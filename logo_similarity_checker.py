import cv2
import json
import requests
import numpy as np
import pandas as pd
import time
import urllib3
import io
import imagehash
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from PIL import Image

# deactivate ssl warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# global paths
OUTPUT_DIR = Path('output_extraction')
LOGOS_DIR = OUTPUT_DIR / 'extracted_logos'
BLACKLIST_DIR = Path('blacklist_references')  # images that will be ignored
LOGOS_DIR.mkdir(parents=True, exist_ok=True)

# thresholds for similarity checks
THRESHOLD_COLOR = 0.98  # 98% color histogram correlation
THRESHOLD_SHAPE = 0.35  # fourier descriptor distance
THRESHOLD_ASPECT = 0.1  # 10% maximum aspect ratio difference
THRESHOLD_PHASH = 5     # maximum hamming distance for pHash

""" Phase 1: logo extraction from the input domains """
class LogoExtractor:
    def __init__(self, output_dir: str = LOGOS_DIR, timeout: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })

    """ fetches HTML content of a URL """
    def fetch_html(self, url):
        try:
            response = self.session.get(url, timeout=self.timeout, verify=False)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser'), response.url
        except:
            return None, None

    """ filtering logic for removing complex photos and banners """
    def is_junk_image(self, img_pil, url):
        try:
            # keep the files that contain logo, brand, icon keywords
            url_lower = url.lower()
            if 'logo' in url_lower or 'brand' in url_lower or 'icon' in url_lower:
                return False

            # remove large photos: eg banners
            w, h = img_pil.size
            ratio = w / h
            if ratio > 5 or ratio < 0.2:
                return True

            # remove photos with lots of colors
            thumb = img_pil.resize((64, 64), Image.Resampling.NEAREST)
            colors = thumb.getcolors(maxcolors=4096)
            if colors is None or len(colors) > 1024:
                return True

            return False
        except Exception:
            return False

    def validate_and_save(self, url, domain, strategy_name):
        try:
            resp = self.session.get(url, timeout=8, verify=False)
            img = Image.open(io.BytesIO(resp.content))
            img = img.convert("RGBA")

            # don't filter google
            if (strategy_name != 'google_fallback'):
                if self.is_junk_image(img, url):
                    return None, "FILTERED"

            # add padding so that the resulted image is squared for fourier analysis
            desired_size = max(img.size)
            new_img = Image.new("RGBA", (desired_size, desired_size), (255, 255, 255, 0))
            new_img.paste(img, ((desired_size - img.size[0]) // 2,
                                (desired_size - img.size[1]) // 2))

            new_img = new_img.resize((256, 256), Image.Resampling.LANCZOS)

            filename = f"{domain.replace('.', '_')}_{strategy_name}.png"
            filepath = self.output_dir / filename
            new_img.save(filepath, "PNG")

            return str(filepath), "OK"
        except Exception:
            return None, "ERROR"

    """ extraction pipeline: meta tags, img tags, google favicon """
    def extract(self, domain):

        candidates = []

        base_url = f"https://{domain}"
        soup, final_url = self.fetch_html(base_url)

        if not soup:
            soup, final_url = self.fetch_html(f"http://{domain}")

        if soup:
            base = final_url or base_url

            # meta tags (OpenGraph)
            for prop in ['og:image', 'og:image:url', 'twitter:image']:
                tag = soup.find('meta', property=prop) or soup.find('meta', attrs={'name': prop})
                if tag and tag.get('content'):
                    candidates.append((urljoin(base, tag['content']), 'meta_tag'))

            # img tags with keyword "logo"
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if not src: continue
                alt = str(img.get('alt', '')).lower()
                cls = str(img.get('class', '')) + str(img.get('id', ''))
                if 'logo' in alt or 'brand' in alt or 'logo' in cls.lower() or 'logo' in src.lower():
                    candidates.append((urljoin(base, src), 'img_tag'))

        # fallback: get google favicon
        candidates.append((
            f"https://www.google.com/s2/favicons?domain={domain}&sz=128",
            "google_fallback"
        ))

        seen_urls = set()
        filtered_count = 0

        for url, strategy in candidates:
            if url in seen_urls or not url.startswith('http'): continue
            seen_urls.add(url)

            path, status = self.validate_and_save(url, domain, strategy)

            if status == "FILTERED": filtered_count += 1
            if status == "OK":
                return {
                    "domain": domain,
                    "success": True,
                    "strategy": strategy,
                    "path": path,
                    "filtered_candidates": filtered_count
                }

        return {
            "domain": domain,
            "success": False,
            "filtered_candidates": filtered_count
        }

""" Phase 2: logo analysis and feature extraction """
class LogoAnalyzer:
    def __init__(self):
        # set 1000 features for high precision
        self.orb = cv2.ORB_create(nfeatures=1000)

        # load bad image hashes
        self.bad_hashes = []
        if BLACKLIST_DIR.exists():
            for file in BLACKLIST_DIR.iterdir():
                if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    img = Image.open(file).convert("RGBA")
                    self.bad_hashes.append(imagehash.phash(img))

    """ extract features from an image path """
    def get_features(self, path):
        try:
            img_cv = cv2.imread(str(path))
            if img_cv is None: return None

            # check for similar bad images using perceptual hashing
            pil_img = Image.open(path)
            curr_phash = imagehash.phash(pil_img)
            for bad_h in self.bad_hashes:
                if curr_phash - bad_h <= 6: return "JUNK"  # similar to a bad image

            # get image dimensions and aspect ratio
            h, w = img_cv.shape[:2]
            aspect = w / h

            # compute fourier descriptors for the contour of the image
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            fourier = np.zeros(32)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 50:
                    cnt_complex = np.array([pt[0][0] + 1j * pt[0][1] for pt in cnt])
                    fft = np.fft.fft(cnt_complex)
                    if len(fft) > 33:
                        fourier = np.abs(fft[1:33])
                        fourier = fourier / (fourier[0] + 1e-10)

            # compute color histogram in hue-saturation-value space
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            # extract keypoints and orb descriptors for geometric text matching
            kp, des = self.orb.detectAndCompute(gray, None)

            return {
                "aspect": aspect,
                "color": hist.flatten(),
                "fourier": fourier,
                "phash": curr_phash,
                "orb_desc": des,
                "orb_kp": kp,
                "path": str(path)
            }
        except:
            return None

    """ checks geometric alignment using RANSAC """
    def check_geometric_match(self, des1, des2, kp1, kp2):

        if des1 is None or des2 is None: return False
        if len(des1) < 2 or len(des2) < 2: return False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except:
            return False

        # test ratio
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 10: return False

        # geometric verification using RANSAC
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is None: return False

            return np.sum(mask) >= 12  # at least 12 points perfectly aligned
        except:
            return False

    """ main similarity check between two feature sets """
    def are_similar(self, f1, f2):

        # filter by perceptual hash
        if f1["phash"] - f2["phash"] > THRESHOLD_PHASH: return False

        # filter by aspect ratio
        if abs(f1["aspect"] - f2["aspect"]) > THRESHOLD_ASPECT: return False

        # filter by color histogram
        color_sim = cv2.compareHist(f1["color"], f2["color"], cv2.HISTCMP_CORREL)
        if color_sim < THRESHOLD_COLOR: return False

        # filter by fourier descriptors
        shape_dist = np.linalg.norm(f1["fourier"] - f2["fourier"])
        if shape_dist > THRESHOLD_SHAPE: return False

        # filter by geometric matching
        if not self.check_geometric_match(f1["orb_desc"], f2["orb_desc"], f1["orb_kp"], f2["orb_kp"]):
            return False

        return True


""" Phase 3: complete pipeline execution """
def run_pipeline():
    start_time = time.time()

    df = pd.read_parquet('logos.snappy.parquet')

    domains = df['domain'].tolist()
    total = len(domains)

    print(f"\n[START] processing {total} domains")

    stats = {
        "success": 0,
        "failed": 0,
        "filtered_junk": 0,
        "strategies": defaultdict(int)
    }

    # parallel logo extraction
    extractor = LogoExtractor()
    extracted = []

    print("\nPHASE 1: LOGO EXTRACTION")

    with ThreadPoolExecutor(max_workers=50) as exec:
        futures = {exec.submit(extractor.extract, d): d for d in domains}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            stats["filtered_junk"] += res.get("filtered_candidates", 0)

            if res["success"]:
                extracted.append(res)
                stats["success"] += 1
                stats["strategies"][res["strategy"]] += 1
            else:
                 stats["failed"] += 1

            if i % 500 == 0:
                print(f" Progress: {i}/{total} | Success: {stats['success']}")


    print(f"\nPHASE 2: GEOMETRIC ANALYSIS for {len(extracted)} images\n")

    analyzer = LogoAnalyzer()
    features = {}
    junk_detected = 0

    for i, item in enumerate(extracted, 1):
        feat = analyzer.get_features(item["path"])
        if feat == "JUNK":
            junk_detected += 1
        elif feat:
            features[item["domain"]] = feat  # use the domain as key

        if i % 500 == 0:
            print(f" Extract features: {i}/{len(extracted)}")

    # start grouping based on similarity
    keys = list(features.keys())
    parents = {k: k for k in keys}

    def find(i):
        if parents[i] == i: return i
        parents[i] = find(parents[i]);
        return parents[i]

    print(f"\nPHASE 3: GROUPING {len(keys)} elements\n")

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            if analyzer.are_similar(features[k1], features[k2]):
                r1, r2 = find(k1), find(k2)
                if r1 != r2: parents[r1] = r2

        if i % 500 == 0:
            print(f"Grouping: {i}/{len(keys)}")

    groups = defaultdict(list)
    for k in keys: groups[find(k)].append(k)

    output = [{"group_id": i, "size": len(v), "domains": v} for i, v in enumerate(groups.values())]
    output = sorted(output, key=lambda x: x['size'], reverse=True)

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=4)

    print("\n")
    print(f"Time: {time.time() - start_time:.1f}s")
    print(f"Processed images: {len(extracted)}")
    print(f"Removed images (blacklisted): {junk_detected}")
    print(f"Total groups: {len(groups)}")

if __name__ == "__main__":
    run_pipeline()