import concurrent.futures
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
import pandas as pd

class LogoExtractor:
    def __init__(self, output_dir: str = 'logos', timeout: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def _fetch_html(self, url):
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, 'html.parser'), resp.url
        except:
            return None, None

    """ Downloads, checks if the image is valid and saves it """
    def _validate_and_save(self, url, domain, strategy_name):
        try:
            # google favicon api handling distinct
            if strategy_name == 'google_favicon':
                resp = self.session.get(url, timeout=5)
            else:
                resp = self.session.get(url, stream=True, timeout=5)

            resp.raise_for_status()
            content = resp.content

            # checks if the content length is less than 100 bytes
            # so that it is not an empty pixel
            if len(content) < 100: return None

            # checks the extension of the image
            ext = '.png'  # default
            if 'svg' in url.lower() or b'<svg' in content[:100]:
                ext = '.svg'
            elif 'jpg' in url.lower() or 'jpeg' in url.lower():
                ext = '.jpg'
            elif 'ico' in url.lower():
                ext = '.ico'

            # save
            filename = f"{domain.replace('.', '_')}_{strategy_name}{ext}"
            filepath = self.output_dir / filename

            with open(filepath, 'wb') as f:
                f.write(content)

            return str(filepath)

        except Exception:
            return None

    def extract(self, domain):
        base_url = f"https://{domain}"
        soup, final_url = self._fetch_html(base_url)

        # if the site does not work with https, try http
        if not soup:
            base_url = f"http://{domain}"
            soup, final_url = self._fetch_html(base_url)

        candidates = []

        if soup:
            base = final_url or base_url

            # 1. open graph image meta tag
            og_img = soup.find('meta', property='og:image')
            if og_img and og_img.get('content'):
                candidates.append((urljoin(base, og_img['content']), 'meta_og'))

            # 2. apple touch icon
            apple_link = soup.find('link', rel=lambda x: x and 'apple-touch-icon' in x.lower())
            if apple_link and apple_link.get('href'):
                candidates.append((urljoin(base, apple_link['href']), 'apple_touch'))

            # 3. explicit images (img_logo_keyword + svg_img combined)
            # search for images that have "logo" in src, class or id
            for img in soup.find_all('img'):
                src = img.get('src')
                if not src: continue

                raw_attrs = str(img.attrs).lower()
                # proitize SVGs or files with "logo" in name
                if 'logo' in raw_attrs or '.svg' in src.lower():
                    candidates.append((urljoin(base, src), 'img_in_page'))

        # 4. google favicon fallback - highest success rate
        # but the resolution is sometimes low
        candidates.append((
            f"https://www.google.com/s2/favicons?domain={domain}&sz=128",
            'google_fallback'
        ))

        # limit to first 5 candidates from page + google fallback
        seen_urls = set()
        for url, strategy in candidates:
            if url in seen_urls: continue
            seen_urls.add(url)

            saved_path = self._validate_and_save(url, domain, strategy)
            if saved_path:
                return {
                    "domain": domain,
                    "success": True,
                    "strategy": strategy,
                    "path": saved_path
                }

        return {"domain": domain, "success": False, "error": "No logo found"}

def process_batch(domains, max_workers=10):
    extractor = LogoExtractor()
    results = []
    total = len(domains)

    print(f"Starting extraction for {total} domains with {max_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_domain = {executor.submit(extractor.extract, d): d for d in domains}

        count = 0
        for future in concurrent.futures.as_completed(future_to_domain):
            count += 1
            res = future.result()
            results.append(res)

            # status
            status = "OK" if res['success'] else "FAIL"
            strat = res.get('strategy') or res.get('error')
            print(f"[{count}/{total}] {res['domain']}: {status} ({strat})")

    return results


if __name__ == "__main__":
    try:
        df = pd.read_parquet('logos.snappy.parquet')
        test_domains = df['domain'].tolist()
    except Exception as e:
        print(f"Failed to read parquet file: {e}")
        test_domains = []

    if test_domains:
        results = process_batch(test_domains, max_workers=10)

        success = sum(1 for r in results if r['success'])
        print(f"\n{'=' * 60}")
        print(f"Succes rate: {success}/{len(results)} ({success / len(results) * 100:.1f}%)")