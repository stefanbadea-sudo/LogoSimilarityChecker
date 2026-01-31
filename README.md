# Logo Similarity Checker
> A geometric approach to logo clustering without machine learning algorithms

---
## Overview

This project tackles the problem of **logo similarity clustering** across a dataset 
consisting of 4,384 company websites without using traditional ML clustering algorithms 
like DBSCAN or k-means. Instead, it employs a **cascade filtering pipeline** inspired by 
mathematical signal processing and computer vision techniques.

**Key Achievement**: > 99.0% extraction success rate with 1,639 distinct groups identified.

**Key Insights:**
- Logos are **perceptually simple** but mathematically complex
- Humans recognize logos through multiple channels: shape, color, spatial layout
- The problem is essentially a **multi-dimensional similarity search**
- Traditional ML clustering treats features as black boxes; we need interpretable filters

### Design
> "Rather than asking 'are these logos the same?', ask 'in how many ways are they different?'"

This led to a **cascade filtering approach** where each filter eliminates false 
matches progressively, inspired by:

- **Fourier Analysis**: Decomposing shapes into frequency components
- **Perceptual Hashing**: How humans perceive image similarity
- **RANSAC**: Geometric consistency verification from computer vision

--- 
## Architecture

> The solution is structured into three main phases:
---
### Phase 1: Multi-Strategy Logo Extraction

**Strategy Cascade** (executed in order until success):

1. **Meta Tags** (OpenGraph/Twitter)
   - `og:image`, `og:image:url`, `twitter:image`
   - Social media platforms enforce logo quality

2. **IMG Tags with Semantic Filtering**
   - Keywords: "logo", "brand", "icon" in alt/class/id/src
   - Developers often use semantic naming

3. **Google Favicon Fallback**
   - `https://www.google.com/s2/favicons?domain={domain}&sz=128`
   - Google's CDN provides coverage as ultimate fallback

**Heuristic Junk Filtering** (applied during extraction):
- **Aspect Ratio Check**: Reject if `ratio > 5` or `ratio < 0.2` (banners/slim badges)
- **Color Complexity**: Reject if >1024 unique colors in 64×64 thumbnail (photos)
- **URL Whitelist**: Always keep if "logo", "brand", "icon" in URL

**Why This Works:**
- Diversified sourcing ensures high coverage
- Early filtering reduces downstream processing
- Google fallback guarantees we never fail completely

---
### Phase 2: Geometric Feature Extraction

 **6-dimensional feature space**

#### 1. **Perceptual Hash (pHash)**
```python
pHash = imagehash.phash(image)  # 64-bit hash
distance = hamming_distance(hash1, hash2)
```
- Robust to minor variations (resizing, compression)
- **Threshold**: less than 5 bits difference


#### 2. **Aspect Ratio**
```python
aspect_ratio = width / height
```

- Logos maintain proportions across usage
- **Threshold**: ±10% difference allowed
- **Catches**: Different logos with similar colors


#### 3. **Color Histogram (HSV)**
```python
hist = cv2.calcHist([hsv], [0,1], None, [32,32], [0,180,0,256])
similarity = cv2.compareHist(hist1, hist2, HISTCMP_CORREL)
```

- Brand colors are sacred; exact matches likely same logo
- **Threshold**: 98% correlation
- **Inspired by**: Content-based image retrieval systems

#### 4. **Fourier Shape Descriptors**
```python
fft = np.fft.fft(contour_complex)
descriptors = abs(fft[1:33]) / abs(fft[0])
```

- Describes shape independent of position/rotation/scale
- **Mathematical Foundation**: Frequency domain representation
- **Threshold**: Euclidean distance < 0.35
- **Why 32 components**: Balance between detail and noise

**Mathematical Intuition:**
Logos have characteristic frequencies:
- **Low frequencies**: Overall shape (circle, rectangle, text width)
- **Mid frequencies**: Details (curves, angles, letter shapes)
- **High frequencies**: Noise (filtered out)

#### 5. **ORB Keypoint Descriptors**
```python
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(gray, None)

```
- Detects distinctive corners/edges (especially text)
- **Use Case**: Text-heavy logos (company names)
- **1000 features**: High precision over speed

#### 6. **RANSAC Geometric Verification**
```python
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
inliers = np.sum(mask) >= 12
```

- Final check that keypoints align geometrically
- **Prevents**: False matches from scattered similar features
- **Threshold**: at least 12 geometrically consistent points

### Phase 3: Union-Find Clustering

I used union find (disjoint set) data structure to efficiently 
cluster logos based on similarity relationships established by the cascade filters.

```python
def find(i):
    if parents[i] == i: return i
    parents[i] = find(parents[i])  # path compression
    return parents[i]

# Merge similar logos
if are_similar(logo1, logo2):
    root1, root2 = find(logo1), find(logo2)
    if root1 != root2: 
        parents[root1] = root2  # union
```

---

## Results

### Extraction Performance
```
Total Domains:              4,384
Successfully Extracted:     4,383  (99.98%)
Failed Extractions:         1      (0.02%)
```

### Filtering Pipeline
```
Input Images:               4,383
Junk Filtered (Heuristic):  267    (6.1%)
Valid Features Extracted:   3,214  (73.3%)
Blacklisted:                267
```

### Clustering Results
```
Total Groups Formed:        1,639
Unique Logos:               1,339  (81.7%)
Multi-Member Groups:        300    (18.3%)
Largest Cluster:            222 domains (AAMCO group)
```

### Top 10 Largest Groups
1. **AAMCO** 222 domains
2. **Mazda** 102 domains
3. **Culligan** 73 domains
4. **Kia** 49 domains
5. **B. Braun** 35 domains
6. **Nestlé** 32 domains
7. **MEDEF** 31 domains
8. **Spitex** 29 domains
9. **Airbnb** 29 domains
10. **Veolia** 28 domains


### Performance Metrics
```
Total Execution Time:       394.7 seconds (~6.6 minutes)
Average Throughput:         11.1 domains/second
Extraction Phase:           180s (45.6%)
Feature Analysis:           120s (30.4%)
Clustering:                 95s  (24.0%)
```

---

### Computational Complexity Comparison

| Dataset Size    | Brute Force | Union-Find | Speedup      |
|-----------------|-------------|------------|--------------|
| 4,384 (current) | 6.5 min     | 6.6 min    | 1x (current) |
| 10,000          | 1.5 h       | 15 min     | 6x           |
| 100,000         | 173 h       | 2.5 h      | 69x          |
| 1,000,000       | 19,841 h    | 25 h       | 793x         |

At current scale, extraction dominates. At larger scales, Union-Find becomes critical.

---

### Output Structure
```
output_extraction/
├── extracted_logos/          # Downloaded logo images
│   ├── example.png
│   ├── another_example.png
│   └── ...
└── results.json              # Clustering results
```

---

## Edge Cases Handled

1. **SSL Certificate Errors**: Disabled verification with warnings
2. **Timeout Protection**: 5-second timeout per request
3. **Missing Images**: Fallback to next strategy
4. **Duplicate URLs**: De-duplication in extraction phase
5. **Empty Contours**: Zero-padding for Fourier transform
7. **Insufficient Keypoints**: Return false for RANSAC if <10 matches

---

## Author
**Stefan Badea**  
Computer Science Student @ University Politehnica of Bucharest