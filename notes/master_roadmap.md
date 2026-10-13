# CSE340 — Digital Image Processing (IIIT-Delhi)
## Final Master Learning Roadmap — 2026 · v3 Quiz-Aligned Course Corpus

> **Purpose:** This is the source-of-truth study plan for learning CSE340 from foundations to course completion.
>
> **v3 update (23 Aug 2026):** The roadmap is now aligned to the *actual uploaded 2026 course corpus*: Lecture 0 plus the professor's Lectures 1–3, and the uploaded Gonzalez & Woods *Digital Image Processing*, 4th Edition Global Edition, Chapters 1–4. The professor's slides are the highest-priority quiz source; the textbook is used selectively for deeper theory, worked examples, and problem practice rather than read linearly.
>
> **v2 foundation retained:** full Camera & Imaging Systems track: camera optics/sensors/ISP, X-ray, CT, MRI, ultrasound, PET/SPECT, microscopy, thermal/IR, multispectral/hyperspectral, stereo/ToF/structured light, LiDAR, OCT, and SAR, with explicit revisit points throughout the DIP syllabus.
>
> **Assignment research lane added:** Bilateral Filtering → Non-Local Means → future improved-NLM/comparative denoising research is parked beside Lecture 2 and activated only after the core quiz material is mastered.

---


# 0A. 2026 Actual-Course Overlay — READ THIS FIRST

This section overrides the generic semester ordering whenever it conflicts with the **professor's actual lecture sequence**.

## 0A.1 Current source hierarchy

Use sources in this order:

```text
1. Professor lecture slides
      ↓
2. Exact Gonzalez & Woods sections aligned to those slides
      ↓
3. Selected Gonzalez & Woods problems/examples
      ↓
4. Our deep conceptual derivations and hand practice
      ↓
5. NumPy / SciPy / OpenCV validation
      ↓
6. External tutorials and research papers
```

The purpose is **not** to consume a 300-page book linearly.

The purpose is:

\[
\boxed{
\text{Lecture requirement}
\rightarrow
\text{book depth}
\rightarrow
\text{worked reasoning}
\rightarrow
\text{quiz recall}
\rightarrow
\text{assignment/research extension}
}
\]

### Corpus currently indexed

| Source | Role |
|---|---|
| Lecture 0 — Signals, Linear Systems, Convolution, Fourier Tools | Mathematical foundation |
| Lecture 1 — Image Acquisition & Intensity Transformations | Quiz/core lecture |
| Lecture 2 — Smoothing: Low-Pass Spatial Filtering | Quiz/core lecture + Assignment-1 anchor |
| Lecture 3 — Sharpening & Edge Detection | Quiz/core lecture |
| Gonzalez & Woods 4E, Chapters 1–4 | Selective theory, examples, problem bank |
| MIT/CSAIL Bilateral Filtering course material supplied by student | Future Assignment-1 depth source |
| Nir Sharon / improved NLM material supplied by student | Future Assignment-1 research lane |

---

## 0A.2 Important source labels

Every topic in our study should be mentally tagged as one of:

- **[SLIDE CORE]** — explicitly part of the professor's lecture; highest quiz priority.
- **[BOOK CORE]** — directly supports a slide topic and should be read selectively.
- **[SLIDE-ONLY CORE]** — professor teaches it even though it is not prominent in uploaded Chapters 1–4; still quiz-important.
- **[BEYOND-TEXTBOOK]** — explicitly marked as such by the professor; learn after core unless emphasized in class.
- **[ASSIGNMENT PARK]** — research/comparison idea saved for later; do not let it derail quiz preparation.

This distinction is crucial. **A topic is not low-priority merely because Gonzalez does not cover it in the uploaded chapters. If it is in the professor's core slide sequence, it is quiz material.**

---

# 0B. Current Progress Ledger

## Lecture 0 — Mathematical Bootstrap

| Module | Status | What it means |
|---|---|---|
| L0.1 Signals | ✅ Taught | continuous/discrete, transforms, impulses |
| L0.2 Systems | ✅ Taught | linearity + added shift-invariance bridge |
| L0.3 Why convolution exists | ✅ Taught | impulse decomposition → system response |
| L0.4 Discrete convolution deep | ✅ Taught · mastery drill pending | asymmetric kernels, padding, 2-D bridge |
| L0.5 Continuous convolution deep | ✅ Taught · mastery drill pending | graphical overlap + piecewise integration |
| L0.6 Convolution in 2-D images | ✅ Taught | kernels, PSF, smoothing, edges |
| L0.7 Correlation | ✅ Taught | similarity, template matching, convolution distinction |
| L0.8 Fourier intuition | ✅ Taught | spatial frequency, magnitude, phase |
| L0.9 Fourier Series + FT | ✅ Taught | periodic → harmonics; aperiodic → continuous FT |
| L0.10 Convolution theorem | ✅ Taught | \(x*h \leftrightarrow XH\), frequency response |
| **L0.11 DFT deep dive** | **▶ NEXT** | build a 4-point DFT from scratch |
| L0.12 Circular convolution | ⬜ Pending | why DFT multiplication wraps |
| L0.13 Zero padding | ⬜ Pending | recover linear convolution via DFT |
| L0.14 FFT | ⬜ Pending | DFT vs efficient algorithm |

### Current exact position

\[
\boxed{\textbf{We are immediately before L0.11 — DFT Deep Dive}}
\]

Lectures 1–3 have now been **ingested and aligned**, but they have **not yet received our full deep-teaching pass**. Previewing their ideas during convolution does not count as mastery.

---

# 0C. Professor-Actual Lecture Map

## Lecture 1 — Image Acquisition & Intensity Transformations

### Part I — Image Acquisition [SLIDE CORE]

#### Digital camera
Deep-study chain:

```text
Scene
→ reflected light
→ lens
→ sensor / photosite
→ electrical charge
→ ADC
→ sampling + quantization
→ digital image
```

Then:

```text
photosite alone → brightness only
→ Bayer CFA (RGGB)
→ missing two colors per photosite
→ demosaicing
→ possible zippering / color-fringing artifacts
```

**Professor-specific details to know**
- Bayer RGGB: 25% R, 50% G, 25% B.
- More green is linked in the slide to human sensitivity near 555 nm.
- Demosaicing is interpolation and can fail at sharp/high-frequency color boundaries.

**Book bridge**
- Ch. 2.3 Image Sensing and Acquisition.
- Ch. 2.4 Image Sampling and Quantization.
- Do **not** read all surrounding material before the slide concepts are clear.

#### MRI [SLIDE CORE]

Quiz-level conceptual chain:

```text
spin signal acquisition
→ samples populate k-space F(u,v)
→ k-space is spatial-frequency data
→ IFFT
→ reconstructed image f(x,y)
```

Know:
- scanner does not directly "photograph pixels";
- center of k-space → gross shape / contrast;
- outer k-space → edges / fine detail.

Revisit after DFT/2-D Fourier.

#### CT [SLIDE CORE]

```text
rotating X-ray source/detector
→ projection at angle θ
→ line integrals = Radon transform g(r,θ)
→ stack angles = sinogram
→ backprojection
→ filtered backprojection
```

At first pass, understand geometry and information flow. Full reconstruction mathematics is later.

#### Ultrasound [SLIDE CORE]

```text
pulse
→ echo
→ travel time → depth
→ echo amplitude → brightness
→ A-line
→ sweep many A-lines
→ B-mode image
```

Remember the slide's approximate tissue sound speed:

\[
c \approx 1540\ \mathrm{m/s}
\]

and the conceptual depth equation:

\[
d=\frac{ct}{2}
\]

(the factor 2 is transmit + return travel).

---

### Part II — Intensity Transformations [SLIDE CORE + BOOK CORE]

Core model:

\[
\boxed{s=T(r)}
\]

Point processing means the output pixel depends only on the input intensity at the **same location**.

Learn deeply:

1. Negative
   \[
   s=(L-1)-r
   \]

2. Log
   \[
   s=c\log(1+r)
   \]

3. Power law / gamma
   \[
   s=cr^\gamma
   \]

4. Contrast stretching — piecewise linear.

5. Intensity-level slicing — emphasize \([A,B]\).

6. Bit-plane slicing — decompose pixel binary significance.

For an 8-bit image:

\[
f(x,y)=\sum_{k=0}^{7} 2^k b_k(x,y)
\]

High-order planes carry most visible structure; all weighted planes together reproduce the image exactly.

### Selected Gonzalez problem bank for Lecture 1

Do **not** solve every Chapter-2/3 problem.

High-value set:

- Ch3 **3.1** — derive full-range intensity stretching.
- **3.2** — parameterized contrast-stretching function.
- **3.3–3.5** — bit-plane/intensity-slicing reasoning.
- Ch2 **2.5–2.8** — camera/image resolution style calculations if the professor emphasizes acquisition numericals.
- Ch2 **2.12–2.13** — quantization/false-contouring style reasoning after sampling/quantization is formally covered.

Histogram problems 3.6 onward belong to the histogram lecture, not this Lecture-1 pass.

---

## Lecture 2 — Smoothing: Low-Pass Spatial Filtering

This is a **major quiz + assignment junction**.

### Part I — Box / Mean Filtering [SLIDE CORE + BOOK CORE]

Transition:

\[
s(x,y)=T[r(x,y)]
\]

point processing

to

\[
s(x,y)=T[\text{window around }(x,y)]
\]

neighborhood processing.

Know padding:
- zero;
- replicate;
- reflect.

For an \(m\times n\) box:

\[
w(s,t)=\frac1{mn}
\]

and coefficients sum to 1 so a constant patch remains constant.

Deep reasoning:
- larger windows → stronger blur;
- fine structures disappear before large structures;
- box filter is content-blind;
- abrupt window boundary and equal weights are limitations.

### Part II — Gaussian Filtering [SLIDE CORE + BOOK CORE]

\[
w(s,t)=K
\exp\left(
-\frac{s^2+t^2}{2\sigma^2}
\right)
\]

where \(K\) normalizes the discrete kernel to sum to 1.

Understand:
- distance-dependent weighting;
- \(\sigma\) controls spatial spread/blur;
- separability;
- 2-D Gaussian as outer product of 1-D vectors;
- why sequential 1-D passes reduce arithmetic.

Gonzalez-specific useful fact:
- practical Gaussian support is on the order of \(6\sigma\times6\sigma\), rounded to a suitable odd kernel.

### Binomial Approximation [BEYOND-TEXTBOOK IN PROFESSOR SLIDES]

Professor explicitly labels this material as beyond Gonzalez.

Know conceptually:
- normalized Pascal-triangle rows approximate a Gaussian;
- outer product gives 2-D separable binomial kernel;
- integer/power-of-two arithmetic can be computationally cheap.

Do not spend more time here than on Gaussian unless class emphasis suggests otherwise.

### Part III — Why Averaging Is Low-Pass [SLIDE CORE]

Must explain in words:

\[
\boxed{
\text{slow variation survives;\ fast variation cancels}
}
\]

Do not rely only on Fourier terminology. The lecture intentionally explains low-pass behavior directly in the spatial domain first.

### Part IV — Bilateral Filtering [SLIDE CORE + ASSIGNMENT ANCHOR]

This is **not optional just because it is beyond the traditional Chapter-3 box/Gaussian sequence**.

Core weight:

\[
w(p,q)
=
\exp\left(
-\frac{\|p-q\|^2}{2\sigma_s^2}
\right)
\exp\left(
-\frac{(I(p)-I(q))^2}{2\sigma_r^2}
\right)
\]

Use notation \(\sigma_s\) for the spatial parameter in our notes to avoid ambiguity.

Interpret each factor:

\[
\text{spatial term}
=
\text{is }q\text{ close to }p?
\]

\[
\text{range term}
=
\text{does }q\text{ have similar intensity?}
\]

Then:

\[
\boxed{\text{bilateral weight = spatial closeness × intensity similarity}}
\]

Parameter reasoning:
- larger \(\sigma_s\) → wider spatial neighborhood;
- smaller \(\sigma_r\) → stronger refusal to cross intensity edges;
- very large \(\sigma_r\) → range term becomes less discriminating → behavior approaches Gaussian smoothing.

Key exam explanation:
> Gaussian knows **where** a neighbor is. Bilateral knows both **where it is** and **what intensity it has**.

Applications in slides:
- edge-preserving denoising;
- base/detail decomposition;
- tone/detail manipulation.

### Part V — Non-Local Means [SLIDE CORE + ASSIGNMENT ANCHOR]

Central observation:

\[
\boxed{\text{natural images contain repeated patches; noise does not repeat coherently}}
\]

Move from bilateral:

```text
compare center-pixel intensities
```

to NLM:

```text
compare entire patches
```

Professor-slide weight:

\[
w(p,q)
=
\exp\left(
-\frac{\|P(p)-P(q)\|^2}{h^2}
\right)
\]

where \(P(p)\) is a patch centered at \(p\).

Understand:
- **non-local** = candidate patches may be far away;
- **patch-based** = similarity uses neighborhood structure rather than one pixel;
- **robustness** = patch comparison averages information;
- **cost** = every candidate requires a patch comparison.

### The filter progression that must be reconstructible from memory

```text
BOX
weights know nothing about image
        ↓
GAUSSIAN
weights know spatial distance
        ↓
BILATERAL
weights know spatial distance + intensity difference
        ↓
NON-LOCAL MEANS
weights know patch similarity + can search far away
```

This progression is likely more important than memorizing isolated definitions.

### Selected Gonzalez problem bank for Lecture 2

High-value:
- **3.20** — separable convolution mechanics.
- **3.23–3.25** — Gaussian separability/properties.
- **3.26** — why spatial arrangement matters even with identical histograms.
- **3.27–3.28** — repeated/composite Gaussian filtering.
- **3.29** — limiting behavior of repeated low-pass filtering.
- **3.31** — sum-1 kernel and intensity preservation.
- **3.33** — point source / defocus / Gaussian PSF reasoning.
- **3.34–3.35** — kernel size versus structures removed.

Bilateral and NLM practice must come primarily from professor slides + our original questions because they are not a central part of the uploaded Gonzalez Chapters 1–4.

---

## Lecture 3 — Sharpening & Edge Detection

### Part I — Derivatives [SLIDE CORE + BOOK CORE]

Discrete first difference:

\[
\frac{\partial f}{\partial x}
\approx
f(x+1)-f(x)
\]

Discrete second difference:

\[
\frac{\partial^2 f}{\partial x^2}
\approx
f(x+1)+f(x-1)-2f(x)
\]

Behavior table to know conceptually:

| Structure | First derivative | Second derivative |
|---|---|---|
| flat | zero | zero |
| ramp | nonzero along ramp | mainly ramp ends |
| step | single/thicker response | positive/negative pair with zero crossing |
| fine line | responds at sides | stronger/finer response |
| noise sensitivity | lower | higher |

### Part II — Laplacian [SLIDE CORE + BOOK CORE]

\[
\nabla^2 f
=
\frac{\partial^2f}{\partial x^2}
+
\frac{\partial^2f}{\partial y^2}
\]

4-neighbor discrete form:

\[
\nabla^2f
=
f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
\]

Know all four sign-convention masks:

```text
 0  1  0      1  1  1
 1 -4  1      1 -8  1
 0  1  0      1  1  1
```

and their negatives.

Sharpening:

\[
g=f+c\nabla^2f
\]

The sign \(c\) depends on which Laplacian sign convention is used.

Composite 4-neighbor sharpening mask:

```text
 0 -1  0
-1  5 -1
 0 -1  0
```

8-neighbor counterpart has center 9 and \(-1\) elsewhere.

### Part III — Unsharp Masking & Highboost [SLIDE CORE + BOOK CORE]

\[
f_{\text{blur}}=\text{LPF}(f)
\]

\[
g_{\text{mask}}=f-f_{\text{blur}}
\]

\[
g=f+k\,g_{\text{mask}}
\]

- \(k=1\): unsharp masking.
- \(k>1\): highboost.
- too-large \(k\): halos and amplified noise.

Deep meaning:

\[
\boxed{
\text{blur removes detail}
\Rightarrow
f-\text{blur}
=
\text{detail estimate}
}
\]

### Part IV — Gradient Operators [SLIDE CORE + BOOK CORE]

\[
\nabla f=[G_x,G_y]
\]

\[
|\nabla f|
=
\sqrt{G_x^2+G_y^2}
\approx |G_x|+|G_y|
\]

Know:
- gradient direction = direction of greatest intensity increase;
- gradient is perpendicular/across an edge;
- horizontal derivative mask responds strongly to vertical edges and vice versa.

Must recognize Roberts, Prewitt and Sobel masks.

Important comparison:
- Roberts: 2×2, thin/noisy, half-pixel localization issue.
- Prewitt: 3×3, centered, equal orthogonal smoothing weights.
- Sobel: 3×3, center line weighted 2, more built-in smoothing.

### Part V — Edge Detection [SLIDE CORE]

The professor's core sequence continues beyond a plain gradient.

Know why:

\[
\boxed{\text{derivatives amplify noise}}
\]

so serious detectors:

```text
smooth
→ differentiate
→ decide
```

#### LoG / Marr–Hildreth

Gaussian smoothing + Laplacian.

Edge is identified through **zero crossings** of the second-derivative response.

#### Canny — five stages

1. Gaussian smoothing.
2. Gradient magnitude + direction.
3. Non-maximum suppression.
4. Double threshold.
5. Hysteresis tracking.

Parameter intuition:
- small \(\sigma\): keeps detail but admits noise;
- large \(\sigma\): suppresses noise but loses/rounds fine structure;
- higher thresholds: fewer/weaker contours survive.

### Part VI — Explicit Professor "Beyond the Textbook"

Keep as enrichment unless emphasized in quiz discussion:

- Scharr;
- Kirsch / compass operators;
- SUSAN;
- structure tensor / Harris;
- steerable filters;
- scale space;
- Perona–Malik anisotropic diffusion;
- precision–recall evaluation;
- learned edge detectors.

Do **not** let these displace Laplacian/Sobel/Canny mastery.

### Selected Gonzalez problem bank for Lecture 3

High-value:
- **3.38** — order of smoothing and Laplacian.
- **3.40** — \(-8\) vs \(-4\) Laplacian sharpness.
- **3.41** — derive one-pass unsharp mask.
- **3.42** — relationship between Laplacian subtraction and unsharp mask.
- **3.43** — gradient isotropy.
- **3.44** — separability of high-pass / Roberts / Sobel kernels.

Canny and some LoG details should be practiced from the professor slides and our generated problems rather than relying on Ch1–4 problem numbering.

---

# 0D. Lecture-0 ↔ Gonzalez Chapter-4 Bridge

Our Lecture-0 deep dive is not wasted side study. It directly prepares the professor's next announced block: **frequency domain**.

Selected current-book problems for the foundation we already learned:

- **4.3–4.4** — convolution with impulses / shift property.
- **4.11** — prove continuous convolution theorem.
- **4.15–4.18** — transform-pair, periodicity/shift, and discrete convolution-theorem reasoning.
- **4.21** — reason about stripe width and 2-D spectrum locations.
- Later, after DFT/FFT teaching, select FFT/filtering problems rather than doing the whole chapter.

Do not start the entire Chapter 4 problem set now.

---

# 0E. Provisional Quiz-1 Corpus

Until the professor announces a narrower boundary, use this **working** quiz corpus:

```text
Lecture 0 mathematical foundation
+
Lecture 1
+
Lecture 2
+
Lecture 3
```

This is a **planning assumption**, not a claim about the official quiz boundary.

### Quiz-1 priority tiers

#### Tier A — must derive / calculate
- discrete convolution and correlation;
- full/same/valid and padding;
- point processing \(s=T(r)\);
- negative/log/gamma numerical mappings;
- bit-plane reasoning;
- box/Gaussian weighted average;
- Gaussian normalization/separability;
- Laplacian masks and sharpening signs;
- unsharp/highboost equations;
- \(G_x,G_y\), gradient magnitude;
- small hand calculations with Sobel/Prewitt;
- DFT basics once L0.11 is completed.

#### Tier B — must explain clearly
- camera pipeline;
- Bayer/demosaicing/artifacts;
- MRI k-space center vs outer region;
- CT projection/Radon/sinogram/backprojection;
- ultrasound A-line/B-mode/depth;
- why averaging is low-pass;
- Gaussian vs bilateral;
- bilateral parameter effects;
- NLM patch/self-similarity logic;
- first vs second derivatives;
- why derivatives amplify noise;
- LoG zero crossing;
- Canny five stages.

#### Tier C — recognize / compare unless emphasized
- binomial approximation;
- NLM regression extension;
- Scharr;
- Kirsch;
- SUSAN;
- structure tensor/Harris;
- steerable filters;
- scale space;
- anisotropic diffusion;
- precision–recall edge-detector comparison.

---

# 0F. How We Will Handle the 317-Page Book

We **will not** try to keep 100k–250k+ source tokens verbatim in working memory.

Instead, use an indexed/retrieval workflow:

```text
Professor slide topic
        ↓
retrieve exact book section
        ↓
retrieve nearby examples/figures
        ↓
retrieve only relevant end-of-chapter problems
        ↓
teach deeply
        ↓
record mastery + problem IDs in this roadmap
```

This scales much better than linear reading and prevents irrelevant textbook sections from consuming quiz time.

For each future lecture you upload:

```text
1. Parse slide structure
2. Extract exact formulas / definitions / diagrams
3. Map to Gonzalez sections
4. Flag slide-only or beyond-textbook content
5. Select book examples/problems
6. Build deep lesson sequence
7. Add quiz traps
8. Add assignment connection
9. Update this progress ledger
```

---

# 0G. Assignment-1 Research Parking Lot — Bilateral + NLM

**Status: PARKED. Do not start full research until Lecture-2 core is mastered and the actual assignment brief is available.**

The likely research spine suggested by the current slide material and the external resources supplied by the student is:

```text
Box / Gaussian baseline
        ↓
Bilateral
        ↓
Non-Local Means
        ↓
Improved / robust NLM variants
        ↓
comparative denoising study
```

Possible future baselines:
- box;
- Gaussian;
- median when noise type warrants it;
- bilateral;
- classical NLM;
- an improved NLM variant;
- BM3D only if the assignment scope and implementation budget justify it.

Possible study variables:
- additive Gaussian-noise level;
- \(\sigma_s\);
- \(\sigma_r\);
- NLM patch size;
- NLM search-window size;
- NLM parameter \(h\);
- computational cost.

Possible metrics:
- PSNR;
- SSIM;
- edge preservation;
- texture preservation;
- runtime/memory.

Possible ablations:
- pixel similarity vs patch similarity;
- local search vs wider/non-local search;
- bilateral range term removed;
- NLM patch size;
- low-rank/robust extension versus ordinary NLM.

External sources already supplied for the later research phase:
1. **MIT/CSAIL — A Gentle Introduction to Bilateral Filtering and its Applications**  
   Use later for bilateral intuition, implementation, applications, variants and limitations.
2. **Nir Sharon / collaborators — Improving Non-Local Means operators**  
   Future lane: affinity/operator viewpoint, low-rank improvement, efficient matrix-function approximation, and denoising comparisons.

### Assignment rule

\[
\boxed{
\text{No research rabbit hole before the base method is understood mathematically.}
}
\]

When activated, the assignment track will use:

```text
theory
→ from-scratch implementation
→ controlled synthetic tests
→ natural-image tests
→ metrics
→ ablations
→ research comparison
→ report
```

---

# 0H. Immediate Study Order From Here

Do this next:

```text
L0.11 DFT Deep Dive
→ L0.12 Circular Convolution
→ L0.13 Zero Padding
→ L0.14 FFT
```

Then start the professor's actual content:

```text
Lecture 1 Deep
→ Lecture 1 mastery questions
→ selected Gonzalez problems

Lecture 2 Deep
→ Bilateral/NLM core
→ Lecture 2 mastery questions
→ selected Gonzalez + original problems
→ Assignment-1 research lane may activate here

Lecture 3 Deep
→ derivative/Laplacian/gradient/Canny
→ Lecture 3 mastery questions
→ selected Gonzalez + original problems

Then
→ closed-book Quiz-1 synthesis
→ weak-topic repair
→ ≥85% mock threshold
```

This means we continue building the mathematical foundation **and** remain synchronized with the actual course.

---

# 0. How to Use This Roadmap

Do **not** treat this as a book to read from top to bottom once.

Use it in four parallel ways:

1. **Course Track** — follow the official week-by-week order.
2. **Quiz Track** — maintain closed-book mathematical mastery because quizzes are the highest-risk component.
3. **Build Track** — implement important algorithms first with NumPy, then with OpenCV/scikit-image.
4. **Depth Track** — read selected papers and connect classical DIP to modern computer vision only after the course concept is understood.

The rule for declaring a topic learned is:

\[
\boxed{
\text{Intuition}
+
\text{Mathematics}
+
\text{Hand Calculation}
+
\text{Implementation}
+
\text{Quiz Recall}
}
\]

If one of these is missing, the topic is not yet fully mastered.

---

# 1. Course DNA

| Item | CSE340 plan |
|---|---|
| **Course** | CSE340 / CSE540 / ECE350 — Digital Image Processing |
| **Credits** | 4 |
| **Course type** | Classical, mathematical Digital Image Processing |
| **Not primarily** | Deep learning, CNN training, object detection, foundation models |
| **Mandatory prerequisites** | MTH100 Maths I, MTH201 Probability & Statistics |
| **Desirable prerequisite** | ECE250 Signals & Systems |
| **Primary textbook for the current uploaded corpus** | Rafael C. Gonzalez & Richard E. Woods, *Digital Image Processing*, **4th ed., Global Edition** (uploaded Chapters 1–4). Older roadmap references to 3E are superseded by this uploaded 4E mapping for quiz study. |
| **Main tools** | Python, NumPy, Matplotlib, OpenCV, SciPy, scikit-image, Jupyter |
| **Course endpoint** | Image formation → enhancement → transform analysis → restoration → compression → morphology → segmentation → feature representation |

## 1.1 The entire course in one line

```text
Image
→ Represent
→ Sample
→ Transform
→ Filter
→ Restore
→ Compress
→ Segment
→ Describe
```

## 1.2 Official learning outcomes translated into practical skills

By the end of the course you should be able to:

- reason about images in the **spatial domain**;
- derive and analyze **image transforms**;
- move between **spatial and Fourier domains**;
- understand what information compression removes and preserves;
- understand classical **segmentation and feature representation**;
- implement the main algorithms yourself rather than treating OpenCV as a black box.

---

# 2. Your Actual 2026 Assessment Strategy

Your current semester grading is:

```text
40% — 4 Quizzes
40% — Assignments
10% — Midsem take-home / mini-project style
10% — Endsem take-home / mini-project style
```

The older TechTree assessment table is not used here for strategy.

## 2.1 Strategic consequence

Your semester should be optimized as:

\[
\boxed{
\text{Quiz Mastery}
>
\text{Assignment Execution}
>
\text{Mid/End-sem Integration}
}
\]

Assignments are large in weight, but they are also the component where coding tools, references, debugging, and permitted LLM assistance make recovery easier.

Quizzes are different:

- no debugging loop;
- no external tools;
- small mathematical mistakes are expensive;
- four quizzes together account for 40%.

So this roadmap deliberately uses **closed-book mastery** as the pacing mechanism.

---

# 3. The Learning Method — Use This for Every Topic

For every new topic, complete the following seven layers.

## Layer 1 — Intuition

Answer:

> What problem is this method solving?

Example:

> Gaussian smoothing suppresses rapidly changing local intensity variations.

## Layer 2 — Mathematics

Know the relevant equation and what every term means.

Example:

\[
g(x,y)
=
\sum_s\sum_t h(s,t)f(x-s,y-t)
\]

## Layer 3 — Hand Calculation

Perform the operation on a tiny synthetic example.

Examples:

- 3×3 convolution;
- 4-level histogram equalization;
- a tiny DFT;
- one erosion/dilation step;
- Sobel on a small patch.

## Layer 4 — From-Scratch Implementation

Use NumPy before using a high-level library.

Tag this:

```text
[NUMPY]
```

## Layer 5 — Library Implementation

Verify using OpenCV, SciPy, or scikit-image.

Tag this:

```text
[LIB]
```

## Layer 6 — Visual Experiment

Change parameters and explain why the output changes.

Example:

```text
Gaussian σ = 0.5
Gaussian σ = 1
Gaussian σ = 3
Gaussian σ = 8
```

## Layer 7 — Quiz Recall

Without notes, explain:

- concept;
- equation;
- one example;
- one common trap.

### Mastery tags

Use:

```text
[INTUITION] [MATH] [HAND] [NUMPY] [LIB] [QUIZ]
```

A topic is **quiz-ready** only when at least:

```text
[MATH] ✓
[HAND] ✓
[QUIZ] ✓
```

are green.

---

# 4. Complete Knowledge Tree

```text
MATHEMATICAL FOUNDATION
│
├── Matrices / vectors
├── Functions and coordinates
├── Probability
├── Complex numbers
├── Sinusoids
└── Convolution
        │
        ▼
DIGITAL IMAGE FUNDAMENTALS
│
├── Image formation
├── Pixels
├── Resolution
├── Sampling
├── Quantization
├── Interpolation
└── Geometric transformations
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
CAMERA & IMAGING SYSTEMS TRACK                  CORE DIP CONTINUES
│
├── Pinhole / lens camera model
├── Focal length, FOV, aperture, exposure, depth of field
├── CCD / CMOS sensors, Bayer CFA, demosaicing
├── X-ray radiography
├── CT
├── MRI
├── Ultrasound
├── PET / SPECT
├── Microscopy / confocal / electron microscopy
├── Thermal / infrared
├── Multispectral / hyperspectral
├── Stereo / ToF / structured-light depth
├── LiDAR
├── OCT
└── SAR / radar imaging
        │
        ▼
SPATIAL IMAGE PROCESSING
│
├── Intensity transformations
├── Histograms
├── Histogram equalization
├── Convolution vs correlation
├── Smoothing
├── Sharpening
└── Image registration
        │
        ▼
FOURIER / FREQUENCY DOMAIN
│
├── Sinusoids
├── Fourier transform
├── DFT
├── 2-D DFT
├── Magnitude / phase
├── DFT properties
├── Sampling theorem
├── Aliasing
└── Frequency-domain filtering
        │
        ├──────────────────┐
        ▼                  ▼
IMAGE RESTORATION       COMPRESSION
│                       │
├── Degradation model   ├── Redundancy
├── Noise models        ├── DCT
├── Blur / PSF          ├── Quantization
├── Inverse filtering   ├── Zig-zag / coding concepts
├── Wiener filtering    └── JPEG
└── CLS / regularization
        │
        ▼
COLOR PROCESSING
│
├── RGB
├── CMY / CMYK
├── HSV / HSI
└── YCbCr
        │
        ▼
MATHEMATICAL MORPHOLOGY
│
├── Structuring element
├── Dilation
├── Erosion
├── Opening
└── Closing
        │
        ▼
SEGMENTATION
│
├── Point / line detection
├── Gradient
├── Sobel
├── Laplacian
├── Canny
├── Thresholding
├── Otsu
└── Region processing
        │
        ▼
FEATURE REPRESENTATION — DEPTH EXTENSION
│
├── Connected components
├── Contours
├── Moments / shape
├── Harris corners
├── Descriptors
└── SIFT intuition
```

---

# 5. Phase 0 — Mathematical Bootcamp

Run this **in parallel with Week 1**. Do not delay the official course waiting for perfect mathematics.

Target: approximately 7–10 focused sessions.

---

## 5.1 Matrices and vectors

Know:

\[
A+B,\quad AB,\quad A^T
\]

and understand:

- shape/dimensions;
- matrix multiplication;
- elementwise multiplication;
- vectors;
- norms;
- inverse;
- coordinates;
- homogeneous coordinates.

### Minimum hand practice

- multiply two 2×2 matrices;
- transform a 2-D coordinate;
- apply a 3×3 homogeneous transform;
- distinguish matrix multiplication from element-wise multiplication.

---

## 5.2 Functions and image coordinates

Move mentally from:

\[
f(x)
\]

to:

\[
f(x,y)
\]

Then:

\[
f(x,y)\rightarrow f[m,n]
\]

Continuous image:

\[
f(x,y)
\]

Digital sampled image:

\[
f[m,n]
\]

The pixel is not merely a colored square — it is a sampled value of an underlying spatial signal.

---

## 5.3 Probability essentials

Know:

\[
E[X],\qquad Var(X),\qquad \sigma^2
\]

Distributions relevant to this course:

- Gaussian;
- uniform;
- impulse / salt-and-pepper;
- Poisson.

You do not need a complete probability course. You need enough to reason about image noise and Wiener restoration.

---

## 5.4 Complex numbers

Essential for Fourier theory.

\[
j=\sqrt{-1}
\]

Euler:

\[
e^{j\theta}
=
\cos\theta+j\sin\theta
\]

For:

\[
z=a+jb
\]

Magnitude:

\[
|z|
=
\sqrt{a^2+b^2}
\]

Phase:

\[
\phi
=
\tan^{-1}\left(\frac{b}{a}\right)
\]

This leads directly to:

\[
F(u,v)
=
|F(u,v)|e^{j\phi(u,v)}
\]

### Minimum mastery

You should be able to move between:

```text
real + imaginary
↔
magnitude + phase
```

---

## 5.5 Sinusoids

Understand:

\[
f(x)
=
A\sin(2\pi fx+\phi)
\]

where:

- \(A\) — amplitude;
- \(f\) — frequency;
- \(\phi\) — phase.

Later, image frequency means **spatial frequency**.

---

## 5.6 Convolution before images

Start in 1-D.

Then move to:

\[
g[m,n]
=
\sum_k\sum_l
f[k,l]h[m-k,n-l]
\]

Master:

- kernel;
- flipping;
- sliding;
- multiply;
- sum;
- boundaries/padding.

Convolution is one of the most important concepts in the entire course.

---

# 6. Software Setup

Install or use an environment with:

```text
Python
NumPy
Matplotlib
OpenCV
SciPy
scikit-image
Jupyter
```

Suggested imports:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import filters, morphology, restoration, segmentation, transform
```

## Rule

Use libraries to **verify** understanding, not replace it.

For core algorithms:

```text
Math
→ NumPy
→ Library
→ Compare
```

---

# 7. Official Semester Roadmap

---

# Week 1 — Introduction & Imaging Systems

## Official focus

- Introduction
- Imaging systems
- Applications

## Learn

### What is an image?

\[
f(x,y)
\]

Types:

- grayscale;
- RGB;
- binary;
- multispectral;
- depth.

### Imaging model

\[
f(x,y)
=
i(x,y)r(x,y)
\]

where:

- \(i(x,y)\) — illumination;
- \(r(x,y)\) — reflectance.

### Acquisition pipeline

```text
Scene
→ Illumination
→ Object reflectance
→ Optics
→ Sensor
→ Analog signal
→ Sampling
→ Quantization
→ Digital image
```

## Coding lab

Inspect:

```python
image.shape
image.dtype
image.min()
image.max()
```

Do:

- load;
- display;
- crop;
- inspect pixels;
- modify pixels;
- split RGB channels;
- convert to grayscale.

## Hand / quiz questions

You must answer:

- What makes an image digital?
- What is a pixel mathematically?
- Spatial resolution vs intensity resolution?
- What does bit depth mean?
- What does image acquisition do?

## Deliverable

`01_image_fundamentals/week1_image_anatomy.ipynb`

---

# Week 1+ — Camera & Imaging Systems Track

This track is deliberately placed immediately after the general Week 1 introduction.

The goal is **not** to turn CSE340 into a full medical-imaging or optical-engineering course. The goal is to understand where digital images come from, what the pixel values physically mean, what limitations are introduced during acquisition, and why later DIP operations are needed.

Use this mental model for every modality:

\[
\boxed{
\text{Physical phenomenon}
\rightarrow
\text{sensor measurement}
\rightarrow
\text{analog signal}
\rightarrow
\text{sampling/quantization}
\rightarrow
\text{digital image}
}
\]

For each imaging system, always ask:

1. What physical quantity is being measured?
2. What sensor converts it into a signal?
3. What does one pixel/voxel mean?
4. What determines spatial resolution?
5. What are the dominant noise/artifact sources?
6. What DIP operations become useful afterward?

---

## 7.1A — How a Digital Camera Works

This is the most important acquisition system to understand because it directly connects optics, sampling, color, noise, geometry, and image processing.

### 7.1A.1 Scene to image

A simplified camera pipeline is:

```text
Scene illumination
→ light reflected/emitted by objects
→ lens
→ aperture
→ image formation on sensor plane
→ CCD/CMOS sensor
→ photoelectric conversion
→ analog amplification
→ analog-to-digital conversion
→ raw sensor image
→ demosaicing / white balance / color correction
→ tone mapping / gamma / compression
→ final digital image
```

A camera image is therefore **not** a perfect copy of the world. It is the result of:

```text
optics
+
sensor physics
+
sampling
+
electronics
+
image signal processing
```

---

### 7.1A.2 Pinhole camera model

Start with the ideal pinhole camera.

Conceptually:

```text
3-D point
→ perspective projection
→ 2-D image point
```

For a point \((X,Y,Z)\):

\[
x=f\frac{X}{Z}
\]

\[
y=f\frac{Y}{Z}
\]

where \(f\) is focal length.

This is the conceptual basis of:

- perspective;
- apparent size vs distance;
- camera calibration;
- stereo vision;
- projective geometry.

You do not need a full computer-vision geometry course here, but the projection model is worth understanding.

---

### 7.1A.3 Focal length and field of view

Focal length affects:

- magnification;
- field of view;
- perspective framing.

For a sensor dimension \(d\), a simplified field-of-view relation is:

\[
FOV
=
2\tan^{-1}\left(\frac{d}{2f}\right)
\]

So:

```text
short focal length → wide field of view
long focal length  → narrow field of view
```

---

### 7.1A.4 Aperture

The aperture controls how much light enters the camera.

Commonly expressed as:

\[
N=\frac{f}{D}
\]

where:

- \(N\) — f-number;
- \(f\) — focal length;
- \(D\) — aperture diameter.

Conceptually:

```text
larger aperture
→ more light
→ shallower depth of field

smaller aperture
→ less light
→ deeper depth of field
```

Also know that extremely small apertures can introduce diffraction blur.

---

### 7.1A.5 Exposure

Image brightness depends strongly on:

```text
aperture
+
shutter time
+
sensor gain / ISO
```

#### Shutter time

Longer exposure:

```text
more photons
+
more motion blur risk
```

Shorter exposure:

```text
less motion blur
+
less captured light
```

#### Gain / ISO

Higher electronic gain can brighten the signal but also amplifies noise.

This connects directly to later DIP topics:

```text
low-light image
→ noise
→ denoising / restoration
```

---

### 7.1A.6 Focus and depth of field

Learn the conceptual distinction between:

- in-focus detail;
- defocus blur;
- motion blur;
- optical diffraction.

Blur can later be modeled approximately through a point-spread function:

\[
g=h*f+\eta
\]

which is exactly the restoration model used later in the course.

---

### 7.1A.7 CCD and CMOS image sensors

Both convert incident photons into electrical measurements.

At course level, understand:

```text
incoming photons
→ charge / electrical response
→ readout
→ digitization
```

You should know that practical sensors introduce:

- shot noise;
- read noise;
- dark current;
- saturation;
- fixed-pattern effects;
- limited dynamic range.

These are physical reasons why denoising and restoration exist.

---

### 7.1A.8 Sensor resolution vs image resolution

Do not equate:

```text
more megapixels
=
automatically better image
```

Final spatial detail depends on:

```text
lens quality
sensor size
pixel pitch
focus
motion
diffraction
signal-to-noise ratio
sampling
processing
```

This is an important DIP lesson: **resolution is a system property, not merely an image-size number.**

---

### 7.1A.9 Bayer color filter array

A typical single-sensor color camera does not directly measure full RGB at every sensor location.

A Bayer-like color filter array samples different colors at neighboring pixels.

Conceptually:

```text
sensor location measures mostly R, G, or B
→ incomplete color samples
→ demosaicing
→ estimated full RGB image
```

This connects directly to:

- sampling;
- interpolation;
- color processing;
- artifacts.

---

### 7.1A.10 Demosaicing

Demosaicing estimates missing color values at each pixel.

This is an excellent real example of interpolation.

Potential artifacts include:

- false color;
- zipper artifacts;
- moiré.

Revisit this during:

```text
Week 2 interpolation
Week 10 color
```

---

### 7.1A.11 White balance and color correction

Different illuminants change measured channel values.

Camera pipelines therefore often perform:

```text
raw sensor RGB-like measurements
→ white balance
→ color correction
→ display color space
```

This explains why image color is not simply “what the sensor saw.”

Revisit during Week 10.

---

### 7.1A.12 Dynamic range and saturation

A sensor can only represent a finite range of measurable intensities.

Too much signal:

```text
→ saturation / clipping
```

Too little signal:

```text
→ noise-dominated dark regions
```

This connects to:

- histogram interpretation;
- contrast enhancement;
- HDR imaging;
- quantization.

---

### 7.1A.13 Camera artifacts to recognize

Know the physical cause of:

- motion blur;
- defocus blur;
- sensor noise;
- rolling-shutter distortion;
- clipping;
- chromatic aberration;
- vignetting;
- aliasing / moiré;
- compression artifacts.

You do not need to derive every correction method, but you should learn to classify the artifact before choosing a DIP solution.

---

### 7.1A.14 Camera practical lab

Perform:

```text
same scene
→ different exposure
→ different ISO/gain
→ different resize factors
→ different JPEG quality
```

Measure and inspect:

- histogram;
- noise;
- saturation;
- blur;
- color changes;
- compression artifacts.

Suggested deliverable:

`01_image_fundamentals/camera_pipeline_and_artifacts.ipynb`

---

# 7.1B — Major Imaging Modalities

For this course, learn each modality at **three depths**:

### Depth 1 — Early semester

```text
physical principle
→ sensor
→ what pixel values mean
```

### Depth 2 — When the mathematics appears

Connect it to:

```text
sampling
Fourier
reconstruction
noise
restoration
segmentation
```

### Depth 3 — Optional research extension

Study modality-specific physics/algorithms only if relevant to your projects.

---

## 7.1B.1 X-ray Radiography

### Physical principle

X-rays pass through the object/body and are attenuated differently by different materials.

Simplified attenuation model:

\[
I
=
I_0 e^{-\mu x}
\]

where:

- \(I_0\) — incident intensity;
- \(I\) — measured transmitted intensity;
- \(\mu\) — attenuation coefficient;
- \(x\) — path length.

### Image formation

```text
X-ray source
→ object/tissue
→ differential attenuation
→ detector
→ 2-D projection image
```

### Pixel meaning

Pixel intensity relates to transmitted X-ray signal and therefore the total attenuation along the beam path.

### DIP connections

- contrast enhancement;
- denoising;
- dynamic range;
- segmentation;
- projection imaging.

---

## 7.1B.2 Computed Tomography — CT

CT acquires X-ray projections from many directions and reconstructs cross-sectional slices.

### Pipeline

```text
rotating X-ray source + detector
→ many projections
→ projection data / sinogram
→ reconstruction
→ CT slice
→ 3-D volume from many slices
```

### Core conceptual idea

Unlike ordinary X-ray:

```text
X-ray = projection
CT    = reconstructed cross-section / volume
```

### Pixel / voxel meaning

CT values are related to X-ray attenuation properties of tissue/material.

### DIP connections

CT should be revisited during:

```text
Fourier transforms
sampling
reconstruction
filtering
inverse problems
noise
segmentation
```

### Later concept

Filtered backprojection is a classical CT reconstruction method.

You do not need its full derivation unless your instructor extends reconstruction, but you should understand:

```text
projection measurements
→ mathematically invert imaging process
→ reconstructed image
```

---

## 7.1B.3 Magnetic Resonance Imaging — MRI

MRI does not use X-ray attenuation.

At introductory level, understand:

```text
strong static magnetic field
→ hydrogen nuclei exhibit magnetic behavior
→ radio-frequency excitation
→ measured electromagnetic response
→ spatial encoding using gradients
→ k-space data
→ reconstruction
→ MR image
```

### Very important DIP connection

MRI measurements are naturally represented in a spatial-frequency domain called **k-space**.

Conceptually:

\[
\boxed{
k\text{-space}
\xrightarrow{\text{inverse Fourier transform}}
\text{MRI image}
}
\]

Do not attempt to master all MR spin physics before Fourier.

### Early-semester understanding

Know:

- strong magnetic field;
- RF excitation;
- gradient encoding;
- received signal;
- k-space;
- reconstruction.

### Revisit during Fourier

Then learn:

- why center of k-space is associated strongly with broad contrast/low spatial frequencies;
- why outer k-space contributes finer spatial detail;
- what undersampling can do;
- how motion affects reconstruction.

### DIP connections

- Fourier transform;
- sampling;
- reconstruction;
- inverse problems;
- noise;
- segmentation.

---

## 7.1B.4 Ultrasound Imaging

### Physical principle

A piezoelectric transducer converts:

```text
electrical pulse
→ acoustic wave
```

and returning echoes back into electrical signals.

### Pipeline

```text
electrical excitation
→ transducer
→ ultrasound pulse
→ propagation through tissue/material
→ reflection/scattering at interfaces
→ returning echo
→ transducer
→ signal processing
→ image
```

### Depth

Echo arrival time helps estimate reflector depth:

\[
d
=
\frac{ct}{2}
\]

where:

- \(c\) — propagation speed;
- \(t\) — round-trip time.

### Pixel meaning

Brightness in B-mode ultrasound is related to echo strength.

### Important artifacts / properties

- speckle;
- acoustic shadow;
- enhancement;
- attenuation;
- limited axial/lateral resolution.

### DIP connections

- filtering;
- denoising;
- speckle;
- edge/region segmentation;
- contrast enhancement;
- temporal processing for video ultrasound.

---

## 7.1B.5 PET — Positron Emission Tomography

PET is primarily functional/metabolic imaging.

Conceptually:

```text
radioactive tracer
→ positron emission
→ annihilation
→ pair of gamma photons
→ coincidence detection
→ many detected events
→ tomographic reconstruction
```

### Pixel / voxel meaning

Values relate to detected tracer activity rather than ordinary optical brightness.

### DIP connections

- tomography;
- reconstruction;
- counting statistics;
- noise;
- smoothing;
- registration with CT/MRI;
- segmentation.

---

## 7.1B.6 SPECT

Concept:

```text
gamma-emitting tracer
→ gamma camera measurements
→ multiple viewing angles
→ tomographic reconstruction
```

Connections:

- reconstruction;
- attenuation effects;
- noise;
- multimodal registration.

---

## 7.1B.7 Optical Microscopy

Pipeline:

```text
illumination
→ specimen
→ objective lens
→ magnified optical image
→ camera/sensor
→ digital micrograph
```

Important concepts:

- optical resolution;
- diffraction;
- point-spread function;
- depth of field;
- noise;
- staining/contrast.

### DIP connections

- deconvolution;
- restoration;
- contrast enhancement;
- segmentation;
- cell/object counting;
- morphology.

---

## 7.1B.8 Confocal Microscopy

Confocal imaging rejects much out-of-focus light and enables optical sectioning.

Conceptually:

```text
focused illumination
→ scanned specimen
→ pinhole rejects out-of-focus light
→ optical slice
→ stack slices
→ 3-D volume
```

DIP connections:

- 3-D images;
- deconvolution;
- segmentation;
- volumetric processing.

---

## 7.1B.9 Electron Microscopy

Uses electrons rather than visible photons.

At course level, understand:

```text
electron source
→ interaction with specimen
→ detector
→ very high-resolution image
```

Connections:

- noise;
- contrast;
- extremely fine spatial detail;
- segmentation;
- texture analysis.

---

## 7.1B.10 Thermal / Infrared Imaging

Thermal cameras measure infrared radiation related to scene temperature/emissivity.

Pipeline:

```text
infrared radiation
→ IR optics
→ thermal detector
→ calibration
→ temperature-like/intensity image
```

Important:

> False-color thermal images may look RGB, but the underlying measurement can be a single thermal/radiometric channel.

DIP connections:

- calibration;
- contrast stretching;
- pseudo-color;
- thresholding;
- segmentation;
- temporal analysis.

---

## 7.1B.11 Multispectral Imaging

Captures several wavelength bands.

Representation:

\[
I(x,y,\lambda_k)
\]

Instead of only:

```text
R G B
```

you may have:

```text
Blue
Green
Red
Red-edge
Near IR
...
```

Connections:

- multi-channel processing;
- band ratios;
- segmentation;
- remote sensing;
- material discrimination.

---

## 7.1B.12 Hyperspectral Imaging

Captures many narrow spectral bands.

Think of each pixel as containing a spectrum:

\[
\mathbf{x}_{pixel}
=
[x_{\lambda_1},x_{\lambda_2},...,x_{\lambda_n}]
\]

So data becomes a spectral-spatial cube:

\[
I(x,y,\lambda)
\]

DIP connections:

- dimensionality;
- denoising;
- spectral signatures;
- segmentation/classification;
- dimensionality reduction as an extension.

---

## 7.1B.13 Stereo Depth

Two cameras view the same scene from different positions.

Pipeline:

```text
left image + right image
→ correspondence / disparity
→ triangulation
→ depth
```

Simple conceptual relationship:

\[
Z
\propto
\frac{fB}{d}
\]

where:

- \(f\) — focal length;
- \(B\) — camera baseline;
- \(d\) — disparity.

Connections:

- geometric transforms;
- interpolation;
- correspondence;
- depth images.

---

## 7.1B.14 Time-of-Flight — ToF Depth Camera

Measures light travel time or phase.

Conceptually:

```text
emit modulated / pulsed light
→ reflection
→ measure delay/phase
→ estimate distance
→ depth image
```

Simplified pulsed relationship:

\[
d
=
\frac{ct}{2}
\]

Connections:

- depth-image noise;
- missing values;
- multipath;
- filtering;
- segmentation.

---

## 7.1B.15 Structured-Light Depth

Projects a known pattern onto a scene.

Pipeline:

```text
project pattern
→ observe deformation
→ infer geometry
→ depth map
```

Connections:

- image geometry;
- correspondence;
- calibration;
- depth processing.

---

## 7.1B.16 LiDAR

LiDAR sends laser light and measures return time/phase.

Conceptually:

```text
laser emission
→ reflection
→ timing/ranging
→ 3-D points
```

Data may appear as:

- point cloud;
- depth/range image;
- intensity image.

Connections:

- sampling;
- rasterization;
- range-image filtering;
- segmentation;
- sensor fusion.

---

## 7.1B.17 Optical Coherence Tomography — OCT

OCT uses optical interference/coherence methods to obtain depth-resolved structural information.

Conceptually:

```text
low-coherence light
→ interference measurement
→ depth profile
→ repeated scans
→ cross-sectional image
```

Connections:

- reconstruction;
- speckle;
- denoising;
- segmentation.

---

## 7.1B.18 SAR / Radar Imaging

Synthetic Aperture Radar uses microwave/radar measurements and signal processing to form an image.

Conceptually:

```text
transmit radar
→ receive reflections over motion/aperture
→ coherent signal processing
→ reconstructed radar image
```

Important characteristics:

- coherent imaging;
- speckle;
- non-optical contrast;
- geometry/distortion.

Connections:

- Fourier/signal processing;
- reconstruction;
- filtering;
- speckle;
- remote sensing.

---

# 7.1C — Imaging Modality Comparison Table

| Modality | Energy / physical principle | Sensor / measurement | Output meaning | Strong DIP connection |
|---|---|---|---|---|
| RGB camera | visible photons | CCD/CMOS | reflected light / color | sampling, color, interpolation, noise |
| X-ray | ionizing X-rays | X-ray detector | projection attenuation | contrast, denoising |
| CT | many X-ray projections | rotating detectors | reconstructed attenuation volume | Fourier, reconstruction, inverse problems |
| MRI | magnetic resonance + RF | receiver coil | MR signal-derived tissue contrast | Fourier/k-space, sampling |
| Ultrasound | acoustic echoes | piezoelectric transducer | echo amplitude vs depth | speckle, filtering, segmentation |
| PET | gamma coincidence events | detector ring | tracer activity | tomography, statistics |
| SPECT | gamma emissions | gamma camera | tracer distribution | tomography, noise |
| Microscopy | visible/fluorescent light | camera/detector | microstructure | PSF, deconvolution, segmentation |
| Thermal | infrared radiation | IR detector | temperature/emissivity-related signal | calibration, thresholding |
| Multispectral | selected wavelength bands | spectral sensor | several bands per pixel | multi-channel processing |
| Hyperspectral | many narrow bands | spectrometer/imager | spectrum per pixel | spectral-spatial processing |
| Stereo | triangulation | two cameras | depth/disparity | geometry, correspondence |
| ToF | light travel time / phase | ToF sensor | distance | depth filtering |
| LiDAR | laser ranging | photodetector | 3-D range points | sampling, segmentation |
| OCT | optical interference | interferometric detector | depth-resolved optical structure | reconstruction, speckle |
| SAR | coherent microwave radar | radar receiver | radar reflectivity | Fourier, speckle, reconstruction |

---

# 7.1D — The Revisit Strategy

Do **not** study every imaging modality to full physics depth in Week 1.

Instead, revisit them when their mathematical connection appears.

## Week 2 — Sampling / interpolation

Revisit:

```text
camera sensor
Bayer sampling
demosaicing
stereo/depth grids
```

Ask:

> What information was sampled, and what happens when sampling is insufficient?

---

## Weeks 3–4 — Histograms / filtering

Revisit:

```text
low-light camera
X-ray contrast
thermal images
microscopy
```

Ask:

> What does the intensity distribution physically represent?

---

## Weeks 5–8 — Fourier / frequency domain

Revisit:

```text
MRI k-space
CT reconstruction intuition
SAR
optical blur
sampling / aliasing
```

Ask:

> Is the measured information already related to frequency-domain data, and what does reconstruction require?

---

## Week 9 — Restoration

Revisit:

```text
camera defocus / motion blur
microscope PSF
ultrasound speckle
CT/MRI noise
```

Use:

\[
g=h*f+\eta
\]

and ask:

> What does \(h\) physically mean for this imaging system?

---

## Week 10 — Color

Revisit:

```text
camera ISP
Bayer CFA
white balance
YCbCr
thermal false color
multispectral/hyperspectral data
```

Ask:

> Are these channels actual physical spectral measurements, reconstructed colors, or visualization choices?

---

## Week 11 — Compression

Revisit:

```text
JPEG camera pipeline
medical-image compression concerns
spectral-image storage
```

Ask:

> Which information is safe to discard, and which may be diagnostically/analytically important?

---

## Weeks 12–13 — Morphology / segmentation

Apply the same operations to:

```text
natural camera image
X-ray / CT
MRI
ultrasound
microscopy
thermal
depth maps
```

This reinforces the important DIP idea:

\[
\boxed{
\text{Different acquisition physics}
\rightarrow
\text{different image meaning}
\rightarrow
\text{many shared mathematical processing tools}
}
\]

---

# Week 2 — Digital Image Fundamentals

## Official focus

- Human visual system
- Digital image formation
- Sampling and quantization
- Interpolation
- Geometric transformation

---

## 7.2.1 Human visual system

Learn only the DIP-relevant parts:

- rods vs cones;
- brightness adaptation;
- contrast;
- spatial sensitivity;
- perceived vs physical brightness.

Avoid spending too much time on biology.

---

## 7.2.2 Sampling

Continuous:

\[
f(x,y)
\]

Sampled:

\[
f[m,n]
\]

Sampling discretizes **space**.

Understand:

- spatial sampling frequency;
- resolution;
- undersampling;
- reconstruction;
- aliasing.

Nyquist connection:

\[
f_s\ge 2f_{\max}
\]

---

## 7.2.3 Quantization

Quantization discretizes **intensity**.

For \(k\) bits:

\[
L=2^k
\]

Examples:

\[
4\text{-bit}=16
\]

\[
8\text{-bit}=256
\]

\[
10\text{-bit}=1024
\]

---

## 7.2.4 Interpolation

### Nearest neighbour

- fastest;
- blocky;
- uses closest sample.

### Bilinear

- uses four neighbours;
- interpolate across both axes.

### Bicubic

- larger neighbourhood;
- smoother;
- more computation.

### Hand mastery

Solve at least one complete bilinear interpolation problem manually.

---

## 7.2.5 Geometric transforms

### Translation

\[
\begin{bmatrix}
x'\\
y'\\
1
\end{bmatrix}
=
\begin{bmatrix}
1&0&t_x\\
0&1&t_y\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
\]

### Scaling

\[
\begin{bmatrix}
s_x&0&0\\
0&s_y&0\\
0&0&1
\end{bmatrix}
\]

### Rotation

\[
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\]

Also understand:

- affine transformation;
- homogeneous coordinates;
- forward vs inverse mapping;
- interpolation during warping.

## Mini-project

```text
Original
→ Downsample
→ Upsample with nearest
→ Upsample with bilinear
→ Upsample with bicubic
→ compare visually
→ compare MSE
```

## Deliverable

`02_sampling_quantization/resampling_interpolation.ipynb`

---

# Weeks 3–4 — Spatial Image Processing

This is a high-priority quiz block.

## Official focus

- Registration
- Intensity transformations
- Histogram processing
- Convolution
- Spatial filtering

---

## 7.3.1 Intensity transformations

### Negative

\[
s=L-1-r
\]

### Log

\[
s=c\log(1+r)
\]

### Gamma / power-law

\[
s=cr^\gamma
\]

Typically:

\[
\gamma<1
\]

brightens many images, while:

\[
\gamma>1
\]

darkens them.

Learn the mapping visually, not only algebraically.

---

## 7.3.2 Contrast stretching

Understand:

- piecewise-linear transforms;
- clipping;
- thresholding;
- dynamic range expansion.

---

## 7.3.3 Histograms

\[
h(r_k)=n_k
\]

Normalized:

\[
p(r_k)=\frac{n_k}{MN}
\]

Learn to infer:

- underexposure;
- overexposure;
- low contrast;
- high contrast.

---

## 7.3.4 Histogram equalization

Core mapping:

\[
s_k
=
(L-1)
\sum_{j=0}^{k}
p(r_j)
\]

Interpretation:

> use the cumulative distribution of intensity to redistribute available levels.

### Quiz target

Solve **at least 5 complete small histogram-equalization problems manually**.

---

## 7.3.5 Correlation vs convolution

Convolution flips the kernel.

Correlation does not.

This is a classic trap.

---

## 7.3.6 Smoothing filters

Learn:

- mean;
- weighted mean;
- Gaussian;
- median.

### Mean kernel

\[
\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}
\]

### Gaussian

\[
G(x,y)
=
\frac{1}{2\pi\sigma^2}
e^{-\frac{x^2+y^2}{2\sigma^2}}
\]

Understand:

\[
\sigma\uparrow
\Rightarrow
\text{stronger smoothing}
\]

### Median

Important for impulse noise.

> **2026 professor-sequence note:** Median is useful textbook context, but Lecture 2's actual progression goes beyond Gaussian smoothing into **Bilateral Filtering and Non-Local Means**. Those topics are therefore slide-core even though they are not central in the uploaded Gonzalez Chapters 1–4.

### Bilateral filtering — 2026 slide core

\[
w(p,q)
=
\exp\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right)
\exp\left(-\frac{(I(p)-I(q))^2}{2\sigma_r^2}\right)
\]

Learn it as:

```text
Gaussian spatial closeness
×
intensity similarity
=
edge-aware weighted average
```

Do not merely memorize the equation. Be able to explain what happens as \(\sigma_s\) and \(\sigma_r\) change and why a large \(\sigma_r\) makes the behavior approach ordinary Gaussian smoothing.

### Non-Local Means — 2026 slide core

\[
w(p,q)
=
\exp\left(
-\frac{\|P(p)-P(q)\|^2}{h^2}
\right)
\]

Learn the progression:

```text
single-pixel similarity
→ patch similarity
→ distant self-similar structures contribute
→ stronger denoising of repeated texture
→ higher computational cost
```

**Assignment research is parked here** and activated only after this core is mastered.

---

## 7.3.7 Sharpening

Learn:

- gradients;
- Laplacian;
- unsharp masking;
- high-boost filtering.

Conceptually:

```text
blurred / low-frequency content
vs
rapid variation / high-frequency content
```

---

## 7.3.8 Registration

Pipeline:

```text
Reference image + Moving image
→ correspondences / similarity
→ estimate transform
→ warp
→ alignment
```

Know:

- translation;
- rotation;
- scale;
- affine;
- interpolation.

---

## Build target

Implement:

```python
negative()
gamma_transform()
histogram_manual()
hist_equalization_manual()
correlate2d_manual()
convolve2d_manual()
mean_filter_manual()
gaussian_filter_manual()
median_filter_manual()
laplacian_filter_manual()
```

Then compare with OpenCV/SciPy.

## Deliverable

`05_spatial_filtering/spatial_processing_toolkit.ipynb`

---

# Weeks 5–6 — Fourier Transform and DFT

This is the mathematically hardest block. Give it extra time.

## Official focus

- Fourier transform
- Convolution
- Sampling and reconstruction
- Aliasing
- DFT

---

## 7.4.1 Spatial frequency

Slow change in intensity:

```text
low spatial frequency
```

rapid changes, edges, texture:

```text
high spatial frequency
```

---

## 7.4.2 Fourier idea

Represent an image using frequency components:

\[
f(x,y)
\leftrightarrow
F(u,v)
\]

Spatial domain:

\[
(x,y)
\]

Frequency domain:

\[
(u,v)
\]

---

## 7.4.3 2-D DFT

Recognize and understand:

\[
F(u,v)
=
\sum_{x=0}^{M-1}
\sum_{y=0}^{N-1}
f(x,y)
e^{-j2\pi
\left(
\frac{ux}{M}
+
\frac{vy}{N}
\right)}
\]

Inverse:

\[
f(x,y)
=
\frac{1}{MN}
\sum_u
\sum_v
F(u,v)
e^{j2\pi
\left(
\frac{ux}{M}
+
\frac{vy}{N}
\right)}
\]

Do not memorize this as a string of symbols. Be able to explain:

- \(M,N\);
- \(x,y\);
- \(u,v\);
- exponential basis;
- normalization;
- forward/inverse sign.

---

## 7.4.4 Magnitude and phase

If:

\[
F=R+jI
\]

then:

\[
|F|
=
\sqrt{R^2+I^2}
\]

\[
\phi
=
\tan^{-1}(I/R)
\]

### Required experiment

Take two images:

```text
Image A magnitude + Image B phase
Image B magnitude + Image A phase
```

Reconstruct both.

Goal:

> understand why phase carries strong structural information.

---

## 7.4.5 FFT vs DFT

Understand:

- DFT = mathematical transform;
- FFT = efficient algorithm for computing DFT.

Do not say “FFT is another transform.”

---

## 7.4.6 `fftshift`

Raw FFT often places DC at a corner.

`fftshift` puts zero frequency at the center.

After shifting:

```text
center      → low frequencies
farther out → high frequencies
```

---

## 7.4.7 DFT/Fourier properties

Master:

### Linearity

\[
af+bg
\leftrightarrow
aF+bG
\]

### Translation

Spatial translation changes frequency-domain phase.

### Scaling

### Rotation

Rotation in spatial domain corresponds to rotation in frequency domain.

### Conjugate symmetry

For real images:

\[
F(-u,-v)=F^*(u,v)
\]

### Separability

2-D DFT can be computed as consecutive 1-D transforms.

### Convolution theorem

\[
f*h
\leftrightarrow
FH
\]

This is one of the most important bridges in the course.

---

## 7.4.8 Aliasing

When:

\[
f_s<2f_{\max}
\]

different high frequencies become indistinguishable after sampling.

Visual symptoms:

- jagged edges;
- moiré;
- false texture;
- repeating artifacts.

## Required labs

1. FFT of constant image.
2. FFT of vertical stripe pattern.
3. FFT of horizontal stripe pattern.
4. FFT of rotated stripes.
5. FFT magnitude of natural image.
6. Change only phase.
7. Demonstrate aliasing by downsampling without appropriate prefiltering.

## Deliverable

`06_fourier/fourier_intuition_and_dft.ipynb`

---

# Weeks 7–8 — Frequency-Domain Filtering

## Official focus

- DFT properties
- Filtering in frequency domain

## Core pipeline

```text
f(x,y)
→ FFT
→ F(u,v)
→ multiply by H(u,v)
→ G(u,v)
→ inverse FFT
→ g(x,y)
```

or:

\[
G(u,v)=H(u,v)F(u,v)
\]

---

## 7.5.1 Low-pass filters

Purpose:

```text
suppress high frequencies
→ smoothing
```

Learn:

- Ideal LPF;
- Butterworth LPF;
- Gaussian LPF.

---

## 7.5.2 High-pass filters

Purpose:

```text
suppress low-frequency content
→ emphasize edges / detail
```

Learn:

- Ideal HPF;
- Butterworth HPF;
- Gaussian HPF.

---

## 7.5.3 Ringing

Understand why a sharp frequency cutoff can cause oscillatory spatial artifacts.

Connect:

```text
Ideal cutoff
→ abrupt frequency discontinuity
→ ringing in spatial domain
```

---

## 7.5.4 The major bridge

You should now see:

```text
Spatial smoothing
↔
Frequency-domain low-pass filtering
```

and:

```text
Spatial sharpening / edge emphasis
↔
High-frequency emphasis
```

This is the point where the course should begin to feel like one coherent system.

## Build

Implement masks yourself:

```python
ideal_lpf()
gaussian_lpf()
butterworth_lpf()
ideal_hpf()
gaussian_hpf()
butterworth_hpf()
```

Compare:

- cutoff frequency;
- Butterworth order;
- spatial ringing;
- blur strength.

## Deliverable

`07_frequency_filtering/frequency_filter_comparison.ipynb`

---

# Week 9 — Image Restoration

## Official focus

- Noise models
- Linear and non-linear filters
- Wiener filtering
- Constrained least-squares filtering
- Reconstruction if time permits

---

## 7.6.1 Enhancement vs restoration

Enhancement:

> make the image look better.

Restoration:

> estimate the original using a degradation model.

---

## 7.6.2 Degradation model

\[
g(x,y)
=
h(x,y)*f(x,y)+\eta(x,y)
\]

Frequency form:

\[
G(u,v)
=
H(u,v)F(u,v)+N(u,v)
\]

where:

- \(f\) — original;
- \(h\) — degradation / PSF;
- \(g\) — observed image;
- \(\eta\) — noise.

---

## 7.6.3 Noise models

Know:

### Gaussian

\[
p(z)
=
\frac{1}{\sqrt{2\pi}\sigma}
e^{-(z-\mu)^2/(2\sigma^2)}
\]

### Salt-and-pepper

Sparse extreme pixels.

### Uniform

### Poisson

---

## 7.6.4 Filters vs noise

Build intuition:

```text
Salt-and-pepper → median is often strong
Gaussian noise  → averaging/Gaussian/Wiener-type reasoning
Blur + noise    → restoration/deconvolution problem
```

Do not memorize one-filter-per-noise blindly. Understand assumptions.

---

## 7.6.5 Inverse filtering

Naive idea:

\[
\hat F(u,v)
=
\frac{G(u,v)}{H(u,v)}
\]

Problem:

\[
H(u,v)\approx0
\]

causes noise amplification / numerical instability.

---

## 7.6.6 Wiener filtering

Concept:

> undo degradation while accounting for noise.

Typical form:

\[
\hat F
=
\left[
\frac{H^*}
{|H|^2+S_\eta/S_f}
\right]G
\]

Understand:

- \(H^*\);
- inverse component;
- noise-to-signal tradeoff.

---

## 7.6.7 Constrained least squares

Think:

\[
\text{reconstruction error}
+
\lambda
\text{regularization}
\]

This is a powerful conceptual bridge to modern inverse problems and machine learning.

## Lab

Generate:

- clean image;
- Gaussian blur;
- motion blur;
- Gaussian noise;
- impulse noise.

Compare:

- mean;
- Gaussian;
- median;
- inverse;
- Wiener;
- optionally constrained/regularized methods.

Evaluate:

\[
MSE
\]

\[
PSNR
\]

and optionally SSIM.

## Deliverable

`08_restoration/restoration_benchmark.ipynb`

---

# Week 10 — Color Image Fundamentals

## Official focus

Color Image Fundamentals.

---

## 7.7.1 RGB

Additive model:

\[
(R,G,B)
\]

Understand channels and composite color.

---

## 7.7.2 CMY / CMYK

Normalized:

\[
C=1-R
\]

\[
M=1-G
\]

\[
Y=1-B
\]

---

## 7.7.3 HSV / HSI

Intuition:

- Hue — color identity;
- Saturation — purity;
- Value/Intensity — brightness-like component.

Understand why separating brightness-like information from chromatic identity can simplify some image-processing tasks.

---

## 7.7.4 YCbCr

```text
Y  → luminance-like component
Cb → chroma
Cr → chroma
```

Important because it connects directly to JPEG.

## Lab

For one image:

- visualize R/G/B;
- visualize HSV channels;
- visualize Y/Cb/Cr;
- alter only luminance;
- alter only chroma;
- compare effects.

## Deliverable

`09_color/color_space_exploration.ipynb`

---

# Week 11 — DCT and Image Compression

## Official focus

- DCT
- Image Compression

---

## 7.8.1 Why compression is possible

Learn the idea of redundancy:

- coding redundancy;
- interpixel/spatial redundancy;
- psychovisual redundancy.

---

## 7.8.2 Lossless vs lossy

Lossless:

> exact reconstruction.

Lossy:

> approximate reconstruction in exchange for stronger compression.

JPEG is the key course example.

---

## 7.8.3 DCT

Concept:

```text
8×8 spatial pixels
→ cosine-basis coefficients
```

Smooth image blocks tend to concentrate much energy into lower-frequency coefficients.

Understand:

- DC coefficient;
- AC coefficients;
- energy compaction.

---

## 7.8.4 JPEG pipeline

Memorize and understand:

```text
RGB
→ YCbCr
→ optional chroma subsampling
→ 8×8 blocks
→ level shift
→ 2-D DCT
→ quantization
→ zig-zag scan
→ run-length / symbol preparation
→ entropy coding
→ JPEG bitstream
```

Decoder conceptually reverses:

```text
entropy decode
→ inverse ordering
→ dequantization
→ IDCT
→ blocks
→ YCbCr
→ RGB
```

---

## 7.8.5 Quantization

\[
C_q(i,j)
=
\operatorname{round}
\left(
\frac{C(i,j)}{Q(i,j)}
\right)
\]

This is where much of JPEG's irreversible loss is introduced.

---

## Lab 1 — JPEG quality study

Test:

```text
100
90
70
50
30
10
```

Record:

- file size;
- MSE;
- PSNR;
- visual artifacts.

---

## Lab 2 — Simplified JPEG

Implement the conceptual pipeline yourself.

You do **not** need to create a standards-compliant JPEG encoder.

The goal is:

> understand DCT + quantization + reconstruction.

## Deliverable

`10_dct_compression/jpeg_from_concepts.ipynb`

---

# Week 12 — Mathematical Morphology

## Official focus

Morphological Image Processing.

## Core object

\[
\boxed{\text{Structuring element}}
\]

Morphology asks how a shape changes when probed by a chosen structuring element.

---

## 7.9.1 Dilation

\[
A\oplus B
\]

Conceptually:

> expand/grow foreground.

---

## 7.9.2 Erosion

\[
A\ominus B
\]

Conceptually:

> shrink foreground.

---

## 7.9.3 Opening

\[
A\circ B
=
(A\ominus B)\oplus B
\]

Often useful for removing small foreground structures.

---

## 7.9.4 Closing

\[
A\bullet B
=
(A\oplus B)\ominus B
\]

Often useful for filling small holes/gaps or joining nearby structures.

---

## Hand mastery

Use a tiny binary image and 3×3 structuring element.

Manually calculate:

- dilation;
- erosion;
- opening;
- closing.

This is ideal quiz material.

## Lab

Use noisy binary shapes and systematically change:

- structuring-element size;
- square vs disk;
- operation order.

## Deliverable

`11_morphology/morphology_by_hand_and_code.ipynb`

---

# Week 13 — Image Segmentation

## Official focus

- Point detection
- Line detection
- Edge detection
- Thresholding

---

## 7.10.1 Classification vs detection vs segmentation

Classification:

> What is present?

Detection:

> Where is the object?

Segmentation:

> Which pixels belong to it?

---

## 7.10.2 Gradient

\[
\nabla f
=
\begin{bmatrix}
\partial f/\partial x\\
\partial f/\partial y
\end{bmatrix}
\]

Magnitude:

\[
|\nabla f|
=
\sqrt{G_x^2+G_y^2}
\]

Direction:

\[
\theta
=
\tan^{-1}\left(\frac{G_y}{G_x}\right)
\]

---

## 7.10.3 Sobel

Typical:

\[
G_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}
\]

\[
G_y=
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix}
\]

Manually apply to a small patch.

---

## 7.10.4 Laplacian

\[
\nabla^2f
=
\frac{\partial^2f}{\partial x^2}
+
\frac{\partial^2f}{\partial y^2}
\]

Typical kernel:

\[
\begin{bmatrix}
0&1&0\\
1&-4&1\\
0&1&0
\end{bmatrix}
\]

Understand:

- first derivative vs second derivative;
- noise sensitivity;
- sharpening/edge behavior.

---

## 7.10.5 Canny

Pipeline:

```text
Gaussian smoothing
→ gradient magnitude and orientation
→ non-maximum suppression
→ double threshold
→ edge tracking by hysteresis
```

Know **why** each stage exists.

---

## 7.10.6 Thresholding

Global:

\[
g(x,y)
=
\begin{cases}
1,&f(x,y)>T\\
0,&\text{otherwise}
\end{cases}
\]

Learn:

- global;
- adaptive/local;
- histogram-based thresholding.

---

## 7.10.7 Otsu

Class probabilities:

\[
\omega_0,\omega_1
\]

Class means:

\[
\mu_0,\mu_1
\]

Between-class variance:

\[
\sigma_B^2
=
\omega_0\omega_1(\mu_0-\mu_1)^2
\]

Select:

\[
T^*
=
\arg\max_T\sigma_B^2(T)
\]

## Lab

Build:

```text
Raw image
→ denoise
→ edge or threshold
→ morphology
→ connected components
→ final segmented objects
```

## Deliverable

`12_segmentation/classical_segmentation_pipeline.ipynb`

---

# Week 13+ — Feature Representation Depth Extension

The weekly table ends at segmentation, but the broader course outcomes include feature representation.

Do this **after course-critical material**.

Learn:

- connected components;
- contours;
- area/perimeter;
- centroid;
- image moments;
- shape descriptors;
- corners;
- Harris;
- basic descriptor intuition;
- SIFT at a conceptual level.

---

## Harris corner detector

Local second-moment matrix:

\[
M=
\sum
w(x,y)
\begin{bmatrix}
I_x^2&I_xI_y\\
I_xI_y&I_y^2
\end{bmatrix}
\]

Response:

\[
R
=
\det(M)
-
k[\operatorname{trace}(M)]^2
\]

Intuition:

```text
both eigenvalues small → flat
one large              → edge
both large             → corner
```

---

# 8. The Four-Quiz System

The exact quiz boundaries should be updated when your instructor announces them. Until then, use this default partition.

| Quiz | Working coverage | Highest-priority material |
|---|---|---|
| **Q1 — provisional until professor confirms boundary** | Lecture 0 + Lectures 1–3 | Signals/convolution/Fourier foundations; acquisition; intensity transforms; smoothing; bilateral/NLM; sharpening; gradients; edge detection |
| **Q2 — provisional** | Frequency-domain block after Lecture 3 | DFT/2-D DFT, magnitude/phase, aliasing, properties, convolution theorem, LPF/HPF/selective filtering |
| **Q3 — provisional** | Restoration + color | Noise, degradation model, inverse/Wiener/CLS, color spaces |
| **Q4 — provisional** | Compression + morphology + segmentation | DCT/JPEG, morphology, segmentation, thresholding, Otsu |

> Once an official quiz boundary is announced, replace the provisional row rather than guessing.

---

# 9. Quiz Preparation Protocol

For each quiz:

## D−7 to D−5 — Understand

- lecture material;
- textbook;
- intuition;
- examples.

No panic memorization.

## D−4 — Formula Day

Derive and condense equations into one sheet.

## D−3 — Numerical Day

Manual calculations only.

Examples:

- bilinear interpolation;
- histogram equalization;
- convolution;
- DFT;
- morphology;
- Sobel/Otsu.

## D−2 — Mixed Closed-Book Set

No notes.

No LLM.

No computer.

## D−1 — Mock Quiz

Target:

\[
\boxed{\ge85\%}
\]

If below target:

```text
Do not reread everything.
Identify weak topics.
Drill only those topics.
Retest.
```

---

# 10. Quiz Notebook

Maintain a separate notebook:

```text
DIP_QUIZ_BOOK.md
```

Every topic gets exactly:

```text
1. Concept
2. Equation
3. One worked example
4. Common trap
5. One comparison question
```

Example:

## Convolution

**Concept:** weighted neighborhood operation with kernel reversal.

**Equation:**

\[
g[m,n]
=
\sum_k\sum_l f[k,l]h[m-k,n-l]
\]

**Worked example:** one 3×3 calculation.

**Trap:** correlation does not flip the kernel.

**Comparison:** convolution vs multiplication in the Fourier domain.

---

# 11. High-Probability Quiz Question Bank

## Fundamentals

- How many levels exist with \(k\) bits?
- Compute memory required for an \(M\times N\) image.
- Sampling vs quantization.
- Spatial vs intensity resolution.
- Explain aliasing.
- Manual nearest-neighbour interpolation.
- Manual bilinear interpolation.
- Apply coordinate transformation.

## Spatial domain

- Negative/log/gamma output.
- Calculate histogram.
- Perform histogram equalization.
- Apply a 3×3 convolution kernel.
- Correlation vs convolution.
- Mean vs Gaussian vs median.
- Effect of Gaussian \(\sigma\).
- Laplacian sharpening.

## Fourier

- Compute a tiny DFT.
- Explain FFT vs DFT.
- Interpret DC.
- Magnitude vs phase.
- Fourier shift property.
- Conjugate symmetry.
- Convolution theorem.
- Explain aliasing using frequency reasoning.
- LPF vs HPF.
- Ideal vs Butterworth vs Gaussian.
- Why ideal filters ring.

## Restoration

- Identify a noise model.
- Select a sensible filter.
- Write degradation model.
- Why inverse filtering is unstable.
- Explain Wiener tradeoff.
- Enhancement vs restoration.

## Color

- RGB vs CMY.
- HSV interpretation.
- YCbCr components.
- Why chroma can be compressed more aggressively.

## Compression

- DCT intuition.
- DC vs AC.
- Full JPEG sequence.
- Why quantization is lossy.
- Zig-zag purpose.
- Lossy vs lossless.
- File size vs quality tradeoff.

## Morphology

- Structuring element.
- Hand erosion.
- Hand dilation.
- Opening vs closing.
- Effect of larger structuring element.

## Segmentation

- Gradient magnitude.
- Apply Sobel.
- First vs second derivative.
- Canny stages.
- Global vs adaptive threshold.
- Compute Otsu objective.
- Edge detection vs segmentation.

---


# 11A. Slide-Aligned Quiz Question Bank — Current Corpus

This bank takes priority over generic questions until the first quiz.

## Lecture 0

- Explain why convolution arises from impulse decomposition of an LSI/LTI system.
- Compute a discrete convolution using an asymmetric kernel.
- Compare convolution and correlation using an asymmetric kernel.
- Solve a continuous rectangle-convolution problem piecewise.
- Explain PSF and why blur can be modeled by convolution.
- Define spatial frequency.
- Explain why sharp edges require high-frequency components.
- Explain magnitude vs phase.
- Explain \(h \leftrightarrow H\).
- Derive/explain \(x*h \leftrightarrow XH\).
- Next: calculate a 4-point DFT manually.

## Lecture 1

- Draw the scene → lens → sensor → ADC → image pipeline and explain information loss.
- Why does a raw photosite not directly give RGB?
- Why is Bayer RGGB green-heavy?
- What is demosaicing and where can zippering/color artifacts occur?
- What exactly is k-space in the professor's MRI abstraction?
- Center vs outer k-space: what image information is retained?
- Define CT projection, Radon transform, sinogram, backprojection.
- Why does simple backprojection blur?
- Explain A-line and B-mode ultrasound.
- Derive/use \(d=ct/2\).
- Distinguish point processing from neighborhood processing.
- Apply negative/log/gamma to sample intensities.
- Explain effects of \(\gamma<1\), \(\gamma=1\), \(\gamma>1\).
- Build/interpret a contrast-stretching piecewise map.
- Explain intensity slicing with/without background.
- Convert a pixel to binary and identify its bit-plane contributions.

## Lecture 2

- Compare zero/replicate/reflect padding.
- Compute a 3×3 mean-filter output.
- Explain why normalization preserves flat-region intensity.
- Why does increasing window size blur more?
- Derive/sample a small Gaussian kernel conceptually and normalize it.
- What does \(\sigma\) change?
- Explain separability and arithmetic savings.
- Why is averaging a low-pass operation without mentioning Fourier first?
- Why does Gaussian smoothing blur real edges?
- Write and explain every factor in the bilateral weight.
- What happens as \(\sigma_r\to\) very large?
- Why does bilateral preserve an edge that Gaussian crosses?
- Explain bilateral base/detail decomposition.
- Explain why NLM compares patches instead of pixels.
- What do patch size, search region and \(h\) control?
- Why is NLM computationally expensive?
- Compare Box → Gaussian → Bilateral → NLM in one coherent progression.

## Lecture 3

- Compute first and second finite differences on flat/ramp/step signals.
- Why is the second derivative more sensitive to fine detail and noise?
- Derive the 4-neighbor discrete Laplacian.
- Recognize 4-neighbor, 8-neighbor and negated Laplacian masks.
- Choose the correct sharpening sign in \(g=f+c\nabla^2f\).
- Derive the composite center-5 sharpening mask.
- Explain unsharp masking from blur subtraction.
- Distinguish \(k=1\) from \(k>1\).
- Why can large \(k\) create halos?
- Compute \(G_x,G_y\) and gradient magnitude on a small patch.
- Explain why a horizontal derivative mask detects vertical edges.
- Compare Roberts, Prewitt and Sobel.
- Why do derivatives amplify noise?
- Explain smooth → differentiate → decide.
- Explain LoG zero crossings.
- Reconstruct the five Canny stages from memory and explain the purpose of each.
- Explain the effect of Canny \(\sigma\) and thresholds.

## Closed-book standard

A topic is not quiz-ready until you can:

```text
define it
→ derive/justify it
→ calculate a small example
→ predict a visual outcome
→ compare it with the neighboring method
```


# 12. Assignment System — 40%

Use LLM help as:

```text
Tutor
+ mathematical checker
+ debugger
+ test-case generator
+ report reviewer
```

not merely:

```text
assignment answer generator
```

Otherwise there is a dangerous failure mode:

```text
High assignment grade
+
weak closed-book understanding
=
quiz loss
```

---


## 12.0A Assignment-1 Bilateral / NLM Research Lane

**Activation condition:** do not begin the full research comparison until:
1. the official assignment statement is available, and
2. Lecture 2 core concepts are at least **Taught + hand-practice complete**.

When activated, use this staged plan:

```text
A. Reproduce simple Gaussian baseline
B. Reproduce bilateral filter
C. Reproduce classical NLM
D. Design controlled noise experiments
E. Compare edge + texture retention
F. Add one research extension only
G. Write ablations and limitations
```

Potential future extension supplied by the student:
- low-rank / operator-improvement view of NLM;
- Chebyshev-polynomial implementation ideas;
- two-stage improved NLM;
- BM3D comparison if justified.

Potential bilateral depth source supplied by the student:
- MIT/CSAIL *A Gentle Introduction to Bilateral Filtering and its Applications*.

**Do not let Assignment 1 become a giant denoising survey.** One clean hypothesis + controlled comparison is preferable to many shallow methods.

---

## 12.1 Nine-step assignment workflow

```text
1. Read the problem
2. Identify the DIP concept
3. Write the relevant mathematics
4. Write pseudocode
5. Attempt implementation
6. Use LLM to debug / explain
7. Validate against library or known result
8. Analyze the output, not just produce it
9. Generate 3 closed-book quiz questions from the assignment
```

---

## 12.2 Two-implementation rule

For every central algorithm:

```text
A. From scratch — NumPy
B. Reference — OpenCV / SciPy / scikit-image
```

Then test:

\[
\text{manual/NumPy}
\approx
\text{library}
\]

within expected boundary / numerical differences.

---

# 13. Private Assignment Progression

These are not predictions of your instructor's exact assignments. They are your **practice build sequence**.

## A1 — Image fundamentals

Build:

- image reading;
- grayscale conversion;
- resampling;
- quantization;
- nearest/bilinear;
- geometric transforms.

## A2 — Spatial processing

Build:

- histograms;
- equalization;
- convolution;
- Gaussian;
- median;
- sharpening.

## A3 — Fourier

Build:

```text
image
→ FFT
→ spectrum
→ frequency mask
→ LPF / HPF
→ inverse FFT
```

## A4 — Restoration

Build degradation simulator:

```text
clean
→ blur
→ noise
→ restoration methods
→ comparison
```

## A5 — Compression

Build simplified JPEG.

## A6 — Morphology + segmentation

```text
noisy binary / grayscale image
→ threshold
→ morphology
→ component extraction
→ final regions
```

---

# 14. Midsem — 10%

Treat this as an integration exercise rather than a separate topic.

## Strong default template

### Image Enhancement & Restoration Benchmark

```text
Original
→ apply controlled degradations
→ restore using several methods
→ calculate metrics
→ compare visually
→ explain why methods differ
```

Candidate methods:

- mean;
- Gaussian;
- median;
- frequency LPF;
- Wiener.

Metrics:

\[
MSE
\]

\[
PSNR
\]

Optional:

\[
SSIM
\]

### Strong report structure

```text
1. Problem
2. Degradation model
3. Methods
4. Experimental setup
5. Metrics
6. Results
7. Visual analysis
8. Failure cases
9. Conclusion
```

---

# 15. Endsem — 10%

A strong full-course architecture is:

```text
Raw image
→ enhancement
→ denoising
→ color-space choice
→ segmentation
→ morphological cleanup
→ connected components / features
→ output representation
```

Possible domains:

- document cleanup;
- coin segmentation;
- microscopy cells;
- industrial defect regions;
- road markings/sign preprocessing;
- medical-image preprocessing;
- surface-texture segmentation;
- simple object counting.

Keep the intellectual center on **classical DIP**.

Do not replace the pipeline with YOLO or a pretrained segmentation model unless the instructor explicitly permits it and the point is a comparison.

---

# 16. Why This Course Is Useful in Your Existing CV/AI Terms

This section is a bridge, not extra exam syllabus.

## 16.1 Classical convolution → CNN convolution

DIP teaches:

```text
fixed designed kernel
→ local response
```

CNNs extend the idea to:

```text
learned kernels
→ feature maps
```

Understanding classical convolution removes much of the mystery from convolutional networks.

---

## 16.2 Sampling → resizing / feature-map downsampling

DIP:

```text
sampling
undersampling
aliasing
anti-aliasing
```

Modern CV:

```text
image resize
stride
pooling
patchification
multi-scale features
```

Aliasing still matters.

---

## 16.3 Geometric transforms → data augmentation

DIP:

- translation;
- rotation;
- scaling;
- affine mapping;
- interpolation.

ML:

- random rotation;
- random crop;
- scale jitter;
- affine augmentation.

The same mathematics sits underneath both.

---

## 16.4 Histograms / color spaces → preprocessing

DIP helps you understand:

- illumination shifts;
- contrast normalization;
- channel distributions;
- RGB vs HSV/YCbCr;
- why preprocessing can help or damage a downstream model.

---

## 16.5 Fourier theory → model/image frequency behavior

Useful for reasoning about:

- blur;
- texture;
- sharpening;
- high-frequency noise;
- periodic artifacts;
- compression artifacts;
- adversarial high-frequency perturbations;
- what a model may be exploiting in an image.

---

## 16.6 Restoration → inverse problems

The degradation equation:

\[
g=h*f+\eta
\]

is conceptually upstream of modern:

- deblurring;
- super-resolution;
- denoising;
- computational imaging;
- learned inverse models.

---

## 16.7 Segmentation → deep segmentation

Classical:

```text
gradient
threshold
morphology
region
```

Modern:

```text
learned dense feature representation
→ pixel-wise prediction
```

The evaluation problem is modern; the core spatial reasoning is not new.

---

## 16.8 Compression → why training images contain artifacts

JPEG introduces:

- block processing;
- DCT;
- quantization;
- frequency-selective loss.

That matters when datasets contain:

- low-quality web images;
- repeated recompression;
- block artifacts;
- low-frequency/high-frequency bias.

---

# 17. Resource Stack

Do not use all resources simultaneously.

---

## Imaging Systems Companion Resources

These are optional companions for understanding acquisition physics. They are **not higher priority than your course lecture material**.

### Digital cameras / computational photography

Use:
- camera model and image formation material from standard computer-vision / computational-photography courses;
- OpenCV camera calibration and geometric transformation documentation for practical geometry.

Main things to learn:
- pinhole projection;
- lens/focal length/FOV;
- exposure;
- image sensor;
- Bayer sampling;
- demosaicing;
- camera artifacts.

### Medical imaging

For CT/MRI/PET/SPECT, use a medical-imaging introduction only to understand:
- acquisition principle;
- signal formation;
- reconstruction idea;
- what a pixel/voxel represents.

Do not let medical physics displace the core DIP syllabus.

### Ultrasound

Focus on:
- pulse-echo;
- time-of-flight;
- B-mode formation;
- speckle;
- attenuation/shadowing;
- resolution.

### Microscopy

Focus on:
- optical resolution;
- PSF;
- diffraction;
- deconvolution;
- segmentation.

### Thermal / spectral / depth

Focus on:
- what physical quantity each sensor measures;
- channel/depth representation;
- calibration/noise;
- how those arrays become ordinary DIP inputs.

---

## Tier 1 — Course-critical

### A. IIIT-Delhi course material

Highest priority.

Use instructor slides, examples, tutorials, and assignment wording as the best guide to quiz style.

TechTree:
https://techtree.iiitd.edu.in/

### B. Gonzalez & Woods

**Rafael C. Gonzalez, Richard E. Woods — Digital Image Processing, 3rd ed.**

Use as the main theory reference.

### Suggested chapter order

```text
Week 1       → Ch. 1
Week 2       → Ch. 2
Weeks 3–4    → Ch. 3
Weeks 5–8    → Ch. 4
Week 9       → Ch. 5
Week 10      → Ch. 6
Week 11      → Ch. 8
Week 12      → Ch. 9
Week 13      → Ch. 10
Features     → Ch. 11
```

Do not read unrelated chapters merely for completeness.

---

## Tier 2 — Video/theory companion

### NPTEL — Digital Image Processing

Use when lecture/book treatment is not clicking.

NPTEL course portal:
https://nptel.ac.in/courses/117105079

Use especially for:

- sampling;
- interpolation;
- Fourier;
- restoration;
- segmentation;
- morphology.

---

## Tier 3 — Coding references

### OpenCV tutorials

https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html

Useful for:

- geometric transforms;
- filtering;
- thresholding;
- morphology;
- gradient/Canny;
- histograms;
- Fourier examples.

### scikit-image

https://scikit-image.org/docs/stable/

Useful for:

- restoration;
- registration;
- transforms;
- morphology;
- segmentation;
- features.

### NumPy FFT

https://numpy.org/doc/stable/reference/routines.fft.html

### SciPy ndimage

https://docs.scipy.org/doc/scipy/tutorial/ndimage.html

---

## Tier 4 — Mathematics rescue

### MIT OpenCourseWare — Signals and Systems

https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/

Use only when you need a deeper explanation of:

- convolution;
- Fourier transform;
- Fourier properties;
- sampling;
- interpolation.

---

# 18. Research Paper Reading Ladder

Papers are a **depth track**, not a substitute for quiz preparation.

---

## Paper 1 — Otsu, 1979

**Nobuyuki Otsu — “A Threshold Selection Method from Gray-Level Histograms.”**

Read during:

```text
thresholding / segmentation
```

Main learning:

- formulate threshold selection as class-separation optimization;
- connect histograms to statistical decision criteria.

DOI:

https://doi.org/10.1109/TSMC.1979.4310076

Difficulty:

```text
2 / 5
```

---

## Paper 2 — Canny, 1986

**John Canny — “A Computational Approach to Edge Detection.”**

Read during:

```text
edge detection
```

Main learning:

- detection;
- localization;
- single response;
- why edge detection is an optimization/problem-design question.

DOI:

https://doi.org/10.1109/TPAMI.1986.4767851

Difficulty:

```text
3.5 / 5
```

---

## Paper 3 — Harris & Stephens, 1988

**“A Combined Corner and Edge Detector.”**

Read after:

```text
gradients + edge detection
```

Archive:

https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.html

Difficulty:

```text
3 / 5
```

---

## Paper 4 — Lowe, 2004

**David G. Lowe — “Distinctive Image Features from Scale-Invariant Keypoints.”**

Read only after feature material.

Author resource:

https://www.cs.ubc.ca/~lowe/keypoints/

Difficulty:

```text
4 / 5
```

---

## Technical standard — JPEG

**ITU-T T.81 / ISO/IEC 10918-1**

Reference:

https://www.itu.int/rec/T-REC-T.81/en

Do not read the whole standard for the course.

Study the conceptual architecture:

```text
color transform
→ blocks
→ DCT
→ quantization
→ coefficient ordering/coding
```

---

# 19. Resource Priority When Time Is Short

Use:

\[
\boxed{
\text{Lecture}
>
\text{Lecture Problems}
>
\text{Gonzalez}
>
\text{Quiz Practice}
>
\text{Assignment}
>
\text{NPTEL}
>
\text{Papers}
}
\]

Do **not** sacrifice a 10%-scale quiz to finish a research paper.

---

# 20. Repository Structure

Create:

```text
digital-image-processing/
│
├── README.md
│
├── 00_math_bootcamp/
│   ├── matrices.ipynb
│   ├── probability.ipynb
│   ├── complex_numbers.ipynb
│   └── convolution_1d.ipynb
│
├── 01_image_fundamentals/
│   ├── week1_image_anatomy.ipynb
│   ├── camera_pipeline_and_artifacts.ipynb
│   └── imaging_modalities_notes.md
├── 01b_imaging_systems/
│   ├── camera/
│   ├── xray_ct/
│   ├── mri/
│   ├── ultrasound/
│   ├── pet_spect/
│   ├── microscopy/
│   ├── thermal_spectral/
│   └── depth_lidar_oct_sar/
├── 02_sampling_quantization/
├── 03_interpolation_geometry/
├── 04_histograms/
├── 05_spatial_filtering/
├── 06_fourier/
├── 07_frequency_filtering/
├── 08_restoration/
├── 09_color/
├── 10_dct_compression/
├── 11_morphology/
├── 12_segmentation/
├── 13_features/
│
├── assignments/
├── midsem/
├── endsem/
│
├── quiz_questions/
│   ├── quiz_1/
│   ├── quiz_2/
│   ├── quiz_3/
│   └── quiz_4/
│
├── formula_sheet/
│   ├── Q1.md
│   ├── Q2.md
│   ├── Q3.md
│   └── Q4.md
│
├── papers/
└── results/
```

This becomes a reusable classical-computer-vision reference after the semester.

---

# 21. Weekly Operating Rhythm

For a normal week:

## Day 1 — Lecture Reconstruction

**60–90 min**

- go through professor material;
- rewrite concepts in your own words;
- identify unfamiliar mathematical terms.

## Day 2 — Mathematics

**60–90 min**

- derive;
- calculate;
- solve tiny synthetic problems.

## Day 3 — NumPy Implementation

**60–120 min**

No high-level helper for the core algorithm.

## Day 4 — Library + Experiment

- OpenCV/scikit-image;
- parameter sweeps;
- visual comparisons;
- explain differences.

## Day 5 — Quiz Mode

No:

```text
computer
LLM
notes
```

Do closed-book recall and hand calculations.

## Day 6 — Assignment / Project

Use LLM support as allowed.

## Day 7 — Consolidation

Create one page containing:

```text
Topic
→ core idea
→ equations
→ one worked example
→ algorithm
→ common mistakes
→ 3 quiz questions
```

---

# 22. Weekly Time Budget

Approximate outside-class baseline:

```text
2.0 h — theory
1.5 h — manual mathematical work
1.5 h — coding
1.0 h — quiz recall
1.0 h — assignment/project
```

Total:

```text
~7 h/week
```

Before a quiz:

```text
40% manual questions
25% theory recall
20% error correction
15% coding / assignment maintenance
```

---

# 23. Immediate 7-Day Starting Plan

Do not jump directly to Fourier.

## Day 1

```text
Image representation
Pixels
Matrices
Coordinate conventions
```

Deliver:

`image_anatomy.ipynb`

## Day 2

```text
Sampling
Quantization
Bit depth
Resolution
```

Deliver:

one hand worksheet + `sampling_quantization.ipynb`

## Day 3

```text
Nearest interpolation
Bilinear interpolation
Bicubic intuition
```

Solve bilinear manually.

## Day 4

```text
Translation
Scaling
Rotation
Affine transforms
Homogeneous coordinates
```

## Day 5

```text
1-D convolution
2-D convolution
correlation
kernel flipping
padding
```

## Day 6

```text
Intensity transforms
Histograms
Histogram equalization
```

## Day 7 — Quiz 0

Closed-book.

Coverage:

- image representation;
- sampling;
- quantization;
- interpolation;
- geometric transform;
- convolution;
- histograms.

Target:

\[
\ge85\%
\]

Then enter the Weeks 3–4 spatial-processing block.

---

# 24. Mastery Scorecard

Rate every major topic:

```text
0 — never seen
1 — recognize
2 — can explain
3 — can solve with notes
4 — can solve closed-book
5 — can teach + implement + debug
```

Maintain:

| Topic | Target before quiz |
|---|---:|
| Sampling | 5 |
| Quantization | 5 |
| Interpolation | 4 |
| Geometric transforms | 4 |
| Histograms | 5 |
| Convolution | 5 |
| Spatial filtering | 5 |
| Fourier/DFT | 5 |
| DFT properties | 5 |
| Frequency filtering | 5 |
| Restoration | 4 |
| Color | 4 |
| DCT/JPEG | 5 |
| Morphology | 5 |
| Segmentation | 5 |
| Harris/SIFT depth | 3 |

---

# 25. Topic Priority Tiers

## S Tier — Must become automatic

- convolution;
- sampling and aliasing;
- histograms/equalization;
- DFT/Fourier;
- DFT properties;
- frequency filtering;
- DCT/JPEG;
- morphology;
- edge detection.

## A Tier — Strong mastery

- interpolation;
- geometric transforms;
- intensity transforms;
- image registration;
- noise models;
- Wiener filtering;
- thresholding/Otsu;
- color spaces.

## B Tier — Understand, but do not overspend time

- detailed visual-system biology;
- advanced reconstruction beyond course coverage;
- advanced descriptors;
- deep SIFT derivation unless explicitly tested.

Re-rank after:

```text
professor slides
+
assignment style
+
Quiz 1
```

provide actual evidence.

---

# 26. Semester Mastery Checklist

## Camera & imaging systems

- [ ] I can explain the full digital-camera pipeline from photons to final image.
- [ ] I understand the pinhole camera model conceptually.
- [ ] I understand focal length and field of view.
- [ ] I understand aperture, shutter time, gain/ISO, focus, and depth of field.
- [ ] I can explain how CCD/CMOS sensors digitize light.
- [ ] I understand Bayer sampling and why demosaicing is interpolation.
- [ ] I can identify common camera artifacts: noise, clipping, blur, moiré, rolling shutter, compression.
- [ ] I can explain how X-ray projection imaging works.
- [ ] I can explain the difference between X-ray and CT.
- [ ] I understand that CT reconstructs slices/volumes from many projections.
- [ ] I can explain MRI at the level of magnetic field → RF → gradients → k-space → reconstruction.
- [ ] I understand why MRI connects strongly to the Fourier transform.
- [ ] I can explain ultrasound pulse/echo image formation and the \(d=ct/2\) relationship.
- [ ] I understand PET/SPECT as tracer-based tomographic imaging.
- [ ] I understand microscopy as an optical imaging system with PSF/diffraction limits.
- [ ] I understand thermal imaging as infrared measurement, not ordinary RGB photography.
- [ ] I understand multispectral vs hyperspectral imaging.
- [ ] I understand stereo, ToF, structured light, and LiDAR as different routes to depth.
- [ ] I have basic intuition for OCT and SAR.
- [ ] For any modality, I can answer: physical signal, sensor, pixel meaning, resolution limit, noise/artifacts, relevant DIP operations.

## Foundations

- [ ] I can explain digital image formation.
- [ ] I understand pixels as sampled values.
- [ ] I can distinguish spatial/intensity resolution.
- [ ] I understand sampling.
- [ ] I understand quantization.
- [ ] I can explain aliasing.
- [ ] I can solve bilinear interpolation.
- [ ] I can apply geometric transforms.

## Spatial processing

- [ ] I can compute an image histogram.
- [ ] I can manually perform histogram equalization.
- [ ] I can explain convolution vs correlation.
- [ ] I can hand-apply a kernel.
- [ ] I understand Gaussian filtering.
- [ ] I understand median filtering.
- [ ] I understand Laplacian sharpening.
- [ ] I understand registration.

## Fourier

- [ ] I understand complex exponentials.
- [ ] I understand spatial frequency.
- [ ] I understand the Fourier transform.
- [ ] I understand 2-D DFT.
- [ ] I understand FFT vs DFT.
- [ ] I understand magnitude and phase.
- [ ] I understand DC.
- [ ] I understand `fftshift`.
- [ ] I know major DFT properties.
- [ ] I can explain convolution theorem.
- [ ] I understand frequency filtering.

## Restoration

- [ ] I can write the degradation model.
- [ ] I recognize common noise models.
- [ ] I understand inverse filtering.
- [ ] I understand why inverse filtering fails.
- [ ] I understand Wiener filtering.
- [ ] I understand regularization/CLS conceptually.

## Color

- [ ] RGB.
- [ ] CMY/CMYK.
- [ ] HSV/HSI.
- [ ] YCbCr.
- [ ] I understand luminance vs chrominance.

## Compression

- [ ] I understand redundancy.
- [ ] I understand DCT.
- [ ] I understand DC/AC coefficients.
- [ ] I can explain the JPEG pipeline from memory.
- [ ] I understand quantization.
- [ ] I understand why JPEG is lossy.
- [ ] I understand quality/file-size tradeoffs.

## Morphology

- [ ] Structuring element.
- [ ] Erosion by hand.
- [ ] Dilation by hand.
- [ ] Opening.
- [ ] Closing.
- [ ] I understand element-size effects.

## Segmentation

- [ ] Point/line detection concepts.
- [ ] Gradient.
- [ ] Sobel.
- [ ] Laplacian.
- [ ] Canny.
- [ ] Global threshold.
- [ ] Adaptive threshold.
- [ ] Otsu.
- [ ] Morphological post-processing.

## Feature depth

- [ ] Connected components.
- [ ] Contours.
- [ ] Moments.
- [ ] Harris intuition.
- [ ] Basic SIFT intuition.

---

# 27. Definition of Course Completion

You have genuinely learned CSE340 when you can take an unfamiliar image-processing problem and reason:

```text
What information is present?
What degradation or transformation occurred?
Should I work spatially or in frequency?
What operation is mathematically justified?
What artifacts will the operation introduce?
How can I evaluate whether it helped?
```

The goal is not to memorize 40 OpenCV function names.

The goal is to understand:

\[
\boxed{
\text{what an image contains}
+
\text{how image information changes}
+
\text{how to manipulate it intentionally}
}
\]

That foundation transfers directly into computer vision, computational imaging, deep learning, multimodal perception, medical imaging, and visual-model debugging.

---

# 28. Our Study Protocol Going Forward

For every new lecture/topic, use this fixed sequence:

```text
1. Intuition
2. Mathematics
3. Hand-worked example
4. NumPy implementation
5. OpenCV/scikit-image validation
6. Visual experiment
7. Quiz questions
8. Assignment connection
9. One-page consolidation
```

Before every quiz:

```text
Course slides
→ concept audit
→ manual problem set
→ closed-book mock quiz
→ error diagnosis
→ targeted repair
→ second mock
```

For assignments:

```text
Understand first
→ code second
→ LLM debug third
→ validate
→ explain
→ convert assignment into quiz knowledge
```

---

# 29. Final Learning Spine

Keep this one sequence in your head throughout the semester:

\[
\boxed{
\text{Image Formation}
\rightarrow
\text{Camera \& Imaging Systems}
\rightarrow
\text{Sampling}
\rightarrow
\text{Spatial Processing}
\rightarrow
\text{Fourier}
\rightarrow
\text{Frequency Filtering}
\rightarrow
\text{Restoration}
\rightarrow
\text{Color}
\rightarrow
\text{DCT/JPEG}
\rightarrow
\text{Morphology}
\rightarrow
\text{Segmentation}
\rightarrow
\text{Features}
}
\]

And beneath every stage:

\[
\boxed{
\text{Theory}
\leftrightarrow
\text{Math}
\leftrightarrow
\text{Hand Problems}
\leftrightarrow
\text{Code}
}
\]

That is the operating system for the course.

---

# 30. Source Basis

This roadmap was consolidated from:

1. IIIT-Delhi TechTree CSE340/CSE540/ECE350 Digital Image Processing course structure.
2. Your 2026 assessment structure: 40% quizzes, 40% assignments, 10% midsem, 10% endsem.
3. Gonzalez & Woods as the course textbook.
4. NPTEL Digital Image Processing as the main secondary teaching resource.
5. OpenCV, NumPy/SciPy, and scikit-image documentation for implementation.
6. MIT Signals & Systems as a mathematical fallback for Fourier, convolution, sampling, and interpolation.
7. Selected classical literature: Otsu, Canny, Harris & Stephens, Lowe/SIFT, and the JPEG standard.

---

## Final rule

> **Do not let code familiarity fool you into thinking a DIP topic is learned.  
> If you cannot explain the equation and solve a small version by hand, it is not yet quiz-ready.**


---

# v3 Change Log — 23 Aug 2026

- Switched the active textbook mapping to the uploaded **Gonzalez & Woods 4E Global Edition, Chapters 1–4**.
- Added professor-actual Lecture 1–3 map.
- Added source-priority tags: slide core / book core / slide-only core / beyond-textbook / assignment park.
- Recorded Lecture-0 progress through **L0.10 Convolution Theorem**.
- Set **L0.11 DFT Deep Dive** as the next study module.
- Added selected Gonzalez problem IDs instead of attempting all end-of-chapter exercises.
- Added Bilateral Filtering and Non-Local Means as 2026 slide-core material.
- Added Bilateral/NLM Assignment-1 research parking lane.
- Added current slide-aligned quiz bank.
- Made Quiz-1 boundary explicitly provisional until the professor announces it.
- Preserved the complete v2 imaging-systems track and generic semester roadmap beneath this actual-course overlay.
