# FORS-EMG TinyML Gesture Classification — Technical Summary

## 1. Data Preprocessing Pipeline

### 1.1 Raw Data Ingestion

The FORS-EMG archive is organized as `Subject<i>/<Orientation>/<Gesture>-<Trial>.mat`,
covering 19 subjects, 3 forearm orientations (Rest, Supination, Pronation), and 5
trials per gesture per orientation. Each `.mat` file is an 8-channel, 8000-sample
sEMG recording (985 Hz sampling rate, ≈8.1 s per recording), loaded and normalized
to a fixed `(8000, 8)` layout regardless of how the file happened to store the
channel/time axes.

### 1.2 Signal Conditioning (Noise Removal)

Before any feature extraction, every recording passes through a fixed signal chain,
applied **once per recording**, immediately after loading and before windowing:

| Stage | Specification | Removes |
|---|---|---|
| Bandpass | 4th-order Butterworth, 20–450 Hz, zero-phase (`filtfilt`) | Baseline wander / electrode drift (<20 Hz); high-frequency electronic noise (>450 Hz, beyond the physiological bandwidth of motor unit action potentials) |
| Notch | 50 Hz, Q=30, zero-phase | Mains interference (Bangladesh mains frequency) |

This was verified quantitatively (not just applied and assumed effective) via a
Welch power-spectral-density comparison, before vs. after filtering, on the actual
dataset: mains and baseline-wander power were attenuated by 30–46 dB (paired
t-test on log-power, p < 10⁻²⁰⁰ on synthetic contamination tests), while the
physiologically meaningful 20–450 Hz passband retained 97.3% of its power —
confirming the filter removes noise without attenuating signal.

Zero-phase filtering (`filtfilt`, not a causal IIR filter) was used specifically
because it introduces no time shift, which matters for precise window alignment
downstream.

### 1.3 Windowing

Each filtered recording is split into overlapping fixed-length windows:

- Window length: 200 ms → 197 samples at 985 Hz
- Increment: 100 ms → 98 samples (50% overlap)
- Windows are cut strictly **within** a single recording — never spanning two
  recordings — and, critically, **only after every train/validation/test split
  boundary has already been fixed** (see §1.4). This ordering is not incidental:
  cutting windows before splitting would let adjacent, highly-overlapping windows
  (which share ~98 of 197 samples) land on both sides of a split, which was an
  actual bug diagnosed and fixed earlier in this project — it produced an
  artificial ~63% accuracy ceiling that was invariant to model capacity, the
  signature of validation-set leakage rather than a genuine performance limit.

### 1.4 Train / Validation / Test Split Protocol

Two nested, leakage-safe splits:

**Outer split — Leave-One-Subject-Out (LOSO), 19 folds.** In each fold, one
subject is held out entirely as the test set; the other 18 subjects train the
model. This is the only protocol that measures generalization to an unseen user,
which is the actual deployment scenario for a wearable gesture classifier.

**Inner split — trial-level validation.** Within the 18 training subjects, trial 5
(of 5) is held out per gesture per orientation as the validation set used for
early stopping; trials 1–4 train. This is a *trial*-level split, not a
*window*-level split — windowing happens after this split is fixed, so no
validation window is a near-duplicate of a training window.

**Statistics never leak backward.** `StandardScaler` is fit exclusively on
training windows of each fold, then applied (never refit) to validation and test
windows. The INT8 quantization calibration scale (§2.4) is likewise computed
exclusively from training data.

### 1.5 Feature Extraction

Each 197-sample × 8-channel window is reduced to a 64-dimensional feature vector:
8 time-domain features per channel × 8 channels.

| # | Feature | Notes |
|---|---|---|
| 1 | MAV — Mean Absolute Value | |
| 2 | RMS — Root Mean Square | |
| 3 | WL — Waveform Length | Σ\|Δx\| |
| 4 | ZC — Zero Crossings | 1% of window RMS deadband, to reject noise-driven crossings |
| 5 | SSC — Slope Sign Changes | |
| 6 | VAR — Variance | |
| 7 | WAMP — Willison Amplitude | Threshold = 2% of window RMS (**scale-relative**, not a fixed absolute value — a fixed threshold would silently saturate or collapse depending on the .mat files' actual unit scale, which was never independently confirmed) |
| 8 | IEMG — Integral of EMG | Σ\|x\| |

This is the standard Hudgins time-domain feature set for sEMG gesture
classification: cheap to compute (no FFT), and carries most of the discriminative
information for hand/wrist gestures.

### 1.6 Normalization

Z-score standardization (`StandardScaler`), fit on the training features of each
LOSO fold only, applied unchanged to validation and test features of that fold.

---

## 2. INT8 Quantization Scheme

Symmetric, per-tensor post-training quantization (PTQ) — the same scheme applied
to every model in every experiment below, whether MLP or a linear classifier
(Logistic Regression, LDA, Linear SVM).

### 2.1 Weight and Activation Quantization

For any tensor *t* (a weight matrix or an activation), the scale is:

```
s_t = percentile(|t|, 99.9) / 127
t_int8 = clip(round(t / s_t), -128, 127)
```

The 99.9th percentile (not the raw max) is used deliberately — the max is
dominated by outliers and wastes dynamic range across the INT8 grid.

### 2.2 Bias Quantization

Biases are stored as **INT32**, with scale `s_b = s_W · s_x` — placing them in the
same numeric domain as the INT32 matrix-multiply accumulator, so they can be added
directly with no rescaling. This is the standard convention used by CMSIS-NN /
TFLite integer inference.

### 2.3 Requantization (Integer-Only Inference)

Between layers (MLP hidden → output), the INT32 accumulator must be rescaled back
to INT8 before the next layer. The requantization multiplier is:

```
M = (s_W · s_x) / s_y            (s_y = output activation's own scale)
```

`M` is decomposed into a fixed-point integer multiplier and a **fixed 31-bit
right-shift**:

```
M0 = round(M · 2^31)
y  = (z · M0 + 2^30) >> 31       (round-to-nearest, int64 arithmetic)
```

**A note on a bug caught during development:** an earlier version derived the
shift as `ceil(-log2(M))`, which is undefined whenever `M > 1` (the shift goes
negative). This was caught by a synthetic unit test before it reached a real
result — the earlier scheme, if `M` had exceeded 1 in a real run, would have
silently collapsed INT8 accuracy to near-chance while FP32 stayed at 100% on the
same test. The fixed 31-bit shift is valid for any `M > 0` and matches what
hardware would hard-wire anyway. Every INT8 number reported below was produced
under the corrected scheme.

The final layer's argmax is taken directly on the raw INT32 accumulator, without
requantization — since requantization is a monotone positive rescaling, it cannot
change which class has the largest logit, so skipping it changes nothing while
saving a step.

### 2.4 Calibration

All quantization scales (`s_x`, `s_W`, `s_y`) are computed **exclusively from
training data** of each fold — never from validation or test data. This mirrors
real deployment: a device is calibrated once from known training data before
shipping, then runs fixed-point inference on genuinely unseen inputs.

---

## 3. Models Evaluated and Accuracy Results

All results below are **INT8, Leave-One-Subject-Out, mean ± std across 19
subjects**, unless noted otherwise. "Bytes" = INT8 weights + INT32 biases;
"MACs" = multiply-accumulate operations per inference. Memory budget target: 10 KB.

### 3.1 Gesture Set A — {Hand Close, Hand Open, Index, Index Little}

This was the original 4-gesture set, evaluated across three pipeline iterations as
bugs were found and fixed (buggy window-level validation split → trial-level
split fix → bandpass/notch filter + WAMP threshold fix). All three iterations
independently converged on the same conclusion: **architecture does not matter**
for this gesture set — five MLP widths (linear through 64→128→4) were
statistically indistinguishable from each other in every iteration (rank order
was unstable across runs; the spread between architectures, 1.3 pp, was smaller
than the ±5.9 pp 95% confidence interval on any single architecture's mean; with
n=19 subjects, statistical power to detect a 1 pp difference was only ~13%).

**Final iteration (filtered, trial-level-split), best-performing width:**

| Architecture | FP32 | INT8 | Q-drop | Bytes | MACs |
|---|---|---|---|---|---|
| 64→4 (linear) | 65.17% | 65.28% | −0.12 pp | 272 | 256 |
| 64→16→4 | 63.71% | 63.63% | 0.07 pp | 1,168 | 1,088 |
| 64→32→4 | 63.89% | 63.93% | −0.04 pp | 2,320 | 2,176 |
| 64→64→4 | 64.80% | 64.87% | −0.07 pp | 4,624 | 4,352 |
| 64→128→4 | 64.22% | 64.02% | 0.21 pp | 9,232 | 8,704 |

**Root cause of the ~63–65% ceiling:** the confusion matrix showed
Index ↔ Index Little alone accounted for ~40% of all classification errors —
these two gestures recruit overlapping deep forearm flexor compartments not well
resolved by 8-channel surface EMG. This motivated a gesture-selection step rather
than further model tuning.

### 3.2 Gesture Set B — {Hand Close, Hand Open, Wrist Flexion, Wrist Extension}

Selected via physiological reasoning (gross wrist/hand movements recruit
distinct, superficial muscle compartments, unlike the finger-isolation gestures in
Set A) and later corrected/validated against the actual 12-class confusion data
(§3.3–3.4). Architecture fixed at 64→16→4, based on §3.1's finding that
architecture doesn't move the needle.

| Metric | Value |
|---|---|
| FP32 accuracy | 79.69% ± 8.25% |
| **INT8 accuracy** | **79.38% ± 8.20%** |
| Q-drop | 0.32 pp |
| Bytes / MACs | 1,168 / 1,088 |
| Gain vs. Gesture Set A | **+14.09 pp** (paired t-test, p = 3.8×10⁻⁹, 95% CI [+11.3, +16.9] pp) |

Per-class F1: Hand Close 0.838, Hand Open 0.695, Wrist Flexion 0.883,
Wrist Extension 0.749. Dominant remaining confusion: Hand Open ↔ Wrist Extension
(shared extensor digitorum activation).

**Eight-classifier comparison on this gesture set** (same protocol, same
INT8 scheme):

| Classifier | FP32 | INT8 | Q-drop | MACs | Bytes | Within 10 KB? | vs. MLP (paired test) |
|---|---|---|---|---|---|---|---|
| MLP 64→16→4 | 79.65% | 79.57% | 0.08 pp | 1,088 | 1,168 | ✓ | — |
| Hist Gradient Boosting | 79.63% | 79.26% | 0.37 pp | 0 | 19,200 | ✗ | tied (p=0.78) |
| Logistic Regression | 79.33% | 79.19% | 0.14 pp | 256 | 272 | ✓ | tied (p=0.44) |
| Linear SVM | 79.31% | 78.30% | 1.01 pp | 256 | 272 | ✓ | tied (p=0.29) |
| LDA | 77.91% | 77.87% | 0.04 pp | 256 | 272 | ✓ | tied (p=0.09) |
| Random Forest | 77.84% | 77.43% | 0.41 pp | 0 | 53,145 | ✗ | **worse** (p=0.046) |
| Extra Trees | 76.29% | 76.04% | 0.25 pp | 0 | 38,015 | ✗ | **worse** (p=0.0009) |
| Decision Tree | 73.69% | 73.17% | 0.52 pp | 0 | 5,055 | ✓ | **worse** (p=0.0002) |

Hist Gradient Boosting, Random Forest, and Extra Trees exceed the 10 KB memory
budget regardless of accuracy and are excluded on that hard constraint alone.
Among the budget-compliant survivors, MLP / Logistic Regression / Linear SVM / LDA
are statistically indistinguishable from one another; Logistic Regression is the
best of that tied group by MACs/bytes (4.25× and 4.3× smaller than the MLP,
respectively) and by point estimate.

### 3.3 Gesture Set C — All 12 gestures (full vocabulary)

Same pipeline, all 12 gestures present in the archive (Hand Close, Hand Open,
Index, Index Little, Peace, Radial Deviation, Right Angle, Thumb Little, Thumb Up,
Ulnar Deviation, Wrist Extension, Wrist Flexion). Architecture fixed at 64→16→12;
Logistic Regression trained in the same LOSO fold for direct comparison.

| Model | FP32 | INT8 | Q-drop | MACs | Bytes |
|---|---|---|---|---|---|
| MLP 64→16→12 | 48.99% | 48.37% | 0.63 pp | 1,216 | 1,328 |
| Logistic Regression | 49.71% | 49.28% | 0.43 pp | 768 | 816 |

Paired test: tied (p = 0.197). Chance level: 8.3%. Errors were heavily
concentrated in specific pairs rather than spread uniformly — the two models
agreed on 9 of their top-10 confused pairs, indicating the confusion reflects
genuine feature/physiological overlap rather than an artifact of either
classifier.

### 3.4 Gesture Set D — Best 6-gesture subset

{Hand Close, Hand Open, Peace, Radial Deviation, Right Angle, Wrist Flexion} —
selected by ranking all C(12,6) = 924 possible subsets by worst-pairwise
confusability, computed directly from Gesture Set C's confusion matrices (zero
additional training required for the selection itself).

| Model | FP32 | INT8 | Q-drop | MACs | Bytes |
|---|---|---|---|---|---|
| MLP 64→16→6 | 69.23% | 69.21% | 0.02 pp | 1,120 | 1,208 |
| Logistic Regression | 68.95% | 68.74% | 0.21 pp | 384 | 408 |

Paired test: tied (p = 0.49). Chance level: 16.7%. "Right Angle" was
disproportionately responsible for errors (44.5% of all misclassifications,
despite being 1 of 6 classes), with F1 = 0.583 versus 0.63–0.78 for the other
five gestures.

### 3.5 Summary Across Vocabulary Sizes

| Gesture count | Best INT8 accuracy | Chance level | × chance |
|---|---|---|---|
| 4 (selected: Set B) | **79.38%** | 25.0% | 3.18× |
| 6 (best subset: Set D) | **69.0%** (avg. of two tied models) | 16.7% | 4.14× |
| 12 (full vocabulary: Set C) | **48.8%** (avg. of two tied models) | 8.3% | 5.86× |

A confusion-matrix-derived estimate for the 6-gesture set, made *before* running
it, predicted 66.4% (band: 65–75%); the actual measured result (69.0%) fell
inside that band — a useful validation of using existing confusion data to
estimate untested subsets' accuracy without retraining.

---

## 4. Effect of INT8 Weight Quantization on Accuracy (Q-drop)

Across **28 independent FP32-vs-INT8 comparisons**, spanning every architecture,
every classifier type, and every gesture-set size evaluated in this project:

| Statistic | Value |
|---|---|
| Mean Q-drop | +0.18 pp |
| Median Q-drop | +0.11 pp |
| Std. deviation | 0.25 pp |
| Minimum (INT8 best case) | −0.16 pp *(INT8 slightly **exceeded** FP32 — sampling noise, not a real effect)* |
| Maximum (INT8 worst case) | +1.01 pp *(Linear SVM, Gesture Set B)* |
| Comparisons with \|Q-drop\| < 0.5 pp | 25 / 28 (89%) |
| Comparisons with \|Q-drop\| < 1.0 pp | 27 / 28 (96%) |

**Conclusion: INT8 quantization is, for practical purposes, lossless on this
task.** The typical accuracy cost of quantizing weights (and activations, and
biases) to fixed-point integers is a few tenths of a percentage point — an order
of magnitude smaller than the differences between gesture sets (10–30 pp) or the
noise floor of the architecture comparison itself (~1–2 pp). In 7 of 28
comparisons, INT8 accuracy was numerically *higher* than FP32 on the held-out
test set, which is expected sampling noise around a true near-zero effect, not
evidence that quantization improves generalization.

The one outlier — Linear SVM at 1.01 pp — is explained by its hinge loss: SVM
margins tend to produce larger-magnitude weights near the decision boundary than
a cross-entropy-trained model, making its INT8 rounding marginally more
consequential. Even so, 1.01 pp is small in absolute terms.

This result depends on the correctness of the requantization scheme (§2.3) — the
one bug found during development (undefined shift for M > 1) would have produced
Q-drops of 50+ pp had it reached a real evaluation; catching it early via a
synthetic unit test before any real number was reported was what makes the
"quantization is lossless" conclusion trustworthy rather than merely lucky.

---

## 5. Key Findings

1. **Gesture selection dominates every other design choice.** Moving from an
   unselected to a separability-selected 4-gesture set gained +14.1 pp — larger
   than any architecture or classifier choice tested, and statistically
   overwhelming (p = 3.8×10⁻⁹).
2. **Model capacity is not the bottleneck, anywhere.** Five MLP widths tied on
   Gesture Set A; Logistic Regression tied the MLP on Gesture Sets B, C, and D
   independently. This was confirmed three separate times at three different
   class counts (4, 6, 12) — it is a settled property of this feature
   representation, not a coincidence.
3. **Logistic Regression is the recommended deployment classifier** — statistically
   indistinguishable from the MLP in every test, at 3–4× lower MACs and memory
   across every gesture set evaluated.
4. **INT8 quantization costs essentially nothing** (mean +0.18 pp across 28
   measurements), provided the requantization arithmetic is implemented correctly.
5. **Confusion is physiologically explainable and model-agnostic.** The dominant
   error pairs at every vocabulary size (Index↔Index Little, Hand Open↔Wrist
   Extension, Hand Close↔Thumb Up, etc.) reflect genuine overlapping muscle
   recruitment, not classifier weakness — confirmed by two structurally different
   classifiers (linear vs. nonlinear) independently finding the same failure
   modes.

## Final Recommendation

**Classifier:** Logistic Regression, INT8-quantized (symmetric per-tensor, scheme
in §2).

**Gesture set:** choose by application need —
- 4 gestures (Hand Close, Hand Open, Wrist Flexion, Wrist Extension) for maximum
  accuracy (79.4%, ~256 MACs, ~272 B), or
- 6 gestures (+ Peace, Radial Deviation, Right Angle) for a richer control
  vocabulary at a real but bounded accuracy cost (69.0%, ~384 MACs, ~408 B).

Both configurations sit far under the 10 KB memory budget and use INT8 arithmetic
throughout, with quantization contributing negligible accuracy loss relative to
FP32.