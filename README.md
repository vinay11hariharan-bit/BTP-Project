# BTP — Low-Power sEMG Gesture Classifier ASIC

B.Tech project, Department of Electrical Engineering, IIT Bombay (BioNICS group, advisor Prof. Laxmeesha Somappa).
Team: Vinay Hariharan (lead), Sudhindra, Yash, Digvijay.

Surface EMG from 8–24 forearm electrodes is classified into hand/wrist gestures by a fixed-function,
fully integer digital core: per-channel time-domain features, an integer shift-normaliser, and an
INT8 Linear Discriminant Analysis (LDA) classifier with one multiplier. Everything is verified
bit-exactly against a Python golden model and evaluated subject-independently on the public
FORS-EMG dataset.

## Headline results

All accuracies are leave-one-subject-out (LOSO) on FORS-EMG, 4 gestures
(Hand Close, Hand Open, Wrist Flexion, Wrist Extension), mean ± 95% CI over subjects.

| Result | Value |
|---|---|
| INT8 LDA, integer front end (software reference) | 77.79 ± 4.39 % |
| Bit-exact chip model (golden model, 8-bit ADC, chip windows) | 77.82 ± 4.10 % |
| Model search: MLP / Logistic Regression / LDA on float features | 77.9–79.6 %, statistically tied |
| INT8 quantisation cost (28 comparisons) | +0.18 pp on average |
| RTL vs golden model, whole chip, C = 8 … 24 channels | bit-exact (x_q, scores, class) |
| Worst-case Stage 2 + 3 time (C = 24) | 850 of 1015 cycles per sample at 1 MHz |

Details: [`information.txt`](information.txt) (model search), [`RTL/README.md`](RTL/README.md) (hardware),
and the explorations section below.

## Repository map

```
.
├── README.md                       this file
├── information.txt                 technical summary of the model search (preprocessing, INT8 scheme, results)
├── K_4_LDA_Analysis.py             MAIN pipeline: loading, filtering, windowing, integer features, INT8 LDA, LOSO
├── lda_golden_model.py             bit-exact Python model of the chip (Stages 1-3); training + test vectors
├── K_4_LDA_Golden_LOSO.py          chip model vs software LDA on FORS-EMG (paired LOSO)
├── K_4_LDA_HW_Experiments.py       choice of the multiplier-free feature set (H1-H6) and the shift normaliser
├── K_4_LDA_Calib.py                LDA with per-user calibration (blending class means / covariance)
├── fors_emg_feature_selection_lda.py  nested-LOSO feature-type selection for the LDA
├── classifier_classification.py    8-classifier comparison (linear, trees, MLP) under one protocol
├── K_4_model_MLP.py                INT8 MLP benchmark, 4 gestures
├── K_6_model_LR_comparison.py      MLP vs Logistic Regression, best 6-gesture subset
├── K_4_CNN_LOSO.py                 raw-signal CNNs (depthwise-separable, two-branch) vs INT8 LDA
├── K_4_CNN_Calib.py                CNN with per-user fine-tuning
├── K_4_HDC_Way1_vs_Way2.py         exploration: hyperdimensional computing on features vs on samples
├── K_4_HDC_Calib.py                exploration: HDC with on-chip-style per-user calibration
├── K_4_TM_Calib.py                 exploration: Tsetlin machine with per-user calibration
├── K_2_CTBAI_Stopping.py           exploration: correlated best-arm-identification stopping rules
├── RTL/                            Verilog of the chip + chip-level cocotb testbench (see RTL/README.md)
└── archive/                        FORS-EMG dataset (Subject1 … Subject19)
```

## Setup

```bash
pip install numpy scipy scikit-learn tabulate tqdm        # every script
pip install torch                                          # MLP and CNN scripts only
brew install icarus-verilog && pip install cocotb          # RTL simulation (macOS; apt install iverilog on Linux)
```

Set the dataset path once: `BASE_DIR` at the top of `K_4_LDA_Analysis.py` must point to the folder
that contains `Subject1/ … Subject19/` (in this repo: `archive/FORS-EMG Dataset/FORS-EMG Dataset/FORS-EMG`).
Every other script imports its data loading from `K_4_LDA_Analysis.py`, so all scripts see the same
filter, windows and folds. All scripts print results only and write no files.

## Evaluation protocol (shared by every script)

- 20–450 Hz band-pass + 50 Hz notch, causal (as the hardware would run it).
- 200 ms windows (196–197 samples at 985 Hz), 100 ms stride.
- Leave-one-subject-out: the held-out subject is never seen in training, normalisation or calibration.
  Inside the training subjects, trial 5 is the validation set; trials 1–4 train.
- Windows are cut after the split, so overlapping windows never leak across it.
- Comparisons are paired per subject (Wilcoxon signed-rank test, 95% CI of the paired difference).

## Development history (short)

1. **Gesture set.** The original set {Hand Close, Hand Open, Index, Index Little} capped at about 65 %:
   Index ↔ Index Little alone caused about 40 % of errors. Switching to Hand Close, Hand Open, Wrist
   Flexion, Wrist Extension gained +14.1 pp (p = 3.8 × 10⁻⁹). Larger vocabularies: 6 gestures 69.0 %, 12 gestures 48.8 %.
2. **Model.** Capacity is not the bottleneck: five MLP widths, Logistic Regression and LDA are
   statistically tied; trees that fit the 10 KB budget are worse. LDA was chosen for the ASIC
   (closed-form training, 160 MACs, 176 B).
3. **Integer front end.** Features rebuilt from add, subtract, absolute value, compare, shift and a
   leading-one detector (SABS, WL, ZC, SSC, Mitchell-squared POW), with an integer `(x − μ) >> k`
   normaliser: no accuracy loss against the float reference.
4. **Hardware.** Scope fixed in Oct 2026 to an LDA-only fixed-function core (CNN/ISA work by Yash and
   Digvijay explored the programmable alternative). Extensible mother–daughter design: one compute
   core, 1–3 daisy-chained 8-channel ADC pods (8–24 electrodes) set by one register.
5. **Verification.** A bit-exact golden model (`lda_golden_model.py`) drives exhaustive unit tests and a
   chip-level cocotb testbench; the chip reproduces the golden model's decisions bit for bit.

## Branches

- `main` — the LDA chip described here.
- `processor_flexibility` — the chip as a configurable feature front end for a microcontroller:
  on-chip LDA removed, real TI ADS1299 ADC interface, SPI slave link to an MCU, on-chip IIR filter,
  configurable windows, fault flags. See `LDAcore_with_proc/README.md` on that branch.

## Explorations (Nov 2026)

Alternative classifiers and decision rules, each evaluated with the same LOSO protocol on the chip's
own INT8 features. Calibration = one trial of the new user (every gesture once); the test uses that
user's other trials only.

| Script | Idea | Result |
|---|---|---|
| `K_4_HDC_Way1_vs_Way2.py` | Hyperdimensional computing, no multipliers. Way 1 encodes the 40 features, Way 2 encodes per-sample envelopes with 3-step sequences | Way 1 60.2 %, Way 2 67.0 %, LDA 77.4 % (18 subjects) |
| `K_4_HDC_Calib.py` | HDC prototypes updated on chip with the calibration trial (additions only) | Way 2: 66.7 → 77.9 %, matches the LDA (78.9 % with calibration) |
| `K_4_TM_Calib.py` | Tsetlin machine (logic-only clauses, learning automata) with calibration | 72.1 → 80.7 ± 2.7 %; worst user 72 % vs 61 % for LDA; memory too large for this chip |
| `K_2_CTBAI_Stopping.py` | Binary decisions as two-armed correlated Gaussian best-arm identification (cost-weighted, joint pulls); decide when confident | error per decision 6.6 % → 0.2–1.1 % on 5 of 6 gesture pairs in about 1.8 s with the mixture-martingale rule |

A consolidated project report is in preparation.
