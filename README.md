# BTP — Low-Power sEMG Gesture Classifier ASIC

B.Tech project, Department of Electrical Engineering, IIT Bombay (BioNICS group, advisor Prof. Laxmeesha Somappa).
Team: Vinay Hariharan, Sudhindra, Yash, Digvijay.

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

Details: [`docs/technical_summary.md`](docs/technical_summary.md) (model search) and [`RTL/README.md`](RTL/README.md) (hardware).

## Repository map

```
.
├── README.md
├── analysis/                        software pipeline and model search (all on FORS-EMG, LOSO)
│   ├── K_4_LDA_Analysis.py          MAIN pipeline: loading, filtering, windowing, integer features, INT8 LDA
│   ├── K_4_LDA_HW_Experiments.py    choice of the multiplier-free feature set (H1-H6) and the shift normaliser
│   ├── fors_emg_feature_selection_lda.py  nested-LOSO feature-type selection
│   ├── classifier_classification.py 8-classifier comparison (linear, trees, MLP)
│   ├── K_4_model_MLP.py             INT8 MLP benchmark, 4 gestures
│   ├── K_6_model_LR_comparison.py   MLP vs Logistic Regression, best 6-gesture subset
│   ├── K_4_LDA_Merge_vs_Drop.py     all 12 gestures merged into 4 classes vs dropping 8 (chip LDA)
│   ├── K_4_LDA_Calib.py             LDA with per-user calibration
│   ├── K_4_CNN_LOSO.py              raw-signal CNNs vs INT8 LDA
│   └── K_4_CNN_Calib.py             CNN with per-user fine-tuning
├── golden_model/
│   ├── lda_golden_model.py          bit-exact Python model of the chip (Stages 1-3); training + test vectors
│   └── K_4_LDA_Golden_LOSO.py       chip model vs software LDA on FORS-EMG (paired LOSO)
├── RTL/                             Verilog of the chip + chip-level cocotb testbench (RTL/README.md)
├── docs/
│   ├── technical_summary.md         model search: preprocessing, INT8 scheme, results
│   └── report/                      project report (LaTeX source, figures, PDF)
└── archive/                         FORS-EMG dataset (Subject<i>/<orientation>/<gesture>-<trial>.mat)
```

## Setup

```bash
pip install numpy scipy scikit-learn tabulate tqdm        # every script
pip install torch                                          # MLP and CNN scripts only
brew install icarus-verilog && pip install cocotb          # RTL simulation (macOS; apt install iverilog on Linux)
```

Run every script from the repository root, for example `python analysis/K_4_LDA_Analysis.py`. The dataset is found automatically in `archive/`; to use another
copy, set `BASE_DIR` at the top of `analysis/K_4_LDA_Analysis.py`. The scripts in `golden_model/` import their data loading from `analysis/K_4_LDA_Analysis.py`, so every script sees the
same filter, windows and folds. Scripts print their results; the older analysis scripts also save a results CSV in the working directory.

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
   Merging all 12 gestures into 4 classes of 3 instead of dropping 8 (grouping chosen from the training
   subjects) gives 60.9 % against 77.8 %, worse on all 19 subjects, so gestures are dropped.
2. **Model.** Capacity is not the bottleneck: five MLP widths, Logistic Regression and LDA are
   statistically tied; trees that fit the 10 KB budget are worse. LDA was chosen for the ASIC
   (closed-form training, 160 MACs, 176 B).
3. **Integer front end.** Features rebuilt from add, subtract, absolute value, compare, shift and a
   leading-one detector (SABS, WL, ZC, SSC, Mitchell-squared POW), with an integer `(x − μ) >> k`
   normaliser: no accuracy loss against the float reference.
4. **Hardware.** Scope fixed in Oct 2026 to an LDA-only fixed-function core (CNN/ISA work by Yash and
   Digvijay explored the programmable alternative). Extensible mother–daughter design: one compute
   core, 1–3 daisy-chained 8-channel ADC pods (8–24 electrodes) set by one register.
5. **Verification.** A bit-exact golden model (`golden_model/lda_golden_model.py`) drives exhaustive unit tests and a
   chip-level cocotb testbench; the chip reproduces the golden model's decisions bit for bit.

## Multi-board system (`main`)

The chip on `main` is built to be chained. Each board is one `emg_top` with one 8-channel ADC
pod; 1–3 boards in a chain act as one LDA classifier for 8–24 electrodes.

```
 ADC pod (ch 0-2)      ADC pod (ch 3-5)      ADC pod (ch 6-7)
       │                     │                     │
 ┌─────▼──────┐        ┌─────▼──────┐        ┌─────▼──────┐
 │ Board 0    │◄─sums──│ Board 1    │◄─sums──│ Board 2    │
 │ (head)     │──run──►│            │──run──►│ (tail)     │
 └─────┬──────┘        └────────────┘        └────────────┘
       ▼ class_out = argmax(0, D1, D2, D3)
```

- **How it splits the work.** Every board runs Stages 1–2 (features, normaliser) on its own channels
  and Stage 3 on its own slice of the weights. Stage 3 uses the K−1 difference rows W′_c = W_c − W_0,
  so each board only produces three partial sums p_1..p_3.
- **What crosses between boards.** Only those three 32-bit sums, bit-serially, LSB first, tail → head.
  Each board adds its own sums with one full adder as the bits pass. The head adds the bias b′ and
  picks argmax(0, D_1, D_2, D_3), which equals the single-chip LDA decision.
- **Synchronisation.** One 256 kHz clock for all boards. The head's `core_en` is passed down the chain
  (`run`), so every board's sample timer starts in the same cycle. A wired-AND ready line starts the
  sum transfer once every board has finished Stage 3.
- **Verified.** `python3 RTL/tb/verify_all.py` (from `RTL/`: `python3 tb/verify_all.py`, about 4 min)
  runs 18 checks and prints ALL PASS or what failed:
  - `emg_chain_link` alone, 1–4 boards, 3000 random strides each (large, small, negative and tied
    sums; K−1 = 3, 2, 1), checked against sums and argmax computed inside the testbench;
  - 8 deliberate bugs put into `emg_chain_link` one at a time: the unit test must catch every one;
  - the whole chip on subject 6, trial 1 of FORS-EMG (4 gestures, 7840 samples) for the splits
    8, 4+4, 2+6, 6+2, 5+3 and 3+3+2: all 79 decisions bit-exact with the golden model, which
    itself equals the original single-chip LDA on all 79. 72 of 76 single-gesture windows are correct.

Details, ports and commands: [`RTL/README.md`](RTL/README.md).

## Branches

- `main` — the chained multi-board version described above.
- `single_chip` — the earlier single-board LDA chip (two clocks, 1 MHz core, up to 24 channels on one
  core, cocotb chip test). This was `main` until 12 Nov 2026.
- `processor_flexibility` — the chip as a configurable feature front end for a microcontroller:
  on-chip LDA removed, real TI ADS1299 ADC interface, SPI slave link to an MCU, on-chip IIR filter,
  configurable windows, fault flags. See `LDAcore_with_proc/README.md` on that branch.

The full project report is in [`docs/report/`](docs/report/).
