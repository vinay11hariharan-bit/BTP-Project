"""
FORS-EMG TinyML Benchmark — LDA variant
=======================================
Evaluates an INT8 Linear Discriminant Analysis classifier on the selected
4-gesture subset, subject-independently (leave-one-subject-out).

Revision 3 (ASIC-friendly integer features) — FINAL CONFIGURATION
-------------------------------------------------------------------
Defaults are the validated fully-integer front end (config H6 + shift norm in
K_4_LDA_HW_Experiments.py; 77.79 ± 4.39 % vs float reference 77.88 %):
  FILTER_MODE      = "causal"     (4 bandpass + 1 notch biquad, forward only)
  FEATURE_MODE     = "hw"         16-bit integer filtered signal
  HW_FEATURE_TYPES = SABS, WL, ZC_NODB, SSC, POW_M1   (40 inputs)
  INPUT_NORM       = "shift"      (x − μ) >> k → INT8
  Classifier       : INT8 LDA, 176 B, 160 MACs, argmax on INT32 accumulator
Set FEATURE_MODE = "float" / INPUT_NORM = "standard" to reproduce the float
reference {MAV, WL, ZC, SSC, VAR}.

* FEATURE_MODE = "hw" switches to integer features computed with only
  add/subtract, absolute value, compare, shift and leading-one detection
  (see extract_hw_features and K_4_LDA_HW_Experiments.py). "float" is the
  reference pipeline.
* INPUT_NORM = "shift" replaces z-scoring AND the float INT8 input quantiser
  by an integer stage: q = clip((x − μ) >> k), μ and k integers per feature.
* Default float feature set is the validated 5-type set
  {MAV, WL, ZC, SSC, VAR} (config D of K_4_LDA_Experiments.py).

Revision 2 (deployment-matched pipeline)
----------------------------------------
* FILTER_MODE = "causal": the bandpass + notch now run as a causal SOS cascade
  (scipy sosfilt), exactly what firmware can run sample-by-sample. The old
  zero-phase filtfilt is non-causal and applies |H|² instead of |H|, so the
  device would see different features from the ones the model was trained on.
  "zerophase" reproduces the original pipeline bit-for-bit.
* FEATURE_TYPES: which of the 8 time-domain feature types feed the LDA.
  Default is the 6-type set {MAV, RMS, WL, ZC, SSC, VAR}: IEMG is dropped
  (IEMG = L·MAV ⇒ identical LDA) and WAMP is dropped (its 2%-of-RMS threshold
  saturates it at ~98% of its maximum ⇒ near-constant).
* LOG_AMPLITUDE: if True, the gain-dependent features (MAV, RMS, WL, VAR,
  IEMG) are replaced by their natural log before standardisation. See
  K_4_LDA_Experiments.py for the controlled comparison.

What changed vs. the MLP version
--------------------------------
* Classifier: sklearn LinearDiscriminantAnalysis (lsqr solver, Ledoit-Wolf
  shrinkage) replaces the 64→16→4 PyTorch MLP. No torch dependency.
* LDA is a closed-form fit: no optimiser, epochs, or early stopping. The
  trial-5 split is still made exactly as before, so LDA is fit on EXACTLY the
  windows the MLP trained on (trials 1-4 of the 18 training subjects). Trial 5
  is now used only to report a held-out-trial (within-population) accuracy.
* Quantisation: the LDA decision function g_k(x) = w_k·x + b_k is a single
  64→4 linear layer, so the same symmetric per-tensor INT8 PTQ applies
  unchanged (INT8 weights, INT32 biases, argmax on the integer accumulator).
* Reporting: train / val / test accuracy, FP32 and INT8, per fold and
  aggregated (mean ± std, plus a t-based 95% CI on the test mean).
* The confusion matrix is built from the SAME per-fold predictions that produce
  the reported accuracies (the MLP script retrained a second set of models).

Configuration decided from prior analysis
------------------------------------------
* Gestures: Hand Close, Hand Open, Wrist Flexion, Wrist Extension — chosen via
  two-stage pairwise-separability search over all 12 FORS-EMG gestures, after
  the original {Hand Close, Hand Open, Index, Index Little} set showed the
  Index/Index Little pair alone accounted for ~40% of all classification
  errors.

Protocol (unchanged)
--------------------
* Data loading: project-native loader — iterates Subject<i>/<Orientation>/
  <gesture>-<trial>.mat, handles both (8000,8) and (8,8000) mat layouts,
  carries full metadata (subject, orientation, trial, rec_id).
* Features: 8 TD features × 8 channels = 64-D input vector per window.
* Windowing: 200 ms windows, 100 ms increment (50% overlap) at 985 Hz.
  Windows are cut AFTER the LOSO split — no window ever crosses the
  train/test boundary.
* Statistics (StandardScaler) fit on training windows ONLY.
* Preprocessing: 20-450 Hz bandpass + 50 Hz mains notch, applied once per
  recording before windowing.
* Quantisation: symmetric per-tensor INT8 PTQ; biases as INT32.
  Requant multiplier M = sw·sx/sy decomposed as M0·2^-n (only used for
  hidden layers; LDA has none, so inference is one INT8 GEMV + argmax).

Configuration
-------------
  Set BASE_DIR below to the root folder that contains Subject1/, Subject2/, …
  Then run:
      python fors_emg_benchmark_lda.py
"""

import os, sys, time, warnings, csv
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfilt_zi, tf2sos
from scipy.stats import t as student_t
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings("ignore")
np.random.seed(0)

# ─── Dataset configuration ────────────────────────────────────────────────────
# ↓ Edit this to match your local path
BASE_DIR = r"/Users/vinayhariharan/Projects/BTP/NN_model/archive/FORS-EMG Dataset/FORS-EMG Dataset/FORS-EMG"

CLASSES = [0, 1, 2, 3]
GESTURE_FILE_MAP = {
    0: "Hand_Close",
    1: "Hand_Open",
    2: "Wrist_Flexion",
    3: "Wrist_Extension",
}
GESTURE_NAMES = {
    0: "Hand Close",
    1: "Hand Open",
    2: "Wrist Flexion",
    3: "Wrist Extension",
}
FOREARM_ORIENTATIONS = ["Rest", "Supination", "Pronation"]
N_TRIALS = 5
N_SUBJECTS = 19   # subjects 1–19
VAL_TRIAL_ID = 5  # trial held out from each training subject

# ─── Signal / feature constants ───────────────────────────────────────────────
FS          = 985
WIN_SAMPLES = int(200 * FS / 1000)   # 197 samples  @ 200 ms
INC_SAMPLES = int(100 * FS / 1000)   #  98 samples  @ 100 ms
N_CHANNELS  = 8
N_CLASSES   = len(CLASSES)            # 4

# ─── Feature configuration ────────────────────────────────────────────────────
# extract_features_from_windows() always computes all 8 types (64 columns,
# layout ch*8 + type); transform_features() then selects FEATURE_TYPES and
# optionally log-transforms the gain-dependent ones.
ALL_FEATURE_NAMES  = ["MAV", "RMS", "WL", "ZC", "SSC", "VAR", "WAMP", "IEMG"]
N_FEATURES_RAW     = len(ALL_FEATURE_NAMES)                       # 8
AMPLITUDE_FEATURES = {"MAV", "RMS", "WL", "VAR", "IEMG"}          # scale with channel gain
FEATURE_TYPES      = ["MAV", "WL", "ZC", "SSC", "VAR"]            # validated 5-type set
LOG_AMPLITUDE      = False
LOG_FLOOR          = 1e-12                                        # guards log(0) only

# ─── Hardware-friendly (integer, multiplier-free) feature configuration ───────
# Primitives (all per channel, per window of L integer samples x[n]):
#   SABS      Σ|x|                                  (= L·MAV; add + conditional negate)
#   WL        Σ|x[n]−x[n−1]|                        (sub, abs, add)
#   SSC       #{n : Δx[n], Δx[n−1] strictly opposite sign}  (sign-bit compare, count)
#   ZC_MAV14  sign-bit change with deadband (|x| << 14) ≥ Σ|x|   (≈ 1% RMS; shift, compare)
#   ZC_NODB   sign-bit change, no deadband           (XOR of sign bits, count)
#   POW       Σx²   exact (serial shift-add squarer / time-multiplexed)
#   POW_M<k>  Σ Mitchell-k approximate x²            (leading-one detect, shift, sub)
#   SABS2     (Σ|x|)²  one square per channel per WINDOW (serial shift-add)
FEATURE_MODE      = "hw"                    # "float" | "hw"
HW_FEATURE_TYPES  = ["SABS", "WL", "ZC_NODB", "SSC", "POW_M1"]   # validated H6
SIGNAL_BITS       = 16                      # word length of the filtered signal
ZC_DEADBAND_SHIFT = 14                      # ZC_MAV14 threshold = Σ|x| / 2^14
INPUT_NORM        = "shift"                 # "standard" (float z-score) | "shift" (integer)
HW_FULL_SCALE     = None                    # set at runtime by set_hw_full_scale()

N_FEATURES  = len(HW_FEATURE_TYPES) if FEATURE_MODE == "hw" else len(FEATURE_TYPES)
INPUT_DIM   = N_CHANNELS * N_FEATURES

# ─── Quantisation constants ───────────────────────────────────────────────────
INT8_MAX  = 127
INT8_MIN  = -128
CALIB_PCT = 99.9   # percentile for robust scale computation
REQUANT_SHIFT = 31 # fixed-point precision of the requantisation multiplier

# ─── LDA configuration ────────────────────────────────────────────────────────
# Ledoit-Wolf shrinkage keeps the pooled covariance well-conditioned. With all
# 8 types it is REQUIRED: IEMG = L·MAV makes the covariance exactly singular.
# With the 6-type set it is still needed in practice: VAR ≈ RMS² (mean ≈ 0
# after the bandpass) and MAV/RMS/VAR have |ρ| ≈ 0.998, so the covariance is
# near-singular. Shrinkage Σ̂ₖ = (1−αₖ)Sₖ + αₖ·diag(Sₖ) bounds its condition
# number with a data-driven αₖ.
LDA_SOLVER    = "lsqr"
LDA_SHRINKAGE = "auto"   # Ledoit-Wolf

# ─── Preprocessing filter configuration ───────────────────────────────────────
APPLY_FILTER    = True
FILTER_MODE     = "causal"      # "causal" (deployable) | "zerophase" (original)
BANDPASS_HZ     = (20.0, 450.0)
MAINS_NOTCH_HZ  = 50.0
NOTCH_Q         = 30.0


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_single_subject(
    base_dir: str,
    subject_id: int,
    classes: list      = CLASSES,
    gesture_file_map   = GESTURE_FILE_MAP,
    orientations: list = FOREARM_ORIENTATIONS,
    n_trials: int      = N_TRIALS,
) -> tuple:
    """
    Loads raw sEMG recording matrices and per-recording metadata for ONE
    subject.

    Returns
    -------
    recordings : list of np.ndarray, each (8000, 8) float64
    recording_meta : list of dict with keys:
        label, subject, orientation, trial, rec_id
    """
    recordings     = []
    recording_meta = []

    subject_dir = os.path.join(base_dir, f"Subject{subject_id}")
    if not os.path.isdir(subject_dir):
        return recordings, recording_meta

    rec_id = 0
    for orientation in orientations:
        orient_dir = os.path.join(subject_dir, orientation)
        for class_idx in classes:
            prefix = gesture_file_map[class_idx]
            for trial in range(1, n_trials + 1):
                fpath = os.path.join(orient_dir, f"{prefix}-{trial}.mat")
                if not os.path.exists(fpath):
                    continue

                mat = sio.loadmat(fpath)
                for key, val in mat.items():
                    if key.startswith("_") or not isinstance(val, np.ndarray):
                        continue
                    if val.ndim != 2:
                        continue
                    if val.shape == (8000, 8):
                        data = val.astype(np.float64)
                    elif val.shape == (8, 8000):
                        data = val.T.astype(np.float64)
                    else:
                        continue

                    recordings.append(data)
                    recording_meta.append({
                        "label":       class_idx,
                        "subject":     subject_id,
                        "orientation": orientation,
                        "trial":       trial,
                        "rec_id":      rec_id,
                    })
                    rec_id += 1
                    break

    return recordings, recording_meta


def load_all_subjects(base_dir: str = BASE_DIR) -> list:
    if not os.path.isdir(base_dir):
        print(f"\n  ERROR: dataset root not found:\n    {base_dir}")
        print("  Set BASE_DIR at the top of this file to the folder that "
              "contains Subject1/, Subject2/, … and re-run.\n")
        sys.exit(1)

    subjects = []
    for sid in tqdm(range(1, N_SUBJECTS + 1), desc="Loading subjects"):
        recs, meta = _load_single_subject(base_dir, sid)
        if not recs:
            print(f"  WARNING: no recordings found for Subject{sid} — skipping.")
            continue
        subjects.append({
            "subject_id": sid,
            "recordings": recs,
            "labels":     [m["label"] for m in meta],
            "meta":       meta,
        })
    return subjects


# ─── Signal conditioning (bandpass + mains notch) ─────────────────────────────

def _design_filters(fs=FS, band=BANDPASS_HZ, notch_freq=MAINS_NOTCH_HZ, notch_q=NOTCH_Q):
    """
    Returns the filter in both forms:
      'ba'  : ((b_bp, a_bp), (b_n, a_n))  — original zero-phase path, unchanged
      'sos' : (n_sections, 6) cascade     — 4 bandpass biquads + 1 notch biquad,
              the exact coefficients to put in firmware (e.g. CMSIS-DSP
              arm_biquad_cascade_df2T_f32; note CMSIS omits a0 and negates a1,a2).
    """
    nyq = fs / 2.0
    low  = band[0] / nyq
    high = min(band[1], nyq - 1.0) / nyq
    b_bp, a_bp = butter(4, [low, high], btype="bandpass")
    b_n,  a_n  = iirnotch(notch_freq, notch_q, fs)
    sos = np.vstack([butter(4, [low, high], btype="bandpass", output="sos"),
                     tf2sos(b_n, a_n)])
    return {"ba": ((b_bp, a_bp), (b_n, a_n)), "sos": sos}


def apply_preprocessing_filter(sig_channels_first, filt, mode=FILTER_MODE):
    """
    sig_channels_first : (n_channels, n_samples)
    causal    : one forward pass of the SOS cascade (what the MCU runs). The
                state is initialised to the steady state for the first sample
                (sosfilt_zi · x[0]), which removes the start-up transient from
                a DC offset — equivalent to a filter that has been running
                continuously on the device before the recording starts.
    zerophase : original filtfilt path (non-causal, |H|² magnitude).
    """
    if mode == "zerophase":
        (b_bp, a_bp), (b_n, a_n) = filt["ba"]
        sig = filtfilt(b_bp, a_bp, sig_channels_first, axis=-1)
        return filtfilt(b_n, a_n, sig, axis=-1)
    if mode == "causal":
        sos = filt["sos"]
        zi = sosfilt_zi(sos)[:, None, :] * sig_channels_first[None, :, 0:1]
        y, _ = sosfilt(sos, sig_channels_first, axis=-1, zi=zi)
        return y
    raise ValueError(f"unknown FILTER_MODE {mode!r}")


def filter_all_subjects(subjects, apply_filter=APPLY_FILTER, mode=None):
    mode = FILTER_MODE if mode is None else mode
    if not apply_filter:
        print("  Preprocessing filter: DISABLED (APPLY_FILTER=False)")
        return subjects

    filt = _design_filters()
    print(f"  Preprocessing filter: {BANDPASS_HZ[0]:.0f}-{BANDPASS_HZ[1]:.0f} Hz "
          f"bandpass + {MAINS_NOTCH_HZ:.0f} Hz notch ({mode})")

    for s in tqdm(subjects, desc="Filtering signals"):
        filtered = []
        for rec in s["recordings"]:
            sig_cf = rec.T
            sig_filtered = apply_preprocessing_filter(sig_cf, filt, mode)
            filtered.append(sig_filtered.T.astype(np.float64))
        s["recordings"] = filtered
    return subjects


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features_from_windows(windows):
    """
    windows : (N, C, L)  →  features : (N, 64)
    8 features per channel: MAV, RMS, WL, ZC, SSC, VAR, WAMP, IEMG
    """
    N, C, L = windows.shape
    x    = windows
    dx   = np.diff(x, axis=2)
    xabs = np.abs(x)

    mav  = xabs.mean(axis=2)
    rms  = np.sqrt((x**2).mean(axis=2))
    wl   = np.abs(dx).sum(axis=2)

    thr  = rms * 0.01
    sx   = np.sign(x)
    zc   = ((sx[:,:,1:] != sx[:,:,:-1]) &
            ((xabs[:,:,:-1] >= thr[:,:,None]) |
             (xabs[:,:,1:]  >= thr[:,:,None]))).sum(axis=2)

    ssc  = ((dx[:,:,1:] * dx[:,:,:-1]) < 0).sum(axis=2)

    var  = x.var(axis=2)
    wamp_thr = (rms * 0.02)[:, :, None]
    wamp = (np.abs(dx) > wamp_thr).sum(axis=2)
    iemg = xabs.sum(axis=2)

    feat = np.stack([mav, rms, wl, zc, ssc, var, wamp, iemg], axis=2)
    return feat.reshape(N, C * 8).astype(np.float32)


def feature_columns(feature_types):
    """Column indices (into the 64-D raw feature matrix) for the given types,
    channel-major, types in the order given."""
    idx = [ALL_FEATURE_NAMES.index(f) for f in feature_types]
    return np.array([c * N_FEATURES_RAW + f for c in range(N_CHANNELS) for f in idx])


def transform_features(F_raw, feature_types=None, log_amplitude=None):
    """
    F_raw : (N, 64) output of extract_features_from_windows
    Selects the requested feature types and, if log_amplitude, replaces every
    gain-dependent feature (MAV, RMS, WL, VAR, IEMG) by its natural log.
    ZC and SSC are counts with RMS-relative thresholds (gain-invariant) and are
    left linear.
    """
    feature_types = FEATURE_TYPES if feature_types is None else feature_types
    log_amplitude = LOG_AMPLITUDE if log_amplitude is None else log_amplitude
    X = F_raw[:, feature_columns(feature_types)].astype(np.float64)
    if log_amplitude:
        amp = np.array([f in AMPLITUDE_FEATURES for f in feature_types] * N_CHANNELS)
        X[:, amp] = np.log(np.maximum(X[:, amp], LOG_FLOOR))
    return X.astype(np.float32)


# ─── Hardware-friendly integer features ───────────────────────────────────────

def set_hw_full_scale(subjects):
    """
    Fixed-point full scale of the filtered signal = max |x| over the dataset,
    so the SIGNAL_BITS-bit representation never clips. This is a fixed
    hardware gain (like choosing an ADC range), not a fitted parameter: it uses
    no labels, and LDA predictions are invariant to a global gain, so it only
    sets the quantisation step.
    """
    global HW_FULL_SCALE
    HW_FULL_SCALE = float(max(np.abs(r).max() for s in subjects for r in s["recordings"]))
    return HW_FULL_SCALE


def quantize_signal(x, full_scale=None, bits=None):
    """Round the filtered signal onto a signed `bits`-bit integer grid."""
    full_scale = HW_FULL_SCALE if full_scale is None else full_scale
    bits = SIGNAL_BITS if bits is None else bits
    q = 2 ** (bits - 1) - 1
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * (q / full_scale)),
                   -q - 1, q).astype(np.int64)


def leading_one(v):
    """Priority encoder: floor(log2 v) for integer v > 0, −1 for v = 0.
    (frexp is exact for integers < 2^53.)"""
    v = np.asarray(v, dtype=np.int64)
    out = np.full(v.shape, -1, dtype=np.int64)
    nz = v > 0
    out[nz] = np.frexp(v[nz].astype(np.float64))[1] - 1
    return out


def mitchell_square(a, levels):
    """
    Multiplier-free approximation of a² for integers a ≥ 0.
    With p = leading_one(a) and r = a − 2^p (0 ≤ r < 2^p):
        a² = 2^{2p} + 2^{p+1}·r + r²  =  [(a << (p+1)) − (1 << 2p)]  +  r²
    Each level adds the bracket (shift + subtract) and recurses on r².
    Truncating after k levels drops the final r_k² ≥ 0, so the result never
    overestimates, and the relative error is < 4^{−k} (25 % for k = 1,
    6.25 % for k = 2); k ≥ popcount(a) is exact.
    """
    r  = np.asarray(a, dtype=np.int64).copy()
    sq = np.zeros_like(r)
    for _ in range(levels):
        nz = r > 0
        if not nz.any():
            break
        p = np.where(nz, leading_one(r), 0)
        sq += np.where(nz, (r << (p + 1)) - (np.int64(1) << (2 * p)), 0)
        r   = np.where(nz, r - (np.int64(1) << p), 0)
    return sq


def hw_feature_dict(win_int, names):
    """
    win_int : (N, C, L) int64 filtered, quantised windows
    Returns {name: (N, C) int64} for each requested primitive.
    """
    x  = np.asarray(win_int, dtype=np.int64)
    a  = np.abs(x)
    need = set(names)
    out = {}
    sabs = a.sum(axis=2)
    if "SABS" in need:
        out["SABS"] = sabs
    if "SABS2" in need:
        out["SABS2"] = sabs * sabs
    if need & {"WL", "SSC"}:
        d = np.diff(x, axis=2)
        if "WL" in need:
            out["WL"] = np.abs(d).sum(axis=2)
        if "SSC" in need:
            d1, d0 = d[:, :, 1:], d[:, :, :-1]
            out["SSC"] = (((d1 > 0) & (d0 < 0)) | ((d1 < 0) & (d0 > 0))).sum(axis=2)
    if need & {"ZC_MAV14", "ZC_NODB"}:
        neg = x < 0                                   # sign bit
        flip = neg[:, :, 1:] != neg[:, :, :-1]
        if "ZC_NODB" in need:
            out["ZC_NODB"] = flip.sum(axis=2)
        if "ZC_MAV14" in need:
            big = (a << ZC_DEADBAND_SHIFT) >= sabs[:, :, None]   # |x| ≥ Σ|x| / 2^14
            out["ZC_MAV14"] = (flip & (big[:, :, :-1] | big[:, :, 1:])).sum(axis=2)
    if "POW" in need:
        out["POW"] = (x * x).sum(axis=2)
    for n in need:
        if n.startswith("POW_M"):
            out[n] = mitchell_square(a, int(n[5:])).sum(axis=2)
    missing = need - set(out)
    if missing:
        raise ValueError(f"unknown hardware feature(s): {sorted(missing)}")
    return out


def extract_hw_features(win_int, feature_types=None):
    """(N, C, L) int windows → (N, C·T) float64, channel-major like the float path."""
    feature_types = HW_FEATURE_TYPES if feature_types is None else feature_types
    d = hw_feature_dict(win_int, feature_types)
    N, C = next(iter(d.values())).shape
    return np.stack([d[f] for f in feature_types], axis=2).reshape(N, C * len(feature_types)).astype(np.float64)


def compute_features(windows):
    """Dispatch on FEATURE_MODE: float reference or integer hardware features."""
    if FEATURE_MODE == "hw":
        if HW_FULL_SCALE is None:
            raise RuntimeError("HW_FULL_SCALE not set — call set_hw_full_scale(subjects) first")
        return extract_hw_features(quantize_signal(windows), HW_FEATURE_TYPES)
    return transform_features(extract_features_from_windows(windows))


class ShiftNormalizer:
    """
    Integer-only replacement for StandardScaler + the INT8 input quantiser:
        q_f = clip( round((x_f − μ_f) / 2^{k_f}), −128, 127 )
    μ_f = round(training mean) (integer), and k_f = ceil(log2(P99.9|x_f − μ_f| / 127))
    is the smallest shift that fits the 99.9th percentile into INT8. k_f < 0
    means a left shift. In hardware: one subtract and one arithmetic shift
    (with a round-half-up add of 2^{k−1}) per feature. The output is already
    INT8, so the classifier uses an input scale of exactly 1.
    """
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = np.round(X.mean(axis=0))
        dev = np.maximum(np.percentile(np.abs(X - self.mu_), CALIB_PCT, axis=0), 1.0)
        self.k_ = np.ceil(np.log2(dev / INT8_MAX)).astype(np.int64)
        return self

    def transform(self, X):
        d = np.round(np.asarray(X, dtype=np.float64) - self.mu_).astype(np.int64)
        out = np.empty_like(d)
        for f, k in enumerate(self.k_):
            if k > 0:
                out[:, f] = (d[:, f] + (np.int64(1) << (k - 1))) >> k
            else:
                out[:, f] = d[:, f] << (-k)
        return np.clip(out, INT8_MIN, INT8_MAX).astype(np.float64)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_normalizer(mode=None):
    mode = INPUT_NORM if mode is None else mode
    return ShiftNormalizer() if mode == "shift" else StandardScaler()


def windows_from_recordings(recordings: list, labels: list) -> tuple:
    all_win, all_lbl = [], []
    for rec, lbl in zip(recordings, labels):
        sig    = rec.T
        n_samp = sig.shape[1]
        starts = np.arange(0, n_samp - WIN_SAMPLES + 1, INC_SAMPLES)
        for s in starts:
            all_win.append(sig[:, s : s + WIN_SAMPLES])
        all_lbl.extend([lbl] * len(starts))

    if len(all_win) == 0:
        return (np.empty((0, N_CHANNELS, WIN_SAMPLES), dtype=np.float32),
                np.empty((0,), dtype=np.int64))
    return (np.stack(all_win).astype(np.float32),
            np.array(all_lbl, dtype=np.int64))


# ─── INT8 PTQ ─────────────────────────────────────────────────────────────────
# Same arithmetic as the MLP version, refactored to (a) take a list of (W, b)
# layers instead of an nn.Module, and (b) split calibration from inference so
# one calibration can be applied to the train, val and test sets.

def compute_scale(arr):
    """Symmetric per-tensor scale from 99.9-percentile of |arr|."""
    v = np.percentile(np.abs(arr.ravel()), CALIB_PCT)
    return float(v) / INT8_MAX if v > 0 else 1.0

def quantize_to_int8(arr, scale):
    return np.clip(np.round(arr / scale), INT8_MIN, INT8_MAX).astype(np.int8)

def calibrate_ptq(layers, X_calib, input_scale=None):
    """
    layers  : list of (W (out,in), b (out,)) float arrays, in forward order
    X_calib : normalised training features
    Returns (sx_in, layer_info) — everything integer inference needs.
    """
    # input_scale=1.0 when the inputs are already INT8 (ShiftNormalizer)
    sx_in = compute_scale(X_calib) if input_scale is None else float(input_scale)
    sx    = sx_in
    x_fp  = X_calib.astype(np.float64)
    layer_info = []
    for i, (W, b) in enumerate(layers):
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        sw = compute_scale(W)

        z = x_fp @ W.T + b
        is_last = (i == len(layers) - 1)
        if is_last:
            sy = compute_scale(z)
        else:
            sy = compute_scale(np.maximum(0, z))
            x_fp = np.maximum(0, z)

        M  = (sw * sx) / sy
        M0 = int(np.round(M * (1 << REQUANT_SHIFT))) if M > 0 else 0

        W_q = quantize_to_int8(W, sw).astype(np.int32)
        b_real = b / (sw * sx)
        if np.max(np.abs(b_real)) >= 2**31 - 1:
            raise OverflowError(
                f"Layer {i}: bias/(sw·sx) = {np.max(np.abs(b_real)):.3e} "
                "does not fit in INT32.")
        b_q = np.round(b_real).astype(np.int32)
        layer_info.append((W_q, b_q, sy, M0, REQUANT_SHIFT, is_last))
        sx = sy
    return sx_in, layer_info

def int8_infer(qparams, X):
    """Integer-only inference; returns argmax class INDEX per row."""
    sx_in, layer_info = qparams
    x_q = np.clip(np.round(X / sx_in), INT8_MIN, INT8_MAX).astype(np.int32)
    for W_q, b_q, sy, M0, n_shift, is_last in layer_info:
        z = (x_q.astype(np.int64) @ W_q.T.astype(np.int64)
             + b_q[None, :].astype(np.int64))
        if is_last:
            # Requantisation is a positive monotone scaling → argmax unchanged.
            return np.argmax(z, axis=1)
        z_req = ((z * np.int64(M0)) + (np.int64(1) << (n_shift - 1))) >> n_shift
        x_q   = np.maximum(0, np.clip(z_req, INT8_MIN, INT8_MAX)).astype(np.int32)


# ─── LDA ──────────────────────────────────────────────────────────────────────

def build_lda():
    return LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE)

def lda_as_linear_layer(lda):
    """
    Express the fitted LDA as one linear layer (W, b) with class = argmax(Wx+b).

    With shared covariance Σ̂ and class means μ_k, the discriminant is
        g_k(x) = μ_kᵀΣ̂⁻¹x − ½ μ_kᵀΣ̂⁻¹μ_k + log π_k
    so W[k] = Σ̂⁻¹μ_k (sklearn coef_) and b[k] = −½ μ_kᵀΣ̂⁻¹μ_k + log π_k
    (sklearn intercept_). For K = 2 sklearn stores a single discriminant
    g = g_1 − g_0; it is expanded to two rows [0; g] so argmax still applies.
    """
    W = np.asarray(lda.coef_, dtype=np.float64)
    b = np.asarray(lda.intercept_, dtype=np.float64)
    if W.shape[0] == 1:
        W = np.vstack([np.zeros_like(W[0]), W[0]])
        b = np.array([0.0, b[0]])
    return W, b

# Deployed model: INT8 W (K×D) + INT32 b (K).
PARAM_BYTES = INPUT_DIM * N_CLASSES + N_CLASSES * 4          # 208 B for 6 types
NUM_MACS    = INPUT_DIM * N_CLASSES                          # 192 for 6 types
DEPLOYED_PARAMS = INPUT_DIM * N_CLASSES + N_CLASSES
# Quantities actually ESTIMATED from data during fitting (the relevant count
# for overfitting): K class means (K·D) + one symmetric pooled covariance
# (D(D+1)/2) + K−1 free priors. Shrinkage lowers the effective count further.
ESTIMATED_PARAMS = N_CLASSES * INPUT_DIM + INPUT_DIM * (INPUT_DIM + 1) // 2 + (N_CLASSES - 1)
MODEL_LABEL = f"LDA {INPUT_DIM}→{N_CLASSES}"


# ─── LOSO-CV ──────────────────────────────────────────────────────────────────

def _val_split_by_trial(recordings, labels, meta, val_trial_id=VAL_TRIAL_ID):
    """Trial `val_trial_id` → val; all other trials → train (unchanged)."""
    train_recs, train_lbls = [], []
    val_recs,   val_lbls   = [], []
    for rec, lbl, m in zip(recordings, labels, meta):
        if m["trial"] == val_trial_id:
            val_recs.append(rec);   val_lbls.append(lbl)
        else:
            train_recs.append(rec); train_lbls.append(lbl)
    return train_recs, train_lbls, val_recs, val_lbls


def loso_cv(subjects):
    """
    Leave-One-Subject-Out CV. Per fold:
      train = trials 1-4 of the 18 training subjects   (LDA fit, scaler fit, PTQ calib)
      val   = trial 5    of the 18 training subjects   (report only)
      test  = all trials of the held-out subject
    Returns a list of per-fold dicts with accuracies and test predictions.
    """
    folds = []
    for test_idx in tqdm(range(len(subjects)), desc=f"  {MODEL_LABEL:18s}", leave=False):
        test_s   = subjects[test_idx]
        train_ss = [s for i, s in enumerate(subjects) if i != test_idx]

        X_te_win, y_te = windows_from_recordings(test_s['recordings'], test_s['labels'])

        all_train_recs, all_train_lbls = [], []
        all_val_recs,   all_val_lbls   = [], []
        for s in train_ss:
            tr_r, tr_l, va_r, va_l = _val_split_by_trial(
                s['recordings'], s['labels'], s['meta'])
            all_train_recs.extend(tr_r); all_train_lbls.extend(tr_l)
            all_val_recs.extend(va_r);   all_val_lbls.extend(va_l)

        X_tr_win, y_tr = windows_from_recordings(all_train_recs, all_train_lbls)
        X_va_win, y_va = windows_from_recordings(all_val_recs,   all_val_lbls)

        X_tr_feat = compute_features(X_tr_win); del X_tr_win
        X_va_feat = compute_features(X_va_win); del X_va_win
        X_te_feat = compute_features(X_te_win); del X_te_win

        sc     = make_normalizer()
        X_tr_n = sc.fit_transform(X_tr_feat).astype(np.float64)
        X_va_n = sc.transform(X_va_feat).astype(np.float64)
        X_te_n = sc.transform(X_te_feat).astype(np.float64)

        # ── LDA fit (closed form, train windows only) ────────────────────────
        lda = build_lda().fit(X_tr_n, y_tr)
        W, b = lda_as_linear_layer(lda)

        # ── INT8 PTQ: calibrate on train windows only ────────────────────────
        qparams = calibrate_ptq([(W, b)], X_tr_n,
                                input_scale=1.0 if INPUT_NORM == "shift" else None)

        sets = {"train": (X_tr_n, y_tr), "val": (X_va_n, y_va), "test": (X_te_n, y_te)}
        acc, preds = {}, {}
        for name, (X, y) in sets.items():
            p_fp = lda.predict(X)
            p_q8 = lda.classes_[int8_infer(qparams, X)]
            acc[f"{name}_fp32"]  = accuracy_score(y, p_fp)
            acc[f"{name}_int8"]  = accuracy_score(y, p_q8)
            acc[f"{name}_agree"] = float(np.mean(p_fp == p_q8))
            preds[name] = (y, p_fp, p_q8)

        # Sanity: (W, b) must reproduce sklearn's FP32 decisions.
        p_lin = lda.classes_[np.argmax(X_te_n @ W.T + b, axis=1)]
        n_mis = int(np.sum(p_lin != preds["test"][1]))
        if n_mis > 0:
            print(f"\n  WARNING: fold S{test_s['subject_id']:02d}: linear-layer form "
                  f"disagrees with lda.predict on {n_mis} test windows.")

        folds.append(dict(
            subject_id = test_s['subject_id'],
            n_train = len(y_tr), n_val = len(y_va), n_test = len(y_te),
            y_te = y_te, p_te_fp32 = preds["test"][1], p_te_int8 = preds["test"][2],
            **acc,
        ))
    return folds


# ─── Reporting helpers ────────────────────────────────────────────────────────

def _stat(folds, key):
    v = np.array([f[key] for f in folds])
    n = len(v)
    half = (student_t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return v.mean(), v.std(), half


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  FORS-EMG TinyML Benchmark — LDA")
    print("=" * 65)

    subjects = load_all_subjects(BASE_DIR)
    total_recs = sum(len(s['recordings']) for s in subjects)
    print(f"\n  Subjects loaded  : {len(subjects)}")
    print(f"  Total recordings : {total_recs}")
    print(f"  Target gestures  : {', '.join(GESTURE_NAMES[c] for c in CLASSES)}")
    print(f"  Orientations     : {', '.join(FOREARM_ORIENTATIONS)}")
    print(f"  Window           : {WIN_SAMPLES} samples ({200} ms), "
          f"increment {INC_SAMPLES} samples ({100} ms)")
    if FEATURE_MODE == "hw":
        print(f"  Feature types    : {HW_FEATURE_TYPES}  (integer hw, {SIGNAL_BITS}-bit signal)")
    else:
        print(f"  Feature types    : {FEATURE_TYPES}  (log-amplitude: {LOG_AMPLITUDE})")
    print(f"  Input norm       : {INPUT_NORM}")
    print(f"  Input dim        : {INPUT_DIM}  ({N_CHANNELS} ch × {N_FEATURES} feats)")
    print(f"  Filter mode      : {FILTER_MODE}")
    print(f"  Classifier       : LDA (solver={LDA_SOLVER}, shrinkage={LDA_SHRINKAGE})")

    sample_rec = subjects[0]['recordings'][0]
    print(f"\n  Raw signal scale check (Subject{subjects[0]['subject_id']}, "
          f"first recording):")
    print(f"    shape={sample_rec.shape}  min={sample_rec.min():.4f}  "
          f"max={sample_rec.max():.4f}  mean={sample_rec.mean():.4f}  "
          f"std={sample_rec.std():.4f}")

    subjects = filter_all_subjects(subjects, apply_filter=APPLY_FILTER)
    if FEATURE_MODE == "hw":
        fs_ = set_hw_full_scale(subjects)
        print(f"  HW full scale    : {fs_:.6g} → {SIGNAL_BITS}-bit signed "
              f"(LSB = {fs_ / (2 ** (SIGNAL_BITS - 1) - 1):.4g})")

    all_labels = [lbl for s in subjects for lbl in s['labels']]
    print(f"\n  Class balance (recordings):")
    for c in CLASSES:
        cnt = all_labels.count(c)
        print(f"    [{c}] {GESTURE_NAMES[c]:14s}: {cnt:4d}  "
              f"({cnt/len(all_labels)*100:.1f}%)")

    # ── Budget ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Model budget (INT8 weights + INT32 biases)")
    print(f"{'─'*65}")
    print(tabulate([[MODEL_LABEL, f"{PARAM_BYTES:,}", f"{NUM_MACS:,}",
                     "✓" if PARAM_BYTES <= 10_000 else "✗ OVER"]],
                   headers=["Model", "Param bytes", "MACs", "≤10 KB?"],
                   tablefmt="rounded_outline"))
    # Worst-case accumulator: |Σ x_q·w_q| ≤ D·128·127, plus |b_q|.
    print(f"  Worst-case |accumulator| before bias: {INPUT_DIM*128*127:,} "
          f"(fits INT32 with large margin)")

    # ── Overfitting-risk check ───────────────────────────────────────────────
    total_windows = 0
    for s in subjects:
        w, _ = windows_from_recordings(s['recordings'], s['labels'])
        total_windows += len(w)
    avg_test_fold      = total_windows / len(subjects)
    avg_train_pool     = avg_test_fold * (len(subjects) - 1)
    avg_train_post_val = avg_train_pool * (1 - 1.0 / N_TRIALS)

    print(f"\n{'─'*65}")
    print("  Overfitting-risk check")
    print(f"{'─'*65}")
    print(f"  Total windows (all subjects) : {total_windows:,.0f}")
    print(f"  Avg. training windows/fold   : {avg_train_post_val:,.0f}  "
          f"(after held-out val trial)")
    print(tabulate(
        [["Deployed (W, b)",            f"{DEPLOYED_PARAMS:,}",
          f"{avg_train_post_val / DEPLOYED_PARAMS:,.1f}x"],
         ["Estimated (means + Σ + priors)", f"{ESTIMATED_PARAMS:,}",
          f"{avg_train_post_val / ESTIMATED_PARAMS:,.1f}x"]],
        headers=["Parameter count", "Params", "Samples per param"],
        tablefmt="rounded_outline"))
    print("  The 'estimated' row is the honest one for LDA: the fit estimates")
    print("  class means and a pooled covariance, of which W,b is a function.")

    # ── LOSO-CV ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  LOSO-CV ({len(subjects)}-fold) — one fold per subject")
    print(f"{'─'*65}\n")
    folds = loso_cv(subjects)

    # ── Aggregate results ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Results  (mean ± std over LOSO folds)")
    print(f"{'─'*65}")
    n_tr_subj = len(subjects) - 1
    split_desc = {
        "train": f"Train (trials 1-4, {n_tr_subj} subj)",
        "val":   f"Val   (trial {VAL_TRIAL_ID}, {n_tr_subj} subj)",
        "test":  "Test  (held-out subject)",
    }
    rows, summary = [], {}
    for sp in ("train", "val", "test"):
        fm, fs, fh = _stat(folds, f"{sp}_fp32")
        qm, qs, qh = _stat(folds, f"{sp}_int8")
        am, _, _   = _stat(folds, f"{sp}_agree")
        summary[sp] = dict(fp32_mean=fm, fp32_std=fs, fp32_ci=fh,
                           int8_mean=qm, int8_std=qs, int8_ci=qh,
                           q_drop=fm - qm, agree=am)
        rows.append([split_desc[sp],
                     f"{fm*100:.2f}±{fs*100:.2f}",
                     f"{qm*100:.2f}±{qs*100:.2f}",
                     f"{(fm-qm)*100:.2f}",
                     f"{am*100:.2f}"])
    print(tabulate(rows,
                   headers=["Split", "FP32 %", "INT8 %", "Q-drop pp", "FP32/INT8 agree %"],
                   tablefmt="rounded_outline"))

    te, tr = summary["test"], summary["train"]
    print(f"\n  Test INT8 accuracy : {te['int8_mean']*100:.2f}% "
          f"(95% CI ±{te['int8_ci']*100:.2f} pp, t-dist, n={len(folds)} folds)")
    print(f"  Test FP32 accuracy : {te['fp32_mean']*100:.2f}% "
          f"(95% CI ±{te['fp32_ci']*100:.2f} pp)")
    print(f"  Train−test gap     : {(tr['fp32_mean']-te['fp32_mean'])*100:.2f} pp FP32, "
          f"{(tr['int8_mean']-te['int8_mean'])*100:.2f} pp INT8")
    print(f"  Param bytes        : {PARAM_BYTES:,} / 10,000")
    print(f"  MACs/inference     : {NUM_MACS:,}")

    # ── Per-subject breakdown ────────────────────────────────────────────────
    print(f"\n  Per-subject accuracy ({MODEL_LABEL}):")
    frows = [[f"S{f['subject_id']:02d}",
              f"{f['train_fp32']:.2%}", f"{f['train_int8']:.2%}",
              f"{f['test_fp32']:.2%}",  f"{f['test_int8']:.2%}",
              f"{(f['test_fp32']-f['test_int8'])*100:.2f}pp",
              f"{(f['train_int8']-f['test_int8'])*100:.2f}pp"]
             for f in folds]
    print(tabulate(frows,
                   headers=["Test subj", "Train FP32", "Train INT8",
                            "Test FP32", "Test INT8", "Q-drop (test)",
                            "Gap INT8 (tr−te)"],
                   tablefmt="simple"))

    # ── Per-class diagnosis (same predictions as the reported accuracies) ────
    all_true = np.concatenate([f["y_te"] for f in folds])
    all_pred = np.concatenate([f["p_te_int8"] for f in folds])
    print(f"\n  Per-class accuracy (INT8, test, pooled over all LOSO folds):")
    print(classification_report(
        all_true, all_pred, labels=CLASSES,
        target_names=[GESTURE_NAMES[c] for c in CLASSES], digits=3))

    cm = confusion_matrix(all_true, all_pred, labels=CLASSES)
    print("  Confusion matrix (rows=true, cols=predicted):")
    cm_rows = [[GESTURE_NAMES[c]] + list(cm[i]) for i, c in enumerate(CLASSES)]
    print(tabulate(cm_rows,
                   headers=["True \\ Pred"] + [GESTURE_NAMES[c] for c in CLASSES],
                   tablefmt="simple"))

    # ── CSV export ───────────────────────────────────────────────────────────
    with open("fors_emg_lda_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Split", "FP32_mean", "FP32_std", "FP32_ci95",
                    "INT8_mean", "INT8_std", "INT8_ci95", "Qdrop_pp",
                    "Agree", "Bytes", "MACs"])
        for sp, r in summary.items():
            w.writerow([MODEL_LABEL, sp,
                        round(r['fp32_mean'], 6), round(r['fp32_std'], 6), round(r['fp32_ci'], 6),
                        round(r['int8_mean'], 6), round(r['int8_std'], 6), round(r['int8_ci'], 6),
                        round(r['q_drop'] * 100, 4), round(r['agree'], 6),
                        PARAM_BYTES, NUM_MACS])

    with open("fors_emg_lda_folds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TestSubject", "n_train", "n_val", "n_test",
                    "train_fp32", "train_int8", "val_fp32", "val_int8",
                    "test_fp32", "test_int8"])
        for fd in folds:
            w.writerow([fd['subject_id'], fd['n_train'], fd['n_val'], fd['n_test']] +
                       [round(fd[k], 6) for k in
                        ("train_fp32", "train_int8", "val_fp32", "val_int8",
                         "test_fp32", "test_int8")])

    print(f"\n  Results saved → fors_emg_lda_results.csv, fors_emg_lda_folds.csv")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
