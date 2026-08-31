"""
FORS-EMG TinyML Architecture Benchmark
=======================================
Evaluates the selected INT8 MLP on the selected 4-gesture subset.

Configuration decided from prior analysis
------------------------------------------
* Gestures: Hand Close, Hand Open, Wrist Flexion, Wrist Extension — chosen via
  two-stage pairwise-separability search over all 12 FORS-EMG gestures, after
  the original {Hand Close, Hand Open, Index, Index Little} set showed the
  Index/Index Little pair alone accounted for ~40% of all classification
  errors (fine finger movements recruiting overlapping forearm muscles,
  poorly resolved by surface EMG).
* Architecture: 64→16→4 (1,168 B, 1,088 MACs) — the 5-way architecture sweep
  on the original gesture set found all candidates statistically tied
  (rank-order unstable across 3 runs, spread 1.3 pp vs. ±5.9 pp 95% CI), so
  architecture is fixed here rather than re-swept; edit HIDDEN_WIDTHS below to
  re-enable the sweep on this gesture set if desired.

Protocol
--------
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
* Training: FP32 PyTorch MLP, Adam + CosineAnnealingLR, early stopping with
  a trial-level (not window-level) validation split.
* Quantisation: symmetric per-tensor INT8 PTQ; biases as INT32.
  Requant multiplier M = sw·sx/sy decomposed as M0·2^-n (pure integer
  hardware arithmetic — no float at inference time).

Configuration
-------------
  Set BASE_DIR below to the root folder that contains Subject1/, Subject2/, …
  Then run:
      python fors_emg_benchmark.py
"""

import os, sys, time, warnings, csv
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

# ─── Dataset configuration ────────────────────────────────────────────────────
# ↓ Edit this to match your local path
BASE_DIR = r"/Users/vinayhariharan/Projects/BTP/NN_model/archive/FORS-EMG Dataset/FORS-EMG Dataset/FORS-EMG"

# Gesture class index → file prefix
# Updated from the original {Hand_Close, Hand_Open, Index, Index_Little} set
# based on the two-stage pairwise-separability search: Index vs Index_Little
# was the dominant confusion (~40% of all errors), while wrist flexion/
# extension recruit distinct superficial muscle compartments and should be
# far more separable on 8-channel surface EMG.
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

# ─── Signal / feature constants ───────────────────────────────────────────────
FS          = 985
WIN_SAMPLES = int(200 * FS / 1000)   # 197 samples  @ 200 ms
INC_SAMPLES = int(100 * FS / 1000)   #  98 samples  @ 100 ms
N_CHANNELS  = 8
N_FEATURES  = 8                       # per channel
N_CLASSES   = len(CLASSES)            # 4
INPUT_DIM   = N_CHANNELS * N_FEATURES # 64

# ─── Quantisation constants ───────────────────────────────────────────────────
INT8_MAX  = 127
INT8_MIN  = -128
CALIB_PCT = 99.9   # percentile for robust scale computation
REQUANT_SHIFT = 31 # fixed-point precision of the requantisation multiplier

# ─── Architecture search space ────────────────────────────────────────────────
# Fixed to the single selected architecture. The prior 5-way sweep showed all
# candidates were statistically indistinguishable (spread 1.3 pp vs. ±5.9 pp
# 95% CI, unstable rank order across runs, ~13% power to detect 1 pp) — the
# accuracy ceiling was set by gesture separability, not model capacity. 64→16→4
# is the defensible pick: smallest model retaining nonlinear capacity, tied
# with every larger candidate. Re-add widths here if you want the full sweep
# on this new gesture set instead of just this one architecture.
HIDDEN_WIDTHS = [16]

# ─── Preprocessing filter configuration ───────────────────────────────────────
# Standard sEMG conditioning: removes DC offset / motion artifact below the
# bandpass low edge and high-frequency noise above the high edge, then a notch
# at the mains frequency. FORS-EMG was collected in Bangladesh → 50 Hz mains;
# change to 60.0 if your recording setup used 60 Hz power.
APPLY_FILTER    = True
BANDPASS_HZ     = (20.0, 450.0)
MAINS_NOTCH_HZ  = 50.0
NOTCH_Q         = 30.0

# ─── Training hyper-parameters ────────────────────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.2
BATCH_SIZE   = 128
MAX_EPOCHS   = 200
PATIENCE     = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    subject.  This is your original load_fors_emg() function, unchanged in
    logic, extended only to accept subject_id as an argument so the benchmark
    can call it for each of the 19 subjects.

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
        return recordings, recording_meta   # subject folder absent → skip silently

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
                    # Normalise to (8000, 8) regardless of storage orientation
                    if val.shape == (8000, 8):
                        data = val.astype(np.float64)
                    elif val.shape == (8, 8000):
                        data = val.T.astype(np.float64)
                    else:
                        continue   # unexpected shape — skip

                    recordings.append(data)
                    recording_meta.append({
                        "label":       class_idx,
                        "subject":     subject_id,
                        "orientation": orientation,
                        "trial":       trial,
                        "rec_id":      rec_id,
                    })
                    rec_id += 1
                    break   # one valid array per .mat file is enough

    return recordings, recording_meta


def load_all_subjects(base_dir: str = BASE_DIR) -> list:
    """
    Calls _load_single_subject() for each of the 19 subjects and packages
    the results into the per-subject structure expected by LOSO-CV.

    Returns
    -------
    subjects : list of dicts, each with:
        'subject_id'  : int
        'recordings'  : list of (8000, 8) float64 arrays   [raw signal]
        'labels'      : list of int   [class index 0-3, one per recording]
        'meta'        : list of dict  [full metadata per recording]
    """
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
    """Design the bandpass + notch filter coefficients once (cheap, reused)."""
    nyq = fs / 2.0
    low  = band[0] / nyq
    high = min(band[1], nyq - 1.0) / nyq
    b_bp, a_bp = butter(4, [low, high], btype="bandpass")
    b_n,  a_n  = iirnotch(notch_freq, notch_q, fs)
    return (b_bp, a_bp), (b_n, a_n)


def apply_preprocessing_filter(sig_channels_first, filt_coeffs):
    """
    sig_channels_first : (n_channels, n_samples)
    Applies zero-phase (filtfilt) 4th-order Butterworth bandpass, then a
    zero-phase notch at the mains frequency, independently per channel.
    """
    (b_bp, a_bp), (b_n, a_n) = filt_coeffs
    sig = filtfilt(b_bp, a_bp, sig_channels_first, axis=-1)
    sig = filtfilt(b_n,  a_n,  sig,                axis=-1)
    return sig


def filter_all_subjects(subjects, apply_filter=APPLY_FILTER):
    """
    Applies the bandpass+notch filter ONCE per recording, in place, before any
    windowing or LOSO splitting happens. Filtering once here (rather than
    inside the LOSO loop) avoids redundant recomputation — each recording is
    reused as training data in 18 of the 19 folds.
    """
    if not apply_filter:
        print("  Preprocessing filter: DISABLED (APPLY_FILTER=False)")
        return subjects

    filt_coeffs = _design_filters()
    print(f"  Preprocessing filter: {BANDPASS_HZ[0]:.0f}-{BANDPASS_HZ[1]:.0f} Hz "
          f"bandpass + {MAINS_NOTCH_HZ:.0f} Hz notch")

    for s in tqdm(subjects, desc="Filtering signals"):
        filtered = []
        for rec in s["recordings"]:
            # rec is (8000, 8) per your loader's convention
            sig_cf = rec.T                                   # → (8, 8000)
            sig_filtered = apply_preprocessing_filter(sig_cf, filt_coeffs)
            filtered.append(sig_filtered.T.astype(np.float64))   # back to (8000, 8)
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

    mav  = xabs.mean(axis=2)                              # (N, C)
    rms  = np.sqrt((x**2).mean(axis=2))
    wl   = np.abs(dx).sum(axis=2)

    # ZC with 1% RMS deadband
    thr  = rms * 0.01
    sx   = np.sign(x)
    zc   = ((sx[:,:,1:] != sx[:,:,:-1]) &
            ((xabs[:,:,:-1] >= thr[:,:,None]) |
             (xabs[:,:,1:]  >= thr[:,:,None]))).sum(axis=2)

    # SSC
    ssc  = ((dx[:,:,1:] * dx[:,:,:-1]) < 0).sum(axis=2)

    var  = x.var(axis=2)
    # WAMP: threshold relative to signal scale (2% of window RMS), NOT a fixed
    # absolute value. A fixed threshold (e.g. 0.002) silently saturates or
    # collapses to zero if the .mat files store raw ADC codes rather than
    # calibrated microvolts — this dataset's units were never verified, so a
    # scale-relative threshold is the safe choice. rms is (N,C); broadcast
    # against dx (N,C,L-1) via the trailing None axis.
    wamp_thr = (rms * 0.02)[:, :, None]          # (N, C, 1)
    wamp = (np.abs(dx) > wamp_thr).sum(axis=2)   # (N, C)
    iemg = xabs.sum(axis=2)

    feat = np.stack([mav, rms, wl, zc, ssc, var, wamp, iemg], axis=2)
    return feat.reshape(N, C * 8).astype(np.float32)


def windows_from_recordings(recordings: list, labels: list) -> tuple:
    """
    Slide fixed-length windows over each raw recording without crossing the
    recording boundary.  This is the correct windowing approach for LOSO: you
    call this separately for each split so windows never straddle train/test.

    Parameters
    ----------
    recordings : list of (8000, 8) float64 arrays   — your loader's output
    labels     : list of int   — one label per recording

    Returns
    -------
    windows : (N_windows, N_channels, WIN_SAMPLES)  float32
    y       : (N_windows,)  int64
    """
    all_win, all_lbl = [], []
    for rec, lbl in zip(recordings, labels):
        # rec is (8000, 8) — your loader guarantees this
        sig    = rec.T                                      # → (8, 8000)
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

def compute_scale(arr):
    """Symmetric per-tensor scale from 99.9-percentile of |arr|."""
    v = np.percentile(np.abs(arr.ravel()), CALIB_PCT)
    return float(v) / INT8_MAX if v > 0 else 1.0

def quantize_to_int8(arr, scale):
    return np.clip(np.round(arr / scale), INT8_MIN, INT8_MAX).astype(np.int8)

def ptq_and_infer(model_fp32, X_calib, X_test):
    """
    Post-training INT8 quantization and integer-arithmetic inference.
    Calibration uses X_calib (training features, already normalised).
    """
    model_fp32.eval()

    # Collect linear layers in order
    linears = [(n, m) for n, m in model_fp32.named_modules()
               if isinstance(m, nn.Linear)]

    # Input scale
    sx = compute_scale(X_calib)

    # Per-layer calibration (FP32 forward on calibration set)
    x_fp = X_calib.copy()
    layer_info = []
    for i, (name, lin) in enumerate(linears):
        W = lin.weight.detach().numpy()
        b = lin.bias.detach().numpy()
        sw = compute_scale(W)

        z = x_fp @ W.T + b
        is_last = (i == len(linears) - 1)
        if is_last:
            sy = compute_scale(z)
        else:
            sy = compute_scale(np.maximum(0, z))
            x_fp = np.maximum(0, z)

        # Requantisation multiplier M = sw*sx/sy, expressed as (M0, shift) so
        # that y = (z*M0) >> shift reproduces round(M*z) with integer ops only.
        # A FIXED non-negative shift is used deliberately: deriving the shift
        # from ceil(-log2(M)) breaks whenever M > 1, because the shift goes
        # negative and a negative right-shift is undefined. A fixed shift is
        # valid for any M > 0 and is what hardware would hard-wire anyway.
        M = (sw * sx) / sy
        n_shift = REQUANT_SHIFT
        M0 = int(np.round(M * (1 << REQUANT_SHIFT))) if M > 0 else 0

        W_q  = quantize_to_int8(W, sw).astype(np.int32)
        b_q  = np.round(b / (sw * sx)).astype(np.int32)
        layer_info.append((W_q, b_q, sy, M0, n_shift, is_last))
        sx = sy

    # Integer-arithmetic inference on test set
    x_q = np.clip(np.round(X_test / compute_scale(X_calib)),
                  INT8_MIN, INT8_MAX).astype(np.int32)

    for W_q, b_q, sy, M0, n_shift, is_last in layer_info:
        z = (x_q.astype(np.int64) @ W_q.T.astype(np.int64)
             + b_q[None, :].astype(np.int64))
        if is_last:
            # argmax on the raw accumulator — requantisation is a monotone
            # positive scaling and cannot change which logit is largest.
            return np.argmax(z, axis=1)
        # Requantise (round-to-nearest, int64) + ReLU
        z_req = ((z * np.int64(M0)) + (np.int64(1) << (n_shift - 1))) >> n_shift
        x_q   = np.maximum(0, np.clip(z_req, INT8_MIN, INT8_MAX)).astype(np.int32)

    return np.argmax(z, axis=1)   # fallback (linear model)


# ─── MLP ──────────────────────────────────────────────────────────────────────

def build_mlp(hidden):
    if hidden is None:
        return nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
    return nn.Sequential(
        nn.Linear(INPUT_DIM, hidden),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(hidden, N_CLASSES),
    )

def param_bytes(hidden):
    if hidden is None:
        return INPUT_DIM * N_CLASSES + N_CLASSES * 4
    return (INPUT_DIM*hidden + hidden*4 + hidden*N_CLASSES + N_CLASSES*4)

def num_params(hidden):
    """Real trainable parameter count (weights + biases), for overfitting
    diagnostics — distinct from param_bytes(), which accounts INT8 weight
    bytes + INT32 bias bytes for the hardware memory budget."""
    if hidden is None:
        return INPUT_DIM * N_CLASSES + N_CLASSES
    return (INPUT_DIM*hidden + hidden) + (hidden*N_CLASSES + N_CLASSES)

def num_macs(hidden):
    if hidden is None:
        return INPUT_DIM * N_CLASSES
    return INPUT_DIM*hidden + hidden*N_CLASSES

def arch_label(hidden):
    return f"64→{N_CLASSES}" if hidden is None else f"64→{hidden}→{N_CLASSES}"


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(X_tr, y_tr, X_val, y_val, hidden):
    model = build_mlp(hidden).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv), yv).item()
        if vloss < best_loss - 1e-4:
            best_loss  = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model.cpu()


# ─── LOSO-CV ──────────────────────────────────────────────────────────────────

def _val_split_by_trial(recordings, labels, meta, val_trial_id=5):
    """
    Split one subject's recordings into train and val using trial index.

    Val = all recordings from the last trial number (default: trial 5) across
    every orientation and class.  Train = everything else.
    This ensures the val windows come from genuinely unseen contractions rather
    than time-adjacent segments of the same contraction — which is what a
    last-10%-of-windows split produces and what caused the 63% plateau.

    Parameters
    ----------
    recordings, labels, meta : outputs of load_all_subjects for one subject
    val_trial_id : int  — trial number held out for validation (1-indexed)

    Returns
    -------
    Four lists: train_recs, train_lbls, val_recs, val_lbls
    """
    train_recs, train_lbls = [], []
    val_recs,   val_lbls   = [], []

    for rec, lbl, m in zip(recordings, labels, meta):
        if m["trial"] == val_trial_id:
            val_recs.append(rec);   val_lbls.append(lbl)
        else:
            train_recs.append(rec); train_lbls.append(lbl)

    return train_recs, train_lbls, val_recs, val_lbls


def loso_cv(subjects, hidden):
    """
    Leave-One-Subject-Out cross-validation.

    Key fixes vs. the initial version:
    1. Val split is by trial index, not by window index.  Trial 5 (the last
       trial per gesture per orientation) is reserved as val for every subject
       in the training pool.  This gives ~16 val recordings per training
       subject (4 classes × 3 orientations × 1 trial) with zero window overlap
       with the remaining training trials.
    2. StandardScaler is fit ONLY on the training recordings (trials 1-4),
       then applied to both val and test — no statistics from val or test
       contaminate the scaler.
    3. PTQ calibration uses only training windows (not val), consistent with
       how the scale would be derived on a real device before deployment.
    """
    fp32_accs, int8_accs = [], []
    n = len(subjects)

    for test_idx in tqdm(range(n),
                         desc=f"  {arch_label(hidden):18s}",
                         leave=False):
        # ── Outer LOSO split ─────────────────────────────────────────────────
        test_s   = subjects[test_idx]
        train_ss = [s for i, s in enumerate(subjects) if i != test_idx]

        # Test set: ALL recordings from the held-out subject (windowed after split)
        X_te_win, y_te = windows_from_recordings(
            test_s['recordings'], test_s['labels'])

        # ── Inner trial-based val split across all training subjects ──────────
        # For each training subject, trial 5 → val, trials 1-4 → train.
        # Using the last trial (highest index) as val is deterministic and
        # avoids any form of look-ahead selection bias.
        all_train_recs, all_train_lbls = [], []
        all_val_recs,   all_val_lbls   = [], []

        for s in train_ss:
            tr_r, tr_l, va_r, va_l = _val_split_by_trial(
                s['recordings'], s['labels'], s['meta'], val_trial_id=5)
            all_train_recs.extend(tr_r); all_train_lbls.extend(tr_l)
            all_val_recs.extend(va_r);   all_val_lbls.extend(va_l)

        # Window AFTER every split boundary is fixed
        X_tr_win, y_tr_all = windows_from_recordings(all_train_recs, all_train_lbls)
        X_va_win, y_va     = windows_from_recordings(all_val_recs,   all_val_lbls)

        # ── Feature extraction ───────────────────────────────────────────────
        X_tr_feat = extract_features_from_windows(X_tr_win)
        X_va_feat = extract_features_from_windows(X_va_win)
        X_te_feat = extract_features_from_windows(X_te_win)

        # ── Normalisation: scaler fit on TRAIN only, applied to val + test ────
        sc      = StandardScaler()
        X_tr_n  = sc.fit_transform(X_tr_feat)
        X_va_n  = sc.transform(X_va_feat)
        X_te_n  = sc.transform(X_te_feat)

        # ── FP32 training with proper early stopping on held-out val trials ───
        model = train_model(X_tr_n, y_tr_all, X_va_n, y_va, hidden)

        # ── FP32 evaluation on test subject ───────────────────────────────────
        model.eval()
        with torch.no_grad():
            preds_fp = model(
                torch.tensor(X_te_n, dtype=torch.float32)
            ).argmax(1).numpy()
        fp32_accs.append(accuracy_score(y_te, preds_fp))

        # ── INT8 PTQ: calibrate on training windows only, infer on test ───────
        preds_q8 = ptq_and_infer(model, X_tr_n, X_te_n)
        int8_accs.append(accuracy_score(y_te, preds_q8))

    return fp32_accs, int8_accs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  FORS-EMG TinyML Architecture Benchmark")
    print("=" * 65)

    # 1. Load raw recordings per subject (using your exact loader)
    subjects = load_all_subjects(BASE_DIR)
    total_recs = sum(len(s['recordings']) for s in subjects)
    print(f"\n  Subjects loaded  : {len(subjects)}")
    print(f"  Total recordings : {total_recs}")
    print(f"  Target gestures  : {', '.join(GESTURE_NAMES[c] for c in CLASSES)}")
    print(f"  Orientations     : {', '.join(FOREARM_ORIENTATIONS)}")
    print(f"  Window           : {WIN_SAMPLES} samples ({200} ms), "
          f"increment {INC_SAMPLES} samples ({100} ms)")
    print(f"  Input dim        : {INPUT_DIM}  ({N_CHANNELS} ch × {N_FEATURES} feats)")
    print(f"  Device           : {DEVICE}")

    # ── Raw signal scale diagnostic ─────────────────────────────────────────
    # Confirms whether feature thresholds (e.g. WAMP, ZC deadband) are sensibly
    # calibrated to this dataset's actual units. If |max| is O(1e2-1e3), the
    # .mat files likely store raw ADC codes rather than calibrated microvolts.
    sample_rec = subjects[0]['recordings'][0]
    print(f"\n  Raw signal scale check (Subject{subjects[0]['subject_id']}, "
          f"first recording):")
    print(f"    shape={sample_rec.shape}  min={sample_rec.min():.4f}  "
          f"max={sample_rec.max():.4f}  mean={sample_rec.mean():.4f}  "
          f"std={sample_rec.std():.4f}")

    # 2. Bandpass + notch filtering (once, before any windowing)
    subjects = filter_all_subjects(subjects, apply_filter=APPLY_FILTER)

    # Class balance across all subjects
    all_labels = [lbl for s in subjects for lbl in s['labels']]
    print(f"\n  Class balance (recordings):")
    for c in CLASSES:
        cnt = all_labels.count(c)
        print(f"    [{c}] {GESTURE_NAMES[c]:14s}: {cnt:4d}  "
              f"({cnt/len(all_labels)*100:.1f}%)")

    # 2. Budget table
    print(f"\n{'─'*65}")
    print("  Architecture budget (INT8 weights + INT32 biases)")
    print(f"{'─'*65}")
    brows = [[arch_label(h), f"{param_bytes(h):,}", f"{num_macs(h):,}",
              "✓" if param_bytes(h) <= 10_000 else "✗ OVER"]
             for h in HIDDEN_WIDTHS]
    print(tabulate(brows,
                   headers=["Architecture","Param bytes","MACs","≤10 KB?"],
                   tablefmt="rounded_outline"))

    # ── Overfitting-risk diagnostic ─────────────────────────────────────────
    # Estimates the samples-per-parameter ratio each architecture will see
    # during LOSO training. This uses actual windowed counts from this run
    # (not an assumption) — computed once here, before the (slow) LOSO sweep,
    # so you see it regardless of how long training takes.
    total_windows = 0
    for s in subjects:
        w, _ = windows_from_recordings(s['recordings'], s['labels'])
        total_windows += len(w)
    avg_test_fold      = total_windows / len(subjects)
    avg_train_pool     = avg_test_fold * (len(subjects) - 1)
    avg_train_post_val = avg_train_pool * (1 - 1.0 / N_TRIALS)  # 1 of N_TRIALS held out as val

    print(f"\n{'─'*65}")
    print("  Overfitting-risk check (real parameter count vs. training windows)")
    print(f"{'─'*65}")
    print(f"  Total windows (all subjects) : {total_windows:,.0f}")
    print(f"  Avg. training windows/fold   : {avg_train_post_val:,.0f}  "
          f"(after held-out val trial)")
    orows = [[arch_label(h), f"{num_params(h):,}",
              f"{avg_train_post_val / num_params(h):,.1f}x"]
             for h in HIDDEN_WIDTHS]
    print(tabulate(orows,
                   headers=["Architecture","Real params","Samples per param"],
                   tablefmt="rounded_outline"))
    print("  Rule of thumb: ≥5-10x samples per parameter is comfortable for a")
    print("  regularised shallow MLP. Below ~3x, treat results with caution.")

    # 3. LOSO-CV sweep
    print(f"\n{'─'*65}")
    print(f"  LOSO-CV ({len(subjects)}-fold) — one fold per subject")
    print(f"{'─'*65}\n")
    results = {}
    for h in HIDDEN_WIDTHS:
        fp32, int8 = loso_cv(subjects, h)
        label = arch_label(h)
        results[label] = dict(
            fp32_mean=np.mean(fp32), fp32_std=np.std(fp32),
            int8_mean=np.mean(int8), int8_std=np.std(int8),
            q_drop=np.mean(fp32)-np.mean(int8),
            pb=param_bytes(h), macs=num_macs(h),
            fp32_folds=fp32, int8_folds=int8
        )

    # 4. Results
    print(f"\n{'─'*65}")
    print("  Results  (mean ± std over subjects)")
    print(f"{'─'*65}")
    rows = [[lbl,
             f"{r['fp32_mean']*100:.2f}±{r['fp32_std']*100:.2f}",
             f"{r['int8_mean']*100:.2f}±{r['int8_std']*100:.2f}",
             f"{r['q_drop']*100:.2f}",
             f"{r['pb']:,}",
             f"{r['macs']:,}"]
            for lbl, r in results.items()]
    print(tabulate(rows,
                   headers=["Architecture","FP32 %","INT8 %","Q-drop pp",
                             "Bytes","MACs"],
                   tablefmt="rounded_outline"))

    best = max(results, key=lambda k: results[k]['int8_mean'])
    r = results[best]
    print(f"\n  ★  Recommended: {best}")
    print(f"     INT8 accuracy : {r['int8_mean']*100:.2f}% ± {r['int8_std']*100:.2f}%")
    print(f"     Q-drop        : {r['q_drop']*100:.2f} pp")
    print(f"     Param bytes   : {r['pb']:,} / 10,000")
    print(f"     MACs/inference: {r['macs']:,}")

    # 5. Per-subject breakdown for best architecture
    print(f"\n  Per-subject INT8 accuracy ({best}):")
    frows = [[f"S{i+1:02d}", f"{f:.2%}", f"{q:.2%}", f"{(f-q)*100:.2f}pp"]
             for i,(f,q) in enumerate(zip(r['fp32_folds'],r['int8_folds']))]
    print(tabulate(frows,
                   headers=["Subject","FP32","INT8","Q-drop"],
                   tablefmt="simple"))

    # 6. Diagnosis — rerun best arch on all subjects to get per-class confusion
    print(f"\n  Per-class accuracy (INT8, {best}, pooled over all LOSO folds):")
    best_h  = results[best].get('hidden', None)
    # hidden is not stored in results dict yet — recover it
    best_h  = HIDDEN_WIDTHS[list(results.keys()).index(best)]

    all_true, all_pred = [], []
    for test_idx in range(len(subjects)):
        test_s   = subjects[test_idx]
        train_ss = [s for i, s in enumerate(subjects) if i != test_idx]
        X_te_win, y_te = windows_from_recordings(test_s['recordings'], test_s['labels'])

        all_tr_recs, all_tr_lbls = [], []
        all_va_recs, all_va_lbls = [], []
        for s in train_ss:
            tr_r,tr_l,va_r,va_l = _val_split_by_trial(
                s['recordings'], s['labels'], s['meta'])
            all_tr_recs.extend(tr_r); all_tr_lbls.extend(tr_l)
            all_va_recs.extend(va_r); all_va_lbls.extend(va_l)

        X_tr_w, y_tr_w = windows_from_recordings(all_tr_recs, all_tr_lbls)
        X_va_w, y_va_w = windows_from_recordings(all_va_recs, all_va_lbls)

        sc2 = StandardScaler()
        X_tr_n2 = sc2.fit_transform(extract_features_from_windows(X_tr_w))
        X_va_n2 = sc2.transform(extract_features_from_windows(X_va_w))
        X_te_n2 = sc2.transform(extract_features_from_windows(X_te_win))

        m2 = train_model(X_tr_n2, y_tr_w, X_va_n2, y_va_w, best_h)
        pq = ptq_and_infer(m2, X_tr_n2, X_te_n2)
        all_true.extend(y_te.tolist())
        all_pred.extend(pq.tolist())

    from sklearn.metrics import confusion_matrix, classification_report
    all_true = np.array(all_true); all_pred = np.array(all_pred)
    print(classification_report(
        all_true, all_pred,
        target_names=[GESTURE_NAMES[c] for c in CLASSES],
        digits=3))

    cm = confusion_matrix(all_true, all_pred)
    print("  Confusion matrix (rows=true, cols=predicted):")
    cm_rows = [[GESTURE_NAMES[c]] + list(cm[i]) for i,c in enumerate(CLASSES)]
    print(tabulate(cm_rows,
                   headers=["True \\ Pred"] + [GESTURE_NAMES[c] for c in CLASSES],
                   tablefmt="simple"))

    # 7. CSV export
    with open("fors_emg_results.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Architecture","FP32_mean","FP32_std",
                    "INT8_mean","INT8_std","Qdrop_pp","Bytes","MACs"])
        for lbl, r in results.items():
            w.writerow([lbl,
                        round(r['fp32_mean'],6), round(r['fp32_std'],6),
                        round(r['int8_mean'],6), round(r['int8_std'],6),
                        round(r['q_drop']*100,4),
                        r['pb'], r['macs']])
    print(f"\n  Results saved → fors_emg_results.csv")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
