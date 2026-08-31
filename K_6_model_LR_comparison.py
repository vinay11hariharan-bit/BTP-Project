"""
FORS-EMG TinyML Benchmark — Best 6-Gesture Subset (Validation Run)
=====================================================================
Validates the "best k=6" gesture subset identified by analysing the actual
12-class confusion matrix (from fors_emg_benchmark_12class.py's real run) —
not a fresh search, an empirical estimate that this run either confirms or
corrects.

Selected gestures: Hand Close, Hand Open, Peace, Radial Deviation,
Right Angle, Wrist Flexion — chosen by ranking all C(12,6)=924 possible
6-gesture subsets by worst-pair confusability (the most-confused pair within
the subset), computed directly from the real trained-classifier confusion
matrices (MLP and Logistic Regression, averaged). This subset's worst pair
sits at 5.6% confusability, vs. 13.5% for the previously-tested 4-gesture set
{Hand Close, Hand Open, Wrist Flexion, Wrist Extension} — so despite having
50% more classes, it should be LESS confused on its hardest pair, not more.

What this run is actually testing
------------------------------------
The confusion-matrix analysis gave a "floor" accuracy of 53.5% (a genuine
lower bound: what the 12-way classifier got right when restricted to these
6 classes, treating any prediction that fell on an excluded gesture as
wrong) and a calibrated central estimate around 66%, extrapolated from the
ONE data point available (the 4-gesture set's floor undershot its real
retrained accuracy by 19.3 pp). This run replaces that extrapolation with
an actual number.

Pipeline (identical to fors_emg_benchmark_12class.py)
--------------------------------------------------------
* Logistic Regression trains alongside the MLP in the SAME LOSO fold —
  identical splits, features, and scaler, so the comparison is exact.
* 20-450 Hz bandpass + 50 Hz mains notch, applied once per recording.
* 200 ms windows / 100 ms increment, cut AFTER every split boundary.
* LOSO across subjects; trial-level (not window-level) validation split.
* StandardScaler + INT8 PTQ calibration scale, both fit on training data only.
* Architecture fixed at 64→16→6 (same 16-unit hidden layer as every other
  run in this series, so any accuracy change reflects gesture separability,
  not a confounded capacity change). Budget: 1,208 B / 1,120 MACs — trivial.

Difference from fors_emg_benchmark_12class.py
--------------------------------------------------
Gestures are a FIXED list of 6 (not auto-discovered), since we deliberately
want exactly this subset — but the script still verifies these 6 file
prefixes actually exist somewhere in your archive before running, so a
naming mismatch fails loudly instead of silently loading zero recordings.

Configuration
-------------
  Set BASE_DIR below to the root folder that contains Subject1/, Subject2/, …
  Then run:
      python fors_emg_benchmark_6class.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy import stats
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

# ─── Dataset configuration ────────────────────────────────────────────────────
# ↓ Edit this to match your local path (same archive fors_emg_benchmark.py uses)
BASE_DIR = r"/Users/vinayhariharan/Projects/BTP/NN_model/archive/FORS-EMG Dataset/FORS-EMG Dataset/FORS-EMG"

FOREARM_ORIENTATIONS = ["Rest", "Supination", "Pronation"]
N_TRIALS   = 5
N_SUBJECTS = 19   # subjects 1–19

# The best-k=6 subset by worst-pair confusability, computed from the actual
# 12-class confusion matrices (see module docstring). File prefixes below
# are the exact spellings confirmed by the 12-gesture run's auto-discovery
# on your real archive — not guessed.
SELECTED_GESTURE_PREFIXES = [
    "Hand_Close", "Hand_Open", "Peace",
    "Radial_Deviation", "Right_Angle", "Wrist_Flexion",
]


def verify_gestures_exist(prefixes, base_dir=BASE_DIR,
                          orientations=FOREARM_ORIENTATIONS,
                          n_subjects=N_SUBJECTS):
    """
    Confirms every requested gesture prefix actually has at least one .mat
    file somewhere in the archive, BEFORE running anything else. A fixed
    hardcoded gesture list (unlike auto-discovery) fails silently if a name
    is wrong — this check turns that into a loud, specific error instead of
    a run that quietly loads zero recordings for a mistyped gesture.
    """
    found_prefixes = set()
    for sid in range(1, n_subjects + 1):
        for ori in orientations:
            d = os.path.join(base_dir, f"Subject{sid}", ori)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(".mat") and "-" in fn[:-4]:
                    found_prefixes.add(fn[:-4].rsplit("-", 1)[0])

    missing = [p for p in prefixes if p not in found_prefixes]
    if missing:
        print(f"\n  ERROR: {len(missing)} requested gesture(s) not found "
              f"anywhere in the archive:")
        for m in missing:
            print(f"    ✗ {m}")
        print(f"\n  Gestures that WERE found in the archive:")
        for f in sorted(found_prefixes):
            flag = " (requested)" if f in prefixes else ""
            print(f"    {f}{flag}")
        print(f"\n  Fix SELECTED_GESTURE_PREFIXES at the top of this file to "
              f"match the exact names above, then re-run.\n")
        sys.exit(1)
    print(f"  Verified all {len(prefixes)} requested gestures exist in the archive:")
    for p in prefixes:
        print(f"    ✓ {p}")


verify_gestures_exist(SELECTED_GESTURE_PREFIXES)

CLASSES          = list(range(len(SELECTED_GESTURE_PREFIXES)))
GESTURE_FILE_MAP = {i: name for i, name in enumerate(SELECTED_GESTURE_PREFIXES)}
GESTURE_NAMES    = {i: name.replace("_", " ") for i, name in enumerate(SELECTED_GESTURE_PREFIXES)}

# ─── Signal / feature constants ───────────────────────────────────────────────
FS          = 985
WIN_SAMPLES = int(200 * FS / 1000)   # 197 samples  @ 200 ms
INC_SAMPLES = int(100 * FS / 1000)   #  98 samples  @ 100 ms
N_CHANNELS  = 8
N_FEATURES  = 8                       # per channel
N_CLASSES   = len(CLASSES)            # 12
INPUT_DIM   = N_CHANNELS * N_FEATURES # 64

# ─── Quantisation constants ───────────────────────────────────────────────────
INT8_MAX  = 127
INT8_MIN  = -128
CALIB_PCT = 99.9   # percentile for robust scale computation
REQUANT_SHIFT = 31 # fixed-point precision of the requantisation multiplier

# ─── Architecture ──────────────────────────────────────────────────────────────
# Fixed at the same 16-unit hidden layer used for the 4-gesture run, on
# purpose — see module docstring. Budget at N_CLASSES=6: 1,208 B /
# 1,120 MACs, still comfortably under 10 KB.
HIDDEN_WIDTHS = [16]

# ─── Preprocessing filter configuration ───────────────────────────────────────
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

LR_MAX_ITER  = 300   # Logistic Regression solver iterations (lbfgs, multinomial)

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
    subject. Identical logic to fors_emg_benchmark.py's loader.

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
    """Calls _load_single_subject() for every subject 1..N_SUBJECTS."""
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
    nyq = fs / 2.0
    low  = band[0] / nyq
    high = min(band[1], nyq - 1.0) / nyq
    b_bp, a_bp = butter(4, [low, high], btype="bandpass")
    b_n,  a_n  = iirnotch(notch_freq, notch_q, fs)
    return (b_bp, a_bp), (b_n, a_n)


def apply_preprocessing_filter(sig_channels_first, filt_coeffs):
    (b_bp, a_bp), (b_n, a_n) = filt_coeffs
    sig = filtfilt(b_bp, a_bp, sig_channels_first, axis=-1)
    sig = filtfilt(b_n,  a_n,  sig,                axis=-1)
    return sig


def filter_all_subjects(subjects, apply_filter=APPLY_FILTER):
    """Filters ONCE per recording, before any windowing or LOSO splitting."""
    if not apply_filter:
        print("  Preprocessing filter: DISABLED (APPLY_FILTER=False)")
        return subjects

    filt_coeffs = _design_filters()
    print(f"  Preprocessing filter: {BANDPASS_HZ[0]:.0f}-{BANDPASS_HZ[1]:.0f} Hz "
          f"bandpass + {MAINS_NOTCH_HZ:.0f} Hz notch")

    for s in tqdm(subjects, desc="Filtering signals"):
        filtered = []
        for rec in s["recordings"]:
            sig_cf = rec.T
            sig_filtered = apply_preprocessing_filter(sig_cf, filt_coeffs)
            filtered.append(sig_filtered.T.astype(np.float64))
        s["recordings"] = filtered
    return subjects


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features_from_windows(windows):
    """windows : (N, C, L) → features : (N, 64). Identical to fors_emg_benchmark.py."""
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


def windows_from_recordings(recordings: list, labels: list) -> tuple:
    """Slides windows over each recording without crossing recording boundaries."""
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

def compute_scale(arr):
    v = np.percentile(np.abs(arr.ravel()), CALIB_PCT)
    return float(v) / INT8_MAX if v > 0 else 1.0

def quantize_to_int8(arr, scale):
    return np.clip(np.round(arr / scale), INT8_MIN, INT8_MAX).astype(np.int8)

def ptq_and_infer(model_fp32, X_calib, X_test):
    """Post-training INT8 quantisation + integer-arithmetic inference."""
    model_fp32.eval()
    linears = [(n, m) for n, m in model_fp32.named_modules()
               if isinstance(m, nn.Linear)]

    sx = compute_scale(X_calib)
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

        # Fixed non-negative shift — valid for any M>0 (see fors_emg_benchmark.py
        # for why deriving the shift from ceil(-log2(M)) breaks when M>1).
        M = (sw * sx) / sy
        n_shift = REQUANT_SHIFT
        M0 = int(np.round(M * (1 << REQUANT_SHIFT))) if M > 0 else 0

        W_q  = quantize_to_int8(W, sw).astype(np.int32)
        b_q  = np.round(b / (sw * sx)).astype(np.int32)
        layer_info.append((W_q, b_q, sy, M0, n_shift, is_last))
        sx = sy

    x_q = np.clip(np.round(X_test / compute_scale(X_calib)),
                  INT8_MIN, INT8_MAX).astype(np.int32)

    for W_q, b_q, sy, M0, n_shift, is_last in layer_info:
        z = (x_q.astype(np.int64) @ W_q.T.astype(np.int64)
             + b_q[None, :].astype(np.int64))
        if is_last:
            return np.argmax(z, axis=1)
        z_req = ((z * np.int64(M0)) + (np.int64(1) << (n_shift - 1))) >> n_shift
        x_q   = np.maximum(0, np.clip(z_req, INT8_MIN, INT8_MAX)).astype(np.int32)

    return np.argmax(z, axis=1)


def quantize_linear_predict(W, b, X_calib, X_test):
    """
    Integer-only inference for a linear model (logits = W x + b), e.g.
    Logistic Regression's coef_/intercept_. Symmetric per-tensor INT8
    weights, INT32-domain bias — identical scheme to the MLP's PTQ, just
    without a hidden layer or requantisation step (a single linear layer's
    argmax is invariant to the accumulator's scale, so none is needed).
    """
    sx, sw = compute_scale(X_calib), compute_scale(W)
    Wq = np.clip(np.round(W / sw), INT8_MIN, INT8_MAX).astype(np.int64)
    bq = np.round(np.asarray(b) / (sw * sx)).astype(np.int64)
    xq = np.clip(np.round(X_test / sx), INT8_MIN, INT8_MAX).astype(np.int64)
    return np.argmax(xq @ Wq.T + bq[None, :], axis=1)


def linear_param_bytes():
    """INT8 weights + INT32 bias for a single 64->N_CLASSES linear layer."""
    return INPUT_DIM * N_CLASSES + N_CLASSES * 4

def linear_num_params():
    return INPUT_DIM * N_CLASSES + N_CLASSES

def linear_macs():
    return INPUT_DIM * N_CLASSES


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
    """Trial 5 -> val, trials 1-4 -> train. Same rationale as fors_emg_benchmark.py:
    a window-level split puts near-duplicate overlapping windows on both sides."""
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
    Leave-One-Subject-Out cross-validation for BOTH the MLP and a Logistic
    Regression classifier, trained within the SAME fold on the SAME
    train/val/test split, features, and scaler — so the comparison is exact,
    not just "close enough." LogReg reuses X_tr_n/y_tr_all/X_te_n/y_te that
    are already computed for the MLP, so this is nearly free to add; it does
    not repeat feature extraction, windowing, or splitting.

    Returns a dict keyed by model name, each with:
        fp32, int8   : list of per-subject accuracies (len == n_subjects)
        y_true, y_pred : pooled arrays across all folds (INT8 predictions)
    """
    mlp_fp32, mlp_int8, mlp_true, mlp_pred = [], [], [], []
    lr_fp32,  lr_int8,  lr_true,  lr_pred  = [], [], [], []
    n = len(subjects)

    for test_idx in tqdm(range(n),
                         desc=f"  {arch_label(hidden):18s}",
                         leave=False):
        test_s   = subjects[test_idx]
        train_ss = [s for i, s in enumerate(subjects) if i != test_idx]

        X_te_win, y_te = windows_from_recordings(
            test_s['recordings'], test_s['labels'])

        all_train_recs, all_train_lbls = [], []
        all_val_recs,   all_val_lbls   = [], []
        for s in train_ss:
            tr_r, tr_l, va_r, va_l = _val_split_by_trial(
                s['recordings'], s['labels'], s['meta'], val_trial_id=5)
            all_train_recs.extend(tr_r); all_train_lbls.extend(tr_l)
            all_val_recs.extend(va_r);   all_val_lbls.extend(va_l)

        X_tr_win, y_tr_all = windows_from_recordings(all_train_recs, all_train_lbls)
        X_va_win, y_va     = windows_from_recordings(all_val_recs,   all_val_lbls)

        X_tr_feat = extract_features_from_windows(X_tr_win)
        X_va_feat = extract_features_from_windows(X_va_win)
        X_te_feat = extract_features_from_windows(X_te_win)

        sc      = StandardScaler()
        X_tr_n  = sc.fit_transform(X_tr_feat)
        X_va_n  = sc.transform(X_va_feat)
        X_te_n  = sc.transform(X_te_feat)

        # ── MLP ──────────────────────────────────────────────────────────────
        model = train_model(X_tr_n, y_tr_all, X_va_n, y_va, hidden)
        model.eval()
        with torch.no_grad():
            preds_fp = model(
                torch.tensor(X_te_n, dtype=torch.float32)
            ).argmax(1).numpy()
        mlp_fp32.append(accuracy_score(y_te, preds_fp))

        preds_q8 = ptq_and_infer(model, X_tr_n, X_te_n)
        mlp_int8.append(accuracy_score(y_te, preds_q8))
        mlp_true.extend(y_te.tolist()); mlp_pred.extend(preds_q8.tolist())

        # ── Logistic Regression ─────────────────────────────────────────────
        # Same train set (X_tr_n, y_tr_all) the MLP just used — val split is
        # irrelevant here since LogReg has no early-stopping / epochs.
        lr = LogisticRegression(max_iter=LR_MAX_ITER)
        lr.fit(X_tr_n, y_tr_all)
        lr_p32 = lr.predict(X_te_n)
        lr_fp32.append(accuracy_score(y_te, lr_p32))

        lr_p8 = quantize_linear_predict(lr.coef_, lr.intercept_, X_tr_n, X_te_n)
        lr_int8.append(accuracy_score(y_te, lr_p8))
        lr_true.extend(y_te.tolist()); lr_pred.extend(lr_p8.tolist())

    return {
        arch_label(hidden): {
            "fp32": mlp_fp32, "int8": mlp_int8,
            "y_true": np.array(mlp_true), "y_pred": np.array(mlp_pred),
            "bytes": param_bytes(hidden), "macs": num_macs(hidden),
        },
        "Logistic Regression": {
            "fp32": lr_fp32, "int8": lr_int8,
            "y_true": np.array(lr_true), "y_pred": np.array(lr_pred),
            "bytes": linear_param_bytes(), "macs": linear_macs(),
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  FORS-EMG TinyML Benchmark — Best 6-Gesture Subset (Validation)")
    print("=" * 65)

    subjects = load_all_subjects(BASE_DIR)
    total_recs = sum(len(s['recordings']) for s in subjects)
    print(f"\n  Subjects loaded  : {len(subjects)}")
    print(f"  Total recordings : {total_recs}")
    print(f"  Target gestures  : {N_CLASSES}  "
          f"({', '.join(GESTURE_NAMES[c] for c in CLASSES)})")
    print(f"  Orientations     : {', '.join(FOREARM_ORIENTATIONS)}")
    print(f"  Window           : {WIN_SAMPLES} samples ({200} ms), "
          f"increment {INC_SAMPLES} samples ({100} ms)")
    print(f"  Input dim        : {INPUT_DIM}  ({N_CHANNELS} ch × {N_FEATURES} feats)")
    print(f"  Device           : {DEVICE}")

    sample_rec = subjects[0]['recordings'][0]
    print(f"\n  Raw signal scale check (Subject{subjects[0]['subject_id']}, "
          f"first recording):")
    print(f"    shape={sample_rec.shape}  min={sample_rec.min():.4f}  "
          f"max={sample_rec.max():.4f}  mean={sample_rec.mean():.4f}  "
          f"std={sample_rec.std():.4f}")

    subjects = filter_all_subjects(subjects, apply_filter=APPLY_FILTER)

    all_labels = [lbl for s in subjects for lbl in s['labels']]
    print(f"\n  Class balance (recordings):")
    for c in CLASSES:
        cnt = all_labels.count(c)
        print(f"    [{c:2d}] {GESTURE_NAMES[c]:16s}: {cnt:4d}  "
              f"({cnt/len(all_labels)*100:.1f}%)")

    print(f"\n{'─'*65}")
    print("  Architecture / classifier budget (INT8 weights + INT32 biases)")
    print(f"{'─'*65}")
    brows = [[arch_label(h), f"{param_bytes(h):,}", f"{num_macs(h):,}",
              "✓" if param_bytes(h) <= 10_000 else "✗ OVER"]
             for h in HIDDEN_WIDTHS]
    brows.append(["Logistic Regression", f"{linear_param_bytes():,}",
                  f"{linear_macs():,}",
                  "✓" if linear_param_bytes() <= 10_000 else "✗ OVER"])
    print(tabulate(brows,
                   headers=["Model","Param bytes","MACs","≤10 KB?"],
                   tablefmt="rounded_outline"))

    total_windows = 0
    for s in subjects:
        w, _ = windows_from_recordings(s['recordings'], s['labels'])
        total_windows += len(w)
    avg_test_fold      = total_windows / len(subjects)
    avg_train_pool     = avg_test_fold * (len(subjects) - 1)
    avg_train_post_val = avg_train_pool * (1 - 1.0 / N_TRIALS)

    print(f"\n{'─'*65}")
    print("  Overfitting-risk check (real parameter count vs. training windows)")
    print(f"{'─'*65}")
    print(f"  Total windows (all subjects) : {total_windows:,.0f}")
    print(f"  Avg. training windows/fold   : {avg_train_post_val:,.0f}  "
          f"(after held-out val trial)")
    orows = [[arch_label(h), f"{num_params(h):,}",
              f"{avg_train_post_val / num_params(h):,.1f}x"]
             for h in HIDDEN_WIDTHS]
    orows.append(["Logistic Regression", f"{linear_num_params():,}",
                  f"{avg_train_post_val / linear_num_params():,.1f}x"])
    print(tabulate(orows,
                   headers=["Model","Real params","Samples per param"],
                   tablefmt="rounded_outline"))
    print("  Rule of thumb: ≥5-10x samples per parameter is comfortable for a")
    print("  regularised shallow MLP. Below ~3x, treat results with caution.")
    print("  (Logistic Regression's convex loss makes it far less overfitting-")
    print("   prone than this ratio alone would suggest for a neural net.)")

    print(f"\n{'─'*65}")
    print(f"  LOSO-CV ({len(subjects)}-fold) — MLP and Logistic Regression, "
          f"same folds")
    print(f"{'─'*65}\n")
    # loso_cv trains both models per fold on identical splits/features, so
    # there is exactly one HIDDEN_WIDTHS entry here (fixed architecture —
    # see module docstring) but two model results come back from one call.
    raw = loso_cv(subjects, HIDDEN_WIDTHS[0])
    results = {}
    for label, d in raw.items():
        fp32, int8 = d["fp32"], d["int8"]
        results[label] = dict(
            fp32_mean=np.mean(fp32), fp32_std=np.std(fp32),
            int8_mean=np.mean(int8), int8_std=np.std(int8),
            q_drop=np.mean(fp32)-np.mean(int8),
            pb=d["bytes"], macs=d["macs"],
            fp32_folds=fp32, int8_folds=int8,
            y_true=d["y_true"], y_pred=d["y_pred"],
        )

    print(f"\n{'─'*65}")
    print("  Results  (mean ± std over subjects)")
    print(f"{'─'*65}")
    order = sorted(results, key=lambda k: -results[k]["int8_mean"])
    rows = [[lbl,
             f"{results[lbl]['fp32_mean']*100:.2f}±{results[lbl]['fp32_std']*100:.2f}",
             f"{results[lbl]['int8_mean']*100:.2f}±{results[lbl]['int8_std']*100:.2f}",
             f"{results[lbl]['q_drop']*100:.2f}",
             f"{results[lbl]['pb']:,}",
             f"{results[lbl]['macs']:,}"]
            for lbl in order]
    print(tabulate(rows,
                   headers=["Model","FP32 %","INT8 %","Q-drop pp",
                             "Bytes","MACs"],
                   tablefmt="rounded_outline"))
    print(f"  Chance level: {100/N_CLASSES:.1f}%  ({N_CLASSES} classes)")

    # ── Paired significance test — same 19 subjects, so between-subject
    # variance cancels and this is the correct test (not two independent
    # samples: MLP and LogReg were evaluated on the identical test windows).
    mlp_label = arch_label(HIDDEN_WIDTHS[0])
    lr_label  = "Logistic Regression"
    d = np.array(results[lr_label]["int8_folds"]) - np.array(results[mlp_label]["int8_folds"])
    t, p = stats.ttest_rel(results[lr_label]["int8_folds"], results[mlp_label]["int8_folds"])
    ci = stats.t.ppf(0.975, len(d)-1) * d.std(ddof=1) / np.sqrt(len(d))
    verdict = ("BETTER than MLP" if p < 0.05 and d.mean() > 0 else
               "WORSE than MLP"  if p < 0.05 and d.mean() < 0 else
               "statistically tied with MLP")
    print(f"\n{'─'*65}")
    print(f"  Paired test: Logistic Regression vs. {mlp_label}  (same {len(d)} subjects)")
    print(f"{'─'*65}")
    print(f"  Mean difference : {d.mean()*100:+.2f} pp  "
          f"(95% CI {(d.mean()-ci)*100:+.2f} to {(d.mean()+ci)*100:+.2f})")
    print(f"  Paired t-test   : t={t:.3f}, p={p:.4f}")
    print(f"  Verdict         : Logistic Regression is {verdict}")

    # ── Per-subject accuracy, both models side by side ─────────────────────
    print(f"\n  Per-subject INT8 accuracy (both models):")
    srows = [[f"S{i+1:02d}"] + [f"{results[lbl]['int8_folds'][i]*100:.1f}" for lbl in order]
             for i in range(len(subjects))]
    srows.append(["MEAN"] + [f"{results[lbl]['int8_mean']*100:.1f}" for lbl in order])
    print(tabulate(srows, headers=["Subj"] + order, tablefmt="simple"))

    # ── Per-model detail: classification report, confusion matrix, most-
    # confused pairs. Both models get this, so you can see whether they
    # struggle on the SAME gesture pairs or different ones.
    for lbl in order:
        r = results[lbl]
        print(f"\n{'═'*65}")
        print(f"  {lbl}  —  INT8 {r['int8_mean']*100:.2f}% ± {r['int8_std']*100:.2f}%")
        print(f"{'═'*65}")
        print(classification_report(
            r['y_true'], r['y_pred'],
            target_names=[GESTURE_NAMES[c] for c in CLASSES],
            digits=3))

        cm = confusion_matrix(r['y_true'], r['y_pred'], labels=CLASSES)
        print("  Confusion matrix (rows=true, cols=predicted; names truncated):")
        short_names = [GESTURE_NAMES[c][:9] for c in CLASSES]
        cm_rows = [[short_names[i]] + list(cm[i]) for i in range(N_CLASSES)]
        print(tabulate(cm_rows,
                       headers=["True \\ Pred"] + short_names,
                       tablefmt="simple"))

        print(f"\n  10 most-confused gesture pairs (off-diagonal, symmetrised):")
        pair_err = []
        for i in range(N_CLASSES):
            for j in range(i+1, N_CLASSES):
                e = cm[i, j] + cm[j, i]
                if e > 0:
                    pair_err.append((e, GESTURE_NAMES[CLASSES[i]], GESTURE_NAMES[CLASSES[j]]))
        pair_err.sort(reverse=True)
        total_err = cm.sum() - np.trace(cm)
        for e, a, b in pair_err[:10]:
            print(f"    {e:6,d} errors ({e/total_err*100:4.1f}% of all errors)   "
                  f"{a} ↔ {b}")

    with open("fors_emg_results_6class.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model","FP32_mean","FP32_std",
                    "INT8_mean","INT8_std","Qdrop_pp","Bytes","MACs"])
        for lbl in order:
            r2 = results[lbl]
            w.writerow([lbl,
                        round(r2['fp32_mean'],6), round(r2['fp32_std'],6),
                        round(r2['int8_mean'],6), round(r2['int8_std'],6),
                        round(r2['q_drop']*100,4),
                        r2['pb'], r2['macs']])
    print(f"\n  Results saved → fors_emg_results_6class.csv")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
