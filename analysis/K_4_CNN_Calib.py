"""
FORS-EMG — global raw-signal CNN + post-training calibration on one subject
==========================================================================
1. Global model: trained on every subject except TARGET (trials 1-4 train,
   trial 5 early stopping), i.e. the LOSO fold for TARGET.
2. Calibration: the global model is fine-tuned on TARGET's own data using
   1 trial and 2 trials (each starting from the same global weights).
3. All three (global, 1-trial, 2-trial) are tested on the same TARGET trials,
   none of which is used for calibration.

Requires K_4_LDA_Analysis.py (revision 3) for loading, filtering and windowing.
Prints results only; writes no files.

Run:  python K_4_CNN_Calib.py
"""

import copy, contextlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score
from tabulate import tabulate

import K_4_LDA_Analysis as B

# ─── Configuration ────────────────────────────────────────────────────────────
MODELS       = ["sep", "two"]
TARGET       = 15
TRAIN_TRIALS = (1, 2, 3, 4)
VAL_TRIAL    = 5
CALIB_SETS   = {"1 trial": (1,), "2 trials": (1, 2)}

# global training
MAX_EPOCHS   = 150
PATIENCE     = 20
BATCH        = 512
LR           = 2e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTH = 0.1
DROPOUT      = 0.3
GAIN_SIGMA   = 0.3

# calibration (fine-tuning, fixed epochs: no target data is spare for early stopping)
CAL_EPOCHS   = 30
CAL_BATCH    = 128
CAL_LR       = 5e-4

EVAL_BATCH   = 8192
SEED         = 0

# ─── Device / speed ───────────────────────────────────────────────────────────
AMP = None
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def amp():
    return (torch.autocast(DEVICE.type, dtype=AMP) if AMP is not None
            else contextlib.nullcontext())


# ─── Models ───────────────────────────────────────────────────────────────────

class ConvBN(nn.Sequential):
    """Conv1d + BatchNorm + ReLU (BN foldable into the conv for deployment)."""
    def __init__(self, cin, cout, k, stride=1, groups=1):
        super().__init__(nn.Conv1d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


def separable(cin, cout, k, stride):
    return [ConvBN(cin, cin, k, stride, groups=cin), ConvBN(cin, cout, 1)]


class SepCNN(nn.Module):
    """Option 2: depthwise-separable, temporal then spatial."""
    def __init__(self, n_ch=8, n_cls=4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBN(n_ch, 32, 15, groups=n_ch), ConvBN(32, 32, 1),
            *separable(32, 48, 7, 2), *separable(48, 64, 7, 2), *separable(64, 64, 5, 2))
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.head(self.features(x).mean(2))


class TwoBranchCNN(nn.Module):
    """Option 5c: temporal ∥ spatial branches, merged, then joint layers."""
    def __init__(self, n_ch=8, n_cls=4):
        super().__init__()
        self.tb = nn.Sequential(ConvBN(n_ch, 32, 15, 2, groups=n_ch),
                                ConvBN(32, 32, 7, groups=n_ch))
        self.sb = nn.Sequential(ConvBN(n_ch, 32, 1, 2), ConvBN(32, 32, 1))
        self.joint = nn.Sequential(*separable(64, 64, 5, 2), *separable(64, 64, 5, 2))
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(64, n_cls))

    def forward(self, x):
        z = self.joint(torch.cat([self.tb(x), self.sb(x)], 1))
        return self.head(z.mean(2))


MODEL_DEFS = {"sep": ("Option 2: depthwise-separable", SepCNN),
              "two": ("Option 5c: two-branch merged", TwoBranchCNN)}


# ─── Data (filtering parallel over all CPU cores) ─────────────────────────────

def _subject_windows(s, filt):
    W, y, tr = [], [], []
    for rec, lbl, m in zip(s["recordings"], s["labels"], s["meta"]):
        sig = B.apply_preprocessing_filter(rec.T, filt, "causal").T
        w, yy = B.windows_from_recordings([sig], [lbl])
        W.append(w.astype(np.float32)); y.append(yy)
        tr.append(np.full(len(yy), m["trial"]))
    y = np.concatenate(y)
    return np.concatenate(W), y, np.full(len(y), s["subject_id"]), np.concatenate(tr)


def build_windows(subjects):
    filt = B._design_filters()
    out = Parallel(n_jobs=-1)(delayed(_subject_windows)(s, filt) for s in subjects)
    return [np.concatenate(z) for z in zip(*out)]


# ─── Training / evaluation (all on device) ────────────────────────────────────

def batch(Xd, ib, inv_sig, augment=False):
    xb = Xd[ib] * inv_sig
    if augment and GAIN_SIGMA > 0:
        xb = xb * torch.exp(GAIN_SIGMA * torch.randn(xb.shape[0], xb.shape[1], 1,
                                                     device=xb.device))
    return xb


@torch.no_grad()
def accuracy(model, Xd, y, idx, inv_sig):
    model.eval()
    idx_d = torch.as_tensor(idx, device=DEVICE)
    with amp():
        pred = torch.cat([model(batch(Xd, idx_d[i:i + EVAL_BATCH], inv_sig)).argmax(1)
                          for i in range(0, len(idx_d), EVAL_BATCH)])
    return accuracy_score(y[idx], pred.cpu().numpy())


@torch.no_grad()
def eval_loss(model, Xd, yd, idx_d, inv_sig):
    model.eval()
    tot = torch.zeros((), device=DEVICE)
    with amp():
        for i in range(0, len(idx_d), EVAL_BATCH):
            ib = idx_d[i:i + EVAL_BATCH]
            tot += F.cross_entropy(model(batch(Xd, ib, inv_sig)).float(), yd[ib],
                                   reduction="sum")
    return tot.item() / len(idx_d)


def fit(model, Xd, yd, idx_tr, inv_sig, epochs, lr, bs, seed, idx_va=None):
    """Train with AdamW + cosine LR. Early stopping only if idx_va is given."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(enabled=AMP is torch.float16)
    tr_d = torch.as_tensor(idx_tr, device=DEVICE)
    va_d = None if idx_va is None else torch.as_tensor(idx_va, device=DEVICE)
    best, best_state, wait, ep = float("inf"), None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr_d[torch.randperm(len(tr_d), device=DEVICE)]
        for i in range(0, len(perm), bs):
            ib = perm[i:i + bs]
            with amp():
                loss = F.cross_entropy(model(batch(Xd, ib, inv_sig, augment=True)).float(),
                                       yd[ib], label_smoothing=LABEL_SMOOTH)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        if va_d is None:
            continue
        v = eval_loss(model, Xd, yd, va_d, inv_sig)
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ep


# ─── Main ─────────────────────────────────────────────────────────────────────

def mean_ci(v):
    v = np.asarray(v) * 100
    return f"{v.mean():.2f} ± {student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)):.2f}"


def main():
    t0 = time.time()
    W, y, sid, trial = build_windows(B.load_all_subjects(B.BASE_DIR))
    Xd = torch.from_numpy(W).to(DEVICE)
    yd = torch.from_numpy(y.astype(np.int64)).to(DEVICE)
    ids = np.unique(sid)
    used = sorted({t for ts in CALIB_SETS.values() for t in ts})
    print(f"Device {DEVICE} (AMP {AMP}) | windows {len(y):,} | {len(ids)} LOSO folds")

    acc = {k: [] for k in MODELS}          # global model, all trials of held-out subject
    calib_rows = []
    for fi, t in enumerate(ids):
        tf = time.time()
        src = sid != t
        idx_tr = np.where(src & np.isin(trial, TRAIN_TRIALS))[0]
        idx_va = np.where(src & (trial == VAL_TRIAL))[0]
        idx_all = np.where(sid == t)[0]

        sub = np.random.default_rng(SEED).choice(idx_tr, size=min(8000, len(idx_tr)),
                                                 replace=False)
        inv_sig = torch.from_numpy((1.0 / W[sub].std(axis=(0, 2))).astype(np.float32)
                                   )[None, :, None].to(DEVICE)
        line = []
        for k in MODELS:
            name, cls = MODEL_DEFS[k]
            torch.manual_seed(SEED)
            g, ep = fit(cls().to(DEVICE), Xd, yd, idx_tr, inv_sig,
                        MAX_EPOCHS, LR, BATCH, SEED, idx_va)
            acc[k].append(accuracy(g, Xd, y, idx_all, inv_sig))
            line.append(f"{k} {acc[k][-1]:.2%} ({ep} ep)")

            if t == TARGET:                # calibration only on the target fold
                idx_te = np.where((sid == t) & ~np.isin(trial, used))[0]
                row = [name, f"{accuracy(g, Xd, y, idx_te, inv_sig) * 100:.2f}"]
                for ts in CALIB_SETS.values():
                    ic = np.where((sid == t) & np.isin(trial, ts))[0]
                    m, _ = fit(copy.deepcopy(g), Xd, yd, ic, inv_sig,
                               CAL_EPOCHS, CAL_LR, CAL_BATCH, SEED + 1)
                    row.append(f"{accuracy(m, Xd, y, idx_te, inv_sig) * 100:.2f}")
                calib_rows.append(row)
        print(f"  [{fi+1:2d}/{len(ids)}] S{t:02d}  " + "  |  ".join(line)
              + f"  [{time.time()-tf:.0f}s]", flush=True)

    s = int(np.where(ids == TARGET)[0][0])
    print(f"\nGlobal CNN before calibration (LOSO, each subject tested on all its trials)")
    print(tabulate([[MODEL_DEFS[k][0], mean_ci(acc[k]), f"{acc[k][s]*100:.2f}"] for k in MODELS],
                   headers=["Model", f"Mean over {len(ids)} subjects (± 95% CI) %",
                            f"S{TARGET:02d} %"], tablefmt="simple"))
    print("\n" + tabulate([[f"S{t:02d}"] + [f"{acc[k][i]*100:.2f}" for k in MODELS]
                           for i, t in enumerate(ids)],
                          headers=["Subject"] + MODELS, tablefmt="simple"))

    te_trials = sorted(set(trial[(sid == TARGET) & ~np.isin(trial, used)]))
    print(f"\nCalibration on S{TARGET:02d} (test trials {te_trials})")
    print(tabulate(calib_rows, headers=["Model", "Global %"]
                   + [f"Calib {n} %" for n in CALIB_SETS], tablefmt="simple"))
    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
