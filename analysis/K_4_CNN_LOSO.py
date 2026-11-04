"""
FORS-EMG — raw-signal CNNs (option 2 and option 5c, large), subject-independent LOSO
=================================================================================
Trains two CNNs directly on the filtered signal window (no hand-made features)
and compares them, per held-out subject, with the validated INT8 LDA (hardware
features SABS, WL, ZC_NODB, SSC, POW_M1 + shift normaliser).

Requires K_4_LDA_Analysis.py (revision 3) in the same folder: data loading,
causal filter, windowing and the LDA reference are imported from it, so the
CNNs see exactly the same windows and folds as the LDA. Prints results only;
writes no files.

Models (every conv is Conv1d + BatchNorm + ReLU; BN is folded into the conv
weights before INT8, so the deployed network has no BN)
-------------------------------------------------------------------------------
  sep  Option 2 — depthwise-separable, temporal then spatial
         stem  depthwise 8→32 (4 temporal filters per electrode), k=15
               pointwise 32→32
         3 separable blocks (depthwise k, stride 2, then pointwise):
               32→48 (k=7) → 48→64 (k=7) → 64→64 (k=5)
         global average pool → FC 64→4

  two  Option 5c — temporal ∥ spatial branches, merged, then joint layers
         temporal branch: depthwise 8→32 (4 per electrode), k=15, stride 2
                          grouped 32→32 within each electrode, k=7
         spatial branch:  pointwise 8→32, stride 2
                          pointwise 32→32
         merge: concatenate feature maps → 64 channels
         2 separable blocks: 64→64 (k=5, stride 2) twice
         global average pool → FC 64→4

Training
--------
  * Same split as the LDA: trials 1-4 of the 18 training subjects train,
    trial 5 of those subjects is used only for early stopping, every trial of
    the held-out subject tests.
  * AdamW, cosine learning-rate schedule, label smoothing, up to MAX_EPOCHS
    with early stopping on validation loss.
  * Augmentation (training only): each electrode of each window is multiplied
    by a random gain exp(N(0, GAIN_SIGMA²)). This imitates electrode-contact
    and skin-impedance differences between people, the main cause of the
    train-test gap.
  * The whole window tensor is placed on the GPU once (≈0.6 GB), so batches
    never cross the CPU-GPU boundary.

INT8 simulation: BN folded into conv weights; weights and every stored
activation quantised to symmetric per-tensor INT8 with 99.9th-percentile
scales calibrated on training windows only; biases kept float (INT32 in
hardware). The output is an argmax and is not quantised.

Run:  python K_4_CNN_LOSO.py
"""

import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t as student_t, wilcoxon
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tabulate import tabulate

import K_4_LDA_Analysis as B

# ─── Configuration ────────────────────────────────────────────────────────────
MODELS        = ["sep", "two"]
TRAIN_TRIALS  = (1, 2, 3, 4)
VAL_TRIAL     = 5
MAX_EPOCHS    = 150
PATIENCE      = 20
BATCH         = 512
LR            = 2e-3
WEIGHT_DECAY  = 1e-3
LABEL_SMOOTH  = 0.1
DROPOUT       = 0.3
GAIN_SIGMA    = 0.3            # per-electrode log-gain augmentation (0 = off)
CALIB_WINDOWS = 8192
EVAL_BATCH    = 8192
SEED          = 0
LDA_HW_TYPES  = ["SABS", "WL", "ZC_NODB", "SSC", "POW_M1"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBN(nn.Module):
    """Conv1d (padding k//2) + BatchNorm. fold() merges BN into the conv."""
    def __init__(self, cin, cout, k, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, groups=groups)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))

    @torch.no_grad()
    def fold(self):
        if isinstance(self.bn, nn.Identity):
            return
        s = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        self.conv.weight.mul_(s[:, None, None])
        self.conv.bias.copy_((self.conv.bias - self.bn.running_mean) * s + self.bn.bias)
        self.bn = nn.Identity()


class QSim(nn.Module):
    """_q(name, x): identity in training, records |x| in "calib" mode,
    rounds to INT8 in "quant" mode."""
    def __init__(self):
        super().__init__()
        self.qmode, self.qscales, self.qstats = None, {}, {}

    def _q(self, name, x):
        if self.qmode == "calib":
            v = x.detach().abs().flatten()
            if v.numel() > 200_000:
                v = v[torch.randint(v.numel(), (200_000,), device=v.device)]
            self.qstats.setdefault(name, []).append(v.float().cpu())
        elif self.qmode == "quant":
            s = self.qscales[name]
            x = torch.clamp(torch.round(x / s), B.INT8_MIN, B.INT8_MAX) * s
        return x

    def _run(self, prefix, layers, x):
        for i, layer in enumerate(layers):
            x = self._q(f"{prefix}{i}", F.relu(layer(x)))
        return x


def separable(cin, cout, k, stride):
    return [ConvBN(cin, cin, k, stride, groups=cin), ConvBN(cin, cout, 1)]


# ─── Models ───────────────────────────────────────────────────────────────────

class SepCNN(QSim):
    """Option 2: temporal (depthwise) then spatial (pointwise), repeated."""
    def __init__(self, n_ch=8, n_cls=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [ConvBN(n_ch, 32, 15, 1, groups=n_ch), ConvBN(32, 32, 1)]
            + separable(32, 48, 7, 2) + separable(48, 64, 7, 2) + separable(64, 64, 5, 2))
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(64, n_cls)

    def forward(self, x):
        x = self._run("l", self.layers, self._q("in", x))
        return self.fc(self.drop(self._q("gap", x.mean(dim=2))))


class TwoBranchCNN(QSim):
    """Option 5c: temporal branch ∥ spatial branch, feature maps merged,
    then joint separable layers."""
    def __init__(self, n_ch=8, n_cls=4):
        super().__init__()
        self.tb = nn.ModuleList([ConvBN(n_ch, 32, 15, 2, groups=n_ch),
                                 ConvBN(32, 32, 7, 1, groups=n_ch)])
        self.sb = nn.ModuleList([ConvBN(n_ch, 32, 1, 2), ConvBN(32, 32, 1)])
        self.joint = nn.ModuleList(separable(64, 64, 5, 2) + separable(64, 64, 5, 2))
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(64, n_cls)

    def forward(self, x):
        x = self._q("in", x)
        t = self._run("t", self.tb, x)
        s = self._run("s", self.sb, x)
        z = self._run("j", self.joint, torch.cat([t, s], dim=1))
        return self.fc(self.drop(self._q("gap", z.mean(dim=2))))


MODEL_DEFS = {
    "sep": ("Option 2: depthwise-separable", SepCNN),
    "two": ("Option 5c: two-branch merged", TwoBranchCNN),
}


def count_macs(model, length):
    macs = 0
    def hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv1d):
            macs += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]
        elif isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv1d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, B.N_CHANNELS, length))
    for h in hs:
        h.remove()
    return macs


def deployed_params(model):
    """Weights + biases after BN folding (BN parameters disappear)."""
    return sum(m.weight.numel() + m.bias.numel() for m in model.modules()
               if isinstance(m, (nn.Conv1d, nn.Linear)))


# ─── Data ─────────────────────────────────────────────────────────────────────

def build_windows(subjects):
    filt = B._design_filters()
    W, y, sid, trial = [], [], [], []
    full_scale = 0.0
    for s in tqdm(subjects, desc="Filtering + windowing"):
        for rec, lbl, m in zip(s["recordings"], s["labels"], s["meta"]):
            sig = B.apply_preprocessing_filter(rec.T, filt, "causal").T
            full_scale = max(full_scale, float(np.abs(sig).max()))
            w, yy = B.windows_from_recordings([sig], [lbl])
            W.append(w); y.append(yy)
            sid.append(np.full(len(yy), s["subject_id"]))
            trial.append(np.full(len(yy), m["trial"]))
    W = np.concatenate(W).astype(np.float32)
    Xlda = np.vstack([B.extract_hw_features(B.quantize_signal(W[i:i + 4000], full_scale),
                                            LDA_HW_TYPES)
                      for i in range(0, len(W), 4000)])
    return W, Xlda, np.concatenate(y), np.concatenate(sid), np.concatenate(trial)


# ─── Training / evaluation (everything stays on the GPU) ──────────────────────

def batch(Xd, idx_d, inv_sig, augment=False):
    xb = Xd[idx_d] * inv_sig
    if augment and GAIN_SIGMA > 0:
        xb = xb * torch.exp(GAIN_SIGMA * torch.randn(xb.shape[0], xb.shape[1], 1,
                                                     device=xb.device))
    return xb


@torch.no_grad()
def predict(model, Xd, idx, inv_sig):
    model.eval()
    idx_d = torch.from_numpy(idx).to(DEVICE)
    out = [model(batch(Xd, idx_d[i:i + EVAL_BATCH], inv_sig)).argmax(1)
           for i in range(0, len(idx), EVAL_BATCH)]
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def eval_loss(model, Xd, yd, idx_d, inv_sig):
    model.eval()
    tot = 0.0
    for i in range(0, len(idx_d), EVAL_BATCH):
        ib = idx_d[i:i + EVAL_BATCH]
        tot += F.cross_entropy(model(batch(Xd, ib, inv_sig)), yd[ib], reduction="sum").item()
    return tot / len(idx_d)


def train_cnn(cls, Xd, yd, idx_tr, idx_va, inv_sig, seed):
    torch.manual_seed(seed)
    model = cls().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    tr_d = torch.from_numpy(idx_tr).to(DEVICE)
    va_d = torch.from_numpy(idx_va).to(DEVICE)
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, epochs = float("inf"), None, 0, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = tr_d[torch.randperm(len(tr_d), generator=g).to(DEVICE)]
        for i in range(0, len(perm), BATCH):
            ib = perm[i:i + BATCH]
            loss = F.cross_entropy(model(batch(Xd, ib, inv_sig, augment=True)), yd[ib],
                                   label_smoothing=LABEL_SMOOTH)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        epochs = ep + 1
        v = eval_loss(model, Xd, yd, va_d, inv_sig)
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, epochs


@torch.no_grad()
def int8_sim(model, Xd, idx_calib, inv_sig):
    """Fold BN, calibrate activation scales, quantise weights to INT8."""
    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, ConvBN):
            mod.fold()
    m.qmode, m.qstats = "calib", {}
    idx_d = torch.from_numpy(idx_calib).to(DEVICE)
    for i in range(0, len(idx_d), 2048):
        m(batch(Xd, idx_d[i:i + 2048], inv_sig))
    m.qscales = {k: B.compute_scale(torch.cat(v).numpy()) for k, v in m.qstats.items()}
    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            w = mod.weight.detach().cpu().numpy()
            s = B.compute_scale(w)
            mod.weight.copy_(torch.from_numpy(
                np.clip(np.round(w / s), B.INT8_MIN, B.INT8_MAX) * s).to(mod.weight))
    m.qmode = "quant"
    return m


def lda_reference(Xtr, ytr, Xte, yte):
    sc = B.ShiftNormalizer()
    Xtr_n, Xte_n = sc.fit_transform(Xtr), sc.transform(Xte)
    lda = B.build_lda().fit(Xtr_n, ytr)
    W, b = B.lda_as_linear_layer(lda)
    q = B.calibrate_ptq([(W, b)], Xtr_n, input_scale=1.0)
    return accuracy_score(yte, lda.classes_[B.int8_infer(q, Xte_n)])


# ─── Statistics ───────────────────────────────────────────────────────────────

def mean_ci(v):
    v = np.asarray(v, dtype=float); n = len(v)
    return v.mean(), student_t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)


def paired_row(a, b):
    d = (np.asarray(a) - np.asarray(b)) * 100
    m, h = mean_ci(d)
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    lo, hi = m - h, m + h
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "no detectable difference"
    return [f"{m:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{p:.3f}",
            f"{(d > 0).sum()}/{(d < 0).sum()}", verdict]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 76)
    print("  FORS-EMG — raw-signal CNNs (large) vs INT8 LDA (paired LOSO)")
    print("=" * 76)
    subjects = B.load_all_subjects(B.BASE_DIR)
    Wnp, Xlda, y, sid, trial = build_windows(subjects)
    ids = [s["subject_id"] for s in subjects]
    del subjects
    L = Wnp.shape[2]
    Xd = torch.from_numpy(Wnp).to(DEVICE)
    yd = torch.from_numpy(y.astype(np.int64)).to(DEVICE)
    print(f"\n  Windows: {len(y):,} × {B.N_CHANNELS} ch × {L} samples   Device: {DEVICE}")
    print(f"  Training: ≤{MAX_EPOCHS} epochs (patience {PATIENCE}), batch {BATCH}, "
          f"gain augmentation σ = {GAIN_SIGMA}")

    rows = [["LDA (hardware features)", 4 * 40 + 4, 160]]
    for k in MODELS:
        name, cls = MODEL_DEFS[k]
        m = cls()
        for mod in m.modules():
            if isinstance(mod, ConvBN):
                mod.fold()
        rows.append([name, deployed_params(m), count_macs(m, L)])
    print(tabulate(rows, headers=["Model", "Deployed params", "MACs / window"],
                   tablefmt="simple"))

    res = {"lda": []}
    res.update({k: [] for k in MODELS})
    in_train = np.isin(trial, TRAIN_TRIALS)
    rng = np.random.default_rng(SEED)

    for fi, t in enumerate(ids):
        tf = time.time()
        idx_tr = np.where((sid != t) & in_train)[0]
        idx_va = np.where((sid != t) & (trial == VAL_TRIAL))[0]
        idx_te = np.where(sid == t)[0]

        sub = rng.choice(idx_tr, size=min(8000, len(idx_tr)), replace=False)
        inv_sig = torch.from_numpy((1.0 / Wnp[sub].std(axis=(0, 2))).astype(np.float32)
                                   )[None, :, None].to(DEVICE)
        calib = rng.choice(idx_tr, size=min(CALIB_WINDOWS, len(idx_tr)), replace=False)

        res["lda"].append(dict(test_int8=lda_reference(Xlda[idx_tr], y[idx_tr],
                                                       Xlda[idx_te], y[idx_te])))
        line = [f"S{t:02d}", f"LDA {res['lda'][-1]['test_int8']:.2%}"]

        for k in MODELS:
            model, ep = train_cnn(MODEL_DEFS[k][1], Xd, yd, idx_tr, idx_va, inv_sig,
                                  SEED + fi)
            mq = int8_sim(model, Xd, calib, inv_sig)
            r = dict(
                epochs=ep,
                train_fp32=accuracy_score(y[idx_tr], predict(model, Xd, idx_tr, inv_sig)),
                val_fp32=accuracy_score(y[idx_va], predict(model, Xd, idx_va, inv_sig)),
                test_fp32=accuracy_score(y[idx_te], predict(model, Xd, idx_te, inv_sig)),
                test_int8=accuracy_score(y[idx_te], predict(mq, Xd, idx_te, inv_sig)),
            )
            res[k].append(r)
            line.append(f"{k} {r['test_fp32']:.2%} (INT8 {r['test_int8']:.2%}, {ep} ep)")
            del model, mq
        print(f"  [{fi+1:2d}/{len(ids)}] " + "  |  ".join(line) + f"  [{time.time()-tf:.0f}s]",
              flush=True)

    col = lambda k, m: np.array([r[m] for r in res[k]])

    print(f"\n{'─'*76}\n  Accuracy (mean over {len(ids)} held-out subjects; ± = 95% CI)\n{'─'*76}")
    srows = []
    m, h = mean_ci(col("lda", "test_int8"))
    srows.append(["LDA (hardware features)", "", "", "", f"{m*100:.2f} ± {h*100:.2f}", ""])
    for k in MODELS:
        mf, hf = mean_ci(col(k, "test_fp32"))
        mq, hq = mean_ci(col(k, "test_int8"))
        srows.append([MODEL_DEFS[k][0],
                      f"{col(k,'train_fp32').mean()*100:.2f}",
                      f"{col(k,'val_fp32').mean()*100:.2f}",
                      f"{mf*100:.2f} ± {hf*100:.2f}",
                      f"{mq*100:.2f} ± {hq*100:.2f}",
                      f"{col(k,'epochs').mean():.0f}"])
    print(tabulate(srows, headers=["Model", "Train FP32 %", "Val FP32 %",
                                   "Test FP32 %", "Test INT8 %", "Epochs"],
                   tablefmt="simple"))

    print(f"\n{'─'*76}\n  Paired comparisons (test, per subject)\n{'─'*76}")
    prow = []
    for k in MODELS:
        prow.append([f"{k} INT8 vs LDA INT8"] + paired_row(col(k, "test_int8"), col("lda", "test_int8")))
        prow.append([f"{k} INT8 vs {k} FP32"] + paired_row(col(k, "test_int8"), col(k, "test_fp32")))
    if len(MODELS) == 2:
        a, b = MODELS[1], MODELS[0]
        prow.append([f"{a} INT8 vs {b} INT8"] + paired_row(col(a, "test_int8"), col(b, "test_int8")))
    print(tabulate(prow, headers=["Comparison", "Δ pp", "95% CI", "Wilcoxon p",
                                  "Up/Down", "Verdict"], tablefmt="simple"))

    print(f"\n  Per-subject test INT8 accuracy (%):")
    print(tabulate([[f"S{t:02d}", f"{res['lda'][i]['test_int8']*100:.2f}"]
                    + [f"{res[k][i]['test_int8']*100:.2f}" for k in MODELS]
                    for i, t in enumerate(ids)],
                   headers=["Subject", "LDA"] + MODELS, tablefmt="simple"))
    print(f"\n  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 76)


if __name__ == "__main__":
    main()
