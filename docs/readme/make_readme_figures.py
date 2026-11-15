"""
Banner and animations for the top-level README.

Everything is computed from this repository's own data and code; nothing is hand-drawn:
  * RTL/tb/mb_data/       real FORS-EMG stream (subject 6, trial 1) + the chip model trained
                          without subject 6, exactly what the multi-board testbench runs;
  * golden_model/lda_golden_model.py   bit-exact Stages 1-3 of the chip;
  * analysis/K_4_LDA_Calib.py          the per-user calibration (needs the dataset in archive/).

Run from the repository root:
    python docs/readme/make_readme_figures.py          # everything (about 2 min)
    python docs/readme/make_readme_figures.py fast     # skip the two that load the whole dataset
Writes docs/readme/*.gif and banner.png.
"""
import sys
import types
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "golden_model"), str(ROOT / "analysis")]
try:
    import tabulate  # noqa: F401  (only the analysis scripts' printing needs it)
except ImportError:
    sys.modules["tabulate"] = types.SimpleNamespace(tabulate=lambda *a, **k: "")
import lda_golden_model as G  # noqa: E402

NAMES = ["Hand Close", "Hand Open", "Wrist Flexion", "Wrist Extension"]
COL = ["#1f5fbf", "#f59e0b", "#2ca02c", "#dc2626"]
INK, GREY, PALE, NAVY = "#0b1a33", "#8a94a6", "#e8edf5", "#13304f"
FS = 985
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK})


def save(anim, name, fps):
    path = OUT / name
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(anim._fig)
    print(f"  {name:28s} {path.stat().st_size / 1e6:5.2f} MB")


# ─── data of the multi-board testbench ───────────────────────────────────────

def load_mb():
    d = ROOT / "RTL" / "tb" / "mb_data"
    a = np.loadtxt(d / "samples.csv", delimiter=",", skiprows=1, dtype=np.int64)
    lab, x = a[:, 1], a[:, 2:]
    mu, k, W, b = np.zeros(40, np.int64), np.zeros(40, np.int64), [], None
    for line in open(d / "model.txt"):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if p[0] == "mu":
            mu[int(p[1])], k[int(p[1])] = int(p[2]), int(p[3])
        elif p[0] == "w":
            W.append([int(v) for v in p[2:]])
        elif p[0] == "b":
            b = np.array([int(v) for v in p[1:]], np.int64)
    exp = np.loadtxt(d / "expected.txt", dtype=np.int64)
    W = np.array(W, np.int64)

    raw, _ = G.window_features(x, 8)                 # Stage 1, bit-exact
    q = G.normalize(raw, mu, k)                      # Stage 2
    D = q @ W.T + b                                  # Stage 3, K-1 difference form
    pred = np.argmax(np.c_[np.zeros(len(D), np.int64), D], axis=1)   # first max = strict '>'
    assert np.array_equal(pred, exp[:, 3]), "does not reproduce expected.txt"
    return dict(x=x, lab=lab, mu=mu, k=k, W=W, b=b, raw=raw, q=q, D=D, pred=pred,
                true=exp[:, 2], mixed=exp[:, 5])


# ─── 1. a real recording through the chip ────────────────────────────────────

def emg_to_decision(mb):
    x, q, D, pred, true, mixed = (mb[k] for k in ("x", "q", "D", "pred", "true", "mixed"))
    N, S = len(x), len(D)
    t = np.arange(N) / FS
    VIEW = 12 * G.SUB_WIN

    fig = plt.figure(figsize=(12, 6.6), dpi=80, facecolor="white")
    gs = fig.add_gridspec(4, 2, width_ratios=[2.1, 1], height_ratios=[1, 1, 1.0, 0.3],
                          hspace=0.75, wspace=0.16, left=0.055, right=0.985, top=0.87, bottom=0.05)
    ax_sig = fig.add_subplot(gs[0:2, 0]); ax_q = fig.add_subplot(gs[2, 0])
    ax_sc = fig.add_subplot(gs[0:2, 1]); ax_dec = fig.add_subplot(gs[2, 1]); ax_tl = fig.add_subplot(gs[3, :])
    fig.suptitle("A real recording through the bit-exact chip model", fontsize=16, fontweight="bold", color=INK, y=0.975)
    fig.text(0.5, 0.915, "FORS-EMG subject 6 (never seen in training) · 8 electrodes · 8-bit ADC at 985 Hz · "
             "one decision every 98 samples (99.5 ms)", ha="center", color=GREY, fontsize=11)

    # signals
    sc = np.percentile(np.abs(x), 99.5, axis=0)
    for c in range(8):
        ax_sig.plot(t, (7 - c) + 0.45 * np.clip(x[:, c] / sc[c], -1.1, 1.1), color=INK, lw=0.6)
    for g in range(4):
        a, z = g * 1960 / FS, (g + 1) * 1960 / FS
        ax_sig.axvspan(a, z, color=COL[g], alpha=0.09, lw=0)
    g_lbl = ax_sig.text(0.005, 0.985, "", transform=ax_sig.transAxes, ha="left", va="top",
                        fontsize=11, fontweight="bold")
    ax_sig.set_yticks(range(8)); ax_sig.set_yticklabels([f"ch{7 - c}" for c in range(8)], fontsize=9)
    ax_sig.set_ylim(-0.75, 8.45); ax_sig.set_xlabel("time (s)", fontsize=10)
    ax_sig.set_title("Electrode signals, 196-sample window sliding by 98", loc="left")
    win = Rectangle((0, -0.7), 196 / FS, 9.0, fc="#ffd84d", alpha=0.25, ec=INK, lw=2)
    ax_sig.add_patch(win)

    # features
    im = ax_q.imshow(q[1].reshape(8, 5).T, cmap="RdBu_r", vmin=-128, vmax=127, aspect="auto")
    ax_q.set_yticks(range(5)); ax_q.set_yticklabels(["SABS", "WL", "ZC", "SSC", "POW"], fontsize=9)
    ax_q.set_xticks(range(8)); ax_q.set_xticklabels([f"ch{c}" for c in range(8)], fontsize=9)
    ax_q.set_title("Stages 1-2: 40 INT8 features  x_q = clip((Σ − μ) >> k)", loc="left")
    cells = [[ax_q.text(c, f, "", ha="center", va="center", fontsize=8.5) for c in range(8)] for f in range(5)]

    # scores
    M = np.abs(D).max() * 1.12
    ax_sc.axvline(0, color=GREY, lw=1)
    bars = ax_sc.barh([3, 2, 1, 0], [0, 0, 0, 0], color=COL, height=0.62)
    vals = [ax_sc.text(0, 3 - c, "", va="center", fontsize=10, color=INK) for c in range(4)]
    ax_sc.set_yticks([3, 2, 1, 0]); ax_sc.set_yticklabels(NAMES, fontsize=10)
    ax_sc.set_xlim(-M, M); ax_sc.set_xticks([])
    ax_sc.set_title("Stage 3: INT8 LDA scores → argmax", loc="left")
    ax_sc.text(0.5, -0.06, "D_c = W′_c · x_q + b′_c   (Hand Close ≡ 0)", transform=ax_sc.transAxes,
               ha="center", va="top", color=GREY, fontsize=9.5)
    for s in ("top", "right", "left"):
        ax_sc.spines[s].set_visible(False)

    # decision
    ax_dec.axis("off")
    d_lbl = ax_dec.text(0.0, 0.95, "chip decision", fontsize=10, color=GREY, va="top")
    d_name = ax_dec.text(0.0, 0.72, "", fontsize=21, fontweight="bold", va="top")
    d_true = ax_dec.text(0.0, 0.36, "", fontsize=11, color=INK, va="top")
    d_acc = ax_dec.text(0.0, 0.08, "", fontsize=10, color=GREY, va="top")

    # timeline
    tl = np.ones((2, S - 1, 4))
    tl_im = ax_tl.imshow(tl, aspect="auto", interpolation="nearest")
    ax_tl.set_yticks([0, 1]); ax_tl.set_yticklabels(["true", "chip"], fontsize=9)
    ax_tl.set_xticks([]); ax_tl.set_title("every decision so far", loc="left", fontsize=10, color=GREY)
    rgba = [matplotlib.colors.to_rgba(c) for c in COL]
    ok = tot = 0
    frames = list(range(1, S)) + [S - 1] * 10

    def update(fi):
        nonlocal ok, tot
        s = frames[fi]
        end = (s + 1) * G.SUB_WIN
        lo = max(0, end - VIEW)
        ax_sig.set_xlim(lo / FS, max(end, VIEW) / FS)
        win.set_x((end - 196) / FS)
        g_now = int(mb["lab"][end - 1])
        g_lbl.set_text(f"performing: {NAMES[g_now]}"); g_lbl.set_color(COL[g_now])
        for c, tl_ in enumerate(ax_sc.get_yticklabels()[::-1]):
            tl_.set_fontweight("bold" if c == pred[s] else "normal")
            tl_.set_color(COL[c] if c == pred[s] else GREY)
        Q = q[s].reshape(8, 5).T
        im.set_data(Q)
        for f in range(5):
            for c in range(8):
                v = int(Q[f, c]); cells[f][c].set_text(str(v))
                cells[f][c].set_color("white" if abs(v) > 75 else INK)
        sc4 = [0, *D[s]]
        for c in range(4):
            bars[c].set_width(sc4[c])
            bars[c].set_alpha(1.0 if c == pred[s] else 0.28)
            vals[c].set_text(f"{sc4[c]:,}")
            vals[c].set_x(sc4[c] + (M * 0.02 if sc4[c] >= 0 else -M * 0.02))
            vals[c].set_ha("left" if sc4[c] >= 0 else "right")
        d_name.set_text("→ " + NAMES[pred[s]]); d_name.set_color(COL[pred[s]])
        if mixed[s]:
            d_true.set_text("window spans two gestures")
        else:
            d_true.set_text(f"true: {NAMES[true[s]]}   " + ("✓" if pred[s] == true[s] else "✗"))
        if fi < S - 1:
            if not mixed[s]:
                tot += 1; ok += int(pred[s] == true[s])
            tl[0, s - 1] = rgba[true[s]]; tl[1, s - 1] = rgba[pred[s]]
            tl_im.set_data(tl)
        d_acc.set_text(f"single-gesture windows correct: {ok} / {tot}")
        return []

    save(FuncAnimation(fig, update, frames=len(frames)), "emg_to_decision.gif", fps=7)


# ─── 2. Stage 1: 98 samples → five running sums, no multiplier ───────────────

def stage1_features(mb):
    x, mu, k, raw = mb["x"], mb["mu"], mb["k"], mb["raw"]
    s = 45                                                       # a Wrist Flexion stride
    n0 = s * G.SUB_WIN
    ch = int(np.argmax(np.abs(x[n0:n0 + 98]).sum(0)))
    xs = [int(v) for v in x[:, ch]]

    def bank(start):
        rows = []
        for n in range(start, start + G.SUB_WIN):
            xv, xp = xs[n], xs[n - 1]
            d1, d0 = xv - xp, xs[n - 1] - xs[n - 2]
            rows.append((abs(xv), abs(d1), int((xv < 0) != (xp < 0)),
                         int((d1 > 0 > d0) or (d1 < 0 < d0)), G.mitchell_sq(abs(xv))))
        return np.array(rows, np.int64)

    cur, prev = bank(n0), bank(n0 - G.SUB_WIN)
    tot_cur, tot_prev = cur.sum(0), prev.sum(0)
    assert np.array_equal(tot_cur + tot_prev, raw[s, 5 * ch:5 * ch + 5]), "not bit-exact"
    run = np.cumsum(cur, axis=0)
    TYPES = ["SABS", "WL", "ZC", "SSC", "POW"]
    OPS = ["|x|", "|x − x₋₁|", "sign(x) ≠ sign(x₋₁)", "slope changes sign", "Mitchell |x|²"]

    fig = plt.figure(figsize=(12, 5.6), dpi=80, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.25, 1], hspace=0.42, wspace=0.08,
                          left=0.05, right=0.98, top=0.86, bottom=0.04)
    ax_w = fig.add_subplot(gs[0, :]); ax_t = fig.add_subplot(gs[1, 0]); ax_b = fig.add_subplot(gs[1, 1])
    fig.suptitle("Stage 1: each sample updates five sums with add, compare and shift only",
                 fontsize=15, fontweight="bold", color=INK, y=0.975)
    fig.text(0.5, 0.905, f"channel {ch}, real Wrist Flexion data · one channel per clock cycle · "
             "two 98-sample banks make one 196-sample window", ha="center", color=GREY, fontsize=11)

    idx = np.arange(-98, 98)
    seg = np.array(xs[n0 - 98:n0 + 98])
    A = np.abs(seg).max() * 1.15
    ax_w.axvspan(-98.5, -0.5, color=PALE, lw=0)
    ax_w.text(-49, A * 0.93, "bank 0: filled during the previous stride", ha="center", va="top", color=GREY, fontsize=10)
    ax_w.text(49, A * 0.93, "bank 1: filling now", ha="center", va="top", color=INK, fontsize=10, fontweight="bold")
    ax_w.plot(idx[:98], seg[:98], color=GREY, lw=1)
    ax_w.axhline(0, color=GREY, lw=0.8)
    line, = ax_w.plot([], [], color=NAVY, lw=1.4)
    zc_pts, = ax_w.plot([], [], "|", color="#f59e0b", ms=14, mew=2.4, label="zero crossing")
    ssc_pts, = ax_w.plot([], [], "^", color="#2ca02c", ms=6, label="slope sign change")
    dot, = ax_w.plot([], [], "o", color="#dc2626", ms=9)
    ax_w.set_xlim(-99, 98); ax_w.set_ylim(-A, A)
    ax_w.set_ylabel("ADC code", fontsize=10); ax_w.set_xticks([-98, -49, 0, 49, 97])
    ax_w.legend(loc="lower left", fontsize=9, frameon=False, ncol=2)

    ax_t.axis("off")
    txt = [ax_t.text(0.0, 0.93 - 0.165 * r, "", family="DejaVu Sans Mono", fontsize=11.5, va="top", color=INK)
           for r in range(6)]
    head = ax_t.text(0.0, 1.08, "", fontsize=12, fontweight="bold", color=INK, va="top")

    ax_b.set_xlim(0, 1.32); ax_b.set_ylim(-0.6, 4.6); ax_b.axis("off")
    ax_b.set_title("running sums in bank 1", loc="left")
    bb = ax_b.barh(range(4, -1, -1), [0] * 5, color=[COL[0], COL[1], COL[3], COL[2], NAVY], height=0.6)
    for r, name in enumerate(TYPES):
        ax_b.add_patch(Rectangle((0, 4 - r - 0.3), 1, 0.6, fc="none", ec=PALE, lw=1))
        ax_b.text(-0.02, 4 - r, name, ha="right", va="center", fontsize=10)
    bval = [ax_b.text(1.03, 4 - r, "", va="center", fontsize=10.5, family="DejaVu Sans Mono") for r in range(5)]

    js = [j for j in range(12) for _ in range(2)] + list(range(12, 98, 2)) + [97]
    frames = [("s", j) for j in js] + [("n", 97)] * 24

    def update(fi):
        kind, j = frames[fi]
        n = n0 + j
        line.set_data(idx[98:98 + j + 1], seg[98:98 + j + 1])
        dot.set_data([j], [seg[98 + j]])
        zc = [i for i in range(j + 1) if cur[i, 2]]
        ssc = [i - 1 for i in range(j + 1) if cur[i, 3]]
        zc_pts.set_data(zc, [0] * len(zc))
        ssc_pts.set_data(ssc, seg[[98 + i for i in ssc]] if ssc else [])
        for r in range(5):
            bb[r].set_width(run[j, r] / tot_cur[r] if tot_cur[r] else 0)
            bval[r].set_text(f"{run[j, r]:,}")
        if kind == "s":
            xv, xp = xs[n], xs[n - 1]
            a = abs(xv)
            p = a.bit_length() - 1
            head.set_text(f"sample {j:2d} of 98:   x = {xv},   x₋₁ = {xp}")
            for r in range(4):
                txt[r].set_text(f"{TYPES[r]:<5s} += {cur[j, r]:<4d}  {OPS[r]}")
            if a:
                txt[4].set_text(f"POW   += {cur[j, 4]:<4d}  ({a}<<{p + 1}) − (1<<{2 * p})")
                txt[5].set_text(f"                exact {a}² = {a * a}: no multiplier")
            else:
                txt[4].set_text("POW   += 0     x = 0"); txt[5].set_text("")
            for t_ in txt:
                t_.set_color(INK)
        else:
            f0 = 5 * ch
            head.set_text("Stage 2, once per stride: x_q = (bank0 + bank1 − μ) >> k")
            for r in range(5):
                R = int(tot_prev[r] + tot_cur[r]); m, kk = int(mu[f0 + r]), int(k[f0 + r])
                qv = int(G.normalize(np.int64(R), np.int64(m), np.int64(kk)))
                txt[r].set_text(f"{TYPES[r]:<5s} ({R:>6,} − {m:>6,}) >> {kk:<2d} = {qv:>4d}")
                txt[r].set_color(NAVY)
            txt[5].set_text("→ 5 of the 40 INT8 inputs of the LDA"); txt[5].set_color(GREY)
        return []

    save(FuncAnimation(fig, update, frames=len(frames)), "stage1_features.gif", fps=8)


# ─── 3. Multi-board chain: partial sums added bit-serially ───────────────────

def chain_serial_add(mb):
    q, W, b, D, pred, true, mixed = (mb[k] for k in ("q", "W", "b", "D", "pred", "true", "mixed"))
    cand = [s for s in range(1, len(D)) if pred[s] == true[s] == 3 and not mixed[s] and D[s].min() < 0 < D[s].max()]
    s = cand[len(cand) // 2]
    split = [[0, 1, 2], [3, 4, 5], [6, 7]]                       # head, middle, tail
    own = []
    for bd, chs in enumerate(split):
        f = [5 * c + t for c in chs for t in range(5)]
        own.append([int(W[c, f] @ q[s, f]) + (int(b[c]) if bd == 0 else 0) for c in range(3)])
    assert [sum(o[c] for o in own) for c in range(3)] == [int(v) for v in D[s]]

    def bits(v):
        return [(v & 0xFFFFFFFF) >> i & 1 for i in range(32)]

    # serial adders: tail passes its own bits, every other board adds own + incoming + carry
    ownb = [[bits(own[bd][c]) for c in range(3)] for bd in range(3)]
    outb, carry = [[None] * 3 for _ in range(3)], [[None] * 3 for _ in range(3)]
    for c in range(3):
        inc = [0] * 32
        for bd in (2, 1, 0):
            cy, o, cs = 0, [], []
            for i in range(32):
                a, bb_ = ownb[bd][c][i], inc[i]
                o.append(a ^ bb_ ^ cy); cy = (a & bb_) | (cy & (a ^ bb_)); cs.append(cy)
            outb[bd][c], carry[bd][c], inc = o, cs, o
    total = [sum(v << i for i, v in enumerate(outb[0][c])) for c in range(3)]
    total = [v - (1 << 32) if v >= 1 << 31 else v for v in total]
    assert total == [int(v) for v in D[s]]

    fig = plt.figure(figsize=(12, 6.0), dpi=85, facecolor="white")
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    X0 = [0.025, 0.355, 0.685]; BW = 0.29
    TITLE = ["Board 0 · HEAD · ch 0–2", "Board 1 · ch 3–5", "Board 2 · TAIL · ch 6–7"]
    cyc = sorted(set(range(16)) | set(range(16, 98, 3)) | {33, 65, 97})
    frames = []
    for t in cyc:
        frames += [t] * (4 if t in (33, 65) else 14 if t == 97 else 1)
    CW = (BW - 0.03) / 32

    def word_row(x0, y, bvals, upto, hl, color):
        for i in range(32):
            xx = x0 + 0.015 + (31 - i) * CW
            known = i <= upto
            fc = (color if bvals[i] else "white") if known else "#f3f5f9"
            ax.add_patch(Rectangle((xx, y), CW * 0.86, 0.034, fc=fc, ec=INK if i == hl else "#c5ccd8",
                                   lw=2.2 if i == hl else 0.6))

    def update(fi):
        t = frames[fi]
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 0.975, "Three boards act as one LDA: only three 32-bit partial sums cross, one bit per clock",
                ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
        ax.text(0.5, 0.92, f"real stride of subject 6 · tail → head, LSB first · one full adder + carry flip-flop "
                f"per board · LCLK cycle {t}", ha="center", va="top", fontsize=11, color=GREY)
        done_words = []
        for bd in range(3):
            x0 = X0[bd]
            g = t - (2 - bd)                                          # bit index this board handles now
            w, i = (min(max(g, 0), 95) // 32, g % 32) if 0 <= g < 96 else ((2, 31) if g >= 96 else (0, -1))
            ax.add_patch(FancyBboxPatch((x0, 0.30), BW, 0.56, boxstyle="round,pad=0.006,rounding_size=0.012",
                                        fc="#f7f9fc", ec="#e07a1f" if bd == 0 else NAVY, lw=2,
                                        ls="--" if bd == 0 else "-"))
            ax.text(x0 + 0.012, 0.835, TITLE[bd], fontsize=11.5, fontweight="bold",
                    color="#e07a1f" if bd == 0 else NAVY, va="top")
            for c in range(3):
                lab = f"p{'₁₂₃'[c]}" + (f" + b′{'₁₂₃'[c]}" if bd == 0 else "")
                ax.text(x0 + 0.015, 0.765 - 0.045 * c, f"{lab} = {own[bd][c]:>6,}", fontsize=10.5,
                        family="DejaVu Sans Mono", va="top",
                        color=INK if c == w and 0 <= g < 96 else GREY,
                        fontweight="bold" if c == w and 0 <= g < 96 else "normal")
            active = 0 <= g < 96
            ax.text(x0 + 0.015, 0.59, f"own word {w + 1} (MSB … LSB)", fontsize=9, color=GREY, va="top")
            word_row(x0, 0.525, ownb[bd][w], 31 if active or g >= 96 else -1, i if active else -9, NAVY)
            out_lab = "running total D" if bd == 0 else "sum sent to the left"
            ax.text(x0 + 0.015, 0.495, out_lab, fontsize=9, color=GREY, va="top")
            word_row(x0, 0.43, outb[bd][w], i if active else (31 if g >= 96 else -1), i if active else -9,
                     "#e07a1f" if bd == 0 else "#2ca02c")
            cy = carry[bd][w][i] if active else 0
            ax.text(x0 + 0.015, 0.385, ("bit %2d   carry → %d" % (i, cy)) if active
                    else ("waiting for the first bit" if g < 0 else "done"),
                    fontsize=10, family="DejaVu Sans Mono", color=INK, va="top")
            if bd > 0:                                              # wire to the left neighbour
                xa, xb = x0 - 0.003, X0[bd - 1] + BW + 0.003
                ax.add_patch(FancyArrowPatch((xa, 0.36), (xb, 0.36), arrowstyle="-|>", mutation_scale=16,
                                             color="#2ca02c", lw=2))
                if active:
                    ax.text((xa + xb) / 2, 0.39, str(outb[bd][w][i]), ha="center", fontsize=13,
                            fontweight="bold", color="#2ca02c")
            if bd == 0:
                done_words = [c for c in range(3) if t - 2 >= 32 * c + 31]
        # head's decision panel
        ax.add_patch(FancyBboxPatch((0.025, 0.035), 0.95, 0.2, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc="white", ec="#e0e5ee", lw=1.5))
        ax.text(0.04, 0.215, "head: compare each total as its last bit arrives", fontsize=10.5, color=GREY, va="top")
        best, bc = 0, 0
        ax.text(0.04, 0.155, "Hand Close ≡ 0", fontsize=12, color=COL[0], va="top", fontweight="bold")
        for c in range(3):
            xx = 0.23 + 0.17 * c
            if c in done_words:
                if total[c] > best:
                    best, bc = total[c], c + 1
                ax.text(xx, 0.155, f"D{'₁₂₃'[c]} = {total[c]:,}", fontsize=12, color=COL[c + 1], va="top",
                        fontweight="bold")
            else:
                ax.text(xx, 0.155, f"D{'₁₂₃'[c]} = …", fontsize=12, color="#c5ccd8", va="top")
        if len(done_words) == 3:
            ax.text(0.74, 0.17, f"→ {NAMES[bc]}", fontsize=19, fontweight="bold", color=COL[bc], va="top")
            ax.text(0.74, 0.085, "argmax{0, D₁, D₂, D₃} · same as one chip", fontsize=9.5, color=GREY, va="top")
        else:
            ax.text(0.74, 0.15, f"best so far: {NAMES[bc]}", fontsize=11, color=GREY, va="top")
        ax.text(0.04, 0.085, "the decision equals the single-chip LDA bit for bit (79 / 79 on real data, 6 board splits)",
                fontsize=9.5, color=GREY, va="top")
        return []

    save(FuncAnimation(fig, update, frames=len(frames)), "chain_serial_add.gif", fps=8)


# ─── 4. calibration on all 18 users (numbers from docs/results/) ─────────────

def calibration_bars():
    rows, on = [], False
    for line in open(ROOT / "docs" / "results" / "lda_calibration_18_subjects.txt", encoding="utf-8"):
        if line.startswith("Calibration on every subject"):
            on = True; continue
        if on and line[:1] == "S" and line[1:3].isdigit():
            p = line.split(); rows.append((p[0], float(p[1]), float(p[2])))
        if on and line.startswith("Mean"):
            break
    rows.sort(key=lambda r: r[1])
    sub, g, c = [r[0] for r in rows], np.array([r[1] for r in rows]), np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(11, 6.0), dpi=80, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.1)
    fig.suptitle("Per-user calibration: one recording of each gesture", fontsize=16, fontweight="bold", color=INK, y=0.97)
    fig.text(0.5, 0.885, "global INT8 LDA (18 other people) vs the same model blended with the new user's statistics "
             "(λ = 0.5) · tested on trials 3-5", ha="center", color=GREY, fontsize=10.5)
    y = np.arange(len(sub))
    ax.set_yticks(y); ax.set_yticklabels(sub, fontsize=9.5)
    ax.set_xlim(55, 100); ax.set_ylim(-1, len(sub) + 0.6)
    ax.set_xlabel("accuracy on the new user (%)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color=PALE)
    ax.scatter(g, y, s=46, color=GREY, zorder=3, label="global")
    segs = [ax.plot([g[i], g[i]], [i, i], color="#2ca02c", lw=3.2, solid_capstyle="round")[0] for i in y]
    heads = ax.scatter(g, y, s=60, color="#2ca02c", zorder=4, label="calibrated, 1 trial")
    ax.axvline(g.mean(), color=GREY, ls="--", lw=1.2)
    ax.text(g.mean(), len(sub) - 0.1, f"mean {g.mean():.1f}%", ha="center", color=GREY, fontsize=10)
    mline = ax.axvline(g.mean(), color="#2ca02c", ls="--", lw=1.2)
    mtxt = ax.text(g.mean(), len(sub) + 0.45, "", ha="center", color="#2ca02c", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    note = ax.text(99.5, 0.6, "", ha="right", fontsize=11, color=INK)
    n = 28
    frames = [0] * 6 + list(range(n + 1)) + [n] * 16

    def update(fi):
        f = frames[fi] / n
        f = 1 - (1 - f) ** 2
        cur = g + f * (c - g)
        for i in y:
            segs[i].set_xdata([g[i], cur[i]])
        heads.set_offsets(np.c_[cur, y])
        mline.set_xdata([cur.mean()] * 2); mtxt.set_x(cur.mean())
        mtxt.set_text(f"mean {cur.mean():.1f}%" if f > 0 else "")
        note.set_text(f"+{(c - g).mean():.1f} pp · {(c > g).sum()} / {len(g)} users improve" if f >= 1 else "")
        return []

    save(FuncAnimation(fig, update, frames=len(frames)), "calibration.gif", fps=12)


# ─── 5. LDA feature space: a new user, before and after calibration ──────────

def lda_space():
    import K_4_LDA_Analysis as B
    import K_4_LDA_Calib as KC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    T = KC.TARGET
    from joblib import parallel_backend
    with parallel_backend("threading"):          # worker processes would not see the tabulate stub
        X, y, sid, trial = KC.build_features(B.load_all_subjects(B.BASE_DIR))
    itr = np.where((sid != T) & np.isin(trial, KC.TRAIN_TRIALS))[0]
    ite = np.where((sid == T) & ~np.isin(trial, (1, 2)))[0]
    ica = np.where((sid == T) & (trial == 1))[0]
    sc = B.ShiftNormalizer()
    Xtr = sc.fit_transform(X[itr]); Xn = sc.transform(X)
    lda = B.build_lda().fit(Xtr, y[itr])
    Sg = getattr(lda, "covariance_", None)
    if Sg is None:
        Sg = KC.class_stats(Xtr, y[itr], lda.classes_)[1]
    mu_t = KC.class_stats(Xn[ica], y[ica], lda.classes_)[0]

    def int8_pred(m):
        Wl, bl = B.lda_as_linear_layer(m)
        qp = B.calibrate_ptq([(Wl, bl)], Xtr, input_scale=1.0)
        return m.classes_[B.int8_infer(qp, Xn[ite])]

    lams = np.linspace(0, KC.LAMBDA, 11)
    preds = [int8_pred(lda)] + [int8_pred(KC.calibrate(lda, Sg, Xn[ica], y[ica], l)) for l in lams[1:]]
    accs = [np.mean(p == y[ite]) * 100 for p in preds]
    print(f"  S{T:02d}: global {accs[0]:.2f}%  calibrated (1 trial, λ={KC.LAMBDA}) {accs[-1]:.2f}%")

    proj = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2).fit(Xtr, y[itr])
    rng = np.random.default_rng(0)
    bg = rng.choice(len(itr), 5000, replace=False)
    P_bg = proj.transform(Xtr[bg]); y_bg = y[itr][bg]
    show = rng.choice(len(ite), 700, replace=False)
    P_te = proj.transform(Xn[ite][show]); y_te = y[ite][show]
    mu_g = proj.transform(lda.means_)
    mu_path = [proj.transform((1 - l) * lda.means_ + l * mu_t) for l in lams]

    fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=80, facecolor="white")
    fig.subplots_adjust(left=0.03, right=0.72, top=0.86, bottom=0.04)
    fig.suptitle("What the LDA sees, and what calibration changes", fontsize=16, fontweight="bold", color=INK, y=0.97)
    fig.text(0.5, 0.905, "40 integer features projected on the LDA's two main directions · "
             "colours of the new user's windows = true gesture, ✗ = chip-style INT8 decision wrong",
             ha="center", color=GREY, fontsize=9.8)
    for g in range(4):
        m = y_bg == g
        ax.scatter(*P_bg[m].T, s=5, color=COL[g], alpha=0.07, lw=0)
    lo, hi = np.percentile(np.vstack([P_bg, P_te]), [0.7, 99.3], axis=0)
    pad = (hi - lo) * 0.08
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(PALE)
    dots = ax.scatter([], [], s=16, lw=0)
    crosses = ax.scatter([], [], s=26, marker="x", lw=1.3)
    gm = ax.scatter(*mu_g.T, s=260, marker="o", facecolor="white", edgecolor=COL, lw=2.5, zorder=5)
    cm = ax.scatter(*mu_g.T, s=300, marker="*", color=COL, edgecolor=INK, lw=0.8, zorder=6)
    trails = [ax.plot([], [], color=COL[g], lw=2, zorder=4)[0] for g in range(4)]

    side = fig.add_axes([0.74, 0.05, 0.25, 0.8]); side.axis("off")
    for g in range(4):
        side.scatter([0.04], [0.95 - 0.06 * g], s=60, color=COL[g]); side.text(0.1, 0.95 - 0.06 * g, NAMES[g],
                                                                            va="center", fontsize=10.5)
    side.scatter([0.04], [0.66], s=110, facecolor="white", edgecolor=INK, lw=2)
    side.text(0.1, 0.66, "class centre, global model", va="center", fontsize=9.5)
    side.scatter([0.04], [0.60], s=150, marker="*", color=GREY, edgecolor=INK)
    side.text(0.1, 0.60, "class centre after calibration", va="center", fontsize=9.5)
    side.set_xlim(0, 1); side.set_ylim(0, 1)
    phase = side.text(0.0, 0.47, "", fontsize=12, fontweight="bold", color=INK, va="top", wrap=True)
    accv = side.text(0.0, 0.27, "", fontsize=30, fontweight="bold", color=INK, va="top")
    accl = side.text(0.0, 0.12, "", fontsize=10, color=GREY, va="top")

    NB = 14
    order = np.arange(len(show))
    frames = [("intro", 0)] * 8 + [("in", i) for i in range(1, NB + 1)] + [("in", NB)] * 8 + \
             [("cal", j) for j in range(len(lams)) for _ in range(2)] + [("cal", len(lams) - 1)] * 18

    def draw(n_show, j):
        pr = preds[j][show]
        idx = order[:n_show]
        good, bad = idx[pr[idx] == y_te[idx]], idx[pr[idx] != y_te[idx]]
        dots.set_offsets(P_te[good] if len(good) else np.empty((0, 2)))
        dots.set_color([COL[v] for v in y_te[good]])
        crosses.set_offsets(P_te[bad] if len(bad) else np.empty((0, 2)))
        crosses.set_color([COL[v] for v in y_te[bad]])

    def update(fi):
        kind, i = frames[fi]
        if kind == "intro":
            draw(0, 0); cm.set_alpha(0)
            phase.set_text(f"Global model: trained on\n18 other people")
            accv.set_text(""); accl.set_text("")
        elif kind == "in":
            draw(int(len(show) * i / NB), 0); cm.set_alpha(0)
            phase.set_text(f"New user (S{T:02d}) puts on\nthe band")
            accv.set_text(f"{accs[0]:.1f}%"); accv.set_color("#dc2626")
            accl.set_text(f"INT8 accuracy on all {len(ite):,}\nof S{T:02d}'s test windows")
        else:
            draw(len(show), i)
            cm.set_alpha(1); cm.set_offsets(mu_path[i])
            for g in range(4):
                trails[g].set_data([p[g, 0] for p in mu_path[:i + 1]], [p[g, 1] for p in mu_path[:i + 1]])
            phase.set_text(f"Calibrate with one recording\nper gesture: blend λ = {lams[i]:.2f}")
            accv.set_text(f"{accs[i]:.1f}%"); accv.set_color("#2ca02c" if i else "#dc2626")
            accl.set_text(f"INT8 accuracy on all {len(ite):,}\nof S{T:02d}'s test windows")
        return []

    save(FuncAnimation(fig, update, frames=len(frames)), "lda_space.gif", fps=7)


# ─── banner ──────────────────────────────────────────────────────────────────

def banner(mb):
    x = mb["x"]
    fig = plt.figure(figsize=(16, 4.2), dpi=100, facecolor=INK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(INK)
    sc = np.percentile(np.abs(x), 99.5, axis=0)
    n = len(x); xs = 0.40 + 0.6 * np.arange(n) / n
    for c in range(8):
        yy = 0.9 - 0.105 * c + 0.045 * np.clip(x[:, c] / sc[c], -1.2, 1.2)
        for g in range(4):
            sl = slice(g * 1960, (g + 1) * 1960)
            ax.plot(xs[sl], yy[sl], color=COL[g], lw=0.55, alpha=0.75)
    grad = np.linspace(1, 0, 256)[None, :] ** 1.6
    ax.imshow(np.dstack([np.full((1, 256, 3), matplotlib.colors.to_rgb(INK)), grad[..., None]]),
              extent=(0.38, 0.62, 0, 1), aspect="auto", zorder=3)
    ax.add_patch(Rectangle((0, 0), 0.385, 1, color=INK, zorder=3))
    ax.text(0.035, 0.74, "sEMG gesture classifier", fontsize=38, fontweight="bold", color="white", zorder=4, va="center")
    ax.text(0.035, 0.53, "on a tiny, fully integer chip", fontsize=25, color="#9fb6d8", zorder=4, va="center")
    ax.text(0.035, 0.31, "forearm EMG → add/compare/shift features → INT8 LDA → a gesture every 99.5 ms",
            fontsize=13.5, color="white", zorder=4, va="center")
    ax.text(0.035, 0.17, "bit-exact Verilog · 1-3 chained boards, 8-24 electrodes · B.Tech project, EE, IIT Bombay",
            fontsize=12, color="#9fb6d8", zorder=4, va="center")
    for g in range(4):
        ax.text(0.425 + 0.15 * g, 0.03, NAMES[g], color=COL[g], fontsize=12, fontweight="bold", zorder=4)
    fig.savefig(OUT / "banner.png", facecolor=INK)
    plt.close(fig)
    print(f"  banner.png                   {(OUT / 'banner.png').stat().st_size / 1e6:5.2f} MB")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    mb = load_mb()
    print(f"mb_data: {len(mb['x'])} samples, {len(mb['D'])} strides, decisions reproduce expected.txt")
    jobs = {"banner": lambda: banner(mb), "emg": lambda: emg_to_decision(mb), "stage1": lambda: stage1_features(mb),
            "chain": lambda: chain_serial_add(mb), "calib": calibration_bars, "lda": lda_space}
    for name, fn in jobs.items():
        if only in ("all", name) or (only == "fast" and name != "lda"):
            fn()
