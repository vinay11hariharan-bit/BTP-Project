"""Figures of the paper that are not drawn by docs/report/make_figures.py.

The numbers are the results quoted in the paper (docs/results/). The system, board and waveform pictures are
crops of the presentation slides; that step runs only if the slide renders are given in DECK.
Run from this folder:  python make_paper_figures.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "legend.fontsize": 7, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
INK, ACC, MUTE, TEAL, GOLD = "#1f3b4d", "#c0392b", "#9aa8b1", "#1a8a7a", "#d4a017"
D = Path(__file__).resolve().parent
OUT = D / "figures"; OUT.mkdir(exist_ok=True)
DECK = D / "slides"          # optional: MultiBoard_System.png, MultiBoard_OneBoard.png, pg-19..21.png


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)


# ---- merge vs drop
names = ["Drop to 4\ngestures", "4 groups of 3\n(4-class LDA)", "4 groups of 3\n(12-class, mapped)", "Earlier 6-group\nmap (6-class)", "Earlier 6-group\nmap (12-class)", "12 gestures\n(no merging)"]
m = [77.82, 60.91, 63.07, 72.38, 70.71, 46.41]; ci = [4.10, 4.55, 5.07, 4.11, 4.62, 5.55]
chance = [25, 25, 25, 50, 50, 8.3]
fig, ax = plt.subplots(figsize=(7.0, 2.4))
x = np.arange(len(m))
ax.bar(x, m, 0.6, yerr=ci, capsize=3, color=[ACC] + [INK] * 4 + [MUTE], error_kw=dict(lw=0.8))
for i, (v, c) in enumerate(zip(m, chance)):
    ax.text(i, v + ci[i] + 1.5, f"{v:.1f}", ha="center", fontsize=7)
    ax.plot([i - 0.3, i + 0.3], [c, c], color="k", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7); ax.set_ylim(0, 95); ax.set_ylabel("held-out accuracy (%)")
ax.set_title("Merging gestures into classes vs dropping gestures (19 subjects, chip INT8 LDA)")
ax.text(-0.4, 90, "black line: chance (majority class for the 6-group map)", fontsize=6.5, ha="left")
save(fig, "merge_drop")

# ---- CNN vs LDA per subject (user's 150-epoch LOSO run, INT8)
lda = [87.38, 65.9, 66.23, 76.08, 67.88, 86.77, 72.33, 81.23, 79.92, 79.81, 85.6, 82.92, 67.31, 84.04, 62.19, 70.06, 91.54, 80.52, 90.38]
sep = [83.81, 71.02, 66.77, 68.06, 73.31, 89.38, 79, 85.27, 77.92, 83.08, 80.5, 86.02, 69.54, 84.73, 71.73, 72.94, 93.08, 72.25, 94.48]
two = [87.19, 75.44, 68.23, 74.92, 77.48, 88.27, 78.35, 81.92, 81.48, 84.33, 79.88, 86.94, 70.58, 86.38, 69.27, 72.94, 94.33, 76.12, 95.04]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
o = np.argsort(lda); xs = np.arange(19)
a.plot(xs, np.array(lda)[o], "o-", color=INK, ms=3, lw=1, label="LDA (160 MACs)")
a.plot(xs, np.array(sep)[o], "s", color=GOLD, ms=3, label="CNN option 2 (751,584 MACs)")
a.plot(xs, np.array(two)[o], "^", color=ACC, ms=3, label="CNN option 5c (594,400 MACs)")
a.set_xticks(xs); a.set_xticklabels([f"S{i + 1:02d}" for i in o], rotation=90, fontsize=6)
a.set_ylabel("held-out accuracy (%)"); a.set_title("Per subject, sorted by the LDA"); a.legend(loc="lower right")
for name, macs, acc, col in [("LDA", 160, 77.79, INK), ("CNN 5c", 594400, 80.48, ACC), ("CNN 2", 751584, 79.10, GOLD)]:
    b.scatter(macs, acc, color=col, s=25, zorder=3); b.annotate(name, (macs, acc), xytext=(4, -9), textcoords="offset points", fontsize=7)
b.set_xscale("log"); b.set_xlim(60, 3e6); b.set_ylim(74, 83); b.set_xlabel("MACs per decision"); b.set_ylabel("mean accuracy (%)")
b.set_title("Accuracy against cost")
save(fig, "cnn_lda")

# ---- calibration
L = {"S02": [69.17, 80.38], "S03": [66.39, 84.20], "S04": [76.91, 79.97], "S05": [68.61, 83.12], "S06": [87.78, 91.39], "S07": [71.46, 82.01],
     "S08": [80.45, 85.17], "S09": [82.71, 87.81], "S10": [81.11, 81.98], "S11": [86.91, 87.26], "S12": [85.31, 88.72], "S13": [67.29, 87.36],
     "S14": [86.15, 88.61], "S15": [59.34, 89.48], "S16": [69.86, 75.97], "S17": [90.07, 92.53], "S18": [83.99, 93.96], "S19": [91.08, 94.41]}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
k = sorted(L, key=lambda s: L[s][0]); xs = np.arange(len(k))
a.vlines(xs, [L[s][0] for s in k], [L[s][1] for s in k], color=MUTE, lw=1.5)
a.plot(xs, [L[s][0] for s in k], "o", color=INK, ms=3.5, label="global (LOSO) model")
a.plot(xs, [L[s][1] for s in k], "o", color=ACC, ms=3.5, label="after 1 calibration trial")
a.set_xticks(xs); a.set_xticklabels(k, rotation=90, fontsize=6); a.set_ylabel("accuracy on trials 3-5 (%)")
a.set_title("INT8 LDA, 4 gestures, 18 subjects"); a.legend(loc="lower right")
cats = ["LDA\n(model C)", "CNN\nsmall", "CNN\ntwo"]; before = [71.4, 75.4, 78.1]; after = [91.9, 92.2, 94.3]
xx = np.arange(3)
b.bar(xx - 0.18, before, 0.36, color=MUTE, label="no calibration"); b.bar(xx + 0.18, after, 0.36, color=INK, label="1 trial")
for i in range(3):
    b.text(i - 0.18, before[i] + 0.8, f"{before[i]:.1f}", ha="center", fontsize=6.5); b.text(i + 0.18, after[i] + 0.8, f"{after[i]:.1f}", ha="center", fontsize=6.5)
b.set_xticks(xx); b.set_xticklabels(cats); b.set_ylim(60, 100); b.set_ylabel("group accuracy (%)")
b.set_title("CNN checkpoint (6 groups)"); b.legend(loc="lower right")
save(fig, "calibration")

# ---- timing at 256 kHz
fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 2.3), gridspec_kw=dict(height_ratios=[1, 1]))
for ax in (a, b):
    ax.grid(False); ax.set_yticks([]); ax.spines["left"].set_visible(False)
a.broken_barh([(0, 12)], (0.55, 0.35), color=GOLD); a.broken_barh([(12, 128)], (0.55, 0.35), color=MUTE); a.broken_barh([(140, 8)], (0.55, 0.35), color=INK)
a.broken_barh([(140, 8), (153, 42)], (0.05, 0.35), color=INK)
a.text(76, 0.72, "SPI read: 64 bits x 2 cycles", ha="center", va="center", fontsize=7, color="white")
a.text(150, 0.72, " Stage 1 (8)", va="center", fontsize=7); a.text(174, 0.22, "Stage 2 (42)", ha="center", va="center", fontsize=7, color="white")
a.text(-2, 0.72, "every sample", ha="right", va="center", fontsize=7); a.text(-2, 0.22, "98th sample", ha="right", va="center", fontsize=7)
a.axvline(260, color="k", lw=0.8, ls="--"); a.text(258, 1.0, "next CONVST", ha="right", fontsize=6.5)
a.set_xlim(-45, 265); a.set_ylim(0, 1.1); a.set_xlabel("clock cycles within one sample (260 cycles = 1.016 ms at 256 kHz)")
segs = [(0, 8, INK, ""), (10, 42, INK, "Stage 2"), (52, 125, INK, "Stage 3: 3 x (40 + 1) + 2"), (177, 4, GOLD, ""), (181, 100, TEAL, "chain: 96 + N + 2")]
for s0, w, c, t in segs:
    b.broken_barh([(s0, w)], (0.3, 0.45), color=c)
    if t: b.text(s0 + w / 2, 0.52, t, ha="center", va="center", fontsize=7, color="white")
b.text(285, 0.52, " decision", va="center", fontsize=7, color=ACC)
b.set_xlim(-45, 330); b.set_ylim(0, 1.0); b.set_xlabel("clock cycles after the last sample of a stride (8 channels per board)")
b.text(-2, 0.52, "per stride", ha="right", va="center", fontsize=7)
save(fig, "timing")

# ---- crop deck pictures (system, one board, waveforms)
def crop(src, dst, box):
    im = Image.open(src); w, h = im.size
    im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).save(OUT / dst)
if DECK.exists():
  crop(DECK / "MultiBoard_System.png", "system.png", (0.0, 0.122, 1.0, 1.0))
  crop(DECK / "MultiBoard_OneBoard.png", "board.png", (0.0, 0.14, 1.0, 1.0))
  for i, n in [(19, "wave_startup.png"), (20, "wave_strides.png"), (21, "wave_stride.png")]:
      crop(DECK / f"pg-{i:02d}.png", n, (0.0, 0.175, 1.0, 0.862))
print("ok")
