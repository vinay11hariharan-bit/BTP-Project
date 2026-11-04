"""Figures for the report. Every number below is copied from a script output listed in the
report's appendix (or from docs/technical_summary.md). Run: python docs/report/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
INK, ACC, MUTE = "#1f3b4d", "#c0392b", "#8a9aa5"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# 1. Gesture vocabulary
fig, ax = plt.subplots(figsize=(3.4, 1.9))
labels = ["Set A\n4 incl. Index", "Set B\n4 selected", "Set D\n6 selected", "Set C\nall 12"]
acc = [65.28, 79.38, 69.0, 48.8]
chance = [25, 25, 16.7, 8.3]
x = range(len(labels))
ax.bar(x, acc, color=[MUTE, ACC, INK, INK], width=0.6)
ax.scatter(x, chance, marker="_", s=300, color="k", zorder=3, label="chance")
for i, a in enumerate(acc):
    ax.text(i, a + 1.5, f"{a:.1f}", ha="center")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("INT8 LOSO accuracy (%)"); ax.set_ylim(0, 92); ax.legend(frameon=False, loc="upper right")
save(fig, "gesture_sets")

# 2. Classifier accuracy vs memory (Gesture Set B)
fig, ax = plt.subplots(figsize=(3.4, 2.1))
pts = [("MLP 64-16-4", 1168, 79.57), ("HistGB", 19200, 79.26), ("LogReg", 272, 79.19),
       ("Linear SVM", 272, 78.30), ("LDA", 272, 77.87), ("Random forest", 53145, 77.43),
       ("Extra trees", 38015, 76.04), ("Decision tree", 5055, 73.17)]
for name, b, a in pts:
    c = ACC if name == "LDA" else (MUTE if b > 10240 else INK)
    ax.scatter(b, a, color=c, s=18, zorder=3)
    ax.annotate(name, (b, a), xytext=(4, -3 if name != "LogReg" else 4), textcoords="offset points", fontsize=6.5)
ax.axvline(10240, color="k", lw=0.6, ls="--"); ax.text(11500, 73.4, "10 KB budget", fontsize=6.5)
ax.set_xscale("log"); ax.set_xlabel("Model memory (bytes, log)"); ax.set_ylabel("INT8 accuracy (%)")
ax.set_ylim(72.5, 80.5)
save(fig, "classifiers")

# 3. Cycle budget
fig, ax = plt.subplots(figsize=(3.4, 1.9))
C = [8, 12, 16, 20, 24]
meas = [290, 430, 570, 710, 850]
ax.plot(C, meas, "o-", color=ACC, label="Stage 2 + 3 (measured, RTL)")
ax.plot(C, [c for c in C], "s-", color=INK, ms=3, label="Stage 1 per sample")
ax.axhline(1015, color="k", ls="--", lw=0.8); ax.text(24, 1045, "sample period: 1015 cycles", fontsize=6.5, ha="right")
ax.set_xlabel("Channels C"); ax.set_ylabel("clk_sys cycles"); ax.set_xticks(C); ax.set_ylim(0, 1300)
ax.legend(frameon=False, loc="upper left", fontsize=6.5)
save(fig, "cycles")
print("figures in", OUT)

