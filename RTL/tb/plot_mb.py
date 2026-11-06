"""Pictures of a multi-board run (GTKWave style). From RTL/, after python tb/run_mb.py 4,4:
    python tb/plot_mb.py waveform/mb_4_4.vcd ../docs/waveforms
Draws one stride (stride 21: local Stage 3, barrier, serial add, decision) and the whole run.
"""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_waveforms as P

def parse(vcd):
    ids, path = {}, []
    f = open(vcd)
    for line in f:
        t = line.split()
        if not t: continue
        if t[0] == "$scope": path.append(t[2])
        elif t[0] == "$upscope": path.pop()
        elif t[0] == "$var": ids.setdefault(t[3], []).append((".".join(path[1:] + [t[4]]), int(t[2])))
        elif t[0] == "$enddefinitions": break
    sig = collections.defaultdict(list); wid = {n: w for v in ids.values() for n, w in v}; T = 0
    for line in f:
        c = line[0]
        if c == "#": T = int(line[1:]); continue
        if c in "01xz":
            for n, _ in ids.get(line[1:].strip(), []): sig[n].append((T, c))
        elif c == "b":
            v, k = line[1:].split()
            for n, _ in ids.get(k, []): sig[n].append((T, v))
    # split vectors of the tb into bits
    for n in list(sig):
        if "." not in n and wid[n] in (2, 3) and n in ("convst", "cs_n", "rdy", "dat", "vld", "cls_valid"):
            for b in range(wid[n]):
                bn = f"{n}[{b}]"; wid[bn] = 1
                sig[bn] = [(t, v.zfill(wid[n])[-1 - b] if len(v) > 1 or v in "01" else v) for t, v in sig[n]]
    return dict(sig), wid

S, W = parse(sys.argv[1])
P.S, P.WID = S, W
P.OUTDIR = Path(sys.argv[2]); P.OUTDIR.mkdir(exist_ok=True)
def rises(n): return [t for t, v in S[n] if v == "1"]
dec = rises("brd[0].u_chip.u_link.dec_valid"); sd = rises("brd[0].u_chip.stride_done")
k = 21
t_dec = dec[k] / 1e6
t_sd = max(t for t in sd if t / 1e6 < t_dec) / 1e6
b0, b1 = "brd[0].u_chip.", "brd[1].u_chip."
P.draw("mb_stride.png", "Two boards (4 + 4 channels), one stride: local Stage 3, barrier, serial add, decision at the head",
       [("convst[0]", "CONVST (shared)"), ("cs_n[0]", "board0 CS_N"), ("cs_n[1]", "board1 CS_N"),
        (b0 + "stride_done", "board0 stride_done"), (b0 + "mac_busy", "board0 Stage 3"), (b0 + "score", "board0 partial"),
        (b1 + "stride_done", "board1 stride_done"), (b1 + "mac_busy", "board1 Stage 3"), (b1 + "score", "board1 partial"),
        ("rdy[0]", "board0 rdy"), ("rdy[1]", "board1 rdy"), ("all_rdy", "all_rdy (wired-AND)"),
        ("vld[1]", "UP valid  1->0"), ("dat[1]", "UP data   1->0"),
        (b0 + "u_link.best", "head best total"), (b0 + "u_link.dec_valid", "head decision"), (b0 + "class_out", "class_out")],
       t_sd - 40, t_dec + 40, unit="us")
P.draw("mb_overview.png", "Two boards on real FORS-EMG (subject 6, trial 1): 8 s, 79 decisions",
       [("convst[0]", "CONVST"), ("all_rdy", "all_rdy"), (b0 + "u_link.dec_valid", "decision"), (b0 + "class_out", "class_out"), ("si", "sample index")],
       0, 8100, unit="ms")
print("ok", t_sd, t_dec)
