"""Run one trial on the FPGA and compare with the golden model (runs on the PYNQ).

Put emg_fpga.bit, emg_fpga.hwh, this file and the trial .npz in one folder on the board, then
    sudo python3 run_trial.py trial_s6_t1.npz            # fast: 10 MHz, a few seconds
    sudo python3 run_trial.py trial_s6_t1.npz --real     # real time: 256 kHz, ~33 s for 32000 samples
or in Jupyter:  %run run_trial.py trial_s6_t1.npz
"""
import sys
import time

import numpy as np
from pynq import MMIO, Overlay

NAMES = ["Hand Close", "Hand Open", "Wrist Flex", "Wrist Ext"]
CTRL, STATUS, NSAMP, NCFG, NWANT, ID = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x1C
CFG, DEC, SAMPLES = 0x1000, 0x2000, 0x40000


class EmgFpga:
    def __init__(self, bit="emg_fpga.bit"):
        self.ol = Overlay(bit)                   # also sets FCLK0 to 10 MHz (from the .hwh)
        base = self.ol.ip_dict["emg"]["phys_addr"] if "emg" in self.ol.ip_dict else 0x40000000
        self.mm = MMIO(base, 0x80000)
        assert self.mm.read(ID) == 0x454D4701, "emg_axi not found: wrong bitstream?"

    def run(self, samples, cfg, nwant, fast=True, timeout=None):
        mm = self.mm
        mm.write(CTRL, 0)                        # hold the chips in reset while loading
        for i, (brd, addr, data) in enumerate(cfg):
            mm.write(CFG + 8 * i, int(data))
            mm.write(CFG + 8 * i + 4, (int(brd) << 12) | int(addr))
        u = samples.astype(np.int8).view(np.uint8).astype(np.uint32)
        hi = (u[:, 0] << 24) | (u[:, 1] << 16) | (u[:, 2] << 8) | u[:, 3]
        lo = (u[:, 4] << 24) | (u[:, 5] << 16) | (u[:, 6] << 8) | u[:, 7]
        for i in range(len(u)):
            mm.write(SAMPLES + 8 * i, int(hi[i]))
            mm.write(SAMPLES + 8 * i + 4, int(lo[i]))
        mm.write(NSAMP, len(u))
        mm.write(NCFG, len(cfg))
        mm.write(NWANT, nwant)

        t0 = time.time()
        mm.write(CTRL, 3 if fast else 1)
        if timeout is None:
            timeout = 5 + len(u) / 984.6 * (0.1 if fast else 1.5)
        while True:
            st = mm.read(STATUS)
            if st & 2 or time.time() - t0 > timeout:
                break
            time.sleep(0.05)
        dt = time.time() - t0
        n = (st >> 16) & 0x3FF
        words = [mm.read(DEC + 4 * i) for i in range(n)]
        cls = np.array([w & 7 for w in words])
        at = np.array([w >> 16 for w in words])
        return cls, at, {"done": bool(st & 2), "overrun": bool(st & 4), "config_done": bool(st & 1),
                         "seconds": dt}


def main():
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    fast = "--real" not in sys.argv
    t = np.load(path)
    golden, true, mixed = t["golden"], t["true"], t["mixed"].astype(bool)

    print(f"subject {int(t['subject'])} trial {int(t['trial'])}: {len(t['samples'])} samples, "
          f"{len(t['cfg'])} config writes, {len(golden)} decisions expected, "
          f"{'fast (10 MHz)' if fast else 'real time (256 kHz)'}")
    fpga = EmgFpga()
    t0 = time.time()
    cls, at, st = fpga.run(t["samples"], t["cfg"], len(golden), fast)
    print(f"loaded and ran in {time.time() - t0:.1f} s (run {st['seconds']:.1f} s): {st}")

    n = min(len(cls), len(golden))
    print("\n  #   sample  time_s  fpga         golden       true")
    for i in range(n):
        mark = "" if cls[i] == golden[i] else "   <-- differs"
        print(f"{i:3d} {at[i]:8d} {at[i] / 984.6:7.2f}  {NAMES[cls[i]]:12s} {NAMES[golden[i]]:12s} "
              f"{NAMES[true[i]]:12s}{' (mixed)' if mixed[i] else ''}{mark}")

    same = int(np.sum(cls[:n] == golden[:n]))
    full = ~mixed[:n]
    print(f"\n{n}/{len(golden)} decisions read, {same} equal to the golden model"
          f"{'  -> BIT-EXACT' if same == n == len(golden) and not st['overrun'] else '  -> CHECK'}")
    print(f"accuracy on single-gesture windows: FPGA {np.mean(cls[:n][full] == true[:n][full]) * 100:.1f}%, "
          f"golden {np.mean(golden[:n][full] == true[:n][full]) * 100:.1f}%")


if __name__ == "__main__":
    main()
