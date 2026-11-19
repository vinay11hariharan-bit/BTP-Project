# FPGA test: two-board LDA classifier on a PYNQ-Z1 / Z2

Two copies of the chip (`RTL/emg_top.v`) run on one FPGA as a head + tail chain (4 + 4 channels).
Everything else is on the FPGA too, so no PC, ADC or Python is needed while it runs:

- the 171 configuration writes (LDA weights, bias, settings) from a ROM
- 7840 real FORS-EMG samples (subject 6, trial 1) from a ROM, sent over SPI by a fake ADC per board
- the 79 decisions of the Python golden model, compared with the chip's output

The design uses only the FPGA fabric (no Zynq PS, no block design).

## Files

| Path | What it is |
|---|---|
| `rtl/fpga_top.v` | top level: ROMs, config loader, the two chips, checker, LEDs |
| `rtl/fpga_clk.v` | 125 MHz → 10 MHz (MMCM), gated down to the chip's 256 kHz |
| `rtl/fpga_pod.v` | fake 8-channel ADC (same SPI timing as the testbench model) |
| `mem/*.mem` | config writes, samples, expected decisions |
| `constraints/pynq.xdc` | pins of the PYNQ-Z1/Z2 |
| `build.tcl` | builds the bitstream from the command line |
| `project.tcl` | makes a normal Vivado project instead |
| `sim/tb_fpga.v` | simulation of the whole thing (Icarus Verilog) |

The chip RTL is read from `../RTL`, so keep the whole repo, not just this folder.

## Build (Vivado 2020.x or newer, any edition, part xc7z020clg400-1)

Command line:

```
cd fpga
vivado -mode batch -source build.tcl
```

The bitstream is `fpga/out/emg_fpga.bit`. Timing and resource reports are in `fpga/out/`.

GUI: Tools → Run Tcl Script → `fpga/project.tcl`, then Generate Bitstream.

The design is small: about 2k LUTs, 2k flip-flops, 29 BRAM18 and 2 DSPs.

## Load

- Vivado Hardware Manager: Open Target → Auto Connect → Program Device → `emg_fpga.bit`
- or on the PYNQ, from Jupyter (copy the .bit to the board first):

```python
from pynq import Bitstream
Bitstream("/home/xilinx/emg_fpga.bit").download()
```

## Run and read the result

| Control | Function |
|---|---|
| BTN0 | restart the run (it also starts by itself after loading) |
| SW0 | 0 = real time (256 kHz, one decision every 99.5 ms, about 8 s in total); 1 = fast (10 MHz, about 0.2 s) |

| LED | Meaning |
|---|---|
| LD0 | toggles at every decision |
| LD1 | all 79 decisions done |
| LD2 | **PASS**: done and every decision equals the golden model |
| LD3 | **FAIL**: a decision differed, or the chip reported overrun |
| LD4 (RGB) | current gesture: red = Hand Close, green = Hand Open, blue = Wrist Flexion, white = Wrist Extension |

A good run ends with LD1 and LD2 on and LD3 off.
If SW0 is changed during a run, press BTN0 to restart.

## Simulate first (optional)

```
cd fpga/mem
iverilog -g2005 -DSIM -s tb_fpga -o tb.vvp ../../RTL/*.v ../rtl/*.v ../sim/tb_fpga.v
vvp -n tb.vvp
```

It prints each decision and ends with `FPGA_TOP PASS: 79 decisions equal to the golden model`.
