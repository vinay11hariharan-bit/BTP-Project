#!/bin/bash
mkdir -p logs
run() {
  name=$1; shift
  printf "%-14s running... " "$name"
  "$@" > logs/$name.log 2>&1
  r=$(grep -oE 'PASS=[01] FAIL=[01]' logs/$name.log | head -1)
  echo "${r:-ERROR, see logs/$name.log}"
}
run 01_top        python tb/test_emg_top.py
run 02_1x8        python tb/test_emg_chain.py
run 03_3x8        python tb/test_emg_chain.py --pods 3
run 04_4x6        python tb/test_emg_chain.py --pods 4 --pod-ch 6
run 05_4x4        python tb/test_emg_chain.py --pods 4 --pod-ch 4
run 06_4x4_C14    python tb/test_emg_chain.py --pods 4 --pod-ch 4 --channels 14
run 07_6x4        python tb/test_emg_chain.py --pods 6 --pod-ch 4
run 08_C1         python tb/test_emg_chain.py --channels 1
run 09_td32       python tb/test_emg_chain.py --t-d 32
run 10_cfg0       python tb/test_emg_chain.py --pods 2 --cfg-dev-ch 0
run 11_shift20    python tb/test_emg_chain.py --shift 20
run 12_lo_unused  python tb/test_emg_chain.py --pods 2 --channels 12 --leadoff 14
run 13_miss16     python tb/test_emg_chain.py --pods 1 --channels 16
run 14_miss20     python tb/test_emg_chain.py --pods 2 --channels 20
run 15_cfg8       python tb/test_emg_chain.py --pods 3 --pod-ch 4 --cfg-dev-ch 8
run 16_cfg4       python tb/test_emg_chain.py --pods 2 --cfg-dev-ch 4
run 17_faults     python tb/test_emg_chain.py --pods 3 --leadoff 4,9 --stuck 16,17 --quiet 2
run 18_lo_frames  python tb/test_emg_chain.py --leadoff 3 --leadoff-frames 200-210
run 19_drop       python tb/test_emg_chain.py --drop-frame 150
run 20_nodrdy     python tb/test_emg_chain.py --no-drdy
run 21_late       python tb/test_emg_chain.py --late-drdy
run 22_badc       python tb/test_emg_chain.py --bad-c
run 23_td130_FAIL python tb/test_emg_chain.py --t-d 130
run 24_gap3_FAIL  python tb/test_emg_chain.py --cmd-gap 3
