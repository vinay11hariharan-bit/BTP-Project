# build.tcl - makes emg_fpga.bit for the PYNQ-Z1/Z2, no project and no GUI needed
#   cd fpga
#   vivado -mode batch -source build.tcl
# Reports and the bitstream go to fpga/out/.
set here [file dirname [file normalize [info script]]]
set out  $here/out
file mkdir $out

read_verilog [glob $here/../RTL/*.v]
read_verilog [glob $here/rtl/*.v]
read_mem     [glob $here/mem/*.mem]
read_xdc     $here/constraints/pynq.xdc

synth_design -top fpga_top -part xc7z020clg400-1
opt_design
place_design
route_design

report_utilization    -file $out/utilization.txt
report_timing_summary -file $out/timing.txt
write_bitstream -force $out/emg_fpga.bit
puts "Bitstream: $out/emg_fpga.bit"
