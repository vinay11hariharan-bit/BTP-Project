# build.tcl - Vivado project + block design (Zynq PS + emg_axi) + bitstream, in one go
#   GUI:          Tools > Run Tcl Script... > fpga/build.tcl
#   command line: cd fpga ; vivado -mode batch -source build.tcl
# Output in fpga/out/: emg_fpga.bit and emg_fpga.hwh (copy BOTH to the PYNQ), reports.
set here [file dirname [file normalize [info script]]]
set proj $here/vivado_proj
set out  $here/out
file mkdir $out

create_project emg_fpga $proj -part xc7z020clg400-1 -force
add_files [glob $here/../RTL/*.v]
add_files [glob $here/rtl/*.v]
add_files -fileset constrs_1 $here/constraints/pynq.xdc
update_compile_order -fileset sources_1

# ---- block design: Zynq PS, 10 MHz FCLK0, M_AXI_GP0 -> emg_axi (RTL module reference)
create_bd_design emg
set ps [create_bd_cell -type ip -vlnv [get_ipdefs -filter {NAME == processing_system7}] ps7]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {10}] $ps
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} $ps

create_bd_cell -type module -reference emg_axi emg
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/ps7/M_AXI_GP0} Slave {/emg/s_axi} intc_ip {New AXI Interconnect} master_apm {0}} \
    [get_bd_intf_pins emg/s_axi]

create_bd_port -dir O -from 3 -to 0 led
connect_bd_net [get_bd_pins emg/led] [get_bd_ports led]

assign_bd_address
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]] {
    set_property range  512K       $seg
    set_property offset 0x40000000 $seg
}
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files emg.bd] -top]
add_files -norecurse $wrapper
set_property top emg_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---- synthesis, implementation, bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "implementation failed, see the Messages tab / $proj/emg_fpga.runs/impl_1/runme.log"
}
open_run impl_1
report_utilization    -file $out/utilization.txt
report_timing_summary -file $out/timing.txt

file copy -force [lindex [glob $proj/emg_fpga.runs/impl_1/*.bit] 0] $out/emg_fpga.bit
set hwh [glob -nocomplain $proj/emg_fpga.gen/sources_1/bd/emg/hw_handoff/emg.hwh \
                           $proj/emg_fpga.srcs/sources_1/bd/emg/hw_handoff/emg.hwh]
file copy -force [lindex $hwh 0] $out/emg_fpga.hwh
puts "Done: $out/emg_fpga.bit and $out/emg_fpga.hwh"
