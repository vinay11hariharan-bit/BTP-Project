# project.tcl - same design as build.tcl, but as a normal Vivado project for the GUI
#   Vivado: Tools > Run Tcl Script... > fpga/project.tcl   (or: vivado -source project.tcl)
# Then click "Generate Bitstream".
set here [file dirname [file normalize [info script]]]

create_project emg_fpga $here/vivado_proj -part xc7z020clg400-1 -force
add_files [glob $here/../RTL/*.v]
add_files [glob $here/rtl/*.v]
add_files [glob $here/mem/*.mem]
add_files -fileset constrs_1 $here/constraints/pynq.xdc
set_property top fpga_top [current_fileset]
update_compile_order -fileset sources_1
