## PYNQ-Z1 / PYNQ-Z2 (same pins on both boards), xc7z020clg400-1
## The clock comes from the Zynq (FCLK0, 10 MHz), so only the LEDs need pins.

## LD0 decision (toggles), LD1 config done, LD2 done, LD3 overrun
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN N16 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports {led[3]}]
set_false_path -to [get_ports {led[*]}]
