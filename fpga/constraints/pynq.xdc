## PYNQ-Z1 / PYNQ-Z2 (same pins on both boards), xc7z020clg400-1

## 125 MHz clock
set_property -dict {PACKAGE_PIN H16 IOSTANDARD LVCMOS33} [get_ports sysclk]
create_clock -name sysclk -period 8.000 [get_ports sysclk]

## buttons
set_property -dict {PACKAGE_PIN D19 IOSTANDARD LVCMOS33} [get_ports {btn[0]}]
set_property -dict {PACKAGE_PIN D20 IOSTANDARD LVCMOS33} [get_ports {btn[1]}]

## switches
set_property -dict {PACKAGE_PIN M20 IOSTANDARD LVCMOS33} [get_ports {sw[0]}]
set_property -dict {PACKAGE_PIN M19 IOSTANDARD LVCMOS33} [get_ports {sw[1]}]

## LEDs
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN N16 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

## RGB LED 4
set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports led4_r]
set_property -dict {PACKAGE_PIN G17 IOSTANDARD LVCMOS33} [get_ports led4_g]
set_property -dict {PACKAGE_PIN L15 IOSTANDARD LVCMOS33} [get_ports led4_b]

## buttons and switches are slow and asynchronous; LEDs need no timing
set_false_path -from [get_ports {btn[*] sw[*]}]
set_false_path -to   [get_ports {led[*] led4_*}]
