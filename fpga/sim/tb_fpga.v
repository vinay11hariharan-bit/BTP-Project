`timescale 1ns/1ps
// tb_fpga - runs fpga_top (compiled with -DSIM) on the real stream and checks the LEDs.
// From fpga/mem:  iverilog -g2005 -DSIM -s tb_fpga -o tb.vvp ../../RTL/*.v ../rtl/*.v ../sim/tb_fpga.v
//                 vvp -n tb.vvp
module tb_fpga;
    reg clk = 1'b0;
    always #50 clk = ~clk;                    // 10 MHz: the "fast" clock of the board
    reg  [1:0] btn = 2'b01;                   // hold reset first
    wire [3:0] led;
    wire r, g, b;

    fpga_top dut (.sysclk(clk), .btn(btn), .sw(2'b01), .led(led), .led4_r(r), .led4_g(g), .led4_b(b));

    initial begin
        #2000 btn = 2'b00;
        wait (led[1] || led[3]);              // done or fail
        #1000;
        if (led[2] && !led[3] && !dut.sim_bad) $display("FPGA_TOP PASS: %0d decisions equal to the golden model", dut.NDEC);
        else                   $display("FPGA_TOP FAIL: led = %b", led);
        $finish;
    end
    initial begin
        #400_000_000;                          // 0.4 s of 10 MHz is far more than needed
        $display("FPGA_TOP TIMEOUT: led = %b, decisions %0d", led, dut.ndec);
        $finish;
    end
endmodule
