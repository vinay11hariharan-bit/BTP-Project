`timescale 1ns/1ps
// fpga_clk - clocks for the FPGA test (Zynq-7000 PL)
//
// 125 MHz board clock -> MMCM -> 10 MHz (clk10, free running).
// clk_emg is the 10 MHz clock gated by a BUFGCE:
//   fast = 0 : one edge every 39 cycles -> 256.4 kHz, the chip's real clock (real-time run)
//   fast = 1 : every edge -> 10 MHz, same cycles 39x faster (whole stream in ~0.2 s)
// The chip only sees clock edges, so both modes give the same cycle-by-cycle behaviour.
// With SIM defined the primitives are skipped and clk_emg is the input clock.
module fpga_clk (
    input  wire clk125,
    input  wire fast,          // asynchronous switch
    output wire clk10,
    output wire clk_emg,
    output wire locked
);
`ifdef SIM
    assign clk10   = clk125;
    assign clk_emg = clk125;
    assign locked  = 1'b1;
`else
    wire fb, fb_buf, out0;

    // VCO = 125 MHz x 8 = 1000 MHz, CLKOUT0 = 1000 / 100 = 10 MHz
    MMCME2_BASE #(
        .CLKIN1_PERIOD    (8.000),
        .CLKFBOUT_MULT_F  (8.000),
        .DIVCLK_DIVIDE    (1),
        .CLKOUT0_DIVIDE_F (100.000)
    ) u_mmcm (
        .CLKIN1   (clk125),
        .CLKFBIN  (fb_buf),
        .CLKFBOUT (fb),
        .CLKOUT0  (out0),
        .LOCKED   (locked),
        .PWRDWN   (1'b0),
        .RST      (1'b0),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6()
    );
    BUFG u_fb  (.I(fb),   .O(fb_buf));
    BUFG u_c10 (.I(out0), .O(clk10));

    // enable for the gated clock, made in the clk10 domain
    reg [1:0] fast_s = 2'b00;
    reg [5:0] div    = 6'd0;
    reg       ce     = 1'b0;
    always @(posedge clk10) begin
        fast_s <= {fast_s[0], fast};
        div    <= (div == 6'd38) ? 6'd0 : div + 1'b1;
        ce     <= fast_s[1] | (div == 6'd38);
    end

    BUFGCE u_gate (.I(out0), .CE(ce), .O(clk_emg));
`endif
endmodule
