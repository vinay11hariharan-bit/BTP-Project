`timescale 1ns/1ps
// emg_clk_gate - the chip clock, made from the 10 MHz FCLK0 of the Zynq with a BUFGCE
//   fast = 0 : one edge in 39 -> 256.4 kHz, the chip's real clock (real time)
//   fast = 1 : every edge     -> 10 MHz, the same cycles 39x faster
// The chip only sees clock edges, so both modes give the same results.
// With SIM defined the clock passes straight through.
module emg_clk_gate (
    input  wire clk,           // 10 MHz
    input  wire fast,
    output wire clk_emg
);
`ifdef SIM
    assign clk_emg = clk;
`else
    reg [5:0] div = 6'd0;
    reg       ce  = 1'b0;
    always @(posedge clk) begin
        div <= (div == 6'd38) ? 6'd0 : div + 1'b1;
        ce  <= fast | (div == 6'd38);
    end
    BUFGCE u_gate (.I(clk), .CE(ce), .O(clk_emg));
`endif
endmodule
