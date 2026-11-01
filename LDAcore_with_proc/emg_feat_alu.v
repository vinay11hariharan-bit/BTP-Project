// emg_feat_alu — Stage 1 per-sample feature ALU (combinational, one channel per cycle).
//
// YOUR MODULE. Keep the port names and widths. Check with:
//     python tb/test_feat_alu.py my_rtl          (exhaustive: all 196,608 input combinations)
//
//   d1     = x[n] - x[n-1]
//   sabs   = |x[n]|                                          0..128
//   wl     = |d1|                                            0..255
//   zc     = 1 when x[n] and x[n-1] have different signs     (sign = MSB; 0 counts as positive)
//   ssc    = (d1 > 0 and d0 < 0) or (d1 < 0 and d0 > 0)       d0 = previous d1, given as 2 sign bits
//   pow    = Mitchell-1 square of sabs                        (instantiate emg_mitchell_sq)
//   d1_pos = (d1 > 0),  d1_neg = (d1 < 0)                     written back as the next d0
//
// Questions to answer before writing:
//   * What is the range of d1, and how many signed bits does it need? What goes wrong
//     if you subtract in 8 bits?
//   * |x| by conditional negation: what is |-128| in 8 bits, and is it still right if
//     sabs is read as unsigned?
//   * d1 = 0 must give d1_pos = d1_neg = 0. Does your d1_pos handle that?
`default_nettype none

module emg_feat_alu (
    input  wire signed [7:0] x,          // x[n]
    input  wire signed [7:0] x_prev,     // x[n-1]
    input  wire              d0_pos,     // previous difference > 0
    input  wire              d0_neg,     // previous difference < 0
    output wire [7:0]        sabs,
    output wire [7:0]        wl,
    output wire              zc,
    output wire              ssc,
    output wire [14:0]       pow,
    output wire              d1_pos,
    output wire              d1_neg
);
	wire signed [8:0] d1; 
	assign d1 = {x[7],x} - {x_prev[7],x_prev};
	assign sabs = (x[7])? (-x):x;
       	wire [8:0] d1_abs = (d1[8])? (-d1):d1;
	assign wl = d1_abs[7:0];
	assign zc = x[7]^x_prev[7];
	assign d1_neg = d1[8];
        assign d1_pos = (~d1[8]) & (|d1);
        assign ssc = (d1_pos & d0_neg) | (d1_neg & d0_pos);
        emg_mitchell_sq u_mitchell_sq (
       	 .a  (sabs),
       	 .sq (pow)
   	 );

endmodule

`default_nettype wire
