// emg_mitchell_sq — Mitchell-1 approximate square of a in [0, 128] (combinational).
//
// YOUR MODULE. Keep the ports. Tested through emg_feat_alu:
//     python tb/test_feat_alu.py my_rtl
//
//   p  = position of the leading one of a          (a = 0b0010_1101 -> p = 5)
//   sq = (a << (p+1)) - (1 << 2p)
//   sq = 0 for a = 0
//
// Questions to answer before writing:
//   * How wide must a << (p+1) be for the largest input, a = 128?
//   * What does the formula give for a = 0 if you don't special-case it?
//   * What is the largest sq, and why do 15 output bits suffice?
`default_nettype none

module emg_mitchell_sq (
    input  wire [7:0]  a,      // |x|, 0..128
    output wire [14:0] sq
);

    reg  [2:0]  p;
    wire [3:0]  shift_a  = {1'b0, p} + 4'd1; // 4-bit to hold p+1 up to 8 (when p=7)
    wire [3:0]  shift_2p = {p, 1'b0};        // 2*p up to 14 (when p=7)
    wire [15:0] term1    = ({8'd0, a} << shift_a);
    wire [15:0] term2    = (16'd1 << shift_2p);
    wire [15:0] diff     = term1 - term2;

    always @(*) begin
        casez (a)
            8'b1???_????: p = 3'd7;
            8'b01??_????: p = 3'd6;
            8'b001?_????: p = 3'd5;
            8'b0001_????: p = 3'd4;
            8'b0000_1???: p = 3'd3;
            8'b0000_01??: p = 3'd2;
            8'b0000_001?: p = 3'd1;
            8'b0000_0001: p = 3'd0;
            default:      p = 3'd0;
        endcase
    end

    assign sq = (a != 8'd0) ? diff[14:0] : 15'd0;
endmodule

`default_nettype wire
