// emg_shift_norm — Stage 2: bank combine + shift-normalisation to INT8.
//
// For every feature f = 5*ch + t  (ch = 0..C-1, t = 0..4: SABS, WL, ZC, SSC, POW):
//
//     raw_f  = bank0[ch].field_t + bank1[ch].field_t            (196-sample window sum)
//     x_q[f] = clip( ((raw_f - mu_f) + 2^(k_f - 1)) >>> k_f , -128, 127 ),   1 <= k_f <= 22
//
// written to the x_q register file in order f = 0, 1, ..., 5C-1, one per cycle.
//
// Pipeline (both memories have 1-cycle read latency):
//
//   ISSUE   cycle i   : norm_re, norm_addr = f;  s2_re, s2_addr = ch  only when t == 0
//   COMPUTE cycle i+1 : norm_mu/norm_k and s2_q0/s2_q1 are valid -> x_q[f] -> xq_we
//
//   cycle    : 0    1    2    3    4    5    6   ...  D   D+1
//   issue f  : 0    1    2    3    4    5    6   ... D-1
//   s2_re    : 1    0    0    0    0    1    0
//   compute f:      0    1    2    3    4    5   ... D-1
//   done     :                                          1
//
// The bank word is read once per channel: the RAM output holds while s2_re is low,
// so it stays valid for that channel's 5 features without a local register.
//
// Bit widths (per bank, 98 samples):
//   fields  SABS <= 98*128, WL <= 98*255, ZC, SSC <= 98, POW <= 98*2^14 < 2^21
//   raw_f   <= 2 * 98 * 2^14 = 3,211,264 < 2^22                         -> 22 bits unsigned
//   d       = raw_f - mu_f + 2^(k_f-1),  mu_f in [0, 2^22 - 1], k_f in [1, 22]
//           in [-(2^22 - 1) + 1, (2^22 - 1) + 2^21]  subset of [-2^23, 2^23 - 1] -> 24 bits signed
//   d >>> k_f has the same range (shift only shrinks it)                  -> 24 bits, then clip to 8
`default_nettype none

module emg_shift_norm #(
    parameter CMAX = 24
) (
    input  wire              clk,
    input  wire              rst_n,          // asynchronous, active low
    input  wire [4:0]        cfg_n_ch,       // C, 1..CMAX

    input  wire              start,          // 1-cycle pulse (Stage 1 stride_done); ignored while busy
    output reg               busy,
    output reg               done,           // 1-cycle pulse after the last x_q write

    // feature banks (inside emg_feat_accum)
    output wire              s2_re,
    output wire [4:0]        s2_addr,        // channel
    input  wire [63:0]       s2_q0,          // bank 0 word, 1 cycle after s2_re
    input  wire [63:0]       s2_q1,          // bank 1 word

    // normalisation memory {mu_f, k_f}
    output wire              norm_re,
    output wire [6:0]        norm_addr,      // feature index f
    input  wire [21:0]       norm_mu,        // 1 cycle after norm_re
    input  wire [4:0]        norm_k,

    // x_q register file for Stage 3
    output wire              xq_we,
    output wire [6:0]        xq_addr,
    output wire signed [7:0] xq_data
);
    wire [4:0] ch_last = cfg_n_ch - 5'd1;

    // ================================================================== issue stage
    reg       iss;          // issue stage active
    reg [4:0] ch;           // channel 0..C-1
    reg [2:0] t;            // feature type 0..4
    reg [6:0] f;            // feature index 0..5C-1 (counts, no 5*ch + t multiply)

    wire iss_last = iss && (ch == ch_last) && (t == 3'd4);

    assign norm_re   = iss;
    assign norm_addr = f;
    assign s2_re     = iss && (t == 3'd0);
    assign s2_addr   = ch;

    // ================================================================== compute stage
    reg       cv;           // compute stage valid  (= iss delayed one cycle)
    reg [2:0] ct;           // t delayed one cycle
    reg [6:0] cf;           // f delayed one cycle
    reg       c_last;       // this compute is the last feature

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;  done <= 1'b0;
            iss  <= 1'b0;  ch <= 5'd0;  t <= 3'd0;  f <= 7'd0;
            cv   <= 1'b0;  ct <= 3'd0;  cf <= 7'd0; c_last <= 1'b0;
        end else begin
            // start
            if (start && !busy) begin
                busy <= 1'b1;
                iss  <= 1'b1;
                ch   <= 5'd0;  t <= 3'd0;  f <= 7'd0;
            end

            // issue advance logic 
            if (iss) begin
                f <= f + 7'd1;
                if (t == 3'd4) begin
                    t  <= 3'd0;
                    ch <= ch + 5'd1; //next channel logic 
                end else begin
                    t  <= t + 3'd1; //next feature in that channel  
                end
                if (iss_last) iss <= 1'b0;
            end

            // issue -> compute
            cv     <= iss;
            ct     <= t;
            cf     <= f;
            c_last <= iss_last;

            // finish: done pulses in the cycle after the last write
            done <= c_last;
            if (c_last) busy <= 1'b0;
        end
    end

    // select field t from both bank words (bank word layout: see emg_feat_accum)
    reg [20:0] fld0, fld1;
    always @(*) begin
        case (ct)
            3'd0:    begin fld0 = s2_q0[13:0];  fld1 = s2_q1[13:0];  end   // SABS
            3'd1:    begin fld0 = s2_q0[28:14]; fld1 = s2_q1[28:14]; end   // WL
            3'd2:    begin fld0 = s2_q0[35:29]; fld1 = s2_q1[35:29]; end   // ZC
            3'd3:    begin fld0 = s2_q0[42:36]; fld1 = s2_q1[42:36]; end   // SSC
            default: begin fld0 = s2_q0[63:43]; fld1 = s2_q1[63:43]; end   // POW
        endcase
    end

    wire        [21:0] raw = fld0 + fld1;

    // Every operand is made signed (zero-extended first), so the whole expression is
    // signed and >>> is an arithmetic shift. Mixing in one unsigned operand would make
    // the expression unsigned and >>> would silently become a logical shift.
    wire signed [23:0] rnd = 24'sd1 << (norm_k - 5'd1);                    // 2^(k_f - 1)
    wire signed [23:0] d   = $signed({2'b00, raw}) - $signed({2'b00, norm_mu}) + rnd;
    wire signed [23:0] sh  = d >>> norm_k;

    assign xq_we   = cv;
    assign xq_addr = cf;
    assign xq_data = (sh > 24'sd127)  ?  8'sd127 :
                     (sh < -24'sd128) ? -8'sd128 : sh[7:0];
endmodule

`default_nettype wire
