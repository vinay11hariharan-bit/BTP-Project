// emg_shift_norm — Stage 2: bank combine + shift-normalisation to INT8.
//
// For every feature f = 5*ch + t  (ch = 0..C-1, t = 0..4: SABS, WL, ZC, SSC, POW):
//
//     raw_f  = sum of field_t over the used banks of channel ch   (window = last N banks, B*N samples)
//     d_f    = raw_f - mu_f                                       (mu_f: 24 bits, 0 .. 2^24 - 1)
//     k_f > 0:  x_q[f] = clip( (d_f + 2^(k_f - 1)) >>> k_f , -128, 127 )   shrink, rounded
//     k_f <= 0: x_q[f] = clip( d_f << -k_f , -128, 127 )                   stretch a small-range feature
//     k_f is a signed 6-bit number, -32 .. 31.
//
// Step 2 (1 Nov 2026): mu_f widened from 22 to 24 bits and k_f made signed (was 1 .. 22), as in
// the golden model's requant() and the agreed design (implementation plan, item 7).
//
// Step 1 (31 Oct 2026): up to 4 banks are read in parallel; bank i is counted only if
// bank_use[i] = 1 (completed since reset and i < N). Banks not yet filled count as 0.
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
// Bit widths (B <= 100 samples per bank, N <= 4 banks):
//   fields  SABS <= 100*128, WL <= 100*255, ZC, SSC <= 100, POW <= 100*2^14 < 2^21
//   raw_f   <= 4 * 100 * 2^14 = 6,553,600 < 2^23                       -> 23 bits unsigned
//   d_f     = raw_f - mu_f in [-(2^24 - 1), 2^23 - 1]                -> fits 26 bits signed
//   k > 0:  d_f + 2^(k-1) with k <= 31: |.| < 2^24 + 2^30 < 2^33     -> 34 bits signed, then >>> k
//   k <= 0: the left shift by s = -k (0 .. 32) is never formed when it would overflow: the result
//           is outside -128..127 exactly when d_f > (127 >> s) or d_f < -(128 >> s), and then the
//           clipped value is given directly. Otherwise |d_f| <= 128 >> s, so d_f << s fits 8 bits.
//
// Dead-channel check: while computing the WL feature (t = 1) of channel ch, dead_mask[ch]
// is set to (raw WL window sum == 0), i.e. the 8-bit value did not change once in the
// window. A missing channel (no electrode or pod, chain input tied to a fixed level) or a
// stuck one (saturated amplifier) gives the same value every sample, so WL = 0. This assumes
// the noise on a connected electrode moves the 8-bit value at least once per window: check
// that on real recordings at the chosen front-end gain. dead_mask is cleared at start, so
// bits C .. CMAX-1 stay 0, and it is valid from done until the next start.
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
    input  wire [63:0]       s2_q2,          // bank 2 word
    input  wire [63:0]       s2_q3,          // bank 3 word
    input  wire [3:0]        bank_use,       // bank i is part of the window (stable while busy)

    // normalisation memory {mu_f, k_f}
    output wire              norm_re,
    output wire [6:0]        norm_addr,      // feature index f
    input  wire [23:0]       norm_mu,        // 1 cycle after norm_re
    input  wire signed [5:0] norm_k,         // -32 .. 31

    // x_q register file for Stage 3
    output wire              xq_we,
    output wire [6:0]        xq_addr,
    output wire signed [7:0] xq_data,

    // dead-channel flags, one per channel, valid from done until the next start
    output reg  [CMAX-1:0]   dead_mask
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
    reg [4:0] cch;          // ch delayed one cycle
    reg       c_last;       // this compute is the last feature

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;  done <= 1'b0;
            iss  <= 1'b0;  ch <= 5'd0;  t <= 3'd0;  f <= 7'd0;
            cv   <= 1'b0;  ct <= 3'd0;  cf <= 7'd0; cch <= 5'd0; c_last <= 1'b0;
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
            cch    <= ch;
            c_last <= iss_last;

            // finish: done pulses in the cycle after the last write
            done <= c_last;
            if (c_last) busy <= 1'b0;
        end
    end

    // select field t from each bank word (bank word layout: see emg_feat_accum);
    // a bank that is not part of the window gives 0
    function [20:0] field;
        input [63:0] q;
        input [2:0]  t_sel;
        input        use_bank;
        begin
            if (!use_bank)          field = 21'd0;
            else case (t_sel)
                3'd0:    field = {7'd0,  q[13:0]};    // SABS
                3'd1:    field = {6'd0,  q[28:14]};   // WL
                3'd2:    field = {14'd0, q[35:29]};   // ZC
                3'd3:    field = {14'd0, q[42:36]};   // SSC
                default: field = q[63:43];            // POW
            endcase
        end
    endfunction

    wire [22:0] raw = field(s2_q0, ct, bank_use[0]) + field(s2_q1, ct, bank_use[1])
                    + field(s2_q2, ct, bank_use[2]) + field(s2_q3, ct, bank_use[3]);

    // dead-channel check: cleared at start, then set from the WL window sum of each channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                 dead_mask      <= {CMAX{1'b0}};
        else if (start && !busy)    dead_mask      <= {CMAX{1'b0}};
        else if (cv && ct == 3'd1)  dead_mask[cch] <= (raw == 23'd0);
    end

    // Every operand is made signed (zero-extended first), so the whole expression is
    // signed and >>> is an arithmetic shift. Mixing in one unsigned operand would make
    // the expression unsigned and >>> would silently become a logical shift.
    wire               k_pos = (norm_k > 6'sd0);                              // shrink
    wire        [4:0]  kr    = norm_k[4:0];                                    // k when k > 0 (1..31)
    wire        [5:0]  sl    = -norm_k;                                        // -k when k <= 0 (0..32)
    wire signed [33:0] d     = $signed({11'd0, raw}) - $signed({10'd0, norm_mu});

    // k > 0: round, then arithmetic shift right
    wire signed [33:0] rnd   = k_pos ? (34'sd1 <<< (kr - 5'd1)) : 34'sd0;      // 2^(k_f - 1)
    wire signed [33:0] shr   = (d + rnd) >>> kr;
    wire signed [7:0]  q_r   = (shr > 34'sd127)  ?  8'sd127 :
                               (shr < -34'sd128) ? -8'sd128 : shr[7:0];

    // k <= 0: shift left, clipped without forming an overflowing product
    wire signed [33:0] hi    =   34'sd127 >>> sl;                              // 127 >> s
    wire signed [33:0] lo    = -(34'sd128 >>> sl);                             // -(128 >> s)
    wire signed [33:0] shl   = d <<< sl;
    wire signed [7:0]  q_l   = (d > hi) ?  8'sd127 :
                               (d < lo) ? -8'sd128 : shl[7:0];

    assign xq_we   = cv;
    assign xq_addr = cf;
    assign xq_data = k_pos ? q_r : q_l;
endmodule

`default_nettype wire
