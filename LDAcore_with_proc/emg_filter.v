// emg_filter — the input filter (step 5, 1 Nov 2026), between the ADC receiver and Stage 1.
//
// Every sample, for each channel c = 0 .. C-1:
//   x16 = top 16 bits of the channel's 24-bit ADS1299 code (emg_spi_adc.rx16_flat)
//   5 sections in Direct Form I, in this order:
//     0, 1  high-pass   numerator g * (1, -2, 1), denominator (1, a1, a2)
//     2, 3  low-pass    numerator g * (1,  2, 1), denominator (1, a1, a2)
//     4     notch       numerator (b0, b1, b2),   denominator (1, a1, a2)
//   x8 = clip( round( y_notch / 2^(S + shift) ), -128, 127 )   S = 4, shift = cfg_shift (0..8)
// This is golden_model.py ChipRef._filter with the "target" settings, bit for bit.
//
// Number format. Coefficients: signed 16 bits with 14 fractional bits (16384 = 1.0). Stored
// numbers: S = 4 fractional bits (decided 31 Oct). Section 0's input is x16 << 4 (20 bits).
// Each section: acc = sum of coefficient x value (exact), then y = (acc + 2^13) >>> 14.
// A section with g = 16384 gives exactly num << 14, the golden model's "no gain" case.
//
// Widths (golden_model.filter_widths, worst case over ANY 16-bit input, designed coefficients):
//   section output |y| < 2^21 (sections 0-3, stored as 22 bits) and < 2^22 (notch, 23 bits)
//   g x num: num = p0 +- 2 p1 + p2 < 2^23 -> multiplier 16 x 24 bits
//   sums: 39 bits; acc here is 42 bits.
// Coefficients written over SPI must be checked with filter_widths first: a filter that is
// louder than the designed one could overflow these widths.
//
// Memory: 254 bits per channel in one RAM word (emg_tp_ram, 1-cycle read):
//   [15:0] x16[n-1]   [31:16] x16[n-2]
//   section k = 0..3: [32+44k +: 22] y[n-1], [54+44k +: 22] y[n-2]
//   notch:            [208 +: 23] y[n-1],    [231 +: 23] y[n-2]
// First sample: after reset, and again whenever core_en or the filter is off, the next sample of
// each channel loads x16 into both input-history registers and starts every section's memory at
// 0 (the RAM is not read for it). For a constant input this is the exact steady state, because
// section 0 is a high-pass: the electrode offset gives no jump and the notch does not ring.
//
// Schedule: one multiply per clock with ONE shared 16 x 24 multiplier. Per channel: read the
// memory (2 cycles), 17 multiplies (3 per section 0-3, 5 for the notch), write back (1 cycle):
// 20 cycles, so 480 cycles of the 1000-cycle sample period at C = 24.
// start (frame_start from the CDC) begins a run; done pulses after the last channel, and then
// x8_flat holds the new samples. A start while busy sets overrun (sticky) and is ignored.
`default_nettype none

module emg_filter #(
    parameter CMAX = 24
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                enable,        // core_en & filter on; 0 re-arms the first-sample start
    input  wire [4:0]          cfg_n_ch,      // C
    input  wire [4:0]          cfg_shift,     // output shift; above 8 acts as 8
    input  wire [17*16-1:0]    coef_flat,     // index 3k + {0 g, 1 a1, 2 a2} (k = 0..3), 12..16 notch b0 b1 b2 a1 a2

    input  wire                start,
    input  wire [16*CMAX-1:0]  rx16_flat,     // stable for ~1 ms after start
    output reg                 busy,
    output reg                 done,
    output reg                 overrun,
    output reg  [8*CMAX-1:0]   x8_flat        // channel c at [8c +: 8], valid from done
);
    localparam integer S  = 4;
    localparam integer F  = 14;
    localparam integer MW = 254;              // memory word

    // ------------------------------------------------------------------ coefficients
    function signed [15:0] coef;
        input [4:0] i;
        begin
            coef = coef_flat[16*i +: 16];
        end
    endfunction

    // ------------------------------------------------------------------ state
    localparam [2:0] T_IDLE = 3'd0, T_RD = 3'd1, T_LD = 3'd2, T_MUL = 3'd3, T_WR = 3'd4;
    reg  [2:0]  tst;
    reg  [4:0]  ch;
    reg  [2:0]  sec;            // section 0..4
    reg  [2:0]  stp;            // multiply inside the section
    reg  [CMAX-1:0] first;      // channel c has not had a sample since the last re-arm

    reg  signed [15:0] x16w, u1w;            // this sample, x16[n-1]
    reg  signed [21:0] y1s [0:3], y2s [0:3]; // sections 0-3: y[n-1], y[n-2] (old, then new)
    reg  signed [22:0] y1n, y2n;             // notch
    reg  signed [23:0] p0, p1, p2;           // this section's input history
    reg  signed [41:0] acc;

    wire [MW-1:0] mem_q;
    wire [MW-1:0] mem_d = {y2n, y1n, y2s[3], y1s[3], y2s[2], y1s[2], y2s[1], y1s[1], y2s[0], y1s[0], u1w, x16w};

    emg_tp_ram #(.W(MW), .DEPTH(CMAX), .AW(5)) u_mem (
        .clk(clk), .we(tst == T_WR), .waddr(ch), .wdata(mem_d),
        .re(tst == T_RD), .raddr(ch), .rdata(mem_q));

    // ------------------------------------------------------------------ one multiply
    wire        notch  = (sec == 3'd4);
    wire        last   = notch ? (stp == 3'd4) : (stp == 3'd2);
    wire signed [22:0] y1o = notch ? y1n : {y1s[sec[1:0]][21], y1s[sec[1:0]]};   // section's y[n-1]
    wire signed [22:0] y2o = notch ? y2n : {y2s[sec[1:0]][21], y2s[sec[1:0]]};   // section's y[n-2]
    wire signed [25:0] num = sec[1] ? (p0 + 2 * p1 + p2) : (p0 - 2 * p1 + p2);  // high-pass: sec 0, 1
    reg  signed [15:0] m_c;
    reg  signed [23:0] m_x;
    reg                m_neg;
    always @(*) begin
        m_neg = 1'b0;
        if (!notch) begin
            case (stp)
                3'd0:    begin m_c = coef(5'd3 * sec + 5'd0); m_x = num[23:0]; end
                3'd1:    begin m_c = coef(5'd3 * sec + 5'd1); m_x = y1o;  m_neg = 1'b1; end
                default: begin m_c = coef(5'd3 * sec + 5'd2); m_x = y2o;  m_neg = 1'b1; end
            endcase
        end else begin
            case (stp)
                3'd0:    begin m_c = coef(5'd12); m_x = p0; end
                3'd1:    begin m_c = coef(5'd13); m_x = p1; end
                3'd2:    begin m_c = coef(5'd14); m_x = p2; end
                3'd3:    begin m_c = coef(5'd15); m_x = y1o; m_neg = 1'b1; end
                default: begin m_c = coef(5'd16); m_x = y2o; m_neg = 1'b1; end
            endcase
        end
    end
    wire signed [39:0] prod  = m_c * m_x;                                       // the multiplier
    wire signed [41:0] acc_n = (stp == 3'd0 ? 42'sd0 : acc) + (m_neg ? -prod : prod);
    wire signed [41:0] y_r   = (acc_n + (42'sd1 <<< (F - 1))) >>> F;            // section output

    // output: round, shift by S + shift, clip
    wire [3:0]         sh    = S + ((cfg_shift > 5'd8) ? 4'd8 : cfg_shift[3:0]);   // 4 .. 12
    wire signed [22:0] y_out = y_r[22:0];
    wire signed [23:0] o     = ($signed({y_out[22], y_out}) + (24'sd1 <<< (sh - 1))) >>> sh;
    wire        [7:0]  o8    = (o > 24'sd127) ? 8'h7F : (o < -24'sd128) ? 8'h80 : o[7:0];

    // channel ch's inputs from the RAM word, or the first-sample start
    wire signed [15:0] x_in = rx16_flat[16*ch +: 16];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tst <= T_IDLE; ch <= 5'd0; sec <= 3'd0; stp <= 3'd0;
            first <= {CMAX{1'b1}};
            busy <= 1'b0; done <= 1'b0; overrun <= 1'b0;
            x8_flat <= {8*CMAX{1'b0}};
            x16w <= 16'sd0; u1w <= 16'sd0; y1n <= 23'sd0; y2n <= 23'sd0;
            p0 <= 24'sd0; p1 <= 24'sd0; p2 <= 24'sd0; acc <= 42'sd0;
            for (i = 0; i < 4; i = i + 1) begin y1s[i] <= 22'sd0; y2s[i] <= 22'sd0; end
        end else begin
            done <= 1'b0;
            if (!enable) first <= {CMAX{1'b1}};
            if (start && busy) overrun <= 1'b1;

            case (tst)
            T_IDLE:
                if (start) begin
                    busy <= 1'b1;
                    ch   <= 5'd0;
                    tst  <= T_RD;
                end
            T_RD:                                   // memory read issued (re = 1 in this cycle)
                tst <= T_LD;
            T_LD: begin                             // memory word arrives: load the working registers
                x16w <= x_in;
                if (first[ch]) begin
                    u1w <= x_in;
                    p0  <= {{4{x_in[15]}}, x_in, 4'd0};
                    p1  <= {{4{x_in[15]}}, x_in, 4'd0};
                    p2  <= {{4{x_in[15]}}, x_in, 4'd0};
                    for (i = 0; i < 4; i = i + 1) begin y1s[i] <= 22'sd0; y2s[i] <= 22'sd0; end
                    y1n <= 23'sd0; y2n <= 23'sd0;
                end else begin
                    u1w <= mem_q[15:0];
                    p0  <= {{4{x_in[15]}}, x_in, 4'd0};
                    p1  <= {{4{mem_q[15]}}, mem_q[15:0], 4'd0};
                    p2  <= {{4{mem_q[31]}}, mem_q[31:16], 4'd0};
                    for (i = 0; i < 4; i = i + 1) begin
                        y1s[i] <= mem_q[32 + 44 * i +: 22];
                        y2s[i] <= mem_q[54 + 44 * i +: 22];
                    end
                    y1n <= mem_q[208 +: 23];
                    y2n <= mem_q[231 +: 23];
                end
                sec <= 3'd0; stp <= 3'd0;
                tst <= T_MUL;
            end
            T_MUL: begin
                acc <= acc_n;
                if (!last) begin
                    stp <= stp + 3'd1;
                end else begin
                    // section done: its memory moves on, its outputs are the next section's inputs
                    p0 <= y_r[23:0];
                    p1 <= {y1o[22], y1o};
                    p2 <= {y2o[22], y2o};
                    stp <= 3'd0;
                    if (!notch) begin
                        y2s[sec[1:0]] <= y1s[sec[1:0]];
                        y1s[sec[1:0]] <= y_r[21:0];
                        sec <= sec + 3'd1;
                    end else begin
                        y2n <= y1n;
                        y1n <= y_out;
                        x8_flat[8*ch +: 8] <= o8;
                        tst <= T_WR;
                    end
                end
            end
            T_WR: begin                             // memory write (we = 1 in this cycle)
                first[ch] <= 1'b0;
                if (ch == cfg_n_ch - 5'd1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    tst  <= T_IDLE;
                end else begin
                    ch  <= ch + 5'd1;
                    tst <= T_RD;
                end
            end
            default: tst <= T_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
