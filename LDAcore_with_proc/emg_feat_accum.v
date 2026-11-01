// emg_feat_accum — Stage 1: per-sample features summed into banks.
//
// Step 1 of the new design (31 Oct 2026): up to 4 banks with a rotating bank pointer, and the
// bank size B and the bank count N are settings (cfg_bank_len, cfg_n_banks) instead of a fixed
// 98-sample stride and 2 banks.
//
//   * Every sample, for each channel, the 5 features of the INT8 sample are added into the bank
//     the pointer points at. The first sample of a bank overwrites instead of adding.
//   * After B samples the bank is complete: stride_done pulses and Stage 2 sums the last N banks.
//   * swap (from emg_core_ctrl, after Stage 2) moves the pointer on: 0, 1, .., N-1, 0, ...
//   * bank_used[i] = bank i has been completed at least once since reset. Stage 2 counts only
//     those banks, so right after reset the window holds only the banks seen so far (the others
//     count as 0) instead of whatever the RAM held before. This is the golden model's rule.
//   * The per-channel history (previous sample, previous slope) carries across bank boundaries.
//   * Dead check (step 4, 1 Nov 2026): bank_chg[i][c] = channel c's 24-bit code changed at least
//     once while bank i was filled (chg_flat from emg_spi_adc, one bit per channel per sample;
//     the first sample of a bank overwrites). chg_win[c] = OR of bank_chg[i][c] over the banks of
//     the window (bank_use). Stage 2 turns it into the dead mask: dead = no change in the window.
//
// Bank word, 64 bits per channel per bank: {POW 21, SSC 7, ZC 7, WL 15, SABS 14}.
// Largest values at B = 100: SABS 12,800 < 2^14, WL 25,500 < 2^15, ZC and SSC 100 < 2^7,
// POW 1,638,400 < 2^21, so no field can overflow for B <= 100.
//
// Write B and N only while core_en = 0 and before the first frame after reset: the pointer and
// bank_used are cleared only by reset.
`default_nettype none

module emg_feat_accum #(
    parameter CMAX = 24
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire [4:0]        cfg_n_ch,
    input  wire [6:0]        cfg_bank_len,   // B, 1..100 samples per bank
    input  wire [2:0]        cfg_n_banks,    // N, 1..4 banks per window

    input  wire              frame_start,
    input  wire [8*CMAX-1:0] rx_flat,
    input  wire [CMAX-1:0]   chg_flat,       // code changed since the previous sample (per channel)
    output reg               busy,
    output reg               sample_done,
    output reg               stride_done,    // a bank was completed
    output reg               overrun,

    input  wire              swap,           // move the bank pointer on (after Stage 2)
    output reg  [1:0]        bank_ptr,       // bank being filled
    output reg  [3:0]        bank_used,      // bank i completed at least once since reset

    input  wire              s2_re,
    input  wire [4:0]        s2_addr,
    output wire [63:0]       s2_q0,
    output wire [63:0]       s2_q1,
    output wire [63:0]       s2_q2,
    output wire [63:0]       s2_q3,
    input  wire [3:0]        bank_use,       // banks of the window (bank_used, first N)
    output wire [CMAX-1:0]   chg_win         // code changed in the window (per channel)
);

    reg        rd_active;
    reg  [4:0] rd_ch;

    reg        wr_valid;
    reg  [4:0] wr_ch;
    reg  [7:0] wr_x;

    reg  [6:0] cnt_win;          // sample number inside the current bank, 0..B-1
    reg  [CMAX-1:0] bank_chg [0:3];   // code changed while bank i was filled (dead check)
    reg        first_sample;

    wire start_accept = frame_start & ~busy;

    wire       s1_re   = start_accept | rd_active;
    wire [4:0] s1_rch  = start_accept ? 5'd0 : rd_ch;
    wire [7:0] s1_rx_c = rx_flat[s1_rch * 8 +: 8];

    wire last_write_cycle = wr_valid & ~rd_active;
    wire win_last_sample  = (cnt_win == (cfg_bank_len - 7'd1));

    // next pointer: 0, 1, .., N-1, 0
    wire [2:0] ptr_inc  = {1'b0, bank_ptr} + 3'd1;
    wire [1:0] ptr_next = (ptr_inc >= cfg_n_banks) ? 2'd0 : ptr_inc[1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_active    <= 1'b0;
            rd_ch        <= 5'd0;
            wr_valid     <= 1'b0;
            wr_ch        <= 5'd0;
            wr_x         <= 8'd0;
            cnt_win      <= 7'd0;
            first_sample <= 1'b1;
            busy         <= 1'b0;
            sample_done  <= 1'b0;
            stride_done  <= 1'b0;
            overrun      <= 1'b0;
            bank_ptr     <= 2'd0;
            bank_used    <= 4'd0;
            bank_chg[0]  <= {CMAX{1'b0}};
            bank_chg[1]  <= {CMAX{1'b0}};
            bank_chg[2]  <= {CMAX{1'b0}};
            bank_chg[3]  <= {CMAX{1'b0}};
        end else begin
            sample_done <= 1'b0;
            stride_done <= 1'b0;

            if (frame_start && busy) begin
                overrun <= 1'b1;
            end

            if (swap && !busy && !frame_start) begin
                bank_ptr <= ptr_next;
            end

            if (start_accept) begin
                busy <= 1'b1;
                if (cfg_n_ch > 5'd1) begin
                    rd_active <= 1'b1;
                    rd_ch     <= 5'd1;
                end else begin
                    rd_active <= 1'b0;
                    rd_ch     <= 5'd0;
                end
            end else if (rd_active) begin
                if (rd_ch == (cfg_n_ch - 5'd1)) begin
                    rd_active <= 1'b0;
                end else begin
                    rd_ch <= rd_ch + 5'd1;
                end
            end

            wr_valid <= s1_re;
            if (s1_re) begin
                wr_ch <= s1_rch;
                wr_x  <= s1_rx_c;
            end

            if (last_write_cycle) begin
                busy         <= 1'b0;
                sample_done  <= 1'b1;
                stride_done  <= win_last_sample;
                first_sample <= 1'b0;
                // chg_flat is stable for the whole sample (like rx_flat); first sample overwrites
                bank_chg[bank_ptr] <= chg_flat | ((cnt_win == 7'd0) ? {CMAX{1'b0}} : bank_chg[bank_ptr]);

                if (win_last_sample) begin
                    cnt_win             <= 7'd0;
                    bank_used[bank_ptr] <= 1'b1;   // this bank now holds B real samples
                end else begin
                    cnt_win <= cnt_win + 7'd1;
                end
            end
        end
    end

    wire [9:0] hist_rdata;
    wire [9:0] hist_wdata;

    emg_tp_ram #(
        .DEPTH (CMAX),
        .W (10)
    ) u_hist_ram (
        .clk   (clk),
        .we    (wr_valid),
        .waddr (wr_ch),
        .wdata (hist_wdata),
        .re    (s1_re),
        .raddr (s1_rch),
        .rdata (hist_rdata)
    );

    // Stage 1 reads the bank being filled; between samples Stage 2 reads all banks.
    wire       bank_re    = (busy | start_accept) ? s1_re  : s2_re;
    wire [4:0] bank_raddr = (busy | start_accept) ? s1_rch : s2_addr;
    wire [63:0] bank_wdata;
    wire [63:0] bank_q [0:3];

    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : g_bank
            emg_tp_ram #(
                .DEPTH (CMAX),
                .W (64)
            ) u_bank_ram (
                .clk   (clk),
                .we    (wr_valid & (bank_ptr == gi)),
                .waddr (wr_ch),
                .wdata (bank_wdata),
                .re    (bank_re),
                .raddr (bank_raddr),
                .rdata (bank_q[gi])
            );
        end
    endgenerate

    assign chg_win = (bank_use[0] ? bank_chg[0] : {CMAX{1'b0}}) | (bank_use[1] ? bank_chg[1] : {CMAX{1'b0}})
                   | (bank_use[2] ? bank_chg[2] : {CMAX{1'b0}}) | (bank_use[3] ? bank_chg[3] : {CMAX{1'b0}});

    assign s2_q0 = bank_q[0];
    assign s2_q1 = bank_q[1];
    assign s2_q2 = bank_q[2];
    assign s2_q3 = bank_q[3];

    wire signed [7:0] x_prev = first_sample ? 8'sd0 : $signed(hist_rdata[9:2]);
    wire              d0_pos = first_sample ? 1'b0  : hist_rdata[1];
    wire              d0_neg = first_sample ? 1'b0  : hist_rdata[0];

    wire [7:0]  alu_sabs;
    wire [7:0]  alu_wl;
    wire        alu_zc;
    wire        alu_ssc;
    wire [14:0] alu_pow;
    wire        alu_d1_pos;
    wire        alu_d1_neg;

    emg_feat_alu u_feat_alu (
        .x      (wr_x),
        .x_prev (x_prev),
        .d0_pos (d0_pos),
        .d0_neg (d0_neg),
        .sabs   (alu_sabs),
        .wl     (alu_wl),
        .zc     (alu_zc),
        .ssc    (alu_ssc),
        .pow    (alu_pow),
        .d1_pos (alu_d1_pos),
        .d1_neg (alu_d1_neg)
    );

    assign hist_wdata = {wr_x, alu_d1_pos, alu_d1_neg};

    wire [63:0] active_bank_q = bank_q[bank_ptr];
    wire [63:0] base_word     = (cnt_win == 7'd0) ? 64'd0 : active_bank_q;   // first sample overwrites

    wire [13:0] acc_sabs = base_word[13:0]  + {6'd0, alu_sabs};
    wire [14:0] acc_wl   = base_word[28:14] + {7'd0, alu_wl};
    wire [6:0]  acc_zc   = base_word[35:29] + {6'd0, alu_zc};
    wire [6:0]  acc_ssc  = base_word[42:36] + {6'd0, alu_ssc};
    wire [20:0] acc_pow  = base_word[63:43] + {6'd0, alu_pow};

    assign bank_wdata = {acc_pow, acc_ssc, acc_zc, acc_wl, acc_sabs};

endmodule

`default_nettype wire
