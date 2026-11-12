// emg_top — one board of the multi-board classifier. Wiring only.
//
// Changes from the single-chip version (branch single_chip):
//   * one clock (clk_sys, 256 kHz): the SPI side runs on it too, so cdc_toggle_sync and
//     the second rst_sync are gone; frame_start is a toggle edge detect here.
//   * at most 8 channels per board (CMAX = 8): banks, x_q, mu/k and W memories shrink.
//   * Stage 3 holds the K-1 difference rows (NCLS = 3), and its result goes to
//     u_link, which adds the partial sums of all boards; the head's u_link decides.
//   * cfg_board selects which board a configuration write is for (board id from u_link).
//
// Instance names are kept: u_rst_sys, u_cfg, u_adc, u_accum, u_norm, u_xq, u_mac, u_ctrl,
// plus u_link (emg_chain_link).
`default_nettype none

module emg_top #(
    parameter CMAX       = 8,
    parameter SAMPLE_DIV = 260,         // 256 kHz / 260 = 984.6 Hz
    parameter NDIFF      = 3            // K-1 rows in Stage 3
) (
    // clock and reset (clk_sys comes from the head board, LCLK)
    input  wire        clk_sys,
    input  wire        arst_n,

    // ADC pods (SPI daisy chain)
    output wire        adc_convst,
    input  wire        adc_drdy_n,
    output wire        adc_cs_n,
    output wire        adc_sclk,
    input  wire        adc_miso,

    // configuration write port
    input  wire        cfg_we,
    input  wire [11:0] cfg_addr,
    input  wire [31:0] cfg_wdata,
    input  wire [1:0]  cfg_board,       // write goes to the board with this id

    // board chain
    input  wire        prs_up_i,        // neighbour present upstream / downstream
    input  wire        prs_dn_i,
    input  wire [1:0]  id_i,
    output wire [1:0]  id_o,
    input  wire        run_i,
    output wire        run_o,
    output wire        rdy_o,           // open drain on the board: wired-AND -> all_rdy_i
    input  wire        all_rdy_i,
    input  wire        up_dat_i,        // from the board downstream
    input  wire        up_vld_i,
    output wire        up_dat_o,        // to the board upstream
    output wire        up_vld_o,

    // results
    output wire        class_valid,
    output wire [2:0]  class_out,
    output wire        primed,
    output wire        overrun
);

    // -------------------------------------------------------------------------
    // Reset synchronisers
    // -------------------------------------------------------------------------
    wire rst_sys_n;

    rst_sync u_rst_sys (
        .clk    (clk_sys),
        .arst_n (arst_n),
        .rst_n  (rst_sys_n)
    );

    // -------------------------------------------------------------------------
    // Configuration registers (clk_sys)
    // -------------------------------------------------------------------------
    wire        [4:0]  cfg_n_ch;
    wire               core_en;
    wire               norm_re;
    wire        [6:0]  norm_addr;
    wire        [21:0] norm_mu;
    wire        [4:0]  norm_k;
    wire               w_re;
    wire        [9:0]  w_addr;
    wire signed [7:0]  w_data;
    wire               b_re;
    wire        [2:0]  b_addr;
    wire signed [31:0] b_data;

    wire [1:0] my_id;
    wire       cfg_we_b = cfg_we & (cfg_board == my_id);

    emg_cfg_regs #(
        .NF (5 * CMAX),
        .NW (NDIFF * 5 * CMAX)
    ) u_cfg (
        .clk       (clk_sys),
        .rst_n     (rst_sys_n),
        .cfg_we    (cfg_we_b),
        .cfg_addr  (cfg_addr),
        .cfg_wdata (cfg_wdata),
        .cfg_n_ch  (cfg_n_ch),
        .core_en   (core_en),
        .norm_re   (norm_re),
        .norm_addr (norm_addr),
        .norm_mu   (norm_mu),
        .norm_k    (norm_k),
        .w_re      (w_re),
        .w_addr    (w_addr),
        .w_data    (w_data),
        .b_re      (b_re),
        .b_addr    (b_addr),
        .b_data    (b_data)
    );

    // -------------------------------------------------------------------------
    // ADC front-end, same clock: frame_start = edge of frame_tog
    // -------------------------------------------------------------------------
    wire [8*CMAX-1:0] rx_flat;
    wire              frame_tog;
    wire              frame_start;
    wire              run;              // common enable, driven by u_link below

    emg_spi_adc #(
        .CMAX       (CMAX),
        .SAMPLE_DIV (SAMPLE_DIV)
    ) u_adc (
        .clk_spi    (clk_sys),
        .rst_n      (rst_sys_n),
        .cfg_n_ch   (cfg_n_ch),
        .core_en    (run),              // common enable from the chain
        .adc_convst (adc_convst),
        .adc_drdy_n (adc_drdy_n),
        .adc_cs_n   (adc_cs_n),
        .adc_sclk   (adc_sclk),
        .adc_miso   (adc_miso),
        .rx_flat    (rx_flat),
        .frame_tog  (frame_tog)
    );

    reg frame_tog_d;
    always @(posedge clk_sys or negedge rst_sys_n)
        if (!rst_sys_n) frame_tog_d <= 1'b0;
        else            frame_tog_d <= frame_tog;

    assign frame_start = frame_tog ^ frame_tog_d;

    // -------------------------------------------------------------------------
    // Stage 1: feature accumulator
    // -------------------------------------------------------------------------
    wire        busy1;
    wire        sample_done;
    wire        stride_done;
    wire        swap;
    wire        bank_sel;
    wire        s2_re;
    wire [4:0]  s2_addr;
    wire [63:0] s2_q0, s2_q1;

    emg_feat_accum #(
        .CMAX (CMAX)
    ) u_accum (
        .clk         (clk_sys),
        .rst_n       (rst_sys_n),
        .cfg_n_ch    (cfg_n_ch),
        .frame_start (frame_start),
        .rx_flat     (rx_flat),
        .busy        (busy1),
        .sample_done (sample_done),
        .stride_done (stride_done),
        .overrun     (overrun),
        .swap        (swap),
        .bank_sel    (bank_sel),
        .s2_re       (s2_re),
        .s2_addr     (s2_addr),
        .s2_q0       (s2_q0),
        .s2_q1       (s2_q1)
    );

    // -------------------------------------------------------------------------
    // Stage 2: shift normaliser -> x_q RAM
    // -------------------------------------------------------------------------
    wire              norm_start;
    wire              norm_busy;
    wire              norm_done;
    wire              xq_we;
    wire        [6:0] xq_addr;
    wire signed [7:0] xq_data;

    emg_shift_norm #(
        .CMAX (CMAX)
    ) u_norm (
        .clk       (clk_sys),
        .rst_n     (rst_sys_n),
        .cfg_n_ch  (cfg_n_ch),
        .start     (norm_start),
        .busy      (norm_busy),
        .done      (norm_done),
        .s2_re     (s2_re),
        .s2_addr   (s2_addr),
        .s2_q0     (s2_q0),
        .s2_q1     (s2_q1),
        .norm_re   (norm_re),
        .norm_addr (norm_addr),
        .norm_mu   (norm_mu),
        .norm_k    (norm_k),
        .xq_we     (xq_we),
        .xq_addr   (xq_addr),
        .xq_data   (xq_data)
    );

    wire       xq_re;
    wire [6:0] xq_raddr;
    wire [7:0] xq_q;

    //ram to hold the data in between after the normalization
    emg_tp_ram #(
        .W     (8),
        .DEPTH (5 * CMAX),
        .AW    (7)
    ) u_xq (
        .clk   (clk_sys),
        .we    (xq_we),
        .waddr (xq_addr),
        .wdata (xq_data),
        .re    (xq_re),
        .raddr (xq_raddr),
        .rdata (xq_q)
    );

    // -------------------------------------------------------------------------
    // Stage 3: LDA multiply-accumulate
    // -------------------------------------------------------------------------
    wire               mac_start;
    wire               mac_busy;
    wire               score_valid;
    wire               mac_class_valid;
    wire        [2:0]  score_cls;
    wire        [2:0]  mac_class_out;
    wire signed [31:0] score;

    emg_lda_mac #(
        .NCLS (NDIFF)
    ) u_mac (
        .clk         (clk_sys),
        .rst_n       (rst_sys_n),
        .cfg_n_ch    (cfg_n_ch),
        .start       (mac_start),
        .busy        (mac_busy),
        .xq_re       (xq_re),
        .xq_addr     (xq_raddr),
        .xq_data     (xq_q),
        .w_re        (w_re),
        .w_addr      (w_addr),
        .w_data      (w_data),
        .b_re        (b_re),
        .b_addr      (b_addr),
        .b_data      (b_data),
        .score_valid (score_valid),
        .score_cls   (score_cls),
        .score       (score),
        .class_valid (mac_class_valid),
        .class_out   (mac_class_out)
    );

    // -------------------------------------------------------------------------
    // Board chain: adds the partial sums of all boards, head decides
    // -------------------------------------------------------------------------
    wire       dec_valid;
    wire [2:0] dec_cls;

    emg_chain_link #(
        .NV (NDIFF)
    ) u_link (
        .clk         (clk_sys),
        .rst_n       (rst_sys_n),
        .prs_up_i    (prs_up_i),
        .prs_dn_i    (prs_dn_i),
        .is_head     (),
        .is_tail     (),
        .id_i        (id_i),
        .id_o        (id_o),
        .my_id       (my_id),
        .core_en     (core_en),
        .run_i       (run_i),
        .run_o       (run_o),
        .run         (run),
        .cfg_we      (cfg_we_b),
        .cfg_addr    (cfg_addr),
        .cfg_wdata   (cfg_wdata),
        .score_valid (score_valid),
        .score_cls   (score_cls),
        .score       (score),
        .mac_done    (mac_class_valid),
        .rdy_o       (rdy_o),
        .all_rdy_i   (all_rdy_i),
        .up_dat_i    (up_dat_i),
        .up_vld_i    (up_vld_i),
        .up_dat_o    (up_dat_o),
        .up_vld_o    (up_vld_o),
        .dec_valid   (dec_valid),
        .dec_cls     (dec_cls)
    );

    // -------------------------------------------------------------------------
    // Core sequencer (masks the first decision, drives the output pins)
    // -------------------------------------------------------------------------
    emg_core_ctrl u_ctrl (
        .clk             (clk_sys),
        .rst_n           (rst_sys_n),
        .stride_done     (stride_done),
        .norm_start      (norm_start),
        .norm_done       (norm_done),
        .swap            (swap),
        .mac_start       (mac_start),
        .mac_class_valid (dec_valid),
        .mac_class_out   (dec_cls),
        .class_valid     (class_valid),
        .class_out       (class_out),
        .primed          (primed)
    );

endmodule

`default_nettype wire
