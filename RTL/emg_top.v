// emg_top — M9: chip top level. Wiring only.
//
// Keep the ports, parameters AND these instance names; the chip testbench
// looks inside u_norm and u_mac to tell you which stage is wrong:
//
//   u_rst_sys   rst_sync         clk_sys reset
//   u_rst_spi   rst_sync         clk_spi reset
//   u_cfg       emg_cfg_regs     (clk_sys)
//   u_adc       emg_spi_adc      (clk_spi)  pass CMAX, SAMPLE_DIV
//   u_cdc       cdc_toggle_sync  frame_tog -> frame_start in clk_sys
//   u_accum     emg_feat_accum   Stage 1, rx_flat directly from u_adc (stable ~1 ms)
//   u_norm      emg_shift_norm   Stage 2
//   u_xq        emg_tp_ram       x_q RAM, 120 x 8: written by u_norm, read by u_mac
//   u_mac       emg_lda_mac      Stage 3
//   u_ctrl      emg_core_ctrl
//
// cfg_n_ch from u_cfg goes to u_adc, u_accum, u_norm, u_mac; core_en to u_adc.
// overrun comes from u_accum; class_valid, class_out, primed from u_ctrl.
// The block diagram on the interface page shows every connection.
`default_nettype none

module emg_top #(
    parameter CMAX       = 24,
    parameter SAMPLE_DIV = 8120
) (
    // clocks and reset
    input  wire        clk_sys,
    input  wire        clk_spi,
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

    // results
    output wire        class_valid,
    output wire [2:0]  class_out,
    output wire        primed,
    output wire        overrun
);

    // -------------------------------------------------------------------------
    // Reset synchronisers
    // -------------------------------------------------------------------------
    wire rst_sys_n, rst_spi_n;

    rst_sync u_rst_sys (
        .clk    (clk_sys),
        .arst_n (arst_n),
        .rst_n  (rst_sys_n)
    );

    rst_sync u_rst_spi (
        .clk    (clk_spi),
        .arst_n (arst_n),
        .rst_n  (rst_spi_n)
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

    emg_cfg_regs u_cfg (
        .clk       (clk_sys),
        .rst_n     (rst_sys_n),
        .cfg_we    (cfg_we),
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
    // ADC front-end (clk_spi) and frame CDC into clk_sys
    // -------------------------------------------------------------------------
    wire [8*CMAX-1:0] rx_flat;
    wire              frame_tog;
    wire              frame_start;

    emg_spi_adc #(
        .CMAX       (CMAX),
        .SAMPLE_DIV (SAMPLE_DIV)
    ) u_adc (
        .clk_spi    (clk_spi),
        .rst_n      (rst_spi_n),
        .cfg_n_ch   (cfg_n_ch),
        .core_en    (core_en),
        .adc_convst (adc_convst),
        .adc_drdy_n (adc_drdy_n),
        .adc_cs_n   (adc_cs_n),
        .adc_sclk   (adc_sclk),
        .adc_miso   (adc_miso),
        .rx_flat    (rx_flat),
        .frame_tog  (frame_tog)
    );

    cdc_toggle_sync u_cdc (
        .clk       (clk_sys),
        .rst_n     (rst_sys_n),
        .tog_in    (frame_tog),
        .pulse_out (frame_start)
    );

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
        .DEPTH (120),
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

    emg_lda_mac u_mac (
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
    // Core sequencer
    // -------------------------------------------------------------------------
    emg_core_ctrl u_ctrl (
        .clk             (clk_sys),
        .rst_n           (rst_sys_n),
        .stride_done     (stride_done),
        .norm_start      (norm_start),
        .norm_done       (norm_done),
        .swap            (swap),
        .mac_start       (mac_start),
        .mac_class_valid (mac_class_valid),
        .mac_class_out   (mac_class_out),
        .class_valid     (class_valid),
        .class_out       (class_out),
        .primed          (primed)
    );

endmodule

`default_nettype wire
