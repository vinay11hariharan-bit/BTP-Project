// emg_top — M9: chip top level. Wiring only.
//
// Step 3 (1 Nov 2026): the chip extracts features; the gesture decision runs on the
// microcontroller. The on-chip LDA (Stage 3, emg_lda_mac) and the 45-pin settings bus are gone;
// the microcontroller reads frames and writes / reads settings over SPI (emg_spi_slave).
//
// Keep the ports, parameters AND these instance names; the testbenches look inside them:
//
//   u_rst_sys   rst_sync         clk_sys reset
//   u_rst_spi   rst_sync         clk_spi reset
//   u_cfg       emg_cfg_regs     settings (clk_sys)
//   u_adc       emg_spi_adc      ADS1299 link (clk_spi), pass CMAX, SAMPLE_DIV
//   u_cdc       cdc_toggle_sync  frame_tog -> frame_start in clk_sys
//   u_cdc_miss  cdc_toggle_sync  miss_tog  -> frame_miss  in clk_sys
//   u_filt      emg_filter       input filter (step 5): u_adc.rx16_flat -> x8_flat, when 0x7FC = 1
//   u_accum     emg_feat_accum   Stage 1: u_filt.x8_flat after u_filt.done (filter on), or
//                                u_adc.rx_flat after frame_start (filter off, the old direct path)
//   u_norm      emg_shift_norm   Stage 2
//   u_xq        emg_tp_ram       x_q RAM, 120 x 8: written by u_norm, read by u_mcu for frames
//   u_ctrl      emg_core_ctrl    sequencing, skip rule, frame header, status
//   u_mcu       emg_spi_slave    microcontroller link (clk_sys)
//
// Step 5 (1 Nov 2026): the input filter (u_filt) between the ADC receiver and Stage 1, switched on
// by setting 0x7FC. A filter overrun (a frame while the filter is busy) counts as an overrun too.
//
// Step 4 (1 Nov 2026): rest flag (u_norm.rest_sum < T_REST, in u_ctrl) and the dead check on the
// 24-bit codes (u_adc.chg_flat -> u_accum.chg_win -> u_norm.dead_mask). chg_flat is stable with
// rx_flat, so it needs no synchroniser either.
//
// u_adc.lo_flat is ORed into u_ctrl.frame_lo and u_adc.hdr_err goes to u_ctrl.frame_hdr_err:
// like rx_flat, both are stable for ~1 ms after frame_tog, so u_ctrl samples them at
// frame_start without a synchroniser. The ADC pods are TI ADS1299 devices in daisy-chain mode.
`default_nettype none

module emg_top #(
    parameter CMAX       = 24,
    parameter SAMPLE_DIV = 8000                // clk_spi cycles per sample: 8 MHz / 8000 = 1 kHz
) (
    // clocks and reset
    input  wire        clk_sys,
    input  wire        clk_spi,
    input  wire        arst_n,

    // ADC pods (ADS1299 SPI daisy chain, chip = master)
    output wire        adc_convst,     // -> START of all devices
    input  wire        adc_drdy_n,
    output wire        adc_cs_n,
    output wire        adc_sclk,
    input  wire        adc_miso,       // <- DOUT of device 0
    output wire        adc_mosi,       // -> DIN of all devices (setup commands)

    // microcontroller (SPI, chip = slave)
    input  wire        mcu_cs_n,
    input  wire        mcu_sclk,
    input  wire        mcu_mosi,
    output wire        mcu_miso,
    output wire        mcu_drdy        // 1 = a new frame is ready (cleared when CS falls)
);

    // -------------------------------------------------------------------------
    // Reset synchronisers
    // -------------------------------------------------------------------------
    wire rst_sys_n, rst_spi_n;

    rst_sync u_rst_sys (.clk(clk_sys), .arst_n(arst_n), .rst_n(rst_sys_n));
    rst_sync u_rst_spi (.clk(clk_spi), .arst_n(arst_n), .rst_n(rst_spi_n));

    // -------------------------------------------------------------------------
    // Microcontroller link and settings (clk_sys)
    // -------------------------------------------------------------------------
    wire               cfg_we;
    wire        [11:0] cfg_addr;
    wire        [31:0] cfg_wdata;
    wire               rd_req;
    wire        [11:0] rd_addr;
    wire        [31:0] rd_data;
    wire        [4:0]  cfg_n_ch;
    wire               core_en;
    wire               cfg_err;
    wire        [3:0]  cfg_dev_ch;
    wire        [4:0]  cfg_shift;
    wire        [6:0]  cfg_bank_len;
    wire        [2:0]  cfg_n_banks;
    wire        [5:0]  cfg_warmup;
    wire        [20:0] cfg_t_rest;
    wire               cfg_dead_low8;
    wire               cfg_filt_on;
    wire   [17*16-1:0] coef_flat;
    wire        [20:0] f_rest_sum;
    wire        [7:0]  cmd_byte;
    wire               cmd_last;
    wire               cmd_tog;
    wire               norm_re;
    wire        [6:0]  norm_addr;
    wire        [23:0] norm_mu;
    wire signed [5:0]  norm_k;
    wire        [7:0]  status;

    emg_cfg_regs #(
        .CMAX      (CMAX)
    ) u_cfg (
        .clk          (clk_sys),
        .rst_n        (rst_sys_n),
        .cfg_we       (cfg_we),
        .cfg_addr     (cfg_addr),
        .cfg_wdata    (cfg_wdata),
        .cfg_n_ch     (cfg_n_ch),
        .core_en      (core_en),
        .cfg_err      (cfg_err),
        .cfg_dev_ch   (cfg_dev_ch),
        .cfg_shift    (cfg_shift),
        .cfg_bank_len (cfg_bank_len),
        .cfg_n_banks  (cfg_n_banks),
        .cfg_warmup   (cfg_warmup),
        .cfg_t_rest   (cfg_t_rest),
        .cfg_dead_low8(cfg_dead_low8),
        .cfg_filt_on  (cfg_filt_on),
        .coef_flat    (coef_flat),
        .cmd_byte     (cmd_byte),
        .cmd_last     (cmd_last),
        .cmd_tog      (cmd_tog),
        .norm_re      (norm_re),
        .norm_addr    (norm_addr),
        .norm_mu      (norm_mu),
        .norm_k       (norm_k),
        .rd_req       (rd_req),
        .rd_addr      (rd_addr),
        .rd_data      (rd_data),
        .status       (status),
        .rest_sum     (f_rest_sum)
    );

    wire              cs_low;
    wire              frame_pub;
    wire              xq_we_raw;
    wire              xq_we_en;
    wire              xq_we = xq_we_raw & xq_we_en;    // skip rule: no writes while a read is on
    wire        [6:0] xq_addr;
    wire signed [7:0] xq_data;
    wire              xq_re;
    wire        [6:0] xq_raddr;
    wire        [7:0] xq_q;
    wire        [7:0] f_count;
    wire        [7:0] f_flags;
    wire       [23:0] f_dead;

    emg_spi_slave u_mcu (
        .clk        (clk_sys),
        .rst_n      (rst_sys_n),
        .mcu_cs_n   (mcu_cs_n),
        .mcu_sclk   (mcu_sclk),
        .mcu_mosi   (mcu_mosi),
        .mcu_miso   (mcu_miso),
        .mcu_drdy   (mcu_drdy),
        .cs_low     (cs_low),
        .cfg_n_ch   (cfg_n_ch),
        .frame_pub  (frame_pub),
        .xq_written (xq_we),
        .f_count    (f_count),
        .f_flags    (f_flags),
        .f_dead     (f_dead),
        .xq_re      (xq_re),
        .xq_raddr   (xq_raddr),
        .xq_q       (xq_q),
        .cfg_we     (cfg_we),
        .cfg_addr   (cfg_addr),
        .cfg_wdata  (cfg_wdata),
        .rd_req     (rd_req),
        .rd_addr    (rd_addr),
        .rd_data    (rd_data)
    );

    // -------------------------------------------------------------------------
    // ADC front-end (clk_spi) and frame CDC into clk_sys
    // -------------------------------------------------------------------------
    wire [8*CMAX-1:0] rx_flat;
    wire [16*CMAX-1:0] rx16_flat;
    wire [CMAX-1:0]   lo_flat;
    wire [CMAX-1:0]   chg_flat;
    wire              hdr_err;
    wire              frame_tog;
    wire              frame_start;
    wire              miss_tog;
    wire              frame_miss;

    emg_spi_adc #(
        .CMAX       (CMAX),
        .SAMPLE_DIV (SAMPLE_DIV)
    ) u_adc (
        .clk_spi    (clk_spi),
        .rst_n      (rst_spi_n),
        .cfg_n_ch   (cfg_n_ch),
        .cfg_dev_ch (cfg_dev_ch),
        .cfg_shift  (cfg_shift),
        .cfg_dead_low8 (cfg_dead_low8),
        .core_en    (core_en),
        .cmd_byte   (cmd_byte),
        .cmd_last   (cmd_last),
        .cmd_tog    (cmd_tog),
        .adc_convst (adc_convst),
        .adc_drdy_n (adc_drdy_n),
        .adc_cs_n   (adc_cs_n),
        .adc_sclk   (adc_sclk),
        .adc_miso   (adc_miso),
        .adc_mosi   (adc_mosi),
        .rx_flat    (rx_flat),
        .rx16_flat  (rx16_flat),
        .lo_flat    (lo_flat),
        .chg_flat   (chg_flat),
        .hdr_err    (hdr_err),
        .frame_tog  (frame_tog),
        .miss_tog   (miss_tog)
    );

    cdc_toggle_sync u_cdc      (.clk(clk_sys), .rst_n(rst_sys_n), .tog_in(frame_tog), .pulse_out(frame_start));
    cdc_toggle_sync u_cdc_miss (.clk(clk_sys), .rst_n(rst_sys_n), .tog_in(miss_tog),  .pulse_out(frame_miss));

    // -------------------------------------------------------------------------
    // Input filter (step 5)
    // -------------------------------------------------------------------------
    wire              filt_busy;
    wire              filt_done;
    wire              filt_overrun;
    wire [8*CMAX-1:0] x8_flat;

    emg_filter #(
        .CMAX (CMAX)
    ) u_filt (
        .clk       (clk_sys),
        .rst_n     (rst_sys_n),
        .enable    (core_en & cfg_filt_on),
        .cfg_n_ch  (cfg_n_ch),
        .cfg_shift (cfg_shift),
        .coef_flat (coef_flat),
        .start     (frame_start & cfg_filt_on),
        .rx16_flat (rx16_flat),
        .busy      (filt_busy),
        .done      (filt_done),
        .overrun   (filt_overrun),
        .x8_flat   (x8_flat)
    );

    wire              s1_start = cfg_filt_on ? filt_done : frame_start;
    wire [8*CMAX-1:0] s1_x8    = cfg_filt_on ? x8_flat   : rx_flat;

    // -------------------------------------------------------------------------
    // Stage 1: feature accumulator
    // -------------------------------------------------------------------------
    wire        busy1;
    wire        sample_done;
    wire        stride_done;
    wire        overrun;
    wire        acc_overrun;
    assign      overrun = acc_overrun | filt_overrun;
    wire        swap;
    wire [1:0]  bank_ptr;
    wire [3:0]  bank_used;
    wire        s2_re;
    wire [4:0]  s2_addr;
    wire [63:0] s2_q0, s2_q1, s2_q2, s2_q3;
    wire [CMAX-1:0] chg_win;
    // banks that form the window: completed since reset and below N
    wire [3:0]  bank_use = bank_used & ((4'd1 << cfg_n_banks) - 4'd1);

    emg_feat_accum #(
        .CMAX (CMAX)
    ) u_accum (
        .clk         (clk_sys),
        .rst_n       (rst_sys_n),
        .cfg_n_ch    (cfg_n_ch),
        .cfg_bank_len(cfg_bank_len),
        .cfg_n_banks (cfg_n_banks),
        .frame_start (s1_start),
        .rx_flat     (s1_x8),
        .chg_flat    (chg_flat),
        .busy        (busy1),
        .sample_done (sample_done),
        .stride_done (stride_done),
        .overrun     (acc_overrun),
        .swap        (swap),
        .bank_ptr    (bank_ptr),
        .bank_used   (bank_used),
        .s2_re       (s2_re),
        .s2_addr     (s2_addr),
        .s2_q0       (s2_q0),
        .s2_q1       (s2_q1),
        .s2_q2       (s2_q2),
        .s2_q3       (s2_q3),
        .bank_use    (bank_use),
        .chg_win     (chg_win)
    );

    // -------------------------------------------------------------------------
    // Stage 2: shift normaliser -> x_q RAM
    // -------------------------------------------------------------------------
    wire            norm_start;
    wire            norm_busy;
    wire            norm_done;
    wire [CMAX-1:0] dead_mask;
    wire     [20:0] rest_sum;

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
        .s2_q2     (s2_q2),
        .s2_q3     (s2_q3),
        .bank_use  (bank_use),
        .norm_re   (norm_re),
        .norm_addr (norm_addr),
        .norm_mu   (norm_mu),
        .norm_k    (norm_k),
        .xq_we     (xq_we_raw),
        .xq_addr   (xq_addr),
        .xq_data   (xq_data),
        .chg_win   (chg_win),
        .dead_mask (dead_mask),
        .rest_sum  (rest_sum)
    );

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
    // Core sequencer, skip rule, frame header
    // -------------------------------------------------------------------------
    emg_core_ctrl #(
        .CMAX (CMAX)
    ) u_ctrl (
        .clk           (clk_sys),
        .rst_n         (rst_sys_n),
        .cfg_warmup    (cfg_warmup),
        .cfg_err       (cfg_err),
        .cfg_t_rest    (cfg_t_rest),
        .rest_sum      (rest_sum),
        .overrun       (overrun),
        .cs_low        (cs_low),
        .stride_done   (stride_done),
        .norm_start    (norm_start),
        .norm_done     (norm_done),
        .swap          (swap),
        .xq_we_en      (xq_we_en),
        .dead_mask     (dead_mask),
        .frame_start   (frame_start),
        .frame_lo      (|lo_flat),
        .frame_hdr_err (hdr_err),
        .frame_miss    (frame_miss),
        .frame_pub     (frame_pub),
        .f_count       (f_count),
        .f_flags       (f_flags),
        .f_dead        (f_dead),
        .f_rest_sum    (f_rest_sum),
        .status        (status)
    );

endmodule

`default_nettype wire
