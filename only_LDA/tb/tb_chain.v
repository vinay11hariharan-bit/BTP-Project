`timescale 1ns/1ps
// =============================================================================
// tb  -  chip + ADS1299 daisy chain (testbench top level, driven by tb/test_emg_chain.py)
//
//          DAISY_IN     DOUT  DAISY_IN     DOUT              DAISY_IN    DOUT
//    0 ──▶ [ dev NPOD-1 ] ──▶ [ dev NPOD-2 ] ──▶ ... ──▶ [ dev 0 ] ──▶ adc_miso
//
//  * Device 0 is wired to MISO, so it sends first: its electrodes are channels
//    0 .. PCH-1, device 1 has channels PCH .. 2*PCH-1, and so on.
//  * START (= adc_convst), CS_N, SCLK and DIN (= adc_mosi) go to every device.
//  * The chip has one DRDY_N pin. Here it goes low only when every device's DRDY_N is
//    low (OR of the active-low lines), as a board would do with a gate. DRDY_MISSING = 1
//    holds it high, as if the DRDY_N line were pulled up and no device answered.
//  * The DAISY_IN of the last device is tied to 0, so channels beyond the real devices
//    read as 0 and their status words as 0000 (not 1100).
//  * Python writes each electrode's 24-bit code to `codes` and its lead-off state to
//    `loffp` / `loffn` before each START. Electrode c: codes[24c +: 24], loffp[c], loffn[c].
//
// Timing (ns), from the ADS1299 datasheet at DVDD = 3.3 V and a 2.048 MHz clock:
//    t_clk   488.28   one ADS1299 clock period (tCLK)
//    t_conv  504400   single-shot conversion at the 8 kSPS setting (DR = 001): 1033 tCLK
//    t_h     10       old DOUT bit held after an SCLK rise (tDOHD)
//    t_d     17       new DOUT bit valid after an SCLK rise (tDOPD max; 32 at DVDD 1.8 V)
//    t_csdod 20       CS_N fall -> DOUT driven (tCSDOD max)
//    t_csdoz 10       CS_N rise -> DOUT high-Z (tCSDOZ max)
// The chip samples MISO 125 ns after each SCLK rise, so t_d must stay below 125 ns.
// ads1299_model reads these as tb.t_clk, tb.t_conv, tb.t_h, tb.t_d, tb.t_csdod, tb.t_csdoz.
// =============================================================================
`default_nettype none

module tb #(
    parameter integer NPOD         = 1,      // ADS1299 devices in the chain
    parameter integer PCH          = 8,      // electrodes per device: 8, 6 or 4
    parameter integer DRDY_MISSING = 0,      // 1 = DRDY_N never falls
    parameter integer CMAX         = 24,
    parameter integer SAMPLE_DIV   = 8120,
    parameter real    t_clk        = 488.28,
    parameter real    t_conv       = 504400.0,
    parameter real    t_h          = 10.0,
    parameter real    t_d          = 17.0,
    parameter real    t_csdod      = 20.0,
    parameter real    t_csdoz      = 10.0
) ();
    localparam integer NPHYS = NPOD * PCH;   // electrodes physically present

    // ---------------------------------------------------------------- driven by Python
    reg                 clk_sys   = 1'b0;
    reg                 clk_spi   = 1'b0;
    reg                 arst_n    = 1'b0;
    reg                 cfg_we    = 1'b0;
    reg  [11:0]         cfg_addr  = 12'd0;
    reg  [31:0]         cfg_wdata = 32'd0;
    reg  [24*NPHYS-1:0] codes     = {24*NPHYS{1'b0}};
    reg  [NPHYS-1:0]    loffp     = {NPHYS{1'b0}};
    reg  [NPHYS-1:0]    loffn     = {NPHYS{1'b0}};
    reg                 drdy_block = 1'b0;   // 1 = hold DRDY_N high (Python uses it to lose one sample)

    // ---------------------------------------------------------------- chip pins
    wire       adc_convst, adc_drdy_n, adc_cs_n, adc_sclk, adc_miso, adc_mosi;
    wire       class_valid, primed, overrun, chan_fault, link_err, cfg_err;
    wire [2:0] class_out;

    emg_top #(
        .CMAX       (CMAX),
        .SAMPLE_DIV (SAMPLE_DIV)
    ) u_top (
        .clk_sys     (clk_sys),
        .clk_spi     (clk_spi),
        .arst_n      (arst_n),
        .adc_convst  (adc_convst),
        .adc_drdy_n  (adc_drdy_n),
        .adc_cs_n    (adc_cs_n),
        .adc_sclk    (adc_sclk),
        .adc_miso    (adc_miso),
        .adc_mosi    (adc_mosi),
        .cfg_we      (cfg_we),
        .cfg_addr    (cfg_addr),
        .cfg_wdata   (cfg_wdata),
        .class_valid (class_valid),
        .class_out   (class_out),
        .primed      (primed),
        .overrun     (overrun),
        .chan_fault  (chan_fault),
        .link_err    (link_err),
        .cfg_err     (cfg_err)
    );

    // ---------------------------------------------------------------- device chain
    wire [NPOD:0]   chain;   // chain[k] = DOUT of device k = DAISY_IN of device k-1
    wire [NPOD-1:0] drdy;

    assign chain[NPOD] = 1'b0;                          // DAISY_IN of the last device
    assign adc_miso    = chain[0];
    assign adc_drdy_n  = (DRDY_MISSING || drdy_block) ? 1'b1 : |drdy;

    // per-device state for the Python checks
    wire [NPOD-1:0]   pod_rdatac, pod_chset_ok;
    wire [8*NPOD-1:0] pod_config1, pod_config3, pod_config4, pod_loff_sensp, pod_loff_sensn;
    wire [32*NPOD-1:0] pod_viol;

    genvar k;
    generate
        for (k = 0; k < NPOD; k = k + 1) begin : pod
            ads1299_model #(.NCH(PCH)) u_pod (
                .start    (adc_convst),
                .cs_n     (adc_cs_n),
                .sclk     (adc_sclk),
                .din      (adc_mosi),
                .daisy_in (chain[k+1]),
                .dout     (chain[k]),
                .drdy_n   (drdy[k]),
                .codes_in (codes[24*PCH*k +: 24*PCH]),
                .loffp_in (loffp[PCH*k +: PCH]),
                .loffn_in (loffn[PCH*k +: PCH])
            );
            assign pod_rdatac[k]               = u_pod.rdatac;
            assign pod_chset_ok[k]             = u_pod.chset_ok;
            assign pod_config1[8*k +: 8]       = u_pod.config1;
            assign pod_config3[8*k +: 8]       = u_pod.config3;
            assign pod_config4[8*k +: 8]       = u_pod.config4;
            assign pod_loff_sensp[8*k +: 8]    = u_pod.loff_sensp;
            assign pod_loff_sensn[8*k +: 8]    = u_pod.loff_sensn;
            assign pod_viol[32*k +: 32]        = u_pod.viol;
        end
    endgenerate
endmodule

`default_nettype wire
