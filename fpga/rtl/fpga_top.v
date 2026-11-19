`timescale 1ns/1ps
// fpga_top - two emg_top boards (head + tail) on one PYNQ-Z1/Z2, fed with real FORS-EMG data
//
// Everything the testbench tb_mb.v does is done here in hardware:
//   * cfg ROM     : the 171 configuration writes (cfg_4_4.mem), sent one every 2 cycles after reset
//   * sample ROM  : 7840 samples of subject 6, trial 1 (samples.mem); a new sample after every
//                   CONVST of the head, one fpga_pod per board, each seeing its own 4 channels first
//   * checker     : every decision of the head is compared with the golden model (expected.mem)
// Board 0 has channels 0-3, board 1 channels 4-7 (4 + 4 split).
//
// Buttons / switches / LEDs (PYNQ-Z1 and Z2 use the same pins):
//   btn0 : restart the run           sw0 : 0 = real time (256 kHz, a decision every 99.5 ms)
//                                          1 = fast (10 MHz, the whole stream in ~0.2 s)
//   led0 : toggles at every decision  led1 : all 79 decisions done
//   led2 : PASS (done, every decision equal to the golden model)
//   led3 : FAIL (a decision differed, or the chip reported overrun)
//   RGB LED 4 : current gesture  red = Hand Close, green = Hand Open, blue = Wrist Flexion,
//               white = Wrist Extension (dimmed)
module fpga_top #(
    parameter NS   = 7840,            // samples in samples.mem
    parameter NCFG = 171,             // writes in cfg_4_4.mem
    parameter NDEC = 79,              // decisions in expected.mem
    parameter C0   = 4                // channels of board 0 (board 1 gets the rest of the 8)
) (
    input  wire       sysclk,         // 125 MHz
    input  wire [1:0] btn,
    input  wire [1:0] sw,
    output wire [3:0] led,
    output wire       led4_r,
    output wire       led4_g,
    output wire       led4_b
);
    // ------------------------------------------------------------------ clocks and reset
    wire clk10, clk, locked;
    fpga_clk u_clk (.clk125(sysclk), .fast(sw[0]), .clk10(clk10), .clk_emg(clk), .locked(locked));

    // power-on / button reset, released synchronously and held for 16 cycles
    reg [4:0] rst_cnt = 5'd0;
    wire      rst_req = btn[0] | ~locked;
    always @(posedge clk or posedge rst_req)
        if (rst_req)          rst_cnt <= 5'd0;
        else if (!rst_cnt[4]) rst_cnt <= rst_cnt + 1'b1;
    wire rst_n = rst_cnt[4];

    // ------------------------------------------------------------------ ROMs
    reg [45:0] cfg_rom [0:NCFG-1];    // {board[1:0], addr[11:0], data[31:0]}
    reg [63:0] smp_rom [0:NS-1];      // channel 0 in the top byte
    reg [2:0]  exp_rom [0:NDEC-1];    // golden decision of stride 1..NDEC
    initial begin
        $readmemh("cfg_4_4.mem",  cfg_rom);
        $readmemh("samples.mem",  smp_rom);
        $readmemh("expected.mem", exp_rom);
    end

    // ------------------------------------------------------------------ configuration
    reg [7:0]  ci;
    reg        cfg_we, cfg_done, phase;
    reg [11:0] cfg_addr;
    reg [31:0] cfg_wdata;
    reg [1:0]  cfg_board;
    wire [45:0] cfg_word = cfg_rom[ci];                 // small: distributed ROM

    // wait 16 cycles after reset: emg_top synchronises its reset internally and would miss
    // the first writes otherwise
    reg [4:0] wait_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ci <= 8'd0; phase <= 1'b0; cfg_done <= 1'b0; cfg_we <= 1'b0; wait_cnt <= 5'd0;
            cfg_addr <= 12'd0; cfg_wdata <= 32'd0; cfg_board <= 2'd0;
        end else if (!wait_cnt[4]) begin
            wait_cnt <= wait_cnt + 1'b1;
        end else if (!cfg_done) begin
            phase <= ~phase;
            if (!phase) begin
                {cfg_board, cfg_addr, cfg_wdata} <= cfg_word;
                cfg_we <= 1'b1;
            end else begin
                cfg_we <= 1'b0;
                if (ci == NCFG - 1) cfg_done <= 1'b1;
                else                ci <= ci + 1'b1;
            end
        end else begin
            cfg_we <= 1'b0;
        end
    end

    // ------------------------------------------------------------------ the two boards
    wire [1:0] convst, cs_n, sclk, miso, drdy_n, cls_valid, primed, overrun, rdy;
    wire [2:0] cls_out0, cls_out1;
    wire [1:0] id01, id_tail_o;
    wire       run01, run_tail_o;
    wire       dat10, vld10, dat_head_o, vld_head_o;
    wire       all_rdy = &rdy;                            // the wired-AND of the board

    emg_top u_head (
        .clk_sys(clk), .arst_n(rst_n),
        .adc_convst(convst[0]), .adc_drdy_n(drdy_n[0]), .adc_cs_n(cs_n[0]), .adc_sclk(sclk[0]), .adc_miso(miso[0]),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata), .cfg_board(cfg_board),
        .prs_up_i(1'b0), .prs_dn_i(1'b1), .id_i(2'd0), .id_o(id01), .run_i(1'b0), .run_o(run01),
        .rdy_o(rdy[0]), .all_rdy_i(all_rdy),
        .up_dat_i(dat10), .up_vld_i(vld10), .up_dat_o(dat_head_o), .up_vld_o(vld_head_o),
        .class_valid(cls_valid[0]), .class_out(cls_out0), .primed(primed[0]), .overrun(overrun[0]));

    emg_top u_tail (
        .clk_sys(clk), .arst_n(rst_n),
        .adc_convst(convst[1]), .adc_drdy_n(drdy_n[1]), .adc_cs_n(cs_n[1]), .adc_sclk(sclk[1]), .adc_miso(miso[1]),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata), .cfg_board(cfg_board),
        .prs_up_i(1'b1), .prs_dn_i(1'b0), .id_i(id01), .id_o(id_tail_o), .run_i(run01), .run_o(run_tail_o),
        .rdy_o(rdy[1]), .all_rdy_i(all_rdy),
        .up_dat_i(1'b0), .up_vld_i(1'b0), .up_dat_o(dat10), .up_vld_o(vld10),
        .class_valid(cls_valid[1]), .class_out(cls_out1), .primed(primed[1]), .overrun(overrun[1]));

    // ------------------------------------------------------------------ sample stream and pods
    reg [12:0] si;
    reg [63:0] cur;
    reg        convst0_d;
    always @(posedge clk) cur <= smp_rom[si];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            si <= 13'd0; convst0_d <= 1'b0;
        end else begin
            convst0_d <= convst[0];
            if (convst[0] & ~convst0_d && si != NS - 1)   // the pods latch cur on this edge
                si <= si + 1'b1;
        end
    end

    fpga_pod u_pod0 (.clk(clk), .rst_n(rst_n), .convst(convst[0]), .cs_n(cs_n[0]), .sclk(sclk[0]),
                     .sdo(miso[0]), .drdy_n(drdy_n[0]), .sample_in(cur));
    fpga_pod u_pod1 (.clk(clk), .rst_n(rst_n), .convst(convst[1]), .cs_n(cs_n[1]), .sclk(sclk[1]),
                     .sdo(miso[1]), .drdy_n(drdy_n[1]), .sample_in(cur << (8 * C0)));

    // ------------------------------------------------------------------ checker
    reg [6:0] ndec;
    reg       done, fail, beat;
    reg [2:0] last_cls;
    reg [2:0] exp_cls;
    always @(posedge clk) exp_cls <= exp_rom[ndec];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ndec <= 7'd0; done <= 1'b0; fail <= 1'b0; beat <= 1'b0; last_cls <= 3'd0;
        end else begin
            if (|overrun) fail <= 1'b1;
            if (cls_valid[0]) begin
                last_cls <= cls_out0;
                beat     <= ~beat;
                if (!done) begin
                    if (cls_out0 != exp_cls) fail <= 1'b1;
                    if (ndec == NDEC - 1) done <= 1'b1;
                    else                  ndec <= ndec + 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------------ LEDs
    assign led = {fail, done & ~fail, done, beat};

    reg [3:0] pwm = 4'd0;                                   // RGB at 1/16 brightness
    always @(posedge clk10) pwm <= pwm + 1'b1;
    wire on = (pwm == 4'd0) & rst_n;
    assign led4_r = on & (last_cls == 3'd0 || last_cls == 3'd3);
    assign led4_g = on & (last_cls == 3'd1 || last_cls == 3'd3);
    assign led4_b = on & (last_cls == 3'd2 || last_cls == 3'd3);

`ifdef SIM
    // for the simulation testbench
    reg sim_bad = 1'b0;                                  // a decision that is X or differs
    always @(posedge clk) if (cls_valid[0] && !done && cls_out0 !== exp_cls) sim_bad <= 1'b1;
    always @(posedge clk) if (cls_valid[0])
        $display("DEC %0d %0d expected %0d%s", ndec, cls_out0, exp_cls,
                 (!done && cls_out0 != exp_cls) ? "  <-- MISMATCH" : "");
`endif
endmodule
