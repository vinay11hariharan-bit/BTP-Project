`timescale 1ns/1ps
// emg_bench - two emg_top boards (head + tail, C0 + 8-C0 channels) and everything around them
//
// Runs on the chip clock. While rst_req is low:
//   1. waits 16 cycles (emg_top synchronises its reset), then sends the ncfg config writes,
//      one every 2 cycles, reading them from the cfg memory (cfg_idx -> cfg_word)
//   2. plays nsamp samples from the sample memory: a new sample at every CONVST of the head,
//      one fpga_pod per board, board 1 seeing its own channels first
//      (after the last one the last sample is held)
//   3. writes the first nwant decisions of the head to the decision memory as
//      {samples converted so far, class}; done goes high after the last of them.
module emg_bench #(
    parameter C0 = 4                          // channels of board 0
) (
    input  wire        clk,
    input  wire        rst_req,               // asynchronous, high = hold everything in reset
    input  wire [8:0]  ncfg,
    input  wire [15:0] nsamp,
    input  wire [9:0]  nwant,

    output wire [7:0]  cfg_idx,
    input  wire [45:0] cfg_word,              // {board[1:0], addr[11:0], data[31:0]}, read without a clock
    output reg  [14:0] smp_idx,
    input  wire [63:0] smp_word,              // registered read of smp_idx, channel 0 in the top byte

    output reg         dec_we,
    output reg  [9:0]  dec_idx,
    output reg  [31:0] dec_word,              // {sample index[15:0], 13'd0, class[2:0]}

    output reg         cfg_done,
    output reg         done,
    output reg         overrun,
    output reg  [9:0]  ndec,
    output reg         beat                   // toggles at every decision
);
    // ------------------------------------------------------------------ reset
    reg [4:0] rst_cnt;
    always @(posedge clk or posedge rst_req)
        if (rst_req)          rst_cnt <= 5'd0;
        else if (!rst_cnt[4]) rst_cnt <= rst_cnt + 1'b1;
    wire rst_n = rst_cnt[4];

    // ------------------------------------------------------------------ configuration
    reg [7:0]  ci;
    reg [4:0]  wait_cnt;
    reg        phase, cfg_we;
    reg [11:0] cfg_addr;
    reg [31:0] cfg_wdata;
    reg [1:0]  cfg_board;
    assign cfg_idx = ci;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ci <= 8'd0; wait_cnt <= 5'd0; phase <= 1'b0; cfg_we <= 1'b0; cfg_done <= 1'b0;
            cfg_addr <= 12'd0; cfg_wdata <= 32'd0; cfg_board <= 2'd0;
        end else if (!wait_cnt[4]) begin
            wait_cnt <= wait_cnt + 1'b1;
        end else if (!cfg_done) begin
            phase <= ~phase;
            if (!phase) begin
                {cfg_board, cfg_addr, cfg_wdata} <= cfg_word;
                cfg_we <= (ncfg != 9'd0);
            end else begin
                cfg_we <= 1'b0;
                if (ci + 1'b1 >= ncfg) cfg_done <= 1'b1;
                else                   ci <= ci + 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------ the two boards
    wire [1:0] convst, cs_n, sclk, miso, drdy_n, cls_valid, primed, ovr, rdy;
    wire [2:0] cls0, cls1;
    wire [1:0] id01, id_tail;
    wire       run01, run_tail, dat10, vld10, dat_head, vld_head;
    wire       all_rdy = &rdy;                // the wired-AND of the board

    emg_top u_head (
        .clk_sys(clk), .arst_n(rst_n),
        .adc_convst(convst[0]), .adc_drdy_n(drdy_n[0]), .adc_cs_n(cs_n[0]), .adc_sclk(sclk[0]), .adc_miso(miso[0]),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata), .cfg_board(cfg_board),
        .prs_up_i(1'b0), .prs_dn_i(1'b1), .id_i(2'd0), .id_o(id01), .run_i(1'b0), .run_o(run01),
        .rdy_o(rdy[0]), .all_rdy_i(all_rdy),
        .up_dat_i(dat10), .up_vld_i(vld10), .up_dat_o(dat_head), .up_vld_o(vld_head),
        .class_valid(cls_valid[0]), .class_out(cls0), .primed(primed[0]), .overrun(ovr[0]));

    emg_top u_tail (
        .clk_sys(clk), .arst_n(rst_n),
        .adc_convst(convst[1]), .adc_drdy_n(drdy_n[1]), .adc_cs_n(cs_n[1]), .adc_sclk(sclk[1]), .adc_miso(miso[1]),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata), .cfg_board(cfg_board),
        .prs_up_i(1'b1), .prs_dn_i(1'b0), .id_i(id01), .id_o(id_tail), .run_i(run01), .run_o(run_tail),
        .rdy_o(rdy[1]), .all_rdy_i(all_rdy),
        .up_dat_i(1'b0), .up_vld_i(1'b0), .up_dat_o(dat10), .up_vld_o(vld10),
        .class_valid(cls_valid[1]), .class_out(cls1), .primed(primed[1]), .overrun(ovr[1]));

    // ------------------------------------------------------------------ sample stream
    reg        convst_d;
    reg [15:0] nconv;                         // CONVSTs seen so far
    wire       conv_rise = convst[0] & ~convst_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            smp_idx <= 15'd0; convst_d <= 1'b0; nconv <= 16'd0;
        end else begin
            convst_d <= convst[0];
            if (conv_rise) begin              // the pods latch smp_word on this edge
                nconv <= nconv + 1'b1;
                if (smp_idx + 1'b1 < nsamp) smp_idx <= smp_idx + 1'b1;
            end
        end
    end

    fpga_pod u_pod0 (.clk(clk), .rst_n(rst_n), .convst(convst[0]), .cs_n(cs_n[0]), .sclk(sclk[0]),
                     .sdo(miso[0]), .drdy_n(drdy_n[0]), .sample_in(smp_word));
    fpga_pod u_pod1 (.clk(clk), .rst_n(rst_n), .convst(convst[1]), .cs_n(cs_n[1]), .sclk(sclk[1]),
                     .sdo(miso[1]), .drdy_n(drdy_n[1]), .sample_in(smp_word << (8 * C0)));

    // ------------------------------------------------------------------ decisions
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dec_we <= 1'b0; dec_idx <= 10'd0; dec_word <= 32'd0;
            ndec <= 10'd0; done <= 1'b0; overrun <= 1'b0; beat <= 1'b0;
        end else begin
            dec_we <= 1'b0;
            if (|ovr) overrun <= 1'b1;
            if (cls_valid[0] && !done) begin
                dec_we   <= 1'b1;
                dec_idx  <= ndec;
                dec_word <= {nconv, 13'd0, cls0};
                ndec     <= ndec + 1'b1;
                beat     <= ~beat;
                if (ndec + 1'b1 >= nwant) done <= 1'b1;
            end
        end
    end
endmodule
