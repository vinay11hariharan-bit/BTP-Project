`timescale 1ns/1ps
// tb_mb — 1 to 3 emg_top boards in a chain, each with its own ADC pod, fed with real
// FORS-EMG samples (mb_data/samples.txt). Started by run_mb.py, which sets the
// parameters, writes the configuration file and checks the decisions.
//
//   board 0 (head) channels 0 .. C0-1, board 1 the next C1, board 2 the next C2.
//   CONVST of the head paces the stream: one new 8-channel sample per CONVST.
//   Printed lines:  DEC <n> <class> <time_us>   for every decision at the head.
module tb;
    parameter NB  = 2;              // boards
    parameter C0  = 4, C1 = 4, C2 = 0;
    parameter NS  = 7840;           // samples in the stream
    parameter CFG = "cfg.txt";
    parameter VCD = "mb.vcd";

    // pod timing used by adc_pod_model (ns)
    real t_conv = 20000.0, t_h = 10.0, t_d = 100.0, t_en = 50.0;

    reg clk = 1'b0;
    always #1953.125 clk = ~clk;    // 256 kHz, shared by all boards
    reg arst_n = 1'b0;

    // ------------------------------------------------------------ config bus (shared)
    reg        cfg_we = 1'b0;
    reg [11:0] cfg_addr = 12'd0;
    reg [31:0] cfg_wdata = 32'd0;
    reg [1:0]  cfg_board = 2'd0;

    // ------------------------------------------------------------ sample stream
    reg [63:0] smem [0:NS-1];
    integer    si = 0;
    reg [63:0] cur;
    initial begin
        $readmemh("mb_data/samples.txt", smem);
        cur = smem[0];
    end

    // ------------------------------------------------------------ chain nets
    wire [1:0] id   [0:NB];
    wire       run  [0:NB];
    wire [NB:0] dat, vld;          // dat[j] / vld[j]: from board j to board j-1
    wire [NB-1:0] rdy;
    wire       all_rdy = &rdy;      // wired-AND on the boards
    assign id[0] = 2'd0;  assign run[0] = 1'b0;
    assign dat[NB] = 1'b0; assign vld[NB] = 1'b0;

    wire [NB-1:0] convst, cs_n, sclk, miso, drdy_n, cls_valid, primed, overrun;
    wire [2:0]    cls_out [0:NB-1];

    genvar j;
    generate for (j = 0; j < NB; j = j + 1) begin : brd
        localparam OFF = (j == 0) ? 0 : (j == 1) ? C0 : C0 + C1;

        emg_top u_chip (
            .clk_sys     (clk),
            .arst_n      (arst_n),
            .adc_convst  (convst[j]),
            .adc_drdy_n  (drdy_n[j]),
            .adc_cs_n    (cs_n[j]),
            .adc_sclk    (sclk[j]),
            .adc_miso    (miso[j]),
            .cfg_we      (cfg_we),
            .cfg_addr    (cfg_addr),
            .cfg_wdata   (cfg_wdata),
            .cfg_board   (cfg_board),
            .prs_up_i    (j > 0),
            .prs_dn_i    (j < NB - 1),
            .id_i        (id[j]),
            .id_o        (id[j+1]),
            .run_i       (run[j]),
            .run_o       (run[j+1]),
            .rdy_o       (rdy[j]),
            .all_rdy_i   (all_rdy),
            .up_dat_i    (dat[j+1]),
            .up_vld_i    (vld[j+1]),
            .up_dat_o    (dat[j]),
            .up_vld_o    (vld[j]),
            .class_valid (cls_valid[j]),
            .class_out   (cls_out[j]),
            .primed      (primed[j]),
            .overrun     (overrun[j])
        );

        adc_pod_model u_pod (
            .convst    (convst[j]),
            .cs_n      (cs_n[j]),
            .sclk      (sclk[j]),
            .sdi       (1'b0),
            .sdo       (miso[j]),
            .drdy_n    (drdy_n[j]),
            .sample_in (cur << (8 * OFF))    // this board's channels first
        );

        // this board, by phase: reset/identity, config, start, sampling, Stage 2-3, chain, decision
        initial #0.001 $dumpvars(0,
            u_chip.rst_sys_n, u_chip.u_link.is_head, u_chip.u_link.is_tail, u_chip.my_id,
            u_chip.cfg_we_b, u_chip.cfg_n_ch, u_chip.u_link.nwords,
            u_chip.core_en, u_chip.u_link.head_en, u_chip.run,
            u_chip.frame_start,
            u_chip.stride_done, u_chip.norm_busy, u_chip.mac_busy,
            u_chip.score_valid, u_chip.score_cls, u_chip.score,
            u_chip.u_link.go, u_chip.u_link.word, u_chip.u_link.bit_n, u_chip.u_link.carry,
            u_chip.u_link.shreg, u_chip.u_link.best, u_chip.u_link.best_cls,
            u_chip.u_link.dec_valid, u_chip.u_link.dec_cls, u_chip.class_out);
    end endgenerate

    // Waveform: the shared nets here, each board's own signals in the generate block below.
    // Everything is dumped from 1 ps (all $dumpvars calls must be at the same time step).
    // Only slow or short-lived signals are kept, so the 8 s run stays about 1-2 MB.
    initial #0.001 $dumpvars(0, arst_n, cfg_we, cfg_board, cfg_addr, cfg_wdata,
                                convst, cs_n, rdy, all_rdy, dat, vld, cls_valid, primed, si);

    // next sample after every CONVST of the head (the pods have latched the current one)
    always @(negedge convst[0]) if (arst_n) begin      // (the X -> 0 at reset is not a CONVST)
        si = si + 1;
        if (si < NS) cur = smem[si];
    end

    // ------------------------------------------------------------ configuration
    task cfg_write(input [1:0] brd_id, input [11:0] a, input [31:0] d);
        begin
            @(negedge clk);
            cfg_board = brd_id; cfg_addr = a; cfg_wdata = d; cfg_we = 1'b1;
            @(negedge clk);
            cfg_we = 1'b0;
        end
    endtask

    integer fd, rc, nw;
    reg [31:0] a_b, a_a, a_d;
    integer ndec = 0;
    integer k;

    initial begin
        $dumpfile(VCD);

        #20000 arst_n = 1'b1;
        #20000;
        fd = $fopen(CFG, "r");
        nw = 0;
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h %h\n", a_b, a_a, a_d);
            if (rc == 3) begin cfg_write(a_b[1:0], a_a[11:0], a_d); nw = nw + 1; end
        end
        $fclose(fd);
        $display("CFG %0d writes", nw);
    end

    always @(posedge clk) if (cls_valid[0]) begin
        $display("DEC %0d %0d %0d", ndec, cls_out[0], $time / 1000);
        ndec = ndec + 1;
    end

`ifdef SHOW_SCORES
    always @(posedge clk) begin
        if (brd[0].u_chip.score_valid) $display("S0 %0d %0d", brd[0].u_chip.score_cls, brd[0].u_chip.score);
        if (NB > 1) if (brd[1].u_chip.score_valid) $display("S1 %0d %0d", brd[1].u_chip.score_cls, brd[1].u_chip.score);
    end
`endif

    always @(posedge clk) if (|overrun) begin
        $display("ERROR overrun at %0t", $time); $finish;
    end

    initial begin
        wait (si >= NS);
        #3000000;                       // let the last stride finish (read + Stages 2-3 + chain)
        $display("END samples %0d decisions %0d", si, ndec);
        $finish;
    end
endmodule
