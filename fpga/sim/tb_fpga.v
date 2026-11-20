`timescale 1ns/1ps
// tb_fpga - drives emg_axi over AXI4-Lite the way the Python driver does:
// writes the config and the samples, starts the run, waits for done and reads the decisions.
// From fpga/mem:  iverilog -g2005 -DSIM -s tb_fpga -o tb.vvp ../../RTL/*.v ../rtl/*.v ../sim/tb_fpga.v
//                 vvp -n tb.vvp
module tb_fpga;
    parameter NS = 7840, NCFG = 171, NDEC = 79;

    reg clk = 1'b0, rstn = 1'b0;
    always #50 clk = ~clk;                        // 10 MHz

    reg  [18:0] awaddr, araddr;
    reg  [31:0] wdata;
    reg         awvalid = 0, wvalid = 0, arvalid = 0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [31:0] rdata;
    wire [1:0]  bresp, rresp;
    wire [3:0]  led;

    emg_axi dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'd0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
        .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(1'b1),
        .led(led));

    task axi_wr(input [18:0] a, input [31:0] d);
        begin
            @(posedge clk) begin awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1; end
            while (awvalid || wvalid) @(posedge clk) begin
                if (awready) awvalid <= 0;
                if (wready)  wvalid  <= 0;
            end
            while (!bvalid) @(posedge clk);
        end
    endtask

    reg [31:0] rd;
    task axi_rd(input [18:0] a);
        begin
            @(posedge clk) begin araddr <= a; arvalid <= 1; end
            @(posedge clk) while (!arready) @(posedge clk);
            arvalid <= 0;
            while (!rvalid) @(posedge clk);
            rd = rdata;
        end
    endtask

    reg [63:0] smp [0:NS-1];
    reg [45:0] cfg [0:NCFG-1];
    reg [2:0]  exp_cls [0:NDEC-1];
    integer i, bad;

    initial begin
        $readmemh("samples.mem",  smp);
        $readmemh("cfg_4_4.mem",  cfg);
        $readmemh("expected.mem", exp_cls);
        #1000 rstn = 1;

        axi_rd(19'h0001C);
        if (rd !== 32'h454D4701) begin $display("FPGA_TOP FAIL: ID = %h", rd); $finish; end
        for (i = 0; i < NCFG; i = i + 1) begin
            axi_wr(19'h01000 + 8*i,     cfg[i][31:0]);
            axi_wr(19'h01000 + 8*i + 4, cfg[i][45:32]);
        end
        for (i = 0; i < NS; i = i + 1) begin
            axi_wr(19'h40000 + 8*i,     smp[i][63:32]);
            axi_wr(19'h40000 + 8*i + 4, smp[i][31:0]);
        end
        axi_wr(19'h00008, NS);
        axi_wr(19'h0000C, NCFG);
        axi_wr(19'h00010, NDEC);
        axi_wr(19'h00000, 32'd3);                 // run, fast

        rd = 0;
        while (!rd[1]) begin
            repeat (2000) @(posedge clk);
            axi_rd(19'h00004);
        end
        $display("status %h: %0d decisions, overrun %0d", rd, rd[25:16], rd[2]);

        bad = rd[2];
        for (i = 0; i < NDEC; i = i + 1) begin
            axi_rd(19'h02000 + 4*i);
            if (rd[2:0] !== exp_cls[i]) begin
                bad = bad + 1;
                $display("DEC %0d at sample %0d: %0d expected %0d  <-- MISMATCH", i, rd[31:16], rd[2:0], exp_cls[i]);
            end
        end
        if (bad == 0) $display("FPGA_TOP PASS: %0d decisions equal to the golden model", NDEC);
        else          $display("FPGA_TOP FAIL: %0d bad", bad);
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("FPGA_TOP TIMEOUT");
        $finish;
    end
endmodule
