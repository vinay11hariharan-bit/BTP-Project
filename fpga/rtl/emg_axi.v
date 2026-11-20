`timescale 1ns/1ps
// emg_axi - the two-board test bench behind an AXI4-Lite port, so the Zynq (PYNQ / Python)
// can load a trial, run it and read the decisions back.
//
// Byte addresses (32-bit accesses only):
//   0x00000  CTRL    rw  [0] run (0 = hold in reset, 1 = start)  [1] fast (10 MHz instead of 256 kHz)
//   0x00004  STATUS  r   [0] config done  [1] done  [2] overrun  [25:16] decisions so far
//   0x00008  NSAMP   rw  samples to play (up to 32768)
//   0x0000C  NCFG    rw  config writes to send (up to 256)
//   0x00010  NWANT   rw  decisions to record before done (up to 1023)
//   0x0001C  ID      r   0x454D4701
//   0x01000  CFG     w   write i: +8i = data, +8i+4 = {board[1:0], addr[11:0]}
//   0x02000  DEC     r   decision i: +4i = {samples converted[15:0], 13'd0, class[2:0]}
//   0x40000  SAMPLES w   sample i: +8i = ch0..ch3, +8i+4 = ch4..ch7 (ch0 / ch4 in bits 31:24)
// Change CFG and SAMPLES only while run = 0.
module emg_axi (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire        s_axi_aclk,            // FCLK0, 10 MHz
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        s_axi_aresetn,

    input  wire [18:0] s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [18:0] s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,

    output wire [3:0]  led                    // LD0 decision, LD1 config done, LD2 done, LD3 overrun
);
    wire clk = s_axi_aclk;

    // ------------------------------------------------------------------ write channel
    reg        aw_ok, w_ok;
    reg [18:0] waddr;
    reg [31:0] wdata;
    wire       wr = aw_ok & w_ok;

    assign s_axi_awready = ~aw_ok & ~s_axi_bvalid;
    assign s_axi_wready  = ~w_ok  & ~s_axi_bvalid;
    assign s_axi_bresp   = 2'b00;

    always @(posedge clk) begin
        if (!s_axi_aresetn) begin
            aw_ok <= 1'b0; w_ok <= 1'b0; s_axi_bvalid <= 1'b0;
        end else begin
            if (s_axi_awvalid && s_axi_awready) begin waddr <= s_axi_awaddr; aw_ok <= 1'b1; end
            if (s_axi_wvalid  && s_axi_wready)  begin wdata <= s_axi_wdata;  w_ok  <= 1'b1; end
            if (wr)                                 begin aw_ok <= 1'b0; w_ok <= 1'b0; s_axi_bvalid <= 1'b1; end
            if (s_axi_bvalid && s_axi_bready)   s_axi_bvalid <= 1'b0;
        end
    end

    wire wr_reg = wr && waddr[18:12] == 7'd0;
    wire wr_cfg = wr && waddr[18:12] == 7'd1;
    wire wr_smp = wr && waddr[18];

    // ------------------------------------------------------------------ registers
    reg        run, fast;
    reg [15:0] nsamp;
    reg [8:0]  ncfg;
    reg [9:0]  nwant;

    always @(posedge clk) begin
        if (!s_axi_aresetn) begin
            run <= 1'b0; fast <= 1'b0; nsamp <= 16'd0; ncfg <= 9'd0; nwant <= 10'd0;
        end else if (wr_reg) begin
            case (waddr[4:2])
                3'd0: {fast, run} <= wdata[1:0];
                3'd2: nsamp       <= wdata[15:0];
                3'd3: ncfg        <= wdata[8:0];
                3'd4: nwant       <= wdata[9:0];
                default: ;
            endcase
        end
    end

    // ------------------------------------------------------------------ memories
    wire        clk_emg;
    wire [7:0]  cfg_idx;
    wire [14:0] smp_idx;
    reg  [63:0] smp_word;
    wire        dec_we;
    wire [9:0]  dec_idx;
    wire [31:0] dec_word;

    reg [31:0] cfg_dat [0:255];               // distributed RAM, read without a clock
    reg [13:0] cfg_ba  [0:255];
    always @(posedge clk)
        if (wr_cfg) begin
            if (waddr[2]) cfg_ba[waddr[10:3]]  <= wdata[13:0];
            else          cfg_dat[waddr[10:3]] <= wdata;
        end

    reg [31:0] smp_hi [0:32767];              // block RAM, written by the PS, read by the chip side
    reg [31:0] smp_lo [0:32767];
    always @(posedge clk)
        if (wr_smp) begin
            if (waddr[2]) smp_lo[waddr[17:3]] <= wdata;
            else          smp_hi[waddr[17:3]] <= wdata;
        end
    always @(posedge clk_emg) smp_word <= {smp_hi[smp_idx], smp_lo[smp_idx]};

    reg [31:0] dec_ram [0:1023];
    always @(posedge clk_emg) if (dec_we) dec_ram[dec_idx] <= dec_word;

    // ------------------------------------------------------------------ the bench
    wire       cfg_done, done, overrun, beat;
    wire [9:0] ndec;

    emg_clk_gate u_gate (.clk(clk), .fast(fast), .clk_emg(clk_emg));

    emg_bench #(.C0(4)) u_bench (
        .clk(clk_emg), .rst_req(~run | ~s_axi_aresetn),
        .ncfg(ncfg), .nsamp(nsamp), .nwant(nwant),
        .cfg_idx(cfg_idx), .cfg_word({cfg_ba[cfg_idx], cfg_dat[cfg_idx]}),
        .smp_idx(smp_idx), .smp_word(smp_word),
        .dec_we(dec_we), .dec_idx(dec_idx), .dec_word(dec_word),
        .cfg_done(cfg_done), .done(done), .overrun(overrun), .ndec(ndec), .beat(beat));

    assign led = {overrun, done, cfg_done, beat};

    // ------------------------------------------------------------------ read channel
    reg        r_pend;
    reg [18:0] raddr;
    reg [31:0] dec_q;

    assign s_axi_arready = ~r_pend & ~s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;

    always @(posedge clk) begin
        if (!s_axi_aresetn) begin
            r_pend <= 1'b0; s_axi_rvalid <= 1'b0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                raddr  <= s_axi_araddr;
                dec_q  <= dec_ram[s_axi_araddr[11:2]];
                r_pend <= 1'b1;
            end
            if (r_pend) begin
                r_pend       <= 1'b0;
                s_axi_rvalid <= 1'b1;
                if (raddr[18:12] == 7'd2)
                    s_axi_rdata <= dec_q;
                else case (raddr[4:2])
                    3'd0: s_axi_rdata <= {30'd0, fast, run};
                    3'd1: s_axi_rdata <= {6'd0, ndec, 13'd0, overrun, done, cfg_done};
                    3'd2: s_axi_rdata <= {16'd0, nsamp};
                    3'd3: s_axi_rdata <= {23'd0, ncfg};
                    3'd4: s_axi_rdata <= {22'd0, nwant};
                    3'd7: s_axi_rdata <= 32'h454D4701;
                    default: s_axi_rdata <= 32'd0;
                endcase
            end
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
        end
    end
endmodule
