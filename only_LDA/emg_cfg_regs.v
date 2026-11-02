// emg_cfg_regs — M7: trained constants and configuration registers, written at boot.
//
// One write port, synchronous to clk (the host interface in front of it is still TBD).
// Address map, on cfg_we:
//   0x000 - 0x077   norm memory   mem[f]          <= cfg_wdata[26:0] = {mu_f[21:0], k_f[4:0]}
//   0x100 - 0x3CF   weight memory mem[addr-0x100] <= cfg_wdata[7:0]  = W[cls][f] at 0x100 + cls*D + f
//   0x400 - 0x405   bias memory   mem[addr-0x400] <= cfg_wdata[31:0] = b[cls]
//   0x7F0           cfg_n_ch      <= cfg_wdata[4:0]   (reset value 8)
//   0x7F1           core_en       <= cfg_wdata[0]     (reset value 0)
//   0x7F2           cfg_dev_ch    <= cfg_wdata[3:0]   channels per ADS1299 device: 8, 6 or 4 (reset value 8)
//   0x7F3           cfg_shift     <= cfg_wdata[4:0]   24-bit code -> INT8 right shift, 0..16 (reset value 0)
//   0x7F4           ADC command   cmd_byte <= cfg_wdata[7:0], cmd_last <= cfg_wdata[8], cmd_tog flips:
//                                 emg_spi_adc sends the byte to all ADS1299 devices (core_en = 0 only,
//                                 wait >= 10 us between writes, see emg_spi_adc)
//   any other address: ignored
// Write the constants and 0x7F0 - 0x7F3 only while core_en = 0.
//
// Channel-count check: a write to 0x7F0 whose full 32-bit value is 0 or above CMAX (24) sets
// cfg_err = 1 (a chip pin) and turns core_en off. While cfg_err = 1, writing 1 to 0x7F1 is
// refused: core_en stays 0, so the core cannot start with a channel count it cannot handle.
// Writing a valid C (1..24) clears cfg_err. The refused core_en write is not remembered: the
// host must write 1 to 0x7F1 again after fixing C.
//
// The memories are emg_tp_ram instances; their read ports are the norm_* / w_* / b_*
// ports (1-cycle latency). norm_mu = word[26:5], norm_k = word[4:0].
`default_nettype none

module emg_cfg_regs #(
    parameter integer CMAX = 24               // largest channel count the core can handle
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               cfg_we,
    input  wire [11:0]        cfg_addr,
    input  wire [31:0]        cfg_wdata,
    output reg  [4:0]         cfg_n_ch,
    output reg                core_en,
    output reg                cfg_err,     // 1 = the last C written was 0 or above CMAX
    output reg  [3:0]         cfg_dev_ch,  // ADC front end (emg_spi_adc)
    output reg  [4:0]         cfg_shift,
    output reg  [7:0]         cmd_byte,
    output reg                cmd_last,
    output reg                cmd_tog,
    input  wire               norm_re,     // Stage 2
    input  wire [6:0]         norm_addr,
    output wire [21:0]        norm_mu,
    output wire [4:0]         norm_k,
    input  wire               w_re,        // Stage 3
    input  wire [9:0]         w_addr,
    output wire signed [7:0]  w_data,
    input  wire               b_re,        // Stage 3
    input  wire [2:0]         b_addr,
    output wire signed [31:0] b_data
);
    // ---------------------------------------------------------------- address decode
    wire        sel_norm = (cfg_addr[11:7] == 5'd0)    && (cfg_addr[6:0] < 7'd120);   // 0x000-0x077
    wire        sel_w    = (cfg_addr >= 12'h100)       && (cfg_addr < 12'h3D0);       // 0x100-0x3CF
    wire        sel_b    = (cfg_addr[11:3] == 9'h080)  && (cfg_addr[2:0] < 3'd6);     // 0x400-0x405
    wire [11:0] w_off    = cfg_addr - 12'h100;                                       // 0..719

    // ---------------------------------------------------------------- memories
    wire [26:0] norm_q;

    emg_tp_ram #(.W(27), .DEPTH(120), .AW(7)) u_norm_mem (
        .clk(clk), .we(cfg_we & sel_norm), .waddr(cfg_addr[6:0]), .wdata(cfg_wdata[26:0]),
        .re(norm_re), .raddr(norm_addr), .rdata(norm_q));

    emg_tp_ram #(.W(8), .DEPTH(720), .AW(10)) u_w_mem (
        .clk(clk), .we(cfg_we & sel_w), .waddr(w_off[9:0]), .wdata(cfg_wdata[7:0]),
        .re(w_re), .raddr(w_addr), .rdata(w_data));

    emg_tp_ram #(.W(32), .DEPTH(6), .AW(3)) u_b_mem (
        .clk(clk), .we(cfg_we & sel_b), .waddr(cfg_addr[2:0]), .wdata(cfg_wdata),
        .re(b_re), .raddr(b_addr), .rdata(b_data));

    assign norm_mu = norm_q[26:5];
    assign norm_k  = norm_q[4:0];

    // ---------------------------------------------------------------- registers
    wire c_bad_w = (cfg_wdata == 32'd0) || (cfg_wdata > CMAX);    // C being written is unusable

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_n_ch   <= 5'd8;
            core_en    <= 1'b0;
            cfg_err    <= 1'b0;
            cfg_dev_ch <= 4'd8;
            cfg_shift  <= 5'd0;
            cmd_byte   <= 8'd0;
            cmd_last   <= 1'b0;
            cmd_tog    <= 1'b0;
        end else if (cfg_we) begin
            if (cfg_addr == 12'h7F0) begin
                cfg_n_ch <= cfg_wdata[4:0];
                cfg_err  <= c_bad_w;
                if (c_bad_w) core_en <= 1'b0;            // stop the core at once
            end
            if (cfg_addr == 12'h7F1) core_en    <= cfg_wdata[0] & ~cfg_err;   // refused while C is bad
            if (cfg_addr == 12'h7F2) cfg_dev_ch <= cfg_wdata[3:0];
            if (cfg_addr == 12'h7F3) cfg_shift  <= cfg_wdata[4:0];
            if (cfg_addr == 12'h7F4) begin
                cmd_byte <= cfg_wdata[7:0];
                cmd_last <= cfg_wdata[8];
                cmd_tog  <= ~cmd_tog;
            end
        end
    end
endmodule

`default_nettype wire
