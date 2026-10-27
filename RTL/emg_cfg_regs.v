// emg_cfg_regs — M7: trained constants and configuration registers, written at boot.
//
// One write port, synchronous to clk (the host interface in front of it is still TBD).
// Address map, on cfg_we:
//   0x000 - 0x077   norm memory   mem[f]          <= cfg_wdata[26:0] = {mu_f[21:0], k_f[4:0]}
//   0x100 - 0x3CF   weight memory mem[addr-0x100] <= cfg_wdata[7:0]  = W[cls][f] at 0x100 + cls*D + f
//   0x400 - 0x405   bias memory   mem[addr-0x400] <= cfg_wdata[31:0] = b[cls]
//   0x7F0           cfg_n_ch      <= cfg_wdata[4:0]   (reset value 8)
//   0x7F1           core_en       <= cfg_wdata[0]     (reset value 0)
//   any other address: ignored
// Write the constants only while core_en = 0.
//
// The memories are emg_tp_ram instances; their read ports are the norm_* / w_* / b_*
// ports (1-cycle latency). norm_mu = word[26:5], norm_k = word[4:0].
`default_nettype none

module emg_cfg_regs (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               cfg_we,
    input  wire [11:0]        cfg_addr,
    input  wire [31:0]        cfg_wdata,
    output reg  [4:0]         cfg_n_ch,
    output reg                core_en,
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
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_n_ch <= 5'd8;
            core_en  <= 1'b0;
        end else if (cfg_we) begin
            if (cfg_addr == 12'h7F0) cfg_n_ch <= cfg_wdata[4:0];
            if (cfg_addr == 12'h7F1) core_en  <= cfg_wdata[0];
        end
    end
endmodule

`default_nettype wire
