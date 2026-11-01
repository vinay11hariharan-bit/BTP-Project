// emg_cfg_regs — M7: trained constants and configuration registers.
//
// Step 3 (1 Nov 2026): written and read over SPI by the microcontroller (emg_spi_slave turns a
// write-setting command into one cfg_we pulse, a read-setting command into one rd_req pulse).
// The LDA weight and bias memories (0x100-0x3CF, 0x400-0x405) are gone with the on-chip LDA.
// Address map, on cfg_we:
//   0x000 - 0x077   norm memory   mem[f]          <= cfg_wdata[29:0] = {k_f[5:0], mu_f[23:0]}
//                                 mu_f unsigned 24 bits, k_f signed 6 bits (-32..31); step 2, 1 Nov 2026
//                                 (was {mu_f[21:0], k_f[4:0]} with k_f 1..22)
//   0x7F0           cfg_n_ch      <= cfg_wdata[4:0]   (reset value 8)
//   0x7F1           core_en       <= cfg_wdata[0]     (reset value 0)
//   0x7F2           cfg_dev_ch    <= cfg_wdata[3:0]   channels per ADS1299 device: 8, 6 or 4 (reset value 8)
//   0x7F3           cfg_shift     <= cfg_wdata[4:0]   24-bit code -> INT8 right shift, 0..16 (reset value 0)
//   0x7F4           ADC command   cmd_byte <= cfg_wdata[7:0], cmd_last <= cfg_wdata[8], cmd_tog flips:
//                                 emg_spi_adc sends the byte to all ADS1299 devices (core_en = 0 only,
//                                 wait >= 10 us between writes, see emg_spi_adc)
//   0x7F5           cfg_bank_len  <= cfg_wdata[6:0]   bank size B, 1..100 samples (reset value 98)
//   0x7F6           cfg_n_banks   <= cfg_wdata[2:0]   banks per window N, 1..4     (reset value 2)
//   0x7F7           cfg_warmup    <= cfg_wdata[5:0]   read-outs not sent out after reset, 0..63
//                                                     (reset value 1; use >= N - 1)
//   0x7F8           cfg_t_rest    <= cfg_wdata[20:0]  rest threshold T_REST (reset value 0 = never rest); step 4
//   0x7F9           cfg_dead_low8 <= cfg_wdata[0]     dead-check rule: 0 = all 24 bits of the code,
//                                                     1 = low 8 bits only (reset value 0); step 4
//   0x7FA           status (read only): the live flags of emg_core_ctrl
//   0x7FC           cfg_filt_on   <= cfg_wdata[0]     1 = the filter feeds Stage 1 (reset value 0 = the old
//                                                     direct path, clip(code >>> cfg_shift)); step 5.
//                                                     With the filter on, 0x7F3 is the filter's output
//                                                     shift, 0..8 (above 8 acts as 8).
//   0x7C0 - 0x7D0   filter coefficients, signed 16 bits, 14 fractional bits (cfg_wdata[15:0]); step 5.
//                   0x7C0 + 3k + {0, 1, 2} = g, a1, a2 of section k = 0..3 (high-pass 0, 1; low-pass
//                   2, 3); 0x7CC - 0x7D0 = notch b0, b1, b2, a1, a2. Reset values = the designed
//                   filter at 1 kHz (golden_model.design_filter, "target"): band-pass 20-450 Hz,
//                   notch 50 Hz Q 30. Check any other set with golden_model.filter_widths first.
//   0x7FB           rest sum (read only) of the last published frame, for setting T_REST; step 4
//   any other address: ignored
// Read (rd_req, rd_data valid one cycle later): 0x000-0x077 (only while core_en = 0: Stage 2 has
// the memory's read port while the core runs), 0x7F0-0x7F3, 0x7F5-0x7FC, 0x7C0-0x7D0. Others read 0.
// Write the constants and 0x7F0 - 0x7F3, 0x7F5 - 0x7F9, 0x7FC, 0x7C0 - 0x7D0 only while core_en = 0, before the first
// frame after reset. The reset values of 0x7F5 - 0x7F7 give the old behaviour (98 x 2, first
// decision masked); step 1 of the new design (31 Oct 2026) added them.
//
// Range check: a write to 0x7F0 whose full 32-bit value is 0 or above CMAX (24), to 0x7F5 that
// is 0 or above 100, or to 0x7F6 that is 0 or above 4, marks that setting bad. cfg_err (a chip
// pin) = 1 while any setting is bad, and a bad write turns core_en off. While cfg_err = 1,
// writing 1 to 0x7F1 is refused: core_en stays 0, so the core cannot start with a setting it
// cannot handle. Writing a valid value clears that setting's mark. The refused core_en write is
// not remembered: the host must write 1 to 0x7F1 again after fixing the setting.
//
// The norm memory is an emg_tp_ram; its read port serves Stage 2 (norm_*) and, when Stage 2 is
// not reading, the settings read (1-cycle latency). norm_mu = word[23:0], norm_k = word[29:24].
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
    output wire               cfg_err,     // 1 = C, B or N was last written out of range
    output reg  [3:0]         cfg_dev_ch,  // ADC front end (emg_spi_adc)
    output reg  [4:0]         cfg_shift,
    output reg  [6:0]         cfg_bank_len,
    output reg  [2:0]         cfg_n_banks,
    output reg  [5:0]         cfg_warmup,
    output reg  [20:0]        cfg_t_rest,
    output reg                cfg_dead_low8,
    output reg                cfg_filt_on,
    output wire [17*16-1:0]   coef_flat,   // filter coefficients, index i at [16i +: 16]
    output reg  [7:0]         cmd_byte,
    output reg                cmd_last,
    output reg                cmd_tog,
    input  wire               norm_re,     // Stage 2
    input  wire [6:0]         norm_addr,
    output wire [23:0]        norm_mu,
    output wire signed [5:0]  norm_k,
    input  wire               rd_req,      // settings read (SPI)
    input  wire [11:0]        rd_addr,
    output wire [31:0]        rd_data,     // valid the cycle after rd_req
    input  wire [7:0]         status,      // live flags (emg_core_ctrl), address 0x7FA
    input  wire [20:0]        rest_sum     // last published rest sum (emg_core_ctrl), address 0x7FB
);
    // ---------------------------------------------------------------- address decode
    wire        sel_norm = (cfg_addr[11:7] == 5'd0)    && (cfg_addr[6:0] < 7'd120);   // 0x000-0x077
    wire        rd_norm  = (rd_addr[11:7] == 5'd0)     && (rd_addr[6:0] < 7'd120);

    // ---------------------------------------------------------------- memories
    wire [29:0] norm_q;

    emg_tp_ram #(.W(30), .DEPTH(120), .AW(7)) u_norm_mem (
        .clk(clk), .we(cfg_we & sel_norm), .waddr(cfg_addr[6:0]), .wdata(cfg_wdata[29:0]),
        .re(norm_re | (rd_req & rd_norm)), .raddr(norm_re ? norm_addr : rd_addr[6:0]), .rdata(norm_q));

    // ---------------------------------------------------------------- filter coefficients
    reg  signed [15:0] coef [0:16];
    wire        sel_coef = (cfg_addr >= 12'h7C0) && (cfg_addr <= 12'h7D0);
    wire        rd_coef  = (rd_addr  >= 12'h7C0) && (rd_addr  <= 12'h7D0);
    genvar gi;
    generate
        for (gi = 0; gi < 17; gi = gi + 1) begin : g_coef
            assign coef_flat[16*gi +: 16] = coef[gi];
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            coef[ 0] <= 16'sd16384;   // section 0 g
            coef[ 1] <= -16'sd29085;   // section 0 a1
            coef[ 2] <= 16'sd12941;   // section 0 a2
            coef[ 3] <= 16'sd16384;   // section 1 g
            coef[ 4] <= -16'sd31041;   // section 1 a1
            coef[ 5] <= 16'sd14905;   // section 1 a2
            coef[ 6] <= 16'sd9172;    // section 2 g
            coef[ 7] <= 16'sd24129;   // section 2 a1
            coef[ 8] <= 16'sd9031;    // section 2 a2
            coef[ 9] <= 16'sd16384;   // section 3 g
            coef[10] <= 16'sd27903;   // section 3 a1
            coef[11] <= 16'sd12963;   // section 3 a2
            coef[12] <= 16'sd16299;   // notch b0
            coef[13] <= -16'sd31002;   // notch b1
            coef[14] <= 16'sd16299;   // notch b2
            coef[15] <= -16'sd31002;   // notch a1
            coef[16] <= 16'sd16213;   // notch a2
        end else if (cfg_we && sel_coef) begin
            coef[cfg_addr[4:0]] <= cfg_wdata[15:0];
        end
    end

    // settings read: registers are captured at rd_req, the norm memory answers one cycle later
    reg        rd_was_norm;
    reg [31:0] rd_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_was_norm <= 1'b0;
            rd_reg      <= 32'd0;
        end else if (rd_req) begin
            rd_was_norm <= rd_norm;
            case (rd_addr)
                12'h7F0: rd_reg <= {27'd0, cfg_n_ch};
                12'h7F1: rd_reg <= {31'd0, core_en};
                12'h7F2: rd_reg <= {28'd0, cfg_dev_ch};
                12'h7F3: rd_reg <= {27'd0, cfg_shift};
                12'h7F5: rd_reg <= {25'd0, cfg_bank_len};
                12'h7F6: rd_reg <= {29'd0, cfg_n_banks};
                12'h7F7: rd_reg <= {26'd0, cfg_warmup};
                12'h7F8: rd_reg <= {11'd0, cfg_t_rest};
                12'h7F9: rd_reg <= {31'd0, cfg_dead_low8};
                12'h7FA: rd_reg <= {24'd0, status};
                12'h7FB: rd_reg <= {11'd0, rest_sum};
                12'h7FC: rd_reg <= {31'd0, cfg_filt_on};
                default: rd_reg <= rd_coef ? {16'd0, coef[rd_addr[4:0]]} : 32'd0;
            endcase
        end
    end
    assign rd_data = rd_was_norm ? {2'b00, norm_q} : rd_reg;

    assign norm_mu = norm_q[23:0];
    assign norm_k  = norm_q[29:24];

    // ---------------------------------------------------------------- registers
    wire c_bad_w = (cfg_wdata == 32'd0) || (cfg_wdata > CMAX);    // C being written is unusable
    wire b_bad_w = (cfg_wdata == 32'd0) || (cfg_wdata > 32'd100);  // B being written is unusable
    wire n_bad_w = (cfg_wdata == 32'd0) || (cfg_wdata > 32'd4);    // N being written is unusable
    reg  c_bad, b_bad, n_bad;
    assign cfg_err = c_bad | b_bad | n_bad;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_n_ch   <= 5'd8;
            core_en    <= 1'b0;
            c_bad      <= 1'b0;
            b_bad      <= 1'b0;
            n_bad      <= 1'b0;
            cfg_bank_len <= 7'd98;
            cfg_n_banks  <= 3'd2;
            cfg_warmup   <= 6'd1;
            cfg_t_rest   <= 21'd0;
            cfg_dead_low8 <= 1'b0;
            cfg_filt_on  <= 1'b0;
            cfg_dev_ch <= 4'd8;
            cfg_shift  <= 5'd0;
            cmd_byte   <= 8'd0;
            cmd_last   <= 1'b0;
            cmd_tog    <= 1'b0;
        end else if (cfg_we) begin
            if (cfg_addr == 12'h7F0) begin
                cfg_n_ch <= cfg_wdata[4:0];
                c_bad    <= c_bad_w;
                if (c_bad_w) core_en <= 1'b0;            // stop the core at once
            end
            if (cfg_addr == 12'h7F5) begin
                cfg_bank_len <= cfg_wdata[6:0];
                b_bad        <= b_bad_w;
                if (b_bad_w) core_en <= 1'b0;
            end
            if (cfg_addr == 12'h7F6) begin
                cfg_n_banks <= cfg_wdata[2:0];
                n_bad       <= n_bad_w;
                if (n_bad_w) core_en <= 1'b0;
            end
            if (cfg_addr == 12'h7F7) cfg_warmup <= cfg_wdata[5:0];
            if (cfg_addr == 12'h7F8) cfg_t_rest <= cfg_wdata[20:0];
            if (cfg_addr == 12'h7F9) cfg_dead_low8 <= cfg_wdata[0];
            if (cfg_addr == 12'h7FC) cfg_filt_on   <= cfg_wdata[0];
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
