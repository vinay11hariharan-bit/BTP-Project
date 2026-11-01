// emg_spi_slave — the chip's link to the microcontroller (step 3, 1 Nov 2026).
//
// The microcontroller is the SPI master, the chip the slave. SPI mode 0 (SCLK idle low, data
// sampled on the rising edge), most significant bit first. Pins: mcu_cs_n, mcu_sclk, mcu_mosi,
// mcu_miso and mcu_drdy (data ready).
//
// How the chip follows the link: everything runs on clk_sys (1 MHz). CS, SCLK and MOSI pass
// through 2-flip-flop synchronisers, and a rising SCLK edge is found by comparing the
// synchronised SCLK with its value one cycle earlier. The three inputs go through identical
// chains, so the chip sees them in the same order as they happened.
//   * At a rising SCLK edge the chip takes the MOSI bit and then moves MISO to the next bit
//     (2 to 3 us after the edge). The master samples MISO at the NEXT rising edge, so this works
//     when every SCLK high and low phase is at least 2 us: SCLK at most 250 kHz.
//   * Rules for the master: SCLK low when CS falls; first rising SCLK edge at least 4 us after
//     CS falls; CS rises at least 2 us after the last falling SCLK edge; CS stays high at least
//     4 us between commands.
//
// Every command is one CS-low period that starts with one command byte. While the master sends,
// the chip sends 0s.
//   0x01 read frame     -> chip sends the frame (below) from the second byte on
//   0x02 write setting  -> master sends 2 address bytes (12 bits used) + 4 value bytes, MSB
//                          first; the write happens when the last bit arrives (cfg_we pulse)
//   0x03 read setting   -> master sends 2 address bytes, then 1 spare byte (anything), then
//                          reads 4 value bytes. The spare byte gives the chip time to fetch a
//                          value stored in memory (the norm memory has a 1-cycle read).
//   other               -> ignored, chip sends 0s
// A command cut short by CS rising does nothing.
//
// Frame (5C + 6 bytes, C = cfg_n_ch):
//   0         frame counter (read-out number, mod 256)
//   1         flags (see emg_core_ctrl)
//   2, 3, 4   dead-channel mask, channel 0 = byte 2 bit 0
//   5 .. 5C+4 the 5C features, signed 8-bit, from the x_q RAM (f = 5*ch + type)
//   5C+5      checksum = sum of bytes 0 .. 5C+4, kept to 8 bits. It is sent INVERTED if the frame
//             changed while it was being read (a new read-out was published, or Stage 2 wrote a
//             feature, while CS was low), so a mixed frame always fails the check.
//   after it  0s
//
// mcu_drdy: set when a new frame is published (frame_pub), cleared when CS falls.
// cs_low: synchronised CS = low, for the skip rule in emg_core_ctrl.
`default_nettype none

module emg_spi_slave (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        mcu_cs_n,       // asynchronous pins
    input  wire        mcu_sclk,
    input  wire        mcu_mosi,
    output reg         mcu_miso,
    output reg         mcu_drdy,
    output wire        cs_low,         // synchronised: a command is in progress

    input  wire [4:0]  cfg_n_ch,       // C, for the frame length
    input  wire        frame_pub,      // 1-cycle pulse: a new frame is ready
    input  wire        xq_written,     // Stage 2 wrote a feature this cycle
    input  wire [7:0]  f_count,        // frame header, stable between frame_pub pulses
    input  wire [7:0]  f_flags,
    input  wire [23:0] f_dead,

    output reg         xq_re,          // x_q RAM read port (1-cycle latency, output held)
    output reg  [6:0]  xq_raddr,
    input  wire [7:0]  xq_q,

    output reg         cfg_we,         // write port into emg_cfg_regs
    output reg  [11:0] cfg_addr,
    output reg  [31:0] cfg_wdata,
    output reg         rd_req,         // read port of emg_cfg_regs: rd_data valid 1 cycle later
    output reg  [11:0] rd_addr,
    input  wire [31:0] rd_data
);
    // ---------------------------------------------------------------- synchronisers
    reg [1:0] cs_s, sclk_s, mosi_s;
    reg       sclk_d, cs_d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cs_s <= 2'b11; sclk_s <= 2'b00; mosi_s <= 2'b00; sclk_d <= 1'b0; cs_d <= 1'b1;
        end else begin
            cs_s   <= {cs_s[0],   mcu_cs_n};
            sclk_s <= {sclk_s[0], mcu_sclk};
            mosi_s <= {mosi_s[0], mcu_mosi};
            sclk_d <= sclk_s[1];
            cs_d   <= cs_s[1];
        end
    end
    wire cs_n    = cs_s[1];
    wire cs_fall = cs_d & ~cs_n;
    wire rise    = ~cs_n & sclk_s[1] & ~sclk_d;
    assign cs_low = ~cs_n;

    // ---------------------------------------------------------------- byte engine
    reg  [2:0]  bit_cnt;        // bit inside the current byte, 0..7
    reg  [3:0]  byte_cnt;       // bytes completed in this command, saturating at 15
    reg  [6:0]  in_sh;          // bits 7..1 of the byte coming in
    reg  [7:0]  out_sh;         // byte going out, MSB on MISO
    reg  [7:0]  cmd;
    reg  [7:0]  fidx;           // frame byte that is loaded at the next byte boundary
    reg  [7:0]  cks;            // running checksum
    reg         dirty;          // the frame changed during this read
    reg  [31:0] val;            // read value
    reg         rd_req_d;       // rd_data is valid while rd_req_d = 1

    wire [7:0]  in_byte  = {in_sh, mosi_s[1]};                 // the byte that completes now
    wire        byte_end = rise & (bit_cnt == 3'd7);
    wire [7:0]  cmd_now  = (byte_cnt == 4'd0) ? in_byte : cmd;
    wire [7:0]  n_feat   = {cfg_n_ch, 2'b00} + {3'd0, cfg_n_ch};  // 5C
    wire [7:0]  cks_idx  = n_feat + 8'd5;                       // index of the checksum byte

    // frame byte fidx (header and checksum from registers, features from the x_q RAM output)
    reg  [7:0]  fbyte;
    always @(*) begin
        if      (fidx == 8'd0)    fbyte = f_count;
        else if (fidx == 8'd1)    fbyte = f_flags;
        else if (fidx == 8'd2)    fbyte = f_dead[7:0];
        else if (fidx == 8'd3)    fbyte = f_dead[15:8];
        else if (fidx == 8'd4)    fbyte = f_dead[23:16];
        else if (fidx <  cks_idx) fbyte = xq_q;
        else if (fidx == cks_idx) fbyte = dirty ? ~cks : cks;
        else                      fbyte = 8'd0;
    end

    // byte to send after the byte that completes now
    reg  [7:0]  next_out;
    always @(*) begin
        next_out = 8'd0;
        if (cmd_now == 8'h01)
            next_out = fbyte;
        else if (cmd_now == 8'h03 && byte_cnt >= 4'd3 && byte_cnt <= 4'd6)
            next_out = val[31 - 8 * (byte_cnt - 4'd3) -: 8];        // value bytes after the spare byte
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_cnt <= 3'd0; byte_cnt <= 4'd0; in_sh <= 7'd0; out_sh <= 8'd0; cmd <= 8'd0;
            fidx <= 8'd0; cks <= 8'd0; dirty <= 1'b0; val <= 32'd0; rd_req_d <= 1'b0;
            mcu_miso <= 1'b0; mcu_drdy <= 1'b0;
            xq_re <= 1'b0; xq_raddr <= 7'd0;
            cfg_we <= 1'b0; cfg_addr <= 12'd0; cfg_wdata <= 32'd0;
            rd_req <= 1'b0; rd_addr <= 12'd0;
        end else begin
            xq_re  <= 1'b0;
            cfg_we <= 1'b0;
            rd_req <= 1'b0;
            rd_req_d <= rd_req;
            if (rd_req_d) val <= rd_data;                      // emg_cfg_regs answers 1 cycle after rd_req

            if (frame_pub) mcu_drdy <= 1'b1;
            if (~cs_n && (frame_pub || xq_written)) dirty <= 1'b1;

            if (cs_n) begin                                    // idle
                mcu_miso <= 1'b0;
            end
            if (cs_fall) begin                                 // a command starts
                bit_cnt <= 3'd0; byte_cnt <= 4'd0; out_sh <= 8'd0; cmd <= 8'd0;
                fidx <= 8'd0; cks <= 8'd0; dirty <= 1'b0;
                mcu_miso <= 1'b0;
                mcu_drdy <= 1'b0;
            end else if (rise) begin
                in_sh   <= {in_sh[5:0], mosi_s[1]};
                bit_cnt <= bit_cnt + 3'd1;
                if (!byte_end) begin
                    out_sh   <= {out_sh[6:0], 1'b0};
                    mcu_miso <= out_sh[6];
                end else begin
                    // a byte is complete: act on it, then load the next outgoing byte
                    if (byte_cnt != 4'd15) byte_cnt <= byte_cnt + 4'd1;
                    if (byte_cnt == 4'd0) cmd <= in_byte;
                    out_sh   <= next_out;
                    mcu_miso <= next_out[7];
                    if (cmd_now == 8'h01) begin                // frame: count the byte, fetch the next
                        if (fidx < cks_idx) cks <= cks + fbyte;
                        if (fidx != 8'hFF) fidx <= fidx + 8'd1;
                        if (fidx + 8'd1 >= 8'd5 && fidx + 8'd1 < cks_idx) begin
                            xq_re    <= 1'b1;
                            xq_raddr <= fidx[6:0] + 7'd1 - 7'd5;
                        end
                    end
                    if (cmd_now == 8'h02 || cmd_now == 8'h03) begin
                        if (byte_cnt == 4'd1) cfg_addr <= {in_byte[3:0], 8'd0};
                        if (byte_cnt == 4'd2) begin
                            cfg_addr[7:0] <= in_byte;
                            if (cmd_now == 8'h03) begin
                                rd_req  <= 1'b1;
                                rd_addr <= {cfg_addr[11:8], in_byte};
                            end
                        end
                        if (cmd_now == 8'h02 && byte_cnt >= 4'd3 && byte_cnt <= 4'd6) begin
                            cfg_wdata <= {cfg_wdata[23:0], in_byte};
                            if (byte_cnt == 4'd6) cfg_we <= 1'b1;   // last value byte: write
                        end
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
