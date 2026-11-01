`timescale 1ns/1ps
// =============================================================================
// emg_spi_adc  (M8)  -  ADC front-end controller for TI ADS1299 devices, clk_spi (8 MHz)
//
// The pods are ADS1299 / ADS1299-6 / ADS1299-4 devices (8, 6 or 4 electrodes) in
// daisy-chain mode: all share START (= adc_convst), CS_N, SCLK and DIN (= adc_mosi),
// DOUT of the first device drives adc_miso, and DOUT of device k+1 goes to DAISY_IN of
// device k. The devices are set up once (single-shot mode, RDATAC) through the command
// port before core_en is set; see "Commands" below and emg_cfg_regs.
//
// Sampling: every SAMPLE_DIV clk_spi cycles (8000 = 1 ms = 1 kHz) adc_convst goes
// high for CONVST_W cycles. In single-shot mode this starts one conversion in every
// device. When adc_drdy_n falls, this block reads the frame over SPI at clk_spi/2 (4 MHz).
//
// Frame format (ADS1299 datasheet, RDATAC): each device sends a 24-bit status word
// {1100, LOFF_STATP[7:0], LOFF_STATN[7:0], GPIO[7:4]} and then 24 bits per channel,
// two's complement, MSB first. Device 0 (on MISO) comes first. The read stops after
// channel C-1, so a device that is only partly used is not read to its end.
//   bits per frame = 24 * (C + ceil(C / cfg_dev_ch))        (C = 24, 8 per device: 648 bits, 162 us)
//
// What this block keeps from each frame, published together when the read ends:
//   rx_flat[8c +: 8]  channel c as INT8:  clip( code >>> cfg_shift , -128, 127 )
//                     (arithmetic shift = floor division by 2^cfg_shift). The training code
//                     must convert recordings the same way (lda_golden_model.ads_code_to_int8).
//   lo_flat[c]        lead-off of channel c: LOFF_STATP bit | LOFF_STATN bit of its device
//                     (bit i of each = input i+1 of that device).
//   rx16_flat[16c +: 16]  channel c's top 16 bits (code >>> 8), the filter's input (step 5).
//   chg_flat[c]       channel c's 24-bit code differs from its previous code (step 4, dead check).
//                     With cfg_dead_low8 = 1 only the low 8 bits are compared. The first code of
//                     a channel after reset has nothing to compare with and gives 0. The previous
//                     code is kept per channel (24 x 24 bits) and updated as each word arrives, so
//                     a read cut short (already a link error) leaves the early channels updated.
//   hdr_err           some status word in this frame did not start with 1100: a device is
//                     missing, the chain is broken, or cfg_dev_ch does not match the devices.
//   frame_tog         toggles once per published frame.
// miss_tog toggles when a CONVST comes while the previous read has not finished (no DRDY_N
// fall in the whole sample period, or a read too slow): no frame was published. A read that is
// cut short this way, or by core_en going low, stops SCLK at once but keeps CS_N low for
// CS_HOLD cycles before raising it, so the ADS1299 timing rules hold even in a fault.
//
// SPI mode 1 (CPOL = 0, CPHA = 1), as the ADS1299 requires: SCLK idles low, the device
// changes DOUT after each SCLK rising edge (tDOPD <= 17 ns at DVDD 3.3 V, 32 ns at 1.8 V),
// and this block samples adc_miso at the next SCLK falling edge, 125 ns after the rise.
// It drives adc_mosi at SCLK rising edges, so DIN is stable 125 ns before and after the
// falling edge where the device latches it (needs tDIST = tDIHD = 10 ns).
//
// Datasheet timing handled here (tCLK = 488 ns at the 2.048 MHz ADS1299 clock):
//   START high >= 2 tCLK (977 ns)        CONVST_W = 16 cycles = 2 us
//   last SCLK fall -> CS_N high >= 4 tCLK  CS_HOLD  = 24 cycles = 3 us
//   CS_N high time >= 2 tCLK             CMD_GAP  = 24 cycles = 3 us (after a command)
//   CS_N low -> first SCLK >= 6 ns       CS_SETUP = 2 cycles = 250 ns
//
// Commands (core_en = 0 only): each write to the command register (emg_cfg_regs, 0x7F4)
// sends one byte on adc_mosi, MSB first. cmd_last = 1 raises CS_N after the byte (end of
// a command); cmd_last = 0 keeps CS_N low for the next byte (multi-byte WREG must stay in
// one CS_N-low period). The host must wait at least 10 us between command writes: a byte
// takes 2 us, CS_HOLD + CMD_GAP 6 us, plus up to 3 clk_spi cycles to cross into this clock
// domain. A write that arrives earlier is lost. Bytes written while core_en = 1 are ignored.
//
// Other choices (unchanged from the previous version):
//  * core_en and the command toggle come from clk_sys: each passes a 2-FF synchroniser.
//    cfg_n_ch, cfg_dev_ch and cfg_shift only change while core_en = 0, so they are stable
//    whenever they are used and need no synchroniser. The command byte is stable from its
//    toggle until the next write (host waits 10 us).
//  * cfg_n_ch above CMAX is treated as CMAX. cfg_n_ch = 0 reads nothing.
//    cfg_dev_ch = 0 or above 8 is treated as 8. cfg_shift above 16 is treated as 16.
//  * adc_drdy_n is asynchronous: 2-FF synchroniser, then its falling edge is detected.
//  * adc_miso is NOT synchronised. It is sampled at a known point of the SCLK this block
//    generates (constrain it with set_input_delay in STA).
// =============================================================================
module emg_spi_adc #(
    parameter integer CMAX       = 24,   // max channels -> rx_flat width 8*CMAX
    parameter integer SAMPLE_DIV = 8000, // clk_spi cycles between CONVST pulses (1 kHz)
    parameter integer CONVST_W   = 16,   // CONVST (= START) high time, clk_spi cycles
    parameter integer CS_SETUP   = 2,    // clk_spi cycles from CS_N low to the first SCLK rise (1..15)
    parameter integer CS_HOLD    = 24,   // clk_spi cycles from the last SCLK fall to CS_N high (1..63)
    parameter integer CMD_GAP    = 24    // clk_spi cycles CS_N stays high after a command (1..63)
) (
    input  wire                clk_spi,    // 8 MHz
    input  wire                rst_n,      // already synchronised to clk_spi
    input  wire [4:0]          cfg_n_ch,   // C, quasi-static
    input  wire [3:0]          cfg_dev_ch, // channels per ADC device (8, 6 or 4), quasi-static
    input  wire [4:0]          cfg_shift,  // 24-bit code -> INT8 right shift, quasi-static
    input  wire                cfg_dead_low8, // dead check on the low 8 bits only, quasi-static
    input  wire                core_en,    // from clk_sys domain

    input  wire [7:0]          cmd_byte,   // from clk_sys, stable after cmd_tog changes
    input  wire                cmd_last,
    input  wire                cmd_tog,    // toggles once per command-byte write

    output reg                 adc_convst, // chip pin -> START of all devices
    input  wire                adc_drdy_n, // chip pin, asynchronous
    output reg                 adc_cs_n,   // chip pin
    output reg                 adc_sclk,   // chip pin, 4 MHz during a transfer
    input  wire                adc_miso,   // chip pin, DOUT of device 0
    output reg                 adc_mosi,   // chip pin -> DIN of all devices

    output reg  [8*CMAX-1:0]   rx_flat,    // channel c at [8c +: 8]
    output reg  [16*CMAX-1:0]  rx16_flat,  // channel c's top 16 bits at [16c +: 16] (filter input)
    output reg  [CMAX-1:0]     lo_flat,    // lead-off flag of channel c
    output reg  [CMAX-1:0]     chg_flat,   // code of channel c changed since its previous sample
    output reg                 hdr_err,    // bad status header in this frame
    output reg                 frame_tog,  // toggles when rx_flat, lo_flat, chg_flat, hdr_err update
    output reg                 miss_tog    // toggles when a sample period produced no frame
);

    localparam integer TW = $clog2(SAMPLE_DIV);
    // Sized copies of the parameters, so every compare is width-matched.
    localparam [4:0]    CMAX5     = CMAX[4:0];
    localparam [TW-1:0] T_LAST    = SAMPLE_DIV[TW-1:0] - 1'b1;
    localparam [TW-1:0] T_CONVST  = CONVST_W[TW-1:0];
    localparam [5:0]    SETUP_END = CS_SETUP[5:0] - 1'b1;
    localparam [5:0]    HOLD_END  = CS_HOLD[5:0]  - 1'b1;
    localparam [5:0]    GAP_END   = CMD_GAP[5:0]  - 1'b1;

    // ---------------------------------------------------------------- inputs
    // core_en and the command toggle: 2-FF synchronisers (they cross from clk_sys).
    // adc_drdy_n: 2-FF synchroniser + one more stage to find the falling edge.
    reg en_m, en_s;
    reg ct_m, ct_s, ct_d;
    reg drdy_m, drdy_s, drdy_d;
    wire drdy_fall = drdy_d & ~drdy_s;
    wire cmd_go    = ct_s ^ ct_d;

    always @(posedge clk_spi or negedge rst_n) begin
        if (!rst_n) begin
            en_m   <= 1'b0;  en_s   <= 1'b0;
            ct_m   <= 1'b0;  ct_s   <= 1'b0;  ct_d   <= 1'b0;
            drdy_m <= 1'b1;  drdy_s <= 1'b1;  drdy_d <= 1'b1;
        end else begin
            en_m   <= core_en;     en_s   <= en_m;
            ct_m   <= cmd_tog;     ct_s   <= ct_m;    ct_d   <= ct_s;
            drdy_m <= adc_drdy_n;  drdy_s <= drdy_m;  drdy_d <= drdy_s;
        end
    end

    // Effective configuration (out-of-range values clamped).
    wire [4:0] c_eff   = (cfg_n_ch > CMAX5) ? CMAX5 : cfg_n_ch;
    wire [3:0] dev_eff = (cfg_dev_ch == 4'd0 || cfg_dev_ch > 4'd8) ? 4'd8 : cfg_dev_ch;
    wire [4:0] sh_eff  = (cfg_shift > 5'd16) ? 5'd16 : cfg_shift;

    // ------------------------------------------------------- CONVST timer
    // timer runs 0 .. SAMPLE_DIV-1 while enabled. tick marks the start of a
    // sample period. The first tick comes right after enable.
    reg  [TW-1:0] timer;
    wire          tick = en_s && (timer == {TW{1'b0}});

    always @(posedge clk_spi or negedge rst_n) begin
        if (!rst_n) begin
            timer      <= {TW{1'b0}};
            adc_convst <= 1'b0;
        end else begin
            if (!en_s || timer == T_LAST) timer <= {TW{1'b0}};
            else                              timer <= timer + 1'b1;
            adc_convst <= en_s && (timer < T_CONVST);
        end
    end

    // ------------------------------------------------- 24-bit word -> INT8
    reg  [22:0]        sr23;                       // first 23 bits of the current word
    wire signed [23:0] word    = {sr23, adc_miso};  // complete when bit_cnt == 23
    reg  [4:0]         sh;                          // cfg_shift latched at the start of a read
    wire signed [23:0] shifted = word >>> sh;
    wire        [7:0]  word_q8 = (shifted >  24'sd127) ? 8'h7F :
                                 (shifted < -24'sd128) ? 8'h80 : shifted[7:0];

    // ------------------------------------------------------ state machine
    localparam [3:0] S_IDLE   = 4'd0,  // nothing to do
                     S_WAIT   = 4'd1,  // CONVST sent, wait for DRDY_N fall
                     S_SETUP  = 4'd2,  // read: CS_N low, wait before the first SCLK
                     S_SHIFT  = 4'd3,  // read: clock the frame in
                     S_HOLD   = 4'd4,  // read: CS_N still low after the last SCLK fall
                     S_CSETUP = 4'd5,  // command: CS_N low, wait before the first SCLK
                     S_CSHIFT = 4'd6,  // command: clock one byte out
                     S_CWAIT  = 4'd7,  // command: CS_N low, waiting for the next byte
                     S_CHOLD  = 4'd8,  // command: CS_N low after the last byte
                     S_CGAP   = 4'd9,  // command: CS_N high, minimum gap
                     S_ABORT  = 4'd10; // transfer cut short: SCLK stopped, CS_N still low for CS_HOLD

    reg [3:0]          state;
    reg [5:0]          cnt;        // setup / hold / gap counter
    reg [4:0]          n_ch;       // C latched at the start of the read
    reg [3:0]          dev_ch;     // channels per device, latched
    reg                in_status;  // the current word is a status word
    reg [4:0]          ch_glob;    // channel the current data word belongs to
    reg [3:0]          ch_in_dev;  // its index inside its device
    reg [4:0]          bit_cnt;    // bit within the 24-bit word, 0 = MSB
    reg [7:0]          cmd_sr;     // command byte being sent
    reg                cmd_end;    // raise CS_N after this byte
    reg [2:0]          cmd_bit;    // bits of the command byte already sent
    reg [8*CMAX-1:0]   shadow;     // frame being assembled
    reg [16*CMAX-1:0]  shadow16;
    reg [CMAX-1:0]     lo_shadow;
    reg [CMAX-1:0]     chg_shadow;
    reg [24*CMAX-1:0]  prev_code;  // previous 24-bit code of each channel (dead check)
    reg [CMAX-1:0]     have_prev;  // channel c has had a code since reset
    reg                low8;       // cfg_dead_low8 latched at the start of a read
    wire [23:0]        code_diff = word ^ prev_code[24*ch_glob +: 24];
    wire               code_chg  = have_prev[ch_glob] & (low8 ? (|code_diff[7:0]) : (|code_diff));
    reg                hdr_shadow;
    reg  [3:0]         abort_next; // state to enter after S_ABORT

    wire data_state = (state == S_WAIT) || (state == S_SETUP) ||
                      (state == S_SHIFT) || (state == S_HOLD);
    wire cmd_state  = (state >= S_CSETUP) && (state <= S_CGAP);

    integer i;

    always @(posedge clk_spi or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            adc_cs_n   <= 1'b1;
            adc_sclk   <= 1'b0;
            adc_mosi   <= 1'b0;
            cnt        <= 6'd0;
            n_ch       <= 5'd0;
            dev_ch     <= 4'd8;
            sh         <= 5'd0;
            in_status  <= 1'b1;
            ch_glob    <= 5'd0;
            ch_in_dev  <= 4'd0;
            bit_cnt    <= 5'd0;
            sr23       <= 23'd0;
            cmd_sr     <= 8'd0;
            cmd_end    <= 1'b0;
            cmd_bit    <= 3'd0;
            shadow     <= {8*CMAX{1'b0}};
            shadow16   <= {16*CMAX{1'b0}};
            lo_shadow  <= {CMAX{1'b0}};
            chg_shadow <= {CMAX{1'b0}};
            prev_code  <= {24*CMAX{1'b0}};
            have_prev  <= {CMAX{1'b0}};
            low8       <= 1'b0;
            hdr_shadow <= 1'b0;
            rx_flat    <= {8*CMAX{1'b0}};
            rx16_flat  <= {16*CMAX{1'b0}};
            lo_flat    <= {CMAX{1'b0}};
            chg_flat   <= {CMAX{1'b0}};
            hdr_err    <= 1'b0;
            frame_tog  <= 1'b0;
            miss_tog   <= 1'b0;
            abort_next <= S_IDLE;
        end else if ((en_s && cmd_state) || (!en_s && data_state)) begin
            // Mode changed: stop at once. A partial frame or command is dropped. If CS_N is
            // low, it stays low for CS_HOLD cycles first, as the ADS1299 needs (S_ABORT).
            adc_sclk <= 1'b0;
            adc_mosi <= 1'b0;
            cnt      <= 6'd0;
            abort_next <= S_IDLE;
            state    <= adc_cs_n ? S_IDLE : S_ABORT;
        end else if (tick) begin
            // New sample period. A read still pending means the last period gave no frame.
            // A read cut short keeps CS_N low for CS_HOLD cycles before it is released.
            if (state != S_IDLE) miss_tog <= ~miss_tog;
            adc_sclk <= 1'b0;
            adc_mosi <= 1'b0;
            cnt      <= 6'd0;
            abort_next <= S_WAIT;
            state    <= adc_cs_n ? S_WAIT : S_ABORT;
        end else begin
            case (state)
            S_IDLE:
                if (!en_s && cmd_go) begin                 // start a command
                    cmd_sr   <= cmd_byte;
                    cmd_end  <= cmd_last;
                    cmd_bit  <= 3'd0;
                    cnt      <= 6'd0;
                    adc_cs_n <= 1'b0;
                    state    <= S_CSETUP;
                end

            // ---------------------------------------------------- data read
            S_WAIT:
                if (drdy_fall) begin
                    if (c_eff == 5'd0) begin
                        state <= S_IDLE;                   // nothing to read
                    end else begin
                        n_ch       <= c_eff;
                        dev_ch     <= dev_eff;
                        sh         <= sh_eff;
                        low8       <= cfg_dead_low8;
                        in_status  <= 1'b1;                // every device starts with status
                        ch_glob    <= 5'd0;
                        ch_in_dev  <= 4'd0;
                        bit_cnt    <= 5'd0;
                        cnt        <= 6'd0;
                        shadow     <= {8*CMAX{1'b0}};      // unused bytes stay 0
                        shadow16   <= {16*CMAX{1'b0}};
                        lo_shadow  <= {CMAX{1'b0}};
                        chg_shadow <= {CMAX{1'b0}};
                        hdr_shadow <= 1'b0;
                        adc_cs_n   <= 1'b0;
                        state      <= S_SETUP;
                    end
                end

            S_SETUP:
                if (cnt == SETUP_END) state <= S_SHIFT;
                else                  cnt   <= cnt + 1'b1;

            S_SHIFT:
                if (!adc_sclk) begin
                    // SCLK rising edge: the devices put out their next bit.
                    adc_sclk <= 1'b1;
                end else begin
                    // SCLK falling edge: sample the bit put out 125 ns ago.
                    adc_sclk <= 1'b0;
                    if (bit_cnt != 5'd23) begin
                        sr23    <= {sr23[21:0], adc_miso};
                        bit_cnt <= bit_cnt + 1'b1;
                    end else begin
                        bit_cnt <= 5'd0;                   // a 24-bit word is complete
                        if (in_status) begin
                            if (word[23:20] != 4'b1100) hdr_shadow <= 1'b1;
                            for (i = 0; i < 8; i = i + 1)
                                if (i < dev_ch && ch_glob + i < n_ch)
                                    lo_shadow[ch_glob + i] <= word[12 + i] | word[4 + i];
                            in_status <= 1'b0;
                            ch_in_dev <= 4'd0;
                        end else begin
                            shadow[8*ch_glob +: 8] <= word_q8;
                            shadow16[16*ch_glob +: 16] <= word[23:8];
                            chg_shadow[ch_glob]    <= code_chg;
                            prev_code[24*ch_glob +: 24] <= word;
                            have_prev[ch_glob]     <= 1'b1;
                            if (ch_glob == n_ch - 1'b1) begin
                                cnt   <= 6'd0;             // last channel: stop clocking
                                state <= S_HOLD;
                            end else begin
                                ch_glob <= ch_glob + 1'b1;
                                if (ch_in_dev == dev_ch - 1'b1) begin
                                    in_status <= 1'b1;     // next device starts
                                    ch_in_dev <= 4'd0;
                                end else begin
                                    ch_in_dev <= ch_in_dev + 1'b1;
                                end
                            end
                        end
                    end
                end

            S_HOLD:
                if (cnt == HOLD_END) begin
                    adc_cs_n  <= 1'b1;
                    rx_flat   <= shadow;                   // publish the whole frame
                    rx16_flat <= shadow16;
                    lo_flat   <= lo_shadow;
                    chg_flat  <= chg_shadow;
                    hdr_err   <= hdr_shadow;
                    frame_tog <= ~frame_tog;
                    state     <= S_IDLE;
                end else begin
                    cnt <= cnt + 1'b1;
                end

            // ---------------------------------------------------- command
            S_CSETUP:
                if (cnt == SETUP_END) state <= S_CSHIFT;
                else                  cnt   <= cnt + 1'b1;

            S_CSHIFT:
                if (!adc_sclk) begin
                    // SCLK rising edge: put out the next bit, MSB first.
                    adc_sclk <= 1'b1;
                    adc_mosi <= cmd_sr[7];
                    cmd_sr   <= {cmd_sr[6:0], 1'b0};
                end else begin
                    // SCLK falling edge: the devices latch DIN here.
                    adc_sclk <= 1'b0;
                    cmd_bit  <= cmd_bit + 1'b1;
                    if (cmd_bit == 3'd7) begin
                        cnt   <= 6'd0;
                        state <= cmd_end ? S_CHOLD : S_CWAIT;
                    end
                end

            S_CWAIT: begin
                adc_mosi <= 1'b0;
                if (cmd_go) begin                          // next byte of the same command
                    cmd_sr  <= cmd_byte;
                    cmd_end <= cmd_last;
                    cmd_bit <= 3'd0;
                    cnt     <= 6'd0;
                    state   <= S_CSETUP;
                end
            end

            S_CHOLD: begin
                adc_mosi <= 1'b0;
                if (cnt == HOLD_END) begin
                    adc_cs_n <= 1'b1;
                    cnt      <= 6'd0;
                    state    <= S_CGAP;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end

            S_CGAP:
                if (cnt == GAP_END) state <= S_IDLE;
                else                cnt   <= cnt + 1'b1;

            // ---------------------------------------------------- cut short
            S_ABORT:
                if (cnt == HOLD_END) begin
                    adc_cs_n <= 1'b1;
                    state    <= abort_next;
                end else begin
                    cnt <= cnt + 1'b1;
                end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
