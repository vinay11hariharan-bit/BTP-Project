`timescale 1ns/1ps
// =============================================================================
// emg_spi_adc  (M8)  -  ADC front-end controller, clk_spi domain (8 MHz)
//
// Every SAMPLE_DIV clk_spi cycles (8120 cycles = 1.015 ms = 985 Hz) this block
// pulses adc_convst to all ADC pods. When adc_drdy_n falls, it reads 8*C bits
// out of the pod daisy chain over SPI at clk_spi/2 (4 MHz) into a shadow
// register. It then copies the complete frame to rx_flat and toggles
// frame_tog in the same clock edge.
//
// Decisions on top of spec card M8:
//  1. Channel order = arrival order. The first byte out of the chain is
//     channel 0 and lands in rx_flat[7:0], the next byte is channel 1, and
//     so on. Each byte arrives MSB first.
//  2. SPI mode 0. SCLK idles low. The ADC changes adc_miso after each SCLK
//     falling edge, and this block samples adc_miso at each SCLK rising edge.
//     That gives the ADC half an SCLK period (125 ns) to drive the new bit.
//  3. No adc_drdy_n fall after a CONVST -> no frame. rx_flat keeps the old
//     frame and frame_tog does not toggle. Every CONVST restarts the read
//     sequence, so a missing or late DRDY_N can never stall the block. A
//     read still in progress when the next CONVST fires is dropped.
//  4. rx_flat bits above 8*C are 0.
//
// Other choices:
//  * core_en comes from the clk_sys domain, so it passes through a 2-FF
//    synchroniser. cfg_n_ch only changes while core_en = 0 (spec), so it is
//    stable whenever the synchronised enable is 1 and needs no synchroniser.
//  * cfg_n_ch above CMAX is treated as CMAX. cfg_n_ch = 0 reads nothing.
//  * adc_drdy_n is an off-chip signal with no timing relation to clk_spi, so
//    it passes through a 2-FF synchroniser before its falling edge is found.
//  * adc_miso is NOT synchronised. It is sampled at a known point of the SCLK
//    this block generates (constrain it with set_input_delay in STA).
// =============================================================================
module emg_spi_adc #(
    parameter integer CMAX     = 24,    // max channels -> rx_flat width 8*CMAX
    parameter integer SAMPLE_DIV = 8120, // clk_spi cycles between CONVST pulses
    parameter integer CONVST_W = 2,     // CONVST high time, clk_spi cycles
    parameter integer CS_SETUP = 2      // extra clk_spi cycles from CS_N low
                                        // to the first SCLK rise (1..15)
) (
    input  wire                clk_spi,    // 8 MHz
    input  wire                rst_n,      // already synchronised to clk_spi
    input  wire [4:0]          cfg_n_ch,   // C, quasi-static
    input  wire                core_en,    // from clk_sys domain

    output reg                 adc_convst, // chip pin, shared by all pods
    input  wire                adc_drdy_n, // chip pin, asynchronous
    output reg                 adc_cs_n,   // chip pin
    output reg                 adc_sclk,   // chip pin, 4 MHz during a burst
    input  wire                adc_miso,   // chip pin, SDO of the last pod

    output reg  [8*CMAX-1:0]   rx_flat,    // channel c at [8c +: 8]
    output reg                 frame_tog   // toggles when rx_flat updates
);

    localparam integer TW = $clog2(SAMPLE_DIV);
    // Sized copies of the parameters, so every compare is width-matched.
    localparam [4:0]    CMAX5     = CMAX[4:0];
    localparam [TW-1:0] T_LAST    = SAMPLE_DIV[TW-1:0] - 1'b1;
    localparam [TW-1:0] T_CONVST  = CONVST_W[TW-1:0];
    localparam [3:0]    SETUP_END = CS_SETUP[3:0] - 1'b1;

    // ---------------------------------------------------------------- inputs
    // core_en: 2-FF synchroniser (crosses from clk_sys).
    reg en_m, en_s;
    // adc_drdy_n: 2-FF synchroniser + one more stage to find the falling edge.
    reg drdy_m, drdy_s, drdy_d;
    wire drdy_fall = drdy_d & ~drdy_s;

    always @(posedge clk_spi or negedge rst_n) begin
        if (!rst_n) begin
            en_m   <= 1'b0;  en_s   <= 1'b0;
            drdy_m <= 1'b1;  drdy_s <= 1'b1;  drdy_d <= 1'b1;
        end else begin
            en_m   <= core_en;     en_s   <= en_m;
            drdy_m <= adc_drdy_n;  drdy_s <= drdy_m;  drdy_d <= drdy_s;
        end
    end

    // Effective channel count: values above CMAX are clamped to CMAX.
    wire [4:0] c_eff = (cfg_n_ch > CMAX5) ? CMAX5 : cfg_n_ch;

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

    // ------------------------------------------------------ read sequence
    localparam [1:0] S_IDLE  = 2'd0,   // wait for the next tick
                     S_WAIT  = 2'd1,   // CONVST sent, wait for DRDY_N fall
                     S_SETUP = 2'd2,   // CS_N low, first bit settling
                     S_SHIFT = 2'd3;   // clock the bits out

    reg [1:0]          state;
    reg [3:0]          setup_cnt;
    reg [4:0]          n_bytes;   // C latched at the start of the burst
    reg [4:0]          byte_idx;  // channel being received
    reg [2:0]          bit_idx;   // bit within the byte, 0 = MSB
    reg [6:0]          byte_sr;   // first 7 bits of the current byte
    reg [8*CMAX-1:0]   shadow;    // frame being assembled

    always @(posedge clk_spi or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            adc_cs_n  <= 1'b1;
            adc_sclk  <= 1'b0;
            setup_cnt <= 4'd0;
            n_bytes   <= 5'd0;
            byte_idx  <= 5'd0;
            bit_idx   <= 3'd0;
            byte_sr   <= 7'd0;
            shadow    <= {8*CMAX{1'b0}};
            rx_flat   <= {8*CMAX{1'b0}};
            frame_tog <= 1'b0;
        end else if (!en_s) begin
            // Disabled: release the bus at once. A partial frame is dropped.
            state    <= S_IDLE;
            adc_cs_n <= 1'b1;
            adc_sclk <= 1'b0;
        end else if (tick) begin
            // New sample period: whatever was pending is abandoned.
            state    <= S_WAIT;
            adc_cs_n <= 1'b1;
            adc_sclk <= 1'b0;
        end else begin
            case (state)
            S_IDLE: ;

            S_WAIT:
                if (drdy_fall) begin
                    if (c_eff == 5'd0) begin
                        state <= S_IDLE;            // nothing to read
                    end else begin
                        n_bytes   <= c_eff;
                        byte_idx  <= 5'd0;
                        bit_idx   <= 3'd0;
                        setup_cnt <= 4'd0;
                        shadow    <= {8*CMAX{1'b0}}; // unused bytes stay 0
                        adc_cs_n  <= 1'b0;
                        state     <= S_SETUP;
                    end
                end

            S_SETUP:
                if (setup_cnt == SETUP_END) state <= S_SHIFT;
                else                           setup_cnt <= setup_cnt + 1'b1;

            S_SHIFT:
                if (!adc_sclk) begin
                    // SCLK rising edge: sample the bit the ADC has been
                    // driving since the previous falling edge.
                    adc_sclk <= 1'b1;
                    byte_sr  <= {byte_sr[5:0], adc_miso};
                    if (bit_idx == 3'd7)
                        shadow[8*byte_idx +: 8] <= {byte_sr, adc_miso};
                    bit_idx  <= bit_idx + 1'b1;     // wraps 7 -> 0
                end else begin
                    // SCLK falling edge: the ADC shifts out its next bit.
                    adc_sclk <= 1'b0;
                    if (bit_idx == 3'd0) begin      // a byte just finished
                        if (byte_idx == n_bytes - 1'b1) begin
                            adc_cs_n  <= 1'b1;
                            rx_flat   <= shadow;    // publish whole frame
                            frame_tog <= ~frame_tog;
                            state     <= S_IDLE;
                        end else begin
                            byte_idx  <= byte_idx + 1'b1;
                        end
                    end
                end
            endcase
        end
    end

endmodule
