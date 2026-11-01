// emg_core_ctrl — sequences Stage 1 -> 2, and makes the frame header for the microcontroller.
//
// Step 3 (1 Nov 2026): the on-chip LDA (Stage 3) is gone. After Stage 2 the 5C features are in
// the x_q RAM, and this block publishes them as a frame over SPI (emg_spi_slave).
//
//   stride_done (Stage 1, bank complete) -> norm_start : start Stage 2
//   norm_done   (Stage 2 finished)       -> swap       : move the bank pointer
//                                        -> frame_pub  : the frame is ready (unless skipped)
//
// Read-out r = 0, 1, 2, ... counts Stage 2 runs since reset.
//
// Skip rule (option B): if a command is in progress on the SPI link (cs_low) when Stage 2
// starts, this read-out is not written: xq_we_en = 0 for the whole Stage 2 run, the frame
// header is not updated and frame_pub does not pulse. The next published frame has the
// "skipped" flag set. So a late read can lose a frame but never sees a mix of two.
//
// Frame header, latched at frame_pub:
//   f_count = r mod 256
//   f_flags bit 0  rest          rest_sum < T_REST (0x7F8): the summed size of all channels in this
//                                window is below the threshold (step 4). T_REST = 0: never.
//           bit 1  chan_fault    a channel was dead (dead_mask) in this window, or reported
//                                lead-off by its ADS1299 in a frame since the previous read-out
//           bit 2  link_err      a bad status header or a missing frame since the previous read-out
//           bit 3  cfg_err       a setting (C, B or N) is out of range
//           bit 4  skipped       at least one read-out was skipped since the previous frame
//           bit 5  valid         r >= cfg_warmup: the window is past the warm-up
//           bit 6  overrun       an ADC frame arrived while Stage 1 was busy (since reset)
//   f_dead  = dead_mask of this read-out
//   f_rest_sum = rest_sum of this read-out (readable at 0x7FB, for setting T_REST)
//
// status (config address 0x7FA, readable at any time, also when no frames come):
//   the same bits, live: link_err rises AT ONCE on a bad header or a missing frame (when the ADC
//   link breaks no read-outs happen, so no frames come and the microcontroller must poll this).
`default_nettype none

module emg_core_ctrl #(
    parameter CMAX = 24
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire [5:0]      cfg_warmup,       // read-outs before the window is valid
    input  wire            cfg_err,
    input  wire [20:0]     cfg_t_rest,       // rest threshold (0x7F8)
    input  wire [20:0]     rest_sum,         // from emg_shift_norm (stable at norm_done)
    input  wire            overrun,          // from emg_feat_accum (sticky)
    input  wire            cs_low,           // from emg_spi_slave: a command is in progress
    input  wire            stride_done,      // from emg_feat_accum
    output wire            norm_start,       // to emg_shift_norm.start
    input  wire            norm_done,        // from emg_shift_norm.done
    output wire            swap,             // to emg_feat_accum.swap
    output reg             xq_we_en,         // Stage 2 may write the x_q RAM
    input  wire [CMAX-1:0] dead_mask,        // from emg_shift_norm (stable at norm_done)
    input  wire            frame_start,      // a new ADC frame (from the CDC), 1-cycle pulse
    input  wire            frame_lo,         // OR of emg_spi_adc.lo_flat (stable at frame_start)
    input  wire            frame_hdr_err,    // emg_spi_adc.hdr_err      (stable at frame_start)
    input  wire            frame_miss,       // a sample period gave no frame (from the CDC)
    output reg             frame_pub,        // 1-cycle pulse: a new frame is ready
    output reg  [7:0]      f_count,
    output reg  [7:0]      f_flags,
    output reg  [23:0]     f_dead,
    output reg  [20:0]     f_rest_sum,       // of the last published frame (0x7FB)
    output wire [7:0]      status            // live flags, config address 0x7FA
);
    assign norm_start = stride_done;
    assign swap       = norm_done;

    wire lo_now  = frame_start & frame_lo;
    wire err_now = (frame_start & frame_hdr_err) | frame_miss;

    reg       lo_seen;      // a lead-off was reported since the previous read-out
    reg       err_seen;     // a link error happened since the previous read-out
    reg       link_err;     // live
    reg       fault_last;   // chan_fault of the last read-out
    reg       valid_last;   // valid of the last read-out
    reg       rest_last;    // rest of the last read-out
    reg       skipped;      // a read-out was skipped since the last published frame
    reg [7:0] rcnt;         // read-outs since reset, mod 256
    reg [5:0] wcnt;         // read-outs since reset, saturating at 63 (warm-up)

    wire       fault_now = (|dead_mask) | lo_seen | lo_now;
    wire       link_now  = err_seen | err_now;
    wire       valid_now = (wcnt >= cfg_warmup);
    wire       rest_now  = (rest_sum < cfg_t_rest);
    wire [23:0]     dm   = dead_mask;                    // zero-extended to 24 channels

    assign status = {1'b0, overrun, valid_last, skipped, cfg_err, link_err, fault_last, rest_last};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            xq_we_en  <= 1'b1;
            frame_pub <= 1'b0;
            f_count   <= 8'd0;
            f_flags   <= 8'd0;
            f_dead    <= 24'd0;
            f_rest_sum <= 21'd0;
            rest_last <= 1'b0;
            lo_seen   <= 1'b0;
            err_seen  <= 1'b0;
            link_err  <= 1'b0;
            fault_last <= 1'b0;
            valid_last <= 1'b0;
            skipped   <= 1'b0;
            rcnt      <= 8'd0;
            wcnt      <= 6'd0;
        end else begin
            frame_pub <= 1'b0;
            if (lo_now)  lo_seen <= 1'b1;
            if (err_now) begin
                err_seen <= 1'b1;
                link_err <= 1'b1;
            end
            if (stride_done) xq_we_en <= ~cs_low;            // skip rule, decided at Stage 2 start
            if (norm_done) begin
                if (xq_we_en) begin                           // publish
                    frame_pub <= 1'b1;
                    f_count   <= rcnt;
                    f_flags   <= {1'b0, overrun, valid_now, skipped, cfg_err, link_now, fault_now, rest_now};
                    f_dead    <= dm;
                    f_rest_sum <= rest_sum;
                    skipped   <= 1'b0;
                end else begin
                    skipped   <= 1'b1;
                end
                fault_last <= fault_now;
                valid_last <= valid_now;
                rest_last  <= rest_now;
                rcnt       <= rcnt + 8'd1;
                if (wcnt != 6'd63) wcnt <= wcnt + 6'd1;
                lo_seen    <= lo_now;                         // start counting for the next read-out
                err_seen   <= err_now;
                if (!err_seen && !err_now) link_err <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
