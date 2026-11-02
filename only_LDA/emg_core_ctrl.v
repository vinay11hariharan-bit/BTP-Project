// emg_core_ctrl — M6: sequences Stage 1 -> 2 -> 3 and masks the first decision after reset.
//
//   stride_done (Stage 1, 98th sample done) -> norm_start     : start Stage 2
//   norm_done   (Stage 2 finished)          -> swap, mac_start : flip the banks, start Stage 3
//   mac_class_valid / mac_class_out         -> class_valid / class_out at the chip pins,
//                                              except the first one after reset
//
// The hand-offs are plain wires: every source is already a registered 1-cycle pulse.
// swap is safe at norm_done: Stage 2 + 3 take at most ~850 cycles after stride_done,
// so Stage 1 is idle (the next frame arrives ~1015 cycles after the last one).
//
// First decision: after reset the partner bank holds garbage, so the first window is
// only half real. primed is set by the first Stage 3 result; only results that arrive
// while primed = 1 go out. class_out is registered and held until the next decision.
//
// chan_fault: updated together with every class_out that goes out, and held until the next
// one. 1 = in the data behind this decision, at least one channel
//   * had a WL window sum of 0 over the 196-sample window (emg_shift_norm.dead_mask), or
//   * was reported lead-off by its ADS1299 in any frame since the previous decision
//     (frame_lo, sampled at frame_start).
// so the decision may be wrong. Like class_out, it is not updated for the masked first decision.
//
// link_err: 1 = the ADC link is broken. It is set AT ONCE (not at the next decision) when
//   * a frame arrives with a bad ADS1299 status header (frame_hdr_err, sampled at frame_start):
//     a device is missing, the chain is broken, or cfg_dev_ch does not match the devices, or
//   * a sample period passed with no frame (frame_miss): no DRDY_N fall, so no decisions
//     come at all, which is why this flag cannot wait for the next decision.
// It clears at a decision when no such event happened since the previous decision.
`default_nettype none

module emg_core_ctrl (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       stride_done,      // from emg_feat_accum
    output wire       norm_start,       // to emg_shift_norm.start
    input  wire       norm_done,        // from emg_shift_norm.done
    output wire       swap,             // to emg_feat_accum.swap
    output wire       mac_start,        // to emg_lda_mac.start
    input  wire       mac_class_valid,  // from emg_lda_mac
    input  wire [2:0] mac_class_out,
    input  wire       dead_any,         // OR of emg_shift_norm.dead_mask (stable at mac_class_valid)
    input  wire       frame_start,      // a new ADC frame (from the CDC), 1-cycle pulse
    input  wire       frame_lo,         // OR of emg_spi_adc.lo_flat (stable at frame_start)
    input  wire       frame_hdr_err,    // emg_spi_adc.hdr_err      (stable at frame_start)
    input  wire       frame_miss,       // a sample period gave no frame (from the CDC), 1-cycle pulse
    output reg        class_valid,      // chip output: 1-cycle pulse per decision
    output reg  [2:0] class_out,        // chip output: held until the next decision
    output reg        primed,           // chip output: a full window has been seen
    output reg        chan_fault,       // chip output: a channel was dead or lead-off behind this decision
    output reg        link_err          // chip output: the ADC link is broken
);
    assign norm_start = stride_done;
    assign swap       = norm_done;
    assign mac_start  = norm_done;

    wire lo_now  = frame_start & frame_lo;
    wire err_now = (frame_start & frame_hdr_err) | frame_miss;

    reg  lo_seen;     // a lead-off was reported since the previous decision
    reg  err_seen;    // a link error happened since the previous decision

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_valid <= 1'b0;
            class_out   <= 3'd0;
            primed      <= 1'b0;
            chan_fault  <= 1'b0;
            link_err    <= 1'b0;
            lo_seen     <= 1'b0;
            err_seen    <= 1'b0;
        end else begin
            class_valid <= 1'b0;
            if (lo_now)  lo_seen <= 1'b1;
            if (err_now) begin
                err_seen <= 1'b1;
                link_err <= 1'b1;
            end
            if (mac_class_valid) begin
                if (primed) begin
                    class_valid <= 1'b1;
                    class_out   <= mac_class_out;
                    chan_fault  <= dead_any | lo_seen | lo_now;
                end
                primed   <= 1'b1;
                lo_seen  <= lo_now;                       // start counting for the next decision
                err_seen <= err_now;
                if (!err_seen && !err_now) link_err <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
