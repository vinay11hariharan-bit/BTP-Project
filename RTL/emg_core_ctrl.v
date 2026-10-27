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
    output reg        class_valid,      // chip output: 1-cycle pulse per decision
    output reg  [2:0] class_out,        // chip output: held until the next decision
    output reg        primed            // chip output: a full window has been seen
);
    assign norm_start = stride_done;
    assign swap       = norm_done;
    assign mac_start  = norm_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_valid <= 1'b0;
            class_out   <= 3'd0;
            primed      <= 1'b0;
        end else begin
            class_valid <= 1'b0;
            if (mac_class_valid) begin
                if (primed) begin
                    class_valid <= 1'b1;
                    class_out   <= mac_class_out;
                end
                primed <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
