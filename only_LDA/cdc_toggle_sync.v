`timescale 1ns/1ps
// =============================================================================
// cdc_toggle_sync  (M8)  -  carries one event across clock domains
//
// The source domain flips tog_in once per event (emg_spi_adc.frame_tog). In
// the destination domain (clk_sys), two flip-flops (s1, s2) synchronise the
// level, and a third (s3) remembers s2 from one cycle earlier. s2 XOR s3 is 1
// for exactly one clk cycle after each change of tog_in, in either direction.
//
// Latency: the pulse appears after the 2nd clk edge that follows the toggle,
// or the 3rd if s1 went metastable and settled to the old value
// (= "2-3 clk_sys cycles" in the spec).
//
// Rules for the user of this block:
//  * Toggles must be at least 2 clk periods apart. Two toggles inside one
//    clk period can cancel and be lost. (Frames are 1 ms apart = 1000 cycles.)
//  * The reset value (0) must equal the source's reset value
//    (frame_tog resets to 0). Otherwise a false pulse follows reset.
//  * Data that goes with the event (rx_flat) is not synchronised here. It
//    must stay unchanged from the toggle until the receiver has read it.
//
// Layout note: s1 and s2 form a synchroniser. Keep them next to each other
// with no logic between them, and tell the STA/DFT flow they are
// synchroniser cells (no timing check on the tog_in -> s1 path).
// =============================================================================
module cdc_toggle_sync (
    input  wire clk,        // destination clock (clk_sys)
    input  wire rst_n,      // destination-domain reset (from rst_sync)
    input  wire tog_in,     // frame_tog from the clk_spi domain
    output wire pulse_out   // frame_start, one clk cycle per toggle
);
    reg s1, s2, s3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1 <= 1'b0;
            s2 <= 1'b0;
            s3 <= 1'b0;
        end else begin
            s1 <= tog_in;   // may go metastable: has a full cycle to settle
            s2 <= s1;       // clean copy of tog_in
            s3 <= s2;       // s2 one cycle ago
        end
    end

    assign pulse_out = s2 ^ s3;
endmodule
