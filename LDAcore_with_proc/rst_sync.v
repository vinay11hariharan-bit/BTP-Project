`timescale 1ns/1ps
// =============================================================================
// rst_sync  (M8)  -  asynchronous assert, synchronous de-assert reset
//
// One instance per clock domain (clk_sys and clk_spi).
//  * arst_n falls -> both flip-flops clear at once, so rst_n falls at once,
//    with or without a running clock.
//  * arst_n rises -> r1 takes the constant 1 at the next clk edge, r2 copies
//    it at the edge after. rst_n therefore rises exactly at a clk edge, 2 edges
//    after arst_n (3 if arst_n rose so close to an edge that r1 went
//    metastable and settled to 0). Every flip-flop of the domain leaves reset
//    on that same edge.
//
// Layout note: r1 and r2 form a synchroniser. Keep them next to each other
// and tell the STA flow they are synchroniser cells.
// =============================================================================
module rst_sync (
    input  wire clk,      // the domain's clock
    input  wire arst_n,   // chip reset pin, asynchronous, active low
    output wire rst_n     // reset for this domain, active low
);
    reg r1, r2;

    always @(posedge clk or negedge arst_n) begin
        if (!arst_n) begin
            r1 <= 1'b0;
            r2 <= 1'b0;
        end else begin
            r1 <= 1'b1;
            r2 <= r1;
        end
    end

    assign rst_n = r2;
endmodule
