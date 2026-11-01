`default_nettype none

module emg_tp_ram #(
    parameter W     = 64,
    parameter DEPTH = 24,
    parameter AW    = 5
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [W-1:0]  wdata,
    input  wire          re,
    input  wire [AW-1:0] raddr,
    output reg  [W-1:0]  rdata
);

    reg [W-1:0] mem [0:DEPTH-1];
// no read-write collisions possible
    always @(posedge clk) begin
        if (we) begin
            mem[waddr] <= wdata;
        end
        if (re) begin
            rdata <= mem[raddr];
        end
    end

endmodule

`default_nettype wire
