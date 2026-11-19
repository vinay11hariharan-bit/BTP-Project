`timescale 1ns/1ps
// fpga_pod - synthesizable stand-in for one 8-channel ADC pod (adc_pod_model.v in the testbench)
//
// Runs on the chip's own clock, so the SPI pins are just registers of the same clock:
//   CONVST rise     : latch sample_in, DRDY_N high; CONV_CYC cycles later load the
//                     shift register and pull DRDY_N low.
//   CS_N low        : SDO = bit 63 of the shift register (channel 0, MSB first).
//   after each SCLK rise (the chip samples SDO at that rise) the register shifts left by one,
//   so the next bit is ready long before the next rise.
module fpga_pod #(
    parameter CONV_CYC = 5            // ~20 us at 256 kHz, like the testbench
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        convst,
    input  wire        cs_n,
    input  wire        sclk,
    output wire        sdo,
    output reg         drdy_n,
    input  wire [63:0] sample_in
);
    reg [63:0] held, sr;
    reg        convst_d, sclk_d;
    reg [3:0]  conv_cnt;

    assign sdo = sr[63];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            held <= 64'd0; sr <= 64'd0; drdy_n <= 1'b1;
            convst_d <= 1'b0; sclk_d <= 1'b0; conv_cnt <= 4'd0;
        end else begin
            convst_d <= convst;
            sclk_d   <= sclk;
            if (convst & ~convst_d) begin
                held     <= sample_in;
                drdy_n   <= 1'b1;
                conv_cnt <= CONV_CYC[3:0];
            end else if (conv_cnt != 4'd0) begin
                conv_cnt <= conv_cnt - 1'b1;
                if (conv_cnt == 4'd1) begin
                    sr     <= held;
                    drdy_n <= 1'b0;
                end
            end else if (!cs_n && sclk && !sclk_d) begin
                sr <= {sr[62:0], 1'b0};
            end
        end
    end
endmodule
