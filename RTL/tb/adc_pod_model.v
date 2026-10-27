`timescale 1ns/1ps
// =============================================================================
// adc_pod_model  -  behavioural model of one 8-channel ADC pod (testbench only)
//
// SPI mode 0 slave with a daisy-chain input:
//  * CONVST rising edge: the 8 input values are sampled (all at once) and
//    DRDY_N goes high. tb.t_conv later the 64-bit output register is loaded
//    and DRDY_N goes low.
//  * Output register = {ch0, ch1, ..., ch7}, so ch0 leaves first, MSB first.
//  * CS_N falling edge: SDO shows the first bit tb.t_en later.
//  * Each SCLK falling edge (CS_N low): the register shifts left by one and
//    takes SDI (the SDO of the pod before it in the chain) into bit 0.
//    SDO keeps the old bit for tb.t_h, is unknown (x) until tb.t_d, then
//    shows the new bit. The x window makes a wrongly timed sample visible.
//  * Tie SDI of the first pod in the chain to 0. The pod wired to MISO sends
//    its bytes first, so its 8 electrodes are channels 0-7.
// =============================================================================
module adc_pod_model (
    input  wire        convst,
    input  wire        cs_n,
    input  wire        sclk,
    input  wire        sdi,
    output wire        sdo,
    output reg         drdy_n,
    input  wire [63:0] sample_in
);
    reg [63:0] held;
    reg [63:0] sr;
    reg        sdo_r;
    assign sdo = sdo_r;

    initial begin
        drdy_n = 1'b1;
        sr     = 64'd0;
        sdo_r  = 1'bz;
    end

    always @(posedge convst) begin
        held   = sample_in;
        drdy_n = 1'b1;
        #(tb.t_conv);
        sr     = held;
        drdy_n = 1'b0;
    end

    always @(negedge sclk) if (!cs_n) sr <= {sr[62:0], sdi};

    // CS_N is checked again after each delay: when CS_N rises together with the
    // last SCLK fall, the pod must stay tri-stated instead of driving SDO late.
    always @(negedge sclk) if (!cs_n) begin
        #(tb.t_h)          if (!cs_n) sdo_r = 1'bx;
        #(tb.t_d - tb.t_h) if (!cs_n) sdo_r = sr[63];
    end

    always @(negedge cs_n) begin
        sdo_r = 1'bx;
        #(tb.t_en) sdo_r = sr[63];
    end

    always @(posedge cs_n) sdo_r = 1'bz;
endmodule
