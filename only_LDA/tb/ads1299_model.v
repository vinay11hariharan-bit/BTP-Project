`timescale 1ns/1ps
// =============================================================================
// ads1299_model  -  behavioural model of one TI ADS1299 / ADS1299-6 / ADS1299-4
//                   (NCH = 8, 6 or 4 electrodes), testbench only
//
// What is modelled (numbers from the ADS1299 datasheet, SBAS499C):
//  * SPI mode 1: DIN and DAISY_IN are latched on SCLK falling edges. DOUT changes after
//    SCLK rising edges: the old bit is held for tb.t_h (tDOHD, 10 ns), is unknown (x)
//    until tb.t_d (tDOPD, 17 ns at DVDD 3.3 V), then shows the new bit. The first rising
//    edge after CS_N falls puts out the MSB. CS_N low -> DOUT driven (x) after tb.t_csdod,
//    CS_N high -> DOUT high-Z after tb.t_csdoz.
//  * Commands on DIN (CS_N low): SDATAC (11h), RDATAC (10h), RESET (06h),
//    WREG (4rh, 0nh, n+1 data bytes), RREG (2rh, 0nh: ignored). Other bytes are ignored,
//    including the 00h that DIN carries during data reads. The device starts in RDATAC.
//  * RDATAC + single-shot (CONFIG4 bit 3): a START rising edge raises DRDY_N, starts one
//    conversion, and tb.t_conv later loads the output register and pulls DRDY_N low.
//    Output register = {status, ch1, ..., chNCH}, 24 bits each, MSB first, with
//    status = {1100, LOFF_STATP, LOFF_STATN, 0000}. LOFF_STATP bit i = electrode i is off
//    (from loffp_in) when CONFIG4 bit 1 (PD_LOFF_COMP) and LOFF_SENSP bit i are set,
//    the same for N.
//  * Daisy chain: each SCLK rising edge after the first shifts the register left and
//    takes the DAISY_IN bit latched at the previous falling edge into bit 0, so the next
//    device's frame follows this one on DOUT.
//
// Datasheet rules checked: each break adds 1 to `viol` and prints a message.
//    START high >= 2 tCLK                     CS_N high >= 2 tCLK (tCSH)
//    last SCLK fall -> CS_N high >= 4 tCLK    multi-byte command: >= 4 tCLK between byte ends
//    WREG only after SDATAC                   CS_N must not rise inside a command
//    START only in RDATAC + single-shot mode
// tCLK = tb.t_clk (488.28 ns for the 2.048 MHz clock).
// =============================================================================
module ads1299_model #(
    parameter integer NCH = 8            // electrodes per device: 8, 6 or 4
) (
    input  wire              start,
    input  wire              cs_n,
    input  wire              sclk,
    input  wire              din,
    input  wire              daisy_in,
    output wire              dout,
    output reg               drdy_n,
    input  wire [24*NCH-1:0] codes_in,  // electrode i at [24i +: 24], two's complement
    input  wire [NCH-1:0]    loffp_in,  // electrode i positive input off
    input  wire [NCH-1:0]    loffn_in   // electrode i negative input off
);
    localparam integer NB = 24 * (NCH + 1);   // bits in the output register

    // ------------------------------------------------------------ registers
    reg [7:0] regs [0:23];
    reg [7:0] config1, config3, config4, loff_sensp, loff_sensn;   // copies, for the testbench
    reg       rdatac;
    reg       chset_ok;                        // CH1SET .. CHnSET all written with 60h
    integer   viol;

    task reset_regs;
        integer r;
        begin
            for (r = 0; r < 24; r = r + 1) regs[r] = 8'h00;
            regs[8'h00] = (NCH == 8) ? 8'h3E : (NCH == 6) ? 8'h3D : 8'h3C;
            regs[8'h01] = 8'h96;  regs[8'h02] = 8'hC0;  regs[8'h03] = 8'h60;
            for (r = 5; r <= 12; r = r + 1) regs[r] = 8'h61;
            config1 = 8'h96;  config3 = 8'h60;  config4 = 8'h00;
            loff_sensp = 8'h00;  loff_sensn = 8'h00;  chset_ok = 1'b0;
            rdatac = 1'b1;
        end
    endtask

    task write_reg(input [4:0] a, input [7:0] v);
        integer r;
        begin
            regs[a] = v;
            config1 = regs[8'h01];  config3 = regs[8'h03];  config4 = regs[8'h17];
            loff_sensp = regs[8'h0F];  loff_sensn = regs[8'h10];
            chset_ok = 1'b1;
            for (r = 0; r < NCH; r = r + 1) if (regs[5 + r] != 8'h60) chset_ok = 1'b0;
        end
    endtask

    // ------------------------------------------------------------ DIN: commands
    reg [7:0] in_sr;
    integer   in_bits;
    integer   wr_left;
    reg [4:0] wr_addr;
    reg       expect_cnt, is_wreg, in_multi;
    realtime  last_byte_end, last_fall, last_cs_rise, start_rise;
    reg       dly_q;                           // DAISY_IN latched at the last falling edge

    initial begin
        reset_regs;
        viol = 0;  in_bits = 0;  wr_left = 0;  wr_addr = 0;
        expect_cnt = 0;  is_wreg = 0;  in_multi = 0;
        last_byte_end = -1.0e9;  last_fall = -1.0e9;  last_cs_rise = -1.0e9;  start_rise = -1.0e9;
        dly_q = 1'b0;  drdy_n = 1'b1;
    end

    task bad(input [8*72-1:0] msg);
        begin
            viol = viol + 1;
            $display("%0.3f us  ADS1299 %m: %0s", $realtime / 1000.0, msg);
        end
    endtask

    task process_byte(input [7:0] b);
        begin
            if (expect_cnt) begin
                wr_left    = b[4:0] + 1;
                expect_cnt = 0;
            end else if (wr_left > 0) begin
                if (is_wreg) write_reg(wr_addr, b);
                wr_addr = wr_addr + 1;
                wr_left = wr_left - 1;
                if (wr_left == 0) in_multi = 0;
            end else if (b == 8'h11) begin
                rdatac = 1'b0;
            end else if (b == 8'h10) begin
                rdatac = 1'b1;
            end else if (b == 8'h06) begin
                reset_regs;
            end else if (b[7:5] == 3'b010) begin   // WREG
                if (rdatac) bad("WREG sent while in RDATAC mode (send SDATAC first)");
                is_wreg = 1;  wr_addr = b[4:0];  expect_cnt = 1;  in_multi = 1;
            end else if (b[7:5] == 3'b001) begin   // RREG: not modelled, skip its bytes
                is_wreg = 0;  wr_addr = b[4:0];  expect_cnt = 1;  in_multi = 1;
            end
        end
    endtask

    always @(negedge sclk) if (!cs_n) begin
        dly_q     = daisy_in;
        last_fall = $realtime;
        in_sr     = {in_sr[6:0], din};
        in_bits   = in_bits + 1;
        if (in_bits == 8) begin
            in_bits = 0;
            if (in_multi && ($realtime - last_byte_end) < 4.0 * tb.t_clk)
                bad("less than 4 tCLK between bytes of a multi-byte command");
            last_byte_end = $realtime;
            process_byte(in_sr);
        end
    end

    // ------------------------------------------------------------ CS_N
    reg first;                                  // next SCLK rise is the first of this CS_N low
    reg dout_r;
    assign dout = dout_r;
    initial begin first = 1'b1;  dout_r = 1'bz; end

    always @(negedge cs_n) begin
        if ($realtime - last_cs_rise < 2.0 * tb.t_clk) bad("CS_N high for less than 2 tCLK");
        first = 1'b1;
        #(tb.t_csdod) if (!cs_n) dout_r = 1'bx;
    end

    always @(posedge cs_n) begin
        if ($realtime - last_fall < 4.0 * tb.t_clk) bad("CS_N rose less than 4 tCLK after the last SCLK fall");
        // a data read cut off inside a byte is not a command (DIN is 0 in RDATAC reads)
        if (in_multi || expect_cnt || (in_bits != 0 && !rdatac)) bad("CS_N rose in the middle of a command");
        last_cs_rise = $realtime;
        in_bits = 0;  in_multi = 0;  expect_cnt = 0;  wr_left = 0;
        #(tb.t_csdoz) if (cs_n) dout_r = 1'bz;
    end

    // ------------------------------------------------------------ conversion
    reg [NB-1:0]       sr;
    reg [24*NCH-1:0]   held;
    reg [7:0]          lp8, ln8;
    integer            i;

    always @(posedge start) begin
        start_rise = $realtime;
        if (!rdatac || !config4[3]) begin
            bad("START while not in RDATAC single-shot mode");
        end else begin
            held   = codes_in;
            lp8    = 8'h00;
            ln8    = 8'h00;
            if (config4[1])
                for (i = 0; i < NCH; i = i + 1) begin
                    lp8[i] = loffp_in[i] & loff_sensp[i];
                    ln8[i] = loffn_in[i] & loff_sensn[i];
                end
            drdy_n = 1'b1;
            #(tb.t_conv);
            sr[NB-1 -: 24] = {4'b1100, lp8, ln8, 4'b0000};
            for (i = 0; i < NCH; i = i + 1)
                sr[NB-1-24*(i+1) -: 24] = held[24*i +: 24];
            drdy_n = 1'b0;
        end
    end

    always @(negedge start)
        if ($realtime - start_rise < 2.0 * tb.t_clk) bad("START high for less than 2 tCLK");

    // ------------------------------------------------------------ DOUT
    reg nb;
    always @(posedge sclk) if (!cs_n) begin
        if (first) first = 1'b0;               // first rise: put out the MSB
        else       sr = {sr[NB-2:0], dly_q};   // later rises: shift, DAISY_IN enters at bit 0
        nb = sr[NB-1];
        #(tb.t_h)          if (!cs_n) dout_r = 1'bx;
        #(tb.t_d - tb.t_h) if (!cs_n) dout_r = nb;
    end
endmodule
