`default_nettype none

module emg_feat_accum #(
    parameter CMAX    = 24,
    parameter SUB_WIN = 98
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire [4:0]        cfg_n_ch,

    input  wire              frame_start,
    input  wire [8*CMAX-1:0] rx_flat,
    output reg               busy,
    output reg               sample_done,
    output reg               stride_done,
    output reg               overrun,

    input  wire              swap,
    output reg               bank_sel,

    input  wire              s2_re,
    input  wire [4:0]        s2_addr,
    output wire [63:0]       s2_q0,
    output wire [63:0]       s2_q1
);

    reg        rd_active;
    reg  [4:0] rd_ch;

    reg        wr_valid;
    reg  [4:0] wr_ch;
    reg  [7:0] wr_x;

    reg  [6:0] cnt_win;
    reg        first_sample;

    wire start_accept = frame_start & ~busy;//real start 

    wire       s1_re   = start_accept | rd_active;
    wire [4:0] s1_rch  = start_accept ? 5'd0 : rd_ch;
    wire [7:0] s1_rx_c = rx_flat[s1_rch * 8 +: 8];

    wire last_write_cycle = wr_valid & ~rd_active;
    wire win_last_sample  = (cnt_win == (SUB_WIN[6:0] - 7'd1));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_active    <= 1'b0;
            rd_ch        <= 5'd0;
            wr_valid     <= 1'b0;
            wr_ch        <= 5'd0;
            wr_x         <= 8'd0;
            cnt_win      <= 7'd0;
            first_sample <= 1'b1;
            busy         <= 1'b0;
            sample_done  <= 1'b0;
            stride_done  <= 1'b0;
            overrun      <= 1'b0;
            bank_sel     <= 1'b0;
        end else begin
            sample_done <= 1'b0;
            stride_done <= 1'b0;

            if (frame_start && busy) begin // overrun condition 
                overrun <= 1'b1;
            end

            if (swap && !busy && !frame_start) begin
                bank_sel <= ~bank_sel;
            end

            if (start_accept) begin //actual start 
                busy <= 1'b1;
                if (cfg_n_ch > 5'd1) begin
                    rd_active <= 1'b1;
                    rd_ch     <= 5'd1;
                end else begin // if channels less than 2
                    rd_active <= 1'b0;
                    rd_ch     <= 5'd0;
                end
            end else if (rd_active) begin  //rd_ch inc logic
                if (rd_ch == (cfg_n_ch - 5'd1)) begin //read done 
                    rd_active <= 1'b0;
                end else begin
                    rd_ch <= rd_ch + 5'd1;
                end
            end

            wr_valid <= s1_re;
            if (s1_re) begin
                wr_ch <= s1_rch;
                wr_x  <= s1_rx_c;
            end

            if (last_write_cycle) begin
                busy         <= 1'b0;
                sample_done  <= 1'b1;
                stride_done  <= win_last_sample;
                first_sample <= 1'b0;

                if (win_last_sample) begin
                    cnt_win <= 7'd0;
                end else begin
                    cnt_win <= cnt_win + 7'd1;
                end
            end
        end
    end

    wire [9:0] hist_rdata;
    wire [9:0] hist_wdata;
//hist bank 
    emg_tp_ram #(
        .DEPTH (CMAX),
        .W (10)
    ) u_hist_ram (
        .clk   (clk),
        .we    (wr_valid),
        .waddr (wr_ch),
        .wdata (hist_wdata),
        .re    (s1_re),
        .raddr (s1_rch),
        .rdata (hist_rdata)
    );

    //these two bank should only be readable when frame ends and calculation done 
    wire       bank_re    = (busy | start_accept) ? s1_re  : s2_re; 
    wire [4:0] bank_raddr = (busy | start_accept) ? s1_rch : s2_addr;

    wire       bank0_we   = wr_valid & (~bank_sel);
    wire       bank1_we   = wr_valid & ( bank_sel);
    wire [63:0] bank_wdata;

//bank 0
    emg_tp_ram #(
        .DEPTH (CMAX),
        .W (64)
    ) u_bank0_ram (
        .clk   (clk),
        .we    (bank0_we),
        .waddr (wr_ch),
        .wdata (bank_wdata),
        .re    (bank_re),
        .raddr (bank_raddr),
        .rdata (s2_q0)
    );
//bank 1
    emg_tp_ram #(
        .DEPTH (CMAX),
        .W (64)
    ) u_bank1_ram (
        .clk   (clk),
        .we    (bank1_we),
        .waddr (wr_ch),
        .wdata (bank_wdata),
        .re    (bank_re),
        .raddr (bank_raddr),
        .rdata (s2_q1)
    );

    wire signed [7:0] x_prev = first_sample ? 8'sd0 : $signed(hist_rdata[9:2]);
    wire              d0_pos = first_sample ? 1'b0  : hist_rdata[1];
    wire              d0_neg = first_sample ? 1'b0  : hist_rdata[0];

    wire [7:0]  alu_sabs;
    wire [7:0]  alu_wl;
    wire        alu_zc;
    wire        alu_ssc;
    wire [14:0] alu_pow;
    wire        alu_d1_pos;
    wire        alu_d1_neg;
//alu instantiation
    emg_feat_alu u_feat_alu (
        .x      (wr_x),
        .x_prev   (x_prev),
        .d0_pos (d0_pos),
        .d0_neg (d0_neg),
        .sabs   (alu_sabs),
        .wl     (alu_wl),
        .zc     (alu_zc),
        .ssc    (alu_ssc),
        .pow    (alu_pow),
        .d1_pos (alu_d1_pos),
        .d1_neg (alu_d1_neg)
    );

    assign hist_wdata = {wr_x, alu_d1_pos, alu_d1_neg};

    wire [63:0] active_bank_q = bank_sel ? s2_q1 : s2_q0;
    wire [63:0] base_word     = (cnt_win == 7'd0) ? 64'd0 : active_bank_q;

    wire [13:0] acc_sabs = base_word[13:0]  + {6'd0, alu_sabs};
    wire [14:0] acc_wl   = base_word[28:14] + {7'd0, alu_wl};
    wire [6:0]  acc_zc   = base_word[35:29] + {6'd0, alu_zc};
    wire [6:0]  acc_ssc  = base_word[42:36] + {6'd0, alu_ssc};
    wire [20:0] acc_pow  = base_word[63:43] + {6'd0, alu_pow};

    assign bank_wdata = {acc_pow, acc_ssc, acc_zc, acc_wl, acc_sabs};

endmodule

`default_nettype wire
