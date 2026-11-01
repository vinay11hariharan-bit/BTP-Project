// emg_lda_mac — Stage 3: 6-class single-MAC LDA with streaming argmax.
//
// YOUR MODULE. Keep the port names and widths exactly as they are (the testbench
// depends on them); you may change reg <-> wire. Replace every TODO. Check with:
//     checked by the chip testbench: python tb/test_emg_top.py
//
// For each class cls = 0..5, with D = 5C features:
//     score[cls] = b[cls] + sum_{f=0}^{D-1} x_q[f] * W[cls][f]          (exact INT32)
// W[cls][f] is at weight address cls*D + f: walk it with ONE counter that increments
// on every MAC (no cls*D multiply). After each class, pulse score_valid for one cycle
// with score_cls = cls and score = its final value (in order 0, 1, ..., 5).
// Keep a running maximum: class 0 always loads it; a later class replaces it only if
// its score is STRICTLY greater (ties keep the lower class index).
// After class 5, pulse class_valid for one cycle with class_out = the winning class,
// and drop busy.
//
// All three memories have 1-cycle read latency: data appears on xq_data / w_data /
// b_data in the cycle AFTER you assert the matching *_re with the address.
`default_nettype none

module emg_lda_mac #(
    parameter NCLS = 6
) (
    input  wire               clk,
    input  wire               rst_n,         // asynchronous, active low
    input  wire [4:0]         cfg_n_ch,      // C, 1..24  ->  D = 5C (40..120)

    input  wire               start,         // 1-cycle pulse (Stage 2 done); ignored while busy
    output reg                busy,

    // x_q register file (written by Stage 2)
    output wire               xq_re,
    output wire [6:0]         xq_addr,       // f
    input  wire signed [7:0]  xq_data,  // feature data 

    // weight memory, W[cls][f] at cls*D + f
    output wire               w_re,
    output wire [9:0]         w_addr,
    input  wire signed [7:0]  w_data,

    // bias memory
    output wire               b_re,
    output wire [2:0]         b_addr,        // cls
    input  wire signed [31:0] b_data,

    // per-class result (for debug and the testbench)
    output reg                score_valid,   // 1-cycle pulse per class
    output reg  [2:0]         score_cls,
    output wire signed [31:0] score,

    // final decision
    output reg                class_valid,   // 1-cycle pulse
    output reg  [2:0]         class_out
); 
    wire [6:0] f_last   = {cfg_n_ch, 2'b00} + {2'b00, cfg_n_ch} - 7'd1;
    wire [2:0] cls_last = NCLS[2:0] - 3'd1;

    // ------------------------------------------------------------------ issue stage
    reg        iss_active;
    reg        iss_bias;      // 1 = bias issue cycle, 0 = feature issue cycle 
    reg  [2:0] c_class;       // 0 .. NCLS-1
    reg  [6:0] c_feature;     // 0 .. D-1
    reg  [9:0] w_addr_cnt;    // 0 .. NCLS*D-1 (max 719)

    wire start_accept = start & ~busy;

    assign b_re    = iss_active & iss_bias;
    assign b_addr  = c_class;

    assign xq_re   = iss_active & ~iss_bias;//xq and w's inputs should be same as they are always multipied at the same edge 
    assign xq_addr = c_feature;

    assign w_re    = iss_active & ~iss_bias;
    assign w_addr  = w_addr_cnt;

    wire iss_last_f   = (c_feature == f_last); 
    wire iss_last_cls = (c_class == cls_last);
   
    reg               mac_bias;
    reg               mac_feat;
    reg               mac_last_f;
    reg               mac_last_cls;
    reg  [2:0]        mac_cls;
    
    reg  signed [31:0] acc;
    reg  signed [31:0] score_reg;
    reg  signed [31:0] max_score;

    wire signed [15:0] prod    = xq_data * w_data;
    wire signed [31:0] prod_32 = {{16{prod[15]}}, prod}; //sign extending the product
    wire signed [31:0] next_acc = acc + prod_32;

    assign score = score_reg;

    // ------------------------------------------------------------------ argmax + outputs
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy         <= 1'b0;
            iss_active   <= 1'b0;
            iss_bias     <= 1'b0;
            c_class      <= 3'd0;
            c_feature    <= 7'd0;
            w_addr_cnt   <= 10'd0;

            mac_bias     <= 1'b0;
            mac_feat     <= 1'b0;
            mac_last_f   <= 1'b0;
            mac_last_cls <= 1'b0;
            mac_cls      <= 3'd0;

            acc          <= 32'sd0;
            score_reg    <= 32'sd0;
            max_score    <= 32'sd0;

            score_valid  <= 1'b0;
            score_cls    <= 3'd0;
            class_valid  <= 1'b0;
            class_out    <= 3'd0;
        end else begin
            score_valid <= 1'b0;
            class_valid <= 1'b0;

            // 1. Issue stage state machine
            if (start_accept) begin
                busy       <= 1'b1;
                iss_active <= 1'b1;
                iss_bias   <= 1'b1;
                c_class    <= 3'd0;
                c_feature  <= 7'd0;
                w_addr_cnt <= 10'd0;
            end else if (iss_active) begin
                if (iss_bias) begin
                    iss_bias  <= 1'b0;
                    c_feature <= 7'd0;
                end else begin
                    w_addr_cnt <= w_addr_cnt + 10'd1;
                    if (iss_last_f) begin
                        c_feature <= 7'd0;
                        if (iss_last_cls) begin
                            iss_active <= 1'b0;
                        end else begin
                            c_class  <= c_class + 3'd1;
                            iss_bias <= 1'b1;
                        end
                    end else begin
                        c_feature <= c_feature + 7'd1;
                    end
                end
            end

            // 2. Pipeline registers from Issue stage to MAC stage (1-cycle RAM latency)
            mac_bias     <= iss_active & iss_bias;
            mac_feat     <= iss_active & ~iss_bias;
            mac_last_f   <= iss_active & ~iss_bias & iss_last_f;
            mac_last_cls <= iss_active & ~iss_bias & iss_last_f & iss_last_cls;
            mac_cls      <= c_class;

            // 3. MAC accumulation
            if (mac_bias) begin
                acc <= b_data;
            end else if (mac_feat) begin
                acc <= next_acc;
            end

            // 4. Per-class score output, running argmax, and final completion
            if (mac_last_f) begin
                score_valid <= 1'b1;
                score_cls   <= mac_cls;
                score_reg   <= next_acc;

                if ((mac_cls == 3'd0) || (next_acc > max_score)) begin
                    max_score <= next_acc;
                    class_out <= mac_cls;
                end

                if (mac_last_cls) begin
                    class_valid <= 1'b1;
                    busy        <= 1'b0;
                end
            end
        end
    end   
endmodule

`default_nettype wire
