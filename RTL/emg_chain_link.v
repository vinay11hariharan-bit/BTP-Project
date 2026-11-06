// emg_chain_link — joins up to 4 boards into one LDA classifier.
//
// Every board runs Stages 1-3 on its own channels (at most 8). Stage 3 is loaded with the
// K-1 difference rows W'_c = W_c - W_0 (c = 1..K-1), so its "scores" are partial sums
//     p_c = sum over this board's features of W'_cf * q_f      (+ b'_c on the head only).
// The decision only needs the totals D_c = sum over boards of p_c:
//     class = argmax { 0, D_1, ..., D_{K-1} }       (strict >, ties keep the lower class)
// which is exactly argmax_c s_c of the single-chip LDA.
//
// Chain, head = board with nothing upstream, tail = board with nothing downstream:
//
//   host -- [board 0] -- [board 1] -- [board 2]
//            head                       tail
//   run   ---------->  ---------->           common enable, passes straight through
//   id    ---------->  ---------->           board number: head 0, then +1 per board
//   up    <----------  <----------           partial sums, bit-serial, LSB first
//   rdy   ----+-----------+-------------+     wired-AND: high when every board is done
//
// Per stride:
//   1. Each board's Stage 3 runs; its time depends on the channel count, and boards with
//      fewer channels also finish their SPI read earlier, so they are not in step.
//   2. Stage 3 done: rdy = 1. The slowest board decides when all_rdy rises. rdy is cleared
//      on that rising edge (the same cycle on every board), not by the local stride_done,
//      so a fast board can never pair its new result with a slow board's old one.
//   3. On the all_rdy rising edge the tail starts sending its K-1 partials (32 bits each,
//      LSB first). Each board adds its own partial bit by bit (one full adder + carry) and
//      forwards the sum one clock later, so the chain adds only 1 cycle per board.
//   4. The head keeps the incoming totals, runs the argmax as each word completes and
//      pulses dec_valid / dec_cls after the last word.
//
// K-1 (number of words) is a register: cfg write 0x7F2 <= K-1 (1..3, reset 3).
`default_nettype none

module emg_chain_link #(
    parameter NV = 3,                       // max words per decision (K-1 for 4 gestures)
    parameter WB = 32                       // bits per word
) (
    input  wire               clk,
    input  wire               rst_n,

    // position in the chain (a neighbour present pulls the pin high)
    input  wire               prs_up_i,     // board upstream present
    input  wire               prs_dn_i,     // board downstream present
    output wire               is_head,
    output wire               is_tail,
    input  wire        [1:0]  id_i,
    output wire        [1:0]  id_o,
    output wire        [1:0]  my_id,

    // common enable
    input  wire               core_en,      // this board's own enable register
    input  wire               run_i,
    output wire               run_o,
    output wire               run,          // to the ADC interface of this board

    // configuration (this board's writes only)
    input  wire               cfg_we,
    input  wire        [11:0] cfg_addr,
    input  wire        [31:0] cfg_wdata,

    // local Stage 3
    input  wire               score_valid,
    input  wire        [2:0]  score_cls,
    input  wire signed [31:0] score,
    input  wire               mac_done,

    // barrier
    output reg                rdy_o,
    input  wire               all_rdy_i,

    // bit-serial partial sums
    input  wire               up_dat_i,
    input  wire               up_vld_i,
    output reg                up_dat_o,
    output reg                up_vld_o,

    // decision (head only)
    output reg                dec_valid,
    output reg         [2:0]  dec_cls
);
    localparam BW = $clog2(NV * WB);

    // ------------------------------------------------------------- position, id, enable
    assign is_head = ~prs_up_i;
    assign is_tail = ~prs_dn_i;
    assign my_id   = is_head ? 2'd0 : id_i;
    assign id_o    = my_id + 2'd1;

    reg en_r;                                   // head: registered enable for the whole chain
    always @(posedge clk or negedge rst_n)
        if (!rst_n) en_r <= 1'b0;
        else        en_r <= core_en;

    assign run   = is_head ? en_r : run_i;
    assign run_o = run;

    // ------------------------------------------------------------- K-1 register
    reg [1:0] n_val;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)                              n_val <= NV;
        else if (cfg_we && cfg_addr == 12'h7F2)  n_val <= cfg_wdata[1:0];

    wire [BW-1:0] n_bits = n_val * WB;

    // ------------------------------------------------------------- local partial sums
    reg [WB-1:0] part [0:NV-1];
    integer i;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) for (i = 0; i < NV; i = i + 1) part[i] <= {WB{1'b0}};
        else if (score_valid && score_cls < NV) part[score_cls] <= score;

    // ------------------------------------------------------------- barrier
    reg all_d;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) all_d <= 1'b0;
        else        all_d <= all_rdy_i;

    wire go = all_rdy_i & ~all_d;               // same cycle on every board

    always @(posedge clk or negedge rst_n)
        if (!rst_n)        rdy_o <= 1'b0;
        else if (go)       rdy_o <= 1'b0;
        else if (mac_done) rdy_o <= 1'b1;

    // ------------------------------------------------------------- tail: stream source
    reg [BW-1:0] t_left;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)                 t_left <= {BW{1'b0}};
        else if (go && is_tail)     t_left <= n_bits;
        else if (t_left != 0)       t_left <= t_left - 1'b1;

    wire bit_in = is_tail ? 1'b0       : up_dat_i;
    wire vld_in = is_tail ? (t_left != 0) : up_vld_i;

    // ------------------------------------------------------------- serial adder
    reg  [BW-1:0] bcnt;                         // position in the stream
    reg           carry;
    wire [1:0]    widx = bcnt / WB;
    wire [4:0]    bidx = bcnt % WB;
    wire          mine = part[widx][bidx];
    wire          sum  = bit_in ^ mine ^ carry;
    wire          cout = (bit_in & mine) | (carry & (bit_in ^ mine));
    wire          last_bit  = (bidx == WB - 1);
    wire          last_word = (widx == n_val - 1'b1);

    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            bcnt <= {BW{1'b0}};  carry <= 1'b0;
            up_dat_o <= 1'b0;    up_vld_o <= 1'b0;
        end else begin
            up_vld_o <= vld_in;
            up_dat_o <= vld_in & sum;
            if (go) begin
                bcnt <= {BW{1'b0}};  carry <= 1'b0;
            end else if (vld_in) begin
                bcnt  <= (last_bit && last_word) ? {BW{1'b0}} : bcnt + 1'b1;
                carry <= last_bit ? 1'b0 : cout;
            end
        end

    // ------------------------------------------------------------- head: collect + argmax
    reg        [WB-1:0] acc;                    // word being received, LSB first
    reg signed [WB-1:0] best;
    reg         [2:0]   best_cls;
    wire signed [WB-1:0] word = {sum, acc[WB-1:1]};

    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            acc <= {WB{1'b0}};  best <= {WB{1'b0}};  best_cls <= 3'd0;
            dec_valid <= 1'b0;  dec_cls <= 3'd0;
        end else begin
            dec_valid <= 1'b0;
            if (go) begin
                best <= {WB{1'b0}};  best_cls <= 3'd0;      // class 0 has D_0 = 0
            end else if (is_head && vld_in) begin
                acc <= word;
                if (last_bit) begin
                    if (word > best) begin
                        best <= word;  best_cls <= widx + 3'd1;
                    end
                    if (last_word) begin
                        dec_valid <= 1'b1;
                        dec_cls   <= (word > best) ? widx + 3'd1 : best_cls;
                    end
                end
            end
        end
endmodule

`default_nettype wire
