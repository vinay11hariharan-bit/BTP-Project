// emg_chain_link — lets several boards act as one LDA classifier.
//
// Each board only has the features of its own channels, so its Stage 3 can only give a
// partial score. We load Stage 3 with the difference rows W'_c = W_c - W_0 (c = 1..K-1),
// add the partial scores of all boards, and pick
//
//     class = argmax{ 0, D_1, ..., D_(K-1) }        D_c = sum of p_c over all boards
//
// which gives the same answer as the single-chip LDA. The head adds the bias b'_c in its
// own Stage 3, the other boards load b' = 0.
//
// The boards sit in a line. The one with nothing upstream is the head (id 0, it decides);
// the one with nothing downstream is the tail (it starts the transfer).
//
// Every stride:
//   - a board raises rdy when its Stage 3 is done. rdy is a wired-AND on the boards, so
//     all_rdy goes high only when the slowest board is done.
//   - on that edge every board drops rdy (same cycle everywhere) and the tail starts
//     shifting out its K-1 partials, 32 bits each, LSB first.
//   - every other board adds its own partial to the incoming bits with one full adder and
//     passes the sum on a clock later. The head keeps the totals and does the argmax.
//
// rdy is cleared on the common edge and not on the local stride_done: a board with fewer
// channels finishes its SPI read early and would otherwise mix its new partial with an
// old one from a slower board.
//
// The number of words (K-1) is written at 0x7F2, default 3.
`default_nettype none

module emg_chain_link #(
    parameter NV = 3,       // max partial sums per board (K-1)
    parameter WB = 32       // bits per partial sum
) (
    input  wire               clk,
    input  wire               rst_n,

    // where we are in the chain (pins pulled high by a neighbour)
    input  wire               prs_up_i,
    input  wire               prs_dn_i,
    output wire               is_head,
    output wire               is_tail,
    input  wire        [1:0]  id_i,
    output wire        [1:0]  id_o,
    output wire        [1:0]  my_id,

    // start: the head's core_en runs every board
    input  wire               core_en,
    input  wire               run_i,
    output wire               run_o,
    output wire               run,

    // config writes for this board
    input  wire               cfg_we,
    input  wire        [11:0] cfg_addr,
    input  wire        [31:0] cfg_wdata,

    // our own Stage 3 results
    input  wire               score_valid,
    input  wire        [2:0]  score_cls,
    input  wire signed [31:0] score,
    input  wire               mac_done,

    output reg                rdy_o,
    input  wire               all_rdy_i,

    // serial link towards the head
    input  wire               up_dat_i,
    input  wire               up_vld_i,
    output reg                up_dat_o,
    output reg                up_vld_o,

    // final decision (only used on the head)
    output reg                dec_valid,
    output reg         [2:0]  dec_cls
);

    // position and id
    assign is_head = ~prs_up_i;
    assign is_tail = ~prs_dn_i;
    assign my_id   = is_head ? 2'd0 : id_i;
    assign id_o    = my_id + 2'd1;

    // The head registers its enable once; everyone else just passes it on, so all
    // boards see run rise in the same cycle and their CONVST timers stay together.
    reg head_en;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) head_en <= 1'b0;
        else        head_en <= core_en;

    assign run   = is_head ? head_en : run_i;
    assign run_o = run;

    // number of words per decision
    reg [1:0] nwords;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)                             nwords <= NV;
        else if (cfg_we && cfg_addr == 12'h7F2) nwords <= cfg_wdata[1:0];

    // keep our partial sums as Stage 3 hands them out
    reg [WB-1:0] part [0:NV-1];
    integer i;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)
            for (i = 0; i < NV; i = i + 1) part[i] <= 0;
        else if (score_valid && score_cls < NV)
            part[score_cls] <= score;

    // wait for everyone
    reg all_rdy_d;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) all_rdy_d <= 1'b0;
        else        all_rdy_d <= all_rdy_i;

    wire go = all_rdy_i & ~all_rdy_d;

    always @(posedge clk or negedge rst_n)
        if (!rst_n)        rdy_o <= 1'b0;
        else if (go)       rdy_o <= 1'b0;
        else if (mac_done) rdy_o <= 1'b1;

    // The tail has nobody below it, so it makes its own valid and adds to zeros.
    reg [6:0] tail_left;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)               tail_left <= 0;
        else if (go && is_tail)   tail_left <= nwords * WB;
        else if (tail_left != 0)  tail_left <= tail_left - 1'b1;

    wire in_bit = is_tail ? 1'b0 : up_dat_i;
    wire in_vld = is_tail ? (tail_left != 0) : up_vld_i;

    // Serial add: which word and which bit we are on, plus one carry.
    reg [1:0] word;
    reg [4:0] bit_n;
    reg       carry;

    wire my_bit    = part[word][bit_n];
    wire sum       = in_bit ^ my_bit ^ carry;
    wire carry_nxt = (in_bit & my_bit) | (carry & (in_bit ^ my_bit));
    wire last_bit  = (bit_n == WB - 1);
    wire last_word = (word == nwords - 1'b1);

    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            word <= 0; bit_n <= 0; carry <= 1'b0;
            up_dat_o <= 1'b0; up_vld_o <= 1'b0;
        end else begin
            up_vld_o <= in_vld;
            up_dat_o <= in_vld & sum;

            if (go) begin
                word <= 0; bit_n <= 0; carry <= 1'b0;
            end else if (in_vld) begin
                bit_n <= bit_n + 1'b1;          // wraps 31 -> 0 by itself
                carry <= last_bit ? 1'b0 : carry_nxt;
                if (last_bit)
                    word <= last_word ? 2'd0 : word + 1'b1;
            end
        end

    // Head: shift each total in, compare when it is complete. Class 0 is the baseline
    // (D_0 = 0), and a strict > keeps the lower class on a tie, like the single chip.
    reg        [WB-1:0] shreg;
    reg signed [WB-1:0] best;
    reg         [2:0]   best_cls;
    wire signed [WB-1:0] total = {sum, shreg[WB-1:1]};

    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            shreg <= 0; best <= 0; best_cls <= 0;
            dec_valid <= 1'b0; dec_cls <= 0;
        end else begin
            dec_valid <= 1'b0;

            if (go) begin
                best     <= 0;
                best_cls <= 0;
            end else if (is_head && in_vld) begin
                shreg <= total;
                if (last_bit) begin
                    if (total > best) begin
                        best     <= total;
                        best_cls <= word + 1'b1;
                    end
                    if (last_word) begin
                        dec_valid <= 1'b1;
                        dec_cls   <= (total > best) ? word + 1'b1 : best_cls;
                    end
                end
            end
        end

endmodule

`default_nettype wire
