`timescale 1ns/1ps
// tb_chain_unit — self-checking test of emg_chain_link alone, 1 to 4 boards.
//
// Each board gets a small stand-in for Stage 3: after a random delay it hands out its
// partial sums (score_valid, one per cycle) and then pulses mac_done. The testbench
// adds the partials itself, takes argmax{0, totals} with ties going to the lower class,
// and checks the head's decision. Nothing here comes from the Python model.
//
// Checked every stride:
//   - exactly one dec_valid at the head, with the expected class
//   - no dec_valid from any other board
//   - ids 0,1,2,... and run reaching every board in the same cycle (checked once)
// Values: random 32-bit partials (kept so the sum fits), small ones, negative ones,
// forced ties, all-negative (class 0 must win), and nwords = 3, 2, 1.
//
//   iverilog -g2005 -s tb_chain_unit -Ptb_chain_unit.NB=3 -o u.vvp emg_chain_link.v tb/tb_chain_unit.v
//   vvp -n u.vvp            ->  prints "UNIT PASS ..." or "UNIT FAIL ..."
module tb_chain_unit;
    parameter NB      = 3;
    parameter STRIDES = 3000;
    parameter SEED    = 1;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg        cfg_we = 0;
    reg [11:0] cfg_addr = 0;
    reg [31:0] cfg_wdata = 0;
    reg        core_en = 0;

    reg               sv   [0:NB-1];
    reg        [2:0]  scls [0:NB-1];
    reg signed [31:0] sc   [0:NB-1];
    reg               done [0:NB-1];

    wire [1:0] id  [0:NB];
    wire       run [0:NB];
    wire [NB:0] dat, vld;
    wire [NB-1:0] rdy, dv, run_seen;
    wire [2:0] dc  [0:NB-1];
    wire [1:0] my_id [0:NB-1];
    wire all_rdy = &rdy;

    assign id[0] = 2'd3;            // must be ignored by the head
    assign run[0] = 1'b0;
    assign dat[NB] = 1'b0;
    assign vld[NB] = 1'b0;

    genvar j;
    generate for (j = 0; j < NB; j = j + 1) begin : b
        emg_chain_link u (
            .clk(clk), .rst_n(rst_n),
            .prs_up_i(j > 0), .prs_dn_i(j < NB - 1), .is_head(), .is_tail(),
            .id_i(id[j]), .id_o(id[j+1]), .my_id(my_id[j]),
            .core_en(j == 0 ? core_en : 1'b0), .run_i(run[j]), .run_o(run[j+1]), .run(run_seen[j]),
            .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
            .score_valid(sv[j]), .score_cls(scls[j]), .score(sc[j]), .mac_done(done[j]),
            .rdy_o(rdy[j]), .all_rdy_i(all_rdy),
            .up_dat_i(dat[j+1]), .up_vld_i(vld[j+1]), .up_dat_o(dat[j]), .up_vld_o(vld[j]),
            .dec_valid(dv[j]), .dec_cls(dc[j]));
    end endgenerate

    // ------------------------------------------------------------- bookkeeping
    integer errors = 0, decs = 0, extra = 0, k, n, t, w, s;
    integer seed = SEED;
    reg signed [31:0] p   [0:NB-1][0:2];
    integer           dly [0:NB-1];
    reg signed [31:0] tot [0:2];
    reg signed [31:0] best;
    reg [2:0] exp_cls;
    reg       got;
    reg [2:0] got_cls;
    reg [1:0] nw;
    integer   kind;

    always @(posedge clk) begin
        if (dv[0]) begin got = 1; got_cls = dc[0]; decs = decs + 1; end
        for (k = 1; k < NB; k = k + 1)
            if (dv[k]) begin
                extra = extra + 1;
                if (extra < 5) $display("ERROR dec_valid from board %0d", k);
            end
    end

    task cfg(input [11:0] a, input [31:0] d);
        begin
            @(negedge clk); cfg_addr = a; cfg_wdata = d; cfg_we = 1;
            @(negedge clk); cfg_we = 0;
        end
    endtask

    // random partial of the chosen kind; keeps |sum over NB boards| < 2^31
    function signed [31:0] pick(input integer kind_i);
        reg signed [31:0] r;
        begin
            r = $random(seed);
            case (kind_i)
                0: pick = r >>> 3;                          // large, any sign (|r| < 2^28)
                1: pick = r % 1000;                         // small, any sign
                2: pick = -((r & 32'h0FFFFFFF) + 1);        // strictly negative
                default: pick = r % 50;                     // narrow range -> many ties
            endcase
        end
    endfunction

    task one_stride;
        integer longest;
        begin
            // values
            kind = $unsigned($random(seed)) % 5;
            for (k = 0; k < NB; k = k + 1)
                for (w = 0; w < 3; w = w + 1)
                    p[k][w] = pick(kind == 4 ? 3 : kind);
            if (kind == 4 && NB > 1) begin                   // force an exact tie of totals
                p[0][1] = p[0][0];
                for (k = 1; k < NB; k = k + 1) p[k][1] = p[k][0];
            end
            for (w = 0; w < 3; w = w + 1) begin
                tot[w] = 0;
                for (k = 0; k < NB; k = k + 1) tot[w] = tot[w] + p[k][w];
            end
            best = 0; exp_cls = 0;
            for (w = 0; w < nw; w = w + 1)
                if (tot[w] > best) begin best = tot[w]; exp_cls = w + 1; end

            // each board finishes at its own time
            longest = 0;
            for (k = 0; k < NB; k = k + 1) begin
                dly[k] = $unsigned($random(seed)) % 300;
                if (dly[k] > longest) longest = dly[k];
            end
            got = 0;
            for (t = 0; t <= longest + 4; t = t + 1) begin
                @(negedge clk);
                for (k = 0; k < NB; k = k + 1) begin
                    sv[k] = 0; done[k] = 0;
                    if (t >= dly[k] && t < dly[k] + 3) begin
                        sv[k] = 1; scls[k] = t - dly[k]; sc[k] = p[k][t - dly[k]];
                    end
                    if (t == dly[k] + 3) done[k] = 1;
                end
            end
            @(negedge clk);
            for (k = 0; k < NB; k = k + 1) begin sv[k] = 0; done[k] = 0; end

            // wait for the decision
            for (t = 0; t < 200 && !got; t = t + 1) @(negedge clk);
            if (!got) begin
                errors = errors + 1;
                if (errors < 10) $display("ERROR stride %0d: no decision", n);
            end else if (got_cls !== exp_cls) begin
                errors = errors + 1;
                if (errors < 10)
                    $display("ERROR stride %0d nwords %0d: got %0d expected %0d (totals %0d %0d %0d)",
                             n, nw, got_cls, exp_cls, tot[0], tot[1], tot[2]);
            end
            repeat (5) @(negedge clk);
            if (got && decs != n + 1) begin            // a second decision for this stride?
                errors = errors + 1;
                if (errors < 10) $display("ERROR stride %0d: %0d decisions so far", n, decs);
            end
        end
    endtask

    initial begin
        for (k = 0; k < NB; k = k + 1) begin sv[k] = 0; done[k] = 0; scls[k] = 0; sc[k] = 0; end
        repeat (3) @(negedge clk);
        rst_n = 1;

        // ids and the common enable
        for (k = 0; k < NB; k = k + 1)
            if (my_id[k] !== k) begin
                errors = errors + 1; $display("ERROR board %0d has id %0d", k, my_id[k]);
            end
        cfg(12'h7F1, 1); core_en = 1;
        @(negedge clk); @(negedge clk);
        if (run_seen !== {NB{1'b1}}) begin
            errors = errors + 1; $display("ERROR run did not reach every board: %b", run_seen);
        end

        n = 0;
        for (s = 0; s < 3; s = s + 1) begin
            nw = 3 - s;                                       // 3, 2, 1 words
            cfg(12'h7F2, nw);
            repeat (STRIDES / 3) begin
                one_stride;
                n = n + 1;
            end
        end

        if (errors == 0 && extra == 0 && decs == n)
            $display("UNIT PASS  boards %0d  strides %0d  decisions %0d", NB, n, decs);
        else
            $display("UNIT FAIL  boards %0d  strides %0d  decisions %0d  errors %0d  extra %0d",
                     NB, n, decs, errors, extra);
        $finish;
    end
endmodule
