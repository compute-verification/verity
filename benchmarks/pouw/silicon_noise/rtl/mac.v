// Baselines: one 4-wide dot-product MAC step, registered accumulator.

// s8 x s8 -> s32: acc <= acc + sum_i a_i * b_i.
module mac_s8_dp4 (
    input clk,
    input [31:0] a, b,
    output reg [31:0] acc
);
    wire signed [7:0] a0 = a[7:0], a1 = a[15:8], a2 = a[23:16], a3 = a[31:24];
    wire signed [7:0] b0 = b[7:0], b1 = b[15:8], b2 = b[23:16], b3 = b[31:24];
    wire signed [17:0] s = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    always @(posedge clk) acc <= acc + {{14{s[17]}}, s};
endmodule

// FP8 E4M3 x E4M3 -> FP32 accumulate, one rounding a step (area model, not NVIDIA's exact semantics).
// Products are exact (4-bit significands, 8-bit product); the four products and the accumulator are aligned to the
// largest exponent in a 48-bit window, summed, normalized and rounded to nearest even. NaN/Inf handling omitted.
module mac_fp8_dp4 (
    input clk,
    input [31:0] a, b,
    output reg [31:0] acc
);
    function [3:0] sig4;  // 1.mmm, subnormal 0.mmm
        input [7:0] x;
        sig4 = {(x[6:3] != 0), x[2:0]};
    endfunction
    function [3:0] exp4;
        input [7:0] x;
        exp4 = (x[6:3] == 0) ? 4'd1 : x[6:3];
    endfunction
    wire [7:0] pa [0:3];
    wire [7:0] pb [0:3];
    assign pa[0] = a[7:0];  assign pa[1] = a[15:8];  assign pa[2] = a[23:16]; assign pa[3] = a[31:24];
    assign pb[0] = b[7:0];  assign pb[1] = b[15:8];  assign pb[2] = b[23:16]; assign pb[3] = b[31:24];
    wire [7:0] pm [0:3];   // product significand, 2.6 fixed point
    wire [8:0] pe [0:3];   // product exponent, biased to FP32 (e_a + e_b - 14 + 127)
    wire ps [0:3];
    genvar i;
    generate for (i = 0; i < 4; i = i + 1) begin : prod
        assign pm[i] = sig4(pa[i]) * sig4(pb[i]);
        assign pe[i] = exp4(pa[i]) + exp4(pb[i]) + 9'd113;
        assign ps[i] = pa[i][7] ^ pb[i][7];
    end endgenerate
    wire [7:0] ae = acc[30:23];
    wire [23:0] am = {(ae != 0), acc[22:0]};
    wire [8:0] e01 = (pe[0] > pe[1]) ? pe[0] : pe[1];
    wire [8:0] e23 = (pe[2] > pe[3]) ? pe[2] : pe[3];
    wire [8:0] ep = (e01 > e23) ? e01 : e23;
    wire [8:0] emax = (ep > {1'b0, ae}) ? ep : {1'b0, ae};
    // align into a 48-bit window whose bit 46 is weight 2^(emax-127+1)
    function signed [49:0] align;
        input [23:0] m; input [8:0] e; input [8:0] em; input s; input [5:0] pos;
        reg [8:0] d; reg [47:0] w;
        begin
            d = em - e;
            w = ({24'd0, m} << pos) >> ((d > 47) ? 47 : d);
            align = s ? -$signed({2'b00, w}) : $signed({2'b00, w});
        end
    endfunction
    wire signed [49:0] t0 = align({16'd0, pm[0]}, pe[0], emax, ps[0], 6'd40);
    wire signed [49:0] t1 = align({16'd0, pm[1]}, pe[1], emax, ps[1], 6'd40);
    wire signed [49:0] t2 = align({16'd0, pm[2]}, pe[2], emax, ps[2], 6'd40);
    wire signed [49:0] t3 = align({16'd0, pm[3]}, pe[3], emax, ps[3], 6'd40);
    wire signed [49:0] ta = align(am, {1'b0, ae}, emax, acc[31], 6'd23);
    wire signed [49:0] sum = t0 + t1 + t2 + t3 + ta;
    wire sgn = sum[49];
    wire [48:0] mag = sgn ? -sum[48:0] : sum[48:0];
    // leading-zero count
    reg [5:0] lz; integer k;
    always @(*) begin
        lz = 6'd49;
        for (k = 0; k < 49; k = k + 1) if (mag[k]) lz = 6'd48 - k;
    end
    wire [48:0] norm = mag << lz;
    wire [23:0] keep = norm[48:25];
    wire g = norm[24], st = |norm[23:0];
    wire up = g & (st | keep[0]);
    wire [24:0] rnd = keep + up;
    wire [9:0] eo = emax + 10'd2 - lz + rnd[24];
    wire [22:0] mo = rnd[24] ? rnd[23:1] : rnd[22:0];
    always @(posedge clk) acc <= (mag == 0) ? 32'd0 : {sgn, eo[7:0], mo};
endmodule
