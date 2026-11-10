// SI-ALG: the arithmetic and format alternatives, each against baseline.v's MACs.
// Same convention: combinational, y is the accumulator after one MAC step.

// ---- Wide integer accumulators: the s8x4 dot product into s48 or s64. ----
module mac_s8x4_s48 (
    input  signed [7:0]  a0, a1, a2, a3, b0, b1, b2, b3,
    input         [47:0] acc,
    output        [47:0] y
);
    wire signed [47:0] d = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    assign y = acc + d;
endmodule

module mac_s8x4_s64 (
    input  signed [7:0]  a0, a1, a2, a3, b0, b1, b2, b3,
    input         [63:0] acc,
    output        [63:0] y
);
    wire signed [63:0] d = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    assign y = acc + d;
endmodule

// ---- s8x4 dot product accumulated in F_p (int8 embeds exactly while p > 2 |bound|). ----
// acc in [0, p); |d| <= 65536 < p, so one correction either way.
module mac_s8x4_modp #(parameter W = 31, parameter [64:0] P = 65'h7fffffff) (
    input  signed [7:0]    a0, a1, a2, a3, b0, b1, b2, b3,
    input         [W-1:0]  acc,
    output        [W-1:0]  y
);
    wire signed [17:0]  d = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    wire signed [W+1:0] t = $signed({2'b00, acc}) + d;
    wire signed [W+1:0] tp = t + $signed({1'b0, P[W:0]});
    wire signed [W+1:0] tm = t - $signed({1'b0, P[W:0]});
    assign y = t[W+1] ? tp[W-1:0] : ((t >= $signed({1'b0, P[W:0]})) ? tm[W-1:0] : t[W-1:0]);
endmodule

module mac_s8x4_m31 (input signed [7:0] a0, a1, a2, a3, b0, b1, b2, b3,
                     input [30:0] acc, output [30:0] y);
    mac_s8x4_modp #(.W(31), .P(64'h7fffffff)) u (a0, a1, a2, a3, b0, b1, b2, b3, acc, y);
endmodule

module mac_s8x4_bb (input signed [7:0] a0, a1, a2, a3, b0, b1, b2, b3,
                    input [30:0] acc, output [30:0] y);
    mac_s8x4_modp #(.W(31), .P(64'h78000001)) u (a0, a1, a2, a3, b0, b1, b2, b3, acc, y);
endmodule

module mac_s8x4_m61 (input signed [7:0] a0, a1, a2, a3, b0, b1, b2, b3,
                     input [60:0] acc, output [60:0] y);
    mac_s8x4_modp #(.W(61), .P(64'h1fffffffffffffff)) u (a0, a1, a2, a3, b0, b1, b2, b3, acc, y);
endmodule

module mac_s8x4_gl (input signed [7:0] a0, a1, a2, a3, b0, b1, b2, b3,
                    input [63:0] acc, output [63:0] y);
    mac_s8x4_modp #(.W(64), .P(64'hffffffff00000001)) u (a0, a1, a2, a3, b0, b1, b2, b3, acc, y);
endmodule

// ---- Full-width field MACs: y = acc + a*b in F_p, every operand in [0, p). ----
// Mersenne-31: x = hi*2^31 + lo == hi + lo.
module mac_m31_full (input [30:0] a, b, acc, output [30:0] y);
    wire [61:0] x  = a * b;
    wire [32:0] s  = {2'b0, x[30:0]} + {2'b0, x[61:31]} + {2'b0, acc};
    wire [31:0] s1 = {1'b0, s[30:0]} + {29'd0, s[32:31]};
    wire [31:0] s2 = s1 - 32'h7fffffff;
    assign y = (s1 >= 32'h7fffffff) ? s2[30:0] : s1[30:0];
endmodule

// BabyBear p = 15*2^27 + 1, Montgomery form with R = 2^32: y = acc + a*b*R^-1 mod p.
module mac_bb_full (input [30:0] a, b, acc, output [30:0] y);
    localparam [31:0] P    = 32'h78000001;
    localparam [31:0] NINV = 32'h77ffffff;   // -p^-1 mod 2^32
    wire [63:0] x = a * b + {1'b0, acc, 32'd0};
    wire [31:0] m = x[31:0] * NINV;
    wire [64:0] t = {1'b0, x} + m * P;
    wire [32:0] u = t[64:32];                              // < (p/R + 2) p < 2.47 p
    wire [32:0] u1 = (u >= {1'b0, P}) ? (u - {1'b0, P}) : u;
    wire [32:0] u2 = (u1 >= {1'b0, P}) ? (u1 - {1'b0, P}) : u1;
    assign y = u2[30:0];
endmodule

// Mersenne-61.
module mac_m61_full (input [60:0] a, b, acc, output [60:0] y);
    localparam [62:0] P = 63'h1fffffffffffffff;
    wire [121:0] x  = a * b;
    wire [62:0]  s  = {2'b0, x[60:0]} + {2'b0, x[121:61]} + {2'b0, acc};
    wire [62:0]  s1 = {2'b0, s[60:0]} + {61'd0, s[62:61]};
    wire [62:0]  s2 = s1 - P;
    assign y = (s1 >= P) ? s2[60:0] : s1[60:0];
endmodule

// Goldilocks p = 2^64 - 2^32 + 1: 2^64 == 2^32 - 1, 2^96 == -1.
module mac_gl_full (input [63:0] a, b, acc, output [63:0] y);
    localparam [66:0] P   = 67'h0ffffffff00000001;
    localparam [66:0] EPS = 67'h000000000ffffffff;
    wire [127:0] x = a * b;
    // r = lo + x2*EPS - x3 + acc + p, in (0, 4*2^64)
    wire [66:0] r  = {3'b0, x[63:0]} + {35'd0, x[95:64]} * EPS - {35'd0, x[127:96]} + {3'b0, acc} + P;
    wire [66:0] r2 = {3'b0, r[63:0]} + {64'd0, r[66:64]} * EPS;
    wire [66:0] r3 = (r2 >= P) ? (r2 - P) : r2;
    wire [66:0] r4 = (r3 >= P) ? (r3 - P) : r3;
    assign y = r4[63:0];
endmodule

// ---- Kulisch (exact fixed-point) accumulation of FP8 E4M3 products. ----
// Every E4M3 product is an integer multiple of 2^-18 below 2^18, so a 64-bit
// two's-complement accumulator at LSB 2^-18 holds any sum of up to 2^27 products exactly.
module e4m3_prod_fx (input [7:0] a, b, output signed [37:0] p, output nan);
    wire [3:0] ea = a[6:3], eb = b[6:3];
    wire [3:0] ma = {ea != 4'd0, a[2:0]};
    wire [3:0] mb = {eb != 4'd0, b[2:0]};
    wire [3:0] xa = (ea == 4'd0) ? 4'd1 : ea;
    wire [3:0] xb = (eb == 4'd0) ? 4'd1 : eb;
    wire [7:0] mp = ma * mb;
    wire [4:0] sh = {1'b0, xa} + {1'b0, xb} - 5'd2;          // product exponent + 18, in [0, 28]
    wire [36:0] mag = {29'd0, mp} << sh;
    assign p   = (a[7] ^ b[7]) ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
    assign nan = (a[6:0] == 7'h7f) || (b[6:0] == 7'h7f);
endmodule

module mac_fp8_k64 (input [7:0] a, b, input [63:0] acc, output [63:0] y, output nan);
    wire signed [37:0] p;
    e4m3_prod_fx u (a, b, p, nan);
    assign y = acc + {{26{p[37]}}, p};
endmodule

module mac_fp8x4_k64 (input [7:0] a0, a1, a2, a3, b0, b1, b2, b3,
                      input [63:0] acc, output [63:0] y, output nan);
    wire signed [37:0] p0, p1, p2, p3;
    wire n0, n1, n2, n3;
    e4m3_prod_fx u0 (a0, b0, p0, n0);
    e4m3_prod_fx u1 (a1, b1, p1, n1);
    e4m3_prod_fx u2 (a2, b2, p2, n2);
    e4m3_prod_fx u3 (a3, b3, p3, n3);
    wire signed [63:0] d = p0 + p1 + p2 + p3;
    assign y = acc + d;
    assign nan = n0 | n1 | n2 | n3;
endmodule

// ---- NVFP4: 16 E2M1 x E2M1 products, an exact block sum, times the two E4M3 block
// scales, into a 64-bit Kulisch accumulator at LSB 2^-20 (the global FP32 scale is
// applied once per output, outside the MAC). ----
module e2m1_prod (input [3:0] a, b, output signed [8:0] p);
    // E2M1 value = M * 2^(E-1): exp 0 -> M = m, E = 0; else M = 2 + m, E = exp - 1.
    wire [1:0] ma = (a[2:1] == 2'd0) ? {1'b0, a[0]} : {1'b1, a[0]};
    wire [1:0] mb = (b[2:1] == 2'd0) ? {1'b0, b[0]} : {1'b1, b[0]};
    wire [1:0] xa = (a[2:1] == 2'd0) ? 2'd0 : (a[2:1] - 2'd1);
    wire [1:0] xb = (b[2:1] == 2'd0) ? 2'd0 : (b[2:1] - 2'd1);
    wire [3:0] mp = ma * mb;
    wire [7:0] mag = {4'd0, mp} << ({1'b0, xa} + {1'b0, xb});   // product * 4, an integer <= 144
    assign p = (a[3] ^ b[3]) ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
endmodule

module mac_nvfp4_b16_k64 (input [63:0] a, b, input [7:0] sa, sb,
                          input [63:0] acc, output [63:0] y, output nan);
    wire signed [8:0] p [0:15];
    genvar k;
    generate for (k = 0; k < 16; k = k + 1) begin : g
        e2m1_prod u (a[4*k +: 4], b[4*k +: 4], p[k]);
    end endgenerate
    wire signed [13:0] blk = p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7]
                           + p[8] + p[9] + p[10] + p[11] + p[12] + p[13] + p[14] + p[15];
    // Block-scale product, as e4m3_prod_fx: M * 2^(sh - 18), sh in [0, 28].
    wire [3:0] ea = sa[6:3], eb = sb[6:3];
    wire [3:0] ma = {ea != 4'd0, sa[2:0]};
    wire [3:0] mb = {eb != 4'd0, sb[2:0]};
    wire [3:0] xa = (ea == 4'd0) ? 4'd1 : ea;
    wire [3:0] xb = (eb == 4'd0) ? 4'd1 : eb;
    wire [7:0] ms = ma * mb;
    wire [4:0] sh = {1'b0, xa} + {1'b0, xb} - 5'd2;
    wire signed [22:0] v = blk * $signed({1'b0, ms});      // block * 4 * scale mantissa product
    wire signed [63:0] vx = {{41{v[22]}}, v};
    wire signed [63:0] d = (sa[7] ^ sb[7]) ? -(vx <<< sh) : (vx <<< sh);
    assign y = acc + d;
    assign nan = (sa[6:0] == 7'h7f) || (sb[6:0] == 7'h7f);
endmodule

// ---- Binary fields, carry-less (GF(2)[x] / f): y = acc ^ a*b. ----
function automatic [254:0] clmul128;
    input [127:0] a, b;
    input integer n;
    integer i;
    begin
        clmul128 = 0;
        for (i = 0; i < 128; i = i + 1)
            if (i < n && b[i]) clmul128 = clmul128 ^ ({127'd0, a} << i);
    end
endfunction

// 4-wide GF(2^8) dot product, AES polynomial 0x11b, one reduction for the sum.
module mac_gf8x4 (input [7:0] a0, a1, a2, a3, b0, b1, b2, b3, input [7:0] acc, output [7:0] y);
    wire [254:0] q0 = clmul128({120'd0, a0}, {120'd0, b0}, 8);
    wire [254:0] q1 = clmul128({120'd0, a1}, {120'd0, b1}, 8);
    wire [254:0] q2 = clmul128({120'd0, a2}, {120'd0, b2}, 8);
    wire [254:0] q3 = clmul128({120'd0, a3}, {120'd0, b3}, 8);
    wire [14:0] q = q0[14:0] ^ q1[14:0] ^ q2[14:0] ^ q3[14:0];
    reg  [14:0] r;
    integer i;
    always @* begin
        r = q;
        for (i = 14; i >= 8; i = i - 1)
            if (r[i]) r = r ^ (15'h11b << (i - 8));
    end
    assign y = acc ^ r[7:0];
endmodule

// GF(2^n) single MAC with f = x^n + low(x).
module mac_gf2n #(parameter N = 32, parameter [127:0] LOW = 128'h8d) (
    input [N-1:0] a, b, acc, output [N-1:0] y);
    wire [254:0] q = clmul128({{(128-N){1'b0}}, a}, {{(128-N){1'b0}}, b}, N);
    reg  [254:0] r;
    integer i;
    always @* begin
        r = q;
        for (i = 2 * N - 2; i >= N; i = i - 1)
            if (r[i]) r = r ^ ({127'd0, LOW} << (i - N)) ^ ({254'd0, 1'b1} << i);
    end
    assign y = acc ^ r[N-1:0];
endmodule

module mac_gf32 (input [31:0] a, b, acc, output [31:0] y);
    mac_gf2n #(.N(32), .LOW(128'h8d)) u (a, b, acc, y);          // x^32 + x^7 + x^3 + x^2 + 1
endmodule

module mac_gf64 (input [63:0] a, b, acc, output [63:0] y);
    mac_gf2n #(.N(64), .LOW(128'h1b)) u (a, b, acc, y);          // x^64 + x^4 + x^3 + x + 1
endmodule

module mac_gf128 (input [127:0] a, b, acc, output [127:0] y);
    mac_gf2n #(.N(128), .LOW(128'h87)) u (a, b, acc, y);         // x^128 + x^7 + x^2 + x + 1 (GHASH's)
endmodule

// ---- RNS: four 8-bit prime channels (251, 241, 239, 233; product 3.37e9 > 2^31),
// each a 4-wide residue dot product accumulated mod its prime. Residue conversion of
// s8 operands is outside the MAC (once per operand load). ----
module rns_ch #(parameter [7:0] M = 8'd251) (
    input [7:0] a0, a1, a2, a3, b0, b1, b2, b3, acc, output [7:0] y);
    localparam [7:0] C = 8'd0 - M;                       // 2^8 == C (mod M)
    wire [17:0] s0 = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + acc;   // < 2^18
    wire [15:0] s1 = s0[7:0] + s0[17:8] * C;            // fold 2^8 -> C
    wire [12:0] s2 = s1[7:0] + s1[15:8] * C;
    wire [10:0] s3 = s2[7:0] + s2[12:8] * C;
    wire [9:0]  s4 = s3[7:0] + s3[10:8] * C;
    wire [9:0]  s5 = s4[7:0] + s4[9:8] * C;              // < 2^8 + 3C
    wire [9:0]  s6 = (s5 >= {2'b0, M}) ? s5 - M : s5;
    wire [9:0]  s7 = (s6 >= {2'b0, M}) ? s6 - M : s6;
    assign y = s7[7:0];
endmodule

module mac_rns4_s8x4 (input [31:0] a0, a1, a2, a3, b0, b1, b2, b3, input [31:0] acc, output [31:0] y);
    rns_ch #(.M(8'd251)) c0 (a0[7:0],   a1[7:0],   a2[7:0],   a3[7:0],   b0[7:0],   b1[7:0],   b2[7:0],   b3[7:0],   acc[7:0],   y[7:0]);
    rns_ch #(.M(8'd241)) c1 (a0[15:8],  a1[15:8],  a2[15:8],  a3[15:8],  b0[15:8],  b1[15:8],  b2[15:8],  b3[15:8],  acc[15:8],  y[15:8]);
    rns_ch #(.M(8'd239)) c2 (a0[23:16], a1[23:16], a2[23:16], a3[23:16], b0[23:16], b1[23:16], b2[23:16], b3[23:16], acc[23:16], y[23:16]);
    rns_ch #(.M(8'd233)) c3 (a0[31:24], a1[31:24], a2[31:24], a3[31:24], b0[31:24], b1[31:24], b2[31:24], b3[31:24], acc[31:24], y[31:24]);
endmodule
