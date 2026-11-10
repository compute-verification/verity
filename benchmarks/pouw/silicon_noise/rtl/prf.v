// Candidate noise generators, each one cycle of a pipelined or unrolled core. synth.py divides by the bits a cycle
// yields to price a noise bit.

// Trivium, 64 state updates a cycle: 64 keystream bits a cycle, 288-bit state.
module trivium_x64 (
    input clk,
    input load,
    input [287:0] init,
    output [63:0] z
);
    reg [287:0] s;
    reg [287:0] n;
    reg [63:0] zz;
    integer k;
    reg t1, t2, t3;
    always @(*) begin
        n = s;
        for (k = 0; k < 64; k = k + 1) begin
            t1 = n[65] ^ n[92];
            t2 = n[161] ^ n[176];
            t3 = n[242] ^ n[287];
            zz[k] = t1 ^ t2 ^ t3;
            t1 = t1 ^ (n[90] & n[91]) ^ n[170];
            t2 = t2 ^ (n[174] & n[175]) ^ n[263];
            t3 = t3 ^ (n[285] & n[286]) ^ n[68];
            n = {n[286:177], t2, n[175:93], t1, n[91:0], t3};
        end
    end
    assign z = zz;
    always @(posedge clk) s <= load ? init : n;
endmodule

// ChaCha quarter-round (combinational); an unrolled ChaCha-R block is 4R of these plus 16 adds.
module chacha_qr (
    input [31:0] a, b, c, d,
    output [31:0] a2, b2, c2, d2
);
    wire [31:0] a1 = a + b;
    wire [31:0] d1x = d ^ a1;
    wire [31:0] d1 = {d1x[15:0], d1x[31:16]};
    wire [31:0] c1 = c + d1;
    wire [31:0] b1x = b ^ c1;
    wire [31:0] b1 = {b1x[19:0], b1x[31:20]};
    assign a2 = a1 + b1;
    wire [31:0] d2x = d1 ^ a2;
    assign d2 = {d2x[23:0], d2x[31:24]};
    assign c2 = c1 + d2;
    wire [31:0] b2x = b1 ^ c2;
    assign b2 = {b2x[24:0], b2x[31:25]};
endmodule

// One registered stage of a pipelined ChaCha: a column round (four quarter-rounds) on a 512-bit state.
module chacha_round (
    input clk,
    input [511:0] x,
    output reg [511:0] y
);
    wire [31:0] w [0:15];
    wire [31:0] o [0:15];
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : un
        assign w[i] = x[32 * i +: 32];
    end endgenerate
    chacha_qr q0 (w[0], w[4], w[8], w[12], o[0], o[4], o[8], o[12]);
    chacha_qr q1 (w[1], w[5], w[9], w[13], o[1], o[5], o[9], o[13]);
    chacha_qr q2 (w[2], w[6], w[10], w[14], o[2], o[6], o[10], o[14]);
    chacha_qr q3 (w[3], w[7], w[11], w[15], o[3], o[7], o[11], o[15]);
    always @(posedge clk) y <= {o[15], o[14], o[13], o[12], o[11], o[10], o[9], o[8],
                                o[7], o[6], o[5], o[4], o[3], o[2], o[1], o[0]};
endmodule

// The 16 feed-forward adds of a ChaCha block.
module chacha_ff (
    input [511:0] x, k,
    output [511:0] y
);
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : ff
        assign y[32 * i +: 32] = x[32 * i +: 32] + k[32 * i +: 32];
    end endgenerate
endmodule

// One registered stage of a pipelined SHA-512: a round and a schedule step, state and 16-word window registered.
module sha512_round (
    input clk,
    input [511:0] st,
    input [1023:0] wv,
    input [63:0] kt,
    output reg [511:0] st2,
    output reg [1023:0] wv2
);
    function [63:0] rotr; input [63:0] x; input [5:0] n; rotr = (x >> n) | (x << (64 - n)); endfunction
    wire [63:0] a = st[511:448], b = st[447:384], c = st[383:320], d = st[319:256];
    wire [63:0] e = st[255:192], f = st[191:128], g = st[127:64], h = st[63:0];
    wire [63:0] w0 = wv[63:0];
    wire [63:0] s1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
    wire [63:0] ch = (e & f) ^ (~e & g);
    wire [63:0] t1 = h + s1 + ch + kt + w0;
    wire [63:0] s0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
    wire [63:0] mj = (a & b) ^ (a & c) ^ (b & c);
    wire [63:0] t2 = s0 + mj;
    wire [63:0] w1 = wv[127:64], w9 = wv[639:576], w14 = wv[959:896];
    wire [63:0] sg0 = rotr(w1, 1) ^ rotr(w1, 8) ^ (w1 >> 7);
    wire [63:0] sg1 = rotr(w14, 19) ^ rotr(w14, 61) ^ (w14 >> 6);
    wire [63:0] w16 = sg1 + w9 + sg0 + w0;
    always @(posedge clk) begin
        st2 <= {t1 + t2, a, b, c, d + t1, e, f, g};
        wv2 <= {w16, wv[1023:64]};
    end
endmodule

// One registered round of Keccak-f[1600] (theta, rho, pi, chi, iota).
module keccak_round (
    input clk,
    input [1599:0] s,
    input [63:0] rc,
    output reg [1599:0] o
);
    function [63:0] rotl; input [63:0] x; input [5:0] n; rotl = (x << n) | (x >> (64 - n)); endfunction
    wire [63:0] A [0:24];
    genvar i;
    generate for (i = 0; i < 25; i = i + 1) begin : un
        assign A[i] = s[64 * i +: 64];
    end endgenerate
    wire [63:0] C [0:4];
    wire [63:0] D [0:4];
    generate for (i = 0; i < 5; i = i + 1) begin : th
        assign C[i] = A[i] ^ A[i + 5] ^ A[i + 10] ^ A[i + 15] ^ A[i + 20];
    end endgenerate
    generate for (i = 0; i < 5; i = i + 1) begin : th2
        assign D[i] = C[(i + 4) % 5] ^ rotl(C[(i + 1) % 5], 1);
    end endgenerate
    localparam [8 * 25 - 1:0] RHO = {8'd14, 8'd56, 8'd61, 8'd2, 8'd18, 8'd8, 8'd21, 8'd15, 8'd45, 8'd41, 8'd39, 8'd25,
                                     8'd43, 8'd10, 8'd3, 8'd20, 8'd55, 8'd6, 8'd44, 8'd28, 8'd27, 8'd36, 8'd62, 8'd1, 8'd0};
    wire [63:0] B [0:24];
    generate for (i = 0; i < 25; i = i + 1) begin : rp
        // lane (x, y) = i -> (y, 2x + 3y)
        assign B[(i / 5) + 5 * ((2 * (i % 5) + 3 * (i / 5)) % 5)] = rotl(A[i] ^ D[i % 5], RHO[8 * i +: 6]);
    end endgenerate
    wire [63:0] X [0:24];
    generate for (i = 0; i < 25; i = i + 1) begin : chi
        assign X[i] = B[i] ^ (~B[5 * (i / 5) + (i % 5 + 1) % 5] & B[5 * (i / 5) + (i % 5 + 2) % 5]);
    end endgenerate
    integer j;
    reg [1599:0] t;
    always @(*) begin
        for (j = 0; j < 25; j = j + 1) t[64 * j +: 64] = X[j];
        t[63:0] = X[0] ^ rc;
    end
    always @(posedge clk) o <= t;
endmodule
