// Per-accumulator units: the hidden noise walk and the sketch registers that absorb each k-step's word.

// The hidden walk: z <= z + n for a B-bit noise word n a step (zero-extended), z W bits wide. Not readable.
module noise_walk #(parameter B = 8, parameter W = 32) (
    input clk,
    input clr,
    input [B - 1:0] n,
    output reg [W - 1:0] z
);
    always @(posedge clk) z <= clr ? {W{1'b0}} : z + {{(W - B){1'b0}}, n};
endmodule

// XOR-rotate sketch: s <= rotl(s, 1) ^ (acc + z).
module sk_xorrot (
    input clk,
    input [31:0] acc, z,
    output reg [31:0] s
);
    wire [31:0] w = acc + z;
    always @(posedge clk) s <= {s[30:0], s[31]} ^ w;
endmodule

// XOR-rotate sketch over acc ^ z (no adder; for FP accumulators, whose bit pattern is mixed with the walk).
module sk_xorrot_x (
    input clk,
    input [31:0] acc, z,
    output reg [31:0] s
);
    always @(posedge clk) s <= {s[30:0], s[31]} ^ acc ^ z;
endmodule

// CRC-32 (IEEE 802.3, reflected) of the 32-bit word acc + z into s.
module sk_crc32 (
    input clk,
    input [31:0] acc, z,
    output reg [31:0] s
);
    wire [31:0] w = acc + z;
    reg [31:0] c; integer k;
    always @(*) begin
        c = s ^ w;
        for (k = 0; k < 32; k = k + 1) c = c[0] ? (c >> 1) ^ 32'hEDB88320 : (c >> 1);
    end
    always @(posedge clk) s <= c;
endmodule

// GF(2^32) Horner sketch: s <= s * r ^ (acc + z), r a public step coin, modulus x^32 + x^7 + x^3 + x^2 + 1.
module sk_gf32 (
    input clk,
    input [31:0] acc, z, r,
    output reg [31:0] s
);
    wire [31:0] w = acc + z;
    reg [62:0] p; reg [31:0] q; integer i;
    always @(*) begin
        p = 63'd0;
        for (i = 0; i < 32; i = i + 1) if (r[i]) p = p ^ ({31'd0, s} << i);
        for (i = 62; i >= 32; i = i - 1) if (p[i]) p = p ^ (63'h8D << (i - 32));
        q = p[31:0];
    end
    always @(posedge clk) s <= q ^ w;
endmodule

// NH over a column pair: s <= s + (acc0 + z0) * (acc1 + z1) mod 2^32.
module sk_nh (
    input clk,
    input [31:0] acc0, z0, acc1, z1,
    output reg [31:0] s
);
    wire [31:0] w0 = acc0 + z0, w1 = acc1 + z1;
    wire [31:0] p = w0 * w1;
    always @(posedge clk) s <= s + p;
endmodule

// Linear sketch mod 2^32: s <= s + r * (acc + z), r a public step coin.
module sk_lin (
    input clk,
    input [31:0] acc, z, r,
    output reg [31:0] s
);
    wire [31:0] w = acc + z;
    always @(posedge clk) s <= s + r * w;
endmodule
