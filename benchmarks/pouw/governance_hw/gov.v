// Blocks the always-on logging design adds beside FT, each one combinational step (state in, state out), as in
// SI-ALG's and OCC-SILICON's flows: registers are priced separately at a DFF a bit.

// One SMSP's issue-class counters at dispatch. Each clock the scheduler issues at most one warp instruction; its
// pipe select (cls, 0..11) is already decoded to route it. The class counter gains the active-lane count, so the
// counters are lane operations, not warp issues. Counters 12 and 13 count register-file write enables and
// transcript absorbs, which the seal carries side by side.
module cnt_step(input v, input [3:0] cls, input [31:0] am, input we, input ab, input [671:0] c, output [671:0] y);
  reg [5:0] pc;
  integer i;
  always @* begin
    pc = 6'd0;
    for (i = 0; i < 32; i = i + 1) pc = pc + am[i];
  end
  genvar k;
  generate
    for (k = 0; k < 12; k = k + 1) begin : lanes
      wire hit = v & (cls == k);
      assign y[48 * k +: 48] = c[48 * k +: 48] + (hit ? {42'b0, pc} : 48'b0);
    end
  endgenerate
  assign y[48 * 12 +: 48] = c[48 * 12 +: 48] + {47'b0, we};
  assign y[48 * 13 +: 48] = c[48 * 13 +: 48] + {47'b0, ab};
endmodule

// GF(2^64) multiplication modulo x^64 + x^4 + x^3 + x + 1: the step of an order-independent polynomial hash of
// vector-pipe results, sum over results of v * K_w^seq, priced against FT's in-order Keccak round.
module gf64_mul(input [63:0] a, input [63:0] b, output reg [63:0] y);
  reg [126:0] p;
  integer i, j;
  always @* begin
    p = 127'd0;
    for (i = 0; i < 64; i = i + 1)
      if (b[i]) p = p ^ ({63'd0, a} << i);
    for (j = 126; j >= 64; j = j - 1)
      if (p[j]) p = p ^ (127'b11011 << (j - 64));
    y = p[63:0];
  end
endmodule
