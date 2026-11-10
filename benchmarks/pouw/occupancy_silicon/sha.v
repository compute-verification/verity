// OCC-SILICON P-C: an iterative SHA-512 engine (one round a clock), as combinational steps priced like SI-ALG's MACs.

module sha512_round(input [63:0] a, b, c, d, e, f, g, h, input [63:0] w, input [63:0] k,
                    output [63:0] a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n);
  wire [63:0] s1 = {e[13:0], e[63:14]} ^ {e[17:0], e[63:18]} ^ {e[40:0], e[63:41]};
  wire [63:0] ch = (e & f) ^ (~e & g);
  wire [63:0] t1 = h + s1 + ch + k + w;
  wire [63:0] s0 = {a[27:0], a[63:28]} ^ {a[33:0], a[63:34]} ^ {a[38:0], a[63:39]};
  wire [63:0] mj = (a & b) ^ (a & c) ^ (b & c);
  wire [63:0] t2 = s0 + mj;
  assign a_n = t1 + t2;
  assign b_n = a; assign c_n = b; assign d_n = c;
  assign e_n = d + t1;
  assign f_n = e; assign g_n = f; assign h_n = g;
endmodule

// The message schedule as a 16-word shift register: w0 leaves, the new word enters at w15.
module sha512_sched(input [1023:0] w, output [1023:0] w_n);
  wire [63:0] w0 = w[63:0], w1 = w[127:64], w9 = w[639:576], w14 = w[959:896];
  wire [63:0] g0 = {w1[0], w1[63:1]} ^ {w1[7:0], w1[63:8]} ^ (w1 >> 7);
  wire [63:0] g1 = {w14[18:0], w14[63:19]} ^ {w14[60:0], w14[63:61]} ^ (w14 >> 6);
  assign w_n = {g1 + w9 + g0 + w0, w[1023:64]};
endmodule

module sha512_fin(input [511:0] hh, input [511:0] s, output [511:0] hh_n);
  genvar i;
  generate for (i = 0; i < 8; i = i + 1) begin : add
    assign hh_n[64*i +: 64] = hh[64*i +: 64] + s[64*i +: 64];
  end endgenerate
endmodule
