// The plane-limited keyed absorb: four Keccak-f[1600] rounds in one clock, each absorbing 8 accumulator words (256
// bits) into lanes x = 0..3 of plane y = 0 only, so no two absorbed bits share a column. 32 words a clock, as FT.
// keccak1600_rnd is SI-FOLD's round (rtl.py), emitted beside this file by gov_synth.py --fold.
module ft_plane4(input [1599:0] st, input [1023:0] m, input [255:0] rc, output [1599:0] y);
  wire [1599:0] s1, s2, s3;
  keccak1600_rnd r0(.st(st), .m({768'b0, m[255:0]}), .rc(rc[63:0]), .y(s1));
  keccak1600_rnd r1(.st(s1), .m({768'b0, m[511:256]}), .rc(rc[127:64]), .y(s2));
  keccak1600_rnd r2(.st(s2), .m({768'b0, m[767:512]}), .rc(rc[191:128]), .y(s3));
  keccak1600_rnd r3(.st(s3), .m({768'b0, m[1023:768]}), .rc(rc[255:192]), .y(y));
endmodule
