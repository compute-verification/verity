// SI baseline MACs. Combinational: y is the accumulator after one MAC step; the
// accumulator register is priced separately from its width (synth.py).

// A: 4-wide s8 x s8 dot product into a wrapping s32 accumulator (IMMA's lane op).
module mac_s8x4_s32 (
    input  signed [7:0]  a0, a1, a2, a3,
    input  signed [7:0]  b0, b1, b2, b3,
    input         [31:0] acc,
    output        [31:0] y
);
    wire signed [31:0] d = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    assign y = acc + d;
endmodule

// B: FP8 E4M3 x E4M3, exact product, added to an FP32 accumulator with one
// round-to-nearest-even (an FMA with FP8 inputs). Subnormals in and out, overflow
// to infinity, any NaN in gives 0x7FC00000. E4M3 has no infinity; S.1111.111 is NaN.
module mac_fp8_fp32 (
    input  [7:0]  a,
    input  [7:0]  b,
    input  [31:0] acc,
    output [31:0] y
);
    // E4M3 decode: value = M * 2^(E - 10), M = {implicit, mantissa}, E = max(exp, 1).
    wire        sa = a[7], sb = b[7];
    wire [3:0]  ea = a[6:3], eb = b[6:3];
    wire [3:0]  ma = {ea != 4'd0, a[2:0]};
    wire [3:0]  mb = {eb != 4'd0, b[2:0]};
    wire [3:0]  xa = (ea == 4'd0) ? 4'd1 : ea;
    wire [3:0]  xb = (eb == 4'd0) ? 4'd1 : eb;
    wire        nan_a = (a[6:0] == 7'h7f);
    wire        nan_b = (b[6:0] == 7'h7f);

    // Product: Mp * 2^(xa + xb - 20), Mp <= 225.
    wire        sp = sa ^ sb;
    wire [7:0]  mp = ma * mb;
    reg  [2:0]  lzp;
    always @* begin
        casez (mp)
            8'b1???????: lzp = 3'd0;
            8'b01??????: lzp = 3'd1;
            8'b001?????: lzp = 3'd2;
            8'b0001????: lzp = 3'd3;
            8'b00001???: lzp = 3'd4;
            8'b000001??: lzp = 3'd5;
            8'b0000001?: lzp = 3'd6;
            default:     lzp = 3'd7;
        endcase
    end
    wire        pz = (mp == 8'd0);
    // As an FP32-like operand: 24-bit significand with bit 23 set, biased exponent.
    // value = M24 * 2^(e - 150); Mp * 2^(xa+xb-20) = (Mp << (16+lz)) * 2^(xa+xb-36-lz)
    // so e = xa + xb + 114 - lz, in [109, 144].
    wire [23:0] pm = pz ? 24'd0 : ({mp, 16'd0} << lzp);
    wire [8:0]  pe = pz ? 9'd1 : ({5'd0, xa} + {5'd0, xb} + 9'd114 - {6'd0, lzp});

    // FP32 decode, subnormal as exponent 1 with implicit 0.
    wire        sc = acc[31];
    wire [7:0]  ec = acc[30:23];
    wire [22:0] fc = acc[22:0];
    wire        c_inf = (ec == 8'hff) && (fc == 23'd0);
    wire        c_nan = (ec == 8'hff) && (fc != 23'd0);
    wire [23:0] cm = {ec != 8'd0, fc};
    wire [8:0]  ce = (ec == 8'd0) ? 9'd1 : {1'b0, ec};

    // Order by magnitude: x is the larger.
    wire        c_big = (ce > pe) || ((ce == pe) && (cm >= pm));
    wire        sx = c_big ? sc : sp;
    wire        sy = c_big ? sp : sc;
    wire [23:0] mx = c_big ? cm : pm;
    wire [23:0] my = c_big ? pm : cm;
    wire [8:0]  ex = c_big ? ce : pe;
    wire [8:0]  ey = c_big ? pe : ce;
    wire [8:0]  d  = ex - ey;

    // Align with guard, round and a jammed sticky bit.
    wire [26:0] yx  = {my, 3'b000};
    wire [26:0] ysh = (d >= 9'd27) ? 27'd0 : (yx >> d);
    wire [26:0] ylost_mask = (d >= 9'd27) ? {27{1'b1}} : ~({27{1'b1}} << d);
    wire        ystk = |(yx & ylost_mask);
    wire [26:0] yal = {ysh[26:1], ysh[0] | ystk};
    wire [26:0] xal = {mx, 3'b000};
    wire        sub = sx ^ sy;
    wire [27:0] s = sub ? ({1'b0, xal} - {1'b0, yal}) : ({1'b0, xal} + {1'b0, yal});

    // Normalize.
    reg  [4:0]  lz;
    integer     i;
    always @* begin
        lz = 5'd27;
        for (i = 0; i < 27; i = i + 1)
            if (s[i]) lz = 5'd26 - i[4:0];
    end
    wire [8:0]  lzx = {4'd0, lz};
    wire [8:0]  room = ex - 9'd1;
    wire [8:0]  sh = (lzx < room) ? lzx : room;
    wire [26:0] sl = s[26:0] << sh;
    wire        carry = s[27];
    wire [23:0] mant = carry ? s[27:4] : sl[26:3];
    wire        g    = carry ? s[3] : sl[2];
    wire        stk  = carry ? (s[2] | s[1] | s[0]) : (sl[1] | sl[0]);
    wire [8:0]  er   = carry ? (ex + 9'd1) : (ex - sh);

    // Round to nearest even.
    wire        up = g & (stk | mant[0]);
    wire [24:0] mr = {1'b0, mant} + {24'd0, up};
    wire [23:0] mf = mr[24] ? mr[24:1] : mr[23:0];
    wire [8:0]  ef = mr[24] ? (er + 9'd1) : er;
    wire        zero = (s == 28'd0);
    wire        ovf = (ef >= 9'd255);
    wire [7:0]  efield = mf[23] ? ef[7:0] : 8'd0;
    wire        szero = sub ? 1'b0 : sx;

    assign y = (c_nan | nan_a | nan_b) ? 32'h7fc00000 :
               c_inf                   ? acc :
               zero                    ? {szero, 31'd0} :
               ovf                     ? {sx, 8'hff, 23'd0} :
                                         {sx, efield, mf[22:0]};
endmodule
