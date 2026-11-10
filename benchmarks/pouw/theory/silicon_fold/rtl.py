"""Verilog for the baseline MACs and every accumulate-path mixer rho, emitted from Python so the wiring matches `rho.py`.

Every module is combinational: the next state from the state and the absorbed words. State registers are priced
separately (`synth.py`), as flip-flops a state bit. Bit layouts:
- 32-bit words little-endian in a bus: word i at [32 i +: 32]; 64-bit lanes likewise.
- SHA-512: st[64 i +: 64] = (a, b, c, d, e, f, g, h)[i].
- Keccak-p[b]: lane (x, y) at [w (x + 5 y) +: w], w = b / 25; the absorbed words XOR into the low `rate` bits.
- AES: byte i at [8 i +: 8], column-major as FIPS-197.
- Xoodoo: lane (y, x) at [32 (4 y + x) +: 32].
"""

from __future__ import annotations

import rho


def _rotr(x: str, r: int, w: int) -> str:
    r %= w
    if r == 0:
        return x
    return f"{{{x}[{r - 1}:0], {x}[{w - 1}:{r}]}}"


def _rotl(x: str, r: int, w: int) -> str:
    return _rotr(x, w - (r % w), w)


def mac4_s(w: int) -> str:
    """Four s{w} x s{w} products added into a 32-bit accumulator (s9 carries an s8 operand plus an s8 noise word)."""
    p = 2 * w
    lines = [f"module mac4_s{w}(input [{4 * w - 1}:0] a, input [{4 * w - 1}:0] b, input [31:0] acc, output [31:0] y);"]
    for i in range(4):
        lines.append(f"  wire signed [{p - 1}:0] p{i} = $signed(a[{w * i + w - 1}:{w * i}]) * $signed(b[{w * i + w - 1}:{w * i}]);")
    ext = " + ".join(f"{{{{{32 - p}{{p{i}[{p - 1}]}}}}, p{i}}}" for i in range(4))
    lines.append(f"  assign y = acc + {ext};")
    lines.append("endmodule")
    return "\n".join(lines)


def mac4_s8() -> str:
    return mac4_s(8)


def noise_add9() -> str:
    """One operand's on-chip rank-1 noise: y = a + s*v (s = +-1 as a sign bit), an s8 + s8 -> s9 add-or-subtract."""
    return """
module noise_add9(input [7:0] a, input [7:0] v, input s, output [8:0] y);
  wire signed [8:0] sa = {a[7], a};
  wire signed [8:0] sv = {v[7], v};
  assign y = s ? sa - sv : sa + sv;
endmodule"""


def fp8_mac() -> str:
    """E4M3 x E4M3 + FP32 -> FP32: exact product, one rounding (RNE), FP32 subnormals flushed, no NaN or Inf inputs."""
    return r"""
module fp8_mac(input [7:0] a, input [7:0] b, input [31:0] acc, output [31:0] y);
  wire ea0 = |a[6:3];
  wire eb0 = |b[6:3];
  wire [3:0] ma = {ea0, a[2:0]};
  wire [3:0] mb = {eb0, b[2:0]};
  wire [4:0] ep = (ea0 ? {1'b0, a[6:3]} : 5'd1) + (eb0 ? {1'b0, b[6:3]} : 5'd1);
  wire [7:0] mp = ma * mb;
  wire sp = a[7] ^ b[7];
  reg [2:0] lz;
  always @* begin
    casez (mp)
      8'b1???????: lz = 0; 8'b01??????: lz = 1; 8'b001?????: lz = 2; 8'b0001????: lz = 3;
      8'b00001???: lz = 4; 8'b000001??: lz = 5; 8'b0000001?: lz = 6; default: lz = 7;
    endcase
  end
  wire pz = (mp == 0);
  wire [7:0] epx = pz ? 8'd0 : ({3'b0, ep} + 8'd114 - {5'b0, lz});
  wire [23:0] fp = pz ? 24'd0 : ({mp, 16'b0} << lz);
  wire cz = (acc[30:23] == 0);
  wire [7:0] ecx = acc[30:23];
  wire [23:0] fc = cz ? 24'd0 : {1'b1, acc[22:0]};
  wire sc = acc[31];
  // larger magnitude first
  wire swap = {epx, fp} > {ecx, fc};
  wire sx = swap ? sp : sc, sy = swap ? sc : sp;
  wire [7:0] ex = swap ? epx : ecx, ey = swap ? ecx : epx;
  wire [23:0] fx = swap ? fp : fc, fy = swap ? fc : fp;
  wire [7:0] d = ex - ey;
  wire [26:0] fy27 = {fy, 3'b0};
  wire [26:0] fysh = (d > 8'd26) ? 27'd0 : (fy27 >> d);
  wire [26:0] lost = (d > 8'd26) ? fy27 : (fy27 & ~(27'h7ffffff << d));
  wire stk = |lost;
  wire [26:0] fya = {fysh[26:1], fysh[0] | stk};
  wire [27:0] s = (sx == sy) ? ({1'b0, fx, 3'b0} + {1'b0, fya}) : ({1'b0, fx, 3'b0} - {1'b0, fya});
  reg [4:0] nz;
  integer i;
  always @* begin
    nz = 5'd27;
    for (i = 0; i <= 26; i = i + 1) if (s[i]) nz = 5'd26 - i[4:0];
  end
  wire carry = s[27];
  wire [26:0] sn = carry ? {s[27:2], s[1] | s[0]} : (s[26:0] << nz);
  wire [9:0] en = carry ? ({2'b0, ex} + 10'd1) : ({2'b0, ex} - {5'b0, nz});
  wire g = sn[2], r = sn[1], st = sn[0], lsb = sn[3];
  wire up = g & (r | st | lsb);
  wire [24:0] fr = {1'b0, sn[26:3]} + {24'b0, up};
  wire [23:0] ff = fr[24] ? fr[24:1] : fr[23:0];
  wire [9:0] ef = fr[24] ? en + 10'd1 : en;
  wire zero = (s == 0) | (cz & pz);
  wire under = ef[9] | (ef == 0);
  wire over = (ef >= 10'd255) & ~ef[9];
  assign y = zero ? 32'd0 : under ? {sx, 31'd0} : over ? {sx, 8'hff, 23'd0} : {sx, ef[7:0], ff[22:0]};
endmodule
"""


def sha512_rnd() -> str:
    w = ["a", "b", "c", "d", "e", "f", "g", "h"]
    lines = ["module sha512_rnd(input [511:0] st, input [63:0] w, input [63:0] k, output [511:0] y);"]
    for i, n in enumerate(w):
        lines.append(f"  wire [63:0] {n} = st[{64 * i + 63}:{64 * i}];")
    lines.append(f"  wire [63:0] s1 = {_rotr('e', 14, 64)} ^ {_rotr('e', 18, 64)} ^ {_rotr('e', 41, 64)};")
    lines.append("  wire [63:0] ch = (e & f) ^ (~e & g);")
    lines.append("  wire [63:0] t1 = h + s1 + ch + k + w;")
    lines.append(f"  wire [63:0] s0 = {_rotr('a', 28, 64)} ^ {_rotr('a', 34, 64)} ^ {_rotr('a', 39, 64)};")
    lines.append("  wire [63:0] mj = (a & b) ^ (a & c) ^ (b & c);")
    lines.append("  wire [63:0] t2 = s0 + mj;")
    lines.append("  assign y = {g, f, e, d + t1, c, b, a, t1 + t2};")
    lines.append("endmodule")
    return "\n".join(lines)


def _rom(name: str, abits: int, dbits: int, values: list[int]) -> str:
    lines = [f"module {name}(input [{abits - 1}:0] t, output reg [{dbits - 1}:0] y);", "  always @* begin", "    case (t)"]
    for i, v in enumerate(values):
        lines.append(f"      {abits}'d{i}: y = {dbits}'h{v:x};")
    lines += [f"      default: y = {dbits}'h0;", "    endcase", "  end", "endmodule"]
    return "\n".join(lines)


def sha512_krom() -> str:
    return _rom("sha512_krom", 7, 64, rho.SHA512_K)


def keccak_rcrom() -> str:
    return _rom("keccak_rcrom", 5, 64, rho.KECCAK_RC64)


KECCAK_RATE = {1600: 1024, 800: 512, 400: 256, 200: 128}


def keccak_rnd(b: int) -> str:
    w = b // 25
    r = KECCAK_RATE[b]
    n = f"keccak{b}_rnd"
    L = [f"module {n}(input [{b - 1}:0] st, input [{r - 1}:0] m, input [{w - 1}:0] rc, output [{b - 1}:0] y);"]
    L.append(f"  wire [{b - 1}:0] s = st ^ {{{b - r}'b0, m}};")
    for x in range(5):
        for yy in range(5):
            i = x + 5 * yy
            L.append(f"  wire [{w - 1}:0] a{x}{yy} = s[{w * i + w - 1}:{w * i}];")
    for x in range(5):
        L.append(f"  wire [{w - 1}:0] c{x} = a{x}0 ^ a{x}1 ^ a{x}2 ^ a{x}3 ^ a{x}4;")
    for x in range(5):
        L.append(f"  wire [{w - 1}:0] d{x} = c{(x - 1) % 5} ^ {_rotl(f'c{(x + 1) % 5}', 1, w)};")
    for x in range(5):
        for yy in range(5):
            L.append(f"  wire [{w - 1}:0] t{x}{yy} = a{x}{yy} ^ d{x};")
    for x in range(5):
        for yy in range(5):
            bx, by = yy, (2 * x + 3 * yy) % 5
            L.append(f"  wire [{w - 1}:0] b{bx}{by} = {_rotl(f't{x}{yy}', rho.KECCAK_ROT[x][yy], w)};")
    outs = []
    for yy in range(5):
        for x in range(5):
            e = f"b{x}{yy} ^ (~b{(x + 1) % 5}{yy} & b{(x + 2) % 5}{yy})"
            if x == 0 and yy == 0:
                e = f"({e}) ^ rc"
            L.append(f"  wire [{w - 1}:0] o{x}{yy} = {e};")
    for i in reversed(range(25)):
        outs.append(f"o{i % 5}{i // 5}")
    L.append(f"  assign y = {{{', '.join(outs)}}};")
    L.append("endmodule")
    return "\n".join(L)


def _qr(p: str, a: str, b: str, c: str, d: str) -> tuple[list[str], tuple[str, str, str, str]]:
    L = []
    L.append(f"  wire [31:0] {p}a1 = {a} + {b};")
    L.append(f"  wire [31:0] {p}x1 = {d} ^ {p}a1;")
    L.append(f"  wire [31:0] {p}d1 = {_rotl(p + 'x1', 16, 32)};")
    L.append(f"  wire [31:0] {p}c1 = {c} + {p}d1;")
    L.append(f"  wire [31:0] {p}x2 = {b} ^ {p}c1;")
    L.append(f"  wire [31:0] {p}b1 = {_rotl(p + 'x2', 12, 32)};")
    L.append(f"  wire [31:0] {p}a2 = {p}a1 + {p}b1;")
    L.append(f"  wire [31:0] {p}x3 = {p}d1 ^ {p}a2;")
    L.append(f"  wire [31:0] {p}d2 = {_rotl(p + 'x3', 8, 32)};")
    L.append(f"  wire [31:0] {p}c2 = {p}c1 + {p}d2;")
    L.append(f"  wire [31:0] {p}x4 = {p}b1 ^ {p}c2;")
    L.append(f"  wire [31:0] {p}b2 = {_rotl(p + 'x4', 7, 32)};")
    return L, (f"{p}a2", f"{p}b2", f"{p}c2", f"{p}d2")


def chacha_qr() -> str:
    L = ["module chacha_qr(input [127:0] st, output [127:0] y);"]
    body, (a, b, c, d) = _qr("q", "st[31:0]", "st[63:32]", "st[95:64]", "st[127:96]")
    L += body
    L.append(f"  assign y = {{{d}, {c}, {b}, {a}}};")
    L.append("endmodule")
    return "\n".join(L)


def chacha_dround() -> str:
    """Absorb 16 words (XOR) then one ChaCha double round: four column QRs, then four diagonal QRs."""
    L = ["module chacha_dround(input [511:0] st, input [511:0] m, output [511:0] y);", "  wire [511:0] s = st ^ m;"]
    v = [f"s[{32 * i + 31}:{32 * i}]" for i in range(16)]
    for i in range(4):
        body, out = _qr(f"c{i}", v[i], v[4 + i], v[8 + i], v[12 + i])
        L += body
        v[i], v[4 + i], v[8 + i], v[12 + i] = out
    for j, (a, b, c, d) in enumerate(((0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))):
        body, out = _qr(f"g{j}", v[a], v[b], v[c], v[d])
        L += body
        v[a], v[b], v[c], v[d] = out
    L.append(f"  assign y = {{{', '.join(reversed(v))}}};")
    L.append("endmodule")
    return "\n".join(L)


def aes_rnd() -> str:
    """One full AES round (SubBytes, ShiftRows, MixColumns, AddRoundKey) with the absorbed 128 bits as its round key."""
    L = ["module aes_rnd(input [127:0] st, input [127:0] k, output [127:0] y);"]
    L.append("  function [7:0] sb(input [7:0] x);")
    L.append("    case (x)")
    for i, v in enumerate(rho.AES_SBOX):
        L.append(f"      8'h{i:02x}: sb = 8'h{v:02x};")
    L.append("    endcase")
    L.append("  endfunction")
    L.append("  function [7:0] xt(input [7:0] x); xt = {x[6:0], 1'b0} ^ (x[7] ? 8'h1b : 8'h00); endfunction")
    for i in range(16):
        L.append(f"  wire [7:0] u{i} = sb(st[{8 * i + 7}:{8 * i}]);")
    for i in range(16):
        L.append(f"  wire [7:0] r{i} = u{(i + 4 * (i % 4)) % 16};")
    outs = []
    for c in range(4):
        a0, a1, a2, a3 = (f"r{4 * c + j}" for j in range(4))
        L.append(f"  wire [7:0] m{4 * c} = xt({a0}) ^ xt({a1}) ^ {a1} ^ {a2} ^ {a3};")
        L.append(f"  wire [7:0] m{4 * c + 1} = {a0} ^ xt({a1}) ^ xt({a2}) ^ {a2} ^ {a3};")
        L.append(f"  wire [7:0] m{4 * c + 2} = {a0} ^ {a1} ^ xt({a2}) ^ xt({a3}) ^ {a3};")
        L.append(f"  wire [7:0] m{4 * c + 3} = xt({a0}) ^ {a0} ^ {a1} ^ {a2} ^ xt({a3});")
    for i in reversed(range(16)):
        outs.append(f"m{i} ^ k[{8 * i + 7}:{8 * i}]")
    L.append(f"  assign y = {{{', '.join(outs)}}};")
    L.append("endmodule")
    return "\n".join(L)


def ascon_rnd() -> str:
    """Absorb 320 bits (XOR) then one Ascon permutation round with round constant c."""
    L = ["module ascon_rnd(input [319:0] st, input [319:0] m, input [7:0] c, output [319:0] y);", "  wire [319:0] s = st ^ m;"]
    for i in range(5):
        L.append(f"  wire [63:0] x{i} = s[{64 * i + 63}:{64 * i}];")
    L.append("  wire [63:0] i2 = x2 ^ {56'b0, c};")
    L.append("  wire [63:0] p0 = x0 ^ x4, p4 = x4 ^ x3, p2 = i2 ^ x1, p1 = x1, p3 = x3;")
    L.append("  wire [63:0] q0 = p0 ^ (~p1 & p2), q1 = p1 ^ (~p2 & p3), q2 = p2 ^ (~p3 & p4), q3 = p3 ^ (~p4 & p0), q4 = p4 ^ (~p0 & p1);")
    L.append("  wire [63:0] v1 = q1 ^ q0, v0 = q0 ^ q4, v3 = q3 ^ q2, v2 = ~q2, v4 = q4;")
    rots = [(19, 28), (61, 39), (1, 6), (10, 17), (7, 41)]
    for i, (r1, r2) in enumerate(rots):
        L.append(f"  wire [63:0] z{i} = v{i} ^ {_rotr(f'v{i}', r1, 64)} ^ {_rotr(f'v{i}', r2, 64)};")
    L.append("  assign y = {z4, z3, z2, z1, z0};")
    L.append("endmodule")
    return "\n".join(L)


def xoodoo_rnd() -> str:
    """Absorb 384 bits (XOR) then one Xoodoo round with round constant rc."""
    L = ["module xoodoo_rnd(input [383:0] st, input [383:0] m, input [31:0] rc, output [383:0] y);", "  wire [383:0] s = st ^ m;"]
    for yy in range(3):
        for x in range(4):
            i = 4 * yy + x
            L.append(f"  wire [31:0] a{yy}{x} = s[{32 * i + 31}:{32 * i}];")
    for x in range(4):
        L.append(f"  wire [31:0] p{x} = a0{x} ^ a1{x} ^ a2{x};")
    for x in range(4):
        px = f"p{(x - 1) % 4}"
        L.append(f"  wire [31:0] e{x} = {_rotl(px, 5, 32)} ^ {_rotl(px, 14, 32)};")
    for yy in range(3):
        for x in range(4):
            L.append(f"  wire [31:0] t{yy}{x} = a{yy}{x} ^ e{x};")
    # rho-west
    for x in range(4):
        L.append(f"  wire [31:0] w0{x} = t0{x};")
        L.append(f"  wire [31:0] w1{x} = t1{(x - 1) % 4};")
        L.append(f"  wire [31:0] w2{x} = {_rotl(f't2{x}', 11, 32)};")
    L.append("  wire [31:0] j00 = w00 ^ rc;")
    for x in range(1, 4):
        L.append(f"  wire [31:0] j0{x} = w0{x};")
    for yy in (1, 2):
        for x in range(4):
            L.append(f"  wire [31:0] j{yy}{x} = w{yy}{x};")
    for yy in range(3):
        for x in range(4):
            L.append(f"  wire [31:0] k{yy}{x} = j{yy}{x} ^ (~j{(yy + 1) % 3}{x} & j{(yy + 2) % 3}{x});")
    outs = {}
    for x in range(4):
        outs[(0, x)] = f"k0{x}"
        outs[(1, x)] = _rotl(f"k1{x}", 1, 32)
        outs[(2, x)] = _rotl(f"k2{(x - 2) % 4}", 8, 32)
    L.append(f"  assign y = {{{', '.join(outs[(i // 4, i % 4)] for i in reversed(range(12)))}}};")
    L.append("endmodule")
    return "\n".join(L)


def speck64_rnd() -> str:
    return "\n".join([
        "module speck64_rnd(input [63:0] st, input [31:0] k, output [63:0] y);",
        "  wire [31:0] x = st[31:0], v = st[63:32];",
        f"  wire [31:0] xn = ({_rotr('x', 8, 32)} + v) ^ k;",
        f"  wire [31:0] vn = {_rotl('v', 3, 32)} ^ xn;",
        "  assign y = {vn, xn};",
        "endmodule",
    ])


def simon64_rnd() -> str:
    return "\n".join([
        "module simon64_rnd(input [63:0] st, input [31:0] k, output [63:0] y);",
        "  wire [31:0] x = st[31:0], v = st[63:32];",
        f"  wire [31:0] f = ({_rotl('x', 1, 32)} & {_rotl('x', 8, 32)}) ^ {_rotl('x', 2, 32)};",
        "  assign y = {x, v ^ f ^ k};",
        "endmodule",
    ])


def nv_arx() -> str:
    return "\n".join([
        "module nv_arx(input [31:0] t, input [31:0] w, output [31:0] y);",
        "  wire [31:0] s = t + w;",
        f"  assign y = {_rotl('s', 7, 32)} ^ t;",
        "endmodule",
    ])


def affine() -> str:
    return "\n".join([
        "module affine(input [31:0] t, input [31:0] w, output [31:0] y);",
        f"  assign y = t * 32'h{rho.AFFINE_A:08x} + w;",
        "endmodule",
    ])


MODULES = {
    "mac4_s8": mac4_s8,
    "mac4_s9": lambda: mac4_s(9),
    "noise_add9": noise_add9,
    "fp8_mac": fp8_mac,
    "sha512_rnd": sha512_rnd,
    "sha512_krom": sha512_krom,
    "keccak_rcrom": keccak_rcrom,
    "keccak1600_rnd": lambda: keccak_rnd(1600),
    "keccak800_rnd": lambda: keccak_rnd(800),
    "keccak400_rnd": lambda: keccak_rnd(400),
    "keccak200_rnd": lambda: keccak_rnd(200),
    "chacha_qr": chacha_qr,
    "chacha_dround": chacha_dround,
    "aes_rnd": aes_rnd,
    "ascon_rnd": ascon_rnd,
    "xoodoo_rnd": xoodoo_rnd,
    "speck64_rnd": speck64_rnd,
    "simon64_rnd": simon64_rnd,
    "nv_arx": nv_arx,
    "affine": affine,
}


def verilog(name: str) -> str:
    return MODULES[name]() + "\n"
