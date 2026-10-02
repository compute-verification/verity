"""BLS signature verification on BLS12-381 with signatures in G1 and keys in G2 (min-sig), stdlib only.

It serves `verity_pouw.beacon`, which checks drand's `bls-unchained-g1-rfc9380` rounds against a pinned group key. It
verifies only; it holds no secrets, so it is not constant-time.
- **The curves:** E: y² = x³ + 4 over Fp, and E′: y² = x³ + 4(1 + u) over Fp2 = Fp[u]/(u² + 1).
- **The pairing:** the optimal ate pairing, with Miller loop over |x| = 0xd201000000010000 and a plain final
  exponentiation. Fp12 is Fp[w]/(w¹² − 2w⁶ + 2) with u = w⁶ − 1, and E′ untwists to E by (x, y) ↦ (x·w⁻², y·w⁻³).
- **Hash-to-curve:** RFC 9380's BLS12381G1_XMD:SHA-256_SSWU_RO_ (expand_message_xmd, the simplified SWU map on the
  11-isogenous curve, the 11-isogeny, and h_eff = 0xd201000000010001), with constants from RFC 9380 §8.8.1 and
  Appendix E.2.
- **Points:** ZCash's compressed encodings: 48 bytes in G1, 96 in G2, with the compression, infinity and sign flags in
  the top three bits.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

P = 0x1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB
R = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001
ATE = 0xD201000000010000                     # |x|, the curve parameter's magnitude (x is negative)
H_EFF = 0xD201000000010001
G2_GEN = ((0x024AA2B2F08F0A91260805272DC51051C6E47AD4FA403B02B4510B647AE3D1770BAC0326A805BBEFD48056C8C121BDB8,
           0x13E02B6052719F607DACD3A088274F65596BD0D09920B61AB5DA61BBDC7F5049334CF11213945D57E5AC7D055D042B7E),
          (0x0CE5D527727D6E118CC9CDC6DA2E351AADFD9BAA8CBDD3A76D429A695160D12C923AC9CC3BACA289E193548608B82801,
           0x0606C4A02EA734CC32ACD2B02BC28B99CB3E287E85A763AF267492AB572E99AB3F370D275CEC1DA1AAA9075FF05F79BE))

# ---- Fp and Fp2 ---------------------------------------------------------------------------------------------------


def inv(a: int) -> int:
    return pow(a, P - 2, P)


def sqrt(a: int) -> int | None:
    """A square root in Fp (p ≡ 3 mod 4), or None."""
    r = pow(a, (P + 1) // 4, P)
    return r if r * r % P == a % P else None


def f2_mul(a, b):
    return ((a[0] * b[0] - a[1] * b[1]) % P, (a[0] * b[1] + a[1] * b[0]) % P)


def f2_add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)


def f2_sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)


def f2_inv(a):
    d = inv((a[0] * a[0] + a[1] * a[1]) % P)
    return (a[0] * d % P, -a[1] * d % P)


def f2_scale(a, k: int):
    return (a[0] * k % P, a[1] * k % P)


# ---- Fp12 = Fp[w]/(w^12 - 2w^6 + 2) -------------------------------------------------------------------------------


def f12_mul(a, b):
    c = [0] * 23
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                c[i + j] += x * y
    for k in range(22, 11, -1):
        t = c[k]
        if t:
            c[k - 6] += 2 * t
            c[k - 12] -= 2 * t
    return tuple(x % P for x in c[:12])


def f12_pow(a, e: int):
    out = ONE12
    for bit in bin(e)[2:]:
        out = f12_mul(out, out)
        if bit == "1":
            out = f12_mul(out, a)
    return out


def f12_inv(a):
    """The inverse by the extended Euclidean algorithm on polynomials over Fp, against the modulus."""
    lm, hm = [1] + [0] * 12, [0] * 13
    low, high = list(a) + [0], [2, 0, 0, 0, 0, 0, P - 2, 0, 0, 0, 0, 0, 1]

    def deg(p):
        d = len(p) - 1
        while d and not p[d]:
            d -= 1
        return d

    while deg(low):
        r = [0] * 13
        n, d = [x for x in high], deg(low)
        li = inv(low[d])
        while deg(n) >= d and any(n):
            s, q = deg(n) - d, n[deg(n)] * li % P
            r[s] = (r[s] + q) % P
            for i in range(d + 1):
                n[i + s] = (n[i + s] - q * low[i]) % P
        nm = [x for x in hm]
        for i in range(13):
            for j in range(13 - i):
                nm[i + j] = (nm[i + j] - lm[i] * r[j]) % P
        lm, low, hm, high = nm, n, lm, low
    i0 = inv(low[0])
    return tuple(x * i0 % P for x in lm[:12])


ONE12 = (1,) + (0,) * 11
W = (0, 1) + (0,) * 10
W_INV = f12_inv(W)
W2_INV, W3_INV = f12_mul(W_INV, W_INV), f12_mul(f12_mul(W_INV, W_INV), W_INV)


def embed(a) -> tuple:
    """An Fp2 element a0 + a1·u as an element of Fp12, u = w⁶ − 1."""
    return ((a[0] - a[1]) % P, 0, 0, 0, 0, 0, a[1] % P, 0, 0, 0, 0, 0)


def fp12(x: int) -> tuple:
    return (x % P,) + (0,) * 11


# ---- G1 (affine over Fp; None is the point at infinity) -------------------------------------------------------------


def g1_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == b[0]:
        if (a[1] + b[1]) % P == 0:
            return None
        lam = 3 * a[0] * a[0] * inv(2 * a[1]) % P
    else:
        lam = (b[1] - a[1]) * inv(b[0] - a[0]) % P
    x = (lam * lam - a[0] - b[0]) % P
    return (x, (lam * (a[0] - x) - a[1]) % P)


def g1_mul(a, k: int):
    out = None
    for bit in bin(k)[2:]:
        out = g1_add(out, out)
        if bit == "1":
            out = g1_add(out, a)
    return out


def g1_on_curve(a) -> bool:
    return a is not None and (a[1] * a[1] - a[0] ** 3 - 4) % P == 0


def g1_compress(a) -> bytes:
    """ZCash's 48-byte encoding of an affine G1 point (not infinity)."""
    head = bytearray(a[0].to_bytes(48, "big"))
    head[0] |= 0x80 | (0x20 if a[1] > (P - 1) // 2 else 0)
    return bytes(head)


def g1_decompress(data: bytes):
    """A compressed G1 point in the prime-order subgroup, or None (malformed, off the curve, infinity, or outside the
    subgroup)."""
    if len(data) != 48 or not data[0] & 0x80 or data[0] & 0x40:
        return None
    sign, x = bool(data[0] & 0x20), int.from_bytes(bytes([data[0] & 0x1F]) + data[1:], "big")
    if x >= P:
        return None
    y = sqrt((x ** 3 + 4) % P)
    if y is None:
        return None
    if (y > (P - 1) // 2) != sign:
        y = P - y
    pt = (x, y)
    return pt if g1_mul(pt, R) is None else None


# ---- G2 (affine over Fp2) -------------------------------------------------------------------------------------------

B2 = (4, 4)


def g2_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == b[0]:
        if f2_add(a[1], b[1]) == (0, 0):
            return None
        lam = f2_mul(f2_scale(f2_mul(a[0], a[0]), 3), f2_inv(f2_scale(a[1], 2)))
    else:
        lam = f2_mul(f2_sub(b[1], a[1]), f2_inv(f2_sub(b[0], a[0])))
    x = f2_sub(f2_sub(f2_mul(lam, lam), a[0]), b[0])
    return (x, f2_sub(f2_mul(lam, f2_sub(a[0], x)), a[1]))


def g2_mul(a, k: int):
    out = None
    for bit in bin(k)[2:]:
        out = g2_add(out, out)
        if bit == "1":
            out = g2_add(out, a)
    return out


def g2_on_curve(a) -> bool:
    return a is not None and f2_sub(f2_mul(a[1], a[1]), f2_add(f2_mul(f2_mul(a[0], a[0]), a[0]), B2)) == (0, 0)


def g2_compress(a) -> bytes:
    """ZCash's 96-byte encoding: x's u-coefficient, then its constant, with the sign of y read on its u-coefficient
    first (lexicographic order on (y1, y0))."""
    (x0, x1), (y0, y1) = a
    larger = y1 > (P - 1) // 2 if y1 else y0 > (P - 1) // 2
    head = bytearray(x1.to_bytes(48, "big"))
    head[0] |= 0x80 | (0x20 if larger else 0)
    return bytes(head) + x0.to_bytes(48, "big")


# ---- the pairing ----------------------------------------------------------------------------------------------------


def _line(t, q, p_x: int, p_y: int):
    """The line through t and q on E′ (the tangent when t = q), untwisted, at the G1 point (p_x, p_y), in Fp12."""
    if t[0] == q[0] and t[1] != q[1]:
        return f12_sub(fp12(p_x), f12_mul(embed(t[0]), W2_INV))
    if t == q:
        lam = f2_mul(f2_scale(f2_mul(t[0], t[0]), 3), f2_inv(f2_scale(t[1], 2)))
    else:
        lam = f2_mul(f2_sub(q[1], t[1]), f2_inv(f2_sub(q[0], t[0])))
    dx = f12_sub(fp12(p_x), f12_mul(embed(t[0]), W2_INV))
    return f12_sub(f12_sub(fp12(p_y), f12_mul(embed(t[1]), W3_INV)), f12_mul(f12_mul(embed(lam), W_INV), dx))


def f12_sub(a, b):
    return tuple((x - y) % P for x, y in zip(a, b))


def miller(q, p):
    """f_{|x|, q}(p) for q on E′ and p on E, both affine and not infinity."""
    f, t = ONE12, q
    for i in range(ATE.bit_length() - 2, -1, -1):
        f = f12_mul(f12_mul(f, f), _line(t, t, *p))
        t = g2_add(t, t)
        if ATE >> i & 1:
            f = f12_mul(f, _line(t, q, *p))
            t = g2_add(t, q)
    return f


FINAL = (P ** 12 - 1) // R


def pairings_equal(q1, p1, q2, p2) -> bool:
    """e(p1, q1) = e(p2, q2), by one final exponentiation of f_{q1}(p1) · f_{q2}(−p2)."""
    f = f12_mul(miller(q1, p1), miller(q2, (p2[0], P - p2[1])))
    return f12_pow(f, FINAL) == ONE12


# ---- hash to G1 (RFC 9380, BLS12381G1_XMD:SHA-256_SSWU_RO_) ---------------------------------------------------------

ISO_A = 0x144698A3B8E9433D693A02C96D4982B0EA985383EE66A8D8E8981AEFD881AC98936F8DA0E0F97F5CF428082D584C1D
ISO_B = 0x12E2908D11688030018B12E8753EEE3B2016C1F0F24F4070A0B9C14FCEF35EF55A23215A316CEAA5D1CC48E98E172BE0
ISO_Z = 11
#: The 11-isogeny's polynomials, constant term first (RFC 9380 Appendix E.2)
XNUM = (
    0x11A05F2B1E833340B809101DD99815856B303E88A2D7005FF2627B56CDB4E2C85610C2D5F2E62D6EAEAC1662734649B7,
    0x17294ED3E943AB2F0588BAB22147A81C7C17E75B2F6A8417F565E33C70D1E86B4838F2A6F318C356E834EEF1B3CB83BB,
    0x0D54005DB97678EC1D1048C5D10A9A1BCE032473295983E56878E501EC68E25C958C3E3D2A09729FE0179F9DAC9EDCB0,
    0x1778E7166FCC6DB74E0609D307E55412D7F5E4656A8DBF25F1B33289F1B330835336E25CE3107193C5B388641D9B6861,
    0x0E99726A3199F4436642B4B3E4118E5499DB995A1257FB3F086EEB65982FAC18985A286F301E77C451154CE9AC8895D9,
    0x1630C3250D7313FF01D1201BF7A74AB5DB3CB17DD952799B9ED3AB9097E68F90A0870D2DCAE73D19CD13C1C66F652983,
    0x0D6ED6553FE44D296A3726C38AE652BFB11586264F0F8CE19008E218F9C86B2A8DA25128C1052ECADDD7F225A139ED84,
    0x17B81E7701ABDBE2E8743884D1117E53356DE5AB275B4DB1A682C62EF0F2753339B7C8F8C8F475AF9CCB5618E3F0C88E,
    0x080D3CF1F9A78FC47B90B33563BE990DC43B756CE79F5574A2C596C928C5D1DE4FA295F296B74E956D71986A8497E317,
    0x169B1F8E1BCFA7C42E0C37515D138F22DD2ECB803A0C5C99676314BAF4BB1B7FA3190B2EDC0327797F241067BE390C9E,
    0x10321DA079CE07E272D8EC09D2565B0DFA7DCCDDE6787F96D50AF36003B14866F69B771F8C285DECCA67DF3F1605FB7B,
    0x06E08C248E260E70BD1E962381EDEE3D31D79D7E22C837BC23C0BF1BC24C6B68C24B1B80B64D391FA9C8BA2E8BA2D229,
)
XDEN = (
    0x08CA8D548CFF19AE18B2E62F4BD3FA6F01D5EF4BA35B48BA9C9588617FC8AC62B558D681BE343DF8993CF9FA40D21B1C,
    0x12561A5DEB559C4348B4711298E536367041E8CA0CF0800C0126C2588C48BF5713DAA8846CB026E9E5C8276EC82B3BFF,
    0x0B2962FE57A3225E8137E629BFF2991F6F89416F5A718CD1FCA64E00B11ACEACD6A3D0967C94FEDCFCC239BA5CB83E19,
    0x03425581A58AE2FEC83AAFEF7C40EB545B08243F16B1655154CCA8ABC28D6FD04976D5243EECF5C4130DE8938DC62CD8,
    0x13A8E162022914A80A6F1D5F43E7A07DFFDFC759A12062BB8D6B44E833B306DA9BD29BA81F35781D539D395B3532A21E,
    0x0E7355F8E4E667B955390F7F0506C6E9395735E9CE9CAD4D0A43BCEF24B8982F7400D24BC4228F11C02DF9A29F6304A5,
    0x0772CAACF16936190F3E0C63E0596721570F5799AF53A1894E2E073062AEDE9CEA73B3538F0DE06CEC2574496EE84A3A,
    0x14A7AC2A9D64A8B230B3F5B074CF01996E7F63C21BCA68A81996E1CDF9822C580FA5B9489D11E2D311F7D99BBDCC5A5E,
    0x0A10ECF6ADA54F825E920B3DAFC7A3CCE07F8D1D7161366B74100DA67F39883503826692ABBA43704776EC3A79A1D641,
    0x095FC13AB9E92AD4476D6E3EB3A56680F682B4EE96F7D03776DF533978F31C1593174E4B4B7865002D6384D168ECDD0A,
    0x01,
)
YNUM = (
    0x090D97C81BA24EE0259D1F094980DCFA11AD138E48A869522B52AF6C956543D3CD0C7AEE9B3BA3C2BE9845719707BB33,
    0x134996A104EE5811D51036D776FB46831223E96C254F383D0F906343EB67AD34D6C56711962FA8BFE097E75A2E41C696,
    0x00CC786BAA966E66F4A384C86A3B49942552E2D658A31CE2C344BE4B91400DA7D26D521628B00523B8DFE240C72DE1F6,
    0x01F86376E8981C217898751AD8746757D42AA7B90EEB791C09E4A3EC03251CF9DE405ABA9EC61DECA6355C77B0E5F4CB,
    0x08CC03FDEFE0FF135CAF4FE2A21529C4195536FBE3CE50B879833FD221351ADC2EE7F8DC099040A841B6DAECF2E8FEDB,
    0x16603FCA40634B6A2211E11DB8F0A6A074A7D0D4AFADB7BD76505C3D3AD5544E203F6326C95A807299B23AB13633A5F0,
    0x04AB0B9BCFAC1BBCB2C977D027796B3CE75BB8CA2BE184CB5231413C4D634F3747A87AC2460F415EC961F8855FE9D6F2,
    0x0987C8D5333AB86FDE9926BD2CA6C674170A05BFE3BDD81FFD038DA6C26C842642F64550FEDFE935A15E4CA31870FB29,
    0x09FC4018BD96684BE88C9E221E4DA1BB8F3ABD16679DC26C1E8B6E6A1F20CABE69D65201C78607A360370E577BDBA587,
    0x0E1BBA7A1186BDB5223ABDE7ADA14A23C42A0CA7915AF6FE06985E7ED1E4D43B9B3F7055DD4EBA6F2BAFAAEBCA731C30,
    0x19713E47937CD1BE0DFD0B8F1D43FB93CD2FCBCB6CAF493FD1183E416389E61031BF3A5CCE3FBAFCE813711AD011C132,
    0x18B46A908F36F6DEB918C143FED2EDCC523559B8AAF0C2462E6BFE7F911F643249D9CDF41B44D606CE07C8A4D0074D8E,
    0x0B182CAC101B9399D155096004F53F447AA7B12A3426B08EC02710E807B4633F06C851C1919211F20D4C04F00B971EF8,
    0x0245A394AD1ECA9B72FC00AE7BE315DC757B3B080D4C158013E6632D3C40659CC6CF90AD1C232A6442D9D3F5DB980133,
    0x05C129645E44CF1102A159F748C4A3FC5E673D81D7E86568D9AB0F5D396A7CE46BA1049B6579AFB7866B1E715475224B,
    0x15E6BE4E990F03CE4EA50B3B42DF2EB5CB181D8F84965A3957ADD4FA95AF01B2B665027EFEC01C7704B456BE69C8B604,
)
YDEN = (
    0x16112C4C3A9C98B252181140FAD0EAE9601A6DE578980BE6EEC3232B5BE72E7A07F3688EF60C206D01479253B03663C1,
    0x1962D75C2381201E1A0CBD6C43C348B885C84FF731C4D59CA4A10356F453E01F78A4260763529E3532F6102C2E49A03D,
    0x058DF3306640DA276FAAAE7D6E8EB15778C4855551AE7F310C35A5DD279CD2ECA6757CD636F96F891E2538B53DBF67F2,
    0x16B7D288798E5395F20D23BF89EDB4D1D115C5DBDDBCD30E123DA489E726AF41727364F2C28297ADA8D26D98445F5416,
    0x0BE0E079545F43E4B00CC912F8228DDCC6D19C9F0F69BBB0542EDA0FC9DEC916A20B15DC0FD2EDEDDA39142311A5001D,
    0x08D9E5297186DB2D9FB266EAAC783182B70152C65550D881C5ECD87B6F0F5A6449F38DB9DFA9CCE202C6477FAAF9B7AC,
    0x166007C08A99DB2FC3BA8734ACE9824B5EECFDFA8D0CF8EF5DD365BC400A0051D5FA9C01A58B1FB93D1A1399126A775C,
    0x16A3EF08BE3EA7EA03BCDDFABBA6FF6EE5A4375EFA1F4FD7FEB34FD206357132B920F5B00801DEE460EE415A15812ED9,
    0x1866C8ED336C61231A1BE54FD1D74CC4F9FB0CE4C6AF5920ABC5750C4BF39B4852CFE2F7BB9248836B233D9D55535D4A,
    0x167A55CDA70A6E1CEA820597D94A84903216F763E13D87BB5308592E7EA7D4FBC7385EA3D529B35E346EF48BB8913F55,
    0x04D2F259EEA405BD48F010A01AD2911D9C6DD039BB61A6290E591B36E636A5C871A5C29F4F83060400F8B49CBA8F6AA8,
    0x0ACCBB67481D033FF5852C1E48C50C477F94FF8AEFCE42D28C0F9A88CEA7913516F968986F7EBBEA9684B529E2561092,
    0x0AD6B9514C767FE3C3613144B45F1496543346D98ADF02267D5CEEF9A00D9B8693000763E3B90AC11E99B138573345CC,
    0x02660400EB2E4F3B628BDD0D53CD76F2BF565B94E72927C1CB748DF27942480E420517BD8714CC80D1FADC1326ED06F7,
    0x0E0FA1D816DDC03E6B24255E0D7819C171C40F65E273B853324EFCD6356CAA205CA2F570F13497804415473A1D634B8F,
    0x01,
)


def expand_message_xmd(msg: bytes, dst: bytes, n: int) -> bytes:
    """RFC 9380 §5.3.1 with SHA-256."""
    ell = -(-n // 32)
    if ell > 255 or len(dst) > 255:
        raise ValueError("expand_message_xmd: output or DST too long")
    dst_prime = dst + bytes([len(dst)])
    b0 = hashlib.sha256(bytes(64) + msg + n.to_bytes(2, "big") + b"\0" + dst_prime).digest()
    b = [hashlib.sha256(b0 + b"\1" + dst_prime).digest()]
    for i in range(2, ell + 1):
        b.append(hashlib.sha256(bytes(x ^ y for x, y in zip(b0, b[-1])) + bytes([i]) + dst_prime).digest())
    return b"".join(b)[:n]


def _poly(coeffs, x: int) -> int:
    out = 0
    for c in reversed(coeffs):
        out = (out * x + c) % P
    return out


def _sswu(u: int):
    """RFC 9380 §6.6.2's simplified SWU on E′₁₁: y² = x³ + A′x + B′, then the 11-isogeny to E."""
    zu2 = ISO_Z * u * u % P
    tv = (zu2 * zu2 + zu2) % P
    x1 = ISO_B * inv(ISO_Z * ISO_A) % P if tv == 0 else (-ISO_B * inv(ISO_A) * (1 + inv(tv))) % P
    gx1 = (x1 ** 3 + ISO_A * x1 + ISO_B) % P
    y = sqrt(gx1)
    x = x1
    if y is None:
        x = zu2 * x1 % P
        y = sqrt((x ** 3 + ISO_A * x + ISO_B) % P)
    if u % 2 != y % 2:
        y = P - y
    return (_poly(XNUM, x) * inv(_poly(XDEN, x)) % P, y * _poly(YNUM, x) * inv(_poly(YDEN, x)) % P)


def hash_to_g1(msg: bytes, dst: bytes):
    """RFC 9380's hash_to_curve for BLS12381G1_XMD:SHA-256_SSWU_RO_."""
    uniform = expand_message_xmd(msg, dst, 128)
    u0, u1 = (int.from_bytes(uniform[64 * i:64 * (i + 1)], "big") % P for i in (0, 1))
    return g1_mul(g1_add(_sswu(u0), _sswu(u1)), H_EFF)


@lru_cache(maxsize=256)
def verify_g1(pk, message: bytes, signature: bytes, dst: bytes) -> bool:
    """A min-sig BLS signature on `message`: e(σ, g2) = e(H(m), pk), with σ decompressed into G1's subgroup."""
    sig = g1_decompress(signature)
    if sig is None:
        return False
    return pairings_equal(G2_GEN, sig, pk, hash_to_g1(message, dst))
