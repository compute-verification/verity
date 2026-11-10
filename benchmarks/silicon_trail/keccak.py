"""The logging chain's step, exactly: one round of Keccak-p[25w] after XORing an input into the absorbed lanes.

A state is a list of 25 lanes of w bits, lane (x, y) at index x + 5y, bit z of a lane at bit z of the integer, so that
state bit w(5y + x) + z is a[x, y, z] as in the Keccak reference and the Ketje v2 document (section 2).  Widths
w = 1, 2, 4, 8, 16, 32, 64 give Keccak-p[25], ..., Keccak-p[1600].

    step:  s_i = R(s_{i-1} xor P(x_i)),   R = iota . chi . pi . rho . theta

P places 4w input bits in four lanes.  Two placements:

    plane  lanes (0,0), (1,0), (2,0), (3,0): plane y = 0 of the plain round.  This is what the synthesized absorber
           does (m[255:0] into state bits 0..255).
    diag   lanes (0,0), (1,1), (2,2), (3,3): Ketje Major's.  Ketje v2 runs the twisted round pi . R . pi^-1 and absorbs
           into plane y = 0 of the twisted state; in the plain round's coordinates that plane is this diagonal
           (`twisted_step` and `selfcheck.py` check the equivalence).

The round constant is a parameter.  It never matters to which inputs collide: iota adds a constant to lane (0, 0),
which both placements absorb into, so a chain with constant rc on inputs (x_1, ..., x_n) is the chain with constant 0
on (x_1, x_2 xor rc, ..., x_n xor rc), xor rc at the end (`selfcheck.py`).
"""

from __future__ import annotations

WIDTHS = (1, 2, 4, 8, 16, 32, 64)
PLACEMENTS = {
    "plane": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "diag": ((0, 0), (1, 1), (2, 2), (3, 3)),
}


def rho_offsets() -> list[list[int]]:
    """rho's offsets r[x][y] = (t + 1)(t + 2)/2 along (x, y) = M^t (1, 0), M = [[0, 1], [2, 3]]; r[0][0] = 0."""
    r = [[0] * 5 for _ in range(5)]
    x, y = 1, 0
    for t in range(24):
        r[x][y] = (t + 1) * (t + 2) // 2
        x, y = y, (2 * x + 3 * y) % 5
    return r


RHO = rho_offsets()


def round_constants(w: int, n: int = 24) -> list[int]:
    """RC[i] for round indices 0..n-1: bit 2^j - 1 of lane (0, 0) is rc[j + 7i], rc the LFSR x^8 + x^6 + x^5 + x^4 + 1."""
    ell = w.bit_length() - 1
    out = []
    for i in range(n):
        c = 0
        for j in range(ell + 1):
            t = (j + 7 * i) % 255
            reg = 1
            for _ in range(t):
                reg <<= 1
                if reg & 0x100:
                    reg ^= 0x171
            if reg & 1:
                c |= 1 << ((1 << j) - 1)
        out.append(c)
    return out


def last_round_constant(w: int) -> int:
    """The constant of Keccak-p[25w, 1], the last round of Keccak-f[25w] (index 12 + 2l - 1): Ketje's step."""
    ell = w.bit_length() - 1
    return round_constants(w, 12 + 2 * ell)[-1]


def rot(v: int, r: int, w: int) -> int:
    r %= w
    return ((v << r) | (v >> (w - r))) & ((1 << w) - 1) if r else v


def theta(a: list[int], w: int) -> list[int]:
    c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
    d = [c[(x - 1) % 5] ^ rot(c[(x + 1) % 5], 1, w) for x in range(5)]
    return [a[i] ^ d[i % 5] for i in range(25)]


def theta_inv(a: list[int], w: int) -> list[int]:
    """theta's inverse, by linear algebra on the 5w column parities (theta fixes a state's parity pattern p to p + E(p))."""
    n = 5 * w
    rows = []
    for i in range(n):
        x, z = divmod(i, w)
        v = 1 << i
        v ^= 1 << (((x + 1) % 5) * w + z)
        v ^= 1 << (((x - 1) % 5) * w + (z + 1) % w)
        rows.append(v)
    # column i of the matrix is rows[i]: parity bit i of the input contributes rows[i] to the output parities
    par = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
    target = sum(par[x] << (x * w) for x in range(5))
    basis: dict[int, tuple[int, int]] = {}
    for i, v in enumerate(rows):
        tag = 1 << i
        while v:
            top = v.bit_length() - 1
            if top not in basis:
                basis[top] = (v, tag)
                break
            bv, bt = basis[top]
            v ^= bv
            tag ^= bt
    sol = 0
    t = target
    while t:
        top = t.bit_length() - 1
        bv, bt = basis[top]
        t ^= bv
        sol ^= bt
    p = [(sol >> (x * w)) & ((1 << w) - 1) for x in range(5)]
    d = [p[(x - 1) % 5] ^ rot(p[(x + 1) % 5], 1, w) for x in range(5)]
    return [a[i] ^ d[i % 5] for i in range(25)]


def rho_pi(a: list[int], w: int) -> list[int]:
    b = [0] * 25
    for x in range(5):
        for y in range(5):
            b[y + 5 * ((2 * x + 3 * y) % 5)] = rot(a[x + 5 * y], RHO[x][y], w)
    return b


def rho_pi_inv(b: list[int], w: int) -> list[int]:
    a = [0] * 25
    for x in range(5):
        for y in range(5):
            a[x + 5 * y] = rot(b[y + 5 * ((2 * x + 3 * y) % 5)], -RHO[x][y], w)
    return a


def pi(a: list[int]) -> list[int]:
    b = [0] * 25
    for x in range(5):
        for y in range(5):
            b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y]
    return b


def pi_inv(b: list[int]) -> list[int]:
    a = [0] * 25
    for x in range(5):
        for y in range(5):
            a[x + 5 * y] = b[y + 5 * ((2 * x + 3 * y) % 5)]
    return a


def chi(b: list[int], w: int) -> list[int]:
    m = (1 << w) - 1
    return [b[x + 5 * y] ^ ((b[(x + 1) % 5 + 5 * y] ^ m) & b[(x + 2) % 5 + 5 * y]) for y in range(5) for x in range(5)]


def linear(a: list[int], w: int) -> list[int]:
    """L = pi . rho . theta, the round's linear part."""
    return rho_pi(theta(a, w), w)


def linear_inv(b: list[int], w: int) -> list[int]:
    return theta_inv(rho_pi_inv(b, w), w)


def keccak_round(a: list[int], rc: int, w: int) -> list[int]:
    s = chi(linear(a, w), w)
    s[0] ^= rc
    return s


def place(x: int, w: int, placement: str = "plane") -> list[int]:
    """P: input bits wj .. wj + w - 1 go to the j-th absorbed lane."""
    s = [0] * 25
    m = (1 << w) - 1
    for j, (lx, ly) in enumerate(PLACEMENTS[placement]):
        s[lx + 5 * ly] = (x >> (w * j)) & m
    return s


def unplace(s: list[int], w: int, placement: str = "plane") -> int:
    return sum(s[lx + 5 * ly] << (w * j) for j, (lx, ly) in enumerate(PLACEMENTS[placement]))


def step(s: list[int], x: int, w: int, placement: str = "plane", rc: int | None = None) -> list[int]:
    p = place(x, w, placement)
    return keccak_round([u ^ v for u, v in zip(s, p)], last_round_constant(w) if rc is None else rc, w)


def chain(s: list[int], xs: list[int], w: int, placement: str = "plane", rc: int | None = None) -> list[int]:
    """F_s(x): the state after absorbing xs from start state s."""
    for x in xs:
        s = step(s, x, w, placement, rc)
    return s


def twisted_step(s: list[int], x: int, w: int, rc: int | None = None) -> list[int]:
    """Ketje v2's step on the twisted state: absorb into plane y = 0, then pi . R . pi^-1."""
    p = place(x, w, "plane")
    u = pi_inv([a ^ b for a, b in zip(s, p)])
    return pi(keccak_round(u, last_round_constant(w) if rc is None else rc, w))


def keccak_f(a: list[int], w: int) -> list[int]:
    ell = w.bit_length() - 1
    for rc in round_constants(w, 12 + 2 * ell):
        a = keccak_round(a, rc, w)
    return a


def to_int(s: list[int], w: int) -> int:
    return sum(v << (w * i) for i, v in enumerate(s))


def from_int(v: int, w: int) -> list[int]:
    m = (1 << w) - 1
    return [(v >> (w * i)) & m for i in range(25)]
