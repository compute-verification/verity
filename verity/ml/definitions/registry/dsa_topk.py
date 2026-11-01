"""[GLM-5.3] The DSA indexer's top-k of a row past TOPK keys, under the pattern for data-dependent structure: a static
superset of wires plus selection values computed in the circuit.

* The selected-key order is a value: the served kernel's `topk_indices` row, whose order (the above-threshold keys' places come
  from shared-memory atomics) no circuit computes.  The Program takes it from its `run` parameter (`derive.RunInputs`).
* The selected rows are read through `DsaWaksmanRows`, a Waksman permutation network over the row's whole history (the static
  candidate superset): its switch settings are the run's value (`waksman_settings` of the served order), and the network's wiring
  is static.  `DsaWaksmanOrder` routes each history row's index through the same settings, so its output is the order the rows
  arrive in.
* `DsaTopKOrder` checks in gates that this order is a permutation of the top-TOPK set the circuit computes from the row's logits,
  under the tie rule below; its bit is one of the Program's checks, which the verifier requires to be 1.

So wrong settings can only route an order the relation accepts, a permutation of the right set, which changes only the order
sparse MLA accumulates the rows in; this is not a witness exemption, since the relation is checked in the circuit.

The served kernel (vLLM d9105ea80): decode takes `cooperative_topk` at most 64 rows a call (sparse_attn_indexer.py:615-642),
whose rows of 2049..16384 keys run `histogram_4096_topk` (topk_histogram_4096.cuh:380-548): a 4096-bin histogram of each logit's
fp16 top 12 bits (`extract_coarse_bin_N`, monotone in the logit, -0 below +0), the keys above the threshold bin written in
`atomicAdd` order, and the threshold bin's resolved by a warp ballot on `(score >, then idx <)` at most 64 ties, else by a 4-round
radix on `convert_to_uint32_v2` (cuh:71-74, also persistent_topk.cuh:50-53).

The tie rule (`topk_set`): key(s) = `convert_to_uint32_v2`(s) = ~bits for a negative word, bits | 2^31 otherwise; j precedes j'
when key_j > key_j', or key_j = key_j' and j < j'.  The top-TOPK set is the TOPK first indices.  Within one coarse bin the
ballot's float order is this key order (±0 never share a bin), so the kernel selects this set but where its row has
  - exactly equal logits straddling the boundary in a threshold bin of more than 64 keys (the radix places equal keys by atomics),
  - more than TOPK keys in the threshold bin (its tie buffer keeps the first TOPK by atomics), or
  - a NaN or an infinity in the threshold bin (the ballot's float comparison is no order there);
where it does not, the relation refuses the served order: a visible failure, never a wrong acceptance.  `persistent_topk` (more
than 64 rows) places ties the same way (below); the prefill's `top_k_per_row_prefill` keys by
`convert_to_uint32_v2` too, but its tie placement is unread.  `top_k_per_row_decode`, which vLLM does not call on sm_90, places
ties otherwise: the relation refuses its order on rows with ties at the boundary.

On an H200 (sm_90), on synthetic rows of 2049..16384 keys, `cooperative_topk` at 1, 8 and 64 rows a
call and `persistent_topk` emit orders the relation accepts on every random, bf16-rounded, ballot-tie and signed-zero row, and
orders it refuses on only the exceptions above (65, 100 and 1000 equal keys straddling the boundary, and a row of 3000 equal
keys); two identical calls gave different orders on every row.

The network is the arbitrary-size Waksman network (Beauquier and Darrot's AS-Waksman) of N rows with its outputs past K pruned.
Of n >= 3 rows, h = n // 2: an input column of h switches, switch i over rows 2i and 2i+1 sending its first output to the top
subnetwork's input i and its second to the bottom's (row n - 1 of an odd n goes to the bottom's input h unswitched); a top
subnetwork of h rows and a bottom one of n - h; and an output column, switch j over the subnetworks' outputs j giving outputs 2j
and 2j + 1 (an odd n's output n - 1 is the bottom's output h; an even n's last switch is fixed straight).  Two rows are one
switch.  Outputs at or past K are dropped with every switch that feeds only them (`_sub_live`): a switch with one live output is a
pick.  A switch is `Select`(s, b, a) for its first output and the XOR of a, b and that output for its second (`Xor3x16`), so
its settings bit s is 1 for crossed.  The settings are laid out input column, top, bottom, output column, recursively; there are
`waksman_switches`(N, K): N ceil(log2 N) - 2^ceil(log2 N) + 1 unpruned, 20,493 at N = 2049 (K = 2048 prunes none).  Every
permutation has settings (the column's constraints form paths and even cycles, two-coloured by `_route`).

Registered functions (`_v1`; their Boolean versions in `boolean_dsa_topk` are the definitions, these the accelerators tests hold
equal to them):
  DsaTopKOrder{N,TOPK}        (scores: N f32, order: TOPK i32) -> bit: order is a permutation of `topk_set`(scores, TOPK)
  DsaTopKOrderCheck{N}x{K}    the same function as one primitive (`order_check`), DsaTopKOrder's body
  Xor3x16, Xor3x32            (a, b, c: 16 or 32 bits) -> a ^ b ^ c
  DsaWaksmanSwap{W,E}         (s: bit, a, b: W x E bits) -> 2 x W x E bits: (a, b) when s is 0, (b, a) when 1
  DsaWaksmanPick{W,E}         (s: bit, a, b: W x E bits) -> W x E bits: a when s is 0, b when 1
  DsaWaksmanRows{N,K,W,E}     (sw: waksman_switches(N, K) bits, rows: N x W x E bits) -> K x W x E bits: the network at sw;
                              at `waksman_settings`(order, N), slot s is rows[order[s]]
  DsaWaksmanOrder{N,TOPK}     (sw: waksman_switches(N, TOPK) bits) -> TOPK i32: `DsaWaksmanRows{N,TOPK,1,32}` of the indices 0..N-1
"""
from __future__ import annotations

import functools

from verity.ml.circuits.defs import PrimitiveDefinition, bind, composite, primitive
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.refs import Coll, concat, strided_view
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.registry.prims import BF16, BIT, F32, I32, SelectBf16, SelectI32

M32 = 0xFFFFFFFF

_CONF = ("model-tested on synthetic rows: the tie rule read from vLLM d9105ea80 topk_histogram_4096.cuh:69-84, 380-548 "
         "(cooperative_topk's rows of 2049..16384 keys) and held by the served cooperative_topk's and persistent_topk's orders "
         f"({EVIDENCE['dsa_topk_order_probe']['run_record']}); a captured served decode past TOPK keys is its hardware check")
_ORDERS: dict[tuple[int, int], PrimitiveDefinition] = {}


def topk_key(bits: int) -> int:
    """`convert_to_uint32_v2` on an f32 word: an unsigned key whose order is the logits' (a larger float a larger key, -0 below
    +0, a positive NaN above +inf and a negative one below -inf)."""
    bits &= M32
    return ~bits & M32 if bits & 0x80000000 else bits | 0x80000000


def topk_set(scores, TOPK: int) -> set[int]:
    """The TOPK indices first under the tie rule: key descending, then index ascending."""
    return set(sorted(range(len(scores)), key=lambda j: (-topk_key(scores[j]), j))[:TOPK])


def order_ok(scores, order, TOPK: int) -> int:
    """1 when the TOPK words ``order`` (u32) are distinct and are `topk_set`(scores, TOPK), else 0."""
    got = [int(w) & M32 for w in order]
    if len(got) != TOPK:
        raise ValueError(f"an order of {len(got)} slots for TOPK {TOPK}")
    return int(len(set(got)) == TOPK and set(got) == topk_set(scores, TOPK))


def check_statics(N: int, TOPK: int) -> None:
    if not 1 <= TOPK <= N < 1 << 31:
        raise ValueError(f"DsaTopKOrder: a row of N keys selects 1 <= TOPK <= N (N={N}, TOPK={TOPK})")


def order_check(N: int, TOPK: int) -> PrimitiveDefinition:
    """`DsaTopKOrderCheck{N}x{TOPK}_v1`: `order_ok` over a row of N f32 scores and a TOPK-slot order."""
    check_statics(N, TOPK)
    p = _ORDERS.get((N, TOPK))
    if p is None:
        def ev(*w: int, _N=N, _K=TOPK) -> int:
            return order_ok(w[:_N], w[_N:], _K)

        p = PrimitiveDefinition(f"DsaTopKOrderCheck{N}x{TOPK}", 1, (("scores", Array(N, F32)), ("order", Array(TOPK, I32))), BIT,
                                ev, doc=order_check.__doc__ or "", conformance=_CONF)
        _ORDERS[(N, TOPK)] = p
    return p


@composite("DsaTopKOrder", 1, ["N", "TOPK"], lambda S: ((("scores", Array(S.N, F32)), ("order", Array(S.TOPK, I32))), BIT),
           conformance=_CONF)
def DsaTopKOrder(B, S, scores, order):
    """The selection relation of a row of N keys: 1 when ``order``'s TOPK slots are distinct indices in [0, N) forming the
    top-TOPK set of ``scores`` under the module docstring's tie rule, in any order; else 0.  `DsaTopKOrder_v2` is it on bits."""
    return B.call(order_check(S.N, S.TOPK), scores, order)


def _sub_live(n: int, k: int) -> tuple[int, int, list[int]]:
    """Of a network of n >= 3 rows whose outputs 0..k-1 are live: the top's and the bottom's live outputs (each a prefix) and the
    output switches that have a setting, ascending (those feeding a live output; an even n's last is fixed)."""
    h, c = n // 2, -(-k // 2)
    k2 = min(h, c) + (k == n) if n % 2 else min(h, c) - (k == n - 1)
    return min(h, c), k2, [j for j in range(h - (n % 2 == 0)) if 2 * j < k]


@functools.lru_cache(maxsize=None)
def waksman_switches(n: int, k: int) -> int:
    """The settings bits of a network of n rows with outputs 0..k-1 live."""
    if n <= 2:
        return n - 1
    k1, k2, outs = _sub_live(n, k)
    return n // 2 + waksman_switches(n // 2, k1) + waksman_switches(n - n // 2, k2) + len(outs)


def _route(n: int, dest: list[int]):
    """The settings that send row p to output dest[p] (a permutation of range(n)): None for one row, the bit for two, else (input
    column, top's, bottom's, output column).  Row p's colour is the subnetwork it crosses (0 top): a switch's two rows, and the
    two rows bound for one output switch, differ; an odd n's last row and the row bound for its last output go bottom; the row
    bound for an even n's output n - 2 goes top (its last switch is fixed)."""
    if n == 1:
        return None
    if n == 2:
        return dest[0]
    h = n // 2
    src = [0] * n
    for p, d in enumerate(dest):
        src[d] = p
    colour = [-1] * n

    def paint(p: int, c: int) -> None:
        stack = [(p, c)]
        while stack:
            p, c = stack.pop()
            if colour[p] == c:
                continue
            if colour[p] != -1:
                raise AssertionError(f"Waksman routing: row {p} of {n} is coloured both ways")
            colour[p] = c
            if p < 2 * h:
                stack.append((p ^ 1, 1 - c))
            if dest[p] < 2 * h:
                stack.append((src[dest[p] ^ 1], 1 - c))

    if n % 2:
        paint(n - 1, 1)
        paint(src[n - 1], 1)
    else:
        paint(src[n - 2], 0)
    for p in range(n):
        if colour[p] == -1:
            paint(p, 0)
    top, bot = [0] * h, [0] * (n - h)
    for p, d in enumerate(dest):
        if colour[p]:
            bot[p // 2 if p < 2 * h else h] = d // 2 if d < 2 * h else h
        else:
            top[p // 2] = d // 2
    return [colour[2 * i] for i in range(h)], _route(h, top), _route(n - h, bot), [colour[src[2 * j]] for j in range(h)]


def _bits(n: int, k: int, tree) -> list[int]:
    if n <= 2:
        return [] if n == 1 else [tree]
    ins, top, bot, outs_ = tree
    k1, k2, outs = _sub_live(n, k)
    return ins + _bits(n // 2, k1, top) + _bits(n - n // 2, k2, bot) + [outs_[j] for j in outs]


def waksman_settings(order, N: int) -> list[int]:
    """The `DsaWaksmanRows{N,K}` settings (K = len(order)) that put rows[order[s]] at output s, the rows no slot selects at
    outputs K..N-1 in ascending order (outputs the network drops)."""
    got = [int(x) for x in order]
    if not 1 <= len(got) <= N or len(set(got)) != len(got) or not all(0 <= x < N for x in got):
        raise ValueError(f"a Waksman order is 1..{N} distinct indices in [0, {N}), not {len(got)} of them")
    dest: list = [None] * N
    for s, p in enumerate(got):
        dest[p] = s
    rest = iter(range(len(got), N))
    return _bits(N, len(got), _route(N, [d if d is not None else next(rest) for d in dest]))


def waksman_apply(sw, rows: list, k: int) -> list:
    """The network of len(rows) rows with outputs 0..k-1 live at settings ``sw``, on Python values: `DsaWaksmanRows`' reference."""
    n = len(rows)
    if len(sw) != waksman_switches(n, k):
        raise ValueError(f"{len(sw)} settings for a network of {n} rows, {k} live")
    if n <= 2:
        return list(rows) if n == 1 else (rows[::-1] if sw[0] else list(rows))[:k]
    h = n // 2
    k1, k2, outs = _sub_live(n, k)
    s1, s2 = waksman_switches(h, k1), waksman_switches(n - h, k2)
    pairs = [(rows[2 * i + 1], rows[2 * i]) if sw[i] else (rows[2 * i], rows[2 * i + 1]) for i in range(h)]
    top = waksman_apply(sw[h:h + s1], [a for a, _ in pairs], k1)
    bot = waksman_apply(sw[h + s1:h + s1 + s2], [b for _, b in pairs] + list(rows[2 * h:]), k2)
    out = []
    for x, j in zip(sw[h + s1 + s2:], outs):
        out += [bot[j], top[j]] if x else [top[j], bot[j]]
    if n % 2 == 0 and k >= n - 1:
        out += [top[h - 1]] + ([bot[h - 1]] if k == n else [])
    if n % 2 and k == n:
        out.append(bot[h])
    return out[:k]


@primitive("Xor3x16", 1, [("a", BF16), ("b", BF16), ("c", BF16)], BF16, conformance="exact-model-tested")
def Xor3x16(a: int, b: int, c: int) -> int:
    return (a ^ b ^ c) & 0xFFFF


@primitive("Xor3x32", 1, [("a", I32), ("b", I32), ("c", I32)], I32, conformance="exact-model-tested")
def Xor3x32(a: int, b: int, c: int) -> int:
    return (a ^ b ^ c) & M32


_SELECT, _XOR3 = {16: SelectBf16, 32: SelectI32}, {16: Xor3x16, 32: Xor3x32}


def _refuse_row(b: dict) -> str | None:
    if b.get("E") is not None and b["E"] not in (16, 32):
        return f"rows of 16- or 32-bit words (E={b['E']})"
    if b.get("W") is not None and b["W"] < 1:
        return f"rows of W >= 1 words (W={b['W']})"
    return None


def _refuse_net(b: dict) -> str | None:
    N, K = b.get("N"), b.get("K", b.get("TOPK"))
    if N is not None and N < 2:
        return f"a network of N >= 2 rows (N={N})"
    if N is not None and K is not None and not 1 <= K <= N:
        return f"1 <= K <= N live outputs (N={N}, K={K})"
    return _refuse_row(b)


def _row(S):
    return Array(S.W, Value(S.E))


def _sig_switch(S, two: bool):
    R = _row(S)
    return (("s", BIT), ("a", R), ("b", R)), Array(2, R) if two else R


@composite("DsaWaksmanSwap", 1, ["W", "E"], lambda S: _sig_switch(S, True), conformance="exact-model-tested", refuse=_refuse_row)
def DsaWaksmanSwap(B, S, s, a, b):
    """One switch over two rows of W E-bit words: (a, b) when s is 0, (b, a) when 1.  The first output is `Select`(s, b, a) a word,
    the second `Xor3x<E>`(a, b, first)."""
    lo = B.batch(_SELECT[S.E], s, b, a, axes=(None, 0, 0))
    return Coll(Array(2, _row(S)), concat([lo.refs, B.batch(_XOR3[S.E], a, b, lo, axes=(0, 0, 0)).refs]))


@composite("DsaWaksmanPick", 1, ["W", "E"], lambda S: _sig_switch(S, False), conformance="exact-model-tested", refuse=_refuse_row)
def DsaWaksmanPick(B, S, s, a, b):
    """A switch whose second output is dropped: a when s is 0, b when 1 (`Select`(s, b, a) a word)."""
    return B.batch(_SELECT[S.E], s, b, a, axes=(None, 0, 0))


def network(B, N: int, K: int, R, sw, rows, swap, pick, sub):
    """`DsaWaksmanRows`' body over its callees, for the word and the Boolean versions: ``swap`` and ``pick`` the switches over two
    rows of type R, and ``sub(n, k)`` the network of n >= 2 rows with k live outputs."""
    if N == 2:
        return B.call(swap, sw[0], rows[0], rows[1]) if K == 2 else Coll(Array(1, R), B.call(pick, sw[0], rows[0], rows[1]).refs)
    L, h = R.leaves, N // 2
    k1, k2, outs = _sub_live(N, K)
    s1, s2 = waksman_switches(h, k1), waksman_switches(N - h, k2)
    col = B.batch(swap, sw[0:h], Coll(Array(h, R), strided_view(rows, 0, L, 2 * L, 1, h * L)),
                  Coll(Array(h, R), strided_view(rows, L, L, 2 * L, 1, h * L)), axes=(0, 0, 0))
    top = Coll(Array(h, R), strided_view(col, 0, L, 2 * L, 1, h * L))
    bot = Coll(Array(N - h, R), concat([strided_view(col, L, L, 2 * L, 1, h * L), *([rows[N - 1].refs] if N % 2 else [])]))
    t = top if h == 1 else B.call(sub(h, k1), sw[h:h + s1], top)
    u = B.call(sub(N - h, k2), sw[h + s1:h + s1 + s2], bot)
    o, m = h + s1 + s2, sum(2 * j + 1 < K for j in outs)
    parts = [B.batch(swap, sw[o:o + m], t[0:m], u[0:m], axes=(0, 0, 0)).refs] if m else []
    if len(outs) > m:
        parts.append(B.call(pick, sw[o + m], t[m], u[m]).refs)
    if N % 2 == 0 and K >= N - 1:
        parts += [t[h - 1].refs] + ([u[h - 1].refs] if K == N else [])
    if N % 2 and K == N:
        parts.append(u[h].refs)
    return Coll(Array(K, R), concat(parts))


def _sig_rows(S):
    R = _row(S)
    return (("sw", Array(waksman_switches(S.N, S.K), BIT)), ("rows", Array(S.N, R))), Array(S.K, R)


@composite("DsaWaksmanRows", 1, ["N", "K", "W", "E"], _sig_rows, conformance="exact-model-tested", refuse=_refuse_net)
def DsaWaksmanRows(B, S, sw, rows):
    """The Waksman network of N rows of W E-bit words with outputs 0..K-1 live, at settings ``sw`` (the module docstring's
    network): at `waksman_settings`(order, N), slot s is rows[order[s]]."""
    return network(B, S.N, S.K, _row(S), sw, rows, bind(DsaWaksmanSwap, W=S.W, E=S.E), bind(DsaWaksmanPick, W=S.W, E=S.E),
                   lambda n, k: bind(DsaWaksmanRows, N=n, K=k, W=S.W, E=S.E))


@composite("DsaWaksmanOrder", 1, ["N", "TOPK"], lambda S: ((("sw", Array(waksman_switches(S.N, S.TOPK), BIT)),), Array(S.TOPK, I32)),
           conformance="exact-model-tested", refuse=_refuse_net)
def DsaWaksmanOrder(B, S, sw):
    """The order `DsaWaksmanRows{N,K=TOPK}` reads its rows in at settings ``sw``: the network over row j's index j (`Const32`), so
    slot s is the index of the row slot s holds."""
    tags = Coll(Array(S.N, Array(1, I32)), concat([B.call(const(32, j)).refs for j in range(S.N)]))
    return Coll(Array(S.TOPK, I32), B.call(bind(DsaWaksmanRows, N=S.N, K=S.TOPK, W=1, E=32), sw, tags).refs)


DSA_TOPK_COMPOSITES = [DsaTopKOrder, DsaWaksmanSwap, DsaWaksmanPick, DsaWaksmanRows, DsaWaksmanOrder]
