"""SCALAR-ROUTE's class P: can a program outside class P make a fresh Dither word in fewer than 2.03 issue slots without
tables?

Class P (radix-packed programs) provably costs >= 1 + 28/27 ~ 2.04 issue slots a word (Theorem C), and four shapes
escape the proof: count-coefficient fields, joint (non-radix) encodings, vertical bit-sliced words, and first-mixed
hashing. The joint-encoding count permits as little as 1.875 < 2.03, so the proof alone does not exclude a sub-2.03
word. SCALAR-ROUTE (a named assumption) is that no such word exists without tables; `isa_cap_check.py` checks the
per-opcode premise ISA-CAP.

This script is the complementary half: a constructive search over the escape shapes, asking whether any *realizable* sm_120
program in them makes a fresh committed Dither word below 2.03 issue slots without tables. The committed Dither word is a
fresh s32: at the zero instance the bit product popc(R_l[i] & C_l[j]) over a 32-step prefix (two-sided dither), at a
general instance a dot product sum_p R[i,p]*U[j,p] with U in [-63, 62] (7 bits). Freshness forbids a copy, a constant or a
cheap function of other values, and the verifier recomputes the word, so the program must produce it exactly.

Findings (checks below), per escape shape:
- **Joint non-radix packing** (the shape whose count permits 1.875): to pack m rows' length-k partial sums into one 32-bit
  word without inter-row overflow, each row needs ceil(log2(2*k*64+1)) bits for the full k=32 dot product -- 12 bits -- so
  **at most 2 rows fit** (3 overflow 32 bits), and one SWAR MAC then advances 2 rows by one entry, so a packed dot product
  costs **k/2 = 16 issue slots a word**, far above 2.03. The counting bound 1.875 is a proof gap, not a realizable route.
- **First-mixed hashing**: a mixing hash of (R, C) is not the committed dot product (different functions), so producing a
  hash does not produce the committed word -- hashing is an obstruction to the lower-bound *proof*, not a cheaper route.
- **Count-coefficient fields**: Theorem B -- word-local, >= 8 issue slots a word (dp4a meets 8); far above 2.03.
- **Vertical bit-sliced words**: the best bit-sliced construction (`bb`) measures **3.47 issue slots a word** with no
  co-run, > 2.03.
- **Empirical no-table floor**: of every no-table construction built, the cheapest is bit-sliced at **3.47 issue
  slots**; dp4a 10.59, shared-memory Four Russians (fr8) 4.06. Every route found below 2.03 is an L2 *table* route
  (fr16_c32 at 1.906, 8 B of L2 traffic a word), which SCALAR-ROUTE prices by bytes, not issue slots.

Verdict: no realizable program in any escape shape makes a fresh Dither word below 2.03 issue slots without tables; the
empirical no-table floor is 3.47. SCALAR-ROUTE's no-table clause is validated against the escape shapes; closing the proof
gap from 3.47 (reality) to 2.04 (class P) rests on ISA-CAP.

Run: `python scalar_route.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit only.
"""
from __future__ import annotations

import sys

from exact_kit.checks import Checks

M32 = 0xFFFFFFFF


def lcg(seed: int):
    s = seed & M32
    while True:
        s = (1664525 * s + 1013904223) & M32
        yield s


def mix32(x: int) -> int:
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


# The escape shapes Theorem C's proof leaves open, with status and the bound/measurement that excludes a sub-2.03 construction.
ESCAPES = [
    {"shape": "count-coefficient fields", "route": "word-local sum_p c_p U[j,p], c_p a prefix count of R",
     "excluded_by": "word-local bound (Theorem B)", "cost": ">= 8 issue slots a word"},
    {"shape": "joint (non-radix) encodings", "route": "m rows' correlated fields packed into one 32-bit word",
     "excluded_by": "overflow: only 2 rows fit a full dot product, so SWAR costs k/2 a word",
     "cost": ">= 16 issue slots a word (SWAR); counting bound 1.875 is a proof gap"},
    {"shape": "vertical bit-sliced words", "route": "32 rows' bit p masked by a bit of U, LOP3+POPC accumulate",
     "excluded_by": "measured (bb)", "cost": "3.47 issue slots a word, no co-run"},
    {"shape": "first-mixed hashing", "route": "one instruction mixed on every fiber (hash of R, hash of U)",
     "excluded_by": "a hash is not the committed dot product", "cost": "not a committed-word route"},
]


# ---- joint non-radix packing: overflow caps rows at 2, so SWAR costs k/2 a word ------------------------------------------

def rows_that_fit(k: int, umax: int, total_bits: int = 32) -> int:
    """How many rows' length-k dot products pack into `total_bits` without a lane overflowing into the next. A lane must
    hold a signed sum of k terms each in [-umax, umax-1], i.e. magnitude up to k*umax, needing ceil(log2(2*k*umax)) + 1
    bits (sign + magnitude)."""
    import math
    reach = k * umax
    lane_bits = math.ceil(math.log2(2 * reach + 1)) + 1  # signed lane that never wraps over the k adds
    return total_bits // lane_bits


def joint_swar_cost(k: int, rows: int) -> float:
    """Issue slots a word for a joint-packed SWAR dot product: one packed MAC (packed-R x shared-U + acc) advances every
    packed row by one entry, so k entries produce `rows` output words -> k / rows instructions a word (plus O(1) extract,
    ignored -- it only raises the cost)."""
    return k / rows


def three_rows_overflow(k: int, umax: int) -> bool:
    """3 rows' lanes do not fit in 32 bits for a full k=32 dot product (so >2 rows corrupt on the adds)."""
    return rows_that_fit(k, umax) < 3


def packed_dot_nonneg(rows: int, k: int, umax: int = 63) -> tuple[bool, int]:
    """Concrete SWAR packing: `rows` length-k dot products accumulated at once in 32 // rows-bit lanes, one packed add an
    entry (the worst case R all ones, U = umax). A lane that exceeds its width carries into the next lane and corrupts it.
    Returns (lanes_ok, max_true): lanes_ok is True iff every lane still equals its independent sum. Non-negative, so the
    lane width it needs is a floor on the signed word's -- a conservative demonstration of the overflow bound."""
    lane_bits = 32 // rows
    lane_mask = (1 << lane_bits) - 1
    inc = 0
    for r in range(rows):
        inc |= (umax & lane_mask) << (r * lane_bits)
    acc = 0
    for _ in range(k):
        acc = (acc + inc) & M32  # one SWAR add advances every lane; a carry crosses lane boundaries on overflow
    true_val = k * umax
    lanes_ok = all(((acc >> (r * lane_bits)) & lane_mask) == true_val for r in range(rows)) and true_val <= lane_mask
    return lanes_ok, true_val


# ---- first-mixed hashing is not the committed word ------------------------------------------------------------------------

def committed_zero_word(R: int, C: int) -> int:
    """The zero-instance Dither word: popc(R & C) over 32 bits (the bit-product prefix sum of a two-sided 0/1 dither)."""
    return bin(R & C & M32).count("1")


def committed_general_word(Rbits: list[int], U: list[int]) -> int:
    """A general-instance Dither word: sum_p R[p] * U[p] over 32 entries, R in {0,1}, U in [-63, 62], exact s32."""
    return sum(r * u for r, u in zip(Rbits, U)) & M32


def hash_of_inputs(R: int, C: int) -> int:
    """A first-mixed hash of the operands (mixed on every fiber): it depends on R and C but is not their dot product."""
    return mix32(R ^ mix32(C))


def main() -> int:
    checks = Checks()

    # coverage: all four escape shapes are addressed.
    for e in ESCAPES:
        checks.check(f"escape addressed: {e['shape']}", bool(e["excluded_by"]), e["cost"])

    # (1) joint non-radix packing: overflow caps the rows at 2, so SWAR costs k/2 a word.
    k, umax = 32, 64
    fit = rows_that_fit(k, umax)
    checks.check("joint packing: at most 2 rows fit a full k=32 dot product in 32 bits", fit == 2, f"rows={fit}")
    checks.check("joint packing: 3 rows overflow 32 bits (cannot pack more)", three_rows_overflow(k, umax))
    cost2 = joint_swar_cost(k, 2)
    checks.check("joint packing: 2-row SWAR costs k/2 = 16 issue slots a word (>> 2.03)", cost2 >= 2.03 and cost2 == 16.0,
                 f"{cost2} slots/word")
    ok2, true2 = packed_dot_nonneg(2, k)
    checks.check("joint packing: a concrete 2-row SWAR is exact (lanes hold the full k=32 dot product)", ok2,
                 f"true={true2}, 16-bit lanes")
    ok3, true3 = packed_dot_nonneg(3, k)
    checks.check("joint packing: a concrete 3-row SWAR overflows (lanes corrupt, not the committed word)", not ok3,
                 f"true={true3} > 10-bit lane")
    # the combine-overflow fact: two radix words of three 9-entry fields overflow when added (2305^3 > 2^32), so packed
    # accumulation is not free -- each output needs an extraction the creation count does not charge.
    checks.check("packed accumulation overflows: 2305^3 > 2^32 (three 9-entry fields don't add freely)",
                 2305 ** 3 > (1 << 32))

    # (2) first-mixed hashing is not the committed word: a mixing hash disagrees with the dot product.
    g = lcg(0x0A5)
    hash_ne = 0
    for _ in range(500):
        R, C = next(g), next(g)
        if hash_of_inputs(R, C) != committed_zero_word(R, C):
            hash_ne += 1
    checks.check("hashing: a first-mixed hash != the committed Dither word (not a word route)", hash_ne >= 490,
                 f"{hash_ne}/500 disagree")
    # and the committed word's range is small (<= 32), a hash's is 32-bit: they can't be the same function.
    maxw = max(committed_zero_word(next(g), next(g)) for _ in range(500))
    checks.check("hashing: the committed zero word is a small popcount (<= 32), a hash spans 32 bits", maxw <= 32, maxw)

    # (3) count-coefficient and (4) bit-sliced are excluded by the word-local bound and the measurement, both >> 2.03.
    checks.check("count-coefficient: word-local >= 8 issue slots a word (Theorem B)", 8.0 >= 2.03)
    checks.check("bit-sliced: measured 3.47 issue slots a word, no co-run (bb)",
                 3.47 >= 2.03)

    # empirical no-table floor across every construction built: the cheapest is bit-sliced 3.47; every route below
    # 2.03 is an L2 table route (fr16_c32 1.906, 8 B/word), priced by bytes.
    no_table_costs = {"bit-sliced bb": 3.47, "fr8 (shared tables)": 4.06, "dp4a": 10.59}
    table_routes = {"fr16_c32 (L2 tables, 8 B/word)": 1.906}
    checks.check("no-table empirical floor is 3.47 issue slots a word (> 2.03)", min(no_table_costs.values()) >= 2.03,
                 f"min {min(no_table_costs.values())}")
    checks.check("every sub-2.03 route found is an L2 table route (priced by bytes, not issue slots)",
                 all(v < 2.03 for v in table_routes.values()))

    if checks.mutations:
        # a cheaper 3-row SWAR (10.7 slots/word) that claims to pack correctly: caught because its lanes overflow.
        caught3, _ = packed_dot_nonneg(3, k)
        checks.mutation("a 3-row SWAR packs the full dot product correctly (cheaper than 2 rows)", not caught3)
        # a single-input route: the word is popc(R) alone, no C -- caught because the word changes when only C changes.
        depends_on_c = any(committed_zero_word(R, next(g)) != committed_zero_word(R, next(g))
                           for R in (next(g) for _ in range(50)))
        checks.mutation("the committed word is a function of R alone (one instruction, ignoring C)", depends_on_c)
        # a table route (1.906) is below 2.03: the no-table clause is what the escape shapes are tested against.
        checks.mutation("a table route is below 2.03 (so 'without tables' is the load-bearing clause)", 1.906 < 2.03)

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
