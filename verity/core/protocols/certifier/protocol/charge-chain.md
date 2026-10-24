Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The charge's carry across windows (`charge_chain`)

`code_v1` carries the decisions' KT counts and their context min(run, 2) from window to window. So a window's charge
depends on every decision before it, and `ChargeCount_v1{N}` takes that carry (`n0[3]`, `n1[3]`, `run[2]`) as an
input and returns the next window's. The session's first carry is the registered book's priors and run 0, public.
Every later carry is a function of the session's decisions so far: a value, committed like any other.

**The chain.** Each window has a produced value `replay-charge` of two hm96 rows (frame-v3-sha512, `hm96-sha512/row/v2`,
96 bits each: n0 3×15, n1 3×15, run 2, zero padding): its carry-in (row 0) and its carry-out (row 1).

- Window w's row 0 is window w − 1's row 1 under the same salt, so its opening b‖c is the same.
- Window 0's row 0 is the priors' row under the value's public salt (`registered.supplied_salts`), whose opening the
  verifier computes from the registered book.
- Row 1 is under a fresh salt.
- The charging unit reads both rows. `ChargeLink_v1{N}` is `ChargeCount_v1{N}` run from row 0's carry, and its ok also
  requires that the charge's carry' is row 1.
- The verifier checks the openings on every session (`check_chain`): `CH-first`, window 0's carry-in isn't the priors
  under their public salt; `CH-carry`, a window's carry-in isn't the previous window's carry-out; `CH-rows`, a window
  isn't two rows.

**A restart is refused.** By induction over the windows, the carry each window is charged from is the one `code_v1`
carries over the whole session. Window 0's carry-in is the priors (`CH-first`). Each window's carry-out is its charge's
carry' from its carry-in (the unit's ok). Each later carry-in is the previous carry-out (`CH-carry`). A prover that
restarts the counts mid-session, at the priors or at counts fitted to the coming window, either breaks the chain, which
the verifier refuses, or claims a unit whose ok is 0, which fails that unit's proof. On the fixture's
async-kv-pressure stream, window 1 costs 23 bits from window 0's carry, 20 from the priors and 16 from fitted counts.
The tests check that each forged carry is refused, whether it is committed under a fresh salt or under the public
start salt, and whether window 0's carry-out claims it.

**Why a value of its own.** The carry-out isn't a field of `replay_chain`'s window row. `ReplayLoad` writes that row
and checks it in full at the window's start, before any decision is coded, so a carry-out there would need ReplayLoad
to leave a field unchecked and change its pins. A sibling value under the same carry rule gives the same guarantee and
leaves the scan's chain as it is.

**The compile slice takes the chain** ([compile-slice.md](compile-slice.md)):

- ``cin`` and ``cout`` are the unit's reads of `replay-charge` rows 0 and 1 (`charge_chain.reads()`).
- Its charge is `charge_chain.build_link(C, D, book, cin, cout, en, bits)`, `ChargeLink_v1{D}`, over the decisions
  the window loads (next section). `book` is `book_fields(D)`, the public tables without the carry. `bits` are `adv`'s
  D decisions, and en_i = i < its `count`. Its ok is the verdict's `code` bit, and its bits are held to c_w (`charge`).
- The verifier runs `charge_chain.check_chain` on the charge rows' openings.
- The prover commits each window with `commit_window(program, book, carry', registration, carry=previous)`.

## One carry for units and windows

The charge's one carried representation is `replay-charge`'s two rows above, and `replay_chain`'s state row holds none
of it: the charge runs whole in the window's load unit, so no step unit carries a coder state, KT counts, a context or
a code position.

- **Where.** The load unit (unit 0, `ReplayLoad`'s) runs `ChargeLink_v1{D}` from row 0 to row 1 on the decisions it
  loads: `adv`'s `dec`, en_i = i < `count`, bit_i = dec_i. Those are the window's own decisions, the ones A5 charges to
  it. The carried decisions at the front of window row 1's `dec` were charged by the window that loaded them.
  `code_v1` codes the decisions in session order, the order the loads append them, and a slot with en 0 changes
  nothing (the tests check this on the fixture's 41 windows at D = 32). So the load's charge is `code_v1`'s. A per-ask
  charge, the compile slice's before #3886, is the same only when no decision carries: it codes a carried decision in
  the window that reads it.
- **The carry, in order:** `n0[3]` and `n1[3]` (15 bits each, `code_v1.KT`'s counts, both halved once
  2·(n0 + n1) + 2 would pass 2^16), `run` (2 bits, min(run, 2)), then 4 zero bits: 96. `run` is `code_v1`'s context,
  the decisions since the last 1, over decisions only, so a forced admission doesn't reset it as it does the replay's
  `pc`. The session starts at the registered book's priors and run 0, under the value's public salt. Every window
  carries all of it.
- **What resets.** The count's state (low 32, high 32, used 1, ok 1, bits 20) starts afresh in each window's unit:
  n_arrived (en's popcount) first, then each decision (`BookKT_v1{3}` under `run`, `ArithCountStep_v1{18, 1, 20}` on
  its frequencies, `run`'s update), then `ZERO_COUNTS`' four 0s and `ArithCountFinish_v1{20}`. None of it leaves the
  unit. There is no segment and no position: `ArithCountStep` counts the bits a segment would hold, so the slice has
  no ``seg`` and calls neither `ArithEncodeStep_v1` nor `ArithEncodeFinish_v1`.
- **The end check** is in the same unit: its ok (each model's and the count's, the finish's, and carry' = row 1)
  joins ``code``, and bits ≤ c_w is ``charge``. `ReplayClose` checks no charge.
- **The reference** is `charge_chain.reference_link` (`arith_encode.reference_charge` with carry' = row 1), and the
  circuit is `build_link`. ENC-REPLAY's 228 carried bits (coder 68, KT counts 3 × 24 each for n0 and n1, `pc` 2,
  position 14) become 0 bits in the state row, and #3801's rows don't change.

**Cost.** `ChargeCount_v1{N}` and 96 ANDs (the carry-out's equality and the ok): 19,632 at N = 2, 50,760 at N = 14
(pinned), 97,159 at the class's D = 32, and 1,256,873 per ask at its P·A = 486. In a lowering, `ChargeCount_v1` adds copies of the words it carries from one
decision to the next, about 175 rows a decision (`size`'s ``carry``). Without them, each carried word's form, and so
the build, grows with every decision before it. With them, no AND operand is wider than 2,166 terms at any N. Per
window: two rows, the charging unit's two reads (each a frame-v3 path of depth 1) and 256 record bytes.
