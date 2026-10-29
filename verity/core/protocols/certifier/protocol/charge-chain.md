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
  charge, the compile slice's earlier one, is the same only when no decision carries: it codes a carried decision in
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
  circuit is `build_link`. The replay's 228 carried bits (coder 68, KT counts 3 × 24 each for n0 and n1, `pc` 2,
  position 14) become 0 bits in the state row, and the chain's rows don't change.

**Cost.** `ChargeCount_v1{N}` and 96 ANDs (the carry-out's equality and the ok): 19,632 at N = 2, 50,760 at N = 14
(pinned), 97,159 at the class's D = 32, and 1,256,873 per ask at its P·A = 486. In a lowering, `ChargeCount_v1` adds copies of the words it carries from one
decision to the next, about 175 rows a decision (`size`'s ``carry``). Without them, each carried word's form, and so
the build, grows with every decision before it. With them, no AND operand is wider than 2,166 terms at any N. Per
window: two rows, the charging unit's two reads (each a frame-v3 path of depth 1) and 256 record bytes.

## The whole window (`window_charge`)

`ChargeCount_v1{N}` codes `code_v1`'s four other counts (n_hits, n_resumes, n_order, n_lag) as 0 (`ZERO_COUNTS`), so it
charges only a window that sends decisions alone. Advice v1 sends two of them:
- **n_hits** counts the hits sent past the pool bound: one shortfall, in hit units, for each lookup the replay reads.
- **n_order** counts the order entries: one for each arrival that deviates from release order, under the client's
  ingress.

The other two are always 0 under v1. The replay derives preemptions and the certified O forces each lag row, so `v1.py`
refuses a window that sends resumes or lag bits. `ChargeCount_v2{N, H, O}` charges the whole window, its decisions and
both counts with their values, as one `code_v1` segment with one count and one finish. With H = O = 0 its charge, counts
and run are `ChargeCount_v1{N}`'s; the tests check this on the fixture's windows and on random ones.

**The symbols**, in `code_v1._window`'s order. Each is a (cum, freq, total) step of code.py's 32-bit `ArithEncoder`,
counted by `ArithCountStep_v1{18, BIN, 20}`:
1. n_arrived = a, then each decision's KT bit at context min(run, 2) (BIN = 1), as in v1.
2. n_hits = h, then each hit's value s ∈ [0, 2^16].
3. n_resumes = 0.
4. n_order = o, then for each entry, with p its position and r its rank:
   - its gap g = p − pos ∈ [0, 2^24];
   - its rank step z = zigzag(r − rank) ∈ [0, 2^25 + 1], where zigzag(d) = 2d for d ≥ 0 and −2d − 1 for d < 0;
   - then pos := p + 1 and rank := r + 1.

   Each entry's order_len is 1, a table restricted to one bin: its symbol (0, f, f) changes no coder state, so it adds
   no bits.
5. n_lag = 0.
6. `ArithCountFinish_v1{20}`.

The window's charge is the bits counted, the length of `code_v1`'s segment for the window. pos and rank are
`code_v1._State`'s step and index, each + 1, so they start at 0.

**An integer** v ∈ [0, hi] under its table's frequencies f is `code.Binned.put(v, 0, hi)` (`binned_symbols`):
- **The bin.** Let j = bitlen(v), with bitlen(0) = 0, and top = bitlen(hi). The table is restricted to bins 0 … top,
  each with f_i ≥ 1, and the escape drops out because nothing feasible lies outside them. So the total is
  T = Σ_{i ≤ top} f_i ≤ 2^16, and the bin's symbol is (Σ_{i<j} f_i, f_j, T).
- **The offset.** u = v − a_j is uniform over the bin's n_j = min(2^j − 1, hi) − a_j + 1 values, with a_0 = 0 and
  a_j = 2^(j−1). It is coded as `code.put_uniform(u, n)`:
  - n ≤ 1: nothing;
  - 1 < n ≤ 2^16: (u, 1, n);
  - n > 2^16: put_uniform(⌊u / 2^15⌋, ⌊(n − 1) / 2^15⌋ + 1), then (u mod 2^15, 1, 2^15).

| value | hi | its table's bins in the unit | offset steps by bin j |
|---|---|---|---|
| a count over S slots (n_arrived, n_hits, n_order) | the book's cap for it | 0 … bitlen(S) | j ≤ 1 none; else one, n_j = 2^(j−1) |
| n_resumes, n_lag | the book's cap | 0 | none |
| a hit | 2^16 (`hit_units`) | 0 … 17 | j ≤ 1 none; 2 … 16 one; 17 none |
| a gap | 2^24 (`gap`) | 0 … 25 | j ≤ 1 none; 2 … 17 one; 18 … 24 two; 25 none |
| a rank step | 2^25 + 1 (2·`index` + 1) | 0 … 26 | j ≤ 1 none; 2 … 17 one; 18 … 25 two; 26 one (n = 2) |

- **T.** T is always the sum to the cap's top bin, even when the unit holds fewer bins.
- **A count's offset.** The book's cap for a count is at least 2^bitlen(S) − 1 (`window_book` refuses one that isn't), so
  n_j = 2^(j−1) for every bin a count of at most S reaches. The unit uses 2^bitlen(S) − 1 for hi, with the same n_j.
- **A wide offset's first chunk.** When n_j = 2^(j−1) > 2^16, that chunk is (⌊u / 2^15⌋, 1, 2^(j−16)).

**The book** (`window_book`) is the registered book's Exp-Golomb tables (`lo` 0, a frequency for every bin up to the
cap's), held to v1's value caps (`v1.TABLE_CAPS`: hit_units 2^16, gap and index 2^24, order_len 1), with
0 ≤ N, H, O < 2^16. The unit's public inputs are each symbol's frequencies at the bins it reaches, and T, 17 bits each.

**The circuit.** For each value:
- A one-hot of bitlen(v) comes from its prefix ORs.
- (cum, freq) are selected from the table's 22-bit prefix sums.
- The uniform step's total and enable are constants of the bin, selected by XORs.
- A bin wider than 2^16 takes the second step.

A disabled slot changes nothing. A field's count is its enabled slots, and a field with no slots has no inputs and a
count of 0.

**The carry** is v1's 92 bits (n0[3], n1[3], run), then pos (32 bits) and rank (25 bits). Each field is least significant
bit first, and the 149 bits are zero-padded to a 160-bit row (`ROW_BITS`). The session's first carry is the book's priors,
run 0, pos 0 and rank 0. The carry is chained as `replay-charge`'s above, under its own produced value `window-charge`:
- two rows, the carry-in and the carry-out;
- window 0's carry-in under the value's public salt;
- the verifier's `CH-first`, `CH-carry` and `CH-rows` (`window_charge.check_chain`).

`ChargeLink_v2{N, H, O}` is `ChargeCount_v2` run from row 0, and its ok also requires that carry' is row 1. After each
decision, each hit, and each order entry's gap and rank step, the carried words are committed. So no AND operand
widens with the slots before it, and the build is linear in N + H + O.

**Refusals** (ok = 0):
- `ChargeCount_v1`'s: a model's, the coder's (a zero frequency, cum + freq past the total, more than 18 scalings) and
  a count past 2^20;
- a carry-in run past 2 or rank past 2^24 + 1;
- an enabled hit past 2^16;
- an enabled entry with p < pos, a gap past 2^24, r past 2^24, or p = 2^32 − 1, whose pos wouldn't fit;
- a selected cumulative frequency past 2^16.

**The bound.** The unit's charge is `code_v1`'s segment length for the whole window, so "The bound" (PROTOCOL.md) covers
every advice v1 window, not just decisions-only ones: `encode` is injective on the values and its strings are
prefix-free. The sampled leak bound takes the count this gives, at most `2^c` advice values charged `c` or less, as its
premise (`hAs` in `Security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean`). It rests on two facts:
- *Each count's code is prefix-free under a public prior.* The prior is the registered table, fixed before the session.
  The ideal length of v is ℓ(v) = −log2(f_j / T) + log2 m_j. Here m_j is n_j when n_j ≤ 2^16, and
  (⌊(n_j − 1) / 2^15⌋ + 1) · 2^15 ≥ n_j for the two chunks. So

  \[ \sum_{v=0}^{hi} 2^{-\ell(v)} = \sum_{j \le top} \frac{f_j}{T} \cdot \frac{n_j}{m_j} \le 1 \]

  Equality holds under v1's caps. They are powers of two (with 2·`index` + 1 for the rank step), so every n_j is at
  most 2^16 or a power of two at least 2^17, and m_j = n_j.
- *The window's symbols form a prefix-free sequence code given the carry-in.* Each symbol's model is fixed by the book,
  the caps and the symbols before it:
  - a count comes before the values it counts and fixes how many follow;
  - a decision's (f0, f1) comes from the carried KT counts and the decisions before it;
  - each value's range is a constant;
  - a uniform step's total is fixed by its bin.

  So `code_v1._window` decodes the symbols back one by one, and no window's sequence is a proper prefix of another's.
  At each step the coder splits its interval among the alphabet in proportion to the frequencies, which sum to at most
  the total: bins 0 … bitlen(S) of a count's table reach at most T.

The Lean definitions (`Security/Definitions/NetworkCertifier/WindowCharge.lean`) mirror `reference_window` one for one,
and the window difftest holds them to it: the certifier program's `NetworkCertifier.WindowCharge`, proved equal to them
(`CodeChargesWhole`), gives what `reference_window` gives on 4,000 random instances (`lean/scripts/window_vectors.py`).

**The slice.** The compile slice's class runs inside the pool bound with the network certifier's ingress, so its windows send no
hits and no order entries. It keeps `ChargeLink_v1{D}` over `replay-charge` as above, and nothing in its wiring changes.
A class whose windows send hits or order entries does the following:
- It calls `window_charge.build_link(C, N, H, O, book, row_in, row_out, en, bits, hen, short, oen, p, r)` over
  `window-charge`'s rows (`window_charge.reads()`), in place of `charge_chain.build_link` over `replay-charge`'s.
- Its `hen` and `oen` are the replay's, as `en` is (en_i = i < `count`): a hit slot for each lookup past the pool
  bound, an order slot for each arrival out of release order, enabled slots first, never the prover's. Nothing reads a
  disabled slot's value: the charge counts only enabled slots, so a slot arrangement or a disabled value the prover
  chose would pass log2(C(H, h) · C(O, o)) bits and those values uncharged (`test_network_certifier_window_charge_slots.py`:
  every arrangement and disabled values past every cap give one charge and one carry).
- It holds the returned bits to c_w as now. The tests check this inline: c_w equal to the charge accepts, and one bit
  less refuses.
- Its H and O are fixed per class, like D. A window that sends more hits or entries than its slots can't be charged,
  so that window's unit fails.

**Cost** (`size`, ANDs). `window_charge.circuit-check.json` pins C-Flock's lowering, which adds one row for each commit
copy of the carry: 53,734 at `{1, 1, 1}` and 40,280 at `{2, 2, 0}`.

| unit | ANDs |
|---|---|
| `ChargeCount_v2{2, 0, 0}` | 19,560 (`ChargeCount_v1{2}` and 24 for the rank check) |
| `ChargeCount_v2{1, 1, 1}` | 53,287 |
| `ChargeCount_v2{2, 2, 0}` | 39,803 |
| `ChargeCount_v2{2, 0, 2}` | 74,901 |
| `ChargeCount_v2{4, 2, 2}` | 100,586 |
| `ChargeLink_v2` | the count's and 160 for the link |

A hit slot costs about 8,600 (two steps of about 4,000 each and its selection). An order slot costs about 25,800 (six
steps and their selections). A count over two or more slots adds its offset step.
