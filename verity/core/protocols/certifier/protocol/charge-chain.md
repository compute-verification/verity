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

**The compile slice's multi-window form.** `CompileSlice_v1` (#3661) charges one window from public constants: its
`ChargeCount_v1{P·A}` takes the priors as part of the public book. Its multi-window form takes the chain as follows:

- `charge_in` and `charge_out` are the unit's reads of `replay-charge` rows 0 and 1 (`charge_chain.reads()`), next to
  the state and window rows it reads from `replay_chain`.
- The call is `charge_chain.build_link(C, N, book, row_in, row_out, en, bits)` in place of `build_charge`. `book` is
  `book_fields(N)`, the public tables without the carry, and `en` and `bits` are the scan's asks as now. Its ok goes
  into `code`, and its bits into `charge`, held to c_w as now.
- The verifier runs `charge_chain.check_chain` on the charge rows' openings beside `replay_chain.check_chain`.
- The prover commits each window with `commit_window(program, book, carry', registration, carry=previous)`.
- Where the replay is split into units, the charge sits in the load unit, over window row 1's `dec` with
  en_i = i < `left`. That is the scan's order, since `ReplayAdvance` reads the decisions at `dh` in turn and the next
  load refuses unread ones. There it is `ChargeLink_v1{D}`.

**Cost.** `ChargeCount_v1{N}` and 96 ANDs (the carry-out's equality and the ok): 19,632 at N = 2 and 50,760 at N = 14
(pinned). Per window: two rows, the charging unit's two reads (each a frame-v3 path of depth 1) and 256 record bytes.
