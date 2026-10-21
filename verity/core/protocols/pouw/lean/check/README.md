# `pouw_check`: PoUW's window check as a core protocol

One `Core.Protocol` (`verity/core/protocols/interface`), `PouwCheck.protocol`, id `verity/pouw/v1`, Lean core only:
PoUW's verifier of record (`verity/core/protocols/pouw/verifier.py`), with `verity/core/protocols/pouw/core_check.py` the
Python reference held to it.

| field | here |
| --- | --- |
| `Params` | the verifier's own: the check item (`entries`, `scheme`, `template`, `tile_work`, and optionally `coins`, the window's coin mode, `live` or `seeded`), `eps` and `delta` as ratio strings, `grid_bits`, the declared `range` of tile counts, the window's `context`, the check query's SHA-512; the template one `PouwCheck.tiles` pins, with its tile work and query; and the rows' fixed `length` (`{max_tokens, stop}`, stop `fixed`: Daniel, 19 Oct, ASK-39) |
| `Commitment` | the window's registration, one-stage's record (`registration.record`) as canonical JSON, its window `{kind, context, public}` holding `window.REGISTERED`, whose `length` item must be a fixed-length window's at the verifier's max_tokens (`PouwWindow.lengthProblem`; a window without one, registered before the rule, is refused as `window-form: no-fixed-length`) |
| `digest` | the record's own: SHA-512(format ‖ 0 ‖ the canonical record), `registration.digest` |
| `Evidence` | the proof service's answer for a registration, read by `Core.Answer.ofJson` (protocol one-stage's): on acceptance the profile, read by `Core.Integrity.ofJson`, and the public outputs `{"draw": d}`, `d` the verifier's draw as one-stage's `draw.py` writes it (`subset`, `bernoulli`, `stratified` or `work`, with exactly `check_draw`'s keys) |
| `check` | `PouwCheck.check`: PoUW's rule over the registration (`registered`), then the service's answer to it (`outcome`); every refusal is the first rule it breaks, with `core_check.py`'s reason |
| `Public` | `window.PUBLIC`: `verdict`, `work`, `check`, `sampling`, `length`, `drawn`, `root_W` |
| `Profile` | the service's `Core.Integrity`, passed through once it is the registered partition's N units drawn subset:K′ at the window's δ, and the draw is subset:K′ over N |

| module | `core_check.py` |
| --- | --- |
| `PouwCheck.Json` | `sorted(dict)`, `_nat`, `_canon` |
| `PouwCheck.Tile` | `TILES`: each tile template the verifier accepts, with its tile work, its check query's SHA-512 and its one circuit's, and a salted template's rows (`Salted`) |
| `PouwCheck.Inputs` | `params`, `commitment`, `digest`, `unit_draw`, `evidence`, with their refusals (`answer_of` and `integrity` are the interface's decoders) |
| `PouwCheck.Check` | `expected_items` and `check` |
| `PouwCheck.Protocol` | the `Core.Protocol` value |
| `PouwCheck.Opening` | `opening`, `window_domain`, `verify_path`, `public_bc`: one unit's opening beside the check (`served_commit.check`'s rule), a certified window's under its certified salt |
| `PouwCheck.Pass` | `pass_epoch` and `draw`, with `coins_of`, `opening_of`, `coin_reason` and `weights_root` under `COIN_VERSIONS`: the window's pass and its draw (`served_commit.Challenger`'s rules) |
| `PouwCheck.Rounds` | `salt`, `rounds`: a live window's salt coin, opened once after its open root (`salt.window_open_root` of its record and the session the auditor's ledger reserved), and its round coins, each opened once after its recorded round root (`window.rounds_reason`'s rules, `PouwWindow.roundsReason`), both read through the challenger's session the ledger opens for the window (`SampledProofs.Challenger.auditedSession`, `sessionNamed`) |
| `PouwCheck.Difftest` | the difftest's functions behind `pouw-check-difftest` (`opening`, `registered`, `pass`, `draw`, `salt`, `rounds`) |
| `PouwCheck.DifftestMain` | `pouw-check-difftest`'s server, and its `coins n`: `n` bytes of `IO.getRandomBytes` (A3), the challenger's entropy (`verifier.coins`; `tests/test_coins.py`) |
| `PouwCheck.Credit` | `credit.py`'s `credited`, `drawn`, `outcome`, `window`, `period`, `period_credit` and `a_link`: a served window's credit bits fixed before its draw coin, its drawn units the draw coin's over its own layout, no partial credit over them, a period counted only closed and complete, and its A trees tied to its round roots (RT4-INT-32, -34 and -42) |
| `PouwCheck.CreditDifftest` | the difftest's `credit`, `drawn`, `outcome`, `window`, `period` and `a-link` behind `pouw-check-difftest` (`CreditDifftest.all`, the functions `DifftestMain` serves) |

`opening` holds a tile proof's public file to the registered window: the unit drawn (or opened undrawn, the filler's),
the entries root's domain one-stage's served domain of the registered program and partition, the opening's entry under
its salt that root's leaf, the unit one of the entry's tiles, the proof one instance over the circuit `PouwCheck.tiles`
pins for the check's template (its header's `circuit_sha512`, which `flock-verify verify --zk` binds to the circuit it
verifies against), and each opened row's b ‖ c its side tree's leaf at the unit's row. An unsalted template's proof's
first two ports, `a0` and `b0` (`PC8_PORTS`), are an A row and a B row of the fp8-p schemas. A salted template's
(`Salted`: today the H1 tile unit at K = 64, R = 4, C = 8) are its R A rows, its C B rows and its public row `pub`, each
`hm96-sha512/row/v2`, its entry's side trees holding R rows per tile row and C per tile column. Each schema is read by
port name, and C-Flock reads port k's b ‖ c at the body's k-th place, in the circuit's port order, so a header whose
keys, in key order (serde's), don't list the opened side rows first and in port order (`portsOf`) is refused, never
read in its own order (every pinned template's side ports sort in port order). The public row's
b ‖ c must be `publicBc` of the window's context, the certified salt the verifier passes (`opening`'s `salt`,
`core_check.certified_salt`, the salt `passEpoch` checked), the unit's position and the entry's side roots. Only a
certified window opens a salted template's units, and only under a 64-byte salt; it opens no unsalted template's unit,
so a certified window passed no salt opens nothing. So the circuit a window's units are proved over is
the pinned one, not whatever builds it: a change to a template's Definition or to C-Flock's lowering lands its new digest
in `PouwCheck/Tile.lean` (and `core_check.TILES`), which benchmarks/pouw's staged-unit test enforces at K = 128. A
template has one circuit, so its units proved under an earlier lowering no longer open (the 9 to 10 Oct served windows,
whose `Pc8TileSide` K = 4096 circuit was c7aaeee4…).
`served_commit.check` keeps the draw, the level and the population's domain and calls the opening rule for the rest.

`passEpoch` re-starts the window's pass's epoch from its manifest under the challenger's own coin commitment: a pass
salted by epoch coins carries the challenger's commitment and its salt coin, opened after the weights' root; a pass
recorded before epoch coins (read only in a replay) carries a beacon; either way the manifest's salt is the epoch's, and
the window's context names the pass by its SHA-512. A certified window's verifier holds the network certifier's salt for
the PoUW window its context names (`<prefix>/pouw-window/<p>/<the pass's SHA-512>`), the window's stream and prover, and
the record the certifier made before it drew that salt (`pouwRecord` of the weights' root, the stream and the prover,
`salt.pouw_record`), and passes only a pass whose salt is `pouwSalt` of them (`salt.pouw_salt`) over those weights, a v3
manifest; no other verifier passes a v3 manifest. The difftest's pass cases (below) hold honest certified
windows and a mutation per certified refusal too. `drawCheck` holds the window's drawn units to the epoch's draw coin,
opened after the registration: `Flock.Draw.subset` of the key it derives under the stream, the registration and its
receipt. The coin is a seeded epoch's `draw` coin, or a live epoch's `pouw/draw` coin 0 (`PouwWindow.LIVE_SCHEME`, a
certified window's). A live window's coins are `{key, record}`, the challenger's signed record and the key it names,
and its input carries the challenger key the window's parameters pin (`key`): `drawCheck` refuses coins naming another
key, a live draw with no pin, and seeded coins under a pin (`coin-key`), verifies the record's signature under the
pinned key (`SampledProofs.Challenger.recordReason`, `challenger.record_reason`) and reads the commitment from it,
refuses a bare commitment, and checks the opening against it
(`SampledProofs.Challenger.liveReason`). Both read the current coin version and, only
in a replay of a recorded pass, the e2e-service passes' (`epoch-coins/v2`, whose binding and weights' root keep the
first 32 bytes of `tagged`).

`PouwCheck.Credit` holds a served window's credit to what it fixed before each coin (`credit.py`, the Python reference).
`credited`: each manifest row (the window's calls in order, `index` 0, 1, …) carries its call's verdict as `credit`, a
credited row its z table's root (`tables.z`), and the draw root over the rows (`calls_root`), the last round's receipts
and every round's root and coin must be the one root the challenger's log shows its draw coin followed, read through the
session the auditor's ledger opens for the window; so a bit changed after the draw coin is refused, and the credited calls
are the rows whose bit is true. `drawn`: the units the draw coin draws, never a list handed in: `drawCheck`'s rule under
the law `work:k` (`workUnits`), over the window's layout (`h1window.program`'s arguments, its units the compiled
Program's, call by call, each call's tiles row block major), one stratum per K (`strata`), each unit's work R·C·K, at
K = min(27,713, N) (`law`). `outcome`: the window is accepted only when every unit `drawn` draws has exactly one proof
outcome and it is an accept; a failed or missing one refuses the whole window. `window` is both on one served window,
the stream its root, the registration its draw root and the receipt `{"receipts": …}`. `period`: the windows presented
are exactly those the auditor's ledger lists in the period's close, in its order, and the close is complete, so a period
with no close or with a window missing, out of order or not its own earns nothing; `periodCredit` also requires each of
them to prove the circuit the period's open committed (its descriptor's SHA-512,
`SampledProofs.Challenger.creditedSession`) and credits it (`window`), a refused one refusing the period. `aLink`: each round's root (the one the
challenger received before that round's coin) is `credit.round_root` of the serving process's own round root and that
round's rows of every registered A value (`…/a/<K>`) as aligned subtrees (`credit.cover`), and each A value's registered
root must be the frame-v3 SHA-512 tree over those subtrees and its domain's pads.

The properties are stated in `Security/Properties/Pouw/Check.lean` and proved in
`Security/Properties/Proofs/Pouw/Check.lean`.

`tests/test_difftest.py`, the suite `verity-pouw-check-difftest`, which `check` runs after the Lean builds, holds
`core_check.case` to `pouw-check serve` on the shared harness (`verity/core/difftest/`) as a `ProtocolPort` with the Lean
of record, comparing whole answers, reasons included. Its cases are `tests/pouw_check_cases.py`: honest
windows and one mutation per refusal, the case's own framing among them. A run needs an accept, a reject and an
undecided. The same suite puts `pouw-check-difftest`'s `opening` against `core_check.opening` as a `Port`, comparing
reasons, over `tests/pouw_opening_cases.py`: honest openings and one mutation per refusal (or per reading both sides
share: hex as `bytes.fromhex` reads it, the header's keys sorted), every mutation also replayed over a few seeds, and
every reason reached. Its `pass` and `draw` go against `core_check.pass_epoch` and `core_check.draw` the same way, over
`tests/pouw_pass_cases.py`: honest passes of each coin version and a recorded beacon's, honest draws, and one mutation
per refusal. Its `rounds` goes against `core_check.rounds` over `tests/pouw_rounds_cases.py`: an honest live window's
round roots and coins, read through the challenger's session (its record and close signed under the window's pinned key,
its release log, and the auditor's signed ledger opening and ending it for the window), and one mutation per refusal (an
in-process challenger's coins, an unsigned or tampered log, a ledger signed under another key or cut before the session's
end, a second session for the window, roots the challenger's log doesn't show,
a coin opened twice, one opened before its round's root, a foreign epoch's, a seeded epoch's commitment or another
layout's). `verity/core/protocols/pouw/tests/test_pouw_core_check.py` checks that every mutation reaches the refusal it
aims at. Its `credit`, `drawn`, `outcome`, `window`, `period` and `a-link` go against `credit.credited`,
`credit.drawn`, `credit.outcome`, `credit.window`, `credit.period_credit` and `credit.a_link` over
`tests/pouw_credit_cases.py` (a credit bit flipped after the draw coin, units the prover chose, a drawn unit failed or
missing, a period unclosed or a window of it missing or out of order, an A tree the round roots didn't commit, and one
mutation per other refusal), and `verity/core/protocols/pouw/tests/test_pouw_credit.py`
holds each mutation to its refusal.

~~~text
lake build                                                         the library, pouw-check and pouw-check-difftest
lake exe pouw-check id | check PARAMS COMMITMENT EVIDENCE | serve    the interface's command line
uv run tools/verity/check/suites.py verity-pouw-check-difftest     the difftest
python tools/verity/lean/check.py verity/core/protocols/pouw/lean/check   the Lean check
~~~
