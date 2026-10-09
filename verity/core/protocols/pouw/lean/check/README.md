# `pouw_check`: PoUW's window check as a core protocol

One `Core.Protocol` (`verity/core/protocols/interface`), `PouwCheck.protocol`, id `verity/pouw/v1`: the port of
`verity/core/protocols/pouw/core_check.py`, Lean core only. The Python keeps running until its properties
are proven.

| field | here |
| --- | --- |
| `Params` | the verifier's own: the check item (`entries`, `scheme`, `template`, `tile_work`), `eps` and `delta` as ratio strings, `grid_bits`, the declared `range` of tile counts, the window's `context`, the check query's SHA-512 |
| `Commitment` | the window's registration, one-stage's record (`registration.record`) as canonical JSON, its window `{kind, context, public}` holding `window.REGISTERED` |
| `digest` | the record's own: SHA-512(format ‖ 0 ‖ the canonical record), `registration.digest` |
| `Evidence` | the proof service's answer for a registration, read by `Core.Answer.ofJson` (protocol one-stage's): on acceptance the profile, read by `Core.Integrity.ofJson`, and the public outputs `{"draw": d}`, `d` the verifier's draw as one-stage's `draw.py` writes it (`subset`, `bernoulli`, `stratified` or `work`, with exactly `check_draw`'s keys) |
| `check` | `PouwCheck.check`: PoUW's rule over the registration (`registered`), then the service's answer to it (`outcome`); every refusal is the first rule it breaks, with `core_check.py`'s reason |
| `Public` | `window.PUBLIC`: `verdict`, `work`, `check`, `sampling`, `drawn`, `root_W` |
| `Profile` | the service's `Core.Integrity`, passed through once it is the registered partition's N units drawn subset:K′ at the window's δ, and the draw is subset:K′ over N |

| module | `core_check.py` |
| --- | --- |
| `PouwCheck.Json` | `sorted(dict)`, `_nat`, `_canon` |
| `PouwCheck.Inputs` | `params`, `commitment`, `digest`, `unit_draw`, `evidence`, with their refusals (`answer_of` and `integrity` are the interface's decoders) |
| `PouwCheck.Check` | `expected_items` and `check` |
| `PouwCheck.Protocol` | the `Core.Protocol` value |
| `PouwCheck.Opening` | `opening`, `window_domain`, `verify_path`: one unit's opening beside the check (`served_commit.check`'s rule) |
| `PouwCheck.Pass` | `pass_epoch` and `draw`, with `coins_of`, `opening_of`, `coin_reason` and `weights_root` under `COIN_VERSIONS`: the window's pass and its draw (`served_commit.Challenger`'s rules) |
| `PouwCheck.Difftest` | the difftest's functions behind `pouw-check-difftest` (`opening`, `pass`, `draw`) |

`opening` holds a tile proof's public file to the registered window: the unit drawn (or opened undrawn, the filler's),
the entries root's domain one-stage's served domain of the registered program and partition, the opening's entry under
its salt that root's leaf, the unit one of the entry's tiles, the proof one instance whose first two rows (in its header's
key order, serde's) are an A row and a B row of the fp8-p schemas, and each row's b ‖ c its side tree's leaf at the
unit's row. `served_commit.check` keeps the draw, the level and the population's domain and calls `core_check.opening`
for the rest.

`passEpoch` re-starts the window's pass's epoch from its manifest under the challenger's own coin commitment: a pass
salted by epoch coins carries the challenger's commitment and its salt coin, opened after the weights' root; a pass
recorded before epoch coins (read only in a replay) carries a beacon; either way the manifest's salt is the epoch's, and
the window's context names the pass by its SHA-512. `drawCheck` holds the window's drawn units to the epoch's draw coin,
opened after the registration: `Flock.Draw.subset` of the key it derives under the stream, the registration and its
receipt. Both read the current coin version and, only in a replay of a recorded pass, the e2e-service passes'
(`epoch-coins/v2`, whose binding and weights' root keep the first 32 bytes of `tagged`).

The properties are stated in `Security/Properties/Pouw/Check.lean` and proved in
`Security/Properties/Proofs/Pouw/Check.lean`.

`tests/test_difftest.py`, the suite `verity-pouw-check-difftest`, which `check` runs after the Lean builds, puts
`pouw-check serve` on the shared harness (`verity/core/difftest/`) as a `ProtocolPort` against
`core_check.case`, comparing whole answers, reasons included. Its cases are `tests/pouw_check_cases.py`: honest
windows and one mutation per refusal, the case's own framing among them. A run needs an accept, a reject and an
undecided. The same suite puts `pouw-check-difftest`'s `opening` against `core_check.opening` as a `Port`, comparing
reasons, over `tests/pouw_opening_cases.py`: honest openings and one mutation per refusal (or per reading both sides
share: hex as `bytes.fromhex` reads it, the header's keys sorted), every mutation also replayed over a few seeds, and
every reason reached. Its `pass` and `draw` go against `core_check.pass_epoch` and `core_check.draw` the same way, over
`tests/pouw_pass_cases.py`: honest passes of each coin version and a recorded beacon's, honest draws, and one mutation
per refusal. `verity/core/protocols/pouw/tests/test_pouw_core_check.py` checks that every mutation reaches the refusal it
aims at.

~~~text
lake build                                                         the library, pouw-check and pouw-check-difftest
lake exe pouw-check id | check PARAMS COMMITMENT EVIDENCE | serve    the interface's command line
uv run tools/verity/check/suites.py verity-pouw-check-difftest     the difftest
python tools/verity/lean/check.py verity/core/protocols/pouw/lean/check   the Lean check
~~~
