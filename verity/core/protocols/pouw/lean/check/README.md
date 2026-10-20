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
| `PouwCheck.Rounds` | `rounds`: a live window's round coins, each opened once after its recorded round root (`window.rounds_reason`'s rules, `PouwWindow.roundsReason`, over the proof service's challenger, `SampledProofs.Challenger`) |
| `PouwCheck.Difftest` | the difftest's functions behind `pouw-check-difftest` (`opening`, `pass`, `draw`, `rounds`) |

`opening` holds a tile proof's public file to the registered window: the unit drawn (or opened undrawn, the filler's),
the entries root's domain one-stage's served domain of the registered program and partition, the opening's entry under
its salt that root's leaf, the unit one of the entry's tiles, the proof one instance over the circuit `PouwCheck.tiles`
pins for the check's template (its header's `circuit_sha512`, which `flock-verify verify --zk` binds to the circuit it
verifies against), and each opened row's b ‖ c its side tree's leaf at the unit's row. An unsalted template's proof's
first two rows (in its header's key order, serde's) are an A row and a B row of the fp8-p schemas. A salted template's
(`Salted`: today the H1 tile unit at K = 64, R = 4, C = 8) are its R A rows, its C B rows and its public row `pub`, each
`hm96-sha512/row/v2`, its entry's side trees holding R rows per tile row and C per tile column, and the public row's
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
round roots and coins, and one mutation per refusal (a coin opened twice, one opened before its round's root, a foreign
epoch's, a seeded epoch's commitment or another layout's). `verity/core/protocols/pouw/tests/test_pouw_core_check.py` checks that every mutation reaches the refusal it
aims at.

~~~text
lake build                                                         the library, pouw-check and pouw-check-difftest
lake exe pouw-check id | check PARAMS COMMITMENT EVIDENCE | serve    the interface's command line
uv run tools/verity/check/suites.py verity-pouw-check-difftest     the difftest
python tools/verity/lean/check.py verity/core/protocols/pouw/lean/check   the Lean check
~~~
