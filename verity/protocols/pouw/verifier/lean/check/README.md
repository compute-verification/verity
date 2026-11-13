# `pouw_check`: PoUW's execution phase check as a core protocol

One `Core.Protocol` (`verity/protocols/interface`), `PouwCheck.protocol`, id `verity/pouw/v1`, Lean core only:
PoUW's verifier (`verity/protocols/pouw/verifier/verifier.py`), with `verity/protocols/pouw/verifier/core_check.py` the
Python reference held to it.

| field | here |
| --- | --- |
| `Params` | the verifier's own: the check item (`entries`, `scheme`, `template`, `tile_work`, and optionally `coins`, the execution phase's coin mode, `live` or `seeded`), `eps` and `delta` as ratio strings, `grid_bits`, the declared `range` of tile counts, the execution phase's `context`, the check query's SHA-512; the template one `PouwCheck.tiles` pins, with its tile work and query; and the rows' fixed `length` (`{max_tokens, stop}`, stop `fixed`) |
| `Commitment` | the execution phase's registration, one-stage's record (`registration.record`) as canonical JSON, its execution phase `{kind, context, public}` holding `window.REGISTERED`, whose `length` item must be a fixed-length execution phase's at the verifier's max_tokens (`PouwWindow.lengthProblem`; an execution phase without one, registered before the rule, is refused as `window-form: no-fixed-length`) |
| `digest` | the record's own: SHA-512(format ‖ 0 ‖ the canonical record), `registration.digest` |
| `Evidence` | the proof system's answer for a registration, read by `Core.Answer.ofJson` (protocol one-stage's): on acceptance the profile, read by `Core.Integrity.ofJson`, and the public outputs `{"draw": d}`, `d` the verifier's sample as one-stage's `draw.py` writes it (`subset`, `bernoulli`, `stratified` or `work`, with exactly `check_draw`'s keys) |
| `check` | `PouwCheck.check`: PoUW's rule over the registration (`registered`), then the service's answer to it (`outcome`); every refusal is the first rule it breaks, with `core_check.py`'s reason |
| `Public` | `window.PUBLIC`: `verdict`, `work`, `check`, `sampling`, `length`, `drawn`, `root_W` |
| `Profile` | the service's `Core.Integrity`, passed through once it is the registered partition's N units sampled subset:K′ at the execution phase's δ, and the sample is subset:K′ over N |

A second `Core.Protocol`, `PouwCheck.Period.protocol`, id `verity/pouw/period/v1` (`pouw-period`), is a credit period's credit
and onsite's PoUW subprotocol (`Core.Onsite.components`), with `verity/protocols/pouw/verifier/period_check.py` its Python
(`tests/test_period_difftest.py`). It credits a credit period all or nothing: `Credit.periodCredit` accepts only when every
execution phase of the credit period's close is credited.

| field | here |
| --- | --- |
| `Params` | the verifier's own `{key, main, name}`: the challenger key it pins, its copy of main's first-parent history, the credit period's name |
| `Commitment` | the auditor's ledger, the lines the challenger signed, as canonical JSON: its credit period's open commits the circuit, K, T, the weights and the fixed length before any coins are flipped |
| `digest` | SHA-512(`verity/pouw/period/ledger/v1` ‖ 0 ‖ the ledger's bytes) |
| `Evidence` | `{pins, windows}`: the challenger host's pins log, and each execution phase's evidence in the close's order (`Credit.PERIOD_KEYS`, each sampled proof block's and each round's proof outcome the verifier's verdict) |
| `check` | `Credit.periodCredit`; a refusal is `code: why`, and it never leaves a credit period undecided |
| `Public` | `Credit.PeriodCredit.toJson`: each execution phase's credited calls and credited served work, K, δ and each execution phase's ε |
| `Profile` | `{period, windows}`: the credit period's name and its number of execution phases |

| module | `core_check.py` |
| --- | --- |
| `PouwCheck.Json` | `sorted(dict)`, `_nat`, `_canon` |
| `PouwCheck.Tile` | `TILES`: each tile template the verifier accepts, with its tile work, its check query's SHA-512 and its one circuit's, and a salted template's rows (`Salted`) |
| `PouwCheck.Inputs` | `params`, `commitment`, `digest`, `unit_draw`, `evidence`, with their refusals (`answer_of` and `integrity` are the interface's decoders) |
| `PouwCheck.Check` | `expected_items` and `check` |
| `PouwCheck.Protocol` | the `Core.Protocol` value |
| `PouwCheck.Period` | `period_check.py`: the credit period's `Core.Protocol` value over `Credit.periodCredit`, its decoders and its digest |
| `PouwCheck.PeriodMain` | `pouw-period`'s entry point |
| `PouwCheck.Opening` | `opening`, `window_domain`, `verify_path`, `public_bc`: one unit's opening beside the check (`served_commit.check`'s rule), a certified execution phase's under its certified salt |
| `PouwCheck.Pass` | `pass_epoch` and `draw`, with `coins_of`, `opening_of`, `coin_reason` and `weights_root` under `COIN_VERSIONS`: the execution phase's pass and its sample (`served_commit.Challenger`'s rules); `live_salt`, a live Pearl-C pass's salt from the salt-only session the auditor's ledger reserved, opened on its salt execution phase's root (`epochOpenRoot` of its weights' root and its `pass_index`) and ended complete |
| `PouwCheck.Rounds` | `salt`, `rounds`: a live execution phase's salt coin, opened once after its open root (`salt.window_open_root` of its record and the session the auditor's ledger reserved), and its round coins, each opened once after its recorded round root (`window.rounds_reason`'s rules, `PouwWindow.roundsReason`), both read through the challenger's session the ledger opens for the execution phase (`Spot.Challenger.auditedSession`, `sessionNamed`) |
| `PouwCheck.Difftest` | the difftest's functions behind `pouw-check-difftest` (`opening`, `registered`, `pass`, `draw`, `salt`, `live-salt`, `rounds`, `chunks`) |
| `PouwCheck.DifftestMain` | `pouw-check-difftest`'s server, and its `coins n`: `n` bytes of `IO.getRandomBytes` (A3), the challenger's entropy (`verifier.coins`; `tests/test_coins.py`) |
| `PouwCheck.Credit` | `credit.py`'s `_window_outcomes` and `_period_outcomes` (its entries `window` and `period_credit` on each sampled proof block's proof outcome in place of the proof files they verify themselves, until this runs the verifier in-process), and the steps they compose (`Credit.Internal`'s `credited`, `outcome`, `aLink`, `registration` and `weights`; `drawn`, `layoutOf`, `period`, `committedWeights`): a served execution phase's credit bits fixed before its sampling coin, its round log the challenger's, its A trees tied to its round roots, its sampled proof blocks the sampling coin's over its committed circuit's layout, no partial credit over them, its entries and its sampled proof blocks' proofs its own registration's, its B rows the ones its credit period's open signed the weights' root of, its served rows the fixed length its credit period's open committed (`Internal.length`, `committedLength`), and a credit period counted only closed and complete |
| `PouwCheck.CreditDifftest` | the difftest's `credit`, `drawn`, `outcome`, `window`, `period` and `a-link` behind `pouw-check-difftest` (`CreditDifftest.all`, the functions `DifftestMain` serves) |
| `PouwCheck.LeafOwn` | not `credit.py`'s: an execution phase unit's leaf statements' tile and the placement of the values they read in the execution phase's tables (`placeOf`: its round's A table of K, the weights' B, its call's z and receipt tables, each from the unit's base row), from the execution phase's committed descriptor and round log alone, and `pouw-leaf-own-window`, `PouwWindow.Lanes.leafOwn?` at that placement (`tests/test_leaf_own.py`) |
| `PouwCheck.LeafOwnMain` | `pouw-leaf-own-window`'s entry point |
| `PouwCheck.Lanes` | `credit.py`'s lanes form: a sampled leaf at a K the circuit files commit as `lane<K>` and `combine<K>` (`lanesFiles`, `formOf`) verified as its lanes and combine statements (`leafAsks`), each against its own leaf Program (`leafOf`, a population of one) and the `--registered` file credit builds from the descriptor (`leafRegisteredFile`: the execution phase's tables at the leaf's place, its partials at the produced record's root under the execution phase's `draw_root`, its known rows the verifier's own), then the produced record's opened roots and shared openings (`outcomes`); a leaf outside `LeafLanesSound`'s reach is refused (`lanesShape`, `lanes-shape`) |
| `PouwCheck.Verified` | `credit.py`'s `_asks` and `_verified`: a served execution phase's statements verified, the verdict `credit.window` and `credit.period_credit` take each sampled proof block's outcome from (`credit._lean_verified`). The verifier runs outside, so `verified` is asked twice: it names each run (the statement, its K's committed circuit file, the call's `--registered` file and the partition object, up to the first statement a check before the verifier refuses), then judges the verifier's answers (a run that fails or refuses, a rejected session, a session that opened its call's public table or the ok rows elsewhere than the `--registered` file's value and root, or a unit of another call, refuses the execution phase); an execution phase's lanes evidence adds each sampled lanes-form leaf's runs after the call statements' (`PouwCheck.Lanes`) |
| `PouwCheck.LiveVerified` | `credit.py`'s `_live_asks` and `_live_verified`: a served execution phase's live@<s> statements verified, the verdict `credit.window` and `credit.period_credit` take each round's live outcome from (`credit._lean_live_verified`), asked twice as `verified` is: it names each run (the statement, the committed live circuit file, round s's `--registered` file, `live_reads.reads_file`'s, and the partition object, up to the first statement a check before the verifier refuses), then judges the verifier's answers (a run that fails or refuses, a rejected session, or a session that proved another unit than its round's, refuses the execution phase) |
| `PouwCheck.CommittedFiles` | `credit.py`'s `committed_files`: the circuit files a credit period's open committed (`circuit_files`, each K's template's and the live Program's, or a lanes-form K's `lane<K>` and `combine<K>` with `leaf_programs`), which `credit.period_credit` reads from here (`credit._lean_committed_files`) |
| `PouwCheck.VerifiedDifftest` | the difftest's `credit-verified`, `credit-live-verified` and `credit-committed-files` behind `pouw-check-difftest` (`VerifiedDifftest.all`, `CreditDifftest.all`'s then it, the functions `DifftestMain` serves) |

`opening` holds a tile proof's public file to the registered execution phase: the unit sampled (or opened unsampled, the filler's),
the entries root's domain one-stage's served domain of the registered program and partition, the opening's entry under
its salt that root's leaf, the unit one of the entry's tiles, the proof one instance over the circuit `PouwCheck.tiles`
pins for the check's template (its header's `circuit_sha512`, which `flock-verify verify --zk` binds to the circuit it
verifies against), and each opened row's b ‖ c its side tree's leaf at the unit's row. An unsalted template's proof's
first two ports, `a0` and `b0` (`PC8_PORTS`), are an A row and a B row of the fp8-p schemas. A salted template's
(`Salted`: today the H1 tile unit at K = 64, R = 4, C = 8) are its R A rows, its C B rows and its public row `pub`, each
`hm96-sha512/row/v2`, its entry's side trees holding R rows per tile row and C per tile column. Each schema is read by
port name, and Flock reads port k's b ‖ c at the body's k-th place, in the circuit's port order, so a header whose
keys, in key order (serde's), don't list the opened side rows first and in port order (`portsOf`) is refused, never
read in its own order (every pinned template's side ports sort in port order). The public row's
b ‖ c must be `publicBc` of the execution phase's context, the certified salt the verifier passes (`opening`'s `salt`,
`core_check.certified_salt`, the salt `passEpoch` checked), the unit's position and the entry's side roots. Only a
certified execution phase opens a salted template's units, and only under a 64-byte salt; it opens no unsalted template's unit,
so a certified execution phase passed no salt opens nothing. So the circuit an execution phase's units are proved over is
the pinned one, not whatever builds it: a change to a template's Definition or to Flock's lowering lands its new digest
in `PouwCheck/Tile.lean` (and `core_check.TILES`), which benchmarks/pouw's staged-unit test enforces at K = 128. A
template has one circuit, so its units proved under an earlier lowering no longer open.
`served_commit.check` keeps the sample, the level and the population's domain and calls the opening rule for the rest.

`passEpoch` re-starts the execution phase's pass's execution phase from its manifest under the challenger's own coin commitment: a pass
salted by execution-phase coins carries the challenger's commitment and its salt coin, opened after the weights' root; a live
Pearl-C pass, whose manifest's coins are a live session's, is read only through its verifier's `live` inputs (`{key,
ledger, main, pins, session}`): the salt-only session the auditor's ledger reserved, opened on the pass's salt execution phase's
root (`epochOpenRoot` of its weights' root and its `pass_index`, computed by the check) and ended complete, the session and its commitment taken from the ledger and the session's files, never the manifest
(`livePassSalt`, `liveSaltCheck`, LIVESALT2); a pass recorded before the execution phase's coins (read only in a replay) carries a
beacon; either way the manifest's salt is the execution phase's, and
the execution phase's context names the pass by its SHA-512. A certified execution phase's verifier holds the network certifier's salt for
the PoUW execution phase its context names (`<prefix>/pouw-window/<p>/<the pass's SHA-512>`), the execution phase's stream and prover, and
the record the certifier made before it sampled that salt (`pouwRecord` of the weights' root, the stream and the prover,
`salt.pouw_record`), and passes only a pass whose salt is `pouwSalt` of them (`salt.pouw_salt`) over those weights, a v3
manifest; no other verifier passes a v3 manifest. The difftest's pass cases (below) hold honest certified
execution phases and a mutation per certified refusal too. `drawCheck` holds the execution phase's sampled proof blocks to the execution phase's sampling coin,
opened after the registration: `Flock.Draw.subset` of the key it derives under the stream, the registration and its
receipt. The coin is a seeded execution phase's `draw` coin, or a live execution phase's `pouw/draw` coin 0 (`PouwWindow.LIVE_SCHEME`, a
certified execution phase's). A live execution phase's coins are `{key, record}`, the challenger's signed record and the key it names,
and its input carries the challenger key the execution phase's parameters pin (`key`): `drawCheck` refuses coins naming another
key, a live sample with no pin, and seeded coins under a pin (`coin-key`), verifies the record's signature under the
pinned key (`Spot.Challenger.recordReason`, `challenger.record_reason`) and reads the commitment from it,
refuses a bare commitment, and checks the opening against it
(`Spot.Challenger.liveReason`). Both read the current coin version and, only
in a replay of a recorded pass, the e2e-service passes' (`epoch-coins/v2`, whose binding and weights' root keep the
first 32 bytes of `tagged`).

`PouwCheck.Credit` holds a served execution phase's credit to what it fixed before each coin (`credit.py`, the Python reference).
Its entries are `window` and `periodCredit`; the steps they compose are internal (`Credit.Internal`, which only the
difftest reaches), so no caller credits an execution phase with a step skipped.
`credited`: each manifest row (the execution phase's calls in order, `index` 0, 1, …) carries its call's verdict as `credit`, a
credited row its z table's root (`tables.z`), and the sample root over the rows (`calls_root`), the last round's receipts
and every round's root and coin must be the one root the challenger's log shows its sampling coin followed, read through the
session the auditor's ledger opens for the execution phase; so a bit changed after the sampling coin is refused, and the credited
calls are the rows whose bit is true. `drawn`: the units the sampling coin selects, never a list handed in: `drawCheck`'s rule
under the sampling design `work:k` (`workUnits`), over the execution phase's layout (`h1window.program`'s arguments, its units the compiled
Program's, call by call, each call's tiles row block major), one stratum per K (`strata`), each unit's work R·C·K, at
the K its credit period's open committed (`law`; `committedK` reads `draw_k`, and an execution phase with fewer units than K is refused).
`servedRows`: each manifest row states its call's served rows m_s (`served`, under `calls_root`, so fixed before the
sampling coin), a natural at most the T table rows its credit period's open committed (`committedRows` reads `rows`, and an
execution phase whose A slot M is not `slotOf` T, the least power of two at or above T and at least R, is refused), one m_s for
every call of a step (call c at step `c / (C/S)`); else the
execution phase is refused (`manifest`). A credited call with m_s = 0 is no credit. `established`: the ε that sample established at
δ = 2⁻⁴⁰ over the execution phase's credited served work, `PouwWindow.epsAt` (`k_prime` inverted) as `epsAt K δ C W 1`, C the
credited calls' served work (`servedWork`, m_s·n·K a call, never its M slot rows') and W the execution phase's units'
(`callWork`, the sample's population, padded slots included), which `periodCredit` reports with C as the execution phase's
`work`, null when it establishes no ε ≤ 1; PoUW has no ε target. That m_s counts the step's input-ids table's served
slots is not yet proved: it is the serving process's statement, fixed before the sampling coin. `outcome`: the execution phase is accepted only when every unit `drawn` samples has exactly one
proof outcome and it is an accept; a failed or missing one refuses the whole execution phase. `layoutOf`: the layout is the
committed circuit's, never the execution phase's evidence: read from the descriptor whose SHA-512 the credit period's open committed
(its program function's record, its templates' A and B rows for R and C), whose root's nodes other than inputs must be
each call's (M/R)·(n/C) instances of K's template in order. `window` runs every step on one served execution phase and its one
round log, crediting only when all accept: `credited`, `roundsCheck` (the log's roots are the ones the challenger
received, `core_check.rounds`), `aLink` over the execution phase's registered entries, `layoutOf`, `servedRows`, `length`, then `outcome`'s rule with the
stream its root, the registration its sample root and the receipt `{"receipts": …}`, then `registration`. `period`: the execution phases
presented are exactly those the auditor's ledger lists in the credit period's close, in its order, and the close is complete,
so a credit period with no close or with an execution phase missing, out of order or not its own earns nothing; `periodCredit` also
requires each of them to have run from a commit the host's pins log records on the auditor's first-parent list of main
and to prove the circuit the credit period's open committed (its descriptor's SHA-512,
`Spot.Challenger.creditedSession` over `pinnedSession`, the same bytes `layoutOf` reads) and credits it
(`window`) sampled at the credit period's K, a refused one refusing
the credit period. Every step that reads the challenger's session (`credited`, `roundsCheck`, `saltCheck`) reads it through
`pinnedSession`, the pins log (`pins`) and the first-parent list (`main`) given beside the ledger. `aLink`: each round's root (the one the challenger received before that round's coin) is `credit.round_root`
of the serving process's own round root and that round's rows of every registered A value as aligned subtrees
(`credit.cover`): round s's share of an execution phase's `…/a/<K>`, or the whole of round s's A table of K `…/a/<K>@<s>` (the M rows
of each of round s's calls of K, call after call, its calls the C/S calls from s·C/S) in round s; and each A value's registered root must be the frame-v3 SHA-512 tree over those
subtrees and its domain's pads. `registration`: each sampled proof block's proof outcome is the verifier's verdict for its session
(`PROOF_KEYS`: the SHA-512 of the descriptor it verified the session against and the session's registered finding,
the Lean verifier's `registered`, each port's value and the root its reads were opened against); each A value's domain, and
each other value's a proof opened, is the one one-stage's registration gives it under the committed descriptor's
SHA-512 and, for every value but the weights' `…/b/<K>`, the execution phase's open root (`Spot.Registered.domainAt`); and each proof was verified against that descriptor, opened only values the execution phase registers once, each at
its registered root, and opened its unit's K's A value, the execution phase's `…/a/<K>` or its own call's round's `…/a/<K>@<s>`. So entries of another execution phase or circuit, a proof verified against another registration, and an entry whose root
is not the one its sessions opened are refused. `length`: the credit period's open commits each request's public max_tokens
with the public-cap done source (`committedLength`, `{max_tokens, stop}`, stop `fixed`), and the
served rows m_s each step's manifest rows fixed before the sampling coin (`servedRows`) are that fixed length's schedule,
every request admitted in round 0: round 0 serves at least one row a request, round s ≥ 1 one decode row for each
request whose max_tokens is past s. A credit period whose open commits no length is
not of record (`no-fixed-length`), an EOS-keyed stop mode is refused (`eos-keyed-row`), and a round serving more or fewer
rows ran a request past its public max_tokens or stopped one before it (`length-mismatch`).

The properties are stated in `security/Properties/Pouw/Check.lean` and proved in
`security/Proofs/Properties/Pouw/Check.lean`.

`tests/test_difftest.py`, the suite `verity-pouw-check-difftest`, which `check` runs after the Lean builds, holds
`core_check.case` to `pouw-check serve` on the shared harness (`tools/difftest/`) as a `ProtocolPort` with the Lean
of record, comparing whole answers, reasons included. Its cases are `tests/pouw_check_cases.py`: honest
execution phases and one mutation per refusal, the case's own framing among them. A run needs an accept, a reject and an
undecided. The same suite puts `pouw-check-difftest`'s `opening` against `core_check.opening` as a `Port`, comparing
reasons, over `tests/pouw_opening_cases.py`: honest openings and one mutation per refusal (or per reading both sides
share: hex as `bytes.fromhex` reads it, the header's keys sorted), every mutation also replayed over a few seeds, and
every reason reached. Its `pass` and `draw` go against `core_check.pass_epoch` and `core_check.draw` the same way, over
`tests/pouw_pass_cases.py`: honest passes of each coin version and a recorded beacon's, honest samples, and one mutation
per refusal. Its `rounds` goes against `core_check.rounds` over `tests/pouw_rounds_cases.py`: an honest live execution phase's
round roots and coins, read through the challenger's session (its record and close signed under the execution phase's pinned key,
its release log, and the auditor's signed ledger opening and ending it for the execution phase), and one mutation per refusal (an
in-process challenger's coins, an unsigned or tampered log, a ledger signed under another key or cut before the session's
end, a second session for the execution phase, roots the challenger's log doesn't show,
a coin opened twice, one opened before its round's root, a foreign execution phase's, a seeded execution phase's commitment or another
layout's). `verity/protocols/pouw/verifier/tests/test_pouw_core_check.py` checks that every mutation reaches the refusal it
aims at. Its `window` and `period` go against `credit._window_outcomes` and `credit._period_outcomes` (`credit.window`
and `credit.period_credit` on proof outcomes, which they derive from an execution phase's proof files by running the
verifier; `test_pouw_credit_statements.py` holds that step), and its `credit`, `drawn`,
`outcome` and `a-link` against the steps they run, `credit._credited`, `credit.drawn`, `credit._outcome` and
`credit._a_link`, over
`tests/pouw_credit_cases.py` (a credit bit flipped after the sampling coin, units the prover chose, a sampled proof block failed or
missing, a credit period unclosed or an execution phase of it missing or out of order, an A tree the round roots didn't commit, a
descriptor whose record names another layout than its nodes, a round log the challenger never received, entries or a
sampled proof block's proof of another registration than the execution phase's, an entry whose root is not its sessions', an execution phase of a
credit period whose B rows, one row's s_w or every code, are not the ones the credit period's open signed, a credit period whose open signs
no weights' root, an execution phase whose request runs past its public max_tokens or stops before it, a credit period whose open
commits no fixed length or an EOS-keyed one, and one mutation per other refusal), and `verity/protocols/pouw/verifier/tests/test_pouw_credit.py`
holds each mutation to its refusal.

`tests/test_verified_difftest.py` puts `credit-verified` against `credit._asks` and `credit._verified` with the Lean
of record, over `tests/pouw_verified_cases.py` (an execution phase's statements, null or answered as the served path answers,
and one mutation per refusal before the verifier runs and of its answers), and holds the served
`credit._lean_verified` to `credit._verified`'s verdict and verifier runs. `tests/test_live_verified_difftest.py` does
the same for `credit-live-verified` against `credit._live_asks` and `credit._live_verified`, over
`tests/pouw_live_verified_cases.py` (an execution phase's live statements, its live tables' entries and each step's m_s), and
holds the served `credit._lean_live_verified` to `credit._live_verified`; and `credit-committed-files` against
`credit.committed_files`, on the ledgers of `pouw_credit_cases.period_instance` whose open commits circuit files of each
form.

~~~text
lake build                                                         the library, pouw-check and pouw-check-difftest
lake exe pouw-check id | check PARAMS COMMITMENT EVIDENCE | serve    the interface's command line
uv run tools/verity/check/suites.py verity-pouw-check-difftest     the difftest
python tools/verity/lean/check.py verity/protocols/pouw/verifier/lean/check   the Lean check
~~~
