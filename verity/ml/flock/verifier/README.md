# Flock verifier: runners and cross-checks

Flock's verifier is the Lean package in `verity/core/service/flock/` (`flock_verifier`, Lean 4, no
dependencies, toolchain pinned in its `lean-toolchain`). Its definitions are the specification, and its docstrings say
what each check is. This directory holds what runs it and what holds it to other implementations: the agreement job
against upstream's Rust verifier, the agreement scripts for single parts, the mutant and fuzz generators, and the vectors.
Upstream's Rust verifier is a cross-check, never a verifier.

The verifier replays a recorded session offline, on the CPU, in interactive mode only: every coin comes from the
verifier's live session record, and the transcript never hashes to derive a challenge. Fiat–Shamir, the union prover,
forks and grinding are refused where they could appear.

## Where each part is in the Lean

The protocol is the Flock paper (Bünz, Rothblum, Wang, ePrint 2026/1329: §4 and Appendices A–C). Upstream's code
(succinctlabs/flock `b684b12`) was read only for byte formats, constants and the order of operations; the places where
the verifier follows upstream because the paper is silent are listed below.

| part | Lean (`verity/core/service/flock/`) |
|---|---|
| fields: GF(2^128) in GHASH form, the AES field and its embedding φ8, GF(2^256) | `Flock.Field` |
| SHA-512, the hm96-sha512 hiding leaves | `Flock.Hash`, `Flock.Hm96` |
| the pipeline `verify : Config × Record × Proofs × PublicInput → Verdict` | `Flock.Verify` |
| the pinned configuration: profile, reps, flavour, link shape, coin scheme, statement versions | `Flock.Tags` |
| the session record (`flock-live-session/v1`), its session checks and its one canonical text | `Flock.Record`, `Flock.RecordText` |
| the replay transcript (transcript-v2 framing; coins read from the record, never derived) | `Flock.Transcript` |
| the unit sample (subset, bernoulli, stratified, work; derived by the verifier, never taken as stated) | `Flock.Draw` (`HmRow.stratifiedLaw`, `HmRow.workLaw`; commitment readers sampled in full: `Registered.checkFull`) |
| a `--zk` session's keyed coin tree | `Flock.CoinTree` (`checkFresh`) |
| the proof container (bincode, refused unless canonical, every nonce 0) | `Flock.Proof` (`decodeProof_canonical`), `Flock.ZkProof` |
| statement binding, zerocheck, lincheck, the claims handed to the opening | `Flock.Piop` |
| claim batching and ring switching | `Flock.Opening` |
| Ligerito | `Flock.Ligerito`, `Flock.ZkLigerito` |
| Merkle caps and openings | `Flock.Merkle` |
| the `fast100` schedule, `m = 22 … 35` | `Flock.Schedule` |
| statements: block matrices, regions, Σ, the statement digest, `Hello` | `Flock.Statement`, `Flock.Net` |
| the circuit file, lookup slots, row leaves, the proved range | `Flock.Circuit`, `Flock.Lookup`, `Flock.RowLeaf`, `Flock.Comp` (`checkInRange`) |
| frame-v3 roots of the public file | `Flock.HmRow` (`checkPublic`) |
| the partition invariant and the width rule, on the verifier's own program | `Flock.Partition` (`checkCut`), `Flock.Qword`, `Flock.Qcall`, `Flock.Program`, `Flock.Extract`, `Flock.Sigma` |
| hm96-sha512 rows, their nets, template queries, registered reads | `Flock.HmRow`, `Flock.HmNets`, `Flock.TemplateQuery`, `Flock.Registered`, `Flock.KnownRead` |
| hidden outputs and links | `HmOut`, `HmLinks` (in `Flock/HmRow.lean`) |
| the recursion's units and their composition | `Flock.RecOpen`, `Flock.RecResiduals`, `Flock.RecOpening`, `Flock.RecVStar`, `Flock.RecSetup`, `Flock.RecCompose`, and `VStar` |
| the typed statement | `Flock.Typed`, `Flock.CircuitType`, `Flock.Layout`, `Flock.DeriveAll` (`deriveChecked`) |
| the templated statement | `Flock.Tmpl` and its `Flock.Tmpl*` modules |
| sessions of several tables | `Stmt.setupTables`, `Draw.tablesWhy` |
| `--zk` sessions: the mask slot, and Δ and regions kept off it | `Flock.Zk` (`shapeOk`, `maskFree`, `SLOT_LOG_MIN`) |
| plain leaves, a recursive session's inner statement | `Tags.withPlainLeaves`, `Registered.checkDescription` |
| fail-closed: the forms the soundness proof covers | `Flock.ProvedScope` (`check`, `checkP`, `scopeOkI`, `drawLaw`, `programHeld`) |
| the ZK server's contract, frames, coins, `Hello` and session loop | `Flock.Firewall` and `Flock/Firewall/` |

`verify` fails closed: after every check passes, `Flock.ProvedScope` refuses each statement form, sampling design or setting
the soundness theorem doesn't cover, with `refused: <why>: a statement form the soundness proof does not cover`, so a
session that fails a check keeps that check's reason. Under `--rec-inner` that includes a V* session over an inner session
whose record holds no draw (`Flock.RecSetup.innerDrawn`), one at a holo or sparse inner shape (`Flock.RecSetup.denseOnly`:
`RecursiveKS` covers the dense form only, and the compiled model refuses the same sessions, `RecDeploy.acc_dense`), and a
recursive session without `--zk` (`Flock.RecSetup.zkOnly`).

`verify` accepts a session only at `verity/flock-circuit` or `verity/flock-circuit+plain-leaves` (`ProvedScope.served`),
on live coins (`--coins os`, or under `--zk` the keyed coin tree). Every other version it sets up and checks, so the
agreement corpus and the recorded fixtures replay, and then refuses the session (`ProvedScope.servedForm`, last, once
every check has passed): a `+seed-injection` one as a selftest build's statement, the rest as superseded; a plain session
on seed coins (`--coins seed`, the default without `--zk`) is refused the same way. A typed version, or one before hm96
rows, is refused earlier, for its form (`ProvedScope.check`).

## Guarantees and premises

- The one soundness theorem over everything `verify` accepts: `FailClosed.flock_verify_sound`
  (`Security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean`); per table at `fast100`, 2^-205 without `--zk`:
  `flock_verify_sound_205` (`FailClosed/Numeric.lean`); the public ties: `flock_verify_public_ties`
  (`FailClosed/PublicTies.lean`).
- The guarantee statements: `Security/Definitions/Flock/Guarantees/` (`ClaimsStated`, `SamplingClaimsStated`,
  `RecursionClaimsStated`, and the record sizing in `Audit.lean`); the proved ones are the `Flock.SecurityProofs.*`
  entries of `Security/lean-audit.json`, among them `EndToEnd`, `EndToEndHidden`, `ZeroKnowledgeHidden`, `MaskedZK`,
  `RecursiveKS`, `RecursiveZK`, `PublicAudit` and `IntegrityHorizon`, each stated in a `Statement.lean` (or
  `KSStatement.lean`, `PublicAudit.lean`) of its area under `Security/Proofs/Flock/`.
- The named premises: `Security/Proofs/Flock/Soundness/Assumptions.lean` (`UniformRandomBytes`, `HmRowComputes`),
  `Assumptions/Custody.lean` (`RecordCustodyExec`, `RecordCustodyExecH`), and the rest of the lock's `assumptions` list
  (`Model.Statement.LinkLayout`, `SHA512CRStrict`, the `Zk.RecordCustodyZK*` family); what each means:
  `Security/Proofs/Flock/Soundness/ASSUMPTIONS.md`.
- The trusted code: the running binary also trusts the toolchain's compiler and runtime and the C compiler and library
  under them, and no declaration swaps a definition for code no proof reads (`verity/core/service/flock/README.md`,
  enforced by `verity/ml/flock/tests/test_lean_verifier.py`). The agreement corpus's coverage rule is
  `verity/ml/flock/tests/test_vectors_coverage.py`.
- The lemmas the verifier's parts rest on: `merkle_binding` (`Security/Proofs/Flock/Verifier/Merkle.lean`),
  `squeeze_answers` and `squeeze_retained` (`Verifier/Transcript.lean`), `G_irreducible` and `Q_irreducible`
  (`Security/Proofs/Flock/Level3/`). The model they're about: `Security/Definitions/Flock/`.

## Running it

~~~text
cd verity/core/service/flock && lake build    # .lake/build/bin/flock-verify and the other executables
flock-verify verify [--statement ID] [--archive DIR] --circuit F --public pub.bin [--tables DIR] [--coins seed|os] [--zk] \
    [--program F] [--partition F] [--registered F] [--session-tables J] [--plain-tables] [--upstream-plain] --session DIR...
flock-verify statement [--statement ID] --circuit F --public pub.bin [--tables DIR]   # digest, Σ, Hello
flock-verify rec-compose --rec-inner F --rec-rows F --rec-run F [--archive DIR] LOG... # a recursive statement's verdict
~~~

- `--archive DIR` reads every object (circuit, public file, partition, program, tables) by its SHA-512 from the
  verifier's own store, never from the prover's session directory; without it the flags are paths, for the development
  and agreement tools.
- `--program`, `--partition` and `--registered` are the verifier's own copies of the program, the partition object and
  the registered reads. Without `--program` every session is refused: a verifier that can't read the Program can't know
  what it verifies.
- `--tables DIR` holds the pinned MUFU tables (`<name>.u32`), for circuits with lookup slots.
- `--coins` binds the coin scheme a plain session's `Hello` names: `seed`, the default, which `verify` refuses once every
  check has passed, or `os` (per-round OS coins, `FC_COINS=os`). A `--zk` session's coins are its coin tree.
- A plain proof file is taken in one form: by default the deployed prover's, with the `zk_level0 = None` byte after the
  pair; `--upstream-plain` takes upstream's builds' form.

Each session `DIR` holds `session.json` (the `flock-live-session/v1` record) and each table's `<table>.rep0.bin` and
`<table>.rep1.bin`. `verify` prints one `VERDICT {"accepted", "session", "why", …}` line per session and exits 0 iff all
are accepted.

A recursive statement is one `verify --zk` per V* part, and no part sees another. Each part's `VERDICT` names its part
(`rec_part`), the SHA-512 of the shared rows file it checked (`rec_rows_sha512`), of the inner shape file it read
(`rec_inner_sha512`), of the inner session's record and its transcript (`rec_inner_session_sha512`,
`rec_inner_transcript_sha512`) and of every input V*'s parts share (`rec_shared`: the inner shape, the inner statement's
files and how it is set up, the inner session's record, the coin server's transcript, the registration), whether it ran
`--zk` and its session's coin key. `rec-compose` (`Flock.RecCompose.check`) gives the statement's verdict: it accepts only
when there is one log per V* statement with one `VERDICT`, every part is accepted under `--zk` without an identifier at
the one rows file the prover staged (`acc0`, the tops, the `rec-c<k>` rows) and the composition's own inner shape, every
part read one set of the other shared inputs, in the compiled verifier's form, and no two sessions share a coin key
(`RecCompose.Accepts`, `check_accepts`; `Proofs.Flock.Claims.Recursion.ComposeParts.composed_parts` reads it at the
compiled verifier's parts).

It also holds the parts to one run, C10 `OneRun`, through `--rec-run`: the challenger's interactive-run record
(`verity/rec-run/v1`, written by `sampled_proofs.challenger.RunRecord`). The record stages the rows file and `rec_shared`
before any session, logs every session the challenger opened for the run's parts (a retry's too) and is closed after the
last; it holds no coin, since each part's coins are drawn fresh at its own Hello and never committed before the run. No
live challenger writes it yet: C-Flock's `serve` logs no Hello to a run record, so a recursive run composes only where
its record is given. `rec-compose` refuses a part
with two sessions or none, a session opened and not composed, a composed session the record didn't open, a staging after
the first Hello, a second staging or one of other inputs, and a record that isn't closed (`RecCompose.checkRun`;
`check_oneRun` proves acceptance gives `OneRun`, and `ComposeParts.composed_run` reads it at the compiled verifier's
parts). The record is the auditor's: it is read from the challenger's host (`trusted-host`), never from the prover's
bundle. `verity_flock.rec_compose` is its Python mirror (`tests/test_lean_rec_compose.py`). The pod's Lean step
(`85-rec-reprice.sh`, `lean_verdicts.py --compose`) gives every part those files and composes V*'s parts over the run's
record: on loopback the pod is the challenger's host, and under `MODES=firewall` (one session per staged statement) it
lists every session its coin servers served (`opens.jsonl`) and assembles the record from them after the sessions
(`lean_verdicts.loopback_record`), the staged set computed from the files every part was given
(`rec_compose.shared_of_args`), not read from any part's `VERDICT`. Under `MODES=dev` or `host`, with WARM + RUNS sessions
a part, it doesn't compose the statement.

`ID` names a statement version (`Flock.Tags`): `verity/flock-circuit`, the default, whose rows are `hm96-sha512/row/v1`
leaves and whose outputs are all hidden rows; `verity/flock-circuit/types`, the typed statement;
`verity/flock-circuit/tmpl`, the templated statement, parsed and refused; `verity/flock-circuit+plain-leaves`, a recursive
session's inner statement, checked without `--zk`; `+seed-injection` on any of them, as the prover's selftest build pins
it, which `verify` checks and then refuses. The names pinned at older commits (`verity/flock-circuit@<commit>`, back to `verity/flock-netlist/v1`) are still
parsed, so recorded sessions keep their names, and each one before hm96-sha512 rows is refused for its form. Those with
keyed-BLAKE3 row leaves and SHA-256 trees (`@19c7269a`, `@fd02e847`, `@631567f7` and `verity/flock-netlist/v1`) need
`--comp`, the BLAKE3 compression circuit, and the verifier and `hash-test` carry BLAKE3 for them alone. The pure-block and
IR-frame statements are not in the Lean verifier.

The other `flock-verify` subcommands run one part on its own: `field-test`, `hash-test`, `merkle-test`, `hm96-test` (on
`verity/ml/commitments/hm96/vectors.json`) and `comp-check` (`Flock.Comp`'s checks); `draw` and `draw-test` (the unit
sample); `unit-cut`, `cut-check`, `qword-program`, `qcall-program`, `input-ids`, `sigma-check` and `template-query` (the
partition and program checks); `net-check`, `region-words`, `typed-expand`, `typed-rows`, `circuit-type` and `layout`
(statement setup); `archive-put` (the object store); `evaluate`; and `rec-inner`, `rec-statements`, `rec-transcript` and
`rec-record` (the recursion). The other executables are `flock-firewall` (`contract`, `frame`, `tap`, `coins`, `hello`,
`session`, `finish`: the ZK server), `flock-rows`, `flock-witness`, `flock-rec-opening` and
`flock-rec-sparse`.

## Agreement with upstream

`ci.py` is the agreement job (`check`'s `lean-agreement` step). For every replayable set of `vectors.json`
(`flock-verifier-vectors/v1`) that has an upstream binary (`upstream.json`, built by `upstream-build.sh` from
`ci-bundle.sh`'s bundle, run by `ci-pod.sh`), it fetches the set's inputs from the store, regenerates its proof mutants
(`redigest.py`), retained-bytes sessions and seeded fuzz cases (`fuzz.py`), adds its forgeries, and runs `agree.py`, which
compares the Lean verifier's verdict with upstream's (`Server::from_record`, then `finish`) on every session:

~~~text
research run --tool flock-agreement --campaign flock-verifier -- python ci.py --lean flock-verify \
    --upstream fd02e847=flock-circuit --upstream 9294e161=flock-netlist --work DIR --seed S --fuzz N
python agree.py --lean flock-verify --upstream flock-circuit [--statement ID] --circuit F --public pub.bin \
    [--expect mutants.json]... [--jobs J] --out DIR --sessions DIR...
~~~

`agree.py` writes `agreement.tsv` and `result.json`, and exits 1 if the verifiers disagree, or miss an expected verdict,
on any session. `--jobs J` deals each public file's sessions into chunks run by `J` process pairs, with the same
output. A disagreement is resolved against the paper and the Lean, never by copying upstream. Each version's
vectors patch (`netlist-vectors.patch`, `circuit-vectors*.patch`) applies to the prover commit it names and adds
`vectors`, `export-comp` and `replay --sessions DIR...` to upstream's binary; build inside upstream `b684b12` with the
patches applied (`cargo build --release -p flock-live --bin <binary>`). The statements no pinned upstream build has
(hidden outputs, `--zk`) agree with this repository's Rust prover instead: `ci.py --live` holds Lean's verdicts on
`vectors.json`'s `live.sets` to the live server's, and `agree.py --zk` holds them to the live build's `replay --zk`.

The Lean verifier is deliberately stricter than upstream in these places, each refused by Lean and accepted by upstream.
A set marked `lean_refuses`, `lean_refuses_drawn` or `lean_refuses_drawless` compares a session refused for its form,
sampling design or missing sample alone as accepted.

| id | Lean refuses |
|---|---|
| D1 | `root_F ≠ Σ` |
| D2 | a stream other than the configured rep streams |
| D3 | `link.sigma ≠ Σ`; upstream refuses it too from `b3baabd8` |
| D4 | retained round bytes that don't hash to the round's digest; upstream refuses them too from `631567f7` |
| D5 | a row that doesn't open its registered root, or a port named twice under `--registered` |
| D6 | a malformed `--zk` `coin_tree` |
| D7 | `sha512x3` or `hm96` nets other than the verifier's |
| D8, D9 | no unit in a VU; no VU in a block, or a count that isn't a power of two |
| D10 | an `identifier` other than the verifier's own `--identifier F` |
| D11 | an output group past its slot |
| D12 | a statement form, sampling design or session the soundness proof doesn't cover (`Flock.ProvedScope`), and a `--zk` mask slot below 2^13 bits or met by Δ or a region |
| D13 | a `rec-open` unit whose name, bits, rows or net aren't the verifier's |
| D14 | a sample that doesn't take in full each unit reading a round's commitment |
| D15 | an inner statement's registered description that isn't the registration |
| D16 | a V* session whose instance labels aren't its sampled proof blocks |
| D17 | a plain file in the form its statement doesn't pin, or a byte after a proof's end |
| D18 | a non-empty `matrix_evals` |
| D19 | a V* cap session that doesn't read its level's tops at the verifier's shared rows |
| D20 | a record not in its canonical text |
| D21 | a nonzero proof-of-work nonce |
| D22 | the verifier's own `--program` with a reference that reads a leaf at another width than its type gives that position (`E_REINTERPRET`, `verity/ml/circuits/PROTOCOL.md` §4.4): `Flock.Program.Widths`, which `NciCheck` decodes with too, refuses it at setup with the codec's message; upstream reads no program. `qword_program_agree.py --widths` holds the decoder to the codec on NCI's mistyped descriptors |

Where the paper is silent, the verifier follows upstream:

| id | topic | the verifier |
|---|---|---|
| A1 | zerocheck coins | six extra coins `r_skip`, squeezed and never read |
| A2 | last coin | Ligerito's final claim-batching coin is read, but an honest proof passes for every value |
| A3 | vanishing on S | imposed by interpolating over `S ∪ Λ` with `S = 0`: degree ≤ 127 |
| A4 | `v†` | the prover sends `final_c_eval`; the verifier compares it with its interpolation |
| A5 | residual sumcheck | `(G(1), G(∞))` of the eq-stripped quadratic, `G(0)` derived, `inv(0) = 0` |
| A6 | pinned coordinates | address bits 6–8 in F8 via φ8, 9–12 in F |
| A7 | lincheck | the verifier computes `α·A_0ᵀe + B_0ᵀe`; `k_log − 6` rounds, top bit first; 64 skip values; a fresh φ8 coin |
| A8 | constant wire | an extra coin `β` on the pin column, which must be 1 in every block |
| A9 | ring switching | each claim switched with its own 7 coins, then batched; Ligerito's K sumcheck replaces the standalone one |
| A10 | packing | the 7 low address bits, basis `x^v` |
| A11 | Ligerito | stratified queries, K folding challenges, out-of-domain at level 0, a clear residual of `2^e` values |
| A12 | proof of work | none: every nonce must be `0` |
| A13 | statement binding | label, statement digest and cap absorbed before the first coin |
| A14 | extra claims | region claims at the link points, opened in the same batch, their points and values not absorbed |
| A15 | `root_F` | required to be Σ |
| A16 | `m > 35` | refused: no `fast100` schedule |

## The other scripts

Each holds one Lean part to its Python or Rust counterpart, or generates inputs for the agreement job:

| script | what |
|---|---|
| `transcript_check.py` | rebuilds every round of a session independently of the Lean (framing, the order of absorbs and squeezes, the batched target) and compares each round's digest and coin count with the record |
| `redigest.py` | proof mutants: one value of one rep edited, its round digests rebuilt, every coin kept; `--retain` writes retained-bytes sessions |
| `fuzz.py` | seeded single mutations of an honest session (proof bytes, values, coins, digests, record fields) |
| `selftest_records.py` | the prover's selftest cases as replayable sets |
| `record_text.py` | the record's canonical text against Lean's |
| `zk_mutants.py` | `--zk` session mutants |
| `template_query_agree.py`, `qword_agree.py`, `qword_program_agree.py`, `qcall_agree.py`, `cut_check_agree.py`, `unit_cut_agree.py`, `layout_agree.py`, `circuit_type_agree.py` | each Lean part against its Python counterpart in `verity.ml` |
| `firewall_agree.py`, `firewall_controls.sh`, `firewall_finish_agree.py`, `firewall_finish_agree.sh` | the ZK server's contract and session loop against the inner phase's proxy, and each tamper control held to a visible refusal |
| `tool.py` | the agreement job's store declaration: `ci-pod.sh` with its settings, the shipped bundle as its input |

`vectors.json` lists the honest sessions, the record-level negatives, the forgeries and the replayable and live sets; the
bytes stay in the evidence store. `lookup_v2_vectors.json` holds `Flock.Lookup.buildV2`'s tables and their
`rows_sha512`, which the Rust and Python mirrors must reproduce.

## Tests

`verity/ml/flock/tests/test_lean_verifier.py` pins the toolchain, rejects `native_decide`, `implemented_by`, `extern` and
`unsafe` in the sources, and runs the Lean-backed tests when `lake` is on the PATH; the `test_lean_*.py` modules verify
fetched fixtures (`fixtures/artifacts.json`) with the built verifier.
