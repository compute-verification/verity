# Flock verifier of record

`PROTOCOL.md` is the specification. `lean/` is the executable verifier written from it, in Lean 4 (toolchain pinned in
`../../../core/service/flock/lean-toolchain`, no dependencies). It replays recorded sessions offline: interactive coins only, CPU only. Upstream's
Rust verifier is the cross-validation oracle (§17.1).

Scope today: the circuit statement in strict mode, including lookup slots (built from the pinned MUFU tables), tail stages
and wires. The default, `verity/flock-circuit`, is the format of C-Flock's circuit prover (M0) at commit `eb90718f` (§16.10):
- serving rows are `hm96-sha512/row/v1` leaves proved in the circuit;
- every identity and binding digest is SHA-512;
- the proof's Merkle leaves are `hm96-sha512/v1`;
- the unit draw is live.

The older versions keep their names: `9294e161` (`verity/flock-netlist/v1`), `19c7269a`, `fd02e847` and `631567f7` (the
keyed-BLAKE3 row leaf, SHA-256 or SHA-512 unsalted trees). Four things are parameters: the byte tags (`Flock.Tags`), the in-circuit row leaf (`Flock.RowLeaf`), the proof's
Merkle scheme (`Flock.MerkleScheme`: SHA-256, or SHA-512 with 64-byte siblings) and the Halevi–Micali leaf (`Flock.Hm96`,
generic over the hash). The legacy module (pure-block, IR frame) is not written yet.

`verify` fails closed (§16.15): every session of a statement whose form the soundness proof does not cover is refused
with `refused: <why>: a statement form the soundness proof does not cover` (`Flock.ProvedScope.check`), once every check
has passed, so a session that fails a check keeps that check's reason. That refuses every version before `hm96-sha512`
rows, the typed statements, public input ports, several tables without `--zk` (unless `--plain-tables`: a statement with
public outputs, or hidden outputs and `hm96-sha512/v1` leaves, at up to 64 tables, at a subset draw, with the verifier's
own `--program` the one META's `program_sha512` names, `ProvedScope.checkP` and `ProvedScope.lawWhy` at `plainTables`,
which `flock_verify_sound`'s `SoundExecJ` and `SoundExecHJ` cover) and more than 2^16 with it (`ProvedScope.zkTableCap`),
shared-row public files (except at hidden outputs and `hm96-sha512/v1` leaves with the verifier's own `--registered F`,
which `FailClosed.SoundZHJR` covers under `--zk` and `FailClosed.SoundExecHR` without it), and circuits
outside `Layout.scopeOk` (with public outputs: a unit reading more than one input group, a tail that computes the outputs,
a `sha512/row/v2` or segmented row, among others). Under hidden outputs the circuit check is `ProvedScope.scopeOkI`, the
general instance form: a unit reading any number of input groups and tail stages returning the outputs (a repeated last
stage from any of its slots, `out_net_ports` of `[slot, port]` pairs), once the wires
compute every slot from the leaves (`ProvedScope.wiresAcyclic`), with `sha512/row/v2` and `sha512/row-seg/v1` rows.
Those statements still parse, set up and verify (`statement` prints their digest and Σ), and the agreement check
compares a session refused for its form alone as accepted (D12, `vectors.json`'s `lean_refuses`). A session drawn by a
sampling design no soundness claim draws by is refused at setup, with
`refused: <why>: no soundness claim draws by this law` (`Flock.ProvedScope.drawLaw`): a bernoulli draw, a work draw with
or without a closure, and a subset draw without `--zk` at a statement with public outputs (`lean_refuses_drawn`). So is a session without a draw, with
`refused: a session without a draw: every soundness claim is about a drawn session (D12)`, last, after the form's
refusal (`lean_refuses_drawless`). Under `--rec-inner`, a V* session whose draw is not the `subset` draw of every unit of
its population is refused after that, and so is a recursive session (`--rec-part`, `--rec-bindings` or `--rec-rows`)
without `--rec-inner` (`Flock.RecSetup.fullDraw`, PROTOCOL.md). Last of all, a session verified without the verifier's
own `--program` is refused (`refused: a session without the verifier's program (--program): …`,
`Flock.ProvedScope.programHeld`), with or without `--zk`: a verifier that cannot read the Program cannot know what it
verifies, and the agreement check compares a session refused for that alone as accepted (`agree.py`'s `NO_PROGRAM`).

A recursive statement is one `verify --zk` per V* part, and no part sees another. Each part's `VERDICT` names its part
(`rec_part`) and the SHA-512 of the shared rows file it checked (`rec_rows_sha512`). `flock-verify rec-compose
--rec-inner F --rec-rows F LOG...` (`Flock.RecCompose.check`) gives the statement's verdict: it accepts only when there
is one log per V* statement and every part is accepted at the one rows file the prover staged (`acc0`, the tops, the
`rec-c<k>` rows). The pod's Lean step (`85-rec-reprice.sh`, `lean_verdicts.py --compose`) gives every part that one file.

## Build and run

~~~text
cd verity/core/service/flock && lake build          # .lake/build/bin/flock-verify
flock-verify verify [--statement ID] [--archive DIR] --circuit F [--comp comp.rows] --public pub.bin [--tables DIR] [--coins seed|os] [--upstream-plain] --session DIR...
flock-verify statement [--statement ID] --circuit F [--comp comp.rows] --public pub.bin [--tables DIR] [--coins seed|os]   # digest, Σ, Hello
~~~

Each plain proof file is taken in one form (PROTOCOL.md §8.1): by default the deployed prover's, the zk patch's build,
which writes a `zk_level0 = None` byte after the pair; `--upstream-plain` takes upstream's builds' form, without it, as
the agreement job's sessions from upstream's binaries are. Each refuses the other.

`ID` names a version:
- `verity/flock-circuit`: the default, and the statement M0 stages today: `@967b8d06`'s with every output a hidden row
  (§16.13), so the public file holds roots, hiding leaves and structure only (M0's transcript is not hiding; the soundness
  theorem covers `Stmt.setupHidden`'s statements in the forms of §16.15, `FailClosed.flock_headline_exec_hidden`). Before
  #757 the name meant `@e51e2b86`, and `/types` meant `types@210d32e1`.
  It needs no `--comp`, and it pins
  `seed_injection: false`, as a proving build does. A circuit that binds a `verity/partition/v1` digest needs
  `--partition F`, the verifier's own copy of the partition object (§16.10).
  Its verdicts carry `partition = {rule, query, digest, units, note}`: `units` is `as stated` (not derived) until the verifier
  decodes the program and evaluates the query itself (§16.10).
- `verity/flock-circuit+seed-injection`: the same statement as the circuit prover's (M0) selftest build pins it, for its recorded cases.
- `verity/flock-circuit+plain-leaves`: the same statement with plain SHA-512 Merkle leaves (`flock-leaf/sha512-unsalted`,
  §16.16), a recursive session's inner statement (`FLOCK_PLAIN_LEAVES=1`), ZK off. `verify` checks it without `--zk`,
  under the hidden-output headlines at the plain leaf (`Refine.leafOfId`), and refuses it under `--zk`.
- `verity/flock-circuit/types` (and `+seed-injection`): the typed statement (§16.11) with hidden outputs (§16.13);
  `verity/flock-circuit/types@210d32e1` is its form with public outputs.
- `verity/flock-circuit@cec3e347` and `verity/flock-circuit/types@cec3e347` (and `+seed-injection`): the two before, for
  statements composed up to verity `cec3e347`, whose `--zk` identity names the pads and every table's draws ChaCha20
  streams of the prover's seed; the default's names the OS's bytes (§16.14).
- `verity/flock-circuit@b3727bd9` and `verity/flock-circuit/types@b3727bd9` (and `+seed-injection`): `@cec3e347`'s, for
  statements composed up to verity `b3727bd9`, whose META `leaf_scheme.salt_source` names a ChaCha20 expansion of a
  per-proof OS key; from `@cec3e347` on it names the OS's bytes, each tree's read once (§16.10, §16.13).
- `verity/flock-circuit@967b8d06` (and `+seed-injection`): PR #83 from `68ae79f2`, whose identity names the round digest,
  statement digest and Σ as SHA-512, and from `967b8d06` shared-row public files (header `shared_rows`, §16.10), with
  public outputs. Runs staged from `68ae79f2` until hidden outputs use this name. A proving build's session of this
  statement or a later one (the hidden-output and typed ones) on per-round OS coins (`FC_COINS=os`) needs `--coins os`,
  which binds the identity that mode writes, not who drew the coins (PROTOCOL.md §7.1, §17.2); a seeded one needs
  `--coins seed`, the default.
- `verity/flock-circuit@e51e2b86` (and `+seed-injection`): PR #83 from `e51e2b86`, for the runs staged before `68ae79f2`.
- `verity/flock-circuit@eb90718f` (and `+seed-injection`): the same without the e2e bindings.
- `verity/flock-circuit@631567f7`, `@fd02e847` and `@19c7269a`, and `verity/flock-netlist/v1`. These need `--comp`.

A session whose record carries a `unit_draw` is verified against the drawn units' statement, which the verifier derives
from the population's public file and the draw (§7.3, §16.10).

`--tables` names the directory of the pinned MUFU tables (`<name>.u32`), for circuits with lookup slots. Each `DIR` holds `session.json` (the
`flock-live-session/v1` record) and `<table>.rep0.bin`, `<table>.rep1.bin` (the proofs the server received; the table is
`circuit`, or `nl` for the older tags). The statement is set up once (about 20 s: `D_b3` is recomputed from `comp.rows`). The command prints
one `VERDICT {"accepted", "session", "why"}` line per session and exits 0 iff all are accepted. The component checks are `field-test`, `hash-test` (SHA-256, SHA-512, BLAKE3), `merkle-test`,
`hm96-test` (on PR #88's `hm96/vectors.json`) and `comp-check`. The toolchain is v4.34.0, the revision the soundness
package's Mathlib and ArkLib pin, so that one build covers both.

`verity/ml/flock/tests/test_lean_verifier.py` pins the toolchain and rejects `native_decide`, `implemented_by`, `extern` and
`unsafe` in the sources.

## Inputs and vectors

| artifact | what |
|---|---|
| `art:04259cdd` | `comp.rows`: the BLAKE3 compression circuit (`flock-r1cs-rows/v1`), `D_b3 = 3f96431c…` |
| `art:799dab88` | the RoPE stage: circuit file `net.txt`, the verifier's public file `pub-8.bin`, the prover's `inst-8.bin` |
| `art:ef080b64` | replayable sessions `honest/` (accept) and `r-break/` (reject S13/R1), with upstream's verdicts, at `9294e161` |
| `art:5715c8f9` | the RoPE stage at `19c7269a`: `circuit.txt`, `pub-8.bin`, `inst-8.bin` |
| `art:b9e86ae1` | `honest/` and `r-break/` at `19c7269a` |
| `art:d9837120`, `art:bd9f7efb` | the RoPE stage and its vectors at `fd02e847` (the final format) |
| `art:fb5157c5`, `art:77cb4bec`, `art:925d6833` | RMSNorm Triton n128 at `fd02e847` (two lookup slots, three tail stages): stage, vectors, the MUFU tables |

~~~text
research data fetch art:ef080b64 --to /tmp/v     # likewise the other two
~~~

`transcript_check.py` (PROTOCOL.md §18's spec self-check) matches every recorded round on the first vector of all 50 honest
sets, both reps, `m` 25 to 34 (`art:430c513a`).

`redigest.py SESSION_DIR OUT_DIR` writes the proof mutants: one value of one rep's proof is edited, and that rep's round
digests and `proof_sha256` are rebuilt from the edited proof (with `transcript_check.py`'s spec transcript), keeping
every coin. Only the check the edit breaks can then reject. `OUT_DIR/mutants.json` lists each mutant's expected verdict.

## Agreement with upstream

The vectors patch of each version applies to its PR #83 commit (`git apply` in a checkout):
- `circuit-vectors-eb90718f.patch`, for `eb90718f` (build Flock with PR #83's `flock-sha512-b684b12.patch` and `flock-live`
  with `--features sha512 --features seed-injection`), patches `flock-circuit`. It has `vectors`, which records the honest
  session (this prover refuses R-BREAK's second level-0 tree under shared salts), and `replay`, which replays drawn records
  under the drawn statement (`--drawn-only`: only the drawn units' statement is built). It drops `forgeries`, and it records
  the selftest's two draw cases under `--record-dir`.
- `netlist-vectors.patch`, for `9294e161`, patches `flock-netlist`;
- `circuit-vectors.patch`, for `631567f7` (the SHA-512 statement: build Flock with PR #83's `flock-sha512-b684b12.patch` and
  `flock-live` with `--features sha512`), patches `flock-circuit`. Its `fd02e847` version is in commit a2c4e9fd and its
  `19c7269a` version in 9605a77d.

Each patch adds three subcommands:
- `vectors` records the honest and R-BREAK sessions through a loopback transport;
- `export-comp` writes `comp.rows`;
- `replay --sessions DIR...` gives upstream's offline verdict (`Server::from_record`, then `finish`).

The two older patches also fix `from_record` for a table that commits no public words; PR #83 carries the fix from
`a2a7e7f4` on. Build inside upstream `b684b12` with PR #83's
patches applied (`cargo build --release -p flock-live --bin <binary>`). Then compare the two verifiers:

~~~text
python agree.py --lean flock-verify --upstream flock-circuit [--statement ID] --circuit F [--comp comp.rows] --public pub.bin \
    [--expect mutants.json]... --out DIR --sessions DIR...
~~~

`--jobs J` runs `J` verifier processes at once: each public file's sessions are dealt into chunks of four or more, one Lean
and one upstream process per chunk, and the verdicts are reassembled, so every output is the same as with `--jobs 1`. `ci.py
--session-jobs K` shares `K` such processes among all the sets it runs (by default the cores, and the available memory at 8
GB a process). Set 8 on a 4-core VM: 156 s with one process, 62 s with four.

It writes `agreement.tsv` and `result.json`, and exits 1 if the verifiers disagree, or miss an expected verdict, on any
session. At `eb90718f` (RoPE, 4 instances; `vectors.json` set 8, `art:c49cb61e`; recorded run `r20260927-060508-66c0`,
result `art:294b6cf1`), 25 of 25 sessions agree:
- the CPU selftest's 19 recorded cases (`selftest --record-dir`, `selftest_records.py`);
- its two draw cases (a drawn session, and one that ignores the draw);
- the honest vector;
- two draw negatives (the record's draw altered, and dropped);
- a salt mutant (an opened salt flipped, with `proof_sha512` rebuilt), which Lean rejects at the Merkle opening.

Triton RMSNorm n128 at `eb90718f` (lookup slots; `vectors.json` set 9, `art:187679d7`, recorded on a CPU pod in
`r20260927-061936-560b`): every selftest case, the draw cases included, 25 of 25. Its unit reads two input groups, so it
is D12 since §16.15: the Lean verifier refuses every session, and upstream gives each its verdict. `agree.py` replays each
drawn session through upstream in its own process (`replay --drawn-only`), since upstream leaks a statement's matrices
and two RMSNorm statements do not fit in 15 GB.

Recorded runs:
- `art:f1f3c2aa` (`9294e161`): 25 of 25 sessions agree;
- `art:81645236` (`19c7269a`): 25 of 25 agree;
- `art:78ef0a6e` (`fd02e847`): 27 of 27 agree, with the retained-bytes sessions.

These versions are before `hm96-sha512` rows, so their sets (0 to 7) are D12 since §16.15.

`redigest.py --retain` writes a session with every round's framed bytes retained, and two negatives of it.

Two more inputs feed the agreement check:
- **Forgeries.** The patch's `forgeries` subcommand records every PR #83 selftest forgery that runs a session as a
  replayable set, with the expected verdict in `case.json`. A forgery that is a false statement carries its own `pub.bin`,
  which `agree.py` picks up. RoPE at `fd02e847`: `art:d306dfbd`, 14 of 14 agree.
- **Fuzzing.** `fuzz.py SESSION_DIR OUT_DIR --seed S --n N` writes seeded single mutations of an honest session: proof
  bytes, proof values with rebuilt digests, coins, round digests and record fields. Its `fuzz.json` gives `agree.py` the
  cases that have a fixed expectation.

`ci.py` is the agreement job. It takes every replayable set of `vectors.json` that has an upstream binary, fetches its
inputs from the store, regenerates the mutants, retained-bytes sessions and fuzz cases, adds its forgeries, and runs
`agree.py`:

~~~text
research run --tool flock-agreement --campaign flock-verifier -- python ci.py --lean flock-verify \
    --upstream fd02e847=flock-circuit --upstream 9294e161=flock-netlist --work DIR --seed S --fuzz N
~~~

### The statements' own tests and sets

- Typed statement (§16.11): on RoPE (`D = 64`) and SiLU·mul (`I = 8192`) the derived rows are the old statement's unit
  section byte for byte (`flock-verify typed-expand`, `test_lean_typed_statement.py`); on #273's GEMM the statement sets
  up and the verifier refuses all 20 of the prover's recorded CPU selftest sessions, the 3 honest ones too, for their
  form (§16.15, `test_lean_typed_template.py`); on typed attention the derived rows are the writer's `rows/` byte for
  byte (`test_lean_typed_reads.py`).
- Template queries (§16.10): `template_query_agree.py` checks the decoding against #131's
  `template_instance_vectors.json` (16 of 16: the population, the ranges, every unit's node, member and gates, the
  refusals word for word, the malformed parameters).
- The outer firewall (`Flock.Firewall.Session`, `flock-firewall session`, live PROTOCOL.md §10): the Lean session loop
  owns the auditor's connection and drives each firewalled rep's frame (`Flock.Firewall.Frames` over
  `Flock.Firewall.Frame`). It places every round (the algebraic phase's scheduled round, `Skeleton.rounds`; the
  opening's, `Drive.openStep`; or the inner proof's, the firewall's own), takes the device's round unmasked from the
  prover (`firewall::Gate`), makes the pass decision against the round in its place (`check`), masks what passes and
  releases the scheduled round in place of one that fails. The rep's pads and the pads tree's salts are the frame's own
  OS draws, with the root it computes, and the coins' four refusals (an outer coin 1, the mask slot's rank, level 0's
  `ρ_0, ρ_1` dependent over F128, the inner proof's `β = 0`) are the frame's; the inner proof's rounds leave through it
  too, in the form it fixes, with `(ρ, σ)` and `Y` its own from its draws and the coins (`innerOwn`), so of the inner
  proof's rounds only `final_c'` and `τ`'s commitment are Rust's. A rep the frame stops ends the session at `Finish`, not
  accepted. `flock-firewall tap` serves the prover's every other OS read (`test_firewall_frame.py` on hand-made reps: the
  draws, the release, the pass decision, the refusals, the inner proof's rounds against `zk_veil::prove_inner`'s
  formulas, the coin and prior protocol, the tap; the ρ and β stops on `live/src/coin_stop_vectors.json`, which
  `zk_veil`'s `rust_coin_stops_agree_with_the_lean_frames_vectors` holds Rust's `rho_independent` and `prove_inner` to;
  the pass decision on `live/src/pass_vectors.json`). `firewall_controls.sh` runs an honest firewalled session and each
  `FC_FIREWALL_TAMPER` control through the loop, and holds each control to a visible refusal.
- The firewall's contract (`Flock.Firewall.Contract`, `flock-firewall contract`), the property every firewall's sessions
  must have, beside the inner phase's program (`Flock.Firewall.run`, which `test_firewall_agreement.py` holds the proxy
  to): `firewall_agree.py` runs the inner phase's firewall (`rec_live.Proxy`) and the contract on the same transcript
  (`verity_flock.firewall_transcript`) over 22 cases: the honest sessions accepted by both, the red team's six on #1270
  refused by both or sent as the schedule places them, and tamper controls the firewall passes and the contract refuses.
  The contract holds each transcript to the verifier's public input (`flock-firewall contract --public FILE`,
  `verity/firewall-public/v0`: the schedule, the relations, the statement's values and the names of the firewall's own
  values), never to the transcript's own copy; without one it refuses whatever leans on it. `test_firewall_contract.py`
  also cuts each of review r6's leaks on #1323 at its item (`firewall_agree.leaks`: the commitments' count and place, a
  tape number naming another's value, the public input from the transcript, a name of the firewall's choosing), and runs
  the contract on every session of `test_firewall_agreement.py`. `--sessions`
  replays recorded sessions from their coins and salts. The K = 4096 sessions it replayed before (`r20261006-070620-e02d`'s,
  4 of 4 accepted by both) are `rec/inner-transcript/v0`'s, from before rec-step3's tops and level0, and today's firewall
  refuses their schedule. A v1 session at that size has been recorded since (`r20261006-021550-4e4b`, rec-reprice's
  `85-rec-reprice.sh STEP=inner K=4096`), but its outputs keep no `private.json`, whose salts the replay needs. The outer phase's rounds are the Lean session loop's
  own (the bullet above), not a transcript the contract decides after the fact.
- Set 13 (GEMM k1024, tail outputs, recorded on a CPU pod) agreed 27 of 27, and set 14 (a shared-row RoPE file whose six
  instances repeat three rows) 23 of 23 with upstream on M0's selftest records, `row_ref_claim_false` included, before
  §16.15; both are D12 now (a unit reading two input groups, a shared-row file), so the Lean verifier refuses every
  session. Set 15 (the same instances per instance) agrees 22 of 22. The `967b8d06` vectors patch writes each false
  statement as the verifier's file (`<case>.pub.bin`) under its real public digest, so those negatives reach the proof's
  openings, not the session parameters.
- Hidden outputs (§16.13): PR #83's pinned build has none of these statements, so check's agreement step replays none,
  and the agreement is between Lean and this repository's Rust prover, not upstream's verifier. `vectors.json`
  `live.sets` 2 to 5 are M0's selftest records at `455a01049` (flat and typed, RoPE and the GEMM coordinate;
  `ci.py --live`: 20/20, 21/21, 20/20, 21/21, reproduced by red-team-proofs-554 in `art:bc6367e3`). On flat RoPE the honest
  sessions pass every check, so digest and Σ are the prover's (they have no draw, so their verdict is that refusal, D12),
  and every negative is refused by its check, `output_row_committed_false` (an
  output row's `b ‖ c` not the circuit's outputs) included. Flat GEMM (a unit reading two input groups, a tail stage
  returning the outputs) is the general instance form, which the check passes, so its honest sessions meet the same
  draw-less refusal (set 3: 21/21); both typed sets are D12 (§16.15), every session refused for its form. `test_lean_hidden_outputs.py` verifies four of each set's
  cases (the honest session and the three false statements) from a fetched fixture
  (`verity/ml/flock/tests/fixtures/hidden-outputs`), so `check` runs it. A replayable set needs a pinned upstream build
  with `flock-circuit replay` (the selftest's `record_replays_offline` already replays in process).
  `test_lean_registered_reads.py` stages its registered-read statements over a hidden-output RoPE circuit (`statement`,
  and `verify` of each of two tables' setups), and set 16's upstream session covers `@967b8d06` (a shared-row file, D12:
  it sets up under its reads and is refused for its form).
- `--zk` sessions (§16.14): no pinned upstream build has them, so check's agreement step replays none. `agree.py --zk`
  compares Lean's verdicts with the live build's `replay --zk`, and `test_lean_zk.py` with its server's, on the sessions of
  `art:8126390c` (`@967b8d06+seed-injection`).
