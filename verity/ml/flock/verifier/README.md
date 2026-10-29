# Flock verifier of record

The verifier of record is the Lean verifier in `../../../core/service/flock/` (`flock-verify`, Lean 4, toolchain pinned
in its `lean-toolchain`, no dependencies). `PROTOCOL.md` here is its outline, section by section. It replays recorded
sessions offline: interactive coins only, CPU only. Upstream's Rust verifier is a cross-check, not a verifier (below).
This directory holds the outline, the agreement scripts and the vectors.

## What it verifies

The default statement, `verity/flock-circuit`, is the format of C-Flock's circuit prover (M0, `verity_flock.circuit`):
serving rows are `hm96-sha512/row/v1` leaves proved in the circuit, every output is a hidden row, every identity and
binding digest is SHA-512, the proof's Merkle leaves are `hm96-sha512/v1`, and the unit draw is live. Its variants:

- `verity/flock-circuit/types`: the typed statement (`PROTOCOL.md` §16.11);
- `verity/flock-circuit/tmpl`: the templated statement, which the verifier parses and refuses until its proof lands;
- `verity/flock-circuit+plain-leaves`: a recursive session's inner statement, plain SHA-512 leaves, checked without
  `--zk` (§16.16);
- `+seed-injection` on any of them: the statement as the prover's selftest build pins it, for its recorded cases.

`verify` fails closed (§16.15): a session whose statement form, draw or setting no soundness theorem covers is refused,
once every check has passed, with `refused: <why>: …`, so a session that fails a check keeps that check's reason.
`flock_verify_sound` is the one theorem over everything it accepts (`Security/Proofs/Flock/`). Under `--rec-inner`
that includes a session at a holo or sparse inner shape (`Flock.RecSetup.denseOnly`: `RecursiveKS` covers the dense
form only, and the compiled model refuses the same sessions, `RecDeploy.acc_dense`) and a recursive session without
`--zk` (`Flock.RecSetup.zkOnly`).

A recursive statement is one `verify --zk` per V* part, and no part sees another. Each part's `VERDICT` names its part
(`rec_part`), the SHA-512 of the shared rows file it checked (`rec_rows_sha512`), of the inner shape file it read
(`rec_inner_sha512`), of the inner session's record and its transcript (`rec_inner_session_sha512`,
`rec_inner_transcript_sha512`) and of every input V*'s parts share (`rec_shared`: the inner shape, the inner
statement's files and how it is set up, the inner session's record, the coin server's transcript, the registration),
whether it ran `--zk` and its session's coin key. `flock-verify rec-compose --rec-inner F --rec-rows F LOG...`
(`Flock.RecCompose.check`) gives the statement's verdict: it accepts only when there is one log per V* statement with
one `VERDICT`, every part is accepted under `--zk` without an identifier at the one rows file the prover staged (`acc0`,
the tops, the `rec-c<k>` rows) and the composition's own inner shape, every part read one set of the other shared
inputs, in the compiled verifier's form, and no two sessions share a coin key (`RecCompose.Accepts`, `check_accepts`;
`Proofs.Flock.Claims.Recursion.ComposeParts.composed_parts` reads it at the compiled verifier's parts).
`verity_flock.rec_compose` is its Python mirror (`tests/test_lean_rec_compose.py`). The pod's Lean step
(`85-rec-reprice.sh`, `lean_verdicts.py --compose`) gives every part those files.

It doesn't check that the parts' sessions are one run of the compiled verifier: that is the open condition C10
`OneRun` (one session per part, every session the coin server opened for the statement's parts recorded, the staging
before the first part's Hello), which needs the service's interactive-run record, a later PR. Until it holds, a prover
can retry a part and hand over only the accepted session, and the recursive verdict is not a claim a guarantee backs.

**Archived formats.** The statement names pinned at older commits (`verity/flock-circuit@<commit>`, back to
`verity/flock-netlist/v1`) are still parsed, so recorded sessions keep their names, and every one before `hm96-sha512`
rows is refused for its form. Those with keyed-BLAKE3 row leaves and SHA-256 trees (`@19c7269a`, `@fd02e847`,
`@631567f7`, `verity/flock-netlist/v1`, which need `--comp`, the BLAKE3 compression circuit) are archived: the Lean
verifier and `hash-test` still carry BLAKE3 for them alone. The pure-block and IR-frame statements are not in the Lean
verifier.

## Build and run

~~~text
cd verity/core/service/flock && lake build          # .lake/build/bin/flock-verify
flock-verify verify [--statement ID] [--archive DIR] --circuit F --public pub.bin [--tables DIR] [--coins seed|os] [--zk] \
    [--program F] [--partition F] [--registered F] [--upstream-plain] --session DIR...
flock-verify statement [--statement ID] --circuit F --public pub.bin [--tables DIR]   # digest, Σ, Hello
~~~

- `--archive DIR` reads every object by its SHA-512 from the verifier's own store (§16.8); without it the flags are
  paths, for the development and agreement tools.
- `--program`, `--partition` and `--registered` are the verifier's own copies of the program, the partition object and
  the registered reads (§16.10). Without `--program` every session is refused: a verifier that cannot read the Program
  cannot know what it verifies.
- `--tables DIR` holds the pinned MUFU tables (`<name>.u32`), for circuits with lookup slots.
- `--coins os` binds the identity of a proving build's session on per-round OS coins (`FC_COINS=os`); `seed`, the
  default, a seeded one (§7). A `--zk` session's coins are its coin tree.
- Each plain proof file is taken in one form (§8.1): by default the deployed prover's, which writes a `zk_level0 = None`
  byte after the pair; `--upstream-plain` takes upstream's builds' form, as the agreement job's sessions from upstream's
  binaries are.

Each session `DIR` holds `session.json` (the `flock-live-session/v1` record) and each table's `<table>.rep0.bin` and
`<table>.rep1.bin`. `verify` prints one `VERDICT {"accepted", "session", "why", …}` line per session and exits 0 iff all
are accepted. The component checks are `field-test`, `hash-test`, `merkle-test`, `hm96-test` (on
`verity/ml/commitments/hm96/vectors.json`) and `comp-check`.

## Agreement with upstream

`ci.py` is the agreement job (`check`'s `lean-agreement` step). For every replayable set of `vectors.json` that has an
upstream binary (`upstream.json`, built by `upstream-build.sh` from `ci-bundle.sh`'s bundle, run by `ci-pod.sh`), it
fetches the set's inputs from the store, regenerates its proof mutants (`redigest.py`), retained-bytes sessions and
seeded fuzz cases (`fuzz.py`), adds its forgeries, and runs `agree.py`, which compares the Lean verifier's verdict with
upstream's on every session:

~~~text
research run --tool flock-agreement --campaign flock-verifier -- python ci.py --lean flock-verify \
    --upstream fd02e847=flock-circuit --upstream 9294e161=flock-netlist --work DIR --seed S --fuzz N
python agree.py --lean flock-verify --upstream flock-circuit [--statement ID] --circuit F --public pub.bin \
    [--expect mutants.json]... --out DIR --sessions DIR...
~~~

`agree.py` writes `agreement.tsv` and `result.json`, and exits 1 if the verifiers disagree, or miss an expected verdict,
on any session. `--jobs J` deals each public file's sessions into chunks run by `J` process pairs, with the same output.
The Lean verifier is deliberately stricter than upstream in the places `PROTOCOL.md` §17.1 lists (D1–D21); a set marked
`lean_refuses`, `lean_refuses_drawn` or `lean_refuses_drawless` compares a session refused for its form, draw or missing
draw alone as accepted.

Each version's vectors patch (`netlist-vectors.patch`, `circuit-vectors*.patch`) applies to the M0 commit it names and
adds three subcommands to upstream's binary: `vectors` (record honest and negative sessions through a loopback
transport), `export-comp` and `replay --sessions DIR...` (upstream's offline verdict: `Server::from_record`, then
`finish`). Build inside upstream `b684b12` with M0's patches applied (`cargo build --release -p flock-live --bin <binary>`).

The statements no pinned upstream build has (hidden outputs, `--zk`) agree with this repository's Rust prover instead:
`ci.py --live` holds Lean's verdicts on `vectors.json`'s `live.sets` to the live server's, and `agree.py --zk` compares Lean's verdicts with the live build's
`replay --zk`.

Other scripts here hold one Lean part to its Python or Rust counterpart: `transcript_check.py` (the transcript and layout
rebuilt from `PROTOCOL.md` alone), `template_query_agree.py`, `qword_agree.py`, `qword_program_agree.py`, `qcall_agree.py`,
`cut_check_agree.py`, `unit_cut_agree.py`, `layout_agree.py`, `circuit_type_agree.py`, `record_text.py`, `zk_mutants.py`,
and the firewall's (`firewall_agree.py`, `firewall_controls.sh`, `firewall_finish_agree.py`): the ZK server's contract
and session loop against the inner phase's proxy, and each tamper control held to a visible refusal.

## Tests

`verity/ml/flock/tests/test_lean_verifier.py` pins the toolchain, rejects `native_decide`, `implemented_by`, `extern` and
`unsafe` in the sources, and runs the Lean-backed tests when `lake` is on the PATH; the `test_lean_*.py` modules verify
fetched fixtures (`fixtures/artifacts.json`) with the built verifier.
