# Flock verifier of record

`PROTOCOL.md` is the specification. `lean/` is the executable verifier written from it, in Lean 4 (toolchain pinned in
`lean/lean-toolchain`, no dependencies). It replays recorded sessions offline: interactive coins only, CPU only. Upstream's
Rust verifier is the cross-validation oracle (§17.1).

Scope today: the circuit statement in strict mode, without lookup slots. It covers two versions of PR #83: `9294e161`
(tags `verity/flock-netlist/v1`) and `19c7269a` (tags `verity/flock-circuit`), both with the keyed-BLAKE3 row leaf. The
byte tags (`Flock.Tags`) and the in-circuit row leaf (`Flock.RowLeaf`) are parameters. The legacy module (pure-block, IR
frame) is not written yet.

## Build and run

~~~text
cd backends/flock/verifier/lean && lake build          # .lake/build/bin/flock-verify
flock-verify verify [--statement ID] --circuit F --comp comp.rows --public pub.bin [--coins seed|os] --session DIR...
flock-verify statement [--statement ID] --circuit F --comp comp.rows --public pub.bin     # digest, Σ, Hello
~~~

`ID` is `verity/flock-circuit` (the default) or `verity/flock-netlist/v1`. Each `DIR` holds `session.json` (the
`flock-live-session/v1` record) and `<table>.rep0.bin`, `<table>.rep1.bin` (the proofs the server received; the table is
`circuit`, or `nl` for the older tags). The statement is set up once (about 20 s: `D_b3` is recomputed from `comp.rows`). The command prints
one `VERDICT {"accepted", "session", "why"}` line per session and exits 0 iff all are accepted. `field-test`,
`hash-test` and `comp-check` run the component checks.

`backends/flock/tests/test_lean_verifier.py` pins the toolchain and rejects `native_decide`, `implemented_by`, `extern` and
`unsafe` in the sources.

## Inputs and vectors

| artifact | what |
|---|---|
| `art:04259cdd` | `comp.rows`: the BLAKE3 compression circuit (`flock-r1cs-rows/v1`), `D_b3 = 3f96431c…` |
| `art:799dab88` | the RoPE stage: circuit file `net.txt`, the verifier's public file `pub-8.bin`, the prover's `inst-8.bin` |
| `art:ef080b64` | replayable sessions `honest/` (accept) and `r-break/` (reject S13/R1), with upstream's verdicts, at `9294e161` |
| `art:5715c8f9` | the RoPE stage at `19c7269a`: `circuit.txt`, `pub-8.bin`, `inst-8.bin` |
| `art:b9e86ae1` | `honest/` and `r-break/` at `19c7269a` |

~~~text
research data fetch art:ef080b64 --to /tmp/v     # likewise the other two
~~~

`redigest.py SESSION_DIR OUT_DIR` writes the proof mutants: one value of one rep's proof is edited, and that rep's round
digests and `proof_sha256` are rebuilt from the edited proof (with `transcript_check.py`'s spec transcript), keeping
every coin. Only the check the edit breaks can then reject. `OUT_DIR/mutants.json` lists each mutant's expected verdict.

## Agreement with upstream

The vectors patch of each version applies to its PR #83 commit (`git apply` in a checkout):
- `netlist-vectors.patch`, for `9294e161`, patches `flock-netlist`;
- `circuit-vectors.patch`, for `19c7269a`, patches `flock-circuit`.

Each patch adds three subcommands:
- `vectors` records the honest and R-BREAK sessions through a loopback transport;
- `export-comp` writes `comp.rows`;
- `replay --sessions DIR...` gives upstream's offline verdict (`Server::from_record`, then `finish`).

Each patch also fixes `from_record` for a table that commits no public words. Build inside upstream `b684b12` with PR #83's
patches applied (`cargo build --release -p flock-live --bin <binary>`). Then compare the two verifiers:

~~~text
python agree.py --lean flock-verify --upstream flock-circuit [--statement ID] --circuit F --comp comp.rows --public pub.bin \
    [--expect mutants.json]... --out DIR --sessions DIR...
~~~

It writes `agreement.tsv` and `result.json`, and exits 1 if the verifiers disagree, or miss an expected verdict, on any
session. Recorded runs: `art:f1f3c2aa` (`9294e161`) and `art:81645236` (`19c7269a`), 25 of 25 sessions agreeing in each.
