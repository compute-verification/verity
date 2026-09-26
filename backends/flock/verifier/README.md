# Flock verifier of record

`PROTOCOL.md` is the specification. `lean/` is the executable verifier written from it, in Lean 4 (toolchain pinned in
`lean/lean-toolchain`, no dependencies). It replays recorded sessions offline: interactive coins only, CPU only. Upstream's
Rust verifier is the cross-validation oracle (§17.1).

Scope today: the circuit statement (PR #83 at `9294e161`, tags `verity/flock-netlist/v1`), strict mode, without lookup
slots. The legacy module (pure-block, IR frame) is not written yet.

## Build and run

~~~text
cd backends/flock/verifier/lean && lake build          # .lake/build/bin/flock-verify
flock-verify verify --circuit net.txt --comp comp.rows --public pub.bin [--coins seed|os] --session DIR...
~~~

Each `DIR` holds `session.json` (the `flock-live-session/v1` record) and `nl.rep0.bin`, `nl.rep1.bin` (the proofs the
server received). The statement is set up once (about 20 s: `D_b3` is recomputed from `comp.rows`). The command prints
one `VERDICT {"accepted", "session", "why"}` line per session and exits 0 iff all are accepted. `field-test`,
`hash-test` and `comp-check` run the component checks.

`backends/flock/tests/test_lean_verifier.py` pins the toolchain and rejects `native_decide`, `implemented_by`, `extern` and
`unsafe` in the sources.

## Inputs and vectors

| artifact | what |
|---|---|
| `art:04259cdd` | `comp.rows`: the BLAKE3 compression circuit (`flock-r1cs-rows/v1`), `D_b3 = 3f96431c…` |
| `art:799dab88` | the RoPE stage: circuit file `net.txt`, the verifier's public file `pub-8.bin`, the prover's `inst-8.bin` |
| `art:ef080b64` | replayable sessions `honest/` (accept) and `r-break/` (reject S13/R1), each with upstream's verdicts |

~~~text
research data fetch art:ef080b64 --to /tmp/v     # likewise the other two
~~~

`redigest.py SESSION_DIR OUT_DIR` writes the proof mutants: one value of one rep's proof is edited, and that rep's round
digests and `proof_sha256` are rebuilt from the edited proof (with `transcript_check.py`'s spec transcript), keeping
every coin. Only the check the edit breaks can then reject. `OUT_DIR/mutants.json` lists each mutant's expected verdict.

## Agreement with upstream

`netlist-vectors.patch` applies to PR #83 at `9294e161` (`git apply` in a checkout). It adds three subcommands to
`flock-netlist`: `vectors` (records the honest and R-BREAK sessions through a loopback transport), `export-comp` (writes
`comp.rows`) and `replay --sessions DIR...` (upstream's offline verdict, `Server::from_record` then `finish`). It also
fixes `from_record` for a table that commits no public words. Build it inside upstream `b684b12` with PR #83's patches
applied (`cargo build --release -p flock-live --bin flock-netlist`), then run `replay` on the same directories as
`flock-verify`. Both verifiers must agree on every session.
