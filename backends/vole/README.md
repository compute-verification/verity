# VOLE: QuickSilver on the WP2 checker

Frozen: kept, still in the tables, not developed. A designated-verifier ZK backend: the WP2 checker
(`verity_numerical.checker`) exported as a SIEVE Circuit-IR relation and proved with Diet Mac'n'Cheese, the QuickSilver
prover of `GaloisInc/swanky`. The verifier holds a secret MAC key, so a transcript convinces no one else; results carry
`zk_mode = "designated-verifier"` (`backends/AGENTS.md`). Older notes call it "Candidate C" or "Candidate V"; it is not
C-Flock.

- `PROTOCOL.md`: the cost model the lane started from.
- `ASSUMPTIONS.md`: the assumption stack as measured, read from swanky's source at the pinned commit.
- `export_ir.py`: the checker as a relation (the compressed v2 realisation; `export_ir_v1.py` is v1): `census`, `unit`,
  `tu`, `vu`, `negatives`.
- `patches/`: the swanky patch (a logUp lookup plugin, output-free multiplication checks, a cached LPN matrix,
  `TCP_NODELAY`); see its README.
- `bench.py`: runs the prover and verifier as two processes through a byte-counting relay.
- `run_all.sh`: the recorded measurement driver; `profile.sh` and `profile_buckets.py`: a callgrind profile of the prover.

swanky is not vendored: `run_all.sh` expects a checkout at `/workspace/swanky` at commit
`1b05bddab3e02c2cc4917d08c80a629286a4a4fc` with `patches/` applied, and refuses any other commit. The recorded runs are
the ledger rows `backends/numerical/reports/ledger/c-vole.jsonl` (relation v1) and `c-compress.jsonl` (v2).

## Running

There are no tests, and `check` runs nothing here. The census needs only the workspace:

~~~sh
uv run python backends/vole/export_ir.py census        # gate and committed-element counts, v2 against v1
~~~

Proving runs on a CPU pod, after building `dietmc` in the patched checkout:

~~~sh
python3 backends/vole/patches/apply.py /workspace/swanky --fixed-lpn
CARGO_TARGET_DIR=/workspace/target-swanky-v2 cargo build --release -p diet-mac-and-cheese --bin dietmc
uv run research run --on <cpu pod> --project verity --source . -- \
    bash backends/vole/run_all.sh small                # or: vu B | threads B T | field F B | variants [B] | profile [B]
~~~
