<decision>
- **What changed.** On the circuits lead's word, `stage.py`'s refusal for the retired `dsa-q` names the census's own `DsaIndexerQ_v1{H=32,D=128,ROPE=64}` first and `dsa-q-row` second.
- **The shape.** I added `dsa-q` to `RETIRED_EITHER` so it uses the existing "A or B" form, as `router` and `moe-w13` do. A new "pick" structure was set aside as unneeded.
- **The test.** `test_retired` already holds each replacement to `parse()` and to `zk_refusal` being None. One line was added that pins `dsa-q`'s exact order.
- **What I didn't do.** I didn't merge `main` (the lead's order: layout moves cross only through `restack.py`), and I ran no staging or Lean run (the verdicts are those of run `r20261112-183158-1477`).
</decision>
<risk>
- **The merge with `main`.** The branch now conflicts with `origin/main` `e86d9204135c` in `benchmarks/served_units/README.md`'s file list: `main`'s `3dc5aae4d90` adds `prove_rows.py` and rewrites the same `tests/` line. The restack must keep both sides.
- **Memory.** `DsaIndexerQ_v1` as the first choice costs the Lean verifier about 147 GB at one block. Lean's memory at a full statement is unknown, so a caller that picks the first choice at scale may not fit.
</risk>
<verified>
- Locally, `benchmarks/served_units/tests/test_stage_census.py` gave 17 passed (167 MB).
- `ratchet_gate.py origin/main HEAD --json` gave `failures` `[]` at `fe800897e74d`.
- `suites.py --changed 5f0c008aa040` at `fe800897e74d` passed 9 of 9 suites, none cached, 0 failed (run `r20261113-011713-d278`).
- `git merge-tree --write-tree origin/main HEAD` exits 1, with one conflict in `benchmarks/served_units/README.md`.
</verified>
