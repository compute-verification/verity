[2026-11-12T18:23Z] a8471ff0a372
VERIFYSHARDS (now SORT, bc-2e121eb7) to training, Thu 12 Nov 18:27Z. **Your 18:21Z stop is read: piece 3 is stopped. I built and ran nothing for it** [RAN: no `research run` launched; one local branch `cursor/training-verifyshards-dd43` at `origin/main` `30d6fcc92f`, no commit, not pushed]. Starting piece 10 now; the table of twenty verdicts by 1 PM PST.

**Three things I read before the stop, for whoever holds piece 3** (all READ, none run):

1. **proofs' own worker already wrote the Lean leg and plans to run it.** SERVEDCARD's `lean-shard.sh` is in the notes, `lanes/proofs/overhead-run/lean-shard.sh` (notes commit `7a15e67600`, 18:13Z): it takes a units file and the GPU run's `out/kept/NAME/`, restages each statement with `stage.py` at the run's commit, holds the circuit to the kept `stage.json`'s `circuit_sha512` and `pin.txt`, writes `program-word.json`, and runs `verity bench prove lean-verify --repeat 1` a kept session, one row a session in `out/lean.tsv`. It is being rehearsed on 4 statements in `r20261112-181214-fea5` (cpu-11, ETA about 20:30Z). Its 18:14Z entry in `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5` says "The shards run on CPU nodes from Fri about 06:15Z, head blocks first, on nodes with at least 256 GB". So benchmarks and SERVEDCARD are both set to run the same leg unless one says so.
2. **Tonight's run is 70 statements of 36 block types** (98.19% of the ranked proving time), not about 95 of about 52 [SERVEDCARD's 18:13Z note, `lanes/proofs/20261112T1813Z-report-servedcard-overhead-step0.md`].
3. **What `lean-shard.sh` lacks against proofs' done-condition:** its row has wall seconds and peak GiB but no CPU-seconds (the record `lean-verify.json` has them: `verifier.runs[0].cpu_s`), no node and no block type; and a rerun starts again from the first session (it rewrites `lean.tsv`'s header). `benchmarks/prove/lean_verify.sh` on `main` verifies one art's proof a run and neither restages nor shards.

The plan itself, step 3 of the overhead run's table, is in the same file, the entry "PRF-502 SERVEDCARD (bc-013624b7), Wed 11 Nov 05:12Z".

