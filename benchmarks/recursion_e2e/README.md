# The recursion bench (`verity bench recursion_e2e`)

The recursive session end to end on the CPU, timed stage by stage. `run.py` drives
`verity/ml/flock/pod/85-rec-reprice.sh`'s steps with `GPU=0`, one research run's `$OUT` for all of them, and gathers
their records into `$OUT/recursion_e2e.json`:

| stage | step | what is timed |
| --- | --- | --- |
| inner | `build`, `inner` | the inner statement staged (plain SHA-512 leaves), then proved without `--zk` (M0, `FC_COINS=os`) through the rec proxy (`verity_flock.rec_live`), whose coin server sees only salted tops and message commitments and answers `Register` with the full draw |
| V* staging | `vstage` | the verifier's fold points and comb (`InnerFold.lean`), then V*'s statements (each Ligerito level's `RecOpen`, its `RecCap`, the algebra's `InnerRepCheck` parts) staged and lowered (`verity_flock.rec_vstage`) |
| outer prove | `oprove` | every V* statement proved with `--zk` on loopback, every unit drawn, and replayed by upstream's verifier |
| verify | `lean` | each V* session verified by the Lean `flock-verify` built at the checked-out commit: `verify --zk --rec-inner-session --rec-inner-transcript`, against the verifier's own coins and the coin server's record |
| negative | `lean` | each honest V* session verified again over a copy of the inner session with rep 0's first `--neg` coin flipped (in `session.json` and the coin server's record alike, `rec_inner.py forge`), which Lean must refuse |

Each stage has its wall time, its peak resident memory (the largest process, `/usr/bin/time -v`) and its proof bytes.
`headline` holds the totals. The run exits 0 only when every step ran, every honest statement was accepted and every
negative was refused.

## Run it

From a clean checkout of the commit to measure. cpu-4 and cpu-5 cap a research run at 64 GB:

~~~sh
research run --on vy-nebius-cpu-5 --project verity --source . --cwd source --mem-gb 60 -- \
  python3 benchmarks/recursion_e2e/run.py run                    # K=64, N=16, one timed session per prove
python3 benchmarks/recursion_e2e/run.py store RUN                # from a VM: the result as an evidence/v1 art: id
~~~

`verity` isn't on the nodes' path, so the job runs `run.py` with `python3`; `verity bench recursion_e2e run` is the same
anywhere `verity` is installed. A node's shell has no store keys, so the job's own `research data put` usually fails
(`out/art.txt` says so), and `store RUN` from a VM records the result.

What `run` takes:

- `--k K --n N`: the inner statement, `N` instances of `GemmCoordinate_v2{K=K,DOT=HopperBF16WgmmaDot16_v1}` under
  `Q_template_instance` (`verity/ml/flock/pod/rec_reprice.py`). The defaults, 64 and 16, are small enough for a CPU node.
- `--runs R`: timed sessions per prove, with no warm one.
- `--neg COIN...`: `challenge` (the default) and/or `query`; `--neg` alone runs no negative.
- `--baseline`: today's `--zk` on the same statement with hm96 leaves, on loopback (`inner.baseline_zk`).
- `--pcpus P` (default 3/4 of the job's cores, the rest the verifier's), `--jobs J` (Lean verifications at once, default 2).

The inputs are pinned by the commit, `K` and `N`: the statement, its program, partition and draw are derived from them.
The prover and Lean builds are cached under `$FLOCK_WORK` by their sources, as `85-rec-reprice.sh` keys them. Unless
it is set, `$FLOCK_WORK` is the first of `/workspace/jobs/rec-reprice`, `/workspace/research/cache/rec-reprice` and
`~/.cache/verity/rec-reprice` that the job can write, recorded with `research keep`. A cold node builds C-Flock's
prover and `flock-verify` first. A run's staged statements
and sessions go under `$FLOCK_WORK/data/recbench-RUN_ID` and are removed when it ends, unless `--keep-data`.

`summarize DIR` rebuilds `DIR/recursion_e2e.json` from a fetched run's `out/`.
