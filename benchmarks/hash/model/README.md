# Commit overhead model

`commit_model.py` (stdlib only) projects commit overhead as a share of serve time from four things: the bytes committed per
prefill and decode step, the stock serve time per step, the leaf size, and a hash's throughput. A throughput is either measured
GB/s or compressions per second divided by compressions per leaf. The hash lane's candidates are compared through it.

~~~text
python3 benchmarks/hash/model/commit_model.py --card               # the recommendation card's tables, one per GPU
python3 benchmarks/hash/model/commit_model.py --card --hm96 'H200/SHA-512/pre=520,300' --hm96 'H200/SHA-512/step=410,180'
python3 benchmarks/hash/model/commit_model.py                      # the card, SHA-512 today, and the GB/s a hash needs for 5%
python3 benchmarks/hash/model/commit_model.py --gbps 900 --leaf 4096 --hash-name X
python3 benchmarks/hash/model/commit_model.py --rate 7.13e9 --per-leaf 44 --leaf 4096
python3 benchmarks/hash/model/commit_model.py --check              # the seeds reproduce circuits' cells; exit 1 if not
~~~

## The card

`--card` prints one table per GPU (RTX PRO 6000, H200, B200) for Qwen3-8B B32 compiled. Each row is a granularity: today's
per-Call bytes at 256 B leaves, then per-Call, per-block, per-layer and residual-folded per-layer at 4 KiB. Each column is a
hash in hm96 format: SHA-512, keyed BLAKE3, SHA-512 r32 and r24. A cell is a whole run's hashing over its stock serve time,
first with the salts precomputed off the step and then, in brackets, with the salts on the step. A ✓ marks a value under
`--target`. Under each table, `smallest_change` names the first row where SHA-512 is under the target.

- The rates are `HM96`: GB/s at 4 KiB for each mode. Prefill reads the 1 GiB window's rate and decode the 142 MB one.
  `--hm96 GPU/HASH/MODE=PREFILL[,DECODE]` replaces one; a single number sets both phases.
- The 256 B row is served rows-sha512 as measured (`SERVED_256`, hashing GPU ms per step), which runs with the salts on the
  step. Its "pre" value removes the salt's 5 of 14 compressions. The other hashes are not measured at 256 B ("n.m.").
- "Pre" only counts if the salt phase runs outside busy GPU time: idle time, ahead of the step, or another device.

## The model

- A phase hashes `kept bytes / GB/s` per step. Its overhead is that time divided by the phase's stock serve time. The run's
  overhead is the hashing summed over every step, divided by the serve time summed the same way.
- Hashing is assumed not to overlap serving. On H200 every commit arm's wall time is its GPU-busy time, and SHA-512 shares the
  integer issue slots with the GEMMs (note:circuits-rows-sha512-floor, note:circuits-commit-defaults-overlap).
- A coarser proof unit keeps only a fraction of the bytes inside the layers. The logits are kept whole, because they are
  committed for decode tokens anyway. The fractions come from the bytes per token per layer that circuits measured:
  - `call`: today's compiled Commit, 84,736 B (Llama-3.2-1B) and 145,180 B (Qwen3-8B).
  - `block`: a layer's attention block and MLP block.
  - `layer`: K, V and the two hidden vectors that vLLM's fused add-norm reads.
  - `layer-folded`: K, V and their sum as one hidden vector. These are the "per layer" rows of
    note:circuits-commit-bytes-width-dma and `granularity.py`'s `layer-folded` unit.
- Left out: gathers, staging copies, device-to-host transfer, host work, and the tree's per-level latency. Served SHA-512
  hashing measured 1.20× the floor on Llama-3.2-1B (302 ms against 252 ms, run r20261017-234859-fb61) and 1.14× on Qwen3-8B
  (1090 ms against 952 ms, run r20261018-031955-d2b3). The lane's baseline note, note:hash/20261018T0623Z-draft-baseline,
  breaks the rest down.

## Seeds

Every seed sits in `WORKLOADS` and `RATES` with its run id.

- Committed bytes per step come from the layouts recorded by runs r20261018-012341-e3b1 (Llama-3.2-1B) and
  r20261018-031955-d2b3 (Qwen3-8B), at B32, prompt 512, mixed traffic.
- Serve times are each run's stock arm: model-kernel GPU milliseconds per step. Llama-3.2-1B runs on one H200 (vy-glm53-s1).
  Qwen3-8B is one workload per GPU: RTX PRO 6000 (r20261018-065034-aa8f), H200 (r20261018-031955-d2b3) and B200
  (r20261018-065600-a6f3). Each has 4 prefill steps of 2,560-token chunks and 64 decode steps. The decode bytes of the 63
  steps in the layouts are spread over 64, so the run still commits 62.927 GB.
- The hm96 rates are runs r20261018-074305-3fd5 (RTX PRO 6000), -074230-9ff4 (H200), -074256-26b4 (B200) and -074239-8faa
  (L40S, which has rates but no serve times).
- GLM-5.3 is the Commit of record, r20261016-213336-6c35, taken per rank and at the run level only. Its serve time is the
  control run's wall, because eager GLM is host-bound.
- SHA-512 has three rate seeds:
  - measured on H200 by `rows_floor` (run r20261017-234343-61c6);
  - H200's raw rate, 4.79 G compressions/s;
  - RTX PRO 6000's (sm_120) raw rate, 7.13 G compressions/s (run r20261016-110413-c5d1).
- The two raw rates are divided by today's compressions per leaf: 14 at 256 B and 44 at 4 KiB (`sha512_leaf_compressions`).
- The older tables, from `table_today` and `table_required`, cover only the H200 workloads. Their sm_120 column is set
  against H200's serve times, so it compares hash speed only. The card uses each GPU's own serve times.
- `--check` holds the seeds to circuits' table: ten cells of prefill and run overhead at 256 B and 4 KiB, for per-Call and
  per-layer units, within 5%.
