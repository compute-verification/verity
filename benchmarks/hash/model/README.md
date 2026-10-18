# Commit overhead model

`commit_model.py` (stdlib only) projects commit overhead as a share of serve time from four things: the bytes committed per
prefill and decode step, the stock serve time per step, the leaf size, and a hash's throughput. A throughput is either measured
GB/s or compressions per second divided by compressions per leaf. The hash lane's candidates are compared through it.

~~~text
python3 benchmarks/hash/model/commit_model.py                      # SHA-512 today, and the GB/s a hash needs for 5%
python3 benchmarks/hash/model/commit_model.py --gbps 900 --leaf 4096 --hash-name X
python3 benchmarks/hash/model/commit_model.py --rate 7.13e9 --per-leaf 44 --leaf 4096
python3 benchmarks/hash/model/commit_model.py --check              # the seeds reproduce circuits' cells; exit 1 if not
~~~

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
- Serve times are each run's `stock-prof` arm: nsys GPU milliseconds per step on one H200 (vy-glm53-s1).
- GLM-5.3 is the Commit of record, r20261016-213336-6c35, taken per rank and at the run level only. Its serve time is the
  control run's wall, because eager GLM is host-bound.
- SHA-512 has three rate seeds:
  - measured on H200 by `rows_floor` (run r20261017-234343-61c6);
  - H200's raw rate, 4.79 G compressions/s;
  - RTX PRO 6000's (sm_120) raw rate, 7.13 G compressions/s (run r20261016-110413-c5d1).
- The two raw rates are divided by today's compressions per leaf: 14 at 256 B and 44 at 4 KiB (`sha512_leaf_compressions`).
- The sm_120 column is set against H200's serve times. It compares hash speed only, because no serve times on sm_120 are
  recorded for these workloads.
- `--check` holds the seeds to circuits' table: ten cells of prefill and run overhead at 256 B and 4 KiB, for per-Call and
  per-layer units, within 5%.
