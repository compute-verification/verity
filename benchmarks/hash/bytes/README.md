# Fewer SHA-512 compressions, same hash

`bytes_model.py` prices each way to commit fewer SHA-512 compressions without changing the hash, through the hash lane's
commit model (`benchmarks/hash/model/commit_model.py`). It imports the model and calls it; nothing in the model is copied. A
row is either a value that could leave the served Commit, or a change to the leaf format, or both. Its price is the points of
a Qwen3-8B B32 run's stock serve time it saves on RTX PRO 6000, H200 and B200.

~~~text
python3 benchmarks/hash/bytes/bytes_model.py              # the ranked table: points of serve saved per GPU
python3 benchmarks/hash/bytes/bytes_model.py --formats    # per format: compressions a leaf, serial chain, opening ANDs
python3 benchmarks/hash/bytes/bytes_model.py --commands   # each row as commit_model.py's own command line
python3 benchmarks/hash/bytes/bytes_model.py --json
uv run python benchmarks/hash/bytes/bytes_model.py --check  # counts reproduce the protocols' and the card's numbers
~~~

## Baseline

The baseline is the card's best SHA-512 cell: per-Call units with fused epilogues (PRF-62/64/68), 71,680 B a token a layer
and 31.33 GB a run, in 4 KiB `hm96-sha512/row-seg/v1` leaves under a binary `frame-v3-sha512` tree. Prefill uses sharaw's
rate, the salts are read ahead, and the decode commit runs on a second stream. That costs 2.93%, 11.9% and 17.4% of serve on
RTX PRO 6000, H200 and B200. `--check` reproduces these values.

## How a row is priced

- **Bytes.** A row removes bytes a token a layer (`saves`) and, optionally, the logits. The new bytes per step go to
  `commit_model.set_fused_bytes`, so the decode rate is read at the smaller window, as the model does.
- **Format.** A format changes the data-phase compressions that a committed byte costs, meaning what still runs on the step
  once the salts and every position-only block are computed ahead. Today that is 44 a 4 KiB leaf: a row of 39, a leaf of 3
  and a tree share of 2.
  - Prefill is throughput-bound, so its rate scales by today's compressions a byte over the new format's.
  - Decode is latency-bound. Its window time is the fixed part of the 5-40 MB fit, plus the serial chain's change at 4.8 µs a
    compression, plus the rest of today's time scaled like prefill (intpipe sections 5 and 6).
- **KV rows.** An attention unit reads K and V one head slice a key, so `kv_fmt` can keep the KV rows at 4 KiB while the
  other bytes take big leaves. A row's compressions per byte mix the two formats by share.
- **Command line.** Each row ends in `commit_model.fused_cost`, with `--fused-bytes` and `--hm96` overrides. `--commands`
  prints them, so the model run alone gives the same cell.

Not in the model:

- The extra cost of a unit's larger cone (int8 scales, a sampler over the logits row) shows up at proving, not at serve.
- Two-stage rows leave out the interiors committed after the draw for drawn replay blocks, which is a fraction of the bytes
  that the profile sets.

The rule each row touches and its owner are in the table.

## PRF-70

- `attention_prologue.py` counts the attention element unit's ANDs a key with and without q/k norm and RoPE recomputed in
  the unit, from `compile_bits` of the repo's circuits (`--check` recomputes them). It then runs the sampling optimizer's
  Qwen3-8B plan at proofs' budgets for each variant: `baseline`, `prologue`, `kv-credited` and `q-prologue`.
- `two_stage.py` gives two-stage sampling's serve bytes (the option Daniel retired in ASK-25, 18 Oct 19:00Z: everything
  is committed at serve time; kept as what it would have saved) and SHA-512 overhead per GPU, with K and V kept at the boundary
  (condition (c)). It also gives the replay cost of a drawn layer or block: recompute plus interior commit, per draw and
  per GPU-week.

~~~text
uv run python benchmarks/hash/bytes/attention_prologue.py [--check|--json]
uv run python benchmarks/hash/bytes/two_stage.py [--json]
~~~
