# POUS benchmark

One harness and one dataset for every POUS implementation. The metric is the frozen effective weight bandwidth (EWB):
the plaintext weight bytes that one pass decodes from `C`, per second. Every weight is decoded on every pass, at the
scheme's certified parameters.

- `weight_sets.json` (`pous-weight-sets/v1`) is the dataset. It holds the weight sets as the serving engine holds them,
  with their checkpoint pins:
  - Qwen2.5-0.5B: every linear weight and the tied head, 15,075 blocks of 64 KiB;
  - Qwen2.5-7B: layer 0 and the head, 23,744 blocks.
- `bench.py` is the harness: one command per (scheme, implementation, weight set, hardware). It writes `bench-result.json`
  (`verity-pous/bench-result/v1`), and `result.json` under `research run`.
  - Every implementation passes the same gates:
    - the scheme's pinned conformance vectors;
    - the implementation's own gates: known answers, and decoded blocks equal to the reference's;
    - the timed audit, plus a negative control;
    - `certificate`: the scheme's certificate covers the audit's k and deadline.
  - Each run reports the storage profile the certificate gives the accepted audit, gaps included.
- `pod.sh` runs it on a pod as a research Attempt.

Implementations:

- `reference`: `verity_pous` on CPU processes.
- `vllm-gpu`: the GPU kernels of `verity_vllm.engine.pous` (#172, #188), from their own tree, through its frozen
  harness `verity-vllm pous-dense-bench`, with the flags of the band's run of record.

A new one is a class in `bench.py`'s `IMPLS`.

~~~bash
PYTHONPATH=packages/verity/src:protocols/pous python benchmarks/pous/bench.py --impl reference \
  --weight-set qwen2.5-0.5b/linears+head --hardware local --out /tmp/pous-bench --max-segments 1   # about 50 s

git -C <#188 checkout> archive -o impl.tar 4e49163193c3750dad580f3a28f5428fbb722a8a
research run --on <L40S pod> --project verity --source . --cwd source --send impl.tar --campaign pous \
  -- bash benchmarks/pous/pod.sh IMPL=vllm-gpu IMPL_COMMIT=4e49163193c3750dad580f3a28f5428fbb722a8a
~~~
