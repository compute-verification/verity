# mempool: vLLM's page pool as a reference machine

vLLM's page pool as memory-checked RAM, and the pool vectors `MemCheck_v1` is tested on:

- `pool_ram.py`: the page pool as a machine over word-addressed memories, every read and write one access under offline
  memory checking. It is the module the other two import.
- `difftest.py`: that machine against vLLM's own pool, as the scheduler replay models it, over the encodings' traces
  and synthetic ones.
- `vectors.py`: the machine's log as `MemCheck_v1`'s pool vectors.

The last two are commands, and each one's docstring has its arguments.

## Interface

- **Provides:** the two commands above, and the fixture `vectors.py` writes for `tests/verity/circuits/test_memcheck.py`:
  `tests/verity/circuits/fixtures/memcheck_pool.json`.
- **Loaded by:** nothing. The test that reads the fixture loads no code from here.
- **Loads:** the vLLM adapter's advice code (`verity_vllm.pipeline.advice_code`) and the scheduler replay with its
  traces under `benchmarks/network_e2e/encodings`.
