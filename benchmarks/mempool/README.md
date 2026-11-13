# mempool: vLLM's page pool as a reference machine

`CacheStep_v3`, the network certifier's cache step, checks a window's calls to vLLM's page pool as memory-checked RAM.
This directory holds a prototype of the reference that step is held to, and what the step costs:

- `pool_ram.py`: the page pool as a machine over word-addressed memories, every read and write one access under offline
  memory checking. It is the module the other four import.
- `difftest.py`: that machine against vLLM's own pool, as the scheduler replay models it, over the encodings' traces
  and synthetic ones.
- `stream.py`: `CacheStep_v3`'s specification (`cache_step_v3.Stream`) held to the machine after every pool call;
  `--vectors` writes the windows the cache step's test replays.
- `vectors.py`: the machine's log as `MemCheck_v1`'s pool vectors.
- `cost.py`: the cache step's cost in AND gates, in the busiest window of synthetic loads.

The last four are commands, and each one's docstring has its arguments.

## Interface

- **Provides:** the four commands above, and the two fixtures they write for tests elsewhere:
  `verity/protocols/certifier/verifier/tests/fixtures/cache_step_v3_vectors.json` (`stream.py --vectors`) and
  `verity/circuits/tests/fixtures/memcheck_pool.json` (`vectors.py`).
- **Loaded by:** nothing. The tests that read the two fixtures load no code from here.
- **Loads:** the certifier's cache step and compile slices (`verity.protocols.certifier.prover`),
  `verity.protocols.certifier.verifier.head_check`, `verity.circuits.defs`, `verity.commitments.gates`, and the
  scheduler replay with its traces under `benchmarks/network_e2e/encodings`.
