# verity_vllm.check.replay

The sampled replay of a Commit. While vLLM served, the adapter committed the values its Programs name. The replay
checks a sample of them without the engine: it samples rows of each request's Program, reads their operands from the
committed values and the registered weights, evaluates each row by its family's row kernel, and compares the result
with the committed words bit for bit.

A Commit runs the replay in its own process, or later on a CPU from a sealed bundle (`verity-vllm commit-replay`). It
refuses a replay whose tree does not evaluate the Definitions the Build of record bound.

Each module's docstring says what it holds. [`DESIGN.md`](DESIGN.md) lists which names other code imports from each
module.
