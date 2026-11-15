# compile_cost: what compiling a served window's advice costs

`run.py` compiles one class-G window's advice into its Programs (`verity_vllm.program.predict.steps.compile_steps`), as
`adapters/vllm/tools/advice_compile.py` runs it, and records wall seconds and peak memory by phase. Its docstring has
the command that runs it on a node, the window, the arguments, the numbers it reports and their targets; `--guard` fails
unless each Program digest is the pinned one.

## Interface

- **Provides:** `run.py`, a command, and the two result files it writes under `--out`.
- **Loaded by:** nothing.
- **Loads:** `verity_vllm.program.predict`, `verity_vllm.engine`, `verity.circuits` and
  `verity.protocols.certifier.verifier`.
