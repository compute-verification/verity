# compile_cost: what compiling a served window's advice costs

`run.py` compiles one class-G window's advice into its Programs (`verity_vllm.program.predict.steps.compile_steps`), as
`integrations/vllm/tools/advice_compile.py` runs it, and records wall seconds and peak memory by phase. Its docstring has
the window, the arguments, the numbers it reports and their targets; `--guard` fails unless each Program digest is the
pinned one.

~~~text
research run --on auto --cwd source --mem-gb 64 --timeout 60m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu &&
  .venv/bin/python benchmarks/compile_cost/run.py --window long-w0 --out "$OUT" [--guard]'
~~~

## Interface

- **Provides:** `run.py`, a command, and the two result files it writes under `--out`.
- **Loaded by:** nothing.
- **Loads:** `verity_vllm.program.predict` (`phi`, `steps`), `verity_vllm.engine`, `verity.ml.circuits` and
  `verity.core.protocols.certifier`.
