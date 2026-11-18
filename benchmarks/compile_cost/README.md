# compile_cost: what compiling an execution phase's advice costs

The vLLM adapter compiles a served execution phase's advice into its request Programs
(`adapters/vllm/tools/advice_compile.py`). This benchmark measures what that compiling costs, in wall time and peak
memory by phase, for GLM-5.3 at TP8 on H200 and one rank's request Programs. Its two numbers are the `long-w0`
requests' wall minutes and peak GiB on one core, beside their targets in `run.py`'s docstring.

## Try it

~~~sh
verity bench compile_cost --window one-300 --layers 4 --out DIR
~~~

This needs no GPU and takes seconds: `--layers 4` keeps only the model's first four layers, so its digests are
another model's and its numbers are a scaling measurement. It prints the time and peak memory of each phase, then one
line, `COMPILE-COST window=.. wall_min=.. peak_gib=.. digests=..`, and writes `DIR/compile_cost.json` and
`DIR/summary.json`.

The measurement of record is the whole model on `long-w0`, which needs about 64 GB, so it runs on a node.
`run.py`'s docstring has that command, every option, and what each file holds; `--guard` fails unless each Program
digest is the pinned one.
