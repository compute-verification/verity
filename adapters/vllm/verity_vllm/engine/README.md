# verity_vllm.engine

The vLLM engine the adapter serves with, pinned to one version. The rest of the adapter reaches vLLM through this
package: how the engine is built, the environment and switches it is served under, the facts of the models and cards
it serves, and the seams where the adapter attaches to a running engine, such as the engine core between steps and
the model runner's step.

It also holds the exact int8 linear's kernels, and the two commands that make a checkpoint: `verity-vllm
int8-checkpoint` and `verity-vllm fp8-checkpoint`.

`tests/test_vllm_public_api.py` lists which of its modules are public to code outside the adapter. The package's
docstring says what it holds, and [`DESIGN.md`](DESIGN.md) says what each module gives, what loads the
package and what it loads.
