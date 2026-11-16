# verity_vllm.observe

The capture of an engine run. While the pinned vLLM runs eagerly, an observer records every dispatcher call and Triton
launch, and the only thing it writes is a raw observation log. The Match reads that log as evidence of what vLLM
executed.

This package holds the observer and the log's readers. [`fold/`](fold/README.md) turns a log back into a Program, so
that it can be held to the Program the Build derived.

The package's docstring lists its modules, and [`DESIGN.md`](DESIGN.md) says what the package loads, what loads it,
and which names other code imports from each module.
