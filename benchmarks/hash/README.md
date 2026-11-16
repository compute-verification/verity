# benchmarks/hash

This directory measures the GPU code PoUW's served linear runs to commit and fold what it computes. Each kernel is held
bit for bit to its CPU reference or Definition before anything is timed. Nothing here changes a committed format, and
nothing a verifier runs reads this code.

It holds one benchmark, [`dither/`](dither/README.md): PoUW's receipts served in compiled vLLM, timed against stock
vLLM int8. Its kernels are frozen.

Each script's header says how to run it, from the source tree's root, and on what hardware. A run writes its results
to `$RESEARCH_RUN_DIR/out` and reads the files it is given from `$RESEARCH_RUN_DIR/inputs`, and its GPU work runs under
a GPU lease. [`DESIGN.md`](DESIGN.md) lists the files that code outside this directory loads.
