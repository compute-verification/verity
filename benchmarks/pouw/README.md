# PoUW benchmarks

The measuring side of `verity/core/protocols/pouw`: Pearl-C's kernels on the RTX PRO 6000 Blackwell (sm_120), their
harness, the served vLLM path and its audits, and the CPU accounting. How a kernel is designed, gated, timed and landed is
the kernel-engineering skill (`.agents/skills/kernel-engineering/SKILL.md`).

Compute accounting's standards are in `machine_model/README.md`: the design registry (one file a design under decision
79's names, Dither-1S, Dither-2S and Dither-C, which `eps_budget.py` prices and `machine-model eps` reads), the ε
vocabulary, how a facet is measured (gates, noise, margins, clocks, row status), and the per-device data layout.

`uv run python benchmarks/pouw/contents.py` prints what is here as a table: each directory or file, the hardware it runs
on, and what it is, one entry a file in `contents/paths/` (`order` places it). A new benchmark adds its own entry and edits
no other file; this README holds none of the tables, and `tests/test_contents.py` fails on one here and on an entry that
names a path the tree doesn't have.

~~~bash
uv run python benchmarks/pouw/contents.py   # what is here, from contents/
uv run python benchmarks/pouw/gamma.py      # CPU
~~~

Served claims (`pearl_c_vllm/`) are measured against graphed stock FP8: stock vLLM `quantization="fp8"` compiled with its
CUDA graphs, `e2e.py`'s ``fp8-graphs`` mode (``--fp8-graphs``).

## Where the filler's tools live

Each has one copy, in `filler/`; change it there. A new arm is a class in an arm module, and a new tile or fold is a flag
of `pfill.cuh`, not a copy of the kernel. Its tests are one suite: `uv run tools/verity/check/suites.py benchmarks/pouw/filler`.

`contents.py filler` lists each tool with its files, one entry a file in `contents/filler/`.

## Where the GEMM kernels and the committer live

Each has one copy; a later kernel includes an earlier one rather than copying it, so change the file an entry names. Each
directory's `*.sh` builds and runs it through `research run`, and each `*ref.py` judges its dumps on the CPU.

`contents.py kernels` lists each kernel with its files and the files it includes, one entry a file in
`contents/kernels/`.

`edited_gemm/egemm_ref.py` imports the E2E slice's `derive.py` from `e2e_slice/` beside it (the slice's receipt and noise
derivation, as cursor/egemm-ea24 has it);
`edited_gemm/receipt_folds_r9.py` is the r9 folds' reference as recorded, held by its sha256, so leave it
unedited.
