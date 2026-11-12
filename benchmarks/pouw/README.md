# PoUW benchmarks

The measuring side of `verity/protocols/pouw/verifier`: Pearl-C's kernels on the RTX PRO 6000 Blackwell (sm_120), their
harness, the served vLLM path and its audits, and the CPU accounting. How a kernel is designed, gated, timed and landed is
the kernel-engineering skill (`.agents/skills/kernel-engineering/SKILL.md`).

Compute accounting's standards are in `benchmarks/machine_model/README.md`: the design registry (one file a design under
decision 79's names, Dither-1S, Dither-2S and Dither-C, which `verity_machine_model.skip` prices, `eps_budget.py`
prints and `machine-model eps` reads), the ε vocabulary, how a facet is measured (gates, noise, margins, clocks, row
status), and the per-device data layout.

`uv run python benchmarks/pouw/contents.py` prints what is here as a table: each directory or file, the hardware it runs
on, and what it is, one entry a file in `contents/paths/` (`order` places it). A new benchmark adds its own entry and edits
no other file; this README holds none of the tables, and `tests/test_contents.py` fails on one here and on an entry that
names a path the tree doesn't have.

~~~bash
uv run python benchmarks/pouw/contents.py   # what is here, from contents/
~~~

`verity bench pouw NAME [ARGS]` runs `NAME/`'s runner with ARGS. `run.py` here is that front door: a directory's runner
is its `run.sh`, else its `run.py`, and given no NAME it names the directories that have one. A runner that needs a GPU
or a whole node is the command of a `research run`, as its header says.

**What a caller may import.** The Python files directly here are the front door (`run.py`), the table (`contents.py`),
a run's result (`run_result.py`), the settings (`settings.py`) and the CPU calculators. Each states an `__all__`, the
names its readers take, but four. `pearlc_census.py` states none: its readers are `pearlc_k256_gate.py`,
`pearlc_k256_gate.sh` and their tests. `settings.py` states none either: code reads a setting it declares through
`verity_config.settings.of(__file__)`, and only its test imports it. `lean_headline.py` and `lean_headline_dc.py` state
none of their own: each gives the names of the machine model's modules it stands for (their `__all__`) to the files
that load it by this path. No code outside `benchmarks/pouw/` imports any of these files; inside it their readers are
each other, three scripts of `pearl_c_vllm/` (`run_result`), `theory/cpw_floor.py` (`lean_headline`) and the tests.

**What it never does.** The files directly here hold no kernel and build none. A calculator runs on a CPU and writes
only the files its arguments name, but for three: `budget.py --sweep` and `--sampling-sweep` rewrite its two data files
here, and `lean_headline.py` and `lean_headline_dc.py` write the Lean files they generate. `run_result.py` imports the
settings registry and the standard library, and writes a run's `result.json` only under a `research run`. `run.py`
starts the runner it is asked for and nothing else.

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

`edited_gemm/`'s references import `floored_gemm/derive.py`'s `tile_receipts` (the duo receipts of one 128 x 128 tile)
beside `floored_gemm/ref.py`;
`edited_gemm/receipt_folds_r9.py` is the r9 folds' reference as recorded, held by its sha256, so leave it
unedited.
