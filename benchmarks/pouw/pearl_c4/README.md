# pearl_c4: Pearl-C4's kernels on sm_120, checked byte for byte

Pearl-C4 is Pearl-C on FP4 operands that carry one scale a scale group (`pearl-c-nvfp4-v0` and `pearl-c-mxfp4-v0`). This
directory holds its device kernels for the RTX PRO 6000 (`pearl_c4.cu`, sm_120a), the check that every buffer they write
is the scalar reference's (`verity.protocols.pouw.prover.schemes.pearl_c4`), and the reference replay of what a device
ran. Its timing is self-timing for development; a number of record is the GPU harness's (`../harness/`).

## The device check, `run.sh`

`run.sh` is the runner: the command of a `research run` on one leased RTX PRO 6000, under `gpu-lease 1`. It runs from
the shipped tree, which is built off the GPU machine, so nothing is compiled on the card:

~~~text
bash benchmarks/pouw/pearl_c4/build.sh [OUT]        # with CUDA 12.9's nvcc; OUT defaults to /tmp/pearl-c4-ship
bash benchmarks/pouw/pearl_c4/ship.sh [OUT] [REPO]  # OUT as a one-commit repository for `research run --source`
bash run.sh [check]                                 # in the shipped tree, on the card
bash run.sh time [MxNxK ...]
~~~

- `build.sh` compiles `pearl_c4.cu` to `pearl_c4.cubin` and refuses it unless `sass.py` passes: no warning, spill, stack
  frame or local memory in any kernel, no FP32 instruction with `.FTZ`, and each kernel's tensor-core and cast
  instructions as counted. It writes the fixtures (`fixture.py`, from the numpy twin `twin.py` and the scheme's hashing)
  for both formats at two shapes, and the tree's `MANIFEST.sha256`.
- `run.sh check` refuses a tree that differs from its manifest and a GPU that is not the one leased, then runs
  `dev.py check`: every kernel on both fixtures at every GEMM geometry, each buffer compared byte for byte, the rows'
  commitment roots and the seeds included.
- `run.sh time` runs the check first, then `dev.py time` for each format at the shapes given.

A run writes, under `$RESEARCH_RUN_DIR/out`: `identity.txt`, `env.txt`, `inputs.check`, `toolchain.txt`,
`smi_trace.csv`, `check.jsonl` and, when timing, `time.jsonl`. It exits 3 when the GPU's identity is refused, 5 when
the tree does not match its manifest and 4 when a buffer differs, in which case nothing is timed.

## The reference replay, on a CPU

These recompute rows and words in the clear, a development shortcut: the protocol's check is a zero-knowledge proof,
and a replay's ACCEPT is never its verdict.

| Script | What it does | What it writes |
| --- | --- | --- |
| `replay.py prove DUMP --run RUN --out FILE` | makes a transcript from what a timed call dumped: the commitments, the records and the sampled tiles' openings | the transcript |
| `replay.py verify FILE --out REPORT` | recomputes the transcript's samples and every opened tile, with the tile's inert work against its bound | the report, and one ACCEPT or REJECT line |
| `real.py CAPTURE OUT [--layers RE] [--tiles N]` | judges a model's captured activations and weights as the scheme judges a unit: admission per row, the screened blocks, and sampled tiles' inert work | a JSON file a task and `summary.json`; exit 99 while tasks remain |
| `real_table.py SUMMARY.json...` | `real.py`'s summaries as one table a linear | the table, and its rows with `--json` |

`f1_fast.py` is the vectorised evaluation of the scheme's F1′ item that `real.py --f1 fast` uses.

## Who reads it

The tests in `../tests/` import `twin.py`, `replay.py`, `real.py`, `real_table.py` and `f1_fast.py` by name and start
`sass.py` and `fixture.py` by their command lines, and `../settings.py` declares the variables `dev.py` reads. `dev.py`
runs beside the cubin on the card with the standard library and numpy alone, so it imports nothing of the tree. Nothing
here is a verifier: no script writes a verdict a guarantee reads, and only `run.sh` and the `dev.py` it starts touch a
GPU.
