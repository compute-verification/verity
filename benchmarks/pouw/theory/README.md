# PoUW theory checks

Exact checks of the premises PoUW's resource accounting rests on: each script decides one claim a premise or a Lean
docstring makes, exhaustively at a small width or over a modelled ISA. A script stays here only while a test or the
Lean reads it.

Most of them are about the fold receipts: the duo step and the fx1 and li folds that the served H1 and J10-R designs
absorb committed words with, and the TS premises their accounting reads (`Accounting.Assumptions.TS`: `FoldOpt`,
`NoSum`, `ScalarRoute`, `SideUnits`, `WordsWritten`). Dither, PoUW's no-fold design, has none of them. When the TS fold
premises are retired, their scripts are deleted with them.

## Running

- **A check** runs on `exact_kit`'s registry (`exact_kit/README.md`):

  ```text
  uv run python benchmarks/pouw/theory/NAME.py [--quick] [--mutations] [--verdict PATH]
  ```

  It prints `N/N checks OK, M/M mutations caught` and exits nonzero when a check fails or a mutation goes uncaught.
  `--mutations` runs each check's mutations as well, and `--verdict` writes the verdict JSON (under `research run`, to
  `$OUT/NAME.verdict.json`). `--quick` samples, in the four scripts that take it: `fold_kernel_check.py`,
  `fold_root_check.py`, `fold_wide32.py` and `zero_routes_check.py`.
- **A C enumerator** (`*.c`) is built by the script or test that drives it, with `gcc -O2 -fopenmp` (`-O3 -march=native`
  for `duo_kernel.c`). Its header comment gives its arguments and output, and `duo_proj.c` runs on its own.
- **An sm_120 probe** (`*.sh` beside its `*.cu`) runs from a pushed commit through `research run` on an RTX PRO 6000.
  Its header has the command; the `*_count.py` beside it counts the SASS and sets the device's timings beside the count.
- **The tests** are `tests/test_theory_*.py`, in the `verity-pouw-benchmarks` suite:
  `uv run tools/verity/check/suites.py` runs them when their inputs change.

## The scripts

What each one checks, and the Lean or the test that reads it.

### The fold's cost: `FoldOpt`

- `fold_opt.py`: no sm_120 program computes duo's per-word fold in fewer instructions than honest's 3 (LOP3, IMAD,
  LEA.HI), by a superoptimizer search and a lower bound exact over Z/2^32. `TS.lean` cites it for `FoldOpt`.
- `fold_opt_fx.py`: the same for sm_120's fx1 and fx2 chains, at least 4 and 5 issue slots a word, and no cheaper on the
  design's binding row. `--wide3` builds and runs `fold_wide3.c`.
- `fold_wide3.c`: `FoldOpt`'s two-word case at width 4: whether a program of at most 3 instructions writes a pair as its
  last instruction's two words, over `fold_opt_fx.py`'s modelled ISA and every sm_120 form that writes two words.
- `fold_wide32.py`: that two-word search at the full 32-bit word, for the forms that reducing mod 2^4 doesn't commute
  with (IMAD.HI, LEA.HI's rotate, IMAD.WIDE's high word, the 64-bit carries).
- `fold_routes.py`: whether any program writes the sm_120, H200 or B200 fold bit-exactly in fewer priced slots than that
  device's certificate charges, over the routes one and two instructions deep.
- `fold_min.py`: the cheapest bit-exact fold at each device's certificate (sm_120's fx1, the H200's li, the B200's li),
  and whether `FoldOpt`'s stated cost holds there. Cited by `HeadlineDc.lean`.

### Fresh words and the menu: `NoSum`, `ScalarRoute`, `SideUnits`

- `isa_escape.py`: no instruction or unit outside the TS menu heads a committed word more cheaply than the menu allows.
  Cited by `Sheet.lean`'s `SheetSemantics` and `FpG1.lean`.
- `nosum.py`: no fixed-function unit on sm_120 writes a fresh committed sum-of-products word (`NoSum`, claim
  `accounting-ts-nosum`).
- `scalar_route.py`: whether a program outside class P makes a fresh word in fewer than 2.03 issue slots without tables
  (`ScalarRoute`), over the four shapes class P's proof doesn't cover.
- `isa_cap_check.py` with `isa_cap.c`: two radix packings that create more incidences in one sm_120 instruction than a
  cap of 32/7, checked over every entry. They are why no per-instruction cap is a premise (`TS.lean`).
- `kappa_count.py`: the per-opcode creation cap κ = 65·d from each device's census, the Lean's `ClassPSlots`
  (`created_le_65`).
- `scope_offload.py`: how much of a credited GPU's int8 work a helper off the die (the host, a peer GPU, a NIC, device
  memory) can do, and which bound applies.

### The receipt's binding

- `duo_kernel.c`, driven by `duo_kernel_sweep.py`: duo's all-state kernel at small widths, the word runs that leave a
  committed value unchanged on every start state (`receipts/README.md`).
- `duo_proj.c`: word runs whose served receipts differ by one constant on every start state. Cited by `Receipt.lean`,
  for the three-word constant kernel beside `DuoWordProjNoKernel`.
- `fold_kernel.c`, driven by `fold_kernel_check.py`: the whole all-state kernel K_R of the same-word R-round XOR-in fold
  at width 8 or less, an independent check of `edited_gemm`'s depth-R binding check.
- `fold_root_kernel.c`, driven by `fold_root_check.py` and `fold_root_sweep.py`: K_R by a meet in the middle at width 16
  or less, and K_R = K_1 under the scaled key rule.
- `fold_open_shape_check.py`: the three keys the reduced key rule admits where K_R = K_1 fails.
- `chain_kernel.c`: runs of two across adjacent words absorbed different numbers of times, by a meet in the middle.

### The checked strip and attention's noise

- `checked_strip.py`: the exact check of the two-sided noise whose served output is a checked word. It is the one module
  other code imports (`edited_gemm`'s `cgemm_ref.py`, `fpx_design`, `tests/test_segment_h1.py`), and `edited_gemm`'s
  references build on its `pv_words`, `pv_served`, `co32c_words` and `digits`.
- `narrow_u.py`: that check with U on ±N for N = 8, 16, 32 (`edited_gemm`'s `co32c-u*`).
- `zero_routes_check.py`: `checked_strip.py`'s priced zero-instance routes.
- `attn_noise.py`: attention's committed words at every operand value, and its noise's h. Core's
  `verity/ml/protocols/pouw/circuit/compile.py` cites its `h_table` for the attention forms H1 is to carry.

### The sm_120 unit's checkpoint and issue

- `cpw_sass.sh` with `cpw_sass.cu`, counted by `cpw_sass_count.py`: the sm_120 unit with its per-set warp checkpoint
  (`PouwWarpCheck_v2` at fx1), each checkpoint route's SASS and its rate on the device.
- `cpw_routes.sh` with `cpw_routes.cu`: the checkpoint routes that merge on the memory system or the uniform datapath.
  `tests/test_census_cover.py`'s SASS fixtures hold its runs' timed loops.
- `cpw_floor.py`: the unit's floor from the sm_120 headline's Lean bound, beside what each checkpoint route adds.
- `a2u_sat.sh` with `a2u_sat.cu`, counted by `a2u_sat_count.py`: whether a uniform-datapath instruction takes an issue
  slot beside a saturating FFMA stream (`Sheet.lean`'s `SheetDispatch`, A2u).
