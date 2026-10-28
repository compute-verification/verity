# The machine model

PoUW's price vectors for sm_120 (rs-machine-model §8): the LP that prices every warp instruction as a convex combination
of the machine's rows, its inputs as data, the attack families, and `machine-model`, which re-solves a named vector and
checks every attack against it. The reading it reports is a design's core ε (`machine-model eps`, "ε in the
tensor–scalar model" below). Each named vector (`machine-model list`) re-solves here to the record it was published with
(τ, τ at 600 W, U/F_low and margin), at the vertex the legacy scripts landed on (`tests/test_vectors.py`), and `solve`
says whether (π₃₂), PRICES-32's load half and π_c ≥ 16 hold at its prices (`data/floors.csv`), and lists every copy
route's price a word beside its measured cost.

This file holds no table of the data. `machine-model render` prints every one as Markdown, read from `data/` (or a
device's directory) alone: the vectors, the epochs and their census, the rows, the floors, the designs with their noise
legend, the honest kernels, the proof blocks, the MMA forms, the table route, the attack families, the devices and θ.
`tests/test_render.py` fails on a table here, so a new vector, design or census row touches only its own data file.

~~~bash
uv run machine-model                      # every command's usage, each from its module's docstring in verity_machine_model/commands/
uv run machine-model list                 # the named vectors
uv run machine-model render               # every table of data/ as Markdown
uv run machine-model render vectors designs   # some of them; --device sm_90 render: a device's
uv run machine-model solve VECTOR         # the vector's record (τ, τ at 600 W, U/F_low, margin), θ, the cheapest attacks, the check, the claims
uv run machine-model solve                # model.toml's `current`, the vector of record
uv run machine-model census VECTOR --kind ATOMS64
uv run machine-model attacks VECTOR -n 3  # the cheapest attacks of every design
uv run machine-model adopt ~/.research/runs/RUN/out/census.csv EPOCH   # a probe's census at the last epoch
uv run machine-model --data /tmp/candidate solve VF3l   # a copy of data/ with a candidate change
uv run machine-model designs              # the registry: names, aliases, noise per product, fold, unit, coins, arm
uv run machine-model eps                  # Dither-1S's ε at q* = 8: every term, the LP's mixes, the ε(q*) curve
uv run machine-model eps dither-2s --q 2.2 --json
uv run machine-model eps Bc --checking sampled   # any alias; the sampling layer on its own line
uv run machine-model eps --table         # every design
uv run machine-model eps --table --operands floor   # the cheater's operands at the noise's floor φ_del
uv run machine-model eps --table --credit floor     # ε a fraction of the floor program ♭, not honest's kernel
uv run machine-model eps --table --credit floor --q 2.16 --lookups   # with QSTAR's table route against L2
uv run machine-model eps --table dither-1s --fold-sweep 1,1.5,2,2.5,3
uv run machine-model eps --table --credit floor --lookups --index-bits 2   # 2-bit entries on the indexed operand
uv run machine-model eps --table --credit floor --uni-issue 0.92   # a uniform instruction takes an issue slot
uv run machine-model eps --device sm89 --table --credit floor --operands floor   # the L40S: Dither-1S/2S/C at L = 16, 32
uv run machine-model eps --device sm100 --table --credit floor --operands floor   # the B200 at L = 32 to 256
uv run machine-model headline             # each family's design of record at the data-sheet and measured readings
uv run machine-model --device sm_100 sheet   # the data sheet: its DS lines against the census, each peak at its clock
uv run machine-model --device sm_90 sheet --eps   # the headline's sheet reading on the sheet's lines, A3 at f_max
uv run machine-model --device sm_100 sheet --eps --uni 0.1 0.02   # ... with a uniform instruction at 0.1, 0.02 a slot
uv run machine-model capacity 0.745/duo:0.6+lane:0.3 0.255/served:0.15   # the verified share σ of a window's capacity
uv run machine-model counts e-l7qs 2048 28672 4096   # H's closed-form instruction counts and model bound at a shape
uv run machine-model counts e-l7qs --check           # every ncu observation of H against them
uv run machine-model --device sm_90 theta    # θ: each figure at its clock, d by route, Strassen's exchange rate, B_RF, r_H, γ, units, c_sha
uv run machine-model render theta            # θ's rates, capacities, clocks, exact paths, FP8 atoms, units and SHA-512, with sources
uv run machine-model cover --holes           # the coverage table's counts by mark and its holes; no flag: the table, --check: its problems
~~~

The data, each kind of thing a file or a line of its own:

- `data/model.toml`: the constants (R*, the honest kernel's u a MAC, at 600 W too), the LP's ρ, tie and x_min, the honest
  GEMM, the CUDA kinds, the adopted design, the epochs in order and the `current` vector.
- `data/rows.csv`: one line per row of the machine (a pipe, a port, a ceiling or a joint vertex): its status
  (`measured`, `ASSUMED`, `provisional`, `joint`, `datasheet`), the runs or notes it rests on, and what it is.
- `data/census/EPOCH.csv`: the census lines from EPOCH, a file for each epoch that brought any: one line per instruction
  class on a row, its measured rate (warp instructions/SM/clk, with `noise` its derating), or the SMSP-cycles it holds
  the row, or a derived u; the epochs it holds (`since`, its file's epoch; `until`) and its runs.
- `data/census/runs.toml` and `data/census/kinds.toml`: the run lists and kind lists many census lines share (a refit's
  every line cites every run its fit read), each cited in a line's `runs` or `kind` as `@NAME`.
- `data/vectors/NAME.toml`: a named vector: its epoch, the rows and credits it uses in LP order, its summary and note,
  and the numbers it was published with.
- `data/attacks/FAMILY.jsonl`: an attack family: one program a line, its design, and the epochs it holds.
- `data/floors.csv`: what the claims read: each root-writing kind at 32 MACs a word ((π₃₂)), each load into registers at
  16 (PRICES-32's load half) and each copy route at 16 (π_c ≥ 16, `copy16`: every kind that writes registers, a UR word
  at the 32 lanes it serves), with the words a warp instruction of it writes.
- `data/honest/NAME.toml`: an honest kernel's instruction profile a MMA (every class it issues, the fold inside it), the
  least operand delivery any program pays a MMA, and its measured SMSP-cycles a MMA, with its source.
- `data/counts/NAME.toml`: an honest kernel's executed warp instructions by class as a closed form in the shape (a
  stage's, a tile's, a raster group crossing's and a launch's, per variant), the ncu observations they are fitted to and the tolerance that holds
  them.
- `data/designs/NAME.toml`: one design of the registry ("Designs" below): its name, title and aliases, the noise on each
  product, its honest profile, L, the noise's operand-delivery floor φ_del, the fold and filler (lane instructions a
  word by class), plain (unfolded) MMAs beside its work, the proof block, the coins, its perplexity arm, and ε_skip's and
  ε_mand's parts with sources.
- `data/designs.toml`: what every design reads unless its file sets it: honest's profile, L, the proof block, the coins,
  the fold (duo), ε_clock, Theorem S's δ, p and checks, honest's baseline program (dp4a) and the routes, all sm_120's.
- `data/adapt.toml`: the proof blocks ε_adapt prices (SPEC's Theorem D): whole dot products (the designs' unit, decision
  72), whole with operands recomputed one stage back, per-stage coins, and the segment spot check it replaced.
- `data/forms.toml`: the MMA forms the cheater's tensor route mixes: each census kind's shape, split, sparsity, why its
  product is exact, and its SASS.
- `data/table.toml`: the cheater's table route (`eps --lookups`): lane instructions and L2 bytes a word, and the L2
  bandwidth it is bound by (the most L2 read in any measured pattern, + 2%).
- `data/headline.toml`: the per-family headline's spec: each GPU family's design of record, the vector each reading
  prices it on, the conditions no computation shows, and its scope, per node ("The headline is per node").
- `data/theta.toml` (and `data/devices/sm_90/theta.toml`, the H200's): θ, the ACCEL model's parameters:
  every tensor rate by format and shape, every lane pipe, the register file, shared memory, L1, L2 and HBM, the
  capacities and clocks, each with its vendor document and page first, then the census or run that measured it, or the
  run that would (`gap`); the exact paths (b1, INT4, FP4) with their ISA reference; the FP8 atoms' semantics; the
  units that write registers, which B_RF's LP reads; a verdict for every unit RT-UNITS lists (`[[unit]]`); the
  public SHA-512 figures our compressions are read against (`[[public]]`); our kernel's SASS a compression
  (`[sha512_mix]`), which SHA-512's ceiling on the card's pipes reads; and the host the card sits in on the node we
  run (`host.*`: cores a GPU, SHA-512 and AMX INT8 a core, DRAM a socket).
- `data/devices/sm_NN/`: another device's data ("Devices" below).
- `data/cover/`: the coverage table ("Coverage" below): the envelope (`envelope.toml`: MODEL-MIN's `Accel.Op` and the
  proposed entries), a row per opcode and a
  verdict per unit outside the SM pipes (`rows.toml`), and the SASS corpus's opcodes by form (`corpus.json`, a fetched
  fixture: "Coverage" below).

The code:

- `verity_machine_model/model.py`: reading `data/` (or a device's directory).
- `verity_machine_model/lp.py`: the LP (Charnes–Cooper; stage 1 U/F_low, stage 2 τ), as the legacy `rs-price-99.py`
  `solve` computes it.
- `verity_machine_model/attacks.py`: an attack's X, the cheapest of a design, and the check: every price at or below
  measured cost, ρ, every credit, every attack at X ≥ 1; a kind's price a word, what is under its floor, and a claim's
  kinds priced a word beside their measured cost.
- `verity_machine_model/adopt.py`: merging a measured census into `data/census/` at the last epoch, each (kind, row) at
  the faster of its two lines.
- `verity_machine_model/resolve.py`: re-solving a named vector from the data.
- `verity_machine_model/eps.py`: the ε budget of the tensor–scalar model
  and ε_machine's LP.
- `verity_machine_model/counts.py`: n(H)(m, n, k) from that closed form, the model bound it gives at any shape the
  kernel runs, its fit and its check.
- `verity_machine_model/capacity.py`: the verified share σ = 1 − hidden(n̄)/hidden(0) of a window's capacity,
  hidden(n̄) the useful work any program could still add beside the pooled verified load n̄ (the filler workstream,
  decision 77).
- `verity_machine_model/redteam.py`: the red-team audit of a headline design against a device's committed census.
- `verity_machine_model/headline.py`: the per-family headline, every number computed from the design files and vectors.
- `verity_machine_model/theta.py`: reading `theta.toml` and holding it to its rules, and what it derives on the data
  sheet and measured: d, one delivery in INT8 MAC units by route; Strassen's ALU-for-tensor-core exchange rate at the
  honest kernel's L; B_RF, the most register bits all units write a SM-clock, and c ≈ ρ₈/B_RF; and, for each honest
  profile `honest` lists (duo's and LI's), r_H against the fold's r_min and, measured, γ against the lone INT8 MMA;
  the mma.sync D ≠ C seed's d; Strassen's exchange at the no-fold receipt's L too; each tensor-over-lane ratio per
  second (`tick`); each RT-UNITS unit's verdict and the most a bounded one adds; and c_sha, our SHA-512 against each
  public figure, with the card's ceiling for our kernel's SASS, with and without IMAD.HI (`sha_ceiling`).
- `verity_machine_model/cover.py`: reading the coverage table, the opcodes it must cover and what fails it.
- `verity_machine_model/cli.py` and `verity_machine_model/commands/`: the `machine-model` command. `cli.py` only
  dispatches; each command is a module of `commands/` with its usage as its docstring, `add(sub)` its arguments and
  `run(args)` its body, so a new command is a new module and touches no other.

An *epoch* is a state of the census and the attacks, named after the vector that introduced it. A census line or an attack
holds from its `since` epoch (empty: the first) through its `until` epoch (empty: the last), so every vector re-solves on
the data it was published on while new measurements land beside it. The census lines from an epoch are its own file, so
the census grows by a file an epoch; the tests keep every file under a quarter of the repository's 256 KiB cap. A kind with no coefficient on a vector's rows is
refused, and the check fails on it.

## Measure a facet

A facet is a row of Π: one class's rate, or a joint frontier of two or more classes. It is measured by a
microbenchmark that knows nothing about any algorithm (`../opcode_probe/`, on `../gpu_kit/`). Its numbers enter
`data/` only from a run that passes every gate below. A refused probe stays in the run as a visible failure.

- **The SASS histogram is pinned.** `gpu_kit/sass_gate.py` runs on the binary in the same job, before the timed run.
  It requires:
  - two clock reads around the timed region;
  - the declared loops, with each loop's exact opcode counts (`--expect`);
  - no LDL or STL;
  - a sink that every timed instruction reaches;
  - with `--smem`, the bank check.

  A loop whose counts differ measured something else, and none of its timing is read. The run records the toolkit
  (ptxas's version), because placement moves between releases.
- **Poison and the no-write control.** `gk::poison_gate` writes 0xA5 over the output on the kernel's own stream, runs
  the kernel and judges the output against its twin. A control that writes nothing must then fail the same judge.
- **Clocks.** `gk::timed_items` sizes every arm's item to 40 ms and warms up until the SM clock is steady. It rotates
  the arms' order rep by rep and reads NVML's SM clock and throttle reasons around every item. An item tagged
  `[clock-gap]`, `[throttled]`, `[unsteady]` or `[no-clock]` doesn't count. Rates are per SMSP-cycle, so the clock
  enters ε only as ε_clock. sm_120's runs are at the node's 2,092 MHz lock, where every program runs at 2,099.5 to
  2,100.0 MHz. f_max is the hardware's 2,430 MHz.
- **The 2% noise rule.** A rate enters the census derated by its `noise`, 0.02. `opcode_probe/eps1.py facets` scales a
  joint frontier's coefficients by 1 − margin (default 0.02), so a program measured on the hull still meets its facet
  with 2% to spare. `--prior data/rows.csv` keeps the old rows' vertices in the hull, and `--emit EPOCH` prints census
  lines.
- **The margin rule (decision 76).** A limit found by a schedule search takes, as its margin, the spread of the
  search's latest extension, once two successive extensions agree within the 2% noise. It replaces blanket factors
  such as the 1.07 that divided VF3n-16's JI3n rows.
- **Row status.** Every line of `data/rows.csv` states its status:
  - `measured`: a gated run of its own;
  - `joint`: a facet of a measured frontier;
  - `provisional`: in use before its gate, or closed;
  - `ASSUMED`: no run measures it;
  - `datasheet`: a vendor document states it (a data sheet's peak, the Programming Guide's throughput table, Nsight
    Compute's pipe assignment), at its exact value: a device's `DS_*` rows and its vector `DS-NN` ("Devices").

  A row a re-measurement violates closes, whatever its status ("Adopt a probe's census").

A uniform-datapath row (UP, or UNI's coefficient on issue or a joint issue row) has its own standard, from DATASHEET
P10. ptxas decides which datapath a warp-uniform chain runs on, and moves
it to lanes unless the loop holds it there, so the run must:

1. Prove the placement from SASS, never from source. The gate pins each timed loop's exact uniform-opcode and MMA
   counts.
2. Pin the toolkit the node pins (ptxas 13.0) and record its version.
3. Put the chain in a loop with MMAs whose other vector work reads their results. That is the only place ptxas 13.0
   keeps it.
4. Carry the chain round the loop into the sink, with its operands from a kernel parameter and LDCU inside the loop.
5. Hold the vector stream fixed while the uniform count K takes at least four values. The cost is the least-squares
   slope of SMSP-cycles against the uniform count read from SASS.
6. Report the vector issue rate at K = 0. A "takes the slot" reading holds only over the range measured (0.60–0.68 in
   P10).
7. Fit UNI's share of a vector instruction's coefficient on the joint rows at load ≤ 0.98, and refit it whenever the
   joint rows are refit. UP is the fastest uniform form alone, with no margin.
8. Record the SASS, every gate's verdict, rows.jsonl and the fit (`eps1.py uniform OUT`) in the run.

## Add a measurement

A new measurement is lines of data and a vector file, not a script:

1. **The row**, if the measurement is a new row: a line in `data/rows.csv` with its status, its run ids
   (`rYYYYMMDD-HHMMSS-xxxx`, a `note:` id, or `@NAME` for a note `data/citations.toml` names) and what it measures.
2. **The census lines**: the new epoch's file, `data/census/EPOCH.csv`, with one line for each instruction class the row
   binds: exactly one of `rate` (the measured warp instructions/SM/clk; `noise` 0.02 for the usual 2% margin), `cycles`
   or `u`, `since` the new epoch, and its runs. A run list that many lines share (a refit's) goes in
   `data/census/runs.toml` once, and its lines cite it as `@NAME`. A refit of an existing line sets the old line's `until`
   to the last epoch it held, in the old line's file, and adds the new value to the new epoch's file.
3. **The epoch**: append the new vector's name to `epochs` in `data/model.toml`.
4. **The vector**: copy the vector it builds on to `data/vectors/NAME.toml`, set `name` and `epoch`, add the row to `rows`
   (and any credit to `credits`), and drop `[published]` and `[legacy]`.
5. `uv run machine-model solve NAME`: it prints τ, τ at 600 W, U/F_low and the margin, and checks every price, credit and
   attack. Record the printed numbers under `[published]` once you publish them; the tests then hold the vector to them.

A new attack family is a new `data/attacks/FAMILY.jsonl` (`since` the epoch it joins); the check prices every attack of
every vector's epoch. A new root-writing kind or register load also needs its line in `data/floors.csv`, which `solve`
reads for the claims, and a new kind that writes registers its `copy16` line (`tests/test_data.py`'s `MEMORY_ONLY`
lists the kinds that write none).

## Copy routes

Per-row keys close (RT) only if every copy route costs π_c ≥ 16 MACs a word:
MOV, any landed word, UR fan-out, every SHFL mode, and any kind whose output can equal an input word. The `copy16`
claim reads every kind that writes registers, a superset of those, so a route can't be missed. VF3k-16 and VF3l-16 fail
it on the UR fan-out alone (UNI 2 and UNI64 1 MACs a lane-word); the candidates `VF3l-16-copy16`,
`VF3l-16-copy16-fan32`, `VF3l-16-copy32-fan16` and `VF3l-16-copy32` credit the routes they name, each at or below its
measured cost. No claim needs them now: only XOR-in's and LI's (RT) did, and both folds left the ε designs (a fold
cheaper than duo moves the lane threshold to dp4a's 8.5, under "ε in the tensor–scalar model"), so they stay as records.
Theorem O's every-run floor reads only (π₃₂) and the load half: a UR word is one value, so the lanes it serves past the
first are copies, credited only by copy luck.

## Designs

A design is one file, `data/designs/NAME.toml`, under decision 79's name. The names tonight's notes, data and
perplexity arms used are its `aliases`, and every command takes either. It is read on top of its `base` (the design it
varies, if it names one) and of `data/designs.toml`, which holds what every design shares: honest's profile, L, the
proof block, the coins, the fold, ε_clock, Theorem S's budget, the baseline and the routes. Each top-level key a file
sets replaces the inherited one whole, and `name`, `title`, `summary` and `aliases` are never inherited.
`uv run machine-model designs` prints the registry (`--json` too), `uv run machine-model render designs` prints it as a
Markdown table with the noise leaves (`NOISE` in `eps.py`) and the peel sums kernels (`SUMS`) as legends, and `eps
--json` carries a design's entry under `registry`.

Beside its names, a design file has these keys:

- `[noise]`: a leaf of `NOISE` in `eps.py` for each of `linears`, `lm_head`, `qk` and `pv`.
- `sums`: the kernel that forms a peel's sums, `vsum` (the default) or `vfold` (`SUMS` in `eps.py`).
- `honest`, `L`, `work` and `plain`.
- `[fold]` and `[filler]`: lane instructions a word by class. A filler runs on lanes (decision 73). A filler states
  `p`, the probability a deficient block of it passes its check, and `luck`, the theorem that proves it: 0 for a
  receipt the check recomputes (`Accounting.Guarantees.FillerLuck.PinnedFillBlocksDeficit`), 0.854 for pfill's cells
  (`FillerBlocksDeficit`), or `p = 1` with no `luck`, no luck credit.
- `[operands]`: the noise's floor φ_del.
- `unit`: a table of `data/adapt.toml`. It is whole dot products (decision 72).
- `[coins]`: `bits`, the width of a noised entry's coin, one σ a request.
- `[table]`: `bits` on the operand a table route indexes, by default the coins' width and never below it.
- `ppl`: a spec in `benchmarks/ppl/specs/`, or `""` while the design's arm isn't on main.
- The ε terms' accounting: `[skip.served]` and `[mand.PART]`, each with its `eps` and `source`, and `[clock]`, `[luck]`,
  `[baseline]` and `[routes]`. `[skip.served]` is the number SKIP's calculator, `../eps_budget.py`, gives the design's
  `[noise]` and `sums` (`design_skip`: the most over Llama-3.1-8B and 70B at 2K, 8K and 32K, prefill and decode, every
  served output free), and the pouw suite's `test_eps_budget` fails, printing the number to write, when they differ.

A new design is a file with its own name, title and summary, and a `base` if it varies one. The tests refuse:

- an alias that names two designs;
- a noise leaf or a proof block that isn't defined;
- a `[skip]` that isn't `eps_budget.py`'s number for the design's noise, or a noise leaf it can't price;
- an arm that isn't a spec;
- a filler off lanes.

## ε in the tensor–scalar model

`eps` reports two layers that are never mixed. The protocol with 100% checking, every word and
operand verified, has the core ε (`--checking full`, the default):

    1 − ε_core = (1 − ε_skip − ε_mand − ε_machine)(1 − ε_sched)(1 − ε_clock)

The sampling compiler adds ε_sample = ε_luck + ε_adapt, on its own line and in its own columns (`--checking sampled`).
It is a function of the proof block, its budget and its selection rule, and 0 at 100% sampling. The two layers meet only
in `Eps.total()`, for the record, never to compare designs. The core ε under the floor credit (decision 74) is the
guarantee's real-time reading. τ, the price vectors' real-time factor, and Theorem O's every-run floor were the reading
before it (decision 23). `solve` still prints τ as a vector's record, but τ is no longer the headline. Every term is a
fraction of honest time:

- **ε_skip**: honest work Verify doesn't require: peeled strips, clean products, unchecked limbs, the vector work a
  checked strip leaves free. Set by the design's `[skip.served]`, SKIP's calculator's number for its noise
  (`../eps_budget.py`, `design_skip`).
- **ε_mand**: the fold or the filler done in fewer instructions than honest's. Set by `[mand.PART]`: 0 for duo, under
  FOLD's named assumption.
- **ε_machine = ε_cert + ε_route**: the cheater's best mix over Π at q*, from the LP below. ε_cert = 1 − π*·n(H)/B(H) is
  honest's own free compute at the certificate π*; ε_route is what the cheater's routes add. Set by the vector and the
  design.
- **ε_sched**: honest's measured time above its model time; 0 unmeasured. Set by `[measured]` in
  `data/honest/NAME.toml`.
- **ε_clock**: 1 − f_H/f_max: 0.1391, honest at the node's 2,092 MHz lock against sm_120's 2,430 MHz. Set by `[clock]`.
- **ε_luck**: Theorem S (sampling layer): an accepted cheater leaves d̄ = ln(1/δ)/((1 − p)·S) of the words wrong, still
  folded. Set by `[luck]`: δ = 2⁻⁴¹, p = ¾, S = 48,000.
- **ε_adapt**: sampling layer: what steering instances in unproved regions gains under one σ a request (SPEC's Theorem
  D). Set by the proof block (`data/adapt.toml`).
- **Fan**: a proof block's fan-out, the adaptive MACs per MAC that fails for certain. ε_luck + ε_adapt ≤ max(1/(1 − p),
  1 + Fan)·(ln(1/δ)/S)·N_w·w*/B(H), so adaptivity is free while Fan ≤ 3 at p = ¾. Whole dot products have Fan 14 at
  gate/up and 17.9 at 8B's lm_head. Set by the proof block's `source`.
- **credit**: what ε is a fraction of: `H`, honest's kernel; `model`, its model bound B(H)/f_max; `floor`, the floor
  program ♭'s bound at f_max (adopted, decision 74). Under `floor`, ε_sched, ε_clock and honest's overhead above ♭ go
  to γ. Set by `--credit`.
- **q***: the fewest lane instructions, past the fold, of a right fresh committed word whose production reads no
  data-dependent address (SPEC §1.3). Assumed, never measured. Set by `--q`, default 8.
- **q*_T, b_T**: a table word's lane instructions, and the bytes it reads from its memory level (QSTAR's lookups). Set
  by `data/table.toml`, `--lookup-q`, `--lookup-bytes`.
- **q*_U**: q* on the uniform datapath, in uniform instructions. Out of the core since a uniform instruction takes an
  issue slot (decision 73). Set by `--q-uniform`.
- **threshold**: the least q* at which ε_machine reaches its floor (the scalar routes closed). Printed by `eps`.

SPEC's SCALAR-ROUTE states q*, q*_T and b_T as one named assumption: a right fresh word costs at least 2.03 issue slots,
or at least 1.22 slots and 3.5 B of table loads (6.0 B for single-field tables). Duo puts the lane threshold near that
2.03 (2.022 on VF3n-16, operands free). A cheaper fold breaks this. With 1 or 1.5 lane instructions a word (LI,
XOR-in), the fold leaves lane time idle beside the tensor route. The credited program fills that time with dp4a words,
so the cheater's lane words save time until q* reaches dp4a's 8.5 (`test_a_cheaper_fold_raises_the_threshold`). That
is why XOR-in and LI are not ε candidates.

Π is the current vector's rows at its epoch, read as the price LP reads them (SMSP-cycles a warp instruction a row).
ε_machine is one LP, SPEC's §0.8. A program writes every committed word on one
of four routes, folds every word and does the filler, and its time is its largest row. The routes are:

- the tensor pipe: `L/32` MMAs for 128 words, with their operand delivery;
- CUDA lanes: q* lane instructions a word, in any mix of the CUDA classes;
- the design's baseline program: dp4a, 8 IDP4A and its loads a word;
- the uniform datapath: q* uniform instructions a word, the fold there too.

Honest is the tensor route at the profile's cost plus the baseline, whose words belong to the baseline and not to ε.
The cheater is every route at q*. Its operand delivery is free by default (SPEC's conservative choice).
`--operands floor` prices it at the noise's floor φ_del, the design's `[operands]`: 1/39 a MMA where B̃ stays resident
at the zero instance, 0.20 where both operands are fresh (SPEC §3.4). `--operands honest` prices it at honest's.
The LP's duals are the certificate π*. ε_luck is the same LP with Theorem S's budget spent on unwritten words (still
folded) and undone filler, each deficient block weighted by 1 − p (¾ a word, the filler's own `p` a filler block), so
the terms add up to its time exactly. At `p = 1` the cheater leaves the filler undone even with every block checked.

Under `--credit model` or `--credit floor`, honest's time above the credit leaves ε for γ, which is printed beside it.
`floor` is the floor program ♭ of SPEC §0.7a (Corollary D3): the tensor route at φ_del, the fold, the fillers and the
baseline, with no overhead. Decision 74 adopts it. γ is honest's time a useful MMA over the plain GEMM's model time.
It is measured when the design runs the profile's kernel. Otherwise it is the measured kernel's time plus the model's
difference between the design and that kernel (PREDICT's arms: within −0.7% to +8.4% of measured on lane work), and
the model's alone without a measured profile.

The other options:

- `--fill X`: X lane instructions a word of verified filler on lanes. It dilutes what the cheater saves on operands,
  at the cost of γ. `dither-2s-fill` is Dither-2S with 4.5.
- `--lookups`: a fifth route for the cheater, the table route (`data/table.toml`, QSTAR's register-indexed lookups).
  It costs q*_T lane instructions a word from the lanes' class pool plus b_T bytes of L2 reads, against an L2 facet at
  the card's bandwidth, with honest's own L2 traffic uncharged. `--lookup-q`, `--lookup-bytes` and `--l2` override the
  file. Its defaults are SCALAR-ROUTE as restated: 1.22 slots and 3.5 B of L2 a word. A design with b-bit noise
  entries on the operand the tables index (`[table] bits`, or `--index-bits B`) pays b times the route's slots and
  bytes (QSTAR's b-bit count), and at b = 2 the route closes.
- ε_machine's floor and the threshold: the floor is ε_machine with the lane and uniform routes closed (the table route
  stays as set), which no q* lowers. The threshold is the least q* at which ε_machine reaches it.
- `--uni-issue C`: charges each uniform instruction C times an ALU instruction's cost on the issue rows too (`issue`
  and the joint `JI…_0` row). The census left that out from VF3 to VF3o-16, and PREDICT's u16s_l7qs measured it at
  0.92. With it, the uniform route closes and a filler gains nothing there. From VF3p-16
  (decision 73) the census charges it itself, and `--uni-issue` changes nothing.
- `--checking sampled`: charges ε_adapt for the design's own proof block (`unit`, whole dot products: +0.35%).
  `--adapt UNIT` charges another (`data/adapt.toml`): `segment`, the spot check decision 72 replaced (≈ 21%), or
  `whole-recompute` and `per-stage` (0). The Python `eps()` charges none unless given one.

## The headline is per node

`machine-model headline`'s ε bounds work done by the node:
an accepted window's node, its GPUs, host CPU and peer GPUs together, performed at least (1 − ε) of the
work the window credits. The device that did it is left to the attribution layer. The floor argument places no work: every
credited word is a fresh function of the post-coin noise, so it was computed after the coin, but nothing ties a
committed tile to the GPU that computed it. The transcript is a Merkle root over tile digests, so a helper holding the
salt ships one 64-byte digest a tile. The model prices the work on one device's rows, which makes ε a share of the
credited time `T_cred` on that device, not of that device's own time.

A per-GPU reading of ε comes only through the attribution layer, under two conditions:

- **LOC's host deduction** (decision 37, LOC: the node runs its own verified compute): what the host CPU can compute
  of the work the window credits is deducted exactly from the GPUs' credit. On int8 that is AMX for a tile's products and AVX-512
  for the fold and credited fillers, each bit-exact.
- **The peer-GPU premise**: every billable GPU on the node is credited each window, so no uncredited peer computes a
  credited GPU's tiles over NVLink. Otherwise the node is the unit.

Counted per GPU without them, as the headline's ε plus the host's or a peer's share of the credited time, the host's
AMX and AVX-512 (and NVLink peers) take every device over 1% on some real host. RT-SCOPE measures the host's
rate on each host family and prints the counted ε at both readings.

## Adopt a probe's census

An opcode probe (the `opcode_probe` beside this directory) writes its gated rows as `census.csv` in this
schema, with no epochs. After steps 3 and 4 above, `uv run machine-model adopt CENSUS EPOCH` merges it at `EPOCH`, which
must be the last epoch. Each (kind, row) keeps the faster of its line at the epoch before and the measured one: a row's
u must be a lower bound on what every program pays on it, so a line slower than the hardware makes a price that isn't
sound. A faster line closes the old one at the epoch before (in the old line's file), adds its own to
`data/census/EPOCH.csv` and adds its runs to the row's in `rows.csv`; a slower one changes nothing. A probe case the census has no row for (VF3k-16's peer-CTA DSMEM atomic) is a new row and, if it is
slower than its opcode's other cases, its own kind (`ATOME32P`): a line for the old kind there would overstate it.

A joint row a re-measurement finds violated (a measured point past 1) is unsound whatever the faster rule says: it closes
at the epoch before, and the new frontier's rows replace it. `opcode_probe/eps1.py facets --prior data/rows.csv --emit
EPOCH` prints them, the upper hull of the new points and the old rows' vertices with the 2% margin. A vector's own file
says what it closed and what replaced it (its `summary` and `note`; `machine-model render vectors` lists them all), so a
new vector adds nothing here. The examples that follow run to VF3q-16. VF3m-16 closed R2-R4,
KIMAD, KHFMA2 and KSHFL that way for EIMAD, EHFMA2 and ESHFL, and replaced the ASSUMED SHFL by its measured 8 SMSP-cycles.
VF3n-16 closed the joint issue rows JI3e_0..2 for JI3n_0..2: IMMA m16n8k16 and
TF32 m16n8k4 beside IADD3 load JI3e_1 past 1. Each JI3n row is JI3e's refit over every native mma.sync form's programs and
VF3e's at the 2% margin, then divided by 1.07 for the 6.9% one program's time varied between builds; MMA takes every form's
least, and each form is its own kind on them (`IMMA16816`, `HMMA1684T`, …, with its `copy16` line). VF3o-16 replaced
the 1.07 with a measured margin (EPS1 TENSOR's M1c, the tensor_kinds probe's schedule search):
600 schedules of the forms that set JI3n (lane op, register layout, 2 to 6 warps a SMSP, placement) join the refit, and
each JI3o row is divided by the most a searched program loads the row fitted to M1 alone, over 0.98 (1.038, 1.026,
1.025), so a search as wide again finding as much again stays covered; the probe's refit with its search runs prints
the rows. VF3p-16 put a uniform-datapath instruction on the issue rows (decision 73; EPS1 P10, `opcode_probe/eps1.py
uniform` over `probes/uniform.cu`): in the honest step at 60-68% issue it costs 0.6-1.1
SMSP-cycles, so UNI is 1 on issue and 0.7416 of a vector instruction's coefficient on each JI3o row, the share that holds
every uniform.cu point at the 2% margin (0.8168 holds them exactly), and UP rises to the fastest form alone, 0.766 a
SMSP-cycle. A uniform row's measurement is valid only on ptxas 13.0 with the uniform chain gated by its exact uniform and
MMA counts in SASS, inside a loop whose vector work reads the MMAs' results (ptxas moves the chain to lanes otherwise).
VF3q-16 is FACET-BREAK's refit (`facet_break/loads.py family`, runs `@FB`):
3505 programs over every facet beat 18 rows, and each beaten family closes for the upper hull of its old rows' vertices
and the measured programs at the 2% margin. R1 and KFFMA0 become EFFMA0..6 (interleaved IMMA + FFMA beat R1 by 41%);
JI3o_0 becomes JI3q_0..11 (HMMA.16816.F16 beside FFMA beat it by 9.8%), each form's hull taken alone and every row at a
vector coefficient one of those hulls has, each form at the most it carries beside it, while JI3o_1 and JI3o_2 hold; LDG
becomes ELDG0..2 over LDG128 and LDG32; KALU*, KSEL*, KALU64*, KIMADW*, KW20 and KDMUL0 their E families; R6r becomes
ER6r0..1 over LDS128 and UBLKCP4K (broadcast LDS.128 beside a 4 KB bulk copy out of shared memory beat it by 5.1%, at
LDS128's fastest program); TLD2 on ROW_TLDW2 scales to its fastest program. The six loads the census priced at 0 (LDS.64, LDG.E.64, the
256-bit LDG.E.ENL2.256 and the L1-skipping LDG.E.NA.128, .128.STRONG.GPU and .ENL2.256.STRONG.GPU) are kinds with rows
of their own at their fastest program, the L1-skipping three on one row of the L2 bytes they read (ROW_L2RD).

## Where the data came from

The legacy chain (the notes repo's `lanes/compute-accounting/gpu-model/rs-price-*.py`, kept as history) built each
vector in a script that loaded the previous one. `data/` is those scripts' LP inputs, captured at each named vector's
solve (that directory's `machine-model-export.py`) and normalized (`machine-model-convert.py`, which regenerates
`data/` byte for byte, all but `model.toml`): a census line's value is its rate, cycles or u, whichever form reproduces
the scripts' float exactly. Two differences from the
scripts: an instruction class's measured cost is the largest of its census lines at the epoch (the scripts froze it before
VF3f added `KFFMA0`, which is FFMA's F row rounded to 10 places, 4e-10 relative above it, inside the check's 1e-9); and
a line's runs are the runs its row's description cites, else the runs its vector's script cites.

## Devices

`data/` is sm_120's, the RTX PRO 6000 at the node's 2,092 MHz lock, and every command reads it unless `--device` or
`--data` names another directory. Another device's measurements go in `data/devices/sm_NN/`. That is the standard for
sm_80 (the A100-SXM4-80GB at default boost, EPS1 AMPERE, written by `tensor_kinds/sm80.py census`), sm_89 (the L40S,
EPS1 ADA), sm_90 (the H200) and sm_100 (the B200, EPS1 DC, whose census `dc/census.py` writes), and for any later device.
`machine-model render devices` lists each directory a command can price, with its epochs, rows, census lines, vectors,
designs and honest kernels. The directory follows these rules:

- It holds `rows.csv` and `census.csv` in `data/`'s schema, and its rows follow "Measure a facet". Its probe's
  `census.py` writes them from gated runs.
- An MMA kind is one SASS form, as on sm_120's JI3n rows.
- Its measured census is `census.csv`, which its writer rewrites whole. `dc/census.py --write` keeps only the uniform
  lines `dc/uniform.py` owns. A refit within it closes the old line with `until` and opens the new one with `since`,
  as sm_90's and sm_100's `-ri` epochs do. An epoch whose lines no writer produces (the data sheet's `DS-90` and
  `DS-100`) is a file of its own, `census/<EPOCH>.csv`, read after `census.csv`. `rows(DIR)`, `entries(m, DIR)` and
  `census(m, epoch, DIR)` take the directory.
- To price a design on the device, it gains the rest of `data/`'s files, and `--device sm_NN` (or `--data DIR`) reads
  it:
  - `model.toml`: its epoch, its vector and its CUDA lane classes, and optionally `uniform_forms`, the lane classes
    with a uniform-datapath form: a filler whose mix has another class has no uniform route (sm_90's datapath is
    integer only, so an FFMA filler stays on lanes). Without it every filler may move there. The constants only
    `solve` reads stay out.
  - `vectors/NAME.toml`: its rows.
  - `honest/`: its honest kernel at each L. `[mma] class` names the MMA's census class (sm_120's is `MMA`).
  - `designs.toml` and `designs/`: each design at each L, read on the device's `designs.toml` (the proof block and the
    coins), with the device's honest kernel, operand floor φ_del, clock and ε_clock, luck budget and baseline. The
    noise, ε_skip and ε_mand are sm_120's design's, and its title is that design's at L.
  - `table.toml`: the table route at the device's SMs and f_max. Without an `l2`, the L2 isn't measured, and the route
    is bound by its slots alone.

  `data/adapt.toml` holds for every device. A census with no uniform-datapath line (`UNI`) closes the uniform route.

Each device also has a data-sheet vector, `DS-NN`: its `DS_*` rows, which a vendor
document states, at their exact values, and no measured or joint row. They are `DS_issue` (every kind 1 SMSP-cycle,
uniform included), `DS_tensor` (each MMA form's MACs over the data sheet's dense peak for its type), the pipe rows of
the Programming Guide's throughput table on Nsight Compute's pipes, and `DS_smem` (one request of 128 B a SM-clock for
each shared-memory instruction, whatever its width, since a broadcast load reads as little as one word). `eps --vector
DS-NN` prices a design on the axioms alone. The current vector is the measured one, and every row is some vector's.
sm_120's is `DS-120`, an epoch of its own after VF whose lines close where they open, so it moves no other vector.
sm_90's `DS-90` and sm_100's `DS-100` are the same, after DC-90 and DC-100, in `data/devices/sm_90/census/DS-90.csv` and
`data/devices/sm_100/census/DS-100.csv`. Each also reads one measured row, `DS_uniform`: no document states the
uniform datapath's rate, so the sheet's uniform pipe is the census's UP line. Their constants are each directory's
`sheet.toml`, with the document that
states each one and its grade: `machine-model --device sm_90 sheet` derives every
DS line from it, checks it against the census file and checks each per-clock peak against the figure its document prints
at the clock it's quoted at. `sheet --lean` prints the Lean module `sheet.toml` names
(`Security/Definitions/Accounting/Sheets/`), which the tests hold to that output; its `Kind.omega` is each tensor class's A6 ω, its accumulator's words (an f64 element, a
DMMA's, two: each type's `element_words`). `sheet --eps` prices the headline design's sheet reading on the sheet's own lines (the
headline's ε, exactly), with A3 read per second at f_max instead (each peak its figure over SMs and f_max), and finds
how much slower every tensor line can be before that ε moves. `sheet --eps --uni C …` prices it with a uniform
instruction at each C of an issue slot, A2u's sweep, which `eps --uni-issue` doesn't do on a DS vector.

The L40S (`data/devices/sm_89/`) prices Dither-1S, Dither-2S and Dither-C at L = 16
(IMMA.16816, where T1 is tight) and at L = 32 (IMMA.16832, half the tensor pipe's words), on ADA's measured honest step:
duo's fold, 19.07 and 22.93 SMSP-cycles a MMA. Its clock is f_max = 2,520 MHz, with honest at 1,680 MHz at L = 16 and
1,585 MHz at L = 32 under the enforced 325 W cap. Its operand floors are SPEC §3.4's register count applied to each
form: 1/94 and 7/72 a MMA for m16n8k16, 1/39 and 0.20 for m16n8k32. ε_skip at L = 16 scales L = 32's strips by their
widths. `eps --table` prints γ_f, γ with the clock gap f_max/f_H.

The H200 and B200 (`data/devices/sm_90/` and `sm_100/`) price the same three designs
at L = 32, 64, 128 and 256 on model honest profiles: one IGMMA.64x256x32 (4096 words a warp) or UTCIMMA.128x256x32
(32768 words), then duo on every committed word. Neither has a measured honest kernel or f_H yet, so γ is the model's
and ε_clock is 0. Both read their operands from shared memory by descriptor, so the operand floor is no instruction for
every noise. The B200's accumulator is in TMEM, so its floor is the tcgen05.ld.32x32b.x32 that bring each committed word
to registers, 1024/L a MMA. Honest's profile counts that load in its fold, and the design's fold leaves it out, so a
word written on lanes doesn't pay it. The baseline is dp4a on IMAD's class, as on sm_120, with its loads unpriced (the
census has no IDP4A or LDS kind).

## Coverage

MODEL-MIN states the accelerator as a small envelope ISA and one premise: every instruction a card executes is
dominated by an envelope op at no fewer units of the same resource. `data/cover/` is that premise's evidence on the H200
(sm_90) and the RTX PRO 6000 (sm_120), MODEL-COVER: evidence, not trusted base (no verifier and no Lean reads it).
`machine-model cover` prints it as Markdown, a table a card, with the counts by mark and every hole.

- **The envelope** (`envelope.toml`) is MODEL-MIN's `Accel.Op` as drafted when the table was built, with its pipes,
  formats and each card's numbers (ρ a SM-clock, format weights): `mma`, `mmaSp`, `lane` (a lane opcode on a pipe: at
  most the pipe's `n_in` words in, 2 out), `move`, `atom`, `red`, `branch` and none. What the table needs that `Accel`
  lacks is marked `proposed` with the MODEL-MIN item asking for it: the `collective` op and resource, `tex`, and the
  formats `e3m2`, `e2m3`, `mxf8` and `mxf6`.
- **The pipes**: an opcode goes on the pipe a run or the CPG places it on, else on `fma`, which charges only the issue
  slot every instruction takes (coarse, and sound on any pipe).
- **What must be covered**: every opcode of NVIDIA's family tables (CUDA Binary Utilities 12.9 §4.3 and §4.4 in
  `../sass_census/data/bu_12_9.json`, CUDA 13's in `sass_isa_tables.json` beside it), every dotted form of the CUDA 13
  libraries' SASS corpus (`corpus.json`, with instruction counts), every census kind, and every unit θ lists
  (`theta.UNITS`). The three JSON tables are generated, so they are fetched fixtures (`fixtures/artifacts.json`,
  gitignored): `suites.py` fetches them before the tests, and `uv run research data fetch-fixtures
  benchmarks/pouw/machine_model/data/cover benchmarks/pouw/sass_census/data` brings them into a checkout. `rows.toml`,
  the reviewed table, stays in Git.
- **A row** (`rows.toml`) names its opcodes, devices, census kinds, mark, envelope op with its pipe, stores or formats,
  units a warp instruction, register words, rate (θ keys, or `rate_value` with its unit), argument and source. A row
  with `forms` (regexes over the dotted form) covers those forms; the opcode's plain row covers the rest. A unit outside
  the SM pipes is a `[[unit]]` with the same marks.
- **The marks**: `covered` (dominated at the rate), `coarse` (dominated at a coarser bound: a pipe's share, the issue
  slot, the data sheet's peak), `hole` (`low`, bounded at a few percent of a resource; `medium`, measured past a ρ;
  `open`, no rate known; or `outside the card`) and `absent`. An opcode's mark is its worst row's. A row resting on a
  proposed entry is counted twice: with it, and as a hole on `Accel.Op` alone.
- **The rule**: each rate of a covered or coarse row, in its op's units a SM-clock (tensor MACs at the format's weight,
  halved for `mmaSp`; a pipe's lanes, `fma`'s the issue slot's 128; fetches; a store's words), is at most its resource's
  ρ, and a lane's or collective's words at most its op's.

`tests/test_cover.py` fails when an opcode, corpus form, census kind or unit has no row, a form matches two rows, a row
names an op, pipe, format or θ key that doesn't exist, a dominated row's rate or words exceed its op's, or a hole has no
severity.

The evidence a row's source cites, each a `research run` with its verdict line:

- `../sass_census/lowering.sh` (any node, `--on auto`) compiles each case of `lowering.cu` for sm_90a and sm_120a and
  checks the SASS it lowers to or that ptxas refuses it (`verdict: N/N cases lower as expected`).
- On one leased RTX PRO 6000 (`--on vy-nebius-1`, through `../opcode_probe/run.sh`): `../opcode_probe/probes/match.cu`
  (MATCH.ANY by class count and .U64, beside LOP3 and FFMA) and `collective.cu` (REDUX and VOTE beside LOP3, REDUX beside
  MATCH.ANY), whose verdict is `../opcode_probe/collective_verdict.py`; both build for sm_90a too
  (`../dc/emp.sh build`, then `emp.sh time` on the H200). `pipes.cu` times each lane opcode beside its pipe's partner
  (LOP3, MUFU.EX2 or DFMA), and `pipes_verdict.py` holds each to the pipe's ρ.
- `../sass_census/multimem.sh` reads sm_120's multicast support and runs LDGMC and `multimem.red` on unicast memory.
