# The ACCEL model of each measured card: design

## Epochs, rows and the census

An *epoch* is a state of the census and the attacks, named after the vector that introduced it. A census line or an
attack holds from its `since` epoch (empty: the first) through its `until` epoch (empty: the last), so every vector
re-solves on the data it was published on while new measurements land beside it. An epoch's census lines are a file of
their own, so the census grows by a file an epoch, and the tests keep every file under a quarter of the repository's 256
KiB limit. A class's measured cost at an epoch is the largest of its census lines there. A kind with no coefficient on a
vector's rows is refused, and the check fails on it.

A *row* is a ceiling of the card: one class's rate, a port, or a joint frontier (a facet) of two or more classes. Π is a
vector's rows at its epoch, read as SMSP-cycles a warp instruction a row; a program takes at least its largest row.

## The check and the claims

`solve` checks every price at or below its class's measured cost, every CUDA lane class at least ρ of the MMA's per-lane
price, every credit of the vector, and every attack of the epoch at X ≥ x_min. It then reads the claims in `floors.csv`,
each a per-word price floor, and lists what is under each:

- `pi32`, (π₃₂): every root-writing kind at ≥ 32 MACs a word.
- `loads16`: every load into registers at ≥ 16 MACs a word.
- `copy16`, π_c ≥ 16: every copy route at ≥ 16 MACs a word. A copy route is MOV, any landed word, the UR fan-out, every
  SHFL mode, and any kind whose output can equal an input word; the claim reads every kind that writes registers, a
  superset, so no route is missed, and a UR word at the 32 lanes it serves. `solve` lists each route's price a word
  beside its measured cost.

A UR word is one value, so the lanes it serves past the first are copies. No ε design needs `copy16`: only the LI and
XOR-in folds' per-row keys did, and neither fold is an ε design ("ε in the tensor–scalar model").

## Measure a facet

A facet is measured by a microbenchmark that knows nothing about any algorithm (`../pouw/opcode_probe/`, on `../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/gpu_kit/`).
Its numbers enter `data/` only from a run that passes every gate below. A refused probe stays in the run as a visible
failure.

- **The SASS histogram is pinned.** `gpu_kit/sass_gate.py` runs on the binary in the same job, before the timed run. It
  requires two clock reads around the timed region, the declared loops with each loop's exact opcode counts
  (`--expect`), no LDL or STL, a sink that every timed instruction reaches, and with `--smem` the bank check. A loop
  whose counts differ measured something else, and none of its timing is read. The run records ptxas's version, because
  placement moves between releases.
- **Poison and the no-write control.** `gk::poison_gate` writes 0xA5 over the output on the kernel's own stream, runs
  the kernel and judges the output against its twin. A control that writes nothing must then fail the same judge.
- **Clocks.** `gk::timed_items` sizes every arm's item to 40 ms and warms up until the SM clock is steady. It rotates
  the arms' order rep by rep and reads NVML's SM clock and throttle reasons around every item. An item tagged
  `[clock-gap]`, `[throttled]`, `[unsteady]` or `[no-clock]` doesn't count. Rates are per SMSP-cycle, so the clock
  enters ε only as ε_clock. sm_120's runs are at the node's 2,092 MHz lock; f_max is the hardware's 2,430 MHz.
- **The 2% noise rule.** A rate enters the census derated by its `noise`, 0.02. `opcode_probe/eps1.py facets` scales a
  joint frontier's coefficients by 1 − margin (default 0.02), so a program measured on the hull still meets its facet
  with 2% to spare. `--prior data/rows.csv` keeps the old rows' vertices in the hull, and `--emit EPOCH` prints census
  lines.
- **The margin rule.** A limit found by a schedule search takes, as its margin, the spread of the search's latest
  extension, once two successive extensions agree within the 2% noise. No blanket factor stands in for it.
- **Row status.** Every line of `data/rows.csv` states its status:
  - `measured`: a gated run of its own;
  - `joint`: a facet of a measured frontier;
  - `provisional`: in use before its gate, or closed;
  - `ASSUMED`: no run measures it;
  - `datasheet`: a vendor document states it (a data sheet's peak, the Programming Guide's throughput table, Nsight
    Compute's pipe assignment), at its exact value: a device's `DS_*` rows and its vector `DS-NN` ("Devices").

  A row a re-measurement violates closes, whatever its status ("Adopt a probe's census").

A uniform-datapath row (UP, or UNI's coefficient on issue or a joint issue row) has its own standard. ptxas decides
which datapath a warp-uniform chain runs on, and moves it to lanes unless the loop holds it there, so the run must:

1. Prove the placement from SASS, never from source. The gate pins each timed loop's exact uniform-opcode and MMA
   counts.
2. Pin the toolkit the node pins (ptxas 13.0) and record its version.
3. Put the chain in a loop with MMAs whose other vector work reads their results, the only place ptxas 13.0 keeps it.
4. Carry the chain round the loop into the sink, with its operands from a kernel parameter and LDCU inside the loop.
5. Hold the vector stream fixed while the uniform count K takes at least four values. The cost is the least-squares
   slope of SMSP-cycles against the uniform count read from SASS.
6. Report the vector issue rate at K = 0. A "takes the slot" reading holds only over the range measured.
7. Fit UNI's share of a vector instruction's coefficient on the joint rows at load ≤ 0.98, and refit it whenever the
   joint rows are refit. UP is the fastest uniform form alone, with no margin.
8. Record the SASS, every gate's verdict, rows.jsonl and the fit (`eps1.py uniform OUT`) in the run.

## Add a measurement

A new measurement is lines of data and a vector file, not a script:

1. **The row**, if the measurement is a new row: a line in `data/rows.csv` with its status, the runs it rests on (run
   ids, or `@NAME` for a note `data/citations.toml` names) and what it measures.
2. **The census lines**: the new epoch's file, `data/census/EPOCH.csv`, with one line for each instruction class the row
   binds: exactly one of `rate` (the measured warp instructions/SM/clk; `noise` 0.02 for the usual 2% margin), `cycles`
   or `u`, `since` the new epoch, and its runs. A run list that many lines share goes in `data/census/runs.toml` once,
   and its lines cite it as `@NAME`. A refit of an existing line sets the old line's `until` to the last epoch it held,
   in the old line's file, and adds the new value to the new epoch's file.
3. **The epoch**: append the new vector's name to `epochs` in `data/model.toml`.
4. **The vector**: copy the vector it builds on to `data/vectors/NAME.toml`, set `name` and `epoch`, add the row to
   `rows` (and any credit to `credits`), and drop `[published]` and `[legacy]`.
5. `uv run machine-model solve NAME`: it prints τ, U/F_low and the margin, and checks every price, credit and attack.
   Record the printed numbers under `[published]` once you publish them; the tests then hold the vector to them.

A new attack family is a new `data/attacks/FAMILY.jsonl` (`since` the epoch it joins); the check prices every attack of
every vector's epoch. A new root-writing kind or register load also needs its line in `data/floors.csv`, and a new kind
that writes registers its `copy16` line (`tests/benchmarks/machine_model/test_data.py`'s `MEMORY_ONLY` lists the kinds that write none).

## Adopt a probe's census

An opcode probe (`../pouw/opcode_probe/`) writes its gated rows as `census.csv` in this schema, with no epochs. After steps 3
and 4 above, `uv run machine-model adopt CENSUS EPOCH` merges it at `EPOCH`, which must be the last epoch. Each (kind,
row) keeps the faster of its line at the epoch before and the measured one: a row's u must be a lower bound on what
every program pays on it, so a line slower than the hardware makes a price that isn't sound. A faster line closes the
old one at the epoch before (in the old line's file), adds its own to `data/census/EPOCH.csv` and adds its runs to the
row's in `rows.csv`; a slower one changes nothing. A probe case the census has no row for is a new row and, if it is
slower than its opcode's other cases, its own kind: a line for the old kind there would overstate it.

A joint row a re-measurement finds violated (a measured point past 1) is unsound whatever the faster rule says: it
closes at the epoch before, and the new frontier's rows replace it. `opcode_probe/eps1.py facets --prior data/rows.csv
--emit EPOCH` prints them: the upper hull of the new points and the old rows' vertices, with the 2% margin. Each MMA
form is its own kind on the joint rows (`IMMA16816`, `HMMA1684T`, …, each with its `copy16` line). A vector's own file
says what it closed and what replaced it (its `summary` and `note`); `machine-model render vectors` lists them all.

## Designs

A design is one file, `data/designs/NAME.toml`. Its `aliases` are its other names, and every command takes either. It is
read on top of its `base` (the design it varies, if it names one) and of `data/designs.toml`, which holds what every
design shares: honest's profile, L, the proof block, the coins, the fold, ε_clock, the sampling budget, the baseline and
the routes. Each top-level key a file sets replaces the inherited one whole, and `name`, `title`, `summary` and
`aliases` are never inherited. `machine-model designs` prints the registry (`--json` too), `machine-model render
designs` prints it with the noise leaves (`NOISE` in `eps.py`) and the peel sums kernels (`SUMS`) as legends, and `eps
--json` carries a design's entry under `registry`.

Beside its names, a design file has these keys:

- `[noise]`: a leaf of `NOISE` for each of `linears`, `lm_head`, `qk` and `pv`.
- `sums`: the kernel that forms a peel's sums, `vsum` (the default) or `vfold` (`SUMS`).
- `honest`, `L`, `work` (folded MMAs) and `plain` (unfolded MMAs beside them).
- `[fold]` and `[filler]`: lane instructions a word by class. A filler runs on lanes. It states `p`, the probability a
  deficient block of it passes its check, and `luck`, the theorem that proves it: 0 for a receipt the check recomputes
  (`ResourceAccounting.Guarantees.FillerLuck.PinnedFillBlocksDeficit`), 0.854 for pfill's cells (`FillerBlocksDeficit`), or `p =
  1` with no `luck`, no luck credit.
- `[operands]`: the noise's operand-delivery floor φ_del.
- `unit`: a proof block of `data/adapt.toml`, whole dot products by default.
- `[coins]`: `bits`, the width of a noised entry's coin, one σ a request.
- `[table]`: `bits` on the operand a table route indexes, by default the coins' width and never below it.
- `ppl`: a spec in `benchmarks/ppl/specs/`, or `""` while the design's arm isn't on main.
- The ε terms' accounting: `[skip.served]` and `[mand.PART]`, each with its `eps` and `source`, and `[clock]`, `[luck]`,
  `[baseline]` and `[routes]`. `[skip.served]` is the number `skip.design_skip` gives the design's `[noise]` and `sums`
  (the most over Llama-3.1-8B and 70B at 2K, 8K and 32K, prefill and decode, every served output free),
  and the pouw suite's `test_eps_budget` fails, printing the number to write, when they differ.

A new design is a file with its own name, title and summary, and a `base` if it varies one. The tests refuse an alias
that names two designs, a noise leaf or a proof block that isn't defined, a `[skip]` that isn't `design_skip`'s number
for the design's noise, an arm that isn't a spec, and a filler off lanes.

## Devices

`data/` is sm_120's, the RTX PRO 6000 at the node's 2,092 MHz lock, and every command reads it unless `--device` or
`--data` names another directory. Another device's measurements go in `data/devices/sm_NN/`: sm_80 (the A100-SXM4-80GB
at default boost), sm_89 (the L40S), sm_90 (the H200) and sm_100 (the B200). `machine-model render devices` lists each
directory a command can price, with its epochs, rows, census lines, vectors, designs and honest kernels. A device
directory follows these rules:

- It holds `rows.csv` and `census.csv` in `data/`'s schema, and its rows follow "Measure a facet". Its probe's
  `census.py` writes them from gated runs.
- An MMA kind is one SASS form.
- Its measured census is `census.csv`, which its writer rewrites whole (`dc/census.py --write` keeps only the uniform
  lines `dc/uniform.py` owns). A refit within it closes the old line with `until` and opens the new one with `since`. An
  epoch whose lines no writer produces (the data sheet's `DS-90` and `DS-100`) is a file of its own in the device's
  census directory, read after `census.csv`. `rows(DIR)`, `entries(m, DIR)` and `census(m, epoch, DIR)` take the
  directory.
- To price a design on the device, it gains the rest of `data/`'s files:
  - `model.toml`: its epoch, its vector and its CUDA lane classes, and optionally `uniform_forms`, the lane classes with
    a uniform-datapath form: a filler whose mix has another class has no uniform route. Without it every filler may move
    there. The constants only `solve` reads stay out.
  - `vectors/NAME.toml`: its rows.
  - `honest/`: its honest kernel at each L. `[mma] class` names the MMA's census class (sm_120's is `MMA`).
  - `designs.toml` and `designs/`: each design at each L, with the device's honest kernel, operand floor φ_del, clock
    and ε_clock, luck budget and baseline. The noise, ε_skip and ε_mand are sm_120's design's.
  - `table.toml`: the table route, at the card file's `sm.count` and `clock.sm`. Without an `l2`, the route is bound by
    its slots alone.

  `data/adapt.toml` holds for every device. A census with no uniform-datapath line (`UNI`) closes the uniform route.

Each device also has a data-sheet vector, `DS-NN`: its `DS_*` rows, which a vendor document states, at their exact
values, and no measured or joint row. They are `DS_issue` (every kind 1 SMSP-cycle, uniform included), `DS_tensor` (each
MMA form's MACs over the data sheet's dense peak for its type), the pipe rows of the Programming Guide's throughput
table on Nsight Compute's pipes, and `DS_smem` (one 128 B request a SM-clock for each shared-memory instruction,
whatever its width). Each also reads one measured row, `DS_uniform`: no document states the uniform datapath's rate, so
the sheet's uniform pipe is the census's UP line. `eps --vector DS-NN` prices a design on the vendor's numbers alone.
sm_120's is `DS-120`, an epoch of its own whose lines close where they open, so it moves no other vector; sm_90's and
sm_100's are in `data/devices/sm_90/census/DS-90.csv` and `data/devices/sm_100/census/DS-100.csv`.

A device's data-sheet constants are its `sheet.toml`, each with the document that states it and its grade, and its
value read from the card file (`card = "EXPR"`) where the card holds it:

- `machine-model --device sm_90 sheet` derives every DS line from it, checks them against the census file, and checks
  each per-clock peak against the figure its document prints at the clock it's quoted at.
- `sheet --lean` prints the Lean module `sheet.toml` names (`security/Definitions/Registration/Sheets/`), which the tests
  hold to that output and whose `machine_eq_census` proves the machine is the census.
- `sheet --eps` prices the headline design's sheet reading on the sheet's own lines, with each peak read per second at
  f_max, and finds how much slower every tensor line can be before that ε moves; `sheet --eps --uni C …` prices it with
  a uniform instruction at each C of an issue slot.

The H200 and B200 have model honest profiles (one IGMMA.64x256x32 or UTCIMMA.128x256x32, then duo on every committed
word) and no measured f_H, so γ is the model's and ε_clock is 0. Both read their operands from shared memory by
descriptor, so the operand floor is no instruction. The B200's accumulator is in TMEM, so its floor is the
tcgen05.ld.32x32b.x32 that brings each committed word to registers, 1024/L a MMA.
