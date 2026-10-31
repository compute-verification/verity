# The machine model

`machine-model` prices PoUW's designs on the ACCEL model 𝓜 of each measured card. For a card it holds, as data, the rows
(the measured ceilings: a pipe, a port, or a joint frontier of several classes), the census that places every warp
instruction class on them, the attack families and the designs. From them it solves the price vectors and reports a
design's ε. Capacity is the peak rate per operand class, read at the card's clock: a row is a ceiling no program beats,
and the model has no power or energy term. The directory and the command keep the name "machine model"; everywhere
else 𝓜 is the ACCEL model (`GLOSSARY.md`).

What it computes:

- **Price vectors** (`solve`): the LP that prices every warp instruction class as a convex combination of the rows. Each
  named vector re-solves to the τ, U/F_low and margin it was published with, and every attack of its epoch is checked
  against it.
- **ε** (`eps`): a design's core ε in the tensor–scalar model, every term from data ("ε in the tensor–scalar model").
- **The headline** (`headline`): each GPU family's design of record at the data-sheet and measured readings, the numbers
  the Lean headline guarantees are exported from ("The headline is per node").
- **The card table** (`theta`, `sheet`): each card's rates, capacities and clocks with their sources, and its data-sheet
  lines.
- **The opcode table** (`cover`): the evidence for the card premise ("The opcode table").

This file holds no table of the data. `machine-model render` prints every one as Markdown from a data directory alone:
the vectors, the epochs and their census, the rows, the floors, the designs, the honest kernels, the proof blocks, the
MMA forms, the table route, the headline's designs, the attack families, the devices and the card table.
`tests/test_render.py` fails on a table here, so a new vector, design or census row touches only its own data file.

## Commands

`machine-model` with no command prints every command's usage, each its module's docstring.

~~~bash
uv run machine-model list                 # the named vectors
uv run machine-model render               # every table of data/ as Markdown
uv run machine-model render vectors designs   # some of its sections
uv run machine-model solve [VECTOR]       # τ, U/F_low, margin, the cheapest attacks, the check, the claims
uv run machine-model census VECTOR --kind ATOMS64
uv run machine-model attacks VECTOR -n 3  # the cheapest attacks of every design
uv run machine-model adopt RUN_DIR/out/census.csv EPOCH   # a probe's census at the last epoch
uv run machine-model --data DIR solve VECTOR   # a copy of data/ with a candidate change
uv run machine-model designs              # the design registry
uv run machine-model eps [DESIGN]         # a design's ε at q* = 8: every term, the LP's mixes, the ε(q*) curve
uv run machine-model eps --table          # every design, one line each
uv run machine-model eps --table --credit floor --operands floor --lookups
uv run machine-model eps --device sm_89 --table --credit floor --operands floor
uv run machine-model headline             # each family's design of record at the data-sheet and measured readings
uv run machine-model redteam [DESIGN]     # the binding and near-binding rows, and how far each can move
uv run machine-model margin [DESIGN]      # the worst core with every near measured row moved 2% at once
uv run machine-model --device sm_90 sheet [--eps]   # the data sheet's lines against the census
uv run machine-model --device sm_90 theta # the card table and what it derives
uv run machine-model capacity 0.745/duo:0.6+lane:0.3 0.255/served:0.15   # the verified share σ of a node's capacity
uv run machine-model counts e-l7qs 2048 28672 4096   # an honest kernel's closed-form instruction counts at a shape
uv run machine-model counts e-l7qs --check           # every ncu observation against them
uv run machine-model cover [--holes|--check]   # the opcode table
~~~

`--device sm_NN` reads `data/devices/sm_NN/` ("Devices"); `--data DIR` reads any data directory. `solve` exits 1 when
the vector fails its check or doesn't reproduce a published number.

## Layout

### Data

`data/` is sm_120's, the RTX PRO 6000. Each kind of thing is a file or a line of its own:

- `data/model.toml`: the constants (R*, the honest kernel's u a MAC), the LP's ρ, tie and x_min, the honest GEMM, the
  CUDA lane classes, the adopted design, the epochs in order and the `current` vector.
- `data/rows.csv`: one line a row: its status ("Measure a facet"), the runs it rests on, and what it is.
- `data/census/EPOCH.csv`: the census lines an epoch brought: a class on a row at a measured rate (warp
  instructions/SM/clk, derated by `noise`), the SMSP-cycles it holds the row, or a derived u, with `since`, `until` and
  its runs. `data/census/runs.toml` and `data/census/kinds.toml` hold the run and kind lists many lines share, cited as
  `@NAME`.
- `data/vectors/NAME.toml`: a named vector: its epoch, its rows and credits in LP order, its summary and note, the
  numbers it was published with (`[published]`) and, for the oldest vectors, the full-precision values the tests hold
  them to (`[legacy]`).
- `data/attacks/FAMILY.jsonl`: an attack family: one program a line, its design, and the epochs it holds.
- `data/floors.csv`: what the claims read, a kind a line ("The check and the claims").
- `data/honest/NAME.toml`: an honest kernel's instruction profile a MMA, the least operand delivery any program pays a
  MMA, and its measured SMSP-cycles a MMA, with its source.
- `data/counts/NAME.toml`: an honest kernel's executed warp instructions by class as a closed form in the shape, the ncu
  observations it is fitted to, and the tolerance that holds them.
- `data/designs/NAME.toml` and `data/designs.toml`: the design registry and what every design inherits ("Designs").
- `data/adapt.toml`: the proof blocks ε_adapt prices.
- `data/forms.toml`: the MMA forms the cheater's tensor route mixes: each census kind's shape, split, sparsity, why its
  product is exact, and its SASS.
- `data/table.toml`: the cheater's table route: lane instructions and L2 bytes a word, and the L2 bandwidth it is bound
  by.
- `data/headline.toml`: each GPU family's design of record, the vector each reading prices it on, the conditions no
  computation shows, and its scope.
- `data/theta.toml`: the card table: every tensor rate by format and shape, every lane pipe, the register file, shared
  memory, L1, L2 and HBM, the capacities and clocks, each with its vendor document first, then the run that measured it
  or the run that would (`gap`); the exact paths (b1, INT4, FP4); the FP8 atoms' semantics; the units that write
  registers; a verdict for every unit outside the SM pipes (`[[unit]]`); the public SHA-512 figures (`[[public]]`) and
  our kernel's SASS a compression (`[sha512_mix]`); and the host the card sits in (`host.*`). The module and command
  keep the name `theta`.
- `data/citations.toml`: the notes a row or census line cites by `@NAME`.
- `data/cover/`: the opcode table ("The opcode table").
- `data/devices/sm_NN/`: another card's data ("Devices").

### Code

`verity_machine_model/`. Every reader takes the data directory, `data: Path = DATA` (sm_120's) unless given another.

- `model`: reading a data directory: `model() -> Model`, `rows()`, `census(m, epoch) -> Census`, `measured_cost(c)`,
  `vector(name) -> Vector`, `vectors()`, `attacks(m, epoch)`, `floors()`, and `device(name)` / `devices()` for the
  device directories.
- `lp`: the price LP (Charnes–Cooper; stage 1 U/F_low, stage 2 τ): `solve(...) -> Solution`, `price`, `cost`.
- `attacks`: `x_of` (an attack's X, its price over its tensor time), `cheapest`, `margin`, `check` (every price at or
  below measured cost, ρ, every credit, every attack at X ≥ x_min), `under_floor` and `priced` (the claims).
- `resolve`: `resolve(name) -> Resolved`, a named vector re-solved: `tau`, `uf_low`, `margin`, the cheapest attacks,
  `failures`, `floors` and `copies`; `numbers()`, and `unreproduced()` against its `[published]`.
- `adopt`: `adopt(path, epoch) -> list[Adoption]`, a probe's census merged at the last epoch.
- `eps`: designs and ε: `design(name) -> Design`, `designs()`, `aliases()`, `honest(profile) -> Honest`, `profiles()`,
  `machine(vector) -> Machine` (Π), `setup(design, machine) -> Setup`, `eps(design, machine, q) -> Eps`, `threshold`.
- `capacity`: the verified share σ = 1 − hidden(n̄)/hidden(0) of a node's capacity, hidden(n̄) the useful work any
  program could still add beside the pooled verified load n̄: `parse(spec) -> Work`, `hidden`, `pool`.
- `counts`: n(H)(m, n, k), an honest kernel's instruction counts from its closed form, the model bound it certifies at
  any shape, its fit and its check: `load(name) -> Counts`, `fit`, `names()`.
- `headline`: the per-family headline: `row(family, reading)`, `rows(reading)`, `record()`, `text()`.
- `redteam`: from a device's census alone, how much room an adversary has to beat a row: `audit`, `margin_audit`,
  `least_filler`, `uniform_gap`.
- `theta`: the card table: `theta(data) -> Theta`, `check`, and what it derives on the data sheet and measured
  (`routes`, `strassen`, `bound`, `folds`, `sha`, `sha_ceiling`).
- `sheet`: a card's data sheet (`sheet.toml`): `sheet(data) -> Sheet`, `lines`, `differences` against the census,
  `reading` and `moves` (the headline's sheet reading and how far a tensor line can slow before it moves), `lean`.
- `cover`: the opcode table: `load() -> Cover`, `problems`, `counts`, `holes`.
- `strassen`: naive tiled against Strassen in the tensor-core model: per-product costs (`costs`), uniform-depth tables
  (`depth`) and the least cost over every recursion tree (`sched`).
- `cli` and `commands/`: the command. `cli.main(argv)` only dispatches. Each command is a module of `commands/` whose
  docstring is its usage, `add(sub)` its arguments and `run(args)` its body, so a new command is a new module and
  touches no other.

## Epochs, rows and the census

An *epoch* is a state of the census and the attacks, named after the vector that introduced it. A census line or an
attack holds from its `since` epoch (empty: the first) through its `until` epoch (empty: the last), so every vector
re-solves on the data it was published on while new measurements land beside it. An epoch's census lines are a file of
their own, so the census grows by a file an epoch, and the tests keep every file under a quarter of the repository's 256
KiB cap. A class's measured cost at an epoch is the largest of its census lines there. A kind with no coefficient on a
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

A facet is measured by a microbenchmark that knows nothing about any algorithm (`../opcode_probe/`, on `../gpu_kit/`).
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
that writes registers its `copy16` line (`tests/test_data.py`'s `MEMORY_ONLY` lists the kinds that write none).

## Adopt a probe's census

An opcode probe (`../opcode_probe/`) writes its gated rows as `census.csv` in this schema, with no epochs. After steps 3
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
  (`Accounting.Guarantees.FillerLuck.PinnedFillBlocksDeficit`), 0.854 for pfill's cells (`FillerBlocksDeficit`), or `p =
  1` with no `luck`, no luck credit.
- `[operands]`: the noise's operand-delivery floor φ_del.
- `unit`: a proof block of `data/adapt.toml`, whole dot products by default.
- `[coins]`: `bits`, the width of a noised entry's coin, one σ a request.
- `[table]`: `bits` on the operand a table route indexes, by default the coins' width and never below it.
- `ppl`: a spec in `benchmarks/ppl/specs/`, or `""` while the design's arm isn't on main.
- The ε terms' accounting: `[skip.served]` and `[mand.PART]`, each with its `eps` and `source`, and `[clock]`, `[luck]`,
  `[baseline]` and `[routes]`. `[skip.served]` is the number `../eps_budget.py` gives the design's `[noise]` and `sums`
  (`design_skip`: the most over Llama-3.1-8B and 70B at 2K, 8K and 32K, prefill and decode, every served output free),
  and the pouw suite's `test_eps_budget` fails, printing the number to write, when they differ.

A new design is a file with its own name, title and summary, and a `base` if it varies one. The tests refuse an alias
that names two designs, a noise leaf or a proof block that isn't defined, a `[skip]` that isn't `eps_budget.py`'s number
for the design's noise, an arm that isn't a spec, and a filler off lanes.

## ε in the tensor–scalar model

`eps` reports two layers that are never mixed. The protocol with 100% checking, every word and operand verified, has the
core ε (`--checking full`, the default), Corollary D3's (`Accounting.Guarantees.TS.FloorCreditD3`):

    1 − ε_core = (1 − ε_skip − ε_mand − ε_machine − ε_luck)(1 − ε_sched)(1 − ε_clock)

The sampling compiler adds ε_sample = ε_adapt on its own line and in its own columns (`--checking sampled`): a function
of the proof block and its selection rule, 0 at 100% sampling. The two layers meet only in `Eps.total()`, for the
record, never to compare designs. `solve` prints τ, the price vectors' real-time factor, as a vector's record; the core
ε under the floor credit is the guarantee's reading. Every term is a fraction of honest time:

- **ε_skip**: honest work the check doesn't require: peeled strips, clean products, unchecked limbs, the vector work a
  checked strip leaves free. The design's `[skip.served]`.
- **ε_mand**: the fold or the filler done in fewer instructions than honest's. The design's `[mand.PART]`: 0 for duo.
- **ε_machine = ε_cert + ε_route**: the cheater's best mix over Π at q*, from the LP below. ε_cert = 1 − π*·n(H)/B(H) is
  honest's own free compute at the certificate π*; ε_route is what the cheater's routes add.
- **ε_luck**: the sampling bound inside the same LP. Every check is one draw, so an accepted cheater leaves d̄ =
  ln(1/δ)/((1 − p)·S) of the words wrong over S spot checks, still folded, each filler's blocks at their own p. The
  design's `[luck]`.
- **ε_sched**: honest's measured time above its model time; 0 unmeasured. `[measured]` in `data/honest/NAME.toml`.
- **ε_clock**: 1 − f_H/f_max, honest's clock against the card's maximum. The design's `[clock]`.
- **ε_adapt** (sampled layer): what steering instances in unproved regions gains under one σ a request, the proof
  block's (`data/adapt.toml`).
- **credit**: what ε is a fraction of: `H`, honest's kernel; `model`, its model bound B(H)/f_max; `floor`, the floor
  program ♭'s bound at f_max. Under `model` or `floor`, ε_sched, ε_clock and honest's overhead above the credit leave ε
  for γ, printed beside it. `--credit`.
- **γ**: honest's time a useful MMA over the plain GEMM's model time: measured when the design runs the profile's
  kernel, else the measured kernel's time plus the model's difference between the design and that kernel, else the
  model's alone.
- **q***: the fewest lane instructions, past the fold, of a right fresh committed word whose production reads no
  data-dependent address. Assumed, never measured (`ScalarRoute`, `Accounting.Assumptions.TS`). `--q`, default 8.
- **q*_T, b_T**: a table word's lane instructions, and the bytes it reads from its memory level. `data/table.toml`,
  `--lookup-q`, `--lookup-bytes`.
- **q*_U**: q* on the uniform datapath, in uniform instructions. `--q-uniform`, default q*.
- **threshold**: the least q* at which ε_machine reaches its floor (the scalar routes closed). Printed by `eps`.

ε_machine is one LP. A program writes every committed word, folds every word and does the filler, and its time is its
largest row. A word is written on one of four routes:

- the tensor pipe: `L/32` MMAs for 128 words, on any MMA form of `data/forms.toml`, with their operand delivery;
- CUDA lanes: q* lane instructions a word, in any mix of the CUDA classes;
- the design's baseline program: dp4a, 8 IDP4A and its loads a word;
- the uniform datapath: q* uniform instructions a word, the fold there too.

Honest is the tensor route at the profile's cost plus the baseline, whose words belong to the baseline and not to ε. The
cheater is every route at q*. Its operand delivery is free by default; `--operands floor` prices it at the design's
φ_del and `--operands honest` at honest's. The LP's duals are the certificate π*. At `p = 1` the cheater leaves the
filler undone even with every block checked. With 1 or 1.5 lane instructions a word, a fold leaves lane time idle beside
the tensor route; the credited program fills it with dp4a words, so the cheater's lane words save time until q* reaches
dp4a's 8.5 (`test_a_cheaper_fold_raises_the_threshold`). That is why the LI and XOR-in folds are not ε designs.

The other options:

- `--fill X`: X lane instructions a word of verified filler on lanes, which dilutes what the cheater saves on operands
  at the cost of γ.
- `--lookups`: a fifth route for the cheater, the table route (`data/table.toml`): q*_T lane instructions a word from
  the lanes' class pool plus b_T bytes of L2 reads, against an L2 facet at the card's bandwidth, with honest's own L2
  traffic uncharged. `--lookup-q`, `--lookup-bytes` and `--l2` override the file. A design with b-bit noise entries on
  the operand the tables index (`[table] bits`, or `--index-bits B`) pays b times the route's slots and bytes.
- `--uni-issue C`: charges each uniform instruction C times an ALU instruction's cost on the issue rows too (`issue` and
  the joint `JI…_0` row). A census that already charges it there is unchanged.
- `--checking sampled`: adds ε_adapt for the design's own proof block; `--adapt UNIT` charges another of
  `data/adapt.toml`. The Python `eps()` charges none unless given one.

## The headline is per node

`machine-model headline`'s ε, and the Lean headline guarantees exported from this data
(`Accounting.Guarantees.TS.Sm120SheetHeadline` and `Sm120MeasuredHeadline`), bound work done by the node: an accepted
window's node, its GPUs, host CPU and peer GPUs together, performed at least (1 − ε) of the work the window credits. The
device that did it is left to the attribution layer. The floor argument places no work: every credited word is a fresh
function of the post-coin noise, so it was computed after the coin, but nothing ties a committed tile to the GPU that
computed it. The transcript is a Merkle root over tile digests, so a helper holding the salt ships one 64-byte digest a
tile. The model prices the work on one device's rows, which makes ε a share of the credited time `T_cred` on that
device, not of that device's own time.

A per-GPU reading of ε comes only through the attribution layer, under two conditions:

- **The host deduction**: what the host CPU can compute of the work the window credits is deducted exactly from the
  GPUs' credit. On int8 that is AMX for a tile's products and AVX-512 for the fold and credited fillers, each bit-exact.
- **The peer-GPU premise**: every billable GPU on the node is credited each window, so no uncredited peer computes a
  credited GPU's tiles over NVLink. Otherwise the node is the unit.

Counted per GPU without them, the host's AMX and AVX-512 (and NVLink peers) take every device over 1% on some real host.

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
  - `table.toml`: the table route at the device's SMs and f_max. Without an `l2`, the route is bound by its slots alone.

  `data/adapt.toml` holds for every device. A census with no uniform-datapath line (`UNI`) closes the uniform route.

Each device also has a data-sheet vector, `DS-NN`: its `DS_*` rows, which a vendor document states, at their exact
values, and no measured or joint row. They are `DS_issue` (every kind 1 SMSP-cycle, uniform included), `DS_tensor` (each
MMA form's MACs over the data sheet's dense peak for its type), the pipe rows of the Programming Guide's throughput
table on Nsight Compute's pipes, and `DS_smem` (one 128 B request a SM-clock for each shared-memory instruction,
whatever its width). Each also reads one measured row, `DS_uniform`: no document states the uniform datapath's rate, so
the sheet's uniform pipe is the census's UP line. `eps --vector DS-NN` prices a design on the vendor's numbers alone.
sm_120's is `DS-120`, an epoch of its own whose lines close where they open, so it moves no other vector; sm_90's and
sm_100's are in `data/devices/sm_90/census/DS-90.csv` and `data/devices/sm_100/census/DS-100.csv`.

A device's data-sheet constants are its `sheet.toml`, each with the document that states it and its grade:

- `machine-model --device sm_90 sheet` derives every DS line from it, checks them against the census file, and checks
  each per-clock peak against the figure its document prints at the clock it's quoted at.
- `sheet --lean` prints the Lean module `sheet.toml` names (`Security/Definitions/Accounting/Sheets/`), which the tests
  hold to that output and whose `machine_eq_census` proves the machine is the census.
- `sheet --eps` prices the headline design's sheet reading on the sheet's own lines, with each peak read per second at
  f_max, and finds how much slower every tensor line can be before that ε moves; `sheet --eps --uni C …` prices it with
  a uniform instruction at each C of an issue slot.

The H200 and B200 have model honest profiles (one IGMMA.64x256x32 or UTCIMMA.128x256x32, then duo on every committed
word) and no measured f_H, so γ is the model's and ε_clock is 0. Both read their operands from shared memory by
descriptor, so the operand floor is no instruction. The B200's accumulator is in TMEM, so its floor is the
tcgen05.ld.32x32b.x32 that brings each committed word to registers, 1024/L a MMA.

## The opcode table

The card premise (`CardWithinAccel`) is that every instruction a card executes is dominated by an ACCEL op (`Accel.Op`)
at no fewer units of the same resource. `data/cover/` is its evidence on the H200 (sm_90) and the RTX PRO 6000 (sm_120):
evidence, not trusted base, since no verifier and no Lean reads it. `machine-model cover` prints it as Markdown, a table
a card, with the counts by mark and every hole.

- **The envelope** (`envelope.toml`) is `Accel.Op` with its pipes, formats and each card's numbers (ρ a SM-clock, format
  weights): `mma`, `mmaSp`, `lane` (a lane opcode on a pipe: at most the pipe's `n_in` words in, 2 out), `move`, `atom`,
  `red`, `branch` and none. What the table needs that `Accel` lacks is marked `proposed`: the `collective` op and
  resource, `tex`, and the formats `e3m2`, `e2m3`, `mxf8` and `mxf6`.
- **The pipes**: an opcode goes on the pipe a run or the CUDA Programming Guide places it on, else on `fma`, which
  charges only the issue slot every instruction takes (coarse, and sound on any pipe).
- **What must be covered**: every opcode of NVIDIA's family tables (CUDA Binary Utilities 12.9 §4.3 and §4.4 in
  `../sass_census/data/bu_12_9.json`, CUDA 13's in `sass_isa_tables.json` beside it), every dotted form of the CUDA 13
  libraries' SASS corpus (`corpus.json`, with instruction counts), every census kind, and every unit the card table
  lists (`theta.UNITS`). The three JSON tables are generated, so they are fetched fixtures (`fixtures/artifacts.json`,
  gitignored): `suites.py` fetches them before the tests, and `uv run research data fetch-fixtures
  benchmarks/pouw/machine_model/data/cover benchmarks/pouw/sass_census/data` brings them into a checkout. `rows.toml`,
  the reviewed table, stays in Git.
- **A row** (`rows.toml`) names its opcodes, devices, census kinds, mark, envelope op with its pipe, stores or formats,
  units a warp instruction, register words, rate (card-table keys, or `rate_value` with its unit), argument and source.
  A row with `forms` (regexes over the dotted form) covers those forms; the opcode's plain row covers the rest. A unit
  outside the SM pipes is a `[[unit]]` with the same marks.
- **The marks**: `covered` (dominated at the rate), `coarse` (dominated at a coarser bound: a pipe's share, the issue
  slot, the data sheet's peak), `hole` (`low`, bounded at a few percent of a resource; `medium`, measured past a ρ;
  `open`, no rate known; or `outside the card`) and `absent`. An opcode's mark is its worst row's. A row resting on a
  proposed entry is counted twice: with it, and as a hole on `Accel.Op` alone.
- **The rule**: each rate of a covered or coarse row, in its op's units a SM-clock (tensor MACs at the format's weight,
  halved for `mmaSp`; a pipe's lanes, `fma`'s the issue slot's 128; fetches; a store's words), is at most its resource's
  ρ, and a lane's or collective's words at most its op's.

`tests/test_cover.py` fails when an opcode, corpus form, census kind or unit has no row, a form matches two rows, a row
names an op, pipe, format or card-table key that doesn't exist, a dominated row's rate or words exceed its op's, or a
hole has no severity.

The evidence a row's source cites, each a `research run` with its verdict line:

- `../sass_census/lowering.sh` (`--on auto`) compiles each case of `lowering.cu` for sm_90a and sm_120a and checks the
  SASS it lowers to or that ptxas refuses it (`verdict: N/N cases lower as expected`).
- On one leased RTX PRO 6000, through `../opcode_probe/run.sh`: `../opcode_probe/probes/match.cu` (MATCH.ANY by class
  count and .U64, beside LOP3 and FFMA) and `collective.cu` (REDUX and VOTE beside LOP3, REDUX beside MATCH.ANY), whose
  verdict is `../opcode_probe/collective_verdict.py`; both build for sm_90a too (`../dc/emp.sh build`, then `emp.sh
  time` on the H200). `pipes.cu` times each lane opcode beside its pipe's partner (LOP3, MUFU.EX2 or DFMA), and
  `pipes_verdict.py` holds each to the pipe's ρ.
- `../sass_census/multimem.sh` reads sm_120's multicast support and runs LDGMC and `multimem.red` on unicast memory.
