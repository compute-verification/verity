# The machine model

PoUW's price vectors for sm_120 (rs-machine-model §8): the LP that prices every warp instruction as a convex combination
of the machine's rows, its inputs as data, the attack families, and `machine-model`, which re-solves a named vector and
checks every attack against it. Each named vector (VF … VF3o-16) re-solves here to the τ, τ at 600 W, U/F_low and
margin it was published with, at the vertex the legacy scripts landed on (`tests/test_vectors.py`), and `solve` says
whether (π₃₂), PRICES-32's load half and π_c ≥ 16 hold at its prices (`data/floors.csv`), and lists every copy route's
price a word beside its measured cost.

~~~bash
uv run machine-model list                 # the named vectors
uv run machine-model solve VF3o-16        # τ, τ at 600 W, U/F_low, margin, θ, the cheapest attacks, the check, the claims
uv run machine-model solve                # model.toml's `current` (VF3o-16, the vector of record since 17 Oct after VF3n-16; VF3l-16 by decision 62 before them)
uv run machine-model census VF3o-16 --kind ATOMS64
uv run machine-model attacks VF3o-16 -n 3 # the cheapest attacks of every design
uv run machine-model adopt ~/.research/runs/RUN/out/census.csv EPOCH   # a probe's census at the last epoch
uv run machine-model --data /tmp/candidate solve VF3l   # a copy of data/ with a candidate change
uv run machine-model eps                  # Dither's ε budget at q* = 8: every term, the LP's mixes, the ε(q*) curve
uv run machine-model eps dither-two-sided --q 2.2 --json
uv run machine-model eps --table         # every design
uv run machine-model eps --table --operands floor   # the cheater's operands at the noise's floor φ_del
uv run machine-model eps --table --credit floor     # ε a fraction of the floor program ♭, not honest's kernel
uv run machine-model eps --table --credit floor --q 2.16 --lookups   # with QSTAR's table route against L2
uv run machine-model eps --table dither-duo --fold-sweep 1,1.5,2,2.5,3
uv run machine-model eps --table --credit floor --uni-issue 0.92   # a uniform instruction takes an issue slot
uv run machine-model eps --device sm89 --table --credit floor --operands floor   # the L40S: Dither-1S/2S/C at L = 16, 32
uv run machine-model eps --device sm100 --table --credit floor --operands floor   # the B200 at L = 32 to 256
~~~

| Path | What |
|---|---|
| `data/model.toml` | The constants (R*, the honest kernel's u a MAC, at 600 W too), the LP's ρ, tie and x_min, the honest GEMM, the CUDA kinds, the adopted design and the epochs in order |
| `data/rows.csv` | One line per row of the machine (a pipe, a port, a ceiling or a joint vertex): its status (`measured`, `ASSUMED`, `provisional`, `joint`), the runs or notes it rests on, and what it is |
| `data/census/EPOCH.csv` | The census lines from EPOCH, a file for each epoch that brought any: one line per instruction class on a row, its measured rate (warp instructions/SM/clk, with `noise` its derating), or the SMSP-cycles it holds the row, or a derived u; the epochs it holds (`since`, its file's epoch; `until`) and its runs |
| `data/census/runs.toml` | The run lists many census lines share (a refit's every line cites every run its fit read), each cited in a line's `runs` as `@NAME` |
| `data/vectors/NAME.toml` | A named vector: its epoch, the rows and credits it uses in LP order, its note, and the numbers it was published with |
| `data/attacks/FAMILY.jsonl` | An attack family: one program a line, its design, and the epochs it holds |
| `data/devices/sm_NN/` | Another device's data ("Devices" below): `sm_89`, the L40S, from EPS1 ADA; `sm_90`, the H200, and `sm_100`, the B200, from EPS1 DC (`dc/census.py` writes their census) |
| `data/floors.csv` | What the claims read: each root-writing kind at 32 MACs a word ((π₃₂)), each load into registers at 16 (PRICES-32's load half) and each copy route at 16 (π_c ≥ 16, `copy16`: every kind that writes registers, a UR word at the 32 lanes it serves), with the words a warp instruction of it writes |
| `verity_machine_model/lp.py` | The LP (Charnes–Cooper; stage 1 U/F_low, stage 2 τ), as the legacy `rs-price-99.py` `solve` computes it |
| `verity_machine_model/attacks.py` | An attack's X, the cheapest of a design, and the check: every price at or below measured cost, ρ, every credit, every attack at X ≥ 1; a kind's price a word, what is under its floor, and a claim's kinds priced a word beside their measured cost |
| `verity_machine_model/adopt.py` | Merging a measured census into `data/census/` at the last epoch, each (kind, row) at the faster of its two lines |
| `verity_machine_model/resolve.py`, `cli.py` | Re-solving a named vector from the data, and the command |
| `data/honest/NAME.toml` | An honest kernel's instruction profile a MMA (every class it issues, the fold inside it), the least operand delivery any program pays a MMA, and its measured SMSP-cycles a MMA: `e-l7qs` is PROFILE's measured kernel; `c-l7qs` (co32c's linear), `h-l7qs` (cohalf's QK^T) and `pv-l7qs` (P·V) are Dither-C's, measured on `benchmarks/pouw/edited_gemm/`'s bit-exact kernels; `dither-duo-l32` the provisional one the tests' hand arithmetic reads |
| `data/table.toml` | The cheater's table route (`eps --lookups`): lane instructions and L2 bytes a word, and the L2 bandwidth it is bound by (QSTAR's, provisional until a DATASHEET row) |
| `data/designs/NAME.toml` | A protocol's parameters: its honest profile, L, its noise's operand-delivery floor φ_del, the fold and filler (lane instructions a word by class), plain (unfolded) MMAs beside its work, ε_skip's and ε_mand's parts with sources, ε_clock, Theorem S's δ, p and checks, and honest's baseline program (dp4a) |
| `verity_machine_model/eps.py` | `eps`: the ε budget of the tensor–scalar model (note:20261017T0310Z-report-eps1-program §2) and ε_machine's LP |

An *epoch* is a state of the census and the attacks, named after the vector that introduced it. A census line or an attack
holds from its `since` epoch (empty: the first) through its `until` epoch (empty: the last), so every vector re-solves on
the data it was published on while new measurements land beside it. The census lines from an epoch are its own file, so
the census grows by a file an epoch; the tests keep every file under a quarter of the repository's 256 KiB cap. A kind with no coefficient on a vector's rows is
refused, and the check fails on it.

## Add a measurement

A new measurement is lines of data and a vector file, not a script:

1. **The row**, if the measurement is a new row: a line in `data/rows.csv` with its status, its run ids
   (`rYYYYMMDD-HHMMSS-xxxx`, or a `note:` id) and what it measures.
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

Per-row keys close (RT) only if every copy route costs π_c ≥ 16 MACs a word (note:20261016T1747Z-draft-fullfloor-s-r12):
MOV, any landed word, UR fan-out, every SHFL mode, and any kind whose output can equal an input word. The `copy16`
claim reads every kind that writes registers, a superset of those, so a route can't be missed. VF3k-16 and VF3l-16 fail
it on the UR fan-out alone (UNI 2 and UNI64 1 MACs a lane-word); the candidates `VF3l-16-copy16`,
`VF3l-16-copy16-fan32`, `VF3l-16-copy32-fan16` and `VF3l-16-copy32` credit the routes they name, each at or below its
measured cost. They wait for XOR-in. Theorem O's every-run floor reads only (π₃₂) and the load half: a UR word is one
value, so the lanes it serves past the first are copies, credited only by copy luck.

## ε in the tensor–scalar model

`eps` reports two layers that are never mixed (Daniel, 17 Oct 06:29Z). The protocol with 100% checking, every word and
operand verified, has the core ε, 1 − ε_core = (1 − ε_skip − ε_mand − ε_machine)(1 − ε_sched)(1 − ε_clock)
(`--checking full`, the default). The sampling compiler adds ε_sample = ε_luck + ε_adapt, a function of the proof unit,
its budget and its selection rule, 0 at 100% sampling, on its own line and in its own columns (`--checking sampled`).
Each term is a fraction of honest time. Π is the current vector's rows at its epoch, read as the price LP reads them (SMSP-cycles a warp
instruction a row). ε_machine is one LP, SPEC's §0.8 (note:20261017T0324Z-draft-eps1-spec): a program writes every
committed word on one of four routes, folds every word and does the filler, and its time is its largest row. The routes
are the tensor pipe (`L/32` MMAs for 128 words, with their operand delivery), CUDA lanes at q* lane instructions a word
in any mix of the CUDA classes, the design's baseline program (dp4a: 8 IDP4A and its loads a word) and the uniform
datapath (q* uniform instructions a word, the fold there too). Honest is the tensor route at the profile's cost plus the
baseline, whose words belong to the baseline and not to ε, and the cheater is every route at q*. The cheater's operand
delivery is free by default (SPEC's conservative choice), `--operands floor` prices it at the noise's floor φ_del (the
design's `[operands]`: 1/39 a MMA where B̃ stays resident at the zero instance, 0.20 where both operands are fresh, SPEC
§3.4) and `--operands honest` at honest's. The LP's duals are the certificate π*, at which ε_machine is ε_cert + ε_route.
ε_luck is the same LP with Theorem S's budget ln(1/δ)/S spent on unwritten words (still folded) and undone filler, each
deficient block weighted by 1 − p (¾ a word, 0.854 a filler block; SPEC §5), so the terms add up to its time exactly.
ε_sched needs a measured profile (`[measured]`) and is otherwise 0, honest certified at the model's bound.
`--credit model` reads ε as a fraction of honest's model bound B(H) (PROFILE's reading: ε_sched and ε_clock leave ε),
and `--credit floor` as a fraction of the floor program ♭ (SPEC §0.7a, Corollary D3): the tensor route at φ_del, the
fold, the fillers and the baseline, with no overhead. Honest's time above the credit then leaves ε for γ (printed beside
it); which reading a guarantee certifies is the coordinator's call. γ is honest's time a useful MMA over the plain
GEMM's model time: measured when the design runs the profile's kernel, otherwise the measured kernel's time plus the
model's difference between the design and that kernel (PREDICT's arms: within −0.7% to +8.4% of measured on lane
work), and the model's alone without a measured profile.
`--upfill` adds verified filler that keeps the uniform datapath busy for the credited program's time, which the cheater
must then run on lanes or leave in place; it is not adopted (decision 73: a uniform instruction takes an issue slot, so
UPFILL costs honest issue time, EPS1 P10, r20261017-055801-e051) and stays only for the tests' comparison. `--fill X` adds X lane instructions a word of verified filler on lanes,
which dilutes what the cheater saves on operands at the cost of γ (`dither-two-sided-fill` is two-sided dither with
4.5). `--lookups` opens a fifth route for the cheater, the table route (`data/table.toml`, QSTAR's register-indexed
lookups): q_T lane instructions a word from the lanes' class pool plus 8 B of L2 reads, against an L2 facet at the
card's bandwidth, honest's own L2 traffic uncharged (`--lookup-q`, `--l2` override the file). ε_machine's floor is its
value with the lane and uniform routes closed (the table route, priced by its bytes, stays as set), which no q* lowers,
and the threshold is the least q* at which ε_machine reaches it. `--uni-issue C` charges each uniform instruction C
times an ALU instruction's cost on the issue rows too (`issue` and the joint `JI…_0` row), which the census left out
from VF3 to VF3o-16 and PREDICT's u16s_l7qs measured at 0.92 (r20261017-045133-67bf). The uniform route then closes
without UPFILL, and UPFILL costs honest issue time while the cheater does its checked work on lanes, 32 values a warp
instruction. From VF3p-16 (decision 73) the census charges it there itself, and `--uni-issue` changes nothing. ε_adapt is the proof unit's, SPEC's
Theorem D under the one-σ order (`data/adapt.toml`): today's segment spot check ≈ 21%, whole dot products +0.35%, and 0
for whole-dot-product proofs that recompute their operands one stage back or for per-stage coins. `--checking sampled`
charges today's by default and `--adapt UNIT` another; the Python `eps()` charges none unless given one. The table
route's defaults (`data/table.toml`) are SCALAR-ROUTE as restated: 1.22 slots and 3.5 B of L2 a word, `--lookup-bytes 6`
for single-field tables. A design with b-bit noise entries on the operand the tables index (`[table] bits`, or
`--index-bits B`) pays b times the route's slots and bytes (QSTAR's b-bit count); at b = 2 the route closes.

## Adopt a probe's census

An opcode probe (the `opcode_probe` beside this directory, #2160) writes its gated rows as `census.csv` in this
schema, with no epochs. After steps 3 and 4 above, `uv run machine-model adopt CENSUS EPOCH` merges it at `EPOCH`, which
must be the last epoch. Each (kind, row) keeps the faster of its line at the epoch before and the measured one: a row's
u must be a lower bound on what every program pays on it, so a line slower than the hardware makes a price that isn't
sound. A faster line closes the old one at the epoch before (in the old line's file), adds its own to
`data/census/EPOCH.csv` and adds its runs to the row's in `rows.csv`; a slower one changes nothing. A probe case the census has no row for (VF3k-16's peer-CTA DSMEM atomic) is a new row and, if it is
slower than its opcode's other cases, its own kind (`ATOME32P`): a line for the old kind there would overstate it.

A joint row a re-measurement finds violated (a measured point past 1) is unsound whatever the faster rule says: it closes
at the epoch before, and the new frontier's rows replace it. `opcode_probe/eps1.py facets --prior data/rows.csv --emit
EPOCH` prints them, the upper hull of the new points and the old rows' vertices with the 2% margin. VF3m-16 closed R2-R4,
KIMAD, KHFMA2 and KSHFL that way for EIMAD, EHFMA2 and ESHFL, and replaced the ASSUMED SHFL by its measured 8 SMSP-cycles.
VF3n-16 closed the joint issue rows JI3e_0..2 for JI3n_0..2 (note:20261017T0321Z-draft-eps1-tensor): IMMA m16n8k16 and
TF32 m16n8k4 beside IADD3 load JI3e_1 past 1. Each JI3n row is JI3e's refit over every native mma.sync form's programs and
VF3e's at the 2% margin, then divided by 1.07 for the 6.9% one program's time varied between builds; MMA takes every form's
least, and each form is its own kind on them (`IMMA16816`, `HMMA1684T`, …, with its `copy16` line). VF3o-16 replaced
the 1.07 with a measured margin (EPS1 TENSOR's M1c, r20261017-052102-d832, the tensor_kinds probe's schedule search):
600 schedules of the forms that set JI3n (lane op, register layout, 2 to 6 warps a SMSP, placement) join the refit, and
each JI3o row is divided by the most a searched program loads the row fitted to M1 alone, over 0.98 (1.038, 1.026,
1.025), so a search as wide again finding as much again stays covered; the probe's refit with its search runs prints
the rows.

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
sm_89 (the L40S, EPS1 ADA), sm_90 (the H200) and sm_100 (the B200, EPS1 DC), and for any later device. The directory
follows these rules:

- It holds `rows.csv` and `census.csv` in `data/`'s schema, and its rows follow "Measure a facet". Its probe's
  `census.py` writes them from gated runs.
- An MMA kind is one SASS form, as on sm_120's JI3n rows.
- It has one epoch, its lines without `since` or `until`. `rows(DIR)`, `entries(m, DIR)` and `census(m, epoch, DIR)`
  take the directory.
- To price a design on the device, it gains the rest of `data/`'s files, and `--device sm_NN` (or `--data DIR`) reads
  it:
  - `model.toml`: its epoch, its vector and its CUDA lane classes. The constants only `solve` reads stay out.
  - `vectors/NAME.toml`: its rows.
  - `honest/`: its honest kernel at each L. `[mma] class` names the MMA's census class (sm_120's is `MMA`).
  - `designs/`: each design at each L, with the device's honest kernel, operand floor φ_del, clock and ε_clock, luck
    budget and baseline. The noise, ε_skip and ε_mand are sm_120's design's.
  - `table.toml`: the table route at the device's SMs and f_max. Without an `l2`, the L2 isn't measured, and the route
    is bound by its slots alone.

  `data/adapt.toml` holds for every device. A census with no uniform-datapath line (`UNI`) closes the uniform route.

The L40S (`data/devices/sm_89/`, note:20261017T0605Z-draft-eps1-ada) prices Dither-1S, Dither-2S and Dither-C at L = 16
(IMMA.16816, where T1 is tight) and at L = 32 (IMMA.16832, half the tensor pipe's words), on ADA's measured honest step:
duo's fold, 19.07 and 22.93 SMSP-cycles a MMA. Its clock is f_max = 2,520 MHz, with honest at 1,680 MHz at L = 16 and
1,585 MHz at L = 32 under the enforced 325 W cap. Its operand floors are SPEC §3.4's register count applied to each
form: 1/94 and 7/72 a MMA for m16n8k16, 1/39 and 0.20 for m16n8k32. ε_skip at L = 16 scales L = 32's strips by their
widths. `eps --table` prints γ_f, γ with the clock gap f_max/f_H.

The H200 and B200 (`data/devices/sm_90/` and `sm_100/`, note:20261017T0452Z-draft-eps1-dc) price the same three designs
at L = 32, 64, 128 and 256 on model honest profiles: one IGMMA.64x256x32 (4096 words a warp) or UTCIMMA.128x256x32
(32768 words), then duo on every committed word. Neither has a measured honest kernel or f_H yet, so γ is the model's
and ε_clock is 0. Both read their operands from shared memory by descriptor, so the operand floor is no instruction for
every noise. The B200's accumulator is in TMEM, so its floor is the tcgen05.ld.32x32b.x32 that bring each committed word
to registers, 1024/L a MMA. Honest's profile counts that load in its fold, and the design's fold leaves it out, so a
word written on lanes doesn't pay it. The baseline is dp4a on IMAD's class, as on sm_120, with its loads unpriced (the
census has no IDP4A or LDS kind).
