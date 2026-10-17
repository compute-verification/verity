# The machine model

PoUW's price vectors for sm_120 (rs-machine-model §8): the LP that prices every warp instruction as a convex combination
of the machine's rows, its inputs as data, the attack families, and `machine-model`, which re-solves a named vector and
checks every attack against it. Each named vector (VF … VF3k-16) re-solves here to the τ, τ at 600 W, U/F_low and
margin it was published with, at the vertex the legacy scripts landed on (`tests/test_vectors.py`), and `solve` says
whether (π₃₂) and PRICES-32's load half hold at its prices (`data/floors.csv`).

~~~bash
uv run machine-model list                 # the named vectors
uv run machine-model solve VF3k-16        # τ, τ at 600 W, U/F_low, margin, θ, the cheapest attacks, the check, the claims
uv run machine-model solve                # model.toml's `current` (VF3k-16, decision 17's active vector)
uv run machine-model census VF3k-16 --kind ATOMS64
uv run machine-model attacks VF3k-16 -n 3 # the cheapest attacks of every design
uv run machine-model adopt ~/.research/runs/RUN/out/census.csv EPOCH   # a probe's census at the last epoch
uv run machine-model --data /tmp/candidate solve VF3l   # a copy of data/ with a candidate change
~~~

| Path | What |
|---|---|
| `data/model.toml` | The constants (R*, the honest kernel's u a MAC, at 600 W too), the LP's ρ, tie and x_min, the honest GEMM, the CUDA kinds, the adopted design and the epochs in order |
| `data/rows.csv` | One line per row of the machine (a pipe, a port, a ceiling or a joint vertex): its status (`measured`, `ASSUMED`, `provisional`, `joint`), the runs or notes it rests on, and what it is |
| `data/census.csv` | One line per instruction class on a row: its measured rate (warp instructions/SM/clk, with `noise` its derating), or the SMSP-cycles it holds the row, or a derived u; the epochs it holds (`since`, `until`) and its runs |
| `data/vectors/NAME.toml` | A named vector: its epoch, the rows and credits it uses in LP order, its note, and the numbers it was published with |
| `data/attacks/FAMILY.jsonl` | An attack family: one program a line, its design, and the epochs it holds |
| `data/floors.csv` | What the claims read: each root-writing kind at 32 MACs a word ((π₃₂)) and each load into registers at 16 (PRICES-32's load half), with the words a warp instruction of it writes |
| `verity_machine_model/lp.py` | The LP (Charnes–Cooper; stage 1 U/F_low, stage 2 τ), as the legacy `rs-price-99.py` `solve` computes it |
| `verity_machine_model/attacks.py` | An attack's X, the cheapest of a design, and the check: every price at or below measured cost, ρ, every credit, every attack at X ≥ 1; a kind's price a word, and what is under its floor |
| `verity_machine_model/adopt.py` | Merging a measured census into `data/` at the last epoch, each (kind, row) at the faster of its two lines |
| `verity_machine_model/resolve.py`, `cli.py` | Re-solving a named vector from the data, and the command |

An *epoch* is a state of the census and the attacks, named after the vector that introduced it. A census line or an attack
holds from its `since` epoch (empty: the first) through its `until` epoch (empty: the last), so every vector re-solves on
the data it was published on while new measurements land beside it. A kind with no coefficient on a vector's rows is
refused, and the check fails on it.

## Add a measurement

A new measurement is lines of data and a vector file, not a script:

1. **The row**, if the measurement is a new row: a line in `data/rows.csv` with its status, its run ids
   (`rYYYYMMDD-HHMMSS-xxxx`, or a `note:` id) and what it measures.
2. **The census lines**: one line in `data/census.csv` for each instruction class the row binds, with exactly one of
   `rate` (the measured warp instructions/SM/clk; `noise` 0.02 for the usual 2% margin), `cycles` or `u`, `since` the new
   epoch, and its runs. A refit of an existing line sets the old line's `until` to the last epoch it held and adds the new
   value with `since` the new epoch.
3. **The epoch**: append the new vector's name to `epochs` in `data/model.toml`.
4. **The vector**: copy the vector it builds on to `data/vectors/NAME.toml`, set `name` and `epoch`, add the row to `rows`
   (and any credit to `credits`), and drop `[published]` and `[legacy]`.
5. `uv run machine-model solve NAME`: it prints τ, τ at 600 W, U/F_low and the margin, and checks every price, credit and
   attack. Record the printed numbers under `[published]` once you publish them; the tests then hold the vector to them.

A new attack family is a new `data/attacks/FAMILY.jsonl` (`since` the epoch it joins); the check prices every attack of
every vector's epoch. A new root-writing kind or register load also needs its line in `data/floors.csv`, which `solve`
reads for the claims.

## Adopt a probe's census

An opcode probe (the `opcode_probe` beside this directory, #2160) writes its gated rows as `census.csv` in this
schema, with no epochs. After steps 3 and 4 above, `uv run machine-model adopt CENSUS EPOCH` merges it at `EPOCH`, which
must be the last epoch. Each (kind, row) keeps the faster of its line at the epoch before and the measured one: a row's
u must be a lower bound on what every program pays on it, so a line slower than the hardware makes a price that isn't
sound. A faster line closes the old one at the epoch before, adds its own from `EPOCH` and adds its runs to the row's in
`rows.csv`; a slower one changes nothing. A probe case the census has no row for (VF3k-16's peer-CTA DSMEM atomic) is a new row and, if it is
slower than its opcode's other cases, its own kind (`ATOME32P`): a line for the old kind there would overstate it.

## Where the data came from

The legacy chain (the notes repo's `lanes/compute-accounting/gpu-model/rs-price-*.py`, kept as history) built each
vector in a script that loaded the previous one. `data/` is those scripts' LP inputs, captured at each named vector's
solve (that directory's `machine-model-export.py`) and normalized (`machine-model-convert.py`, which regenerates
`data/` byte for byte, all but `model.toml`): a census line's value is its rate, cycles or u, whichever form reproduces
the scripts' float exactly. Two differences from the
scripts: an instruction class's measured cost is the largest of its census lines at the epoch (the scripts froze it before
VF3f added `KFFMA0`, which is FFMA's F row rounded to 10 places, 4e-10 relative above it, inside the check's 1e-9); and
a line's runs are the runs its row's description cites, else the runs its vector's script cites.
