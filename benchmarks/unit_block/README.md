# One Boolean Definition as one proof block

`run.py` builds a word primitive's Boolean version as one Flock proof block (`verity_flock.class_statement.UnitStager`), each
parameter one `hm96-sha512/row/v2` bit row and its return one more, and measures two things against Flock's proved range of
2^27 witness bits (`verity_flock.circuit.K_MAX`):

- **the select alone**: the lowered unit, its ANDs and rows, and its slot of 2^`unit_log` bits;
- **with its rows' openings**: the whole block `circuit.compose` lays out, the unit beside every row's SHA-512 chain
  (`sha512x3` slots of three compressions) and hm96 slot, in 2^`k_log` bits.

`compose` refuses a block past 2^27. `run.py` lifts that bound while it builds, so a block past it is measured, and its record
says `fits: false` with the refusal it would get. `--stage N` stages N instances on the built circuit, each checked against the
word primitive, the first parameter's value in each from `--first`.

~~~text
research run --on vy-nebius-cpu-6 --cwd source --mem-gb 64 --timeout 60m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu &&
  d=$(mktemp -d) && trap "rm -rf $d" EXIT &&
  .venv/bin/python benchmarks/unit_block/run.py GatherF32x50304_v1 --first 0,50303,50304,-1 --stage 4 --scratch "$d" --out "$OUT"'
~~~

It writes `unit-block-WORD.json`: the model `circuit.unit_witness` reads off the Definition, the select's slot, the block's
`k_log`, ranges and rows with their compressions, each as a share of 2^27, the build's time and peak memory, and the staged
instances' outputs.

## Interface

- **Provides:** `run.py`, a command, and the result file it writes.
- **Loaded by:** nothing.
- **Loads:** `verity.definitions`, `verity.circuits`, `verity.commitments.rowleaf` and Flock (`verity_flock`).
