# unit_lowering

What writing a PoUW H1 circuit file costs, and the file it writes, for three statements of an R × C tile (the leaf):

- `lane`: the leaf's lane statement `PouwH1LaneZy_v1{TILE}`. It is one circuit for all the leaf's lanes, each 4 rows by
  8 columns, with the lane index a public input.
- `combine`: the leaf's `PouwH1Combine_v1{TILE}`, which joins the lanes.
- `tile`: the whole H1 tile unit, the auditor's lowering whose file a PoUW period-open commits as `k<K>`
  (`python -m tests.protocols.pouw_h1_circuits --lower K R C BIND PATH` from `integrations/vllm`).

The tile is a parameter. 32 × 32 is the H1 kernel's warp tile, never a choice made for proving. The lane and the combine are
what can be proved once the tile is past the block (`verity/core/protocols/pouw/leaf_lanes.py`, `benchmarks/pouw/h1_lanes/`).
`lanes.py` stages each one as `h1_lanes/run.py`'s `stager` does, with its `RC` set to the leaf, bound to the Program a served
window's verifier derives for it. They exist only at an R that is a multiple of 16: a lane reads the leaf's R admit bits as
one row, and a row is whole 16-bit words (`h1zy._leaf`). So at 4 × 8 and 8 × 8 the tile itself is the statement; a 4 × 8 tile
is one lane. Every lowering runs with `HZ_STAGE=fast` and with `H1_TMPL` unset (flat) or `structural`.

## The two numbers, the guard and the target

1. **Peak RSS in GiB** to write the K 2048 lane file, and the combine file, the lowering child's `ru_maxrss`, at each leaf.
2. **Wall minutes** of the same child.

Each file's META `k_log` must be at most 27, C-Flock's block cap (compose's `K_MAX`, the prover's `IN_RANGE_K_LOG`, Lean's
`Stmt.InRange`). Compose gives the unit section a power-of-two slot beside the SHA and hm96 ranges (`circuit._ranges`), so
k_log ≤ 27 needs the unit under 2^26 rows. A record whose k_log is past 27 makes `run.py` exit 1.

The guard holds the flat file's bytes. A shape passes when its SHA-512 equals `reference.json`'s (d40fe21d26's lowering) and
its k_log is at most 27. The tile is bound to `{"partition": "0"*128, "program": "0"*128}`.

- `--guard` checks `PINS`: (64, 4 × 8) and (256, 8 × 8) tiles, the lane at K 448 and the combine at K 448. Together they take
  about 10 min and 12 GiB.
- `--guard all` adds `WIDE`: the (2048, 4 × 8) tile and the (64, 32 × 32) tile, about 40 and 56 GiB.

`benchmarks/pouw/tests/test_hidden_zk_stage.py` must still pass.

| | K 2048 lane | K 2048 combine |
| --- | --- | --- |
| target | ≤ 16 GiB, ≤ 15 min, k_log ≤ 27 | the same |
| stretch | lanes at K 8192 and K 14336 in ≤ 64 GiB, ≤ 60 min and 2^27 | |

Main's lowering (the approach `baseline`) and every approach since are on the leaderboard below.

## Running it

Every lowering runs remotely (AGENTS.md "Tests"). The file goes in a scratch directory, and only numbers go in `$OUT`:

```text
research run --on auto --cwd source --mem-gb 64 --timeout 60m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu &&
  d=$(mktemp -d) && trap "rm -rf $d" EXIT &&
  .venv/bin/python benchmarks/unit_lowering/run.py --unit lane --k 2048 --tile 16,16 --scratch "$d" --out "$OUT" [--guard]'
```

- `run.py --k K --tile R,C [--unit tile|lane|combine] [--form flat|structural] --scratch DIR --out DIR [--guard [all]]
  [--tree PATH]` writes `OUT/unit_lowering.json`. The record holds `unit`, `k`, `tile`, `form`, `peak_gib`, `wall_min`,
  `bytes`, `ands`, `k_log`, `sha512` and `commit`, plus each section's rows. It prints `UNIT-LOWERING unit=.. k=.. tile=RxC
  form=.. peak_gib=.. wall_min=.. ands=.. bytes=.. k_log=.. sha512=<16 hex>`.
- A tile is R = 4 i rows by C = 8 j columns (i j lanes, `h1tile.Layout.lanes`), at a K that is a multiple of 32 from 64. The
  1 × 1 and 4 × 4 tiles fail in `h1tile._xor`. A lane or combine also needs R to be a multiple of 16, and the combine needs
  K from 384.
- `--guard` exits 1 and names each shape whose SHA-512 or k_log fails. `--tree PATH` runs another checkout's lowering (a
  worktree of your approach) under this harness.
- `fit.py RECORD...` fits ANDs, bytes, peak and wall on one unit's terms and projects them:
  - lane and combine, one leaf at a time: 1 and K, projected to K 2048, 8192 and 14336 at that leaf;
  - tile: 1, R C, R C K and (R + C) K, projected to (2048, 32 × 32), (8192, 32 × 32) and (128, 32 × 32).
- `mem_profile.py --k K --tile R,C [--unit tile|lane|combine] --scratch DIR --out DIR [--frames N]` runs the lowering
  in-process. It reports:
  - each phase's peak: `unit_lowering`, `ir_lower.lower`, `ir_lower.layout`, `circuit.compose`, the netlist, the digests;
  - a census of live objects by type at the peak;
  - with `--frames`, tracemalloc's allocation sites.
- `verity bench unit_lowering ...` is `run.py ...`.

`reference.json` comes from `run.py --pin-reference`, or from measured runs of d40fe21d26's lowering, one `run` id each.

## The leaderboard

Every measured run gets a label:

```text
research data label RUN unit_lowering '{"unit":"lane","k":2048,"tile":[R,C],"form":"flat","peak_gib":..,"wall_min":..,
  "ands":..,"bytes":..,"k_log":..,"peak_b_per_and":..,"file_b_per_and":..,"guard":"pass|fail|n/a","approach":"<approach>",
  "commit":"<branch>@<sha>"}' --value-json --by LANE --ref RUN
```

Runs of d40fe21d26's lowering carry the approach `baseline`. To read the leaderboard:
`research data sql "select target, by, value from labels where key='unit_lowering'"`.
