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

## What's broken

- **The tile at K 2048 can't be proved.** On 32 × 32 it is about 736M ANDs, a 2^30 block, and C-Flock caps a block at 2^27
  (compose's `K_MAX`, the prover's `IN_RANGE_K_LOG`, Lean's `Stmt.InRange`).
- **Its flat file also never finished.** Run r20261028-151803-ed19 was killed out of memory at 600 GiB after 2 h 50 min, with
  no file. The same limit stopped K 4096 on 23 Oct (templated 32 × 32 past 600 GiB) and 24 Oct (killed at 898.1 GiB after
  8,661 s), and EmbeddingRead_v1's flat compose on 25 Oct (431 GB).
- **The lane needs about 1.76 KB of memory per AND.** At K 2048 that is 40.1 GiB, 2.5 times the target, though its 12.7 min
  meets the target's time (r20261028-192717-8754).
- **Past K 4096 the lane can't fit C-Flock's block.** Compose gives the unit section a power-of-two slot and puts the SHA and
  hm96 ranges beside it (`circuit._ranges`), so k_log ≤ 27 needs the unit under 2^26 rows, about 67M. A lane has
  1.73M + 11,060 K ANDs: about 92M at K 8192 and 160M at K 14336. The lane's k_log was 25 at K 448, 26 at K 1024 and
  K 2048, and 27 at K 4096.

## The two numbers and the guard

1. **Peak RSS in GiB** to write the K 2048 lane file, and the combine file, the lowering child's `ru_maxrss`, at each leaf.
2. **Wall minutes** of the same child.

Each file's META `k_log` must be at most 27. A record whose k_log is past that makes `run.py` exit 1.

The guard holds the flat file's bytes. A shape passes when its SHA-512 equals `reference.json`'s (d40fe21d26's lowering) and
its k_log is at most 27. The tile is bound to `{"partition": "0"*128, "program": "0"*128}`.

- `--guard` checks `PINS`: (64, 4 × 8) and (256, 8 × 8) tiles, the lane at K 448 and the combine at K 448. Together they take
  about 10 min and 12 GiB.
- `--guard all` adds `WIDE`: the (2048, 4 × 8) tile (40 GiB, 13 min) and the (64, 32 × 32) tile (56 GiB, 18 min).

`benchmarks/pouw/tests/test_hidden_zk_stage.py` must still pass.

| | K 2048 lane | K 2048 combine |
| --- | --- | --- |
| baseline (d40fe21d26) | 40.11 GiB, 12.67 min, k_log 26 (r20261028-192717-8754) | 2.80 GiB, 1.00 min, k_log 25 (r20261028-192737-c341) |
| target | ≤ 16 GiB, ≤ 15 min, k_log ≤ 27 | the same |
| stretch | lanes at K 8192 and K 14336 in ≤ 64 GiB, ≤ 60 min and 2^27 | |

A lane's ANDs are linear in K: 1.73M + 11,060 K, within 1.4% of each lane measured. That is about 92M at K 8192 and 160M at
K 14336. Its peak and wall time grow faster than that past K 2048:

| K | peak GiB per K | wall min per K | peak B/AND | file B/AND |
| --- | --- | --- | --- | --- |
| 448 → 1024 | 0.0168 | 0.0051 | 1,797 → 1,693 | 373 → 428 |
| 1024 → 2048 | 0.0190 | 0.0060 | 1,693 → 1,764 | 428 → 489 |
| 2048 → 4096 | 0.0323 | 0.0076 | 1,764 → 2,428 | 489 → 571 |

The K 4096 lane (r20261028-193352-e765) took 106.2 GiB and 28.1 min, with k_log 27. If the last step's rates hold, a lane
needs at least 238 GiB and 59 min at K 8192, and 437 GiB and 105 min at K 14336. `fit.py`'s linear peak doesn't describe
this: its residual over the four lanes is 46%.

The fixed part is about 2.5 GiB and 0.8 min, which is the combine's whole cost. So meeting the target at K 2048 takes at most
about 590 B a lane AND. For the block, the stretch takes a unit of under 2^26 rows (about 66.8M ANDs): 28% fewer ANDs at
K 8192 and 59% fewer at K 14336.

## By leaf, at K 2048

A lane is 4 rows by 8 columns at any leaf, so one statement's cost barely depends on the leaf. These are all measured, flat
on d40fe21d26's lowering:

| leaf | statement | run | ANDs | peak GiB | wall min | k_log |
| --- | --- | --- | --- | --- | --- | --- |
| 4 × 8 | tile (its one lane) | r20261028-195458-d820 | 24,353,765 | 39.783 | 12.32 | 26 |
| 8 × 8 | tile (two lanes) | r20261028-195504-f270 | 45,063,832 | 74.334 | 23.82 | 27 |
| 16 × 16 | lane (8 lanes, one circuit) | r20261028-195336-48bf | 24,374,031 | 39.957 | 12.61 | 26 |
| 16 × 16 | combine | r20261028-195348-bdfb | 4,223 | 2.797 | 0.81 | 23 |
| 32 × 32 | lane (32 lanes, one circuit) | r20261028-192717-8754 | 24,411,115 | 40.110 | 12.67 | 26 |
| 32 × 32 | combine | r20261028-192737-c341 | 4,223 | 2.799 | 1.00 | 25 |

The 8 × 8 tile costs 1.85 lanes and is at the cap (k_log 27): it is the largest tile that is one block at K 2048, and its
74.3 GiB and 23.8 min miss the target as a lane does, at the same 1.77 KB an AND. The tile as one statement is past the block
from 16 × 16. `fit.py` over the nine tiles puts it at about 159M ANDs at 16 × 16, and 606M at 32 × 32.

## The baseline

These are main's lowering at d40fe21d26 (the harness's branch changes none of it), flat, and labelled with the approach
`baseline`. "ANDs" is the unit section's AND rows. Each file also holds the fixed `sha512x3` (244,206 ANDs) and `hm96`
(241,490 ANDs) sections.

| unit | K | tile | run | ANDs | bytes | peak GiB | wall min | k_log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combine | 448 | 32 × 32 | r20261028-192707-13f4 | 1,023 | 56,466,630 | 2.801 | 0.89 | 25 |
| combine | 2048 | 32 × 32 | r20261028-192737-c341 | 4,223 | 59,390,029 | 2.799 | 1.00 | 25 |
| lane | 448 | 32 × 32 | r20261028-192650-520b | 6,587,943 | 2,456,417,158 | 11.025 | 3.55 | 25 |
| lane | 1024 | 32 × 32 | r20261028-192712-4f61 | 13,126,706 | 5,613,470,942 | 20.699 | 6.48 | 26 |
| lane | 2048 | 32 × 32 | r20261028-192717-8754 | 24,411,115 | 11,928,476,042 | 40.110 | 12.67 | 26 |
| lane | 4096 | 32 × 32 | r20261028-193352-e765 | 46,979,920 | 26,845,806,510 | 106.223 | 28.15 | 27 |
| combine | 448 | 16 × 16 | r20261028-195343-dc43 | 1,023 | 55,787,617 | 2.797 | 0.81 | 23 |
| combine | 2048 | 16 × 16 | r20261028-195348-bdfb | 4,223 | 56,569,248 | 2.797 | 0.81 | 23 |
| lane | 448 | 16 × 16 | r20261028-195330-37f4 | 6,551,065 | 2,452,561,376 | 10.936 | 3.44 | 25 |
| lane | 2048 | 16 × 16 | r20261028-195336-48bf | 24,374,031 | 11,925,211,408 | 39.957 | 12.61 | 26 |
| tile | 2048 | 4 × 8 | r20261028-195458-d820 | 24,353,765 | 11,966,464,478 | 39.783 | 12.32 | 26 |
| tile | 2048 | 8 × 8 | r20261028-195504-f270 | 45,063,832 | 23,734,216,333 | 74.334 | 23.82 | 27 |
| tile | 64 | 4 × 8 | r20261028-184657-1d80 | 2,384,742 | 660,464,727 | 4.912 | 1.79 | |
| tile | 64 | 8 × 8 | r20261028-190833-eb4e | 3,819,091 | 1,155,450,279 | 6.687 | 1.98 | |
| tile | 256 | 4 × 8 | r20261028-190839-670c | 4,155,238 | 1,517,594,569 | 7.594 | 2.34 | |
| tile | 64 | 8 × 16 | r20261028-190845-6837 | 6,683,373 | 2,141,785,547 | 10.236 | 3.24 | |
| tile | 256 | 8 × 8 | r20261028-190851-b98d | 7,336,803 | 2,871,022,299 | 12.061 | 3.53 | |
| tile | 64 | 16 × 16 | r20261028-190910-0fa3 | 11,787,763 | 4,110,837,122 | 16.895 | 4.92 | |
| tile | 2048 | 4 × 8 | r20261028-190932-398d | 24,353,765 | 11,966,464,478 | 39.781 | 12.64 | |
| tile | 64 | 32 × 32 | r20261028-190926-461a | 41,138,509 | 16,625,178,423 | 55.934 | 17.65 | |
| tile | 128 | 32 × 32 | r20261028-191720-a7b2 | 59,560,013 | 26,409,786,801 | 83.364 | 27.43 | |

The tile runs above predate run.py's read of k_log. `fit.py` over the nine tiles (ANDs within 10%, peak within 7%) puts
(2048, 32 × 32) at about 606M ANDs, 900 GiB and 320 min. That is consistent with ed19's kill at 600 GiB after 170 min, and the
32 lanes of K 2048 hold 781M ANDs between them.

## Where the memory goes

`mem_profile.py --unit lane` on the 32 × 32 leaf's lane, at K 2048 (r20261028-200838-4945, 24,411,115 ANDs) and K 4096
(r20261028-200857-090d, 46,979,920 ANDs). A phase's peak includes its sub-phases', and its seconds leave out the census:

| phase | K 2048: peak GiB | B/AND | seconds | K 4096: peak GiB | B/AND | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| `ir_lower.lower_gates` | 30.10 | 1,323 | 208 | 86.85 | 1,984 | 433 |
| `ir_lower.layout` (`fast_stage.ands_layout`) | **40.11** | 1,764 | 189 | **106.23** | 2,428 | 330 |
| `circuit.compose` (the netlist, the dummy) | 38.02 | 1,672 | 262 | 103.66 | 2,369 | 596 |
| `write_text` | 37.03 | 1,628 | 21 | 101.13 | 2,311 | 48 |
| the whole lowering | 40.11 | 1,764 | 727 | 106.23 | 2,428 | 1,479 |

The peak is the layout, on top of everything `lower_gates` built. These are the live objects, per unit AND, at the two
phases' exits. The census counts objects the garbage collector tracks, by `sys.getsizeof`, so ints and the allocator's slack
aren't in it:

| type | what | `lower_gates`, K 2048 | K 4096 | `ands_layout`, K 2048 | K 4096 |
| --- | --- | --- | --- | --- | --- |
| frozenset | the linear forms' variable sets (`gf2.L.s`) | 454 | 1,080 | 27 | 15 |
| tuple | the gates' operand forms | 621 | 680 | 621 | 680 |
| list | the layout's rows | 36 | 33 | 698 | 773 |
| L | the linear forms (`gf2.L`) | 100 | 99 | 98 | 97 |

Two things grow faster than the ANDs:

- **The variable sets.** There are 2.0M of them at K 2048 and 3.0M at K 4096, averaging 5.7 KB and 17.1 KB each, so they hold
  10.3 GiB and 47.3 GiB at `lower_gates`' exit. That is the superlinear term in the lane's peak. The layout frees them.
- **The rows' nonzeros.** There are 57 an AND at K 2048 (1.40G) and 66 at K 4096 (3.11G), each an 8-byte pointer in a Python
  list. Compose's time follows them: its netlist took 160 s and 367 s.

The small tile has the same shape, at a smaller scale. At (64, 8 × 16) (r20261028-190803-e482) the peak is the same phase at
1,645 B an AND, with the sets at 92.

**The next lever.** Meeting 16 GiB at K 2048 takes about 590 B a lane AND. `lower_gates` alone ends at 1,323, so no single
one of the three is enough: the sets, the gates' tuples and the rows all have to leave Python objects, or never be held
whole. Lowering and laying out gate by gate would hold only the live forms: write each row as it's made, and drop a form once
its last reader is laid out. If the rows must be held, int32 arrays take 228 B an AND at K 2048 (5.2 GiB) and 264 at K 4096
(11.6 GiB). The sets' growth with K says the forms should be kept short as they're built, not only stored more compactly.

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
