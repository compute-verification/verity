# Lowering PoUW's H1 circuit files

This benchmark measures what writing a PoUW H1 circuit file costs, in peak memory and wall time, and holds the file it
writes to a pinned digest. It lowers three statements of an R × C tile (the leaf): the lane statement, one circuit for
all the leaf's lanes; the combine, which joins the lanes; and the whole H1 tile, the auditor's lowering whose file a
PoUW period-open commits. It proves nothing.

## Try it

Every lowering runs remotely (AGENTS.md "Tests"). The file goes in a scratch directory, and only numbers go in `$OUT`:

```text
research run --on auto --cwd source --mem-gb 64 --timeout 60m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu &&
  d=$(mktemp -d) && trap "rm -rf $d" EXIT &&
  .venv/bin/python benchmarks/unit_lowering/run.py --unit lane --k 2048 --tile 16,16 --scratch "$d" --out "$OUT" [--guard]'
```

`run.py` prints one `UNIT-LOWERING unit=.. k=.. tile=RxC form=.. peak_gib=.. wall_min=.. ands=.. bytes=.. k_log=..
sha512=<16 hex>` line and writes `OUT/unit_lowering.json`.

## The two numbers, the guard and the target

1. **Peak RSS in GiB** to write the K 2048 lane file, and the combine file, the lowering child's `ru_maxrss`, at each leaf.
2. **Wall minutes** of the same child.

Each file's META `k_log` must be at most 27, Flock's largest block (compose's `K_MAX`, the prover's `IN_RANGE_K_LOG`, Lean's
`Stmt.InRange`). Compose gives the unit section a power-of-two slot beside the SHA and hm96 ranges (`circuit._ranges`), so
k_log ≤ 27 needs the unit under 2^26 rows. A record whose k_log is past 27 makes `run.py` exit 1.

The guard holds the flat file's bytes. A shape passes when its SHA-512 equals `reference.json`'s (d40fe21d26's lowering) and
its k_log is at most 27. The tile is bound to `{"partition": "0"*128, "program": "0"*128}`.

- `--guard` checks `PINS`: (64, 4 × 8) and (256, 8 × 8) tiles, the lane at K 448 and the combine at K 448. Together they take
  about 10 min and 12 GiB.
- `--guard all` adds `WIDE`: the (2048, 4 × 8) tile and the (64, 32 × 32) tile, about 40 and 56 GiB.

`tests/benchmarks/pouw/test_hidden_zk_stage.py` must still pass.

| | K 2048 lane | K 2048 combine |
| --- | --- | --- |
| target | ≤ 16 GiB, ≤ 15 min, k_log ≤ 27 | the same |
| stretch | lanes at K 8192 and K 14336 in ≤ 64 GiB, ≤ 60 min and 2^27 | |

Main's lowering (the approach `baseline`) and every approach since are on the leaderboard ([`DESIGN.md`](DESIGN.md)).

[`DESIGN.md`](DESIGN.md) has the three statements in full, what a caller may import, every command and flag, and the
leaderboard.
