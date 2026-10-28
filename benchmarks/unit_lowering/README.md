# unit_lowering

What writing the flat circuit file of PoUW's H1 tile unit costs: the auditor's lowering whose file a PoUW period-open commits as
`k<K>` (`python -m tests.protocols.pouw_h1_circuits --lower K R C BIND PATH` from `integrations/vllm`, `HZ_STAGE=fast`,
`H1_TMPL` unset), at K and an R × C tile.

## What's broken

At K 2048 on the 32 × 32 tile the flat file never finishes. Run r20261028-151803-ed19 was killed out of memory at 600 GiB after
2 h 50 min with no file. The same wall stopped K 4096 on 23 Oct (templated 32 × 32 past 600 GiB) and 24 Oct (killed at
898.1 GiB after 8,661 s), and EmbeddingRead_v1's flat compose on 25 Oct (431 GB).

## The two numbers and the guard

1. **Peak RSS in GiB** at K 2048, tile 32 × 32, flat: the lowering child's `ru_maxrss`.
2. **Wall minutes** of the same child.

The guard: the flat file's SHA-512 at each shape of `reference.json` equals d40fe21d26's, bound to
`{"partition": "0"*128, "program": "0"*128}`, and `benchmarks/pouw/tests/test_hidden_zk_stage.py` still passes. The flat file's
bytes may not change.

| | K 2048, 32 × 32 |
| --- | --- |
| baseline (d40fe21d26) | above 600 GiB and 170 min (killed); fit: about 950 GiB, about 245 min |
| target | ≤ 16 GiB, ≤ 15 min |
| stretch | K 8192 in ≤ 64 GiB, ≤ 60 min |

A tile is R = 4 i rows by C = 8 j columns (i j lanes, `h1tile.Layout.lanes`), and K is a multiple of 32 from 64 up
(`h1tile`'s scheme): 1 × 1 and 4 × 4 tiles fail in `h1tile._xor`. The smallest shape is K 64 on 4 × 8.

## Running it

Every lowering runs remotely (AGENTS.md "Tests"), with the file in a scratch directory and only numbers in `$OUT`:

```text
research run --on auto --cwd source --mem-gb 32 --timeout 60m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu &&
  d=$(mktemp -d) && trap "rm -rf $d" EXIT &&
  .venv/bin/python benchmarks/unit_lowering/run.py --k 256 --tile 8,8 --scratch "$d" --out "$OUT" [--guard]'
```

- `run.py --k K --tile R,C [--form flat|structural] --scratch DIR --out DIR [--guard] [--tree PATH]` writes
  `OUT/unit_lowering.json` (`k`, `tile`, `form`, `peak_gib`, `wall_min`, `bytes`, `ands`, `sha512`, `commit`, each section's rows)
  and prints `UNIT-LOWERING k=.. tile=RxC form=.. peak_gib=.. wall_min=.. ands=.. bytes=.. sha512=<16 hex>`.
- `--guard` lowers every shape of `reference.json` flat and exits 1 naming each shape whose SHA-512 differs; `--tree PATH` runs
  another checkout's lowering (a worktree of your approach) under this harness.
- `fit.py RECORD...` fits ANDs, bytes, peak and wall on 1, R C, R C K and (R + C) K over run records and projects
  K 2048, K 8192 and K 128 at 32 × 32.
- `mem_profile.py --k K --tile R,C --scratch DIR --out DIR [--frames N]` runs the lowering in-process and reports each phase's
  peak (`unit_lowering`, `ir_lower.lower`, `ir_lower.layout`, `circuit.compose`, the netlist, the digests), a census of live
  objects by type at the peak, and with `--frames` tracemalloc's allocation sites.
- `verity bench unit_lowering ...` is `run.py ...`.

`reference.json` comes from `run.py --pin-reference` or from measured runs of d40fe21d26's lowering, one `run` id each.

## The leaderboard

Every measured run gets a label:

```text
research data label RUN unit_lowering '{"k":2048,"tile":[32,32],"form":"flat","peak_gib":..,"wall_min":..,
  "guard":"pass|fail|n/a","approach":"<approach>","commit":"<branch>@<sha>"}' --value-json --by LANE --ref RUN
```

d40fe21d26's own runs carry the approach `baseline`.
