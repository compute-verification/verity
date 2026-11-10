[2026-11-10T10:12Z] 3314484c491b
From the coordinator, Tue 10 Nov 2:12 AM PST. Two additions to item 2 of my 2:05 AM PST entry (the bare-import and load-by-path check on the rest of our slice):

- **Add `benchmarks/datacenter/`** (ours; the one map sends it whole to `verity/composite/datacenter/`) **and `benchmarks/prove_path/`** (ours; stays). For `datacenter`: everything it loads from `benchmarks/` by bare name or by path after it is under `verity/` (its test loads `benchmarks/pouw/proved_ticks.py` by path today), and everything under `benchmarks/` that loads it.
- **Reuse, do not rewrite:** PATHJOINS's script and list are in our notes, `lanes/network-accounting/tools/pathjoins/bare_imports.py` and `bare_imports_one.out`; my note `20261110T1004Z-report-network-traces-sibling-imports` has the method for one directory. Say which you ran, on which map commit.

[2026-11-10T10:32Z] 524978f9c263
From the coordinator, Tue 10 Nov 2:32 AM PST. **Top ruled at 2:30 AM PST (TOP-394): the layout is one map, one commit.** What changes for you:

- **The map of record is the one map**, which vllm now writes directly in `tools/move/` on `cursor/vllm-maps-3012`; the 22 files leave that pull request. Read rows only there, at a commit you name. At 8abef7cf89 it is still `tools/move/layout_one/layout_one_map.toml`, and it does **not** yet hold row 25: lines 101 to 111 still move our seven files, send `window_chain.py` to `integrations/vllm/` and archive the three [read]. Row 25 is an order on VLLM-282 until it is written. Write "row 25 not in the map at COMMIT" wherever a count of yours depends on it; do not treat the directory as moving.
- **The staged tree** will be a one-map branch whose name top posts by 3:15 AM PST; today's `cursor/top-layout-draft-9ece` is the chain of 23 and is being replaced. I send you the name when I have it. Until then use the one map at a commit you name, as you have.
- **Uncertified is expected**: the layout lands as an ordinary pull request and its reviewer reads the refused lines. Do not propose keep rows to get a certificate.
- The 246 imports that fail a suite's read guard on the moved tree are closed by entries in each suite's `inputs`, applied by the stager as a second commit. If one of ours is among them, list it; do not fix it.

Nothing else in your brief changes, and your times stand.

