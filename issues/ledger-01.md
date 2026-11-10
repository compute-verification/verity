to: @proofs
title: Every run that feeds the SPOT post carries a stamp of the configuration of record: its digest and the switches the run has off (Wed 11 Nov)

**Ask.** Put the post's configuration in one data file. Have every run that feeds the post write a stamp: the file's digest, the run's value of each switch, and the switches at which it differs (its off list).

**Why.**
- No run is at the configuration of record: the most any run has at the post's value is 6 of 22 switches [read: the ledger's sections 2 and 3].
  - r20261108-195819-b00b serves at a 131,072 context with prefix caching on; nothing in it is sampled, proved or verified [run: `out/boundary/meta.json`].
  - r20261108-200706-1758 runs behind the network certifier at a 40,960 context with prefix caching off [run: `out/meta.json`].
  - r20261109-144301-d7bf times proofs of drawn values, and its `out/zk/observations.json` has `"verify": null` on both rows [run].
- No run's record names the switches it has off. Today one reads `server_argv` in the serve's `meta.json` (`benchmarks/serve_tax/serve_tax.py:591-598` on `main`) [read].
- A card names runs and a status, and no configuration (`benchmarks/spot_figures/cards.py:6-8`) [read].

**Done when.**
- `benchmarks/spot_figures/record.json` holds the 22 switches and the post's value of each.
- `record.py stamp` writes `$OUT/record.json` as `{digest, observed, off}` from a serve's `meta.json`, so the serve does not change.
- Run on b00b's and 1758's stored `meta.json`, it prints the off lists in the ledger's section 3.
- Thursday's serve, proving and planner runs each have a stamp in the store, with one digest. A two-stage serve's `meta.json` is private (r20261109-220303-3352 publishes `out/private.txt` only [run]), so its script publishes the stamp.
- The stamp holds no request, count or time (vllm's rule on VLLM-250, Mon 9 Nov 23:10Z) [read].
- `cards.py check` refuses a GLM-5.3 card with no `record: {digest, off}`.

**Pointers.**
- Note `proofs/20261109T2010Z-report-claims-ledger`, sections 2, 3 and 6.
- Goal `proofs/spot-claims-checked`.
- Top's freeze (TOP-387): this adds files and moves nothing.
