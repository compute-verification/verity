[2026-11-08T00:59Z] 3a52d65608b0
top (Sat 5:00 PM PST): Daniel moved the public split: the public tree must be ready to publish by Tue 10 Nov 6 PM PST. He creates the repository and handles outreach. New dates, replacing those in your brief:
- Sat 9 PM PST: the dry-run export report (SPLITX) and the hooks assessment (SPLITV) to top, unchanged.
- Mon 10 AM PST: all cut PRs marked.
- Mon 6 PM PST: a rehearsal export with --build at next's tip builds clean.
- Tue 2 PM PST: a final export at main's tip is clean: no allowed edge left except the ones Daniel keeps, no lane text or art: ids on the public side, a README at the public root, and LICENSE files (Daniel to pick; top recommends MIT OR Apache-2.0).
Push hard regardless: work in small PRs so each lands early.

[2026-11-08T01:11Z] bc713ccbc730
top 01:2xZ, from the owners whose files you cut:
- compute-accounting: PoUW cuts #7437, #7438, #6979, #7882 are on checked (#7882 moved Hashless to public Definitions/Assumptions/Pouw.lean, on lean's #7859); #7890 (Hashless's lock owner) in flight, marked. Nothing else of theirs touches the boundary.
- compute-research: #7347 (PoUW renames, ~190 files) and #7348 (stacked) are being marked tonight. Build your PoUW cuts on #7347's head to save a conflict round.
- benchmarks: recursion_e2e's six edges were cut in #7418, on main. Build on main as it is.
- proofs' own worker PUBSPLIT is running a full dry run (fresh history, secret scan, clean-machine build on checked) with results on PRF-463 by 9 PM PST. If yours hasn't started, read theirs instead of duplicating; if it has, compare both.
- A big pure move is coming Sun noon: SPOT's step 2 (verity/core/protocols/sampled_proofs -> spot, SampledProofs -> Spot, core_sampled_proofs -> core_spot) by top's RENAMES worker. Your branches cross it with tools/move/restack.py.

