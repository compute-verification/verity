[2026-11-12T20:12Z] 6059e96ebd50
PUT, Thu 12 Nov, 12:12 PM PST

**All three checks pass; putting now.**

1. Fetch: art:55b35cfb… fetched to a directory outside any checkout; each of bodies/*.json has the sha256 bodies.json names (algorithm-policy 864688df47ed85f4…, frontier-fit a1c878ff75814936…, standard-workloads 78a409b4509f6cfa…, size-scaling d59b340eedd613ce…). RAN (sha256sum on the fetched files, against bodies.json).
2. Checkout: origin's cursor/training-put-55a1 is d76ae46e6f005334c2e4d8c952a85282fdf48d4d (git ls-remote); a detached worktree at it, 0 status lines, te_core tree 11f297ba3e3dcb53…. RAN.
3. Nobody PUT since staging: a GET at 2026-11-12T20:11:38Z gives each of the four panels the same sha256 and updated_at as served/meta.json (algorithm-policy 2026-11-11T05:39:00Z, frontier-fit 2026-11-05T04:47:24Z, standard-workloads 2026-11-05T07:28:18Z, size-scaling 2026-11-04T18:06:50Z), and the GET's panels.json is byte-identical to served/panels.json. The route also serves training/methods, which none of the four puts touch. RAN (te_core.console.served from the clean checkout).
4. Dry puts: the README's four lines with --dry, from benchmarks/covert_witness of that checkout, each print "console.put would refuse: nothing" with the sha256, bytes, rows and note length bodies.json has, rc 0; each output is byte-identical to S/dry/NAME.txt. The checkout still has 0 status lines afterwards. RAN.

[2026-11-12T20:12Z] e6cf49d59133
SORT, TRN-146: #8950 stops. Its by-width plans are superseded by #9112, and what is left of it is proofs' decision. I'm changing its verdict from "still wanted" to "superseded in part, proofs decides".

What #8950 is for: each row of the break-even driver (`benchmarks/spot_figures/breakeven/headline.py`) names the plan its design is held to: the planner's own, by width, or by width at a held κ.

- The carry is pushed: `cursor/carry-8950-dc90` at 611f0327c7ede5926a3432c41aa27e8aa5172311, #8950's head with `main` 30d6fcc92ff7 merged in (2 files: `headline.py`, `tests/test_headline.py`). RAN (`git ls-remote`). Do not take it as is.
- Its quick suites, `r20261112-192338-9a48`, ended rc 1 with the last line `6 suites: 5 passed (0 cached), 1 failed`. The `benchmarks/spot_figures` suite had 233 passed and 1 failed: `test_headline.py::test_a_row_s_design_is_held_to_its_plan`, `assert ('template' is None)` at line 307. RAN (fetched stdout.log).
- Why it fails: #9112 (1f64540734, landed as 862a214346) made a two-stage line's strata one per block type: `STRATA = "template"` in `total_cost.py`, passed as `stratify=tc.STRATA` by `breakeven/common.py`. Its message says this replaces "the planner's width-stratified design, whose strata of several types the verifier's rule cannot state". With `stratify="template"`, `two_stage_design` tries only the "template" threshold, so #8950's `widths=True` does nothing and the design's `merged` is "template". READ (`git log -1 1f64540734`; `verity/protocols/spot/optimize.py` `two_stage_design`; `common.py`).
- The question for proofs is whether a by-width row is still wanted at all. Forcing `stratify=True` would make the test pass, but it would price a design the verifier can't register, so I haven't done it. The held-κ rows (`kappas`) could stay as by-template rows at a held κ, but that is a new plan, not a carry. Nothing was pushed past the merge.

