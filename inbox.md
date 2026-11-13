[2026-11-13T02:24Z] 6f0c76e21bd7
TOPCODE, Thu 12 Nov 6:23 PM PST.

**The named steps are in as you ruled, but "none has a forward-only step" is false: 23 other entries' records hold forward-only steps their configs count, so under the rule as ruled those 23 entries' verdicts move from equal to differs; I need your ruling on them before the PR.**

What is pushed on `cursor/training-topcode-9478` (head efb10c7e07):
- 594067e3e2, of_record: `record_only_steps` (beside `step`; `[[held."ENTRY".record_only_steps]]`, `number` in the record from 1, and `why`), one paragraph in `of_record.toml`'s header and one in the module docstring. The steps fact compares the config's steps with the record's less the named ones. The verdict says "N named steps of the record's program alone, T tokens". The modal line ("which n of its N steps have") and E's size ("after N") take the record's steps less the named, the count the config's steps stand for. Differences: a named step with a backward; a named number the record lacks; a record's step with no backward the entry doesn't name ("name each in the entry's `record_only_steps` with a `why`, or find why the record's program takes it"). Tests: both probes named, equal; one unnamed, differs; a named step that trains, differs; a named number past the record, differs. RAN locally: `tests/test_of_record.py` 21 pass (slow included), ruff clean.
- eef4a10b40: the 81M config at 5,387 steps (its schedule over them), its two probes named: step 1, `cut_probes("theta")`, and step 5389, `cut_probes("end")` (READ, the record's `out/train.py` lines 1152 and 1215). RAN locally: `steps_alone` on the record gives no difference, 2 named steps, 524,288 tokens.
- efb10c7e07: the 238M config at 20,205 steps, the plan's `cap_gates` 1.0084763e22 (READ, `out/plan.json` of r20261106-183414-1416, run 16) over 3.37075e14 MACs × 1,480.7. The trace (RAN r20261113-014549-a81c) gives ⌊…⌋ = 20,205.

The finding: RAN locally, `steps_alone` over all 34 entries' records (the 28 records, fetched earlier from their `art:` ids). 24 entries get a new difference: mine and 23 others. Their records hold steps with no backward, and their configs' steps equal the record's whole count (READ, `attacks/*.toml`), so the configs train those steps:
- ref-optim entries (10): 3 to 24 forward-only steps late in the run at the modal step's tokens. Examples: `s-t20-1.1x-ref-optim-fullmuon-758` at 643, 704 and 758; `xl-t20-1.1x-ref-optim-fullmuon-fe-wsd` 24 steps from 2044.
- fullmuon entries and their -ema twins (12): one forward-only step, the record's last, at 32,768 tokens where the modal step has 65,536 or 196,608.
- `m-t20-2x-hybrid-grow-depth` (GROW's): 25 steps, 1, 322, 644, … every ~321.
- 10 entries have none (the two uncoded topcode, the five shallow, the three qat).
INFERRED: these are evaluation forwards (a probe of the loss), counted by the harness as steps; I have not read their programs.

So the rule as ruled turns 23 equal verdicts into differs, and each fix is that entry's config's steps less its forward-only ones, plus a `why` read from its program. That changes 23 configs that aren't mine. Choose one:
(a) Keep the rule. The 23 differ until their owners name the steps and correct the steps (GROW for `m-t20-2x-hybrid-grow-depth`). My PR's body lists them. That is honest: their configs do train more steps than their records did.
(b) Keep the rule, and I name all 23 in this PR after reading each program, with each config's steps lowered. That is a larger PR touching other lanes' configs.
(c) Narrow the unnamed-step difference to records whose entry names any step, so the 23 stay as they are until someone looks. I'd advise against this: it leaves a known miscount silent.
I'd go with (a). Until you answer I hold the PR body's verdict paragraph and change nothing outside my two configs.

Runs:
- dry run r20261113-020146-1c6c: done, rc 0, at the old steps. I'm rerunning both configs' dry runs at efb10c7e07.
- every-entry comparisons r20261113-015001-f3a4 (GROW's head) and r20261113-015028-3b8e (dea3ccc1a1): still running on vy-nebius-cpu-14 and -10.
- suites r20261113-020600-ec6d: still running on vy-nebius-cpu-1.
I'm also launching the every-entry comparison at efb10c7e07, which will show the 23 moves as RAN.

