[2026-11-14T07:14Z] d9d1adc5682d
## Coordinator, Sat 14 Nov 07:14Z: the overhead run: fix PR yours, leg A go, leg B shape 1 after a leg-b-ready note

Read `three-goals-state` (its head) and `overhead-run-sized` whole. My word, leg by leg.

**The fix pull request: yours.** SERVEDCARD has no live turn. Merge its `cursor/prove-time-tmpl-unit-6e08` at `499739a84f` into a branch of your own from `origin/main` (never push to its branch), rerun its tests, open it unmarked with `Goal proofs/spot-overhead-measured`, and say in the body whose work it carries. Add the table's reader for shape 1's rows in the same pull request, and make the row say what it is: measured from warm sessions (WARM=1, RUNS=4), not "steady state". A row never borrows the steady rows' label.

**Leg A (CPU, 136 sessions, about 1.9 core-hours): go**, as soon as the fix's scripts are at a pushed commit. Nodes by free slots, not cpu-6.

**Leg B: shape 1 (`stage_prove.sh`).** The goal asks a Lean verdict on every proving session, and shape 2 keeps no session for Lean. **But no GPU job and no staging on node A yet.** First, on CPU only, a note `leg-b-ready` with:
1. the `ks_form.py` pass over the 128 types: which the form mirror accepts (the goal excludes the ones it refuses), so the job's real count;
2. node A's free disk now, read not remembered, and the staging waves under `WIN_DISK_GB` that fit it, or the subset that fits;
3. what booking node A's hours takes, from `calendars/node-a.md` itself and the writing-runs skill: who writes the row, and whether it needs a lead's word on Slack (then I give it). Do not write the calendar row yourself unless the calendar's own text says a worker may;
4. the exact commands of the staging and of the two-card prove, with `--max-hours` or the lease that bounds each, and the budget line they run under.
I give the GPU word from that note. Its hour: by Sun 15 Nov 12:00Z, so the Sun 18:15Z staging start can still hold.

**Leg C (CPU): go once leg B's sessions exist**, same outputs as leg A.

**The other two goals** (`spot-verifier-cost`, `serving-cost-three-parts`): as your state note sizes them; their pull requests keep their days. If the three goals' texts are older than `main`, write the sentences that are now false in your next note, and I rewrite the goals.

