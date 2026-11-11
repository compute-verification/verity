[2026-11-11T02:17Z] 2e83643662c5
REMOTEPY to coordinator, Tue 10 Nov 6:17 PM PST. **Item 5 is pushed: `cursor/net-remote-nonlean-9f54` at `d38d1641cd03`, one commit (`d38d1641cd`) on vllm's `c1c9244e02b9`; `tests/test_vllm_boundary.py` and `tests/test_vllm_public_api.py` on that tree, on the VM: "6 passed in 10.63s".** The commit changes `benchmarks/prove_path/prove_path.py` alone, 3 lines added and 5 removed, and the push was a fast-forward from `48c8f46e7193` [run: `git diff --name-only c1c9244e02b9 HEAD`, `git push`]. The run at the head follows; I name it next.

[2026-11-11T02:19Z] dace42ad6e49
LEANMERGE to network-accounting, Tue 10 Nov 6:19 PM PST. **NET-325, tonight's report: the refusal is in `factsOf`, the whole change is written and pushed, Lean passes on all three files and the package builds. I wait on one run, the suites: `r20261111-020707-bbfd`. Wed 11 Nov noon PST holds.**

**Branch:** `cursor/net325-drawn-le-units-22f3` at `28528994d9a5806c82af8d9b6a78a97b6e1a82ea`, two commits over `2f30e1709f1e`, five files, no lock, no `--update`.

**The place: `factsOf`, your preference.** One line, `unless drawn ≤ l.size do refuse …`, between the sampling design and the bound [read: `NetworkCheck/Period.lean:1132`]. It holds of every scan and every audit, tied or not. I looked first, and nothing accepted today turns into a refusal:
- Before this change no `#guard` of `Period.lean` called `factsOf` or `check`, and every literal facts record in them has `drawn ≤ units` [read].
- No tracked JSON holds a profile level or a facts record that samples above its size [run: a scan of the tree's JSON on the VM].
- The mirror's accepted kinds all sample within the level, and the users outside the suite ask only for accepted or honest cases (`case("accept", …)`, `build("rows-honest", …)`) [read].
- Every old change's case is byte for byte what it was (540 cases, seeds 0 to 2, both sizes), and the mirror answers them as before [run: VM, pure Python].

**What is written.**
1. *The check.* The refusal, with the reason "…: profile: K sampled of N units: more than the level holds", and two `#guard`s: a level sampled whole or in part gives its facts, `subset:k` and stratified; 3 of 2, and strata of 2 of 1 and 1 of 1, are refused before the bound is read.
2. *The mirror.* The same refusal in `_facts`, at the same place in the findings' order [read: `period_case.py:951`]. Four changes: `audit-oversampled` and `rows-audit-oversampled` (an audit, tied or not), `rows-tied-oversampled` (the tied audit of the aimed scan), `rows-answer-oversampled` (a scan). One test, `test_a_profile_that_samples_more_than_its_level_holds_is_refused`: each change at four seeds, Lean and the mirror both refuse at the sample, with the bound gone too, and both accept the same period with that level sampled whole. It asserts that a tied and an untied audit, a scan, `subset` and stratified were all met.
3. *The property.* `RowsTiedChargedAt` gains the conjunct `f.drawn ≤ f.units` [read: `Security/Properties/NetworkPeriod.lean:151`]. `RowsTiedCharged`'s docstring now says a tied audit that is not among `charged` is sampled whole, and the sentence on `units ≤ drawn` is gone. The proof reads it off the refusal by one new lemma, `factsOf_sampled_le_units`. The guarantee's name and both lemmas' names stand. `_honest` and `_nontrivial` are untouched: the nontrivial lemma's failing check does not change.
4. *Nothing else of the check.* `auditCheck`, `auditChecks`, `chargedOf` and `rowsWindowCheck` are as they were. Not NET-328.

**What has run**, each at the head, `--on auto`, each on vy-nebius-cpu-3 [run]:
- `r20261111-020401-220f`, try on `NetworkCheck/Period.lean`: "lean exit 0 in 13 s", "axiom probe: no theorems in it", exit 0.
- `r20261111-020401-504e`, try on the statement's file: "lean exit 0 in 24 s", exit 0. Its build of the imports printed "Built NetworkCheck.Period (2.5s)".
- `r20261111-021117-eeaf`, try on the proofs file: "lean exit 0 in 35 s", "axiom probe: none of its 37 theorems uses an axiom beyond propext, Classical.choice and Quot.sound", exit 0.
- `r20261111-020707-4179`, the period check's package, `lean_changed.py --build` on its 13 files: "Build completed successfully (70 jobs).", exit 0.
- `r20261111-020401-efe8`, my first try on the proofs file: exit 1 before Lean read the file. "error: read-only file system (error code: 30)" at `…/lean-records_build-0/Security/Proofs/.lake/packages/leansqlite/.lake`. That is cpu-3's Lean slot tree `build/0`, not the file: the same try on slot `build/1` passed. One for infra if you want it.

**Who else reads what I changed** [read: `git grep` on both]. The same three files on my branch and on `cursor/net200-fail-forms-cb67` at `2edb2e080d4c`: the check's file, the statement's file, the proofs file. Each elaborates on mine. FAILFORMS's `Fails.lean` names none of `factsOf`, `auditCheck`, `auditChecks`, `chargedOf` and unfolds none: it needs no line. Its branch does not hold `2f30e1709f1e`, so its copies of the three are `main`'s.

**What is left.**
- The suites: `r20261111-020707-bbfd`, `suites.py verity/core/protocols/network verity-core-composition verity-network-traces`, launched 6:07 PM PST, took its slot at once, in flock's `lake build` at 6:19 PM PST. **I end my turn on it: resume me when it ends.** Only it runs Lean against the mirror on the new cases.
- The base: not named yet. `cursor/lean-remote-merge-fc8d` is at `eb56c217f831` (six branches, without `2f30e1709f1e`), so I merged nothing. When lean names the ten-branch head I merge it and try the proofs file again.
- Then the end report, the note, and the one comment on NET-325.

**Least sure of.**
- The refusal compares the level's totals. A stratum that samples more than its own size, inside a level that still holds the sum, is accepted, as your `drawn ≤ l.size` says. Say if you want each stratum held too.
- The retired-terms lint will warn on four of the files, by the code's own names (`drawn`, the profile's `"draw"` key): 1 to 3 sites each beyond the base's.
- Not run: the kernel replay, `check.py`, a `--siblings` try. The lemma's name is new in its namespace [read: `git grep`].

**Apart from this job:** `r20261111-000406-1d52` ended on its own with rc 1 at 5:30 PM PST. I posted its verdict on LEAN-249 at 6:09 PM PST and sent you the same lines.

My note: `lanes/network-accounting/20261111T0217Z-state-net325-leanmerge.md`.

