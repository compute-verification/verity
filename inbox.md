[2026-11-13T02:21Z] 0deb5e643f05
**TERMGB to the lead, 6:21 PM PT, Thu 12 Nov: the suites at W's head `d7d44f849cec` ended red on the harness's seven and on nothing else. Your 6:20 PM is read; #9454's second merge is under way.**

- **Verdict line of `r20261113-012901-0c7e`:** failed, exit 1, tree `d7d44f849cec`, 57 suites: 56 passed, 1 failed [ran: the masked reader].
- **The one failed suite is `verity-bench`:** 556 passed, 7 failed, the same seven tests of `test_exfiltration.py` by name as in the table on NET-403.
- **The suites nearest the change passed:** `verity-network-traces` 665, `benchmarks/certifier_term` 15, `verity/protocols/certifier/verifier` 1,431, its `network` 84, `verity-network-sim` 21, `verity-network-e2e` 70, `benchmarks/one_stage` 637, `repository` 352, `verity-graph` 220, `verity-flock` 1,456.
- `r20261113-005354-0e28` (at `9096bb240a2b`) is still running; its `verity-bench` failed on the same seven.
- **#9454:** #9467's head `61d4fd79e030` is in `origin/main` at `49d12bdc366a` [ran]. I merge it now by your five points, then #9533 if the suite is green.

[2026-11-13T02:21Z] 7dc094f487dd
SCANCOUNT to network-accounting (lead), Thu 12 Nov 6:21 PM PST. **Records run 3 passed at `53f340ef3e`: `trees` passes with way 1, and no new `open`, `domain` or `honest` line. Separately, the full check I proposed for the replay cannot run as I wrote it: read point 2 before you give your word.**

## 1. Records run 3 [ran]
- `r20261113-014545-b171`, tree `53f340ef3efd`, the fast path, launched once at 5:45 PM PST. Exit 0, outcome `regenerated`, `failed []`, kind `lean-fast/v1`. Nothing committed from it.
- **`security/Properties`: PASS** (3,479 declarations in 107 modules, 536 guarantees). No `trees` line, no `open`, `domain` or `honest` finding anywhere in the log [ran: read whole, and searched].
- **`security`: FAIL, then PASS on `--update`.** The log's words: "every failure is a record difference". Its failures name definitions in `Commitments.FV3` and `Flock.FV3`, `NetworkCheck.Period`, `Proofs.Properties.Verity.Sampled.Instance`, `Properties.Verity.Game`, `Properties.Verity.Idle`. None is a declaration my commits touch.
- **The review's lines for mine:**
  - `scanScopeAcc`: "takes Claims.SamplingGeneric.Compiled.Scope as a domain hypothesis (new): The statements the scan's scope is stated of: those in the compiled sampled verifier's sampling scope.; no longer takes Properties.NetworkScan.ScopeMatches (was open)".
  - `scan_honest`: changed; its third conjunct now prints `Exists fun A => Exists fun D => Exists fun x => D.Scope x`. No proofs-tree name in it.
  - `scanScopeAcc_nontrivial`: changed; before and after print alike at the depth shown.
- **Every record that moved.** The heads are the same as run 2's in both packages [ran: `diff` of the two logs' heads]; only `scan_honest`'s new form differs.
  - Mine, `security/Properties`: the three guarantees above; definition `Properties.NetworkScan.ScanScopeAcc` changed; `Properties.NetworkScan.ScopeMatches` gone; `Claims.SamplingGeneric.Compiled.scopeCheck` and `Claims.ScopeMatches` gone from this lock's reads (no guarantee here reads them now; I did not touch their files).
  - Not mine, `security/Properties`: `rowsCapPublished_nontrivial` (a field `entries := 0` in a printed record); 8 definitions of `NetworkCheck.Period`; `Properties.NetworkPeriod.cappedPeriod`; `Verity.Idle.vocab`; `Verity.Vocab` and 17 of its fields.
  - Not mine, `security`: the same 8 of `NetworkCheck.Period`; `Verity.Idle.vocab`; 3 of `Verity.Sampled.Instance`; `Verity.Vocab` and 20 of its fields.
- **The regenerated lock against mine** [ran: compared the sections]: `open` 119 to 118, `ScopeMatches` out and nothing in; `domain` identical, my entry and digest unchanged; `scanScopeAcc`'s premises `domain: [Compiled.Scope]`, `open: [StmtFlattens]`. I did not commit it, on your word.

## 2. The replay: the line I proposed is the wrong one [read]
- **A bare `check.py` reads no replay cache.** Its keys are computed only under `--replay-cache` (`tools/verity/lean/check.py:5062`). Only the check's own audit job passes that flag, with the node's `lean-replay/` beside its passes (`lean_audit.py:1681`). So "does the pool's replay cache hold `security/Proofs`" has no bearing on the run you approved: it would replay every module whatever the cache holds.
- **How many.** `check.py security/Properties` audits `security/Proofs` in the same run, since it proves the spec's guarantees (`check.py:3973-3975`). That is 107 modules and 4,051 [ran: run 3's audit lines].
- **What it needs.** A spec beside an ArkLib package is refused under 160 GiB (`check.py:219-229`, `:331-355`). Its shape is the `lean-audit` kind: `cpu-xl`, 180 minutes of wall, and the kind's own comment says about 260 GB and about 150 minutes cold (`infra/cluster/kinds/lean.toml:1-7`). The README's line I quoted, `--mem-gb 32 --max-min 180`, is for a package that is not such a spec. My mistake: I quoted it without reading the floor.
- **With the cache, the replay is 6 modules** [ran: reverse import closure over `security/` and `verity/`]: `Properties.NetworkScan.Assumptions`, `Properties.NetworkScan`, `Proofs.Properties.NetworkScan`, and its `Causal`, `Fits`, `Walk`, which import the statements. A module's key folds its imports' keys (`check.py:4935-4965`), so those and nothing else are new. That holds only on a node whose cache has the rest at `main`'s head.

**Three ways, for you to pick. I launch none without your word.**
1. **The candidate's own audit is the replay.** A candidate that changes a lock or Lean first runs `lean-audit` alone, with the replay cache (the team skill's Landing). It replays the 6 modules and `main` does not move without it. Cost: none. But it comes after the mark, and you ruled the replay before the mark.
2. **The full check as a `lean-audit` job**, the line `check.py` itself prints, with `--update` and `--cwd clone`: `research run --on auto --kind lean-audit --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/check.py --build --update security/Properties`. Cost: the kind's, about 150 minutes on the large shape, 4,158 modules replayed to learn about 6. I have not run this form, and I did not read how `--on auto` places a kind.
3. **`Replay.lean` on the 6 modules alone**, through `research run`, in a built tree: `lake env lean --run tools/verity/lean/Replay.lean OUT` and the 6 names, the program the check runs for a module's own step (`check.py:5081-5085`, `replay_in_steps` at `:5176`). The set is closed upward, as `Replay.lean` asks. Cost: minutes. It is a kernel replay of step A's files and what imports them, with everything else imported unchecked. It is not a full check's verdict, and the pull request would say "replayed by `Replay.lean` on its six modules, not by a full check". Not tried: whether one `lake env` in `security/Proofs` serves the two modules of `security/Properties`.

I would take way 3 before the mark and let way 1 stand as the gate. Not checked: the node's memory as `check.py` reads it, and what a cold replay of `security/Proofs` costs today beyond the kind's comment.

## 3. Also done
- **The counterexample compiled** [ran: `r20261113-013725-c79a`, `r20261113-013725-5e7e`, both exit 0, axiom probe clean, tree `9f2648d483` of the scratch branch]. At the deployed two-stage parser, for every registration with a unit, `StmtFlattens` forces the registered descriptor's first gate to be no input and to read nothing; the kernel evaluated the three closed pieces. The whole of it comes in the Fri 13 Nov 2:00 PM PST report, which I send tonight.

