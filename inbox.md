[2026-11-14T06:17Z] 57b7a9ac8ae2

READCALC, Fri 13 Nov, 10:16 PM PST.

**#9889 is NOT READY on one must-fix: the test counts as "listed" every definition any module of a lock reads, where its docstring and the body say only an assumptions module's, so a citation `rho.lean` calls absent passes it.**

a6494c1f6a1c2884d11fb19d6ed70426df4b2809 NOT READY, 1 must-fix.

What I read: `benchmarks/rho/tests/test_lean_names.py` whole (the only file), `rho/lean.py` lines 19-38, `rho/min_cut.py` lines 60-72, the rho suite's `pyproject.toml` inputs, `training_slowdown/calculator.py` lines 134-160. Sampled: the body's harness table (rows 1 and 2 checked, the rest not opened). Not opened: `classes.py` and `envelopes.py` beyond the cited lines.

## MUST FIX

1. `benchmarks/rho/tests/test_lean_names.py:39` (`listed()`): `out |= {d for m, g in (lock.get("reads") or {}).items() for d in ...}` takes the definitions of every module the lock records, about 35,000 names over 2,009 modules. The docstring (lines 5-7, "a definition an assumptions module holds") and the body ("`rho.lean`'s reading: ... a definition read by an assumptions module") say only an assumptions module's, which is what `rho/lean.py:35-37` reads.
   - RAN on my VM: append `MUT = "Nci.Policy"` to `rho/classes.py`. The test passes (3 passed), and `rho.lean.status("Nci.Policy")` returns `absent`.
   - Fix (RAN on my VM: at the head 3 passed, and the same mutation now fails `test_every_lean_name_the_calculator_cites_is_in_a_lock`):
     ```
     out |= {d for m, g in (lock.get("reads") or {}).items() if "Assumptions" in m.split(".")
             for d in (g.get("definitions") or {})}
     ```
     Or take `rho.lean`'s own sets instead of a second reading of the locks, plus the guarantee keys `nci/arguments.py` reads.
   - It changes no figure today. RAN: all 34 listed names get `proved` or `assumption` from `rho.lean.status`.

## SHOULD FIX

1. `test_lean_names.py:23` (`NAME`): the last part must be capitalized, but 648 of the locks' guarantee keys end in a lowercase part (137 in `security/lean-audit.json`, 511 in `security/Properties/lean-audit.json`; RAN, a scan of the keys).
   - RAN: `MUT = "Nci.Guarantees.noSuchBound"` in `rho/classes.py` passes the test.
   - Fix: `NAME = re.compile(r"(?:[A-Z][A-Za-z0-9_']*\.)+[A-Za-z_][A-Za-z0-9_']*")`. RAN: at the head it still finds 35 names and passes 3 tests, and the mutation fails.
2. Two other ways a citation escapes the test (RAN, each passes 3 tests): a name built inside a function (`def _mut(): return f"{AREA}.Guarantees.NoSuchBound"` in `min_cut.py`), and a name inside prose (`"rests on Nci.Guarantees.NoSuchBound"`). The docstring's definition of a citation already rules these out, but its first line ("Every Lean name the calculator cites is one a lock lists") reads wider.
   - Fix: say so in the first line ("every string of the calculator that is a Lean name"). Or also scan backticked names in strings. INFERRED, from a loose scan I RAN: that would flag `Accounting.Hardware` (`rho/hardware.py:1`, a Lean definition in `security/Definitions/Accounting/Physical.lean` that no lock lists), so it would need an entry or a fix.

## NOTE

1. `ALLOWED` "only shrinks" is enforced for stale entries only (`test_allowed_only_shrinks`, lines 93-96). Nothing can stop an addition, so the docstring and the failure message state that part as a rule. That is fine; the body says it right ("a stale entry fails").
2. The toy test (lines 99-108) exercises `cited()` only and never `listed()`, which is why must-fix 1 went unseen. A toy lock with a non-assumptions `reads` entry would cover it.

## For the lead to read

- `test_lean_names.py:31-40` (`listed()`) decides what "a lock lists" means. This is must-fix 1.
- `test_lean_names.py:54-68` (`cited()`) decides what a citation is: a whole string of capitalized dotted parts, under a root some lock name has, that is not a prefix of a listed name.
- `test_lean_names.py:71-82` (`calculator_cites()`) decides where citations are looked for. It scans every `.py` under `benchmarks/rho/` except `tests/`, plus the built values of the top-level `rho` modules (module constants and strings, and classes' uppercase attributes, nested at most 3 deep).

## The body's claims

- 35 cited, 34 listed, `min_cut.HE_CHARGED` the one in `ALLOWED`: RAN on my VM (`calculator_cites()` and `listed()`). Same at the head merged onto `main` 8e018c9331f (local merge 11539f76dca): RAN, 3 passed, 35 cited, the same one missing.
- "It reads the two locks `rho.lean` reads and the suite declares": READ `rho/lean.py:19`, `benchmarks/rho/pyproject.toml` `inputs`. True.
- "`rho.lean`'s reading: ... a definition read by an assumptions module": FALSE of the test's code. This is must-fix 1.
- `HE_CHARGED` is composed from the registry's row: READ `min_cut.py:60-69`. True.
- The 42 tracked locks: RAN `git ls-files '*lean-audit.json'`, 42. `HECharged` and its `_honest` and `_nontrivial` companions are only under `shape.frozen` in `security/lean-audit.json`, with no guarantee: RAN a JSON scan at the head and at `origin/main`. True.
- When `cursor/hecharged-from-count-a8e2` lands, `test_allowed_only_shrinks` fails: RAN. Merged locally with #9519's head d198a01da9e (local commit 69fe071f65), it fails with "ALLOWED names a lock now lists ...: ['Nci.Cut.Guarantees.HECharged']". True.
- #9877 changes nothing in that: RAN, #9877's head merged with #9889's (local 54a8f51226). The test passes, 35 cited, the same one missing, and #9877's diff has no line naming `HE_CHARGED`. Whichever of #9889 and #9519 lands second deletes the `ALLOWED` entry, whether #9877 lands or not.
- The harness table, sampled: READ `calculator.py:134-137`, where `theorem()` returns `HE` while its docstring says "`HEShared` where the products share arrays". True. READ `min_cut.py:198`, where `statement` is `HE` at F = 1 and else `HE_SHARED`. True. READ: `RESTS_ON`, `RESTS_ON_CELL`, `he_rests_on`, `SHARED_LABEL` and the β_eff text exist in `formalism.py`. The other rows I did not check.
- Suites `r20261114-015642-3bbb`: READ (`research fetch --all`, `stdout.log`). `tree_sha` is a6494c1f6a1c2884d11fb19d6ed70426df4b2809, the head. Last line: "5 suites: 5 passed (0 cached), 0 failed; 29.1 min of test time". rho: 615 passed in 30 s.
- ruff: RAN on my VM, "All checks passed!". The ratchet gate: not rerun.

## Tree rules, paths, time

- Paths are today's layout: `benchmarks/rho/` until Monday's move.
- The new file holds no run id, PR number, date, name or measured number, and no Lean name as a literal.
- Time added: 3 tests, 0.4-1.1 s on my VM (RAN). The rho suite is 615 tests in 30 s in the author's run.

Runs I launched: none (every check above was a single test file on my VM, or a local `git merge-tree`, nothing pushed).

The whole report is note:20261114T0616Z-report-readcalc-9889 (synced).

