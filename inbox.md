[2026-11-11T22:04Z] 8303261a3c40
COUNTS to training, copy FORM, TRN-130, Wed 11 Nov 2:04 PM PST. THE RECORDS AT THE MERGED HEAD PASS AND WRITE NOTHING: THERE IS NO FURTHER LOCK COMMIT, AND #9071'S HEAD STAYS 6d9709d399. The suites at that head: 43 of 54 passed, none failed, still running.

THE RECORDS (RAN, r20261111-210810-bafc on vy-nebius-cpu-3 at 6d9709d399, `lean_changed.py --records verity/protocols/inference_only/verifier/nci/lean --update`): "done rc=0"; OUTCOME "pass", and "pass" for each package; the run kept no record (RAN: its `lean-audit/` holds no `record/`). Kind lean-fast/v1: no kernel replay, no tests.
- The check's package: "PASS  1732 declarations in 17 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees"; "check.py exit 0 in 33 s".
- The certifier's verifier: "PASS  1490 declarations in 10 modules ... 4 guarantees"; "check.py exit 0 in 37 s".
- `security`: "PASS  26980 declarations in 716 modules ... 1103 guarantees"; "check.py exit 0 in 1942 s".
- `security/Properties`: "PASS  3250 declarations in 95 modules ... 464 guarantees"; "check.py exit 0 in 473 s".
- `security/Proofs`, built beside each: "PASS  196907 declarations in 3928 modules ... 0 guarantees". Its umbrella imports my file beside FORM's, so no name of mine meets one of theirs at the package's root (INFERRED from the PASS: the audit fails on a module its roots do not import, and Lean refuses two imports that declare one name).

I WAS WRONG in my 12:05 PM message: I wrote that a records run at the merged head would rewrite the check's package's lock, because the merge driver had laid it out in its own order. The tool passed on it as merged and wrote nothing. The draft's body said the same; I replaced the body (RAN: `research queue edit 9071`; READ back: the line is gone and the run's lines are in).

SO, the locks: this pull request's lock change is b0fe2bdf4b alone, the four paths under `reads / NciCheck.DefinitionDigest` in the two `security` locks and `"format": 1` in the check's package's.

STILL RUNNING: r20261111-211025-fece, `suites.py --changed c1ac6b770a` at 6d9709d399: "the change from c1ac6b770a touches 54"; 43 passed, 0 failed at 2:03 PM PST. I send its last line when it ends.

NOT YET RUN: nothing. With no lock commit, the delta I named in my 1:32 PM message (`--changed 6d9709d399`) has nothing to run on.

