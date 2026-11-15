[2026-11-15T07:46Z] 205ea013847d
ROWSPROTO2, TOP-424 tie mend PUSHED (11:47 PM PST). Branch cursor/na-prove-path-tie-b51f, one commit b8ffef108f1 on architecture 009698aa9c0a. Nothing pushed to architecture or collect.
Cause confirmed. The subprocess's sys.path named verity/protocols/proofs/spot/verifier/tests, which holds none of test_pearltie, test_remotetie, window_tie or settings.py (none of them is anywhere under it). All four live in benchmarks/prove_path/tests (HERE). 00dac86f8bf moved the two tie tests there, and 2bac3457712 moved window_tie; neither updated the path. records_audit is in HERE.parent, which stays. Nothing else is needed from spot/verifier/tests, so I dropped that entry.
benchmarks/prove_path/tests/test_records_audit.py:718
before: sys.path[:0] = [{str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "tests")!r}, {str(HERE.parent)!r}]
after:  sys.path[:0] = [{str(HERE)!r}, {str(HERE.parent)!r}]
Ran on this VM (CPU, no Lean needed), ci-like path on the worktree, fixtures fetched:
- the test (-k): 1 passed in 25 s; without the mend, ModuleNotFoundError 'test_pearltie', as in ci's run.
- the whole file: 52 passed.
- py_compile ok; ruff 178 -> 177 (one long line fewer).
- tests/test_repository.py: 17 passed.
Read guard: test_records_audit.py is in the suite's `untraced` list (keyed on the suite's whole tree), and the mend points only at benchmarks/prove_path/tests and benchmarks/prove_path, both inside the suite. Fine.

[2026-11-15T07:54Z] d0d17505cb83
DEVICEGONE to network-accounting (lead): the new head is 70863c040628 on cursor/na-lock-hand-lines-b51f (parent 2baf913ea3e6), pushed only there, Sat 14 Nov 11:54 PM PST.

The two added lines, in security/lean-audit.json `shape.frozen`, right after the EgressRegistered line. The form takes no owner: a frozen entry is a string.
   "Accounting.Guarantees.B4UnderId": "unstated: stated in the spec and proved by Accounting.SecurityProofs.B4UnderId in Proofs.Accounting.Actual.LinkEnds, listed as a guarantee after the landing, NET-450, Wed 18 Nov",
   "Accounting.Guarantees.B4Via": "unstated: stated in the spec, no theorem proves it yet; Proofs.Accounting.Actual.LinkEnds derives it from B4 only at a Site whose every place is its own link end (b4Via_of_ownLinkEnds), NET-450, Wed 18 Nov",

What proves each [read security/Proofs/Accounting/Actual/LinkEnds.lean; the file and Definitions/Accounting/Guarantees/LinkEnds.lean are the same at 009698aa9c0a as on my base]:
- B4UnderId is proved by `theorem B4UnderId : Guarantees.B4UnderId.{u, v, w}` (line 50), both directions of the iff, with no hypothesis. Its module is reached: Proofs/Accounting/Actual.lean:15 imports it.
- B4Via is not proved. The definition file says so itself (line 9: "`B4Via` is not proved here"). The nearest result is `b4Via_of_ownLinkEnds` (line 60): from `Guarantees.B4`, at a Site T with `∀ p, T.linkPlace p = p` and `T.RatedOn β Set.univ`, it gives `B4Under T.linkPlace T.placeEnds β`. That is B4Via's body at such a Site only, under B4.

Not ours [read]: 09c3b46379e ("SPOT's caller's checks move out of SPOT's check into spot_flock's Caller") came in through cursor/proofs-caller-top415-caller-check-7ed4, the proofs lane, by the merges of origin/architecture into that branch. So CallerCheckMeans and CallerCheckSound are proofs', and I wrote nothing for them.

Rechecks [ran]:
- The lock parses, and check.py's load() takes all three locks. category() gives `unstated` for both reasons.
- merge.merge has 0 conflicts in all three locks against each branch, with both entries kept:
  - LOCKPASS 5cf7ba575ea3;
  - lean's branch at 610b864973ed, and at its newer head 9477aac45347 (it moved after your message);
  - TERMGB2 f930518dcd8d.
- tests/test_lean_packages.py and tests/test_repository.py: 51 passed.

Least sure: whether the shape check, given facts, takes the B4UnderId entry as `unstated`, or wants B4UnderId listed now because a theorem states it (shape.unstated-category only checks the prefix). No build ran.
Not looked at: whether B4 itself, which b4Via_of_ownLinkEnds takes, is proved.
TIMER: none

