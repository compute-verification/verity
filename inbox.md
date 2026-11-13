[2026-11-13T01:03Z] 86d8a0c38c9d
KEY, Thu 12 Nov, 5:03 PM PST.

**A slip in my last message, found at 5:01 PM: the suite I ran at 9ddabfeafb is not the one that holds the new test, so the right two are launched now; and the security proofs package's check passed.**

The slip
- RAN r20261112-234514-8867 ended `1 suites: 1 passed (0 cached), 0 failed`, but it ran `verity/protocols/certifier/verifier`. READ: `test_nci_record_check.py` is in `verity/protocols/certifier/verifier/network`, a suite of its own under that directory (its own `pyproject.toml`). I named the parent. So that run says nothing of the option.
- How I found it: RAN r20261113-005740-a3ea and r20261113-005919-ba30 read that suite's logs on the nodes. It collected 1447 tests at 9ddabfeafb and 1447 at ea620a89cf, where one more was due.
- RAN r20261113-010201-8cc1, launched 01:02Z at 9ddabfeafb: `suites.py verity/protocols/certifier/verifier/network repository --changed ea620a89cf --quick`. The network suite holds the test file, the only file that loads the script. `repository` scans the whole tree, so it reads both changed files. It waits for a check slot.
- Unchanged: the test passes locally at 9ddabfeafb (5 passed in `test_nci_record_check.py`), and the network suite passed at ea620a89cf in r20261112-232245-e582 (90 passed).
- The PR body says this, and names r20261112-234514-8867 as the wrong suite.

The security proofs package's check, at ea620a89cf
- RAN r20261112-232445-9cb4: `AUDIT: PASS; reports in …/lean-audit`. Its first line: `security/Proofs: PASS  198512 declarations in 3967 modules; axioms propext, Classical.choice, Quot.sound`, 5,780 s in all.
- INFERRED: it stands at 9ddabfeafb, whose diff from ea620a89cf is the script and its test file alone.

So at this hour every check you asked for has passed but the two suites at the new head: the five groups, the three tries, the nci package's check, the proofs package's check, the served twelve, and the ratchet at exactly #9474's 3 sites.

In flight: r20261113-010201-8cc1 alone. I send its last line when it ends.

