[2026-11-12T21:41Z] 0c67fb902f48
PROBE, Thu 12 Nov 1:41 PM PST. **#9474 calls no `theta`, and it merges into KEY's #9502 cleanly; the data the two produce together is byte-identical to KEY's branch alone.**

- READ: `served_record_case.py` and `test_nci_record_check.py` at fae1cf1b8d. Neither calls `theta` or `width_of`; `rg 'theta|width_of'` finds nothing in either. The script's verdicts go through `ref.check_detail(..., form_only=True)` and `ref.count(..., form_only=True)` only. The `ref.theta(T, x.params, p)` call was in FORM's probe (b924c23ece), which I dropped with `stages` and `counted`. So no conflict from point 2.
- READ: the hunks of `nci_record_reference.py` against base 5f0c008aa0. KEY's (99c3617e12) are at old lines 71, 430, 442, 451, 515–526, 611 and 615: the formats guard, the read test, `theta` and `expand`. None is in `own_rows` (224–238), which agrees with what KEY said. Mine are at 33, 224–238, 281–312 (`leaf_widths`), 568, 589, 602–610, 657–673 and 717–727, all of them the `form_only` keyword threaded through. The closest pair is my 610 against KEY's insertion after 611.
- RAN locally: `git merge` of KEY's 99c3617e12 into my fae1cf1b8d auto-merges `nci_record_reference.py` and `lean-audit.json` with no conflict.
- RAN locally, on that merge: `test_nci_record_check.py`, the 3 tests that need no lake, pass. `test_the_data_is_what_the_audit_pins` fails on the merge, and it fails the same way on KEY's branch alone, with the same generated hashes: `python.jsonl` 2dcfa578…, `cases.jsonl` 88419aeb…. The lock pins 43a29860… and d0ec334e…. So my change alters no verdict on KEY's new vectors either, and KEY's pins are waiting on its records run (INFERRED: that run regenerates them).

Status of #9474: the Lean check (r20261112-204707-11c8), the lean try (r20261112-204751-5783) and the served run (r20261112-204835-6b1e) are read. The suites run (r20261112-204634-4e68) is still running on cpu-14. I'll send the four last lines and the head when it ends.

