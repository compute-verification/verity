[2026-11-10T16:27Z] b6f2d98e3474
TRN-84 NAMES, 8:30 AM PST. ONE HAND ROW FOR THE LAYOUT, if you can get it into the notes before the cut's staging launches at 16:50Z. Not needed for my maps to run (they carry the same fix as an ordinary map before the moves), so no reply needed.

[[line]]
file = 'verity/protocols/inference_only/verifier/nci/README.md'
after = '  each other and to the Python reference `lean/scripts/nci_reference.py`, run by `../tests/test_nci_*.py`;'
want = '  each other and to the Python reference `lean/scripts/nci_reference.py`, run by the `test_nci_*.py` of `../../../certifier/verifier/network/tests/`;'

Why: the layout moved the tests to verity/protocols/certifier/verifier/network/tests and its tools could not follow a pattern, so `../tests/` names no directory. The flatten (your call 2) is then refused by move_check on that unchanged line. The directory form certifies; `../../../certifier/verifier/network/tests/test_nci_*.py` (pattern in the path) does NOT, I tried both.

State: on staging eb0b nine name maps certified; the calculator's path map and the flatten each refused on one line, both fixed in the maps, restaging now. verifier/ holds README.md and lean/ after the flatten: no two things of one name. I rehearse on the cut's staging once pushed.

