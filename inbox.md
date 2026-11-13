[2026-11-13T04:48Z] 8c0c126fe8d3
FIELDS, Thu 20:48 PST

**All three runs ended rc 0. At `fe31b503bb` the records hold with nothing regenerated and all 88 suites pass. #9565 now waits only on #9502; #9474 and #9502 are both still open.**

- RAN r20261113-021149-517d (rc 0, at `66a3543f46`): records for `verity/protocols/spot/verifier/lean` and its requirers. Last lines: `security/Properties: check.py exit 0 in 623 s ... (0 files changed there)` and `OUTCOME {"outcome": "pass", "package": "verity/protocols/spot/verifier/lean", ...}`. No record was kept.
- RAN r20261113-021149-636b: cancelled 3 minutes in, for the merge. It has no suite result.
- RAN r20261113-021552-c5aa (rc 0, at `fe31b503bb`): records for `verity/circuits/lean` and its requirers. Every package passes, `security/Proofs` included, so the merged proofs build. `OUTCOME pass`; no record was kept, so the committed `security/Properties` record holds on the merged tree.
- RAN r20261113-021552-e198 (rc 0, at `fe31b503bb`): `suites.py --changed 49d12bdc36 --quick`. Last line: `88 suites: 88 passed (0 cached), 0 failed; 251.3 min of test time`. The network suite (85 passed) runs `test_nci_record_difftest.py` whole, the 671-case data pin included.
- RAN e198, on `verity-vllm`: it passed with 14363 tests passed and 401 skipped, in 3577 s. 9eba's one-hour limit cut it at 55 minutes because the suite is that long, not because a test hung; e198's three-hour limit let it finish.
- #9565's body has these runs. Next: once #9502 is on `main`, I merge `main` once, combine the cases, regenerate the data, the `runs` entry and the records once, run the touched suites, and send you the head and the run ids.

