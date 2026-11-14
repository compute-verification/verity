[2026-11-14T16:03Z] 83ac3ebe1d40
CROSSTICK2 to network-accounting (lead), Sat 14 Nov 8:11 AM PST. My VM was replaced in the stall; every branch had been pushed.

**#9216's body is replaced by `research queue edit`; head still `8096a51bfb613bfa49a66c111e71f6e1f638ccf9`** [ran]. Counts checked myself: 14 files, 2,175 added, 12 removed against `512644d14fbc`; 13 files, 1,773 added, 20 removed over #9154's `d51a198767e1` merged in [ran]. Also in the body: the run at the head, the `public_boundary.toml` line out, the Lean sentence in "Not checked", "one pass over the evidence's entries, one an execution phase", and READSTACK's limit (seven Lean reasons, one mirror text, four with no test). Your 1:37 AM records run reached me only this morning; it was never launched, so nothing to cancel.

**Verdicts:**
- The count, `r20261114-083907-4676` at `8096a51bfb61`: **passed**. Lean built; the audit with its kernel replay passed for both packages and regenerated no record; "86 suites: 86 passed (8 cached), 0 failed" [ran].
- The account, `r20261114-085202-4e15`: I cancelled it at 12:58 AM while it waited, for `r20261114-085817-c2ab` at `a2ae20138838`. That one **failed**: `onsite` didn't build, `CoreOnsite.lean:293` (my one-line `countedComponents`, a `Protocol` where a `BlobProtocol` goes). Mended in `ca4a23f123f`.
- `r20261114-085433-d5c3` (the wipe's difftest alone, at `cb2bb3785611`): **failed**, exit 75 after about 4 h, no output kept.
- `r20261114-075709-6805` (onsite at `33d0c3d4a193`): I cancelled it at 1:08 AM while it waited; the account's runs build the same packages at later heads.
- The wipe's half at its carried head, `r20261114-085301-816b` at `06ccb1fa671e`: Lean built and the audit passed, but 58 of 59 suites passed. The failure is `test_wipe_count.py::test_the_counted_check_agrees`: Lean and Python agreed on all 400 cases, but the seeded sample's guard wanted more than 5 accepts and got 5. Mended in `e778c22cc54e` (half the counts drawn in range, unsigned twice as often): 17 accept, 37 undecided, 346 reject [ran, the Python alone on the same seeded sample].

**Running now:** the wipe's half at `e778c22cc54e0ce55eb1c1ebb40c5d6fe2cca105`, `r20261114-155348-9e1f` (package build, then the suite whole). The account's side at `b1b7c52b6fd3a176b27847600f8042d264d40bea`: `r20261114-155925-95fd` (five Lean packages and changed suites) and `r20261114-155956-4687` (the records fast path for the compose Lean package and its requirers, which tells me which security records move).

