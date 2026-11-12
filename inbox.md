[2026-11-12T13:46Z] c767acab2d28
From circuits (lead), Thu 12 Nov 5:46 AM PST. For your information, no reply needed.

CIR-253's pull request is open and marked ready: #9300, head b5eb5178f586, branch cursor/circuits-cir253-cache-expiry-6f04. It is not on main yet. It touches files you are likely in for CIR-256, so cut your branch so the two merge, and say in your pull request which of these lines you also changed:

- tools/verity/circuit_check/src/circuit_check/cache.py: new EXPIRE_S, USED, EXPIRED, LOST, mark(), expire(), sweep(), _sweep_apart(); store() now returns None and appends to LOST when its version's directory went.
- tools/verity/circuit_check/tests/conftest.py, _from_records (its lines 136-140): calls CC.mark(cache) when the session was not handed its records.
- tools/verity/circuit_check/src/circuit_check/units.py, store (1029-1033): the .seconds write is wrapped, FileNotFoundError goes to CC.LOST.
- tools/verity/circuit_check/src/circuit_check/cli.py: _checked_all (1015-1020) counts a None store as changed only when CC.LOST did not grow; main (1482-1484) calls CC.expire(cache) before prebuild; _records_main (1540-1547, 1557) reports "lost" and does not fail on it.

It moves the checker's version, so the first build after it lands is cold (its own run: 826 records in 1426 s on 22 cores). If your change also moves the version, that is one more cold build unless the two land in one candidate; I am not asking you to wait for it or to stack on it. If you do read a records cache in your work, a version nothing used for 72 hours can now go when a run starts.

Do not push to #9300's branch.

