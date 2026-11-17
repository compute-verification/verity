[2026-11-17T17:38Z] 45b45a9b5c2f
Lead, Tue 17 Nov 9:42 AM PST: another worker (NOFLAKE) is deleting the timed end-to-end tests from tests/benchmarks/attester_load/test_attester_load.py (the  fixture and the tests that take it) and may factor a small pure function out of run.py's main for writing measurements. Do not add or change tests that run the stub arm for real; if your change needs a test of metrics/members, test pure functions on a fixed document. Expect a small merge in run.py and that test file once its PR lands.

