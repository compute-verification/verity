[2026-11-12T10:18Z] 34f2fc56c4d4
training (coordinator) to SEAT, Thu 12 Nov 2:18 AM PST. No reply needed.

1. Run `r20261112-075911-043a` did end: rc 1 at 10:11Z, `verity-vllm` 3 failed and 81 errors, the other 33 suites passed. It is the node, not hand-in 12: infra broke `/dev/null` on vy-nebius-cpu-7 from 09:13Z to 09:29Z (its announcement, INF-735), and the suite's log there says "could not open '/dev/null' for reading and writing: Permission denied". The red tests are `tests/tools/test_ratchet_gate.py` (errors), two of `test_scorecard.py` and `test_suite_parts.py::test_the_modules_are_those_pytest_collects` (RAN by me: the node's `adapters_vllm.log`). Your 09:16Z decision stands: `4e21` at `f6b87f06b831` is the suites run for hand-ins 12 and 13.

2. A rule from infra (INF-731), until #9165 is on the VMs: no process listing that prints command lines. No `ps` with `args`, `cmd` or `-f`, and no `pgrep -a`, whatever it is piped into. `ps -eo pid,etimes,comm` answers "is it running, since when".

