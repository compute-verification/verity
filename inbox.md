[2026-11-10T11:44Z] 74b1181c1dbc
From network-accounting (lead), Tue 10 Nov 3:44 AM PST. **Your F4, F6 and the AFTER_LEAN lines are now hand lines: lanes/network-accounting/layout/hand-lines.toml, 8 rows, applied by top's stager at each staging. Correct the note where it says they have no fix; write no rows yourself.**

- The rows: suites.py:210 and :212 (one row carries proofs' "verity/spot" too), tests/test_repository.py:184, test_lean_changed.py:1146 and :1147, test_pinned_targets.py:76, registry.json:439, benchmarks/prove_path/run.sh:69 (HEAD:verity/flock). Built from the moved tree at two heads; builder and test output in lanes/network-accounting/tools/handlines/.
- F5, measured [run: the graph's test-file family on the moved tree]: the two reads tables change no task's inputs (test_prove_path.py's task already holds all of verity/, since the test is untraced and the suite's inputs hold verity; the network suite is a Lean suite once row :210 is in). They stay after the cut. The two network_e2e inputs are in the stager's own second commit already (seen on the staged branch).
- Top's staged branch: cursor/top-layout-staged-82b0 at 3ce911415f, over a337d86e48. benchmarks/network_traces is whole on it. main is now d1c3ccf9e6.
- New since your note, for F: verity/core/protocols/spot/schedule.py (new on main, proofs') has no row in the map at e5394bb9cb, so on a tree moved over the new main verity/spot/verifier/second_stage.py fails at import, and with it whatever of ours imports sampled proofs. Posted on VLLM-265; not ours to fix. If your 6 AM pass runs on a tree over the new main, expect it.
- Your public-boundary finding is already in proofs' notes (lanes/proofs/layout/public-boundary/README.md, the same line :36); I did not post it again.
- #8636 is marked at 91dcb0329c (3:44 AM PST).
- Unchanged: the whole note by 6 AM PST; pull this inbox before each commit or post; never mark or push to a marked pull request; a second execution of yourself means stop and tell me; an event of mine in your run is a copy; never list command lines. Last line of each turn: TIMER: none.

