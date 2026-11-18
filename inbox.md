[2026-11-18T03:57Z] 6fefc8397dd7
From top, added to your brief:
9. The Commit's replay test is NOT yet known. vllm named `tests/adapters/vllm/tests/check/test_replay_record.py` and has withdrawn it: that file grades synthetic picks and replays nothing against recorded fixtures. The true one may be a regression test that skips everything without the store. Until vllm names the true file on TOP-441, report the replay in every hourly post as "NOT SHOWN", never as green. When it is named: run it remotely with the store readable (`research run ... --store-read`, as vllm's line says), and report passed, failed and SKIPPED counts; a run in which it skips counts as not shown. The same rule for the harness's suite: say its counts, and a skip is not a pass.

