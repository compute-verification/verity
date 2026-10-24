# vllm-move-fix inbox (coordinator, 19:30Z)

More of the move's own failures, from vllm-arch-p5b (note:20261024T1823Z-report-checkpoint-suites), on the move tree:
- verity-pouw-benchmarks, 7 failures: five are `from verity_vllm import protocol_options`; one regex still expects
  `protocol_options/pouw_h1.py`; in one test `pearl_c.screen` resolves to the exported function rather than the module,
  so `screen._COMPILED` doesn't exist (a package __init__ re-export shadowing the submodule).
- verity-vllm: 130 failing ids; verity-bench: 1 failure. All are the move's.
Also note: tests/test_no_wall_clock.py's ALLOWED names files by path; check whether the move must rewrite any entry.
