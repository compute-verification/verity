[2026-11-17T04:24Z] 6bdbc4ad8335
MODEL: Claude Opus 5.5
DRAWMAP 8:25 PM PST: the cause of the two served_zk failures. Each is a rename made on one side of a pair whose other side is a Lean text the map holds.

1. test_the_cli_runs_the_window_through_the_service (select replied ERROR, not SAMPLED). Proofs' Lean `verity-verify-proof draw` prints its sample as a line tagged "DRAW {...}" (FlockVerify.lean:1709, held). The phase-1 map renamed the tag in served_commit.py's lean_sampler to `startswith("SAMPLE ")`, and in the test's stand-in Lean to print 'SAMPLE '. The reader still slices x[5:], the length of "DRAW ", so it parsed "LE {...}" and the select failed. Against the real Lean binary it would match nothing at all.
2. test_check_opens_the_sampled_proof_blocks_rows_only. The test's `"not drawn" in SC.check(...)` became "not sampled". The producer, core_check.py:860 "unit {u} is not drawn", is held because Lean's Opening.lean:155 states the same message.

The fix keeps both on both sides: a quoted "DRAW " and a quoted "not drawn" are held. The regenerated map differs from 1515f0fbb62's by exactly those two keep patterns and changes 3 files. It is pushed as e1acc87d653 on cursor/pouw-draw-map-1de3.

Reproduction on a node, one named file: r20261117-042103-156f runs pytest tests/benchmarks/pouw/test_served_zk.py at e1acc87d653. I'll send its end, then update handins/draw-map.{toml,md}.

The Lean map has no such pair: "is not drawn" is held there on both sides because other lanes' Lean spells it, and the line tag is not in its scope.

The Lean build relaunch: I'm waiting for ci's batch to free the slot (stages by 8:45 PM PST). I'll read queue status and the slot first, then launch with a 45-minute cap.

