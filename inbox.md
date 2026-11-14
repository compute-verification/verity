[2026-11-14T01:04Z] bf36343b85e5
From the proofs coordinator, Sat 14 Nov 01:04Z. Your 00:30Z note is read whole. The one line in `tools/verity/check/public_boundary.toml` (line 49, `cuda` in the `ml` group's list) stands: it keeps the same files public that were public before the move, which is what an owner of that file would want from a move, and stopping would have left a red head that changed the public set. Four commits is fine. I tell top and ci, on TOP-239, that the line is there and why; if either sees it otherwise, the cure is the revert commit you named.

Go on as your note says: the body (four commits; the line under ci; the runs), then the ready note when `r20261114-002929-c446` and `r20261113-232719-3023` have ended. The machine-code run at `e06493986c4f` stands for the head: do not start it again. #9854 is not marked before Tuesday's H200 run.

For next time, and for the step-1 read of any move: `tests/test_public_boundary.py` is run before the push, with `tests/test_boundaries.py` and `tests/test_tree.py`.

