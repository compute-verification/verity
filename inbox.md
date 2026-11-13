[2026-11-13T06:13Z] 6b50f0da98f1
QUERY, Thu 12 Nov, 10:13 PM PST.

**#9374 is merged with `main` `99cc585fa8f2` at VERSION 4.26.0, head `7f3b0b2cbd55`, and all five checks pass there with every library entry unmoved; it is still a draft.**

Head `7f3b0b2cbd550e80077309d7e913a1afb9cd9bcb`, pushed with no force (RAN). There are two commits: the merge `f6ce25b830`, then `7f3b0b2cbd`, VERSION 4.26.0 in `__init__.VERSION`, `products.CAPTURE`, the grader's `CAPTURE` and test_products' pin. `PRICED_AS_TRACED_FROM` stays 4.23.0, and the README names no version (READ). Title and body are updated with `research queue edit` (794 words; the Lane, Refs and Goal lines kept) (RAN).

Step 1:
- The conflict in `library.py`: `PARTS["premise"]`. The other side is #9272's batch premise (`work.P_BATCH`, `f6a5e89ffb`), not #9039's. Both texts stand: the batch premise first, then the attention premise. `test_library` asserts each (READ: tests/test_library.py; RAN: passes in the suites run).
- After `ca6c4c2a5483`, #9039 added only two merges of `main` (`3132133f30`, `f5a4a2b051`) (READ: git log). Their own resolutions are formalism's imports (`op_key`, `P_BATCH`) and `run.py`'s record (`tf32` beside `threads`). Neither meets this change. `gates.py` and `grade.py` take no resolution edit, and in `grade.py` and `work.py` the auto-merge keeps my lines beside `main`'s unchanged (READ: git show --cc, git diff origin/main). Nothing on `main` since `99cc585fa8f2` touches `benchmarks/training_slowdown` (READ: git diff to `b843a10af4`).

The five checks, at `7f3b0b2cbd55`:
- `r20261113-054016-4430` (RAN): grader 31 and training_slowdown 660 passed, fresh; ruff clean; ratchet gate `5d132e70ed76..7f3b0b2cbd55: 1 change(s), 0 failed`. Last line: `suites rc 0 ruff rc 0 ratchet rc 0`.
- `r20261113-054052-a4e1`, unmarked (RAN): 22 failed plus the grader's 2. The names equal those of `r20261112-223533-da90`, compared line for line. Passed goes from 95 to 99 because of `main`'s four new tests in those files (#9538's three `SOURCES` cases and #9482's thread-count test) (READ: git diff). Last line: `mutation unmarked: training_slowdown rc 1 grader rc 1`.
- `r20261113-054121-4175`, widened (RAN): 13 failed, the same names as `r20261112-223559-d294`; passed goes from 104 to 108 for the same reason. Last line: `mutation widened: training_slowdown rc 1 grader rc 0`.
- `r20261113-054146-9821`, the library on 46 entries (RAN): all exit 0. 37 are traced: none refused, every call marked bfloat16, and no attention row in M. Last line: `lib exit 1`. That counts the nine entries that need a record: the eight as before, plus `main`'s new `l16-t20-1.1x-shallow-depthshare-4`.
- `r20261113-054202-8dfe`, xl (RAN): traced clean, with 139 attention rows, all bfloat16. Last line: `lib exit 0`.
- On the 43 entries and xl, every `library.json` equals the one at `8e09000bffbb` (`r20261112-223626-da4b`, `r20261112-223637-7d01`) but for `step.capture` and `step.capture_code` (4.25.0 to 4.26.0). No entry moved, so no #9039 commit is named (RAN: compare of every field).
- Beyond the brief, `r20261113-054240-49b6` (RAN) runs `main`'s three new l16 entries at `99cc585fa8f2`. The two traced ones (`-102`, `-40326`) fall 1.948 in |C| here, inside l16's 1.91 to 1.95, and what binds is unchanged. Not run: `main`'s two new xl entries (`-3884`, `shallow-depthshare-4`).

Handoff: `20261113T0613Z-handoff-QUERY` (synced).

