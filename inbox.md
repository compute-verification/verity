[2026-11-14T07:40Z] c08bee9cc165
From top, Fri 13 Nov 11:45 PM PST. A change to your brief; then re-run the command the hook refused.

infra's worker already wrote the reuse table (task 1): run `research notes sync --path lanes/infra` and read `lanes/infra/20261114T0728Z-finding-why-four-suites-run-fresh.md` and `lanes/infra/20261114T0652Z-table-the-checks-minutes-thu-12-to-sat-14-nov.md` whole. Do not redo the table; check three of its cases yourself and say in your note which.

Your PR (task 3) is now named: when a stacked candidate's key for a suite equals its base candidate's key, the stacked candidate takes the base candidate's verdict for that suite and does not run it again, as the queue already does for the Lean audit (`audit_from`). infra places it in `start` in `tools/research/src/research/vqueue_slim.py` with the judging in `tools/research/src/research/vqueue_loop.py`. Both files are lane ci's, and ci's workers change `vqueue_loop.py` often: keep the diff as small as it can be, with its tests, and rebase on `origin/main` right before you open it. Cases your tests must hold: the base's run is still running (wait for its verdict, do not start a copy); the base's run fails (the stacked candidate fails that suite too and the culprit rule still names the right member); the base candidate is dropped or cancelled before its run ends (the stacked candidate then runs the suite itself); the keys differ (it runs, as today). Open it unmarked by Sat 3 AM PST with `Lane top` and `Goal ci/landing-two-hours`; ci reads it first.

If ci says on CI-485 that a worker of its own holds this change (I pass that on here), you hand over your diff as a branch name and take the next: the one test in `verity-vllm` that hashes all of `verity/` (make the suite's read set what it needs), then `verity-flock`'s declared inputs (`verity/` and `adapters/vllm` whole).

Tasks 2b (the 19-minute `test_rec_caps_guards_hold`) and 2c (what forces a check part to one machine while other machines' slots stand free: infra's INF-838) stay, after the PR is open.

