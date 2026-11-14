# CIR-278 LINTFIX checkpoint, 2026-11-14T07:42:10Z

LINTFIX on CIR-278, the lead's 07:12Z brief: #9835's last verdicts, changes 1 and 2, then 3 to 5.

#9835: branch `cursor/circuits-cir278-lint-self-fixes-6f04` at `c7ecec3f5d971ec19cd854ee6b29f990a59e07e6` (repo_ruff.json back in main's order; its diff against main is the frozen.py line alone). Its repository run at that head: r20261114-071633-e289, waiting for a slot.

Change 1: branch `cursor/circuits-cir278-stale-terms-6f04` at `90917ec7584ff23fac47662f62ec2c93a1a04e60`, pushed: 12 circuits entries (28 uses) dropped from repo_retired_terms.json by hand, no other line moved. Its repository run: r20261114-071723-5e38, running. Pull request opens when it is read.

Change 2: branch `cursor/circuits-cir278-cap-senses-6f04` at `f24cd5eecd61852566daa420f11e8c9fdd37a13d`, pushed, cut from change 1's head: six other senses of "cap" and two file lists; 1,302 sites no longer counted, none gained; circuits' entries: 27 removed, 31 lowered, by hand. Its repository run: launched.

Runs at #9835's former head 7bc0 (ids only):
- r20261114-060030-384e: done, read on its node, source_sha 7bc07334513b: 22 suites, 22 passed.
- r20261114-060151-8531: verity-flock, running.

Left: read 5e38 and open change 1; read e289; 8531's verdicts on CIR-278; change 2's run and pull request by 11:00Z; changes 3 to 5 by 17:00Z.

Stuck: nothing.
