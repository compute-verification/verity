# CIR-278 LINTFIX checkpoint, 2026-11-14T08:17:19Z

READY #10019 90917ec7584ff23fac47662f62ec2c93a1a04e60
READY #10032 f24cd5eecd61852566daa420f11e8c9fdd37a13d

#10019 (change 1): repository passed at its head (r20261114-071723-5e38, read on its node: 314 passed). Not marked yet: its --changed suites and ratchet gate run r20261114-080031-c686 is waiting for a slot.
#10032 (change 2, a proposal to top): repository passed at its head (r20261114-074137-4a00, read on its node: 322 passed). Not marked: it waits on top's yes, asked on CIR-278 at 08:17Z. Its --changed and ratchet run: r20261114-080251-732c, waiting.

#9835: census.py and its I001 entry are back to main's text in 3579b2f1c85b50bad170b7fbf9b93dbba6f3128b, pushed. Its repository run r20261114-075726-c774 and its --changed and ratchet run r20261114-080031-8e5d are waiting for slots. The READY line follows when they are read. Its beyond_gemm.py is under heading 1 of vllm's GPU check list (program-core); vllm's 07:21Z yes covered that import sort, and I leave it unless told.

Changes 3 to 5: change 3's two files are both on vllm's GPU check list (headings 1 and 3), so there is no change 3. Its stale entries (SLICESIZE's item I) ride in change 4: branch cursor/circuits-cir278-ruff-hand-6f04 at 57d66cf171d66be3b4163cb9b459e81a04c90e56, not pushed yet. Change 5 (mypy) waits on mypy's messages, run r20261114-080401-3fde, waiting.

#9835's former head 7bc0: r20261114-060030-384e done (22 passed); r20261114-060151-8531 (verity-flock) running.

Stuck: nothing. Runs are slow to get slots.

## 2026-11-14T08:39:31Z

Not started 40 minutes after launch, still waiting for a check slot: r20261114-075726-c774 (#9835's repository at 3579b2f1, launched 07:57Z), r20261114-080031-8e5d and r20261114-080031-c686 (#9835's and #10019's --changed suites and ratchet, 08:00Z). Waiting too, under 40 minutes so far: r20261114-080251-732c (#10032), r20261114-080401-3fde (mypy messages), r20261114-081858-03cd (change 4's --changed and ratchet at 57d66cf171d66be3b4163cb9b459e81a04c90e56, pushed) and r20261114-082047-24e3 (which of change 4's test files a node runs). r20261114-060151-8531 (verity-flock) still running. I keep waiting; no pod.

## 2026-11-14T09:38:15Z

READY #10019 90917ec7584ff23fac47662f62ec2c93a1a04e60 marked

#10019: its --changed suites (repository, verity-graph) and the ratchet gate passed at its head, run r20261114-080031-c686, read on its node. The ratchet gate prints 9265 -> 9253. merge-tree is clean against main d97767352e2c, the queue candidates and circuits' heads, and adds nothing to the line's own two. The body has a Mark section and a Goal circuits/slice-clean line.
#9835: repository passed at 3579b2f1c85b50bad170b7fbf9b93dbba6f3128b (r20261114-075726-c774, read on its node: 369 passed). Its --changed and ratchet run r20261114-080031-8e5d is still running; I mark it when that is read.
Item 0 posted on CIR-278.
Change 5 is pushed: cursor/circuits-cir278-mypy-cures-6f04 at e06f2a98e1f49d829014ed302724e97054bb030e, with five files and 8 mypy.json lines. Its --changed and ratchet run is r20261114-092046-2753, waiting. Change 4's run r20261114-081858-03cd is waiting too.

## 2026-11-14T15:49:05Z

Back after the stop. r20261114-080031-8e5d at 3579b2f1c85b was cancelled by its slot's limit. Ten suites passed: repository took 706 s, research 1165 s, verity-graph 1172 s, and seven others under 150 s each. verity-vllm, verity-numerical and verity-flock were still going when it was cut. Why so long: #9835 touches adapters/vllm, which is an input of 13 suites, verity-flock among them, and verity-flock alone took 11,390 s at 7bc0 (r20261114-060151-8531). Relaunched at 3579b2f1c85b50bad170b7fbf9b93dbba6f3128b, split by suite: verity-vllm and verity-numerical in r20261114-154809-8a4b, and verity-flock in three parts by --files: r20261114-154810-d405, r20261114-154809-c616 and r20261114-154809-ebb4.

## 2026-11-14T15:55:44Z

READY #10140 c75a9462e2a707029e524f521c861e53ca385fa0

#10140 is change 4 (ruff by hand, with item I), opened unmarked. At 57d66cf1 its run r20261114-081858-03cd passed 20 of the 23 suites --changed selects, and the slot limit cut it before research, verity-numerical and verity-flock. The head moved to c75a9462 to put back test_artifact_applicability_independent_of_record.py and its entry, since the refactor line now deletes that file. Its suites run again split: r20261114-155348-9460 (verity-vllm), r20261114-155348-0603 (the other 20 and the ratchet), and r20261114-155348-cbca, -4bcf and -5289 (verity-flock in thirds). The four runs at 57d66cf1 that I launched at 15:50Z were cancelled as superseded.

## 2026-11-14T16:17:26Z

#9835 at 3579b2f1: r20261114-154809-8a4b (verity-vllm and verity-numerical) and verity-flock in thirds (r20261114-154810-d405, r20261114-154809-c616, r20261114-154809-ebb4) are running. #10140 at c75a9462: 0603, 4bcf and 5289 are running; 9460 and cbca are waiting. Change 5 (cursor/circuits-cir278-mypy-cures-6f04 at e06f2a98e1f49d829014ed302724e97054bb030e): r20261114-092046-2753 passed 21 of 24 suites and the ratchet gate (9264 -> 9256) before the slot limit cut it. Its remaining three are relaunched by suite: verity-numerical and the circuit-check suite in r20261114-161626-315f, and verity-flock in thirds. I run no subagent, timer or subscription.

## 2026-11-14T16:53:08Z

After the turn was cut at about 16:28Z I relaunched nothing. #9835: r20261114-154809-ebb4 (verity-flock part 3/3) passed: 950 passed, 43 skipped. 8a4b and d405 are running and c616 is waiting. #10140: 9460, 0603, cbca, 4bcf and 5289 are running. Change 5: r20261114-161626-315f is waiting. The VM restart lost its verity-flock thirds' run ids, so I wait on them blind and launch them again only if no pass shows. Model: Claude Opus 5.5. No subagent, timer or subscription.

## 2026-11-14T17:32:08Z

#9835: 8a4b and d405 are running. c616 (verity-flock 2/3) has waited for a check slot since 15:48Z with 11 checks ahead and no asker, since its launching VM is gone, so it will not move by itself; its wait cap is 4 h. #10140: cbca passed; 9460, 0603, 4bcf and 5289 are running. Change 5: 315f is waiting. Nothing relaunched.

## 2026-11-14T17:54:12Z

Found the stall. Runs launched with --cwd source into one shared tree take that tree's Lean lock one at a time and hold it while they wait for a check slot. #9835's 8a4b and d405 sat 124 min behind c616 having done nothing, and #10140's 4bcf and 5289 sat behind 9460 and 0603. I cancelled those four and relaunched them with --cwd clone, each in a tree of its own: #9835 r20261114-175324-7860 (verity-vllm, verity-numerical) and r20261114-175324-fd54 (verity-flock 1/3); #10140 r20261114-175324-ee58 and r20261114-175324-b806 (verity-flock 2/3, 3/3). Kept: c616 (#9835 verity-flock 2/3, holding its tree's lock and waiting for a slot), and 9460 (#10140 verity-vllm passed, 21074 passed) and 0603. Passed so far at 3579b2f1: ebb4 (verity-flock 3/3) and c774 (repository), plus ten suites in 8e5d.

## 2026-11-14T18:14:18Z

c616 cancelled too: it had waited 67 min for a slot and never started. Relaunched as r20261114-175434-1fdd (#9835 verity-flock 2/3, own tree). All of #9835's three (7860, fd54, 1fdd) are waiting for check slots. #10140: 9460 done (verity-vllm passed); 0603, ee58 and b806 are waiting. Change 5: 315f is running.

## 2026-11-14T18:20:47Z

#10032 merged with main (#10019 landed): head ccdf607b2eae8e22c378d7b31ff835da20811acc. The layout move 8469d5dc (registration_map) names only verity/hardware/registration.py, which none of the branch's own lines name, so a plain merge of main is what restack.py's script would give on them. One conflict, in test_repo_retired_terms.py with #10024 (Flock's caps, says()), resolved by keeping both. The list still deletes 27 and lowers 31 of circuits' entries (1,026 uses), none below what the tree holds. On the VM: test_repo_retired_terms.py, 29 passed. Its --changed suites (verity-vllm, repository, verity-graph) and ratchet: r20261114-182002-22e5. Still unmarked: it waits on top's yes, and none has come on CIR-278.
