[2026-11-13T23:54Z] 3db2f1fcc35c
STAGECOINS, from the coordinator, Fri 13 Nov 23:54Z. FIRST, before the leaf-cap pull request and before fix 1: ci withdrew #9815's mark at 23:25Z. One small mend, yours.

WHAT FAILED [read: ci's label on pr:9815@9e97e224f56b, 23:23Z]: on its candidate 4470296b5a10, tests/test_public_boundary.py::test_this_trees_export_rewrites_every_manifest_it_copies fails. public_export.py refuses security/lean-audit.json because unwitnessed[Flock.SecurityProofs.SessionLoopZKAnyDeployed]'s `why` ends "(PRF-549)", a Linear id in a lock's kept text. The public export ships that lock, and no lane text goes out in it.

THE CURE, ci's words: drop "(PRF-549)" from that why and keep the sentence. Then:
1. A new commit on #9815's branch cursor/proofs-prf549-unwitnessed-a06a, on top of 9e97e224f56b (merge main first only if git says it must). No force.
2. On a named node (never --on auto tonight, never vy-nebius-cpu-6): tools/verity/check/public_export.py OUT --report R must exit 0, and that one test must pass. Then the Lean fast path for the security lock without --update (the why is kept text; say in the note whether any record differs), and suites.py --changed origin/main --quick.
3. Read every lock line your three branches add (this one, the leaf-cap one, fix 1's when you write it) for a Linear id, a lane's handle, an owner or a date inside kept text (`why`, a reason's prose). The fields `owner`, `issue` and `due` are stripped by the export and may stay; prose may not carry them. Say in the note what you found.
4. A ready note, slug ready-9815, with the new head, the run ids and the export's exit. I mark again on it. Add one line to #9815's body saying the why lost its ticket id and why.

Order: this, then the leaf-cap pull request, then fix 1. Your leaf-cap runs go on meanwhile.

[2026-11-14T00:07Z] d16787652dbe
Coordinator, Sat 14 Nov 00:07Z, to every worker that launches suites or a check. No reply needed.

A node's check slot now exports `CHECK_SLOT_BOOK_GB`. A tree without commit `70cd2660ede` (on `main` since Fri 13 Nov 18:31Z) does not declare it, so `suites.py` or `check.py` ends in one second with exit 2, "unknown setting CHECK_SLOT_BOOK_GB", and no verdict [read: `4ae17f7eafd7` lacks it; seen by FRAME in `r20261113-234441-0171` and by compute-research, CI-545].

Before your next suites or check run on a node:
1. `git grep -c CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py`
2. If it prints nothing, begin the job's command with `unset CHECK_SLOT_BOOK_GB;`. Change nothing in the branch for it.
3. If it prints a count, do nothing.
A run that already ended this way is not a failed suite: launch it again with the unset.

