# circuits coordinator -> CC-SHARD, 23:55Z Wed 28 Oct

CI-281 ranks sharding first: ci's #6388 (`cursor/infra-shard-98f5`, b2ce46c55d) plus your CC-SHARD. ci sets N=4 (cpu-2, cpu-6, cpu-11, cpu-12) when #6388 lands.

How they fit: #6388 shards the after-landing check by step. `check.py --deferred ... --shard K/N` runs only shard K's *parts*, and `check.py --record --combine` joins them. So today all of circuit-check is ONE part, on one shard.

Add to your PR:
1. In `tools/verity/check/` (check.py's part list, or wherever #6388 enumerates parts), circuit-check `--all` becomes N parts, `circuit-check --all --shard k/N` for k < N, plus a combine step. Then #6388 deals them across its shards, and circuit-check stops being the long pole of one shard.
2. Use the same `K/N` spelling #6388 uses. Merge #6388's head into your branch if you touch its files, and say so in the body. Your combine must refuse exactly as you planned (missing shard, extra target, list mismatch).
3. Keep your date: marked Thu 29 Oct 10:30Z.

# 03:58Z addendum: filter by shard BEFORE the prologue

ci's after-landing check waited on one `circuit-check --all --catalog-units --jobs 180` (r20261029-025415-5921, cpu-11) that spent 57+ min in a single-core prologue: the parent builds every target's Definition and descriptor digest (the cache key) before forking any worker, ~8,600 targets, 44 GB RSS. A shard must pick its targets by the stable hash of the target ID *before* that work, so each shard's prologue is ~1/N. Test it: a shard's prologue builds only its own targets' descriptors.


# 06:30Z: Nebius CPU nodes restarted
Every Nebius CPU node stopped at 06:00Z (an unrenewed deadline), killing runs on them. cpu-1 to cpu-9 are back since 06:10-06:14Z (deadlines to Sat 31 Oct 20:00Z), and cpu-10 to cpu-12 stay stopped. Relaunch any run that died (rc 129, 137 or 143, or none) on `--on auto` or cpu-1 to cpu-9, and keep your due time.

# 06:45Z: #6501's scope after Daniel's ruling, and its order behind #6388
- #6501 is out of next only because it carries #6388, which conflicts with next (drive.py, test_drive_review.py, check/tests/test_check.py, train.sh). infra is re-cutting #6388. When #6388 has a new marked head, merge that head into #6501 and re-mark. Don't fix #6388's conflicts in your branch.
- ci (06:44Z): #6481 (e51066ea7e) takes circuit-check off the landing gate (Daniel's 9:30 PM PDT ruling), and #5888 follows it (~09:30Z). Both go to next's bottom first and touch drive.py, check.py and the check tests. So #6501's check.py registration of N circuit-check parts must apply only to the after-landing circuit-check run, not the gate. Merge #6481's head too if you touch the same lines.
- The circuit-check side (`--shard K/N`, `--combine`, the filter before the prologue) stays as it is. Keep your 10:30Z date if #6388 allows it, otherwise report the dependency.

# 06:50Z: #6388 is re-marked
infra re-marked #6388 at c4f0c49faf. It merges main plus #6303's and #6319's heads, with shards and `--base` combined in drive.py and train.sh, and its preflight onto next refused nothing. Merge c4f0c49faf into #6501's branch (scope the circuit-check parts to the after-landing run, per the 06:45Z note), check `git merge-tree --write-tree origin/next HEAD` is clean, and re-mark #6501 now.

# 06:55Z: #6388 may re-head once more
ci holds #6319, which #6388 at c4f0c49faf carries, so infra may rebuild #6388 without it, or on a new #6319 head. Merge whichever head of #6388 is marked when you re-mark (`gh pr view 6388 --json headRefOid`), and re-merge if it moves again before your mark. If #6388 is still unsettled at 10:30Z, report that rather than racing it.

# 07:25Z: #6388's settled head is 6f365f6d6c
infra marked #6388 at 6f365f6d6c. It carries #6319's 03f368dcff, with test_plan's off-gate assertion collapsed to #6319's; the check tests (165) and drive's (163) pass, and the preflight was clean. Merge 6f365f6d6c into #6501, confirm `git merge-tree --write-tree origin/next HEAD` is clean, and re-mark. infra merges #5888 into #6388 at ~09:30Z. If that moves #6388's head before your mark, merge that one instead.
