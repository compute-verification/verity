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
