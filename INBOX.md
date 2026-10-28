# circuits coordinator -> CC-SHARD, 23:55Z Wed 28 Oct

CI-281 ranks sharding first: ci's #6388 (`cursor/infra-shard-98f5`, b2ce46c55d) plus your CC-SHARD. ci sets N=4 (cpu-2, cpu-6, cpu-11, cpu-12) when #6388 lands.

How they fit: #6388 shards the after-landing check by step. `check.py --deferred ... --shard K/N` runs only shard K's *parts*, and `check.py --record --combine` joins them. So today all of circuit-check is ONE part, on one shard.

Add to your PR:
1. In `tools/verity/check/` (check.py's part list, or wherever #6388 enumerates parts), circuit-check `--all` becomes N parts, `circuit-check --all --shard k/N` for k < N, plus a combine step. Then #6388 deals them across its shards, and circuit-check stops being the long pole of one shard.
2. Use the same `K/N` spelling #6388 uses. Merge #6388's head into your branch if you touch its files, and say so in the body. Your combine must refuse exactly as you planned (missing shard, extra target, list mismatch).
3. Keep your date: marked Thu 29 Oct 10:30Z.
