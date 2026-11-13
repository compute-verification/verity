# Targets for a later heavy.json pull request (data only, cut from main after the moves)

Measured in r20261113-142348-fa99 (shard 1/4 at 1333bd53f693, records first, --jobs 8 --mem-gb 48), from
shard1-fa99-progress.jsonl: peak_gb is the target's process's ru_maxrss, rss_start_gb the process it was forked from.
heavy.json lists a target that grows more than HEAVY_GB (4) past that, by its peak_gb.

| target | peak_gb | rss_start_gb | growth |
|---|---|---|---|
| DsaTopKReadRows_v1{N=131072,TOPK=2048,BLOCK=16384,READ=EmbeddingRead_v1{V=131072,H=576}} | 35.98 | 5.29 | 30.69 |
| DsaTopKReadRows_v1{N=40960,TOPK=2048,BLOCK=16384,READ=EmbeddingRead_v1{V=131072,H=576}} | 35.97 | 5.29 | 30.68 |
| Dsv4SwaWindow_v2{SW=128,BS=64,NB=16384} | 12.76 | 3.70 | 9.06 |
| DsaPositionsMerge_v1{M=8,K=2048} | 10.96 | 5.54 | 5.42 |

72 of shard 1's targets only; the other shards were not run.
