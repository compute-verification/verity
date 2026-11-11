[2026-11-11T21:09Z] ac478544c1e0
REPLOT to training, TRN-111, Wed 11 Nov 1:10 PM PST (21:10Z). After the cut: branches whole, both builds have slots and are running; nothing launched twice.

BRANCHES (READ by me, git ls-remote): cursor/training-put-55a1 at 947e4be33c (te_core tree 11f297ba3e, cut from 225e85e26e); cursor/training-tokens-correction-55a1 (#9010) at 9941a3daa2; cursor/training-replot-55a1 untouched at 225e85e26e. Every commit of mine was pushed before the cut. The VM restarted: its scratch is gone (worktrees, fetched runs, my three comparison scripts); worktrees are back and the scripts rewritten, now kept where a restart does not reach.

THE THREE RUNS (READ by me on the node, 21:02Z to 21:09Z):
1. r20261111-201517-d96c (te_core 7de182a661, for the three counts): took a slot at 20:52:42Z, 64 GB reserved. Plan and layer as before (0 to trace, 379 held, 2 left). At 21:06Z both builds are in te_core.algorithm_policy; measured peak 29 GB so far. size-scaling's step comes next.
2. r20261111-201517-c982 (the put branch's code): took a slot at 20:47:12Z. Both builds finished their tables and entered the size-scaling step at 21:05:51Z and 21:05:57Z; peak 30 GB.
3. r20261111-201712-071f is not a size-scaling run: it was the put branch's standard-workloads build, 13 GB. It FAILED at 20:28:32Z, before the builder ran: my option.sh fetched the put branch's bundle before the one it stands on ("Repository lacks these prerequisite commits"). I mended the order and relaunched it before the cut as r20261111-203105-ffb5, with your ruling in it (--branch cursor/training-put-55a1, HEAD 947e4be33c). That one took a slot and FAILED at 20:46:04Z in the builder's first fetch: the node's store no longer holds the trace files of art:4be9df21... (READ by me: 9 of its 31 files are "remote" there now, the eight traces among them; the 15:20Z build read them locally), and a run's workload on the node has no remote. Same builder, same arts; the node dropped its local copies of preserved files in between.
   What I do about it, unless you say otherwise: fetch on the VM exactly what the builder fetches (9 trace arts, 16 records, the starts' start.json; by art id, so the same bytes) into a store of its own, send that with the run, and point the builder's `research data fetch` at it. No change to the builder. It is filling now.

WHAT I OWE NEXT:
- 1:45 PM PST: nothing, both builds hold slots. I will say so only if d96c stalls.
- By 2:30 PM PST: the three counts of size-scaling from d96c, falls first, with the 238M row and the two rows you named.
- After c982 and the standard-workloads build: per panel, equal or every cell and note character that differs (the put branch against 7de182a661).
- Thu 12 Nov noon PST: the staged artifact, the note's check, #9010's "What is put".

