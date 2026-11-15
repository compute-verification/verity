[2026-11-15T22:46Z] 5d7c6597169c
top, 22:46Z Sun 15 Nov. One small change is yours after all, AFTER the table (the table by 23:45Z stays first). ci and infra both said the stats row is not theirs tonight; ci reads yours at tier 2.

The change, as ci stated it: a shard's peak is kept when its slot's scope ends, in the slot's record (tools/check/slot.py), and `research queue stats` reads it. Two things the stats then print that they do not print today:

1. Distinct shards waiting for a slot (one per candidate and part), with twins counted apart. Today's "runs in line" counts each shard once per machine it is queued on (the first free slot runs it, the others exit 75), so 98 read as a backlog when about ten shards waited.
2. By part, over the shards that ended in the window: cores at the highest, anon memory at the highest, memory.peak. So the next slot sizing reads from the record and nobody samples cgroups by hand again.

Rules:
- Base on main (9bec18de4f1f now). Stay out of tools/research/src/research/vqueue_loop.py.
- Two other changes touch slot.py: kernels' #10265 and infra's SLOTFIT (not open yet). Read `gh pr diff 10265`, keep your lines apart from its hunks, and say in the body which of its lines are near yours; whichever lands later merges the others.
- If the record a slot writes at its end cannot hold the peak without a change on the machines (a deploy), say exactly what infra reruns; you deploy nothing.
- Tests beside the code, in the owning suite; one named test file locally at most, the suite through `research run --on auto --cwd source` launched from your tree, with remote.json's source_sha read before you believe it.
- Open it UNMARKED: `research queue open --head BRANCH --sha SHA --title T --body-file F --by top`, body with `Lane top`, and a first line asking ci for its tier-2 read. Number on your note's first line. By 02:30Z; if it will not be done by then, say so in the note at 01:30Z with what is left.

