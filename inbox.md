[2026-11-16T04:40Z] 2bd2af8642ce
From top, 04:40Z Mon 16 Nov. THE NEW TOP IS HERE: carry onto it now.

The final base is candidate `queue/21c1e5f2fff1` = 21c1e5f2fff136cfe9e84214c85352ac22657b03 [read by me: git ls-remote +
first-parent log]. It is `queue/269cc620ef65` plus two merges and nothing else: #10386 (PoUB: γ's terms from the Lean
verifier, Poub.Gamma) and #10440 (bench memory: a refused --local session exits 1). ci built it at 8:34 PM PST, depth is
back to 0, and the pins hold #10350 alone, so nothing else comes onto the chain before the move.

1. `git fetch origin 'refs/heads/queue/*:refs/remotes/origin/queue/*'`, then carry onto 21c1e5f2fff1 with carry.sh as
   you did on 269c (steps A to D). Whatever you already settled on 269c holds; the new part is only what #10386 and
   #10440 touch (the PoUB verifier's Lean and Python, benchmarks' memory files). A Lean file of theirs that the move
   renames or rewrites: the mover's own rewrite only, no hand edit of a Lean literal; name each such file in the note.
2. Push to `cursor/top-ontology-move-824c-r4` (a plain push on top of fa91140ba464, never a force).
3. Run F on that pushed head: verity-vllm --quick, research, the root suite, and PoUB's Lean-built suite, each detached
   with `research run`, launched FROM your carry's tree (its remote.json source_sha must be the pushed head).
4. Write on line 1 of your note: the pushed head, the base 21c1e5f2fff1, the run ids, and for each suite its counts
   beside the base's. Sync the note. Then wait on the runs in calls under 20 minutes; when all have ended, write their
   verdicts on line 1 and end your turn.

If the candidate 21c1 goes red and ci drops it, I tell you here; until then it is the base.

