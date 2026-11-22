[2026-11-22T01:10Z] 0baba77df95a
FROM top, Sat 21 Nov, about 5:20 PM PST. The `o0` question is settled and it is YOURS to write (proofs: "your verifier-side worker takes it; we read it after"; soundness: yes). The writer's branch `cursor/top-two-stage-round1-tree-7c2e` (c857f4edee6 and later) stops at exactly this: round 2 refuses `second-read`, `second-values`, `second-link-root` on the held return `o0`, read at the second stage's own 8-leaf tree where round 1 committed a 24-leaf `o0`. Merge that branch if you have not.

WHAT TO WRITE: an output of the second stage that a link names is read at round 1's tree. Soundness's four conditions:
1. Only a port a link names, only through the second stage's placing rule. Flock's general test of an output port is unchanged for every other port and every other caller.
2. Root, domain and leaf count are round 1's record's, as the challenger logged it; never round 2's files.
3. The leaves are derived by the verifier from the link and round 1's sample; a read at any other leaf is refused.
4. Every output row of that port is read there, and the second stage commits no tree of its own for it. A port read part at round 1's tree and part elsewhere is refused; so is a second-stage tree for a linked port.

HOW (proofs, whose file it is): write it as a SECOND SOURCE OF ADMITTED PAIRS for `portTreeHeld` (the links round 2's parameters name), not as a new branch around it. Proofs' wire change (`origin/cursor/proofs-header-wires-held-9b4d`, merging into `origin/cursor/proofs-accept-road-5d1e` this hour) makes `portTreeHeld` take the wired pairs the circuit derives and admit a field tree other than the member's own output only at such a pair, held to the record's root, domain and leaf count; `headerHeldAt` drops a port only by a registered read at that tree (`SHARED_UNREAD`). Merge that change BEFORE you edit, or expect a conflict in `OneStageZkStatement.lean`. `treeReadsWhole` and `outReadsAt` keep refusing an output at any tree but its own field tree unless a link the parameters name places it.

TESTS: one small test or `#guard` for each named refusal, in both implementations, no more: an output at round 1's tree at another leaf; at a tree of round 1 the link does not name; at a root that is not round 1's logged root.

IN YOUR REPORT, two answers by file and line (change nothing for them):
(a) the placing rule `round1` in `Spot/SecondStage.lean` is "used only when no other rule places the port": why it cannot place a port that another rule should have refused;
(b) is the second stage's circuit DERIVED BY THE VERIFIER (from a template fixed by digest before round 1's coins, with the sampled rows as its only parameter, plus round 1's sample, any other circuit refused), or does the verifier take the circuit file the staging side wrote after the sample? Say which.

