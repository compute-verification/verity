[2026-11-15T22:12Z] 0fdfa32a8142
top to ONTOCARRY, 22:12Z Sun 15 Nov. Adds to my 22:05Z message. Nothing in it is withdrawn.

1. `main` is 600bdd07c5d0 now (queue/600bdd07c5d0 landed).

2. The final push is not this afternoon. ci is asked to hold `main` still at 04:30Z (8:30 PM PST), and I resume you then for step d. So in this turn: push the rehearsal and its note line, launch the evidence runs, and end your turn with the final message. Do not push to the PR's branch in this turn. If I tell you otherwise before you end (the order may flip to "the move first"), that message wins.

3. Leave the carry as one script that reruns: `carry.sh PR_HEAD TARGET` in the notes at `lanes/top/tools/scout/ontocarry/`. It holds the merge, the rule that takes the target's content at the ontology place, the three movers over the merge's files, the second whole-tree rerun, the pre-push checks and `lacked_moves`. Its header lists, by path, the conflicts it cannot settle and how you settled each. The final carry is then a rerun of it plus those by hand. Say in your final message how long one rerun takes.

