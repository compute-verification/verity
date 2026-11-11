[2026-11-11T21:40Z] 1c236d34eb2c
From top (your coordinator), Wed 11 Nov 1:40 PM PST. One addition to your brief, for the shape "swarm with an integrator". Nothing else changes, and the PR is still due by 8 PM PST, unmarked.

Since your launch two more swarms started, on Daniel's direct ask to the leads at about 1:06 PM PST: memory-accounting's (MEM-269, integrator MEM-270, six seats) and compute-research's for compute accounting's code (CMP-409, integrator CMP-410). With vllm's (VLLM-341) that is three. I ruled on CMP-409 and MEM-269 at 1:21 PM PST: each is a trial on TOP-402; one set of rules across the three, a swarm posts only its differences; one lint list for the whole tree.

Read before you write that shape's section:
1. MEM-269's description and its comment of 1:35 PM PST (seven rules drawn from six leads' answers and vllm's first day), and the comment of 1:36 PM PST (its five differences from VLLM-341).
2. CMP-409's description and comments.

What the section should hold, each line citing the issue it comes from:
- the rules the three share, stated once (what a seat may edit, what is frozen and by whose word, how a hand-in is made, how seats are reached, how the integrator lands: how many pull requests a day and what goes apart);
- how it fails, from the record: vllm's train broke this morning on an import of a deleted module inside a function, which the quick tier does not resolve (MEM-269, rule 1); files in a Lean package directory mixed into an ordinary hand-in (rule 2);
- the trial's number: list entries and ruff findings removed a seat a day, and the swarm's pull requests the queue ejected;
- that two score tools exist now, in the notes and not in the repo (vllm's lanes/vllm/tools/swarm_* and memory-accounting's lanes/memory-accounting/tools/swarm.py, whose `gone HEAD` check works on any directory). Say so as an open item with its owner unnamed; do not move a tool in this PR.

Mark each claim you did not check yourself as the lead's.

