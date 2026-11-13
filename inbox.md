[2026-11-13T07:18Z] 57c4ba9ed24a
Circuits' coordinator, Thu 12 Nov, about 11:20 PM PST (07:18Z). Read once, then run your command again. Daniel, 11:03 PM PST: the repository's structure comes first tonight. top lands two chains of path-only moves on main, target Fri 13 Nov 7 AM PST. Chain 1: configuration, settings, bench instances, flat files. Chain 2: verity/definitions/ leaves (the served circuits to adapters/vllm/verity_vllm/definitions/, ops, silicon, kernels, private_circuits and the cut's four files to verity/circuits/), the protocols regroup under verity/protocols/{proofs,accounting,policy,experimental}/, and the Lean packages follow.
What it means for you:
1. Do not rebase or hand-merge your branch across a move, tonight or after. Once both chains are on main, a branch is carried with tools/move/restack.py (read its docstring first): it merges the move's base, runs the move's own script on your branch, then merges main. I tell you when both chains are on main.
2. Mark nothing. No pull request of circuits is marked before both chains are on main.
3. Runs already launched ship a pinned source and are not touched by the moves. A run you launch after a chain lands must name its source commit; do not take main's head by default without saying so.
4. Keep your task and your return time. If your return holds a branch, say in it which of its files sit under verity/definitions/, verity/protocols/, verity/hardware/, verity/configuration/ or security/.

[2026-11-13T07:38Z] a06e0689e104
Circuits' coordinator (07:38Z Fri 13 Nov, which is Thu 12 Nov late evening PST). Read once, then run your command again. Two changes to the second pass's bounds, after Daniel's word tonight that each lead takes the machines its priorities need. Everything else stands: record v4, cheapest first, the lookup's form read before staging, back by Fri 13 Nov 10 AM PST (18:00Z).
1. Up to FOUR runs at a time at 64 GB each (it was two), each still at most 21,000 s. Split the types not yet reached so that no type is staged twice. Record v4 names, for each verdict, the run that gave it.
2. My yes, recorded here, for the types the 64 GB rule left out (the 13 whose Lean check and the one whose stager your estimate puts past 64 GB): ONE run at a time at --mem-gb 160, at most 21,000 s, over the types your estimate puts at 128 GB or under, cheapest first. A type estimated past 128 GB stays not run and is listed with its estimate. Write each such type's measured peak into the record. This run counts as one of the four.
3. If a run waits in the queue more than ten minutes before it starts, put its minutes in your return.

[2026-11-13T07:42Z] 130380163253
Circuits' coordinator (07:42Z Fri 13 Nov, which is Thu 12 Nov late evening PST). Read once, then run your command again. From infra's announcement to every lead tonight (INF-783, not explained yet): a worker on its own cloud machine may wake, on a resume or mid-turn, on a NEW machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own. Until it is explained, three rules. Nothing else in your brief changes.
1. Commit and push before you end a turn and before every long run or long wait. What you fetched or generated and did not put on origin or in the store is lost with the machine.
2. Act on no event that does not name you. A timer named circuits-..., a Slack or Linear event, or a follow-up that reads as addressed to the coordinator is mine: do nothing with it, and say in your return that it reached you, with its time.
3. On waking, and whenever your tree looks unfamiliar, read uptime, git status and git log -1 before you trust the tree. If the machine is new, check your branches out again from origin and say so in your return.

