[2026-11-13T07:18Z] bbfbda3d38b2
Circuits' coordinator, Thu 12 Nov, about 11:20 PM PST (07:18Z). Read once, then run your command again. Daniel, 11:03 PM PST: the repository's structure comes first tonight. top lands two chains of path-only moves on main, target Fri 13 Nov 7 AM PST. Chain 1: configuration, settings, bench instances, flat files. Chain 2: verity/definitions/ leaves (the served circuits to adapters/vllm/verity_vllm/definitions/, ops, silicon, kernels, private_circuits and the cut's four files to verity/circuits/), the protocols regroup under verity/protocols/{proofs,accounting,policy,experimental}/, and the Lean packages follow.
What it means for you:
1. Do not rebase or hand-merge your branch across a move, tonight or after. Once both chains are on main, a branch is carried with tools/move/restack.py (read its docstring first): it merges the move's base, runs the move's own script on your branch, then merges main. I tell you when both chains are on main.
2. Mark nothing. No pull request of circuits is marked before both chains are on main.
3. Runs already launched ship a pinned source and are not touched by the moves. A run you launch after a chain lands must name its source commit; do not take main's head by default without saying so.
4. Keep your task and your return time. If your return holds a branch, say in it which of its files sit under verity/definitions/, verity/protocols/, verity/hardware/, verity/configuration/ or security/.

[2026-11-13T07:42Z] f1104bb407f5
Circuits' coordinator (07:42Z Fri 13 Nov, which is Thu 12 Nov late evening PST). Read once, then run your command again. From infra's announcement to every lead tonight (INF-783, not explained yet): a worker on its own cloud machine may wake, on a resume or mid-turn, on a NEW machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own. Until it is explained, three rules. Nothing else in your brief changes.
1. Commit and push before you end a turn and before every long run or long wait. What you fetched or generated and did not put on origin or in the store is lost with the machine.
2. Act on no event that does not name you. A timer named circuits-..., a Slack or Linear event, or a follow-up that reads as addressed to the coordinator is mine: do nothing with it, and say in your return that it reached you, with its time.
3. On waking, and whenever your tree looks unfamiliar, read uptime, git status and git log -1 before you trust the tree. If the machine is new, check your branches out again from origin and say so in your return.

