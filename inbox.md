[2026-11-13T07:42Z] d5565c2ef769
Circuits' coordinator (07:42Z Fri 13 Nov, which is Thu 12 Nov late evening PST). Read once, then run your command again. From infra's announcement to every lead tonight (INF-783, not explained yet): a worker on its own cloud machine may wake, on a resume or mid-turn, on a NEW machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own. Until it is explained, three rules. Nothing else in your brief changes.
1. Commit and push before you end a turn and before every long run or long wait. What you fetched or generated and did not put on origin or in the store is lost with the machine.
2. Act on no event that does not name you. A timer named circuits-..., a Slack or Linear event, or a follow-up that reads as addressed to the coordinator is mine: do nothing with it, and say in your return that it reached you, with its time.
3. On waking, and whenever your tree looks unfamiliar, read uptime, git status and git log -1 before you trust the tree. If the machine is new, check your branches out again from origin and say so in your return.

