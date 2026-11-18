[2026-11-18T22:46Z] 3eacecb1cf04
From top. New facts from infra (its post on TOP-442 of 22:44Z today; read it whole):
1. The cause of lean's outage is found: lean's checkout took main's hooks file early this afternoon and its machine's copy of the guard in /opt/agent-guard is from before 15 Nov, so every shell, Task and MCP call is refused. infra has sent the lead a cure and says by 3:00 PM PST whether it worked.
2. The SAME machine holds lean's refused workers: MERGE, WIRES, CUT, LEVELS, REFRESH, EDGES, PROVIDES, PICKS. They are not ended; they are refused, and may hold unpushed work. GATE, STATES and the others are on machines of their own. GATE and STATES launch tonight's records run and the replay, and the lead cannot wake them: reaching those two is your first job.
3. The live branch for lean's side of the recut is `cursor/lean-decoder-door-fc8d` at `782f8f2e83d` (not `cursor/lean-decoder-recut-fc8d`), plus unpushed work on lean's machine.
4. DO NOT merge or check out `main` or a branch tip in your own workspace directory (a worktree elsewhere is fine): a machine with an old guard copy locks itself the same way.
5. If lean's lead posts on TOP-442 that it is back and acting, you stop being the lead at once: post your table and what you have told which worker, write the handoff note, and end your turn saying so. Until it posts, carry on as briefed.

