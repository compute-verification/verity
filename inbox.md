[2026-11-11T06:44Z] 8391667cf96f
onecalc: YOUR BRIEF, written again (Tue 10 Nov 10:45 PM PST). Your first turn's text went with the stop, and my resume message told you to go on from a brief you no longer hold. This is the whole of it; where it differs from what you pieced together, this wins.

WHY. Daniel, 9:59 PM PST: the machinery that calculates the slowdown a protocol or a demo induces must be one thing, clear and easy to maintain. Today three places compute it: the calculator (verity/protocols/inference_only/rho, whose min_cut is the one Python statement of Lean's HE), training_slowdown/formalism.py (1,533 lines, a second copy of the bound: 128,564 beside 129,298 at one point, 0.57% apart), and the trace pricer (capture/, flops.py). Issue TRN-101; goal step training/grade-reads-calculator, due Thu 12 Nov 6 PM PST.

PULL REQUEST 1, done when:
1. The grade's slowdown is the calculator's bound of record. My provisional ruling: it is the solved count (min_cut's), not formalism's closed form.
2. formalism.py's copy of the bound is deleted.
3. A test holds the benchmark to reaching the calculator through one module.
4. The body lists every printed figure that changes, and by how much. No figure changes silently.

TWO STOPS.
- BEFORE you delete anything: send me by inbox what formalism.py holds that is NOT a copy of the bound, each with who reads it. I rule where each goes. Go on with 1, 3 and 4 meanwhile.
- If any printed figure moves by more than 0.57%, stop and send me the figure, both values and the point.

WHAT TO COMPARE AGAINST. The published table was rebuilt and republished tonight at 10:36 PM PST (the capture fix: two sensitivities double in twelve rows; no slowdown moved). Its tables are art:b760642b7efede6c9475603284d12c4a6111fa638101be30a7957c8407b7d57c (458 rows, uncut).

NEIGHBOURS. library (bc-ffb1fc16-88b8-53d7-b009-7d6affef6e3e, TRN-100, same base) prints its table from the calculator and will ask you for the one module's name: answer it early. PAIR's #8816 (head 62b53e40ad24) adds a test on the calculator's count of wrongly accepted bits: read its three files and do not move what it tests.

LATER, not tonight: PR 2 (what the grade still needs of the trace pricer, the rest deleted, or a dated plan if it waits on the step compiler), and one entry point from (policy, step) to the slowdown and what binds, with a README of ten lines.

BRANCH. cursor/training-one-calculator-f980 off #8809's head 95239b187833. Push every commit as you make it. Open the PR as a draft through `research queue open --issue TRN-101 --by training --draft`. Do NOT mark; I say when.

RULES. No GPU run. No measured result in the tree. launch-1111/COMMON-1111.md's other rules stand.

TELL ME by inbox: at your first push (branch, head); the list of what is not a copy; and at the end, with the table of moved figures.
WHERE YOU RUN (read by me on the machine, 10:42 PM PST): you are on the coordinator's own VM, not one of your own: 4 cores, 16 GB, 42 GB of disk free, shared by me, library, onecalc and resume. COMMON-1111.md's line "you are a cloud agent on your own VM" is wrong for you, and its section on tonight's move is over: the move is on main since 7:26 PM PST (5c095bb4ab70).
- Here: one `uv sync` in your own worktree, and small test files. Suites, real-shape traces, and anything over about 2 GB or a few minutes go to a node: `research run --on auto ...`. This VM's kernel has killed a 9 GB and an 11.7 GB process this week.
- Write only inside your own worktree (/home/ubuntu/wt-NAME) and /tmp/NAME/. /cursor/stores/self is MY store: read it, never write it. Leave alone /tmp/mainwt, /tmp/replotwt, /tmp/overlay*, the other worktrees, ~/.research/runs, and the tmux server (it is mine: no attach, no send-keys, no kill).
- Never kill a process you did not start, and only by PID.

