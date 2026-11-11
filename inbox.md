[2026-11-11T06:44Z] ccb927a8cc1c
library: YOUR BRIEF, written again (Tue 10 Nov 10:45 PM PST). The first one went with your first turn. This is the whole of it; where it differs from what you pieced together, this wins.

WHY. Daniel, 9:59 PM PST: the library of training algorithms holds one (routes/subset.py, 127 lines), and he wants many, so agents can hill-climb three plots: (1) how much the policy slows standard frontier training algorithms; (2) policy-induced against algorithm-induced inefficiency at small scale; (3) how both scale with effective training compute. Issue TRN-100; goal step training/algorithm-library-first-eight, due Thu 12 Nov 6 PM PST.

THE FIRST PULL REQUEST, done when:
1. An entry is a route (code and dials) or a config of one: no second API. Honest and covert entries are one kind of thing. A new entry costs a config or under about 100 lines.
2. At least 8 standard algorithms are entries, each held to a torch reference by a test at a tiny shape.
3. Each entry's step is traced on a CPU (real shapes on a node).
4. One command prints plot 1's table from the calculator: entry, slowdown under the inference-only policy, what binds.
5. Entries the route API can't express are listed with the reason, in the body.

WHICH EIGHT. Choose for spread in what a step carries and reads, since that is what the policy prices. My list to start from: AdamW, SGD with momentum, Muon, Adafactor, Lion, one second-order method (Shampoo or SOAP), gradient accumulation, an average of the weights (EMA). Read the 18 rows of the published panel training/standard-workloads first and keep its names where an entry is the same algorithm. Send me your eight and your entry design by inbox BEFORE entry 4; go on meanwhile. The later twelve (not this PR) include the forward-only ones (evolution strategies, forward-forward, zeroth-order) and the token-routed mixture of experts.

THE CALCULATOR. The table's slowdown comes from the calculator (verity/protocols/inference_only/rho), never from training_slowdown/formalism.py's copy of the bound: onecalc (bc-2f024cb4-e0ea-5389-84f4-6c6d22e0f980, TRN-101, branch cursor/training-one-calculator-f980, same base as yours) is deleting that copy and naming the one module the benchmark reaches the calculator through. Ask it by inbox for that module's name and import only through it.

BRANCH. cursor/training-algorithm-library-6e3e off #8809's head 95239b187833. Push every commit as you make it. Open the PR as a draft (`research queue open --issue TRN-100 --by training --draft`) once two entries and the command exist. Do NOT mark; I say when.

RULES. No GPU run and no attack run (TRN-73's pause). No measured result in the tree. launch-1111/COMMON-1111.md's other rules stand.

TELL ME by inbox: at your first push (branch, head); your eight and the design before entry 4; and at the end, with the table your command printed and the run it came from.
WHERE YOU RUN (read by me on the machine, 10:42 PM PST): you are on the coordinator's own VM, not one of your own: 4 cores, 16 GB, 42 GB of disk free, shared by me, library, onecalc and resume. COMMON-1111.md's line "you are a cloud agent on your own VM" is wrong for you, and its section on tonight's move is over: the move is on main since 7:26 PM PST (5c095bb4ab70).
- Here: one `uv sync` in your own worktree, and small test files. Suites, real-shape traces, and anything over about 2 GB or a few minutes go to a node: `research run --on auto ...`. This VM's kernel has killed a 9 GB and an 11.7 GB process this week.
- Write only inside your own worktree (/home/ubuntu/wt-NAME) and /tmp/NAME/. /cursor/stores/self is MY store: read it, never write it. Leave alone /tmp/mainwt, /tmp/replotwt, /tmp/overlay*, the other worktrees, ~/.research/runs, and the tmux server (it is mine: no attach, no send-keys, no kill).
- Never kill a process you did not start, and only by PID.

