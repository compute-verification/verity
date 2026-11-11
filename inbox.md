[2026-11-11T03:11Z] 76d219ff0db7
From harness (TRN-90), Tue 10 Nov, 7:13 PM PST. (i) and (ii) are answered; code follows. "Run" = run by me on my VM (CPU, kernel 6.12, today's sandbox); "read" = read in code.

(i) NOT sound, and not one hidden path. Hiding a root closes only the direct reach.
- Run: job B's route process lists and rewrites job A's state.json. /proc/PID/root, cwd and mem are denied to it, for A's own process and for A's route process, so ptrace and /proc are not a way out. A signal to either is permitted.
- Not one path: a route's process itself writes under the runs root (its own $OUT: E, --weights), and the job's scratch, where each kept state passes before the job's own process hashes it, is tempfile's shared directory. It would take "hide <root>/runs, keep my own run dir, scratch inside it": a new mount step in sandbox.py.
- The way out as the same user is FILES, and no hidden state root closes it. Run: a sandboxed process leaves a bytecode file under a cache prefix the user can write, and a later process outside any sandbox runs it in place of the module's source. On a node that prefix is <root>/cache/pycache, set for every run of a shipped tree, and <root>/src/<sha> with its .venv is shared by the runs of a commit (read, tools/research/src/research/remote.py). Likewise the login files that `bash -lc` reads. Code left there runs as the user outside every sandbox and can rewrite any state.
- The reach is wider than kept states: B's route can rewrite A's $OUT/run.json, the result, before custody takes it. That holds on #8680 with or without a resume.
- Infra ALREADY has the boundary (read, tools/research/src/research/jobuser.py): `research run --job-user USER` runs the workload in a unit with the host read-only but for its run dir, every other run's dir and the launch requests out of sight, no /home, its own /tmp. Three things keep kind training-attack out of it today: RestrictNamespaces=yes (our sandbox's unshare fails there), a stop exits 128+signal instead of the job's 75, and a read-only host leaves `uv sync` no tree to write. The ask for infra is those three, not a new boundary; wording in my final message.

(ii) SOUND for its windows, and small. The command already lists out/state/state.json in the stopped run's record, and that listing carries the store's sha256 of the file (read, store/local.py `listing`). The command passes it to the job; the job's own process refuses a state.json of any other sha256. The four SHA-512 in it are then the stored record's, and (a) binds the tensors as both processes load them. No network at the restart.
- Closes: every rewrite after custody has taken the stopped run's output: the node's store cache (the unhashed cached blob too), transit, $OUT from the fetch to the load. Also a state planted for a job the command never found stopped.
- Left: a rewrite on the stopping node before custody takes the output (the store then holds the rewritten state); --local, which has no stored copy; node scratch unless the record's copy also goes to $OUT (I will add that if it stays small); and the argv itself, which lies in <root>/requests on the node (read).

Doing now: (ii), one commit with its tests. Not doing (i) inside our sandbox tonight unless you say so: it would rebuild part of infra's unit and still leave the files above open.

Node run r20261111-024638-3f74: 13 of 27 suites reported at 7:09 PM PST, all passed (benchmarks/training_slowdown 280, cluster 304, repository 279); still running.

