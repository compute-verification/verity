[2026-11-11T03:46Z] 9886a1c8c900
From HARNESS (TRN-90, #8775). Tonight's part is done and pushed; I end my turn and wait for #8680's crossed head.

HEAD: 239cab25ba121684908008a5c7a5ca38c8160ac7 on cursor/training-job-resume-5277. A draft, unmarked. main not merged, nothing pushed to #8680. One commit: "a job continues only from the state the stopped run's record holds" ((ii)). No commit for (i).

NODE RUNS (both read by me, tree_sha read):
- r20261111-024638-3f74 at 938b4a38f841, cpu-3: rc 0, 27 suites, 27 passed, 0 failed.
- r20261111-032019-f2dc at 239cab25ba12, cpu-2, the one more run: rc 0, 27 suites, 27 passed, 0 failed. benchmarks/training_slowdown 281 passed. --quick runs no slow test; the through-processes test ran on my VM (test_resume.py 10 passed).

(i) NOT sound, not one path (sent 03:11Z). (ii) sound for its windows, done: closes a whole rewrite after custody has taken the stopped run's output; leaves the stopping node before custody, NODE_STATE, --local.

THE ASK FOR INFRA (58 words):
For kind training-attack, let each job run under `research run --job-user`: its unit already hides other runs, /home, /tmp and the launch requests. Three things stop us: RestrictNamespaces=yes (our route sandbox needs user, mount, network and PID namespaces), a stop exits 128+signal where the job must exit 75, and the read-only host leaves `uv sync` no writable tree.

SIGNAL: a PID namespace in the sandbox would take away the route's signal (read, not run; #8680's sandbox.py), but not the stop: the job's own process sets no limit on a step's time, so a route that does not return is stopped by the runner at the run's time, exit 75, and continued.

DISK: nothing in the tree records a node's disk; `research disk-hold` reads it on the machine at launch. 44 to 55 GB on each GPU node needs infra's word.

NOTE: training-20261111T0345Z-handoff-trn90-stored-record-cross-job (the crossing's command is in it).

