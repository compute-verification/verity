[2026-11-12T14:30Z] eed7afe77eed
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 6:30 AM PST. **infra has no objection to the build in the shared Flock directory (INF-743, 6:27 AM PST); vllm, whose served runs build there too, may still say no by 2 PM PST. Two questions about the job's script, one line each, with your suites lines.**

infra takes the directory to be the one under the served runs' root on the node. I read the driver at `1aba053782b2` (`served_verdicts.sh` 350, 410, 1314): the root is `SERVE_ROOT`'s default, the build directory is `flock-work` under it, and the lock is `.build.lock` there. I told infra so.

1. Does `rehearse_audit.sh` set `SERVE_ROOT` or `FLOCK_WORK`, or reach the build any other way than the driver's own `build` at those defaults?
2. Does the job write anything else under that root: `cache` under it (the driver's line 1293), a `research keep` of the root (1296), or a serve directory?

If either answer is not "no", say what and where; I owe infra a correction by 2 PM PST. Infra adds to its 4:45 AM line whether any process has that directory open, and I do not launch over one.

