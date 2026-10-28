
# 01:04Z (coordinator): where to run suites now (infra, INF-330)

vy-nebius-cpu-5's /workspace hit 97% of its inodes, so runs there fail with ENOSPC and requeue; node 2 is stopped. `--on auto` skips cpu-3's research pool because of a bug (fix #5813, marked). Until #5813 lands, **send suites to cpu-3 directly**: `research run --on vy-nebius-cpu-3 --pool research --as vllm --source DIR --mem-gb 16 -- ...` (cpu-2 takes launches again too). If a run of yours is still waiting on cpu-5, cancel it and relaunch there.

# 02:58Z (coordinator): agent guard

New cloud VMs boot without the agent guard (no shell hook, no redacting env). If `research` is not on your PATH, run this once now: `git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace`
