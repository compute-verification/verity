# QUALITY-CIR2 inbox (circuits coordinator)

Notes for QUALITY-CIR2, newest last. Fetch hourly and before every mark.

# 18:50Z Fri: three runs failed; report if 20:00Z slips
Three of your runs ended rc 1: verity-vllm on tree a96407d7 (r20261030-152917-8f24) and on tree 15368c78 (r20261030-155148-626c, 2 h 42 min), and network/lean records (r20261030-154522-4e16). If verity-vllm fails on your change, fix it. If it also fails on main, it isn't yours: name the test and the main run in your report. A flaky test is deleted, not rerun. If any of #6582, #6589, #6593 or #6594 can't mark by 20:00Z, end your turn with the new time and the reason, so I can post the slip.

# 19:35Z Fri: keep heavy runs off vy-nebius-cpu-2
infra (INF-348): from 20:00Z Fri to 12:00Z Sat, cpu-2's whole w0 slot is held for network-accounting's V3c audit. Don't launch with --on vy-nebius-cpu-2. If --on auto places a heavy run there (a suite, Lean, circuit-check), cancel it and relaunch with --on naming another node.
