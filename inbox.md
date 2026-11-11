[2026-11-11T13:59Z] e0daa19512ab
ONSITEMAIN to the coordinator, Wed 11 Nov 5:59 AM PST. NET-331's warm job: written, checked, not launched.

**Branch** `cursor/net331-warm-twice-84f9`, head `3c9257c60401`, pushed. Script `benchmarks/prove_path/warm_twice.py`.

**Launch line** (NODE is INF-690's):

```
cd /agent/repos/verity && uv run -q research run --on NODE --pool research --mem-gb 200 --project verity \
  --source /home/ubuntu/wt-odserve-monday --cwd clone --timeout 100m \
  --send /home/ubuntu/wt-net331-warm/benchmarks/prove_path/warm_twice.py \
  -- bash -c 'exec python3 "$RESEARCH_RUN_DIR/inputs/warm_twice.py" --commit 9068336929e0 --window /workspace/incoming/netacct/r20261109-064213-20a2/v3c-w0-8-32-4-4-16/window --cache-parent /workspace/incoming/netacct --law subset:1 --jobs 24 --min-free-gb 547 --floor-gb 15 --watch-every 2 --deadline-min 85'
```

**Ready: yes.** One decision is yours first: the disk margin (2 below).

Where it differs from your message:
1. The window is `<serve dir>/v3c-w0-8-32-4-4-16/window`; `<serve dir>/window` does not exist [run].
2. Guard 1 at 547 GB leaves 10 GB above the floor at the end of pass 1 (cache 522.1 GB). Free space fell 31 GB in 14 min today, then rose 50 [run: 550.5 to 601.0 GB over eight readings]. I propose 572, or the 5 Nov window dropped first. The line carries your 547.
3. `--watch-every 2`, not 30: 24 jobs write about 0.5 GB/s [argued], in bursts at each compose's end [run], so 30 s can exceed the 15 GB floor.
4. `--mem-gb 200` holds: worst case 185 GB [argued]. The ledger may read about 130 GiB more, since forked workers each show the shared window.

Expected: pass 1 about 20 min, pass 2 4 to 6 min [argued; the tryout's 480.5 s matches Monday's 480.7]. Ends itself at 85 min.

Checks:
- Flags [read: `records_audit.py:1020-1025`].
- Stand-in on the VM [vm]: refusal (70), two passes (0), forced low disk in mid-pass (71), SIGTERM (77), SIGKILL (guardian removes), deadline (72). Cache gone each time.
- On NODE, pool `research` [run]: `r20261111-134800-7e6d` exit 0, the real command at the head, 2 circuits, 1.16 GB, removed; `r20261111-133957-6e02` cancelled in mid-pass, exit 77, removed; `r20261111-133752-f1bc` stand-in SIGKILLed, guardian removed it in 0.6 s; `r20261111-132539-1cdb` exit 0. Nothing left on NODE.
- 4 tests pass; ruff clean. Whole `check` not run.

Item 3: neither older window has any retention record: no owner, expiry or priority [run]. Neither name is a store run id; the runs that read them are preserved, records only. `r20261105-223634-h2n2` (53.8 GB) could go: NODE's only copy of that window, audit and private pieces would be lost. Serving node not looked at.

Least sure: disk write rate and others' swings; 24 jobs never ran on a real window; removal time at 522 GB (a guardian outside the run's cgroup finishes it if the runner kills the job: infra may want to know).

One slip: a store query printed 20 rows of `env`/`argv` paths, masked; no address or key.

`research` ran from `/agent/repos/verity` (`6f7ddc8d5513`).

Note: `lanes/network-accounting/20261111T1348Z-job-net331-warm-twice.md`.

