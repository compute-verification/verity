# Live-console panels

`verity_console.py` builds Verity's panels for the site's live console (`/admin/live` and `/console` on website-docs) and PUTs
them with the `panels:write` key. This is the master copy. The hosts run copies of it, so deploy from here rather than editing
on a host.

- **`--group node1`** runs on vy-nebius-1 every 5 minutes, via `infra/nebius/verity-console.{service,timer}`. It builds:
  - the node's Prometheus and Kueue panels (`verity/node1-*`);
  - the pool panels (`infra/*`), from `/workspace/usage/infra-pool.json`;
  - the proofs hill-climb tables (`verity/hillclimb-*`), from `/workspace/usage/hillclimb/*.json`.

  It needs `util_collect.py` beside it, and the `cluster` package (`infra/cluster/src/cluster`) in `cluster/` beside it: the
  GPU usage rows (`cluster.usage`) it records in `/workspace/usage/usage-hourly.jsonl`, which `research usage` reads, and
  derives `/workspace/usage/held-idle-hourly.jsonl` from. Each pool's lease ends that pool's idle before a shrink, read from its
  running `n1_lease.py loop` in `/proc` (three variables of its environment, nothing else), so the console runs as the loops'
  user, `research`; when it can't read them, node 1's hours wait for a pass that can.
- **`--group store`** runs on the control pod every 10 minutes, from its `console-loop.sh`. It builds the evidence-store panels
  (build, PoUW overhead, coverage, merge trains). It needs `ovd.py`, a copy of the Project store's
  `scripts/overnight-dashboard.py`, and the `research` package. `console-loop.sh` is the loop, started at boot by the pod's
  `/post_start.sh`; it reads the R2 credentials from `/root/.config/verity/r2.env` and the store from `/workspace/steward/store`.

Deploy to vy-nebius-1 with `research deploy` (`tools/research/src/research/pods/nebius/deploy.toml` lists the two scripts, the `cluster` tree and both
units; it reloads systemd after a unit changes, and the timer's next pass runs the new files):

~~~sh
research deploy install vy-nebius-1 tools/research/console/{verity_console.py,util_collect.py} \
    /workspace/research/console/cluster infra/nebius/verity-console.{service,timer} --ref origin/main --by WHO --apply
~~~

Deploy to the control pod (`vy-control-verity`, `nv7h6w1pairkcu`). The loop runs `verity_console.py` fresh each time, so the next
run picks up a copied file without a restart:

~~~sh
for f in verity_console.py ovd.py console-loop.sh; do
  research pods ssh nv7h6w1pairkcu -- "cat > /workspace/console/$f" < tools/research/console/$f
done
~~~

Before a deploy, run it without publishing: `python3 verity_console.py --group node1 --key /nonexistent --out /tmp/vc-dry`.
