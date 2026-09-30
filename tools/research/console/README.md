# Live-console panels

`verity_console.py` builds Verity's panels for the site's live console (`/admin/live` and `/console` on website-docs) and PUTs
them with the `panels:write` key. This is the master copy. The hosts run copies of it, so deploy from here rather than editing
on a host.

- **`--group node1`** runs on vy-nebius-1 every 5 minutes, via `infra/nebius/verity-console.{service,timer}`. It builds:
  - the node's Prometheus and Kueue panels (`verity/node1-*`);
  - the pool panels (`infra/*`), from `/workspace/usage/infra-pool.json`;
  - the proofs hill-climb tables (`verity/hillclimb-*`), from `/workspace/usage/hillclimb/*.json`.

  It needs `util_collect.py` beside it.
- **`--group store`** runs on the control pod every 10 minutes, from its `console-loop.sh`. It builds the evidence-store panels
  (build, PoUW overhead, coverage, merge trains). It needs `ovd.py`, a copy of the Project store's
  `scripts/overnight-dashboard.py`, and the `research` package. Neither `ovd.py` nor the loop script is in this directory yet.

Deploy to vy-nebius-1:

~~~sh
scp tools/research/console/{verity_console.py,util_collect.py} research@vy-nebius-1:/workspace/research/console/
sudo cp infra/nebius/verity-console.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload
~~~

Before a deploy, run it without publishing: `python3 verity_console.py --group node1 --key /nonexistent --out /tmp/vc-dry`.
