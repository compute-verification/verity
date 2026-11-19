# Live-console panels

The site's live console (`/admin/live` and `/console` on website-docs) shows the store as panels: the last day's checks.
`verity_console.py` builds Verity's panels and PUTs them with the `panels:write` key.
This directory is the master copy. The hosts run copies of it, so deploy from here rather than editing on a host.

`status_export.py`, beside it, writes the docs site's status document, `verity/status/v0`: every guarantee the tree's
`lean-audit.json` locks list, proved outright or resting on named assumptions, and the results of record its tables and
demos name.

## Try it

~~~sh
python3 tools/research/console/status_export.py --root . --rev HEAD --out /tmp/status.json
~~~

This reads the tree at one commit through git and writes the status document, then prints one line: the commit, the
guarantees counted by state, and the document's size and digest. It needs nothing. Without `--facts`, a Lean audit's
output directory (a `check` run's `lean-audit/`), a guarantee whose facts it doesn't hold is `unknown`, never `proved`;
run it at the commit those facts were built from.

## The store group

`--group store` runs on the control pod every 10 minutes, from its `console-loop.sh`. It builds `verity/merge-checks`, the
last day's `check` Attempts with the main commit each landed as (its `landed` label); the site's Results view reads every
other store result from the store's index. It needs the `research` package. `console-loop.sh` is the loop, started at boot by the pod's
`/post_start.sh`; it reads the R2 credentials from `/root/.config/verity/r2.env` and the store from `/workspace/steward/store`.

## Deploying

Deploy to the control pod (`vy-control-verity`). The loop runs `verity_console.py` fresh each time, so the next
run picks up a copied file without a restart:

~~~sh
for f in verity_console.py console-loop.sh; do
  research pods ssh vy-control-verity -- "cat > /workspace/console/$f" < tools/research/console/$f
done
~~~

Before a deploy, run it without publishing: `python3 verity_console.py --group store --key /nonexistent --out /tmp/vc-dry`.
