---
name: writing-runs
description: >
  How to write a job for `research run` in Verity so its custody stays small: pick the working directory (`--cwd source`, or
  `--cwd clone` for a job that writes into the tree),
  write what to keep under `$OUT` (the run dir's `out/`), and keep scratch, caches and build
  trees in a `mktemp -d` directory outside the run dir. Custody uploads the run's own logs and records, all of `$OUT`, the files
  `--declared-output GLOB` names, and other files only up to 256 MiB; the rest it lists by path, size and sha256 without bytes.
  Use when writing or launching a
  `research run` command, a job's custody upload is slow or huge, custody printed "trimmed N undeclared file(s)", or a run's
  output is missing from its run record.
---

# Writing runs

`research run` starts the workload in its run dir unless you pass `--cwd`. On a machine, custody publishes that dir to the
store's remote once the workload exits. Every byte the workload leaves there is a byte custody reads, and it may upload it
too. Run `r20260930-080414-bae0` started a build in its run dir, and custody spent 54 minutes on 9.5 GiB before it was
killed. Whoever writes the job decides what to keep.

A local `research run` (no `--on`, as in a pod) publishes only its telemetry and the files its `result.json` lists, unless it
passes `--run-record`. With it, the attempt carries the same selection custody's run record does. The Nebius Sky templates
with a free-form `CMD` (`port-capture`, `prover-bench`, `prover-dev`) pass it.

## What custody keeps

- **Always uploaded:** the run's own files, meaning the logs, `job.json`, `status.json`, `events.jsonl`, `result.json` and
  the other records the harness writes (`custody.RUN_FILES`).
- **`$OUT`, always, in full:** the harness creates the run dir's `out/` before the workload starts and gives the workload its
  path as `$OUT`. Custody uploads everything under it, as if every launch passed `--declared-output 'out/*'`.
- **Declared outputs, beside `out/`:** each `--declared-output GLOB` is matched against paths relative to the run dir, and
  every matching file is uploaded. The glob uses `fnmatch`, so `*` also matches across `/`: `res/*` is all of `res/`,
  and `*` is everything.
- **Everything else is trimmed:** undeclared files are uploaded in path order while their total stays within 256 MiB
  (`custody.UNDECLARED_CAP_BYTES`). The rest are listed in the run record's `meta.trimmed` with path, size and sha256, and
  no bytes. Custody still pins a trimmed file by its hash, so changing it afterwards breaks `.custody`.

Custody prints `custody trimmed N undeclared file(s)` when it trims. The cap exists only so that runs launched before this
rule keep working. Once lanes declare their outputs it drops to 0, and undeclared files are never uploaded.

## The norms

1. **Run from a directory you chose.** Use `--cwd source` (or `source/SUB`) for a command that reads the checkout, `--cwd clone`
   (or `clone/SUB`) for one that writes into it (a build, a Lean `.lake`): on a machine every run of the commit shares the
   source tree, and a clone is the run's own, removed when it ends; the workload imports the tree's Python from it too. Locally
   only the cwd moves: an editable install still imports from, and writes beside `__file__` in, the invoking checkout. Or pass
   an absolute `--cwd`. A relative path in the command
   that resolves against the run dir is the failure above.
2. **Put what you want kept in `$OUT`.** The workload finds it in `$OUT` whatever its cwd is (it is
   `$RESEARCH_RUN_DIR/out`), and custody always sends all of it to the store, so the launch declares nothing. A result that must
   live elsewhere in the run dir needs `--declared-output GLOB`; quote the glob so your shell doesn't expand it.
3. **Keep outputs small:** the numbers, JSON, logs and the digests of big things. An artifact that must survive but is
   large (a model, a proof, a trace over about 100 MiB) goes into the store with `research data put --kind K --file P
   --preserve` from inside the job, and you record its `art:` id in `$OUT`.
4. **Put scratch in a `mktemp -d` directory that you delete,** as in the example, or elsewhere outside the run dir. A job's
   `$TMPDIR` may be unset (queued jobs on vy-nebius-1 have none), so `$TMPDIR/x` can become `/x`. Caches, build trees
   (`target/`, `.lake/`, `node_modules/`), downloaded datasets and checkpoints never go in the run dir.
5. **Record what a job retains outside its run dir, and delete it only through `research retention rm`.** A job that keeps
   data on the node (PoUW's passes) runs `research keep DIR --owner @LANE --expires never --priority keep` as soon as `DIR`
   exists; lowering it takes top's yes. A brief that deletes says `research retention rm PATH --approved-by
   @OWNER --ref THREAD --apply`, never `rm -rf`, so a stale step fails closed. A worker that hits "Permission denied" or a
   refusal stops and reports it: someone made that path read-only or kept it on purpose. On 6 Oct a stale step ran
   `chmod -R u+w` and `rm -rf` on serve `r20261006-201412-b28d`'s cited passes, about 27 GB on each node, and they are gone.
   `retention rm` deletes passes only once their record names a preserved artifact.

~~~bash
research run --on M --cwd source -- \
  bash -c 'd=$(mktemp -d); trap "rm -rf $d" EXIT; python bench.py --scratch "$d" --out "$OUT"'
~~~

Before a long run on a machine, try the same command locally for a minute. Check that `ls` of the run dir shows only your
output directory beside the harness's files.
