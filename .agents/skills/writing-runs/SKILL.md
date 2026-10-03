---
name: writing-runs
description: >
  How to write a job for `research run` in Verity so its custody stays small: pick the working directory (`--cwd source`),
  write results under one output directory, declare them with `--declared-output GLOB`, and keep scratch, caches and build
  trees in a `mktemp -d` directory outside the run dir. Custody uploads the run's own logs and records, the declared files, and undeclared
  files only up to 256 MiB; the rest it lists by path, size and sha256 without bytes. Use when writing or launching a
  `research run` command, a job's custody upload is slow or huge, custody printed "trimmed N undeclared file(s)", or a run's
  output is missing from its run record.
---

# Writing runs

`research run` starts the workload in its run dir unless you pass `--cwd`. On a machine, custody publishes that dir to the
store's remote once the workload exits. Every byte the workload leaves there is a byte custody reads, and it may upload it
too. Run `r20260930-080414-bae0` started a build in its run dir, and custody spent 54 minutes on 9.5 GiB before it was
killed. Whoever writes the job decides what to keep.

## What custody keeps

- **Always uploaded:** the run's own files, meaning the logs, `job.json`, `status.json`, `events.jsonl`, `result.json` and
  the other records the harness writes (`custody.RUN_FILES`).
- **Declared outputs:** each `--declared-output GLOB` is matched against paths relative to the run dir, and every matching
  file is uploaded. The glob uses `fnmatch`, so `*` also matches across `/`: `out/*` is all of `out/`, and `*` is everything.
- **Everything else is trimmed:** undeclared files are uploaded in path order while their total stays within 256 MiB
  (`custody.UNDECLARED_CAP_BYTES`). The rest are listed in the run record's `meta.trimmed` with path, size and sha256, and
  no bytes. Custody still pins a trimmed file by its hash, so changing it afterwards breaks `.custody`.

Custody prints `custody trimmed N undeclared file(s)` when it trims. The cap exists only so that runs launched before this
rule keep working. Once lanes declare their outputs it drops to 0, and undeclared files are never uploaded.

## The norms

1. **Run from a directory you chose.** Use `--cwd source` (or `source/SUB`) for a command that reads the checkout, `--cwd clone`
   (or `clone/SUB`) for one that writes into it (a build, a Lean `.lake`): on a machine every run of the commit shares the
   source tree, and a clone is the run's own, removed when it ends. Or pass an absolute `--cwd`. A relative path in the command
   that resolves against the run dir is the failure above.
2. **Put results in one directory and declare it.** The workload finds its run dir in `$RESEARCH_RUN_DIR` whatever its cwd
   is. For example, it writes to `$RESEARCH_RUN_DIR/out/` and the launch passes `--declared-output 'out/*'`. Quote the glob so
   your shell doesn't expand it.
3. **Keep outputs small:** the numbers, JSON, logs and the digests of big things. An artifact that must survive but is
   large (a model, a proof, a trace over about 100 MiB) goes into the store with `research data put --kind K --file P
   --preserve` from inside the job, and you record its `art:` id in `out/`.
4. **Put scratch in a `mktemp -d` directory that you delete,** as in the example, or elsewhere outside the run dir. A job's
   `$TMPDIR` may be unset (queued jobs on vy-nebius-1 have none), so `$TMPDIR/x` can become `/x`. Caches, build trees
   (`target/`, `.lake/`, `node_modules/`), downloaded datasets and checkpoints never go in the run dir.

~~~bash
research run --on M --cwd source --declared-output 'out/*' -- \
  bash -c 'd=$(mktemp -d); trap "rm -rf $d" EXIT; python bench.py --scratch "$d" --out "$RESEARCH_RUN_DIR/out"'
~~~

Before a long run on a machine, try the same command locally for a minute. Check that `ls` of the run dir shows only your
output directory beside the harness's files.
