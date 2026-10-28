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

`research run` starts the workload in its run dir unless you pass `--cwd`. On a machine, custody reads every byte left
there when the workload exits and publishes to the store:

- always, the run's logs and records (`custody.RUN_FILES`) and all of `$OUT` (`$RESEARCH_RUN_DIR/out`, whatever the cwd);
- each `--declared-output GLOB` (quote it; `fnmatch` on paths relative to the run dir, so `*` crosses `/`);
- other files in path order up to 256 MiB in all (`custody.UNDECLARED_CAP_BYTES`, kept for older launches; it drops to 0
  once lanes declare their outputs). The rest are listed in `meta.trimmed` by path, size and sha256, and custody prints
  `custody trimmed N undeclared file(s)`; a trimmed file is still pinned by hash, so changing it breaks `.custody`.

A local `research run` (no `--on`) publishes only its telemetry and the files its `result.json` lists, unless
`--run-record` gives it custody's selection; the Nebius Sky templates with a free-form `CMD` (`port-capture`,
`prover-bench`, `prover-dev`) pass it.

## The norms

1. **Pick the cwd:** `--cwd source` (or `source/SUB`) to read the checkout, `--cwd clone` (or `clone/SUB`) to write into
   it (a build, a Lean `.lake`), or an absolute path; never a relative path that resolves against the run dir. On a machine
   every run of a commit shares the source tree, while a clone is the run's own, removed when it ends, and the workload
   imports its Python from it. Locally only the cwd moves: an editable install imports from, and writes beside `__file__`
   in, the invoking checkout.
2. **Keep results in `$OUT`, small:** numbers, JSON, logs, digests of big things. A large artifact that must survive (a
   model, a proof, a trace past about 100 MiB) goes there too: a node holds no store write key, so `research data put` in
   a job fails. Cite it as the run id and its path under `out/`; an uncited run record leaves the store 7 days after upload
   (the store README §7.4), and a label or note naming the run keeps it. Don't copy in what the commit regenerates (a circuit, a build).
3. **Scratch, caches, build trees (`target/`, `.lake/`, `node_modules/`), datasets and checkpoints stay out of the run
   dir**, in a `mktemp -d` directory you delete. `$TMPDIR` may be unset (queued jobs on vy-nebius-1): `$TMPDIR/x` is `/x`.
4. **Data kept outside the run dir** is registered once it exists, `research keep DIR --owner @LANE --expires never
   --priority keep` (lowering it takes top's yes), and deleted only by `research retention rm PATH --approved-by @OWNER
   --ref THREAD --apply`, never `rm -rf`, so a stale step fails closed; it deletes passes only once their record names a
   preserved artifact. On "Permission denied" or a refusal, stop and report it: someone made the path read-only or kept
   it on purpose.

~~~bash
research run --on M --cwd source -- \
  bash -c 'd=$(mktemp -d); trap "rm -rf $d" EXIT; python bench.py --scratch "$d" --out "$OUT"'
~~~

Try a long run's command on its machine for a minute first (`--timeout 2m --wait`), never on an agent VM (AGENTS.md
"Tests"), and check that the run dir holds only your output directory beside the harness's files.
