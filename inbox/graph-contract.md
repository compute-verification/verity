# The fine-grained check: contract v1 (Daniel approved, Sun 1 Nov 2026)

Verity computes a task graph that depends only on the commit. Infra's executor runs any part of it with one result cache,
one artifact store and one event stream. Drive only reads results. Daniel builds the dashboard on the events and the graph.

## Decisions in force
1. Keys come from declared inputs, not traced reads. A sandbox enforces the declaration; an undeclared read is an error.
2. Keys include repository paths. A pure move reruns what it moved, once. No key is built to survive moves.
3. Runners report facts. What gates, what may stay red and what a red triggers is drive's config, never a runner's.
4. Lean moves to the module system, so a proof-only edit rebuilds one module (Lean v4.34.0; every dependency is ported).
5. `tasks.toml` holds families (tens of entries), not one row per task.
6. No `owner` field anywhere in the graph, results or events. Who gets paged is drive's policy, from paths.

## Layout (new package `tools/verity/graph/`, distribution `verity-graph`, import `verity_graph`)
- `verity_graph/contract.py`: schema names, canonical JSON, the key rule, file digests.
- `verity_graph/families/<family>.py`: `def tasks(root: Path, files: list[str]) -> list[dict]`. `files` is `git ls-files`
  at the commit (POSIX, relative). Returns graph/v2 task dicts without `key`; the core adds digests and keys.
- `verity_graph/runners/<kind>.py`: `def run(task: dict, work: Path) -> dict`, returning result/v1's fields.
- `tasks.toml`: one `[family.<name>]` table each (files covered, runner kind, env, about template, resources, timeout).
- CLI: `verity graph [--json] [--family F]` lists graph/v2; `verity graph run ID --work DIR` runs one task and prints
  result/v1; `verity graph check --local [--family F]` is the serial reference executor (no cache, no parallelism).
  `verity run` and `verity tasks` already exist and mean other things.

## graph/v2
```json
{"schema": "verity-graph/v2", "commit": "<sha>",
 "envs": {"py": {"spec": ["uv.lock", "pyproject.toml"], "digest": "sha512:<hex>"},
          "lean": {"spec": ["<lean-toolchain>", "<lake-manifest.json>", "..."], "digest": "sha512:<hex>"}},
 "files": {"<path>": "sha512:<hex>"},
 "tasks": [{"id": "lean-module:Security/Proofs/Flock/Claims/Recursion/AccTies.lean",
            "kind": "lean-module", "about": "compile and kernel-replay one module", "env": "lean",
            "argv": ["verity", "graph", "run", "<id>"], "vars": {},
            "inputs": ["Security/Proofs/Flock/Claims/Recursion/AccTies.lean"],
            "reads": [{"task": "lean-module:...", "output": "olean"}],
            "outputs": [{"name": "olean", "path": "out/AccTies.olean"}],
            "resources": {"cores": 1, "mem_gb": 6, "disk_gb": 1}, "timeout_s": 900,
            "key": null}]}
```
- Same bytes on any machine for one commit: no timings, store reads, environment or machine state.
- `files` holds every input file of any task, once. A runner's own source files are inputs of every task of its kind.
- An env's digest is SHA-512 over its spec files' digests.
- `key` is set when the task reads no artifact; else null, and the executor computes it when the reads' digests exist.

## The key rule
```
key = "sha512:" + hex(SHA-512(b"verity-key/v1\n" + canonical({
        "env": <env digest>, "kind": ..., "argv": [...], "vars": {...},
        "inputs":  sorted [[path, file digest], ...],
        "reads":   sorted [[task id, output name, output digest], ...],
        "outputs": sorted [[name, path], ...]})))
canonical(x) = json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
file digest = "sha512:" + hex(SHA-512(file bytes))
```
`about`, `resources` and `timeout_s` stay out of the key. Using the reads' output digests gives early cutoff: an artifact
rebuilt to the same bytes leaves its readers' keys unchanged. `tests/key_vectors.json` holds shared vectors both sides
(Verity and the executor) must reproduce.

## result/v1 (printed by `verity graph run`)
```json
{"schema": "verity-result/v1", "task": "<id>", "key": "sha512:<hex>", "status": "pass|fail|error",
 "seconds": 41.2, "cpu_s": 39.8, "peak_rss_gb": 3.1, "outputs": {"olean": "sha512:<hex>"},
 "undeclared_reads": [], "detail": {}, "log": "sha512:<hex>"}
```
`error`: the task couldn't run (worker died, disk full, over its memory ceiling, an undeclared read); never a result, the
executor retries it. `fail`: a result, never retried.

## events/v1 (the executor writes one JSON line per event; Daniel's dashboard reads only these and the graph)
```json
{"schema": "verity-events/v1", "ts": "...", "submission": "...", "purpose": "after-landing|ready|lean-check|nightly|adhoc",
 "commit": "<sha>", "task": "<id>", "key": "...", "event": "queued|hit|started|finished|errored|cancelled",
 "worker": "...", "wait_s": 0, "seconds": 0, "status": "...", "peak_rss_gb": 0, "hit_from": "<submission>"}
```

## Families
| family | tasks (main, 1 Nov) | lister and runner |
|---|---|---|
| test-file | 2,091 | G1 |
| lean-module | 4,424 | G2 |
| lean-lock | ~30 | G2 |
| circuit-binding, circuit-coverage, circuit-catalog | 197 + 1 + 1 | G3 |
| build (Flock verifier Lean build, flock-circuit-build, lean-agreement) | a handful | G0 |

## Invariants, each with a test
1. `verity graph` gives the same bytes in two checkouts of one commit at different paths.
2. It opens nothing outside the tree (an audit hook on `open`).
3. Keys reproduce `tests/key_vectors.json`.
4. No `owner` field.
5. It lists the whole graph in under 60 s on one core.
6. Errors never become results; a fail is never retried.
