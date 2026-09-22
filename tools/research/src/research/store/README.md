# research.store — the evidence store

Four objects, two relations. Everything else is a typed use of them.

~~~text
Tool         a reusable computation (a small Python declaration around an existing command)
Derivation   a requested computation: H(tool, input artifact ids, key params, key conditions)   -- "have we done this?"
Attempt      one execution of a Derivation = one `research run` (attempt id == run id)          -- "what happened?"
Artifact     an immutable, typed, content-addressed object (manifest + optional payload)

Replica      where an Artifact's bytes live (local cache, repo fixture, remote) -- subordinate
Label        an assertion about an Artifact/Attempt/Derivation, with provenance -- never a truth field
~~~

Dataset = Artifact kind `dataset-snapshot/v1`. Result = `bench-result/v1`. Verdict = `verification-verdict/v1`.
Fixture = `fixture/v1`. Proof = `proof/v1`. Probe capture = `tc-evidence/v1`. There is no Record and no Dataset object.

CLI: `research data …`. Package: `research.store`. Id prefixes: `art:`, `drv:`. Attempt id = run id as-is.
Stdlib only (the `research` rule); the S3 client is hand-rolled SigV4.

## 1. Canonical JSON and identity

`canon.dumps(obj) -> bytes`: `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`
encoded UTF-8. Floats render by Python `repr` (shortest round-trip); `-0.0` is rejected (use `0.0`); NaN/Inf are rejected;
ints are unbounded; keys must be `str`. `canon.sha256(obj) -> hex`.

~~~text
art:<sha256 hex of the canonical artifact manifest>     the manifest embeds payload digests, so the id covers the bytes
drv:<sha256 hex of the canonical derivation document>   no source revision inside (see §6)
reuse_key = sha256 hex of canon({"derivation": drv_id, "source": <source identity>})   None for dirty trees
attempt id = run id (r20260922-105905-2021)
~~~

Two agents freezing the same member set produce the same snapshot id (members are sorted and de-duplicated; `name`, `note`
and `query` are part of the content, keys absent when not given). Two attempts producing byte-identical outputs produce the
same artifact id; provenance lives on the Attempt (§4), not inside the artifact.

## 2. Artifact manifest — `research/artifact/v0.1`

~~~json
{
  "schema": "research/artifact/v0.1",
  "kind": "bench-result/v1",
  "meta": {"...": "small inline typed data; canonical size <= 256 KiB"},
  "refs": {"proof": "art:...", "inputs": ["art:...", "art:..."]},
  "payload": null
}
~~~

`payload` is `null`, a file, or a tree:

~~~json
{"type": "file", "sha256": "<hex>", "bytes": 1234, "name": "proof.bin"}
{"type": "tree", "sha256": "<hex of canon(files)>", "bytes": 5678,
 "files": [{"path": "tiles_random.npz", "sha256": "<hex>", "bytes": 100}, ...]}
~~~

Tree `files` are sorted by `path`; paths are relative with `/` separators, no `..`, `.` or empty components, no absolute, no
duplicates, and no path that is both a file and a directory of another (`a` and `a/b`); `bytes` is the sum; an empty tree is
legal. A file payload's `name` is a bare file name. Every file (and a file payload) is one blob at `objects/sha256/<hex>`.
`refs` values are `art:` ids or lists of them (the store checks the syntax, not existence — a ref may point at an artifact that
lives only remotely). `meta` may contain `art:` strings too; only `refs` is indexed as edges.

Manifest identity excludes producer, time, labels. The manifest file is `manifests/<sha256 hex>.json` (canonical bytes).

Kinds are strings `<name>/v<int>`. The store validates only the envelope; kind-specific validation is the producer's
(`kinds.py` holds the known kinds and a one-line description each; `put` warns on an unknown kind, never refuses).

## 3. Layout

Local store root: `~/.research/store` (or `$RESEARCH_STORE`). Remote uses the same relative keys.

~~~text
objects/sha256/<hex>                 blobs (content-addressed; write temp + hard-link, mode 0444; never overwritten)
manifests/<hex>.json                 artifact manifests (canonical bytes; hex == art id hex)
attempts/<run id>.json               attempt manifests (canonical bytes; immutable once published)
labels/<target>/<ts>-<by>.json       one immutable file per assertion; <target> = id with ':' -> '_'
snapshots are ordinary artifacts (kind dataset-snapshot/v1)
trees/<hex>/                         LOCAL ONLY: where `fetch ART` (no --to) materialises a tree payload
catalog.sqlite                       LOCAL ONLY: rebuildable index (§5); never uploaded
scratch/<reuse_key>/<name>/          LOCAL ONLY: tool caches namespaced by reuse_key (§6.5)
~~~

Python API: `research.store.local.LocalStore(root=None, *, config_path=None, repo=None)` (root from `$RESEARCH_STORE`, config
per §7, repo = `$RESEARCH_REPO` or the nearest ancestor of cwd holding `fixtures/artifacts.json`). Methods: `put_artifact`,
`put_manifest`, `get_manifest`, `has_local`, `fetch`, `where`, `put_attempt`, `get_attempt`, `push_attempt`, `add_label`,
`labels`, `snapshot`, `verify`, `push`, `reindex`, `index`, `remote`. Errors: `StoreError` > `NotFound`, `AttemptExists`,
`IntegrityError`; `remote.RemoteConflict`, `remote.ConfigError`. A file payload fetched without `to` is returned as its
read-only object path (no copy); with `to` it is copied to `to/<name>` when `to` is a directory, else to `to`.

## 4. Attempt manifest — `research/attempt/v0.1`

Published by `research run` at completion or failure (both — a failed attempt is evidence). Built from `job.json`,
`status.json`, `failure.json`, `result.json`, `research.env.conditions()`, and the Tool declaration when one applies.

~~~json
{
  "schema": "research/attempt/v0.1",
  "id": "r20260922-105905-2021",
  "tool": {"name": "bench_vu", "version": "1"},
  "derivation": "drv:...",
  "reuse_key": "<hex>|null",
  "argv": ["python", "-m", "..."],
  "params": {"rtt_ms": 0, "mode": "interactive", "...": "every parsed parameter"},
  "key_params": ["mode", "soundness", "batch"],
  "inputs": {"instances": "art:..."},
  "conditions": {"hw.family": "h100", "hw.sku": "NVIDIA H100 80GB HBM3", "driver": "...", "cuda": "...", "...": "..."},
  "key_conditions": ["hw.family"],
  "source": {"commit": "<sha>", "tree": "<sha>", "dirty": false, "dirty_digest": null, "lockfiles": {"uv.lock": "<sha256>"}},
  "closure": {"tools/tc_probe/tc_probe.py": "<sha256>", "...": "path -> content sha256 of every declared source file"},
  "execution": {"start_utc": "...", "end_utc": "...", "wall_s": 12.3, "exclusive": false, "load_start": 1.2, "load_end": 1.1,
                "host": "vy-g5", "project": "verity", "campaign": "r21-silicon", "run_dir": "/workspace/research/runs/r...",
                "stage": "bench", "remote": {"machine": "vy-g5", "pod_id": "...", "root": "...", "source_sha": "..."} },
  "cost": {"provider": "runpod", "pod_id": "...", "rate_usd_per_hour": 2.69, "usd": 0.0092, "shared": false},
  "status": {"state": "done|failed|cancelled", "rc": 0, "class": "SUCCESS", "failure": "<failure.json why>", "validation": "passed",
             "result": "valid (research/result/v0.1)|absent|invalid: ...", "timed_out": false},
  "outputs": {"result": "art:...", "run_files": "art:..."},
  "telemetry": {"events": "art:...|null", "resources": "art:...|null"}
}
~~~

`status.state` is status.json's state as recorded (`done` / `failed` / `cancelled`), `argv` and `params.argv` are the harness's
redacted argv, `execution.run_dir` is where the attempt executed (the machine's run dir for a remote run). A harness
`--input name=path` that points at a fetched artifact becomes `inputs.<name> = art:...`; any other `--input` enters the
derivation by content as the key param `input.<name> = sha256:<hex>` (`inputs` holds `art:` ids only).

Without a Tool declaration (`research run -- CMD` with no `--tool`): `tool = {"name": "<argv[0] basename>", "version": null}`,
`key_params = []`, `key_conditions = []`, so `derivation = drv(H(tool name, inputs, {}, {}))` — a real but coarse
derivation; `reuse_key` is still computed (clean tree) so `lookup` is honest. `params` then holds `{"argv": argv}`.

`research run` writes everything it computed before launch into job.json's optional `store` section (`tool {name, version, spec}`,
`derivation`, `reuse_key`, `key_params`, `key_conditions`, `params`, `key_params_resolved`, `inputs`, `closure`, `conditions`, `scratch`,
`campaign`, `store_root`); `publish` reads it back, so the manifest says what was requested, not what a re-parse would guess.
For a run that predates the store, `conditions` are reconstructed from what job.json recorded (SKU, driver, hostname, commit) and
everything else is `null`.

`attempts/<id>.json` is written once. Publishing an id that exists with different content is an error (`AttemptExists`), never an
overwrite; the same canonical content again is a no-op. The store validates the envelope only (`model.validate_attempt_doc`):
schema, id syntax, every top-level key above present (`reuse_key`, `closure`, `cost`, `telemetry` may be `null`), `tool.name` a
string, `derivation` a `drv:` id, `inputs`/`outputs` values `art:` ids, the whole document canonicalisable.

## 5. Index — `catalog.sqlite` (rebuildable, local only)

`research data reindex` wipes and rebuilds it from `manifests/`, `attempts/`, `labels/` (local), then, with `--remote`,
from the remote listing too (manifests and attempts the remote has and we lack are fetched and cached first). Deleting the
file and reindexing gives the same query results as the incremental index (tested). Tables (denormalised; add columns freely,
the file is disposable):

~~~text
artifacts(id, kind, bytes, payload_type, manifest_json, local INTEGER, remote INTEGER, repo_path TEXT)
artifact_meta(id, key, value)            flattened meta with dotted keys, scalars only (for --where)
refs(src, name, dst)
attempts(id, tool, tool_version, derivation, reuse_key, state, rc, validation, start_utc, wall_s, host, project, campaign,
         cost_usd, source_commit, source_dirty, manifest_json)
attempt_params(attempt, key, value)      flattened params
attempt_conditions(attempt, key, value)
attempt_outputs(attempt, name, artifact)
attempt_inputs(attempt, name, artifact)
labels(target, key, value, by, ts, ref, path)
replicas(id, location, verified_utc, detail)   verifications (§7); the one table a rebuild keeps, since it is not a local fact
~~~

Flattening: nested objects get dotted keys, lists numeric components (`families.0`), scalars are stored as text (strings raw,
everything else `json.dumps`: `true`, `null`, `1.5`); `--where` and `--label` compare that text. `local` means manifest + every
blob present here; `remote` means pushed (from a listing or a recorded push/verification).

`research.store.index.Index(path)`: `index_artifact(manifest_doc, art_id, local=None, remote=None, repo_path=None)` (None keeps the
current flag), `index_attempt(doc)`, `index_label(label, path)`, `set_presence(art_id, *, local, remote, repo_path)`,
`set_replica(art_id, location, verified_utc, detail)`, `rebuild(artifacts, attempts, labels)`, `artifact(id)`, `artifacts()`,
`attempt(run_id)`, `attempts_by_reuse_key(key)`, `attempts_by_derivation(drv)`, `attempts_by_output(art_id)`,
`select_artifacts(kind=None, where=None, label=None, attempt=None, tool=None, derivation=None)`, `labels(target, key=None)`,
`replicas(art_id)`, `sql(query, params=())` and `sql_with_columns(...)` on a read-only (`?mode=ro`) connection. Attempt queries
return the attempt documents; artifact queries return `{id, kind, bytes, payload_type, manifest, local, remote, repo_path}`.

## 6. Tools, derivations, conditions, closures, scratch

### 6.1 Tool declaration (`research.store.tool`)

~~~python
from research.store.tool import Tool

TC_PROBE = Tool(
    name="tc_probe", version="1",
    command=["python", "tools/tc_probe/tc_probe.py"],
    closure=["tools/tc_probe/**", "packages/verity/src/verity/ml/tc/**"],   # repo-relative globs -> file manifest at run time
    key_params=["instruction", "families", "seed", "n_random", "replay_golden"],
    key_conditions=["hw.family"],
    param_artifacts=["replay_golden", "replay_instances"],                     # path-valued params: identity = content, never pathname
    caches=["nvcc"],                                                            # names for d.scratch(name)
    produces=["tc-evidence/v1", "run-files/v1"],
)
~~~

Declarations live next to the tool they describe (`tools/tc_probe/tool.py`, `backends/direct/ligero/tool.py`; the example above is
illustrative, the files follow the scripts' real argparse) and are found by `research run --tool <dotted.module:NAME>` or
`--tool <name>` via the registry in `research/store/tools_registry.py` (a dict name -> dotted path; keep it tiny). A declaration
imports nothing but `research.store.tool`: it must load on the control machine and in a shipped source tree alike (`load` falls
back to executing `<source root>/<module path>.py` by file when the module is not on sys.path).

### 6.2 Params

`research run` records the **full parsed argv** (`params.argv`) plus, when the Tool provides `parse(argv) -> dict`, every
parsed parameter. `key_params` selects the subset entering the derivation. A path-valued key param is replaced by the
content identity of what it points at: an existing artifact id if the path is a fetched artifact, else `sha256:<hex>` of the
file (or tree) — never the pathname. Non-key params are still recorded and queryable.

### 6.3 Conditions — `research.env.conditions()`

One canonical baseline collected by the runner for every attempt; lanes never hand-write it. Dotted keys, scalar values,
`null` when unknown (never omitted, never guessed):

~~~text
source.commit  source.tree  source.dirty  source.dirty_digest  source.lockfiles.<name>
rt.os  rt.kernel  rt.python  rt.rust  rt.cuda  rt.driver  rt.torch  rt.triton  rt.nvcc
hw.family  hw.sku  hw.count  hw.vram_mb  hw.cpu  hw.hostname
exec.exclusive  exec.load_start  exec.load_end  exec.project  exec.campaign  exec.runs_root
~~~

`hw.family` comes from ONE SKU->family map inside `research/env.py` (`SKU_FAMILY`: `h100`, `h200`, `a100`, `l40s`, `l40`,
`rtx4090`, `rtx5090`, `b200`, `a10`, `t4`; `cpu` when there is no GPU, `unknown` for a GPU the table does not know, `null` when the
probe itself failed); nobody else normalises. Tools may add keys under `tool.<name>.<key>` via `Tool.conditions(env) -> dict`.
`source.lockfiles.<name>` is one key per lockfile found at the source root (`uv.lock`, `Cargo.lock`), hashed from the files
themselves; `env.source_identity(conds)` gives the nested `source` block used by the manifest and the reuse key. A shipped
`git archive` has no `.git`: the control side passes the commit/tree/dirty it knew (`research run --source-identity`, added by
`--on --source`) and the machine fills only what git could not tell it. `rt.torch` / `rt.triton` are the runner interpreter's
distributions (`importlib.metadata`, never imported); a workload in another venv may differ.

### 6.4 Derivation id and reuse key

~~~python
derivation_doc = {"tool": {"name": ..., "version": ...}, "inputs": {name: art_id}, "params": {k: key params, content-resolved},
                  "conditions": {k: conditions[k] for k in key_conditions}}
drv_id    = "drv:" + canon.sha256(derivation_doc)
reuse_key = canon.sha256({"derivation": drv_id, "source": {"tree": source.tree, "lockfiles": source.lockfiles}})  if not dirty else None
~~~

`lookup` matches on `reuse_key` (exact source + exact request: conservative, false misses not false hits).
`select --derivation drv:…` and `diff` group on `drv_id` across revisions. Closure-scoped reuse (a third key from the recorded
closure manifest) is a later, explicit optimisation once a Tool's closure has earned trust — it must not be the default.
The Triton cache incident (W=31 binaries served to a W=22 tree) is why.

### 6.5 Scratch / caches

`d.scratch(name)` -> `<store>/scratch/<reuse_key>/<name>/` (for dirty trees: `<store>/scratch/dirty-<run id>/<name>/`).
`research run --scratch triton` exports `TRITON_CACHE_DIR` there; `--scratch <name>` exports `RESEARCH_SCRATCH_<NAME>`.
Keyed by reuse_key ON PURPOSE: every commit gets a cold cache; the recompile cost is the price of the safe bias. Do not
"fix" this to derivation_id or a source-independent key without a measured reason and a per-tool `cache_scope` declaration.

## 7. Remote and replicas

~~~python
class Remote(Protocol):
    def put(self, key: str, path: Path) -> None            # upload under an immutable key; RemoteConflict if the key holds a different size
    def get(self, key: str, path: Path) -> None            # KeyError when absent
    def head(self, key: str) -> dict | None               # {"bytes": n} or None
    def list(self, prefix: str) -> list[str]
    def describe(self) -> str                              # for `where`
~~~

`FsRemote(root)` (a directory, also the test double) and `S3Remote(endpoint, bucket, access_key, secret_key, region="auto")`
(SigV4 with signed payload hashes, stdlib `urllib`+`hmac`, path-style `<endpoint>/<bucket>/<key>`; single PUT for objects
<= 4 GiB; above that `put` raises `NotImplementedError` — the planned design is multipart with per-part sha256 recorded in a
sidecar key `<key>.parts.json`). The signer is tested against the AWS documented vectors; R2 itself has not been exercised yet.
Config `~/.research/store.toml` (or `$RESEARCH_STORE_CONFIG`):

~~~toml
[remote]
type = "s3"                       # or "fs" with path = "/vault/store"
bucket = "verity-research"
endpoint = "https://<account>.r2.cloudflarestorage.com"
region = "auto"                   # optional
access_key_id_env = "RESEARCH_R2_ACCESS_KEY_ID"        # names of env vars; the file never holds secrets
secret_access_key_env = "RESEARCH_R2_SECRET_ACCESS_KEY"
session_token_env = "AWS_SESSION_TOKEN"                # optional (this is the default): set when the key pair is a temporary credential
~~~

A temporary credential (R2 temp-access credentials, STS-style) is a key pair plus a session token; when the named env var is set
the token is sent as `x-amz-security-token` inside the SigV4 signed headers on every request, when unset nothing changes.

No remote configured (no file, or no `[remote]` table) => the local store is the only replica and `where` says so; nothing
fails, nothing is called PRESERVED (`verify` returns `preserved=False, reason="no remote configured"`; `push` refuses). A
`[remote]` table that is present but unusable (unknown type, missing field, credential env vars unset) is `ConfigError` from
`load_remote`, on purpose: a broken vault must not look like "no vault". `LocalStore` keeps that distinction without making the
local store unusable: `store.remote` degrades to `None` and keeps the reason in `store.remote_error`; reads, puts, labels and
attempt publication proceed local-only, and `push` / `push_attempt` / `research data push` raise the reason (e.g. which env vars
to set). `research run` reports which of the three happened: `PRESERVED on the remote`, `NOT preserved: <error>`, or `local only
(<reason>)`; `research data push RUN` (or `--pending`) preserves later, with credentials in the environment
(`set -a; source ~/.config/verity/r2.env; set +a` for the R2 vault). `where` still answers for the local and repo replicas and reports the remote row as `UNUSABLE <error>` (never "not configured", never absent).

**PRESERVED** means: remote `head` size matches AND a full read-back sha256 matches, for the manifest and every blob (objects
< 4 GiB; above that, per-part sha256 from the sidecar checked against the parts read back — not implemented). ETags are not
content hashes. `research data verify ART` performs it (exit 1 unless PRESERVED); `put --preserve` performs it after upload;
`LocalStore.push(art, preserve=True)` returns the verify report; the `replicas` table records `verified_utc` per location.

Replica resolution order for `fetch`: local objects -> repo replica (`fixtures/artifacts.json`: `{art id: repo-relative
path}` of the payload — a file, or a tree's directory; every blob is sha256-checked against the manifest before use, mismatch
warns on stderr and falls through) -> remote (downloaded bytes are checked too: `IntegrityError` on mismatch). The repo map
supplies bytes only; the manifest must still be resolvable locally or remotely. Bytes that arrive from a repo or the remote are
cached into `objects/`. `research data pull --from MACHINE RUN`
copies an attempt manifest plus its outputs' manifests and blobs from a machine's local store over ssh (uses
`research.remote` helpers) — the same-host tier of the planner until R2 is configured.

## 8. Labels

`labels/<target>/<ts>-<by>.json`:

~~~json
{"schema": "research/label/v0.1", "target": "art:...", "key": "proof_class", "value": "COMPLETE_ZK_BACKEND",
 "by": "red-team-zk-2", "ts": "2026-09-22T11:20:29Z", "ref": "art:<findings>|r<run id>|<url>"}
~~~

Append-only, one file per assertion (`O_EXCL`, mode 0444), never read-modify-write. The file name uses the compact ts
(`20260922T112029Z`), `by` restricted to `[A-Za-z0-9._-]`, and a `~N` suffix when the same asserter labels twice in one
second (`labels()` orders by ts, then file name with `~N` read numerically, so history order survives past `~9`). Several
assertions that belong together (one ledger row's fields, one audit's verdicts) share `by`, `ts` and `ref`; `ref` is the key
to group them by when one target carries several rows. `research data labels
TARGET` shows the latest assertion per (key, by); `--history` shows every one in ts order; `--key K --by PATTERN` (fnmatch)
filter. `label TARGET KEY VALUE` stores VALUE as a string unless `--value-json`. The store never picks a winner; decision
tables do (`proof_class=… by=red-team-*`).

## 9. CLI (v0.1)

~~~text
research data put  --kind K [--meta JSON|@file] [--ref name=art:...]... [--file PATH | --tree DIR] [--preserve]   -> art id
research data show ART|ATTEMPT|DRV [--json]   manifest (+ producing attempts) / attempt / attempts with that derivation
research data where ART [--json]              replicas: local / repo / remote (+ verified_utc)
research data fetch ART [--to DIR]            materialise payload (file or tree); prints path
research data verify ART [--json]             PRESERVED check against the remote (exit 1 unless PRESERVED)
research data push [ART|RUN ...] [--pending] [--json]
                                              preserve on the remote: blobs + manifest uploaded and read back (PRESERVED) per artifact;
                                              a run id pushes the attempt manifest and every artifact it references; --pending pushes
                                              every local artifact without a verified remote replica and every attempt the remote lacks
research data lookup --tool NAME [--param k=v]... [--input name=art:...]... [--condition k=v]... [--source-tree SHA] [-- ARGV...]
                                              -> attempts with that reuse_key (or --any-source: that derivation); ARGV after -- is
                                                 parsed by the Tool exactly as `run` would, --param overrides; missing key conditions
                                                 come from this machine's conditions(); --source-tree hashes that tree's lockfiles
research data select [--kind K] [--where meta.key=value]... [--tool T] [--derivation D] [--attempt R] [--label k=v] [--json|--ids]
research data snapshot --name NAME (--members art:... | --select <select flags>) [--note TEXT]  -> art id (dataset-snapshot/v1)
research data labels TARGET [--history] [--key K] [--by PAT] [--json]
research data label  TARGET KEY VALUE --by WHO [--ref REF] [--value-json]
research data attempt publish RUN_DIR|RUN_ID [--tool ...] [--force] | --all [--runs-dir D]   (what `research run` calls; after the fact for old runs)
research data pull --from MACHINE RUN --project P    (the machine is bound to a project, as for `run --on`)
research data reindex [--remote]
research data sql "SELECT ..." [--json]      power-user escape hatch; read-only connection
~~~

No query DSL: flags for common facets + `sql`. Every command takes `--store ROOT`. `--ref name=a,b` or a repeated name makes a
list. `--members` repeats or comma-separates; `--select` takes the rest of the line as select flags and records them as
`meta.query` (so a snapshot frozen by query and one frozen by an explicit member list differ). `put --preserve` exits 1 unless
PRESERVED (the id is still printed). `research/store/cli.py` exposes `COMMANDS: dict[str, fn(argv) -> int]` and `main(argv)`;
`cli_prov.py` (lookup / pull / attempt) is merged into `COMMANDS` when importable. `research/cli.py` dispatches `data`.

## 10. `research run` integration

At completion or failure, in the runner (local and machine side alike), `store.attempt.publish(run_dir, tool=...)`:

1. `result.json`, if valid, becomes an artifact of kind `result.kind` (default `bench-result/v1`; `tc_probe` writes
   `tc-evidence/v1`): `meta` = the result document minus `artifacts`/`measurement_files` paths, `refs.run_files` = the
   run-files artifact. Paths listed in `result.artifacts` + `measurement_files` become ONE `run-files/v1` tree artifact.
2. `events.jsonl` / `resources.jsonl` become `telemetry/v1` file artifacts when present.
3. The attempt manifest (§4) is written to `attempts/<run id>.json` and indexed.
4. With a remote configured: blobs and manifests are uploaded and verified; `execution.preserved` is recorded per output.

`publish` never raises into the runner: a failure is written to `<run dir>/store_publish_error.json` and reported on one line;
the workload's exit status stands. It is idempotent (an attempt already in the store is a no-op) and refuses a run whose
status.json is missing or not terminal (`--force` overrides, for the CLI only).
`research fetch RUN --all` continues to work as before; `research data pull` complements it by moving store objects.
Active run state stays in the run registry; the store only ever sees completed/failed attempts.

## 11. Known kinds (`kinds.py`)

~~~text
bench-result/v1          a result.json (measurements, validation, fingerprint) + refs.run_files, refs.proof?
tc-evidence/v1           one tensor-core probe capture: meta {instruction, arch, sku, families, counts, model, model_agrees,
                         first_mismatches, seed, n} ; payload tree = probe_results.json, capture_sample.json, tiles_*.npz
dataset-snapshot/v1      meta {name, note, query?}; refs.members = [art:...]; payload null
verification-verdict/v1  meta {result: PASS|FAIL, verifier, detail}; refs.proof, refs.verifier_attempt
proof/v1                 payload = proof bytes; meta {statement_digest, backend, params}
fixture/v1               a repo-replicated input (golden records, instances); payload file/tree
run-files/v1             the files a result.json pointed at, as one tree
telemetry/v1             events.jsonl or resources.jsonl of an attempt
redteam-findings/v1      meta {findings: [...]}; refs.target
ledger-row/v1            one frozen overhead-ledger row whose run is not an attempt here: meta = the ledger/v1 row; payload null;
                         the lane's claims are labels on it (`verity_numerical.bench.ledger_to_store`)
~~~

## 12. Design rules

1. **A derivation key is a claim of dependency completeness.** Bias to false misses. Exact reuse includes the source tree.
2. **Immutable objects only.** Blobs, manifests, attempts, labels are write-once; the index is disposable.
3. **Labels are assertions, not fields.** Keep every assertion; policy chooses.
4. **Verify uploads.** Size + content read-back before anything is called PRESERVED. Never trust ETags.
5. **Producers never carry verifier truth.** A proof artifact may hold proof bytes, the claimed statement, prover metadata;
   the verifier's chain/spec/keys come from its own closure or an explicit trusted input artifact (the `chain.txt` lesson).
6. **Content, not pathname**, is the identity of every input.
7. **`conditions()` owns normalisation.** One SKU->family map.
8. **Canonical JSON everywhere** ids are computed.

## 13. Deferred (each arrives with a named failure)

`diff` (F9), `independent` (U3, descriptive not authoritative: shared closure files / lockfiles / inputs), `impact` (F4),
`export` (F5/U5), `gc` (F11), pod teardown gating on PRESERVED (F7), closure-scoped reuse, replica planner beyond the
three-tier order above.
