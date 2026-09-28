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
reuse_key = sha256 hex of canon({"derivation": drv_id, "source": <source identity>})   None for dirty trees and undeclared runs
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
steward/lease.json                   REMOTE ONLY, not a store record: the steward lease, the one mutable key, changed by conditional
                                     PUT only (research/steward.py); the store never reads, lists or indexes it
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
`key_params = ["argv"]`, `key_conditions = []`, so `derivation = drv(H(tool name, inputs, {"argv": redacted argv}, {}))`: the
whole command line is the request, and `reuse_key` is `null` whatever the tree (argv cannot name the files, environment or data
the command reads, so nothing may be served as its reuse). `params` then holds `{"argv": argv}`. The rule used to be
`key_params = []` with a reuse key: every undeclared `python …` run of one clean tree shared one derivation and one reuse key.
Publishing applies the rule to a record written by an older runner (or with no `store` section): it is re-derived.

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
replicas(id, location, verified_utc, detail, verified)   verifications (§7: verified = etag-md5 | sha256-readback); the one table a rebuild
                                                          keeps, since it is not a local fact; `verified` is added to an older catalog on open
~~~

Flattening: nested objects get dotted keys, lists numeric components (`families.0`), scalars are stored as text (strings raw,
everything else `json.dumps`: `true`, `null`, `1.5`); `--where` and `--label` compare that text. `local` means manifest + every
blob present here; `remote` means pushed (from a listing or a recorded push/verification).

`research.store.index.Index(path)`: `index_artifact(manifest_doc, art_id, local=None, remote=None, repo_path=None)` (None keeps the
current flag), `index_attempt(doc)`, `index_label(label, path)`, `set_presence(art_id, *, local, remote, repo_path)`,
`set_replica(art_id, location, verified_utc, detail, verified=None)`, `rebuild(artifacts, attempts, labels)`, `artifact(id)`, `artifacts()`,
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

`d.scratch(name)` -> `<store>/scratch/<reuse_key>/<name>/` (no reuse key -- a dirty tree, or no declared Tool:
`<store>/scratch/run-<run id>/<name>/`).
`research run --scratch triton` exports `TRITON_CACHE_DIR` there; `--scratch <name>` exports `RESEARCH_SCRATCH_<NAME>`.
Keyed by reuse_key ON PURPOSE: every commit gets a cold cache; the recompile cost is the price of the safe bias. Do not
"fix" this to derivation_id or a source-independent key without a measured reason and a per-tool `cache_scope` declaration.

## 7. Remote and replicas

~~~python
class Remote(Protocol):
    def put(self, key: str, path: Path) -> dict            # conditional create under an immutable key -> {"action": created|skipped|exists, ...};
                                                           # RemoteConflict if the key holds a different size or (single-part ETag) MD5
    def get(self, key: str, path: Path) -> None            # KeyError when absent
    def head(self, key: str) -> dict | None               # {"bytes": n, "etag": md5 hex | None} or None
    def list(self, prefix: str) -> list[str]
    def describe(self) -> str                              # for `where`
~~~

`FsRemote(root)` (a directory, also the test double; its `etag` is the file's MD5) and `S3Remote(endpoint, bucket, access_key,
secret_key, region="auto", session_token=None)` (SigV4 with signed payload hashes, stdlib `urllib`+`hmac`, path-style
`<endpoint>/<bucket>/<key>`; one PUT for objects <= 5 GiB, the service's PutObject limit; above that `push` raises `StoreError`
naming multipart as unimplemented — the planned design is multipart with per-part sha256 recorded in a sidecar key
`<key>.parts.json`). `S3Remote.put` is HEAD-then-`PUT If-None-Match: *`: a 412 means another writer created the key first
(`exists`, fine for immutable content); a 400/501 means the service lacks conditional PUT, so it is switched off for the rest of
the session and the plain PUT is used (the push report carries a note). Cloudflare R2 honours `If-None-Match: *` (checked live).
The signer is tested against the AWS documented vectors and the client against R2. Config `~/.research/store.toml` (or
`$RESEARCH_STORE_CONFIG`):

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

Without that file (and with `$RESEARCH_STORE_CONFIG` unset) the environment stands in: `R2_ENDPOINT`, `R2_BUCKET` (must equal
the bucket `tools/research/store.pod.toml` names), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. A file that exists wins.

A temporary credential (R2 temp-access credentials, STS-style) is a key pair plus a session token; when the named env var is set
the token is sent as `x-amz-security-token` inside the SigV4 signed headers on every request, when unset nothing changes.

### 7.1 Temporary credentials for pods — `research data mint-credential`

A GPU pod gets a short-lived, bucket-scoped credential instead of the parent R2 key (`credentials.py`, stdlib):

~~~text
research data mint-credential [--ttl 6h] [--permission object-read-write|object-read-only|admin-*] [--prefix P]... [--object KEY]...
                              [--actions PutObject,HeadObject,GetObject] [--via auto|local|api] [--env] [--endpoint U] [--bucket B] [--account-id A]
~~~

Inputs come from flags > `store.toml [remote]` (endpoint, bucket, the env var *names* of the parent key) > `R2_ENDPOINT` /
`R2_BUCKET` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`; the account id from `R2_ACCOUNT_ID` / `CLOUDFLARE_ACCOUNT_ID` or the
`<account>.r2.cloudflarestorage.com` host. Two paths, `auto` preferring the first:

- **local** (no API call): sign a JWT (HS256) with the parent *secret access key* — Cloudflare's documented client-side signing,
  ported claim for claim from their reference implementation: `{bucket, scope | actions, paths?: {prefixPaths, objectPaths}, sub:
  account id, iss: parent key id, aud: endpoint host, iat, exp}`; the temporary key id is the parent's, the temporary secret is the
  sha256 hex of the JWT, the session token is `base64("jwt/" + JWT)`. Checked live against `verity-dev` (2026-09-22): a 10-minute
  `object-read-only` credential HEADs a manifest (200) and is refused a PUT (403); `--prefix` + `--actions` are enforced (PUT inside
  the prefix 200, HEAD outside it 403, a GET not in `actions` 403). One deviation from the reference TypeScript: R2 answers 400 to a
  JWT that carries both `scope` and `actions`, so `--actions` *replaces* `--permission` in the claims. Only this path can scope by
  action.
- **api**: `POST /client/v4/accounts/{account}/r2/temp-access-credentials` with `CLOUDFLARE_API_TOKEN` (body `{bucket,
  parentAccessKeyId, permission, ttlSeconds, prefixes?, objects?}`), used when the parent secret is not in the environment. Unit-tested
  against a fake transport; not exercised live (no API token in `r2.env`).

TTL <= 7 days. Output is JSON (`accessKeyId`, `secretAccessKey`, `sessionToken`, `expires_utc`, `via`, `bucket`, `scope`, ...) or, with
`--env`, the three `export AWS_ACCESS_KEY_ID= ... AWS_SECRET_ACCESS_KEY= ... AWS_SESSION_TOKEN=` lines for a pod bootstrap; nothing is
ever written to disk by the tool. A pod with those three variables and the same `store.toml` pushes through `S3Remote` as usual
(the token rides in the signed headers, above). A temporary credential cannot mint another (the tool refuses when
`AWS_SESSION_TOKEN` is set); revoking the parent token kills every child.

No remote configured (no file, or no `[remote]` table) => the local store is the only replica and `where` says so; nothing
fails, nothing is called PRESERVED (`verify` returns `preserved=False, reason="no remote configured"`; `push` refuses). A
`[remote]` table that is present but unusable (unknown type, missing field, credential env vars unset) is `ConfigError` from
`load_remote`, on purpose: a broken vault must not look like "no vault". `LocalStore` keeps that distinction without making the
local store unusable: `store.remote` degrades to `None` and keeps the reason in `store.remote_error`; reads, puts, labels and
attempt publication proceed local-only, and `push` / `push_attempt` / `research data push` raise the reason (e.g. which env vars
to set). `research run` reports which of the three happened: `PRESERVED on the remote`, `NOT preserved: <error>`, or `local only
(<reason>)`; `research data push RUN` (or `--pending`) preserves later, with credentials in the environment
(`set -a; source ~/.config/verity/r2.env; set +a` for the R2 vault). `where` still answers for the local and repo replicas and reports the remote row as `UNUSABLE <error>` (never "not configured", never absent).

**PRESERVED** means: for the manifest and every blob, the remote holds an object of the right size whose content was checked,
by one of two checks, and the replica record (`replicas.verified`) says which:

~~~text
etag-md5          `push --verify head` (default): HEAD after upload; Content-Length == bytes and the single-part ETag == hex MD5 of
                  the local bytes (S3 and R2 make a single-PUT object's ETag its MD5; a `-` in the ETag marks multipart, which is not
                  an MD5, and such an object falls back to read-back).  One extra round trip per object.
sha256-readback   `push --verify readback`, `verify`, `put --preserve`'s manifest: download and sha256 against the manifest.  Costs
                  the object's bytes again.  Manifests are always read back (they are small); blobs on request.
~~~

`LocalStore.push(art, preserve=True, jobs=32, verify="head")` uploads the blobs with `jobs` threads (per blob: HEAD; present with
the right size and MD5 -> skipped and thereby verified; else conditional PUT, then the check), and the manifest **after** every
blob is present and verified — a manifest is never visible before its blobs, and a failed blob withholds it. One failed object
makes the artifact NOT preserved without aborting a `research data push` batch. The report has the `verify` shape (`preserved`,
`checked`, `errors`, `verified_utc`) plus `verified`, `objects`, `uploaded`, `skipped`, `bytes`, `seconds`, `notes`; results are
in manifest order whatever the completion order. `research data verify ART` is the standalone read-back check (exit 1 unless
PRESERVED); the `replicas` table records `verified_utc` and `verified` per location (`where` shows `verified_utc`).

Replica resolution order for `fetch`: local objects -> repo replica (`fixtures/artifacts.json`: `{art id: repo-relative
path}` of the payload — a file, or a tree's directory; every blob is sha256-checked against the manifest before use, mismatch
warns on stderr and falls through) -> remote (downloaded bytes are checked too: `IntegrityError` on mismatch). The repo map
supplies bytes only; the manifest must still be resolvable locally or remotely. Bytes that arrive from a repo or the remote are
cached into `objects/`. A repo fixture is registered bare (`fixture/v1`, `meta {}`, `refs {}`), so its id is its bytes (plus the
bare file name); when the bytes change on purpose, `research data refresh-fixtures [--preserve]` re-registers it and moves the old
id into the map's `previous` block (`{"previous": {"art:<new>": ["art:<old>", ...]}}`), which readers skip: `replica_map()` takes
`art:` keys only. `research data pull --from MACHINE RUN`
copies an attempt manifest plus its outputs' manifests and blobs from a machine's local store over ssh (uses
`research.remote` helpers) — the same-host tier of the planner until R2 is configured.

### 7.2 Remote-first: write-through, verdicts, refresh, parity

- **Write-through.** With `[remote] write_through = true` in store.toml, or `RESEARCH_WRITE_THROUGH=1`, or (no store.toml, no
  `$RESEARCH_STORE_CONFIG`) `R2_ENDPOINT` + `R2_BUCKET` in the environment, `research data label` and `put` upload in the same
  command. The local write stands whatever the upload does; the command says `on <remote>`, `NOT preserved (<why>)` or `local only
  (write-through off)`, never nothing. Attempt publishing already uploads whenever a remote is usable. Off on the laptop until its
  store.toml sets the flag.
- **Pending queue.** `research data pending` lists what this store holds that the remote lacks (one listing per prefix): artifacts
  with no verified remote replica whose manifest the remote lacks or whose blobs are all local, attempts, assertions. `push
  --pending` drains exactly that. An artifact pulled from the remote without its blobs is the remote's already and is not queued.
- **Verdicts do not depend on the uploader.** `preserved --mode recorded` (the FINAL check) trusts a recorded verified replica and,
  where there is none (a pod's upload), runs the `head` check against the remote, which records the replica when it passes.
  `head_fallback=False` keeps it offline (the reaper, which bounds what it reads back). Abbreviated `art:` ids the local catalog
  does not know are resolved against the remote's `manifests/<prefix>` listing.
- **Refresh.** `research data refresh` fetches only the manifests (re-hashed to their ids), attempts and assertions this store
  lacks, and indexes just those. A fresh machine builds its catalog this way.
- **Atomic rebuild.** `reindex` builds the new catalog beside the old one, carries the replica records over, and renames it into
  place once complete; an open `LocalStore` notices the new file and reopens it.
- **Parity.** `research data parity --fresh DIR [--render] [--json-out F]` (`parity.py`) builds a store at DIR from the remote alone
  and compares it with this one on content: ids and manifest bytes, attempt documents, label content hashes, every verified replica
  row confirmed by HEAD (manifest + blobs, sizes), and with `--render` the `bench.tables` markdown. Every difference gets a cause
  (`pending here`, `not pulled`, `uploaded during the run`); anything else fails (exit 1). Read-only for the store under test.

## 8. Labels

`labels/<target>/<ts>-<by>.json` locally; on the remote `labels/<target>/<sha256 of the canonical assertion>.json` (content key,
immutable, append-only; a target's assertion set is the listing of its prefix):

~~~json
{"schema": "research/label/v0.1", "target": "art:...", "key": "proof_class", "value": "COMPLETE_ZK_BACKEND",
 "by": "red-team-zk-2", "ts": "2026-09-22T11:20:29Z", "ref": "art:<findings>|r<run id>|<url>"}
~~~

Assertions are durable with what they describe (`labels_sync.py`): `push RUN` / `attempt publish` / `push ART` push the assertions
about the attempt, its derivation and its named artifacts (`attempt_targets`); `push --pending` and `labels-sync` push every local
one; `reindex --remote`, `labels-sync` and `pull --from MACHINE` bring remote / machine assertions into the local layout and index.
`preserved` is a three-leg gate — attempt manifest, artifacts, labels — and names the leg that fails; a preserved Attempt is not
preserved until its assertions are on the remote. `labels TARGET --remote` shows each assertion as both / LOCAL ONLY / REMOTE ONLY.
Conflicting assertions coexist on both sides.

Append-only, one file per assertion (`O_EXCL`, mode 0444), never read-modify-write. The file name uses the compact ts
(`20260922T112029Z`), `by` restricted to `[A-Za-z0-9._-]`, and a `~N` suffix when the same asserter labels twice in one
second (`labels()` orders by ts, then file name with `~N` read numerically, so history order survives past `~9`). Several
assertions that belong together (one ledger row's fields, one audit's verdicts) share `by`, `ts` and `ref`; `ref` is the key
to group them by when one target carries several rows. `research data labels
TARGET` shows the latest assertion per (key, by); `--history` shows every one in ts order; `--key K --by PATTERN` (fnmatch)
filter. `label TARGET KEY VALUE` stores VALUE as a string unless `--value-json`. The store never picks a winner; decision
tables do (`proof_class=… by=red-team-*`).

### 8.1 Vocabulary (`vocab.py`)

The store validates a label's envelope only; the *words* are pinned in `research/store/vocab.py` so a producer, a verifier and
a table read the same key. One constant per key (`vocab.PROOF_CLASS`, never the string), grouped by who asserts it, each with a
value form: `enum` (a closed set), `bool` (`true` | `false`), `int`, `number`, `text`, `list`. `research data vocab [--json]`
prints the table; the groups are:

- **result** — the producing lane about its own result (`ledger_to_store`, or `label` by hand): `candidate` (enum: the backend
  families A-GKR, B-Ligero, C-Flock, D-SP1, and the frozen ledger's B-AIR, C (the VOLE candidate), SP1 (D-SP1) and shared),
  `proof_class` (enum: the contract's classes + COMPONENT), `soundness`, `K`, `B`, `hardware`,
  `authentication` (enum: included | included-hash | excluded), `campaign`, `relation`, `mode` (enum: interactive | fiat-shamir), `zk` (bool),
  `overhead`, `seconds_per_vu`, `track` (enum), `scope` (enum: unit | vu | component), `label`, `note`, `assumptions` (list),
  `omitted` (list), `domain` (`total` | `finite-only`: the operand domain the statement covers), `breakthrough` (bool); grouping and configuration a table selects on: `lane`, `row`, `sweep`, `arm`, `stage`,
  `source`, `superseded_by`, `interaction_attempt_of` (the cell an interaction attempt belongs to), `hash`, `sharing`, `fold` (int), `pipeline` (int).
- **verification** — an independent verifier, never the producer: `verified` (enum: accepted | rejected | not-transferable |
  no-dumps), `verifier`, `verifier_seconds`, `same_device` (bool), `below_bar` (bool: verified, but the whole-proof bound misses the tables' 2^-128 bar,
  so it is shown in the below-bar drill-down and never as a Table 2 cell), `outcome`; `independently_verified` (bool) is the back-fill's
  spelling of `verified` — accepted by every reader, deprecated for new writes; `verified_by` is a ledger row's free text and
  carries no verdict — ignored.
- **evidence** — probe captures and input sets: `instruction`, `model`, `model_agrees` (bool), `model_status`,
  `model_rejected`, `evidence`, `family`, `instances_dataset`, `instances_tier`, `mismatches`, `target`, `component`, `jobs`, `dtype`.

Enum and bool values are compared as the catalog stores them (`scalar_text`: strings raw, everything else JSON), so `true` and
`"true"` are one word and `"True"` is another. `research data label` writes the canonical spelling of an old key or value
(`axis` -> `sweep`, `tree` -> `source`, `batch` -> `B`) and refuses — exit 2, nothing written, the reason on stderr — an unknown key
(naming the nearest known one), a measurement (`l`, `subbatches`, `rtt_ms_path`, ...: the number belongs in the result or the
artifact's meta), a ref written as a label, a value outside its enum, or a word where the form is bool / int / number.
`--off-vocab` writes it anyway, for a concept the vocabulary has not learnt yet; add the word to `vocab.py` next. Readers stay
lenient: whatever older writers stored is still read, old spellings as their canonical key. Consumers (`bench.tables`, the trust
module) import the constants; the store imports no consumer.

## 9. CLI (v0.1)

~~~text
research data put  --kind K [--meta JSON|@file] [--ref name=art:...]... [--file PATH | --tree DIR] [--preserve]   -> art id
research data show ART|ATTEMPT|DRV [--json]   manifest (+ producing attempts) / attempt / attempts with that derivation
research data where ART [--json]              replicas: local / repo / remote (+ verified_utc)
research data fetch ART [--to DIR]            materialise payload (file or tree); prints path
research data verify ART [--json]             PRESERVED check against the remote (exit 1 unless PRESERVED)
research data push [ART|RUN ...] [--pending] [--jobs N] [--verify head|readback] [--json]
                                              preserve on the remote: blobs uploaded with N threads (default 32), each verified (head:
                                              size + MD5 ETag, default; readback: download + sha256), then the manifest (PRESERVED); one
                                              line per artifact with objects, uploaded/skipped, MB, seconds, objects/s and the check done;
                                              a run id pushes the attempt manifest and every artifact it references; --pending pushes
                                              every local artifact without a verified remote replica and every attempt the remote lacks
research data lookup --tool NAME [--param k=v]... [--input name=art:...]... [--condition k=v]... [--source-tree SHA] [-- ARGV...]
                                              -> attempts with that reuse_key (or --any-source: that derivation); ARGV after -- is
                                                 parsed by the Tool exactly as `run` would, --param overrides; missing key conditions
                                                 come from this machine's conditions(); --source-tree hashes that tree's lockfiles
research data select [--kind K] [--where meta.key=value]... [--tool T] [--derivation D] [--attempt R] [--label k=v] [--json|--ids]
research data snapshot --name NAME (--members art:... | --select <select flags>) [--note TEXT]  -> art id (dataset-snapshot/v1)
research data labels TARGET [--history] [--key K] [--by PAT] [--remote] [--json]
                                              --remote: both / LOCAL ONLY (not durable) / REMOTE ONLY per assertion; exit 1 if any is local only
research data label  TARGET KEY VALUE --by WHO [--ref REF] [--value-json] [--off-vocab]   refuses what the vocabulary (§8.1) cannot place
research data preserved (RUN|ART)... [--mode head|readback|recorded] [--json]
                                              the durability gate: attempt manifest + artifacts + labels all on the remote, else the failing legs
research data vocab [--json]                  the label vocabulary (§8.1): key, asserter group, value form, enum values, status
research data refresh-fixtures [--preserve] [--dry-run] [--repo DIR] [--json]
                                              re-register every fixtures/artifacts.json entry whose bytes changed (§7): put as a bare
                                              fixture/v1 under the new id (--preserve pushes it), rewrite the entry, keep the old id under
                                              the map's `previous` block; prints old -> new; idempotent; exit 1 on a missing path or conflict
research data doctor [--clean] [--runs-dir D] [--json]
                                              run dirs holding a store_publish_error.json (§10): stale (the run is an attempt in this store)
                                              vs REAL (it is not); read-only unless --clean, which deletes the stale files and nothing else
research data attempt publish RUN_DIR|RUN_ID [--tool ...] [--force] | --all [--runs-dir D]   (what `research run` calls; after the fact for old runs)
research data pull --from MACHINE RUN --project P    (the machine is bound to a project, as for `run --on`)
research data reindex [--remote]
research data sql "SELECT ..." [--json]      power-user escape hatch; read-only connection
research data mint-credential [--ttl 6h] [--permission P] [--prefix P]... [--actions A,B] [--via auto|local|api] [--env]   (§7.1)
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

### 10.1 Run custody on the remote — `research run --on M` (`--custody-r2`, the default; `custody.py`)

With `--custody-r2` (the default with `--on`; `--no-custody-r2` opts out) the run's custody does not depend on the launcher's disk:
the launcher mints an action-scoped temporary key
(§7.1, local signing; `--custody-ttl`, default 6h, at most 24h, must outlast the run and its upload: `research run` refuses a `--timeout` longer than 6h without an explicit `--custody-ttl`; a run over 4 GiB gets a `size_warning`, since uploads don't resume; no `DeleteObject`/`CopyObject`) and stages it with a
secret-free store.toml under `<root>/requests/<run id>/custody/` (mode 600; never in request.json, remote.json or the workload's
environment). Without the parent key on the launcher the launch is refused. After the workload exits, the runner publishes the
attempt with one more output, `outputs.run_record`: a `run-record/v1` tree of every file in the run dir (stdout.log, stderr.log,
launcher.log, job.json, status.json, outputs; markers excluded), pushes and verifies it (`preserved.check`, head; retried), deletes
the key, and writes `<run dir>/.custody` (`research/run-custody/v0.1`: `verified_from`, the file list with sizes and sha256).
The marker is written only when every file now in the run dir is in the record by content; from the record on, the runner's own
output goes to `<root>/requests/<run id>/custody/publish.log`, not launcher.log, so nothing it prints can make the run dir differ
from its record. A failure whose key had expired says so in the marker's `error`. With `--no-custody-r2` attempts are unchanged
(no `run_record`).

- The pod guard accepts a valid `.custody` like a valid `.fetched`: every file listed, none changed at or after `verified_from`.
- `research data custody RUN [--publish]` checks the same predicate from any machine (and rewrites the marker); `--publish` puts a
  local run dir on the remote first. `research data custody --triage [--verify]` sorts a runs root into `fetched`, `r2`,
  `attempt-no-record`, `attempt-unpreserved`, `no-custody`, `running`, `never-ran`, `waived`; a waiver or never-ran verdict is a
  label: `research data label RUN custody waived|never-ran --by WHO`.
- `research pods drain|unpreserved --custody-r2` and `research notes watch --reap --custody-r2` (`pods.drain.pod_custody`) refresh
  the catalog from the remote, require every attempt tied to the pod to be preserved, and read the pod guard's state for its run
  dirs: without a guard state younger than 10 min, custody is unknown and nothing is terminated.

## 11. Known kinds (`kinds.py`)

~~~text
bench-result/v1          a result.json (measurements, validation, fingerprint) + refs.run_files, refs.proof?
bench-point/v1           a sweep harness's raw result for its chosen point (result.json kind bench-point/v1): kept by the
                         run's attempt with its run files; `bench.cell register` makes the cell's one bench-result/v1
tc-evidence/v1           one tensor-core probe capture: meta {instruction, arch, sku, families, counts, model, model_agrees,
                         first_mismatches, seed, n} ; payload tree = probe_results.json, capture_sample.json, tiles_*.npz
dataset-snapshot/v1      meta {name, note, query?}; refs.members = [art:...]; payload null
verification-verdict/v1  meta {result: PASS|FAIL, verifier, detail}; refs.proof, refs.verifier_attempt
proof/v1                 payload = proof bytes; meta {statement_digest, backend, params}
fixture/v1               a repo-replicated input (golden records, instances); payload file/tree
run-files/v1             the files a result.json pointed at, as one tree
run-record/v1            every file of a run dir (logs included) as one tree; outputs.run_record of a --custody-r2 attempt (§10.1)
telemetry/v1             events.jsonl or resources.jsonl of an attempt
redteam-findings/v1      meta {findings: [...]}; refs.target
ledger-row/v1            one frozen overhead-ledger row whose run is not an attempt here: meta = the ledger/v1 row; payload null;
                         the lane's claims are labels on it (`verity_numerical.bench.ledger_to_store`)
trust-dossier/v1         the computed trust level of one numerical target (`python tools/tc_probe/trust.py --publish`): meta =
                         the TrustReport (level, criteria with evidence ids, D-entries, references, missing items, triggers, procedure);
                         refs.cited = every artifact it rests on, refs.previous = earlier dossiers; payload null; labels target,
                         instruction, level, anchor
~~~

## 12. Design rules

1. **A derivation key is a claim of dependency completeness.** Bias to false misses. Exact reuse includes the source tree.
2. **Immutable objects only.** Blobs, manifests, attempts, labels are write-once; the index is disposable.
3. **Labels are assertions, not fields.** Keep every assertion; policy chooses.
4. **Verify uploads.** Size + a content check before anything is called PRESERVED, and the record says which check: a single-part
   ETag is the object's MD5 and is accepted as the transport check (`etag-md5`); a multipart ETag is not a hash and never is; the
   manifest's sha256 stays the identity and read-back (`sha256-readback`) stays available for anything that matters more than time.
5. **Producers never carry verifier truth.** A proof artifact may hold proof bytes, the claimed statement, prover metadata;
   the verifier's chain/spec/keys come from its own closure or an explicit trusted input artifact (the `chain.txt` lesson).
6. **Content, not pathname**, is the identity of every input.
7. **`conditions()` owns normalisation.** One SKU->family map.
8. **Canonical JSON everywhere** ids are computed.

## 13. Deferred (each arrives with a named failure)

`diff` (F9), `independent` (U3, descriptive not authoritative: shared closure files / lockfiles / inputs), `impact` (F4),
`export` (F5/U5), `gc` (F11), pod teardown gating on PRESERVED (F7), closure-scoped reuse, replica planner beyond the
three-tier order above.
