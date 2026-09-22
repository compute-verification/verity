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

Two agents freezing the same member set produce the same snapshot id. Two attempts producing byte-identical outputs
produce the same artifact id; provenance lives on the Attempt (§4), not inside the artifact.

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

Tree `files` are sorted by `path`; paths are relative, no `..`, no absolute. Every file (and a file payload) is one blob at
`objects/sha256/<hex>`. `refs` values are `art:` ids or lists of them (the store checks the syntax, not existence — a ref may
point at an artifact that lives only remotely). `meta` may contain `art:` strings too; only `refs` is indexed as edges.

Manifest identity excludes producer, time, labels. The manifest file is `manifests/<sha256 hex>.json` (canonical bytes).

Kinds are strings `<name>/v<int>`. The store validates only the envelope; kind-specific validation is the producer's
(`kinds.py` holds the known kinds and a one-line description each; `put` warns on an unknown kind, never refuses).

## 3. Layout

Local store root: `~/.research/store` (or `$RESEARCH_STORE`). Remote uses the same relative keys.

~~~text
objects/sha256/<hex>                 blobs (content-addressed; write temp + rename; never overwritten)
manifests/<hex>.json                 artifact manifests (canonical bytes; hex == art id hex)
attempts/<run id>.json               attempt manifests (canonical bytes; immutable once published)
labels/<target>/<ts>-<by>.json       one immutable file per assertion; <target> = id with ':' -> '_'
snapshots are ordinary artifacts (kind dataset-snapshot/v1)
catalog.sqlite                       LOCAL ONLY: rebuildable index (§5); never uploaded
scratch/<reuse_key>/<name>/          LOCAL ONLY: tool caches namespaced by reuse_key (§6.5)
~~~

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
                "host": "vy-g5", "project": "verity", "campaign": "r21-silicon", "run_dir": "/workspace/research/runs/r..."},
  "cost": {"provider": "runpod", "pod_id": "...", "rate_usd_per_hour": 2.69, "usd": 0.0092, "shared": false},
  "status": {"state": "exited", "rc": 0, "class": null, "failure": null, "validation": "passed"},
  "outputs": {"result": "art:...", "run_files": "art:..."},
  "telemetry": {"events": "art:...|null", "resources": "art:...|null"}
}
~~~

Without a Tool declaration (`research run -- CMD` with no `--tool`): `tool = {"name": "<argv[0] basename>", "version": null}`,
`key_params = []`, `key_conditions = []`, `inputs = {}`, so `derivation = drv(H(tool name, {}, {}, {}))` — a real but coarse
derivation; `reuse_key` is still computed (clean tree) so `lookup` is honest. `params` then holds `{"argv": argv}`.

`attempts/<id>.json` is written once. Publishing an id that exists with different content is an error, never an overwrite.

## 5. Index — `catalog.sqlite` (rebuildable, local only)

`research data reindex` deletes and rebuilds it from `manifests/`, `attempts/`, `labels/` (local), then, with `--remote`,
from the remote listing too. Tables (denormalised; add columns freely, the file is disposable):

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
~~~

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

Declarations live next to the tool they describe (`tools/tc_probe/tool.py`, `backends/.../tool.py`) and are found by
`research run --tool <dotted.module:NAME>` or `--tool <name>` via the registry in `research/store/tools_registry.py`
(a dict name -> dotted path; keep it tiny).

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

`hw.family` comes from ONE SKU->family map inside `conditions()` (`h100`, `a100`, `l40s`, `rtx4090`, `rtx5090`, `b200`,
`cpu`, `unknown`); nobody else normalises. Tools may add keys under `tool.<name>.<key>` via `Tool.conditions(env) -> dict`.

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
    def put(self, key: str, path: Path) -> None            # upload under an immutable key; refuse to overwrite a different object
    def get(self, key: str, path: Path) -> None
    def head(self, key: str) -> dict | None               # {"bytes": n} or None
    def list(self, prefix: str) -> list[str]
~~~

`FsRemote(root)` (a directory, also the test double) and `S3Remote(endpoint, bucket, region="auto", access_key, secret)`
(SigV4, stdlib `urllib`+`hmac`; single PUT for objects <= 4 GiB, multipart above with per-part sha256 recorded in a sidecar
key `<key>.parts.json`). Config `~/.research/store.toml` (or `$RESEARCH_STORE_CONFIG`):

~~~toml
[remote]
type = "s3"                       # or "fs" with path = "/vault/store"
bucket = "verity-research"
endpoint = "https://<account>.r2.cloudflarestorage.com"
access_key_id_env = "RESEARCH_R2_ACCESS_KEY_ID"        # names of env vars; the file never holds secrets
secret_access_key_env = "RESEARCH_R2_SECRET_ACCESS_KEY"
~~~

No remote configured => the local store is the only replica and `where` says so; nothing fails, nothing is called PRESERVED.

**PRESERVED** means: remote `head` size matches AND a full read-back sha256 matches (objects < 4 GiB; above that, per-part
sha256 from the sidecar checked against the parts read back). ETags are not content hashes. `research data verify ART`
performs it; `put --preserve` performs it after upload; the local `replicas` view records `verified_utc` per location.

Replica resolution order for `fetch`: local objects -> repo replica (`fixtures/artifacts.json`: `{art id: repo-relative
path}`, verified against the manifest on open, fall back on mismatch) -> remote. `research data pull --from MACHINE RUN`
copies an attempt manifest plus its outputs' manifests and blobs from a machine's local store over ssh (uses
`research.remote` helpers) — the same-host tier of the planner until R2 is configured.

## 8. Labels

`labels/<target>/<ts>-<by>.json`:

~~~json
{"schema": "research/label/v0.1", "target": "art:...", "key": "proof_class", "value": "COMPLETE_ZK_BACKEND",
 "by": "red-team-zk-2", "ts": "2026-09-22T11:20:29Z", "ref": "art:<findings>|r<run id>|<url>"}
~~~

Append-only, one file per assertion, never read-modify-write. `research data labels TARGET` lists all; `--history` orders
by ts; `--key K --by PATTERN` filters. The store never picks a winner; decision tables do (`proof_class=… by=red-team-*`).

## 9. CLI (v0.1)

~~~text
research data put  --kind K [--meta JSON|@file] [--ref name=art:...]... [--file PATH | --tree DIR] [--preserve]   -> art id
research data show ART|ATTEMPT|DRV [--json]
research data where ART                       replicas: local / repo / remote (+ verified_utc)
research data fetch ART [--to DIR]            materialise payload (file or tree); prints path
research data verify ART                      PRESERVED check against the remote
research data lookup --tool NAME [--param k=v]... [--input name=art:...]... [--condition k=v]... [--source-tree SHA]
                                              -> attempts with that reuse_key (or --any-source: that derivation)
research data select [--kind K] [--where meta.key=value]... [--tool T] [--derivation D] [--attempt R] [--label k=v] [--json]
research data snapshot --name NAME (--members art:... | --select <select flags>) [--note TEXT]  -> art id (dataset-snapshot/v1)
research data labels TARGET [--history] [--key K] [--by PAT]
research data label  TARGET KEY VALUE --by WHO [--ref REF]
research data attempt publish RUN_DIR [--tool ...]   (what `research run` calls; also usable after the fact for old runs)
research data pull --from MACHINE RUN
research data reindex [--remote]
research data sql "SELECT ..."               power-user escape hatch; read-only connection
~~~

No query DSL: flags for common facets + `sql`.

## 10. `research run` integration

At completion or failure, in the runner (local and machine side alike), `store.attempt.publish(run_dir, tool=...)`:

1. `result.json`, if valid, becomes an artifact of kind `result.kind` (default `bench-result/v1`; `tc_probe` writes
   `tc-evidence/v1`): `meta` = the result document minus `artifacts`/`measurement_files` paths, `refs.run_files` = the
   run-files artifact. Paths listed in `result.artifacts` + `measurement_files` become ONE `run-files/v1` tree artifact.
2. `events.jsonl` / `resources.jsonl` become `telemetry/v1` file artifacts when present.
3. The attempt manifest (§4) is written to `attempts/<run id>.json` and indexed.
4. With a remote configured: blobs and manifests are uploaded and verified; `execution.preserved` is recorded per output.

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
