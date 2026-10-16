# research.store: pod credentials, preservation and replica resolution

The detail of the store README's §7.1 (`../README.md`, the contract): the temporary credentials `research data mint-credential`
gives a pod, what the store does with no usable remote, what PRESERVED means and how `push` checks it, and the replicas `fetch`
reads, in order. The S3 client, the remote config and §7's other sections are in the store README.

## Temporary credentials for pods: `research data mint-credential`

A GPU pod gets a short-lived, bucket-scoped credential instead of the parent R2 key (`../credentials.py`, stdlib):

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
  sha256 hex of the JWT, the session token is `base64("jwt/" + JWT)`. R2 enforces the permission, `--prefix` and `--actions` (checked live). One deviation from the reference TypeScript: R2 answers 400 to a
  JWT that carries both `scope` and `actions`, so `--actions` *replaces* `--permission` in the claims. Only this path can scope by
  action.
- **api**: `POST /client/v4/accounts/{account}/r2/temp-access-credentials` with `CLOUDFLARE_API_TOKEN` (body `{bucket,
  parentAccessKeyId, permission, ttlSeconds, prefixes?, objects?}`), used when the parent secret is not in the environment. Unit-tested
  against a fake transport; not exercised live (no API token in `r2.env`).

TTL <= 7 days. Output is JSON (`accessKeyId`, `secretAccessKey`, `sessionToken`, `expires_utc`, `via`, `bucket`, `scope`, ...) or, with
`--env`, the three `export AWS_ACCESS_KEY_ID= ... AWS_SECRET_ACCESS_KEY= ... AWS_SESSION_TOKEN=` lines for a pod bootstrap; nothing is
ever written to disk by the tool. A pod with those three variables and the same `store.toml` pushes through `S3Remote` as usual
(the token rides in the signed headers: the store README, §7). A temporary credential cannot mint another (the tool refuses when
`AWS_SESSION_TOKEN` is set); revoking the parent token kills every child.

## No remote, or an unusable one

No remote configured (no file, or no `[remote]` table) => the local store is the only replica and `where` says so; nothing
fails, nothing is called PRESERVED (`verify` returns `preserved=False, reason="no remote configured"`; `push` refuses). A
`[remote]` table that is present but unusable (unknown type, missing field, credential env vars unset) is `ConfigError` from
`load_remote`, on purpose: a broken vault must not look like "no vault". `LocalStore` keeps that distinction without making the
local store unusable: `store.remote` degrades to `None` and keeps the reason in `store.remote_error`; reads, puts, labels and
attempt publication proceed local-only, and `push` / `push_attempt` / `research data push` raise the reason (e.g. which env vars
to set). `research run` reports which of the three happened: `PRESERVED on the remote`, `NOT preserved: <error>`, or `local only
(<reason>)`; `research data push RUN` (or `--pending`) preserves later, with credentials in the environment
(`set -a; source ~/.config/verity/r2.env; set +a` for the R2 vault). `where` still answers for the local and repo replicas and reports the remote row as `UNUSABLE <error>` (never "not configured", never absent).

## PRESERVED, and how `push` checks it

**PRESERVED** means: for the manifest and every blob, the remote holds an object of the right size whose content was checked,
by one of two checks, and the replica record (`replicas.verified`) says which:

~~~text
etag-md5          `push --verify head` (default): HEAD after upload; Content-Length == bytes and the single-part ETag == hex MD5 of
                  the local bytes (S3 and R2 make a single-PUT object's ETag its MD5; a `-` in the ETag marks multipart, which is not
                  an MD5, and such an object falls back to read-back). One extra round trip per object.
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

## Replica resolution for `fetch`

Replica resolution order for `fetch`: local objects -> repo replica (`fixtures/artifacts.json`: `{art id: repo-relative
path}` of the payload — a file, or a tree's directory; every blob is sha256-checked against the manifest before use, mismatch
warns on stderr and falls through) -> remote (downloaded bytes are checked too: `IntegrityError` on mismatch). The repo map
supplies bytes only; the manifest must still be resolvable locally or remotely. Bytes that arrive from a repo or the remote are
cached into `objects/`. A repo fixture is registered bare (`fixture/v1`, `meta {}`, `refs {}`), so its id is its bytes (plus the
bare file name); when the bytes change on purpose, `research data refresh-fixtures [--preserve]` re-registers it and moves the old
id into the map's `previous` block (`{"previous": {"art:<new>": ["art:<old>", ...]}}`), which readers skip: `replica_map()` takes
`art:` keys only. A registry entry git does not track is fetched instead (the store README, §7.3). `research data pull --from MACHINE RUN`
copies an attempt manifest plus its outputs' manifests and blobs from a machine's local store over ssh (uses
`research.remote` helpers) — the same-host tier of the planner until R2 is configured.
