# The verifier's auditor host

`host.py` is vy-auditor-1's verifier service (INF-179): the remote bandwidth audit's verifier
(`benchmarks/remote_session/verifier.py`) run only as `auditor`, from a tree infra pinned, one run at a time, behind the
gate `vy-verifier-gate` that every key of the host's `research` is forced to. Its docstring is the reference: who reaches
the gate, what runs, a run, a bound run, a replay and the store. The host's deployed files are
`tools/research/src/research/pods/nebius/verifier/`, and infra deploys `host.py` and owns their entries in
`tools/research/src/research/pods/nebius/deploy.toml`.

## A run

~~~text
vy-verifier-gate run --peer NAME --port P --period P --pin SHA512 (--paced SPEC ... [--pace-pass-ms MS]
                     | --sessions N [--windows W] [--window-ms MS] [--samples S])
                     [--minutes M] [--jitter-ms MS] [--quantile Q] [--lifetime L] [--verdict lean|both]
                     [--model ROLE [--engines E] --arena-blocks A]
~~~

A lane names a run's parameters, each held to a whitelist, and never a path, a host, a key or a node's capacity: those
are the host's, from `/etc/vy/verifier.json`. `--pin` is the SHA-512 of the challenger's public key, passed to
verifier.py as `--challenger-pin`.

A bound run (`--model ROLE`, ROLE of `[A-Z0-9_]`, with `--arena-blocks A` and optionally `--engines E`, 1 to 8) binds
the registered store to a pinned model, which `auditor_line.py`'s `bound` check needs. The gate adds verifier.py's
`--checkpoint <models>/<ROLE>` and its `--pin`, core's pin in the pinned tree
(`verity/core/protocols/pouw/lean/check/models.json`). It refuses the run before starting anything when the role has no
directory there, when the pinned tree has no `models.json`, or when `--model` comes without `--arena-blocks`.

## A bound role's checkpoint

`<models>` is `verifier.json`'s `models`, default `/var/lib/vy-verifier/models`. Infra provisions one directory per role:
every file the role's `models.json` entry pins, at the pinned revision, as a root-held regular file (0644, in a root-held
0755 directory). Copy the files out of a download, never point at an HF cache's snapshot: its files are symlinks into
the cache's blobs, which `store_layout.check` refuses. Qwen2.5-32B in BF16, role `QWEN25_32B_BF16`, about 66 GB:

~~~sh
hf download Qwen/Qwen2.5-32B --revision 1818d35814b8319459f4bd55ed1ac8709630f003 --local-dir /tmp/q
install -d -m 0755 -o root -g root /var/lib/vy-verifier/models/QWEN25_32B_BF16
install -m 0644 -o root -g root /tmp/q/* /var/lib/vy-verifier/models/QWEN25_32B_BF16/
~~~

verifier.py holds every pinned file to its SHA-512 before it listens, so the files are trusted because they match the
pin, not because of where they came from. A run binds to a role once the pinned tree's `models.json` pins it.

## Tests

`tests/test_host.py` runs the gate in-process with its root steps injected. It holds the whitelist to verifier.py's own
parser, including a bound run's binding to a `models.json`-shaped pin. It also runs one session through
`tests/stub_verifier.py` (slow). Run the suite as `uv run tools/verity/check/suites.py benchmarks/live_verifier`, on a
node (the testing skill).
