# The verifier's auditor host

`host.py` is vy-auditor-1's verifier service: the verifier of proof of useful bandwidth (PoUB)
(`benchmarks/remote_session/verifier.py`) run only as `auditor`, from a tree infra pinned, one run at a time, behind the
gate `vy-verifier-gate` that every key of the host's `research` is forced to. Its docstring is the reference: who reaches
the gate, what runs, a run, a bound run, a replay and the store. The host's deployed files are
`tools/research/src/research/pods/nebius/verifier/`, and infra deploys `host.py` and owns their entries in
`tools/research/src/research/pods/nebius/deploy.toml`.

## A run

~~~text
vy-verifier-gate run --peer NAME --port P --period P --pin SHA512 (--paced SPEC ... [--pace-pass-ms MS]
                     | --sessions N [--windows W] [--window-ms MS] [--samples S])
                     [--minutes M] [--jitter-ms MS] [--quantile Q] [--lifetime L]
                     [--model ROLE [--engines E] --arena-blocks A]
~~~

A lane names a run's parameters, each held to a whitelist, and never a path, a host, a key or a node's capacity: those
are the host's, from `/etc/vy/verifier.json`. `--pin` is the SHA-512 of the challenger's public key, passed to
verifier.py as `--challenger-pin`.

The verifier credits a session's coins only from a challenger that ran main's code (`coins.trusted`, through
`challenger.pinned_session`). It fetches the challenger's pins log through the challenger's gate (`pins`) at each
session's end, beside the ledger, and it reads main's first-parent history from `/opt/vy-verifier/main-first-parent.txt`
(`--challenger-main`). Infra writes that file, one 40-hex commit a line, whenever it pins either host:

~~~sh
git fetch origin main && git rev-list --first-parent origin/main > main-first-parent.txt
install -m 0644 -o root -g root main-first-parent.txt /opt/vy-verifier/main-first-parent.txt   # on the auditor host
~~~

The gate refuses a run before starting anything while the file is missing. A challenger pinned after the file was
written isn't on it, so its sessions are refused as `coins` until infra writes the file again.

A bound run (`--model ROLE`, ROLE of `[A-Z0-9_]`, with `--arena-blocks A` and optionally `--engines E`, 1 to 8) binds
the registered store to a pinned model, which `auditor_line.py`'s `bound` check needs. The gate adds verifier.py's
`--checkpoint <models>/<ROLE>` and its `--pin`, core's pin in the pinned tree
(`verity/core/protocols/pouw/lean/check/models.json`). It refuses the run before starting anything when the role has no
directory there, when the pinned tree has no `models.json`, or when `--model` comes without `--arena-blocks`.

## A bound role's checkpoint

`<models>` is `verifier.json`'s `models`, default `/var/lib/vy-verifier/models`. Infra provisions one directory per role:
every file the role's `models.json` entry pins, at the pinned revision, as a root-held regular file (0644, in a root-held
0755 directory). Copy the files out of a download, never point at an HF cache's snapshot: its files are symlinks into
the cache's blobs, which `remote_bandwidth.store.check` refuses. Qwen2.5-32B in BF16, role `QWEN25_32B_BF16`, about 66 GB:

~~~sh
hf download Qwen/Qwen2.5-32B --revision 1818d35814b8319459f4bd55ed1ac8709630f003 --local-dir /tmp/q
install -d -m 0755 -o root -g root /var/lib/vy-verifier/models/QWEN25_32B_BF16
install -m 0644 -o root -g root /tmp/q/* /var/lib/vy-verifier/models/QWEN25_32B_BF16/
~~~

verifier.py holds every pinned file to its SHA-512 before it listens, so the files are trusted because they match the
pin, not because of where they came from. A run binds to a role once the pinned tree's `models.json` pins it.

## Tests

`tests/test_host.py` runs the gate in-process with its root steps injected. It holds the whitelist to verifier.py's own
parser, including a bound run's binding to a `models.json`-shaped pin and the host's main list. It also runs one
session through `tests/stub_verifier.py` (slow). Run the suite as
`uv run tools/verity/check/suites.py benchmarks/live_verifier`, on a node (the testing skill).
