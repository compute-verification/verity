[2026-11-11T05:43Z] 877da795e067
From the infra lead, Wed 11 Nov 05:45Z, for JOBUSER (INF-669). No change to your order or its hour; three facts to carry into the PR.

1. I ran the socket test tonight on cpu-6 as `research` with training's `sandbox.py` at `df79b8259d48` (scripts in the notes: `lanes/infra/tools/inf669-sock.sh`, `inf669-dns.sh`). From inside its sandbox `systemd-run --user` returned 0 and its process ran in the node's own namespaces with eth0 up and the hidden directory readable. And with no interface up `resolvectl query github.com` printed an address. With `--hide /run` both are closed. training now hides `/run` whole and `/tmp/tmux-<uid>` in its own sandbox and keeps a refusal for any socket still visible.

2. For your unit: `unit_argv` empties `/run` and binds `/run/systemd/resolve` back. systemd-resolved's varlink socket `io.systemd.Resolve` is in that directory, so a job inside a `--job-user` unit can still ask the resolver for a name. Say in the PR body what the unit does and does not close here. If binding only the two resolv.conf files instead of the directory keeps name lookups working for jobs that have the network (check how `/etc/resolv.conf` resolves on the nodes: it points at `stub-resolv.conf` and the stub listens on 127.0.0.53), propose it as a separate small commit with a test, or say why not. Do not widen the PR beyond that.

3. I owe training a run of the same test inside a `--job-user` unit at your PR's head, by Thu 12 Nov. Put in your final message the exact command I run on a node for that (the unit with `--job-namespaces user,mnt,net`, then training's `sandbox.py -- bash inner.sh`), and what each line should print when the unit holds.

