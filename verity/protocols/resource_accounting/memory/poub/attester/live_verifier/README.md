# The verifier's auditor device

This is the auditor device's verifier service. `host.py` runs PoUB's verifier on a host of the auditor's, only as the
user `auditor`, from a tree infra pinned, one run at a time, behind one gate, `vy-verifier-gate`, that every key of the host's `research` is forced to. The same gate
runs the memory attester's parties, and the same `host.py`, pinned apart, is the compose gate.

`host.py`'s docstring is the reference: who reaches the gate, what runs, a run, a replay and the store. The host's
deployed files are `tools/research/src/research/pods/nebius/verifier/`, and infra deploys `host.py`.

## Try it

Its tests run the gate in-process, with its root steps injected. Run them on a node:

~~~text
uv run tools/check/suites.py verity/protocols/resource_accounting/memory/poub/attester/live_verifier
~~~

## Who may reach a party

`party_log wipe|pous|bw [LOG]` answers root on the host alone (`sudo vy-verifier-gate party_log ...`): the gate refuses
it over ssh to every key, named or not, since a party's log holds what its device received and signed. The parties write
their files as before.

A party verb (`wipe_party`, `challenger_party`, `bw_party`) answers only a caller from a registered node, so every answer
of a party goes into a node. The wrapper `vy-verifier-gate` passes the ssh caller's address (the first word of
`SSH_CONNECTION`, which sudo would otherwise drop) as `--from`. The gate refuses the verb unless that address is one of a
peer's `addresses` in `/etc/vy/verifier.json` (`node_of`), whatever the key. A peer lists the addresses its ssh to the
host comes from:

~~~json
"peers": {"NODE": {"host_bytes": 1843153879040, "ingress_bytes_per_s": 4.4e9, "addresses": ["ADDRESS"]}}
~~~
