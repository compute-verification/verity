# research.store: pod credentials, preservation and replica resolution

A result counts as kept only once the store's remote holds it. This part of the store says three things: the short-lived
key a pod writes to the remote with in place of the parent key, when an artifact counts as PRESERVED there, and where
`fetch` looks for an artifact's bytes, in order.

## Try it

Take any `art:` id from the repository's `artifacts.json`:

~~~sh
research data verify ART
~~~

This reads the artifact's manifest and every blob back from the remote, checks each against its sha256, and prints
`PRESERVED` with the remote and how many objects it read back. It needs the store's credentials in the environment and
exits 1 unless the artifact is PRESERVED.

| Command | What it decides |
| --- | --- |
| `research data mint-credential` | the short-lived, bucket-scoped key a pod or a cluster job holds |
| `research data push`, `research data verify` | whether an artifact is PRESERVED on the remote, and by which check |
| `research data fetch` | which replica an artifact's bytes come from: local, then the repository, then the remote |

[`DESIGN.md`](DESIGN.md) has each in full: the minting paths, the store with no usable remote, the three checks and the
push order, and the replicas `fetch` reads.
