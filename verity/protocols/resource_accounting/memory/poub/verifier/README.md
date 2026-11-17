# PoUB

PoUB is the memory protocol that tells the auditor how much of a node's memory read rate is left uncertified. The node
reads its registered store in passes, each under a live coin, and answers challenges about what it read. The verifier
times the passes and checks the answers against the store's committed contents, and its ledger turns an accepted
session into the share of the read rate left uncertified, with its error.

Its specification states what it does: "PoUB v0: a verifier with no hardware at the site certifies, window by window,
how many reads of a node's registered store the node made."

This directory is the verifier's side. Its definition is the Lean package in `lean/`, whose process decides every
session; the Python here keeps the books beside it and is held to it by a difftest. The prover's side is in
`../prover/`, and the memory attester's in `../attester/`.

## Try it

A whole session on the CPU, both processes on this machine over a test store, from the tree's root:

~~~text
uv run verity bench memory --loopback
~~~

It prints the session's share and its error, says that no node was audited, and exits 0. It certifies nothing. It
builds the Lean verifier in the tree first, so it needs Lean (`bash tools/verity/lean/setup.sh
verity/protocols/resource_accounting/memory/poub/verifier/lean`). The [memory benchmark](../../../../../../benchmarks/memory/README.md) runs a
session on a GPU while vLLM serves from the store.

## What runs today

A session runs end to end on live coins: the verifier and its challenger as a party of their own, on a host the developer
doesn't control (`verifier_party.py`), or every party in one process (`period.py`), against the prover's passes over
its registered store. A period's case replays through the Lean (`Poub.Protocol`), and the session harness runs
the Lean verifier beside the books (below). Not all of it is zero knowledge yet:

- **Openings are in the clear.** Each window's openings go to the verifier party as they are (`wire.encode_openings`),
  and it checks them itself (`audit.AuditRun.verify`). Those checks are the specification of what each zero-knowledge
  proof block proves (`PROTOCOL.md` §8), but no session proves them: `opening.py` evaluates the opening unit on the
  honest witness, and only `benchmarks/one_stage/a8.py` proves one window's units in zero knowledge.
- **Plans are in the clear.** The verifier sees which blocks each pass covers (`PROTOCOL.md` §8, "What the plan
  shows").
- **The requirement's units and the stacking property are inputs, not proofs.** The ledger credits the served
  circuit's reads only on the plan and memory-map units' verdict (`audit.RequirementLink`), and blocks appended after
  the seed only on the stacking property's (`audit.PostSeedVerdict`); without them it credits the node's reads of its
  registered store. Only `benchmarks/one_stage/a7.py` proves the units.

[`PROTOCOL.md`](PROTOCOL.md) has the semantics, the byte formats and the record. [`DESIGN.md`](DESIGN.md) holds the
entry points, which side is the definition and why the Lean decides alone, replaying a recorded session, and what each
module does.
