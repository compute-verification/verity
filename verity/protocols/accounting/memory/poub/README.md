# PoUB

The bandwidth audit establishes, epoch by epoch, how many bits of a registered store the hardware read from its
memory. It establishes it from a store committed before any coin, passes over it each keyed by a live coin and
committed by its root before the epoch's deadline, and openings of sampled receipts checked against the store.

Two declarations exist in `scheme.py`, each its own protocol with its own verifier and store, and no setting turns one
into the other: `bandwidth` (PoUB-P2, `verity/remote-bandwidth/p2/v0`) over PoUS's store `C*`, the one an application
lists; and `poub` (the remote session, `verity/remote-bandwidth/v0`) over the node's own store, re-keyed each lifetime
with probation (`refresh`), which no application lists and which holds the design against stacking.

The guarantee: `Properties.Poub.Line.LineHeld` (`security/Properties/Poub/Remote/Line.lean`), deployed, is `poub`'s; `bandwidth`'s `SketchBoundP2` is drafted.

Not yet: openings and plans reach the verifier in the clear, no session proves its opening unit, the requirement link and the stacking property are inputs to the ledger, and nothing starts either from `run`.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the declarations | `BANDWIDTH` and `POUB`: names, protocols, parties, settings, each one's own term | `scheme.py` |
| the prover | `bandwidth`: passes over PoUS's `C*` on the card that serves from it; `poub`: a binder keeping the node store current from the served run, and its passes | `prover/p2_stage/`, `prover/binder.py`, `prover/node_store/`, `prover/commit_session.py` |
| the Lean verifier | `poub-p2-protocol` (`Poub.P2Protocol`); `poub-verifier` and `poub-protocol` (`Poub.Verifier`, `Poub.Protocol`): the session an event a line, then its case | `verifier/lean/` |
| the verifier party | the verifier and its challenger: coins, deadlines, sampled receipts, the ledger | `verifier/p2_parties.py`, `verifier/session_parties.py`, `verifier/audit.py`, `verifier/ledger.py` |
| the auditor's device | the host that runs the verifier party, behind one gate | `attester/live_verifier/host.py` |
| its circuits | `BlockOpeningUnit_v2`: a sampled receipt's opening, `poub`'s only | `verifier/commitments/gates/block_check.py` |
| its device code | `poub`'s passes and receipts on the GPU; `bandwidth` reads PoUS's decode | `prover/codec_tensor/`, `prover/poub_session/`, `prover/word_codec/` |

## What varies

| implementation | store: PoUS's `C*` | store: the node's own | coins from the verifier party | refresh | opening unit | Lean verifier |
| --- | --- | --- | --- | --- | --- | --- |
| `bandwidth` | served | never written | served | never written | never written | served |
| `poub` | never written | exists | exists | exists | exists | exists |

Both take a pass's seed from its epoch's challenge and the previous pass's root, and send each pass's root and then the
epoch's sampled openings to the verifier party; `bandwidth`'s closer also reads PoUS's memory attester.

A new implementation supplies: a store with its registration (its blocks and their tree), the passes over it, and a Lean verifier of its session.
Fixed for every implementation: the live coins and the epoch's deadline, the receipt layout of its own term (`DELTA_SCHEME`), the sampled openings and the pooled bound.

## Interfaces

- From the workload: `bandwidth`, nothing of its own: PoUS's store on the same exchange (`depends`); `poub`, through
  `verity/hooks.py`, `Declared`, `RegisteredValues` and the `steps` events, filing `ReuseOldestFreed`, `FixRanges`,
  `RequireProperty` and `StartBeside` (its session's process).
- From `run`: today nothing. `bandwidth` runs in PoUS's piece after the serve (`benchmarks/pous/served_piece.py`), and
  `poub` in a session of its own (`benchmarks/poub_session/`); the coins are the verifier party's own.
- It gives: `bandwidth`, the bytes read (`q_bytes`) and their error (`eps`) to accounting's record (`bandwidth`);
  `poub`, its session's verdict and line, in no combined record. Today no field of the policy reads either.

## Try it

~~~sh
NODE_KIND=rtx-pro-6000-server-x8 uv run verity bench memory --loopback
~~~

A whole `poub` session on the CPU, both parties on this machine over a test store, the Lean verifier built first: it
prints the share of the node's read rate left uncertified with its error, then `a measurement, not an audit`, and exits 0.
