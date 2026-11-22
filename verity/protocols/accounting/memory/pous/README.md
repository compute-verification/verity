# PoUS

PoUS, proof of useful space, establishes that the hardware's memory held at least a stated number of bits of the
registered values through a session. It establishes it from a commitment to those values made before the memory
attester's salt, the store `C*` encoded under that salt, and the memory attester's timed openings of sampled blocks.

The guarantee: `Pous.Guarantees.P2SlackFamilyUncond` (`security/Definitions/Pous/Guarantees/P2.lean`).

Not yet: setup's check runs on the developer's node, PoUS starts after the serve and loads the weights again, and no field of the policy reads what it gives.

The memory wipe is a protocol of its own: `../wipe/`.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the declaration | `POUS`: the name `pous`, the option `pous:p2-16448/v4`, its hooks, parties and settings | `scheme.py` |
| the prover | commits the registered values, encodes `C*` on the card, decodes each due block's weight from `C*`, answers the timed reads | `prover/served.py`, `prover/weight_store.py`, `prover/period_prover.py` |
| the Lean verifier | `pous-run-protocol`: setup's decode of `C*` against the commitment, then the signed session at its point | `verifier/check/` |
| the auditor's device | the memory attester: the salt, a coin tree committed before the session, each answer timed on its clock, the session signed | `attester/challenger_session.py`, `attester/memory_attester.py`, `attester/lean/` |
| the reference | the codecs behind one API, setup, the timed audit and the period's answer, in Python | `verifier/` |
| its circuits | `PousP2SetupCheck_v2`, one unit a store block; the session's, decode's, answer's and verdict's circuit templates | `prover/setup_circuit.py`, `prover/*_circuit.py` |
| its device code | P2's encode and decode on the GPU (`libp2v1.so`) | `prover/p2/p2_store/`, `prover/p2/p2_gpu/` |

## What varies

| implementation | codec | GPU encode and decode | store tree | the memory attester's session | Lean verifier |
| --- | --- | --- | --- | --- | --- |
| P2 v4, `p2-16448/v4` | served | served | served | served | served |
| P2 v2, `p2-16448/v2` | exists | exists | exists | exists | never written |
| band chain, `band-chain/d12/v1` | exists | exists | exists | never written | never written |
| dense chain, `dense-chain/v1` | exists | exists | exists | never written | never written |
| P3, `p3/v1` | exists | never written | exists | never written | never written |

The codecs are `verifier/schemes/`' registry; the band and dense GPU code is the benchmark's (`benchmarks/pous/band_gpu/`).
P2 v2 keys its blocks with ChaCha8, which the spec does not allow; no declaration names it, and the GPU store refuses it.

A new implementation supplies: a codec (`verifier/codec.py`'s `Codec`: one segment encoded and decoded) with its certificate at its parameters, its GPU encode and decode, and a Lean verifier at its point.
Fixed for every implementation: the commitment before the salt, the store tree over each block's `sha512/row/v2` digest (`verifier/verifier.py`), the memory attester's coin tree and clock, and the hooks it attaches through.

## Interfaces

- From the workload, through `verity/hooks.py`: `RegisteredValues` (the registered values once resident, and which block
  reads each as its weight) and `BlockDue` (a block about to be computed); it answers `ReleaseValue` (each weight freed,
  restored decoded from `C*`) and `ComputeBlockWith` (its kernel decodes the due block's weight from `C*`).
- From `run`: today nothing. The adapter's launcher starts it after the serve (`benchmarks/pous/served_piece.py`), which
  loads the weights onto the card again; its salt and coins are the memory attester's own.
- It gives: the bits held (`s_bits`) and their error (`eps`), certified or not, to accounting's record (`space`); the
  bandwidth audit reads its store `C*` (`../poub/`); today no field of the policy reads it.

## Try it

~~~sh
uv run tools/check/suites.py verity/protocols/accounting/memory/pous/verifier/check
~~~

It builds the Lean verifier and holds each executable to the Python on the CPU: setup, the timed audit's wire on the
spec's vectors, a recorded run's verdict and the verdict unit against its circuit template. It ends `1 suites: 1 passed`.
