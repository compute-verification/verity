# The memory wipe

The wipe bounds how many bits of a node's prior memory content can survive an accepted wipe. The memory attester
streams labels under a key fresh for the wipe until the fill is complete, then reads `k` of them back, one at a time,
each against a deadline on its own clock.

The guarantee: `Pous.Guarantees.StreamedWipeResidual` (`security/Definitions/Pous/Guarantees/StreamedWipeResidual.lean`), under its premise `StreamedWipe.Covers`.

Not yet: the served wipe takes no signed network egress count over its span, so it publishes `certified: false` and a bound equal to all declared memory.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the declaration | `WIPE`: the name `wipe`, its settings, parties and launcher | `scheme.py` |
| the prover | fills the card's memory with the labels and answers the reads, before the workload's engine loads | `prover/wipe_served.py`, `prover/wipe_run.py`, `prover/wipe_store.py` |
| the Lean verifier | `wipe-protocol`: the signed session's verdict, then `uncounted`'s public outputs | `../pous/attester/lean/MemoryAttester/WipeProtocol.lean` |
| the auditor's device | the memory attester: the key, the reads, their deadlines, the session signed | `attester/wipe_challenger.py`, `../pous/attester/` |
| the reference | the commitment, the batch records and the verdict, in Python | `verifier/streamed_wipe.py`, `verifier/wipe_period.py` |
| its device code | the label streamer, on the card or the host | `attester/wipe_stream.c`, `attester/wipe_stream.py` |

## What varies

| implementation | label store | Lean verifier | egress count |
| --- | --- | --- | --- |
| on the card (`store: cuda`) | served | served | never written |
| in host memory (`store: host`) | exists | served | never written |
| in Python (`store: python`) | exists | served | never written |

The counted check, `checkCounted` (`../pous/attester/lean/MemoryAttester/WipeCount.lean`), certifies at the count its
evidence states; only tests call it (`tests/test_wipe_counted_callers.py`), since nothing yet ties that count to the
network attester's.
On the card, the judge holds the fill to the card file's fill floor, `store.wipe_fill`, whose entry is a placeholder;
the memory between it and `store.dram` is filled by no label and is reported beside the verdict as
`uncovered_card_bytes`.

A new implementation supplies: a label store and streamer that fill the declared memory and answer each read by its deadline.
Fixed for every implementation: the labels' key and SHA-512, the memory attester's reads and clock, and `wipe-protocol`.

## Interfaces

- From the workload: nothing; it runs whole before the engine loads, and touches no weight.
- From `run`: the settings (`period.json`'s `wipe`), the memory attester's binding, and its key's pin.
- It gives: `foreign_bits`, `eps` and `certified` to accounting's record (`wipe`); onsite's soundness takes `WipeSound`.

## Try it

~~~sh
bash benchmarks/pous/smoke/wipe.sh
~~~

A rehearsal on the processor that certifies nothing: the memory attester runs on this machine under a key made for the
run, the fill is in host memory, and its last line is `WIPE {...}` with the verdict and `wipe-protocol`'s SHA-512.
