# PoUS

PoUS, proof of useful space for model weights, is how the auditor learns how much of the registered store each device
held. The memory wipe is part of it.

It runs in two steps. At setup, the developer's node commits to the weights before any salt exists, the verifier samples
a salt after the commitment, and the node encodes the weights under that salt into the store, which the verifier checks
decodes to what was committed. Then, in timed rounds, the memory attester challenges blocks of the store on live coins,
times each answer and signs the session's records, and the verifier checks each round's timing and its block's digest.
The memory attester is the auditor's device, on a machine the developer doesn't control.

This directory is the verifier's side. Its Lean executables, under `check/`, give the verdicts of record. Its Python
is a spec-level reference that matches the Lean models' definitions and is the oracle for the GPU kernels. It is pure
Python and slow by design. The prover's side is in `../prover/`, and the memory attester's in
[`../attester/`](../attester/README.md).

## Try it

The whole lifecycle at a toy geometry, from the tree's root after `uv sync --all-packages --extra torch-cpu`. It needs no
GPU and finishes at once:

~~~python
from verity.protocols.resource_accounting.memory.pous.verifier import audit, band, codec, setup
from verity.protocols.resource_accounting.memory.pous.verifier.layout import TensorLayout, decode_tensor

layout = TensorLayout.of(band.BandParams(16, 64, 4), [("w1", 200), ("w2", 300)])  # a toy geometry, not a deployed one
params, prims = layout.params, band.reference(layout.params)
w = layout.pack({"w1": bytes(200), "w2": bytes(range(256)) + bytes(44)})
commitment = setup.commit(params, w)                                   # the server, before any salt exists
salt = setup.sample_salt(b"verifier-secret", commitment)                 # the verifier
C = [params.codec.encode_segment(params, prims, salt, s, g)             # the server encodes C*
     for s, g in enumerate(codec.split(params, w))]
vk = setup.verifier_setup(params, prims, commitment, salt, encoded=C)   # the verifier decodes C*, checks the commitment
blocks = [b for seg in C for b in seg]
assert decode_tensor(layout, prims, salt, "w1", blocks.__getitem__) == bytes(200)
v = audit.UnsignedRounds(vk, audit.challenge_key(b"audit-secret", vk))  # k = 111 timed rounds
print(audit.run_timed(v, blocks.__getitem__).accepted)                  # True
~~~

## The codecs

| Codec | Its place |
| --- | --- |
| The band scheme, `band-chain/d12/v1` | The default. |
| The dense scheme | Selectable. |
| P2, `p2-16448/v4` | Experimental and never the default, but what a period runs. |
| P3 | The fast research codec. |

[`DESIGN.md`](DESIGN.md) states what each codec and the wipe rest on, word for word, and holds the entry points, the
lifecycle through the generic protocol, the memory attester's records and P2's construction.
