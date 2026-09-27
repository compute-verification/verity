# POUS P3 (`verity_pous`)

A spec-level reference for P3, the proof-of-unique-storage encoding: two layers of labels over the band graph
`B(220, 12)` with a transpose connector, `H` an overwrite chain on a double-width permutation, and a tweakable `P`.
It matches the Lean model's definitions (`p3Scheme` over `p3Model'`) and is the oracle for later GPU kernels. Pure
Python, stdlib plus `verity`. It is slow by design: one 14.4 MB segment encodes in about 21 s.

`PROTOCOL.md` describes the lifecycle, the API and the byte formats. `DISCREPANCIES.md` lists where the Lean model,
the prose spec and this code differ.

~~~python
from verity_pous import audit, codec, scheme, setup
from verity_pous.layout import Layout, decode_tensor
from verity_pous.params import OPERATING_POINT
from verity_pous.primitives import reference
from verity_pous.verifier import VerifierKey

layout = Layout.of(OPERATING_POINT, [("w1", n1), ("w2", n2)])         # tensors -> segments of W
params, prims = layout.params, reference(layout.params)
w = layout.pack({"w1": b1, "w2": b2})
commitment = setup.commit(params, w)                                   # server, before any salt exists
salt = setup.draw_salt(verifier_secret, commitment)                    # verifier
segs = list(codec.split(params, w))
C = [scheme.encode_segment(params, prims, salt, s, g) for s, g in enumerate(segs)]    # per segment, on bytes
vk = setup.verifier_key(params, prims, segs, commitment, salt, encoded=C)            # the verifier's own encoding
blocks = [b for seg in C for b in seg]
w1 = decode_tensor(layout, prims, salt, "w1", blocks.__getitem__)      # decode one tensor
v = audit.TimedVerifier(vk, audit.challenge_key(audit_secret, vk))     # Δ = 0.5 ms (+ rtt_allowance_ns)
result = audit.run_timed(v, blocks.__getitem__)
~~~

Tests: `uv run pytest protocols/pous` (about 8 s). The 64 KB-label segment vector runs with `VERITY_POUS_SLOW=1`.
Regenerate the vectors with `python protocols/pous/tests/pous_vectors.py`, and the Lean vectors with
`tests/lean/generate.sh`.
