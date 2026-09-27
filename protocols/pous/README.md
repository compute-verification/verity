# POUS (`verity_pous`)

A spec-level reference for POUS, the proof-of-unique-storage encoding, with two codecs behind one API:

- **the dense scheme** (`verity_pous.dense`, the secure-first default): one layer on the complete DAG, `H` an overwrite
  chain on a double-width permutation (Lean's `chainDense`);
- **P3** (`verity_pous.scheme`, the fast research codec): two layers over the band graph `B(220, 12)` with a transpose
  connector and a tweakable `P` (Lean's `p3Scheme`).

It matches the Lean models' definitions and is the oracle for GPU kernels. Pure Python, stdlib plus `verity`. It is
slow by design: a 32 MiB dense segment encodes in about 8 minutes, and a 14.4 MB P3 segment in about 21 s.

`PROTOCOL.md` describes the lifecycle, the API and the byte formats. `DISCREPANCIES.md` lists where the Lean model,
the prose spec and this code differ.

~~~python
from verity_pous import audit, codec, dense, setup
from verity_pous.layout import Layout, decode_tensor

layout = Layout.of(dense.SECURE_FIRST, [("w1", n1), ("w2", n2)])      # tensors -> segments of W
params, prims = layout.params, dense.reference(layout.params)
w = layout.pack({"w1": b1, "w2": b2})
commitment = setup.commit(params, w)                                   # server, before any salt exists
salt = setup.draw_salt(verifier_secret, commitment)                    # verifier
segs = list(codec.split(params, w))
C = [params.codec.encode_segment(params, prims, salt, s, g) for s, g in enumerate(segs)]   # per segment
vk = setup.verifier_key(params, prims, segs, commitment, salt, encoded=C)            # the verifier's own encoding
blocks = [b for seg in C for b in seg]
w1 = decode_tensor(layout, prims, salt, "w1", blocks.__getitem__)      # decode one tensor
v = audit.TimedVerifier(vk, audit.challenge_key(audit_secret, vk))     # k = 111, Δ = 0.5 ms (+ rtt_allowance_ns)
result = audit.run_timed(v, blocks.__getitem__)
~~~

For P3, use `params.OPERATING_POINT` and `primitives.reference`; everything else is the same.

Tests: `uv run pytest protocols/pous` (about 8 s). P3's 64 KB-label segment vector runs with `VERITY_POUS_SLOW=1`.
Regenerate the vectors with `python protocols/pous/tests/pous_vectors.py`, and the Lean vectors with
`tests/lean/generate.sh`.
