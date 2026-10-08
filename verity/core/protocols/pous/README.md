# POUS (`verity.core.protocols.pous`)

A spec-level reference for POUS, proof of useful space for model weights, with three codecs behind one API:

- **the band scheme** (`verity.core.protocols.pous.band`, the secure-first default): one layer on the band DAG with in-degree 12, `H`
  an overwrite chain on a double-width permutation (Lean's `chainBand`). Deploy it only while the effective round
  budget stays at or below 12 calls (the cliff);
- **the dense scheme** (`verity.core.protocols.pous.dense`, selectable): the same on the complete DAG (Lean's `chainDense`);
- **P2** (`verity.core.protocols.pous.schemes.p2`, **experimental**): the squaring-based scheme, proved in abstract M1 under the
  pinned `P2SlackFamilyUncond` (k = 105), with its instantiation assumption not accepted; never the default.
  `p2-16448/v4` has the spec's (v4) counter-mode SHA-512 keys, `p2-16448/v2` ChaCha8 keys, which the spec does not allow;
- **P3** (`verity.core.protocols.pous.scheme`, the fast research codec): two layers over the band graph `B(220, 12)` with a transpose
  connector and a tweakable `P` (Lean's `p3Scheme`).

It matches the Lean models' definitions and is the oracle for GPU kernels. Pure Python, stdlib plus `verity`. It is
slow by design: a 32 MiB band segment decodes in about 25 s on one core, a dense one in about 8 minutes, and a
14.4 MB P3 segment in about 21 s.

`PROTOCOL.md` describes the lifecycle, the API and the byte formats. `DISCREPANCIES.md` lists where the Lean model,
the prose spec and this code differ.

~~~python
from verity.core.protocols.pous import audit, band, codec, setup
from verity.core.protocols.pous.layout import Layout, decode_tensor

layout = Layout.of(band.SECURE_FIRST, [("w1", n1), ("w2", n2)])       # tensors -> segments of W
params, prims = layout.params, band.reference(layout.params)
w = layout.pack({"w1": b1, "w2": b2})
commitment = setup.commit(params, w)                                   # server, before any salt exists
salt = setup.draw_salt(verifier_secret, commitment)                    # verifier
C = [params.codec.encode_segment(params, prims, salt, s, g)             # server: C*, per segment
     for s, g in enumerate(codec.split(params, w))]
vk = setup.verifier_setup(params, prims, commitment, salt, encoded=C)   # verifier: decodes C*, checks the commitment
blocks = [b for seg in C for b in seg]
w1 = decode_tensor(layout, prims, salt, "w1", blocks.__getitem__)      # decode one tensor
v = audit.UnsignedRounds(vk, audit.challenge_key(audit_secret, vk))     # k = 111, Δ = 0.5 ms (+ rtt_allowance_ns)
result = audit.run_timed(v, blocks.__getitem__)
~~~

For dense, use `dense.POINT_64K` and `dense.reference`; for P3, `params.OPERATING_POINT` and `primitives.reference`.
Everything else is the same.

The same lifecycle through the generic protocol. A scheme implements only `encode_segment` and `decode_segment`, and
the memory challenger refuses any `k` or deadline its certificate doesn't cover:

~~~python
from verity.core.protocols.pous import protocol, schemes

scheme = schemes.get("band-chain/d12/v1", segments=layout.params.segments)   # or schemes.Band(layout.params)
commitment = protocol.commit(scheme, w)
salt = protocol.draw_salt(verifier_secret, commitment)
C = protocol.encode(scheme, w, salt)                                           # server
vk = protocol.verifier_key(scheme, commitment, salt, w=w)                      # PRIVATE: the verifier encodes
# vk = protocol.verifier_key(scheme, commitment, salt, mode=protocol.PUBLIC, encoded=C)   # the same guarantee
v = protocol.TimedAudit(scheme, vk, protocol.challenge_key(audit_secret, vk), rtt_allowance_ns=rtt)
v.run(protocol.responder(scheme, blocks.__getitem__))
print(v.profile().statement())                                                 # Pous.SecurityProofs.BandMultiMeetsFamily
~~~

On the deployed path the timed rounds run on the memory challenger (`memory_challenger`), an auditor-trusted device
inside the node's isolation boundary. It commits its indices in a coin tree before the session, hashes each whole
answer after reading its receipt time (`block_digest`, its `sha512/row/v2` digest: the inner digest of C's leaf, which
`answer_circuit.py` opens), and signs the session's round records once. `verdict` checks a
signed session in the clear and publishes only the verdict. It accepts S1's LMS over SHA-512 at `lms.DEPLOYED`
(`memory_challenger.LMS`) and nothing else: a test-signed or unsigned record is refused.

~~~python
from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc

signer = mc.LmsSigner(key, lms.LeafState(state_path, lms.public_key(key), lms.DEPLOYED.H))  # key = lms.keygen(lms.DEPLOYED, seed)
coins = mc.CoinTree.draw(challenger_secret, session_id, params.blocks, k)          # before the session
c = mc.MemoryChallenger(coins, signer, limit_ns=500_000 + rtt, block_bytes=params.block_bytes)
signed = c.run(blocks.__getitem__)                                                  # witness: never sent in the clear
v = mc.verdict(signed, c.commitment, c.openings(), lambda i: mc.block_digest(blocks[i]), trusted={signer.public_key})
print(v.public())                                                                   # {"accepted": ...}
~~~

Tests: `uv run pytest verity/core/protocols/pous` (about 8 s). P3's 64 KB-label segment vector runs with `VERITY_POUS_SLOW=1`.
Regenerate the vectors with `python verity/core/protocols/pous/tests/pous_vectors.py`, and the Lean vectors with
`tests/lean/generate.sh`.
