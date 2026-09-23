"""Ligerito (Novakovic-Angeris 2025, eprint 2025/1187) over BabyBear: the recursive-Ligero polynomial commitment scheme as a
succinct backend for the same relation IR.

``params.py``  proof-size / soundness / round-count calculator (Table 1 of the paper reproduced; our batch sizes; byte model
               pinned to the prototype's measured proofs) -- lane ligerito-design.
``ref.py``     pure-Python/numpy reference of the Ligerito PCS over BabyBear with degree-6 extension challenges
               (commit / prove-eval / verify at toy sizes) -- the spec the GPU prototype and the Rust verifier are
               checked against -- lane ligerito-design.
``pcs.py``, ``transcript.py``, ``proto/``: lane ligerito-proto's torch/cupy prover + numpy verifier (Phase A go/no-go
               measurement); now owned by lane ligerito-pcs-fast.
``layout.py``, ``sumcheck.py``: compile.System -> multilinear layout -> batched zero-check -> evaluation claims -- lane
               ligerito-sumcheck.
``zk.py``: lane ligerito-zk's masking (mask cells in the committed polynomial, per-column RS padding, Libra-style
zero-check mask) and the step-0 committed coins (PROTOCOL.md 8c) -- the hooks ``pcs.py`` and ``sumcheck.py`` call.
``prove.py``, ``proof.py``, ``run.py``: lane ligerito-relation's glue (witness -> layout -> commit -> zero-check -> PCS
               evaluation proof), the ``LGTO0001`` wire format, gates and bench-result/v1.
"""
