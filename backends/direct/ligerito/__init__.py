"""Ligerito (Novakovic-Angeris 2025, eprint 2025/1187) over BabyBear: the recursive-Ligero polynomial commitment scheme as a
succinct backend for the same relation IR.

``params.py``  proof-size / soundness / round-count calculator (Table 1 of the paper reproduced; our batch sizes; byte model
               pinned to the prototype's measured proofs) -- lane ligerito-design.
``ref.py``     pure-Python/numpy reference of the Ligerito PCS over BabyBear with degree-6 extension challenges
               (commit / prove-eval / verify at toy sizes) -- the spec the GPU prototype and the Rust verifier are
               checked against -- lane ligerito-design.
``pcs.py``, ``transcript.py``, ``proto/``: lane ligerito-proto's torch/cupy prover + numpy verifier (Phase A go/no-go
               measurement); now owned by lane ligerito-pcs-fast.
"""
