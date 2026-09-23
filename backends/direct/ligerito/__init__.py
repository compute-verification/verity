"""Ligerito (Novakovic-Angeris 2025, eprint 2025/1187): a succinct backend for the same relation IR (lane ligerito-design).

``params.py``  proof-size / soundness / round-count calculator (Table 1 of the paper reproduced; our batch sizes).
``ref.py``     pure-Python/numpy reference of the Ligerito PCS over BabyBear with degree-6 extension challenges
               (commit / prove-eval / verify at toy sizes) -- the spec the GPU prototype and the Rust verifier are
               checked against.
"""
