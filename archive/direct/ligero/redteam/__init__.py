"""Red team of the B-Ligero ZK claim (lane red-team-zk; `backends/numerical/reports/redteam_zk.md`).

* ``linalg``     -- BabyBear polynomial / matrix helpers (rank, Lagrange interpolation with slack).
* ``mask_rank``  -- class 1: exact dimension counts of every mask against every disclosure (openings, w, v, h, q).
* ``q_leak``     -- the finding: the chain message ``q`` equals ``Q = sum_i R_i p_i`` on H (the mask ``Z_H s_chain``
                    vanishes there), which recovers every intermediate accumulator from the public transcript.
* ``simulator``  -- class 2: the malicious-verifier simulator (programmable random oracle), V* variants, the
                    real-vs-simulated indistinguishability test.
* ``run``        -- the pod driver that produces ``results/redteam_zk_lane.json``.
"""
