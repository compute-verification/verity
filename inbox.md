[2026-11-12T07:26Z] 220533211cb9
From your lead (@kernels), Thu 07:26Z: compute-accounting said YES to the wide row's design (CMP-380) and wants ONE HOST GUARD in the same pull request: the refusal at 129 segments or more moves from run.py:550-551 into the host launch wrapper of the three hashing kernels, so every caller meets it before any launch; host code only, the cubins' sha256 shown unchanged; once the wide path exists the wrapper sends a row over 128 segments to it. The fold's 0.56 ms is to be measured on the H200. inbox/KMOVE.md's last bullet.

