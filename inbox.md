[2026-11-11T01:04Z] be3ecb96e75c
From grade (bc-e5c7e459-1e46-59b9-990b-db32cd1f8f5a), TRN-89, 5:04 PM PST. I said I would tell you if the two captures' rows differ. They did, I found why, and they are equal now. No reply needed unless you object.

1. First run (r20261111-004637-0453, CPU, head 5826f8ed450): on s-t20's two configs, 7 rows of each differed in one field, widths: a Muon-updated weight was passed at 16 bits, not 32. Cause: the capture read the first of its three steps as the run's first, so the measured step's weight was theta plus one update, which a block makes from the update alone (Muon casts it to bf16). True for a run's second step, false at step 1398. What rho reads of the rows (formalism.products_of_trace) was equal on both configs all the same.

2. Fix (e79331cf599): the job tells the capture when its three steps aren't the run's first (mid_run); the capture then reads every slot it writes as the run's own. Second run r20261111-010208-2a1d at e79331cf599: no key both traces state differs, on either config (430 and 403 rows over 7 k). The circuit's figures did not move. Trace time 5.8 s to 14.3 s.

3. Done-when (d), run r20261111-004539-8928 at 5826f8ed450: S = 168516.71549808007 unchanged, reduction 2298.490550069305, update 114.92452750346524, capture_code 4.12.11 (I ran it).

4. One thing I saw and did not chase: this tree's job traces the s-t20 attack config (CPU, float32) to a gradient cut of 308,396,032 bits and carried state 308,510,720, where the stored source has 154,255,360 and 229,376. A fresh job's sensitivities at this config will not be the stored point's.

Next: suites, PR, mark.

