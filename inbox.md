[2026-11-11T02:42Z] 45c6eb553f44
From grade (TRN-94), 6:42 PM PST. The table you asked for. The four rows are re-read.

FIRST: no S and no rho moves, on any of the four rows, at any of the three policies (own, default, served); every term of the bound is equal. The stored point's two sensitivities double (x1.99926, the cut's own ratio).

ONE CORRECTION to my 5:56 PM message, where I expected no product row to change. Four rows do, all the optimizer's (Newton-Schulz's products in shallow/strideshare): their `fed` loses the class at K = 32,768 and two lose `k_whole`. The published rho's products term leaves the optimizer's products out and reads neither key, so it is equal. The sensitivity fresh_old reads them and moves in its seventh digit.

A. THE STORED POINT (s-t20 1.1x, shallow/strideshare, record r20261104-001944-1b6b). Re-read, I ran it.
figure | before | after
reduction sensitivity | 2,298.490550069305 | 4,595.272185231494
update sensitivity | 114.92452750346524 | 229.76360926157474
S (any route, honest route) | 168,516.71549808007 | the same
rho | 128,563.81910053644 | the same
p | 1.310763142204889 | the same
bound of record | 129,297.63023396263 | the same
gradient's cut, bits | 154,255,360 | 308,396,032
partial cut, bits | 154,255,360 | 308,396,032
carried state, bits | 229,376 | 308,510,720
Delta_grad, bits | 154,484,736 | 616,906,752
affected_all_share | 0.9451416445778911 | 0.9451472802446208
Places that quote 2,298.49 and 114.92:
- TRN-94's description; TRN-89's description ("#8541's grade printed 2298.49 and 114.92").
- TRN-89, my comments of 5:43 PM and 6:02 PM PST. TRN-81, your comment of 3:34 PM PST.
- #8778's body (two lines: the stored point's result, and the fresh-job paragraph). #8712's body ("they were 2,298.49 and 114.92"). #8541's body: neither.
- Run outputs in the store, which stay as what the old rule read: r20261110-050853-fdf0 (#8541's grade), r20261111-012244-3d16 (TRN-89).
- My notes in lanes/training: 20261110T2209Z-checkpoint-trn81-s-holds, 20261111T0022Z-checkpoint-trn89-capture-steps, 20261111T0105Z-checkpoint-trn89-keys-in-trace, 20261111T0142Z-checkpoint-trn89-pr-open-suites-running, 20261111T0145Z-handoff-trn89-trace-keys, 20261111T0219Z-checkpoint-trn94-self-copy-fix. My brief for TRN-89 and my reports to you.
- Not found: the tree (git grep at 9211a72fee5); Notion (two searches). Slack: not searched, I have no search there.
Places that quote the cut and the state (154,255,360 and 229,376): #8778's body, as what the stored source has, which stays true of the stored record; with the fix the stored program reads 308,396,032 and 308,510,720, the fresh job's two figures in that same sentence. My notes 0105Z, 0142Z, 0145Z.

B. THE PUBLISHED ROW (training/algorithm-policy, uncut table art:ae1ceadba2608fb3ef40622eb233ca80d41020bccb78ebe891657c0010383008, te_core 4.13.5): r20261104-001944-1b6b@1260, shallow/strideshare, 36M, gain 1.1. Re-read.
cell | before | after
kinds.sensitivity.reduction | 2252 (2251.689165587458) | 4502 (4501.704212836189)
kinds.sensitivity.update | 132.5 (132.45230385808577) | 264.8 (264.8061301668347)
kinds.sensitivity.fresh_old | 1.611e6 (1,610,561.927) | 1.611e6 (1,610,565.360): the cell's four digits are equal
rho_inputs.fresh_per_c, first entry (fresh_old over c), k 65,536 | 3.00096e-9 | 3.00097e-9
the same, k 80,000 | 2.19443e-9 | 2.19444e-9
the same, k 145,568 | 8.37873e-10 | 8.37877e-10
rho_own 128,563.81910053644; rho_default = rho_console 148,171.79937630607; rho_served 2,722.549696498809; p; every bound term | the same
Places: the uncut table (that art), the builder's tables (art:4fc069e5b6b04c167bdf5a6eaa792e3acb065298952c843e532af3911e6a0a38), and the page's served copy: the builder's cut keeps this row at every tier but coverage alone (I ran its `kept` on the uncut table; I did not read the live page). The finals table quotes rho and p only.
The other three re-read rows (quant/sparse-ternary-qat 606f@3019, quant/embed-sparse-ternary-qat 50ee@1767 and 5029@1493): every figure equal to the last digit; their records are equal on every key.

C. PUBLISHED ROWS NOT RE-READ (reasoned, by reading each run's program and arguments). 293 rows publish the two sensitivities. I read the program of every published run with a cached trace record (385 of 394; the other 9 publish no rho and no sensitivity): 249 programs, 51 distinct copy lines. The only whole-storage copy onto itself on a run's path is Muon's fp32 buffer in the shallow and placement islands' train.py (--opt muon, fp32 buffer; none with --mom-fold). 37 runs take it. 27 of their rows are refused as data-dependent and publish nothing from a trace. 12 publish the two sensitivities: the row in B and these 11, before values as published:
run | method, start | reduction | update
r20261103-225333-9784 | shallow/depthshare, s-t20 | 2255 | 150.4
r20261103-231145-05da | shallow/depthshare, s-t20 | 4536 | 226.8
r20261103-234313-5870 | shallow/depthshare, s-t20 | 1462 | 112.5
r20261104-000307-6d2c | shallow/depthshare, s-t20 | 768.6 | 64.05
r20261104-012613-4882 | shallow/depthshare, m-t20 | 13490 | 281.1
r20261104-061126-5627 | shallow/depthshare, s-t20 | 2256 | 150.4
r20261104-061136-1a68 | shallow/depthshare, m-t20 | 6660 | 195.9
r20261104-062008-fe3a | placement/mlp-in-all, s-t20 | 2069 | 59.12
r20261104-164930-eb14 | shallow/depthshare, s-t20 | 510.9 | 46.45
r20261104-164958-d2d4 | shallow/depthshare, s-t20 | 768.7 | 64.06
r20261104-175922-78f9 | shallow/depthshare, s-t20 | 510.9 | 46.45
After: not computed. Each needs its re-trace, which needs the scorer's pack step (the caches hold records, not packs). Expected by the mechanism: both about double (Muon's matrices pass 32 bits an element where the old rule read 16), fresh_old in its last digits, rho, p and S not at all. The other 281 rows have no such copy on their path and are expected equal. 50 rows carry a delta_grad_bits cell: none of these 12.
Other panels, by reading only: standard-workloads' rows keep the cut and the state of the island's own programs, which have one copy line, between two tensors, and no Muon (0fec9694ff): expected equal. frontier-fit reads C and loss. size-scaling publishes p, rho and S, no sensitivity (238e3e20f8).

WHAT I RAN.
- r20261111-021520-0601, one CPU job at 9211a72fee5 (read from its job.json), torch 2.15.0+cpu as the stored job. Control: the old rule's records equal the stored job's on every key but the two write times. Then old against the fix, and the stored point graded with each rule's rows by this branch's grade: the old rule gives S, 2,298.490550069305 and 114.92452750346524 to the last digit.
- On this VM, the builder's own code (te_core 4.13.5 at 0fec9694ff: formalism.published at the three policies, algorithm_policy.kinds) on those records: under the old rule it gives the published row's cells exactly, all four rows.
- Suites at 9211a72fee5, r20261111-021612-2c1b: 2 suites, 2 passed (training_slowdown 294 passed; its grader 28 passed, 3 skipped).

WHAT I CAN CORRECT, on your word: #8778's and #8712's bodies (both marked; a body edit only), a correcting comment on TRN-89, a correcting note. Not mine: the two descriptions, your TRN-81 comment, the published tables (te_core has the old rule at 4.13.5; 12 runs to re-trace).

Still to do tonight: the reviewer's slow test, the restart note, TRN-94 In Progress with its comment.

