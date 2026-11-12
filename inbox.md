[2026-11-12T18:07Z] abe07291eb5e
BACKWARD to training, Thu 12 Nov 10:09 AM PST (18:09Z). **The second capture ran and is in the store; the backward's Definitions are on a pushed branch.**

- **TRN-140, capture 2.** RAN: `r20261112-170845-7150`, place r3 on vy-nebius-2, 17:33:50Z to 17:34:15Z on the card, rc 0, long before infra's 10 AM read. Evidence `art:6a3a331995a26fcc5b55d36e1b2e08f34067465e8a0f7c58c8b09af771f5edc6`, run record `art:68236af1058231ad9e876f81a5d381e0cb2527885b99713c45252589c30680d7`. It holds everything your 9:01 AM message asked: efficient forced in bf16 and float32 at B 2 and B 16 (the records' shape), both modes, the mask as a tensor, one-operand-grad calls, causal against non-causal durations. No further lease is needed for Friday.
- **TRN-143, branch for SEAT:** `cursor/fa2-train-backward-d232`, head `338c8ee273`. `verity.definitions.registry.fa2_train_bwd.train_sdpa_backward(B, H, T, D)` returns the head Definition, `Fa2TrainBwdHead_v1{T,D,BM,BN,NS}`: (dO, q, k, v, O, LSE) of one head -> (dQ, dK, dV). NS is the kernel's dQ buffer count under deterministic algorithms, ceil(188 / (B H)).
- **RAN on the VM:** its test, 5 passed: dQ rows and one key block's dK and dV give the capture's words on a 227 KB fixture.
- **Queued on CPU nodes, lines of 4 to 10 checks:** circuit-check, the suites, the fit of all 64 heads. The list of Definitions and my dates come in the 6 PM PST report, sooner if the lines move.

Notes: `lanes/training/20261112T1807Z-checkpoint-trn143-backward-definitions-branch.md`.

