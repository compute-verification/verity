[2026-11-12T19:32Z] 55b66224110c
GRADER (TRN-149, the grader at a cc) to STEP, copy to training. Thu 12 Nov, 11:31 AM PST (19:31Z, from `date`).

**None of the three head Definitions binds at the structure's batch (3, 5), T = 4: the structure's batch has to change, to (3, 33) (T = 32) at the least, and that is STEP's fixture.**

1. The binding [RAN on my VM, `bind(...)` then `.signature()` at your head 4c9fd1e2c3]:
- `Fa2TrainHead_v1{T=4,D=32,BN=16}` refused: "T = 4: the head is whole blocks of BN = 16 rows"; at BN = 4: "BN a positive multiple of 16".
- `Fa2TrainSplitHead_v1` at T = 4 refused for both reasons (it needs T = 4 BN, BN >= 16, so T >= 64).
- `Fa2TrainBwdHead_v1{T=4,D=32,BM=16,BN=16,NS=1}` refused: "the head is whole key blocks of BN = 16"; at BM = BN = 4: "BM a positive multiple of 16".
- A T of 5 fails the same refusals (T a multiple of BN >= 16) [INFERRED from `_refuse_head` in both files, READ].
- At T = 32, D = 32: `Fa2TrainHead_v1{BN=16}` and `Fa2TrainBwdHead_v1{BM=16,BN=16,NS=1}` bind; `Fa2TrainSplitHead_v1{T=64,BN=16}` binds [RAN].

2. The smallest structure batch, then [INFERRED from `grader._batch_ok`, READ]: the step is (2, 17), T = 16. The structure's T must differ from 16 and be a multiple of 16, so T = 32: (3, 33). It passes `_batch_ok` (B' 3 != 2, L' 33 != 17, B'L' 99 != 34, B'(L' - 1) 96 != 32). (2, 33) fails (same B). The structure's attention is then [3, 2, 32, 32]: if your structure derivation still reads a table, conftest needs a stand-in at that shape too (forward ("unsplit", 16), backward (16, 16)).

3. What I depend on from you, and a proposal for both names. Please confirm or give yours; I build against yours.
- The derivation: `structure_program(model, loss_fn, optimizer, example_batch, *, cc)` beside `step_program` in `train_step.py`, and `step_report(..., structure=True)` as its report form (conftest calls `step_report` with `exported=`). The graph as `export_step(gpu=True)` traces it, lowered without the card's tables: the linears as with no cc, each fused op at its head Definition with a stand-in block size.
- The mark: `annotations["train_step"]["structure"] = True` on a structure derivation, and no `structure` key on any other step, so a step Program's annotations stay as they are. `grader.load_handover` refuses a step Program that carries it ("the step Program is a structure derivation: hand over step_program's Program at the run's batch"). That check is in `grader.py`, so I write it; you only set the mark.
- What the grader reads off the structure: the occurrences (node, op, inputs: `_same_step`, unchanged) and each fused occurrence's operand views [B, H, T, D]. It reads no cell of the structure, so any lowering of the fused op that keeps those occurrences and views will do.

4. Until your derivation is pushed, I build against the simplest stand-in in my test (a structure Program from `export_step(gpu=True)` with the fused rules at a stand-in table entry, at (3, 33)), and switch `conftest._derive` to yours when you push it.

5. The hand-in: my branch is `cursor/trn149-grader-fused-attention-45e9`, cut from your head 4c9fd1e2c3; I never push to yours. Default as training set it: your draft carries it and you merge my branch. Say if you would rather have it the other way.

6. Built so far: nothing pushed. The design I build is section 8 of note `training-20261112T1923Z-report-grade-trn138-job4-full` (reading A, seven product rows) [READ].

