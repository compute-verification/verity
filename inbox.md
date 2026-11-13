[2026-11-13T16:00Z] bad9e68e8127
WALKER, Fri 13 Nov, 8:00 AM PST.

**The node run does not name what stops s-t20. At s(2,64,512) the five largest rises are all one thing, the walk copying the whole optimizer state, new in #9558; at s-t20 that is 0.8 GiB a copy, about 5 GiB in all, against the 91 GiB rise measured there. Whatever rises at s-t20 is under 0.05 GiB at this shape.**

- **The run** (RAN: r20261113-154055-e13f, #9558 with #9676 and #9730 merged, 7d20a8576a, s(2,64,512), 16 GB, alone): build 49.9 s, reading before the pass 102.3 s, pass 227.3 s over 64,233 segments, 70 claims, none refused, 1 point. Resident 2.235 GiB at the pass's start, 5.088 GiB at its highest.
- **1. The five largest rises** (RAN; the state is θ and AdamW's two moments, 58,609,152 words, 8 bytes a word in the walk):

  | GiB added | where in the walk | Definition | what is taken | words | kept after |
  |---|---|---|---|---|---|
  | 0.458 | the pass's start, before any node | `Root[Loop_v1]` | every input sampled (`Walk.inputs`) | the state and the batch | yes |
  | 0.438 and 0.437 | `TrainStep`'s return, iterations 2 and 1 | `Derived[TrainStep]_v1` | a `Concat` of 169 parts | 58,609,153 | yes |
  | 0.437 | the scan's first carry, before iteration 1 | `Root[Loop_v1]` | a `Concat` of 168 parts | 58,609,152 | yes |
  | 0.436 | the scan's end, after iteration 2 | `Root[Loop_v1]` | carry and outputs joined | 58,609,153 | no |
  | 0.435, twice | `Iteration`'s return, both iterations | `Iteration_v1` | a `Concat` of 2 parts | 58,609,153 | yes |

  None of the five is an op. The largest op is far below them: `F32Fma_v1` as a batch of 720,896 members in `TrainStep` (node 2497, operands (1,1), (1,720896), (1,720896)), 0.048 GiB, gone when it returns; 51 of the 60 largest segments are such batches. The largest product is `LinearRef_v1{K=128,N=1408}` as a batch of 512 (node 1170, operands (1,65536) and (1,180224)), 0.036 GiB. No row check adds more than 0.02 GiB.
- **Which code** (READ: `points.py` at 7d20a8576a; RAN: `origin/main` at 4ae17f7eafd7 has no `points.py`): **new in #9558**, all five. `Walk.inputs` samples every input. `Walk.take` joins a `Concat`'s parts with `np.concatenate`, and `Walk.scan` joins the carry and the outputs the same way. Each makes a fresh array of the whole state.
- **2. Before iteration 2's first product?** (RAN) Partly. One iteration is 753 products here too, the count both s-t20 runs stopped at. Resident memory is 3.13 GiB at iteration 1's start, about 3.9 GiB after its 753rd product, 4.27 GiB at iteration 2's start, and 4.7 GiB at its highest in between: the optimizer's new state, then `TrainStep`'s return, then `Iteration`'s return, three copies. The pass's highest, 5.088 GiB, is later, at the scan's end.
- **Scaling to s-t20** (INFERRED): each of the five is 8 bytes times the state's words. Exponent 0 in ctx and in batch. In vocab only the tied embedding grows, 3 × vocab × 512 words: 58,609,152 words become 106,974,720, so 0.797 GiB a copy. The pass's whole rise here is 2.85 GiB, 6.5 copies; at s-t20 that is about 5.2 GiB.
- **So this does not explain s-t20** (INFERRED). RAN, b24e: the pass there held 2 to 8 GiB above its start up to the 674th product, then rose 91 GiB in 340 s and was still rising. A segment that reaches 91 GiB there and is under 0.048 GiB here grows by more than 1,900 times. ctx × vocab is 1,000 times from this shape, so it is steeper than that: ctx² × vocab, ctx × vocab², or a higher power of one. I can't name the node from this run: it kept only the 60 largest segments.
- **What would name it, not launched**: the same run at three small shapes in one job, s(2,64,512), s(2,128,512) and s(2,64,2048), keeping every segment with its arrays' sizes. Segment k is the same node at each shape, so each segment's size gives its exponent in ctx and in vocab exactly, and I list those that reach more than 10 GiB at s-t20. About 35 minutes, 16 GB, `--on auto`. Say go, and whether it may run beside the trace or after it.
- **A correction to 7:41 AM.** I wrote that the tracer is slow because it walks a deep call stack. RAN, the trace's smoke: the reading's stack is 13 to 38 frames deep. So the cost is the tracer's fixed cost per allocation (RAN on the VM: about 1 µs a step against 46 ns untraced), not the walk.
- **The trace is out** (RAN): r20261113-154950-9ed5, #9730's head 5efe513c14, s(2,64,512), 12 GB, 2-hour limit, launched 7:50 AM PST when the node run had ended. Its smoke passed. It prints a table at 0.1 GiB traced, then at 3 times the last or 15 minutes after it, at most four before the end's. I expect its end near 8:45 AM.
- **Bodies**: #9730's s-t20 row has no column-case cell now, and says both runs stopped in #9558's pass. #9558's s-t20 section is next, with this run's 753.
- No merge of `main`, no restack. The three heads are unchanged. No fix.

