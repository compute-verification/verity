[2026-11-13T15:09Z] 40d80e35ef8e
WALKER, Fri 13 Nov, 7:09 AM PST.

**b24e stopped itself in #9558's pass at 153.0 GiB, at the same place 54ec did, so s-t20 has no claims, N or `Work` yet. The memory is the pass's, not the reading's: it rose by 91 GiB within 340 s around the pass's 753rd product, and was still rising. I told you at 6:05 AM it had room to finish, and that was wrong.**

- **The end** (RAN: r20261113-114851-b24e, ended 6:57 AM PST, ask 160 GB, stop at 144 GiB):
  - Stopped 4,985 s into the pass, 11,285 s into the run. The pass never returned, so there are no claims, no N and no `Work` to hash.
  - Up to the stop: 753 products evaluated, 12 nonfinite nodes, 0 NaN nodes, 0 origins.
  - Reading before the pass, as sent: 5,387.4 s, highest 58.8 GiB.
- **The pass's memory** (RAN):

  | into the pass | products | resident |
  |---|---|---|
  | 0 s | 0 | 56.8 GiB |
  | 445 s to 4,645 s | 79 to 674 | 59.6 to 64.9 GiB |
  | 4,945 s | 753 | 134.4 GiB |
  | 4,985 s (stop) | 753 | 153.0 GiB |

  The last 18.6 GiB came in 40 s.
- **54ec stopped at the same product** (RAN): 753 products, 5,217 s into its pass. It entered the pass at 124.9 GiB, so it had about 19 GiB of room to its stop, and I read its stop as slow growth in the pass. It was this same rise.
- **Where** (INFERRED, not seen): 753 is one iteration's products. RAN: at s(2,64,512), where the pass finishes, it walks at least 1,423 products over the two iterations, and the nonfinite nodes go from 12 to 24 in the second. So the rise is in what the pass evaluates between iteration 1's last product and iteration 2's first: the end of the backward pass, the optimizer's update, or the scan's carry. RAN: at s(2,64,512) the resident memory steps up between the 674th and the 1,069th product, from 3.6 GiB to a peak of 5.0 GiB.
- **What it means** (INFERRED): s-t20 does not pass at a 160 GB ask under any reader. #9730 leaves 56.8 GiB at the pass's start, and the pass then needs more than 96 GiB on top at that one step. This is #9558's to fix or to state, not #9676's or #9730's.
- **Not known: which node.** The harness prints no Definition or node at a stop. Two ways to find it, neither launched:
  - s(2,64,512) with the pass, the peak reset at each node of the walk, printing the nodes that raise it most: about 7 minutes, 16 GB.
  - s-t20 again with the walk's Definition and node on each heartbeat: about 3 hours, 160 GB.

  I'd take the first. Say go, after the trace.
- **The trace**: I am writing it now and launch it next: one run, `--on auto`, #9730's head at s(2,256,512), 32 GB. The untraced run peaked at 10.15 GiB. The tracer's table, a snapshot and the count of what each cache holds all grow with the number of live blocks, which I don't know before the run; my estimate is 15 to 24 GiB in all. I send the table when it ends.
- Next: #9558's body says s-t20 is not read "because of the reader's memory". I change it to say the pass stops on memory at 160 GB, with both runs. #9730's s-t20 row loses its column-case cell: the run stopped before the reader returned it.
- No merge of `main`. The three heads are unchanged.

