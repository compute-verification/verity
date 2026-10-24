Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The replay at the served shape

How a real served window becomes `CompileSlice_v1`'s input, and what the replay's parts cost at the shapes served
windows need: `ReplayStep_v1` (`replay_step`), with `replay_chain`'s `ReplayAdvance` beside each step and `ReplayLoad`
once a window.

**A served window as the slice's input** (`benchmarks/network_e2e/encodings/replay_vectors.py`):
- `served_window(records, w)` cuts window w out of a served scheduler trace (the recorder's `advice_trace.jsonl`): the
  header, the arrivals the engine core added before the window's first step, and its steps, renumbered from that step.
  It also returns the session index of the window's first arrival. It refuses a window the trace doesn't open and
  close, and one that opens on an engine still holding a request.
- `slice_inputs(trace, R=, W=, A=, SEG=, base=, session=, Q=, D=, P=, K=)` replays that window from the empty engine
  under sync scheduling (the slice's `ReplayStep_v1` is sync) and refuses it if that schedule isn't the trace's or a
  step falls outside (R, W, A). It returns the slice's statics and its `Slice` (arrivals, decisions, executed steps),
  the ids being session indices, and each step's rows, in `compile_slice_smallkv.json`'s layout. Q, D, P and K
  default to the window's own counts; a larger public class pads the window into it, and a smaller one is refused.
- `served_record(seal, w, requests, base)` gives window w's served rows as the Commit's seal recorded them
  (`SamplerInputsSource.rows`), with the map from a runner's request id to its arrival id, the input of
  `served_rows.step_rows`.
- `python3 replay_vectors.py TRACE --slice W [--served SEAL] [--max-model-len L] [--R R --W W --A A] [--Q Q --D D
  --P P --K K] --out FIXTURE.json` writes all three as one fixture.

**vLLM run r20261024-141912-8eb1** (Llama-3.2-1B under vLLM's AsyncScheduler: max_num_seqs 8, max_num_batched_tokens
2048, max_model_len 512, chunked prefill, no prefix caching, 101,626 KV blocks) served 480 requests in 6,169 steps and
339 windows. Its records are `benchmarks/network_e2e/encodings/tests/fixtures/vllm-8eb1/`: the scheduler trace, and
window 212's seal.
- The async replay reproduces all 6,169 steps (6,624 decisions, 466 requests arrived on an ask).
- Every window opens on a drained engine, so each is a slice from the empty engine.
- 314 of the 339 windows have the served schedule under sync scheduling too, so today's sync slice proves them as
  served. The other 25 need the slice at AS = 1 (below).
- Over the windows: at most 4 requests, 4 running, none waiting, 4 taken in one step (so A = 5), 32 decisions and 54
  async units (52 steps).

**w212** (`verity/core/protocols/certifier/tests/fixtures/compile_slice_w212.json`) is window 212: requests 308–310 of
the session, 20 steps, decisions `[0]*6 + [1] + [0]*6 + [1]`.
- Its statics: (R, W, A) = (2, 1, 2), IW = 9 (the session's 480 ids), Q = K = 3, D = 14, P = 20, SEG = 64, at the
  run's engine (`engine`).
- Its keys: `R`, `W`, `A`, `IW`, `Q`, `D`, `P`, `K`, `SEG` and `engine` for `CS.statics`; `arrivals`, `decisions` and
  `steps` for `CS.Slice`; `rows`, the replay's step rows; `served`, the seal's runner rows a step, and `ids`, for
  `served_rows.step_rows(S, served[u], ids)`.
- `CompileSlice_v1` accepts it with the seal's rows in ``srv`` (0x3F, a 13-bit code), at 222,668 ANDs; 8eb1 serves
  without prefix caching, so the slice makes no cache call and ``cache`` is 1. The circuit is its reference on it and on
  four negatives: a decision flipped (``load`` and ``rows``; the decision it leaves unread carries), a charge one bit short
  (``charge``), an arrival's prompt length changed (``rows``), and a served row dropped (``rows``)
  (`verity/core/protocols/certifier/tests/test_network_certifier_compile_slice_w212.py`).
- `ReplayStep_v1` at AS = 1 is w212's served scheduler on all 22 of its async units, circuit and reference
  (`benchmarks/network_e2e/encodings/tests/test_served_window.py`).
- At 8eb1's engine class, w212 is `verity/core/protocols/certifier/tests/fixtures/compile_slice_w212_class.json`
  (`--slice 212 --served SEAL --R 8 --W 1 --A 9 --Q 4 --D 32 --P 54 --K 4`): its 3 arrivals in 4 slots, its 14
  decisions in 32, and 34 idle steps after its 20, whose served rows are zero. The slice takes it as it is: its
  reference accepts the window with the seal's rows (0x3F), and each negative fails the flags it fails at w212's own
  statics (the same tests, over both fixtures).
- `CompileSlice_v1` as it stands can't build its circuit at the class. The builder's wires are GF(2) forms over
  committed bits, and the coding carries the coder's state (`low`, `high`, `x`, `used`), the KT counts and the code
  position from ask to ask without committing them, so each ask adds about 19 bits to their forms. The class has
  P·A = 486 asks, so the build's nonzeros are quadratic in the window: nnz_AB is 23.5 million at P = 2 and 97.8
  million at P = 4 (82,813 and 193,566 ANDs), and at P = 54 the build passed 352 GB unfinished (run
  r20261024-184641-e2af).
- Measured with those words committed once an ask (`Circuit.commit`, a patch of the build that the slice doesn't make
  yet), the forms stay linear in the window: nnz_AB is 4.6 million at P = 2 and 24.3 million at P = 8. At the class the
  unit is 3,181,376 ANDs, 66,815 of them the commit rows (3,114,561 without them), with nnz_AB 176.7 million, built in
  142 s at 15.3 GB. Its circuit is its reference on the window and the four negatives (0x3F, 0x2E, 0x37, 0x2F, 0x2F;
  run r20261024-201333-a08a). At w212's own statics the same patch adds 5,425 commit rows to the 222,668 ANDs and
  cuts nnz_AB from 119.6 million to 11.9 million.
- The statics are public, so a window proved at its own class shows its counts (3 requests, 20 steps). A session
  proves every window at one class, padded with idle steps and empty slots, and that shows only the session's
  maximum: 8eb1's engine class in the table below.

**Cost at the served shapes** (run r20261024-175929-6f49). A step unit is `ReplayStep_v1` and `ReplayAdvance`, and a
window is P step units and one `ReplayLoad`. Under async scheduling a unit is a `schedule()` and the update of the
step before it, so P counts units, a few more than the steps. The chain's rows don't carry AS or the walk entries yet
(PROTOCOL.md, "Open"). The AS and EW rows count the chain with the fields that item names, checked equal to
`ReplayStep_v1`'s own. EW = bitlen(NB): the walk entries matter only with prefix caching, and while the no-eviction
bound holds the table has N = NB − 1 entries. 8eb1 serves without prefix caching, so its EW = 17 row is what it would
cost with it.

| shape | (R, W, A) | Q, D, P, K | AS, EW | ReplayStep_v1 | ReplayAdvance | step unit | ReplayLoad | window |
|---|---|---|---|---|---|---|---|---|
| w212 | (2, 1, 2) | 3, 14, 20, 3 | sync | 2,441 | 620 | 3,061 | 734 | 61,954 |
| | | 3, 14, 22, 3 | 1, 0 | 2,522 | 627 | 3,149 | 746 | 70,024 |
| | | | 1, 17 | 2,658 | 691 | 3,349 | 778 | 74,456 |
| 8eb1's largest window | (4, 1, 5) | 4, 32, 54, 4 | sync | 5,478 | 1,199 | 6,677 | 1,352 | 361,910 |
| | | | 1, 0 | 5,698 | 1,248 | 6,946 | 1,408 | 376,492 |
| | | | 1, 17 | 6,089 | 1,376 | 7,465 | 1,472 | 404,582 |
| 8eb1's engine class | (8, 1, 9) | 4, 32, 54, 4 | sync | 11,680 | 1,774 | 13,454 | 1,560 | 728,076 |
| | | | 1, 0 | 12,373 | 1,827 | 14,200 | 1,616 | 768,416 |
| | | | 1, 17 | 13,444 | 2,115 | 15,559 | 1,760 | 841,946 |
| async heavy | (256, 1, 35) | 1000, 1000, 651, 1000 | sync | 524,296 | 80,959 | 605,255 | 769,234 | 394,790,239 |
| | | | 1, 0 | 564,404 | 83,381 | 647,785 | 790,754 | 422,498,789 |
| | | | 1, 18 | 647,330 | 92,597 | 739,927 | 795,362 | 482,487,839 |

- *8eb1's largest window* is the maximum over its windows (w290 has the most units and decisions). *8eb1's engine
  class* holds any window of its engine: R = max_num_seqs, A = R + 1, and W = 1, since no 8eb1 window waits.
- *Async heavy* is TRACE-CORPUS's heavy trace under async scheduling
  (art:e86ec6d2d6b50e1469013dea2376897af1fcc329f8661c8efc06504f8ce99d3a, from run r20261024-064542-ae66): 1,000
  requests at 300 per second, 256 sequences, max_num_batched_tokens 8192, whole prefills, prefix caching, 172,615 KV
  blocks. Its corpus window is the whole trace. The replay reproduces its 649 steps in 651 units, every unit in class
  (256, 1, 35). Its sync `ReplayAdvance`, 80,959 ANDs, is the corpus run's at sync heavy (r20261024-091809-297f).
- Async scheduling adds 3–8% to a step, and the walk entries 5–15% more. At R = 256 about half the step is
  compacting the running list (341,241 of the heavy step's 647,330 ANDs).
- On their own, w212's 20 step units and the load are 61,954 ANDs, under a third of the slice's 222,668. Most of the
  slice is the asks' coding, the charge and the served rows.
- Not counted: `CacheStep_v1`'s calls with prefix caching, whose linear table at N = NB − 1 is far past any unit
  (PROTOCOL.md, "Scaling: offline memory checking"), and the coding, which goes with the decisions, not the steps.

**Splitting a window into units.** The slice holds a whole window in one unit, its state rows wires between the steps.
That works for 8eb1's windows: at its engine class a window's replay is at most 841,946 ANDs. The async heavy window
is 651 step units of 739,927 ANDs, 4.8 × 10^8 in all, so it splits into the units `replay_chain` already defines,
linked by its rows:
- **Unit 0, the load:** `ReplayLoad`, 795,362 ANDs. It reads state rows 0 and 1 and window rows 0 and 1, and the
  window's K certified arrivals and D decisions at public positions.
- **Unit u = 1 … P, one step:** `ReplayStep_v1` and `ReplayAdvance`, 739,927 ANDs (647,330 and 92,597). It reads
  state rows u and u + 1 and window row 1. A step past the window's executed count is idle and costs the same.
- **The links:** a window commits P + 2 state rows of 20,320 bits and two window rows of 37,024 bits, and makes
  3P + 4 = 1,957 reads, each a frame-v3 path of depth 10.
- **k steps a unit** (not built; `replay_chain` is k = 1 and the slice is k = P): a unit reads state row j and writes
  row j + k, the k − 1 rows between them wires. That divides the state rows and reads by k at k step units a unit.
- **What the split adds to the slice:** the asks' coding runs across steps, so a split slice also carries the encoder's
  state from unit to unit (its interval, the KT counts, the plain context and the codeword's position). That's a group
  `replay_chain`'s state row doesn't have yet.

**For the 25 async windows.** The slice feeds `ReplayStep_v1` through its `_in_fields` and zeroes what its rows don't
carry, and its engine has no `async_scheduling`. At AS = 1 the step also reads `run_fin R×1 | run_cap R×1 |
wait_cap W×1` after `free` and `arr_cap A×1` after `arr_O`. In `replay_chain` that is PROTOCOL.md's Open item, which
also puts `q_cap` in the window row and `a_cap` in the warden's slots. `slice_inputs` then takes the engine's mode as it
is. The walk entries (`run_pfx`, `row_pfx`) come only with prefix caching, which 8eb1 doesn't serve.
