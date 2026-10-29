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
- `slice_inputs(trace, R=, W=, A=, base=, session=, Q=, D=, P=, K=, AS=0)` replays that window from the empty engine
  under sync scheduling, or at `AS=1` under the trace's async scheduling, and refuses it if that schedule isn't the
  trace's or a step falls outside (R, W, A). It returns the slice's statics and its `Slice` (arrivals, decisions, executed steps),
  the ids being session indices, and each step's rows, in `compile_slice_smallkv.json`'s layout. Q, D, P and K
  default to the window's own counts; a larger public class pads the window into it, and a smaller one is refused.
- `served_record(seal, w, requests, base)` gives window w's served rows as the Commit's seal recorded them
  (`SamplerInputsSource.rows`), with the map from a runner's request id to its arrival id, the input of
  `served_rows.step_rows`.
- `python3 replay_vectors.py TRACE --slice W [--served SEAL] [--max-model-len L] [--R R --W W --A A] [--Q Q --D D
  --P P --K K] [--AS 1] --out FIXTURE.json` writes all three as one fixture. At AS = 1 a unit is a `schedule()` and
  the update of the step before it, an arrival is (id, P, O, cap), and the seal's steps are placed at the units that
  ran rows (`by_unit`).

**vLLM session 8eb1** (Llama-3.2-1B under vLLM's AsyncScheduler: max_num_seqs 8, max_num_batched_tokens
2048, max_model_len 512, chunked prefill, no prefix caching, 101,626 KV pages) served 480 requests in 6,169 steps and
339 windows. Its records are `benchmarks/network_e2e/encodings/tests/fixtures/vllm-8eb1/`: the scheduler trace, and
windows 212's and 1's seals.
- The async replay reproduces all 6,169 steps (6,624 decisions, 466 requests arrived on an ask).
- Every window opens on a drained engine, so each is a slice from the empty engine.
- 314 of the 339 windows have the served schedule under sync scheduling too, so the sync slice proves them as
  served. The other 25 need the slice at AS = 1 (below).
- Over the windows: at most 4 requests, 4 running, none waiting, 4 taken in one step (so A = 5), 32 decisions and 54
  async units (52 steps).

**w212** (`verity/core/protocols/certifier/tests/fixtures/compile_slice_w212.json`) is window 212: requests 308–310 of
the session, 20 steps, decisions `[0]*6 + [1] + [0]*6 + [1]`.
- Its statics: (R, W, A) = (2, 1, 2), IW = 9 (the session's 480 ids), Q = K = 3, D = 14, P = 20, at the run's engine
  (`engine`).
- Its keys: `R`, `W`, `A`, `IW`, `Q`, `D`, `P`, `K` and `engine` for `CS.statics`; `arrivals`, `decisions` and `steps`
  for `CS.Slice`; `rows`, the replay's step rows; `served`, the seal's runner rows a step, and `ids`, for
  `served_rows.step_rows(S, served[u], ids)`.
- `CompileSlice_v1` accepts it with the seal's rows in ``srv`` (0x3F; 27 bits from the session's carry), at 117,164
  ANDs; 8eb1 serves without prefix caching, so the slice makes no cache call and ``cache`` is 1. The circuit is its
  reference on it and on its negatives ([compile-slice.md](compile-slice.md))
  (`verity/core/protocols/certifier/tests/test_network_certifier_compile_slice_w212.py`).
- `ReplayStep_v1` at AS = 1 is w212's served scheduler on all 22 of its async units, circuit and reference
  (`benchmarks/network_e2e/encodings/tests/test_served_window.py`).
- At 8eb1's engine class, its AsyncScheduler included (AS = 1, the trace's: `--slice 212 --served SEAL --R 8 --W 1
  --A 9 --Q 4 --D 32 --P 54 --K 4`), w212 is
  `verity/core/protocols/certifier/tests/fixtures/compile_slice_w212_class_async.json`: its 3 arrivals in 4 slots,
  each with its cap, its 14 decisions in 32, and 33 idle steps after its 21 units. Its proof in zero knowledge is
  [compile-slice.md](compile-slice.md)'s.
- At the sync class, the same sizes under a sync scheduler (`--AS 0`), w212 is
  `verity/core/protocols/certifier/tests/fixtures/compile_slice_w212_class.json`: 34 idle steps after its 20, whose
  served rows are zero. The slice takes it as it is: it accepts the window with the seal's rows (0x3F), and each
  negative fails the flags it fails at w212's own statics (the same tests, over both fixtures).
- **The slice** charges by `ChargeLink_v1{32}` over the committed decisions, its carry committed once a window
  (`charge_chain`). At the sync class it is 857,866 ANDs, 97,159 of them `ChargeLink_v1{32}` and 5,509 its copies of
  the words it carries from decision to decision. Its circuit is its reference on the window and every negative
  ([compile-slice.md](compile-slice.md)).
- The statics are public, so a window proved at its own class shows its counts (3 requests, 20 steps). A session
  proves every window at one class, padded with idle steps and empty slots, and that shows only the session's
  maximum: 8eb1's engine class in the table below.

**Cost at the served shapes** (`replay_step.size` and `replay_chain.size`). A step unit is `ReplayStep_v1` and
`ReplayAdvance`, and a window is P step units and one `ReplayLoad`. Under async scheduling a unit is a `schedule()` and
the update of the step before it, so P counts units, a few more than the steps. The AS and EW rows are the chain at
those statics (`replay_chain.Shape`'s AS and EW), its rows `ReplayStep_v1`'s fields. EW = bitlen(NB): the walk entries matter only with prefix caching, and while the no-eviction
bound holds the table has N = NB − 1 entries. 8eb1 serves without prefix caching, so its EW = 17 row is what it would
cost with it.

| shape | (R, W, A) | Q, D, P, K | AS, EW | ReplayStep_v1 | ReplayAdvance | step unit | ReplayLoad | window |
|---|---|---|---|---|---|---|---|---|
| w212 | (2, 1, 2) | 3, 14, 20, 3 | sync | 2,441 | 610 | 3,051 | 845 | 61,865 |
| | | 3, 14, 22, 3 | 1, 0 | 2,522 | 617 | 3,139 | 857 | 69,915 |
| | | | 1, 17 | 2,658 | 681 | 3,339 | 889 | 74,347 |
| 8eb1's largest window | (4, 1, 5) | 4, 32, 54, 4 | sync | 5,478 | 1,186 | 6,664 | 1,736 | 361,592 |
| | | | 1, 0 | 5,698 | 1,235 | 6,933 | 1,792 | 376,174 |
| | | | 1, 17 | 6,089 | 1,363 | 7,452 | 1,856 | 404,264 |
| 8eb1's engine class | (8, 1, 9) | 4, 32, 54, 4 | sync | 11,680 | 1,761 | 13,441 | 1,944 | 727,758 |
| | | | 1, 0 | 12,373 | 1,814 | 14,187 | 2,000 | 768,098 |
| | | | 1, 17 | 13,444 | 2,102 | 15,546 | 2,144 | 841,628 |
| async heavy | (256, 1, 35) | 1000, 1000, 651, 1000 | sync | 524,296 | 80,938 | 605,234 | 789,234 | 394,796,568 |
| | | | 1, 0 | 564,404 | 83,360 | 647,764 | 810,754 | 422,505,118 |
| | | | 1, 18 | 647,330 | 92,576 | 739,906 | 815,362 | 482,494,168 |

- *8eb1's largest window* is the maximum over its windows (w290 has the most units and decisions). *8eb1's engine
  class* holds any window of its engine: R = max_num_seqs, A = R + 1, and W = 1, since no 8eb1 window waits.
- *Async heavy* is the trace corpus's heavy trace under async scheduling: 1,000
  requests at 300 per second, 256 sequences, max_num_batched_tokens 8192, whole prefills, prefix caching, 172,615 KV
  pages. Its corpus window is the whole trace. The replay reproduces its 649 steps in 651 units, every unit in class
  (256, 1, 35).
- Async scheduling adds 3–8% to a step, and the walk entries 5–15% more. At R = 256 about half the step is
  compacting the running list (341,241 of the heavy step's 647,330 ANDs).
- On their own, w212's 20 step units and the load are 61,865 ANDs, about half the slice's 117,164. Most of the rest is
  the charge, `ChargeLink_v1{14}` (50,760, and its 2,359 copies); the served rows' equality is the remainder.
- `ReplayClose`, once a session, is 8 ANDs at w212's shape and 11 at 8eb1's.
- Not counted: `CacheStep_v1`'s calls with prefix caching, whose linear table at N = NB − 1 is far past any unit
  (PROTOCOL.md, "Scaling: offline memory checking"), and the charge, which goes with the decisions, not the steps
  (`ChargeLink_v1{D}`: 97,159 ANDs and 5,509 copies at D = 32).

**Splitting a window into units.** The slice holds a whole window in one unit, its state rows wires between the steps.
That works for 8eb1's windows: at its engine class a window's replay is at most 841,628 ANDs. The async heavy window
is 651 step units of 739,906 ANDs, 4.8 × 10^8 in all, so it splits into the units `replay_chain` already defines,
linked by its rows:
- **Unit 0, the load:** `ReplayLoad`, 815,362 ANDs. It reads state rows 0 and 1 and window rows 0 and 1, and the
  window's K certified arrivals and D decisions at public positions.
- **Unit u = 1 … P, one step:** `ReplayStep_v1` and `ReplayAdvance`, 739,906 ANDs (647,330 and 92,576). It reads
  state rows u and u + 1 and window row 1. A step past the window's executed count is idle and costs the same.
- **The links:** a window commits P + 2 state rows of 20,320 bits and two window rows of 37,024 bits, and makes
  3P + 4 = 1,957 reads, each a frame-v3 path of depth 10.
- **k steps a unit** (not built; `replay_chain` is k = 1 and the slice is k = P): a unit reads state row j and writes
  row j + k, the k − 1 rows between them wires. That divides the state rows and reads by k at k step units a unit.
- **What the split adds to the slice:** no charge state crosses step units. The charge is the load unit's,
  `ChargeLink_v1{D}` over window row 1's decisions, and its carry crosses windows in `charge_chain`'s rows
  ([charge-chain.md](charge-chain.md)).

**For the 25 async windows.** Their sync replay isn't the served schedule, so `slice_inputs` refuses them at AS = 0.
At AS = 1 the step also reads `run_fin R×1 | run_cap R×1 | wait_cap W×1` after `free` and `arr_cap A×1` after
`arr_O`. `replay_chain`'s rows carry them at its AS (PROTOCOL.md, "The rows'"), with an arrival's cap as `q_cap` in the
window row and `a_cap` in the network certifier's slots. The slice takes its chain's shape from its statics
(`compile_slice.chain`), so at an async engine its rows widen and its circuit hands the step these fields.
**w1** is the smallest of them with more than one request: window 1, requests 4–7 of the session, decisions
`[1, 1, 1]`, 38 steps; its sync replay diverges from the trace at step 2.
- At 8eb1's engine class under its async scheduling it is
  `verity/core/protocols/certifier/tests/fixtures/compile_slice_w1_class.json` (`--slice 1 --served SEAL --R 8 --W 1
  --A 9 --Q 4 --D 32 --P 54 --K 4 --AS 1`, the seal's records being `seal-w1.jsonl`): AS = 1, EW = 0 (no prefix
  caching), 38 units, the last schedule on the drained engine dropped. Its charge is from the session's carry after
  window 0, whose decisions `[1, 1, 1]` charge 24 bits; w1's then charge 21.
- `ReplayStep_v1` at AS = 1 and the class is w1's served scheduler on all 39 of its async units, circuit and reference
  (`benchmarks/network_e2e/encodings/tests/test_served_window.py`).
- `CompileSlice_v1` takes it at AS = 1, with two parts that sync didn't need:
  - **The reference at AS = 1.** It reads its entries by `replay_chain`'s names at its shape (`running_names`,
    `waiting_names`, `arrival_names`) in `_state`, `_row`, `_inputs` and `check`'s decoding of `cert` and `arr`,
    where it built sync tuples and `case` raised `MalformedRow` on w1's first state row. The build fed the step by
    name already.
  - **The served rows matched by request.** `served_rows.step_rows` gives a row the slot of its position in the
    runner's batch, and at w1 that isn't the scheduler's running slot at two units: at unit 0 the batch is requests 4,
    6, 5, 7, and at unit 5 request 5 holds slot 0 while its last output is in flight, so request 6, in slot 1, is the
    batch's first row. The rows are the replay's at every unit as (request, kind, start, n)
    (`verity/core/protocols/certifier/tests/test_network_certifier_compile_slice_w1.py`). At AS = 1 ``wit`` carries
    ``map``, R × R bits a step, and ``rows`` holds served row b to the slot its map row names: at most one 1 in each
    map row and column, a slot's column set iff the slot has a row, and the served row equal to that slot's (an empty
    map row's served row zero). The map is then unique, the identity when the batch is in slot order, so the check is
    as strict as the slot-for-slot one, which AS = 0 keeps. It costs R² products of a slot's row and 2·R·(R − 1)
    one-to-one ANDs a step.
- The slice accepts w1 (0x3F, 21 bits) at its own statics (4, 1, 5), Q 4, D 3, P 38, K 4
  (`verity/core/protocols/certifier/tests/fixtures/compile_slice_w1.json`) and at the class. The negatives fail as
  at w212: late admission 0x2E (`load` and `rows`; at the class only, past w1's own D), charge 0x37, longer prompt
  0x2F, dropped row 0x2F. So does the one the map opens: unit 0's first two served rows (requests 4 and 6) with their
  requests swapped, 0x2F. The circuit is its reference on all of them: at w1's own statics 314,936 ANDs, and at the
  class 1,021,302, of which the map is 123,120: its products P·R²·32 = 110,592 at the class's 33-bit rows (P 54,
  R 8), its one-to-one checks 54·2·8·7 = 6,048, and 6,480 in the zero check over its 20,736 mismatch bits, against
  the 14,256 bits the slot-for-slot comparison reads.
- The walk entries (`run_pfx`, `row_pfx`) come only with prefix caching, which 8eb1 doesn't serve.
