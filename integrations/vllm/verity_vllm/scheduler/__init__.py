"""vLLM's scheduler as Boolean Definitions: the network audit's compiler for a vLLM server, which replays what the
scheduler decided from the committed advice and the certified arrivals.

The advice format and its decoder (advice v0, `verity.core.protocols.certifier.advice`) and the charge, the encoder and
the network certifier's rows stay in `verity.core.protocols.certifier`; what is vLLM's scheduler is here. Each module
takes the circuits and their evaluator from `verity`:

- `replay_step`: `ReplayStep_v1`, one vLLM v1 scheduler step, with `reference` its spec. It takes the committed queue
  state, the certified arrivals and the committed decision bits to the step's rows (request, kind, start, n) and the
  next state, under sync or async scheduling and with `CacheStep_v1`'s walk entries, and refuses what its size class or
  the prefix layer's counts can't express. It is held step for step to the replay
  (`benchmarks/network_e2e/encodings/replay_vectors.py`);
- `advice_check`: `AdviceCheck_v1{L}`, the compile slice's proof block's first part, 1 exactly when the committed bits
  (at most L) decode to a v0 advice, up to the three refusals of `decode` it defers (unique ids, a source's KV computed
  there, release order per request); its AND count at each L is `size(L)`;
- `replay_chain`: `ReplayStep_v1`'s state handed from one step proof block to the next as the state rows of a produced
  value a window, with private heads into the window row every step reads. `ReplayAdvance` sits beside each step and
  checks its arrivals and decision bits are the window row's at the heads; `ReplayLoad` runs once a window and carries
  the decisions the last window left unread; `ReplayClose` ends the session with none unread. The windows join by equal
  openings and the session starts from the empty rows under public salts (`check_chain`). Its cost at a shape is `size`;
- `compile_slice`: `CompileSlice_v1`, with `check` its reference, composes `ReplayLoad`, `ReplayStep_v1` and
  `ReplayAdvance` on shared wires, `CacheStep_v1`'s calls on the certified tokens checking the hits `ReplayStep_v1`
  reads, and `ChargeLink_v1` over the decisions the window commits, from its carry-in to its carry-out, for one window.
  It holds the charge to the declared one and the step rows to its committed ``srv`` row, the served rows. The evictable
  and returned counts are a stand-in held at 0 (`verity/core/protocols/certifier/protocol/compile-slice.md`), and
  `benchmarks/network_traces/compile_slice_zk.py` proves it in zero knowledge;
- `compile_slice_v2`: `CompileSlice_v2`, `CompileSlice_v1` with its prefix cache replaced by `CacheStep_v2`'s pool, its
  calls in vLLM's order on the pool rows in ``wit``, so eviction, preemption and a leaving request's returns are in the
  slice (`benchmarks/network_e2e/encodings/compile_slice_pool.py` holds its calls to the replay's `_KV` over whole
  traces);
- `compile_slice_v3`: `CompileSlice_v3`, `CompileSlice_v2` computing what v2 checks. From the advice (DA decisions and
  the step count), the arrivals' ingress fields and their registered output lengths and tokens, it computes the served
  rows and the state, window and pool rows the window leaves, and holds them to the registered ``srv`` and ``cout`` it
  reads; its one output is the 16-bit verdict. So its only free input is the advice, ℓ_w = DA + PW bits (`free_width`),
  and the charge is ℓ_w;
- `served_rows`: the step rows a served window ran, from the server's kept rows, committed before the sample under the
  run root as a hiding-commitment registration, a row each block of G steps (`compile_slice.CompileSliceBlocks_v1`).
  `leaf_ok` holds a proof block's row to its registered leaf, and `reads_port` is the same read as the Lean verifier's
  registered reads take it;
- `served_tie`: `ServedRowsTie_v2`, with `check` its reference, holds a window's served rows to the positions its steps'
  Commit holds (the step-input port, `port_chunks`), so the rows registered at serve time are the batch the server ran;
- `served_statics`: `ServedStaticsTie_v1`, with `check` its reference, holds each block's registered served rows to the
  RoPE statics of the Programs its steps ran (`verity/core/protocols/certifier/protocol/served-statics.md`);
- `cache_step`: `CacheStep_v1`, one scheduled row's pass over a table of whole cached units, which checks the row's hit
  and owns or appends the units the row computes (`verity/core/protocols/certifier/PROTOCOL.md`, "Advice v1");
- `cache_step_v2`: `CacheStep_v2`, one call's pass over vLLM's prefix-cache block pool: which slots hold each cached
  block, the free queue's order by stamps, eviction named by a threshold stamp the call checks, and a leaving request's
  returns.

Each Definition's circuit-check binding is its module's sidecar (`<module>.circuit-check.py`, pins beside it).
"""
