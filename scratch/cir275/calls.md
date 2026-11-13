# What each call in the list of kernel calls must carry

At `33e504b391bf843bff35a165bf3f87e9d2fa55b4`. Paths are under `adapters/vllm/verity_vllm/program/`; `file:line` is where the
field is read or set today.

There is no list of kernel calls yet. The decoder writes each call straight into a `verity.circuits.defs.Builder`:
`_Model.call` (`predict/build.py:461-462`, `self.B.call(fn, *args)`), with the Builder made at `predict/build.py:739`. The
fields below are what the kernels' circuits and the cut read today, either from that Builder's body or from the per-request
plan beside it (`predict/steps.py` `RequestUnit`). The nearest thing to the list today is `predict/step_needs.py`. Its
`StepNeeds` (74-274) gives each Definition by id, with its calls, for one DSA engine step, and `Need` (35-41) holds a
(route, statics) pair with its id or its gap. It names Definitions and counts calls. It carries none of the places below.

## Per call

1. **Kernel.** The role the decoder asks for: `common.kind(role, **statics)` (`predict/common.py:23-33`), resolved by the
   vocabulary `PROFILE_B1_EAGER_V3.kind` (`predict/common.py:28`; `frontend/rules/vocab.py`). The roles are bound in
   `_Specs.__init__` (`predict/build.py:295-355`), `_Specs.linear`/`gemm_bias`/`attention` (`predict/build.py:360-399`),
   and inline in `predict/dsa.py`/`dsv4.py`'s `_Model`. The cut reads the resulting Definition's family and id:
   `window_circuit._Program.lower` (`predict/window_circuit.py:304-313`, `node.form`, `node.fn.name` in `FAMILIES`,
   `node.fn.id`). `window_rows.Served` refuses a served row of another Definition than its class node's
   (`predict/window_rows.py:356-357`).
2. **Settings.** The statics of that role, as `_Specs` passes them:
   - linear shapes `K`, `N` (`predict/build.py:309, 353-354, 360, 367`);
   - `DOT`, the target's k16 step (`predict/build.py:293, 348, 360`);
   - norm `N`, `EPS`, `TARGET`, `M` (`predict/build.py:300-301, 310, 313`);
   - embedding `V`, `H` or `VS`, `H`, `START` (`predict/build.py:295, 298`);
   - collective `N` (`predict/build.py:299, 315`);
   - rope `NHEADS`, `D` (`predict/build.py:319`);
   - activation `I` (`predict/build.py:320`);
   - MoE `E`, `K`, `N`, `TOPK`, `H` and the router's form (`predict/build.py:345-355`);
   - attention `T`, `NH`, `KVH`, `D`, `CAP`, or under target-keyed kinds `BN`, `DOT`, `INV`, `MASKED_FROM`,
     `FA2_MASKED_FROM` (`predict/build.py:388, 393-399`);
   - the scalar chains' `N`, `C` (`predict/build.py:322-331`).

   The target facts these come from are read through `frontend/targets.py`: `uses_accepted_kinds` (511-513),
   `tensor_core_dot` (134-139), `moe_expert_dot` (142-147), `attention_geometry` (500-508), `fa3_masked_from` (296-300),
   `fa2_masked_from` (311-315), `fa2_check_inf` (303-308) and `fa3_check_inf` (289-293), all over a `TargetProfile`
   (`frontend/target_profile.py:53-624`; `target.int8_parts` at `predict/build.py:282`). The cut reads statics back off the
   bound Definition (`node.fn.bindings`, e.g. `predict/window_circuit.py:321`).
3. **The launch it ran in**, wherever a setting reads it:
   - the query row's index among its sequence's rows, the sequence's own rows and the launch's `max_seqlen_q`
     (`_Specs.attention`, `predict/build.py:378-399`; computed in `_Model.forward`, `predict/build.py:535-542`, by
     `common.launch_seqlen_q`, `predict/common.py:351-363`);
   - the CUDA-graph mode and the tokens the forward runs at, every query row's key count, and the step's prefills
     (`common._Launch.graph`/`tokens`/`keys`/`prefills`, `predict/common.py:229-292`, over `frontend/targets.py:406-497`).
4. **Its step and position.** Which request step and position the call belongs to:
   - the forward's `pos0` and rows (`predict/build.py:494-559`);
   - the advice row `RowRef(request, digest, start, n, samples)` (`predict/steps.py:192-199`);
   - the cut labels each unit by (slot, step), a position opening at each embedding
     (`predict/window_circuit.py:314-317`, `window_rows.positions`, `predict/window_rows.py:92-98`).
5. **Each input's source.** Either another call's output, as that call's words [lo, hi), or a parameter's words. The
   parameters are `input_ids`/`prompt`, `seed` and the seed rule's parameters, `prefix`, `last_token`, `peers_k` and the
   `weights` record field (`predict/build.py:723-741`). The cut reads them as `("n", node, lo, hi)` or `("p", name, lo, hi)`
   (`window_circuit._Program.segments`, `predict/window_circuit.py:241-256`). It reads the writer rows behind them
   (`rows`, 258-274), a weight read as one run of a field (`weight`, 294-299), and each port as a read or a feed (`_one`,
   369-385).
6. **Each output.** The call's return fields and their widths (`predict/window_circuit.py:306-308`,
   `node.fn.signature()[1]`), laid out as the unit's output rows (`predict/window_circuit.py:382-384`).
7. **Where the server committed each input and output.**
   - For every served row, `{spec, step, op_path, members, row}` (`window_rows.Served`, `predict/window_rows.py:340-468`):
     the Definition spec the record binds it at, the request step, the op path (module/fx node), the member names and the
     row.
   - `read(step, op_path, member, lo, hi)`, a member's elements at the request's arrival plus its step, from
     `CommittedStore.words`; and `bound(step, op_path, member)` (`predict/window_rows.py:341-344, 358-360`).
   - The cut matches calls to those rows by site: the op path less its last part, the members and an ordinal
     (`_sites`, `predict/window_rows.py:304-314`). It respells op paths across arrivals (`_respelt`, 280-301) and folds
     narrowed linears (`_fold_narrowed`, 321-336).
   - Weights are committed per parameter `weights/<label>@<words>` (`window_weights._param`,
     `predict/window_weights.py:52-60`) as hm96 rows (`weight_value`, 114-125) and pinned by root (`registration_pins`,
     307-324).
   - The scan state's rows are committed by `window_rows.state_value` (`predict/window_rows.py:701-715`).
8. **Its peers**, for a collective at TP > 1: the peer Inputs it reads, in call order (`common._Peers`,
   `predict/common.py:182-204`; shapes `predict/build.py:731-736`), which `tp_binding.py` binds across ranks.

## Per request (the list's header)

These come from `RequestUnit` (`predict/steps.py:203-229`).

- request index and id (209, 216): `WorkloadRequest` (`predict/steps.py:532`);
- `LP`, `T`, `first_step` (211-212, 215): `request_program` (473-485), then `build_program`
  (`predict/build.py:681`), and `predict/steps.py:532-533`;
- `C` and `producers`, the cached prefix (217-218, `prefix()` 240-245): `predict/build.py:727-728`,
  `predict/steps.py:533`;
- `policy` and `seed` (213-214): `predict/build.py:709, 724-725, 744-748` (`_Stochastic`: the seed Input, `S` splits,
  temperature and top-p bits);
- `stop`, its length rule (225): `with_checks` (`predict/build.py:764`);
- `edge`, the history groups a window carries out (224): `with_edge` (`predict/build.py:765`), `edge_of`
  (`predict/steps.py:531`);
- `launch`, `chunks`, `entries` (222-223, 229): `predict/build.py:752-755` (`launch_batch`, `in_launch`, `launch_rows`);
- the target facts GP-01 reads: `batch_invariant` and `lag()` (`predict/steps.py:531, 534`).

## Not established

- Whether the cut needs item 3 at all once item 2 is given. Today the launch enters only through settings (`MASKED_FROM`,
  `BN`, the graph mode). Its reads at `predict/build.py:535-542` are the decoder's, not the cut's.
- Item 7 is read today by op path and member, which are vLLM's names (fx nodes, module paths). Whether the list carries
  those names or a place the server assigns per call is open: the agreed line says only "where the server committed each
  input and output". Nothing in `program/` assigns such a place today.
- `predict/dsa.py` and `dsv4.py` bind their kinds inline. Their fields were not traced call by call; the field list above
  is read from `build.py`'s `_Specs`, `steps.py` and the window lowering.
