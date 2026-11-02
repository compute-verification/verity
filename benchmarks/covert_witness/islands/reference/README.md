# Island `reference`: lower W*, the least |C|ρ that reaches a target loss, at every policy

This is the island's program for a researcher agent (autoresearch's program.md; the tree names it README.md). The
search brief says what the search is for; this file says how a candidate runs and what it must not do.

**Nothing is capped** but the fixed training-FLOP budget and GPU-second cap (below): any update, any size or format of E,
any layers. te_core scores what the run did by the training formalism v1.

**Seed** (`train.py` as committed): AdamW + Muon continuation of every tensor (Muon on the blocks' matrices, AdamW on the embedding and norms), E every tensor in full.

This is full training: a complete schedule of it is a candidate for honest H, the least |C| of any valid record that reaches the target by a complete schedule. The reference directions (ref-optim, ref-data) search that side; beat this seed.

## Your loop

1. Pick a start (`s50`, `s100`, `m50`, `m100`, and the ladder's starts once `starts_store.json` lists them; m100 is the
   main one) and a hypothesis.
2. Edit `islands/reference/train.py` only (or your direction's `islands/<direction>/`).
   Commit it: the run takes only that island from HEAD, sent as a tar and named in the record, and measures it with
   the pinned commit's fixed side (`pin.py`).
3. Run the candidate: `python benchmarks/covert_witness/launch.py START --island reference --note "HYPOTHESIS" -- TRAIN_ARGS`
   (`--island` defaults to your direction's island). It runs on
   the next of nodes.txt's nodes (one GPU, the node's `gpu` slot pool), waits for it, puts the run's `out/` in the evidence store
   (`research data put --kind evidence/v1 --tree ... --preserve`) and prints one JSON row (also appended to
   `~/.research/covert-witness.jsonl`): `valid`, `bpb`, `E_bytes`, `phi`, `lowest_updated_layer`, `train_flops`
   (the harness's), `gpu_seconds`, the checks, the run, the art id, and the te_core ledger's ids. `--no-wait` returns at
   once; `--collect RUN` collects later.
4. Keep or discard: keep the commit when the row is `valid` and `python -m te_core.frontier --formalism` shows its
   records lowering W* (or, for an honest method, H) at some target and policy for that start; otherwise revert it.
   Every candidate is logged either way: its record is in the store, and the ledger keeps the row.
5. Report each kept result as the run id, its art id, the witness record's `score`, `E`, `structure` and `train`, and
   the frontier's W*, H and S at the reference policy.

Keep at most two candidates of this island in flight, so the islands share the node.

## The score and the budget

- Score (`python -m te_core.frontier --formalism`): per start, target ℓ* (a drop below the start's bpb,
  `te_core.frontier.TARGETS`) and policy (c, k) (`te_core.policy`'s grid, in decades from one token's inference block to
  the start's server block): W* is the least |C|ρ of a valid record that reaches ℓ*, honest H the least |C| of a valid
  record that reaches ℓ* by a complete schedule, and S = W*/H ≥ 1. |C| is the harness's count in Boolean gates
  (`te_core.gates`); ρ is 1 where Fit holds and Cover's target elsewhere (`te_core.formalism`). bpb is `score.bpb`, bits
  per byte on a fresh slice of the search pool (shard 013, data version v2.2) at context 1024, computed by the
  evaluation job (`evaluate.py`) on decode(θ, E) after your run ends. Compare only within a start.
- Budget: train.py runs inside the harness (`te_core.run --script`, `te_core/README.md`), which counts every
  multiply-accumulate it computes through torch and times its steps. 2x its MACs ≤ `prepare.TRAIN_FLOP_BUDGET` x
  `prepare.HARNESS_FLOP_SLACK` (9e16 x 1.1: the count includes the elementwise and optimizer work `prepare.train_flops`
  leaves out, 1.06-1.09x of it on the reference seed), and its steps' wall seconds ≤ `prepare.GPU_SECONDS_CAP` (3000: the
  harness runs eager and counted, about 2.4x a compiled run, and stops a run at 1.2x the cap). A step is the work from one
  `prepare.TrainData.next` call to the next. Your step log is still priced (the witness record's `train`) but doesn't
  decide the budget.
- Curve: the harness keeps the first E you encode in each 5% of the run, and the one you submit; the evaluation job
  scores each, and each is a record in the ledger. Any checkpoint that reaches ℓ* counts for W*; for H only a complete
  schedule's does (a run's last checkpoint).
- The formalism's quantities: ingest writes each checkpoint's (`te_core.formalism.of_record`): gates and token blocks
  per step structure, the gates that depend on the update, and Δ_grad where te_core measures it. τ, a, u(x) and Δ are
  stubs until te_core measures them (τ two token ids, a = d_model x 16 bits, u spread evenly with u(1) = Δ = 8·|E|),
  each labelled in the row, so today what lowers ρ is a smaller E and a smaller dependent share. A step whose weights you
  declare (`prepare.declare_step`) is still checked as a counterexample family (te_core's README) and recorded.

## The witnesses, which evaluate.py checks

θ's digest; E and its format; decode(θ, E), which is the model (the one you pass to `submit` isn't kept), with θ's
tensors, shapes and dtypes (so the model's size and its evaluation FLOPs and weight bytes per byte are the start's); 8·|E| bits; the declared `changed` set,
which must cover every tensor that differs, from which φ (the share of forward weight-matmul FLOPs whose weight differs and
has an updated node upstream) and the lowest updated layer are computed; `trainable`, which must cover `changed`; the
harness's record, which must be this E's and find nothing invalid (tokens computed outside a step, weights written where
the counter can't see, a failed script), and its FLOPs and seconds; the training job's process tree, which must have
done no work the harness didn't count (`tree_counted`). `valid` is every check passing.

## Rules

- Never edit the fixed side (`prepare.py`, `evaluate.py`, `candidate.sh`, `sandbox.py`, `launch.py`, `pin.py`,
  `te_core/`, `data.py`, `starts.py`, `model.py`, `sealed.py`, `promote.py`; the run uses the pinned commit's anyway),
  and never run `evaluate.py` yourself. train.py runs in a sandbox with your island, the starts, `train.bin`, the
  Python and its output directory only, read-only but the output: no held-out set, sealed side, `val.bin` or network.
  Never read `prepare.UNREADABLE`, and start no process: its CPU seconds count against the run (`tree_counted`). Each score you see
  is on a fresh slice of the search pool sampled after your weights are hashed; promotions are decided on a sealed set, and
  the final set is never scored for the search.
- Train only on `prepare.TrainData` (the start's tokenizer's training stream) or data you generate from θ.
- train.py may not name the harness's internals or a route around torch's dispatcher (`te_core/run.py`'s `FORBIDDEN`:
  Triton, CUDA graphs, custom extensions, subprocesses and the like); the harness refuses such a file.
- Report any loophole in the evaluator, the checker or the harness instead of exploiting it: work the harness doesn't
  count, a way to make E smaller than the update it carries, a model that differs from decode(θ, E) and passes, a way to
  the held-out sets. A decoder change is an evaluator change (ask, don't add one).
