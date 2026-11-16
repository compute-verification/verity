Part of network attestation's [PROTOCOL.md](../PROTOCOL.md).

## The compile slice's charge, and a served execution phase at its engine's class

`CompileSlice_v1` (`verity_vllm.compiler.compile_slice`) is one execution phase of the compile slice's proof block in one Boolean
Definition. Its circuit, rows, verdict flags, reference and stand-ins are `verity_vllm.compiler.compile_slice`'s
docstring. This file covers how the slice charges an execution phase, what binds the charge's inputs, and a served execution phase
at its engine's class. `CompileSlice_v4` (`verity_vllm.compiler.compile_slice_v4`) is the slice computing its schedule rather
than checking it, with vLLM's block pool in place of the evictable and returned counts' stand-in, its pool calls
answered by `CacheStep_v3`'s memory-checked stream. It has no ChargeLink: the served rows and last rows are registered
reads it holds to the computed ones, and its advice input is exactly ℓ_w gates wide. Until a Lean lemma shows its
checked witness unique, that witness counts as free input at its width, so its charge is ℓ_w plus the witness's width
(`free_width`). `CompileSlice_v5` (`verity_vllm.compiler.compile_slice_v5`) is v4 at its execution phase's activity: the keys are
one a block the execution phase keys, from a ragged token row registered with each slot's keyed block count, RB's touched list
is the addresses the stream touches, and each running slot's request number is carried in the slot row ``cout``
registers. Its charge is v4's, at its own checked witness's width. `CompileSlice_v6` and `PoolWindow_v1`
(`verity_vllm.compiler.compile_slice_v6`) are v5 cut in two proof blocks with the block pool carried between execution phases: `PoolWindow_v1`
opens the pool row under the previous execution phase's output root, registered before its coins, and commits the
row its stream leaves as the next. Both are proved every execution phase, and a chain restarts from the empty pool's root after a
failed one. A request open at the execution phase's edge is cut, having emitted its ``a_O``, and continues in the next one
as an ordinary arrival; each keyed block's page is copied into the execution phase's carried port from its source, its
write or the previous carried port (`PageCopy_v1`). The charge is ℓ_w plus `PoolWindow_v1`'s checked rows (the left row,
the blocks' sources and the stream's witness) at their widths.

**The charge** is `charge_chain.ChargeLink_v1{D}` (`build_link`; [charge-chain.md](charge-chain.md)):
- Its inputs:
  - ``pub``'s book: the registered book's public tables (`charge_chain.book_fields(D)`);
  - ``cin`` and ``cout``, the execution phase's carry-in and carry-out: its reads of `replay-charge` rows 0 and 1;
  - ``adv``'s D decision bits, with en_i = i < ``count``.
- Its ok is the verdict's ``code`` bit, and its bits are held to the declared charge c (``charge``).
- An execution phase charges the decisions it commits, in commit order (advice v1). `ReplayLoad` appends them after the decisions
  the last execution phase left unread, and `ReplayAdvance` reads the execution-phase row's in turn. So the scan reads them in commit
  order and only consumes them, and `ReplayClose` refuses a session that leaves one unread.
- A decision an execution phase commits but its steps don't read is charged in that execution phase and read in a later one.
- A forced admission (an arrival into a quiet engine, taken unasked) is no decision, so it takes no slot. ``run``, the
  charge's context, counts decisions only, as `code_v1`'s does, and the admission doesn't reset it as it resets the
  scan's ``pc``. Each of 8eb1's 339 execution phases opens with one. Over them the chain is `code_v1`'s charge of the decisions
  alone (its other counts 0) on every execution phase, 5,500 bits; charged as a 1, the forced admissions would change 330
  execution phases' charges (`tests/benchmarks/network_e2e/encodings/test_served_window.py`).
- The slice has no coder state, codeword or context counter: ChargeLink_v1 reads its contexts and estimates from ``cin``
  and the committed decisions. Its counts other than the decisions are coded as 0 (`arith_encode.ZERO_COUNTS`, the
  still-sent fields' stand-in).
- **What a proof of the slice covers** (the phase-18 attack): the compile slice's charge, the execution phase's
  decisions. The chain is `code_v1`'s whole charge only where an execution phase's hits and order are 0 as well.
  - Hits are 0 on 8eb1, which serves without prefix caching: its trace header says so (``enable_prefix_caching``
    false, `test_served_window.py`).
  - Order is 0 only under release-order serving, and 8eb1 wasn't served in release order. The order's charge is the
    I/O tie's (`io_tie`, a draft), outside the slice's proof.

`CompileSlice_v7` (`verity_vllm.compiler.compile_slice_v7`) is v1 with `window_charge.ChargeLink_v2{D, 0, 0, K}` in
place of this charge, so it charges advice v1's length terms too ([charge-chain.md](charge-chain.md), "The slice").

**What binds the charge's inputs** (the phase-14 attack: `ChargeCount_v1` on its own is faithful, but its enables, its bits
and its carry are witness until the slice binds them):
- *`en` is the scan's own asks.* It is ``count``'s, and the scan can't read past ``count``. `ReplayLoad` refuses
  committed bits past it, and `ReplayAdvance` refuses a step whose reads pass what is left. Advice that drops a
  decision the scan asks is refused:
  - by ``load`` when the execution-phase row keeps the decision;
  - by ``steps`` (and ``rows``) when the execution-phase row drops it too, since the scan's ask then reads past the count.
  On w212 both cases charge less than the honest 27 bits (24).
- *`bit` is the committed decisions:* ``adv``'s row, which the verifier holds to the advice registration's leaf.
- *The carry is the session's:* ``cout`` must be the charge's carry' from ``cin`` (``code``). The verifier holds ``cin``
  and ``cout`` by their b‖c to the session's carry chain (`charge_chain.check_chain`), beside the advice, served-row and
  certified-record bindings. The proof block accepts a restarted carry-in (the priors, or counts fitted to the execution phase),
  and the chain refuses it: ``CH-first`` on a session's first execution phase, ``CH-carry`` after it.
- The slice-level cases are `tests/adapters/vllm/tests/compiler/test_network_attester_compile_slice_phase14.py`.

**Every advised execution phase is proved, never sampled.** The chain binds only proved proof blocks: an unproved one's
carry-out is whatever the prover likes. And `en` is pinned only by the slice's own scan. So the compile slice (the load
proof block, where the replay is split into proof blocks) is sampled `All`, as N2's proof blocks are (PROTOCOL.md, A3).

**A served execution phase at its engine's class: w212.** An execution phase's request and step counts are private, and the
class is protocol structure. So a session proves every execution phase at its engine's class: (R, W, A) = (8, 1, 9), Q 4,
D 32, P 54, K 4, no prefix caching, and 8eb1's AsyncScheduler (AS = 1). The scheduler is engine configuration, so it
is the class's and never an execution phase's: an AS chosen per execution phase would show whether its schedule matched sync
order. w212 (requests 308–310, 21 proof blocks, 14 decisions) is padded into it
(`tests/verity/protocols/resource_accounting/network/verifier/fixtures/compile_slice_w212_class_async.json`: an empty arrival slot, 18
decisions past the count with en 0, and 33 idle steps). The seal's rows are in ``srv``
(`verity_vllm.compiler.served_rows.step_rows`), each held to its request's slot through ``map``.
- It proves w212's charge of its decisions. Its order's charge is the I/O tie's, outside this proof (What a proof of
  the slice covers, above).
- `replay_vectors --slice` replays an execution phase at the trace's own scheduler unless asked for the sync class (`--AS 0`).
- The proof block is 1,021,302 ANDs, 123,120 of them the map's ([served-shape.md](served-shape.md)). Of them,
  ChargeLink_v1{32} is 97,159, and 5,509 are its copies of the words it carries from decision to decision
  ([charge-chain.md](charge-chain.md)).
- Without prefix caching the slice makes no CacheStep_v1 call and ``cache`` is 1. So a made-up cache table can't be
  expressed at 8eb1's class; the demonstration class (prefix caching, 59,341 ANDs) is where it is refused, by ``cache``
  alone.
- **The sync class** is the same sizes under a sync scheduler (AS = 0): w212's 20 steps and 34 idle ones
  (`compile_slice_w212_class.json`), 857,866 ANDs, and at w212's own statics, which would show its counts, 117,164.
  The circuit is its reference on the honest execution phase and every negative at both
  (`tests/adapters/vllm/tests/compiler/test_network_attester_compile_slice_w212.py`), and the padding changes no
  verdict and no charge. At AS = 1 the class's circuit is its reference on w1
  (`test_network_attester_compile_slice_w1.py`), and on w212.
- The honest execution phase charges 27 bits from the session's carry into it, ((58, 50), (58, 0), (392, 48)) at run 0. It
  would charge 28 from the priors and 25 from counts fitted to it.
- That carry is the chain over execution phases 0 … 211, each one's decisions as its own async replay asks them
  (`replay.decisions` on `replay_vectors.served_window`), 3,448 bits.
  - Over all 339 execution phases the chain charges 5,500 bits, against 5,593 if every one restarted at the priors.
  - The sync replay (`slice_inputs` at AS = 0) asks the same decisions on 309 execution phases, asks others on 5 (30, 59, 118,
    147 and 235) and refuses 25. So the chain is the class's, each execution phase's decisions as the AsyncScheduler asks them.
- The phase-10 attack's passes (a pass at every ask, where the served execution phase admitted requests 309 and 310 at the seventh
  and fourteenth: twenty-one at the class, at the asks of its 21 proof blocks, and twenty at the sync class) charge 23
  bits.
  - With the served rows in ``srv``, they are refused by ``rows``.
  - With their own rows there, the proof block accepts them, and the native ``srv`` b‖c check refuses them.
