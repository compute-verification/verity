Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The compile slice's charge, and its proof at a served engine's class

`CompileSlice_v1` (`compile_slice`) is one window of the compile unit in one Boolean Definition. Its circuit, rows,
verdict flags, reference and stand-ins are `compile_slice`'s docstring. This file covers how the slice charges a
window, what binds the charge's inputs, and the slice's proof of record on a served window.

**The charge** is `charge_chain.ChargeLink_v1{D}` (`build_link`; [charge-chain.md](charge-chain.md)):
- Its inputs:
  - ``pub``'s book: the registered book's public tables (`charge_chain.book_fields(D)`);
  - ``cin`` and ``cout``, the window's carry-in and carry-out: its reads of `replay-charge` rows 0 and 1;
  - ``adv``'s D decision bits, with en_i = i < ``count``.
- Its ok is the verdict's ``code`` bit, and its bits are held to the declared charge c (``charge``).
- A window charges the decisions it commits, in commit order (advice v1). `ReplayLoad` appends them after the decisions
  the last window left unread, and `ReplayAdvance` reads the window row's in turn. So the scan reads them in commit
  order and only consumes them, and `ReplayClose` refuses a session that leaves one unread.
- A decision a window commits but its steps don't read is charged in that window and read in a later one.
- The slice has no coder state, codeword or context counter: ChargeLink_v1 reads its contexts and estimates from ``cin``
  and the committed decisions. Its counts other than the decisions are coded as 0 (`arith_encode.ZERO_COUNTS`, the
  still-sent fields' stand-in).

**What binds the charge's inputs** (REDTEAM phase 14: `ChargeCount_v1` on its own is faithful, but its enables, its bits
and its carry are witness until the slice binds them):
- *`en` is the scan's own asks.* It is ``count``'s, and the scan can't read past ``count``. `ReplayLoad` refuses
  committed bits past it, and `ReplayAdvance` refuses a step whose reads pass what is left. Advice that drops a
  decision the scan asks is refused:
  - by ``load`` when the window row keeps the decision;
  - by ``steps`` (and ``rows``) when the window row drops it too, since the scan's ask then reads past the count.
  On w212 both cases charge less than the honest 27 bits (24).
- *`bit` is the committed decisions:* ``adv``'s row, which the verifier holds to the advice registration's leaf.
- *The carry is the session's:* ``cout`` must be the charge's carry' from ``cin`` (``code``). The verifier holds ``cin``
  and ``cout`` by their b‖c to the session's carry chain (`charge_chain.check_chain`), beside the advice, served-row and
  certified-record bindings. The unit accepts a restarted carry-in (the priors, or counts fitted to the window), and
  the chain refuses it: ``CH-first`` on a session's first window, ``CH-carry`` after it.
- The slice-level cases are `verity/core/protocols/certifier/tests/test_network_certifier_compile_slice_phase14.py`.

**Every advised window is proved, never sampled.** The chain binds only proved units: an unproved window's carry-out is
whatever the prover likes. And `en` is pinned only by the slice's own scan. So the compile slice (the load unit, where
the replay is split into units) is drawn `All`, as N2's units are (PROTOCOL.md, A3).

**The proof of record: w212 at vLLM 8eb1's engine class.** A window's request and step counts are private, and the
class is protocol structure. So a session proves every window at its engine's class: (R, W, A) = (8, 1, 9), Q 4,
D 32, P 54, K 4, no prefix caching. w212 (requests 308–310, 20 steps, 14 decisions) is padded into it
(`verity/core/protocols/certifier/tests/fixtures/compile_slice_w212_class.json`: an empty arrival slot, 18 decisions
past the count with en 0, and 34 idle steps), with the seal's rows in ``srv`` (`served_rows.step_rows`).
- The unit is 852,357 ANDs, about 97,000 of them ChargeLink_v1{32}. At w212's own statics, which would show its
  counts, it is 114,805.
- Without prefix caching the slice makes no CacheStep_v1 call and ``cache`` is 1. So a made-up cache table can't be
  expressed at 8eb1's class; the demonstration class (prefix caching, 58,521 ANDs) is where it is refused, by ``cache``
  alone.
- The circuit is its reference on the honest window and every negative, at both statics
  (`verity/core/protocols/certifier/tests/test_network_certifier_compile_slice_w212.py`). The padding changes no
  verdict and no charge.
- The honest window charges 27 bits from the session's carry into it, ((58, 50), (58, 0), (392, 48)) at run 0. It
  would charge 28 from the priors and 25 from counts fitted to it.
- That carry is the chain over windows 0 … 211, each window's decisions as its own async replay asks them
  (`replay.decisions` on `replay_vectors.served_window`), 3,448 bits.
  - Over all 339 windows the chain charges 5,500 bits, against 5,593 if every window restarted at the priors.
  - The sync replay (`slice_inputs`) asks the same decisions on 309 windows, asks others on 5 (30, 59, 118, 147 and
    235) and refuses 25. So the chain is what an async slice commits.
  - The ZK run's earlier windows are committed by its driver and not proved in that run (a stand-in).
- REDTEAM phase 10's twenty passes (a pass at every ask, where the served window admitted requests 309 and 310 at the
  seventh and fourteenth) charge 23 bits.
  - With the served rows in ``srv``, they are refused by ``rows``.
  - With their own rows there, the unit accepts them, and the native ``srv`` b‖c check refuses them.

**In zero knowledge** (`benchmarks/network_traces/compile_slice_zk.py --slice w212`, and `--slice demo`), each unit is
proved by C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`. The verifier opens only ``pub``
and the verdict, and runs the bindings, the carry chain's among them, on the units' b‖c. What the verifier received
holds no witness piece: the ``wit`` and ``srv`` rows, every row's salt, and the ``cin`` and ``cout`` rows whole.
