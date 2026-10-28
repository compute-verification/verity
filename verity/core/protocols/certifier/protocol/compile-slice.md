Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The compile slice's charge, and a served window at its engine's class

`CompileSlice_v1` (`compile_slice`) is one window of the compile unit in one Boolean Definition. Its circuit, rows,
verdict flags, reference and stand-ins are `compile_slice`'s docstring. This file covers how the slice charges a
window, what binds the charge's inputs, and a served window at its engine's class. `CompileSlice_v2`
(`compile_slice_v2`) is the same slice with vLLM's block pool (`CacheStep_v2`) in place of the evictable and returned
counts' stand-in; its charge is this one. `CompileSlice_v3` (`compile_slice_v3`) is v2 computing its schedule rather
than checking it, with no ChargeLink, the served rows and last rows registered reads it holds to the computed ones: its
advice input is exactly ℓ_w gates wide, so the charge is ℓ_w.

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
- A forced admission (an arrival into a quiet engine, taken unasked) is no decision, so it takes no slot. ``run``, the
  charge's context, counts decisions only, as `code_v1`'s does, and the admission doesn't reset it as it resets the
  scan's ``pc``. Each of 8eb1's 339 windows opens with one. Over them the chain is `code_v1`'s charge of the decisions
  alone (its other counts 0) on every window, 5,500 bits; charged as a 1, the forced admissions would change 330
  windows' charges (`benchmarks/network_e2e/encodings/tests/test_served_window.py`).
- The slice has no coder state, codeword or context counter: ChargeLink_v1 reads its contexts and estimates from ``cin``
  and the committed decisions. Its counts other than the decisions are coded as 0 (`arith_encode.ZERO_COUNTS`, the
  still-sent fields' stand-in).
- **What a proof of the slice covers** (the phase-18 attack): the compile slice's charge, the window's
  decisions. The chain is `code_v1`'s whole charge only where a window's hits and order are 0 as well.
  - Hits are 0 on 8eb1, which serves without prefix caching: its trace header says so (``enable_prefix_caching``
    false, `test_served_window.py`).
  - Order is 0 only under release-order serving, and 8eb1 wasn't served in release order. The order's charge is the
    I/O tie's (`io_tie`, a draft), outside the slice's proof.

**What binds the charge's inputs** (the phase-14 attack: `ChargeCount_v1` on its own is faithful, but its enables, its bits
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

**A served window at its engine's class: w212.** A window's request and step counts are private, and the
class is protocol structure. So a session proves every window at its engine's class: (R, W, A) = (8, 1, 9), Q 4,
D 32, P 54, K 4, no prefix caching, and 8eb1's AsyncScheduler (AS = 1). The scheduler is engine configuration, so it
is the class's and never a window's: an AS chosen per window would show whether the window's schedule matched sync
order. w212 (requests 308–310, 21 units, 14 decisions) is padded into it
(`verity/core/protocols/certifier/tests/fixtures/compile_slice_w212_class_async.json`: an empty arrival slot, 18
decisions past the count with en 0, and 33 idle steps). The seal's rows are in ``srv`` (`served_rows.step_rows`), each
held to its request's slot through ``map``.
- It proves w212's charge of its decisions. Its order's charge is the I/O tie's, outside this proof (What a proof of
  the slice covers, above).
- `compile_slice_zk` refuses a w212 fixture whose engine isn't the trace header's. `replay_vectors --slice` replays a
  window at the trace's own scheduler unless asked for the sync class (`--AS 0`).
- The unit is 1,021,302 ANDs, 123,120 of them the map's ([served-shape.md](served-shape.md)). Of them,
  ChargeLink_v1{32} is 97,159, and 5,509 are its copies of the words it carries from decision to decision
  ([charge-chain.md](charge-chain.md)).
- Without prefix caching the slice makes no CacheStep_v1 call and ``cache`` is 1. So a made-up cache table can't be
  expressed at 8eb1's class; the demonstration class (prefix caching, 59,341 ANDs) is where it is refused, by ``cache``
  alone.
- **The sync class** is the same sizes under a sync scheduler (AS = 0): w212's 20 steps and 34 idle ones
  (`compile_slice_w212_class.json`), 857,866 ANDs, and at w212's own statics, which would show its counts, 117,164.
  The circuit is its reference on the honest window and every negative at both
  (`verity/core/protocols/certifier/tests/test_network_certifier_compile_slice_w212.py`), and the padding changes no
  verdict and no charge. At AS = 1 the class's circuit is its reference on w1
  (`test_network_certifier_compile_slice_w1.py`), and on w212.
- The honest window charges 27 bits from the session's carry into it, ((58, 50), (58, 0), (392, 48)) at run 0. It
  would charge 28 from the priors and 25 from counts fitted to it.
- That carry is the chain over windows 0 … 211, each window's decisions as its own async replay asks them
  (`replay.decisions` on `replay_vectors.served_window`), 3,448 bits.
  - Over all 339 windows the chain charges 5,500 bits, against 5,593 if every window restarted at the priors.
  - The sync replay (`slice_inputs` at AS = 0) asks the same decisions on 309 windows, asks others on 5 (30, 59, 118,
    147 and 235) and refuses 25. So the chain is the class's, each window's decisions as the AsyncScheduler asks them.
- The phase-10 attack's passes (a pass at every ask, where the served window admitted requests 309 and 310 at the seventh
  and fourteenth: twenty-one at the class, at the asks of its 21 units, and twenty at the sync class) charge 23 bits.
  - With the served rows in ``srv``, they are refused by ``rows``.
  - With their own rows there, the unit accepts them, and the native ``srv`` b‖c check refuses them.

**In zero knowledge** (`benchmarks/network_traces/compile_slice_zk.py --slice w212`, and `--slice demo`), each unit is
proved by C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`. The verifier opens only ``pub``
and the verdict, and runs the bindings, the carry chain's among them, on the units' b‖c. What the verifier received
holds no witness piece: the ``wit`` and ``srv`` rows, every row's salt, and the ``cin`` and ``cout`` rows whole.
- The scan's rule is the lane's one, field-aligned (`leak_scan.scan`, `benchmarks/network_traces/leak_scan.py`). It
  replaces the phase-18 attack's rule.
  - A find of a witness piece or carry row counts only if it lies inside one field: one read C-Flock's proof format
    makes of a repetition's stream (`circuit.rep*.bin`, `redigest.layout`), or one record of a round message
    (`rounds[*].msg`, as `transcript_check.T` frames them). A match across two fields doesn't count, and neither does
    hex a nibble off.
  - A counted find is excused only if its bytes are in what the statement fixes: ``pub``'s rows (`pub-*.bin`), the
    statement or the circuit's definition, the streams' integer fields (lengths and parameters, not nonces), and the
    round messages' headers, domain and labels.
  - No piece is excused for its byte count or its witness class. A differential against a proof of a different witness
    is evidence for a manual analysis only.
  - The rule misses a leak written at a byte shift across fields.
- The driver scans for every piece but those the staged circuit holds, and for each carry row whole.
  `compile_slice_zk.py rescan` takes a finished run's scan again under the rule, its pieces recomputed; a salt the run
  drew isn't, so a salt it found reads leaked.
  - Each witness row is cut whole (`leak_scan.public_pieces`). A row shorter than the 16-byte piece is found whole, as
    the carry rows are: at G < P, the ``srv`` rows of a few steps (10 bytes at the demo's G = 2).
  - Earlier runs cut with `n2_zk.public_pieces`, which skipped a row's bytes past its last 16-byte boundary and every
    row shorter than 16 bytes.
