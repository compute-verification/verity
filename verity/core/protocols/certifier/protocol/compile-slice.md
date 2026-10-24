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
- A forced admission (an arrival into a quiet engine, taken unasked) is no decision, so it takes no slot. ``run``, the
  charge's context, counts decisions only, as `code_v1`'s does, and the admission doesn't reset it as it resets the
  scan's ``pc``. Each of 8eb1's 339 windows opens with one. Over them the chain is `code_v1`'s charge on every window,
  5,500 bits; charged as a 1, the forced admissions would change 330 windows' charges
  (`benchmarks/network_e2e/encodings/tests/test_served_window.py`).
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
- The unit is 857,866 ANDs. Of them, ChargeLink_v1{32} is 97,159, and 5,509 are its copies of the words it carries
  from decision to decision (#3887; [charge-chain.md](charge-chain.md)). At w212's own statics, which would show its
  counts, it is 117,164.
- Without prefix caching the slice makes no CacheStep_v1 call and ``cache`` is 1. So a made-up cache table can't be
  expressed at 8eb1's class; the demonstration class (prefix caching, 59,341 ANDs) is where it is refused, by ``cache``
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
is scanned for every witness piece: the ``wit`` and ``srv`` rows, every row's salt, and the ``cin`` and ``cout`` rows
whole. A piece found that isn't excluded is leaked, and fails the run.
- The scan is `zk_scan`'s, every ZK driver's (`benchmarks/network_traces/zk_scan.py`). A piece found is excluded if and
  only if its value occurs, raw or as hex at an even offset into a run of hex digits, in the bytes the statement fixes:
  - the staged circuit and ``pub``'s rows (`pub-*.bin`), whole;
  - each round message's framing (`rounds[*].msg`, decoded): its records' headers, the domain's and labels' bytes and
    the padding (C-Flock's §6.1);
  - each proof stream's container fields (`circuit.rep*.bin`): the lengths and parameters its layout reads, and each
    ring switch's grinding nonce, which the verifier requires to be 0 (C-Flock's §8.3).
- The rest of the streams and messages is the prover's: its elements, hashes, observed bytes and every other nonce.
  Taken whole, the streams would exclude every piece the scan could find there. No piece is excluded for its byte count
  (REDTEAM's rule, note:network-accounting/20261024T0558Z-finding-redteam-v3, phase 18). A carry row (12 bytes) is one
  piece, whole.

**The proof of record is run r20261024-211633-861e** (#3886 at 6e29d34e7c, the w212 and demonstration slices).
- At w212: 10 units of 857,866 ANDs (k_log 23). m0's verifier and Lean `verify --zk` accept, and the audit is complete
  with no wrong unit (δ 2⁻²⁰).
  - The honest unit's word is 0x3f, and it is accepted.
  - Seven negatives are refused by their words: flip 0x2e, charge 0x37, arrival 0x2f, cleared 0x3e, carry 0x3b,
    cleared-read 0x2d and sixpass 0x2f.
  - restart is refused by ``CH-carry``, and sixpass-own by the native ``srv`` check.
- At the demonstration class: 12 units of 59,341 ANDs, accepted and complete, every verdict as its reference says. The
  made-up table is refused by ``cache`` alone (0x1f), and restart by ``CH-first``.
- The verifier received 347,887,598 bytes at w212 and 103,356,161 at the demonstration class. The run's own scan left
  out the pieces with at most one nonzero byte, and found none of the other 837 and 654 witness pieces, and no ``cin``
  or ``cout`` row.
- Its scan rerun under `zk_scan`, on the rows rebuilt from its verdict words, leaks nothing
  (art:645dbbe3f4a8fa5ec24f174ef1675038aadedd232e301f0d3abf976c9f8c8e94; the salts aren't kept).
  - At w212, it finds ten of the rows' 380 pieces: five of the six below, on the framing and the container's fields,
    and five the circuit holds. The sixth (``20`` in its eleventh byte) isn't found, since its only matches are hex at
    odd offsets, straddling nibbles.
  - At the demonstration class, it finds one all-zero piece and the restart's zero ``cin`` row. The circuit and the
    container's fields hold both.
- REDTEAM's check of the left-out pieces is
  art:c8f215f4e454914b21e339fff69d533f04c11e1958a95e6e09d69b4481384973 (the check and its logs).
  - Of w212's 44 sparse pieces, 38 occur nowhere in what the verifier received. The other six occur only in the proof
    streams and round messages.
  - The demonstration class's restart starts from the zero carry. Its row occurs in the same files' zero padding.
- Each of the six sits on framing:
  - the count and padding fields of round 0's protocol descriptor and of `flock-pcs-open-batch-v0`'s message;
  - the 5 and 8 fields of the 176-byte rounds;
  - in the proof stream, the length 128 of `initial_cap` and of a ring switch's `s_hat_v`, each behind a grinding nonce
    that the verifier's reader requires to be 0. The `s_hat_v` window's one other byte is the last byte of the
    previous field element, 0 there (0x92 in the demonstration class's proof).
- Each message hit is the same 16 bytes at the paired round of the demonstration class's proof, whose circuit and
  witness differ.
- Run r20261024-202836-67d7, at #3886's head before #3887's copies (852,357 ANDs), has the same verdicts. Its six
  flagged pieces are the same framing (REDTEAM's differential, run r20261024-211716-ae3b). `zk_scan` excludes five of
  them and doesn't find the sixth, as at 861e.

**The slice's ZK runs with its coder**, before the charge was `ChargeLink_v1`:
- At the demonstration class (49,113 ANDs a unit), the honest unit and ten negatives, with the made-up table refused
  by ``cache`` alone (run r20261024-162328-03c7).
- At corpus-smallkv's first seven steps (7,562,498 ANDs a unit, m = 29), the honest unit and twelve negatives (run
  r20261024-175616-105c). The made-up table was refused by ``cache`` alone, and the six passes with their own served
  row by the native ``srv`` check.
- Their scans rerun under `zk_scan` (art:645dbbe3f4a8fa5ec24f174ef1675038aadedd232e301f0d3abf976c9f8c8e94):
  - At the demonstration class, one all-zero piece is found. The circuit and the container's fields hold it, so
    nothing is leaked.
  - At the corpus class, two of the four pieces found are leaked, so that run fails the scan.
  - One leaked piece is a cache table's token, ``05`` and fifteen zeros. It sits where each stream holds a Ligerito
    nonce list's length (5) and its first nonce. The query and consistency nonces are messages the verifier never
    checks (C-Flock's §8.3), and Verity's prover sets them to 0. The same piece leaks in the corpus class's other runs,
    r20261024-145236-cf49 and r20261024-205837-adc8.
  - The other is a walk entry, fifteen zeros and ``c0``: the tail of an all-zero element and the next element's first
    byte, a one-in-256 match.
