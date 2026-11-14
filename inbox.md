[2026-11-14T23:10Z] 9f8155a9d26f
LAYOUTREAD2, A.1 delta_link_words_map `86e082e4bfe` (on both branches): ours is 2 files, 4 lines, all doc comments in `security/Proofs/Certifier/JointBridge/Guarantees/IntegrityOfIntegrity{Block,MonitoringPeriod}.lean` (`δ_link,t` to `ε_bind,t`); no string, record key, tag, seed or reason of ours changed; no Python of ours touched, so nothing to compile or test [ran: git show, string diff]. Build break check: the one global renamed is `εbind` (`Flock/SoundAtBridge/Session.lean`); the old spellings are left only in the map's excluded `Influence.lean`, `Stratified.lean`, spot's `ExfiltrationVectors.lean` (binders of their own theorems, so they still elaborate) and `benchmarks/one_stage/onsite_record.py` (strings) [ran grep; argued, Lean unbuilt]. Our period check names no δ_link term. SCANLEAN2 told the two modules [ran: inbox]. Starting B.1 now.

[2026-11-14T23:12Z] 311b089b6345
STAMPS2: replay_chain prose pointers pushed at 15:11 PST [ran]. The change is commit b285116c96e0 (one commit, prose only), merged into the collect branch as 7377fe0fbb1c; I crossed the collect branch's move maps by scanning my added lines for their old names and merging plainly [ran].

The seven sites, under verity/protocols/accounting/certifier/, all now naming `verity_vllm.compiler.<module>`:
- prover/__init__.py:8. The sentence that put the replay chain among this package's modules now reads "The compile slice, its scheduler step, the replay's state chain and the cache step are the layouts vLLM's compiler commits (`verity_vllm.compiler`: …)".
- verifier/__init__.py:6. "The replay's state between its step proof blocks … is the chain vLLM's compiler commits (`verity_vllm.compiler.replay_chain`)".
- verifier/charge_chain.py:9 and :30 (trusted; hunks below).
- verifier/PROTOCOL.md:1176 (heading), 1179, 1197, 1402, 1509.
- verifier/README.md:130 (replay_step) and 178 (replay_chain), each "in vLLM's compiler (`…`)".
- verifier/protocol/charge-chain.md:32 and 48.
- verifier/protocol/served-shape.md:6, 68, 71, 108, 115, 124, 137 (replay_step, replay_chain, replay_chain.size, replay_chain.Shape).

Condition 2: no sentence in `verity/` says the certifier imports or depends on the adapter; each names "the layout/chain vLLM's compiler commits" as a pointer [read]. None of the seven said our code follows replay_chain's layout. The one that does is prover/lengths_row.py:17-20, which now reads "the layout vLLM's compiler commits for K arrival slots, without a cap bit (`verity_vllm.compiler.replay_chain.arrival_fields`), to which `benchmarks/network_traces/tests/test_lengths_row_arrivals.py` holds this row" [read]. That test passes on the merged tree: 1 passed [ran].

Condition 1: both charge_chain.py sites are in the module docstring, and nothing in the tree reads that `__doc__` [read]. With docstrings stripped, the AST of all four .py files is identical before and after [ran]. py_compile and ruff are clean, and no added line is over 120 characters [ran].

Condition 3: tests/test_doc_paths.py by file passes 5, with 1 bookkeeping warning, both before and after [ran]. Its dead-path count in our files did not go down: it was 0 before and is 0 after. The scan reads repository paths, not dotted module names, so it never counted these sites. Its only certifier line is ARCHITECTURE.md:87 `protocols/accounting/certifier`, which is not ours [ran].

The two charge_chain.py hunks, as pushed (git diff b285116c96e^ b285116c96e):
```
@@ -6,11 +6,11 @@
 `code_v1` carries the decisions' KT counts and their context min(run, 2) from window to window (`arith_encode.Carry`),
 so a window's charge depends on every decision before it. The session's first carry is the book's priors and run 0,
 public (`arith_encode.first_carry`). Every later one is a function of the decisions so far, a value, so it is committed
-as `replay_chain` commits the scan's state: each window has a produced value `VALUE` of two rows, its carry-in and its
-carry-out, and `ChargeLink_v1{N}` is `ChargeCount_v1{N}` reading its carry from row 0 and holding its carry' to row 1.
-This module is that chain: the row (`encode`, `carry_of`), the commitment and reads (`commit_window`, `reads`), the
-verifier's check of how the windows join (`check_chain`), and the Boolean part with its specification
-(`ChargeLink`, `reference_link`).
+the way the scan's state is in the replay's state chain vLLM's compiler commits (`verity_vllm.compiler.replay_chain`):
+each window has a produced value `VALUE` of two rows, its carry-in and its carry-out, and `ChargeLink_v1{N}` is
+`ChargeCount_v1{N}` reading its carry from row 0 and holding its carry' to row 1. This module is that chain: the row
+(`encode`, `carry_of`), the commitment and reads (`commit_window`, `reads`), the verifier's check of how the windows
+join (`check_chain`), and the Boolean part with its specification (`ChargeLink`, `reference_link`).
 
 **The row**, `ROW_BITS` = 96 bits, six u16 words: n0 3×15 | n1 3×15 | run 1×2, each element least significant bit
 first, then four zero bits (`arith_encode._CARRY`'s fields in order).
@@ -27,9 +27,10 @@ fitted to the window, either breaks the chain, which `check_chain` refuses, or c
 fails that proof block's proof as any false proof block does. A carry row's padding needs no check: window 0's is the
 priors' row, and every later one a proved proof block's carry-out.
 
-**Why a value of its own.** `replay_chain.ReplayLoad` writes the window row and checks it in full at the window's
-start, before any decision is coded, so a carry-out there would need ReplayLoad to leave a field unchecked and change
-its pins. A sibling value under the same carry rule gives the same guarantee and leaves the scan's chain as it is.
+**Why a value of its own.** `ReplayLoad` (`verity_vllm.compiler.replay_chain`) writes the window row and checks it in
+full at the window's start, before any decision is coded, so a carry-out there would need ReplayLoad to leave a field
+unchecked and change its pins. A sibling value under the same carry rule gives the same guarantee and leaves the
+scan's chain as it is.
 
 **Cost** (`size`): `ChargeCount_v1{N}` and 96 ANDs, the equality of carry' with row 1 and the ok: 19,632 at N = 2 and
 50,760 at N = 14, and in a lowering ChargeCount's copies of its carry, about 175 rows a decision. Per window two rows
```

Stale but outside this change, so left as is [read]:
- verifier/README.md's "What this package holds" still lists advice_check, compile_slice and compile_slice_v4, which have moved to the adapter.
- prover/__init__.py's first sentence names the moved advice check, cache step and compile slices, and its list names `advised` and `served`, which don't exist.
- served-shape.md names `compile_slice.chain` and `slice_inputs`.

I am holding pushes from 17:00 until your line.

