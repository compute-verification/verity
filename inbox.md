[2026-11-14T18:18Z] af33e234060f
WALKER, Sat 14 Nov, 10:18 AM PST.

**Both merges are pushed, and neither is marked: A (#9558) 4dd7d026ba487b057fc683d2db25f5087dfa1aa4 merges #9877's head 9adca32c755aec052cfcf939dc8e77983960dee2 into c0082a114ab5; C (#9730) 4014519f62535cf0c8ea267b6e5c377875a0b004 merges 4dd7d026ba48 into 46b9519a24c2. Each has one hand-resolved hunk pair, printed in full below; the rest merged without conflict.**

- A's hand resolution: in `benchmarks/circuit_properties/README.md`, A's paragraph on the point check and zero rows is kept whole, and #9877's two lines on `work.bound` (it hands `work.products` to `min_cut.slowdown_of`) replace the old two. READ: merged `work.bound` calls `mc.slowdown_of(..., work=products(work))`, and `products` prices each group's `macs`, from which the zero share is already taken.
- C's hand resolution: in `benchmarks/circuit_properties/tests/test_work.py`, C's guarded-attention block (`GUARDED`, `_guarded_attention` and its two tests) is kept, and `test_clique` is dropped, as #9877 dropped it (`work.clique` moved to `min_cut`; its other use became `MC.clique`). RAN (ast): 60 test functions, #9558's 58 and C's two, none twice and none dropped; no top-level name is defined twice. C's README takes A's resolved text: C doesn't touch the README.
- C's body cites `work.py` by line. #9877 moves every cited line up by one, each unchanged (RAN, difflib between 46b9519a24c2 and 4014519f6253), and the body's numbers are now the new head's.
- RAN on the VM, `suites.py circuit_properties benchmarks/training_slowdown --changed 3d62f5507b01 --quick --fresh`. With `main` merged, the change from 3d62f5507b01 reaches 23 training_slowdown test modules, run whole:
  - at A 4dd7d026ba48: `2 suites: 2 passed` (circuit-properties 144, training_slowdown 743);
  - at C 4014519f6253: `2 suites: 2 passed` (154, 746);
  - `tests/test_one_slowdown.py` (#9877's entry function test): 3 passed at each head.
  - Ruff on C's `test_work.py`: clean.
- Not launched: no run. Not marked: either head.

A, `git show --remerge-diff 4dd7d026ba48`, 2 hunk(s):
```
diff --git a/benchmarks/circuit_properties/README.md b/benchmarks/circuit_properties/README.md
remerge CONFLICT (content): Merge conflict in benchmarks/circuit_properties/README.md
index efe0cfd5689..6a51c0242ad 100644
--- a/benchmarks/circuit_properties/README.md
+++ b/benchmarks/circuit_properties/README.md
@@ -65,7 +65,6 @@ by a shared weight would then read one pair of values twice. A carried word's ro
 shift or a rounding to a narrower float (`work.ROUNDINGS`) whatever else it reads, as a weight's update does, and by a
 multiply or a divide only where every other argument is one no input reaches, one word every member of every enclosing
 batch reads, or one whose own rows are shown distinct (a weight times a weight); any other carried rows may repeat, and
-<<<<<<< ours
 a multiply by a stated 0 reaches nothing. Each charged operand whose rows these rules show distinct or specific is
 tested at a point (`points.check`): the Program evaluated once, θ's words generic (four binades about 2^-3, every sign)
 and every input's one positive word repeated, as a submitter may choose, and an operand specific and not distinct at a
@@ -91,35 +90,8 @@ weights against a stated constant, rows equal only where an input's word is nega
 a submitter found by trying circuits against the point's seed, which is public (SHA-512 of the Program's digest); and
 two nodes that compute rows from one array's words into two arrays give each
 distinct rows.
-`work.bound(work, c)` is the most blocks over its
-sub-works, through the calculator's `min_cut` at each sub-work's least β, none from `HEAD_K` on. The module docstring
-||||||| base
-a multiply by a stated 0 reaches nothing. One more premise holds the reading below the work, its parts each held as a
-fact by a test: θ's rows differ at every width a rounding reads them, and a word no input reaches is nonzero; an update
-that can't change a weight or cancels it for a reason no constant shows (W plus z plus -1 times z) still makes θ′ and
-keeps its rows distinct; a weight combined with its own words still counts as a weight with distinct rows, though the
-combination may make its rows equal whatever the weight: cancelled against itself (M plus -1 times M; a multiplier v
-plus -1 times v, v registered or carried), or times its own rows in another order (M times swap(M), whose two rows are
-equal for every M), so the looked-up rows plus such an array are charged as weights though they repeat with the
-tokens; so does one word every member of every batch reads as a multiplier (a learning rate, a
-decay, an input's or computed from inputs, taken as nonzero); and two nodes that compute rows from one array's words
-into two arrays give each distinct rows.
-`work.bound(work, c)` is the most blocks over its
-sub-works, through the calculator's `min_cut` at each sub-work's least β, none from `HEAD_K` on. The module docstring
-=======
-a multiply by a stated 0 reaches nothing. One more premise holds the reading below the work, its parts each held as a
-fact by a test: θ's rows differ at every width a rounding reads them, and a word no input reaches is nonzero; an update
-that can't change a weight or cancels it for a reason no constant shows (W plus z plus -1 times z) still makes θ′ and
-keeps its rows distinct; a weight combined with its own words still counts as a weight with distinct rows, though the
-combination may make its rows equal whatever the weight: cancelled against itself (M plus -1 times M; a multiplier v
-plus -1 times v, v registered or carried), or times its own rows in another order (M times swap(M), whose two rows are
-equal for every M), so the looked-up rows plus such an array are charged as weights though they repeat with the
-tokens; so does one word every member of every batch reads as a multiplier (a learning rate, a
-decay, an input's or computed from inputs, taken as nonzero); and two nodes that compute rows from one array's words
-into two arrays give each distinct rows.
 `work.bound(work, c)` hands the charged products (`work.products`: MACs at `keep`, outputs at most 16 bits) to the
 calculator's `min_cut.slowdown_of`, which chooses the sub-work, none from `HEAD_K` on. The module docstring
->>>>>>> theirs
 has the rules and which way each moves the bound; the training benchmark's `tests/test_program_reading.py` holds a
 training loop's reading beside its trace's, never above it.
 
```

C, `git show --remerge-diff 4014519f6253`, 2 hunk(s):
```
diff --git a/benchmarks/circuit_properties/tests/test_work.py b/benchmarks/circuit_properties/tests/test_work.py
remerge CONFLICT (content): Merge conflict in benchmarks/circuit_properties/tests/test_work.py
index 4033ffca4a8..da8b254fae4 100644
--- a/benchmarks/circuit_properties/tests/test_work.py
+++ b/benchmarks/circuit_properties/tests/test_work.py
@@ -1434,7 +1434,6 @@ def test_a_chain_deeper_than_the_recursion_limit_is_read_as_a_short_one(depth):
         [(g.product, g.weight, g.macs, g.reason) for g in plain.groups]
 
 
-<<<<<<< 46b9519a24c (Merge branch 'cursor/trn145-random-point-refusal-914d' into cursor/trn145-mask-structure-eb6e Co-authored-by: danielreuter <danielreuter@users.noreply.github.com>)
 #: the guarded attention's variants, each with the nodes it reads from runs and those it reads word by word: its form,
 #: and each off it by one word or reaching a branch a causal mask never does. A condition that is not a constant stops
 #: the reading at the select, before the softmax's rows, and so does a compare with a word of a node whose words
@@ -1530,15 +1529,6 @@ def test_the_guarded_softmax_charges_what_the_unguarded_one_does():
     assert (w.N, w.keep, w.left_out) == (plain.N, plain.keep, plain.left_out)
 
 
-def test_clique():
-    tri = [frozenset(e) for e in ((1, 2), (2, 3), (1, 3), (3, 4))]
-    assert WK.clique(tri) == 3
-    assert WK.clique(tri + [frozenset(e) for e in ((1, 4), (2, 4))]) == 4
-    assert WK.clique([frozenset((1, 2))]) == 2 and WK.clique([]) == 0
-
-
-=======
->>>>>>> 4dd7d026ba4 (Merge commit '9adca32c755aec052cfcf939dc8e77983960dee2' into HEAD)
 #: the suite's Programs built by name, each held to its reading in `READINGS`
 CASES = {
     "train": loop, "carried": carried_loop, "lookup": lambda: _lookup_loop(False),
```

