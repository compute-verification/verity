/-!
The tensor fold's write floor with the lever and a fresh R a step (hash_lever.py part F, K 2,048, rank 2): any program
that outputs the sealed state writes at least 1.773 registers a word, at least twice stock (0.8673).

Quantities are a word's, in thousandths: `w` the program's register writes, `wp` those that form products with R or Q
and `wb` those that carry operand bits, `p` the products formed and `bits` the operand bits written.
- W1: both kinds are register writes.
- H32: a register write forms at most 32 products with s8 R or Q entries.
- DistinctWrites: a register write carries at most 32 operand bits.
- ByteEntry: every row-plane's product (one row, one step, one byte position) is formed by one route, a correction Δ
  added to products the program already holds (scratch, a partner row of the same step, COUNTER's decomposition, the
  walk split), and the program forms at least that route's products and bits: 8 a nonzero s8 digit of Δ, and the
  digit's width. The state adds Z's 128 bits and Z·Q's 128 products a row a step. `f0`..`f3` and `fz` are the cheapest
  route's products plus bits, summed over row-planes and over 32: checked, per byte, by hash_lever.py.
Core Lean only: `lean TensorFloor.lean`.
-/

theorem tensor_floor (w wp wb p bits f0 f1 f2 f3 fz : Nat)
    (h32 : p ≤ 32 * wp)
    (distinct : bits ≤ 32 * wb)
    (writes : wp + wb ≤ w)
    (entry : 32 * (f0 + f1 + f2 + f3 + fz) ≤ p + bits)
    (byte0 : 489 ≤ f0) (byte1 : 489 ≤ f1) (byte2 : 489 ≤ f2) (byte3 : 272 ≤ f3) (state : 31 ≤ fz) :
    1770 ≤ w ∧ 2 * 868 ≤ w := by
  omega

/-- The same count at the finer grain the run prints (products plus bits, not yet over 32). -/
theorem tensor_floor_fine (w wp wb p bits f0 f1 f2 f3 fz : Nat)
    (h32 : p ≤ 32 * wp)
    (distinct : bits ≤ 32 * wb)
    (writes : wp + wb ≤ w)
    (entry : f0 + f1 + f2 + f3 + fz ≤ p + bits)
    (byte0 : 15663 ≤ f0) (byte1 : 15665 ≤ f1) (byte2 : 15679 ≤ f2) (byte3 : 8728 ≤ f3) (state : 1000 ≤ fz) :
    1773 ≤ w ∧ 2 * 868 ≤ w := by
  omega

#print axioms tensor_floor
#print axioms tensor_floor_fine
