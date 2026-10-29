/-!
The tensor fold's write floor with the lever and a fresh R a step (hash_lever.py part F, K 2,048, rank 2): any program
that outputs the sealed state writes at least 1.242 registers a word, at least 1.43 times stock (0.8673).

Quantities are a word's, in thousandths: `w` the program's register writes, `wp` those that form products with R or Q
and `wb` those that carry operand bits, `p` the products formed and `bits` the operand bits written.
- W1: both kinds are register writes.
- H32: a register write forms at most 32 products with s8 R or Q entries.
- DistinctWrites: a register write carries at most 32 operand bits.
- ByteEntry: every row-plane's product (one row, one step, one byte position) is formed by one route, a correction Δ
  added to products the program already holds (scratch, a partner row of the same step, COUNTER's decomposition, the
  walk split), and the program forms at least that route's products and bits: 8 products a nonzero s8 digit of Δ every
  step, and the digit's width where Δ moved since the route's last step. The state adds Z's 128 bits and Z·Q's 128
  products a row a step. `f0`..`f3` and `fz` are the cheapest route's products plus bits over row-planes: checked, per
  byte, by hash_lever.py.
Core Lean only: `lean TensorFloor.lean`.
-/

theorem tensor_floor (w wp wb p bits f0 f1 f2 f3 fz : Nat)
    (h32 : p ≤ 32 * wp)
    (distinct : bits ≤ 32 * wb)
    (writes : wp + wb ≤ w)
    (entry : f0 + f1 + f2 + f3 + fz ≤ p + bits)
    (byte0 : 10196 ≤ f0) (byte1 : 13127 ≤ f1) (byte2 : 8974 ≤ f2) (byte3 : 6461 ≤ f3) (state : 1000 ≤ fz) :
    1242 ≤ w ∧ 143 * 868 ≤ 100 * w := by
  omega

#print axioms tensor_floor
