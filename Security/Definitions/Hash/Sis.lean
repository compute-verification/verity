import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Fintype.Pi
import Definitions.Hash.Tree

/-!
# An SIS leaf under a SHA-512 tree (trusted)

The leaf's inner digest is linear: a row read as `m` short integer digits `w` (int8 values, or nibbles, with the
leftover-hash randomness as further digits when the leaf hides natively) is mapped to `u = A·w mod q` by a public
matrix `A ∈ Z_q^{n×m}` (on int8 tensor cores: `A` in base-256 digits, an int32 accumulation each), and the leaf input is
the format's tag, which names the parameter set, followed by `u`'s encoding (`leafIn`). The tree above it is
`Definitions/Hash/Tree.lean`'s, over `H` (SHA-512 when deployed).

The verifier takes `A` and the parameter set from its own statement, never from an opening, and refuses an opening
whose digits are out of range or whose length is not `m` (the length is in the type: `Fin P.m → ℤ`); a verifier that
did not check the length let `ajtai-leaf-3`'s collisions through.

**SIS.** A solution for `A` is a nonzero integer vector `z` with every entry at most `β = hi − lo` in absolute value
and `A·z = 0 mod q` (`Solution`). Two accepted openings at one position whose digits differ, with no collision of `H`
among what the two paths hashed, have the same `u`, so their difference is a solution (`Guarantees.SisTreeBinding`).
For a fixed public `A` that is all a statement can say: some finder hard-codes a solution of any one matrix. Hardness is
stated over a uniform `A` (`Assumptions.SISHard`); a matrix derived from a public label takes it through a
common-reference-string step (`hash-derived-matrix`), and one derived from the epoch's live coin is the uniform draw of
the window game (`Guarantees.SisEpochBinding`).

**Cost.** A finder's and a prover's `cost` are their work in one declared unit (as `Claims.Finder.cost` is SHA-512
calls); `reduce` adds the `m` subtractions that turn a prover's double opening into a candidate solution.
-/

namespace Hash.Sis

/-- An SIS parameter set: `n` rows, `m` digits a leaf, modulus `q`, digits in `[lo, hi]`, and the security in bits
the parameter set is claimed at (`bits`, from the lattice estimator). -/
structure Params where
  n : ℕ
  m : ℕ
  q : ℕ
  one_lt_q : 1 < q
  lo : ℤ
  hi : ℤ
  bits : ℕ

instance instNeZeroNatQ (P : Params) : NeZero P.q := ⟨by have := P.one_lt_q; omega⟩

/-- A public matrix: `n × m` over `Z_q`. -/
abbrev Mat (P : Params) : Type := Fin P.n → Fin P.m → ZMod P.q

/-- A leaf's digits. -/
abbrev Digits (P : Params) : Type := Fin P.m → ℤ

/-- The linear digest `u = A·w mod q`. -/
def image (P : Params) (A : Mat P) (w : Digits P) : Fin P.n → ZMod P.q := fun i => ∑ j, A i j * (w j : ZMod P.q)

/-- **An SIS solution for `A`**: nonzero, every entry within `β = hi − lo`, and `A·z = 0 mod q`. -/
def Solution (P : Params) (A : Mat P) (z : Digits P) : Prop :=
  z ≠ 0 ∧ (∀ j, |z j| ≤ P.hi - P.lo) ∧ image P A z = 0

/-- Bytes per coordinate of `u`: enough for `q − 1`. -/
def byteWidth (q : ℕ) : ℕ := (Nat.size (q - 1) + 7) / 8

/-- `v` in `w` bytes, big-endian. -/
def encNat (w v : ℕ) : Msg := (List.range w).reverse.map fun j => ((v / 256 ^ j) % 256).toUInt8

/-- `u`'s encoding: each coordinate's residue in `byteWidth q` bytes, in order. -/
def encImage (P : Params) (u : Fin P.n → ZMod P.q) : Msg :=
  (List.finRange P.n).flatMap fun i => encNat (byteWidth P.q) (u i).val

/-- The tree's format: the leaf tag (it names the parameter set), the node tag, and the depth. -/
structure Format where
  leafTag : Msg
  nodeTag : Msg
  depth : ℕ

/-- An opening: the leaf's digits and the sibling digests from the leaf level up. -/
structure Opening (P : Params) where
  w : Digits P
  sibs : List Digest

/-- The leaf input of digits `w`: the tag, then `A·w`'s encoding. -/
def leafIn (P : Params) (F : Format) (A : Mat P) (w : Digits P) : Msg := F.leafTag ++ encImage P (image P A w)

/-- The verifier accepts an opening at position `i` against root `r`: its digits are in range and its path recomputes
`r`. -/
def accepts (P : Params) (F : Format) (H : Msg → Digest) (A : Mat P) (r : Digest) (i : ℕ) (o : Opening P) : Prop :=
  (∀ j, P.lo ≤ o.w j ∧ o.w j ≤ P.hi) ∧ Path.accepts H F.nodeTag F.depth r i (leafIn P F A o.w) o.sibs

/-- Every input an opening's check hashed. -/
def hashed (P : Params) (F : Format) (H : Msg → Digest) (A : Mat P) (i : ℕ) (o : Opening P) : List Msg :=
  Path.hashed H F.nodeTag i (leafIn P F A o.w) o.sibs

/-- What a prover outputs: a root, a position and two openings there. -/
abbrev Claim (P : Params) : Type := Digest × ℕ × Opening P × Opening P

/-- **A binding break**: both openings accepted at one position against one root, with different digits. -/
def Break (P : Params) (F : Format) (H : Msg → Digest) (A : Mat P) (c : Claim P) : Prop :=
  accepts P F H A c.1 c.2.1 c.2.2.1 ∧ accepts P F H A c.1 c.2.1 c.2.2.2 ∧ c.2.2.1.w ≠ c.2.2.2.w

/-- Every input the two openings' checks hashed. -/
def claimHashed (P : Params) (F : Format) (H : Msg → Digest) (A : Mat P) (c : Claim P) : List Msg :=
  hashed P F H A c.2.1 c.2.2.1 ++ hashed P F H A c.2.1 c.2.2.2

/-- A double opening's candidate solution: the difference of the digits. -/
def diff {P : Params} (o₁ o₂ : Opening P) : Digits P := o₁.w - o₂.w

/-- An SIS finder: its coins, its output on a matrix and each coin, and its cost there. -/
structure Finder (P : Params) where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  out : Mat P → S → Option (Digits P)
  cost : Mat P → S → ℕ

attribute [instance] Finder.fin Finder.ne

/-- A prover against a matrix it is given: its coins, its claim on each matrix and coin, and its cost there. Its coins
are drawn independently of the matrix; anything it computed before the matrix was drawn is in its definition. -/
structure Prover (P : Params) where
  Ω : Type
  [fin : Fintype Ω]
  [ne : Nonempty Ω]
  out : Mat P → Ω → Claim P
  cost : Mat P → Ω → ℕ

attribute [instance] Prover.fin Prover.ne

/-- The finder a prover gives: run it, and output its two openings' difference, at its cost plus `m`. -/
def reduce {P : Params} (V : Prover P) : Finder P where
  S := V.Ω
  out A ω := some (diff (V.out A ω).2.2.1 (V.out A ω).2.2.2)
  cost A ω := V.cost A ω + P.m

end Hash.Sis
