import Proofs.Flock.Soundness.Audit.FlockLink
import Proofs.Flock.Soundness.Binding.Leaf

/-!
# The layout of registered leaves, and its value binding

The link theorem (`flock_batched_linkSoundE`) takes the binding of wires to registered commit strings as a named
structure, `Partition.ValueBinding`. This file builds one (`Layout.binding`) from:

* **the layout** (`Layout`): the commit strings (`Pos`: the leaves under the serving roots) and the values behind them
  (`Val`: a row), the commit string that holds each wire and each value's bit on it, the leaf digest the registration
  commits to at each commit string (`regLeaf`), and what a drawn unit's table's witness reads at a commit string: a value
  and a salt, read at fixed addresses (`row`, `salt`: the bits at `rowAt`, the bytes at `saltAt`). One fact about the
  layout, `decode_row`: in a satisfying witness, the decoder's bit on each of the unit's wires is the bit of the row the
  witness holds at the wire's commit string;
* **a leaf scheme per commit string**, with its explicit extractor (`ExplicitLeaf`);
* **the registration fact** (`Layout.Registered`): in a satisfying witness, the leaf of what the witness reads at each of
  the unit's commit strings is the registered one. For `hm96-sha512/row/v1` rows it comes from the circuit's row hashing,
  `Assumptions.HmRowComputes` (`Binding.Rows`).

The binding's commitment is the pair (commit string, leaf digest), and its `collide` is the leaf scheme's extractor, so
`collide_spec` is proved: two openings of one commit string with one leaf and different values give two different byte
strings with one SHA-512 digest (`Refine.H512`).
-/

namespace FlockSoundness.Binding

open Game Model Accounting Extract Refine Audit

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- The message type of drawn unit `u`'s table. -/
abbrev Msg (S : Finset (Fin n)) (R : Reg) (u : Fin n) : Type :=
  Message 𝔽 (Audit.Partition.tabOf plan tab S R u).sch.k0 (Audit.Partition.tabOf plan tab S R u).sch.level₀.logCols

/-- The witness a message of drawn unit `u`'s table packs. -/
noncomputable abbrev wit (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : Msg plan tab S R u) :
    Witness (Audit.Partition.tabOf plan tab S R u).S.m :=
  witnessOf execArith (Audit.Partition.tabOf plan tab S R u).S.m (Audit.Partition.tabOf plan tab S R u).sch.k0
    (Audit.Partition.tabOf plan tab S R u).sch.level₀.logCols M

/-- A message of drawn unit `u`'s table packs a witness that satisfies the table's statement. -/
def Sat (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : Msg plan tab S R u) : Prop :=
  (Audit.Partition.tabOf plan tab S R u).S.Satisfies (wit plan tab S R u M)

/-- The byte whose bit `t` is `b t`, bit 0 the least significant. -/
def byteOf (b : Fin 8 → Bool) : UInt8 := UInt8.ofNat (∑ t : Fin 8, if b t then 2 ^ t.val else 0)

/-- The `N` bits a witness holds at the addresses `a`. -/
def bitsAt {m N : ℕ} (z : Witness m) (a : Fin N → Fin (2 ^ m)) : Fin N → Bool := fun i => bit (z (a i))

/-- The `N` bytes a witness holds at the addresses `a`: byte `j`'s bit `t` is the witness's bit at `a j t`. -/
def bytesAt {m N : ℕ} (z : Witness m) (a : Fin N → Fin 8 → Fin (2 ^ m)) : List UInt8 :=
  List.ofFn fun j => byteOf (bitsAt z (a j))

/-- **The layout of the drawn units' committed wires in registered leaves.** The commit strings (`Pos`, the leaves under
the serving roots) and their values (`Val`), the commit string holding each wire (`pos`) and a value's bit on it (`bit`),
and the registered leaf digests (`regLeaf`). Where a drawn unit's table's witness holds each commit string's value and
salt, at fixed addresses: the value's `rowBits p` bits (`rowAt`), which `rowOf` reads as a value, and the salt's 192
bytes, bit by bit (`saltAt`). `decode_row`: in a satisfying witness the decoder's bit on each of the unit's wires is the
bit of the value the witness holds at the wire's commit string. -/
structure Layout (decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab) where
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  [inhPos : Inhabited Pos]
  Val : Type
  [finVal : Fintype Val]
  [decVal : DecidableEq Val]
  [inhVal : Inhabited Val]
  pos : Fin C.N → Pos
  bit : Val → Fin C.N → Bool
  regLeaf : Reg → Pos → Dig
  rowBits : Pos → ℕ
  rowOf : (p : Pos) → (Fin (rowBits p) → Bool) → Val
  rowAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → (p : Pos) → Fin (rowBits p) →
    Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.m)
  saltAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Pos → Fin 192 → Fin 8 →
    Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.m)
  decode_row : ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u,
    decode S R u M g = bit (rowOf (pos g) (bitsAt (wit plan tab S R u M) (rowAt S R u (pos g)))) g

attribute [instance] Layout.finPos Layout.decPos Layout.inhPos Layout.finVal Layout.decVal Layout.inhVal

variable {P plan tab} {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace Layout

variable (lay : Layout P plan tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val)

/-- The value a drawn unit's table's message holds at commit string `p`: `rowOf` of the witness's bits at `rowAt`. -/
noncomputable def row (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : Msg plan tab S R u) (p : lay.Pos) : lay.Val :=
  lay.rowOf p (bitsAt (wit plan tab S R u M) (lay.rowAt S R u p))

/-- The salt it holds there: the witness's 192 bytes at `saltAt`. -/
noncomputable def salt (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : Msg plan tab S R u) (p : lay.Pos) : List UInt8 :=
  bytesAt (wit plan tab S R u M) (lay.saltAt S R u p)

/-- **The registration fact**: in a satisfying witness of a drawn unit's table, the leaf of the value and salt the
witness holds at each of the unit's commit strings is the registered leaf there. -/
def Registered : Prop :=
  ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u,
    (scheme (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)

/-- An opening of a commit string: the commit string, a value and a salt. -/
abbrev Opening : Type := lay.Pos × lay.Val × List UInt8

/-- What a drawn unit's table's message opens commit string `p` to: `p`, and the value and salt it holds there. -/
noncomputable def openingAt (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : Msg plan tab S R u) (p : lay.Pos) : lay.Opening :=
  (p, lay.row S R u M p, lay.salt S R u M p)

/-- An opening's commitment: its commit string and its leaf there. -/
def com (o : lay.Opening) : lay.Pos × Dig := (o.1, (scheme o.1).leaf o.2.1 o.2.2)

/-- **The collision from two openings**: the extractor of the first one's commit string. -/
def collide (o o' : lay.Opening) : List UInt8 × List UInt8 := (scheme o.1).pair o.2.1 o.2.2 o'.2.1 o'.2.2

theorem collide_spec (o o' : lay.Opening) (hc : lay.com scheme o = lay.com scheme o') (hv : o.2.1 ≠ o'.2.1) :
    Collides (lay.collide scheme o o') := by
  obtain ⟨p, v, y⟩ := o
  obtain ⟨p', v', y'⟩ := o'
  simp only [com, Prod.mk.injEq] at hc
  obtain ⟨rfl, hl⟩ := hc
  exact (scheme p).pair_spec v y v' y' hv hl

/-- **The value binding of a layout**, given its leaf schemes and the registration fact: an opening is a commit string, a
value and a salt; its commitment the commit string and its leaf; the registered commitment at `p` is `p` and the
registered leaf; a drawn unit's table's message opens `p` to the value and salt it holds there; and `collide` is the leaf
scheme's extractor. -/
noncomputable def binding (hreg : lay.Registered scheme) : P.ValueBinding plan tab decode H512 where
  Pos := lay.Pos
  Val := lay.Val
  pos := lay.pos
  bit := lay.bit
  Opening := lay.Opening
  inhOp := ⟨(default, default, [])⟩
  Com := lay.Pos × Dig
  val o := o.2.1
  com := lay.com scheme
  cs R p := (p, lay.regLeaf R p)
  opening S R u M p := lay.openingAt S R u M p
  opening_spec S R u hu M hM g hg :=
    ⟨congrArg (Prod.mk (lay.pos g)) (hreg S R u hu M hM g hg), (lay.decode_row S R u hu M hM g hg).symm⟩
  collide := lay.collide scheme
  collide_spec o o' hc hv := lay.collide_spec scheme o o' hc hv

end Layout

end FlockSoundness.Binding
