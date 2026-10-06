/-
Level 3, L3-A: the statement's level-0 matrices `A_0`, `B_0` over GF(2), as the verifier's lincheck fold builds them
(`Stmt.fold`: PROTOCOL.md §10, §16.1). They are what `fold_get`, `baseOf_get`, `mask_term` and `lookupTerm` read:
- the fold writes each slot's base at the slot's columns, later ranges over earlier, so column `j`'s entries come from the
  last range covering `j` (`lastCover`, as `rangesVal`), and only from rows in the same slot;
- within a slot, a net's entry at `(i, i')` is the count, mod 2, of `i'` in its row `i` (`sparseEntry`, the terms of
  `sparseT`; GF(2^128) has characteristic 2); a lookup slot's B side adds the table's product-row entries (`lookupEntry`, the
  terms of `lookupTerm`); a mask slot is the identity on both sides (its base is `(α + 1)·e_lo`);
- Δ's pairs `(i, c)` add at row `i`, column `c`, counted mod 2.
Where no range covers `j` the matrices are Δ alone. No layout hypothesis is used.
-/
import Proofs.Flock.Level3.FoldStmt

open Flock Lookup

namespace FlockLevel3

/-- The last range of `rs` covering column `c`: the one whose base the fold leaves at `c` (`rangesVal`). -/
def lastCover (rs : List Range) (c : ℕ) : Option Range :=
  rs.foldl (fun acc r => if covers r c then some r else acc) none

/-- A sparse matrix's entry at row `i`, column `i'`: the count, mod 2, of `i'` among row `i`'s columns (`sparseT`'s terms). -/
def sparseEntry (m : Sparse) (i i' : ℕ) : ZMod 2 :=
  if i < m.rows then (((List.range m.len[i]!).filter fun t => m.col i t = i').length : ZMod 2) else 0

/-- A lookup slot's B entries read from its table (`lookupTerm`'s terms): product row `prod0 + j·nh + h`, for `j < bits` and
`h < nh = 2^(n − lo)`, reads low minterm `loTop + l`, `l < 2^lo`, when bit `j` of `table[h·2^lo + l]` is set. -/
def lookupEntry : Option LookupTable → ℕ → ℕ → ZMod 2
  | none, _, _ => 0
  | some t, i, i' =>
    let nh := 2 ^ (t.n - t.lo)
    let nlo := 2 ^ t.lo
    if t.prod0 ≤ i ∧ i < t.prod0 + t.bits * nh ∧ t.loTop ≤ i' ∧ i' < t.loTop + nlo ∧
        (word t nlo (2 ^ t.bits - 1) ((i - t.prod0) % nh) (i' - t.loTop)).testBit ((i - t.prod0) / nh) then 1 else 0

/-- A slot type's local A entry at row `i`, column `i'`. -/
def slotEntryA (st : Stmt) : Kind → ℕ → ℕ → ZMod 2
  | .net n, i, i' => sparseEntry (st.c.nets.getD n default).2.a i i'
  | .mask, i, i' => if i = i' then 1 else 0
  | .comp, i, i' => sparseEntry st.leaf.comp.a i i'

/-- A slot type's local B entry at row `i`, column `i'`, a lookup slot's table entries included. -/
def slotEntryB (st : Stmt) : Kind → ℕ → ℕ → ZMod 2
  | .net n, i, i' => sparseEntry (st.c.nets.getD n default).2.b i i' + lookupEntry (st.c.nets.getD n default).2.lookup i i'
  | .mask, i, i' => if i = i' then 1 else 0
  | .comp, i, i' => sparseEntry st.leaf.comp.b i i'

/-- The ranges' share of a level-0 matrix at row `r`, column `j`: the last range covering `j`, if `r` is in the same slot. -/
def rangesEntry (st : Stmt) (entry : Kind → ℕ → ℕ → ZMod 2) (r j : ℕ) : ZMod 2 :=
  match lastCover st.c.ranges.toList j with
  | some rg => if r / 2 ^ rg.slotLog = j / 2 ^ rg.slotLog then entry rg.kind (r % 2 ^ rg.slotLog) (j % 2 ^ rg.slotLog) else 0
  | none => 0

/-- Δ's pairs at row `r`, column `j`, counted mod 2. -/
def deltaEntry (pairs : Array (ℕ × ℕ)) (r j : ℕ) : ZMod 2 := ((pairs.toList.filter (· = (r, j))).length : ZMod 2)

/-- **The level-0 A matrix** the verifier folds: `Stmt.fold α lw ρ` is `α·A_0ᵀe + B_0ᵀe` for `e = lw ⊗ eq(ρ)`. -/
def placedA (st : Stmt) : Matrix (Fin (2 ^ st.c.kLog)) (Fin (2 ^ st.c.kLog)) (ZMod 2) :=
  fun r j => rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val

/-- **The level-0 B matrix** the verifier folds. -/
def placedB (st : Stmt) : Matrix (Fin (2 ^ st.c.kLog)) (Fin (2 ^ st.c.kLog)) (ZMod 2) :=
  fun r j => rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val

end FlockLevel3
