import FlockLevel3.Folded
import FlockSoundness.Protocol.Refine.Tapes

/-!
# The statement decoding, part 1: the executable's layout is a block R1CS (PROTOCOL.md §10, §16.1)

The executable never holds `A₀` and `B₀`: its lincheck fold (`Stmt.fold`) reads them off the circuit's layout. `stmtOf st`
is the model statement those matrices make. Entry `(i, c)` of `A₀` or `B₀` counts, mod 2:
* the covering range's slot matrix at `(i mod 2^sl, c mod 2^sl)`, when row `i` is in column `c`'s slot. The covering
  range is the last in the layout that covers `c`, since the fold overwrites (`lastCover`). A mask slot is the identity
  on both sides;
* for a lookup slot, on the B side, the product rows its table's words send to `c` (`lkCnt`);
* the pairs `(i, c)` of Δ_A or Δ_B.

`ofCircuit_fold`: the fold `Setup.ofCircuit` hands the verifier realizes `stmtOf st` (`FoldRealizes`, R3), given the
layout facts `StmtWF`: `6 ≤ k_log ≤ m`, the pin inside the block, and every range's slots between 2^6 and 2^k_log. The
regions are a parameter here.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

/-! ### The matrices -/

/-- The occurrences of column `c` in row `i` of a sparse matrix: what `foldT` adds at `c` (none past its rows or its
width). -/
def spCnt (m : Sparse) (i c : ℕ) : ℕ :=
  if i < m.rows ∧ c < m.width then ((Finset.range m.len[i]!).filter fun t => m.col i t = c).card else 0

/-- The product rows a lookup table sends to column `c` (§16.5): row `prod0 + j·2^(n−lo) + h` for each set bit `j < bits`
of word `(h, l)` with `loTop + l = c`. -/
def lkCnt (lk : LookupTable) (i c : ℕ) : ℕ :=
  ((Finset.range (2 ^ (lk.n - lk.lo)) ×ˢ Finset.range (2 ^ lk.lo) ×ˢ Finset.range 32).filter fun x =>
    lk.loTop + x.2.1 = c ∧ (Lookup.word lk (2 ^ lk.lo) (2 ^ lk.bits - 1) x.1 x.2.1).testBit x.2.2 = true ∧
      x.2.2 < lk.bits ∧ lk.prod0 + x.2.2 * 2 ^ (lk.n - lk.lo) + x.1 = i).card

/-- A slot type's A side at local row `i` and column `c`. -/
def slotA (st : Stmt) : Kind → ℕ → ℕ → ℕ
  | .comp => spCnt st.leaf.comp.a
  | .net j => spCnt (st.c.nets.getD j default).2.a
  | .mask => fun i c => if i = c then 1 else 0

/-- A lookup slot's table rows, inside the B matrix's width. -/
def lkSide (b : Sparse) : Option LookupTable → ℕ → ℕ → ℕ
  | some lk => fun i c => if c < b.width then lkCnt lk i c else 0
  | none => fun _ _ => 0

/-- A slot type's B side at local row `i` and column `c`; a lookup slot adds its table's product rows. -/
def slotB (st : Stmt) : Kind → ℕ → ℕ → ℕ
  | .comp => spCnt st.leaf.comp.b
  | .net j => fun i c => spCnt (st.c.nets.getD j default).2.b i c +
      lkSide (st.c.nets.getD j default).2.b (st.c.nets.getD j default).2.lookup i c
  | .mask => fun i c => if i = c then 1 else 0

/-- The last range of `rs` whose slots cover column `c`: the one whose base the fold leaves at `c`. -/
def lastCover : List Range → ℕ → Option Range
  | [], _ => none
  | r :: rs, c => (lastCover rs c).or (if covers r c then some r else none)

/-- The ranges' share of entry `(i, c)`: the covering range's slot side, when row `i` is in column `c`'s slot. -/
def rangeCnt (side : Kind → ℕ → ℕ → ℕ) (rs : List Range) (i c : ℕ) : ℕ :=
  match lastCover rs c with
  | none => 0
  | some r => if i / 2 ^ r.slotLog = c / 2 ^ r.slotLog then side r.kind (i % 2 ^ r.slotLog) (c % 2 ^ r.slotLog) else 0

theorem lastCover_spec : ∀ (rs : List Range) (c : ℕ) (r : Range), lastCover rs c = some r → r ∈ rs ∧ covers r c
  | [], _, _, h => by cases h
  | r₀ :: rs, c, r, h => by
      rw [lastCover] at h
      cases h' : lastCover rs c with
      | some r' =>
          rw [h'] at h
          simp only [Option.some_or, Option.some.injEq] at h
          subst h
          obtain ⟨hm, hc⟩ := lastCover_spec rs c _ h'
          exact ⟨List.mem_cons_of_mem _ hm, hc⟩
      | none =>
          rw [h'] at h
          by_cases hc : covers r₀ c
          · simp only [hc, ite_true, Option.none_or, Option.some.injEq] at h
            subst h
            exact ⟨List.mem_cons_self, hc⟩
          · simp [hc] at h

/-! ### Counting in characteristic 2 -/

/-! ### One slot -/

/-! ### The statement -/

/-- The pairs `(i, c)` of Δ. -/
def dCnt (d : Array (ℕ × ℕ)) (i c : ℕ) : ℕ := d.toList.count (i, c)

/-- `A₀` read off the executable's layout: the ranges' A sides and Δ_A, mod 2. -/
def matA (st : Stmt) (i c : ℕ) : ZMod 2 := (rangeCnt (slotA st) st.c.ranges.toList i c + dCnt st.da i c : ℕ)

/-- `B₀` read off the executable's layout: the ranges' B sides, lookup tables included, and Δ_B, mod 2. -/
def matB (st : Stmt) (i c : ℕ) : ZMod 2 := (rangeCnt (slotB st) st.c.ranges.toList i c + dCnt st.db i c : ℕ)

/-- The layout facts that reading the fold as matrices needs. -/
structure StmtWF (st : Stmt) : Prop where
  hk6 : 6 ≤ st.c.kLog
  hk : st.c.kLog ≤ st.m
  pin : st.pin < 2 ^ st.c.kLog
  slots : ∀ r ∈ st.c.ranges.toList, 6 ≤ r.slotLog ∧ r.slotLog ≤ st.c.kLog

/-- **The model statement of an executable statement** (§16.1): its `m`, `k_log` and pin, the matrices its layout
makes, and the given regions. -/
def stmtOf (st : Stmt) (h : StmtWF st) (regions : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    Model.Statement where
  m := st.m
  kLog := st.c.kLog
  hk := h.hk
  hk6 := h.hk6
  pin := ⟨st.pin, h.pin⟩
  A₀ := fun i c => matA st i.val c.val
  B₀ := fun i c => matB st i.val c.val
  regions := regions

end FlockSoundness.Refine
