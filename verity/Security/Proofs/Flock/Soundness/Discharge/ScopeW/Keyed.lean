import Proofs.Flock.Soundness.Discharge.ScopeW.RowEnc
import Proofs.Flock.Soundness.Discharge.ScopeW.Rows
import Proofs.Flock.Soundness.Discharge.Hm.Site
import Proofs.Flock.Soundness.Discharge.ScopeW.Commit
import Proofs.Flock.Soundness.Discharge.ScopeW.RowVal

set_option Elab.async false

/-!
# `rs` at the production program's places, each row under its own schema

`Layout.keyedRowsOuts` hashes every row with `Layout.rowEnc`, the bytes `sha512/row/v1` hashes, so `hHm`
(`Hm.keyedRowsOuts_computes`) takes every row to be a `sha512/row/v1` row (`hv1`). `keyedRowsW` is its readers' rows
(`Layout.keyReaders`, the public outputs registered by `outBitOf`) under each row's own schema (`encW`, `pairW`), with
`keyedRowsOuts`' copies and pins: `Layout.keyProg_copies`, `keyProg_bcPinned` and `keyProg_valPinned` read no row's
shape. Its layout is `Layout.keyedLayout`, unchanged.

`hHm` at the four shapes is `keyedRowsOutsW_computes`: the `b ‖ c` at a row's `hm96` output is the commit string of
`rowEncOf` of the row's value (`bc_site_commitW` at every shape, with `rowEncOf_row`).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Binding Placed
open FlockSoundness.Discharge.Layout (keyN portRowW KeysRow OutsOut PortReads BcSite keyedLayout keyReaders keyPlace
  outBitOf outsOf outsOut_outsOf keysRow_of_parse bitCode bitCode_inj)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) {popOf : Reg → Public}

/-- **`rs` at the production program's places, each row under its own schema**: `Layout.keyedRows`' readers, copies and
pins, with `encW` and `pairW` in place of `rowEnc`. -/
noncomputable def keyedRowsW (outs : Finset (Fin (keyN c hU n))) (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) (hkeys : KeysRow c hU (portRowW c)) :
    HmRowsP (keyedLayout hparse outs saltCol popOf) :=
  rowsP (keyReaders (portRowW c) hparse outs saltCol popOf (outBitOf c hU n outs popOf)) (keyPlace hparse outs)
    (encW c n (Fin (keyN c hU n))) (pairW c n (Fin (keyN c hU n))) pairW_spec bitCode bitCode_inj
    (Layout.keyProg_copies hparse outs saltCol popOf _)
    (Layout.keyProg_bcPinned hparse outs saltCol popOf _ hsite)
    (Layout.keyProg_valPinned hparse outs saltCol popOf _ hkeys (Const.constCols_keyProg hU hparse outs)
      (Zero.zeroCols_keyProg hU hparse outs) (Layout.keyProg_hout hparse ho hsite houts))

/-- **`keyedRowsW` at the returned outputs' bits** (`outsOf`), `KeysRow` from the parse: `Layout.keyedRowsOuts`'
form. -/
noncomputable def keyedRowsOutsW {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    HmRowsP (keyedLayout hparse (outsOf c hU n) saltCol popOf) :=
  keyedRowsW hparse saltCol (outsOf c hU n) ho hsite (outsOut_outsOf c hU n) (keysRow_of_parse hc h1 hpr)

/-- **`HmRowComputes` at `keyedRowsOutsW`'s rows** (`hHm`), at the `hm96` slots' salt columns: in a satisfying witness
of a drawn unit's table, each row port's `b ‖ c` at the unit's site is the commit string of `rowEncOf` of the row's
value under its port's schema, at every row of the four shapes (`hW`, which `scopeOkW_ports` gives). -/
theorem keyedRowsOutsW_computes {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyedRowsOutsW hparse Hm.saltCol hc h1 hpr ho hsite).Computes := by
  unfold HmRowsP.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyedLayout hparse (outsOf c hU n) Hm.saltCol popOf).pos g = pos
  rcases pos with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hmem : c.ports[q] ∈ c.ports.toList := Array.getElem_mem_toList q.isLt
    have hpq : (si.acc S R u hu).st.c.ports[q.val]! = c.ports[q] := by
      rw [hce, getElem!_pos c.ports q.val q.isLt]
      rfl
    have hle : portRowW c q ≤ Finset.univ.sup (portRowW c) + 1 :=
      (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
    have hb := bc_site_commitW (si.acc S R u hu) (by rw [hce]; exact ho) (by rw [hce]; exact hW) hM (si.o S R u hu)
      (si.hvu S R u hu) hq
    rw [hpq] at hb
    show Binding.bytesAt _ (Layout.keyBcAddr si S R u q) = Binding.commitString
      (rowEncOf c.ports[q] (W := Finset.univ.sup (portRowW c) + 1)
        (Layout.widthOf (Finset.univ.sup (portRowW c) + 1) (portRowW c q),
          Layout.padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) =>
            Layout.keyRowAddr si S R u q k.val)))
      (Binding.bytesAt _ (Layout.keySaltAddr si Hm.saltCol S R u (.inl (u', q)) (hU := hU)))
    have e1 : Layout.keyBcAddr si S R u q = fun (j : Fin 128) (t : Fin 8) =>
        Layout.siteAddr (si.acc S R u hu) (si.o S R u hu)
          (Layout.bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [Layout.keyBcAddr, hu, dite_true]
    have e2 : Layout.keySaltAddr si Hm.saltCol S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        Layout.siteAddr (si.acc S R u hu) (si.o S R u hu)
          (Hm.saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [Layout.keySaltAddr, hu, dite_true]
    rw [rowEncOf_row (hW _ hmem) ((parse_facts hc).v1Segs _ hmem) (parse_bits hc _ hmem)
      (show portRowW c q = 16 * c.ports[q].words from rfl) hle, e1, e2, hb]
    refine congrArg (fun r => Binding.commitString (rowMsgOf c.ports[q] r) _) ?_
    congr 1
    funext j
    congr 1
    funext t
    have hj : (8 * j.val + t.val) / 8 = j.val := by omega
    have ht : (8 * j.val + t.val) % 8 = t.val := by omega
    show _ = bit (Binding.wit plan tab S R u M (Layout.keyRowAddr si S R u q (8 * j.val + t.val)))
    simp only [Layout.keyRowAddr, hu, ↓reduceDIte, hj, ht]
  · intro h
    exact (Bool.false_ne_true h).elim

end FlockSoundness.Discharge.ScopeW
