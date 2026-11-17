module

public import Proofs.Flock.VBridge.Family.Registered
import all Init.Data.String.Defs
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Registered.Binding

/-! Two sessions' registrant records at one registered root and position agree up to a SHA-512 collision, which
`Accepted` excludes (`registrants_eq_or_collides`, `rowsShared`). -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Refine FlockSoundness.Binding FlockSoundness.Registered Audit

section Two

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Layout P plan tab decode} {rs : HmRowsX lay}
  {C' : Verity.Boolean.Circuit} {n' : ℕ} {P' : Audit.Partition C' n'} {Reg₁ : Type} {mPts' : ℕ}
  {plan' : Finset (Fin n') → Reg₁ → List (TabSpec mPts')}
  {tab' : (S : Finset (Fin n')) → (R : Reg₁) → Fin n' → Fin (plan' S R).length}
  {decode' : Audit.Partition.DecoderB (C := C') (Reg := Reg₁) plan' tab'}
  {lay' : Layout P' plan' tab' decode'} {rs' : HmRowsX lay'}

theorem registrants_eq_or_collides (rd : ReadsX lay rs) (rd' : ReadsX lay' rs') {p : lay.Pos} {p' : lay'.Pos}
    (hp : rd.isRead p) (hp' : rd'.isRead p') (hport : rd.port p = rd'.port p') (hat : rd.readAt p = rd'.readAt p') :
    rs.enc p (rd.wv p) = rs'.enc p' (rd'.wv p') ∨
      Collides (⟨rd.port p, rd.readAt p, rd.leaf p, rd.wpath p, rd'.leaf p', rd'.wpath p'⟩ : TwoOpenings).pair ∨
      Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p) (rs'.enc p' (rd'.wv p')) (rd'.ws p')) := by
  by_cases hl : rd.leaf p = rd'.leaf p'
  · by_cases he : rs.enc p (rd.wv p) = rs'.enc p' (rd'.wv p')
    · exact Or.inl he
    · exact Or.inr (Or.inr (hm96Pair_spec _ _ _ _ he hl))
  · refine Or.inr (Or.inl (TwoOpenings.collides _ ⟨hl, rd.committed p hp, ?_⟩))
    show Flock.Registered.opensLeaf (rd.port p) (rd'.leaf p').1 (rd.readAt p) (rd'.wpath p') = true
    rw [hport, hat]
    exact rd'.committed p' hp'

end Two

section Enc

open FlockSoundness.Discharge.Layout

theorem rowPrefix_head (w : ℕ) : (Flock.HmRow.rowPrefix w).data.toList.head? = some 118 := by
  have hA : ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16]).data.toList.head? = some 118 := by
    decide
  unfold Flock.HmRow.rowPrefix Flock.Hm96.pad
  rw [ByteArray.data_append, ByteArray.data_append, Array.toList_append, Array.toList_append, List.head?_append,
    List.head?_append, hA]
  rfl

theorem rowPrefixV2_head (l : ℕ) : (Flock.HmRow.rowPrefixV2 l).data.toList.head? = some 118 := by
  have hA : "verity/sha512-row/v2".toUTF8.data.toList.head? = some 118 := by decide
  rw [FlockLevel3.RowV2.prefix_data, List.append_assoc, List.append_assoc, List.head?_append, hA]
  rfl

theorem rowPre_head (v2 : Bool) (w : ℕ) : (FlockSoundness.Discharge.Hidden.rowPre v2 w).data.toList.head? = some 118 := by
  unfold FlockSoundness.Discharge.Hidden.rowPre
  split
  · exact rowPrefixV2_head _
  · exact rowPrefix_head _

theorem rowPre_length (v2 : Bool) (w : ℕ) : (FlockSoundness.Discharge.Hidden.rowPre v2 w).data.toList.length = 128 := by
  rw [Array.length_toList]
  exact FlockSoundness.Discharge.Hidden.size_rowPre v2 w

theorem rowEncP_head_shaped {pre : ℕ → ByteArray} (hpre : ∀ w, (pre w).data.toList.head? = some 118) {W : ℕ}
    {v : BitVal W} (h : RowShaped v) : (rowEncP pre v).head? = some 118 := by
  simp [rowEncP, h, List.head?_append, hpre]

theorem rowEncP_head_other {pre : ℕ → ByteArray} {W : ℕ} {v : BitVal W} (h : ¬ RowShaped v) :
    (rowEncP pre v).head? = some 0 := by
  simp [rowEncP, h, List.head?_append, List.head?_replicate]

theorem rowEncP_row_length {pre : ℕ → ByteArray} (hpre : ∀ w, (pre w).data.toList.length = 128) {W : ℕ}
    {v : BitVal W} (h : RowShaped v) : (rowEncP pre v).length = 128 + 2 * (v.1.val / 16) := by
  simp only [rowEncP, h, ↓reduceIte, List.length_append, hpre, rowBytes, List.length_ofFn]

theorem rowEncP_other_length {pre : ℕ → ByteArray} {W : ℕ} {v : BitVal W} (h : ¬ RowShaped v) :
    (rowEncP pre v).length = 128 + W + (2 * W + 1) := by
  simp only [rowEncP, h, ↓reduceIte, List.length_append, List.length_replicate, bitCode_length]

theorem get_of_rowBytes₂ {W W' : ℕ} {v : BitVal W} {w : BitVal W'} (hv : RowShaped v) (h : rowBytes v = rowBytes w)
    (i : ℕ) (hi : i < W) (hi' : i < W') (hiv : i < v.1.val) : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ := by
  have hj : i / 8 < 2 * (v.1.val / 16) := by have := hv.1; omega
  have hl : i / 8 < (rowBytes v).length := by simp only [rowBytes, List.length_ofFn]; exact hj
  have e := List.getElem_of_eq h hl
  simp only [rowBytes, List.getElem_ofFn] at e
  have := congrFun (byteOf_inj e) ⟨i % 8, Nat.mod_lt _ (by norm_num)⟩
  simp only [BitVal.get, show 8 * (i / 8) + i % 8 = i by omega, hi, hi', ↓reduceDIte] at this
  exact this

theorem rowEncP_get_eq {pre pre' : ℕ → ByteArray} (hl : ∀ w, (pre w).data.toList.length = 128)
    (hl' : ∀ w, (pre' w).data.toList.length = 128) (hh : ∀ w, (pre w).data.toList.head? = some 118)
    (hh' : ∀ w, (pre' w).data.toList.head? = some 118) {W W' : ℕ} {v : BitVal W} {w : BitVal W'}
    (h : rowEncP pre v = rowEncP pre' w) (i : ℕ) (hi : i < W) (hi' : i < W') : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ := by
  by_cases hv : RowShaped v <;> by_cases hw : RowShaped w
  · have hlen := congrArg List.length h
    rw [rowEncP_row_length hl hv, rowEncP_row_length hl' hw] at hlen
    have hk : v.1.val = w.1.val := by have := hv.1; have := hw.1; omega
    simp only [rowEncP, hv, hw, ↓reduceIte] at h
    obtain ⟨-, hb⟩ := List.append_inj h (by rw [hl, hl'])
    by_cases hiv : i < v.1.val
    · exact get_of_rowBytes₂ hv hb i hi hi' hiv
    · rw [hv.2 ⟨i, hi⟩ (by show v.1.val ≤ i; omega), hw.2 ⟨i, hi'⟩ (by show w.1.val ≤ i; omega)]
  · have := congrArg List.head? h
    rw [rowEncP_head_shaped hh hv, rowEncP_head_other hw] at this
    exact absurd this (by decide)
  · have := congrArg List.head? h
    rw [rowEncP_head_other hv, rowEncP_head_shaped hh' hw] at this
    exact absurd this (by decide)
  · have hlen := congrArg List.length h
    rw [rowEncP_other_length hv, rowEncP_other_length hw] at hlen
    obtain rfl : W = W' := by omega
    simp only [rowEncP, hv, hw, ↓reduceIte] at h
    rw [bitCode_inj (List.append_inj h rfl).2]

theorem rowEncPre_get_eq {b b' : Bool} {W W' : ℕ} {v : BitVal W} {w : BitVal W'}
    (h : rowEncP (FlockSoundness.Discharge.Hidden.rowPre b) v = rowEncP (FlockSoundness.Discharge.Hidden.rowPre b') w)
    (i : ℕ) (hi : i < W) (hi' : i < W') : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ :=
  rowEncP_get_eq (rowPre_length b) (rowPre_length b') (rowPre_head b) (rowPre_head b') h i hi hi'

theorem encPortH_of_none {p : Flock.Port} {W : ℕ} (h : p.segs = none) :
    FlockSoundness.Discharge.HiddenExec.encPortH (W := W) p =
      rowEncP (FlockSoundness.Discharge.Hidden.rowPre p.v2) := by
  simp [FlockSoundness.Discharge.HiddenExec.encPortH, h]

theorem bits_of_encPortH {p p' : Flock.Port} {W W' : ℕ} {v : BitVal W} {w : BitVal W'} (hs : p.segs = none)
    (hs' : p'.segs = none)
    (h : FlockSoundness.Discharge.HiddenExec.encPortH p v = FlockSoundness.Discharge.HiddenExec.encPortH p' w)
    (b : ℕ) (hb : b < W) (hb' : b < W') : v.2 ⟨b, hb⟩ = w.2 ⟨b, hb'⟩ :=
  rowEncPre_get_eq ((congrFun (encPortH_of_none hs) v).symm.trans (h.trans (congrFun (encPortH_of_none hs') w)))
    b hb hb'

end Enc

section family

open FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.Model
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockSoundness.Discharge.HiddenExec (encPortH)

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- Holds where every session's circuit is in V*'s scope (`readsUnsegC3` for C3's families); the registered check
does not give it, because it checks a segmented row in its own leaf schema rather than refusing it. -/
def ReadsUnsegmented : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s : Fin F.stmts.size) (u : Fin (F.Vs s).nv)
    (q : Fin (F.Vs s).c.ports.size), (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.Vs s).c.ports[q.val]!.segs = none

/-- Discharged by `Accepted`'s `¬ RegColl` (`noRegistrantCollision`), which the recursive audit prices once; no
finder behind the transfer's `cr/sha-512` term prices it, because each sees one session's record. -/
def NoRegistrantCollision : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    ¬ Collides (⟨(F.zs R ω t R₂ s).rd.port (.inl (u, q)), (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)),
        (F.zs R ω t R₂ s).rd.leaf (.inl (u, q)), (F.zs R ω t R₂ s).rd.wpath (.inl (u, q)),
        (F.zs R ω t R₂ s').rd.leaf (.inl (u', q')), (F.zs R ω t R₂ s').rd.wpath (.inl (u', q'))⟩ : TwoOpenings).pair ∧
      ¬ Collides (hm96Pair
        (encPortH (F.Vs s).c.ports[q.val]! (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))))
        ((F.zs R ω t R₂ s).rd.ws (.inl (u, q)))
        (encPortH (F.Vs s').c.ports[q'.val]! (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q'))))
        ((F.zs R ω t R₂ s').rd.ws (.inl (u', q'))))

def SharedRows : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    encPortH (F.Vs s).c.ports[q.val]! (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))) =
      encPortH (F.Vs s').c.ports[q'.val]! (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q')))

theorem sharedRows_of (h : F.NoRegistrantCollision dg N k Rw) : F.SharedRows dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat
  obtain ⟨h1, h2⟩ := h R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat
  rcases registrants_eq_or_collides (F.zs R ω t R₂ s).rd (F.zs R ω t R₂ s').rd hr hr' hport hat with he | hc | hc
  · exact he
  · exact absurd hc h1
  · exact absurd hc h2

theorem rowsShared_of_sharedRows (h : F.SharedRows dg N k Rw) (hu : F.ReadsUnsegmented dg N k Rw) :
    F.RowsShared dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat b hb hb'
  exact bits_of_encPortH (hu R ω t R₂ τ o ha s u q hr) (hu R ω t R₂ τ o ha s' u' q' hr')
    (h R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat) b _ _

theorem rowsShared_of (h : F.NoRegistrantCollision dg N k Rw) (hu : F.ReadsUnsegmented dg N k Rw) :
    F.RowsShared dg N k Rw :=
  F.rowsShared_of_sharedRows dg N k Rw (F.sharedRows_of dg N k Rw h) hu

theorem noRegistrantCollision : F.NoRegistrantCollision dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' _ _
  exact ⟨fun hc => ha.2.1 ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inl hc⟩,
    fun hc => ha.2.1 ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inr hc⟩⟩

theorem sharedRows : F.SharedRows dg N k Rw := F.sharedRows_of dg N k Rw (F.noRegistrantCollision dg N k Rw)

theorem rowsShared (hu : F.ReadsUnsegmented dg N k Rw) : F.RowsShared dg N k Rw :=
  F.rowsShared_of dg N k Rw (F.noRegistrantCollision dg N k Rw) hu

end VStarFamily

end family

end FlockVBridge.Family
