import Proofs.Flock.Soundness.Registered.Binding
import Proofs.Flock.Soundness.Discharge.Layout.RowEnc
import Proofs.Flock.VBridge.Family.Registered

/-!
# Two registrants' records at one registered root and position

`RowsShared` (`Family/Registered.lean`) asks that two sessions' registrant records (`Reads`) agree where they read one
registered root at one position. Each record's opening opens that root there (`Reads.committed`), so they agree up to a
SHA-512 collision, in a pair fixed by the two records rather than by a run:

* `registrants_eq_or_collides`: two registered reads at one root and position carry one row's bytes (`enc`), or their
  openings collide (`TwoOpenings.collides`), or hm96's leaves do (`hm96Pair_spec`);
* `rowEnc_get_eq`: one row's bytes (`rowEnc`, what `sha512/row/v1` hashes) fix its bits at every width, so two
  sessions' values agree at each bit both widths hold;
* `SharedRows`, `rowsShared_of_sharedRows`: `RowsShared` where two sessions' registrant records at one root and
  position hold one row's bytes, as they do when the family takes them from one choice;
* `NoRegistrantCollision`, `sharedRows_of`, `rowsShared_of`: `SharedRows`, and so `RowsShared`, where no two such
  records give either collision.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Refine FlockSoundness.Binding FlockSoundness.Registered Audit

section Two

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Layout P plan tab decode} {rs : HmRows lay}
  {C' : Circuit} {n' : ℕ} {P' : Audit.Partition C' n'} {Reg₁ : Type} {mPts' : ℕ}
  {plan' : Finset (Fin n') → Reg₁ → List (TabSpec mPts')}
  {tab' : (S : Finset (Fin n')) → (R : Reg₁) → Fin n' → Fin (plan' S R).length}
  {decode' : Audit.Partition.DecoderB (C := C') (Reg := Reg₁) plan' tab'}
  {lay' : Layout P' plan' tab' decode'} {rs' : HmRows lay'}

/-- **Two registrants' rows at one registered root and position are one row's bytes, or SHA-512 collides**: where two
records' registered reads `p`, `p'` name one root (`port`) and one position (`readAt`), each record's opening opens it
there (`Reads.committed`), so their leaves are one, or the two openings conflict (`TwoOpenings.collides`); one leaf
over two rows' bytes is hm96's collision (`hm96Pair_spec`). -/
theorem registrants_eq_or_collides (rd : Reads lay rs) (rd' : Reads lay' rs') {p : lay.Pos} {p' : lay'.Pos}
    (hp : rd.isRead p) (hp' : rd'.isRead p') (hport : rd.port p = rd'.port p') (hat : rd.readAt p = rd'.readAt p') :
    rs.enc (rd.wv p) = rs'.enc (rd'.wv p') ∨
      Collides (⟨rd.port p, rd.readAt p, rd.leaf p, rd.wpath p, rd'.leaf p', rd'.wpath p'⟩ : TwoOpenings).pair ∨
      Collides (hm96Pair (rs.enc (rd.wv p)) (rd.ws p) (rs'.enc (rd'.wv p')) (rd'.ws p')) := by
  by_cases hl : rd.leaf p = rd'.leaf p'
  · by_cases he : rs.enc (rd.wv p) = rs'.enc (rd'.wv p')
    · exact Or.inl he
    · exact Or.inr (Or.inr (hm96Pair_spec _ _ _ _ he hl))
  · refine Or.inr (Or.inl (TwoOpenings.collides _ ⟨hl, rd.committed p hp, ?_⟩))
    show Flock.Registered.opensLeaf (rd.port p) (rd'.leaf p').1 (rd.readAt p) (rd'.wpath p') = true
    rw [hport, hat]
    exact rd'.committed p' hp'

end Two

/-! ## One row's bytes at two widths -/

section Enc

open FlockSoundness.Discharge.Layout

/-- `sha512/row/v1`'s first block begins with the label's `v`. -/
theorem rowPrefix_head (w : ℕ) : (Flock.HmRow.rowPrefix w).data.toList.head? = some 118 := by
  have hA : ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16]).data.toList.head? = some 118 := by
    decide
  unfold Flock.HmRow.rowPrefix Flock.Hm96.pad
  rw [ByteArray.data_append, ByteArray.data_append, Array.toList_append, Array.toList_append, List.head?_append,
    List.head?_append, hA]
  rfl

theorem rowEnc_head_shaped {W : ℕ} {v : BitVal W} (h : RowShaped v) : (rowEnc v).head? = some 118 := by
  simp [rowEnc, h, List.head?_append, rowPrefix_head]

theorem rowEnc_head_other {W : ℕ} {v : BitVal W} (h : ¬ RowShaped v) : (rowEnc v).head? = some 0 := by
  simp [rowEnc, h, List.head?_append, List.head?_replicate]

/-- `get_of_rowBytes` at two widths. -/
theorem get_of_rowBytes₂ {W W' : ℕ} {v : BitVal W} {w : BitVal W'} (hv : RowShaped v) (h : rowBytes v = rowBytes w)
    (i : ℕ) (hi : i < W) (hi' : i < W') (hiv : i < v.1.val) : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ := by
  have hj : i / 8 < 2 * (v.1.val / 16) := by have := hv.1; omega
  have hl : i / 8 < (rowBytes v).length := by simp only [rowBytes, List.length_ofFn]; exact hj
  have e := List.getElem_of_eq h hl
  simp only [rowBytes, List.getElem_ofFn] at e
  have := congrFun (byteOf_inj e) ⟨i % 8, Nat.mod_lt _ (by norm_num)⟩
  simp only [BitVal.get, show 8 * (i / 8) + i % 8 = i by omega, hi, hi', ↓reduceDIte] at this
  exact this

/-- **One row's bytes fix its bits at every width** (`rowEnc_inj` at two widths): a row's length fixes its words and
its bytes its bits, both zero past the words; the label's first byte (`rowPrefix_head`) tells a row from any other
value; two other values are `bitCode`s of one width. -/
theorem rowEnc_get_eq {W W' : ℕ} {v : BitVal W} {w : BitVal W'} (h : rowEnc v = rowEnc w) (i : ℕ) (hi : i < W)
    (hi' : i < W') : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ := by
  by_cases hv : RowShaped v <;> by_cases hw : RowShaped w
  · have hl := congrArg List.length h
    rw [rowEnc_row_length hv, rowEnc_row_length hw] at hl
    have hk : v.1.val = w.1.val := by have := hv.1; have := hw.1; omega
    simp only [rowEnc, hv, hw, ↓reduceIte] at h
    obtain ⟨-, hb⟩ := List.append_inj h (by rw [rowPrefix_length, rowPrefix_length])
    by_cases hiv : i < v.1.val
    · exact get_of_rowBytes₂ hv hb i hi hi' hiv
    · rw [hv.2 ⟨i, hi⟩ (by show v.1.val ≤ i; omega), hw.2 ⟨i, hi'⟩ (by show w.1.val ≤ i; omega)]
  · have := congrArg List.head? h
    rw [rowEnc_head_shaped hv, rowEnc_head_other hw] at this
    exact absurd this (by decide)
  · have := congrArg List.head? h
    rw [rowEnc_head_other hv, rowEnc_head_shaped hw] at this
    exact absurd this (by decide)
  · have hl := congrArg List.length h
    rw [rowEnc_other_length hv, rowEnc_other_length hw] at hl
    obtain rfl : W = W' := by omega
    simp only [rowEnc, hv, hw, ↓reduceIte] at h
    rw [bitCode_inj (List.append_inj h rfl).2]

end Enc

/-! ## At the family -/

section family

open FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.Model
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockSoundness.Discharge.Layout (rowEnc)

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **No two sessions' registrant records collide at one registered root and position.** At every accepted outer
proof, where two sessions' registered reads name one root and one position, neither their openings
(`TwoOpenings.pair`) nor their hm96 leaves (`hm96Pair`) are a SHA-512 collision. The pair is fixed by the history (each
record is a field of the session the family builds there), not by a session's run, so no finder of the transfer's
`cr/sha-512` terms (`ZkOuter.CR`: each session's tables, its registered values and its registered paths, each against
that session's own record) outputs it. -/
def NoRegistrantCollision : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    ¬ Collides (⟨(F.zs R ω t R₂ s).rd.port (.inl (u, q)), (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)),
        (F.zs R ω t R₂ s).rd.leaf (.inl (u, q)), (F.zs R ω t R₂ s).rd.wpath (.inl (u, q)),
        (F.zs R ω t R₂ s').rd.leaf (.inl (u', q')), (F.zs R ω t R₂ s').rd.wpath (.inl (u', q'))⟩ : TwoOpenings).pair ∧
      ¬ Collides (hm96Pair (rowEnc (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))))
        ((F.zs R ω t R₂ s).rd.ws (.inl (u, q))) (rowEnc (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q'))))
        ((F.zs R ω t R₂ s').rd.ws (.inl (u', q'))))

/-- **Two sessions' registrant records at one registered root and position hold one row's bytes.** At every accepted
outer proof, where two sessions' registered reads name one root and one position, the registrants' values there have
one `rowEnc`. It holds where the family's sessions take the registrant's opening at a root and position from one
choice, the same in every session; `sharedRows_of` gives it from `NoRegistrantCollision` otherwise. -/
def SharedRows : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    rowEnc (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))) = rowEnc (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q')))

/-- **`SharedRows` where no two registrant records collide** (`registrants_eq_or_collides`). -/
theorem sharedRows_of (h : F.NoRegistrantCollision dg N k Rw) : F.SharedRows dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat
  obtain ⟨h1, h2⟩ := h R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat
  rcases registrants_eq_or_collides (F.zs R ω t R₂ s).rd (F.zs R ω t R₂ s').rd hr hr' hport hat with he | hc | hc
  · exact he
  · exact absurd hc h1
  · exact absurd hc h2

/-- **`RowsShared` from one row's bytes** (`SharedRows`, `rowEnc_get_eq`). -/
theorem rowsShared_of_sharedRows (h : F.SharedRows dg N k Rw) : F.RowsShared dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat b hb hb'
  exact rowEnc_get_eq (h R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat) b _ _

/-- **`RowsShared` where no two registrant records collide**. -/
theorem rowsShared_of (h : F.NoRegistrantCollision dg N k Rw) : F.RowsShared dg N k Rw :=
  F.rowsShared_of_sharedRows dg N k Rw (F.sharedRows_of dg N k Rw h)

end VStarFamily

end family

end FlockVBridge.Family
