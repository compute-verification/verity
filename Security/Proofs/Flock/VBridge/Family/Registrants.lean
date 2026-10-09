import Proofs.Flock.Soundness.Registered.Binding

/-!
# Two registrants' records at one registered root and position

`RowsShared` (`Family/Registered.lean`) asks that two sessions' registrant records (`Reads`) agree where they read one
registered root at one position. Each record's opening opens that root there (`Reads.committed`), so they agree up to a
SHA-512 collision, in a pair fixed by the two records rather than by a run:

* `registrants_eq_or_collides`: two registered reads at one root and position carry one row's bytes (`enc`), or their
  openings collide (`TwoOpenings.collides`), or hm96's leaves do (`hm96Pair_spec`).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Refine FlockSoundness.Binding FlockSoundness.Registered Audit

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

end FlockVBridge.Family
