import Proofs.Flock.Soundness.Registered.ReadsX

/-!
# The registered reads' term at the outcome's registration

`ReadsX.prob_off_le` bounds accepting while a registered read is off at one fixed registration. In the live audit the
registration is the outcome's first component, so `ReadsX.prob_off_le_at` takes it from the outcome (`reg`), with the
same finder: one play of the game, and the extractor's pair at the off read.
-/

namespace FlockSoundness.Registered

open Game Refine Binding Audit
open scoped ENNReal

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace ReadsX

variable {lay : Layout P plan tab decode} {rs : HmRowsX lay} (rd : ReadsX lay rs)

/-- **`prob_off_le` at the outcome's registration** (`cr/sha-512`, `SHA512CRStrict`): in a game `G` played by `σ` whose
accepted outcomes the verifier's registered check holds on at the registration `reg o`, through the outcome's paths, the
probability of accepting while some registered read's leaf under `reg o` is not the registrant's is at most
`qR²/2^513`, for the finder that plays `G` once and outputs the extractor's pair (`pick`) at `reg o`. -/
theorem prob_off_le_at {β : Type} (G : Game β) (σ : Strategy G) (reg : β → Reg) (accept : β → Prop)
    (path : β → lay.Pos → ByteArray) (hchk : ∀ o, accept o → rd.Checked (reg o) (path o)) {qR : ℝ}
    (hR : (StrictCR.oneRun G σ fun o => rd.pick (reg o) (path o)).CR H512 qR) :
    prob (fun o => accept o ∧ rd.Off (reg o)) G σ ≤ ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  have hle : prob (fun o => accept o ∧ rd.Off (reg o)) G σ ≤
      prob (fun o => StrictCR.IsColl H512 (rd.pick (reg o) (path o))) G σ :=
    prob_mono (fun o ⟨ha, hoff⟩ =>
      (rd.openings (reg o) (path o) (rd.offAt (reg o))).isColl (rd.offAt_conflict (hchk o ha) hoff)) G σ
  have hev : (prob (fun o => StrictCR.IsColl H512 (rd.pick (reg o) (path o))) G σ).toReal ≤
      qR ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    have := StrictCR.Finder.CR.le hR
    rwa [StrictCR.oneRun_ev] at this
  have hfin : prob (fun o => StrictCR.IsColl H512 (rd.pick (reg o) (path o))) G σ ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  exact hle.trans ((ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 hev)

end ReadsX

end FlockSoundness.Registered
