import Proofs.Flock.Claims.KS.RepDist
import Proofs.Flock.Claims.KS.Rounds
import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Soundness.ExecPlacement
import Proofs.Flock.Soundness.Refine.Regions

/-!
# A rep's level caps at the plain fork round's count

With four lanes at every level after level 0 (`fast100_lanes`), rep `r`'s level-`ℓ + 1` cap is `capRound k0 (ℓ + 1)`
coins into Ligerito, after the zerocheck's, the lincheck's and the opening's coins: the count `forkRound` reads after
the link message and rep `r`'s `r · R` earlier rounds.

* `ligSlotDist_capRound`: Ligerito's part of the count is `capRound`'s;
* `dist_lockRep_capRound`: so every path of rep `r` that meets the cap meets it after that many coins, at every level
  (past the last cap there is no such node);
* `repDist_preLig`, `dist_lockRep_plain`: at a plain table's statement, two points per region and `fast100`, that count
  is `preLig spec st + capRound k0 (ℓ + 1)`;
* `forkRound_eq`: the plain fork round is the link message, rep `r`'s `r · R` earlier rounds and that count.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- **Ligerito's level-`ℓ + 1` cap is `capRound k0 (ℓ + 1)` coins in**, with four lanes after level 0. -/
theorem ligSlotDist_capRound (k0 : ℕ) (levs : List Accounting.Level) (hl : ∀ lv ∈ levs, lv.lanes = 4) (ℓ : ℕ)
    (hℓ : ℓ < levs.length) : ligSlotDist levs ℓ + k0 + 1 + 1 = capRound k0 (ℓ + 1) := by
  cases ℓ with
  | zero =>
    have h0 : ligSlotDist levs 0 = 0 := rfl
    have h1 : capRound k0 (0 + 1) = k0 + 2 := ite_eq_left (by omega)
    omega
  | succ p =>
    have h0 : ligSlotDist levs (p + 1) = 7 + recSlotDist levs p := rfl
    have h1 : capRound k0 (p + 1 + 1) = k0 + 13 + 11 * (p + 1 + 1 - 2) := ite_eq_right (by omega)
    have h2 := recSlotDist_eq levs p hl hℓ
    omega

/-- **A rep's level-`ℓ + 1` cap, at the plain fork round's count**: `capRound k0 (ℓ + 1)` coins into Ligerito after
the zerocheck's, the lincheck's and the opening's, at every level of a schedule with four lanes after level 0 (past
the last cap there is no such node). -/
theorem dist_lockRep_capRound (A : Model.Arith F K) (r : ℕ) (hr : r < 2) (S : Model.Statement) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (C₀ : Model.Oracle F) (extra : List (Model.Claim F S.m)) (hl : ∀ lv ∈ l₁ :: rest, lv.lanes = 4) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1))
      (capRound k0 (ℓ + 1) + (extra.length + 1 + 1 + 1) + (S.kLog - 6 + 3) + (S.m - 6 + 3))
      (lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra) := by
  by_cases hlt : ℓ < (l₁ :: rest).length
  · have h := dist_lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra ℓ
    unfold repSlotDist at h
    rw [← ligSlotDist_capRound k0 (l₁ :: rest) hl ℓ hlt]
    exact h
  · exact .noSlot (noSlot_lockRep_ge A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra ℓ
      (by simp only [List.length_cons] at hlt ⊢; omega))

omit [Fintype F] [DecidableEq F] in
/-- **The model's rounds before Ligerito are `preLig`'s**: at a plain table's statement and two points per region, the
zerocheck's, the lincheck's and the opening's coins of `dist_lockRep_capRound` are `preLig spec st`. -/
theorem repDist_preLig {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} (L : FlockSoundness.Layout st)
    (hp : spec.points = 2) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → F) (k0 ℓ : ℕ) :
    capRound k0 (ℓ + 1) +
        ((Model.extraClaims (L.model (Refine.regionsOf st)) ptLocal mPts pts).length + 1 + 1 + 1) +
        ((L.model (Refine.regionsOf st)).kLog - 6 + 3) + ((L.model (Refine.regionsOf st)).m - 6 + 3) =
      preLig spec st + capRound k0 (ℓ + 1) := by
  rw [FlockSoundness.extraClaims_length]
  have hR : (L.model (Refine.regionsOf st)).regions.length = st.regions.size := by
    show (Refine.regionsOf st).length = st.regions.size
    simp [Refine.regionsOf]
  rw [hR]
  show _ = (3 + (st.m - 6)) + (3 + (st.c.kLog - 6)) + (2 + spec.points * st.regions.size + 1) + capRound k0 (ℓ + 1)
  rw [hp]
  show capRound k0 (ℓ + 1) + (2 * st.regions.size + 1 + 1 + 1) + (st.c.kLog - 6 + 3) + (st.m - 6 + 3) = _
  omega

/-- **A plain table's rep caps at `preLig + capRound`**: at the `fast100` schedule and two points per region, rep `r`'s
level-`ℓ + 1` cap in `lockAfterL`'s reps is `preLig spec st + capRound k0 (ℓ + 1)` coins in (past the last cap there is
no such node). -/
theorem dist_lockRep_plain {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} (L : FlockSoundness.Layout st)
    (hp : spec.points = 2) (A : Model.Arith F K) (r : ℕ) (hr : r < 2) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (C₀ : Model.Oracle F) (ptLocal mPts : ℕ)
    (pts : Fin 2 → Fin mPts → F) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1)) (preLig spec st + capRound k0 (ℓ + 1))
      (lockRep (D := D) A r hr (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm C₀
        (Model.extraClaims (L.model (Refine.regionsOf st)) ptLocal mPts pts)) := by
  have h := dist_lockRep_capRound (D := D) A r hr (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm C₀
    (Model.extraClaims (L.model (Refine.regionsOf st)) ptLocal mPts pts) (fast100_lanes hsch) ℓ
  rwa [repDist_preLig L hp ptLocal mPts pts k0 ℓ] at h

/-- **The plain fork round, split at rep `r`'s start**: the link message, rep `r`'s `r · R` earlier rounds, then
`dist_lockRep_plain`'s count. -/
theorem forkRound_eq (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) (k0 r ℓ : ℕ) :
    forkRound spec st k0 r ℓ = 1 + r * repRounds false spec st + (preLig spec st + capRound k0 (ℓ + 1)) := by
  unfold forkRound
  exact Nat.add_assoc _ _ _

end FlockDeploy.KS
