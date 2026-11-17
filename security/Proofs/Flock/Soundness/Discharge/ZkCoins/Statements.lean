module

public import Proofs.Flock.Soundness.Discharge.ZkCoins.SessionDefs
public import Proofs.Flock.Soundness.CROnly.Hm96Defs
import Mathlib.Combinatorics.Matroid.Init

/-! The coin step's guarantees, as statements. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game
open scoped ENNReal

def CommittedLe : Prop :=
  ∀ {I Blk Salt B Cs D α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [Fintype Salt] [Nonempty Salt]
    [AddCommGroup B] [Fintype B] (p : α → Prop) (g : TGame I Blk α), g.Distinct →
    ∀ (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) (δ₁ : ℝ≥0∞),
      Claims.Hm96Hiding My cs δ₁ →
      ∀ τ : (I → D) → TGame.CStrat Salt g, probC p g dig My cs H τ ≤ value p g.erase + Fintype.card I * δ₁

def CommittedLeHm96 : Prop :=
  ∀ {E I Blk Cs D α : Type} [Field E] [Fintype E] [DecidableEq E], Fintype.card E = 2 →
    ∀ [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Cs]
      (p : (Fin (512 + 1536 - 1) → E) → α → Prop) (g : (Fin (512 + 1536 - 1) → E) → TGame I Blk α),
      (∀ κ, (g κ).Distinct) →
      ∀ (dig : (Fin (512 + 1536 - 1) → E) → I → Blk → (Fin 512 → E)) (cs : (Fin 1536 → E) → Cs),
      (Finset.univ.image cs).card ≤ 2 ^ 512 →
      ∀ (H : (Fin (512 + 1536 - 1) → E) → (Fin 512 → E) × Cs → D)
        (τ : (κ : Fin (512 + 1536 - 1) → E) → (I → D) → TGame.CStrat (Fin 1536 → E) (g κ)),
        avg (fun κ => probC (p κ) (g κ) (dig κ) (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) (τ κ)) ≤
          avg (fun κ => value (p κ) (g κ).erase) + Fintype.card I * ENNReal.ofReal (1 / 2 ^ 257)

def SessionCoinsSound : Prop :=
  ∀ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith F K), A.Correct → ∀ (ts : List (ZkSession.TabZK F mPts)) (j : Fin ts.length)
    {Blk Salt B Cs D : Type} [Fintype Blk] [Nonempty Blk] [Fintype Salt] [Nonempty Salt] [AddCommGroup B] [Fintype B]
    (dc : Decoders Blk) (R : ℕ) (dig : SessIdx ts.length R → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D)
    (δ₁ : ℝ≥0∞), Claims.Hm96Hiding My cs δ₁ →
    ∀ τ : (SessIdx ts.length R → D) → TGame.CStrat Salt (sessionZKC dc R A ts),
      probC (fun outs => ZkSession.acceptedZK ts outs ∧ ¬ (ts.get j).Committed A (outs.getD j ZkSession.dfltZK).2)
          (sessionZKC dc R A ts) dig My cs H τ ≤
        (ts.get j).bound + ((1 + 2 * ts.length * R : ℕ) : ℝ≥0∞) * δ₁

def SessionCoinsHm96 : Prop :=
  ∀ {F K E : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Field E] [Fintype E]
    [DecidableEq E] {mPts : ℕ} (A : Model.Arith F K), A.Correct → Fintype.card E = 2 →
    ∀ (ts : List (ZkSession.TabZK F mPts)) (j : Fin ts.length) {Blk Cs D : Type} [Fintype Blk] [Nonempty Blk] [DecidableEq Cs]
      (dc : Decoders Blk) (R : ℕ) (dig : (Fin (512 + 1536 - 1) → E) → SessIdx ts.length R → Blk → (Fin 512 → E))
      (cs : (Fin 1536 → E) → Cs), (Finset.univ.image cs).card ≤ 2 ^ 512 →
      ∀ (H : (Fin (512 + 1536 - 1) → E) → (Fin 512 → E) × Cs → D)
        (τ : (Fin (512 + 1536 - 1) → E) → (SessIdx ts.length R → D) → TGame.CStrat (Fin 1536 → E) (sessionZKC dc R A ts)),
        avg (fun κ => probC (fun outs => ZkSession.acceptedZK ts outs ∧ ¬ (ts.get j).Committed A (outs.getD j ZkSession.dfltZK).2)
            (sessionZKC dc R A ts) (dig κ) (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) (τ κ)) ≤
          (ts.get j).bound + ((1 + 2 * ts.length * R : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 257)

end FlockSoundness.Discharge.ZkCoins
