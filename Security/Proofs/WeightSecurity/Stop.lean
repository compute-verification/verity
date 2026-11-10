module

public import Definitions.WeightSecurity.Served
public import Proofs.Flock.EndToEnd.Period.Union
import Proofs.Flock.Soundness.Game

/-!
# A period of served windows: the lead and the incorrect proof blocks left undetected

A deployment's monitoring period is a run of windows (`FlockSoundness.Game.periodRun`), each recorded as a
`WeightSecurity.Served`.

* `period_lead`: if at every strategy satisfying `good` a window's records all carry one number `w` of incorrect proof
  blocks and its sample misses them all with probability at most `x ^ w`, then the windows before the first sampled
  incorrect proof block carry at least `M` (`Served.lead`) with probability at most `x ^ M`, for `x ≤ 1`. It is
  `period_horizon` stopped at the first window whose sample holds one.
* `period_undetected`: if each such window is also accepted with a sampled incorrect proof block with probability at
  most `ε i`, the run leaves `M` incorrect proof blocks undetected (`Served.Undetected`) with probability at most
  `Σ_t ε (i + t) + x ^ M`.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open FlockSoundness.Game
open scoped ENNReal

variable {β : ℕ → Type} {O V Tick Adv : Type}

theorem period_lead (G : (i : ℕ) → Verity.Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → Served O V Tick Adv)
    (good : (i : ℕ) → Strategy (G i) → Prop) (x : ℝ≥0∞) (hx : x ≤ 1)
    (hstage : ∀ i τ, good i τ → ∃ w : ℕ, (∀ b, (rec i τ b).incorrect = w) ∧
      prob (fun b => ¬ (rec i τ b).sampledIncorrect) (G i) τ ≤ x ^ w) :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)) (M : ℕ), PeriodGood G good i T s →
      value (fun l => M ≤ Served.lead l) (periodRun G rec i T s) ≤ x ^ M
  | _, _, _, 0, _ => by
      rw [pow_zero]
      exact value_le_one _ _
  | 0, _, _, M + 1, _ => by
      classical
      show value (fun l => M + 1 ≤ Served.lead l) (Verity.Game.ret []) ≤ _
      simp [Served.lead]
  | T + 1, i, s, M + 1, hs => by
      obtain ⟨w, hw, hp⟩ := hstage i _ hs.1
      have h := value_runBind_le_mul (fun _ => periodG G (i + 1) T)
        (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
        (fun l => M + 1 ≤ Served.lead l) (fun b => ¬ (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect)
        (x ^ (M + 1 - w)) (G i) s
        (atEnds_mono (fun b s' hg => by
          classical
          rw [value_map]
          by_cases hm : (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect
          · rw [ite_eq_right (not_not.2 hm)]
            refine (value_mono (q := fun _ => False) (fun l hl => ?_) _).trans_eq (value_false _)
            simp [Served.lead, hm] at hl
          · rw [ite_eq_left hm]
            refine (value_mono (fun l hl => ?_) _).trans
              (period_lead G rec good x hx hstage T (i + 1) s' (M + 1 - w) hg)
            simp only [Served.lead, ite_eq_right hm, hw b] at hl
            omega) (G i) s hs.2)
      calc value (fun l => M + 1 ≤ Served.lead l) (periodRun G rec i (T + 1) s)
          ≤ prob (fun b => ¬ (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect) (G i) (Strategy.ofBind _ (G i) s) *
              x ^ (M + 1 - w) := h
        _ ≤ x ^ w * x ^ (M + 1 - w) := by gcongr
        _ = x ^ (w + (M + 1 - w)) := (pow_add _ _ _).symm
        _ ≤ x ^ (M + 1) := pow_le_pow_of_le_one zero_le hx (by omega)

theorem period_undetected (G : (i : ℕ) → Verity.Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → Served O V Tick Adv)
    (good : (i : ℕ) → Strategy (G i) → Prop) (ε : ℕ → ℝ≥0∞) (x : ℝ≥0∞) (hx : x ≤ 1)
    (hacc : ∀ i τ, good i τ → prob (fun b => (rec i τ b).accepted ∧ (rec i τ b).sampledIncorrect) (G i) τ ≤ ε i)
    (hstage : ∀ i τ, good i τ → ∃ w : ℕ, (∀ b, (rec i τ b).incorrect = w) ∧
      prob (fun b => ¬ (rec i τ b).sampledIncorrect) (G i) τ ≤ x ^ w)
    (T i : ℕ) (s : Strategy (periodG G i T)) (M : ℕ) (hs : PeriodGood G good i T s) :
    value (Served.Undetected M) (periodRun G rec i T s) ≤ ∑ t ∈ Finset.range T, ε (i + t) + x ^ M := by
  show value (fun l => (∃ r ∈ l, r.accepted ∧ r.sampledIncorrect) ∨ M ≤ Served.lead l) (periodRun G rec i T s) ≤ _
  exact (value_or_le _ _ _).trans (add_le_add
    (period_union G rec (fun r => r.accepted ∧ r.sampledIncorrect) good ε hacc T i s hs)
    (period_lead G rec good x hx hstage T i s M hs))

end WeightSecurity.SecurityProofs
