module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.PlansProof
public import Proofs.Flock.RecGeneric.AtFlockPlans
public import Proofs.Flock.Claims.Recursion.KS
public import Proofs.Flock.Claims.Recursion.LastBindsR

/-!
# The Flock instance: well formed, its inner and outer extractions (F4, F5, F8)

* **F4**: the element-level Flock is a well-formed inner protocol (`protoWFOfFlock`, from the plans' reads), and
  `KSFlockLin` is its `Proto.KSAbortLin` (`kSAbortLinOfFlockLin`), its three plans being `B`, `FC` and `F0` at the runs
  (`kSAbortLinOfFlockLin_of`, from `forkPlanIsBAbort`, `capPlanIsFC` and `conflictPlanIsF0`).
* **F5**: today's `E_O` at the compiled record is knowledge sound at every statement from each session's
  (`outerKSOfSessions`): `PrBad_EO_le` sums the sessions' failures, and `Σ_s √(γ_s·k·p) = √((Σ_s √(γ_s·k))²·p)`.
* **F8**: `Proto.LastCoinFree` at the element-level Flock (`Proto.ofFlock`) is `DE`'s `Bound.LastStepFree`: one
  statement (`lastCoinFree_ofFlock`). So it holds at every `H` at each bound record whose `DE` steps as the deployed
  reader (`RecDeploy.lastCoinFree_ofFlock_R`, from `RecDeploy.lastStepFree_of_R`) or the deployed plain verifier
  (`RecDeploy.lastCoinFree_ofFlock_V`, from `RecDeploy.lastStepFree_of_V`): L1's last-coin hypothesis holds at the
  deployment with no premise.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

theorem fin3_forall {P : Fin 3 → Prop} (h0 : P 0) (h1 : P 1) (h2 : P 2) : ∀ i, P i := by
  intro i; fin_cases i
  exacts [h0, h1, h2]

/-- **F4, well formed**: the element-level Flock's answers decode, its `κ` is nonnegative where its tables' errors are,
its weights and tails are nonnegative, and its plans meet a bad run at most `0`, `2` and `k + 1` times as often as one
run is bad (`forkPlanReads`, `capPlanReads`, `conflictPlanReads`). -/
theorem protoWFOfFlock {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) : ProtoWFOfFlock Bd decI Cp := by
  intro hdec hε
  show _ ∧ _ ∧ _ ∧ _ ∧ _
  refine ⟨hdec, fun x => ?_, ?_, ?_, ?_⟩
  · show 0 ≤ Claims.Flock.κ Bd.DE x
    unfold Claims.Flock.κ
    have h1 : (0 : ℝ) < 2 ^ (-(Bd.DE.waits x : ℝ)) := by positivity
    have h2 : (0 : ℝ) ≤ 2 * Bd.DE.k x / Bd.DE.cap x := by positivity
    have h3 := Real.exp_pos (-(Bd.DE.k x : ℝ) / 8)
    have h4 := hε x
    linarith
  · exact fin3_forall (P := fun i => ∀ x, 0 ≤ ((Proto.ofFlock Bd decI Cp).terms i).w x)
      (fun x => show 0 ≤ 4 * (Bd.DE.V.k x : ℝ) by positivity)
      (fun x => show 0 ≤ 4 * (Cp.width x : ℝ) by positivity)
      (fun _ => show (0 : ℝ) ≤ 4 by norm_num)
  · exact fin3_forall (P := fun i => ∀ x m, 0 ≤ ((Proto.ofFlock Bd decI Cp).terms i).tail x m)
      (fun _ _ => show (0 : ℝ) ≤ 0 from le_rfl)
      (fun _ m => show 0 ≤ 1 / ((m : ℝ) + 1) by positivity)
      (fun x m => show 0 ≤ ((Bd.DE.k x : ℝ) + 1) / ((m : ℝ) + 1) by positivity)
  · exact fin3_forall
      (P := fun i => ∀ m, (((Proto.ofFlock Bd decI Cp).terms i).F m).ReadsLe ((Proto.ofFlock Bd decI Cp).terms i).d)
      (fun _ => forkPlanReads Coin Bd.DE.V.k) (fun m => capPlanReads Coin Cp m)
      (fun m => conflictPlanReads Coin Bd.DE.k m)

/-- **F4, the inner protocol's KS**: wherever `B`'s, `FC`'s and `F0`'s plans are `B`, `FC` and `F0` at a protocol's
runs that may abort (F3), `KSFlockLin` is the element-level Flock's `Proto.KSAbortLin`: its three terms are
`KSFlockLin`'s, at the loop bounds `m 1` and `m 2`. -/
theorem kSAbortLinOfFlockLin_of {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (hB : ForkPlanIsBAbort Coin) (hC : CapPlanIsFC Coin) (hF : ConflictPlanIsF0 Coin) (H : Msg → Msg)
    (Bd : Bound Coin Col T) (Cp : Caps Coin) : KSAbortLinOfFlockLin H Bd Cp := by
  intro hKS decI hdec x Ω _ _ P A _ _ keep m
  change Fin 3 → ℕ at m
  have h := hKS decI hdec x Ω P A keep (m 1) (m 2)
  show (Inner.ofFlock Bd.DE decI).PrBadAbort H x P keep ≤ Claims.Flock.κ Bd.DE x + ∑ i : Fin 3,
      (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).w x *
        (((![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).F (m i)).PrAt H (runModel H Bd.DE.V P x keep) x +
          (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).tail x (m i))
  rw [Fin.sum_univ_three]
  show _ ≤ _ + (4 * (Bd.DE.V.k x : ℝ) *
        ((forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H (runModel H Bd.DE.V P x keep) x + 0) +
      4 * (Cp.width x : ℝ) * ((capPlan Cp (m 1)).PrAt H (runModel H Bd.DE.V P x keep) x + 1 / ((m 1 : ℝ) + 1)) +
      4 * ((conflictPlan (Coin := Coin) Bd.DE.k (m 2)).PrAt H (runModel H Bd.DE.V P x keep) x +
        ((Bd.DE.k x : ℝ) + 1) / ((m 2 : ℝ) + 1)))
  rw [hB H Bd.DE.V x Ω P A keep, hC H Bd.DE.V Cp x Ω P A keep (m 1), hF H Bd.DE.V Bd.DE.k x Ω P A keep (m 2)]
  first | linarith | exact h.trans (le_of_eq (by ring))

/-- **F4, the inner protocol's KS**: `KSFlockLin` is the element-level Flock's `Proto.KSAbortLin`, its plans being
`B`, `FC` and `F0` at a protocol's runs that may abort (`forkPlanIsBAbort`, `capPlanIsFC`, `conflictPlanIsF0`). -/
theorem kSAbortLinOfFlockLin {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Msg → Msg) (Bd : Bound Coin Col T) (Cp : Caps Coin) : KSAbortLinOfFlockLin H Bd Cp :=
  kSAbortLinOfFlockLin_of (forkPlanIsBAbort Coin) (capPlanIsFC Coin) (conflictPlanIsF0 Coin) H Bd Cp

/-- **F5**: each session's `E^rep` failing with probability at most `κ_s + √(γ_s·k(x)·Pr[B])`
(`Claims.Recursion.OuterKS`) gives `Comp.OuterKSAt` at `Comp.ofRec`, at every decoder that meets the spec and every
statement. -/
theorem outerKSOfSessions {Coin Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col']
    [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (IV : Verifier Coin) {W : Type} (I : Claims.Recursion.Inner Coin W)
    (DO : Claims.Flock.Deploy Coin Col' T') : OuterKSOfSessions H D IV I DO := by
  intro hO decO hdec x Ω _ _ P
  have hp0 : 0 ≤ PrB H (D.record IV).V P x := PrB_nonneg H (D.record IV).V P x
  have hG0 : 0 ≤ ∑ s : Fin ((D.record IV).nS x), Real.sqrt (((D.record IV).nums x s).γ * (D.record IV).V.k x) :=
    Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
  refine (PrBad_EO_le H (D.record IV) I DO decO x P).trans ?_
  show _ ≤ ∑ s : Fin ((D.record IV).nS x), ((D.record IV).nums x s).κ +
    Real.sqrt ((∑ s : Fin ((D.record IV).nS x), Real.sqrt (((D.record IV).nums x s).γ * (D.record IV).V.k x)) ^ 2 *
      PrB H (D.record IV).V P x)
  rw [Real.sqrt_mul' _ hp0, Real.sqrt_sq hG0, Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun s _ => (hO decO hdec x Ω P s).trans (le_of_eq ?_)
  rw [Real.sqrt_mul (mul_nonneg (Nums.γ_nonneg _) (Nat.cast_nonneg _))]

/-- **F8 is `Bound.LastStepFree`**: the element-level Flock's last step reads no last coin wherever `DE`'s does. -/
theorem lastCoinFree_ofFlock {Coin Col T : Type} {H : Msg → Msg} {Bd : Bound Coin Col T} (h : Bd.LastStepFree H)
    (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) : (Proto.ofFlock Bd decI Cp).LastCoinFree H :=
  h

end Claims.RecGeneric

namespace RecDeploy

open Claims (Msg)
open Claims.Recursion Claims.RecGeneric
open FlockDeploy (Coin FinalP)

/-- **F8 at the deployed reader**: at every bound record whose `DE` steps as `FlockDeploy.VR rd`, at every `H`. -/
theorem lastCoinFree_ofFlock_R {Col T : Type} [DecidableEq Col] (H : Msg → Msg) (rd : Msg → Option FinalP)
    (Bd : Bound Coin Col T) (hV : Bd.DE.V = FlockDeploy.VR rd) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) :
    (Proto.ofFlock Bd decI Cp).LastCoinFree H :=
  lastCoinFree_ofFlock (lastStepFree_of_R H rd Bd hV) decI Cp

/-- **F8 at the deployed plain verifier**: at every bound record whose `DE` steps as `FlockDeploy.V false`, at every
`H`. -/
theorem lastCoinFree_ofFlock_V {Col T : Type} [DecidableEq Col] (H : Msg → Msg) (Bd : Bound Coin Col T)
    (hV : Bd.DE.V = FlockDeploy.V false) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) :
    (Proto.ofFlock Bd decI Cp).LastCoinFree H :=
  lastCoinFree_ofFlock (lastStepFree_of_V H Bd hV) decI Cp

end RecDeploy
