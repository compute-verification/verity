module

public import Proofs.WeightSecurity.TwoProd
public import Proofs.WeightSecurity.TwoRegWitnesses
public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Witnesses.SamplingTwo
import Proofs.Core.Game.Basic
import Proofs.Core.ValueSalts
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Witnesses.SamplingTwoReg
import Proofs.WeightSecurity.TwoProdPeriod

/-!
# The two-stage period's product form says something

`CompilerPeriodTwoProd`'s premise holds at an instance, with the premises its committed-trace forms add
(`periodTwoProd_regVal`, `periodTwoLead_regVal`), and `PeriodTwoLead` and `PeriodTwoProd` fail where it doesn't.

* `periodTwoProd_honest`: at `twoLTwo`'s verification phases, `twoA` is knowledge sound and every phase's commitment
  binds and its sessions meet the two-stage contract (`compilerKSTwo_honest`). The instance is a toy of two sessions
  over SHA-512, and no served verifier.
* `periodTwoLead_nontrivial`, `periodTwoProd_nontrivial`: at `sksLTwo`'s verification phases, which accept after one
  round and sample nothing, both statements fail at the committed boundary trace and `δ = 1/2`. One verification phase,
  played by `twoStrat` at the silent prover, is accepted surely and samples no proof block
  (`acceptedSampledAt_sksLTwo`: the hypothesis holds at `ε = 0`), while its escape is `0` (`escAt_sksLTwo`: its
  committed boundary trace opens nothing, so its replay block holds an incorrect proof block at every interior): the
  event holds surely and the bound is `1/2`.
* `sksLTwo_not_sessions`: so `sksLTwo` does not meet the two-stage contract, the premise `compilerPeriodTwoProd` takes.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open Claims.Witness (silentP run_accV_out)
open Claims.Witness.SamplingGeneric
open Claims.Witness.Sampling (toyStmt toyPart)
open FlockSoundness.Game (periodG)
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

/-- **A period of one two-stage verification phase**, played by `twoStrat`: the run flips the phase's coins and
records the phase at the count and the escape its release, statement and prover fix. -/
theorem twoAtRun_twoStrat (H : Msg → Msg) (L : CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (m : (O → V) × Tick × Adv)
    (x : Msg) (P : Prover Unit) :
    twoAtRun H (fun _ => L) val 1 (twoStrat L m x P) =
      .coin (Fin (L.V.k x) → Coin) fun r =>
        .ret [twoAtRec H L (val 0 m x P) (regCountAt L x (val 0 m x P)) (escAt L x (val 0 m x P)) (m, ⟨x, P, r⟩)] :=
  rfl

/-- One verification phase whose sample holds none of its incorrect proof blocks and whose escape is `0` leaves
escapes that multiply below every `δ > 0`. -/
theorem undetectedBelow_one {w : ServedEsc O V Tick Adv} {δ : ℝ} (hδ : 0 < δ) (hs : ¬ w.sampledIncorrect)
    (he : w.escape = 0) : ServedEsc.UndetectedBelow δ [w] := by
  refine Or.inr ?_
  have h0 : ServedEsc.leadEscape [w] = 0 := by simp [ServedEsc.leadEscape, hs, he]
  rw [h0]
  exact ENNReal.ofReal_pos.2 hδ

/-- One accepted verification phase whose escape is `0` is a period whose phases are all accepted and whose escapes
multiply below every `δ > 0`. -/
theorem prodBelow_one {w : ServedEsc O V Tick Adv} {δ : ℝ} (hδ : 0 < δ) (ha : w.accepted) (he : w.escape = 0) :
    (∀ r ∈ [w], r.accepted) ∧ ([w].map ServedEsc.escape).prod < ENNReal.ofReal δ := by
  refine ⟨fun r hr => ?_, ?_⟩
  · rw [List.mem_singleton.1 hr]
    exact ha
  · rw [List.map_singleton, List.prod_singleton, he]
    exact ENNReal.ofReal_pos.2 hδ

/-- **A period of one verification phase that samples no incorrect proof block and whose escape is `0`** leaves
escapes below every `δ > 0` undetected surely. -/
theorem value_undetectedBelow_one (H : Msg → Msg) (L : CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (m : (O → V) × Tick × Adv)
    (x : Msg) (P : Prover Unit) {δ : ℝ} (hδ : 0 < δ)
    (hs : ∀ r : Fin (L.V.k x) → Coin, ¬ ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () (val 0 m x P) r)
    (he : escAt L x (val 0 m x P) = 0) :
    Verity.Game.value (ServedEsc.UndetectedBelow δ) (twoAtRun H (fun _ => L) val 1 (twoStrat L m x P)) = 1 := by
  rw [twoAtRun_twoStrat, Verity.Game.value_coin_ret]
  exact Verity.ValueSalts.prCoin_of_forall fun r => undetectedBelow_one hδ (hs r) he

/-- **A period of one verification phase that is accepted surely and whose escape is `0`** is accepted throughout,
with escapes that multiply below every `δ > 0`, surely. -/
theorem value_prodBelow_one (H : Msg → Msg) (L : CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (m : (O → V) × Tick × Adv)
    (x : Msg) (P : Prover Unit) {δ : ℝ} (hδ : 0 < δ)
    (ha : ∀ r : Fin (L.V.k x) → Coin, (run H P L.V x () r).out = .acc)
    (he : escAt L x (val 0 m x P) = 0) :
    Verity.Game.value
        (fun l : List (ServedEsc O V Tick Adv) =>
          (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.escape).prod < ENNReal.ofReal δ)
        (twoAtRun H (fun _ => L) val 1 (twoStrat L m x P)) = 1 := by
  rw [twoAtRun_twoStrat, Verity.Game.value_coin_ret]
  exact Verity.ValueSalts.prCoin_of_forall fun r => prodBelow_one hδ (ha r) he

/-! ## The premises hold together -/

/-- **The product form's premises hold together** at `twoLTwo`'s verification phases: `twoA` is knowledge sound and
every phase's commitment binds and its sessions meet the two-stage contract (`compilerKSTwo_honest`), which are
`periodTwoProd_regVal`'s and `periodTwoLead_regVal`'s premises; the last conjunct is `CompilerPeriodTwoProd`'s one. -/
theorem periodTwoProd_honest :
    (twoA Bool).KS FlockDeploy.sha512 ∧
      (∀ _t : ℕ, (twoLTwo Bool).commit.Binds FlockDeploy.sha512 ∧ (twoLTwo Bool).Sessions FlockDeploy.sha512) ∧
      ∀ _t : ℕ, (twoLTwo Bool).Sessions FlockDeploy.sha512 := by
  obtain ⟨hA, hB, hS, -, -⟩ := compilerKSTwo_honest
  exact ⟨hA, fun _ => ⟨hB, hS⟩, fun _ => hS⟩

/-! ## The statements fail at a verifier that samples nothing -/

/-- At boundary values that commit nothing at the input, `toyTwo`'s `P̄_a` at best interiors is `0`: its one replay
block holds an incorrect proof block at every interior. -/
theorem toyTwo_paBestAt {val : ℕ → Option Bool} (hv : val 0 = none) :
    toyTwo.PaBestAt toyStmt toyPart val = 0 := by
  unfold Claims.Sampling.Design.Two.PaBestAt
  exact toyTwo_Pa (toy_footprintAt hv _)

open Classical in
/-- `sksLTwo`'s escape is `0` at boundary values that commit nothing at the input. -/
theorem escAt_sksLTwo (x : Msg) {val : ℕ → Option Bool} (hv : val 0 = none) : escAt (sksLTwo Bool) x val = 0 := by
  have hval : escAt (sksLTwo Bool) x val = ENNReal.ofReal (toyTwo.PaBestAt toyStmt toyPart val) :=
    ite_eq_left inScopeTwo_toy
  rw [hval, toyTwo_paBestAt hv, ENNReal.ofReal_zero]

/-- `sksLTwo` samples nothing: at no coins does its sample hold a proof block. -/
theorem sksLTwo_samples_none (H : Msg → Msg) (x : Msg) (P : Prover Unit) (val : ℕ → Option Bool)
    (r : Fin ((sksLTwo Bool).V.k x) → Bool) :
    ¬ ∃ u ∈ (sksLTwo Bool).checked x r, u ∈ (sksLTwo Bool).FregTwoAt H P x () val r :=
  fun ⟨u, hu, _⟩ => absurd (show u ∈ ([] : List ℕ) from hu) (by simp)

/-- So at no coins is `sksLTwo` accepted with a sampled incorrect proof block: the event's probability is `0` at
every statement, prover and boundary values. -/
theorem acceptedSampledAt_sksLTwo (H : Msg → Msg) (x : Msg) (P : Prover Unit) (val : ℕ → Option Bool) :
    acceptedSampledAt H (sksLTwo Bool) x P val = 0 := by
  unfold acceptedSampledAt
  exact Eq.trans (Pr_congr fun r => iff_false_intro fun h => sksLTwo_samples_none H x P val r h.2) Pr_false

/-- `sksLTwo`'s escape at its committed boundary trace, at the empty statement and the silent prover, is `0`: the
trace opens nothing (`sksL_regVal`). -/
theorem escAt_sksLTwo_regVal :
    escAt (sksLTwo Bool) [] ((sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ()) = 0 :=
  escAt_sksLTwo [] (val := (sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ()) (sksL_regVal _ silentP [] () 0)

/-- **`PeriodTwoLead` fails** at `sksLTwo`'s verification phases, the committed boundary trace and `δ = 1/2`: one
verification phase, at the empty statement and the silent prover, samples nothing and has escape `0`, so its escapes
multiply below `1/2` surely, while the bound is `1/2`. -/
theorem periodTwoLead_nontrivial :
    ¬ PeriodTwoLead (O := Unit) (V := Unit) (Tick := Unit) (Adv := Unit) FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) := by
  intro h
  let m : (Unit → Unit) × Unit × Unit := (fun _ => (), (), ())
  have hv : Verity.Game.value (ServedEsc.UndetectedBelow (1 / 2))
      (twoAtRun FlockDeploy.sha512 (fun _ => sksLTwo Bool)
        (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) 1
        (twoStrat (sksLTwo Bool) m [] silentP)) = 1 :=
    value_undetectedBelow_one FlockDeploy.sha512 (sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) m [] silentP (by norm_num)
      (fun r => sksLTwo_samples_none _ _ _ _ r) escAt_sksLTwo_regVal
  have h1 := h 1 (fun _ => 0) (twoStrat (sksLTwo Bool) m [] silentP)
    (fun _ _ _ _ => (acceptedSampledAt_sksLTwo _ _ _ _).le) (1 / 2)
  rw [hv] at h1
  norm_num at h1

/-- **`PeriodTwoProd` fails** at `sksLTwo`'s verification phases, the committed boundary trace and `δ = 1/2`: one
verification phase, at the empty statement and the silent prover, is accepted surely with escape `0`, while the bound
is `1/2`. -/
theorem periodTwoProd_nontrivial :
    ¬ PeriodTwoProd (O := Unit) (V := Unit) (Tick := Unit) (Adv := Unit) FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) := by
  intro h
  let m : (Unit → Unit) × Unit × Unit := (fun _ => (), (), ())
  have hv : Verity.Game.value
      (fun l : List (ServedEsc Unit Unit Unit Unit) =>
        (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.escape).prod < ENNReal.ofReal (1 / 2))
      (twoAtRun FlockDeploy.sha512 (fun _ => sksLTwo Bool)
        (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) 1
        (twoStrat (sksLTwo Bool) m [] silentP)) = 1 :=
    value_prodBelow_one FlockDeploy.sha512 (sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) m [] silentP (by norm_num)
      (fun r => run_accV_out FlockDeploy.sha512 silentP [] () r) escAt_sksLTwo_regVal
  have h1 := h 1 (fun _ => 0) (twoStrat (sksLTwo Bool) m [] silentP)
    (fun _ _ _ _ => (acceptedSampledAt_sksLTwo _ _ _ _).le) (1 / 2)
  rw [hv] at h1
  norm_num at h1

/-- **`sksLTwo` does not meet the two-stage contract**, the premise `compilerPeriodTwoProd` takes: under it,
`PeriodTwoLead` would hold at `sksLTwo`'s verification phases. -/
theorem sksLTwo_not_sessions : ¬ (sksLTwo Bool).Sessions FlockDeploy.sha512 := fun hS =>
  periodTwoLead_nontrivial
    (compilerPeriodTwoProd Bool (toyA Bool) Unit Unit Unit Unit FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      (fun _ _ x P => (sksLTwo Bool).regVal FlockDeploy.sha512 P x ()) fun _ => hS).1

end WeightSecurity.SecurityProofs
