module

public import Proofs.WeightSecurity.TwoReg
public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Witnesses.SamplingTwo
import Proofs.Core.Game.Basic
import Proofs.Flock.Claims.Witnesses.SamplingTwoReg
import Proofs.Flock.Claims.Witnesses.SamplingTwoRegMiss

/-!
# The two-stage period's statements say something

`CompilerPeriodTwoReg`'s premises hold together, and `PeriodTwoReg` fails where they don't.

* `periodTwoReg_honest`: at `twoLTwo`'s windows every window meets `CompilerKSTwo`'s premises (`compilerKSTwo_honest`),
  and the design `twoTwo` catches each replay block at rate `1` and samples each cell at rate `1` (`twoTwo_firstRate`,
  `twoTwo_secondRate`).
* `periodTwoReg_nontrivial`: at `sksLTwo`'s windows, which accept after one round and sample nothing, `PeriodTwoReg`
  fails at both rates `1`. One window, played by `twoStrat` at the silent prover, leaves its replay block, wrong at the
  committed boundary (`toy_wrongAtBoundaryAt`), undetected surely, while the bound is `0`: `ε_reg,2` is `0` at every
  prover.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open Claims.Witness (silentP prB_accV)
open Claims.Witness.SamplingGeneric
open Claims.Witness.Sampling (toyStmt toyPart)
open FlockSoundness.Game (periodG)
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

/-- **One two-stage window's strategy**: release `m`, then send the statement `x` and the prover `P`. -/
def twoStrat (L : CompiledTwo Coin A) {Rls : Type} (m : Rls) (x : Msg) (P : Prover Unit) :
    Verity.Game.Strategy (periodG (fun _ : ℕ => twoGame L Rls) 0 1) :=
  ⟨m, (x, P), fun _ => PUnit.unit⟩

/-- **A period of one two-stage window**, played by `twoStrat`: the run flips the window's coins and records the
window at the count its statement and prover fix. -/
theorem twoRegRun_twoStrat (H : Msg → Msg) (L : CompiledTwo Coin A) (m : (O → V) × Tick × Adv) (x : Msg)
    (P : Prover Unit) :
    twoRegRun H (fun _ => L) 1 (twoStrat L m x P) =
      .coin (Fin (L.V.k x) → Coin) fun r => .ret [twoRegRec H L (regCount H L x P) (m, ⟨x, P, r⟩)] :=
  rfl

/-- **`CompilerPeriodTwoReg`'s premises hold together** at `twoLTwo`'s windows and both rates `1`: `twoA` is knowledge
sound, each window's commitment binds and its sessions meet the two-stage contract (`compilerKSTwo_honest`), and its
one statement's design is `twoTwo`, whose stages catch each replay block and sample each cell surely. -/
theorem periodTwoReg_honest :
    (twoA Bool).KS FlockDeploy.sha512 ∧
      (∀ _t : ℕ, (twoLTwo Bool).commit.Binds FlockDeploy.sha512 ∧ (twoLTwo Bool).Sessions FlockDeploy.sha512) ∧
      (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
      ∀ (_t : ℕ) x s Pt d, (twoLTwo Bool).parse x = some s → (twoLTwo Bool).partition s = some Pt →
        (twoLTwo Bool).two s = some d → InScopeTwo Pt d s → d.FirstRate 1 ∧ d.SecondRate Pt 1 := by
  obtain ⟨hA, hB, hS, -, -⟩ := compilerKSTwo_honest
  refine ⟨hA, fun _ => ⟨hB, hS⟩, zero_le_one, le_rfl, zero_le_one, le_rfl, fun _ x s Pt d hs hP hd _ => ?_⟩
  obtain rfl : s = twoStmt := (Option.some.inj hs).symm
  obtain rfl : Pt = twoPart := (Option.some.inj hP).symm
  obtain rfl : d = twoTwo := (Option.some.inj hd).symm
  exact ⟨twoTwo_firstRate, twoTwo_secondRate⟩

open Classical in
/-- **`PeriodTwoReg` fails** at `sksLTwo`'s windows and both rates `1`: one window, at the empty statement and the
silent prover, leaves its replay block wrong at the committed boundary undetected surely, while the bound is `0`. -/
theorem periodTwoReg_nontrivial :
    ¬ PeriodTwoReg (O := Unit) (V := Unit) (Tick := Unit) (Adv := Unit) FlockDeploy.sha512 (fun _ => sksLTwo Bool)
      1 1 := by
  intro h
  let m : (Unit → Unit) × Unit × Unit := (fun _ => (), (), ())
  have hε : ∀ (x : Msg) (P : Prover Unit), (sksLTwo Bool).εregTwo FlockDeploy.sha512 x P = 0 := by
    intro x P
    have hb : PrB FlockDeploy.sha512 (sksL Bool).V P x = 0 := prB_accV _ P x
    have h1 : (sksL Bool).εreg FlockDeploy.sha512 x P = 0 := by
      unfold Compiled.εreg
      rw [sksL_κs, hb, bconf_pr _ (sksL Bool) (fun _ _ _ => rfl) Unit P x]
      simp
    unfold CompiledTwo.εregTwo
    rw [sksLTwo_bconfTwo, Real.sqrt_zero, mul_zero, add_zero]
    exact h1
  have hcnt : 1 ≤ regCount FlockDeploy.sha512 (sksLTwo Bool) [] silentP := by
    have hval : regCount FlockDeploy.sha512 (sksLTwo Bool) [] silentP =
        (wrongAtBoundaryAt toyStmt toyPart toyTwo.part
          ((sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ())).card :=
      ite_eq_left inScopeTwo_toy
    rw [hval]
    exact Finset.card_pos.2 ⟨0, toy_wrongAtBoundaryAt (val := (sksLTwo Bool).regVal FlockDeploy.sha512 silentP [] ())
      (sksL_regVal _ silentP [] () 0)⟩
  have hund : ∀ w : Served Unit Unit Unit Unit, ¬ w.sampledIncorrect → 1 ≤ w.incorrect →
      Served.Undetected 1 [w] :=
    fun w hs hn => Or.inr (by simpa [Served.lead, hs] using hn)
  have hU : ∀ r : Fin ((sksLTwo Bool).V.k []) → Bool, Served.Undetected 1
      [twoRegRec FlockDeploy.sha512 (sksLTwo Bool) (regCount FlockDeploy.sha512 (sksLTwo Bool) [] silentP)
        (m, ⟨[], silentP, r⟩)] := fun r =>
    hund _ (fun ⟨u, hu, _⟩ => absurd (show u ∈ ([] : List ℕ) from hu) (by simp)) hcnt
  have hv : Verity.Game.value (Served.Undetected 1)
      (twoRegRun FlockDeploy.sha512 (fun _ => sksLTwo Bool) 1 (twoStrat (sksLTwo Bool) m [] silentP)) = 1 := by
    rw [twoRegRun_twoStrat, Verity.Game.value_coin_ret]
    simp only [Verity.Game.prCoin, hU, if_true]
    exact Verity.Game.avg_const 1
  have h1 := h 1 (fun _ => 0) (twoStrat (sksLTwo Bool) m [] silentP) (fun _ _ _ => (hε _ _).le) 1
  rw [hv] at h1
  norm_num at h1

end WeightSecurity.SecurityProofs
