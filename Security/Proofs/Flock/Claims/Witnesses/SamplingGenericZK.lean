module

public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.Claims.ZK.Compose
public import Proofs.Flock.Claims.ZK.Split
public import Proofs.Flock.Claims.ZKOut
public import Proofs.Flock.Masked
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob

/-!
# Refutation witnesses: the generic sampled compiler's zero knowledge

The proof system's zero knowledge (`ZkArg.ZK`) and a masked compiled verifier's (`ZkCompiled.ZK`) each fail at a toy
instance (`Claims.ZKOut`'s own, `Claims.ZKOut_fails`, is beside `Claims.zkOutLeak`).

* `ZKOut` fails at a prover that talks (`talkP`) against a verifier that accepts everything, with a silent simulator and
  no error: the honest transcript is never empty and the simulator's always is (`talk_leaks`, through
  `Claims.zkOutLeak`): `zkOut_talk_fails`.
* `talkArg`, a proof system whose sessions run `talkP` and whose simulator is silent, is that instance:
  `ZkArg.ZK_fails`.
* `talkL`, a masked compiled verifier of one session over `talkArg` under a commitment whose every opening passes, so
  the toy witness is in the sampling relation: its composed simulator says nothing while the prover talks, at an error
  of `0`: `ZkCompiled.ZK_fails`.

The lock's `meets` witnesses: the deployed records are masked (`Claims.Flock.masked_meets_zk`,
`Claims.Sampling.masked_meets_zk`), and `echo_meets`: `CompilerZK`'s antecedents hold together at a nonempty relation.
`echoArg` is `talkArg` with a simulator that asks the verifier its answer to the talking prover's `[0]` (`echoSimOut`),
so it is zero knowledge at error `0` with every witness in its relation (`echoArg_zk`); `echoL`, `talkL`'s one-session
masked compiled verifier over it, meets the zero-knowledge contract, and the toy witness is in its sampling relation.

At Flock, `zkA_zk`: Flock's deployed masked session as a proof system with a public outcome (`zkA`) is zero
knowledge, by `MaskedZK` through `ZKmalR`'s two-sided bound (`zkOut_of_zkmalR`). `zkFlock_meets` puts it under a masked
compiled verifier of one session (`zkL`) that meets the zero-knowledge contract, but only vacuously in its `witness`
field: `zkL`'s commitment opens nothing, so its sampling relation is empty. A nonempty one would need the contract's
`witness` there, a member of Flock's relation (`Flock.Readable.relation`) at each statement the run admits, so a
statement the deployed parser accepts and a witness its tables hold of, and the tree constructs none.
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Recursion (spBind spUnder)
open Claims.Witness.Sampling (toyStmt toyPart toyWit toy_faulty toyWit_val accCm rejCm rejCm_faulty)

/-! ## `ZKOut` fails at a talking prover -/

/-- A prover that says `[0]`. -/
def talkP : Prover (Unit × Unit) := ⟨fun _ _ _ => .ret [0]⟩

/-- **A verifier that accepts everything**, after one round. -/
def acceptAllVs : VStar Unit := ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩

/-- A simulator that outputs the empty transcript. -/
def silentSimOut : SimOut Unit where
  C _ := Unit
  prog _ _ _ := .ret []

/-- **The talking prover leaks**: against the verifier that accepts everything its transcript is never empty, the
silent simulator's always is, and the error is `0`. -/
theorem talk_leaks (H : Msg → Msg) :
    ¬ Pr (fun ω : Unit × Unit => (runStar H talkP acceptAllVs [] ω ()).msgs ≠ []) ≤
        (Fintype.card Unit : ℝ) * Pr (fun p : Unit × silentSimOut.C [] =>
          (silentSimOut.given p.1).out H acceptAllVs () [] p.2 ≠ []) + ∑ _o : Unit, (0 : ℝ) := by
  intro h
  have h1 : Pr (fun ω : Unit × Unit => (runStar H talkP acceptAllVs [] ω ()).msgs ≠ []) = 1 := by
    rw [Pr_congr fun ω => iff_true_intro (by simp [runStar, stepWith, acceptAllVs, talkP, OC.eval])]
    exact Claims.ZKMal.Pr_true
  have h2 : Pr (fun p : Unit × silentSimOut.C [] =>
      (silentSimOut.given p.1).out H acceptAllVs () [] p.2 ≠ []) = 0 := by
    rw [Pr_congr fun p => iff_false_intro (by simp [Simulator.out, SimOut.given, silentSimOut, SP.eval])]
    exact Pr_false
  rw [h1, h2] at h
  norm_num at h

/-- **`ZKOut` fails at the talking prover**, its relation every witness, its outcome trivial, its simulator silent and
its error `0`: its leak form (`Claims.zkOutLeak`) would bound the talking prover's leak. -/
theorem zkOut_talk_fails (H : Msg → Msg) :
    ¬ ZKOut (fun _ => Unit) (fun _ => Unit) H (fun _ => 1) (fun _ _ => talkP) (fun _ (_ : Unit) => True)
      (fun _ _ => ()) silentSimOut (fun _ _ _ _ _ _ => 0) := fun hz =>
  talk_leaks H (Claims.zkOutLeak (fun _ => Unit) (fun _ => Unit) H (fun _ => 1) (fun _ _ => talkP)
    (fun _ (_ : Unit) => True) (fun _ _ => ()) silentSimOut (fun _ _ _ _ _ _ => 0) hz [] () trivial Unit acceptAllVs ()
    rfl (· ≠ []))

/-! ## `ZkArg.ZK` and `ZkCompiled.ZK` fail at the talking prover -/

/-- **A proof system whose prover talks and whose simulator is silent**: `toyA` with every witness in its relation,
one round per session, the talking prover, the trivial outcome, the silent simulator and error `0`. -/
def talkArg (Coin : Type) : ZkArg Coin where
  toArg := { toyA Coin with R := fun _ _ => True }
  kz _ := 1
  Nonce := Unit
  Rest := Unit
  P _ := talkP
  St := Unit
  out _ _ := ()
  S := silentSimOut
  εzk _ _ _ _ _ _ := 0

/-- The router of one session from the first round, at the empty statement. -/
def talkStep (_ : Msg) (hist : List (Msg × Msg)) : Step := .session 0 [] hist

/-- **A masked compiled verifier of one session over `talkArg`**: the toy statement, every opening passing (`accCm`),
the verifier that accepts after its one round, the session's, and the reference prover the composed one. -/
def talkL (Coin : Type) : ZkCompiled Coin (talkArg Coin) where
  V := accV Coin
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := accCm
  stage2 _ := Finset.univ
  checked _ _ := [0]
  sh _ _ := ()
  κb _ _ := 0
  γb _ _ := 0
  rounds _ _ := Finset.univ
  session x _ v := some (x, v)
  iroots _ _ _ := []
  reveal _ _ _ _ := []
  openCalls _ := 0
  nOut _ := 1
  nIn _ := 1
  Tape := (Fin 1 → Unit) × (Fin 1 → Unit)
  P _ := ⟨fun _ _ _ => .ret [0]⟩
  δc _ _ := 1
  step := talkStep
  k₀ _ := 0
  M _ := 1
  n _ _ := 1
  wit _ _ _ := .ret ()
  CB _ _ := Unit
  tapeOf _ t := t

section Talk

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- `talkL` meets the zero-knowledge contract: its one session is the run, and every witness is in `talkArg`'s
relation. -/
theorem talkL_sessions (H : Msg → Msg) : (talkL Coin).ZkSessions H where
  rounds _ _ _ := by simp [ZkCompiled.start, talkL, accV]
  routed _ _ _ := ⟨fun _ h => absurd h (Nat.not_lt_zero _), fun j hj hist _ _ => by
    obtain rfl : j = 0 := Nat.lt_one_iff.1 hj
    simp [ZkCompiled.start, talkL, talkStep, Step.stmt]⟩
  shaped _ _ _ _ _ _ _ := ⟨rfl, ⟨Equiv.refl _⟩⟩
  witness _ _ _ _ _ _ _ := trivial
  honest _ _ _ _ _ _ _ := rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- The toy witness opens the one input under `accCm`, so no proof block is incorrect. -/
theorem talkL_rhon (H : Msg → Msg) :
    Claims.Sampling.Rhon H (talkL Coin).commit (talkL Coin).parse (talkL Coin).partition [] toyWit :=
  ⟨toyStmt, toyPart, rfl, rfl, by
    show Claims.Sampling.faulty H accCm toyStmt toyPart toyWit = ∅
    simp [toy_faulty, toyWit_val]⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem talkL_msgs (H : Msg → Msg) (w : Claims.Sampling.Wit)
    (t : (Fin ((talkL Coin).M []) → (talkArg Coin).Nonce) × (Fin ((talkL Coin).M []) → (talkArg Coin).Rest)) :
    (runStar H ((talkL Coin).prover w []) acceptAllVs [] t ()).msgs ≠ [] := by
  simp [runStar, stepWith, acceptAllVs, ZkCompiled.prover, talkL, Prover.comap, OC.eval]

theorem simAt_silent (C : Type) (xS : Msg) (o : Unit) (c : C) : simAt silentSimOut C xS o c = .ret [] := by
  unfold simAt
  split <;> rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- The composed simulator at `talkL` says nothing. -/
theorem talkL_simOut (H : Msg → Msg) (o : List Msg × List Unit) (c : (talkL Coin).sim.C []) :
    ((talkL Coin).sim.given o).out H acceptAllVs () [] c = [] := by
  obtain ⟨l, _ | ⟨u, os⟩⟩ := o
  · rfl
  · have h : 0 < (talkL Coin).M [] := Nat.one_pos
    show ((spBind (spUnder [] (simAt silentSimOut ((talkL Coin).CB [] (⟨0, h⟩ : Fin ((talkL Coin).M [])).val) [] u
      (c ⟨0, h⟩))) fun τ =>
      (talkL Coin).simSessions [] (u :: os) c [] ([] ++ τ)).eval H _).1 = []
    rw [simAt_silent]
    rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem talkL_err (H : Msg → Msg) (w : Claims.Sampling.Wit) (o : List Msg × List (talkArg Coin).St) :
    (talkL Coin).err H [] w Unit acceptAllVs () o = 0 := by
  unfold ZkCompiled.err
  refine Finset.sum_eq_zero fun j _ => (congrArg _ (funext fun t => ?_)).trans (Ex_const (0 : ℝ))
  dsimp only
  split <;> rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`ZkCompiled.ZK` fails at `talkL`**: at the toy witness, which no proof block faults, the talking prover's
transcript is never empty, the composed simulator's always is, and the error is `0`. -/
theorem zkCompiled_talk_fails (H : Msg → Msg) : ¬ (talkL Coin).ZK H := by
  intro h
  have h1 := h (fun x => Claims.Sampling.leafLayer H (talkL Coin).commit (talkL Coin).parse x toyWit) [] toyWit
    ⟨talkL_rhon _, rfl⟩ Unit acceptAllVs () rfl
    (Claims.Sampling.leafLayer H (talkL Coin).commit (talkL Coin).parse [] toyWit, [()]) (· ≠ [])
  rw [Pr_congr fun t => iff_true_intro (by exact ⟨rfl, talkL_msgs _ _ t⟩), Claims.ZKMal.Pr_true,
    Pr_congr fun c => iff_false_intro (by exact not_not_intro (talkL_simOut _ _ c)), Pr_false, talkL_err,
    add_zero] at h1
  norm_num at h1

end Talk

/-! ## `CompilerZK`'s antecedents hold where a witness is in the sampling relation -/

/-- **A simulator that echoes the talking prover**: it asks the verifier its answer to `[0]` and outputs that round,
the talking prover's transcript against every verifier of one round. -/
def echoSimOut : SimOut Unit where
  C _ := Unit
  prog _ _ _ := .ask [[0]] fun b => .ret [([0], b)]

/-- Against a verifier of one round, the talking prover's transcript is its `[0]` and the verifier's answer. -/
theorem talk_msgs (H : Msg → Msg) {Θ : Type} (Vs : VStar Θ) (x : Msg) (θ : Θ) (hk : Vs.k x = 1)
    (ω : Unit × Unit) : (runStar H talkP Vs x ω θ).msgs = [([0], ((Vs.next x [[0]] θ).eval H).1)] := by
  unfold runStar
  rw [hk]
  cases hb : ((Vs.next x [[0]] θ).eval H).1 <;> simp [stepWith, talkP, OC.eval, hb]

/-- **The echoing proof system**: `talkArg` with the simulator that echoes the talking prover. -/
def echoArg (Coin : Type) : ZkArg Coin := { talkArg Coin with S := echoSimOut }

/-- **The echoing proof system is zero knowledge** at error `0`, at every witness, all in its relation: the echoing
simulator's transcript is the talking prover's. -/
theorem echoArg_zk {Coin : Type} (H : Msg → Msg) : (echoArg Coin).ZK H := by
  intro x w _ Θ Vs θ hk o Ev
  change Pr (fun ω : Unit × Unit => () = o ∧ Ev (runStar H talkP Vs x ω θ).msgs) ≤
    Pr (fun c : Unit => Ev ((echoSimOut.given o).out H Vs θ x c)) + 0
  rw [add_zero]
  by_cases hE : Ev [([0], ((Vs.next x [[0]] θ).eval H).1)]
  · refine (Pr_le_one _).trans (le_of_eq ?_)
    exact ((Pr_congr fun c : Unit => iff_true_intro (show Ev ((echoSimOut.given o).out H Vs θ x c) from hE)).trans
      Claims.ZKMal.Pr_true).symm
  · refine (le_of_eq ?_).trans (Pr_nonneg _)
    exact (Pr_congr fun ω : Unit × Unit => iff_false_intro fun h : () = o ∧ Ev (runStar H talkP Vs x ω θ).msgs =>
      hE (talk_msgs H Vs x θ hk ω ▸ h.2)).trans Pr_false

/-- **A masked compiled verifier of one session over the echoing proof system**: `talkL`'s parts. -/
def echoL (Coin : Type) : ZkCompiled Coin (echoArg Coin) := { talkL Coin with }

section Echo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- `echoL` meets the zero-knowledge contract, as `talkL` does: the contract reads the proof system's simulator only
through its coins, `Unit` at both. -/
theorem echoL_sessions (H : Msg → Msg) : (echoL Coin).ZkSessions H :=
  have h := talkL_sessions (Coin := Coin) H
  ⟨h.rounds, h.routed, h.shaped, h.witness, h.honest⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`CompilerZK`'s antecedents hold together at a nonempty relation**: the echoing proof system is zero knowledge, the
one-session masked compiled verifier over it meets the zero-knowledge contract, and the toy witness, which no proof
block faults, is in its sampling relation, so the contract's `witness` field holds of a witness there. -/
theorem echo_meets (H : Msg → Msg) :
    (echoArg Coin).ZK H ∧ (echoL Coin).ZkSessions H ∧
      Claims.Sampling.Rhon H (echoL Coin).commit (echoL Coin).parse (echoL Coin).partition [] toyWit :=
  ⟨echoArg_zk H, echoL_sessions H, talkL_rhon (Coin := Coin) H⟩

end Echo

/-! ## Flock's masked session meets the zero-knowledge contract -/

/-- A simulator, as one given a trivial outcome. -/
def oneOut (S : Simulator) : SimOut Unit where
  C := S.C
  prog x _ c := S.prog x c

/-- **`ZKmalR` gives `ZKOut`** with a trivial outcome: the honest view's probability of an event is at most the
simulator's plus the error, the two-sided bound's one side. -/
theorem zkOut_of_zkmalR {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {k : Msg → ℕ} {P : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ}
    (h : ZKmalR H k P R S εzk γ) :
    ZKOut (fun _ => N) (fun _ => Ω) H k (fun w _ => P w) R (fun _ _ => ()) (oneOut S)
      (fun x w _ Vs θ _ => εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) := by
  intro x w hR _ Vs θ hk o Ev
  have h1 := sub_le_iff_le_add'.1 ((le_abs_self _).trans (h x w hR _ Vs θ hk Ev))
  refine (le_of_eq (Pr_congr fun ω => ?_)).trans h1
  exact and_iff_right rfl

open Claims.ZKMal.Masked FlockSoundness.Discharge in
/-- **Flock's deployed masked session** (`Flock.Guarantees.MaskedZK`'s record), at one coin per rep round and
`t = 1`. -/
noncomputable def zkFlock :=
  Claims.ZKMal.zkDeployed _root_.Flock.Readable.relation (fun _ _ => 1)
    (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE) 1

open Claims.ZKMal.Masked FlockSoundness.Discharge in
open Classical in
/-- `MaskedZK`'s zero-knowledge error at `t`. -/
noncomputable def zkEps (t : ℕ) (x : Msg) : ℝ :=
  if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
    8 * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
      8 * t * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1

/-- `MaskedZK`'s fork factor at `t`, `4·t·3^t`. -/
noncomputable def zkγ (t : ℕ) (_ : Msg) : ℝ := 4 * t * 3 ^ t

theorem zkFlock_zk : ZKmalR FlockDeploy.sha512 (fun x => zkFlock.V.k x + 1) zkFlock.prover
    (Claims.Flock.R zkFlock.parse) zkFlock.S (zkEps 1) (zkγ 1) :=
  (_root_.Flock.SecurityProofs.MaskedZK (fun _ _ => 1) 1 Nat.one_pos).2

/-- **Flock's deployed masked session as a proof system with a public outcome**: `Arg.ofFlock` at its record, its
sessions' rounds `V.k + 1` (`Hello` first), its outcome trivial, and its simulator and error `MaskedZK`'s. -/
noncomputable def zkA : ZkArg FlockDeploy.Coin where
  toArg := Arg.ofFlock zkFlock.toDeploy
  kz x := zkFlock.V.k x + 1
  Nonce := zkFlock.Nonce
  Rest := zkFlock.Rest
  P := zkFlock.prover
  St := Unit
  out _ _ := ()
  S := oneOut zkFlock.S
  εzk x w _ Vs θ _ := zkEps 1 x + zkγ 1 x * (Fork FlockDeploy.sha512 (zkFlock.prover w) Vs θ x).Pr FlockDeploy.sha512

/-- `zkA` is zero knowledge with a public outcome: `MaskedZK`, through `zkOut_of_zkmalR`. -/
theorem zkA_zk : zkA.ZK FlockDeploy.sha512 := zkOut_of_zkmalR zkFlock_zk

/-- The router of one session from the first round, at the run's statement. -/
def oneStep (x : Msg) (hist : List (Msg × Msg)) : Step := .session 0 x hist

/-- **A masked compiled verifier of one session over Flock's**: the toy statement, one session of Flock's masked
record at the run's statement from the first round, its rounds the session's, and the reference prover the composed
one. Its commitment opens nothing (`rejCm`), so no witness is honest there: the tree states no member of Flock's
relation, so the session's witness has none to be. -/
noncomputable def zkL : ZkCompiled FlockDeploy.Coin zkA where
  V := ⟨fun x => zkFlock.V.k x + 1, fun _ _ _ => .ret .rej⟩
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := rejCm
  stage2 _ := Finset.univ
  checked _ _ := [0]
  sh x _ := zkA.shape x
  κb x _ := zkA.κ x
  γb x _ := zkA.γ x
  rounds _ _ := Finset.univ
  session x _ v := some (x, v)
  iroots _ _ _ := []
  reveal _ _ _ _ := []
  openCalls _ := 0
  nOut _ := 1
  nIn _ := 1
  Tape := (Fin 1 → zkFlock.Nonce) × (Fin 1 → zkFlock.Rest)
  P _ := ⟨fun x hist t => (zkFlock.prover ⟨[]⟩).next x hist (t.1 ⟨0, Nat.one_pos⟩, t.2 ⟨0, Nat.one_pos⟩)⟩
  δc _ _ := 1
  step := oneStep
  k₀ _ := 0
  M _ := 1
  n x _ := zkFlock.V.k x + 1
  wit _ _ _ := .ret ⟨[]⟩
  CB x _ := zkFlock.S.C x
  tapeOf _ t := t

theorem rhon_rejCm (H : Msg → Msg) (x : Msg) (w : Claims.Sampling.Wit) :
    ¬ Claims.Sampling.Rhon H rejCm (fun _ => some toyStmt) (fun _ => some toyPart) x w := by
  rintro ⟨s, P, hs, hP, hf⟩
  obtain rfl := Option.some.inj hs
  obtain rfl := Option.some.inj hP
  rw [rejCm_faulty] at hf
  exact Finset.singleton_ne_empty 0 hf

theorem zkL_sessions (H : Msg → Msg) : zkL.ZkSessions H where
  rounds _ _ _ := by simp [ZkCompiled.start, zkL]
  routed _ _ _ := ⟨fun _ h => absurd h (Nat.not_lt_zero _), fun j hj hist _ _ => by
    obtain rfl : j = 0 := Nat.lt_one_iff.1 hj
    simp [ZkCompiled.start, zkL, oneStep, Step.stmt]⟩
  shaped _ _ _ _ _ _ _ := ⟨rfl, ⟨Equiv.refl _⟩⟩
  witness x w h _ _ _ _ := absurd h (rhon_rejCm H x w)
  honest _ _ _ _ _ _ _ := rfl

/-- **`CompilerZK`'s antecedents hold together** at Flock's deployed masked session: as a proof system with a public
outcome it is zero knowledge (`MaskedZK`), and the one-session masked compiled verifier over it meets the
zero-knowledge contract, its `witness` field vacuously (`rhon_rejCm`); `echo_meets` is the antecedents at a nonempty
relation. -/
theorem zkFlock_meets : zkA.ZK FlockDeploy.sha512 ∧ zkL.ZkSessions FlockDeploy.sha512 :=
  ⟨zkA_zk, zkL_sessions _⟩

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

/-- **`ZkArg.ZK` fails** at some proof system. -/
theorem ZkArg.ZK_fails {Coin : Type} (H : Msg → Msg) : ∃ A : ZkArg Coin, ¬ A.ZK H :=
  ⟨Witness.SamplingGeneric.talkArg Coin, Witness.SamplingGeneric.zkOut_talk_fails H⟩

/-- **`ZkCompiled.ZK` fails** at some masked compiled verifier. -/
theorem ZkCompiled.ZK_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : ZkArg Coin) (L : ZkCompiled Coin A), ¬ L.ZK H :=
  ⟨_, _, Witness.SamplingGeneric.zkCompiled_talk_fails H⟩

end Claims.SamplingGeneric
