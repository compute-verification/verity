module

public import Proofs.Flock.Claims.Recursion.Deploy
public import Proofs.Flock.Claims.Recursion.DeployZK
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
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.BitVec

/-! The recursion's deployed stop `stopsRec` and its record `deployRecStop`; whether the staged numbers read the coins
is left open, and the stop covers both cases. -/

@[expose] public section

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open scoped NNReal

variable {N Ω : Claims.Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
  [∀ x, Nonempty (Ω x)]

/-- **An inner zerocheck outer coin is `1`** in the transcript `ms`: at some outer coin slot of the inner statement
(`FlockComplete.outerSlots`), the verifier's answer is a coin's bytes whose element there is `1` (`FlockComplete.Bad`
on the answers). -/
def badAt (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : Prop :=
  ∃ p ∈ FlockComplete.outerSlots (xIOf x), ∃ c : Coin,
    (ms[p.1]?).map Prod.snd = some (Claims.Out.msg c.bytes) ∧ c.at p.2 = 1

/-- The rounds whose starts the stop is read at, and the one after: the inner rounds of `IV` and the staging round. -/
def stopRounds (IV : Claims.Verifier Coin) (x : Claims.Msg) : ℕ := IV.k (xIOf x) + kS x

/-- **An answer before the sessions is not a coin's bytes**: among the inner and staging rounds of `ms`, the verifier
answered one with something other than `.msg c.bytes`. -/
def uncoinedAt (IV : Claims.Verifier Coin) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : Prop :=
  ∃ p ∈ ms.take (stopRounds IV x), ∀ c : Coin, p.2 ≠ Claims.Out.msg c.bytes

/-- **Some part's staged numbers are not `nums`'** once `ms` has passed the inner and staging rounds: the compiled
verifier refuses that part's session (`ract`). -/
def mismatchAt (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    Prop :=
  stopRounds IV x ≤ ms.length ∧ ∃ s < nS x, numsOf (staged st x (ms.take (stopRounds IV x)) s) ≠ nums st x s

/-- **Some part's staged V\* statement is one the masked verifier can't accept** at SHA-512 once `ms` has passed the
inner and staging rounds: `readyO` at the run's own staged statement, not `pick`'s. -/
def unreadyAt (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    Prop :=
  stopRounds IV x ≤ ms.length ∧
    ∃ s < nS x, readyO FlockDeploy.sha512 (staged st x (ms.take (stopRounds IV x)) s) = false

open Classical in
/-- **The deployed stop** at the statement `x` and the transcript `ms`: out of scope or an inner statement the inner
verifier can't accept at SHA-512 (read off the statement), an answer before the sessions that is not a coin's bytes,
an inner zerocheck outer coin at `1`, or a part's staged V\* statement the masked verifier can't accept or whose
numbers are not the statement's. -/
noncomputable def stopsRec (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (st : Stage) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) : Bool :=
  !scopeCheck ivScope x || !FlockComplete.ready FlockDeploy.sha512 (xIOf x) || decide (uncoinedAt IV x ms) ||
    decide (badAt x ms) || decide (unreadyAt IV st x ms) || decide (mismatchAt IV st x ms)

/-- **The reference prover with the deployed stop**: `refP`, except that once `stopsRec` has held at the start of a
round up to `stopRounds` it sends the empty message. -/
noncomputable def refPStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (w : Claims.Flock.Wit) : Claims.Prover (Tape b) :=
  ⟨fun x hist ω =>
    if (List.range (min hist.length (stopRounds IV x) + 1)).any
        (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) then .ret []
    else (refP IV b w).next x hist ω⟩

/-- **The compiled verifier with its reference prover's stop**: `deployRec`, its reference prover `refPStop` and its
stop `stopsRec`, which reads no oracle; the reference prover's tape at each statement `x` sampled from the finite nonce
and rest `N x × Ω x` by `tapeOf x`, the outer simulators `SO`, and the inner rounds' commit strings `nAt` in the
layout `innerMsg`. -/
noncomputable def deployRecStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg) :
    Claims.Recursion.StopDeploy Coin :=
  { deployRec IV ivScope b with
    Tape := Tape b
    P := refPStop IV ivScope b
    Nonce := N
    Rest := Ω
    tapeOf := tapeOf
    SO := SO
    nAt := nAt
    innerMsg := innerMsg
    stops := fun _ => stopsRec IV ivScope b.st }

theorem prover_deployRecStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (w : Claims.Flock.Wit) (x : Claims.Msg) :
    (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).prover w x =
      ⟨fun x' hist ω => (refPStop IV ivScope b w).next x' hist (tapeOf x ω)⟩ :=
  rfl

theorem refPStop_halted (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg))
    (ω : Tape b)
    (h : (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (stopRounds IV x) x (asRounds hist) = true) :
    (refPStop IV ivScope b w).next x hist ω = .ret [] := by
  have hl : (asRounds hist).length = hist.length := List.length_map _
  have h' : (List.range (min hist.length (stopRounds IV x) + 1)).any
      (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) = true := by
    rw [← hl]; exact h
  simp only [refPStop, h', ↓reduceIte]

theorem refPStop_live (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg))
    (ω : Tape b)
    (h : (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (stopRounds IV x) x (asRounds hist) = false) :
    (refPStop IV ivScope b w).next x hist ω = (refP IV b w).next x hist ω := by
  have hl : (asRounds hist).length = hist.length := List.length_map _
  have h' : (List.range (min hist.length (stopRounds IV x) + 1)).any
      (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) = false := by
    rw [← hl]; exact h
  simp only [refPStop, h', Bool.false_eq_true, ↓reduceIte]

theorem halted_of_statement (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (n : ℕ) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out))
    (h : scopeCheck ivScope x = false ∨ FlockComplete.ready FlockDeploy.sha512 (xIOf x) = false) :
    (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H n x ms = true := by
  unfold Claims.Recursion.StopDeploy.halted
  refine List.any_eq_true.2 ⟨0, List.mem_range.2 (by omega), ?_⟩
  change stopsRec IV ivScope b.st x (ms.take 0) = true
  rcases h with h | h <;> simp [stopsRec, h]

/-- **Every answer of `ms` is a coin's bytes**, as the compiled verifier sends them. -/
def Coined (ms : List (Claims.Msg × Claims.Out)) : Prop := ∀ p ∈ ms, ∃ c : Coin, p.2 = Claims.Out.msg c.bytes

theorem not_uncoinedAt (IV : Claims.Verifier Coin) (x : Claims.Msg) {ms : List (Claims.Msg × Claims.Out)}
    (hc : Coined ms) : ¬ uncoinedAt IV x ms := fun ⟨p, hp, hn⟩ =>
  let ⟨c, hpc⟩ := hc p (List.mem_of_mem_take hp)
  hn c hpc

theorem Coined.take {ms : List (Claims.Msg × Claims.Out)} (hc : Coined ms) (i : ℕ) : Coined (ms.take i) :=
  fun p hp => hc p (List.mem_of_mem_take hp)

theorem stopsRec_live (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (st : Stage) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) (hs : scopeCheck ivScope x = true)
    (hr : FlockComplete.ready FlockDeploy.sha512 (xIOf x) = true) (hl : ms.length < stopRounds IV x)
    (hc : Coined ms) (hb : ¬ badAt x ms) : stopsRec IV ivScope st x ms = false := by
  have hu : ¬ unreadyAt IV st x ms := fun h => absurd h.1 (by omega)
  have hm : ¬ mismatchAt IV st x ms := fun h => absurd h.1 (by omega)
  simp [stopsRec, hs, hr, not_uncoinedAt IV x hc, hb, hu, hm]

theorem halted_live (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out))
    (hs : scopeCheck ivScope x = true) (hr : FlockComplete.ready FlockDeploy.sha512 (xIOf x) = true)
    (hl : ms.length < stopRounds IV x) (hc : Coined ms) (hb : ∀ i ≤ ms.length, ¬ badAt x (ms.take i)) :
    (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (stopRounds IV x) x ms = false := by
  unfold Claims.Recursion.StopDeploy.halted
  refine List.any_eq_false.2 fun i hi => ?_
  have hi' : i ≤ ms.length := by have := List.mem_range.1 hi; omega
  change ¬ stopsRec IV ivScope b.st x (ms.take i) = true
  rw [stopsRec_live IV ivScope b.st x (ms.take i) hs hr (by rw [List.length_take]; omega) (hc.take i) (hb i hi')]
  exact Bool.false_ne_true

theorem zkStop_deployRecStop_iff (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin)
    (ivScope : Claims.Msg → Bool) (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (relO : Readable.Parsed → Readable.Relation) (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b)
    (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg) :
    Claims.Recursion.ZKStop H (deployRecStop IV ivScope b tapeOf SO nAt innerMsg) I (deployZkRec relO) ↔
      ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
        Claims.Hm96Hiding (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).mask
            ((deployRecStop IV ivScope b tapeOf SO nAt innerMsg).toDeploy.saltHash H) δ₁ →
          Claims.ZKmalRx N Ω H (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).V.k
            (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).prover
            (Claims.Recursion.R (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).toDeploy I)
            (Claims.Recursion.SimStop (deployRecStop IV ivScope b tapeOf SO nAt innerMsg) I (SO t) H)
            (Claims.Recursion.εzkStop (deployRecStop IV ivScope b tapeOf SO nAt innerMsg) I t δ₁)
            (Claims.Recursion.γzk (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).toDeploy t) :=
  ⟨And.right, fun h => ⟨rfl, h⟩⟩

end RecDeploy
