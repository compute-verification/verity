module

public import Proofs.Flock.Claims.Recursion.Deploy
public import Proofs.Flock.Claims.Recursion.DeployZK
public import Proofs.Flock.Claims.ZK.Layout
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-! The recursion's zero knowledge by sequential composition (`zkAt`, `zkmalRx`). -/

@[expose] public section

namespace Claims.ZKMal

open Claims

theorem interact_ne_nil (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), 0 < n → interact ask pm n hist ≠ []
  | 0, _, h => absurd h (lt_irrefl 0)
  | n + 1, hist, _ => by
      simp only [interact]
      split <;> simp

theorem recGoesOn_interact (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), 0 < n →
      Claims.Recursion.goesOn (interact ask pm n hist) = goesOn n (interact ask pm n hist)
  | 0, _, h => absurd h (lt_irrefl 0)
  | n + 1, hist, _ => by
      simp only [interact]
      cases ask (hist.map Prod.fst ++ [pm hist]) with
      | msg m =>
          simp only
          rw [goesOn_cons]
          rcases Nat.eq_zero_or_pos n with rfl | hn
          · simp [interact, Claims.Recursion.goesOn, goesOn]
          · rw [← recGoesOn_interact ask pm n _ hn]
            obtain ⟨p, τ, hτ⟩ := List.exists_cons_of_ne_nil (interact_ne_nil ask pm n (hist ++ [(pm hist, m)]) hn)
            rw [hτ]
            simp [Claims.Recursion.goesOn, List.getLast?_cons_cons]
      | acc => simp [Claims.Recursion.goesOn, goesOn, isMsg]
      | rej => simp [Claims.Recursion.goesOn, goesOn, isMsg]

end Claims.ZKMal

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open Claims.ZKMal (Layout ZKmalAt SeqTape iterSimG iterProver)

variable (IV : Claims.Verifier Coin) (b : Brings)

def kPre (x : Claims.Msg) : ℕ := IV.k (xIOf x) + kS x

noncomputable def preP {T : Type} (pre : T → b.ref.TI × (ℕ → Salt)) (w : Claims.Flock.Wit) : Claims.Prover T :=
  ⟨fun x hist t => if hist.length < IV.k (xIOf x) then (b.ref.PI w).next (xIOf x) hist (pre t).1 else
    commitThen (fun cs => .ret (stageMsg b.hs x (asRounds (hist.take (IV.k (xIOf x)))) cs))
      ((List.range (b.nC x)).map fun j =>
        ((b.ref.digests w (pre t).1 (hist.take (IV.k (xIOf x)))).getD j 0, (pre t).2 j)) []⟩

theorem refP_pre {T : Type} (pre : T → b.ref.TI × (ℕ → Salt)) (w : Claims.Flock.Wit) (x : Claims.Msg)
    (hist : List (Claims.Msg × Claims.Msg)) (ω : Tape b) (t : T) (hω : pre t = (ω.1, ω.2.1))
    (h : hist.length < kPre IV x) : (refP IV b w).next x hist ω = (preP IV b pre w).next x hist t := by
  unfold kPre at h
  by_cases h1 : hist.length < IV.k (xIOf x)
  · simp only [refP, preP, h1, ↓reduceIte, hω]
  · simp only [refP, preP, h1, h, ↓reduceIte, hω]

def refWit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit :=
  fun w p h s => b.ref.wit w p.1 h s

noncomputable def xsOf (x : Claims.Msg) (s : ℕ) (h : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  (vstar b.st x (asRounds (h.take (kPre IV x)))).getD s []

noncomputable def lay {T : Type} (pre : T → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit) :
    Layout Claims.Flock.Wit Claims.Flock.Wit T where
  k₀ := kPre IV
  n x s := (nums b.st x s).rounds + 1
  xs := xsOf IV b
  ws s w t x h := wit w (pre t) (h.take (IV.k (xIOf x))) s

noncomputable def parts (T : Type) : Layout Claims.Flock.Wit Claims.Flock.Wit T where
  k₀ := kPre IV
  n x s := (nums b.st x s).rounds + 1
  xs := xsOf IV b
  ws _ _ _ _ _ := ⟨[]⟩

theorem lay_start {T : Type} (pre : T → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit) :
    (lay IV b pre wit).start = (parts IV b T).start :=
  Layout.start_congr rfl rfl

noncomputable def refPM {T TS : Type} (pre : T → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover TS) (w : Claims.Flock.Wit) : Claims.Prover (T × (ℕ → TS)) :=
  ⟨fun x hist ω => ((lay IV b pre wit).prover (preP IV b pre) PS (nS x) w).next x hist ω⟩

noncomputable def kM (x : Claims.Msg) : ℕ :=
  kPre IV x + ((List.range (nS x)).map fun s => (nums b.st x s).rounds + 1).sum

theorem parts_start (T : Type) (x : Claims.Msg) :
    ∀ m, (parts IV b T).start x m = kPre IV x + ((List.range m).map fun s => (nums b.st x s).rounds + 1).sum
  | 0 => by simp [Layout.start, parts]
  | m + 1 => by
      rw [Layout.start, parts_start T x m, List.range_succ, List.map_append, List.sum_append]
      simp [parts, add_assoc]

theorem kM_eq (x : Claims.Msg) : kM IV b x = (V IV b.st).k x + nS x := by
  have e : ∀ m, ((List.range m).map fun s => (nums b.st x s).rounds + 1).sum =
      ((List.range m).map fun s => (nums b.st x s).rounds).sum + m := by
    intro m
    induction m with
    | zero => simp
    | succ m ih => simp only [List.range_succ, List.map_append, List.sum_append, ih]; simp; omega
  simp only [kM, kPre, V, e]
  omega

theorem parts_start_pos (T : Type) (x : Claims.Msg) (j : ℕ) : 0 < (parts IV b T).start x j := by
  rw [parts_start]
  simp [kPre, kS]

noncomputable def simM (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (S₀ : Claims.Simulator) (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) : Claims.Simulator where
  C x := (iterSimG T₀ 0 NB ΩB (Layout.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω))) S₀ (fun j x => (SO x j).C)
    (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) (nS x)).C x
  prog x := (iterSimG T₀ 0 NB ΩB (Layout.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω))) S₀ (fun j x => (SO x j).C)
    (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) (nS x)).prog x

theorem zkAt (H : Claims.Msg → Claims.Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (T₀ : SeqTape) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (RA RF : Claims.Rel Claims.Flock.Wit) (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg) (ε₀ g : ℝ) (eB : ℕ → ℝ)
    (h₀ : ZKmalAt H (kPre IV x) (preP IV b pre) (RA x) x (S₀.C x) (S₀.prog x) ε₀ 0)
    (hS : ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      ZKmalAt H ((nums b.st x j).rounds + 1) PS (RF (xsOf IV b x j u)) (xsOf IV b x j u) (SO x j).C
        ((SO x j).prog (xsOf IV b x j u)) (eB j) g)
    (hW : ∀ w, RA x w → ∀ j < nS x, ∀ (t₀ : T₀.N × T₀.Ω) (u : List (Claims.Msg × Claims.Msg)),
      u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
        RF (xsOf IV b x j u) (wit w (pre t₀) (u.take (IV.k (xIOf x))) j))
    (he0 : ∀ j < nS x, 0 ≤ eB j) (hg : 0 ≤ g) :
    ZKmalAt H (kM IV b x) (fun w => (refPM IV b pre wit PS w).comap (Layout.flatten T₀ 0 NB ΩB (nS x))) (RA x) x
      ((simM IV b T₀ NB ΩB S₀ SO).C x) ((simM IV b T₀ NB ΩB S₀ SO).prog x)
      (ε₀ + ∑ j ∈ Finset.range (nS x), eB j) (nS x * g) := by
  have hst := lay_start IV b pre wit
  have h := Claims.ZKMal.zkmalAt_layout T₀ 0 (lay IV b pre wit) H x (nS x) (preP IV b pre) PS RA RF S₀
    (fun j x => (SO x j).C) (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn)
    ε₀ 0 g eB h₀ (fun j hj u hu => hS j hj u (by
      have hl := Layout.reach_length T₀ 0 NB ΩB (lay IV b pre wit) H _ RA x hu
      rwa [hst] at hl))
    (fun j hj => Layout.relOnAt_of T₀ 0 NB ΩB (lay IV b pre wit) H _ RA RF x fun w hw t₀ u hu =>
      hW w hw j hj t₀ u (by rwa [hst] at hu))
    (fun j _ w _ Θ Vs θ ωA => by
      rw [Claims.ZKMal.runStar_msgs]
      exact Claims.ZKMal.recGoesOn_interact _ _ _ _ (by
        show 0 < (lay IV b pre wit).start x j
        rw [hst]
        exact parts_start_pos IV b _ x j))
    (by simp) he0 hg
  rw [Layout.iterSimG_seq_congr T₀ 0 NB ΩB (lay IV b pre wit) (parts IV b _) rfl rfl rfl, hst, parts_start] at h
  simp only [Nat.cast_zero, zero_add] at h
  exact h.of_equiv _ (Equiv.refl _) (Equiv.refl _) fun _ _ _ => rfl

theorem zkmalRx (H : Claims.Msg → Claims.Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (T₀ : SeqTape) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (RA RF : Claims.Rel Claims.Flock.Wit) (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (ε₀ g : Claims.Msg → ℝ) (eB : ℕ → Claims.Msg → ℝ)
    (h₀ : ∀ x, ZKmalAt H (kPre IV x) (preP IV b pre) (RA x) x (S₀.C x) (S₀.prog x) (ε₀ x) 0)
    (hS : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      ZKmalAt H ((nums b.st x j).rounds + 1) PS (RF (xsOf IV b x j u)) (xsOf IV b x j u) (SO x j).C
        ((SO x j).prog (xsOf IV b x j u)) (eB j x) (g x))
    (hW : ∀ x w, RA x w → ∀ j < nS x, ∀ (t₀ : T₀.N × T₀.Ω) (u : List (Claims.Msg × Claims.Msg)),
      u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
        RF (xsOf IV b x j u) (wit w (pre t₀) (u.take (IV.k (xIOf x))) j))
    (he0 : ∀ j x, 0 ≤ eB j x) (hg : ∀ x, 0 ≤ g x) :
    Claims.ZKmalRx (fun x => (T₀.iter 0 NB ΩB (nS x)).N) (fun x => (T₀.iter 0 NB ΩB (nS x)).Ω) H (kM IV b)
      (fun w x => (refPM IV b pre wit PS w).comap (Layout.flatten T₀ 0 NB ΩB (nS x))) RA (simM IV b T₀ NB ΩB S₀ SO)
      (fun x => ε₀ x + ∑ j ∈ Finset.range (nS x), eB j x) (fun x => nS x * g x) := fun x =>
  zkAt IV b H T₀ pre wit PS RA RF S₀ SO x (ε₀ x) (g x) (fun j => eB j x) (h₀ x) (hS x) (hW x) (fun j _ => he0 j x)
    (hg x)

end RecDeploy
