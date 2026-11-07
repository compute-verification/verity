module

public import Proofs.Flock.Claims.ZK.Server
public import Definitions.Claims.ZKOut
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
import Proofs.Flock.Claims.Prob

/-!
# The deferred server per outcome

`ZkServer.zk_deferred` bounds the deferred server's transcript within the factor `2` of `leaky1`, the simulator that
guesses whether the session stopped. Where the session stopped is public: the server defers every stop to the session's
last round, so it is one bit of the transcript (`stopOf`), and with the verdict (`verdictOf`) it is the session's
outcome (`outD`). Given the outcome the simulator guesses nothing: it is `leaky1` at the outcome's bit (`simOutD`), and
the server is zero knowledge per outcome at the factor `1` (`ZkServer.zkOut_deferred`, `Claims.ZKOut`'s form at an
outcome).

* `stopOf`, `verdictOf`, `outD`: where the session stopped, its verdict, its outcome.
* `interact_stopAt_self`, `interact_stopAt_last`, `interact_stopAt_last_ne`, `cutT_stopOf`: a run cut at its last
  round or not at all is the run cut where `stopOf` says it stopped.
* `simOutD`, `simOutD_out`: `leaky1` given the outcome.
* `ZkServer.zkOut_deferred`: the deferred server per outcome.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-- **Where the deferred server stopped** in a transcript `τ` of `n` rounds: whether round `n - 1`, the session's last,
carries the stop. -/
def stopOf (stop : Msg) (n : ℕ) (τ : List (Msg × Out)) : Bool :=
  decide ((τ[n - 1]?).map Prod.fst = some stop)

/-- **The verdict a transcript ends on**: `some true` at an accept, `some false` at a reject, `none` where it ends on a
message or is empty. -/
def verdictOf (τ : List (Msg × Out)) : Option Bool :=
  match τ.getLast? with
  | some (_, .acc) => some true
  | some (_, .rej) => some false
  | _ => none

/-- **The deferred server's outcome** at `x`, of its `k x` rounds: the verdict, and whether the session stopped at its
last round. -/
def outD (stop : Msg) (k : Msg → ℕ) (x : Msg) (τ : List (Msg × Out)) : Option Bool × Bool :=
  (verdictOf τ, stopOf stop (k x) τ)

/-- A run stopped at its length is the run. -/
theorem interact_stopAt_self (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (n : ℕ) :
    interact ask (stopAt pm n stop) n [] = interact ask pm n [] :=
  interact_congr ask pm (stopAt pm n stop) n [] fun i hi => stopAt_of_lt (by
    rw [List.nil_append, length_histOf_take]
    have := interact_length_le ask pm n []
    omega)

/-- A run whose last round carries the stop is the run stopped at its last round. -/
theorem interact_stopAt_last (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (n : ℕ)
    (hs : ((interact ask pm n [])[n - 1]?).map Prod.fst = some stop) :
    interact ask (stopAt pm (n - 1) stop) n [] = interact ask pm n [] := by
  refine interact_congr ask pm (stopAt pm (n - 1) stop) n [] fun i hi => ?_
  have hl : ([] ++ histOf ((interact ask pm n []).take i)).length = i := by
    rw [List.nil_append, length_histOf_take]; omega
  by_cases hin : i < n - 1
  · exact stopAt_of_lt (by rw [hl]; exact hin)
  · have hn := interact_length_le ask pm n []
    have hie : i = n - 1 := by omega
    rw [stopAt_of_ge (h := [] ++ histOf ((interact ask pm n []).take i)) (by rw [hl]; exact hin)]
    have hg := interact_getElem_fst ask pm n [] i hi
    rw [← hie] at hs
    rw [hs] at hg
    exact Option.some.inj hg

/-- A run stopped at its last round whose last round doesn't carry the stop ended before it, so it is the run. -/
theorem interact_stopAt_last_ne (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (n : ℕ)
    (hs : ¬ ((interact ask (stopAt pm (n - 1) stop) n [])[n - 1]?).map Prod.fst = some stop) :
    interact ask pm n [] = interact ask (stopAt pm (n - 1) stop) n [] := by
  refine interact_congr ask (stopAt pm (n - 1) stop) pm n [] fun i hi => ?_
  have hl : ([] ++ histOf ((interact ask (stopAt pm (n - 1) stop) n []).take i)).length = i := by
    rw [List.nil_append, length_histOf_take]; omega
  by_cases hin : i < n - 1
  · exact (stopAt_of_lt (by rw [hl]; exact hin)).symm
  · exfalso
    have hn := interact_length_le ask (stopAt pm (n - 1) stop) n []
    have hie : i = n - 1 := by omega
    have hg := interact_getElem_fst ask (stopAt pm (n - 1) stop) n [] i hi
    rw [stopAt_of_ge (h := [] ++ histOf ((interact ask (stopAt pm (n - 1) stop) n []).take i))
      (by rw [hl]; exact hin), hie] at hg
    exact hs hg

/-- **A run cut at its last round or not at all is the run cut where it stopped**: the run of `pm` cut at `n - c` is
that run cut at `n - b`, with `b` whether the cut run's last round carries the stop (`stopOf`). -/
theorem cutT_stopOf (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (n : ℕ) (c : Fin 2) :
    cutT ask stop (n - c) n (interact ask pm n []) =
      cutT ask stop (n - (stopOf stop n (cutT ask stop (n - c) n (interact ask pm n []))).toNat) n
        (interact ask pm n []) := by
  rw [← interact_stopAt, ← interact_stopAt]
  have hA := interact_stopAt_self ask pm stop n
  rcases (by have := c.isLt; omega : (c : ℕ) = 0 ∨ (c : ℕ) = 1) with h0 | h1
  · rw [h0, Nat.sub_zero, hA]
    cases hb : stopOf stop n (interact ask pm n []) with
    | false => rw [Bool.toNat_false, Nat.sub_zero, hA]
    | true =>
      rw [Bool.toNat_true, interact_stopAt_last ask pm stop n (by unfold stopOf at hb; exact of_decide_eq_true hb)]
  · rw [h1]
    cases hb : stopOf stop n (interact ask (stopAt pm (n - 1) stop) n []) with
    | true => rw [Bool.toNat_true]
    | false =>
      rw [Bool.toNat_false, Nat.sub_zero, hA]
      exact (interact_stopAt_last_ne ask pm stop n (by unfold stopOf at hb; exact of_decide_eq_false hb)).symm

/-- **`leaky1` given the outcome**: `Sim`'s transcript cut where the outcome says the session stopped, at its last
round or not at all. -/
def simOutD (Sim : Simulator) (k : Msg → ℕ) (stop : Msg) : SimOut (Option Bool × Bool) where
  C := Sim.C
  prog x o c := (leaky1 Sim k stop).prog x ((⟨o.2.toNat, Bool.toNat_lt _⟩ : Fin 2), c)

theorem simOutD_out {Θ : Type} (H : Msg → Msg) (Sim : Simulator) (k : Msg → ℕ) (stop : Msg) (Vs : VStar Θ) (θ : Θ)
    (x : Msg) (o : Option Bool × Bool) (c : Sim.C x) :
    ((simOutD Sim k stop).given o).out H Vs θ x c =
      cutT (fun as => ((Vs.next x as θ).eval H).1) stop (k x - o.2.toNat) (k x) (Sim.out H Vs θ x c) :=
  leaky1_out H Sim k stop Vs θ x ((⟨o.2.toNat, Bool.toNat_lt _⟩ : Fin 2), c)

namespace ZkServer

/-- **The deferred server per outcome**: under `zk_deferred`'s premises, at each outcome `o` (`outD`: the verdict and
whether the session stopped at its last round), the server's transcript shows `o` and an event with probability at most
the simulator's given `o` (`simOutD`, `leaky1` at `o`'s bit) of the event, plus `ε_zk + γ·Fork`: `Claims.ZKOut`'s form
at an outcome, at the factor `1`. The released transcript is `P`'s run cut at its last round or not at all
(`split_deferred`), and `o`'s bit says which (`cutT_stopOf`), so the event at that cut is one `P` meets `ZKmalR` at. -/
theorem zkOut_deferred {W N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (S : ZkServer W (N × Ω))
    (fill : Msg → ℕ → ℕ → W → N × Ω → List (Msg × Msg) → S.Val) (H : Msg → Msg) (k : Msg → ℕ)
    (P : W → Prover (N × Ω)) (Rl : Rel W) (Sim : Simulator) (εzk γ : Msg → ℝ) (hZK : ZKmalR H k P Rl Sim εzk γ)
    (x : Msg) (w : W) (hR : Rl x w) (wk : S.Worker) (hRef : S.RefDOwn H P fill k wk x w) (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = k x) (o : Option Bool × Bool) (Ev : List (Msg × Out) → Prop) :
    Pr (fun ω : (N × Ω) × S.Chk =>
        outD S.stop k x (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs = o ∧
          Ev (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs) ≤
      Pr (fun c : Sim.C x => Ev (((simOutD Sim k S.stop).given o).out H Vs θ x c)) +
        (εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) := by
  classical
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  set cut : List (Msg × Out) → List (Msg × Out) := cutT ask S.stop (k x - o.2.toNat) (k x) with hcut
  have hsrv : ∀ ω : (N × Ω) × S.Chk, (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs =
      interact ask (S.nextD fill k wk x w ω.1 ω.2) (k x) [] := fun ω => by
    rw [runStar_msgs, hk]; rfl
  have hP : ∀ ρ : N × Ω, (runStar H (P w) Vs x ρ θ).msgs =
      interact ask (fun h => (((P w).next x h ρ).eval H).1) (k x) [] := fun ρ => by
    rw [runStar_msgs, hk]
  have hcutP : ∀ ω : (N × Ω) × S.Chk, outD S.stop k x (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs = o →
      (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs = cut (runStar H (P w) Vs x ω.1 θ).msgs := by
    intro ω ho
    obtain ⟨c, hc⟩ := S.split_deferred H P fill k x w wk hRef ask ω.1 ω.2
    have hb : stopOf S.stop (k x) (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs = o.2 := congrArg Prod.snd ho
    rw [hsrv, hc] at hb ⊢
    rw [hP, hcut, ← hb]
    exact cutT_stopOf ask _ S.stop (k x) c
  have hm : Pr (fun c : Sim.C x => outD S.stop k x (cut (Sim.out H Vs θ x c)) = o ∧ Ev (cut (Sim.out H Vs θ x c))) ≤
      Pr (fun c : Sim.C x => Ev (((simOutD Sim k S.stop).given o).out H Vs θ x c)) :=
    Pr_mono fun c hc => by rw [simOutD_out]; exact hc.2
  calc Pr (fun ω : (N × Ω) × S.Chk => outD S.stop k x (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs = o ∧
          Ev (runStar H (S.proverD fill k wk w) Vs x ω θ).msgs)
      ≤ Pr (fun ω : (N × Ω) × S.Chk => outD S.stop k x (cut (runStar H (P w) Vs x ω.1 θ).msgs) = o ∧
          Ev (cut (runStar H (P w) Vs x ω.1 θ).msgs)) :=
        Pr_mono fun ω hω => by rw [← hcutP ω hω.1]; exact hω
    _ = Pr (fun ρ : N × Ω => outD S.stop k x (cut (runStar H (P w) Vs x ρ θ).msgs) = o ∧
          Ev (cut (runStar H (P w) Vs x ρ θ).msgs)) :=
        Pr_fst (fun ρ : N × Ω => outD S.stop k x (cut (runStar H (P w) Vs x ρ θ).msgs) = o ∧
          Ev (cut (runStar H (P w) Vs x ρ θ).msgs))
    _ ≤ Pr (fun c : Sim.C x => outD S.stop k x (cut (Sim.out H Vs θ x c)) = o ∧ Ev (cut (Sim.out H Vs θ x c))) +
          (εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) := by
        have h := hZK x w hR Θ Vs θ hk fun τ => outD S.stop k x (cut τ) = o ∧ Ev (cut τ)
        linarith [le_abs_self (Pr (fun ρ : N × Ω => outD S.stop k x (cut (runStar H (P w) Vs x ρ θ).msgs) = o ∧
          Ev (cut (runStar H (P w) Vs x ρ θ).msgs)) -
          Pr (fun c : Sim.C x => outD S.stop k x (cut (Sim.out H Vs θ x c)) = o ∧ Ev (cut (Sim.out H Vs θ x c))))]
    _ ≤ _ := by linarith

end ZkServer

end Claims.ZKServer
