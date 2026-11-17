module

public import Proofs.Flock.Claims.Complete.Runs
public import Proofs.Flock.Claims.Complete.Zk.Coins
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
import Proofs.Flock.Claims.Complete.Json.Goals
import Proofs.Flock.Claims.Prob

/-! The masked verifier's error at any number of tables (`ready`, `δc`), and completeness for the deployed prover from
acceptance at the good coins (`complete_of`). -/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock FlockDeploy

theorem next_msg {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {m : Claims.Msg} (h : ((FlockDeploy.next true x as cs).eval H).1 = .msg m) :
    ∃ c, cs.getLast? = some c ∧ m = c.bytes := by
  cases ha : act true x as cs with
  | rej =>
    rw [next_rej H ha] at h
    cases h
  | coin c =>
    rw [next_coin H ha] at h
    refine ⟨c, ?_, (Claims.Out.msg.inj h).symm⟩
    unfold act at ha
    repeat' split at ha
    all_goals first | (cases ha; done) | (cases ha; assumption)
  | decide s spec sts link c₁ mids f =>
    simp only [FlockDeploy.next, ha] at h
    unfold outOf at h
    rw [Claims.OC.eval_bind] at h
    simp only [Claims.OC.eval] at h
    split at h <;> simp at h

theorem stepWith_congr_coins (H : Claims.Msg → Claims.Msg) {Ω : Type} {P P' : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {k : ℕ} (r : Fin k → Coin)
    (hP : ∀ hist : List (Claims.Msg × Claims.Msg),
      hist.map Prod.snd = (List.range hist.length).map (fun j => (Claims.coinSeq r j).bytes) →
        P.next x hist ω = P'.next x hist ω) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Transcript),
      as.length = hist.length →
      hist.map Prod.snd = (List.range hist.length).map (fun j => (Claims.coinSeq r j).bytes) →
      Claims.stepWith H P x ω (fun as => (V true).next x as ((List.range as.length).map (Claims.coinSeq r))) n hist
          as t =
        Claims.stepWith H P' x ω (fun as => (V true).next x as ((List.range as.length).map (Claims.coinSeq r))) n
          hist as t
  | 0, _, _, _, _, _ => rfl
  | n + 1, hist, as, t, hl, hh => by
    simp only [Claims.stepWith]
    rw [hP hist hh]
    generalize hvb : ((V true).next x (as ++ [((P'.next x hist ω).eval H).1])
      ((List.range (as ++ [((P'.next x hist ω).eval H).1]).length).map (Claims.coinSeq r))).eval H = vb
    obtain ⟨b, l⟩ := vb
    cases b with
    | msg m =>
      have hm : ((FlockDeploy.next true x (as ++ [((P'.next x hist ω).eval H).1])
          ((List.range (as ++ [((P'.next x hist ω).eval H).1]).length).map (Claims.coinSeq r))).eval H).1 =
            .msg m := by
        change (((V true).next x _ _).eval H).1 = _
        rw [hvb]
      obtain ⟨c, hc, hmc⟩ := next_msg hm
      have hc' : c = Claims.coinSeq r hist.length := by
        rw [List.length_append, List.length_singleton, List.range_succ, List.map_append, List.map_singleton,
          List.getLast?_append, List.getLast?_singleton, Option.some_or, hl] at hc
        exact (Option.some.inj hc).symm
      have hm' : m = (Claims.coinSeq r hist.length).bytes := hmc.trans (congrArg Coin.bytes hc')
      refine stepWith_congr_coins H r hP n _ _ _ (by simp only [List.length_append, List.length_singleton, hl]) ?_
      rw [List.map_append, List.map_singleton, List.length_append, List.length_singleton, List.range_succ,
        List.map_append, List.map_singleton, hh]
      exact congrArg (fun b => _ ++ [b]) hm'
    | acc => rfl
    | rej => rfl

theorem outerHit_coins {x : Claims.Msg} {k : ℕ} {r : Fin k → Coin} (hr : ¬ BadZk x r) (n : ℕ) :
    outerHit x ((List.range n).map fun j => (Claims.coinSeq r j).bytes) = false := by
  unfold outerHit
  rw [List.any_eq_false]
  intro p hp
  simp only [List.length_map, List.length_range, Bool.and_eq_true, decide_eq_true_eq, not_and]
  intro hpn
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hpn, Option.map_some, Option.getD_some,
    JsonRT.coinOf_bytes]
  exact fun h => hr (Or.inl ⟨p, hp, h⟩)

theorem run_honestPR (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    (r : Fin ((V true).k x) → Coin) (hr : ¬ BadZk x r) :
    Claims.run H (honestPR w) (V true) x ω r = Claims.run H (honestP (maskWOf x w ω.mask)) (V true) x ω r :=
  stepWith_congr_coins H r (fun hist hh => honestPR_next (by rw [hh]; exact outerHit_coins hr _)) _ _ _ _ rfl rfl

def MaskFree (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ x w (μ : ℕ → ℕ → F128), Claims.Flock.Scope (deployZk rel) x → Claims.Flock.R (deployZk rel).parse x w →
    Claims.Flock.R (deployZk rel).parse x (maskWOf x w μ)

theorem complete_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel)
    (hacc : ∀ x w, Claims.Flock.Scope (deployZk rel) x → Claims.Flock.R (deployZk rel).parse x w →
      ready H x = true → ∀ (ω : TapeZk) (r : Fin ((V true).k x) → Coin), ¬ BadZk x r →
        (Claims.run H (honestP w) (V true) x ω r).out = .acc) :
    Claims.Flock.Complete H (deployZk rel) := by
  intro _ x w hs hr ω
  show 1 - δc H x ≤ _
  unfold δc
  split
  · rename_i hready
    calc 1 - εZk x ≤ 1 - Claims.Pr (fun r : Fin ((V true).k x) → Coin => BadZk x r) := by
          linarith [pr_badZk_le hMR x (rankSlotsZk_layout hs) (k := (V true).k x)]
      _ = Claims.Pr (fun r : Fin ((V true).k x) → Coin => ¬ BadZk x r) := (Pr_not _).symm
      _ ≤ Claims.Pr (fun r : Fin ((V true).k x) → Coin =>
            (Claims.run H (honestPR w) (V true) x ω r).out = .acc) :=
          Claims.Pr_mono fun r hr' => by
            rw [run_honestPR H x w ω r hr']
            exact hacc x _ hs (hMF x w ω.mask hs hr) hready ω r hr'
  · rw [sub_self]
    exact Claims.Pr_nonneg _

/-- **A setup with several tables is never `readyZk`**, so `δcZk H` is `1` there. -/
theorem readyZk_tables {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h : 1 < sts.size) : readyZk H x = false := by
  unfold readyZk
  rw [hs]
  have : (sts.size == 1) = false := by simp; omega
  simp [this]

end FlockComplete.ZkTables
