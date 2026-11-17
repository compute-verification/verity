module

public import Proofs.Flock.Claims.Bridge.Transport
public import Proofs.Flock.Claims.Deploy
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
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import VCVio.Prelude.Core

/-! Committed coins, the `--zk` Hello round, are within `δ_hide` of fresh ones. -/

@[expose] public section

namespace Claims.Bridge

open FlockSoundness.Game (ind)

/-- **A coin tree** over `k` rounds: the root of the coins `r` at the tree's randomness `τ` (`hello`), sent before the
prover's first message, and round `j`'s opening (`opn`), sent with the answer to round `j`. -/
structure CoinTree (Coin T : Type) (k : ℕ) where
  hello : (Fin k → Coin) → T → Msg
  opn : (Fin k → Coin) → T → Fin k → Msg

/-- The openings of rounds `< k` as a sequence, empty past them. -/
def openSeq {k : ℕ} (π : Fin k → Msg) (i : ℕ) : Msg := if h : i < k then π ⟨i, h⟩ else []

/-- **A prover against committed coins**: `a_j = P*^H(x, root, a_1, b_1, …, b_{j−1}, π_1, …, π_{j−1}; ω)`, with the
root and the openings of the rounds answered so far. -/
structure CProver (Ω : Type) where
  next : Msg → Msg → List (Msg × Msg) → List Msg → Ω → OC Msg

def CProver.at {Ω : Type} (P : CProver Ω) (h : Msg) (π : ℕ → Msg) : Prover Ω :=
  ⟨fun x hist ω => P.next x h hist ((List.range hist.length).map π) ω⟩

def runC {Coin Ω T : Type} [Inhabited Coin] (H : Msg → Msg) (P : CProver Ω) (V : Verifier Coin) (x : Msg)
    (C : CoinTree Coin T (V.k x)) (ω : Ω) (r : Fin (V.k x) → Coin) (τ : T) : Transcript :=
  run H (P.at (C.hello r τ) (openSeq (C.opn r τ))) V x ω r

/-- **An online simulator** of a coin tree: a root from its tape alone, and round `j`'s opening from its tape and the
answers to rounds `0, …, j`. -/
structure OnlineSim where
  Tape : Type
  fin : Fintype Tape
  ne : Nonempty Tape
  hello : Tape → Msg
  opn : Tape → List Msg → Msg

instance OnlineSim.instFintypeTape (S : OnlineSim) : Fintype S.Tape := S.fin

instance OnlineSim.instNonemptyTape (S : OnlineSim) : Nonempty S.Tape := S.ne

def OnlineSim.opnSeq {Coin : Type} [Inhabited Coin] (S : OnlineSim) (ans : Coin → Msg) {k : ℕ} (σ : S.Tape)
    (r : Fin k → Coin) : Fin k → Msg :=
  fun j => S.opn σ ((List.range (j.val + 1)).map (ans ∘ coinSeq r))

def SimHides {Coin T : Type} [Fintype Coin] [Inhabited Coin] [Fintype T] {k : ℕ} (C : CoinTree Coin T k)
    (ans : Coin → Msg) (S : OnlineSim) (δ_hide : ℝ) : Prop :=
  ∀ Ev : (Fin k → Coin) → Msg → (Fin k → Msg) → Prop,
    |Pr (fun p : (Fin k → Coin) × T => Ev p.1 (C.hello p.1 p.2) (C.opn p.1 p.2)) -
      Pr (fun p : (Fin k → Coin) × S.Tape => Ev p.1 (S.hello p.2) (S.opnSeq ans p.2 p.1))| ≤ δ_hide

/-- Not proved here: a statistically hiding tree has such a simulator, and `δ_hide` is then the hm96 leaves'
hiding over the rounds. -/
def TreeHides {Coin T : Type} [Fintype Coin] [Inhabited Coin] [Fintype T] {k : ℕ} (C : CoinTree Coin T k)
    (ans : Coin → Msg) (δ_hide : ℝ) : Prop :=
  ∃ S : OnlineSim, SimHides C ans S δ_hide

/-- **The verifier answers with its coin**: every message it answers a round with is `ans` of that round's coin, the
last of the coins it is given. -/
def AnswersCoin {Coin : Type} (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (ans : Coin → Msg) : Prop :=
  ∀ (as : List Msg) (cs : List Coin) (c : Coin) (m : Msg), ((V.next x as (cs ++ [c])).eval H).1 = .msg m → m = ans c

def simProver {Ω : Type} (S : OnlineSim) (k : Msg → ℕ) (P : CProver Ω) : Prover (Ω × S.Tape) :=
  ⟨fun x hist ω => P.next x (S.hello ω.2) hist ((List.range hist.length).map fun i =>
    if i < k x then S.opn ω.2 ((hist.take (i + 1)).map Prod.snd) else []) ω.1⟩

def Answered {Coin : Type} (ans : Coin → Msg) (r : ℕ → Coin) (hist : List (Msg × Msg)) : Prop :=
  ∀ i (hi : i < hist.length), hist[i].2 = ans (r i)

theorem answered_nil {Coin : Type} (ans : Coin → Msg) (r : ℕ → Coin) : Answered ans r [] :=
  fun _ hi => absurd hi (Nat.not_lt_zero _)

theorem answered_snoc {Coin : Type} {ans : Coin → Msg} {r : ℕ → Coin} {hist : List (Msg × Msg)}
    (h : Answered ans r hist) (a : Msg) : Answered ans r (hist ++ [(a, ans (r hist.length))]) := by
  intro i hi
  rw [List.length_append, List.length_singleton] at hi
  rcases Nat.lt_or_ge i hist.length with hl | hl
  · rw [List.getElem_append_left hl]
    exact h i hl
  · have hi' : i = hist.length := by omega
    subst hi'
    simp

theorem stepWith_congr {Coin : Type} (H : Msg → Msg) {Ω Ω' : Type} (P : Prover Ω) (P' : Prover Ω')
    (V : Verifier Coin) (x : Msg) (ω : Ω) (ω' : Ω') (ans : Coin → Msg) (r : ℕ → Coin) (hV : AnswersCoin H V x ans)
    (hP : ∀ hist, Answered ans r hist → P.next x hist ω = P'.next x hist ω') :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), Answered ans r hist →
      hist.length = as.length →
      stepWith H P x ω (fun as => V.next x as ((List.range as.length).map r)) n hist as t =
        stepWith H P' x ω' (fun as => V.next x as ((List.range as.length).map r)) n hist as t := by
  intro n
  induction n with
  | zero => intro _ _ _ _ _; rfl
  | succ n ih =>
    intro hist as t hA hl
    simp only [stepWith]
    rw [hP hist hA]
    generalize hvb : (V.next x (as ++ [((P'.next x hist ω').eval H).1])
      ((List.range (as ++ [((P'.next x hist ω').eval H).1]).length).map r)).eval H = vb
    obtain ⟨b, l⟩ := vb
    cases b with
    | msg m =>
      have hm : m = ans (r hist.length) := by
        refine hV (as ++ [((P'.next x hist ω').eval H).1]) ((List.range as.length).map r) (r as.length) m ?_
          |>.trans (by rw [hl])
        have e : (List.range (as ++ [((P'.next x hist ω').eval H).1]).length).map r =
            (List.range as.length).map r ++ [r as.length] := by
          simp [List.range_succ]
        rw [← e, hvb]
      subst hm
      exact ih _ _ _ (answered_snoc hA _) (by simp [hl])
    | acc => rfl
    | rej => rfl

theorem run_simProver {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) {Ω : Type}
    (S : OnlineSim) (P : CProver Ω) (ans : Coin → Msg) (hV : AnswersCoin H V x ans) (ω : Ω) (σ : S.Tape)
    (r : Fin (V.k x) → Coin) :
    run H (simProver S V.k P) V x (ω, σ) r = run H (P.at (S.hello σ) (openSeq (S.opnSeq ans σ r))) V x ω r := by
  unfold run
  refine stepWith_congr H _ _ V x _ _ ans (coinSeq r) hV (fun hist hA => ?_) _ _ _ _ (answered_nil _ _) rfl
  simp only [simProver, CProver.at]
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi' := List.mem_range.1 hi
  simp only [openSeq, OnlineSim.opnSeq]
  split_ifs with hk
  · congr 1
    refine List.ext_getElem (by simp only [List.length_map, List.length_take, List.length_range]; omega)
      fun n h1 h2 => ?_
    simp only [List.length_map, List.length_take] at h1
    simp only [List.getElem_map, List.getElem_take, List.getElem_range, Function.comp]
    exact hA n (by omega)
  · rfl

theorem abs_Ex_sub_le {α : Type} [Fintype α] [Nonempty α] {f g : α → ℝ} {δ : ℝ} (h : ∀ a, |f a - g a| ≤ δ) :
    |Ex f - Ex g| ≤ δ := by
  have e : Ex f - Ex g = Ex fun a => f a - g a := by
    unfold Ex
    rw [← sub_div, ← Finset.sum_sub_distrib]
  rw [e, abs_le]
  constructor
  · calc -δ = Ex fun _ : α => -δ := (Ex_const _).symm
      _ ≤ _ := Ex_mono fun a => (abs_le.1 (h a)).1
  · calc _ ≤ Ex fun _ : α => δ := Ex_mono fun a => (abs_le.1 (h a)).2
      _ = δ := Ex_const _

theorem runC_near {Coin Ω T : Type} [Fintype Coin] [Inhabited Coin] [Fintype Ω] [Nonempty Ω] [Fintype T]
    (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (C : CoinTree Coin T (V.k x)) (ans : Coin → Msg) (S : OnlineSim)
    {δ_hide : ℝ} (hS : SimHides C ans S δ_hide) (hV : AnswersCoin H V x ans) (P : CProver Ω)
    (Ev : Ω → (Fin (V.k x) → Coin) → Transcript → Prop) :
    |Pr (fun p : Ω × (Fin (V.k x) → Coin) × T => Ev p.1 p.2.1 (runC H P V x C p.1 p.2.1 p.2.2)) -
      Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => Ev p.1.1 p.2 (run H (simProver S V.k P) V x p.1 p.2))| ≤
      δ_hide := by
  have hL : Pr (fun p : Ω × (Fin (V.k x) → Coin) × T => Ev p.1 p.2.1 (runC H P V x C p.1 p.2.1 p.2.2)) =
      Ex fun ω => Pr fun q : (Fin (V.k x) → Coin) × T =>
        Ev ω q.1 (run H (P.at (C.hello q.1 q.2) (openSeq (C.opn q.1 q.2))) V x ω q.1) := by
    rw [Pr_eq_Ex, Ex_prod]
    simp only [Pr_eq_Ex]
    rfl
  have hR : Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) =>
        Ev p.1.1 p.2 (run H (simProver S V.k P) V x p.1 p.2)) =
      Ex fun ω => Pr fun q : (Fin (V.k x) → Coin) × S.Tape =>
        Ev ω q.1 (run H (P.at (S.hello q.2) (openSeq (S.opnSeq ans q.2 q.1))) V x ω q.1) := by
    have e : (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) =>
          Ev p.1.1 p.2 (run H (simProver S V.k P) V x p.1 p.2)) =
        fun p => Ev p.1.1 p.2 (run H (P.at (S.hello p.1.2) (openSeq (S.opnSeq ans p.1.2 p.2))) V x p.1.1 p.2) := by
      funext p
      obtain ⟨⟨ω, σ⟩, r⟩ := p
      rw [run_simProver H V x S P ans hV]
    rw [e, Pr_eq_Ex, Ex_prod, Ex_prod]
    simp only [Pr_eq_Ex, Ex_prod]
    exact congrArg Ex (funext fun ω => Ex_comm _)
  rw [hL, hR]
  exact abs_Ex_sub_le fun ω => hS fun r h π => Ev ω r (run H (P.at h (openSeq π)) V x ω r)

/-- The bound a knowledge-soundness proof charges in `κ`. -/
theorem Pr_runC_le {Coin Ω T : Type} [Fintype Coin] [Inhabited Coin] [Fintype Ω] [Nonempty Ω] [Fintype T]
    (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (C : CoinTree Coin T (V.k x)) (ans : Coin → Msg) (S : OnlineSim)
    {δ_hide : ℝ} (hS : SimHides C ans S δ_hide) (hV : AnswersCoin H V x ans) (P : CProver Ω)
    (Ev : Ω → (Fin (V.k x) → Coin) → Transcript → Prop) :
    Pr (fun p : Ω × (Fin (V.k x) → Coin) × T => Ev p.1 p.2.1 (runC H P V x C p.1 p.2.1 p.2.2)) ≤
      Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => Ev p.1.1 p.2 (run H (simProver S V.k P) V x p.1 p.2)) +
        δ_hide := by
  have := (abs_le.1 (runC_near H V x C ans S hS hV P Ev)).2
  linarith

theorem Pr_runC_le_of_treeHides {Coin Ω T : Type} [Fintype Coin] [Inhabited Coin] [Fintype Ω] [Nonempty Ω]
    [Fintype T] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (C : CoinTree Coin T (V.k x)) (ans : Coin → Msg)
    {δ_hide : ℝ} (hT : TreeHides C ans δ_hide) (hV : AnswersCoin H V x ans) (P : CProver Ω) :
    ∃ S : OnlineSim, ∀ Ev : Ω → (Fin (V.k x) → Coin) → Transcript → Prop,
      Pr (fun p : Ω × (Fin (V.k x) → Coin) × T => Ev p.1 p.2.1 (runC H P V x C p.1 p.2.1 p.2.2)) ≤
        Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => Ev p.1.1 p.2 (run H (simProver S V.k P) V x p.1 p.2)) +
          δ_hide :=
  let ⟨S, hS⟩ := hT
  ⟨S, Pr_runC_le H V x C ans S hS hV P⟩

theorem act_coin {zk : Bool} {x : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} {c : FlockDeploy.Coin}
    (h : FlockDeploy.act zk x as cs = .coin c) : cs.getLast? = some c := by
  unfold FlockDeploy.act at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem outOf_ne_msg (H : Msg → Msg) (d : OCV Unit) (m : Msg) : ((FlockDeploy.outOf d).eval H).1 ≠ .msg m := by
  unfold FlockDeploy.outOf
  rw [OC.eval_bind]
  simp only [OC.eval]
  split <;> simp

theorem answersCoin_deploy (zk : Bool) (H : Msg → Msg) (x : Msg) :
    AnswersCoin H (FlockDeploy.V zk) x FlockDeploy.Coin.bytes := by
  intro as cs c m h
  change ((FlockDeploy.next zk x as (cs ++ [c])).eval H).1 = .msg m at h
  cases ha : FlockDeploy.act zk x as (cs ++ [c]) with
  | rej =>
    rw [FlockDeploy.next_rej H ha] at h
    cases h
  | coin c' =>
    rw [FlockDeploy.next_coin H ha] at h
    have hc : c = c' := by simpa using act_coin ha
    rw [← Out.msg.inj h, hc]
  | decide s spec sts link c₁ mids f =>
    simp only [FlockDeploy.next, ha] at h
    exact absurd h (outOf_ne_msg _ _ _)

end Claims.Bridge
