module

public import Batteries.Data.ByteArray
public import Proofs.Flock.Claims.KS.BatchMove
public import Proofs.Flock.Claims.KS.BatchSched
public import Proofs.Flock.Claims.KS.SimLead
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
import Proofs.Flock.Claims.KS.BatchSteps
import Std.Tactic.BVDecide.Normalize.Prop
import VCVio.Prelude.Core

/-! The simulated strategy of `J` tables follows the batch's schedule where the live prover submits its rounds stream by stream. -/

@[expose] public section

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Tables Claims.Bridge
open FlockSoundness.Game.Lock (WaitR FollowLP Paused Front Sess consP headS deadT FirstS predS DepAt)

local notation "𝔽" => FlockLevel3.GF128

section Batch

variable {ζ : Type}

theorem tr_list (Φ : (l : List (Game ζ)) → Strategy (batch l) → Prop) {l l' : List (Game ζ)} (e : l = l')
    {σ : Strategy (batch l)} {σ' : Strategy (batch l')} (h : HEq σ σ') (hΦ : Φ l' σ') : Φ l σ := by
  subst e
  cases h
  exact hΦ

theorem ofFn_split {J : ℕ} (f : Fin J → Game ζ) (j : Fin J) :
    List.ofFn f = (List.ofFn f).take j ++ f j :: (List.ofFn f).drop (j + 1) := by
  rw [← ofFn_update, Function.update_eq_self]

theorem round_heq_gen {A : Type} {C : Game A → Game ζ} (c : Ctx C) {J : ℕ} (f : Fin J → Game ζ) (t : Fin J)
    {E : Game A → Game (List ζ)} (hE : ∀ g, batch (List.ofFn (Function.update f t (C g))) = E g) {g₀ : Game A}
    {Cm : Type} [Fintype Cm] [Nonempty Cm] (kk : Cm → Game A) (s : Steps g₀ (.coin Cm kk))
    (ψ : Strategy (.coin Cm (((c.comp (Ctx.batchFn f t)).congr hE).coin kk).next)) :
    HEq (playSteps (((c.comp (Ctx.batchFn f t)).congr hE).lift s)
        (playSteps (((c.comp (Ctx.batchFn f t)).congr hE).coin kk).steps ψ))
      (playSteps ((Ctx.batchAt ((List.ofFn f).take t) ((List.ofFn f).drop (t + 1))).lift
          ((c.lift s).trans (c.coin kk).steps))
        (playSteps ((Ctx.batchAt ((List.ofFn f).take t) ((List.ofFn f).drop (t + 1))).coin
          (c.coin kk).next).steps ψ)) := by
  have h₁ := playSteps_congr_coin (Ctx.batchAt ((List.ofFn f).take t) ((List.ofFn f).drop (t + 1)))
    (fun g => congrArg batch (ofFn_update f t g).symm) (c.coin kk).next ψ
  have h₂ := playSteps_congr_lift (Ctx.batchAt ((List.ofFn f).take t) ((List.ofFn f).drop (t + 1)))
    (fun g => congrArg batch (ofFn_update f t g).symm) ((c.lift s).trans (c.coin kk).steps) _ _ h₁
  refine (playSteps_congr_lift (c.comp (Ctx.batchFn f t)) hE s _ _
    (playSteps_congr_coin (c.comp (Ctx.batchFn f t)) hE kk ψ)).trans ?_
  rw [playSteps_comp_lift]
  exact (heq_of_eq (congrArg (playSteps ((Ctx.batchFn f t).lift (c.lift s)))
    (playSteps_trans ((Ctx.batchFn f t).lift (c.coin kk).steps) ((Ctx.batchFn f t).coin (c.coin kk).next).steps
      ψ))).trans ((heq_of_eq (playSteps_lift_trans _ _ _ _).symm).trans h₂)

theorem paused_round' {b : Option ℕ} {Inner : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop}
    {C : Type} [Fintype C] [Nonempty C] {X : Game ζ} {N : C → Game ζ} (T : Steps X (.coin C N)) (B : List (Game ζ))
    (Inner' : C → (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop)
    (hInner : ∀ D (ψ : Strategy (.coin C ((Ctx.batchAt [] B).coin N).next)),
      (∀ c, Inner' c D (N c :: B) (cast (congrArg Strategy (((Ctx.batchAt [] B).coin N).eq c)) (ψ c))) →
        Inner D (X :: B) (playSteps ((Ctx.batchAt [] B).lift T) (playSteps ((Ctx.batchAt [] B).coin N).steps ψ)))
    (A : List (Game ζ)) (np : ℕ) (hnp : A.length = np) {D : List ζ → Prop}
    (ψ : Strategy (.coin C ((Ctx.batchAt A B).coin N).next))
    (hP : ∀ c, Paused (predS b) (Inner' c) np D (A ++ N c :: B)
      (cast (congrArg Strategy (((Ctx.batchAt A B).coin N).eq c)) (ψ c))) :
    Paused b Inner np D (A ++ X :: B)
      (playSteps ((Ctx.batchAt A B).lift T) (playSteps ((Ctx.batchAt A B).coin N).steps ψ)) := by
  subst hnp
  exact paused_round T B Inner' hInner A ψ hP

theorem sess_dead' {α₁ β₁ : Type} {b : Option ℕ} {nL : ℕ} {f : α₁ × β₁ → Game ζ} {g₁ : Game α₁} {g₂ : Game β₁}
    {Dead : List ζ → Prop} (hd : DepAt 0 Dead) {l : List (Game ζ)} (hl : l ≠ []) {σ : Strategy (batch l)}
    (hσ : ∀ x, Lock.Reach _ σ x → Dead x) : Sess b nL f g₁ g₂ Dead l σ := by
  cases l with
  | nil => exact absurd rfl hl
  | cons X B => exact Lock.sess_dead hd hσ

end Batch

section Round

variable {P OutT Cp Op : Type} [Nonempty Op] {J : ℕ} (out : Fin J → Cp → P × P → Op → OutT) (caps : Fin J → Cp)

abbrev ctxB (gs : Fin J → Fin 2 → Game P) (t : Fin J) :=
  Ctx.batchAt ((List.ofFn (tabG out caps gs)).take t) ((List.ofFn (tabG out caps gs)).drop (t + 1))

abbrev ctx₀ (gs : Fin J → Fin 2 → Game P) (t : Fin J) :=
  (Ctx.left (gs t 1)).comp (Ctx.bind (KgJ out t (caps t)))

abbrev ctx₁ (gs : Fin J → Fin 2 → Game P) (t : Fin J) :=
  (Ctx.right (gs t 0)).comp (Ctx.bind (KgJ out t (caps t)))

omit [Nonempty Op] in
theorem ofFn_tabG_upd₀ (gs : Fin J → Fin 2 → Game P) (t : Fin J) (g : Game P) :
    List.ofFn (tabG out caps (upd gs t 0 g)) =
      (List.ofFn (tabG out caps gs)).take t ++ (interleave g (gs t 1)).bind (KgJ out t (caps t)) ::
        (List.ofFn (tabG out caps gs)).drop (t + 1) := by
  rw [← ofFn_update]
  congr 1
  funext j'
  by_cases h : j' = t
  · subst h
    rw [Function.update_self]
    simp [tabG, upd]
  · rw [Function.update_of_ne h]
    simp [tabG, upd, h]

omit [Nonempty Op] in
theorem ofFn_tabG_upd₁ (gs : Fin J → Fin 2 → Game P) (t : Fin J) (g : Game P) :
    List.ofFn (tabG out caps (upd gs t 1 g)) =
      (List.ofFn (tabG out caps gs)).take t ++ (interleave (gs t 0) g).bind (KgJ out t (caps t)) ::
        (List.ofFn (tabG out caps gs)).drop (t + 1) := by
  rw [← ofFn_update]
  congr 1
  funext j'
  by_cases h : j' = t
  · subst h
    rw [Function.update_self]
    simp [tabG, upd]
  · rw [Function.update_of_ne h]
    simp [tabG, upd, h]

omit [Nonempty Op] in
theorem length_take_ofFn (gs : Fin J → Fin 2 → Game P) (t : Fin J) :
    ((List.ofFn (tabG out caps gs)).take t).length = t := by
  simp [t.isLt.le]

abbrev roundS (gs : Fin J → Fin 2 → Game P) (t : Fin J) (r : Fin 2) {Cm : Type} [Fintype Cm] [Nonempty Cm]
    (kk : Cm → Game P) (s : Steps (gs t r) (.coin Cm kk))
    (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t r (kk x)))) : Strategy (posOf out caps gs) :=
  playSteps ((Steps.ofEq (posOf_upd_self out caps gs t r).symm).trans ((ctxAt out caps gs t r).lift s))
    (playSteps ((ctxAt out caps gs t r).coin kk).steps fun x =>
      cast (congrArg Strategy (((ctxAt out caps gs t r).coin kk).eq x).symm) (ψn x))

theorem round_heq_ofEq {A : Type} {C : Game A → Game (List OutT)} (c : Ctx C) {g : Game (List OutT)} {g₁ : Game A}
    (e₁ : g = C g₁) {Cm : Type} [Fintype Cm] [Nonempty Cm] (kk : Cm → Game A) (s : Steps g₁ (.coin Cm kk))
    (ψ : Strategy (.coin Cm (c.coin kk).next)) :
    HEq (playSteps ((Steps.ofEq e₁).trans (c.lift s)) (playSteps (c.coin kk).steps ψ))
      (playSteps (c.lift s) (playSteps (c.coin kk).steps ψ)) := by
  rw [playSteps_trans]
  exact playSteps_ofEq_heq _ _

theorem step0 {gs : Fin J → Fin 2 → Game P} (t : Fin J) {Cm : Type} [Fintype Cm] [Nonempty Cm] (kk : Cm → Game P)
    (s : Steps (gs t 0) (.coin Cm kk)) (Φ : (l : List (Game OutT)) → Strategy (batch l) → Prop)
    (Φ' : Cm → (l : List (Game OutT)) → Strategy (batch l) → Prop)
    (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t 0 (kk x))))
    (hB : ∀ ψ : Strategy (.coin Cm ((ctxB out caps gs t).coin ((ctx₀ out caps gs t).coin kk).next).next),
      (∀ x, Φ' x _ (cast (congrArg Strategy (((ctxB out caps gs t).coin ((ctx₀ out caps gs t).coin kk).next).eq x))
        (ψ x))) →
      Φ _ (playSteps ((ctxB out caps gs t).lift (((ctx₀ out caps gs t).lift s).trans
          ((ctx₀ out caps gs t).coin kk).steps))
        (playSteps ((ctxB out caps gs t).coin ((ctx₀ out caps gs t).coin kk).next).steps ψ)))
    (hN : ∀ x, Φ' x (List.ofFn (tabG out caps (upd gs t 0 (kk x)))) (ψn x)) :
    Φ (List.ofFn (tabG out caps gs))
      (playSteps ((Steps.ofEq (posOf_upd_self out caps gs t 0).symm).trans ((ctxAt out caps gs t 0).lift s))
        (playSteps ((ctxAt out caps gs t 0).coin kk).steps fun x =>
          cast (congrArg Strategy (((ctxAt out caps gs t 0).coin kk).eq x).symm) (ψn x))) :=
  tr_list Φ (ofFn_split (tabG out caps gs) t)
    ((round_heq_ofEq (ctxAt out caps gs t 0) (posOf_upd_self out caps gs t 0).symm kk s _).trans
      (round_heq_gen (ctx₀ out caps gs t) (tabG out caps gs) t (posOf_upd₀ out caps gs t) kk s _))
    (hB _ fun x => tr_list (Φ' x)
      ((congrArg (fun X => (List.ofFn (tabG out caps gs)).take t ++ X :: (List.ofFn (tabG out caps gs)).drop (t + 1))
        (((ctx₀ out caps gs t).coin kk).eq x)).trans (ofFn_tabG_upd₀ out caps gs t (kk x)).symm)
      ((cast_heq _ _).trans (cast_heq _ _)) (hN x))

theorem step1 {gs : Fin J → Fin 2 → Game P} (t : Fin J) {Cm : Type} [Fintype Cm] [Nonempty Cm] (kk : Cm → Game P)
    (s : Steps (gs t 1) (.coin Cm kk)) (Φ : (l : List (Game OutT)) → Strategy (batch l) → Prop)
    (Φ' : Cm → (l : List (Game OutT)) → Strategy (batch l) → Prop)
    (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t 1 (kk x))))
    (hB : ∀ ψ : Strategy (.coin Cm ((ctxB out caps gs t).coin ((ctx₁ out caps gs t).coin kk).next).next),
      (∀ x, Φ' x _ (cast (congrArg Strategy (((ctxB out caps gs t).coin ((ctx₁ out caps gs t).coin kk).next).eq x))
        (ψ x))) →
      Φ _ (playSteps ((ctxB out caps gs t).lift (((ctx₁ out caps gs t).lift s).trans
          ((ctx₁ out caps gs t).coin kk).steps))
        (playSteps ((ctxB out caps gs t).coin ((ctx₁ out caps gs t).coin kk).next).steps ψ)))
    (hN : ∀ x, Φ' x (List.ofFn (tabG out caps (upd gs t 1 (kk x)))) (ψn x)) :
    Φ (List.ofFn (tabG out caps gs))
      (playSteps ((Steps.ofEq (posOf_upd_self out caps gs t 1).symm).trans ((ctxAt out caps gs t 1).lift s))
        (playSteps ((ctxAt out caps gs t 1).coin kk).steps fun x =>
          cast (congrArg Strategy (((ctxAt out caps gs t 1).coin kk).eq x).symm) (ψn x))) :=
  tr_list Φ (ofFn_split (tabG out caps gs) t)
    ((round_heq_ofEq (ctxAt out caps gs t 1) (posOf_upd_self out caps gs t 1).symm kk s _).trans
      (round_heq_gen (ctx₁ out caps gs t) (tabG out caps gs) t (posOf_upd₁ out caps gs t) kk s _))
    (hB _ fun x => tr_list (Φ' x)
      ((congrArg (fun X => (List.ofFn (tabG out caps gs)).take t ++ X :: (List.ofFn (tabG out caps gs)).drop (t + 1))
        (((ctx₁ out caps gs t).coin kk).eq x)).trans (ofFn_tabG_upd₁ out caps gs t (kk x)).symm)
      ((cast_heq _ _).trans (cast_heq _ _)) (hN x))

theorem front_step {k : ℕ} {S : (List OutT → Prop) → (l : List (Game OutT)) → Strategy (batch l) → Prop}
    {gs : Fin J → Fin 2 → Game P} (t : Fin J) (r : Fin 2) {Cm : Type} [Fintype Cm] [Nonempty Cm] (kk : Cm → Game P)
    (s : Steps (gs t r) (.coin Cm kk)) (m c : ℕ) (ns : List ℕ) {D : List OutT → Prop}
    (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t r (kk x))))
    (hN : ∀ x, Paused (some m) (Front k S (c :: ns)) t D (List.ofFn (tabG out caps (upd gs t r (kk x)))) (ψn x)) :
    Paused (some (m + 1)) (Front k S ((c + 1) :: ns)) t D (List.ofFn (tabG out caps gs))
      (playSteps ((Steps.ofEq (posOf_upd_self out caps gs t r).symm).trans ((ctxAt out caps gs t r).lift s))
        (playSteps ((ctxAt out caps gs t r).coin kk).steps fun x =>
          cast (congrArg Strategy (((ctxAt out caps gs t r).coin kk).eq x).symm) (ψn x))) := by
  rcases r with ⟨_ | _ | _, hr⟩
  · exact step0 out caps t kk s (fun l σ => Paused (some (m + 1)) (Front k S ((c + 1) :: ns)) t D l σ)
      (fun _ l σ => Paused (some m) (Front k S (c :: ns)) t D l σ) ψn
      (fun ψ hP => paused_round' _ _ (fun _ => Front k S (c :: ns)) (fun D ψ hF => front_round k S c ns _ _ D ψ hF)
        _ t (length_take_ofFn out caps gs t) ψ hP) hN
  · exact step1 out caps t kk s (fun l σ => Paused (some (m + 1)) (Front k S ((c + 1) :: ns)) t D l σ)
      (fun _ l σ => Paused (some m) (Front k S (c :: ns)) t D l σ) ψn
      (fun ψ hP => paused_round' _ _ (fun _ => Front k S (c :: ns)) (fun D ψ hF => front_round k S c ns _ _ D ψ hF)
        _ t (length_take_ofFn out caps gs t) ψ hP) hN
  · omega

theorem sess_step0 {gs : Fin J → Fin 2 → Game P} (t : Fin J) {Cm : Type} [Fintype Cm] [Nonempty Cm]
    (kk : Cm → Game P) (s : Steps (gs t 0) (.coin Cm kk)) (b : Option ℕ) (c : ℕ) {D : List OutT → Prop}
    (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t 0 (kk x))))
    (hN : ∀ x, Paused (predS b) (Sess (predS b) c (KgJ out t (caps t)) (kk x) (gs t 1)) t D
      (List.ofFn (tabG out caps (upd gs t 0 (kk x)))) (ψn x)) :
    Paused b (Sess b (c + 1) (KgJ out t (caps t)) (gs t 0) (gs t 1)) t D (List.ofFn (tabG out caps gs))
      (playSteps ((Steps.ofEq (posOf_upd_self out caps gs t 0).symm).trans ((ctxAt out caps gs t 0).lift s))
        (playSteps ((ctxAt out caps gs t 0).coin kk).steps fun x =>
          cast (congrArg Strategy (((ctxAt out caps gs t 0).coin kk).eq x).symm) (ψn x))) :=
  step0 out caps t kk s (fun l σ => Paused b (Sess b (c + 1) (KgJ out t (caps t)) (gs t 0) (gs t 1)) t D l σ)
    (fun x l σ => Paused (predS b) (Sess (predS b) c (KgJ out t (caps t)) (kk x) (gs t 1)) t D l σ) ψn
    (fun ψ hP => paused_round' _ _ (fun x => Sess (predS b) c (KgJ out t (caps t)) (kk x) (gs t 1))
      (fun D ψ hS => sess_round0 (gs t 1) s _ D ψ hS) _ t (length_take_ofFn out caps gs t) ψ hP) hN

theorem sess_step1 {gs : Fin J → Fin 2 → Game P} (t : Fin J) {Cm : Type} [Fintype Cm] [Nonempty Cm]
    (kk : Cm → Game P) (s : Steps (gs t 1) (.coin Cm kk)) (b : Option ℕ) {a : P} (g₂ : Game P)
    {D : List OutT → Prop} (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t 1 (kk x))))
    (hN : ∀ x, Paused (predS b) (Sess (predS b) 0 (KgJ out t (caps t)) (.ret a) g₂) t D
      (List.ofFn (tabG out caps (upd gs t 1 (kk x)))) (ψn x)) :
    Paused b (Sess b 0 (KgJ out t (caps t)) (.ret a) g₂) t D (List.ofFn (tabG out caps gs))
      (playSteps ((Steps.ofEq (posOf_upd_self out caps gs t 1).symm).trans ((ctxAt out caps gs t 1).lift s))
        (playSteps ((ctxAt out caps gs t 1).coin kk).steps fun x =>
          cast (congrArg Strategy (((ctxAt out caps gs t 1).coin kk).eq x).symm) (ψn x))) :=
  step1 out caps t kk s (fun l σ => Paused b (Sess b 0 (KgJ out t (caps t)) (.ret a) g₂) t D l σ)
    (fun _ l σ => Paused (predS b) (Sess (predS b) 0 (KgJ out t (caps t)) (.ret a) g₂) t D l σ) ψn
    (fun ψ hP => paused_round' _ _ (fun _ => Sess (predS b) 0 (KgJ out t (caps t)) (.ret a) g₂)
      (fun D ψ hS => sess_round1 _ _ D ψ hS) _ t (length_take_ofFn out caps gs t) ψ hP) hN

end Round

theorem strategy_mapDead_coin_castR {β α : Type} {pL : β → Prop} {pg : α → Prop}
    (f : (g : Game α) → Strategy g → Strategy g) {C C' : Type} [Fintype C] [Nonempty C] [Fintype C'] [Nonempty C']
    {k : C → Game β} {g g' : C' → Game α} (e : C' ≃ C) (H : ∀ x, g x = g' x) (d : ∀ x, Sim pL pg (k (e x)) (g x))
    (τ : Strategy (.coin C k)) :
    (Sim.mapDead f (Sim.coin e fun x => Sim.castR (H x) (d x) : Sim pL pg (.coin C k) (.coin C' g'))).strategy τ =
      fun x => cast (congrArg Strategy (H x)) ((Sim.mapDead f (d x)).strategy (τ (e x))) :=
  funext fun x => (congrArg (fun d => Sim.strategy d (τ (e x))) (mapDead_castR f (H x) (d x))).trans
    (strategy_castR (H x) _ _)

section SchedJ

variable {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
  (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
  (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (caps : Fin J → Cp) (rootB : Array ByteArray)
  (pts : Fin 2 → Fin mPts → 𝔽) (nul : Op) {Dead : List OutT → Prop}
  {G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P}

/-- Stream `2 · t + r` is rep `r` of table `t`. -/
def SchedLJ : (n : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → List ℕ →
    Strategy ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) → Prop
  | 0, _, ss, _ => ss = []
  | n + 1, rs, ss, σ => match σ with
    | ⟨none, _⟩ => ss = []
    | ⟨some s, τ⟩ => ∃ ss', ss = (2 * s.tab.val + s.rep.val) :: ss' ∧
        ∀ c, SchedLJ n (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)) ss' (τ c)

theorem reps_of_schedLJ : ∀ (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (ss : List ℕ)
    (σ : Strategy ((liveRoundsJ dg J n rs).bind (KLJ rootB pts))), SchedLJ dg rootB pts n rs ss σ →
      ∀ (t : Fin J) (r : Fin 2), 2 * t.val + r.val ∉ ss → ∀ o, Lock.Reach _ σ o → o.reps t r = rs t r
  | 0, _, _, _, _, _, _, _, _, h => by cases h; rfl
  | _ + 1, _, _, ⟨none, _⟩, _, _, _, _, _, h => by cases h; rfl
  | n + 1, rs, _, ⟨some s, τ⟩, hS, t, r, hn, o, h => by
      obtain ⟨ss', rfl, hS'⟩ := hS
      obtain ⟨c, h⟩ := h
      refine (reps_of_schedLJ n _ ss' (τ c) (hS' c) t r (fun hm => hn (List.mem_cons_of_mem _ hm)) o h).trans
        (addRoundJ_ne rs _ fun e => hn ?_)
      obtain ⟨rfl, rfl⟩ := e
      exact List.mem_cons_self

omit [Nonempty Op] in
theorem noAcc_of_schedLJ (hfin : ∀ t r proofs, fin t r proofs = []) {n : ℕ} {rs : Fin J → Fin 2 → List Flock.Round}
    {gs : Fin J → Fin 2 → Game P}
    (hI : InvJ out ok pL opn caps ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs))
    {ss : List ℕ} {Pr : Strategy ((liveRoundsJ dg J n rs).bind (KLJ rootB pts))}
    (hS : SchedLJ dg rootB pts n rs ss Pr) {t : Fin J} {r : Fin 2} (hn : 2 * t.val + r.val ∉ ss)
    (hg : ∀ a, gs t r ≠ .ret a) : ∀ o, Lock.Reach _ Pr o → ¬ pL o := by
  intro o ho hp
  obtain ⟨pend, hpl, -⟩ := hI o (out_of_reach ho) hp
  have h0 := hpl t r
  have hr : restJ dec one fin rs t r o = [] := by
    simp only [restJ, reps_of_schedLJ dg rootB pts n rs ss Pr hS t r hn o ho, List.drop_length, hfin, tapeFrom,
      List.map_nil, List.append_nil]
  rw [hr] at h0
  exact hg _ (PlaysT.nil_inv h0)

theorem dead_answerWith (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) {caps' : Fin J → Cp}
    {gs' : Fin J → Fin 2 → Game P} (hinh : ∀ j r, AllInh (gs' j r)) (σ : Strategy (posOf out caps' gs')) :
    ∀ x, Lock.Reach _ (answerWith nul (posOf out caps' gs') σ) x → Dead x :=
  fun x hx => by
    obtain ⟨pend, hp⟩ := OutN.of_posOf out nul caps' gs' dz
      (reach_answerWith nul (allInh_posOf out caps' hinh) σ x hx) j
    exact hnul x _ _ hp

theorem dead_mapDead_dead (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) {β : Type} {pβ : β → Prop}
    {pg : List OutT → Prop} {L : Game β} {caps' : Fin J → Cp} {gs' : Fin J → Fin 2 → Game P}
    (hinh : ∀ j r, AllInh (gs' j r)) (d : Sim pβ pg L (posOf out caps' gs')) (hd : ∃ hp σ, d = Sim.dead hp σ)
    (s : Strategy L) : ∀ x, Lock.Reach _ ((Sim.mapDead (answerWith nul) d).strategy s) x → Dead x := by
  obtain ⟨hp, σ, rfl⟩ := hd
  exact dead_answerWith out nul j dz hnul hinh σ

theorem dead_noAcc (j : Fin J) (dz : OutT) (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs)
    {β : Type} {pβ : β → Prop} {pg : List OutT → Prop} {L : Game β} {gs : Fin J → Fin 2 → Game P}
    (d : Sim pβ pg L (posOf out caps gs)) (hA : Sim.All (fun b _ => pβ b) (DeadAt out G) d) (s : Strategy L)
    (hno : ∀ b, Lock.Reach L s b → ¬ pβ b) :
    ∀ x, Lock.Reach _ ((Sim.mapDead (answerWith nul) d).strategy s) x → Dead x := by
  intro x hx
  rcases reach_mapDead (answerWith nul) d hA s x hx with ⟨b, hb, hp⟩ | ⟨_, σ, ⟨caps', _, gs', hinh, _, rfl⟩, hr⟩
  · exact absurd hp (hno b hb)
  · exact dead_answerWith out nul j dz hnul hinh σ x hr

theorem coin_ind (j : Fin J) (dz : OutT) (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs)
    {n : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {t : Fin J} {r : Fin 2} {bb : ByteArray} {k : ℕ}
    {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r))
    (next : (c : Fin k → 𝔽) → (gs' : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn caps (roundL dg rootB pts n rs t r bb k c) gs'
        (restJ dec one fin (addRoundJ rs t r (roundOf dg bb k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs t r bb k c) (posOf out caps gs'))
    (τ : Strategy (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs t r bb k)))
    (Φ : Strategy (posOf out caps gs) → Prop)
    (Ψ : (gs' : Fin J → Fin 2 → Game P) → Strategy (posOf out caps gs') → Prop)
    (hdead : ∀ σ, (∀ x, Lock.Reach _ σ x → Dead x) → Φ σ)
    (hstep : ∀ (Cm : Type) (_ : Fintype Cm) (_ : Nonempty Cm) (kk : Cm → Game P) (s : Steps (gs t r) (.coin Cm kk))
      (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t r (kk x)))), (∀ x, Ψ _ (ψn x)) →
        Φ (roundS out caps gs t r kk s ψn))
    (hnext : ∀ c gs' h' hI', Sim.All (fun b _ => pL b) (DeadAt out G) (next c gs' h' hI') →
      Ψ gs' ((Sim.mapDead (answerWith nul) (next c gs' h' hI')).strategy (τ c))) :
    ∀ (g' : Game P) (s : Steps (gs t r) g') (hg : AllInh g')
      (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs t r bb k)) (upd gs t r g')
        (upd (restJ dec one fin rs) t r (midJ dec one fin rs t r))),
      Sim.All (fun b _ => pL b) (DeadAt out G)
        (coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs t r bb k gs h next g' hg hI) →
      Φ (playSteps ((Steps.ofEq (posOf_upd_self out caps gs t r).symm).trans ((ctxAt out caps gs t r).lift s))
        ((Sim.mapDead (answerWith nul)
          (coinJ out ok dflt pL opn dg dec one fin caps rootB pts n rs t r bb k gs h next g' hg hI)).strategy τ))
  | .ret _, _, hg, _, _ => hdead _ fun x hx =>
      dead_answerWith out nul j dz hnul (allInh_upd h t r hg) _ x ((reach_playSteps _ _).1 hx)
  | .send _ _, _, hg, _, _ => hdead _ fun x hx =>
      dead_answerWith out nul j dz hnul (allInh_upd h t r hg) _ x ((reach_playSteps _ _).1 hx)
  | @Game.coin _ Cm _ _ kk, s, hg, hI, hA => by
      rcases coinJ_coin out ok dflt pL opn dg dec one fin caps rootB pts h next kk hg hI with hd | ⟨e, hI', he⟩
      · exact hdead _ fun x hx => dead_mapDead_dead out nul j dz hnul (allInh_upd h t r hg) _ hd τ x
          ((reach_playSteps _ _).1 hx)
      · rw [he] at hA ⊢
        rw [mapDead_ofSteps, strategy_ofSteps, strategy_mapDead_coin_castR]
        exact hstep Cm _ _ kk s _ fun x => hnext (e x) _ _ (hI' x) (all_castR_inv _ _ (all_ofSteps_inv _ _ hA x))

theorem round_ind (j : Fin J) (dz : OutT) (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs)
    {n : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {t : Fin J} {r : Fin 2} {bb : ByteArray} {k : ℕ}
    {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs t r bb k)) gs (restJ dec one fin rs))
    (next : (c : Fin k → 𝔽) → (gs' : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn caps (roundL dg rootB pts n rs t r bb k c) gs'
        (restJ dec one fin (addRoundJ rs t r (roundOf dg bb k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs t r bb k c) (posOf out caps gs'))
    (τ : Strategy (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs t r bb k)))
    (Φ : Strategy (posOf out caps gs) → Prop)
    (Ψ : (gs' : Fin J → Fin 2 → Game P) → Strategy (posOf out caps gs') → Prop)
    (hdead : ∀ σ, (∀ x, Lock.Reach _ σ x → Dead x) → Φ σ)
    (hstep : ∀ (Cm : Type) (_ : Fintype Cm) (_ : Nonempty Cm) (kk : Cm → Game P) (s : Steps (gs t r) (.coin Cm kk))
      (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t r (kk x)))), (∀ x, Ψ _ (ψn x)) →
        Φ (roundS out caps gs t r kk s ψn))
    (hnext : ∀ c gs' h' hI', Sim.All (fun b _ => pL b) (DeadAt out G) (next c gs' h' hI') →
      Ψ gs' ((Sim.mapDead (answerWith nul) (next c gs' h' hI')).strategy (τ c)))
    (hA : Sim.All (fun b _ => pL b) (DeadAt out G)
      (roundJ out ok dflt pL opn dg dec one fin caps rootB pts n rs t r bb k gs h hI next)) :
    Φ ((Sim.mapDead (answerWith nul)
      (roundJ out ok dflt pL opn dg dec one fin caps rootB pts n rs t r bb k gs h hI next)).strategy τ) := by
  rcases hf : follow (gs t r) (dec t (rs t r).length bb) with _ | g'
  · exact hdead _ (dead_mapDead_dead out nul j dz hnul h _
      (roundJ_none out ok dflt pL opn dg dec one fin caps rootB pts h next hI hf) τ)
  · rw [roundJ_some out ok dflt pL opn dg dec one fin caps rootB pts h next hI hf] at hA ⊢
    rw [mapDead_ofSteps, strategy_ofSteps]
    exact coin_ind out ok dflt pL opn dg dec one fin caps rootB pts nul j dz hnul h next τ Φ Ψ hdead hstep hnext g'
      (stepsOfFollow _ _ hf) _ _ (all_ofSteps_inv _ _ hA)

def OnSched (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (ss : List ℕ)
    (Φ : (gs : Fin J → Fin 2 → Game P) → Strategy (posOf out caps gs) → Prop) : Prop :=
  ∀ n rs gs (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn caps ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs)) Pr,
    SchedLJ dg rootB pts n rs ss Pr →
    Sim.All (fun b _ => pL b) (DeadAt out G) (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI) →
    Φ gs ((Sim.mapDead (answerWith nul)
      (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI)).strategy Pr)

set_option hygiene false in
local notation "OnS" => OnSched out ok dflt pL opn dg dec one fin caps rootB pts nul

theorem onSched_cons (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (t : Fin J) (r : Fin 2) {p : ℕ}
    (hp : p = 2 * t.val + r.val) {ss : List ℕ}
    {Φ Ψ : (gs : Fin J → Fin 2 → Game P) → Strategy (posOf out caps gs) → Prop}
    (hdead : ∀ gs σ, (∀ x, Lock.Reach _ σ x → Dead x) → Φ gs σ)
    (hstep : ∀ gs (Cm : Type) (_ : Fintype Cm) (_ : Nonempty Cm) (kk : Cm → Game P) (s : Steps (gs t r) (.coin Cm kk))
      (ψn : (x : Cm) → Strategy (posOf out caps (upd gs t r (kk x)))), (∀ x, Ψ _ (ψn x)) →
        Φ gs (roundS out caps gs t r kk s ψn))
    (hss : OnS G ss Ψ) : OnS G (p :: ss) Φ := by
  intro n rs gs h hI Pr hS hA
  rcases n with _ | n
  · exact absurd hS (List.cons_ne_nil _ _)
  rcases Pr with ⟨_ | ⟨tab, rep, bb, k⟩, τ⟩
  · exact absurd hS (List.cons_ne_nil _ _)
  obtain ⟨ss', e, hS'⟩ := hS
  obtain ⟨e₁, rfl⟩ := List.cons.inj e
  dsimp only at e₁
  have := r.isLt
  have := rep.isLt
  have e₂ : tab = t := Fin.ext (by omega)
  have e₃ : rep = r := Fin.ext (by omega)
  subst tab rep
  exact round_ind out ok dflt pL opn dg dec one fin caps rootB pts nul j dz hnul h
    (hI.mono fun _ h => .send (some (⟨t, r, bb, k⟩ : SubmitJ J)) h)
    (fun _ gs' h' hI' => simRJ out ok dflt pL opn dg dec one fin caps rootB pts n _ gs' h' hI') τ (Φ gs) Ψ
    (hdead gs) (hstep gs) (fun c gs' h' hI' hA' => hss n _ gs' h' hI' (τ c) (hS' c) hA') (hA (some ⟨t, r, bb, k⟩))

theorem onSched_of (hfin : ∀ t r proofs, fin t r proofs = []) (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (ss : List ℕ)
    {Φ : (gs : Fin J → Fin 2 → Game P) → Strategy (posOf out caps gs) → Prop}
    (hΦ : ∀ gs σ, (∀ (t : Fin J) (r : Fin 2), 2 * t.val + r.val ∉ ss → (∀ a, gs t r ≠ .ret a) →
      ∀ x, Lock.Reach _ σ x → Dead x) → Φ gs σ) : OnS G ss Φ :=
  fun _ _ gs _ hI Pr hS hA => hΦ gs _ fun _ _ hn hg => dead_noAcc out caps nul j dz hnul _ hA Pr
    (noAcc_of_schedLJ out ok pL opn dg dec one fin caps rootB pts hfin hI hS hn hg)

variable {out ok dflt pL opn dg dec one fin caps rootB pts nul}

theorem onSched_mono {ss : List ℕ} {Φ Φ' : (gs : Fin J → Fin 2 → Game P) → Strategy (posOf out caps gs) → Prop}
    (h : OnS G ss Φ) (hΦ : ∀ gs σ, Φ gs σ → Φ' gs σ) : OnS G ss Φ' :=
  fun n rs gs h' hI Pr hS hA => hΦ _ _ (h n rs gs h' hI Pr hS hA)

theorem pred_cond {b : Option ℕ} {N N' : ℕ} (hN : N = N' + 1) (hb : b = none ∨ ∃ m, b = some m ∧ m < N) :
    predS b = none ∨ ∃ m, predS b = some m ∧ m < N' := by
  rcases hb with rfl | ⟨_ | m, rfl, hm⟩
  · exact .inl rfl
  · exact .inl rfl
  · exact .inr ⟨m, rfl, by omega⟩

omit [Nonempty Op] in
theorem paused_sess_dead (j : Fin J) (hD : DepAt j.val Dead) {b : Option ℕ} {nL : ℕ} {g₁ g₂ : Game P}
    (gs : Fin J → Fin 2 → Game P) (σ : Strategy (posOf out caps gs)) (hσ : ∀ x, Lock.Reach _ σ x → Dead x) :
    Paused b (Sess b nL (KgJ out j (caps j)) g₁ g₂) j Dead (List.ofFn (tabG out caps gs)) σ :=
  Lock.paused_dead (m := 0) (fun _ _ _ hd hl hσ => sess_dead' hd (List.ne_nil_of_length_pos hl) hσ) j.val hD
    (by simp) hσ

omit [Nonempty Op] in
theorem paused_front_dead (j : Fin J) (hD : DepAt j.val Dead) {b : Option ℕ} {k : ℕ}
    {S : (List OutT → Prop) → (l : List (Game OutT)) → Strategy (batch l) → Prop}
    (hS : ∀ D l σ, DepAt 0 D → l ≠ [] → (∀ x, Lock.Reach (batch l) σ x → D x) → S D l σ) (t : ℕ) (ns : List ℕ)
    (ht : t + ns.length = j.val) (gs : Fin J → Fin 2 → Game P) (σ : Strategy (posOf out caps gs))
    (hσ : ∀ x, Lock.Reach _ σ x → Dead x) : Paused b (Front k S ns) t Dead (List.ofFn (tabG out caps gs)) σ :=
  Lock.paused_dead (fun _ _ _ hd hl hσ => Lock.front_dead k hS ns hd hl hσ) t (ht ▸ hD)
    (by rw [List.length_ofFn, ht]; exact j.isLt) hσ

theorem onSched_spent (hfin : ∀ t r proofs, fin t r proofs = []) (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (hD : DepAt j.val Dead)
    (ss : List ℕ) (hss : ∀ p ∈ ss, 2 * j.val ≤ p) (a : P) (g₂ : Game P) :
    OnS G ss fun gs σ =>
      Paused none (Sess none 0 (KgJ out j (caps j)) (.ret a) g₂) j Dead (List.ofFn (tabG out caps gs)) σ :=
  onSched_of out ok dflt pL opn dg dec one fin caps rootB pts nul hfin j dz hnul ss fun gs σ hnd => by
    by_cases hall : ∀ i : Fin J, i.val < j.val → ∀ r, ∃ a, gs i r = .ret a
    · refine Lock.paused_sess_spent j σ (fun i hi hl => ?_) (by simp)
      have hiJ : i < J := by simpa using hl
      obtain ⟨a₀, h₀⟩ := hall ⟨i, hiJ⟩ hi 0
      obtain ⟨a₁, h₁⟩ := hall ⟨i, hiJ⟩ hi 1
      refine ⟨Op, fun o => .ret (out ⟨i, hiJ⟩ (caps ⟨i, hiJ⟩) (a₀, a₁) o), ?_⟩
      rw [List.getElem_ofFn]
      simp only [tabG, h₀, h₁]
      rfl
    · push Not at hall
      obtain ⟨i, hi, r, hr⟩ := hall
      have := r.isLt
      exact paused_sess_dead j hD gs σ (hnd i r (fun hm => by have := hss _ hm; omega) hr)

theorem onSched_rep1 (hfin : ∀ t r proofs, fin t r proofs = []) (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (hD : DepAt j.val Dead)
    (rest : List ℕ) (hrest : ∀ p ∈ rest, 2 * j.val + 2 ≤ p) (a : P) (g₂ : Game P) :
    ∀ c : ℕ, OnS G (List.replicate c (2 * j.val + 1) ++ rest) fun gs σ =>
      ∀ b : Option ℕ, (b = none ∨ ∃ m, b = some m ∧ m < c) →
        Paused b (Sess b 0 (KgJ out j (caps j)) (.ret a) g₂) j Dead (List.ofFn (tabG out caps gs)) σ := by
  have hss : ∀ c, ∀ p ∈ List.replicate c (2 * j.val + 1) ++ rest, 2 * j.val ≤ p := fun c p hp => by
    rcases List.mem_append.1 hp with hp | hp
    · rw [List.eq_of_mem_replicate hp]
      omega
    · have := hrest p hp
      omega
  intro c
  induction c with
  | zero =>
    intro n rs gs h hI Pr hS hA b hb
    obtain rfl : b = none := by
      rcases hb with hb | ⟨m, -, hm⟩
      · exact hb
      · omega
    exact onSched_spent hfin j dz hnul hD _ (hss 0) a g₂ n rs gs h hI Pr hS hA
  | succ c ih =>
    intro n rs gs h hI Pr hS hA b hb
    rcases b with _ | m
    · exact onSched_spent hfin j dz hnul hD _ (hss (c + 1)) a g₂ n rs gs h hI Pr hS hA
    · exact onSched_cons out ok dflt pL opn dg dec one fin caps rootB pts nul j dz hnul j 1 rfl
        (Φ := fun gs σ =>
          Paused (some m) (Sess (some m) 0 (KgJ out j (caps j)) (.ret a) g₂) j Dead (List.ofFn (tabG out caps gs)) σ)
        (Ψ := fun gs σ => ∀ b : Option ℕ, (b = none ∨ ∃ m, b = some m ∧ m < c) →
          Paused b (Sess b 0 (KgJ out j (caps j)) (.ret a) g₂) j Dead (List.ofFn (tabG out caps gs)) σ)
        (fun gs σ hσ => paused_sess_dead j hD gs σ hσ)
        (fun gs _ _ _ kk s ψn hN => sess_step1 out caps j kk s (some m) g₂ ψn fun x => hN x _ (pred_cond rfl hb))
        ih n rs gs h hI Pr hS hA

theorem onSched_rep0 (hfin : ∀ t r proofs, fin t r proofs = []) (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (hD : DepAt j.val Dead) (R : ℕ)
    (rest : List ℕ) (hrest : ∀ p ∈ rest, 2 * j.val + 2 ≤ p) :
    ∀ c : ℕ, OnS G (List.replicate c (2 * j.val) ++ (List.replicate R (2 * j.val + 1) ++ rest)) fun gs σ =>
      ∀ b : Option ℕ, (b = none ∨ ∃ m, b = some m ∧ m < c + R) →
        Paused b (Sess b c (KgJ out j (caps j)) (gs j 0) (gs j 1)) j Dead (List.ofFn (tabG out caps gs)) σ := by
  intro c
  induction c with
  | zero =>
    intro n rs gs h hI Pr hS hA b hb
    by_cases hr : ∃ a, gs j 0 = .ret a
    · obtain ⟨a, ha⟩ := hr
      rw [ha]
      exact onSched_rep1 hfin j dz hnul hD rest hrest a (gs j 1) R n rs gs h hI Pr hS hA b
        (by rwa [Nat.zero_add] at hb)
    · push Not at hr
      refine paused_sess_dead j hD gs _ (dead_noAcc out caps nul j dz hnul _ hA Pr
        (noAcc_of_schedLJ out ok pL opn dg dec one fin caps rootB pts hfin hI hS (t := j) (r := 0) ?_ hr))
      intro hm
      simp only [List.replicate_zero, List.nil_append, List.mem_append, List.mem_replicate] at hm
      rcases hm with ⟨-, hm⟩ | hm
      · simp at hm
      · have := hrest _ hm
        simp at this
  | succ c ih =>
    refine onSched_cons out ok dflt pL opn dg dec one fin caps rootB pts nul j dz hnul j 0 rfl
      (Ψ := fun gs σ => ∀ b : Option ℕ, (b = none ∨ ∃ m, b = some m ∧ m < c + R) →
        Paused b (Sess b c (KgJ out j (caps j)) (gs j 0) (gs j 1)) j Dead (List.ofFn (tabG out caps gs)) σ)
      (fun gs σ hσ b _ => paused_sess_dead j hD gs σ hσ)
      (fun gs _ _ _ kk s ψn hN b hb => sess_step0 out caps j kk s b c ψn fun x => ?_) ih
    have h' := hN x (predS b) (pred_cond (by omega) hb)
    rwa [upd_self, upd_ne gs (kk x) (show ¬ (j = j ∧ (1 : Fin 2) = 0) by simp)] at h'

theorem onSched_front (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (hD : DepAt j.val Dead) {k : ℕ}
    {S : (List OutT → Prop) → (l : List (Game OutT)) → Strategy (batch l) → Prop}
    (hS : ∀ D l σ, DepAt 0 D → l ≠ [] → (∀ x, Lock.Reach (batch l) σ x → D x) → S D l σ) (t : Fin J) (r : Fin 2)
    {p : ℕ} (hp : p = 2 * t.val + r.val) (ns : List ℕ) (hlen : t.val + (ns.length + 1) = j.val) (g₁ g₂ : Game P)
    (M off : ℕ) (ss : List ℕ)
    (hK : OnS G ss fun gs σ => gs j 0 = g₁ → gs j 1 = g₂ →
      Paused (some (M + off)) (Front k S (off :: ns)) t Dead (List.ofFn (tabG out caps gs)) σ) :
    ∀ c : ℕ, OnS G (List.replicate c p ++ ss) fun gs σ => gs j 0 = g₁ → gs j 1 = g₂ →
      Paused (some (M + (off + c))) (Front k S ((off + c) :: ns)) t Dead (List.ofFn (tabG out caps gs)) σ := by
  have hne : ∀ r' : Fin 2, ¬ (j = t ∧ r' = r) := fun r' h => by
    have := congrArg Fin.val h.1
    omega
  intro c
  induction c with
  | zero => exact hK
  | succ c ih =>
    exact onSched_cons out ok dflt pL opn dg dec one fin caps rootB pts nul j dz hnul t r hp
      (Ψ := fun gs σ => gs j 0 = g₁ → gs j 1 = g₂ →
        Paused (some (M + (off + c))) (Front k S ((off + c) :: ns)) t Dead (List.ofFn (tabG out caps gs)) σ)
      (fun gs σ hσ _ _ => paused_front_dead j hD hS t ((off + (c + 1)) :: ns) hlen gs σ hσ)
      (fun gs _ _ _ kk s ψn hN e₁ e₂ => front_step out caps t r kk s (M + (off + c)) (off + c) ns ψn fun x =>
        hN x ((upd_ne gs (kk x) (hne 0)).trans e₁) ((upd_ne gs (kk x) (hne 1)).trans e₂)) ih

def fromS (Rf : ℕ → ℕ) : ℕ → ℕ → List ℕ
  | _, 0 => []
  | t, d + 1 => List.replicate (Rf t) (2 * t) ++ (List.replicate (Rf t) (2 * t + 1) ++ fromS Rf (t + 1) d)

def nsFrom (Rf : ℕ → ℕ) : ℕ → ℕ → List ℕ
  | _, 0 => []
  | t, d + 1 => (Rf t + Rf t) :: nsFrom Rf (t + 1) d

theorem length_nsFrom (Rf : ℕ → ℕ) : ∀ t d, (nsFrom Rf t d).length = d
  | _, 0 => rfl
  | t, d + 1 => congrArg (· + 1) (length_nsFrom Rf (t + 1) d)

theorem onSched_tabs (hfin : ∀ t r proofs, fin t r proofs = []) (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (hD : DepAt j.val Dead)
    (Rf : ℕ → ℕ) (k R : ℕ) (hk : k < R + R) (rest : List ℕ) (hrest : ∀ p ∈ rest, 2 * j.val + 2 ≤ p)
    (g₁ g₂ : Game P) :
    ∀ d t, t + d = j.val →
      OnS G (fromS Rf t d ++ (List.replicate R (2 * j.val) ++ (List.replicate R (2 * j.val + 1) ++ rest)))
        fun gs σ => gs j 0 = g₁ → gs j 1 = g₂ →
          Paused (some ((nsFrom Rf t d).sum + k)) (Front k (Sess (some k) R (KgJ out j (caps j)) g₁ g₂)
            (nsFrom Rf t d)) t Dead (List.ofFn (tabG out caps gs)) σ := by
  intro d
  induction d with
  | zero =>
    intro t ht n rs gs h hI Pr hS hA e₁ e₂
    obtain rfl : t = j.val := by simpa using ht
    subst e₁ e₂
    show Paused (some (0 + k)) _ _ _ _ _
    rw [Nat.zero_add]
    exact onSched_rep0 hfin j dz hnul hD R rest hrest R n rs gs h hI Pr hS hA (some k) (.inr ⟨k, rfl, hk⟩)
  | succ d ih =>
    intro t ht
    have htJ : t < J := by
      have := j.isLt
      omega
    have hS : ∀ D l σ, DepAt 0 D → l ≠ [] → (∀ x, Lock.Reach (batch l) σ x → D x) →
        Sess (some k) R (KgJ out j (caps j)) g₁ g₂ D l σ := fun _ _ _ hd hl hσ => sess_dead' hd hl hσ
    have hlen : (⟨t, htJ⟩ : Fin J).val + ((nsFrom Rf (t + 1) d).length + 1) = j.val := by
      dsimp only
      rw [length_nsFrom]
      omega
    have h0 := onSched_front j dz hnul hD hS ⟨t, htJ⟩ 0 (p := 2 * t) rfl (nsFrom Rf (t + 1) d) hlen g₁ g₂
      ((nsFrom Rf (t + 1) d).sum + k) (Rf t) _
      (onSched_mono (onSched_front j dz hnul hD hS ⟨t, htJ⟩ 1 (p := 2 * t + 1) rfl (nsFrom Rf (t + 1) d) hlen g₁ g₂
        ((nsFrom Rf (t + 1) d).sum + k) 0 _
        (onSched_mono (ih (t + 1) (by omega)) fun _ _ hP e₁ e₂ => Lock.paused_front_zero k _ _ t (hP e₁ e₂)) (Rf t))
        fun _ _ hP e₁ e₂ => by simpa only [Nat.zero_add] using hP e₁ e₂) (Rf t)
    simp only [fromS, List.append_assoc]
    refine onSched_mono h0 fun gs σ hP e₁ e₂ => ?_
    have h' := hP e₁ e₂
    rwa [show (nsFrom Rf (t + 1) d).sum + k + (Rf t + Rf t) = (nsFrom Rf t (d + 1)).sum + k by
      simp only [nsFrom, List.sum_cons]
      omega] at h'

theorem front_simRJ (hfin : ∀ t r proofs, fin t r proofs = []) (j : Fin J) (dz : OutT)
    (hnul : ∀ outs cap pend, outs.getD j dz = out j cap pend nul → Dead outs) (hD : DepAt j.val Dead)
    (Rf : ℕ → ℕ) (k R : ℕ) (hk : k < R + R) (rest : List ℕ) (hrest : ∀ p ∈ rest, 2 * j.val + 2 ≤ p) (n : ℕ)
    (rs : Fin J → Fin 2 → List Flock.Round) (gs : Fin J → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn caps ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs))
    (Pr : Strategy ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)))
    (hS : SchedLJ dg rootB pts n rs
      (fromS Rf 0 j ++ (List.replicate R (2 * j.val) ++ (List.replicate R (2 * j.val + 1) ++ rest))) Pr)
    (hA : Sim.All (fun b _ => pL b) (DeadAt out G)
      (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI)) :
    Front k (Sess (some k) R (KgJ out j (caps j)) (gs j 0) (gs j 1)) (nsFrom Rf 0 j) Dead
      (List.ofFn (tabG out caps gs))
      ((Sim.mapDead (answerWith nul) (simRJ out ok dflt pL opn dg dec one fin caps rootB pts n rs gs h hI)).strategy
        Pr) :=
  onSched_tabs hfin j dz hnul hD Rf k R hk rest hrest (gs j 0) (gs j 1) j.val 0 (Nat.zero_add _) n rs gs h hI Pr hS hA
    rfl rfl

end SchedJ

end FlockDeploy.KS
