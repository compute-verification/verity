import Proofs.Flock.Claims.KS.BatchMove

/-!
# The simulated strategy of `J` tables follows the batch's schedule

The simulation of a session of `J` tables (`simRJ`) follows the live prover round by round, each round moving one rep
of one table (`ctxAt`). When the live prover submits its rounds stream by stream (`SchedLJ`), the tables in front of
table `j` first, each rep 0 then rep 1, then table `j`'s, the simulated strategy, with every dead branch answering the
openings with `nul`, follows the batch's schedule (`Lock.Front` over the tables in front, then `Lock.Sess` at table
`j`), short of runs `Dead` reads at table `j` alone:

* `tr_list`, `round_heq_gen`, `ofFn_tabG_upd₀`, `ofFn_tabG_upd₁`, `step0`, `step1`: a round through `ctxAt` is a
  round through `Ctx.batchAt` at the table's position;
* `front_step`, `sess_step0`, `sess_step1`: so a round at a table in front, and at the session's table's rep 0 and
  rep 1, keeps the schedule one coin further on.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Tables Claims.Bridge
open FlockSoundness.Game.Lock (WaitR FollowLP Paused Front Sess consP headS deadT FirstS predS DepAt)

local notation "𝔽" => FlockLevel3.GF128

section Batch

variable {ζ : Type}

/-- A predicate of a batch's strategy read along an equality of the batch's games and a strategy equal to it. -/
theorem tr_list (Φ : (l : List (Game ζ)) → Strategy (batch l) → Prop) {l l' : List (Game ζ)} (e : l = l')
    {σ : Strategy (batch l)} {σ' : Strategy (batch l')} (h : HEq σ σ') (hΦ : Φ l' σ') : Φ l σ := by
  subst e
  cases h
  exact hΦ

theorem ofFn_split {J : ℕ} (f : Fin J → Game ζ) (j : Fin J) :
    List.ofFn f = (List.ofFn f).take j ++ f j :: (List.ofFn f).drop (j + 1) := by
  rw [← ofFn_update, Function.update_eq_self]

/-- **A round through a table's context at a position of a batch over `Fin J`**, read through `Ctx.batchAt` there:
the table's path to its coin, then the coin. -/
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

/-- `paused_round` with the paused count given. -/
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

end Batch

/-! ## A round of the simulation at a position of the batch -/

section Round

variable {P OutT Cp Op : Type} [Nonempty Op] {J : ℕ} (out : Fin J → Cp → P × P → Op → OutT) (caps : Fin J → Cp)

/-- The batch's context at table `t`'s position. -/
abbrev ctxB (gs : Fin J → Fin 2 → Game P) (t : Fin J) :=
  Ctx.batchAt ((List.ofFn (tabG out caps gs)).take t) ((List.ofFn (tabG out caps gs)).drop (t + 1))

/-- Table `t`'s context at rep 0. -/
abbrev ctx₀ (gs : Fin J → Fin 2 → Game P) (t : Fin J) :=
  (Ctx.left (gs t 1)).comp (Ctx.bind (KgJ out t (caps t)))

/-- Table `t`'s context at rep 1. -/
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

theorem round_heq_ofEq {A : Type} {C : Game A → Game (List OutT)} (c : Ctx C) {g : Game (List OutT)} {g₁ : Game A}
    (e₁ : g = C g₁) {Cm : Type} [Fintype Cm] [Nonempty Cm] (kk : Cm → Game A) (s : Steps g₁ (.coin Cm kk))
    (ψ : Strategy (.coin Cm (c.coin kk).next)) :
    HEq (playSteps ((Steps.ofEq e₁).trans (c.lift s)) (playSteps (c.coin kk).steps ψ))
      (playSteps (c.lift s) (playSteps (c.coin kk).steps ψ)) := by
  rw [playSteps_trans]
  exact playSteps_ofEq_heq _ _

/-- **A round at rep 0 of table `t`**, read through `Ctx.batchAt` at the table's position: a predicate `Φ` there
from `Φ'` at each coin's next position. -/
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

/-- **A round at rep 1 of table `t`**, read through `Ctx.batchAt` at the table's position. -/
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

/-- **A round at a table in front**, at either rep: the paused games keep waiting, and the table advances one coin. -/
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

/-- **A round at rep 0 of the session's table**: the paused games keep waiting, and rep 0 leads one coin. -/
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

/-- **A round at rep 1 of the session's table** once rep 0 has ended: the paused games keep waiting. -/
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

end FlockDeploy.KS
