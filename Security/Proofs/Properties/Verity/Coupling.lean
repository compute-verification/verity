module

public import Properties.Verity.Coupling
public import Properties.Verity.Idle
public import Proofs.Properties.Verity.Subprotocols
public import Proofs.Core.Game.BatchInduced
public import Proofs.Core.Blocks
public import Proofs.Pouw.Prob

/-!
# The causal game's coins, flipped up front: the coupling and the bounds it carries

`play_couples` proves the coupling (`Verity.PlayCouples`, a lemma every guarantee here rests on) by induction over the
device's recursion (`releases`), from every state whose families' coins left `fs` the coin commitment's families `fs₀`
reach (`Coupling.Shape`), for every continuation:

* at a fresh coin, the coin the device samples is the coin of `t` its family's coins left name, and the rest of the play
  reads no coin of that family it has read already (`Coupling.outK_congr`, over `Coupling.Agree`), so sampling it apart is
  sampling `t` (`Coupling.avg_lazy`). Each coin's continuation resolves the coin trees' coins after its own release, so
  one resolution serves them all, glued by the release that extends the log (`Coupling.AgreeFrom`);
* at a coin tree's `Hello` or an opening, the device samples the tree's sample or the opened leaf's sample, and the
  resolution takes the one at which the continuation's probability is largest (`avg_le_of_max`), so the play's
  probability is at most that resolution's over the coins flipped up front;
* where a tree-line guard fails, the releases end in the play and at the coins flipped up front alike
  (`Coupling.prob_ifPos`, `Coupling.prob_ifNeg`).

The device's sample at the close is the last coordinate of the coins flipped up front (`Coupling.avg_prod`). The lower half is the upper half of the complement (`Coupling.prob_add_not`). From it: `coins_couple`
(`Verity.CoinsCouple`); `target_of_periods` (`Verity.TargetOfPeriods`: `PeriodRun.Sound` at the coupled run's own
coordinate, its verdict and claim carried by the coupling, and the rest of the event by `BreakBound` and the forger's
`Unforgeable`); `wipe_target` (`Verity.WipeTarget`, through `Wipe.OfRun.prob_kept_le`); and `soundness_of_upFront`
(`Verity.SoundnessOfUpFront`, through `soundness_at_of_targets`). Each guarantee's witnesses are at the idle deployment
(`Properties.Verity.Idle`).
-/

@[expose] public section

namespace Verity

open Accounting
open scoped ENNReal

namespace Coupling

open Game

/-! ## Averages -/

theorem avg_eq_const {C : Type} [Fintype C] [Nonempty C] {f : C → ℝ≥0∞} {x : ℝ≥0∞} (h : ∀ c, f c = x) :
    avg f = x := by
  rw [show f = fun _ => x from funext h, avg_const]

theorem avg_equiv {C D : Type} [Fintype C] [Fintype D] (e : C ≃ D) (f : D → ℝ≥0∞) :
    avg (fun c => f (e c)) = avg f := by
  unfold avg
  rw [Fintype.card_congr e]
  exact Fintype.sum_equiv e _ _ (fun _ => rfl)

theorem avg_prod {C D : Type} [Fintype C] [Fintype D] (f : C × D → ℝ≥0∞) :
    avg f = avg fun c => avg fun d => f (c, d) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  simp only [Finset.mul_sum, mul_assoc]

/-- **Lazy sampling**: sampling a coordinate `π t` of a uniform `t` and reading the rest of `t` where the coordinate is
not read is sampling the coordinate apart. -/
theorem avg_lazy {T X : Type} [Fintype T] [Fintype X] [Nonempty X] (π : T → X) (upd : T → X → T)
    (h1 : ∀ t x, π (upd t x) = x) (h2 : ∀ t, upd t (π t) = t) (h3 : ∀ t x y, upd (upd t x) y = upd t y)
    (Ψ : X → T → ℝ≥0∞) (hΨ : ∀ x t y, Ψ x (upd t y) = Ψ x t) :
    avg (fun t => Ψ (π t) t) = avg fun x => avg fun t => Ψ x t := by
  let φ : X × T ≃ X × T :=
    { toFun := fun p => (π p.2, upd p.2 p.1), invFun := fun p => (π p.2, upd p.2 p.1),
      left_inv := fun p => by simp [h1, h3, h2], right_inv := fun p => by simp [h1, h3, h2] }
  calc avg (fun t => Ψ (π t) t) = avg fun _ : X => avg fun t => Ψ (π t) t := (avg_const _).symm
    _ = avg fun p : X × T => Ψ (π p.2) (upd p.2 p.1) := by rw [avg_prod]; simp only [hΨ]
    _ = avg fun p : X × T => Ψ (φ p).1 (φ p).2 := rfl
    _ = avg fun p : X × T => Ψ p.1 p.2 := avg_equiv φ (fun p => Ψ p.1 p.2)
    _ = _ := avg_prod _

/-! ## The coins flipped up front, as the device takes them -/

/-- `t` with coin `n` of the first family labelled `l` set to `x`. -/
def _root_.Verity.UpFront.set : {fs : List (String × ℕ)} → UpFront fs → String → ℕ → (Fin 512 → Bool) → UpFront fs
  | [], t, _, _, _ => t
  | (l', _) :: _, t, l, n, x =>
    if l' = l then ((fun m => if m.val = n then x else t.1 m), t.2) else (t.1, UpFront.set t.2 l n x)

/-- `fs` has the families of `fs₀`, in order, each with at most its coins. -/
def Shape : List (String × ℕ) → List (String × ℕ) → Prop
  | [], [] => True
  | (l₀, n₀) :: fs₀, (l, n) :: fs => l = l₀ ∧ n ≤ n₀ ∧ Shape fs₀ fs
  | _, _ => False

/-- `t` and `t'` agree on every coin the families `fs` have left. -/
def Agree : (fs₀ fs : List (String × ℕ)) → UpFront fs₀ → UpFront fs₀ → Prop
  | [], _, _, _ => True
  | (_, _) :: _, [], t, t' => t = t'
  | (_, _) :: fs₀, (_, c) :: fs, t, t' => (∀ m : Fin _, m.val < c → t.1 m = t'.1 m) ∧ Agree fs₀ fs t.2 t'.2

theorem shape_refl : ∀ fs : List (String × ℕ), Shape fs fs
  | [] => trivial
  | (_, _) :: fs => ⟨rfl, le_rfl, shape_refl fs⟩

theorem agree_refl : ∀ (fs₀ fs : List (String × ℕ)) (t : UpFront fs₀), Agree fs₀ fs t t
  | [], _, _ => trivial
  | (_, _) :: _, [], _ => rfl
  | (_, _) :: fs₀, (_, _) :: fs, t => ⟨fun _ _ => rfl, agree_refl fs₀ fs t.2⟩

theorem read_set : ∀ {fs₀ : List (String × ℕ)} (t : UpFront fs₀) (l : String) (n : ℕ) (x : Fin 512 → Bool),
    n < coinsLeft fs₀ l → (t.set l n x).read l n = x
  | [], _, _, _, _, h => absurd h (Nat.not_lt_zero _)
  | (l', c) :: fs₀, t, l, n, x, h => by
    by_cases hl : l' = l
    · simp only [coinsLeft, hl, ite_true] at h
      simp [UpFront.set, UpFront.read, hl, h]
    · simp only [coinsLeft, hl, ite_false] at h
      simp only [UpFront.set, UpFront.read, hl, ite_false]
      exact read_set t.2 l n x h

theorem set_read : ∀ {fs₀ : List (String × ℕ)} (t : UpFront fs₀) (l : String) (n : ℕ),
    t.set l n (t.read l n) = t
  | [], _, _, _ => rfl
  | (l', c) :: fs₀, t, l, n => by
    by_cases hl : l' = l
    · simp only [UpFront.set, UpFront.read, hl, ite_true]
      refine Prod.ext (funext fun m => ?_) rfl
      by_cases hm : m.val = n
      · subst hm; simp
      · simp [hm]
    · simp only [UpFront.set, UpFront.read, hl, ite_false]
      exact Prod.ext rfl (set_read t.2 l n)

theorem set_set : ∀ {fs₀ : List (String × ℕ)} (t : UpFront fs₀) (l : String) (n : ℕ) (x y : Fin 512 → Bool),
    (t.set l n x).set l n y = t.set l n y
  | [], _, _, _, _, _ => rfl
  | (l', c) :: fs₀, t, l, n, x, y => by
    by_cases hl : l' = l
    · simp only [UpFront.set, hl]
      refine Prod.ext (funext fun m => ?_) rfl
      by_cases hm : m.val = n <;> simp [hm]
    · simp only [UpFront.set, hl]
      exact Prod.ext rfl (set_set t.2 l n x y)

theorem coinsLeft_le : ∀ {fs₀ fs : List (String × ℕ)} (l : String), Shape fs₀ fs → coinsLeft fs l ≤ coinsLeft fs₀ l
  | [], [], _, _ => le_rfl
  | (l₀, n₀) :: fs₀, (l₁, n₁) :: fs, l, ⟨h1, h2, h3⟩ => by
    subst h1
    by_cases hl : l₁ = l
    · simp [coinsLeft, hl, h2]
    · simp only [coinsLeft, hl, ite_false]
      exact coinsLeft_le l h3

theorem coinsLeft_take : ∀ {fs fs' : List (String × ℕ)} (l : String), takeCoin l fs = some fs' →
    coinsLeft fs' l + 1 = coinsLeft fs l
  | [], _, _, h => by simp [takeCoin] at h
  | (l', c) :: fs, fs', l, h => by
    by_cases hl : l' = l
    · simp only [takeCoin, hl, ite_true] at h
      cases c with
      | zero => simp at h
      | succ c => simp only [Option.some.injEq] at h; subst h; simp [coinsLeft, hl]
    · simp only [takeCoin, hl, ite_false] at h
      obtain ⟨fs'', h', rfl⟩ := Option.map_eq_some_iff.1 h
      simp only [coinsLeft, hl, ite_false]
      exact coinsLeft_take l h'

theorem shape_take : ∀ {fs₀ fs fs' : List (String × ℕ)} (l : String), Shape fs₀ fs → takeCoin l fs = some fs' →
    Shape fs₀ fs'
  | [], [], _, _, _, h => by simp [takeCoin] at h
  | (l₀, n₀) :: fs₀, (l₁, n₁) :: fs, fs', l, ⟨h1, h2, h3⟩, h => by
    by_cases hl : l₁ = l
    · simp only [takeCoin, hl, ite_true] at h
      cases n₁ with
      | zero => simp at h
      | succ c => simp only [Option.some.injEq] at h; subst h; exact ⟨hl ▸ h1, by omega, h3⟩
    · simp only [takeCoin, hl, ite_false] at h
      obtain ⟨fs'', h', rfl⟩ := Option.map_eq_some_iff.1 h
      exact ⟨h1, h2, shape_take l h3 h'⟩

theorem agree_take : ∀ {fs₀ fs fs' : List (String × ℕ)} (l : String) {t t' : UpFront fs₀}, Shape fs₀ fs →
    takeCoin l fs = some fs' → Agree fs₀ fs t t' → Agree fs₀ fs' t t'
  | [], [], _, _, _, _, _, h, _ => by simp [takeCoin] at h
  | (l₀, n₀) :: fs₀, (l₁, n₁) :: fs, fs', l, t, t', ⟨h1, h2, h3⟩, h, ⟨a1, a2⟩ => by
    by_cases hl : l₁ = l
    · simp only [takeCoin, hl, ite_true] at h
      cases n₁ with
      | zero => simp at h
      | succ c => simp only [Option.some.injEq] at h; subst h; exact ⟨fun m hm => a1 m (by omega), a2⟩
    · simp only [takeCoin, hl, ite_false] at h
      obtain ⟨fs'', h', rfl⟩ := Option.map_eq_some_iff.1 h
      exact ⟨a1, agree_take l h3 h' a2⟩

theorem read_congr : ∀ {fs₀ fs : List (String × ℕ)} {t t' : UpFront fs₀} (l : String) (n : ℕ), Shape fs₀ fs →
    Agree fs₀ fs t t' → n < coinsLeft fs l → t.read l n = t'.read l n
  | [], [], _, _, _, _, _, _, h => absurd h (Nat.not_lt_zero _)
  | (l₀, n₀) :: fs₀, (l₁, n₁) :: fs, t, t', l, n, ⟨h1, h2, h3⟩, ⟨a1, a2⟩, h => by
    subst h1
    by_cases hl : l₁ = l
    · simp only [coinsLeft, hl, ite_true] at h
      simp only [UpFront.read, hl, ite_true, lt_of_lt_of_le h h2, dite_true]
      exact a1 _ h
    · simp only [coinsLeft, hl, ite_false] at h
      simp only [UpFront.read, hl, ite_false]
      exact read_congr l n h3 a2 h

theorem agree_set : ∀ {fs₀ fs : List (String × ℕ)} (t : UpFront fs₀) (l : String) (n : ℕ) (x : Fin 512 → Bool),
    Shape fs₀ fs → coinsLeft fs l ≤ n → Agree fs₀ fs t (t.set l n x)
  | [], [], _, _, _, _, _, _ => trivial
  | (l₀, n₀) :: fs₀, (l₁, n₁) :: fs, t, l, n, x, ⟨h1, h2, h3⟩, h => by
    subst h1
    by_cases hl : l₁ = l
    · simp only [coinsLeft, hl, ite_true] at h
      simp only [Agree, UpFront.set, hl]
      exact ⟨fun m hm => by simp [show m.val ≠ n by omega], agree_refl _ _ _⟩
    · simp only [coinsLeft, hl, ite_false] at h
      simp only [Agree, UpFront.set, hl]
      exact ⟨fun _ _ => rfl, agree_set t.2 l n x h3 h⟩


/-! ## The play at the coins flipped up front reads only the coins left and the coin trees' coins from its log on -/

/-- `ρ` and `ρ'` resolve every coin tree's coins alike after every log that extends `pre`. -/
def AgreeFrom (pre : List Release) (ρ ρ' : TreeCoins) : Prop :=
  ∀ w, pre <+: w → ρ.hello w = ρ'.hello w ∧ ρ.opening w = ρ'.opening w

theorem AgreeFrom.mono {pre pre' : List Release} {ρ ρ' : TreeCoins} (h : AgreeFrom pre ρ ρ') (hp : pre <+: pre') :
    AgreeFrom pre' ρ ρ' := fun w hw => h w (hp.trans hw)

theorem exists_le_of_eq {X : Type} {a b : ℝ≥0∞} {g : X → ℝ≥0∞} (e : a = b) (h : ∃ x, b ≤ g x) :
    ∃ x, a ≤ g x := e ▸ h

theorem outK_congr (fs₀ : List (String × ℕ)) {β : Type} : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Release) (f : List Release → Game β) (σ : Game.Strategy ((releases k fs ts).bind f))
    (t t' : UpFront fs₀) (ρ ρ' : TreeCoins), Shape fs₀ fs → Agree fs₀ fs t t' → AgreeFrom pre ρ ρ' →
      outK fs₀ t ρ k fs ts pre f σ = outK fs₀ t' ρ' k fs ts pre f σ
  | 0, _, _, _, _, _, _, _, _, _, _, _, _ => rfl
  | k + 1, fs, ts, pre, f, σ, t, t', ρ, ρ', hs, ha, hρ => by
    conv_lhs => rw [outK.eq_def]
    conv_rhs => rw [outK.eq_def]
    rcases σ with ⟨_ | ⟨⟨l⟩ | ⟨spec⟩ | ⟨i, s, n⟩, b⟩, σ⟩
    · rfl
    · dsimp only at σ ⊢
      revert σ
      generalize hc : takeCoin l fs = c
      cases c with
      | none => intro σ; rfl
      | some fs' =>
        intro σ
        have hlt : coinsLeft fs' l < coinsLeft fs l := by have := coinsLeft_take l hc; omega
        dsimp only
        rw [read_congr l _ hs ha hlt]
        rw [outK_congr fs₀ k fs' ts _ _ _ t t' ρ ρ' (shape_take l hs hc) (agree_take l hs hc ha)
          (hρ.mono (List.prefix_append _ _))]
    · dsimp only
      by_cases hc : treeLines ts < maxTreeLines ∧ TreeSpecOk spec
      · simp only [dite_eq_left hc]
        generalize Game.Strategy.ifPos hc σ = σ'
        rw [show ρ.hello pre spec b = ρ'.hello pre spec b from congrFun (congrFun (hρ pre List.prefix_rfl).1 spec) b]
        rw [outK_congr fs₀ k fs _ _ _ _ t t' ρ ρ' hs ha (hρ.mono (List.prefix_append _ _))]
      · simp only [dite_eq_right hc]
    · dsimp only at σ ⊢
      by_cases hc : treeLines ts < maxTreeLines
      swap
      · simp only [dite_eq_right hc]
      simp only [dite_eq_left hc]
      generalize Game.Strategy.ifPos hc σ = σ'
      clear σ
      revert σ'
      generalize ts[i]? = oT
      cases oT with
      | none => intro σ; rfl
      | some T =>
        intro σ
        dsimp only at σ ⊢
        revert σ
        generalize T.next s n = oN
        cases oN with
        | none => intro σ; rfl
        | some p =>
          obtain ⟨s', r'⟩ := p
          intro σ
          dsimp only
          rw [show ρ.opening pre i s n b T s' r' = ρ'.opening pre i s n b T s' r' by
            rw [(hρ pre List.prefix_rfl).2]]
          rw [outK_congr fs₀ k fs _ _ _ _ t t' ρ ρ' hs ha (hρ.mono (List.prefix_append _ _))]

/-! ## Probabilities over the coins -/

open Classical in
theorem prCoin_equiv {C D : Type} [Fintype C] [Fintype D] (e : C ≃ D) (p : D → Prop) :
    prCoin (fun c => p (e c)) = prCoin p :=
  avg_equiv e fun d => if p d then 1 else 0

open Classical in
theorem prCoin_or_le {C : Type} [Fintype C] (p q : C → Prop) :
    prCoin (fun c => p c ∨ q c) ≤ prCoin p + prCoin q := by
  unfold prCoin
  rw [← avg_add]
  refine avg_mono fun c => ?_
  by_cases hp : p c
  · simp only [hp, true_or, ite_true]
    exact le_self_add
  · by_cases hq : q c <;> simp [hp, hq]

/-- **A bound at one coordinate.** If, at every value of the other coins, the probability over coordinate `i` of `E`
is at most `δ`, so is the probability over all the coins. -/
theorem pr_le_of_update {I : Type} [Fintype I] [DecidableEq I] {Ω : I → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (i : I) (E : ((j : I) → Ω j) → Prop) {δ : ℝ}
    (h : ∀ ξ, Pouw.pr (fun x : Ω i => E (Function.update ξ i x)) ≤ δ) : Pouw.pr E ≤ δ := by
  let e : ((j : I) → Ω j) ≃ ((j : { j // j ≠ i }) → Ω j) × Ω i :=
    (Equiv.piSplitAt i Ω).trans (Equiv.prodComm _ _)
  obtain ⟨x₀⟩ : Nonempty (Ω i) := inferInstance
  rw [← Pouw.pr_equiv e.symm E]
  refine Pouw.pr_prod_le _ δ ((Pouw.pr_nonneg _).trans (h (Classical.arbitrary _))) fun g => ?_
  have key : (fun x => E (e.symm (g, x))) = fun x => E (Function.update (e.symm (g, x₀)) i x) := by
    funext x
    congr 1
    funext j
    by_cases hj : j = i
    · subst hj
      simp [e, Equiv.piSplitAt_symm_apply]
    · simp [e, Equiv.piSplitAt_symm_apply, hj]
  exact (congrArg Pouw.pr key).trans_le (h _)

theorem prCoin_eq_ofReal_pr {C : Type} [Fintype C] (p : C → Prop) :
    prCoin p = ENNReal.ofReal (Pouw.pr p) := by
  classical
  rw [prCoin_eq_card p]
  unfold Pouw.pr Claims.Pr Claims.prUniform
  rcases (Fintype.card C).eq_zero_or_pos with h | h
  · have := Fintype.card_eq_zero_iff.mp h
    simp
  · rw [ENNReal.ofReal_div_of_pos (by exact_mod_cast h), ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]

/-- An event of a value every play's outcome is mapped to has the probability of that value's event under a sample of a
type with one value. -/
theorem prob_const_eq {O X : Type} [Fintype X] [Nonempty X] (P : X → Prop) (x : X) (hX : ∀ y, y = x)
    (G : Game O) (t : Strategy G) : prob (fun _ => P x) G t = value P (draw X) := by
  rw [show draw X = .coin X fun y => .ret (id y) from rfl, value_coin_ret]
  by_cases hp : P x
  · rw [show (fun _ : O => P x) = fun _ => True from funext fun _ => propext (iff_true_intro hp), prob_true,
      show (fun y : X => P (id y)) = fun _ => True from funext fun y => propext (iff_true_intro (hX y ▸ hp))]
    simp [prCoin, avg_const]
  · rw [show (fun _ : O => P x) = fun _ => False from funext fun _ => propext (iff_false_intro hp), prob_false,
      show (fun y : X => P (id y)) = fun _ => False from funext fun y => propext (iff_false_intro (hX y ▸ hp)),
      prCoin_false]


theorem prob_assocMap {α β γ : Type} (p : γ → Prop) (h : α → β) (f : β → Game γ) :
    ∀ (G : Game α) (s : Strategy ((G.map h).bind f)),
      prob p ((G.map h).bind f) s = prob p (G.bind fun a => f (h a)) (Strategy.assocMap h f G s)
  | .ret _, _ => rfl
  | .send _ k, s => prob_assocMap p h f (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      show avg (fun c => prob p (((k c).map h).bind f) (s c)) = avg fun c => prob p _ _
      exact congrArg avg (funext fun c => prob_assocMap p h f (k c) (s c))

theorem prob_ifPos {β γ : Type} (p : γ → Prop) (f : β → Game γ) {c : Prop} [h : Decidable c] {A B : Game β}
    (hc : c) (s : Strategy ((if c then A else B).bind f)) :
    prob p ((if c then A else B).bind f) s = prob p (A.bind f) (Strategy.ifPos hc s) := by
  cases h with
  | isTrue _ => rfl
  | isFalse hn => exact absurd hc hn

theorem prob_ifNeg {β γ : Type} (p : γ → Prop) (f : β → Game γ) {c : Prop} [h : Decidable c] {A B : Game β}
    (hc : ¬ c) (s : Strategy ((if c then A else B).bind f)) :
    prob p ((if c then A else B).bind f) s = prob p (B.bind f) (Strategy.ifNeg hc s) := by
  cases h with
  | isTrue hp => exact absurd hp hc
  | isFalse _ => rfl

open Classical in
theorem prCoin_add_not {C : Type} [Fintype C] [Nonempty C] (p : C → Prop) :
    prCoin p + prCoin (fun c => ¬ p c) = 1 := by
  unfold prCoin
  rw [← avg_add]
  exact avg_eq_const fun c => by by_cases h : p c <;> simp [h]

theorem prob_add_not {α : Type} (p : α → Prop) :
    ∀ (g : Game α) (s : Strategy g), prob p g s + prob (fun a => ¬ p a) g s = 1
  | .ret a, _ => by
      classical
      simp only [prob]
      by_cases hp : p a <;> simp [hp]
  | .send _ k, s => prob_add_not p (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      rw [prob_coin, prob_coin, ← avg_add]
      exact avg_eq_const fun c => prob_add_not p (k c) (s c)

/-- Some resolution of the coin trees' coins. -/
def TreeCoins.any : TreeCoins where
  hello _ _ _ := (fun _ => false, fun _ _ => (fun _ _ => false, fun _ => false))
  opening _ _ _ _ _ T s' r := ⟨T.sample.2 s' r, rfl⟩

/-- An opening's place in the device's log: everything its coins' type reads. -/
structure OKey where
  i : ℕ
  s : ℕ
  n : ℕ
  b : ByteArray
  T : OpenTree
  s' : Fin T.spec.streams.size
  r : Fin (2 ^ T.spec.roundsLog)

theorem agreeFrom_refl (pre : List Release) (ρ : TreeCoins) : AgreeFrom pre ρ ρ := fun _ _ => ⟨rfl, rfl⟩

theorem ne_of_prefix {pre w : List Release} {r : Release} (h : pre ++ [r] <+: w) : w ≠ pre := by
  rintro rfl
  have := h.length_le
  simp at this

open Classical in
/-- **The play's probability is at most the coins flipped up front's**, at some resolution of the coin trees' coins:
from the device's state, for every continuation `f` and strategy. -/
theorem upper (fs₀ : List (String × ℕ)) {β : Type} (q : β → Prop) : ∀ (k : ℕ) (fs : List (String × ℕ))
    (ts : List OpenTree) (pre : List Release) (f : List Release → Game β)
    (σ : Game.Strategy ((releases k fs ts).bind f)), Shape fs₀ fs →
      ∃ ρ : TreeCoins, prob q ((releases k fs ts).bind f) σ ≤
        avg fun t : UpFront fs₀ => prob q (f (outK fs₀ t ρ k fs ts pre f σ).1) (outK fs₀ t ρ k fs ts pre f σ).2
  | 0, _, _, _, _, _, _ => ⟨TreeCoins.any, (avg_eq_const fun _ => rfl).ge⟩
  | k + 1, fs, ts, pre, f, σ, hs => by
    conv => enter [1, ρ, 2, 1, t]; rw [outK.eq_def]
    rcases σ with ⟨_ | ⟨⟨l⟩ | ⟨spec⟩ | ⟨i, s, n⟩, b⟩, σ⟩
    · exact ⟨TreeCoins.any, (avg_eq_const fun _ => rfl).ge⟩
    · change ∃ ρ, prob q _ σ ≤ _
      dsimp only at σ ⊢
      revert σ
      generalize hc : takeCoin l fs = c
      cases c with
      | none => intro σ; exact ⟨TreeCoins.any, (avg_eq_const fun _ => rfl).ge⟩
      | some fs' =>
        intro σ
        dsimp only
        have hs' := shape_take l hs hc
        have hlt : coinsLeft fs' l < coinsLeft fs l := by have := coinsLeft_take l hc; omega
        have hlt₀ : coinsLeft fs' l < coinsLeft fs₀ l := lt_of_lt_of_le hlt (coinsLeft_le l hs)
        let R : (Fin 512 → Bool) → Release := fun x => ⟨.fresh l, b, List.ofFn x⟩
        let Ψ : TreeCoins → (Fin 512 → Bool) → UpFront fs₀ → ℝ≥0∞ := fun ρ x t =>
          prob q (f (R x :: (outK fs₀ t ρ k fs' ts (pre ++ [R x]) (fun w => f (R x :: w))
            (Strategy.assocMap (R x :: ·) f _ (σ x))).1))
            (outK fs₀ t ρ k fs' ts (pre ++ [R x]) (fun w => f (R x :: w))
              (Strategy.assocMap (R x :: ·) f _ (σ x))).2
        choose ρs hρs using fun x => upper fs₀ q k fs' ts (pre ++ [R x]) (fun w => f (R x :: w))
          (Strategy.assocMap (R x :: ·) f _ (σ x)) hs'
        let ρ : TreeCoins :=
          { hello := fun w => if h : ∃ x, pre ++ [R x] <+: w then (ρs h.choose).hello w
              else (ρs fun _ => false).hello w
            opening := fun w => if h : ∃ x, pre ++ [R x] <+: w then (ρs h.choose).opening w
              else (ρs fun _ => false).opening w }
        have hagree : ∀ x, AgreeFrom (pre ++ [R x]) (ρs x) ρ := by
          intro x w hw
          have h : ∃ x', pre ++ [R x'] <+: w := ⟨x, hw⟩
          have hx : h.choose = x := by
            have h2 := List.prefix_of_prefix_length_le h.choose_spec hw (by simp)
            have h3 := h2.eq_of_length (by simp)
            have h5 := congrArg Release.coins (List.cons_eq_cons.1 (List.append_cancel_left h3)).1
            dsimp only [R] at h5
            exact List.ofFn_injective h5
          constructor
          · simp only [ρ]; rw [dite_eq_left_of_eq_true (eq_true h), hx]
          · simp only [ρ]; rw [dite_eq_left_of_eq_true (eq_true h), hx]
        refine ⟨ρ, ?_⟩
        have hlazy := avg_lazy (fun t : UpFront fs₀ => t.read l (coinsLeft fs' l))
          (fun t x => t.set l (coinsLeft fs' l) x) (fun t x => read_set t l _ x hlt₀) (fun t => set_read t l _)
          (fun t x y => set_set t l _ x y) (Ψ ρ) (fun x t y => by
            simp only [Ψ]
            rw [outK_congr fs₀ k fs' ts _ _ _ t (t.set l (coinsLeft fs' l) y) ρ ρ hs'
              (agree_set t l _ y hs' le_rfl) (agreeFrom_refl _ _)])
        calc prob q _ σ
            = avg fun x => prob q ((releases k fs' ts).bind fun w => f (R x :: w))
                (Strategy.assocMap (R x :: ·) f _ (σ x)) :=
              congrArg avg (funext fun x => prob_assocMap q (R x :: ·) f _ (σ x))
          _ ≤ avg fun x => avg fun t => Ψ (ρs x) x t := avg_mono fun x => hρs x
          _ = avg fun x => avg fun t => Ψ ρ x t := by
              refine congrArg avg (funext fun x => congrArg avg (funext fun t => ?_))
              simp only [Ψ]
              rw [outK_congr fs₀ k fs' ts _ _ _ t t (ρs x) ρ hs' (agree_refl _ _ _) (hagree x)]
          _ = avg fun t => Ψ ρ (t.read l (coinsLeft fs' l)) t := hlazy.symm
    · change ∃ ρ, prob q _ σ ≤ _
      dsimp only at σ ⊢
      by_cases hc : treeLines ts < maxTreeLines ∧ TreeSpecOk spec
      swap
      · conv => enter [1, ρ, 2, 1, t]; rw [dite_eq_right hc]
        exact exists_le_of_eq (prob_ifNeg q f hc σ) ⟨TreeCoins.any, (avg_eq_const fun _ => rfl).ge⟩
      conv => enter [1, ρ, 2, 1, t]; rw [dite_eq_left hc]
      refine exists_le_of_eq (prob_ifPos q f hc σ) ?_
      generalize Game.Strategy.ifPos hc σ = σ'
      clear σ
      let R : TreeSample spec → Release := fun x =>
        ⟨.hello spec, b, bitsOfBytes (treeRoot b spec x ++ treeKey x.1)⟩
      let T : TreeSample spec → OpenTree := fun x => ⟨b, spec, x, fun _ => 0⟩
      let Ψ : TreeCoins → TreeSample spec → UpFront fs₀ → ℝ≥0∞ := fun ρ x t =>
        prob q (f (R x :: (outK fs₀ t ρ k fs (ts ++ [T x]) (pre ++ [R x]) (fun w => f (R x :: w))
          (Strategy.assocMap (R x :: ·) f _ (σ' x))).1))
          (outK fs₀ t ρ k fs (ts ++ [T x]) (pre ++ [R x]) (fun w => f (R x :: w))
            (Strategy.assocMap (R x :: ·) f _ (σ' x))).2
      choose ρs hρs using fun x => upper fs₀ q k fs (ts ++ [T x]) (pre ++ [R x]) (fun w => f (R x :: w))
        (Strategy.assocMap (R x :: ·) f _ (σ' x)) hs
      obtain ⟨x, hx⟩ := avg_le_of_max fun x => avg fun t => Ψ (ρs x) x t
      let ρ : TreeCoins :=
        { hello := fun w sp ν => if w = pre ∧ ν = b then Function.update (fun sp => (ρs x).hello w sp ν) spec x sp
            else (ρs x).hello w sp ν
          opening := (ρs x).opening }
      have hρx : ρ.hello pre spec b = x := by simp [ρ]
      have hagree : AgreeFrom (pre ++ [R x]) (ρs x) ρ := by
        intro w hw
        refine ⟨funext fun sp => funext fun ν => ?_, rfl⟩
        simp [ρ, ne_of_prefix hw]
      refine ⟨ρ, ?_⟩
      rw [hρx]
      calc prob q _ σ'
          = avg fun x => prob q ((releases k fs (ts ++ [T x])).bind fun w => f (R x :: w))
              (Strategy.assocMap (R x :: ·) f _ (σ' x)) :=
            congrArg avg (funext fun x => prob_assocMap q (R x :: ·) f _ (σ' x))
        _ ≤ avg fun x => avg fun t => Ψ (ρs x) x t := avg_mono fun x => hρs x
        _ ≤ avg fun t => Ψ (ρs x) x t := hx
        _ = avg fun t => Ψ ρ x t := by
            refine congrArg avg (funext fun t => ?_)
            simp only [Ψ]
            rw [outK_congr fs₀ k fs _ _ _ _ t t (ρs x) ρ hs (agree_refl _ _ _) hagree]
    · change ∃ ρ, prob q _ σ ≤ _
      dsimp only at σ ⊢
      by_cases hc : treeLines ts < maxTreeLines
      swap
      · conv => enter [1, ρ, 2, 1, t]; rw [dite_eq_right hc]
        exact exists_le_of_eq (prob_ifNeg q f hc σ) ⟨TreeCoins.any, (avg_eq_const fun _ => rfl).ge⟩
      conv => enter [1, ρ, 2, 1, t]; rw [dite_eq_left hc]
      refine exists_le_of_eq (prob_ifPos q f hc σ) ?_
      generalize Game.Strategy.ifPos hc σ = σ'
      clear σ
      revert σ'
      generalize ts[i]? = oT
      cases oT with
      | none => intro σ; exact ⟨TreeCoins.any, (avg_eq_const fun _ => rfl).ge⟩
      | some T =>
        intro σ
        dsimp only at σ ⊢
        revert σ
        generalize T.next s n = oN
        cases oN with
        | none => intro σ; exact ⟨TreeCoins.any, (avg_eq_const fun _ => rfl).ge⟩
        | some p =>
          obtain ⟨s', r'⟩ := p
          intro σ
          dsimp only at σ ⊢
          have : Nonempty (TreeOpenable T.nonce T.spec T.sample s' r') := ⟨⟨T.sample.2 s' r', rfl⟩⟩
          let R : TreeOpenable T.nonce T.spec T.sample s' r' → Release := fun d =>
            ⟨.opening i s n, b, TreeDraw.opened d.1 n⟩
          let Ψ : TreeCoins → TreeOpenable T.nonce T.spec T.sample s' r' → UpFront fs₀ → ℝ≥0∞ := fun ρ d t =>
            prob q (f (R d :: (outK fs₀ t ρ k fs (ts.set i (T.bump s')) (pre ++ [R d]) (fun w => f (R d :: w))
              (Strategy.assocMap (R d :: ·) f _ (σ d))).1))
              (outK fs₀ t ρ k fs (ts.set i (T.bump s')) (pre ++ [R d]) (fun w => f (R d :: w))
                (Strategy.assocMap (R d :: ·) f _ (σ d))).2
          choose ρs hρs using fun d => upper fs₀ q k fs (ts.set i (T.bump s')) (pre ++ [R d])
            (fun w => f (R d :: w)) (Strategy.assocMap (R d :: ·) f _ (σ d)) hs
          obtain ⟨d, hd⟩ := avg_le_of_max fun d => avg fun t => Ψ (ρs d) d t
          let ρ : TreeCoins :=
            { hello := (ρs d).hello
              opening := fun w i' s₁ n' b' T' s₂ r₂ => if w = pre then
                  Function.update (fun key : OKey => (ρs d).opening w key.i key.s key.n key.b key.T key.s' key.r)
                    ⟨i, s, n, b, T, s', r'⟩ d ⟨i', s₁, n', b', T', s₂, r₂⟩
                else (ρs d).opening w i' s₁ n' b' T' s₂ r₂ }
          have hρd : ρ.opening pre i s n b T s' r' = d := by simp [ρ]
          have hagree : AgreeFrom (pre ++ [R d]) (ρs d) ρ := by
            intro w hw
            refine ⟨rfl, ?_⟩
            funext i' s₁ n' b' T' s₂ r₂
            simp [ρ, ne_of_prefix hw]
          refine ⟨ρ, ?_⟩
          rw [hρd]
          calc prob q _ σ
              = avg fun d => prob q ((releases k fs (ts.set i (T.bump s'))).bind fun w => f (R d :: w))
                  (Strategy.assocMap (R d :: ·) f _ (σ d)) :=
                congrArg avg (funext fun d => prob_assocMap q (R d :: ·) f _ (σ d))
            _ ≤ avg fun d => avg fun t => Ψ (ρs d) d t := avg_mono fun d => hρs d
            _ ≤ avg fun t => Ψ (ρs d) d t := hd
            _ = avg fun t => Ψ ρ d t := by
                refine congrArg avg (funext fun t => ?_)
                simp only [Ψ]
                rw [outK_congr fs₀ k fs _ _ _ _ t t (ρs d) ρ hs (agree_refl _ _ _) hagree]

end Coupling

/-- **The coupling of the play to the coins flipped up front.** For every families `fs`, device signer `D`, parameters,
committed files, commitment, strategy `σ` and event `q`, the probability of `q` in the play of `σ` is at least its
probability over the coins flipped up front (the families' coins and the device's sample at the close) at some resolution
of the coin trees' coins, and at most its probability at another. -/
def PlayCouples : Prop :=
  ∀ (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (play fs D θ F c)) (q : Exchange → Prop),
    (∃ ρ : TreeCoins, Game.prCoin (fun t => q (outUp fs D θ F c ρ σ t)) ≤ Game.prob q (play fs D θ F c) σ) ∧
      ∃ ρ : TreeCoins, Game.prob q (play fs D θ F c) σ ≤ Game.prCoin fun t => q (outUp fs D θ F c ρ σ t)

open Game in
/-- **`PlayCouples`**: the upper half from the device's state after the request (`Coupling.upper`), the close's sample
being the last coordinate of the coins flipped up front (`Coupling.avg_prod`); the lower half from the upper half of the
complement, whose probabilities with the event's sum to one in the play and over the coins flipped up front. -/
theorem play_couples : PlayCouples := by
  have up : ∀ (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
      (σ : Game.Strategy (play fs D θ F c)) (q : Exchange → Prop),
      ∃ ρ : TreeCoins, Game.prob q (play fs D θ F c) σ ≤ Game.prCoin fun t => q (outUp fs D θ F c ρ σ t) := by
    intro fs D θ F c σ q
    obtain ⟨ρ, h⟩ := Coupling.upper fs q (frames fs) fs [] [] (finish D θ F c σ.1) σ.2 (Coupling.shape_refl fs)
    refine ⟨ρ, h.trans (le_of_eq ?_)⟩
    unfold Game.prCoin
    rw [Coupling.avg_prod]
    rfl
  intro fs D θ F c σ q
  refine ⟨?_, up fs D θ F c σ q⟩
  obtain ⟨ρ, h⟩ := up fs D θ F c σ fun o => ¬ q o
  refine ⟨ρ, ?_⟩
  have e1 := Coupling.prob_add_not q (play fs D θ F c) σ
  have e2 := Coupling.prCoin_add_not fun t => q (outUp fs D θ F c ρ σ t)
  have hA : Game.prCoin (fun t => ¬ q (outUp fs D θ F c ρ σ t)) ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (e2 ▸ le_add_self)
  refine ENNReal.le_of_add_le_add_right hA ?_
  rw [e2, ← e1]
  exact add_le_add le_rfl h

/-! ## Each subprotocol's target from its bound over the coins flipped up front -/

/-- **`CoinsCouple`**: the play's probability of the event is at most its probability over the coins flipped up front at
some resolution of the coin trees' coins (`play_couples`), which the bound takes at every resolution. -/
theorem coins_couple : CoinsCouple := by
  intro ι α _ d V acc k a h θ F c σ
  obtain ⟨ρ, hρ⟩ := (play_couples (V.families θ) (V.signer θ) θ F c σ fun o => acc o ∧ ¬ V.Holds k θ F c σ a o).2
  exact hρ.trans (h θ F c σ ρ)

/-- **`TargetOfPeriods`**: at the resolution `play_couples` gives, the coupled run's coins carry the play's event into
the run's (its verdict accepting while its claim fails, which `PeriodRun.Sound` bounds at the run's own coordinate
within `δ`) or the rest of the event (`BreakBound`, whose forgery term `Unforgeable` bounds by `ε_forge`). -/
theorem target_of_periods : TargetOfPeriods := by
  intro ι α _ d V acc k b h θ F c σ
  unfold Vocab.error Accounting.condError
  split_ifs with hcr
  · classical
    obtain ⟨-, hcr, hun⟩ := hcr
    obtain ⟨ρ, hρ⟩ := (play_couples (V.families θ) (V.signer θ) θ F c σ
      fun o => acc o ∧ ¬ V.Holds k θ F c σ (.ofAudit b) o).2
    obtain ⟨q, δ, hs, hδ, hb⟩ := h θ F c σ ρ
    have hf : ENNReal.ofReal ((V.forger (.ofAudit b) θ F c σ).Pr lmsVerify) ≤
        ENNReal.ofReal (V.epsForge θ (V.callsMul θ (.ofAudit b) * V.cost θ F c σ + V.callsAdd θ (.ofAudit b))
          (V.signatures θ)) :=
      ENNReal.ofReal_le_ofReal (hun (V.forger_calls _ θ F c σ) (V.forger_signed _ θ F c σ))
    show _ ≤ V.delta θ (.ofAudit b) (V.cost θ F c σ) +
      ENNReal.ofReal (V.epsForge θ (V.callsMul θ (.ofAudit b) * V.cost θ F c σ + V.callsAdd θ (.ofAudit b))
        (V.signatures θ)) +
      V.loss (.ofAudit b) θ (ENNReal.ofReal ((V.finder (.ofAudit b) θ F c σ).Pr sha512Msg))
    rw [add_assoc]
    refine hρ.trans ((Coupling.prCoin_equiv q.coins _).symm.trans_le ?_)
    refine (Game.prCoin_mono (q := fun ξ => (q.run.accepts b ξ ∧ ¬ q.run.Holds (V.certOf θ) q.price b ξ) ∨
      (acc (outUp (V.families θ) (V.signer θ) θ F c ρ σ (q.coins ξ)) ∧
        (¬ V.Logged (.ofAudit b) (outUp (V.families θ) (V.signer θ) θ F c ρ σ (q.coins ξ)) ∨ q.brk ξ))) ?_).trans
      ((Coupling.prCoin_or_le _ _).trans (add_le_add ?_ (le_trans (hb hcr) (add_le_add hf le_rfl))))
    · intro ξ ⟨ha, hh⟩
      by_cases hl : V.Logged (.ofAudit b) (outUp (V.families θ) (V.signer θ) θ F c ρ σ (q.coins ξ))
      · by_cases hk : q.brk ξ
        · exact Or.inr ⟨ha, Or.inr hk⟩
        · exact Or.inl ⟨q.verdict ξ ha hl hk, fun hw => hh (q.world ξ hw)⟩
      · exact Or.inr ⟨ha, Or.inl hl⟩
    · rw [Coupling.prCoin_eq_ofReal_pr]
      exact (ENNReal.ofReal_le_ofReal (Coupling.pr_le_of_update (q.run.own b)
        (fun ξ => q.run.accepts b ξ ∧ ¬ q.run.Holds (V.certOf θ) q.price b ξ) hs)).trans hδ
  · exact le_top

/-- **`WipeTarget`**: where `acc` holds and the wipe's claim fails, the wipe accepts and the play keeps foreign content
past the threshold, which the survivor bound on the run bounds (`Wipe.OfRun.prob_kept_le`). -/
theorem wipe_target : WipeTarget := by
  intro ι α _ d V _ _ _ _ acc k ε hS hW θ F c σ
  refine le_trans (Game.prob_mono (q := fun o => (V.wipeRun θ F c σ).accepts o ∧
    (V.wipeRun θ F c σ).Kept (V.foreignBits θ) o)
    (fun o ho => ⟨hW.verdict θ F c σ o ho.1, Classical.byContradiction ho.2⟩) _ σ) ?_
  exact (Wipe.OfRun.prob_kept_le (hS θ F) _ σ (hW.preserves θ F c σ)).trans (hW.error θ F c σ)

/-- **`SoundnessOfUpFront`**: each subprotocol's bound at its component or as its target gives its target
(`Vocab.sound_of`), and its bound over the coins flipped up front its bound at its component (`coins_couple`); the
targets give soundness (`soundness_at_of_targets`). -/
theorem soundness_of_upFront : SoundnessOfUpFront := by
  intro ι α _ d V k h
  exact soundness_at_of_targets ι α d V k fun a ha =>
    (h a ha).elim (V.sound_of k a) fun h' =>
      h'.elim (fun hu => V.sound_of k a (coins_couple ι α d V _ k a hu)) id

/-! ## Witnesses, at the idle deployment -/

namespace Idle

/-- At `exact`, PoUW's target fails at remote for a verifier that accepts everything: its claim fails on every play
after the empty commitment (`pouw_fails`), and every such play has probability one (`prob_play_eq_one`). -/
theorem pouw_nontrivial : ¬ exact.TargetFor (fun _ => True) .remote .pouw := by
  intro h
  have h1 := prob_play_eq_one (exact.families .empty) (D := exact.signer .empty) .empty #[.empty] .empty
    (fun o => (fun _ : Exchange => True) o ∧
      ¬ exact.Holds .remote .empty #[.empty] .empty (exact.honest .empty #[.empty] .empty) .pouw o)
    (fun o ho => ⟨trivial, pouw_fails .remote _ _ _ o ho⟩) (exact.honest .empty #[.empty] .empty)
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h .empty #[.empty] .empty _)).trans_lt
    (lt_of_eq_of_lt (error_exact _ _ _ _ _) zero_lt_one))

end Idle

/-- **`CoinsCouple`'s honest lemma**: at the idle deployment, PoUW meets its bound at its component at remote over the
coins flipped up front, its error being `⊤`. -/
theorem coins_couple_honest :
    Idle.unbounded.UpFrontFor (fun o => ComponentAccepts .remote .pouw o.sent) .remote .pouw := by
  intro θ F c σ ρ
  rw [Idle.error_unbounded]
  exact le_top

/-- **`CoinsCouple`'s nontrivial lemma**: its conclusion fails at remote's PoUW for a verifier that accepts everything,
at the idle deployment whose errors are all `0`. -/
theorem coins_couple_nontrivial : ¬ Idle.exact.TargetFor (fun _ => True) .remote .pouw := Idle.pouw_nontrivial

/-- **`TargetOfPeriods`' honest lemma**: at the idle deployment publishing `⊤` (`Idle.unbounded`), PoUW's plays at
remote couple to the period run that charges nothing and accepts everything (`Idle.coupling`), within `δ = 1`; no check
reads the device's log, and the coupling has no break, so the rest of the event is empty. -/
theorem target_of_periods_honest :
    (Idle.vocab ⊤ 1).PeriodsCouple (fun o => ComponentAccepts .remote .pouw o.sent) .remote .pouw := by
  intro θ F c σ ρ
  refine ⟨Idle.coupling ⊤ _ .remote θ F c σ ρ, 1, fun _ => Pouw.pr_le_one _, le_top, fun _ => ?_⟩
  refine le_trans (Game.prCoin_mono (q := fun _ => False) ?_) (by rw [Game.prCoin_false]; exact zero_le)
  rintro ξ ⟨-, hl | hk⟩
  · exact hl fun log h => by simp [Idle.vocab] at h
  · exact hk

/-- **`TargetOfPeriods`' nontrivial lemma**: its conclusion fails at remote's PoUW for a verifier that accepts
everything, at the idle deployment whose errors are all `0`. -/
theorem target_of_periods_nontrivial : ¬ Idle.exact.TargetFor (fun _ => True) .remote .pouw := Idle.pouw_nontrivial

/-- **`WipeTarget`'s honest lemma**: at the idle deployment, whose wipe accepts nothing, the wipe meets its survivor
bound at `ε = 0`, and every play couples to the wipe's game at the verdict that accepts nothing: the run holds no
foreign content, one seed and one spot read. -/
theorem wipe_target_honest :
    (∀ θ F, (Idle.unbounded.wipe θ F).SurvivorBound (Idle.unbounded.foreignBits θ) fun _ _ _ => 0) ∧
      Idle.unbounded.WipeCoupling (fun _ => False) fun _ _ _ _ _ => 0 := by
  refine ⟨fun θ F S_φ _ hold keep => ?_, ⟨fun θ F c σ P => ?_, fun _ _ _ _ _ h => h.elim, fun _ _ _ _ => zero_le⟩⟩
  · refine (Game.value_mono (q := fun _ => False) (fun x hx => hx.1) _).trans (le_of_eq ?_)
    rw [show Game.draw ((Fin S_φ → Bool) × Idle.unbounded.Seed × Idle.unbounded.Spots) =
      .coin _ fun x => .ret (id x) from rfl, Game.value_coin_ret, Game.prCoin_false]
  · exact Coupling.prob_const_eq P (fun i => i.elim0, (), ())
      (fun y => Prod.ext (funext fun i => i.elim0) (Prod.ext rfl rfl)) _ σ

/-- **`WipeTarget`'s nontrivial lemma**: its conclusion fails at onsite's wipe for a verifier that accepts everything,
at the idle deployment whose errors are all `0`, where every play keeps its (empty) foreign content. -/
theorem wipe_target_nontrivial : ¬ Idle.exact.TargetFor (fun _ => True) .onsite .wipe :=
  Idle.exact.sound_of_nontrivial .onsite .wipe .empty #[.empty] .empty (Idle.exact.honest .empty #[.empty] .empty)
    (fun _ h => h ⟨Nat.le_refl 0, funext fun i => i.elim0⟩) (lt_of_eq_of_lt (Idle.error_exact _ _ _ _ _) zero_lt_one)

/-- **`SoundnessOfUpFront`'s honest lemma**: its hypotheses hold at the idle deployment's compiled remote composite,
sampled proofs, PoUW and the inference-only policy each meeting its bound at its component. -/
theorem soundness_of_upFront_honest :
    Idle.unbounded.SubprotocolBound .remote .sampled ∧ Idle.unbounded.SubprotocolBound .remote .pouw ∧
      Idle.unbounded.SubprotocolBound .remote .inferenceOnly :=
  ⟨.inl (Idle.componentBound_unbounded _ _), .inl (Idle.componentBound_unbounded _ _),
    .inl (Idle.componentBound_unbounded _ _)⟩

/-- **`SoundnessOfUpFront`'s nontrivial lemma**: its conclusion fails at remote for a verifier that accepts
everything, at the idle deployment whose errors are all `0`. -/
theorem soundness_of_upFront_nontrivial : ¬ Idle.exact.SoundnessFor .remote fun _ => True :=
  Idle.nontrivial_at .remote (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

end Verity
