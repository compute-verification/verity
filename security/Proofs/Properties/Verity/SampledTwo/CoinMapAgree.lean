module

public import Proofs.Properties.Verity.SampledTwo.CoinMapLaw

/-!
# The two-stage leg's coin map: its sampling design with the agreement in the event (`coinMapHolds`)

`CoinMapHolds` (`Proofs.Properties.Verity.SampledTwo.CoinMap`) is `CoinMapLaw` at the coin map's frame, with the
agreement `ElemAgrees` inside the event and `hide = gapOf`. It is proved as `lawAt` is, along `relK` one coin node at a
time, now counting the mapped play only where every release after the releases before agrees with the run's coins.

* **The sampling design with an agreement** (`LawA`, `law_idA`, `law_optA`, `law_iteA`, `law_keepA`, `law_setA`, `law_writeA`):
  `LawHolds` counting the mapped play only where an agreement holds. Below a coin node the agreement is the subtree's
  own; a kept node or a node written with an independent value gives the node's agreement wherever the subtree's holds,
  and a node written through `W` gives it wherever the written value is good, at the chance it is not.
* **A release agrees** (`RelGood`, `LogGood`, `AgFrom`): where the locator puts a release and its elements fit, it
  carries the round's coin's next elements; `AgFrom` holds of an output whose log's releases after `pre` all do.
  `LogGood` after `pre` reads the run's coins only where the releases after `pre` read (`logGood_congr`).
* **Along the releases** (`law_placeKA`): the sampling design with `AgFrom` holds along `relK` at `gapK`. A fresh coin's release
  is the run's elements; an opening's is wherever its class at the run's elements is not empty, which `gapRound`'s
  middle term prices.
* **To `ElemAgrees`** (`devBits_getD`, `elemAgrees_of_logGood`): where every release of a log agrees, the bits `devBits`
  collects for a round are the run's coin's, so `dev`'s elements are the run's below what the releases carry.
* **The sampling design** (`law_placeA`, `coinMapHolds`).
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open Claims
open scoped ENNReal

variable {β : Type}

theorem ind_mono {p q : Prop} [Decidable p] [Decidable q] (h : p → q) (x : ℝ≥0∞) :
    (if p then x else 0) ≤ (if q then x else 0) := by
  by_cases hp : p
  · rw [if_pos hp, if_pos (h hp)]
  · rw [if_neg hp]
    exact zero_le

theorem ind_le_one {p : Prop} [Decidable p] {x : ℝ≥0∞} (hx : x ≤ 1) : (if p then x else 0) ≤ 1 := by
  by_cases hp : p
  · rw [if_pos hp]
    exact hx
  · rw [if_neg hp]
    exact zero_le

/-- An `if` is the same at any two instances that decide its condition. -/
theorem ite_inst {α : Type} {p : Prop} (d₁ d₂ : Decidable p) (x y : α) : @ite α p d₁ x y = @ite α p d₂ x y := by
  rw [Subsingleton.elim d₁ d₂]

/-- **An output a play reaches**: the play's output at any tape. -/
theorem playReaches_tapeOut : ∀ (g : Game β) (s : Strategy g) (t : Tape g s), PlayReaches g s (Tape.out g s t)
  | .ret _, _, _ => rfl
  | .send _ k, s, t => playReaches_tapeOut (k s.1) s.2 t
  | @Game.coin _ C _ _ k, s, t =>
    let t' : C × ((c : C) → Tape (k c) (s c)) := t
    ⟨t'.1, playReaches_tapeOut (k t'.1) (s t'.1) (t'.2 t'.1)⟩

/-! ## The sampling design with an agreement -/

section Node

variable {R : Type} [Fintype R] [Nonempty R]

open Classical in
/-- **The sampling design with the agreement `Ag`**: as `LawHolds`, the mapped play counted only where `Ag` holds of the coins and
its output. -/
def LawA (g : Game β) (s : Strategy g) (P : R → Tape g s → Tape g s) (Ag : R → β → Prop) (G : ℝ≥0∞) : Prop :=
  ∀ H : β → ℝ≥0∞, (∀ b, H b ≤ 1) →
    avg (fun t : Tape g s => H (Tape.out g s t)) ≤
      avg (fun r : R => avg fun t : Tape g s =>
        if Ag r (Tape.out g s (P r t)) then H (Tape.out g s (P r t)) else 0) + G

theorem law_monoA {g : Game β} {s : Strategy g} {P : R → Tape g s → Tape g s} {Ag : R → β → Prop} {G G' : ℝ≥0∞}
    (h : LawA g s P Ag G) (hG : G ≤ G') : LawA g s P Ag G' :=
  fun H hH => (h H hH).trans (add_le_add le_rfl hG)

theorem law_congrA {g : Game β} {s : Strategy g} {P P' : R → Tape g s → Tape g s} {Ag : R → β → Prop} {G : ℝ≥0∞}
    (hP : ∀ r t, P r t = P' r t) (h : LawA g s P' Ag G) : LawA g s P Ag G := by
  have : P = P' := funext fun r => funext fun t => hP r t
  subst this
  exact h

theorem law_idA {g : Game β} {s : Strategy g} {Ag : R → β → Prop} (h : ∀ r t, Ag r (Tape.out g s t)) :
    LawA g s (fun _ t => t) Ag 0 := by
  classical
  intro H _
  refine le_of_eq ?_
  rw [add_zero]
  exact (avg_const _).symm.trans
    (congrArg avg (funext fun r => congrArg avg (funext fun t => (if_pos (h r t)).symm)))

theorem law_sendA {M : Type} (kf : M → Game β) (m : M) (s' : Strategy (kf m))
    {P : R → Tape (kf m) s' → Tape (kf m) s'} {Ag : R → β → Prop} {G : ℝ≥0∞} (h : LawA (kf m) s' P Ag G) :
    LawA (Game.send M kf) ⟨m, s'⟩ P Ag G := h

theorem law_alongA {g g' : Game β} (h : g = g') {Pl : R → (s : Strategy g') → Tape g' s → Tape g' s}
    {Ag : R → β → Prop} {G : Strategy g' → ℝ≥0∞} (hl : ∀ s, LawA g' s (fun r => Pl r s) Ag (G s)) (s : Strategy g) :
    LawA g s (fun r => along h (Pl r) s) Ag (alongVal h G s) := by
  subst h
  exact hl s

theorem law_optA {γ : Type} (n : Game β) (y : γ → Game β) {Ag : R → β → Prop}
    {fn : R → (s : Strategy n) → Tape n s → Tape n s}
    {fy : R → (a : γ) → (s : Strategy (y a)) → Tape (y a) s → Tape (y a) s} {gn : Strategy n → ℝ≥0∞}
    {gy : (a : γ) → Strategy (y a) → ℝ≥0∞} (hn : ∀ s, LawA n s (fun r => fn r s) Ag (gn s))
    (hy : ∀ a s, LawA (y a) s (fun r => fy r a s) Ag (gy a s)) :
    ∀ (o : Option γ) (s : Strategy (optGame o n y)),
      LawA (optGame o n y) s (fun r => placeOpt n y (fn r) (fy r) o s) Ag (gapOpt n y gn gy o s)
  | none, s => hn s
  | some a, s => hy a s

theorem law_iteA {c : Prop} (dc : Decidable c) (G₁ G₂ : Game β) {Ag : R → β → Prop}
    {f₁ : R → (s : Strategy G₁) → Tape G₁ s → Tape G₁ s} {f₂ : R → (s : Strategy G₂) → Tape G₂ s → Tape G₂ s}
    {g₁ : Strategy G₁ → ℝ≥0∞} {g₂ : Strategy G₂ → ℝ≥0∞} (h₁ : ∀ s, LawA G₁ s (fun r => f₁ r s) Ag (g₁ s))
    (h₂ : ∀ s, LawA G₂ s (fun r => f₂ r s) Ag (g₂ s)) :
    ∀ s : Strategy (@ite _ c dc G₁ G₂),
      LawA (@ite _ c dc G₁ G₂) s (fun r => placeIte dc G₁ G₂ s (f₁ r) (f₂ r)) Ag (gapIte dc G₁ G₂ s g₁ g₂) := by
  cases dc with
  | isFalse _ => exact h₂
  | isTrue _ => exact h₁

/-- **A kept coin node**: the node's agreement holds wherever the subtree's does. -/
theorem law_keepA {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) {f : R → (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s}
    {Ag : R → β → Prop} {Ag' : C → R → β → Prop} {G : C → ℝ≥0∞}
    (hup : ∀ c r τ, Ag' c r (Tape.out (k c) (s c) (f r c (s c) τ)) → Ag r (Tape.out (k c) (s c) (f r c (s c) τ)))
    (h : ∀ c, LawA (k c) (s c) (fun r => f r c (s c)) (Ag' c) (G c)) :
    LawA (@Game.coin β C iF iN k) s (fun r => placeCoin k id (f r) s) Ag (avg G) := by
  classical
  intro H hH
  let B' : C → R → ℝ≥0∞ := fun c r => avg fun τ : Tape (k c) (s c) =>
    if Ag' c r (Tape.out (k c) (s c) (f r c (s c) τ)) then H (Tape.out (k c) (s c) (f r c (s c) τ)) else 0
  have e : ∀ r, avg (fun t : Tape (@Game.coin β C iF iN k) s =>
      if Ag r (Tape.out _ s (placeCoin k id (f r) s t)) then H (Tape.out _ s (placeCoin k id (f r) s t)) else 0) =
        avg fun c => avg fun τ : Tape (k c) (s c) =>
          if Ag r (Tape.out (k c) (s c) (f r c (s c) τ)) then H (Tape.out (k c) (s c) (f r c (s c) τ)) else 0 :=
    fun r => avg_tape_coin s fun c τ =>
      if Ag r (Tape.out (k c) (s c) (f r c (s c) τ)) then H (Tape.out (k c) (s c) (f r c (s c) τ)) else 0
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => H (Tape.out _ s t))
      = avg fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) τ) :=
        avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) τ)
    _ ≤ avg fun c => (avg fun r : R => B' c r) + G c := avg_mono fun c => h c H hH
    _ = (avg fun c => avg fun r : R => B' c r) + avg G := avg_add _ _
    _ = (avg fun r : R => avg fun c => B' c r) + avg G := congrArg (· + avg G) (avg_swap _)
    _ ≤ _ := add_le_add (avg_mono fun r => (avg_mono fun c => avg_mono fun τ =>
        ind_mono (hup c r τ) _).trans_eq (e r).symm) le_rfl

/-- **A coin node written with `v r`**, independent of what the map and the subtree's agreement read after it: the
node's agreement holds wherever the written subtree's does. -/
theorem law_setA {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) {f : R → (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s}
    {Ag : R → β → Prop} {Ag' : C → R → β → Prop} {G : C → ℝ≥0∞} (v : R → C) (Rel : R → R → Prop)
    (hv : Splits v Rel) (hf : ∀ c r r' τ, Rel r r' → f r c (s c) τ = f r' c (s c) τ)
    (hAg : ∀ c r r' o, Rel r r' → (Ag' c r o ↔ Ag' c r' o))
    (hup : ∀ r τ, Ag' (v r) r (Tape.out (k (v r)) (s (v r)) (f r (v r) (s (v r)) τ)) →
      Ag r (Tape.out (k (v r)) (s (v r)) (f r (v r) (s (v r)) τ)))
    (h : ∀ c, LawA (k c) (s c) (fun r => f r c (s c)) (Ag' c) (G c)) :
    LawA (@Game.coin β C iF iN k) s (fun r => placeCoin k (fun _ => v r) (f r) s) Ag (avg G) := by
  classical
  intro H hH
  let B' : C → R → ℝ≥0∞ := fun c r => avg fun τ : Tape (k c) (s c) =>
    if Ag' c r (Tape.out (k c) (s c) (f r c (s c) τ)) then H (Tape.out (k c) (s c) (f r c (s c) τ)) else 0
  have e : ∀ r, avg (fun t : Tape (@Game.coin β C iF iN k) s =>
      if Ag r (Tape.out _ s (placeCoin k (fun _ => v r) (f r) s t)) then
        H (Tape.out _ s (placeCoin k (fun _ => v r) (f r) s t)) else 0) =
        avg fun τ : Tape (k (v r)) (s (v r)) =>
          if Ag r (Tape.out (k (v r)) (s (v r)) (f r (v r) (s (v r)) τ)) then
            H (Tape.out (k (v r)) (s (v r)) (f r (v r) (s (v r)) τ)) else 0 :=
    fun r => avg_tape_at s (v r) fun τ =>
      if Ag r (Tape.out (k (v r)) (s (v r)) (f r (v r) (s (v r)) τ)) then
        H (Tape.out (k (v r)) (s (v r)) (f r (v r) (s (v r)) τ)) else 0
  have hB : ∀ c r r', Rel r r' → B' c r = B' c r' := fun c r r' hr =>
    congrArg avg (funext fun τ => by
      rw [hf c r r' τ hr]
      exact if_congr (hAg c r r' _ hr) rfl rfl)
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => H (Tape.out _ s t))
      = avg fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) τ) :=
        avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) τ)
    _ ≤ avg fun c => (avg fun r : R => B' c r) + G c := avg_mono fun c => h c H hH
    _ = (avg fun c => avg fun r : R => B' c r) + avg G := avg_add _ _
    _ = (avg fun r : R => B' (v r) r) + avg G := congrArg (· + avg G) (hv B' hB).symm
    _ ≤ _ := add_le_add (avg_mono fun r =>
        (avg_mono fun τ => ind_mono (hup r τ) _).trans_eq (e r).symm) le_rfl

open Classical in
/-- **A coin node written through `W` at `e r`**, independent of what the map and the subtree's agreement read after
it: the node's agreement holds wherever the written subtree's does and `e r` is good; the node costs `W`'s shortfall
and the chance `e r` is not good. -/
theorem law_writeA {C P : Type} {iF : Fintype C} {iN : Nonempty C} [Fintype P] [Nonempty P] {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) {f : R → (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s}
    {Ag : R → β → Prop} {Ag' : C → R → β → Prop} {G : C → ℝ≥0∞} (W : P → C → C) (e : R → P) (Rel : R → R → Prop)
    (he : Splits e Rel) (hf : ∀ c r r' τ, Rel r r' → f r c (s c) τ = f r' c (s c) τ)
    (hAg : ∀ c r r' o, Rel r r' → (Ag' c r o ↔ Ag' c r' o)) (Good : P → Prop) [DecidablePred Good]
    (hup : ∀ r d τ, Good (e r) → Ag' (W (e r) d) r (Tape.out (k (W (e r) d)) (s (W (e r) d))
      (f r (W (e r) d) (s (W (e r) d)) τ)) →
        Ag r (Tape.out (k (W (e r) d)) (s (W (e r) d)) (f r (W (e r) d) (s (W (e r) d)) τ)))
    (h : ∀ c, LawA (k c) (s c) (fun r => f r c (s c)) (Ag' c) (G c)) :
    LawA (@Game.coin β C iF iN k) s (fun r => placeCoin k (W (e r)) (f r) s) Ag
      (shortfall W + avg (fun p => if Good p then 0 else 1) + avg fun p => avg fun c => G (W p c)) := by
  classical
  intro H hH
  let A : C → ℝ≥0∞ := fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) τ)
  let B' : C → R → ℝ≥0∞ := fun c r => avg fun τ : Tape (k c) (s c) =>
    if Ag' c r (Tape.out (k c) (s c) (f r c (s c) τ)) then H (Tape.out (k c) (s c) (f r c (s c) τ)) else 0
  let B : C → R → ℝ≥0∞ := fun c r => avg fun τ : Tape (k c) (s c) =>
    if Ag r (Tape.out (k c) (s c) (f r c (s c) τ)) then H (Tape.out (k c) (s c) (f r c (s c) τ)) else 0
  have hA : ∀ c, A c ≤ 1 := fun c =>
    (avg_mono fun τ => hH (Tape.out (k c) (s c) τ)).trans_eq (avg_const 1)
  have hB'1 : ∀ c r, B' c r ≤ 1 := fun c r =>
    (avg_mono fun τ => ind_le_one (hH _)).trans_eq (avg_const 1)
  have e₁ : ∀ r, avg (fun t : Tape (@Game.coin β C iF iN k) s =>
      if Ag r (Tape.out _ s (placeCoin k (W (e r)) (f r) s t)) then
        H (Tape.out _ s (placeCoin k (W (e r)) (f r) s t)) else 0) = avg fun c => B (W (e r) c) r := fun r =>
    avg_tape_map s (W (e r)) fun c τ =>
      if Ag r (Tape.out (k (W (e r) c)) (s (W (e r) c)) (f r (W (e r) c) (s (W (e r) c)) τ)) then
        H (Tape.out (k (W (e r) c)) (s (W (e r) c)) (f r (W (e r) c) (s (W (e r) c)) τ)) else 0
  have hB : ∀ p r r', Rel r r' → (avg fun c => B' (W p c) r) = avg fun c => B' (W p c) r' := fun p r r' hr =>
    congrArg avg (funext fun c => congrArg avg (funext fun τ => by
      rw [hf (W p c) r r' τ hr]
      exact if_congr (hAg (W p c) r r' _ hr) rfl rfl))
  have hgood : ∀ r, (avg fun c => B' (W (e r) c) r) ≤ (avg fun c => B (W (e r) c) r) + if Good (e r) then 0 else 1 :=
    fun r => by
      by_cases hg : Good (e r)
      · rw [if_pos hg, add_zero]
        exact avg_mono fun c => avg_mono fun τ => ind_mono (hup r c τ hg) _
      · rw [if_neg hg]
        exact le_add_left ((avg_mono fun c => hB'1 (W (e r) c) r).trans_eq (avg_const 1))
  have hbad : (avg fun r : R => if Good (e r) then (0 : ℝ≥0∞) else 1) = avg fun p => if Good p then 0 else 1 :=
    calc (avg fun r : R => if Good (e r) then (0 : ℝ≥0∞) else 1)
        = avg fun p => avg fun _ : R => if Good p then (0 : ℝ≥0∞) else 1 :=
          he (fun p _ => if Good p then 0 else 1) fun _ _ _ _ => rfl
      _ = avg fun p => if Good p then 0 else 1 := by simp only [avg_const]
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => H (Tape.out _ s t))
      = avg A := avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) τ)
    _ ≤ avg (fun p : P => avg fun c : C => A (W p c)) + shortfall W := avg_le_write1 W A hA
    _ ≤ avg (fun p : P => avg fun c : C => (avg fun r : R => B' (W p c) r) + G (W p c)) + shortfall W :=
        add_le_add (avg_mono fun p => avg_mono fun c => h (W p c) H hH) le_rfl
    _ = (avg fun p : P => avg fun c : C => avg fun r : R => B' (W p c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W := by
        simp only [avg_add]
    _ = (avg fun p : P => avg fun r : R => avg fun c : C => B' (W p c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W :=
        congrArg (fun z => z + (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W)
          (congrArg avg (funext fun p => avg_swap fun (c : C) (r : R) => B' (W p c) r))
    _ = (avg fun r : R => avg fun c : C => B' (W (e r) c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W :=
        congrArg (fun z => z + (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W)
          (he (fun p r => avg fun c => B' (W p c) r) hB).symm
    _ ≤ (avg fun r : R => (avg fun c : C => B (W (e r) c) r) + if Good (e r) then 0 else 1) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W :=
        add_le_add (add_le_add (avg_mono hgood) le_rfl) le_rfl
    _ = (avg fun r : R => avg fun c : C => B (W (e r) c) r) + (avg fun p => if Good p then 0 else 1) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W := by
        rw [avg_add, hbad]
    _ = _ := by
        have hX : (avg fun r : R => avg fun c : C => B (W (e r) c) r) =
            avg fun r : R => avg fun t : Tape (@Game.coin β C iF iN k) s =>
              if Ag r (Tape.out _ s (placeCoin k (W (e r)) (f r) s t)) then
                H (Tape.out _ s (placeCoin k (W (e r)) (f r) s t)) else 0 :=
          congrArg avg (funext fun r => (e₁ r).symm)
        rw [hX]
        ring

end Node

/-! ## A release agrees with the run's coins -/

section Along

variable (L : Locate) {K : ℕ}

/-- **A release agrees with the run's coins**, after the releases `pre`: where the locator puts it and its elements
fit (`slot`), it carries the round's coin's next elements. -/
def RelGood (r : Fin K → FlockDeploy.Coin) (pre : List Verity.Release) (rl : Verity.Release) : Prop :=
  ∀ (j : Fin K) (u : ℕ), slot L K pre rl.frame rl.answers (widthOf rl.frame) = some (j, u) →
    rl.coins = openBits (r j) u (widthOf rl.frame)

/-- **Every release of `log` agrees**, after the releases `pre`. -/
def LogGood (r : Fin K → FlockDeploy.Coin) : List Verity.Release → List Verity.Release → Prop
  | _, [] => True
  | pre, rl :: log => RelGood L r pre rl ∧ LogGood r (pre ++ [rl]) log

/-- **The releases after `pre` of an output's log agree** (`lg` the output's log). -/
def AgFrom (lg : β → List Verity.Release) (r : Fin K → FlockDeploy.Coin) (pre : List Verity.Release) (o : β) : Prop :=
  LogGood L r pre ((lg o).drop pre.length)

/-- **A fresh coin's written bits are an opening's of 4 elements.** -/
theorem ofFn_freshVal (c : FlockDeploy.Coin) (u : ℕ) : List.ofFn (freshVal c u) = openBits c u 4 := rfl

theorem relGood_of_none {r : Fin K → FlockDeploy.Coin} {pre : List Verity.Release} {rl : Verity.Release}
    (h : slot L K pre rl.frame rl.answers (widthOf rl.frame) = none) : RelGood L r pre rl := fun _ _ hs => by
  rw [h] at hs
  cases hs

theorem relGood_congr {r r' : Fin K → FlockDeploy.Coin} {pre : List Verity.Release} (h : AgreeFrom L pre r r')
    (rl : Verity.Release) : RelGood L r pre rl ↔ RelGood L r' pre rl := by
  unfold RelGood
  refine forall_congr' fun j => forall_congr' fun u => imp_congr_right fun hs => ?_
  obtain ⟨-, hu, -⟩ := slot_spec L hs
  rw [openBits_congr (c := r j) (c' := r' j) fun a ha => h j a (by rw [← hu]; exact ha)]

theorem logGood_congr {r r' : Fin K → FlockDeploy.Coin} :
    ∀ (log pre : List Verity.Release), AgreeFrom L pre r r' → (LogGood L r pre log ↔ LogGood L r' pre log)
  | [], _, _ => Iff.rfl
  | rl :: log, pre, h => and_congr (relGood_congr L h rl) (logGood_congr log (pre ++ [rl]) (h.snoc L rl))

theorem agFrom_nil {lg : β → List Verity.Release} {r : Fin K → FlockDeploy.Coin} {pre : List Verity.Release} {o : β}
    (h : lg o = pre ++ []) : AgFrom L lg r pre o := by
  unfold AgFrom
  rw [h, List.append_nil, List.drop_length]
  exact trivial

theorem agFrom_cons {lg : β → List Verity.Release} {r : Fin K → FlockDeploy.Coin} {pre log : List Verity.Release}
    {rl : Verity.Release} {o : β} (h : lg o = pre ++ rl :: log) :
    AgFrom L lg r pre o ↔ RelGood L r pre rl ∧ AgFrom L lg r (pre ++ [rl]) o := by
  have e₁ : (lg o).drop pre.length = rl :: log := by
    rw [h]
    exact List.drop_left
  have e₂ : (lg o).drop (pre ++ [rl]).length = log := by
    rw [h, show pre ++ rl :: log = (pre ++ [rl]) ++ log by simp]
    exact List.drop_left
  unfold AgFrom
  rw [e₁, e₂]
  exact Iff.rfl

/-! ## The sampling design with the agreement along the releases -/

/-- The continuation `f` after the releases `pre` ends in outputs whose log is `pre` and the releases after. -/
def Records (lg : β → List Verity.Release) (f : List Verity.Release → Game β) (pre : List Verity.Release) : Prop :=
  ∀ (log : List Verity.Release) (s : Strategy (f log)) (o : β), PlayReaches (f log) s o → lg o = pre ++ log

theorem Records.cons {lg : β → List Verity.Release} {f : List Verity.Release → Game β} {pre : List Verity.Release}
    (h : Records lg f pre) (rl : Verity.Release) : Records lg (fun log => f (rl :: log)) (pre ++ [rl]) :=
  fun log s o ho => by
    rw [h _ s o ho]
    simp

theorem Records.out {lg : β → List Verity.Release} {f : List Verity.Release → Game β} {pre : List Verity.Release}
    (h : Records lg f pre) (rl : Verity.Release) (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (s : Strategy (relK k fs ts fun log => f (rl :: log))) (τ : Tape (relK k fs ts fun log => f (rl :: log)) s) :
    ∃ log, lg (Tape.out _ s τ) = pre ++ rl :: log := by
  obtain ⟨log, s', h'⟩ := Verity.Sampled.playReaches_relK k fs ts _ s (playReaches_tapeOut _ s τ)
  exact ⟨log, h _ s' _ h'⟩

/-- The sampling design with `AgFrom` along `relK k`, at every layout, trees, releases before and continuation that records them. -/
def RecLawA {β : Type} (lg : β → List Verity.Release) {k : ℕ} (rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k)
    (gr : GapRec β k) : Prop :=
  ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β),
    Records lg f pre → ∀ s : Strategy (relK k fs ts f),
      LawA (relK k fs ts f) s (fun r => rc r fs ts pre f s) (fun r o => AgFrom L lg r pre o) (gr fs ts pre f s)

theorem law_leafA {lg : β → List Verity.Release} {f : List Verity.Release → Game β} {pre : List Verity.Release}
    (hf : Records lg f pre) (s : Strategy (f [])) :
    LawA (R := Fin K → FlockDeploy.Coin) (f []) s (fun _ t => t) (fun r o => AgFrom L lg r pre o) 0 :=
  law_idA fun _ t => agFrom_nil L (hf [] s _ (playReaches_tapeOut _ s t))

theorem law_freshA {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    {gr : GapRec β k} (hlaw : RecLawA L lg rc gr) (hrc : RecCongr L rc) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (hf : Records lg f pre) (l : String)
    (b : ByteArray) (s : Strategy (freshG k fs ts f l b)) :
    LawA (freshG k fs ts f l b) s (fun r => placeFresh L r k (rc r) fs ts pre f l b s)
      (fun r o => AgFrom L lg r pre o) (gapFresh k gr fs ts pre f l b s) := by
  unfold placeFresh gapFresh
  refine law_optA _ _ (fun s => law_leafA L hf s) (fun fs' s' => ?_) _ s
  rcases hs : slot L K pre (.fresh l) b 4 with _ | ⟨i, u⟩
  · refine law_keepA s' (Ag' := fun x r o => AgFrom L lg r (pre ++ [⟨.fresh l, b, List.ofFn x⟩]) o) ?_ fun x => ?_
    · intro x r τ hA
      exact (agFrom_cons L (hf.out ⟨.fresh l, b, List.ofFn x⟩ k fs' ts _ _).choose_spec).2
        ⟨relGood_of_none L hs, hA⟩
    · exact hlaw _ _ _ _ (hf.cons _) _
  · have hv := splits_fresh i u (slot_spec L hs).2.2
    refine law_setA (R := Fin K → FlockDeploy.Coin) s'
      (Ag' := fun x r o => AgFrom L lg r (pre ++ [⟨.fresh l, b, List.ofFn x⟩]) o) _ _ hv ?_ ?_ ?_ fun x => ?_
    · intro x r r' τ hr
      exact hrc _ _ _ _ _ _ r r' (offChunk_agree L hs rfl hr _)
    · intro x r r' o hr
      exact logGood_congr L _ _ (offChunk_agree L hs rfl hr _)
    · intro r τ hA
      refine (agFrom_cons L (hf.out ⟨.fresh l, b, List.ofFn (freshVal (r i) u)⟩ k fs' ts _ _).choose_spec).2
        ⟨fun j u' hs' => ?_, hA⟩
      have e := hs.symm.trans hs'
      cases e
      dsimp only [widthOf]
      exact ofFn_freshVal _ _
    · exact hlaw _ _ _ _ (hf.cons _) _

theorem law_helloA {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    {gr : GapRec β k} (hlaw : RecLawA L lg rc gr) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (hf : Records lg f pre)
    (spec : Flock.CoinTree.Spec) (ν : ByteArray) (s : Strategy (helloG k fs ts f spec ν)) :
    LawA (helloG k fs ts f spec ν) s (fun r => placeHello k (rc r) fs ts pre f spec ν s)
      (fun r o => AgFrom L lg r pre o) (gapHello k gr fs ts pre f spec ν s) := by
  unfold placeHello gapHello
  refine law_iteA _ _ _ (fun s' => ?_) (fun s => law_leafA L hf s) s
  refine law_keepA s' (Ag' := fun tr r o => AgFrom L lg r
    (pre ++ [⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩]) o) ?_ fun tr => ?_
  · intro tr r τ hA
    exact (agFrom_cons L (hf.out ⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩ k fs _ _
      _).choose_spec).2 ⟨relGood_of_none L rfl, hA⟩
  · exact hlaw _ _ _ _ (hf.cons _) _

theorem law_roundA {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    {gr : GapRec β k} (hlaw : RecLawA L lg rc gr) (hrc : RecCongr L rc) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (hf : Records lg f pre) (i sn n : ℕ)
    (b : ByteArray) (T : OpenTree) (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog))
    (s : Strategy (openP k fs ts f i sn n b T p)) :
    LawA (openP k fs ts f i sn n b T p) s (fun r => placeRound L r k (rc r) fs ts pre f i sn n b T p s)
      (fun r o => AgFrom L lg r pre o) (gapRound L K k gr fs ts pre f i sn n b T p s) := by
  classical
  unfold placeRound gapRound
  rcases hs : slot L K pre (.opening i sn n) b n with _ | ⟨j, u⟩
  · refine law_keepA s (Ag' := fun dr r o => AgFrom L lg r
      (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩]) o) ?_ fun dr => ?_
    · intro dr r τ hA
      exact (agFrom_cons L (hf.out ⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ k fs _ _ _).choose_spec).2
        ⟨relGood_of_none L hs, hA⟩
    · exact hlaw _ _ _ _ (hf.cons _) _
  · obtain ⟨-, -, hu⟩ := slot_spec L hs
    have hv := splits_open j u n hu
    refine law_monoA (law_congrA (P' := fun r : Fin K → FlockDeploy.Coin =>
        placeCoin _ (fun dr => openW n (bitsOf (openElems (r j) u n)) dr)
        (fun dr s₅ t₅ => rc r fs (ts.set i (T.bump p.1)) (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩])
          (fun log => f (⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ :: log)) s₅ t₅) s) ?_
      (law_writeA (R := Fin K → FlockDeploy.Coin) s
        (Ag' := fun dr r o => AgFrom L lg r (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩]) o)
        (G := fun dr => gr fs (ts.set i (T.bump p.1)) (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩])
          (fun log => f (⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ :: log)) (s dr))
        (fun (e : Fin n → Fin 128 → Bool) dr => openW n (bitsOf e) dr) _ _ hv ?_ ?_
        (fun e => (Finset.univ.filter fun e' : TreeOpenable T.nonce T.spec T.sample p.1 p.2 =>
          TreeDraw.opened e'.1 n = bitsOf e).Nonempty) ?_ fun dr => ?_)) ?_
    · intro r t
      show placeCoin _ (fun dr => openW n (openBits (r j) u n) dr) _ s t = _
      rw [openBits_eq]
    · intro dr r r' τ hr
      exact hrc _ _ _ _ _ _ r r' (offChunk_agree L hs rfl hr _)
    · intro dr r r' o hr
      exact logGood_congr L _ _ (offChunk_agree L hs rfl hr _)
    · intro r d τ hg hA
      refine (agFrom_cons L (hf.out ⟨.opening i sn n, b,
        TreeDraw.opened (openW n (bitsOf (openElems (r j) u n)) d).1 n⟩ k fs _ _ _).choose_spec).2
        ⟨fun j' u' hs' => ?_, hA⟩
      have e := hs.symm.trans hs'
      cases e
      show TreeDraw.opened (openW n (bitsOf (openElems (r j) u n)) d).1 n = openBits (r j) u n
      rw [openBits_eq]
      exact (Finset.mem_filter.1 (pickIn_mem hg d)).2
    · exact hlaw _ _ _ _ (hf.cons _) _
    · exact le_rfl

theorem law_openA {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    {gr : GapRec β k} (hlaw : RecLawA L lg rc gr) (hrc : RecCongr L rc) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (hf : Records lg f pre) (i sn n : ℕ)
    (b : ByteArray) (s : Strategy (openG k fs ts f i sn n b)) :
    LawA (openG k fs ts f i sn n b) s (fun r => placeOpen L r k (rc r) fs ts pre f i sn n b s)
      (fun r o => AgFrom L lg r pre o) (gapOpen L K k gr fs ts pre f i sn n b s) := by
  unfold placeOpen gapOpen
  refine law_iteA _ _ _ (fun s' => ?_) (fun s => law_leafA L hf s) s
  refine law_optA _ _ (fun s => law_leafA L hf s) (fun T s₃ => ?_) _ s'
  refine law_optA _ _ (fun s => law_leafA L hf s) (fun p s₄ => ?_) _ s₃
  exact law_roundA L hlaw hrc fs ts pre f hf i sn n b T p s₄

/-- **The sampling design with the agreement along the releases**: at every layout, trees, releases before and continuation that
records them, the coin map `placeK` is uniform up to `gapK`, counted where the releases after `pre` agree. -/
theorem law_placeKA {β : Type} (lg : β → List Verity.Release) : ∀ (k : ℕ) (fs : List (String × ℕ))
    (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β), Records lg f pre →
    ∀ s : Strategy (relK k fs ts f),
      LawA (R := Fin K → FlockDeploy.Coin) (relK k fs ts f) s (fun r => placeK L r k fs ts pre f s)
        (fun r o => AgFrom L lg r pre o) (gapK L K k fs ts pre f s)
  | 0, _, _, _, _, hf, s => law_leafA L hf s
  | _ + 1, _, _, _, _, hf, ⟨none, s'⟩ => law_leafA L hf s'
  | k + 1, fs, ts, pre, f, hf, ⟨some (.fresh l, b), s'⟩ => by
    refine law_sendA _ _ _ ?_
    exact law_congrA (fun r t => rfl) (law_freshA L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f hf s => law_placeKA lg k fs ts pre f hf s)
      (fun fs ts pre f s t r r' h => placeK_congr L k fs ts pre f s t r r' h) fs ts pre f hf l b s')
  | k + 1, fs, ts, pre, f, hf, ⟨some (.hello spec, ν), s'⟩ => by
    refine law_sendA _ _ _ ?_
    exact law_congrA (fun r t => rfl) (law_helloA L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f hf s => law_placeKA lg k fs ts pre f hf s) fs ts pre f hf spec ν s')
  | k + 1, fs, ts, pre, f, hf, ⟨some (.opening i sn n, b), s'⟩ => by
    refine law_sendA _ _ _ ?_
    exact law_congrA (fun r t => rfl) (law_openA L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f hf s => law_placeKA lg k fs ts pre f hf s)
      (fun fs ts pre f s t r r' h => placeK_congr L k fs ts pre f s t r r' h) fs ts pre f hf i sn n b s')

/-! ## To the device's release -/

/-- **Once a round is past its coin, no release after fits.** -/
theorem fitOf_zero (i : ℕ) : ∀ (log q : List Verity.Release), 65536 < usedOf L i [] q → fitOf L i q log = 0
  | [], _, _ => rfl
  | rl :: log, q, h => by
    have hs := usedOf_snoc L i q rl
    have h1 : ¬ (locAt L q rl.frame rl.answers = some i ∧ usedOf L i [] q + widthOf rl.frame ≤ 65536) :=
      fun ⟨_, hw⟩ => by omega
    have h2 := fitOf_zero i log (q ++ [rl]) (by rw [hs]; split_ifs <;> omega)
    simp only [fitOf, if_neg h1, h2, Nat.add_zero]

/-- **Where every release agrees, the bits collected for round `i` are its coin's**: bit `t` of what the releases of
`log` after `q` carry for round `i`, below what those that fit carry, is the run's coin's bit past the `usedOf … q`
elements `q` gave it. -/
theorem devBits_getD (r : Fin K → FlockDeploy.Coin) (i : Fin K) :
    ∀ (log q : List Verity.Release), LogGood L r q log → ∀ t, t < 128 * fitOf L i.val q log →
      (devBits L i.val q log).getD t false = coinBit (r i) (128 * usedOf L i.val [] q + t)
  | [], q, _, t, ht => by simp [fitOf] at ht
  | rl :: log, q, ⟨hg, hgs⟩, t, ht => by
    have hsnoc := usedOf_snoc L i.val q rl
    simp only [devBits]
    simp only [fitOf] at ht
    by_cases hloc : locAt L q rl.frame rl.answers = some i.val
    · rw [if_pos hloc]
      rw [if_pos hloc] at hsnoc
      by_cases hfitw : usedOf L i.val [] q + widthOf rl.frame ≤ 65536
      · rw [if_pos ⟨hloc, hfitw⟩] at ht
        have hslot : slot L K q rl.frame rl.answers (widthOf rl.frame) = some (i, usedOf L i.val [] q) := by
          simp only [slot, hloc, dif_pos i.isLt, if_pos hfitw, Fin.eta]
        have hc := hg i _ hslot
        have hlen : rl.coins.length = 128 * widthOf rl.frame := by
          rw [hc]
          unfold openBits
          exact List.length_ofFn
        by_cases htw : t < 128 * widthOf rl.frame
        · have hg' : t < (openBits (r i) (usedOf L i.val [] q) (widthOf rl.frame)).length := by
            unfold openBits
            rw [List.length_ofFn]
            exact htw
          rw [List.getD_append rl.coins (devBits L i.val (q ++ [rl]) log) false t (by rw [hlen]; exact htw), hc,
            List.getD_eq_getElem _ false hg']
          simp only [openBits, List.getElem_ofFn]
        · rw [List.getD_append_right rl.coins (devBits L i.val (q ++ [rl]) log) false t (by rw [hlen]; omega), hlen,
            devBits_getD r i log (q ++ [rl]) hgs (t - 128 * widthOf rl.frame) (by omega), hsnoc]
          congr 1
          omega
      · rw [if_neg (fun h => hfitw h.2), fitOf_zero L i.val log (q ++ [rl]) (by rw [hsnoc]; omega)] at ht
        omega
    · rw [if_neg hloc] at hsnoc
      rw [if_neg (fun h => hloc h.1)] at ht
      rw [if_neg hloc, List.nil_append, devBits_getD r i log (q ++ [rl]) hgs t (by omega), hsnoc, Nat.add_zero]

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **Where every release of an output's log agrees, `ElemAgrees` holds.** -/
theorem elemAgrees_of_logGood {A : Live₂} {x : Msg} (reads : ℕ → Prop) {o : Verity.Exchange} (r : Coins A x)
    (h : LogGood L r [] o.log) : ElemAgrees L A x reads o r := by
  intro i _ c hc j hj
  unfold dev at hc
  by_cases h0 : usedOf L i.val [] o.log = 0
  · rw [if_pos h0] at hc
    cases hc
  · rw [if_neg h0] at hc
    cases hc
    refine eq_of_bit fun b hb => ?_
    dsimp only
    rw [bit_elemOfBits _ hb]
    have hj' := j.isLt
    rw [devBits_getD L r i o.log [] h (128 * j.val + b) (by omega)]
    unfold coinBit FlockDeploy.Coin.at
    have e₁ : (128 * usedOf L i.val [] [] + (128 * j.val + b)) / 128 = j.val := by
      simp only [usedOf]
      omega
    have e₂ : (128 * usedOf L i.val [] [] + (128 * j.val + b)) % 128 = b := by
      simp only [usedOf]
      omega
    rw [e₁, e₂, dif_pos hj']

/-! ## The sampling design -/

/-- **The coin map is uniform up to the hiding term, counted where the releases agree**, at the period's game. -/
theorem law_placeA (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (K : ℕ) :
    LawA (R := Fin K → FlockDeploy.Coin) (V.game θ F c) σ (fun r t => place V θ F c σ L r t)
      (fun r o => AgFrom L Verity.Exchange.log r [] o) (gapOf V θ F c σ L K) :=
  law_alongA (game_eq V θ F c)
    (Pl := fun r s t => placeK L r (frames (V.families θ)) (V.families θ) [] [] (finish (V.signer θ) θ F c s.1) s.2 t)
    (G := fun s => gapK L K (frames (V.families θ)) (V.families θ) [] [] (finish (V.signer θ) θ F c s.1) s.2)
    (fun s => by
      rcases s with ⟨m, s'⟩
      refine law_sendA _ _ _ ?_
      exact law_congrA (fun r t => rfl) (law_placeKA L Verity.Exchange.log _ _ _ _ _
        (fun log _ _ h => by rw [List.nil_append]; exact Verity.Sampled.log_of_finish h) s')) σ

end Along

/-- **The coin map's law holds** (`CoinMapHolds`): at every vocabulary, statement, strategy, locator, prover and set
of read rounds, `CoinMapLaw` at the frame of `L`, the agreement `ElemAgrees` and `hide = gapOf`. -/
theorem coinMapHolds {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] :
    CoinMapHolds (d := d) (α := α) := by
  classical
  intro V A x θ F c σ L P reads E
  have h := law_placeA L V θ F c σ (A.L.V.k x) (fun b => if E b then 1 else 0) fun b => by
    by_cases hb : E b <;> simp [hb]
  calc Verity.Game.prob E (V.game θ F c) σ
      = prCoin fun t : Tape (V.game θ F c) σ => E (Tape.out _ σ t) := prob_eq_prCoin _ _ _
    _ ≤ (avg fun r : Coins A x => avg fun t : Tape (V.game θ F c) σ =>
          if AgFrom L Verity.Exchange.log r [] (Tape.out _ σ (place V θ F c σ L r t)) then
            (if E (Tape.out _ σ (place V θ F c σ L r t)) then 1 else 0) else 0) + gapOf V θ F c σ L (A.L.V.k x) := h
    _ ≤ (avg fun r : Coins A x => avg fun t : Tape (V.game θ F c) σ =>
          if E (Tape.out _ σ (place V θ F c σ L r t)) ∧
            ElemAgrees L A x reads (Tape.out _ σ (place V θ F c σ L r t)) r then 1 else 0) +
          gapOf V θ F c σ L (A.L.V.k x) := by
        refine add_le_add (avg_mono fun r => avg_mono fun t => ?_) le_rfl
        by_cases hA : AgFrom L Verity.Exchange.log r [] (Tape.out _ σ (place V θ F c σ L r t))
        · by_cases hE : E (Tape.out _ σ (place V θ F c σ L r t))
          · rw [if_pos hA, if_pos hE, if_pos ⟨hE, elemAgrees_of_logGood L reads r hA⟩]
          · rw [if_pos hA, if_neg hE]
            exact zero_le
        · rw [if_neg hA]
          exact zero_le
    _ = prCoin (fun s : TapeOf V θ F c σ × Coins A x =>
          E ((frame V A x θ F c σ L P).out s.1 s.2) ∧
            ElemAgrees L A x reads ((frame V A x θ F c σ L P).out s.1 s.2) s.2) +
          gapOf V θ F c σ L (A.L.V.k x) := by
        unfold prCoin
        rw [avg_prod, avg_swap]
        exact congrArg (· + gapOf V θ F c σ L (A.L.V.k x)) (congrArg avg (funext fun t =>
          congrArg avg (funext fun r => ite_inst _ _ _ _)))
    _ = _ := by rw [prCoin_eq_ofReal_pr]

end Verity.SampledTwo.CoinMap
