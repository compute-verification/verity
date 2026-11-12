module

public import Definitions.Core.Game.TwoRun
public import Proofs.Core.Game.Seq
public import Proofs.Core.Game.Prob
public import Definitions.Core.Interaction

/-!
# The most likely revealed message, and the two-continuation finder

A play may fix a message of the prover's by a digest of it and reveal the message only later. A reading of the play that
needs the message where the digest fixes it can take the most likely one there (`revealMode`): of the messages the plays
from that node reveal, one with the largest probability. A play that reveals another message misses. Two independent
continuations from the node then reveal two different messages with one digest, a collision: a continuation that
reveals `w` differs from every continuation that reveals another message, and those are at least as likely as a miss,
since the most likely message is at least as likely as `w`. So at one node the miss squared is at most the chance of a
collision (`sq_reveal_miss_le`), and by Jensen over the coins before the node the same holds along the play
(`DigestCut.sq_miss_le`).

* `DigestCut`: where along each play the message is fixed (`here`), and what the play's output reveals for it.
* `DigestCut.miss`, and `DigestCut.missP` in `ℝ≥0∞`, averaged as `prob` is: the probability that the play reveals, at
  its cut, a message other than the most likely one there.
* `DigestCut.forkG`, `DigestCut.forkS`: the play up to its cut, then two independent continuations, with the messages
  they reveal.
* `CoinTape`: a uniform tape for a strategy's plays, which gives the play its sampling design (`expect_tape`, `pr_tape`). A fork's
  tape is the two-continuation finder, a `Claims.Finder` at a SHA-512 budget on every coin (`DigestCut.finder`).
* `DigestCut.miss_le`, `DigestCut.missP_le`: the miss is at most the square root of the finder's success; at
  `cr/sha-512` (`Claims.StrictBirthday`) at the budget `q`, at most `√(q²/2^513)` (`DigestCut.missP_le_strict`).
* `DigestCut.prob_le_revealMode_add_missP`: at a cut, an event of plays that reveal a message is at most as likely as
  the event with the most likely message revealed, plus the miss.

`revealMode`, `DigestCut`, its fork, `CoinTape` and `DigestCut.finder`, with the lemmas they are built from, are
`Definitions.Core.Game.TwoRun`'s; `DigestCut.miss` and `DigestCut.missP` are here.
-/

@[expose] public section

namespace Verity

namespace Game

open scoped ENNReal

variable {α : Type}

/-! ## The outputs a strategy reaches, and the most likely revealed message -/

variable {V : Type}

theorem revealPr_le_revealMode [Nonempty V] (rv : α → Option V) (g : Game α) (s : Strategy g) (w : V) :
    revealPr rv g s w ≤ revealPr rv g s (revealMode rv g s) :=
  (exists_revealMode rv g s).choose_spec w

theorem ind_reveal_split (rv : α → Option V) (w : V) (o : α) :
    ind (fun o => ∃ u, rv o = some u) o =
      ind (fun o => ∃ u, rv o = some u ∧ u ≠ w) o + ind (fun o => rv o = some w) o := by
  unfold ind
  rcases h : rv o with _ | u
  · simp [h]
  · by_cases hu : u = w
    · subst hu
      simp [h]
    · simp [h, hu]

/-- **At one node, the miss squared is at most the fork**: the probability that the play reveals a message other than
the most likely one, squared, is at most the expectation over two independent plays of any `F` that is at least `1`
wherever the two reveal different messages. -/
theorem sq_reveal_miss_le [Nonempty V] (rv : α → Option V) (g : Game α) (s : Strategy g) (F : α → α → ℝ)
    (hF0 : ∀ o o', 0 ≤ F o o')
    (hF1 : ∀ o o' w w', rv o = some w → rv o' = some w' → w ≠ w' → 1 ≤ F o o') :
    expect (ind fun o => ∃ w, rv o = some w ∧ w ≠ revealMode rv g s) g s ^ 2 ≤
      expect (fun o => expect (F o) g s) g s := by
  set c := revealMode rv g s
  set M := expect (ind fun o => ∃ w, rv o = some w ∧ w ≠ c) g s
  set A := expect (ind fun o => ∃ w, rv o = some w) g s
  have hsplit : ∀ w, A = expect (ind fun o => ∃ u, rv o = some u ∧ u ≠ w) g s + revealPr rv g s w := fun w => by
    rw [revealPr, ← expect_add]
    exact congrArg (fun f => expect f g s) (funext (ind_reveal_split rv w))
  have hrow : ∀ o, M * ind (fun o => ∃ w, rv o = some w) o ≤ expect (F o) g s := by
    intro o
    rcases h : rv o with _ | w
    · have : ind (fun o => ∃ w, rv o = some w) o = 0 := by
        unfold ind
        exact ite_eq_right (by simp [h])
      rw [this, mul_zero]
      exact expect_nonneg (hF0 o) g s
    · have h1 : ind (fun o => ∃ w, rv o = some w) o = 1 := by
        unfold ind
        exact ite_eq_left ⟨w, h⟩
      rw [h1, mul_one]
      have hM : M ≤ expect (ind fun o' => ∃ u, rv o' = some u ∧ u ≠ w) g s := by
        have := hsplit c
        have := hsplit w
        have := revealPr_le_revealMode rv g s w
        linarith
      refine hM.trans (expect_mono (fun o' => ?_) g s)
      unfold ind
      split_ifs with he
      · obtain ⟨u, hu, hne⟩ := he
        exact hF1 o o' w u h hu (Ne.symm hne)
      · exact hF0 o o'
  have hM0 : 0 ≤ M := expect_nonneg (ind_nonneg _) g s
  have hMA : M ≤ A := expect_mono (ind_mono fun o ⟨w, hw, _⟩ => ⟨w, hw⟩) g s
  have hAM : M * A ≤ expect (fun o => expect (F o) g s) g s := by
    rw [← expect_const_mul]
    exact expect_mono hrow g s
  rw [sq]
  exact (mul_le_mul_of_nonneg_left hMA hM0).trans hAM

/-! ## Cuts: where a play fixes a message, and the fork from there -/

namespace DigestCut

/-- **The miss at the cut**: the probability that the play reveals, at its cut, a message other than the most likely
one there (`revealMode`). -/
noncomputable def miss [Nonempty V] : {g : Game α} → DigestCut V g → Strategy g → ℝ
  | _, .skip, _ => 0
  | g, .here rv, s => expect (ind fun o => ∃ w, rv o = some w ∧ w ≠ revealMode rv g s) g s
  | _, .send K, s => miss (K s.1) s.2
  | _, @DigestCut.coin _ _ C iF _ _ K, s => (∑ c : C, miss (K c) (s c)) / (@Fintype.card C iF)

/-- **The miss at the cut, as `prob` averages**: `miss` in `ℝ≥0∞` (`missP_toReal`). -/
noncomputable def missP [Nonempty V] : {g : Game α} → DigestCut V g → Strategy g → ℝ≥0∞
  | _, .skip, _ => 0
  | g, .here rv, s => prob (fun o => ∃ w, rv o = some w ∧ w ≠ revealMode rv g s) g s
  | _, .send K, s => missP (K s.1) s.2
  | _, @DigestCut.coin _ _ C _ _ _ K, s => avg fun c : C => missP (K c) (s c)

theorem miss_nonneg [Nonempty V] : ∀ {g : Game α} (K : DigestCut V g) (s : Strategy g), 0 ≤ K.miss s
  | _, .skip, _ => le_rfl
  | g, .here _, s => expect_nonneg (ind_nonneg _) g s
  | _, .send K, s => miss_nonneg (K s.1) s.2
  | _, @DigestCut.coin _ _ C _ _ _ K, s =>
      div_nonneg (Finset.sum_nonneg fun c _ => miss_nonneg (K c) (s c)) (Nat.cast_nonneg _)

theorem missP_le_one [Nonempty V] : ∀ {g : Game α} (K : DigestCut V g) (s : Strategy g), K.missP s ≤ 1
  | _, .skip, _ => zero_le_one
  | g, .here _, s => prob_le_one _ g s
  | _, .send K, s => missP_le_one (K s.1) s.2
  | _, @DigestCut.coin _ _ C _ _ _ K, s => (avg_mono fun c => missP_le_one (K c) (s c)).trans (avg_const 1).le

theorem missP_toReal [Nonempty V] :
    ∀ {g : Game α} (K : DigestCut V g) (s : Strategy g), (K.missP s).toReal = K.miss s
  | _, .skip, _ => rfl
  | g, .here rv, s => prob_toReal _ g s
  | _, .send K, s => missP_toReal (K s.1) s.2
  | _, @DigestCut.coin _ _ C _ _ _ K, s => by
      simp only [missP, miss, avg]
      have hC : (Fintype.card C : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      have hfin : ∀ c, missP (K c) (s c) ≠ ⊤ := fun c =>
        ne_top_of_le_ne_top ENNReal.one_ne_top (missP_le_one (K c) (s c))
      rw [ENNReal.toReal_sum fun c _ => ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 hC) (hfin c)]
      simp only [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast, missP_toReal (K _) (s _)]
      rw [← Finset.mul_sum, inv_mul_eq_div]

/-- **The miss, squared, is at most the fork's chance of a collision**, where every two messages a cut's plays reveal
have one image under `H` (`Agrees`): at the cut, by `sq_reveal_miss_le`, and over the coins before it by Jensen. -/
theorem sq_miss_le [Nonempty V] {D : Type} (H : V → D) :
    ∀ {g : Game α} (K : DigestCut V g), K.Agrees H → ∀ s : Strategy g,
      K.miss s ^ 2 ≤ expect (ind fun y => ∃ u v, y = some (u, v) ∧ u ≠ v ∧ H u = H v) (K.forkG s) (K.forkS s)
  | _, .skip, _, _ => by
      show (0 : ℝ) ^ 2 ≤ ind _ none
      unfold ind
      simp
  | g, .here rv, hK, s => by
      show _ ≤ expect _ (g.bind _) (Strategy.bind g _ s _)
      rw [expect_bind]
      simp only [expect_map]
      refine sq_reveal_miss_le rv g s _ (fun _ _ => ind_nonneg _ _) fun o o' w w' h h' hne => ?_
      rw [h, h']
      unfold ind
      rw [ite_eq_left ⟨w, w', rfl, hne, hK o o' w w' h h'⟩]
  | _, .send K, hK, s => sq_miss_le H (K s.1) (hK s.1) s.2
  | _, @DigestCut.coin _ _ C _ _ _ K, hK, s => by
      show ((∑ c, miss (K c) (s c)) / Fintype.card C) ^ 2 ≤
        (∑ c, expect _ (forkG (K c) (s c)) (forkS (K c) (s c))) / Fintype.card C
      have hC : (0 : ℝ) < Fintype.card C := by exact_mod_cast Fintype.card_pos
      have hcs : (∑ c, miss (K c) (s c)) ^ 2 ≤ Fintype.card C * ∑ c, miss (K c) (s c) ^ 2 := by
        simpa using sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun c => miss (K c) (s c))
      have hle : ∑ c, miss (K c) (s c) ^ 2 ≤
          ∑ c, expect (ind fun y => ∃ u v, y = some (u, v) ∧ u ≠ v ∧ H u = H v) (forkG (K c) (s c))
            (forkS (K c) (s c)) :=
        Finset.sum_le_sum fun c _ => sq_miss_le H (K c) (hK c) (s c)
      rw [div_pow, div_le_div_iff₀ (by positivity) hC]
      calc (∑ c, miss (K c) (s c)) ^ 2 * Fintype.card C
          ≤ Fintype.card C * (∑ c, miss (K c) (s c) ^ 2) * Fintype.card C :=
            mul_le_mul_of_nonneg_right hcs hC.le
        _ ≤ Fintype.card C * (∑ c, expect (ind fun y => ∃ u v, y = some (u, v) ∧ u ≠ v ∧ H u = H v)
              (forkG (K c) (s c)) (forkS (K c) (s c))) * Fintype.card C := by gcongr
        _ = _ := by ring

/-- **A reading that reveals only messages with one image `d` agrees.** -/
theorem here_agrees {D : Type} (H : V → D) {g : Game α} (rv : α → Option V) (d : D)
    (h : ∀ o w, rv o = some w → H w = d) : (DigestCut.here (g := g) rv).Agrees H :=
  fun o o' w w' hw hw' => (h o w hw).trans (h o' w' hw').symm

/-- **At a cut, an event of plays that reveal a message** is at most as likely as the event with the most likely message
revealed, plus the miss there. -/
theorem prob_le_revealMode_add_missP [Nonempty V] (rv : α → Option V) (g : Game α) (s : Strategy g) (E : α → Prop)
    (hE : ∀ o, E o → ∃ w, rv o = some w) :
    prob E g s ≤
      prob (fun o => E o ∧ rv o = some (revealMode rv g s)) g s + (DigestCut.here (g := g) rv).missP s := by
  refine (prob_mono (fun o ho => ?_) g s).trans (prob_or_le _ _ g s)
  obtain ⟨w, hw⟩ := hE o ho
  by_cases hm : w = revealMode rv g s
  · exact Or.inl ⟨ho, hm ▸ hw⟩
  · exact Or.inr ⟨w, hw, hm⟩

end DigestCut

/-! ## A uniform tape for a strategy's plays -/

instance instNonemptyCoinTape (g : Game α) (s : Strategy g) : Nonempty (CoinTape g s) := CoinTape.nonempty g s

/-- **A uniform dependent function's value at `c` is uniform**: averaging `F` at the value at `c` over the functions
is averaging `F` over the values. -/
theorem sum_pi_eval {C : Type} [Fintype C] [DecidableEq C] {T : C → Type} [∀ c, Fintype (T c)]
    [∀ c, Nonempty (T c)] (c : C) (F : T c → ℝ) :
    (∑ φ : (c' : C) → T c', F (φ c)) / Fintype.card ((c' : C) → T c') = (∑ t, F t) / Fintype.card (T c) := by
  have hR : (Fintype.card ((c' : {c' // c' ≠ c}) → T c') : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  rw [Fintype.sum_equiv (Equiv.piSplitAt c T) (fun φ => F (φ c)) (fun p => F p.1) (fun φ => rfl),
    Fintype.sum_prod_type, Fintype.card_congr (Equiv.piSplitAt c T), Fintype.card_prod]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Nat.cast_mul]
  rw [← Finset.mul_sum, mul_comm (Fintype.card (T c) : ℝ), mul_div_mul_left _ _ hR]

/-- **A uniform tape gives the play its sampling design**: the average over tapes of `f` at the output is `f`'s expectation. -/
theorem expect_tape (f : α → ℝ) : ∀ (g : Game α) (s : Strategy g),
    (∑ x : CoinTape g s, f (CoinTape.run g s x)) / Fintype.card (CoinTape g s) = expect f g s
  | .ret a, _ => by
      show (∑ _x : PUnit, f a) / Fintype.card PUnit = f a
      simp
  | .send _ k, s => expect_tape f (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      classical
      let _ : ∀ c, Fintype (CoinTape (k c) (s c)) := fun c => CoinTape.fintype (k c) (s c)
      have _ : ∀ c, Nonempty (CoinTape (k c) (s c)) := fun c => CoinTape.nonempty (k c) (s c)
      show (∑ x : C × ((c : C) → CoinTape (k c) (s c)), f (CoinTape.run (k x.1) (s x.1) (x.2 x.1))) /
          Fintype.card (C × ((c : C) → CoinTape (k c) (s c))) = (∑ c, expect f (k c) (s c)) / Fintype.card C
      have hP : (Fintype.card ((c : C) → CoinTape (k c) (s c)) : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      have key : ∀ c, (∑ φ : (c' : C) → CoinTape (k c') (s c'), f (CoinTape.run (k c) (s c) (φ c))) =
          expect f (k c) (s c) * Fintype.card ((c' : C) → CoinTape (k c') (s c')) := fun c => by
        have h1 := sum_pi_eval (T := fun c => CoinTape (k c) (s c)) c (fun t => f (CoinTape.run (k c) (s c) t))
        rw [expect_tape f (k c) (s c)] at h1
        exact (div_eq_iff hP).1 h1
      rw [Fintype.sum_prod_type, Fintype.card_prod, Finset.sum_congr rfl fun c _ => key c, ← Finset.sum_mul,
        Nat.cast_mul, mul_div_mul_right _ _ hP]

theorem pr_tape (p : α → Prop) (g : Game α) (s : Strategy g) :
    Claims.Pr (fun x : CoinTape g s => p (CoinTape.run g s x)) = expect (ind p) g s := by
  classical
  rw [← expect_tape]
  unfold Claims.Pr Claims.prUniform
  congr 1
  rw [Finset.natCast_card_filter]
  exact Finset.sum_congr rfl fun x _ => by unfold ind; split_ifs <;> simp_all

namespace DigestCut

theorem finder_cost {g : Game α} (K : DigestCut Claims.Msg g) (s : Strategy g) (q : ℕ) (x : (K.finder s q).S) :
    (K.finder s q).cost x = q := rfl

theorem finder_pr (H : Claims.Msg → Claims.Msg) {g : Game α} (K : DigestCut Claims.Msg g) (s : Strategy g)
    (q : ℕ) : (K.finder s q).Pr H =
      expect (ind fun y => ∃ u v, y = some (u, v) ∧ Claims.Collision H u v) (K.forkG s) (K.forkS s) :=
  pr_tape (fun y => ∃ u v, y = some (u, v) ∧ Claims.Collision H u v) (K.forkG s) (K.forkS s)

/-- **The miss is at most the square root of the two-continuation finder's success**, where every two messages a cut's
plays reveal have one image under `H` (`Agrees`). -/
theorem miss_le (H : Claims.Msg → Claims.Msg) {g : Game α} (K : DigestCut Claims.Msg g) (hK : K.Agrees H)
    (s : Strategy g) (q : ℕ) : K.miss s ≤ Real.sqrt ((K.finder s q).Pr H) := by
  rw [finder_pr]
  exact (le_abs_self _).trans (Real.abs_le_sqrt (sq_miss_le H K hK s))

/-- **The miss at `cr/sha-512`**: where the finder at `q` calls meets the strict birthday bound, the miss is at most
`√(q²/2^513)`. -/
theorem miss_le_strict (H : Claims.Msg → Claims.Msg) {g : Game α} (K : DigestCut Claims.Msg g) (hK : K.Agrees H)
    (s : Strategy g) (q : ℕ) (hCR : Claims.StrictBirthday H (K.finder s q)) :
    K.miss s ≤ Real.sqrt ((q : ℝ) ^ 2 / 2 ^ (513 : ℝ)) :=
  (miss_le H K hK s q).trans (Real.sqrt_le_sqrt (hCR q fun _ => le_rfl))

theorem missP_le (H : Claims.Msg → Claims.Msg) {g : Game α} (K : DigestCut Claims.Msg g) (hK : K.Agrees H)
    (s : Strategy g) (q : ℕ) : K.missP s ≤ ENNReal.ofReal (Real.sqrt ((K.finder s q).Pr H)) := by
  rw [← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (K.missP_le_one s)), missP_toReal]
  exact ENNReal.ofReal_le_ofReal (miss_le H K hK s q)

theorem missP_le_strict (H : Claims.Msg → Claims.Msg) {g : Game α} (K : DigestCut Claims.Msg g) (hK : K.Agrees H)
    (s : Strategy g) (q : ℕ) (hCR : Claims.StrictBirthday H (K.finder s q)) :
    K.missP s ≤ ENNReal.ofReal (Real.sqrt ((q : ℝ) ^ 2 / 2 ^ (513 : ℝ))) := by
  rw [← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (K.missP_le_one s)), missP_toReal]
  exact ENNReal.ofReal_le_ofReal (miss_le_strict H K hK s q hCR)

end DigestCut

end Game

end Verity
