import Proofs.Accounting.TS.Drawn

/-!
# An honest coin-keyed proof layer meets (I8)'s premises

The positive twin of RT3-LOGIC-1's `coinLayer`: a served round's unit takes the round's salt σ_w and outputs a
receipt keyed by it (`window.round_key`, `h1tile.live_salt`), committed after σ_w in a tree of its own, while the
round's leaves, committed before σ_w, hold its operands. `keyedLayer n` is such a window over `n + 1` rounds, one proof a round: each
proof accepted, opening its round's operand (`false`), its relation reading σ_w and the receipt the proof gives on the
path (the operand keyed by σ_w). The receipt moves with σ_w (`keyedLayer_receipt_moves`), so as a leaf it would fork
with every redraw of σ_w, as `coinLayer` does. Read by the relation instead, binding (both forms) and proof soundness
hold at `0` and the relations pin (`keyedLayer_meets_budget`), so the restated premises hold for an honest
coin-dependent layer at the budget `S·(√δ_fork + n_π·δ_ZK) ≤ 2^-41`, `S = 48,000` (`keyedLayer_meets_budget_dc`).

The sampled form needs no restatement: `DrawnLayer` binds each position at its own cut, a unit's outputs at the draw,
after the coins its noise reads. `keyedDrawn n` is the served shape over `n + 1` rounds, one coin before each round's
draw (the salt) and one unit, whose slots are its operand (cut before the salt) and its receipt (cut at the draw), the
operand keyed by the salt. Its receipt moves with the salt (`keyedDrawn_receipt_moves`), and it meets binding at every
position's cut (both forms), joint soundness and pinning at `0`, within the budget (`keyedDrawn_meets_budget`).

`guess_rel_last`: for a prover that guesses round `w`'s receipt (`true`) instead of keying its operand, the committed
values satisfy every relation on one path and fail the last round's on its continuation that redraws the last coin, so
`ProofLayer.Of` no longer has the ideal game accept every last coin (RT3-LOGIC's `game_accepts_last`): the relation
reads the coin the receipt is keyed by.
-/

namespace Accounting.SecurityProofs.TS.Keyed

open Finset Accounting.TS Pouw.RelativeSecurity

theorem coinAt_splice {Ω : Type} : ∀ {n : ℕ} (w : Fin n) (x z : Paths Ω n),
    coinAt (splice (w : ℕ) x z) w = coinAt z w
  | 0, w, _, _ => w.elim0
  | _ + 1, ⟨0, _⟩, _, _ => rfl
  | _ + 1, ⟨w + 1, h⟩, x, z => coinAt_splice ⟨w, Nat.lt_of_succ_lt_succ h⟩ x.2 z.2

theorem coinAt_const {Ω : Type} (a : Ω) : ∀ {n : ℕ} (w : Fin n), coinAt (Paths.const a n) w = a
  | 0, w => w.elim0
  | _ + 1, ⟨0, _⟩ => rfl
  | _ + 1, ⟨w + 1, h⟩ => coinAt_const a ⟨w, Nat.lt_of_succ_lt_succ h⟩

theorem prob_zero {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : ∀ ω, ¬ p ω) : prob p = 0 := by
  classical
  rw [prob_eq_card, filter_false_of_mem fun ω _ => h ω, card_empty, Nat.cast_zero, zero_div]

/-- A unit's receipt: its operand keyed by the round's coin. -/
def key (v c : Bool) : Bool := xor v c

theorem key_inj {v c : Bool} (h : key false c = key v c) : v = false := by
  revert h
  cases v <;> cases c <;> decide

theorem key_flip (c : Bool) : key false c ≠ key false (!c) := by
  cases c <;> decide

/-- **An honest coin-keyed layer** over `n + 1` rounds, one proof a round: every proof accepted, opening its round's
leaf, the operand `false`, committed before σ_w; its relation, on the path, that the receipt the proof gives there (the
honest one, the operand keyed by σ_w) is the opened operand keyed by σ_w; a window right when every operand is the
honest one. -/
def keyedLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted _ _ _ := True
  opened _ _ _ := false
  rel x w _ v := key false (coinAt x w) = key v (coinAt x w)
  right _ c := ∀ w i, c w i = false

theorem keyedLayer_committed (n : ℕ) (x : Paths Bool (n + 1)) (w : Fin (n + 1)) (i : Fin 1) :
    (keyedLayer n).committed x w i = false := rfl

/-- **The receipt moves with σ_w**: on every path, redrawing round `w`'s coin changes round `w`'s receipt. -/
theorem keyedLayer_receipt_moves (n : ℕ) (w : Fin (n + 1)) (x : Paths Bool (n + 1)) :
    ∃ z, key false (coinAt x w) ≠ key false (coinAt (splice (w : ℕ) x z) w) := by
  refine ⟨Paths.const (!coinAt x w) (n + 1), ?_⟩
  rw [coinAt_splice, coinAt_const]
  exact key_flip _

theorem keyedLayer_binding (n : ℕ) : (keyedLayer n).BindingAt 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨_, i, hi⟩ := h
  exact hi (keyedLayer_committed n x w i).symm

/-- The fork form at `δ_fork` in every round: `BindingOfFork`'s right side at most `δ_fork`. -/
def ForkAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) (δfork : ℝ) : Prop :=
  ∀ w, prob (fun p : Paths Ω S × Paths Ω S => pf.Forked w p.1 p.2) ≤ δfork

theorem keyedLayer_fork (n : ℕ) : ForkAt (keyedLayer n) 0 := by
  intro w
  refine le_of_eq (prob_zero fun p h => ?_)
  obtain ⟨_, _, i, hi⟩ := h
  exact hi rfl

theorem keyedLayer_sound (n : ℕ) : (keyedLayer n).SoundAt 0 := by
  intro w i
  refine le_of_eq (prob_zero fun x h => ?_)
  exact h.2 rfl

theorem keyedLayer_pins (n : ℕ) : (keyedLayer n).Pins := by
  intro x c h w i
  exact key_inj (h w i)

/-- **The honest coin-keyed layer meets the restated premises at the budget**, over `S = n + 1` rounds: binding at
`δ_bind = 0` and its fork form at `δ_fork = 0`, proof soundness at `δ_ZK = 0` and pinning, with
`S·(δ_bind + n_π·δ_ZK)` and `S·(√δ_fork + n_π·δ_ZK)` both within `2^-41`. -/
theorem keyedLayer_meets_budget (n : ℕ) :
    ∃ δbind δfork δzk : ℝ, (keyedLayer n).BindingAt δbind ∧ ForkAt (keyedLayer n) δfork ∧
      (keyedLayer n).SoundAt δzk ∧ (keyedLayer n).Pins ∧
      ((n + 1 : ℕ) : ℝ) * (δbind + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧
      ((n + 1 : ℕ) : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 := by
  refine ⟨0, 0, 0, keyedLayer_binding n, keyedLayer_fork n, keyedLayer_sound n, keyedLayer_pins n, ?_, ?_⟩
  · simp only [mul_zero, add_zero]
    positivity
  · simp only [Real.sqrt_zero, mul_zero, add_zero]
    positivity

/-- `keyedLayer_meets_budget` at the sm_120 and DC headlines' `S = 48,000`. -/
theorem keyedLayer_meets_budget_dc :
    ∃ δbind δfork δzk : ℝ, (keyedLayer 47999).BindingAt δbind ∧ ForkAt (keyedLayer 47999) δfork ∧
      (keyedLayer 47999).SoundAt δzk ∧ (keyedLayer 47999).Pins ∧
      (48000 : ℝ) * (δbind + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧
      (48000 : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 := by
  obtain ⟨δb, δf, δz, hb, hf, hz, hp, h1, h2⟩ := keyedLayer_meets_budget 47999
  have e : ((47999 + 1 : ℕ) : ℝ) = 48000 := by norm_num
  rw [e] at h1 h2
  exact ⟨δb, δf, δz, hb, hf, hz, hp, h1, h2⟩

/-- A prover that guesses each round's receipt (`true`) rather than keying its operand: proofs accepted, the operand
`false` opened, and the relation that the guessed receipt is the operand keyed by σ_w. -/
def guessLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted _ _ _ := True
  opened _ _ _ := false
  rel x w _ v := true = key v (coinAt x w)
  right _ c := ∀ w i, c w i = false

/-- **The relation reads the last coin**: on the all-`true` path the guessing prover's committed values satisfy every
relation, and on its continuation that redraws the last round's coin as `false` they fail the last round's. So `Of`'s
tie to the ideal game reads the coin the receipt is keyed by, and a prover deficient in the last round is accepted by
the ideal game on its lucky coins only (RT3-LOGIC's `game_accepts_last` had it accepted on every last coin). -/
theorem guess_rel_last (n : ℕ) :
    (∀ w i, (guessLayer n).rel (Paths.const true (n + 1)) w i
        ((guessLayer n).committed (Paths.const true (n + 1)) w i)) ∧
      ¬ ∀ w i, (guessLayer n).rel (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) w i
        ((guessLayer n).committed (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) w i) := by
  refine ⟨fun w _ => ?_, fun h => ?_⟩
  · show true = key false (coinAt (Paths.const true (n + 1)) w)
    rw [coinAt_const]
    rfl
  · have hl := h (Fin.last n) 0
    change true = key false (coinAt (splice ((Fin.last n : Fin (n + 1)) : ℕ) (Paths.const true (n + 1))
      (Paths.const false (n + 1))) (Fin.last n)) at hl
    rw [coinAt_splice, coinAt_const] at hl
    exact absurd hl (by decide)

/-! ## The sampled form -/

theorem coinAt_spliceAt {C B : Type} {m : ℕ} : ∀ {n : ℕ} (w : Fin n) (k : ℕ) (x z : Paths (Paths C m × B) n),
    coinAt (spliceAt (w : ℕ) k x z) w = (splice k (coinAt x w).1 (coinAt z w).1, (coinAt z w).2)
  | 0, w, _, _, _ => w.elim0
  | _ + 1, ⟨0, _⟩, _, _, _ => rfl
  | _ + 1, ⟨w + 1, h⟩, k, x, z => coinAt_spliceAt ⟨w, Nat.lt_of_succ_lt_succ h⟩ k x.2 z.2

/-- The plurality of continuations that each open one value, the same on every continuation, is that value. -/
theorem plural_eq {X V : Type} [Fintype X] [Nonempty V] (sp : X → X → X) (opens : X → V → Prop) (f : X → V)
    (h : ∀ y v, opens y v ↔ v = f y) (x z₀ : X) (hf : ∀ z, f (sp x z) = f x) : plural sp opens x = f x := by
  classical
  have hA : ∀ z v, Alone opens (sp x z) v ↔ v = f x := fun z v => by
    rw [← hf z]
    exact ⟨fun hv => (h _ v).1 hv.1, fun hv => ⟨(h _ v).2 hv, fun v' hv' => ((h _ v').1 hv').trans hv.symm⟩⟩
  by_contra hne
  have h1 := Plurality.tally_le_plural (sp := sp) (opens := opens) x (f x)
  have h0 : tally sp opens x (plural sp opens x) = 0 := by
    unfold tally
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro z _ hz
    exact hne ((hA z _).1 hz)
  have hpos : 0 < tally sp opens x (f x) := by
    unfold tally
    rw [Finset.card_pos]
    exact ⟨z₀, Finset.mem_filter.2 ⟨Finset.mem_univ _, (hA z₀ (f x)).2 rfl⟩⟩
  omega

/-- Round `w`'s salt on a path of rounds with one coin before the draw. -/
def salt {n : ℕ} (x : Paths (Paths Bool 1 × Unit) n) (w : Fin n) : Bool := (coinAt x w).1.1

/-- Round `w`'s honest values: its operand (`false`), and its receipt, the operand keyed by the salt. -/
def val {n : ℕ} (x : Paths (Paths Bool 1 × Unit) n) (w : Fin n) (k : Bool) : Bool :=
  if k then key false (salt x w) else false

theorem salt_spliceAt {n : ℕ} (w : Fin n) (x z : Paths (Paths Bool 1 × Unit) n) :
    salt (spliceAt (w : ℕ) 1 x z) w = salt x w :=
  congrArg (fun p => p.1.1) (coinAt_spliceAt w 1 x z)

/-- **An honest coin-keyed sampled layer** over `n + 1` rounds, each with one coin before its draw (the salt) and one
unit, whose slots are its operand (`false`, position `(w, false)`, cut before the salt) and its receipt (position
`(w, true)`, cut at the draw, after the salt): every proof accepted, opening the honest values; the unit's relation and
rightness, that the receipt is the operand keyed by the salt. -/
def keyedDrawn (n : ℕ) : DrawnLayer Bool Unit (n + 1) 1 1 Unit Bool (Fin (n + 1) × Bool) Bool where
  draw _ _ := ()
  pos w _ k := (w, k)
  coin _ _ := 0
  cut ℓ := (ℓ.1, if ℓ.2 then Fin.last 1 else 0)
  src ℓ := if ℓ.2 then some (ℓ.1, ()) else none
  accepted _ _ _ := True
  opened x w _ k := val x w k
  rel x w _ v := v true = key (v false) (salt x w)
  right x w _ v := v true = key (v false) (salt x w)

theorem keyedDrawn_opens (n : ℕ) (y : Paths (Paths Bool 1 × Unit) (n + 1)) (w : Fin (n + 1)) (k v : Bool) :
    (keyedDrawn n).Opens y (w, k) v ↔ v = val y w k := by
  constructor
  · rintro ⟨w', i, k', hp, -, hv⟩
    have hp' : (w', k') = (w, k) := hp
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hp'
    exact hv.symm
  · intro hv
    exact ⟨w, 0, k, rfl, trivial, hv.symm⟩

theorem keyedDrawn_val_cont (n : ℕ) (x z : Paths (Paths Bool 1 × Unit) (n + 1)) (w : Fin (n + 1)) (k : Bool) :
    val ((keyedDrawn n).cont (w, k) x z) w k = val x w k := by
  cases k
  · rfl
  · show key false (salt (spliceAt (w : ℕ) 1 x z) w) = key false (salt x w)
    rw [salt_spliceAt]

theorem keyedDrawn_committed (n : ℕ) (x : Paths (Paths Bool 1 × Unit) (n + 1)) (w : Fin (n + 1)) (k : Bool) :
    (keyedDrawn n).committed x (w, k) = val x w k := by
  have h := plural_eq ((keyedDrawn n).cont (w, k)) (fun y v => (keyedDrawn n).Opens y (w, k) v)
    (fun y => val y w k) (fun y v => keyedDrawn_opens n y w k v) x x (fun z => keyedDrawn_val_cont n x z w k)
  exact h

/-- **The receipt moves with the salt**: redrawing round `w`'s salt (from its coin `0` on) changes its receipt, so a
cut before the salt would fail binding for this honest layer. -/
theorem keyedDrawn_receipt_moves (n : ℕ) (w : Fin (n + 1)) (x : Paths (Paths Bool 1 × Unit) (n + 1)) :
    ∃ z, key false (salt x w) ≠ key false (salt (spliceAt (w : ℕ) 0 x z) w) := by
  refine ⟨Paths.const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) (n + 1), ?_⟩
  have h : salt (spliceAt (w : ℕ) 0 x (Paths.const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) (n + 1))) w =
      !salt x w :=
    (congrArg (fun p => p.1.1)
      (coinAt_spliceAt w 0 x (Paths.const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) (n + 1)))).trans
      (congrArg (fun p => p.1.1) (coinAt_const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) w))
  rw [h]
  exact key_flip _

theorem keyedDrawn_binding (n : ℕ) : (keyedDrawn n).BindingAt 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨i, _, hi⟩ := h
  exact hi (funext fun k => (keyedDrawn_committed n x w k).symm)

theorem keyedDrawn_fork (n : ℕ) (ℓ : Fin (n + 1) × Bool) :
    prob (fun p : Paths (Paths Bool 1 × Unit) (n + 1) × Paths (Paths Bool 1 × Unit) (n + 1) =>
      (keyedDrawn n).Forked ℓ p.1 p.2) ≤ 0 := by
  obtain ⟨w, k⟩ := ℓ
  refine le_of_eq (prob_zero fun p h => ?_)
  obtain ⟨v, v', hv, hv', hne⟩ := h
  rw [keyedDrawn_opens] at hv hv'
  rw [keyedDrawn_val_cont] at hv'
  exact hne (hv.trans hv'.symm)

theorem keyedDrawn_sound (n : ℕ) : (keyedDrawn n).JointSound 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨i, _, hi⟩ := h
  exact hi rfl

theorem keyedDrawn_pins (n : ℕ) : (keyedDrawn n).Pins := by
  intro x w j v h
  exact h

/-- **The honest coin-keyed sampled layer meets the premises at the budget**, over `S = n + 1` rounds: binding at
`δ_bind = 0` and its fork form at every position's cut at `δ_fork = 0`, joint soundness at `δ_ks = 0` and pinning, with
`S·(δ_bind + δ_ks)` and `S·(|L|·√δ_fork + δ_ks)` (`|L| = 2`, a round's positions) both within `2^-41`. -/
theorem keyedDrawn_meets_budget (n : ℕ) :
    ∃ δbind δfork δks : ℝ, (keyedDrawn n).BindingAt δbind ∧
      (∀ ℓ, prob (fun p : Paths (Paths Bool 1 × Unit) (n + 1) × Paths (Paths Bool 1 × Unit) (n + 1) =>
        (keyedDrawn n).Forked ℓ p.1 p.2) ≤ δfork) ∧
      (keyedDrawn n).JointSound δks ∧ (keyedDrawn n).Pins ∧
      ((n + 1 : ℕ) : ℝ) * (δbind + δks) ≤ (1 / 2) ^ 41 ∧
      ((n + 1 : ℕ) : ℝ) * (2 * Real.sqrt δfork + δks) ≤ (1 / 2) ^ 41 := by
  refine ⟨0, 0, 0, keyedDrawn_binding n, keyedDrawn_fork n, keyedDrawn_sound n, keyedDrawn_pins n, ?_, ?_⟩
  · simp only [add_zero, mul_zero]
    positivity
  · simp only [Real.sqrt_zero, mul_zero, add_zero]
    positivity

end Accounting.SecurityProofs.TS.Keyed
