module

public import Proofs.Pouw.TS.Sampled
public import Proofs.Pouw.TS.ProofsSound
public import Proofs.ResourceAccounting.Lemmas.TS

/-! An honest coin-keyed proof layer meets (I8)'s premises. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.TS.Keyed

open Finset ResourceAccounting.TS Pouw.RelativeSecurity

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
relation, and on its continuation that resamples the last round's coin as `false` they fail the last round's, so a prover
deficient in the last round is accepted by the ideal game on its lucky coins only. -/
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

/-- **`guessLayer` lacks `Pins`**: on the all-`false` path, the operand `true` in every round satisfies every relation
(the guess `true` is `true` keyed by `false`), and it is not the honest operand. -/
theorem guessLayer_not_pins (n : ℕ) : ¬ (guessLayer n).Pins := by
  intro h
  have hr := h (Paths.const false (n + 1)) (fun _ _ => true) (fun w _ => by
    show true = key true (coinAt (Paths.const false (n + 1)) w)
    rw [coinAt_const]
    rfl)
  have h0 : true = false := hr 0 0
  cases h0

theorem coinAt_spliceAt {C B : Type} {m : ℕ} : ∀ {n : ℕ} (w : Fin n) (k : ℕ) (x z : Paths (Paths C m × B) n),
    coinAt (spliceAt (w : ℕ) k x z) w = (splice k (coinAt x w).1 (coinAt z w).1, (coinAt z w).2)
  | 0, w, _, _, _ => w.elim0
  | _ + 1, ⟨0, _⟩, _, _, _ => rfl
  | _ + 1, ⟨w + 1, h⟩, k, x, z => coinAt_spliceAt ⟨w, Nat.lt_of_succ_lt_succ h⟩ k x.2 z.2

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

/-- Round `w`'s salt on a path of rounds with one coin before the sample. -/
def salt {n : ℕ} (x : Paths (Paths Bool 1 × Unit) n) (w : Fin n) : Bool := (coinAt x w).1.1

/-- Round `w`'s honest values: its operand (`false`), and its receipt, the operand keyed by the salt. -/
def val {n : ℕ} (x : Paths (Paths Bool 1 × Unit) n) (w : Fin n) (k : Bool) : Bool :=
  if k then key false (salt x w) else false

theorem salt_spliceAt {n : ℕ} (w : Fin n) (x z : Paths (Paths Bool 1 × Unit) n) :
    salt (spliceAt (w : ℕ) 1 x z) w = salt x w :=
  congrArg (fun p => p.1.1) (coinAt_spliceAt w 1 x z)

/-- **An honest coin-keyed sampled layer** over `n + 1` rounds, each with one coin before its sample (the salt) and one
unit, whose slots are its operand (`false`, position `(w, false)`, cut before the salt) and its receipt (position
`(w, true)`, cut at the sample, after the salt): every proof accepted, opening the honest values; the unit's relation and
rightness, that the receipt is the operand keyed by the salt. -/
def keyedSampled (n : ℕ) : SampledLayer Bool Unit (n + 1) 1 1 Unit Bool (Fin (n + 1) × Bool) Bool where
  draw _ _ := ()
  pos w _ k := (w, k)
  coin _ _ := 0
  cut ℓ := (ℓ.1, if ℓ.2 then Fin.last 1 else 0)
  src ℓ := if ℓ.2 then some (ℓ.1, ()) else none
  accepted _ _ _ := True
  opened x w _ k := val x w k
  rel x w _ v := v true = key (v false) (salt x w)
  right x w _ v := v true = key (v false) (salt x w)

theorem keyedSampled_opens (n : ℕ) (y : Paths (Paths Bool 1 × Unit) (n + 1)) (w : Fin (n + 1)) (k v : Bool) :
    (keyedSampled n).Opens y (w, k) v ↔ v = val y w k := by
  constructor
  · rintro ⟨w', i, k', hp, -, hv⟩
    have hp' : (w', k') = (w, k) := hp
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hp'
    exact hv.symm
  · intro hv
    exact ⟨w, 0, k, rfl, trivial, hv.symm⟩

theorem keyedSampled_val_cont (n : ℕ) (x z : Paths (Paths Bool 1 × Unit) (n + 1)) (w : Fin (n + 1)) (k : Bool) :
    val ((keyedSampled n).cont (w, k) x z) w k = val x w k := by
  cases k
  · rfl
  · show key false (salt (spliceAt (w : ℕ) 1 x z) w) = key false (salt x w)
    rw [salt_spliceAt]

theorem keyedSampled_committed (n : ℕ) (x : Paths (Paths Bool 1 × Unit) (n + 1)) (w : Fin (n + 1)) (k : Bool) :
    (keyedSampled n).committed x (w, k) = val x w k := by
  have h := plural_eq ((keyedSampled n).cont (w, k)) (fun y v => (keyedSampled n).Opens y (w, k) v)
    (fun y => val y w k) (fun y v => keyedSampled_opens n y w k v) x x (fun z => keyedSampled_val_cont n x z w k)
  exact h

/-- **The receipt moves with the salt**: redrawing round `w`'s salt (from its coin `0` on) changes its receipt, so a
cut before the salt would fail binding for this honest layer. -/
theorem keyedSampled_receipt_moves (n : ℕ) (w : Fin (n + 1)) (x : Paths (Paths Bool 1 × Unit) (n + 1)) :
    ∃ z, key false (salt x w) ≠ key false (salt (spliceAt (w : ℕ) 0 x z) w) := by
  refine ⟨Paths.const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) (n + 1), ?_⟩
  have h : salt (spliceAt (w : ℕ) 0 x (Paths.const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) (n + 1))) w =
      !salt x w :=
    (congrArg (fun p => p.1.1)
      (coinAt_spliceAt w 0 x (Paths.const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) (n + 1)))).trans
      (congrArg (fun p => p.1.1) (coinAt_const (((!salt x w, ()), ()) : Paths Bool 1 × Unit) w))
  rw [h]
  exact key_flip _

theorem keyedSampled_binding (n : ℕ) : (keyedSampled n).BindingAt 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨i, _, hi⟩ := h
  exact hi (funext fun k => (keyedSampled_committed n x w k).symm)

theorem keyedSampled_fork (n : ℕ) (ℓ : Fin (n + 1) × Bool) :
    prob (fun p : Paths (Paths Bool 1 × Unit) (n + 1) × Paths (Paths Bool 1 × Unit) (n + 1) =>
      (keyedSampled n).Forked ℓ p.1 p.2) ≤ 0 := by
  obtain ⟨w, k⟩ := ℓ
  refine le_of_eq (prob_zero fun p h => ?_)
  obtain ⟨v, v', hv, hv', hne⟩ := h
  rw [keyedSampled_opens] at hv hv'
  rw [keyedSampled_val_cont] at hv'
  exact hne (hv.trans hv'.symm)

theorem keyedSampled_sound (n : ℕ) : (keyedSampled n).JointSound 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨i, _, hi⟩ := h
  exact hi rfl

theorem keyedSampled_pins (n : ℕ) : (keyedSampled n).Pins := by
  intro x w j v h
  exact h

theorem keyedSampled_meets_budget (n : ℕ) :
    ∃ δbind δfork δks : ℝ, (keyedSampled n).BindingAt δbind ∧
      (∀ ℓ, prob (fun p : Paths (Paths Bool 1 × Unit) (n + 1) × Paths (Paths Bool 1 × Unit) (n + 1) =>
        (keyedSampled n).Forked ℓ p.1 p.2) ≤ δfork) ∧
      (keyedSampled n).JointSound δks ∧ (keyedSampled n).Pins ∧
      ((n + 1 : ℕ) : ℝ) * (δbind + δks) ≤ (1 / 2) ^ 41 ∧
      ((n + 1 : ℕ) : ℝ) * (2 * Real.sqrt δfork + δks) ≤ (1 / 2) ^ 41 := by
  refine ⟨0, 0, 0, keyedSampled_binding n, keyedSampled_fork n, keyedSampled_sound n, keyedSampled_pins n, ?_, ?_⟩
  · simp only [add_zero, mul_zero]
    positivity
  · simp only [Real.sqrt_zero, mul_zero, add_zero]
    positivity

section Plays

open ResourceAccounting.Assumptions.TS

theorem key_false (c : Bool) : key false c = c := by cases c <;> rfl

/-- The receipt round `w` publishes on the path: the honest one (the operand `false` keyed by σ_w) in every round but
the last, where the prover skips the round's words and publishes the guess `true`. -/
def rcpt (n : ℕ) (x : Paths Bool (n + 1)) (w : Fin (n + 1)) : Bool :=
  if (w : ℕ) = n then true else key false (coinAt x w)

/-- **A pinning check against a last-round guesser** over `n + 1` rounds, one proof a round: round `w`'s relation
recomputes the operand (`v = false`) and reads the receipt the proof gives on the path, keyed by σ_w; a sound verifier
accepts the proof exactly when its opening satisfies the relation; a window right when every operand is the honest
one. -/
def lastLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted x w _ := rcpt n x w = key false (coinAt x w)
  opened _ _ _ := false
  rel x w _ v := v = false ∧ rcpt n x w = key v (coinAt x w)
  right _ c := ∀ w i, c w i = false

/-- The ideal game split at the receipt: every round but the last honest (share `0`, passing), the last deficient
(share `1`, its words wrong), passing and keeping the honest committed value exactly when its coin is `true`. -/
noncomputable def lastRG : (n : ℕ) → ReceiptGame Bool (n + 1)
  | 0 => .round (fun _ => 1) (fun ω => ω = true) (fun _ => False) (fun ω => ω = true) fun _ => .done
  | n + 1 => .round (fun _ => 0) (fun _ => True) (fun _ => True) (fun _ => True) fun _ => lastRG n

/-- The cheater's play: accepted when every proof is, its run's fixed words `(1 − 1/S)·N_w` at `N_w = 1`. -/
noncomputable def lastPlay (n : ℕ) : Play Bool (n + 1) Unit Unit Unit Unit where
  game := (lastRG n).game
  accepts x := ∀ w, (lastLayer n).Proved x w
  run _ :=
    { run := { n := fun _ => 0, cycles := 0, clock := 0 }, nword := fun _ _ => 0, nfold := fun _ => 0,
      nfill := fun _ => 0, words := fun _ => 0, fixed := 1 - 1 / ((n + 1 : ℕ) : ℝ) }

theorem lastRG_accepts : ∀ (n : ℕ) (x : Paths Bool (n + 1)),
    (lastRG n).game.accepts x ↔ coinAt x (Fin.last n) = true
  | 0, x => by
    show (x.1 = true ∧ True) ↔ x.1 = true
    exact ⟨fun h => h.1, fun h => ⟨h, trivial⟩⟩
  | n + 1, x => by
    show (True ∧ (lastRG n).game.accepts x.2) ↔ coinAt x.2 (Fin.last n) = true
    rw [true_and]
    exact lastRG_accepts n x.2

theorem lastRG_deficiency : ∀ (n : ℕ) (x : Paths Bool (n + 1)), (lastRG n).game.deficiency x = 1
  | 0, _ => by
    show (1 : ℝ) + 0 = 1
    norm_num
  | n + 1, x => by
    show (0 : ℝ) + (lastRG n).game.deficiency x.2 = 1
    rw [zero_add]
    exact lastRG_deficiency n x.2

theorem rcpt_last (n : ℕ) (x : Paths Bool (n + 1)) : rcpt n x (Fin.last n) = true := by
  unfold rcpt
  simp

theorem lastLayer_of (n : ℕ) : (lastLayer n).Of (lastPlay n) := by
  refine ⟨fun _ hx => hx, fun x hx => ?_⟩
  obtain ⟨_, h1⟩ := hx (Fin.last n) 0
  have h2 : rcpt n x (Fin.last n) = key false (coinAt x (Fin.last n)) := h1
  rw [rcpt_last, key_false] at h2
  exact (lastRG_accepts n x).2 h2.symm

theorem lastLayer_binding (n : ℕ) : (lastLayer n).BindingAt 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨_, i, hi⟩ := h
  exact hi rfl

theorem lastLayer_sound (n : ℕ) : (lastLayer n).SoundAt 0 := by
  intro w i
  refine le_of_eq (prob_zero fun x h => ?_)
  exact h.2 ⟨rfl, h.1⟩

theorem lastLayer_pins (n : ℕ) : (lastLayer n).Pins := by
  intro x c h w i
  exact (h w i).1

theorem lastPlay_rightFixed (n : ℕ) : RightFixed (lastPlay n) (lastLayer n) 1 := by
  intro x _ _
  show (1 - (lastRG n).game.deficiency x / ((n + 1 : ℕ) : ℝ)) * 1 ≤ 1 - 1 / ((n + 1 : ℕ) : ℝ)
  exact le_of_eq (by rw [lastRG_deficiency, mul_one])

theorem lastRG_pinned : ∀ n, (lastRG n).Pinned
  | 0 => ⟨fun _ h => h, fun _ => trivial⟩
  | n + 1 => ⟨fun _ _ => trivial, fun _ => lastRG_pinned n⟩

theorem prob_bool_true : prob (fun ω : Bool => ω = true) = 1 / 2 := by
  rw [prob_eq_card]
  have h1 : (Finset.univ.filter fun ω : Bool => ω = true).card = 1 := by decide
  rw [h1, Fintype.card_bool]
  norm_num

theorem prob_bool_all : prob (fun _ : Bool => True) = 1 := by
  rw [prob_eq_card, Finset.filter_true_of_mem fun _ _ => trivial, Finset.card_univ, Fintype.card_bool]
  norm_num

theorem lastRG_receiptLuck : ∀ n, (lastRG n).ReceiptLuck (1 / 2)
  | 0 => by
    refine ⟨?_, fun _ => trivial⟩
    have e1 : (fun ω : Bool => ω = true ∧ ¬ False ∧ (0 : ℝ) < 1) = fun ω => ω = true :=
      funext fun ω => propext ⟨fun h => h.1, fun h => ⟨h, not_false, one_pos⟩⟩
    have e2 : (fun _ : Bool => ¬ False ∧ (0 : ℝ) < 1) = fun _ => True :=
      funext fun _ => propext ⟨fun _ => trivial, fun _ => ⟨not_false, one_pos⟩⟩
    show prob (fun ω : Bool => ω = true ∧ ¬ False ∧ (0 : ℝ) < 1) ≤
      1 / 2 * prob (fun _ : Bool => ¬ False ∧ (0 : ℝ) < 1)
    rw [e1, e2, prob_bool_true, prob_bool_all]
    norm_num
  | n + 1 => by
    refine ⟨?_, fun _ => lastRG_receiptLuck n⟩
    show prob (fun _ : Bool => True ∧ ¬ True ∧ (0 : ℝ) < 0) ≤
      1 / 2 * prob (fun _ : Bool => ¬ True ∧ (0 : ℝ) < 0)
    have h1 : prob (fun _ : Bool => True ∧ ¬ True ∧ (0 : ℝ) < 0) = 0 := prob_zero fun _ h => h.2.1 trivial
    have h2 : prob (fun _ : Bool => ¬ True ∧ (0 : ℝ) < 0) = 0 := prob_zero fun _ h => h.1 trivial
    rw [h1, h2]
    norm_num

theorem lastPlay_proofsSound (n : ℕ) : ProofsSound (lastPlay n) 1 0 0 1 :=
  _root_.ResourceAccounting.SecurityProofs.TS.ProofsSoundOfPinned Bool (n + 1) 1 (lastPlay n) (lastLayer n) 1 0 0
    (lastLayer_of n) (lastLayer_binding n) (lastLayer_sound n) (lastLayer_pins n) (lastPlay_rightFixed n)

/-- **It is a cheater the statement is about**: the all-`true` path is accepted, the ideal game charges it one
deficient round, and its fixed words are `(1 − 1/S)·N_w`, below `N_w` and equal to `ProofsSound`'s bound there. -/
theorem lastPlay_tight (n : ℕ) :
    (lastPlay n).accepts (Paths.const true (n + 1)) ∧
      (lastPlay n).game.deficiency (Paths.const true (n + 1)) = 1 ∧
      ((lastPlay n).run (Paths.const true (n + 1))).fixed < 1 ∧
      (1 - (lastPlay n).game.deficiency (Paths.const true (n + 1)) / ((n + 1 : ℕ) : ℝ)) * 1 =
        ((lastPlay n).run (Paths.const true (n + 1))).fixed := by
  refine ⟨fun w i => ?_, lastRG_deficiency n _, ?_, ?_⟩
  · show rcpt n (Paths.const true (n + 1)) w = key false (coinAt (Paths.const true (n + 1)) w)
    unfold rcpt
    rw [coinAt_const]
    split <;> rfl
  · show 1 - 1 / ((n + 1 : ℕ) : ℝ) < 1
    have : (0 : ℝ) < 1 / ((n + 1 : ℕ) : ℝ) := by positivity
    linarith
  · show (1 - (lastRG n).game.deficiency (Paths.const true (n + 1)) / ((n + 1 : ℕ) : ℝ)) * 1 =
      1 - 1 / ((n + 1 : ℕ) : ℝ)
    rw [lastRG_deficiency, mul_one]

theorem lastPlay_rejects_unlucky (n : ℕ) :
    ¬ (lastPlay n).accepts (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) := by
  intro h
  have h2 : rcpt n (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) (Fin.last n) =
      key false (coinAt (splice ((Fin.last n : Fin (n + 1)) : ℕ) (Paths.const true (n + 1))
        (Paths.const false (n + 1))) (Fin.last n)) := h (Fin.last n) 0
  rw [rcpt_last, coinAt_splice, coinAt_const, key_false] at h2
  cases h2

/-- The honest ideal game: every round's share `0`, every round passing. -/
noncomputable def honestRG : (n : ℕ) → ReceiptGame Bool n
  | 0 => .done
  | n + 1 => .round (fun _ => 0) (fun _ => True) (fun _ => True) (fun _ => True) fun _ => honestRG n

/-- `keyedLayer`'s honest play: accepted when every proof is, its run's fixed words `N_w = 1`. -/
noncomputable def keyedPlay (n : ℕ) : Play Bool (n + 1) Unit Unit Unit Unit where
  game := (honestRG (n + 1)).game
  accepts x := ∀ w, (keyedLayer n).Proved x w
  run _ :=
    { run := { n := fun _ => 0, cycles := 0, clock := 0 }, nword := fun _ _ => 0, nfold := fun _ => 0,
      nfill := fun _ => 0, words := fun _ => 0, fixed := 1 }

theorem honestRG_accepts : ∀ (n : ℕ) (x : Paths Bool n), (honestRG n).game.accepts x
  | 0, _ => trivial
  | n + 1, x => ⟨trivial, honestRG_accepts n x.2⟩

theorem honestRG_deficiency : ∀ (n : ℕ) (x : Paths Bool n), (honestRG n).game.deficiency x = 0
  | 0, _ => rfl
  | n + 1, x => by
    show (0 : ℝ) + (honestRG n).game.deficiency x.2 = 0
    rw [zero_add]
    exact honestRG_deficiency n x.2

theorem keyedPlay_proofsSound (n : ℕ) : ProofsSound (keyedPlay n) 1 0 0 1 :=
  _root_.ResourceAccounting.SecurityProofs.TS.ProofsSoundOfPinned Bool (n + 1) 1 (keyedPlay n) (keyedLayer n) 1 0 0
    ⟨fun _ hx => hx, fun x _ => honestRG_accepts (n + 1) x⟩ (keyedLayer_binding n) (keyedLayer_sound n)
    (keyedLayer_pins n) (fun x _ _ => by
      show (1 - (honestRG (n + 1)).game.deficiency x / ((n + 1 : ℕ) : ℝ)) * 1 ≤ 1
      exact le_of_eq (by rw [honestRG_deficiency, zero_div, sub_zero, mul_one]))

end Plays

end ResourceAccounting.SecurityProofs.TS.Keyed
