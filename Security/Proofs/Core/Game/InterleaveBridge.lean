import Proofs.Core.Game.TwoRun

/-!
# A play advanced between stops, on coins supplied from outside

A strategy `σ` of a causal game `g` (`Verity.Game`) is turned into a strategy of another game whose moves interleave with
coins, a C-Flock game (`FlockSoundness.Game`) among them, by simulating `σ`: the simulation holds a tape of `g` and a
position on it, and walks `σ`'s play to the next node where one of a family of cuts (`DigestCut`) fixes a message (a
stop). There the other game's move is read off the most likely message at each cut fixed there (`revealMode`), and its
coin is written into the tape at the stop before the walk goes on. This module is that simulation on `g`'s side, at
any family of cuts and any rewriting of the tape:

* **Positions** (`Pos`): a node of `g` with `σ`'s strategy there, the cuts there, the tape from there and the tapes of
  the whole play through the node (`Pos.back`, a tape from there put back in place).
* **The walk** (`Pos.toStop`): along the tape, through the prover's messages and the tape's coins, to the first node at
  which a cut is fixed, or to the play's end (`Pos.toStop_stops`). `Pos.consume` marks the cuts fixed at a stop as
  past, and `Pos.rewrite` rewrites the tape from the position.
* **What the walk keeps** (`Pos.Inv`, `Pos.Le`): the play of a tape through the position is the play from the position
  (`Pos.Inv.run`); each cut not yet past reads, on such a tape, what its descent at the position reads, at every
  reading that descends through the prover's messages and the tape's coins (`CutReading`: the miss, the revealed and
  the most likely message); and the walk, consuming and rewriting only extend the position's tapes (`Pos.Le`), so
  every later tape is one through it, and the walk keeps the whole tape (`Pos.toStop_spec`).
* **What a stop fixes** (`Pos.modes`, `Pos.read_here`): at a stop, each cut fixed there reads, on every tape through
  any later position, what it reads at the stop.
* **The coupling through a joint law** (`pr_le_of_joint`): a joint law of another game's outcome and a tape of `g`, whose
  tape marginal is within `h` of the uniform tape's, and on whose support an event of the tape implies one of the
  outcome, bounds the event's probability over the uniform tape by the outcome's plus `h`.
-/

namespace Verity.Game

open scoped ENNReal

variable {α V ι : Type}

namespace DigestCut

/-- **The cut after the prover sends `m`**; a cut fixed at the node is past. -/
def atSend {M : Type} {k : M → Game α} : DigestCut V (.send M k) → (m : M) → DigestCut V (k m)
  | .skip, _ => .skip
  | .here _, _ => .skip
  | .send K, m => K m

/-- **The cut after the coin `c`**; a cut fixed at the node is past. -/
def atCoin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} :
    DigestCut V (@Game.coin _ C iF iN k) → (c : C) → DigestCut V (k c)
  | .skip, _ => .skip
  | .here _, _ => .skip
  | .coin K, c => K c

/-- Whether the cut fixes its message at this node. -/
def isHere {g : Game α} : DigestCut V g → Bool
  | .here _ => true
  | _ => false

/-- **The most likely message the cut fixes at this node**; `none` where it fixes none here. -/
noncomputable def modeHere [Nonempty V] {g : Game α} : DigestCut V g → Strategy g → Option V
  | .here rv, s => some (revealMode rv g s)
  | _, _ => none

theorem modeHere_here [Nonempty V] {g : Game α} {K : DigestCut V g} {s : Strategy g} {w : V}
    (h : K.modeHere s = some w) : ∃ rv, K = .here rv ∧ revealMode rv g s = w := by
  cases K with
  | skip => simp [modeHere] at h
  | here rv => exact ⟨rv, rfl, Option.some.inj h⟩
  | send K => simp [modeHere] at h
  | coin K => simp [modeHere] at h

end DigestCut

/-- **A reading of cuts on tapes**: a value of a cut, a strategy and a tape that reads through the prover's messages
and the tape's coins, as the miss (`DigestCut.missOn`), the revealed message (`revealOn`) and the most likely one
(`modeOn`) do. -/
structure CutReading (α V β : Type) : Type 1 where
  r : {g : Game α} → DigestCut V g → (s : Strategy g) → CoinTape g s → β
  send : ∀ {M : Type} {k : M → Game α} (K : (m : M) → DigestCut V (k m)) (s : Strategy (Game.send M k))
    (y : CoinTape (Game.send M k) s), r (.send K) s y = r (K s.1) s.2 y
  coin : ∀ {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} (K : (c : C) → DigestCut V (k c))
    (s : Strategy (@Game.coin _ C iF iN k)) (y : CoinTape (@Game.coin _ C iF iN k) s),
    r (@DigestCut.coin _ _ C iF iN k K) s y = r (K y.1) (s y.1) (y.2 y.1)

namespace CutReading

variable {β : Type} (ρ : CutReading α V β)

theorem atSend_eq {M : Type} {k : M → Game α} (K : DigestCut V (.send M k)) (s : Strategy (Game.send M k))
    (y : CoinTape (k s.1) s.2) (h : K.isHere = false) (hs : K ≠ .skip) :
    ρ.r K s y = ρ.r (K.atSend s.1) s.2 y := by
  cases K with
  | skip => exact absurd rfl hs
  | here rv => simp [DigestCut.isHere] at h
  | send K => exact ρ.send K s y

open Classical in
theorem atCoin_eq {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (K : DigestCut V (@Game.coin _ C iF iN k)) (s : Strategy (@Game.coin _ C iF iN k))
    (x : CoinTape (@Game.coin _ C iF iN k) s) (y : CoinTape (k x.1) (s x.1)) (h : K.isHere = false)
    (hs : K ≠ .skip) :
    ρ.r K s ((x.1, Function.update x.2 x.1 y) : CoinTape (@Game.coin _ C iF iN k) s) =
      ρ.r (K.atCoin x.1) (s x.1) y := by
  cases K with
  | skip => exact absurd rfl hs
  | here rv => simp [DigestCut.isHere] at h
  | coin K =>
    refine (ρ.coin K s _).trans ?_
    show ρ.r (K x.1) (s x.1) (Function.update x.2 x.1 y x.1) = ρ.r (K x.1) (s x.1) y
    rw [Function.update_self]

end CutReading

/-- **A position of a play of `σ` in `g`**: a node `g'` with `σ`'s strategy there (`s'`), the cuts there (`Ks`), the tape
from there (`x`), and the tapes of the whole play through the node (`back`: a tape from the node, put back in place). -/
structure Pos (g : Game α) (σ : Strategy g) (V ι : Type) : Type 1 where
  g' : Game α
  s' : Strategy g'
  Ks : ι → DigestCut V g'
  x : CoinTape g' s'
  back : CoinTape g' s' → CoinTape g σ

namespace Pos

variable {g : Game α} {σ : Strategy g}

/-- The play's start on the tape `x`, at the cuts `Ks`. -/
def root (Ks : ι → DigestCut V g) (x : CoinTape g σ) : Pos g σ V ι := ⟨g, σ, Ks, x, id⟩

/-- The tape of the whole play the position holds. -/
def tape (p : Pos g σ V ι) : CoinTape g σ := p.back p.x

open Classical in
/-- **The walk to the next stop**: from the node `g'` along the tape, through the prover's messages and the tape's
coins, to the first node at which one of the cuts is fixed, or to the play's end. -/
noncomputable def walk : (g' : Game α) → (s' : Strategy g') → (ι → DigestCut V g') → CoinTape g' s' →
    (CoinTape g' s' → CoinTape g σ) → Pos g σ V ι
  | .ret a, s', Ks, x, back => ⟨.ret a, s', Ks, x, back⟩
  | .send M k, s', Ks, x, back =>
    if ∃ i, (Ks i).isHere then ⟨.send M k, s', Ks, x, back⟩
    else walk (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) x back
  | @Game.coin _ C iF iN k, s', Ks, x, back =>
    if ∃ i, (Ks i).isHere then ⟨@Game.coin _ C iF iN k, s', Ks, x, back⟩
    else walk (k x.1) (s' x.1) (fun i => (Ks i).atCoin x.1) (x.2 x.1)
      fun y => back (x.1, Function.update x.2 x.1 y)

/-- **The next stop from the position** (`walk`). -/
noncomputable def toStop (p : Pos g σ V ι) : Pos g σ V ι := walk p.g' p.s' p.Ks p.x p.back

/-- **The cuts fixed at the position, marked past.** -/
def consume (p : Pos g σ V ι) : Pos g σ V ι where
  __ := p
  Ks i := if (p.Ks i).isHere then .skip else p.Ks i

/-- **The tape from the position, rewritten by `f`.** -/
def rewrite (p : Pos g σ V ι) (f : CoinTape p.g' p.s' → CoinTape p.g' p.s') : Pos g σ V ι where
  __ := p
  x := f p.x

/-- **The most likely message each cut fixed at the position fixes there**; `none` at every other cut. -/
noncomputable def modes [Nonempty V] (p : Pos g σ V ι) (i : ι) : Option V := (p.Ks i).modeHere p.s'

/-- **What a position keeps of the play's start at the cuts `Ks₀`**: the play of a tape through the node is the play
from the node, and each cut either reads, on every tape through the node and at every reading, what its descent at the
node reads, or is past. -/
structure Inv (Ks₀ : ι → DigestCut V g) (p : Pos g σ V ι) : Prop where
  run : ∀ y, CoinTape.run g σ (p.back y) = CoinTape.run p.g' p.s' y
  cut : ∀ i, (∀ (β : Type) (ρ : CutReading α V β) y, ρ.r (Ks₀ i) σ (p.back y) = ρ.r (p.Ks i) p.s' y) ∨
    p.Ks i = .skip

/-- **`q` is a later position than `p`**: every tape through `q` is a tape through `p`. -/
def Le (p q : Pos g σ V ι) : Prop :=
  ∃ h : CoinTape q.g' q.s' → CoinTape p.g' p.s', ∀ y, q.back y = p.back (h y)

theorem Le.refl (p : Pos g σ V ι) : p.Le p := ⟨id, fun _ => rfl⟩

theorem Le.trans {p q r : Pos g σ V ι} (h₁ : p.Le q) (h₂ : q.Le r) : p.Le r := by
  obtain ⟨f, hf⟩ := h₁
  obtain ⟨f', hf'⟩ := h₂
  exact ⟨f ∘ f', fun y => (hf' y).trans (hf (f' y))⟩

theorem inv_root (Ks : ι → DigestCut V g) (x : CoinTape g σ) : (root Ks x).Inv Ks :=
  ⟨fun _ => rfl, fun _ => Or.inl fun _ _ _ => rfl⟩

theorem le_consume (p : Pos g σ V ι) : p.Le p.consume := ⟨id, fun _ => rfl⟩

theorem le_rewrite (p : Pos g σ V ι) (f : CoinTape p.g' p.s' → CoinTape p.g' p.s') : p.Le (p.rewrite f) :=
  ⟨id, fun _ => rfl⟩

theorem inv_rewrite {Ks₀ : ι → DigestCut V g} {p : Pos g σ V ι} (h : p.Inv Ks₀)
    (f : CoinTape p.g' p.s' → CoinTape p.g' p.s') : (p.rewrite f).Inv Ks₀ := ⟨h.run, h.cut⟩

theorem inv_consume {Ks₀ : ι → DigestCut V g} {p : Pos g σ V ι} (h : p.Inv Ks₀) : p.consume.Inv Ks₀ := by
  refine ⟨h.run, fun i => ?_⟩
  show (∀ (β : Type) (ρ : CutReading α V β) y, ρ.r (Ks₀ i) σ (p.back y) =
      ρ.r (if (p.Ks i).isHere then .skip else p.Ks i) p.s' y) ∨
    (if (p.Ks i).isHere then .skip else p.Ks i) = .skip
  split_ifs
  · exact Or.inr rfl
  · exact h.cut i

theorem tape_consume (p : Pos g σ V ι) : p.consume.tape = p.tape := rfl

theorem tape_rewrite (p : Pos g σ V ι) (f : CoinTape p.g' p.s' → CoinTape p.g' p.s') :
    (p.rewrite f).tape = p.back (f p.x) := rfl

/-- **The walk extends the position, keeps the tape and keeps the invariant.** -/
theorem walk_spec (Ks₀ : ι → DigestCut V g) :
    ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
      (back : CoinTape g' s' → CoinTape g σ),
      (⟨g', s', Ks, x, back⟩ : Pos g σ V ι).Inv Ks₀ →
        (⟨g', s', Ks, x, back⟩ : Pos g σ V ι).Le (walk g' s' Ks x back) ∧
          (walk g' s' Ks x back).tape = back x ∧ (walk g' s' Ks x back).Inv Ks₀
  | .ret a, s', Ks, x, back, h => ⟨Le.refl _, rfl, h⟩
  | .send M k, s', Ks, x, back, h => by
    classical
    unfold walk
    split_ifs with hf
    · exact ⟨Le.refl _, rfl, h⟩
    · have hf' : ∀ i, (Ks i).isHere = false := fun i => by
        simpa using fun h' => hf ⟨i, h'⟩
      have h' : (⟨k s'.1, s'.2, fun i => (Ks i).atSend s'.1, x, back⟩ : Pos g σ V ι).Inv Ks₀ := by
        refine ⟨h.run, fun i => ?_⟩
        by_cases hs : Ks i = .skip
        · refine Or.inr ?_
          show (Ks i).atSend s'.1 = .skip
          rw [hs]
          rfl
        rcases h.cut i with hc | hc
        · exact Or.inl fun β ρ y => (hc β ρ y).trans (ρ.atSend_eq (Ks i) s' y (hf' i) hs)
        · exact absurd hc hs
      exact walk_spec Ks₀ (k s'.1) s'.2 _ x back h'
  | @Game.coin _ C iF iN k, s', Ks, x, back, h => by
    classical
    unfold walk
    split_ifs with hf
    · exact ⟨Le.refl _, rfl, h⟩
    · have hf' : ∀ i, (Ks i).isHere = false := fun i => by
        simpa using fun h' => hf ⟨i, h'⟩
      have h' : (⟨k x.1, s' x.1, fun i => (Ks i).atCoin x.1, x.2 x.1,
          fun y => back (x.1, Function.update x.2 x.1 y)⟩ : Pos g σ V ι).Inv Ks₀ := by
        refine ⟨fun y => (h.run _).trans ?_, fun i => ?_⟩
        · show CoinTape.run (k x.1) (s' x.1) (Function.update x.2 x.1 y x.1) = _
          rw [Function.update_self]
        by_cases hs : Ks i = .skip
        · refine Or.inr ?_
          show (Ks i).atCoin x.1 = .skip
          rw [hs]
          rfl
        rcases h.cut i with hc | hc
        · exact Or.inl fun β ρ y => (hc β ρ _).trans (ρ.atCoin_eq (Ks i) s' x y (hf' i) hs)
        · exact absurd hc hs
      obtain ⟨⟨f, hf'⟩, ht, hi⟩ := walk_spec Ks₀ (k x.1) (s' x.1) _ (x.2 x.1) _ h'
      refine ⟨⟨fun y => (x.1, Function.update x.2 x.1 (f y)), hf'⟩, ?_, hi⟩
      rw [ht]
      show back (x.1, Function.update x.2 x.1 (x.2 x.1)) = back x
      rw [Function.update_eq_self]
      rfl

/-- **The next stop extends the position, keeps the tape and keeps the invariant.** -/
theorem toStop_spec {Ks₀ : ι → DigestCut V g} {p : Pos g σ V ι} (h : p.Inv Ks₀) :
    p.Le p.toStop ∧ p.toStop.tape = p.tape ∧ p.toStop.Inv Ks₀ :=
  walk_spec Ks₀ p.g' p.s' p.Ks p.x p.back h

/-- **The walk ends at a stop or at the play's end.** -/
theorem walk_stops : ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
    (back : CoinTape g' s' → CoinTape g σ),
    (∃ i, ((walk g' s' Ks x back).Ks i).isHere) ∨ ∃ a, (walk g' s' Ks x back).g' = .ret a
  | .ret a, _, _, _, _ => Or.inr ⟨a, rfl⟩
  | .send M k, s', Ks, x, back => by
    by_cases hf : ∃ i, (Ks i).isHere
    · rw [walk, ite_eq_left_of_eq_true _ _ (eq_true hf)]
      exact Or.inl hf
    · rw [walk, ite_eq_right_of_eq_false _ _ (eq_false hf)]
      exact walk_stops (k s'.1) s'.2 _ x back
  | @Game.coin _ C iF iN k, s', Ks, x, back => by
    by_cases hf : ∃ i, (Ks i).isHere
    · rw [walk, ite_eq_left_of_eq_true _ _ (eq_true hf)]
      exact Or.inl hf
    · rw [walk, ite_eq_right_of_eq_false _ _ (eq_false hf)]
      exact walk_stops (k x.1) (s' x.1) _ (x.2 x.1) _

theorem toStop_stops (p : Pos g σ V ι) : (∃ i, (p.toStop.Ks i).isHere) ∨ ∃ a, p.toStop.g' = .ret a :=
  walk_stops p.g' p.s' p.Ks p.x p.back

/-- **What a stop fixes, on every later tape**: at a position that keeps the invariant, a cut fixed there reads, on
every tape through any later position and at every reading, what it reads at the position on some tape from there. -/
theorem read_here {Ks₀ : ι → DigestCut V g} {p q : Pos g σ V ι} (h : p.Inv Ks₀) (hpq : p.Le q) {i : ι}
    {rv : α → Option V} (hi : p.Ks i = .here rv) {β : Type} (ρ : CutReading α V β) (y : CoinTape q.g' q.s') :
    ∃ z, ρ.r (Ks₀ i) σ (q.back y) = ρ.r (DigestCut.here rv : DigestCut V p.g') p.s' z := by
  obtain ⟨f, hf⟩ := hpq
  refine ⟨f y, ?_⟩
  rw [hf y]
  rcases h.cut i with hc | hc
  · rw [hc β ρ (f y), hi]
  · rw [hi] at hc
    cases hc

/-- **The play of the position's tape is the play of the whole tape.** -/
theorem run_tape {Ks₀ : ι → DigestCut V g} {p : Pos g σ V ι} (h : p.Inv Ks₀) :
    CoinTape.run g σ p.tape = CoinTape.run p.g' p.s' p.x :=
  h.run p.x

end Pos

/-! ## The coupling through a joint law -/

theorem prob_mono_reaches {p q : α → Prop} :
    ∀ (g : Game α) (s : Strategy g), (∀ a, PlayReaches g s a → p a → q a) → prob p g s ≤ prob q g s
  | .ret a, _, h => by
      classical
      simp only [prob]
      by_cases hp : p a
      · simp [hp, h a rfl hp]
      · simp [hp]
  | .send _ k, s, h => prob_mono_reaches (k s.1) s.2 h
  | @Game.coin _ C _ _ k, s, h => Finset.sum_le_sum fun c _ =>
      mul_le_mul_right (prob_mono_reaches (k c) (s c) fun a ha => h a ⟨c, ha⟩) _

/-- **The coupling through a joint law**: a joint law `W` of an outcome and a tape of `g` at `σ` whose tape is, at the
event `Q`, within `h` of the uniform tape (`hW`), and on whose support `Q` of the tape implies `E` of the outcome (`hE`),
bounds `Q`'s probability over the uniform tape by `E`'s under `W`, plus `h`. -/
theorem pr_le_of_joint {g : Game α} {σ : Strategy g} {β : Type} (W : Game (β × CoinTape g σ)) (w : Strategy W)
    (Q : CoinTape g σ → Prop) (E : β → Prop) (h : ℝ≥0∞)
    (hW : ENNReal.ofReal (Claims.Pr Q) ≤ prob (fun o => Q o.2) W w + h)
    (hE : ∀ o, PlayReaches W w o → Q o.2 → E o.1) :
    ENNReal.ofReal (Claims.Pr Q) ≤ prob (fun o => E o.1) W w + h :=
  hW.trans (add_le_add (prob_mono_reaches W w hE) le_rfl)

end Verity.Game
