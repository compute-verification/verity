module

public import Definitions.Core.Game.Seq
public import Definitions.Core.Interaction

/-!
# The most likely revealed message, a play's cuts, and the two-continuation finder

A play may fix a message of the prover's by a digest of it and reveal the message only later. A reading of the play that
needs the message where the digest fixes it takes the most likely one there (`revealMode`): of the messages the plays
from that node reveal, one with the largest probability.

* `PlayReaches`, `revealPr`, `revealMode`: the outputs a strategy's plays reach, the probability that a play reveals a
  message, and a message of the largest probability. One exists (`exists_revealMode`): only the finitely many messages
  the reached outputs reveal have a probability (`playReaches_finite`, `expect_eq_zero`), and none is negative
  (`revealPr_nonneg`).
* `DigestCut`: where along each play the message is fixed (`here`), and what the play's output reveals for it.
  `DigestCut.Agrees`: every two messages a cut's plays reveal have one image under a hash.
* `DigestCut.forkG`, `DigestCut.forkS`: the play up to its cut, then two independent continuations, with the messages
  they reveal (`DigestCut.pair`).
* `CoinTape`: a uniform tape for a strategy's plays, a finite and inhabited type (`CoinTape.fintype`,
  `CoinTape.nonempty`), and the play a tape reads (`CoinTape.run`).
* `DigestCut.finder`: a fork's tape as the two-continuation finder, a `Claims.Finder` at a SHA-512 budget on every
  coin.

The lemmas here are the ones a definition's proof field needs. What the definitions satisfy (the miss at a cut, and its
bound by the finder's success) is proved in `Proofs.Core.Game.TwoRun`.
-/

@[expose] public section

namespace Verity

namespace Game

variable {α : Type}

/-! ## The outputs a strategy reaches, and the most likely revealed message -/

/-- **An output the plays of `s` in `g` reach**, at some coins. -/
def PlayReaches : (g : Game α) → Strategy g → α → Prop
  | .ret a, _, o => o = a
  | .send _ k, s, o => PlayReaches (k s.1) s.2 o
  | @Game.coin _ C _ _ k, s, o => ∃ c : C, PlayReaches (k c) (s c) o

theorem playReaches_finite : ∀ (g : Game α) (s : Strategy g), {o | PlayReaches g s o}.Finite
  | .ret a, _ => Set.finite_singleton a
  | .send _ k, s => playReaches_finite (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      have h : {o | PlayReaches (@Game.coin _ C _ _ k) s o} = ⋃ c, {o | PlayReaches (k c) (s c) o} := by
        ext o
        simp [PlayReaches]
      rw [h]
      exact Set.finite_iUnion fun c => playReaches_finite (k c) (s c)

theorem expect_eq_zero {f : α → ℝ} :
    ∀ (g : Game α) (s : Strategy g), (∀ o, PlayReaches g s o → f o = 0) → expect f g s = 0
  | .ret a, _, h => h a rfl
  | .send _ k, s, h => expect_eq_zero (k s.1) s.2 h
  | @Game.coin _ C _ _ k, s, h => by
      simp only [expect]
      rw [Finset.sum_eq_zero fun c _ => expect_eq_zero (k c) (s c) fun o ho => h o ⟨c, ho⟩, zero_div]

variable {V : Type}

/-- **The probability that the play reveals `w`**, as `rv` reads the play's output. -/
noncomputable def revealPr (rv : α → Option V) (g : Game α) (s : Strategy g) (w : V) : ℝ :=
  expect (ind fun o => rv o = some w) g s

theorem revealPr_nonneg (rv : α → Option V) (g : Game α) (s : Strategy g) (w : V) : 0 ≤ revealPr rv g s w := by
  have hind : ∀ (p : α → Prop) (a : α), 0 ≤ ind p a := fun p a => by
    unfold ind; split_ifs <;> norm_num
  unfold revealPr
  induction g with
  | ret a => exact hind _ a
  | send M k ih => exact ih s.1 s.2
  | coin C k ih =>
    simp only [expect]
    exact div_nonneg (Finset.sum_nonneg fun c _ => ih c (s c)) (Nat.cast_nonneg _)

/-- Only the finitely many messages the reached outputs reveal have a probability, so one has the largest. -/
theorem exists_revealMode [Nonempty V] (rv : α → Option V) (g : Game α) (s : Strategy g) :
    ∃ c, ∀ w, revealPr rv g s w ≤ revealPr rv g s c := by
  classical
  set T : Set V := {w | ∃ o, PlayReaches g s o ∧ rv o = some w}
  have hT : T.Finite := (((playReaches_finite g s).image rv).preimage (Option.some_injective V).injOn).subset
    fun w ⟨o, ho, hw⟩ => ⟨o, ho, hw⟩
  have h0 : ∀ w, w ∉ T → revealPr rv g s w = 0 := fun w hw =>
    expect_eq_zero g s fun o ho => by
      unfold ind
      rw [ite_eq_right fun he => hw ⟨o, ho, he⟩]
  obtain ⟨c₀⟩ := ‹Nonempty V›
  obtain ⟨c, -, hc⟩ :=
    (insert c₀ hT.toFinset).exists_max_image (revealPr rv g s) ⟨c₀, Finset.mem_insert_self _ _⟩
  refine ⟨c, fun w => ?_⟩
  by_cases hw : w ∈ T
  · exact hc w (Finset.mem_insert_of_mem (hT.mem_toFinset.2 hw))
  · rw [h0 w hw]
    exact revealPr_nonneg rv g s c

/-- **The most likely revealed message**: a message the plays of `s` in `g` reveal, as `rv` reads their outputs, with
the largest probability. -/
noncomputable def revealMode [Nonempty V] (rv : α → Option V) (g : Game α) (s : Strategy g) : V :=
  (exists_revealMode rv g s).choose

/-! ## Cuts: where a play fixes a message, and the fork from there -/

/-- **A cut of the game `g`**: where along each play the prover's message is fixed, by a digest of it, before the play
reveals it. At `here rv` it is fixed, and `rv` reads from the play's output the message the play reveals for it (`none`
where it reveals none); a play through `skip` has no cut. -/
inductive DigestCut (V : Type) : Game α → Type 1
  | skip {g : Game α} : DigestCut V g
  | here {g : Game α} (rv : α → Option V) : DigestCut V g
  | send {M : Type} {k : M → Game α} (K : (m : M) → DigestCut V (k m)) : DigestCut V (.send M k)
  | coin {C : Type} [Fintype C] [Nonempty C] {k : C → Game α} (K : (c : C) → DigestCut V (k c)) :
      DigestCut V (.coin C k)

namespace DigestCut

/-- The two messages two continuations reveal, where both reveal one. -/
def pair : Option V → Option V → Option (V × V)
  | some w, some w' => some (w, w')
  | _, _ => none

/-- **The fork's game**: the play up to its cut, then two independent continuations from there, with the two messages
they reveal; `none` where the play has no cut. -/
def forkG : {g : Game α} → DigestCut V g → Strategy g → Game (Option (V × V))
  | _, .skip, _ => .ret none
  | g, .here rv, _ => g.bind fun o => g.map fun o' => pair (rv o) (rv o')
  | _, .send K, s => forkG (K s.1) s.2
  | _, @DigestCut.coin _ _ C iF iN _ K, s => @Game.coin _ C iF iN fun c => forkG (K c) (s c)

/-- The fork's strategy: `s`, before the cut and in both continuations. -/
def forkS : {g : Game α} → (K : DigestCut V g) → (s : Strategy g) → Strategy (K.forkG s)
  | _, .skip, _ => PUnit.unit
  | g, .here _, s => Strategy.bind g _ s fun _ => Strategy.map _ g s
  | _, .send K, s => forkS (K s.1) s.2
  | _, @DigestCut.coin _ _ C _ _ _ K, s => fun c : C => forkS (K c) (s c)

/-- **Every two messages a cut's plays reveal have one image under `H`**: each opens the digest the cut fixed. -/
def Agrees {D : Type} (H : V → D) : {g : Game α} → DigestCut V g → Prop
  | _, .skip => True
  | _, .here rv => ∀ o o' w w', rv o = some w → rv o' = some w' → H w = H w'
  | _, .send K => ∀ m, Agrees H (K m)
  | _, @DigestCut.coin _ _ C _ _ _ K => ∀ c : C, Agrees H (K c)

end DigestCut

/-! ## A uniform tape for a strategy's plays -/

/-- **A uniform tape for the plays of `s` in `g`**: at each coin, the coin and a tape for each of its continuations. A
uniform tape gives the play its sampling design (`expect_tape`). -/
def CoinTape : (g : Game α) → Strategy g → Type
  | .ret _, _ => PUnit
  | .send _ k, s => CoinTape (k s.1) s.2
  | @Game.coin _ C _ _ k, s => C × ((c : C) → CoinTape (k c) (s c))

/-- The output of the play a tape reads: at each coin, the tape's coin, then that continuation's tape. -/
def CoinTape.run : (g : Game α) → (s : Strategy g) → CoinTape g s → α
  | .ret a, _, _ => a
  | .send _ k, s, x => CoinTape.run (k s.1) s.2 x
  | @Game.coin _ _ _ _ k, s, x => CoinTape.run (k x.1) (s x.1) (x.2 x.1)

open Classical in
/-- A tape's finitely many values. -/
@[instance_reducible]
noncomputable def CoinTape.fintype : (g : Game α) → (s : Strategy g) → Fintype (CoinTape g s)
  | .ret _, _ => inferInstanceAs (Fintype PUnit)
  | .send _ k, s => CoinTape.fintype (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      letI := fun c => CoinTape.fintype (k c) (s c)
      inferInstanceAs (Fintype (C × ((c : C) → CoinTape (k c) (s c))))

theorem CoinTape.nonempty : (g : Game α) → (s : Strategy g) → Nonempty (CoinTape g s)
  | .ret _, _ => ⟨PUnit.unit⟩
  | .send _ k, s => CoinTape.nonempty (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      ⟨((Classical.arbitrary C, fun c => (CoinTape.nonempty (k c) (s c)).some) :
        C × ((c : C) → CoinTape (k c) (s c)))⟩

/-- **A tape's finitely many values**, as the instance (`CoinTape.fintype`). -/
noncomputable instance instFintypeCoinTape (g : Game α) (s : Strategy g) : Fintype (CoinTape g s) :=
  CoinTape.fintype g s

namespace DigestCut

/-- **The two-continuation finder** at the cut `K` along `s`, at `q` SHA-512 calls on every coin: a uniform tape of the
fork (`forkG`, `CoinTape`), its output the two messages the continuations reveal. -/
noncomputable def finder {g : Game α} (K : DigestCut Claims.Msg g) (s : Strategy g) (q : ℕ) : Claims.Finder where
  S := CoinTape (K.forkG s) (K.forkS s)
  fin := CoinTape.fintype _ _
  ne := CoinTape.nonempty _ _
  out := CoinTape.run (K.forkG s) (K.forkS s)
  cost _ := q

end DigestCut

end Game

end Verity
