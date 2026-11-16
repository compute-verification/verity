module

public import Proofs.Flock.ZkAny.Generic

/-!
# Sessions in sequence, per outcome

An audit's sampled sessions run one after another, and the auditor may choose its moves in each from the bytes the
earlier ones released. `PerOutcome.stream` is `PerOutcome.chain` iterated over `ℓ` sessions: where each session is
simulated per outcome at every value of the earlier sessions' randomness, by a simulator that reads only their outcomes
and bytes, the `ℓ` sessions are simulated per outcome at the tuple of their outcomes, the simulators run in turn, with
the errors added.

* `Tup F ℓ`: one entry per session, oldest first; finite and nonempty where each entry is.
* `Tup.rels`, `Tup.outs`: the sessions' bytes and outcomes on their randomness, session `j` reading the earlier
  sessions' randomness and bytes and its own randomness.
* `Tup.sims`, `Tup.errs`: the simulated bytes at the sessions' outcomes, session `j`'s simulator given the earlier
  outcomes and simulated bytes; and the errors added.
* `PerOutcome.mono`, `PerOutcome.comap_sim`, `PerOutcome.map_out`: a larger error, the simulator's coins read through a
  bijection, and the outcomes through an injective map.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **One entry per session, oldest first**: `Tup F 0` is empty and `Tup F (ℓ + 1)` is `Tup F ℓ` and then `F ℓ`. -/
def Tup (F : ℕ → Type) : ℕ → Type
  | 0 => Unit
  | ℓ + 1 => Tup F ℓ × F ℓ

namespace Tup

variable {F : ℕ → Type}

/-- Tuples of finite entries are finite. -/
@[instance_reducible]
def fintype' [∀ j, Fintype (F j)] : ∀ ℓ, Fintype (Tup F ℓ)
  | 0 => inferInstanceAs (Fintype Unit)
  | ℓ + 1 => letI := fintype' ℓ; inferInstanceAs (Fintype (Tup F ℓ × F ℓ))

instance instFintype [∀ j, Fintype (F j)] (ℓ : ℕ) : Fintype (Tup F ℓ) := fintype' ℓ

/-- Tuples of nonempty entries are nonempty. -/
theorem nonempty' [∀ j, Nonempty (F j)] : ∀ ℓ, Nonempty (Tup F ℓ)
  | 0 => ⟨()⟩
  | ℓ + 1 => haveI := nonempty' ℓ; inferInstanceAs (Nonempty (Tup F ℓ × F ℓ))

instance instNonempty [∀ j, Nonempty (F j)] (ℓ : ℕ) : Nonempty (Tup F ℓ) := nonempty' ℓ

variable {A C : ℕ → Type} {St V : Type}

/-- **The bytes of `ℓ` sessions** on their randomness: session `j`'s are `r j` at the earlier sessions' randomness and
bytes and its own randomness. -/
def rels (r : (j : ℕ) → Tup A j → Tup (fun _ => V) j → A j → V) : (ℓ : ℕ) → Tup A ℓ → Tup (fun _ => V) ℓ
  | 0, _ => ()
  | ℓ + 1, p => (rels r ℓ p.1, r ℓ p.1 (rels r ℓ p.1) p.2)

/-- **The outcomes of `ℓ` sessions** on their randomness: session `j`'s is `o j` at the earlier sessions' randomness and
bytes and its own randomness. -/
def outs (o : (j : ℕ) → Tup A j → Tup (fun _ => V) j → A j → St)
    (r : (j : ℕ) → Tup A j → Tup (fun _ => V) j → A j → V) : (ℓ : ℕ) → Tup A ℓ → Tup (fun _ => St) ℓ
  | 0, _ => ()
  | ℓ + 1, p => (outs o r ℓ p.1, o ℓ p.1 (rels r ℓ p.1) p.2)

/-- **The simulated bytes of `ℓ` sessions** at their outcomes `s` and the simulators' coins `c`: session `j`'s are its
simulator `S j` at the earlier outcomes and simulated bytes, its own outcome and its own coins. -/
def sims (S : (j : ℕ) → Tup (fun _ => St) j → Tup (fun _ => V) j → St → C j → V) :
    (ℓ : ℕ) → Tup (fun _ => St) ℓ → Tup C ℓ → Tup (fun _ => V) ℓ
  | 0, _, _ => ()
  | ℓ + 1, s, c => (sims S ℓ s.1 c.1, S ℓ s.1 (sims S ℓ s.1 c.1) s.2 c.2)

/-- **The errors of `ℓ` sessions** at their outcomes, added. -/
def errs (ε : (j : ℕ) → St → ℝ≥0∞) : (ℓ : ℕ) → Tup (fun _ => St) ℓ → ℝ≥0∞
  | 0, _ => 0
  | ℓ + 1, s => errs ε ℓ s.1 + ε ℓ s.2

end Tup

/-- **A larger error.** -/
theorem PerOutcome.mono {Ω St Ωs V : Type} [Fintype Ω] [Fintype Ωs] {out : Ω → St} {rel : Ω → V}
    {S : St → Ωs → V} {ε ε' : St → ℝ≥0∞} (h : PerOutcome out rel S ε) (hε : ∀ s, ε s ≤ ε' s) :
    PerOutcome out rel S ε' :=
  fun s E => (h s E).trans (add_le_add le_rfl (hε s))

/-- **The simulator's coins read through a bijection** `e`: uniform coins of `Ωs'` are uniform coins of `Ωs`. -/
theorem PerOutcome.comap_sim {Ω St Ωs Ωs' V : Type} [Fintype Ω] [Fintype Ωs] [Fintype Ωs'] {out : Ω → St}
    {rel : Ω → V} {S : St → Ωs → V} {ε : St → ℝ≥0∞} (h : PerOutcome out rel S ε) (e : Ωs' ≃ Ωs) :
    PerOutcome out rel (fun s c => S s (e c)) ε :=
  fun s E => (h s E).trans_eq (congrArg (· + ε s) (prCoin_congr_equiv e fun σ => E (S s σ)).symm)

/-- **The outcomes read through an injective map** `f`: at an outcome `f s` the simulator is `S s`, and at one outside
`f`'s range any simulator does, since the outcome never takes it. -/
theorem PerOutcome.map_out {Ω St St' Ωs V : Type} [Fintype Ω] [Fintype Ωs] {out : Ω → St} {rel : Ω → V}
    {S : St → Ωs → V} {ε : St → ℝ≥0∞} (h : PerOutcome out rel S ε) {f : St → St'} (hf : Function.Injective f)
    {S' : St' → Ωs → V} (hS : ∀ s, S' (f s) = S s) {ε' : St' → ℝ≥0∞} (hε : ∀ s, ε s ≤ ε' (f s)) :
    PerOutcome (fun ω => f (out ω)) rel S' ε' := by
  intro s' E
  by_cases hs : ∃ s, f s = s'
  · obtain ⟨s, rfl⟩ := hs
    calc prCoin (fun ω => f (out ω) = f s ∧ E (rel ω)) ≤ prCoin (fun ω => out ω = s ∧ E (rel ω)) :=
          prCoin_mono fun ω hω => ⟨hf hω.1, hω.2⟩
      _ ≤ prCoin (fun σ => E (S s σ)) + ε s := h s E
      _ ≤ prCoin (fun σ => E (S' (f s) σ)) + ε' (f s) := by rw [hS]; exact add_le_add le_rfl (hε s)
  · calc prCoin (fun ω => f (out ω) = s' ∧ E (rel ω)) ≤ prCoin (fun _ : Ω => False) :=
          prCoin_mono fun ω hω => hs ⟨out ω, hω.1⟩
      _ = 0 := by classical rw [prCoin_eq_card]; simp
      _ ≤ _ := zero_le

/-- **Sessions in sequence, per outcome** (`PerOutcome.chain`, iterated). Where each session `j < ℓ` is simulated per
outcome at every value `π` of the earlier sessions' randomness, by its simulator `S j` given the earlier sessions'
outcomes and bytes, the `ℓ` sessions are simulated per outcome at the tuple of their outcomes, by the simulators in
turn, each given the earlier outcomes and simulated bytes, with the errors added. Session `j`'s outcome and bytes may
read all of the earlier sessions' randomness; its simulator reads only what they released and their outcomes. -/
theorem PerOutcome.stream {A C : ℕ → Type} {St V : Type} [∀ j, Fintype (A j)] [∀ j, Nonempty (A j)]
    [∀ j, Fintype (C j)] [∀ j, Nonempty (C j)]
    (o : (j : ℕ) → Tup A j → Tup (fun _ => V) j → A j → St) (r : (j : ℕ) → Tup A j → Tup (fun _ => V) j → A j → V)
    (S : (j : ℕ) → Tup (fun _ => St) j → Tup (fun _ => V) j → St → C j → V) (ε : (j : ℕ) → St → ℝ≥0∞) (ℓ : ℕ)
    (h : ∀ j < ℓ, ∀ π : Tup A j, PerOutcome (o j π (Tup.rels r j π)) (r j π (Tup.rels r j π))
      (S j (Tup.outs o r j π) (Tup.rels r j π)) (ε j)) :
    PerOutcome (Tup.outs o r ℓ) (Tup.rels r ℓ) (Tup.sims S ℓ) (Tup.errs ε ℓ) := by
  induction ℓ with
  | zero =>
    intro s E
    calc prCoin (fun ω : Tup A 0 => Tup.outs o r 0 ω = s ∧ E (Tup.rels r 0 ω))
        ≤ prCoin (fun _ : Tup A 0 => E ()) := prCoin_mono fun _ hω => hω.2
      _ = prCoin (fun _ : Tup C 0 => E ()) :=
          prCoin_congr_equiv (Equiv.refl Unit : Tup A 0 ≃ Tup C 0) fun _ => E ()
      _ ≤ prCoin (fun _ : Tup C 0 => E ()) + 0 := le_self_add
  | succ ℓ ih =>
    exact PerOutcome.chain (out₁ := Tup.outs o r ℓ) (rel₁ := Tup.rels r ℓ) (S₁ := Tup.sims S ℓ)
      (ε₁ := Tup.errs ε ℓ) (out₂ := fun π => o ℓ π (Tup.rels r ℓ π)) (rel₂ := fun π => r ℓ π (Tup.rels r ℓ π))
      (S₂ := S ℓ) (ε₂ := ε ℓ) (ih fun j hj => h j (Nat.lt_succ_of_lt hj)) (h ℓ (Nat.lt_succ_self ℓ))

end FlockSoundness.ZK
