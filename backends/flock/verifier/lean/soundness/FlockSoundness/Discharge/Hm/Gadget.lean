/-
What the rows of the verifier's builder (`Flock.HmNets`) force: a Hoare-style judgement `Sound m P` (whenever every row of
the final state holds, the result satisfies `P`) and the primitive gadgets' rules: a fresh bit, AND, majority, commit.
-/
import Mathlib.Tactic
import Flock.HmNets

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and round Regs

/-- A form's value under the committed bits `z`. -/
def ev (z : ℕ → Bool) (f : Lin) : Bool := f.s.foldr (fun v b => z v ^^ b) f.c

theorem foldr_xor (z : ℕ → Bool) (l : List ℕ) (c d : Bool) :
    l.foldr (fun v b => z v ^^ b) (c ^^ d) = (l.foldr (fun v b => z v ^^ b) c ^^ d) := by
  induction l with
  | nil => rfl
  | cons v l ih => simp only [List.foldr_cons, ih, Bool.xor_assoc]

theorem foldr_symm (z : ℕ → Bool) (s t : List ℕ) :
    (symm s t).foldr (fun v b => z v ^^ b) false =
      (s.foldr (fun v b => z v ^^ b) false ^^ t.foldr (fun v b => z v ^^ b) false) := by
  fun_induction symm s t with
  | case1 t => simp
  | case2 a s => simp
  | case3 a s b t h ih => simp only [List.foldr_cons, ih, Bool.xor_assoc]
  | case4 a s b t h h' ih =>
      simp only [List.foldr_cons, ih]
      cases z a <;> cases z b <;> simp
  | case5 a s b t h h' ih =>
      obtain rfl : a = b := Nat.le_antisymm (Nat.not_lt.mp h') (Nat.not_lt.mp h)
      simp only [List.foldr_cons, ih]
      cases z a <;> simp

theorem ev_eq (z : ℕ → Bool) (f : Lin) : ev z f = (f.s.foldr (fun v b => z v ^^ b) false ^^ f.c) := by
  rw [ev, ← foldr_xor, Bool.false_xor]

@[simp] theorem ev_xor (z : ℕ → Bool) (f g : Lin) : ev z (f ^^^ g) = (ev z f ^^ ev z g) := by
  show ev z (Lin.xor f g) = _
  rw [ev_eq, ev_eq z f, ev_eq z g]
  simp only [Lin.xor, foldr_symm]
  cases f.c <;> cases g.c <;> simp

@[simp] theorem ev_zero (z : ℕ → Bool) : ev z zero = false := rfl
@[simp] theorem ev_one (z : ℕ → Bool) : ev z one = true := rfl
@[simp] theorem ev_var (z : ℕ → Bool) (v : ℕ) : ev z (var v) = z v := by simp [var, ev]

theorem ev_of_isConst {f : Lin} (h : f.isConst = true) (z : ℕ → Bool) : ev z f = f.c := by
  obtain ⟨s, c⟩ := f
  simp only [Lin.isConst, List.isEmpty_iff] at h
  subst h
  rfl

theorem ev_of_s_eq {f g : Lin} (h : f.s = g.s) (z : ℕ → Bool) : ev z f = (ev z g ^^ f.c ^^ g.c) := by
  rw [ev_eq z f, ev_eq z g, h]
  cases f.c <;> cases g.c <;> simp

/-- A row holds under `z`. -/
def rowHolds (z : ℕ → Bool) (r : Row) : Prop := z r.out = (ev z r.a && ev z r.b)

/-- Every row holds under `z`. -/
def stHolds (z : ℕ → Bool) (s : St) : Prop := ∀ r ∈ s.rows, rowHolds z r

/-- **What a builder program guarantees**: its rows only grow, and whenever every row of its final state holds under
`z`, its result satisfies `P z`. No freshness is needed: if the rows hold, they hold. -/
def Sound {α : Type} (m : M α) (P : (ℕ → Bool) → α → Prop) : Prop :=
  ∀ s : St, (∀ r ∈ s.rows, r ∈ (m.run s).2.rows) ∧ ∀ z, stHolds z (m.run s).2 → P z (m.run s).1

namespace Sound

variable {α β : Type}

theorem pure' {a : α} {P : (ℕ → Bool) → α → Prop} (h : ∀ z, P z a) : Sound (pure a) P :=
  fun _ => ⟨fun _ hr => hr, fun z _ => h z⟩

theorem bind' {m : M α} {f : α → M β} {P : (ℕ → Bool) → α → Prop} {Q : (ℕ → Bool) → β → Prop}
    (hm : Sound m P) (hf : ∀ a, Sound (f a) fun z b => P z a → Q z b) : Sound (m >>= f) Q := by
  intro s
  obtain ⟨hs1, hs2⟩ := hm s
  obtain ⟨hf1, hf2⟩ := hf (m.run s).1 (m.run s).2
  refine ⟨fun r hr => hf1 r (hs1 r hr), fun z hz => ?_⟩
  exact hf2 z hz (hs2 z fun r hr => hz r (hf1 r hr))

theorem mono {m : M α} {P Q : (ℕ → Bool) → α → Prop} (hm : Sound m P) (h : ∀ z a, P z a → Q z a) :
    Sound m Q := fun s => ⟨(hm s).1, fun z hz => h z _ ((hm s).2 z hz)⟩

theorem ite' {c : Prop} [Decidable c] {m₁ m₂ : M α} {P : (ℕ → Bool) → α → Prop} (h₁ : c → Sound m₁ P)
    (h₂ : ¬ c → Sound m₂ P) : Sound (if c then m₁ else m₂) P := by
  split_ifs with hc
  · exact h₁ hc
  · exact h₂ hc

end Sound

theorem run_fresh (a b : Lin) (s : St) :
    (fresh a b).run s = (var s.next, { next := s.next + 1, rows := s.rows.push ⟨s.next, a, b⟩ }) := rfl

theorem sound_fresh (a b : Lin) : Sound (fresh a b) fun z r => ev z r = (ev z a && ev z b) := by
  intro s
  rw [run_fresh]
  refine ⟨fun r hr => Array.mem_push_of_mem _ hr, fun z hz => ?_⟩
  have := hz ⟨s.next, a, b⟩ (Array.mem_push_self ..)
  simpa [rowHolds] using this

theorem run_input (n : ℕ) (s : St) :
    (input n).run s = ((List.range n).map (var <| s.next + ·), { s with next := s.next + n }) := rfl

theorem sound_input (n : ℕ) : Sound (input n) fun _ l => l.length = n := by
  intro s
  rw [run_input]
  exact ⟨fun r hr => hr, fun z _ => by simp⟩

theorem sound_and (a b : Lin) : Sound (Flock.HmNets.and a b) fun z r => ev z r = (ev z a && ev z b) := by
  unfold Flock.HmNets.and
  refine Sound.ite' (fun ha => Sound.pure' fun z => ?_) fun ha => Sound.ite' (fun hb => Sound.pure' fun z => ?_)
    fun hb => Sound.ite' (fun he => Sound.pure' fun z => ?_) fun _ => sound_fresh a b
  · rw [ev_of_isConst ha z]; cases a.c <;> simp
  · rw [ev_of_isConst hb z]; cases b.c <;> simp
  · have hab := ev_of_s_eq (beq_iff_eq.mp he) z
    by_cases hc : a.c = b.c
    · simp only [hc] at hab
      simp [hc, hab]
    · simp only [beq_iff_eq, hc, ite_false, ev_zero, hab]
      cases ha' : a.c <;> cases hb' : b.c <;> simp_all

/-- The majority of three bits. -/
def maj3 (a b c : Bool) : Bool := (a && b) ^^ (a && c) ^^ (b && c)

@[simp] theorem isConst_zero : zero.isConst = true := rfl
@[simp] theorem isConst_one : one.isConst = true := rfl

/-- A form that is constant is `zero` or `one`. -/
theorem eq_of_isConst {f : Lin} (h : f.isConst = true) : f = zero ∨ f = one := by
  obtain ⟨s, c⟩ := f
  simp only [Lin.isConst, List.isEmpty_iff] at h
  subst h
  cases c
  · exact Or.inl rfl
  · exact Or.inr rfl

set_option linter.unusedSimpArgs false in
theorem sound_maj (x y w : Lin) :
    Sound (Flock.HmNets.maj x y w) fun z r => ev z r = maj3 (ev z x) (ev z y) (ev z w) := by
  have key : ∀ f : Lin, f.isConst = false ∨ f = zero ∨ f = one := fun f => by
    cases h : f.isConst
    · exact Or.inl rfl
    · exact Or.inr (eq_of_isConst h)
  rcases key x with hx | hx | hx <;> rcases key y with hy | hy | hy <;> rcases key w with hw | hw | hw
  all_goals
    unfold Flock.HmNets.maj
    simp (config := { decide := true }) only [List.filter, hx, hy, hw, isConst_zero, isConst_one, Bool.not_true,
      Bool.not_false, List.length_cons, List.length_nil, ↓reduceIte]
  all_goals first
    | (refine Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => ?_
       simp only [ev_xor, hr]
       cases ev z x <;> cases ev z y <;> cases ev z w <;> rfl)
    | (refine Sound.pure' fun z => ?_
       simp only [ev_zero, ev_one, maj3]
       cases ev z x <;> cases ev z y <;> cases ev z w <;> rfl)

theorem sound_commitBit (f : Lin) : Sound (commitBit f) fun z r => ev z r = ev z f := by
  unfold commitBit
  refine Sound.ite' (fun _ => Sound.pure' fun _ => rfl) fun _ => Sound.mono (sound_fresh _ _) fun z r h => ?_
  simp [h]

end FlockSoundness.Discharge.Hm
