/-
What the rows of the verifier's builder (`Flock.HmNets`) force: a Hoare-style judgement `Sound m P` (whenever every row of
the final state holds, the result satisfies `P`) and the primitive gadgets' rules: a fresh bit, AND, majority, commit.
-/
import Mathlib.Tactic
import Flock.HmNets

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- A form's value under the committed bits `z`. -/
def ev (z : Var → Bool) (f : Lin) : Bool := f.s.foldr (fun v b => z v ^^ b) f.c

theorem foldr_xor (z : Var → Bool) (l : List Var) (c d : Bool) :
    l.foldr (fun v b => z v ^^ b) (c ^^ d) = (l.foldr (fun v b => z v ^^ b) c ^^ d) := by
  induction l with
  | nil => rfl
  | cons v l ih => simp only [List.foldr_cons, ih, Bool.xor_assoc]

theorem ev_eq (z : Var → Bool) (f : Lin) : ev z f = (f.s.foldr (fun v b => z v ^^ b) false ^^ f.c) := by
  rw [ev, ← foldr_xor, Bool.false_xor]

@[simp] theorem ev_xor (z : Var → Bool) (f g : Lin) : ev z (f ^^^ g) = (ev z f ^^ ev z g) := by
  show ev z (Lin.xor f g) = _
  rw [ev_eq, ev_eq z f, ev_eq z g]
  simp only [Lin.xor, Flock.HmNets.foldr_symm]
  cases f.c <;> cases g.c <;> simp

@[simp] theorem ev_zero (z : Var → Bool) : ev z Lin.zero = false := rfl
@[simp] theorem ev_one (z : Var → Bool) : ev z Lin.one = true := rfl
@[simp] theorem ev_var (z : Var → Bool) (v : Var) : ev z (Lin.var v) = z v := by simp [Lin.var, ev]
@[simp] theorem ev_const (z : Var → Bool) (b : Bool) : ev z (Lin.const b) = b := rfl

theorem ev_of_isConst {f : Lin} (h : f.isConst = true) (z : Var → Bool) : ev z f = f.c := by
  obtain ⟨s, c⟩ := f
  simp only [Lin.isConst, List.isEmpty_iff] at h
  subst h
  rfl

theorem ev_of_s_eq {f g : Lin} (h : f.s = g.s) (z : Var → Bool) : ev z f = (ev z g ^^ f.c ^^ g.c) := by
  rw [ev_eq z f, ev_eq z g, h]
  cases f.c <;> cases g.c <;> simp

/-- A row holds under `z`. -/
def rowHolds (z : Var → Bool) (r : Row) : Prop := z r.out = (ev z r.a && ev z r.b)

/-- Every row holds under `z`. -/
def stHolds (z : Var → Bool) (s : St) : Prop := ∀ r ∈ s.rows, rowHolds z r

/-- **What a builder program guarantees**: its rows only grow, and whenever every row of its final state holds under
`z`, its result satisfies `P z`. No freshness is needed: if the rows hold, they hold. -/
def Sound {α : Type} (m : M α) (P : (Var → Bool) → α → Prop) : Prop :=
  ∀ s : St, (∀ r ∈ s.rows, r ∈ (m.run s).2.rows) ∧ ∀ z, stHolds z (m.run s).2 → P z (m.run s).1

namespace Sound

variable {α β : Type}

theorem pure' {a : α} {P : (Var → Bool) → α → Prop} (h : ∀ z, P z a) : Sound (pure a) P :=
  fun _ => ⟨fun _ hr => hr, fun z _ => h z⟩

theorem bind' {m : M α} {f : α → M β} {P : (Var → Bool) → α → Prop} {Q : (Var → Bool) → β → Prop}
    (hm : Sound m P) (hf : ∀ a, Sound (f a) fun z b => P z a → Q z b) : Sound (m >>= f) Q := by
  intro s
  obtain ⟨hs1, hs2⟩ := hm s
  obtain ⟨hf1, hf2⟩ := hf (m.run s).1 (m.run s).2
  refine ⟨fun r hr => hf1 r (hs1 r hr), fun z hz => ?_⟩
  exact hf2 z hz (hs2 z fun r hr => hz r (hf1 r hr))

theorem mono {m : M α} {P Q : (Var → Bool) → α → Prop} (hm : Sound m P) (h : ∀ z a, P z a → Q z a) :
    Sound m Q := fun s => ⟨(hm s).1, fun z hz => h z _ ((hm s).2 z hz)⟩

theorem ite' {c : Prop} [Decidable c] {m₁ m₂ : M α} {P : (Var → Bool) → α → Prop} (h₁ : c → Sound m₁ P)
    (h₂ : ¬ c → Sound m₂ P) : Sound (if c then m₁ else m₂) P := by
  split_ifs with hc
  · exact h₁ hc
  · exact h₂ hc

end Sound

theorem sound_fresh (a b : Lin) : Sound (fresh a b) fun z r => ev z r = (ev z a && ev z b) := by
  intro s
  refine ⟨fun r hr => List.mem_cons_of_mem _ hr, fun z hz => ?_⟩
  have := hz ⟨s.next, a, b⟩ List.mem_cons_self
  simp only [rowHolds] at this
  simpa [fresh, StateT.run] using this

theorem sound_input (n : ℕ) : Sound (input n) fun _ l => l.length = n := by
  intro s
  exact ⟨fun r hr => hr, fun z _ => by simp [input, StateT.run]⟩

theorem sound_and (a b : Lin) : Sound (Flock.HmNets.and a b) fun z r => ev z r = (ev z a && ev z b) := by
  unfold Flock.HmNets.and
  refine Sound.ite' (fun ha => Sound.pure' fun z => ?_) fun ha => Sound.ite' (fun hb => Sound.pure' fun z => ?_)
    fun hb => Sound.ite' (fun he => Sound.pure' fun z => ?_) fun _ => sound_fresh a b
  · rw [ev_of_isConst ha z]; cases a.c <;> simp
  · rw [ev_of_isConst hb z]; cases b.c <;> simp
  · have hab := ev_of_s_eq he z
    by_cases hc : a.c = b.c
    · simp only [hc] at hab
      simp [hc, hab]
    · simp only [hc, ite_false, ev_zero, hab]
      cases ha' : a.c <;> cases hb' : b.c <;> simp_all

/-- The majority of three bits. -/
def maj3 (a b c : Bool) : Bool := (a && b) ^^ (a && c) ^^ (b && c)

theorem sound_maj (x y w : Lin) : Sound (maj x y w) fun z r => ev z r = maj3 (ev z x) (ev z y) (ev z w) := by
  unfold maj
  dsimp only
  split_ifs with h2 hs0 hs2
  all_goals first
    | (refine Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => ?_
       simp only [ev_xor, hr]
       cases ev z x <;> cases ev z y <;> cases ev z w <;> rfl)
    | (refine Sound.pure' fun z => ?_
       have ex := fun h => ev_of_isConst (f := x) h z
       have ey := fun h => ev_of_isConst (f := y) h z
       have ew := fun h => ev_of_isConst (f := w) h z
       cases hx : x.isConst <;> cases hy : y.isConst <;> cases hw : w.isConst <;>
         simp only [hx, hy, hw, List.filter, Bool.not_true, Bool.not_false, List.length_cons, List.length_nil,
           List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at * <;>
         cases hxc : x.c <;> cases hyc : y.c <;> cases hwc : w.c <;>
         simp_all [maj3])

theorem sound_commit (f : Lin) : Sound (commit f) fun z r => ev z r = ev z f := by
  unfold commit
  refine Sound.ite' (fun _ => Sound.pure' fun _ => rfl) fun _ => Sound.mono (sound_fresh _ _) fun z r h => ?_
  simp [h]

end FlockSoundness.Discharge.Hm
