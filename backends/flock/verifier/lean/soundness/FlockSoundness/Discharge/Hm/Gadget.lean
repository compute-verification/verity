/-
What the builder's rows force (`Builder`): a Hoare-style judgement `Sound m P` (whenever every row of the final state
holds, the result satisfies `P`) and the primitive gadgets' rules: a fresh bit, AND, majority, commit.
-/
import Mathlib.Tactic
import FlockSoundness.Discharge.Hm.Builder

namespace FlockSoundness.Discharge.Hm

namespace Lin

/-- Its value under the committed bits `z`. -/
def eval (z : Var → Bool) (f : Lin) : Bool := f.s.foldr (fun v b => z v ^^ b) f.c

theorem foldr_xor (z : Var → Bool) (l : List Var) (c d : Bool) :
    l.foldr (fun v b => z v ^^ b) (c ^^ d) = (l.foldr (fun v b => z v ^^ b) c ^^ d) := by
  induction l with
  | nil => rfl
  | cons v l ih => simp only [List.foldr_cons, ih, Bool.xor_assoc]

theorem foldr_symmDiff (z : Var → Bool) (s t : List Var) :
    (symmDiff s t).foldr (fun v b => z v ^^ b) false =
      (s.foldr (fun v b => z v ^^ b) false ^^ t.foldr (fun v b => z v ^^ b) false) := by
  fun_induction symmDiff s t with
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

theorem eval_eq (z : Var → Bool) (f : Lin) : f.eval z = (f.s.foldr (fun v b => z v ^^ b) false ^^ f.c) := by
  rw [eval, ← foldr_xor, Bool.false_xor]

@[simp] theorem eval_xor (z : Var → Bool) (f g : Lin) : (f ^^^ g).eval z = (f.eval z ^^ g.eval z) := by
  show (xor f g).eval z = _
  rw [eval_eq, eval_eq z f, eval_eq z g]
  simp only [xor, foldr_symmDiff]
  cases f.c <;> cases g.c <;> simp

@[simp] theorem eval_zero (z : Var → Bool) : zero.eval z = false := rfl
@[simp] theorem eval_one (z : Var → Bool) : one.eval z = true := rfl
@[simp] theorem eval_var (z : Var → Bool) (v : Var) : (var v).eval z = z v := by simp [var, eval]
@[simp] theorem eval_const (z : Var → Bool) (b : Bool) : (const b).eval z = b := rfl

theorem eval_of_isConst {f : Lin} (h : f.isConst = true) (z : Var → Bool) : f.eval z = f.c := by
  obtain ⟨s, c⟩ := f
  simp only [isConst, List.isEmpty_iff] at h
  subst h
  rfl

theorem eval_of_s_eq {f g : Lin} (h : f.s = g.s) (z : Var → Bool) : f.eval z = (g.eval z ^^ f.c ^^ g.c) := by
  rw [eval_eq z f, eval_eq z g, h]
  cases f.c <;> cases g.c <;> simp

end Lin

/-- A row holds under `z`. -/
def Row.Holds (z : Var → Bool) (r : Row) : Prop := z r.out = (r.a.eval z && r.b.eval z)

/-- Every row holds under `z`. -/
def St.Holds (z : Var → Bool) (s : St) : Prop := ∀ r ∈ s.rows, r.Holds z

/-- **What a builder program guarantees**: its rows only grow, and whenever every row of its final state holds under
`z`, its result satisfies `P z`. No freshness is needed: if the rows hold, they hold. -/
def Sound {α : Type} (m : M α) (P : (Var → Bool) → α → Prop) : Prop :=
  ∀ s : St, (∀ r ∈ s.rows, r ∈ (m.run s).2.rows) ∧ ∀ z, (m.run s).2.Holds z → P z (m.run s).1

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

theorem sound_fresh (a b : Lin) : Sound (fresh a b) fun z r => r.eval z = (a.eval z && b.eval z) := by
  intro s
  refine ⟨fun r hr => List.mem_cons_of_mem _ hr, fun z hz => ?_⟩
  have := hz ⟨s.next, a, b⟩ List.mem_cons_self
  simp only [Row.Holds] at this
  simpa [fresh, StateT.run] using this

theorem sound_input (n : ℕ) : Sound (input n) fun _ l => l.length = n := by
  intro s
  exact ⟨fun r hr => hr, fun z _ => by simp [input, StateT.run]⟩

theorem sound_and (a b : Lin) : Sound (and a b) fun z r => r.eval z = (a.eval z && b.eval z) := by
  unfold and
  refine Sound.ite' (fun ha => Sound.pure' fun z => ?_) fun ha => Sound.ite' (fun hb => Sound.pure' fun z => ?_)
    fun hb => Sound.ite' (fun he => Sound.pure' fun z => ?_) fun _ => sound_fresh a b
  · rw [Lin.eval_of_isConst ha z]; cases a.c <;> simp
  · rw [Lin.eval_of_isConst hb z]; cases b.c <;> simp
  · have hab := Lin.eval_of_s_eq he z
    by_cases hc : a.c = b.c
    · simp only [hc] at hab
      simp [hc, hab]
    · simp only [hc, ite_false, Lin.eval_zero, hab]
      cases ha' : a.c <;> cases hb' : b.c <;> simp_all

/-- The majority of three bits. -/
def maj3 (a b c : Bool) : Bool := (a && b) ^^ (a && c) ^^ (b && c)

theorem sound_maj (x y w : Lin) : Sound (maj x y w) fun z r => r.eval z = maj3 (x.eval z) (y.eval z) (w.eval z) := by
  unfold maj
  dsimp only
  split_ifs with h2 hs0 hs2
  all_goals first
    | (refine Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => ?_
       simp only [Lin.eval_xor, hr]
       cases x.eval z <;> cases y.eval z <;> cases w.eval z <;> rfl)
    | (refine Sound.pure' fun z => ?_
       have ex := fun h => Lin.eval_of_isConst (f := x) h z
       have ey := fun h => Lin.eval_of_isConst (f := y) h z
       have ew := fun h => Lin.eval_of_isConst (f := w) h z
       cases hx : x.isConst <;> cases hy : y.isConst <;> cases hw : w.isConst <;>
         simp only [hx, hy, hw, List.filter, Bool.not_true, Bool.not_false, List.length_cons, List.length_nil,
           List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at * <;>
         cases hxc : x.c <;> cases hyc : y.c <;> cases hwc : w.c <;>
         simp_all [maj3])

theorem sound_commit (f : Lin) : Sound (commit f) fun z r => r.eval z = f.eval z := by
  unfold commit
  refine Sound.ite' (fun _ => Sound.pure' fun _ => rfl) fun _ => Sound.mono (sound_fresh _ _) fun z r h => ?_
  simp [h]

end FlockSoundness.Discharge.Hm
