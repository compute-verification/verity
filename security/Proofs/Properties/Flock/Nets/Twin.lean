module

public import Flock.HmNets

/-!
# The builder as the kernel runs it

`HmNets.symm` is well-founded recursion, which the kernel doesn't reduce, so no run of the builder evaluates under
`decide`. `symmK` is the same merge by structural recursion (`symmK_eq`), and `beqK` the supports' equality as a `Bool`;
each `…K` below is its `HmNets` namesake with those two in place of `symm` and `decide (a.s = b.s)`, and equal to it
(`scheduleK_eq`, `roundsK_eq`, `ffK_eq` and the lemmas before them).
-/

@[expose] public section

open Flock Flock.HmNets

namespace FlockDeploy.Concrete.Nets

/-! ## The twin -/

/-- `symm` by structural recursion. -/
noncomputable def symmK : List Var → List Var → List Var := fun xs =>
  List.rec (motive := fun _ => List Var → List Var) (fun ys => ys)
    (fun x xs rec => fun ys => List.rec (motive := fun _ => List Var) (x :: xs)
      (fun y ys recY => cond (Nat.blt x y) (x :: rec (y :: ys)) (cond (Nat.blt y x) (y :: recY) (rec ys))) ys) xs

/-- Equality of two supports, by structural recursion. -/
noncomputable def beqK : List Var → List Var → Bool := fun xs =>
  List.rec (motive := fun _ => List Var → Bool)
    (fun ys => List.rec (motive := fun _ => Bool) true (fun _ _ _ => false) ys)
    (fun x _ rec => fun ys => List.rec (motive := fun _ => Bool) false (fun y ys _ => cond (Nat.beq x y) (rec ys) false) ys)
    xs

theorem blt_eq_decide (a b : Nat) : Nat.blt a b = decide (a < b) := by
  rw [Bool.eq_iff_iff, Nat.blt_eq, decide_eq_true_iff]

theorem beq_eq_decide (a b : Nat) : Nat.beq a b = decide (a = b) := by
  rw [Bool.eq_iff_iff, Nat.beq_eq, decide_eq_true_iff]

theorem symmK_eq : ∀ a b : List Var, symmK a b = symm a b
  | [], ys => by simp only [symm]; rfl
  | x :: xs, [] => by simp only [symm]; rfl
  | x :: xs, y :: ys => by
    have e : symmK (x :: xs) (y :: ys) =
        cond (Nat.blt x y) (x :: symmK xs (y :: ys)) (cond (Nat.blt y x) (y :: symmK (x :: xs) ys) (symmK xs ys)) := rfl
    rw [e, symmK_eq xs (y :: ys), symmK_eq (x :: xs) ys, symmK_eq xs ys, blt_eq_decide, blt_eq_decide]
    simp only [symm]
    by_cases h1 : x < y
    · simp [h1]
    · by_cases h2 : y < x <;> simp [h1, h2]

theorem beqK_eq : ∀ a b : List Var, beqK a b = decide (a = b)
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | x :: xs, y :: ys => by
    have e : beqK (x :: xs) (y :: ys) = cond (Nat.beq x y) (beqK xs ys) false := rfl
    rw [e, beqK_eq xs ys, beq_eq_decide]
    by_cases h : x = y <;> simp [h]

noncomputable def xorK (f g : Lin) : Lin := ⟨symmK f.s g.s, f.c ^^ g.c⟩

theorem xorK_eq (f g : Lin) : xorK f g = f ^^^ g := by
  show (⟨symmK f.s g.s, f.c ^^ g.c⟩ : Lin) = ⟨symm f.s g.s, f.c ^^ g.c⟩
  rw [symmK_eq]

theorem xorK_fun : xorK = (· ^^^ ·) := funext fun f => funext fun g => xorK_eq f g

noncomputable def andK (a b : Lin) : M Lin :=
  if a.isConst then pure (if a.c then b else Lin.zero)
  else if b.isConst then pure (if b.c then a else Lin.zero)
  else if beqK a.s b.s then pure (if a.c = b.c then a else Lin.zero)
  else fresh a b

theorem andK_eq (a b : Lin) : andK a b = HmNets.and a b := by
  unfold andK HmNets.and
  rw [beqK_eq]
  by_cases h : a.s = b.s <;> simp [h]

/-- `maj`, whose branch with two constants commits nothing and so is `maj`'s own. -/
noncomputable def majK (x y z : Lin) : M Lin :=
  if 2 ≤ ([x, y, z].filter (·.isConst)).length then maj x y z
  else do
    let r ← andK (xorK x z) (xorK y z)
    pure (xorK z r)

theorem majK_eq (x y z : Lin) : majK x y z = maj x y z := by
  unfold majK
  split
  · rfl
  · rename_i h
    simp only [maj, h, ↓reduceIte, xorK_eq, andK_eq]

noncomputable def xorWK (a b : Word) : Word := List.zipWith xorK a b
noncomputable def xor3K (a b c : Word) : Word := xorWK (xorWK a b) c

theorem xorWK_eq (a b : Word) : xorWK a b = xorW a b := by
  simp only [xorWK, xorW, xorK_fun]

theorem xor3K_eq (a b c : Word) : xor3K a b c = xor3 a b c := by
  simp only [xor3K, xor3, xorWK_eq]

noncomputable def chWK (e f g : Word) : M Word :=
  (List.zip e (List.zip f g)).mapM fun x => do
    let r ← andK x.1 (xorK x.2.1 x.2.2)
    pure (xorK x.2.2 r)

theorem chWK_eq (e f g : Word) : chWK e f g = chW e f g := by
  simp only [chWK, chW, xorK_eq, andK_eq]

noncomputable def majWK (a b c : Word) : M Word := (List.zip a (List.zip b c)).mapM fun x => majK x.1 x.2.1 x.2.2

theorem majWK_eq (a b c : Word) : majWK a b c = majW a b c := by
  simp only [majWK, majW, majK_eq]

noncomputable def add2GoK : Word → Word → Lin → Nat → M Word
  | x :: a, y :: b, c, i =>
    if a.isEmpty then pure [xorK (xorK x y) c]
    else do
      let c₁ ← majK x y c
      let c₂ ← carryCommit i c₁
      let r ← add2GoK a b c₂ (i + 1)
      pure ((xorK (xorK x y) c) :: r)
  | _, _, _, _ => pure []

theorem add2GoK_eq : ∀ (a b : Word) (c : Lin) (i : Nat), add2GoK a b c i = add2Go a b c i
  | x :: a, y :: b, c, i => by
    simp only [add2GoK, add2Go, xorK_eq, majK_eq]
    congr
    funext c₁
    congr
    funext c₂
    rw [add2GoK_eq a b c₂ (i + 1)]
  | [], _, _, _ => by simp only [add2GoK, add2Go]
  | _ :: _, [], _, _ => by simp only [add2GoK, add2Go]

noncomputable def add2K (a b : Word) : M Word := add2GoK a b Lin.zero 0

theorem add2K_eq (a b : Word) : add2K a b = add2 a b := add2GoK_eq a b Lin.zero 0

noncomputable def addNK : Word → List Word → M Word
  | acc, [] => pure acc
  | acc, x :: xs => do
    let acc ← add2K acc x
    addNK acc xs

theorem addNK_eq : ∀ (acc : Word) (xs : List Word), addNK acc xs = addN acc xs
  | acc, [] => rfl
  | acc, x :: xs => by
    simp only [addNK, addN, add2K_eq]
    congr
    funext acc'
    exact addNK_eq acc' xs

noncomputable def bsig0WK (a : Word) : Word := xor3K (rot a 28) (rot a 34) (rot a 39)
noncomputable def bsig1WK (e : Word) : Word := xor3K (rot e 14) (rot e 18) (rot e 41)
noncomputable def ssig0WK (w : Word) : Word := xor3K (rot w 1) (rot w 8) (shr w 7)
noncomputable def ssig1WK (w : Word) : Word := xor3K (rot w 19) (rot w 61) (shr w 6)

noncomputable def schedStepK (w : List Word) : M (List Word) := do
  let t := w.length
  let x ← addNK (w.getD (t - 16) []) [ssig0WK (w.getD (t - 15) []), w.getD (t - 7) [], ssig1WK (w.getD (t - 2) [])]
  let x ← commitW x
  pure (w ++ [x])

theorem schedStepK_eq (w : List Word) : schedStepK w = schedStep w := by
  simp only [schedStepK, schedStep, addNK_eq, ssig0WK, ssig1WK, ssig0W, ssig1W, xor3K_eq]

noncomputable def scheduleK : Nat → List Word → M (List Word)
  | 0, w => pure w
  | n + 1, w => do
    let w ← schedStepK w
    scheduleK n w

theorem scheduleK_eq : ∀ (n : Nat) (w : List Word), scheduleK n w = schedule n w
  | 0, w => rfl
  | n + 1, w => by
    simp only [scheduleK, schedule, schedStepK_eq]
    congr
    funext w'
    exact scheduleK_eq n w'

noncomputable def roundWK (k : BitVec 64) (wt : Word) (r : Regs Word) : M (Regs Word) := do
  let ch ← chWK r.e r.f r.g
  let t1 ← addNK r.h [bsig1WK r.e, ch, constW k, wt]
  let mj ← majWK r.a r.b r.c
  let t2 ← add2K (bsig0WK r.a) mj
  let e ← add2K r.d t1
  let e ← commitW e
  let a ← add2K t1 t2
  let a ← commitW a
  pure ⟨a, r.a, r.b, r.c, e, r.e, r.f, r.g⟩

theorem roundWK_eq (k : BitVec 64) (wt : Word) (r : Regs Word) : roundWK k wt r = roundW k wt r := by
  simp only [roundWK, roundW, chWK_eq, addNK_eq, majWK_eq, add2K_eq, bsig1WK, bsig0WK, bsig1W, bsig0W, xor3K_eq]

noncomputable def roundsK (w : List Word) : Nat → Nat → Regs Word → M (Regs Word)
  | 0, _, r => pure r
  | n + 1, t, r => do
    let r ← roundWK (kAt t) (w.getD t []) r
    roundsK w n (t + 1) r

theorem roundsK_eq (w : List Word) : ∀ (n t : Nat) (r : Regs Word), roundsK w n t r = rounds w n t r
  | 0, _, _ => rfl
  | n + 1, t, r => by
    simp only [roundsK, rounds, roundWK_eq]
    congr
    funext r'
    exact roundsK_eq w n (t + 1) r'

/-- The feed-forward of a compression: each chaining word plus its register. -/
def ff (cv : List Word) (r : Regs Word) : M (List Word) := (cv.zip r.toList).mapM fun x => add2 x.1 x.2

noncomputable def ffK (cv : List Word) (r : Regs Word) : M (List Word) := (cv.zip r.toList).mapM fun x => add2K x.1 x.2

theorem ffK_eq (cv : List Word) (r : Regs Word) : ffK cv r = ff cv r := by
  simp only [ffK, ff, add2K_eq]

end FlockDeploy.Concrete.Nets
