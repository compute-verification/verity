import Flock.RecAlgebra
import Proofs.Flock.Soundness.Refine.Arith

/-!
# V*'s algebra over symbolic values, evaluated

`run_sim`: `run` over the symbolic values (`Lin`), evaluated at the message elements' values, is `run` over F on the
evaluated messages, with the same coins drawn: the residuals are the symbolic residuals evaluated, and the coin-derived
points are the same.

* `Lin.eval` reads a symbolic value at the elements' values; it is a `Hom`: it commutes with every operation `run`
  uses (zero, `^`, `sm`, `mv` where it succeeds, constants).
* `Sim f x y`: whenever the coin computation `x` succeeds, `y` succeeds with `f` of its result, the same coins drawn;
  `SimL` is the same through Ligerito's state. Each function `run` calls has its lemma, in the order `run` calls them.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine FlockLevel3

/-! ## Evaluating symbolic values -/

/-- A dict's value: `Σ x · g k` over its entries. -/
def tEval {κ : Type} (g : κ → F128) : Terms κ → F128
  | [] => 0
  | (k, x) :: r => x * g k + tEval g r

theorem tEval_addAt {κ : Type} [DecidableEq κ] (g : κ → F128) :
    ∀ (ts : Terms κ) (k : κ) (v : F128), tEval g (addAt ts k v) = tEval g ts + v * g k
  | [], k, v => by
    simp only [addAt, tEval]
    (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_zero]; ring)
  | (k', x) :: r, k, v => by
    unfold addAt
    split
    · rename_i h; subst h; simp only [tEval]
      (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)
    · simp only [tEval, tEval_addAt g r k v]
      (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)

theorem tEval_mergeT {κ : Type} [DecidableEq κ] (g : κ → F128) :
    ∀ (b a : Terms κ), tEval g (mergeT a b) = tEval g a + tEval g b
  | [], a => by
    simp only [mergeT, List.foldl_nil, tEval]
    (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
  | (k, x) :: r, a => by
    have := tEval_mergeT g r (addAt a k x)
    simp only [mergeT, List.foldl_cons] at this ⊢
    rw [this, tEval_addAt]; simp only [tEval]
    (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)

theorem tEval_scale {κ : Type} (g : κ → F128) (s : F128) :
    ∀ ts : Terms κ, tEval g (ts.map fun kv => (kv.1, s * kv.2)) = s * tEval g ts
  | [] => by
    simp only [List.map_nil, tEval]
    (apply fOf_injective; simp only [fOf_mul, fOf_zero]; ring)
  | (k, x) :: r => by
    simp only [List.map_cons, tEval, tEval_scale g s r]
    (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)

/-- A symbolic value at the elements' values `val`: `Σ t[k]·val k + c + Σ q[(a, b)]·val a·val b`. -/
def Lin.eval (val : Key → F128) (a : Lin) : F128 :=
  tEval val a.t + a.c.getD 0 + tEval (fun ab => val ab.1 * val ab.2) a.q

theorem Lin.eval_xor (val : Key → F128) (a b : Lin) : (a.xor b).eval val = a.eval val + b.eval val := by
  simp only [Lin.eval, Lin.xor, tEval_mergeT]
  rcases a with ⟨at', ac, aq⟩
  rcases b with ⟨bt, bc, bq⟩
  cases ac <;> cases bc <;> simp only [Option.getD]
  all_goals (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem Lin.eval_scale (val : Key → F128) (s : F128) (a : Lin) : (a.scale s).eval val = s * a.eval val := by
  simp only [Lin.eval, Lin.scale, tEval_scale]
  cases a.c <;> simp only [Option.map, Option.getD]
  all_goals (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_zero]; ring)

theorem Lin.eval_mv (val : Key → F128) {x y z : Lin} (h : x.mv y = .ok z) : z.eval val = x.eval val * y.eval val := by
  unfold Lin.mv at h
  split at h
  · rename_i kx a ky b
    split at h
    · rename_i hab
      obtain ⟨rfl, rfl⟩ := hab
      cases h
      simp only [Lin.eval, tEval, Option.getD]
      (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_zero, fOf_one]; ring)
    · cases h
  · cases h

theorem Lin.eval_zero (val : Key → F128) : Lin.eval val {} = 0 := by
  simp only [Lin.eval, tEval, Option.getD]; (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem Lin.eval_cst (val : Key → F128) (v : F128) : Lin.eval val { c := some v } = v := by
  simp only [Lin.eval, tEval, Option.getD]; (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem Lin.eval_var (val : Key → F128) (k : Key) : (Lin.var k).eval val = val k := by
  simp only [Lin.eval, Lin.var, tEval, Option.getD]
  (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_zero, fOf_one]; ring)

/-! ## Maps that commute with the operations -/

/-- `h` commutes with `run`'s operations: zero, `^`, `sm`, `mv` where it succeeds, and constants. -/
structure Hom {α β : Type} [Val α] [Val β] (h : α → β) : Prop where
  zero : h Val.zero = Val.zero
  add : ∀ a b, h (Val.add a b) = Val.add (h a) (h b)
  sm : ∀ c a, h (Val.sm c a) = Val.sm c (h a)
  mv : ∀ a b z, Val.mv a b = .ok z → Val.mv (h a) (h b) = .ok (h z)
  cst : ∀ c, h (Val.cst c) = Val.cst c

/-- Evaluation is a `Hom` from the symbolic values to F. -/
theorem Lin.eval_hom (val : Key → F128) : Hom (Lin.eval val) where
  zero := Lin.eval_zero val
  add := Lin.eval_xor val
  sm := Lin.eval_scale val
  mv := fun a b z h => by
    show Except.ok (Lin.eval val a * Lin.eval val b) = .ok (Lin.eval val z)
    rw [Lin.eval_mv val h]
  cst := Lin.eval_cst val

section Hom

variable {α β : Type} [Val α] [Val β] {h : α → β}

/-- A K value's image, coordinate-wise. -/
def KV.map (h : α → β) (x : KV α) : KV β := (h x.1, h x.2)

/-- A round polynomial's image, coefficient-wise. -/
def KV3.map (h : α → β) (q : KV α × KV α × KV α) : KV β × KV β × KV β :=
  (KV.map h q.1, KV.map h q.2.1, KV.map h q.2.2)

theorem Hom.foldl_add (hh : Hom h) {γ : Type} (f : γ → α) (g : γ → β) (hfg : ∀ c, h (f c) = g c) :
    ∀ (l : List γ) (acc : α), h (l.foldl (fun acc c => Val.add acc (f c)) acc) =
      l.foldl (fun acc c => Val.add acc (g c)) (h acc)
  | [], _ => rfl
  | c :: l, acc => by simp only [List.foldl_cons]; rw [hh.foldl_add f g hfg l, hh.add, hfg]

theorem Hom.dotV (hh : Hom h) (cs : Array F128) (xs : Array α) : h (dotV cs xs) = dotV cs (xs.map h) := by
  unfold Flock.RecAlgebra.dotV
  rw [← Array.foldl_toList, ← Array.foldl_toList,
    hh.foldl_add (fun cx : F128 × α => Val.sm cx.1 cx.2) (fun cx => Val.sm cx.1 (h cx.2)) (fun cx => hh.sm _ _),
    hh.zero]
  simp only [Array.toList_zip, Array.toList_map]
  rw [List.zip_map_right, List.foldl_map]
  rfl

theorem Hom.zipWith_add (hh : Hom h) (a b : Array α) :
    (a.zipWith Val.add b).map h = (a.map h).zipWith Val.add (b.map h) := by
  rw [Array.zipWith_map, Array.map_zipWith]
  simp only [hh.add]

theorem Hom.kadd (hh : Hom h) (x y : KV α) : KV.map h (kadd x y) = kadd (KV.map h x) (KV.map h y) := by
  simp only [KV.map, Flock.RecAlgebra.kadd, hh.add]

theorem Hom.kc (hh : Hom h) (c : F256) (x : KV α) : KV.map h (kc c x) = kc c (KV.map h x) := by
  simp only [KV.map, Flock.RecAlgebra.kc, hh.add, hh.sm]

theorem Hom.kf (hh : Hom h) (c : F256) (x : α) : KV.map h (kf c x) = kf c (h x) := by
  simp only [KV.map, Flock.RecAlgebra.kf, hh.sm]

theorem Hom.fk (hh : Hom h) (b : F128) (x : KV α) : KV.map h (fk b x) = fk b (KV.map h x) := by
  simp only [KV.map, Flock.RecAlgebra.fk, hh.sm]

theorem Hom.kofF (hh : Hom h) (y : α) : KV.map h (kofF y) = kofF (h y) := by
  simp only [KV.map, Flock.RecAlgebra.kofF, hh.zero]

theorem Hom.kaddAll (hh : Hom h) (xs : List (KV α)) :
    KV.map h (kaddAll xs) = kaddAll (xs.map (KV.map h)) := by
  unfold Flock.RecAlgebra.kaddAll
  suffices ∀ (l : List (KV α)) (acc : KV α), KV.map h (l.foldl Flock.RecAlgebra.kadd acc) =
      (l.map (KV.map h)).foldl Flock.RecAlgebra.kadd (KV.map h acc) by
    rw [this]; simp only [KV.map, hh.zero]
  intro l
  induction l with
  | nil => intro _; rfl
  | cons x l ih => intro acc; simp only [List.foldl_cons, List.map_cons]; rw [ih, hh.kadd]

theorem Hom.ofMsg (hh : Hom h) (mm : KV α × KV α) (s : KV α) :
    KV3.map h (ofMsg mm s) = ofMsg (KV.map h mm.1, KV.map h mm.2) (KV.map h s) := by
  simp only [KV3.map, Flock.RecAlgebra.ofMsg, hh.kadd]

theorem Hom.kaddAll_kf (hh : Hom h) (cs : Array F256) (xs : Array α) :
    KV.map h (Flock.RecAlgebra.kaddAll ((cs.zip xs).toList.map fun fa => Flock.RecAlgebra.kf fa.1 fa.2)) =
      Flock.RecAlgebra.kaddAll ((cs.zip (xs.map h)).toList.map fun fa => Flock.RecAlgebra.kf fa.1 fa.2) := by
  rw [hh.kaddAll, List.map_map]
  congr 1
  rw [Array.toList_zip, Array.toList_zip, Array.toList_map, List.zip_map_right, List.map_map]
  apply List.map_congr_left
  intro fa _
  simp only [Function.comp, Prod.map, id, hh.kf]

theorem getElem!_map_of {γ δ : Type} [Inhabited γ] [Inhabited δ] (f : γ → δ) (hf : f default = default)
    (xs : Array γ) (i : Nat) : (xs.map f)[i]! = f xs[i]! := by
  simp only [getElem!_def, Array.getElem?_map]
  cases xs[i]? <;> simp [hf]

theorem getD_map (hh : Hom h) (xs : Array α) (i : Nat) : (xs.map h).getD i Val.zero = h (xs.getD i Val.zero) := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases xs[i]? <;> simp [hh.zero]

end Hom

/-! ## Computations and their images -/

/-- `y` is `x`'s image under `f`: when `x` succeeds, `y` succeeds with `f` of its result, the same coins drawn. -/
def Sim {A B : Type} (f : A → B) (x : CM A) (y : CM B) : Prop :=
  ∀ k src a k', x k src = .ok (a, k') → y k src = .ok (f a, k')

namespace Sim

variable {A B C D : Type}

theorem refl (x : CM A) : Sim id x x := fun _ _ _ _ hx => hx

theorem mono {f g : A → B} {x : CM A} {y : CM B} (hs : Sim f x y) (hfg : ∀ a, f a = g a) : Sim g x y := by
  intro k src a k' hx
  rw [hs k src a k' hx, hfg]

theorem pure' {f : A → B} {a : A} {b : B} (hab : f a = b) : Sim f (Pure.pure a) (Pure.pure b) := by
  intro k src a' k' hx
  have e : (Pure.pure a : CM A) k src = .ok (a, k) := rfl
  rw [e] at hx
  cases hx
  rw [hab]; rfl

theorem bind {f : A → B} {g : C → D} {x : CM A} {y : CM B} {x' : A → CM C} {y' : B → CM D}
    (hs : Sim f x y) (h' : ∀ a, Sim g (x' a) (y' (f a))) : Sim g (x >>= x') (y >>= y') := by
  intro k src c k' hx
  have e1 : (x >>= x') k src = (x k src).bind (fun p => x' p.1 p.2 src) := rfl
  have e2 : (y >>= y') k src = (y k src).bind (fun p => y' p.1 p.2 src) := rfl
  rw [e1] at hx
  rw [e2]
  cases hxk : x k src with
  | error e => rw [hxk] at hx; cases hx
  | ok p =>
    rw [hxk] at hx
    rw [hs k src p.1 p.2 hxk]
    exact h' p.1 p.2 src c k' hx

theorem bind_same {g : C → D} {x : CM A} {x' : A → CM C} {y' : A → CM D}
    (h' : ∀ a, Sim g (x' a) (y' a)) : Sim g (x >>= x') (x >>= y') :=
  bind (refl x) h'

theorem throw' {f : A → B} {y : CM B} (e : String) : Sim f (throw e) y := by
  intro k src a k' hx
  cases hx

theorem ite' {f : A → B} {c : Prop} [Decidable c] {x₁ x₂ : CM A} {y₁ y₂ : CM B} (h₁ : c → Sim f x₁ y₁)
    (h₂ : ¬c → Sim f x₂ y₂) : Sim f (if c then x₁ else x₂) (if c then y₁ else y₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

theorem liftE' {f : A → B} {e₁ : Except String A} {e₂ : Except String B}
    (he : ∀ a, e₁ = .ok a → e₂ = .ok (f a)) : Sim f (liftE e₁) (liftE e₂) := by
  cases e₁ with
  | error e => exact throw' e
  | ok a => rw [he a rfl]; exact pure' rfl

end Sim

/-! ## The maps on messages, points and Ligerito's state -/

section Maps

variable {α β : Type} [Val α] [Val β]

/-- The messages' image under `h`. -/
def Messages.map (h : α → β) (M : Messages α) : Messages β where
  ab := M.ab.map h
  c := M.c.map h
  zc := M.zc.map (KV.map h)
  fa := h M.fa
  fb := h M.fb
  lc := M.lc.map (KV.map h)
  zp := M.zp.map h
  rs := M.rs.map (·.map h)
  rsT := M.rsT.map (·.map h)
  ood := M.ood.map h
  msgs := M.msgs.map fun mm => (KV.map h mm.1, KV.map h mm.2)
  yr := M.yr.map h
  comb := M.comb.map h
  cons := fun i ea => (M.cons i ea).map h
  tAtom := M.tAtom.map h

def ZcPts.map (h : α → β) (p : ZcPts α) : ZcPts β := ⟨p.z, p.rho, p.r, h p.vc⟩

def LcPts.map (h : α → β) (p : LcPts α) : LcPts β := ⟨p.fold, h p.w, p.rPrime, p.rIn⟩

def AClaim.map (h : α → β) (c : AClaim α) : AClaim β := ⟨h c.value, c.skip, c.x⟩

def RsPts.map (h : α → β) (p : RsPts α) : RsPts β := ⟨h p.T, p.gamma, p.sws⟩

def Info.map (h : α → β) (i : Info α) : Info β := ⟨i.zc.map h, i.lc.map h, i.rs.map h, i.lg⟩

def LSt.map (h : α → β) (st : LSt α) : LSt β := ⟨KV.map h st.s, KV3.map h st.q, st.msg, st.ood, st.oods, st.conss⟩

end Maps

/-! ## Ligerito's state: computations and their images -/

/-- `y` is `x`'s image: from the image of a state, `y` succeeds with `f` of `x`'s result and `fs` of its state. -/
def SimL {α β A B : Type} (fs : LSt α → LSt β) (f : A → B) (x : LM α A) (y : LM β B) : Prop :=
  ∀ st, Sim (fun p => (f p.1, fs p.2)) (x st) (y (fs st))

namespace SimL

variable {α β A B C D : Type} {fs : LSt α → LSt β}

theorem bind {f : A → B} {g : C → D} {x : LM α A} {y : LM β B} {x' : A → LM α C} {y' : B → LM β D}
    (hs : SimL fs f x y) (h' : ∀ a, SimL fs g (x' a) (y' (f a))) : SimL fs g (x >>= x') (y >>= y') := by
  intro st
  exact Sim.bind (x' := fun p : A × LSt α => x' p.1 p.2) (y' := fun p : B × LSt β => y' p.1 p.2) (hs st)
    fun p => h' p.1 p.2

theorem lift {f : A → B} {x : CM A} {y : CM B} (hs : Sim f x y) :
    SimL fs f (StateT.lift x : LM α A) (StateT.lift y : LM β B) := by
  intro st
  exact Sim.bind hs fun a => Sim.pure' rfl

theorem bind_lift {g : C → D} {x : CM A} {x' : A → LM α C} {y' : A → LM β D}
    (h' : ∀ a, SimL fs g (x' a) (y' a)) :
    SimL fs g ((StateT.lift x : LM α A) >>= x') ((StateT.lift x : LM β A) >>= y') :=
  bind (lift (Sim.refl x)) h'

theorem pure' {f : A → B} {a : A} {b : B} (hab : f a = b) : SimL fs f (Pure.pure a) (Pure.pure b) := by
  intro st
  exact Sim.pure' (by rw [hab])

theorem throw' {f : A → B} {y : LM β B} (e : String) : SimL fs f (throw e) y := by
  intro st k src a k' hx
  cases hx

theorem ite' {f : A → B} {c : Prop} [Decidable c] {x₁ x₂ : LM α A} {y₁ y₂ : LM β B} (h₁ : c → SimL fs f x₁ y₁)
    (h₂ : ¬c → SimL fs f x₂ y₂) : SimL fs f (if c then x₁ else x₂) (if c then y₁ else y₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

theorem get' : SimL fs fs (get : LM α (LSt α)) (get : LM β (LSt β)) := by
  intro st
  exact Sim.pure' rfl

theorem set' {s : LSt α} {s' : LSt β} (hs : fs s = s') :
    SimL fs id (set s : LM α PUnit) (set s' : LM β PUnit) := by
  intro st
  exact Sim.pure' (by rw [hs])

theorem modify' {m : LSt α → LSt α} {m' : LSt β → LSt β} (hm : ∀ st, fs (m st) = m' (fs st)) :
    SimL fs id (modify m : LM α PUnit) (modify m' : LM β PUnit) := by
  intro st
  exact Sim.pure' (by rw [hm])

end SimL

/-! ## `run` and its image -/

section Run

variable {α β : Type} [Val α] [Val β] {h : α → β} (hh : Hom h)
include hh

theorem zcRounds_sim (r : Array F128) :
    ∀ (l : List (α × α)) (l' : List (β × β)) (i : Nat) (claim : α) (claim' : β) (rho : Array F128),
      l' = l.map (KV.map h) → claim' = h claim →
      Sim (fun p => (h p.1, p.2)) (zcRounds r l i claim rho) (zcRounds r l' i claim' rho)
  | [], _, _, _, _, _, rfl, rfl => Sim.pure' rfl
  | (g1, ginf) :: rest, _, i, claim, _, rho, rfl, rfl => by
    simp only [zcRounds, List.map_cons, KV.map]
    refine Sim.bind_same fun rr => ?_
    exact zcRounds_sim r rest _ (i + 1) _ _ (rho.push rr) rfl (by simp only [hh.add, hh.sm])

theorem zerocheck_sim (m : Nat) (M : Messages α) :
    Sim (fun p => (h p.1, p.2.map h)) (zerocheck m M) (zerocheck m (M.map h)) := by
  unfold zerocheck
  refine Sim.bind_same fun _ => Sim.bind_same fun r => Sim.bind_same fun z => ?_
  refine Sim.bind (zcRounds_sim hh _ M.zc.toList _ 0 _ _ #[] (by simp [Messages.map]) ?_) fun ⟨claim, rho⟩ => ?_
  · simp only [Messages.map, hh.add, hh.dotV, ← hh.zipWith_add]
  · refine Sim.bind (Sim.liftE' (f := h) fun a ha => hh.mv _ _ _ ha) fun ab => Sim.pure' ?_
    simp only [ZcPts.map, Messages.map, hh.add, hh.dotV]

theorem lcRounds_sim :
    ∀ (l : List (α × α)) (l' : List (β × β)) (claim : α) (claim' : β) (ts : Array F128),
      l' = l.map (KV.map h) → claim' = h claim →
      Sim (fun p => (h p.1, p.2)) (lcRounds l claim ts) (lcRounds l' claim' ts)
  | [], _, _, _, _, rfl, rfl => Sim.pure' rfl
  | (q1, qinf) :: rest, _, claim, _, ts, rfl, rfl => by
    simp only [lcRounds, List.map_cons, KV.map]
    refine Sim.bind_same fun t => ?_
    exact lcRounds_sim rest _ _ _ (ts.push t) rfl (by simp only [hh.add, hh.sm])

theorem combDot_hom : ∀ (l : List (α × α)) (acc a : α),
    combDot l acc = .ok a → combDot (l.map (KV.map h)) (h acc) = .ok (h a)
  | [], acc, a, hc => by cases hc; rfl
  | (cl, zl) :: rest, acc, a, hc => by
    simp only [combDot, bind, Except.bind] at hc
    split at hc
    · cases hc
    · rename_i x hx
      have := combDot_hom rest _ a hc
      simp only [List.map_cons, KV.map, combDot, bind, Except.bind, hh.mv _ _ _ hx]
      rw [← hh.add]; exact this

theorem lincheck_sim (kLog : Nat) (M : Messages α) (zc : ZcPts α) :
    Sim (fun p => (h p.1, p.2.map h)) (lincheck kLog M zc) (lincheck kLog (M.map h) (zc.map h)) := by
  unfold lincheck
  refine Sim.bind_same fun alpha => Sim.bind_same fun beta => ?_
  refine Sim.bind (lcRounds_sim hh M.lc.toList _ _ _ #[] (by simp [Messages.map]) ?_) fun ⟨claim, ts⟩ => ?_
  · simp only [Messages.map, hh.add, hh.sm, hh.cst]
  · refine Sim.bind (Sim.liftE' (f := h) fun a ha => ?_) fun acc => Sim.bind_same fun rPrime => Sim.pure' ?_
    · have := combDot_hom hh _ _ a ha
      rw [hh.zero, show (KV.map h : KV α → KV β) = Prod.map h h from rfl] at this
      simpa [Messages.map, Array.zip_map] using this
    · simp only [LcPts.map, ZcPts.map, Messages.map, hh.add, hh.dotV]

theorem claimsOf_map (m kLog : Nat) (zc : ZcPts α) (lc : LcPts α) (extras : Array Claim) :
    claimsOf m kLog (zc.map h) (lc.map h) extras = (claimsOf m kLog zc lc extras).map (AClaim.map h) := by
  simp only [claimsOf, ZcPts.map, LcPts.map, Array.map_append, Array.map_map]
  congr 1
  · simp [AClaim.map]
  · apply Array.map_congr_left
    intro e _
    simp only [Function.comp, AClaim.map, hh.cst]

theorem rsClaim_sim (M : Messages α) (k : Nat) (cl : AClaim α) :
    Sim (fun p => (h p.1, h p.2.1, p.2.2)) (rsClaim M k cl) (rsClaim (M.map h) k (cl.map h)) := by
  unfold rsClaim
  refine Sim.bind_same fun rpp => Sim.pure' ?_
  have e : ∀ (xs : Array (Array α)), (xs.map (·.map h))[k]! = (xs[k]!).map h :=
    fun xs => getElem!_map_of (fun a : Array α => a.map h) (by show Array.map h #[] = #[]; simp) xs k
  simp only [Messages.map, AClaim.map, e, hh.add, hh.dotV]

theorem rsClaims_sim (M : Messages α) : ∀ (cls : List (AClaim α)) (k : Nat),
    Sim (fun p => (p.1.map h, p.2.1.map h, p.2.2)) (rsClaims M cls k) (rsClaims (M.map h) (cls.map (AClaim.map h)) k)
  | [], _ => Sim.pure' (by simp)
  | cl :: rest, k => by
    simp only [rsClaims, List.map_cons]
    refine Sim.bind (rsClaim_sim hh M k cl) fun ⟨out, beta, rpp, z⟩ => ?_
    refine Sim.bind (rsClaims_sim M rest (k + 1)) fun ⟨outs, betas, pts⟩ => Sim.pure' ?_
    simp only [Array.map_append, List.map_toArray, List.map_cons, List.map_nil]

theorem ringSwitch_sim (M : Messages α) (claims : Array (AClaim α)) :
    Sim (fun p => (p.1.map h, p.2.map h)) (ringSwitch M claims) (ringSwitch (M.map h) (claims.map (AClaim.map h))) := by
  unfold ringSwitch
  simp only [Array.toList_map, Array.size_map]
  refine Sim.bind (rsClaims_sim hh M claims.toList 0) fun ⟨outs, betas, pts⟩ => ?_
  exact Sim.bind_same fun gamma => Sim.pure' (by simp only [RsPts.map, hh.dotV])

omit [Val α] [Val β] hh in
theorem nextMsg_sim (M : Messages α) :
    SimL (LSt.map h) (fun mm : KV α × KV α => (KV.map h mm.1, KV.map h mm.2)) (nextMsg M) (nextMsg (M.map h)) := by
  unfold nextMsg
  refine SimL.bind SimL.get' fun st => ?_
  simp only [Messages.map, LSt.map, Array.getElem?_map]
  cases M.msgs[st.msg]? with
  | none => exact SimL.throw' _
  | some mm => exact SimL.bind (SimL.set' rfl) fun _ => SimL.pure' rfl

omit [Val α] [Val β] hh in
theorem nextOod_sim (M : Messages α) : SimL (LSt.map h) h (nextOod M) (nextOod (M.map h)) := by
  unfold nextOod
  refine SimL.bind SimL.get' fun st => ?_
  simp only [Messages.map, LSt.map, Array.getElem?_map]
  cases M.ood[st.ood]? with
  | none => exact SimL.throw' _
  | some y => exact SimL.bind (SimL.set' rfl) fun _ => SimL.pure' rfl

theorem challenge_sim (M : Messages α) : SimL (LSt.map h) id (challenge M) (challenge (M.map h)) := by
  unfold challenge
  refine SimL.bind_lift fun rk => SimL.bind SimL.get' fun st => ?_
  obtain ⟨s, ⟨a0, a1, a2⟩, msg, ood, oods, conss⟩ := st
  refine SimL.bind (SimL.set' ?_) fun _ => SimL.bind (nextMsg_sim (h := h) M) fun mm =>
    SimL.bind (SimL.modify' fun st => ?_) fun _ => SimL.pure' rfl
  · simp only [LSt.map, KV3.map, hh.kadd, hh.kc]
  · simp only [LSt.map, hh.ofMsg]

theorem challenges_sim (M : Messages α) : ∀ n, SimL (LSt.map h) id (challenges M n) (challenges (M.map h) n)
  | 0 => SimL.pure' rfl
  | n + 1 => by
    simp only [challenges]
    exact SimL.bind (challenge_sim hh M) fun rk => SimL.bind (challenges_sim M n) fun rest => SimL.pure' rfl

theorem addScaled_sim (p : KV α × KV α × KV α) (p' : KV β × KV β × KV β) (b : F128) (hp : p' = KV3.map h p) :
    SimL (LSt.map h) id (addScaled p b) (addScaled p' b) := by
  subst hp
  exact SimL.modify' fun st => by simp only [LSt.map, KV3.map, hh.kadd, hh.fk]

theorem oodIntro_sim (M : Messages α) (dim split : Nat) :
    SimL (LSt.map h) id (oodIntro M dim split) (oodIntro (M.map h) dim split) := by
  unfold oodIntro
  refine SimL.bind_lift fun zz => SimL.bind (nextOod_sim (h := h) M) fun y =>
    SimL.bind (nextMsg_sim (h := h) M) fun mm =>
    SimL.bind_lift fun b => SimL.bind (addScaled_sim hh _ _ b ?_) fun _ => SimL.modify' fun st => ?_
  · rw [hh.ofMsg, hh.kofF]
  · simp only [LSt.map, hh.kadd, hh.kofF, hh.sm]

theorem query_sim (sch : Schedule) (M : Messages α) (i : Nat) (fold : Array F256) :
    SimL (LSt.map h) (fun p => (p.1, p.2.1, KV.map h p.2.2)) (query sch M i fold) (query sch (M.map h) i fold) := by
  unfold query
  refine SimL.bind_lift fun ps => SimL.bind_lift fun al => SimL.pure' ?_
  simp only [Messages.map, hh.kaddAll_kf]

theorem queryFold_sim (sch : Schedule) (M : Messages α) (i cols : Nat) (fold : Array F256) :
    SimL (LSt.map h) id (queryFold sch M i cols fold) (queryFold sch (M.map h) i cols fold) := by
  unfold queryFold
  refine SimL.bind (query_sim hh sch M i fold) fun ⟨pos, al, e⟩ => SimL.bind (nextMsg_sim (h := h) M) fun mm =>
    SimL.bind_lift fun b => SimL.bind (addScaled_sim hh _ _ b ?_) fun _ => SimL.modify' fun st => ?_
  · rw [hh.ofMsg]
  · simp only [LSt.map, hh.kadd, hh.fk]

theorem levelsLoop_sim (sch : Schedule) (M : Messages α) (R : Nat) :
    ∀ fuel i split levels ext, SimL (LSt.map h) id (levelsLoop sch M R fuel i split levels ext)
      (levelsLoop sch (M.map h) R fuel i split levels ext)
  | 0, _, _, _, _ => SimL.pure' rfl
  | fuel + 1, i, split, levels, ext => by
    simp only [levelsLoop]
    refine SimL.ite' (fun _ => SimL.throw' _) fun _ => ?_
    refine SimL.bind (challenges_sim hh M 4) fun chs => SimL.ite' (fun _ => ?_) fun _ => ?_
    · exact SimL.bind (oodIntro_sim hh M _ _) fun _ => SimL.bind (oodIntro_sim hh M _ _) fun _ =>
        SimL.bind (queryFold_sim hh sch M _ _ chs) fun _ => levelsLoop_sim sch M R fuel _ _ _ _
    · refine SimL.bind (query_sim hh sch M R chs) fun ⟨pos, al, e⟩ => SimL.bind_lift fun b =>
        SimL.bind (SimL.modify' fun st => ?_) fun _ => levelsLoop_sim sch M R fuel _ _ _ _
      simp only [LSt.map, hh.kadd, hh.fk]

theorem ligeritoRun_sim (sch : Schedule) (M : Messages α) :
    SimL (LSt.map h) id (ligeritoRun sch M) (ligeritoRun sch (M.map h)) := by
  unfold ligeritoRun
  refine SimL.bind_lift fun zz => SimL.bind (nextOod_sim (h := h) M) fun y => SimL.bind_lift fun b =>
    SimL.bind (SimL.modify' fun st => ?_) fun _ => SimL.bind (nextMsg_sim (h := h) M) fun mm =>
    SimL.bind (SimL.modify' fun st => ?_) fun _ => SimL.bind (challenges_sim hh M _) fun initCh =>
    SimL.bind (oodIntro_sim hh M _ _) fun _ => SimL.bind (oodIntro_sim hh M _ _) fun _ =>
    SimL.bind (queryFold_sim hh sch M _ _ _) fun _ =>
    SimL.bind (levelsLoop_sim hh sch M _ _ _ _ _ _) fun _ => SimL.pure' rfl
  · simp only [LSt.map, hh.kadd, hh.kofF, hh.sm]
  · simp only [LSt.map, hh.ofMsg]

theorem ligerito_sim (sch : Schedule) (M : Messages α) (tIn : α) (sws : Array Switched) :
    Sim (fun p => (p.1.map h, p.2)) (ligerito sch M tIn sws) (ligerito sch (M.map h) (h tIn) sws) := by
  unfold ligerito
  have h0 : LSt.map h { s := kofF tIn, q := ((Val.zero, Val.zero), (Val.zero, Val.zero), (Val.zero, Val.zero)) } =
      { s := kofF (h tIn), q := ((Val.zero, Val.zero), (Val.zero, Val.zero), (Val.zero, Val.zero)) } := by
    simp only [LSt.map, KV3.map, hh.kofF]
    simp only [KV.map, hh.zero]
  refine Sim.bind (f := fun p => (p.1, LSt.map h p.2)) ?_ fun ⟨⟨initCh, levels, ext⟩, st⟩ => ?_
  · have := ligeritoRun_sim hh sch M { s := kofF tIn, q := ((Val.zero, Val.zero), (Val.zero, Val.zero),
      (Val.zero, Val.zero)) }
    rw [h0] at this
    exact this
  · simp only [LSt.map, Messages.map, Array.size_map]
    refine Sim.bind_same fun src => Sim.bind_same fun k => Sim.ite' (fun _ => Sim.throw' _) fun _ => ?_
    refine Sim.ite' (fun _ => Sim.throw' _) fun _ => Sim.bind_same fun res => ?_
    refine Sim.ite' (fun _ => Sim.throw' _) fun _ => Sim.pure' ?_
    have hl : KV.map h (kaddAll ((Array.range res.size).toList.map fun j =>
        kc res[j]! (M.yr.getD (2 * j) Val.zero, M.yr.getD (2 * j + 1) Val.zero))) =
        kaddAll ((Array.range res.size).toList.map fun j =>
          kc res[j]! ((M.yr.map h).getD (2 * j) Val.zero, (M.yr.map h).getD (2 * j + 1) Val.zero)) := by
      rw [hh.kaddAll, List.map_map]
      congr 1
      apply List.map_congr_left
      intro j _
      simp only [Function.comp]
      rw [hh.kc]
      simp only [KV.map, getD_map hh]
    rw [← hl, ← hh.kadd]
    simp [KV.map]

theorem run_sim (sh : Shape) (M : Messages α) (extras : Array Claim) :
    Sim (fun p => (p.1.map h, p.2.map h)) (run sh M extras) (run sh (M.map h) extras) := by
  unfold run
  refine Sim.ite' (fun _ => Sim.throw' _) fun _ => ?_
  refine Sim.bind (zerocheck_sim hh _ M) fun ⟨r0, zc⟩ => Sim.bind (lincheck_sim hh _ M zc) fun ⟨r1, lc⟩ => ?_
  simp only [claimsOf_map hh]
  refine Sim.bind (ringSwitch_sim hh M _) fun ⟨rsr, rs⟩ => ?_
  have ht : (M.map h).tAtom.getD (rs.map h).T = h (M.tAtom.getD rs.T) := by
    simp only [Messages.map, RsPts.map]
    cases M.tAtom <;> rfl
  simp only [ht]
  refine Sim.bind (ligerito_sim hh _ M _ _) fun ⟨lgr, lg⟩ => Sim.pure' ?_
  simp [Info.map, RsPts.map, hh.add]

end Run

end Flock.RecAlgebra
