import Proofs.Flock.VBridge.Algebra.Rows
import Proofs.Flock.VBridge.Algebra.Ligerito

/-!
# `run`'s residuals have one skeleton at every coin source

`run` (RecAlgebra.lean:532–539) over the symbolic messages draws coins only to scale values (`sm`), to weigh them
(`dotV`, `kc`, `kf`, `fk`) and to fill the coin-derived points; it branches on no coin's value. The keys a residual
holds, whether it has a constant and its products depend on the coins only through how many are drawn, which is the
same at any two sources that hand out rounds of the sizes asked (`Sized`). So the residuals at the zero coins and at a
rep's coins have one skeleton whenever both runs succeed (`residualLins_skel`), and `verifierRows` returns whenever
`structureOf` does and `run` succeeds at the rep's coins (`verifierRows_ok`).

The values related are linear (no products) except the zerocheck's and the lincheck's residuals, whose products are
`mv` of message elements and so the same at both sources.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine
open Val (zero add sm mv cst)

/-! ## Linear values of one skeleton -/

/-- Two linear values (no products) with the same elements in the same order and a constant alike present. -/
def Lin.LSk (a a' : Lin) : Prop :=
  a.t.map Prod.fst = a'.t.map Prod.fst ∧ a.c.isSome = a'.c.isSome ∧ a.q = [] ∧ a'.q = []

theorem addAt_keys {κ : Type} [DecidableEq κ] (k : κ) (v v' : F128) :
    ∀ {t t' : Terms κ}, t.map Prod.fst = t'.map Prod.fst → (addAt t k v).map Prod.fst = (addAt t' k v').map Prod.fst
  | [], [], _ => rfl
  | (a, x) :: r, (a', x') :: r', h => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨rfl, h⟩ := h
    by_cases hk : a = k <;> simp [addAt, hk, h, addAt_keys k v v' h]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

theorem mergeT_keys {κ : Type} [DecidableEq κ] :
    ∀ {b b' a a' : Terms κ}, a.map Prod.fst = a'.map Prod.fst → b.map Prod.fst = b'.map Prod.fst →
      (mergeT a b).map Prod.fst = (mergeT a' b').map Prod.fst
  | [], [], _, _, ha, _ => ha
  | (k, x) :: r, (k', x') :: r', a, a', ha, hb => by
    simp only [List.map_cons, List.cons.injEq] at hb
    obtain ⟨rfl, hb⟩ := hb
    exact mergeT_keys (b := r) (b' := r') (addAt_keys k x x' ha) hb
  | [], _ :: _, _, _, _, h => by simp at h
  | _ :: _, [], _, _, _, h => by simp at h

theorem xor_isSome (a b : Lin) : (Lin.xor a b).c.isSome = (a.c.isSome || b.c.isSome) := by
  simp only [Lin.xor]
  cases a.c <;> cases b.c <;> rfl

namespace Lin.LSk

theorem refl {a : Lin} (h : a.q = []) : Lin.LSk a a := ⟨rfl, rfl, h, h⟩

theorem zero' : Lin.LSk (zero : Lin) (zero : Lin) := ⟨rfl, rfl, rfl, rfl⟩

theorem cst' {v v' : F128} : Lin.LSk (cst v : Lin) (cst v' : Lin) := ⟨rfl, rfl, rfl, rfl⟩

theorem add' {a a' b b' : Lin} (ha : Lin.LSk a a') (hb : Lin.LSk b b') :
    Lin.LSk (add a b : Lin) (add a' b' : Lin) := by
  show Lin.LSk (Lin.xor a b) (Lin.xor a' b')
  refine ⟨mergeT_keys ha.1 hb.1, by rw [xor_isSome, xor_isSome, ha.2.1, hb.2.1], ?_, ?_⟩
  · show mergeT a.q b.q = []
    rw [ha.2.2.1, hb.2.2.1]; rfl
  · show mergeT a'.q b'.q = []
    rw [ha.2.2.2, hb.2.2.2]; rfl

theorem sm' {s s' : F128} {a a' : Lin} (ha : Lin.LSk a a') : Lin.LSk (sm s a : Lin) (sm s' a' : Lin) := by
  show Lin.LSk (Lin.scale s a) (Lin.scale s' a')
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [Lin.scale, List.map_map]
    exact ha.1
  · simp only [Lin.scale, Option.isSome_map]
    exact ha.2.1
  · simp [Lin.scale, ha.2.2.1]
  · simp [Lin.scale, ha.2.2.2]

theorem skel {a a' : Lin} (h : Lin.LSk a a') : Lin.Skel a a' := ⟨h.1, h.2.1, h.2.2.1.trans h.2.2.2.symm⟩

end Lin.LSk

theorem Lin.Skel.refl' (a : Lin) : Lin.Skel a a := ⟨rfl, rfl, rfl⟩

theorem Lin.Skel.add' {a a' b b' : Lin} (ha : Lin.Skel a a') (hb : Lin.Skel b b') :
    Lin.Skel (add a b : Lin) (add a' b' : Lin) := by
  show Lin.Skel (Lin.xor a b) (Lin.xor a' b')
  refine ⟨mergeT_keys ha.1 hb.1, by rw [xor_isSome, xor_isSome, ha.2.1, hb.2.1], ?_⟩
  show mergeT a.q b.q = mergeT a'.q b'.q
  rw [ha.2.2, hb.2.2]

/-- Two K values of one linear skeleton. -/
def KSk (x x' : KV Lin) : Prop := Lin.LSk x.1 x'.1 ∧ Lin.LSk x.2 x'.2

/-- Two round polynomials of one linear skeleton. -/
def K3Sk (p p' : KV Lin × KV Lin × KV Lin) : Prop := KSk p.1 p'.1 ∧ KSk p.2.1 p'.2.1 ∧ KSk p.2.2 p'.2.2

/-- Two messages of one linear skeleton. -/
def MSk (mm mm' : KV Lin × KV Lin) : Prop := KSk mm.1 mm'.1 ∧ KSk mm.2 mm'.2

namespace KSk

theorem zero' : KSk ((zero, zero) : KV Lin) (zero, zero) := ⟨Lin.LSk.zero', Lin.LSk.zero'⟩

theorem kadd' {x x' y y' : KV Lin} (hx : KSk x x') (hy : KSk y y') : KSk (kadd x y) (kadd x' y') :=
  ⟨Lin.LSk.add' hx.1 hy.1, Lin.LSk.add' hx.2 hy.2⟩

theorem kc' {c c' : F256} {x x' : KV Lin} (hx : KSk x x') : KSk (kc c x) (kc c' x') :=
  ⟨Lin.LSk.add' (Lin.LSk.sm' hx.1) (Lin.LSk.sm' hx.2), Lin.LSk.add' (Lin.LSk.sm' hx.1) (Lin.LSk.sm' hx.2)⟩

theorem kf' {c c' : F256} {y y' : Lin} (hy : Lin.LSk y y') : KSk (kf c y) (kf c' y') :=
  ⟨Lin.LSk.sm' hy, Lin.LSk.sm' hy⟩

theorem fk' {b b' : F128} {x x' : KV Lin} (hx : KSk x x') : KSk (fk b x) (fk b' x') :=
  ⟨Lin.LSk.sm' hx.1, Lin.LSk.sm' hx.2⟩

theorem kofF' {y y' : Lin} (hy : Lin.LSk y y') : KSk (kofF y) (kofF y') := ⟨hy, Lin.LSk.zero'⟩

end KSk

theorem ofMsg_sk {mm mm' : KV Lin × KV Lin} {s s' : KV Lin} (hm : MSk mm mm') (hs : KSk s s') :
    K3Sk (ofMsg mm s) (ofMsg mm' s') :=
  ⟨hm.1, KSk.kadd' hs hm.2, hm.2⟩

/-! ## Lists of related values -/

section Lists

variable {γ δ ε ζ : Type}

theorem forall₂_foldl {Rl : γ → δ → Prop} {Ra : ε → ζ → Prop} {f : ε → γ → ε} {g : ζ → δ → ζ}
    (hf : ∀ a a' x x', Ra a a' → Rl x x' → Ra (f a x) (g a' x')) {l : List γ} {l' : List δ}
    (hl : List.Forall₂ Rl l l') : ∀ {a a'}, Ra a a' → Ra (l.foldl f a) (l'.foldl g a') := by
  induction hl with
  | nil => exact fun h => h
  | cons hx _ ih => exact fun h => ih (hf _ _ _ _ h hx)

theorem forall₂_map {R : γ → δ → Prop} {S : ε → ζ → Prop} {f : γ → ε} {g : δ → ζ}
    (hfg : ∀ a b, R a b → S (f a) (g b)) {l : List γ} {l' : List δ} (hl : List.Forall₂ R l l') :
    List.Forall₂ S (l.map f) (l'.map g) := by
  induction hl with
  | nil => exact .nil
  | cons hx _ ih => exact .cons (hfg _ _ hx) ih

theorem forall₂_map_same {S : ε → ζ → Prop} {f : γ → ε} {g : γ → ζ} (hfg : ∀ a, S (f a) (g a)) :
    ∀ l : List γ, List.Forall₂ S (l.map f) (l.map g)
  | [] => .nil
  | a :: l => .cons (hfg a) (forall₂_map_same hfg l)

theorem forall₂_append {R : γ → δ → Prop} {l₁ l₂ : List γ} {l₁' l₂' : List δ} (h₁ : List.Forall₂ R l₁ l₁')
    (h₂ : List.Forall₂ R l₂ l₂') : List.Forall₂ R (l₁ ++ l₂) (l₁' ++ l₂') := by
  induction h₁ with
  | nil => exact h₂
  | cons hx _ ih => exact .cons hx ih

theorem forall₂_mono {R S : γ → δ → Prop} (hRS : ∀ a b, R a b → S a b) {l : List γ} {l' : List δ}
    (h : List.Forall₂ R l l') : List.Forall₂ S l l' := by
  induction h with
  | nil => exact .nil
  | cons hx _ ih => exact .cons (hRS _ _ hx) ih

theorem forall₂_length {R : γ → δ → Prop} {l : List γ} {l' : List δ} (h : List.Forall₂ R l l') :
    l.length = l'.length := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]

theorem forall₂_zip {R : ε → ζ → Prop} :
    ∀ {cs : List γ} {cs' : List δ} {xs : List ε} {xs' : List ζ}, cs.length = cs'.length → List.Forall₂ R xs xs' →
      List.Forall₂ (fun p p' => R p.2 p'.2) (cs.zip xs) (cs'.zip xs')
  | [], [], _, _, _, _ => .nil
  | _ :: _, _ :: _, _, _, _, .nil => .nil
  | _ :: _, _ :: _, _ :: _, _ :: _, h, .cons hx hl => .cons hx (forall₂_zip (by simpa using h) hl)
  | [], _ :: _, _, _, h, _ => by simp at h
  | _ :: _, [], _, _, h, _ => by simp at h

end Lists

theorem forall₂_self {l : List Lin} (h : ∀ x ∈ l, x.q = []) : List.Forall₂ Lin.LSk l l := by
  induction l with
  | nil => exact .nil
  | cons a l ih =>
    exact .cons (Lin.LSk.refl (h a (by simp))) (ih fun x hx => h x (by simp [hx]))

theorem arr_self {xs : Array Lin} (h : ∀ x ∈ xs, x.q = []) : List.Forall₂ Lin.LSk xs.toList xs.toList :=
  forall₂_self fun x hx => h x (by simpa using hx)

theorem forall₂_zipWith :
    ∀ {l₁ l₁' l₂ l₂' : List Lin}, List.Forall₂ Lin.LSk l₁ l₁' → List.Forall₂ Lin.LSk l₂ l₂' →
      List.Forall₂ Lin.LSk (List.zipWith add l₁ l₂) (List.zipWith add l₁' l₂')
  | _, _, _, _, .nil, _ => .nil
  | _, _, _, _, .cons _ _, .nil => .nil
  | _, _, _, _, .cons ha hl, .cons hb hl' => .cons (Lin.LSk.add' ha hb) (forall₂_zipWith hl hl')

theorem dotV_sk {cs cs' : Array F128} {xs xs' : Array Lin} (hc : cs.size = cs'.size)
    (hx : List.Forall₂ Lin.LSk xs.toList xs'.toList) : Lin.LSk (dotV cs xs) (dotV cs' xs') := by
  unfold dotV
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_zip, Array.toList_zip]
  exact forall₂_foldl (Rl := fun p p' : F128 × Lin => Lin.LSk p.2 p'.2) (Ra := Lin.LSk)
    (fun a a' x x' ha hx => Lin.LSk.add' ha (Lin.LSk.sm' hx))
    (forall₂_zip (cs := cs.toList) (cs' := cs'.toList) (by rw [Array.length_toList, Array.length_toList]; exact hc)
      hx) Lin.LSk.zero'

theorem kaddAll_sk {xs xs' : List (KV Lin)} (h : List.Forall₂ KSk xs xs') : KSk (kaddAll xs) (kaddAll xs') := by
  unfold kaddAll
  exact forall₂_foldl (Rl := KSk) (Ra := KSk) (fun _ _ _ _ ha hx => KSk.kadd' ha hx) h KSk.zero'

theorem kaddAll_kf_sk {fs fs' : Array F256} {xs xs' : Array Lin} (hf : fs.size = fs'.size)
    (hx : List.Forall₂ Lin.LSk xs.toList xs'.toList) :
    KSk (kaddAll ((fs.zip xs).toList.map fun fa => kf fa.1 fa.2))
      (kaddAll ((fs'.zip xs').toList.map fun fa => kf fa.1 fa.2)) := by
  rw [Array.toList_zip, Array.toList_zip]
  exact kaddAll_sk (forall₂_map (fun _ _ h => KSk.kf' h)
    (forall₂_zip (cs := fs.toList) (cs' := fs'.toList) (by rw [Array.length_toList, Array.length_toList]; exact hf)
      hx))

theorem sz_eqTable (r : Array F128) : (eqTable r).size = 2 ^ r.size := by
  have h : ∀ (l : List F128) (t : Array F128), (l.foldl eqStep t).size = t.size * 2 ^ l.length := by
    intro l
    induction l with
    | nil => intro t; simp
    | cons a l ih =>
      intro t
      rw [List.foldl_cons, ih, List.length_cons, pow_succ]
      simp only [eqStep, Array.size_append, Array.size_map]
      ring
  unfold eqTable
  rw [← Array.foldl_toList, h, Array.length_toList]
  simp

theorem sz_eqTableK (r : Array F256) : (eqTableK r).size = 2 ^ r.size := by
  rw [eqTableK_foldl, ← Array.foldl_toList, foldl_eqStepK_size]
  simp

/-! ## The symbolic messages are linear -/

/-- Messages whose elements are linear, with consistency sums that do not depend on `eq(α)`. -/
structure MLin (M : Messages Lin) : Prop where
  ab : ∀ x ∈ M.ab, x.q = []
  c : ∀ x ∈ M.c, x.q = []
  zc : ∀ p ∈ M.zc, p.1.q = [] ∧ p.2.q = []
  fa : M.fa.q = []
  fb : M.fb.q = []
  lc : ∀ p ∈ M.lc, p.1.q = [] ∧ p.2.q = []
  zp : ∀ x ∈ M.zp, x.q = []
  rs : ∀ k : Nat, ∀ x ∈ M.rs[k]!, x.q = []
  rsT : ∀ k : Nat, ∀ x ∈ M.rsT[k]!, x.q = []
  ood : ∀ x ∈ M.ood, x.q = []
  msgs : ∀ mm ∈ M.msgs, MSk mm mm
  yr : ∀ x ∈ M.yr, x.q = []
  cons : ∀ i ea, ∀ x ∈ M.cons i ea, x.q = []
  consEq : ∀ i ea ea', M.cons i ea = M.cons i ea'
  tAtom : ∀ x, M.tAtom = some x → x.q = []

theorem range_map_lin {n : Nat} {g : Nat → Lin} (hg : ∀ i, (g i).q = []) :
    ∀ x ∈ (Array.range n).map g, x.q = [] := by
  intro x hx
  rw [Array.mem_map] at hx
  obtain ⟨i, -, rfl⟩ := hx
  exact hg i

theorem range_map_lin₂ {n : Nat} {g : Nat → Lin × Lin} (hg : ∀ i, (g i).1.q = [] ∧ (g i).2.q = []) :
    ∀ p ∈ (Array.range n).map g, p.1.q = [] ∧ p.2.q = [] := by
  intro p hp
  rw [Array.mem_map] at hp
  obtain ⟨i, -, rfl⟩ := hp
  exact hg i

theorem getElem!_lin {xs : Array (Array Lin)} (h : ∀ ys ∈ xs, ∀ x ∈ ys, x.q = []) :
    ∀ k : Nat, ∀ x ∈ xs[k]!, x.q = [] := by
  intro k x hx
  by_cases hk : k < xs.size
  · rw [getElem!_pos xs k hk] at hx
    exact h _ (Array.getElem_mem hk) x hx
  · rw [getElem!_neg xs k hk] at hx
    change x ∈ (#[] : Array Lin) at hx
    simp at hx

theorem symbolic_mlin (sh : Shape) : MLin (symbolic sh) where
  ab := range_map_lin fun _ => rfl
  c := range_map_lin fun _ => rfl
  zc := range_map_lin₂ fun _ => ⟨rfl, rfl⟩
  fa := rfl
  fb := rfl
  lc := range_map_lin₂ fun _ => ⟨rfl, rfl⟩
  zp := range_map_lin fun _ => rfl
  rs := getElem!_lin fun ys hys => by
    simp only [symbolic, Array.mem_map, Array.mem_range] at hys
    obtain ⟨k, -, rfl⟩ := hys
    exact range_map_lin fun _ => rfl
  rsT := getElem!_lin fun ys hys => by
    simp only [symbolic, Array.mem_map, Array.mem_range] at hys
    obtain ⟨k, -, rfl⟩ := hys
    exact range_map_lin fun _ => rfl
  ood := range_map_lin fun _ => rfl
  msgs := by
    intro mm hmm
    simp only [symbolic, Array.mem_map, Array.mem_range] at hmm
    obtain ⟨i, -, rfl⟩ := hmm
    exact ⟨⟨Lin.LSk.refl rfl, Lin.LSk.refl rfl⟩, ⟨Lin.LSk.refl rfl, Lin.LSk.refl rfl⟩⟩
  yr := range_map_lin fun _ => rfl
  cons := fun _ _ => range_map_lin fun _ => rfl
  consEq := fun _ _ _ => rfl
  tAtom := by
    intro x hx
    simp only [symbolic, Option.some.injEq] at hx
    subst hx
    rfl

theorem mem_of_get? {γ : Type} {xs : Array γ} {i : Nat} {a : γ} (h : xs[i]? = some a) : a ∈ xs := by
  obtain ⟨hi, rfl⟩ := Array.getElem?_eq_some_iff.mp h
  exact Array.getElem_mem hi

/-! ## Two runs at two coin sources -/

/-- A coin source that hands out rounds of the sizes asked. -/
def Sized (src : Coins) : Prop := ∀ k n r, src.round k n = .ok r → r.size = n

theorem sized_zeros : Sized Coins.zeros := by
  intro k n r h
  cases h
  simp

theorem sized_ofRounds (rounds : Array (Array F128)) : Sized (Coins.ofRounds rounds) := by
  intro k n r h
  simp only [Coins.ofRounds] at h
  split at h
  · split at h
    · cases h; assumption
    · cases h
  · cases h

/-- `x` and `y` at two sized coin sources: whenever both succeed, their results are related. -/
def RS {A B : Type} (R : A → B → Prop) (x : CM A) (y : CM B) : Prop :=
  ∀ (src src' : Coins) (k k' : Nat) a b (j j' : Nat), Sized src → Sized src' → x k src = .ok (a, j) →
    y k' src' = .ok (b, j') → R a b

namespace RS

variable {A B C D : Type}

theorem triv {x : CM A} {y : CM B} : RS (fun _ _ => True) x y := fun _ _ _ _ _ _ _ _ _ _ _ _ => trivial

theorem bind {R : A → B → Prop} {R' : C → D → Prop} {x : CM A} {y : CM B} {f : A → CM C} {g : B → CM D}
    (hs : RS R x y) (h : ∀ a b, R a b → RS R' (f a) (g b)) : RS R' (x >>= f) (y >>= g) := by
  intro src src' k k' c d j j' hz hz' hx hy
  obtain ⟨a, k₁, hx₁, hx₂⟩ := cm_bind hx
  obtain ⟨b, k₁', hy₁, hy₂⟩ := cm_bind hy
  exact h a b (hs src src' k k' a b k₁ k₁' hz hz' hx₁ hy₁) src src' k₁ k₁' c d j j' hz hz' hx₂ hy₂

theorem pure' {R : A → B → Prop} {a : A} {b : B} (h : R a b) : RS R (Pure.pure a) (Pure.pure b) := by
  intro src src' k k' a' b' j j' _ _ hx hy
  cases cm_pure hx
  cases cm_pure hy
  exact h

theorem liftE' {R : A → B → Prop} {e : Except String A} {e' : Except String B}
    (h : ∀ a b, e = .ok a → e' = .ok b → R a b) : RS R (liftE e) (liftE e') := by
  intro src src' k k' a b j j' _ _ hx hy
  obtain ⟨a₁, ha, hr⟩ := liftE_ok hx
  obtain ⟨b₁, hb, hr'⟩ := liftE_ok hy
  cases hr
  cases hr'
  exact h _ _ ha hb

theorem guard {R : A → B → Prop} {c c' : Prop} [Decidable c] [Decidable c'] {x₁ x₂ : CM A} {y₁ y₂ : CM B}
    (hx : ∀ k src p, x₁ k src ≠ .ok p) (hy : ∀ k src p, y₁ k src ≠ .ok p) (h : ¬c → ¬c' → RS R x₂ y₂) :
    RS R (if c then x₁ else x₂) (if c' then y₁ else y₂) := by
  intro src src' k k' a b j j' hz hz' hxo hyo
  by_cases hc : c
  · simp only [hc, ↓reduceIte] at hxo
    exact absurd hxo (hx _ _ _)
  by_cases hc' : c'
  · simp only [hc', ↓reduceIte] at hyo
    exact absurd hyo (hy _ _ _)
  simp only [hc, hc', ↓reduceIte] at hxo hyo
  exact h hc hc' src src' k k' a b j j' hz hz' hxo hyo

end RS

theorem coinVec_rs (n : Nat) : RS (fun v v' : Array F128 => v.size = v'.size) (coinVec n) (coinVec n) := by
  intro src src' k k' a b j j' hz hz' hx hy
  have e : ∀ {src : Coins} {k j : Nat} {v : Array F128}, Sized src → coinVec n k src = .ok (v, j) → v.size = n := by
    intro src k j v hs h
    rcases coinVec_ok h with ⟨rfl, rfl, -⟩ | ⟨-, hr, -⟩
    · rfl
    · exact hs _ _ _ hr
  rw [e hz hx, e hz' hy]

/-! ## Ligerito's state at two coin sources -/

/-- Two Ligerito states alike: running sums and round polynomials of one linear skeleton, as many messages and OOD
values read. -/
structure SR (st st' : LSt Lin) : Prop where
  s : KSk st.s st'.s
  q : K3Sk st.q st'.q
  msg : st.msg = st'.msg
  ood : st.ood = st'.ood

/-- `x` and `y` from alike states at two sized coin sources: whenever both succeed, related results and alike
states. -/
def RSL {A B : Type} (R : A → B → Prop) (x : LM Lin A) (y : LM Lin B) : Prop :=
  ∀ st st', SR st st' → RS (fun p p' => R p.1 p'.1 ∧ SR p.2 p'.2) (x st) (y st')

namespace RSL

variable {A B C D : Type}

theorem bind {R : A → B → Prop} {R' : C → D → Prop} {x : LM Lin A} {y : LM Lin B} {f : A → LM Lin C}
    {g : B → LM Lin D} (hs : RSL R x y) (h : ∀ a b, R a b → RSL R' (f a) (g b)) : RSL R' (x >>= f) (y >>= g) := by
  intro st st' hst
  exact RS.bind (f := fun p : A × LSt Lin => f p.1 p.2) (g := fun p : B × LSt Lin => g p.1 p.2) (hs st st' hst)
    fun p p' hp => h p.1 p'.1 hp.1 p.2 p'.2 hp.2

theorem lift {R : A → B → Prop} {x : CM A} {y : CM B} (hs : RS R x y) :
    RSL R (StateT.lift x : LM Lin A) (StateT.lift y : LM Lin B) := by
  intro st st' hst
  exact RS.bind hs fun a b hab => RS.pure' ⟨hab, hst⟩

theorem liftT {x : CM A} {y : CM B} : RSL (fun _ _ => True) (StateT.lift x : LM Lin A) (StateT.lift y : LM Lin B) :=
  lift RS.triv

theorem pure' {R : A → B → Prop} {a : A} {b : B} (h : R a b) : RSL R (Pure.pure a) (Pure.pure b) :=
  fun _ _ hst => RS.pure' ⟨h, hst⟩

theorem throw' {R : A → B → Prop} {y : LM Lin B} (e : String) : RSL R (throw e) y := by
  intro st st' _ src src' k k' a b j j' _ _ hx _
  cases hx

theorem ite' {R : A → B → Prop} {c : Prop} [Decidable c] {x₁ x₂ : LM Lin A} {y₁ y₂ : LM Lin B}
    (h₁ : c → RSL R x₁ y₁) (h₂ : ¬c → RSL R x₂ y₂) : RSL R (if c then x₁ else x₂) (if c then y₁ else y₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

theorem get' : RSL SR (get : LM Lin (LSt Lin)) (get : LM Lin (LSt Lin)) := fun _ _ hst => RS.pure' ⟨hst, hst⟩

theorem set' {s s' : LSt Lin} (h : SR s s') :
    RSL (fun _ _ => True) (set s : LM Lin PUnit) (set s' : LM Lin PUnit) :=
  fun _ _ _ => RS.pure' ⟨trivial, h⟩

theorem modify' {m m' : LSt Lin → LSt Lin} (hm : ∀ st st', SR st st' → SR (m st) (m' st')) :
    RSL (fun _ _ => True) (modify m : LM Lin PUnit) (modify m' : LM Lin PUnit) :=
  fun st st' hst => RS.pure' ⟨trivial, hm st st' hst⟩

end RSL

/-! ## `run`'s stages at two coin sources -/

section Run

variable {M : Messages Lin} (hM : MLin M)
include hM

omit hM in
theorem zcRounds_rs (r r' : Array F128) :
    ∀ (l : List (Lin × Lin)) (i : Nat) (claim claim' : Lin) (rho rho' : Array F128),
      (∀ p ∈ l, p.1.q = [] ∧ p.2.q = []) → Lin.LSk claim claim' →
      RS (fun p p' => Lin.LSk p.1 p'.1) (zcRounds r l i claim rho) (zcRounds r' l i claim' rho')
  | [], _, _, _, _, _, _, hc => RS.pure' hc
  | (g1, ginf) :: rest, i, claim, claim', rho, rho', hl, hc => by
    have hg := hl (g1, ginf) (by simp)
    simp only [zcRounds]
    refine RS.bind RS.triv fun rr rr' _ => ?_
    exact zcRounds_rs r r' rest (i + 1) _ _ _ _ (fun p hp => hl p (by simp [hp]))
      (Lin.LSk.add' (Lin.LSk.add' (Lin.LSk.sm' (Lin.LSk.sm' (Lin.LSk.add' hc (Lin.LSk.sm' (Lin.LSk.refl hg.1)))))
        (Lin.LSk.sm' (Lin.LSk.refl hg.1))) (Lin.LSk.sm' (Lin.LSk.refl hg.2)))

theorem zerocheck_rs (m : Nat) :
    RS (fun p p' => Lin.Skel p.1 p'.1 ∧ Lin.LSk p.2.vc p'.2.vc) (zerocheck m M) (zerocheck m M) := by
  unfold zerocheck
  refine RS.bind RS.triv fun _ _ _ => RS.bind RS.triv fun _ _ _ => RS.bind RS.triv fun z z' _ => ?_
  have hpc : Lin.LSk (dotV (lagrange nodesL den6Inv z) M.c) (dotV (lagrange nodesL den6Inv z') M.c) :=
    dotV_sk (by rw [lagrange_size, lagrange_size]) (arr_self hM.c)
  have hab : List.Forall₂ Lin.LSk (M.ab.zipWith add M.c).toList (M.ab.zipWith add M.c).toList := by
    rw [Array.toList_zipWith]
    exact forall₂_zipWith (arr_self hM.ab) (arr_self hM.c)
  refine RS.bind (zcRounds_rs _ _ M.zc.toList 0 _ _ #[] #[] (fun p hp => hM.zc p (by simpa using hp))
    (Lin.LSk.add' (dotV_sk (by rw [combWeights_size, combWeights_size]) hab) hpc))
    fun ⟨claim, rho⟩ ⟨claim', rho'⟩ hc => ?_
  refine RS.bind (RS.liftE' (R := fun a b => a = b) fun a b ha hb => Except.ok.inj (ha.symm.trans hb))
    fun ab ab' (he : ab = ab') => RS.pure' ⟨?_, hpc⟩
  subst he
  exact Lin.Skel.add' (Lin.LSk.skel hc) (Lin.Skel.refl' ab)

omit hM in
theorem lcRounds_rs :
    ∀ (l : List (Lin × Lin)) (claim claim' : Lin) (ts ts' : Array F128),
      (∀ p ∈ l, p.1.q = [] ∧ p.2.q = []) → Lin.LSk claim claim' →
      RS (fun p p' => Lin.LSk p.1 p'.1) (lcRounds l claim ts) (lcRounds l claim' ts')
  | [], _, _, _, _, _, hc => RS.pure' hc
  | (q1, qinf) :: rest, claim, claim', ts, ts', hl, hc => by
    have hg := hl (q1, qinf) (by simp)
    simp only [lcRounds]
    refine RS.bind RS.triv fun t t' _ => ?_
    have he0 := Lin.LSk.add' hc (Lin.LSk.refl hg.1)
    exact lcRounds_rs rest _ _ _ _ (fun p hp => hl p (by simp [hp]))
      (Lin.LSk.add' (Lin.LSk.add' (Lin.LSk.sm' (Lin.LSk.refl hg.2))
        (Lin.LSk.sm' (Lin.LSk.add' (Lin.LSk.add' he0 (Lin.LSk.refl hg.1)) (Lin.LSk.refl hg.2)))) he0)

theorem lincheck_rs (kLog : Nat) (zc zc' : ZcPts Lin) :
    RS (fun p p' => Lin.Skel p.1 p'.1 ∧ Lin.LSk p.2.w p'.2.w) (lincheck kLog M zc) (lincheck kLog M zc') := by
  unfold lincheck
  refine RS.bind RS.triv fun _ _ _ => RS.bind RS.triv fun _ _ _ => ?_
  refine RS.bind (lcRounds_rs M.lc.toList _ _ #[] #[] (fun p hp => hM.lc p (by simpa using hp))
    (Lin.LSk.add' (Lin.LSk.add' (Lin.LSk.sm' (Lin.LSk.refl hM.fa)) (Lin.LSk.refl hM.fb)) Lin.LSk.cst'))
    fun ⟨claim, ts⟩ ⟨claim', ts'⟩ hc => ?_
  refine RS.bind (RS.liftE' (R := fun a b => a = b) fun a b ha hb => Except.ok.inj (ha.symm.trans hb))
    fun acc acc' (he : acc = acc') => RS.bind RS.triv fun r r' _ => RS.pure' ⟨?_, ?_⟩
  · subst he
    exact Lin.Skel.add' (Lin.Skel.refl' acc) (Lin.LSk.skel hc)
  · exact dotV_sk (by rw [skipWeights_size, skipWeights_size]) (arr_self hM.zp)

omit hM in
theorem extras_sk : ∀ {l l' : List Claim}, l.length = l'.length →
    List.Forall₂ (fun cl cl' : AClaim Lin => Lin.LSk cl.value cl'.value)
      (l.map fun e => (⟨cst e.value, e.skip, e.x⟩ : AClaim Lin)) (l'.map fun e => ⟨cst e.value, e.skip, e.x⟩)
  | [], [], _ => .nil
  | _ :: _, _ :: _, h => .cons Lin.LSk.cst' (extras_sk (by simpa using h))
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

omit hM in
theorem claimsOf_sk {m kLog : Nat} {zc zc' : ZcPts Lin} {lc lc' : LcPts Lin} {extras extras' : Array Claim}
    (hw : Lin.LSk lc.w lc'.w) (hvc : Lin.LSk zc.vc zc'.vc) (hx : extras.size = extras'.size) :
    List.Forall₂ (fun cl cl' : AClaim Lin => Lin.LSk cl.value cl'.value) (claimsOf m kLog zc lc extras).toList
      (claimsOf m kLog zc' lc' extras').toList := by
  simp only [claimsOf, Array.toList_append, Array.toList_map]
  exact .cons hw (.cons hvc (extras_sk (by simpa using hx)))

theorem rsClaim_rs (k : Nat) {cl cl' : AClaim Lin} (hcl : Lin.LSk cl.value cl'.value) :
    RS (fun p p' => Lin.LSk p.1 p'.1 ∧ Lin.LSk p.2.1 p'.2.1) (rsClaim M k cl) (rsClaim M k cl') := by
  unfold rsClaim
  refine RS.bind (coinVec_rs 7) fun rpp rpp' hr => RS.pure' ⟨?_, ?_⟩
  · exact Lin.LSk.add' (dotV_sk (by simp) (arr_self (hM.rs k))) hcl
  · exact dotV_sk (by rw [sz_eqTable, sz_eqTable, hr]) (arr_self (hM.rsT k))

theorem rsClaims_rs {cls cls' : List (AClaim Lin)}
    (h : List.Forall₂ (fun cl cl' : AClaim Lin => Lin.LSk cl.value cl'.value) cls cls') :
    ∀ k, RS (fun p p' => List.Forall₂ Lin.LSk p.1.toList p'.1.toList ∧ List.Forall₂ Lin.LSk p.2.1.toList p'.2.1.toList)
      (rsClaims M cls k) (rsClaims M cls' k) := by
  induction h with
  | nil => intro k; exact RS.pure' ⟨.nil, .nil⟩
  | cons hcl _ ih =>
    intro k
    simp only [rsClaims]
    refine RS.bind (rsClaim_rs hM k hcl) fun ⟨out, beta, rpp, z⟩ ⟨out', beta', rpp', z'⟩ h1 => ?_
    refine RS.bind (ih (k + 1)) fun ⟨outs, betas, pts⟩ ⟨outs', betas', pts'⟩ h2 => RS.pure' ⟨?_, ?_⟩
    · simp only [Array.toList_append]
      exact .cons h1.1 h2.1
    · simp only [Array.toList_append]
      exact .cons h1.2 h2.2

theorem ringSwitch_rs {claims claims' : Array (AClaim Lin)}
    (h : List.Forall₂ (fun cl cl' : AClaim Lin => Lin.LSk cl.value cl'.value) claims.toList claims'.toList) :
    RS (fun p p' => List.Forall₂ Lin.LSk p.1.toList p'.1.toList ∧ Lin.LSk p.2.T p'.2.T) (ringSwitch M claims)
      (ringSwitch M claims') := by
  have hsz : claims'.size = claims.size := by simpa using (forall₂_length h).symm
  unfold ringSwitch
  rw [hsz]
  refine RS.bind (rsClaims_rs hM h 0) fun ⟨outs, betas, pts⟩ ⟨outs', betas', pts'⟩ h1 => ?_
  exact RS.bind (coinVec_rs _) fun gamma gamma' hg => RS.pure' ⟨h1.1, dotV_sk hg h1.2⟩

theorem nextMsg_rs : RSL MSk (nextMsg M) (nextMsg M) := by
  unfold nextMsg
  refine RSL.bind RSL.get' fun st st' hst => ?_
  rw [← hst.msg]
  cases hm : M.msgs[st.msg]? with
  | none => exact RSL.throw' _
  | some mm =>
    exact RSL.bind (RSL.set' ⟨hst.s, hst.q, rfl, hst.ood⟩) fun _ _ _ => RSL.pure' (hM.msgs mm (mem_of_get? hm))

theorem nextOod_rs : RSL Lin.LSk (nextOod M) (nextOod M) := by
  unfold nextOod
  refine RSL.bind RSL.get' fun st st' hst => ?_
  rw [← hst.ood]
  cases hy : M.ood[st.ood]? with
  | none => exact RSL.throw' _
  | some y =>
    exact RSL.bind (RSL.set' ⟨hst.s, hst.q, hst.msg, rfl⟩) fun _ _ _ =>
      RSL.pure' (Lin.LSk.refl (hM.ood y (mem_of_get? hy)))

theorem challenge_rs : RSL (fun _ _ => True) (challenge M) (challenge M) := by
  unfold challenge
  refine RSL.bind RSL.liftT fun rk rk' _ => RSL.bind RSL.get' fun st st' hst => ?_
  obtain ⟨s, ⟨a0, a1, a2⟩, msg, ood, oods, conss⟩ := st
  obtain ⟨s', ⟨a0', a1', a2'⟩, msg', ood', oods', conss'⟩ := st'
  obtain ⟨hs, ⟨h0, h1, h2⟩, hmsg, hood⟩ := hst
  refine RSL.bind (RSL.set' ⟨KSk.kadd' (KSk.kadd' h0 (KSk.kc' h1)) (KSk.kc' h2), ⟨h0, h1, h2⟩, hmsg, hood⟩)
    fun _ _ _ => RSL.bind (nextMsg_rs hM) fun mm mm' hm => ?_
  exact RSL.bind (RSL.modify' fun st st' h => ⟨h.s, ofMsg_sk hm h.s, h.msg, h.ood⟩) fun _ _ _ => RSL.pure' trivial

theorem challenges_rs : ∀ n, RSL (fun a a' : Array F256 => a.size = a'.size) (challenges M n) (challenges M n)
  | 0 => RSL.pure' rfl
  | n + 1 => by
    simp only [challenges]
    exact RSL.bind (challenge_rs hM) fun _ _ _ => RSL.bind (challenges_rs n) fun rest rest' h =>
      RSL.pure' (by simp [h])

omit hM in
theorem addScaled_rs {p p' : KV Lin × KV Lin × KV Lin} {b b' : F128} (hp : K3Sk p p') :
    RSL (fun _ _ => True) (addScaled p b) (addScaled p' b') :=
  RSL.modify' fun _ _ h =>
    ⟨h.s, ⟨KSk.kadd' h.q.1 (KSk.fk' hp.1), KSk.kadd' h.q.2.1 (KSk.fk' hp.2.1), KSk.kadd' h.q.2.2 (KSk.fk' hp.2.2)⟩,
      h.msg, h.ood⟩

theorem oodIntro_rs (dim split : Nat) : RSL (fun _ _ => True) (oodIntro M dim split) (oodIntro M dim split) := by
  unfold oodIntro
  refine RSL.bind RSL.liftT fun _ _ _ => RSL.bind (nextOod_rs hM) fun y y' hy =>
    RSL.bind (nextMsg_rs hM) fun mm mm' hm => RSL.bind RSL.liftT fun b b' _ => ?_
  exact RSL.bind (addScaled_rs (ofMsg_sk hm (KSk.kofF' hy))) fun _ _ _ =>
    RSL.modify' fun _ _ h => ⟨KSk.kadd' h.s (KSk.kofF' (Lin.LSk.sm' hy)), h.q, h.msg, h.ood⟩

theorem cons_sk (i : Nat) (ea ea' : Array F128) :
    List.Forall₂ Lin.LSk (M.cons i ea).toList (M.cons i ea').toList := by
  rw [hM.consEq i ea' ea]
  exact arr_self (hM.cons i ea)

theorem query_rs (sch : Schedule) (i : Nat) {fold fold' : Array F256} (hf : fold.size = fold'.size) :
    RSL (fun p p' => KSk p.2.2 p'.2.2) (query sch M i fold) (query sch M i fold') := by
  unfold query
  refine RSL.bind RSL.liftT fun _ _ _ => RSL.bind RSL.liftT fun _ _ _ => RSL.pure' ?_
  exact kaddAll_kf_sk (by rw [sz_eqTableK, sz_eqTableK, hf]) (cons_sk hM _ _ _)

theorem queryFold_rs (sch : Schedule) (i cols : Nat) {fold fold' : Array F256} (hf : fold.size = fold'.size) :
    RSL (fun _ _ => True) (queryFold sch M i cols fold) (queryFold sch M i cols fold') := by
  unfold queryFold
  refine RSL.bind (query_rs hM sch i hf) fun ⟨pos, al, e⟩ ⟨pos', al', e'⟩ he =>
    RSL.bind (nextMsg_rs hM) fun mm mm' hm => RSL.bind RSL.liftT fun b b' _ => ?_
  exact RSL.bind (addScaled_rs (ofMsg_sk hm he)) fun _ _ _ =>
    RSL.modify' fun _ _ h => ⟨KSk.kadd' h.s (KSk.fk' he), h.q, h.msg, h.ood⟩

theorem levelsLoop_rs (sch : Schedule) (R : Nat) :
    ∀ fuel i split (levels levels' : Array (Array F256)) ext,
      RSL (fun p p' => p.2 = p'.2) (levelsLoop sch M R fuel i split levels ext)
        (levelsLoop sch M R fuel i split levels' ext)
  | 0, _, _, _, _, _ => RSL.pure' rfl
  | fuel + 1, i, split, levels, levels', ext => by
    simp only [levelsLoop]
    refine RSL.ite' (fun _ => RSL.throw' _) fun _ => ?_
    refine RSL.bind (challenges_rs hM 4) fun chs chs' hc => RSL.ite' (fun _ => ?_) fun _ => ?_
    · exact RSL.bind (oodIntro_rs hM _ _) fun _ _ _ => RSL.bind (oodIntro_rs hM _ _) fun _ _ _ =>
        RSL.bind (queryFold_rs hM sch _ _ hc) fun _ _ _ => levelsLoop_rs sch R fuel _ _ _ _ _
    · refine RSL.bind (query_rs hM sch R hc) fun ⟨pos, al, e⟩ ⟨pos', al', e'⟩ he =>
        RSL.bind RSL.liftT fun b b' _ => ?_
      exact RSL.bind (RSL.modify' fun _ _ h => ⟨KSk.kadd' h.s (KSk.fk' he), h.q, h.msg, h.ood⟩) fun _ _ _ =>
        levelsLoop_rs sch R fuel _ _ _ _ _

theorem ligeritoRun_rs (sch : Schedule) :
    RSL (fun p p' => p.2.2 = p'.2.2) (ligeritoRun sch M) (ligeritoRun sch M) := by
  unfold ligeritoRun
  refine RSL.bind RSL.liftT fun _ _ _ => RSL.bind (nextOod_rs hM) fun y y' hy => RSL.bind RSL.liftT fun b b' _ => ?_
  refine RSL.bind (RSL.modify' fun _ _ h => ⟨KSk.kadd' h.s (KSk.kofF' (Lin.LSk.sm' hy)), h.q, h.msg, h.ood⟩)
    fun _ _ _ => RSL.bind (nextMsg_rs hM) fun mm mm' hm => ?_
  refine RSL.bind (RSL.modify' fun _ _ h => ⟨h.s, ofMsg_sk hm h.s, h.msg, h.ood⟩) fun _ _ _ =>
    RSL.bind (challenges_rs hM _) fun initCh initCh' hc => ?_
  refine RSL.bind (oodIntro_rs hM _ _) fun _ _ _ => RSL.bind (oodIntro_rs hM _ _) fun _ _ _ =>
    RSL.bind (queryFold_rs hM sch _ _ hc) fun _ _ _ => ?_
  exact RSL.bind (levelsLoop_rs hM sch _ _ _ _ _ _ _) fun ⟨levels, ext⟩ ⟨levels', ext'⟩ he => RSL.pure' he

theorem getD_yr (i : Nat) : (M.yr.getD i zero).q = [] := by
  rw [Array.getD_eq_getD_getElem?]
  cases hy : M.yr[i]? with
  | none => rfl
  | some x => exact hM.yr x (mem_of_get? hy)

theorem ligerito_rs (sch : Schedule) {tIn tIn' : Lin} (sws sws' : Array Switched) (ht : Lin.LSk tIn tIn') :
    RS (fun p p' => List.Forall₂ Lin.LSk p.1.toList p'.1.toList) (ligerito sch M tIn sws)
      (ligerito sch M tIn' sws') := by
  unfold ligerito
  refine RS.bind (ligeritoRun_rs hM sch _ _ ⟨KSk.kofF' ht, ⟨KSk.zero', KSk.zero', KSk.zero'⟩, rfl, rfl⟩)
    fun ⟨⟨initCh, levels, ext⟩, st⟩ ⟨⟨initCh', levels', ext'⟩, st'⟩ ⟨(he : ext = ext'), hst⟩ => ?_
  subst he
  refine RS.bind RS.triv fun _ _ _ => RS.bind RS.triv fun _ _ _ =>
    RS.guard (fun _ _ _ h => by cases h) (fun _ _ _ h => by cases h) fun _ _ => ?_
  refine RS.guard (fun _ _ _ h => by cases h) (fun _ _ _ h => by cases h) fun _ _ => ?_
  refine RS.bind (RS.liftE' (R := fun r r' : Array F256 => r.size = r'.size) fun a b ha hb => by
    rw [(finalWeights_ok ha).2.2.2.2.1, (finalWeights_ok hb).2.2.2.2.1]) fun res res' hres => ?_
  refine RS.guard (fun _ _ _ h => by cases h) (fun _ _ _ h => by cases h) fun _ _ => RS.pure' ?_
  rw [← hres]
  have hl := kaddAll_sk (forall₂_map_same (S := KSk)
    (f := fun j => kc res[j]! (M.yr.getD (2 * j) zero, M.yr.getD (2 * j + 1) zero))
    (g := fun j => kc res'[j]! (M.yr.getD (2 * j) zero, M.yr.getD (2 * j + 1) zero))
    (fun j => KSk.kc' ⟨Lin.LSk.refl (getD_yr hM _), Lin.LSk.refl (getD_yr hM _)⟩) (Array.range res.size).toList)
  have hfin := KSk.kadd' hl hst.s
  exact .cons hfin.1 (.cons hfin.2 .nil)

theorem run_rs (sh : Shape) (extras extras' : Array Claim) :
    RS (fun p p' => List.Forall₂ Lin.Skel p.1.toList p'.1.toList) (run sh M extras) (run sh M extras') := by
  unfold run
  refine RS.guard (fun _ _ _ h => by cases h) (fun _ _ _ h => by cases h) fun h₁ h₂ => ?_
  have hx : extras.size = extras'.size := by omega
  refine RS.bind (zerocheck_rs hM _) fun ⟨r0, zc⟩ ⟨r0', zc'⟩ h0 =>
    RS.bind (lincheck_rs hM _ zc zc') fun ⟨r1, lc⟩ ⟨r1', lc'⟩ h1 => ?_
  refine RS.bind (ringSwitch_rs hM (claimsOf_sk h1.2 h0.2 hx)) fun ⟨rsr, rs⟩ ⟨rsr', rs'⟩ h2 => ?_
  have ht : Lin.LSk (M.tAtom.getD rs.T) (M.tAtom.getD rs'.T) := by
    cases hA : M.tAtom with
    | none => exact h2.2
    | some x => exact Lin.LSk.refl (hM.tAtom x hA)
  refine RS.bind (ligerito_rs hM _ _ _ ht) fun ⟨lgr, lg⟩ ⟨lgr', lg'⟩ h3 => RS.pure' ?_
  simp only [Array.toList_append]
  exact forall₂_append (forall₂_append (forall₂_append (.cons h0.1 (.cons h1.1 .nil))
    (forall₂_mono (fun _ _ h => Lin.LSk.skel h) h2.1)) (forall₂_mono (fun _ _ h => Lin.LSk.skel h) h3))
    (.cons (Lin.LSk.skel (Lin.LSk.add' ht h2.2)) .nil)

end Run

theorem residualLins_run {sh : Shape} {coins : Coins} {extras : Array Claim} {r : Array Lin × Info Lin}
    (h : residualLins sh coins extras = .ok r) : ∃ j, run sh (symbolic sh) extras 0 coins = .ok (r, j) := by
  unfold residualLins at h
  change (Prod.fst <$> run sh (symbolic sh) extras 0 coins) = _ at h
  cases hr : run sh (symbolic sh) extras 0 coins with
  | error e => rw [hr] at h; cases h
  | ok p =>
    obtain ⟨a, j⟩ := p
    rw [hr] at h
    cases h
    exact ⟨j, rfl⟩

/-- **The residuals at the zero coins and at a rep's coins have one skeleton** whenever `run` succeeds at both
(RecAlgebra.lean:810–811, 532–539). -/
theorem residualLins_skel {sh : Shape} {rounds : Array (Array F128)} {extras₀ extras : Array Claim}
    {lins₀ lins : Array Lin} {info₀ info : Info Lin}
    (hz : residualLins sh Coins.zeros extras₀ = .ok (lins₀, info₀))
    (hl : residualLins sh (Coins.ofRounds rounds) extras = .ok (lins, info)) :
    List.Forall₂ Lin.Skel lins₀.toList lins.toList := by
  obtain ⟨j, hj⟩ := residualLins_run hz
  obtain ⟨j', hj'⟩ := residualLins_run hl
  exact run_rs (symbolic_mlin sh) sh extras₀ extras _ _ 0 0 _ _ j j' sized_zeros (sized_ofRounds rounds) hj hj'

/-- **`verifierRows` returns** (RecAlgebra.lean:824–829) whenever `structureOf` does and `run` succeeds at the rep's
coins: its S is `structureOf`'s, and v is the coefficient slots, then the comb. -/
theorem verifierRows_ok {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb : Array F128}
    {lins : Array Lin} {info : Info Lin} {S : GfS} (hS : structureOf sh = .ok S)
    (hl : residualLins sh (Coins.ofRounds rounds) extras = .ok (lins, info)) :
    ∃ b : Built, build sh lins = .ok b ∧ b.S = S ∧
      verifierRows sh rounds extras comb = .ok (b.vVals ++ comb, info, b.kappa) := by
  have hS' := hS
  unfold structureOf at hS'
  cases hz : residualLins sh Coins.zeros (extrasLike sh) with
  | error e => rw [hz] at hS'; cases hS'
  | ok p =>
    obtain ⟨lins₀, info₀⟩ := p
    exact verifierRows_ok_of_skel hS hz hl (residualLins_skel hz hl)

/-- `verifierRows` returns exactly when `structureOf` does and `run` succeeds at the rep's coins. -/
theorem verifierRows_ok_iff {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb : Array F128} :
    (∃ x, verifierRows sh rounds extras comb = .ok x) ↔
      (∃ S, structureOf sh = .ok S) ∧ ∃ r, residualLins sh (Coins.ofRounds rounds) extras = .ok r := by
  constructor
  · rintro ⟨x, hx⟩
    unfold verifierRows at hx
    cases hl : residualLins sh (Coins.ofRounds rounds) extras with
    | error e => rw [hl] at hx; cases hx
    | ok p =>
      refine ⟨?_, p, rfl⟩
      cases hS : structureOf sh with
      | ok S => exact ⟨S, rfl⟩
      | error e =>
        exfalso
        obtain ⟨lins, info⟩ := p
        rw [hl] at hx
        simp only [bind, Except.bind] at hx
        cases hb : build sh lins with
        | error e' => rw [hb] at hx; cases hx
        | ok b =>
          rw [hb, hS] at hx
          simp [Except.toOption] at hx
  · rintro ⟨⟨S, hS⟩, ⟨lins, info⟩, hl⟩
    obtain ⟨b, -, -, h⟩ := verifierRows_ok (comb := comb) hS hl
    exact ⟨_, h⟩

end Flock.RecAlgebra
