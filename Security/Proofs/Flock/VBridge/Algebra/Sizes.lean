import Proofs.Flock.VBridge.Algebra.RunSkel

/-!
# `verifierRows` reads a record's rounds only through their sizes

`run` (RecAlgebra.lean:532–539) branches on the coins only through how many each round holds: the values it computes
from them feed `finalWeights` (RecAlgebra.lean:456–496), whose checks read only sizes. So at two records whose rounds
hold as many coins each, with as many extra claims, `run` succeeds at one exactly when at the other
(`residualLins_sizes`), and so does `verifierRows` (`verifierRows_sizes`).
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-- Two records whose rounds hold as many coins each. -/
def SameSizes (r r' : Array (Array F128)) : Prop := r.toList.map Array.size = r'.toList.map Array.size

theorem SameSizes.symm {r r' : Array (Array F128)} (h : SameSizes r r') : SameSizes r' r := Eq.symm h

theorem SameSizes.size {r r' : Array (Array F128)} (h : SameSizes r r') : r.size = r'.size := by
  simpa using congrArg List.length h

theorem SameSizes.get {r r' : Array (Array F128)} (h : SameSizes r r') (k : Nat) :
    r[k]?.map Array.size = r'[k]?.map Array.size := by
  simpa [List.getElem?_map] using congrArg (·[k]?) h

/-- Two records' rounds of some coins, which `Coins.ofRounds` reads, hold as many coins each when the records' do. -/
theorem SameSizes.filter {r r' : Array (Array F128)} (h : SameSizes r r') :
    SameSizes (r.filter (·.size ≠ 0)) (r'.filter (·.size ≠ 0)) := by
  have e : ∀ l : List (Array F128), (l.filter (·.size ≠ 0)).map Array.size = (l.map Array.size).filter (· ≠ 0) :=
    fun l => (List.filter_map (f := Array.size) (p := fun n => decide (n ≠ 0)) (l := l)).symm
  unfold SameSizes at h ⊢
  rw [Array.toList_filter, Array.toList_filter, e, e, h]

theorem round_sizes {r r' : Array (Array F128)} (h : SameSizes r r') {k n : Nat} {v : Array F128}
    (hv : (Coins.ofRounds r).round k n = .ok v) : ∃ v', (Coins.ofRounds r').round k n = .ok v' := by
  have hk := h.filter.get k
  simp only [Coins.ofRounds] at hv ⊢
  cases h1 : (r.filter (·.size ≠ 0))[k]? with
  | none => rw [h1] at hv; cases hv
  | some a =>
    rw [h1] at hv hk
    cases h2 : (r'.filter (·.size ≠ 0))[k]? with
    | none => rw [h2] at hk; cases hk
    | some a' =>
      rw [h2] at hk
      simp only [Option.map_some, Option.some.injEq] at hk
      by_cases hn : a.size = n
      · exact ⟨a', by simp [← hk, hn]⟩
      · simp [hn] at hv

theorem coinVec_of {n k : Nat} {src : Coins} {v : Array F128} (hn : n ≠ 0) (h : src.round k n = .ok v) :
    coinVec n k src = .ok (v, k + 1) := by
  unfold coinVec
  split
  · contradiction
  · change (match src.round k n with
      | .ok r => (set (k + 1) >>= fun _ => pure r : CM (Array F128))
      | .error e => (throw e : CM (Array F128))) k src = _
    rw [h]
    rfl

theorem cm_bind_of {A B : Type} {x : CM A} {f : A → CM B} {k k₁ : Nat} {src : Coins} {a : A}
    (h : x k src = .ok (a, k₁)) : (x >>= f) k src = f a k₁ src := by
  have e : (x >>= f) k src = (x k src).bind (fun p => f p.1 p.2 src) := rfl
  rw [e, h]
  rfl

/-! ## Two runs at two records of one shape -/

/-- `x` at a record and `y` at any record whose rounds have the same sizes: whenever `x` succeeds, `y` does, with as
many coins drawn and related results. -/
def RF {A B : Type} (R : A → B → Prop) (x : CM A) (y : CM B) : Prop :=
  ∀ r r', SameSizes r r' → ∀ k a j, x k (Coins.ofRounds r) = .ok (a, j) →
    ∃ b, y k (Coins.ofRounds r') = .ok (b, j) ∧ R a b

namespace RF

variable {A B C D : Type}

theorem bind {R : A → B → Prop} {R' : C → D → Prop} {x : CM A} {y : CM B} {f : A → CM C} {g : B → CM D}
    (hs : RF R x y) (h : ∀ a b, R a b → RF R' (f a) (g b)) : RF R' (x >>= f) (y >>= g) := by
  intro r r' hr k c j hx
  obtain ⟨a, k₁, hx₁, hx₂⟩ := cm_bind hx
  obtain ⟨b, hy₁, hab⟩ := hs r r' hr k a k₁ hx₁
  obtain ⟨d, hy₂, hcd⟩ := h a b hab r r' hr k₁ c j hx₂
  exact ⟨d, by rw [cm_bind_of hy₁]; exact hy₂, hcd⟩

theorem pure' {R : A → B → Prop} {a : A} {b : B} (h : R a b) : RF R (Pure.pure a) (Pure.pure b) := by
  intro r r' _ k a' j hx
  cases cm_pure hx
  exact ⟨b, rfl, h⟩

theorem liftE' {R : A → B → Prop} {e : Except String A} {e' : Except String B}
    (h : ∀ a, e = .ok a → ∃ b, e' = .ok b ∧ R a b) : RF R (liftE e) (liftE e') := by
  intro r r' _ k a j hx
  obtain ⟨a₁, ha, hr⟩ := liftE_ok hx
  cases hr
  obtain ⟨b, rfl, hab⟩ := h _ ha
  exact ⟨b, rfl, hab⟩

theorem throw' {R : A → B → Prop} {y : CM B} (e : String) : RF R (throw e) y := by
  intro _ _ _ _ _ _ hx
  cases hx

theorem ite' {R : A → B → Prop} {c c' : Prop} [Decidable c] [Decidable c'] (hc : c ↔ c') {x₁ x₂ : CM A}
    {y₁ y₂ : CM B} (h₁ : c → RF R x₁ y₁) (h₂ : ¬c → RF R x₂ y₂) :
    RF R (if c then x₁ else x₂) (if c' then y₁ else y₂) := by
  by_cases h : c
  · simp only [h, hc.mp h, ↓reduceIte]
    exact h₁ h
  · simp only [h, mt hc.mpr h, ↓reduceIte]
    exact h₂ h

theorem read' : RF (fun s s' : Coins => ∀ k, s.drawnAll k = s'.drawnAll k) (read : CM Coins) read := by
  intro r r' hr k a j hx
  have e : ∀ src : Coins, (read : CM Coins) k src = .ok (src, k) := fun _ => rfl
  rw [e] at hx
  cases hx
  exact ⟨_, e _, fun k => by
    show (k == (r.filter (·.size ≠ 0)).size) = (k == (r'.filter (·.size ≠ 0)).size)
    rw [hr.filter.size]⟩

theorem get' : RF (· = ·) (get : CM Nat) get := by
  intro r r' hr k a j hx
  have e : ∀ src : Coins, (get : CM Nat) k src = .ok (k, k) := fun _ => rfl
  rw [e] at hx
  cases hx
  exact ⟨_, e _, rfl⟩

end RF

theorem coinVec_rf (n : Nat) : RF (fun v v' : Array F128 => v.size = v'.size) (coinVec n) (coinVec n) := by
  intro r r' hr k v j hx
  rcases coinVec_ok hx with ⟨rfl, rfl, rfl⟩ | ⟨hn, hv, rfl⟩
  · exact ⟨#[], rfl, rfl⟩
  · obtain ⟨v', hv'⟩ := round_sizes hr hv
    refine ⟨v', coinVec_of hn hv', ?_⟩
    show v.size = v'.size
    rw [sized_ofRounds r _ _ _ hv, sized_ofRounds r' _ _ _ hv']

theorem coinOne_rf : RF (fun _ _ => True) coinOne coinOne := by
  unfold coinOne
  exact RF.bind (coinVec_rf 1) fun _ _ _ => RF.pure' trivial

theorem coinK2_rf : RF (fun _ _ => True) coinK2 coinK2 := by
  unfold coinK2
  exact RF.bind (coinVec_rf 2) fun _ _ _ => RF.pure' trivial

/-! ## Ligerito's state at two records -/

/-- Two OOD contexts of one dimension and split. -/
def OodSz (o o' : OodCtx) : Prop := o.z.size = o'.z.size ∧ o.split = o'.split

/-- Two consistency contexts of one width and level. -/
def ConsSz (c c' : ConsCtx) : Prop := c.cols = c'.cols ∧ c.start = c'.start

/-- Two Ligerito states alike: as many messages and OOD values read, contexts of one shape. -/
structure SF (st st' : LSt Lin) : Prop where
  msg : st.msg = st'.msg
  ood : st.ood = st'.ood
  oods : List.Forall₂ OodSz st.oods.toList st'.oods.toList
  conss : List.Forall₂ ConsSz st.conss.toList st'.conss.toList

theorem SF.oods' {st st' : LSt Lin} (h : SF st st') {z z' : Array F128} (hz : z.size = z'.size) (b b' : F128)
    (sp : Option Nat) :
    List.Forall₂ OodSz (st.oods.push ⟨z, b, sp⟩).toList (st'.oods.push ⟨z', b', sp⟩).toList := by
  simp only [Array.toList_push]
  exact forall₂_append h.oods (.cons ⟨hz, rfl⟩ .nil)

theorem SF.conss' {st st' : LSt Lin} (h : SF st st') (cols : Nat) (pos pos' : Array Nat) (al al' : Array F128)
    (b b' : F128) (i : Nat) :
    List.Forall₂ ConsSz (st.conss.push ⟨cols, pos, al, b, i⟩).toList
      (st'.conss.push ⟨cols, pos', al', b', i⟩).toList := by
  simp only [Array.toList_push]
  exact forall₂_append h.conss (.cons ⟨rfl, rfl⟩ .nil)

/-- `x` and `y` from alike states at two records of one shape: whenever `x` succeeds, `y` does, with related results
and alike states. -/
def RFL {A B : Type} (R : A → B → Prop) (x : LM Lin A) (y : LM Lin B) : Prop :=
  ∀ st st', SF st st' → RF (fun p p' => R p.1 p'.1 ∧ SF p.2 p'.2) (x st) (y st')

namespace RFL

variable {A B C D : Type}

theorem bind {R : A → B → Prop} {R' : C → D → Prop} {x : LM Lin A} {y : LM Lin B} {f : A → LM Lin C}
    {g : B → LM Lin D} (hs : RFL R x y) (h : ∀ a b, R a b → RFL R' (f a) (g b)) : RFL R' (x >>= f) (y >>= g) := by
  intro st st' hst
  exact RF.bind (f := fun p : A × LSt Lin => f p.1 p.2) (g := fun p : B × LSt Lin => g p.1 p.2) (hs st st' hst)
    fun p p' hp => h p.1 p'.1 hp.1 p.2 p'.2 hp.2

theorem lift {R : A → B → Prop} {x : CM A} {y : CM B} (hs : RF R x y) :
    RFL R (StateT.lift x : LM Lin A) (StateT.lift y : LM Lin B) := by
  intro st st' hst
  exact RF.bind hs fun a b hab => RF.pure' ⟨hab, hst⟩

theorem pure' {R : A → B → Prop} {a : A} {b : B} (h : R a b) : RFL R (Pure.pure a) (Pure.pure b) :=
  fun _ _ hst => RF.pure' ⟨h, hst⟩

theorem throw' {R : A → B → Prop} {y : LM Lin B} (e : String) : RFL R (throw e) y := by
  intro st st' _ r r' _ k a j hx
  cases hx

theorem ite' {R : A → B → Prop} {c : Prop} [Decidable c] {x₁ x₂ : LM Lin A} {y₁ y₂ : LM Lin B}
    (h₁ : c → RFL R x₁ y₁) (h₂ : ¬c → RFL R x₂ y₂) : RFL R (if c then x₁ else x₂) (if c then y₁ else y₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

theorem get' : RFL SF (get : LM Lin (LSt Lin)) (get : LM Lin (LSt Lin)) := fun _ _ hst => RF.pure' ⟨hst, hst⟩

theorem set' {s s' : LSt Lin} (h : SF s s') :
    RFL (fun _ _ => True) (set s : LM Lin PUnit) (set s' : LM Lin PUnit) :=
  fun _ _ _ => RF.pure' ⟨trivial, h⟩

theorem modify' {m m' : LSt Lin → LSt Lin} (hm : ∀ st st', SF st st' → SF (m st) (m' st')) :
    RFL (fun _ _ => True) (modify m : LM Lin PUnit) (modify m' : LM Lin PUnit) :=
  fun st st' hst => RF.pure' ⟨trivial, hm st st' hst⟩

end RFL

/-! ## `finalWeights` checks sizes only -/

theorem foldlM_of {α β : Type} {f : β → α → Except String β} {P : α → Prop} (hf : ∀ b a, P a → ∃ r, f b a = .ok r) :
    ∀ (l : List α) (b : β), (∀ a ∈ l, P a) → ∃ r, l.foldlM f b = .ok r
  | [], b, _ => ⟨b, rfl⟩
  | a :: l, b, h => by
    obtain ⟨r, hr⟩ := hf b a (h a (by simp))
    obtain ⟨r', hr'⟩ := foldlM_of hf l r fun x hx => h x (by simp [hx])
    exact ⟨r', by rw [List.foldlM_cons, hr]; exact hr'⟩

theorem foldlM_bind_of {α β γ : Type} {f : β → α → Except String β} {g : β → Except String γ} {P : α → Prop}
    (hf : ∀ b a, P a → ∃ r, f b a = .ok r) (l : List α) (b : β) (hl : ∀ a ∈ l, P a) (hg : ∀ r, ∃ s, g r = .ok s) :
    ∃ s, (l.foldlM f b >>= g) = .ok s := by
  obtain ⟨r, hr⟩ := foldlM_of hf l b hl
  obtain ⟨s, hs⟩ := hg r
  exact ⟨s, by rw [hr]; exact hs⟩

theorem foldl_size_const {β : Type} {f : Array F256 → β → Array F256} {N : Nat} (hf : ∀ a b, (f a b).size = N) :
    ∀ (l : List β) (a : Array F256), a.size = N → (l.foldl f a).size = N
  | [], _, h => h
  | b :: l, a, _ => foldl_size_const hf l (f a b) (hf a b)

theorem basis_size (sws : Array Switched) (q : Array F256) (e : Nat) : (basis sws q e).size = 2 ^ e := by
  unfold basis
  rw [← Array.foldl_toList]
  exact foldl_size_const (fun _ _ => by simp) _ _ (by simp)

/-- `finalWeights` returns when its size checks pass (RecAlgebra.lean:456–496). -/
theorem finalWeights_of {sch : Schedule} {ext : Nat} {initCh : Array F256} {levels : Array (Array F256)}
    {oods : Array OodCtx} {conss : Array ConsCtx} {sws : Array Switched}
    (c1 : (origE initCh levels).size + ext = sch.n)
    (hood : ∀ o ∈ oods.toList, o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext)
    (hcons : ∀ cs ∈ conss.toList, (consFixed levels cs.start).size + ext = cs.cols) :
    ∃ res, finalWeights sch ext initCh levels oods conss sws = .ok res := by
  unfold finalWeights
  dsimp only
  have c1' : ¬((levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh).size + ext ≠ sch.n) := not_not.mpr c1
  have c2' : ¬((basis sws (levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh) ext).size ≠ 2 ^ ext) :=
    not_not.mpr (basis_size _ _ _)
  rw [ite_eq_right c1', ite_eq_right c2', ← Array.foldlM_toList]
  refine foldlM_bind_of (P := fun o => o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext) ?_ _ _ hood
    fun r => ?_
  rotate_left
  · rw [← Array.foldlM_toList]
    refine foldlM_of (P := fun cs => (consFixed levels cs.start).size + ext = cs.cols) ?_ _ _ hcons
    intro b cs hcs
    split
    · rename_i hne; exact absurd hcs hne
    · exact ⟨_, rfl⟩
  · intro b o ho
    obtain ⟨z, beta, sp⟩ := o
    cases sp <;>
    · simp only [oodFixed, origE] at ho
      dsimp only
      split
      · rename_i hne; exact absurd ho hne
      · exact ⟨_, rfl⟩

theorem foldl_sizes {f : Array F256 → Array F256 → Array F256}
    (hf : ∀ a a' l l' : Array F256, a.size = a'.size → l.size = l'.size → (f a l).size = (f a' l').size) :
    ∀ (L L' : List (Array F256)) (a a' : Array F256), L.map Array.size = L'.map Array.size → a.size = a'.size →
      (L.foldl f a).size = (L'.foldl f a').size
  | [], [], _, _, _, h => h
  | l :: L, l' :: L', a, a', hL, h => by
    simp only [List.map_cons, List.cons.injEq] at hL
    exact foldl_sizes hf L L' _ _ hL.2 (hf _ _ _ _ h hL.1)
  | [], _ :: _, _, _, hL, _ => by simp at hL
  | _ :: _, [], _, _, hL, _ => by simp at hL

theorem tails_sizes (a a' l l' : Array F256) (ha : a.size = a'.size) (hl : l.size = l'.size) :
    (a ++ l.extract 1 l.size).size = (a' ++ l'.extract 1 l'.size).size := by
  simp [ha, hl]

theorem drop_sizes {levels levels' : Array (Array F256)}
    (hl : levels.toList.map Array.size = levels'.toList.map Array.size) (s : Nat) :
    (levels.extract s levels.size).toList.map Array.size = (levels'.extract s levels'.size).toList.map Array.size := by
  rw [extract_toList, extract_toList, List.map_drop, List.map_drop, hl]

theorem origE_sizes {initCh initCh' : Array F256} {levels levels' : Array (Array F256)}
    (hi : initCh.size = initCh'.size) (hl : levels.toList.map Array.size = levels'.toList.map Array.size) :
    (origE initCh levels).size = (origE initCh' levels').size := by
  unfold origE
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  exact foldl_sizes tails_sizes _ _ _ _ hl hi

theorem consFixed_sizes {levels levels' : Array (Array F256)}
    (hl : levels.toList.map Array.size = levels'.toList.map Array.size) (s : Nat) :
    (consFixed levels s).size = (consFixed levels' s).size := by
  unfold consFixed
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  exact foldl_sizes tails_sizes _ _ _ _ (drop_sizes hl s) rfl

theorem oodFixed_sizes {levels levels' : Array (Array F256)} {orig orig' : Array F256} {o o' : OodCtx}
    (hl : levels.toList.map Array.size = levels'.toList.map Array.size) (hq : orig.size = orig'.size)
    (hs : o.split = o'.split) : (oodFixed levels orig o).1.size = (oodFixed levels' orig' o').1.size := by
  obtain ⟨z, beta, sp⟩ := o
  obtain ⟨z', beta', sp'⟩ := o'
  dsimp only at hs
  subst hs
  cases sp with
  | none => exact hq
  | some s =>
    simp only [oodFixed]
    rw [← Array.foldl_toList, ← Array.foldl_toList]
    refine foldl_sizes (fun a a' l l' ha hl => ?_) _ _ _ _ (drop_sizes hl s) rfl
    have e : a.isEmpty = a'.isEmpty := by simp [Array.isEmpty, ha]
    by_cases hE : a'.isEmpty <;> simp [e, hE, ha, hl]

theorem forall₂_mem_right {γ δ : Type} {R : γ → δ → Prop} :
    ∀ {l : List γ} {l' : List δ}, List.Forall₂ R l l' → ∀ b ∈ l', ∃ a ∈ l, R a b
  | _, _, .nil, _, hb => absurd hb List.not_mem_nil
  | a :: _, _ :: _, .cons hab h, b, hb => by
    rcases List.mem_cons.mp hb with rfl | hb
    · exact ⟨a, List.mem_cons_self, hab⟩
    · obtain ⟨a', ha', h'⟩ := forall₂_mem_right h b hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

/-- `finalWeights` returns on any arguments of the sizes of arguments it returns on. -/
theorem finalWeights_sizes {sch : Schedule} {ext : Nat} {initCh initCh' : Array F256}
    {levels levels' : Array (Array F256)} {oods oods' : Array OodCtx} {conss conss' : Array ConsCtx}
    {sws sws' : Array Switched} {res : Array F256} (h : finalWeights sch ext initCh levels oods conss sws = .ok res)
    (hi : initCh.size = initCh'.size) (hl : levels.toList.map Array.size = levels'.toList.map Array.size)
    (ho : List.Forall₂ OodSz oods.toList oods'.toList) (hc : List.Forall₂ ConsSz conss.toList conss'.toList) :
    ∃ res', finalWeights sch ext initCh' levels' oods' conss' sws' = .ok res' := by
  obtain ⟨c1, -, hood, hcons, -⟩ := finalWeights_ok h
  have hq := origE_sizes hi hl
  refine finalWeights_of (by rw [← hq]; exact c1) (fun o' ho' => ?_) (fun cs' hc' => ?_)
  · obtain ⟨o, hmem, hz, hs⟩ := forall₂_mem_right ho o' ho'
    rw [← hz, hood o hmem, oodFixed_sizes hl hq hs]
  · obtain ⟨cs, hmem, hcols, hst⟩ := forall₂_mem_right hc cs' hc'
    rw [← hcols, ← hst, ← consFixed_sizes hl, hcons cs hmem]

/-! ## `run`'s stages at two records -/

section Run

variable {M : Messages Lin}

theorem zcRounds_rf (r r' : Array F128) :
    ∀ (l : List (Lin × Lin)) (i : Nat) (claim claim' : Lin) (rho rho' : Array F128),
      RF (fun _ _ => True) (zcRounds r l i claim rho) (zcRounds r' l i claim' rho')
  | [], _, _, _, _, _ => RF.pure' trivial
  | (g1, ginf) :: rest, i, claim, claim', rho, rho' => by
    simp only [zcRounds]
    exact RF.bind coinOne_rf fun _ _ _ => zcRounds_rf r r' rest _ _ _ _ _

theorem zerocheck_rf (m : Nat) : RF (fun _ _ => True) (zerocheck m M) (zerocheck m M) := by
  unfold zerocheck
  refine RF.bind (coinVec_rf _) fun _ _ _ => RF.bind (coinVec_rf _) fun _ _ _ => RF.bind coinOne_rf fun _ _ _ => ?_
  refine RF.bind (zcRounds_rf _ _ M.zc.toList 0 _ _ #[] #[]) fun ⟨claim, rho⟩ ⟨claim', rho'⟩ _ => ?_
  exact RF.bind (RF.liftE' (R := fun _ _ => True) fun a ha => ⟨a, ha, trivial⟩) fun _ _ _ => RF.pure' trivial

theorem lcRounds_rf :
    ∀ (l : List (Lin × Lin)) (claim claim' : Lin) (ts ts' : Array F128),
      RF (fun _ _ => True) (lcRounds l claim ts) (lcRounds l claim' ts')
  | [], _, _, _, _ => RF.pure' trivial
  | (q1, qinf) :: rest, claim, claim', ts, ts' => by
    simp only [lcRounds]
    exact RF.bind coinOne_rf fun _ _ _ => lcRounds_rf rest _ _ _ _

theorem lincheck_rf (kLog : Nat) (zc zc' : ZcPts Lin) :
    RF (fun _ _ => True) (lincheck kLog M zc) (lincheck kLog M zc') := by
  unfold lincheck
  refine RF.bind coinOne_rf fun _ _ _ => RF.bind coinOne_rf fun _ _ _ => ?_
  refine RF.bind (lcRounds_rf M.lc.toList _ _ #[] #[]) fun ⟨claim, ts⟩ ⟨claim', ts'⟩ _ => ?_
  exact RF.bind (RF.liftE' (R := fun _ _ => True) fun a ha => ⟨a, ha, trivial⟩) fun _ _ _ =>
    RF.bind coinOne_rf fun _ _ _ => RF.pure' trivial

theorem rsClaim_rf (k : Nat) (cl cl' : AClaim Lin) : RF (fun _ _ => True) (rsClaim M k cl) (rsClaim M k cl') := by
  unfold rsClaim
  exact RF.bind (coinVec_rf 7) fun _ _ _ => RF.pure' trivial

theorem rsClaims_rf : ∀ {cls cls' : List (AClaim Lin)}, cls.length = cls'.length → ∀ k,
    RF (fun _ _ => True) (rsClaims M cls k) (rsClaims M cls' k)
  | [], [], _, _ => RF.pure' trivial
  | cl :: rest, cl' :: rest', h, k => by
    simp only [rsClaims]
    refine RF.bind (rsClaim_rf k cl cl') fun ⟨out, beta, rpp, z⟩ ⟨out', beta', rpp', z'⟩ _ => ?_
    exact RF.bind (rsClaims_rf (by simpa using h) (k + 1)) fun ⟨outs, betas, pts⟩ ⟨outs', betas', pts'⟩ _ =>
      RF.pure' trivial
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h

theorem ringSwitch_rf {claims claims' : Array (AClaim Lin)} (h : claims.size = claims'.size) :
    RF (fun _ _ => True) (ringSwitch M claims) (ringSwitch M claims') := by
  unfold ringSwitch
  rw [← h]
  refine RF.bind (rsClaims_rf (by simpa using h) 0) fun ⟨outs, betas, pts⟩ ⟨outs', betas', pts'⟩ _ => ?_
  exact RF.bind (coinVec_rf _) fun _ _ _ => RF.pure' trivial

theorem nextMsg_rf : RFL (fun _ _ => True) (nextMsg M) (nextMsg M) := by
  unfold nextMsg
  refine RFL.bind RFL.get' fun st st' hst => ?_
  rw [← hst.msg]
  cases M.msgs[st.msg]? with
  | none => exact RFL.throw' _
  | some mm => exact RFL.bind (RFL.set' ⟨rfl, hst.ood, hst.oods, hst.conss⟩) fun _ _ _ => RFL.pure' trivial

theorem nextOod_rf : RFL (fun _ _ => True) (nextOod M) (nextOod M) := by
  unfold nextOod
  refine RFL.bind RFL.get' fun st st' hst => ?_
  rw [← hst.ood]
  cases M.ood[st.ood]? with
  | none => exact RFL.throw' _
  | some y => exact RFL.bind (RFL.set' ⟨hst.msg, rfl, hst.oods, hst.conss⟩) fun _ _ _ => RFL.pure' trivial

theorem challenge_rf : RFL (fun _ _ => True) (challenge M) (challenge M) := by
  unfold challenge
  refine RFL.bind (RFL.lift coinK2_rf) fun rk rk' _ => RFL.bind RFL.get' fun st st' hst => ?_
  obtain ⟨s, ⟨a0, a1, a2⟩, msg, ood, oods, conss⟩ := st
  obtain ⟨s', ⟨a0', a1', a2'⟩, msg', ood', oods', conss'⟩ := st'
  obtain ⟨hmsg, hood, hoods, hconss⟩ := hst
  refine RFL.bind (RFL.set' ⟨hmsg, hood, hoods, hconss⟩) fun _ _ _ => RFL.bind nextMsg_rf fun mm mm' _ => ?_
  exact RFL.bind (RFL.modify' fun _ _ h => ⟨h.msg, h.ood, h.oods, h.conss⟩) fun _ _ _ => RFL.pure' trivial

theorem challenges_rf : ∀ n, RFL (fun a a' : Array F256 => a.size = a'.size) (challenges M n) (challenges M n)
  | 0 => RFL.pure' rfl
  | n + 1 => by
    simp only [challenges]
    exact RFL.bind challenge_rf fun _ _ _ => RFL.bind (challenges_rf n) fun rest rest' h =>
      RFL.pure' (by simp [h])

theorem addScaled_rf {p p' : KV Lin × KV Lin × KV Lin} {b b' : F128} :
    RFL (fun _ _ => True) (addScaled p b) (addScaled p' b') :=
  RFL.modify' fun _ _ h => ⟨h.msg, h.ood, h.oods, h.conss⟩

theorem oodIntro_rf (dim split : Nat) : RFL (fun _ _ => True) (oodIntro M dim split) (oodIntro M dim split) := by
  unfold oodIntro
  refine RFL.bind (RFL.lift (coinVec_rf dim)) fun zz zz' hz => RFL.bind nextOod_rf fun y y' _ =>
    RFL.bind nextMsg_rf fun mm mm' _ => RFL.bind (RFL.lift coinOne_rf) fun b b' _ => ?_
  exact RFL.bind addScaled_rf fun _ _ _ => RFL.modify' fun _ _ h => ⟨h.msg, h.ood, h.oods' hz b b' _, h.conss⟩

theorem query_rf (sch : Schedule) (i : Nat) (fold fold' : Array F256) :
    RFL (fun _ _ => True) (query sch M i fold) (query sch M i fold') := by
  unfold query
  exact RFL.bind (RFL.lift (coinVec_rf _)) fun _ _ _ => RFL.bind (RFL.lift (coinVec_rf _)) fun _ _ _ =>
    RFL.pure' trivial

theorem queryFold_rf (sch : Schedule) (i cols : Nat) (fold fold' : Array F256) :
    RFL (fun _ _ => True) (queryFold sch M i cols fold) (queryFold sch M i cols fold') := by
  unfold queryFold
  refine RFL.bind (query_rf sch i fold fold') fun ⟨pos, al, e⟩ ⟨pos', al', e'⟩ _ =>
    RFL.bind nextMsg_rf fun mm mm' _ => RFL.bind (RFL.lift coinOne_rf) fun b b' _ => ?_
  exact RFL.bind addScaled_rf fun _ _ _ => RFL.modify' fun _ _ h =>
    ⟨h.msg, h.ood, h.oods, h.conss' cols pos pos' al al' b b' i⟩

theorem levelsLoop_rf (sch : Schedule) (R : Nat) :
    ∀ fuel i split (levels levels' : Array (Array F256)) ext,
      levels.toList.map Array.size = levels'.toList.map Array.size →
      RFL (fun p p' => p.1.toList.map Array.size = p'.1.toList.map Array.size ∧ p.2 = p'.2)
        (levelsLoop sch M R fuel i split levels ext) (levelsLoop sch M R fuel i split levels' ext)
  | 0, _, _, _, _, _, hl => RFL.pure' ⟨hl, rfl⟩
  | fuel + 1, i, split, levels, levels', ext, hl => by
    simp only [levelsLoop]
    refine RFL.ite' (fun _ => RFL.throw' _) fun _ => ?_
    refine RFL.bind (challenges_rf 4) fun chs chs' hc => ?_
    have hl' : (levels.push chs).toList.map Array.size = (levels'.push chs').toList.map Array.size := by
      simp [hl, hc]
    refine RFL.ite' (fun _ => ?_) fun _ => ?_
    · exact RFL.bind (oodIntro_rf _ _) fun _ _ _ => RFL.bind (oodIntro_rf _ _) fun _ _ _ =>
        RFL.bind (queryFold_rf sch _ _ _ _) fun _ _ _ => levelsLoop_rf sch R fuel _ _ _ _ _ hl'
    · refine RFL.bind (query_rf sch R _ _) fun ⟨pos, al, e⟩ ⟨pos', al', e'⟩ _ =>
        RFL.bind (RFL.lift coinOne_rf) fun b b' _ => ?_
      exact RFL.bind (RFL.modify' fun _ _ h => ⟨h.msg, h.ood, h.oods, h.conss' _ pos pos' al al' b b' R⟩)
        fun _ _ _ => levelsLoop_rf sch R fuel _ _ _ _ _ hl'

theorem ligeritoRun_rf (sch : Schedule) :
    RFL (fun p p' => p.1.size = p'.1.size ∧ p.2.1.toList.map Array.size = p'.2.1.toList.map Array.size ∧
      p.2.2 = p'.2.2) (ligeritoRun sch M) (ligeritoRun sch M) := by
  unfold ligeritoRun
  refine RFL.bind (RFL.lift (coinVec_rf _)) fun zz zz' hz => RFL.bind nextOod_rf fun y y' _ =>
    RFL.bind (RFL.lift coinOne_rf) fun b b' _ => ?_
  refine RFL.bind (RFL.modify' fun _ _ h => ⟨h.msg, h.ood, h.oods' hz b b' none, h.conss⟩)
    fun _ _ _ => RFL.bind nextMsg_rf fun mm mm' _ => ?_
  refine RFL.bind (RFL.modify' fun _ _ h => ⟨h.msg, h.ood, h.oods, h.conss⟩) fun _ _ _ =>
    RFL.bind (challenges_rf _) fun initCh initCh' hc => ?_
  refine RFL.bind (oodIntro_rf _ _) fun _ _ _ => RFL.bind (oodIntro_rf _ _) fun _ _ _ =>
    RFL.bind (queryFold_rf sch _ _ _ _) fun _ _ _ => ?_
  exact RFL.bind (levelsLoop_rf sch _ _ _ _ _ _ _ rfl) fun ⟨levels, ext⟩ ⟨levels', ext'⟩ he =>
    RFL.pure' ⟨hc, he.1, he.2⟩

theorem ligerito_rf (sch : Schedule) (tIn tIn' : Lin) (sws sws' : Array Switched) :
    RF (fun _ _ => True) (ligerito sch M tIn sws) (ligerito sch M tIn' sws') := by
  unfold ligerito
  refine RF.bind (ligeritoRun_rf sch _ _ ⟨rfl, rfl, .nil, .nil⟩)
    fun ⟨⟨initCh, levels, ext⟩, st⟩ ⟨⟨initCh', levels', ext'⟩, st'⟩ ⟨⟨hi, hl, (he : ext = ext')⟩, hst⟩ => ?_
  subst he
  refine RF.bind RF.read' fun s s' hs => RF.bind RF.get' fun k k' (hk : k = k') => ?_
  subst hk
  refine RF.ite' (by rw [hs k]) (fun _ => RF.throw' _) fun _ => ?_
  refine RF.ite' (by rw [hst.msg, hst.ood]) (fun _ => RF.throw' _) fun _ => ?_
  refine RF.bind (RF.liftE' (R := fun _ _ => True) fun res hres => ?_) fun res res' _ =>
    RF.ite' Iff.rfl (fun _ => RF.throw' _) fun _ => RF.pure' trivial
  obtain ⟨res', h'⟩ := finalWeights_sizes (sws' := sws') hres hi hl hst.oods hst.conss
  exact ⟨res', h', trivial⟩

theorem run_rf (sh : Shape) {extras extras' : Array Claim} (hx : extras.size = extras'.size) :
    RF (fun _ _ => True) (run sh M extras) (run sh M extras') := by
  unfold run
  refine RF.ite' (by rw [hx]) (fun _ => RF.throw' _) fun _ => ?_
  refine RF.bind (zerocheck_rf _) fun ⟨r0, zc⟩ ⟨r0', zc'⟩ _ =>
    RF.bind (lincheck_rf _ zc zc') fun ⟨r1, lc⟩ ⟨r1', lc'⟩ _ => ?_
  refine RF.bind (ringSwitch_rf (by simp [claimsOf, hx])) fun ⟨rsr, rs⟩ ⟨rsr', rs'⟩ _ => ?_
  exact RF.bind (ligerito_rf _ _ _ _ _) fun _ _ _ => RF.pure' trivial

end Run

/-- **`run` reads a record's rounds only through their sizes**, and the extra claims only through their number
(RecAlgebra.lean:810–811, 532–539). -/
theorem residualLins_sizes {sh : Shape} {r r' : Array (Array F128)} {extras extras' : Array Claim}
    {p : Array Lin × Info Lin} (hr : SameSizes r r') (hx : extras.size = extras'.size)
    (h : residualLins sh (Coins.ofRounds r) extras = .ok p) :
    ∃ p', residualLins sh (Coins.ofRounds r') extras' = .ok p' := by
  obtain ⟨j, hj⟩ := residualLins_run h
  obtain ⟨p', hp', -⟩ := run_rf (M := symbolic sh) sh hx r r' hr 0 p j hj
  refine ⟨p', ?_⟩
  unfold residualLins
  change (Prod.fst <$> run sh (symbolic sh) extras' 0 (Coins.ofRounds r')) = _
  rw [hp']
  rfl

/-- **`verifierRows` reads a record's rounds only through their sizes** (RecAlgebra.lean:824–829): it returns at a
record, extra claims and comb exactly when it does at any record whose rounds hold as many coins each, as many extra
claims and any comb. -/
theorem verifierRows_sizes {sh : Shape} {r r' : Array (Array F128)} {extras extras' : Array Claim}
    {comb comb' : Array F128} (hr : SameSizes r r') (hx : extras.size = extras'.size) :
    (∃ x, verifierRows sh r extras comb = .ok x) ↔ ∃ x, verifierRows sh r' extras' comb' = .ok x := by
  rw [verifierRows_ok_iff, verifierRows_ok_iff]
  constructor
  · rintro ⟨hS, p, hp⟩
    exact ⟨hS, residualLins_sizes hr hx hp⟩
  · rintro ⟨hS, p, hp⟩
    exact ⟨hS, residualLins_sizes (r := r') (r' := r) hr.symm hx.symm hp⟩

end Flock.RecAlgebra
