import Proofs.Flock.VBridge.Algebra.Build

/-!
# When `verifierRows` returns

`verifierRows sh rounds extras comb` (RecAlgebra.lean:824–829) builds S from the symbolic residuals at the rep's coins
and refuses unless it is `structureOf sh`'s, built at coins all zero. `build` reads a residual's elements, in order,
whether it has a constant, and its products (with their coefficients, which must be 1); the elements' coefficients
and the constant's value go into v and `kappa`, never into S. So S is the same at two lists of residuals of the same
skeleton (`Lin.Skel`, `build_skel`), and `verifierRows` returns, with v the coefficient slots then the comb, whenever
`run` at the rep's coins gives residuals of the zero coins' skeleton (`verifierRows_ok_of_skel`).
-/

namespace Flock.RecAlgebra

open Flock

/-- Two symbolic values with the same elements in the same order, a constant alike present, and the same products. -/
def Lin.Skel (f f' : Lin) : Prop := f.t.map Prod.fst = f'.t.map Prod.fst ∧ f.c.isSome = f'.c.isSome ∧ f.q = f'.q

/-! ## Two `build` states alike but for v's values and `kappa`'s entries -/

/-- `kappa`'s shape: its entries' sizes. -/
def kShape (K : Array (Array (Array F128))) : Array (Array Nat) := K.map (·.map Array.size)

theorem amap_modify {α β : Type} {F : α → β} {g : α → α} (hg : ∀ a, F (g a) = F a) (xs : Array α) (i : Nat) :
    (xs.modify i g).map F = xs.map F := by
  apply Array.ext
  · simp
  · intro j h1 h2
    simp only [Array.getElem_map, Array.getElem_modify]
    split <;> simp [hg]

theorem kShape_modify (K : Array (Array (Array F128))) (level r l : Nat) (x : F128) :
    kShape (K.modify level (·.modify r (·.modify l (· + x)))) = kShape K := by
  unfold kShape
  exact amap_modify (fun a => amap_modify (fun b => Array.size_modify) a r) K level

theorem kcond_iff (K : Array (Array (Array F128))) (level r l : Nat) :
    (level < K.size ∧ r < K[level]!.size ∧ l < (K[level]!)[r]!.size) ↔
      (level < (kShape K).size ∧ r < (kShape K)[level]!.size ∧ l < ((kShape K)[level]!)[r]!) := by
  have e1 : (kShape K)[level]! = K[level]!.map Array.size :=
    getElem!_map_of (fun a : Array (Array F128) => a.map Array.size) (by show Array.map Array.size #[] = #[]; simp) K
      level
  have e2 : (K[level]!.map Array.size)[r]! = (K[level]!)[r]!.size :=
    getElem!_map_of Array.size rfl K[level]! r
  rw [e1, e2, kShape, Array.size_map, Array.size_map]

/-- Two `build` states alike: the same operands, operand table and v names, as many v values, `kappa` of one shape. -/
structure BRel (st st' : BSt) : Prop where
  ops : st.ops = st'.ops
  opOf : st.opOf = st'.opOf
  vNames : st.vNames = st'.vNames
  vVals : st.vVals.size = st'.vVals.size
  kappa : kShape st.kappa = kShape st'.kappa

theorem BRel.refl (st : BSt) : BRel st st := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `y` is `x` at an alike state: whenever `x` succeeds, `y` succeeds from an alike state with a related result and an
alike state. -/
def BSim {A B : Type} (R : A → B → Prop) (x : BM A) (y : BM B) : Prop :=
  ∀ st st' a st₁, BRel st st' → x st = .ok (a, st₁) → ∃ b st₁', y st' = .ok (b, st₁') ∧ R a b ∧ BRel st₁ st₁'

namespace BSim

variable {A B C D : Type}

theorem bind {R : A → B → Prop} {R' : C → D → Prop} {x : BM A} {y : BM B} {f : A → BM C} {g : B → BM D}
    (hs : BSim R x y) (h : ∀ a b, R a b → BSim R' (f a) (g b)) : BSim R' (x >>= f) (y >>= g) := by
  intro st st' c st₂ hrel hx
  obtain ⟨a, st₁, hx₁, hx₂⟩ := bm_bind hx
  obtain ⟨b, st₁', hy₁, hab, hrel₁⟩ := hs st st' a st₁ hrel hx₁
  obtain ⟨d, st₂', hy₂, hcd, hrel₂⟩ := h a b hab st₁ st₁' c st₂ hrel₁ hx₂
  refine ⟨d, st₂', ?_, hcd, hrel₂⟩
  have e : (y >>= g) st' = (y st').bind (fun p => g p.1 p.2) := rfl
  rw [e, hy₁]
  exact hy₂

theorem pure' {R : A → B → Prop} {a : A} {b : B} (h : R a b) : BSim R (Pure.pure a) (Pure.pure b) := by
  intro st st' a' st₁ hrel hx
  cases hx
  exact ⟨b, st', rfl, h, hrel⟩

theorem throw' {R : A → B → Prop} {y : BM B} (e : String) : BSim R (throw e) y := by
  intro st st' a st₁ hrel hx
  cases hx

theorem throw_bind' {R : C → D → Prop} {f : A → BM C} {y : BM D} (e : String) : BSim R (throw e >>= f) y := by
  intro st st' a st₁ hrel hx
  cases hx

theorem get' : BSim BRel (get : BM BSt) (get : BM BSt) := by
  intro st st' a st₁ hrel hx
  cases hx
  exact ⟨st', st', rfl, hrel, hrel⟩

theorem set' {s s' : BSt} (h : BRel s s') : BSim (fun _ _ => True) (set s : BM PUnit) (set s' : BM PUnit) := by
  intro st st' a st₁ hrel hx
  cases hx
  exact ⟨PUnit.unit, s', rfl, trivial, h⟩

theorem liftB' {R : A → B → Prop} {e : Except String A} {e' : Except String B}
    (h : ∀ a, e = .ok a → ∃ b, e' = .ok b ∧ R a b) : BSim R (liftB e) (liftB e') := by
  intro st st' a st₁ hrel hx
  obtain ⟨he, rfl⟩ := liftB_ok hx
  obtain ⟨b, hb, hab⟩ := h a he
  subst hb
  exact ⟨b, st', rfl, hab, hrel⟩

theorem ite_iff {R : A → B → Prop} {c c' : Prop} [Decidable c] [Decidable c'] {x₁ x₂ : BM A} {y₁ y₂ : BM B}
    (hc : c ↔ c') (h₁ : c → BSim R x₁ y₁) (h₂ : ¬c → BSim R x₂ y₂) :
    BSim R (if c then x₁ else x₂) (if c' then y₁ else y₂) := by
  by_cases h : c
  · have h' : c' := hc.mp h
    simp only [h, h', ↓reduceIte]
    exact h₁ h
  · have h' : ¬c' := fun h' => h (hc.mpr h')
    simp only [h, h', ↓reduceIte]
    exact h₂ h

end BSim

/-! ## `build`'s passes at alike states -/

theorem filter_keys {p : Key × F128 → Bool} (hp : ∀ a b : Key × F128, a.1 = b.1 → p a = p b) :
    ∀ {l l' : List (Key × F128)}, l.map Prod.fst = l'.map Prod.fst →
      (l.filter p).map Prod.fst = (l'.filter p).map Prod.fst
  | [], [], _ => rfl
  | (k, x) :: l, (k', x') :: l', h => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨rfl, h⟩ := h
    have e : p (k, x') = p (k, x) := hp _ _ rfl
    by_cases hq : p (k, x) <;> simp [hq, e, filter_keys hp h]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

theorem mapM_keys {g : Key × F128 → Except String (Nat × (Key × F128))}
    (hg : ∀ x x' p, x.1 = x'.1 → g x = .ok p → ∃ p', g x' = .ok p' ∧ (p.1, p.2.1) = (p'.1, p'.2.1)) :
    ∀ {xs xs' : List (Key × F128)} {keyed : List (Nat × (Key × F128))}, xs.map Prod.fst = xs'.map Prod.fst →
      xs.mapM g = .ok keyed → ∃ keyed', xs'.mapM g = .ok keyed' ∧
        keyed.map (fun p => (p.1, p.2.1)) = keyed'.map (fun p => (p.1, p.2.1))
  | [], [], _, _, hm => by
    cases hm
    exact ⟨[], rfl, rfl⟩
  | x :: xs, x' :: xs', keyed, h, hm => by
    simp only [List.map_cons, List.cons.injEq] at h
    rw [List.mapM_cons] at hm ⊢
    cases hx : g x with
    | error e => rw [hx] at hm; cases hm
    | ok p =>
      cases hm' : xs.mapM g with
      | error e => rw [hx, hm'] at hm; cases hm
      | ok ks =>
        rw [hx, hm'] at hm
        cases hm
        obtain ⟨p', hp', hpp⟩ := hg x x' p h.1 hx
        obtain ⟨ks', hks', hk⟩ := mapM_keys hg h.2 hm'
        refine ⟨p' :: ks', by rw [hp', hks']; rfl, ?_⟩
        simp only [List.map_cons, List.cons.injEq]
        exact ⟨hpp, hk⟩
  | [], _ :: _, _, h, _ => by simp at h
  | _ :: _, [], _, h, _ => by simp at h

/-- Sorting by an order of the elements keeps two lists' elements alike. -/
theorem sortBy_keys {o : Key × F128 → Except String Nat} (ho : ∀ x x' : Key × F128, x.1 = x'.1 → o x = o x')
    {xs xs' ys : List (Key × F128)} (h : xs.map Prod.fst = xs'.map Prod.fst) (hs : sortBy o xs = .ok ys) :
    ∃ ys', sortBy o xs' = .ok ys' ∧ ys.map Prod.fst = ys'.map Prod.fst := by
  unfold sortBy at hs ⊢
  have hg : ∀ x x' p, x.1 = x'.1 → (fun x => (do return ((← o x), x) : Except String (Nat × (Key × F128)))) x =
      .ok p → ∃ p', (fun x => (do return ((← o x), x) : Except String (Nat × (Key × F128)))) x' = .ok p' ∧
        (p.1, p.2.1) = (p'.1, p'.2.1) := by
    intro x x' p hx hp
    dsimp only at hp ⊢
    cases hox : o x with
    | error e => rw [hox] at hp; cases hp
    | ok n =>
      rw [hox] at hp
      cases hp
      exact ⟨(n, x'), by rw [← ho x x' hx, hox]; rfl, by simp [hx]⟩
  cases hm : xs.mapM (fun x => (do return ((← o x), x) : Except String (Nat × (Key × F128)))) with
  | error e => rw [hm] at hs; cases hs
  | ok keyed =>
    rw [hm] at hs
    cases hs
    obtain ⟨keyed', hm', hk⟩ := mapM_keys hg h hm
    rw [hm']
    refine ⟨_, rfl, ?_⟩
    have e : ∀ l : List (Nat × (Key × F128)),
        ((l.mergeSort fun a b => decide (a.1 ≤ b.1)).map (·.2)).map Prod.fst =
          ((l.map fun p => (p.1, p.2.1)).mergeSort fun a b => decide (a.1 ≤ b.1)).map Prod.snd := by
      intro l
      have h1 : (l.mergeSort fun a b => decide (a.1 ≤ b.1)).map (fun p => (p.1, p.2.1)) =
          (l.map fun p => (p.1, p.2.1)).mergeSort fun a b => decide (a.1 ≤ b.1) :=
        List.map_mergeSort fun _ _ _ _ => rfl
      rw [← h1, List.map_map, List.map_map]
      rfl
    rw [e, e, hk]

section Build

variable {sh : Shape} {comb0 : Nat}

theorem varOp_sim (k : Key) : BSim Eq (varOp sh comb0 k) (varOp sh comb0 k) := by
  unfold varOp
  refine BSim.bind BSim.get' fun st st' hrel => ?_
  rw [← hrel.opOf]
  cases st.opOf.lookup k with
  | some i => exact BSim.pure' rfl
  | none =>
    refine BSim.bind (BSim.liftB' fun t ht => ⟨t, ht, rfl⟩) fun t t' (ht : t = t') => ?_
    subst ht
    refine BSim.bind (BSim.set' ?_) fun _ _ _ => BSim.pure' (by rw [hrel.ops])
    exact ⟨by simp [hrel.ops], by simp [hrel.ops], hrel.vNames, hrel.vVals, hrel.kappa⟩

theorem coefOps_sim (ri : Nat) : ∀ (l l' : List (Key × F128)), l.map Prod.fst = l'.map Prod.fst →
    BSim Eq (coefOps sh comb0 ri l) (coefOps sh comb0 ri l')
  | [], [], _ => BSim.pure' rfl
  | (k, x) :: rest, (k', x') :: rest', h => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨rfl, h⟩ := h
    simp only [coefOps]
    refine BSim.bind BSim.get' fun st st' hrel => ?_
    refine BSim.bind (BSim.set' ?_) fun _ _ _ => ?_
    · exact ⟨by simp [hrel.ops, hrel.vNames], hrel.opOf, by simp [hrel.vNames], by simp [hrel.vVals], hrel.kappa⟩
    refine BSim.bind (varOp_sim _) fun o o' (ho : o = o') => ?_
    exact BSim.bind (coefOps_sim ri rest rest' h) fun ps ps' (hp : ps = ps') =>
      BSim.pure' (by rw [hrel.ops, ho, hp])
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

theorem prodOps_sim (ri : Nat) : ∀ l : List ((Key × Key) × F128),
    BSim Eq (prodOps sh comb0 ri l) (prodOps sh comb0 ri l)
  | [] => BSim.pure' rfl
  | ((a, b), x) :: rest => by
    simp only [prodOps]
    refine BSim.ite_iff Iff.rfl (fun _ => BSim.throw_bind' _) fun _ => ?_
    refine BSim.bind (varOp_sim a) fun oa oa' (h1 : oa = oa') => ?_
    refine BSim.bind (varOp_sim b) fun ob ob' (h2 : ob = ob') => ?_
    exact BSim.bind (prodOps_sim ri rest) fun ps ps' (h3 : ps = ps') => BSim.pure' (by rw [h1, h2, h3])

theorem kapAdd_sim (r : Nat) : ∀ (l l' : List (Key × F128)), l.map Prod.fst = l'.map Prod.fst →
    BSim (fun _ _ => True) (kapAdd r l) (kapAdd r l')
  | [], [], _ => BSim.pure' trivial
  | (k, x) :: rest, (k', x') :: rest', h => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨rfl, h⟩ := h
    simp only [kapAdd]
    cases k
    case cons level l =>
      refine BSim.bind BSim.get' fun st st' hrel => ?_
      refine BSim.ite_iff (by rw [kcond_iff st.kappa, kcond_iff st'.kappa, hrel.kappa]) (fun _ => ?_) fun _ => ?_
      · refine BSim.bind (BSim.set' ?_) fun _ _ _ => kapAdd_sim r rest rest' h
        exact ⟨hrel.ops, hrel.opOf, hrel.vNames, hrel.vVals, by
          dsimp only
          rw [kShape_modify, kShape_modify, hrel.kappa]⟩
      · exact BSim.throw' _
    all_goals exact kapAdd_sim r rest rest' h
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

theorem accTerms_sim {fin ri : Nat} {l l' : List (Key × F128)} (h : l.map Prod.fst = l'.map Prod.fst) :
    BSim Eq (accTerms sh fin ri l) (accTerms sh fin ri l') := by
  have he : l.isEmpty = l'.isEmpty := by
    cases l <;> cases l' <;> simp_all
  unfold accTerms
  refine BSim.ite_iff (by rw [he]) (fun _ => BSim.pure' rfl) fun _ =>
    BSim.ite_iff Iff.rfl (fun _ => BSim.throw' _) fun _ => ?_
  refine BSim.bind (kapAdd_sim (ri - fin) l l' h) fun _ _ _ => ?_
  cases wi sh (.acc1 (ri - fin)) <;> cases wi sh (.acc0 (ri - fin)) <;>
    first | exact BSim.pure' rfl | exact BSim.throw' _

theorem constTerm_sim {ri : Nat} {c c' : Option F128} (h : c.isSome = c'.isSome) :
    BSim Eq (constTerm ri c) (constTerm ri c') := by
  cases c <;> cases c' <;> simp at h
  · exact BSim.pure' rfl
  · simp only [constTerm]
    refine BSim.bind BSim.get' fun st st' hrel => BSim.bind (BSim.set' ?_) fun _ _ _ =>
      BSim.pure' (by rw [hrel.vNames])
    exact ⟨hrel.ops, hrel.opOf, by simp [hrel.vNames], by simp [hrel.vVals], hrel.kappa⟩

theorem buildRes_sim {fin ri : Nat} {f f' : Lin} (hs : Lin.Skel f f') :
    BSim Eq (buildRes sh fin comb0 ri f) (buildRes sh fin comb0 ri f') := by
  unfold buildRes
  refine BSim.bind (accTerms_sim (filter_keys (fun a b h => by simp only [h]) hs.1)) fun acc acc' (h1 : acc = acc') =>
    ?_
  refine BSim.bind (R := fun l l' : List (Key × F128) => l.map Prod.fst = l'.map Prod.fst)
    (BSim.liftB' fun l hl =>
      sortBy_keys (fun x x' hx => by simp only [hx]) (filter_keys (fun a b h => by simp only [h]) hs.1) hl)
    fun l1 l1' h2 => ?_
  refine BSim.bind (coefOps_sim ri l1 l1' h2) fun cm cm' (h3 : cm = cm') => ?_
  refine BSim.bind (BSim.liftB' fun l hl => ⟨l, by rw [← hs.2.2]; exact hl, rfl⟩) fun l2 l2' (h4 : l2 = l2') => ?_
  subst h4
  refine BSim.bind (prodOps_sim ri l2) fun pm pm' (h5 : pm = pm') => ?_
  exact BSim.bind (constTerm_sim hs.2.1) fun cp cp' (h6 : cp = cp') => BSim.pure' (by rw [h1, h3, h5, h6])

theorem buildAll_sim {fin : Nat} {fs fs' : List Lin} (hs : List.Forall₂ Lin.Skel fs fs') :
    ∀ ri, BSim Eq (buildAll sh fin comb0 fs ri) (buildAll sh fin comb0 fs' ri) := by
  induction hs with
  | nil => intro _; exact BSim.pure' rfl
  | cons hf _ ih =>
    intro ri
    simp only [buildAll]
    exact BSim.bind (buildRes_sim hf) fun r r' (h1 : r = r') =>
      BSim.bind (ih (ri + 1)) fun rs rs' (h2 : rs = rs') => BSim.pure' (by rw [h1, h2])

end Build

theorem slotsOf_skel {f f' : Lin} (h : Lin.Skel f f') : slotsOf f = slotsOf f' := by
  have hl := congrArg List.length (filter_keys (p := fun x => !x.1.isCons) (fun a b h => by simp only [h]) h.1)
  simp only [List.length_map] at hl
  unfold slotsOf
  rw [hl, h.2.1]

theorem slots_skel {fs fs' : List Lin} (hs : List.Forall₂ Lin.Skel fs fs') : fs.map slotsOf = fs'.map slotsOf := by
  induction hs with
  | nil => rfl
  | cons hf _ ih => rw [List.map_cons, List.map_cons, slotsOf_skel hf, ih]

/-- `build`'s S at residuals of one skeleton is the same, and `build` succeeds at both or neither. -/
theorem build_skel {sh : Shape} {lins lins' : Array Lin} {b : Built}
    (hs : List.Forall₂ Lin.Skel lins.toList lins'.toList) (hb : build sh lins = .ok b) :
    ∃ b', build sh lins' = .ok b' ∧ b'.S = b.S := by
  have hc : (lins'.toList.map slotsOf).sum = (lins.toList.map slotsOf).sum := by rw [slots_skel hs]
  unfold build at hb ⊢
  rw [hc]
  generalize (lins.toList.map slotsOf).sum = comb0 at hb ⊢
  generalize sh.sched.levels.map (fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) = K at hb ⊢
  simp only at hb ⊢
  cases hB : (buildAll sh (2 + sh.claims) comb0 lins.toList 0).run { kappa := K } with
  | error e => rw [hB] at hb; cases hb
  | ok p =>
    obtain ⟨res, st⟩ := p
    rw [hB] at hb
    simp only [bind, Except.bind] at hb
    have hB' : buildAll sh (2 + sh.claims) comb0 lins.toList 0 { kappa := K } = .ok (res, st) := hB
    obtain ⟨res', st', hB2, hres, hrel⟩ :=
      buildAll_sim (sh := sh) (comb0 := comb0) (fin := 2 + sh.claims) hs 0 _ _ _ _ (BRel.refl _) hB'
    have hB2' : (buildAll sh (2 + sh.claims) comb0 lins'.toList 0).run { kappa := K } = .ok (res', st') := hB2
    rw [hB2']
    simp only [bind, Except.bind]
    split at hb
    · cases hb
    · rename_i hn
      cases hb
      have hcond : (st'.vNames.size ≠ comb0 ∨ st'.vVals.size ≠ comb0) = False := by
        rw [← hrel.vNames, ← hrel.vVals]
        exact eq_false hn
      simp only [hcond, ↓reduceIte]
      refine ⟨_, rfl, ?_⟩
      rw [← hrel.vNames, ← hrel.ops, ← hres]
      try rfl

/-- **`verifierRows` returns** when `run` at the rep's coins gives residuals of the skeleton those at coins all zero
have (the ones `structureOf` builds S from): then its S is `structureOf`'s, and v is the coefficient slots, then the
comb. -/
theorem verifierRows_ok_of_skel {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb : Array F128}
    {lins₀ lins : Array Lin} {info₀ info : Info Lin} {S : GfS} (hS : structureOf sh = .ok S)
    (hz : residualLins sh CoinSource.zeros (extrasLike sh) = .ok (lins₀, info₀))
    (hl : residualLins sh (CoinSource.ofRounds rounds) extras = .ok (lins, info))
    (hs : List.Forall₂ Lin.Skel lins₀.toList lins.toList) :
    ∃ b : Built, build sh lins = .ok b ∧ b.S = S ∧
      verifierRows sh rounds extras comb = .ok (b.vVals ++ comb, info, b.kappa) := by
  have hS' := hS
  unfold structureOf at hS
  rw [hz] at hS
  simp only [bind, Except.bind] at hS
  cases hb₀ : build sh lins₀ with
  | error e => rw [hb₀] at hS; cases hS
  | ok b₀ =>
    rw [hb₀] at hS
    (try simp only at hS)
    cases hS
    obtain ⟨b, hb, hbS⟩ := build_skel hs hb₀
    refine ⟨b, hb, hbS, ?_⟩
    unfold verifierRows
    rw [hl]
    simp only [bind, Except.bind]
    rw [hb]
    simp only [hS', Except.toOption, hbS, ne_eq, not_true_eq_false, ↓reduceIte]
    try rfl

end Flock.RecAlgebra
