import Proofs.Flock.VBridge.Algebra.Sim

/-!
# What `_build`'s passes compute

`buildAll_spec`: `build`'s passes over the symbolic residuals give residuals whose value (`resValue`), on any S and v
extending the passes' operands and coefficient slots, is each symbolic residual's terms but the consistency sums
evaluated (`noCons`), plus the running sums w holds where it has consistency sums (`resForm`); the passes add each
residual's coefficients on the consistency sums into `kappa`, and only Ligerito's final check, residuals `fin` and
`fin + 1`, has any.

* `valOf` is a message element's value: its element of w, its ring switch's transpose of w, the comb's element (at a
  holo shape plus w's `h`, `hOf`, as `atomOp` reads it), or the level's consistency sum.
* `kapSum kappa r cons` is the final check's component `r` on the levels' consistency sums `cons`.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-! ## Values -/

/-- A holo shape's `h[l]` as w holds it, the comb operand's second term (`atomOp`); zero off a holo shape. -/
def hOf (sh : Shape) (w : Array F128) (l : Nat) : F128 :=
  if sh.holo then match wi sh (.h l) with
    | some i => w[i]!
    | none => 0
  else 0

/-- A message element's value at w, the comb and the levels' consistency sums. -/
def valOf (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) : Key → F128
  | .rsT k u => match wi sh (.rs k 0) with
    | some i => termValue w comb (.wt i u)
    | none => 0
  | .comb l => comb[l]! + hOf sh w l
  | .cons level l => cons level l
  | k => match wi sh k with
    | some i => w[i]!
    | none => 0

theorem hOf_plain {sh : Shape} (hp : sh.holo = false) (w : Array F128) (l : Nat) : hOf sh w l = 0 := by
  simp [hOf, hp]

theorem hOf_holo {sh : Shape} (hh : sh.holo = true) (w : Array F128) {l i : Nat} (hi : wi sh (.h l) = some i) :
    hOf sh w l = w[i]! := by
  simp [hOf, hh, hi]

/-- `Σ_{i < n} f i`. -/
def rangeSum (n : Nat) (f : Nat → F128) : F128 := (List.range n).foldl (fun acc i => acc + f i) 0

/-- `Σ_level Σ_l κ[level][r][l] · cons level l`: the final check's component `r` on the consistency sums. -/
def kapSum (kappa : Array (Array (Array F128))) (r : Nat) (cons : Nat → Nat → F128) : F128 :=
  rangeSum kappa.size fun level => rangeSum (kappa[level]!)[r]!.size fun l => ((kappa[level]!)[r]!)[l]! * cons level l

/-- A symbolic value's terms on the consistency sums. -/
def consOf (f : Lin) : Terms Key := f.t.filter (·.1.isCons)

/-- A symbolic value but its terms on the consistency sums, at `val`. -/
def noCons (val : Key → F128) (f : Lin) : F128 :=
  tEval val (f.t.filter (!·.1.isCons)) + f.c.getD 0 + tEval (fun ab => val ab.1 * val ab.2) f.q

/-! ## Sums -/

theorem f128_add_zero (x : F128) : x + 0 = x := by
  (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem f128_zero_add (x : F128) : 0 + x = x := by
  (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem foldl_shift {γ : Type} (g : γ → F128) :
    ∀ (l : List γ) (c : F128), l.foldl (fun x a => x + g a) c = c + l.foldl (fun x a => x + g a) 0
  | [], c => by simp only [List.foldl_nil]; (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
  | a :: l, c => by
    simp only [List.foldl_cons]
    rw [foldl_shift g l (c + g a), foldl_shift g l (0 + g a)]
    (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem afoldl_shift {γ : Type} (g : γ → F128) (xs : Array γ) (c : F128) :
    xs.foldl (fun x a => x + g a) c = c + xs.foldl (fun x a => x + g a) 0 := by
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  exact foldl_shift g _ c

theorem afoldl_append {γ : Type} (g : γ → F128) (xs ys : Array γ) :
    (xs ++ ys).foldl (fun x a => x + g a) 0 = xs.foldl (fun x a => x + g a) 0 + ys.foldl (fun x a => x + g a) 0 := by
  rw [Array.foldl_append, afoldl_shift]

theorem afoldl_single {γ : Type} (g : γ → F128) (a : γ) : (#[a] : Array γ).foldl (fun x a => x + g a) 0 = g a := by
  simp only [List.foldl_toArray', List.foldl_cons, List.foldl_nil]
  (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem afoldl_empty {γ : Type} (g : γ → F128) : (#[] : Array γ).foldl (fun x a => x + g a) 0 = 0 := rfl

theorem tEval_append {κ : Type} (g : κ → F128) :
    ∀ (a b : Terms κ), tEval g (a ++ b) = tEval g a + tEval g b
  | [], b => by
    simp only [List.nil_append, tEval]
    (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
  | (k, x) :: r, b => by
    simp only [List.cons_append, tEval, tEval_append g r b]
    (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)

theorem tEval_perm {κ : Type} (g : κ → F128) {a b : Terms κ} (hp : a.Perm b) : tEval g a = tEval g b := by
  induction hp with
  | nil => rfl
  | cons x _ ih => obtain ⟨k, v⟩ := x; simp only [tEval, ih]
  | swap x y l =>
    obtain ⟨k, v⟩ := x; obtain ⟨k', v'⟩ := y; simp only [tEval]
    (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)
  | trans _ _ ih₁ ih₂ => rw [ih₁, ih₂]

theorem tEval_congr {κ : Type} {g g' : κ → F128} :
    ∀ (l : Terms κ), (∀ kx ∈ l, g kx.1 = g' kx.1) → tEval g l = tEval g' l
  | [], _ => rfl
  | (k, x) :: r, hg => by
    simp only [tEval]
    rw [hg (k, x) (by simp), tEval_congr r fun kx hk => hg kx (by simp [hk])]

theorem lin_eval_split (val : Key → F128) (f : Lin) : f.eval val = noCons val f + tEval val (consOf f) := by
  have hp := List.filter_append_perm (fun kx : Key × F128 => kx.1.isCons) f.t
  rw [Lin.eval, noCons, consOf, ← tEval_perm val hp, tEval_append]
  (apply fOf_injective; simp only [fOf_add]; ring)

theorem rangeSum_congr {n : Nat} {f g : Nat → F128} (hfg : ∀ i < n, f i = g i) : rangeSum n f = rangeSum n g := by
  unfold rangeSum
  apply List.foldl_ext
  intro acc i hi
  rw [hfg i (List.mem_range.mp hi)]

theorem rangeSum_single {n i₀ : Nat} {f g : Nat → F128} (d : F128) (hi : i₀ < n) (hne : ∀ i < n, i ≠ i₀ → f i = g i)
    (h₀ : f i₀ = g i₀ + d) : rangeSum n f = rangeSum n g + d := by
  induction n with
  | zero => omega
  | succ n ih =>
    unfold rangeSum
    rw [List.range_succ, List.foldl_append, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    by_cases hn : i₀ = n
    · subst hn
      have : rangeSum i₀ f = rangeSum i₀ g := rangeSum_congr fun i hi => hne i (by omega) (by omega)
      unfold rangeSum at this
      rw [this, h₀]
      (apply fOf_injective; simp only [fOf_add]; ring)
    · have := ih (by omega) fun i hi hi' => hne i (by omega) hi'
      unfold rangeSum at this
      rw [this, hne n (by omega) (Ne.symm hn)]
      (apply fOf_injective; simp only [fOf_add]; ring)

theorem rangeSum_zero {n : Nat} {f : Nat → F128} (hf : ∀ i < n, f i = 0) : rangeSum n f = 0 := by
  rw [rangeSum_congr (g := fun _ => 0) hf]
  clear hf
  unfold rangeSum
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, ih]; simp only [List.foldl_cons, List.foldl_nil]
    (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem getElem!_modify_eq {γ : Type} [Inhabited γ] (xs : Array γ) (i : Nat) (g : γ → γ) (hi : i < xs.size) :
    (xs.modify i g)[i]! = g xs[i]! := by
  simp only [getElem!_def, Array.getElem?_modify, ite_true, Array.getElem?_eq_getElem hi, Option.map_some]

theorem getElem!_modify_ne {γ : Type} [Inhabited γ] (xs : Array γ) {i j : Nat} (g : γ → γ) (hij : i ≠ j) :
    (xs.modify i g)[j]! = xs[j]! := by
  simp only [getElem!_def, Array.getElem?_modify, hij, ite_false]

/-- A consistency sum added into `kappa`: component `r` gains `x · cons level l`, the others are unchanged. -/
theorem kapSum_modify (K : Array (Array (Array F128))) (cons : Nat → Nat → F128) {level r l : Nat} (x : F128)
    (h1 : level < K.size) (h2 : r < K[level]!.size) (h3 : l < ((K[level]!)[r]!).size) (r' : Nat) :
    kapSum (K.modify level (·.modify r (·.modify l (· + x)))) r' cons =
      kapSum K r' cons + (if r' = r then x * cons level l else 0) := by
  unfold kapSum
  rw [Array.size_modify]
  by_cases hr : r' = r
  · subst hr
    simp only [ite_true]
    refine rangeSum_single _ h1 (fun i _ hi => ?_) ?_
    · rw [getElem!_modify_ne _ _ (Ne.symm hi)]
    · rw [getElem!_modify_eq _ _ _ h1, getElem!_modify_eq _ _ _ h2, Array.size_modify]
      refine rangeSum_single _ h3 (fun i _ hi => ?_) ?_
      · rw [getElem!_modify_ne _ _ (Ne.symm hi)]
      · rw [getElem!_modify_eq _ _ _ h3]; (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)
  · simp only [hr, ite_false, f128_add_zero]
    apply rangeSum_congr
    intro i _
    by_cases hi : level = i
    · subst hi
      rw [getElem!_modify_eq _ _ _ h1, getElem!_modify_ne _ _ (Ne.symm hr)]
    · rw [getElem!_modify_ne _ _ hi]

/-! ## `_build`'s state -/

abbrev BM := StateT BSt (Except String)

theorem liftB_ok {A : Type} {e : Except String A} {st st' : BSt} {a : A} (h : liftB e st = .ok (a, st')) :
    e = .ok a ∧ st' = st := by
  cases e with
  | error e => cases h
  | ok a' => cases h; exact ⟨rfl, rfl⟩

theorem bm_bind {A B : Type} {x : BM A} {f : A → BM B} {st : BSt} {r : B × BSt}
    (h : (x >>= f) st = .ok r) : ∃ a st', x st = .ok (a, st') ∧ f a st' = .ok r := by
  have e : (x >>= f) st = (x st).bind (fun p => f p.1 p.2) := rfl
  rw [e] at h
  cases hx : x st with
  | error e => rw [hx] at h; cases h
  | ok p => rw [hx] at h; exact ⟨p.1, p.2, rfl, h⟩

section Build

variable (sh : Shape) (comb0 : Nat)

/-- The op table's operands for message elements: each is the element's operand (`atomOp`); v's names and values
align. -/
def OpsOk (st : BSt) : Prop :=
  st.vNames.size = st.vVals.size ∧
    ∀ k i, st.opOf.lookup k = some i → ∃ op, atomOp sh comb0 k = .ok op ∧ st.ops[i]? = some op

/-- Operands `F` and v values `V` extend `st`'s. -/
def ExtTo (st : BSt) (F : Array (Array Term)) (V : Array F128) : Prop :=
  (∃ o, F = st.ops ++ o) ∧ ∃ vv, V = st.vVals ++ vv

/-- `st'` extends `st`. -/
def Ext (st st' : BSt) : Prop := ExtTo st st'.ops st'.vVals

variable {sh comb0}

theorem Ext.refl (st : BSt) : Ext st st := ⟨⟨#[], by simp⟩, ⟨#[], by simp⟩⟩

theorem Ext.trans {a b : BSt} {F : Array (Array Term)} {V : Array F128} (hab : Ext a b) (hb : ExtTo b F V) :
    ExtTo a F V := by
  obtain ⟨⟨o1, h1⟩, ⟨v1, h2⟩⟩ := hab
  obtain ⟨⟨o2, h3⟩, ⟨v2, h4⟩⟩ := hb
  exact ⟨⟨o1 ++ o2, by rw [h3, h1, Array.append_assoc]⟩, ⟨v1 ++ v2, by rw [h4, h2, Array.append_assoc]⟩⟩

theorem ExtTo.ops? {a : BSt} {F : Array (Array Term)} {V : Array F128} (hab : ExtTo a F V) {i : Nat}
    {x : Array Term} (hi : a.ops[i]? = some x) : F[i]? = some x := by
  obtain ⟨⟨o, h⟩, _⟩ := hab
  have hlt : i < a.ops.size := by
    rcases Nat.lt_or_ge i a.ops.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none h'] at hi; cases hi
  rw [h, Array.getElem?_append_left hlt, hi]

theorem ExtTo.vVals? {a : BSt} {F : Array (Array Term)} {V : Array F128} (hab : ExtTo a F V) {j : Nat} {x : F128}
    (hj : a.vVals[j]? = some x) : V[j]? = some x := by
  obtain ⟨_, ⟨vv, h⟩⟩ := hab
  have hlt : j < a.vVals.size := by
    rcases Nat.lt_or_ge j a.vVals.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none h'] at hj; cases hj
  rw [h, Array.getElem?_append_left hlt, hj]

theorem Ext.push (st : BSt) (n : VName) (x : F128) (op : Array Term) :
    Ext st { st with vNames := st.vNames.push n, vVals := st.vVals.push x, ops := st.ops.push op } :=
  ⟨⟨#[op], Array.push_eq_append⟩, ⟨#[x], Array.push_eq_append⟩⟩

theorem OpsOk.push {st : BSt} (hi : OpsOk sh comb0 st) (n : VName) (x : F128) (op : Array Term) :
    OpsOk sh comb0 { st with vNames := st.vNames.push n, vVals := st.vVals.push x, ops := st.ops.push op } := by
  refine ⟨by simp [hi.1], fun k i hk => ?_⟩
  obtain ⟨t, ht, ho⟩ := hi.2 k i hk
  exact ⟨t, ht, (Ext.push st n x op).ops? ho⟩

variable (w comb : Array F128) (cons : Nat → Nat → F128)

theorem opValue_single {S : GfS} {o : Nat} {t : Term} (v : Array F128) (h : S.ops[o]? = some #[t]) :
    opValue S w v o = termValue w v t := by
  simp only [opValue, getElem!_def, h]
  exact afoldl_single _ _

/-- The products of operand pairs, summed. -/
def mulSum (S : GfS) (v : Array F128) (ps : Array (Nat × Nat)) : F128 :=
  ps.foldl (fun x ab => x + opValue S w v ab.1 * opValue S w v ab.2) 0

theorem mulSum_cons (S : GfS) (v : Array F128) (p : Nat × Nat) (ps : Array (Nat × Nat)) :
    mulSum w S v (#[p] ++ ps) = opValue S w v p.1 * opValue S w v p.2 + mulSum w S v ps := by
  simp only [mulSum, afoldl_append, afoldl_single]

/-- A message element's operand is its term, or at a holo shape a comb's term and its `h`. -/
theorem atomOp_cases {k : Key} {op : Array Term} (h : atomOp sh comb0 k = .ok op) :
    ∃ t, atomTerm sh comb0 k = .ok t ∧ (op = #[t] ∨ sh.holo = true ∧ ∃ i, op = #[t, .w i]) := by
  unfold atomOp at h
  cases ht : atomTerm sh comb0 k with
  | error e => rw [ht] at h; cases h
  | ok t =>
    rw [ht] at h
    refine ⟨t, rfl, ?_⟩
    simp only [bind, Except.bind] at h
    split at h
    · split at h
      · cases h
        exact .inr ⟨by assumption, _, rfl⟩
      · cases h
    · cases h
      exact .inl rfl

/-- Off a holo shape, a message element's operand is its term alone. -/
theorem atomOp_plain {k : Key} {op : Array Term} (hp : sh.holo = false) (h : atomOp sh comb0 k = .ok op) :
    ∃ t, atomTerm sh comb0 k = .ok t ∧ op = #[t] := by
  obtain ⟨t, ht, h1 | ⟨h2, -⟩⟩ := atomOp_cases h
  · exact ⟨t, ht, h1⟩
  · rw [hp] at h2; cases h2

/-- Off the comb, a message element's operand is its term alone. -/
theorem atomOp_single {k : Key} {op : Array Term} (hk : ∀ l, k ≠ .comb l) (h : atomOp sh comb0 k = .ok op) :
    ∃ t, atomTerm sh comb0 k = .ok t ∧ op = #[t] := by
  obtain ⟨t, ht, h1 | ⟨-, i, h2⟩⟩ := atomOp_cases h
  · exact ⟨t, ht, h1⟩
  · exfalso
    unfold atomOp at h
    rw [ht] at h
    simp only [bind, Except.bind] at h
    split at h
    · exact hk _ rfl
    · cases h
      simp at h2

/-- At a holo shape, a comb's operand is its slot and its `h`. -/
theorem atomOp_comb {l : Nat} {op : Array Term} (hh : sh.holo = true) (h : atomOp sh comb0 (.comb l) = .ok op) :
    ∃ i, wi sh (.h l) = some i ∧ op = #[.v (comb0 + l), .w i] := by
  unfold atomOp at h
  simp only [atomTerm, bind, Except.bind, hh] at h
  cases hw : wi sh (.h l) with
  | none => rw [hw] at h; cases h
  | some i => rw [hw] at h; cases h; exact ⟨i, rfl, rfl⟩

theorem atom_val {k : Key} {t : Term} (V : Array F128) (ha : atomTerm sh comb0 k = .ok t)
    (hk : ∀ l, k ≠ .comb l) : termValue w (V ++ comb) t = valOf sh w comb cons k := by
  cases k
  case comb l => exact absurd rfl (hk l)
  all_goals clear hk
  all_goals (simp only [atomTerm, valOf] at ha ⊢; (try split at ha) <;> (try cases ha) <;> (try simp_all [termValue]))

/-- A message element's operand sums to its value, at either shape. -/
theorem atomOp_val {k : Key} {op : Array Term} {V : Array F128} (h : atomOp sh comb0 k = .ok op)
    (hV : V.size = comb0) : op.foldl (fun x t => x + termValue w (V ++ comb) t) 0 = valOf sh w comb cons k := by
  by_cases hk : ∃ l, k = .comb l
  · obtain ⟨l, rfl⟩ := hk
    have hv : termValue w (V ++ comb) (.v (comb0 + l)) = comb[l]! := by
      rw [termValue, getElem!_def, getElem!_def, Array.getElem?_append_right (by omega),
        show comb0 + l - V.size = l from by omega]
    show _ = comb[l]! + hOf sh w l
    cases hh : sh.holo
    · obtain ⟨t, ht, rfl⟩ := atomOp_plain hh h
      simp only [atomTerm, Except.ok.injEq] at ht
      subst ht
      rw [afoldl_single, hv, hOf_plain hh, f128_add_zero]
    · obtain ⟨i, hi, rfl⟩ := atomOp_comb hh h
      simp only [List.foldl_toArray', List.foldl_cons, List.foldl_nil]
      rw [f128_zero_add, hv, hOf_holo hh w hi]
      rfl
  · obtain ⟨t, ht, rfl⟩ := atomOp_single (fun l e => hk ⟨l, e⟩) h
    rw [afoldl_single]
    exact atom_val w comb cons V ht fun l e => hk ⟨l, e⟩

theorem opValue_atom {S : GfS} {o : Nat} {k : Key} {op : Array Term} {V : Array F128} (h : S.ops[o]? = some op)
    (ha : atomOp sh comb0 k = .ok op) (hV : V.size = comb0) : opValue S w (V ++ comb) o = valOf sh w comb cons k := by
  simp only [opValue, getElem!_def, h]
  exact atomOp_val w comb cons ha hV

theorem varOp_spec {k : Key} {st st' : BSt} {o : Nat} (hi : OpsOk sh comb0 st)
    (hr : varOp sh comb0 k st = .ok (o, st')) :
    OpsOk sh comb0 st' ∧ Ext st st' ∧ st'.vNames = st.vNames ∧ st'.vVals = st.vVals ∧ st'.kappa = st.kappa ∧
      ∃ op, atomOp sh comb0 k = .ok op ∧ st'.ops[o]? = some op := by
  unfold varOp at hr
  obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
  cases hg
  cases hl : st.opOf.lookup k with
  | some i =>
    simp only [hl] at hr
    cases hr
    exact ⟨hi, Ext.refl _, rfl, rfl, rfl, hi.2 k _ hl⟩
  | none =>
    simp only [hl] at hr
    obtain ⟨t', st2, ht, hr⟩ := bm_bind hr
    obtain ⟨ha, h2⟩ := liftB_ok ht
    subst st2
    obtain ⟨_, st3, hs, hr⟩ := bm_bind hr
    cases hs
    cases hr
    refine ⟨⟨hi.1, fun k' i' hk' => ?_⟩, ⟨⟨#[t'], Array.push_eq_append⟩, ⟨#[], by simp⟩⟩, rfl, rfl,
      rfl, t', ha, by simp⟩
    simp only [List.lookup_append] at hk'
    cases hk'' : st.opOf.lookup k' with
    | some j =>
      rw [hk''] at hk'
      cases hk'
      obtain ⟨t2, h2, h3⟩ := hi.2 k' _ hk''
      exact ⟨t2, h2, (Ext.push st (.const 0) 0 t').ops? h3⟩
    | none =>
      rw [hk''] at hk'
      simp only [List.lookup, Option.none_or] at hk'
      split at hk'
      · rename_i hkk
        cases hk'
        have : k' = k := by simpa using hkk
        subst this
        exact ⟨t', ha, by simp⟩
      · cases hk'

theorem coefOps_spec (ri : Nat) : ∀ (l : List (Key × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    OpsOk sh comb0 st → coefOps sh comb0 ri l st = .ok (ps, st') →
    OpsOk sh comb0 st' ∧ Ext st st' ∧ st'.kappa = st.kappa ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V → V.size = comb0 →
        mulSum w S (V ++ comb) ps = tEval (valOf sh w comb cons) l)
  | [], st, st', ps, hi, hr => by
    cases hr
    exact ⟨hi, Ext.refl _, rfl, fun _ _ _ _ => rfl⟩
  | (k, x) :: rest, st, st', ps, hi, hr => by
    obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    obtain ⟨o, st3, hv, hr⟩ := bm_bind hr
    obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
    cases hr
    have hA := hi.push (.coef ri k) x #[.v st.vNames.size]
    obtain ⟨h3i, h3e, h3n, h3v, h3k, op, hop, ho⟩ := varOp_spec hA hv
    obtain ⟨h4i, h4e, h4k, hval⟩ := coefOps_spec ri rest st3 st' ps' h3i hc
    refine ⟨h4i, (Ext.push st _ _ _).trans (h3e.trans h4e), by rw [h4k, h3k], fun S V hSV hV => ?_⟩
    rw [mulSum_cons, hval S V hSV hV, tEval]
    have hslot : S.ops[st.ops.size]? = some #[Term.v st.vNames.size] := (h3e.trans (h4e.trans hSV)).ops? (by simp)
    have hx : V[st.vNames.size]? = some x := (h3e.trans (h4e.trans hSV)).vVals? (by simp [hi.1])
    have hlt : st.vNames.size < V.size := by
      rcases Nat.lt_or_ge st.vNames.size V.size with h' | h'
      · exact h'
      · rw [Array.getElem?_eq_none h'] at hx; cases hx
    rw [opValue_single w _ hslot, opValue_atom w comb cons (h4e.trans hSV |>.ops? ho) hop hV]
    simp only [termValue, getElem!_def, Array.getElem?_append_left hlt, hx]

theorem prodOps_spec (ri : Nat) : ∀ (l : List ((Key × Key) × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    OpsOk sh comb0 st → prodOps sh comb0 ri l st = .ok (ps, st') →
    OpsOk sh comb0 st' ∧ Ext st st' ∧ st'.kappa = st.kappa ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V → V.size = comb0 →
        mulSum w S (V ++ comb) ps = tEval (fun ab => valOf sh w comb cons ab.1 * valOf sh w comb cons ab.2) l)
  | [], st, st', ps, hi, hr => by
    cases hr
    exact ⟨hi, Ext.refl _, rfl, fun _ _ _ _ => rfl⟩
  | ((a, b), x) :: rest, st, st', ps, hi, hr => by
    simp only [prodOps] at hr
    split at hr
    · obtain ⟨_, _, h1, _⟩ := bm_bind hr
      cases h1
    · rename_i hx1
      obtain ⟨oa, st2, ha, hr⟩ := bm_bind hr
      obtain ⟨ob, st3, hb, hr⟩ := bm_bind hr
      obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
      cases hr
      obtain ⟨h2i, h2e, -, -, h2k, opa, hopa, hoa⟩ := varOp_spec hi ha
      obtain ⟨h3i, h3e, -, -, h3k, opb, hopb, hob⟩ := varOp_spec h2i hb
      obtain ⟨h4i, h4e, h4k, hval⟩ := prodOps_spec ri rest st3 st' ps' h3i hc
      refine ⟨h4i, h2e.trans (h3e.trans h4e), by rw [h4k, h3k, h2k], fun S V hSV hV => ?_⟩
      have hx : x = 1 := by simpa using hx1
      rw [mulSum_cons, hval S V hSV hV, tEval, hx,
        opValue_atom w comb cons ((h3e.trans (h4e.trans hSV)).ops? hoa) hopa hV,
        opValue_atom w comb cons ((h4e.trans hSV).ops? hob) hopb hV]
      (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_one]; ring)

/-- A key's consistency sum, or zero. -/
def consVal : Key → F128
  | .cons level l => cons level l
  | _ => 0

theorem kapAdd_spec (r : Nat) : ∀ (l : List (Key × F128)) (st st' : BSt), kapAdd r l st = .ok ((), st') →
    st'.ops = st.ops ∧ st'.vVals = st.vVals ∧ st'.vNames = st.vNames ∧ st'.opOf = st.opOf ∧
      ∀ r', kapSum st'.kappa r' cons = kapSum st.kappa r' cons + (if r' = r then tEval (consVal cons) l else 0)
  | [], st, st', hr => by
    cases hr
    refine ⟨rfl, rfl, rfl, rfl, fun r' => ?_⟩
    split <;> simp only [tEval, f128_add_zero]
  | (k, x) :: rest, st, st', hr => by
    have hstep : ∃ st1, kapAdd r rest st1 = .ok ((), st') ∧ st1.ops = st.ops ∧ st1.vVals = st.vVals ∧
        st1.vNames = st.vNames ∧ st1.opOf = st.opOf ∧
        ∀ r', kapSum st1.kappa r' cons = kapSum st.kappa r' cons + (if r' = r then x * consVal cons k else 0) := by
      simp only [kapAdd] at hr
      cases k
      case cons level l =>
        obtain ⟨st0, st2, hg, hr⟩ := bm_bind hr
        cases hg
        split at hr
        · rename_i hb
          obtain ⟨_, st1, hs, hr⟩ := bm_bind hr
          cases hs
          exact ⟨_, hr, rfl, rfl, rfl, rfl, fun r' => kapSum_modify _ cons x hb.1 hb.2.1 hb.2.2 r'⟩
        · obtain ⟨_, _, hs, _⟩ := bm_bind hr
          cases hs
      all_goals
        refine ⟨st, hr, rfl, rfl, rfl, rfl, fun r' => ?_⟩
        have hz : ∀ y : F128, y = 0 →
            kapSum st.kappa r' cons = kapSum st.kappa r' cons + (if r' = r then x * y else 0) := by
          intro y hy; subst hy; split <;> (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_zero]; ring)
        exact hz _ rfl
    obtain ⟨st1, hr1, s1, s2, s3, s4, hs⟩ := hstep
    obtain ⟨e1, e2, e3, e4, ih⟩ := kapAdd_spec r rest st1 st' hr1
    refine ⟨e1.trans s1, e2.trans s2, e3.trans s3, e4.trans s4, fun r' => ?_⟩
    rw [ih, hs, tEval]
    split <;> (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_zero]; ring)

theorem OpsOk.of_eq {st st' : BSt} (hi : OpsOk sh comb0 st) (h1 : st'.ops = st.ops) (h2 : st'.vVals = st.vVals)
    (h3 : st'.vNames = st.vNames) (h4 : st'.opOf = st.opOf) : OpsOk sh comb0 st' := by
  refine ⟨by rw [h2, h3]; exact hi.1, fun k i hk => ?_⟩
  rw [h4] at hk
  rw [h1]
  exact hi.2 k i hk

theorem Ext.of_eq {st st' : BSt} (h1 : st'.ops = st.ops) (h2 : st'.vVals = st.vVals) : Ext st st' :=
  ⟨⟨#[], by rw [h1]; simp⟩, ⟨#[], by rw [h2]; simp⟩⟩

theorem mapM_snd {β : Type} (g : β → Except String (Nat × β)) (hg : ∀ x p, g x = .ok p → p.2 = x) :
    ∀ (xs : List β) (keyed : List (Nat × β)), xs.mapM g = .ok keyed → keyed.map (·.2) = xs
  | [], keyed, h => by cases h; rfl
  | x :: xs, keyed, h => by
    rw [List.mapM_cons] at h
    cases hx : g x with
    | error e => rw [hx] at h; cases h
    | ok p =>
      cases hm : xs.mapM g with
      | error e => rw [hx, hm] at h; cases h
      | ok ks =>
        rw [hx, hm] at h
        cases h
        simp [hg x p hx, mapM_snd g hg xs ks hm]

theorem sortBy_perm {β : Type} {o : β → Except String Nat} {xs ys : List β} (h : sortBy o xs = .ok ys) :
    ys.Perm xs := by
  unfold sortBy at h
  have hg : ∀ x (p : Nat × β), (do return ((← o x), x) : Except String (Nat × β)) = .ok p → p.2 = x := by
    intro x p hp
    cases ho : o x with
    | error e => rw [ho] at hp; cases hp
    | ok n => rw [ho] at hp; cases hp; rfl
  cases hm : xs.mapM (fun x => do return ((← o x), x)) with
  | error e => rw [hm] at h; cases h
  | ok keyed =>
    rw [hm] at h
    cases h
    have := (List.mergeSort_perm keyed fun a b => decide (a.1 ≤ b.1)).map (·.2)
    rwa [mapM_snd _ hg xs keyed hm] at this

theorem mulSum_append (S : GfS) (v : Array F128) (a b : Array (Nat × Nat)) :
    mulSum w S v (a ++ b) = mulSum w S v a + mulSum w S v b := afoldl_append _ a b

theorem resValue_eq (S : GfS) (v : Array F128) (r : Res) :
    resValue S w v r = r.add.foldl (fun x t => x + termValue w v t) 0 + mulSum w S v r.mul := by
  unfold resValue mulSum
  rw [afoldl_shift]

theorem consOf_val (f : Lin) : tEval (consVal cons) (consOf f) = tEval (valOf sh w comb cons) (consOf f) := by
  apply tEval_congr
  intro kx hk
  simp only [consOf, List.mem_filter] at hk
  obtain ⟨k, x⟩ := kx
  cases k <;> simp_all [Key.isCons, consVal, valOf]

theorem accTerms_spec {fin ri : Nat} {l : List (Key × F128)} {st st' : BSt} {acc : Array Term}
    (hr : accTerms sh fin ri l st = .ok (acc, st')) :
    st'.ops = st.ops ∧ st'.vVals = st.vVals ∧ st'.vNames = st.vNames ∧ st'.opOf = st.opOf ∧
      (∀ r', kapSum st'.kappa r' cons =
        kapSum st.kappa r' cons + (if ri = fin + r' then tEval (consVal cons) l else 0)) ∧
      (l ≠ [] → ri = fin ∨ ri = fin + 1) ∧
      ∀ v : Array F128, acc.foldl (fun x t => x + termValue w v t) 0 =
        (if l = [] then 0 else valOf sh w comb cons (.acc1 (ri - fin)) + valOf sh w comb cons (.acc0 (ri - fin))) := by
  unfold accTerms at hr
  split at hr
  · rename_i he
    cases hr
    have : l = [] := List.isEmpty_iff.mp he
    subst this
    refine ⟨rfl, rfl, rfl, rfl, fun r' => ?_, fun h => absurd rfl h, fun v => rfl⟩
    split <;> simp only [tEval, f128_add_zero]
  · rename_i he
    have hne : l ≠ [] := fun h => by simp [h] at he
    split at hr
    · cases hr
    · rename_i hri
      have hfin : ri = fin ∨ ri = fin + 1 := by omega
      obtain ⟨_, st1, hk, hr⟩ := bm_bind hr
      obtain ⟨e1, e2, e3, e4, hks⟩ := kapAdd_spec cons (ri - fin) l st st1 hk
      cases h1 : wi sh (.acc1 (ri - fin)) with
      | none => simp only [h1] at hr; cases hr
      | some i1 =>
        cases h0 : wi sh (.acc0 (ri - fin)) with
        | none => simp only [h1, h0] at hr; cases hr
        | some i0 =>
          simp only [h1, h0] at hr
          cases hr
          refine ⟨e1, e2, e3, e4, fun r' => ?_, fun _ => hfin, fun v => ?_⟩
          · rw [hks]
            split_ifs <;> first | rfl | (exfalso; omega)
          · simp only [hne, ite_false, List.foldl_toArray', List.foldl_cons, List.foldl_nil, termValue, valOf, h1,
              h0, f128_zero_add]

theorem constTerm_spec {ri : Nat} {c : Option F128} {st st' : BSt} {cp : Array Term} (hi : OpsOk sh comb0 st)
    (hr : constTerm ri c st = .ok (cp, st')) :
    OpsOk sh comb0 st' ∧ Ext st st' ∧ st'.kappa = st.kappa ∧
      ∀ (F : Array (Array Term)) (V : Array F128), ExtTo st' F V →
        cp.foldl (fun x t => x + termValue w (V ++ comb) t) 0 = c.getD 0 := by
  cases c with
  | none =>
    cases hr
    exact ⟨hi, Ext.refl _, rfl, fun _ _ _ => rfl⟩
  | some c =>
    unfold constTerm at hr
    obtain ⟨_, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    cases hr
    refine ⟨⟨by simp [hi.1], hi.2⟩, ⟨⟨#[], by simp⟩, ⟨#[c], Array.push_eq_append⟩⟩, rfl, fun F V hFV => ?_⟩
    have hx : V[st.vNames.size]? = some c := hFV.vVals? (by simp [hi.1])
    have hlt : st.vNames.size < V.size := by
      rcases Nat.lt_or_ge st.vNames.size V.size with h' | h'
      · exact h'
      · rw [Array.getElem?_eq_none h'] at hx; cases hx
    simp only [List.foldl_toArray', List.foldl_cons, List.foldl_nil, termValue, getElem!_def,
      Array.getElem?_append_left hlt, hx, Option.getD_some, f128_zero_add]

theorem buildRes_spec {fin ri : Nat} {f : Lin} {st st' : BSt} {res : Res} (hi : OpsOk sh comb0 st)
    (hr : buildRes sh fin comb0 ri f st = .ok (res, st')) :
    OpsOk sh comb0 st' ∧ Ext st st' ∧
      (∀ r', kapSum st'.kappa r' cons =
        kapSum st.kappa r' cons + (if ri = fin + r' then tEval (valOf sh w comb cons) (consOf f) else 0)) ∧
      (consOf f ≠ [] → ri = fin ∨ ri = fin + 1) ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V → V.size = comb0 →
        resValue S w (V ++ comb) res = noCons (valOf sh w comb cons) f + (if consOf f = [] then 0 else
          valOf sh w comb cons (.acc1 (ri - fin)) + valOf sh w comb cons (.acc0 (ri - fin)))) := by
  unfold buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, a2, a3, a4, aks, afin, aval⟩ := accTerms_spec w comb cons h1
  obtain ⟨l1, st1', hs1, hr⟩ := bm_bind hr
  obtain ⟨hl1, h⟩ := liftB_ok hs1
  subst st1'
  obtain ⟨cm, st2, h2, hr⟩ := bm_bind hr
  obtain ⟨l2, st2', hs2, hr⟩ := bm_bind hr
  obtain ⟨hl2, h⟩ := liftB_ok hs2
  subst st2'
  obtain ⟨pm, st3, h3, hr⟩ := bm_bind hr
  obtain ⟨cp, st4, h4, hr⟩ := bm_bind hr
  cases hr
  have hi1 : OpsOk sh comb0 st1 := hi.of_eq a1 a2 a3 a4
  obtain ⟨h2i, h2e, h2k, h2val⟩ := coefOps_spec w comb cons ri l1 st1 st2 cm hi1 h2
  obtain ⟨h3i, h3e, h3k, h3val⟩ := prodOps_spec w comb cons ri l2 st2 st3 pm h2i h3
  obtain ⟨h4i, h4e, h4k, h4val⟩ := constTerm_spec w comb h3i h4
  refine ⟨h4i, (Ext.of_eq a1 a2).trans (h2e.trans (h3e.trans h4e)), fun r' => ?_, afin, fun S V hSV hV => ?_⟩
  · have hc := consOf_val (sh := sh) w comb cons f
    simp only [consOf] at hc ⊢
    rw [h4k, h3k, h2k, aks, hc]
  · rw [resValue_eq, afoldl_append, mulSum_append, aval, h4val S.ops V hSV,
      h2val S V (h3e.trans (h4e.trans hSV)) hV, h3val S V (h4e.trans hSV) hV,
      tEval_perm _ (sortBy_perm hl1), tEval_perm _ (sortBy_perm hl2), noCons]
    simp only [consOf]
    by_cases hc : List.filter (fun x => x.1.isCons) f.t = [] <;> simp only [hc, ite_true, ite_false]
    all_goals (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

/-- Residual `fi.2`'s value as `buildRes` builds it from `fi.1`: its terms but the consistency sums, and the running
sums in their place. -/
def resForm (val : Key → F128) (fin : Nat) (fi : Lin × Nat) : F128 :=
  noCons val fi.1 + (if consOf fi.1 = [] then 0 else val (.acc1 (fi.2 - fin)) + val (.acc0 (fi.2 - fin)))

theorem buildAll_spec {fin : Nat} : ∀ (fs : List Lin) (ri : Nat) (st st' : BSt) (rs : Array Res),
    OpsOk sh comb0 st → buildAll sh fin comb0 fs ri st = .ok (rs, st') →
    OpsOk sh comb0 st' ∧ Ext st st' ∧
      (∀ r', kapSum st'.kappa r' cons = kapSum st.kappa r' cons +
        (if ri ≤ fin + r' then tEval (valOf sh w comb cons) (consOf (fs.getD (fin + r' - ri) {})) else 0)) ∧
      (∀ fi ∈ fs.zipIdx ri, consOf fi.1 ≠ [] → fi.2 = fin ∨ fi.2 = fin + 1) ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V → V.size = comb0 →
        rs.toList.map (resValue S w (V ++ comb)) = (fs.zipIdx ri).map (resForm (valOf sh w comb cons) fin))
  | [], ri, st, st', rs, hi, hr => by
    cases hr
    refine ⟨hi, Ext.refl _, fun r' => ?_, by simp, fun _ _ _ _ => rfl⟩
    have : tEval (valOf sh w comb cons) (consOf (([] : List Lin).getD (fin + r' - ri) {})) = 0 := rfl
    split <;> simp only [this, f128_add_zero]
  | f :: rest, ri, st, st', rs, hi, hr => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    obtain ⟨h1i, h1e, h1k, h1fin, h1val⟩ := buildRes_spec w comb cons hi h1
    obtain ⟨h2i, h2e, h2k, h2fin, h2val⟩ := buildAll_spec rest (ri + 1) st1 st' rs' h1i h2
    refine ⟨h2i, h1e.trans h2e, fun r' => ?_, fun fi hfi => ?_, fun S V hSV hV => ?_⟩
    · rw [h2k, h1k]
      rcases Nat.lt_trichotomy ri (fin + r') with hlt | heq | hgt
      · rw [show fin + r' - ri = (fin + r' - (ri + 1)) + 1 by omega, List.getD_cons_succ]
        simp only [show ¬ri = fin + r' by omega, show ri + 1 ≤ fin + r' by omega, show ri ≤ fin + r' by omega,
          ite_true, ite_false]
        (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
      · simp only [heq, show ¬fin + r' + 1 ≤ fin + r' by omega, Nat.le_refl, Nat.sub_self, List.getD_cons_zero,
          ite_true, ite_false]
        (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
      · simp only [show ¬ri = fin + r' by omega, show ¬ri + 1 ≤ fin + r' by omega, show ¬ri ≤ fin + r' by omega,
          ite_false]
        (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
    · rw [List.zipIdx_cons, List.mem_cons] at hfi
      rcases hfi with rfl | hfi
      · exact h1fin
      · exact h2fin fi hfi
    · rw [List.zipIdx_cons, List.map_cons, Array.toList_append, List.map_append, h2val S V hSV hV]
      show [resValue S w (V ++ comb) r] ++ _ = _
      rw [h1val S V (h2e.trans hSV) hV]
      rfl

theorem kapSum_zero {K : Array (Array (Array F128))}
    (hK : ∀ a b c : Nat, a < K.size → c < ((K[a]!)[b]!).size → ((K[a]!)[b]!)[c]! = 0) (r : Nat) :
    kapSum K r cons = 0 := by
  unfold kapSum
  apply rangeSum_zero
  intro a ha
  apply rangeSum_zero
  intro c hc
  rw [hK a r c ha hc]
  (apply fOf_injective; simp only [fOf_mul, fOf_zero]; ring)

end Build

end Flock.RecAlgebra
