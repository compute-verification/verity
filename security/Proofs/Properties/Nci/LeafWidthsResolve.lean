module

public import Proofs.Properties.Nci.LeafWidths
public import Proofs.Properties.Nci.ResolveWithin
public import Flock.Sigma
public import Mathlib.Tactic

/-!
# Resolution keeps the width

A reference of a body names a leaf of a parameter or of a node's output, and resolution follows it, activation by
activation, to the gate that computes it. This module proves that the gate is as wide as the leaf.

* `wAt_widths`: the width `Widths.wAt` reads at a leaf of a type is that leaf's entry of the type's widths.
* `resolve_node`: resolution ends at a member of a node of a primitive, in an activation it reaches, and the primitive
  returns one value as wide as the leaf the reference names. Each step keeps the width of the leaf it names
  (`step_wide`): going down into a member, by the node's shape (`down_wide`); going up through a parameter, by the
  shape of the argument it is bound to (`up_wide`, with `axis_wide` for a member along an axis of a batch); and in
  both, by the widths the body's references keep (`BodyWide`).
* `arg_wide`: a member's argument leaf is as wide as the callee's parameter leaf, for any callee, composite or
  primitive: it reads the node's shape and the argument's references, and nothing of the callee's body.
* `gate_of_node`: locating the gate of a member of a primitive's node, in an activation resolution reaches, arrives at
  that node in that activation. The walk starts at the root, over every activation resolution reaches (`under_root`);
  from an activation over the node's it steps into the member that is still over it (`Resolve.gateStep_member`), and
  from the node's own it ends at the node (`gateStep_prim`): `walk_step`.
* `resolve_wide`: the two together. A gate a reference resolves to is as wide as the leaf the reference names.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program
open NciCheck.Properties.SecurityProofs.Resolve (Good RefOk BodyOk)

/-! ## A leaf's width is its type's -/

private theorem sizeOf_lt_of_mem {fs : Array Ty} {t : Ty} (h : t ∈ fs) : sizeOf t < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  omega

private theorem sizeOf_snd_lt_of_mem {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs) :
    sizeOf f.2 < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  obtain ⟨k, t⟩ := f
  simp only [Prod.mk.sizeOf_spec] at this ⊢
  omega

private theorem append_left (a b : Array ℕ) (j : ℕ) (hj : j < a.size) : (a ++ b)[j]! = a[j]! := by
  rw [getElem!_pos (a ++ b) j (by rw [Array.size_append]; omega), getElem!_pos a j hj, Array.getElem_append_left hj]

private theorem append_right (a b : Array ℕ) (j : ℕ) (hj : a.size ≤ j) (hb : j < a.size + b.size) :
    (a ++ b)[j]! = b[j - a.size]! := by
  rw [getElem!_pos (a ++ b) j (by rw [Array.size_append]; omega), getElem!_pos b (j - a.size) (by omega),
    Array.getElem_append_right hj]

/-- A fold that only appends keeps what the array already holds. -/
private theorem foldl_append_keep {β : Type} (g : β → Array ℕ) (j : ℕ) :
    ∀ (l : List β) (acc : Array ℕ), j < acc.size → (l.foldl (fun a t => a ++ g t) acc)[j]! = acc[j]! := by
  intro l
  induction l with
  | nil => intro acc _; rfl
  | cons x l ih =>
    intro acc hj
    rw [List.foldl_cons, ih (acc ++ g x) (by rw [Array.size_append]; omega), append_left acc (g x) j hj]

private theorem foldl_add {β : Type} (g : β → ℕ) :
    ∀ (l : List β) (acc : ℕ), l.foldl (fun a t => a + g t) acc = acc + (l.map g).sum := by
  intro l
  induction l with
  | nil => intro acc; rw [List.foldl_nil, List.map_nil, List.sum_nil, Nat.add_zero]
  | cons x l ih => intro acc; rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

/-- Leaf `j` of copies of an element appended one after another is the element's leaf `j` modulo its leaf count. -/
private theorem copies_at {β : Type} (w : Array ℕ) (j : ℕ) :
    ∀ (l : List β) (acc : Array ℕ), acc.size ≤ j → j < acc.size + l.length * w.size →
      (l.foldl (fun a _ => a ++ w) acc)[j]! = w[(j - acc.size) % w.size]! := by
  intro l
  induction l with
  | nil => intro acc h1 h2; rw [List.length_nil, Nat.zero_mul] at h2; omega
  | cons x l ih =>
    intro acc h1 h2
    rw [List.length_cons, Nat.succ_mul] at h2
    rw [List.foldl_cons]
    by_cases h : j < acc.size + w.size
    · rw [foldl_append_keep (fun _ => w) j l (acc ++ w) (by rw [Array.size_append]; exact h),
        append_right acc w j h1 h, Nat.mod_eq_of_lt (by omega)]
    · rw [ih (acc ++ w) (by rw [Array.size_append]; omega) (by rw [Array.size_append]; omega), Array.size_append,
        Nat.mod_eq_sub_mod (show w.size ≤ j - acc.size by omega)]
      congr 2
      omega

/-- The fold `Widths.wAt` runs over a compound type's fields finds the field holding leaf `j` and returns that field's
width at the leaf, which is the entry at `j` of the fields' widths appended in order. -/
private theorem fields_at {β : Type} (lf : β → ℕ) (wf : β → ℕ → ℕ) (g : β → Array ℕ) (j : ℕ)
    (F : Option ℕ × ℕ → β → Option ℕ × ℕ) (hsome : ∀ v n f, F (some v, n) f = (some v, n))
    (hnone : ∀ n f, F (none, n) f = (if j < n + lf f then some (wf f (j - n)) else none, n + lf f)) :
    ∀ (l : List β), (∀ f ∈ l, (g f).size = lf f ∧ ∀ i < lf f, wf f i = (g f)[i]!) → ∀ acc : Array ℕ, acc.size ≤ j →
      j < acc.size + (l.map lf).sum →
      ((l.foldl F (none, acc.size)).1).getD 0 = (l.foldl (fun a t => a ++ g t) acc)[j]! := by
  have keep : ∀ (l : List β) (v n : ℕ), l.foldl F (some v, n) = (some v, n) := by
    intro l
    induction l with
    | nil => intro v n; rfl
    | cons x l ih => intro v n; rw [List.foldl_cons, hsome, ih]
  intro l
  induction l with
  | nil => intro _ acc h1 h2; rw [List.map_nil, List.sum_nil] at h2; omega
  | cons x l ih =>
    intro hl acc h1 h2
    obtain ⟨hx, hwx⟩ := hl x (List.mem_cons_self ..)
    rw [List.map_cons, List.sum_cons] at h2
    rw [List.foldl_cons, List.foldl_cons, hnone]
    by_cases h : j < acc.size + lf x
    · simp only [h, ↓reduceIte]
      rw [keep, foldl_append_keep g j l (acc ++ g x) (by rw [Array.size_append]; omega),
        append_right acc (g x) j h1 (by omega), ← hwx (j - acc.size) (by omega)]
      rfl
    · simp only [h, ↓reduceIte]
      have := ih (fun f hf => hl f (List.mem_cons_of_mem _ hf)) (acc ++ g x) (by rw [Array.size_append]; omega)
        (by rw [Array.size_append]; omega)
      rw [Array.size_append, hx] at this
      exact this

private theorem wAt_widths_all : (t : Ty) → ∀ j, j < t.leaves → Widths.wAt t j = (Ty.widths t)[j]!
  | .value w => by
    intro j hj
    rw [Ty.leaves] at hj
    have : j = 0 := by omega
    subst this
    rw [Widths.wAt, Ty.widths]
    rfl
  | .array n e => by
    intro j hj
    rw [Ty.leaves] at hj
    have hL : 0 < e.leaves := by
      rcases Nat.eq_zero_or_pos e.leaves with h | h
      · rw [h] at hj; omega
      · exact h
    have hs : (Ty.widths e).size = e.leaves := Flock.Sigma.widths_length e
    rw [Widths.wAt, Ty.widths]
    have hne : (e.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    simp only [hne, Bool.false_eq_true, ↓reduceIte]
    rw [wAt_widths_all e (j % e.leaves) (Nat.mod_lt _ hL), ← Array.foldl_toList,
      copies_at (Ty.widths e) j (Array.range n).toList #[] (Nat.zero_le _)
        (by rw [Array.toList_range, List.length_range, hs]; show j < 0 + n * e.leaves; omega), hs]
    rfl
  | .tuple fs => by
    intro j hj
    rw [Ty.leaves, ← Array.foldl_toList, foldl_add] at hj
    rw [Widths.wAt, Ty.widths, ← Array.foldl_toList, ← Array.foldl_toList]
    exact fields_at Ty.leaves Widths.wAt Ty.widths j _ (fun _ _ _ => rfl) (fun _ _ => rfl) fs.toList
      (fun f hf => ⟨Flock.Sigma.widths_length f, fun i hi => wAt_widths_all f i hi⟩) #[] (Nat.zero_le _) hj
  | .record fs => by
    intro j hj
    rw [Ty.leaves, ← Array.foldl_toList, foldl_add] at hj
    rw [Widths.wAt, Ty.widths, ← Array.foldl_toList, ← Array.foldl_toList]
    exact fields_at (fun f : String × Ty => f.2.leaves) (fun f => Widths.wAt f.2) (fun f => Ty.widths f.2) j _
      (fun _ _ _ => rfl) (fun _ _ => rfl) fs.toList
      (fun f hf => ⟨Flock.Sigma.widths_length f.2, fun i hi => wAt_widths_all f.2 i hi⟩) #[] (Nat.zero_le _) hj
termination_by t => t
decreasing_by
  all_goals simp only [Ty.array.sizeOf_spec, Ty.tuple.sizeOf_spec, Ty.record.sizeOf_spec]
  · omega
  · exact sizeOf_lt_of_mem (Array.mem_def.mpr hf)
  · exact sizeOf_snd_lt_of_mem (Array.mem_def.mpr hf)

theorem wAt_widths (t : Ty) (j : ℕ) (hj : j < t.leaves) : Widths.wAt t j = (Ty.widths t)[j]! := by
  exact wAt_widths_all t j hj

/-! ## Along an axis -/

/-- Along an axis of an array, leaf `j` of member `m` is as wide as the leaf of the whole it is read from. -/
private theorem axis_wide : ∀ (ax : ℕ) (t : Ty) (ln : ℕ) (mt : Ty) (L : ℕ), t.removeAxis ax = some (ln, mt) →
    t.axisDims ax = some (ln, L) → ∀ j < mt.leaves, ∀ m < ln,
      Widths.wAt t ((j / L * ln + m) * L + j % L) = Widths.wAt mt j := by
  intro ax
  induction ax with
  | zero =>
    intro t ln mt L h hd j hj m hm
    cases t <;> simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at h
    rename_i n e
    obtain ⟨hn, he⟩ := h
    subst hn he
    simp only [Ty.axisDims, Option.some.injEq, Prod.mk.injEq, true_and] at hd
    subst hd
    have hne : (e.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    rw [Widths.wAt]
    simp only [hne, Bool.false_eq_true, ↓reduceIte]
    rw [Nat.div_eq_of_lt hj, Nat.mod_eq_of_lt hj, Nat.zero_mul, Nat.zero_add, Nat.add_comm, Nat.add_mul_mod_self_right,
      Nat.mod_eq_of_lt hj]
  | succ ax ih =>
    intro t ln mt L h hd j hj m hm
    cases t <;> simp only [Ty.removeAxis, Option.map_eq_some_iff, Prod.mk.injEq, reduceCtorEq, Prod.exists] at h
    rename_i n e
    obtain ⟨ln', inner, hr, hln, hmt⟩ := h
    subst hln hmt
    simp only [Ty.axisDims] at hd
    obtain ⟨L', hax, hlv, hdvd, hb⟩ := BodiesOk.axis_spec ax e ln' inner hr
    rw [hd] at hax
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at hax
    subst hax
    rw [BodiesOk.leaves_array] at hj
    have hIpos : 0 < inner.leaves := by
      rcases Nat.eq_zero_or_pos inner.leaves with h0 | h0
      · rw [h0] at hj; omega
      · exact h0
    obtain ⟨s, hs⟩ := hdvd
    have hLpos : 0 < L := by
      rcases Nat.eq_zero_or_pos L with h0 | h0
      · rw [h0, Nat.zero_mul] at hs; omega
      · exact h0
    have hr' : j % inner.leaves < inner.leaves := Nat.mod_lt _ hIpos
    have hjr : j = j % inner.leaves + (j / inner.leaves * s) * L := by
      have h1 := Nat.div_add_mod j inner.leaves
      have h2 : (j / inner.leaves * s) * L = inner.leaves * (j / inner.leaves) := by
        rw [Nat.mul_assoc, Nat.mul_comm s L, ← hs, Nat.mul_comm]
      rw [h2]
      omega
    have hdiv : j / L = j % inner.leaves / L + j / inner.leaves * s := by
      conv => lhs; rw [hjr]
      rw [Nat.add_mul_div_right _ _ hLpos]
    have hmod : j % L = j % inner.leaves % L := by
      conv => lhs; rw [hjr]
      rw [Nat.add_mul_mod_self_right]
    have hA := hb _ hr' m hm
    have e1 : ((j % inner.leaves / L + j / inner.leaves * s) * ln' + m) * L + j % inner.leaves % L =
        ((j % inner.leaves / L * ln' + m) * L + j % inner.leaves % L) + j / inner.leaves * e.leaves := by
      rw [hlv, hs]
      simp only [Nat.mul_add, Nat.mul_comm, Nat.mul_left_comm, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
    have hne : (e.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    have hni : (inner.leaves == 0) = false := by
      rw [beq_eq_false_iff_ne]; omega
    rw [Widths.wAt, Widths.wAt]
    simp only [hne, hni, Bool.false_eq_true, ↓reduceIte]
    rw [hdiv, hmod, e1, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hA]
    exact ih e ln' inner L hr hd (j % inner.leaves) hr' m hm

/-! ## What a step reads -/

private theorem wAt_value (w j : ℕ) : Widths.wAt (.value w) j = w := by
  rw [Widths.wAt]

private theorem wAt_array (n : ℕ) (e : Ty) (j : ℕ) (hL : 0 < e.leaves) :
    Widths.wAt (.array n e) j = Widths.wAt e (j % e.leaves) := by
  have hne : (e.leaves == 0) = false := by
    rw [beq_eq_false_iff_ne]; omega
  rw [Widths.wAt]
  simp only [hne, Bool.false_eq_true, ↓reduceIte]

private theorem wAt_tuple2 (a b : Ty) (j : ℕ) :
    Widths.wAt (.tuple #[a, b]) j = if j < a.leaves then Widths.wAt a j else
      if j < a.leaves + b.leaves then Widths.wAt b (j - a.leaves) else 0 := by
  rw [Widths.wAt, ← Array.foldl_toList]
  simp only [List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.sub_zero]
  by_cases h1 : j < a.leaves
  · simp only [h1, ↓reduceIte, Option.getD_some]
  · simp only [h1, ↓reduceIte]
    by_cases h2 : j < a.leaves + b.leaves
    · simp only [h2, ↓reduceIte, Option.getD_some]
    · simp only [h2, ↓reduceIte, Option.getD_none]

/-- The type a body's sources give a reference in range is the type the reference names. -/
private theorem src_ty (B : Body) (below : ℕ) (r : Ref) (hp : r.p = true → r.i < B.params.size)
    (hn : r.p = false → r.i < below ∧ r.i < B.nodes.size) :
    (srcOf B below).ty? r.p r.i = some (if r.p then (B.params[r.i]!).2 else (B.nodes[r.i]!).out) := by
  unfold Widths.Src.ty? srcOf
  cases hrp : r.p
  · obtain ⟨h1, h2⟩ := hn hrp
    simp only [Bool.false_eq_true, ↓reduceIte, h1]
    rw [Array.getElem?_map, getElem?_pos B.nodes r.i h2, getElem!_pos B.nodes r.i h2]
    rfl
  · have h1 := hp hrp
    simp only [↓reduceIte]
    rw [Array.getElem?_map, getElem?_pos B.params r.i h1, getElem!_pos B.params r.i h1]
    rfl

/-- A reference of a sequence that keeps the widths names a leaf as wide as the leaf it is read as. -/
private theorem seq_wide (P : Prog) (d : ℕ) (sc : Scope) (hd : sc.d = d) (below : ℕ) (t : Ty) (seq : Seq)
    (hbelow : below ≤ (P.defs[d]!).body.nodes.size) (hw : SeqWide (srcOf (P.defs[d]!).body below) t seq) (j : ℕ)
    (hj : j < seq.count) (hr : RefOk P d below (seq.get j)) :
    Widths.wAt (tyOf P sc (seq.get j)) (seq.get j).j = Widths.wAt t j := by
  subst hd
  exact hw j hj _
    (src_ty _ below _ (fun h => (hr.1 h).1) (fun h => ⟨(hr.2 h).1, Nat.lt_of_lt_of_le (hr.2 h).1 hbelow⟩))

/-! ## Resolution keeps the width -/

/-- **Down from a node's leaf.** The step ends at a member of the node, of a primitive that returns one value as wide
as the leaf, or goes into a member, at a leaf of its return as wide as the leaf. -/
private theorem down_wide (P : Prog) (hB : ∀ sc, Good P sc → BodyOk P sc.d)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (sc : Scope) (hg : Good P sc) (hd : sc.d < P.defs.size) (i j : ℕ) (hi : i < (P.defs[sc.d]!).body.nodes.size)
    (hj : j < ((P.defs[sc.d]!).body.nodes[i]!).out.leaves) (x) (h : resolveStep P (sc, ⟨false, i, j⟩) = .ok x) :
    (∃ m, x = .inl (sc.off + (P.defs[sc.d]!).body.offsets[i]! + m) ∧
      (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true ∧ m < ((P.defs[sc.d]!).body.nodes[i]!).members ∧
      (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).ret =
        .value (Widths.wAt ((P.defs[sc.d]!).body.nodes[i]!).out j)) ∨
    ∃ s', x = .inr s' ∧
      Widths.wAt (tyOf P s'.1 s'.2) s'.2.j = Widths.wAt ((P.defs[sc.d]!).body.nodes[i]!).out j := by
  obtain ⟨hfn, -, -, hdown, -⟩ := (hB sc hg).1 i hi
  have hdn := hdown j hj
  obtain ⟨-, ⟨-, -, hprim, hcomp, hshape, -⟩, -⟩ := (hD sc.d hd).2.1 i hi
  obtain ⟨hcp, hcc⟩ := (hD _ hfn).1
  have hret := (hD _ hfn).2.2.1
  rcases hf : ((P.defs[sc.d]!).body.nodes[i]!).form
  · rw [Resolve.rs_prim P sc i j hf] at h
    cases h
    simp only [hf, BodiesOk.Shape] at hshape
    obtain ⟨-, -, -, w, hw⟩ := hcp (hprim hf)
    refine Or.inl ⟨0, rfl, hprim hf, Nat.lt_of_lt_of_eq Nat.zero_lt_one (Resolve.prim_members _ (Or.inl hf)).symm, ?_⟩
    rw [hshape.1, hw, wAt_value]
  · rw [Resolve.rs_call P sc i j hf] at h
    cases h
    simp only [hf] at hdn
    simp only [hf, BodiesOk.Shape] at hshape
    obtain ⟨hrt, hbw⟩ := hW _ hfn hdn.1
    refine Or.inr ⟨_, rfl, ?_⟩
    dsimp only
    refine Eq.trans (seq_wide P ((P.defs[sc.d]!).body.nodes[i]!).fn _ rfl _ _ _ (Nat.le_refl _) hbw.2 j hdn.2
      (hret j hdn.2)) ?_
    rw [hrt, hshape.1]
  · rw [Resolve.rs_batch P sc i j hf] at h
    simp only [hf] at hdn
    simp only [hf, BodiesOk.Shape] at hshape
    obtain ⟨hm, -, hrc⟩ := hdn
    by_cases hcp' : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
    · simp only [hcp', ite_true] at h
      cases h
      obtain ⟨-, -, -, w, hw⟩ := hcp hcp'
      refine Or.inl ⟨_, rfl, hcp', by rw [Resolve.bs_members _ (Or.inl hf)]; exact hm, ?_⟩
      rw [hshape.1, hw, wAt_array _ _ _ (Nat.lt_of_lt_of_eq Nat.zero_lt_one (BodiesOk.leaves_value w).symm), wAt_value]
    · simp only [hcp', Bool.false_eq_true, ite_false] at h
      cases h
      have hcp'' := Bool.eq_false_of_ne_true hcp'
      obtain ⟨-, -, hol⟩ := hcc hcp''
      obtain ⟨hrt, hbw⟩ := hW _ hfn hcp''
      have hLpos : 0 < (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).ret.leaves := by
        rw [hshape.1, BodiesOk.leaves_array] at hj
        rcases Nat.eq_zero_or_pos (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).ret.leaves with h0 | h0
        · rw [h0] at hj; omega
        · exact h0
      refine Or.inr ⟨_, rfl, ?_⟩
      dsimp only
      refine Eq.trans (seq_wide P ((P.defs[sc.d]!).body.nodes[i]!).fn _ rfl _ _ _ (Nat.le_refl _) hbw.2 _
        (hrc hcp'') (hret _ (hrc hcp''))) ?_
      rw [hrt, hshape.1, wAt_array _ _ _ hLpos, hol]
  · simp only [hf, BodiesOk.Shape] at hshape
    obtain ⟨carry, y, hcr, hout, -, -, -⟩ := hshape
    have hcp'' := hcomp (Or.inr hf)
    by_cases hn : ((P.defs[sc.d]!).body.nodes[i]!).n = 0
    · exact absurd h (Resolve.rs_scan0 P sc i j x hf hn)
    rw [Resolve.rs_scan P sc i j carry y hf hn hcr] at h
    cases h
    obtain ⟨-, hrcount, -⟩ := hcc hcp''
    obtain ⟨hrt, hbw⟩ := hW _ hfn hcp''
    rw [hout, BodiesOk.leaves_tuple2, BodiesOk.leaves_array] at hj
    rw [hcr, BodiesOk.leaves_tuple2] at hrcount
    refine Or.inr ⟨_, rfl, ?_⟩
    dsimp only
    by_cases hjc : j < carry.leaves
    · simp only [hjc, ↓reduceIte]
      refine Eq.trans (seq_wide P ((P.defs[sc.d]!).body.nodes[i]!).fn _ rfl _ _ _ (Nat.le_refl _) hbw.2 j
        (by omega) (hret j (by omega))) ?_
      rw [hrt, hcr, hout, wAt_tuple2, wAt_tuple2]
      simp only [hjc, ↓reduceIte]
    · have hy : 0 < y.leaves := by
        rcases Nat.eq_zero_or_pos y.leaves with h0 | h0
        · rw [h0] at hj; omega
        · exact h0
      have hmod := Nat.mod_lt (j - carry.leaves) hy
      have h1 : ¬ carry.leaves + (j - carry.leaves) % y.leaves < carry.leaves := by omega
      have h2 : carry.leaves + (j - carry.leaves) % y.leaves < carry.leaves + y.leaves := by omega
      have h3 : j < carry.leaves + (Ty.array ((P.defs[sc.d]!).body.nodes[i]!).n y).leaves := by
        rw [BodiesOk.leaves_array]; exact hj
      simp only [hjc, ↓reduceIte]
      refine Eq.trans (seq_wide P ((P.defs[sc.d]!).body.nodes[i]!).fn _ rfl _ _ _ (Nat.le_refl _) hbw.2 _
        (by omega) (hret _ (by omega))) ?_
      rw [hrt, hcr, hout, wAt_tuple2, wAt_tuple2]
      simp only [h1, h2, h3, hjc, ↓reduceIte]
      rw [wAt_array _ _ _ hy, Nat.add_sub_cancel_left]

/-- **Up through a parameter's leaf.** The step goes to the parent, at a leaf of the argument the parameter is bound
to, or to the member before, at a leaf of its return; either is as wide as the parameter's leaf. -/
private theorem up_wide (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (parent : Scope) (d off k m i j : ℕ) (hg : Good P (.child parent d off k m))
    (hi : i < (P.defs[d]!).body.params.size) (hj : j < ((P.defs[d]!).body.params[i]!).2.leaves) (x)
    (h : resolveStep P (.child parent d off k m, ⟨true, i, j⟩) = .ok x) :
    ∃ s', x = .inr s' ∧
      Widths.wAt (tyOf P s'.1 s'.2) s'.2.j = Widths.wAt ((P.defs[d]!).body.params[i]!).2 j := by
  have hdd : d < P.defs.size := (Resolve.good_d P hroot hB _ hg).1
  unfold Good at hg
  obtain ⟨hp, hk, hfn, hc, hm, -⟩ := hg
  obtain ⟨hpd, hpp⟩ := Resolve.good_d P hroot hB parent hp
  obtain ⟨-, hargs, hup, -, -⟩ := (hB parent hp).1 k hk
  rw [hfn] at hup
  have hu := hup i hi j hj m hm
  obtain ⟨-, ⟨hsz, -, -, -, hshape, -⟩, -⟩ := (hD parent.d hpd).2.1 k hk
  rw [hfn] at hsz hshape
  obtain ⟨hpar, -, -⟩ := (hD d hdd).1.2 hc
  obtain ⟨hrt, hbwd⟩ := hW d hdd hc
  have hret := (hD d hdd).2.2.1
  have hia : i < ((P.defs[parent.d]!).body.nodes[k]!).args.size := by
    rw [hsz, ← hpar]; exact hi
  have hsw := (hW parent.d hpd hpp).2.1 k hk i hia
  have harg : ∀ t, t < (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.count →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get t))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get t).j =
        Widths.wAt (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).ty t :=
    fun t ht => seq_wide P parent.d parent rfl k _ _ (Nat.le_of_lt hk) hsw t ht (hargs i t ht)
  rw [hpar] at hj ⊢
  rcases hf : ((P.defs[parent.d]!).body.nodes[k]!).form
  · simp only [hf] at hu
    simp only [hf, BodiesOk.Shape] at hshape
    rw [Resolve.rs_up P parent d off k m i j (Or.inl hf)] at h
    cases h
    refine ⟨_, rfl, ?_⟩
    dsimp only
    rw [harg j hu, hshape.2 i hia]
  · simp only [hf] at hu
    simp only [hf, BodiesOk.Shape] at hshape
    rw [Resolve.rs_up P parent d off k m i j (Or.inr hf)] at h
    cases h
    refine ⟨_, rfl, ?_⟩
    dsimp only
    rw [harg j hu, hshape.2 i hia]
  · simp only [hf] at hu
    simp only [hf, BodiesOk.Shape] at hshape
    rw [Resolve.rs_up_batch P parent d off k m i j hf, Resolve.memberRef_eq] at h
    cases h
    refine ⟨_, rfl, ?_⟩
    dsimp only
    rw [harg _ hu]
    rcases hshape.2 i hia with ⟨hdn, hty⟩ | ⟨ax, mt, hrem, hmt, hdims⟩
    · have hidx : Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j = j := by
        simp only [Resolve.memberIdx, hdn]
      rw [hidx, hty]
    · obtain ⟨L, hax, -, -, -⟩ := BodiesOk.axis_spec ax _ _ mt hrem
      have hidx : Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j =
          (j / L * ((P.defs[parent.d]!).body.nodes[k]!).n + m) * L + j % L := by
        simp only [Resolve.memberIdx, hdims, hax]
      rw [hidx, axis_wide ax _ _ mt L hrem hax j (by rw [hmt]; exact hj) m
        (by rw [← Resolve.bs_members _ (Or.inl hf)]; exact hm), hmt]
  · simp only [hf] at hu
    simp only [hf, BodiesOk.Shape] at hshape
    obtain ⟨carry, y, hcr, -, hp0, ha0, hrest⟩ := hshape
    by_cases hi0 : i = 0
    · subst hi0
      simp only [ite_true] at hu
      rw [Resolve.rs_up_scan0 P parent d off k m j hf] at h
      rw [hp0] at hj ⊢
      by_cases hm0 : m = 0
      · subst hm0
        simp only [ite_true] at hu h
        cases h
        refine ⟨_, rfl, ?_⟩
        dsimp only
        rw [harg j hu, ha0]
      · simp only [hm0, ite_false] at hu h
        cases h
        refine ⟨_, rfl, ?_⟩
        dsimp only
        refine Eq.trans (seq_wide P d _ rfl _ _ _ (Nat.le_refl _) hbwd.2 j hu (hret j hu)) ?_
        rw [hrt, hcr, wAt_tuple2]
        simp only [hj, ↓reduceIte]
    · simp only [hi0, ite_false] at hu
      rw [Resolve.rs_up_scan P parent d off k m i j hi0 hf] at h
      cases h
      refine ⟨_, rfl, ?_⟩
      dsimp only
      obtain ⟨hxs, hnx⟩ := hrest i hia hi0
      by_cases hx : i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs
      · obtain ⟨e, hety, hep, hdims⟩ := hxs hx
        simp only [hx, ite_true, hdims, Option.map_some, Option.getD_some] at hu ⊢
        rw [← hep] at hj ⊢
        rw [harg _ hu, hety, wAt_array _ _ _ (by omega), Nat.add_comm, Nat.add_mul_mod_self_right,
          Nat.mod_eq_of_lt hj]
      · simp only [hx, ite_false] at hu ⊢
        rw [harg j hu, hnx hx]

/-- **A member's argument leaf is as wide as the callee's parameter leaf, for any callee.** Member `m` of node `k`
of an activation's body reads leaf `j` of its parameter `i` at a reference of the node's argument `i`: reference
`j` where the node is a call, reference `Resolve.memberIdx _ m j` where it is a batch. The leaf that reference
names in the activation's body is as wide as leaf `j` of the callee's parameter, the callee a composite or a
primitive: nothing here reads the callee's body. -/
theorem arg_wide (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (parent : Scope) (hp : Good P parent) (k : ℕ) (hk : k < (P.defs[parent.d]!).body.nodes.size) (m : ℕ)
    (hm : m < ((P.defs[parent.d]!).body.nodes[k]!).members) (i : ℕ)
    (hi : i < (P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params.size) (j : ℕ)
    (hj : j < ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2.leaves) :
    (((P.defs[parent.d]!).body.nodes[k]!).form = .call →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j).j =
        Widths.wAt ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2 j) ∧
    (((P.defs[parent.d]!).body.nodes[k]!).form = .batch →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
            (Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j)))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
            (Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j)).j =
        Widths.wAt ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2 j) := by
  obtain ⟨hpd, hpp⟩ := Resolve.good_d P hroot hB parent hp
  obtain ⟨-, hargs, -, -, -⟩ := (hB parent hp).1 k hk
  obtain ⟨-, ⟨hsz, hcnt, -, -, hshape, -⟩, -⟩ := (hD parent.d hpd).2.1 k hk
  have hia : i < ((P.defs[parent.d]!).body.nodes[k]!).args.size := by
    rw [hsz]; exact hi
  have hc := hcnt i hia
  have hsw := (hW parent.d hpd hpp).2.1 k hk i hia
  have harg : ∀ t, t < (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.count →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get t))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get t).j =
        Widths.wAt (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).ty t :=
    fun t ht => seq_wide P parent.d parent rfl k _ _ (Nat.le_of_lt hk) hsw t ht (hargs i t ht)
  refine ⟨fun hf => ?_, fun hf => ?_⟩
  · simp only [hf, BodiesOk.Shape] at hshape
    have hty := hshape.2 i hia
    rw [harg j (by rw [hc, hty]; exact hj), hty]
  · simp only [hf, BodiesOk.Shape] at hshape
    rcases hshape.2 i hia with ⟨hdn, hty⟩ | ⟨ax, mt, hrem, hmt, hdims⟩
    · have hidx : Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j = j := by
        simp only [Resolve.memberIdx, hdn]
      rw [hidx, harg j (by rw [hc, hty]; exact hj), hty]
    · obtain ⟨L, hax, -, -, hb⟩ := BodiesOk.axis_spec ax _ _ mt hrem
      have hidx : Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j =
          (j / L * ((P.defs[parent.d]!).body.nodes[k]!).n + m) * L + j % L := by
        simp only [Resolve.memberIdx, hdims, hax]
      have hmn : m < ((P.defs[parent.d]!).body.nodes[k]!).n := by
        rw [← Resolve.bs_members _ (Or.inl hf)]; exact hm
      rw [hidx, harg _ (by rw [hc]; exact hb j (by rw [hmt]; exact hj) m hmn),
        axis_wide ax _ _ mt L hrem hax j (by rw [hmt]; exact hj) m hmn, hmt]

/-- **One step of resolution keeps the width.** From a state that resolves below a bound, a step ends at a member of
a node of the activation, of a primitive that returns one value as wide as the leaf the state names, or it moves to a
state that names a leaf as wide. -/
private theorem step_wide (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (B : ℕ) (s : Scope × Ref) (x) (hs : Resolve.Inv P B s) (h : resolveStep P s = .ok x) :
    (∀ g, x = .inl g → ∃ k m, k < (P.defs[s.1.d]!).body.nodes.size ∧
      (P.defs[((P.defs[s.1.d]!).body.nodes[k]!).fn]!).prim = true ∧
      m < ((P.defs[s.1.d]!).body.nodes[k]!).members ∧ g = s.1.off + (P.defs[s.1.d]!).body.offsets[k]! + m ∧
      (P.defs[((P.defs[s.1.d]!).body.nodes[k]!).fn]!).ret = .value (Widths.wAt (tyOf P s.1 s.2) s.2.j)) ∧
    (∀ s', x = .inr s' → Widths.wAt (tyOf P s'.1 s'.2) s'.2.j = Widths.wAt (tyOf P s.1 s.2) s.2.j) := by
  obtain ⟨sc, ⟨p, i, j⟩⟩ := s
  obtain ⟨hg, hr, -, -⟩ := hs
  dsimp only at hg hr ⊢
  cases p
  · have hi : i < (P.defs[sc.d]!).body.nodes.size := (hr.2 rfl).1
    have hj : j < ((P.defs[sc.d]!).body.nodes[i]!).out.leaves := (hr.2 rfl).2
    have hty : tyOf P sc ⟨false, i, j⟩ = ((P.defs[sc.d]!).body.nodes[i]!).out := rfl
    rw [hty]
    rcases down_wide P hB hD hW sc hg (Resolve.good_d P hroot hB sc hg).1 i j hi hj x h with
      ⟨m, rfl, hp, hm, hw⟩ | ⟨s', rfl, hw⟩
    · exact ⟨fun g hg' => by cases hg'; exact ⟨i, m, hi, hp, hm, rfl, hw⟩, fun s' hs' => by cases hs'⟩
    · exact ⟨fun g hg' => (by cases hg'), fun s'' hs' => by cases hs'; exact hw⟩
  · rcases sc with ⟨d, off, st⟩ | ⟨parent, d, off, k, m⟩
    · unfold Good at hg
      obtain ⟨-, -, rfl⟩ := hg
      exact absurd h (Resolve.rs_top P d off i j x)
    · simp only [Scope.d.eq_2] at hr
      have hi : i < (P.defs[d]!).body.params.size := (hr.1 rfl).1
      have hj : j < ((P.defs[d]!).body.params[i]!).2.leaves := (hr.1 rfl).2
      have hty : tyOf P (.child parent d off k m) ⟨true, i, j⟩ = ((P.defs[d]!).body.params[i]!).2 := rfl
      rw [hty]
      obtain ⟨s', rfl, hw⟩ := up_wide P hroot hB hD hW parent d off k m i j hg hi hj x h
      exact ⟨fun g hg' => (by cases hg'), fun s'' hs' => by cases hs'; exact hw⟩

/-- **Where resolution ends**: at a member of a node of a primitive, in an activation resolution reaches, whose
primitive returns one value as wide as the leaf the reference names. -/
theorem resolve_node (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (sc : Scope) (hsc : Good P sc) (r : Ref) (hr : RefOk P sc.d (P.defs[sc.d]!).body.nodes.size r) (g : ℕ)
    (h : resolve P sc r = .ok g) :
    ∃ sc' k m, Good P sc' ∧ k < (P.defs[sc'.d]!).body.nodes.size ∧
      (P.defs[((P.defs[sc'.d]!).body.nodes[k]!).fn]!).prim = true ∧
      m < ((P.defs[sc'.d]!).body.nodes[k]!).members ∧ g = sc'.off + (P.defs[sc'.d]!).body.offsets[k]! + m ∧
      (P.defs[((P.defs[sc'.d]!).body.nodes[k]!).fn]!).ret = .value (Widths.wAt (tyOf P sc r) r.j) := by
  have hinv : Resolve.Inv P (sc.off + (P.defs[sc.d]!).body.offsets[r.i + 1]!) (sc, r) := by
    unfold Resolve.Inv
    exact ⟨hsc, hr, fun _ => Nat.le_add_right _ _, fun _ => Nat.le_refl _⟩
  exact DescAgree.run_inv "scope resolution" (resolveStep P)
    (fun s => Resolve.Inv P (sc.off + (P.defs[sc.d]!).body.offsets[r.i + 1]!) s ∧
      Widths.wAt (tyOf P s.1 s.2) s.2.j = Widths.wAt (tyOf P sc r) r.j)
    (fun g => ∃ sc' k m, Good P sc' ∧ k < (P.defs[sc'.d]!).body.nodes.size ∧
      (P.defs[((P.defs[sc'.d]!).body.nodes[k]!).fn]!).prim = true ∧
      m < ((P.defs[sc'.d]!).body.nodes[k]!).members ∧ g = sc'.off + (P.defs[sc'.d]!).body.offsets[k]! + m ∧
      (P.defs[((P.defs[sc'.d]!).body.nodes[k]!).fn]!).ret = .value (Widths.wAt (tyOf P sc r) r.j))
    (fun s g hs hx => by
      obtain ⟨k, m, hk, hp, hm, hgg, hw⟩ := (step_wide P hroot hB hD hW _ s _ hs.1 hx).1 g rfl
      rw [hs.2] at hw
      exact ⟨s.1, k, m, hs.1.1, hk, hp, hm, hgg, hw⟩)
    (fun s s' hs hx => ⟨(Resolve.step_ok P hroot hB hL _ s _ hs.1 hx).2 s' rfl,
      ((step_wide P hroot hB hD hW _ s _ hs.1 hx).2 s' rfl).trans hs.2⟩)
    _ _ _ ⟨hinv, rfl⟩ h

/-! ## The walk arrives where the layout says -/

/-- Every activation resolution reaches lies under the root's. -/
private theorem under_root (P : Prog) : ∀ sc, Good P sc → Resolve.Under P.rootScope sc
  | .top d off st, h => by
    obtain ⟨rfl, rfl, rfl⟩ := h
    rfl
  | .child parent d off k m, h => Or.inr (under_root P parent h.1)

/-- An activation over one resolution reaches is one resolution reaches. -/
private theorem under_good (P : Prog) (U : Scope) : ∀ sc, Good P sc → Resolve.Under U sc → Good P U
  | .top d off st, h, hu => by
    have hu' : U = .top d off st := hu
    rw [hu']
    exact h
  | .child parent d off k m, h, hu => by
    have hu' : U = .child parent d off k m ∨ Resolve.Under U parent := hu
    rcases hu' with hu' | hu'
    · rw [hu']
      exact h
    · exact under_good P U parent h.1 hu'

/-- From an activation over another, one step down: the member of one of its nodes that is still over the other. -/
private theorem under_child (U : Scope) : ∀ sc, Resolve.Under U sc → U ≠ sc →
    ∃ d off k m, Resolve.Under (.child U d off k m) sc
  | .top d off st, hu, hne => absurd hu hne
  | .child parent d off k m, hu, hne => by
    have hu' : U = .child parent d off k m ∨ Resolve.Under U parent := hu
    rcases hu' with hu' | hu'
    · exact absurd hu' hne
    · by_cases hp : U = parent
      · rw [hp]
        exact ⟨d, off, k, m, Resolve.under_refl _⟩
      · obtain ⟨d', off', k', m', hc⟩ := under_child U parent hu' hp
        exact ⟨d', off', k', m', Or.inr hc⟩

/-- An activation resolution reaches ends no later than an activation over it. -/
private theorem under_end (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d) (U : Scope) :
    ∀ sc, Good P sc → Resolve.Under U sc → sc.off + (P.defs[sc.d]!).gates ≤ U.off + (P.defs[U.d]!).gates
  | .top d off st, _, hu => by
    have hu' : U = .top d off st := hu
    exact hu' ▸ Nat.le_refl _
  | .child parent d off k m, hg, hu => by
    have hu' : U = .child parent d off k m ∨ Resolve.Under U parent := hu
    rcases hu' with hu' | hu'
    · exact hu' ▸ Nat.le_refl _
    · obtain ⟨hp, hk, hfn, hc, hm, ho⟩ := hg
      have ih := under_end P hroot hB hL U parent hp hu'
      obtain ⟨hpd, hpp⟩ := Resolve.good_d P hroot hB parent hp
      obtain ⟨-, hK, hnodes⟩ := hL parent.d hpd hpp
      have hstep := hnodes k hk
      have hmono := Resolve.offsets_mono P parent.d (hL parent.d hpd) hpp (k + 1) _ (by omega) (Nat.le_refl _)
      rw [hfn] at hstep
      have hle : (m + 1) * (P.defs[d]!).gates ≤
          ((P.defs[parent.d]!).body.nodes[k]!).members * (P.defs[d]!).gates := Nat.mul_le_mul_right _ hm
      rw [Nat.add_mul, Nat.one_mul] at hle
      show off + (P.defs[d]!).gates ≤ _
      omega

/-- **The last step of `Prog.gate`'s walk**: for the gate of member `m` of node `k` of an activation, the node calling
a primitive of one gate, the step from the activation ends at the node. -/
private theorem gateStep_prim (P : Prog) (hL : ∀ d < P.defs.size, Walk.LaidOut P d) (sc : Scope)
    (hd : sc.d < P.defs.size) (hp : (P.defs[sc.d]!).prim = false) {k m : ℕ}
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hg1 : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates = 1) {x : (ℕ × Array ℕ × Scope) ⊕ Scope}
    (h : gateStep P (sc.off + (P.defs[sc.d]!).body.offsets[k]! + m) sc = .ok x) :
    ∃ ops, x = .inl (((P.defs[sc.d]!).body.nodes[k]!).fn, ops, sc) := by
  have hmono := Resolve.offsets_mono P sc.d (hL sc.d hd) hp
  obtain ⟨-, -, hnodes⟩ := hL sc.d hd hp
  have hstep := hnodes k hk
  rw [hg1, Nat.mul_one] at hstep
  unfold gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k' hk'
    obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk'
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    have hkk : k' = k := by
      rcases Nat.lt_trichotomy k' k with hlt | heq | hgt
      · have := hmono (k' + 1) k (by omega) (by omega)
        omega
      · exact heq
      · have := hmono (k + 1) k' (by omega) (by omega)
        omega
    rw [hkk] at h
    simp only [hc, ite_true] at h
    split at h
    · cases h
    · rename_i ops hops
      simp only [pure, Except.pure, Except.ok.injEq] at h
      exact ⟨ops, h.symm⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- **One step of `Prog.gate`'s walk toward a primitive's node.** From an activation over the node's, the step ends at
the node when the activation is the node's, and otherwise goes on to the activation below that is still over the
node's. -/
private theorem walk_step (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d) (sc : Scope)
    (hsc : Good P sc) (k m : ℕ) (hk : k < (P.defs[sc.d]!).body.nodes.size)
    (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hg1 : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates = 1)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) (w : Scope) (hu : Resolve.Under w sc)
    (x : (ℕ × Array ℕ × Scope) ⊕ Scope)
    (h : gateStep P (sc.off + (P.defs[sc.d]!).body.offsets[k]! + m) w = .ok x) :
    (∃ ops, x = .inl (((P.defs[sc.d]!).body.nodes[k]!).fn, ops, sc)) ∨ ∃ w', x = .inr w' ∧ Resolve.Under w' sc := by
  obtain ⟨hd, hpc⟩ := Resolve.good_d P hroot hB sc hsc
  by_cases hw : w = sc
  · rw [hw] at h
    exact Or.inl (gateStep_prim P hL sc hd hpc hk hm hp hg1 h)
  · obtain ⟨d', off', k', m', hc⟩ := under_child w sc hu hw
    obtain ⟨hwd, hwp⟩ := Resolve.good_d P hroot hB w (under_good P w sc hsc hu)
    have hlo := Resolve.under_off P _ sc hsc hc
    have hhi := under_end P hroot hB hL _ sc hsc hc
    obtain ⟨-, hk', hfn', hc', hm', ho'⟩ := under_good P _ sc hsc hc
    obtain ⟨-, hK, hnodes⟩ := hL sc.d hd hpc
    have hstep := hnodes k hk
    rw [hg1, Nat.mul_one] at hstep
    have hend := Resolve.offsets_mono P sc.d (hL sc.d hd) hpc (k + 1) _ (by omega) (Nat.le_refl _)
    have hlo' : off' ≤ sc.off := hlo
    have hhi' : sc.off + (P.defs[sc.d]!).gates ≤ off' + (P.defs[d']!).gates := hhi
    rw [← hfn'] at hc' ho' hhi'
    have hx := Resolve.gateStep_member P hL w hwd hwp hk' hm' hc' (by omega) (by omega) h
    refine Or.inr ⟨_, hx, ?_⟩
    have hch : P.child w k' m' = .child w d' off' k' m' := by
      show Scope.child w _ _ k' m' = _
      rw [← ho', hfn']
    rw [hch]
    exact hc

/-- **The walk arrives where the layout says**: the gate of member `m` of a primitive's node `k`, in an activation
resolution reaches, is located at that node, with its callee for primitive. -/
theorem gate_of_node (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!) (sc : Scope)
    (hsc : Good P sc) (k m : ℕ) (hk : k < (P.defs[sc.d]!).body.nodes.size)
    (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) (p : ℕ) (ops : Array ℕ) (sc' : Scope)
    (h : P.gate (sc.off + (P.defs[sc.d]!).body.offsets[k]! + m) = .ok (p, ops, sc')) :
    p = ((P.defs[sc.d]!).body.nodes[k]!).fn ∧ sc' = sc := by
  obtain ⟨-, hg1, -, -⟩ := (hD _ ((hB sc hsc).1 k hk).1).1.1 hp
  unfold Prog.gate at h
  by_cases hge : sc.off + (P.defs[sc.d]!).body.offsets[k]! + m ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (gateStep P (sc.off + (P.defs[sc.d]!).body.offsets[k]! + m))
    (fun w => Resolve.Under w sc) (fun b => b.1 = ((P.defs[sc.d]!).body.nodes[k]!).fn ∧ b.2.2 = sc) ?_ ?_ _ _
    (p, ops, sc') (under_root P sc hsc) h
  · intro w b hu hb
    rcases walk_step P hroot hB hL sc hsc k m hk hp hg1 hm w hu _ hb with ⟨ops', hx⟩ | ⟨w', hx, -⟩
    · cases hx
      exact ⟨rfl, rfl⟩
    · cases hx
  · intro w w' hu hb
    rcases walk_step P hroot hB hL sc hsc k m hk hp hg1 hm w hu _ hb with ⟨ops', hx⟩ | ⟨w'', hx, hu'⟩
    · cases hx
    · cases hx
      exact hu'

/-- **A gate a reference resolves to is as wide as the leaf it names**, wherever the gate is located. -/
theorem resolve_wide (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (sc : Scope) (hsc : Good P sc) (r : Ref) (hr : RefOk P sc.d (P.defs[sc.d]!).body.nodes.size r) (g : ℕ)
    (h : resolve P sc r = .ok g) (p : ℕ) (ops : Array ℕ) (sc' : Scope) (hg : P.gate g = .ok (p, ops, sc')) :
    (P.defs[p]!).ret = .value (Widths.wAt (tyOf P sc r) r.j) := by
  obtain ⟨sc₁, k, m, hsc₁, hk, hp, hm, rfl, hw⟩ := resolve_node P hroot hB hL hD hW sc hsc r hr g h
  rw [(gate_of_node P hroot hB hL hD sc₁ hsc₁ k m hk hp hm p ops sc' hg).1]
  exact hw

end NciCheck.Properties.SecurityProofs.LeafWidths
