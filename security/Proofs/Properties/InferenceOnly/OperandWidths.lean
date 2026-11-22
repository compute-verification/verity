module

public import Proofs.Properties.InferenceOnly.LeafWidthsResolve
public import Proofs.Properties.InferenceOnly.LeafWidthsDecoded
public import Proofs.Properties.InferenceOnly.BodiesOk
public import Proofs.Properties.InferenceOnly.Resolve
public import Proofs.Properties.InferenceOnly.ReadsExact
public import Properties.Flock.Circuit.OfProgram

/-!
# A gate's operands are no wider than its primitive's parameters

In a decoded Program whose root reads no input node recut and whose nodes read only earlier nodes
(`Prog.orderViolation`), the gate `Prog.gate` locates calls a primitive at a member of a node, and its operands are the
leaves of the node's arguments, resolved, in order (`gate_loc`, `operands_lists`). Each resolves to a gate as wide as
the parameter leaf it is bound to (`LeafWidths.resolve_wide`, `LeafWidths.arg_wide`), or to one `Prog.gate` doesn't
locate, of width 0: `gate_operand_widths`, and its sum, `gate_operand_bits`. The order is needed: `decode`'s width check
reads a node's references to later nodes as no source, so they may name leaves of any width.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.OperandWidths

open Flock.Program Flock.OfProgram
open InferenceOnlyVerifier.Properties.SecurityProofs

/-! ## Lists -/

theorem forall₂_of_get {R : ℕ → ℕ → Prop} :
    ∀ (a b : List ℕ), a.length = b.length → (∀ i (ha : i < a.length) (hb : i < b.length), R a[i] b[i]) →
      List.Forall₂ R a b
  | [], [], _, _ => .nil
  | [], _ :: _, h, _ => absurd h (by simp)
  | _ :: _, [], h, _ => absurd h (by simp)
  | x :: a, y :: b, h, hR =>
    .cons (hR 0 (by simp) (by simp)) (forall₂_of_get a b (by simpa using h)
      fun i ha hb => hR (i + 1) (by simpa using ha) (by simpa using hb))

theorem forall₂_append {R : ℕ → ℕ → Prop} :
    ∀ {a b c d : List ℕ}, List.Forall₂ R a b → List.Forall₂ R c d → List.Forall₂ R (a ++ c) (b ++ d)
  | [], [], _, _, .nil, h => h
  | _ :: _, _ :: _, _, _, .cons h1 h2, h => .cons h1 (forall₂_append h2 h)

theorem forall₂_flatten {R : ℕ → ℕ → Prop} :
    ∀ (A B : List (List ℕ)), A.length = B.length →
      (∀ i (ha : i < A.length) (hb : i < B.length), List.Forall₂ R A[i] B[i]) →
        List.Forall₂ R A.flatten B.flatten
  | [], [], _, _ => .nil
  | [], _ :: _, h, _ => absurd h (by simp)
  | _ :: _, [], h, _ => absurd h (by simp)
  | x :: A, y :: B, h, hR =>
    forall₂_append (hR 0 (by simp) (by simp)) (forall₂_flatten A B (by simpa using h)
      fun i ha hb => hR (i + 1) (by simpa using ha) (by simpa using hb))

theorem sum_le_of_forall₂ : ∀ {a b : List ℕ}, List.Forall₂ (· ≤ ·) a b → a.sum ≤ b.sum
  | [], [], .nil => le_refl _
  | _ :: _, _ :: _, .cons h1 h2 => by
    simp only [List.sum_cons]
    exact Nat.add_le_add h1 (sum_le_of_forall₂ h2)

/-! ## A decoded Program's facts -/

/-- **Every Definition of a decoded Program is laid out**: its nodes' gates follow one another from 0. -/
theorem decoded_laid (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) : ∀ d < P.defs.size, Walk.LaidOut P d := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_full bytes P h hr
  intro d hd hp
  have hl := hinv.2.2.1 d (by rw [← hP]; exact hd)
  rw [← hP] at hl
  obtain ⟨-, h0, hg, hk⟩ := hl.2 hp
  exact ⟨h0, hg, fun k hk' => (hk k hk').2⟩

/-! ## Where a gate is -/

/-- **`Prog.gate` locates a gate at a member of a node calling a primitive**, in an activation resolution reaches,
with that member's operands. -/
theorem gate_loc (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    {g p : ℕ} {ops : Array ℕ} {sc : Scope} (h : P.gate g = .ok (p, ops, sc)) :
    ∃ k m, Resolve.Good P sc ∧ k < (P.defs[sc.d]!).body.nodes.size ∧ ((P.defs[sc.d]!).body.nodes[k]!).fn = p ∧
      (P.defs[p]!).prim = true ∧ m < ((P.defs[sc.d]!).body.nodes[k]!).members ∧
      operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops := by
  rw [Prog.gate_of (Prog.gate_recut h)] at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (gateStep P g) (Resolve.WalkQ P g)
    (fun b => ∃ k m, Resolve.Good P b.2.2 ∧ k < (P.defs[b.2.2.d]!).body.nodes.size ∧
      ((P.defs[b.2.2.d]!).body.nodes[k]!).fn = b.1 ∧ (P.defs[b.1]!).prim = true ∧
      m < ((P.defs[b.2.2.d]!).body.nodes[k]!).members ∧
        operands P b.2.2 ((P.defs[b.2.2.d]!).body.nodes[k]!) m = .ok b.2.1)
    ?_ ?_ _ _ (p, ops, sc)
    ⟨by show Resolve.Good P (.top P.root 0 none); unfold Resolve.Good; exact ⟨rfl, rfl, rfl⟩, Nat.zero_le _,
      by show g - 0 < (P.defs[P.root]!).gates; omega⟩ h
  · intro sc b ⟨hg, hoff, hlt⟩ hb
    obtain ⟨hd, hpc⟩ := Resolve.good_d P hroot hB sc hg
    obtain ⟨h0, hK, hnodes⟩ := hL sc.d hd hpc
    unfold gateStep at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · rename_i k hk
      obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
      have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
          ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
        have := hnodes k hkK
        omega
      obtain ⟨hm1, -, -⟩ := Walk.member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
      by_cases hpc' : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
      · simp only [hpc', ite_true] at hb
        split at hb
        · cases hb
        · rename_i ops' hops
          simp only [pure, Except.pure, Except.ok.injEq, Sum.inl.injEq] at hb
          subst hb
          exact ⟨k, _, hg, hkK, rfl, hpc', hm1, hops⟩
      · simp only [hpc', Bool.false_eq_true, ite_false, pure, Except.pure] at hb
        cases hb
    · simp [throw, throwThe, MonadExceptOf.throw] at hb
  · intro sc sc' ⟨hg, hoff, hlt⟩ hb
    obtain ⟨hd, hpc⟩ := Resolve.good_d P hroot hB sc hg
    obtain ⟨h0, hK, hnodes⟩ := hL sc.d hd hpc
    unfold gateStep at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · rename_i k hk
      obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
      have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
          ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
        have := hnodes k hkK
        omega
      obtain ⟨hm1, hm2, hm3⟩ := Walk.member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
      by_cases hpc' : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
      · simp only [hpc', ite_true] at hb
        split at hb
        · cases hb
        · simp only [pure, Except.pure, Except.ok.injEq] at hb
          cases hb
      · simp only [hpc', Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq, Sum.inr.injEq] at hb
        subst hb
        refine ⟨Resolve.good_child P hg hkK rfl (by simpa using hpc') hm1 rfl, ?_, ?_⟩
        · simp only [Prog.child, Scope.off.eq_2]
          omega
        · simp only [Prog.child, Scope.off.eq_2, Scope.d.eq_2]
          omega
    · simp [throw, throwThe, MonadExceptOf.throw] at hb

/-! ## A member's operands, in order -/

/-- How many references `operands` resolves for argument `i` of `node`. -/
def cntOf (P : Prog) (node : Node) (i : ℕ) : ℕ :=
  if (node.form == .batch && (node.args[i]!).dims.isSome) = true then ((P.defs[node.fn]!).params[i]!).2.leaves
  else (node.args[i]!).seq.count

/-- The reference `operands` resolves for leaf `t` of argument `i` of member `m` of `node`. -/
def refOf (node : Node) (m i t : ℕ) : Ref :=
  if (node.form == .batch) = true then (node.args[i]!).memberRef m t else (node.args[i]!).seq.get t

/-- **`operands` lists each argument's resolved references in turn.** -/
theorem operands_lists (P : Prog) (sc : Scope) (node : Node) (m : ℕ) (ops : Array ℕ)
    (h : operands P sc node m = .ok ops) :
    ∃ L : List (List ℕ), L.length = node.args.size ∧ ops.toList = L.flatten ∧
      ∀ i (hi : i < L.length), L[i].length = cntOf P node i ∧
        ∀ t (ht : t < L[i].length), resolve P sc (refOf node m i t) = .ok L[i][t] := by
  rw [operands_of (operands_cond h)] at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', bind, Except.bind] at h
  split at h
  · cases h
  rename_i ops' hops
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  have key := ReadsExact.forIn_range_idx
    (fun i (r : Array ℕ) => ∃ L : List (List ℕ), L.length = i ∧ r.toList = L.flatten ∧
      ∀ i' (hi : i' < L.length), L[i'].length = cntOf P node i' ∧
        ∀ t (ht : t < L[i'].length), resolve P sc (refOf node m i' t) = .ok L[i'][t])
    _ _ ?_ _ 0 #[] ops' (le_refl _) ⟨[], rfl, rfl, fun i hi => absurd hi (by simp)⟩ hops
  · obtain ⟨L, hlen, hflat, hL⟩ := key
    exact ⟨L, hlen.trans (by simp [Std.Legacy.Range.size]), hflat, hL⟩
  intro i b r hi hb hr
  obtain ⟨L, rfl, hflat, hL⟩ := hb
  split at hr
  · cases hr
  rename_i inner hinner
  simp only [pure, Except.pure, Except.ok.injEq] at hr
  subst hr
  refine ⟨_, rfl, ?_⟩
  have key := ReadsExact.forIn_range_idx
    (fun t (r : Array ℕ) => ∃ l : List ℕ, l.length = t ∧ r.toList = b.toList ++ l ∧
      ∀ t' (ht : t' < l.length), resolve P sc (refOf node m L.length t') = .ok l[t'])
    _ _ ?_ _ 0 b inner (le_refl _) ⟨[], rfl, by simp, fun t ht => absurd ht (by simp)⟩ hinner
  · obtain ⟨l, hl, hrl, hres⟩ := key
    have hl' : l.length = cntOf P node L.length := hl.trans (by simp [Std.Legacy.Range.size, cntOf])
    refine ⟨L ++ [l], by simp, by rw [hrl, hflat, List.flatten_append, List.flatten_singleton], ?_⟩
    intro i' hi'
    simp only [List.length_append, List.length_singleton] at hi'
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi' with hi'' | rfl
    · rw [List.getElem_append_left hi'']
      exact hL i' hi''
    · rw [List.getElem_append_right (le_refl _)]
      simp only [Nat.sub_self, List.getElem_singleton]
      exact ⟨hl', hres⟩
  intro t r' y ht hr' hy
  obtain ⟨l, rfl, hrl, hres⟩ := hr'
  try dsimp only at hy
  split at hy
  · cases hy
  rename_i v hv
  simp only [pure, Except.pure, Except.ok.injEq] at hy
  subst hy
  refine ⟨_, rfl, l ++ [v], by simp, by simp [hrl], fun t' ht' => ?_⟩
  simp only [List.length_append, List.length_singleton] at ht'
  rcases Nat.lt_succ_iff_lt_or_eq.1 ht' with ht'' | rfl
  · rw [List.getElem_append_left ht'']
    exact hres t' ht''
  · rw [List.getElem_append_right (le_refl _)]
    simp only [Nat.sub_self, List.getElem_singleton]
    exact hv

/-! ## The widths -/

theorem gate_of_gateAt {P : Prog} {g p : ℕ} {ops : Array ℕ} (h : gateAt P g = some (p, ops)) :
    ∃ sc, P.gate g = .ok (p, ops, sc) := by
  unfold gateAt at h
  rcases hgo : P.gate g with e | ⟨p', ops', sc'⟩
  · rw [hgo] at h; simp at h
  · rw [hgo] at h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨sc', rfl⟩

/-- **A member's operands are, one by one, as wide as its callee's parameter leaves**, at a node that is no scan in an
activation resolution reaches: each operand `Prog.gate` locates calls a primitive returning one value of that width. -/
theorem node_operand_widths (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ LeafWidths.BodyWide (P.defs[d]!).body)
    (sc : Scope) (hsc : Resolve.Good P sc) (k : ℕ) (hk : k < (P.defs[sc.d]!).body.nodes.size) (m : ℕ)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) (hns : ((P.defs[sc.d]!).body.nodes[k]!).form ≠ .scan)
    (ops : Array ℕ) (hops : operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops) :
    List.Forall₂ (fun g w => ∀ p o s, P.gate g = .ok (p, o, s) → (P.defs[p]!).ret = .value w) ops.toList
      ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).params.toList.flatMap fun q => q.2.widths.toList) := by
  obtain ⟨p, hfn⟩ : ∃ p, ((P.defs[sc.d]!).body.nodes[k]!).fn = p := ⟨_, rfl⟩
  rw [hfn]
  obtain ⟨L, hlen, hflat, hLi⟩ := operands_lists P sc _ m ops hops
  obtain ⟨hd, -⟩ := Resolve.good_d P hroot hB sc hsc
  obtain ⟨-, hnd, -⟩ := (hD sc.d hd).2.1 k hk
  obtain ⟨hsz, hcnt, hpr, -, hshape, -⟩ := hnd
  obtain ⟨hfnlt, -, hup, -, hop⟩ := (hB sc hsc).1 k hk
  rw [hfn] at hsz hpr hshape hop hup hfnlt
  have hform : ((P.defs[sc.d]!).body.nodes[k]!).form = .prim ∨ ((P.defs[sc.d]!).body.nodes[k]!).form = .call ∨
      ((P.defs[sc.d]!).body.nodes[k]!).form = .batch := by
    rcases hf : ((P.defs[sc.d]!).body.nodes[k]!).form
    · exact Or.inl rfl
    · exact Or.inr (Or.inl rfl)
    · exact Or.inr (Or.inr rfl)
    · exact absurd hf hns
  -- each argument resolves as many references as its parameter has leaves
  have hcntL : ∀ i < ((P.defs[sc.d]!).body.nodes[k]!).args.size,
      cntOf P ((P.defs[sc.d]!).body.nodes[k]!) i = ((P.defs[p]!).params[i]!).2.leaves := by
    intro i hi
    unfold cntOf
    rw [hfn]
    split
    · rfl
    · rename_i hnb
      rw [hcnt i hi]
      rcases hform with hf | hf | hf
      · simp only [hf, BodiesOk.Shape] at hshape
        rw [hshape.2 i hi]
      · simp only [hf, BodiesOk.Shape] at hshape
        rw [hshape.2 i hi]
      · simp only [hf, BodiesOk.Shape] at hshape
        rcases hshape.2 i hi with ⟨-, hty⟩ | ⟨ax, mt, hrem, hmt, hdims⟩
        · rw [hty]
        · obtain ⟨L', hax, -⟩ := BodiesOk.axis_spec ax _ _ mt hrem
          exact (hnb (by rw [hf, hdims, hax]; rfl)).elim
  rw [hflat]
  show List.Forall₂ _ L.flatten ((P.defs[p]!).params.toList.map fun q => q.2.widths.toList).flatten
  refine forall₂_flatten _ _ (by simp [hlen, hsz]) fun i ha hb => ?_
  have hia : i < ((P.defs[sc.d]!).body.nodes[k]!).args.size := by rw [← hlen]; exact ha
  have hip : i < (P.defs[p]!).params.size := by rw [← hsz]; exact hia
  obtain ⟨hlc, hres⟩ := hLi i ha
  simp only [List.getElem_map, Array.getElem_toList]
  refine forall₂_of_get _ _ ?_ fun t ht ht' => ?_
  · rw [hlc, hcntL i hia, Flock.Sigma.widths_length, getElem!_pos _ i hip]
  simp only [Array.getElem_toList]
  have hres := hres t ht
  have htc : t < ((P.defs[p]!).params[i]!).2.leaves := by rw [← hcntL i hia, ← hlc]; exact ht
  -- the reference is one of the node's, in range
  have hidx : ∃ u, refOf ((P.defs[sc.d]!).body.nodes[k]!) m i t =
      (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get u ∧
        u < (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.count := by
    by_cases hpp : (P.defs[p]!).prim = true
    · have htc' : t < cntOf P ((P.defs[sc.d]!).body.nodes[k]!) i := hlc ▸ ht
      unfold cntOf at htc'
      rw [hfn] at htc'
      have hu := hop hpp i hia m hm t htc'
      unfold refOf
      by_cases hb : (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true
      · rw [if_pos hb] at hu ⊢
        exact ⟨_, Resolve.memberRef_eq _ _ _, hu⟩
      · rw [if_neg hb] at hu ⊢
        exact ⟨t, rfl, hu⟩
    · have hbp := ((hD p hfnlt).1.2 (by simpa using hpp)).1
      have hu := hup i (by rw [hbp]; exact hip) t (by rw [hbp]; exact htc) m hm
      unfold refOf
      rcases hform with hf | hf | hf
      · exact absurd (hpr hf) hpp
      · simp only [hf] at hu
        rw [if_neg (by rw [hf]; decide)]
        exact ⟨t, rfl, hu⟩
      · simp only [hf] at hu
        rw [if_pos (by rw [hf]; rfl)]
        exact ⟨_, Resolve.memberRef_eq _ _ _, hu⟩
  obtain ⟨u, hu, hul⟩ := hidx
  have href0 := ((hB sc hsc).1 k hk).2.1 i u hul
  have href : Resolve.RefOk P sc.d (P.defs[sc.d]!).body.nodes.size
      ((((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get u) :=
    ⟨href0.1, fun h => ⟨Nat.lt_trans (href0.2 h).1 hk, (href0.2 h).2⟩⟩
  rw [← hu] at href
  -- the leaf it names is as wide as the parameter's
  have harg : Widths.wAt (LeafWidths.tyOf P sc (refOf ((P.defs[sc.d]!).body.nodes[k]!) m i t))
      (refOf ((P.defs[sc.d]!).body.nodes[k]!) m i t).j = Widths.wAt ((P.defs[p]!).params[i]!).2 t := by
    have hw := LeafWidths.arg_wide P hroot hB hD hW sc hsc k hk m hm i (by rw [hfn]; exact hip) t
      (by rw [hfn]; exact htc)
    rw [hfn] at hw
    unfold refOf
    rcases hform with hf | hf | hf
    · rw [if_neg (by rw [hf]; decide)]
      exact hw.1 (Or.inl hf)
    · rw [if_neg (by rw [hf]; decide)]
      exact hw.1 (Or.inr hf)
    · rw [if_pos (by rw [hf]; rfl), Resolve.memberRef_eq]
      exact hw.2 hf
  intro p' o s' hgo'
  rw [LeafWidths.resolve_wide P hroot hB hL hD hW sc hsc _ href _ hres p' o s' hgo', harg,
    LeafWidths.wAt_widths _ _ htc, getElem!_pos (P.defs[p]!).params i hip]
  have htw : t < ((P.defs[p]!).params[i]).2.widths.size := by
    rw [← Array.length_toList, Flock.Sigma.widths_length, ← getElem!_pos (P.defs[p]!).params i hip]
    exact htc
  rw [getElem!_pos _ t htw]

/-- **A gate's operands are, one by one, at most as wide as its primitive's parameter leaves**, in a decoded Program
whose root reads no input node recut and whose nodes read only earlier nodes: an operand `Prog.gate` locates is exactly
as wide, one it doesn't is of width 0. -/
theorem gate_operand_widths (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hov : P.orderViolation = none) (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) (g p : ℕ)
    (ops : Array ℕ)
    (hg : gateAt P g = some (p, ops)) :
    List.Forall₂ (· ≤ ·) (ops.toList.map (widthAt P))
      ((P.defs[p]!).params.toList.flatMap fun q => q.2.widths.toList) := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_full bytes P h hr
  have hD : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hrt, hrp, -⟩ := BodiesOk.decode_root bytes P h hr
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hrt, hrp⟩
  obtain ⟨S, hS0, hS⟩ := BodiesOk.order_closed P hov
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc hg => by
    obtain ⟨hSd, hpd⟩ := BodiesOk.good_closed P S hS0 hrp (fun d hd hp => (hS d hd hp).2) sc hg
    exact BodiesOk.body_ok P hD sc.d (hS sc.d hSd hpd).1
  have hL := decoded_laid bytes P h hr
  have hW := LeafWidths.decoded_wide bytes P h hr
  obtain ⟨sc, hgo⟩ := gate_of_gateAt hg
  obtain ⟨k, m, hsc, hk, hfn, hpp, hm, hops⟩ := gate_loc P hroot hB hL hgo
  obtain ⟨hd, -⟩ := Resolve.good_d P hroot hB sc hsc
  obtain ⟨-, ⟨-, -, -, hcomp, -⟩, -⟩ := (hD sc.d hd).2.1 k hk
  have hns : ((P.defs[sc.d]!).body.nodes[k]!).form ≠ .scan := fun hf => by
    have := hcomp (Or.inr hf)
    rw [hfn, hpp] at this
    cases this
  have hw := node_operand_widths P hroot hB hL hD hW sc hsc k hk m hm hns ops hops
  rw [hfn] at hw
  rw [List.forall₂_map_left_iff]
  refine hw.imp fun {a b} hab => ?_
  rcases hga : gateAt P a with _ | ⟨p', ops'⟩
  · simp only [widthAt, hga]
    exact Nat.zero_le _
  obtain ⟨sc', hgo'⟩ := gate_of_gateAt hga
  simp only [widthAt, hga, hab p' ops' sc' hgo', le_refl]

/-- **A gate's operands are at most as many bits as its primitive's parameters.** -/
theorem gate_operand_bits (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hov : P.orderViolation = none) (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) (g p : ℕ)
    (ops : Array ℕ) (hg : gateAt P g = some (p, ops)) :
    (ops.toList.map (widthAt P)).sum ≤ ((P.defs[p]!).params.toList.flatMap fun q => q.2.widths.toList).sum :=
  sum_le_of_forall₂ (gate_operand_widths bytes P h hov hr g p ops hg)

end InferenceOnlyVerifier.Properties.SecurityProofs.OperandWidths
