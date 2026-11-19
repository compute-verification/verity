module

public import Properties.Flock.Circuit.OfProgram
public import Proofs.Properties.InferenceOnly.RecordLeafWide

/-!
# A gate's operands are as wide as its primitive's parameters

`gate_operand_widths`: on a Program the record check reads, the operands `Prog.gate` locates for a gate, read gate by
gate at their widths, are its primitive's parameter leaves' widths in order. The walk ends at member `m` of a primitive's
node (`gate_last`); `operands` resolves each argument's references in order (`operands_list`), as many as the parameter
has leaves; each resolves to a gate as wide as the leaf its reference names (`LeafWidths.resolve_wide`), located
(`gate_located`), and that leaf is as wide as the parameter's (`LeafWidths.arg_wide`).
-/

@[expose] public section

namespace FlockSoundness.Inline

open Flock.Program (Prog Scope Node Form Defn operands resolve gateStep)
open Flock.OfProgram (widthAt gateAt)
open InferenceOnlyVerifier.Properties.SecurityProofs
open InferenceOnlyVerifier.Properties.Record.SecurityProofs (ReadProg read_wide gate_located widths_size)

/-- A gate `Prog.gate` locates is a gate of the root's. -/
theorem gate_lt {P : Prog} {g : ℕ} {x : ℕ × Array ℕ × Scope} (h : P.gate g = .ok x) :
    g < (P.defs[P.root]!).gates := by
  unfold Prog.gate at h
  by_contra hge
  simp [Nat.not_lt.1 hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- **Where `Prog.gate`'s walk ends**: at member `m` of a primitive's node `k` of an activation resolution reaches, with
that member's operands. -/
theorem gate_last (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    {g p : ℕ} {ops : Array ℕ} {sc : Scope} (h : P.gate g = .ok (p, ops, sc)) :
    ∃ k m, Resolve.Good P sc ∧ k < (P.defs[sc.d]!).body.nodes.size ∧
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true ∧ m < ((P.defs[sc.d]!).body.nodes[k]!).members ∧
      p = ((P.defs[sc.d]!).body.nodes[k]!).fn ∧ operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops := by
  unfold Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (gateStep P g) (Resolve.WalkQ P g)
    (fun b => ∃ k m, Resolve.Good P b.2.2 ∧ k < (P.defs[b.2.2.d]!).body.nodes.size ∧
      (P.defs[((P.defs[b.2.2.d]!).body.nodes[k]!).fn]!).prim = true ∧
      m < ((P.defs[b.2.2.d]!).body.nodes[k]!).members ∧ b.1 = ((P.defs[b.2.2.d]!).body.nodes[k]!).fn ∧
      operands P b.2.2 ((P.defs[b.2.2.d]!).body.nodes[k]!) m = .ok b.2.1) ?_ ?_ _ _ (p, ops, sc)
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
          exact ⟨k, _, hg, hkK, hpc', hm1, rfl, hops⟩
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
        · simp only [Flock.Program.Prog.child, Flock.Program.Scope.off.eq_2]
          omega
        · simp only [Flock.Program.Prog.child, Flock.Program.Scope.off.eq_2, Flock.Program.Scope.d.eq_2]
          omega
    · simp [throw, throwThe, MonadExceptOf.throw] at hb

private theorem forIn_collect {α ε : Type} (S : α → List ℕ → Prop) :
    ∀ (l : List α) (f : α → Array ℕ → Except ε (ForInStep (Array ℕ))) (b r : Array ℕ),
      (∀ x ∈ l, ∀ b r, f x b = .ok r → ∃ ys, S x ys ∧ r = .yield ⟨b.toList ++ ys⟩) →
      forIn l b f = .ok r → ∃ yss, r.toList = b.toList ++ yss.flatten ∧ List.Forall₂ S l yss
  | [], f, b, r, _, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], by simp [h], .nil⟩
  | x :: l, f, b, r, hf, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x b with e | s
    · simp [hx, bind, Except.bind] at h
    · obtain ⟨ys, hS, rfl⟩ := hf x List.mem_cons_self b _ hx
      simp only [hx, bind, Except.bind] at h
      obtain ⟨yss, hr, h2⟩ := forIn_collect S l f _ r (fun y hy => hf y (List.mem_cons_of_mem _ hy)) h
      exact ⟨ys :: yss, by simp [hr], .cons hS h2⟩

private theorem forall₂_singletons {α : Type} {R : α → ℕ → Prop} : ∀ {l : List α} {yss : List (List ℕ)},
    List.Forall₂ (fun t ys => ∃ x, ys = [x] ∧ R t x) l yss → List.Forall₂ R l yss.flatten
  | _, _, .nil => .nil
  | _, _, .cons ⟨_, rfl, hx⟩ h => .cons hx (forall₂_singletons h)

/-- **What `operands` returns, in order**: per argument `i`, the resolutions of its references `t` below the count
`operands` reads it to. -/
theorem operands_list (P : Prog) (sc : Scope) (node : Node) (m : ℕ) (ops : Array ℕ)
    (h : operands P sc node m = .ok ops) :
    ∃ xss : List (List ℕ), ops.toList = xss.flatten ∧
      List.Forall₂ (fun i xs => List.Forall₂ (fun t x =>
          resolve P sc (if (node.form == Form.batch) = true then (node.args[i]!).memberRef m t
            else (node.args[i]!).seq.get t) = .ok x)
          (List.range (if (node.form == Form.batch && (node.args[i]!).dims.isSome) = true then
            ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count)) xs)
        (List.range node.args.size) xss := by
  unfold operands at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', bind, Except.bind] at h
  split at h
  · cases h
  · rename_i ops' hops
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at hops
    rw [List.range_eq_range']
    obtain ⟨xss, hx, h2⟩ := forIn_collect _ _ _ _ _ ?_ hops
    · exact ⟨xss, by simpa using hx, h2⟩
    intro i _ b r hr
    split at hr
    · cases hr
    · rename_i inner hinner
      simp only [pure, Except.pure, Except.ok.injEq] at hr
      subst hr
      simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at hinner
      obtain ⟨yss, hy, h3⟩ := forIn_collect _ _ _ _ _ ?_ hinner
      · refine ⟨yss.flatten, ?_, ?_⟩
        · rw [List.range_eq_range']
          exact forall₂_singletons h3
        · congr 1
          exact Array.ext' hy
      intro t _ b' r' hr'
      split at hr'
      · cases hr'
      · rename_i v hv
        simp only [pure, Except.pure, Except.ok.injEq] at hr'
        subst hr'
        exact ⟨[v], ⟨v, rfl, hv⟩, by congr 1; exact Array.ext' (by simp)⟩

/-- A primitive's node reads as many references of each argument as the parameter has leaves. -/
private theorem cnt_eq {c : Defn} {nd : Node} (hf : nd.form = .prim ∨ nd.form = .batch) {i : ℕ}
    (hcnt : (nd.args[i]!).seq.count = (nd.args[i]!).ty.leaves)
    (hshape : BodiesOk.Shape c nd.form nd.args nd.out nd.n nd.nxs) (hi : i < nd.args.size) :
    (if (nd.form == Form.batch && (nd.args[i]!).dims.isSome) = true then (c.params[i]!).2.leaves
      else (nd.args[i]!).seq.count) = (c.params[i]!).2.leaves := by
  rcases hf with hf | hf
  · simp only [hf, BodiesOk.Shape] at hshape
    simp only [hf, BodiesOk.beq_prim_batch, Bool.false_and, Bool.false_eq_true, ite_false, hcnt, hshape.2 i hi]
  · simp only [hf, BodiesOk.Shape] at hshape
    rcases hshape.2 i hi with ⟨hdn, hty⟩ | ⟨ax, mt, hrem, hmt, hdims⟩
    · simp only [hdn, Option.isSome_none, Bool.and_false, Bool.false_eq_true, ite_false, hcnt, hty]
    · obtain ⟨L, hax, -⟩ := BodiesOk.axis_spec ax _ _ mt hrem
      simp only [hf, BodiesOk.beq_batch_batch, hdims, hax, Option.isSome_some, Bool.and_self, ite_true]

/-- An operand at leaf `t` of argument `i` of member `m` of a primitive's node is as wide as the parameter's leaf. -/
private theorem leaf_width {P : Prog} (hP : ReadProg P) {sc : Scope} (hsc : Resolve.Good P sc) {k m : ℕ}
    (hk : k < (P.defs[sc.d]!).body.nodes.size)
    (hpk : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) {i t a : ℕ}
    (hi : i < ((P.defs[sc.d]!).body.nodes[k]!).args.size)
    (ht : t < ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).params[i]!).2.leaves)
    (hres : resolve P sc (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
        (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).memberRef m t
      else (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get t) = .ok a)
    (hag : a < (P.defs[P.root]!).gates) :
    widthAt P a = ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).params[i]!).2.widths[t]! := by
  obtain ⟨hroot, hB, hL, hD, hW⟩ := read_wide hP
  obtain ⟨hpd, -⟩ := Resolve.good_d P hroot hB sc hsc
  obtain ⟨-, ⟨hsz, hcnt, -, hco, hshape, -⟩, -⟩ := (hD sc.d hpd).2.1 k hk
  obtain ⟨-, hargs, -, -, hop⟩ := (hB sc hsc).1 k hk
  obtain ⟨p', ops', sc', -, hgate, -, -⟩ := gate_located hP hag
  have hi' : i < (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).params.size := hsz ▸ hi
  have haw := LeafWidths.arg_wide P hroot hB hD hW sc hsc k hk m hm i hi' t ht
  have hf : ((P.defs[sc.d]!).body.nodes[k]!).form = .prim ∨ ((P.defs[sc.d]!).body.nodes[k]!).form = .batch := by
    rcases hfm : ((P.defs[sc.d]!).body.nodes[k]!).form
    · exact Or.inl rfl
    · exact absurd (hco (Or.inl hfm)) (by rw [hpk]; decide)
    · exact Or.inr rfl
    · exact absurd (hco (Or.inr hfm)) (by rw [hpk]; decide)
  have htc := cnt_eq hf (hcnt i hi) hshape hi
  have hidx := hop hpk i hi m hm t (by rw [htc]; exact ht)
  rw [Resolve.memberRef_eq] at hres
  have hres' : resolve P sc ((((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get
      (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
        Resolve.memberIdx (((P.defs[sc.d]!).body.nodes[k]!).args[i]!) m t else t)) = .ok a := by
    split <;> simp_all
  have href0 := hargs i _ hidx
  have href : Resolve.RefOk P sc.d (P.defs[sc.d]!).body.nodes.size
      ((((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get
        (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
          Resolve.memberIdx (((P.defs[sc.d]!).body.nodes[k]!).args[i]!) m t else t)) :=
    ⟨href0.1, fun h => ⟨Nat.lt_trans (href0.2 h).1 hk, (href0.2 h).2⟩⟩
  have hret := LeafWidths.resolve_wide P hroot hB hL hD hW sc hsc _ href a hres' p' ops' sc' hgate
  have hw : Flock.Program.Widths.wAt (LeafWidths.tyOf P sc ((((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get
        (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
          Resolve.memberIdx (((P.defs[sc.d]!).body.nodes[k]!).args[i]!) m t else t)))
      ((((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get
        (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
          Resolve.memberIdx (((P.defs[sc.d]!).body.nodes[k]!).args[i]!) m t else t)).j =
      Flock.Program.Widths.wAt ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).params[i]!).2 t := by
    rcases hf with hf | hf
    · simp only [hf, BodiesOk.beq_prim_batch, Bool.false_eq_true, ite_false]
      exact haw.1 (Or.inl hf)
    · simp only [hf, BodiesOk.beq_batch_batch, ite_true]
      exact haw.2 hf
  unfold widthAt gateAt
  rw [hgate]
  simp only [hret, hw, LeafWidths.wAt_widths _ _ ht]

/-- **D3: a gate's operands are as wide as its primitive's parameters**, leaf by leaf in order, on a Program the
record check reads. -/
theorem gate_operand_widths {P : Prog} (hP : ReadProg P) {g p : ℕ} {ops : Array ℕ} {sc : Scope}
    (hg : P.gate g = .ok (p, ops, sc)) :
    ops.toList.map (widthAt P) = (P.defs[p]!).params.toList.flatMap (fun q => q.2.widths.toList) := by
  obtain ⟨hroot, hB, hL, hD, -⟩ := read_wide hP
  have hgg := gate_lt hg
  have hlt := Resolve.gate_operands_lt P hroot hB hL hg
  obtain ⟨k, m, hsc, hk, hpk, hm, rfl, hops⟩ := gate_last P hroot hB hL hg
  obtain ⟨xss, hx, h2⟩ := operands_list P sc _ m ops hops
  obtain ⟨hpd, -⟩ := Resolve.good_d P hroot hB sc hsc
  obtain ⟨-, ⟨hsz, hcnt, -, hco, hshape, -⟩, -⟩ := (hD sc.d hpd).2.1 k hk
  have hf : ((P.defs[sc.d]!).body.nodes[k]!).form = .prim ∨ ((P.defs[sc.d]!).body.nodes[k]!).form = .batch := by
    rcases hfm : ((P.defs[sc.d]!).body.nodes[k]!).form
    · exact Or.inl rfl
    · exact absurd (hco (Or.inl hfm)) (by rw [hpk]; decide)
    · exact Or.inr rfl
    · exact absurd (hco (Or.inr hfm)) (by rw [hpk]; decide)
  rw [hx, List.map_flatten, List.flatMap]
  congr 1
  rw [List.forall₂_iff_getElem] at h2
  obtain ⟨hlen, hall⟩ := h2
  simp only [List.length_range] at hlen
  apply List.ext_getElem (by simp [← hlen, hsz])
  intro i h1 h3
  simp only [List.length_map] at h1
  have hi : i < ((P.defs[sc.d]!).body.nodes[k]!).args.size := by rw [← hlen] at h1; exact h1
  have hxi := hall i (by simp [hi]) h1
  simp only [List.getElem_range] at hxi
  rw [List.forall₂_iff_getElem] at hxi
  obtain ⟨hlen', hall'⟩ := hxi
  simp only [List.length_range, cnt_eq hf (hcnt i hi) hshape hi] at hlen'
  simp only [List.getElem_map, Array.getElem_toList]
  have hi' : i < (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).params.size := hsz ▸ hi
  rw [← getElem!_pos _ i hi']
  apply List.ext_getElem (by simp [← hlen', widths_size])
  intro t ht1 ht2
  simp only [List.length_map] at ht1
  have ht : t < ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).params[i]!).2.leaves := hlen' ▸ ht1
  have hres := hall' t (by simp [cnt_eq hf (hcnt i hi) hshape hi, ht]) ht1
  simp only [List.getElem_range] at hres
  have hmem : xss[i][t] ∈ ops := by
    rw [← Array.mem_toList_iff, hx]
    exact List.mem_flatten.2 ⟨_, List.getElem_mem _, List.getElem_mem _⟩
  simp only [List.getElem_map, Array.getElem_toList]
  rw [leaf_width hP hsc hk hpk hm hi ht hres (Nat.lt_trans (hlt _ hmem) hgg), getElem!_pos]

end FlockSoundness.Inline
