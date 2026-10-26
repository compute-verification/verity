import Properties.NetworkScan
import Proofs.Properties.Nci.BodiesOk

/-!
# The scan's layout fits its Program: where its egress and ingress gates resolve

What `NetworkCheck.Period.scanOf`'s description checks give `Properties.NetworkScan.ScanFits`, on a Program C-Flock's
decoder returns (`BodiesOk.decode_spec`: every Definition `DefOk`), laid out and walked as `NciCheck.Desc.walkable`
checks.

* Down from a node whose outputs its activation computes (`Own`: no scan, a primitive's or an `ownReturns`
  composite's), resolution never goes up through a parameter, so it ends at a gate within the node's gates, within the
  walk's fuel (`resolve_own`).
* A gate within a root node that calls a composite is located below the root, where no input node is
  (`NciCheck.Desc.inputsAtRoot`), so it is no input (`gate_not_input`); a gate within a root input node is located there
  (`gate_at_input`).
* A template member's parameter whose argument is one input node's leaves resolves, one step up and one down, to a gate
  of that input node (`resolve_param_input`).
-/

namespace Properties.NetworkScan.Fits

open Flock.Program (Prog Scope Ref Node Part Seq resolve resolveStep gateStep)
open NciCheck.Properties.SecurityProofs
open NetworkCheck.Period (ownReturns partNames)

/-- A node of body `d` whose outputs are gates its activation computes: no scan, and calling a primitive or an
`ownReturns` composite decoded before `d`. -/
def Own (P : Prog) (d : ℕ) (n : Node) : Prop :=
  n.form ≠ .scan ∧ ((P.defs[n.fn]!).prim = true ∨ (n.fn < d ∧ (ownReturns P)[n.fn]! = true))

theorem ownReturns_size (P : Prog) : (ownReturns P).size = P.defs.size :=
  (Walk.pushFold_spec _ _).1

/-- **What `ownReturns` says of a Definition**: a composite, each name of each part of its return a node of its body
whose outputs it computes (`Own`). -/
theorem ownReturns_spec (P : Prog) {d : ℕ} (hd : d < P.defs.size) (h : (ownReturns P)[d]! = true) :
    (P.defs[d]!).prim = false ∧ ∀ q ∈ (P.defs[d]!).body.ret.parts, ∀ x ∈ partNames q, x.1 = false ∧
      ∃ n, (P.defs[d]!).body.nodes[x.2]? = some n ∧ Own P d n := by
  have hsz := ownReturns_size P
  unfold ownReturns at h hsz
  rw [(Walk.pushFold_spec _ _).2 d hd] at h
  simp only [Bool.and_eq_true, Bool.not_eq_true', Array.all_eq_true_iff_forall_mem] at h
  obtain ⟨hp, hall⟩ := h
  refine ⟨hp, fun q hq x hx => ?_⟩
  have hx' := hall q hq x hx
  obtain ⟨p, k⟩ := x
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at hx'
  obtain ⟨hp0, hm⟩ := hx'
  refine ⟨hp0, ?_⟩
  split at hm
  · rename_i n hn
    refine ⟨n, hn, ?_⟩
    simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hm
    obtain ⟨hs, hpr⟩ := hm
    refine ⟨fun he => absurd (he ▸ hs) (by decide), hpr.imp id fun ⟨hlt, ho⟩ => ⟨hlt, ?_⟩⟩
    rw [Walk.getElem!_extract _ hlt (by rw [(Walk.pushFold_spec _ _).1]; omega)] at ho
    unfold ownReturns
    exact ho
  · cases hm

/-- The names `BodiesOk.partNames` gives a part are among `partNames`'s. -/
theorem names_sub (q : Part) (x : Bool × ℕ) (h : x ∈ BodiesOk.partNames q) : x ∈ partNames q := by
  cases q with
  | affine p i b c =>
    simp only [BodiesOk.partNames] at h
    split at h
    · exact h
    · simp at h
  | strided p i b inner os is c =>
    simp only [BodiesOk.partNames] at h
    split at h
    · exact h
    · simp at h
  | explicit it => exact h

/-- What `NciCheck.Desc.inputsAtRoot` says: no node calls the root, and only the root's are input nodes. -/
theorem inputsAtRoot_spec (P : Prog) (h : NciCheck.Desc.inputsAtRoot P = true) :
    ∀ d < P.defs.size, ∀ n ∈ (P.defs[d]!).body.nodes, n.fn ≠ P.root ∧
      (d = P.root ∨ NciCheck.Desc.isInputNode P n = false) := by
  intro d hd n hn
  unfold NciCheck.Desc.inputsAtRoot at h
  have := Array.all_eq_true_iff_forall_mem.1 (List.all_eq_true.1 h d (List.mem_range.2 hd)) n hn
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, Bool.or_eq_true, beq_iff_eq, Bool.not_eq_true'] at this
  exact this

theorem node_mem (b : Flock.Program.Body) {k : ℕ} (hk : k < b.nodes.size) : b.nodes[k]! ∈ b.nodes := by
  rw [getElem!_pos _ k hk]; exact Array.getElem_mem hk

/-- Two nodes whose gates hold one gate are one node, on offsets not decreasing. -/
theorem offsets_unique (offs : Array ℕ) (K : ℕ) (hmono : ∀ a b, a ≤ b → b ≤ K → offs[a]! ≤ offs[b]!) {k k' y : ℕ}
    (hk : k < K) (hk' : k' < K) (h1 : offs[k]! ≤ y) (h2 : y < offs[k + 1]!) (h1' : offs[k']! ≤ y)
    (h2' : y < offs[k' + 1]!) : k = k' := by
  rcases Nat.lt_trichotomy k k' with h | h | h
  · have := hmono (k + 1) k' (by omega) (by omega); omega
  · exact h
  · have := hmono (k' + 1) k (by omega) (by omega); omega

/-- **The node a step of `Prog.gate`'s walk finds**: the node `k` of the activation whose gates hold `g`; the step ends
there exactly when `k` calls a primitive, at that primitive, and otherwise goes into a member of `k`. -/
theorem gateStep_found (P : Prog) (g : ℕ) (sc : Scope) (x) (h : gateStep P g sc = .ok x) :
    ∃ k < (P.defs[sc.d]!).body.nodes.size, (P.defs[sc.d]!).body.offsets[k]! ≤ g - sc.off ∧
      g - sc.off < (P.defs[sc.d]!).body.offsets[k + 1]! ∧
      (∀ b, x = .inl b → (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true ∧
        b.1 = ((P.defs[sc.d]!).body.nodes[k]!).fn) ∧
      (∀ sc', x = .inr sc' → (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false ∧
        sc'.d = ((P.defs[sc.d]!).body.nodes[k]!).fn) := by
  unfold gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    refine ⟨k, hkK, hpk.1, hpk.2, ?_⟩
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hp, ite_true] at h
      split at h
      · cases h
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨fun b hb => (by cases hb; exact ⟨hp, rfl⟩), fun sc' hb => (by cases hb)⟩
    · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨fun b hb => (by cases hb), fun sc' hb => (by cases hb; exact ⟨by simpa using hp, rfl⟩)⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- A step of `Prog.gate`'s walk finds a node of a Definition the Program has. -/
theorem gateStep_d (P : Prog) (g : ℕ) (sc : Scope) (x) (h : gateStep P g sc = .ok x) : sc.d < P.defs.size := by
  obtain ⟨k, hk, -⟩ := gateStep_found P g sc x h
  refine (Nat.lt_or_ge sc.d P.defs.size).elim id fun hn => ?_
  rw [getElem!_neg P.defs sc.d (by omega)] at hk
  have h0 : (default : Flock.Program.Defn).body.nodes.size = 0 := rfl
  omega

theorem rootScope_d (P : Prog) : P.rootScope.d = P.root := rfl

theorem rootScope_off (P : Prog) : P.rootScope.off = 0 := rfl

theorem child_d (sc : Scope) (d off k m : ℕ) : (Scope.child sc d off k m).d = d := rfl

theorem child_off (sc : Scope) (d off k m : ℕ) : (Scope.child sc d off k m).off = off := rfl

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hL

/-- **A gate of a root node calling a composite is no input**: `Prog.gate`'s walk goes into the node's member, below
the root, where only primitives that are no input are (`NciCheck.Desc.inputsAtRoot`). -/
theorem gate_not_input (hin : NciCheck.Desc.inputsAtRoot P = true) {k g p : ℕ} {ops : Array ℕ} {sc : Scope}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hkc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (h1 : (P.defs[P.root]!).body.offsets[k]! ≤ g) (h2 : g < (P.defs[P.root]!).body.offsets[k + 1]!)
    (h : P.gate g = .ok (p, ops, sc)) : Flock.Program.isInputId (P.defs[p]!).key = false := by
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hspec := inputsAtRoot_spec P hin
  unfold Flock.Program.Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (gateStep P g) (fun a => a.d ≠ P.root ∨ a = P.rootScope)
    (fun b => Flock.Program.isInputId (P.defs[b.1]!).key = false) ?_ ?_ Flock.Steps.FUEL P.rootScope (p, ops, sc)
    (Or.inr rfl) h
  · intro a b ha hb
    have had := gateStep_d P g a _ hb
    obtain ⟨k', hk', -, -, hinl, -⟩ := gateStep_found P g a _ hb
    obtain ⟨hp, hb1⟩ := hinl b rfl
    rcases ha with ha | rfl
    · obtain ⟨-, hd⟩ := hspec a.d had _ (node_mem _ hk')
      rcases hd with hd | hd
      · exact absurd hd ha
      · rw [hb1]
        simpa [NciCheck.Desc.isInputNode, hp] using hd
    · obtain ⟨k'', hk'', h1', h2', hinl', -⟩ := gateStep_found P g P.rootScope _ hb
      simp only [rootScope_d, rootScope_off, Nat.sub_zero] at hk'' h1' h2' hinl'
      have e : k'' = k := offsets_unique _ _ hmono hk'' hk h1' h2' h1 h2
      subst e
      exact absurd (hinl' b rfl).1 (by simp [hkc])
  · intro a a' _ ha'
    have had := gateStep_d P g a _ ha'
    obtain ⟨k', hk', -, -, -, hinr⟩ := gateStep_found P g a _ ha'
    obtain ⟨-, hd'⟩ := hinr a' rfl
    exact Or.inl (hd' ▸ (hspec a.d had _ (node_mem _ hk')).1)

/-- **A gate of a root node calling a primitive is located there**: `Prog.gate`'s walk ends at the root's first step,
at that node's primitive. -/
theorem gate_at_prim {k g p : ℕ} {ops : Array ℕ} {sc : Scope}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hkp : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = true)
    (h1 : (P.defs[P.root]!).body.offsets[k]! ≤ g) (h2 : g < (P.defs[P.root]!).body.offsets[k + 1]!)
    (h : P.gate g = .ok (p, ops, sc)) : p = ((P.defs[P.root]!).body.nodes[k]!).fn := by
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  unfold Flock.Program.Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (gateStep P g) (fun a => a = P.rootScope)
    (fun b => b.1 = ((P.defs[P.root]!).body.nodes[k]!).fn) ?_ ?_ Flock.Steps.FUEL P.rootScope (p, ops, sc) rfl h
  · rintro a b rfl hb
    obtain ⟨k', hk', h1', h2', hinl, -⟩ := gateStep_found P g P.rootScope _ hb
    simp only [rootScope_d, rootScope_off, Nat.sub_zero] at hk' h1' h2' hinl
    have e : k' = k := offsets_unique _ _ hmono hk' hk h1' h2' h1 h2
    subst e
    exact (hinl b rfl).2
  · rintro a a' rfl ha'
    obtain ⟨k', hk', h1', h2', -, hinr⟩ := gateStep_found P g P.rootScope _ ha'
    simp only [rootScope_d, rootScope_off, Nat.sub_zero] at hk' h1' h2' hinr
    have e : k' = k := offsets_unique _ _ hmono hk' hk h1' h2' h1 h2
    subst e
    exact absurd hkp (by simp [(hinr a' rfl).1])

end

/-- **Reference `starts[i] + x` of a sequence** is part `i`'s element `x`, for `x` within the part. -/
theorem seq_get_at (ps : Array Part) {i x : ℕ} (hi : i < ps.size) (hx : x < ps[i]!.count) :
    (Seq.ofParts ps).starts[i]! + x < (Seq.ofParts ps).count ∧
      (Seq.ofParts ps).get ((Seq.ofParts ps).starts[i]! + x) = ps[i]!.get x := by
  obtain ⟨-, -, hst, hcnt⟩ := Walk.ofParts_spec ps
  have hst' : ∀ i < ps.size, (Seq.ofParts ps).starts[i]! + ps[i]!.count =
      ((ps.toList.take (i + 1)).map Part.count).sum := fun i hi => by rw [hst i hi, Walk.sum_take_succ_map ps hi]
  have hmono : ∀ a b, a ≤ b →
      ((ps.toList.take a).map Part.count).sum ≤ ((ps.toList.take b).map Part.count).sum := by
    intro a b hab
    obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le hab
    rw [List.take_add, List.map_append, List.sum_append]
    omega
  have hle : ((ps.toList.take (i + 1)).map Part.count).sum ≤ (Seq.ofParts ps).count := by
    rw [hcnt]
    have := hmono (i + 1) ps.toList.length (by simp only [Array.length_toList]; omega)
    rwa [List.take_length] at this
  have hlt : (Seq.ofParts ps).starts[i]! + x < (Seq.ofParts ps).count := by
    have := hst' i hi; omega
  refine ⟨hlt, ?_⟩
  obtain ⟨i', hi', h1, h2, hget⟩ := Walk.seq_get ps hlt
  have e : i' = i := by
    rcases Nat.lt_trichotomy i' i with h | h | h
    · have := hst' i' hi'
      have := hmono (i' + 1) i (by omega)
      have := hst i hi
      omega
    · exact h
    · have := hst' i hi
      have := hmono (i + 1) i' (by omega)
      have := hst i' hi'
      omega
  subst e
  rw [hget, Nat.add_sub_cancel_left]

/-- Resolution's state on its way down from a node whose outputs its activation computes: a reference to an output leaf
of such a node (`Own`) of a composite's body, the node's gates within `[lo, hi)`. -/
def Down (P : Prog) (lo hi : ℕ) (s : Scope × Ref) : Prop :=
  s.1.d < P.defs.size ∧ (P.defs[s.1.d]!).prim = false ∧ s.2.p = false ∧
    s.2.i < (P.defs[s.1.d]!).body.nodes.size ∧ s.2.j < ((P.defs[s.1.d]!).body.nodes[s.2.i]!).out.leaves ∧
    Own P s.1.d ((P.defs[s.1.d]!).body.nodes[s.2.i]!) ∧
    lo ≤ s.1.off + (P.defs[s.1.d]!).body.offsets[s.2.i]! ∧ s.1.off + (P.defs[s.1.d]!).body.offsets[s.2.i + 1]! ≤ hi

section
variable (P : Prog) (hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]!)
  (hL : ∀ d < P.defs.size, Walk.LaidOut P d) (hc : NciCheck.Desc.calleesFirst P = true)
include hok hL hc

/-- **A composite that `ownReturns` takes returns its own nodes' leaves**: its return's leaf `t` is an output leaf of a
node of its body whose outputs it computes (`Own`). -/
theorem ret_own {d t : ℕ} (hd : d < P.defs.size) (ho : (ownReturns P)[d]! = true)
    (ht : t < (P.defs[d]!).body.ret.count) :
    ((P.defs[d]!).body.ret.get t).p = false ∧ ((P.defs[d]!).body.ret.get t).i < (P.defs[d]!).body.nodes.size ∧
      ((P.defs[d]!).body.ret.get t).j < ((P.defs[d]!).body.nodes[((P.defs[d]!).body.ret.get t).i]!).out.leaves ∧
      Own P d ((P.defs[d]!).body.nodes[((P.defs[d]!).body.ret.get t).i]!) := by
  obtain ⟨-, hown⟩ := ownReturns_spec P hd ho
  obtain ⟨-, hret, ps, hps, -⟩ := (hok d hd).2
  obtain ⟨q, hq, hx⟩ := BodiesOk.names_of _ ps hps t ht
  obtain ⟨hp, n, hn, hown'⟩ := hown q hq _ (names_sub q _ hx)
  dsimp only at hp hn
  have hin := (hret t ht).2 hp
  refine ⟨hp, hin.1, hin.2, ?_⟩
  have e : (P.defs[d]!).body.nodes[((P.defs[d]!).body.ret.get t).i]! = n := by
    rw [getElem!_pos (P.defs[d]!).body.nodes _ hin.1]
    simpa [hin.1] using hn
  rw [e]; exact hown'

/-- **One step down keeps `Down`**: from a node whose outputs its activation computes, resolution ends at one of the
node's gates (a primitive's, or a batch of them) or goes into the member computing the leaf, at a node of the callee's
whose outputs it computes, within the member's gates, one callee deeper. -/
theorem down_step (lo hi : ℕ) (s : Scope × Ref) (hs : Down P lo hi s) :
    (∃ b, resolveStep P s = .ok (.inl b) ∧ lo ≤ b ∧ b < hi) ∨
      ∃ s', resolveStep P s = .ok (.inr s') ∧ Down P lo hi s' ∧
        (NciCheck.Desc.depths P)[s'.1.d]! < (NciCheck.Desc.depths P)[s.1.d]! := by
  obtain ⟨sc, ⟨p, i, j⟩⟩ := s
  obtain ⟨hd, hpr, hp, hi, hj, ⟨hns, hcall⟩, hlo, hhi⟩ := hs
  dsimp only at hd hpr hp hi hj hns hcall hlo hhi
  subst hp
  have hmem := node_mem _ hi
  have hfn : ((P.defs[sc.d]!).body.nodes[i]!).fn < sc.d := Walk.calleesFirst_spec P hc sc.d hd _ hmem
  have hfs : ((P.defs[sc.d]!).body.nodes[i]!).fn < P.defs.size := by omega
  obtain ⟨-, hnok, -⟩ := (hok sc.d hd).2.1 i hi
  have hdown := BodiesOk.down_ok _ _ (hok _ hfs).1 hnok j hj
  have hlay := ((hL sc.d hd) hpr).2.2 i hi
  have hdep := Walk.depths_child P hc hd hpr hmem
  rcases hf : ((P.defs[sc.d]!).body.nodes[i]!).form with _ | _ | _ | _
  · rw [hf] at hdown; dsimp only at hdown
    obtain ⟨-, hg1⟩ := hdown
    rw [Resolve.prim_members _ (Or.inl hf), hg1] at hlay
    exact Or.inl ⟨_, Resolve.rs_prim P sc i j hf, hlo, by omega⟩
  · rw [hf] at hdown; dsimp only at hdown
    obtain ⟨hcp, hjr⟩ := hdown
    have ho : (ownReturns P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! = true := by
      rcases hcall with h | ⟨-, h⟩
      · rw [hcp] at h; cases h
      · exact h
    obtain ⟨hp', hi', hj', hown'⟩ := ret_own P hok hL hc hfs ho hjr
    have hm := Resolve.offsets_mono P _ (hL _ hfs) hcp (_ + 1) _ hi' (le_refl _)
    have hend := ((hL _ hfs) hcp).2.1
    rw [Resolve.prim_members _ (Or.inr hf)] at hlay
    refine Or.inr ⟨_, Resolve.rs_call P sc i j hf, ⟨hfs, hcp, hp', hi', hj', hown', ?_, ?_⟩, hdep⟩
    · simp only [child_d, child_off]; omega
    · simp only [child_d, child_off]; omega
  · rw [hf] at hdown; dsimp only at hdown
    obtain ⟨hjn, hpg, hpr'⟩ := hdown
    rw [Resolve.bs_members _ (Or.inl hf)] at hlay
    have hstep := Resolve.rs_batch P sc i j hf
    have hq := Nat.zero_le (j / (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).outLeaves)
    by_cases hcp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
    · simp only [hcp, ite_true] at hstep
      rw [hpg hcp] at hlay
      exact Or.inl ⟨_, hstep, by omega, by omega⟩
    · have hcp' : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = false := by simpa using hcp
      simp only [hcp', Bool.false_eq_true, ite_false] at hstep
      have ho : (ownReturns P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! = true := by
        rcases hcall with h | ⟨-, h⟩
        · exact absurd h hcp
        · exact h
      obtain ⟨hp', hi', hj', hown'⟩ := ret_own P hok hL hc hfs ho (hpr' hcp')
      have hm := Resolve.offsets_mono P _ (hL _ hfs) hcp' (_ + 1) _ hi' (le_refl _)
      have hend := ((hL _ hfs) hcp').2.1
      have hmul : (j / (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).outLeaves + 1) *
          (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).gates ≤
          ((P.defs[sc.d]!).body.nodes[i]!).n * (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).gates :=
        Nat.mul_le_mul_right _ hjn
      rw [Nat.add_mul, Nat.one_mul] at hmul
      generalize j / (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).outLeaves *
        (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).gates = q at hstep hmul
      refine Or.inr ⟨_, hstep, ⟨hfs, hcp', hp', hi', hj', hown', ?_, ?_⟩, hdep⟩
      · simp only [child_d, child_off]; omega
      · simp only [child_d, child_off]; omega
  · exact absurd hf hns

/-- **Down from a node whose outputs its activation computes, resolution ends within the node's gates**, within the
walk's fuel: each step goes one callee deeper (`down_step`), and the activation's callees nest less deep than the
fuel. -/
theorem resolve_own {lo hi : ℕ} (sc : Scope) (r : Ref) (hs : Down P lo hi (sc, r))
    (hdep : (NciCheck.Desc.depths P)[sc.d]! < Flock.Steps.FUEL) :
    ∃ a, resolve P sc r = .ok a ∧ lo ≤ a ∧ a < hi :=
  Walk.run_of_potential (what := "scope resolution") (Down P lo hi)
    (fun s => (NciCheck.Desc.depths P)[s.1.d]!) (fun b => lo ≤ b ∧ b < hi) (down_step P hok hL hc lo hi)
    Flock.Steps.FUEL (sc, r) hs hdep

/-- **A template member's parameter whose argument is one input node's leaves resolves to that node's gate**: up
through root node `k` (a batch of a composite) to its argument's reference, which is leaf `memberIdx` of root node
`kr`, an input node, and down to that leaf's gate, a gate of `kr`. -/
theorem resolve_param_input (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false) {k m j kr c : ℕ}
    (hk : k < (P.defs[P.root]!).body.nodes.size) (hb : ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    (hcp : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (ha : (((P.defs[P.root]!).body.nodes[k]!).args[0]!).seq.parts = #[.affine false kr 0 c])
    (hin : NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[kr]!) = true)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).n)
    (hp0 : 0 < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params.size)
    (hj : j < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[0]!).2.leaves) :
    ∃ g, resolve P (P.child P.rootScope k m) ⟨true, 0, j⟩ = .ok g ∧ kr < (P.defs[P.root]!).body.nodes.size ∧
      (P.defs[P.root]!).body.offsets[kr]! ≤ g ∧ g < (P.defs[P.root]!).body.offsets[kr + 1]! := by
  have hmem := node_mem _ hk
  have hfn := Walk.calleesFirst_spec P hc P.root hroot.1 _ hmem
  have hfs : ((P.defs[P.root]!).body.nodes[k]!).fn < P.defs.size := by omega
  obtain ⟨-, hnok, hargs⟩ := (hok P.root hroot.1).2.1 k hk
  have hcok := (hok _ hfs).1
  have hup := BodiesOk.up_ok _ _ hcok hnok
  have hbp := (hcok.2 hcp).1
  have hasz : 0 < ((P.defs[P.root]!).body.nodes[k]!).args.size := by rw [hnok.1]; exact hp0
  have hidx := hup 0 (by rw [hbp]; exact hp0) j (by rw [hbp]; exact hj) m
    (by rw [Resolve.bs_members _ (Or.inl hb)]; exact hm)
  rw [hb] at hidx; dsimp only at hidx
  have hget : (((P.defs[P.root]!).body.nodes[k]!).args[0]!).seq.get
      (Resolve.memberIdx (((P.defs[P.root]!).body.nodes[k]!).args[0]!) m j) =
      ⟨false, kr, Resolve.memberIdx (((P.defs[P.root]!).body.nodes[k]!).args[0]!) m j⟩ := by
    have h0 : (#[.affine false kr 0 c] : Array Part)[0]! = .affine false kr 0 c := rfl
    unfold Flock.Program.Seq.get
    rw [if_pos (by simp [ha])]
    simp only [ha, h0, Flock.Program.Part.get, Nat.zero_add]
    try rfl
  have href := ((hargs 0 hasz).2 _ hidx).2
  rw [hget] at href
  obtain ⟨hkr, hlv⟩ : kr < (P.defs[P.root]!).body.nodes.size ∧
      Resolve.memberIdx (((P.defs[P.root]!).body.nodes[k]!).args[0]!) m j <
        ((P.defs[P.root]!).body.nodes[kr]!).out.leaves := href rfl
  set x := Resolve.memberIdx (((P.defs[P.root]!).body.nodes[k]!).args[0]!) m j with hx
  have hmemr := node_mem _ hkr
  have hfnr := Walk.calleesFirst_spec P hc P.root hroot.1 _ hmemr
  have hfsr : ((P.defs[P.root]!).body.nodes[kr]!).fn < P.defs.size := by omega
  obtain ⟨-, hnokr, -⟩ := (hok P.root hroot.1).2.1 kr hkr
  have hcokr := (hok _ hfsr).1
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[kr]!).fn]!).prim = true := by
    simp only [NciCheck.Desc.isInputNode, Bool.and_eq_true] at hin; exact hin.1
  have hdownr := BodiesOk.down_ok _ _ hcokr hnokr x hlv
  have hlayr := ((hL P.root hroot.1) hroot.2).2.2 kr hkr
  have hs1 : resolveStep P (P.child P.rootScope k m, ⟨true, 0, j⟩) =
      .ok (.inr (P.rootScope, ⟨false, kr, x⟩)) := by
    simp only [Flock.Program.Prog.child]
    rw [Resolve.rs_up_batch P P.rootScope _ _ k m 0 j hb, rootScope_d, Resolve.memberRef_eq, ← hx, hget]
  have hrun : ∀ g, resolveStep P (P.rootScope, ⟨false, kr, x⟩) = .ok (.inl g) →
      resolve P (P.child P.rootScope k m) ⟨true, 0, j⟩ = .ok g := by
    intro g hg
    have h2 : Flock.Steps.run "scope resolution" (resolveStep P) (0 + 1 + 1) (P.child P.rootScope k m, ⟨true, 0, j⟩)
        = .ok g := by
      rw [Flock.Steps.run.eq_2, hs1]
      show Flock.Steps.run "scope resolution" (resolveStep P) (0 + 1) (P.rootScope, ⟨false, kr, x⟩) = .ok g
      rw [Flock.Steps.run.eq_2, hg]
      rfl
    exact Flock.Steps.run_mono (m := Flock.Steps.FUEL) (by unfold Flock.Steps.FUEL; norm_num) h2
  rcases hf : ((P.defs[P.root]!).body.nodes[kr]!).form with _ | _ | _ | _
  · rw [hf] at hdownr; dsimp only at hdownr
    rw [Resolve.prim_members _ (Or.inl hf), hdownr.2] at hlayr
    refine ⟨_, hrun _ (Resolve.rs_prim P P.rootScope kr x hf), hkr, ?_, ?_⟩ <;>
      simp only [rootScope_off, rootScope_d] <;> omega
  · exact absurd hprim (by simp [hnokr.2.2.2.1 (Or.inl hf)])
  · rw [hf] at hdownr; dsimp only at hdownr
    have hstep := Resolve.rs_batch P P.rootScope kr x hf
    simp only [rootScope_d] at hstep
    simp only [hprim, ite_true, (hcokr.1 hprim).2.2.1, Nat.div_one] at hstep
    rw [Resolve.bs_members _ (Or.inl hf), (hcokr.1 hprim).2.1] at hlayr
    have hxn := hdownr.1
    rw [(hcokr.1 hprim).2.2.1, Nat.div_one] at hxn
    refine ⟨_, hrun _ hstep, hkr, ?_, ?_⟩ <;> simp only [rootScope_off] <;> omega
  · exact absurd hprim (by simp [hnokr.2.2.2.1 (Or.inr hf)])

end

end Properties.NetworkScan.Fits
