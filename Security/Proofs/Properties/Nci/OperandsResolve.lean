import Proofs.Properties.Nci.Resolve

/-!
# Operands resolve on the walk: `Walk.OperandsResolve`, from `Resolve.BodiesOk`

`Flock.Program.resolve` walks up through parameters and down through node outputs. Numbering the root's activations
parents first and members in order, two places each (`pos`, through `seg`, the activations a body holds before a
node), resolution's state has a potential (`phi`): just past its activation's place for a parameter's leaf, past the
places of node `i`'s members for a leaf of node `i`. Each step down goes into a member of the node, each step up to a
parameter or an earlier node of the parent, and a scan's carry to the iteration before, so each lowers it
(`step_progress`); it stays below four times the root's activations (`phi_lt`), which `Desc.walkBound` keeps within
`Flock.Steps.FUEL`. No step refuses: `Desc.resolvable` gives the root no parameters and every scan of no iterations no
output leaves, and `Resolve.BodyOk` the scan step's `(carry, y)`. So resolution from any state `Resolve.Inv` holds of
ends at a gate (`resolve_ok`), and a primitive node's operands, which `Resolve.BodyOk`'s `OpOk` reads in range, all
resolve (`operands_prim`) in every activation `Prog.gate`'s walk reaches (`reach_walkQ`): `operandsResolve_of` proves
`Walk.OperandsResolve` from `Resolve.BodiesOk`.
-/

namespace NciCheck.Properties.SecurityProofs.OperandsResolve

open Flock.Program

/-- The activations one activation of Definition `d` holds in its nodes before node `k`. -/
def seg (P : Prog) (d k : ℕ) : ℕ :=
  (((P.defs[d]!).body.nodes.toList.take k).map fun n => n.members * (Desc.activations P)[n.fn]!).sum

/-- An activation's place among the root's activations, each given two places: its parent's, then one, then two for
each activation the parent's earlier nodes and earlier members of its node hold. -/
def pos (P : Prog) : Scope → ℕ
  | .top _ _ _ => 0
  | .child parent d _ k m => pos P parent + 1 + 2 * seg P parent.d k + m * (2 * (Desc.activations P)[d]!)

/-- Resolution's potential: just past the activation's place for a parameter's leaf, past the places of node `i`'s
members for a leaf of node `i`. -/
def phi (P : Prog) (s : Scope × Ref) : ℕ :=
  if s.2.p then 2 * pos P s.1 + 1 else 2 * (pos P s.1 + 1 + 2 * seg P s.1.d (s.2.i + 1))

theorem pos_child (P : Prog) (parent : Scope) (d off k m : ℕ) :
    pos P (.child parent d off k m) =
      pos P parent + 1 + 2 * seg P parent.d k + m * (2 * (Desc.activations P)[d]!) := rfl

theorem phi_p (P : Prog) (sc : Scope) (i j : ℕ) : phi P (sc, ⟨true, i, j⟩) = 2 * pos P sc + 1 := rfl

theorem phi_n (P : Prog) (sc : Scope) (i j : ℕ) :
    phi P (sc, ⟨false, i, j⟩) = 2 * (pos P sc + 1 + 2 * seg P sc.d (i + 1)) := rfl

theorem seg_succ (P : Prog) (d k : ℕ) (hk : k < (P.defs[d]!).body.nodes.size) :
    seg P d (k + 1) = seg P d k +
      ((P.defs[d]!).body.nodes[k]!).members * (Desc.activations P)[((P.defs[d]!).body.nodes[k]!).fn]! := by
  unfold seg
  rw [List.take_add_one, List.map_append, List.sum_append, getElem!_pos (P.defs[d]!).body.nodes k hk]
  simp [hk]

theorem seg_mono (P : Prog) (d : ℕ) {a b : ℕ} (h : a ≤ b) : seg P d a ≤ seg P d b := by
  unfold seg
  obtain ⟨c, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [List.take_add, List.map_append, List.sum_append]
  exact Nat.le_add_right _ _

theorem seg_le (P : Prog) (d k : ℕ) : seg P d k ≤ seg P d (P.defs[d]!).body.nodes.size := by
  rcases le_total k (P.defs[d]!).body.nodes.size with h | h
  · exact seg_mono P d h
  · unfold seg
    rw [List.take_of_length_le (by simpa using h), List.take_of_length_le (by simp)]

theorem foldl_add_sum {α : Type} (f : α → ℕ) :
    ∀ (l : List α) (s : ℕ), l.foldl (fun s n => s + f n) s = s + (l.map f).sum
  | [], s => by simp
  | x :: xs, s => by
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [foldl_add_sum f xs]
    omega

/-- **A composite's activations**: its own, and those its nodes hold. -/
theorem activations_seg (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) : (Desc.activations P)[d]! = 1 + seg P d (P.defs[d]!).body.nodes.size := by
  rw [Walk.activations_get P hc hd, hp]
  simp only [Bool.false_eq_true, ite_false]
  have h := foldl_add_sum (fun n : Node => n.members * (Desc.activations P)[n.fn]!)
    (P.defs[d]!).body.nodes.toList 0
  rw [Array.foldl_toList] at h
  rw [h]
  unfold seg
  rw [List.take_of_length_le (by simp)]
  omega

/-- What `Desc.resolvable` checks: the root has no parameters, and a scan of no iterations no output leaves. -/
theorem resolvable_spec (P : Prog) (h : Desc.resolvable P = true) :
    (P.defs[P.root]!).body.params.size = 0 ∧ ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).form = .scan →
        ((P.defs[d]!).body.nodes[k]!).n = 0 → ((P.defs[d]!).body.nodes[k]!).out.leaves = 0 := by
  unfold Desc.resolvable at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  obtain ⟨h1, h2⟩ := h
  refine ⟨by simpa [Array.isEmpty_iff] using h1, fun d hd hp k hk hf hn => ?_⟩
  have h3 := h2 d hd
  rw [hp, Bool.false_or] at h3
  have hmem : (P.defs[d]!).body.nodes[k]! ∈ (P.defs[d]!).body.nodes := by
    rw [getElem!_pos _ k hk]
    exact Array.getElem_mem hk
  have h4 := Array.all_eq_true_iff_forall_mem.1 h3 _ hmem
  rw [hf, hn] at h4
  have e : (Form.scan != Form.scan) = false := rfl
  simpa [e] using h4

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
  (hc : Desc.calleesFirst P = true)
include hroot hB hc

/-- An activation the walk reaches holds its activations among the root's. -/
theorem pos_bound : ∀ sc, Resolve.Good P sc →
    pos P sc + 2 * (Desc.activations P)[sc.d]! ≤ 2 * (Desc.activations P)[P.root]!
  | .top d off st, h => by
    obtain ⟨rfl, -, -⟩ := h
    simp [pos, Scope.d]
  | .child parent d off k m, h => by
    have h' := h
    obtain ⟨hp, hk, hfn, -, hm, -⟩ := h'
    have ih := pos_bound parent hp
    obtain ⟨hpd, hpc⟩ := Resolve.good_d P hroot hB parent hp
    have hs := seg_succ P parent.d k hk
    have hle := seg_le P parent.d (k + 1)
    have ha := activations_seg P hc hpd hpc
    rw [hfn] at hs
    have hmA : (m + 1) * (Desc.activations P)[d]! ≤
        ((P.defs[parent.d]!).body.nodes[k]!).members * (Desc.activations P)[d]! := Nat.mul_le_mul_right _ hm
    have e1 : m * (2 * (Desc.activations P)[d]!) = 2 * (m * (Desc.activations P)[d]!) := by ring
    have e2 : (m + 1) * (Desc.activations P)[d]! = m * (Desc.activations P)[d]! + (Desc.activations P)[d]! := by ring
    rw [pos_child, show (Scope.child parent d off k m).d = d from rfl, e1]
    omega

omit hroot hB in
/-- A state's potential is below its activation's place plus two for each activation it holds. -/
theorem phi_le (sc : Scope) (r : Ref) (hd : sc.d < P.defs.size) (hp : (P.defs[sc.d]!).prim = false) :
    phi P (sc, r) + 1 ≤ 2 * pos P sc + 4 * (Desc.activations P)[sc.d]! := by
  have ha := activations_seg P hc hd hp
  have hle := seg_le P sc.d (r.i + 1)
  unfold phi
  dsimp only
  split <;> omega

/-- **Resolution's potential stays below four times the root's activations.** -/
theorem phi_lt (B : ℕ) (s : Scope × Ref) (hs : Resolve.Inv P B s) :
    phi P s < 4 * (Desc.activations P)[P.root]! := by
  obtain ⟨sc, r⟩ := s
  obtain ⟨hd, hp⟩ := Resolve.good_d P hroot hB sc hs.1
  have h1 := phi_le P hc sc r hd hp
  have h2 := pos_bound P hroot hB hc sc hs.1
  omega

omit hroot in
/-- Down into member `m` of node `i`: below the node's leaves' potential. -/
theorem phi_down (sc : Scope) (hg : Resolve.Good P sc) (i m off : ℕ) (r : Ref)
    (hi : i < (P.defs[sc.d]!).body.nodes.size)
    (hcomp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = false)
    (hm : m < ((P.defs[sc.d]!).body.nodes[i]!).members) :
    phi P (.child sc ((P.defs[sc.d]!).body.nodes[i]!).fn off i m, r) <
      2 * (pos P sc + 1 + 2 * seg P sc.d (i + 1)) := by
  have hfn := ((hB sc hg).1 i hi).1
  have h1 : phi P (.child sc ((P.defs[sc.d]!).body.nodes[i]!).fn off i m, r) + 1 ≤
      2 * (pos P sc + 1 + 2 * seg P sc.d i + m * (2 * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]!)) +
        4 * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! :=
    phi_le P hc (.child sc ((P.defs[sc.d]!).body.nodes[i]!).fn off i m) r hfn hcomp
  have hs := seg_succ P sc.d i hi
  have hmA : (m + 1) * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! ≤
      ((P.defs[sc.d]!).body.nodes[i]!).members * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! :=
    Nat.mul_le_mul_right _ hm
  have e1 : m * (2 * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]!) =
      2 * (m * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]!) := by ring
  have e2 : (m + 1) * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! =
      m * (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! +
        (Desc.activations P)[((P.defs[sc.d]!).body.nodes[i]!).fn]! := by ring
  omega

omit hroot hB in
/-- Back to the iteration before, through a scan's carry: below the parameter's potential. -/
theorem phi_prev (parent : Scope) (d off off' k m j : ℕ) (r : Ref) (hd : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) (hm : m ≠ 0) :
    phi P (.child parent d off' k (m - 1), r) < phi P (.child parent d off k m, ⟨true, 0, j⟩) := by
  have h1 : phi P (.child parent d off' k (m - 1), r) + 1 ≤
      2 * (pos P parent + 1 + 2 * seg P parent.d k + (m - 1) * (2 * (Desc.activations P)[d]!)) +
        4 * (Desc.activations P)[d]! :=
    phi_le P hc (.child parent d off' k (m - 1)) r hd hp
  rw [phi_p, pos_child]
  obtain ⟨m', rfl⟩ := Nat.exists_eq_succ_of_ne_zero hm
  have e : (m' + 1) * (2 * (Desc.activations P)[d]!) =
      m' * (2 * (Desc.activations P)[d]!) + 2 * (Desc.activations P)[d]! := by ring
  simp only [Nat.succ_sub_one, Nat.succ_eq_add_one] at h1 ⊢
  rw [e]
  omega

end

/-- Up to the parent, at a parameter or a node before the child's: below the parameter's potential. -/
theorem phi_up (P : Prog) (parent : Scope) (d off k m i j : ℕ) (r : Ref) (hr : r.p = false → r.i < k) :
    phi P (parent, r) < phi P (.child parent d off k m, ⟨true, i, j⟩) := by
  rw [phi_p, pos_child]
  unfold phi
  dsimp only
  split
  · omega
  · rename_i h
    have := seg_mono P parent.d (show r.i + 1 ≤ k from hr (by simpa using h))
    omega

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
  (hc : Desc.calleesFirst P = true) (hrp : (P.defs[P.root]!).body.params.size = 0)
  (hz : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).form = .scan →
        ((P.defs[d]!).body.nodes[k]!).n = 0 → ((P.defs[d]!).body.nodes[k]!).out.leaves = 0)
include hroot hB hc hrp hz

/-- **Each step of resolution ends at a gate or lowers the potential**, from a state `Resolve.Inv` holds of. -/
theorem step_progress (B : ℕ) (s : Scope × Ref) (hs : Resolve.Inv P B s) :
    (∃ a, resolveStep P s = .ok (.inl a)) ∨ ∃ s', resolveStep P s = .ok (.inr s') ∧ phi P s' < phi P s := by
  obtain ⟨sc, ⟨p, i, j⟩⟩ := s
  obtain ⟨hg, hr, -, -⟩ := hs
  dsimp only at hg hr
  cases p
  · have hi : i < (P.defs[sc.d]!).body.nodes.size := (hr.2 rfl).1
    have hj := (hr.2 rfl).2
    dsimp only at hi hj
    obtain ⟨hd, hpc⟩ := Resolve.good_d P hroot hB sc hg
    obtain ⟨-, -, -, hdown, -⟩ := (hB sc hg).1 i hi
    have hdn := hdown j hj
    rw [phi_n]
    rcases hf : ((P.defs[sc.d]!).body.nodes[i]!).form
    · exact Or.inl ⟨_, Resolve.rs_prim P sc i j hf⟩
    · simp only [hf] at hdn
      exact Or.inr ⟨_, Resolve.rs_call P sc i j hf, phi_down P hB hc sc hg i 0 _ _ hi hdn.1
        (by rw [Resolve.prim_members _ (Or.inr hf)]; omega)⟩
    · simp only [hf] at hdn
      obtain ⟨hm, -, -⟩ := hdn
      rw [Resolve.rs_batch P sc i j hf]
      by_cases hcp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
      · simp only [hcp, ite_true]
        exact Or.inl ⟨_, rfl⟩
      · simp only [hcp, Bool.false_eq_true, ite_false]
        exact Or.inr ⟨_, rfl, phi_down P hB hc sc hg i _ _ _ hi (by simpa using hcp)
          (by rw [Resolve.bs_members _ (Or.inl hf)]; exact hm)⟩
    · simp only [hf] at hdn
      obtain ⟨hcp, hdn⟩ := hdn
      by_cases hn : ((P.defs[sc.d]!).body.nodes[i]!).n = 0
      · have := hz sc.d hd hpc i hi hf hn
        omega
      obtain ⟨carry, y, hret, hm, -⟩ := hdn hn
      rw [Resolve.rs_scan P sc i j carry y hf hn hret]
      have e1 : (if j < carry.leaves then (((P.defs[sc.d]!).body.nodes[i]!).n - 1, j)
          else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)).1 =
          if j < carry.leaves then ((P.defs[sc.d]!).body.nodes[i]!).n - 1 else (j - carry.leaves) / y.leaves := by
        split <;> rfl
      exact Or.inr ⟨_, rfl, phi_down P hB hc sc hg i _ _ _ hi hcp
        (by rw [Resolve.bs_members _ (Or.inr hf), e1]; exact hm)⟩
  · rcases sc with ⟨d, off, st⟩ | ⟨parent, d, off, k, m⟩
    · unfold Resolve.Good at hg
      obtain ⟨rfl, -, -⟩ := hg
      have := (hr.1 rfl).1
      simp only [Scope.d] at this
      omega
    · simp only [Scope.d] at hr
      have hg' := hg
      unfold Resolve.Good at hg'
      obtain ⟨hp, hk, hfn, hcd, hm, -⟩ := hg'
      have hi : i < (P.defs[d]!).body.params.size := (hr.1 rfl).1
      have hj : j < ((P.defs[d]!).body.params[i]!).2.leaves := (hr.1 rfl).2
      obtain ⟨hfn', hargs, hup, -, -⟩ := (hB parent hp).1 k hk
      rw [hfn] at hup hfn'
      have hu := hup i hi j hj m hm
      have harg : ∀ (i' t : ℕ), t < ((((P.defs[parent.d]!).body.nodes[k]!).args[i']!).seq.count) →
          phi P (parent, (((P.defs[parent.d]!).body.nodes[k]!).args[i']!).seq.get t) <
            phi P (.child parent d off k m, ⟨true, i, j⟩) :=
        fun i' t ht => phi_up P parent d off k m i j _ (fun h => ((hargs i' t ht).2 h).1)
      rcases hf : ((P.defs[parent.d]!).body.nodes[k]!).form
      · simp only [hf] at hu
        rw [Resolve.rs_up P parent d off k m i j (Or.inl hf)]
        exact Or.inr ⟨_, rfl, harg i j hu⟩
      · simp only [hf] at hu
        rw [Resolve.rs_up P parent d off k m i j (Or.inr hf)]
        exact Or.inr ⟨_, rfl, harg i j hu⟩
      · simp only [hf] at hu
        rw [Resolve.rs_up_batch P parent d off k m i j hf, Resolve.memberRef_eq]
        exact Or.inr ⟨_, rfl, harg i _ hu⟩
      · simp only [hf] at hu
        by_cases hi0 : i = 0
        · subst hi0
          simp only [ite_true] at hu
          rw [Resolve.rs_up_scan0 P parent d off k m j hf]
          by_cases hm0 : m = 0
          · subst hm0
            simp only [ite_true] at hu ⊢
            exact Or.inr ⟨_, rfl, harg 0 j hu⟩
          · simp only [hm0, ite_false]
            exact Or.inr ⟨_, rfl, phi_prev P hc parent d off _ k m j _ hfn' hcd hm0⟩
        · rw [Resolve.rs_up_scan P parent d off k m i j hi0 hf]
          simp only [hi0, ite_false] at hu
          refine Or.inr ⟨_, rfl, ?_⟩
          by_cases hx : i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs
          · simp only [hx, ite_true] at hu ⊢
            exact harg i _ hu
          · simp only [hx, ite_false] at hu ⊢
            exact harg i _ hu

include hL

/-- **Resolution ends at a gate** from any state `Resolve.Inv` holds of, within `Flock.Steps.FUEL` when four times the
root's activations are. -/
theorem resolve_ok (hfuel : 4 * (Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL) (B : ℕ) (sc : Scope) (r : Ref)
    (hs : Resolve.Inv P B (sc, r)) : ∃ a, resolve P sc r = .ok a := by
  obtain ⟨a, ha, -⟩ := Walk.run_of_potential (what := "scope resolution") (step := resolveStep P) (Resolve.Inv P B)
    (phi P) (fun _ => True)
    (fun s hs' => by
      rcases step_progress P hroot hB hc hrp hz B s hs' with ⟨a, ha⟩ | ⟨s', hs1, hlt⟩
      · exact Or.inl ⟨a, ha, trivial⟩
      · exact Or.inr ⟨s', hs1, (Resolve.step_ok P hroot hB hL B s _ hs' hs1).2 s' rfl, hlt⟩)
    Flock.Steps.FUEL (sc, r) hs (lt_of_lt_of_le (phi_lt P hroot hB hc B _ hs) hfuel)
  exact ⟨a, ha⟩

end

theorem forIn_list_ok {α β ε : Type} :
    ∀ (l : List α) (f : α → β → Except ε (ForInStep β)) (b : β),
      (∀ x ∈ l, ∀ b, ∃ r, f x b = .ok r) → ∃ r, forIn l b f = .ok r
  | [], f, b, _ => ⟨b, rfl⟩
  | x :: l, f, b, hf => by
    rw [List.forIn_cons]
    obtain ⟨r, hr⟩ := hf x List.mem_cons_self b
    rw [hr]
    cases r with
    | done b' => exact ⟨b', rfl⟩
    | yield b' => exact forIn_list_ok l f b' (fun y hy => hf y (List.mem_cons_of_mem _ hy))

theorem forIn_list_ne {α β ε : Type} (l : List α) (f : α → β → Except ε (ForInStep β)) (b : β)
    (hf : ∀ x ∈ l, ∀ b, ∃ r, f x b = .ok r) (e : ε) : forIn l b f ≠ .error e := by
  obtain ⟨r, hr⟩ := forIn_list_ok l f b hf
  rw [hr]
  exact fun h => by cases h

/-- **`operands` returns** when every reference it resolves does. -/
theorem operands_ok (P : Prog) (sc : Scope) (node : Node) (m : ℕ)
    (h : ∀ i < node.args.size, ∀ t < (if (node.form == Form.batch && (node.args[i]!).dims.isSome) = true then
        ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
      ∃ a, resolve P sc (if (node.form == Form.batch) = true then (node.args[i]!).memberRef m t
        else (node.args[i]!).seq.get t) = .ok a) :
    ∃ ops, operands P sc node m = .ok ops := by
  unfold operands
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', bind, Except.bind]
  split
  · rename_i err herr
    refine absurd herr (forIn_list_ne _ _ _ ?_ err)
    intro i hi b
    simp only [List.mem_range'_1] at hi
    split
    · rename_i err' herr'
      refine absurd herr' (forIn_list_ne _ _ _ ?_ err')
      intro t ht b'
      simp only [List.mem_range'_1] at ht
      obtain ⟨a, ha⟩ := h i (by simpa using hi.2) t (by simpa using ht.2)
      rw [ha]
      exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

/-- **The activations `Prog.gate`'s walk reaches** are ones resolution starts from, holding `g`. -/
theorem reach_walkQ {g : ℕ} (hg : g < (P.defs[P.root]!).gates) {sc : Scope} (h : Walk.Reach P g sc) :
    Resolve.WalkQ P g sc := by
  induction h with
  | root =>
    exact ⟨by show Resolve.Good P (.top P.root 0 none); unfold Resolve.Good; exact ⟨rfl, rfl, rfl⟩, Nat.zero_le _,
      by show g - 0 < (P.defs[P.root]!).gates; omega⟩
  | @step sc sc' _ hb ih =>
    obtain ⟨hg, hoff, hlt⟩ := ih
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

/-- **A primitive node's operands resolve** in an activation resolution starts from: `Resolve.BodyOk`'s `OpOk` reads
each argument's leaves in range, and each resolves (`resolve_ok`). -/
theorem operands_prim (hc : Desc.calleesFirst P = true) (hrp : (P.defs[P.root]!).body.params.size = 0)
    (hz : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).form = .scan →
        ((P.defs[d]!).body.nodes[k]!).n = 0 → ((P.defs[d]!).body.nodes[k]!).out.leaves = 0)
    (hfuel : 4 * (Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL) (sc : Scope) (hg : Resolve.Good P sc) (k : ℕ)
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hpk : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (m : ℕ) (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) :
    ∃ ops, operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops := by
  obtain ⟨-, hargs, -, -, hop⟩ := (hB sc hg).1 k hk
  refine operands_ok P sc _ m fun i hi t ht => ?_
  have hidx := hop hpk i hi m hm t ht
  have hr : (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
      (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).memberRef m t
      else (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get t) =
      (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get
        (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
          Resolve.memberIdx (((P.defs[sc.d]!).body.nodes[k]!).args[i]!) m t else t) := by
    split <;> simp only [Resolve.memberRef_eq]
  rw [hr]
  exact resolve_ok P hroot hB hL hc hrp hz hfuel _ sc _
    (Resolve.start_ok P hroot hB hL sc hg k hk _ (hargs i _ hidx))

end

/-- **`Walk.OperandsResolve`, from `Resolve.BodiesOk`**: `Desc.walkable` lays every body out (`Walk.bodiesLaidOut_spec`),
gives the root no parameters and no scan of no iterations output leaves (`resolvable_spec`), and keeps four times the
root's activations within `Flock.Steps.FUEL` (`Desc.walkBound`), so in every activation the walk reaches
(`reach_walkQ`) a primitive node's operands resolve (`operands_prim`). -/
theorem operandsResolve_of (hB : Resolve.BodiesOk) : Walk.OperandsResolve := by
  intro bytes P hdec hov hlo hin hwk han g hg sc hR _ k hk hpk m hm e
  have hB' := hB bytes P hdec hov hlo hin hwk han
  unfold Desc.walkable at hwk
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hwk
  obtain ⟨⟨hc, hbl⟩, hres, hwb⟩ := hwk
  obtain ⟨hrp, hlay, -, -⟩ := Walk.bodiesLaidOut_spec P hbl
  have hg' : g < (P.defs[P.root]!).gates := hg
  have hr : P.root < P.defs.size := by
    refine (Nat.lt_or_ge P.root P.defs.size).elim id fun hn => ?_
    rw [getElem!_neg P.defs P.root (by omega)] at hg'
    have h0 : (default : Flock.Program.Defn).gates = 0 := rfl
    omega
  obtain ⟨hpar, hz⟩ := resolvable_spec P hres
  have hfuel : 4 * (Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL := by
    unfold Desc.walkBound at hwb
    have : ((Desc.activations P)[P.root]! + 1) * 2 ≤
        ((Desc.activations P)[P.root]! + 1) * ((Desc.depths P)[P.root]! + 2) := Nat.mul_le_mul_left _ (by omega)
    omega
  obtain ⟨ops, hops⟩ := operands_prim P ⟨hr, hrp⟩ hB' hlay hc hpar hz hfuel sc
    (reach_walkQ P ⟨hr, hrp⟩ hB' hlay hg' hR).1 k hk hpk m hm
  rw [hops]
  exact fun h => by cases h

end NciCheck.Properties.SecurityProofs.OperandsResolve
