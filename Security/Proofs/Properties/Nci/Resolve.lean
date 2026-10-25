import Proofs.Properties.Nci.Walk

/-!
# `Flock.Program.resolve`'s walk, toward `Walk.OperandsEarlier`

`resolveStep`'s cases, each as the step it takes (`rs_prim`, `rs_call`, `rs_batch`, `rs_scan`, `rs_scan0`, `rs_top`,
`rs_up_scan0`, `rs_up_scan`, `rs_up_batch`, `rs_up`; Lean generates no equations for its match, which shares
`buildNode`'s on `.tuple #[carry, y]`, so each reduces by `whnf`). Resolution keeps a bound (`Inv`): from an activation
the walk reaches (`Good`), a parameter's reference resolves before the activation, and a node's before the node ends.
Up through a parameter, the parent's node names its parameters and earlier nodes (`up_ok`), or a scan's carry the
iteration before; down through a node's output, the member's return names its nodes (`down_ok`); so every step keeps
it (`step_ok`) and resolution ends before its bound (`resolve_lt`). Every operand `operands` returns is such a
resolution (`operands_spec`), started below the primitive node it reads for (`start_ok`), so it is an earlier gate
(`gate_operands_lt`).

What that needs of each body the walk reaches is `BodyOk` (`BodiesOk`, a named `Prop` here, which `bodiesOk` proves in
`Proofs.Properties.Nci.BodiesOk`): its references name parameters and earlier nodes at a leaf, and the indices
resolution reads through each node are in range, which `Flock.Program.decode` and `Prog.orderViolation` check.
`operandsEarlier_of` gives `Walk.OperandsEarlier` from it.
-/

namespace NciCheck.Properties.SecurityProofs.Resolve

open Flock.Program


theorem rs_prim (P : Prog) (sc : Scope) (k j : ℕ)
    (h : ((P.defs[sc.d]!).body.nodes[k]!).form = .prim) :
    resolveStep P (sc, ⟨false, k, j⟩) = .ok (.inl (sc.off + (P.defs[sc.d]!).body.offsets[k]!)) := by
  rcases sc with ⟨d, off, _ | st⟩ | ⟨parent, d, off, k', m⟩ <;>
    (delta resolveStep; (conv => lhs; whnf); rw [h]; rfl)

theorem rs_call (P : Prog) (sc : Scope) (k j : ℕ)
    (h : ((P.defs[sc.d]!).body.nodes[k]!).form = .call) :
    resolveStep P (sc, ⟨false, k, j⟩) = .ok (.inr (.child sc ((P.defs[sc.d]!).body.nodes[k]!).fn
      (sc.off + (P.defs[sc.d]!).body.offsets[k]!) k 0,
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).body.ret.get j)) := by
  rcases sc with ⟨d, off, _ | st⟩ | ⟨parent, d, off, k', m⟩ <;>
    (delta resolveStep; (conv => lhs; whnf); rw [h]; rfl)

theorem rs_batch (P : Prog) (sc : Scope) (k j : ℕ)
    (h : ((P.defs[sc.d]!).body.nodes[k]!).form = .batch) :
    resolveStep P (sc, ⟨false, k, j⟩) =
      let c := P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!
      let o := sc.off + (P.defs[sc.d]!).body.offsets[k]!
      if c.prim then .ok (.inl (o + j / c.outLeaves))
      else .ok (.inr (.child sc ((P.defs[sc.d]!).body.nodes[k]!).fn (o + j / c.outLeaves * c.gates) k
        (j / c.outLeaves), c.body.ret.get (j % c.outLeaves))) := by
  rcases sc with ⟨d, off, _ | st⟩ | ⟨parent, d, off, k', m⟩ <;>
    (delta resolveStep; (conv => lhs; whnf); rw [h]; rfl)

theorem rs_scan (P : Prog) (sc : Scope) (k j : ℕ) (carry y : Ty)
    (h : ((P.defs[sc.d]!).body.nodes[k]!).form = .scan)
    (hn : ((P.defs[sc.d]!).body.nodes[k]!).n ≠ 0)
    (hr : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).ret = .tuple #[carry, y]) :
    resolveStep P (sc, ⟨false, k, j⟩) =
      let c := P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!
      let o := sc.off + (P.defs[sc.d]!).body.offsets[k]!
      let mj := if j < carry.leaves then (((P.defs[sc.d]!).body.nodes[k]!).n - 1, j)
        else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)
      .ok (.inr (.child sc ((P.defs[sc.d]!).body.nodes[k]!).fn (o + mj.1 * c.gates) k mj.1, c.body.ret.get mj.2)) := by
  have hn' : (((P.defs[sc.d]!).body.nodes[k]!).n == 0) = false := by simpa using hn
  rcases sc with ⟨d, off, _ | st⟩ | ⟨parent, d, off, k', m⟩ <;>
  · delta resolveStep; conv => lhs; whnf
    rw [h]
    conv => lhs; whnf
    rw [hn']
    conv => lhs; whnf
    rw [hr]
    rfl


theorem rs_scan0 (P : Prog) (sc : Scope) (k j : ℕ) (x)
    (h : ((P.defs[sc.d]!).body.nodes[k]!).form = .scan)
    (hn : ((P.defs[sc.d]!).body.nodes[k]!).n = 0) : resolveStep P (sc, ⟨false, k, j⟩) ≠ .ok x := by
  have hn' : (((P.defs[sc.d]!).body.nodes[k]!).n == 0) = true := by simpa using hn
  rcases sc with ⟨d, off, _ | st⟩ | ⟨parent, d, off, k', m⟩ <;>
  · delta resolveStep; (conv => lhs; whnf); rw [h]; (conv => lhs; whnf); rw [hn']
    intro h; cases h

theorem rs_top (P : Prog) (d off i j : ℕ) (x) : resolveStep P (.top d off none, ⟨true, i, j⟩) ≠ .ok x := by
  delta resolveStep; intro h; cases h

theorem rs_up_scan0 (P : Prog) (parent : Scope) (d off k m j : ℕ)
    (h : ((P.defs[parent.d]!).body.nodes[k]!).form = .scan) :
    resolveStep P (.child parent d off k m, ⟨true, 0, j⟩) =
      if m = 0 then .ok (.inr (parent, (((P.defs[parent.d]!).body.nodes[k]!).args[0]!).seq.get j))
      else .ok (.inr (.child parent d (off - (P.defs[d]!).gates) k (m - 1), (P.defs[d]!).body.ret.get j)) := by
  delta resolveStep; (conv => lhs; whnf); rw [h]
  by_cases hm : m = 0
  · subst hm; rfl
  · have : (m == 0) = false := by simpa using hm
    (conv => rhs; simp only [hm, ↓reduceIte]); (conv => lhs; whnf); rw [this]; rfl

theorem rs_up_scan (P : Prog) (parent : Scope) (d off k m i j : ℕ) (hi : i ≠ 0)
    (h : ((P.defs[parent.d]!).body.nodes[k]!).form = .scan) :
    resolveStep P (.child parent d off k m, ⟨true, i, j⟩) =
      let a := ((P.defs[parent.d]!).body.nodes[k]!).args[i]!
      .ok (.inr (parent, if i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs
        then a.seq.get (m * (a.dims.map (·.2)).getD 0 + j) else a.seq.get j)) := by
  have : (i == 0) = false := by simpa using hi
  delta resolveStep; (conv => lhs; whnf); rw [h]; (conv => lhs; whnf); rw [this]; rfl

theorem rs_up_batch (P : Prog) (parent : Scope) (d off k m i j : ℕ)
    (h : ((P.defs[parent.d]!).body.nodes[k]!).form = .batch) :
    resolveStep P (.child parent d off k m, ⟨true, i, j⟩) =
      .ok (.inr (parent, (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).memberRef m j)) := by
  delta resolveStep; (conv => lhs; whnf); rw [h]; rfl

theorem rs_up (P : Prog) (parent : Scope) (d off k m i j : ℕ)
    (h : ((P.defs[parent.d]!).body.nodes[k]!).form = .prim ∨ ((P.defs[parent.d]!).body.nodes[k]!).form = .call) :
    resolveStep P (.child parent d off k m, ⟨true, i, j⟩) =
      .ok (.inr (parent, (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j)) := by
  rcases h with h | h <;> (delta resolveStep; (conv => lhs; whnf); rw [h]; rfl)


/-- The activations `Prog.gate`'s walk and `resolve` reach: the root's, and a composite member of a node of one. -/
def Good (P : Prog) : Scope → Prop
  | .top d off st => d = P.root ∧ off = 0 ∧ st = none
  | .child parent d off k m => Good P parent ∧ k < (P.defs[parent.d]!).body.nodes.size ∧
      ((P.defs[parent.d]!).body.nodes[k]!).fn = d ∧ (P.defs[d]!).prim = false ∧
      m < ((P.defs[parent.d]!).body.nodes[k]!).members ∧
      off = parent.off + (P.defs[parent.d]!).body.offsets[k]! + m * (P.defs[d]!).gates

/-- Reference `r` of body `d` names a parameter's leaf or a leaf of a node before `k`. -/
def RefOk (P : Prog) (d k : ℕ) (r : Ref) : Prop :=
  (r.p = true → r.i < (P.defs[d]!).body.params.size ∧ r.j < ((P.defs[d]!).body.params[r.i]!).2.leaves) ∧
  (r.p = false → r.i < k ∧ r.j < ((P.defs[d]!).body.nodes[r.i]!).out.leaves)

def memberIdx (a : Arg) (m t : ℕ) : ℕ := match a.dims with
  | none => t
  | some (d, L) => (t / L * d + m) * L + t % L

theorem memberRef_eq (a : Arg) (m t : ℕ) : a.memberRef m t = a.seq.get (memberIdx a m t) := by
  unfold Arg.memberRef memberIdx; cases a.dims <;> rfl

/-- The indices a parameter's leaf of member `m` of node `nd` (calling `c`) reads, in range. -/
def UpOk (nd : Node) (c : Defn) : Prop :=
  ∀ i < c.body.params.size, ∀ j < (c.body.params[i]!).2.leaves, ∀ m < nd.members,
    match nd.form with
    | .scan => if i = 0 then (if m = 0 then j < (nd.args[0]!).seq.count else j < c.body.ret.count)
      else (if i ≤ nd.nxs then m * ((nd.args[i]!).dims.map (·.2)).getD 0 + j else j) < (nd.args[i]!).seq.count
    | .batch => memberIdx (nd.args[i]!) m j < (nd.args[i]!).seq.count
    | _ => j < (nd.args[i]!).seq.count

/-- The member and leaf a node's output leaf `j` reads, in range, and the callee of the form it names. -/
def DownOk (nd : Node) (c : Defn) : Prop :=
  ∀ j < nd.out.leaves,
    match nd.form with
    | .prim => c.prim = true ∧ c.gates = 1
    | .call => c.prim = false ∧ j < c.body.ret.count
    | .batch => j / c.outLeaves < nd.n ∧ (c.prim = true → c.gates = 1) ∧
        (c.prim = false → j % c.outLeaves < c.body.ret.count)
    | .scan => c.prim = false ∧ (nd.n ≠ 0 → ∃ carry y, c.ret = .tuple #[carry, y] ∧
        (if j < carry.leaves then nd.n - 1 else (j - carry.leaves) / y.leaves) < nd.n ∧
        (if j < carry.leaves then j else carry.leaves + (j - carry.leaves) % y.leaves) < c.body.ret.count)

/-- The indices a primitive node's operands read, in range. -/
def OpOk (nd : Node) (c : Defn) : Prop :=
  ∀ i < nd.args.size, ∀ m < nd.members,
    ∀ t < (if (nd.form == Form.batch && (nd.args[i]!).dims.isSome) = true then (c.params[i]!).2.leaves
      else (nd.args[i]!).seq.count),
    (if (nd.form == Form.batch) = true then memberIdx (nd.args[i]!) m t else t) < (nd.args[i]!).seq.count

/-- What a body the walk reaches holds: its nodes' arguments name parameters and earlier nodes, its return names its
nodes, each at a leaf, and the indices resolution reads through each node are in range. -/
def BodyOk (P : Prog) (d : ℕ) : Prop :=
  (∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn < P.defs.size ∧
    (∀ i : ℕ, ∀ t < (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.count,
      RefOk P d k ((((P.defs[d]!).body.nodes[k]!).args[i]!).seq.get t)) ∧
    UpOk ((P.defs[d]!).body.nodes[k]!) (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!) ∧
    DownOk ((P.defs[d]!).body.nodes[k]!) (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!) ∧
    ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim = true →
      OpOk ((P.defs[d]!).body.nodes[k]!) (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!))) ∧
  ∀ t < (P.defs[d]!).body.ret.count, RefOk P d (P.defs[d]!).body.nodes.size ((P.defs[d]!).body.ret.get t)

theorem offsets_mono (P : Prog) (d : ℕ) (hl : Walk.LaidOut P d) (hp : (P.defs[d]!).prim = false) :
    ∀ a b, a ≤ b → b ≤ (P.defs[d]!).body.nodes.size → (P.defs[d]!).body.offsets[a]! ≤ (P.defs[d]!).body.offsets[b]! := by
  intro a b hab hb
  induction b, hab using Nat.le_induction with
  | base => exact le_refl _
  | succ b _ ih =>
    have := (hl hp).2.2 b (by omega)
    have := ih (by omega)
    omega


/-- Resolution's state `(sc, r)` resolves below `B`: an activation the walk reaches, a reference of its body, and the
gates it may name before `B` (a parameter's lie before the activation, a node's before its end). -/
def Inv (P : Prog) (B : ℕ) (s : Scope × Ref) : Prop :=
  Good P s.1 ∧ RefOk P s.1.d (P.defs[s.1.d]!).body.nodes.size s.2 ∧
    (s.2.p = true → s.1.off ≤ B) ∧ (s.2.p = false → s.1.off + (P.defs[s.1.d]!).body.offsets[s.2.i + 1]! ≤ B)


theorem forIn_list_inv {α β ε : Type} (Q : β → Prop) :
    ∀ (l : List α) (f : α → β → Except ε (ForInStep β)) (b r : β),
      (∀ x ∈ l, ∀ b r, Q b → f x b = .ok r → ∃ b', r = .yield b' ∧ Q b') →
      Q b → forIn l b f = .ok r → Q r
  | [], f, b, r, _, hb, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact h ▸ hb
  | x :: l, f, b, r, hf, hb, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x b with e | s
    · simp [hx, bind, Except.bind] at h
    · obtain ⟨b', rfl, hb'⟩ := hf x (List.mem_cons_self) b _ hb hx
      simp only [hx, bind, Except.bind] at h
      exact forIn_list_inv Q l f b' r (fun y hy => hf y (List.mem_cons_of_mem _ hy)) hb' h

def OpsQ (P : Prog) (sc : Scope) (node : Node) (m : ℕ) (ops : Array ℕ) : Prop :=
  ∀ a ∈ ops, ∃ i < node.args.size, ∃ t < (if (node.form == Form.batch && (node.args[i]!).dims.isSome) = true then
      ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
    resolve P sc (if (node.form == Form.batch) = true then (node.args[i]!).memberRef m t
      else (node.args[i]!).seq.get t) = .ok a

/-- **What `operands` returns**: each operand is the resolution of a reference of an argument `i`, at an index `t` below
the count `operands` reads it to. -/
theorem operands_spec (P : Prog) (sc : Scope) (node : Node) (m : ℕ) (ops : Array ℕ)
    (h : operands P sc node m = .ok ops) : OpsQ P sc node m ops := by
  unfold operands at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', bind, Except.bind] at h
  split at h
  · cases h
  · rename_i ops' hops
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine forIn_list_inv (OpsQ P sc node m) _ _ _ _ ?_ (by simp [OpsQ]) hops
    intro i hi b r hb hr
    simp only [List.mem_range'_1] at hi
    split at hr
    · cases hr
    · rename_i inner hinner
      simp only [pure, Except.pure, Except.ok.injEq] at hr
      subst hr
      refine ⟨_, rfl, ?_⟩
      refine forIn_list_inv _ _ _ _ _ ?_ hb hinner
      intro t ht b' r' hb' hr'
      simp only [List.mem_range'_1] at ht
      split at hr'
      · cases hr'
      · rename_i v hv
        simp only [pure, Except.pure, Except.ok.injEq] at hr'
        subst hr'
        refine ⟨_, rfl, fun a ha => ?_⟩
        rcases Array.mem_push.1 ha with ha | rfl
        · exact hb' a ha
        · exact ⟨i, by simpa using hi.2, t, by simpa using ht.2, hv⟩

theorem prim_members (nd : Node) (h : nd.form = .prim ∨ nd.form = .call) : nd.members = 1 := by
  unfold Node.members; rcases h with h | h <;> rw [h] <;> rfl

theorem bs_members (nd : Node) (h : nd.form = .batch ∨ nd.form = .scan) : nd.members = nd.n := by
  unfold Node.members; rcases h with h | h <;> rw [h] <;> rfl

theorem good_child (P : Prog) {parent : Scope} {d off k m : ℕ} (hp : Good P parent)
    (hk : k < (P.defs[parent.d]!).body.nodes.size) (hfn : ((P.defs[parent.d]!).body.nodes[k]!).fn = d)
    (hc : (P.defs[d]!).prim = false) (hm : m < ((P.defs[parent.d]!).body.nodes[k]!).members)
    (ho : off = parent.off + (P.defs[parent.d]!).body.offsets[k]! + m * (P.defs[d]!).gates) :
    Good P (.child parent d off k m) := by
  unfold Good; exact ⟨hp, hk, hfn, hc, hm, ho⟩

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB

theorem good_d : ∀ sc, Good P sc → sc.d < P.defs.size ∧ (P.defs[sc.d]!).prim = false
  | .top d off st, h => by obtain ⟨rfl, -, -⟩ := h; exact hroot
  | .child parent d off k m, h => by
    obtain ⟨hp, hk, hfn, hc, -, -⟩ := h
    refine ⟨?_, hc⟩
    have := ((hB parent hp).1 k hk).1
    rw [hfn] at this
    exact this

include hL

/-- Up through a parameter: a reference of the parent's node `k` names a parameter or an earlier node. -/
theorem up_ok (B : ℕ) (parent : Scope) (d off k m : ℕ) (hg : Good P (.child parent d off k m)) (hoff : off ≤ B)
    (r : Ref) (hr : RefOk P parent.d k r) : Inv P B (parent, r) := by
  obtain ⟨hp, hk, hfn, hc, hm, ho⟩ := hg
  obtain ⟨hd, hpc⟩ := good_d P hroot hB parent hp
  have hmono := offsets_mono P parent.d (hL _ hd) hpc
  unfold Inv; dsimp only
  refine ⟨hp, ⟨fun h => hr.1 h, fun h => ⟨by show r.i < _; have := (hr.2 h).1; omega, (hr.2 h).2⟩⟩,
    fun _ => by show parent.off ≤ B; omega, fun h => ?_⟩
  have h1 := (hr.2 h).1
  have := hmono (r.i + 1) k (by omega) (by omega)
  show parent.off + _ ≤ B
  omega

/-- Down through a return: into an activation the walk reaches, which ends by `B`. -/
theorem down_ok (B : ℕ) (sc : Scope) (hg : Good P sc) (hend : sc.off + (P.defs[sc.d]!).gates ≤ B)
    (t : ℕ) (ht : t < (P.defs[sc.d]!).body.ret.count) : Inv P B (sc, (P.defs[sc.d]!).body.ret.get t) := by
  obtain ⟨hd, hpc⟩ := good_d P hroot hB sc hg
  have hr := (hB sc hg).2 t ht
  have hmono := offsets_mono P sc.d (hL _ hd) hpc
  have hK := (hL _ hd hpc).2.1
  unfold Inv; dsimp only
  refine ⟨hg, hr, fun _ => by omega, fun h => ?_⟩
  have h1 := (hr.2 h).1
  have := hmono (((P.defs[sc.d]!).body.ret.get t).i + 1) _ (by omega) (le_refl _)
  show sc.off + _ ≤ B
  omega

/-- **One step of resolution keeps `Inv`**: it ends at a gate before `B` or moves to a state that resolves below `B`. -/
theorem step_ok (B : ℕ) (s : Scope × Ref) (x) (hs : Inv P B s) (h : resolveStep P s = .ok x) :
    (∀ a, x = .inl a → a < B) ∧ (∀ s', x = .inr s' → Inv P B s') := by
  obtain ⟨sc, ⟨p, i, j⟩⟩ := s
  obtain ⟨hg, hr, hbp, hbn⟩ := hs
  dsimp only at hg hr hbp hbn
  cases p
  · have hi : i < (P.defs[sc.d]!).body.nodes.size := (hr.2 rfl).1
    have hj := (hr.2 rfl).2
    have hb := hbn rfl
    dsimp only at hi hj hb
    obtain ⟨hd, hpc⟩ := good_d P hroot hB sc hg
    obtain ⟨hfn, -, -, hdown, -⟩ := (hB sc hg).1 i hi
    have hdn := hdown j hj
    have hlay := (hL _ hd hpc).2.2 i hi
    rcases hf : ((P.defs[sc.d]!).body.nodes[i]!).form
    · simp only [hf] at hdn
      rw [rs_prim P sc i j hf] at h
      cases h
      refine ⟨fun a ha => ?_, fun s' hs' => by cases hs'⟩
      cases ha
      rw [prim_members _ (Or.inl hf), hdn.2] at hlay
      omega
    · simp only [hf] at hdn
      rw [rs_call P sc i j hf] at h
      cases h
      refine ⟨fun a ha => (by cases ha), fun s' hs' => ?_⟩
      cases hs'
      rw [prim_members _ (Or.inr hf)] at hlay
      refine down_ok P hroot hB hL B _ (good_child P hg hi rfl hdn.1 (by rw [prim_members _ (Or.inr hf)]; omega) (by omega))
        ?_ j hdn.2
      simp only [Scope.off.eq_2, Scope.d.eq_2]
      omega
    · simp only [hf] at hdn
      rw [rs_batch P sc i j hf] at h
      rw [bs_members _ (Or.inl hf)] at hlay
      obtain ⟨hm, hpg, hrc⟩ := hdn
      by_cases hcp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
      · simp only [hcp, ite_true] at h
        cases h
        refine ⟨fun a ha => ?_, fun s' hs' => by cases hs'⟩
        cases ha
        rw [hpg hcp] at hlay
        omega
      · simp only [hcp, Bool.false_eq_true, ite_false] at h
        cases h
        refine ⟨fun a ha => (by cases ha), fun s' hs' => ?_⟩
        cases hs'
        have hcp' : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = false := by simpa using hcp
        have hle : (j / (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).outLeaves + 1) *
            (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).gates ≤
            ((P.defs[sc.d]!).body.nodes[i]!).n * (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).gates :=
          Nat.mul_le_mul_right _ hm
        rw [Nat.add_mul, Nat.one_mul] at hle
        refine down_ok P hroot hB hL B _ (good_child P hg hi rfl hcp' (by rw [bs_members _ (Or.inl hf)]; exact hm) (by omega))
          ?_ _ (hrc hcp')
        simp only [Scope.off.eq_2, Scope.d.eq_2]
        omega
    · simp only [hf] at hdn
      obtain ⟨hcp, hdn⟩ := hdn
      by_cases hn : ((P.defs[sc.d]!).body.nodes[i]!).n = 0
      · exact absurd h (rs_scan0 P sc i j x hf hn)
      obtain ⟨carry, y, hret, hm, hjj⟩ := hdn hn
      rw [rs_scan P sc i j carry y hf hn hret] at h
      rw [bs_members _ (Or.inr hf)] at hlay
      have e1 : (if j < carry.leaves then (((P.defs[sc.d]!).body.nodes[i]!).n - 1, j)
          else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)).1 =
          if j < carry.leaves then ((P.defs[sc.d]!).body.nodes[i]!).n - 1 else (j - carry.leaves) / y.leaves := by
        split <;> rfl
      have e2 : (if j < carry.leaves then (((P.defs[sc.d]!).body.nodes[i]!).n - 1, j)
          else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)).2 =
          if j < carry.leaves then j else carry.leaves + (j - carry.leaves) % y.leaves := by
        split <;> rfl
      simp only [e1, e2] at h
      cases h
      refine ⟨fun a ha => (by cases ha), fun s' hs' => ?_⟩
      cases hs'
      have hle := Nat.mul_le_mul_right (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).gates hm
      rw [Nat.succ_mul] at hle
      refine down_ok P hroot hB hL B _ (good_child P hg hi rfl hcp (by rw [bs_members _ (Or.inr hf)]; exact hm) (by omega))
        ?_ _ hjj
      simp only [Scope.off.eq_2, Scope.d.eq_2]
      omega
  · rcases sc with ⟨d, off, st⟩ | ⟨parent, d, off, k, m⟩
    · unfold Good at hg
      obtain ⟨-, -, rfl⟩ := hg
      exact absurd h (rs_top P d off i j x)
    · simp only [Scope.d.eq_2, Scope.off.eq_2] at hr hbp
      have hg' := hg
      unfold Good at hg'
      obtain ⟨hp, hk, hfn, hc, hm, ho⟩ := hg'
      have hi : i < (P.defs[d]!).body.params.size := (hr.1 rfl).1
      have hj : j < ((P.defs[d]!).body.params[i]!).2.leaves := (hr.1 rfl).2
      have hoff : off ≤ B := by simpa using hbp
      obtain ⟨-, hargs, hup, -, -⟩ := (hB parent hp).1 k hk
      rw [hfn] at hup
      have hu := hup i hi j hj m hm
      have harg : ∀ (i' t : ℕ), t < ((((P.defs[parent.d]!).body.nodes[k]!).args[i']!).seq.count) →
          Inv P B (parent, (((P.defs[parent.d]!).body.nodes[k]!).args[i']!).seq.get t) :=
        fun i' t ht => up_ok P hroot hB hL B parent d off k m hg hoff _ (hargs i' t ht)
      rcases hf : ((P.defs[parent.d]!).body.nodes[k]!).form
      · simp only [hf] at hu
        rw [rs_up P parent d off k m i j (Or.inl hf)] at h
        cases h
        exact ⟨fun a ha => (by cases ha), fun s' hs' => by cases hs'; exact harg i j hu⟩
      · simp only [hf] at hu
        rw [rs_up P parent d off k m i j (Or.inr hf)] at h
        cases h
        exact ⟨fun a ha => (by cases ha), fun s' hs' => by cases hs'; exact harg i j hu⟩
      · simp only [hf] at hu
        rw [rs_up_batch P parent d off k m i j hf, memberRef_eq] at h
        cases h
        exact ⟨fun a ha => (by cases ha), fun s' hs' => by cases hs'; exact harg i _ hu⟩
      · simp only [hf] at hu
        by_cases hi0 : i = 0
        · subst hi0
          simp only [ite_true] at hu
          rw [rs_up_scan0 P parent d off k m j hf] at h
          by_cases hm0 : m = 0
          · subst hm0
            simp only [ite_true] at hu h
            cases h
            exact ⟨fun a ha => (by cases ha), fun s' hs' => by cases hs'; exact harg 0 j hu⟩
          · simp only [hm0, ite_false] at hu h
            cases h
            refine ⟨fun a ha => (by cases ha), fun s' hs' => ?_⟩
            cases hs'
            obtain ⟨m', rfl⟩ := Nat.exists_eq_succ_of_ne_zero hm0
            rw [Nat.succ_mul] at ho
            refine down_ok P hroot hB hL B _ (good_child P hp hk hfn hc (by omega)
              (by rw [Nat.succ_sub_one]; omega)) ?_ j hu
            simp only [Scope.off.eq_2, Scope.d.eq_2]
            omega
        · simp only [hi0, ite_false] at hu
          rw [rs_up_scan P parent d off k m i j hi0 hf] at h
          cases h
          refine ⟨fun a ha => (by cases ha), fun s' hs' => ?_⟩
          cases hs'
          by_cases hx : i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs
          · simp only [hx, ite_true] at hu ⊢
            exact harg i _ hu
          · simp only [hx, ite_false] at hu ⊢
            exact harg i _ hu

/-- **Resolution stays below its bound.** -/
theorem resolve_lt (B : ℕ) (sc : Scope) (r : Ref) (hs : Inv P B (sc, r)) {a : ℕ} (h : resolve P sc r = .ok a) :
    a < B :=
  DescAgree.run_inv "scope resolution" (resolveStep P) (Inv P B) (· < B)
    (fun s a hs h => (step_ok P hroot hB hL B s _ hs h).1 a rfl)
    (fun s s' hs h => (step_ok P hroot hB hL B s _ hs h).2 s' rfl) _ _ _ hs h

/-- A reference of node `k`'s arguments starts resolution below the node. -/
theorem start_ok (sc : Scope) (hg : Good P sc) (k : ℕ) (hk : k < (P.defs[sc.d]!).body.nodes.size) (r : Ref)
    (hr : RefOk P sc.d k r) : Inv P (sc.off + (P.defs[sc.d]!).body.offsets[k]!) (sc, r) := by
  obtain ⟨hd, hpc⟩ := good_d P hroot hB sc hg
  have hmono := offsets_mono P sc.d (hL _ hd) hpc
  unfold Inv; dsimp only
  refine ⟨hg, ⟨fun h => hr.1 h, fun h => ⟨by have := (hr.2 h).1; omega, (hr.2 h).2⟩⟩, fun _ => by omega, fun h => ?_⟩
  have h1 := (hr.2 h).1
  have := hmono (r.i + 1) k (by omega) (by omega)
  omega

/-- What `Prog.gate`'s walk keeps: an activation resolution reaches, holding `g`. -/
def WalkQ (P : Prog) (g : ℕ) (sc : Scope) : Prop :=
  Good P sc ∧ sc.off ≤ g ∧ g - sc.off < (P.defs[sc.d]!).gates

/-- **Every operand is an earlier gate**, on bodies `BodyOk` wherever the walk reaches. -/
theorem gate_operands_lt {g p : ℕ} {ops : Array ℕ} {sc : Scope} (h : P.gate g = .ok (p, ops, sc)) :
    ∀ a ∈ ops, a < g := by
  unfold Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (gateStep P g) (WalkQ P g) (fun b => ∀ a ∈ b.2.1, a < g) ?_ ?_ _ _ (p, ops, sc)
    ⟨by show Good P (.top P.root 0 none); unfold Good; exact ⟨rfl, rfl, rfl⟩, Nat.zero_le _,
      by show g - 0 < (P.defs[P.root]!).gates; omega⟩ h
  · intro sc b ⟨hg, hoff, hlt⟩ hb
    obtain ⟨hd, hpc⟩ := good_d P hroot hB sc hg
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
          intro a ha
          obtain ⟨i, hi, t, ht, hres⟩ := operands_spec P sc _ _ ops' hops a ha
          obtain ⟨-, hargs, -, -, hop⟩ := (hB sc hg).1 k hkK
          have hidx := hop hpc' i hi _ hm1 t ht
          have hr : (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
              (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).memberRef
                (if (((P.defs[sc.d]!).body.nodes[k]!).form == .prim ||
                  ((P.defs[sc.d]!).body.nodes[k]!).form == .call) = true then 0
                else (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) /
                  (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) t
              else (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get t) =
              (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get
                (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
                  memberIdx (((P.defs[sc.d]!).body.nodes[k]!).args[i]!)
                    (if (((P.defs[sc.d]!).body.nodes[k]!).form == .prim ||
                      ((P.defs[sc.d]!).body.nodes[k]!).form == .call) = true then 0
                    else (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) /
                      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) t
                else t) := by
            split <;> simp only [memberRef_eq]
          rw [hr] at hres
          have := resolve_lt P hroot hB hL _ sc _ (start_ok P hroot hB hL sc hg k hkK _ (hargs i _ hidx)) hres
          omega
      · simp only [hpc', Bool.false_eq_true, ite_false, pure, Except.pure] at hb
        cases hb
    · simp [throw, throwThe, MonadExceptOf.throw] at hb
  · intro sc sc' ⟨hg, hoff, hlt⟩ hb
    obtain ⟨hd, hpc⟩ := good_d P hroot hB sc hg
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
        refine ⟨good_child P hg hkK rfl (by simpa using hpc') hm1 rfl, ?_, ?_⟩
        · simp only [Prog.child, Scope.off.eq_2]
          omega
        · simp only [Prog.child, Scope.off.eq_2, Scope.d.eq_2]
          omega
    · simp [throw, throwThe, MonadExceptOf.throw] at hb

end

/-- **The bodies resolution reaches hold their references in range** (`bodiesOk`, `Proofs.Properties.Nci.BodiesOk`). On
a Program `Desc.evaluate` reads, the body of every activation `Good` reaches is `BodyOk`: its nodes' arguments name its
parameters and earlier nodes, and its return its nodes, each at a leaf, and the members and leaves resolution reads
through each node are in range. -/
def BodiesOk : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    Desc.rootLaidOut P = true → Desc.inputsAtRoot P = true → Desc.walkable P = true →
    (∃ reads, Desc.analyze P = .ok reads) → ∀ sc, Good P sc → BodyOk P sc.d

/-- **`OperandsEarlier`, from `BodiesOk`**: `Desc.walkable` lays every body out with the root a composite
(`Walk.bodiesLaidOut_spec`), so every operand the walk locates is an earlier gate (`gate_operands_lt`). -/
theorem operandsEarlier_of (hB : BodiesOk) : Walk.OperandsEarlier := by
  intro bytes P hdec hov hlo hin hwk han g hg p ops sc hx a ha
  have hwk' := hwk
  unfold Desc.walkable at hwk'
  simp only [Bool.and_eq_true] at hwk'
  obtain ⟨hrp, hlay, -, -⟩ := Walk.bodiesLaidOut_spec P hwk'.1.2
  have hr : P.root < P.defs.size := by
    refine (Nat.lt_or_ge P.root P.defs.size).elim id fun hn => ?_
    have hg' : g < (P.defs[P.root]!).gates := hg
    rw [getElem!_neg P.defs P.root (by omega)] at hg'
    have h0 : (default : Flock.Program.Defn).gates = 0 := rfl
    omega
  exact gate_operands_lt P ⟨hr, hrp⟩ (hB bytes P hdec hov hlo hin hwk han) hlay hx a ha

end NciCheck.Properties.SecurityProofs.Resolve
