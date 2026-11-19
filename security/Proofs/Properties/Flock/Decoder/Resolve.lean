module

public import Proofs.Properties.Flock.Decoder.Walk

/-!
# `Flock.Program.resolve`'s steps and the bound they keep

`resolveStep`'s cases, the activations the walk reaches (`Good`), what each body must hold (`BodyOk`) and the bound
resolution keeps (`Inv`), up through a parameter (`up_ok`) and down through a return (`down_ok`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.Resolve

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

theorem good_base (P : Prog) : ∀ {U : Scope}, Good P U → U.base = P.root
  | .top _ _ _, h => h.1
  | .child parent _ _ _ _, h => good_base P (U := parent) h.1

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
  rw [operands_of (operands_cond h)] at h
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

end

end InferenceOnlyVerifier.Properties.SecurityProofs.Resolve
