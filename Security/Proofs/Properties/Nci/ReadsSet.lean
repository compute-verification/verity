import Proofs.Properties.Nci.ReadsOneNode
import Proofs.Properties.Nci.OperandsResolve

/-!
# `ReadsExact.RootSet` from the decoder's facts: gates located with their operands, and the statements it is proved through

`Prog.gate` locates a gate from the root, activation by activation, and reads its operands in the activation of the
primitive node holding it; `LocR` is that walk from any activation (`run_locR`), with one list of operands
(`locR_unique`), `circuitOf`'s at the root (`locR_root`), a member's gate located from its caller's activation as from
the member's (`locR_child`), and a primitive node's gate with its member's operands (`locR_prim`).

`RootSet` is `NodeSet` at the root's activation (`rootRead_eq`): node `k`'s positions (`nodePos`) name outside it
exactly the gates its gates read outside it. `NodeSet` of a body holds wherever its callees' `UsedExact` does: a
member's gates read outside it the leaves of its parameters its callee's `used` names, which one step up
(`den_up`, `den_up_batch`, `den_up_scan`, `den_up_scan0`) are the positions `argPos` and `carryPos` read, `down`
keeping their gates (`specs_decoded`). `UsedExact` of a Definition holds wherever `NodeSet` holds of its body's nodes:
`summarize`'s `used` is the parameter pieces its nodes' reads of one copy name (`ivsOf_spec`, `norm_mem`), and a gate
position of the body is one of the activation's (`locW_lt`). Callees first, as `memoOf_retOk`.

At a root node calling a primitive, `RootSet` holds outright (`rootSet_prim`): its node read reads each argument whole
(`nodeReads_prim_ok`), `down` keeping each reference's gate (`readPos_agree`), each reference is read by a member's gate
(`operands_ok`, a mapped argument's by the member its index falls in, `tile`), and each operand is one of those
references, resolving before the node.

At a composite node, what its gates read outside it is what its members' parameter leaves that the callee's `used` names
resolve to (`compReads`), each member's activation located as its caller's is (`located_child`). One argument's node
read reads, outside the node, what the members' reads of its parameter leaves resolve to, when its positions are those
leaves' `upIdx` (`readSet`, through `branch_iff`). Its read succeeding (`nodeReads_comp_ok`, `ArgReadOk`), an argument's
read names exactly its members' reads of the parameter (`argSet`, `ArgB`) wherever its members' leaves tile it, and the
node's positions what its gates read outside it wherever its carries' read names the later iterations' (`compSet_of`).
All members at once, `Desc.memberRows` reads per used run the rows `rowsAt` names (`memberRows_rows`), whose indices are
the members' leaves there (`rowsAt_idx`), each through `throughPS` keeping its references' gates (`throughPS_agree`):
the same set (`rowsSet`, `argSet_of`).

A scan's carries, `Desc.carriesRead` reads outside the node exactly what its iterations past the first read at the
carry leaves `used` names (`carrySet`, `CarryRead`). A run the step returns in place is the initial carry's in every
iteration (`back_init`). Any other leaf, followed back through the step's return (`chain_step`), is the initial carry's
leaf the chain reaches within the iterations before it leaves (every such leaf kept, the sort only permuting:
`insertLoop_has`, `qsort_toList_perm`), or what the step returns where the chain leaves, in iteration `m - 1 - e`
(`back_exit`, `ret_den`): a gate of the node, which the read drops (`ret_gate_in`), an `xs` row per iteration
(`throughPS_agree`, else `CalleeSpec` iteration by iteration), or a shared parameter, one read for every iteration
(`up_idx`).

From the decoder's facts (`RootFacts`, `WidthFacts`) and a composite node's shape (`CompShape`), `NodeSet` holds at
every node calling a composite (`nodeSet_comp_of`): every parameter position resolves, `Desc.walkable` keeping
resolution within `Flock.Steps.FUEL` (`param_ok`, from `OperandsResolve.resolve_ok`), a callee's `used` names its
parameter leaves in range (`used_lt`) and its return its gates (`locW_lt`). So `UsedExact` holds of every reached
composite but the root (`usedExact_reached`), and `RootSet` wherever every reference of a node calling a primitive is
read by a member and every node calling a composite is shaped as `CompShape` says (`rootSet_of`).
-/

namespace NciCheck.Properties.SecurityProofs.ReadsExact
open NciCheck.Desc (Piece Seg PS Tgt)
open Flock.Program (Prog)
open DescAgree (nodeCount rootOffs)

/-- The member of node `k` of Definition `d`'s body that holds the body's gate `x`, as `gateStep` takes it. -/
def memberAt (P : Prog) (d k x : Nat) : Nat :=
  if (((P.defs[d]!).body.nodes[k]!).form == .prim || ((P.defs[d]!).body.nodes[k]!).form == .call) = true then 0
  else (x - (P.defs[d]!).body.offsets[k]!) / (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates

/-- Node `k` is the one `gateStep`'s search finds holding gate `x` of Definition `d`'s body. -/
def FindsAt (P : Prog) (d x k : Nat) : Prop :=
  (Array.range (P.defs[d]!).body.nodes.size).find? (fun k =>
    (P.defs[d]!).body.offsets[k]! ≤ x && x < (P.defs[d]!).body.offsets[k + 1]!) = some k

theorem findsAt_unique {P : Prog} {d x k k' : Nat} (h : FindsAt P d x k) (h' : FindsAt P d x k') : k = k' := by
  unfold FindsAt at h h'
  rw [h] at h'
  exact Option.some.inj h'

/-- Gate `g` located from activation `sc` as `Prog.gate` locates it, with its operands `ops`: the node holding it calls
a primitive, and `ops` are its member's operands in `sc`, or a composite, whose member's activation locates it. -/
inductive LocR (P : Prog) : Flock.Program.Scope → Nat → Array Nat → Prop
  | prim {sc : Flock.Program.Scope} {g k : Nat} {ops : Array Nat} :
      FindsAt P sc.d (g - sc.off) k → (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true →
      Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) (memberAt P sc.d k (g - sc.off)) = .ok ops →
      LocR P sc g ops
  | comp {sc : Flock.Program.Scope} {g k : Nat} {ops : Array Nat} :
      FindsAt P sc.d (g - sc.off) k → (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false →
      LocR P (P.child sc k (memberAt P sc.d k (g - sc.off))) g ops → LocR P sc g ops

/-- `gateStep` from activation `sc`: the node holding `g`, and its member's operands when it calls a primitive, else
its member's activation. -/
theorem gateStep_of (P : Prog) (g : Nat) (sc : Flock.Program.Scope) {x}
    (h : Flock.Program.gateStep P g sc = .ok x) :
    ∃ k, FindsAt P sc.d (g - sc.off) k ∧
      (((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true ∧ ∃ ops,
          Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) (memberAt P sc.d k (g - sc.off)) = .ok ops ∧
          x = .inl (((P.defs[sc.d]!).body.nodes[k]!).fn, ops, sc)) ∨
        ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false ∧
          x = .inr (P.child sc k (memberAt P sc.d k (g - sc.off))))) := by
  obtain ⟨k, hk, -⟩ := DescAgree.gateStep_eq P g sc x h
  refine ⟨k, hk, ?_⟩
  rw [gateStep_at P g sc k hk] at h
  have hm : memberAt P sc.d k (g - sc.off) =
      (if (((P.defs[sc.d]!).body.nodes[k]!).form == .prim || ((P.defs[sc.d]!).body.nodes[k]!).form == .call) = true
        then 0 else (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) /
          (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) := rfl
  rw [hm]
  by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
  · simp only [hp, ite_true] at h
    rcases ho : Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!)
        (if (((P.defs[sc.d]!).body.nodes[k]!).form == .prim || ((P.defs[sc.d]!).body.nodes[k]!).form == .call) = true
          then 0 else (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) /
            (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) with e | ops <;> rw [ho] at h
    · cases h
    · simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact Or.inl ⟨hp, ops, rfl, h.symm⟩
  · have hp' : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false := by simpa using hp
    simp only [hp', Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq] at h
    exact Or.inr ⟨hp', h.symm⟩

/-- **`Prog.gate`'s walk locates a gate with the operands it ends at.** -/
theorem run_locR (P : Prog) (g : Nat) : ∀ (n : Nat) (sc : Flock.Program.Scope) {p : Nat} {ops : Array Nat}
    {sc' : Flock.Program.Scope}, Flock.Steps.run "gate location" (Flock.Program.gateStep P g) n sc = .ok (p, ops, sc') →
    LocR P sc g ops
  | 0, _, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, sc, p, ops, sc', h => by
    rw [Flock.Steps.run.eq_2] at h
    rcases hs : Flock.Program.gateStep P g sc with e | x
    · simp [hs, bind, Except.bind] at h
    obtain ⟨k, hk, ⟨hp, ops', ho, rfl⟩ | ⟨hp, rfl⟩⟩ := gateStep_of P g sc hs
    · simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl, -⟩ := h
      exact LocR.prim hk hp ho
    · simp only [hs, bind, Except.bind] at h
      exact LocR.comp hk hp (run_locR P g n _ h)

/-- A gate has one list of operands. -/
theorem locR_unique {P : Prog} {sc : Flock.Program.Scope} {g : Nat} {ops ops' : Array Nat}
    (h : LocR P sc g ops) (h' : LocR P sc g ops') : ops = ops' := by
  induction h generalizing ops' with
  | prim hf hp ho =>
    cases h' with
    | prim hf' _ ho' =>
      obtain rfl := findsAt_unique hf hf'
      rw [ho] at ho'
      exact Except.ok.inj ho'
    | comp hf' hp' _ =>
      obtain rfl := findsAt_unique hf hf'
      rw [hp] at hp'
      cases hp'
  | comp hf hp _ ih =>
    cases h' with
    | prim hf' hp' _ =>
      obtain rfl := findsAt_unique hf hf'
      rw [hp] at hp'
      cases hp'
    | comp hf' _ h'' =>
      obtain rfl := findsAt_unique hf hf'
      exact ih h''

/-- **A gate's operands in `circuitOf`'s circuit are the ones it is located with from the root.** -/
theorem locR_root {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) {g : Nat} (hg : g < gates P) :
    LocR P P.rootScope g c.reads[g]! := by
  obtain ⟨p, sc, hp, -⟩ := circuitOf_gate hc hg
  unfold Flock.Program.Prog.gate at hp
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hp
  simp only [hge, ite_false, bind, Except.bind] at hp
  exact run_locR P g _ _ hp

/-- Gate `x` of a laid-out body of Definition `d`, in member `m` of node `k` (`x - offsets[k] = m * gates + j`, `j`
within the callee's gates): the node `gateStep` finds and the member it takes. -/
theorem findsAt_member (P : Prog) {d k m j : Nat} (hL : Laid P d) (hk : k < (P.defs[d]!).body.nodes.size)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members) (hj : j < (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates) :
    FindsAt P d ((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j) k ∧
      memberAt P d k ((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j) = m := by
  have hmono : ∀ k' < (P.defs[d]!).body.nodes.size,
      (P.defs[d]!).body.offsets[k']! ≤ (P.defs[d]!).body.offsets[k' + 1]! := fun k' hk' => by
    rw [hL k' hk']; exact Nat.le_add_right _ _
  have hxk : (P.defs[d]!).body.offsets[k]! ≤
        (P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j ∧
      (P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j <
        (P.defs[d]!).body.offsets[k + 1]! := by
    rw [hL k hk]
    have hle := Nat.mul_le_mul_right (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates hm
    rw [Nat.succ_mul] at hle
    omega
  refine ⟨find_node hmono hk hxk, ?_⟩
  have hG : 0 < (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates := by omega
  unfold memberAt
  unfold Flock.Program.Node.members at hm
  revert hm
  cases hf : ((P.defs[d]!).body.nodes[k]!).form <;> intro hm
  · exact (Nat.lt_one_iff.1 hm).symm
  · exact (Nat.lt_one_iff.1 hm).symm
  all_goals
    show (_ - _) / _ = m
    rw [show (P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j -
        (P.defs[d]!).body.offsets[k]! = j + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates by omega,
      Nat.add_mul_div_right _ _ hG, Nat.div_eq_of_lt hj, Nat.zero_add]

/-- **A member's gate is located from its caller's activation as from the member's**: in a laid-out body, gate
`j` of member `m` of composite node `k`. -/
theorem locR_child (P : Prog) {sc : Flock.Program.Scope} {k m j : Nat} (hL : Laid P sc.d)
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members)
    (hj : j < (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) {ops : Array Nat} :
    LocR P sc (sc.off + ((P.defs[sc.d]!).body.offsets[k]! +
        m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates + j)) ops ↔
      LocR P (P.child sc k m) (sc.off + ((P.defs[sc.d]!).body.offsets[k]! +
        m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates + j)) ops := by
  obtain ⟨hf, hma⟩ := findsAt_member P hL hk hm hj
  have e : sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates + j) -
      sc.off = (P.defs[sc.d]!).body.offsets[k]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates + j := by
    omega
  constructor
  · intro h
    cases h with
    | prim hf' hp' _ =>
      rw [e] at hf'
      obtain rfl := findsAt_unique hf hf'
      rw [hp] at hp'; cases hp'
    | comp hf' _ h' =>
      rw [e] at hf' h'
      obtain rfl := findsAt_unique hf hf'
      rwa [hma] at h'
  · intro h
    refine LocR.comp (k := k) (by rw [e]; exact hf) hp ?_
    rwa [e, hma]

/-- **A primitive node's gate is located with its member's operands**: in a laid-out body, member `m` of node `k`
calling a primitive of one gate. -/
theorem locR_prim (P : Prog) {sc : Flock.Program.Scope} {k m : Nat} (hL : Laid P sc.d)
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hG : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates = 1)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) {ops : Array Nat} :
    LocR P sc (sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m)) ops ↔
      Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops := by
  obtain ⟨hf, hma⟩ := findsAt_member P hL hk hm (j := 0) (by omega)
  rw [hG, Nat.mul_one, Nat.add_zero] at hf hma
  have e : sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m) - sc.off = (P.defs[sc.d]!).body.offsets[k]! + m := by
    omega
  constructor
  · intro h
    cases h with
    | prim hf' _ ho =>
      rw [e] at hf' ho
      obtain rfl := findsAt_unique hf hf'
      rwa [hma] at ho
    | comp hf' hp' _ =>
      rw [e] at hf'
      obtain rfl := findsAt_unique hf hf'
      rw [hp] at hp'; cases hp'
  · intro h
    exact LocR.prim (k := k) (by rw [e]; exact hf) hp (by rw [e, hma]; exact h)

/-! ## The statements `RootSet` is proved through -/

/-- Gate `a` is one of node `k`'s of activation `sc`. -/
def InNodeAt (P : Prog) (sc : Flock.Program.Scope) (k a : Nat) : Prop :=
  sc.off + (P.defs[sc.d]!).body.offsets[k]! ≤ a ∧ a < sc.off + (P.defs[sc.d]!).body.offsets[k + 1]!

/-- A gate of node `k` of activation `sc`, located from `sc`, reads gate `a`. -/
def NodeReadsAt (P : Prog) (sc : Flock.Program.Scope) (k a : Nat) : Prop :=
  ∃ g, InNodeAt P sc k g ∧ ∃ ops, LocR P sc g ops ∧ a ∈ ops

/-- Every gate of activation `sc` is located from it (as every gate is from the root, `circuitOf` passing). -/
def Located (P : Prog) (sc : Flock.Program.Scope) : Prop :=
  ∀ g, sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates → ∃ ops, LocR P sc g ops

/-- The positions node `k` of body `c` reads (`nodeReads_prim`, `nodeReads_comp`), a composite's before `keep`:
`rootRead`, at any body. -/
def nodePos (c : Desc.Ctx) (node : Flock.Program.Node) (k : Nat) : List (Tgt × Int) :=
  if (c.P.defs[node.fn]!).prim = true then
    (if (node.form == .batch && node.n == 0) = true then [] else
      (List.range node.args.size).flatMap fun i => readPos c k i #[(0, (c.arg k i).count)])
  else if node.members = 0 then []
  else (List.range node.args.size).flatMap (argPos c k node) ++ carryPos c k node

theorem rootRead_eq (P : Prog) (memo : Array Desc.Summary) (node : Flock.Program.Node) (k : Nat) :
    rootRead P memo node k = nodePos (rootCtx P memo) node k := rfl

/-- **Node `k` of Definition `f`'s body reads outside itself exactly the gates its positions name there**, in every
activation of `f` whose gates are located: `RootSet` at the root's. -/
def NodeSet (P : Prog) (memo : Array Desc.Summary) (f k : Nat) : Prop :=
  ∀ sc : Flock.Program.Scope, sc.d = f → Resolve.Good P sc → Located P sc → ∀ node,
    (ctxOf P memo f).node k = .ok node →
    (∀ a : Nat, ((∃ x ∈ nodePos (ctxOf P memo f) node k, den P sc x = .ok a) ∧ ¬ InNodeAt P sc k a) ↔
      (NodeReadsAt P sc k a ∧ ¬ InNodeAt P sc k a)) ∧
    ∀ i (p : Int), (Tgt.ref true i, p) ∈ nodePos (ctxOf P memo f) node k → ∃ a, den P sc (.ref true i, p) = .ok a

/-- **Definition `f`'s summarized parameter reads (`used`) are what its activations read outside themselves**: in an
activation `sc` whose gates are located, a gate reading a gate outside `sc` reads parameter `i`'s leaf `j` there, `j`
in `used[i]`, and each such leaf is read so by a gate of `sc`. -/
def UsedExact (P : Prog) (memo : Array Desc.Summary) (f : Nat) : Prop :=
  ∀ sc : Flock.Program.Scope, sc.d = f → Resolve.Good P sc → Located P sc →
    (∀ g ops a, sc.off ≤ g → g < sc.off + (P.defs[f]!).gates → LocR P sc g ops → a ∈ ops →
      ¬ (sc.off ≤ a ∧ a < sc.off + (P.defs[f]!).gates) →
      ∃ i j, Width.inIvs (((memo[f]!).used[i]?.getD #[]).toList) j ∧
        Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a) ∧
    ∀ i j, Width.inIvs (((memo[f]!).used[i]?.getD #[]).toList) j →
      ∃ g, sc.off ≤ g ∧ g < sc.off + (P.defs[f]!).gates ∧ ∃ ops, LocR P sc g ops ∧
        ∃ a ∈ ops, Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a

/-! ## An argument's reads, and a node's operands -/

theorem forall2_mem_left {α β : Type} {R : α → β → Prop} : ∀ {l : List α} {m : List β}, List.Forall₂ R l m →
    ∀ a ∈ l, ∃ b ∈ m, R a b
  | _, _, .nil, _, ha => absurd ha (by simp)
  | _, _, .cons (b := b) h hrest, a, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨b, List.mem_cons_self, h⟩
    · obtain ⟨b', hb', h'⟩ := forall2_mem_left hrest a ha
      exact ⟨b', List.mem_cons_of_mem _ hb', h'⟩

/-- A position of an agreeing list names the gate what it agrees with resolves to, wherever that resolves. -/
theorem agree_src {P : Prog} {sc : Flock.Program.Scope} {xs : List (Tgt × Int)} {vs : List (Except String Nat)}
    (h : Agree P sc xs vs) {x : Tgt × Int} (hx : x ∈ xs) : ∃ v ∈ vs, ∀ a, v = .ok a → den P sc x = .ok a :=
  forall2_mem_left h x hx

/-- What resolves in a list an agreeing one agrees with, a position of it names. -/
theorem agree_tgt {P : Prog} {sc : Flock.Program.Scope} {xs : List (Tgt × Int)} {vs : List (Except String Nat)}
    (h : Agree P sc xs vs) {v : Except String Nat} (hv : v ∈ vs) {a : Nat} (ha : v = .ok a) :
    ∃ x ∈ xs, den P sc x = .ok a := by
  obtain ⟨x, hx, hxv⟩ := forall2_mem h v hv
  exact ⟨x, hx, hxv a ha⟩

/-- Node `k`'s argument `i` as the context holds it counts its references. -/
theorem argCount_eq {c : Desc.Ctx} {k i : Nat} (hA : ArgOk c k i) :
    (c.arg k i).count = ((c.body.nodes[k]!).args[i]!).seq.count := by
  have h1 := hA.1.2.2.2
  have hl := segsPos_length (c.arg k i).segs.toList (fun g hg => segOk_of_mem _ hA.1 g hg)
  have h2 := congrArg List.length hA.2
  unfold psPos segsPos at h2
  rw [hl, List.length_map, List.length_range] at h2
  rw [h1, h2]

/-- `throughArg`'s read of an argument's span, `oneOf`, names what `down` gives each of its pieces. -/
theorem oneOf_pos (c : Desc.Ctx) (k i lo hi : Nat) {out : Array Seg}
    (h : oneOf c Flock.Steps.FUEL k i lo hi = .ok out) :
    segsPos out = ((c.arg k i).pieces lo hi).toList.flatMap (downPos c) := by
  unfold oneOf at h
  dsimp only at h
  rw [← Array.forIn_toList] at h
  obtain ⟨h1, -⟩ := forIn_pm (downPos c) (fun _ _ => True) _ (fun q r y hy => by
    obtain ⟨segs, hs, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    exact ⟨r ++ segs, hy.symm, by rw [segsPos_append, downPos, hs], fun s _ => Or.inr trivial⟩) _ #[] out
    (ok_bind_pure h)
  rw [h1, segsPos_empty, List.nil_append]

/-- `nodeReads`' reads of node `k`'s argument `i` over `spans` succeed. -/
def ReadsOk (c : Desc.Ctx) (k i : Nat) (spans : Array (Nat × Nat)) : Prop :=
  ∀ x ∈ spans.toList, ∃ out, oneOf c Flock.Steps.FUEL k i x.1 x.2 = .ok out

/-- **An argument's spans, read through `down`, keep the gates their references resolve to** wherever the reads
succeed. -/
theorem readPos_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (hD : DownSpec c sc Flock.Steps.FUEL) {k i : Nat}
    (hA : ArgOk c k i)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {spans : Array (Nat × Nat)}
    (hsp : ∀ x ∈ spans.toList, x.1 < x.2 → x.2 ≤ ((c.body.nodes[k]!).args[i]!).seq.count)
    (hok : ReadsOk c k i spans) :
    Agree c.P sc (readPos c k i spans) ((spansPos spans).map fun t =>
      Flock.Program.resolve c.P sc (((c.body.nodes[k]!).args[i]!).seq.get t)) := by
  unfold readPos spansPos
  rw [List.map_flatMap]
  refine agree_flatMap _ _ _ fun x hx => ?_
  obtain ⟨out, hout⟩ := hok x hx
  rw [← oneOf_pos c k i x.1 x.2 hout]
  exact one_agree c sc _ hD hA hR (hsp x hx) hout

theorem forIn_push_ok {α β ε : Type} (f : β → Except ε α) :
    ∀ (l : List β) (acc out : Array α),
      forIn l acc (fun x (r : Array α) => do let v ← f x; pure (ForInStep.yield (r.push v))) = .ok out →
      ∀ x ∈ l, ∃ a, f x = .ok a
  | [], _, _, _, x, hx => by simp at hx
  | y :: l, acc, out, h, x, hx => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := Walk.except_bind_ok h
    obtain ⟨v, hv, hst⟩ := Walk.except_bind_ok hst
    simp only [pure, Except.pure, Except.ok.injEq] at hst
    subst hst
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨v, hv⟩
    · exact forIn_push_ok f l _ out h x hx

theorem forIn_nest_ok {α β γ ε : Type} (L : β → List γ) (f : β → γ → Except ε α) :
    ∀ (l : List β) (acc out : Array α),
      forIn l acc (fun i (r : Array α) => do
          let r' ← forIn (L i) r (fun t (s : Array α) => do let v ← f i t; pure (ForInStep.yield (s.push v)))
          pure (ForInStep.yield r')) = .ok out →
      ∀ i ∈ l, ∀ t ∈ L i, ∃ a, f i t = .ok a
  | [], _, _, _, i, hi => by simp at hi
  | j :: l, acc, out, h, i, hi => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := Walk.except_bind_ok h
    obtain ⟨r', hx, hst⟩ := Walk.except_bind_ok hst
    simp only [pure, Except.pure, Except.ok.injEq] at hst
    subst hst
    rcases List.mem_cons.1 hi with rfl | hi
    · exact forIn_push_ok (f i) (L i) acc r' hx
    · exact forIn_nest_ok L f l r' out h i hi

/-- **Each of a gate's operands resolves**: every reference of its node's arguments its member reads. -/
theorem operands_ok (P : Prog) (sc : Flock.Program.Scope) (node : Flock.Program.Node) (m : Nat) {ops : Array Nat}
    (h : Flock.Program.operands P sc node m = .ok ops) :
    ∀ i < node.args.size, ∀ t < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
        ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
      ∃ a, Flock.Program.resolve P sc (if (node.form == .batch) = true then (node.args[i]!).memberRef m t
        else (node.args[i]!).seq.get t) = .ok a := by
  unfold Flock.Program.operands at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have h' := ok_bind_pure h
  intro i hi t ht
  exact forIn_nest_ok _ _ _ _ _ h' i (by simp [List.mem_range'_1, Std.Legacy.Range.size]; omega) t
    (by simpa [List.mem_range'_1, Std.Legacy.Range.size] using ht)

/-- **A batch's members tile a mapped argument**: each of its `n · M` references is some member's `memberIdx` of one of
its `M` leaves, `L` dividing `M`. -/
theorem tile {n L M idx : Nat} (hM : L ∣ M) (h : idx < n * M) :
    ∃ m < n, ∃ t < M, (t / L * n + m) * L + t % L = idx := by
  obtain ⟨s, rfl⟩ := hM
  have hL : 0 < L := by
    rcases Nat.eq_zero_or_pos L with h0 | h0
    · subst h0; simp at h
    · exact h0
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with h0 | h0
    · subst h0; simp at h
    · exact h0
  have hq : idx / L < n * s := (Nat.div_lt_iff_lt_mul hL).2 (by
    rw [show n * s * L = n * (L * s) by rw [Nat.mul_assoc, Nat.mul_comm s L]]; exact h)
  have hqs : idx / L / n < s := (Nat.div_lt_iff_lt_mul hn).2 (by rw [Nat.mul_comm]; exact hq)
  refine ⟨idx / L % n, Nat.mod_lt _ hn, idx / L / n * L + idx % L, ?_, ?_⟩
  · have h1 : (idx / L / n + 1) * L ≤ s * L := Nat.mul_le_mul_right _ hqs
    have h2 := Nat.mod_lt idx hL
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm s L] at h1
    omega
  · have e1 : (idx / L / n * L + idx % L) / L = idx / L / n := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hL, Nat.div_eq_of_lt (Nat.mod_lt _ hL), Nat.zero_add]
    have e2 : (idx / L / n * L + idx % L) % L = idx % L := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_mod]
    rw [e1, e2, Nat.div_add_mod' (idx / L) n, Nat.div_add_mod' idx L]

/-! ## A primitive root node -/

/-- A loop that succeeds took a step at each element, each step's element one `Q` holds of. -/
theorem forIn_each {β σ ε : Type} (f : β → σ → Except ε (ForInStep σ)) (Q : β → Prop)
    (hf : ∀ x r y, f x r = .ok y → Q x ∧ ∃ r', y = .yield r') :
    ∀ (l : List β) (acc out : σ), forIn l acc f = .ok out → ∀ x ∈ l, Q x
  | [], _, _, _, x, hx => by simp at hx
  | y :: l, acc, out, h, x, hx => by
    rw [List.forIn_cons] at h
    rcases hy : f y acc with e | st <;> rw [hy] at h
    · cases h
    obtain ⟨hq, r', rfl⟩ := hf y acc _ hy
    rcases List.mem_cons.1 hx with rfl | hx
    · exact hq
    · exact forIn_each f Q hf l r' out h x hx

/-- A loop each of whose steps goes on succeeds. -/
theorem forIn_ok_of {β σ ε : Type} (f : β → σ → Except ε (ForInStep σ)) :
    ∀ (l : List β) (acc : σ), (∀ x ∈ l, ∀ r, ∃ r', f x r = .ok (.yield r')) → ∃ out, forIn l acc f = .ok out
  | [], acc, _ => ⟨acc, rfl⟩
  | x :: l, acc, h => by
    obtain ⟨r', hr'⟩ := h x List.mem_cons_self acc
    obtain ⟨out, ho⟩ := forIn_ok_of f l r' (fun y hy => h y (List.mem_cons_of_mem _ hy))
    exact ⟨out, by rw [List.forIn_cons, hr']; exact ho⟩

theorem bind_pure_ok_of {α : Type} {x : Except String α} (h : ∃ y, x = .ok y) :
    ∃ y, (x >>= fun r => pure r) = .ok y := by
  obtain ⟨y, rfl⟩ := h
  exact ⟨y, rfl⟩

/-- `oneOf` succeeds where `down` does on each piece of the span. -/
theorem oneOf_ok (c : Desc.Ctx) (k i lo hi : Nat)
    (h : ∀ q ∈ ((c.arg k i).pieces lo hi).toList, ∃ segs, Desc.down c Flock.Steps.FUEL q = .ok segs) :
    ∃ out, oneOf c Flock.Steps.FUEL k i lo hi = .ok out := by
  unfold oneOf
  dsimp only
  rw [← Array.forIn_toList]
  apply bind_pure_ok_of
  apply forIn_ok_of
  intro q hq r
  obtain ⟨segs, hs⟩ := h q hq
  exact ⟨r ++ segs, by simp only [hs, bind, Except.bind, pure, Except.pure]⟩

/-- `nodeReads`' read of argument `i` over `spans`, succeeding, succeeds span by span. -/
theorem readLoop_ok (c : Desc.Ctx) (k i : Nat) (spans : Array (Nat × Nat)) (acc out : Array Seg)
    (h : forIn spans acc (fun (x : Nat × Nat) __s => do
        let __s ← forIn ((c.arg k i).pieces x.1 x.2) __s fun q __s => do
          let __do_lift ← Desc.down c Flock.Steps.FUEL q
          pure (ForInStep.yield (__s ++ __do_lift))
        pure (ForInStep.yield __s)) = .ok out) :
    ReadsOk c k i spans := by
  rw [← Array.forIn_toList] at h
  intro x hx
  refine oneOf_ok c k i x.1 x.2 (forIn_each _
    (fun x => ∀ q ∈ ((c.arg k i).pieces x.1 x.2).toList, ∃ segs, Desc.down c Flock.Steps.FUEL q = .ok segs)
    (fun x r y hy => ?_) _ _ _ h x hx)
  obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
  simp only [pure, Except.pure, Except.ok.injEq] at hy
  refine ⟨fun q hq => ?_, r', hy.symm⟩
  rw [← Array.forIn_toList] at hr'
  exact forIn_each _ (fun q => ∃ segs, Desc.down c Flock.Steps.FUEL q = .ok segs) (fun q r y hy => by
    obtain ⟨segs, hs, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    exact ⟨⟨segs, hs⟩, _, hy.symm⟩) _ _ _ hr' q hq

/-- **A primitive's node read succeeding reads each argument whole**, unless it is a batch of no members. -/
theorem nodeReads_prim_ok (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (hn : c.node k = .ok node)
    (hp : (c.P.defs[node.fn]!).prim = true) {rs : Array Seg} (h : Desc.nodeReads c k = .ok rs)
    (hb : ¬ (node.form == .batch && node.n == 0) = true) :
    ∀ i < node.args.size, ReadsOk c k i #[(0, (c.arg k i).count)] := by
  unfold Desc.nodeReads at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨node', hn', h⟩ := Walk.except_bind_ok h
  rw [hn] at hn'
  injection hn' with e
  subst e
  simp only [hp, ↓reduceIte] at h
  split at h
  · exact absurd ‹_› hb
  · obtain ⟨out, hout, -⟩ := Walk.except_bind_ok h
    intro i hi
    exact forIn_each _ (fun i => ReadsOk c k i #[(0, (c.arg k i).count)]) (fun i r y hy => by
      obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨readLoop_ok c k i _ #[] lift (ok_bind_pure hl), _, hy.symm⟩) _ _ _ hout i
      (by simp [List.mem_range'_1]; omega)

/-- A gate position names, at the root, the gate it is at. -/
theorem den_root_gate (P : Prog) {w : Nat} {p : Int} {a : Nat} (h : den P P.rootScope (.gate w, p) = .ok a) :
    p = (a : Int) := by
  simp only [den] at h
  split at h
  · cases h
  · rename_i hp
    injection h with h
    have ho : P.rootScope.off = 0 := rfl
    rw [ho, Nat.zero_add] at h
    rw [← h, Int.toNat_of_nonneg (by omega)]

/-- A primitive node's reference of argument `i`, member `m`'s `t`-th, is its sequence's at the index `operands` reads. -/
theorem ref_get (node : Flock.Program.Node) (i m t : Nat) :
    (if (node.form == .batch) = true then (node.args[i]!).memberRef m t else (node.args[i]!).seq.get t) =
      (node.args[i]!).seq.get (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t else t) := by
  split <;> simp only [Resolve.memberRef_eq]

/-- **`RootSet` at a root node calling a primitive**: its positions name exactly the gates its gates read, none of
them its own, given the node's references in range and resolving below it (`hLow`), its operands' indices in range
(`hOp`), and every reference read by some member unless it is a batch of none (`hCov`). -/
theorem rootSet_prim (P : Prog) (memo : Array Desc.Summary) (c : Circuit) (hc : circuitOf P = .ok c)
    (hl : Desc.rootLaidOut P = true) (hL : Laid P P.root) {k : Nat} (hk : k < nodeCount P)
    {node : Flock.Program.Node} {rs : Array Seg} (hn : (rootCtx P memo).node k = .ok node)
    (hp : (P.defs[node.fn]!).prim = true) (hG : (P.defs[node.fn]!).gates = 1)
    (hrs : Desc.nodeReads (rootCtx P memo) k = .ok rs) (hgt : ∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w)
    (hD : DownSpec (rootCtx P memo) P.rootScope Flock.Steps.FUEL)
    (hA : ∀ i < node.args.size, ArgOk (rootCtx P memo) k i)
    (hR : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count,
      Resolve.RefOk P P.root k ((node.args[i]!).seq.get t))
    (hLow : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∀ a,
      Flock.Program.resolve P P.rootScope ((node.args[i]!).seq.get t) = .ok a → a < (rootOffs P)[k]!)
    (hOp : Resolve.OpOk node (P.defs[node.fn]!))
    (hCov : ¬ (node.form == .batch && node.n == 0) = true → ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
      ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
        ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
      (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t) :
    ∀ a : Nat, (∃ w, (Tgt.gate w, (a : Int)) ∈ rootRead P memo node k) ↔
      ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! ∧ a ∉ nodeGates P k := by
  have hkb := node_lt hn
  have hg := node_get hn
  change k < (P.defs[P.root]!).body.nodes.size at hkb
  change (P.defs[P.root]!).body.nodes[k]! = node at hg
  obtain ⟨-, -, hK, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  have hend := DescAgree.offs_mono (rootOffs P) (nodeCount P) hmono (k + 1) (nodeCount P) (by omega) le_rfl
  have hoff : (rootOffs P)[k + 1]! = (rootOffs P)[k]! + node.members := by
    have := hL k hkb
    rw [hg, hG, Nat.mul_one] at this
    exact this
  have hgate : ∀ x ∈ rootRead P memo node k, ∃ w, x.1 = .gate w := by
    obtain ⟨out, rfl, hpos, -⟩ := nodeReads_prim (rootCtx P memo) k node hn hp rs hrs
    intro x hx
    rw [rootRead] at hx
    simp only [hp, ↓reduceIte] at hx
    rw [← hpos] at hx
    obtain ⟨s, hs, hx⟩ := List.mem_flatMap.1 hx
    obtain ⟨g, hgs, hxg⟩ := (mem_segPos_split s x).2 hx
    obtain ⟨q, hq, hxq⟩ := segPos_tgt hxg
    obtain ⟨w, hw⟩ := hgt g (Array.mem_flatMap.2 ⟨s, Array.mem_toList_iff.1 hs, Array.mem_toList_iff.1 hgs⟩) q
      (Array.mem_toList_iff.1 hq)
    exact ⟨w, hxq.trans hw⟩
  have hd : P.rootScope.d = P.root := rfl
  have hloc : ∀ m < node.members, ∀ ops, LocR P P.rootScope ((rootOffs P)[k]! + m) ops ↔
      Flock.Program.operands P P.rootScope node m = .ok ops := by
    intro m hm ops
    have := locR_prim P (sc := P.rootScope) (k := k) (m := m) hL hkb (by rw [hd, hg]; exact hp)
      (by rw [hd, hg]; exact hG) (by rw [hd, hg]; exact hm) (ops := ops)
    rw [hd, hg, show P.rootScope.off = 0 from rfl, Nat.zero_add] at this
    exact this
  by_cases hb : (node.form == .batch && node.n == 0) = true
  · have hm0 : node.members = 0 := by
      simp only [Bool.and_eq_true, beq_iff_eq] at hb
      unfold Flock.Program.Node.members
      rw [hb.1, hb.2]
      simp
    intro a
    constructor
    · rintro ⟨w, hx⟩
      rw [rootRead] at hx
      simp only [hp, hb, ↓reduceIte] at hx
      simp at hx
    · rintro ⟨g, hgk, -, -⟩
      have := (DescAgree.mem_nodeGates P k g).1 hgk
      omega
  have hg' : (rootCtx P memo).body.nodes[k]! = node := node_get hn
  have hok := nodeReads_prim_ok (rootCtx P memo) k node hn hp hrs hb
  have hag : ∀ i < node.args.size, Agree P P.rootScope
      (readPos (rootCtx P memo) k i #[(0, ((rootCtx P memo).arg k i).count)])
      ((List.range (node.args[i]!).seq.count).map fun t =>
        Flock.Program.resolve P P.rootScope ((node.args[i]!).seq.get t)) := by
    intro i hi
    have hc0 := argCount_eq (hA i hi)
    have := readPos_agree (rootCtx P memo) P.rootScope hD (hA i hi) (by rw [hg']; exact hR i hi)
      (spans := #[(0, ((rootCtx P memo).arg k i).count)])
      (fun x hx _ => by simp only [List.mem_singleton] at hx; subst hx; exact hc0.le) (hok i hi)
    rw [hg'] at this hc0
    have hsp : spansPos #[(0, ((rootCtx P memo).arg k i).count)] = List.range (node.args[i]!).seq.count := by
      simp [spansPos, List.range_eq_range', hc0]
    rw [hsp] at this
    exact this
  have hread : rootRead P memo node k = (List.range node.args.size).flatMap fun i =>
      readPos (rootCtx P memo) k i #[(0, ((rootCtx P memo).arg k i).count)] := by
    rw [rootRead]
    simp only [hp, hb, ↓reduceIte, Bool.false_eq_true]
  intro a
  constructor
  · rintro ⟨w, hx⟩
    rw [hread] at hx
    obtain ⟨i, hi, hx⟩ := List.mem_flatMap.1 hx
    have hi' := List.mem_range.1 hi
    obtain ⟨v, hv, hxv⟩ := agree_src (hag i hi') hx
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hv
    have ht' := List.mem_range.1 ht
    obtain ⟨m, hm, t', ht'', hidx⟩ := hCov hb i hi' t ht'
    have hops := (hloc m hm _).1 (locR_root hc (g := (rootOffs P)[k]! + m) (by omega))
    obtain ⟨a', ha'⟩ := operands_ok P _ _ m hops i hi' t' ht''
    rw [ref_get, hidx] at ha'
    have hpa := den_root_gate P (hxv a' ha')
    have haa : a = a' := by exact_mod_cast hpa
    subst haa
    refine ⟨(rootOffs P)[k]! + m, (DescAgree.mem_nodeGates P k _).2 ⟨by omega, by omega⟩, ?_, fun hin => ?_⟩
    · exact (operands_mem P _ _ m _ hops a).2 ⟨i, hi', t', ht'', by rw [ref_get, hidx]; exact ha'⟩
    · have h1 := hLow i hi' t ht' a ha'
      have h2 := (DescAgree.mem_nodeGates P k a).1 hin
      omega
  · rintro ⟨g, hgk, ha, -⟩
    have hgr := (DescAgree.mem_nodeGates P k g).1 hgk
    obtain ⟨m, rfl⟩ : ∃ m, g = (rootOffs P)[k]! + m := ⟨g - (rootOffs P)[k]!, by omega⟩
    have hm : m < node.members := by omega
    have hops := (hloc m hm _).1 (locR_root hc (by omega))
    obtain ⟨i, hi, t', ht', hres⟩ := (operands_mem P _ _ m _ hops a).1 ha
    rw [ref_get] at hres
    have hlt := hOp i hi m hm t' ht'
    obtain ⟨x, hx, hxa⟩ := agree_tgt (hag i hi) (List.mem_map.2 ⟨_, List.mem_range.2 hlt, rfl⟩) hres
    have hxr : x ∈ rootRead P memo node k := by
      rw [hread]
      exact List.mem_flatMap.2 ⟨i, List.mem_range.2 hi, hx⟩
    obtain ⟨w, hw⟩ := hgate x hxr
    obtain ⟨x1, x2⟩ := x
    simp only at hw
    subst hw
    have := den_root_gate P hxa
    subst this
    exact ⟨w, hxr⟩

/-! ## A composite node's reads, member by member -/

/-- **A member's activation has its gates located** where its caller's has: a member's gate is located from its
caller's activation as from its own (`locR_child`). -/
theorem located_child (P : Prog) {sc : Flock.Program.Scope} {k m : Nat} (hL : Laid P sc.d)
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members)
    (hend : (P.defs[sc.d]!).body.offsets[k + 1]! ≤ (P.defs[sc.d]!).gates) (hloc : Located P sc) :
    Located P (P.child sc k m) := by
  intro g h1 h2
  change sc.off + (P.defs[sc.d]!).body.offsets[k]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤ g at h1
  change g < sc.off + (P.defs[sc.d]!).body.offsets[k]! +
    m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates + (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates at h2
  have hle := Nat.mul_le_mul_right (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates hm
  rw [Nat.succ_mul] at hle
  have hK := hL k hk
  obtain ⟨ops, h⟩ := hloc g (by omega) (by omega)
  have e : g = sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
      (g - (sc.off + (P.defs[sc.d]!).body.offsets[k]! +
        m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates))) := by omega
  rw [e] at h ⊢
  exact ⟨ops, (locR_child P hL hk hp hm (by omega)).1 h⟩

/-- **A composite node reads outside itself exactly what its members' summarized parameter reads resolve to there**:
member `m`'s gates read outside the member the leaves of its parameters its callee's `used` names (`UsedExact`). -/
theorem compReads (P : Prog) (memo : Array Desc.Summary) {sc : Flock.Program.Scope} {k : Nat} (hL : Laid P sc.d)
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false)
    (hend : (P.defs[sc.d]!).body.offsets[k + 1]! ≤ (P.defs[sc.d]!).gates) (hloc : Located P sc)
    (hgood : Resolve.Good P sc) (hU : UsedExact P memo ((P.defs[sc.d]!).body.nodes[k]!).fn) (a : Nat) :
    (NodeReadsAt P sc k a ∧ ¬ InNodeAt P sc k a) ↔
      ((∃ m < ((P.defs[sc.d]!).body.nodes[k]!).members, ∃ i j,
        Width.inIvs (((memo[((P.defs[sc.d]!).body.nodes[k]!).fn]!).used[i]?.getD #[]).toList) j ∧
        Flock.Program.resolve P (P.child sc k m) ⟨true, i, j⟩ = .ok a) ∧ ¬ InNodeAt P sc k a) := by
  have hK := hL k hk
  constructor
  · rintro ⟨⟨g, ⟨hg1, hg2⟩, ops, hgl, ha⟩, hout⟩
    refine ⟨?_, hout⟩
    have hG : 0 < (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
      rcases Nat.eq_zero_or_pos (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates with h0 | h0
      · rw [h0, Nat.mul_zero] at hK; omega
      · exact h0
    have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
        ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by omega
    have hdm := Nat.div_add_mod (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!)
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates
    have hm : (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) / (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates <
        ((P.defs[sc.d]!).body.nodes[k]!).members := (Nat.div_lt_iff_lt_mul hG).2 hx
    have hj := Nat.mod_lt (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) hG
    set m := (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) / (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates
    set jj := (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) % (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates
    rw [Nat.mul_comm] at hdm
    have e : g = sc.off + ((P.defs[sc.d]!).body.offsets[k]! +
        m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates + jj) := by omega
    rw [e] at hgl
    have hc := (locR_child P hL hk hp hm hj).1 hgl
    rw [← e] at hc
    have hle := Nat.mul_le_mul_right (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates hm
    rw [Nat.succ_mul] at hle
    obtain ⟨i, j, hij, hr⟩ := (hU (P.child sc k m) rfl (Resolve.good_child P hgood hk rfl hp hm rfl)
      (located_child P hL hk hp hm hend hloc)).1 g ops a
      (by show _ + _ + _ ≤ g; omega) (by show g < _ + _ + _ + _; omega) hc ha (fun ⟨h1, h2⟩ => by
        change sc.off + (P.defs[sc.d]!).body.offsets[k]! +
          m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤ a at h1
        change a < sc.off + (P.defs[sc.d]!).body.offsets[k]! +
          m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
            (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates at h2
        exact hout ⟨by omega, by omega⟩)
    exact ⟨m, hm, i, j, hij, hr⟩
  · rintro ⟨⟨m, hm, i, j, hij, hr⟩, hout⟩
    refine ⟨?_, hout⟩
    obtain ⟨g, hg1, hg2, ops, hgl, a', ha', hr'⟩ :=
      (hU (P.child sc k m) rfl (Resolve.good_child P hgood hk rfl hp hm rfl)
        (located_child P hL hk hp hm hend hloc)).2 i j hij
    rw [hr] at hr'
    obtain rfl := Except.ok.inj hr'
    change sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤ g at hg1
    change g < sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
        (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates at hg2
    have hle := Nat.mul_le_mul_right (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates hm
    rw [Nat.succ_mul] at hle
    have e : g = sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
        (g - (sc.off + (P.defs[sc.d]!).body.offsets[k]! +
          m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates))) := by omega
    rw [e] at hgl
    have hc := (locR_child P hL hk hp hm (by omega)).2 hgl
    rw [← e] at hc
    exact ⟨g, ⟨by omega, by omega⟩, ops, hc, ha'⟩

/-! ## An argument's positions against its members' reads -/

theorem mem_spansPos (xs : Array (Nat × Nat)) (t : Nat) : t ∈ spansPos xs ↔ Width.inIvs xs.toList t := by
  unfold spansPos Width.inIvs
  simp only [List.mem_flatMap, List.mem_range'_1]
  constructor
  · rintro ⟨p, hp, h1, h2⟩; exact ⟨p, hp, h1, by omega⟩
  · rintro ⟨p, hp, h1, h2⟩; exact ⟨p, hp, h1, by omega⟩

/-- **Positions agreeing with references `L` name exactly what the members' reads `B` resolve to**, when `L` holds
exactly the references those reads go up to (`idx`), each read going up to its reference's gate (`hup`) and each
succeeding (`hsucc`). -/
theorem branch_iff {P : Prog} {sc : Flock.Program.Scope} {xs : List (Tgt × Int)} {L : List Nat}
    (R : Nat → Except String Nat) (S : Nat → Nat → Except String Nat) (B : Nat → Nat → Prop) (idx : Nat → Nat → Nat)
    (hag : Agree P sc xs (L.map R)) (hcov : ∀ t, t ∈ L ↔ ∃ m j, B m j ∧ t = idx m j)
    (hup : ∀ m j a, B m j → S m j = .ok a → R (idx m j) = .ok a)
    (hsucc : ∀ m j, B m j → ∃ a, S m j = .ok a) (a : Nat) :
    (∃ x ∈ xs, den P sc x = .ok a) ↔ ∃ m j, B m j ∧ S m j = .ok a := by
  constructor
  · rintro ⟨x, hx, hxa⟩
    obtain ⟨v, hv, hxv⟩ := agree_src hag hx
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hv
    obtain ⟨m, j, hB, rfl⟩ := (hcov t).1 ht
    obtain ⟨a', ha'⟩ := hsucc m j hB
    have := hxv a' (hup m j a' hB ha')
    rw [hxa] at this
    obtain rfl := Except.ok.inj this
    exact ⟨m, j, hB, ha'⟩
  · rintro ⟨m, j, hB, hS⟩
    exact agree_tgt hag (List.mem_map.2 ⟨idx m j, (hcov _).2 ⟨m, j, hB, rfl⟩, rfl⟩) (hup m j a hB hS)

/-- The index into argument `i` of node `node` that member `m`'s parameter leaf `j` reads (a scan's carry past the
first iteration aside): a batch member's `memberIdx`, a scan iteration's `xs` slice, else `j`. -/
def upIdx (node : Flock.Program.Node) (i m j : Nat) : Nat :=
  match node.form with
  | .batch => Resolve.memberIdx (node.args[i]!) m j
  | .scan => if i ≠ 0 ∧ i ≤ node.nxs then m * ((node.args[i]!).dims.map (·.2)).getD 0 + j else j
  | _ => j

/-- **Up through a member's parameter leaf**: what it resolves to, its node's reference `upIdx` resolves to in the
parent (a scan's carry past the first iteration aside). -/
theorem up_idx (P : Prog) (sc : Flock.Program.Scope) {k m i j a : Nat} {node : Flock.Program.Node}
    (hnode : (P.defs[sc.d]!).body.nodes[k]! = node) (hnc : node.form = .scan → i = 0 → m = 0)
    (h : Flock.Program.resolve P (P.child sc k m) ⟨true, i, j⟩ = .ok a) :
    Flock.Program.resolve P sc ((node.args[i]!).seq.get (upIdx node i m j)) = .ok a := by
  unfold Flock.Program.Prog.child at h
  dsimp only at h
  unfold upIdx
  rcases hf : node.form with _ | _ | _ | _
  · exact den_up P sc hnode (Or.inl hf) h
  · exact den_up P sc hnode (Or.inr hf) h
  · have := den_up_batch P sc hnode hf h
    rwa [Resolve.memberRef_eq] at this
  · by_cases hi : i = 0
    · subst hi
      have hm := hnc hf rfl
      subst hm
      have := den_up_scan0 P sc hnode hf h
      simp only [↓reduceIte] at this
      simpa using this
    · have := den_up_scan P sc hnode hf hi h
      by_cases hx : i ≤ node.nxs
      · simp only [hx, ↓reduceIte] at this
        simpa [hi, hx] using this
      · simp only [hx, ↓reduceIte] at this
        simpa [hi, hx] using this

/-- **A read of argument `i` over `spans` names exactly what the members' reads `B` resolve to**, where the spans'
indices are exactly the references those reads go up to (`upIdx`). -/
theorem readSet (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hD : DownSpec c sc Flock.Steps.FUEL)
    {k i : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hA : ArgOk c k i)
    (hR : ∀ t < (node.args[i]!).seq.count, Resolve.RefOk c.P c.d k ((node.args[i]!).seq.get t))
    {spans : Array (Nat × Nat)} (hsp : ∀ x ∈ spans.toList, x.1 < x.2 → x.2 ≤ (node.args[i]!).seq.count)
    (hok : ReadsOk c k i spans) (B : Nat → Nat → Prop)
    (hcov : ∀ t, t ∈ spansPos spans ↔ ∃ m j, B m j ∧ t = upIdx node i m j)
    (hB : ∀ m j, B m j → node.form = .scan → i = 0 → m = 0)
    (hsucc : ∀ m j, B m j → ∃ a, Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a) (a : Nat) :
    (∃ x ∈ readPos c k i spans, den c.P sc x = .ok a) ↔
      ∃ m j, B m j ∧ Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a := by
  have hg : c.body.nodes[k]! = node := node_get hn
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact hg
  have hag := readPos_agree c sc hD hA (by rw [hg]; exact hR) (by rw [hg]; exact hsp) hok
  rw [hg] at hag
  exact branch_iff _ _ B (upIdx node i) hag hcov
    (fun m j a hb h => up_idx c.P sc hnode (hB m j hb) h) hsucc a

/-! ## A composite node's arguments, read as its members read their parameters -/

/-- Argument `i` of `node` is read member by member (`nodeReads`): a batch's mapped argument or a scan's `xs`. -/
def perMember (node : Flock.Program.Node) (i : Nat) : Bool :=
  match node.form with
  | .batch => node.args[i]!.dims.isSome
  | .scan => decide (1 ≤ i) && decide (i ≤ node.nxs)
  | _ => false

theorem argPos_eq (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (i : Nat) :
    argPos c k node i =
      if (node.form == .scan && i == 0) = true then readPos c k 0 ((c.memo[node.fn]!).used[i]?.getD #[])
      else if (!perMember node i) = true then readPos c k i ((c.memo[node.fn]!).used[i]?.getD #[])
      else if ((c.memo[node.fn]!).used[i]?.getD #[]).covers
          (((c.P.defs[node.fn]!).params[i]?.map (·.2.leaves)).getD 0) = true then
        readPos c k i #[(0, (c.arg k i).count)]
      else match Desc.memberRows c node k i ((c.memo[node.fn]!).used[i]?.getD #[]) with
        | .ok (some ps) => ps.toList.flatMap (downPos c)
        | _ => (List.range node.members).flatMap fun m =>
          readPos c k i (Desc.argSpans node i m ((c.memo[node.fn]!).used[i]?.getD #[])) := by
  unfold argPos perMember
  cases node.form <;> rfl

/-- `nodeReads`' read of argument `i` of a composite's node `k` succeeds, branch by branch as `argPos` reads it. -/
def ArgReadOk (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (i : Nat) : Prop :=
  if (node.form == .scan && i == 0) = true then ReadsOk c k 0 ((c.memo[node.fn]!).used[i]?.getD #[])
  else if (!perMember node i) = true then ReadsOk c k i ((c.memo[node.fn]!).used[i]?.getD #[])
  else if ((c.memo[node.fn]!).used[i]?.getD #[]).covers
      (((c.P.defs[node.fn]!).params[i]?.map (·.2.leaves)).getD 0) = true then
    ReadsOk c k i #[(0, (c.arg k i).count)]
  else match Desc.memberRows c node k i ((c.memo[node.fn]!).used[i]?.getD #[]) with
    | .ok (some ps) => ∀ p ∈ ps.toList, ∃ segs, Desc.down c Flock.Steps.FUEL p = .ok segs
    | .ok none => ∀ m < node.members,
      ReadsOk c k i (Desc.argSpans node i m ((c.memo[node.fn]!).used[i]?.getD #[]))
    | .error _ => False

/-- **A composite's node read succeeding reads each argument as `argPos` does** (`ArgReadOk`), and a scan's carries. -/
theorem nodeReads_comp_ok (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (hn : c.node k = .ok node)
    (hp : (c.P.defs[node.fn]!).prim = false) {rs : Array Seg} (h : Desc.nodeReads c k = .ok rs)
    (hm : node.members ≠ 0) :
    (∀ i < node.args.size, ArgReadOk c k node i) ∧
      ((node.form == .scan && node.n ≥ 2) = true →
        ∃ segs, Desc.carriesRead c k ((c.memo[node.fn]!).used[0]?.getD #[]) = .ok segs) := by
  unfold Desc.nodeReads at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨node', hn', h⟩ := Walk.except_bind_ok h
  rw [hn] at hn'
  injection hn' with e
  subst e
  rw [if_neg (by rw [hp]; decide)] at h
  split at h
  · rename_i hm'
    exact absurd (by simpa using hm') hm
  obtain ⟨r1, h1, h⟩ := Walk.except_bind_ok h
  refine ⟨fun i hi => forIn_each _ (ArgReadOk c k node) (fun i r y hy => ?_) _ _ _ h1 i
    (by rw [List.mem_range'_1]; omega), fun hsc => ?_⟩
  · rw [ArgReadOk]
    rcases hform : node.form with _ | _ | _ | _ <;> simp only [hform, perMember] at hy ⊢
    all_goals
      split at hy
      · rename_i hs
        obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact ⟨by rw [if_pos hs]; exact readLoop_ok c k 0 _ #[] lift (ok_bind_pure hl), _, hy.symm⟩
      · rename_i hs
        split at hy
        · rename_i hpm
          obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          exact ⟨by rw [if_neg hs, if_pos hpm]; exact readLoop_ok c k i _ #[] lift (ok_bind_pure hl), _, hy.symm⟩
        · rename_i hpm
          split at hy
          · rename_i hcv
            obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
            simp only [pure, Except.pure, Except.ok.injEq] at hy
            exact ⟨by rw [if_neg hs, if_neg hpm, if_pos hcv]; exact readLoop_ok c k i _ #[] lift (ok_bind_pure hl),
              _, hy.symm⟩
          · rename_i hcv
            obtain ⟨mr, hmr, hy⟩ := Walk.except_bind_ok hy
            rcases mr with _ | ps
            · have hy := walkAtMost_bind _ _ _ hy
              obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              refine ⟨?_, _, hy.symm⟩
              rw [if_neg hs, if_neg hpm, if_neg hcv, hmr]
              intro m hm
              refine forIn_each _ (fun m => ReadsOk c k i (Desc.argSpans node i m
                ((c.memo[node.fn]!).used[i]?.getD #[]))) (fun m r y hy => ?_) _ _ _ hr' m
                (by rw [List.mem_range'_1]; omega)
              obtain ⟨lift, hl, hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              exact ⟨readLoop_ok c k i _ #[] lift (ok_bind_pure hl), _, hy.symm⟩
            · obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              refine ⟨?_, _, hy.symm⟩
              rw [if_neg hs, if_neg hpm, if_neg hcv, hmr]
              rw [← Array.forIn_toList] at hr'
              intro p hp'
              exact forIn_each _ (fun p => ∃ segs, Desc.down c Flock.Steps.FUEL p = .ok segs) (fun p r y hy => by
                obtain ⟨segs, hs, hy⟩ := Walk.except_bind_ok hy
                simp only [pure, Except.pure, Except.ok.injEq] at hy
                exact ⟨⟨segs, hs⟩, _, hy.symm⟩) _ _ _ hr' p hp'
  · rw [if_pos hsc] at h
    obtain ⟨cr, hcr, -⟩ := Walk.except_bind_ok h
    exact ⟨cr, hcr⟩

/-- Member `m`'s leaf `j` of parameter `i`, among those `u` names, as an argument's read reads it (`argPos`): a scan's
iterations after the first read their carry through `carriesRead`. -/
def ArgB (node : Flock.Program.Node) (u : Desc.Ivs) (i m j : Nat) : Prop :=
  m < node.members ∧ (node.form = .scan → i = 0 → m = 0) ∧ Width.inIvs u.toList j

/-- **A composite node's positions name outside it exactly what its gates read there**, where each argument's read
names its members' reads of the parameter (`ArgB`) and the carries' read the later iterations' reads of their carry. -/
theorem compSet_of (P : Prog) (memo : Array Desc.Summary) {sc : Flock.Program.Scope} {k : Nat} (hL : Laid P sc.d)
    (hk : k < (P.defs[sc.d]!).body.nodes.size) {node : Flock.Program.Node}
    (hg : (P.defs[sc.d]!).body.nodes[k]! = node) (hp : (P.defs[node.fn]!).prim = false)
    (hend : (P.defs[sc.d]!).body.offsets[k + 1]! ≤ (P.defs[sc.d]!).gates) (hloc : Located P sc)
    (hgood : Resolve.Good P sc) (hU : UsedExact P memo node.fn)
    (hbig : ∀ i j, Width.inIvs (((memo[node.fn]!).used[i]?.getD #[]).toList) j → i < node.args.size)
    (hArg : ∀ i < node.args.size, ∀ a, (∃ x ∈ argPos (ctxOf P memo sc.d) k node i, den P sc x = .ok a) ↔
      ∃ m j, ArgB node ((memo[node.fn]!).used[i]?.getD #[]) i m j ∧
        Flock.Program.resolve P (P.child sc k m) ⟨true, i, j⟩ = .ok a)
    (hCarry : ∀ a, ((∃ x ∈ carryPos (ctxOf P memo sc.d) k node, den P sc x = .ok a) ∧ ¬ InNodeAt P sc k a) ↔
      ((∃ m, node.form = .scan ∧ 1 ≤ m ∧ m < node.members ∧ ∃ j,
        Width.inIvs (((memo[node.fn]!).used[0]?.getD #[]).toList) j ∧
        Flock.Program.resolve P (P.child sc k m) ⟨true, 0, j⟩ = .ok a) ∧ ¬ InNodeAt P sc k a)) (a : Nat) :
    ((∃ x ∈ nodePos (ctxOf P memo sc.d) node k, den P sc x = .ok a) ∧ ¬ InNodeAt P sc k a) ↔
      (NodeReadsAt P sc k a ∧ ¬ InNodeAt P sc k a) := by
  subst hg
  rw [compReads P memo hL hk hp hend hloc hgood hU a]
  have hp' : ((ctxOf P memo sc.d).P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false := hp
  unfold nodePos
  simp only [hp', Bool.false_eq_true, ↓reduceIte]
  by_cases h0 : ((P.defs[sc.d]!).body.nodes[k]!).members = 0
  · simp only [h0, ↓reduceIte, List.not_mem_nil, false_and, exists_false, Nat.not_lt_zero]
  simp only [h0, ↓reduceIte]
  constructor
  · rintro ⟨⟨x, hx, hxa⟩, hout⟩
    refine ⟨?_, hout⟩
    rcases List.mem_append.1 hx with hx | hx
    · obtain ⟨i, hi, hx⟩ := List.mem_flatMap.1 hx
      rw [List.mem_range] at hi
      obtain ⟨m, j, ⟨hm, -, hj⟩, hr⟩ := (hArg i hi a).1 ⟨x, hx, hxa⟩
      exact ⟨m, hm, i, j, hj, hr⟩
    · obtain ⟨⟨m, -, -, hm, j, hj, hr⟩, -⟩ := (hCarry a).1 ⟨⟨x, hx, hxa⟩, hout⟩
      exact ⟨m, hm, 0, j, hj, hr⟩
  · rintro ⟨⟨m, hm, i, j, hj, hr⟩, hout⟩
    refine ⟨?_, hout⟩
    have hi := hbig i j hj
    by_cases hc : ((P.defs[sc.d]!).body.nodes[k]!).form = .scan ∧ i = 0 ∧ 1 ≤ m
    · obtain ⟨hs, rfl, h1⟩ := hc
      obtain ⟨⟨x, hx, hxa⟩, -⟩ := (hCarry a).2 ⟨⟨m, hs, h1, hm, j, hj, hr⟩, hout⟩
      exact ⟨x, List.mem_append_right _ hx, hxa⟩
    · obtain ⟨x, hx, hxa⟩ := (hArg i hi a).2
        ⟨m, j, ⟨hm, fun hs hi0 => Nat.eq_zero_of_not_pos fun h => hc ⟨hs, hi0, h⟩, hj⟩, hr⟩
      exact ⟨x, List.mem_append_left _ (List.mem_flatMap.2 ⟨i, List.mem_range.2 hi, hx⟩), hxa⟩

/-- Spans whose indices are below `n` end at or before it. -/
theorem spans_le {spans : Array (Nat × Nat)} {n : Nat} (h : ∀ t ∈ spansPos spans, t < n) :
    ∀ x ∈ spans.toList, x.1 < x.2 → x.2 ≤ n := by
  intro x hx hlt
  have := h (x.2 - 1) ((mem_spansPos spans _).2 ⟨x, hx, by omega, by omega⟩)
  omega

/-- Where an argument is read as the parameter (a scan's initial carry, or an argument not mapped over the members),
member `m`'s leaf `j` reads its reference `j`. -/
theorem upIdx_shared (node : Flock.Program.Node) (i m j : Nat)
    (h : (node.form == .scan && i == 0) = true ∨ (!perMember node i) = true) :
    upIdx node i m j = j := by
  unfold upIdx
  unfold perMember at h
  cases hf : node.form with
  | prim => rfl
  | call => rfl
  | batch =>
    simp only [hf] at h
    rcases hd : (node.args[i]!).dims with _ | ⟨d, L⟩
    · simp [Resolve.memberIdx, hd]
    · simp [hd] at h
      exact absurd h.1 (by decide)
  | scan =>
    simp only [hf] at h
    dsimp only
    split
    · rename_i hc
      rcases h with h | h <;> simp at h <;> omega
    · rfl

/-- Member `m`'s spans of a mapped argument (`Desc.argSpans`) are its used leaves' indices `upIdx`. -/
theorem spansPos_argSpans (node : Flock.Program.Node) (i m : Nat) (u : Desc.Ivs) {lv : Nat}
    (hpm : perMember node i = true)
    (hu : ∀ j, Width.inIvs u.toList j → j < lv)
    (hL0 : ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 → lv = 0) :
    spansPos (Desc.argSpans node i m u) = (spansPos u).map (upIdx node i m) := by
  unfold perMember at hpm
  cases hf : node.form with
  | prim => simp [hf] at hpm
  | call => simp [hf] at hpm
  | batch =>
    rw [argSpans_batch node i m u hf (fun d L hd h0 p hp => ?_)]
    · unfold upIdx; rw [hf]
    · have := hL0 d L hd h0
      by_contra hlt
      have := hu p.1 ⟨p, hp, Nat.le_refl _, by omega⟩
      omega
  | scan =>
    simp only [hf, Bool.and_eq_true, decide_eq_true_eq] at hpm
    rw [argSpans_xs node i m u hf hpm]
    unfold upIdx; rw [hf]
    simp [show i ≠ 0 by omega, hpm.2]

/-- **One argument's read names exactly what its members' reads of the parameter resolve to** (`ArgB`), where its
members' leaves tile it (`upIdx`, in range wherever used), its reads succeed (`ArgReadOk`), and so do its members'
reads (`hsucc`); all members at once (`Desc.memberRows`) as `hrows` says. -/
theorem argSet (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hD : DownSpec c sc Flock.Steps.FUEL)
    {k i : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hA : ArgOk c k i)
    (hR : ∀ t < (node.args[i]!).seq.count, Resolve.RefOk c.P c.d k ((node.args[i]!).seq.get t))
    (hm0 : node.members ≠ 0) (hok : ArgReadOk c k node i) {lv : Nat}
    (hlv : ((c.P.defs[node.fn]!).params[i]?.map (·.2.leaves)).getD 0 = lv)
    (hu : ∀ j, Width.inIvs (((c.memo[node.fn]!).used[i]?.getD #[]).toList) j → j < lv)
    (hL0 : ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 → lv = 0)
    (hlo : ∀ m < node.members, ∀ j < lv, (node.form = .scan → i = 0 → m = 0) →
      upIdx node i m j < (node.args[i]!).seq.count)
    (htile : ∀ t < (node.args[i]!).seq.count, ∃ m < node.members, ∃ j < lv, upIdx node i m j = t)
    (hsucc : ∀ m j, ArgB node ((c.memo[node.fn]!).used[i]?.getD #[]) i m j →
      ∃ a, Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a)
    (hrows : perMember node i = true → ∀ ps,
      Desc.memberRows c node k i ((c.memo[node.fn]!).used[i]?.getD #[]) = .ok (some ps) →
      (∀ p ∈ ps.toList, ∃ segs, Desc.down c Flock.Steps.FUEL p = .ok segs) → ∀ a,
      (∃ x ∈ ps.toList.flatMap (downPos c), den c.P sc x = .ok a) ↔
        ∃ m j, ArgB node ((c.memo[node.fn]!).used[i]?.getD #[]) i m j ∧
          Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a)
    (a : Nat) :
    (∃ x ∈ argPos c k node i, den c.P sc x = .ok a) ↔
      ∃ m j, ArgB node ((c.memo[node.fn]!).used[i]?.getD #[]) i m j ∧
        Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a := by
  set u := (c.memo[node.fn]!).used[i]?.getD #[] with hudef
  have hm0' : 0 < node.members := Nat.pos_of_ne_zero hm0
  have hB : ∀ m j, ArgB node u i m j → node.form = .scan → i = 0 → m = 0 := fun m j h => h.2.1
  -- a read of `u` as the parameter itself
  have shared : (node.form == .scan && i == 0) = true ∨ (!perMember node i) = true → ReadsOk c k i u →
      ((∃ x ∈ readPos c k i u, den c.P sc x = .ok a) ↔
        ∃ m j, ArgB node u i m j ∧ Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a) := by
    intro hsh hok'
    have hup := fun m j => upIdx_shared node i m j hsh
    refine readSet c sc hd hD hn hA hR (spans_le fun t ht => ?_) hok' _ (fun t => ?_) hB hsucc a
    · have := hlo 0 hm0' t (hu t ((mem_spansPos u t).1 ht)) (fun _ _ => rfl)
      rwa [hup] at this
    · rw [mem_spansPos]
      constructor
      · intro ht; exact ⟨0, t, ⟨hm0', fun _ _ => rfl, ht⟩, (hup 0 t).symm⟩
      · rintro ⟨m, j, ⟨-, -, hj⟩, rfl⟩; rwa [hup]
  rw [argPos_eq]
  unfold ArgReadOk at hok
  rw [← hudef] at hok ⊢
  by_cases h1 : (node.form == .scan && i == 0) = true
  · rw [if_pos h1] at hok ⊢
    have hi0 : i = 0 := by simp only [Bool.and_eq_true, beq_iff_eq] at h1; exact h1.2
    subst hi0
    exact shared (.inl h1) hok
  rw [if_neg h1] at hok ⊢
  by_cases h2 : (!perMember node i) = true
  · rw [if_pos h2] at hok ⊢
    exact shared (.inr h2) hok
  rw [if_neg h2] at hok ⊢
  have hpm : perMember node i = true := by simpa using h2
  have hnc : node.form = .scan → i ≠ 0 := by
    intro hs h0; subst h0; unfold perMember at hpm; rw [hs] at hpm; simp at hpm
  have hcnt := argCount_eq hA
  rw [node_get hn] at hcnt
  by_cases h3 : u.covers (((c.P.defs[node.fn]!).params[i]?.map (·.2.leaves)).getD 0) = true
  · rw [if_pos h3] at hok ⊢
    rw [hlv] at h3
    refine readSet c sc hd hD hn hA hR (fun x hx _ => ?_) hok _ (fun t => ?_) hB hsucc a
    · simp only [List.mem_singleton] at hx
      rw [hx, hcnt]
    · rw [mem_spansPos]
      constructor
      · rintro ⟨x, hx, h1, h2⟩
        simp only [List.mem_singleton] at hx
        subst hx
        rw [hcnt] at h2
        obtain ⟨m, hm, j, hj, rfl⟩ := htile t h2
        exact ⟨m, j, ⟨hm, fun hs hi => absurd hi (hnc hs), covers_mem u lv h3 j hj⟩, rfl⟩
      · rintro ⟨m, j, ⟨hm, hs, hj⟩, rfl⟩
        refine ⟨(0, (c.arg k i).count), by simp, Nat.zero_le _, ?_⟩
        rw [hcnt]
        exact hlo m hm j (hu j hj) hs
  rw [if_neg h3] at hok ⊢
  split at hok
  · rename_i ps hps
    rw [hps]
    exact hrows hpm ps hps hok a
  · rename_i hps
    rw [hps]
    simp only [List.mem_flatMap, List.mem_range]
    have hsp : ∀ m < node.members, ∀ t ∈ spansPos (Desc.argSpans node i m u), t < (node.args[i]!).seq.count := by
      intro m hm t ht
      rw [spansPos_argSpans node i m u hpm hu hL0] at ht
      obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ht
      exact hlo m hm j (hu j ((mem_spansPos u j).1 hj)) (fun hs hi => absurd hi (hnc hs))
    have hcov : ∀ m < node.members, ∀ t, t ∈ spansPos (Desc.argSpans node i m u) ↔
        ∃ m' j, (m' = m ∧ ArgB node u i m' j) ∧ t = upIdx node i m' j := by
      intro m hm t
      rw [spansPos_argSpans node i m u hpm hu hL0, List.mem_map]
      constructor
      · rintro ⟨j, hj, rfl⟩
        exact ⟨m, j, ⟨rfl, hm, fun hs hi => absurd hi (hnc hs), (mem_spansPos u j).1 hj⟩, rfl⟩
      · rintro ⟨m', j, ⟨rfl, -, -, hj⟩, rfl⟩
        exact ⟨j, (mem_spansPos u j).2 hj, rfl⟩
    have key := fun m (hm : m < node.members) => readSet c sc hd hD hn hA hR (spans_le (hsp m hm)) (hok m hm)
      (fun m' j => m' = m ∧ ArgB node u i m' j) (hcov m hm) (fun m' j h => h.2.2.1)
      (fun m' j h => hsucc m' j h.2) a
    constructor
    · rintro ⟨x, ⟨m, hm, hx⟩, hxa⟩
      obtain ⟨m', j, ⟨rfl, hB'⟩, hr⟩ := (key m hm).1 ⟨x, hx, hxa⟩
      exact ⟨m', j, hB', hr⟩
    · rintro ⟨m, j, hB', hr⟩
      obtain ⟨x, hx, hxa⟩ := (key m hB'.1).2 ⟨m, j, ⟨rfl, hB'⟩, hr⟩
      exact ⟨x, ⟨m, hB'.1, hx⟩, hxa⟩
  · exact absurd hok id

/-! ## All members' reads of a mapped argument at once (`Desc.memberRows`) -/

/-- The pieces of rows `Desc.memberRows` reads argument `i` through for the used run `[lo, hi)`: per row of a batch's
mapped axis (a scan's `xs` one), one piece of gate-0 positions, a member a row. -/
def rowsAt (node : Flock.Program.Node) (i lo hi : Nat) : Array Piece :=
  match node.form, (node.args[i]!).dims with
  | .scan, _ =>
    #[{ tgt := .gate 0, base := (lo : Int), stride := 1, len := hi - lo, rows := node.members,
        rowStride := ((((node.args[i]!).dims.map (·.2)).getD 0 : Nat) : Int) }]
  | .batch, some (d, L) =>
    if L == 0 then #[] else
    (Array.range ((hi + L - 1) / L - lo / L)).filterMap fun q' =>
      let q := lo / L + q'
      let (s, e) := (max lo (q * L), min hi ((q + 1) * L))
      if s < e then some { tgt := .gate 0, base := ((q * d * L + (s - q * L) : Nat) : Int), stride := 1,
                           len := e - s, rows := node.members, rowStride := (L : Int) } else none
  | _, _ => #[]

/-- The argument indices a piece of gate-0 positions names. -/
def idxOf (q : Piece) : List Nat := (pos q).map (·.2.toNat)

theorem mem_idxOf_row {b : Nat} {len rows L : Nat} (t : Nat) :
    t ∈ idxOf { tgt := .gate 0, base := (b : Int), stride := 1, len := len, rows := rows, rowStride := (L : Int) } ↔
      ∃ r < rows, ∃ u < len, t = b + r * L + u := by
  unfold idxOf pos
  simp only [List.mem_map, List.mem_flatMap, List.mem_range]
  constructor
  · rintro ⟨x, ⟨r, hr, u, hu, rfl⟩, rfl⟩
    exact ⟨r, hr, u, hu, by simp only [Int.mul_one]; omega⟩
  · rintro ⟨r, hr, u, hu, rfl⟩
    exact ⟨_, ⟨r, hr, u, hu, rfl⟩, by simp only [Int.mul_one]; omega⟩

theorem rowsAt_nonneg (node : Flock.Program.Node) (i lo hi : Nat) :
    ∀ q ∈ (rowsAt node i lo hi).toList, ∀ x ∈ pos q, 0 ≤ x.2 := by
  intro q hq x hx
  unfold rowsAt at hq
  split at hq
  · simp only [List.mem_singleton] at hq
    subst hq
    unfold pos at hx
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx
    obtain ⟨r, -, u, -, rfl⟩ := hx
    dsimp only
    positivity
  · split at hq
    · simp at hq
    · rw [Array.mem_toList_iff, Array.mem_filterMap] at hq
      obtain ⟨q', -, hq⟩ := hq
      dsimp only at hq
      split at hq
      · injection hq with hq
        subst hq
        unfold pos at hx
        simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx
        obtain ⟨r, -, u, -, rfl⟩ := hx
        dsimp only
        positivity
      · cases hq
  · simp at hq

/-- **The rows `Desc.memberRows` reads for the used run `[lo, hi)` name exactly its members' leaves there**, as
indices `upIdx` into a mapped argument. -/
theorem rowsAt_idx (node : Flock.Program.Node) (i lo hi : Nat) (hpm : perMember node i = true)
    (hL0 : ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 → hi ≤ lo) (t : Nat) :
    t ∈ (rowsAt node i lo hi).toList.flatMap idxOf ↔
      ∃ m < node.members, ∃ j, lo ≤ j ∧ j < hi ∧ t = upIdx node i m j := by
  unfold perMember at hpm
  unfold rowsAt upIdx
  cases hf : node.form with
  | prim => simp [hf] at hpm
  | call => simp [hf] at hpm
  | scan =>
    simp only [hf, Bool.and_eq_true, decide_eq_true_eq] at hpm
    simp only [List.flatMap, List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil,
      List.append_nil, mem_idxOf_row]
    simp only [if_pos (show i ≠ 0 ∧ i ≤ node.nxs from ⟨by omega, hpm.2⟩)]
    constructor
    · rintro ⟨r, hr, u, hu, rfl⟩
      exact ⟨r, hr, lo + u, by omega, by omega, by ring⟩
    · rintro ⟨r, hr, j, h1, h2, rfl⟩
      exact ⟨r, hr, j - lo, by omega, by
        have : lo + (j - lo) = j := by omega
        rw [← this]; ring_nf; omega⟩
  | batch =>
    simp only [hf] at hpm
    rcases hd : (node.args[i]!).dims with _ | ⟨d, L⟩
    · simp [hd] at hpm
    simp only
    by_cases hL : L = 0
    · have := hL0 d L hd hL
      simp only [hL, beq_self_eq_true, ↓reduceIte, List.flatMap, List.map_nil, List.flatten_nil,
        List.not_mem_nil, false_iff, not_exists, not_and]
      intro m _ j h1 h2; omega
    have hLp : 0 < L := Nat.pos_of_ne_zero hL
    simp only [beq_iff_eq, hL, ↓reduceIte, List.mem_flatMap, Array.mem_toList_iff, Array.mem_filterMap,
      Array.mem_range]
    constructor
    · rintro ⟨q, ⟨q', hq', hq⟩, ht⟩
      split at hq
      · rename_i hse
        injection hq with hq
        subst hq
        rw [mem_idxOf_row] at ht
        obtain ⟨r, hr, u, hu, rfl⟩ := ht
        refine ⟨r, hr, max lo ((lo / L + q') * L) + u, by omega, by omega, ?_⟩
        unfold Resolve.memberIdx
        rw [hd]
        dsimp only
        have hq1 : (lo / L + q') * L ≤ max lo ((lo / L + q') * L) + u := by omega
        have hq2 : max lo ((lo / L + q') * L) + u < (lo / L + q' + 1) * L := by
          have := Nat.lt_of_lt_of_le (Nat.lt_of_le_of_lt (Nat.le_refl _) (by omega :
            max lo ((lo / L + q') * L) + u < min hi ((lo / L + q' + 1) * L))) (Nat.min_le_right _ _)
          exact this
        have hdiv : (max lo ((lo / L + q') * L) + u) / L = lo / L + q' := by
          exact Nat.div_eq_of_lt_le hq1 hq2
        have hmod : (max lo ((lo / L + q') * L) + u) % L = max lo ((lo / L + q') * L) + u - (lo / L + q') * L := by
          have := Nat.div_add_mod (max lo ((lo / L + q') * L) + u) L
          rw [hdiv] at this
          rw [Nat.mul_comm] at this
          omega
        rw [hdiv, hmod]
        have e1 : (lo / L + q') * d * L = (lo / L + q') * L * d := by ring
        rw [e1]
        zify [hq1, Nat.le_max_right lo ((lo / L + q') * L)]
        ring
      · cases hq
    · rintro ⟨r, hr, j, h1, h2, rfl⟩
      have hjq := Nat.div_add_mod j L
      have hjm := Nat.mod_lt j hLp
      have hq0 : lo / L ≤ j / L := Nat.div_le_div_right h1
      have hq1 : j / L < (hi + L - 1) / L := by
        rw [Nat.lt_div_iff_mul_lt hLp]
        have := Nat.div_mul_le_self j L
        omega
      have e : lo / L + (j / L - lo / L) = j / L := by omega
      have hs : max lo (j / L * L) ≤ j := by
        have := Nat.div_mul_le_self j L; omega
      have he : j < min hi ((j / L + 1) * L) := by
        rw [Nat.add_mul, Nat.one_mul]
        have : j < j / L * L + L := by rw [Nat.mul_comm]; omega
        omega
      refine ⟨{ tgt := .gate 0,
                 base := ((j / L * d * L + (max lo (j / L * L) - j / L * L) : Nat) : Int), stride := 1,
                 len := min hi ((j / L + 1) * L) - max lo (j / L * L), rows := node.members,
                 rowStride := (L : Int) }, ⟨j / L - lo / L, by omega, ?_⟩, ?_⟩
      · simp only [e]
        rw [if_pos (by omega)]
      rw [mem_idxOf_row]
      refine ⟨r, hr, j - max lo (j / L * L), by omega, ?_⟩
      unfold Resolve.memberIdx
      rw [hd]
      dsimp only
      have hjL : j % L = j - j / L * L := by
        have := Nat.div_add_mod j L; rw [Nat.mul_comm] at this; omega
      rw [hjL]
      have h3 : j / L * L ≤ max lo (j / L * L) := Nat.le_max_right _ _
      have e1 : j / L * d * L = j / L * L * d := by ring
      rw [e1]
      zify [hs, h3, Nat.div_mul_le_self j L]
      ring

/-- **What `down` keeps of the piece `throughPS` gives of an argument**: the gates the argument's references at the
piece's indices resolve to, position by position. -/
theorem throughPS_agree (c : Desc.Ctx) (sc : Flock.Program.Scope) (hD : DownSpec c sc Flock.Steps.FUEL) {k i : Nat}
    (hA : ArgOk c k i)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {q bp : Piece} (hq0 : ∀ x ∈ pos q, 0 ≤ x.2)
    (hqc : ∀ t ∈ idxOf q, t < ((c.body.nodes[k]!).args[i]!).seq.count)
    (h : Desc.throughPS (c.arg k i) q = some bp) {segs : Array Seg}
    (hd : Desc.down c Flock.Steps.FUEL bp = .ok segs) :
    Agree c.P sc (segsPos segs) ((idxOf q).map fun t =>
      Flock.Program.resolve c.P sc (((c.body.nodes[k]!).args[i]!).seq.get t)) := by
  set a := (c.body.nodes[k]!).args[i]! with hadef
  obtain ⟨hokA, hposA⟩ := hA
  rw [← hadef] at hposA
  obtain ⟨s, pa, hsegs, hps, hsr, hpr, hbpe⟩ := throughPS_some h
  have hpa : pos pa = (List.range a.seq.count).map fun t => refPos (a.seq.get t) := by
    rw [← hposA, psPos_flat _ hokA (by rw [hsegs]; simpa using hsr)]
    simp [hsegs, hps, posAll]
  rw [pos_eq_map] at hpa
  obtain ⟨hcnt, hpt⟩ := map_range_eq hpa
  have hpaJ : ∀ J < a.seq.count, (pa.tgt, pa.base + (J : Int) * pa.stride) = refPos (a.seq.get J) := by
    intro J hJ
    unfold Desc.Piece.count at hcnt
    have hr1 : pa.rows = 1 := by
      rcases Nat.lt_or_ge pa.rows 1 with h | h
      · rw [show pa.rows = 0 by omega, Nat.zero_mul] at hcnt; omega
      · omega
    have hJl : J < pa.len := by rw [hr1, Nat.one_mul] at hcnt; omega
    rw [← hpt J (by unfold Desc.Piece.count; omega)]
    simp only [ppt, Prod.mk.injEq, true_and, Nat.div_eq_of_lt hJl, Nat.mod_eq_of_lt hJl]
    simp
  have hpb : pos bp = (pos q).map fun y => (pa.tgt, pa.base + y.2 * pa.stride) := by
    rw [hbpe]
    unfold pos
    rw [List.map_flatMap]
    apply flatMap_congr
    intro r _
    rw [List.map_map]
    apply List.map_congr_left
    intro t _
    simp only [Function.comp, Prod.mk.injEq, true_and]
    ring
  have hJ : ∀ y ∈ pos q, (pa.tgt, pa.base + y.2 * pa.stride) = refPos (a.seq.get y.2.toNat) ∧
      y.2.toNat < a.seq.count := by
    intro y hy
    have hc := hqc y.2.toNat (List.mem_map.2 ⟨y, hy, rfl⟩)
    refine ⟨?_, hc⟩
    rw [← hpaJ _ hc, Int.toNat_of_nonneg (hq0 y hy)]
  have hxs : ∀ x ∈ pos bp, ∃ J < a.seq.count, x = refPos (a.seq.get J) := by
    intro x hx
    rw [hpb] at hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    exact ⟨_, (hJ y hy).2, (hJ y hy).1⟩
  have hDS := hD _ segs (fun x hx' => by obtain ⟨J, -, rfl⟩ := hxs x hx'; simp [refPos])
    (fun x hx' k' hk' => by
      obtain ⟨J, hJ', rfl⟩ := hxs x hx'
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR J hJ').2 hp').2
      simp only [refPos]
      exact_mod_cast this) hd
  rw [hpb, List.map_map] at hDS
  unfold idxOf
  rw [List.map_map]
  convert hDS using 1
  apply List.map_congr_left
  intro y hy
  simp only [Function.comp]
  rw [(hJ y hy).1, den_refPos]

/-- A loop each of whose steps on an element of `l` goes on keeping `I`, growing its state (`Le`) and doing what
`R` says of the element, or stops with `D`: ending without `D`, it kept `I`, grew its state, and did `R` of each. -/
theorem forIn_cover {α σ : Type} (f : α → σ → Except String (ForInStep σ)) (D I : σ → Prop) (Le : σ → σ → Prop)
    (R : α → σ → Prop) (hrefl : ∀ s, Le s s) (htrans : ∀ s1 s2 s3, Le s1 s2 → Le s2 s3 → Le s1 s3)
    (hRm : ∀ x s1 s2, R x s1 → Le s1 s2 → R x s2) :
    ∀ (l : List α) (s out : σ), (∀ x ∈ l, ∀ s y, I s → f x s = .ok y →
      (∃ s', y = .yield s' ∧ I s' ∧ Le s s' ∧ R x s') ∨ (∃ s', y = .done s' ∧ D s')) →
      I s → forIn l s f = .ok out → ¬ D out → I out ∧ Le s out ∧ ∀ x ∈ l, R x out
  | [], s, out, _, hs, h, _ => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨hs, hrefl _, fun x hx => by simp at hx⟩
  | x :: l, s, out, hf, hs, h, hD => by
    rw [List.forIn_cons] at h
    rcases hx : f x s with e | y <;> rw [hx] at h
    · exact absurd h (by intro h; cases h)
    rcases hf x List.mem_cons_self s y hs hx with ⟨s', rfl, hs', hle, hr⟩ | ⟨s', rfl, hq⟩
    · obtain ⟨h1, h2, h3⟩ := forIn_cover f D I Le R hrefl htrans hRm l s' out
        (fun x hx => hf x (List.mem_cons_of_mem _ hx)) hs' h hD
      refine ⟨h1, htrans _ _ _ hle h2, fun z hz => ?_⟩
      rcases List.mem_cons.1 hz with rfl | hz
      · exact hRm _ _ _ hr h2
      · exact h3 z hz
    · change Except.ok s' = Except.ok out at h
      injection h with h
      subst h
      exact absurd hq hD

/-- **What `Desc.memberRows` gives**: for each used run, the pieces `throughPS` gives of its rows (`rowsAt`), and
nothing else. -/
theorem memberRows_rows (c : Desc.Ctx) (node : Flock.Program.Node) (k i : Nat) (u : Desc.Ivs) {ps : Array Piece}
    (h : Desc.memberRows c node k i u = .ok (some ps)) :
    (∀ bp ∈ ps.toList, ∃ p ∈ u.toList, ∃ q ∈ (rowsAt node i p.1 p.2).toList,
      Desc.throughPS (c.arg k i) q = some bp) ∧
    (∀ p ∈ u.toList, ∀ q ∈ (rowsAt node i p.1 p.2).toList, ∃ bp ∈ ps.toList,
      Desc.throughPS (c.arg k i) q = some bp) := by
  let T : Array Piece → Prop := fun out => ∀ bp ∈ out.toList, ∃ p ∈ u.toList,
    ∃ q ∈ (rowsAt node i p.1 p.2).toList, Desc.throughPS (c.arg k i) q = some bp
  let I : Option (Option (Array Piece)) × Array Piece → Prop := fun st => st.1 = none ∧ T st.2
  let D : Option (Option (Array Piece)) × Array Piece → Prop := fun st => st.1 = some none
  let Le : Option (Option (Array Piece)) × Array Piece → Option (Option (Array Piece)) × Array Piece → Prop :=
    fun s1 s2 => ∀ bp ∈ s1.2.toList, bp ∈ s2.2.toList
  let R : Nat × Nat → Option (Option (Array Piece)) × Array Piece → Prop := fun p st =>
    ∀ q ∈ (rowsAt node i p.1 p.2).toList, ∃ bp ∈ st.2.toList, Desc.throughPS (c.arg k i) q = some bp
  have hrefl : ∀ s, Le s s := fun s bp h => h
  have htrans : ∀ s1 s2 s3, Le s1 s2 → Le s2 s3 → Le s1 s3 := fun _ _ _ h1 h2 bp h => h2 bp (h1 bp h)
  have hRm : ∀ x s1 s2, R x s1 → Le s1 s2 → R x s2 := by
    intro x s1 s2 h1 h2 q hq
    obtain ⟨bp, hb, hbp⟩ := h1 q hq
    exact ⟨bp, h2 bp hb, hbp⟩
  have hpush : ∀ (out : Array Piece) bp, (∀ bp' ∈ out.toList, bp' ∈ (out.push bp).toList) ∧ bp ∈ (out.push bp).toList :=
    fun out bp => ⟨fun bp' h => by simp [h], by simp⟩
  unfold Desc.memberRows at h
  extract_lets a out0 at h
  obtain ⟨st, hst, h⟩ := Walk.except_bind_ok h
  rw [← Array.forIn_toList] at hst
  have key := forIn_cover _ D I Le R hrefl htrans hRm u.toList (none, out0) st ?_ ⟨rfl, fun bp hb => by
    simp [out0] at hb⟩ hst
  · obtain ⟨r, out⟩ := st
    by_cases hD : D (r, out)
    · change r = some none at hD
      subst hD
      simp [pure, Except.pure] at h
    obtain ⟨⟨hr, hT⟩, -, hR⟩ := key hD
    change r = none at hr
    subst hr
    simp only [pure, Except.pure, Except.ok.injEq, Option.some.injEq] at h
    subst h
    exact ⟨hT, fun p hp q hq => hR p hp q hq⟩
  intro x hxu s y hs hf
  obtain ⟨lo, hi⟩ := x
  extract_lets r out jp at hf
  dsimp (config := { zeta := false }) only at hf
  have hjp : ∀ y, jp (rowsAt node i lo hi) = .ok y →
      (∃ s', y = .yield s' ∧ I s' ∧ Le s s' ∧ R (lo, hi) s') ∨ (∃ s', y = .done s' ∧ D s') := by
    intro y hy
    simp only [jp] at hy
    obtain ⟨st2, hst2, hy⟩ := Walk.except_bind_ok hy
    rw [← Array.forIn_toList] at hst2
    have key2 := forIn_cover _ D I Le (fun q st => ∃ bp ∈ st.2.toList, Desc.throughPS (c.arg k i) q = some bp)
      hrefl htrans (fun q s1 s2 h1 h2 => by obtain ⟨bp, hb, hbp⟩ := h1; exact ⟨bp, h2 bp hb, hbp⟩)
      (rowsAt node i lo hi).toList (none, out) st2 ?_ ⟨rfl, hs.2⟩ hst2
    · obtain ⟨r2, out2⟩ := st2
      by_cases hD2 : D (r2, out2)
      · change r2 = some none at hD2
        subst hD2
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact .inr ⟨_, hy.symm, rfl⟩
      obtain ⟨⟨hr2, hT2⟩, hle2, hR2⟩ := key2 hD2
      change r2 = none at hr2
      subst hr2
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact .inl ⟨_, hy.symm, ⟨rfl, hT2⟩, hle2, fun q hq => hR2 q hq⟩
    intro q hq s2 y2 hs2 hf2
    split at hf2
    · rename_i bp hbp
      simp only [pure, Except.pure, Except.ok.injEq] at hf2
      refine .inl ⟨_, hf2.symm, ⟨rfl, fun bp' hb' => ?_⟩, (hpush _ bp).1, bp, (hpush _ bp).2, hbp⟩
      rw [Array.toList_push, List.mem_append, List.mem_singleton] at hb'
      rcases hb' with hb' | rfl
      · exact hs2.2 bp' hb'
      · exact ⟨(lo, hi), hxu, q, hq, hbp⟩
    · simp only [pure, Except.pure, Except.ok.injEq] at hf2
      exact .inr ⟨_, hf2.symm, rfl⟩
  split at hf
  · rename_i hform
    obtain ⟨rows, hrw, hf⟩ := Walk.except_bind_ok hf
    simp only [pure, Except.pure, Except.ok.injEq] at hrw
    subst hrw
    refine hjp y ?_
    unfold rowsAt
    simp only [hform]
    convert hf
    show _ = (Option.map (fun x => ((x.2 : Nat) : Int)) (node.args[i]!).dims).getD 0
    cases (node.args[i]!).dims <;> rfl
  · rename_i d L hform hdims
    have hd' : (node.args[i]!).dims = some (d, L) := hdims
    split at hf
    · rename_i hL
      obtain ⟨rows, hrw, hf⟩ := Walk.except_bind_ok hf
      simp only [pure, Except.pure, Except.ok.injEq] at hrw
      subst hrw
      refine hjp y ?_
      unfold rowsAt
      simp only [hform, hd', hL, ↓reduceIte]
      exact hf
    · rename_i hL
      have hf := walkAtMost_bind _ _ _ hf
      obtain ⟨rows, hrw, hf⟩ := Walk.except_bind_ok hf
      simp only [pure, Except.pure, Except.ok.injEq] at hrw
      subst hrw
      refine hjp y ?_
      unfold rowsAt
      simp only [hform, hd', hL, Bool.false_eq_true, ↓reduceIte]
      exact hf
  · rename_i hns hnb
    obtain ⟨rows, hrw, hf⟩ := Walk.except_bind_ok hf
    simp only [pure, Except.pure, Except.ok.injEq] at hrw
    subst hrw
    refine hjp y ?_
    have : rowsAt node i lo hi = #[] := by
      unfold rowsAt
      split
      · rename_i hs; exact absurd hs hns
      · rename_i d L hb hd; exact absurd hd (hnb d L hb)
      · rfl
    rw [this]
    exact hf

/-- **All members' reads of a mapped argument at once** (`Desc.memberRows`) name exactly what the members' reads of
the parameter's used leaves resolve to (`ArgB`). -/
theorem rowsSet (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hD : DownSpec c sc Flock.Steps.FUEL)
    {k i : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hA : ArgOk c k i)
    (hR : ∀ t < (node.args[i]!).seq.count, Resolve.RefOk c.P c.d k ((node.args[i]!).seq.get t))
    (hpm : perMember node i = true) {lv : Nat}
    (hu : ∀ j, Width.inIvs (((c.memo[node.fn]!).used[i]?.getD #[]).toList) j → j < lv)
    (hL0 : ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 → lv = 0)
    (hlo : ∀ m < node.members, ∀ j < lv, (node.form = .scan → i = 0 → m = 0) →
      upIdx node i m j < (node.args[i]!).seq.count)
    (hsucc : ∀ m j, ArgB node ((c.memo[node.fn]!).used[i]?.getD #[]) i m j →
      ∃ a, Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a)
    {ps : Array Piece} (hps : Desc.memberRows c node k i ((c.memo[node.fn]!).used[i]?.getD #[]) = .ok (some ps))
    (hok : ∀ p ∈ ps.toList, ∃ segs, Desc.down c Flock.Steps.FUEL p = .ok segs) (a : Nat) :
    (∃ x ∈ ps.toList.flatMap (downPos c), den c.P sc x = .ok a) ↔
      ∃ m j, ArgB node ((c.memo[node.fn]!).used[i]?.getD #[]) i m j ∧
        Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a := by
  set u := (c.memo[node.fn]!).used[i]?.getD #[] with hudef
  have hg : c.body.nodes[k]! = node := node_get hn
  have hnode : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact hg
  have hnc : node.form = .scan → i ≠ 0 := by
    intro hs h0; subst h0; unfold perMember at hpm; rw [hs] at hpm; simp at hpm
  obtain ⟨hmem, hcovr⟩ := memberRows_rows c node k i u hps
  have hL0' : ∀ p ∈ u.toList, ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 → p.2 ≤ p.1 := by
    intro p hp d L hdl h0
    have := hL0 d L hdl h0
    by_contra hlt
    have := hu p.1 ⟨p, hp, Nat.le_refl _, by omega⟩
    omega
  have hidx : ∀ p ∈ u.toList, ∀ q ∈ (rowsAt node i p.1 p.2).toList, ∀ t ∈ idxOf q,
      ∃ m < node.members, ∃ j, Width.inIvs u.toList j ∧ t = upIdx node i m j := by
    intro p hp q hq t ht
    obtain ⟨m, hm, j, h1, h2, rfl⟩ := (rowsAt_idx node i p.1 p.2 hpm (hL0' p hp) t).1
      (List.mem_flatMap.2 ⟨q, hq, ht⟩)
    exact ⟨m, hm, j, ⟨p, hp, h1, h2⟩, rfl⟩
  have hhg : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    rw [hg]; exact hR
  -- one piece's read
  have one : ∀ p ∈ u.toList, ∀ q ∈ (rowsAt node i p.1 p.2).toList, ∀ bp,
      Desc.throughPS (c.arg k i) q = some bp → (∃ segs, Desc.down c Flock.Steps.FUEL bp = .ok segs) →
      ((∃ x ∈ downPos c bp, den c.P sc x = .ok a) ↔
        ∃ m j, (ArgB node u i m j ∧ upIdx node i m j ∈ idxOf q) ∧
          Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a) := by
    intro p hp q hq bp hbp ⟨segs, hsegs⟩
    have hag := throughPS_agree c sc hD hA hhg (rowsAt_nonneg node i p.1 p.2 q hq) (fun t ht => by
      obtain ⟨m, hm, j, hj, rfl⟩ := hidx p hp q hq t ht
      rw [hg]
      exact hlo m hm j (hu j hj) (fun hs hi => absurd hi (hnc hs))) hbp hsegs
    rw [hg] at hag
    have e : downPos c bp = segsPos segs := by unfold downPos; rw [hsegs]
    rw [e]
    refine branch_iff _ _ _ (upIdx node i) hag (fun t => ⟨fun ht => ?_, ?_⟩)
      (fun m j a hB h => up_idx c.P sc hnode hB.1.2.1 h) (fun m j hB => hsucc m j hB.1) a
    · obtain ⟨m, hm, j, hj, rfl⟩ := hidx p hp q hq t ht
      exact ⟨m, j, ⟨⟨hm, fun hs hi => absurd hi (hnc hs), hj⟩, ht⟩, rfl⟩
    · rintro ⟨m, j, ⟨-, h⟩, rfl⟩; exact h
  constructor
  · rintro ⟨x, hx, hxa⟩
    obtain ⟨bp, hbp, hx⟩ := List.mem_flatMap.1 hx
    obtain ⟨p, hp, q, hq, hq'⟩ := hmem bp hbp
    obtain ⟨m, j, ⟨hB, -⟩, hr⟩ := (one p hp q hq bp hq' (hok bp hbp)).1 ⟨x, hx, hxa⟩
    exact ⟨m, j, hB, hr⟩
  · rintro ⟨m, j, hB, hr⟩
    obtain ⟨p, hp, h1, h2⟩ := hB.2.2
    have ht : upIdx node i m j ∈ (rowsAt node i p.1 p.2).toList.flatMap idxOf :=
      (rowsAt_idx node i p.1 p.2 hpm (hL0' p hp) _).2 ⟨m, hB.1, j, h1, by omega, rfl⟩
    obtain ⟨q, hq, htq⟩ := List.mem_flatMap.1 ht
    obtain ⟨bp, hbp, hq'⟩ := hcovr p hp q hq
    obtain ⟨x, hx, hxa⟩ := (one p hp q hq bp hq' (hok bp hbp)).2 ⟨m, j, ⟨hB, htq⟩, hr⟩
    exact ⟨x, List.mem_flatMap.2 ⟨bp, hbp, hx⟩, hxa⟩

/-- **One argument's read names exactly its members' reads of the parameter** (`argSet`, all members at once by
`rowsSet`). -/
theorem argSet_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hD : DownSpec c sc Flock.Steps.FUEL)
    {k i : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hA : ArgOk c k i)
    (hR : ∀ t < (node.args[i]!).seq.count, Resolve.RefOk c.P c.d k ((node.args[i]!).seq.get t))
    (hm0 : node.members ≠ 0) (hok : ArgReadOk c k node i) {lv : Nat}
    (hlv : ((c.P.defs[node.fn]!).params[i]?.map (·.2.leaves)).getD 0 = lv)
    (hu : ∀ j, Width.inIvs (((c.memo[node.fn]!).used[i]?.getD #[]).toList) j → j < lv)
    (hL0 : ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 → lv = 0)
    (hlo : ∀ m < node.members, ∀ j < lv, (node.form = .scan → i = 0 → m = 0) →
      upIdx node i m j < (node.args[i]!).seq.count)
    (htile : ∀ t < (node.args[i]!).seq.count, ∃ m < node.members, ∃ j < lv, upIdx node i m j = t)
    (hsucc : ∀ m j, ArgB node ((c.memo[node.fn]!).used[i]?.getD #[]) i m j →
      ∃ a, Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a) (a : Nat) :
    (∃ x ∈ argPos c k node i, den c.P sc x = .ok a) ↔
      ∃ m j, ArgB node ((c.memo[node.fn]!).used[i]?.getD #[]) i m j ∧
        Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, i, j⟩ = .ok a :=
  argSet c sc hd hD hn hA hR hm0 hok hlv hu hL0 hlo htile hsucc
    (fun hpm _ hps hok' a => rowsSet c sc hd hD hn hA hR hpm hu hL0 hlo hsucc hps hok' a) a

/-! ## A scan's carries, read as its later iterations read their carry -/

/-- A loop inserting `g t` for each `t` of `l` keeps what it started with and inserts each. -/
theorem insertLoop_has {α : Type} (g : α → Nat) :
    ∀ (l : List α) (s0 s : Std.HashSet Nat),
      forIn (m := Except String) l s0 (fun t s => pure (ForInStep.yield (s.insert (g t)))) = .ok s →
      (∀ x ∈ s0, x ∈ s) ∧ ∀ t ∈ l, g t ∈ s
  | [], s0, s, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact ⟨fun x hx => hx, fun t ht => by simp at ht⟩
  | t :: l, s0, s, h => by
    rw [List.forIn_cons] at h
    obtain ⟨h1, h2⟩ := insertLoop_has g l _ s h
    refine ⟨fun x hx => h1 x (Std.HashSet.mem_insert.2 (.inr hx)), fun t' ht' => ?_⟩
    rcases List.mem_cons.1 ht' with rfl | ht'
    · exact h1 _ (Std.HashSet.mem_insert.2 (.inl (beq_self_eq_true _)))
    · exact h2 t' ht'

/-- **Back to the initial carry**: iteration `m ≥ 1`'s carry leaf `f 0`, the step's summary naming each return
position `f s` carry leaf `f (s + 1)` for `s < m`, is the first iteration's carry leaf `f m`. -/
theorem back_init (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hform : node.form = .scan) (hRet : RetOk c.P c.memo node.fn) (f : Nat → Nat)
    {m a : Nat} (hm : 1 ≤ m)
    (hs : ∀ s < m, (psPos (c.memo[node.fn]!).ret)[f s]? = some (.ref true 0, (f (s + 1) : Int)))
    (hv : Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, f 0⟩ = .ok a) :
    Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, f m⟩ = .ok a := by
  have h1 := carry_back c sc hd hn hform (by omega : m ≠ 0) hv
  have := carry_init c sc hd hn hform hRet f (m := m - 1) (fun s hs' => hs s (by omega)) h1
  rwa [show m - 1 + 1 = m by omega] at this

/-- **Back to where the carry leaves**: iteration `m`'s carry leaf `f 0`, the step's summary naming each return
position `f s` carry leaf `f (s + 1)` for `s < e < m`, is iteration `m - 1 - e`'s return position `f e`. -/
theorem back_exit (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hform : node.form = .scan) (hRet : RetOk c.P c.memo node.fn) (f : Nat → Nat)
    {e m a : Nat} (he : e < m)
    (hs : ∀ s < e, (psPos (c.memo[node.fn]!).ret)[f s]? = some (.ref true 0, (f (s + 1) : Int)))
    (hv : Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, f 0⟩ = .ok a) :
    Flock.Program.resolve c.P (c.P.child sc k (m - 1 - e)) ((c.P.defs[node.fn]!).body.ret.get (f e)) = .ok a := by
  have h1 := carry_back c sc hd hn hform (by omega : m ≠ 0) hv
  exact carry_chain c sc hd hn hform hRet f (m := m - 1) e (by omega) hs h1

/-- A step's return position, read in an iteration, names in that iteration's activation what it resolves to. -/
theorem ret_den (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat} {node : Flock.Program.Node}
    (hn : c.node k = .ok node) (hRet : RetOk c.P c.memo node.fn) {it h a : Nat} {x : Tgt × Int}
    (hx : (psPos (c.memo[node.fn]!).ret)[h]? = some x)
    (hv : Flock.Program.resolve c.P (c.P.child sc k it) ((c.P.defs[node.fn]!).body.ret.get h) = .ok a) :
    den c.P (c.P.child sc k it) x = .ok a := by
  obtain ⟨-, -, -, hag, -⟩ := hRet
  have hd' : (c.P.child sc k it).d = node.fn := by rw [child_eq c sc hd hn it]; rfl
  have hA := hag _ hd'
  have hlt : h < (c.P.defs[node.fn]!).body.ret.count := by
    have := (List.getElem?_eq_some_iff.1 hx).1
    rw [hA.length_eq] at this; simpa using this
  exact agree_get hA hx (by rw [List.getElem?_map, List.getElem?_range hlt]; rfl) a hv

/-- The chain's step: each position before its last (any, around a cycle) returns the carry leaf the next one is. -/
theorem chain_step {P : Prog} {memo : Array Desc.Summary} {f : Nat} (hR : RetOk P memo f) {j : Nat}
    {ch : Desc.Chain} (hok : ChainOk (memo[f]!).ret j ch) {s : Nat} (hs : ch.exit = none ∨ s + 1 < ch.hist.size) :
    (psPos (memo[f]!).ret)[ch.at s]? = some (.ref true 0, (ch.at (s + 1) : Int)) := by
  rcases hs with hx | hs
  · obtain ⟨q, hq, ht, hb⟩ := chainAt_cyc hok hx s
    exact carryPos_of_retAt hR hq ht hb
  · obtain ⟨q, hq, ht, hb⟩ := hok.2.2.1 s hs
    rw [chainAt_lt ch (by omega : s < ch.hist.size), chainAt_lt ch hs]
    exact carryPos_of_retAt hR hq ht hb

/-- A gate a step's return names, read in an iteration, is one of the scan node's. -/
theorem ret_gate_in (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) {k : Nat}
    {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hlaid : (c.P.defs[sc.d]!).body.offsets[k + 1]! =
      (c.P.defs[sc.d]!).body.offsets[k]! + node.members * (c.P.defs[node.fn]!).gates)
    {it : Nat} (hit : it < node.members) {w : Nat} {p : Int} (hp : p.toNat < (c.P.defs[node.fn]!).gates) {a : Nat}
    (h : den c.P (c.P.child sc k it) (.gate w, p) = .ok a) : InNodeAt c.P sc k a := by
  rw [child_eq c sc hd hn it] at h
  simp only [den] at h
  split at h
  · cases h
  · injection h with h
    subst h
    have hle := Nat.mul_le_mul_right (c.P.defs[node.fn]!).gates hit
    rw [Nat.succ_mul] at hle
    unfold InNodeAt
    change _ ≤ sc.off + (c.P.defs[sc.d]!).body.offsets[k]! + it * (c.P.defs[node.fn]!).gates + p.toNat ∧
      sc.off + (c.P.defs[sc.d]!).body.offsets[k]! + it * (c.P.defs[node.fn]!).gates + p.toNat < _
    rw [hlaid]
    omega

/-- Gate `a` is what an iteration of scan node `k` past the first reads at a leaf of its carry `u` names. -/
def CarryRead (P : Prog) (sc : Flock.Program.Scope) (k : Nat) (node : Flock.Program.Node) (u : Desc.Ivs)
    (a : Nat) : Prop :=
  ∃ m, 1 ≤ m ∧ m < node.members ∧ ∃ j, Width.inIvs u.toList j ∧
    Flock.Program.resolve P (P.child sc k m) ⟨true, 0, j⟩ = .ok a

/-- Every gate outside node `k` that a position of `out` names is a carry read (`CarryRead`). -/
def CarrySnd (P : Prog) (sc : Flock.Program.Scope) (k : Nat) (node : Flock.Program.Node) (u : Desc.Ivs)
    (out : Array Seg) : Prop :=
  ∀ x ∈ segsPos out, ∀ v, den P sc x = .ok v → ¬ InNodeAt P sc k v → CarryRead P sc k node u v

/-- Every gate outside node `k` that iterations `1 .. T` read at carry leaf `j` a position of `out` names. -/
def CarryCov (P : Prog) (sc : Flock.Program.Scope) (k T j : Nat) (out : Array Seg) : Prop :=
  ∀ m, 1 ≤ m → m ≤ T → ∀ v, Flock.Program.resolve P (P.child sc k m) ⟨true, 0, j⟩ = .ok v → ¬ InNodeAt P sc k v →
    ∃ x ∈ segsPos out, den P sc x = .ok v

/-- `out'` holds every position of `out`. -/
def SegsLe (out out' : Array Seg) : Prop := ∀ x ∈ segsPos out, x ∈ segsPos out'

theorem segsLe_app (out e : Array Seg) : SegsLe out (out ++ e) := by
  intro x hx; rw [segsPos_append]; exact List.mem_append_left _ hx

theorem segsLe_trans (o1 o2 o3 : Array Seg) (h1 : SegsLe o1 o2) (h2 : SegsLe o2 o3) : SegsLe o1 o3 :=
  fun x hx => h2 x (h1 x hx)

theorem carryCov_mono {P : Prog} {sc : Flock.Program.Scope} {k T j : Nat} {o1 o2 : Array Seg}
    (h : CarryCov P sc k T j o1) (hl : SegsLe o1 o2) : CarryCov P sc k T j o2 := by
  intro m h1 h2 v hv hout
  obtain ⟨x, hx, hxv⟩ := h m h1 h2 v hv hout
  exact ⟨x, hl x hx, hxv⟩

theorem carrySnd_app {P : Prog} {sc : Flock.Program.Scope} {k : Nat} {node : Flock.Program.Node} {u : Desc.Ivs}
    {out e : Array Seg} (h : CarrySnd P sc k node u out)
    (he : ∀ x ∈ segsPos e, ∀ v, den P sc x = .ok v → ¬ InNodeAt P sc k v → CarryRead P sc k node u v) :
    CarrySnd P sc k node u (out ++ e) := by
  intro x hx
  rw [segsPos_append] at hx
  rcases List.mem_append.1 hx with hx | hx
  · exact h x hx
  · exact he x hx

/-- **What `Desc.carriesRead` reads of a scan's carries, outside the node, is exactly what its iterations past the
first read at the leaves of their carry `used` names** (`CarryRead`). -/
theorem carrySet (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d)
    (hD : DownSpec c sc Flock.Steps.FUEL) (hC : CalleeSpec c sc Flock.Steps.FUEL) (hT : ThroughSpec c sc Flock.Steps.FUEL)
    {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .scan) (hn2 : 2 ≤ node.n)
    (hRet : RetOk c.P c.memo node.fn)
    (hlaid : (c.P.defs[sc.d]!).body.offsets[k + 1]! =
      (c.P.defs[sc.d]!).body.offsets[k]! + node.members * (c.P.defs[node.fn]!).gates)
    (hgw : ∀ w p, (Tgt.gate w, p) ∈ psPos (c.memo[node.fn]!).ret → p.toNat < (c.P.defs[node.fn]!).gates)
    (hpa : (c.P.defs[node.fn]!).body.params.size ≤ node.args.size)
    (hA : ∀ i < node.args.size, ArgOk c k i)
    (hR : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count,
      Resolve.RefOk c.P c.d k ((node.args[i]!).seq.get t))
    (hlo : ∀ i, 1 ≤ i → i < (c.P.defs[node.fn]!).body.params.size → ∀ m < node.members,
      ∀ j < ((c.P.defs[node.fn]!).body.params[i]!).2.leaves, upIdx node i m j < (node.args[i]!).seq.count)
    (hu0 : ∀ j, Width.inIvs (((c.memo[node.fn]!).used[0]?.getD #[]).toList) j →
      j < (c.P.defs[node.fn]!).body.ret.count)
    (hsucc : ∀ m j, 1 ≤ m → m < node.members → Width.inIvs (((c.memo[node.fn]!).used[0]?.getD #[]).toList) j →
      ∃ a, Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, j⟩ = .ok a)
    {segs : Array Seg} (h : Desc.carriesRead c k ((c.memo[node.fn]!).used[0]?.getD #[]) = .ok segs) (a : Nat) :
    ((∃ x ∈ segsPos segs, den c.P sc x = .ok a) ∧ ¬ InNodeAt c.P sc k a) ↔
      (CarryRead c.P sc k node ((c.memo[node.fn]!).used[0]?.getD #[]) a ∧ ¬ InNodeAt c.P sc k a) := by
  set u := (c.memo[node.fn]!).used[0]?.getD #[] with hudef
  unfold Desc.carriesRead at h
  obtain ⟨node', hn', h⟩ := Walk.except_bind_ok h
  rw [hn] at hn'
  cases hn'
  extract_lets fuel r T init out0 seen0 at h
  have hnode := node_get hn
  have hmem : node.members = node.n := members_scan hform
  have hT1 : 1 ≤ T := by show 1 ≤ node.n - 1; omega
  have hTm : T + 1 = node.members := by show node.n - 1 + 1 = node.members; omega
  have hR5 := hRet
  obtain ⟨hps, hpos, -, hag, hpp⟩ := hR5
  have hlenR : (psPos r).length = (c.P.defs[node.fn]!).body.ret.count := by
    have := (hag (.top node.fn 0 none) rfl).length_eq
    simpa using this
  let G : Nat → Prop := fun l => 0 < (c.P.defs[node.fn]!).body.params.size ∧
    l < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves
  have hcarry : ∀ h j' : Nat, (psPos r)[h]? = some (.ref true 0, (j' : Int)) → G j' := by
    intro h j' hx
    have := (hpos _ (List.mem_of_getElem? hx)).2.2 0 rfl
    exact ⟨this.1, by have h2 : ((j' : Nat) : Int) < _ := this.2; exact_mod_cast h2⟩
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hfin := forIn_cover _ (fun _ => False) (CarrySnd c.P sc k node u) SegsLe
    (fun (x : Nat × Nat) out => ∀ j, x.1 ≤ j → j < x.2 → CarryCov c.P sc k T j out)
    (fun o x hx => hx) segsLe_trans (fun x o1 o2 h hl j h1 h2 => carryCov_mono (h j h1 h2) hl) _ out0 segs ?_
    (fun x hx => by simp [out0, segsPos] at hx) h' id
  · obtain ⟨hS, -, hCov⟩ := hfin
    constructor
    · rintro ⟨⟨x, hx, hxa⟩, hout⟩
      exact ⟨hS x hx a hxa hout, hout⟩
    · rintro ⟨⟨m, hm1, hm, j, ⟨p, hp, hj1, hj2⟩, hr⟩, hout⟩
      exact ⟨hCov p hp j hj1 hj2 m hm1 (by omega) a hr hout, hout⟩
  intro x hx o y hI hy
  obtain ⟨aa, bb⟩ := x
  dsimp (config := { zeta := false }) only at hy
  obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
  simp only [pure, Except.pure, Except.ok.injEq] at hy
  subst hy
  rw [← Array.forIn_toList] at hv
  have hab : aa < bb → bb ≤ (psPos r).length := fun hlt => by
    rw [hlenR]; have := hu0 (bb - 1) ⟨(aa, bb), hx, by dsimp only; omega, by dsimp only; omega⟩; omega
  have hju : ∀ j, aa ≤ j → j < bb → Width.inIvs u.toList j := fun j h1 h2 => ⟨(aa, bb), hx, h1, h2⟩
  obtain ⟨hcov, -⟩ := carryRuns_cover r hps hpp aa bb hab
  have hrun : ∀ x ∈ (Desc.carryRuns r aa bb).toList, ∀ j, x.1 ≤ j → j < x.2.1 → Width.inIvs u.toList j := by
    intro x hx' j h1 h2
    have : j ∈ (Desc.carryRuns r aa bb).toList.flatMap (fun x => List.range' x.1 (x.2.1 - x.1)) :=
      List.mem_flatMap.2 ⟨x, hx', List.mem_range'_1.2 ⟨h1, by omega⟩⟩
    rw [hcov, List.mem_range'_1] at this
    exact hju j this.1 (by omega)
  have hfin2 := forIn_cover _ (fun _ => False) (CarrySnd c.P sc k node u) SegsLe
    (fun (x : Nat × Nat × Bool) out => ∀ j, x.1 ≤ j → j < x.2.1 → CarryCov c.P sc k T j out)
    (fun o x hx => hx) segsLe_trans (fun x o1 o2 h hl j h1 h2 => carryCov_mono (h j h1 h2) hl) _ o v ?_ hI hv id
  · obtain ⟨hS, hle, hCov⟩ := hfin2
    refine .inl ⟨v, rfl, hS, hle, fun j h1 h2 => ?_⟩
    have : j ∈ List.range' aa (bb - aa) := List.mem_range'_1.2 ⟨h1, by omega⟩
    rw [← hcov] at this
    obtain ⟨x, hx', hjx⟩ := List.mem_flatMap.1 this
    rw [List.mem_range'_1] at hjx
    exact hCov x hx' j hjx.1 (by omega)
  intro x hx' o' y hI' hy
  obtain ⟨lo, hi, idb⟩ := x
  dsimp (config := { zeta := false }) only at hy
  have hrun' : ∀ j, lo ≤ j → j < hi → Width.inIvs u.toList j := fun j h1 h2 => hrun _ hx' j h1 h2
  split at hy
  · rename_i hxid
    obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    have hids := carryRuns_id r hps hpp aa bb _ hx' hxid
    have hag0 := init_agree c sc hn hT (by omega) (fun j h1 h2 => hcarry j j (hids j h1 h2)) hv'
    dsimp only at hag0 hids
    have hback : ∀ m j w, 1 ≤ m → lo ≤ j → j < hi →
        Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, j⟩ = .ok w →
        Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, j⟩ = .ok w :=
      fun m j w hm h1 h2 hw => back_init c sc hd hn hform hRet (fun _ => j) hm (fun _ _ => hids j h1 h2) hw
    refine .inl ⟨_, rfl, carrySnd_app hI' ?_, segsLe_app _ _, fun j h1 h2 m hm1 hmT w hw hout => ?_⟩
    · intro x' hx'' w hw hout
      obtain ⟨val, hval, hxv⟩ := agree_src hag0 hx''
      obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hval
      rw [List.mem_range'_1] at hj
      obtain ⟨w', hw'⟩ := hsucc 1 j le_rfl (by omega) (hrun' j hj.1 (by omega))
      have := hxv w' (hback 1 j w' le_rfl hj.1 (by omega) hw')
      rw [hw] at this
      obtain rfl := Except.ok.inj this
      exact ⟨1, le_rfl, by omega, j, hrun' j hj.1 (by omega), hw'⟩
    · obtain ⟨x', hx'', hxv⟩ := agree_tgt hag0 (List.mem_map.2 ⟨j, List.mem_range'_1.2 ⟨h1, by
        change j < hi at h2; omega⟩, rfl⟩) (hback m j w hm1 h1 h2 hw)
      exact ⟨x', by rw [segsPos_append]; exact List.mem_append_right _ hx'', hxv⟩
  have hy := walkAtMost_bind _ _ _ hy
  obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
  simp only [pure, Except.pure, Except.ok.injEq] at hy
  subst hy
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hv
  have hfin3 := forIn_cover _ (fun _ => False) (CarrySnd c.P sc k node u) SegsLe
    (fun (j : Nat) out => CarryCov c.P sc k T j out)
    (fun o x hx => hx) segsLe_trans (fun x o1 o2 h hl => carryCov_mono h hl) _ o' v ?_ hI' hv id
  · obtain ⟨hS, hle, hCov⟩ := hfin3
    exact .inl ⟨v, rfl, hS, hle, fun j h1 h2 => hCov j (List.mem_range'_1.2 ⟨h1, by
      change j < hi at h2; simp only [Std.Legacy.Range.size]; omega⟩)⟩
  intro j hj o'' y hI'' hy
  simp only [Std.Legacy.Range.size, List.mem_range'_1] at hj
  have hju' := hrun' j hj.1 (by omega)
  obtain ⟨ch, hch, hy⟩ := Walk.except_bind_ok hy
  have hok := chainOf_ok _ j ch hch
  extract_lets e upto jp at hy
  have he : e = ch.hist.size - 1 := rfl
  have hupto : upto = if ch.exit.isSome = true then min T e else T := rfl
  have hsz := hok.1
  obtain ⟨s1, hs1, hy⟩ := Walk.except_bind_ok hy
  dsimp only at hy
  have hat0 : ch.at 0 = j := by rw [chainAt_lt ch hsz]; exact hok.2.1
  have hstep : ∀ s, (ch.exit = none ∨ s + 1 < ch.hist.size) →
      (psPos r)[ch.at s]? = some (.ref true 0, (ch.at (s + 1) : Int)) := fun s hs => chain_step hRet hok hs
  have hup : ∀ t, t ≤ upto → t ≤ T ∧ (ch.exit = none ∨ t ≤ e) := by
    intro t ht
    rw [hupto] at ht
    rcases hx : ch.exit with _ | q
    · rw [hx] at ht; simp only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at ht; exact ⟨ht, .inl rfl⟩
    · rw [hx] at ht; simp only [Option.isSome_some, ↓reduceIte] at ht; exact ⟨by omega, .inr (by omega)⟩
  have hinit : ∀ t, 1 ≤ t → t ≤ upto → ∀ w,
      Flock.Program.resolve c.P (c.P.child sc k t) ⟨true, 0, j⟩ = .ok w ↔
        Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, ch.at t⟩ = .ok w := by
    intro t h1 h2 w
    obtain ⟨hT', hor⟩ := hup t h2
    have fwd : ∀ w, Flock.Program.resolve c.P (c.P.child sc k t) ⟨true, 0, j⟩ = .ok w →
        Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, ch.at t⟩ = .ok w := fun w hw =>
      back_init c sc hd hn hform hRet ch.at h1 (fun s hs => hstep s (hor.imp id (fun h => by omega)))
        (by rw [hat0]; exact hw)
    refine ⟨fwd w, fun hw => ?_⟩
    obtain ⟨w', hw'⟩ := hsucc t j h1 (by omega) hju'
    have := fwd w' hw'
    rw [hw] at this
    obtain rfl := Except.ok.inj this
    exact hw'
  have hGt : ∀ t, 1 ≤ t → t ≤ upto → G (ch.at t) := by
    intro t h1 h2
    obtain ⟨-, hor⟩ := hup t h2
    have := hstep (t - 1) (hor.imp id (fun h => by omega))
    rw [show t - 1 + 1 = t by omega] at this
    exact hcarry _ _ this
  let G' : Nat → Prop := fun l => ∃ t, 1 ≤ t ∧ t ≤ upto ∧ ch.at t = l
  have hjp : ∀ sn y, (∀ l ∈ sn, G' l) → (∀ t, 1 ≤ t → t ≤ upto → ch.at t ∈ sn) → jp () sn = .ok y →
      (∃ s', y = .yield s' ∧ CarrySnd c.P sc k node u s' ∧ SegsLe o'' s' ∧ CarryCov c.P sc k T j s') ∨
        (∃ s', y = .done s' ∧ False) := by
    intro sn y hG hcs hy
    simp only [jp] at hy
    obtain ⟨v1, hv1, hy⟩ := Walk.except_bind_ok hy
    rw [← Array.forIn_toList] at hv1
    have hfin4 := forIn_cover _ (fun _ => False) (CarrySnd c.P sc k node u) SegsLe
      (fun (l : Nat) out => ∀ w, Flock.Program.resolve c.P (c.P.child sc k 0) ⟨true, 0, l⟩ = .ok w →
        ∃ x ∈ segsPos out, den c.P sc x = .ok w)
      (fun o x hx => hx) segsLe_trans
      (fun l o1 o2 h hl w hw => by obtain ⟨x, hx, hxw⟩ := h w hw; exact ⟨x, hl x hx, hxw⟩) _ o'' v1 ?_ hI'' hv1 id
    swap
    · intro l hl o3 y3 hI3 hy3
      obtain ⟨v', hv', hy3⟩ := Walk.except_bind_ok hy3
      simp only [pure, Except.pure, Except.ok.injEq] at hy3
      subst hy3
      obtain ⟨t, ht1, ht2, rfl⟩ :=
        hG _ (Std.HashSet.mem_toArray.1 (Array.mem_toList_iff.1 (qsort_mem (fun x => x) _ hl)))
      have hag1 := init_agree c sc hn hT (by omega)
        (fun j' h1 h2 => by rw [show j' = ch.at t by omega]; exact hGt t ht1 ht2) hv'
      have htT := (hup t ht2).1
      refine .inl ⟨_, rfl, carrySnd_app hI3 ?_, segsLe_app _ _, fun w hw => ?_⟩
      · intro x' hx' w hw hout
        obtain ⟨val, hval, hxv⟩ := agree_src hag1 hx'
        obtain ⟨j'', hj'', rfl⟩ := List.mem_map.1 hval
        rw [List.mem_range'_1] at hj''
        rw [show j'' = ch.at t by omega] at hxv
        obtain ⟨w', hw'⟩ := hsucc t j ht1 (by omega) hju'
        have := hxv w' ((hinit t ht1 ht2 w').1 hw')
        rw [hw] at this
        obtain rfl := Except.ok.inj this
        exact ⟨t, ht1, by omega, j, hju', hw'⟩
      · obtain ⟨x', hx', hxv⟩ := agree_tgt hag1
          (List.mem_map.2 ⟨ch.at t, List.mem_range'_1.2 ⟨le_rfl, by omega⟩, rfl⟩) hw
        exact ⟨x', by rw [segsPos_append]; exact List.mem_append_right _ hx', hxv⟩
    obtain ⟨hS4, hle4, hR4⟩ := hfin4
    have hcovI : ∀ t, 1 ≤ t → t ≤ upto → ∀ w,
        Flock.Program.resolve c.P (c.P.child sc k t) ⟨true, 0, j⟩ = .ok w → ∃ x ∈ segsPos v1, den c.P sc x = .ok w :=
      fun t h1 h2 w hw => hR4 (ch.at t)
        ((Properties.RemoteAudit.qsort_toList_perm _ _).mem_iff.2
          (Array.mem_toList_iff.2 (Std.HashSet.mem_toArray.2 (hcs t h1 h2)))) w ((hinit t h1 h2 w).1 hw)
    have hfinal : ∀ s', y = .yield s' → CarrySnd c.P sc k node u s' → SegsLe v1 s' →
        (∀ m, e < m → m ≤ T → ch.exit ≠ none → ∀ w,
          Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, j⟩ = .ok w → ¬ InNodeAt c.P sc k w →
            ∃ x ∈ segsPos s', den c.P sc x = .ok w) →
        (∃ s', y = .yield s' ∧ CarrySnd c.P sc k node u s' ∧ SegsLe o'' s' ∧ CarryCov c.P sc k T j s') ∨
          (∃ s', y = .done s' ∧ False) := by
      intro s' hys hS hle hX
      refine .inl ⟨s', hys, hS, segsLe_trans _ _ _ hle4 hle, fun m h1 h2 w hw hout => ?_⟩
      by_cases hmu : m ≤ upto
      · obtain ⟨x, hx, hxw⟩ := hcovI m h1 hmu w hw
        exact ⟨x, hle x hx, hxw⟩
      · rw [hupto] at hmu
        rcases hx : ch.exit with _ | q
        · rw [hx] at hmu; simp only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at hmu; omega
        · rw [hx] at hmu
          simp only [Option.isSome_some, ↓reduceIte] at hmu
          exact hX m (by omega) h2 (by rw [hx]; simp) w hw hout
    have hnode' : (c.P.defs[sc.d]!).body.nodes[k]! = node := by rw [hd]; exact hnode
    split at hy
    · rename_i q hq0
      split at hy
      · rename_i heT
        have hqr := hok.2.2.2.1 q hq0
        have hhe : ch.at e = ch.hist[e]! := chainAt_lt ch (by omega)
        have hxq : (psPos r)[ch.hist[e]!]? = some (q.tgt, q.base) := retAt_pos _ hps hpp hqr.1
        have hlt : ch.hist[e]! < (c.P.defs[node.fn]!).body.ret.count := by
          rw [← hlenR]; exact (List.getElem?_eq_some_iff.1 hxq).1
        have hb0 : 0 ≤ q.base := (hpos _ (List.mem_of_getElem? hxq)).1
        have hE : ∀ it, it < T - e → ∀ w,
            Flock.Program.resolve c.P (c.P.child sc k (it + 1 + e)) ⟨true, 0, j⟩ = .ok w ↔
              Flock.Program.resolve c.P (c.P.child sc k it) ((c.P.defs[node.fn]!).body.ret.get ch.hist[e]!) =
                .ok w := by
          intro it hit w
          have fwd : ∀ w, Flock.Program.resolve c.P (c.P.child sc k (it + 1 + e)) ⟨true, 0, j⟩ = .ok w →
              Flock.Program.resolve c.P (c.P.child sc k it) ((c.P.defs[node.fn]!).body.ret.get ch.hist[e]!) =
                .ok w := by
            intro w hw
            have := back_exit c sc hd hn hform hRet ch.at (e := e) (m := it + 1 + e) (by omega)
              (fun s hs => hstep s (.inr (by omega))) (by rw [hat0]; exact hw)
            rwa [show it + 1 + e - 1 - e = it by omega, hhe] at this
          refine ⟨fwd w, fun hw => ?_⟩
          obtain ⟨w', hw'⟩ := hsucc (it + 1 + e) j (by omega) (by omega) hju'
          have := fwd w' hw'
          rw [hw] at this
          obtain rfl := Except.ok.inj this
          exact hw'
        have hEd : ∀ it w, Flock.Program.resolve c.P (c.P.child sc k it)
            ((c.P.defs[node.fn]!).body.ret.get ch.hist[e]!) = .ok w →
              den c.P (c.P.child sc k it) (q.tgt, q.base) = .ok w :=
          fun it w hw => ret_den c sc hd hn hRet hxq hw
        have hcE : ∀ m, e < m → m ≤ T → ∀ w, Flock.Program.resolve c.P (c.P.child sc k m) ⟨true, 0, j⟩ = .ok w →
            Flock.Program.resolve c.P (c.P.child sc k (m - 1 - e))
              ((c.P.defs[node.fn]!).body.ret.get ch.hist[e]!) = .ok w ∧ m - 1 - e < T - e :=
          fun m h1 h2 w hw => ⟨(hE (m - 1 - e) (by omega) w).1 (by rwa [show m - 1 - e + 1 + e = m by omega]),
            by omega⟩
        split at hy
        · rename_i w0 hqt
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine hfinal v1 hy.symm hS4 (fun x hx => hx) (fun m h1 h2 _ w hw hout => absurd ?_ hout)
          obtain ⟨hE', hit⟩ := hcE m h1 h2 w hw
          have := hEd _ w hE'
          rw [hqt] at this
          exact ret_gate_in c sc hd hn hlaid (by omega)
            (hgw w0 q.base (by rw [← hqt]; exact List.mem_of_getElem? hxq)) this
        · cases hy
        · rename_i i hqt
          have hi0 : i ≠ 0 := fun h => hqr.2 (by rw [hqt, h])
          have hiP := (hpos _ (List.mem_of_getElem? hxq)).2.2 i hqt
          have hia : i < node.args.size := by omega
          obtain ⟨b, hb⟩ : ∃ b : Nat, q.base = (b : Int) := ⟨q.base.toNat, (Int.toNat_of_nonneg hb0).symm⟩
          have hbl : b < ((c.P.defs[node.fn]!).body.params[i]!).2.leaves := by
            have h2 : (b : Int) < (((c.P.defs[node.fn]!).body.params[i]!).2.leaves : Int) := by
              have := hiP.2; rw [hb] at this; exact this
            exact_mod_cast h2
          have hEr : ∀ it w, Flock.Program.resolve c.P (c.P.child sc k it)
              ((c.P.defs[node.fn]!).body.ret.get ch.hist[e]!) = .ok w →
                Flock.Program.resolve c.P sc ((node.args[i]!).seq.get (upIdx node i it b)) = .ok w := by
            intro it w hw
            have := hEd it w hw
            rw [hqt, hb] at this
            simp only [den, Int.toNat_natCast] at this
            rw [if_neg (by omega)] at this
            exact up_idx c.P sc hnode' (fun _ h => absurd h hi0) this
          split at hy
          · rename_i hxs
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hxs
            have hLe : (Option.map (fun x => ((x.2 : Nat) : Int)) (node.args[i]?.bind fun x => x.dims)).getD 0 =
                ((((node.args[i]!).dims.map (·.2)).getD 0 : Nat) : Int) := by
              rw [Array.getElem?_eq_getElem hia, getElem!_pos node.args i hia]
              simp only [Option.bind_some]
              cases (node.args[i]).dims <;> rfl
            rw [hLe, hb] at hy
            have hup' : ∀ it, upIdx node i it b = b + it * (((node.args[i]!).dims.map (·.2)).getD 0) + 0 := by
              intro it
              unfold upIdx
              rw [hform]
              dsimp only
              rw [if_pos ⟨hi0, hxs.2⟩]
              omega
            split at hy
            · rename_i bp hbp
              obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              set L := ((node.args[i]!).dims.map (·.2)).getD 0 with hL
              obtain ⟨pp, hpp⟩ : ∃ pp : Piece, pp = ⟨.gate 0, (b : Int), 1, 1, T - e, (L : Int)⟩ := ⟨_, rfl⟩
              rw [← hpp] at hbp
              have hq0 : ∀ x ∈ pos pp, 0 ≤ x.2 := by
                intro x hx
                rw [hpp] at hx
                unfold pos at hx
                simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx
                obtain ⟨r, -, u', -, rfl⟩ := hx
                dsimp only
                positivity
              have hcov : ∀ t, t ∈ idxOf pp ↔ ∃ it j', (it < T - e ∧ j' = 0) ∧ t = upIdx node i it b := by
                intro t
                rw [hpp, mem_idxOf_row]
                constructor
                · rintro ⟨r', hr', u', hu', rfl⟩
                  exact ⟨r', 0, ⟨hr', rfl⟩, by rw [hup']; omega⟩
                · rintro ⟨it, j', ⟨hit, -⟩, rfl⟩
                  exact ⟨it, hit, 0, by omega, hup' it⟩
              have hqc : ∀ t ∈ idxOf pp, t < ((c.body.nodes[k]!).args[i]!).seq.count := by
                intro t ht
                obtain ⟨it, -, ⟨hit, -⟩, rfl⟩ := (hcov t).1 ht
                rw [hnode]
                exact hlo i (by omega) hiP.1 it (by omega) b hbl
              have hag := throughPS_agree c sc hD (hA i hia) (by rw [hnode]; exact hR i hia) hq0 hqc hbp hv'
              rw [hnode] at hag
              have key := branch_iff (P := c.P) (sc := sc)
                (fun t => Flock.Program.resolve c.P sc ((node.args[i]!).seq.get t))
                (fun it _ => Flock.Program.resolve c.P (c.P.child sc k (it + 1 + e)) ⟨true, 0, j⟩)
                (fun it j' => it < T - e ∧ j' = 0) (fun it _ => upIdx node i it b) hag hcov
                (fun it j' w hB hS => hEr it w ((hE it hB.1 w).1 hS))
                (fun it j' hB => hsucc (it + 1 + e) j (by omega) (by omega) hju')
              refine hfinal (v1 ++ v') hy.symm (carrySnd_app hS4 ?_) (segsLe_app _ _)
                (fun m h1 h2 _ w hw hout => ?_)
              · intro x hx w hw hout
                obtain ⟨it, j', ⟨hit, -⟩, hr⟩ := (key w).1 ⟨x, hx, hw⟩
                exact ⟨it + 1 + e, by omega, by omega, j, hju', hr⟩
              · obtain ⟨x, hx, hxw⟩ := (key w).2
                  ⟨m - 1 - e, 0, ⟨by omega, rfl⟩, by rwa [show m - 1 - e + 1 + e = m by omega]⟩
                exact ⟨x, by rw [segsPos_append]; exact List.mem_append_right _ hx, hxw⟩
            · have hy := walkAtMost_bind _ _ _ hy
              obtain ⟨sB, hsB, hy⟩ := Walk.except_bind_ok hy
              simp only [pure, Except.pure, Except.ok.injEq] at hy
              rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hsB
              have hfin5 := forIn_cover _ (fun _ => False) (CarrySnd c.P sc k node u) SegsLe
                (fun (it : Nat) out => ∀ w, Flock.Program.resolve c.P (c.P.child sc k it)
                  ((c.P.defs[node.fn]!).body.ret.get ch.hist[e]!) = .ok w → ∃ x ∈ segsPos out, den c.P sc x = .ok w)
                (fun o x hx => hx) segsLe_trans
                (fun l o1 o2 h hl w hw => by obtain ⟨x, hx, hxw⟩ := h w hw; exact ⟨x, hl x hx, hxw⟩) _ v1 sB ?_
                hS4 hsB id
              swap
              · intro it hit o5 y5 hI5 hy5
                simp only [Std.Legacy.Range.size, List.mem_range'_1] at hit
                obtain ⟨v', hv', hy5⟩ := Walk.except_bind_ok hy5
                simp only [pure, Except.pure, Except.ok.injEq] at hy5
                subst hy5
                have hag := hC k it _ _ v' (by rw [hnode]; omega) (fun _ => by rw [hnode]; omega) hv'
                rw [hnode] at hag
                refine .inl ⟨_, rfl, carrySnd_app hI5 ?_, segsLe_app _ _, fun w hw => ?_⟩
                · intro x hx w hw hout
                  obtain ⟨val, hval, hxv⟩ := agree_src hag hx
                  obtain ⟨j'', hj'', rfl⟩ := List.mem_map.1 hval
                  rw [List.mem_range'_1] at hj''
                  rw [show j'' = ch.hist[e]! by omega] at hxv
                  obtain ⟨w', hw'⟩ := hsucc (it + 1 + e) j (by omega) (by omega) hju'
                  have := hxv w' ((hE it (by omega) w').1 hw')
                  rw [hw] at this
                  obtain rfl := Except.ok.inj this
                  exact ⟨it + 1 + e, by omega, by omega, j, hju', hw'⟩
                · obtain ⟨x, hx, hxw⟩ := agree_tgt hag
                    (List.mem_map.2 ⟨ch.hist[e]!, List.mem_range'_1.2 ⟨le_rfl, by omega⟩, rfl⟩) hw
                  exact ⟨x, by rw [segsPos_append]; exact List.mem_append_right _ hx, hxw⟩
              obtain ⟨hS5, hle5, hR5⟩ := hfin5
              refine hfinal sB hy.symm hS5 hle5 (fun m h1 h2 _ w hw hout => ?_)
              obtain ⟨hE', hit⟩ := hcE m h1 h2 w hw
              exact hR5 (m - 1 - e) (by simp only [Std.Legacy.Range.size, List.mem_range'_1]; omega) w hE'
          · rename_i hxs
            obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
            simp only [pure, Except.pure, Except.ok.injEq] at hy
            have hag := hC k 0 _ _ v' (by rw [hnode]; omega) (fun _ => by rw [hnode]; omega) hv'
            rw [hnode] at hag
            have hsh : ∀ it, upIdx node i it b = b := by
              intro it
              unfold upIdx
              rw [hform]
              simp only [Bool.and_eq_true, decide_eq_true_eq, not_and] at hxs
              dsimp only
              rw [if_neg (fun h => hxs (by omega) h.2)]
            refine hfinal (v1 ++ v') hy.symm (carrySnd_app hS4 ?_) (segsLe_app _ _) (fun m h1 h2 _ w hw hout => ?_)
            · intro x hx w hw hout
              obtain ⟨val, hval, hxv⟩ := agree_src hag hx
              obtain ⟨j'', hj'', rfl⟩ := List.mem_map.1 hval
              rw [List.mem_range'_1] at hj''
              rw [show j'' = ch.hist[e]! by omega] at hxv
              obtain ⟨w', hw'⟩ := hsucc (0 + 1 + e) j (by omega) (by omega) hju'
              have := hxv w' ((hE 0 (by omega) w').1 hw')
              rw [hw] at this
              obtain rfl := Except.ok.inj this
              exact ⟨0 + 1 + e, by omega, by omega, j, hju', hw'⟩
            · obtain ⟨hE', -⟩ := hcE m h1 h2 w hw
              obtain ⟨w', hw'⟩ := hsucc (0 + 1 + e) j (by omega) (by omega) hju'
              have h0 := (hE 0 (by omega) w').1 hw'
              have r1 := hEr _ w hE'
              have r2 := hEr 0 w' h0
              rw [hsh] at r1 r2
              rw [r1] at r2
              obtain rfl := Except.ok.inj r2
              obtain ⟨x, hx, hxw⟩ := agree_tgt hag
                (List.mem_map.2 ⟨ch.hist[e]!, List.mem_range'_1.2 ⟨le_rfl, by omega⟩, rfl⟩) h0
              exact ⟨x, by rw [segsPos_append]; exact List.mem_append_right _ hx, hxw⟩
      · simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact hfinal v1 hy.symm hS4 (fun x hx => hx) (fun m h1 h2 _ => absurd h2 (by omega))
    · rename_i hne
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      refine hfinal v1 hy.symm hS4 (fun x hx => hx) (fun m h1 h2 hx => ?_)
      rcases hq : ch.exit with _ | q
      · exact absurd hq hx
      · exact absurd hq (hne q)
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hs1
  have hG1 : ∀ l ∈ s1, G' l :=
    insertLoop_mem G' (fun t => ch.hist[t]!) _ _ _ (fun x hx => by simp [seen0] at hx) (fun t ht => by
      rw [List.mem_range'_1] at ht
      exact ⟨t, ht.1, by omega, chainAt_lt ch (by omega)⟩) hs1
  have hC1 : ∀ t, 1 ≤ t → t ≤ min upto e → ch.at t ∈ s1 := by
    intro t h1 h2
    rw [chainAt_lt ch (by omega)]
    exact (insertLoop_has (fun t => ch.hist[t]!) _ _ _ hs1).2 t (List.mem_range'_1.2 ⟨h1, by omega⟩)
  split at hy
  · rename_i hue
    have hx : ch.exit = none := by
      rcases hx : ch.exit with _ | q
      · rfl
      · exfalso
        rw [hupto, hx] at hue
        simp only [Option.isSome_some, ↓reduceIte] at hue
        omega
    obtain ⟨hc, -⟩ := hok.2.2.2.2 hx
    split at hy
    · obtain ⟨s2, hs2, hy⟩ := Walk.except_bind_ok hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hs2
      refine hjp s2 y (insertLoop_mem G' (fun t => ch.hist[t]!) _ _ _ hG1 (fun t ht => ?_) hs2) (fun t h1 h2 => ?_) hy
      · rw [List.mem_range'_1] at ht
        by_cases ht0 : t = 0
        · subst ht0
          refine ⟨ch.hist.size, by omega, by omega, ?_⟩
          unfold Desc.Chain.at
          rw [if_neg (by omega), show ch.cyc = 0 by omega, Nat.sub_zero, Nat.mod_self]
        · exact ⟨t, by omega, by omega, chainAt_lt ch (by omega)⟩
      · obtain ⟨hk1, hk2⟩ := insertLoop_has (fun t => ch.hist[t]!) _ _ _ hs2
        by_cases hte : t ≤ e
        · exact hk1 _ (hC1 t h1 (by omega))
        · rw [chainAt_idx]
          obtain ⟨hlt, -⟩ := cyc_idx ch.hist.size ch.cyc t hc
          refine hk2 _ (List.mem_range'_1.2 ⟨?_, by omega⟩)
          unfold cycIdx
          split <;> omega
    · obtain ⟨s2, hs2, hy⟩ := Walk.except_bind_ok hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hs2
      refine hjp s2 y (insertLoop_mem G' (fun t => ch.at t) _ _ _ hG1 (fun t ht => ?_) hs2) (fun t h1 h2 => ?_) hy
      · rw [List.mem_range'_1] at ht
        exact ⟨t, by omega, by omega, rfl⟩
      · obtain ⟨hk1, hk2⟩ := insertLoop_has (fun t => ch.at t) _ _ _ hs2
        by_cases hte : t ≤ e
        · exact hk1 _ (hC1 t h1 (by omega))
        · exact hk2 t (List.mem_range'_1.2 ⟨by omega, by omega⟩)
  · exact hjp s1 y hG1 (fun t h1 h2 => hC1 t h1 (by omega)) hy

/-! ## A composite node's reads, its arguments' and its carries' together -/

/-- **A composite node's positions name outside it exactly what its gates read there** (`compSet_of`), its arguments'
reads by `argSet_of` and its carries' by `carrySet`, its members' reads succeeding as its callee's `UsedExact` says,
wherever its read succeeds and its arguments are shaped as the decoder holds them. -/
theorem compNodeSet (P : Prog) (memo : Array Desc.Summary) {sc : Flock.Program.Scope} {k : Nat} (hL : Laid P sc.d)
    (hk : k < (P.defs[sc.d]!).body.nodes.size) {node : Flock.Program.Node}
    (hn : (ctxOf P memo sc.d).node k = .ok node) (hp : (P.defs[node.fn]!).prim = false)
    (hend : (P.defs[sc.d]!).body.offsets[k + 1]! ≤ (P.defs[sc.d]!).gates) (hloc : Located P sc)
    (hgood : Resolve.Good P sc) (hU : UsedExact P memo node.fn) (hRet : RetOk P memo node.fn)
    (hD : DownSpec (ctxOf P memo sc.d) sc Flock.Steps.FUEL) (hC : CalleeSpec (ctxOf P memo sc.d) sc Flock.Steps.FUEL)
    (hT : ThroughSpec (ctxOf P memo sc.d) sc Flock.Steps.FUEL)
    {rs : Array Seg} (hrs : Desc.nodeReads (ctxOf P memo sc.d) k = .ok rs)
    (hA : ∀ i < node.args.size, ArgOk (ctxOf P memo sc.d) k i)
    (hR : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count,
      Resolve.RefOk P sc.d k ((node.args[i]!).seq.get t))
    (hprm : (P.defs[node.fn]!).params = (P.defs[node.fn]!).body.params)
    (hpa : (P.defs[node.fn]!).body.params.size = node.args.size)
    (hused : ∀ i j, Width.inIvs (((memo[node.fn]!).used[i]?.getD #[]).toList) j →
      i < (P.defs[node.fn]!).body.params.size ∧ j < ((P.defs[node.fn]!).body.params[i]!).2.leaves)
    (hL0 : ∀ i < node.args.size, ∀ d L, (node.args[i]!).dims = some (d, L) → L = 0 →
      ((P.defs[node.fn]!).body.params[i]!).2.leaves = 0)
    (hlo : ∀ i < node.args.size, ∀ m < node.members, ∀ j < ((P.defs[node.fn]!).body.params[i]!).2.leaves,
      (node.form = .scan → i = 0 → m = 0) → upIdx node i m j < (node.args[i]!).seq.count)
    (htile : node.members ≠ 0 → ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
      ∃ j < ((P.defs[node.fn]!).body.params[i]!).2.leaves, upIdx node i m j = t)
    (hgw : ∀ w p, (Tgt.gate w, p) ∈ psPos (memo[node.fn]!).ret → p.toNat < (P.defs[node.fn]!).gates)
    (hu0 : node.form = .scan → ∀ j, Width.inIvs (((memo[node.fn]!).used[0]?.getD #[]).toList) j →
      j < (P.defs[node.fn]!).body.ret.count) (a : Nat) :
    ((∃ x ∈ nodePos (ctxOf P memo sc.d) node k, den P sc x = .ok a) ∧ ¬ InNodeAt P sc k a) ↔
      (NodeReadsAt P sc k a ∧ ¬ InNodeAt P sc k a) := by
  have hg : (P.defs[sc.d]!).body.nodes[k]! = node := node_get hn
  have hlaid : (P.defs[sc.d]!).body.offsets[k + 1]! =
      (P.defs[sc.d]!).body.offsets[k]! + node.members * (P.defs[node.fn]!).gates := by
    have := hL k hk; rw [hg] at this; exact this
  by_cases hm0 : node.members = 0
  · have hp' : ((ctxOf P memo sc.d).P.defs[node.fn]!).prim = false := hp
    unfold nodePos
    simp only [hp', Bool.false_eq_true, ↓reduceIte, hm0, List.not_mem_nil, false_and, exists_false, false_iff,
      not_and]
    rintro ⟨g, ⟨h1, h2⟩, -⟩
    rw [hlaid, hm0, Nat.zero_mul, Nat.add_zero] at h2
    omega
  have hsucc : ∀ m < node.members, ∀ i j, Width.inIvs (((memo[node.fn]!).used[i]?.getD #[]).toList) j →
      ∃ a, Flock.Program.resolve P (P.child sc k m) ⟨true, i, j⟩ = .ok a := by
    intro m hm i j hij
    have hd' : ((ctxOf P memo sc.d).P.child sc k m).d = node.fn := by
      rw [child_eq (ctxOf P memo sc.d) sc rfl hn m]; rfl
    have hp2 : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false := by rw [hg]; exact hp
    have hm2 : m < ((P.defs[sc.d]!).body.nodes[k]!).members := by rw [hg]; exact hm
    obtain ⟨g, -, -, ops, -, a, -, hr⟩ := (hU (P.child sc k m) hd'
      (Resolve.good_child P hgood hk rfl hp2 hm2 rfl) (located_child P hL hk hp2 hm2 hend hloc)).2 i j hij
    exact ⟨a, hr⟩
  obtain ⟨hok, hcr⟩ := nodeReads_comp_ok (ctxOf P memo sc.d) k node hn hp hrs hm0
  have hlv : ∀ i < node.args.size, (((P.defs[node.fn]!).params[i]?.map (·.2.leaves)).getD 0) =
      ((P.defs[node.fn]!).body.params[i]!).2.leaves := by
    intro i hi
    rw [hprm, Array.getElem?_eq_getElem (by omega), getElem!_pos _ i (by omega)]
    rfl
  refine compSet_of P memo hL hk hg hp hend hloc hgood hU (fun i j hij => by have := (hused i j hij).1; omega)
    (fun i hi a => ?_) (fun a => ?_) a
  · exact argSet_of (ctxOf P memo sc.d) sc rfl hD hn (hA i hi) (hR i hi) hm0 (hok i hi) (hlv i hi)
      (fun j hj => (hused i j hj).2) (hL0 i hi) (hlo i hi) (htile hm0 i hi)
      (fun m j hB => hsucc m hB.1 i j hB.2.2) a
  · by_cases hs : node.form = .scan ∧ 2 ≤ node.n
    · have hsc : (node.form == .scan && decide (node.n ≥ 2)) = true := by
        rw [Bool.and_eq_true, hs.1, decide_eq_true_eq]; exact ⟨rfl, hs.2⟩
      obtain ⟨segs, hsegs⟩ := hcr hsc
      have e : carryPos (ctxOf P memo sc.d) k node = segsPos segs := by
        unfold carryPos; rw [if_pos hsc, hsegs]
      rw [e]
      have hpa' : ((ctxOf P memo sc.d).P.defs[node.fn]!).body.params.size = node.args.size := hpa
      have key := carrySet (ctxOf P memo sc.d) sc rfl hD hC hT hn hs.1 hs.2 hRet hlaid hgw (by omega) hA hR
        (fun i h1 hi m hm j hj => hlo i (by omega) m hm j hj (fun _ h => absurd h (by omega))) (hu0 hs.1)
        (fun m j _ hm hj => hsucc m hm 0 j hj) hsegs a
      refine key.trans ?_
      unfold CarryRead
      constructor
      · rintro ⟨⟨m, h1, h2, j, hj, hr⟩, hout⟩; exact ⟨⟨m, hs.1, h1, h2, j, hj, hr⟩, hout⟩
      · rintro ⟨⟨m, -, h1, h2, j, hj, hr⟩, hout⟩; exact ⟨⟨m, h1, h2, j, hj, hr⟩, hout⟩
    · have e : carryPos (ctxOf P memo sc.d) k node = [] := by
        have hsc : ¬ (node.form == .scan && decide (node.n ≥ 2)) = true := fun h => by
          simp only [Bool.and_eq_true, decide_eq_true_eq] at h; exact hs ⟨form_eq_scan h.1, h.2⟩
        unfold carryPos; rw [if_neg hsc]
      rw [e]
      simp only [List.not_mem_nil, false_and, exists_false, false_iff, not_and]
      rintro ⟨m, hf, h1, h2, -⟩ -
      have := members_scan hf
      exact hs ⟨hf, by omega⟩

/-! ## `summarize`'s parameter reads -/

/-- One piece of `summarize`'s parameter reads: a piece naming parameter `i`'s leaves joins `i`'s. -/
def pushRef (q : Piece) (acc : Array (Array Piece)) : Array (Array Piece) :=
  match q.tgt with
  | .ref true i => acc.modify i (·.push q)
  | _ => acc

theorem forIn_pure_fold {α σ : Type} (g : α → σ → σ) (f : α → σ → Except String (ForInStep σ))
    (hf : ∀ x s, f x s = .ok (.yield (g x s))) :
    ∀ (l : List α) (s out : σ), forIn l s f = .ok out → out = l.foldl (fun s x => g x s) s
  | [], s, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact h.symm
  | x :: l, s, out, h => by
    rw [List.forIn_cons, hf] at h
    exact forIn_pure_fold g f hf l _ out h

theorem pushRef_get (q : Piece) (acc : Array (Array Piece)) (i : Nat) :
    (pushRef q acc).size = acc.size ∧ ∀ q', q' ∈ ((pushRef q acc)[i]!).toList ↔
      q' ∈ (acc[i]!).toList ∨ (i < acc.size ∧ q' = q ∧ q.tgt = .ref true i) := by
  unfold pushRef
  split
  · rename_i i0 hq
    refine ⟨Array.size_modify .., fun q' => ?_⟩
    by_cases hi : i < acc.size
    · rw [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i hi, Array.getElem_modify]
      by_cases e : i0 = i
      · subst e; simp [hi, hq]
      · have : ¬ Tgt.ref true i0 = Tgt.ref true i := fun h => e (by injection h)
        simp [e, hq, this]
    · rw [getElem!_neg _ i (by simpa using hi), getElem!_neg _ i hi]
      simp [hi]
  · rename_i hq
    refine ⟨rfl, fun q' => ?_⟩
    have : ¬ q.tgt = .ref true i := fun h => hq i h
    simp [this]

theorem pushRef_fold : ∀ (l : List Piece) (acc : Array (Array Piece)),
    (l.foldl (fun a q => pushRef q a) acc).size = acc.size ∧ ∀ i q,
      q ∈ ((l.foldl (fun a q => pushRef q a) acc)[i]!).toList ↔
        q ∈ (acc[i]!).toList ∨ (i < acc.size ∧ q ∈ l ∧ q.tgt = .ref true i)
  | [], acc => ⟨rfl, fun i q => by simp⟩
  | x :: l, acc => by
    obtain ⟨h1, h2⟩ := pushRef_fold l (pushRef x acc)
    have hs := (pushRef_get x acc 0).1
    rw [List.foldl_cons]
    refine ⟨by rw [h1, hs], fun i q => ?_⟩
    rw [h2, (pushRef_get x acc i).2, hs]
    simp only [List.mem_cons]
    constructor
    · rintro ((h | ⟨g1, rfl, g3⟩) | ⟨g1, g2, g3⟩)
      · exact .inl h
      · exact .inr ⟨g1, .inl rfl, g3⟩
      · exact .inr ⟨g1, .inr g2, g3⟩
    · rintro (h | ⟨g1, rfl | g2, g3⟩)
      · exact .inl (.inl h)
      · exact .inl (.inr ⟨g1, rfl, g3⟩)
      · exact .inr ⟨g1, g2, g3⟩

/-- One segment of `summarize`'s: a segment of more than one copy is passed over. -/
def segRefs (acc : Array (Array Piece)) (s : Seg) : Array (Array Piece) :=
  if s.reps > 1 then acc else s.ps.toList.foldl (fun a q => pushRef q a) acc

theorem segRefs_fold : ∀ (l : List Seg) (acc : Array (Array Piece)),
    (l.foldl segRefs acc).size = acc.size ∧ ∀ i q,
      q ∈ ((l.foldl segRefs acc)[i]!).toList ↔
        q ∈ (acc[i]!).toList ∨ (i < acc.size ∧ ∃ s ∈ l, s.reps ≤ 1 ∧ q ∈ s.ps.toList ∧ q.tgt = .ref true i)
  | [], acc => ⟨rfl, fun i q => by simp⟩
  | x :: l, acc => by
    obtain ⟨h1, h2⟩ := segRefs_fold l (segRefs acc x)
    have hx : (segRefs acc x).size = acc.size ∧ ∀ i q, q ∈ ((segRefs acc x)[i]!).toList ↔
        q ∈ (acc[i]!).toList ∨ (i < acc.size ∧ x.reps ≤ 1 ∧ q ∈ x.ps.toList ∧ q.tgt = .ref true i) := by
      unfold segRefs
      split
      · rename_i hr
        exact ⟨rfl, fun i q => ⟨.inl, fun h => h.elim id fun h' => absurd h'.2.1 (by omega)⟩⟩
      · rename_i hr
        obtain ⟨g1, g2⟩ := pushRef_fold x.ps.toList acc
        refine ⟨g1, fun i q => ?_⟩
        rw [g2]
        constructor
        · rintro (h | ⟨h1, h2, h3⟩)
          · exact .inl h
          · exact .inr ⟨h1, by omega, h2, h3⟩
        · rintro (h | ⟨h1, -, h2, h3⟩)
          · exact .inl h
          · exact .inr ⟨h1, h2, h3⟩
    rw [List.foldl_cons]
    refine ⟨by rw [h1, hx.1], fun i q => ?_⟩
    rw [h2, hx.2, hx.1]
    constructor
    · rintro ((h | ⟨g1, g2, g3, g4⟩) | ⟨g1, s, hs, g2, g3, g4⟩)
      · exact .inl h
      · exact .inr ⟨g1, x, List.mem_cons_self, g2, g3, g4⟩
      · exact .inr ⟨g1, s, List.mem_cons_of_mem _ hs, g2, g3, g4⟩
    · rintro (h | ⟨g1, s, hs, g2, g3, g4⟩)
      · exact .inl (.inl h)
      · rcases List.mem_cons.1 hs with rfl | hs
        · exact .inl (.inr ⟨g1, g2, g3, g4⟩)
        · exact .inr ⟨g1, s, hs, g2, g3, g4⟩

theorem forIn_pure_eq {α σ : Type} (g : α → σ → σ) (f : α → σ → Except String (ForInStep σ))
    (hf : ∀ x s, f x s = .ok (.yield (g x s))) :
    ∀ (l : List α) (s : σ), forIn l s f = .ok (l.foldl (fun s x => g x s) s)
  | [], s => rfl
  | x :: l, s => by
    rw [List.forIn_cons, hf]
    exact forIn_pure_eq g f hf l _

/-- **`summarize`'s parameter reads**: per parameter, the pieces naming its leaves of the body's node reads of one
copy. -/
theorem summarize_acc (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (h : Desc.summarize P memo d = .ok s) (hp : (P.defs[d]!).prim = false) :
    ∃ acc : Array (Array Piece), s.used = acc.map (fun qs => Desc.Ivs.norm (Desc.ivsOf qs)) ∧
      acc.size = (P.defs[d]!).body.params.size ∧ ∀ i q, q ∈ (acc[i]!).toList ↔
        (i < (P.defs[d]!).body.params.size ∧ ∃ k < (P.defs[d]!).body.nodes.size, ∃ rs,
          Desc.nodeReads (ctxOf P memo d) k = .ok rs ∧
          ∃ g ∈ rs.toList, g.reps ≤ 1 ∧ q ∈ g.ps.toList ∧ q.tgt = .ref true i) := by
  unfold Desc.summarize at h
  extract_lets defn body c ret0 acc0 at h
  rw [if_neg (by simp [defn, hp])] at h
  obtain ⟨ret, -, h⟩ := Walk.except_bind_ok h
  obtain ⟨acc, hacc, h⟩ := Walk.except_bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  refine ⟨acc, rfl, ?_⟩
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hacc
  have := forIn_range_idx (fun t (a : Array (Array Piece)) => a.size = body.params.size ∧ ∀ i q,
      q ∈ (a[i]!).toList ↔ (i < body.params.size ∧ ∃ k < t, ∃ rs, Desc.nodeReads c k = .ok rs ∧
        ∃ g ∈ rs.toList, g.reps ≤ 1 ∧ q ∈ g.ps.toList ∧ q.tgt = .ref true i)) _ body.nodes.size ?_
    body.nodes.size 0 acc0 acc (by simp) ⟨by simp [acc0], fun i q => ?_⟩ (by simpa using hacc)
  · rw [Nat.zero_add] at this
    exact this
  · intro t r y ht ⟨hsz, hI⟩ hy
    obtain ⟨rs, hrs, hy⟩ := Walk.except_bind_ok hy
    obtain ⟨r', hr', hy⟩ := Walk.except_bind_ok hy
    simp only [Except.ok.injEq] at hy
    refine ⟨r', hy.symm, ?_⟩
    rw [← Array.forIn_toList] at hr'
    have e := forIn_pure_fold (fun x a => segRefs a x) _ (fun x a => ?_) _ _ _ hr'
    · obtain ⟨g1, g2⟩ := segRefs_fold rs.toList r
      rw [e]
      refine ⟨by rw [g1, hsz], fun i q => ?_⟩
      rw [g2, hI, hsz]
      constructor
      · rintro (⟨h1, k, hk, rs', h2, h3⟩ | ⟨h1, h2⟩)
        · exact ⟨h1, k, by omega, rs', h2, h3⟩
        · exact ⟨h1, t, by omega, rs, hrs, h2⟩
      · rintro ⟨h1, k, hk, rs', h2, h3⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
        · exact .inl ⟨h1, k, hk, rs', h2, h3⟩
        · rw [hrs] at h2
          obtain rfl := Except.ok.inj h2
          exact .inr ⟨h1, h3⟩
    · unfold segRefs
      split
      · rfl
      · rw [← Array.forIn_toList, forIn_pure_eq (fun q a => pushRef q a) _ (fun q a => ?_)]
        · rfl
        · unfold pushRef
          rcases q.tgt with ⟨_ | _, i⟩ | w <;> rfl
  · simp only [Nat.not_lt_zero, false_and, exists_false, and_false, iff_false]
    intro hq
    by_cases hi : i < acc0.size
    · rw [getElem!_pos acc0 i hi] at hq
      simp [acc0] at hq
    · rw [getElem!_neg acc0 i hi] at hq
      change q ∈ (#[] : Array Piece).toList at hq
      simp at hq

/-- **`summarize`'s `used`, leaf by leaf**: parameter `i`'s leaf `j` is in it exactly when a piece of a node's read of
one copy names it. -/
theorem summarize_used (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (h : Desc.summarize P memo d = .ok s) (hp : (P.defs[d]!).prim = false) (i j : Nat) :
    Width.inIvs ((s.used[i]?.getD #[]).toList) j ↔
      (i < (P.defs[d]!).body.params.size ∧ ∃ k < (P.defs[d]!).body.nodes.size, ∃ rs,
        Desc.nodeReads (ctxOf P memo d) k = .ok rs ∧ ∃ g ∈ rs.toList, g.reps ≤ 1 ∧
          ∃ q ∈ g.ps.toList, q.tgt = .ref true i ∧ ∃ x ∈ q.pis.toList, Width.inPI x j) := by
  obtain ⟨acc, hu, hsz, hmem⟩ := summarize_acc P memo d s h hp
  rw [hu]
  by_cases hi : i < acc.size
  · rw [Array.getElem?_map, Array.getElem?_eq_getElem hi, Option.map_some, Option.getD_some, norm_mem,
      Width.ivsOf_spec, ← getElem!_pos acc i hi]
    simp only [Array.toList_flatMap, List.mem_flatMap]
    constructor
    · rintro ⟨x, ⟨q, hq, hx⟩, hj⟩
      obtain ⟨h1, k, hk, rs, hrs, g, hg, h2, h3, h4⟩ := (hmem i q).1 hq
      exact ⟨h1, k, hk, rs, hrs, g, hg, h2, q, h3, h4, x, hx, hj⟩
    · rintro ⟨h1, k, hk, rs, hrs, g, hg, h2, q, h3, h4, x, hx, hj⟩
      exact ⟨x, ⟨q, (hmem i q).2 ⟨h1, k, hk, rs, hrs, g, hg, h2, h3, h4⟩, hx⟩, hj⟩
  · rw [Array.getElem?_map, Array.getElem?_eq_none (by omega), Option.map_none, Option.getD_none]
    simp only [Width.inIvs, List.not_mem_nil, false_and, exists_false, false_iff, not_and]
    intro h1; omega

/-! ## A node's parameter reads, in `used` -/

/-- `summarize` reads every node of the body. -/
theorem summarize_reads (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (h : Desc.summarize P memo d = .ok s) (hp : (P.defs[d]!).prim = false) :
    ∀ k < (P.defs[d]!).body.nodes.size, ∃ rs, Desc.nodeReads (ctxOf P memo d) k = .ok rs := by
  unfold Desc.summarize at h
  extract_lets defn body c ret0 acc0 at h
  rw [if_neg (by simp [defn, hp])] at h
  obtain ⟨ret, -, h⟩ := Walk.except_bind_ok h
  obtain ⟨acc, hacc, -⟩ := Walk.except_bind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hacc
  have := forIn_range_idx (fun t (_ : Array (Array Piece)) => ∀ k < t, ∃ rs, Desc.nodeReads c k = .ok rs) _
    body.nodes.size ?_ body.nodes.size 0 acc0 acc (by simp) (fun k hk => absurd hk (Nat.not_lt_zero _))
    (by simpa using hacc)
  · rw [Nat.zero_add] at this
    exact this
  · intro t r y ht hI hy
    obtain ⟨rs, hrs, hy⟩ := Walk.except_bind_ok hy
    obtain ⟨r', -, hy⟩ := Walk.except_bind_ok hy
    simp only [Except.ok.injEq] at hy
    refine ⟨r', hy.symm, fun k hk => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
    · exact hI k hk
    · exact ⟨rs, hrs⟩

/-- **A parameter position a node reads is a leaf `summarize`'s `used` names**, the segments the node's read splits
naming positions at or past 0, of a copy at least, and of more than one only of gates or of copies all alike. -/
theorem paramPos_used (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (h : Desc.summarize P memo d = .ok s) (hp : (P.defs[d]!).prim = false) {k : Nat}
    (hk : k < (P.defs[d]!).body.nodes.size) {node : Flock.Program.Node} (hn : (ctxOf P memo d).node k = .ok node)
    (hout : ∀ g, ((∃ i, FromDown (ctxOf P memo d) k i g) ∨ FromRows (ctxOf P memo d) k node g ∨
      FromCarry (ctxOf P memo d) k node g) → (∀ x ∈ segPos g, 0 ≤ x.2) ∧ RepsOk g)
    {i : Nat} {p : Int} (hx : (Tgt.ref true i, p) ∈ nodePos (ctxOf P memo d) node k)
    (hi : i < (P.defs[d]!).body.params.size) :
    Width.inIvs ((s.used[i]?.getD #[]).toList) p.toNat := by
  obtain ⟨rs, hrs⟩ := summarize_reads P memo d s h hp k hk
  obtain ⟨out, hin, hpos, hfrom⟩ : ∃ out : Array Seg,
      (∀ g ∈ (out.flatMap Seg.split).toList, ∀ q, g.ps[0]? = some q → (∀ w, q.tgt ≠ .gate w) → g ∈ rs.toList) ∧
      segsPos out = nodePos (ctxOf P memo d) node k ∧
      ∀ g ∈ out, (∃ i, FromDown (ctxOf P memo d) k i g) ∨ FromRows (ctxOf P memo d) k node g ∨
        FromCarry (ctxOf P memo d) k node g := by
    by_cases hpr : ((ctxOf P memo d).P.defs[node.fn]!).prim = true
    · obtain ⟨out, e, hp', hf⟩ := nodeReads_prim _ k node hn hpr rs hrs
      refine ⟨out, fun g hg _ _ _ => by rw [e]; exact hg, ?_, fun g hg => .inl (hf g hg)⟩
      rw [hp']
      unfold nodePos
      rw [if_pos hpr]
    · have hpr' : ((ctxOf P memo d).P.defs[node.fn]!).prim = false := by simpa using hpr
      have := nodeReads_comp _ k node hn hpr' rs hrs
      by_cases hm : node.members = 0
      · exfalso
        unfold nodePos at hx
        rw [if_neg hpr, if_pos hm] at hx
        cases hx
      · rw [if_neg hm] at this
        obtain ⟨out, e, hp', hf⟩ := this
        refine ⟨out, fun g hg q hq hgate => ?_, ?_, hf⟩
        · rw [e, Array.toList_filter, List.mem_filter]
          refine ⟨hg, ?_⟩
          unfold keep
          rw [hq]
          dsimp only
          cases ht : q.tgt with
          | gate w => exact absurd ht (hgate w)
          | ref b i => rfl
        · rw [hp']
          unfold nodePos
          rw [if_neg hpr, if_neg hm]
  rw [← hpos] at hx
  obtain ⟨g0, hg0, hx0⟩ := List.mem_flatMap.1 hx
  obtain ⟨hnn, hrep⟩ := hout g0 (hfrom g0 (Array.mem_toList_iff.1 hg0))
  obtain ⟨g, hg, hxg⟩ := (mem_segPos_split g0 _).2 hx0
  have hgo : g ∈ (out.flatMap Seg.split).toList := by
    rw [Array.toList_flatMap]
    exact List.mem_flatMap.2 ⟨g0, hg0, hg⟩
  obtain ⟨q, hq, rfl⟩ : ∃ q ∈ g0.ps.toList, g = Seg.one q := by
    revert hg hxg
    unfold Desc.Seg.split
    dsimp only
    split
    · rename_i h1
      intro hg _
      simp only [Array.toList_map, List.mem_map] at hg
      obtain ⟨q, hq, rfl⟩ := hg
      refine ⟨q, ?_, rfl⟩
      unfold Desc.Seg.alike at hq
      split at hq <;> exact hq
    · rename_i h1
      intro hg hxg
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
      subst hg
      exfalso
      have hal : Seg.alike g0 = g0 := by
        unfold Desc.Seg.alike
        split
        · rename_i h2
          exfalso
          apply h1
          unfold Desc.Seg.alike
          rw [if_pos h2]
          rfl
        · rfl
      rw [hal] at h1 hxg
      have hr1 : g0.reps > 1 := by
        have := hrep.1
        simp only [beq_iff_eq] at h1
        omega
      rcases hrep.2 hr1 with hgt | hsh
      · obtain ⟨q, hq, ht⟩ := segPos_tgt hxg
        obtain ⟨w, hw⟩ := hgt q hq
        rw [hw] at ht
        cases ht
      · unfold Desc.Seg.alike at hal
        rw [if_pos (by simp [hr1, hsh])] at hal
        have := congrArg Seg.reps hal
        simp only at this
        omega
  rw [segPos_one] at hxg
  have htgt : q.tgt = .ref true i := (mem_pos_tgt hxg).symm
  have hp0 : 0 ≤ p := hnn _ hx0
  unfold pos at hxg
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq] at hxg
  obtain ⟨r, hr, t, ht, -, hpe⟩ := hxg
  have hnn' : ∀ e ∈ Width.elems q, 0 ≤ e := by
    intro e he
    obtain ⟨r', hr', t', ht', rfl⟩ := Width.mem_elems.1 he
    have hmem : (q.tgt, q.base + (r' : Int) * q.rowStride + (t' : Int) * q.stride) ∈ segPos (Seg.one q) := by
      rw [segPos_one]
      unfold pos
      simp only [List.mem_flatMap, List.mem_range, List.mem_map]
      exact ⟨r', by omega, t', ht', rfl⟩
    exact hnn _ ((mem_segPos_split g0 _).1 ⟨_, hg, hmem⟩)
  refine (summarize_used P memo d s h hp i p.toNat).2 ⟨hi, k, hk, rs, hrs, Seg.one q,
    hin _ hgo q (by simp [Seg.one]) (fun w hw => by rw [hw] at htgt; cases htgt), by simp [Seg.one], q,
    by simp [Seg.one], htgt, (Width.pis_spec q hnn' p.toNat).2 ⟨r, by omega, t, ht, by omega⟩⟩

/-! ## `UsedExact` from the body's nodes -/

theorem locR_finds {P : Prog} {sc : Flock.Program.Scope} {g : Nat} {ops : Array Nat} (h : LocR P sc g ops) :
    ∃ k, FindsAt P sc.d (g - sc.off) k := by
  cases h with
  | prim hf _ _ => exact ⟨_, hf⟩
  | comp hf _ _ => exact ⟨_, hf⟩

theorem findsAt_bounds {P : Prog} {d x k : Nat} (h : FindsAt P d x k) :
    k < (P.defs[d]!).body.nodes.size ∧ (P.defs[d]!).body.offsets[k]! ≤ x ∧ x < (P.defs[d]!).body.offsets[k + 1]! := by
  obtain ⟨hk, hp⟩ := DescAgree.find?_range_some h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
  exact ⟨hk, hp⟩

theorem ctx_node (P : Prog) (memo : Array Desc.Summary) (d : Nat) {k : Nat} (hk : k < (P.defs[d]!).body.nodes.size) :
    (ctxOf P memo d).node k = .ok ((P.defs[d]!).body.nodes[k]!) := by
  unfold Desc.Ctx.node
  change (match (P.defs[d]!).body.nodes[k]? with | some n => pure n | none => throw "program") = _
  rw [Array.getElem?_eq_getElem hk, getElem!_pos _ k hk]
  rfl

theorem laid_le {P : Prog} {d : Nat} (hL : Laid P d) {a b : Nat} (hab : a ≤ b)
    (hb : b ≤ (P.defs[d]!).body.nodes.size) : (P.defs[d]!).body.offsets[a]! ≤ (P.defs[d]!).body.offsets[b]! := by
  induction b, hab using Nat.le_induction with
  | base => exact le_refl _
  | succ b hab ih =>
    rw [hL b (by omega)]
    exact Nat.le_trans (ih (by omega)) (Nat.le_add_right _ _)

/-- **What an activation's gates read outside it is a leaf `used` names** (`UsedExact`'s first part), wherever each
node reads outside itself what its positions name there, and the segments its read splits name positions at or past 0,
no node's output, parameters in range and the body's gates, of a copy at least and of more than one only of gates or
of copies all alike. -/
theorem usedExact_a (P : Prog) (memo : Array Desc.Summary) (s : Desc.Summary) (sc : Flock.Program.Scope)
    (h : Desc.summarize P memo sc.d = .ok s) (hp : (P.defs[sc.d]!).prim = false) (hL : Laid P sc.d)
    (hlast : (P.defs[sc.d]!).body.offsets[(P.defs[sc.d]!).body.nodes.size]! = (P.defs[sc.d]!).gates)
    (hN : ∀ k < (P.defs[sc.d]!).body.nodes.size, ∀ node, (ctxOf P memo sc.d).node k = .ok node → ∀ a,
      (NodeReadsAt P sc k a ∧ ¬ InNodeAt P sc k a) →
        ∃ x ∈ nodePos (ctxOf P memo sc.d) node k, den P sc x = .ok a)
    (hout : ∀ k < (P.defs[sc.d]!).body.nodes.size, ∀ node, (ctxOf P memo sc.d).node k = .ok node → ∀ g,
      ((∃ i, FromDown (ctxOf P memo sc.d) k i g) ∨ FromRows (ctxOf P memo sc.d) k node g ∨
        FromCarry (ctxOf P memo sc.d) k node g) →
      (∀ x ∈ segPos g, 0 ≤ x.2) ∧ RepsOk g)
    (hpos : ∀ k < (P.defs[sc.d]!).body.nodes.size, ∀ node, (ctxOf P memo sc.d).node k = .ok node →
      ∀ x ∈ nodePos (ctxOf P memo sc.d) node k, (∀ k', x.1 ≠ .ref false k') ∧
        (∀ i, x.1 = .ref true i → i < (P.defs[sc.d]!).body.params.size) ∧
        (∀ w, x.1 = .gate w → x.2 < ((P.defs[sc.d]!).gates : Int))) :
    ∀ g ops a, sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates → LocR P sc g ops → a ∈ ops →
      ¬ (sc.off ≤ a ∧ a < sc.off + (P.defs[sc.d]!).gates) →
      ∃ i j, Width.inIvs ((s.used[i]?.getD #[]).toList) j ∧ Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a := by
  intro g ops a hg1 _ hgl ha hout'
  obtain ⟨k, hf⟩ := locR_finds hgl
  obtain ⟨hk, hb1, hb2⟩ := findsAt_bounds hf
  have hend : (P.defs[sc.d]!).body.offsets[k + 1]! ≤ (P.defs[sc.d]!).gates := by
    rw [← hlast]
    exact laid_le hL (by omega) (le_refl _)
  have hn := ctx_node P memo sc.d hk
  obtain ⟨x, hx, hden⟩ := hN k hk _ hn a
    ⟨⟨g, ⟨by omega, by omega⟩, ops, hgl, ha⟩, fun ⟨h1, h2⟩ => hout' ⟨by omega, by omega⟩⟩
  obtain ⟨hnf, hpi, hgw⟩ := hpos k hk _ hn x hx
  obtain ⟨t, p⟩ := x
  cases t with
  | gate w =>
    exfalso
    have hpw := hgw w rfl
    simp only [den] at hden
    split at hden
    · cases hden
    · injection hden with e
      exact hout' ⟨by omega, by omega⟩
  | ref b i =>
    cases b with
    | false => exact absurd rfl (hnf i)
    | true =>
      simp only [den] at hden
      split at hden
      · cases hden
      · rename_i hp0
        exact ⟨i, p.toNat, paramPos_used P memo sc.d s h hp hk hn (hout k hk _ hn) hx (hpi i rfl), hden⟩

/-- The segments a node's read splits: every read is one of them, each whose first piece names no gate is a read, and
they name the node's positions. -/
theorem nodeReads_out (c : Desc.Ctx) (k : Nat) (node : Flock.Program.Node) (hn : c.node k = .ok node) {rs : Array Seg}
    (hrs : Desc.nodeReads c k = .ok rs) :
    ∃ out : Array Seg, (∀ g ∈ rs.toList, g ∈ (out.flatMap Seg.split).toList) ∧
      (∀ g ∈ (out.flatMap Seg.split).toList, ∀ q, g.ps[0]? = some q → (∀ w, q.tgt ≠ .gate w) → g ∈ rs.toList) ∧
      segsPos out = nodePos c node k ∧
      ∀ g ∈ out, (∃ i, FromDown c k i g) ∨ FromRows c k node g ∨ FromCarry c k node g := by
  by_cases hpr : (c.P.defs[node.fn]!).prim = true
  · obtain ⟨out, e, hp', hf⟩ := nodeReads_prim _ k node hn hpr rs hrs
    refine ⟨out, fun g hg => by rw [e] at hg; exact hg, fun g hg _ _ _ => by rw [e]; exact hg, ?_,
      fun g hg => .inl (hf g hg)⟩
    rw [hp']
    unfold nodePos
    rw [if_pos hpr]
  · have hpr' : (c.P.defs[node.fn]!).prim = false := by simpa using hpr
    have := nodeReads_comp _ k node hn hpr' rs hrs
    by_cases hm : node.members = 0
    · rw [if_pos hm] at this
      subst this
      refine ⟨#[], fun g hg => by simp at hg, fun g hg => by simp at hg, ?_, fun g hg => by simp at hg⟩
      unfold nodePos
      rw [if_neg hpr, if_pos hm, segsPos_empty]
    · rw [if_neg hm] at this
      obtain ⟨out, e, hp', hf⟩ := this
      refine ⟨out, fun g hg => ?_, fun g hg q hq hgate => ?_, ?_, hf⟩
      · rw [e, Array.toList_filter, List.mem_filter] at hg
        exact hg.1
      · rw [e, Array.toList_filter, List.mem_filter]
        refine ⟨hg, ?_⟩
        unfold keep
        rw [hq]
        dsimp only
        cases ht : q.tgt with
        | gate w => exact absurd ht (hgate w)
        | ref b i => rfl
      · rw [hp']
        unfold nodePos
        rw [if_neg hpr, if_neg hm]

/-- A read of one copy that `Seg.split` makes is one piece of the segment. -/
theorem split_one {s g : Seg} (hr : s.reps ≠ 0) (hg : g ∈ (Seg.split s).toList) (h1 : g.reps ≤ 1) :
    ∃ q ∈ s.ps.toList, g = Seg.one q := by
  revert hg
  unfold Desc.Seg.split
  dsimp only
  split
  · intro hg
    simp only [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, hq, rfl⟩ := hg
    refine ⟨q, ?_, rfl⟩
    unfold Desc.Seg.alike at hq
    split at hq <;> exact hq
  · rename_i hne
    intro hg
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    subst hg
    exfalso
    have hal : Seg.alike s = s := by
      unfold Desc.Seg.alike
      split
      · rename_i hc
        exfalso
        apply hne
        unfold Desc.Seg.alike
        rw [if_pos hc]
        rfl
      · rfl
    rw [hal] at h1 hne
    simp only [beq_iff_eq] at hne
    omega

/-- **A leaf `used` names is a parameter position a node reads**, the segments its read splits naming positions at or
past 0, of pieces of positive count, of a copy at least. -/
theorem used_nodePos (P : Prog) (memo : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (h : Desc.summarize P memo d = .ok s) (hp : (P.defs[d]!).prim = false)
    (hout : ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (ctxOf P memo d).node k = .ok node → ∀ g,
      ((∃ i, FromDown (ctxOf P memo d) k i g) ∨ FromRows (ctxOf P memo d) k node g ∨
        FromCarry (ctxOf P memo d) k node g) →
      (∀ x ∈ segPos g, 0 ≤ x.2) ∧ g.reps ≠ 0 ∧ ∀ q ∈ g.ps.toList, 0 < q.count)
    {i j : Nat} (hij : Width.inIvs ((s.used[i]?.getD #[]).toList) j) :
    ∃ k < (P.defs[d]!).body.nodes.size, ∃ node, (ctxOf P memo d).node k = .ok node ∧
      (Tgt.ref true i, (j : Int)) ∈ nodePos (ctxOf P memo d) node k := by
  obtain ⟨-, k, hk, rs, hrs, g, hg, hg1, q, hq, htgt, x', hx', hj⟩ := (summarize_used P memo d s h hp i j).1 hij
  have hn := ctx_node P memo d hk
  refine ⟨k, hk, _, hn, ?_⟩
  obtain ⟨out, hsub, -, hpos, hfrom⟩ := nodeReads_out _ k _ hn hrs
  have hgo := hsub g hg
  rw [Array.toList_flatMap, List.mem_flatMap] at hgo
  obtain ⟨s0, hs0, hgs⟩ := hgo
  obtain ⟨hnn, hr0, hcnt⟩ := hout k hk _ hn s0 (hfrom s0 (Array.mem_toList_iff.1 hs0))
  obtain ⟨q0, hq0, rfl⟩ := split_one hr0 hgs hg1
  have e : q = q0 := by simpa [Seg.one] using hq
  subst e
  have hsub' : ∀ y ∈ pos q, y ∈ segPos s0 := fun y hy =>
    (mem_segPos_split s0 y).1 ⟨_, hgs, by rw [segPos_one]; exact hy⟩
  have hrow : 1 ≤ q.rows := by
    have := hcnt q hq0
    unfold Desc.Piece.count at this
    rcases Nat.eq_zero_or_pos q.rows with h0 | h0
    · rw [h0, Nat.zero_mul] at this; omega
    · exact h0
  have hnn' : ∀ e ∈ Width.elems q, 0 ≤ e := by
    intro e he
    obtain ⟨r', hr', t', ht', rfl⟩ := Width.mem_elems.1 he
    refine hnn _ (hsub' (q.tgt, _) ?_)
    unfold pos
    simp only [List.mem_flatMap, List.mem_range, List.mem_map]
    exact ⟨r', by omega, t', ht', rfl⟩
  obtain ⟨r, hr, t, ht, hje⟩ := (Width.pis_spec q hnn' j).1 ⟨x', hx', hj⟩
  rw [← hpos]
  unfold segsPos
  refine List.mem_flatMap.2 ⟨s0, hs0, hsub' _ ?_⟩
  unfold pos
  simp only [List.mem_flatMap, List.mem_range, List.mem_map]
  exact ⟨r, by omega, t, ht, by rw [htgt, hje]⟩

/-- **Each leaf `used` names is read by a gate of the activation** (`UsedExact`'s second part), wherever each node
reads outside itself what its positions name there, its parameter positions resolve and are in range, a parameter's
leaf resolves before the activation, and the segments its read splits name positions at or past 0, of pieces of positive count, of a copy at
least. -/
theorem usedExact_b (P : Prog) (memo : Array Desc.Summary) (s : Desc.Summary) (sc : Flock.Program.Scope)
    (h : Desc.summarize P memo sc.d = .ok s) (hp : (P.defs[sc.d]!).prim = false) (hL : Laid P sc.d)
    (hlast : (P.defs[sc.d]!).body.offsets[(P.defs[sc.d]!).body.nodes.size]! = (P.defs[sc.d]!).gates)
    (hN : ∀ k < (P.defs[sc.d]!).body.nodes.size, ∀ node, (ctxOf P memo sc.d).node k = .ok node → ∀ a,
      ((∃ x ∈ nodePos (ctxOf P memo sc.d) node k, den P sc x = .ok a) ∧ ¬ InNodeAt P sc k a) →
        NodeReadsAt P sc k a)
    (hS : ∀ k < (P.defs[sc.d]!).body.nodes.size, ∀ node, (ctxOf P memo sc.d).node k = .ok node →
      ∀ i (p : Int), (Tgt.ref true i, p) ∈ nodePos (ctxOf P memo sc.d) node k → ∃ a, den P sc (.ref true i, p) = .ok a)
    (hout : ∀ k < (P.defs[sc.d]!).body.nodes.size, ∀ node, (ctxOf P memo sc.d).node k = .ok node → ∀ g,
      ((∃ i, FromDown (ctxOf P memo sc.d) k i g) ∨ FromRows (ctxOf P memo sc.d) k node g ∨
        FromCarry (ctxOf P memo sc.d) k node g) →
      (∀ x ∈ segPos g, 0 ≤ x.2) ∧ g.reps ≠ 0 ∧ ∀ q ∈ g.ps.toList, 0 < q.count)
    (hpi : ∀ k < (P.defs[sc.d]!).body.nodes.size, ∀ node, (ctxOf P memo sc.d).node k = .ok node →
      ∀ i (p : Int), (Tgt.ref true i, p) ∈ nodePos (ctxOf P memo sc.d) node k →
        i < (P.defs[sc.d]!).body.params.size ∧ p < (((P.defs[sc.d]!).body.params[i]!).2.leaves : Int))
    (hlt : ∀ i j a, i < (P.defs[sc.d]!).body.params.size → j < ((P.defs[sc.d]!).body.params[i]!).2.leaves →
      Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a → a < sc.off) :
    ∀ i j, Width.inIvs ((s.used[i]?.getD #[]).toList) j →
      ∃ g, sc.off ≤ g ∧ g < sc.off + (P.defs[sc.d]!).gates ∧ ∃ ops, LocR P sc g ops ∧
        ∃ a ∈ ops, Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a := by
  intro i j hij
  obtain ⟨k, hk, node, hn, hx⟩ := used_nodePos P memo sc.d s h hp hout hij
  obtain ⟨a, ha⟩ := hS k hk node hn i j hx
  have hr : Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a := by
    simp only [den] at ha
    rwa [if_neg (by omega), Int.toNat_natCast] at ha
  have hij' := hpi k hk node hn i j hx
  have hlo := hlt i j a hij'.1 (by have := hij'.2; omega) hr
  obtain ⟨g, ⟨hg1, hg2⟩, ops, hgl, hag⟩ := hN k hk node hn a ⟨⟨_, hx, ha⟩, fun ⟨h1, _⟩ => by omega⟩
  have hend : (P.defs[sc.d]!).body.offsets[k + 1]! ≤ (P.defs[sc.d]!).gates := by
    rw [← hlast]
    exact laid_le hL (by omega) (le_refl _)
  exact ⟨g, by omega, by omega, ops, hgl, a, hag, hr⟩

/-! ## `UsedExact` of every reached Definition, callees first -/

theorem usedExact_congr {P : Prog} {m m' : Array Desc.Summary} {f : Nat} (e : m[f]! = m'[f]!)
    (h : UsedExact P m f) : UsedExact P m' f := by
  unfold UsedExact at h ⊢
  rw [← e]
  exact h

theorem retOk_congr {P : Prog} {m m' : Array Desc.Summary} {f : Nat} (e : m[f]! = m'[f]!)
    (h : RetOk P m f) : RetOk P m' f := by
  unfold RetOk at h ⊢
  rw [← e]
  exact h

/-- **`memoOf` summarizes each reached Definition but the root over the summaries before it.** -/
theorem memoOf_prefix (P : Prog) (memo : Array Desc.Summary) (h : memoOf P = .ok memo) :
    ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root →
      ∃ r : Array Desc.Summary, r.size = f ∧ (∀ f' < f, r[f']! = memo[f']!) ∧
        Desc.summarize P r f = .ok memo[f]! := by
  unfold memoOf at h
  extract_lets seen memo0 at h
  have h' := ok_bind_pure h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h'
  have := forIn_range_idx (fun t (r : Array Desc.Summary) => r.size = t ∧
      ∀ f < t, (Desc.reached P)[f]! = true → f ≠ P.root →
        ∃ r' : Array Desc.Summary, r'.size = f ∧ (∀ f' < f, r'[f']! = r[f']!) ∧ Desc.summarize P r' f = .ok r[f]!)
    _ P.defs.size ?_ P.defs.size 0 memo0 memo
    (by simp) ⟨rfl, fun f hf => absurd hf (Nat.not_lt_zero _)⟩ (by simpa using h')
  · exact fun f hf => this.2 f (by simpa using hf)
  intro t r y htP ⟨hsz, hI⟩ hy
  obtain ⟨v, rfl, hv⟩ : ∃ v, y = .yield (r.push v) ∧
      ((seen[t]! = true ∧ ¬t = P.root) → Desc.summarize P r t = .ok v) := by
    split at hy
    · rcases hs : Desc.summarize P r t with e | v
      · rw [hs] at hy; cases hy
      · rw [hs] at hy
        change Except.ok (ForInStep.yield (r.push v)) = Except.ok y at hy
        injection hy with hy
        exact ⟨v, hy.symm, fun _ => rfl⟩
    · rename_i hc
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨default, hy.symm, fun h => absurd h hc⟩
  have hget : ∀ f' < t, (r.push v)[f']! = r[f']! := fun f' hf' => by
    rw [getElem!_pos (r.push v) f' (by simp; omega), Array.getElem_push_lt (by omega), ← getElem!_pos r f' (by omega)]
  refine ⟨_, rfl, by simp [hsz], fun f hf hrf hfr => ?_⟩
  rcases Nat.lt_succ_iff_lt_or_eq.1 hf with hf | rfl
  · obtain ⟨r', h1, h2, h3⟩ := hI f hf hrf hfr
    exact ⟨r', h1, fun f' hf' => (h2 f' hf').trans (hget f' (by omega)).symm, by rw [hget f hf]; exact h3⟩
  · refine ⟨r, hsz, fun f' hf' => (hget f' (by omega)).symm, ?_⟩
    subst hsz
    rw [getElem!_pos (r.push v) r.size (by simp), Array.getElem_push_eq]
    exact hv ⟨hrf, hfr⟩

/-- **`UsedExact` of every reached composite Definition but the root**, callees first: wherever it holds of each one
summarized over the summaries before it (`memoOf_prefix`) once it holds there of its composite callees. -/
theorem memoOf_usedExact (P : Prog) (memo : Array Desc.Summary) (h : memoOf P = .ok memo)
    (hcf : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn < d)
    (hnr : ∀ d < P.defs.size, (Desc.reached P)[d]! = true →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn ≠ P.root)
    (hStep : ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → (P.defs[f]!).prim = false →
      ∀ r : Array Desc.Summary, r.size = f → (∀ f' < f, r[f']! = memo[f']!) →
      Desc.summarize P r f = .ok memo[f]! →
      (∀ k < (P.defs[f]!).body.nodes.size, (P.defs[((P.defs[f]!).body.nodes[k]!).fn]!).prim = false →
        UsedExact P r ((P.defs[f]!).body.nodes[k]!).fn) →
      UsedExact P memo f) :
    ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → (P.defs[f]!).prim = false →
      UsedExact P memo f := by
  intro f
  induction f using Nat.strong_induction_on with
  | _ f ih =>
    intro hf hrf hfr hp
    obtain ⟨r, hsz, hpre, hs⟩ := memoOf_prefix P memo h f hf hrf hfr
    refine hStep f hf hrf hfr hp r hsz hpre hs fun k hk hpk => ?_
    have hlt := hcf f hf k hk
    exact usedExact_congr (hpre _ hlt).symm (ih _ hlt (by omega) (reached_closed P hcf f hrf hp k hk)
      (hnr f hf hrf k hk) hpk)

/-- A segment a node's read splits is one `down` (or `Desc.memberRows`, or `Desc.carriesRead`) gives. -/
theorem nodePos_seg (P : Prog) (r : Array Desc.Summary) (d : Nat) (s : Desc.Summary)
    (h : Desc.summarize P r d = .ok s) (hp : (P.defs[d]!).prim = false) {k : Nat}
    (hk : k < (P.defs[d]!).body.nodes.size) {node : Flock.Program.Node} (hn : (ctxOf P r d).node k = .ok node)
    {x : Tgt × Int} (hx : x ∈ nodePos (ctxOf P r d) node k) :
    ∃ g, ((∃ i, FromDown (ctxOf P r d) k i g) ∨ FromRows (ctxOf P r d) k node g ∨ FromCarry (ctxOf P r d) k node g) ∧
      x ∈ segPos g := by
  obtain ⟨rs, hrs⟩ := summarize_reads P r d s h hp k hk
  obtain ⟨out, -, -, hpos, hfrom⟩ := nodeReads_out _ k node hn hrs
  rw [← hpos] at hx
  obtain ⟨g, hg, hxg⟩ := List.mem_flatMap.1 hx
  exact ⟨g, hfrom g (Array.mem_toList_iff.1 hg), hxg⟩

/-- **`UsedExact` of a composite Definition from its body's nodes**: summarized over `r`, wherever `NodeSet` holds of
each node over `r`, the segments their reads split are as `down` gives them (`ShapeOk`, `RepsOk`), the gate positions
they name are the body's, and a parameter resolves before every activation resolution reaches. -/
theorem usedExact_of (P : Prog) (r memo : Array Desc.Summary) (f : Nat)
    (h : Desc.summarize P r f = .ok memo[f]!) (hp : (P.defs[f]!).prim = false) (hL : Laid P f)
    (hlast : (P.defs[f]!).body.offsets[(P.defs[f]!).body.nodes.size]! = (P.defs[f]!).gates)
    (hN : ∀ k < (P.defs[f]!).body.nodes.size, NodeSet P r f k)
    (hsh : ∀ k < (P.defs[f]!).body.nodes.size, ∀ node, (ctxOf P r f).node k = .ok node → ∀ g,
      ((∃ i, FromDown (ctxOf P r f) k i g) ∨ FromRows (ctxOf P r f) k node g ∨ FromCarry (ctxOf P r f) k node g) →
      ShapeOk P f g ∧ RepsOk g)
    (hgw : ∀ k < (P.defs[f]!).body.nodes.size, ∀ node, (ctxOf P r f).node k = .ok node →
      ∀ w (p : Int), (Tgt.gate w, p) ∈ nodePos (ctxOf P r f) node k → p < ((P.defs[f]!).gates : Int))
    (hlt : ∀ sc : Flock.Program.Scope, sc.d = f → Resolve.Good P sc →
      ∀ i j a, i < (P.defs[f]!).body.params.size → j < ((P.defs[f]!).body.params[i]!).2.leaves →
        Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a → a < sc.off) :
    UsedExact P memo f := by
  intro sc hd hgood hloc
  subst hd
  have hNS := fun k hk node hn => hN k hk sc rfl hgood hloc node hn
  refine ⟨usedExact_a P r memo[sc.d]! sc h hp hL hlast
    (fun k hk node hn a ha => ((hNS k hk node hn).1 a).2 ha |>.1)
    (fun k hk node hn g hg => ⟨fun x hx => ((hsh k hk node hn g hg).1.2.1 x hx).1, (hsh k hk node hn g hg).2⟩)
    (fun k hk node hn x hx => ?_),
    usedExact_b P r memo[sc.d]! sc h hp hL hlast
    (fun k hk node hn a ha => (((hNS k hk node hn).1 a).1 ha).1)
    (fun k hk node hn => (hNS k hk node hn).2)
    (fun k hk node hn g hg => ⟨fun x hx => ((hsh k hk node hn g hg).1.2.1 x hx).1,
      by have := (hsh k hk node hn g hg).1.2.2.2; omega, (hsh k hk node hn g hg).1.2.2.1⟩)
    (fun k hk node hn i p hx => by
      obtain ⟨g, hg, hxg⟩ := nodePos_seg P r sc.d _ h hp hk hn hx
      exact ((hsh k hk node hn g hg).1.2.1 _ hxg).2.2 i rfl)
    (hlt sc rfl hgood)⟩
  obtain ⟨g, hg, hxg⟩ := nodePos_seg P r sc.d _ h hp hk hn hx
  obtain ⟨-, hnf, hpi⟩ := (hsh k hk node hn g hg).1.2.1 x hxg
  refine ⟨hnf, fun i hi => (hpi i hi).1, fun w hw => ?_⟩
  obtain ⟨t, p⟩ := x
  simp only at hw
  subst hw
  exact hgw k hk node hn w p hx

/-! ## `RootSet` at a root node calling a composite -/

/-- At the root every position a node reads names a gate: `down`'s segments name no node's output and the root has no
parameters (`rootShape`). -/
theorem rootRead_gate (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    {k : Nat} {node : Flock.Program.Node} (hn : (rootCtx P memo).node k = .ok node) {rs : Array Seg}
    (hrs : Desc.nodeReads (rootCtx P memo) k = .ok rs) :
    ∀ x ∈ rootRead P memo node k, ∃ w, x.1 = .gate w := by
  intro x hx
  rw [rootRead_eq] at hx
  obtain ⟨out, -, -, hpos, hfrom⟩ := nodeReads_out _ k node hn hrs
  rw [← hpos] at hx
  obtain ⟨g, hg, hxg⟩ := List.mem_flatMap.1 hx
  obtain ⟨-, hnf, hpi⟩ := (rootShape P memo hm hF hn (hfrom g (Array.mem_toList_iff.1 hg))).2.1 x hxg
  obtain ⟨t, p⟩ := x
  cases t with
  | gate w => exact ⟨w, rfl⟩
  | ref b i =>
    cases b with
    | false => exact absurd rfl (hnf i)
    | true =>
      have := (hpi i rfl).1
      rw [hF.2.2.1] at this
      omega

/-- **`RootSet` at a root node calling a composite, from `NodeSet` at the root's activation**: the root's activation
is one resolution reaches, its gates located (`locR_root`), each with `circuitOf`'s operands. -/
theorem rootSet_comp (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (c : Circuit) (hc : circuitOf P = .ok c) (hl : Desc.rootLaidOut P = true) {k : Nat} (hk : k < nodeCount P)
    {node : Flock.Program.Node} {rs : Array Seg} (hn : (rootCtx P memo).node k = .ok node)
    (hrs : Desc.nodeReads (rootCtx P memo) k = .ok rs) (hN : NodeSet P memo P.root k) :
    ∀ a : Nat, ((∃ w, (Tgt.gate w, (a : Int)) ∈ rootRead P memo node k) ∧ a ∉ nodeGates P k) ↔
      ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! ∧ a ∉ nodeGates P k := by
  obtain ⟨-, -, hK, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  have hend := DescAgree.offs_mono (rootOffs P) (nodeCount P) hmono (k + 1) (nodeCount P) (by omega) le_rfl
  have hgood : Resolve.Good P P.rootScope := ⟨rfl, rfl, rfl⟩
  have hloc : Located P P.rootScope := fun g _ h2 =>
    ⟨_, locR_root hc (g := g) (by change g < 0 + (P.defs[P.root]!).gates at h2; unfold gates; omega)⟩
  have hin : ∀ a, InNodeAt P P.rootScope k a ↔ a ∈ nodeGates P k := by
    intro a
    rw [DescAgree.mem_nodeGates]
    change 0 + (P.defs[P.root]!).body.offsets[k]! ≤ a ∧ a < 0 + (P.defs[P.root]!).body.offsets[k + 1]! ↔ _
    simp only [Nat.zero_add]
    rfl
  have hrd : ∀ a, NodeReadsAt P P.rootScope k a ↔ ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! := by
    intro a
    constructor
    · rintro ⟨g, hg, ops, hl', ha⟩
      have hg' := (hin g).1 hg
      have hgr := (DescAgree.mem_nodeGates P k g).1 hg'
      have := locR_unique hl' (locR_root hc (g := g) (by omega))
      subst this
      exact ⟨g, hg', ha⟩
    · rintro ⟨g, hg, ha⟩
      have hgr := (DescAgree.mem_nodeGates P k g).1 hg
      exact ⟨g, (hin g).2 hg, _, locR_root hc (by omega), ha⟩
  obtain ⟨hiff, -⟩ := hN P.rootScope rfl hgood hloc node (by rw [← rootCtx_eq]; exact hn)
  have hgt := rootRead_gate P memo hm hF hn hrs
  intro a
  have key := hiff a
  rw [← rootCtx_eq, ← rootRead_eq] at key
  constructor
  · rintro ⟨⟨w, hw⟩, hout⟩
    have hden : den P P.rootScope (Tgt.gate w, (a : Int)) = .ok a := by
      simp only [den]
      rw [if_neg (by omega)]
      change Except.ok (0 + (a : Int).toNat) = _
      simp
    obtain ⟨hr, -⟩ := key.1 ⟨⟨_, hw, hden⟩, fun h => hout ((hin a).1 h)⟩
    obtain ⟨g, hg, ha⟩ := (hrd a).1 hr
    exact ⟨g, hg, ha, hout⟩
  · rintro ⟨g, hg, ha, hout⟩
    obtain ⟨⟨x, hx, hxa⟩, -⟩ := key.2 ⟨(hrd a).2 ⟨g, hg, ha⟩, fun h => hout ((hin a).1 h)⟩
    obtain ⟨w, hw⟩ := hgt x hx
    obtain ⟨t, p⟩ := x
    simp only at hw
    subst hw
    have := den_root_gate P hxa
    subst this
    exact ⟨⟨w, hx⟩, hout⟩

/-! ## `RootSet` at a root node calling a primitive -/

/-- **An activation resolution reaches is of a reached composite** (`reached_root`, `reached_closed`). -/
theorem good_reached (P : Prog) (hF : RootFacts P) : ∀ sc, Resolve.Good P sc →
    sc.d < P.defs.size ∧ (Desc.reached P)[sc.d]! = true ∧ (P.defs[sc.d]!).prim = false
  | .top d off st, h => by
    obtain ⟨rfl, -, -⟩ := h
    exact ⟨hF.1, reached_root P hF.1, hF.2.1⟩
  | .child parent d off k m, h => by
    obtain ⟨hpg, hk, hfn, hc, -, -⟩ := h
    obtain ⟨hd, hr, hp⟩ := good_reached P hF parent hpg
    have hcl := reached_closed P hF.2.2.2.2.2.2.1 parent.d hr hp k hk
    have hlt := hF.2.2.2.2.2.2.1 parent.d hd k hk
    rw [hfn] at hcl hlt
    exact ⟨by change d < P.defs.size; omega, hcl, hc⟩

/-- **`RootSet` at a root node calling a primitive**, `rootSet_prim`'s premises from the decoder's facts (`RootFacts`,
`WidthFacts`, `Desc.walkable`) and resolution staying below the node (`Resolve.start_ok`, `Resolve.resolve_lt`), but
that every reference of the node is read by a member unless it is a batch of none (`hCov`). -/
theorem rootSet_prim_of (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) (hwk : Desc.walkable P = true) (c : Circuit) (hc : circuitOf P = .ok c)
    (hl : Desc.rootLaidOut P = true) {k : Nat} (hk : k < nodeCount P)
    {node : Flock.Program.Node} {rs : Array Seg} (hn : (rootCtx P memo).node k = .ok node)
    (hp : (P.defs[node.fn]!).prim = true) (hrs : Desc.nodeReads (rootCtx P memo) k = .ok rs)
    (hgt : ∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w)
    (hCov : ¬ (node.form == .batch && node.n == 0) = true → ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
      ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
        ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
      (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t) :
    ∀ a : Nat, (∃ w, (Tgt.gate w, (a : Int)) ∈ rootRead P memo node k) ↔
      ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! ∧ a ∉ nodeGates P k := by
  obtain ⟨hr, hpr, -, hDec, -, hSO, -, -, hB⟩ := id hF
  have hBr := hB P.root hr (reached_root P hr)
  have hkb : k < (P.defs[P.root]!).body.nodes.size := node_lt hn
  have hg : (P.defs[P.root]!).body.nodes[k]! = node := node_get hn
  have hfn : node.fn < P.defs.size := by have := (hBr.1 k hkb).1; rwa [hg] at this
  have hL : Laid P P.root := ((hWF P.root hr).2 hpr).1
  have hG : (P.defs[node.fn]!).gates = 1 := (hWF node.fn hfn).1 hp
  have hR0 := root_retOk P memo hm hF
  rw [rootCtx_eq] at hR0
  have hD : DownSpec (rootCtx P memo) P.rootScope Flock.Steps.FUEL :=
    (specs_decoded P memo hr hDec hBr hR0 (hSO memo P.root) P.rootScope rfl Flock.Steps.FUEL).1
  have hA : ∀ i < node.args.size, ArgOk (rootCtx P memo) k i := fun i hi =>
    argOk_decoded P memo hr hDec k i hkb (by change i < ((P.defs[P.root]!).body.nodes[k]!).args.size; rw [hg]; exact hi)
  have hR : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count,
      Resolve.RefOk P P.root k ((node.args[i]!).seq.get t) := by
    intro i _ t ht
    have := (hBr.1 k hkb).2.1 i t (by rw [hg]; exact ht)
    rwa [hg] at this
  have hOp : Resolve.OpOk node (P.defs[node.fn]!) := by
    have := (hBr.1 k hkb).2.2.2.2 (by rw [hg]; exact hp)
    rwa [hg] at this
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hr, hpr⟩
  have hB' : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc hsc => by
    obtain ⟨hd, hrd, -⟩ := good_reached P hF sc hsc
    exact hB sc.d hd hrd
  have hLO : ∀ d < P.defs.size, Walk.LaidOut P d := by
    have hwk' := hwk
    unfold Desc.walkable at hwk'
    simp only [Bool.and_eq_true] at hwk'
    exact (Walk.bodiesLaidOut_spec P hwk'.1.2).2.1
  have hLow : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∀ a,
      Flock.Program.resolve P P.rootScope ((node.args[i]!).seq.get t) = .ok a → a < (rootOffs P)[k]! := by
    intro i hi t ht a ha
    have := Resolve.resolve_lt P hroot hB' hLO _ P.rootScope _
      (Resolve.start_ok P hroot hB' hLO P.rootScope ⟨rfl, rfl, rfl⟩ k hkb _ (hR i hi t ht)) ha
    change a < 0 + (P.defs[P.root]!).body.offsets[k]! at this
    change a < (P.defs[P.root]!).body.offsets[k]!
    omega
  exact rootSet_prim P memo c hc hl hL hk hn hp hG hrs hgt hD hA hR hLow hOp hCov

/-! ## `NodeSet` at a node calling a primitive -/

/-- **`NodeSet` at a node calling a primitive**, in any activation: its positions name outside it exactly what its
gates read there, given its arguments' references in range, its operands' indices in range (`hOp`) and every reference
read by a member unless it is a batch of none (`hCov`), and each position it reads resolves, the member's gate being
located (`Located`). -/
theorem nodeSet_prim (P : Prog) (memo : Array Desc.Summary) {d k : Nat} (hL : Laid P d)
    (hkb : k < (P.defs[d]!).body.nodes.size) (hend : (P.defs[d]!).body.offsets[k + 1]! ≤ (P.defs[d]!).gates)
    {node : Flock.Program.Node} {rs : Array Seg} (hn : (ctxOf P memo d).node k = .ok node)
    (hp : (P.defs[node.fn]!).prim = true) (hG : (P.defs[node.fn]!).gates = 1)
    (hrs : Desc.nodeReads (ctxOf P memo d) k = .ok rs)
    (hD : ∀ sc : Flock.Program.Scope, sc.d = d → DownSpec (ctxOf P memo d) sc Flock.Steps.FUEL)
    (hA : ∀ i < node.args.size, ArgOk (ctxOf P memo d) k i)
    (hR : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, Resolve.RefOk P d k ((node.args[i]!).seq.get t))
    (hOp : Resolve.OpOk node (P.defs[node.fn]!))
    (hCov : ¬ (node.form == .batch && node.n == 0) = true → ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
      ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
        ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
      (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t) :
    NodeSet P memo d k := by
  intro sc hd _ hloc node' hn'
  rw [hn] at hn'
  injection hn' with e
  subst e
  subst hd
  have hg : (P.defs[sc.d]!).body.nodes[k]! = node := node_get hn
  have hoff : (P.defs[sc.d]!).body.offsets[k + 1]! = (P.defs[sc.d]!).body.offsets[k]! + node.members := by
    have := hL k hkb
    rw [hg, hG, Nat.mul_one] at this
    exact this
  have hloc' : ∀ m < node.members, ∀ ops, LocR P sc (sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m)) ops ↔
      Flock.Program.operands P sc node m = .ok ops := by
    intro m hm ops
    have := locR_prim P (sc := sc) (k := k) (m := m) hL hkb (by rw [hg]; exact hp) (by rw [hg]; exact hG)
      (by rw [hg]; exact hm) (ops := ops)
    rw [hg] at this
    exact this
  have hpc : ((ctxOf P memo sc.d).P.defs[node.fn]!).prim = true := hp
  by_cases hb : (node.form == .batch && node.n == 0) = true
  · have hm0 : node.members = 0 := by
      simp only [Bool.and_eq_true, beq_iff_eq] at hb
      unfold Flock.Program.Node.members
      rw [hb.1, hb.2]
      simp
    have hnil : nodePos (ctxOf P memo sc.d) node k = [] := by
      unfold nodePos
      rw [if_pos hpc, if_pos hb]
    rw [hnil]
    refine ⟨fun a => ?_, fun i p hx => by simp at hx⟩
    simp only [List.not_mem_nil, false_and, exists_false, false_iff, not_and]
    rintro ⟨g, ⟨h1, h2⟩, -⟩
    rw [hoff, hm0] at h2
    omega
  have hg' : (ctxOf P memo sc.d).body.nodes[k]! = node := node_get hn
  have hok := nodeReads_prim_ok (ctxOf P memo sc.d) k node hn hpc hrs hb
  have hag : ∀ i < node.args.size, Agree P sc
      (readPos (ctxOf P memo sc.d) k i #[(0, ((ctxOf P memo sc.d).arg k i).count)])
      ((List.range (node.args[i]!).seq.count).map fun t =>
        Flock.Program.resolve P sc ((node.args[i]!).seq.get t)) := by
    intro i hi
    have hc0 := argCount_eq (hA i hi)
    have := readPos_agree (ctxOf P memo sc.d) sc (hD sc rfl) (hA i hi) (by rw [hg']; exact hR i hi)
      (spans := #[(0, ((ctxOf P memo sc.d).arg k i).count)])
      (fun x hx _ => by simp only [List.mem_singleton] at hx; subst hx; exact hc0.le) (hok i hi)
    rw [hg'] at this hc0
    have hsp : spansPos #[(0, ((ctxOf P memo sc.d).arg k i).count)] = List.range (node.args[i]!).seq.count := by
      simp [spansPos, List.range_eq_range', hc0]
    rw [hsp] at this
    exact this
  have hread : nodePos (ctxOf P memo sc.d) node k = (List.range node.args.size).flatMap fun i =>
      readPos (ctxOf P memo sc.d) k i #[(0, ((ctxOf P memo sc.d).arg k i).count)] := by
    unfold nodePos
    rw [if_pos hpc, if_neg hb]
  have hmem : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members, ∃ ops,
      LocR P sc (sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m)) ops ∧
      Flock.Program.operands P sc node m = .ok ops ∧ ∃ a,
        Flock.Program.resolve P sc ((node.args[i]!).seq.get t) = .ok a ∧ a ∈ ops := by
    intro i hi t ht
    obtain ⟨m, hm, t', ht'', hidx⟩ := hCov hb i hi t ht
    obtain ⟨ops, hl⟩ := hloc (sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m)) (by omega) (by omega)
    have hops := (hloc' m hm ops).1 hl
    obtain ⟨a', ha'⟩ := operands_ok P _ _ m hops i hi t' ht''
    refine ⟨m, hm, ops, hl, hops, a', by rw [ref_get, hidx] at ha'; exact ha',
      (operands_mem P _ _ m _ hops a').2 ⟨i, hi, t', ht'', ha'⟩⟩
  refine ⟨fun a => ⟨?_, ?_⟩, fun i p hx => ?_⟩
  · rintro ⟨⟨x, hx, hxa⟩, hout⟩
    rw [hread] at hx
    obtain ⟨i, hi, hx⟩ := List.mem_flatMap.1 hx
    have hi' := List.mem_range.1 hi
    obtain ⟨v, hv, hxv⟩ := agree_src (hag i hi') hx
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hv
    obtain ⟨m, hm, ops, hl, -, a', ha', hin⟩ := hmem i hi' t (List.mem_range.1 ht)
    rw [hxv a' ha'] at hxa
    obtain rfl : a' = a := Except.ok.inj hxa
    exact ⟨⟨_, ⟨by omega, by omega⟩, ops, hl, hin⟩, hout⟩
  · rintro ⟨⟨g, ⟨hg1, hg2⟩, ops, hl, ha⟩, hout⟩
    obtain ⟨m, rfl⟩ : ∃ m, g = sc.off + ((P.defs[sc.d]!).body.offsets[k]! + m) :=
      ⟨g - sc.off - (P.defs[sc.d]!).body.offsets[k]!, by omega⟩
    have hm : m < node.members := by omega
    have hops := (hloc' m hm ops).1 hl
    obtain ⟨i, hi, t', ht', hres⟩ := (operands_mem P _ _ m _ hops a).1 ha
    rw [ref_get] at hres
    have hlt := hOp i hi m hm t' ht'
    obtain ⟨x, hx, hxa⟩ := agree_tgt (hag i hi) (List.mem_map.2 ⟨_, List.mem_range.2 hlt, rfl⟩) hres
    refine ⟨⟨x, ?_, hxa⟩, hout⟩
    rw [hread]
    exact List.mem_flatMap.2 ⟨i, List.mem_range.2 hi, hx⟩
  · rw [hread] at hx
    obtain ⟨i', hi, hx⟩ := List.mem_flatMap.1 hx
    have hi' := List.mem_range.1 hi
    obtain ⟨v, hv, hxv⟩ := agree_src (hag i' hi') hx
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hv
    obtain ⟨-, -, -, -, -, a', ha', -⟩ := hmem i' hi' t (List.mem_range.1 ht)
    exact ⟨a', hxv a' ha'⟩

/-! ## `NodeSet` at a primitive node from the decoder's facts, over the summaries before a Definition -/

/-- **`NodeSet` at a node calling a primitive, from the decoder's facts** (`RootFacts`, `WidthFacts`): in a reached
composite, over summaries whose callees' hold `RetOk`, wherever every reference of the node is read by a member. -/
theorem nodeSet_prim_of (P : Prog) (r : Array Desc.Summary) (hF : RootFacts P) (hWF : WidthFacts P) {d : Nat}
    (hd : d < P.defs.size) (hrd : (Desc.reached P)[d]! = true) (hpd : (P.defs[d]!).prim = false)
    (hR0 : ∀ k node, (ctxOf P r d).node k = .ok node → RetOk P r node.fn) {k : Nat}
    {node : Flock.Program.Node} {rs : Array Seg} (hn : (ctxOf P r d).node k = .ok node)
    (hp : (P.defs[node.fn]!).prim = true) (hrs : Desc.nodeReads (ctxOf P r d) k = .ok rs)
    (hCov : ¬ (node.form == .batch && node.n == 0) = true → ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
      ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
        ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
      (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t) :
    NodeSet P r d k := by
  obtain ⟨-, -, -, hDec, -, hSO, -, -, hB⟩ := hF
  have hBd := hB d hd hrd
  have hkb : k < (P.defs[d]!).body.nodes.size := node_lt hn
  have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
  have hfn : node.fn < P.defs.size := by have := (hBd.1 k hkb).1; rwa [hg] at this
  obtain ⟨hL, hlast, -⟩ := (hWF d hd).2 hpd
  have hend : (P.defs[d]!).body.offsets[k + 1]! ≤ (P.defs[d]!).gates := by
    rw [← hlast]
    exact laid_le hL (by omega) (le_refl _)
  refine nodeSet_prim P r hL hkb hend hn hp ((hWF node.fn hfn).1 hp) hrs
    (fun sc hsc => (specs_decoded P r hd hDec hBd hR0 (hSO r d) sc hsc Flock.Steps.FUEL).1)
    (fun i hi => argOk_decoded P r hd hDec k i hkb
      (by change i < ((P.defs[d]!).body.nodes[k]!).args.size; rw [hg]; exact hi))
    (fun i _ t ht => ?_) ?_ hCov
  · have := (hBd.1 k hkb).2.1 i t (by rw [hg]; exact ht)
    rwa [hg] at this
  · have := (hBd.1 k hkb).2.2.2.2 (by rw [hg]; exact hp)
    rwa [hg] at this

/-- **A reached composite's callees hold `RetOk` over the summaries before it** (`memoOf_retOk`, `retOk_congr`). -/
theorem prefix_retOk (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P) {f : Nat}
    (hf : f < P.defs.size) (hrf : (Desc.reached P)[f]! = true) (hpf : (P.defs[f]!).prim = false)
    {r : Array Desc.Summary} (hpre : ∀ f' < f, r[f']! = memo[f']!) :
    ∀ k node, (ctxOf P r f).node k = .ok node → RetOk P r node.fn := by
  obtain ⟨-, -, -, hDec, hRP, hSO, hcf, hnr, hB⟩ := hF
  intro k node hn
  have hk : k < (P.defs[f]!).body.nodes.size := node_lt hn
  have hg : (P.defs[f]!).body.nodes[k]! = node := node_get hn
  have hlt := hcf f hf k hk
  rw [← hg]
  exact retOk_congr (hpre _ hlt).symm (memoOf_retOk P memo hm hRP hDec hcf hnr hB hSO _ (by omega)
    (reached_closed P hcf f hrf hpf k hk) (hnr f hf hrf k hk))

/-! ## A parameter's leaf resolves before the activation -/

theorem good_bodyOk (P : Prog) (hF : RootFacts P) : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc hsc => by
  obtain ⟨hd, hrd, -⟩ := good_reached P hF sc hsc
  exact hF.2.2.2.2.2.2.2.2 sc.d hd hrd

theorem laidOut_of (P : Prog) (hwk : Desc.walkable P = true) : ∀ d < P.defs.size, Walk.LaidOut P d := by
  unfold Desc.walkable at hwk
  simp only [Bool.and_eq_true] at hwk
  exact (Walk.bodiesLaidOut_spec P hwk.1.2).2.1

/-- **A parameter's leaf resolves before the activation**, in any activation resolution reaches (`Resolve.resolve_lt`). -/
theorem param_lt (P : Prog) (hF : RootFacts P) (hwk : Desc.walkable P = true) (sc : Flock.Program.Scope)
    (hg : Resolve.Good P sc) {i j a : Nat} (hi : i < (P.defs[sc.d]!).body.params.size)
    (hj : j < ((P.defs[sc.d]!).body.params[i]!).2.leaves)
    (h : Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a) : a < sc.off :=
  Resolve.resolve_lt P ⟨hF.1, hF.2.1⟩ (good_bodyOk P hF) (laidOut_of P hwk) sc.off sc ⟨true, i, j⟩
    ⟨hg, ⟨fun _ => ⟨hi, hj⟩, fun h => by cases h⟩, fun _ => le_refl _, fun h => by cases h⟩ h

/-! ## Segments, widths and `UsedExact` of every reached Definition from the decoder's facts -/

/-- **Every segment a body node's read splits is as `down` gives it** (`ShapeOk`, `RepsOk`), in a decoded Definition
whose callees' summaries hold `RetOk`: `rootShape` in any body (`piece_shape`, `throughPS_shape`,
`carriesRead_shape`), with `reps_fuel` through `carriesRead_of`. -/
theorem bodySegs (P : Prog) (r : Array Desc.Summary) {d : Nat} (hd : d < P.defs.size) (hDec : Decoded P)
    (hBd : Resolve.BodyOk P d) (hR : ∀ k node, (ctxOf P r d).node k = .ok node → RetOk P r node.fn)
    (hSO : ScanOut (ctxOf P r d)) {k : Nat} {node : Flock.Program.Node} (hn : (ctxOf P r d).node k = .ok node)
    {s : Seg} (hs : (∃ i, FromDown (ctxOf P r d) k i s) ∨ FromRows (ctxOf P r d) k node s ∨
      FromCarry (ctxOf P r d) k node s) : ShapeOk P d s ∧ RepsOk s := by
  have hSh := shapes_decoded P r hd hDec hBd hR hSO Flock.Steps.FUEL
  have hA := argOk_decoded P r hd hDec
  have hRp := reps_fuel (ctxOf P r d) Flock.Steps.FUEL
  have hPS : ∀ k node i p bp segs, (ctxOf P r d).node k = .ok node →
      Desc.throughPS ((ctxOf P r d).arg k i) p = some bp →
      Desc.down (ctxOf P r d) Flock.Steps.FUEL bp = .ok segs → ∀ s ∈ segs.toList, ShapeOk P d s := by
    intro k node i p bp segs hn hp hd'
    have hk := node_lt hn
    by_cases hi' : i < ((ctxOf P r d).body.nodes[k]!).args.size
    · exact throughPS_shape _ _ hSh.1 (hA k i hk hi') (ctxOf_argCount P r d k i hk hi') ((hBd.1 k hk).2.1 i)
        hp hd'
    · rw [throughPS_empty (ctxOf_arg_ge P r d k i hi')] at hp
      cases hp
  have hk := node_lt hn
  rcases hs with ⟨i, lo, hi, q, hq, segs, hd', hs⟩ | ⟨i, u, ps, hmr, p, hp, segs, hd', hs⟩ | ⟨hfs, hn2, segs, hc, hs⟩
  · refine ⟨?_, hRp.1 q segs hd' s (Array.mem_toList_iff.2 hs)⟩
    by_cases hi' : i < ((ctxOf P r d).body.nodes[k]!).args.size
    · exact piece_shape _ _ hSh.1 (hA k i hk hi') (ctxOf_argCount P r d k i hk hi') ((hBd.1 k hk).2.1 i) hq hd'
        s (Array.mem_toList_iff.2 hs)
    · rw [pieces_empty (ctxOf_arg_ge P r d k i hi')] at hq
      simp at hq
  · refine ⟨?_, hRp.1 p segs hd' s (Array.mem_toList_iff.2 hs)⟩
    obtain ⟨p', hp'⟩ := memberRows_mem _ node k i u hmr p (Array.mem_toList_iff.2 hp)
    exact hPS k node i p' p segs hn hp' hd' s (Array.mem_toList_iff.2 hs)
  · refine ⟨carriesRead_shape _ hR hSh.2.1 hSh.2.2.1 hPS k node segs hn hfs hn2 hc s (Array.mem_toList_iff.2 hs), ?_⟩
    exact carriesRead_of RepsOk _ hR (fun k _ _ _ m a b segs _ _ h => hRp.2.1 k m a b segs h)
      (fun k _ _ _ lo hi segs _ h => hRp.2.2.1 k 0 0 _ segs h) (fun _ _ _ _ bp segs _ _ h => hRp.1 bp segs h)
      k node segs hn hfs hn2 hc s (Array.mem_toList_iff.2 hs)

theorem retW_congr {P : Prog} {m m' : Array Desc.Summary} {f : Nat} (e : m[f]! = m'[f]!)
    (h : RetW P m f) : RetW P m' f := by
  unfold RetW at h ⊢
  rw [← e]
  exact h

/-- **A reached composite's callees name their gates with their widths over the summaries before it**
(`memoOf_retW`, `retW_congr`). -/
theorem prefix_retW (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) {f : Nat} (hf : f < P.defs.size) (hrf : (Desc.reached P)[f]! = true)
    (hpf : (P.defs[f]!).prim = false) {r : Array Desc.Summary} (hpre : ∀ f' < f, r[f']! = memo[f']!) :
    ∀ k node, (ctxOf P r f).node k = .ok node → RetW P r node.fn := by
  obtain ⟨-, -, -, hDec, hRP, hSO, hcf, hnr, hB⟩ := hF
  intro k node hn
  have hk : k < (P.defs[f]!).body.nodes.size := node_lt hn
  have hg : (P.defs[f]!).body.nodes[k]! = node := node_get hn
  have hlt := hcf f hf k hk
  rw [← hg]
  exact retW_congr (hpre _ hlt).symm (memoOf_retW P memo hm hRP hDec hWF hcf hnr hB hSO _ (by omega)
    (reached_closed P hcf f hrf hpf k hk) (hnr f hf hrf k hk)).2

/-- **Every segment a body node's read splits names the body's gates with their widths** (`GateW`): `rootW` in any
decoded composite whose callees' summaries hold `RetOk` and name theirs. -/
theorem bodyW (P : Prog) (r : Array Desc.Summary) {d : Nat} (hd : d < P.defs.size) (hpd : (P.defs[d]!).prim = false)
    (hDec : Decoded P) (hWF : WidthFacts P) (hBd : Resolve.BodyOk P d)
    (hR : ∀ k node, (ctxOf P r d).node k = .ok node → RetOk P r node.fn)
    (hRW : ∀ k node, (ctxOf P r d).node k = .ok node → RetW P r node.fn) {k : Nat} {node : Flock.Program.Node}
    (hn : (ctxOf P r d).node k = .ok node) {s : Seg}
    (hs : (∃ i, FromDown (ctxOf P r d) k i s) ∨ FromRows (ctxOf P r d) k node s ∨
      FromCarry (ctxOf P r d) k node s) : GateW P d s := by
  have hW := widths_decoded P r hd hpd hDec hWF hBd hR hRW Flock.Steps.FUEL
  have hA := argOk_decoded P r hd hDec
  have hPS : ∀ k node i p bp segs, (ctxOf P r d).node k = .ok node →
      Desc.throughPS ((ctxOf P r d).arg k i) p = some bp →
      Desc.down (ctxOf P r d) Flock.Steps.FUEL bp = .ok segs → ∀ s ∈ segs.toList, GateW P d s := by
    intro k node i p bp segs hn hp hd'
    have hk := node_lt hn
    by_cases hi' : i < ((ctxOf P r d).body.nodes[k]!).args.size
    · exact throughPS_w _ _ hW.1 (hA k i hk hi') (ctxOf_argCount P r d k i hk hi') ((hBd.1 k hk).2.1 i) hp hd'
    · rw [throughPS_empty (ctxOf_arg_ge P r d k i hi')] at hp
      cases hp
  have hk := node_lt hn
  rcases hs with ⟨i, lo, hi, q, hq, segs, hd', hs⟩ | ⟨i, u, ps, hmr, p, hp, segs, hd', hs⟩ | ⟨hfs, hn2, segs, hc, hs⟩
  · by_cases hi' : i < ((ctxOf P r d).body.nodes[k]!).args.size
    · exact piece_w _ _ hW.1 (hA k i hk hi') (ctxOf_argCount P r d k i hk hi') ((hBd.1 k hk).2.1 i) hq hd'
        s (Array.mem_toList_iff.2 hs)
    · rw [pieces_empty (ctxOf_arg_ge P r d k i hi')] at hq
      simp at hq
  · obtain ⟨p', hp'⟩ := memberRows_mem _ node k i u hmr p (Array.mem_toList_iff.2 hp)
    exact hPS k node i p' p segs hn hp' hd' s (Array.mem_toList_iff.2 hs)
  · refine carriesRead_of (GateW P d) _ hR ?_ ?_ hPS k node segs hn hfs hn2 hc s (Array.mem_toList_iff.2 hs)
    · intro k' node' hn' hf' m a b segs' hm' _ h'
      refine hW.2.1 k' m a b segs' hm' ?_ h'
      have hk' := node_lt hn'
      have hg : (P.defs[d]!).body.nodes[k']! = node' := node_get hn'
      exact (((hWF d hd).2 hpd).2.2 k' hk').2 (Or.inr (by rw [hg]; exact hf'))
    · intro k' node' hn' hm0 lo hi' segs' hlv h'
      exact init_w _ hn' hW.2.2.1 hm0 hlv h'

/-- **A gate position a body node reads is one of the body's gates** (`bodyW`, `locW_lt`). -/
theorem bodyGate_lt (P : Prog) (r : Array Desc.Summary) {d : Nat} (s : Desc.Summary) (h : Desc.summarize P r d = .ok s)
    (hd : d < P.defs.size) (hpd : (P.defs[d]!).prim = false) (hDec : Decoded P) (hWF : WidthFacts P)
    (hBd : Resolve.BodyOk P d) (hR : ∀ k node, (ctxOf P r d).node k = .ok node → RetOk P r node.fn)
    (hRW : ∀ k node, (ctxOf P r d).node k = .ok node → RetW P r node.fn) (hSO : ScanOut (ctxOf P r d))
    {k : Nat} (hk : k < (P.defs[d]!).body.nodes.size) {node : Flock.Program.Node}
    (hn : (ctxOf P r d).node k = .ok node) (w : Nat) (p : Int)
    (hx : (Tgt.gate w, p) ∈ nodePos (ctxOf P r d) node k) : p < ((P.defs[d]!).gates : Int) := by
  obtain ⟨g, hg, hxg⟩ := nodePos_seg P r d s h hpd hk hn hx
  obtain ⟨hL, hlast, -⟩ := (hWF d hd).2 hpd
  have h1 := locW_lt (bodyW P r hd hpd hDec hWF hBd hR hRW hn hg w p hxg) hL hlast
  have h0 := ((bodySegs P r hd hDec hBd hR hSO hn hg).1.2.1 _ hxg).1
  omega

/-- **`UsedExact` of every reached composite Definition but the root**, with the decoder's facts (`RootFacts`,
`WidthFacts`, `Desc.walkable`), wherever every reference of a primitive node of a reached composite is read by a
member unless it is a batch of none (`hCov`) and, over the summaries before a Definition, `NodeSet` holds of each of its nodes calling a composite
once its composite callees' `UsedExact` does there (`hComp`): `memoOf_usedExact`, each step `usedExact_of` with
`nodeSet_prim_of`, `bodySegs`, `bodyGate_lt` and `param_lt`. -/
theorem usedExact_all (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) (hwk : Desc.walkable P = true)
    (hCov : ∀ d < P.defs.size, (Desc.reached P)[d]! = true → (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (P.defs[d]!).body.nodes[k]! = node →
      (P.defs[node.fn]!).prim = true → ¬ (node.form == .batch && node.n == 0) = true →
      ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
        ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
          ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
        (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t)
    (hComp : ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → (P.defs[f]!).prim = false →
      ∀ r : Array Desc.Summary, r.size = f → (∀ f' < f, r[f']! = memo[f']!) →
      Desc.summarize P r f = .ok memo[f]! →
      (∀ k < (P.defs[f]!).body.nodes.size, (P.defs[((P.defs[f]!).body.nodes[k]!).fn]!).prim = false →
        UsedExact P r ((P.defs[f]!).body.nodes[k]!).fn) →
      ∀ k < (P.defs[f]!).body.nodes.size, (P.defs[((P.defs[f]!).body.nodes[k]!).fn]!).prim = false →
        NodeSet P r f k) :
    ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → (P.defs[f]!).prim = false →
      UsedExact P memo f := by
  refine memoOf_usedExact P memo hm hF.2.2.2.2.2.2.1 hF.2.2.2.2.2.2.2.1
    fun f hf hrf hfr hp r hsz hpre hs hU => ?_
  have hDec := hF.2.2.2.1
  have hBd := hF.2.2.2.2.2.2.2.2 f hf hrf
  have hSO := hF.2.2.2.2.2.1 r f
  have hR := prefix_retOk P memo hm hF hf hrf hp hpre
  have hRW := prefix_retW P memo hm hF hWF hf hrf hp hpre
  obtain ⟨hL, hlast, -⟩ := (hWF f hf).2 hp
  refine usedExact_of P r memo f hs hp hL hlast (fun k hk => ?_)
    (fun k hk node hn g hg => bodySegs P r hf hDec hBd hR hSO hn hg)
    (fun k hk node hn w p hx => bodyGate_lt P r _ hs hf hp hDec hWF hBd hR hRW hSO hk hn w p hx)
    (fun sc hsc hg i j a hi hj h => by subst hsc; exact param_lt P hF hwk sc hg hi hj h)
  have hn := ctx_node P r f hk
  obtain ⟨rs, hrs⟩ := summarize_reads P r f _ hs hp k hk
  rcases hpc : (P.defs[((P.defs[f]!).body.nodes[k]!).fn]!).prim with _ | _
  · exact hComp f hf hrf hfr hp r hsz hpre hs hU k hk hpc
  · exact nodeSet_prim_of P r hF hWF hf hrf hp hR hn hpc hrs (hCov f hf hrf hp k hk _ rfl hpc)


/-! ## `NodeSet` at a composite node from the decoder's facts, and `RootSet` -/

/-- **Every parameter leaf in range resolves**, in any activation resolution reaches: `Desc.walkable` keeps resolution
within `Flock.Steps.FUEL` (`OperandsResolve.resolve_ok`). -/
theorem param_ok (P : Prog) (hF : RootFacts P) (hwk : Desc.walkable P = true) (sc : Flock.Program.Scope)
    (hg : Resolve.Good P sc) {i j : Nat} (hi : i < (P.defs[sc.d]!).body.params.size)
    (hj : j < ((P.defs[sc.d]!).body.params[i]!).2.leaves) :
    ∃ a, Flock.Program.resolve P sc ⟨true, i, j⟩ = .ok a := by
  have hw := hwk
  unfold Desc.walkable at hw
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hw
  obtain ⟨⟨hc, -⟩, hres, hwb⟩ := hw
  obtain ⟨-, hz⟩ := OperandsResolve.resolvable_spec P hres
  have hfuel : 4 * (Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL := by
    unfold Desc.walkBound at hwb
    have : ((Desc.activations P)[P.root]! + 1) * 2 ≤
        ((Desc.activations P)[P.root]! + 1) * ((Desc.depths P)[P.root]! + 2) := Nat.mul_le_mul_left _ (by omega)
    omega
  exact OperandsResolve.resolve_ok P ⟨hF.1, hF.2.1⟩ (good_bodyOk P hF) (laidOut_of P hwk) hc hF.2.2.1 hz hfuel
    sc.off sc ⟨true, i, j⟩ ⟨hg, ⟨fun _ => ⟨hi, hj⟩, fun h => by cases h⟩, fun _ => le_refl _, fun h => by cases h⟩

/-- **Every parameter position a body node reads resolves**, in any activation of the body resolution reaches: the
segments its read splits name parameter leaves in range (`bodySegs`), and each resolves (`param_ok`). -/
theorem nodePos_ok (P : Prog) (r : Array Desc.Summary) (hF : RootFacts P) (hwk : Desc.walkable P = true) {d : Nat}
    (hd : d < P.defs.size) (hrd : (Desc.reached P)[d]! = true)
    (hR : ∀ k node, (ctxOf P r d).node k = .ok node → RetOk P r node.fn) {k : Nat} {node : Flock.Program.Node}
    {rs : Array Seg} (hn : (ctxOf P r d).node k = .ok node) (hrs : Desc.nodeReads (ctxOf P r d) k = .ok rs)
    (sc : Flock.Program.Scope) (hsd : sc.d = d) (hg : Resolve.Good P sc) :
    ∀ i (p : Int), (Tgt.ref true i, p) ∈ nodePos (ctxOf P r d) node k → ∃ a, den P sc (.ref true i, p) = .ok a := by
  intro i p hx
  obtain ⟨out, -, -, hpos, hfrom⟩ := nodeReads_out _ k node hn hrs
  rw [← hpos] at hx
  obtain ⟨g, hg', hxg⟩ := List.mem_flatMap.1 hx
  have hsh := bodySegs P r hd hF.2.2.2.1 (hF.2.2.2.2.2.2.2.2 d hd hrd) hR (hF.2.2.2.2.2.1 r d) hn
    (hfrom g (Array.mem_toList_iff.1 hg'))
  obtain ⟨h0, -, hpi⟩ := hsh.1.2.1 _ hxg
  obtain ⟨hi, hp⟩ := hpi i rfl
  dsimp only at h0 hi hp
  subst hsd
  obtain ⟨a, ha⟩ := param_ok P hF hwk sc hg hi (j := p.toNat) (by omega)
  refine ⟨a, ?_⟩
  simp only [den]
  rw [if_neg (by omega)]
  exact ha

/-- **A leaf `used` names is a parameter leaf in range**, of every reached composite but the root: a position a node
reads (`used_nodePos`), as `down` gives it (`bodySegs`). -/
theorem used_lt (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P) {f : Nat}
    (hf : f < P.defs.size) (hrf : (Desc.reached P)[f]! = true) (hfr : f ≠ P.root)
    (hpf : (P.defs[f]!).prim = false) :
    ∀ i j, Width.inIvs (((memo[f]!).used[i]?.getD #[]).toList) j →
      i < (P.defs[f]!).body.params.size ∧ j < ((P.defs[f]!).body.params[i]!).2.leaves := by
  intro i j hij
  obtain ⟨r, -, hpre, hs⟩ := memoOf_prefix P memo hm f hf hrf hfr
  have hR := prefix_retOk P memo hm hF hf hrf hpf hpre
  have hsh : ∀ k node, (ctxOf P r f).node k = .ok node → ∀ g, ((∃ i, FromDown (ctxOf P r f) k i g) ∨
      FromRows (ctxOf P r f) k node g ∨ FromCarry (ctxOf P r f) k node g) → ShapeOk P f g ∧ RepsOk g :=
    fun k node hn g hg => bodySegs P r hf hF.2.2.2.1 (hF.2.2.2.2.2.2.2.2 f hf hrf) hR (hF.2.2.2.2.2.1 r f) hn hg
  obtain ⟨k, hk, node, hn, hx⟩ := used_nodePos P r f _ hs hpf
    (fun k _ node hn g hg => ⟨fun x hx => ((hsh k node hn g hg).1.2.1 x hx).1,
      by have := (hsh k node hn g hg).1.2.2.2; omega, (hsh k node hn g hg).1.2.2.1⟩) hij
  obtain ⟨g, hg, hxg⟩ := nodePos_seg P r f _ hs hpf hk hn hx
  obtain ⟨hi, hj⟩ := ((hsh k node hn g hg).1.2.1 _ hxg).2.2 i rfl
  exact ⟨hi, by omega⟩

/-- What the decoder establishes of a node calling a composite, as `compNodeSet` reads it: the callee's parameters
are its body's, one per argument, a dimension of no leaves goes to a parameter of none, each member's parameter leaf
is a reference of the argument, at a node with members every reference is one, and a scan's carry is within its
callee's return. -/
def CompShape (P : Prog) (node : Flock.Program.Node) : Prop :=
  (P.defs[node.fn]!).params = (P.defs[node.fn]!).body.params ∧
  (P.defs[node.fn]!).body.params.size = node.args.size ∧
  (∀ i < node.args.size, ∀ e L, (node.args[i]!).dims = some (e, L) → L = 0 →
    ((P.defs[node.fn]!).body.params[i]!).2.leaves = 0) ∧
  (∀ i < node.args.size, ∀ m < node.members, ∀ j < ((P.defs[node.fn]!).body.params[i]!).2.leaves,
    (node.form = .scan → i = 0 → m = 0) → upIdx node i m j < (node.args[i]!).seq.count) ∧
  (node.members ≠ 0 → ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
    ∃ j < ((P.defs[node.fn]!).body.params[i]!).2.leaves, upIdx node i m j = t) ∧
  (node.form = .scan → ((P.defs[node.fn]!).body.params[0]!).2.leaves ≤ (P.defs[node.fn]!).body.ret.count)

/-- **`NodeSet` at a node calling a composite, from the decoder's facts** (`RootFacts`, `WidthFacts`, `CompShape`): in
a reached composite, over summaries whose callees' hold `RetOk` and `RetW`, wherever the callee's `UsedExact` holds and
its `used` names its parameter leaves in range: `compNodeSet`, with the gates its return names in range (`locW_lt`)
and every parameter position resolving (`nodePos_ok`). -/
theorem nodeSet_comp_of (P : Prog) (r : Array Desc.Summary) (hF : RootFacts P) (hWF : WidthFacts P)
    (hwk : Desc.walkable P = true) {d : Nat} (hd : d < P.defs.size) (hrd : (Desc.reached P)[d]! = true)
    (hpd : (P.defs[d]!).prim = false)
    (hR0 : ∀ k node, (ctxOf P r d).node k = .ok node → RetOk P r node.fn)
    (hRW : ∀ k node, (ctxOf P r d).node k = .ok node → RetW P r node.fn) {k : Nat}
    {node : Flock.Program.Node} {rs : Array Seg} (hn : (ctxOf P r d).node k = .ok node)
    (hp : (P.defs[node.fn]!).prim = false) (hrs : Desc.nodeReads (ctxOf P r d) k = .ok rs)
    (hU : UsedExact P r node.fn)
    (hused : ∀ i j, Width.inIvs (((r[node.fn]!).used[i]?.getD #[]).toList) j →
      i < (P.defs[node.fn]!).body.params.size ∧ j < ((P.defs[node.fn]!).body.params[i]!).2.leaves)
    (hS : CompShape P node) :
    NodeSet P r d k := by
  obtain ⟨hprm, hpa, hL0, hlo, htile, hsc0⟩ := hS
  obtain ⟨-, -, -, hDec, -, hSO, -, -, hB⟩ := id hF
  have hBd := hB d hd hrd
  have hkb : k < (P.defs[d]!).body.nodes.size := node_lt hn
  have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
  have hfn : node.fn < P.defs.size := by have := (hBd.1 k hkb).1; rwa [hg] at this
  obtain ⟨hL, hlast, -⟩ := (hWF d hd).2 hpd
  obtain ⟨hLf, hlastf, -⟩ := (hWF node.fn hfn).2 hp
  have hend : (P.defs[d]!).body.offsets[k + 1]! ≤ (P.defs[d]!).gates := by
    rw [← hlast]
    exact laid_le hL (by omega) (le_refl _)
  intro sc hsd hgood hloc node' hn'
  rw [hn] at hn'
  injection hn' with e
  subst e
  have hok := nodePos_ok P r hF hwk hd hrd hR0 hn hrs sc hsd hgood
  subst hsd
  have hS := specs_decoded P r hd hDec hBd hR0 (hSO r sc.d) sc rfl Flock.Steps.FUEL
  refine ⟨fun a => compNodeSet P r hL hkb hn hp hend hloc hgood hU (hR0 k node hn) hS.1 hS.2.1 hS.2.2.1 hrs
    (fun i hi => argOk_decoded P r hd hDec k i hkb
      (by change i < ((P.defs[sc.d]!).body.nodes[k]!).args.size; rw [hg]; exact hi))
    (fun i _ t ht => by have := (hBd.1 k hkb).2.1 i t (by rw [hg]; exact ht); rwa [hg] at this)
    hprm hpa hused hL0 hlo htile (fun w p hx => locW_lt (hRW k node hn w p hx) hLf hlastf)
    (fun hs j hj => by have := (hused 0 j hj).2; have := hsc0 hs; omega) a, hok⟩

/-- **`UsedExact` of every reached composite Definition but the root, from the decoder's facts** (`RootFacts`,
`WidthFacts`, `Desc.walkable`), wherever every reference of a primitive node is read by a member unless it is a batch
of none (`hCov`) and every node calling a composite is shaped as `CompShape` says: `usedExact_all`, each composite
node's `NodeSet` by `nodeSet_comp_of` over the summaries before its Definition (`prefix_retOk`, `prefix_retW`,
`used_lt`). -/
theorem usedExact_reached (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) (hwk : Desc.walkable P = true)
    (hCov : ∀ d < P.defs.size, (Desc.reached P)[d]! = true → (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (P.defs[d]!).body.nodes[k]! = node →
      (P.defs[node.fn]!).prim = true → ¬ (node.form == .batch && node.n == 0) = true →
      ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
        ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
          ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
        (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t)
    (hShape : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (P.defs[d]!).body.nodes[k]! = node →
      (P.defs[node.fn]!).prim = false → CompShape P node) :
    ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → (P.defs[f]!).prim = false →
      UsedExact P memo f := by
  refine usedExact_all P memo hm hF hWF hwk hCov fun f hf hrf _ hp r _ hpre hs hU k hk hpc => ?_
  obtain ⟨-, -, -, -, -, -, hcf, hnr, -⟩ := id hF
  have hn := ctx_node P r f hk
  obtain ⟨rs, hrs⟩ := summarize_reads P r f _ hs hp k hk
  have hlt := hcf f hf k hk
  refine nodeSet_comp_of P r hF hWF hwk hf hrf hp (prefix_retOk P memo hm hF hf hrf hp hpre)
    (prefix_retW P memo hm hF hWF hf hrf hp hpre) hn hpc hrs (hU k hk hpc) ?_ (hShape f hf k hk _ rfl hpc)
  rw [hpre _ hlt]
  exact used_lt P memo hm hF (by omega) (reached_closed P hcf f hrf hp k hk) (hnr f hf hrf k hk) hpc

/-- **`NodeSet` at a root node calling a composite, from the decoder's facts**: `nodeSet_comp_of` at the root, its
callee's `UsedExact` by `usedExact_reached`. -/
theorem rootNodeSet (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) (hwk : Desc.walkable P = true)
    (hU : ∀ f < P.defs.size, (Desc.reached P)[f]! = true → f ≠ P.root → (P.defs[f]!).prim = false →
      UsedExact P memo f)
    (hShape : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (P.defs[d]!).body.nodes[k]! = node →
      (P.defs[node.fn]!).prim = false → CompShape P node)
    {k : Nat} {node : Flock.Program.Node} {rs : Array Seg} (hn : (rootCtx P memo).node k = .ok node)
    (hrs : Desc.nodeReads (rootCtx P memo) k = .ok rs) (hp : (P.defs[node.fn]!).prim = false) :
    NodeSet P memo P.root k := by
  obtain ⟨hr, hpr, -, -, -, -, hcf, hnr, -⟩ := id hF
  have hrr := reached_root P hr
  have hkb : k < (P.defs[P.root]!).body.nodes.size := node_lt hn
  have hg : (P.defs[P.root]!).body.nodes[k]! = node := node_get hn
  have hfn : node.fn < P.defs.size := by have := hcf P.root hr k hkb; rw [hg] at this; omega
  have hrf : (Desc.reached P)[node.fn]! = true := by
    have := reached_closed P hcf P.root hrr hpr k hkb; rwa [hg] at this
  have hfr : node.fn ≠ P.root := by have := hnr P.root hr hrr k hkb; rwa [hg] at this
  exact nodeSet_comp_of P memo hF hWF hwk hr hrr hpr (prefix_retOk P memo hm hF hr hrr hpr fun _ _ => rfl)
    (prefix_retW P memo hm hF hWF hr hrr hpr fun _ _ => rfl) hn hp hrs (hU node.fn hfn hrf hfr hp)
    (used_lt P memo hm hF hfn hrf hfr hp) (hShape P.root hr k hkb node hg hp)

/-- **`RootSet`, from the decoder's facts** (`RootFacts`, `WidthFacts`): every reference of a node calling a
primitive is read by a member unless it is a batch of none (`hCov`), and every node calling a composite is shaped as
`CompShape` says: at a root node calling a primitive `rootSet_prim_of`, at one calling a composite `rootSet_comp`
with `rootNodeSet`. -/
theorem rootSet_of
    (hF : ∀ bytes P, Flock.Program.decode bytes = .ok P → P.orderViolation = none → RootFacts P)
    (hWF : ∀ bytes P, Flock.Program.decode bytes = .ok P → Desc.walkable P = true → WidthFacts P)
    (hCov : ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P →
      ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (P.defs[d]!).body.nodes[k]! = node →
      (P.defs[node.fn]!).prim = true → ¬ (node.form == .batch && node.n == 0) = true →
      ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
        ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
          ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
        (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t)
    (hShape : ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P →
      ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (P.defs[d]!).body.nodes[k]! = node →
      (P.defs[node.fn]!).prim = false → CompShape P node) :
    RootSet := by
  intro bytes P memo c hd ho hl hi hw hm hc k hk node rs hn hrs hg a
  have hF' := hF bytes P hd ho
  have hWF' := hWF bytes P hd hw
  have hgk : (P.defs[P.root]!).body.nodes[k]! = node := node_get hn
  rcases hp : (P.defs[node.fn]!).prim with _ | _
  · have := rootSet_comp P memo hm hF' c hc hl hk hn hrs
      (rootNodeSet P memo hm hF' hWF' hw
        (usedExact_reached P memo hm hF' hWF' hw (fun d hd' _ _ => hCov bytes P hd d hd') (hShape bytes P hd))
        (hShape bytes P hd) hn hrs hp) a
    simpa using this
  · have := rootSet_prim_of P memo hm hF' hWF' hw c hc hl hk hn hp hrs hg
      (hCov bytes P hd P.root hF'.1 k hk node hgk hp) a
    simpa using this

end NciCheck.Properties.SecurityProofs.ReadsExact
