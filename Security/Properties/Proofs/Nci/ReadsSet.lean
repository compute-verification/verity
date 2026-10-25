import Properties.Proofs.Nci.ReadsOneNode

/-!
# Toward `ReadsExact.RootSet`: gates located with their operands, and the statements it is proved through

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

At a composite node, what its gates read outside it is what its members' parameter leaves that the callee's `used`
names resolve to (`compReads`), each member's activation located as its caller's is (`located_child`). One argument's node read reads, outside the
node, what the members' reads of its parameter leaves resolve to, when its positions are those leaves' `upIdx`
(`readSet`, through `branch_iff`). Its read succeeding (`nodeReads_comp_ok`, `ArgReadOk`), an argument's read names
exactly its members' reads of the parameter (`argSet`, `ArgB`) wherever its members' leaves tile it, and the node's
positions what its gates read outside it wherever its carries' read names the later iterations' (`compSet_of`). All
members at once, `Desc.memberRows` reads per used run the rows `rowsAt` names (`memberRows_rows`), whose indices are
the members' leaves there (`rowsAt_idx`), each through `throughPS` keeping its references' gates (`throughPS_agree`):
the same set (`rowsSet`, `argSet_of`).
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
  ∀ sc : Flock.Program.Scope, sc.d = f → Located P sc → ∀ node, (ctxOf P memo f).node k = .ok node → ∀ a : Nat,
    ((∃ x ∈ nodePos (ctxOf P memo f) node k, den P sc x = .ok a) ∧ ¬ InNodeAt P sc k a) ↔
      (NodeReadsAt P sc k a ∧ ¬ InNodeAt P sc k a)

/-- **Definition `f`'s summarized parameter reads (`used`) are what its activations read outside themselves**: in an
activation `sc` whose gates are located, a gate reading a gate outside `sc` reads parameter `i`'s leaf `j` there, `j`
in `used[i]`, and each such leaf is read so by a gate of `sc`. -/
def UsedExact (P : Prog) (memo : Array Desc.Summary) (f : Nat) : Prop :=
  ∀ sc : Flock.Program.Scope, sc.d = f → Located P sc →
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
(`hOp`), and every reference read by some member (`hCov`). -/
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
    (hCov : ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
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
    obtain ⟨m, hm, t', ht'', hidx⟩ := hCov i hi' t ht'
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
    (hU : UsedExact P memo ((P.defs[sc.d]!).body.nodes[k]!).fn) (a : Nat) :
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
    obtain ⟨i, j, hij, hr⟩ := (hU (P.child sc k m) rfl (located_child P hL hk hp hm hend hloc)).1 g ops a
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
      (hU (P.child sc k m) rfl (located_child P hL hk hp hm hend hloc)).2 i j hij
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
    (hU : UsedExact P memo node.fn)
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
  rw [compReads P memo hL hk hp hend hloc hU a]
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

end NciCheck.Properties.SecurityProofs.ReadsExact
