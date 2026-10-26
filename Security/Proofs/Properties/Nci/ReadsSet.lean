import Proofs.Properties.Nci.ReadsOneNode

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
(`readSet`, through `branch_iff`).
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

end NciCheck.Properties.SecurityProofs.ReadsExact
