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

end NciCheck.Properties.SecurityProofs.ReadsExact
