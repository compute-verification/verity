import Proofs.Properties.Nci
import Properties.NciDesc

/-!
# `NciCheck.Desc.evaluate` agrees with `NciCheck.evaluate`

`descAgrees_of` proves `Properties.DescAgrees` from two statements about the decoder and the description-sized analysis,
each a named `Prop` here, both proved: the first by `Walk.gatesLocated_of`, from `Resolve.bodiesOk`, and the second by
`Novel.novelExact_of`, from `Novel.readsExact`:

* `GatesLocated`: on a Program the decoder returns and `Desc.evaluate` reads (topological by `Prog.orderViolation`, its
  root laid out, its input nodes the root's, `Desc.walkable`, and `Desc.analyze` resolving its references), every gate of
  the canonical layout is located and reads only earlier gates (`GateOk`), so `NciCheck.circuitOf` lays it out.
* `NovelExact`: on such a Program, for a division whose IUs are unions of root nodes, the novel reads `Desc.novelWidth`
  counts per IU, from what `Desc.analyze` resolves each root node to read, are `NciCheck.ius`'s, gate by gate.

The rest is proved: `Desc.evaluate`'s refusals and folds (`descEvaluate_ok`, `nodeThetaFold_ok`, `nodeOwnerFold_ok`),
the root's layout (`rootLaidOut_spec`, `node_unique`, `node_exists`), that a root node's gates are input gates exactly
when it is an input node (`gate_input`, through `Prog.gate`'s walk and `inputsAtRoot`), that every verdict has an
expansion, and that on each the unrolled check accepts θ and the owners, counts the same IUs, and the same sizes
(`sizes_eq_tally`).
-/

namespace NciCheck.Properties.SecurityProofs.DescAgree

open Lean (Json toJson)
open Flock.Program (Prog)


/-- The root's body, its number of nodes, its offsets, and which of its nodes are input nodes, as `Desc.evaluate` reads
them. -/
def rootBody (P : Prog) : Flock.Program.Body := (P.defs[P.root]!).body
def nodeCount (P : Prog) : ℕ := (rootBody P).nodes.size
def rootOffs (P : Prog) : Array ℕ := (rootBody P).offsets
def inputs (P : Prog) : Array Bool := (rootBody P).nodes.map (Desc.isInputNode P)
def registered (P : Prog) : Array Bool := (rootBody P).nodes.map fun n => isRegisteredId (P.defs[n.fn]!).key

/-- `Desc.evaluate`'s fold over θ by root node. -/
def nodeThetaFold (P : Prog) (theta : Array Json) : Except String (Array Bool) :=
  (List.range theta.size).foldlM (fun m t => Desc.thetaStep (inputs P) (registered P) m theta[t]!) (Array.replicate (nodeCount P) false)

/-- `Desc.evaluate`'s fold over the owners by root node. -/
def nodeOwnerFold (P : Prog) (owner : Array Json) : Except String (Array (Option ℕ)) :=
  (List.range (nodeCount P)).foldlM (fun own k => own.push <$>
    Desc.ownerStep (inputs P)[k]! ((rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0) owner[k]!) #[]

theorem descEvaluate_ok (j out : Json) (h : Desc.evaluate j = .ok out) :
    ∃ i P reads mask own, parse j = .ok i ∧ decodeProgram i.program = .ok P ∧
      P.orderViolation = none ∧ Desc.rootLaidOut P = true ∧ Desc.inputsAtRoot P = true ∧ Desc.walkable P = true ∧
      Desc.analyze P = .ok reads ∧ nodeThetaFold P i.theta = .ok mask ∧ i.owner.size = nodeCount P ∧
      nodeOwnerFold P i.owner = .ok own ∧ iuCount own ≤ nodeCount P ∧ (∀ u < iuCount own, u ∈ usedOf own) ∧
      out = verdict (P.defs[P.root]!).gates
        ((Array.range (iuCount own)).map fun u => ((Desc.sizes (rootOffs P) own (iuCount own))[u]!,
          Desc.novelWidth (Desc.intoOf (rootOffs P) own mask reads (iuCount own))[u]!))
        (failure i.gamma i.num i.den ((Array.range (iuCount own)).map fun u =>
          ((Desc.sizes (rootOffs P) own (iuCount own))[u]!,
            Desc.novelWidth (Desc.intoOf (rootOffs P) own mask reads (iuCount own))[u]!))) := by
  unfold Desc.evaluate at h
  rcases h1 : parse j with e | i <;> simp only [h1, bind, Except.bind] at h
  · cases h
  rcases h2 : decodeProgram i.program with e | P <;> simp only [h2] at h
  · cases h
  simp only [pure, Except.pure] at h
  by_cases h3 : P.orderViolation.isSome = true
  · simp only [h3, ite_true, throw, throwThe, MonadExceptOf.throw] at h; cases h
  simp only [h3, Bool.false_eq_true, ite_false] at h
  by_cases h4 : (!(Desc.rootLaidOut P && Desc.inputsAtRoot P)) = true
  · simp only [h4, ite_true, throw, throwThe, MonadExceptOf.throw] at h; cases h
  simp only [h4, Bool.false_eq_true, ite_false] at h
  by_cases h4w : (!Desc.walkable P) = true
  · simp only [h4w, ite_true, throw, throwThe, MonadExceptOf.throw] at h; cases h
  simp only [h4w, Bool.false_eq_true, ite_false] at h
  rcases h5 : Desc.analyze P with e | reads <;> simp only [h5] at h
  · cases h
  rcases h6 : nodeThetaFold P i.theta with e | mask <;> simp only [nodeThetaFold, inputs, registered, rootBody, nodeCount] at h6 <;> simp only [h6] at h
  · cases h
  by_cases h7 : (i.owner.size != (P.defs[P.root]!).body.nodes.size) = true
  · simp only [h7, ite_true, throw, throwThe, MonadExceptOf.throw] at h; cases h
  simp only [h7, Bool.false_eq_true, ite_false] at h
  split at h
  · cases h
  rename_i own h8
  replace h8 : nodeOwnerFold P i.owner = .ok own := h8
  by_cases h9 : iuCount own > (P.defs[P.root]!).body.nodes.size
  · simp only [h9, ite_true, throw, throwThe, MonadExceptOf.throw] at h; cases h
  simp only [h9, ite_false] at h
  by_cases h10 : (!(List.range (iuCount own)).all (usedOf own).contains) = true
  · simp only [h10, ite_true, throw, throwThe, MonadExceptOf.throw] at h; cases h
  simp only [h10, Bool.false_eq_true, ite_false, Except.ok.injEq] at h
  refine ⟨i, P, reads, mask, own, rfl, h2, ?_, ?_, ?_, by simpa using h4w, h5, h6, ?_, h8, by simp [nodeCount, rootBody]; omega, ?_, h.symm⟩
  · simpa using h3
  · simp at h4; exact h4.1
  · simp at h4; exact h4.2
  · simpa [nodeCount, rootBody] using h7
  · intro u hu
    simp at h10
    exact h10 u hu

theorem descThetaStep_ok (inp reg m m' : Array Bool) (t : Json) (h : Desc.thetaStep inp reg m t = .ok m') :
    ∃ k, natOf? t = some k ∧ k < inp.size ∧ inp[k]! = true ∧ reg[k]! = true ∧ m[k]! = false ∧
      m' = m.set! k true := by
  unfold Desc.thetaStep at h
  rcases hk : natOf? t with _ | k <;> simp only [hk] at h
  · cases h
  by_cases h1 : k ≥ inp.size
  · simp [h1, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  by_cases h2 : inp[k]! = true
  · by_cases h2' : reg[k]! = true
    · by_cases h3 : m[k]! = true
      · simp [h1, h2, h2', h3, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
      · simp [h1, h2, h2', h3, pure, Except.pure] at h
        exact ⟨k, rfl, by omega, h2, h2', by simpa using h3, h.symm⟩
    · simp [h1, h2, h2', throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · simp [h1, h2, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

theorem getElem!_set!' {α : Type} [Inhabited α] (xs : Array α) (k : ℕ) (x : α) (i : ℕ) (hk : k < xs.size) :
    (xs.set! k x)[i]! = if i = k then x else xs[i]! := by
  by_cases hi : i < xs.size
  · rw [getElem!_pos (xs.set! k x) i (by simpa using hi), getElem!_pos xs i hi]
    simp only [Array.set!_eq_setIfInBounds, Array.getElem_setIfInBounds]
    by_cases h : i = k <;> simp_all [eq_comm]
  · rw [getElem!_neg (xs.set! k x) i (by simpa using hi), getElem!_neg xs i hi]
    by_cases h : i = k
    · omega
    · simp [h]

/-- What θ's fold by node holds: each entry an input node no earlier entry names, and the mask the nodes named. -/
theorem nodeThetaFold_ok (P : Prog) (theta : Array Json) (mask : Array Bool) (h : nodeThetaFold P theta = .ok mask) :
    mask.size = nodeCount P ∧
      (∀ t < theta.size, ∃ k, natOf? theta[t]! = some k ∧ k < nodeCount P ∧ (inputs P)[k]! = true ∧
        (registered P)[k]! = true ∧ ∀ t' < t, natOf? theta[t']! ≠ some k) ∧
      ∀ k < nodeCount P, (mask[k]! = true ↔ ∃ t < theta.size, natOf? theta[t]! = some k) := by
  have hinp : (inputs P).size = nodeCount P := by simp [inputs, nodeCount]
  have := foldlM_range_ok _ _ (fun n (m : Array Bool) => m.size = nodeCount P ∧
      (∀ t < n, ∃ k, natOf? theta[t]! = some k ∧ k < nodeCount P ∧ (inputs P)[k]! = true ∧
        (registered P)[k]! = true ∧ ∀ t' < t, natOf? theta[t']! ≠ some k) ∧
      ∀ k < nodeCount P, (m[k]! = true ↔ ∃ t < n, natOf? theta[t]! = some k)) ?_ ?_ theta.size mask h
  · exact this
  · refine ⟨by simp, fun t ht => absurd ht (by omega), fun k hk => ?_⟩
    simp [getElem!_pos (Array.replicate (nodeCount P) false) k (by simpa using hk)]
  · rintro n m m' ⟨hs, ht, hm⟩ hstep
    obtain ⟨k, hk, hkK, hin, hreg, hfresh, rfl⟩ := descThetaStep_ok _ _ _ _ _ hstep
    rw [hinp] at hkK
    refine ⟨by simp [hs], fun t htn => ?_, fun k' hk' => ?_⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 htn with htn | rfl
      · exact ht t htn
      · refine ⟨k, hk, hkK, hin, hreg, fun t' ht' heq => ?_⟩
        have := (hm k hkK).2 ⟨t', ht', heq⟩
        rw [hfresh] at this
        cases this
    · rw [getElem!_set!' _ _ _ _ (by omega)]
      by_cases hkk : k' = k
      · subst hkk
        simp only [ite_true, true_iff]
        exact ⟨n, by omega, hk⟩
      · simp only [hkk, ite_false, hm k' hk']
        constructor
        · rintro ⟨t, ht', h'⟩; exact ⟨t, by omega, h'⟩
        · rintro ⟨t, ht', h'⟩
          rcases Nat.lt_succ_iff_lt_or_eq.1 ht' with ht' | rfl
          · exact ⟨t, ht', h'⟩
          · rw [hk] at h'; exact absurd (Option.some.inj h').symm hkk

theorem descOwnerStep_ok (inp held : Bool) (e : Json) (o : Option ℕ) (h : Desc.ownerStep inp held e = .ok o) :
    o = (if held then natOf? e else none) ∧
      (held = true → (e = .str "input" ∧ inp = true) ∨ ((∃ u, natOf? e = some u) ∧ inp = false)) := by
  unfold Desc.ownerStep at h
  by_cases ht : isInputTag e = true
  · obtain rfl := (isInputTag_iff e).1 ht
    cases held <;> cases inp <;>
      simp [isInputTag, natOf?_str, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h ⊢ <;>
      exact h.symm
  · have he : e ≠ .str "input" := fun h' => ht ((isInputTag_iff e).2 h')
    rcases hn : natOf? e with _ | u <;> cases held <;> cases inp <;>
      simp [ht, hn, he, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h ⊢ <;>
      exact h.symm

/-- What the owners' fold by node holds: per root node with gates, its IU or `none` for `"input"`, as its kind demands,
and `none` for a node without gates. -/
theorem nodeOwnerFold_ok (P : Prog) (owner : Array Json) (own : Array (Option ℕ)) (h : nodeOwnerFold P owner = .ok own) :
    own.size = nodeCount P ∧ ∀ k < nodeCount P,
      own[k]! = (if (rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0 then natOf? owner[k]! else none) ∧
      ((rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0 →
        (owner[k]! = .str "input" ∧ (inputs P)[k]! = true) ∨
          ((∃ u, natOf? owner[k]! = some u) ∧ (inputs P)[k]! = false)) := by
  refine foldlM_range_ok _ _ (fun n (own : Array (Option ℕ)) => own.size = n ∧ ∀ k < n,
      own[k]! = (if (rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0 then natOf? owner[k]! else none) ∧
      ((rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0 →
        (owner[k]! = .str "input" ∧ (inputs P)[k]! = true) ∨
          ((∃ u, natOf? owner[k]! = some u) ∧ (inputs P)[k]! = false))) ⟨rfl, fun k hk => absurd hk (by omega)⟩
    ?_ _ own h
  rintro n own own' ⟨hs, hk⟩ hstep
  rcases hx : Desc.ownerStep (inputs P)[n]! (decide ((rootOffs P)[n + 1]! - (rootOffs P)[n]! > 0)) owner[n]! with e | o
  · rw [hx] at hstep
    cases hstep
  simp only [hx, Functor.map, Except.map, Except.ok.injEq] at hstep
  subst hstep
  obtain ⟨ho, hkind⟩ := descOwnerStep_ok _ _ _ _ hx
  refine ⟨by simp [hs], fun k hk' => ?_⟩
  rcases Nat.lt_succ_iff_lt_or_eq.1 hk' with hlt | rfl
  · rw [getElem!_push_lt _ _ (by omega)]
    exact hk k hlt
  · have e1 : (own.push o)[own.size]! = o := getElem!_push_size _ _
    rw [hs] at e1
    rw [e1]
    refine ⟨?_, fun hg => hkind (by simpa using hg)⟩
    rw [ho]
    by_cases hg : (rootOffs P)[own.size + 1]! - (rootOffs P)[own.size]! > 0 <;> simp [hg]


/-- What `rootLaidOut` checks. -/
theorem rootLaidOut_spec (P : Prog) (h : Desc.rootLaidOut P = true) :
    (rootOffs P).size = nodeCount P + 1 ∧ (rootOffs P)[0]! = 0 ∧ (rootOffs P)[nodeCount P]! = gates P ∧
      ∀ k < nodeCount P, (rootOffs P)[k]! ≤ (rootOffs P)[k + 1]! := by
  simp only [Desc.rootLaidOut, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

theorem offs_mono (offs : Array ℕ) (K : ℕ) (hmono : ∀ k < K, offs[k]! ≤ offs[k + 1]!) :
    ∀ a b, a ≤ b → b ≤ K → offs[a]! ≤ offs[b]! := by
  intro a b hab hb
  induction b with
  | zero => obtain rfl : a = 0 := by omega
            exact Nat.le_refl _
  | succ b ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 (Nat.lt_succ_of_le hab) with h | h
    · exact Nat.le_trans (ih (by omega) (by omega)) (hmono b (by omega))
    · subst h; exact Nat.le_refl _

/-- Each gate is in one root node. -/
theorem node_unique (offs : Array ℕ) (K : ℕ) (hmono : ∀ k < K, offs[k]! ≤ offs[k + 1]!) {g k k' : ℕ}
    (hk : k < K) (hk' : k' < K) (h1 : offs[k]! ≤ g ∧ g < offs[k + 1]!) (h2 : offs[k']! ≤ g ∧ g < offs[k' + 1]!) :
    k = k' := by
  rcases Nat.lt_trichotomy k k' with h | h | h
  · have := offs_mono offs K hmono (k + 1) k' (by omega) (by omega); omega
  · exact h
  · have := offs_mono offs K hmono (k' + 1) k (by omega) (by omega); omega

theorem node_exists (offs : Array ℕ) (K : ℕ) (h0 : offs[0]! = 0) : ∀ g, g < offs[K]! →
    ∃ k < K, offs[k]! ≤ g ∧ g < offs[k + 1]! := by
  induction K with
  | zero => intro g hg; omega
  | succ K ih =>
    intro g hg
    by_cases h : g < offs[K]!
    · obtain ⟨k, hk, h'⟩ := ih g h
      exact ⟨k, by omega, h'⟩
    · exact ⟨K, by omega, by omega, hg⟩

theorem find?_range_some {n : ℕ} {p : ℕ → Bool} {k : ℕ} (h : (Array.range n).find? p = some k) :
    k < n ∧ p k = true :=
  ⟨Array.mem_range.1 (Array.mem_of_find?_eq_some h), Array.find?_some h⟩

theorem run_inv {α β : Type} (what : String) (step : α → Flock.V (β ⊕ α)) (Q : α → Prop) (R : β → Prop)
    (hR : ∀ a b, Q a → step a = .ok (.inl b) → R b) (hQ : ∀ a a', Q a → step a = .ok (.inr a') → Q a') :
    ∀ n a b, Q a → Flock.Steps.run what step n a = .ok b → R b
  | 0, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, a, b, hq, h => by
    rw [Flock.Steps.run.eq_2] at h
    rcases hs : step a with e | (b' | a')
    · simp [hs, bind, Except.bind] at h
    · simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact h ▸ hR a b' hq hs
    · simp only [hs, bind, Except.bind] at h
      exact run_inv what step Q R hR hQ n a' b (hQ a a' hq hs) h

theorem gateStep_eq (P : Prog) (g : ℕ) (sc : Flock.Program.Scope) (x) (h : Flock.Program.gateStep P g sc = .ok x) :
    ∃ k, (Array.range (P.defs[sc.d]!).body.nodes.size).find? (fun k =>
        (P.defs[sc.d]!).body.offsets[k]! ≤ g - sc.off && g - sc.off < (P.defs[sc.d]!).body.offsets[k + 1]!) = some k ∧
      (((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true ∧
          ∃ ops, x = .inl (((P.defs[sc.d]!).body.nodes[k]!).fn, ops, sc)) ∨
        ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false ∧
          ∃ sc', x = .inr sc' ∧ sc'.d = ((P.defs[sc.d]!).body.nodes[k]!).fn)) := by
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    refine ⟨k, hk, ?_⟩
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · left
      simp only [hp, ite_true] at h
      split at h
      · cases h
      · rename_i ops hops
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨hp, ops, h.symm⟩
    · right
      simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨by simpa using hp, _, h.symm, rfl⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem inputsAtRoot_nodes (P : Prog) (h : Desc.inputsAtRoot P = true) (d k : ℕ)
    (hk : k < (P.defs[d]!).body.nodes.size) :
    ((P.defs[d]!).body.nodes[k]!).fn ≠ P.root ∧ (d = P.root ∨ Desc.isInputNode P ((P.defs[d]!).body.nodes[k]!) = false) := by
  by_cases hd : d < P.defs.size
  · simp only [Desc.inputsAtRoot, List.all_eq_true, List.mem_range, Array.all_eq_true] at h
    have := h d hd k hk
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, Bool.or_eq_true, beq_iff_eq, Bool.not_eq_true'] at this
    rw [getElem!_pos _ k hk]
    exact this
  · rw [getElem!_neg P.defs d hd] at hk
    have h0 : (default : Flock.Program.Defn).body.nodes.size = 0 := rfl
    omega

theorem inputs_get (P : Prog) {k : ℕ} (hk : k < nodeCount P) :
    (inputs P)[k]! = Desc.isInputNode P ((rootBody P).nodes[k]!) := by
  rw [getElem!_pos (inputs P) k (by simp only [inputs, Array.size_map]; exact hk),
    getElem!_pos (rootBody P).nodes k (by exact hk)]
  simp only [inputs, Array.getElem_map]
  rfl

/-- **A root node's gates are input gates exactly when it is an input node.** The gate's primitive is the node's callee
when that is a primitive, and else a primitive some body below the root calls, which `inputsAtRoot` makes no input. -/
theorem gate_input (P : Prog) (hl : Desc.rootLaidOut P = true) (hi : Desc.inputsAtRoot P = true) {g p : ℕ}
    {ops : Array ℕ} {sc : Flock.Program.Scope} (h : P.gate g = .ok (p, ops, sc)) {k : ℕ} (hk : k < nodeCount P)
    (hg : (rootOffs P)[k]! ≤ g ∧ g < (rootOffs P)[k + 1]!) :
    Flock.Program.isInputId (P.defs[p]!).key = (inputs P)[k]! := by
  obtain ⟨-, -, -, hmono⟩ := rootLaidOut_spec P hl
  unfold Flock.Program.Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  have hF : Flock.Steps.FUEL = (Flock.Steps.FUEL - 1) + 1 := by unfold Flock.Steps.FUEL; omega
  rw [hF, Flock.Steps.run.eq_2] at h
  generalize Flock.Steps.FUEL - 1 = fuel at h
  rw [inputs_get P hk]
  rcases hs : Flock.Program.gateStep P g P.rootScope with e | x
  · simp [hs, bind, Except.bind] at h
  obtain ⟨k0, hf, hx⟩ := gateStep_eq P g P.rootScope x hs
  obtain ⟨hk0, hp0⟩ := find?_range_some hf
  have hd : P.rootScope.d = P.root := rfl
  have ho : P.rootScope.off = 0 := rfl
  simp only [hd, ho, Nat.sub_zero, Bool.and_eq_true, decide_eq_true_eq] at hk0 hp0
  have hk0K : k0 < nodeCount P := hk0
  obtain rfl := node_unique (rootOffs P) (nodeCount P) hmono hk0K hk hp0 hg
  simp only [hd] at hx
  rcases hx with ⟨hprim, ops', rfl⟩ | ⟨hprim, sc', rfl, hsc'⟩
  · simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -, -⟩ := h
    simp [Desc.isInputNode, rootBody, hprim]
  · simp only [hs, bind, Except.bind] at h
    have hnot : Desc.isInputNode P ((rootBody P).nodes[k0]!) = false := by
      simp [Desc.isInputNode, rootBody, hprim]
    rw [hnot]
    refine run_inv "gate location" (Flock.Program.gateStep P g) (fun sc => sc.d ≠ P.root)
      (fun b => Flock.Program.isInputId (P.defs[b.1]!).key = false) ?_ ?_ fuel sc' (p, ops, sc) ?_ h
    · intro a b ha hb
      obtain ⟨k', hf', hx'⟩ := gateStep_eq P g a _ hb
      obtain ⟨hk', -⟩ := find?_range_some hf'
      rcases hx' with ⟨hprim', ops', hb'⟩ | ⟨-, _, hb', -⟩
      · cases hb'
        rcases (inputsAtRoot_nodes P hi a.d k' hk').2 with hr | hr
        · exact absurd hr ha
        · simpa [Desc.isInputNode, hprim'] using hr
      · cases hb'
    · intro a a' ha ha'
      obtain ⟨k', hf', hx'⟩ := gateStep_eq P g a _ ha'
      obtain ⟨hk', -⟩ := find?_range_some hf'
      rcases hx' with ⟨-, _, hb'⟩ | ⟨-, sc'', hb', hd'⟩
      · cases hb'
      · cases hb'
        rw [hd']
        exact (inputsAtRoot_nodes P hi a.d k' hk').1
    · rw [hsc']
      exact (inputsAtRoot_nodes P hi P.root k0 hk0K).1

/-- An input node's gates are its callee's: `Prog.gate` stops at a primitive callee. -/
theorem gate_input_fn (P : Prog) (hl : Desc.rootLaidOut P = true) {g p : ℕ}
    {ops : Array ℕ} {sc : Flock.Program.Scope} (h : P.gate g = .ok (p, ops, sc)) {k : ℕ} (hk : k < nodeCount P)
    (hg : (rootOffs P)[k]! ≤ g ∧ g < (rootOffs P)[k + 1]!) (hin : (inputs P)[k]! = true) :
    p = ((rootBody P).nodes[k]!).fn := by
  obtain ⟨-, -, -, hmono⟩ := rootLaidOut_spec P hl
  rw [inputs_get P hk] at hin
  unfold Flock.Program.Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  have hF : Flock.Steps.FUEL = (Flock.Steps.FUEL - 1) + 1 := by unfold Flock.Steps.FUEL; omega
  rw [hF, Flock.Steps.run.eq_2] at h
  generalize Flock.Steps.FUEL - 1 = fuel at h
  rcases hs : Flock.Program.gateStep P g P.rootScope with e | x
  · simp [hs, bind, Except.bind] at h
  obtain ⟨k0, hf, hx⟩ := gateStep_eq P g P.rootScope x hs
  obtain ⟨hk0, hp0⟩ := find?_range_some hf
  have hd : P.rootScope.d = P.root := rfl
  have ho : P.rootScope.off = 0 := rfl
  simp only [hd, ho, Nat.sub_zero, Bool.and_eq_true, decide_eq_true_eq] at hk0 hp0
  have hk0K : k0 < nodeCount P := hk0
  have hkk : k0 = k := node_unique (rootOffs P) (nodeCount P) hmono hk0K hk hp0 hg
  subst hkk
  simp only [hd] at hx
  rcases hx with ⟨hprim, ops', rfl⟩ | ⟨hprim, sc', rfl, hsc'⟩
  · simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    exact h.1.symm
  · simp [Desc.isInputNode, rootBody, hprim] at hin

theorem registered_get (P : Prog) {k : ℕ} (hk : k < nodeCount P) :
    (registered P)[k]! = isRegisteredId (P.defs[((rootBody P).nodes[k]!).fn]!).key := by
  rw [getElem!_pos (registered P) k (by simp only [registered, Array.size_map]; exact hk),
    getElem!_pos (rootBody P).nodes k (by exact hk)]
  simp only [registered, Array.getElem_map]
  rfl

/-- θ's gates: each entry's node's, in order. -/
def thetaGates (P : Prog) (theta : Array Json) : List ℕ :=
  theta.toList.flatMap fun t => ((natOf? t).map (nodeGates P)).getD []

theorem mem_nodeGates (P : Prog) (k g : ℕ) :
    g ∈ nodeGates P k ↔ (rootOffs P)[k]! ≤ g ∧ g < (rootOffs P)[k + 1]! := by
  simp only [nodeGates, List.mem_range'_1, rootOffs, rootBody]
  omega

/-- **The case `Desc.isInputNode` once got wrong.** A root node that calls a composite holds no input gate, whatever the
composite's key. `Input16_v1` is a well-formed key of a static-free composite, which the decoder admits, and
`Desc.isInputNode` read an input node off the key alone, so it admitted such a node's computed gates to θ and to no IU,
where `evaluate` refuses them (`theta-not-input`, `gate-in-no-iu`). The difftest's last six vectors are such programs. -/
theorem call_gates_computed (P : Prog) (hl : Desc.rootLaidOut P = true) (hi : Desc.inputsAtRoot P = true) {k : ℕ}
    (hk : k < nodeCount P) (hprim : (P.defs[((rootBody P).nodes[k]!).fn]!).prim = false) {g : ℕ}
    (hg : g ∈ nodeGates P k) : ¬ IsInput P g := by
  rintro ⟨p, ops, sc, hgate, hp⟩
  rw [gate_input P hl hi hgate hk ((mem_nodeGates P k g).1 hg), inputs_get P hk] at hp
  simp [Desc.isInputNode, hprim] at hp

theorem mem_thetaGates (P : Prog) (theta : Array Json) (g : ℕ) :
    g ∈ thetaGates P theta ↔ ∃ t < theta.size, ∃ k, natOf? theta[t]! = some k ∧ g ∈ nodeGates P k := by
  simp only [thetaGates, List.mem_flatMap, Array.mem_toList_iff]
  constructor
  · rintro ⟨a, ha, hg⟩
    obtain ⟨t, ht, rfl⟩ := Array.mem_iff_getElem.1 ha
    rcases hk : natOf? theta[t] with _ | k <;> rw [hk] at hg
    · simp at hg
    · exact ⟨t, ht, k, by rw [getElem!_pos theta t ht]; exact hk, by simpa using hg⟩
  · rintro ⟨t, ht, k, hk, hg⟩
    rw [getElem!_pos theta t ht] at hk
    exact ⟨theta[t], Array.getElem_mem ht, by rw [hk]; simpa using hg⟩

theorem thetaGates_nodup (P : Prog) (theta : Array Json)
    (hmono : ∀ k < nodeCount P, (rootOffs P)[k]! ≤ (rootOffs P)[k + 1]!)
    (hθ : ∀ t < theta.size, ∃ k, natOf? theta[t]! = some k ∧ k < nodeCount P ∧ ∀ t' < t, natOf? theta[t']! ≠ some k) :
    (thetaGates P theta).Nodup := by
  unfold thetaGates List.Nodup
  rw [List.pairwise_flatMap]
  refine ⟨fun a _ => ?_, ?_⟩
  · rcases natOf? a with _ | k
    · simp
    · exact List.nodup_range' (s := (P.defs[P.root]!).body.offsets[k]!) 1
  · rw [List.pairwise_iff_getElem]
    intro i j hi hj hij x hx y hy hxy
    subst hxy
    simp only [Array.length_toList] at hi hj
    simp only [Array.getElem_toList] at hx hy
    obtain ⟨kj, hkj, hkjK, hfresh⟩ := hθ j hj
    obtain ⟨ki, hki, hkiK, -⟩ := hθ i hi
    rw [getElem!_pos theta j hj] at hkj
    rw [getElem!_pos theta i hi] at hki
    rw [hki] at hx
    rw [hkj] at hy
    have hx' := (mem_nodeGates P ki x).1 (by simpa using hx)
    have hy' := (mem_nodeGates P kj x).1 (by simpa using hy)
    obtain rfl := node_unique (rootOffs P) (nodeCount P) hmono hkiK hkjK hx' hy'
    exact hfresh i hij (by rw [getElem!_pos theta i hi]; exact hki)

/-- An expansion's entries are θ's gates. -/
theorem expands_get {theta' : Array Json} {M : List ℕ} (h : theta'.toList.map natOf? = M.map some) :
    theta'.size = M.length ∧ ∀ t (ht : t < theta'.size), natOf? theta'[t]! = some (M[t]'(by
      have := congrArg List.length h; simp at this; omega)) := by
  have hl : theta'.size = M.length := by have := congrArg List.length h; simpa using this
  refine ⟨hl, fun t ht => ?_⟩
  have := congrArg (fun l => l[t]?) h
  simp only [List.getElem?_map, Array.getElem?_toList] at this
  rw [getElem!_pos theta' t ht]
  rw [Array.getElem?_eq_getElem ht, List.getElem?_eq_getElem (by omega)] at this
  simpa using this

theorem rep_length {α : Type} (offs : Array ℕ) (x : ℕ → α) (h0 : offs[0]! = 0) : ∀ K,
    (∀ k < K, offs[k]! ≤ offs[k + 1]!) →
      ((List.range K).flatMap fun k => List.replicate (offs[k + 1]! - offs[k]!) (x k)).length = offs[K]!
  | 0, _ => by simp [h0]
  | K + 1, hmono => by
    rw [List.range_succ, List.flatMap_append, List.length_append,
      rep_length offs x h0 K (fun k hk => hmono k (by omega))]
    have := hmono K (by omega)
    simp
    omega

theorem rep_get {α : Type} (offs : Array ℕ) (x : ℕ → α) (h0 : offs[0]! = 0) : ∀ K,
    (∀ k < K, offs[k]! ≤ offs[k + 1]!) → ∀ k < K, ∀ g, offs[k]! ≤ g → g < offs[k + 1]! →
      ((List.range K).flatMap fun k => List.replicate (offs[k + 1]! - offs[k]!) (x k))[g]? = some (x k)
  | 0, _, k, hk, _, _, _ => absurd hk (by omega)
  | K + 1, hmono, k, hk, g, h1, h2 => by
    have hm : ∀ k < K, offs[k]! ≤ offs[k + 1]! := fun k hk => hmono k (by omega)
    have hlen := rep_length offs x h0 K hm
    rw [List.range_succ, List.flatMap_append]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
    · have hle := offs_mono offs K hm (k + 1) K (by omega) (by omega)
      rw [List.getElem?_append_left (by omega)]
      exact rep_get offs x h0 K hm k hk g h1 h2
    · rw [List.getElem?_append_right (by omega)]
      simp only [hlen, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.getElem?_replicate]
      simp only [show g - offs[k]! < offs[k + 1]! - offs[k]! by omega, ite_true]


/-! ## The expansion -/

theorem getElem!_of_getElem? {α : Type} [Inhabited α] {xs : Array α} {g : ℕ} {v : α} (h : xs[g]? = some v) :
    xs[g]! = v := by
  have hg : g < xs.size := by
    rcases Nat.lt_or_ge g xs.size with hg | hg
    · exact hg
    · rw [Array.getElem?_eq_none hg] at h; cases h
  rw [getElem!_pos xs g hg]
  rw [Array.getElem?_eq_getElem hg] at h
  exact Option.some.inj h

theorem natOf?_toJson (n : ℕ) : natOf? (toJson n) = some n := by
  simp [natOf?, toJson, Lean.JsonNumber.fromNat]

/-- θ expanded: θ's gates as JSON. -/
def thetaOf (P : Prog) (theta : Array Json) : Array Json := ((thetaGates P theta).map fun g => toJson g).toArray

/-- The owners expanded: each root node's owner once per gate of it. -/
def ownerOf' (P : Prog) (owner : Array Json) : Array Json :=
  ((List.range (nodeCount P)).flatMap fun k =>
    List.replicate ((rootOffs P)[k + 1]! - (rootOffs P)[k]!) owner[k]!).toArray

end DescAgree

open DescAgree
open Lean (Json toJson)
open Flock.Program (Prog)

/-! ## The statements `descAgrees_of` takes -/

/-- **The layout is located.** On a Program the decoder returns that `Desc.evaluate` reads (topological, its root
laid out, its input nodes the root's, `Desc.walkable`, and its references resolved by `Desc.analyze`), every gate of the
canonical layout is located and reads only earlier gates. Without `Desc.walkable` it is false: `Prog.gate` walks at most
`Flock.Steps.FUEL` steps to an operand, and a nested scan of `2^32 × 2^32` identity steps needs `2^64`. -/
def GatesLocated : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    Desc.rootLaidOut P = true → Desc.inputsAtRoot P = true → Desc.walkable P = true →
    (∃ reads, Desc.analyze P = .ok reads) → ∀ g < gates P, GateOk P g

/-- **The novel reads by node are the gates'.** On such a Program, and a division `dv` of its layout whose IUs are
unions of root nodes (root node `k`'s gates all in `own[k]`, θ every gate of the nodes `mask` marks, input nodes, in no
IU), `Desc.novelWidth` of what `Desc.intoOf` collects per IU from `Desc.analyze`'s reads is `NciCheck.ius`'s novel
reads. -/
def NovelExact : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (reads : Array (Array Desc.Seg)) (mask : Array Bool) (own : Array (Option ℕ))
    (c : Circuit) (dv : Division),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → Desc.analyze P = .ok reads → circuitOf P = .ok c →
    mask.size = nodeCount P → own.size = nodeCount P → dv.owner.size = gates P →
    (∀ k < nodeCount P, mask[k]! = true → (inputs P)[k]! = true) →
    (∀ k < nodeCount P, own[k]! ≠ none → (inputs P)[k]! = false ∧ nodeGates P k ≠ []) →
    (∀ k < nodeCount P, ∀ g ∈ nodeGates P k, dv.owner[g]! = own[k]! ∧ (g ∈ dv.theta ↔ mask[k]! = true)) →
    dv.n = iuCount own →
    ∀ u < dv.n, Desc.novelWidth (Desc.intoOf (rootOffs P) own mask reads dv.n)[u]! = (ius c dv)[u]!.2

theorem iuCount_eq (a b : Array (Option ℕ))
    (h : ∀ u, (∃ g < a.size, a[g]! = some u) ↔ (∃ k < b.size, b[k]! = some u)) : iuCount a = iuCount b := by
  obtain ⟨ha1, ha2⟩ := iuCount_spec a
  obtain ⟨hb1, hb2⟩ := iuCount_spec b
  have hab : iuCount a ≤ iuCount b := by
    rcases ha2 with h0 | ⟨g, hg, hga⟩
    · omega
    · obtain ⟨k, hk, hkb⟩ := (h _).1 ⟨g, hg, hga⟩
      have := hb1 k hk _ hkb
      omega
  have hba : iuCount b ≤ iuCount a := by
    rcases hb2 with h0 | ⟨k, hk, hkb⟩
    · omega
    · obtain ⟨g, hg, hga⟩ := (h _).2 ⟨k, hk, hkb⟩
      have := ha1 g hg _ hga
      omega
  omega

theorem ius_fst (c : Circuit) (dv : Division) {u : ℕ} (hu : u < dv.n) : (ius c dv)[u]!.1 = (tally c dv)[u]!.1 := by
  have hs := (tally_spec c dv).1
  rw [getElem!_pos (ius c dv) u (by simp [ius, hs, hu]), getElem!_pos (tally c dv) u (by rw [hs]; exact hu)]
  simp [ius]

/-- **`Desc.evaluate` agrees with `evaluate`, given `GatesLocated` and `NovelExact`.** -/
theorem descAgrees_of (hG : GatesLocated) (hN : NovelExact) : Properties.DescAgrees := by
  intro j out hev
  obtain ⟨i, P, reads, mask, own, hpar, hdp, hov, hlo, hin, hwk, han, hθ, hos, hof, hn, hused, rfl⟩ :=
    descEvaluate_ok j out hev
  have hdec := (decodeProgram_ok _ _).1 hdp
  have hi : Parses j i := (parse_ok j i).1 hpar
  have hgl := hG _ P hdec hov hlo hin hwk ⟨reads, han⟩
  obtain ⟨c, hc⟩ := (circuitOf_isOk P).2 hgl
  obtain ⟨hsz, h0, hK, hmono⟩ := rootLaidOut_spec P hlo
  obtain ⟨hms, hθt, hθm⟩ := nodeThetaFold_ok P i.theta mask hθ
  obtain ⟨hown_s, hown⟩ := nodeOwnerFold_ok P i.owner own hof
  have hng : ∀ k g, g ∈ nodeGates P k ↔ (rootOffs P)[k]! ≤ g ∧ g < (rootOffs P)[k + 1]! := mem_nodeGates P
  -- a root node's gates: in the circuit, and input gates exactly when it is an input node
  have hcls : ∀ k < nodeCount P, ∀ g ∈ nodeGates P k, g < gates P ∧ (IsInput P g ↔ (inputs P)[k]! = true) := by
    intro k hk g hg
    have hg' := (hng k g).1 hg
    have hle := offs_mono (rootOffs P) (nodeCount P) hmono (k + 1) (nodeCount P) (by omega) (by omega)
    have hlt : g < gates P := by rw [← hK]; omega
    refine ⟨hlt, ?_, fun hk' => ?_⟩
    · rintro ⟨p, ops, sc, hgate, hp⟩
      rw [← gate_input P hlo hin hgate hk hg']; exact hp
    · obtain ⟨p, ops, w, ⟨⟨sc, hgate⟩, -, -⟩, -⟩ := hgl g hlt
      exact ⟨p, ops, sc, hgate, by rw [gate_input P hlo hin hgate hk hg']; exact hk'⟩
  have hnode : ∀ g < gates P, ∃ k < nodeCount P, g ∈ nodeGates P k := by
    intro g hg
    obtain ⟨k, hk, h'⟩ := node_exists (rootOffs P) (nodeCount P) h0 g (by rw [hK]; exact hg)
    exact ⟨k, hk, (hng k g).2 h'⟩
  have hheld : ∀ k g, g ∈ nodeGates P k → (rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0 := by
    intro k g hg; have := (hng k g).1 hg; omega
  have hownk : ∀ k < nodeCount P, ∀ g ∈ nodeGates P k, own[k]! = natOf? i.owner[k]! := by
    intro k hk g hg
    rw [(hown k hk).1, if_pos (hheld k g hg)]
  -- the nodes θ names, and θ's gates without repeats
  have hθ' : ∀ t < i.theta.size, ∃ k, natOf? i.theta[t]! = some k ∧ k < nodeCount P ∧
      ∀ t' < t, natOf? i.theta[t']! ≠ some k := fun t ht => by
    obtain ⟨k, h1, h2, -, -, h4⟩ := hθt t ht
    exact ⟨k, h1, h2, h4⟩
  have hnd := thetaGates_nodup P i.theta hmono hθ'
  have hmemθ : ∀ k < nodeCount P, ∀ g ∈ nodeGates P k, (g ∈ thetaGates P i.theta ↔ mask[k]! = true) := by
    intro k hk g hg
    rw [mem_thetaGates, hθm k hk]
    constructor
    · rintro ⟨t, ht, k', hk', hg'⟩
      obtain ⟨k'', h1, h2, -, -, -⟩ := hθt t ht
      rw [hk'] at h1
      obtain rfl := Option.some.inj h1
      obtain rfl := node_unique (rootOffs P) (nodeCount P) hmono h2 hk ((hng _ g).1 hg') ((hng k g).1 hg)
      exact ⟨t, ht, hk'⟩
    · rintro ⟨t, ht, hk'⟩
      exact ⟨t, ht, k, hk', hg⟩
  -- on every expansion, `evaluate` gives the verdict
  have hagree : ∀ j', Expands j j' → evaluate j' = .ok (verdict (P.defs[P.root]!).gates
        ((Array.range (iuCount own)).map fun u => ((Desc.sizes (rootOffs P) own (iuCount own))[u]!,
          Desc.novelWidth (Desc.intoOf (rootOffs P) own mask reads (iuCount own))[u]!))
        (failure i.gamma i.num i.den ((Array.range (iuCount own)).map fun u =>
          ((Desc.sizes (rootOffs P) own (iuCount own))[u]!,
            Desc.novelWidth (Desc.intoOf (rootOffs P) own mask reads (iuCount own))[u]!)))) := by
    rintro j' ⟨i2, P2, hpar2, hdec2, theta', owner', hpar', hET, hEO⟩
    have hii : i2 = i := parses_unique hpar2 hi
    subst i2
    rw [hdec, Except.ok.injEq] at hdec2
    subst P2
    have hET' : theta'.toList.map natOf? = (thetaGates P i.theta).map some := hET
    obtain ⟨hlen, hget⟩ := expands_get hET'
    have hT : ∀ t < theta'.size, ThetaOk P theta' t := by
      intro t ht
      have hm : (thetaGates P i.theta)[t]'(by omega) ∈ thetaGates P i.theta := List.getElem_mem _
      obtain ⟨t0, ht0, k, hk, hg⟩ := (mem_thetaGates P i.theta _).1 hm
      obtain ⟨k', h1, h2, h3, hr, -⟩ := hθt t0 ht0
      rw [hk] at h1
      obtain rfl := Option.some.inj h1
      obtain ⟨hlt, hinp⟩ := hcls k h2 _ hg
      have hreg : IsRegistered P ((thetaGates P i.theta)[t]'(by omega)) := by
        obtain ⟨p, ops, w, ⟨⟨sc, hgate⟩, -, -⟩, -⟩ := hgl _ hlt
        have hp := gate_input_fn P hlo hgate h2 ((hng _ _).1 hg) h3
        refine ⟨p, ops, sc, hgate, (isRegisteredId_iff _).1 ?_⟩
        rw [hp, ← registered_get P h2]
        exact hr
      refine ⟨_, hget t ht, hlt, hinp.2 h3, hreg, fun t' ht' heq => ?_⟩
      rw [hget t' (by omega), Option.some.injEq] at heq
      exact absurd ((List.Nodup.getElem_inj hnd).1 heq) (by omega)
    have hO : ∀ g < gates P, OwnerOk P owner' g := by
      intro g hg
      obtain ⟨k, hk, hgk⟩ := hnode g hg
      obtain ⟨hin', hnat⟩ := hEO.2 k (by rw [hos]; exact hk) g hgk
      have hinp := (hcls k hk g hgk).2
      rcases (hown k hk).2 (hheld k g hgk) with ⟨he, hik⟩ | ⟨⟨u, hu⟩, hik⟩
      · exact Or.inl ⟨hin'.2 he, hinp.2 hik⟩
      · refine Or.inr ⟨⟨u, by rw [hnat]; exact hu⟩, fun h' => ?_⟩
        rw [hinp.1 h'] at hik; cases hik
    have hE : ¬ EmptyIU owner' := by
      rintro ⟨g, u, hg, hu, u', hu', hall⟩
      rw [hEO.1] at hg
      obtain ⟨k, hk, hgk⟩ := hnode g hg
      rw [(hEO.2 k (by rw [hos]; exact hk) g hgk).2, ← hownk k hk g hgk] at hu
      have hub := (iuCount_spec own).1 k (by rw [hown_s]; exact hk) u hu
      obtain ⟨k', hk', hk'u⟩ := (usedOf_mem own u').1 (hused u' (by omega))
      rw [hown_s] at hk'
      have hk'h : (rootOffs P)[k' + 1]! - (rootOffs P)[k']! > 0 := by
        by_contra hz
        rw [(hown k' hk').1, if_neg hz] at hk'u
        cases hk'u
      rw [(hown k' hk').1, if_pos hk'h] at hk'u
      have hg' : (rootOffs P)[k']! ∈ nodeGates P k' := (hng k' _).2 ⟨Nat.le_refl _, by omega⟩
      have := hall (rootOffs P)[k']! (by rw [hEO.1]; exact (hcls k' hk' _ hg').1)
      rw [(hEO.2 k' (by rw [hos]; exact hk') _ hg').2] at this
      exact this hk'u
    obtain ⟨dv, hdv⟩ := (divisionOf_isOk hc theta' owner').2 ⟨hT, hEO.1, hO, hE⟩
    obtain ⟨-, hdθ, -, -, hdsz, hdown, hdn, -⟩ := divisionOf_ok hc hdv
    have hdvo : ∀ k < nodeCount P, ∀ g ∈ nodeGates P k, dv.owner[g]! = own[k]! := by
      intro k hk g hgk
      rw [hdown g (hcls k hk g hgk).1, (hEO.2 k (by rw [hos]; exact hk) g hgk).2, hownk k hk g hgk]
    have hneq : dv.n = iuCount own := by
      rw [hdn]
      apply iuCount_eq
      intro u
      constructor
      · rintro ⟨g, hg, hgu⟩
        rw [hdsz] at hg
        obtain ⟨k, hk, hgk⟩ := hnode g hg
        exact ⟨k, by rw [hown_s]; exact hk, by rw [← hdvo k hk g hgk]; exact hgu⟩
      · rintro ⟨k, hk, hku⟩
        rw [hown_s] at hk
        have hk'h : (rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0 := by
          by_contra hz
          rw [(hown k hk).1, if_neg hz] at hku
          cases hku
        have hg' : (rootOffs P)[k]! ∈ nodeGates P k := (hng k _).2 ⟨Nat.le_refl _, by omega⟩
        exact ⟨_, by rw [hdsz]; exact (hcls k hk _ hg').1, by rw [hdvo k hk _ hg']; exact hku⟩
    have hnov := hN _ P reads mask own c dv hdec hov hlo hin hwk han hc hms hown_s hdsz
      (fun k hk hm => by
        obtain ⟨t, ht, htk⟩ := (hθm k hk).1 hm
        obtain ⟨k', h1, -, h3, -⟩ := hθt t ht
        rw [htk] at h1
        obtain rfl := Option.some.inj h1
        exact h3)
      (fun k hk hne => by
        have hk'h : (rootOffs P)[k + 1]! - (rootOffs P)[k]! > 0 := by
          by_contra hz
          rw [(hown k hk).1, if_neg hz] at hne
          exact hne rfl
        refine ⟨?_, fun he => ?_⟩
        · rcases (hown k hk).2 hk'h with ⟨he, -⟩ | ⟨-, hik⟩
          · rw [(hown k hk).1, if_pos hk'h, he, natOf?_str] at hne
            exact absurd rfl hne
          · exact hik
        · have hg' : (rootOffs P)[k]! ∈ nodeGates P k := (hng k _).2 ⟨Nat.le_refl _, by omega⟩
          rw [he] at hg'
          cases hg')
      (fun k hk g hgk => ⟨hdvo k hk g hgk, by
        rw [hdθ, ← hmemθ k hk g hgk]
        constructor
        · rintro ⟨t, ht, hgt⟩
          rw [hget t ht, Option.some.injEq] at hgt
          rw [← hgt]
          exact List.getElem_mem _
        · intro hm
          obtain ⟨t, ht, rfl⟩ := List.getElem_of_mem hm
          exact ⟨t, by omega, hget t (by omega)⟩⟩)
      hneq
    rw [(evaluate_ok j' _).2 ⟨_, P, c, dv, (parse_ok j' _).2 hpar', hdp, hc, hdv, rfl⟩]
    have hr : (Array.range (iuCount own)).map (fun u => ((Desc.sizes (rootOffs P) own (iuCount own))[u]!,
        Desc.novelWidth (Desc.intoOf (rootOffs P) own mask reads (iuCount own))[u]!)) = ius c dv := by
      apply Array.ext
      · simp [ius, (tally_spec c dv).1, hneq]
      · intro u hu1 hu2
        have hu : u < dv.n := by simpa [hneq] using hu1
        rw [Array.getElem_map, Array.getElem_range, ← getElem!_pos (ius c dv) u hu2]
        apply Prod.ext
        · rw [ius_fst c dv hu, ← hneq]
          exact sizes_eq_tally c dv (rootOffs P) own h0 (by rw [hown_s, hK, (circuitOf_ok P c hc).1])
            (by rw [hown_s]; exact hmono)
            (fun k hk g h1 h2 => hdvo k (by rw [← hown_s]; exact hk) g ((hng k g).2 ⟨h1, h2⟩)) u hu
        · rw [← hneq]
          exact hnov u hu
    rw [hr, (circuitOf_ok P c hc).1]
    rfl
  refine ⟨?_, hagree⟩
  -- an expansion exists
  obtain ⟨-, -, -, ⟨gv, hgv, hgn⟩, ⟨a, b, hab, ha, hb⟩, hden⟩ := id hi
  refine ⟨Json.mkObj [("program", .str i.program), ("theta", .arr (thetaOf P i.theta)),
      ("owner", .arr (ownerOf' P i.owner)), ("gamma", gv), ("alpha", .arr #[a, b])], i, P, hi, hdec,
    thetaOf P i.theta, ownerOf' P i.owner, ⟨rfl, rfl, rfl, ⟨gv, rfl, hgn⟩, ⟨a, b, rfl, ha, hb⟩, hden⟩, ?_, ?_, ?_⟩
  · show (thetaOf P i.theta).toList.map natOf? = (thetaGates P i.theta).map some
    simp [thetaOf, List.map_map, Function.comp_def, natOf?_toJson]
  · simp only [ownerOf', List.size_toArray]
    rw [rep_length _ _ h0 _ hmono, hK]
  · intro k hk g hgk
    rw [hos] at hk
    have hx := rep_get (rootOffs P) (fun k => i.owner[k]!) h0 (nodeCount P) hmono k hk g ((hng k g).1 hgk).1
      ((hng k g).1 hgk).2
    have : (ownerOf' P i.owner)[g]! = i.owner[k]! := getElem!_of_getElem? (by
      simp only [ownerOf', List.getElem?_toArray]; exact hx)
    rw [this]
    exact ⟨Iff.rfl, rfl⟩

end NciCheck.Properties.SecurityProofs
