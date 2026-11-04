import Proofs.Properties.NetworkScan.Causal

/-!
# The member-precise walk lemma

`Proofs.Properties.NetworkScan.Walk`'s `memberReads`: a gate of member `m` of a root node `k`, a batch or a call of a
composite, reads gates of that member's activation, or a gate some reference of node `k`'s arguments resolves to from
the root. Here that reference is member `m`'s own (`MemberSliceReads`). Leaving the activation up through leaf `j` of
the callee's parameter `i`, resolution reads the argument's reference at `upIdx node m i j`: the member's index
(`Resolve.memberIdx`) for a batch, `j` for a call, with `j` below the parameter's leaves. On an argument mapped over its
first axis that index is `m·L + j` (`memberIdx_slice`), so the members of one batch read disjoint slices of it, which a
wire between two members of one template needs.

* `step_slice`: `WalkProof.step_below`, with that reference where a step leaves the activation;
* `run_slice`: resolution from below the activation ends in its gates, or at the resolution of such a reference;
* `memberSliceReads`: with `WalkProof.gate_below`, the lemma;
* `input_gate`, `arg_gate`: leaf `t` of a batch input node `q` is gate `offsets[q] + t`, a primitive returning one
  leaf, so the members' slices of such a node are disjoint gates;
* `lands_member`: an out leaf of a member of a batch resolves into `InM` at that member;
* `act_node`, `act_not_input`: a member's activation lies in its node's gates, so no input node's gate is in it;
* `InM`, `closure_member`, `depends_member`: at a batch whose every argument is one input node's leaves
  (`NetworkCheck.Period.inputArg`), the gates of member `m`'s activation and those its own references resolve to hold
  what they read in a flattening (`Properties.NetworkScan.Flattens`), so whatever such a gate depends on is among them.
-/

namespace Properties.NetworkCertifier

open Flock.Program
open NciCheck.Properties.SecurityProofs

/-- The index of argument `i`'s reference that member `m` of node `nd` reads through leaf `j` of its callee's
parameter `i` (`Flock.Program.resolveStep`'s step up from a batch's or a call's member): the member's index into the
argument (`Resolve.memberIdx`) for a batch, else `j`. -/
def upIdx (nd : Node) (m i j : ℕ) : ℕ :=
  match nd.form with
  | .batch => Resolve.memberIdx (nd.args[i]!) m j
  | _ => j

/-- **The member-precise walk lemma.** In a Program C-Flock decodes, topological and laid out, a gate `g` in the
activation of member `m` of a root node `k` that calls a composite by a batch or a call reads (its operands,
`Flock.Program.Prog.gate`) only gates of that activation, or the gate member `m`'s own reference of an argument of node
`k` resolves to from the root: argument `i`'s reference at `upIdx`, for a leaf `j` of the callee's parameter `i`. -/
def MemberSliceReads : Prop :=
  ∀ (bytes : ByteArray) (P : Flock.Program.Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    NciCheck.Desc.bodiesLaidOut P = true →
    ∀ k < (P.defs[P.root]!).body.nodes.size, ∀ m < ((P.defs[P.root]!).body.nodes[k]!).members,
      (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false →
      ((P.defs[P.root]!).body.nodes[k]!).form ≠ .scan →
      ∀ g p ops sc, P.gate g = .ok (p, ops, sc) → (P.child P.rootScope k m).off ≤ g →
        g < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates →
        ∀ a ∈ ops,
          ((P.child P.rootScope k m).off ≤ a ∧
            a < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) ∨
          ∃ i j : ℕ, i < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params.size ∧
            j < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params[i]!).2.leaves ∧
            Flock.Program.resolve P P.rootScope ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
              (upIdx ((P.defs[P.root]!).body.nodes[k]!) m i j)) = .ok a

/-- The gates a gate of member `m` of root node `k` depends on stay in: its activation's, and those member `m`'s own
references of node `k`'s arguments resolve to from the root. -/
abbrev InM (P : Prog) (k m x : ℕ) : Prop :=
  ((P.child P.rootScope k m).off ≤ x ∧
    x < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) ∨
  ∃ i j : ℕ, i < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params.size ∧
    j < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params[i]!).2.leaves ∧
    resolve P P.rootScope ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
      (upIdx ((P.defs[P.root]!).body.nodes[k]!) m i j)) = .ok x

end Properties.NetworkCertifier

namespace Properties.NetworkCertifier.MemberSliceProof

open Flock.Program
open NciCheck.Properties.SecurityProofs
open NetworkCheck.Period
open Properties.NetworkScan
open Properties.NetworkScan.WalkProof
open Properties.NetworkScan.CausalProof

theorem upIdx_batch {nd : Node} (h : nd.form = .batch) (m i j : ℕ) :
    upIdx nd m i j = Resolve.memberIdx (nd.args[i]!) m j := by
  simp only [upIdx, h]

theorem upIdx_other {nd : Node} (h : nd.form = .prim ∨ nd.form = .call) (m i j : ℕ) : upIdx nd m i j = j := by
  rcases h with h | h <;> simp only [upIdx, h]

theorem memberRef_upIdx {nd : Node} (h : nd.form = .batch) (m i j : ℕ) :
    (nd.args[i]!).memberRef m j = (nd.args[i]!).seq.get (upIdx nd m i j) := by
  rw [upIdx_batch h, Resolve.memberRef_eq]

theorem get_upIdx {nd : Node} (h : nd.form = .prim ∨ nd.form = .call) (m i j : ℕ) :
    (nd.args[i]!).seq.get j = (nd.args[i]!).seq.get (upIdx nd m i j) := by
  rw [upIdx_other h]

/-- **Member `m`'s slice**: on an argument mapped over its first axis, `(d, L)` its `dims`, member `m` reads leaf
`m·L + j` through leaf `j < L` of its parameter. -/
theorem memberIdx_slice {a : Arg} {d L : ℕ} (hd : a.dims = some (d, L)) {m j : ℕ} (hj : j < L) :
    Resolve.memberIdx a m j = m * L + j := by
  simp only [Resolve.memberIdx, hd, Nat.div_eq_of_lt hj, Nat.mod_eq_of_lt hj, Nat.zero_mul, Nat.zero_add]

/-- **Leaf `t` of a batch input node `q` is its gate `offsets[q] + t`**, its primitive returning one leaf. -/
theorem input_gate (P : Prog) {q t : ℕ} (hq1 : (P.defs[((P.defs[P.root]!).body.nodes[q]!).fn]!).outLeaves = 1)
    (hprim : (P.defs[((P.defs[P.root]!).body.nodes[q]!).fn]!).prim = true)
    (hf : ((P.defs[P.root]!).body.nodes[q]!).form = .batch) :
    ∀ n a, Flock.Steps.run "scope resolution" (resolveStep P) n (P.rootScope, ⟨false, q, t⟩) = .ok a →
      a = (P.defs[P.root]!).body.offsets[q]! + t := by
  intro n a h
  cases n with
  | zero => simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | succ n =>
    rw [Flock.Steps.run.eq_2, Resolve.rs_batch P P.rootScope q t hf] at h
    simp only [rootScope_d, rootScope_off, Nat.zero_add] at h
    simp only [hprim, ite_true, hq1, Nat.div_one, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact h.symm

/-- **A reference at `t` of an argument `inputArg` reads, to a batch input node `q`, resolves to gate
`offsets[q] + t`.** -/
theorem arg_gate (P : Prog) {k i t q a : ℕ} (hq : inputArg P (((P.defs[P.root]!).body.nodes[k]!).args[i]!) = some q)
    (hq1 : (P.defs[((P.defs[P.root]!).body.nodes[q]!).fn]!).outLeaves = 1)
    (hqb : ((P.defs[P.root]!).body.nodes[q]!).form = .batch)
    (h : resolve P P.rootScope ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get t) = .ok a) :
    a = (P.defs[P.root]!).body.offsets[q]! + t := by
  obtain ⟨⟨c, hparts⟩, -, hin⟩ := inputArg_some hq
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[q]!).fn]!).prim = true := by
    simp only [NciCheck.Desc.isInputNode, Bool.and_eq_true] at hin; exact hin.1
  rw [seq_get_one hparts t] at h
  rw [input_gate P (t := 0 + t) hq1 hprim hqb _ a h, Nat.zero_add]

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

omit hroot hL in
/-- **One step of resolution from below `A`**, the activation of member `m` of root node `k` (a batch or a call): it
ends at a gate at or past its activation's offset, moves below `A`, or leaves `A` up through leaf `j` of the callee's
parameter `i`, to member `m`'s own reference of node `k`'s argument `i` at the root. -/
theorem step_slice {A : Scope} {k m : ℕ} (hA : A = P.child P.rootScope k m)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form ≠ .scan) (B : ℕ) (s : Scope × Ref) (x)
    (hs : Resolve.Inv P B s) (hb : Below A s.1) (h : resolveStep P s = .ok x) :
    (∀ a, x = .inl a → s.1.off ≤ a) ∧ (∀ s', x = .inr s' → Below A s'.1 ∨
      (s'.1 = P.rootScope ∧ ∃ i j : ℕ, i < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params.size ∧
        j < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params[i]!).2.leaves ∧
        s'.2 = (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
          (upIdx ((P.defs[P.root]!).body.nodes[k]!) m i j))) := by
  obtain ⟨sc, ⟨p, i, j⟩⟩ := s
  obtain ⟨hg, hr, -, -⟩ := hs
  dsimp only at hg hr hb ⊢
  cases p
  · have hi : i < (P.defs[sc.d]!).body.nodes.size := (hr.2 rfl).1
    have hj := (hr.2 rfl).2
    dsimp only at hi hj
    obtain ⟨-, -, -, hdown, -⟩ := (hB sc hg).1 i hi
    have hdn := hdown j hj
    rcases hf' : ((P.defs[sc.d]!).body.nodes[i]!).form
    · rw [Resolve.rs_prim P sc i j hf'] at h
      cases h
      exact ⟨(fun a ha => by cases ha; omega), fun s' hs' => by cases hs'⟩
    · rw [Resolve.rs_call P sc i j hf'] at h
      cases h
      exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl (Below.child hb)⟩
    · rw [Resolve.rs_batch P sc i j hf'] at h
      by_cases hcp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
      · simp only [hcp, ite_true] at h
        cases h
        exact ⟨(fun a ha => by cases ha; exact Nat.le_add_right_of_le (Nat.le_add_right _ _)), fun s' hs' => by cases hs'⟩
      · simp only [hcp, Bool.false_eq_true, ite_false] at h
        cases h
        exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl (Below.child hb)⟩
    · simp only [hf'] at hdn
      obtain ⟨-, hdn⟩ := hdn
      by_cases hn : ((P.defs[sc.d]!).body.nodes[i]!).n = 0
      · exact absurd h (Resolve.rs_scan0 P sc i j x hf' hn)
      obtain ⟨carry, y, hret, -, -⟩ := hdn hn
      rw [Resolve.rs_scan P sc i j carry y hf' hn hret] at h
      cases h
      exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl (Below.child hb)⟩
  · rcases sc with ⟨d, off, st⟩ | ⟨parent, d, off, k', m'⟩
    · obtain ⟨-, -, rfl⟩ := hg
      exact absurd h (Resolve.rs_top P d off i j x)
    · have hg' := hg
      obtain ⟨-, -, hfn, -, -, -⟩ := hg'
      have hi : i < (P.defs[d]!).body.params.size := (hr.1 rfl).1
      have hj : j < ((P.defs[d]!).body.params[i]!).2.leaves := (hr.1 rfl).2
      rcases below_child hb with he | hpb
      · rw [hA] at he
        simp only [Prog.child, Scope.child.injEq] at he
        obtain ⟨hpar, -, -, hkk, hmm⟩ := he
        subst hpar
        subst k' m'
        subst hfn
        refine ⟨fun a ha => ?_, fun s' hs' => Or.inr ?_⟩
        · rcases hf'' : ((P.defs[P.rootScope.d]!).body.nodes[k]!).form
          · rw [Resolve.rs_up P _ _ off k m i j (Or.inl hf'')] at h; cases h; cases ha
          · rw [Resolve.rs_up P _ _ off k m i j (Or.inr hf'')] at h; cases h; cases ha
          · rw [Resolve.rs_up_batch P _ _ off k m i j hf''] at h; cases h; cases ha
          · exact absurd hf'' hf
        · rcases hf'' : ((P.defs[P.rootScope.d]!).body.nodes[k]!).form
          · rw [Resolve.rs_up P _ _ off k m i j (Or.inl hf'')] at h; cases h; cases hs'
            exact ⟨rfl, i, j, hi, hj, get_upIdx (Or.inl hf'') m i j⟩
          · rw [Resolve.rs_up P _ _ off k m i j (Or.inr hf'')] at h; cases h; cases hs'
            exact ⟨rfl, i, j, hi, hj, get_upIdx (Or.inr hf'') m i j⟩
          · rw [Resolve.rs_up_batch P _ _ off k m i j hf''] at h; cases h; cases hs'
            exact ⟨rfl, i, j, hi, hj, memberRef_upIdx hf'' m i j⟩
          · exact absurd hf'' hf
      · rcases hf'' : ((P.defs[parent.d]!).body.nodes[k']!).form
        · rw [Resolve.rs_up P parent d off k' m' i j (Or.inl hf'')] at h
          cases h
          exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl hpb⟩
        · rw [Resolve.rs_up P parent d off k' m' i j (Or.inr hf'')] at h
          cases h
          exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl hpb⟩
        · rw [Resolve.rs_up_batch P parent d off k' m' i j hf''] at h
          cases h
          exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl hpb⟩
        · by_cases hi0 : i = 0
          · subst hi0
            rw [Resolve.rs_up_scan0 P parent d off k' m' j hf''] at h
            by_cases hm0 : m' = 0
            · simp only [hm0, ite_true] at h
              cases h
              exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl hpb⟩
            · simp only [hm0, ite_false] at h
              cases h
              exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl (Below.child hpb)⟩
          · rw [Resolve.rs_up_scan P parent d off k' m' i j hi0 hf''] at h
            cases h
            exact ⟨(fun a ha => by cases ha), fun s' hs' => by cases hs'; exact Or.inl hpb⟩

/-- **Resolution from below `A`** ends in `A`'s gates, or, in no more steps, at the gate member `m`'s own reference of
an argument of node `k` resolves to from the root. -/
theorem run_slice {A : Scope} {k m : ℕ} (hA : A = P.child P.rootScope k m)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form ≠ .scan) :
    ∀ (n : ℕ) (s : Scope × Ref) (a : ℕ), Resolve.Inv P (A.off + (P.defs[A.d]!).gates) s → Below A s.1 →
      Flock.Steps.run "scope resolution" (resolveStep P) n s = .ok a →
      (A.off ≤ a ∧ a < A.off + (P.defs[A.d]!).gates) ∨
        ∃ i j : ℕ, i < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params.size ∧
          j < ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.params[i]!).2.leaves ∧ ∃ n' ≤ n,
          Flock.Steps.run "scope resolution" (resolveStep P) n'
            (P.rootScope, (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get
              (upIdx ((P.defs[P.root]!).body.nodes[k]!) m i j)) = .ok a
  | 0, _, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, s, a, hs, hb, h => by
    rw [Flock.Steps.run.eq_2] at h
    rcases hx : resolveStep P s with e | (b | s')
    · simp [hx, bind, Except.bind] at h
    · simp only [hx, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      have h1 := (step_slice P hB hA hf _ s _ hs hb hx).1 b rfl
      have h2 := (Resolve.step_ok P hroot hB hL _ s _ hs hx).1 b rfl
      rw [← h]
      exact Or.inl ⟨le_trans (below_off P hroot hB hL hb hs.1).1 h1, h2⟩
    · simp only [hx, bind, Except.bind] at h
      have hs' := (Resolve.step_ok P hroot hB hL _ s _ hs hx).2 s' rfl
      rcases (step_slice P hB hA hf _ s _ hs hb hx).2 s' rfl with hb' | ⟨h1, i, j, hi, hj, h2⟩
      · rcases run_slice hA hf n s' a hs' hb' h with h3 | ⟨i, j, hi, hj, n', hn', h4⟩
        · exact Or.inl h3
        · exact Or.inr ⟨i, j, hi, hj, n', by omega, h4⟩
      · refine Or.inr ⟨i, j, hi, hj, n, by omega, ?_⟩
        obtain ⟨sc', r'⟩ := s'
        dsimp only at h1 h2
        subst h1 h2
        exact h

end

/-- **The member-precise walk lemma.** -/
theorem memberSliceReads : MemberSliceReads := by
  intro bytes P hdec hov hbl k hk m hm hc hf g p ops sc hgate hg1 hg2 a ha
  obtain ⟨hr, hrp, -⟩ := BodiesOk.decode_root bytes P hdec
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := BodiesOk.order_closed P hov
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc hg => by
    obtain ⟨hSd, hpd⟩ := BodiesOk.good_closed P S hS0 hrp (fun d hd hp => (hS d hd hp).2) sc hg
    exact BodiesOk.body_ok P hok sc.d (hS sc.d hSd hpd).1
  obtain ⟨-, hL, -, -⟩ := Walk.bodiesLaidOut_spec P hbl
  obtain ⟨r, hinv', hb, hres⟩ := gate_below P ⟨hr, hrp⟩ hB hL hk hm hc hg1 hg2 hgate a ha
  rcases run_slice P ⟨hr, hrp⟩ hB hL (k := k) (m := m) rfl hf _ _ _ hinv' hb hres with
    h | ⟨i, j, hi, hj, n', hn', h⟩
  · exact Or.inl h
  · exact Or.inr ⟨i, j, hi, hj, Flock.Steps.run_mono hn' h⟩

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

/-- **`InM` holds what each of its gates reads**, in a flattening of a Program C-Flock decodes, topological and laid
out, at a root node `k` batching a composite whose every argument is one input node's leaves. -/
theorem closure_member {bytes : ByteArray} (hdec : decode bytes = .ok P) (hov : P.orderViolation = none)
    (hbl : NciCheck.Desc.bodiesLaidOut P = true) (C : Claims.Sampling.Prog) (hF : Flattens P C)
    {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size) (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (hbatch : ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    (hargs : ∀ b ∈ ((P.defs[P.root]!).body.nodes[k]!).args, (inputArg P b).isSome = true)
    (hinp : ∀ k' < (P.defs[P.root]!).body.nodes.size,
      NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = true →
        ((P.defs[P.root]!).body.nodes[k']!).args = #[])
    (x y : ℕ) (hy : y ∈ (C.gates[x]?.map Claims.Sampling.BGate.reads).getD []) (hx : InM P k m x) :
    InM P k m y := by
  have hxl : x < C.gates.length := by
    by_contra hn
    rw [List.getElem?_eq_none (by omega)] at hy
    simp at hy
  obtain ⟨p, ops, sc, hgate, -, hreads⟩ := hF.2.1 x hxl
  have hyo : y ∈ ops := (hreads y).1 hy
  rcases hx with ⟨hx1, hx2⟩ | ⟨i, j, hi, hj, hres⟩
  · have hf : ((P.defs[P.root]!).body.nodes[k]!).form ≠ .scan := fun h => by rw [hbatch] at h; cases h
    exact memberSliceReads bytes P hdec hov hbl k hk m hm hcomp hf x p ops sc hgate hx1 hx2 y hyo
  · have hBr : Resolve.BodyOk P P.root := hB P.rootScope (good_root P)
    obtain ⟨-, -, hup, -, -⟩ := hBr.1 k hk
    have hu := hup i hi j hj m hm
    simp only [hbatch] at hu
    have ht : upIdx ((P.defs[P.root]!).body.nodes[k]!) m i j <
        (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.count := by
      rw [upIdx_batch hbatch]; exact hu
    have hia := arg_lt ht
    have hmem : ((P.defs[P.root]!).body.nodes[k]!).args[i]! ∈ ((P.defs[P.root]!).body.nodes[k]!).args := by
      rw [getElem!_pos ((P.defs[P.root]!).body.nodes[k]!).args i hia]; exact Array.getElem_mem hia
    obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 (hargs _ hmem)
    obtain ⟨hx1, hx2⟩ := arg_lands P hroot hB hL hk hq ht _ x hres
    obtain ⟨-, hqK, hin⟩ := inputArg_some hq
    have := gate_input P hroot hL hqK hin hx1 hx2 (hinp q hqK hin) hgate
    rw [this] at hyo
    simp at hyo

/-- **An out leaf of a member lands in `InM`**: leaf `j` of root node `ke`, a batch of a composite, resolves only to a
gate of `InM` at member `j / L`, `L` the callee's out leaves. -/
theorem lands_member {ke j a : ℕ} (hk : ke < (P.defs[P.root]!).body.nodes.size)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).prim = false)
    (hbatch : ((P.defs[P.root]!).body.nodes[ke]!).form = .batch)
    (hj : j < ((P.defs[P.root]!).body.nodes[ke]!).out.leaves)
    (ha : resolve P P.rootScope ⟨false, ke, j⟩ = .ok a) :
    InM P ke (j / (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves) a := by
  have hBr : Resolve.BodyOk P P.root := hB P.rootScope (good_root P)
  obtain ⟨-, -, -, hdown, -⟩ := hBr.1 ke hk
  have hdn := hdown j hj
  simp only [hbatch] at hdn
  obtain ⟨hdn1, -, hdn3⟩ := hdn
  have hm : j / (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves <
      ((P.defs[P.root]!).body.nodes[ke]!).members := by
    rw [Resolve.bs_members _ (Or.inl hbatch)]; exact hdn1
  have hgA : Resolve.Good P (P.child P.rootScope ke
      (j / (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves)) :=
    Resolve.good_child P (good_root P) hk rfl hcomp hm rfl
  have hinv' := Resolve.down_ok P hroot hB hL _ _ hgA (le_refl _) _ (hdn3 hcomp)
  have hstep := Resolve.rs_batch P P.rootScope ke j hbatch
  simp only [rootScope_d, hcomp, Bool.false_eq_true, ite_false] at hstep
  unfold Flock.Program.resolve at ha
  rw [fuel_succ] at ha
  generalize hfu : Flock.Steps.FUEL - 1 = fuel at ha
  replace ha := ReadsExact.run_inr hstep ha
  have hf : ((P.defs[P.root]!).body.nodes[ke]!).form ≠ .scan := fun h => by rw [hbatch] at h; cases h
  rcases run_slice P hroot hB hL (k := ke)
      (m := j / (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves) rfl hf _ _ a hinv' Below.self ha with
    h | ⟨i, t, hi, ht, n', hn', h⟩
  · exact Or.inl h
  · exact Or.inr ⟨i, t, hi, ht, Flock.Steps.run_mono (m := Flock.Steps.FUEL) (by omega) h⟩

omit hB in
/-- **Member `m`'s activation lies in its node's gates.** -/
theorem act_node {k m x : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) (hx1 : (P.child P.rootScope k m).off ≤ x)
    (hx2 : x < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) :
    (P.defs[P.root]!).body.offsets[k]! ≤ x ∧ x < (P.defs[P.root]!).body.offsets[k + 1]! := by
  obtain ⟨-, -, hnodes⟩ := hL P.root hroot.1 hroot.2
  have hn := hnodes k hk
  rw [child_off] at hx1 hx2
  have hle : (m + 1) * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates ≤
      ((P.defs[P.root]!).body.nodes[k]!).members * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates :=
    Nat.mul_le_mul_right _ hm
  rw [Nat.add_mul, Nat.one_mul] at hle
  exact ⟨by omega, by omega⟩

omit hB in
/-- **No gate of a composite's activation is an input node's.** -/
theorem act_not_input {k m q x : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (hq : q < (P.defs[P.root]!).body.nodes.size)
    (hin : NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[q]!) = true)
    (hx1 : (P.child P.rootScope k m).off ≤ x)
    (hx2 : x < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates)
    (hxq : (P.defs[P.root]!).body.offsets[q]! ≤ x ∧ x < (P.defs[P.root]!).body.offsets[q + 1]!) : False := by
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hkq : k = q := DescAgree.node_unique _ _ (fun k' hk' => hmono k' (k' + 1) (by omega) (by omega)) hk hq
    (act_node P hroot hL hk hm hx1 hx2) hxq
  subst hkq
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = true := by
    simp only [NciCheck.Desc.isInputNode, Bool.and_eq_true] at hin; exact hin.1
  rw [hcomp] at hprim
  cases hprim

/-- **What a gate of `InM` depends on is in `InM`.** -/
theorem depends_member {bytes : ByteArray} (hdec : decode bytes = .ok P) (hov : P.orderViolation = none)
    (hbl : NciCheck.Desc.bodiesLaidOut P = true) (C : Claims.Sampling.Prog) (hF : Flattens P C)
    {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size) (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (hbatch : ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    (hargs : ∀ b ∈ ((P.defs[P.root]!).body.nodes[k]!).args, (inputArg P b).isSome = true)
    (hinp : ∀ k' < (P.defs[P.root]!).body.nodes.size,
      NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = true →
        ((P.defs[P.root]!).body.nodes[k']!).args = #[])
    {w g : ℕ} (hw : InM P k m w) (hd : NetTiming.Depends C g w) : InM P k m g :=
  depends_closed C (InM P k m) (closure_member P hroot hB hL hdec hov hbl C hF hk hm hcomp hbatch hargs hinp) hd hw

end

end Properties.NetworkCertifier.MemberSliceProof
