import Properties.NetworkScan
import NciCheck.Desc
import Proofs.Properties.Nci.BodiesOk
import Proofs.Properties.Nci.OperandsResolve

/-!
# The walk lemma

**The walk lemma** (`MemberReads`): in a Program C-Flock decodes, topological (`Flock.Program.Prog.orderViolation`) and
laid out (`NciCheck.Desc.bodiesLaidOut`), a gate in the activation of member `m` of a root node `k` calling a composite
(by a batch or a call: a scan's iteration reads the one before) reads only gates of that activation, or a gate one of
node `k`'s argument references resolves to from the root, what its arguments bind. It is where C-Flock's walk
(`Flock.Program.Prog.gate`, `Flock.Program.resolve`) takes a gate's operands, and the scan's causality
(`Proofs.Properties.NetworkScan.Causal`) rests on it.

Fix a member `m` of a root node `k` that calls a composite by a batch or a call, and its activation `A`
(`Prog.child P.rootScope k m`). The activations below `A` (`Below`) are `A` and the members of nodes of one. A step of
resolution (`step_below`) from a state below `A` ends at a gate at or past its activation's offset, moves below `A` (down
into a node's member, up to the parent or a scan's previous iteration below `A`), or leaves `A` up through one of its
parameters, to a reference of node `k`'s argument at the root. With `Resolve.step_ok`'s bound at the end of `A`'s gates,
resolution from below `A` (`run_below`) ends in `A`'s gates or at what a reference of node `k`'s argument resolves to
from the root. `Prog.gate`'s walk to a gate of `A` goes through `A` (`gateStep_root`) and stays below it, and each
operand it returns is such a resolution (`gate_below`): `memberReads`.
-/

namespace Properties.NetworkScan

/-- **The walk lemma.** In a Program C-Flock decodes, topological and laid out, a gate `g` in the activation of member
`m` of a root node `k` that calls a composite by a batch or a call reads (its operands, `Flock.Program.Prog.gate`) only
gates of that activation, or a gate a reference of one of node `k`'s arguments resolves to from the root. -/
def MemberReads : Prop :=
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
          ∃ i t : Nat, t < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.count) ∧
            Flock.Program.resolve P P.rootScope ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get t) = .ok a

end Properties.NetworkScan

namespace Properties.NetworkScan.WalkProof

open Flock.Program
open NciCheck.Properties.SecurityProofs

/-- The activations at or below `A`: `A`, and each member of a node of one. -/
inductive Below (A : Scope) : Scope → Prop
  | self : Below A A
  | child {parent : Scope} {d off k m : ℕ} : Below A parent → Below A (.child parent d off k m)

theorem below_child {A parent : Scope} {d off k m : ℕ} (h : Below A (.child parent d off k m)) :
    A = .child parent d off k m ∨ Below A parent := by
  cases h with
  | self => exact Or.inl rfl
  | child h => exact Or.inr h

theorem inv_mono {P : Prog} {B B' : ℕ} {s : Scope × Ref} (h : Resolve.Inv P B s) (hB : B ≤ B') :
    Resolve.Inv P B' s :=
  ⟨h.1, h.2.1, fun hp => le_trans (h.2.2.1 hp) hB, fun hn => le_trans (h.2.2.2 hn) hB⟩

theorem rootScope_d (P : Prog) : P.rootScope.d = P.root := rfl

theorem rootScope_off (P : Prog) : P.rootScope.off = 0 := rfl

theorem good_root (P : Prog) : Resolve.Good P P.rootScope := ⟨rfl, rfl, rfl⟩

theorem child_d (P : Prog) (k m : ℕ) : (P.child P.rootScope k m).d = ((P.defs[P.root]!).body.nodes[k]!).fn := rfl

theorem child_off (P : Prog) (k m : ℕ) : (P.child P.rootScope k m).off =
    (P.defs[P.root]!).body.offsets[k]! + m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates := by
  have e : (P.child P.rootScope k m).off =
      0 + (P.defs[P.root]!).body.offsets[k]! + m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates := rfl
  rw [e, Nat.zero_add]

/-- Two members of `G` gates each holding `x` are one. -/
theorem member_eq {M m G x : ℕ} (h1 : M * G ≤ x) (h2 : x < M * G + G) (h3 : m * G ≤ x) (h4 : x < m * G + G) :
    M = m := by
  rcases Nat.lt_trichotomy M m with h | h | h
  · have : (M + 1) * G ≤ m * G := Nat.mul_le_mul_right G h
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  · exact h
  · have : (m + 1) * G ≤ M * G := Nat.mul_le_mul_right G h
    rw [Nat.add_mul, Nat.one_mul] at this
    omega

/-- `Prog.gate`'s walk goes down into a member of a node. -/
theorem gateStep_child {P : Prog} {g : ℕ} {sc sc' : Scope} (h : gateStep P g sc = .ok (.inr sc')) :
    ∃ k m, sc' = P.child sc k m := by
  unfold gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hp, ite_true] at h
      split at h
      · cases h
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        cases h
    · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq, Sum.inr.injEq] at h
      exact ⟨_, _, h.symm⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

/-- **An activation below `A` lies within `A`'s gates.** -/
theorem below_off {A : Scope} : ∀ {sc : Scope}, Below A sc → Resolve.Good P sc →
    A.off ≤ sc.off ∧ sc.off + (P.defs[sc.d]!).gates ≤ A.off + (P.defs[A.d]!).gates := by
  intro sc h
  induction h with
  | self => exact fun _ => ⟨le_refl _, le_refl _⟩
  | @child parent d off k m _ ih =>
    intro hg
    obtain ⟨hp, hk, hfn, hc, hm, ho⟩ := hg
    obtain ⟨ih1, ih2⟩ := ih hp
    obtain ⟨hd, hpc⟩ := Resolve.good_d P hroot hB parent hp
    obtain ⟨-, hK, hnodes⟩ := hL parent.d hd hpc
    have hn := hnodes k hk
    rw [hfn] at hn
    have hmono := Resolve.offsets_mono P parent.d (hL _ hd) hpc (k + 1) _ (by omega) (le_refl _)
    have hle : (m + 1) * (P.defs[d]!).gates ≤
        ((P.defs[parent.d]!).body.nodes[k]!).members * (P.defs[d]!).gates := Nat.mul_le_mul_right _ hm
    rw [Nat.add_mul, Nat.one_mul] at hle
    show A.off ≤ off ∧ off + (P.defs[d]!).gates ≤ A.off + (P.defs[A.d]!).gates
    constructor <;> omega

/-- **One step of resolution from below `A`**, the activation of member `m` of root node `k` (a batch or a call): it ends
at a gate at or past its activation's offset, moves below `A`, or leaves `A` up through a parameter, to a reference of
node `k`'s argument at the root. -/
theorem step_below {A : Scope} {k m : ℕ} (hA : A = P.child P.rootScope k m)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form ≠ .scan) (B : ℕ) (s : Scope × Ref) (x)
    (hs : Resolve.Inv P B s) (hb : Below A s.1) (h : resolveStep P s = .ok x) :
    (∀ a, x = .inl a → s.1.off ≤ a) ∧ (∀ s', x = .inr s' → Below A s'.1 ∨
      (s'.1 = P.rootScope ∧ ∃ i t : ℕ, t < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.count) ∧
        s'.2 = (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get t)) := by
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
      exact ⟨fun a ha => by cases ha; omega, fun s' hs' => by cases hs'⟩
    · rw [Resolve.rs_call P sc i j hf'] at h
      cases h
      exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl (Below.child hb)⟩
    · rw [Resolve.rs_batch P sc i j hf'] at h
      by_cases hcp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
      · simp only [hcp, ite_true] at h
        cases h
        exact ⟨fun a ha => by cases ha; omega, fun s' hs' => by cases hs'⟩
      · simp only [hcp, Bool.false_eq_true, ite_false] at h
        cases h
        exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl (Below.child hb)⟩
    · simp only [hf'] at hdn
      obtain ⟨-, hdn⟩ := hdn
      by_cases hn : ((P.defs[sc.d]!).body.nodes[i]!).n = 0
      · exact absurd h (Resolve.rs_scan0 P sc i j x hf' hn)
      obtain ⟨carry, y, hret, -, -⟩ := hdn hn
      rw [Resolve.rs_scan P sc i j carry y hf' hn hret] at h
      cases h
      exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl (Below.child hb)⟩
  · rcases sc with ⟨d, off, st⟩ | ⟨parent, d, off, k', m'⟩
    · obtain ⟨-, -, rfl⟩ := hg
      exact absurd h (Resolve.rs_top P d off i j x)
    · have hg' := hg
      obtain ⟨hp, hk', hfn, -, hm', -⟩ := hg'
      have hi : i < (P.defs[d]!).body.params.size := (hr.1 rfl).1
      have hj : j < ((P.defs[d]!).body.params[i]!).2.leaves := (hr.1 rfl).2
      obtain ⟨-, -, hup, -, -⟩ := (hB parent hp).1 k' hk'
      rw [hfn] at hup
      have hu := hup i hi j hj m' hm'
      rcases below_child hb with he | hpb
      · rw [hA] at he
        simp only [Prog.child, Scope.child.injEq] at he
        obtain ⟨hpar, -, -, hkk, hmm⟩ := he
        subst hpar
        subst k' m'
        refine ⟨fun a ha => ?_, fun s' hs' => Or.inr ?_⟩
        · rcases hf'' : ((P.defs[P.rootScope.d]!).body.nodes[k]!).form
          · rw [Resolve.rs_up P _ d off k m i j (Or.inl hf'')] at h; cases h; cases ha
          · rw [Resolve.rs_up P _ d off k m i j (Or.inr hf'')] at h; cases h; cases ha
          · rw [Resolve.rs_up_batch P _ d off k m i j hf''] at h; cases h; cases ha
          · exact absurd hf'' hf
        · rcases hf'' : ((P.defs[P.rootScope.d]!).body.nodes[k]!).form
          · simp only [hf''] at hu
            rw [Resolve.rs_up P _ d off k m i j (Or.inl hf'')] at h; cases h; cases hs'
            exact ⟨rfl, i, j, hu, rfl⟩
          · simp only [hf''] at hu
            rw [Resolve.rs_up P _ d off k m i j (Or.inr hf'')] at h; cases h; cases hs'
            exact ⟨rfl, i, j, hu, rfl⟩
          · simp only [hf''] at hu
            rw [Resolve.rs_up_batch P _ d off k m i j hf'', Resolve.memberRef_eq] at h; cases h; cases hs'
            exact ⟨rfl, i, _, hu, rfl⟩
          · exact absurd hf'' hf
      · rcases hf'' : ((P.defs[parent.d]!).body.nodes[k']!).form
        · rw [Resolve.rs_up P parent d off k' m' i j (Or.inl hf'')] at h
          cases h
          exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl hpb⟩
        · rw [Resolve.rs_up P parent d off k' m' i j (Or.inr hf'')] at h
          cases h
          exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl hpb⟩
        · rw [Resolve.rs_up_batch P parent d off k' m' i j hf''] at h
          cases h
          exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl hpb⟩
        · by_cases hi0 : i = 0
          · subst hi0
            rw [Resolve.rs_up_scan0 P parent d off k' m' j hf''] at h
            by_cases hm0 : m' = 0
            · simp only [hm0, ite_true] at h
              cases h
              exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl hpb⟩
            · simp only [hm0, ite_false] at h
              cases h
              exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl (Below.child hpb)⟩
          · rw [Resolve.rs_up_scan P parent d off k' m' i j hi0 hf''] at h
            cases h
            exact ⟨fun a ha => by cases ha, fun s' hs' => by cases hs'; exact Or.inl hpb⟩

/-- **Resolution from below `A`** ends in `A`'s gates, or at the gate a reference of node `k`'s argument resolves to
from the root, in no more steps. -/
theorem run_below {A : Scope} {k m : ℕ} (hA : A = P.child P.rootScope k m)
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form ≠ .scan) :
    ∀ (n : ℕ) (s : Scope × Ref) (a : ℕ), Resolve.Inv P (A.off + (P.defs[A.d]!).gates) s → Below A s.1 →
      Flock.Steps.run "scope resolution" (resolveStep P) n s = .ok a →
      (A.off ≤ a ∧ a < A.off + (P.defs[A.d]!).gates) ∨
        ∃ i t : ℕ, t < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.count) ∧ ∃ n' ≤ n,
          Flock.Steps.run "scope resolution" (resolveStep P) n'
            (P.rootScope, (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get t) = .ok a
  | 0, _, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, s, a, hs, hb, h => by
    rw [Flock.Steps.run.eq_2] at h
    rcases hx : resolveStep P s with e | (b | s')
    · simp [hx, bind, Except.bind] at h
    · simp only [hx, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      have h1 := (step_below P hroot hB hL hA hf _ s _ hs hb hx).1 b rfl
      have h2 := (Resolve.step_ok P hroot hB hL _ s _ hs hx).1 b rfl
      rw [← h]
      exact Or.inl ⟨le_trans (below_off P hroot hB hL hb hs.1).1 h1, h2⟩
    · simp only [hx, bind, Except.bind] at h
      have hs' := (Resolve.step_ok P hroot hB hL _ s _ hs hx).2 s' rfl
      rcases (step_below P hroot hB hL hA hf _ s _ hs hb hx).2 s' rfl with hb' | ⟨h1, i, t, ht, h2⟩
      · rcases run_below hA hf n s' a hs' hb' h with h3 | ⟨i, t, ht, n', hn', h4⟩
        · exact Or.inl h3
        · exact Or.inr ⟨i, t, ht, n', by omega, h4⟩
      · refine Or.inr ⟨i, t, ht, n, by omega, ?_⟩
        obtain ⟨sc', r'⟩ := s'
        dsimp only at h1 h2
        subst h1 h2
        exact h

/-- **`Prog.gate`'s first step to a gate of `A`** goes into `A`. -/
theorem gateStep_root {k m g : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hg1 : (P.child P.rootScope k m).off ≤ g)
    (hg2 : g < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) :
    gateStep P g P.rootScope = .ok (.inr (P.child P.rootScope k m)) := by
  obtain ⟨-, -, hnodes⟩ := hL P.root hroot.1 hroot.2
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hn := hnodes k hk
  rw [child_off] at hg1 hg2
  have hle : (m + 1) * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates ≤
      ((P.defs[P.root]!).body.nodes[k]!).members * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates :=
    Nat.mul_le_mul_right _ hm
  rw [Nat.add_mul, Nat.one_mul] at hle
  have hin : (P.defs[P.root]!).body.offsets[k]! ≤ g ∧ g < (P.defs[P.root]!).body.offsets[k + 1]! := by omega
  have hmono' : ∀ k' < (P.defs[P.root]!).body.nodes.size,
      (P.defs[P.root]!).body.offsets[k']! ≤ (P.defs[P.root]!).body.offsets[k' + 1]! :=
    fun k' hk' => hmono k' (k' + 1) (by omega) (by omega)
  unfold gateStep
  simp only [bind, Except.bind, rootScope_d, rootScope_off, Nat.sub_zero]
  split
  · rename_i k' hk'
    obtain ⟨hk'K, hpk⟩ := DescAgree.find?_range_some hk'
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    have hkk : k' = k := DescAgree.node_unique _ _ hmono' hk'K hk hpk hin
    subst k'
    simp only [hc, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq, Sum.inr.injEq]
    congr 1
    obtain ⟨-, h2, h3⟩ := Walk.member_of ((P.defs[P.root]!).body.nodes[k]!)
      (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates (g - (P.defs[P.root]!).body.offsets[k]!) (by omega)
    have := member_eq (m := m) h2 (by omega) (by omega) (by omega)
    exact this
  · exfalso
    rename_i hne
    have hnone := Option.eq_none_iff_forall_ne_some.2 hne
    rw [Array.find?_eq_none] at hnone
    exact hnone k (Array.mem_range.2 hk) (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hin)

/-- **The operands of a gate of `A`**: each is a resolution from an activation below `A`, of a state that resolves before
the end of `A`'s gates. -/
theorem gate_below {k m g p : ℕ} {ops : Array ℕ} {sc : Scope} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (hg1 : (P.child P.rootScope k m).off ≤ g)
    (hg2 : g < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates)
    (h : P.gate g = .ok (p, ops, sc)) :
    ∀ a ∈ ops, ∃ r, Resolve.Inv P ((P.child P.rootScope k m).off +
        (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) (sc, r) ∧
      Below (P.child P.rootScope k m) sc ∧ resolve P sc r = .ok a := by
  have hA : Resolve.Good P (P.child P.rootScope k m) :=
    Resolve.good_child P (good_root P) hk rfl hc hm rfl
  have hAin := (below_off P hroot hB hL (A := P.rootScope) (sc := P.child P.rootScope k m)
    (Below.child Below.self) hA).2
  rw [child_d, rootScope_off, rootScope_d, Nat.zero_add] at hAin
  have hrg : g < (P.defs[P.root]!).gates := by omega
  unfold Prog.gate at h
  have hng : ¬ (g ≥ (P.defs[P.root]!).gates) := by omega
  simp only [hng, ite_false, bind, Except.bind] at h
  have hF : Flock.Steps.FUEL = (Flock.Steps.FUEL - 1) + 1 := by unfold Flock.Steps.FUEL; norm_num
  have h0 := gateStep_root P hroot hB hL hk hc hm hg1 hg2
  rw [hF, Flock.Steps.run.eq_2, h0] at h
  simp only [bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (gateStep P g)
    (fun sc => Below (P.child P.rootScope k m) sc ∧ Walk.Reach P g sc)
    (fun b => ∀ a ∈ b.2.1, ∃ r, Resolve.Inv P ((P.child P.rootScope k m).off +
        (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) (b.2.2, r) ∧
      Below (P.child P.rootScope k m) b.2.2 ∧ resolve P b.2.2 r = .ok a) ?_ ?_ _ _ (p, ops, sc)
    ⟨Below.self, Walk.Reach.step Walk.Reach.root h0⟩ h
  · intro sc b ⟨hb, hreach⟩ hstep
    obtain ⟨hg, hoff, hlt⟩ := OperandsResolve.reach_walkQ P hroot hB hL hrg hreach
    obtain ⟨hd, hpc⟩ := Resolve.good_d P hroot hB sc hg
    obtain ⟨h0, hK, hnodes⟩ := hL sc.d hd hpc
    have hend := (below_off P hroot hB hL hb hg).2
    rw [child_d] at hend
    unfold gateStep at hstep
    simp only [bind, Except.bind] at hstep
    split at hstep
    · rename_i k' hk'
      obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk'
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
      have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k']! <
          ((P.defs[sc.d]!).body.nodes[k']!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k']!).fn]!).gates := by
        have := hnodes k' hkK
        omega
      obtain ⟨hm1, -, -⟩ := Walk.member_of ((P.defs[sc.d]!).body.nodes[k']!) _ _ hx
      by_cases hpc' : (P.defs[((P.defs[sc.d]!).body.nodes[k']!).fn]!).prim = true
      · simp only [hpc', ite_true] at hstep
        split at hstep
        · cases hstep
        · rename_i ops' hops
          simp only [pure, Except.pure, Except.ok.injEq, Sum.inl.injEq] at hstep
          subst hstep
          intro a ha
          obtain ⟨i, hi, t, ht, hres⟩ := Resolve.operands_spec P sc _ _ ops' hops a ha
          obtain ⟨-, hargs, -, -, hop⟩ := (hB sc hg).1 k' hkK
          have hidx := hop hpc' i hi _ hm1 t ht
          have hr : (if (((P.defs[sc.d]!).body.nodes[k']!).form == Form.batch) = true then
              (((P.defs[sc.d]!).body.nodes[k']!).args[i]!).memberRef
                (if (((P.defs[sc.d]!).body.nodes[k']!).form == .prim ||
                  ((P.defs[sc.d]!).body.nodes[k']!).form == .call) = true then 0
                else (g - sc.off - (P.defs[sc.d]!).body.offsets[k']!) /
                  (P.defs[((P.defs[sc.d]!).body.nodes[k']!).fn]!).gates) t
              else (((P.defs[sc.d]!).body.nodes[k']!).args[i]!).seq.get t) =
              (((P.defs[sc.d]!).body.nodes[k']!).args[i]!).seq.get
                (if (((P.defs[sc.d]!).body.nodes[k']!).form == Form.batch) = true then
                  Resolve.memberIdx (((P.defs[sc.d]!).body.nodes[k']!).args[i]!)
                    (if (((P.defs[sc.d]!).body.nodes[k']!).form == .prim ||
                      ((P.defs[sc.d]!).body.nodes[k']!).form == .call) = true then 0
                    else (g - sc.off - (P.defs[sc.d]!).body.offsets[k']!) /
                      (P.defs[((P.defs[sc.d]!).body.nodes[k']!).fn]!).gates) t
                else t) := by
            split <;> simp only [Resolve.memberRef_eq]
          rw [hr] at hres
          have hst := Resolve.start_ok P hroot hB hL sc hg k' hkK _ (hargs i _ hidx)
          have hK' : (P.defs[sc.d]!).body.offsets[k']! ≤ (P.defs[sc.d]!).gates := by
            rw [← hK]; exact Resolve.offsets_mono P sc.d (hL _ hd) hpc k' _ (by omega) (le_refl _)
          exact ⟨_, inv_mono hst (by omega), hb, hres⟩
      · simp only [hpc', Bool.false_eq_true, ite_false, pure, Except.pure] at hstep
        cases hstep
    · simp [throw, throwThe, MonadExceptOf.throw] at hstep
  · intro sc sc' ⟨hb, hreach⟩ hstep
    obtain ⟨k', m', rfl⟩ := gateStep_child hstep
    exact ⟨Below.child hb, Walk.Reach.step hreach hstep⟩

end

/-- **The walk lemma.** -/
theorem memberReads : MemberReads := by
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
  rcases run_below P ⟨hr, hrp⟩ hB hL (k := k) (m := m) rfl hf _ _ _ hinv' hb hres with h | ⟨i, t, ht, n', hn', h⟩
  · exact Or.inl h
  · exact Or.inr ⟨i, t, ht, Flock.Steps.run_mono hn' h⟩

end Properties.NetworkScan.WalkProof
