import Proofs.Properties.NetworkScan.Walk
import Proofs.Properties.NetworkScan.Accept

/-!
# The scan is causal, proved

Let `ke` be the egress template's root node, `kl` the later ingress template's, and `S` the gates of `ke`'s members
together with the gates of the input nodes `ke`'s arguments read (`InS`). An egress gate resolves into `S`
(`batch_lands`, from `WalkProof.run_below`). `S` is closed under reads (`closure`): a gate of a member of `ke` reads its
activation's gates or what its arguments' references resolve to (the walk lemma, `WalkProof.memberReads`), which is an
input node's gate when the argument is one input node's leaves (`arg_lands`), and an input node's gate reads nothing
(`gate_input`). So everything an egress gate depends on is in `S` (`depends_closed`). A later bucket's ingress gate is a
gate of the input node `kl`'s row reads (`param_lands`), which is neither `ke`, a composite, nor any input node an
egress argument reads, and a gate lies in one root node's gates (`DescAgree.node_unique`): `causal_of_facts`, from
what `scanOf` checks (`Facts`).

`ScanCausal` is part (3) of `NetTiming.Layout.Fits` at `scanLayout`: `scanOf` takes each template's argument to be one
input node's leaves (`NetworkCheck.Period.inputArg`) and no egress argument to be an ingress row's input node, so an
egress member reads only its activation's gates and its arguments' input gates, and a later bucket's ingress gates are
its row's input node's, which no egress argument names. The walk lemma reads the Program topological, which `ScanCausal`
takes as `P.orderViolation = none`; `scanOf` checks it (`Fits.Accepted.topo`), and `Properties.NetworkScan.scanFits`
reads it from there.
-/

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- **The scan is causal** (`NetTiming.Layout.Fits`'s part (3)). If `scanOf` accepts the scan `j`, its canonical text
`c` decodes to `P` and `P` is topological, then in every flattening `C` of `P` no egress gate of a step depends on an
ingress gate of a later step of the window. -/
def ScanCausal : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P →
    P.orderViolation = none → Flattens P C →
    ∀ t < s.window, ∀ t' < s.window, t < t' → ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s).egress t ℓ σ),
      ∀ i ∈ (scanLayout P s).ingress t', ¬ NetTiming.Depends C i g

end Properties.NetworkScan

namespace Properties.NetworkScan.CausalProof

open Flock.Program
open NciCheck.Properties.SecurityProofs
open NetworkCheck.Period
open Properties.NetworkScan.WalkProof

theorem fuel_succ : Flock.Steps.FUEL = (Flock.Steps.FUEL - 1) + 1 := by unfold Flock.Steps.FUEL; omega

theorem seq_get_one {s : Seq} {q : Part} (h : s.parts = #[q]) (k : ℕ) : s.get k = q.get k := by
  first
  | (unfold Seq.get; rw [h]; rfl)
  | simp [Seq.get, h]

/-- An argument `inputArg` reads is one affine run over an input node's leaves from its first. -/
theorem inputArg_some {P : Prog} {a : Arg} {q : ℕ} (h : inputArg P a = some q) :
    (∃ c, a.seq.parts = #[Part.affine false q 0 c]) ∧ q < (P.defs[P.root]!).body.nodes.size ∧
      NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[q]!) = true := by
  unfold inputArg at h
  split at h
  · rename_i k c hparts
    split at h
    · rename_i n hn
      split at h
      · rename_i hin
        cases h
        obtain ⟨hk, hnk⟩ := Array.getElem?_eq_some_iff.1 hn
        exact ⟨⟨c, Array.ext' hparts⟩, hk, by rw [getElem!_pos (P.defs[P.root]!).body.nodes q hk, hnk]; exact hin⟩
      · cases h
    · cases h
  · cases h

theorem arg_lt {args : Array Arg} {i t : ℕ} (h : t < (args[i]!).seq.count) : i < args.size := by
  by_contra hi
  rw [getElem!_neg args i hi] at h
  exact Nat.not_lt_zero t h

theorem member_div {M G x : ℕ} (hx : x < M * G) : x / G < M ∧ x / G * G ≤ x ∧ x < x / G * G + G := by
  have hG : 0 < G := Nat.pos_of_ne_zero (by rintro rfl; simp at hx)
  have h1 := Nat.div_add_mod x G
  have h2 := Nat.mod_lt x hG
  rw [Nat.mul_comm] at h1
  refine ⟨?_, by omega, by omega⟩
  by_contra hc
  have : M * G ≤ x / G * G := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The gates an egress gate's dependencies stay in: those of root node `ke`'s members, and those of each input node
one of its arguments reads. -/
abbrev InS (P : Prog) (ke x : ℕ) : Prop :=
  ((P.defs[P.root]!).body.offsets[ke]! ≤ x ∧ x < (P.defs[P.root]!).body.offsets[ke + 1]!) ∨
    ∃ b ∈ ((P.defs[P.root]!).body.nodes[ke]!).args, ∃ q, inputArg P b = some q ∧
      (P.defs[P.root]!).body.offsets[q]! ≤ x ∧ x < (P.defs[P.root]!).body.offsets[q + 1]!

/-- Everything a gate of `S` depends on is in `S`, when `S` holds what each of its gates reads. -/
theorem depends_closed (C : Claims.Sampling.Prog) (S : ℕ → Prop)
    (hS : ∀ x y, y ∈ (C.gates[x]?.map Claims.Sampling.BGate.reads).getD [] → S x → S y)
    {i g : ℕ} (h : NetTiming.Depends C i g) (hg : S g) : S i := by
  unfold NetTiming.Depends at h
  revert hg
  induction h with
  | refl => exact id
  | tail _ hbc ih => exact fun hc => ih (hS _ _ hbc hc)

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

/-- **A leaf of an input node resolves to one of its gates.** -/
theorem input_lands {q j : ℕ} (hq : q < (P.defs[P.root]!).body.nodes.size)
    (hin : NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[q]!) = true)
    (hj : j < ((P.defs[P.root]!).body.nodes[q]!).out.leaves) :
    ∀ n a, Flock.Steps.run "scope resolution" (resolveStep P) n (P.rootScope, ⟨false, q, j⟩) = .ok a →
      (P.defs[P.root]!).body.offsets[q]! ≤ a ∧ a < (P.defs[P.root]!).body.offsets[q + 1]! := by
  intro n a h
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[q]!).fn]!).prim = true := by
    simp only [NciCheck.Desc.isInputNode, Bool.and_eq_true] at hin; exact hin.1
  have hBr : Resolve.BodyOk P P.root := hB P.rootScope (good_root P)
  obtain ⟨-, -, -, hdown, -⟩ := hBr.1 q hq
  have hdn := hdown j hj
  obtain ⟨-, -, hnodes⟩ := hL P.root hroot.1 hroot.2
  have hn := hnodes q hq
  cases n with
  | zero => simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | succ n =>
    rw [Flock.Steps.run.eq_2] at h
    rcases hf : ((P.defs[P.root]!).body.nodes[q]!).form
    · rw [Resolve.rs_prim P P.rootScope q j hf] at h
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, rootScope_d, rootScope_off,
        Nat.zero_add] at h
      subst h
      simp only [hf] at hdn
      rw [Resolve.prim_members _ (Or.inl hf), hdn.2] at hn
      omega
    · simp only [hf] at hdn
      exact absurd (hdn.1.symm.trans hprim) (by decide)
    · rw [Resolve.rs_batch P P.rootScope q j hf] at h
      simp only [rootScope_d, rootScope_off, Nat.zero_add] at h
      simp only [hprim, ite_true, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      subst h
      simp only [hf] at hdn
      obtain ⟨h1, h2, -⟩ := hdn
      rw [Resolve.bs_members _ (Or.inl hf), h2 hprim] at hn
      omega
    · simp only [hf] at hdn
      exact absurd (hdn.1.symm.trans hprim) (by decide)

/-- **A reference of an argument `inputArg` reads resolves to a gate of its input node.** -/
theorem arg_lands {k i t q : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hq : inputArg P (((P.defs[P.root]!).body.nodes[k]!).args[i]!) = some q)
    (ht : t < (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.count) :
    ∀ n a, Flock.Steps.run "scope resolution" (resolveStep P) n
        (P.rootScope, (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get t) = .ok a →
      (P.defs[P.root]!).body.offsets[q]! ≤ a ∧ a < (P.defs[P.root]!).body.offsets[q + 1]! := by
  obtain ⟨⟨c, hparts⟩, hqK, hin⟩ := inputArg_some hq
  have hget := seq_get_one hparts t
  have hBr : Resolve.BodyOk P P.root := hB P.rootScope (good_root P)
  have hr := (hBr.1 k hk).2.1 i t ht
  rw [hget] at hr
  have hj : 0 + t < ((P.defs[P.root]!).body.nodes[q]!).out.leaves := (hr.2 rfl).2
  intro n a h
  rw [hget] at h
  exact input_lands P hroot hB hL hqK hin hj n a h

/-- **An egress leaf resolves into `InS`**: a leaf of root node `ke`, a batch of a composite whose every argument is
one input node's leaves, resolves, to a gate of one of its members or of an input node an argument reads. -/
theorem batch_lands (hc : NciCheck.Desc.calleesFirst P = true) (hrp : (P.defs[P.root]!).body.params.size = 0)
    (hz : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).form = .scan →
        ((P.defs[d]!).body.nodes[k]!).n = 0 → ((P.defs[d]!).body.nodes[k]!).out.leaves = 0)
    (hfuel : 4 * (NciCheck.Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL)
    {ke j : ℕ} (hk : ke < (P.defs[P.root]!).body.nodes.size)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).prim = false)
    (hbatch : ((P.defs[P.root]!).body.nodes[ke]!).form = .batch)
    (hargs : ∀ b ∈ ((P.defs[P.root]!).body.nodes[ke]!).args, (inputArg P b).isSome = true)
    (hj : j < ((P.defs[P.root]!).body.nodes[ke]!).out.leaves) :
    ∃ a, resolve P P.rootScope ⟨false, ke, j⟩ = .ok a ∧ InS P ke a := by
  have hBr : Resolve.BodyOk P P.root := hB P.rootScope (good_root P)
  obtain ⟨-, hK, hnodes⟩ := hL P.root hroot.1 hroot.2
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hn := hnodes ke hk
  have hke1 : (P.defs[P.root]!).body.offsets[ke + 1]! ≤ (P.defs[P.root]!).gates := by
    rw [← hK]; exact hmono _ _ (by omega) (le_refl _)
  have hinv : Resolve.Inv P (P.defs[P.root]!).gates (P.rootScope, ⟨false, ke, j⟩) := by
    refine ⟨good_root P, ⟨fun h => absurd h (by decide), fun _ => ⟨hk, hj⟩⟩, fun h => absurd h (by decide),
      fun _ => ?_⟩
    show 0 + (P.defs[P.root]!).body.offsets[ke + 1]! ≤ (P.defs[P.root]!).gates
    omega
  obtain ⟨a, ha⟩ := OperandsResolve.resolve_ok P hroot hB hL hc hrp hz hfuel _ _ _ hinv
  refine ⟨a, ha, ?_⟩
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
  unfold Flock.Program.resolve at ha
  rw [fuel_succ, Flock.Steps.run.eq_2, Resolve.rs_batch P P.rootScope ke j hbatch] at ha
  simp only [rootScope_d] at ha
  simp only [hcomp, Bool.false_eq_true, ite_false, bind, Except.bind] at ha
  have hf : ((P.defs[P.root]!).body.nodes[ke]!).form ≠ .scan := fun h => by rw [hbatch] at h; cases h
  rcases run_below P hroot hB hL rfl hf _ _ a hinv' Below.self ha with ⟨h1, h2⟩ | ⟨i, t, ht, n', -, h⟩
  · rw [child_off] at h1
    rw [child_off, child_d] at h2
    have hle : (j / (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves + 1) *
        (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).gates ≤
        ((P.defs[P.root]!).body.nodes[ke]!).members * (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).gates :=
      Nat.mul_le_mul_right _ hm
    rw [Nat.add_mul, Nat.one_mul] at hle
    exact Or.inl ⟨by omega, by omega⟩
  · have hi := arg_lt ht
    have hmem : ((P.defs[P.root]!).body.nodes[ke]!).args[i]! ∈ ((P.defs[P.root]!).body.nodes[ke]!).args := by
      rw [getElem!_pos ((P.defs[P.root]!).body.nodes[ke]!).args i hi]; exact Array.getElem_mem hi
    obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 (hargs _ hmem)
    exact Or.inr ⟨_, hmem, q, hq, arg_lands P hroot hB hL hk hq ht n' a h⟩

/-- **A later bucket's ingress gate is a gate of its row's input node**: parameter 0 of a member of root node `kl`, a
batch whose argument 0 is the input node `q`'s leaves, resolves, to a gate of `q`. -/
theorem param_lands (hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]!)
    (hc : NciCheck.Desc.calleesFirst P = true) (hrp : (P.defs[P.root]!).body.params.size = 0)
    (hz : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).form = .scan →
        ((P.defs[d]!).body.nodes[k]!).n = 0 → ((P.defs[d]!).body.nodes[k]!).out.leaves = 0)
    (hfuel : 4 * (NciCheck.Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL)
    {kl m j q : ℕ} (hk : kl < (P.defs[P.root]!).body.nodes.size)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).prim = false)
    (hbatch : ((P.defs[P.root]!).body.nodes[kl]!).form = .batch)
    (hm : m < ((P.defs[P.root]!).body.nodes[kl]!).members)
    (hp0 : 0 < (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).params.size)
    (hj : j < ((P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).params[0]!).2.leaves)
    (hq : inputArg P (((P.defs[P.root]!).body.nodes[kl]!).args[0]!) = some q) :
    ∃ a, resolve P (P.child P.rootScope kl m) ⟨true, 0, j⟩ = .ok a ∧
      (P.defs[P.root]!).body.offsets[q]! ≤ a ∧ a < (P.defs[P.root]!).body.offsets[q + 1]! := by
  have hBr : Resolve.BodyOk P P.root := hB P.rootScope (good_root P)
  have hfnlt : ((P.defs[P.root]!).body.nodes[kl]!).fn < P.defs.size := (hBr.1 kl hk).1
  have hbp : (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).body.params =
      (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).params := ((hok _ hfnlt).1.2 hcomp).1
  obtain ⟨-, hK, hnodes⟩ := hL P.root hroot.1 hroot.2
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hn := hnodes kl hk
  have hkl1 : (P.defs[P.root]!).body.offsets[kl + 1]! ≤ (P.defs[P.root]!).gates := by
    rw [← hK]; exact hmono _ _ (by omega) (le_refl _)
  have hle : (m + 1) * (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).gates ≤
      ((P.defs[P.root]!).body.nodes[kl]!).members * (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).gates :=
    Nat.mul_le_mul_right _ hm
  rw [Nat.add_mul, Nat.one_mul] at hle
  have hgA : Resolve.Good P (P.child P.rootScope kl m) := Resolve.good_child P (good_root P) hk rfl hcomp hm rfl
  have hp0' : 0 < (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).body.params.size := by rw [hbp]; exact hp0
  have hj' : j < ((P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).body.params[0]!).2.leaves := by
    rw [hbp]; exact hj
  have hinv : Resolve.Inv P (P.defs[P.root]!).gates (P.child P.rootScope kl m, ⟨true, 0, j⟩) := by
    refine ⟨hgA, ⟨fun _ => ⟨?_, ?_⟩, fun h => absurd h (by decide)⟩, fun _ => ?_, fun h => absurd h (by decide)⟩
    · show 0 < (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).body.params.size
      exact hp0'
    · show j < ((P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).body.params[0]!).2.leaves
      exact hj'
    · show 0 + (P.defs[P.root]!).body.offsets[kl]! + m * (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).gates ≤
        (P.defs[P.root]!).gates
      omega
  obtain ⟨a, ha⟩ := OperandsResolve.resolve_ok P hroot hB hL hc hrp hz hfuel _ _ _ hinv
  refine ⟨a, ha, ?_⟩
  have hstep : resolveStep P (P.child P.rootScope kl m, ⟨true, 0, j⟩) =
      .ok (.inr (P.rootScope, (((P.defs[P.root]!).body.nodes[kl]!).args[0]!).memberRef m j)) :=
    Resolve.rs_up_batch P P.rootScope _ _ kl m 0 j hbatch
  unfold Flock.Program.resolve at ha
  rw [fuel_succ, Flock.Steps.run.eq_2, hstep] at ha
  simp only [bind, Except.bind] at ha
  rw [Resolve.memberRef_eq] at ha
  obtain ⟨-, -, hup, -, -⟩ := hBr.1 kl hk
  have ht := hup 0 hp0' j hj' m hm
  simp only [hbatch] at ht
  exact arg_lands P hroot hB hL hk hq ht _ a ha

/-- `Prog.gate`'s first step to a gate of an input node locates that node's primitive, whose operands are none. -/
theorem gateStep_input {q x : ℕ} {r : (ℕ × Array ℕ × Scope) ⊕ Scope} (hq : q < (P.defs[P.root]!).body.nodes.size)
    (hprim : (P.defs[((P.defs[P.root]!).body.nodes[q]!).fn]!).prim = true)
    (hx1 : (P.defs[P.root]!).body.offsets[q]! ≤ x) (hx2 : x < (P.defs[P.root]!).body.offsets[q + 1]!)
    (hargs : ((P.defs[P.root]!).body.nodes[q]!).args = #[]) (hs : gateStep P x P.rootScope = .ok r) :
    ∃ ops, r = .inl (((P.defs[P.root]!).body.nodes[q]!).fn, ops, P.rootScope) ∧ ops = #[] := by
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hmono' : ∀ k' < (P.defs[P.root]!).body.nodes.size,
      (P.defs[P.root]!).body.offsets[k']! ≤ (P.defs[P.root]!).body.offsets[k' + 1]! :=
    fun k' hk' => hmono k' (k' + 1) (by omega) (by omega)
  unfold gateStep at hs
  simp only [bind, Except.bind, rootScope_d, rootScope_off, Nat.sub_zero] at hs
  split at hs
  · rename_i k' hk'
    obtain ⟨hk'K, hpk⟩ := DescAgree.find?_range_some hk'
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    have hkk : k' = q := DescAgree.node_unique _ _ hmono' hk'K hq
      (by first | exact hpk | exact ⟨of_decide_eq_true hpk.1, of_decide_eq_true hpk.2⟩) ⟨hx1, hx2⟩
    subst k'
    simp only [hprim, ite_true] at hs
    split at hs
    · cases hs
    · rename_i ops hops
      simp only [pure, Except.pure, Except.ok.injEq] at hs
      exact ⟨ops, hs.symm, Walk.operands_nil P _ _ _ ops hargs hops⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at hs

/-- **A gate of an input node reads nothing.** -/
theorem gate_input {q x p : ℕ} {ops : Array ℕ} {sc : Scope} (hq : q < (P.defs[P.root]!).body.nodes.size)
    (hin : NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[q]!) = true)
    (hx1 : (P.defs[P.root]!).body.offsets[q]! ≤ x) (hx2 : x < (P.defs[P.root]!).body.offsets[q + 1]!)
    (hargs : ((P.defs[P.root]!).body.nodes[q]!).args = #[]) (h : P.gate x = .ok (p, ops, sc)) : ops = #[] := by
  obtain ⟨-, hK, -⟩ := hL P.root hroot.1 hroot.2
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hxg : x < (P.defs[P.root]!).gates := by
    have := hmono (q + 1) _ (by omega) (le_refl _)
    omega
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[q]!).fn]!).prim = true := by
    simp only [NciCheck.Desc.isInputNode, Bool.and_eq_true] at hin; exact hin.1
  unfold Prog.gate at h
  have hng : ¬ (x ≥ (P.defs[P.root]!).gates) := by omega
  simp only [hng, ite_false, bind, Except.bind] at h
  rw [fuel_succ, Flock.Steps.run.eq_2] at h
  rcases hs : gateStep P x P.rootScope with e | r
  · simp [hs, bind, Except.bind] at h
  · obtain ⟨ops', rfl, hops⟩ := gateStep_input P hroot hB hL hq hprim hx1 hx2 hargs hs
    simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    rw [← h.2.1, hops]

/-- **`InS` holds what each of its gates reads**, in a flattening of a Program C-Flock decodes, topological and laid
out, at a root node `ke` batching a composite whose every argument is one input node's leaves. -/
theorem closure {bytes : ByteArray} (hdec : decode bytes = .ok P) (hov : P.orderViolation = none)
    (hbl : NciCheck.Desc.bodiesLaidOut P = true) (C : Claims.Sampling.Prog) (hF : Flattens P C)
    {ke : ℕ} (hk : ke < (P.defs[P.root]!).body.nodes.size)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).prim = false)
    (hbatch : ((P.defs[P.root]!).body.nodes[ke]!).form = .batch)
    (hargs : ∀ b ∈ ((P.defs[P.root]!).body.nodes[ke]!).args, (inputArg P b).isSome = true)
    (hinp : ∀ k < (P.defs[P.root]!).body.nodes.size,
      NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = true →
        ((P.defs[P.root]!).body.nodes[k]!).args = #[])
    (x y : ℕ) (hy : y ∈ (C.gates[x]?.map Claims.Sampling.BGate.reads).getD []) (hx : InS P ke x) :
    InS P ke y := by
  have hxl : x < C.gates.length := by
    by_contra hn
    rw [List.getElem?_eq_none (by omega)] at hy
    simp at hy
  obtain ⟨p, ops, sc, hgate, -, hreads⟩ := hF.2.1 x hxl
  have hyo : y ∈ ops := (hreads y).1 hy
  obtain ⟨-, -, hnodes⟩ := hL P.root hroot.1 hroot.2
  rcases hx with ⟨hx1, hx2⟩ | ⟨b, -, q, hq, hx1, hx2⟩
  · have hn := hnodes ke hk
    have hx' : x - (P.defs[P.root]!).body.offsets[ke]! <
        ((P.defs[P.root]!).body.nodes[ke]!).members * (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).gates := by
      omega
    obtain ⟨hm, hm1, hm2⟩ := member_div hx'
    have hle : ((x - (P.defs[P.root]!).body.offsets[ke]!) /
        (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).gates + 1) *
        (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).gates ≤
        ((P.defs[P.root]!).body.nodes[ke]!).members * (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).gates :=
      Nat.mul_le_mul_right _ hm
    rw [Nat.add_mul, Nat.one_mul] at hle
    have hf : ((P.defs[P.root]!).body.nodes[ke]!).form ≠ .scan := fun h => by rw [hbatch] at h; cases h
    rcases memberReads bytes P hdec hov hbl ke hk _ hm hcomp hf x p ops sc hgate
      (by rw [child_off]; omega) (by rw [child_off]; omega) y hyo with h | ⟨i, t, ht, h⟩
    · rw [child_off] at h
      exact Or.inl ⟨by omega, by omega⟩
    · have hi := arg_lt ht
      have hmem : ((P.defs[P.root]!).body.nodes[ke]!).args[i]! ∈ ((P.defs[P.root]!).body.nodes[ke]!).args := by
        rw [getElem!_pos ((P.defs[P.root]!).body.nodes[ke]!).args i hi]; exact Array.getElem_mem hi
      obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 (hargs _ hmem)
      exact Or.inr ⟨_, hmem, q, hq, arg_lands P hroot hB hL hk hq ht _ y h⟩
  · obtain ⟨-, hqK, hin⟩ := inputArg_some hq
    have := gate_input P hroot hB hL hqK hin hx1 hx2 (hinp q hqK hin) hgate
    rw [this] at hyo
    simp at hyo

end

/-- What `scanOf`'s acceptance gives the causal argument, at a window of at least two steps: the egress template's root
node `ke` (the last template) and the later ingress template's `kl` (the second), each a batch of a composite, of
`T` and `T - 1` members; the egress template's return, a u16 and the schedule's row; the later ingress template's
first parameter, the schedule's row, bound to the input node `q'`; the egress template's every argument one input
node's leaves, none `q'`; and the Program laid out within what resolution walks. -/
structure Facts (P : Prog) (s : Schedule) (kl ke q' : ℕ) : Prop where
  last : (templateNodes P).getLast?.getD 0 = ke
  second : (templateNodes P).getD 1 0 = kl
  kl_lt : kl < (P.defs[P.root]!).body.nodes.size
  ke_lt : ke < (P.defs[P.root]!).body.nodes.size
  l_comp : (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).prim = false
  e_comp : (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).prim = false
  l_batch : ((P.defs[P.root]!).body.nodes[kl]!).form = .batch
  e_batch : ((P.defs[P.root]!).body.nodes[ke]!).form = .batch
  l_n : ((P.defs[P.root]!).body.nodes[kl]!).n = s.window - 1
  e_n : ((P.defs[P.root]!).body.nodes[ke]!).n = s.window
  e_ret : (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).ret =
    .tuple #[.array outBits (.value 1), .array (s.allocation * s.cellBytes) (.array 8 (.value 1))]
  l_p0 : 0 < (P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).params.size
  l_row : ((P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).params[0]!).2 =
    .array (s.allocation * s.cellBytes) (.array 8 (.value 1))
  walk : NciCheck.Desc.walkable P = true
  row_in : inputArg P (((P.defs[P.root]!).body.nodes[kl]!).args[0]!) = some q'
  e_args : ∀ b ∈ ((P.defs[P.root]!).body.nodes[ke]!).args, (inputArg P b).isSome = true ∧ inputArg P b ≠ some q'

/-- **Part (3) of `NetTiming.Layout.Fits` at `scanLayout`, from `Facts`.** -/
theorem causal_of_facts {bytes : ByteArray} {P : Prog} {s : Schedule} {kl ke q' : ℕ}
    (hdec : decode bytes = .ok P) (hov : P.orderViolation = none) (F : Facts P s kl ke q')
    (C : Claims.Sampling.Prog) (hF : Flattens P C) :
    ∀ t < s.window, ∀ t' < s.window, t < t' → ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s).egress t ℓ σ),
      ∀ i ∈ (scanLayout P s).ingress t', ¬ NetTiming.Depends C i g := by
  obtain ⟨hr, hrprim, -⟩ := BodiesOk.decode_root bytes P hdec
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := BodiesOk.order_closed P hov
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc hg => by
    obtain ⟨hSd, hpd⟩ := BodiesOk.good_closed P S hS0 hrprim (fun d hd hp => (hS d hd hp).2) sc hg
    exact BodiesOk.body_ok P hok sc.d (hS sc.d hSd hpd).1
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hr, hrprim⟩
  have hwk := F.walk
  unfold NciCheck.Desc.walkable at hwk
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hwk
  obtain ⟨⟨hc, hbl⟩, hres, hwb⟩ := hwk
  obtain ⟨-, hL, -, hinp⟩ := Walk.bodiesLaidOut_spec P hbl
  obtain ⟨hpar, hz⟩ := OperandsResolve.resolvable_spec P hres
  have hfuel : 4 * (NciCheck.Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL := by
    unfold NciCheck.Desc.walkBound at hwb
    have : ((NciCheck.Desc.activations P)[P.root]! + 1) * 2 ≤
        ((NciCheck.Desc.activations P)[P.root]! + 1) * ((NciCheck.Desc.depths P)[P.root]! + 2) :=
      Nat.mul_le_mul_left _ (by omega)
    omega
  have hinp' : ∀ k < (P.defs[P.root]!).body.nodes.size,
      NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = true →
        ((P.defs[P.root]!).body.nodes[k]!).args = #[] := hinp P.root hr hrprim
  have hBr : Resolve.BodyOk P P.root := hB P.rootScope (good_root P)
  -- the egress node's leaves: `T` members of a u16 and the row
  have hfe : ((P.defs[P.root]!).body.nodes[ke]!).fn < P.defs.size := (hBr.1 ke F.ke_lt).1
  obtain ⟨-, -, hol⟩ := (hok _ hfe).1.2 F.e_comp
  obtain ⟨-, hnode, -⟩ := (hok P.root hr).2.1 ke F.ke_lt
  obtain ⟨-, -, -, -, hsh, -⟩ := hnode
  simp only [BodiesOk.Shape, F.e_batch] at hsh
  have hle : ((P.defs[P.root]!).body.nodes[ke]!).out.leaves =
      s.window * (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves := by
    rw [hsh.1, BodiesOk.leaves_array, F.e_n, hol]
  have hL16 : (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves =
      outBits + s.allocation * (8 * s.cellBytes) := by
    simp only [hol, F.e_ret, BodiesOk.leaves_tuple2, BodiesOk.leaves_array, BodiesOk.leaves_value]
    ring
  have h16 : outBits = 16 := rfl
  have hmono := Resolve.offsets_mono P P.root (hL _ hr) hrprim
  have hmono' : ∀ k' < (P.defs[P.root]!).body.nodes.size,
      (P.defs[P.root]!).body.offsets[k']! ≤ (P.defs[P.root]!).body.offsets[k' + 1]! :=
    fun k' hk' => hmono k' (k' + 1) (by omega) (by omega)
  intro t ht t' ht' htt' ℓ σ g hg i hi hdep
  -- the egress gate is in `InS`
  have hegress : ∀ j < (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves,
      InS P ke (gateAt P P.rootScope ⟨false, ke,
        t * (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves + j⟩) := by
    intro j hj
    have hjl : t * (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves + j <
        ((P.defs[P.root]!).body.nodes[ke]!).out.leaves := by
      rw [hle]; exact BodiesOk.lt_mul_of _ _ _ _ ht hj
    obtain ⟨a, ha, hS⟩ := batch_lands P hroot hB hL hc hpar hz hfuel F.ke_lt F.e_comp F.e_batch
      (fun b hb => (F.e_args b hb).1) hjl
    have : gateAt P P.rootScope ⟨false, ke,
        t * (P.defs[((P.defs[P.root]!).body.nodes[ke]!).fn]!).outLeaves + j⟩ = a := by
      unfold gateAt; rw [ha]; rfl
    rw [this]; exact hS
  have hgS : InS P ke g := by
    simp only [scanLayout, NetTiming.slotGates, List.mem_cons, List.mem_map, List.mem_range, F.last] at hg
    rcases hg with rfl | ⟨b, hb, rfl⟩
    · exact hegress 0 (by rw [hL16, h16]; omega)
    · apply hegress
      rw [hL16]
      have := BodiesOk.lt_mul_of σ.val s.allocation (8 * s.cellBytes) b σ.isLt hb
      omega
  -- the ingress gate is a gate of the row's input node `q'`
  have ht'0 : t' ≠ 0 := by omega
  simp only [scanLayout, List.mem_map, List.mem_range, ht'0, ↓reduceIte, F.second] at hi
  obtain ⟨j, hj, rfl⟩ := hi
  have hm : t' - 1 < ((P.defs[P.root]!).body.nodes[kl]!).members := by
    rw [Resolve.bs_members _ (Or.inl F.l_batch), F.l_n]; omega
  have hj' : j < ((P.defs[((P.defs[P.root]!).body.nodes[kl]!).fn]!).params[0]!).2.leaves := by
    simp only [F.l_row, BodiesOk.leaves_array, BodiesOk.leaves_value]
    have : s.allocation * s.cellBytes * (8 * 1) = s.allocation * (8 * s.cellBytes) := by ring
    omega
  obtain ⟨a, ha, h1, h2⟩ := param_lands P hroot hB hL hok hc hpar hz hfuel F.kl_lt F.l_comp F.l_batch hm F.l_p0 hj'
    F.row_in
  have hia : gateAt P (P.child P.rootScope kl (t' - 1)) ⟨true, 0, j⟩ = a := by
    unfold gateAt; rw [ha]; rfl
  rw [hia] at hdep
  obtain ⟨-, hq'K, hq'in⟩ := inputArg_some F.row_in
  have hiS := depends_closed C (InS P ke)
    (closure P hroot hB hL hdec hov hbl C hF F.ke_lt F.e_comp F.e_batch (fun b hb => (F.e_args b hb).1) hinp')
    hdep hgS
  rcases hiS with ⟨h3, h4⟩ | ⟨b, hb, q, hq, h3, h4⟩
  · have hkq := DescAgree.node_unique _ _ hmono' F.ke_lt hq'K ⟨h3, h4⟩ ⟨h1, h2⟩
    subst hkq
    simp only [NciCheck.Desc.isInputNode, Bool.and_eq_true, F.e_comp, Bool.false_eq_true, false_and] at hq'in
  · obtain ⟨-, hqK, -⟩ := inputArg_some hq
    have hqq := DescAgree.node_unique _ _ hmono' hqK hq'K ⟨h3, h4⟩ ⟨h1, h2⟩
    subst hqq
    exact (F.e_args b hb).2 hq

theorem form_eq_batch {f : Flock.Program.Form} (h : (f == .batch) = true) : f = .batch := by
  cases f <;> first | rfl | exact absurd h (by decide)

/-- A template node is a root node calling a composite. -/
theorem templateNodes_mem {P : Prog} {k : ℕ} (h : k ∈ templateNodes P) :
    k < (P.defs[P.root]!).body.nodes.size ∧ (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false := by
  unfold templateNodes at h
  simp only [List.mem_filter, List.mem_range] at h
  obtain ⟨hk, hd⟩ := h
  refine ⟨hk, ?_⟩
  split at hd
  · rename_i d hdef
    obtain ⟨hlt, heq⟩ := Array.getElem?_eq_some_iff.1 hdef
    rw [getElem!_pos P.defs _ hlt, heq]; simpa using hd
  · cases hd

/-- **What `scanOf`'s acceptance gives**, at a window of at least two steps: `Facts`, at the second template's node,
the last's, and the input node the second's row reads. -/
theorem scanOf_facts {j : Lean.Json} {program : String} {s : Schedule} {sc : Scan} {c : String} {P : Prog}
    (h : scanOf j program s = .ok sc) (hc : Flock.canon j = .ok c) (hP : decode c.toUTF8 = .ok P)
    (hT : 2 ≤ s.window) : ∃ kl ke q', Facts P s kl ke q' := by
  unfold scanOf at h
  obtain ⟨c', hc', h⟩ := Fits.bind_ok h
  obtain rfl : c' = c := by rw [hc] at hc'; exact (Except.ok.inj hc').symm
  obtain ⟨P', hP', h⟩ := Fits.bind_ok h
  rw [hP] at hP'
  obtain rfl : P = P' := by cases hP'; rfl
  obtain ⟨_, -, h⟩ := Fits.bind_ok h
  obtain ⟨root, hroot, h⟩ := Fits.bind_ok h
  obtain rfl : root = P.defs[P.root]! := by
    obtain ⟨hlt, e⟩ := Array.getElem?_eq_some_iff.1 (Fits.needSome_ok hroot)
    rw [getElem!_pos P.defs _ hlt, e]
  obtain ⟨⟨xe, de⟩, hback, h⟩ := Fits.bind_ok h
  replace hback := Fits.needSome_ok hback
  obtain ⟨_, -, h⟩ := Fits.bind_ok h
  obtain ⟨⟨⟨xf, df⟩, later⟩, hm, h⟩ := Fits.bind_ok h
  obtain ⟨_, hnames, h⟩ := Fits.bind_ok h
  obtain ⟨_, hcount, h⟩ := Fits.bind_ok h
  obtain ⟨_, hrow, h⟩ := Fits.bind_ok h
  obtain ⟨_, -, h⟩ := Fits.bind_ok h
  obtain ⟨_, hret, h⟩ := Fits.bind_ok h
  obtain ⟨_, -, h⟩ := Fits.bind_ok h
  obtain ⟨_, hwalk, h⟩ := Fits.bind_ok h
  obtain ⟨_, -, h⟩ := Fits.bind_ok h
  obtain ⟨_, -, h⟩ := Fits.bind_ok h
  obtain ⟨_, hrows, h⟩ := Fits.bind_ok h
  obtain ⟨_, hargs, h⟩ := Fits.bind_ok h
  obtain ⟨_, hegr, -⟩ := Fits.bind_ok h
  replace hnames := Fits.need_ok hnames
  replace hcount := Fits.need_ok hcount
  replace hrow := Fits.need_ok hrow
  replace hret := Fits.need_ok hret
  replace hwalk := Fits.need_ok hwalk
  replace hrows := Fits.need_ok hrows
  replace hargs := Fits.need_ok hargs
  replace hegr := Fits.need_ok hegr
  simp only [Bool.and_eq_true] at hcount
  obtain ⟨⟨⟨⟨-, -⟩, hlater⟩, heb⟩, hen⟩ := hcount
  -- the templates: `templateNodes P = [k0, kl, ke]`
  rcases Fits.ingressTemplates_ok hm with ⟨f, -, hx⟩ | ⟨f, b, hpop, hx⟩
  · simp only [Prod.mk.injEq] at hx
    obtain ⟨-, rfl⟩ := hx
    simp only [beq_iff_eq, Option.some.injEq] at hlater
    omega
  simp only [Prod.mk.injEq] at hx
  obtain ⟨rfl, rfl⟩ := hx
  simp only [Array.back?_map, List.back?_toArray, Option.map_eq_some_iff] at hback
  obtain ⟨ke, hke, hge⟩ := hback
  simp only [Array.toList_pop, Array.toList_map] at hpop
  obtain ⟨ys, hys⟩ := List.getLast?_eq_some_iff.1 hke
  rw [hys, List.map_append, List.map_cons, List.map_nil, List.dropLast_concat] at hpop
  obtain ⟨k0, kl, rfl⟩ : ∃ k0 kl, ys = [k0, kl] := by
    rcases ys with _ | ⟨k0, _ | ⟨kl, _ | ⟨k2, ys⟩⟩⟩ <;> simp at hpop
    exact ⟨_, _, rfl⟩
  simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at hpop
  obtain ⟨hf, rfl⟩ := hpop
  simp only [Prod.mk.injEq] at hge hf
  obtain ⟨rfl, rfl⟩ := hge
  obtain ⟨rfl, rfl⟩ := hf
  simp only [List.cons_append, List.nil_append] at hys
  have hmem_l : kl ∈ templateNodes P := by rw [hys]; simp
  have hmem_e : ke ∈ templateNodes P := by rw [hys]; simp
  obtain ⟨hkl, hlc⟩ := templateNodes_mem hmem_l
  obtain ⟨hke', hec⟩ := templateNodes_mem hmem_e
  -- the counts
  dsimp only at hlater
  split at hlater
  case isFalse => simp at hlater
  rename_i hlb
  simp only [decide_eq_true_eq] at hlb
  simp only [beq_iff_eq, Option.some.injEq] at hlater hen
  -- the rows, the arguments and the names
  simp only [Option.map_some, Option.getD_some, Array.append_singleton, List.push_toArray, List.cons_append,
    List.nil_append, List.map_toArray, List.map_cons, List.map_nil, List.all_toArray', List.all_cons, List.all_nil,
    Bool.and_true, Bool.and_eq_true] at hrows
  obtain ⟨-, hr1⟩ := hrows
  obtain ⟨q', hq'⟩ := Option.isSome_iff_exists.1 hr1
  obtain ⟨a0, ha0, hia⟩ := Option.bind_eq_some_iff.1 hq'
  have hrow_in : inputArg P (((P.defs[P.root]!).body.nodes[kl]!).args[0]!) = some q' := by
    obtain ⟨hlt, heq⟩ := Array.getElem?_eq_some_iff.1 ha0
    rw [getElem!_pos ((P.defs[P.root]!).body.nodes[kl]!).args 0 hlt, heq]; exact hia
  simp only [Option.map_some, Option.getD_some, Array.append_singleton, List.push_toArray, List.cons_append,
    List.nil_append, List.all_toArray', List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true,
    Array.all_eq_true] at hargs
  obtain ⟨-, -, hae⟩ := hargs
  simp only [Option.map_some, Option.getD_some, Array.append_singleton, List.push_toArray, List.cons_append,
    List.nil_append, List.map_toArray, List.map_cons, List.map_nil, Array.contains_eq_mem, List.mem_toArray,
    List.mem_cons, List.not_mem_nil, or_false, Bool.decide_or, Bool.not_or, Array.all_eq_true, Bool.and_eq_true,
    Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_false_iff_not] at hegr
  simp only [Option.all_some, Bool.and_eq_true, beq_iff_eq] at hnames
  obtain ⟨⟨-, hnl⟩, -⟩ := hnames
  simp only [Option.all_some, Bool.and_eq_true] at hrow hwalk
  refine ⟨kl, ke, q', ?_⟩
  exact {
    last := by rw [hys]; rfl
    second := by rw [hys]; rfl
    kl_lt := hkl
    ke_lt := hke'
    l_comp := hlc
    e_comp := hec
    l_batch := form_eq_batch hlb.1
    e_batch := form_eq_batch heb
    l_n := hlater
    e_n := hen
    e_ret := BodiesOk.ty_beq hret
    l_p0 := by
      have := congrArg Array.size hnl
      rw [Array.size_map] at this
      rw [this]; decide
    l_row := BodiesOk.ty_beq hrow.2
    walk := hwalk.2
    row_in := hrow_in
    e_args := fun b hb => by
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hb
      refine ⟨hae i hi, fun he => (hegr i hi).2 ?_⟩
      rw [he, hq'] }

/-- **The scan is causal.** -/
theorem scanCausal : ScanCausal := by
  intro j program s sc c P C h hc hP hov hF t ht t' ht' htt'
  obtain ⟨kl, ke, q', F⟩ := scanOf_facts h hc hP (by omega)
  exact causal_of_facts hP hov F C hF t ht t' ht' htt'

end Properties.NetworkScan.CausalProof
