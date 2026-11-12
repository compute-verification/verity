module

public import Properties.NetworkScan.Chain
public import Proofs.Properties.NetworkScan.Causal
public import Proofs.Properties.NetworkScan.Fits
public import Proofs.NetworkCertifier.SampledCharge

/-!
# The scan's cut, proved

`Properties.NetworkScan.ScanViewCut`: at an accepted scan, the free inputs other than the egress units' `state` gates
are checked inputs under no check at all (`NetTiming.CheckedUnique` at the check that holds of every run).

**Checked inputs from what the egress reads** (`NetTiming.checkedUnique_of_reads`), for any program and layout: when no
egress gate of a step depends on an ingress gate of a later step, and no egress gate depends on a gate of `chk`, the
inputs `chk` are checked under every check. Two runs whose ingress agrees through step `t` and whose inputs agree off
`chk` read one input at every gate an egress gate of step `t` depends on (`NetTiming.inputOf_congr_checked`): such a
gate is no later step's ingress, an ingress gate of a step through `t` reads the same ingress bit, and any other gate
reads inputs that agree, being outside `chk`. So the egress gates hold one value in both
(`NetTiming.valOver_congr_inp`).

**The egress reads only the view gates.** Let `ke` be the egress template's root node and `V` the gates of `ke`'s
members together with the view gates (`viewSet`: the leaves of each member's `state` parameter, as gates). An egress
gate is a gate of `ke` (`Fits.egress_gates`). `V` is closed under reads (`view_closure`): a gate of a member of `ke`
reads its activation's gates or what a leaf of one of its parameters resolves to (the walk lemma at the parameters,
`WalkProof.memberReads_params`: a template's body reads only its parameters), and the egress template has one
parameter, `state` (`Fits.Accepted.egress_params`); a view gate is a gate of the input node the `state` argument reads
(`CausalProof.param_lands`), which reads nothing (`CausalProof.gate_input`). So everything an egress gate depends on is
in `V` (`CausalProof.depends_closed`): a gate of `ke`, which is no input (`Fits.gate_not_input`) and so no free input,
or a view gate. Neither is a free input other than a view gate, and the scan is causal (`CausalProof.scanCausal`).
-/

@[expose] public section

namespace NetTiming

open Claims.Sampling

/-- Two runs over the horizon `n` read one input at the gate `i`, when their ingress agrees through step `t`, `i` is no
ingress gate of a later step, and their other inputs agree at `i`. -/
theorem inputOf_congr_checked {L S : ℕ} (lay : BoundaryGates L S) {n t : ℕ} {ins ins' : List (List Bool)}
    {a a' : ℕ → Bool} (hl : ins.length = n) (hl' : ins'.length = n)
    (hins : ∀ j ≤ t, ins.getD j [] = ins'.getD j []) {i : ℕ}
    (hlate : ∀ t' < n, t < t' → i ∉ lay.ingress t') (ha : a i = a' i) :
    lay.inputOf n ins a i = lay.inputOf n ins' a' i := by
  unfold BoundaryGates.inputOf
  rw [hl, hl']
  rcases hR : (List.range n).findSome? (fun j => ((lay.ingress j).findIdx? (· == i)).map (j, ·)) with
    _ | ⟨j, idx⟩
  · simp only [ha]
  · obtain ⟨j', hj', hfj⟩ := List.exists_of_findSome?_eq_some hR
    obtain ⟨idx', hidx, hp⟩ := Option.map_eq_some_iff.mp hfj
    simp only [Prod.mk.injEq] at hp
    obtain ⟨rfl, rfl⟩ := hp
    obtain ⟨hjl, hjg, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hidx
    have hmem : i ∈ lay.ingress j' := by
      have hm := List.getElem_mem hjl
      rwa [beq_iff_eq.1 hjg] at hm
    have hjt : j' ≤ t := by
      by_contra hlt
      exact hlate j' (List.mem_range.1 hj') (by omega) hmem
    simp only
    rw [hins j' hjt]

/-- **Checked inputs from what the egress reads.** When no egress gate of a step depends on an ingress gate of a later
step (`BoundaryGates.Fits`'s part (3)) and no egress gate depends on a gate of `chk`, the inputs `chk` are checked
under every check `valid`: runs whose ingress agrees through a step and whose inputs agree off `chk` give that step
one egress row. -/
theorem checkedUnique_of_reads {L S : ℕ} (C : Prog) (Pt : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (chk : Finset ℕ) (valid : (ℕ → Bool) → Prop)
    (hcausal : ∀ t < n, ∀ t' < n, t < t' → ∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ), ∀ i ∈ lay.ingress t',
      ¬ Depends C i g)
    (hreads : ∀ t < n, ∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ), ∀ i, Depends C i g → i ∉ chk) :
    CheckedUnique C Pt d lay n chk valid := by
  intro t ht ins ins' a a' free hl hl' hins ha _ _ _ ℓ
  apply rowOf_congr
  intro σ g hg
  apply valOver_congr_inp
  intro i hdep
  exact inputOf_congr_checked lay hl hl' hins
    (fun t' ht' htt' hmem => hcausal t ht t' ht' htt' ℓ σ g hg i hmem hdep) (ha i (hreads t ht ℓ σ g hg i hdep))

end NetTiming

namespace Properties.NetworkScan.ViewCutProof

open Flock.Program
open NciCheck.Properties.SecurityProofs
open NetworkCheck.Period

/-- A leaf of the egress template's `state` at a member before `T`, as a gate, is a view gate. -/
theorem mem_viewSet {P : Prog} {s : Schedule} {y t j : ℕ} (ht : t < s.window)
    (hj : j < ((P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).params[0]!).2.leaves)
    (hy : gateAt P (P.child P.rootScope (Fits.egressNode P) t) ⟨true, 0, j⟩ = y) : y ∈ viewSet P s :=
  List.mem_toFinset.2 (List.mem_flatMap.2 ⟨t, List.mem_range.2 ht, List.mem_map.2 ⟨j, List.mem_range.2 hj, hy⟩⟩)

/-- A view gate is a leaf of the egress template's `state` at a member before `T`, as a gate. -/
theorem of_mem_viewSet {P : Prog} {s : Schedule} {y : ℕ} (h : y ∈ viewSet P s) :
    ∃ t < s.window, ∃ j < ((P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).params[0]!).2.leaves,
      gateAt P (P.child P.rootScope (Fits.egressNode P) t) ⟨true, 0, j⟩ = y := by
  obtain ⟨t, ht, hy⟩ := List.mem_flatMap.1 (List.mem_toFinset.1 h)
  obtain ⟨j, hj, hjy⟩ := List.mem_map.1 hy
  exact ⟨t, List.mem_range.1 ht, j, List.mem_range.1 hj, hjy⟩

/-- The gates an egress gate's dependencies stay in: those of the egress template's root node's members, and the view
gates. -/
abbrev InV (P : Prog) (s : Schedule) (x : ℕ) : Prop :=
  ((P.defs[P.root]!).body.offsets[Fits.egressNode P]! ≤ x ∧
      x < (P.defs[P.root]!).body.offsets[Fits.egressNode P + 1]!) ∨ x ∈ viewSet P s

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

/-- **`InV` holds what each of its gates reads**, in a flattening of a Program C-Flock decodes, topological and laid
out, whose egress template's root node batches `T` members of a composite of one parameter, bound to one input node's
leaves: a member's gate reads its activation's gates or a leaf of that parameter, a view gate, and a view gate is a
gate of that input node, which reads nothing. -/
theorem view_closure {bytes : ByteArray} {s : Schedule} (hdec : decode bytes = .ok P)
    (hov : P.orderViolation = none) (hbl : NciCheck.Desc.bodiesLaidOut P = true)
    (hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]!)
    (hc : NciCheck.Desc.calleesFirst P = true) (hrp : (P.defs[P.root]!).body.params.size = 0)
    (hz : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).form = .scan →
        ((P.defs[d]!).body.nodes[k]!).n = 0 → ((P.defs[d]!).body.nodes[k]!).out.leaves = 0)
    (hfuel : 4 * (NciCheck.Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL)
    (C : Claims.Sampling.Prog) (hF : Flattens P C)
    (hk : Fits.egressNode P < (P.defs[P.root]!).body.nodes.size)
    (hcomp : (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).prim = false)
    (hbatch : ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).form = .batch)
    (hn : ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).n = s.window)
    (hp1 : (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).params.size = 1) {q : ℕ}
    (hq : inputArg P (((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).args[0]!) = some q)
    (hinp : ∀ k < (P.defs[P.root]!).body.nodes.size,
      NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = true →
        ((P.defs[P.root]!).body.nodes[k]!).args = #[])
    (x y : ℕ) (hy : y ∈ (C.gates[x]?.map Claims.Sampling.BGate.reads).getD []) (hx : InV P s x) : InV P s y := by
  have hxl : x < C.gates.length := by
    by_contra hnl
    rw [List.getElem?_eq_none (by omega)] at hy
    simp at hy
  obtain ⟨p, ops, sc, hgate, -, hreads⟩ := hF.2.1 x hxl
  have hyo : y ∈ ops := (hreads y).1 hy
  obtain ⟨-, -, hnodes⟩ := hL P.root hroot.1 hroot.2
  have hBr : Resolve.BodyOk P P.root := hB P.rootScope (WalkProof.good_root P)
  have hfnlt : ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn < P.defs.size := (hBr.1 _ hk).1
  have hbp : (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).body.params =
      (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).params := ((hok _ hfnlt).1.2 hcomp).1
  have hmem : ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).members = s.window := by
    rw [Resolve.bs_members _ (Or.inl hbatch), hn]
  rcases hx with ⟨hx1, hx2⟩ | hxv
  · have hnk := hnodes (Fits.egressNode P) hk
    have hx' : x - (P.defs[P.root]!).body.offsets[Fits.egressNode P]! <
        ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).members *
          (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).gates := by
      omega
    obtain ⟨m, hm, hm1, hm2⟩ : ∃ m, m < ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).members ∧
        m * (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).gates ≤
          x - (P.defs[P.root]!).body.offsets[Fits.egressNode P]! ∧
        x - (P.defs[P.root]!).body.offsets[Fits.egressNode P]! <
          m * (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).gates +
            (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).gates :=
      ⟨_, CausalProof.member_div hx'⟩
    have hle : (m + 1) * (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).gates ≤
        ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).members *
          (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).gates :=
      Nat.mul_le_mul_right _ hm
    rw [Nat.add_mul, Nat.one_mul] at hle
    have hf : ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).form ≠ .scan := fun h => by
      rw [hbatch] at h; cases h
    rcases WalkProof.memberReads_params hdec hov hbl hk hm hcomp hf hgate
      (by rw [WalkProof.child_off]; omega) (by rw [WalkProof.child_off]; omega) hyo with h | ⟨i, j, hi, hj, h⟩
    · rw [WalkProof.child_off] at h
      exact Or.inl ⟨by omega, by omega⟩
    · rw [hbp, hp1] at hi
      have hi0 : i = 0 := by omega
      subst hi0
      rw [hbp] at hj
      have hmT : m < s.window := by rw [← hmem]; exact hm
      have hgy : gateAt P (P.child P.rootScope (Fits.egressNode P) m) ⟨true, 0, j⟩ = y := by
        unfold gateAt; rw [h]; rfl
      exact Or.inr (mem_viewSet hmT hj hgy)
  · obtain ⟨t, ht, j, hj, hxg⟩ := of_mem_viewSet hxv
    have htm : t < ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).members := by rw [hmem]; exact ht
    have hp0 : 0 < (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).params.size := by
      rw [hp1]; exact Nat.one_pos
    obtain ⟨a, ha, h1, h2⟩ := CausalProof.param_lands P hroot hB hL hok hc hrp hz hfuel hk hcomp hbatch htm hp0 hj hq
    have hga : gateAt P (P.child P.rootScope (Fits.egressNode P) t) ⟨true, 0, j⟩ = a := by
      unfold gateAt; rw [ha]; rfl
    have hxa : x = a := hxg.symm.trans hga
    rw [hxa] at hgate
    obtain ⟨-, hqK, hin⟩ := CausalProof.inputArg_some hq
    have hnone := CausalProof.gate_input P hroot hL hqK hin h1 h2 (hinp q hqK hin) hgate
    rw [hnone] at hyo
    simp at hyo

end

end Properties.NetworkScan.ViewCutProof

namespace Properties.NetworkScan

open Flock.Program
open NciCheck.Properties.SecurityProofs
open NetworkCheck.Period

/-- **The scan's cut.** At an accepted scan, the free inputs other than the egress units' `state` gates are checked
inputs under the check that holds of every run. -/
theorem scanViewCut : ScanViewCut := by
  intro j program s sc c P C Pt d h hc hP hF
  have hA := Fits.scanOf_accepted h hc hP
  refine NetTiming.checkedUnique_of_reads C Pt d (scanLayout P s) s.window _ _
    (CausalProof.scanCausal j program s sc c P C h hc hP hA.topo hF) ?_
  intro t ht ℓ σ g hg i hdep hichk
  obtain ⟨hr, hrprim, -⟩ := BodiesOk.decode_root c.toUTF8 P hP
  obtain ⟨st, hPd, hinv, -⟩ := BodiesOk.decode_spec c.toUTF8 P hP
  obtain ⟨S, hS0, hS⟩ := BodiesOk.order_closed P hA.topo
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hPd]; exact hinv.1
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc hg => by
    obtain ⟨hSd, hpd⟩ := BodiesOk.good_closed P S hS0 hrprim (fun d hd hp => (hS d hd hp).2) sc hg
    exact BodiesOk.body_ok P hok sc.d (hS sc.d hSd hpd).1
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hr, hrprim⟩
  have hwk := hA.laid.2.2
  unfold NciCheck.Desc.walkable at hwk
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hwk
  obtain ⟨⟨hcf, hbl⟩, hres, hwb⟩ := hwk
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
  -- the egress template's root node: a batch of `T` members of a composite of one parameter, `state`
  obtain ⟨hke, hec⟩ := Fits.templateNodes_mem P hA.egress_mem
  have hp1 : (P.defs[((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).fn]!).params.size = 1 := by
    have h1 := congrArg Array.size hA.egress_params
    rw [Array.size_map] at h1
    have h2 : egressParams.size = 1 := rfl
    exact h1.trans h2
  -- its one argument is one input node's leaves
  obtain ⟨-, hnode, -⟩ := (hok P.root hr).2.1 (Fits.egressNode P) hke
  have h0 : 0 < ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).args.size := by
    have h1 : ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).args.size = 1 := hnode.1.trans hp1
    omega
  have hmem0 : ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).args[0]! ∈
      ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).args := by
    rw [getElem!_pos ((P.defs[P.root]!).body.nodes[Fits.egressNode P]!).args 0 h0]
    exact Array.getElem_mem h0
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 (hA.egress_args _ hmem0)
  -- an egress gate is a gate of that node, and everything it depends on is one, or a view gate
  obtain ⟨-, -, hg1, hg2⟩ := Fits.egress_gates hP hA hF t ht ℓ σ g hg
  have hiS : ViewCutProof.InV P s i := CausalProof.depends_closed C (ViewCutProof.InV P s)
    (ViewCutProof.view_closure P hroot hB hL hP hA.topo hbl hok hcf hpar hz hfuel C hF hke hec hA.egress_batch.1
      hA.egress_batch.2 hp1 hq hinp')
    hdep (Or.inl ⟨hg1, hg2⟩)
  obtain ⟨hfree, hnv⟩ := Finset.mem_sdiff.1 hichk
  rcases hiS with ⟨h1, h2⟩ | hv
  · obtain ⟨hil, hinput, -⟩ := Finset.mem_filter.1 hfree
    obtain ⟨p, ops, sc', hgate, hiff, -⟩ := hF.2.1 i (Finset.mem_range.1 hil)
    have hni := Fits.gate_not_input P hroot hL hA.laid.2.1 hke hec h1 h2 hgate
    exact Bool.false_ne_true (hni.symm.trans (hiff.1 hinput))
  · exact hnv hv

end Properties.NetworkScan
