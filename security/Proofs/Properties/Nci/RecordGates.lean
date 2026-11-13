module

public import Proofs.Properties.Nci.DecodeKeys
public import Proofs.Properties.Nci.RecordTakes
public import Proofs.Properties.Nci.RecordUnits
public import Proofs.Properties.Nci.OperandsResolve
public import Proofs.Properties.Nci.ResolveWithin
public import Mathlib.Tactic

/-!
# A unit's gates

A unit is a member of a root node that is no input, and its gates are that member's place in the node's.

`ReadProg`: what the record check reads of a Program before it reads a unit of it; its last clause, that the root is
the first Definition of its key, the decoder gives (`readProg_of_decode`). The template query passes over the input
nodes of such a Program (`readProg_inputs`), so its ranges are the root's nodes whose callee's key is no input id.
`unit_node` gives a unit its root node and member, from the root's ranges (`ranges_spec`), and `unit_within` puts
the member's gates inside the node's. From those:

* `unitsComputed` (`UnitsComputed`): a gate inside a unit is computed, since its root node is no input node.
* `gate_located`: every gate of the Program is located, holds one value, and reads nothing where it is an input.
* `gatesOwn` (`GatesOwn`): with that, a computed gate lies in a unit, the member of its root node that holds it.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope isInputId)
open Flock.TemplateQuery (TRange)
open NciCheck.Record.SecurityProofs (RootFirst InputsRead ofProg_def ranges_spec evaluate_asked)
open NciCheck.Properties.SecurityProofs

/-- **A Program the record check reads**: it decodes, in order, its root and its bodies laid out and its input
nodes the root's, no root node calls a composite keyed as an input, and its root is the first Definition of its
key. -/
def ReadProg (P : Prog) : Prop :=
  (∃ bytes, Flock.Program.decode bytes = .ok P) ∧ P.orderViolation = none ∧ Desc.rootLaidOut P = true ∧
    Desc.inputsAtRoot P = true ∧ Desc.walkable P = true ∧ NciCheck.RecordText.keyedAsInput P = false ∧ RootFirst P

/-- **A decoded Program that passes the check's reads is one the record check reads**: the decoder puts no
Definition of the root's key before the root (`DecodeKeys.root_first`). -/
theorem readProg_of_decode {bytes : ByteArray} {P : Prog} (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hl : Desc.rootLaidOut P = true) (hin : Desc.inputsAtRoot P = true)
    (hwk : Desc.walkable P = true) (hkey : NciCheck.RecordText.keyedAsInput P = false) : ReadProg P :=
  ⟨⟨bytes, hdec⟩, hov, hl, hin, hwk, hkey, DecodeKeys.root_first bytes P hdec⟩

/-- **The template query passes over the input nodes of a Program the record check reads**: a root node's callee is
a Definition before the root and the first of its key (`DecodeKeys.decode_keys`), and where its key is an input id
it is a primitive, since no root node calls a composite keyed as an input. -/
theorem readProg_inputs {P : Prog} (hP : ReadProg P) : InputsRead P := by
  obtain ⟨⟨bytes, hdec⟩, -, -, -, -, hkey, -⟩ := hP
  obtain ⟨hr, -, -⟩ := BodiesOk.decode_root bytes P hdec
  obtain ⟨st, hPd, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hPd]; exact hinv.1
  obtain ⟨-, huniq⟩ := DecodeKeys.decode_keys bytes P hdec
  intro k hk hI
  have hfn : ((P.defs[P.root]!).body.nodes[k]!).fn < P.root := ((hok P.root hr).2.1 k hk).1
  have hlt : ((P.defs[P.root]!).body.nodes[k]!).fn < P.defs.size := Nat.lt_trans hfn hr
  have hprim : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = true := by
    have e : (P.defs[P.root]!).body.nodes[k]! = (P.defs[P.root]!).body.nodes[k] := getElem!_pos _ k hk
    have hany := Array.any_eq_false.1 hkey k hk
    rw [e] at hI ⊢
    cases hp : (P.defs[((P.defs[P.root]!).body.nodes[k]).fn]!).prim with
    | true => rfl
    | false =>
      exfalso
      simp [hp, hI] at hany
  have hfst : ∀ j < ((P.defs[P.root]!).body.nodes[k]!).fn,
      (P.defs[j]!).key ≠ (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key := by
    intro j hj he
    have h1 := huniq j (Nat.lt_trans hj hlt)
    rw [he, huniq _ hlt] at h1
    exact absurd (Option.some.inj h1) (Nat.ne_of_gt hj)
  rw [ofProg_def hlt hfst]
  exact hprim

/-- **A unit is a member of a root node that is no input**: the node's callee has the unit's template for its key,
which is no input id. -/
theorem unit_node {P : Prog} {ts : Array String} {rs : Array TRange} (hr : P.root < P.defs.size) (hk : RootFirst P)
    (hin : InputsRead P) (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs)
    {tid : String} {u : ℕ} {sc : Scope} (h : UnitAt P rs tid u sc) :
    ∃ k m, k < (P.defs[P.root]!).body.nodes.size ∧
      isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key = false ∧
      tid = (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key ∧
      m < ((P.defs[P.root]!).body.nodes[k]!).members ∧ sc = P.child P.rootScope k m := by
  obtain ⟨hnodes, hall⟩ := ranges_spec hr hk hin hrs
  obtain ⟨i, m, hi, hti, hm, -, hsc⟩ := h
  have hmem : rs[i]! ∈ rs.toList := by
    rw [getElem!_pos rs i hi]
    simp
  obtain ⟨hlt, htmpl, -, hcount⟩ := hall _ hmem
  have hno : rs[i]!.node ∈ rs.toList.map (·.node) := List.mem_map.2 ⟨_, hmem, rfl⟩
  rw [hnodes, List.mem_filter] at hno
  refine ⟨rs[i]!.node, m, hlt, ?_, ?_, ?_, hsc⟩
  · simpa using hno.2
  · rw [← hti, htmpl]
  · rw [← hcount]
    exact hm

/-- A member of a root node: its Definition is the node's callee, and its gates start at the node's offset, after
the members before it. -/
theorem child_root (P : Prog) (k m : ℕ) :
    (P.child P.rootScope k m).d = ((P.defs[P.root]!).body.nodes[k]!).fn ∧
      (P.child P.rootScope k m).off = (P.defs[P.root]!).body.offsets[k]! +
        m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates := by
  refine ⟨rfl, ?_⟩
  show 0 + (P.defs[P.root]!).body.offsets[k]! + m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates = _
  omega

/-- **A member's gates are inside its root node's.** -/
theorem unit_within (P : Prog) (hL : Walk.LaidOut P P.root) (hp : (P.defs[P.root]!).prim = false) {k m : ℕ}
    (hk : k < (P.defs[P.root]!).body.nodes.size) (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members) :
    (P.defs[P.root]!).body.offsets[k]! ≤ (P.child P.rootScope k m).off ∧
      (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates ≤
        (P.defs[P.root]!).body.offsets[k + 1]! := by
  obtain ⟨-, -, hnodes⟩ := hL hp
  obtain ⟨hd, hoff⟩ := child_root P k m
  have hstep := hnodes k hk
  have hle : (m + 1) * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates ≤
      ((P.defs[P.root]!).body.nodes[k]!).members * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates :=
    Nat.mul_le_mul_right _ hm
  rw [Nat.add_mul, Nat.one_mul] at hle
  rw [hoff, hd, hstep]
  constructor <;> omega

/-- What `Desc.walkable` gives of a decoded Program in order: its root a composite Definition of its own, every body
laid out, and every reference of a body that resolution reaches in range. -/
theorem read_bodies {P : Prog} (hP : ReadProg P) :
    P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false ∧ (∀ d < P.defs.size, Walk.LaidOut P d) ∧
      ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := by
  obtain ⟨⟨bytes, hdec⟩, hov, -, -, hwk, -, -⟩ := hP
  obtain ⟨hr, hrp, -⟩ := BodiesOk.decode_root bytes P hdec
  unfold Desc.walkable at hwk
  simp only [Bool.and_eq_true] at hwk
  obtain ⟨-, hlay, -, -⟩ := Walk.bodiesLaidOut_spec P hwk.1.2
  obtain ⟨st, hPd, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := BodiesOk.order_closed P hov
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hPd]; exact hinv.1
  refine ⟨hr, hrp, hlay, fun sc hg => ?_⟩
  obtain ⟨hSd, hpd⟩ := BodiesOk.good_closed P S hS0 hrp (fun d hd hp => (hS d hd hp).2) sc hg
  exact BodiesOk.body_ok P hok sc.d (hS sc.d hSd hpd).1

/-- **A gate inside a unit is computed**: it is a gate of a root node that is no input node. -/
theorem unitsComputed {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) : UnitsComputed P rs := by
  intro tid u sc g p ops sc' hu hlo hhi hgate
  obtain ⟨hr, hrp, hlay, -⟩ := read_bodies hP
  have hir := readProg_inputs hP
  obtain ⟨-, -, hl, hin, -, -, hfirst⟩ := hP
  obtain ⟨k, m, hk, hno, -, hm, rfl⟩ := unit_node hr hfirst hir hrs hu
  obtain ⟨h1, h2⟩ := unit_within P (hlay P.root hr) hrp hk hm
  have hg : (DescAgree.rootOffs P)[k]! ≤ g ∧ g < (DescAgree.rootOffs P)[k + 1]! :=
    ⟨Nat.le_trans h1 hlo, Nat.lt_of_lt_of_le hhi h2⟩
  have hkn : k < DescAgree.nodeCount P := hk
  rw [DescAgree.gate_input P hl hin hgate hkn hg, DescAgree.inputs_get P hkn]
  show Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false
  simp [Desc.isInputNode, hno]

/-- **Every gate of the Program is located**: it has a primitive that returns one value, and an input's gate reads
nothing. -/
theorem gate_located {P : Prog} (hP : ReadProg P) {g : ℕ} (hg : g < gates P) :
    ∃ p ops sc w, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value w ∧
      (isInputId (P.defs[p]!).key = true → ops = #[]) := by
  obtain ⟨hr, hrp, hlay, hB⟩ := read_bodies hP
  obtain ⟨-, -, -, -, hwk, -, -⟩ := hP
  have hwk' := hwk
  unfold Desc.walkable at hwk'
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hwk'
  obtain ⟨⟨hc, hbl⟩, hres, hwb⟩ := hwk'
  obtain ⟨hpar, hz⟩ := OperandsResolve.resolvable_spec P hres
  have hfuel : 4 * (Desc.activations P)[P.root]! ≤ Flock.Steps.FUEL := by
    unfold Desc.walkBound at hwb
    have : ((Desc.activations P)[P.root]! + 1) * 2 ≤
        ((Desc.activations P)[P.root]! + 1) * ((Desc.depths P)[P.root]! + 2) := Nat.mul_le_mul_left _ (by omega)
    omega
  have hg' : g < (P.defs[P.root]!).gates := hg
  obtain ⟨⟨p, ops, sc⟩, hx⟩ := Walk.gate_ok_of P hc hlay hr hrp (Walk.depth_lt_fuel P hwk) hg'
    (fun sc hR _ k hk hpk m hm e => by
      obtain ⟨ops, hops⟩ := OperandsResolve.operands_prim P ⟨hr, hrp⟩ hB hlay hc hpar hz hfuel sc
        (OperandsResolve.reach_walkQ P ⟨hr, hrp⟩ hB hlay hg' hR).1 k hk hpk m hm
      rw [hops]
      exact fun h => by cases h)
  obtain ⟨⟨w, hw⟩, hinp⟩ := Walk.gate_reads_of P hbl hx
  exact ⟨p, ops, sc, w, hx, hw, hinp⟩

/-- **Every gate has its place**: it is located and holds one value; an input gate reads nothing; and a computed
gate lies in the unit that is the member of its root node holding it. -/
theorem gatesOwn {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) : GatesOwn P rs := by
  intro g hg
  obtain ⟨p, ops, sc, w, hgate, hw, hinp⟩ := gate_located hP hg
  refine ⟨p, ops, sc, w, hgate, hw, hinp, fun hcomp => ?_⟩
  obtain ⟨hr, hrp, hlay, -⟩ := read_bodies hP
  have hir := readProg_inputs hP
  obtain ⟨-, -, hl, hin, -, hkey, hfirst⟩ := hP
  obtain ⟨-, h0, hK, -⟩ := DescAgree.rootLaidOut_spec P hl
  obtain ⟨k, hkn, hgk⟩ := DescAgree.node_exists (DescAgree.rootOffs P) (DescAgree.nodeCount P) h0 g
    (by rw [hK]; exact hg)
  have hk : k < (P.defs[P.root]!).body.nodes.size := hkn
  have hnode : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false := by
    have h := DescAgree.gate_input P hl hin hgate hkn hgk
    rw [DescAgree.inputs_get P hkn, hcomp] at h
    exact h.symm
  have hno : isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key = false := by
    have e : (P.defs[P.root]!).body.nodes[k]! = (P.defs[P.root]!).body.nodes[k] := getElem!_pos _ k hk
    have hany := Array.any_eq_false.1 hkey k hk
    rw [e] at hnode ⊢
    cases hI : isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]).fn]!).key with
    | false => rfl
    | true =>
      exfalso
      have hprim : (P.defs[((P.defs[P.root]!).body.nodes[k]).fn]!).prim = false := by
        simpa [Desc.isInputNode, hI] using hnode
      simp [hprim, hI] at hany
  obtain ⟨hnodes, hall⟩ := ranges_spec hr hfirst hir hrs
  have hkin : k ∈ rs.toList.map (·.node) := by
    rw [hnodes, List.mem_filter]
    exact ⟨List.mem_range.2 hk, by simpa using hno⟩
  obtain ⟨r, hrmem, hrk⟩ := List.mem_map.1 hkin
  have hrmem' : r ∈ rs := by simpa using hrmem
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hrmem'
  have e : rs[i]! = rs[i] := getElem!_pos rs i hi
  have hik : rs[i]!.node = k := by rw [e]; exact hrk
  obtain ⟨-, -, -, hcount⟩ := hall rs[i]! (by rw [e]; exact hrmem)
  rw [hik] at hcount
  obtain ⟨-, -, hstep⟩ := hlay P.root hr hrp
  have hstep := hstep k hk
  have hgk' : (P.defs[P.root]!).body.offsets[k]! ≤ g ∧ g < (P.defs[P.root]!).body.offsets[k + 1]! := hgk
  obtain ⟨G, hG⟩ : ∃ G, G = (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates := ⟨_, rfl⟩
  obtain ⟨M, hM⟩ : ∃ M, M = ((P.defs[P.root]!).body.nodes[k]!).members := ⟨_, rfl⟩
  obtain ⟨o, ho⟩ : ∃ o, o = (P.defs[P.root]!).body.offsets[k]! := ⟨_, rfl⟩
  rw [← hG, ← hM, ← ho] at hstep
  rw [← ho, hstep] at hgk'
  rw [← hM] at hcount
  have hxlt : g - o < G * M := by rw [Nat.mul_comm]; omega
  have hGpos : 0 < G := by
    rcases Nat.eq_zero_or_pos G with h | h
    · rw [h, Nat.zero_mul] at hxlt
      omega
    · exact h
  have hdiv : (g - o) / G < M := Nat.div_lt_of_lt_mul hxlt
  have hdm := Nat.div_add_mod (g - o) G
  have hml := Nat.mod_lt (g - o) hGpos
  have hcomm : (g - o) / G * G = G * ((g - o) / G) := Nat.mul_comm _ _
  obtain ⟨hd, hoff⟩ := child_root P k ((g - o) / G)
  refine ⟨rs[i]!.template,
    ((rs.extract 0 i).filter fun r => r.template == rs[i]!.template).foldl (fun a r => a + r.count) 0 +
      (g - o) / G,
    P.child P.rootScope k ((g - o) / G), ⟨i, (g - o) / G, hi, rfl, ?_, rfl, by rw [hik]⟩, ?_, ?_⟩
  · rw [hcount]
    exact hdiv
  · rw [hoff, ← ho, ← hG]
    omega
  · rw [hoff, hd, ← ho, ← hG]
    omega

/-! ## Units apart -/

/-- A range of the root: its node is one of the root's, and its count the node's members. -/
theorem range_node {P : Prog} {ts : Array String} {rs : Array TRange} (hr : P.root < P.defs.size)
    (hk : RootFirst P) (hin : InputsRead P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {i : ℕ} (hi : i < rs.size) :
    rs[i]!.node < (P.defs[P.root]!).body.nodes.size ∧
      rs[i]!.count = ((P.defs[P.root]!).body.nodes[rs[i]!.node]!).members := by
  obtain ⟨-, hall⟩ := ranges_spec hr hk hin hrs
  have hmem : rs[i]! ∈ rs.toList := by
    rw [getElem!_pos rs i hi]
    simp
  obtain ⟨hlt, -, -, hcount⟩ := hall _ hmem
  exact ⟨hlt, hcount⟩

/-- The root's ranges are in the order of their nodes. -/
theorem ranges_sorted {P : Prog} {ts : Array String} {rs : Array TRange} (hr : P.root < P.defs.size)
    (hk : RootFirst P) (hin : InputsRead P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs)
    {i i' : ℕ} (hi : i < rs.size) (hi' : i' < rs.size) (h : i < i') : rs[i]!.node < rs[i']!.node := by
  obtain ⟨hnodes, -⟩ := ranges_spec hr hk hin hrs
  have hs : (rs.toList.map (·.node)).Pairwise (· < ·) := by
    rw [hnodes]
    exact List.Pairwise.filter _ List.pairwise_lt_range
  rw [List.pairwise_iff_getElem] at hs
  have := hs i i' (by simpa using hi) (by simpa using hi') h
  rw [getElem!_pos rs i hi, getElem!_pos rs i' hi']
  simpa using this

/-- Two members of one root node hold no gate in common. -/
theorem members_apart (P : Prog) (k : ℕ) {m m' : ℕ} (h : m < m') :
    (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates ≤
      (P.child P.rootScope k m').off := by
  obtain ⟨hd, hoff⟩ := child_root P k m
  obtain ⟨-, hoff'⟩ := child_root P k m'
  have hle : (m + 1) * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates ≤
      m' * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates := Nat.mul_le_mul_right _ h
  rw [Nat.add_mul, Nat.one_mul] at hle
  rw [hd, hoff, hoff']
  omega

/-- **Units are apart**: two members of the root's ranges, not one member of one range, hold no gate in common. -/
theorem units_apart {P : Prog} {ts : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {i i' m m' : ℕ}
    (hi : i < rs.size) (hi' : i' < rs.size) (hm : m < rs[i]!.count) (hm' : m' < rs[i']!.count)
    (hne : i ≠ i' ∨ m ≠ m') :
    (P.child P.rootScope rs[i]!.node m).off + (P.defs[(P.child P.rootScope rs[i]!.node m).d]!).gates ≤
        (P.child P.rootScope rs[i']!.node m').off ∨
      (P.child P.rootScope rs[i']!.node m').off + (P.defs[(P.child P.rootScope rs[i']!.node m').d]!).gates ≤
        (P.child P.rootScope rs[i]!.node m).off := by
  obtain ⟨hr, hrp, hlay, -⟩ := read_bodies hP
  have hfirst : RootFirst P := hP.2.2.2.2.2.2
  have hir := readProg_inputs hP
  obtain ⟨hk, hc⟩ := range_node hr hfirst hir hrs hi
  obtain ⟨hk', hc'⟩ := range_node hr hfirst hir hrs hi'
  rw [hc] at hm
  rw [hc'] at hm'
  have hw := unit_within P (hlay P.root hr) hrp hk hm
  have hw' := unit_within P (hlay P.root hr) hrp hk' hm'
  rcases Nat.lt_trichotomy i i' with h | h | h
  · have hn := ranges_sorted hr hfirst hir hrs hi hi' h
    have hmono := Resolve.offsets_mono P P.root (hlay P.root hr) hrp (rs[i]!.node + 1) rs[i']!.node hn
      (Nat.le_of_lt hk')
    exact Or.inl (Nat.le_trans hw.2 (Nat.le_trans hmono hw'.1))
  · subst h
    have hmm : m ≠ m' := by
      rcases hne with h | h
      · exact absurd rfl h
      · exact h
    rcases Nat.lt_or_gt_of_ne hmm with h | h
    · exact Or.inl (members_apart P _ h)
    · exact Or.inr (members_apart P _ h)
  · have hn := ranges_sorted hr hfirst hir hrs hi' hi h
    have hmono := Resolve.offsets_mono P P.root (hlay P.root hr) hrp (rs[i']!.node + 1) rs[i]!.node hn
      (Nat.le_of_lt hk)
    exact Or.inr (Nat.le_trans hw'.2 (Nat.le_trans hmono hw.1))

/-! ## A unit's template is the wiring's -/

/-- **Every unit's template is one the root's ranges were asked for.** -/
theorem unit_asked {P : Prog} {ts : Array String} {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (h : UnitAt P rs tid u sc) : tid ∈ ts := by
  obtain ⟨i, m, hi, hti, -, -, -⟩ := h
  have hmem : rs[i]! ∈ rs.toList := by
    rw [getElem!_pos rs i hi]
    simp
  have hc := evaluate_asked hrs _ hmem
  rw [hti] at hc
  simpa using hc

/-- A unit has a template of the wiring, and is a unit of its Definition, where the wiring's templates are the
Program's and the root's ranges were asked for them. -/
theorem unit_template {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P)
      (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) {tid : String} {u : ℕ} {sc : Scope} (hu : UnitAt P rs tid u sc) :
    ∃ t, (tid, t) ∈ W.templates ∧ TemplateOf (P.defs[sc.d]!) t := by
  have hmem : tid ∈ W.templates.map (·.1) := by simpa using unit_asked hrs hu
  obtain ⟨⟨tid', t⟩, ht, rfl⟩ := List.mem_map.1 hmem
  exact ⟨t, ht, (hT tid' t ht).2 u sc hu⟩

/-- **A unit's parameters have their own names** (`PortsNamed`): a unit's Definition is a template's of the
wiring, whose parameters' names are apart. -/
theorem portsNamed {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P)
      (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) : PortsNamed P rs := by
  intro tid u sc i i' hu hi hi' hname
  obtain ⟨t, -, htmpl⟩ := unit_template hrs hT hu
  exact htmpl.1 i i' hi hi' hname

end NciCheck.Properties.Record.SecurityProofs
