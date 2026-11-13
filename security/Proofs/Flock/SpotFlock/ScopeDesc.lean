module

public import Proofs.Flock.SpotFlock.ScopeOf

/-!
# The two-stage scope from the registration's descriptor

`scopeOk_members` (`SpotFlock.ScopeOf`) reduces the two-stage scope at the members to conditions on each member, on the
replay layout and on the design. As stated, two of them read every gate: a member's size bound (`MemberFits`, its
program's gates) and the design's well-formedness on the members' replay partition (`filesReplay`, whose blocks are
the members' gates). Here each is a closed form in the descriptor, and the registration's side of `stmtOk` follows from
checks of the descriptor alone, one step per member, slot and unit:

* `length_memberProg`: a member's program has `memberSize c n = 2 + n·B + 3·(n·R)` gates, `B` and `R` its stacked unit
  rows' inputs and computed rows (`unitIns`, `unitRows`: `upv` copies of the unit net's, from its `inWords`, `constPos`
  and `upv`). The lowering makes an input gate one gate and a row three (`off_size`).
* `strataSizes_filesReplay`, `wellFormed_filesReplay_iff`: the members' replay partition has the replay layout's classes
  and strata, so a design is well formed on it exactly when it is on the layout's strata sizes (`designOk`).
* `fillOk`, `fillOk_iff`: the layout fills the members' files (`FilesFill`) exactly when each unit `x` of member `t` is
  in request `x / n_t`'s slot.
* `memberOk`, `memberFits_of_memberOk`: a member fits (`MemberFits`) where its unit net is built and its descriptor
  fits.
* `descOk`, `two_of_descOk`, `partition_two_of_descOk`: the registration's descriptor decided, and the registration's
  side of `stmtOk`, `(partition h s).isSome && (two h s).isSome`, from it and each member's unit net built (`Built`,
  which the unrolled circuit of a statement `Flock.Desc.parseStmt` parsed has, flat or templated:
  `FlockSoundness.Discharge.Hidden.built_of_parseStmt`; a flat file's, `FlockC3.VStarList.built_of_parse`), where
  every template's reference statement reads as a member's (`templateOk`, a hypothesis here: `descOk` doesn't decide
  it).
-/

@[expose] public section

namespace Flock.Bridge.Members

open Claims.Sampling Properties.Spot

/-! ## A member's program's size -/

theorem gateSize_of_lt {kp : Spot.Bits.KeyProg} {g : ℕ} (hg : g < kp.inputs) : Spot.Bits.gateSize kp g = 1 := by
  simp [Spot.Bits.gateSize, Spot.Bits.KeyProg.row?, hg]

theorem gateSize_of_row {kp : Spot.Bits.KeyProg} {j : ℕ} (hj : j < kp.rows.size) :
    Spot.Bits.gateSize kp (kp.inputs + j) = 3 := by
  simp [Spot.Bits.gateSize, Spot.Bits.KeyProg.row?, hj]

theorem sum_gateSize_inputs (kp : Spot.Bits.KeyProg) :
    ∀ m ≤ kp.inputs, ((List.range m).map (Spot.Bits.gateSize kp)).sum = m
  | 0, _ => rfl
  | m + 1, hm => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_gateSize_inputs kp m (by omega), List.map_cons,
      List.map_nil, List.sum_cons, List.sum_nil, gateSize_of_lt (by omega)]
    omega

theorem sum_gateSize (kp : Spot.Bits.KeyProg) :
    ∀ j ≤ kp.rows.size, ((List.range (kp.inputs + j)).map (Spot.Bits.gateSize kp)).sum = kp.inputs + 3 * j
  | 0, _ => sum_gateSize_inputs kp kp.inputs le_rfl
  | j + 1, hj => by
    rw [← Nat.add_assoc, List.range_succ, List.map_append, List.sum_append, sum_gateSize kp j (by omega),
      List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, gateSize_of_row (by omega)]
    omega

/-- **A key program's lowered size**: one gate an input gate, three a row. -/
theorem off_size (kp : Spot.Bits.KeyProg) : Spot.Bits.off kp kp.size = kp.inputs + 3 * kp.rows.size := by
  rw [Properties.Spot.SecurityProofs.off_of_le kp le_rfl]
  exact sum_gateSize kp kp.rows.size le_rfl

/-- **A member's stacked unit rows' inputs**, `B` of its key program: `upv` copies of the unit net's input columns
(`Spot.Bits.Rows.ofNet`). -/
def unitIns (c : _root_.Flock.Circuit) : ℕ := c.upv * min (c.unit.inWords * _root_.Flock.WORD) c.unit.constPos

/-- **A member's stacked unit rows' computed rows**, `R` of its key program: `upv` copies of the unit net's computed
rows, each copy with its constant row. -/
def unitRows (c : _root_.Flock.Circuit) : ℕ :=
  c.upv * (c.unit.constPos - min (c.unit.inWords * _root_.Flock.WORD) c.unit.constPos + 1)

/-- **A member's lowered size**, from its descriptor: the constant, the zero and `B` source gates a unit, one gate each,
then `R` rows a unit, three gates each. -/
def memberSize (c : _root_.Flock.Circuit) (n : ℕ) : ℕ := 2 + n * unitIns c + 3 * (n * unitRows c)

theorem nIn_bitsRows (c : _root_.Flock.Circuit) : (bitsRows c).nIn = unitIns c := rfl

theorem nComp_bitsRows (c : _root_.Flock.Circuit) : (bitsRows c).nComp = unitRows c := rfl

theorem keyProg_rows_size (c : _root_.Flock.Circuit) (n : ℕ) :
    (Spot.Bits.keyProg c n).rows.size = n * (bitsRows c).nComp := by
  have h1 : (Spot.Bits.keyProg c n).inputs + (Spot.Bits.keyProg c n).rows.size =
      2 + n * (bitsRows c).nIn + n * (bitsRows c).nComp := Flock.Bridge.keyProg_size c n
  have h2 := Flock.Bridge.keyProg_inputs c n
  omega

/-- **A member's program's size** is its descriptor's (`memberSize`). -/
theorem length_memberProg (c : _root_.Flock.Circuit) (n : ℕ) : (memberProg c n).gates.length = memberSize c n := by
  have e : (memberProg c n).gates.length = Spot.Bits.off (Spot.Bits.keyProg c n) (Spot.Bits.keyProg c n).size :=
    Properties.Spot.SecurityProofs.bitsProg_length _ _
  rw [e, off_size, Flock.Bridge.keyProg_inputs, keyProg_rows_size, nIn_bitsRows, nComp_bitsRows]
  rfl

/-! ## The replay partition's strata -/

variable (ms : Spot.Bits.Members) (ns : Array ℕ) (l : Spot.Slots.SlotOrder)

theorem classes_filesReplay : (filesReplay ms ns l).classes =
    l.classes.toList.map fun c => c.toList.map fun s => s.toList.map (filesBlock ms ns) := rfl

/-- **The members' replay partition's strata are the replay layout's**, in slots. -/
theorem strataSizes_filesReplay :
    (filesReplay ms ns l).classes.map strataSizes = l.classes.toList.map fun c => c.toList.map (·.size) := by
  simp [classes_filesReplay, strataSizes, Function.comp_def]

end Flock.Bridge.Members

namespace SpotFlock

open Claims Claims.Sampling Properties.Spot Flock.Bridge.Members

/-! ## The descriptor's checks -/

/-- **A member's descriptor fits the two-stage scope**, decided from the descriptor alone: `unitLog ≤ 32`, an instance a
unit at least (`0 < upv`), and at most `unitClassMax` gates in its program (`memberSize`). -/
def memberOk (ci : _root_.Flock.Circuit × Array ℕ) : Bool :=
  decide (ci.1.unit.unitLog ≤ 32) && decide (0 < ci.1.upv) && decide (memberSize ci.1 ci.2.size ≤ unitClassMax)

theorem memberFits_of_memberOk {ci : _root_.Flock.Circuit × Array ℕ} (hU : FlockSoundness.Built ci.1.unit)
    (h : memberOk ci = true) : MemberFits ci := by
  simp only [memberOk, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hu, hv⟩, hs⟩ := h
  refine ⟨hU, hu, ?_, by rw [length_memberProg]; exact hs⟩
  rw [nComp_bitsRows]
  exact Nat.mul_pos hv (Nat.succ_pos _)

/-- **The replay layout fills the members' files**, decided: unit `x` of member `t`, of `n_t` units a request, is in
request `x / n_t`'s run, whose slot is that request's unit. One check per unit, none per gate. -/
def fillOk (ms : Spot.Bits.Members) (ns : Array ℕ) (l : Spot.Slots.SlotOrder) : Bool :=
  (List.range ms.length).all fun t => (List.range (((ms[t]?).map (·.2.size)).getD 0)).all fun x =>
    decide (0 < ns.getD t 0) && (l.blocks[x / ns.getD t 0]?).any fun b => b.unit == some (x / ns.getD t 0)

section fill

variable {ms : Spot.Bits.Members} {ns : Array ℕ} {l : Spot.Slots.SlotOrder}

theorem filesFill_of_fillOk (h : fillOk ms ns l = true) : FilesFill ms ns l := by
  intro t ht x hx
  have h1 := List.all_eq_true.1 h t (List.mem_range.2 ht)
  rw [List.getElem?_eq_getElem ht, Option.map_some, Option.getD_some] at h1
  have h2 := List.all_eq_true.1 h1 x (List.mem_range.2 hx)
  rw [Bool.and_eq_true, decide_eq_true_eq] at h2
  obtain ⟨hn, hb⟩ := h2
  rcases e : l.blocks[x / ns.getD t 0]? with _ | b
  · rw [e] at hb
    exact absurd hb (by simp)
  · rw [e, Option.any_some, beq_iff_eq] at hb
    refine ⟨x / ns.getD t 0, b, e, hb, Nat.div_mul_le_self x _, ?_⟩
    rw [Nat.add_one_mul]
    exact Nat.lt_div_mul_add hn

theorem fillOk_of_filesFill (h : FilesFill ms ns l) : fillOk ms ns l = true := by
  refine List.all_eq_true.2 fun t ht => ?_
  have ht' := List.mem_range.1 ht
  rw [List.getElem?_eq_getElem ht', Option.map_some, Option.getD_some]
  refine List.all_eq_true.2 fun x hx => ?_
  obtain ⟨ρ, b, hb, hu, h1, h2⟩ := h t ht' x (List.mem_range.1 hx)
  have hn : 0 < ns.getD t 0 := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at h2; omega
  have e : x / ns.getD t 0 = ρ := Nat.div_eq_of_lt_le h1 h2
  rw [Bool.and_eq_true, decide_eq_true_eq, e, hb, Option.any_some, beq_iff_eq]
  exact ⟨hn, hu⟩

/-- **`fillOk` decides `FilesFill`.** -/
theorem fillOk_iff : fillOk ms ns l = true ↔ FilesFill ms ns l := ⟨filesFill_of_fillOk, fillOk_of_filesFill⟩

end fill

/-- **The first stage's design on the replay layout's strata**, decided: one rule per class of `l`, each well formed
over its class's strata sizes in slots (`oneStageOk`). -/
def designOk (d : Design) (l : Spot.Slots.SlotOrder) : Bool :=
  d.rules.length == l.classes.size &&
    (d.rules.zip (l.classes.toList.map fun c => c.toList.map (·.size))).all fun p => oneStageOk p.2 p.1

/-- **A design is well formed on the members' replay partition** exactly when it is on the replay layout's strata. -/
theorem wellFormed_filesReplay_iff {ms : Spot.Bits.Members} {ns : Array ℕ} {l : Spot.Slots.SlotOrder} {d : Design} :
    d.WellFormed (filesReplay ms ns l) ↔ designOk d l = true := by
  show d.rules.length = (filesReplay ms ns l).classes.length ∧
      (∀ p ∈ d.rules.zip (filesReplay ms ns l).classes, p.1.WellFormed (strataSizes p.2)) ↔ _
  unfold designOk
  rw [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, ← strataSizes_filesReplay ms ns l, List.zip_map_right,
    List.forall_mem_map]
  have hlen : (filesReplay ms ns l).classes.length = l.classes.size := by simp [classes_filesReplay]
  rw [hlen]
  exact and_congr_right fun _ => forall₂_congr fun p _ => ⟨oneStageOk_complete, oneStageOk_sound⟩

/-- `descOk`'s checks at the members `ms`, the replay layout `l` and the first stage's design `d`. -/
def descAt (h : Held) (ms : Spot.Bits.Members) (l : Spot.Slots.SlotOrder) (d : Design) : Bool :=
  ms.all memberOk && filesOk h.reg ms l h.k && fillOk ms h.reg.ns l && designOk d l && h.k.length == ms.length &&
    h.k.all (1 ≤ ·)

/-- **The registration's descriptor is in the two-stage scope**, decided over the descriptor alone: the replay layout and
the first stage's design evaluate, every member fits (`memberOk`), the layout fits and fills the members' files
(`filesOk`, `fillOk`), the design is well formed on the layout's strata (`designOk`), and the second stage counts at
least `1` for each member. Its steps are per member, slot and unit; it builds no member's program. -/
def descOk (h : Held) : Bool :=
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some d => descAt h h.ms l d
  | _, _ => false

/-! ## The registration's side of `stmtOk` -/

/-- **`two` returns the members' replay design** where the descriptor's checks pass at the replay layout and the first
stage's design, each member's unit net is built, and every template's reference statement reads as a member's
(`templateOk`). -/
theorem two_of_descAt (h : Held) (s : AuditStatement) {l : Spot.Slots.SlotOrder} {d : Design}
    (hl : (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption = some l) (hd : replayDesign h = some d)
    (hb : ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit) (hok : descAt h h.ms l d = true)
    (ht : ∀ t < h.ms.length, templateOk h t = true) : two h s = some ⟨filesReplay h.ms h.reg.ns l, d, h.k⟩ := by
  simp only [descAt, Bool.and_eq_true, List.all_eq_true, beq_iff_eq, decide_eq_true_eq] at hok
  obtain ⟨⟨⟨⟨⟨hm, hf⟩, hfill⟩, hdes⟩, hk⟩, hk1⟩ := hok
  exact two_eq_filesReplay h s hl hd (fun ci hci => memberFits_of_memberOk (hb ci hci) (hm ci hci))
    (filesFill_of_fillOk hfill) (wellFormed_filesReplay_iff.2 hdes) hk hk1 hf ht

theorem two_of_descOk (h : Held) (s : AuditStatement) (hb : ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit)
    (hok : descOk h = true) (ht : ∀ t < h.ms.length, templateOk h t = true) : (two h s).isSome := by
  unfold descOk at hok
  rcases hl : (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption with _ | l
  · rw [hl] at hok
    simp at hok
  rcases hd : replayDesign h with _ | d
  · rw [hl, hd] at hok
    simp at hok
  rw [hl, hd] at hok
  simp [two_of_descAt h s hl hd hb hok ht]

/-- **The registration's side of `stmtOk`**, `(partition h s).isSome && (two h s).isSome`, where the registration's
descriptor is in the two-stage scope (`descOk`), each member's unit net is built, and every template's reference
statement reads as a member's (`templateOk`). -/
theorem partition_two_of_descOk (h : Held) (s : AuditStatement) (hb : ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit)
    (hok : descOk h = true) (ht : ∀ t < h.ms.length, templateOk h t = true) :
    ((partition h s).isSome && (two h s).isSome) = true := by
  rw [two_of_descOk h s hb hok ht]
  rfl

end SpotFlock
