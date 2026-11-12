module

public import Proofs.Flock.SpotFlock.Honest.Roots
public import Proofs.Flock.SpotFlock.Reads
public import Proofs.Flock.Verifier.Registered

/-!
# `EntryFact` at the one-template instance

At every gate an accepted run's checked unit needs, a registered read is named like the gate's port
(`rootsHold_held`'s reading: `bit_held`, `regNamed_name`), so only `EntryHolds`' registered arm is ever met, and there:

* the port is the session's first of the read's name: the member's first port of the gate's row (`PortsMember`);
* the slot names the entry round 2 reads the value off: both are the boundary port the gate's link names, since the
  interior registration registers no value (`reads_held`), so `readKey` takes the link's port;
* the slot's schema is the read's: the slot has the member's port's row schema (`PortRoots.schemaOf`), and Flock's
  setup checks every registered read against the first port of its name (`Registered.checkPort`, `schema_of_setup`).

The hypotheses are `rootsHold_held`'s: `RefPorts R`, `NeededBits` and `PortsMember`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

/-- **`checkPort` accepting holds the read to the first port of its name's schema.** -/
theorem checkPort_schema {c pub t indices reg} (h : _root_.Flock.Registered.checkPort c pub t indices reg = .ok ()) :
    ∃ p, (List.range c.ports.size).find? (fun p => c.ports[p]!.name == reg.name) = some p ∧
      _root_.Flock.Registered.portSchema c.ports[p]! = reg.schema := by
  unfold _root_.Flock.Registered.checkPort at h
  simp only at h
  split at h
  · rename_i p hp
    split at h
    · exact absurd (_root_.Flock.Registered.except_bind_ok h) fun ⟨_, h1, _⟩ =>
        _root_.Flock.Registered.throw_ne_ok _ _ h1
    · rename_i hsch
      exact ⟨p, hp, by simpa using hsch⟩
  · exact absurd h (_root_.Flock.Registered.throw_ne_ok _ _)

/-- **Each registered read of a session is of the schema of the first port of its name**, in every statement of the
session's setup. -/
theorem schema_of_setup {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {own} (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hown : ins.registered = some own) :
    ∀ st ∈ sts, ∀ reg ∈ own, ∃ p,
      (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p ∧
        _root_.Flock.Registered.portSchema st.c.ports[p]! = reg.schema := by
  intro st hst reg hreg
  obtain ⟨-, hsetup⟩ := FlockDeploy.setupOf_some hset
  have hck : _root_.Flock.checkRegistered st (some own) = .ok () := by
    rw [← hown]
    exact setup_checkRegistered hsetup st hst
  obtain ⟨tb, idx, -, -, hp⟩ := FlockSoundness.Discharge.ZkReg.check_tables
    (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hck)
  exact checkPort_schema (hp reg hreg)

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **`EntryFact` at the instance.** -/
theorem entryFact_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀))
    (hM : PortsMember (held R B desc q c₀)) :
    EntryFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) := by
  intro r hr i hi
  rw [wireMap_held]
  show r.EntryHolds (held R B desc q c₀) id (returned (held R B desc q c₀)) (addressOf (held R B desc q c₀) i) i
  obtain ⟨rp, hb⟩ := hN r hr i hi
  obtain ⟨ht0, hports⟩ := templateOf_held R B desc q c₀ hM hr
  obtain ⟨hrowP, p, hp, hpn, own, reg₀, hown, hregm, hrn⟩ := bit_held R B desc q c₀ hR hM hr hb
  have haddr := address_held R B desc q c₀ (bitAtGate_of_bitOf hb)
  have hport : (addressOf (held R B desc q c₀) i).port r.u = p := by rw [haddr, hpn]
  obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 (heldLink_held R B desc q c₀ rp.row hrowP)
  have hpl : p < (stmtOf R.x).ports.size := List.mem_range.1 (List.mem_of_find?_eq_some hp)
  have hslot : (addressOf (held R B desc q c₀) i).slot.key = .port l.value ∧
      (addressOf (held R B desc q c₀) i).slot.schema = _root_.Flock.PortRoots.schemaOf (stmtOf R.x) p := by
    rw [haddr]
    show (slotAt (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) rp).key = _ ∧
      (slotAt (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) rp).schema = _
    unfold slotAt
    rw [hpn, ht0]
    have hl' : heldLink (held R B desc q c₀) ((held R B desc q c₀).names.getD 0 "") rp.row = some l := hl
    rw [hl']
    exact ⟨rfl, rfl⟩
  have hr' := hr
  obtain ⟨Pt, d, spec, hx, -, -, -, hok, -, -, -, -, -, hset⟩ := hr'
  refine ⟨fun reg hreg => ?_, fun hnone => ?_⟩
  · rw [hport] at hreg
    have hname := regNamed_name R B desc q c₀ hM hr hp hreg
    have hmem : reg ∈ own := by
      unfold regNamed at hreg
      rw [hown, Option.bind_some] at hreg
      exact Array.mem_of_find?_eq_some hreg
    refine ⟨?_, ?_, ?_⟩
    · rw [hport, hports, hname]
      exact hp
    · have hk : readKeyAt (held R B desc q c₀) r.x r.m₁ r.c₀ r.c₁ (templateOf (held R B desc q c₀).ms r.u)
          reg.name reg.value = .port l.value := by
        obtain ⟨st, rec, draw, hdx, hrec, hdraw, -, hnone⟩ := reads_held R B desc q c₀ hR hr
        unfold readKeyAt
        rw [hdx, hrec, hdraw, ht0]
        show readKey (ownOf (held R B desc q c₀) st rec draw r.c₀) R.name reg.name reg.value = _
        unfold readKey
        have hl'' : Spot.SecondStage.linkOf (ownOf (held R B desc q c₀) st rec draw r.c₀) R.name reg.name =
            some l := by
          rw [hname]
          exact hl
        rw [hl'', hnone]
        rfl
      unfold entryAt
      rw [hslot.1, hk]
    · rw [hslot.2]
      have hj : 0 < r.sts.size := CheckedRun.sts_pos hr
      obtain ⟨p', hp', hsch⟩ := schema_of_setup hset hown r.sts[0]!
        (by rw [getElem!_pos r.sts 0 hj]; exact Array.getElem_mem hj) reg hmem
      rw [hports, hname, hp] at hp'
      obtain rfl := Option.some.inj hp'
      rw [hports, getElem!_pos (stmtOf R.x).ports p hpl] at hsch
      unfold _root_.Flock.PortRoots.schemaOf
      rw [Array.getElem?_eq_getElem hpl]
      exact hsch
  · exfalso
    rw [hport] at hnone
    unfold regNamed at hnone
    rw [hown, Option.bind_some, hports] at hnone
    apply Array.find?_eq_none.1 hnone reg₀ hregm
    rw [hrn]
    exact beq_iff_eq.2 (beq_iff_eq.1 (List.find?_some hp)).symm

/-! ## The width at the instance -/

/-- **Each bit's position is inside its port's row**, on `R`'s statement circuit: every source and written bit's
position is one of the `16 w` bits of the first port of its row's name. -/
def RefWidths (R : Ref) : Prop :=
  ∀ y ∈ sourceBits (ms2 R.x) ++ writtenBits (ms2 R.x),
    y.2.2.bit < 16 * ((stmtOf R.x).ports[((List.range (stmtOf R.x).ports.size).find? fun q =>
      (stmtOf R.x).ports[q]!.name == y.2.2.row).getD 0]!).words

/-- **Every wired row has one width at the instance** (the address facts' `RowWidths`, written out): the wiring map
moves nothing, so each bit's gate is its own and only its position's range is asked (`RefWidths`). -/
theorem rowWidths_held (hW : RefWidths R) :
    ∀ i t k rp, BitOf (held R B desc q c₀) t k i rp → ∃ tw rq,
      bitAtGate (held R B desc q c₀) (wireMap (held R B desc q c₀) i) = some (tw, rq) ∧
      rq.bit < 16 * ((memberCircuit (held R B desc q c₀) tw).ports[portNamed (held R B desc q c₀) tw rq.row]!).words ∧
      ((memberCircuit (held R B desc q c₀) tw).ports[portNamed (held R B desc q c₀) tw rq.row]!).words =
        ((memberCircuit (held R B desc q c₀) t).ports[portNamed (held R B desc q c₀) t rp.row]!).words := by
  intro i t k rp hb
  refine ⟨t, rp, by rw [wireMap_held]; exact bitAtGate_of_bitOf hb, ?_, rfl⟩
  have hmem : (i, t, rp) ∈ sourceBits (ms2 R.x) ++ writtenBits (ms2 R.x) := by
    rcases hb.2 with ⟨hs, -⟩ | ⟨-, hw, -, -⟩
    · exact List.mem_append_left _ (List.mem_of_find?_eq_some hs)
    · exact List.mem_append_right _ (List.mem_of_find?_eq_some hw)
  exact hW _ hmem

end SpotFlock.Honest
