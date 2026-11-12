module

public import Proofs.Flock.SpotFlock.Honest.Addresses
public import Proofs.Flock.SpotFlock.NeededBits
public import Proofs.Flock.SpotFlock.ReadsDesign
public import Proofs.Flock.SpotFlock.AddressesAt
public import Proofs.Flock.SpotFlock.Sessions

/-!
# The roots half of `OneTreeFact` at the one-template instance

At `held R B desc q c₀` the wiring map moves nothing (`wireMap_held`), so the first half of `OneTreeFact` is `rfl`.
The second half, `CheckedRun.RootsHold` at every gate an accepted run's checked unit needs, holds here because each
such gate is committed by its proof block and round 2 reads each registered value off the boundary record:

* every needed gate is a source bit, an input, or a written bit at one of `R`'s ports, which a link names and the
  wired program returns (`NeededBits`, `RefPorts.written`), so the proof block commits it and the wired program opens
  it under the statement's roots;
* the interior registration registers no value: round 2's `second-inputs` refuses any value that no link names as
  the opened row of a reference read, and no port of `R` is named like an opening row (`RefPorts.row`), so an accepted
  run's interior record has no `registered` entry (`registeredEntry_none`);
* every registered read is named like a port of the member's circuit, each of which a link names (`PortsMember`,
  `RefPorts.all`), so round 2 reads it under the statement's roots (`readRoots`).

So no registered read is ever missing at a needed gate's port, and the interior arm never applies.

The statement is about the second try's tree, whose interior registration registers no value; at a design whose
interior registers one (the first try's `logits`), `registeredEntry_none` fails and so does this proof. It is at one
key for the run: one `Held`, one stage.

Taken as hypotheses: `NeededBits` (being proved on its own) and `PortsMember`, the statement of the address facts'
`CheckedRun.ports_member` (not yet on this file's base).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

/-- **The session's ports are its member's**: at every accepted run's checked unit, the first table's circuit has
its template's member circuit's ports (the statement of `CheckedRun.ports_member`). -/
def PortsMember (h : Held) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∃ m, h.ms[templateOf h.ms r.u]? = some m ∧ r.sts[0]!.c.ports = m.1.ports

/-- **What the roots ask of `R`'s ports**: every written bit's row is one of them, every port of the statement's
circuit is one of them, the circuit has a port, and none is named like an opening row. -/
structure RefPorts (R : Ref) : Prop where
  written : ∀ y ∈ writtenBits (ms2 R.x), R.ports.contains y.2.2.row = true
  all : ∀ q (hq : q < (stmtOf R.x).ports.size), R.ports.contains ((stmtOf R.x).ports[q]).name = true
  pos : 0 < (stmtOf R.x).ports.size
  row : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false

/-! ## Round 2 refuses an interior registration's value no link opens -/

theorem all_eq_isEmpty {α : Type} (xs : Array α) (g : α → Bool) (hg : ∀ x ∈ xs, g x = false) :
    xs.all g = xs.isEmpty := by
  cases he : xs.isEmpty with
  | false =>
    rw [Bool.eq_false_iff]
    intro hall
    have hpos : 0 < xs.size := by
      rcases Nat.eq_zero_or_pos xs.size with h0 | h0
      · simp [Array.isEmpty, h0] at he
      · exact h0
    have h1 := (Array.all_eq_true.1 hall) 0 hpos
    rw [hg _ (Array.getElem_mem hpos)] at h1
    cases h1
  | true =>
    have h0 : xs.size = 0 := by simpa [Array.isEmpty] using he
    rw [Array.all_eq_true]
    intro i hi
    omega

theorem openedRow_false (own : Spot.SecondStage.Own) (names : Array String)
    (mine : Array (String × Array (Array (String × String)))) (x : Spot.SecondStage.Link)
    (hx : Spot.SecondStage.OPENING_ROW.contains x.port = false) :
    Spot.SecondStage.openedRow own names mine x = false := by
  unfold Spot.SecondStage.openedRow
  simp only [hx, Bool.and_false, Bool.false_and]

/-- **An interior registration round 2 accepts registers no value**, where the verifier fixes no value, fills none
from its own bytes, and no link's port is named like an opening row: each registered value is then one no link
names as an opened row, which `second-inputs` refuses. -/
theorem registeredEntry_none {own : Spot.SecondStage.Own} (hf : own.fixed = #[]) (hs : own.supplied = #[])
    (hrow : ∀ l ∈ own.links.getD #[], Spot.SecondStage.OPENING_ROW.contains l.port = false)
    (hin : Spot.SecondStage.inputsFailure own = none) (v : String) :
    Spot.SecondStage.registeredEntry own.interior v = none := by
  unfold Spot.SecondStage.registeredEntry
  rw [Array.find?_eq_none]
  intro e he hv
  have hv' : Spot.getD e "value" = .str v := by
    revert hv
    cases Spot.getD e "value" <;> simp [Spot.jsonStrIs]
    intro h; rw [h]
  unfold Spot.SecondStage.inputsFailure at hin
  have hall : ∀ (ns : Array String) (mn : Array (String × Array (Array (String × String)))) (w : String),
      ((own.links.getD #[]).filter (·.value == w)).all (Spot.SecondStage.openedRow own ns mn) =
        ((own.links.getD #[]).filter (·.value == w)).isEmpty := fun ns mn w =>
    all_eq_isEmpty _ _ fun l hl => openedRow_false _ _ _ _ (hrow l (Array.mem_filter.1 hl).1)
  have hmine : Spot.SecondStage.ownValues own = #[] := by simp [Spot.SecondStage.ownValues, hs]
  have hn := Array.find?_eq_none.1 hin v (Array.mem_filterMap.2 ⟨e, he, by rw [hv']⟩)
  apply hn
  simp only [hall, hmine, hf, Bool.not_and_self, Bool.not_false, Bool.true_or, Bool.and_true,
    Array.any_empty, Bool.or_false, Bool.not_eq_true']
  rfl

/-! ## The roots at the instance -/

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

theorem readsOk_at {h : Held} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {t : ℕ}
    (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) : readsOk h t x m₁ xS c₀ c₁ = true := by
  unfold stmtOkAt at hok
  cases hp : parse h x with
  | none => rw [hp] at hok; cases hok
  | some s =>
    rw [hp] at hok
    simp only [Bool.and_eq_true] at hok
    exact hok.2

/-- **An accepted run's checked unit is of the one template**, where its session has the member's ports. -/
theorem templateOf_held (hM : PortsMember (held R B desc q c₀)) {r : CheckedRun (held R B desc q c₀).stage}
    (hr : r.Ok (held R B desc q c₀)) :
    templateOf (held R B desc q c₀).ms r.u = 0 ∧ r.sts[0]!.c.ports = (stmtOf R.x).ports := by
  obtain ⟨m, hm, hp⟩ := hM r hr
  obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.1 hm
  have hl : (held R B desc q c₀).ms.length = 1 := rfl
  have h0 : templateOf (held R B desc q c₀).ms r.u = 0 := by omega
  refine ⟨h0, ?_⟩
  rw [h0] at hm
  have hm0 : (held R B desc q c₀).ms[0]? = some (stmtOf R.x, #[0, 1]) := rfl
  rw [hm0] at hm
  rw [hp, ← Option.some.inj hm]

/-- **Every registered read of an accepted run's checked unit is named like one of `R`'s ports.** -/
theorem regName_held (hR : RefPorts R) (hM : PortsMember (held R B desc q c₀))
    {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) (t : ℕ) (row : String)
    {reg : _root_.Flock.Registered.Port}
    (hreg : regNamed r.ins r.sts (portNamed (held R B desc q c₀) t row) = some reg) :
    R.ports.contains reg.name = true := by
  have hports := (templateOf_held R B desc q c₀ hM hr).2
  have hmc : memberCircuit (held R B desc q c₀) t = stmtOf R.x := rfl
  have hqlt : portNamed (held R B desc q c₀) t row < (stmtOf R.x).ports.size := by
    unfold portNamed
    rw [hmc]
    rcases hf : (List.range (stmtOf R.x).ports.size).find? (fun q => (stmtOf R.x).ports[q]!.name == row) with _ | q'
    · exact hR.pos
    · exact List.mem_range.1 (List.mem_of_find?_eq_some hf)
  unfold regNamed at hreg
  cases hi : r.ins.registered with
  | none => rw [hi] at hreg; cases hreg
  | some own =>
    rw [hi, Option.bind_some] at hreg
    have hn : reg.name = (r.sts[0]!.c.ports[portNamed (held R B desc q c₀) t row]!).name :=
      beq_iff_eq.1 (Array.find?_some hreg)
    rw [hn, hports, getElem!_pos (stmtOf R.x).ports _ hqlt]
    exact hR.all _ hqlt

/-- **The roots half of `OneTreeFact` at the instance**: at every gate an accepted run's checked unit needs, the roots
the wired program opens it under are those round 2 reads its registered value under, and a registered read is named
like its port. -/
theorem rootsHold_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀))
    (hM : PortsMember (held R B desc q c₀)) :
    ∀ r : CheckedRun (held R B desc q c₀).stage, r.Ok (held R B desc q c₀) → ∀ i, r.s.prog.Needs r.b i →
      r.RootsHold (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
        (addressOf (held R B desc q c₀) i) (wireMap (held R B desc q c₀) i) := by
  intro r hr i hi
  have hr' := hr
  rw [wireMap_held]
  show r.RootsHold (held R B desc q c₀) id (returned (held R B desc q c₀)) (addressOf (held R B desc q c₀) i) i
  obtain ⟨rp, hb⟩ := hN r hr i hi
  have ht0 := (templateOf_held R B desc q c₀ hM hr).1
  have hbit := bitAtGate_of_bitOf hb
  have haddr : (addressOf (held R B desc q c₀) i).port r.u =
      portNamed (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) rp.row := by
    rw [addressOf_held, addressAt_held, hbit]
  obtain ⟨Pt, d, spec, hx, -, -, -, hok, -, -, -, -, -, hset⟩ := hr
  have hprog : r.s.prog = bitsProgs (held R B desc q c₀).ms := parse_prog _ hx
  have hbd : (r.s.prog.wired id (returned (held R B desc q c₀))).Boundary (cutBlock id r.rb) i = true := by
    rw [hprog, wired_id_keep]
    rcases hb.2 with ⟨-, hin⟩ | ⟨-, hw, hout, -⟩
    · simp [Prog.Boundary, hin]
    · have hmem := List.mem_of_find?_eq_some hw
      have hrow : R.ports.contains rp.row = true := hR.written _ hmem
      have hret : returned (held R B desc q c₀) i = true := by
        unfold returned
        rw [List.any_eq_true]
        refine ⟨_, hmem, ?_⟩
        simp only [beq_self_eq_true, Bool.true_and]
        rw [ht0]
        exact heldLink_held R B desc q c₀ rp.row hrow
      simp [Prog.Boundary, List.mem_filter, hout, hret]
  have hrw : r.rootsWired id (returned (held R B desc q c₀)) i = r.s.roots := by
    unfold CheckedRun.rootsWired
    rw [if_neg]
    rintro ⟨-, hf⟩
    rw [hbd] at hf
    cases hf
  refine ⟨fun reg hreg => ?_, fun hnone => ?_⟩
  · rw [hrw]
    rw [haddr] at hreg
    have hname := regName_held R B desc q c₀ hR hM hr' _ _ hreg
    obtain ⟨st, hdx, hsr⟩ := parse_roots _ hx
    have hok0 := stmtOkAt_of _ hok (t := 0) (by decide)
    obtain ⟨st', rec, draw, hst', hrec, hdraw, hin⟩ := inputs_of_readsOk (readsOk_at hok0)
    rw [hdx] at hst'
    obtain rfl := Option.some.inj hst'
    have hnone : ∀ v, Spot.SecondStage.registeredEntry
        (ownOf (held R B desc q c₀) st rec draw r.c₀).interior v = none :=
      registeredEntry_none rfl rfl
        (fun l hl => by
          have hl' : l ∈ R.ports.map fun p => (⟨R.name, p, p⟩ : Spot.SecondStage.Link) := hl
          obtain ⟨p, hp, rfl⟩ := Array.mem_map.1 hl'
          exact hR.row p hp)
        hin
    have hlink : (Spot.SecondStage.linkOf (ownOf (held R B desc q c₀) st rec draw r.c₀) R.name reg.name).isSome =
        true := heldLink_held R B desc q c₀ reg.name hname
    obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 hlink
    have hri : readsInterior (ownOf (held R B desc q c₀) st rec draw r.c₀)
        ((held R B desc q c₀).names.getD (templateOf (held R B desc q c₀).ms r.u) "") reg.name reg.value = false := by
      rw [ht0]
      show readsInterior (ownOf (held R B desc q c₀) st rec draw r.c₀) R.name reg.name reg.value = false
      unfold readsInterior
      rw [hl, hnone]
      rfl
    unfold readRoots
    rw [hdx, hrec, hdraw, if_neg (by rw [hri]; decide), hsr]
  · exfalso
    have hrowP : R.ports.contains rp.row = true := by
      rcases hb.2 with ⟨hs, -⟩ | ⟨-, hw, -, -⟩
      · exact (List.all_eq_true.1 R.named) _ (List.mem_of_find?_eq_some hs)
      · exact hR.written _ (List.mem_of_find?_eq_some hw)
    have hrowM : rp.row ∈ R.ports := by simpa using hrowP
    have hports := (templateOf_held R B desc q c₀ hM hr').2
    have hlt : templateOf (held R B desc q c₀).ms r.u < (held R B desc q c₀).ms.length := by
      rw [ht0]; decide
    obtain ⟨ins, spec', sts, hset', hnamed⟩ := named_port_of_stmtOk _ hok hlt
    have he : some (ins, spec', sts) = some (r.ins, spec, r.sts) := hset'.symm.trans hset
    obtain ⟨rfl, -, rfl⟩ : ins = r.ins ∧ spec' = spec ∧ sts = r.sts := by simpa using he
    have hlinkM : (⟨R.name, rp.row, rp.row⟩ : Spot.SecondStage.Link) ∈ (held R B desc q c₀).links.getD #[] :=
      show _ ∈ R.ports.map fun p => (⟨R.name, p, p⟩ : Spot.SecondStage.Link) from Array.mem_map.2 ⟨rp.row, hrowM, rfl⟩
    obtain ⟨own, reg, hown, hregm, hrn, hsc⟩ :=
      hnamed rp.row (Or.inr ⟨_, hlinkM, by rw [ht0]; rfl, rfl⟩)
    have hj : 0 < r.sts.size := CheckedRun.sts_pos hr'
    obtain ⟨p, hp⟩ := hsc r.sts[0]! (by rw [getElem!_pos r.sts 0 hj]; exact Array.getElem_mem hj)
    rw [hports] at hp
    have hpn : portNamed (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) rp.row = p := by
      unfold portNamed
      show ((List.range (stmtOf R.x).ports.size).find? fun q => (stmtOf R.x).ports[q]!.name == rp.row).getD 0 = p
      rw [hp]
      rfl
    rw [haddr, hpn] at hnone
    unfold regNamed at hnone
    rw [hown, Option.bind_some, hports] at hnone
    apply Array.find?_eq_none.1 hnone reg hregm
    rw [hrn]
    exact beq_iff_eq.2 (beq_iff_eq.1 (List.find?_some hp)).symm

end SpotFlock.Honest
