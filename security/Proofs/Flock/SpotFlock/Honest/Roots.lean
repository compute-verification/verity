module

public import Proofs.Flock.SpotFlock.Honest.Addresses
public import Proofs.Flock.SpotFlock.NeededBits
public import Proofs.Flock.SpotFlock.ReadsDesign
public import Proofs.Flock.SpotFlock.AddressesAt
public import Proofs.Flock.SpotFlock.Sessions
public import Proofs.Flock.SpotFlock.AddressFacts.Width

/-!
# The roots half of `OneTreeFact` at the one-template instance

At `held R B desc q c₀` the wiring map moves nothing (`wireMap_held`), so the first half of `OneTreeFact` is `rfl`.
The second half, `CheckedRun.RootsHold` at every gate an accepted run's checked unit needs, holds here because each
such gate is committed by its proof block and round 2 reads each registered value off the boundary record:

* every needed gate is a source bit, an input, or a written bit at one of `R`'s ports, which a link names and the
  wired program returns (`NeededBits`, `Ref.named`, `RefPorts.written`), so the proof block commits it and the wired
  program opens it under the statement's roots (`rootsWired_held`);
* the interior registration registers no value: round 2's `second-inputs` refuses any value that no link names as
  the opened row of a reference read, and no port of `R` is named like an opening row (`RefPorts.row`), so an accepted
  run's interior record has no `registered` entry (`registeredEntry_none`, `reads_held`);
* a link names the gate's row, so the session has a registered read of that name (`named_port_of_stmtOk`), and the
  gate's port is the session's first port of that name (`CheckedRun.ports_member`, `bit_held`). So a registered read is always
  named like the gate's port, it is of the gate's row (`regNamed_name`), and round 2 reads it under the statement's
  roots (`readRoots`).

That the interior registers no value is not a premise here: it follows from round 2's `second-inputs` at `R`'s
links. So the run this is true of has the second try's shape, an interior registration with no value; the first
try's, whose interior registers `logits`, round 2 refuses at this instance. It is at one key for the run: one `Held`,
one stage.

Taken as a hypothesis: `NeededBits` (being proved on its own). That a checked unit's session has its member's
ports is `CheckedRun.ports_member`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

/-- **What the roots ask of `R`'s ports**: every written bit's row is one of them, and none is named like an opening
row. -/
structure RefPorts (R : Ref) : Prop where
  written : ∀ y ∈ writtenBits (ms2 R.x), R.ports.contains y.2.2.row = true
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
  simp

theorem readsOk_at {h : Held} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {t : ℕ}
    (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) : readsOk h t x m₁ xS c₀ c₁ = true := by
  unfold stmtOkAt at hok
  cases hp : parse h x with
  | none => rw [hp] at hok; cases hok
  | some s =>
    rw [hp] at hok
    simp only [Bool.and_eq_true] at hok
    exact hok.2

/-! ## An accepted run at the instance -/

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **An accepted run's checked unit is of the one template**, where its session has the member's ports. -/
theorem templateOf_held {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) :
    templateOf (held R B desc q c₀).ms r.u = 0 ∧ r.sts[0]!.c.ports = (stmtOf R.x).ports := by
  obtain ⟨m, hm, hp⟩ := CheckedRun.ports_member hr
  obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.1 hm
  have hl : (held R B desc q c₀).ms.length = 1 := rfl
  have h0 : templateOf (held R B desc q c₀).ms r.u = 0 := by omega
  refine ⟨h0, ?_⟩
  rw [h0] at hm
  have hm0 : (held R B desc q c₀).ms[0]? = some (stmtOf R.x, #[0, 1]) := rfl
  rw [hm0] at hm
  rw [hp, ← Option.some.inj hm]

/-- **What round 2 holds of an accepted run at the instance**: its statement, round 1's record and the second
stage's draw evaluate, the statement's roots are those it decodes to, and the interior registration registers no
value. -/
theorem reads_held (hR : RefPorts R) {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) :
    ∃ st rec draw, decode r.x = some st ∧ jsonMsg r.m₁ = some rec ∧
      secondDraw (held R B desc q c₀) r.c₀ r.c₁ = some draw ∧ r.s.roots = st.roots ∧
      ∀ v, Spot.SecondStage.registeredEntry (ownOf (held R B desc q c₀) st rec draw r.c₀).interior v = none := by
  obtain ⟨Pt, d, spec, hx, -, -, -, hok, -⟩ := hr
  obtain ⟨st, hdx, hsr⟩ := parse_roots _ hx
  have hok0 := stmtOkAt_of _ hok (t := 0) (show 0 < (held R B desc q c₀).ms.length from Nat.zero_lt_one)
  obtain ⟨st', rec, draw, hst', hrec, hdraw, hin⟩ := inputs_of_readsOk (readsOk_at hok0)
  rw [hdx] at hst'
  obtain rfl := Option.some.inj hst'
  refine ⟨st, rec, draw, hdx, hrec, hdraw, hsr, registeredEntry_none rfl rfl (fun l hl => ?_) hin⟩
  have hl' : l ∈ R.ports.map fun p => (⟨R.name, p, p⟩ : Spot.SecondStage.Link) := hl
  obtain ⟨p, hp, rfl⟩ := Array.mem_map.1 hl'
  exact hR.row p hp

/-- **A needed gate's row and port at the instance**: its row is one of `R`'s ports; the member's first port of that
name is `p`, the gate's port; and the session registers a read of that name. -/
theorem bit_held (hR : RefPorts R)
    {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) {k i : ℕ} {rp : Spot.Bits.RowPos}
    (hb : BitOf (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) k i rp) :
    R.ports.contains rp.row = true ∧
      ∃ p, (List.range (stmtOf R.x).ports.size).find? (fun q => (stmtOf R.x).ports[q]!.name == rp.row) = some p ∧
        portNamed (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) rp.row = p ∧
        ∃ own reg, r.ins.registered = some own ∧ reg ∈ own ∧ reg.name = rp.row := by
  have hrowP : R.ports.contains rp.row = true := by
    rcases hb.2 with ⟨hs, -⟩ | ⟨-, hw, -, -⟩
    · exact (List.all_eq_true.1 R.named) _ (List.mem_of_find?_eq_some hs)
    · exact hR.written _ (List.mem_of_find?_eq_some hw)
  refine ⟨hrowP, ?_⟩
  have hrowM : rp.row ∈ R.ports := by simpa using hrowP
  obtain ⟨ht0, hports⟩ := templateOf_held R B desc q c₀ hr
  have hj : 0 < r.sts.size := CheckedRun.sts_pos hr
  obtain ⟨Pt, d, spec, hx, -, -, -, hok, -, -, -, -, -, hset⟩ := hr
  have hlt : templateOf (held R B desc q c₀).ms r.u < (held R B desc q c₀).ms.length := by
    rw [ht0]; exact Nat.zero_lt_one
  obtain ⟨ins, spec', sts, hset', hnamed⟩ := named_port_of_stmtOk _ hok hlt
  have he : some (ins, spec', sts) = some (r.ins, spec, r.sts) := hset'.symm.trans hset
  obtain ⟨rfl, -, rfl⟩ : ins = r.ins ∧ spec' = spec ∧ sts = r.sts := by simpa using he
  have hlinkM : (⟨R.name, rp.row, rp.row⟩ : Spot.SecondStage.Link) ∈ (held R B desc q c₀).links.getD #[] :=
    show _ ∈ R.ports.map fun p => (⟨R.name, p, p⟩ : Spot.SecondStage.Link) from Array.mem_map.2 ⟨rp.row, hrowM, rfl⟩
  obtain ⟨own, reg, hown, hregm, hrn, hsc⟩ :=
    hnamed rp.row (Or.inr ⟨_, hlinkM, by rw [ht0]; rfl, rfl⟩)
  obtain ⟨p, hp⟩ := hsc r.sts[0]! (by rw [getElem!_pos r.sts 0 hj]; exact Array.getElem_mem hj)
  rw [hports] at hp
  refine ⟨p, hp, ?_, own, reg, hown, hregm, hrn⟩
  unfold portNamed
  show ((List.range (stmtOf R.x).ports.size).find? fun q => (stmtOf R.x).ports[q]!.name == rp.row).getD 0 = p
  rw [hp]
  rfl

/-- **A registered read named like the gate's port is of the gate's row.** -/
theorem regNamed_name {r : CheckedRun (held R B desc q c₀).stage}
    (hr : r.Ok (held R B desc q c₀)) {p : ℕ} {row : String}
    (hp : (List.range (stmtOf R.x).ports.size).find? (fun q => (stmtOf R.x).ports[q]!.name == row) = some p)
    {reg : _root_.Flock.Registered.Port} (hreg : regNamed r.ins r.sts p = some reg) : reg.name = row := by
  have hports := (templateOf_held R B desc q c₀ hr).2
  unfold regNamed at hreg
  cases hi : r.ins.registered with
  | none => rw [hi] at hreg; cases hreg
  | some own =>
    rw [hi, Option.bind_some, hports] at hreg
    have hn := Array.find?_some hreg
    have hq := List.find?_some hp
    simp only [beq_iff_eq] at hn hq
    rw [hn, hq]

/-- **A needed gate's address at the instance** is its own position's: the member's first port of its row, and the
place and slot of its position. -/
theorem address_held {i t : ℕ} {rp : Spot.Bits.RowPos} (hbit : bitAtGate (held R B desc q c₀) i = some (t, rp)) :
    addressOf (held R B desc q c₀) i =
      ⟨fun _ => portNamed (held R B desc q c₀) t rp.row, placeAt (held R B desc q c₀) t rp,
        slotAt (held R B desc q c₀) t rp⟩ := by
  rw [addressOf_held, addressAt_held, hbit]

/-- **The wired program opens a needed gate under the statement's roots**: the proof block commits it, an input or
a written bit the wired program returns. -/
theorem rootsWired_held (hR : RefPorts R)
    {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) {k i : ℕ} {rp : Spot.Bits.RowPos}
    (hb : BitOf (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) k i rp) :
    r.rootsWired id (returned (held R B desc q c₀)) i = r.s.roots := by
  have ht0 := (templateOf_held R B desc q c₀ hr).1
  obtain ⟨Pt, d, spec, hx, -⟩ := hr
  have hprog : r.s.prog = Properties.Spot.bitsProgs (held R B desc q c₀).ms := parse_prog _ hx
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
  unfold CheckedRun.rootsWired
  rw [ite_eq_right]
  rintro ⟨-, hf⟩
  rw [hbd] at hf
  cases hf

/-- **The roots half of `OneTreeFact` at the instance**: at every gate an accepted run's checked unit needs, the roots
the wired program opens it under are those round 2 reads its registered value under, and a registered read is named
like its port. -/
theorem rootsHold_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀)) :
    ∀ r : CheckedRun (held R B desc q c₀).stage, r.Ok (held R B desc q c₀) → ∀ i, r.s.prog.Needs r.b i →
      r.RootsHold (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
        (addressOf (held R B desc q c₀) i) (wireMap (held R B desc q c₀) i) := by
  intro r hr i hi
  rw [wireMap_held]
  show r.RootsHold (held R B desc q c₀) id (returned (held R B desc q c₀)) (addressOf (held R B desc q c₀) i) i
  obtain ⟨rp, hb⟩ := hN r hr i hi
  have ht0 := (templateOf_held R B desc q c₀ hr).1
  obtain ⟨hrowP, p, hp, hpn, own, reg₀, hown, hregm, hrn⟩ := bit_held R B desc q c₀ hR hr hb
  have hport : (addressOf (held R B desc q c₀) i).port r.u = p := by
    rw [address_held R B desc q c₀ (bitAtGate_of_bitOf hb), hpn]
  refine ⟨fun reg hreg => ?_, fun hnone => ?_⟩
  · rw [rootsWired_held R B desc q c₀ hR hr hb]
    rw [hport] at hreg
    have hname := regNamed_name R B desc q c₀ hr hp hreg
    obtain ⟨st, rec, draw, hdx, hrec, hdraw, hsr, hnone⟩ := reads_held R B desc q c₀ hR hr
    have hlink : (Spot.SecondStage.linkOf (ownOf (held R B desc q c₀) st rec draw r.c₀) R.name reg.name).isSome =
        true := by
      rw [hname]
      exact heldLink_held R B desc q c₀ rp.row hrowP
    obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 hlink
    have hri : readsInterior (ownOf (held R B desc q c₀) st rec draw r.c₀)
        ((held R B desc q c₀).names.getD (templateOf (held R B desc q c₀).ms r.u) "") reg.name reg.value = false := by
      rw [ht0]
      show readsInterior (ownOf (held R B desc q c₀) st rec draw r.c₀) R.name reg.name reg.value = false
      unfold readsInterior
      rw [hl, hnone]
      rfl
    simp only [readRoots, hdx, hrec, hdraw, hri, Bool.false_eq_true, ↓reduceIte]
    exact hsr
  · exfalso
    rw [hport] at hnone
    unfold regNamed at hnone
    rw [hown, Option.bind_some, (templateOf_held R B desc q c₀ hr).2] at hnone
    apply Array.find?_eq_none.1 hnone reg₀ hregm
    have hq := List.find?_some hp
    simp only [beq_iff_eq] at hq ⊢
    rw [hrn, hq]

/-- **`OneTreeFact` at the instance**: the wiring map moves nothing, and the roots hold (`rootsHold_held`). -/
theorem oneTreeFact_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀)) :
    OneTreeFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) :=
  ⟨fun i => by rw [wireMap_held]; exact ⟨rfl, rfl⟩, rootsHold_held R B desc q c₀ hR hN⟩

end SpotFlock.Honest
