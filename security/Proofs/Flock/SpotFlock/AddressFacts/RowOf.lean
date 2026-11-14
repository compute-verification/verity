module

public import Proofs.Flock.SpotFlock.EntryRoots
public import Proofs.Flock.SpotFlock.AddressFacts.RowParse
public import Proofs.Flock.SpotFlock.AddressFacts.RowRead

/-!
# The row fact of the address premise, at the address map

`SpotFlock.RowFact h (wireMap h) (returned h) (addressOf h)` is the row fact `addressesConnected_of_runs` reads of the
address map: at every accepted run's checked inner block (its unit `r.u`), at each gate the inner block's proof needs,
the inner block's instance is one of its session's, the slot of the gate's address has the leaf count of what the port
reads, and the slot's leaf is the row the session reads (`CheckedRun.RowHolds`). `rowFact_of_runs` proves it from the
wiring facts of each accepted run (`Wiring h`, as `entryFact_of_wiring` takes them) and the three premises below.

The proof follows `CheckedRun.holds_of_wiring`, case by case, at the same bit of the inner block's member:

* **A port a link names** (`rowHolds_linked`), a source bit or a written one. A registered read is named like the
  port. The slot is not sized and counts no units, so under any roots its leaf count is the entry's and its leaf is
  its position: the `per` rows of the inner block's outer block (its replay block), and the cell position's row among
  them. The second stage holds the read to the same row (`SecondStage.linkedRow`, through `readOk_linked` and
  `placedOk_at`). The check's other arm, a value the interior registration registers, is the one `second-inputs`
  excludes.
* **A port a wire names** (`wired_reg`), a source bit the wiring sends to its writer's gate inside the outer block.
  The slot is the writer's port of the interior record, sized, on a stride of the writer's inner blocks an outer
  block. The second stage holds the read to the writer's row (`SecondStage.wiredRow`, through `readOk_wired`), `per`
  rows an outer block, and `per` is the writer's count (`readOk_wired`'s `pw.n = r.per`). The read's leaf count is
  the sample's size times it (`placedOk_leaves`).
* **A template's own port** (the last case of `rowHolds_of_wiring`), a written bit no link names. With a registered
  read, the second stage holds each slot to its own first row (`SecondStage.rowsOk`, through `readOk_own_rows` and
  `rowsOk_at`): one row an inner block where the read's leaf count is the units' (`OwnLeaves`). With none, the row
  is the published tree's (`PublishedRows`).

The premises. Each is a statement about an accepted run that this file takes and does not prove:

* `Sits`: where the second stage puts the checked inner block. Its instance is one of its session's and the session's
  instances are labelled by their slots (`InstanceFact`'s first two clauses); the instance's place in its template's
  part is inside the part, in the cell of the inner block's outer block, at the inner block's place in that cell.
* `OwnLeaves`: a registered read at a template's own written port has one leaf an inner block.
* `PublishedRows`: `RowHolds`' last arm at a template's own written port with no registered read: the public file's
  tree has one leaf an inner block, and the sampled inner block's row is its own place.

That Flock's parse of a registered-reads file keeps each port's reads is `portsOf_reads` (`RowParse`), proved.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart memberProg)
open Lean (Json)

namespace AddressRow

variable {h : Held}

/-! ## The premises -/

/-- **Where the second stage puts a checked inner block**: at an accepted run, the inner block's instance is one of
its session's (`Counted`), the session's instances are labelled by their slots (`LabelledAt` at `drawSlot`), and the
instance's place `v` in its template's part is inside the part, in the cell of the inner block's outer block (the
block's place among the first stage's record's blocks, and the sample's block there the inner block's own), at the
inner block's place in that cell. -/
def Sits (h : Held) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h →
    r.Counted h ∧ LabelledAt (drawSlot (partOf h r) r.ins.read.draw) r.sts ∧
      unitOf r.ins.read.draw (r.n h) < (sampleOf h r).size * (partOf h r).n ∧
      unitOf r.ins.read.draw (r.n h) / (partOf h r).n = placeOf (iroots r.x r.seen r.ρ) ∧
      (sampleOf h r)[unitOf r.ins.read.draw (r.n h) / (partOf h r).n]? = some (cellBlock h r.u) ∧
      unitOf r.ins.read.draw (r.n h) % (partOf h r).n = cellPos h r.u

/-- **A registered read at a template's own written port has one leaf an inner block**: at an accepted run, a
registered read named like the row of a written bit of the inner block's template, at a port no link names, has the
sample's size times the template's count of leaves. -/
def OwnLeaves (h : Held) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ own reg, r.ins.registered = some own → reg ∈ own →
    (∃ (y : ℕ) (rp : Spot.Bits.RowPos), (y, templateOf h.ms r.u, rp) ∈ writtenBits h.ms ∧ rp.row = reg.name) →
    heldLink h (h.names.getD (templateOf h.ms r.u) "") reg.name = none →
    reg.leaves = (sampleOf h r).size * h.reg.ns.getD (templateOf h.ms r.u) 0

/-- **The published rows at a template's own written port with no registered read**: at an accepted run, at the row
of a written bit of the inner block's template that no link names and no registered read is named like, the session
has one table, and over any population its public file loads the port's published tree has one leaf an inner block
and the sampled inner block's row is its own place. -/
def PublishedRows (h : Held) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ (y : ℕ) (rp : Spot.Bits.RowPos),
    (y, templateOf h.ms r.u, rp) ∈ writtenBits h.ms →
    heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = none →
    regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) = none →
      r.ins.read.I.count ≤ 1 ∧
      ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
        (pubPort r.sts[0]!.c pop (portNamed h (templateOf h.ms r.u) rp.row)).leaves =
            (sampleOf h r).size * h.reg.ns.getD (templateOf h.ms r.u) 0 ∧
          rowAt r.sts[0]!.c pop (portNamed h (templateOf h.ms r.u) rp.row) (unitOf r.ins.read.draw (r.n h)) =
            unitOf r.ins.read.draw (r.n h)

/-! ## Rows and cells -/

/-- A cell position's row among its outer block's is the rows' own, below `per`. -/
theorem cellRow_some {rr : Spot.SecondStage.Rows} {c x : ℕ} (hc : cellRow rr c = some x) :
    Spot.SecondStage.rowAt rr.runs c = some x ∧ x < rr.per := by
  unfold cellRow at hc
  obtain ⟨y, hy, hif⟩ := Option.bind_eq_some_iff.1 hc
  have hif' : (if y < rr.per then some y else none) = some x := hif
  by_cases hlt : y < rr.per
  · rw [ite_eq_left hlt] at hif'
    cases hif'
    exact ⟨hy, hlt⟩
  · rw [ite_eq_right hlt] at hif'
    cases hif'

/-- **The row the second stage holds a wired port's slot to**, at the slot `v` places into its part: its cell
position's row, `per` rows after each cell before it. -/
theorem wiredRow_at {p : Spot.SecondStage.Part} {rr : Spot.SecondStage.Rows} {v pos : ℕ}
    (hrow : Spot.SecondStage.wiredRow p rr (p.lo + v) = some pos) :
    ∃ x, cellRow rr (v % p.n) = some x ∧ pos = v / p.n * rr.per + x := by
  have hu : p.unit (p.lo + v) = v % p.n := congrArg (fun k => k % p.n) (Nat.add_sub_cancel_left p.lo v)
  have hs : p.span (p.lo + v) = v / p.n := congrArg (fun k => k / p.n) (Nat.add_sub_cancel_left p.lo v)
  rw [wiredRow_cell, hu, hs] at hrow
  obtain ⟨x, hx, hpos⟩ := Option.map_eq_some_iff.1 hrow
  exact ⟨x, hx, hpos.symm⟩

/-- **The row the second stage holds a linked port's slot to**, at the slot `v` places into its part, where the
sample's block at the slot's cell is `b` and the slot's place in the cell is `c`: `c`'s row, `per` rows after each
outer block before `b`. -/
theorem linked_pos {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {rr : Spot.SecondStage.Rows}
    {v P b c : ℕ} (hrow : Spot.SecondStage.linkedRow own p rr (p.lo + v) = some P)
    (hb : own.sample[v / p.n]? = some b) (hc : v % p.n = c) :
    ∃ x, Spot.SecondStage.rowAt rr.runs c = some x ∧ P = b * rr.per + x := by
  have hu : p.unit (p.lo + v) = v % p.n := congrArg (fun k => k % p.n) (Nat.add_sub_cancel_left p.lo v)
  have hs : p.span (p.lo + v) = v / p.n := congrArg (fun k => k / p.n) (Nat.add_sub_cancel_left p.lo v)
  rw [linkedRow_cell, hu, hs, hb, hc] at hrow
  have hrow' : (cellRow rr c).map (fun x => b * rr.per + x) = some P := hrow
  obtain ⟨x, hx, hP⟩ := Option.map_eq_some_iff.1 hrow'
  exact ⟨x, (cellRow_some hx).1, hP.symm⟩

/-- **Where the wiring sends an inner block at a wired port**: its cell position's writer position `x`, in the
writer's cell of the same outer block. -/
theorem wireUnit_cell {t : ℕ} {port : String} {u tw : ℕ} {wport : String} {u' : ℕ}
    (hy : wireUnit h t port u = some (tw, wport, u')) :
    ∃ x, wireCell h t port (u % h.reg.ns.getD t 0) = some (tw, wport, x) ∧
      u' = u / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + x := by
  unfold wireUnit at hy
  obtain ⟨⟨a, b, x⟩, hc, he⟩ := Option.map_eq_some_iff.1 hy
  have h1 : a = tw := congrArg Prod.fst he
  have h2 : b = wport := congrArg (fun z : ℕ × String × ℕ => z.2.1) he
  have h3 : u / h.reg.ns.getD t 0 * h.reg.ns.getD a 0 + x = u' :=
    congrArg (fun z : ℕ × String × ℕ => z.2.2) he
  subst h1
  subst h2
  exact ⟨x, hc, h3.symm⟩

/-! ## The slot of a row position -/

/-- At a port a link names, a row position's slot is not sized and counts no units, at the rows' position. -/
theorem slotAt_link_shape {t : ℕ} {rp : Spot.Bits.RowPos} {l : Spot.SecondStage.Link}
    (hl : heldLink h (h.names.getD t "") rp.row = some l) :
    (slotAt h t rp).sized = false ∧ (slotAt h t rp).units = 0 ∧ (slotAt h t rp).stride = 0 ∧
      (slotAt h t rp).pos = rp.leaf / h.reg.ns.getD t 0 *
          ((heldRows h (h.names.getD t "") rp.row).map (·.per)).getD 0 +
        ((heldRows h (h.names.getD t "") rp.row).bind fun rows =>
          Spot.SecondStage.rowAt rows.runs (rp.leaf % h.reg.ns.getD t 0)).getD 0 := by
  unfold slotAt
  rw [hl]
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- At a port a link names whose rows give the cell position a row `x`, the slot's position is `x`, `per` rows after
each outer block before the position's. -/
theorem slotAt_link {t : ℕ} {rp : Spot.Bits.RowPos} {l : Spot.SecondStage.Link}
    (hl : heldLink h (h.names.getD t "") rp.row = some l) {rr : Spot.SecondStage.Rows}
    (hrr : heldRows h (h.names.getD t "") rp.row = some rr) {x : ℕ}
    (hx : Spot.SecondStage.rowAt rr.runs (rp.leaf % h.reg.ns.getD t 0) = some x) :
    (slotAt h t rp).pos = rp.leaf / h.reg.ns.getD t 0 * rr.per + x := by
  obtain ⟨-, -, -, h4⟩ := slotAt_link_shape hl
  have e1 : ((heldRows h (h.names.getD t "") rp.row).map (·.per)).getD 0 = rr.per := by
    exact (congrArg (fun o : Option Spot.SecondStage.Rows => (o.map (·.per)).getD 0) hrr).trans rfl
  have e2 : ((heldRows h (h.names.getD t "") rp.row).bind fun rows =>
      Spot.SecondStage.rowAt rows.runs (rp.leaf % h.reg.ns.getD t 0)).getD 0 = x := by
    exact (congrArg (fun o : Option Spot.SecondStage.Rows => (o.bind fun rows =>
        Spot.SecondStage.rowAt rows.runs (rp.leaf % h.reg.ns.getD t 0)).getD 0) hrr).trans
      ((congrArg (fun o : Option ℕ => o.getD 0) hx).trans rfl)
  exact h4.trans (congrArg₂ (fun a b => rp.leaf / h.reg.ns.getD t 0 * a + b) e1 e2)

/-- At a port no link names, a row position's slot is sized and counts no units of its own, at the position's place
in its cell on a stride of the member's inner blocks an outer block. -/
theorem slotAt_free_shape {t : ℕ} {rp : Spot.Bits.RowPos}
    (hl : heldLink h (h.names.getD t "") rp.row = none) :
    (slotAt h t rp).sized = true ∧ (slotAt h t rp).units = 0 ∧
      (slotAt h t rp).stride = h.reg.ns.getD t 0 ∧ (slotAt h t rp).pos = rp.leaf % h.reg.ns.getD t 0 := by
  unfold slotAt
  rw [hl]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem leavesAt_of_unsized (roots : List Msg) {s : Slot} (hs : s.sized = false) (e : Entry) :
    leavesAt roots s e = e.leaves := by
  unfold leavesAt
  exact ite_eq_right fun hc => absurd (hs.symm.trans hc) (by decide)

theorem leavesAt_of_sized (roots : List Msg) {s : Slot} (hs : s.sized = true) (e : Entry) :
    leavesAt roots s e = unitsAt roots s := by
  unfold leavesAt
  exact ite_eq_left hs

/-- A slot of no units and no stride counts no units, under any roots. -/
theorem unitsAt_of_flat (roots : List Msg) {s : Slot} (hu : s.units = 0) (hst : s.stride = 0) :
    unitsAt roots s = 0 := by
  rw [unitsAt, hu, hst, Nat.mul_zero, Nat.add_zero]

/-- A slot that counts no units opens its position. -/
theorem leafAt_of_flat (roots : List Msg) {s : Slot} (hu : unitsAt roots s = 0) (e : Entry) :
    leafAt roots s e = s.pos := by
  unfold leafAt
  exact ite_eq_left hu

/-- **A sized slot under an outer block's roots**, of no units of its own, on a stride of `n`, at position `c`: where
the first stage's record lists `q` blocks and `q · n` is positive, its leaf count is `q · n` and its leaf is `c`, `n`
after each block before the outer block's place. -/
theorem inside_slot {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) {s : Slot} {n c q : ℕ}
    (hlen : (blocksOf rec).length = q) (hs : s.sized = true) (hu : s.units = 0) (hst : s.stride = n)
    (hp : s.pos = c) (hq : 0 < q * n) (e : Entry) :
    leavesAt (irootsOf m₁ ρ) s e = q * n ∧ leafAt (irootsOf m₁ ρ) s e = c + (blocksOf rec).idxOf ρ * n := by
  subst hlen
  have hun : unitsAt (irootsOf m₁ ρ) s = (blocksOf rec).length * n := by
    rw [unitsAt_irootsOf hm, hu, hst, Nat.zero_add]
  have hpos : 0 < unitsAt (irootsOf m₁ ρ) s := by
    rw [hun]
    exact hq
  exact ⟨(leavesAt_of_sized _ hs e).trans hun,
    (leafAt_irootsOf hm ρ hs hpos e).trans (congrArg₂ (fun a b => a + (blocksOf rec).idxOf ρ * b) hp hst)⟩

/-! ## Arithmetic -/

theorem pos_of_lt_mul {v q n : ℕ} (hlt : v < q * n) : 0 < q := by
  rcases Nat.eq_zero_or_pos q with h0 | h0
  · rw [h0, Nat.zero_mul] at hlt
    exact absurd hlt (Nat.not_lt_zero _)
  · exact h0

/-- The writer's row at a wired port: row `x` of the cell at place `idx`, `per` rows a cell, is the writer's inner
block `c · N + x`'s place in its cell, `N` after each cell before, where `N` is `per`. -/
theorem wired_arith {P vd per x x' u' c N idx : ℕ} (hP : P = vd * per + x) (hN : N = per) (hx : x = x')
    (hidx : vd = idx) (hu : u' = c * N + x') (hlt : x' < per) : P = u' % N + idx * N := by
  rw [hP, hu, hx, hidx, hN]
  exact (Nat.add_comm (idx * per) x').trans
    (congrArg (fun t => t + idx * per) ((Nat.mul_add_mod' c per x').trans (Nat.mod_eq_of_lt hlt)).symm)

theorem split_arith {v n c idx : ℕ} (hc : v % n = c) (hidx : v / n = idx) : v = c + idx * n := by
  subst hc
  subst hidx
  exact ((Nat.add_comm (v % n) (v / n * n)).trans (Nat.div_add_mod' v n)).symm

theorem ratio_one {P v L U : ℕ} (hP : P = v * (L / U)) (hL : L = U) (hU : 0 < U) : P = v :=
  hP.trans ((congrArg (fun t => v * t) ((congrArg (fun t => t / U) hL).trans (Nat.div_self hU))).trans
    (Nat.mul_one v))

/-! ## The second stage's checks -/

/-- **At a template's own port of the interior record, the second stage holds each slot to its own first row**: the
last arm of `SecondStage.readOk`, its `rowsOk` clause (`readOk_own` reads its first). -/
theorem readOk_own_rows {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} {v : String} (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = none) (hv : Spot.getD e "value" = .str v)
    (hp : ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome = true)
    (hok : Spot.SecondStage.readOk own named p port e = true) :
    Spot.SecondStage.rowsOk own.sample.size p e (own.sample.size * p.n) (fun u => some (u - p.lo)) = true := by
  unfold Spot.SecondStage.readOk at hok
  rw [hv, hl, hw] at hok
  cases hf : (Spot.Record.byPort own.interior).find? (·.1 == v) with
  | none =>
    rw [hf] at hp
    cases hp
  | some kx =>
    obtain ⟨k, x⟩ := kx
    simp only [hf, Bool.and_eq_true, beq_iff_eq] at hok
    exact hok.2

/-- **An accepted statement's first-stage record lists the sample in its order** (`second-interior-order`), with the
run's statement, the first stage's record and the second stage's sample it was read at. -/
theorem readsOk_order {t : ℕ} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hr : readsOk h t x m₁ xS c₀ c₁ = true) :
    ∃ st rec draw, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      Spot.SecondStage.interiorOrderOk (ownOf h st rec draw c₀) = true := by
  unfold readsOk at hr
  cases hst : decode x with
  | none => rw [hst] at hr; cases hr
  | some st =>
    cases hrec : jsonMsg m₁ with
    | none => rw [hst, hrec] at hr; cases hr
    | some rec =>
      cases hdraw : secondDraw h c₀ c₁ with
      | none => rw [hst, hrec, hdraw] at hr; cases hr
      | some draw =>
        cases hw : FlockDeploy.decode xS with
        | none => rw [hst, hrec, hdraw, hw] at hr; cases hr
        | some w =>
          rw [hst, hrec, hdraw, hw] at hr
          simp only [Bool.and_eq_true] at hr
          obtain ⟨⟨⟨⟨⟨hord, -⟩, -⟩, -⟩, -⟩, -⟩ := hr
          exact ⟨st, rec, draw, rfl, rfl, rfl, hord⟩

end AddressRow

namespace CheckedRun

open AddressRow

variable {h : Held} {r : CheckedRun h.stage}

/-- **The second stage passes each registered read of a checked inner block's session, at the entry Flock parsed**:
`read_of_reg`, with the entry's leaf count and its reads as Flock's port has them (`portsOf_reads`). -/
theorem entry_of_reg (hr : r.Ok h) {st : Roots} {rec draw : Json} (hR : r.Reads st rec draw)
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : r.ins.registered = some own) (hmem : reg ∈ own) :
    ∃ e, Spot.getD e "value" = .str reg.value ∧ Spot.nat? (Spot.getD e "leaves") = some reg.leaves ∧
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec)
        (partOf h r) reg.name e = true ∧
      ∀ u pos, reg.readAt u = some (some pos) → readsAt e u = some pos := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st', rec', draw', w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  have e1 : st' = st := Option.some.inj (hst.symm.trans hR.dec)
  have e2 : rec' = rec := Option.some.inj (hrec.symm.trans hR.msg)
  have e3 : draw' = draw := Option.some.inj (hdraw.symm.trans hR.sample)
  rw [e1, e2, e3] at hc
  obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins (ins_of_setupOf hw hset) hown
  obtain ⟨-, -, hSr, -⟩ := stmtJsonOf_fields h hS
  have hSj : S.registered = jR := by
    have h4 := hSr bR hbR
    rw [hjR] at h4
    exact (Option.some.inj h4).symm
  obtain ⟨kvs, hports, hent⟩ := portsOf_reads (portsOf_of_ofJson hoj)
  obtain ⟨ej, hm, hf, hreads'⟩ := hent reg hmem
  obtain ⟨-, -, -, hlv, hv⟩ := hf
  obtain ⟨-, -, -, -, -, -, hreads, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hents : Spot.entries (Spot.getD S.registered "ports") = kvs.toArray := by
    simp only [hSj, getD_of_ok hports, Spot.entries]
  rw [hents] at hreads
  have hrd : Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec)
      (partAt h (ownOf h st rec draw r.c₀).sample.size (templateOf h.ms r.u)) reg.name ej = true :=
    Array.all_eq_true_iff_forall_mem.1 hreads _ hm
  exact ⟨ej, getD_of_ok hv, (congrArg Spot.nat? (getD_of_ok hlv)).trans (nat_num reg.leaves), hrd, hreads'⟩

/-- **The first stage's record lists the run's sample**, in its order, at an accepted run. -/
theorem blocks_of_ok (hr : r.Ok h) {st : Roots} {rec draw : Json} (hR : r.Reads st rec draw) :
    blocksOf rec = (sampleOf h r).toList := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st', rec', draw', hst, hrec, hdraw, hord⟩ := readsOk_order hat.2
  have e1 : st' = st := Option.some.inj (hst.symm.trans hR.dec)
  have e2 : rec' = rec := Option.some.inj (hrec.symm.trans hR.msg)
  have e3 : draw' = draw := Option.some.inj (hdraw.symm.trans hR.sample)
  rw [e1, e2, e3] at hord
  exact blocksOf_of_interiorOrderOk hord

/-- **The header's position for the checked inner block's instance is the registered read's at the instance's
slot** (`positionFact` at `drawSlot`), where the second stage puts the inner block there (`Sits`). -/
theorem readAt_of_ok (hS : Sits h) (hr : r.Ok h) {own : Array _root_.Flock.Registered.Port}
    {reg : _root_.Flock.Registered.Port} (hown : r.ins.registered = some own) (hmem : reg ∈ own) {q : ℕ}
    (hfind : (List.range r.sts[0]!.c.ports.size).find? (fun p => r.sts[0]!.c.ports[p]!.name == reg.name) = some q)
    {tb} {positions} {paths} (htb : r.sts[0]!.pub.tables = some tb)
    (hent : _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths)) :
    reg.readAt ((partOf h r).lo + unitOf r.ins.read.draw (r.n h)) =
      some (some (positions.getD (_root_.Flock.Registered.ref tb r.sts[0]!.c.ports.size q (r.n h)) 0)) := by
  obtain ⟨hcnt, hlab, -⟩ := hS r hr
  have hpos := sts_pos hr
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hsc := sessionScoped h r.x r.m₁ (unitStmt h r.m₂ r.u) r.c₀ r.c₁ _ (stmtOkAt_unit h hx hok hu)
  exact positionFact (drawSlot (partOf h r) r.ins.read.draw) hset hown hsc hck hpos hlab reg hmem q tb positions
    paths hfind htb hent (r.n h) hcnt

/-- **The row at a needed gate at a port a link names**, a gate the wiring leaves alone. A registered read is named
like the port; the address's slot is not sized and counts no units, so under any roots its leaf count is the
entry's and its leaf its position, the row the second stage holds the read to (`SecondStage.linkedRow`). -/
theorem rowHolds_linked (hS : Sits h) (hr : r.Ok h) {st : Roots} {rec draw : Json}
    (hR : r.Reads st rec draw) {i : ℕ} {rp : Spot.Bits.RowPos}
    (hb : BitOf h (templateOf h.ms r.u) (r.u - Properties.Spot.unitBase h.ms (templateOf h.ms r.u)) i rp)
    (hfind : (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == rp.row) =
      some (portNamed h (templateOf h.ms r.u) rp.row))
    (hname : (r.sts[0]!.c.ports[portNamed h (templateOf h.ms r.u) rp.row]!).name = rp.row)
    (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hfix : wireMap h i = i) {l : Spot.SecondStage.Link}
    (hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = some l) :
    r.RowHolds h (wireMap h) (returned h) (addressOf h i) (wireMap h i) := by
  have haddr : addressOf h i = ⟨fun _ => portNamed h (templateOf h.ms r.u) rp.row,
      placeAt h (templateOf h.ms r.u) rp, slotAt h (templateOf h.ms r.u) rp⟩ :=
    (addressOf_eq h hω i).trans (addressAt_of_fixed hb hfix)
  obtain ⟨own0, reg0, hown0, hmem0, hn0⟩ := reg_of_named hr (Or.inr (link_named h hl))
  have hsome : regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) ≠ none :=
    regNamed_ne_none hname hown0 hmem0 hn0
  obtain ⟨hcnt, -, -, -, hsamp, hmod⟩ := hS r hr
  have hcp : rp.leaf % h.reg.ns.getD (templateOf h.ms r.u) 0 = cellPos h r.u :=
    congrArg (fun k => k % h.reg.ns.getD (templateOf h.ms r.u) 0) hb.1
  have hcb : rp.leaf / h.reg.ns.getD (templateOf h.ms r.u) 0 = cellBlock h r.u :=
    congrArg (fun k => k / h.reg.ns.getD (templateOf h.ms r.u) 0) hb.1
  unfold RowHolds
  simp only [haddr]
  refine ⟨hcnt, fun reg hreg e _ hle => ?_, fun hnone => absurd hnone hsome⟩
  obtain ⟨own, hown, hmem, hn⟩ := regNamed_spec hname hreg
  obtain ⟨ej, -, -, hrd, hreads⟩ := entry_of_reg hr hR hown hmem
  rw [hn] at hrd
  obtain ⟨-, rr, hrr, hcase⟩ := readOk_linked (own := ownOf h st rec draw r.c₀)
    (named := Spot.SecondStage.byTemplate rec) (p := partOf h r) (port := rp.row) (e := ej) (l := l) hl hrd
  have hrr' : heldRows h (h.names.getD (templateOf h.ms r.u) "") rp.row = some rr := hrr
  obtain ⟨hsz, hun, hstr, -⟩ := slotAt_link_shape hl
  refine ⟨(leavesAt_of_unsized _ hsz e).trans hle, fun tb positions paths htb hent => ?_⟩
  rcases hcase with ⟨xe, hxe, -, -⟩ | ⟨-, bj, m, -, -, -, hpl⟩
  · have hno : Spot.SecondStage.registeredEntry rec l.value = none :=
      registeredEntry_none_of_inputs h hR.inputs l.value
    have hxe' : Spot.SecondStage.registeredEntry rec l.value = some xe := hxe
    rw [hno] at hxe'
    cases hxe'
  · have hfind' : (List.range r.sts[0]!.c.ports.size).find?
        (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some (portNamed h (templateOf h.ms r.u) rp.row) := by
      rw [hn]
      exact hfind
    have hP := hreads _ _ (readAt_of_ok hS hr hown hmem hfind' htb hent)
    obtain ⟨-, hrow⟩ := placedOk_at hpl hP
    obtain ⟨x, hx, hPx⟩ := linked_pos hrow hsamp hmod
    have hx' : Spot.SecondStage.rowAt rr.runs (rp.leaf % h.reg.ns.getD (templateOf h.ms r.u) 0) = some x :=
      (congrArg (fun c => Spot.SecondStage.rowAt rr.runs c) hcp).trans hx
    have hpos := slotAt_link hl hrr' hx'
    exact (leafAt_of_flat _ (unitsAt_of_flat _ hun hstr) e).trans
      (hpos.trans ((congrArg (fun b => b * rr.per + x) hcb).trans hPx.symm))

/-- **The row at a needed gate the wiring sends (or leaves) inside the inner block's outer block**, at a port
`rq.row` of template `tw` that no link names: the address's slot is `tw`'s port of the interior record, sized, on a
stride of `tw`'s count, and under the outer block's roots its leaf count is the sample's size times that count and
its leaf the position's place in its cell after the cells before the outer block's. `hread` and `hnone` are what the
session reads there, with a registered read named like the gate's port and with none. -/
theorem rowHolds_inside (hS : Sits h) (hr : r.Ok h) {st : Roots} {rec draw : Json} (hR : r.Reads st rec draw)
    {k i tw : ℕ} {rp rq : Spot.Bits.RowPos} (hb : BitOf h (templateOf h.ms r.u) k i rp)
    (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hw : bitAtGate h (wireMap h i) = some (tw, rq))
    (hlw : heldLink h (h.names.getD tw "") rq.row = none)
    (hin : wireMap h i ∈ cutBlock (wireMap h) r.rb ∧
      (r.s.prog.wired (wireMap h) (returned h)).Boundary (cutBlock (wireMap h) r.rb) (wireMap h i) = false)
    (hread : ∀ reg, regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) = some reg →
      0 < (sampleOf h r).size * h.reg.ns.getD tw 0 ∧ reg.leaves = (sampleOf h r).size * h.reg.ns.getD tw 0 ∧
      ∀ tb positions paths, r.sts[0]!.pub.tables = some tb →
        _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
        positions.getD (_root_.Flock.Registered.ref tb r.sts[0]!.c.ports.size
            (portNamed h (templateOf h.ms r.u) rp.row) (r.n h)) 0 =
          rq.leaf % h.reg.ns.getD tw 0 + (blocksOf rec).idxOf r.ρ * h.reg.ns.getD tw 0)
    (hnone : regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) = none →
      0 < (sampleOf h r).size * h.reg.ns.getD tw 0 ∧ r.ins.read.I.count ≤ 1 ∧
      ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
        (pubPort r.sts[0]!.c pop (portNamed h (templateOf h.ms r.u) rp.row)).leaves =
            (sampleOf h r).size * h.reg.ns.getD tw 0 ∧
          rowAt r.sts[0]!.c pop (portNamed h (templateOf h.ms r.u) rp.row) (unitOf r.ins.read.draw (r.n h)) =
            rq.leaf % h.reg.ns.getD tw 0 + (blocksOf rec).idxOf r.ρ * h.reg.ns.getD tw 0) :
    r.RowHolds h (wireMap h) (returned h) (addressOf h i) (wireMap h i) := by
  have haddr : addressOf h i = ⟨fun _ => portNamed h (templateOf h.ms r.u) rp.row, placeAt h tw rq,
      slotAt h tw rq⟩ :=
    (addressOf_eq h hω i).trans (addressAt_of_bitOf hb hw)
  have hroots : r.rootsWired (wireMap h) (returned h) (wireMap h i) = irootsOf r.m₁ r.ρ := rootsWired_inside hin
  have hir : iroots r.x r.seen r.ρ = irootsOf r.m₁ r.ρ := rfl
  have hlen : (blocksOf rec).length = (sampleOf h r).size :=
    (congrArg List.length (blocks_of_ok hr hR)).trans Array.length_toList
  obtain ⟨hsz, hun, hstr, hp⟩ := slotAt_free_shape hlw
  obtain ⟨hcnt, -⟩ := hS r hr
  unfold RowHolds
  simp only [haddr]
  refine ⟨hcnt, fun reg hreg e _ _ => ?_, fun hno => ?_⟩
  · obtain ⟨hq, hlv, hrow⟩ := hread reg hreg
    obtain ⟨h1, h2⟩ := inside_slot hR.msg r.ρ hlen hsz hun hstr hp hq e
    exact ⟨((congrArg (fun R => leavesAt R (slotAt h tw rq) e) hroots).trans h1).trans hlv.symm,
      fun tb positions paths htb hent =>
        ((congrArg (fun R => leafAt R (slotAt h tw rq) e) hroots).trans h2).trans
          (hrow tb positions paths htb hent).symm⟩
  · obtain ⟨hq, hc, hpub⟩ := hnone hno
    refine ⟨hc, fun pop hpop e _ => ?_⟩
    obtain ⟨hlv, hrow⟩ := hpub pop hpop
    obtain ⟨h1, h2⟩ := inside_slot hR.msg r.ρ hlen hsz hun hstr hp hq e
    exact ⟨((congrArg (fun R => leavesAt R (slotAt h tw rq) e) hir).trans h1).trans hlv.symm,
      ((congrArg (fun R => leafAt R (slotAt h tw rq) e) hir).trans h2).trans hrow.symm⟩

/-- **What a registered read at a port a wire names gives**: the read has the sample's size times the writer's count
of leaves, a positive number, and the header's position for the checked inner block's instance is the writer's inner
block's place in its cell, after the cells before the outer block's. -/
theorem wired_reg (hS : Sits h) (hr : r.Ok h) {st : Roots} {rec draw : Json}
    (hR : r.Reads st rec draw) {rp : Spot.Bits.RowPos} {q : ℕ}
    (hcp : rp.leaf % h.reg.ns.getD (templateOf h.ms r.u) 0 = cellPos h r.u)
    (hfind : (List.range r.sts[0]!.c.ports.size).find? (fun p => r.sts[0]!.c.ports[p]!.name == rp.row) = some q)
    (hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = none) {w : Spot.SecondStage.Wire}
    (hwire : heldWire h (h.names.getD (templateOf h.ms r.u) "") rp.row = some w) {tw : ℕ}
    (htw : heldWriter h w.writer = some tw) {wport : String} {u' : ℕ}
    (hwu : wireUnit h (templateOf h.ms r.u) rp.row rp.leaf = some (tw, wport, u'))
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : r.ins.registered = some own) (hmem : reg ∈ own) (hn : reg.name = rp.row) :
    0 < (sampleOf h r).size * h.reg.ns.getD tw 0 ∧ reg.leaves = (sampleOf h r).size * h.reg.ns.getD tw 0 ∧
      ∀ tb positions paths, r.sts[0]!.pub.tables = some tb →
        _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
        positions.getD (_root_.Flock.Registered.ref tb r.sts[0]!.c.ports.size q (r.n h)) 0 =
          u' % h.reg.ns.getD tw 0 + (blocksOf rec).idxOf r.ρ * h.reg.ns.getD tw 0 := by
  obtain ⟨-, -, hvlt, hdiv, -, hmod⟩ := hS r hr
  have hir : iroots r.x r.seen r.ρ = irootsOf r.m₁ r.ρ := rfl
  have hidx : unitOf r.ins.read.draw (r.n h) / (partOf h r).n = (blocksOf rec).idxOf r.ρ :=
    (hdiv.trans (congrArg placeOf hir)).trans (placeOf_irootsOf hR.msg r.ρ)
  obtain ⟨ej, -, hlv, hrd, hreads⟩ := entry_of_reg hr hR hown hmem
  rw [hn] at hrd
  obtain ⟨xe, pw, rr, -, -, hpw, hrr, hpn, -, hpl⟩ :=
    Spot.SecondStage.readOk_wired (own := ownOf h st rec draw r.c₀) (named := Spot.SecondStage.byTemplate rec)
      (p := partOf h r) (port := rp.row) (e := ej) (w := w) hl hwire hrd
  rw [writerOf_ownOf, htw] at hpw
  have hpw' : partAt h (h.reg.sample (src r.c₀)).size tw = pw := Option.some.inj hpw
  have hN : h.reg.ns.getD tw 0 = rr.per := (congrArg Spot.SecondStage.Part.n hpw').trans hpn
  have hrr' : heldRows h (h.names.getD (templateOf h.ms r.u) "") rp.row = some rr := hrr
  obtain ⟨x', hcell, hu'⟩ := wireUnit_cell hwu
  obtain ⟨w2, r2, -, -, hr2, -, -, hx2⟩ := wireCell_wire h hcell
  have hrr2 : r2 = rr := Option.some.inj (hr2.symm.trans hrr')
  rw [hrr2] at hx2
  have hx3 : cellRow rr (rp.leaf % h.reg.ns.getD (templateOf h.ms r.u) 0) = some x' := hx2
  obtain ⟨-, hx'lt⟩ := cellRow_some hx3
  have hntw : 0 < h.reg.ns.getD tw 0 :=
    Nat.lt_of_lt_of_eq (Nat.lt_of_le_of_lt (Nat.zero_le x') hx'lt) hN.symm
  have hleaves : reg.leaves = (sampleOf h r).size * h.reg.ns.getD tw 0 :=
    (Option.some.inj (hlv.symm.trans (placedOk_leaves hpl))).trans
      (congrArg (fun n => (sampleOf h r).size * n) hN.symm)
  refine ⟨Nat.mul_pos (pos_of_lt_mul hvlt) hntw, hleaves, fun tb positions paths htb hent => ?_⟩
  have hfind' : (List.range r.sts[0]!.c.ports.size).find?
      (fun p => r.sts[0]!.c.ports[p]!.name == reg.name) = some q := by
    rw [hn]
    exact hfind
  have hP := hreads _ _ (readAt_of_ok hS hr hown hmem hfind' htb hent)
  obtain ⟨-, hrow⟩ := placedOk_at hpl hP
  obtain ⟨x, hx, hPx⟩ := wiredRow_at hrow
  have hxx : x = x' :=
    Option.some.inj ((hx.symm.trans (congrArg (cellRow rr) (hmod.trans hcp.symm))).trans hx3)
  exact wired_arith hPx hN hxx hidx hu' hx'lt

/-- **The row at every gate a checked inner block needs**, at the address map: `holds_of_wiring`'s cases, each with
the row the second stage holds the session's read to. -/
theorem rowHolds_of_wiring (W : Wiring h) (hS : Sits h) (hL : OwnLeaves h)
    (hPub : PublishedRows h) (hr : r.Ok h) {i : ℕ} (hi : r.s.prog.Needs r.b i) :
    r.RowHolds h (wireMap h) (returned h) (addressOf h i) (wireMap h i) := by
  obtain ⟨st, rec, draw, hR⟩ := reads_of_ok hr
  obtain ⟨d, h2⟩ := hR.design
  have hω : ∀ j, wireMap h (wireMap h j) = wireMap h j := (rewires_of_two h2 W.first).idem
  have htop := topological_of_two h2
  obtain ⟨rp, hb⟩ := W.needed r hr i hi
  obtain ⟨hfind, -, hname⟩ := port_at W.rows W.ports hr hb
  obtain ⟨-, -, hvlt, hdiv, -, hmod⟩ := hS r hr
  have hcp : rp.leaf % h.reg.ns.getD (templateOf h.ms r.u) 0 = cellPos h r.u :=
    congrArg (fun k => k % h.reg.ns.getD (templateOf h.ms r.u) 0) hb.1
  have hir : iroots r.x r.seen r.ρ = irootsOf r.m₁ r.ρ := rfl
  have hidx : unitOf r.ins.read.draw (r.n h) / (partOf h r).n = (blocksOf rec).idxOf r.ρ :=
    (hdiv.trans (congrArg placeOf hir)).trans (placeOf_irootsOf hR.msg r.ρ)
  rcases hb.2 with ⟨hs, -⟩ | ⟨hs, hw, -, hnin⟩
  · -- a source bit
    have hsm : (i, templateOf h.ms r.u, rp) ∈ sourceBits h.ms := List.mem_of_find?_eq_some hs
    cases hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row with
    | some l =>
      have hfix : wireMap h i = i := wireMap_of_no_target h (wireTarget_of_linked hs hl)
      exact rowHolds_linked hS hr hR hb hfind hname hω hfix hl
    | none =>
      obtain ⟨y, hxy⟩ := wireTarget_of_sourcesNamed h W.named W.move hsm hl
      obtain ⟨tw, wport, u', hwu, hlw, hym, hret⟩ := wireTarget_at hs hxy
      obtain ⟨w, -, hwire, htw, hwp⟩ := wireUnit_wire hwu
      have htw' : heldWriter h w.writer = some tw := htw
      have hwp' : wport = w.wport := hwp
      rw [hwp'] at hlw hym
      have hmap : wireMap h i = y := wireMap_of_target h hxy
      have hbit : bitAtGate h (wireMap h i) = some (tw, Spot.Bits.RowPos.mk w.wport u' rp.bit) := by
        rw [hmap]
        simp only [bitAtGate, source_none_of_written htop hym, written_find W.once hym, Option.map_some]
      have hin := W.inside r hr i hi (by rw [hmap]; exact hret)
        (by rw [hmap]; exact written_not_input htop hym)
      obtain ⟨own0, reg0, hown0, hmem0, hn0⟩ := reg_of_named hr (Or.inl (wire_named h hwire))
      refine rowHolds_inside hS hr hR hb hω hbit hlw hin (fun reg hreg => ?_)
        (fun hn' => absurd hn' (regNamed_ne_none hname hown0 hmem0 hn0))
      obtain ⟨own, hown, hmem, hn⟩ := regNamed_spec hname hreg
      exact wired_reg hS hr hR hcp hfind hl hwire htw' hwu hown hmem hn
  · -- a written bit
    have hwm : (i, templateOf h.ms r.u, rp) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hw
    have hfix : wireMap h i = i := wireMap_of_not_source h hs
    cases hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row with
    | some l => exact rowHolds_linked hS hr hR hb hfind hname hω hfix hl
    | none =>
      have hret : returned h i = false := returned_of_whole W.whole hwm hl
      have huw : heldWire h (h.names.getD (templateOf h.ms r.u) "") rp.row = none :=
        unwired_of_written W.unwired hwm hl
      have hbit : bitAtGate h (wireMap h i) = some (templateOf h.ms r.u, rp) := by
        rw [hfix]
        exact bitAtGate_of_bitOf hb
      have hin := W.inside r hr i hi (by rw [hfix]; exact hret) (by rw [hfix]; exact hnin)
      have hq : 0 < (sampleOf h r).size * h.reg.ns.getD (templateOf h.ms r.u) 0 :=
        Nat.lt_of_le_of_lt (Nat.zero_le _) hvlt
      have hsplit : unitOf r.ins.read.draw (r.n h) =
          rp.leaf % h.reg.ns.getD (templateOf h.ms r.u) 0 +
            (blocksOf rec).idxOf r.ρ * h.reg.ns.getD (templateOf h.ms r.u) 0 :=
        split_arith (hmod.trans hcp.symm) hidx
      refine rowHolds_inside hS hr hR hb hω hbit hl hin (fun reg hreg => ?_) (fun hno => ?_)
      · obtain ⟨own, hown, hmem, hn⟩ := regNamed_spec hname hreg
        have hl' : heldLink h (h.names.getD (templateOf h.ms r.u) "") reg.name = none := by
          rw [hn]
          exact hl
        have huw' : heldWire h (h.names.getD (templateOf h.ms r.u) "") reg.name = none := by
          rw [hn]
          exact huw
        have hport := W.own r hr own reg hown hmem hl' huw' rec hR.msg
        have hleaves := hL r hr own reg hown hmem ⟨i, rp, hwm, hn.symm⟩ hl'
        refine ⟨hq, hleaves, fun tb positions paths htb hent => ?_⟩
        obtain ⟨ej, hv, hlv, hrd, hreads⟩ := entry_of_reg hr hR hown hmem
        have hrows := readOk_own_rows (own := ownOf h st rec draw r.c₀)
          (named := Spot.SecondStage.byTemplate rec) (p := partOf h r) (port := reg.name) (e := ej)
          (v := reg.value) hl' huw' hv hport hrd
        have hfind' : (List.range r.sts[0]!.c.ports.size).find?
            (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) =
              some (portNamed h (templateOf h.ms r.u) rp.row) := by
          rw [hn]
          exact hfind
        have hP := hreads _ _ (readAt_of_ok hS hr hown hmem hfind' htb hent)
        obtain ⟨hU, hPv⟩ := rowsOk_at hrows hlv hP
          (congrArg some (Nat.add_sub_cancel_left (partOf h r).lo (unitOf r.ins.read.draw (r.n h))))
        exact (ratio_one hPv hleaves hU).trans hsplit
      · obtain ⟨hc, hpub⟩ := hPub r hr i rp hwm hl hno
        refine ⟨hq, hc, fun pop hpop => ?_⟩
        obtain ⟨hlv, hrow⟩ := hpub pop hpop
        exact ⟨hlv, hrow.trans hsplit⟩

end CheckedRun

/-- **The row fact at the address map**, from the wiring facts of each accepted run and the three premises: at every
accepted run's checked inner block, the slot of each needed gate's address has the leaf count of what the gate's port
reads and opens the row the session reads. It is the `hRow` of `addressesConnected_of_runs`. -/
theorem rowFact_of_runs {h : Held} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h) (hS : AddressRow.Sits h)
    (hL : AddressRow.OwnLeaves h) (hPub : AddressRow.PublishedRows h) :
    RowFact h (wireMap h) (returned h) (addressOf h) :=
  fun r hr _ hi => CheckedRun.rowHolds_of_wiring (hW r hr) hS hL hPub hr hi

/-- **The row fact at the address map**, from the wiring facts and the three premises. -/
theorem rowFact_of_wiring {h : Held} (W : Wiring h) (hS : AddressRow.Sits h)
    (hL : AddressRow.OwnLeaves h) (hPub : AddressRow.PublishedRows h) :
    RowFact h (wireMap h) (returned h) (addressOf h) :=
  rowFact_of_runs (fun _ _ => W) hS hL hPub

end SpotFlock
