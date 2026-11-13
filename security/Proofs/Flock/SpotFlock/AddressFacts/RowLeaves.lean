module

public import Proofs.Flock.SpotFlock.AddressFacts.RowOf

/-!
# The leaf count of a registered read at a template's own written port

`AddressRow.OwnLeaves h` is the second of the row fact's premises (`RowOf`): at an accepted run, a registered read of
the checked inner block's session at a port its template writes and no link names has one leaf an inner block, the
sample's size times the template's count. This file proves it, `ownLeaves_of_wires`, from the wiring facts of each
accepted run and one condition decided on the auditor's copies of the wires (`wiresChecked`).

The second stage's check of the writer's own session does not give the count. At a template's own port
`SecondStage.readOk` holds the read's root, domain and leaf count to the first stage's record's entry of the port
(`sameAt`) and each slot to its own first row (`rowsOk`), in a tree of any whole number of rows a slot. What pins the
count is the check of a session that reads the port through a wire:

* the wiring check asks every written port no link names to be some wire's writer port (`writtenRead`);
* the wire's reader is a template (`SecondStage.wiresOk`, `reader_of_wire`), whose session reads the wire's port
  registered (`SecondStage.wiredRead`, `wired_entry_of_stmtOk`);
* at that read the second stage takes the first wire at the reader's port where no link names it
  (`SecondStage.readOk`'s second arm), and holds the read to the first stage's record's entry of that wire's writer
  port, of the sample's size times the writer's count of leaves (`readOk_wired_find`, `placedOk_leaves`);
* both sessions' reads are held to one entry of the first stage's record, the first of the port's name, so the
  writer's read has that leaf count too (`sameAt_leaves`).

The third step is where `wiresChecked` is read: no link names the wire's reader port, and the first wire at that port
names the same writer port. Without it a written port could be the writer port only of wires whose reader ports a link
names, or that an earlier wire at the same reader port shadows, and no session's check would pin its leaf count.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable {h : Held}

/-- **Each wire is one the second stage's check reads**, decided on the auditor's copies: no link names the wire's
reader port, and the first wire at that port (`heldWire`, the one `SecondStage.readOk` takes) names the wire's writer
port. The wires the design derives are one a reader port (`SecondStage.wireRuns`, through `gather`), so the second
clause holds of them (`RowWires`); the first is not a fact of the derivation, which gathers the links and the wires
apart: it fails where a derived link and a derived wire name one port of one template. -/
def wiresChecked (h : Held) : Bool :=
  (h.wires.getD #[]).all fun w =>
    (heldLink h w.template w.port).isNone &&
      (heldWire h w.template w.port).any fun w' => w'.writer == w.writer && w'.wport == w.wport

/-! ## Leaf counts the second stage holds to one entry -/

/-- A value the second stage reads as the natural number `n` is the number `n`. -/
theorem num_of_nat {a : Json} {n : ℕ} (hn : Spot.nat? a = some n) : a = .num ⟨Int.ofNat n, 0⟩ := by
  cases a with
  | num m =>
    obtain ⟨i, e⟩ := m
    cases e with
    | succ e => simp [Spot.nat?, Spot.int?] at hn
    | zero =>
      cases i with
      | ofNat k =>
        have hk : k = n := by simpa [Spot.nat?, Spot.int?] using hn
        subst hk
        rfl
      | negSucc k =>
        have hneg : ¬ (0 : Int) ≤ Int.negSucc k := Int.not_le.2 (Int.negSucc_lt_zero k)
        simp [Spot.nat?, Spot.int?, hneg] at hn
  | _ => simp [Spot.nat?, Spot.int?] at hn

/-- **Entries the second stage holds to one another have one leaf count** (`SecondStage.sameAt`, its `leaves`
clause). -/
theorem sameAt_leaves {e x : Json} {n : ℕ} (hs : Spot.SecondStage.sameAt e x = true)
    (hn : Spot.nat? (Spot.getD e "leaves") = some n) : Spot.nat? (Spot.getD x "leaves") = some n := by
  unfold Spot.SecondStage.sameAt at hs
  simp only [Bool.and_eq_true] at hs
  have hj : Spot.jeq (Spot.getD e "leaves") (Spot.getD x "leaves") = true := hs.1.2
  rw [num_of_nat hn] at hj
  exact nat?_of_jeq hj

/-! ## The second stage's two arms, at the entry they find -/

/-- **The wired arm of `SecondStage.readOk`, at the entry of the first stage's record it finds**: at a port a wire
names and no link does, a read the check passes is held to the first entry of the wire's writer port's name, with
the writer's count the rows' `per` and the read's leaf count the sample's size times it (`readOk_wired`, which
keeps only that the entry is one of the record's). -/
theorem readOk_wired_find {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} {w : Spot.SecondStage.Wire} (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = some w)
    (hok : Spot.SecondStage.readOk own named p port e = true) :
    ∃ kx pw r,
      (Spot.Record.byPort own.interior).find? (·.1 == Spot.SecondStage.portOf named w.writer w.wport) = some kx ∧
      Spot.SecondStage.writerOf own w = some pw ∧ Spot.SecondStage.rowsOf own p.template port = some r ∧
      pw.n = r.per ∧ Spot.SecondStage.sameAt e kx.2 = true ∧
      Spot.SecondStage.placedOk own.sample.size p e (own.sample.size * r.per)
        (Spot.SecondStage.wiredRow p r) = true := by
  unfold Spot.SecondStage.readOk at hok
  rw [hl, hw] at hok
  cases hv : Spot.getD e "value" with
  | str v =>
    rw [hv] at hok
    simp only [Bool.and_eq_true, beq_iff_eq] at hok
    obtain ⟨rfl, hok⟩ := hok
    cases hf : (Spot.Record.byPort own.interior).find?
        (·.1 == Spot.SecondStage.portOf named w.writer w.wport) with
    | none => rw [hf] at hok; simp at hok
    | some kx =>
      obtain ⟨k, x⟩ := kx
      cases hp : Spot.SecondStage.writerOf own w with
      | none => rw [hf, hp] at hok; simp at hok
      | some pw =>
        cases hr : Spot.SecondStage.rowsOf own p.template port with
        | none => rw [hf, hp, hr] at hok; simp at hok
        | some r =>
          rw [hf, hp, hr] at hok
          simp only [Bool.and_eq_true, beq_iff_eq] at hok
          exact ⟨(k, x), pw, r, rfl, rfl, rfl, hok.1.2, hok.1.1, hok.2⟩
  | _ => rw [hv] at hok; simp at hok

/-- **The last arm of `SecondStage.readOk`, at the entry of the first stage's record it finds**: at a port no link
or wire names, a read of a port of the first stage's record is of the template's own port and is held to the first
entry of that name (`readOk_own` reads the name, `readOk_own_rows` the rows). -/
theorem readOk_own_find {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} {v : String} (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = none) (hv : Spot.getD e "value" = .str v)
    (hp : ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome = true)
    (hok : Spot.SecondStage.readOk own named p port e = true) :
    ∃ kx, (Spot.Record.byPort own.interior).find? (·.1 == v) = some kx ∧
      v = Spot.SecondStage.portOf named p.template port ∧ Spot.SecondStage.sameAt e kx.2 = true := by
  unfold Spot.SecondStage.readOk at hok
  rw [hv, hl, hw] at hok
  cases hf : (Spot.Record.byPort own.interior).find? (·.1 == v) with
  | none =>
    rw [hf] at hp
    cases hp
  | some kx =>
    obtain ⟨k, x⟩ := kx
    simp only [hf, Bool.and_eq_true, beq_iff_eq] at hok
    exact ⟨(k, x), rfl, hok.1.1, hok.1.2⟩

/-! ## The session that reads a wire -/

/-- The design's check holds each wire's reader to a part of its name (`SecondStage.wiresOk`, its first clause). -/
theorem wiresOk_reader {ps : Array Spot.SecondStage.Part} {own : Spot.SecondStage.Own}
    (hok : Spot.SecondStage.wiresOk ps own = true) {w : Spot.SecondStage.Wire} (hw : w ∈ own.wires.getD #[]) :
    ∃ p ∈ ps, (p.template == w.template) = true := by
  unfold Spot.SecondStage.wiresOk at hok
  cases hws : own.wires with
  | none =>
    rw [hws] at hok
    cases hok
  | some ws =>
    simp only [hws, Option.getD_some] at hok hw
    have h1 := Array.all_eq_true_iff_forall_mem.1 hok w hw
    simp only [Bool.and_eq_true] at h1
    obtain ⟨p, hp, hpw⟩ := Array.any_eq_true'.1 h1.1.1.1
    exact ⟨p, hp, (Bool.and_eq_true_iff.1 hpw).1⟩

/-- **The session of a wire's reader reads the wire's port registered, and the second stage checks that read**: at
session statements the verifier's check passes, template `t`'s statement has an entry of its registered reads at
the port of each wire that names `t` as its reader (`SecondStage.wiredRead`), which passes `SecondStage.readOk`
against the template's part. -/
theorem wired_entry_of_stmtOk {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) {w : Spot.SecondStage.Wire}
    (hw : w ∈ h.wires.getD #[]) (hwt : w.template = h.names.getD t "") :
    ∃ st rec smp e, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some smp ∧
      Spot.SecondStage.readOk (ownOf h st rec smp c₀) (Spot.SecondStage.byTemplate rec)
        (partAt h (ownOf h st rec smp c₀).sample.size t) w.port e = true := by
  have hat := stmtOkAt_of h hok ht
  obtain ⟨s, hx⟩ := parse_of_stmtOkAt h hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, smp, wr, S, hst, hrec, hsmp, -, -, hc⟩ := readsOk_true h hat.2
  obtain ⟨-, -, -, -, -, -, hreads, -, hwired, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  unfold Spot.SecondStage.wiredRead at hwired
  have h1 : (w.template != h.names.getD t "" ||
      (Spot.entries (Spot.getD S.registered "ports")).any (·.1 == w.port)) = true :=
    Array.all_eq_true_iff_forall_mem.1 hwired w hw
  have hany : (Spot.entries (Spot.getD S.registered "ports")).any (·.1 == w.port) = true := by
    rcases Bool.or_eq_true_iff.1 h1 with hne | hany
    · exact absurd hwt (bne_iff_ne.1 hne)
    · exact hany
  obtain ⟨⟨nm, e⟩, hme, hnm⟩ := Array.any_eq_true'.1 hany
  have hnm' : nm = w.port := by simpa using hnm
  have hrd : Spot.SecondStage.readOk (ownOf h st rec smp c₀) (Spot.SecondStage.byTemplate rec)
      (partAt h (ownOf h st rec smp c₀).sample.size t) nm e = true :=
    Array.all_eq_true_iff_forall_mem.1 hreads _ hme
  rw [hnm'] at hrd
  exact ⟨st, rec, smp, e, hst, hrec, hsmp, hrd⟩

end AddressRow

namespace CheckedRun

open AddressRow

variable {h : Held} {r : CheckedRun h.stage}

/-- **A wire's reader is a template**, at an accepted run: the design's check holds the wire's reader to a part of
its name (`wiresOk_reader`), and the parts are the templates' in order. -/
theorem reader_of_wire (hr : r.Ok h) {w : Spot.SecondStage.Wire} (hw : w ∈ h.wires.getD #[]) :
    ∃ t, t < h.ms.length ∧ h.names.getD t "" = w.template := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb0, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, -, -, -, hdes⟩ := design_of_readsOk hat.2
  simp only [Spot.SecondStage.designOk, Bool.and_eq_true] at hdes
  obtain ⟨p, hp, hpt⟩ := wiresOk_reader hdes.2 hw
  have hp' : p ∈ (List.range h.ms.length).map (partAt h (ownOf h st rec draw r.c₀).sample.size) :=
    Array.mem_toList_iff.2 hp
  obtain ⟨t, htm, hpe⟩ := List.mem_map.1 hp'
  subst hpe
  exact ⟨t, List.mem_range.1 htm, eq_of_beq hpt⟩

/-- **A registered read at a template's own written port has one leaf an inner block**, at an accepted run whose
wires the second stage's check reads (`wiresChecked`): the read named like the row of a written bit of the inner
block's template, at a port no link names, has the sample's size times the template's count of leaves. A wire's
reader's session pins the count of the first stage's record's entry of the port, and the writer's own read is held
to the same entry. -/
theorem own_leaves (W : Wiring h) (hwc : wiresChecked h = true) (hr : r.Ok h)
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : r.ins.registered = some own) (hmem : reg ∈ own) {y : ℕ} {rp : Spot.Bits.RowPos}
    (hy : (y, templateOf h.ms r.u, rp) ∈ writtenBits h.ms) (hrow : rp.row = reg.name)
    (hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") reg.name = none) :
    reg.leaves = (sampleOf h r).size * h.reg.ns.getD (templateOf h.ms r.u) 0 := by
  obtain ⟨st, rec, draw, hR⟩ := reads_of_ok hr
  obtain ⟨d, h2⟩ := hR.design
  obtain ⟨-, -, -, -, -, -, hrows, -⟩ := wiring_of_two h h2
  have hwr : writtenRead h = true := writtenRead_of_rows h hrows
  have hok : stmtOk h r.x r.m₁ r.m₂ r.c₀ r.c₁ = true := by
    obtain ⟨Pt, d0, spec, hx, hP, h20, hin, hok, hrest⟩ := hr
    exact hok
  -- the wire whose writer port the written port is
  unfold writtenRead at hwr
  have h1 : ((heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row).isSome ||
      (h.wires.getD #[]).any fun w =>
        w.writer == h.names.getD (templateOf h.ms r.u) "" && w.wport == rp.row) = true :=
    List.all_eq_true.1 hwr (y, templateOf h.ms r.u, rp) hy
  rw [hrow, hl] at h1
  obtain ⟨w, hwm, hwp⟩ := Array.any_eq_true'.1 ((Bool.or_eq_true_iff.1 h1).resolve_left (by simp))
  obtain ⟨hw1, hw2⟩ := Bool.and_eq_true_iff.1 hwp
  have hwwr : w.writer = h.names.getD (templateOf h.ms r.u) "" := eq_of_beq hw1
  have hwwp : w.wport = reg.name := eq_of_beq hw2
  -- the wire the second stage takes at its reader port
  unfold wiresChecked at hwc
  have h5 : ((heldLink h w.template w.port).isNone &&
      (heldWire h w.template w.port).any fun w' => w'.writer == w.writer && w'.wport == w.wport) = true :=
    Array.all_eq_true_iff_forall_mem.1 hwc w hwm
  obtain ⟨hlk, hfirst⟩ := Bool.and_eq_true_iff.1 h5
  have hlk' : heldLink h w.template w.port = none := Option.isNone_iff_eq_none.1 hlk
  cases hw' : heldWire h w.template w.port with
  | none => simp [hw'] at hfirst
  | some w' =>
    simp only [hw', Option.any_some, Bool.and_eq_true, beq_iff_eq] at hfirst
    obtain ⟨hw'w, hw'p⟩ := hfirst
    -- the reader's session's read of the wire's port
    obtain ⟨t', ht', hnt'⟩ := reader_of_wire hr hwm
    obtain ⟨st', rec', draw', e', hst', hrec', hdraw', hrd'⟩ := wired_entry_of_stmtOk hok ht' hwm hnt'.symm
    have e1 : st' = st := Option.some.inj (hst'.symm.trans hR.dec)
    have e2 : rec' = rec := Option.some.inj (hrec'.symm.trans hR.msg)
    have e3 : draw' = draw := Option.some.inj (hdraw'.symm.trans hR.sample)
    rw [e1, e2, e3] at hrd'
    have hlr : heldLink h (h.names.getD t' "") w.port = none := by
      rw [hnt']
      exact hlk'
    have hwr' : heldWire h (h.names.getD t' "") w.port = some w' := by
      rw [hnt']
      exact hw'
    obtain ⟨kx, pw, rr, hfind, hpw, -, hpn, hsame, hpl⟩ :=
      readOk_wired_find (own := ownOf h st rec draw r.c₀) (named := Spot.SecondStage.byTemplate rec)
        (p := partAt h (ownOf h st rec draw r.c₀).sample.size t') (port := w.port) (e := e') (w := w') hlr hwr'
        hrd'
    rw [writerOf_ownOf] at hpw
    obtain ⟨tw, htw, hpwe⟩ := Option.map_eq_some_iff.1 hpw
    have hN : h.reg.ns.getD tw 0 = rr.per := (congrArg Spot.SecondStage.Part.n hpwe).trans hpn
    -- the wire's writer is the inner block's template
    have htw' : (List.range h.ms.length).find? (fun t => h.names.getD t "" == w'.writer) = some tw := htw
    have htwlt : tw < h.ms.length := List.mem_range.1 (List.mem_of_find?_eq_some htw')
    have h0 := List.find?_some htw'
    have h0' : (h.names.getD tw "" == w'.writer) = true := h0
    have htwn : h.names.getD tw "" = w'.writer := eq_of_beq h0'
    have he : tw = templateOf h.ms r.u :=
      List.inj_on_of_nodup_map (names_nodup_of_two h h2) (List.mem_range.2 htwlt) (List.mem_range.2 hR.lt)
        (htwn.trans (hw'w.trans hwwr))
    have hx1 : Spot.nat? (Spot.getD kx.2 "leaves") = some ((ownOf h st rec draw r.c₀).sample.size * rr.per) :=
      sameAt_leaves hsame (placedOk_leaves hpl)
    -- the writer's own read, held to the same entry
    have hl0 : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = none := by
      rw [hrow]
      exact hl
    have huw0 := unwired_of_written W.unwired hy hl0
    have huw : heldWire h (h.names.getD (templateOf h.ms r.u) "") reg.name = none := by
      rw [← hrow]
      exact huw0
    have hport := W.own r hr own reg hown hmem hl huw rec hR.msg
    obtain ⟨ej, hv, hlv, hrd, -⟩ := entry_of_reg hr hR hown hmem
    obtain ⟨kx2, hfind2, hval, hsame2⟩ := readOk_own_find (own := ownOf h st rec draw r.c₀)
      (named := Spot.SecondStage.byTemplate rec) (p := partOf h r) (port := reg.name) (e := ej) (v := reg.value)
      hl huw hv hport hrd
    have hkey : Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) w'.writer w'.wport = reg.value := by
      rw [hw'w, hw'p, hwwr, hwwp]
      exact hval.symm
    rw [hkey] at hfind
    have hkx : kx = kx2 := Option.some.inj (hfind.symm.trans hfind2)
    rw [hkx] at hx1
    have hx2 : Spot.nat? (Spot.getD kx2.2 "leaves") = some reg.leaves := sameAt_leaves hsame2 hlv
    have hLe : reg.leaves = (ownOf h st rec draw r.c₀).sample.size * rr.per :=
      Option.some.inj (hx2.symm.trans hx1)
    exact hLe.trans (congrArg (fun n => (sampleOf h r).size * n)
      (hN.symm.trans (congrArg (fun t => h.reg.ns.getD t 0) he)))

end CheckedRun

/-- **The second premise of the row fact**, from the wiring facts of each accepted run and the wires the second
stage's check reads (`wiresChecked`). -/
theorem AddressRow.ownLeaves_of_wires {h : Held} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hwc : AddressRow.wiresChecked h = true) : AddressRow.OwnLeaves h :=
  fun r hr _ _ hown hmem ⟨_, _, hy, hrow⟩ hl => CheckedRun.own_leaves (hW r hr) hwc hr hown hmem hy hrow hl

/-- **The row fact at the address map**, from the wiring facts of each accepted run, the wires the second stage's
check reads, and the two premises left: where the second stage puts a checked inner block (`Sits`) and the published
rows at a written port with no registered read (`PublishedRows`). -/
theorem rowFact_of_wires {h : Held} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h) (hS : AddressRow.Sits h)
    (hwc : AddressRow.wiresChecked h = true) (hPub : AddressRow.PublishedRows h) :
    RowFact h (wireMap h) (returned h) (addressOf h) :=
  rowFact_of_runs hW hS (AddressRow.ownLeaves_of_wires hW hwc) hPub

end SpotFlock
