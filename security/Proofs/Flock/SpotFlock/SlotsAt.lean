module

public import Proofs.Flock.SpotFlock.RevealAt

/-!
# `SlotsAt` at the base frame, from where each gate's value sits

`slotsAt_slotMap`: where every gate's value has one address that holds at every accepted run's checked unit that needs
it (`Addresses h`), `SlotsAt` holds at the slot map `slotMap h` (each gate's address's slot), the base frame
`revealCode h (addressMap h) c₀` and the layout `layoutFor h`, for every `c₀`. `slotsAt_of` is the same at any address
map whose addresses hold. None of `slotMap`, `revealCode` and `layoutFor` reads `h.frame` (`slotMap_frame`,
`revealCode_frame`, `layoutFor_frame`).

At every accepted run's checked unit the layout puts the unit in a block of its session's first table
(`CheckedRun.blockOf_some`) and each gate at a column of that table (`CheckedRun.wire_lt`, `colAt_lt`).

**An address holds at a run** (`CheckedRun.AddressHolds`): three facts about gate `g`'s address at one checked unit of
an accepted run, at the port the address names in that unit's session (`Address.port`).

* `Counted`: the unit's instance is below the table's count.
* `RowAt`: the address's place is as wide as its port's row in that table, and the bit the row holds at the place, at
  the unit's instance, is the one the layout reads.
* the branch facts, each giving the slot's row as the frame reads it off the gate's roots (`leafAt`, `unitsAt`): at a
  port a registered read is named like (`ReadsRegistered`), the port is the first of its name, the gate's roots are
  those round 2 reads the value under (`readRoots`), and the address's slot names the entry round 2 reads the value off
  (`readKeyAt`), at the read's schema, its leaf the session header's position for the row; at a one-table session's
  other ports (`ReadsInterior`), the gate is one its replay block doesn't commit, and the slot names the template's port
  of the interior record, sized, at the published tree's schema, its units the tree's leaves and its leaf the drawn
  unit's row.

A slot reads its row off the roots, so one slot fits every replay sample: under a replay block's roots, a slot at unit
`p` with stride `n` reads unit `i·n + p` of a tree of `q·n` units, at the `i`-th of `q` sampled replay blocks
(`leafAt_irootsOf`), the second stage's population as the two-stage registration lays it out; under the run's roots it
reads its fixed leaf (`leafAt_decode`).

The row's format is not part of an address: every port of the session's first table is an unsegmented bit row
(`CheckedRun.v2Row_port`), since the pin holds the session's circuit to its template's reference statement's, whose
ports `templateOk` holds to `ProvedScope.v2Row`.

`Addresses h` is a premise, not a theorem, because the verifier's checks don't give it today, and at some registrations
it can't hold:

* the width (`RowAt`'s first fact): one place for every unit that needs the gate, so the ports that hold it in
  different sessions must have one width;
* the copy (`RowAt`'s last fact): the opened bit is a unit column and the row's bit a message column, equal only
  through the session circuit's copy rows, and the place must be one for every unit that needs the gate;
* the row (`Counted` and the slots' leaves): round 2 doesn't hold a session's population to the layout a slot reads
  off the roots, and bounds a registered read's row only to its unit's range of rows; and the unit's instance
  (`rankAt`) is a drawn unit's, which an always-checked unit needn't have;
* the entry (`ReadsRegistered`): round 2 holds a session's registered reads to its template's reference statement's but
  for the run's fields, and which entry it reads a value off (`readKeyAt`) still reads the round-1 record's port names;
* one tree a gate: a gate one unit writes and another reads is needed by both, and the frame checks both openings at
  the gate's one slot under the same roots, so both units' reads of its row must have one root. Round 2 reads a wired
  input at its writer's output port (`SecondStage.wireOf`), one commitment per value; a registration that commits a
  wired input as a tree of its own, under a domain of its own, has two.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

section Columns

open FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

/-- **Every column `colAt` gives at an accepted statement is one of its table's**: below the rows' width it is the
accepted statement's own column (`col_val`), and from the width on it is the pin, the last column. -/
theorem colAt_lt {T : Model.Statement} (a : HiddenExec.AcceptedH T) {gi : ℕ} (hg : gi < a.st.c.g) (x : ℕ) :
    colAt a.st a.L.unit_built gi x < 2 ^ T.kLog := by
  have hw : a.rows.w = (unitRows a.st a.L.unit_built).stackW a.st.c.upv := rfl
  have h0 : 0 < (unitRows a.st a.L.unit_built).stackW a.st.c.upv := Nat.succ_pos _
  by_cases hx : x < a.rows.w
  · have hc := (a.col hg ⟨x, hx⟩).isLt
    rw [col_val a hg ⟨x, hx⟩] at hc
    exact hc
  · have hl : a.rows.w - 1 < a.rows.w := by omega
    have e : colAt a.st a.L.unit_built gi x = colAt a.st a.L.unit_built gi (a.rows.w - 1) := by
      unfold colAt
      rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    have hc := (a.col hg ⟨_, hl⟩).isLt
    rw [col_val a hg ⟨_, hl⟩] at hc
    rw [e]
    exact hc

/-- **`layoutFor h` sends every gate to a column of a session statement's first table**, at a statement the scope
check accepts (`accepted_check`, `colAt_lt`). -/
theorem wire_lt_of {xS : Msg} {p : Parsed} (hp : parseBytes (FlockDeploy.readOf true) xS = some p) (hP : Parses p)
    (hc : p.check = .ok ()) (hh : p.tags.hiddenOutputs = true) (hj : 0 < p.sts.size) (x m₁ : Msg) (u G : ℕ) :
    (layoutFor h).wire x m₁ xS u G < 2 ^ p.sts[0].c.kLog := by
  obtain ⟨T, a, hst, hk, -, -⟩ := accepted_check hP hc hh hj
  have hfs : firstStmt xS = some a.st := by
    rw [hst]
    show (parseBytes (FlockDeploy.readOf true) xS).bind (·.sts[0]?) = _
    rw [hp]
    exact getElem?_pos p.sts 0 hj
  obtain ⟨hty, ⟨ls, hcc⟩, -⟩ := Hidden.Committed.check a.ht a.committed
  have hg0 : 0 < a.st.c.g := Hidden.check_g_pos hty hcc
  rw [← hk]
  classical
  show ((firstStmt xS).map fun st =>
    if hb : Built (lunit st.c) then
      colAt st hb (rankAt h x m₁ xS u % st.c.g)
        (pickAny h (templateOf h.ms u) (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) G)
    else 0).getD 0 < _
  rw [hfs]
  show (if hb : Built (lunit a.st.c) then
      colAt a.st hb (rankAt h x m₁ xS u % a.st.c.g)
        (pickAny h (templateOf h.ms u) (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) G)
    else 0) < _
  rw [dite_eq_left (show Built (lunit a.st.c) from a.L.unit_built)]
  exact colAt_lt a (Nat.mod_lt _ hg0) _

end Columns

namespace CheckedRun

variable (r : CheckedRun h.stage)

/-- **The unit's instance is one of its session's first table's instances.** -/
def Counted : Prop := r.n h < r.sts[0]!.pub.n

/-- **The address's port reads a registered value**: a registered read `reg` is named like the unit's port, it is the
first port of that name, the gate's roots are those round 2 reads `reg`'s value under, and the address's slot names the
entry round 2 reads the value off, at `reg`'s schema, not sized; under those roots its leaf (`leafAt`), at an entry of
`reg`'s leaf count, is the position the session header gives the instance's row. -/
def ReadsRegistered (S : Address) (g : ℕ) : Prop :=
  ∃ reg, regNamed r.ins r.sts (S.port r.u) = some reg ∧
    (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some (S.port r.u) ∧
    r.roots h g = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value ∧
    S.slot.key = readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value ∧
    S.slot.schema = reg.schema ∧ S.slot.sized = false ∧
    ∀ t positions paths, r.sts[0]!.pub.tables = some t →
      _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
      ∀ e, entryAt (r.roots h g) S.slot = some e → e.leaves = reg.leaves →
        leafAt (r.roots h g) S.slot e =
          positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size (S.port r.u) (r.n h)) 0

/-- **The address's port reads the interior record**: no registered read is named like the unit's port, the session has
one table, the gate is one the unit's replay block doesn't commit, and the address's slot names the template's port of
the interior record, sized; over any population the session's public file loads, the slot has the published tree's
schema, and under the replay block's roots its units are the tree's leaves and its leaf (`leafAt`) the drawn unit's
row. -/
def ReadsInterior (S : Address) (g : ℕ) : Prop :=
  regNamed r.ins r.sts (S.port r.u) = none ∧ r.ins.read.I.count ≤ 1 ∧ S.port r.u < r.sts[0]!.c.ports.size ∧
    (g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false) ∧
    S.slot.key = .interior (h.names.getD (templateOf h.ms r.u) "") (r.sts[0]!.c.ports[S.port r.u]!).name ∧
    S.slot.sized = true ∧ ∀ pop,
    _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
    S.slot.schema = (pubPort r.sts[0]!.c pop (S.port r.u)).schema ∧
    unitsAt (iroots r.x r.seen r.ρ) S.slot = (pubPort r.sts[0]!.c pop (S.port r.u)).leaves ∧
    ∀ e, entryAt (iroots r.x r.seen r.ρ) S.slot = some e →
      leafAt (iroots r.x r.seen r.ρ) S.slot e = rowAt r.sts[0]!.c pop (S.port r.u) (unitOf r.ins.read.draw (r.n h))

/-- **Gate `g`'s value sits at address `S`** in the run (the module's docstring). -/
def AddressHolds (S : Address) (g : ℕ) : Prop :=
  r.Counted h ∧ r.RowAt h S g ∧ (r.ReadsRegistered h S g ∨ r.ReadsInterior h S g)

end CheckedRun

/-- **Each gate's slot**: its address's. -/
def slotsOf (addr : ℕ → Option Address) (g : ℕ) : Option Slot := (addr g).map Address.slot

namespace CheckedRun

variable {h} {r : CheckedRun h.stage}

/-- **An accepted run's checked unit's session has a table** (`zkOk_sound`). -/
theorem sts_pos (hr : r.Ok h) : 0 < r.sts.size := by
  obtain ⟨_, _, _, hx, -, -, -, hok, hu, -, -, -, -, hset⟩ := hr
  have hs := sessionOk_of_stmtOkAt h (stmtOkAt_unit h hx hok hu)
  simp only [sessionOk, Bool.and_eq_true] at hs
  obtain ⟨ins, spec, sts, hset', hpos, -⟩ := zkOk_sound hs.1.1.1
  rw [hset] at hset'
  cases hset'
  exact hpos

/-- **The layout puts an accepted run's checked unit in a block of its session's first table**: the session statement
parses (`Claims.Flock.check`), to the setup's tables (`parsed_setupOf`). -/
theorem blockOf_some (hr : r.Ok h) : (layoutFor h).blockOf [] r.m₁ (r.xS h) r.u = some (r.β h) := by
  have hj := sts_pos hr
  obtain ⟨_, _, _, -, -, -, -, -, -, hck, -, -, -, hset⟩ := hr
  have hfs : firstStmt (unitStmt h r.m₂ r.u) = some r.sts[0] := by
    cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
    | none => simp [Claims.Flock.check, hS] at hck
    | some S =>
      unfold FlockDeploy.parse _root_.Flock.Readable.parseOf at hS
      obtain ⟨p, hp, -⟩ := Option.map_eq_some_iff.1 hS
      obtain ⟨rfl, -⟩ := parsed_setupOf hset hp
      show (_root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) (unitStmt h r.m₂ r.u)).bind (·.sts[0]?) = _
      rw [hp]
      exact getElem?_pos r.sts 0 hj
  have hv : (layoutFor h).blockOf [] r.m₁ (unitStmt h r.m₂ r.u) r.u =
      some (rankAt h [] r.m₁ (unitStmt h r.m₂ r.u) r.u / r.sts[0].c.g % 2 ^ (r.sts[0].m - r.sts[0].c.kLog)) := by
    show (firstStmt (unitStmt h r.m₂ r.u)).map _ = _
    rw [hfs]
    rfl
  show (layoutFor h).blockOf [] r.m₁ (unitStmt h r.m₂ r.u) r.u =
    some (((layoutFor h).blockOf [] r.m₁ (unitStmt h r.m₂ r.u) r.u).getD 0)
  rw [hv]
  rfl

/-- **The layout's column for each gate of an accepted run's checked unit is one of its session's first table**: the
session statement passes the scope check (`parsed_of_stmtOkAt`, `wire_lt_of`). -/
theorem wire_lt (hr : r.Ok h) (g : ℕ) : r.wire h g < 2 ^ r.sts[0]!.c.kLog := by
  have hj := sts_pos hr
  obtain ⟨_, _, _, hx, -, -, -, hok, hu, hck, -, -, -, hset⟩ := hr
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨p, hpb, -, hpp, hc, hh⟩ := parsed_of_stmtOkAt h (stmtOkAt_unit h hx hok hu) hS
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hpb
    rw [getElem!_pos r.sts 0 hj]
    exact wire_lt_of h hpb hpp hc hh hj [] r.m₁ r.u g

/-- **Every port of an accepted run's checked unit's session's first table is an unsegmented bit row**
(`ProvedScope.v2Row`): the table's circuit is the one the scope check parses (`accepted_check`), which unrolls, ports
and all (`unrolled_ports`), to its template's reference statement's statement circuit (`pin_of_stmtOkAt`,
`stmt_of_pin`), each of whose ports `templateOk` holds to `v2Row`. -/
theorem v2Row_port (hr : r.Ok h) {q : ℕ} (hq : q < r.sts[0]!.c.ports.size) :
    _root_.Flock.ProvedScope.v2Row r.sts[0]!.c.ports[q]! = true := by
  have hj := sts_pos hr
  obtain ⟨Pt, d, _, hx, hP, h2, hin, hok, hu, hck, hb, hρ, -, hset⟩ := hr
  have hokAt := stmtOkAt_unit h hx hok hu
  have hT := fun t (ht : t < h.ms.length) => templateOk_facts h (templateOk_of_two h h2 ht) ht
  obtain ⟨t, ht, k, hut, -, -⟩ :=
    shape_of_checked h (fun t ht => ⟨(hT t ht).2.1, (hT t ht).2.2⟩) hx hP h2 hin hb hρ
  have htu : templateOf h.ms r.u = t := by
    rw [hut]
    exact templateOf_unitBase ht k.isLt
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨p, hpb, -, hpp, hc, hh⟩ := parsed_of_stmtOkAt h hokAt hS
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hpb
    obtain ⟨w, w₀, hw, hw₀, he⟩ := pin_of_stmtOkAt h hokAt
    rw [htu] at hw₀
    have hok₀ := templateOk_of_two h h2 ht
    unfold templateOk at hok₀
    cases hr₀ : FlockDeploy.readOf true (h.xS₀ t) with
    | none => rw [hr₀] at hok₀; cases hok₀
    | some r₀ =>
      rw [hr₀] at hok₀
      simp only [Bool.and_eq_true] at hok₀
      have hst := stmt_of_pin hw hw₀ he (readOf_of_parseBytes hpb) hr₀
      obtain ⟨T, a, hst0, -, hcirc, -⟩ := accepted_check hpp hc hh hj
      have hports : r.sts[0]!.c.ports = r₀.stmt.ports :=
        calc r.sts[0]!.c.ports = a.st.c.ports := by rw [(getElem!_pos r.sts 0 hj).trans hst0.symm]
          _ = (_root_.Flock.HmOut.unrolled a.st.c).ports := (FlockSoundness.unrolled_ports a.st.c).symm
          _ = r₀.stmt.ports := by
            rw [← hcirc]
            exact congrArg _root_.Flock.Circuit.ports hst
      rw [hports] at hq ⊢
      rw [getElem!_pos r₀.stmt.ports q hq]
      exact Array.all_eq_true.1 hok₀.2 q hq

/-- **An unsegmented bit row's format**: `v2`, with no segments. -/
theorem v2_of_v2Row {p : _root_.Flock.Port} (hp : _root_.Flock.ProvedScope.v2Row p = true) :
    p.v2 = true ∧ p.segs = none := by
  unfold _root_.Flock.ProvedScope.v2Row at hp
  cases hv : p.v2 <;> cases hs : p.segs <;> simp_all

/-- **`SlotsAtGate` at an address that holds**: at a checked unit of an accepted run, at a gate it needs whose address
holds there, the code's opening is the address's (`openAt_revealCode`): its value is the witness's bit at the layout's
column, its row and salt are the address's port's at the unit's instance (`row_opening`, `salt_opening`), and its path
is `pathAt`'s (`path_opening`). -/
theorem slotsAtGate (hr : r.Ok h) {addr : ℕ → Option Address} (c₀ : Code) {g : ℕ} {S : Address} (hsS : addr g = some S)
    (hg : r.s.prog.Needs r.b g) (hH : r.AddressHolds h S g) :
    SlotsAtGate h (slotsOf addr) (revealCode h addr c₀) (layoutFor h) r.x r.s r.m₀ r.m₁ r.m₂ r.c₀ r.c₁ r.rest r.wF r.u
      r.ρ r.ins r.sts g := by
  obtain ⟨hn, hR, hrd⟩ := hH
  have hj := sts_pos hr
  have e0 : r.sts[0]! = r.sts[0] := getElem!_pos r.sts 0 hj
  have hsl : slotsOf addr g = some S.slot := by simp [slotsOf, hsS]
  have hq : S.port r.u < r.sts[0]!.c.ports.size := by
    rcases hrd with ⟨reg, -, hfind, -⟩ | ⟨-, -, hq, -⟩
    · exact List.mem_range.1 (List.mem_of_find?_eq_some hfind)
    · exact hq
  obtain ⟨hv2, hseg⟩ := v2_of_v2Row (v2Row_port hr hq)
  refine ⟨r.β h, 0, r.wire h g, r.n h, S.port r.u, blockOf_some hr, hj, by rw [← e0]; exact wire_lt hr g,
    by rw [Nat.zero_mul, Nat.zero_add]; rfl, by rw [← e0]; exact hn, ?_⟩
  intro o
  have ho : o = r.opening h S g := openAt_revealCode hr hsS hg c₀
  rw [ho]
  refine ⟨rfl, row_opening hsS hv2 hseg hR, salt_opening addr c₀ S g, ?_⟩
  rw [← e0]
  rcases hrd with ⟨reg, hreg, hfind, hroots, hkey, hsch, hsz, hleaf⟩ | ⟨hnone, hc1, hq, hint, hkey, hsz, hI⟩
  · obtain ⟨own, hown, hfo⟩ := Option.bind_eq_some_iff.1 hreg
    refine Or.inl ⟨own, reg, hown, Array.mem_of_find?_eq_some hfo, hfind, hroots,
      fun t positions paths ht hent => ⟨S.slot, hsl, hkey, hsch, hsz, hleaf t positions paths ht hent, ?_⟩⟩
    rw [path_opening addr c₀ S g]
    simp only [pathAt, hreg, ht, hent]
  · have hR' : (r.opening h S g).1 = iroots r.x r.seen r.ρ := by
      rw [roots_opening hr S g, ite_eq_left hint]
    refine Or.inr ⟨hc1, hq, hR', fun own hown reg hreg => ?_, fun pop hpop => ?_⟩
    · have hnf : own.find? (fun reg => reg.name == (r.sts[0]!.c.ports[S.port r.u]!).name) = none := by
        simpa [regNamed, hown] using hnone
      simpa using Array.find?_eq_none.1 hnf reg hreg
    · obtain ⟨hsc, hun, hl⟩ := hI pop hpop
      refine ⟨S.slot, hsl, hkey, hsc, hsz, by rw [hR']; exact hun, fun e he => by rw [hR'] at he ⊢; exact hl e he, ?_⟩
      rw [path_opening addr c₀ S g]
      simp only [pathAt, hnone, hpop]

end CheckedRun

/-- **`SlotsAt` at any address map whose addresses hold**: where each gate a checked unit of an accepted run needs has
an address that holds there (`CheckedRun.slotsAtGate`). -/
theorem slotsAt_of {addr : ℕ → Option Address}
    (hS : ∀ r : CheckedRun h.stage, r.Ok h → ∀ g, r.s.prog.Needs r.b g → ∃ S, addr g = some S ∧ r.AddressHolds h S g)
    (c₀ : Code) : SlotsAt h (slotsOf addr) (revealCode h addr c₀) (layoutFor h) := by
  intro x s Pt d m₀ m₁ m₂ c₀' c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  have hr : CheckedRun.Ok h ⟨x, s, m₀, m₁, m₂, c₀', c₁, rest, wF, u, b, ρ, rb, ins, sts⟩ :=
    ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩
  obtain ⟨S, hsS, hH⟩ := hS _ hr g hg
  exact CheckedRun.slotsAtGate hr c₀ hsS hg hH

/-- **Every gate's value has one address**, fixed before the run, that holds at every accepted run's checked unit that
needs the gate (`CheckedRun.AddressHolds`). The premise `SlotsAt` rests on (the module's docstring). -/
def Addresses : Prop :=
  ∀ g, ∃ S : Address, ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.AddressHolds h S g

open Classical in
/-- **The address map**: each gate's address where it has one that holds at every run that needs it. -/
noncomputable def addressMap (g : ℕ) : Option Address :=
  if hx : ∃ S : Address, ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.AddressHolds h S g then
    some hx.choose
  else none

/-- **The slot map**: each gate's address's slot. -/
noncomputable def slotMap : ℕ → Option Slot := slotsOf (addressMap h)

theorem addressMap_spec (hS : Addresses h) (g : ℕ) :
    ∃ S, addressMap h g = some S ∧ ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.AddressHolds h S g :=
  ⟨(hS g).choose, dite_eq_left (hS g), (hS g).choose_spec⟩

/-- **`SlotsAt` at the slot map** (the module's docstring). -/
theorem slotsAt_slotMap (hS : Addresses h) (c₀ : Code) :
    SlotsAt h (slotMap h) (revealCode h (addressMap h) c₀) (layoutFor h) :=
  slotsAt_of h (fun r hr g hg => by
    obtain ⟨S, hs, hH⟩ := addressMap_spec h hS g
    exact ⟨S, hs, hH r hr hg⟩) c₀

theorem slotMap_frame (f : Code) : slotMap { h with frame := f } = slotMap h := rfl

theorem revealCode_frame (f : Code) (addr : ℕ → Option Address) (c₀ : Code) :
    revealCode { h with frame := f } addr c₀ = revealCode h addr c₀ := rfl

theorem layoutFor_frame (f : Code) : layoutFor { h with frame := f } = layoutFor h := rfl

end SpotFlock
