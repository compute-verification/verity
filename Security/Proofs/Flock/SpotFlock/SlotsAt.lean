module

public import Proofs.Flock.SpotFlock.RevealAt

/-!
# `SlotsAt` at the base frame, from where each gate's value sits

`slotsAt_slotMap`: where every gate's value has one site that holds at every accepted run's checked unit that needs it
(`Sites h`), `SlotsAt` holds at the slot map `slotMap h` (each gate's site's slot), the base frame
`revealCode h (siteMap h) c₀` and the layout `layoutFor h`, for every `c₀`. `slotsAt_of` is the same at any site map
whose sites hold. None of `slotMap`, `revealCode` and `layoutFor` reads `h.frame` (`slotMap_frame`,
`revealCode_frame`, `layoutFor_frame`).

At every accepted run's checked unit the layout puts the unit in a block of its session's first table
(`CheckedRun.blockOf_some`) and each gate at a column of that table (`CheckedRun.wire_lt`, `colAt_lt`).

**A site holds at a run** (`CheckedRun.SiteHolds`): three facts about gate `g`'s site at one checked unit of an
accepted run.

* `Counted`: the unit's instance is below the table's count.
* `RowAt`: the site's port of that table is an unsegmented bit row of its place's width, and the bit it holds at the
  place, at the unit's instance, is the one the layout reads.
* the branch facts: at a port a registered read is named like (`ReadsRegistered`), the port is the first of its name,
  the opening's roots are those round 2 reads the value under (`readRoots`), and the site's slot is the entry round 2
  reads the value off (`readKeyAt`) at the read's schema and the session header's position for the row; at a
  one-table session's other ports (`ReadsInterior`), the gate is one its replay block doesn't commit, and the slot is
  the template's port of the interior record at the published tree's schema and leaf count and the instance's row.

`Sites h` is a premise, not a theorem, because the verifier's checks don't give it today:

* the row format (`RowAt`'s first three facts): the scope admits `sha512/row/v1` and `sha512/row-seg/v1` ports, whose
  rows `SamplingDeploy.row` doesn't encode;
* the copy (`RowAt`'s last fact): the opened bit is a unit column and the row's bit a message column, equal only
  through the session circuit's copy rows, and the place must be one for every unit that needs the gate;
* the slot's position (`Counted` and the slots): an interior row's leaf is its unit's place in the second stage's
  population, `i·n_t + p` for the `i`-th sampled replay block, in a tree of `q·n_t` leaves, so the run's coins set
  both the position and the leaf count; and round 2 bounds a registered read's row only to its unit's range of rows;
* the registered value (`ReadsRegistered`): round 2 lets a port read any registered value, so its slot's key is fixed
  only where each port's registered value is the reference statement's.
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
      rw [if_neg (by omega), if_neg (by omega)]
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

variable (r : CheckedRun)

/-- **The unit's instance is one of its session's first table's instances.** -/
def Counted : Prop := r.n h < r.sts[0]!.pub.n

/-- **The site's port reads a registered value**: a registered read `reg` is named like it, it is the first port of
that name, the opening's roots are those round 2 reads `reg`'s value under, and the site's slot is the entry round 2
reads the value off, at `reg`'s schema and the position the session header gives the instance's row. -/
def ReadsRegistered (S : Site) (g : ℕ) : Prop :=
  ∃ reg, regNamed r.ins r.sts S.port = some reg ∧
    (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some S.port ∧
    rootsAt h r.s r.seen r.u g = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value ∧
    ∀ t positions paths, r.sts[0]!.pub.tables = some t →
      _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
      S.slot = ⟨readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value, reg.schema,
        positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size S.port (r.n h)) 0, none⟩

/-- **The site's port reads the interior record**: no registered read is named like it, the session has one table,
the gate is one the unit's replay block doesn't commit, and the site's slot is the template's port of the interior
record, at the published tree's schema and leaf count and the instance's row. -/
def ReadsInterior (S : Site) (g : ℕ) : Prop :=
  regNamed r.ins r.sts S.port = none ∧ r.ins.read.I.count ≤ 1 ∧ S.port < r.sts[0]!.c.ports.size ∧
    (g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false) ∧ ∀ pop,
    _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
    S.slot = ⟨.interior (h.names.getD (templateOf h.ms r.u) "") (r.sts[0]!.c.ports[S.port]!).name,
      (pubPort r.sts[0]!.c pop S.port).schema, rowAt r.sts[0]!.c pop S.port (unitOf r.ins.read.draw (r.n h)),
      some (pubPort r.sts[0]!.c pop S.port).leaves⟩

/-- **Gate `g`'s value sits at site `S`** in the run (the module's docstring). -/
def SiteHolds (S : Site) (g : ℕ) : Prop :=
  r.Counted h ∧ r.RowAt h S g ∧ (r.ReadsRegistered h S g ∨ r.ReadsInterior h S g)

/-- **`SlotsAt`'s conclusion** at gate `g` of the run's checked unit. -/
def Slots (slot : ℕ → Option Slot) (c : Code) (L : Layout) (g : ℕ) : Prop :=
  ∃ β j y n q, L.blockOf r.x r.m₁ (r.xS h) r.u = some β ∧ ∃ hj : j < r.sts.size,
    y < 2 ^ r.sts[j].c.kLog ∧ L.wire r.x r.m₁ (r.xS h) r.u g = j * 2 ^ r.sts[j].c.kLog + y ∧ n < r.sts[j].pub.n ∧
    ∃ o ∈ c.reveal r.s r.u r.seen r.wF, o.2.pos = g ∧
      o.2.val = _root_.Flock.InnerRows.bitAt r.p r.wF j β y ∧
      o.1 = (if g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false then iroots r.x r.seen r.ρ else r.s.roots) ∧
      SamplingDeploy.row c.place c.bits o.2 = _root_.Flock.InnerRows.rowBytes r.p r.wF j n q ∧
      c.salt o.2 = (_root_.Flock.InnerRows.salt r.p r.wF j n q).data.toList ∧
      ((∃ own reg, r.ins.registered = some own ∧ reg ∈ own ∧
        (List.range r.sts[j].c.ports.size).find? (fun q => r.sts[j].c.ports[q]!.name == reg.name) = some q ∧
        o.1 = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value ∧
        ∀ t positions paths, r.sts[j].pub.tables = some t →
          _root_.Flock.Registered.entry r.sts[j].pub reg = .ok (positions, paths) →
          slot g = some ⟨readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value, reg.schema,
            positions.getD (_root_.Flock.Registered.ref t r.sts[j].c.ports.size q n) 0, none⟩ ∧
          bytes (c.π o.2) = paths.getD (_root_.Flock.Registered.ref t r.sts[j].c.ports.size q n) ByteArray.empty) ∨
      (r.ins.read.I.count ≤ 1 ∧ q < r.sts[j].c.ports.size ∧ o.1 = iroots r.x r.seen r.ρ ∧
        (∀ own, r.ins.registered = some own → ∀ reg ∈ own, reg.name ≠ r.sts[j].c.ports[q]!.name) ∧ ∀ pop,
        _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[j].c r.ins.read.I.publicFile = .ok pop →
        slot g = some ⟨.interior (h.names.getD (templateOf h.ms r.u) "") r.sts[j].c.ports[q]!.name,
            (pubPort r.sts[j].c pop q).schema, rowAt r.sts[j].c pop q (unitOf r.ins.read.draw n),
            some (pubPort r.sts[j].c pop q).leaves⟩ ∧
          bytes (c.π o.2) = pubPath r.sts[j].c pop q (rowAt r.sts[j].c pop q (unitOf r.ins.read.draw n))))

end CheckedRun

/-- **Each gate's slot**: its site's. -/
def slotsOf (site : ℕ → Option Site) (g : ℕ) : Option Slot := (site g).map Site.slot

namespace CheckedRun

variable {h r}

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
    rw [hfs] <;> rfl
  show (layoutFor h).blockOf [] r.m₁ (unitStmt h r.m₂ r.u) r.u =
    some (((layoutFor h).blockOf [] r.m₁ (unitStmt h r.m₂ r.u) r.u).getD 0)
  rw [hv] <;> rfl

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

/-- **`SlotsAt`'s conclusion at a site that holds**: at the base frame over the site map, the layout `layoutFor h`
and the site map's slots. -/
theorem slots (hr : r.Ok h) {site : ℕ → Option Site} (c₀ : Code) {g : ℕ} {S : Site} (hsS : site g = some S)
    (hg : r.s.prog.Needs r.b g) (hH : r.SiteHolds h S g) :
    r.Slots h (slotsOf site) (revealCode h site c₀) (layoutFor h) g := by
  obtain ⟨hn, hR, hrd⟩ := hH
  have hj := sts_pos hr
  have e0 : r.sts[0]! = r.sts[0] := getElem!_pos r.sts 0 hj
  refine ⟨r.β h, 0, r.wire h g, r.n h, S.port, blockOf_some hr, hj, by rw [← e0]; exact wire_lt hr g,
    by rw [Nat.zero_mul, Nat.zero_add] <;> rfl, by rw [← e0]; exact hn,
    r.opening h S g, r.open_mem hr hsS hg, rfl, rfl, r.roots_opening hr S g, r.row_opening hsS hR,
    r.salt_opening site c₀ S g, ?_⟩
  rw [← e0]
  rcases hrd with ⟨reg, hreg, hfind, hroots, hslot⟩ | ⟨hnone, hc1, hq, hint, hslot⟩
  · obtain ⟨own, hown, hfo⟩ := Option.bind_eq_some_iff.1 hreg
    refine Or.inl ⟨own, reg, hown, Array.mem_of_find?_eq_some hfo, hfind, hroots,
      fun t positions paths ht hent => ⟨?_, ?_⟩⟩
    · show (site g).map Site.slot = _
      rw [hsS]
      exact congrArg some (hslot t positions paths ht hent)
    · rw [r.path_opening site c₀ S g]
      simp only [pathAt, hreg, ht, hent]
  · refine Or.inr ⟨hc1, hq, ?_, fun own hown reg hreg => ?_, fun pop hpop => ⟨?_, ?_⟩⟩
    · rw [r.roots_opening hr S g, if_pos hint]
    · have hnf : own.find? (fun reg => reg.name == (r.sts[0]!.c.ports[S.port]!).name) = none := by
        simpa [regNamed, hown] using hnone
      simpa using Array.find?_eq_none.1 hnf reg hreg
    · show (site g).map Site.slot = _
      rw [hsS]
      exact congrArg some (hslot pop hpop)
    · rw [r.path_opening site c₀ S g]
      simp only [pathAt, hnone, hpop]

end CheckedRun

/-- **`SlotsAt` at any site map whose sites hold**: where each gate a checked unit of an accepted run needs has a site
that holds there. -/
theorem slotsAt_of {site : ℕ → Option Site}
    (hS : ∀ r : CheckedRun, r.Ok h → ∀ g, r.s.prog.Needs r.b g → ∃ S, site g = some S ∧ r.SiteHolds h S g)
    (c₀ : Code) : SlotsAt h (slotsOf site) (revealCode h site c₀) (layoutFor h) := by
  intro x s Pt d m₀ m₁ m₂ c₀' c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  have hr : CheckedRun.Ok h ⟨x, s, m₀, m₁, m₂, c₀', c₁, rest, wF, u, b, ρ, rb, ins, sts⟩ :=
    ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩
  obtain ⟨S, hsS, hH⟩ := hS _ hr g hg
  exact CheckedRun.slots hr c₀ hsS hg hH

/-- **Every gate's value has one site**, fixed before the run, that holds at every accepted run's checked unit that
needs the gate (`CheckedRun.SiteHolds`). The premise `SlotsAt` rests on (the module's docstring). -/
def Sites : Prop :=
  ∀ g, ∃ S : Site, ∀ r : CheckedRun, r.Ok h → r.s.prog.Needs r.b g → r.SiteHolds h S g

open Classical in
/-- **The site map**: each gate's site where it has one that holds at every run that needs it. -/
noncomputable def siteMap (g : ℕ) : Option Site :=
  if hx : ∃ S : Site, ∀ r : CheckedRun, r.Ok h → r.s.prog.Needs r.b g → r.SiteHolds h S g then some hx.choose
  else none

/-- **The slot map**: each gate's site's slot. -/
noncomputable def slotMap : ℕ → Option Slot := slotsOf (siteMap h)

theorem siteMap_spec (hS : Sites h) (g : ℕ) :
    ∃ S, siteMap h g = some S ∧ ∀ r : CheckedRun, r.Ok h → r.s.prog.Needs r.b g → r.SiteHolds h S g :=
  ⟨(hS g).choose, dif_pos (hS g), (hS g).choose_spec⟩

/-- **`SlotsAt` at the slot map** (the module's docstring). -/
theorem slotsAt_slotMap (hS : Sites h) (c₀ : Code) :
    SlotsAt h (slotMap h) (revealCode h (siteMap h) c₀) (layoutFor h) :=
  slotsAt_of h (fun r hr g hg => by
    obtain ⟨S, hs, hH⟩ := siteMap_spec h hS g
    exact ⟨S, hs, hH r hr hg⟩) c₀

theorem slotMap_frame (f : Code) : slotMap { h with frame := f } = slotMap h := rfl

theorem revealCode_frame (f : Code) (site : ℕ → Option Site) (c₀ : Code) :
    revealCode { h with frame := f } site c₀ = revealCode h site c₀ := rfl

theorem layoutFor_frame (f : Code) : layoutFor { h with frame := f } = layoutFor h := rfl

end SpotFlock
