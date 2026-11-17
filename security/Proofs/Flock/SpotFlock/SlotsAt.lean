module

public import Proofs.Flock.SpotFlock.RevealAt

/-! `SlotsAt` at the base frame (`slotsAt_slotMap`), from the premise that each gate's value has one address that holds at every accepted run's checked unit that needs it (`Addresses`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

section Columns

open FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

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

def AddressHolds (S : Address) (g : ℕ) : Prop :=
  r.Counted h ∧ r.RowAt h S g ∧ (r.ReadsRegistered h S g ∨ r.ReadsInterior h S g)

end CheckedRun

/-- **Each gate's slot**: its address's. -/
def slotsOf (addr : ℕ → Option Address) (g : ℕ) : Option Slot := (addr g).map Address.slot

namespace CheckedRun

variable {h} {r : CheckedRun h.stage}

theorem sts_pos (hr : r.Ok h) : 0 < r.sts.size := by
  obtain ⟨_, _, _, hx, -, -, -, hok, hu, -, -, -, -, hset⟩ := hr
  have hs := sessionOk_of_stmtOkAt h (stmtOkAt_unit h hx hok hu)
  simp only [sessionOk, Bool.and_eq_true] at hs
  obtain ⟨ins, spec, sts, hset', hpos, -⟩ := zkOk_sound hs.1.1.1
  rw [hset] at hset'
  cases hset'
  exact hpos

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

theorem v2_of_v2Row {p : _root_.Flock.Port} (hp : _root_.Flock.ProvedScope.v2Row p = true) :
    p.v2 = true ∧ p.segs = none := by
  unfold _root_.Flock.ProvedScope.v2Row at hp
  cases hv : p.v2 <;> cases hs : p.segs <;> simp_all

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

theorem slotsAt_of {addr : ℕ → Option Address}
    (hS : ∀ r : CheckedRun h.stage, r.Ok h → ∀ g, r.s.prog.Needs r.b g → ∃ S, addr g = some S ∧ r.AddressHolds h S g)
    (c₀ : Code) : SlotsAt h (slotsOf addr) (revealCode h addr c₀) (layoutFor h) := by
  intro x s Pt d m₀ m₁ m₂ c₀' c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  have hr : CheckedRun.Ok h ⟨x, s, m₀, m₁, m₂, c₀', c₁, rest, wF, u, b, ρ, rb, ins, sts⟩ :=
    ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩
  obtain ⟨S, hsS, hH⟩ := hS _ hr g hg
  exact CheckedRun.slotsAtGate hr c₀ hsS hg hH

/-- Every gate's value has one address, fixed before the run, that holds at every accepted run's checked unit that
needs the gate. A premise, not a theorem: the verifier's checks don't give it, and at some registrations it can't hold
(one width, one copy, one row, one entry and one tree a gate). -/
def Addresses : Prop :=
  ∀ g, ∃ S : Address, ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.AddressHolds h S g

open Classical in
noncomputable def addressMap (g : ℕ) : Option Address :=
  if hx : ∃ S : Address, ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.AddressHolds h S g then
    some hx.choose
  else none

noncomputable def slotMap : ℕ → Option Slot := slotsOf (addressMap h)

theorem addressMap_spec (hS : Addresses h) (g : ℕ) :
    ∃ S, addressMap h g = some S ∧ ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.AddressHolds h S g :=
  ⟨(hS g).choose, dite_eq_left (hS g), (hS g).choose_spec⟩

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
