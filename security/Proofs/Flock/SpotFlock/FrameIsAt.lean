module

public import Proofs.Flock.SpotFlock.FrameAt

/-!
# `FrameIs` at a frame that reads its ports off the roots

`frameIs_frameAt`: at a registration whose frame is `frameAt slot c` (`FrameAt.lean`), `FrameIs` at a layout `L`
follows from two facts, one about `c` and the slots, one about the run's records.

* `SlotsAt slot c L` (`SlotsAtGate` at one gate): at each gate a checked unit needs, the code's opening there, whose
  value and data are `c`'s first opening at the gate under its roots (`Reveal.frameAt`), holds the witness's bit at the
  layout's column, its row and salt are row `q` of unit `n` of the session's table, and the gate's slot names where the
  frame finds that row. At a port the session reads registered (`reg`): the entry round 2 reads the read's value off
  (`readKeyAt`), under the roots it reads it under (`readRoots`), at the read's schema, not sized, its leaf under those
  roots (`leafAt`), at an entry of the read's leaf count, the position the session header gives the row, the opening's
  path being the header's. At a one-table session's other ports: template `t`'s port of the interior record
  (`SlotKey.interior`), under the replay block's roots, at the published tree's schema, sized to its leaf count, its
  leaf the drawn unit's row, the opening's path being the published tree's (`pubPath`).
* `EntriesAt c`: in an accepted run, the entry round 2 reads each registered read's value off has the read's root,
  leaf count and domain id, and template `t`'s port of the interior record at a port a one-table session doesn't read
  registered has its published tree's root and domain id; each port at most `c`'s own depth.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.NetworkCertifier.Bytes (bs)
open Lean (Json)

/-- **Whether round 2 reads a registered read's value `v` at template `t`'s port `port` off the interior record**
(`SecondStage.readOk`): at a port a link names, where the interior registration registers a value `v`; at a port no
link names, where the interior record has a port named `v`, or where neither record has and the boundary record
registers no value `v`. -/
def readsInterior (own : Spot.SecondStage.Own) (t port v : String) : Bool :=
  match Spot.SecondStage.linkOf own t port with
  | some _ => (Spot.SecondStage.registeredEntry own.interior v).isSome
  | none => ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome ||
      (((Spot.Record.byPort own.boundary).find? (·.1 == v)).isNone &&
        (Spot.SecondStage.registeredEntry own.boundary v).isNone)

/-- **The entry round 2 reads a registered read's value `v` at template `t`'s port `port` off**
(`SecondStage.readOk`): at a port a link names, the registered value `v` where the interior registration registers
one, else the boundary port the link names; else the interior record's port named `v`, where it has one (at a port a
wire names, its writer's output port); else the registered value `v`. The entry is the value's, not the reading
port's, so every session that reads one value names one entry. -/
def readKey (own : Spot.SecondStage.Own) (t port v : String) : SlotKey :=
  match Spot.SecondStage.linkOf own t port with
  | some l => if (Spot.SecondStage.registeredEntry own.interior v).isSome then .reg v else .port l.value
  | none => if ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome then .port v else .reg v

variable (h : Held)

/-- **The roots round 2 reads template `t`'s registered read of value `v` at port `port` under**, in the run at
statement `x`, round-1 message `m₁`, round 0's coin and the second stage's, at replay block `ρ`: the replay block's
where it reads the value off the interior record (`readsInterior`), else the statement's. -/
def readRoots (x m₁ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (ρ t : ℕ) (port v : String) : List Msg :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁ with
  | some st, some rec, some draw =>
    if readsInterior (ownOf h st rec draw c₀) (h.names.getD t "") port v then irootsOf m₁ ρ else st.roots
  | _, _, _ => []

/-- **The entry round 2 reads template `t`'s registered read of value `v` at port `port` off**, in the same run
(`readKey`). -/
def readKeyAt (x m₁ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (t : ℕ) (port v : String) : SlotKey :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁ with
  | some st, some rec, some draw => readKey (ownOf h st rec draw c₀) (h.names.getD t "") port v
  | _, _, _ => .reg v

/-- **The reveal opens the session's rows, and gate `g`'s slot names where they are** (the module's docstring), at a
checked unit `u` of the run at statement `x` (parsed `s`), round messages `m₀`, `m₁`, `m₂`, coins `c₀` and `c₁` and the
rest `rest`, at the session witness `wF`, the unit's replay block `ρ` and its session's setup: the layout's facts and
the code's opening at the gate as `FrameIs` states them, its value and data `c`'s first opening there
(`Reveal.frameAt`), and then, for the row's port `q`, either the registered read it is named like, the roots round 2
reads its value under, the gate's slot and the header's path, or, at a one-table session's port read from no registered
read, the replay block's roots, the gate's slot and the published tree's path. -/
def SlotsAtGate (slot : ℕ → Option Slot) (c : Code) (L : Layout) (x : Msg) (s : AuditStatement) (m₀ m₁ m₂ : Msg)
    (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) (wF : Claims.Flock.Wit) (u ρ : ℕ)
    (ins : FlockDeploy.Ins) (sts : Array _root_.Flock.Stmt) (g : ℕ) : Prop :=
  ∃ β j y n q, L.blockOf x m₁ (unitStmt h m₂ u) u = some β ∧ ∃ hj : j < sts.size,
    y < 2 ^ sts[j].c.kLog ∧ L.wire x m₁ (unitStmt h m₂ u) u g = j * 2 ^ sts[j].c.kLog + y ∧ n < sts[j].pub.n ∧
    let o := Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt c) s u
      (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF g
    o.2.val = _root_.Flock.InnerRows.bitAt { ins.read with sts } wF j β y ∧
      SamplingDeploy.row c.place c.bits o.2 = _root_.Flock.InnerRows.rowBytes { ins.read with sts } wF j n q ∧
      c.salt o.2 = (_root_.Flock.InnerRows.salt { ins.read with sts } wF j n q).data.toList ∧
      ((∃ own reg, ins.registered = some own ∧ reg ∈ own ∧
        (List.range sts[j].c.ports.size).find? (fun q => sts[j].c.ports[q]!.name == reg.name) = some q ∧
        o.1 = readRoots h x m₁ c₀ c₁ ρ (templateOf h.ms u) reg.name reg.value ∧
        ∀ t positions paths, sts[j].pub.tables = some t →
          _root_.Flock.Registered.entry sts[j].pub reg = .ok (positions, paths) →
          ∃ sl, slot g = some sl ∧ sl.key = readKeyAt h x m₁ c₀ c₁ (templateOf h.ms u) reg.name reg.value ∧
            sl.schema = reg.schema ∧ sl.sized = false ∧
            (∀ e, entryAt o.1 sl = some e → e.leaves = reg.leaves →
              leafAt o.1 sl e = positions.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) 0) ∧
            bytes (c.π o.2) = paths.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) ByteArray.empty) ∨
      (ins.read.I.count ≤ 1 ∧ q < sts[j].c.ports.size ∧ o.1 = iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ ∧
        (∀ own, ins.registered = some own → ∀ reg ∈ own, reg.name ≠ sts[j].c.ports[q]!.name) ∧ ∀ pop,
        _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop →
        ∃ sl, slot g = some sl ∧ sl.key = .interior (h.names.getD (templateOf h.ms u) "") sts[j].c.ports[q]!.name ∧
          sl.schema = (pubPort sts[j].c pop q).schema ∧ sl.sized = true ∧
          unitsAt o.1 sl = (pubPort sts[j].c pop q).leaves ∧
          (∀ e, entryAt o.1 sl = some e → leafAt o.1 sl e = rowAt sts[j].c pop q (unitOf ins.read.draw n)) ∧
          bytes (c.π o.2) = pubPath sts[j].c pop q (rowAt sts[j].c pop q (unitOf ins.read.draw n))))

/-- **`SlotsAtGate` at every gate each checked unit of an accepted run needs**, at `FrameIs`'s statements and
setup. -/
def SlotsAt (slot : ℕ → Option Slot) (c : Code) (L : Layout) : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) →
    ∀ g, s.prog.Needs b g → SlotsAtGate h slot c L x s m₀ m₁ m₂ c₀ c₁ rest wF u ρ ins sts g

/-- **The entries the slots name have their ports' fields** (the module's docstring), at `FrameIs`'s statements and
setup: under the roots round 2 reads each of the session's registered reads under, the entry it reads the read's value
off has the read's root, leaf count and domain id; under the replay block's roots, template `t`'s port of the interior
record at a port of a one-table session that no registered read is named like has the root and domain id of that
port's published tree over any population the session's public file loads; each port at most `c`'s own depth. -/
def EntriesAt (c : Code) : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) →
    (∀ own reg, ins.registered = some own → reg ∈ own → ∃ e,
      keyAt (readRoots h x m₁ c₀ c₁ ρ (templateOf h.ms u) reg.name reg.value)
        (readKeyAt h x m₁ c₀ c₁ (templateOf h.ms u) reg.name reg.value) = some e ∧
      bytes e.root = reg.root ∧ e.leaves = reg.leaves ∧ bytes e.domain = reg.domain ∧
      _root_.Flock.log2ceil reg.leaves ≤ depth c.own) ∧
    (∀ j (hj : j < sts.size) q pop, ins.read.I.count ≤ 1 → q < sts[j].c.ports.size →
      (∀ own, ins.registered = some own → ∀ reg ∈ own, reg.name ≠ sts[j].c.ports[q]!.name) →
      _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop → ∃ e,
      keyAt (iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ)
        (.interior (h.names.getD (templateOf h.ms u) "") sts[j].c.ports[q]!.name) = some e ∧
      bytes e.root = (pubPort sts[j].c pop q).root ∧ bytes e.domain = (pubPort sts[j].c pop q).domain ∧
      _root_.Flock.log2ceil (pubPort sts[j].c pop q).leaves ≤ depth c.own)

/-- **`FrameIs` at a frame that reads its ports off the roots**: the reveal's opening and the gate's slot
(`SlotsAt`), and the entry the slot names (`EntriesAt`), give the frame's port either branch asks for, at the slot's
leaf (`port_frameAt`), with the reveal's path. -/
theorem frameIs_frameAt {slot : ℕ → Option Slot} {c : Code} {L : Layout} (hf : h.frame = frameAt slot c)
    (hS : SlotsAt h slot c L) (hE : EntriesAt h c) : FrameIs h L := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  obtain ⟨β, j, y, n, q, hβ, hj, hy, hw, hn, hval, hrow, hsalt, hbr⟩ :=
    hS x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  obtain ⟨hEr, hEp⟩ :=
    hE x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset
  rw [hf, show Reveal.frameAt (frameAt slot c) = Reveal.frameAt c from rfl]
  generalize Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt c) s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF g
    = o at hval hrow hsalt hbr ⊢
  refine ⟨β, j, y, n, q, hβ, hj, hy, hw, hn, hval, hrow, hsalt, ?_⟩
  rcases hbr with ⟨own, reg, hown, hreg, hq, hR, hslot⟩ | ⟨hc1, hq, hR, hnr, hslot⟩
  · refine Or.inl ⟨own, reg, hown, hreg, hq, fun t positions paths ht hent => ?_⟩
    obtain ⟨sl, hs, hk, hsc, hsz, hleaf, hπ⟩ := hslot t positions paths ht hent
    obtain ⟨e, he, hr, hl, hd, hcap⟩ := hEr own reg hown hreg
    rw [← hR, ← hk] at he
    have hl' : leavesAt o.1 sl e = reg.leaves := by simp [leavesAt, hsz, hl]
    obtain ⟨P, hPr, hPl, hPd, hPs, hat, hfind⟩ := port_frameAt (c := c) hs he hr hl' hd hsc hcap
    exact ⟨P, hPr, hPl, hPd, hPs, by rw [← hleaf e he hl]; exact hat, hfind, hπ⟩
  · refine Or.inr ⟨hc1, hq, fun pop hpop => ?_⟩
    obtain ⟨sl, hs, hk, hsc, hsz, hun, hleaf, hπ⟩ := hslot pop hpop
    obtain ⟨e, he, hr, hd, hcap⟩ := hEp j hj q pop hc1 hq hnr hpop
    rw [← hR, ← hk] at he
    have hl' : leavesAt o.1 sl e = (pubPort sts[j].c pop q).leaves := by simp [leavesAt, hsz, hun]
    obtain ⟨P, hPr, hPl, hPd, hPs, hat, hfind⟩ := port_frameAt (c := c) hs he hr hl' hd hsc hcap
    exact ⟨P, hPr, hPl, hPd, hPs, by rw [← hleaf e he]; exact hat, hfind, hπ⟩

end SpotFlock
