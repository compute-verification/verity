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
(`SecondStage.readOk`): at a port no link names, where the interior record has a port named `v`, or where neither
record has and the boundary record registers no value `v`. -/
def readsInterior (own : Spot.SecondStage.Own) (t port v : String) : Bool :=
  match Spot.SecondStage.linkOf own t port with
  | some _ => false
  | none => ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome ||
      (((Spot.Record.byPort own.boundary).find? (·.1 == v)).isNone &&
        (Spot.SecondStage.registeredEntry own.boundary v).isNone)

/-- **The entry round 2 reads a registered read's value `v` at template `t`'s port `port` off**
(`SecondStage.readOk`): the boundary port a link names; else the interior record's port named `v`, where it has one
(at a port a wire names, its writer's output port); else the registered value `v`. The entry is the value's, not the
reading port's, so every session that reads one value names one entry. -/
def readKey (own : Spot.SecondStage.Own) (t port v : String) : SlotKey :=
  match Spot.SecondStage.linkOf own t port with
  | some l => .port l.value
  | none => if ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome then .port v else .reg v

/-- **Each value a `mapM` over `Option` returns is one of its arguments' images.** -/
theorem mem_of_mapM_some {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {r : List β}, l.mapM f = some r → ∀ y ∈ r, ∃ x ∈ l, f x = some y
  | [], r, hm => by
    rw [List.mapM_nil] at hm
    cases hm
    intro y hy
    cases hy
  | a :: l, r, hm => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.pure_def] at hm
    obtain ⟨b, hb, hm⟩ := Option.bind_eq_some_iff.1 hm
    obtain ⟨r', hr', hm⟩ := Option.bind_eq_some_iff.1 hm
    have e : r = b :: r' := (Option.some.inj hm).symm
    subst e
    intro y hy
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨a, List.mem_cons_self, hb⟩
    · obtain ⟨x, hx, hfx⟩ := mem_of_mapM_some hr' y hy
      exact ⟨x, List.mem_cons_of_mem a hx, hfx⟩

/-- **A fold that keeps an entry named `v` and adds one at a root whose port is `v` ends with one**: folding `f` over
the roots `rs` from `acc`, where `acc` has an entry named `v` or a root of `rs` names port `v`. -/
theorem fold_named {v : String} {f : Array (String × Json) → Json → Array (String × Json)}
    (keep : ∀ acc r, (∃ x ∈ acc, x.1 = v) → ∃ x ∈ f acc r, x.1 = v)
    (add : ∀ acc r, Spot.getD r "port" = .str v → ∃ x ∈ f acc r, x.1 = v) :
    ∀ (rs : List Json) (acc : Array (String × Json)),
      ((∃ x ∈ acc, x.1 = v) ∨ ∃ r ∈ rs, Spot.getD r "port" = .str v) → ∃ x ∈ rs.foldl f acc, x.1 = v
  | [], _, .inl hx => by rw [List.foldl_nil]; exact hx
  | [], _, .inr ⟨_, hr, _⟩ => by cases hr
  | r :: rs, acc, hx => by
    rw [List.foldl_cons]
    refine fold_named keep add rs (f acc r) ?_
    rcases hx with hx | ⟨r', hr', hp⟩
    · exact .inl (keep acc r hx)
    · rcases List.mem_cons.1 hr' with he | hr'
      · subst he
        exact .inl (add acc _ hp)
      · exact .inr ⟨r', hr', hp⟩

/-- **A port a record's roots name is one `Record.byPort` finds**: each name `namesAt` reads off the record's roots'
ports (as `recordOk` reads them) is the name of one of `Record.byPort`'s entries. -/
theorem byPort_find_of_namesAt {rec : Json} {ps : List String} (hps : namesAt rec "roots" "port" = some ps)
    {v : String} (hv : v ∈ ps) : ((Spot.Record.byPort rec).find? (·.1 == v)).isSome = true := by
  unfold namesAt at hps
  obtain ⟨r, hr, hrv⟩ := mem_of_mapM_some hps v hv
  try dsimp only at hrv
  have hp : Spot.getD r "port" = .str v := by
    cases hj : Spot.getD r "port" <;> rw [hj] at hrv <;> cases hrv
    rfl
  obtain ⟨x, hx, hxv⟩ : ∃ x ∈ Spot.Record.byPort rec, x.1 = v := by
    unfold Spot.Record.byPort
    rw [← Array.foldl_toList]
    refine fold_named ?keep ?add _ #[] (.inr ⟨r, ?mem, hp⟩)
    case keep =>
      intro acc r' ⟨x, hx, hxv⟩
      try dsimp only
      split
      · rename_i p _
        by_cases hpv : p = v
        · exact ⟨(p, r'), Array.mem_push.2 (.inr rfl), hpv⟩
        · refine ⟨x, Array.mem_push.2 (.inl (Array.mem_filter.2 ⟨hx, ?_⟩)), hxv⟩
          simp only [hxv, bne_iff_ne, ne_eq]
          exact fun he => hpv he.symm
      · exact ⟨x, hx, hxv⟩
    case add =>
      intro acc r' hp'
      simp only [hp']
      exact ⟨(v, r'), Array.mem_push.2 (.inr rfl), rfl⟩
    case mem =>
      cases hroots : Spot.getD rec "roots" <;> rw [hroots] at hr
      case arr => simpa using hr
      all_goals simp at hr
  exact Array.find?_isSome.2 ⟨x, hx, by simp [hxv]⟩

variable (h : Held)

/-- **The roots round 2 reads template `t`'s registered read of value `v` at port `port` under**, in the run at
statement `x`, round-1 message `m₁` and the coins of rounds 0 and 1, at replay block `ρ`: the replay block's where it
reads the value off the interior record (`readsInterior`), else the statement's. -/
def readRoots (x m₁ : Msg) (c₀ c₁ : FlockDeploy.Coin) (ρ t : ℕ) (port v : String) : List Msg :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁ with
  | some st, some rec, some draw =>
    if readsInterior (ownOf h st rec draw c₀) (h.names.getD t "") port v then irootsOf m₁ ρ else st.roots
  | _, _, _ => []

/-- **The entry round 2 reads template `t`'s registered read of value `v` at port `port` off**, in the same run
(`readKey`). -/
def readKeyAt (x m₁ : Msg) (c₀ c₁ : FlockDeploy.Coin) (t : ℕ) (port v : String) : SlotKey :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁ with
  | some st, some rec, some draw => readKey (ownOf h st rec draw c₀) (h.names.getD t "") port v
  | _, _, _ => .reg v

/-- **An own port's value is read off the interior record**: in a run whose records name their registrations' names
(`recordsOk`, which `stmtOk` holds: `recordsOk_of_stmtOk`), at a port no link names, a value `Held` lists among the
interior registration's ports is read off the interior record (`readsInterior`), under the replay block's roots, at
that port's entry (`readKey`). So the run doesn't choose the arm of such a read. -/
theorem readsInterior_own {x m₁ : Msg} (hr : recordsOk h x m₁ = true) {st rec} (hd : decode x = some st)
    (hj : jsonMsg m₁ = some rec) {draw c₀} {t port v : String}
    (hl : Spot.SecondStage.linkOf (ownOf h st rec draw c₀) t port = none) (hv : v ∈ h.interiorNames.ports) :
    readsInterior (ownOf h st rec draw c₀) t port v = true ∧ readKey (ownOf h st rec draw c₀) t port v = .port v := by
  simp only [recordsOk, hd, hj, Bool.and_eq_true] at hr
  obtain ⟨ps, hps, -, hmem⟩ := (recordOk_spec hr.2).1
  have hf : ((Spot.Record.byPort (ownOf h st rec draw c₀).interior).find? (·.1 == v)).isSome = true :=
    byPort_find_of_namesAt hps ((hmem v).2 hv)
  refine ⟨?_, ?_⟩
  · unfold readsInterior
    rw [hl]
    simp only [hf, Bool.true_or]
  · unfold readKey
    rw [hl]
    exact ite_eq_left hf

/-- **The reveal opens the session's rows, and gate `g`'s slot names where they are** (the module's docstring), at a
checked unit `u` of the run at statement `x` (parsed `s`), round messages `m₀`, `m₁`, `m₂`, coins `c₀` and `c₁` and the
rest `rest`, at the session witness `wF`, the unit's replay block `ρ` and its session's setup: the layout's facts and
the code's opening at the gate as `FrameIs` states them, its value and data `c`'s first opening there
(`Reveal.frameAt`), and then, for the row's port `q`, either the registered read it is named like, the roots round 2
reads its value under, the gate's slot and the header's path, or, at a one-table session's port read from no registered
read, the replay block's roots, the gate's slot and the published tree's path. -/
def SlotsAtGate (slot : ℕ → Option Slot) (c : Code) (L : Layout) (x : Msg) (s : AuditStatement) (m₀ m₁ m₂ : Msg)
    (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) (wF : Claims.Flock.Wit) (u ρ : ℕ) (ins : FlockDeploy.Ins)
    (sts : Array _root_.Flock.Stmt) (g : ℕ) : Prop :=
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
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
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
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
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
