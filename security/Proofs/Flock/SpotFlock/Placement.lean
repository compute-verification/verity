module

public import Proofs.Flock.SpotFlock.UnitEmbeds
public import Proofs.Flock.Soundness.Discharge.Hidden.Built

@[expose] public section

/-!
# Where a layout places each checked unit, and `Embeds` there

`templateOk_facts`: what `Embeds` reads of SPOT[Flock]'s registration, per template `t`, where the design evaluates
(`templateOk_of_two`). Member `t`'s circuit is the statement circuit (`Readable.Read.stmt`: the circuit the scope check
parses, unrolled) of the template's reference session statement `xS₀ t` (`Held.ms`), whose unit net is built (`Built`,
from the circuit file's descriptor: `stmt_of_parseStmt`, `built_of_parseStmt`) with at most `2^32` gates
(`unitLog ≤ 32`).

`stmt_of_pin`: a session statement that is a reference statement but for its sample, its public file and its registered
reads (`pinOk`) has the reference statement's statement circuit, which reads only the statement's tags, circuit file,
table files and table count.

`PlacesAt h L x m₁ xS u`: the layout places unit `u`, member `t`'s unit `k`, in its session statement `xS`. At a table
`j` of the statement the verifier parses, and at every accepted statement of that table (`HiddenExec.AcceptedH`) whose
circuit unrolls to member `t`'s, `L` puts the unit at a block and a template instance in range, and sends each wire
the unit computes or reads, at its position in the members' program, to the gate of one of the placed unit's columns
on that wire (`unitPlace`). `exists_wire`: a placed unit's columns cover every such wire.

`embeds_of_placesAt`: a layout that places each checked unit in its session statement holds each checked unit's
circuit gates there (`Embeds`). The unit's session statement is pinned to its template's reference
statement, so each of its tables' accepted statements unrolls to the unit's member circuit (`accepted_check`,
`stmt_of_pin`), and `embeds_unitPlace` holds the unit's gates and constants at the placed columns.
-/

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

/-- **Every wire a rows unit computes or reads is one of its columns'**: the unit's own gates are its computed
columns' (`onto`), and each of them reads only its columns' wires (`op`). -/
theorem exists_wire {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : FlockSoundness.Rows}
    (I : P.IsRowsUnit u R) {x : Fin C.N} (hx : P.unit x = some u ∨ ∃ g, P.unit g = some u ∧ x ∈ (C.op g).args) :
    ∃ c, I.wire c = x := by
  rcases hx with hx | ⟨g, hg, hx⟩
  · obtain ⟨i, rfl⟩ := I.onto x hx
    exact ⟨_, rfl⟩
  · obtain ⟨i, rfl⟩ := I.onto g hg
    rw [I.op i] at hx
    simp only [Verity.Boolean.Op.args, List.mem_append, List.mem_map] at hx
    rcases hx with ⟨c, -, rfl⟩ | ⟨c, -, rfl⟩ <;> exact ⟨c, rfl⟩

/-- **Member `t`'s units are its template's**: the unit `unitBase ms t + k` of a member's own range has template `t`. -/
theorem templateOf_unitBase {ms : Spot.Bits.Members} {t k : ℕ} (ht : t < ms.length) (hk : k < ms[t].2.size) :
    templateOf ms (Properties.Spot.unitBase ms t + k) = t := by
  unfold templateOf
  have hf : (List.range ms.length).find? (fun t' => decide (Properties.Spot.unitBase ms t + k <
      Properties.Spot.unitBase ms (t' + 1))) = some t := by
    rw [List.find?_range_eq_some]
    refine ⟨?_, List.mem_range.2 ht, fun j hj => ?_⟩
    · rw [_root_.Flock.Bridge.Members.unitBase_succ ms ht]
      simpa using hk
    · have := _root_.Flock.Bridge.Members.unitBase_mono ms (show j + 1 ≤ t by omega)
      simp only [Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_false_iff_not, not_lt]
      omega
  rw [hf]
  rfl

/-! ## A pinned session statement's circuit -/

/-- **The verifier's parsers read a statement's tags, circuit file, table files and table count from its fields.** -/
theorem ins_fields {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    Tags.byName? w.statement = some s.read.I.tags ∧ s.read.I.circuitFile = w.circuit ∧ s.read.I.tables = w.tables ∧
      s.read.I.count = w.count := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨by assumption, rfl, rfl, rfl⟩)

/-- **What the masked verifier reads of a decoded statement** is its parsers' reading of the decoded fields. -/
theorem ins_of_readOf {x : Msg} {w : FlockDeploy.Raw} {r : Readable.Read} (hw : FlockDeploy.decode x = some w)
    (hr : FlockDeploy.readOf true x = some r) : ∃ s, w.ins = .ok s ∧ s.read = r := by
  unfold FlockDeploy.readOf FlockDeploy.insOf at hr
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 hr
  obtain ⟨w', hw', hs⟩ := Option.bind_eq_some_iff.1 hs
  obtain rfl : w = w' := Option.some.inj (hw.symm.trans hw')
  split_ifs at hs with hz
  · cases hs
  change w.ins.toOption = some s at hs
  cases hi : w.ins with
  | error e => rw [hi] at hs; cases hs
  | ok s' =>
    rw [hi] at hs
    cases hs
    exact ⟨_, rfl, rfl⟩

/-- **A parsed statement is what the verifier read**, with its tables. -/
theorem readOf_of_parseBytes {xS : Msg} {p : Parsed} (hp : parseBytes (FlockDeploy.readOf true) xS = some p) :
    FlockDeploy.readOf true xS = some p.toRead := by
  unfold parseBytes at hp
  obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
  unfold Readable.Read.parse at hrp
  obtain ⟨sts, -, rfl⟩ := Option.map_eq_some_iff.1 hrp
  exact hr

/-- **A pinned session statement has its reference statement's statement circuit**: two statements equal but for
their samples, public files and registered reads have the same tags, circuit file, table files and table count, which
are all that the statement circuit (`Read.stmt`) and the masked verifier's tags (`ZkExec.zkTags`) read. -/
theorem stmt_of_pin {xS xS₀ : Msg} {w w₀ : FlockDeploy.Raw} (hw : FlockDeploy.decode xS = some w)
    (hw₀ : FlockDeploy.decode xS₀ = some w₀)
    (he : { w with draw := none, registered := none, publicFile := .empty } =
      { w₀ with draw := none, registered := none, publicFile := .empty })
    {r r₀ : Readable.Read} (hr : FlockDeploy.readOf true xS = some r) (hr₀ : FlockDeploy.readOf true xS₀ = some r₀) :
    r.stmt = r₀.stmt := by
  have hz : r.zk = true := FlockDeploy.readOf_zk hr
  have hz₀ : r₀.zk = true := FlockDeploy.readOf_zk hr₀
  obtain ⟨s, hs, rfl⟩ := ins_of_readOf hw hr
  obtain ⟨s₀, hs₀, rfl⟩ := ins_of_readOf hw₀ hr₀
  obtain ⟨ht, hc, htb, hn⟩ := ins_fields hs
  obtain ⟨ht₀, hc₀, htb₀, hn₀⟩ := ins_fields hs₀
  have e1 : w.statement = w₀.statement := (congrArg FlockDeploy.Raw.statement he :)
  have e2 : w.circuit = w₀.circuit := (congrArg FlockDeploy.Raw.circuit he :)
  have e3 : w.tables = w₀.tables := (congrArg FlockDeploy.Raw.tables he :)
  have e4 : w.count = w₀.count := (congrArg FlockDeploy.Raw.count he :)
  rw [e1] at ht
  have htags : s.read.I.tags = s₀.read.I.tags := Option.some.inj (ht.symm.trans ht₀)
  have hT : s.read.tags = s₀.read.tags := by
    unfold Readable.Read.tags ZkExec.zkTags
    simp only [hz, hz₀, ↓reduceIte]
    rw [htags, hc, hc₀, e2, hn, hn₀, e4]
  unfold Readable.Read.stmt Readable.Read.circ
  rw [hT, hc, hc₀, e2, htb, htb₀, e3]

/-- **A hidden-output reading's statement circuit is its circuit file's descriptor's** (`Desc.parseStmt`), at untyped
tags: a templated file's committed circuit unrolled, a flat file's parse, which carries no template (`ParseFacts`). -/
theorem stmt_of_parseStmt {r : Readable.Read} {D : Desc} (hh : r.tags.hiddenOutputs = true)
    (ht : r.tags.typed = false) (hD : Desc.parseStmt r.tags r.I.circuitFile r.I.tables = .ok D) :
    r.stmt = D.unroll := by
  unfold Readable.Read.stmt Readable.Read.circ ProvedScope.circuitOf HmOut.parseCommit
  unfold Desc.parseStmt at hD
  simp only [hh, ↓reduceIte]
  cases hi : r.tags.isTmpl
  · simp only [hi, Bool.false_eq_true, ↓reduceIte] at hD ⊢
    unfold Desc.parse at hD
    split at hD
    · next c hc =>
      cases hD
      rw [hc]
      exact unrolled_flat (parse_facts (Hidden.parse_untyped ht hc).1).tmpl
    · cases hD
  · simp only [hi, ↓reduceIte] at hD ⊢
    split at hD
    · next c hc =>
      cases hD
      simp only [hc, Desc.unroll]
    · cases hD

/-! ## Placing each checked unit -/

variable (h : Held)

/-- **A template whose reference statement reads as a member's** (`templateOk`): member `t`'s circuit is the reference
statement's statement circuit (`Held.ms`), whose unit net is built (`Built`, from its circuit file's descriptor:
`stmt_of_parseStmt`, `built_of_parseStmt`) with at most `2^32` gates. -/
theorem templateOk_facts {t : ℕ} (hok : templateOk h t = true) (ht : t < h.ms.length) :
    (FlockDeploy.readOf true (h.xS₀ t)).map Readable.Read.stmt = some h.ms[t].1 ∧ Built h.ms[t].1.unit ∧
      h.ms[t].1.unit.unitLog ≤ 32 := by
  have hms : h.ms[t].1 = stmtOf (h.xS₀ t) := by simp [Held.ms]
  unfold templateOk at hok
  cases hr : FlockDeploy.readOf true (h.xS₀ t) with
  | none => rw [hr] at hok; cases hok
  | some r =>
    rw [hr] at hok
    have hst : h.ms[t].1 = r.stmt := by
      rw [hms]
      unfold stmtOf
      rw [hr]
      rfl
    simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] at hok
    obtain ⟨⟨⟨⟨⟨hh, hty⟩, hD⟩, hlog⟩, -⟩, -⟩ := hok
    rw [hst]
    split at hD
    · next D hD' =>
      refine ⟨rfl, ?_, hlog⟩
      rw [stmt_of_parseStmt hh hty hD']
      exact Hidden.built_of_parseStmt hD' hty
    · cases hD

/-- **Member `t`'s key program's partition**, as Bridge lowers it. -/
noncomputable abbrev keyP (t : ℕ) (ht : t < h.ms.length) (hU : Built h.ms[t].1.unit) :=
  (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size).partition (Discharge.Layout.outsOf h.ms[t].1 hU h.ms[t].2.size)

/-- **Member `t`'s key program's circuit**, as Bridge lowers it. -/
noncomputable abbrev keyC (t : ℕ) (ht : t < h.ms.length) (hU : Built h.ms[t].1.unit) :=
  (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size).circuit (Discharge.Layout.outsOf h.ms[t].1 hU h.ms[t].2.size)

/-- **Member `t`'s unit `k`, placed at template instance `gi` of block `o`** of an accepted statement whose circuit
unrolls to member `t`'s (`unitPlace`). -/
noncomputable abbrev placed (t : ℕ) (ht : t < h.ms.length) (hU : Built h.ms[t].1.unit) {T : Model.Statement}
    (a : HiddenExec.AcceptedH T) (hparse : HmOut.unrolled a.st.c = h.ms[t].1) (k : Fin h.ms[t].2.size) {gi : ℕ}
    (hg : gi < a.st.c.g) (o : Fin (2 ^ T.nbl)) :=
  unitPlace hU (Discharge.Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) a hparse k hg o

/-- **A layout places unit `u` in its session statement `xS`**: for unit `u`, member `t`'s unit `k`, at some table `j`
of the statement the verifier parses, at every accepted statement of that table whose circuit unrolls to member `t`'s,
the layout puts the unit at a block `o` and a template instance `gi` in range, and sends each wire the unit computes or
reads, at its position in the members' program (`Bits.base`, then Bridge's `posOf`), to the gate of one of the placed
unit's columns on that wire (`unitPlace`). -/
def PlacesAt (L : SpotFlock.Layout) (x m₁ xS : Msg) (u : ℕ) : Prop :=
  ∀ p : Parsed, parseBytes (FlockDeploy.readOf true) xS = some p →
  ∀ t (ht : t < h.ms.length) (hU : Built h.ms[t].1.unit) (k : Fin h.ms[t].2.size),
    u = Properties.Spot.unitBase h.ms t + k →
    ∃ j, ∃ hj : j < p.sts.size, ∀ (T : Model.Statement) (a : HiddenExec.AcceptedH T)
      (hparse : HmOut.unrolled a.st.c = h.ms[t].1), a.st = p.sts[j] →
      ∃ (gi : ℕ) (hg : gi < a.st.c.g) (o : Fin (2 ^ T.nbl)), L.blockOf x m₁ xS u = some o.val ∧
        ∀ y, ((keyP h t ht hU).unit y = some k ∨
            ∃ g', (keyP h t ht hU).unit g' = some k ∧ y ∈ ((keyC h t ht hU).op g').args) →
          ∃ cc : Fin (placed h t ht hU a hparse k hg o).R.w, (placed h t ht hU a hparse k hg o).inst.wire cc = y ∧
            L.wire x m₁ xS u (Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y) =
              j * 2 ^ (toReadable relation p).kLog + ((placed h t ht hU a hparse k hg o).col cc).val

/-- **A layout that places each checked unit in its session statement holds each checked unit's gates there**
(`Embeds`). Where the design evaluates, each template's reference statement reads as a member's (`templateOk_facts`).
A checked unit `u` is member `t`'s unit `k`, whose lowering sits in the run's program (`shape_of_checked`); its session
statement passes round 2 at template `t` (`stmtOkAt_unit`, `templateOf_unitBase`), so it is parsed and checked with
hidden outputs (`parsed_of_stmtOkAt`) and pinned to `xS₀ t` (`pin_of_stmtOkAt`). Each of its tables has an accepted
statement whose circuit is the scope check's (`accepted_check`), the pinned statement's statement circuit unrolls it to
`xS₀ t`'s (`stmt_of_pin`), and that is member `t`'s; at the placement the layout gives, the unit's gates and constants
hold (`embeds_unitPlace`). -/
theorem embeds_of_placesAt {L : SpotFlock.Layout}
    (hL : ∀ x s (m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) u, parse h x = some s →
      stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
      PlacesAt h L x m₁ (unitStmt h m₂ u) u) :
    Embeds h L := by
  intro x s Pt d m₁ m₂ c₀ c₁ u b ρ rb S hx hP h2 hin hok hu hS hb hρ hrb
  have hT := fun t (ht : t < h.ms.length) => templateOk_facts h (templateOk_of_two h h2 ht) ht
  obtain ⟨t, ht, k, hut, hsit, rfl⟩ :=
    shape_of_checked h (fun t ht => ⟨(hT t ht).2.1, (hT t ht).2.2⟩) hx hP h2 hin hb hρ
  have hokAt := stmtOkAt_unit h hx hok hu
  obtain ⟨p, hpb, rfl, hpp, hc, hh⟩ := parsed_of_stmtOkAt h hokAt hS
  have htu : templateOf h.ms u = t := by
    rw [hut]
    exact templateOf_unitBase ht k.isLt
  obtain ⟨j, hj, hpl⟩ := hL x s m₁ m₂ c₀ c₁ u hx hok hu p hpb t ht (hT t ht).2.1 k hut
  obtain ⟨T, a, hst, hk, hcirc, hsat⟩ := accepted_check hpp hc hh hj
  have hparse : HmOut.unrolled a.st.c = h.ms[t].1 := by
    obtain ⟨w, w₀, hw, hw₀, he⟩ := pin_of_stmtOkAt h hokAt
    obtain ⟨r₀, hr₀, hstmt⟩ := Option.map_eq_some_iff.1 (hT t ht).1
    rw [htu] at hw₀
    rw [← hcirc, ← hstmt]
    exact stmt_of_pin hw hw₀ he (readOf_of_parseBytes hpb) hr₀
  obtain ⟨gi, hg, o, hβ, hW⟩ := hpl T a hparse hst
  refine ⟨_, _, _, _, k, _, hsit, rfl, o.val, hβ, fun w hR => ?_⟩
  have hE := embeds_unitPlace (hT t ht).2.1 a hparse hj hk k hg o hR (hsat w hR)
    (fun y => L.wire x m₁ (unitStmt h m₂ u) u
      (Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht (hT t ht).2.1) y)) hW
  exact hE

end SpotFlock
