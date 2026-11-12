module

public import Proofs.Flock.Recursive.InnerRows
public import Proofs.Flock.SpotFlock.Rooted
public import Proofs.Flock.SpotFlock.Rows
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.Claims.Scope.Sizes

/-!
# A session row's commit string is the table row its registered read opens

`rows_open`: at a session statement that sets up as a masked Flock statement with registered ports `own`, in Flock's
scope with hidden outputs (`ScopedHidden`), and a Flock witness its relation accepts, row `q` of unit `n` of each
table, as the witness holds it, commits to the table row that the registered read of `q`'s port opens. The commit
string (`Hm96.Default512.commitString`) of the SHA-512 of the bytes the row hash reads (`InnerRows.rowBytes`), under the
salt the `hm96` slot reads (`InnerRows.salt`), opens the port's root at the position and through the path the session
gives the unit (`Registered.opens`). It is `InnerRows.rows_commit_check` (the circuit computes the commit string and the
public file pins it) after `opens_of_setup` (the registered reads open the public file's rows).

`rows_rooted`: where the statement has one table (`count ≤ 1`), the same commit string opens the population's published
tree at port `q` (`pubPort`) at the row of the unit the sample drew as instance `n` (`unitOf`, `rowAt`), through
the path from that row (`pubPath`): it is the population's `b ‖ c` there (`setupHidden_unit`), a leaf of the tree whose
root Flock's `checkPublic` recomputes (`pubPort_opens`).

`decOf_bitAt`: a layout's decoder at table `j`'s column `y` of the unit's block is the witness's bit there, as
`InnerRows.bitAt` reads it, the bit `rowBytes` and `salt` read.

`sessionScoped`: the verifier's check puts each session statement in Flock's scope with hidden outputs, since
`sessionOk` applies Flock's scope check and the hidden-outputs tag (`scoped_of_stmtOkAt`, `scopedHidden_of_checks`),
so `rows_open` and `rows_rooted` apply at every session statement the check passes.

`reveals_of_frameRows`: `Reveals` (`SpotFlock.Opens`) at a layout's decoder from one premise on the registration,
`FrameRows`: for each value a checked unit needs, the layout's wire is a column of a table in the unit's block, and the
code's reveal holds an opening of the witness's bit there that passes wherever the row holding it opens either the
registered read of its port (the session has registered reads) or its population's published tree (a one-table
session). So `sessionOpensTwo_of_rows`: `SessionOpensTwo` from `Embeds` and `FrameRows`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A session statement in Flock's scope with hidden outputs**: where it sets up, its reading passes the verifier's
statement check (`Readable.Read.check`, the first half of its scope refusal) and hides its outputs. -/
def ScopedHidden (xS : Msg) : Prop :=
  ∀ ins spec sts, FlockDeploy.setupOf true xS = some (ins, spec, sts) →
    ins.read.check = .ok () ∧ ins.read.tags.hiddenOutputs = true

/-- **A session statement Flock's scope check passes, with hidden outputs, is `ScopedHidden`**: the scope refusal
(`Parsed.scope`) begins with the statement check, at the reading the setup's tables parse from (`parsed_setupOf`). -/
theorem scopedHidden_of_checks {xS : Msg} (hs : FlockDeploy.scopeCheck true xS = true) (hh : hiddenOf xS = true) :
    ScopedHidden xS := by
  intro ins spec sts hset
  obtain ⟨hins, -⟩ := FlockDeploy.setupOf_some hset
  have hread : FlockDeploy.readOf true xS = some ins.read := by simp [FlockDeploy.readOf, hins]
  refine ⟨?_, by simpa [hiddenOf, hread] using hh⟩
  unfold FlockDeploy.scopeCheck _root_.Flock.Readable.scopeCheckOf at hs
  split at hs
  · rename_i p hp
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hp
    exact ((_root_.Flock.Readable.scope_ok_iff _).1 ((_root_.Flock.Readable.isOk_iff _).1 hs)).1
  · cases hs

/-- **The relation at the setup's reading**: where a session statement sets up and the claims' check accepts a witness,
the reading with the setup's own tables parses, and the witness satisfies the deployed relation there. -/
theorem relS_setupOf {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {w : Claims.Flock.Wit} (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true) :
    _root_.Flock.Readable.Parses { ins.read with sts } ∧
      Claims.Flock.RelS (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation { ins.read with sts }) w := by
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    have hR : Claims.Flock.RelS S w := by simpa [Claims.Flock.check, hS] using hck
    unfold FlockDeploy.parse _root_.Flock.Readable.parseOf at hS
    obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hS
    obtain ⟨rfl, hP⟩ := parsed_setupOf hset hp
    exact ⟨hP, hR⟩

/-- **The decoder at a table's column is the witness's bit there**: where a layout's wire for gate `g` of unit `u` is
table `j`'s column `y` in the session's block circuit, at a session statement whose reading passes the statement check,
the decoded value is bit `y` of the unit's block in table `j`'s share of the witness (`InnerRows.bitAt`). -/
theorem decOf_bitAt {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β : ℕ} {ins : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hc : ins.read.check = .ok ())
    (hβ : L.blockOf x m₁ xS u = some β) (w : Claims.Flock.Wit) {g j y : ℕ} (hj : j < sts.size)
    (hy : y < 2 ^ sts[j].c.kLog) (hw : L.wire x m₁ xS u g = j * 2 ^ sts[j].c.kLog + y) :
    decOf L x m₁ xS w u g = _root_.Flock.InnerRows.bitAt { ins.read with sts } w j β y := by
  have hS' := hS
  unfold FlockDeploy.parse _root_.Flock.Readable.parseOf at hS'
  obtain ⟨p, hp, hpS⟩ := Option.map_eq_some_iff.1 hS'
  obtain ⟨rfl, hP⟩ := parsed_setupOf hset hp
  obtain ⟨hRu, -⟩ := (_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  have hfit := _root_.Flock.Readable.tables_fit (r := ins.read) (_root_.Flock.Readable.tablesOf_of_parses hP)
    hRu.2.1 hpi hj
  have hk : S.kLog = sts[j].c.kLog := by
    have h0 : 0 < sts.size := by omega
    rw [← hpS]
    simp only [_root_.Flock.Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]
    exact hfit.2.1.symm
  rw [decOf_hid hset hS hβ w hj (by rw [hk]; exact hy) (by rw [hk]; exact hw)]
  unfold _root_.Flock.InnerRows.bitAt hidAt
  dsimp only
  rw [getElem!_pos sts j hj, hk]

variable (h : Held)

/-- **A session row's commit string is the table row its registered read opens**: at a session statement that sets up as
a masked Flock statement with registered ports `own`, in Flock's scope with hidden outputs, and a Flock witness its
relation accepts, at each table `j`, for each port `reg` of `own`, the table's port `q` named like it and each unit `n`:
the commit string of row `q` of unit `n`, as the witness holds it (`InnerRows.rowBytes` under `InnerRows.salt`), opens
`reg` at the position its reads give for the unit, through the path the session gives there. -/
theorem rows_open {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {own} {w : Claims.Flock.Wit}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts)) (hown : ins.registered = some own)
    (hsc : ScopedHidden xS)
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true) :
    ∀ j (hj : j < sts.size), ∃ tb idx, sts[j].pub.tables = some tb ∧
      _root_.Flock.Registered.unitIndices sts[j].pub = .ok idx ∧
      ∀ reg ∈ own, ∃ q positions paths,
        (List.range sts[j].c.ports.size).find? (fun q => sts[j].c.ports[q]!.name == reg.name) = some q ∧
        _root_.Flock.Registered.entry sts[j].pub reg = .ok (positions, paths) ∧ ∀ n < sts[j].pub.n,
          _root_.Flock.Registered.ref tb sts[j].c.ports.size q n < tb.counts.getD q 0 ∧
          reg.reads[idx.getD n 0]? =
            some (some (positions.getD (_root_.Flock.Registered.ref tb sts[j].c.ports.size q n) 0)) ∧
          _root_.Flock.Registered.opens reg
            (_root_.Flock.Hm96.Default512.commitString
              (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } w j n q).toArray⟩)
              (_root_.Flock.InnerRows.salt { ins.read with sts } w j n q))
            (positions.getD (_root_.Flock.Registered.ref tb sts[j].c.ports.size q n) 0)
            (paths.getD (_root_.Flock.Registered.ref tb sts[j].c.ports.size q n) ByteArray.empty) = true := by
  obtain ⟨hc, hh⟩ := hsc ins spec sts hset
  obtain ⟨hP, hR⟩ := relS_setupOf hset hck
  intro j hj
  obtain ⟨tb, idx, ht, hidx, hregs⟩ := opens_of_setup hset hown sts[j] (Array.getElem_mem hj)
  refine ⟨tb, idx, ht, hidx, fun reg hreg => ?_⟩
  obtain ⟨q, positions, paths, hq, hent, hn⟩ := hregs reg hreg
  have hqlt : q < sts[j].c.ports.size := List.mem_range.1 (List.mem_of_find?_eq_some hq)
  refine ⟨q, positions, paths, hq, hent, fun n hn' => ?_⟩
  obtain ⟨hlt, hreads, hopens⟩ := hn n hn'
  refine ⟨hlt, hreads, ?_⟩
  have hrow : _root_.Flock.Hm96.Default512.commitString
      (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } w j n q).toArray⟩)
      (_root_.Flock.InnerRows.salt { ins.read with sts } w j n q) =
      _root_.Flock.Registered.tableRow sts[j].pub tb q (_root_.Flock.Registered.ref tb sts[j].c.ports.size q n) := by
    refine (_root_.Flock.InnerRows.rows_commit_check (p := { ins.read with sts }) hP hc hh hR hj hn' hqlt).trans ?_
    show sts[j].pub.digest n q sts[j].c.ports.size = _
    unfold _root_.Flock.Public.digest _root_.Flock.Registered.tableRow _root_.Flock.Registered.ref
    rw [ht]
  rw [hrow]
  exact hopens

/-- **A session row's commit string opens its population's published tree**: at a one-table session statement
(`count ≤ 1`) that sets up as a masked Flock statement, in Flock's scope with hidden outputs, and a Flock witness its
relation accepts, the session's public file loads a population, and at each table `j`, port `q` and unit `n`, the
commit string of row `q` of unit `n`, as the witness holds it, opens port `q`'s published tree there (`pubPort`) at the
row of the unit the sample drew as `n`, through the path from that row. -/
theorem rows_rooted {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {w : Claims.Flock.Wit}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts)) (hsc : ScopedHidden xS)
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true)
    (hc1 : ins.read.I.count ≤ 1) {j n q : ℕ} (hj : j < sts.size) (hn : n < sts[j].pub.n)
    (hq : q < sts[j].c.ports.size) :
    ∃ pop, _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop ∧
      _root_.Flock.Registered.opens (pubPort sts[j].c pop q)
        (_root_.Flock.Hm96.Default512.commitString
          (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } w j n q).toArray⟩)
          (_root_.Flock.InnerRows.salt { ins.read with sts } w j n q))
        (rowAt sts[j].c pop q (unitOf ins.read.draw n))
        (pubPath sts[j].c pop q (rowAt sts[j].c pop q (unitOf ins.read.draw n))) = true := by
  obtain ⟨hc, hh⟩ := hsc ins spec sts hset
  obtain ⟨hP, hR⟩ := relS_setupOf hset hck
  have hrow : _root_.Flock.Hm96.Default512.commitString
      (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } w j n q).toArray⟩)
      (_root_.Flock.InnerRows.salt { ins.read with sts } w j n q) = sts[j].pub.digest n q sts[j].c.ports.size :=
    _root_.Flock.InnerRows.rows_commit_check (p := { ins.read with sts }) hP hc hh hR hj hn hq
  obtain ⟨hRu, -⟩ := (_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  obtain ⟨spec', hset'⟩ : ∃ spec', ins.read.setup = .ok (spec', sts) :=
    _root_.Flock.Readable.setup_of_tablesOf (_root_.Flock.Readable.tablesOf_of_parses hP)
  unfold _root_.Flock.Readable.Read.setup at hset'
  rw [hpi] at hset'
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset'
  obtain ⟨d, hd, hdraw⟩ := hall j hj
  rw [hdraw hc1] at hd
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd
  rw [_root_.Flock.Readable.tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  obtain ⟨pop, hpopT, -, -⟩ := (FlockSoundness.Discharge.Hidden.setupHidden_spec' hd).pub
  obtain ⟨hu, hdig⟩ := setupHidden_unit hd hpopT hn hq
  have hpop := (_root_.Flock.Readable.loadPublicH_tagsAt _ _ _ _ _).symm.trans hpopT
  refine ⟨pop, hpop, ?_⟩
  rw [hrow, hdig]
  exact pubPort_opens hpop hu hq

/-- **The verifier's check puts each session statement in Flock's scope with hidden outputs** (`ScopedHidden`). -/
def SessionScoped : Prop :=
  ∀ x m₁ xS c₀ c₁ t, stmtOkAt h x m₁ xS c₀ c₁ t = true → ScopedHidden xS

/-- **The verifier's check puts each session statement in Flock's scope with hidden outputs**: `sessionOk` applies
Flock's scope check and the hidden-outputs tag (`scoped_of_stmtOkAt`). -/
theorem sessionScoped : SessionScoped h := fun _ _ _ _ _ _ hok =>
  scopedHidden_of_checks (scoped_of_stmtOkAt h hok).1 (scoped_of_stmtOkAt h hok).2

/-- **The frame reveals each value a checked unit needs at its session row**: at the statements `Reveals` reads and the
session's setup, for each gate `g` the unit's correctness reads, the layout's wire for `g` is column `y` of table `j`
in the unit's block `β`, and the reveal holds an opening at `g` of the witness's bit there, under the roots `Reveals`
names, that passes wherever the commit string of row `q` of a unit `n` of the table opens either

* the registered read `reg` of the session's registered ports `own` that table `j`'s port `q` is named like, at the
  position and through the path the session gives the unit; or
* at a one-table session, port `q`'s published tree over the session's population (`pubPort`), at the row of the unit
  the sample drew as `n`, through the path from that row.

A session of more than one table needs only the first: round 2 refuses a session split across Flock tables
(`second-tables`). -/
def FrameRows (L : Layout) : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) →
    ∀ g, s.prog.Needs b g → ∃ β j y n q, L.blockOf x m₁ (unitStmt h m₂ u) u = some β ∧ ∃ hj : j < sts.size,
      y < 2 ^ sts[j].c.kLog ∧ L.wire x m₁ (unitStmt h m₂ u) u g = j * 2 ^ sts[j].c.kLog + y ∧ n < sts[j].pub.n ∧
      ∃ o ∈ (code h).reveal s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF, o.2.pos = g ∧
        o.2.val = _root_.Flock.InnerRows.bitAt { ins.read with sts } wF j β y ∧
        o.1 = (if g ∈ rb ∧ s.prog.Boundary rb g = false then iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ
          else s.roots) ∧
        ((∃ own reg, ins.registered = some own ∧ reg ∈ own ∧
          (List.range sts[j].c.ports.size).find? (fun q => sts[j].c.ports[q]!.name == reg.name) = some q ∧
          ∀ t positions paths, sts[j].pub.tables = some t →
            _root_.Flock.Registered.entry sts[j].pub reg = .ok (positions, paths) →
            _root_.Flock.Registered.opens reg
              (_root_.Flock.Hm96.Default512.commitString
                (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } wF j n q).toArray⟩)
                (_root_.Flock.InnerRows.salt { ins.read with sts } wF j n q))
              (positions.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) 0)
              (paths.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) ByteArray.empty) = true →
            (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 o.1 o.2 = true) ∨
        (ins.read.I.count ≤ 1 ∧ q < sts[j].c.ports.size ∧ ∀ pop,
          _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop →
          _root_.Flock.Registered.opens (pubPort sts[j].c pop q)
              (_root_.Flock.Hm96.Default512.commitString
                (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } wF j n q).toArray⟩)
                (_root_.Flock.InnerRows.salt { ins.read with sts } wF j n q))
              (rowAt sts[j].c pop q (unitOf ins.read.draw n))
              (pubPath sts[j].c pop q (rowAt sts[j].c pop q (unitOf ins.read.draw n))) = true →
            (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 o.1 o.2 = true))

/-- **`Reveals` at a layout's decoder** where the frame reveals each needed value at its session row (`FrameRows`):
the check puts the session statement in Flock's scope with hidden outputs (`sessionScoped`), so the row opens the
registered read of its port (`rows_open`) or its population's published tree (`rows_rooted`), so the opening passes,
and its value, the witness's bit at the wire's column, is the decoded value (`decOf_bitAt`). -/
theorem reveals_of_frameRows (L : Layout) (hf : FrameRows h L) : Reveals h (decOf L) := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb g hg
  have hokU := stmtOkAt_unit h hx hok hu
  have hsx := sessionScoped h x m₁ (unitStmt h m₂ u) c₀ c₁ _ hokU
  have hs := sessionOk_of_stmtOkAt h hokU
  simp only [sessionOk, Bool.and_eq_true] at hs
  obtain ⟨ins, spec, sts, hset, -⟩ := zkOk_sound hs.1.1.1
  obtain ⟨β, j, y, n, q, hβ, hj, hy, hw, hn, o, ho, hpos, hval, hroot, hbr⟩ :=
    hf x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  have hopen : (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 o.1 o.2 = true := by
    rcases hbr with ⟨own, reg, hown, hreg, hq, hread⟩ | ⟨hc1, hq, hrooted⟩
    · obtain ⟨tb, _idx, ht, -, hregs⟩ := rows_open hset hown hsx hck j hj
      obtain ⟨q', positions, paths, hq', hent, hrows⟩ := hregs reg hreg
      obtain rfl : q' = q := Option.some.inj (hq'.symm.trans hq)
      obtain ⟨-, -, hopens⟩ := hrows n hn
      exact hread tb positions paths ht hent hopens
    · obtain ⟨pop, hpop, hopens⟩ := rows_rooted hset hsx hck hc1 hj hn hq
      exact hrooted pop hpop hopens
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h m₂ u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    refine ⟨o, ho, hopen, hpos, ?_, hroot⟩
    rw [hval, decOf_bitAt hset hS (hsx ins spec sts hset).1 hβ wF hj hy hw]

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a layout** where the session statement holds each checked
unit's gates (`Embeds`) and the frame reveals each needed value at its session row (`FrameRows`). -/
theorem sessionOpensTwo_of_rows (L : Layout) (he : Embeds h L) (hf : FrameRows h L) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_embeds h L he (reveals_of_frameRows h L hf)

end SpotFlock
