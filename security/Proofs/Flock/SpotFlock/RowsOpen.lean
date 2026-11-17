module

public import Proofs.Flock.Recursive.InnerRows
public import Proofs.Flock.SpotFlock.Rooted
public import Proofs.Flock.SpotFlock.Rows
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.Claims.Scope.Sizes

/-! A session row's commit string is the table row its registered read opens, or its population's published tree's, so
`SessionOpensTwo` follows from `Embeds` and `FrameRows` (`sessionOpensTwo_of_rows`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

def ScopedHidden (xS : Msg) : Prop :=
  ∀ ins spec sts, FlockDeploy.setupOf true xS = some (ins, spec, sts) →
    ins.read.check = .ok () ∧ ins.read.tags.hiddenOutputs = true

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

def SessionScoped : Prop :=
  ∀ x m₁ xS c₀ c₁ t, stmtOkAt h x m₁ xS c₀ c₁ t = true → ScopedHidden xS

theorem sessionScoped : SessionScoped h := fun _ _ _ _ _ _ hok =>
  scopedHidden_of_checks (scoped_of_stmtOkAt h hok).1 (scoped_of_stmtOkAt h hok).2

/-- A session of more than one table needs only the first branch: round 2 refuses a session split across Flock tables
(`second-tables`). -/
def FrameRows (L : Layout) : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
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

theorem sessionOpensTwo_of_rows (L : Layout) (he : Embeds h L) (hf : FrameRows h L) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_embeds h L he (reveals_of_frameRows h L hf)

end SpotFlock
