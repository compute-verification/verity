module

public import Proofs.Flock.Recursive.InnerRows
public import Proofs.Flock.SpotFlock.Rows
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.Claims.Scope.Sizes

/-!
# A session row's commit string is the table row its registered read opens

`rows_open`: at a run's statement and a session statement the verifier's check passes (`stmtOk`), in Flock's scope with
hidden outputs (`ScopedHidden`), and a Flock witness its relation accepts, row `q` of unit `n` of each table, as the
witness holds it, commits to the table row that the registered read of `q`'s port opens. The commit string
(`Hm96.Default512.commitString`) of the SHA-512 of the bytes the row hash reads (`InnerRows.rowBytes`), under the salt
the `hm96` slot reads (`InnerRows.salt`), opens the port's root at the position and through the path the session gives
the unit (`Registered.opens`). It is `InnerRows.rows_commit_check` (the circuit computes the commit string and the
public file pins it) after `opens_of_stmtOk` (the registered reads open the public file's rows).

`decOf_bitAt`: a layout's decoder at table `j`'s column `y` of the unit's block is the witness's bit there, as
`InnerRows.bitAt` reads it, the bit `rowBytes` and `salt` read.

`sessionScoped`: the verifier's check puts each session statement in Flock's scope with hidden outputs, since
`sessionOk` applies Flock's scope check and the hidden-outputs tag (`scoped_of_stmtOk`, `scopedHidden_of_checks`), so
`rows_open` applies at every session statement the check passes.

`reveals_of_frameRows`: `Reveals` (`SpotFlock.Opens`) at a layout's decoder from one premise on the registration,
`FrameRows`: for each value a checked unit needs, the layout's wire is a column of a table in the unit's block, and the
code's reveal holds an opening of the witness's bit there that passes wherever the registered read of the row holding
it passes. So `sessionOpensTwo_of_rows`: `SessionOpensTwo` from `Embeds` and `FrameRows`.
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
    (hy : y < 2 ^ sts[j].c.kLog) (hw : L.wire u g = j * 2 ^ sts[j].c.kLog + y) :
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

/-- **A session row's commit string is the table row its registered read opens**: at a run's statement and a session
statement the verifier's check passes, in Flock's scope with hidden outputs, and a Flock witness its relation accepts,
the session sets up with the registered ports `own` (`opens_of_stmtOk`), and at each table `j`, for each port `reg` of
`own`, the table's port `q` named like it and each unit `n`: the commit string of row `q` of unit `n`, as the witness
holds it (`InnerRows.rowBytes` under `InnerRows.salt`), opens `reg` at the position its reads give for the unit, through
the path the session gives there. -/
theorem rows_open {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} {s : AuditStatement} {w : Claims.Flock.Wit}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ xS c₀ c₁ = true) (hsc : ScopedHidden xS)
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true) :
    ∃ it ins spec sts own, interiorsOf m₁ = some it ∧ FlockDeploy.setupOf true xS = some (ins, spec, sts) ∧
      ins.registered = some own ∧
      (FlockDeploy.jsonOf "registered reads" (h.registered s.roots it.roots) >>= _root_.Flock.Registered.ofJson) =
        .ok own ∧
      ∀ j (hj : j < sts.size), ∃ t idx, sts[j].pub.tables = some t ∧
        _root_.Flock.Registered.unitIndices sts[j].pub = .ok idx ∧
        ∀ reg ∈ own, ∃ q positions paths,
          (List.range sts[j].c.ports.size).find? (fun q => sts[j].c.ports[q]!.name == reg.name) = some q ∧
          _root_.Flock.Registered.entry sts[j].pub reg = .ok (positions, paths) ∧ ∀ n < sts[j].pub.n,
            _root_.Flock.Registered.ref t sts[j].c.ports.size q n < t.counts.getD q 0 ∧
            reg.reads[idx.getD n 0]? =
              some (some (positions.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) 0)) ∧
            _root_.Flock.Registered.opens reg
              (_root_.Flock.Hm96.Default512.commitString
                (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } w j n q).toArray⟩)
                (_root_.Flock.InnerRows.salt { ins.read with sts } w j n q))
              (positions.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) 0)
              (paths.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) ByteArray.empty) = true := by
  obtain ⟨it, ins, spec, sts, own, hit, hset, hown, hjson, hall⟩ := opens_of_stmtOk h hx hok
  obtain ⟨hc, hh⟩ := hsc ins spec sts hset
  obtain ⟨hP, hR⟩ := relS_setupOf hset hck
  refine ⟨it, ins, spec, sts, own, hit, hset, hown, hjson, fun j hj => ?_⟩
  obtain ⟨t, idx, ht, hidx, hregs⟩ := hall sts[j] (Array.getElem_mem hj)
  refine ⟨t, idx, ht, hidx, fun reg hreg => ?_⟩
  obtain ⟨q, positions, paths, hq, hent, hn⟩ := hregs reg hreg
  have hqlt : q < sts[j].c.ports.size := List.mem_range.1 (List.mem_of_find?_eq_some hq)
  refine ⟨q, positions, paths, hq, hent, fun n hn' => ?_⟩
  obtain ⟨hlt, hreads, hopens⟩ := hn n hn'
  refine ⟨hlt, hreads, ?_⟩
  have hrow : _root_.Flock.Hm96.Default512.commitString
      (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } w j n q).toArray⟩)
      (_root_.Flock.InnerRows.salt { ins.read with sts } w j n q) =
      _root_.Flock.Registered.tableRow sts[j].pub t q (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) := by
    refine (_root_.Flock.InnerRows.rows_commit_check (p := { ins.read with sts }) hP hc hh hR hj hn' hqlt).trans ?_
    show sts[j].pub.digest n q sts[j].c.ports.size = _
    unfold _root_.Flock.Public.digest _root_.Flock.Registered.tableRow _root_.Flock.Registered.ref
    rw [ht]
  rw [hrow]
  exact hopens

/-- **The verifier's check puts each session statement in Flock's scope with hidden outputs** (`ScopedHidden`). -/
def SessionScoped : Prop :=
  ∀ x m₁ xS c₀ c₁, stmtOk h x m₁ xS c₀ c₁ = true → ScopedHidden xS

/-- **The verifier's check puts each session statement in Flock's scope with hidden outputs**: `sessionOk` applies
Flock's scope check and the hidden-outputs tag (`scoped_of_stmtOk`). -/
theorem sessionScoped : SessionScoped h := fun _ _ _ _ _ hok =>
  scopedHidden_of_checks (scoped_of_stmtOk h hok).1 (scoped_of_stmtOk h hok).2

/-- **The frame reveals each value a checked unit needs at its session row**: at the statements `Reveals` reads and the
session's setup, with registered ports `own`, for each gate `g` the unit's correctness reads, the layout's wire for `g`
is column `y` of table `j` in the unit's block `β`, and the reveal holds an opening at `g` of the witness's bit there,
under the roots `Reveals` names, that passes wherever the registered read of `reg`, at table `j`'s port `q` named like
it, opens the commit string of row `q` of a unit `n` of the table. -/
def FrameRows (L : Layout) : Prop :=
  ∀ x s Pt d (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts own,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    FlockDeploy.setupOf true xS = some (ins, spec, sts) → ins.registered = some own →
    ∀ g, s.prog.Needs b g → ∃ β j y n q reg, L.blockOf x m₁ xS u = some β ∧ ∃ hj : j < sts.size,
      y < 2 ^ sts[j].c.kLog ∧ L.wire u g = j * 2 ^ sts[j].c.kLog + y ∧ n < sts[j].pub.n ∧ reg ∈ own ∧
      (List.range sts[j].c.ports.size).find? (fun q => sts[j].c.ports[q]!.name == reg.name) = some q ∧
      ∃ o ∈ (code h).reveal s u (seenAt m₀ m₁ xS c₀ c₁ rest) wF, o.2.pos = g ∧
        o.2.val = _root_.Flock.InnerRows.bitAt { ins.read with sts } wF j β y ∧
        o.1 = (if g ∈ rb ∧ s.prog.Boundary rb g = false then iroots x (seenAt m₀ m₁ xS c₀ c₁ rest) ρ
          else s.roots) ∧
        ∀ t positions paths, sts[j].pub.tables = some t →
          _root_.Flock.Registered.entry sts[j].pub reg = .ok (positions, paths) →
          _root_.Flock.Registered.opens reg
            (_root_.Flock.Hm96.Default512.commitString
              (_root_.Flock.Sha512.hash ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } wF j n q).toArray⟩)
              (_root_.Flock.InnerRows.salt { ins.read with sts } wF j n q))
            (positions.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) 0)
            (paths.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) ByteArray.empty) = true →
          (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 o.1 o.2 = true

/-- **`Reveals` at a layout's decoder** where the frame reveals each needed value at its session row (`FrameRows`):
the check puts the session statement in Flock's scope with hidden outputs (`sessionScoped`), so the registered read of
that row passes (`rows_open`), so the opening does, and its value, the witness's bit at the wire's column, is the
decoded value (`decOf_bitAt`). -/
theorem reveals_of_frameRows (L : Layout) (hf : FrameRows h L) : Reveals h (decOf L) := by
  intro x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb g hg
  have hsx := sessionScoped h x m₁ xS c₀ c₁ hok
  obtain ⟨_it, ins, spec, sts, own, -, hset, hown, -, hall⟩ := rows_open h hx hok hsx hck
  obtain ⟨β, j, y, n, q, reg, hβ, hj, hy, hw, hn, hreg, hq, o, ho, hpos, hval, hroot, hopen⟩ :=
    hf x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb ins spec sts own hx hP h2 hin hok hu hck hb hρ hrb hset hown g hg
  obtain ⟨t, _idx, ht, -, hregs⟩ := hall j hj
  obtain ⟨q', positions, paths, hq', hent, hrows⟩ := hregs reg hreg
  obtain rfl : q' = q := Option.some.inj (hq'.symm.trans hq)
  obtain ⟨-, -, hopens⟩ := hrows n hn
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    refine ⟨o, ho, hopen t positions paths ht hent hopens, hpos, ?_, hroot⟩
    rw [hval, decOf_bitAt hset hS (hsx ins spec sts hset).1 hβ wF hj hy hw]

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a layout** where the session statement holds each checked
unit's gates (`Embeds`) and the frame reveals each needed value at its session row (`FrameRows`). -/
theorem sessionOpensTwo_of_rows (L : Layout) (he : Embeds h L) (hf : FrameRows h L) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_embeds h L he (reveals_of_frameRows h L hf)

end SpotFlock
