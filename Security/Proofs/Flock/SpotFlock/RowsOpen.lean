module

public import Proofs.Flock.Recursive.InnerRows
public import Proofs.Flock.SpotFlock.Frame
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

`commit_block`: the commit string of a row read off the hidden values at `hidAt`, at template instance `i` of block
`β`, as `FrameIs` (`SpotFlock.Frame`) states the frame's row and salt, is `rows_open`'s of unit `β · g + i`, at the
session's block width (`kLog_setupOf`).

`sessionScoped`: the verifier's check puts each session statement in Flock's scope with hidden outputs, since
`sessionOk` applies Flock's scope check and the hidden-outputs tag (`scoped_of_stmtOk`, `scopedHidden_of_checks`), so
`rows_open` applies at every session statement the check passes.

`reveals_of_frameIs`: `Reveals` (`SpotFlock.Opens`) at a layout's decoder from three premises on the registration: each
checked unit has its block (`Embeds`), each value it needs has its row at a unit of a session table in that block
(`RowsIn`), and the frame opens those rows (`FrameIs`). So `sessionOpensTwo_of_frame`: `SessionOpensTwo` from `Embeds`,
`RowsIn` and `FrameIs`.
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

/-- **The session's block width is each table's**: at a session statement that sets up, whose reading passes the
statement check, the claims' statement's `kLog` is table `j`'s (`Readable.tables_fit`). -/
theorem kLog_setupOf {xS : Msg} {S : Claims.Flock.Stmt} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hc : ins.read.check = .ok ()) {j : ℕ}
    (hj : j < sts.size) : S.kLog = sts[j].c.kLog := by
  unfold FlockDeploy.parse _root_.Flock.Readable.parseOf at hS
  obtain ⟨p, hp, hpS⟩ := Option.map_eq_some_iff.1 hS
  obtain ⟨rfl, hP⟩ := parsed_setupOf hset hp
  obtain ⟨hRu, -⟩ := (_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  have hfit := _root_.Flock.Readable.tables_fit (r := ins.read) (_root_.Flock.Readable.tablesOf_of_parses hP)
    hRu.2.1 hpi hj
  have h0 : 0 < sts.size := by omega
  rw [← hpS]
  simp only [_root_.Flock.Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]
  exact hfit.2.1.symm

/-- **`Binding.commitString`'s bytes are the executable's commit string**: of a row's bytes `r` under the salt bytes
`y`, `b ‖ c` is `Hm96.Default512.commitString` of the row's SHA-512 under the salt (`InnerRows.commitString_data`). -/
theorem bytes_commitString (r y : List UInt8) :
    bytes (FlockSoundness.Binding.commitString r y) =
      _root_.Flock.Hm96.Default512.commitString (_root_.Flock.Sha512.hash ⟨r.toArray⟩) ⟨y.toArray⟩ :=
  _root_.Flock.InnerRows.eq_of_data (by
    simpa [bytes] using _root_.Flock.InnerRows.commitString_data r ⟨y.toArray⟩)

/-- **A row read off the hidden values in a block is the unit's**: the commit string of port `q`'s row message and
salt bytes at template instance `i` of table `j`'s block `β`, read off the hidden values at `hidAt`, is the executable
commit string of row `q` of the table's unit `β · g + i` (`InnerRows.rowBytes` under `InnerRows.salt`). -/
theorem commit_block {ins : FlockDeploy.Ins} {sts : Array _root_.Flock.Stmt} (w : Claims.Flock.Wit) {j β i q : ℕ}
    (hj : j < sts.size) (hi : i < sts[j].c.g) :
    bytes (FlockSoundness.Binding.commitString
        (FlockSoundness.Discharge.Hidden.Any.rowMsg sts[j].c.ports[q]!
          (FlockSoundness.Discharge.Hm.bytesOf (fun y => w.hid.getD (hidAt sts.size sts[j].c.kLog j β y) false)
            (2 * sts[j].c.ports[q]!.words) (FlockSoundness.Typed.msgCol sts[j].c i q)))
        (FlockSoundness.Discharge.Hm.bytesOf (fun y => w.hid.getD (hidAt sts.size sts[j].c.kLog j β y) false) 192
          (FlockSoundness.Discharge.Hm.saltCol sts[j].c i q))) =
      _root_.Flock.Hm96.Default512.commitString
        (_root_.Flock.Sha512.hash
          ⟨(_root_.Flock.InnerRows.rowBytes { ins.read with sts } w j (β * sts[j].c.g + i) q).toArray⟩)
        (_root_.Flock.InnerRows.salt { ins.read with sts } w j (β * sts[j].c.g + i) q) := by
  have hg : 0 < sts[j].c.g := Nat.zero_lt_of_lt hi
  have hdiv : (β * sts[j].c.g + i) / sts[j].c.g = β := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hg, Nat.div_eq_of_lt hi, Nat.zero_add]
  have hmod : (β * sts[j].c.g + i) % sts[j].c.g = i := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
  have hbit : (fun y => w.hid.getD (hidAt sts.size sts[j].c.kLog j β y) false) =
      _root_.Flock.InnerRows.bitAt { ins.read with sts } w j β := by
    funext y
    simp only [_root_.Flock.InnerRows.bitAt, hidAt, getElem!_pos sts j hj]
  rw [hbit, bytes_commitString]
  unfold _root_.Flock.InnerRows.rowBytes _root_.Flock.InnerRows.salt
  dsimp only
  rw [getElem!_pos sts j hj, hdiv, hmod]

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

/-- **The layout's rows are the session's**: at the statements `Reveals` reads but the witness, the session
statement's setup and the checked unit's block `β`, each gate the unit's correctness reads has its row
(`L.rowOf … = some (j, i, p)`) at a table `j` of the session statement and a template instance `i` of its circuit,
whose unit in the block, `β · g + i`, is one of the table's. -/
def RowsIn (L : Layout) : Prop :=
  ∀ x s Pt d (m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) u b ρ rb ins spec sts β,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    FlockDeploy.setupOf true xS = some (ins, spec, sts) → L.blockOf x m₁ xS u = some β →
    ∀ g, s.prog.Needs b g → ∃ (j i p : ℕ) (st : _root_.Flock.Stmt), L.rowOf x m₁ xS u g = some (j, i, p) ∧
      sts[j]? = some st ∧ i < st.c.g ∧ β * st.c.g + i < st.pub.n

/-- **`Reveals` at a layout's decoder** where each checked unit has its block (`Embeds`), the layout's rows are the
session's (`RowsIn`) and the frame opens them (`FrameIs`): the check puts the session statement in Flock's scope with
hidden outputs (`sessionScoped`), so the registered read of each row passes on the commit string of the witness's row
(`rows_open`), which is the frame's opening's (`commit_block`), so the opening passes (`opens_frame`). -/
theorem reveals_of_frameIs (L : Layout) (he : Embeds h L) (hr : RowsIn h L) (hF : FrameIs h L) :
    Reveals h (decOf L) := by
  intro x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb g hg
  have hsx := sessionScoped h x m₁ xS c₀ c₁ hok
  obtain ⟨it, ins, spec, sts, own, hit, hset, -, hown, hall⟩ := rows_open h hx hok hsx hck
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨β, hβ, -⟩ := he x s Pt d m₁ xS c₀ c₁ u b ρ rb S hx hP h2 hin hok hu hS hb hρ hrb
    obtain ⟨j, i, p, st, hrow, hst, hi, hn⟩ :=
      hr x s Pt d m₁ xS c₀ c₁ u b ρ rb ins spec sts β hx hP h2 hin hok hu hb hρ hrb hset hβ g hg
    obtain ⟨q, hq, hpos, hval, hroot, hrq, hsalt, reg, hreg, t, positions, paths, hp, ht, hent, hat, hfind, hπ⟩ :=
      hF x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb it ins spec sts S own β hx hP h2 hin hok hu hck hb hρ hrb hit hset
        hS hown hβ g j i p st hg hrow hst
    obtain ⟨hj, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
    obtain ⟨t', _idx, ht', -, hregs⟩ := hall j hj
    rw [ht] at ht'
    cases ht'
    obtain ⟨p', positions', paths', hp', hent', hrows⟩ := hregs reg hreg
    rw [hp] at hp'
    cases hp'
    rw [hent] at hent'
    cases hent'
    obtain ⟨-, -, hopens⟩ := hrows (β * sts[j].c.g + i) hn
    refine ⟨q, hq, ?_, hpos, hval, hroot⟩
    subst hpos
    refine opens_frame h hat hfind ?_
    rw [hrq, hsalt, hπ, kLog_setupOf hset hS (hsx ins spec sts hset).1 hj, commit_block (ins := ins) wF hj hi]
    exact hopens

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a layout** where the session statement holds each checked
unit's gates (`Embeds`), the layout's rows are the session's (`RowsIn`) and the frame opens them (`FrameIs`). -/
theorem sessionOpensTwo_of_frame (L : Layout) (he : Embeds h L) (hr : RowsIn h L) (hF : FrameIs h L) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_embeds h L he (reveals_of_frameIs h L he hr hF)

end SpotFlock
