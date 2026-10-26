import Proofs.Flock.Claims.Recursion.VerdictRowECommit
import Proofs.Flock.Recursive.HPortTie

/-!
# An algebra part's `acc0`/`acc1` slots carry the commit string of the `rec-acc` row they read

Row E's cross-part tie (`held_rows_eq_layR`, `out_rows_eq_layR`) takes two parts' hm96 slots to carry one commit string
(`hcm`); `accCommit_eq` gives it at two staged `.recOpen` parts. This is the same at a staged `.inner` part's running
sums (`innerAccRow_table`):

* its setup ran `checkStmt` (`RecIns.setup_checks`), whose `checkAcc` gives the port `acc0`'s registration in the part's
  shared rows (`site_of_checkAcc`), and whose `checkRows` makes the port the shared rows' entry of its name in every
  field, reads included (`FlockC3.HPort.rowPort_of_checkRows`, `site_reads_of_portDiff`);
* it ran `checkRegistered`, so the slot at unit `u` is the table row that opens the registered root at the position the
  port's reads name for `u` (`checkPort_at`).

The verifier holds the position an `.inner` part reads to the shared rows' entry (`--rec-rows`), not to the chain:
`RecVStar.accReads` names none at an `.inner` statement, so `checkAccReads` checks none there. So each tie takes, as an
equation on the shared rows' entry `a`, the position it reads at `u`: the rec-open slot's chain position
(`innerAccCommit_eq`), or the other algebra part's (`innerInnerCommit_eq`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **An algebra part's `acc0`/`acc1` slot at unit `u` is a registered `rec-acc` row at the position the shared rows
name**: at a staged `.inner` part the witness checks against, with the setup's table labelling its instances in order and
the reads file `own`, port `nm`'s entry `reg` of `own` has `acc0`'s registration in the part's shared rows and the reads
of their entry `a` of that name, and the slot at unit `n` reads as the table row the setup opens against `reg`'s root at
the position `reg` reads for `n`. -/
theorem innerAccRow_table {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {k : ℕ} {name : String} {pt : VStar.Part}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt))
    {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own) {nm : String}
    (hnm : nm ∈ RecVStar.accPorts (.inner k name pt)) {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) {n : ℕ}
    (hnp : n < sts[0].pub.n) :
    ∃ reg rws a0 a, reg ∈ own ∧ reg.name = nm ∧ i.rows = some rws ∧ rws.find? (·.name == "acc0") = some a0 ∧
      FlockVBridge.Family.site reg = FlockVBridge.Family.site a0 ∧ rws.find? (·.name == nm) = some a ∧
      reg.reads = a.reads ∧
      ∃ t positions paths, sts[0].pub.tables = some t ∧
        (List.range sts[0].c.ports.size).find? (fun p => sts[0].c.ports[p]!.name == reg.name) = some q ∧
        Registered.entry sts[0].pub reg = .ok (positions, paths) ∧
        Registered.ref t sts[0].c.ports.size q n < t.counts.getD q 0 ∧
        reg.reads[n]? = some (some (positions.getD (Registered.ref t sts[0].c.ports.size q n) 0)) ∧
        Registered.opens reg (Registered.tableRow sts[0].pub t q (Registered.ref t sts[0].c.ports.size q n))
          (positions.getD (Registered.ref t sts[0].c.ports.size q n) 0)
          (paths.getD (Registered.ref t sts[0].c.ports.size q n) ByteArray.empty) = true ∧
        Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
          (Registered.tableRow sts[0].pub t q (Registered.ref t sts[0].c.ports.size q n)).data.toList := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', -, h0p, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hS' := hS
  obtain ⟨ms, -, -, -, rfl, -, -⟩ := hS'
  obtain ⟨hins, hset⟩ := setupO_some hsu
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  obtain ⟨hsz1, -⟩ := oneFull_staged hS hsu
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hsz : p.sts.size = 1 := by rw [hp]; exact hsz1
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0p
  obtain ⟨L, hz⟩ := layout_of_check_regs hck hx ht hpi h0p
  have hz1 : (L.model (FlockSoundness.Refine.regionsOf p.sts[0])).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨_, hset0⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset0
  rw [hpi] at hset0
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset0
  obtain ⟨dd, hd, -⟩ := hall 0 h0p
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0p
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0p).2.2
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L _ hz1
  subst hc0
  obtain ⟨xr, hxr, hinner, -⟩ := insO_staged hins
  rw [Option.some.inj (hxr.symm.trans hr)] at hinner
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins, hown] at hC
  have hst := checks_stmt hC 0 h0
  obtain ⟨j, f, -, hj, hf, -, -⟩ := checkStmt_own hst
  cases hj
  rw [hf] at hst
  obtain ⟨ss, s', hss, hs', hacc, -⟩ := FlockC3.RegLevels.checkAcc_of_checkStmt hst
  rw [hss] at hsv
  obtain rfl : s' = .inner k name pt := Option.some.inj (hs'.symm.trans hsv)
  obtain ⟨reg, hfr, hv⟩ := FlockC3.RegLevels.value_of_checkAcc hacc nm hnm
  have hmem : reg ∈ own := Array.mem_of_find?_eq_some hfr
  have hname : reg.name = nm := by simpa using Array.find?_some hfr
  obtain ⟨rws, a0, hrws, ha0, hsite⟩ := FlockC3.RegLevels.site_of_checkAcc hacc hmem hv
  obtain ⟨ss2, s2, hss2, hs2, -, hrows, -⟩ := FlockC3.HPort.regChecks_of_checkStmt hst
  obtain rfl : ss2 = ss := Except.ok.inj (hss2.symm.trans hss)
  obtain rfl : s2 = .inner k name pt := Option.some.inj (hs2.symm.trans hs')
  have hrp : nm ∈ RecVStar.rowPorts (.inner k name pt) := by
    have h1 : nm ∈ (pt.S.ports.map (·.1)).filter (fun nm => nm == "acc0" || nm == "acc1") := hnm
    exact (List.mem_filter.1 h1).1
  obtain ⟨rws2, reg2, a, hrws2, hreg2, ha, hdiff⟩ := FlockC3.HPort.rowPort_of_checkRows hrows hrp
  obtain rfl : reg = reg2 := Option.some.inj (hfr.symm.trans hreg2)
  obtain rfl : rws = rws2 := Option.some.inj (hrws.symm.trans hrws2)
  obtain ⟨-, hra⟩ := FlockC3.HPort.site_reads_of_portDiff hdiff
  have hreg := RecIns.setup_registered hset sts[0] (Array.getElem_mem h0)
  rw [hown] at hreg
  obtain ⟨t, idx, htt, hidx2, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hreg)
  rw [hidx] at hidx2
  cases hidx2
  obtain ⟨pp, positions, paths, hpp, hent, hall'⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  rw [← e0] at hpp hall' hnp htt hent ⊢
  have hpp' := hpp
  rw [hname] at hpp'
  obtain rfl : pp = q := find_range_findIdx (f := fun (x : Port) => x.name == nm) hpp' hq
  obtain ⟨hlt, hrdn, hop⟩ := hall' n hnp
  rw [range_getD hnp] at hrdn
  have hnn : n < p.sts[0].n := by rw [setupOne_n hd]; exact hnp
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hsl := slotPub_table ht' hd (by omega) hv1 L hz1 hq' hnp (div_lt_blocks hg0 hnn hnbl)
  rw [digest_tables htt] at hsl
  exact ⟨reg, rws, a0, a, hmem, hname, hrws, ha0, hsite, ha, hra, t, positions, paths, htt, hpp, hent, hlt, hrdn, hop,
    hsl⟩

/-- **An algebra part's `acc0`/`acc1` slot and a rec-open part's `acc`/`acc_out` slot carry one commit string, where the
shared rows put the first at the second's chain position**: at a staged `.inner` part `s` and a staged `.recOpen` part
`s'` of `x` the witnesses check against, with their tables labelled in order and the reads files `own`, `own'`, where the
shared rows' entry `a` of port `nm` reads unit `n` at the chain position of `s'`'s slot at unit `n'` (`hpos`), the two
slots read one commit string, wherever the true calls `L` and the setups' evaluations at `xs ∋ xO, xO'` have no
collision, at a record that lists the registered reads' evaluations (`RegListed`). -/
theorem innerAccCommit_eq {st st' : Stage} {s s' : ℕ} {x xO xO' : Msg} {w w' : Claims.Flock.Wit}
    (hS : StagedPart st s x xO) (hS' : StagedPart st' s' x xO')
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    (hck' : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO' w' = true) {r : XR}
    (hr : readX x = some r) {c c' : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    (hpc' : r.w.parts[s']?.bind partCircuit = some c') {k : ℕ} {name : String} {pt : VStar.Part}
    {l' LANES' HH' m' : ℕ}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt))
    (hsv' : (RecVStar.statements r.shape).toOption.bind (·[s']?) = some (.recOpen l' LANES' HH' m'))
    {sch : Schedule} (hsch : fast100 r.shape.m = some sch) {i i' : RecIns} {spec spec' : SessionSpec}
    {sts sts' : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (hsu' : setupO xO' = some (i', spec', sts'))
    (h0 : 0 < sts.size) (h0' : 0 < sts'.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    (hidx' : Registered.unitIndices sts'[0].pub = .ok (Array.range sts'[0].pub.n))
    {own own' : Array Registered.Port} (hown : i.base.registered = some own)
    (hown' : i'.base.registered = some own') {nm nm' : String} {d' : ℕ}
    (hnm : nm ∈ RecVStar.accPorts (.inner k name pt)) (hnm' : (nm', d') ∈ [("acc", 0), ("acc_out", 1)]) {q q' : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hq' : c'.ports.findIdx? (·.name == nm') = some q') {n n' : ℕ}
    (hnp : n < sts[0].pub.n) (hn' : n' < 2 * (sch.levels[l']!).q) (hnp' : n' < sts'[0].pub.n)
    {rws : Array Registered.Port} {a : Registered.Port} (hrw : i.rows = some rws)
    (ha : rws.find? (·.name == nm) = some a)
    (hpos : a.reads[n]? = some (some (n' / (sch.levels[l']!).q * RecVStar.opensBelow sch sch.levels.size +
      RecVStar.opensBelow sch l' + n' % (sch.levels[l']!).q + d')))
    {Co : Type} {D : Claims.Recursion.Deploy Co} (hD : RegListed D) {L : List (Msg × Msg)}
    (hl : ∀ e ∈ L, FlockDeploy.sha512 e.1 = e.2) {xs : List Msg}
    (hc : (Claims.findColl (L ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    (hx : xO ∈ xs) (hx' : xO' ∈ xs) :
    Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
      Claims.Recursion.VStarB.readForms w' (hmCommit c' q' n') := by
  obtain ⟨reg, rws₁, a0, a₁, hreg, -, hrw₁, ha0, hs, ha₁, hra, t, positions, paths, ht, hp, he, hlt, hrd, -, hrdc⟩ :=
    innerAccRow_table hS hck hr hpc hsv hsu h0 hidx hown hnm hq hnp
  rw [hrw] at hrw₁
  cases hrw₁
  rw [ha] at ha₁
  cases ha₁
  obtain ⟨reg', rws', a0', hreg', -, hrw', ha', hs', t', positions', paths', ht', hp', he', hlt', hps', -, hrd'⟩ :=
    accRow_table hS' hck' hr hpc' hsv' hsch hsu' h0' hidx' hown' hnm' hq' hn' hnp'
  have ho := mem_regOpenings hsu hown (st := sts[0]) (by simp) (by simpa using hreg) (mem_portOpenings ht hp he hlt)
  have ho' := mem_regOpenings hsu' hown' (st := sts'[0]) (by simp) (by simpa using hreg')
    (mem_portOpenings ht' hp' he' hlt')
  have hpp : positions.getD (Registered.ref t sts[0].c.ports.size q n) 0 =
      positions'.getD (Registered.ref t' sts'[0].c.ports.size q' n') 0 := by
    rw [hra, hpos] at hrd
    rw [hps']
    exact (Option.some.inj (Option.some.inj hrd)).symm
  rw [hpp] at ho
  obtain ⟨ms, -, -, -, rfl, -, -⟩ := hS
  obtain ⟨ms', -, -, -, rfl, -, -⟩ := hS'
  have hsite := accRow_site_eq (setupO_some hsu).1 (setupO_some hsu').1 hrw ha0 hs hrw' ha' hs'
  have hbc := bc_eq_of_setup hD hl hc hx hx' ho ho' (sameRoot_of_site hsite)
  rw [hrdc, hrd', hbc]

/-- **Two algebra parts' `acc0`/`acc1` slots carry one commit string, where the shared rows put them at one position**:
at staged `.inner` parts `s` and `s'` of `x` the witnesses check against, with their tables labelled in order and the
reads files `own`, `own'`, where the shared rows' entries `a` (port `nm`) and `a'` (port `nm'`) read units `n` and `n'`
at one position (`hpos`), the two slots read one commit string, wherever the true calls `L` and the setups' evaluations
at `xs ∋ xO, xO'` have no collision, at a record that lists the registered reads' evaluations (`RegListed`). -/
theorem innerInnerCommit_eq {st st' : Stage} {s s' : ℕ} {x xO xO' : Msg} {w w' : Claims.Flock.Wit}
    (hS : StagedPart st s x xO) (hS' : StagedPart st' s' x xO')
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    (hck' : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO' w' = true) {r : XR}
    (hr : readX x = some r) {c c' : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    (hpc' : r.w.parts[s']?.bind partCircuit = some c') {k k' : ℕ} {name name' : String} {pt pt' : VStar.Part}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt))
    (hsv' : (RecVStar.statements r.shape).toOption.bind (·[s']?) = some (.inner k' name' pt'))
    {i i' : RecIns} {spec spec' : SessionSpec}
    {sts sts' : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (hsu' : setupO xO' = some (i', spec', sts'))
    (h0 : 0 < sts.size) (h0' : 0 < sts'.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    (hidx' : Registered.unitIndices sts'[0].pub = .ok (Array.range sts'[0].pub.n))
    {own own' : Array Registered.Port} (hown : i.base.registered = some own)
    (hown' : i'.base.registered = some own') {nm nm' : String}
    (hnm : nm ∈ RecVStar.accPorts (.inner k name pt)) (hnm' : nm' ∈ RecVStar.accPorts (.inner k' name' pt')) {q q' : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hq' : c'.ports.findIdx? (·.name == nm') = some q') {n n' : ℕ}
    (hnp : n < sts[0].pub.n) (hnp' : n' < sts'[0].pub.n)
    {rws rws' : Array Registered.Port} {a a' : Registered.Port} (hrw : i.rows = some rws)
    (hrw' : i'.rows = some rws') (ha : rws.find? (·.name == nm) = some a) (ha' : rws'.find? (·.name == nm') = some a')
    (hpos : a.reads[n]? = a'.reads[n']?)
    {Co : Type} {D : Claims.Recursion.Deploy Co} (hD : RegListed D) {L : List (Msg × Msg)}
    (hl : ∀ e ∈ L, FlockDeploy.sha512 e.1 = e.2) {xs : List Msg}
    (hc : (Claims.findColl (L ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    (hx : xO ∈ xs) (hx' : xO' ∈ xs) :
    Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
      Claims.Recursion.VStarB.readForms w' (hmCommit c' q' n') := by
  obtain ⟨reg, rws₁, a0, a₁, hreg, -, hrw₁, ha0, hs, ha₁, hra, t, positions, paths, ht, hp, he, hlt, hrd, -, hrdc⟩ :=
    innerAccRow_table hS hck hr hpc hsv hsu h0 hidx hown hnm hq hnp
  rw [hrw] at hrw₁
  cases hrw₁
  rw [ha] at ha₁
  cases ha₁
  obtain ⟨reg', rws₂, a0', a₂, hreg', -, hrw₂, ha0', hs', ha₂, hra', t', positions', paths', ht', hp', he', hlt', hrd',
    -, hrdc'⟩ := innerAccRow_table hS' hck' hr hpc' hsv' hsu' h0' hidx' hown' hnm' hq' hnp'
  rw [hrw'] at hrw₂
  cases hrw₂
  rw [ha'] at ha₂
  cases ha₂
  have ho := mem_regOpenings hsu hown (st := sts[0]) (by simp) (by simpa using hreg) (mem_portOpenings ht hp he hlt)
  have ho' := mem_regOpenings hsu' hown' (st := sts'[0]) (by simp) (by simpa using hreg')
    (mem_portOpenings ht' hp' he' hlt')
  have hpp : positions.getD (Registered.ref t sts[0].c.ports.size q n) 0 =
      positions'.getD (Registered.ref t' sts'[0].c.ports.size q' n') 0 := by
    rw [hra, hpos] at hrd
    rw [hra'] at hrd'
    rw [hrd'] at hrd
    exact (Option.some.inj (Option.some.inj hrd)).symm
  rw [hpp] at ho
  obtain ⟨ms, -, -, -, rfl, -, -⟩ := hS
  obtain ⟨ms', -, -, -, rfl, -, -⟩ := hS'
  have hsite := accRow_site_eq (setupO_some hsu).1 (setupO_some hsu').1 hrw ha0 hs hrw' ha0' hs'
  have hbc := bc_eq_of_setup hD hl hc hx hx' ho ho' (sameRoot_of_site hsite)
  rw [hrdc, hrdc', hbc]

end RecDeploy
