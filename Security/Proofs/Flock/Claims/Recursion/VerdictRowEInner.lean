import Proofs.Flock.Claims.Recursion.VerdictRowECommit

/-!
# An algebra part's `acc0`/`acc1` slots carry the commit string of the `rec-acc` row they read

Row E's cross-part tie (`held_rows_eq_layR`, `out_rows_eq_layR`) takes two parts' hm96 slots to carry one commit string
(`hcm`); `accCommit_eq` gives it at two staged `.recOpen` parts. This is the same at a staged `.inner` part's running
sums (`innerAccRow_table`):

* its setup ran `checkStmt` (`RecIns.setup_checks`), whose `checkAcc` gives the port `acc0`'s registration in the part's
  shared rows (`site_of_checkAcc`), and whose `checkAccReads` makes it read the reps' ends (`checkEndReads`): `acc0`
  reads rows `0` and `n`, `acc1` rows `n` and `2 · n`, `n` a rep's openings (`innerReads_of_checkAccReads`), so unit
  `u`'s `acc0`/`acc1` (`d = 0, 1`) reads row `(u + d) · n` (`endRow_eq`);
* it ran `checkRegistered`, so the slot at unit `u` is the table row that opens the registered root at that position
  (`checkPort_at`).

So the ties take only an equation on chain positions, as `accCommit_eq` does: the rec-open slot's
(`innerAccCommit_eq`), or the other algebra part's (`innerInnerCommit_eq`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **`checkAccReads` at an algebra part reads the reps' ends** (`checkEndReads`): each of its `accPorts` is `own`'s
entry of its name, reading `endWant` of rows `0, n` (`acc0`) and `n, 2 · n` (`acc1`), `n` a rep's openings. -/
theorem innerReads_of_checkAccReads {i : RecVStar.Inner} {k : ℕ} {name : String} {pt : VStar.Part}
    {own : Array Registered.Port} (h : RecVStar.checkAccReads i (.inner k name pt) own = .ok ())
    {sch : Schedule} (hsch : fast100 i.m = some sch) {nm : String}
    (hnm : nm ∈ RecVStar.accPorts (.inner k name pt)) :
    ∃ p, own.find? (·.name == nm) = some p ∧
      p.reads = RecVStar.endWant (#[some 0, some (RecVStar.opensBelow sch sch.levels.size)],
        #[some (RecVStar.opensBelow sch sch.levels.size), some (2 * RecVStar.opensBelow sch sch.levels.size)]) nm := by
  have hw : RecVStar.accReads i (.inner k name pt) = .ok none := rfl
  unfold RecVStar.checkAccReads at h
  rw [hw] at h
  obtain ⟨_, h0, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h0
  dsimp only at h
  have he : RecVStar.endReads i (.inner k name pt) =
      .ok (some (#[some 0, some (RecVStar.opensBelow sch sch.levels.size)],
        #[some (RecVStar.opensBelow sch sch.levels.size), some (2 * RecVStar.opensBelow sch sch.levels.size)])) := by
    simp only [RecVStar.endReads, hsch]
    rfl
  unfold RecVStar.checkEndReads at h
  split at h
  · rename_i hem
    rw [List.isEmpty_iff] at hem
    rw [hem] at hnm
    cases hnm
  rw [he] at h
  obtain ⟨_, h0, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h0
  dsimp only at h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  refine FlockC3.PubRows.rows_forIn_each (fun nm => ∃ p, own.find? (·.name == nm) = some p ∧
    p.reads = RecVStar.endWant (#[some 0, some (RecVStar.opensBelow sch sch.levels.size)],
      #[some (RecVStar.opensBelow sch sch.levels.size), some (2 * RecVStar.opensBelow sch sch.levels.size)]) nm)
    ?_ _ h nm hnm
  intro nm' r hr
  try dsimp only at hr
  split at hr
  · rename_i p hf
    split at hr
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2
    · rename_i hne
      refine ⟨⟨p, hf, by simpa using hne⟩, ?_⟩
      cases hr
      rfl
  · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
    cases h2

/-- **Unit `u`'s `acc0`/`acc1` row** (`d = 0, 1`) in `endWant`'s reads is `(u + d) · n`, and `u` is one of the two
reps. -/
theorem endRow_eq {N n x d : ℕ} {nm : String} (hd : (nm, d) ∈ [("acc0", 0), ("acc1", 1)])
    (h : (RecVStar.endWant (#[some 0, some N], #[some N, some (2 * N)]) nm)[n]? = some (some x)) :
    n < 2 ∧ x = (n + d) * N := by
  simp only [List.mem_cons, List.mem_nil_iff, or_false, Prod.mk.injEq] at hd
  rcases hd with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · have e : RecVStar.endWant (#[some 0, some N], #[some N, some (2 * N)]) "acc0" = #[some 0, some N] := rfl
    rw [e] at h
    rcases n with _ | _ | n
    · refine ⟨by omega, ?_⟩; simp at h ⊢; omega
    · refine ⟨by omega, ?_⟩; simp at h ⊢; omega
    · simp at h
  · have e : RecVStar.endWant (#[some 0, some N], #[some N, some (2 * N)]) "acc1" = #[some N, some (2 * N)] := rfl
    rw [e] at h
    rcases n with _ | _ | n
    · refine ⟨by omega, ?_⟩; simp at h ⊢; omega
    · refine ⟨by omega, ?_⟩; simp at h ⊢; omega
    · simp at h

/-- **An algebra part's `acc0`/`acc1` slot at unit `u` is the registered `rec-acc` row at the rep's end**: at a staged
`.inner` part the witness checks against, with the setup's table labelling its instances in order and the reads file
`own`, port `nm`'s (`d = 0, 1` for `acc0`, `acc1`) entry `reg` of `own` has `acc0`'s registration in the part's shared
rows, and the slot at unit `n` (one of the two reps) reads as the table row the setup opens against `reg`'s root at chain
position `(n + d) · opensBelow`. -/
theorem innerAccRow_table {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {k : ℕ} {name : String} {pt : VStar.Part}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt))
    {sch : Schedule} (hsch : fast100 r.shape.m = some sch) {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own) {nm : String} {d : ℕ}
    (hnm : nm ∈ RecVStar.accPorts (.inner k name pt)) (hd : (nm, d) ∈ [("acc0", 0), ("acc1", 1)]) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) {n : ℕ} (hnp : n < sts[0].pub.n) :
    ∃ reg rws a0, reg ∈ own ∧ reg.name = nm ∧ i.rows = some rws ∧ rws.find? (·.name == "acc0") = some a0 ∧
      FlockVBridge.Family.site reg = FlockVBridge.Family.site a0 ∧ n < 2 ∧
      ∃ t positions paths, sts[0].pub.tables = some t ∧
        (List.range sts[0].c.ports.size).find? (fun p => sts[0].c.ports[p]!.name == reg.name) = some q ∧
        Registered.entry sts[0].pub reg = .ok (positions, paths) ∧
        Registered.ref t sts[0].c.ports.size q n < t.counts.getD q 0 ∧
        positions.getD (Registered.ref t sts[0].c.ports.size q n) 0 =
          (n + d) * RecVStar.opensBelow sch sch.levels.size ∧
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
  obtain ⟨dd, hd', -⟩ := hall 0 h0p
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0p
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0p).2.2
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd' (by omega) hv1 L _ hz1
  subst hc0
  obtain ⟨xr, hxr, hinner, -⟩ := insO_staged hins
  rw [Option.some.inj (hxr.symm.trans hr)] at hinner
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins, hown] at hC
  have hst := checks_stmt hC 0 h0
  obtain ⟨j, f, -, hj, hf, -, -⟩ := checkStmt_own hst
  cases hj
  rw [hf] at hst
  obtain ⟨ss, s', hss, hs', hacc, hrd⟩ := FlockC3.RegLevels.checkAcc_of_checkStmt hst
  rw [hss] at hsv
  obtain rfl : s' = .inner k name pt := Option.some.inj (hs'.symm.trans hsv)
  obtain ⟨reg, hfr, hv⟩ := FlockC3.RegLevels.value_of_checkAcc hacc nm hnm
  have hmem : reg ∈ own := Array.mem_of_find?_eq_some hfr
  have hname : reg.name = nm := by simpa using Array.find?_some hfr
  obtain ⟨rws, a0, hrws, ha0, hsite⟩ := FlockC3.RegLevels.site_of_checkAcc hacc hmem hv
  obtain ⟨p0, hp0, hpr⟩ := innerReads_of_checkAccReads hrd hsch hnm
  obtain rfl : reg = p0 := Option.some.inj (hfr.symm.trans hp0)
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
  rw [range_getD hnp, hpr] at hrdn
  obtain ⟨hn2, hpos⟩ := endRow_eq hd hrdn
  have hnn : n < p.sts[0].n := by rw [setupOne_n hd']; exact hnp
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hsl := slotPub_table ht' hd' (by omega) hv1 L hz1 hq' hnp (div_lt_blocks hg0 hnn hnbl)
  rw [digest_tables htt] at hsl
  exact ⟨reg, rws, a0, hmem, hname, hrws, ha0, hsite, hn2, t, positions, paths, htt, hpp, hent, hlt, hpos, hop, hsl⟩

/-- **An algebra part's `acc0`/`acc1` slot and a rec-open part's `acc`/`acc_out` slot carry one commit string at one
chain position**: at a staged `.inner` part `s` and a staged `.recOpen` part `s'` of `x`, at one staging message (`hxO`, `hxO'`), the witnesses check against,
with their tables labelled in order and the reads files `own`, `own'`, where `s`'s slot at unit `n` (`(n + d) · n₀`,
`innerAccRow_table`) and `s'`'s at unit `n'` (`accRow_table`) are one chain position (`hpos`), the two slots read one
commit string, wherever the true calls `L` and the setups' evaluations at `xs ∋ xO, xO'` have no collision, at a record
that lists the registered reads' evaluations (`RegListed`). -/
theorem innerAccCommit_eq {st : Stage} {s s' : ℕ} {x xO xO' : Msg} {w w' : Claims.Flock.Wit}
    (hS : StagedPart st s x xO) (hS' : StagedPart st s' x xO')
    {ms : List (Msg × Claims.Out)} (hxO : xO = staged st x ms s) (hxO' : xO' = staged st x ms s')
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
    (hown' : i'.base.registered = some own') {nm nm' : String} {d d' : ℕ}
    (hnm : nm ∈ RecVStar.accPorts (.inner k name pt)) (hd : (nm, d) ∈ [("acc0", 0), ("acc1", 1)])
    (hnm' : (nm', d') ∈ [("acc", 0), ("acc_out", 1)]) {q q' : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hq' : c'.ports.findIdx? (·.name == nm') = some q') {n n' : ℕ}
    (hnp : n < sts[0].pub.n) (hn' : n' < 2 * (sch.levels[l']!).q) (hnp' : n' < sts'[0].pub.n)
    (hpos : (n + d) * RecVStar.opensBelow sch sch.levels.size =
      n' / (sch.levels[l']!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l' +
        n' % (sch.levels[l']!).q + d')
    {Co : Type} {D : Claims.Recursion.Deploy Co} (hD : RegListed D) {L : List (Msg × Msg)}
    (hl : ∀ e ∈ L, FlockDeploy.sha512 e.1 = e.2) {xs : List Msg}
    (hc : (Claims.findColl (L ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    (hx : xO ∈ xs) (hx' : xO' ∈ xs) :
    Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
      Claims.Recursion.VStarB.readForms w' (hmCommit c' q' n') := by
  obtain ⟨reg, rws, a0, hreg, -, hrw, ha0, hs, -, t, positions, paths, ht, hp, he, -, hps, -, hrdc⟩ :=
    innerAccRow_table hS hck hr hpc hsv hsch hsu h0 hidx hown hnm hd hq hnp
  obtain ⟨reg', rws', a0', hreg', -, hrw', ha', hs', t', positions', paths', ht', hp', he', -, hps', -, hrd'⟩ :=
    accRow_table hS' hck' hr hpc' hsv' hsch hsu' h0' hidx' hown' hnm' hq' hn' hnp'
  have ho := mem_regOpenings hsu hown (st := sts[0]) (by simp) (by simpa using hreg) (mem_portOpenings ht hp he hnp)
  have ho' := mem_regOpenings hsu' hown' (st := sts'[0]) (by simp) (by simpa using hreg')
    (mem_portOpenings ht' hp' he' hnp')
  have hpp : positions.getD (Registered.ref t sts[0].c.ports.size q n) 0 =
      positions'.getD (Registered.ref t' sts'[0].c.ports.size q' n') 0 := by
    rw [hps, hps']
    exact hpos
  rw [hpp] at ho
  subst hxO hxO'
  have hsite := accRow_site_eq (setupO_some hsu).1 (setupO_some hsu').1 hrw ha0 hs hrw' ha' hs'
  have hbc := bc_eq_of_setup hD hl hc hx hx' ho ho' (sameRoot_of_site hsite)
  rw [hrdc, hrd', hbc]

/-- **Two algebra parts' `acc0`/`acc1` slots carry one commit string at one chain position**: at staged `.inner` parts
`s` and `s'` of `x` at one staging message (`hxO`, `hxO'`) the witnesses check against, with their tables labelled in order and the reads files `own`, `own'`,
where port `nm` (`d`) at unit `n` and port `nm'` (`d'`) at unit `n'` read one chain position (`hpos`), the two slots read
one commit string, wherever the true calls `L` and the setups' evaluations at `xs ∋ xO, xO'` have no collision, at a
record that lists the registered reads' evaluations (`RegListed`). -/
theorem innerInnerCommit_eq {st : Stage} {s s' : ℕ} {x xO xO' : Msg} {w w' : Claims.Flock.Wit}
    (hS : StagedPart st s x xO) (hS' : StagedPart st s' x xO')
    {ms : List (Msg × Claims.Out)} (hxO : xO = staged st x ms s) (hxO' : xO' = staged st x ms s')
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    (hck' : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO' w' = true) {r : XR}
    (hr : readX x = some r) {c c' : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    (hpc' : r.w.parts[s']?.bind partCircuit = some c') {k k' : ℕ} {name name' : String} {pt pt' : VStar.Part}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt))
    (hsv' : (RecVStar.statements r.shape).toOption.bind (·[s']?) = some (.inner k' name' pt'))
    {sch : Schedule} (hsch : fast100 r.shape.m = some sch) {i i' : RecIns} {spec spec' : SessionSpec}
    {sts sts' : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (hsu' : setupO xO' = some (i', spec', sts'))
    (h0 : 0 < sts.size) (h0' : 0 < sts'.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    (hidx' : Registered.unitIndices sts'[0].pub = .ok (Array.range sts'[0].pub.n))
    {own own' : Array Registered.Port} (hown : i.base.registered = some own)
    (hown' : i'.base.registered = some own') {nm nm' : String} {d d' : ℕ}
    (hnm : nm ∈ RecVStar.accPorts (.inner k name pt)) (hd : (nm, d) ∈ [("acc0", 0), ("acc1", 1)])
    (hnm' : nm' ∈ RecVStar.accPorts (.inner k' name' pt')) (hd' : (nm', d') ∈ [("acc0", 0), ("acc1", 1)]) {q q' : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hq' : c'.ports.findIdx? (·.name == nm') = some q') {n n' : ℕ}
    (hnp : n < sts[0].pub.n) (hnp' : n' < sts'[0].pub.n)
    (hpos : (n + d) * RecVStar.opensBelow sch sch.levels.size = (n' + d') * RecVStar.opensBelow sch sch.levels.size)
    {Co : Type} {D : Claims.Recursion.Deploy Co} (hD : RegListed D) {L : List (Msg × Msg)}
    (hl : ∀ e ∈ L, FlockDeploy.sha512 e.1 = e.2) {xs : List Msg}
    (hc : (Claims.findColl (L ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    (hx : xO ∈ xs) (hx' : xO' ∈ xs) :
    Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
      Claims.Recursion.VStarB.readForms w' (hmCommit c' q' n') := by
  obtain ⟨reg, rws, a0, hreg, -, hrw, ha0, hs, -, t, positions, paths, ht, hp, he, -, hps, -, hrdc⟩ :=
    innerAccRow_table hS hck hr hpc hsv hsch hsu h0 hidx hown hnm hd hq hnp
  obtain ⟨reg', rws', a0', hreg', -, hrw', ha0', hs', -, t', positions', paths', ht', hp', he', -, hps', -,
    hrdc'⟩ := innerAccRow_table hS' hck' hr hpc' hsv' hsch hsu' h0' hidx' hown' hnm' hd' hq' hnp'
  have ho := mem_regOpenings hsu hown (st := sts[0]) (by simp) (by simpa using hreg) (mem_portOpenings ht hp he hnp)
  have ho' := mem_regOpenings hsu' hown' (st := sts'[0]) (by simp) (by simpa using hreg')
    (mem_portOpenings ht' hp' he' hnp')
  have hpp : positions.getD (Registered.ref t sts[0].c.ports.size q n) 0 =
      positions'.getD (Registered.ref t' sts'[0].c.ports.size q' n') 0 := by
    rw [hps, hps']
    exact hpos
  rw [hpp] at ho
  subst hxO hxO'
  have hsite := accRow_site_eq (setupO_some hsu).1 (setupO_some hsu').1 hrw ha0 hs hrw' ha0' hs'
  have hbc := bc_eq_of_setup hD hl hc hx hx' ho ho' (sameRoot_of_site hsite)
  rw [hrdc, hrdc', hbc]

end RecDeploy
