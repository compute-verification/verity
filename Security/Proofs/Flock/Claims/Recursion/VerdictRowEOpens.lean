import Proofs.Flock.Claims.Recursion.CapOpens
import Proofs.Flock.C3.RegLevels.AccReads

/-!
# A rec-open part's `acc` and `acc_out` slots are registered `rec-acc` rows at their chain positions

Row E's cross-part tie (`held_rows_eq_layR`, `out_rows_eq_layR`) asks two parts' hm96 slots to carry one commit string
(`hcm`). Each part's slot is a table row the setup opens against a registered `rec-acc` root, at a position its unit's
read names. This is that per-part fact, at a staged `.recOpen` part the witness checks against (`accRow_table`):

* the slot is its instance's public row (`slotPub_table`, `digest_tables`);
* the setup ran `checkRegistered` (`RecIns.setup_registered`), so the port's table row opens the registered root at its
  position through its path (`checkPort_at`);
* it ran `checkStmt` (`RecIns.setup_checks`), whose `checkAcc` gives the port `acc0`'s registration in the part's
  shared rows, and whose `checkAccReads` puts unit `n`'s read at `accReads`' row of `n`, one on at `acc_out`
  (`accPort_of_checks`, `accReads_recOpen`).

The unit's label is taken as `n` (`hidx`, as `check5_bytes` takes it) and the part's reads file as present (`hown`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **A staged rec-open part's setup holds the verifier's reads file**: its unit is the verifier's `rec-open/l<L>-h<H>`
(`checkPart`'s name, `RecOpen.check` in the parse), and `RecCoins.check` refuses such a unit without `--registered`. -/
theorem registered_recOpen {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {r : XR}
    (hr : readX x = some r) {l LANES HH m : ℕ}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen l LANES HH m))
    {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) :
    ∃ own, i.base.registered = some own := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss, hss, hsv'⟩, h0p, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hsv
  obtain rfl : sv = .recOpen l LANES HH m := Option.some.inj (hsv'.symm.trans hsv)
  obtain ⟨-, hset⟩ := setupO_some hsu
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have h0 : 0 < sts.size := by subst hp; exact h0p
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0p
  have hpar := FlockVBridge.params_of_check P.recNet ((htab 0 h0p).1.trans (FlockVBridge.name_recOpen l LANES HH m))
  rw [e0] at hpar
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset) 0 h0
  cases ho : i.base.registered with
  | some own => exact ⟨own, rfl⟩
  | none =>
    exfalso
    rw [ho] at hcoin
    unfold RecCoins.check at hcoin
    simp [hpar] at hcoin

/-- **A rec-open part's `acc` (`acc_out`) slot at unit `n` is a registered `rec-acc` row at its chain position (the
next)**: at a staged `.recOpen l` part the witness checks against, with the setup's table labelling its instances in
order and the reads file `own`, port `nm`'s entry `reg` of `own` has `acc0`'s registration in the part's shared rows,
and the slot at unit `n` reads as the table row the setup opens against `reg`'s root, at row
`n / q · N + opensBelow l + n % q`, plus one at `acc_out`. -/
theorem accRow_table {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {l LANES HH m : ℕ}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen l LANES HH m))
    {sch : Schedule} (hsch : fast100 r.shape.m = some sch) {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own) {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) {n : ℕ}
    (hn : n < 2 * (sch.levels[l]!).q) (hnp : n < sts[0].pub.n) :
    ∃ reg rws a0, reg ∈ own ∧ reg.name = nm ∧ i.rows = some rws ∧ rws.find? (·.name == "acc0") = some a0 ∧
      FlockVBridge.Family.site reg = FlockVBridge.Family.site a0 ∧
      ∃ t positions paths, sts[0].pub.tables = some t ∧
        (List.range sts[0].c.ports.size).find? (fun p => sts[0].c.ports[p]!.name == reg.name) = some q ∧
        Registered.entry sts[0].pub reg = .ok (positions, paths) ∧
        Registered.ref t sts[0].c.ports.size q n < t.counts.getD q 0 ∧
        positions.getD (Registered.ref t sts[0].c.ports.size q n) 0 =
          n / (sch.levels[l]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l +
            n % (sch.levels[l]!).q + d ∧
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
  obtain ⟨ss, s', hss, hs', hacc, hrd⟩ := FlockC3.RegLevels.checkAcc_of_checkStmt hst
  rw [hss] at hsv
  obtain rfl : s' = .recOpen l LANES HH m := Option.some.inj (hs'.symm.trans hsv)
  have hl : RecVStar.accPorts (.recOpen l LANES HH m) = ["acc", "acc_out"] := rfl
  have hnm' : nm ∈ RecVStar.accPorts (.recOpen l LANES HH m) := by
    rw [hl]
    simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hnm
    rcases hnm with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> simp
  obtain ⟨reg, hfr, -⟩ := FlockC3.RegLevels.value_of_checkAcc hacc nm hnm'
  have hmem : reg ∈ own := Array.mem_of_find?_eq_some hfr
  have hname : reg.name = nm := by simpa using Array.find?_some hfr
  obtain ⟨⟨rws, a0, hrws, ha0, hsite⟩, hreads⟩ := FlockC3.RegLevels.accPort_of_checks hacc hrd hl
    (FlockC3.RegLevels.accReads_recOpen hsch l LANES HH m) hnm hmem hname
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
  rw [hreads, range_getD hnp, FlockC3.RegLevels.getElem?_accRow _ hn] at hrdn
  have hnn : n < p.sts[0].n := by rw [setupOne_n hd]; exact hnp
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hsl := slotPub_table ht' hd (by omega) hv1 L hz1 hq' hnp (div_lt_blocks hg0 hnn hnbl)
  rw [digest_tables htt] at hsl
  exact ⟨reg, rws, a0, hmem, hname, hrws, ha0, hsite, t, positions, paths, htt, hpp, hent, hlt,
    (Option.some.inj (Option.some.inj hrdn)).symm, hop, hsl⟩

end RecDeploy
