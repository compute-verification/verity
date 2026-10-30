import Proofs.Flock.Recursive.LinksOuter
import Proofs.Flock.Claims.Recursion.CapLowerPin

/-!
# The link part's rows at each instance

At a staged link part (V\*'s statement `.link UB PB N`, at a statement carrying the table `(T, N)`) the witness checks
against, each instance `n < N` fixes its rows:

* `rowOpens_staged`: its `trow`, the bytes at the row's input columns (`trowAt`), opens position `n` of the registered
  table at `T` (`RowOpens`). The `trow` slot is the instance's public row (`slotPub_table`), which the setup opens
  against the verifier's registered `trow` (`checkPort_at`), whose root, leaves, domain and reads `checkTable` fixed
  (`trow_table`); and the slot commits the row (`slotIn_stagedG`).
* `linkValue_slot`, `linkOut_bind`: its `rin` (`rout`) slot reads as check 5's row of `expected R` (`expected W`), over
  the inner statement's input (output) rows' root (`linkSetup_staged`), so the unit's output row is that row's message,
  or SHA-512 collides.
-/

namespace RecDeploy.Links

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

theorem bytes_eq_of_toList {a b : ByteArray} (h : a.data.toList = b.data.toList) : a = b := by
  cases a; cases b
  simp only [ByteArray.mk.injEq]
  exact Array.ext' h

/-! ## The link part's ports -/

/-- **V\*'s link part's row ports**: `trow`, `xyr`, `raux`, `xyw`, `waux`, then the outputs `rin`, `rout`
(`linkUnit_ok`'s input rows and output ports). -/
theorem linkRows_eq {UB PB : ℕ} {u : VStar.UnitC} (hu : VStar.linkUnit UB PB = .ok u) :
    RecVStar.vstarRows (VStar.linkSub UB PB) u =
      [("trow", 64, 1024), ("xyr", 128, 2048), ("raux", VStar.auxWords (UB + PB), 16 * VStar.auxWords (UB + PB)),
        ("xyw", 128, 2048), ("waux", VStar.auxWords (UB + PB), 16 * VStar.auxWords (UB + PB)),
        ("rin", 64, 1024), ("rout", 64, 1024)] := by
  obtain ⟨-, -, hin, hop⟩ := FlockVBridge.RecLink.linkUnit_ok hu
  unfold RecVStar.vstarRows
  rw [hin, hop]
  rfl

/-- **The link part's `trow` port is a row of `LINK_ROW_WORDS` words.** -/
theorem trow_port {c : Circuit} {UB PB : ℕ} {u : VStar.UnitC} (hu : VStar.linkUnit UB PB = .ok u)
    (hrows : c.ports.toList.map (fun q => (q.name, q.words, q.bits)) = RecVStar.vstarRows (VStar.linkSub UB PB) u) :
    ∃ q, c.ports.findIdx? (·.name == "trow") = some q ∧ ∃ hq : q < c.ports.size,
      c.ports[q].words = VStar.LINK_ROW_WORDS := by
  rw [linkRows_eq hu] at hrows
  obtain ⟨q, hq⟩ := CapLower.port_of_rows hrows (e := ("trow", 64, 1024)) (by simp)
  have hq : c.ports.findIdx? (·.name == "trow") = some q := hq
  obtain ⟨hqs, hm⟩ := CapLower.port_mem_rows hrows hq
  refine ⟨q, hq, hqs, ?_⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with h | h | h | h | h | h | h
  · exact (Prod.mk.inj (Prod.mk.inj h).2).1
  all_goals simp at h

/-! ## What the setup checked at the link part -/

/-- **A staged link part's setup**: at a part whose V\* statement is `.link UB PB N`, at a statement carrying the table
`(T, N)`, the setup's one table passed `checkLinks` at the verifier's registered reads `own` and the inner statement's
files `f`: `trow` reads the table (`checkTable`), `rin` and `rout` are the verifier's rows of `expected R` and
`expected W` over the roots of the inner statement's input and output rows, and the table labels its instances in
order and holds at least the `N` instances. -/
theorem linkSetup_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {r : XR}
    (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss) {UB PB N : ℕ}
    (hsv : ss[s]? = some (.link UB PB N)) {c : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    {T : ByteArray} (hl : r.w.links = some (T, N)) {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) :
    ∃ (h0 : 0 < sts.size) (own : Array Registered.Port) (f : RecCoins.Files) (sf : _) (R W : ByteArray),
      i.base.registered = some own ∧ i.src = some (.files f) ∧ f.stmt sts[0].c.unit.name = .ok sf ∧
      RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).1 = .ok R ∧
      RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows sf.2).2 = .ok W ∧
      RecLinkPart.checkTable sts[0].c sts[0].pub own T N = .ok () ∧
      (∃ what, RecCoins.checkValue sts[0].c sts[0].pub own "rin" RecLinkPart.IN_VALUE
        (Array.replicate N (RecLinkRef.expected R).toArray) what = .ok ()) ∧
      (∃ what, RecCoins.checkValue sts[0].c sts[0].pub own "rout" RecLinkPart.OUT_VALUE
        (Array.replicate N (RecLinkRef.expected W).toArray) what = .ok ()) ∧
      Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) ∧ N ≤ sts[0].n ∧
      sts[0].n = sts[0].pub.n := by
  obtain ⟨p, _, _, hx, -, hpi, _, -, h0p, -⟩ := stagedTables hS
  obtain ⟨ms, -, -, -, rfl, -, -⟩ := id hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  obtain ⟨-, hpopn⟩ := oneFull_staged hS hsu
  have h0 : 0 < sts.size := by subst hp; exact h0p
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  obtain ⟨-, hc0, -, hname, -, -, -, -, -, hpop⟩ := stagedLink hS hx h0p hr hss hsv hpc
  have hnm : sts[0].c.unit.name = VStar.linkName UB PB := by rw [← e0, hc0]; exact hname
  obtain ⟨hlk, -, -⟩ := linkCheck_staged hsu h0 hnm
  obtain ⟨T', N', own, f, sf, R, W, hlk', hown, hfs, htab, hsf, hR, hW, hrin, hrout⟩ := checkLinks_ok hlk
  rw [insO_staged_links hins hr, hl] at hlk'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hlk')
  obtain ⟨_, -, -, -, -, -, -, -, -, -, hidx⟩ := checkTable_ok htab
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨_, hset0⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset0
  rw [hpi] at hset0
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset0
  obtain ⟨_, hd, -⟩ := hall 0 h0p
  have hnn := setupOne_n hd
  rw [e0] at hnn hpop
  have hle := hpopn 0 h0
  exact ⟨h0, own, f, sf, R, W, hown, hfs, hsf, hR, hW, htab, hrin, hrout, hidx, by omega, hnn⟩

/-! ## `trow` opens the table -/

/-- **A link part's `trow` slot at instance `n` is a row of the registered table**: at a staged `.link UB PB N` part
the witness checks against, at a statement carrying the table `(T, N)`, the verifier's registered `trow` has root `T`,
`N` leaves, the row schema and the domain the part's program registers (`checkTable_ok`), and the slot at instance
`n < N` reads as a table row that opens it at position `n` (`checkPort_at`, its reads `n ↦ n`). -/
theorem trow_table {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {UB PB N : ℕ} (hsv : ss[s]? = some (.link UB PB N))
    {T : ByteArray} (hl : r.w.links = some (T, N)) {q : ℕ} (hq : c.ports.findIdx? (·.name == "trow") = some q)
    {n : ℕ} (hn : n < N) :
    ∃ reg : Registered.Port, reg.root = T ∧ reg.leaves = N ∧ reg.schema = HmRow.ROW_SCHEMA ∧
      tableDomain c N = some reg.domain ∧ ∃ bc path, Registered.opens reg bc n path = true ∧
        readForms w (hmCommit c q n) = bc.data.toList := by
  obtain ⟨ms, i, spec, sts, rfl, hsu, -⟩ := id hS
  obtain ⟨h0, own, _, _, _, _, hown, -, -, -, -, htab', -, -, hidx, hNn, hnn⟩ :=
    linkSetup_staged hS hr hss hsv hpc hl hsu
  obtain ⟨p, _, _, hx, ht, hpi, _, -, h0p, htab⟩ := stagedTables hS
  obtain ⟨-, hset⟩ := setupO_some hsu
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
  obtain ⟨_, hd, -⟩ := hall 0 h0p
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0p
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0p).2.2
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L _ hz1
  subst hc0
  obtain ⟨reg, hfr, -, hroot, hleaves, -, -, hschema, hdom, hreads, -⟩ := checkTable_ok htab'
  have hmem : reg ∈ own := Array.mem_of_find?_eq_some hfr
  have hrn : reg.name = "trow" := by simpa using Array.find?_some hfr
  have hreg := RecIns.setup_registered hset sts[0] (Array.getElem_mem h0)
  rw [hown] at hreg
  obtain ⟨t, idx, htt, hidx2, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hreg)
  rw [hidx] at hidx2
  cases hidx2
  obtain ⟨pp, positions, paths, hpp, hent, hall'⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  rw [← e0] at hpp hall' htt hdom hNn hnn
  rw [hrn] at hpp
  obtain rfl : pp = q := find_range_findIdx (f := fun (x : Port) => x.name == "trow") hpp hq
  have hnN : n < p.sts[0].n := by omega
  have hnp : n < p.sts[0].pub.n := by omega
  obtain ⟨-, hrdn, hop⟩ := hall' n hnp
  have hrN : ((Array.range N).map some)[n]? = some (some n) := by simp [hn]
  rw [hreads, range_getD hnp, hrN] at hrdn
  rw [← Option.some.inj (Option.some.inj hrdn)] at hop
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hsl := slotPub_table ht' hd (by omega) hv1 L hz1 hq' hnp (div_lt_blocks hg0 hnN hnbl)
  rw [digest_tables htt] at hsl
  exact ⟨reg, hroot, hleaves, hschema, hdom, _, _, hop, hsl⟩

/-- **A link part's `trow` at instance `n` opens the table at `T`**: at a staged `.link UB PB N` part the witness
checks against, at a statement carrying the table `(T, N)`, the bytes at instance `n`'s `trow` input columns
(`trowAt`) open position `n` of the registered table at `T` under the domain its program registers (`RowOpens`): the
slot commits the row (`slotIn_stagedG`, `trow` being an input of `linkSub`), and its commit string is the table row
`trow_table` opens. -/
theorem rowOpens_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {UB PB N : ℕ} (hsv : ss[s]? = some (.link UB PB N))
    {T : ByteArray} (hl : r.w.links = some (T, N)) {dom : ByteArray} (hdom : tableDomain c N = some dom) {n : ℕ}
    (hn : n < N) : RowOpens dom T N n (trowAt c w n) := by
  obtain ⟨u, hu, hrows⟩ := CapLower.rows_staged hS hr hss hsv hpc
  have hu : VStar.linkUnit UB PB = .ok u := hu
  obtain ⟨q, hq, hqs, hk⟩ := trow_port hu hrows
  have hslot : readForms w (hmCommit c q n) =
      FlockSoundness.Binding.commitString (rowPfx c q ++ trowAt c w n) (readForms w (hmSalt c q n)) :=
    MoreSlots.slotIn_stagedG hS hck hr hss hsv hpc hq (show "trow" ∉ ["rin", "rout"] by decide) (.inl hn)
  obtain ⟨reg, hroot, hleaves, hschema, hd, bc, path, hop, hbc⟩ := trow_table hS hck hr hpc hss hsv hl hq hn
  have hdm : reg.domain = dom := Option.some.inj (hd.symm.trans hdom)
  have hmsg : rowMsgOf (trowAt c w n) = ⟨(rowPfx c q ++ trowAt c w n).toArray⟩ := by
    rw [CapLower.rowPfx_of_words hqs hk]
    try rfl
  have hb : Hm96.Default512.commitString (Sha512.hash (rowMsgOf (trowAt c w n)))
      ⟨(readForms w (hmSalt c q n)).toArray⟩ = bc := by
    apply bytes_eq_of_toList
    rw [hmsg, ResLower.commitString_exec, ← hbc, hslot]
    try rfl
  refine ⟨⟨(readForms w (hmSalt c q n)).toArray⟩, path, ?_⟩
  rw [← opens_tablePort hroot hleaves hdm hschema, hb]
  exact hop

/-! ## `rin` and `rout` are check 5's rows -/

/-- **Check 5's row at a link part's output slot**: where the setup's table passed `checkValue` at port `port` over
`N` copies of the row `E`, the part's port `q` (its `port`)'s hm96 slot at instance `n < N` reads as the commit string
of `E` under the verifier's salt of instance `n` (`check5_bytes_lt`). -/
theorem linkValue_slot {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own) {port value : String} {E : Array ℕ}
    {N : ℕ} {what : ℕ → String}
    (hv : RecCoins.checkValue sts[0].c sts[0].pub own port value (Array.replicate N E) what = .ok ())
    {q : ℕ} (hq : c.ports.findIdx? (·.name == port) = some q) {n : ℕ} (hn : n < N) (hNn : N ≤ sts[0].n)
    (hnn : sts[0].n = sts[0].pub.n) :
    ∃ prog, sts[0].c.metaJ.getObjValAs? String "program_sha512" = .ok prog ∧
      readForms w (hmCommit c q n) = (RecCoins.commitRow E (RecCoins.verifierSalt prog value n)).data.toList := by
  have hlt : n < (Array.replicate N E).size := by rw [Array.size_replicate]; exact hn
  obtain ⟨prog, hprog, h⟩ := ResLower.check5_bytes_lt hS hck hr hpc hsu h0 hidx hown
    (salt := fun prog => RecCoins.verifierSalt prog value) hv hq (by omega) hlt (by omega)
  refine ⟨prog, hprog, ?_⟩
  rw [h, getElem!_pos (Array.replicate N E) n hlt, Array.getElem_replicate]

/-- **An output slot check 5 holds to a row carries that row**: where port `q`'s slot at instance `n` commits the
row at its output columns (`slotOut_stagedG`) and reads as check 5's commit string of `E` under `salt`
(`linkValue_slot`), the slot's `sha512/row/v1` message is `E`'s, or the two messages' or the two salts' evaluations
collide (`commit_bind`). -/
theorem linkOut_bind {c : Circuit} {q n : ℕ} {w : Claims.Flock.Wit} {E : Array ℕ} {salt : ByteArray}
    (hout : readForms w (hmCommit c q n) = FlockSoundness.Binding.commitString
      (rowPfx c q ++ readForms w (DeployedR.outRowForms c q n)) (readForms w (hmSalt c q n)))
    (hv : readForms w (hmCommit c q n) = (RecCoins.commitRow E salt).data.toList) :
    rowPfx c q ++ readForms w (DeployedR.outRowForms c q n) =
        ResLower.rowInput (RecCoins.padTo E (Registered.rowWords E.size 16)) ∨
      Claims.Collision FlockDeploy.sha512 (rowPfx c q ++ readForms w (DeployedR.outRowForms c q n))
        (ResLower.rowInput (RecCoins.padTo E (Registered.rowWords E.size 16))) ∨
      Claims.Collision FlockDeploy.sha512 (FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n))
        (FlockSoundness.Refine.K512.saltPrefix ++ salt.data.toList) := by
  apply commit_bind
  rw [← hout, hv, RecCoins.commitRow, ResLower.rowDigest_eq, ResLower.commitString_exec]

end RecDeploy.Links
