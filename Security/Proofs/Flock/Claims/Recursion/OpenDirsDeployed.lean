import Proofs.Flock.Claims.Recursion.ResLowerDeployed
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.C3.RegLevels.OpensRegs

/-!
# Check 5's `dirs` tie at the deployed run

`CopyDirsR` asks that a `rec-open` piece's copy hold, at its position columns (`idxCol`), the `dirs` row of the
position the verifier drew (`DirsTie`). Check 5 (`RecCoins.checkOpen`, at the statement's recursive setup) holds each
instance's public commit string of port `dirs` to `commitRow` of the verifier's own row `openDirs` under `verifierSalt`,
and `layVR` records the part's hm96 calls at `dirs` (`openPorts`, `portCalls_sub_layVR_open`), whose verifier
evaluations are `openSetupSt`. So `port_tie` gives the part's `dirs` bytes as the verifier's row, at every instance of
the table, not only the first two.

* `check5_bytes_lt`: `check5_bytes` at any instance of the table (`div_lt_blocks`).
* `slotDirs_staged`: the `dirs` slot commits at a staged level part.
* `dirsCols_staged`: port `dirs` is row 4 of the `rec-open` unit (`rowsOk`), from column `idxCol LANES HH 0` of each copy
  (`preBits`, `inGroups_of_laid`), so its forms are the copy's position columns.
* `openPos`, `openRows_dirs`: the position `openDirs` registers at instance `i`.
* `dirsTie_staged`: the tie, at the position `openPos` reads off the inner session check 5 loads.
-/

namespace RecDeploy.OpenDirs

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open ResLower

noncomputable section

/-! ## The `rec-open` unit's `dirs` port -/

/-- **Port `dirs` is the circuit's row 4**: the unit's first five rows are `row`, `aux`, `acc`, `coef`, `dirs`
(`rowsOk`). -/
theorem findIdx_dirs {c : Circuit} {L H : ℕ} (h : RecOpen.rowsOk c L H = true) :
    c.ports.findIdx? (·.name == "dirs") = some 4 := by
  have hnm : ∀ j < 5,
      ((Flock.RecOpen.rows L H)[j]!).1 = (#["row", "aux", "acc", "coef", "dirs"] : Array String)[j]! := by
    intro j hj
    interval_cases j <;> rfl
  obtain ⟨h4, hn4, -⟩ := FlockC3.RegLevels.rowsOk_get h (show 4 < 5 by decide)
  refine Array.findIdx?_eq_some_iff_getElem.2 ⟨h4, ?_, fun j hj => ?_⟩
  · exact (congrArg (· == "dirs") (hn4.trans (hnm 4 (by decide)))).trans (by decide)
  · obtain ⟨hj', hnj, -⟩ := FlockC3.RegLevels.rowsOk_get h (show j < 5 by omega)
    intro hc
    have e := (congrArg (· == "dirs") (hnj.trans (hnm j (by omega)))).symm.trans hc
    interval_cases j <;> exact absurd e (by decide)

/-- **Port `dirs` starts at the position columns**: the words of the four rows before it are `idxCol`'s offset. -/
theorem words_pre {c : Circuit} {L H : ℕ} (h : RecOpen.rowsOk c L H = true) :
    16 * (c.ports.extract 0 4).foldl (· + ·.words) 0 = OpenLower.idxCol L H 0 := by
  have hpre : ∀ i ≤ 5, 16 * ((c.ports.toList.take i).map (·.words)).sum = FlockC3.RegLevels.preBits L H i := by
    intro i hi
    induction i with
    | zero => simp [FlockC3.RegLevels.preBits]
    | succ i ih =>
      obtain ⟨hi', -, hwi, -⟩ := FlockC3.RegLevels.rowsOk_get h (show i < 5 by omega)
      have hB : i < (Flock.RecOpen.ports L H).bits.toList.length := by
        simp [Flock.RecOpen.Ports.bits]
        omega
      have hBi : (Flock.RecOpen.ports L H).bits.toList[i] = (Flock.RecOpen.ports L H).bits[i]! := by
        rw [Array.getElem_toList, getElem!_pos _ i (by simpa using hB)]
      unfold FlockC3.RegLevels.preBits at ih ⊢
      rw [List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem (by simpa using hi'),
        List.take_add_one, List.sum_append, List.getElem?_eq_getElem hB, hBi, ← hwi, ← ih (by omega)]
      simp [Nat.mul_add]
  have hst : (c.ports.extract 0 4).foldl (· + ·.words) 0 = ((c.ports.toList.take 4).map (·.words)).sum :=
    extract_words (·.words) c.ports 4
  rw [hst, hpre 4 (by omega), FlockC3.RegLevels.preBits_dirs]
  simp [OpenLower.idxCol, Flock.RecOpen.ports]

theorem idxCol_add (L H h : ℕ) : OpenLower.idxCol L H h = OpenLower.idxCol L H 0 + h := by
  simp [OpenLower.idxCol]

/-- **Where a copy's `dirs` columns sit**: at a staged `rec-open` part, port `dirs` is row 4 (`findIdx_dirs`) of 64
words, and instance `i`'s `dirs` row (`rowHidden_eq`) is copy `(i / g, i % g, 0)`'s columns from `idxCol LANES HH 0`
(`words_pre`, the unit's one input group at column 0: `inGroups_of_laid`), below the unit's constant
(`leaves_le_constPos`). -/
theorem dirsCols_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size) (hdin : 4 < HmOut.nIn c) (i : ℕ) :
    c.ports.findIdx? (·.name == "dirs") = some 4 ∧ ∃ pt : Port, c.ports[4]? = some pt ∧ pt.words = 64 ∧
      ∀ h < 1024, ∃ hl : h < (rowForms c "dirs" 4 i).length,
        Flock.Readable.copyOf p 0 ht w (i / c.g) (i % c.g) 0 (OpenLower.idxCol LANES HH h) =
          formBit w (rowForms c "dirs" 4 i)[h] := by
  obtain ⟨p', r', sv, hx', ht', hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hx] at hx'
  cases hx'
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i0.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have hc0 := partCircuit_staged hS hx ht' hpi hr hpc h0
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht' hpi h0
  obtain ⟨u, -, hupv, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  have hn : p.sts[0].c.unit.name = Flock.RecOpen.unitName LANES HH := (htab 0 h0).1
  have hpar := FlockVBridge.params_of_check P.recNet hn
  obtain ⟨hrk, -⟩ := FlockC3.RegLevels.check_rows P.recNet (.inl hpar)
  obtain ⟨-, hlaid⟩ := Flock.RecOpen.check_ok P.recNet hpar
  have hg := FlockC3.RegLevels.inGroups_of_laid hlaid
  subst hc0
  have hq4 := findIdx_dirs hrk
  obtain ⟨hqs, -, hw4, hb4⟩ := FlockC3.RegLevels.rowsOk_get hrk (show 4 < 5 by decide)
  have hd : (Flock.RecOpen.ports LANES HH).bits[4]! = 1024 := rfl
  have hbits : p.sts[0].c.ports[4].bits = 1024 := by rw [hb4, hd]
  have hwords : p.sts[0].c.ports[4].words = 64 := by omega
  have hout : "dirs" ∉ (VStar.Stmt.recOpen lv LANES HH n).sub.outputs := by
    show "dirs" ∉ ["out", "acc_out"]
    decide
  have hN := in_block hrows hout hq4
  rw [getElem!_pos p.sts[0].c.ports 4 hqs] at hN
  have hrh := rowHidden_eq i hl hq4 hqs hN
  have hpre := words_pre hrk
  have hcol0 : (p.sts[0].c.unit.inGroups.getD 0 default).col = 0 := by rw [hg]; rfl
  refine ⟨hq4, p.sts[0].c.ports[4], Array.getElem?_eq_getElem hqs, hwords, fun h hh => ?_⟩
  have hcp := leaves_le_constPos P hl (by omega)
  rw [hcol0, Nat.zero_mul, Nat.zero_add] at hcp
  have hlen : (rowForms p.sts[0].c "dirs" 4 i).length = 1024 := by simp [rowForms, hdin, hrh, hbits]
  refine ⟨by omega, ?_⟩
  have hel : (rowForms p.sts[0].c "dirs" 4 i)[h]'(by omega) = HmNets.Lin.var (i / p.sts[0].c.g * 2 ^ p.sts[0].c.kLog +
      (p.sts[0].c.unitRange.slot (i % p.sts[0].c.g * p.sts[0].c.upv) +
        ((p.sts[0].c.unit.inGroups.getD 0 default).col * Flock.WORD +
          16 * (p.sts[0].c.ports.extract 0 4).foldl (· + ·.words) 0 + h))) := by
    simp [rowForms, hdin, hrh]
  have hix : OpenLower.idxCol LANES HH h = 16 * (p.sts[0].c.ports.extract 0 4).foldl (· + ·.words) 0 + h := by
    rw [hpre, idxCol_add]
  have hj : OpenLower.idxCol LANES HH h ≤ p.sts[0].c.unit.constPos := by rw [hix]; omega
  rw [hel, formBit_var, hcol0, Nat.zero_mul, Nat.zero_add, ← hix, Claims.Recursion.VStar.copyOf_eq hj]
  simp [Claims.Recursion.VStar.copyIdx, hsz, hupv]

/-! ## The `dirs` slot commits, at every instance -/

/-- **A level part's `dirs` slot commits**: `dirs` is an input of the `rec-open` unit (its outputs are `out` and
`acc_out`), so its row lies in the table's leaves (`in_block`); each instance `i < n` is one of the table's
(`staged_scope`, `oneFull_staged`), so in one of its blocks (`div_lt_blocks`), and `slotCommits_table` applies. -/
theorem slotDirs_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    {qd i : ℕ} (hqd : c.ports.findIdx? (·.name == "dirs") = some qd) (hi : i < n) :
    SlotCommits c "dirs" qd i w := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i0.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have h0' : 0 < sts.size := by subst hp; exact h0
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hle : Flock.RecVStar.population p.sts[0] ≤ p.sts[0].n := by
    rw [e0]
    exact (oneFull_staged hS hsu).2 0 h0'
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨L, regs, hz⟩ := Flock.Readable.layout_of_check hck hx ht hpi h0
  have hz1 : (L.model regs).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, r'', ss'', sv'', hr'', hss'', hsv'', hpop⟩ := staged_scope hS hx h0
  rw [hr] at hr''
  cases hr''
  rw [hss] at hss''
  cases hss''
  rw [hsv] at hsv''
  cases hsv''
  have hpop' : Flock.RecVStar.population p.sts[0] = n := hpop
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨u, -, hupv, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨-, hg, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L regs hz1
  have hb : i / p.sts[0].c.g < 2 ^ p.sts[0].nbl := div_lt_blocks hg (show i < p.sts[0].n by omega) hnbl
  subst hc0
  have hout : "dirs" ∉ (VStar.Stmt.recOpen lv LANES HH n).sub.outputs := by
    show "dirs" ∉ ["out", "acc_out"]
    decide
  exact slotCommits_table ht' hd (by omega) hv1 hl L regs hz1 hqd (in_block hrows hout hqd) hb

/-! ## Check 5's public rows, at every instance -/

/-- **Check 5's row at a slot, at any instance of the table**: `check5_bytes` with the instance's block from the table's
instance count (`div_lt_blocks`) rather than `n < 2`. -/
theorem check5_bytes_lt {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own)
    {salt : String → ℕ → ByteArray} {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (hrows : RecCoins.checkRows salt sts[0].c sts[0].pub own port value rows what = .ok ())
    {q : ℕ} (hq : c.ports.findIdx? (·.name == port) = some q) {n : ℕ} (hnN : n < sts[0].n) (hn : n < rows.size)
    (hnp : n < sts[0].pub.n) :
    ∃ prog, sts[0].c.metaJ.getObjValAs? String "program_sha512" = .ok prog ∧
      readForms w (hmCommit c q n) = (RecCoins.commitRow rows[n]! (salt prog n)).data.toList := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', -, h0p, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
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
  obtain ⟨d, hd, -⟩ := hall 0 h0p
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0p
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0p).2.2
  obtain ⟨-, hg, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L _ hz1
  subst hc0
  have hreg := RecIns.setup_registered hset sts[0] (Array.getElem_mem h0)
  rw [← e0] at hrows hidx hnp hnN hreg ⊢
  rw [hown] at hreg
  obtain ⟨reg, prog, t, pp, positions, paths, hf, hreads, hprog, htt, hpp, hent, hrow⟩ := checkRows_tableRows hrows
  have hmem : reg ∈ own := Array.mem_of_find?_eq_some hf
  have hname : reg.name = port := by simpa using Array.find?_some hf
  obtain ⟨t', idx, ht2, hidx2, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hreg)
  rw [htt] at ht2
  cases ht2
  rw [hidx] at hidx2
  cases hidx2
  obtain ⟨p', positions', paths', hp', hent', hall'⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  obtain ⟨p'', positions'', paths'', hp'', hent'', hsize⟩ := FlockC3.PubRows.checkPort_size (hports _ hmem)
  rw [hname, hpp] at hp' hp''
  cases hp'
  cases hp''
  rw [hent] at hent' hent''
  cases hent'
  cases hent''
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hqq : pp = q := find_range_findIdx (f := fun (x : Port) => x.name == port) hpp hq
  rw [hqq] at hrow hall' hsize
  obtain ⟨hlt, hrd, -⟩ := hall' n hnp
  rw [hreads, range_getD hnp] at hrd
  have hpos : positions.getD (Registered.ref t p.sts[0].c.ports.size q n) 0 = n := by
    rw [Array.getD_eq_getD_getElem?]
    simp [hn] at hrd
    omega
  have hlt' : Registered.ref t p.sts[0].c.ports.size q n < positions.size := by rw [hsize]; exact hlt
  have hsl := slotPub_table ht' hd (by omega) hv1 L hz1 hq' hnp (div_lt_blocks hg hnN hnbl)
  rw [digest_tables htt, hrow _ hlt', getElem!_pos positions _ hlt'] at hsl
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt', Option.getD_some] at hpos
  rw [hpos] at hsl
  refine ⟨prog, hprog, ?_⟩
  rw [hsl]
  simp [Array.getD_eq_getD_getElem?, hn]

/-! ## The position check 5 registers -/

/-- **The position of instance `i`** of a `rec-open` statement at `LANES` and `H`, at the inner session `inn`: rep
`i / q`'s opening `i % q`'s query position at the one level `innerLevels` finds (`openDirs`' order). -/
def openPos (inn : RecCoins.Inner) (LANES H i : ℕ) : ℕ :=
  match inn.algebra with
  | .error _ => 0
  | .ok (sh, extras) => match FlockSoundness.Discharge.Recursion.innerLevels sh LANES H with
    | [l] => match inn.rep sh extras 0, inn.rep sh extras 1 with
      | .ok r0, .ok r1 =>
        ((if i < sh.sched.levels[l]!.q then r0 else r1).info.lg.conss[l]!).pos[i % sh.sched.levels[l]!.q]!
      | _, _ => 0
    | _ => 0

/-- **Check 5's `dirs` row `i` is the `dirsRow` of `openPos`.** -/
theorem openRows_dirs {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (h : FlockSoundness.Discharge.Recursion.openRows inn LANES H = some (cr, dr)) {i : ℕ} (hi : i < dr.size) :
    dr[i]! = RecCoins.dirsRow (openPos inn LANES H i) H := by
  unfold FlockSoundness.Discharge.Recursion.openRows at h
  unfold openPos
  cases ha : inn.algebra with
  | error e => rw [ha] at h; simp at h
  | ok v =>
    obtain ⟨sh, extras⟩ := v
    rw [ha] at h
    simp only at h ⊢
    cases hl : FlockSoundness.Discharge.Recursion.innerLevels sh LANES H with
    | nil => rw [hl] at h; simp at h
    | cons l t =>
      cases t with
      | cons _ _ => rw [hl] at h; simp at h
      | nil =>
        rw [hl] at h
        simp only at h ⊢
        cases h0 : inn.rep sh extras 0 with
        | error _ => rw [h0] at h; simp at h
        | ok r0 =>
          cases h1 : inn.rep sh extras 1 with
          | error _ => rw [h0, h1] at h; simp at h
          | ok r1 =>
            rw [h0, h1] at h
            simp only [Option.some.injEq, Prod.mk.injEq] at h
            obtain ⟨-, rfl⟩ := h
            simp only [FlockSoundness.Discharge.Recursion.openDirs, Array.size_map, Array.size_range] at hi ⊢
            exact (FlockC3.RegLevels.getElem!_map_range _ hi).trans rfl

theorem dirsRow_size (pos H : ℕ) : (RecCoins.dirsRow pos H).size = 64 := by
  simp [RecCoins.dirsRow, RecCoins.DIRS_WORDS]

theorem getD_of_lt {a : Array ℕ} {j : ℕ} (hj : j < a.size) : a.getD j 0 = a[j]! := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some, getElem!_pos a j hj]

/-! ## The tie -/

/-- **Check 5's `dirs` tie at a level part's instance, at `layVR`**: at a staged `rec-open` part whose calls at
`layVR` are true and collision-free with the verifier's evaluations, and whose check 5 runs `checkOpen` (`hso`, `hop`,
`hsp`: the unit is no other recursive unit), the inner session check 5 loads gives `dirs` rows `dr`, and at each
instance `i < n` of them, the copy `(i / g, i % g, 0)` holds `dirsRow` of the position `openPos` reads there
(`DirsTie`): `port_tie` at the `dirs` slot (`slotDirs_staged`) and check 5's public row (`check5_bytes_lt`), with the
verifier's evaluations of it in `openSetupSt` (`hset`), and the slot's forms the copy's position columns
(`dirsCols_staged`). -/
theorem dirsTie_staged {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit}
    (hT : ∀ e ∈ (layVR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((layVR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (layVR elem) IV) xO)).isSome = false)
    (hS : StagedPart b.st s x xO) (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {i0 : RecIns} {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i0, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i0.base.registered = some own)
    (hso : RecCoins.stackedParams sts[0].c.unit.name = none) (hop : RecOpening.params sts[0].c.unit.name = none)
    (hsp : RecSparse.params sts[0].c.unit.name = none)
    {f : RecCoins.Source} (hf : RecCoins.Source.atC3 i0.src i0.inner i0.commits = some f)
    (hset : ∀ u ∈ resSetup xO, u ∈ b.setup xO) (hdin : 4 < HmOut.nIn c) {i : ℕ} (hi : i < n)
    (hnp : i < sts[0].pub.n) :
    ∃ (inn : RecCoins.Inner) (cr dr : Array (Array ℕ)), f.load sts[0].c.unit.name = .ok inn ∧
      FlockSoundness.Discharge.Recursion.openRows inn LANES HH = some (cr, dr) ∧
      (i < dr.size → OpenLower.DirsTie (Flock.Readable.copyOf p 0 ht w (i / c.g) (i % c.g) 0) LANES HH
        (openPos inn LANES HH i)) := by
  obtain ⟨p', r', sv, hx', htyp, hpi, hr', ⟨ss', hss', hsv'⟩, h0p, htab⟩ := stagedTables hS
  rw [hx] at hx'
  cases hx'
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hp : p = { i0.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx htyp hpi h0p
  have hn : p.sts[0].c.unit.name = Flock.RecOpen.unitName LANES HH := (htab 0 h0p).1
  have hpar := FlockVBridge.params_of_check P.recNet hn
  rw [e0] at hpar
  obtain ⟨-, r'', ss'', sv'', hr'', hss'', hsv'', hpop⟩ := staged_scope hS hx h0p
  rw [hr] at hr''
  cases hr''
  rw [hss] at hss''
  cases hss''
  rw [hsv] at hsv''
  cases hsv''
  have hpop' : Flock.RecVStar.population p.sts[0] = n := hpop
  have hle : Flock.RecVStar.population sts[0] ≤ sts[0].n := (oneFull_staged hS hsu).2 0 h0
  rw [e0] at hpop'
  have hnN : i < sts[0].n := by omega
  obtain ⟨-, hset0⟩ := setupO_some hsu
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset0) 0 h0
  rw [hown, hf] at hcoin
  obtain ⟨inn, hinn, hco⟩ := FlockSoundness.Discharge.Recursion.openCheck_of_check hcoin hso hop hsp hpar
  obtain ⟨cr, dr, hrows, -, what, hdv⟩ := FlockSoundness.Discharge.Recursion.open_of_check hco
  refine ⟨inn, cr, dr, hinn, hrows, fun hnr => ?_⟩
  have hHH : HH ≤ 32 := OpenLower.hh_le_staged hr hss hsv hx hS ht
  obtain ⟨hqd, ptd, hptd, hwords, hcols⟩ := dirsCols_staged (w := w) hS hr hss hsv hpc hx ht hdin i
  obtain ⟨prog, hprog, hpv⟩ := check5_bytes_lt hS hck hr hpc hsu h0 hidx hown hdv hqd hnN hnr hnp
  have hst : openSetupSt (RecCoins.Source.atC3 i0.src i0.inner i0.commits) sts[0] =
      commitIns dr (RecCoins.verifierSalt prog RecCoins.DIRS_VALUE) := by
    unfold openSetupSt
    simp only [hpar, hf, hinn, hprog, Except.toOption, hrows]
  have hmemS : ∀ u ∈ openSetupSt (RecCoins.Source.atC3 i0.src i0.inner i0.commits) sts[0], u ∈ b.setup xO :=
    fun u hu => hset u (by
      unfold resSetup; rw [hsu]; exact List.mem_flatMap.2 ⟨sts[0], by simp, List.mem_append_right _ hu⟩)
  rw [hst] at hmemS
  have hci : rowInput (RecCoins.padTo dr[i]! (Registered.rowWords dr[i]!.size 16)) ∈
        commitIns dr (RecCoins.verifierSalt prog RecCoins.DIRS_VALUE) ∧
      (Hm96.Default512.saltPrefix ++ RecCoins.verifierSalt prog RecCoins.DIRS_VALUE i).data.toList ∈
        commitIns dr (RecCoins.verifierSalt prog RecCoins.DIRS_VALUE) := by
    unfold commitIns
    exact ⟨List.mem_flatMap.2 ⟨i, List.mem_range.2 hnr, List.mem_cons_self⟩,
      List.mem_flatMap.2 ⟨i, List.mem_range.2 hnr, List.mem_cons_of_mem _ List.mem_cons_self⟩⟩
  have hfd : rowForms c "dirs" 4 i = ((rowHidden c "dirs" i).getD []).map HmNets.Lin.var := by
    unfold rowForms
    rw [ite_eq_left hdin]
  have htie := port_tie rfl hT hnc
    (portCalls_sub_layVR_open elem hr hss hsv hpc (by simp [openPorts]) hqd hi)
    (by rw [hfd]; exact slotDirs_staged hS hck hr hss hsv hpc hqd hi) hpv.symm
    (hmemS _ hci.1) (hmemS _ hci.2)
  have hdr := openRows_dirs hrows hnr
  have hwd : ptd.words = (RecCoins.padTo dr[i]! (Registered.rowWords dr[i]!.size 16)).size := by
    rw [hdr, hwords]
    simp [RecCoins.padTo, Registered.rowWords, dirsRow_size]
  have hrb := readForms_of_rowInput hptd hwd htie
  intro h hh
  obtain ⟨hl, hcol⟩ := hcols h (by omega)
  have hb16 : h < 16 * (RecCoins.padTo dr[i]! (Registered.rowWords dr[i]!.size 16)).toList.length := by
    rw [Array.length_toList, ← hwd, hwords]
    omega
  have hj : h / 16 < (RecCoins.dirsRow (openPos inn LANES HH i) HH).size := by rw [dirsRow_size]; omega
  rw [hcol, ← bitL_readForms hl, hrb, wordBytes, bitL_wordBytes _ h hb16, List.getD_eq_getElem?_getD,
    Array.getElem?_toList, ← Array.getD_eq_getD_getElem?, getD_padTo, hdr, getD_of_lt hj]

/-! ## `CopyDirsTieR` at `layVR` -/

/-- Piece `q`'s instance of its level's table: rep `r`'s opening `j` is `r · q + j` (`copyAtR`). -/
def instR (xI : Msg) (q : OpenPiece) : ℕ := q.rep * OpenLower.qAt xI q.level + q.query

/-- The deployed record over `layVR elem`. -/
abbrev DR (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (IV : Claims.Verifier Coin) : Claims.Recursion.Deploy Coin :=
  recBound _root_.Flock.Readable.relation b (layVR elem) IV

/-- **What `dirsTie_staged` takes beyond `StagedAt`**, at each piece a level part owns, at its instance `instR`: the
part's circuit, with `dirs` among its inputs; the staged setup's registered rows (`hidx`, `hown`) and check 5's source
(`hf`), its unit no other recursive unit (`hso`, `hop`, `hsp`), the setup listing `openSetupSt` (`hset`); the
instance among the statement's, the public rows' and check 5's `dirs` rows; and `posAt` the position check 5 registers
there (`openPos`, the coin tie). -/
def DirsSideR (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  OpenLower.AtPieces FlockDeploy.sha512 (DR b elem IV) shape untr b cmt (layVR elem) pieces
    fun x cs _ LANES HH s xO _ _ _ q =>
    ∃ (cc : Circuit) (i0 : RecIns) (spec : SessionSpec) (sts : Array Stmt) (own : Array Registered.Port)
      (f : RecCoins.Source),
      (readX x).bind (fun r => r.w.parts[s]?.bind partCircuit) = some cc ∧ 4 < HmOut.nIn cc ∧
      setupO xO = some (i0, spec, sts) ∧ i0.base.registered = some own ∧
      RecCoins.Source.atC3 i0.src i0.inner i0.commits = some f ∧ (∀ u ∈ resSetup xO, u ∈ b.setup xO) ∧
      Registered.unitIndices sts[0]!.pub = .ok (Array.range sts[0]!.pub.n) ∧
      RecCoins.stackedParams sts[0]!.c.unit.name = none ∧ RecOpening.params sts[0]!.c.unit.name = none ∧
      RecSparse.params sts[0]!.c.unit.name = none ∧
      (∀ (r : XR) (ss : Array VStar.Stmt) (lv n : ℕ), readX x = some r → RecVStar.statements r.shape = .ok ss →
        ss[s]? = some (.recOpen lv LANES HH n) → instR (Claims.Recursion.xI (DR b elem IV) x) q < n) ∧
      instR (Claims.Recursion.xI (DR b elem IV) x) q < sts[0]!.pub.n ∧
      ∀ (inn : RecCoins.Inner) (cr dr : Array (Array ℕ)), f.load sts[0]!.c.unit.name = .ok inn →
        FlockSoundness.Discharge.Recursion.openRows inn LANES HH = some (cr, dr) →
        instR (Claims.Recursion.xI (DR b elem IV) x) q < dr.size ∧
        posAt (Claims.Recursion.xI (DR b elem IV) x)
            ((List.range (Bd.DE.V.k (Claims.Recursion.xI (DR b elem IV) x))).map cs) q =
          openPos inn LANES HH (instR (Claims.Recursion.xI (DR b elem IV) x) q)

/-- **`CopyDirsTieR` at the deployed layout `layVR`**, from `DirsSideR`: `dirsTie_staged` at each piece's instance
`instR`, whose block and VU are `copyAtR`'s (`partCircuit_staged`: the part's circuit is the staged table's). -/
theorem copyDirsTieR_layVR {b : Brings} {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {IV : Claims.Verifier Coin} {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}
    {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hside : DirsSideR b elem IV shape untr cmt Bd pieces posAt) :
    OpenLower.CopyDirsTieR FlockDeploy.sha512 (DR b elem IV) shape untr b cmt (layVR elem) Bd pieces posAt := by
  intro x st hx hf cs ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  obtain ⟨cc, i0, spec, sts, own, f, hpc, hdin, hsu, hown, hf', hset, hidx, hso, hop, hsp, hin, hnp, hrows⟩ :=
    hside x st hx hf cs ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  rw [hr] at hpc
  change r.w.parts[s]?.bind partCircuit = some cc at hpc
  obtain ⟨hxp, -, -, -, hT, hnc, -, hck, hS⟩ := hst
  have hp : p = { i0.base.read with sts } := Option.some.inj (hxp.symm.trans (parse_of_setupO hsu))
  have h0 : 0 < sts.size := by subst hp; exact ht
  have e0 : sts[0]! = sts[0] := getElem!_pos sts 0 h0
  rw [e0] at hidx hso hop hsp hnp hrows
  obtain ⟨p', -, -, hx', htyp, hpi, -, -, h0p, -⟩ := stagedTables hS
  rw [hxp] at hx'
  cases hx'
  have hcc := partCircuit_staged hS hxp htyp hpi hr hpc h0p
  obtain ⟨inn, cr, dr, hinn, hor, htie⟩ := dirsTie_staged hT hnc hS hck hr hss hsv hpc hxp ht hsu h0 hidx hown
    hso hop hsp hf' hset hdin (hin r ss lv n hr hss hsv) hnp
  obtain ⟨hdr, hpos⟩ := hrows inn cr dr hinn hor
  show OpenLower.DirsTie (OpenLower.copyR (Claims.Recursion.xI (DR b elem IV) x) w p ht q) LANES HH
    (posAt (Claims.Recursion.xI (DR b elem IV) x)
      ((List.range (Bd.DE.V.k (Claims.Recursion.xI (DR b elem IV) x))).map cs) q)
  rw [hpos]
  have hcp : OpenLower.copyR (Claims.Recursion.xI (DR b elem IV) x) w p ht q =
      Flock.Readable.copyOf p 0 ht w (instR (Claims.Recursion.xI (DR b elem IV) x) q / cc.g)
        (instR (Claims.Recursion.xI (DR b elem IV) x) q % cc.g) 0 := by
    simp only [OpenLower.copyR, OpenLower.copyAtR, instR, getElem!_pos p.sts 0 ht, hcc]
  rw [hcp]
  exact htie hdr

/-- **`CopyDirsR` at the deployed layout `layVR`**, from `DirsSideR` (`copyDirsR_of`: `HH ≤ 32` at a staged part). -/
theorem copyDirsR_layVR {b : Brings} {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {IV : Claims.Verifier Coin} {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}
    {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hside : DirsSideR b elem IV shape untr cmt Bd pieces posAt) :
    OpenLower.CopyDirsR FlockDeploy.sha512 (DR b elem IV) shape untr b cmt (layVR elem) Bd pieces posAt :=
  OpenLower.copyDirsR_of FlockDeploy.sha512 (DR b elem IV) shape untr b cmt (layVR elem) Bd pieces posAt
    (copyDirsTieR_layVR pieces posAt hside)

end

end RecDeploy.OpenDirs
