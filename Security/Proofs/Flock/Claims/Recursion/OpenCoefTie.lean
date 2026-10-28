import Proofs.Flock.Claims.Recursion.OpenDirsDeployed
import Proofs.Flock.Claims.Recursion.InnerExtras
import Proofs.Flock.Claims.Recursion.VerdictRowECoin
import Proofs.Flock.Claims.Recursion.VerdictParts
import Proofs.Flock.Claims.Recursion.CapLowerPin
import Proofs.Flock.Claims.Recursion.VStarPartsSparse
import Proofs.Flock.C3.RegLevels.VRows
import Proofs.Flock.C3.Opening.SessionAcc
import Proofs.Flock.VBridge.AccUnits
import Proofs.Flock.VBridge.Family.Opened
import Proofs.Flock.VBridge.Family.Structure
import Proofs.Flock.Soundness.Refine.LigeritoRun
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.DecompBDeployed
import Proofs.Flock.Claims.Recursion.VerdictRowDCover

/-!
# Check 5's `coef` tie at the deployed run

A `rec-open` piece's copy reads its opening's coefficients at its `coef` columns (`openCoefs`). Check 5
(`RecCoins.checkOpen`) holds each instance's public commit string of port `coef` to `commitRow` of the verifier's own
row `openCoef` under `verifierSalt`: the words of `RecCoins.coefAt` at its rep's `κ` and `eq(α)` at the query. So, as
for `dirs` (`OpenDirs.dirsTie_staged`), `port_tie` gives the copy's `coef` bits as the verifier's row, and the coin side
reads that row off the coins as `Rows.coefRead` (`coefTieR_of`). The record facts are the deployed record's, as for
`dirs`: `DeployedR.layR` records the part's hm96 calls at `coef` (`openPorts`, `portCalls_sub_layR_open`), and check 5's
evaluations of the `coef` rows are `openSetupSt`'s, which the deployed setup lists (`hset`, `resSetup_deployedB`).

* `openCoefs_of_words`: a copy whose `coef` columns hold the bits of `RecCoins.words xs` reads `xs`.
* `coefCols_staged`, `slotCoef_staged`: port `coef` is row 3 of the unit, from column `coefCol LANES HH 0` of each copy,
  and its slot commits.
* `coefTie_staged`: the copy's `coef` columns are check 5's `coef` row, at each instance.
* `CoefTieR`, `coefTieR_of`: check 5's `coef` row at a piece's instance is the words of `coefRead` at the coins, from
  `InnerCoinsR`.
* `CopyCoefTieR`, `copyCoefTieR_of`, `copyCoefTieR_dense`: the tie at every piece a level part owns, at any accepted
  run (`AtPieces`); `CopyCoefTieRP`, `copyCoefTieRP_dense` at each part's own witness (`AtPiecesP`, `JointP`), and
  `copyCoefTieRP_deployed` at the deployed brings, the deployed bound and row D's pieces, with no premise.
-/

namespace RecDeploy.OpenCoef

open Flock
open Claims (Msg)
open FlockDeploy (Coin setupOf tabsOf pointsOf)
open Claims.Recursion.VStarB
open ResLower
open RecDeploy.DeployedR (rowForms portCalls)
open RecDeploy.OpenDirs (DR four_lt_nIn getD_of_lt)
open FlockSoundness.Discharge.Recursion (openRows openCoef openDirs innerLevels)
open FlockSoundness.Refine (shapeOf)

noncomputable section

/-! ## Reading a copy's `coef` columns -/

/-- Column `h` of a `rec-open` copy's `coef` row: `openCoefs`' base. -/
def coefCol (LANES HH h : ℕ) : ℕ := 128 * LANES + 1024 * max 1 ((HH + 1) / 2) + 1024 + h

theorem coefCol_add (L H h : ℕ) : coefCol L H h = coefCol L H 0 + h := by
  simp [coefCol]

theorem words_size (xs : Array F128) : (RecCoins.words xs).size = 8 * xs.size := by
  have h : ∀ l : List F128,
      (l.flatMap fun x => (FlockC3.RegLevels.u16s x.lo ++ FlockC3.RegLevels.u16s x.hi).toList).length =
        8 * l.length := by
    intro l
    induction l with
    | nil => simp
    | cons x l ih => simp [List.flatMap_cons, ih] <;> omega
  unfold RecCoins.words
  rw [← Array.length_toList, Array.toList_flatMap, h, Array.length_toList]

theorem coefAt_size (kap : Array (Array F128)) (L : ℕ) (e : F128) : (RecCoins.coefAt kap L e).size = 2 * L := by
  simp [RecCoins.coefAt] <;> omega

/-- **A copy whose `coef` columns hold the words of `xs` reads `xs`**: element `i` of `openCoefs` is its 128 bits at
`coefCol (128 i)`, which are bits `128 i …` of `RecCoins.words xs` (`words_bit`). -/
theorem openCoefs_of_words {v : ℕ → Bool} {LANES HH : ℕ} {xs : Array F128} (hxs : xs.size = 2 * LANES)
    (hv : ∀ h < 256 * LANES, v (coefCol LANES HH h) = ((RecCoins.words xs)[h / 16]!).testBit (h % 16)) :
    Flock.Readable.openCoefs v LANES HH = xs := by
  apply Array.ext
  · rw [Flock.Readable.openCoefs, FlockVBridge.Family.size_elems, hxs]
  · intro i h1 h2
    have hi : i < 2 * LANES := by rw [← hxs]; exact h2
    simp only [Flock.Readable.openCoefs, FlockVBridge.Family.elems, Array.getElem_ofFn]
    apply FlockVBridge.AccChain.eq_of_bitOf
    intro t ht
    rw [FlockVBridge.bitOf_ofBits _ ht]
    have hb : 128 * i + t < 256 * LANES := by omega
    have e := hv (128 * i + t) hb
    unfold coefCol at e
    simp only [Nat.add_assoc] at e ⊢
    rw [e, FlockC3.RegLevels.words_bit xs (by rw [hxs]; omega), show (128 * i + t) / 128 = i by omega,
      show (128 * i + t) % 128 = t by omega, getElem!_pos xs i h2]

/-! ## The `rec-open` unit's `coef` port -/

/-- **Port `coef` is the circuit's row 3**: the unit's first five rows are `row`, `aux`, `acc`, `coef`, `dirs`
(`rowsOk`). -/
theorem findIdx_coef {c : Circuit} {L H : ℕ} (h : RecOpen.rowsOk c L H = true) :
    c.ports.findIdx? (·.name == "coef") = some 3 := by
  have hnm : ∀ j < 5,
      ((Flock.RecOpen.rows L H)[j]!).1 = (#["row", "aux", "acc", "coef", "dirs"] : Array String)[j]! := by
    intro j hj
    interval_cases j <;> rfl
  obtain ⟨h3, hn3, -⟩ := FlockC3.RegLevels.rowsOk_get h (show 3 < 5 by decide)
  refine Array.findIdx?_eq_some_iff_getElem.2 ⟨h3, ?_, fun j hj => ?_⟩
  · exact (congrArg (· == "coef") (hn3.trans (hnm 3 (by decide)))).trans (by decide)
  · obtain ⟨hj', hnj, -⟩ := FlockC3.RegLevels.rowsOk_get h (show j < 5 by omega)
    intro hc
    have e := (congrArg (· == "coef") (hnj.trans (hnm j (by omega)))).symm.trans hc
    interval_cases j <;> exact absurd e (by decide)

theorem preBits_coef (L H : ℕ) :
    FlockC3.RegLevels.preBits L H 3 = (Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux +
      (Flock.RecOpen.ports L H).acc := by
  simp [FlockC3.RegLevels.preBits, Flock.RecOpen.Ports.bits] <;> omega

/-- **Port `coef` starts at `coefCol`**: the words of the three rows before it are its offset. -/
theorem words_pre3 {c : Circuit} {L H : ℕ} (h : RecOpen.rowsOk c L H = true) :
    16 * (c.ports.extract 0 3).foldl (· + ·.words) 0 = coefCol L H 0 := by
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
  have hst : (c.ports.extract 0 3).foldl (· + ·.words) 0 = ((c.ports.toList.take 3).map (·.words)).sum :=
    extract_words (·.words) c.ports 3
  rw [hst, hpre 3 (by omega), preBits_coef]
  simp [coefCol, Flock.RecOpen.ports]

/-- **Where a copy's `coef` columns sit**: at a staged `rec-open` part, port `coef` is row 3 (`findIdx_coef`) of
`16 LANES` words, at a `LANES` that is a multiple of 8 (`RecOpen.valid`), and instance `i`'s `coef` row
(`rowHidden_eq`) is copy `(i / g, i % g, 0)`'s columns from `coefCol LANES HH 0` (`words_pre3`, the unit's one input
group at column 0), below the unit's constant (`leaves_le_constPos`). -/
theorem coefCols_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size) (hdin : 3 < HmOut.nIn c) (i : ℕ) :
    c.ports.findIdx? (·.name == "coef") = some 3 ∧ ∃ pt : Port, c.ports[3]? = some pt ∧ pt.words = 16 * LANES ∧
      LANES % 8 = 0 ∧ ∀ h < 256 * LANES, ∃ hl : h < (rowForms c "coef" 3 i).length,
        Flock.Readable.copyOf p 0 ht w (i / c.g) (i % c.g) 0 (coefCol LANES HH h) =
          formBit w (rowForms c "coef" 3 i)[h] := by
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
  obtain ⟨hval, hlaid⟩ := Flock.RecOpen.check_ok P.recNet hpar
  simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hval
  obtain ⟨⟨hL8, hLm⟩, -⟩ := hval
  have hg := FlockC3.RegLevels.inGroups_of_laid hlaid
  subst hc0
  have hq3 := findIdx_coef hrk
  obtain ⟨hqs, -, hw3, hb3⟩ := FlockC3.RegLevels.rowsOk_get hrk (show 3 < 5 by decide)
  have hd : (Flock.RecOpen.ports LANES HH).bits[3]! = 256 * LANES := rfl
  have hbits : p.sts[0].c.ports[3].bits = 256 * LANES := by rw [hb3, hd]
  have hwords : p.sts[0].c.ports[3].words = 16 * LANES := by omega
  have hout : "coef" ∉ (VStar.Stmt.recOpen lv LANES HH n).sub.outputs := by
    show "coef" ∉ ["out", "acc_out"]
    decide
  have hN := in_block hrows hout hq3
  rw [getElem!_pos p.sts[0].c.ports 3 hqs] at hN
  have hrh := rowHidden_eq i hl hq3 hqs hN
  have hpre := words_pre3 hrk
  have hcol0 : (p.sts[0].c.unit.inGroups.getD 0 default).col = 0 := by rw [hg]; rfl
  refine ⟨hq3, p.sts[0].c.ports[3], Array.getElem?_eq_getElem hqs, hwords, hLm, fun h hh => ?_⟩
  have hcp := leaves_le_constPos P hl (by omega)
  rw [hcol0, Nat.zero_mul, Nat.zero_add] at hcp
  have hlen : (rowForms p.sts[0].c "coef" 3 i).length = 256 * LANES := by simp [rowForms, hdin, hrh, hbits]
  refine ⟨by omega, ?_⟩
  have hel : (rowForms p.sts[0].c "coef" 3 i)[h]'(by omega) = HmNets.Lin.var (i / p.sts[0].c.g * 2 ^ p.sts[0].c.kLog +
      (p.sts[0].c.unitRange.slot (i % p.sts[0].c.g * p.sts[0].c.upv) +
        ((p.sts[0].c.unit.inGroups.getD 0 default).col * Flock.WORD +
          16 * (p.sts[0].c.ports.extract 0 3).foldl (· + ·.words) 0 + h))) := by
    simp [rowForms, hdin, hrh]
  have hix : coefCol LANES HH h = 16 * (p.sts[0].c.ports.extract 0 3).foldl (· + ·.words) 0 + h := by
    rw [hpre, coefCol_add]
  have hj : coefCol LANES HH h ≤ p.sts[0].c.unit.constPos := by rw [hix]; omega
  rw [hel, formBit_var, hcol0, Nat.zero_mul, Nat.zero_add, ← hix, Claims.Recursion.VStar.copyOf_eq hj]
  simp [Claims.Recursion.VStar.copyIdx, hsz, hupv]

/-- **A level part's `coef` slot commits**: `coef` is an input of the `rec-open` unit (its outputs are `out` and
`acc_out`), so its row lies in the table's leaves (`in_block`); each instance `i < n` is one of the table's
(`staged_scope`, `oneFull_staged`), so in one of its blocks (`div_lt_blocks`), and `slotCommits_table` applies. -/
theorem slotCoef_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    {qc i : ℕ} (hqc : c.ports.findIdx? (·.name == "coef") = some qc) (hi : i < n) :
    SlotCommits c "coef" qc i w := by
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
  have hout : "coef" ∉ (VStar.Stmt.recOpen lv LANES HH n).sub.outputs := by
    show "coef" ∉ ["out", "acc_out"]
    decide
  exact slotCommits_table ht' hd (by omega) hv1 hl L regs hz1 hqc (in_block hrows hout hqc) hb

/-! ## Check 5's `coef` rows -/

/-- **Check 5's `rec-open` rows, opened**: where `openRows` gives rows, the inner session's algebra, the one level of
`LANES` lanes and `H` siblings, both reps, and the `coef` and `dirs` rows built of them. -/
theorem openRows_coef {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (h : openRows inn LANES H = some (cr, dr)) :
    ∃ sh extras l r0 r1, inn.algebra = .ok (sh, extras) ∧ innerLevels sh LANES H = [l] ∧
      inn.rep sh extras 0 = .ok r0 ∧ inn.rep sh extras 1 = .ok r1 ∧ cr = openCoef sh l LANES r0 r1 ∧
      dr = openDirs sh l H r0 r1 := by
  unfold openRows at h
  cases ha : inn.algebra with
  | error e => rw [ha] at h; cases h
  | ok p =>
    obtain ⟨sh, extras⟩ := p
    rw [ha] at h
    dsimp only at h
    cases hl : innerLevels sh LANES H with
    | nil => rw [hl] at h; simp at h
    | cons l t =>
      cases t with
      | cons _ _ => rw [hl] at h; simp at h
      | nil =>
        rw [hl] at h
        dsimp only at h
        cases h0 : inn.rep sh extras 0 <;> cases h1 : inn.rep sh extras 1 <;> rw [h0, h1] at h <;>
          simp only [reduceCtorEq] at h
        rename_i r0 r1
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        exact ⟨sh, extras, l, r0, r1, rfl, hl, h0, h1, h.1.symm, h.2.symm⟩

theorem cr_size_eq {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (h : openRows inn LANES H = some (cr, dr)) : cr.size = dr.size := by
  obtain ⟨sh, extras, l, r0, r1, -, -, -, -, rfl, rfl⟩ := openRows_coef h
  simp [openCoef, openDirs]

/-- **Each of check 5's `coef` rows is `16 LANES` words**: the words of `coefAt`'s `2 LANES` elements. -/
theorem crRow_size {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (h : openRows inn LANES H = some (cr, dr)) {i : ℕ} (hi : i < cr.size) : cr[i]!.size = 16 * LANES := by
  obtain ⟨sh, extras, l, r0, r1, -, -, -, -, rfl, -⟩ := openRows_coef h
  unfold openCoef at hi ⊢
  rw [Array.size_map, Array.size_range] at hi
  rw [FlockC3.RegLevels.getElem!_map_range _ hi]
  simp only [words_size, coefAt_size] <;> omega

/-- **`coef` row `r q + j` is rep `r`'s query `j`**: at level `l` of `q` queries, instance `r q + j` (`r < 2`,
`j < q`) holds the words of rep `r`'s coefficients at `eq(α_l)[j]`. -/
theorem openCoef_get (sh : Shape) (l LANES : ℕ) (r0 r1 : RecCoins.Rep) {r j : ℕ} (hr : r < 2)
    (hj : j < (sh.sched.levels[l]!).q) :
    (openCoef sh l LANES r0 r1)[r * (sh.sched.levels[l]!).q + j]! =
      RecCoins.words (RecCoins.coefAt ((if r = 0 then r0 else r1).kappa[l]!) LANES
        (eqTable ((if r = 0 then r0 else r1).info.lg.conss[l]!).alpha)[j]!) := by
  have hlt : r * (sh.sched.levels[l]!).q + j < (openCoef sh l LANES r0 r1).size := by
    unfold openCoef
    rw [Array.size_map, Array.size_range]
    have : r * (sh.sched.levels[l]!).q ≤ 1 * (sh.sched.levels[l]!).q := Nat.mul_le_mul_right _ (by omega)
    omega
  rw [getElem!_pos (openCoef sh l LANES r0 r1) _ hlt]
  unfold openCoef
  obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
  · simp [hj, Nat.mod_eq_of_lt hj]
  · simp only [Nat.one_mul]
    have h1 : ¬ (sh.sched.levels[l]!).q + j < (sh.sched.levels[l]!).q := by omega
    have h2 : ((sh.sched.levels[l]!).q + j) % (sh.sched.levels[l]!).q = j := by
      rw [Nat.add_mod_left, Nat.mod_eq_of_lt hj]
    simp [h1, h2]

/-! ## The tie at a copy -/

/-- **Check 5's `coef` tie at a level part's instance, at `layR`**: at a staged `rec-open` part whose calls at `layR`
are true and collision-free with the verifier's evaluations, and whose check 5 runs `checkOpen`, the inner session check 5 loads gives `coef` rows `cr`, and at each instance `i < n` of them
the copy `(i / g, i % g, 0)` holds `cr[i]`'s bits at its `coef` columns: `port_tie` at the `coef` slot
(`slotCoef_staged`, whose calls `layR` records, `portCalls_sub_layR_open`) and check 5's public row
(`check5_bytes_lt`), with the verifier's evaluations of it in `openSetupSt` (`hset`), and the slot's forms the copy's
`coef` columns (`coefCols_staged`). -/
theorem coefTie_staged {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV) xO)).isSome = false)
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
    (hset : ∀ u ∈ resSetup xO, u ∈ b.setup xO) {i : ℕ} (hdin : 3 < HmOut.nIn c) (hi : i < n) (hnp : i < sts[0].pub.n) :
    ∃ (inn : RecCoins.Inner) (cr dr : Array (Array ℕ)), f.load sts[0].c.unit.name = .ok inn ∧
      openRows inn LANES HH = some (cr, dr) ∧
      (i < cr.size → ∀ h < 256 * LANES, Flock.Readable.copyOf p 0 ht w (i / c.g) (i % c.g) 0 (coefCol LANES HH h) =
        ((cr[i]!)[h / 16]!).testBit (h % 16)) := by
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
  obtain ⟨cr, dr, hrows, ⟨what, hcv⟩, -⟩ := FlockSoundness.Discharge.Recursion.open_of_check hco
  refine ⟨inn, cr, dr, hinn, hrows, fun hnr => ?_⟩
  obtain ⟨hqc, ptc, hptc, hwords, hL8, hcols⟩ := coefCols_staged (w := w) hS hr hss hsv hpc hx ht hdin i
  obtain ⟨prog, hprog, hpv⟩ := OpenDirs.check5_bytes_lt hS hck hr hpc hsu h0 hidx hown hcv hqc hnN hnr hnp
  have hst : openSetupSt (RecCoins.Source.atC3 i0.src i0.inner i0.commits) sts[0] =
      commitIns cr (RecCoins.verifierSalt prog RecCoins.COEF_VALUE) ++
        commitIns dr (RecCoins.verifierSalt prog RecCoins.DIRS_VALUE) := by
    unfold openSetupSt
    simp only [hpar, hf, hinn, hprog, Except.toOption, hrows]
  have hmemS : ∀ u ∈ commitIns cr (RecCoins.verifierSalt prog RecCoins.COEF_VALUE), u ∈ b.setup xO :=
    fun u hu => hset u (by
      unfold resSetup; rw [hsu]
      exact List.mem_flatMap.2 ⟨sts[0], by simp, List.mem_append_right _ (by rw [hst]; exact List.mem_append_left _ hu)⟩)
  have hci : rowInput (RecCoins.padTo cr[i]! (Registered.rowWords cr[i]!.size 16)) ∈
        commitIns cr (RecCoins.verifierSalt prog RecCoins.COEF_VALUE) ∧
      (Hm96.Default512.saltPrefix ++ RecCoins.verifierSalt prog RecCoins.COEF_VALUE i).data.toList ∈
        commitIns cr (RecCoins.verifierSalt prog RecCoins.COEF_VALUE) := by
    unfold commitIns
    exact ⟨List.mem_flatMap.2 ⟨i, List.mem_range.2 hnr, List.mem_cons_self⟩,
      List.mem_flatMap.2 ⟨i, List.mem_range.2 hnr, List.mem_cons_of_mem _ List.mem_cons_self⟩⟩
  have hfd : rowForms c "coef" 3 i = ((rowHidden c "coef" i).getD []).map HmNets.Lin.var := by
    unfold rowForms
    rw [ite_eq_left hdin]
  have htie := port_tie rfl hT hnc
    (portCalls_sub_layR_open elem hr hss hsv hpc (by simp [DeployedR.openPorts]) hqc hi)
    (by rw [hfd]; exact slotCoef_staged hS hck hr hss hsv hpc hqc hi) hpv.symm
    (hmemS _ hci.1) (hmemS _ hci.2)
  have hsz := crRow_size hrows hnr
  have hwd : ptc.words = (RecCoins.padTo cr[i]! (Registered.rowWords cr[i]!.size 16)).size := by
    rw [hwords]
    simp [RecCoins.padTo, Registered.rowWords, hsz] <;> omega
  have hrb := readForms_of_rowInput hptc hwd htie
  intro h hh
  obtain ⟨hl, hcol⟩ := hcols h hh
  have hb16 : h < 16 * (RecCoins.padTo cr[i]! (Registered.rowWords cr[i]!.size 16)).toList.length := by
    rw [Array.length_toList, ← hwd, hwords]
    omega
  have hj : h / 16 < cr[i]!.size := by rw [hsz]; omega
  rw [hcol, ← bitL_readForms hl, hrb, wordBytes, bitL_wordBytes _ h hb16, List.getD_eq_getElem?_getD,
    Array.getElem?_toList, ← Array.getD_eq_getD_getElem?, getD_padTo, getD_of_lt hj]

/-! ## The coin side -/

/-- **`alphaOf` is the level's context's `α`**: where the contexts are the levels in order (`rows_conss`), the context
starting at level `l` is the `l`-th. -/
theorem alphaOf_eq {info : RecAlgebra.Info RecAlgebra.Lin} {n : ℕ} {qs : ℕ → ℕ}
    (E : info.lg.conss.toList.map (fun c => (c.start, c.pos.size)) = (List.range n).map fun k => (k, qs k))
    {l : ℕ} (hl : l < n) : Rows.alphaOf info l = (info.lg.conss[l]!).alpha := by
  have hsz : info.lg.conss.size = n := by
    have := congrArg List.length E
    simpa using this
  have hst : ∀ k (hk : k < info.lg.conss.size), info.lg.conss[k].start = k := by
    intro k hk
    have e := congrArg (fun L => L[k]?) E
    simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hk, Option.map_some,
      List.getElem?_range (hsz ▸ hk), Option.some.injEq, Prod.mk.injEq] at e
    exact e.1
  have hf : info.lg.conss.find? (·.start == l) = some info.lg.conss[l] := by
    rw [Array.find?_eq_some_iff_getElem]
    refine ⟨by simp [hst l (by omega)], l, by omega, rfl, fun j hj => ?_⟩
    simp [hst j (by omega)]
    omega
  unfold Rows.alphaOf
  rw [hf, getElem!_pos _ l (by omega)]

/-- **The `coef` tie** (`CoefTieR`, the coin side of `CopyCoefTieR`): at a V\* statement `xO` part `s` stages at `x`
and shows the coins `c` at, where check 5 loads the staged table's (table 0's) inner session from its source and
builds `rec-open`'s rows at the level part's `LANES`, `HH` (`openRows`), the `coef` row of a piece `ownQR` gives part
`s`, at instance `rep · q + query`, is the words of the `coef` `coefRead` reads off the coins. -/
def CoefTieR (untr : TrReader) (b : Brings) (pieces : Msg → List OpenPiece) (Bk : Msg → ℕ) (s : ℕ) (x : Msg)
    (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ (_ : 0 < sts.size) (f : RecCoins.Source) (inn : RecCoins.Inner),
      RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[0].c.unit.name = .ok inn →
      ∀ (r : XR) (ss : Array VStar.Stmt) (lv LANES HH n : ℕ), readX x = some r →
        RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
      ∀ (cr dr : Array (Array ℕ)), openRows inn LANES HH = some (cr, dr) →
      ∀ q ∈ pieces (xIOf x), OpenLower.ownQR x q = some s →
        cr[q.rep * OpenLower.qAt (xIOf x) q.level + q.query]! =
          RecCoins.words (Rows.coefRead (xIOf x) ((List.range (Bk (xIOf x))).map c) q)

/-- **`CoefTieR` from `InnerCoinsR`**, at a statement the reader reads at the dense form and pieces in range: as
`posTieR_of`, `openRows`' `coef` row `r q + j` is rep `r`'s query `j`'s coefficients (`openCoef_get`) at the piece's
level, and `verifierRows` gives the same `κ` and contexts at check 5's shape and coins as at the reader's
(`verifierRows_filter`, `verifierRows_comb`, `verifierRows_congr`); `coefRead`'s lanes are the level's (`levelMatch`)
and its `alphaOf` the level's context's (`alphaOf_eq`). -/
theorem coefTieR_of {untr : TrReader} {b : Brings} {pieces : Msg → List OpenPiece} {Bk : Msg → ℕ} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) (hR : OpenLower.PiecesInRangeR pieces)
    (hI : OpenLower.InnerCoinsR untr b Bk s x c) : CoefTieR untr b pieces Bk s x c := by
  intro xO hsh i spec sts hsu ht f inn hf hload r ss lv LANES HH n hr hss hsv cr dr hop q hq ho
  obtain ⟨v, hv, hm, hk, hex, hrd⟩ := hI xO hsh i spec sts hsu ht f inn hf hload
  obtain ⟨sh, extras, l, r0, r1, halg, hl, h0, h1, rfl, -⟩ := openRows_coef hop
  obtain ⟨hext, hsh'⟩ := OpenLower.algebra_ok halg
  obtain ⟨sch, hsch, hshm, hshk, hshc, hshs, hshh⟩ := OpenLower.shape?_ok hsh'
  -- the piece's level is the statement's
  obtain ⟨hq0, r', ss', hr', hss', L', H', n', hsv'⟩ := OpenLower.ownQR_some ho
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv'
  obtain ⟨hlvq, -, -, -⟩ := hsv'
  obtain ⟨hlvs, hLANES, hHH, -⟩ := OpenLower.levelMatch hst hd hr hss hsv
  have hS : OpenLower.schAt (xIOf x) 0 = sch := by
    unfold OpenLower.schAt
    rw [hv]
    dsimp only
    rw [← hm, hsch, Option.getD_some]
  obtain ⟨hrep, hquery⟩ := hR (xIOf x) q hq
  rw [hq0, hS] at hquery
  have hlev : q.level < sch.levels.size := by
    by_contra hge
    rw [getElem!_neg sch.levels q.level hge] at hquery
    exact Nat.not_lt_zero _ hquery
  rw [hS, ← hlvs, hlvq] at hLANES hHH
  have hql : q.level = l := OpenLower.innerLevels_unique hl (by rw [hshs]; exact hlev)
    (by rw [hshs]; exact hLANES.symm) (by rw [hshs]; exact hHH.symm)
  have hl' : l < sh.sched.levels.size := by rw [hshs, ← hql]; exact hlev
  -- check 5's row
  have hqa : OpenLower.qAt (xIOf x) q.level = (sh.sched.levels[l]!).q := by
    unfold OpenLower.qAt
    rw [hS, hshs, hql]
  rw [hqa, openCoef_get sh l LANES r0 r1 hrep (by rw [hshs, ← hql]; exact hquery)]
  refine congrArg RecCoins.words ?_
  -- rep `q.rep`'s rounds and rows
  have hrq : inn.rep sh extras q.rep = .ok (if q.rep = 0 then r0 else r1) := by
    obtain h | h : q.rep = 0 ∨ q.rep = 1 := by omega
    · rw [h]; simpa using h0
    · rw [h]; simpa using h1
  obtain ⟨hrounds, hvr⟩ := RecDeploy.rep_ok hrq
  -- its contexts are the levels in order
  have hseam := FlockSoundness.Refine.fast100_level _ _ hsch
  have h0s : 0 < sh.sched.levels.size := by rw [hshs]; exact (FlockVBridge.Family.fast100_q0 hsch).1
  have hqs : ∀ k < sh.sched.levels.size, ∀ cs,
      (Flock.positions (sh.sched.levels[k]!) cs).size = (sh.sched.levels[k]!).q := by
    rw [hshs]
    exact fun k hk cs => FlockSoundness.Refine.positions_size _
      (hseam _ (by rw [getElem!_pos _ k hk]; exact Array.getElem_mem hk)) cs
  have E := FlockVBridge.Family.rows_conss hvr h0s hqs
  rw [OpenLower.verifierRows_filter (hrd q.rep hrep _ hrounds (OpenLower.verifierRows_pos hvr)),
    Rows.verifierRows_comb] at hvr
  -- the reader's coefficients
  have hexs : ((tabsOf v.2.2)[0]!.extra (pointsOf v.2.1 (((List.range (Bk (xIOf x))).map c).headD default))) =
      .ok extras := by rw [← hex, hext]
  have hLL : FlockVBridge.AccChain.levelLanes
      (shapeOf (tabsOf v.2.2)[0]! ((fast100 (tabsOf v.2.2)[0]!.m).getD default) extras.size) l = LANES := by
    show 2 ^ ((((fast100 (tabsOf v.2.2)[0]!.m).getD default).levels[l]!).lanes) = LANES
    rw [← hm, hsch, Option.getD_some, ← hql]
    exact hLANES.symm
  unfold Rows.coefRead
  rw [hv]
  dsimp only
  rw [hq0, hexs]
  simp only [Except.toOption, Option.getD_some, Nat.mul_zero, Nat.zero_add]
  rw [OpenLower.verifierRows_congr
    (sh := shapeOf (tabsOf v.2.2)[0]! ((fast100 (tabsOf v.2.2)[0]!.m).getD default) extras.size)
    (sh' := sh) (by
      simp only [OpenLower.algFields, Prod.mk.injEq]
      refine ⟨by rw [hshm, hm]; rfl, by rw [hshk, hk]; rfl, by rw [hshc]; rfl, by rw [hshs, ← hm, hsch]; rfl,
        by rw [hshh]; rfl⟩)]
  cases h : RecAlgebra.verifierRows sh (Rows.canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast
      q.rep) extras #[] with
  | error e => rw [h] at hvr; cases hvr
  | ok w =>
    rw [h] at hvr
    obtain ⟨w₁, w₂, w₃⟩ := w
    simp only [Functor.map, Except.map, Except.ok.injEq, Prod.mk.injEq] at hvr
    obtain ⟨-, hi, hk'⟩ := hvr
    dsimp only
    rw [hi, hk', hql, FlockC3.Opening.coefAt_eq, hLL, alphaOf_eq E hl']

/-- **`CoefTieR` at a dense compiled statement**, from the verifier's checks and the pieces' ranges
(`innerCoinsR_staged`). -/
theorem coefTieR_dense {b : Brings} {pieces : Msg → List OpenPiece} {Bk : Msg → ℕ} {s : ℕ} {x : Msg}
    {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement} (hst : parse x = some st0)
    (hd : Claims.Recursion.Form.Dense st0.form) (hR : OpenLower.PiecesInRangeR pieces)
    (hB : RestRead.kI x ≤ Bk (xIOf x)) : CoefTieR recRead b pieces Bk s x c :=
  coefTieR_of hst hd hR (OpenLower.innerCoinsR_staged hst hd hB)

/-! ## The tie at every piece -/

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)

/-- **Check 5's `coef` tie at the copies**: at every piece a level part owns, at any accepted run (`AtPieces`), the
coefficients the piece's copy reads (`openCoefs`) are the ones `coefRead` reads off the coins. -/
def CopyCoefTieR (pieces : Msg → List OpenPiece) : Prop :=
  OpenLower.AtPieces H D shape untr b cmt L pieces fun x c _ LANES HH _ _ w p ht q =>
    Flock.Readable.openCoefs (OpenLower.copyR (Claims.Recursion.xI D x) w p ht q) LANES HH =
      Rows.coefRead (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q

/-- **`CopyCoefTieR` at the parts' own witnesses** (`AtPiecesP`: `JointP`, at `w = ws[s]`). -/
def CopyCoefTieRP (pieces : Msg → List OpenPiece) : Prop :=
  OpenLower.AtPiecesP H D shape untr b cmt L pieces fun x c _ LANES HH _ _ w p ht q =>
    Flock.Readable.openCoefs (OpenLower.copyR (Claims.Recursion.xI D x) w p ht q) LANES HH =
      Rows.coefRead (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q

end Deployed

/-- **`CopyCoefTieR` at the layout of record `layR`**, at the deployed reader's statements at the dense form and pieces
in range, from the deployed setup listing check 5's evaluations (`hset`) and the coin side (`InnerCoinsR`): the part's circuit is the staged table's, with `coef` among its
inputs (`four_lt_nIn`); check 5's site holds the registered reads, the inner session and its unit no other recursive unit
(`copySite_staged`); the piece's instance is below the statement's `n`, which is check 5's `coef` rows (`drSize_staged`,
`cr_size_eq`) and at most the table's public file's, so the labels are `0, …, n − 1` (`CopySite.idx_staged`); then
`coefTie_staged` at the piece's instance gives its `coef` columns as check 5's row there, which is the words of
`coefRead` (`coefTieR_of`), and `openCoefs_of_words` reads them. -/
theorem copyCoefTieR_of {b : Brings} {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {IV : Claims.Verifier Coin} {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}
    {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}
    (pieces : Msg → List OpenPiece)
    (hP : ∀ x, (DR b elem IV).parse x = parse x) (hSh : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hR : OpenLower.PiecesInRangeR pieces) (hset : ∀ xO, ∀ u ∈ resSetup xO, u ∈ b.setup xO)
    (hI : ∀ s x c st0, parse x = some st0 → Claims.Recursion.Form.Dense st0.form →
      OpenLower.InnerCoinsR untr b Bd.DE.V.k s x c) :
    CopyCoefTieR FlockDeploy.sha512 (DR b elem IV) shape untr b cmt (DeployedR.layR elem) Bd pieces := by
  intro x st hx hf cs ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht
  obtain ⟨hxp, -, -, -, hT, hnc, hshow, hck, hS⟩ := hst
  have hx' : parse x = some st := (hP x).symm.trans hx
  have hd := hSh _ hf
  have hxI : Claims.Recursion.xI (DR b elem IV) x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP x]
  have hS' := hS
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i0.base.read with sts } := Option.some.inj (hxp.symm.trans (parse_of_setupO hsu))
  have h0 : 0 < sts.size := by subst hp; exact ht
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  obtain ⟨p', r', sv, hx2, htyp, hpi, hr', ⟨ss', hss', hsv'⟩, h0p, htab⟩ := stagedTables hS
  rw [hxp] at hx2
  cases hx2
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  -- the part's circuit, with `coef` among its inputs
  obtain ⟨vs, hvs, -, hall⟩ := readX_partVStar hr
  rw [hss] at hvs
  cases hvs
  obtain ⟨hs, -⟩ := Array.getElem?_eq_some_iff.1 hsv
  obtain ⟨-, pt, cc, -, hpt, hcc, -, -, -⟩ := hall s hs
  have hpc : r.w.parts[s]?.bind partCircuit = some cc := by rw [hpt, Option.bind_some]; exact hcc
  have hc0 := partCircuit_staged hS hxp htyp hpi hr hpc h0p
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hxp htyp hpi h0p
  have hn : p.sts[0].c.unit.name = Flock.RecOpen.unitName LANES HH := (htab 0 h0p).1
  have hpar := FlockVBridge.params_of_check P.recNet hn
  have hrk := (FlockC3.RegLevels.check_rows P.recNet (.inl hpar)).1
  have hou := Flock.RecOpen.outsOk_of_check P.recNet hpar
  rw [hc0] at hrk hou
  have hdin : 3 < HmOut.nIn cc := by have := four_lt_nIn hrk hou; omega
  -- check 5's site and rows
  rw [e0] at hpar
  obtain ⟨own, f, hown, hf', hso, hop, hsp, -⟩ := CopySite.copySite_staged hsu h0 (.inl (by simp [hpar]))
  obtain ⟨hins, hset0⟩ := setupO_some hsu
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset0) 0 h0
  rw [hown, hf'] at hcoin
  obtain ⟨inn, hinn, hco⟩ := FlockSoundness.Discharge.Recursion.openCheck_of_check hcoin hso hop hsp hpar
  obtain ⟨cr, dr, hrows, ⟨what, hcv⟩, -⟩ := FlockSoundness.Discharge.Recursion.open_of_check hco
  -- the piece's level is the statement's
  obtain ⟨h0q, r2, ss2, hr2, hss2, L2, H2, n2, hsv2⟩ := OpenLower.ownQR_some ho
  rw [hr] at hr2
  cases hr2
  rw [hss] at hss2
  cases hss2
  rw [hsv] at hsv2
  simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv2
  obtain ⟨hlvq, -, -, -⟩ := hsv2
  obtain ⟨hls, -, -, hnq⟩ := OpenLower.levelMatch hx' hd hr hss hsv
  have hql : q.level = s := hlvq.symm.trans hls
  obtain ⟨hrep, hqry⟩ := hR _ q hq
  rw [h0q, hxI, hql] at hqry
  have hlev : s < (OpenLower.schAt (xIOf x) 0).levels.size := by
    by_contra hge
    rw [getElem!_neg _ s hge] at hqry
    exact Nat.not_lt_zero _ hqry
  have hinst : OpenDirs.instR (Claims.Recursion.xI (DR b elem IV) x) q < n := by
    unfold OpenDirs.instR OpenLower.qAt
    rw [hxI, hql, hnq]
    generalize ((OpenLower.schAt (xIOf x) 0).levels[s]!).q = Q at hqry ⊢
    have h1 : q.rep * Q ≤ 1 * Q := Nat.mul_le_mul_right _ (by omega)
    omega
  -- the statement's instances are the table's, at most its public file's
  obtain ⟨-, r3, ss3, sv3, hr3, hss3, hsv3, hpop⟩ := staged_scope hS hxp h0p
  rw [hr] at hr3
  cases hr3
  rw [hss] at hss3
  cases hss3
  rw [hsv] at hsv3
  cases hsv3
  have hpop' : Flock.RecVStar.population p.sts[0] = n := hpop
  have hle := (oneFull_staged hS hsu).2 0 h0
  obtain ⟨-, hts⟩ := tablesOf_of_parse hxp
  obtain ⟨_, hset1⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset1
  rw [hpi] at hset1
  obtain ⟨-, hall1⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset1
  obtain ⟨d, hd1, -⟩ := hall1 0 h0p
  have hnn := setupOne_n hd1
  rw [e0] at hpop' hnn
  have hnp : n ≤ sts[0].pub.n := by omega
  -- check 5's `coef` rows are the statement's instances
  have hIc := hI s x cs st hx' hd xO (And.intro hshow hS) i0 spec sts hsu h0 f inn hf' hinn
  obtain ⟨v, hv, hm, -⟩ := hIc
  have hdrs : dr.size = n := OpenDirs.drSize_staged hx' hd hr hss hsv hlev ⟨v, hv, hm⟩ hrows
  have hcrs : cr.size = n := (cr_size_eq hrows).trans hdrs
  have hidx := CopySite.idx_staged hsu h0 hown hcv (by omega) (CopySite.labels_staged hS hsu h0)
  -- the copy's `coef` columns are check 5's row
  obtain ⟨inn', cr', dr', hinn', hor', htie⟩ := coefTie_staged hT hnc hS hck hr hss hsv hpc hxp ht hsu h0 hidx hown
    hso hop hsp hf' (hset xO)
    hdin hinst (by omega)
  rw [hinn] at hinn'
  cases hinn'
  rw [hrows] at hor'
  cases hor'
  -- which is the words of `coefRead`
  have hct := coefTieR_of hx' hd hR (hI s x cs st hx' hd) xO (And.intro hshow hS) i0 spec sts hsu h0 f inn hf' hinn
    r ss lv LANES HH n hr hss hsv cr dr hrows q (by rw [← hxI]; exact hq) ho
  have hct' : cr[OpenDirs.instR (Claims.Recursion.xI (DR b elem IV) x) q]! =
      RecCoins.words (Rows.coefRead (Claims.Recursion.xI (DR b elem IV) x)
        ((List.range (Bd.DE.V.k (Claims.Recursion.xI (DR b elem IV) x))).map cs) q) := by
    rw [hxI]
    exact hct
  have hcr : OpenDirs.instR (Claims.Recursion.xI (DR b elem IV) x) q < cr.size := by rw [hcrs]; exact hinst
  have hw16 := crRow_size hrows hcr
  rw [hct', words_size] at hw16
  show Flock.Readable.openCoefs (OpenLower.copyR (Claims.Recursion.xI (DR b elem IV) x) w p ht q) LANES HH =
    Rows.coefRead (Claims.Recursion.xI (DR b elem IV) x)
      ((List.range (Bd.DE.V.k (Claims.Recursion.xI (DR b elem IV) x))).map cs) q
  have hcp : OpenLower.copyR (Claims.Recursion.xI (DR b elem IV) x) w p ht q =
      Flock.Readable.copyOf p 0 ht w (OpenDirs.instR (Claims.Recursion.xI (DR b elem IV) x) q / cc.g)
        (OpenDirs.instR (Claims.Recursion.xI (DR b elem IV) x) q % cc.g) 0 := by
    simp only [OpenLower.copyR, OpenLower.copyAtR, OpenDirs.instR, getElem!_pos p.sts 0 ht, hc0]
  rw [hcp]
  refine openCoefs_of_words (by omega) fun h hh => ?_
  rw [htie hcr h hh, hct']

/-- **`CopyCoefTieR` at the deployed reader**, at the dense form: the coin side from the verifier's checks
(`innerCoinsR_staged`). -/
theorem copyCoefTieR_dense {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {IV : Claims.Verifier Coin} {cmt : Msg → ℕ → Msg → Msg}
    {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}
    (pieces : Msg → List OpenPiece)
    (hP : ∀ x, (DR b elem IV).parse x = parse x) (hR : OpenLower.PiecesInRangeR pieces)
    (hset : ∀ xO, ∀ u ∈ resSetup xO, u ∈ b.setup xO) (hB : ∀ x, RestRead.kI x ≤ Bd.DE.V.k (xIOf x)) :
    CopyCoefTieR FlockDeploy.sha512 (DR b elem IV) Claims.Recursion.Form.Dense recRead b cmt (DeployedR.layR elem) Bd
      pieces :=
  copyCoefTieR_of pieces hP (fun _ h => h) hR hset
    fun _ x _ _ hst hd => OpenLower.innerCoinsR_staged hst hd (hB x)

/-- **`CopyCoefTieRP` at the deployed reader**, at the dense form: `copyCoefTieR_dense` at each part's own witness
(`atPiecesP_of`). -/
theorem copyCoefTieRP_dense {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {IV : Claims.Verifier Coin} {cmt : Msg → ℕ → Msg → Msg}
    {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}
    (pieces : Msg → List OpenPiece)
    (hP : ∀ x, (DR b elem IV).parse x = parse x) (hR : OpenLower.PiecesInRangeR pieces)
    (hset : ∀ xO, ∀ u ∈ resSetup xO, u ∈ b.setup xO) (hB : ∀ x, RestRead.kI x ≤ Bd.DE.V.k (xIOf x)) :
    CopyCoefTieRP FlockDeploy.sha512 (DR b elem IV) Claims.Recursion.Form.Dense recRead b cmt (DeployedR.layR elem) Bd
      pieces :=
  OpenLower.atPiecesP_of _ _ _ _ _ _ _ (copyCoefTieR_dense pieces hP hR hset hB)

/-- **`CopyCoefTieRP` at the deployed brings, with no premise**: `copyCoefTieRP_dense` at `bB (deployedB b) …`, whose
setup lists check 5's evaluations (`resSetup_deployedB`), the deployed bound `bdR` (whose `k` is `kI`) and row D's
pieces (`piecesInRangeR_rowDPieces`). Its record is `DR … (bdR …).IVStar`, which is `DRL`. -/
theorem copyCoefTieRP_deployed (rd : Msg → Option FlockDeploy.FinalP) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (cmt : Msg → ℕ → Msg → Msg)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) :
    CopyCoefTieRP FlockDeploy.sha512
      (DR (DecompBR.bB (deployedB b) aux nOf hd rest) elem (bdR rd wE final w).IVStar) Claims.Recursion.Form.Dense
      recRead (DecompBR.bB (deployedB b) aux nOf hd rest) cmt (DeployedR.layR elem) (bdR rd wE final w)
      (Rows.rowDPieces key) :=
  copyCoefTieRP_dense (Rows.rowDPieces key) (fun _ => rfl) (OpenLower.piecesInRangeR_rowDPieces key)
    (resSetup_deployedB b) (fun _ => le_rfl)

end

end RecDeploy.OpenCoef
