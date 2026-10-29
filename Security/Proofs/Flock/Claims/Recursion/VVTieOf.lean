import Proofs.Flock.Claims.Recursion.InnerExtras
import Proofs.Flock.Claims.Recursion.VerdictRowRDep
import Proofs.Flock.Claims.Recursion.ResLowerDeployed
import Proofs.Flock.Claims.Recursion.VerdictRowDTie
import Proofs.Flock.C3.RegLevels.VRows
import Proofs.Flock.VBridge.Family.Join

/-!
# `VVTie` at `Rows.resVRead`'s verifier values

`vvTie_of`: at a dense compiled statement, `ResLower.VVTie` at `vvAt := Rows.resVRead`, wherever the staged
`rec-residuals` statements are plain (`holoOf … = .ok false`). Check 5 (`checkResiduals`) loads the inner session and
registers rep `r`'s `v` row: part `jj`'s elements `vs` of the verifier rows at its coins, the comb last
(`residualRows`). `resVRead` at key `(jj, o, g, 0)` reads group `jj`'s `V` elements of the verifier rows at table 0's
shape, its extra claims at the first coin's link points, rep `o + g`'s `canonRounds` and table 0's fold at them. They
agree:

* the inner session's shape is the part's (check 5 refuses another, `residualsShape_of_check`), its `m` and `k_log`
  table 0's and its rounds `canonRounds` (`staged_innerAt`, `rep_canon`), its extra claims table 0's and its comb table
  0's fold (`staged_bridges`), so `resVRead`'s verifier rows are the rep's `V`, then the comb (`verifierRows_comb`);
* the row's bits are `vOf`'s at `vs` (`rows_vOf`), so its `nv = vs.size` elements (`part_nv`, at the plain shape) are
  those of `V ++ comb` at `vs`;
* key `(jj, o, g, 0)` is instance `o + g` (`ResLower.keyCopy`), so `resVRead`'s rep is check 5's row.

`holoOf` reads META's `holo`, which `RecVStar.checkCircuit` pins to V*'s file's (`checkHolo`): `false` at every
statement over the dense form's plain inner shape (`staged_plain`, by `checkCircuit_plain`). `vvTie_dense` is
`vvTie_of` with `hplain` so discharged.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin setupOf tabsOf pointsOf)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Recursion (residualRows residualRowsOf residualGroups)
open RecDeploy.OpenLower (algFields)

/-! ## Check 5's rows -/

/-- **Check 5's shape is the part's**: an accepted `rec-residuals/m<m>-k<k>-c<cl>-p<j>` unit's inner session has an
algebra whose `m`, `k_log` and claims are the name's. -/
theorem residualsShape_of_check {inn : RecCoins.Inner} {c : Circuit} {pub : Public} {own : Array Registered.Port}
    {m k cl j : ℕ} (h : RecCoins.checkResiduals inn c pub own m k cl j = .ok ()) :
    ∃ sh extras, inn.algebra = .ok (sh, extras) ∧ m = sh.m ∧ k = sh.kLog ∧ cl = sh.claims := by
  unfold RecCoins.checkResiduals at h
  dsimp only at h
  obtain ⟨⟨sh, extras⟩, halg, h⟩ := FlockVBridge.PartS.bind_ok h
  try dsimp only at h
  split at h
  · cases h
  · rename_i hne
    have he : (m, k, cl) = (sh.m, sh.kLog, sh.claims) := by
      simpa only [bne_iff_ne, ne_eq, Decidable.not_not] using hne
    simp only [Prod.mk.injEq] at he
    exact ⟨sh, extras, halg, he.1, he.2.1, he.2.2⟩

theorem residualRows_some {inn : RecCoins.Inner} {sh : _root_.Flock.Shape} {extras : Array _root_.Flock.Claim}
    {j : ℕ} {rows : Array (Array ℕ)} (halg : inn.algebra = .ok (sh, extras)) (h : residualRows inn j = some rows) :
    ∃ gs R P vs, residualGroups sh = .ok gs ∧ gs[j]? = some (R, P, vs) ∧
      residualRowsOf inn sh extras vs = .ok rows := by
  unfold residualRows at h
  rw [halg] at h
  dsimp only at h
  cases hg : residualGroups sh with
  | error e => rw [hg] at h; cases h
  | ok gs =>
    rw [hg] at h
    dsimp only at h
    cases hj : gs[j]? with
    | none => rw [hj] at h; cases h
    | some t =>
      obtain ⟨R, P, vs⟩ := t
      rw [hj] at h
      dsimp only at h
      cases hr : residualRowsOf inn sh extras vs with
      | error e => rw [hr] at h; cases h
      | ok rows' =>
        rw [hr] at h
        cases h
        exact ⟨gs, R, P, vs, rfl, hj, hr⟩

/-- **The part's `v` elements are its group's**: `part`'s `nv` is the number of group `j`'s `V` elements. -/
theorem part_nv {sh : _root_.Flock.Shape} {j : ℕ} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {R P vs : Array ℕ} (hp : RecResiduals.part sh j = .ok (S, ns, nv))
    (hg : residualGroups sh = .ok gs) (hj : gs[j]? = some (R, P, vs)) : nv = vs.size := by
  unfold residualGroups at hg
  obtain ⟨S0, hS0, hg⟩ := FlockVBridge.PartS.bind_ok hg
  unfold RecResiduals.part at hp
  obtain ⟨S1, hS1, hp⟩ := FlockVBridge.PartS.bind_ok hp
  rw [hS0] at hS1
  cases hS1
  dsimp only at hp
  split at hp
  · cases hp
  · obtain ⟨gs', hgs', hp⟩ := FlockVBridge.PartS.bind_ok hp
    rw [hg] at hgs'
    cases hgs'
    try dsimp only at hp
    rw [hj] at hp
    try dsimp only at hp
    obtain ⟨S2, -, hp⟩ := FlockVBridge.PartS.bind_ok hp
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hp
    exact hp.2.2.symm

/-- A row check 5 computes, where its part reaches the comb, computed the comb. -/
theorem comb_of_row {inn : RecCoins.Inner} {sh : _root_.Flock.Shape} {extras : Array _root_.Flock.Claim}
    {vs : Array ℕ} {r : ℕ} {row : Array ℕ}
    (h : (do
      let rep ← inn.rep sh extras r
      let base := rep.v.extract 0 (rep.v.size - 64)
      let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure RecCoins.ZERO_COMB
      let full := base ++ comb
      pure (RecCoins.padTo (RecCoins.words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size)) :
        _root_.Flock.V (Array ℕ)) = .ok row)
    {rep : RecCoins.Rep} (hrep : inn.rep sh extras r = .ok rep) (ha : vs.any (· ≥ rep.v.size - 64) = true) :
    ∃ cb, inn.comb r rep = .ok cb := by
  obtain ⟨rep', hrep', h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hrep] at hrep'
  cases hrep'
  have hsz : (rep.v.extract 0 (rep.v.size - 64)).size = rep.v.size - 64 := by
    simp only [Array.size_extract, Nat.sub_le, Nat.min_eq_left, Nat.sub_zero]
  dsimp only at h
  split at h
  · obtain ⟨cb, hcb, -⟩ := FlockVBridge.PartS.bind_ok h
    exact ⟨cb, hcb⟩
  · rename_i hn
    exact absurd (by rw [hsz]; exact ha) hn

theorem comb_ok {inn : RecCoins.Inner} {r : ℕ} {rep : RecCoins.Rep} {cb : Array _root_.Flock.F128}
    (h : inn.comb r rep = .ok cb) : inn.combOf r rep.info = .ok cb ∧ cb.size = 64 := by
  unfold RecCoins.Inner.comb at h
  cases hc : inn.combOf r rep.info with
  | error e => simp [hc, bind, Except.bind] at h
  | ok c0 =>
    by_cases h64 : c0.size = 64
    · simp [hc, h64, bind, Except.bind, pure, Except.pure] at h
      subst h
      exact ⟨rfl, h64⟩
    · simp [hc, h64, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h

theorem comb_of_rows {inn : RecCoins.Inner} {sh : _root_.Flock.Shape} {extras : Array _root_.Flock.Claim}
    {vs : Array ℕ} {rows : Array (Array ℕ)} (h : residualRowsOf inn sh extras vs = .ok rows) {r : ℕ} (hr : r < 2)
    {rep : RecCoins.Rep} (hrep : inn.rep sh extras r = .ok rep) (ha : vs.any (· ≥ rep.v.size - 64) = true) :
    ∃ cb, inn.comb r rep = .ok cb := by
  unfold residualRowsOf at h
  obtain ⟨a, b, ha', hb', -⟩ := FlockC3.Opening.mapM_two h
  obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
  · exact comb_of_row ha' hrep ha
  · exact comb_of_row hb' hrep ha

/-! ## The copies and the reader -/

theorem portSizes_congr {sh sh' : _root_.Flock.Shape} (h : algFields sh = algFields sh') :
    RecResiduals.portSizes sh = RecResiduals.portSizes sh' := by
  unfold RecResiduals.portSizes
  rw [OpenLower.rounds_congr h]

theorem resVOf_congr {sh sh' : _root_.Flock.Shape} (h : algFields sh = algFields sh') :
    Rows.resVOf sh = Rows.resVOf sh' := by
  funext rounds extras comb p k
  unfold Rows.resVOf
  rw [OpenLower.verifierRows_congr h, OpenLower.structureOf_congr h, portSizes_congr h]

section resVOf

variable {sh : _root_.Flock.Shape} {rounds : Array (Array _root_.Flock.F128)} {extras : Array _root_.Flock.Claim}
  {comb v : Array _root_.Flock.F128} {info : RecAlgebra.Info RecAlgebra.Lin}
  {kappa : Array (Array (Array _root_.Flock.F128))} {S : RecAlgebra.GfS}
  {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {p : ℕ} {R P vs : Array ℕ}

theorem resVOf_some (hv : RecAlgebra.verifierRows sh rounds extras comb = .ok (v, info, kappa))
    (hS : RecAlgebra.structureOf sh = .ok S) (hg : RecResiduals.groups S (RecResiduals.portSizes sh) = .ok gs)
    (hj : gs[p]? = some (R, P, vs)) {k : ℕ} (hk : k < vs.size) :
    Rows.resVOf sh rounds extras comb p k = RecAlgebra.el v vs[k] := by
  unfold Rows.resVOf
  simp [hv, hS, hg, hj, hk]

theorem resVOf_none (hv : RecAlgebra.verifierRows sh rounds extras comb = .ok (v, info, kappa))
    (hS : RecAlgebra.structureOf sh = .ok S) (hg : RecResiduals.groups S (RecResiduals.portSizes sh) = .ok gs)
    (hj : gs[p]? = some (R, P, vs)) {k : ℕ} (hk : vs.size ≤ k) :
    Rows.resVOf sh rounds extras comb p k = 0 := by
  have hk' : vs[k]? = none := Array.getElem?_eq_none hk
  unfold Rows.resVOf
  simp [hv, hS, hg, hj, hk']

end resVOf

theorem getD_nat (a : Array ℕ) (i : ℕ) : a.getD i 0 = a[i]! := by
  rw [Array.getElem!_eq_getD]
  rfl

theorem append_cancel_right' {a a' z : Array _root_.Flock.F128} (h : a ++ z = a' ++ z) : a = a' := by
  have e := congrArg Array.toList h
  simp only [Array.toList_append] at e
  exact Array.toList_inj.1 (List.append_cancel_right e)

/-! ## The tie -/

/-- **`VVTie` at the read verifier values** (`Rows.resVRead`), at a dense compiled statement whose coins run to at least
the deployed plain verifier's inner rounds, wherever its staged `rec-residuals` statements are plain. -/
theorem vvTie_of {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement}
    (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form) (hB : RestRead.kI x ≤ Bk (xIOf x))
    (hplain : ∀ xO, showsPart recRead b s x xO c → ∀ i spec (sts : Array Stmt), setupO xO = some (i, spec, sts) →
      ∀ t (ht : t < sts.size), (RecResiduals.params sts[t].c.unit.name).isSome →
        RecResiduals.holoOf sts[t].c = .ok false) :
    VVTie recRead b Rows.resVRead Bk s x c := by
  intro xO hsh i spec sts hsu t ht m k cl jj hpar f inn rows hf hload hrows sh holo S ns nv hshp hholo hpart o g hr2
  obtain ⟨h1, -⟩ := oneFull_staged hsh.2 hsu
  obtain rfl : t = 0 := by omega
  have h0 : 0 < sts.size := ht
  obtain ⟨hins, hset0⟩ := setupO_some hsu
  -- check 5 at table 0: the inner session's shape is the part's
  have hrs : (RecResiduals.params sts[0].c.unit.name).isSome := by simp [hpar]
  obtain ⟨own, f', hown, hf', hso, hop, hsp, hro⟩ := CopySite.copySite_staged hsu h0 (.inr hrs)
  have hff : f' = f := Option.some.inj (hf'.symm.trans hf)
  subst f'
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset0) 0 h0
  rw [hown, hf] at hcoin
  obtain ⟨inn', hinn', hcr⟩ :=
    FlockSoundness.Discharge.Recursion.residualsCheck_of_check hcoin hso hop hsp (hro hrs) hpar
  have hii : inn' = inn := Except.ok.inj (hinn'.symm.trans hload)
  subst inn'
  obtain ⟨sh', extras, halg, hm', hk', hcl'⟩ := residualsShape_of_check hcr
  obtain ⟨gs, R, P, vs, hgs, hj, hrr⟩ := residualRows_some halg hrows
  obtain ⟨hex, hshI⟩ := OpenLower.algebra_ok halg
  obtain ⟨sch, hsch, hsm, hsk, hscl, hss, hsh0⟩ := OpenLower.shape?_ok hshI
  have hshe : sh = sh' := by
    rw [hm', hk', hcl', hsm, hsk, hscl] at hshp
    exact Option.some.inj (hshp.symm.trans hshI)
  subst sh'
  -- the plain part's `nv` is the group's
  have hh := hholo.symm.trans (hplain xO hsh i spec sts hsu 0 h0 hrs)
  cases hh
  have hwith : ({ sh with holo := false } : _root_.Flock.Shape) = sh := by
    cases sh
    dsimp only at hsh0
    subst hsh0
    rfl
  rw [hwith] at hpart
  have hnv : nv = vs.size := part_nv hpart hgs hj
  -- the row's bits, at the rep's coins
  obtain ⟨-, hbits⟩ := FlockC3.RegLevels.rows_vOf (inn := inn) (sh := sh) (extras := extras) (vs := vs) hrr
  obtain ⟨rep, hrep, hbit⟩ := hbits (o + g) hr2
  obtain ⟨-, V, hV, hvOf⟩ := FlockC3.RegLevels.rep_ok hrep
  have hI := OpenLower.staged_innerAt (b := b) (s := s) (c := c) hst hd hB
  obtain ⟨v, hv, hm, hk, -⟩ := hI xO hsh i hins f _ inn hf hload
  obtain ⟨v1, hv1, hrow⟩ := OpenLower.rep_canon hI hsh hins hf hload hr2 hrep
  have hvv1 : v1 = v := Option.some.inj (hv1.symm.trans hv)
  subst v1
  obtain ⟨hE, hC⟩ := OpenLower.staged_bridges (Bk := Bk) hst hd hB
  have hext := hE xO hsh i spec sts hsu h0 f inn hf hload v hv
  obtain ⟨v2, hv2, hcomb⟩ := hC xO hsh i hins f _ inn hf hload
  have hvv2 : v2 = v := Option.some.inj (hv2.symm.trans hv)
  subst v2
  obtain ⟨V', hV', hVall⟩ := Flock.RecAlgebra.verifierRows_comb hrow
  have hVV : V = V' := append_cancel_right' (hV.symm.trans hV')
  subst V'
  -- the reader's extra claims and shape are the session's
  have hexR : Rows.extrasRead v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c) = extras := by
    unfold Rows.extrasRead
    rw [← hext, hex]
    all_goals rfl
  have hsch' : fast100 (tabsOf v.2.2)[0]!.m = some sch := by rw [← hm]; exact hsch
  have hsR : algFields (Rows.shapeRead v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c)) = algFields sh := by
    unfold Rows.shapeRead
    rw [hexR]
    show ((tabsOf v.2.2)[0]!.m, (tabsOf v.2.2)[0]!.kLog, 2 + extras.size, (fast100 (tabsOf v.2.2)[0]!.m).getD default,
      false) = (sh.m, sh.kLog, sh.claims, sh.sched, sh.holo)
    simp only [hsch', hsm, hsk, hscl, hss, hsh0, hm, hk, Option.getD_some]
  obtain ⟨combR, hcR⟩ : ∃ C, Rows.combRead (tabsOf v.2.2)[0]!
      (Rows.shapeRead v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c))
      (Rows.canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast (o + g)) extras = C :=
    ⟨_, rfl⟩
  have hL : ∀ e, Rows.resVRead (xIOf x) ((List.range (Bk (xIOf x))).map c) (jj, o, g, 0) e =
      Rows.resVOf sh (Rows.canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast (o + g)) extras
        combR jj e := by
    intro e
    unfold Rows.resVRead
    rw [hv]
    dsimp only
    unfold Rows.resVAt
    rw [hexR, hcR, resVOf_congr hsR]
  -- the comb check 5 computes is the reader's
  have hCk : vs.any (· ≥ rep.v.size - 64) = true → inn.comb (o + g) rep = .ok combR := by
    intro ha
    obtain ⟨cb, hcb⟩ := comb_of_rows hrr hr2 hrep ha
    obtain ⟨hc0, h64⟩ := comb_ok hcb
    rw [hcb]
    congr 1
    rw [← hcR]
    simp only [hcomb] at hc0
    obtain ⟨lins, hlins⟩ := FlockVBridge.Family.residualLins_of_verifierRows hrow
    unfold Rows.combRead
    rw [OpenLower.residualLins_congr hsR, hlins]
    dsimp only
    rw [hc0]
    dsimp only
    rw [if_pos h64]
  have hgs' := hgs
  unfold residualGroups at hgs'
  obtain ⟨S0, hS0, hg0⟩ := FlockVBridge.PartS.bind_ok hgs'
  funext e
  rw [hL e]
  unfold wordsElems
  rw [hnv]
  by_cases he : e < vs.size
  · rw [if_pos he, resVOf_some (hVall combR) hS0 hg0 hj he]
    symm
    apply FlockVBridge.AccChain.eq_of_bitOf
    intro t ht
    rw [FlockVBridge.bitOf_ofBits _ ht]
    beta_reduce
    rw [getD_nat]
    have hb := hbit combR hCk (128 * e + t) (by omega)
    rw [show (128 * e + t) / 16 = 8 * e + t / 16 by omega, show (128 * e + t) % 16 = t % 16 by omega] at hb
    rw [hb, FlockVBridge.wbit, show (128 * e + t) / 128 = e by omega, show (128 * e + t) % 128 = t by omega,
      FlockVBridge.Decode.sub_lt _ _ e he, hvOf combR]
    all_goals rfl
  · rw [if_neg he, resVOf_none (hVall combR) hS0 hg0 hj (Nat.le_of_not_lt he)]

/-- **A dense statement's staged tables are plain**: each passed `checkCircuit` at the reader's inner shape
(`checks_stmt`, `checkStmt_circuit`), whose holo flag a dense form leaves off, so its META says no `holo`
(`checkCircuit_plain`). -/
theorem staged_plain {untr : TrReader} {b : Brings} {s : ℕ} {x : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement}
    (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form) {xO : Msg}
    (hsh : showsPart untr b s x xO c) {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) (t : ℕ) (ht : t < sts.size) :
    RecResiduals.holoOf sts[t].c = .ok false := by
  obtain ⟨ms, -, -, -, rfl, -, -⟩ := hsh.2
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  obtain ⟨_, _, _, -, -, hc⟩ := checkStmt_circuit (checks_stmt hC t ht)
  obtain ⟨r', hr', -, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  exact FlockSoundness.Discharge.Recursion.checkCircuit_plain hc hd.1

/-- **`VVTie` at `Rows.resVRead`'s verifier values, from the verifier's checks**: `vvTie_of`, its plain tables by
`staged_plain`. -/
theorem vvTie_dense {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.CompiledStatement}
    (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form) (hB : RestRead.kI x ≤ Bk (xIOf x)) :
    VVTie recRead b Rows.resVRead Bk s x c :=
  vvTie_of hst hd hB fun _ hsh _ _ _ hsu t ht _ => staged_plain hst hd hsh hsu t ht

end RecDeploy.ResLower
