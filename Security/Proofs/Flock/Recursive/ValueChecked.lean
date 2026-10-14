import Proofs.Flock.Recursive.ValueInner
import Proofs.Flock.C3.PubRows.Shape

/-!
# The value tie at a statement check 5 accepted

`value_tie_coll` at the verifier's rows (`valueSrcAll`, at its inner session `src t` and candidate-3 inputs `c3 t`),
from `RecCoins.check` at a statement of session `j`'s circuit, at each unit where `valueRowsAll` has rows:

* `value_tie_stacked`: a `rec-open-stacked` unit's `coef` and `dirs` (`stackedLevel_of_check`, `stacked_of_check`,
  `valueRows_coef`, `valueRows_dirs`);
* `value_tie_opening`: a `rec-opening` unit's `v` (`openingCheck_of_check`, `opening_of_check`, `valueRows_v`);
* `value_tie_open`: a `rec-open` unit's `coef` and `dirs` (`openCheck_of_check`, `open_of_check`, `innerRows_coef`,
  `innerRows_dirs`);
* `value_tie_residuals`: a `rec-residuals` unit's `v` (`residualsCheck_of_check`, `residuals_of_check`, `innerRows_v`);
* `value_tie_sparse`: a `rec-sparse` unit's `v` (`sparseCheck_of_check`, `sparse_of_check`, `innerRows_sparse`);
* `coins_of_session`: where a session accepts and its record draws, the statement setting up a drawn unit's part has
  the session's circuit and passes check 5 (`FlockC3.PubRows.stmt_of_session`'s argument, at `RecCoins.check`).
-/

namespace FlockSoundness.Discharge.Recursion

section Tie

open Game Audit

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ Reg' : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {src : List (Cm × I.Coin) → Option Flock.RecCoins.Source} {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3}
  {f : Flock.RecCoins.Source} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
  {j : Fin S} {sj : Flock.Stmt} {files : Option Flock.RecCoins.Source} {c3' : Flock.RecCoins.C3} {prog : String}
  {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size}

/-- **The value tie at a `rec-open-stacked` unit's `coef` or `dirs`**: where check 5 accepted a statement of session
`j`'s circuit at the verifier's candidate-3 inputs `c3 t`, a registered read `(u, q)` of its `coef` or `dirs` port
(without segments) is of row `u` of `valueRows`' rows there, and the registrant's bit `b` is bit `b % 16` of the row's
word `b / 16`, or `ValueColl` at `valueSrcAll src c3`. -/
theorem value_tie_stacked (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) files (some c3') = .ok ()) (hc3 : c3 t = some c3')
    {LANES H : ℕ} (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = some (LANES, H))
    (hname : ((Vs j).c.ports[q.val]!).name = "coef" ∨ ((Vs j).c.ports[q.val]!).name = "dirs")
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = ((Vs j).c.ports[q.val]!).name →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none) :
    ∃ rows value, valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value) ∧
      u.val < rows.size ∧
      ((∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q),
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          (rows[u.val]!)[b / 16]!.testBit (b % 16)) ∨
      ValueColl zs (valueSrcAll src c3) R ω t R₂) := by
  obtain ⟨o, l, hop, hls, ⟨what, hcv⟩, what', hdv⟩ :=
    stacked_of_check (stackedLevel_of_check hck (by rw [hc]; exact hso))
  rw [hc] at hop hcv hdv
  rcases hname with hn | hn
  · have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
        some (stackedCoefRows o l LANES, Flock.RecCoins.COEF_VALUE) := by
      rw [valueRowsAll_stacked hso, hc3, hn]
      exact valueRows_coef hso hop hls
    exact ⟨_, _, hv, value_tie_coll (zs := zs) (vsrc := valueSrcAll src c3) hcv hprog (valueSrcAll_eq hv hprog) hp
      (fun reg hr hr' => hone reg hr (hr'.trans hn.symm)) hpos hseg⟩
  · have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
        some (stackedDirsRows o l H, Flock.RecCoins.DIRS_VALUE) := by
      rw [valueRowsAll_stacked hso, hc3, hn]
      exact valueRows_dirs hso hop hls
    exact ⟨_, _, hv, value_tie_coll (zs := zs) (vsrc := valueSrcAll src c3) hdv hprog (valueSrcAll_eq hv hprog) hp
      (fun reg hr hr' => hone reg hr (hr'.trans hn.symm)) hpos hseg⟩

/-- **The value tie at a `rec-opening` unit's `v`**: `value_tie_stacked` at part `i`'s `v` row. -/
theorem value_tie_opening (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) files (some c3') = .ok ()) (hc3 : c3 t = some c3')
    (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none) {m cl i : ℕ}
    (hpar : Flock.RecOpening.params (Vs j).c.unit.name = some (m, cl, i))
    (hname : ((Vs j).c.ports[q.val]!).name = "v")
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = ((Vs j).c.ports[q.val]!).name →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none) :
    ∃ rows value, valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value) ∧
      u.val < rows.size ∧
      ((∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q),
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          (rows[u.val]!)[b / 16]!.testBit (b % 16)) ∨
      ValueColl zs (valueSrcAll src c3) R ω t R₂) := by
  obtain ⟨o, rows, what, hop, hrows, hcv⟩ :=
    opening_of_check (openingCheck_of_check hck (by rw [hc]; exact hso) (by rw [hc]; exact hpar))
  rw [hc] at hop hcv
  have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
      some (rows, Flock.RecCoins.V_VALUE) := by
    rw [valueRowsAll_opening hso hpar, hc3, hname]
    exact valueRows_v hso hpar hop hrows
  exact ⟨_, _, hv, value_tie_coll (zs := zs) (vsrc := valueSrcAll src c3) hcv hprog (valueSrcAll_eq hv hprog) hp
    (fun reg hr hr' => hone reg hr (hr'.trans hname.symm)) hpos hseg⟩

/-- **The value tie at a `rec-open` unit's `coef` or `dirs`**: `value_tie_stacked` at `openRows`' rows at the inner
session `src t` check 5 loaded. -/
theorem value_tie_open (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) (some f) (c3 t) = .ok ()) (hf : src t = some f)
    (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none)
    (hop : Flock.RecOpening.params (Vs j).c.unit.name = none) (hsp : Flock.RecSparse.params (Vs j).c.unit.name = none)
    {LANES H : ℕ} (ho : Flock.RecOpen.params (Vs j).c.unit.name = some (LANES, H))
    (hname : ((Vs j).c.ports[q.val]!).name = "coef" ∨ ((Vs j).c.ports[q.val]!).name = "dirs")
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = ((Vs j).c.ports[q.val]!).name →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none) :
    ∃ rows value, valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value) ∧
      u.val < rows.size ∧
      ((∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q),
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          (rows[u.val]!)[b / 16]!.testBit (b % 16)) ∨
      ValueColl zs (valueSrcAll src c3) R ω t R₂) := by
  obtain ⟨inn, hl, hck'⟩ := openCheck_of_check hck (by rw [hc]; exact hso) (by rw [hc]; exact hop)
    (by rw [hc]; exact hsp) (by rw [hc]; exact ho)
  obtain ⟨cr, dr, hrows, ⟨what, hcv⟩, what', hdv⟩ := open_of_check hck'
  rw [hc] at hl hcv hdv
  rcases hname with hn | hn
  · have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
        some (cr, Flock.RecCoins.COEF_VALUE) := by
      rw [hf, valueRowsAll_load hso hop hl, hn]
      exact innerRows_coef hsp ho hrows
    exact ⟨_, _, hv, value_tie_coll (zs := zs) (vsrc := valueSrcAll src c3) hcv hprog (valueSrcAll_eq hv hprog) hp
      (fun reg hr hr' => hone reg hr (hr'.trans hn.symm)) hpos hseg⟩
  · have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
        some (dr, Flock.RecCoins.DIRS_VALUE) := by
      rw [hf, valueRowsAll_load hso hop hl, hn]
      exact innerRows_dirs hsp ho hrows
    exact ⟨_, _, hv, value_tie_coll (zs := zs) (vsrc := valueSrcAll src c3) hdv hprog (valueSrcAll_eq hv hprog) hp
      (fun reg hr hr' => hone reg hr (hr'.trans hn.symm)) hpos hseg⟩

/-- **The value tie at a `rec-residuals` unit's `v`**: `value_tie_stacked` at part `i`'s `residualRows` at the inner
session `src t` check 5 loaded. -/
theorem value_tie_residuals (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) (some f) (c3 t) = .ok ()) (hf : src t = some f)
    (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none)
    (hop : Flock.RecOpening.params (Vs j).c.unit.name = none) (hsp : Flock.RecSparse.params (Vs j).c.unit.name = none)
    (ho : Flock.RecOpen.params (Vs j).c.unit.name = none) {m k cl i : ℕ}
    (hres : Flock.RecResiduals.params (Vs j).c.unit.name = some (m, k, cl, i))
    (hname : ((Vs j).c.ports[q.val]!).name = "v")
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = ((Vs j).c.ports[q.val]!).name →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none) :
    ∃ rows value, valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value) ∧
      u.val < rows.size ∧
      ((∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q),
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          (rows[u.val]!)[b / 16]!.testBit (b % 16)) ∨
      ValueColl zs (valueSrcAll src c3) R ω t R₂) := by
  obtain ⟨inn, hl, hck'⟩ := residualsCheck_of_check hck (by rw [hc]; exact hso) (by rw [hc]; exact hop)
    (by rw [hc]; exact hsp) (by rw [hc]; exact ho) (by rw [hc]; exact hres)
  obtain ⟨rows, what, hrows, hcv⟩ := residuals_of_check hck'
  rw [hc] at hl hcv
  have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
      some (rows, Flock.RecCoins.V_VALUE) := by
    rw [hf, valueRowsAll_load hso hop hl, hname]
    exact innerRows_v hsp ho hres hrows
  exact ⟨_, _, hv, value_tie_coll (zs := zs) (vsrc := valueSrcAll src c3) hcv hprog (valueSrcAll_eq hv hprog) hp
    (fun reg hr hr' => hone reg hr (hr'.trans hname.symm)) hpos hseg⟩

/-- **The value tie at a `rec-sparse` unit's `v`**: `value_tie_stacked` at `sparseRows`' rows at the inner session
`src t` check 5 loaded and the verifier's candidate-3 inputs `c3 t`. -/
theorem value_tie_sparse (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) (some f) (some c3') = .ok ()) (hf : src t = some f)
    (hc3 : c3 t = some c3') (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none)
    (hop : Flock.RecOpening.params (Vs j).c.unit.name = none) {sh : Flock.RecSparse.Shape}
    (hsp : Flock.RecSparse.params (Vs j).c.unit.name = some sh)
    (hname : ((Vs j).c.ports[q.val]!).name = "v")
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = ((Vs j).c.ports[q.val]!).name →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none) :
    ∃ rows value, valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value) ∧
      u.val < rows.size ∧
      ((∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q),
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          (rows[u.val]!)[b / 16]!.testBit (b % 16)) ∨
      ValueColl zs (valueSrcAll src c3) R ω t R₂) := by
  obtain ⟨inn, hl, hck'⟩ := sparseCheck_of_check hck (by rw [hc]; exact hso) (by rw [hc]; exact hop)
    (by rw [hc]; exact hsp)
  obtain ⟨rows, what, hrows, hcv⟩ := sparse_of_check hck'
  rw [hc] at hl hcv hrows
  have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
      some (rows, Flock.RecCoins.SPARSE_V_VALUE) := by
    rw [hf, hc3, valueRowsAll_load hso hop hl, hname]
    exact innerRows_sparse hsp hrows
  exact ⟨_, _, hv, value_tie_coll (zs := zs) (vsrc := valueSrcAll src c3) hcv hprog (valueSrcAll_eq hv hprog) hp
    (fun reg hr hr' => hone reg hr (hr'.trans hname.symm)) hpos hseg⟩

end Tie

section Session

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockSoundness.Discharge.ZkReg FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
open FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.Composed (drawJ partOf)

/-- **A session's statement check 5 accepted**: where the session accepts and its record draws, the statement setting
up a drawn unit's part has the session's circuit and passes `RecCoins.check`, as `ChecksC3` runs it. -/
theorem coins_of_session {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)
    {tr : Exec.Transcript} (hacc : AcceptsZKR zo.I zo.own tr) {coins : Option Flock.RecCoins.Source}
    {c3 : Option Flock.RecCoins.C3}
    (hck : ∃ spec sts, setupOf (zkInputs zo.I) tr.record = .ok (spec, sts) ∧
      sts.forM (Flock.RecCoins.check · (some zo.own) coins c3) = .ok ()) {d : Lean.Json}
    (hd : recordDraw tr.record = some d) (u : Fin nv) :
    ∃ sj : Flock.Stmt, sj.c = c ∧ Flock.RecCoins.check sj (some zo.own) coins c3 = .ok () := by
  obtain ⟨spec, sts, hs, -⟩ := hacc
  obtain ⟨spec', sts', hs', hfm⟩ := hck
  have hst : sts' = sts := congrArg Prod.snd (Except.ok.inj (hs'.symm.trans hs))
  rw [hst] at hfm
  obtain ⟨hsz, hstJ⟩ := ZkHidden.setupOf_tableJH zo.hTags.1 hd hs
  have hjs : (partOf zo.I d u.val).val < sts.size := by rw [hsz]; exact (partOf zo.I d u.val).isLt
  have s := Hidden.setupHidden_spec' (hstJ _ hjs)
  have hcst : sts[(partOf zo.I d u.val).val].c = c := by
    have h1 := s.parse
    rw [ZkHidden.parseH_tagsAt ((zkTags_typed zo.I).trans zo.ht), zo.hcH] at h1
    exact (Except.ok.inj h1).symm
  exact ⟨_, hcst, array_forM_ok hfm _ hjs⟩

end Session

end FlockSoundness.Discharge.Recursion

#print axioms FlockSoundness.Discharge.Recursion.value_tie_stacked

#print axioms FlockSoundness.Discharge.Recursion.value_tie_opening

#print axioms FlockSoundness.Discharge.Recursion.value_tie_open

#print axioms FlockSoundness.Discharge.Recursion.value_tie_residuals

#print axioms FlockSoundness.Discharge.Recursion.value_tie_sparse
