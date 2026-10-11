import Proofs.Flock.Soundness.Discharge.ZkExec.RunsStatements

/-!
# Ligerito under `--zk`, as frames

`zkLigerito_frames`: a `Zk.ligerito` run that succeeds on retained rounds emits `ligFramesZ`. Its steps are the non-ZK
session's (`Exec.ligerito_frames`) with three changes: `LABEL_OPEN` follows the cap; the last lane round's coin is
followed by level 0's hiding frames (`l0Fr`, through `laneFr`); and level 0's queries open rows `EXTRA_LANES` wider
(`queryLevel0_emit`), followed by a pure loop over the extra lanes that emits nothing. The final check is
`Refine.finalProg`'s text returning the lane phase's record (`finalProgZ`), and it emits nothing (`finalProgZ_ok`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Transcript

variable {β : Type}

/-! ## Level 0's queries -/

theorem queryLevel0_emit {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level} {cap : Array ByteArray}
    {fold : Array F256} (hq : lv.q ≠ 0)
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Flock.Transcript} {b : β}
    (h : (Zk.queryLevel0 ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) (hret : AllRetained t.rounds) :
    ∃ qn cn r t₁, p.queryNonces[st.qn]? = some qn ∧ p.consistencyNonces[st.cn]? = some cn ∧
      Emitted t t₁ [.bytes (le64 qn), .sq 2 lv.q, .bytes (le64 cn), .sq 2 ((lv.q - 1).log2 + 1)] ∧
      f r { st with qn := st.qn + 1, cn := st.cn + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold Zk.queryLevel0 at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, hqn, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, hcn, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  have e₃ := absorbBytes_emitted h₃
  obtain ⟨coins, t₄, h₄, h₁⟩ := lift_bind h₁
  obtain ⟨e₄, -⟩ := sampleVec_emitted hq h₄ (by rw [e₃.rounds]; exact hret)
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  have e₅ := absorbBytes_emitted h₅
  obtain ⟨alpha, t₆, h₆, h₁⟩ := lift_bind h₁
  obtain ⟨e₆, -⟩ := sampleVec_emitted (by omega) h₆ (by rw [e₅.rounds, e₄.rounds, e₃.rounds]; exact hret)
  cases hmc : merkleCheck ms cap lv.cap lv.d (positions lv coins) p.initialProof.rows p.initialProof.siblings
      p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    exact ⟨qn, cn, _, _, hqn, hcn, e₃.trans (e₄.trans (e₅.trans e₆)), h₂⟩

/-! ## The lane rounds -/

/-- Lane round `i`'s frames: its `K` coin, then (after the last lane round's coin) level 0's hiding frames, then the
round's message. -/
def laneFr (p : LigeritoProof) (z : Zk.Level0Msgs) (k0 i : ℕ) : List Frame :=
  .sq 2 2 :: ((if i + 1 = k0 then l0Fr z else []) ++ kFr p.messages[1 + i]!)

theorem laneFr_chFr (p : LigeritoProof) (z : Zk.Level0Msgs) (k0 : ℕ) :
    ∀ n a, a + n < k0 → (List.range' a n).flatMap (laneFr p z k0) = chFr p (1 + a) n
  | 0, _, _ => rfl
  | n + 1, a, h => by
      rw [List.range'_succ, List.flatMap_cons, laneFr_chFr p z k0 n (a + 1) (by omega),
        show 1 + (a + 1) = 1 + a + 1 by omega]
      simp only [laneFr, chFr, ite_eq_right (show ¬ a + 1 = k0 by omega), List.nil_append, List.cons_append]

theorem laneFr_flat (p : LigeritoProof) (z : Zk.Level0Msgs) (k0 : ℕ) (hk : 1 ≤ k0) :
    (List.range' 0 k0).flatMap (laneFr p z k0) = chFr p 1 (k0 - 1) ++ .sq 2 2 :: (l0Fr z ++ kFr p.messages[k0]!) := by
  obtain ⟨k, rfl⟩ : ∃ k, k0 = k + 1 := ⟨k0 - 1, by omega⟩
  rw [List.range'_concat, List.flatMap_append, laneFr_chFr p z (k + 1) k 0 (by omega)]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, laneFr, Nat.add_sub_cancel, Nat.zero_add,
    Nat.one_mul, Nat.add_zero, Nat.add_comm 1 k, ite_true]

/-! ## The final check, as `Zk.ligerito` runs it -/

/-- `Zk.ligerito`'s final check (live PROTOCOL.md §7, §13.6) and its record: `Refine.finalProg`'s text, returning the
lane phase's record. -/
def finalProgZ (p : LigeritoProof) (basisAt : Array F256 → Nat → Array F256) (n : Nat) (T : F128)
    (initCh : Array F256) (rho : F256 × F256) (levels : Array (Array F256)) (ext : Nat) : LM Zk.L0Rec := do
    let st ← get
    let orig := levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh
    if orig.size + ext != n then throw "ligerito: |orig| + ext ≠ n"
    let heads := levels.map (fun l => l[0]!)
    let prodFrom := fun (a : Nat) => (heads.extract a heads.size).foldl (fun acc c => acc * cff c) (1 : F256)
    let pi0 := prodFrom 0
    let bres := basisAt orig ext
    if bres.size != 2 ^ ext then throw "ligerito: basis residual length"
    let mut res : Array F256 := bres.map (· * pi0)
    let sv := svTable (n + 1)
    for o in st.oods do
      let (fixed, scale) := match o.split with
        | none => (orig, pi0)
        | some s => (levels.extract s levels.size |>.foldl (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) #[], prodFrom (s + 1))
      if o.z.size != fixed.size + ext then throw "ligerito: ood dimension"
      let head := (Array.range fixed.size).foldl (fun acc t => acc * ((1 : F256) + F256.ofF o.z[t]! + fixed[t]!)) (1 : F256)
      let c := (head * scale).smul o.beta
      res := (Array.range (2 ^ ext)).map fun y =>
        let tail := (Array.range ext).foldl (fun acc j =>
          let zj := o.z[fixed.size + j]!
          acc * (if (y >>> j) % 2 == 1 then zj else (1 : F128) + zj)) (1 : F128)
        res[y]! + c.smul tail
    for cs in st.conss do
      let fixed := (levels.extract cs.start levels.size).foldl (fun acc l => acc ++ l.extract 1 l.size) #[]
      if fixed.size + ext != cs.cols then throw "ligerito: consistency dimension"
      let ea := eqTable cs.alpha
      let scale := prodFrom cs.start
      let mut add : Array F256 := Array.replicate (2 ^ ext) 0
      for j in [0:cs.pos.size] do
        let wh := whats sv cs.pos[j]! cs.cols
        let pj := (Array.range fixed.size).foldl (fun acc i => acc * ((1 : F256) + fixed[i]! * F256.ofF ((1 : F128) + wh[i]!))) (1 : F256)
        let coef := pj.smul (ea[j]!)
        add := (Array.range (2 ^ ext)).map fun y =>
          let tj := (Array.range ext).foldl (fun acc i => if (y >>> i) % 2 == 1 then acc * wh[fixed.size + i]! else acc) (1 : F128)
          add[y]! + coef.smul tj
      res := (Array.range (2 ^ ext)).map fun y => res[y]! + (add[y]! * scale).smul cs.beta
    let yr := p.finalProof.yr
    let lhs := (Array.range (2 ^ ext)).foldl (fun acc y => acc + res[y]! * ⟨yr[2 * y]!, yr[2 * y + 1]!⟩) (0 : F256)
    if lhs != st.s then throw "ligerito: final check failed"
    if st.ood != p.oodValues.size || st.qn != p.queryNonces.size || st.cn != p.consistencyNonces.size
        || st.cbn != p.claimNonces.size then throw "ligerito: leftover proof lists"
    return { target := T, ood := #[((st.oods[0]!).beta, p.oodValues[0]!)], challenges := initCh, rho }

theorem finalProgZ_ok {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {n : ℕ} {T : F128}
    {initCh : Array F256} {rho : F256 × F256} {levels : Array (Array F256)} {ext : ℕ} {st st' : LState}
    {tr tr' : Flock.Transcript} {rec : Zk.L0Rec}
    (h : finalProgZ p basisAt n T initCh rho levels ext st tr = .ok ((rec, st'), tr')) : tr' = tr := by
  unfold finalProgZ at h
  have h1 := lget_bind h
  clear h
  dsimp only at h1
  change ((if ((origE initCh levels).size + ext != n) = true then _ else _ : LM Zk.L0Rec) st tr) = _ at h1
  by_cases c1 : ((origE initCh levels).size + ext != n) = true
  · rw [ite_eq_left c1] at h1; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h1; cases h0
  rw [ite_eq_right c1] at h1
  change ((if ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true then _ else _ : LM Zk.L0Rec) st tr) = _
    at h1
  by_cases c2 : ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true
  · rw [ite_eq_left c2] at h1; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h1; cases h0
  rw [ite_eq_right c2] at h1
  obtain ⟨res₁, st₁, tr₁, hl₁, h2⟩ := lbind_ok h1
  clear h1
  rw [← Array.forIn_toList] at hl₁
  obtain ⟨-, rfl, rfl, -⟩ := forIn_yield_fold _
    (fun res o => oodStep ext (Refine.oodFixed levels (origE initCh levels) o) o res)
    (fun o => o.z.size = (Refine.oodFixed levels (origE initCh levels) o).1.size + ext) (by
      intro o s st tr b st' tr' hb
      change ((if (o.z.size != (Refine.oodFixed levels (origE initCh levels) o).1.size + ext) = true then _ else _ :
        LM (ForInStep (Array F256))) st tr) = _ at hb
      by_cases cz : (o.z.size != (Refine.oodFixed levels (origE initCh levels) o).1.size + ext) = true
      · rw [ite_eq_left cz] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right cz] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      simp only [bne_iff_ne, ne_eq, not_not] at cz
      exact ⟨rfl, rfl, rfl, cz⟩) _ _ _ _ _ _ _ hl₁
  obtain ⟨res₂, st₂, tr₂, hl₂, h3⟩ := lbind_ok h2
  clear h2
  rw [← Array.forIn_toList] at hl₂
  obtain ⟨-, rfl, rfl, -⟩ := forIn_yield_fold _ (consStep (svTable (n + 1)) ext levels)
    (fun cs => (Refine.consFixed levels cs.start).size + ext = cs.cols) (by
      intro cs s st tr b st' tr' hb
      change ((if ((Refine.consFixed levels cs.start).size + ext != cs.cols) = true then _ else _ :
        LM (ForInStep (Array F256))) st tr) = _ at hb
      by_cases cz : ((Refine.consFixed levels cs.start).size + ext != cs.cols) = true
      · rw [ite_eq_left cz] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right cz] at hb
      obtain ⟨add, st₃, tr₃, hadd, hb'⟩ := lbind_ok hb
      clear hb
      rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hadd
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hadd
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
      simp only [bne_iff_ne, ne_eq, not_not] at cz
      exact ⟨rfl, rfl, rfl, cz⟩) _ _ _ _ _ _ _ hl₂
  change ((if (lhsE p.finalProof.yr ext res₂ != st₂.s) = true then _ else _ : LM Zk.L0Rec) st₂ tr₂) = _ at h3
  by_cases c3 : (lhsE p.finalProof.yr ext res₂ != st₂.s) = true
  · rw [ite_eq_left c3] at h3; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h3; cases h0
  rw [ite_eq_right c3] at h3
  by_cases c4 : (st₂.ood != p.oodValues.size || st₂.qn != p.queryNonces.size ||
      st₂.cn != p.consistencyNonces.size || st₂.cbn != p.claimNonces.size) = true
  · rw [ite_eq_left c4] at h3; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h3; cases h0
  rw [ite_eq_right c4] at h3
  obtain ⟨-, -, rfl⟩ := lpure_ok h3
  rfl

/-! ## Ligerito under `--zk` -/

set_option maxHeartbeats 4000000 in
theorem zkLigerito_frames (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) (hk0 : 1 ≤ sch.k0)
    {t t' : Flock.Transcript} {rc : Zk.L0Rec} (hret : AllRetained t.rounds)
    (h : Zk.ligerito ms sch T cap p basisAt zl t = .ok (rc, t')) :
    Emitted t t' (ligFramesZ sch T cap p zl.msgs) := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  unfold Zk.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨hzl, h⟩ := check_bind h
  obtain ⟨_, t₁, h₁, h⟩ := bind_ok h
  have e₁ := label_emitted h₁
  obtain ⟨_, t₂, h₂, h⟩ := bind_ok h
  have e₂ := absorbF_emitted h₂
  obtain ⟨_, t₃, h₃, h⟩ := bind_ok h
  have e₃ := absorbBytes_emitted h₃
  obtain ⟨_, t₃', h₃', h⟩ := bind_ok h
  have eO := label_emitted h₃'
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  have r₃ := eO.retained (e₃.retained (e₂.retained (e₁.retained hret)))
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨e₅, -⟩ := sampleVec_emitted (by omega) h₅ r₃
  obtain ⟨y₀, hy₀, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have e₆ := absorbF_emitted h₆
  obtain ⟨n₀, t₇, hn₀, e₇, h₄⟩ := claimPow_emit h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨e₈, -⟩ := sample_emitted h₈ (e₇.retained (e₆.retained (e₅.retained r₃)))
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, hm₀, e₉, h₄⟩ := nextMsg_emit h₄
  have h₄ := lmodify_bind h₄
  have r₉ := e₉.retained (e₈.retained (e₇.retained (e₆.retained (e₅.retained r₃))))
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨⟨g1, g2, g3, g4, g5⟩, e₁₀⟩ := lforIn_emitted _ (laneFr p zl.msgs sch.k0)
    (fun i _ st => st.msg = 1 + i ∧ st.ood = 1 ∧ st.cbn = 1 ∧ st.qn = 0 ∧ st.cn = 0) sch.k0 (by
      intro i v st tE bv st' tE' hiB hP hretE hb
      obtain ⟨hmsg, hood, hcbn, hqn, hcn⟩ := hP
      obtain ⟨r, tR, hR', hb⟩ := lift_bind hb
      obtain ⟨eR, -⟩ := sampleK_emitted hR' hretE
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        have f₂ := absorbK_emitted h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        have f₃ := absorbK_emitted h₃
        obtain ⟨ρ₀, t₄, h₄, hb⟩ := lift_bind hb
        obtain ⟨f₄, -⟩ := sampleK_emitted h₄ (f₃.retained (f₂.retained (eR.retained hretE)))
        obtain ⟨ρ₁, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨f₅, -⟩ := sampleK_emitted h₅ (f₄.retained (f₃.retained (f₂.retained (eR.retained hretE))))
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, t₆, h₆, hb⟩ := lift_bind hb
        have f₆ := label_emitted h₆
        obtain ⟨_, t₇, h₇, hb⟩ := lift_bind hb
        have f₇ := absorbK_emitted h₇
        obtain ⟨_, t₈, h₈, hb⟩ := lift_bind hb
        have f₈ := absorbSlice_emitted h₈
        have hb := lmodify_bind hb
        obtain ⟨m, t₉, hm, f₉, hb⟩ := nextMsg_emit hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        have hmm : m = p.messages[1 + i]! := by rw [← getElem!_of_getElem? hm, hmsg]
        refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · simp only; omega
        · simp only; omega
        · simp only; omega
        · simp only; omega
        · simp only; omega
        · have e := eR.trans (f₂.trans (f₃.trans (f₄.trans (f₅.trans (f₆.trans (f₇.trans (f₈.trans f₉)))))))
          rw [hmm] at e
          show Emitted _ _ (.sq 2 2 :: ((if i + 1 = sch.k0 then l0Fr zl.msgs else []) ++ kFr p.messages[1 + i]!))
          rw [ite_eq_left hj]
          simpa [l0Fr, kFr] using e
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, t₉, hm, f₉, hb⟩ := nextMsg_emit hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        have hmm : m = p.messages[1 + i]! := by rw [← getElem!_of_getElem? hm, hmsg]
        refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · simp only; omega
        · simp only; omega
        · simp only; omega
        · simp only; omega
        · simp only; omega
        · have e := eR.trans f₉
          rw [hmm] at e
          show Emitted _ _ (.sq 2 2 :: ((if i + 1 = sch.k0 then l0Fr zl.msgs else []) ++ kFr p.messages[1 + i]!))
          rw [ite_eq_right hj]
          simpa using e) sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> rfl) r₉ h₁₀
  have r₁₀ := e₁₀.retained r₉
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have e₁₁ := absorbBytes_emitted h₁₁
  obtain ⟨yA, mA, nA, stA, tA, hyA, hmA, hnA, a1, a2, a3, a4, a5, eA, h₄⟩ :=
    oodIntro_emit (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄ (e₁₁.retained r₁₀)
  obtain ⟨yB, mB, nB, stB, tB, hyB, hmB, hnB, b1, b2, b3, b4, b5, eB, h₄⟩ :=
    oodIntro_emit (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄ (eA.retained (e₁₁.retained r₁₀))
  have hlv0 := hlv 0 (by omega)
  obtain ⟨qn₀, cn₀, ⟨pos0, al0, e0⟩, tQ, hqn₀, hcn₀, eQ, h₄⟩ :=
    queryLevel0_emit hlv0.q_ne_zero h₄ (eB.retained (eA.retained (e₁₁.retained r₁₀)))
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, tM, hmq, eM, h₄⟩ := nextMsg_emit h₄'
  obtain ⟨nq, tN, hnq, eN, h₄⟩ := claimPow_emit h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨e₁₇, -⟩ := sample_emitted h₁₇ (eN.retained (eM.retained (eQ.retained (eB.retained (eA.retained
    (e₁₁.retained r₁₀))))))
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have r₁₇ := e₁₇.retained (eN.retained (eM.retained (eQ.retained (eB.retained (eA.retained (e₁₁.retained r₁₀))))))
  obtain ⟨-, eL⟩ := lforIn_emitted _ (levelFr sch p)
    (fun i v st => i < sch.levels.size - 1 → st.msg = sch.k0 + 4 + 7 * i ∧ st.ood = 3 + 2 * i ∧ st.cbn = 4 + 3 * i ∧
      st.qn = 1 + i ∧ st.cn = 1 + i ∧ v.1 = (sch.levels.getD (i + 1) default).cols + 4) (sch.levels.size - 1) (by
      intro i v st tE bv st' tE' hiB hP hretE hb
      obtain ⟨hmsg, hood, hcbn, hqn, hcn, hsplit⟩ := hP hiB
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨c1, c2, c3, c4, c5, eC⟩ := challenges_emitted p 4 0 _ _ _ _ _ _ hretE hf
      by_cases hi : i + 1 < sch.levels.size - 1
      · have hc1 := hcols (i + 1) (by omega)
        rw [ite_eq_left hi] at hb
        obtain ⟨_, t₂, h₂', hb⟩ := lift_bind hb
        have eK := absorbBytes_emitted h₂'
        obtain ⟨yA, mA, nA, stA, tA, hyA, hmA, hnA, a1, a2, a3, a4, a5, eA, hb⟩ :=
          oodIntro_emit (show v.1 - 4 + 1 ≠ 0 by omega) hb (eK.retained (eC.retained hretE))
        obtain ⟨yB, mB, nB, stB, tB, hyB, hmB, hnB, b1, b2, b3, b4, b5, eB, hb⟩ :=
          oodIntro_emit (show v.1 - 4 + 1 ≠ 0 by omega) hb (eA.retained (eK.retained (eC.retained hretE)))
        have hlvi := hlv (i + 1) (by omega)
        obtain ⟨qnI, cnI, rI, tQ, hqnI, hcnI, eQ, hb⟩ :=
          queryLevel_emit hlvi.q_ne_zero hb (eB.retained (eA.retained (eK.retained (eC.retained hretE))))
        obtain ⟨mq, tM, hmq, eM, hb⟩ := nextMsg_emit hb
        obtain ⟨nq, tN, hnq, eN, hb⟩ := claimPow_emit hb
        obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
        obtain ⟨e₈, -⟩ := sample_emitted h₈ (eN.retained (eM.retained (eQ.retained (eB.retained (eA.retained
          (eK.retained (eC.retained hretE)))))))
        have hb := lmodify_bind hb
        rw [ite_eq_right (show ¬ v.1 - 4 + 1 < 4 by omega)] at hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨_, rfl, fun hi' => ⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · simp only; rw [b1, a1, c1, hmsg]; omega
        · simp only; rw [b2, a2, c2, hood]; omega
        · simp only; rw [b3, a3, c3, hcbn]; omega
        · simp only; rw [b4, a4, c4, hqn]; omega
        · simp only; rw [b5, a5, c5, hcn]; omega
        · simp only; omega
        · have e := eC.trans (eK.trans (eA.trans (eB.trans (eQ.trans (eM.trans (eN.trans e₈))))))
          have hyA' : yA = p.oodValues[3 + 2 * i]! := by rw [← getElem!_of_getElem? hyA, c2, hood]
          have hmA' : mA = p.messages[sch.k0 + 8 + 7 * i]! := by
            rw [← getElem!_of_getElem? hmA, c1, hmsg]; congr 1; omega
          have hnA' : nA = p.claimNonces[4 + 3 * i]! := by rw [← getElem!_of_getElem? hnA, c3, hcbn]
          have hyB' : yB = p.oodValues[4 + 2 * i]! := by
            rw [← getElem!_of_getElem? hyB, a2, c2, hood]; congr 1; omega
          have hmB' : mB = p.messages[sch.k0 + 9 + 7 * i]! := by
            rw [← getElem!_of_getElem? hmB, a1, c1, hmsg]; congr 1; omega
          have hnB' : nB = p.claimNonces[5 + 3 * i]! := by
            rw [← getElem!_of_getElem? hnB, a3, c3, hcbn]; congr 1; omega
          have hqnI' : qnI = p.queryNonces[1 + i]! := by rw [← getElem!_of_getElem? hqnI, b4, a4, c4, hqn]
          have hcnI' : cnI = p.consistencyNonces[1 + i]! := by rw [← getElem!_of_getElem? hcnI, b5, a5, c5, hcn]
          have hmq' : mq = p.messages[sch.k0 + 10 + 7 * i]! := by
            rw [← getElem!_of_getElem? hmq]; simp only; rw [b1, a1, c1, hmsg]; congr 1; omega
          have hnq' : nq = p.claimNonces[6 + 3 * i]! := by
            rw [← getElem!_of_getElem? hnq]; simp only; rw [b3, a3, c3, hcbn]; congr 1; omega
          rw [hmsg, hsplit, hyA', hmA', hnA', hyB', hmB', hnB', hqnI', hcnI', hmq', hnq'] at e
          simpa [levelFr, hi, introFr, queryFr, capFr, kFr, List.append_assoc] using e
      · rw [ite_eq_right hi] at hb
        have hb := lget_bind hb
        by_cases hmc₀ : (st₁.msg != p.messages.size) = true
        · rw [ite_eq_left hmc₀] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hmc₀] at hb
        by_cases hyr : (p.finalProof.yr.size != 2 ^ (v.1 - 4 + 1)) = true
        · rw [ite_eq_left hyr] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hyr] at hb
        obtain ⟨_, st₂, t₂, h₂', hq⟩ := lbind_ok hb
        rw [← Array.forIn_toList] at h₂'
        obtain ⟨hst₂, eY⟩ := forIn_absorbF_emitted _ h₂'
        have hlvR := hlv (sch.levels.size - 1) (by omega)
        obtain ⟨qnI, cnI, rI, tQ, hqnI, hcnI, eQ, hq⟩ :=
          queryLevel_emit hlvR.q_ne_zero hq (eY.retained (eC.retained hretE))
        obtain ⟨nq, tN, hnq, eN, hq⟩ := claimPow_emit hq
        obtain ⟨bq, t₈, h₈, hq⟩ := lift_bind hq
        obtain ⟨e₈, -⟩ := sample_emitted h₈ (eN.retained (eQ.retained (eY.retained (eC.retained hretE))))
        have hq := lmodify_bind hq
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hq
        refine ⟨_, rfl, fun hi' => absurd hi' hi, ?_⟩
        have e := eC.trans (eY.trans (eQ.trans (eN.trans e₈)))
        have hqnI' : qnI = p.queryNonces[1 + i]! := by rw [← getElem!_of_getElem? hqnI, hst₂, c4, hqn]
        have hcnI' : cnI = p.consistencyNonces[1 + i]! := by rw [← getElem!_of_getElem? hcnI, hst₂, c5, hcn]
        have hnq' : nq = p.claimNonces[4 + 3 * i]! := by
          rw [← getElem!_of_getElem? hnq]; simp only; rw [hst₂, c3, hcbn]
        rw [hmsg, hqnI', hcnI', hnq'] at e
        simpa [levelFr, hi, queryFr, List.append_assoc] using e)
    (sch.levels.size - 1) 0 _ _ t₁₇ _ _ t₁₈ (by omega) (fun _ => by
      simp only at g1 g2 g3 g4 g5 ⊢
      exact ⟨by omega, by omega, by omega, by omega, by omega, by omega⟩) r₁₇ hloop
  obtain rfl := finalProgZ_ok hcont
  have e := e₁.trans (e₂.trans (e₃.trans (eO.trans (e₅.trans (e₆.trans (e₇.trans (e₈.trans (e₉.trans (e₁₀.trans
    (e₁₁.trans (eA.trans (eB.trans (eQ.trans (eM.trans (eN.trans (e₁₇.trans eL))))))))))))))))
  rw [laneFr_flat p zl.msgs sch.k0 hk0] at e
  simp only at hy₀ hn₀ hm₀ g1 g2 g3 g4 g5 hyA hmA hnA hyB hmB hnB hqn₀ hcn₀ hmq hnq a1 a2 a3 a4 a5 b1 b2 b3 b4 b5
  have iy₀ : y₀ = p.oodValues[0]! := by rw [← getElem!_of_getElem? hy₀]
  have in₀ : n₀ = p.claimNonces[0]! := by rw [← getElem!_of_getElem? hn₀]
  have im₀ : m₀ = p.messages[0]! := by rw [← getElem!_of_getElem? hm₀]
  have iyA : yA = p.oodValues[1]! := by rw [← getElem!_of_getElem? hyA, g2]
  have imA : mA = p.messages[sch.k0 + 1]! := by rw [← getElem!_of_getElem? hmA, g1]; congr 1; omega
  have inA : nA = p.claimNonces[1]! := by rw [← getElem!_of_getElem? hnA, g3]
  have iyB : yB = p.oodValues[2]! := by rw [← getElem!_of_getElem? hyB, a2, g2]
  have imB : mB = p.messages[sch.k0 + 2]! := by rw [← getElem!_of_getElem? hmB, a1, g1]; congr 1; omega
  have inB : nB = p.claimNonces[2]! := by rw [← getElem!_of_getElem? hnB, a3, g3]
  have iqn₀ : qn₀ = p.queryNonces[0]! := by rw [← getElem!_of_getElem? hqn₀, b4, a4, g4]
  have icn₀ : cn₀ = p.consistencyNonces[0]! := by rw [← getElem!_of_getElem? hcn₀, b5, a5, g5]
  have imq : mq = p.messages[sch.k0 + 3]! := by rw [← getElem!_of_getElem? hmq, b1, a1, g1]; congr 1; omega
  have inq : nq = p.claimNonces[3]! := by rw [← getElem!_of_getElem? hnq, b3, a3, g3]
  rw [iy₀, in₀, im₀, iyA, imA, inA, iyB, imB, inB, iqn₀, icn₀, imq, inq] at e
  simpa only [ligFramesZ, introFr, queryFr, capFr, List.append_assoc, List.cons_append, List.singleton_append,
    List.nil_append] using e

end FlockSoundness.Discharge.ZkExec
