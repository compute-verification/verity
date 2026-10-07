import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLig
import Proofs.Flock.Soundness.Refine.Final

/-!
# `--zk` Ligerito's final check is the model's

`finalProgZ` (`RunsLig.lean`) is `Refine.finalProg`'s text returning the lane phase's record. A run of it that
succeeds establishes `finalProg_ok`'s facts (`finalProgZ_facts`), so the model's final check holds on its state
(`final_refinesZ`, as the non-ZK session's `final_refines`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock FlockLevel3 Refine Transcript Model

theorem finalProgZ_facts {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {n : ℕ} {T : F128}
    {initCh : Array F256} {rho : F256 × F256} {levels : Array (Array F256)} {ext : ℕ} {st st' : LState}
    {tr tr' : Flock.Transcript} {rec : Zk.L0Rec}
    (h : finalProgZ p basisAt n T initCh rho levels ext st tr = .ok ((rec, st'), tr')) :
    st' = st ∧ tr' = tr ∧ (origE initCh levels).size + ext = n ∧
      (basisAt (origE initCh levels) ext).size = 2 ^ ext ∧
      (∀ o ∈ st.oods.toList, o.z.size = (Refine.oodFixed levels (origE initCh levels) o).1.size + ext) ∧
      (∀ cs ∈ st.conss.toList, (Refine.consFixed levels cs.start).size + ext = cs.cols) ∧
      lhsE p.finalProof.yr ext (resE basisAt n initCh levels ext st) = st.s := by
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
  obtain ⟨hres₁, rfl, rfl, hood⟩ := forIn_yield_fold _
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
  obtain ⟨hres₂, rfl, rfl, hcons⟩ := forIn_yield_fold _ (consStep (svTable (n + 1)) ext levels)
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
  obtain ⟨-, rfl, rfl⟩ := lpure_ok h3
  simp only [bne_iff_ne, ne_eq, not_not] at c1 c2 c3
  refine ⟨rfl, rfl, c1, c2, hood, hcons, ?_⟩
  rw [← c3, hres₂, hres₁]
  rfl

/-- **The `--zk` final check is the model's** (§13.6): `final_refines` for `finalProgZ`. -/
theorem final_refinesZ {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {n : ℕ} {T : F128}
    {initCh : Array F256} {rho : F256 × F256} {levels : Array (Array F256)} {ext : ℕ} {st st' : LState}
    {tr tr' : Flock.Transcript} {rec : Zk.L0Rec} (b : ℕ → FlockLevel3.GF128)
    (hl : ∀ l ∈ levels.toList, l.size = 4)
    (hpos : ∀ cs ∈ st.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hbasis : ∀ y < 2 ^ ext, gOf (basisAt (origE initCh levels) ext)[y]! =
      Model.bHat execArith n b ((origE initCh levels).toList.map gOf ++ bitsK ext y))
    (h : finalProgZ p basisAt n T initCh rho levels ext st tr = .ok ((rec, st'), tr')) :
    tr' = tr ∧ Model.finalCheck execArith n b (initCh.toList.map gOf) (levelsOf levels) (st.oods.toList.map oodOf)
      (st.conss.toList.map consOf) ext (yrOf p ext) (gOf st.s) = true := by
  obtain ⟨-, rfl, horig, hbres, hood, hcons, hlhs⟩ := finalProgZ_facts h
  refine ⟨rfl, ?_⟩
  unfold Model.finalCheck
  simp only [Bool.and_eq_true, decide_eq_true_eq, ← origE_toList, List.length_map, Array.length_toList]
  refine ⟨horig, ?_⟩
  rw [← hlhs, gOf_lhsE]
  refine Finset.sum_congr rfl fun y hy => ?_
  rw [Finset.mem_range] at hy
  have hy2 : 2 * y < 2 ^ (ext + 1) := by rw [pow_succ]; omega
  have hy3 : 2 * y + 1 < 2 ^ (ext + 1) := by rw [pow_succ]; omega
  have hyr0 : yrOf p ext ⟨2 * y % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩ = fOf p.finalProof.yr[2 * y]! := by
    simp only [yrOf, Nat.mod_eq_of_lt hy2]
  have hyr1 : yrOf p ext ⟨(2 * y + 1) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩ =
      fOf p.finalProof.yr[2 * y + 1]! := by
    simp only [yrOf, Nat.mod_eq_of_lt hy3]
  rw [hyr0, hyr1, mul_comm execArith.u]
  refine congrArg (· * _) ?_
  symm
  unfold resE
  rw [consFold_get _ _ _ hy, gOf_foldl_add, oodFold_get _ _ hy, gOf_foldl_add,
    getElem!_pos _ y (by simpa [hbres] using hy), Array.getElem_map, gOf_mul,
    ← getElem!_pos _ y (by simpa [hbres] using hy), hbasis y hy, prodFromE_eq levels hl]
  refine congrArg₂ (· + ·) (congrArg₂ (· + ·) rfl ?_) ?_
  · rw [List.map_map]
    refine congrArg List.sum (List.map_congr_left fun o ho => ?_)
    have hz := hood o ho
    cases hsp : o.split with
    | none =>
        have hfs : Refine.oodFixed levels (origE initCh levels) o = (origE initCh levels, prodFromE levels 0) := by
          simp [Refine.oodFixed, hsp]
        rw [hfs] at hz
        simp only [Function.comp, hfs]
        rw [gOf_oodTerm ext _ o y hz, prodFromE_eq levels hl]
        simp only [oodOf, hsp]
    | some a =>
        have hfs : Refine.oodFixed levels (origE initCh levels) o = ((levels.extract a levels.size).foldl
            (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) #[], prodFromE levels (a + 1)) := by
          simp [Refine.oodFixed, hsp]
        rw [hfs] at hz
        simp only [Function.comp, hfs]
        rw [gOf_oodTerm ext _ o y hz, prodFromE_eq levels hl, Refine.oodFixed_some levels hl a]
        simp only [oodOf, hsp]
  · rw [List.map_map]
    refine congrArg List.sum (List.map_congr_left fun cs hcs' => ?_)
    have hc := hcons cs hcs'
    have hle := Refine.consFixed_size_le initCh levels cs.start
    simp only [Function.comp]
    rw [gOf_consTerm (n + 1) ext levels hl cs y hy hc (by omega) (hpos cs hcs')]
    simp only [consOf]

end FlockSoundness.Discharge.ZkExec
