import Proofs.Flock.Claims.KS.LevelsAll
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictOpens

/-!
# Every level's request in an accepted masked rep

`zk_ligerito_reqs` and `zk_rep_reqs` fix a masked rep's first model request: level 0's, at the positions of the
coins of round `k + k0 + 8` of Ligerito's. Here the same run fixes every later one, as `LevelsAll` does for the plain
rep: the requests after it are levels `1, …, R` in order, level `ℓ`'s at the positions of the coins of round
`k + 2 + levRound k0 R ℓ` (`LaterAt`), Ligerito starting at round `k`; the masked lane phase's two `ρ` coins put every
level two rounds after the plain one.

* `qbOut_reqs_laterA`: `qbOut_reqs_later` after any requests.
* `zk_ligerito_reqsL`, `zk_rep_reqsL`: `zk_ligerito_reqs` and `zk_rep_reqs`, with the later requests `LaterAt`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **A query step at level `i + 1` appends that level's request** after any requests `A`: `qbOut_reqs_later`. -/
theorem qbOut_reqs_laterA {D : Type} (r : ℕ) (sch : Flock.Schedule) (i : ℕ)
    (hl : LevelSeam (sch.levels.getD (i + 1) default)) (ch : List 𝕂) (start : ℕ) (stM : LigStateC 𝔽 𝕂 D)
    (cw cα cβ : Array F128) (w : Bool) (m : F256 × F256) {A rest : List Req} {rounds : Array Flock.Round}
    {k : ℕ} (h : stM.reqs = A ++ rest) (hlen : rest.length = i) (hlev : stM.level = i + 1)
    (hlat : LaterAt sch rounds k rest)
    (hcw : cw = (rounds[k + FlockDeploy.KS.levRound sch.k0 (sch.levels.size - 1) (i + 1)]!).coins) :
    ∃ rest', (qbOut r (levelOf (sch.levels.getD (i + 1) default)) ch start stM cw cα cβ w m).reqs = A ++ rest' ∧
      rest'.length = i + 1 ∧ LaterAt sch rounds k rest' := by
  refine ⟨rest ++ [⟨stM.level, Model.positions execArith (levelOf (sch.levels.getD (i + 1) default)).queries
    (levelOf (sch.levels.getD (i + 1) default)).logLen
    (vecOf (levelOf (sch.levels.getD (i + 1) default)).queries cw), ch.length,
    pathLen (levelOf (sch.levels.getD (i + 1) default))⟩], by simp only [qbOut, h, List.append_assoc],
    by simp [hlen], fun j hj => ?_⟩
  rcases Nat.lt_or_ge j rest.length with hj' | hj'
  · rw [List.getElem_append_left hj']
    exact hlat j hj'
  · have hji : j = i := by simp at hj; omega
    subst hji
    rw [List.getElem_append_right hj']
    simp only [hlen, Nat.sub_self, List.getElem_cons_zero]
    refine ⟨hlev, ?_⟩
    rw [hcw]
    exact positions_eq _ hl _

set_option maxHeartbeats 4000000 in
/-- **The hiding Ligerito's run records every level's request** (`zk_ligerito_reqs`): its requests passed, the first
is level 0's at the positions of the coins of round `k + k0 + 8`, and the ones after it are levels `1, …, R` in
order, level `ℓ`'s at the positions of the coins of round `k + 2 + levRound k0 R ℓ` (`LaterAt`), where `k` is the
round Ligerito starts at. -/
theorem zk_ligerito_reqsL (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (t t' : Flock.Transcript)
    (rec : Zk.L0Rec) (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1) (hk0 : 1 ≤ sch.k0)
    (hwf : RoundsWF t.rounds) (h : Zk.ligerito ms sch T cap p basisAt zl t = .ok (rec, t')) :
    ∀ {np : ℕ} (P : ZkSession.Pads0 np 2) (c₀ : 𝔽) (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽),
      ∃ out : ZkSession.LinK 𝕂 np × PendL 𝔽 Dig,
        Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P (schedOf sch) zl.tPad c₀ r Tm b)
          (ligMsgsZ p capDig sch zl.tPad zl.msgs) (consumed t.rounds t.k t'.k) out ∧
        out.2.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
        (∀ req ∈ out.2.reqs, ReqOKZ ms sch cap p req) ∧
        ∃ (R : Req) (rest : List Req), out.2.reqs = R :: rest ∧ R.level = 0 ∧
          R.positions =
            (Flock.positions (sch.levels.getD 0 default) (t.rounds[t.k + 2 + sch.k0 + 6]!).coins).toList ∧
          rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds (t.k + 2) rest := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  unfold Zk.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨hzl, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨hn₅, rfl, hr₅, hk₅⟩ := sampleVec_ok (by omega) h₅
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀, hn₁₀⟩ := lane_coins _ sch.k0 (by
      intro i v st tE bv st' tE' hiB hb
      obtain ⟨r, tR, hR', hb⟩ := lift_bind hb
      obtain ⟨-, hnR, -, hrR, hkR⟩ := sampleK_ok hR'
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        obtain ⟨f₂k, f₂r⟩ := absorbK_ok h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        obtain ⟨f₃k, f₃r⟩ := absorbK_ok h₃
        obtain ⟨ρ₀, t₄, h₄, hb⟩ := lift_bind hb
        obtain ⟨-, hn₄, -, hr₄, hk₄⟩ := sampleK_ok h₄
        obtain ⟨ρ₁, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨-, hn₅, -, hr₅, hk₅⟩ := sampleK_ok h₅
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, t₆, h₆, hb⟩ := lift_bind hb
        obtain ⟨f₆k, f₆r⟩ := label_ok h₆
        obtain ⟨_, t₇, h₇, hb⟩ := lift_bind hb
        obtain ⟨f₇k, f₇r⟩ := absorbK_ok h₇
        obtain ⟨_, t₈, h₈, hb⟩ := lift_bind hb
        obtain ⟨f₈k, f₈r⟩ := absorbSlice_ok h₈
        have hb := lmodify_bind hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨_, rfl, by rw [f₉r, f₈r, f₇r, f₆r, hr₅, hr₄, f₃r, f₂r, hrR], by rw [ite_eq_left hj]; omega, hnR,
          fun _ => ⟨?_, ?_⟩⟩
        · rw [show tE.k + 1 = t₃.k by omega, show tE.rounds = t₃.rounds by rw [f₃r, f₂r, hrR]]
          exact hn₄
        · rw [show tE.k + 2 = t₄.k by omega, show tE.rounds = t₄.rounds by rw [hr₄, f₃r, f₂r, hrR]]
          exact hn₅
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        exact ⟨_, rfl, by rw [f₉r, hrR], by rw [ite_eq_right hj]; omega, hnR, fun h => absurd h hj⟩)
    sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by omega) h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨hnA, hnB, yA, mA, t₁₂, -, -, hr₁₂, hk₁₂, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨hnC, hnD, yB, mB, t₁₃, -, -, hr₁₃, hk₁₃, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  have hlv0 := hlv 0 (by omega)
  have hmc0 := queryLevel0_mc hlv0.q_ne_zero h₄
  obtain ⟨hnQ, hnQ', ⟨pos0, al0, e0⟩, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel0_bind hlv0.q_ne_zero h₄
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₁₇, rfl, hr₁₇, hk₁₇⟩ := sample_ok h₁₇
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hfin := finalProgZ_ok hcont
  rw [hfin]
  have er₃ : t₃'.rounds = t.rounds := by rw [k₃'.2, k₃.2, k₂.2, k₁.2]
  have ek₃ : t₃'.k = t.k := by rw [k₃'.1, k₃.1, k₂.1, k₁.1]
  have er₇ : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, er₃]
  have ek₇ : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, ek₃]
  have er₉ : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, er₇]
  have ek₉ : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, ek₇]
  have er₁₁ : t₁₁.rounds = t.rounds := by rw [k₁₁.2, hr₁₀, er₉]
  have ek₁₁ : t₁₁.k = t.k + 2 + sch.k0 + 2 := by rw [k₁₁.1, hk₁₀, ek₉]
  have er₁₂ : t₁₂.rounds = t.rounds := hr₁₂.trans er₁₁
  have ek₁₂ : t₁₂.k = t.k + 2 + sch.k0 + 4 := by rw [hk₁₂, ek₁₁]
  have er₁₃ : t₁₃.rounds = t.rounds := hr₁₃.trans er₁₂
  have ek₁₃ : t₁₃.k = t.k + 2 + sch.k0 + 6 := by rw [hk₁₃, ek₁₂]
  have er₁₆ : t₁₆.rounds = t.rounds := by rw [k₁₆.2, k₁₅.2, hr₁₄, er₁₃]
  have ek₁₆ : t₁₆.k = t.k + 2 + sch.k0 + 8 := by rw [k₁₆.1, k₁₅.1, hk₁₄, ek₁₃]
  have er₁₇ : t₁₇.rounds = t.rounds := hr₁₇.trans er₁₆
  have ek₁₇ : t₁₇.k = t.k + 2 + sch.k0 + 9 := by rw [hk₁₇, ek₁₆]
  have hd1 : (sch.levels.toList.map levelOf).drop 1 =
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := drop_levels sch 1 (by omega)
  have hL : ∀ (r : ℕ) (b : ℕ → 𝔽) (ini : List 𝕂) (stM : LigStateC 𝔽 𝕂 Dig), stM.level = 1 →
      stM.caps = [capDig p.recursiveCaps[0]!] → (∀ req ∈ stM.reqs, ReqOKZ ms sch cap p req) →
      ∃ pend : PendL 𝔽 Dig, t₁₈.rounds = t₁₇.rounds ∧ t₁₇.k ≤ t₁₈.k ∧
      Plays fOf (Model.recursiveRoundsC execArith r (sch.m - 7) b ini sch.ext
        (levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2) stM)
        (recMsgs p capDig sch.ext (sch.levels.size - 1) 0 (sch.k0 + 4) 3) (consumed t₁₇.rounds t₁₇.k t₁₈.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOKZ ms sch cap p req) ∧ ∃ rest, pend.reqs = stM.reqs ++ rest ∧
        rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds (t.k + 2) rest := by
    intro r b ini stM₀ hlev1 hcaps1 hreqs1
    have hLC := loop_compose _ (sch.levels.size - 1)
      (fun i stM => Model.recursiveRoundsC execArith r (sch.m - 7) b ini sch.ext
        ((sch.levels.toList.map levelOf).drop (i + 1)) stM)
      (fun i v _ tE stM => RoundsWF tE.rounds ∧ v.1 = (sch.levels.getD (i + 1) default).cols + 4 ∧
        v.2.1 = p.recursiveCaps[i]! ∧ stM.level = i + 1 ∧
        stM.caps = (List.range (i + 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
        (∀ req ∈ stM.reqs, ReqOKZ ms sch cap p req) ∧ tE.rounds = t.rounds ∧
        tE.k = t.k + 2 + sch.k0 + 9 + 11 * i ∧
        ∃ rest, stM.reqs = stM₀.reqs ++ rest ∧ rest.length = i ∧ LaterAt sch t.rounds (t.k + 2) rest)
      (fun i _ => recMsgs p capDig sch.ext (sch.levels.size - 1 - i) i (sch.k0 + 4 + 7 * i) (3 + 2 * i))
      (fun _ _ _ pend => pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
        (∀ req ∈ pend.reqs, ReqOKZ ms sch cap p req) ∧ ∃ rest, pend.reqs = stM₀.reqs ++ rest ∧
          rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds (t.k + 2) rest) ?hstep ?hlast
      (sch.levels.size - 1) 0 _ _ _ out st₁₈ t₁₈ stM₀ (by omega) (by omega)
      ⟨by rw [er₁₇]; exact hwf, by show sch.n - sch.k0 + 1 = _; omega, rfl, hlev1, by rw [hcaps1]; rfl, hreqs1,
        er₁₇, by rw [ek₁₇]; omega, [], (List.append_nil _).symm, rfl, laterAt_nil _ _ _⟩
      hloop
    case hstep =>
      clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1
      intro i v stE tE bv stE' tE' stM hi hrel hb
      obtain ⟨hwfE, hsplit, hprev, hlevM, hcapsM, hreqsM, hrE, hkE, hheadM⟩ := hrel
      have hc1 := hcols (i + 1) (by omega)
      have hl1 := hl4 (i + 1) (by omega) (by omega)
      have hl2 := hl4 (i + 1 + 1) (by omega) (by omega)
      try dsimp only at hb
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨hr₁, hk₁, hn₁⟩ := fold_coins p 4 0 #[] chs stE st₁ tE t₁ hf
      rw [ite_eq_left hi] at hb
      obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
      have k₂ := absorbBytes_ok h₂
      obtain ⟨hnA, hnB, yA, mA, t₃, -, -, hr₃, hk₃, hb⟩ := oodIntro_bind (by omega) hb
      obtain ⟨hnC, hnD, yB, mB, t₄, -, -, hr₄, hk₄, hb⟩ := oodIntro_bind (by omega) hb
      have hlvi := hlv (i + 1) (by omega)
      obtain ⟨hnQ, hnQ', hmcQ, t₅, hr₅, hk₅, hb⟩ := queryLevel_bind hlvi.q_ne_zero hb
      obtain ⟨mq, t₆, -, k₆, hb⟩ := nextMsg_bind hb
      obtain ⟨t₇, k₇, hb⟩ := claimPow_bind hb
      obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
      obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
      have hb := lmodify_bind hb
      rw [ite_eq_right (show ¬ v.1 - 4 + 1 < 4 by omega)] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
      have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
      have er₃ : t₃.rounds = tE.rounds := hr₃.trans er₂
      have ek₃ : t₃.k = tE.k + 6 := by rw [hk₃, ek₂]
      have er₄ : t₄.rounds = tE.rounds := hr₄.trans er₃
      have ek₄ : t₄.k = tE.k + 8 := by rw [hk₄, ek₃]
      have er₇ : t₇.rounds = tE.rounds := by rw [k₇.2, k₆.2, hr₅, er₄]
      have ek₇ : t₇.k = tE.k + 10 := by rw [k₇.1, k₆.1, hk₅, ek₄]
      have er₈ : tE'.rounds = tE.rounds := hr₈.trans er₇
      have ek₈ : tE'.k = tE.k + 11 := by rw [hk₈, ek₇]
      have hdim : (levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes = v.1 - 4 + 1 := by
        show (sch.levels.getD (i + 1 + 1) default).cols + (sch.levels.getD (i + 1 + 1) default).lanes = _
        omega
      obtain ⟨⟨rs, sM', qM'⟩, hfoldM, hrsl⟩ := foldRoundsC_len (D := Dig) p 4 (sch.k0 + 4 + 7 * i) stM.s stM.q
        (consumed tE.rounds tE.k (tE.k + 4)) (by rw [consumed_length]; omega) (consumed_sizes hwfE hn₁)
      obtain ⟨⟨sT, qT, ctxs⟩, htwo⟩ := twoOodC_any (D := Dig)
        ((levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes) (stM.levels ++ [rs]).length sM' qM'
        (tE.rounds[tE.k + 4]!).coins (tE.rounds[tE.k + 5]!).coins (tE.rounds[tE.k + 6]!).coins
        (tE.rounds[tE.k + 7]!).coins
        (by rw [RoundsWF.get hwfE, hdim, show tE.k + 4 = t₂.k by omega, ← er₂]; exact hnA)
        (by rw [RoundsWF.get hwfE, show tE.k + 5 = t₂.k + 1 by omega, ← er₂]; exact hnB)
        (by rw [RoundsWF.get hwfE, hdim, show tE.k + 6 = t₃.k by omega, ← er₃]; exact hnC)
        (by rw [RoundsWF.get hwfE, show tE.k + 7 = t₃.k + 1 by omega, ← er₃]; exact hnD) (by omega)
        p.oodValues[3 + 2 * i]! p.oodValues[3 + 2 * i + 1]! p.messages[sch.k0 + 4 + 7 * i + 4]!
        p.messages[sch.k0 + 4 + 7 * i + 5]!
      have hcw : (tE.rounds[tE.k + 8]!).coins.size = (levelOf (sch.levels.getD (i + 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 8 = t₄.k by omega, ← er₄, hnQ]; rfl
      have hcα : (tE.rounds[tE.k + 9]!).coins.size =
          Accounting.ceilLog2 (levelOf (sch.levels.getD (i + 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 9 = t₄.k + 1 by omega, ← er₄, hnQ']; exact hlvi.2.2.2.1
      have hcβ : (tE.rounds[tE.k + 10]!).coins.size = 1 := by
        rw [RoundsWF.get hwfE, show tE.k + 10 = t₇.k by omega, ← er₇, hn₈]
      have hmcE : merkleCheck ms v.2.1 (sch.levels.getD (i + 1) default).cap (sch.levels.getD (i + 1) default).d
          (Flock.positions (sch.levels.getD (i + 1) default) (tE.rounds[tE.k + 8]!).coins)
          (p.recursiveProofs[i]!).rows (p.recursiveProofs[i]!).siblings (p.recursiveProofs[i]!).salts
          (2 ^ (sch.levels.getD (i + 1) default).lanes) = .ok () := by
        rw [show tE.k + 8 = t₄.k by omega, ← er₄]; exact hmcQ
      refine ⟨_, { qbOut r (levelOf (sch.levels.getD (i + 1) default)) rs (stM.levels ++ [rs]).length
            { stM with s := sT, q := qT, levels := stM.levels ++ [rs], oods := stM.oods ++ ctxs }
            (tE.rounds[tE.k + 8]!).coins (tE.rounds[tE.k + 9]!).coins (tE.rounds[tE.k + 10]!).coins true
            p.messages[sch.k0 + 4 + 7 * i + 6]! with
          level := stM.level + 1, caps := stM.caps ++ [capDig p.recursiveCaps[i + 1]!],
          prevLevel := levelOf (sch.levels.getD (i + 1 + 1) default) },
        kMsgs p (sch.k0 + 4 + 7 * i) 4 ++ ⟨Cap Dig, capDig p.recursiveCaps[i + 1]!⟩ ::
          [fMsg p.oodValues[3 + 2 * i]!, kMsg p.messages[sch.k0 + 4 + 7 * i + 4]!,
            fMsg p.oodValues[3 + 2 * i + 1]!, kMsg p.messages[sch.k0 + 4 + 7 * i + 5]!] ++
          [kMsg p.messages[sch.k0 + 4 + 7 * i + 6]!],
        rfl, ⟨by rw [er₈]; exact hwfE, by show v.1 - 4 + 1 = _; omega, rfl,
          by show stM.level + 1 = i + 1 + 1; rw [hlevM],
          by
            show stM.caps ++ [capDig p.recursiveCaps[i + 1]!] = _
            rw [hcapsM, List.range_succ (n := i + 1), List.map_append]
            rfl,
          ?reqs, er₈.trans hrE, by rw [ek₈, hkE]; omega, ?head⟩, er₈, by omega, ?tape, ?plays⟩
      case head =>
        obtain ⟨rest, hrest, hlenE, hlatE⟩ := hheadM
        exact qbOut_reqs_laterA r sch i hlvi rs (stM.levels ++ [rs]).length
          { stM with s := sT, q := qT, levels := stM.levels ++ [rs], oods := stM.oods ++ ctxs }
          (tE.rounds[tE.k + 8]!).coins (tE.rounds[tE.k + 9]!).coins (tE.rounds[tE.k + 10]!).coins true
          p.messages[sch.k0 + 4 + 7 * i + 6]! hrest hlenE hlevM hlatE (by
            rw [hrE, hkE, FlockDeploy.KS.levRound, ite_eq_left hi,
              show t.k + 2 + sch.k0 + 9 + 11 * i + 8 = t.k + 2 + (sch.k0 + 6 + 11 * (i + 1)) by omega])
      case reqs =>
        intro req hreq
        simp only [qbOut, List.mem_append, List.mem_singleton] at hreq
        rcases hreq with hreq | rfl
        · exact hreqsM req hreq
        · exact reqOKZ_succ (ℓ := i + 1) hlevM (by omega) (by omega) hlvi rs (by rw [hrsl, hl1])
            (tE.rounds[tE.k + 8]!).coins (by unfold capAt; rw [ite_eq_right (by omega), hprev]; rfl)
            (by unfold openAt; rw [ite_eq_right (by omega), ite_eq_left (by omega)]; rfl) hmcE
      case tape =>
        obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 - i = j + 2 := ⟨sch.levels.size - 1 - i - 2, by omega⟩
        rw [hj, show sch.levels.size - 1 - (i + 1) = j + 1 by omega,
          show sch.k0 + 4 + 7 * (i + 1) = sch.k0 + 4 + 7 * i + 7 by omega,
          show 3 + 2 * (i + 1) = 3 + 2 * i + 2 by omega]
        simp only [recMsgs]
      case plays =>
        intro msR csR pend hrec
        try dsimp only at hrec ⊢
        rw [drop_levels sch (i + 1) (by omega), drop_levels sch (i + 1 + 1) (by omega)]
        rw [drop_levels sch (i + 1 + 1) (by omega)] at hrec
        have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (i + 1) default)).lanes
            stM.s stM.q) (kMsgs p (sch.k0 + 4 + 7 * i) 4) (consumed tE.rounds tE.k (tE.k + 4)) (rs, sM', qM') := by
          rw [show (levelOf (sch.levels.getD (i + 1) default)).lanes = 4 from hl1]; exact hfoldM
        have hmain := iterNL_model r (sch.m - 7) b ini sch.ext
          (levelOf (sch.levels.getD (i + 1) default)) (levelOf (sch.levels.getD (i + 1 + 1) default))
          ((sch.levels.toList.map levelOf).drop (i + 1 + 1 + 1)) stM hfold' (capDig p.recursiveCaps[i + 1]!) htwo
          (tE.rounds[tE.k + 8]!).coins (tE.rounds[tE.k + 9]!).coins (tE.rounds[tE.k + 10]!).coins hcw hcα hcβ
          hlvi.q_ne_one (hα1 (i + 1) (by omega)) p.messages[sch.k0 + 4 + 7 * i + 6]! hrec
        have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k (tE.k + 4) ++
            [(tE.rounds[tE.k + 4]!).coins, (tE.rounds[tE.k + 5]!).coins, (tE.rounds[tE.k + 6]!).coins,
              (tE.rounds[tE.k + 7]!).coins] ++
            [(tE.rounds[tE.k + 8]!).coins, (tE.rounds[tE.k + 9]!).coins, (tE.rounds[tE.k + 10]!).coins] := by
          rw [ek₈, consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 11 by omega),
            show tE.k + 11 = tE.k + 4 + 7 by omega, consumed_add tE.rounds (tE.k + 4) 7]
          simp [List.range_succ, Nat.add_assoc]
        rw [hcs]
        exact hmain
    case hlast =>
      clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1
      intro i v stE tE bv stE' tE' stM hi hrel hb
      obtain ⟨hwfE, hsplit, hprev, hlevM, hcapsM, hreqsM, hheadM⟩ := hrel
      have hl1 := hl4 (i + 1) (by omega) (by omega)
      try dsimp only at hb
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨hr₁, hk₁, hn₁⟩ := fold_coins p 4 0 #[] chs stE st₁ tE t₁ hf
      rw [ite_eq_right (show ¬ i + 1 < sch.levels.size - 1 by omega)] at hb
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
      obtain ⟨_, st₂, t₂, h₂, hb₂⟩ := lbind_ok hb
      clear hb
      rw [← Array.forIn_toList] at h₂
      obtain ⟨rfl, k₂⟩ := forIn_absorbF_keeps _ h₂
      have hb := hb₂
      clear hb₂
      have hlvR := hlv (sch.levels.size - 1) (by omega)
      obtain ⟨hnQ, hnQ', hmcQ, t₃, hr₃, hk₃, hb⟩ := queryLevel_bind hlvR.q_ne_zero hb
      obtain ⟨t₄, k₄, hb⟩ := claimPow_bind hb
      obtain ⟨bq, t₅, h₅, hb⟩ := lift_bind hb
      obtain ⟨-, hn₅, rfl, hr₅, hk₅⟩ := sample_ok h₅
      have hb := lmodify_bind hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
      have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
      have er₄ : t₄.rounds = tE.rounds := by rw [k₄.2, hr₃, er₂]
      have ek₄ : t₄.k = tE.k + 6 := by rw [k₄.1, hk₃, ek₂]
      have er₅ : tE'.rounds = tE.rounds := hr₅.trans er₄
      have ek₅ : tE'.k = tE.k + 7 := by rw [hk₅, ek₄]
      rw [hi] at hl1
      obtain ⟨⟨rs, sM', qM'⟩, hfoldM, hrsl⟩ := foldRoundsC_len (D := Dig) p 4 (sch.k0 + 4 + 7 * i) stM.s stM.q
        (consumed tE.rounds tE.k (tE.k + 4)) (by rw [consumed_length]; omega) (consumed_sizes hwfE hn₁)
      have hfold' : Plays fOf (Model.foldRoundsC execArith
          (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes stM.s stM.q)
          (kMsgs p (sch.k0 + 4 + 7 * i) 4) (consumed tE.rounds tE.k (tE.k + 4)) (rs, sM', qM') := by
        rw [show (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes = 4 from hl1]; exact hfoldM
      have hcw : (tE.rounds[tE.k + 4]!).coins.size =
          (levelOf (sch.levels.getD (sch.levels.size - 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 4 = t₂.k by omega, ← er₂, hnQ]; rfl
      have hcα : (tE.rounds[tE.k + 5]!).coins.size =
          Accounting.ceilLog2 (levelOf (sch.levels.getD (sch.levels.size - 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 5 = t₂.k + 1 by omega, ← er₂, hnQ']; exact hlvR.2.2.2.1
      have hcβ : (tE.rounds[tE.k + 6]!).coins.size = 1 := by
        rw [RoundsWF.get hwfE, show tE.k + 6 = t₄.k by omega, ← er₄, hn₅]
      have hmcE : merkleCheck ms v.2.1 (sch.levels.getD (sch.levels.size - 1) default).cap
          (sch.levels.getD (sch.levels.size - 1) default).d
          (Flock.positions (sch.levels.getD (sch.levels.size - 1) default) (tE.rounds[tE.k + 4]!).coins)
          p.finalProof.rows p.finalProof.siblings p.finalProof.salts
          (2 ^ (sch.levels.getD (sch.levels.size - 1) default).lanes) = .ok () := by
        rw [show tE.k + 4 = t₂.k by omega, ← er₂]; exact hmcQ
      have hmain := iterLast_model r (sch.m - 7) b ini sch.ext
        (levelOf (sch.levels.getD (sch.levels.size - 1) default)) stM hfold' (yrOf p sch.ext)
        (tE.rounds[tE.k + 4]!).coins (tE.rounds[tE.k + 5]!).coins (tE.rounds[tE.k + 6]!).coins hcw hcα hcβ
        hlvR.q_ne_one (hα1 (sch.levels.size - 1) (by omega))
      have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k (tE.k + 4) ++
          [(tE.rounds[tE.k + 4]!).coins, (tE.rounds[tE.k + 5]!).coins, (tE.rounds[tE.k + 6]!).coins] := by
        rw [ek₅, consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 7 by omega),
          show tE.k + 7 = tE.k + 4 + 3 by omega, consumed_add tE.rounds (tE.k + 4) 3]
        simp [List.range_succ, Nat.add_assoc]
      have hdrop : (sch.levels.toList.map levelOf).drop (i + 1) =
          [levelOf (sch.levels.getD (sch.levels.size - 1) default)] := by
        rw [hi, drop_levels sch (sch.levels.size - 1) (by omega),
          show sch.levels.size - 1 + 1 = sch.levels.size by omega]
        simp
      exact ⟨_, _, rfl, er₅, by omega, by
        rw [hdrop, show sch.levels.size - 1 - i = 1 by omega, hcs]
        simp only [recMsgs]
        exact hmain, by
        refine ⟨?_, fun req hreq => ?_, ?_⟩
        · show stM.caps = _
          rw [hcapsM, hi]
        · simp only [qbOut, List.mem_append, List.mem_singleton] at hreq
          rcases hreq with hreq | rfl
          · exact hreqsM req hreq
          · exact reqOKZ_succ (ℓ := sch.levels.size - 1) (hlevM.trans hi) (by omega) (by omega) hlvR rs
              (by rw [hrsl, hl1]) (tE.rounds[tE.k + 4]!).coins
              (by unfold capAt; rw [ite_eq_right (by omega), hprev, show sch.levels.size - 1 - 1 = i by omega])
              (by unfold openAt; rw [ite_eq_right (by omega), ite_eq_right (lt_irrefl _)]) hmcE
        · obtain ⟨rest, hrest, hlenE, hlatE⟩ := hheadM
          obtain ⟨rest', h1, h2', h3⟩ := qbOut_reqs_laterA r sch i (by rw [hi]; exact hlvR) rs
            (stM.levels ++ [rs]).length { stM with s := sM', q := qM', levels := stM.levels ++ [rs] }
            (tE.rounds[tE.k + 4]!).coins (tE.rounds[tE.k + 5]!).coins (tE.rounds[tE.k + 6]!).coins false (0, 0)
            hrest hlenE hlevM hlatE (by
              rw [hrE, hkE, FlockDeploy.KS.levRound, ite_eq_right (by omega),
                show t.k + 2 + sch.k0 + 9 + 11 * i + 4 = t.k + 2 + (sch.k0 + 2 + 11 * (i + 1)) by omega])
          rw [hi] at h1 h2'
          exact ⟨rest', h1, h2', h3⟩⟩
    obtain ⟨pend, hr, hk, hpl, hfinal⟩ := hLC
    refine ⟨pend, hr, hk, ?_, hfinal⟩
    simp only [Nat.zero_add, Nat.sub_zero, Nat.mul_zero, Nat.add_zero] at hpl
    rw [hd1] at hpl
    exact hpl
  obtain ⟨_, hrL, hkL, -⟩ := hL 0 (fun _ => 0) []
    ⟨fun _ => 0, fun _ => (0, 0, 0), 0, [], [], [], 1, [capDig p.recursiveCaps[0]!], [], ⟨0, 0, 0, 0⟩⟩ rfl rfl
    (by simp)
  intro np P c₀ r Tm b
  have hcs : consumed t.rounds t.k t₁₈.k = (t.rounds[t.k]!).coins :: (t.rounds[t.k + 1]!).coins ::
      (consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0) ++
        (consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2) ++
          ([(t.rounds[t.k + 2 + sch.k0 + 2]!).coins, (t.rounds[t.k + 2 + sch.k0 + 3]!).coins,
              (t.rounds[t.k + 2 + sch.k0 + 4]!).coins, (t.rounds[t.k + 2 + sch.k0 + 5]!).coins] ++
            ([(t.rounds[t.k + 2 + sch.k0 + 6]!).coins, (t.rounds[t.k + 2 + sch.k0 + 7]!).coins,
                (t.rounds[t.k + 2 + sch.k0 + 8]!).coins] ++
              consumed t₁₇.rounds t₁₇.k t₁₈.k)))) := by
    rw [er₁₇, ek₁₇, consumed_cons _ (show t.k < t₁₈.k by omega), consumed_cons _ (show t.k + 1 < t₁₈.k by omega),
      show t.k + 1 + 1 = t.k + 2 from rfl,
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 2 + sch.k0 by omega)
        (show t.k + 2 + sch.k0 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 ≤ t.k + 2 + sch.k0 + 2 by omega)
        (show t.k + 2 + sch.k0 + 2 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 + 2 ≤ t.k + 2 + sch.k0 + 2 + 4 by omega)
        (show t.k + 2 + sch.k0 + 2 + 4 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 + 2 + 4 ≤ t.k + 2 + sch.k0 + 2 + 4 + 3 by omega)
        (show t.k + 2 + sch.k0 + 2 + 4 + 3 ≤ t₁₈.k by omega),
      consumed_add t.rounds (t.k + 2 + sch.k0 + 2) 4, consumed_add t.rounds (t.k + 2 + sch.k0 + 2 + 4) 3,
      show t.k + 2 + sch.k0 + 2 + 4 + 3 = t.k + 2 + sch.k0 + 9 by omega]
    simp [List.range_succ, Nat.add_assoc]
  have hlv01 : (schedOf sch).levels = levelOf (sch.levels.getD 0 default) ::
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := by
    have e0 := drop_levels sch 0 (by omega)
    simp only [List.drop_zero, Nat.zero_add] at e0
    rw [hd1] at e0
    exact e0
  have hz₀ : (t.rounds[t.k]!).coins.size = (schedOf sch).m - 7 := by
    rw [RoundsWF.get hwf, show t.k = t₃'.k from ek₃.symm, ← er₃, hn₅, hn7]; rfl
  have hβ₀ : (t.rounds[t.k + 1]!).coins.size = 1 := by
    rw [RoundsWF.get hwf, show t.k + 1 = t₇.k from ek₇.symm, ← er₇, hn₈]
  have hFl : (consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0)).length = sch.k0 := by
    rw [consumed_length]; omega
  have hFs : ∀ c ∈ consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0), c.size = 2 :=
    consumed_sizes hwf fun j hj => by
      rw [show t.k + 2 + j = t₉.k + j by omega, ← er₉]; exact hn₁₀ j (by omega)
  have hρl : (consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2)).length = 2 := by
    rw [consumed_length]; omega
  have hρs : ∀ c ∈ consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2), c.size = 2 :=
    consumed_sizes hwf fun j hj => by
      rw [show t.k + 2 + sch.k0 + j = t₉.k + (sch.k0 + j) by omega, ← er₉]; exact hn₁₀ (sch.k0 + j) (by omega)
  have hdim0 : sch.n - sch.k0 + 1 = (schedOf sch).m - 7 - (schedOf sch).k0 + 1 := by
    show sch.n - sch.k0 + 1 = sch.m - 7 - sch.k0 + 1
    rw [hn7]
  have hmc0' : merkleCheck ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (Flock.positions (sch.levels.getD 0 default) (t.rounds[t.k + 2 + sch.k0 + 6]!).coins) p.initialProof.rows
      p.initialProof.siblings p.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) =
        .ok () := by
    rw [show t.k + 2 + sch.k0 + 6 = t₁₃.k by omega, ← er₁₃]; exact hmc0
  have hmain := ligeritoPadZC_reqs (D := Dig) P (schedOf sch) zl.tPad c₀ r Tm b hlv01
    (t.rounds[t.k]!).coins (t.rounds[t.k + 1]!).coins hz₀ (by show sch.m - 7 ≠ 1; omega) hβ₀
    (fOf p.oodValues[0]!) (gOf p.messages[0]!.1, gOf p.messages[0]!.2)
    (fun s q => foldRoundsZK_any P p sch.k0 0 1 s q _ hFl hFs)
    ![gOf zl.msgs.e0, gOf zl.msgs.e1] (drawKs_any 2 _ hρl hρs) (gOf zl.msgs.tPrime)
    (fun i => gOf zl.msgs.ybar[i.val]!) (gOf p.messages[sch.k0]!.1, gOf p.messages[sch.k0]!.2)
    (capDig p.recursiveCaps[0]!)
    ⟨0, Model.positions execArith (levelOf (sch.levels.getD 0 default)).queries
      (levelOf (sch.levels.getD 0 default)).logLen
      (vecOf (levelOf (sch.levels.getD 0 default)).queries (t.rounds[t.k + 2 + sch.k0 + 6]!).coins),
      (levelOf (sch.levels.getD 0 default)).lanes, pathLen (levelOf (sch.levels.getD 0 default))⟩
    (Q := fun pend => pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOKZ ms sch cap p req) ∧
      ∃ rest, pend.reqs = ⟨0, Model.positions execArith (levelOf (sch.levels.getD 0 default)).queries
        (levelOf (sch.levels.getD 0 default)).logLen
        (vecOf (levelOf (sch.levels.getD 0 default)).queries (t.rounds[t.k + 2 + sch.k0 + 6]!).coins),
        (levelOf (sch.levels.getD 0 default)).lanes, pathLen (levelOf (sch.levels.getD 0 default))⟩ :: rest ∧
        rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds (t.k + 2) rest)
    (fun s q => twoOodC_any _ 0 s q
      (t.rounds[t.k + 2 + sch.k0 + 2]!).coins (t.rounds[t.k + 2 + sch.k0 + 3]!).coins
      (t.rounds[t.k + 2 + sch.k0 + 4]!).coins (t.rounds[t.k + 2 + sch.k0 + 5]!).coins
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 2 = t₁₁.k by omega, ← er₁₁, hnA, hdim0])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 3 = t₁₁.k + 1 by omega, ← er₁₁, hnB])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 4 = t₁₂.k by omega, ← er₁₂, hnC, hdim0])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 5 = t₁₂.k + 1 by omega, ← er₁₂, hnD])
      (by show sch.m - 7 - sch.k0 + 1 ≠ 1; omega)
      p.oodValues[1]! p.oodValues[2]! p.messages[sch.k0 + 1]! p.messages[sch.k0 + 2]!)
    (fun col st hlev hreqs => by
      obtain ⟨st', hpl, hst'⟩ := queryAndBatchPadC_req r (levelOf (sch.levels.getD 0 default)) col 0 st
        (t.rounds[t.k + 2 + sch.k0 + 6]!).coins (t.rounds[t.k + 2 + sch.k0 + 7]!).coins
        (t.rounds[t.k + 2 + sch.k0 + 8]!).coins
        (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 6 = t₁₃.k by omega, ← er₁₃, hnQ]; rfl)
        (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 7 = t₁₃.k + 1 by omega, ← er₁₃, hnQ'];
            exact hlv0.2.2.2.1)
        (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 8 = t₁₆.k by omega, ← er₁₆, hn₁₇])
        hlv0.q_ne_one (hα1 0 (by omega)) p.messages[sch.k0 + 3]!
      exact ⟨st', hpl, by rw [hst', hreqs, hlev]; rfl⟩)
    (fun ini st hreqs hlev hcaps => by
      obtain ⟨pend, _, _, hpl, hc, hq, rest, hrest, hlenR, hlatR⟩ := hL r b ini st hlev hcaps (by
        rw [hreqs]
        intro req hreq
        rw [List.mem_singleton] at hreq
        subst hreq
        exact reqOKZ_zero rfl hlv0 _ _ hmc0')
      exact ⟨pend, hpl, hc, hq, rest, by rw [hrest, hreqs]; rfl, hlenR, hlatR⟩)
  rw [hcs]
  simp only [ligMsgsZ, List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
  obtain ⟨out, hpl, hcaps, hreqs, rest, hrest, hlenR, hlatR⟩ := hmain
  exact ⟨out, hpl, hcaps, hreqs, _, rest, hrest, rfl, Refine.positions_eq _ hlv0 _, hlenR, hlatR⟩

/-- **`repProg`'s run records every level's request** (`zk_rep_reqs`): its requests passed, the first is level 0's,
and the ones after it are levels `1, …, R` in order, each at the positions of the coins of its query round
(`LaterAt`), Ligerito starting after the masked replay's `m + (kLog − 6)` rounds and the ring switch's `claims + 1`. -/
theorem zk_rep_reqsL {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (extras : Array Flock.Claim) (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam T st sch tPad)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) :
    ∃ pend : ZkLink.PendZ 𝔽 Dig Sl,
      Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend ∧
      pend.lig.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig zp.proof.ligerito.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.lig.reqs, ReqOKZ st.merkle sch zp.proof.cap zp.proof.ligerito req) ∧
      ∃ (R : Req) (rest : List Req), pend.lig.reqs = R :: rest ∧ R.level = 0 ∧
        R.positions = (Flock.positions (sch.levels.getD 0 default)
          (rounds[(st.m + (st.kLog - 6)) + (2 + extras.size + 1) + (sch.k0 + 8)]!).coins).toList ∧
        rest.length = sch.levels.size - 1 ∧
        LaterAt sch rounds ((st.m + (st.kLog - 6)) + (2 + extras.size + 1) + 2) rest := by
  obtain ⟨h25, -, -, hk0, -, hdim, hlv, hα1, -⟩ := zk_schedule_seams _ _ _ hsch
  have hTm : T.S.m = st.m := hseam.hm
  have h7 : 7 ≤ T.S.m := by omega
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨Tb, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok lg =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg', t₄, h₄, h₅⟩ := bind_ok hrun
  clear hrun
  obtain ⟨hlg, ht₄⟩ := pure_ok h₄
  rw [hlg, ht₄] at h₅
  have hwf₀ : RoundsWF (Flock.Transcript.init rounds dom).rounds := hwf
  have hcs3 := replay_cons_size _ _ _ _ h₁
  obtain ⟨hr₁, hk₁, hRP⟩ := zk_replay_plays st _ zp.proof zp.inner.root _ t₁ rp T.S hTm.symm hseam.hk.symm
    (by omega) hwf₀ h₁
  obtain ⟨k₁, zc, row, abM, row', hk₁a, hk₁b, hpz, hpl, hc0, hc1⟩ := hRP T.padsA T.tr
  rw [← hTm] at h₂
  obtain ⟨g₀, g₁, g₂⟩ := two_append_get!
    ({ rp.claims[0]! with value := Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]! } : Flock.Claim)
    ({ rp.claims[1]! with value := Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]! } : Flock.Claim) extras
  obtain ⟨hr₂, hk₂, op, hop, -⟩ := opening_refines T.S.m h7 _
    (⟨fOf (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!), abM.skip, abM.x⟩ ::
      ⟨fOf (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!), execArith.lagS zc.z, zc.r⟩ ::
        Model.extraClaims T.S T.ptLocal mPts pts)
    (by rw [size_two_append]; simp; omega)
    (fun k hk => by
      rcases k with _ | _ | k
      · rw [g₀]; exact hc0 _
      · rw [g₁]; exact hc1 _
      · have hk' : k < (Model.extraClaims T.S T.ptLocal mPts pts).length := by simp at hk; omega
        rw [g₂ k (by omega)]
        exact hext k hk')
    (by rw [size_two_append]; omega) (by rw [size_two_append]; omega) _ t₁ t₂ Tb sws (hr₁ ▸ hwf₀) h₂
  rw [size_two_append] at hop
  obtain ⟨xo, hxo, hxo1⟩ := openingZK_plays h7 _ _ abM.pub ⟨0, execArith.lagS zc.z, zc.r⟩ _ hop
  obtain ⟨xo1, xo2⟩ := xo
  obtain rfl : xo1 = some op := hxo1
  have hwf₂ : RoundsWF t₂.rounds := hr₂ ▸ hr₁ ▸ hwf₀
  obtain ⟨hr₃, hk₃, -⟩ := zk_ligerito_plays st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩
    t₂ t₃ rc hdim hlv hα1 hk0 hwf₂ h₃
  have hlig : ∀ {e : ℕ} (P : ZkSession.Pads0 T.np e), e = 2 → ∀ (Tm : 𝔽) (b : ℕ → 𝔽),
      ∃ out : ZkSession.LinK 𝕂 T.np × PendL 𝔽 Dig,
        Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P T.sch T.t T.c₀ r.val Tm b)
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0) (consumed t₂.rounds t₂.k t₃.k) out ∧
        (out.2.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig zp.proof.ligerito.recursiveCaps[j]!) ∧
          (∀ req ∈ out.2.reqs, ReqOKZ st.merkle sch zp.proof.cap zp.proof.ligerito req) ∧
          ∃ (R : Req) (rest : List Req), out.2.reqs = R :: rest ∧ R.level = 0 ∧
            R.positions = (Flock.positions (sch.levels.getD 0 default)
              (t₂.rounds[t₂.k + 2 + sch.k0 + 6]!).coins).toList ∧
            rest.length = sch.levels.size - 1 ∧ LaterAt sch t₂.rounds (t₂.k + 2) rest) := by
    intro e P he Tm b
    subst he
    rw [hseam.hsch, hseam.ht]
    exact zk_ligerito_reqsL st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩
      t₂ t₃ rc hdim hlv hα1 hk0 hwf₂ h₃ P T.c₀ r.val Tm b
  have hwf₃ : RoundsWF t₃.rounds := hr₃ ▸ hwf₂
  obtain ⟨hr₅, hk₅, hIP⟩ := zk_inner_plays _ zp.inner zp.proof.zerocheck.finalC _ t₃ tr
    (by rw [Array.size_append, Array.size_append, hcs3, ligeritoConstraints_size hv]; rfl) hwf₃ h₅
  have hM : ∃ o, Plays fOf (ZkLink.repMaskedZC (D := Dig) execArith T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r.val
        (Model.extraClaims T.S T.ptLocal mPts pts))
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC]))))
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) o ∧
      (o.2.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig zp.proof.ligerito.recursiveCaps[j]!) ∧
        (∀ req ∈ o.2.reqs, ReqOKZ st.merkle sch zp.proof.cap zp.proof.ligerito req) ∧
        ∃ (R : Req) (rest : List Req), o.2.reqs = R :: rest ∧ R.level = 0 ∧
          R.positions = (Flock.positions (sch.levels.getD 0 default)
            (t₂.rounds[t₂.k + 2 + sch.k0 + 6]!).coins).toList ∧
          rest.length = sch.levels.size - 1 ∧ LaterAt sch t₂.rounds (t₂.k + 2) rest) := by
    unfold ZkLink.repMaskedZC
    refine pq_map ?_
    refine pq_bind hpz ?_ rfl rfl
    refine pq_map ?_
    refine pq_bind hpl ?_ rfl rfl
    refine pq_map ?_
    unfold ZkLink.restZC
    refine pq_bind hxo ?_ rfl rfl
    obtain ⟨out, hout, hne⟩ := hlig T.pads hseam.he op.T op.b
    refine pq_bind hout ?_ rfl (List.append_nil _).symm
    exact ⟨_, Plays.bind' (Plays.recv _ (fOf zp.proof.zerocheck.finalC)) (Plays.pure _) rfl rfl, hne⟩
  have hmsgs : restMsgsZ st sch (2 + extras.size) tPad zp =
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC])))) ++
      [⟨𝔽 × 𝔽, (fOf zp.inner.rho, fOf zp.inner.sigma)⟩, ⟨Dig, toDig zp.inner.tauCommit⟩,
        ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) zp.inner.y⟩] := by
    rw [hTm, hseam.hk, hseam.hnp, hseam.hnP]
    simp [restMsgsZ, innerMsgs]
  have hcoins : rounds.toList.map (·.coins) =
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) ++ consumed t₃.rounds t₃.k tr.k := by
    rw [hr₃, hr₂, hr₁]
    simp only [List.append_assoc]
    rw [← consumed_append _ hk₃ hk₅, ← consumed_append _ hk₂ (hk₃.trans hk₅),
      ← consumed_append _ hk₁b (hk₂.trans (hk₃.trans hk₅)),
      ← consumed_append _ hk₁a (hk₁b.trans (hk₂.trans (hk₃.trans hk₅))), hk]
    exact (consumed_all _).symm
  unfold ZkLink.repZCAt ZkLink.repZC
  rw [hseam.hq]
  refine exQ_recv (toDig zp.inner.root) ?_
  have hmv := ringSwitch_moves _ _ _ (by rw [size_two_append]; omega) h₂
  rw [size_two_append] at hmv
  have hi : t₂.k + 2 + sch.k0 + 6 = (st.m + (st.kLog - 6)) + (2 + extras.size + 1) + (sch.k0 + 8) := by
    have h0 : (Flock.Transcript.init rounds dom).k = 0 := rfl
    rw [hmv.2, hk₁, h0, hTm, hseam.hk]
    omega
  have hi2 : t₂.k + 2 = (st.m + (st.kLog - 6)) + (2 + extras.size + 1) + 2 := by omega
  have hrt : t₂.rounds = rounds := by rw [hr₂, hr₁]; rfl
  refine pq_bindQ hM (fun o ho => ?_) hmsgs hcoins
  obtain ⟨v, hv⟩ := hIP Hs T.dom T.np T.nP T.pos o.1 T.tr (toDig zp.inner.root)
  obtain ⟨hc, hq, R, rest, hR, hl, hp, hlenR, hlatR⟩ := ho
  refine ⟨⟨o.2, v⟩, Plays.bind' hv (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm, hc, hq, R, rest,
    hR, hl, ?_, hlenR, ?_⟩
  · rw [hp, hi, hrt]
  · rw [← hrt, ← hi2]
    exact hlatR

end FlockSoundness.Discharge.ZkExec
