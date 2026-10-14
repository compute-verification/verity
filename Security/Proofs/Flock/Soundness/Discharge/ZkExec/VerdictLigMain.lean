import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLigRun

/-!
# The hiding Ligerito accepts

`zk_ligerito_accepts`: when `Zk.ligerito` accepts, the compiled masked phase's Ligerito, played with the proof's
messages on the coins the executable consumed, ends in a result whose verdict at the proof's openings is true. The
lane phase and level 0's padded queries are new (`lane_vals`, `queryLevel0_val`, `padE_eq`); the level loop is the
non-ZK session's (`LigeritoRun.lean`), with an invariant that drops the non-ZK session's caps and requests (`LoopRelV`):
level 0's request is on `2^k₀ + 4` wide rows, which the non-ZK session's `ReqOK` does not describe, and the verdict
reads neither.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- The level loop's invariant before iteration `i`: the non-ZK session's `LoopRel` without its caps and requests. -/
structure LoopRelV (sch : Flock.Schedule) (p : LigeritoProof) (o : Opens 𝔽 Dig) (i : ℕ)
    (v : ℕ × Array ByteArray × Array (Array F256) × ℕ) (stE : LState) (t : Flock.Transcript)
    (stM : LigStateC 𝔽 𝕂 Dig) : Prop where
  wf : RoundsWF t.rounds
  s : stM.s o = gOf stE.s
  q : stM.q o = quadOf stE.q
  split : v.1 = (sch.levels.getD (i + 1) default).cols + 4
  prev : v.2.1 = p.recursiveCaps[i]!
  size : v.2.2.1.size = i
  lanes : ∀ l ∈ v.2.2.1.toList, l.size = 4
  levels : stM.levels = v.2.2.1.toList.map fun l => l.toList.map gOf
  oods : stM.oods = stE.oods.toList.map oodOf
  cons : stM.cons = stE.conss.toList.map consOf
  pos : ∀ cs ∈ stE.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size
  level : stM.level = i + 1

/-- What the level loop ends with: the non-ZK session's `LoopFinal` without its residual length, caps and requests. -/
structure LoopFinalV (sch : Flock.Schedule) (p : LigeritoProof) (o : Opens 𝔽 Dig) (b : ℕ → 𝔽) (ini : List 𝕂)
    (v : ℕ × Array ByteArray × Array (Array F256) × ℕ) (stE : LState) (pend : PendL 𝔽 Dig) : Prop where
  size : v.2.2.1.size = sch.levels.size - 1
  lanes : ∀ l ∈ v.2.2.1.toList, l.size = 4
  ext : v.2.2.2 = sch.ext
  verdict : pend.verdict o = Model.finalCheck execArith (sch.m - 7) b ini
    (v.2.2.1.toList.map fun l => l.toList.map gOf) (stE.oods.toList.map oodOf) (stE.conss.toList.map consOf) sch.ext
    (yrOf p sch.ext) (gOf stE.s)
  pos : ∀ cs ∈ stE.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size

/-- **Level 0's padded consistency value, at the model's positions**: `consPad_eq` with the model's positions and
consistency coins (`positions_eq`, `ofFn_vecOf`), on a level of rate 1 whose rows the openings give. -/
theorem padE_eq (lv : Flock.Level) (hl : LevelSeam lv) (hrate : lv.rate = 1) (k0 : ℕ) (hk0 : lv.lanes = k0)
    (rows : Array (Array F128)) (o : Opens 𝔽 Dig) (r ℓ : ℕ) (cw cα : Array F128) (fold : Array F256) (tPad : ℕ)
    (rho : F256 × F256) (ρ : Fin 2 → 𝕂) (ybar : Array F256) (hcα : cα.size = Nat.log2 (lv.q - 1) + 1)
    (hk : fold.size = k0) (hrows : ∀ j < (Flock.positions lv cw).size, rows[j]!.size = 2 ^ k0 + 4)
    (hR : ∀ j < lv.q, (o r ℓ j).1 = rows[j]!.toList.map fOf) (hrep : r < 2) (hρ0 : ρ 0 = gOf rho.1)
    (hρ1 : ρ 1 = gOf rho.2) (hy : ybar.size = tPad) (ht : tPad ≤ 2 ^ lv.cols) :
    ((Model.positions execArith (levelOf lv).queries (levelOf lv).logLen (vecOf (levelOf lv).queries cw)).zipIdx.map
      fun pj => execArith.embed (eqList (List.ofFn (vecOf (Accounting.ceilLog2 (levelOf lv).queries) cα)) pj.2) *
        ZkLink.padColumnO execArith (levelOf lv).logLen (levelOf lv).logCols tPad (fOf Zk.PAD_C)
          (fold.toList.map gOf) ρ (extraLane k0 2 r) (fun i => gOf ybar[i.val]!) (o r ℓ pj.2).1 pj.1).sum =
      gOf (consE0 rows lv.q (2 ^ lv.lanes) cα fold +
        extraE rows (Flock.positions lv cw) (2 ^ lv.lanes) r lv.cols rho cα ybar) := by
  have hc : cα.size = Accounting.ceilLog2 (levelOf lv).queries := hcα.trans hl.2.2.2.1
  have hsz := positions_size lv hl cw
  rw [show List.ofFn (vecOf (Accounting.ceilLog2 (levelOf lv).queries) cα) = cα.toList.map fOf from ofFn_vecOf _ cα hc,
    show Model.positions execArith (levelOf lv).queries (levelOf lv).logLen (vecOf (levelOf lv).queries cw) =
      (Flock.positions lv cw).toList from Refine.positions_eq lv hl cw, show (levelOf lv).logLen = lv.cols + 1 by
      show lv.cols + lv.rate = _; rw [hrate], show (levelOf lv).logCols = lv.cols from rfl, hk0, ← hsz]
  exact consPad_eq rows (fun j => (o r ℓ j).1) (Flock.positions lv cw) cα fold k0 r lv.cols tPad rho ρ ybar hk hrows
    (fun j hj => hR j (hsz ▸ hj)) (pos_le_alpha lv hl cw cα hcα) hrep hρ0 hρ1 hy ht

set_option maxHeartbeats 8000000 in
/-- **The hiding Ligerito accepts** (live PROTOCOL.md §7 with §13.3–§13.6). When `Zk.ligerito` accepts from `t` to
`t'` on well-formed rounds, at a schedule whose dimensions and levels are the model's, whose levels' consistency coins
are not one word, with at least one lane round, level 0 of rate 1 and the padding no longer than level 0's message,
and with `basisAt` the model's `bHat` of `b` (`hbasis`), then at any pads and batched value, the compiled masked phase's
Ligerito at `c₀ = PAD_C` and the executable's rep, played with the proof's messages on the rounds consumed, ends in a
result whose verdict at the openings `o` (the proof's opened rows, `ho`) is true. -/
theorem zk_ligerito_accepts (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (b : ℕ → 𝔽) (o : Opens 𝔽 Dig)
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1) (hk0 : 1 ≤ sch.k0)
    (hrate : (sch.levels.getD 0 default).rate = 1) (hpad : zl.tPad ≤ 2 ^ (sch.levels.getD 0 default).cols)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o zl.rep ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    (hbasis : ∀ (orig : Array F256) (y : ℕ), orig.size + sch.ext = sch.m - 7 → y < 2 ^ sch.ext →
      gOf (basisAt orig sch.ext)[y]! = Model.bHat execArith (sch.m - 7) b (orig.toList.map gOf ++ bitsK sch.ext y))
    {t t' : Flock.Transcript} {rec : Zk.L0Rec} (hwf : RoundsWF t.rounds)
    (h : Zk.ligerito ms sch T cap p basisAt zl t = .ok (rec, t')) :
    ∀ {np : ℕ} (P : ZkSession.Pads0 np 2) (Tm : 𝔽),
      ∃ out : ZkSession.LinK 𝕂 np × PendL 𝔽 Dig,
        Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P (schedOf sch) zl.tPad (fOf Zk.PAD_C) zl.rep Tm b)
          (ligMsgsZ p capDig sch zl.tPad zl.msgs) (consumed t.rounds t.k t'.k) out ∧ out.2.verdict o = true := by
  intro np P Tm
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
  obtain ⟨hr₁₀, hk₁₀, hn₁₀, hlane1, hlane2, hs₁₀, hq₁₀, hmsg₁₀, hood₁₀, hoods₁₀, hconss₁₀⟩ :=
    lane_vals p zl.msgs.tPrime _ sch.k0 (by
      intro i v st tE bv st' tE' hiB hb
      obtain ⟨r, tR, hR', hb⟩ := lift_bind hb
      obtain ⟨-, hnR, rfl, hrR, hkR⟩ := sampleK_ok hR'
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        obtain ⟨f₂k, f₂r⟩ := absorbK_ok h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        obtain ⟨f₃k, f₃r⟩ := absorbK_ok h₃
        obtain ⟨ρ₀, t₄, h₄, hb⟩ := lift_bind hb
        obtain ⟨-, hn₄, hρ₀, hr₄, hk₄⟩ := sampleK_ok h₄
        obtain ⟨ρ₁, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨-, hn₅, hρ₁, hr₅, hk₅⟩ := sampleK_ok h₅
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
        obtain ⟨m, t₉, hm, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        have e₃r : t₃.rounds = tE.rounds := by rw [f₃r, f₂r, hrR]
        have e₄r : t₄.rounds = tE.rounds := by rw [hr₄, e₃r]
        refine ⟨_, rfl, by rw [f₉r, f₈r, f₇r, f₆r, hr₅, hr₄, f₃r, f₂r, hrR], by rw [ite_eq_left hj]; omega, hnR,
          rfl, rfl, rfl, rfl, rfl, fun _ => ⟨?_, ?_, ?_, rfl, ?_⟩⟩
        · rw [show tE.k + 1 = t₃.k by omega, ← e₃r]
          exact hn₄
        · rw [show tE.k + 2 = t₄.k by omega, ← e₄r]
          exact hn₅
        · show (ρ₀, ρ₁) = _
          rw [hρ₀, hρ₁, e₃r, e₄r, show t₃.k = tE.k + 1 by omega, show t₄.k = tE.k + 2 by omega]
          rfl
        · rw [show p.messages[st.msg]! = m from getElem!_of_getElem? hm]
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        exact ⟨_, rfl, by rw [f₉r, hrR], by rw [ite_eq_right hj]; omega, hnR, rfl, rfl, rfl, rfl, rfl,
          fun h => absurd h hj⟩)
    sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by omega) h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨hnA, hnB, yA, mA, t₁₂, hyA, hmA, hr₁₂, hk₁₂, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨hnC, hnD, yB, mB, t₁₃, hyB, hmB, hr₁₃, hk₁₃, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  have hlv0 := hlv 0 (by omega)
  obtain ⟨hnQ, hnQ', hmc, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel0_val hlv0.q_ne_zero h₄
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨hxe, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, hmq, k₁₅, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₁₇, rfl, hr₁₇, hk₁₇⟩ := sample_ok h₁₇
  obtain ⟨stL, hstL, h₄⟩ := lmodify_bind' h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hextra : extra = extraE p.initialProof.rows
      (Flock.positions (sch.levels.getD 0 default) (t₁₃.rounds[t₁₃.k]!).coins)
      (2 ^ (sch.levels.getD 0 default).lanes) zl.rep (sch.levels.getD 0 default).cols lane.2
      (t₁₃.rounds[t₁₃.k + 1]!).coins zl.msgs.ybar := by
    rw [hxe]
    rfl
  simp only at hoods₁₀ hconss₁₀ hood₁₀ hmsg₁₀ hq₁₀
  obtain ⟨hzy, hzr⟩ : zl.msgs.ybar.size = zl.tPad ∧ zl.rep < 2 := by simpa [Zk.EXTRA_LANES] using hzl
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
  have hwf₉ : RoundsWF t₉.rounds := er₉ ▸ hwf
  have h0 : 0 < sch.levels.size := by omega
  have hz₀ : (t₃'.rounds[t₃'.k]!).coins.size = (schedOf sch).m - 7 := by
    rw [RoundsWF.get (er₃ ▸ hwf), hn₅, hn7]; rfl
  have hβ₀ : (t₇.rounds[t₇.k]!).coins.size = 1 := by rw [RoundsWF.get (er₇ ▸ hwf), hn₈]
  have hFl : (consumed t₉.rounds t₉.k (t₉.k + sch.k0)).length = sch.k0 := by rw [consumed_length]; omega
  have hFs : ∀ c ∈ consumed t₉.rounds t₉.k (t₉.k + sch.k0), c.size = 2 :=
    consumed_sizes hwf₉ fun j hj => hn₁₀ j (by omega)
  have hc₀ : (t₉.rounds[t₉.k + sch.k0]!).coins.size = 2 := by
    rw [RoundsWF.get hwf₉]; exact hn₁₀ sch.k0 (by omega)
  have hc₁ : (t₉.rounds[t₉.k + sch.k0 + 1]!).coins.size = 2 := by
    rw [RoundsWF.get hwf₉]; exact hn₁₀ (sch.k0 + 1) (by omega)
  have hlaneSz : lane.1.size = sch.k0 := by
    rw [← Array.length_toList, hlane1]
    simp [consumed_length]
  have hini : lane.1.toList.map gOf = (consumed t₉.rounds t₉.k (t₉.k + sch.k0)).map
      fun c => execArith.ofLimbs (fOf c[0]!) (fOf c[1]!) := by
    rw [hlane1]
    simp only [List.nil_append, List.map_map]
    refine List.map_congr_left fun c _ => ?_
    rw [ofLimbs_coin]
    rfl
  have hfold : ∀ s q, ∃ s', Plays fOf (ZkSession.foldRoundsZK execArith P (schedOf sch).k0 0 s q)
      (kMsgs p 1 (sch.k0 - 1)) (consumed t₉.rounds t₉.k (t₉.k + sch.k0)) (lane.1.toList.map gOf, s') := by
    intro s q
    rw [hini]
    exact foldRoundsZK_val P p sch.k0 0 1 s q _ hFl hFs
  have hρ := drawKs_two _ _ hc₀ hc₁
  obtain ⟨ρ, hρdef⟩ : ∃ ρ : Fin 2 → 𝕂, ρ =
      Fin.cons (execArith.ofLimbs (fOf (t₉.rounds[t₉.k + sch.k0]!).coins[0]!)
        (fOf (t₉.rounds[t₉.k + sch.k0]!).coins[1]!))
        (Fin.cons (execArith.ofLimbs (fOf (t₉.rounds[t₉.k + sch.k0 + 1]!).coins[0]!)
          (fOf (t₉.rounds[t₉.k + sch.k0 + 1]!).coins[1]!)) Fin.elim0) := ⟨_, rfl⟩
  rw [← hρdef] at hρ
  have hρ0 : ρ 0 = gOf lane.2.1 := by
    rw [hρdef, hlane2]
    exact ofLimbs_coin _
  have hρ1 : ρ 1 = gOf lane.2.2 := by
    rw [hρdef, hlane2]
    exact ofLimbs_coin _
  have hq₁₀' : st₁₀.q = Poly.ofMsg p.messages[sch.k0]! zl.msgs.tPrime := by
    rw [hq₁₀]
    congr 2
    omega
  have hdimO : sch.n - sch.k0 + 1 = (schedOf sch).m - 7 - (schedOf sch).k0 + 1 := by
    show sch.n - sch.k0 + 1 = sch.m - 7 - sch.k0 + 1
    rw [hn7]
  obtain ⟨sT, qT, htwo, hsT, hqT⟩ := twoOodC_plays (D := Dig) o ((schedOf sch).m - 7 - (schedOf sch).k0 + 1) 0
    (fun _ => gOf zl.msgs.tPrime)
    (fun _ => Model.fromMsg (gOf p.messages[sch.k0]!.1, gOf p.messages[sch.k0]!.2) (gOf zl.msgs.tPrime))
    zl.msgs.tPrime (Poly.ofMsg p.messages[sch.k0]! zl.msgs.tPrime) rfl (quadOf_ofMsg _ _).symm
    (t₁₁.rounds[t₁₁.k]!).coins (t₁₁.rounds[t₁₁.k + 1]!).coins (t₁₂.rounds[t₁₂.k]!).coins
    (t₁₂.rounds[t₁₂.k + 1]!).coins (by rw [RoundsWF.get (er₁₁ ▸ hwf), hnA, hdimO])
    (by rw [RoundsWF.get (er₁₁ ▸ hwf), hnB]) (by rw [RoundsWF.get (er₁₂ ▸ hwf), hnC, hdimO])
    (by rw [RoundsWF.get (er₁₂ ▸ hwf), hnD]) (by show sch.m - 7 - sch.k0 + 1 ≠ 1; omega) yA yB mA mB
  have hopen0 : openAt p (sch.levels.size - 1) 0 =
      (p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts) := by simp [openAt]
  obtain ⟨-, hrows⟩ := merkleCheck_rows hmc
  have hcα₀ : (t₁₃.rounds[t₁₃.k + 1]!).coins.size = Nat.log2 ((sch.levels.getD 0 default).q - 1) + 1 := by
    rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']
  obtain ⟨stQ, hstQ⟩ : ∃ x : LigStateC 𝔽 𝕂 Dig, x = ⟨sT, qT, sch.m - 7 - sch.k0 + 1, [],
      ⟨List.ofFn (vecOf (sch.m - 7) (t₃'.rounds[t₃'.k]!).coins), fOf (t₇.rounds[t₇.k]!).coins[0]!, none⟩ ::
        [oodOf ⟨(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins.getD 0 0, some 0⟩,
          oodOf ⟨(t₁₂.rounds[t₁₂.k]!).coins, (t₁₂.rounds[t₁₂.k + 1]!).coins.getD 0 0, some 0⟩],
      [], 0, [], [], levelOf (sch.levels.getD 0 default)⟩ := ⟨_, rfl⟩
  have hE := padE_eq (sch.levels.getD 0 default) hlv0 hrate sch.k0 hl0 p.initialProof.rows o zl.rep stQ.level
    (t₁₃.rounds[t₁₃.k]!).coins (t₁₃.rounds[t₁₃.k + 1]!).coins lane.1 zl.tPad lane.2 ρ zl.msgs.ybar hcα₀ hlaneSz
    (fun j hj => by rw [hrows j hj, hl0]; rfl) (fun j hj => by rw [hstQ, ho 0 h0 j hj, hopen0]) hzr hρ0 hρ1 hzy
    hpad
  obtain ⟨stM₀, hstM₀⟩ : ∃ x : LigStateC 𝔽 𝕂 Dig, x =
      { qbPadOut zl.rep (levelOf (sch.levels.getD 0 default))
          (ZkLink.padColumnO execArith (levelOf (sch.levels.getD 0 default)).logLen
            (levelOf (sch.levels.getD 0 default)).logCols zl.tPad (fOf Zk.PAD_C) (lane.1.toList.map gOf) ρ
            (extraLane sch.k0 2 zl.rep) (fun i => gOf zl.msgs.ybar[i.val]!)) 0 stQ
          (t₁₃.rounds[t₁₃.k]!).coins (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins mq with
        level := 1, caps := [capDig p.recursiveCaps[0]!], prevLevel := levelOf (sch.levels.getD 1 default) } :=
    ⟨_, rfl⟩
  have hrel₀ : LoopRelV sch p o 0 (sch.n - sch.k0 + 1, p.recursiveCaps[0]!, #[], 0) stL t₁₇ stM₀ := by
    rw [hstM₀]
    refine ⟨er₁₇ ▸ hwf, ?_, ?_, ?_, rfl, rfl, by simp, ?_, ?_, ?_, ?_, rfl⟩
    · simp only [qbPadOut]
      rw [hE, hstQ]
      dsimp only
      rw [hsT, hstL]
      dsimp only
      rw [hs₁₀, hextra, getElem!_eq_getD]
      conv_rhs => rw [gOf_add, gOf_smul]
      rw [mul_comm]
      rfl
    · simp only [qbPadOut]
      rw [hE, hstQ]
      dsimp only
      rw [hqT, hstL]
      dsimp only
      rw [hq₁₀', hextra, getElem!_eq_getD]
      conv_rhs => rw [quadOf_addScaled, quadOf_ofMsg]
      rfl
    · show sch.n - sch.k0 + 1 = _
      rw [← hc0, ← hc01]
    · simp only [qbPadOut]
      rw [hstQ]
      rfl
    · simp only [qbPadOut]
      rw [hstQ, hstL]
      dsimp only
      rw [hoods₁₀, show List.ofFn (vecOf (sch.m - 7) (t₃'.rounds[t₃'.k]!).coins) =
        (t₃'.rounds[t₃'.k]!).coins.toList.map fOf from ofFn_vecOf _ _ hz₀]
      simp [oodOf, getElem!_eq_getD]
    · simp only [qbPadOut]
      rw [hstQ, hstL]
      dsimp only
      rw [hconss₁₀, show List.ofFn (vecOf (Accounting.ceilLog2 (levelOf (sch.levels.getD 0 default)).queries)
          (t₁₃.rounds[t₁₃.k + 1]!).coins) = (t₁₃.rounds[t₁₃.k + 1]!).coins.toList.map fOf from
          ofFn_vecOf _ _ (hcα₀.trans hlv0.2.2.2.1),
        show Model.positions execArith (levelOf (sch.levels.getD 0 default)).queries
          (levelOf (sch.levels.getD 0 default)).logLen
          (vecOf (levelOf (sch.levels.getD 0 default)).queries (t₁₃.rounds[t₁₃.k]!).coins) =
          (Flock.positions (sch.levels.getD 0 default) (t₁₃.rounds[t₁₃.k]!).coins).toList from
          Refine.positions_eq _ hlv0 _]
      rw [← hc0]
      simp [consOf, getElem!_eq_getD, levelOf]
    · rw [hstL]
      dsimp only
      rw [hconss₁₀]
      exact conss_push_pos (fun x hx => by simp at hx) (pos_le_alpha _ hlv0 _ _ hcα₀)
  have hL := loop_compose _ (sch.levels.size - 1)
    (fun i stM => Model.recursiveRoundsC execArith zl.rep (sch.m - 7) b (lane.1.toList.map gOf) sch.ext
      ((sch.levels.toList.map levelOf).drop (i + 1)) stM)
    (fun i v stE tE stM => LoopRelV sch p o i v stE tE stM)
    (fun i stE => recMsgs p capDig sch.ext (sch.levels.size - 1 - i) i stE.msg stE.ood)
    (fun v stE _ pend => LoopFinalV sch p o b (lane.1.toList.map gOf) v stE pend)
    ?hstep ?hlast (sch.levels.size - 1) 0 _ _ _ out st₁₈ t₁₈ stM₀ (by omega) (by omega) hrel₀ hloop
  case hstep =>
    clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1 ho cap
    intro i v stE tE bv stE' tE' stM hi hrel hb
    obtain ⟨hwfE, hsE, hqE, hsplit, hprev, hsize, hlanes, hlevels, hoods, hcons, hposE, hlevel⟩ := hrel
    have hc1 := hcols (i + 1) (by omega)
    have hl1 := hl4 (i + 1) (by omega) (by omega)
    have hl2 := hl4 (i + 1 + 1) (by omega) (by omega)
    try dsimp only at hb
    rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
    obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, hmsg₁, hood₁, -, -, -, hoods₁, hconss₁, hsz₁, -, sM₁, qM₁, hfoldM, hsM₁, hqM₁⟩ :=
      fold_refines p o 4 0 #[] chs stE st₁ tE t₁ stM.s stM.q hsE hqE hwfE hf
    simp only [Array.size_empty, Array.extract_size] at hfoldM
    rw [ite_eq_left hi] at hb
    obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
    have k₂ := absorbBytes_ok h₂
    obtain ⟨hnA, hnB, yA, mA, t₃, hyA, hmA, hr₃, hk₃, hb⟩ := oodIntro_bind (by omega) hb
    obtain ⟨hnC, hnD, yB, mB, t₄, hyB, hmB, hr₄, hk₄, hb⟩ := oodIntro_bind (by omega) hb
    have hlvi := hlv (i + 1) (by omega)
    obtain ⟨hnQ, hnQ', hmc, t₅, hr₅, hk₅, hb⟩ := queryLevel_bind hlvi.q_ne_zero hb
    obtain ⟨mq, t₆, hmq, k₆, hb⟩ := nextMsg_bind hb
    obtain ⟨t₇, k₇, hb⟩ := claimPow_bind hb
    obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
    obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
    obtain ⟨stN, hstN, hb⟩ := lmodify_bind' hb
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
    have hsz : chs.size = 4 := by simpa using hsz₁
    have hlen : (stM.levels ++ [chs.toList.map gOf]).length = i + 1 := by simp [hlevels, hsize]
    have hdim : (levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
        (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes = v.1 - 4 + 1 := by
      show (sch.levels.getD (i + 1 + 1) default).cols + (sch.levels.getD (i + 1 + 1) default).lanes = _
      omega
    obtain ⟨sT, qT, htwo, hsT, hqT⟩ := twoOodC_plays o (v.1 - 4 + 1) (i + 1) sM₁ qM₁ _ _ hsM₁ hqM₁
      (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins (t₃.rounds[t₃.k]!).coins (t₃.rounds[t₃.k + 1]!).coins
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnA]) (by rw [RoundsWF.get (er₂ ▸ hwfE), hnB])
      (by rw [RoundsWF.get (er₃ ▸ hwfE), hnC]) (by rw [RoundsWF.get (er₃ ▸ hwfE), hnD]) (by omega) yA yB mA mB
    have hcapAt : capAt cap p (i + 1) = v.2.1 := by rw [hprev]; simp [capAt]
    have hopen : openAt p (sch.levels.size - 1) (i + 1) =
        (p.recursiveProofs[i]!.rows, p.recursiveProofs[i]!.siblings, p.recursiveProofs[i]!.salts) := by
      simp [openAt, hi]
    have hmc' : merkleCheck ms (capAt cap p (i + 1)) (sch.levels.getD (i + 1) default).cap
        (sch.levels.getD (i + 1) default).d (Flock.positions (sch.levels.getD (i + 1) default) (t₄.rounds[t₄.k]!).coins)
        (openAt p (sch.levels.size - 1) (i + 1)).1 (openAt p (sch.levels.size - 1) (i + 1)).2.1
        (openAt p (sch.levels.size - 1) (i + 1)).2.2 (2 ^ (sch.levels.getD (i + 1) default).lanes) = .ok () := by
      rw [hcapAt, hopen]; exact hmc
    set stQ : LigStateC 𝔽 𝕂 Dig := { stM with
        s := sT, q := qT, levels := stM.levels ++ [chs.toList.map gOf],
        oods := stM.oods ++ [oodOf ⟨(t₂.rounds[t₂.k]!).coins, (t₂.rounds[t₂.k + 1]!).coins.getD 0 0, some (i + 1)⟩,
          oodOf ⟨(t₃.rounds[t₃.k]!).coins, (t₃.rounds[t₃.k + 1]!).coins.getD 0 0, some (i + 1)⟩] } with hstQ
    have hq := qbOut_rel ms sch cap p zl.rep (i + 1) (by omega) hlvi chs (by rw [hsz, hl1])
      (stM.levels ++ [chs.toList.map gOf]).length stQ hlevel o (ho (i + 1) (by omega)) (t₄.rounds[t₄.k]!).coins
      (t₄.rounds[t₄.k + 1]!).coins (t₇.rounds[t₇.k]!).coins (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']) hmc' true mq
    rw [hopen] at hq
    obtain ⟨hqs, hqq, hqc, -, hql, hqo, -, -⟩ := hq
    set qb := qbOut zl.rep (levelOf (sch.levels.getD (i + 1) default)) (chs.toList.map gOf)
      (stM.levels ++ [chs.toList.map gOf]).length stQ (t₄.rounds[t₄.k]!).coins (t₄.rounds[t₄.k + 1]!).coins
      (t₇.rounds[t₇.k]!).coins true mq with hqb
    have hmsgA := getElem!_of_getElem? hmA
    have hoodA := getElem!_of_getElem? hyA
    have hmsgB := getElem!_of_getElem? hmB
    have hoodB := getElem!_of_getElem? hyB
    have hmsgQ := getElem!_of_getElem? hmq
    simp only [hmsg₁, hood₁, Nat.add_assoc, Nat.reduceAdd] at hmsgA hoodA hmsgB hoodB hmsgQ
    subst hstN
    refine ⟨_, { qb with
        level := stM.level + 1, caps := stM.caps ++ [capDig p.recursiveCaps[i + 1]!],
        prevLevel := levelOf (sch.levels.getD (i + 1 + 1) default) },
      kMsgs p stE.msg 4 ++ ⟨Cap Dig, capDig p.recursiveCaps[i + 1]!⟩ ::
        [fMsg p.oodValues[stE.ood]!, kMsg p.messages[stE.msg + 4]!, fMsg p.oodValues[stE.ood + 1]!,
          kMsg p.messages[stE.msg + 5]!] ++ [kMsg p.messages[stE.msg + 6]!],
      rfl, ?rel, er₈, by omega, ?tape, ?plays⟩
    case rel =>
      refine ⟨er₈ ▸ hwfE, ?_, ?_, ?_, rfl, by simp [hsize], ?_, ?_, ?_, ?_, ?_, ?_⟩
      · show qb.s o = _
        rw [hqs]
        simp only [hstQ, hsT, gOf_add, gOf_smul]
        rfl
      · show qb.q o = _
        rw [hqq]
        simp only [ite_true, hstQ, hqT, quadOf_addScaled, quadOf_ofMsg]
        rfl
      · show v.1 - 4 + 1 = _
        omega
      · intro l hl
        simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hl
        rcases hl with hl | rfl
        · exact hlanes l hl
        · exact hsz
      · show qb.levels = _
        rw [hql, hstQ]
        simp [hlevels]
      · show qb.oods = _
        rw [hqo, hstQ]
        simp [hoods, hoods₁]
      · show qb.cons = _
        rw [hqc, hstQ, hlen]
        simp [hcons, hconss₁, hsplit]
      · dsimp only
        rw [hconss₁]
        exact conss_push_pos hposE (pos_le_alpha _ hlvi _ _ (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']))
      · show stM.level + 1 = _
        rw [hlevel]
    case tape =>
      obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 - i = j + 2 := ⟨sch.levels.size - 1 - i - 2, by omega⟩
      rw [hj, show sch.levels.size - 1 - (i + 1) = j + 1 by omega]
      simp only [recMsgs, hmsg₁, hood₁, Nat.add_assoc, Nat.reduceAdd, List.append_assoc, List.cons_append]
    case plays =>
      intro msR csR pend hrec
      rw [drop_levels sch (i + 1) (by omega), drop_levels sch (i + 1 + 1) (by omega)]
      rw [drop_levels sch (i + 1 + 1) (by omega)] at hrec
      have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (i + 1) default)).lanes stM.s stM.q)
          (kMsgs p stE.msg 4) (consumed tE.rounds tE.k t₁.k) (List.map gOf chs.toList, sM₁, qM₁) := by
        rw [show (levelOf (sch.levels.getD (i + 1) default)).lanes = 4 from hl1]; exact hfoldM
      have htwo' : Plays fOf (Model.twoOodC execArith ((levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes) (stM.levels ++ [chs.toList.map gOf]).length sM₁ qM₁)
          [fMsg yA, kMsg mA, fMsg yB, kMsg mB]
          [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₃.rounds[t₃.k]!.coins, t₃.rounds[t₃.k + 1]!.coins]
          (sT, qT, [oodOf ⟨(t₂.rounds[t₂.k]!).coins, (t₂.rounds[t₂.k + 1]!).coins.getD 0 0, some (i + 1)⟩,
            oodOf ⟨(t₃.rounds[t₃.k]!).coins, (t₃.rounds[t₃.k + 1]!).coins.getD 0 0, some (i + 1)⟩]) := by
        rw [hdim, hlen]; exact htwo
      have hmain := iterNL_model zl.rep (sch.m - 7) b (lane.1.toList.map gOf) sch.ext
        (levelOf (sch.levels.getD (i + 1) default)) (levelOf (sch.levels.getD (i + 1 + 1) default))
        ((sch.levels.toList.map levelOf).drop (i + 1 + 1 + 1)) stM hfold' (capDig p.recursiveCaps[i + 1]!) htwo'
        (t₄.rounds[t₄.k]!).coins (t₄.rounds[t₄.k + 1]!).coins (t₇.rounds[t₇.k]!).coins
        (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ]; rfl)
        (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']; exact hlvi.2.2.2.1)
        (by rw [RoundsWF.get (er₇ ▸ hwfE), hn₈]) hlvi.q_ne_one (hα1 (i + 1) (by omega)) mq hrec
      have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k t₁.k ++
          [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₃.rounds[t₃.k]!.coins, t₃.rounds[t₃.k + 1]!.coins] ++
          [t₄.rounds[t₄.k]!.coins, t₄.rounds[t₄.k + 1]!.coins, t₇.rounds[t₇.k]!.coins] := by
        rw [er₂, ek₂, er₃, ek₃, er₄, ek₄, er₇, ek₇, ek₈, hk₁,
          consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 11 by omega),
          show tE.k + 11 = tE.k + 4 + 7 by omega, consumed_add tE.rounds (tE.k + 4) 7]
        simp [List.range_succ, Nat.add_assoc]
      rw [hcs, hmsgA, hoodA, hmsgB, hoodB, hmsgQ]
      simp only [List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
      exact hmain
  case hlast =>
    clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1 ho cap
    intro i v stE tE bv stE' tE' stM hi hrel hb
    obtain ⟨hwfE, hsE, hqE, hsplit, hprev, hsize, hlanes, hlevels, hoods, hcons, hposE, hlevel⟩ := hrel
    have hl1 := hl4 (i + 1) (by omega) (by omega)
    try dsimp only at hb
    rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
    obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, hmsg₁, hood₁, -, -, -, hoods₁, hconss₁, hsz₁, -, sM₁, qM₁, hfoldM, hsM₁, hqM₁⟩ :=
      fold_refines p o 4 0 #[] chs stE st₁ tE t₁ stM.s stM.q hsE hqE hwfE hf
    simp only [Array.size_empty, Array.extract_size] at hfoldM
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
    obtain ⟨hnQ, hnQ', hmc, t₃, hr₃, hk₃, hb⟩ := queryLevel_bind hlvR.q_ne_zero hb
    obtain ⟨t₄, k₄, hb⟩ := claimPow_bind hb
    obtain ⟨bq, t₅, h₅, hb⟩ := lift_bind hb
    obtain ⟨-, hn₅, rfl, hr₅, hk₅⟩ := sample_ok h₅
    obtain ⟨stN, hstN, hb⟩ := lmodify_bind' hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
    have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
    have er₄ : t₄.rounds = tE.rounds := by rw [k₄.2, hr₃, er₂]
    have ek₄ : t₄.k = tE.k + 6 := by rw [k₄.1, hk₃, ek₂]
    have er₅ : tE'.rounds = tE.rounds := hr₅.trans er₄
    have ek₅ : tE'.k = tE.k + 7 := by rw [hk₅, ek₄]
    have hsz : chs.size = 4 := by simpa using hsz₁
    rw [hi] at hsplit hl1
    have hlen : (stM.levels ++ [chs.toList.map gOf]).length = sch.levels.size - 1 := by
      simp [hlevels, hsize]; omega
    have hcapAt : capAt cap p (sch.levels.size - 1) = v.2.1 := by
      rw [hprev]; simp [capAt, show sch.levels.size - 1 ≠ 0 by omega, show sch.levels.size - 1 - 1 = i by omega]
    have hopen : openAt p (sch.levels.size - 1) (sch.levels.size - 1) =
        (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts) := by
      simp [openAt, show sch.levels.size - 1 ≠ 0 by omega]
    have hmc' : merkleCheck ms (capAt cap p (sch.levels.size - 1)) (sch.levels.getD (sch.levels.size - 1) default).cap
        (sch.levels.getD (sch.levels.size - 1) default).d
        (Flock.positions (sch.levels.getD (sch.levels.size - 1) default) (t₂.rounds[t₂.k]!).coins)
        (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).1 (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).2.1
        (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).2.2
        (2 ^ (sch.levels.getD (sch.levels.size - 1) default).lanes) = .ok () := by
      rw [hcapAt, hopen]; exact hmc
    set stQ : LigStateC 𝔽 𝕂 Dig := { stM with s := sM₁, q := qM₁, levels := stM.levels ++ [chs.toList.map gOf] }
      with hstQ
    have hq := qbOut_rel ms sch cap p zl.rep (sch.levels.size - 1) (by omega) hlvR chs (by rw [hsz, hl1])
      (stM.levels ++ [chs.toList.map gOf]).length stQ (by rw [hstQ]; exact hlevel.trans hi) o
      (ho (sch.levels.size - 1) (by omega)) (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins
      (t₄.rounds[t₄.k]!).coins (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']) hmc' false (0, 0)
    rw [hopen] at hq
    obtain ⟨hqs, hqq, hqc, -, hql, hqo, -, -⟩ := hq
    have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes
        stM.s stM.q) (kMsgs p stE.msg 4) (consumed tE.rounds tE.k t₁.k) (List.map gOf chs.toList, sM₁, qM₁) := by
      rw [show (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes = 4 from hl1]; exact hfoldM
    have hmain := iterLast_model zl.rep (sch.m - 7) b (lane.1.toList.map gOf) sch.ext
      (levelOf (sch.levels.getD (sch.levels.size - 1) default)) stM hfold' (yrOf p sch.ext)
      (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins (t₄.rounds[t₄.k]!).coins
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ]; rfl)
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']; exact hlvR.2.2.2.1)
      (by rw [RoundsWF.get (er₄ ▸ hwfE), hn₅]) hlvR.q_ne_one (hα1 (sch.levels.size - 1) (by omega))
    have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k t₁.k ++
        [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₄.rounds[t₄.k]!.coins] := by
      rw [er₂, ek₂, er₄, ek₄, ek₅, hk₁,
        consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 7 by omega),
        show tE.k + 7 = tE.k + 4 + 3 by omega, consumed_add tE.rounds (tE.k + 4) 3]
      simp [List.range_succ, Nat.add_assoc]
    have hdrop : (sch.levels.toList.map levelOf).drop (i + 1) =
        [levelOf (sch.levels.getD (sch.levels.size - 1) default)] := by
      rw [hi, drop_levels sch (sch.levels.size - 1) (by omega), show sch.levels.size - 1 + 1 = sch.levels.size by omega]
      simp
    rw [← hstQ] at hmain
    rw [hdrop, show sch.levels.size - 1 - i = 1 by omega, hcs]
    simp only [recMsgs]
    refine ⟨_, _, rfl, er₅, by omega, hmain, ?_⟩
    subst hstN
    refine ⟨by simp [hsize]; omega, ?_, by simp only; omega, ?_, ?_⟩
    · intro l hl
      simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hl
      rcases hl with hl | rfl
      · exact hlanes l hl
      · exact hsz
    · dsimp only
      rw [hql, hqo, hqc, hqs, hstQ, hlen]
      dsimp only
      rw [hlevels, hoods, hcons, hsM₁, hoods₁, hconss₁,
        show v.1 - 4 = (sch.levels.getD (sch.levels.size - 1) default).cols by omega, gOf_add, gOf_smul]
      simp only [Array.toList_push, List.map_append, List.map_cons, List.map_nil]
      rfl
    · dsimp only
      rw [hconss₁]
      exact conss_push_pos hposE (pos_le_alpha _ hlvR _ _ (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']))
  obtain ⟨pend, hrL, hkL, hplL, hsizeL, hlanesL, hextL, hverL, hposL⟩ := hL
  have hfin : finalProgZ p basisAt sch.n T lane.1 lane.2 out.2.2.1 out.2.2.2 st₁₈ t₁₈ = .ok ((rc', stF), t') :=
    hcont
  rw [hextL] at hfin
  obtain ⟨-, -, horig, -⟩ := finalProgZ_facts hfin
  obtain ⟨rfl, hfc⟩ := final_refinesZ (n := sch.n) b hlanesL hposL
    (fun y hy => by rw [hn7]; exact hbasis _ y (by omega) hy) hfin
  have hlv01 : (schedOf sch).levels = levelOf (sch.levels.getD 0 default) ::
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := by
    have e0 := drop_levels sch 0 h0
    have e1 := drop_levels sch 1 (by omega)
    simp only [List.drop_zero, Nat.zero_add, Nat.reduceAdd] at e0 e1
    rw [e1] at e0
    exact e0
  have hd1 : (sch.levels.toList.map levelOf).drop (0 + 1) =
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := drop_levels sch 1 (by omega)
  rw [hd1, hstM₀, hstQ] at hplL
  obtain ⟨row, hmain⟩ := ligeritoPadZC_model (D := Dig) P (schedOf sch) zl.tPad (fOf Zk.PAD_C) zl.rep Tm b hlv01
    (t₃'.rounds[t₃'.k]!).coins (t₇.rounds[t₇.k]!).coins hz₀ (by show sch.m - 7 ≠ 1; omega) hβ₀
    (fOf p.oodValues[0]!) (gOf p.messages[0]!.1, gOf p.messages[0]!.2) hfold ![gOf zl.msgs.e0, gOf zl.msgs.e1] hρ
    (gOf zl.msgs.tPrime) (fun i => gOf zl.msgs.ybar[i.val]!) p.messages[sch.k0]! (capDig p.recursiveCaps[0]!) htwo
    (t₁₃.rounds[t₁₃.k]!).coins (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins
    (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ]; rfl) (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']; exact hlv0.2.2.2.1)
    (by rw [RoundsWF.get (er₁₆ ▸ hwf), hn₁₇]) hlv0.q_ne_one (hα1 0 h0) mq hplL
  have hcs : consumed t.rounds t.k t'.k = (t₃'.rounds[t₃'.k]!).coins :: (t₇.rounds[t₇.k]!).coins ::
      (consumed t₉.rounds t₉.k (t₉.k + sch.k0) ++
        ([(t₉.rounds[t₉.k + sch.k0]!).coins, (t₉.rounds[t₉.k + sch.k0 + 1]!).coins] ++
          ([(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins, (t₁₂.rounds[t₁₂.k]!).coins,
              (t₁₂.rounds[t₁₂.k + 1]!).coins] ++
            ([(t₁₃.rounds[t₁₃.k]!).coins, (t₁₃.rounds[t₁₃.k + 1]!).coins, (t₁₆.rounds[t₁₆.k]!).coins] ++
              consumed t₁₇.rounds t₁₇.k t'.k)))) := by
    rw [er₃, ek₃, er₇, ek₇, er₉, ek₉, er₁₁, ek₁₁, er₁₂, ek₁₂, er₁₃, ek₁₃, er₁₆, ek₁₆, er₁₇, ek₁₇]
    rw [ek₁₇] at hkL
    rw [consumed_cons _ (show t.k < t'.k by omega), consumed_cons _ (show t.k + 1 < t'.k by omega),
      show t.k + 1 + 1 = t.k + 2 from rfl,
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 2 + sch.k0 by omega)
        (show t.k + 2 + sch.k0 ≤ t'.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 ≤ t.k + 2 + sch.k0 + 9 by omega)
        (show t.k + 2 + sch.k0 + 9 ≤ t'.k by omega),
      consumed_add t.rounds (t.k + 2 + sch.k0) 9]
    simp [List.range_succ, Nat.add_assoc]
  have hyA' := getElem!_of_getElem? hyA
  have hmA' := getElem!_of_getElem? hmA
  have hyB' := getElem!_of_getElem? hyB
  have hmB' := getElem!_of_getElem? hmB
  have hmq' := getElem!_of_getElem? hmq
  simp only [hood₁₀, hmsg₁₀] at hyA' hmA' hyB' hmB' hmq'
  have hLmsg : stL.msg = sch.k0 + 4 := by rw [hstL]; simp only [hmsg₁₀]; omega
  have hLood : stL.ood = 3 := by rw [hstL]; simp only [hood₁₀]
  refine ⟨(row, pend), ?_, ?_⟩
  · rw [hcs]
    unfold ligMsgsZ
    rw [hLmsg, hLood, Nat.sub_zero] at hmain
    rw [show 0 + 1 + sch.k0 = sch.k0 + 1 by omega] at hmA'
    rw [show 0 + 1 + sch.k0 + 1 = sch.k0 + 2 by omega] at hmB'
    rw [show 0 + 1 + sch.k0 + 1 + 1 = sch.k0 + 3 by omega] at hmq'
    rw [show (0 + 1 + 1 : ℕ) = 2 from rfl] at hyB'
    rw [show (0 + 1 : ℕ) = 1 from rfl] at hyA'
    rw [← hyA', ← hmA', ← hyB', ← hmB', ← hmq'] at hmain
    simp only [List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
    exact hmain
  · show pend.verdict o = true
    rw [hverL, ← hn7]
    exact hfc

end FlockSoundness.Discharge.ZkExec
