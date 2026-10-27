import Proofs.Flock.Claims.Recursion.VerdictRowDCover
import Proofs.Flock.Claims.Recursion.RoundSizes
import Proofs.Flock.Claims.Complete.Replay.LigeritoWalk
import Proofs.Flock.Verifier.Hash

/-!
# Row D's `hlev`: the verifier's query positions, and the honest proof's Merkle checks

`DecAt.levels_of_merkleOk` gives row D's `hlev` from FlockComplete's `Lg.MerkleOk` at some query coins `cf` from a
round `k`, and the tie `htie` that every run of the verifier rows queries level `ℓ` at the positions those coins draw at
round `k + qOff sch ℓ`.

* **The tie, outright** (`rows_pos`, `DecAt.htie`): any run of `verifierRows` on rounds with no empty round queries
  level `ℓ` at the positions of the rounds' own coins at `lgStart + qOff sch ℓ`, where `lgStart` is Ligerito's first
  round (after the zerocheck's, the lincheck's and ring switching's), step by step as `ligeritoRun` draws them
  (`levelsLoop_pos`, `ligeritoRun_pos`). So `htie` holds at `cf = cfOf rounds`, `k = lgStart`.
* **`hlev` from `MerkleOk` at any coins that agree there** (`merkleOk_congr`, `DecAt.hlev_of_merkleOk`): `MerkleOk`
  reads its coins only at the queried rounds.
* **The honest prover's `MerkleOk`** (`FlockComplete.Replay.merkleOk_of_repP`): a returning honest rep `repP` on a
  `commitP` of its table passes every level's Merkle check at its replies' coins from Ligerito's first round
  (`repP_opens`, `merkleOk_of_opens`), at any hashes whose digests have the scheme's width. `DecAt.hlev_of_repP`: row
  D's `hlev` at a rep whose decided proof is such a run's output, and whose replies' coins at the queried rounds are its
  recorded rounds'.
* **At the deployed hashes** (`DecAt.hlev_deployed`): `Hashes.ofSha512 Sha512.hash` makes the plain scheme its own
  closures (`plain_ops_std`) with 64-byte digests (`plain_widths_std`), so only the honest rep's `hrun`, `h0` and
  `hcoin` remain.
-/

namespace FlockVBridge.Family

open Flock FlockSoundness.Refine
open Flock.RecAlgebra (Messages LSt Coins)
open FlockComplete.Replay.Lg (qOff)

/-- Ligerito's first round at a shape with `2 + n` claims: after the zerocheck's `3 + (m − 6)` rounds, the lincheck's
`3 + (k_log − 6)` and ring switching's `2 + n + 1`. -/
def lgStart (sh : Shape) (n : ℕ) : ℕ := 3 + (sh.m - 6) + 3 + (sh.kLog - 6) + (2 + n) + 1

/-- The recorded rounds as coins: round `j`'s `t`-th coin, `0` past its end. -/
def cfOf (w : Array (Array F128)) : ℕ → ℕ → F128 := fun j t => (w[j]!).getD t 0

theorem range_cfOf {w : Array (Array F128)} {j q : ℕ} (h : (w[j]!).size = q) :
    (Array.range q).map (cfOf w j) = w[j]! := by
  apply Array.ext
  · simp [h]
  · intro t h1 h2
    simp [cfOf, Array.getD_eq_getD_getElem?, h2]

theorem range_map_cons {α : Type} (f : ℕ → α) {n : ℕ} (hn : 0 < n) :
    (List.range n).map f = f 0 :: (List.range' 1 (n - 1)).map f := by
  obtain ⟨n, rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
  rw [List.range_eq_range', List.range'_succ]
  simp

/-! ## The positions Ligerito queries -/

section Pos

variable {w : Array (Array F128)} [Fact (Flock.RecAlgebra.NoEmpty w)] {M : Messages F128}

/-- V*'s level loop queries level `l` at the positions of round `k₀ + qOff sch l`'s coins, `q` of them, from Ligerito's
first round `k₀`. -/
theorem levelsLoop_pos {sch : Flock.Schedule} {k₀ : ℕ}
    (hq : ∀ i ≤ sch.levels.size - 1, (sch.levels.getD i default).q ≠ 0) :
    ∀ (fuel i split : ℕ) (levels : Array (Array F256)) (e : ℕ) (st : LSt F128) (k : ℕ)
      (out : Array (Array F256) × ℕ) (st' : LSt F128) (k' : ℕ), i + fuel ≤ sch.levels.size - 1 →
      k = k₀ + sch.k0 + 9 + 11 * i →
      Flock.RecAlgebra.levelsLoop sch M (sch.levels.size - 1) fuel i split levels e st k (Coins.ofRounds w) =
        .ok ((out, st'), k') →
      st'.conss.toList.map (·.pos) = st.conss.toList.map (·.pos) ++
          (List.range' (i + 1) fuel).map (fun l => positions (sch.levels.getD l default) w[k₀ + qOff sch l]!) ∧
        ∀ l, i + 1 ≤ l → l < i + 1 + fuel → (w[k₀ + qOff sch l]!).size = (sch.levels.getD l default).q
  | 0, _, _, _, _, st, k, out, st', k', _, _, h => by
    simp only [Flock.RecAlgebra.levelsLoop] at h
    obtain ⟨-, rfl, -⟩ := Flock.RecAlgebra.vpure h
    exact ⟨by simp, fun l h1 h2 => by omega⟩
  | fuel + 1, i, split, levels, e, st, k, out, st', k', hi, hk, h => by
    simp only [Flock.RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (Flock.RecAlgebra.vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := Flock.RecAlgebra.vbind_ok h
    obtain ⟨e₁, -, -, -, -, hc₁, -⟩ :=
      Flock.RecAlgebra.challenges_conv (p := default) 4 0 #[] st st₁ k k₁ chs h₁
    try dsimp only at h
    by_cases hi' : i + 1 < sch.levels.size - 1
    · rw [ite_eq_left hi'] at h
      obtain ⟨u₂, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨u₃, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨u₄, st₄, k₄, h₄, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨e₂, -, -, hc₂, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
      obtain ⟨e₃, -, -, hc₃, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
      obtain ⟨e₄, hz, -, -, mm, -, -, -, -, hc₄, -⟩ := Flock.RecAlgebra.vqueryFold (hq (i + 1) (by omega)) h₄
      obtain ⟨ih, ihs⟩ := levelsLoop_pos (k₀ := k₀) hq fuel (i + 1) _ _ _ st₄ k₄ out st' k' (by omega) (by omega) h
      have hr : k₃ = k₀ + qOff sch (i + 1) := by
        unfold qOff
        rw [ite_eq_left hi']
        omega
      rw [hr] at hc₄ hz
      refine ⟨?_, fun l h1 h2 => ?_⟩
      · rw [ih, hc₄, Array.toList_push, hc₃, hc₂, hc₁, List.range'_succ]
        simp only [List.map_append, List.map_cons, List.map_nil, List.append_assoc, List.cons_append,
          List.nil_append]
      · rcases Nat.eq_or_lt_of_le h1 with rfl | h1'
        · exact hz
        · exact ihs l (by omega) (by omega)
    · rw [ite_eq_right hi'] at h
      obtain ⟨r, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨rfl, -, hz, -, rfl⟩ := Flock.RecAlgebra.vquery (hq _ le_rfl) h₂
      try dsimp only at h
      obtain ⟨-, h⟩ := Flock.RecAlgebra.vcoinOne_bind h
      obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨e₃, -⟩ := Flock.RecAlgebra.vmodify_ok h₃
      obtain rfl : fuel = 0 := by omega
      simp only [Flock.RecAlgebra.levelsLoop] at h
      obtain ⟨-, rfl, -⟩ := Flock.RecAlgebra.vpure h
      have hR : sch.levels.size - 1 = i + 1 := by omega
      have hr : k₁ = k₀ + qOff sch (i + 1) := by
        unfold qOff
        rw [ite_eq_right (by omega)]
        omega
      rw [hr, hR] at e₃
      rw [hr, hR] at hz
      refine ⟨?_, fun l h1 h2 => ?_⟩
      · rw [e₃, Array.toList_push, hc₁]
        simp
      · obtain rfl : l = i + 1 := by omega
        exact hz

/-- **V*'s Ligerito up to its final check queries level `l` at the positions of round `k₀ + qOff sch l`'s coins**, `q`
of them, from its first round `k₀`, recording level `l`'s context `l`-th. -/
theorem ligeritoRun_pos {sch : Flock.Schedule} {st₀ : LSt F128} {k₀ k' : ℕ}
    {out : Array F256 × Array (Array F256) × ℕ} {stV : LSt F128} (h0 : st₀.conss = #[]) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hrun : (Flock.RecAlgebra.ligeritoRun sch M).run st₀ k₀ (Coins.ofRounds w) = .ok ((out, stV), k')) :
    stV.conss.toList.map (·.pos) =
        (List.range sch.levels.size).map (fun l => positions (sch.levels.getD l default) w[k₀ + qOff sch l]!) ∧
      ∀ l < sch.levels.size, (w[k₀ + qOff sch l]!).size = (sch.levels.getD l default).q := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hq : ∀ i ≤ sch.levels.size - 1, (sch.levels.getD i default).q ≠ 0 :=
    fun i hi => (hlv i (by omega)).q_ne_zero
  have hc01 := hcols 0 (by omega)
  rw [StateT.run] at hrun
  unfold Flock.RecAlgebra.ligeritoRun at hrun
  obtain ⟨-, h⟩ := Flock.RecAlgebra.vcoinVec_bind (by omega) hrun
  obtain ⟨y0, -, h⟩ := Flock.RecAlgebra.vnextOod_bind h
  obtain ⟨-, h⟩ := Flock.RecAlgebra.vcoinOne_bind h
  obtain ⟨⟨⟩, stZ, kZ, hZ, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨eZ, rfl⟩ := Flock.RecAlgebra.vmodify_ok hZ
  obtain ⟨m0, -, h⟩ := Flock.RecAlgebra.vnextMsg_bind h
  obtain ⟨⟨⟩, stA, kA, hA, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨eA, rfl⟩ := Flock.RecAlgebra.vmodify_ok hA
  obtain ⟨ic, st₁, k₁, h₁, h⟩ := Flock.RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₄, k₄, h₄, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨lv', ex'⟩, st₅, k₅, h₅, h⟩ := Flock.RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨-, rfl, -⟩ := Flock.RecAlgebra.vpure h
  obtain ⟨e₁, -, -, -, -, hc₁, -⟩ := Flock.RecAlgebra.challenges_conv (p := default) sch.k0 0 #[] _ _ _ _ _ h₁
  obtain ⟨e₂, -, -, hc₂, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
  obtain ⟨e₃, -, -, hc₃, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
  obtain ⟨e₄, hz, -, -, mq, -, -, -, -, hc₄, -⟩ := Flock.RecAlgebra.vqueryFold (hlv 0 (by omega)).q_ne_zero h₄
  have hA0 : stA.conss = #[] := by
    rw [eA, eZ]
    exact h0
  have hr : k₃ = k₀ + qOff sch 0 := by
    unfold qOff
    rw [ite_eq_left (by omega)]
    omega
  rw [hr] at hc₄ hz
  obtain ⟨hL, hLs⟩ := levelsLoop_pos (k₀ := k₀) hq _ _ _ _ _ st₄ _ _ _ _ (by omega) (by omega) h₅
  refine ⟨?_, fun l hl => ?_⟩
  · rw [hL, hc₄, Array.toList_push, hc₃, hc₂, hc₁, hA0, range_map_cons _ (by omega)]
    simp
  · rcases Nat.eq_zero_or_pos l with rfl | hl0'
    · exact hz
    · exact hLs l (by omega) (by omega)

end Pos

/-- `ligerito` returns the contexts of its `ligeritoRun`, run from a state with none. -/
theorem ligerito_run_conss {sch : Flock.Schedule} {M : Messages F128} {tIn : F128} {sws : Array Switched}
    {k k' : ℕ} {src : Flock.RecAlgebra.Coins} {r : Array F128 × Flock.RecAlgebra.LgPts}
    (h : Flock.RecAlgebra.ligerito sch M tIn sws k src = .ok (r, k')) :
    ∃ (st₀ : LSt F128) (out : Array F256 × Array (Array F256) × ℕ) (st : LSt F128) (k₁ : ℕ), st₀.conss = #[] ∧
      (Flock.RecAlgebra.ligeritoRun sch M).run st₀ k src = .ok ((out, st), k₁) ∧ r.2.conss = st.conss := by
  unfold Flock.RecAlgebra.ligerito at h
  obtain ⟨⟨⟨initCh, levels, ext⟩, st⟩, k₁, h1, h⟩ := Flock.RecAlgebra.cm_bind h
  obtain ⟨a, k₂, ha, h⟩ := Flock.RecAlgebra.cm_bind h
  cases ha
  obtain ⟨b, k₃, hb, h⟩ := Flock.RecAlgebra.cm_bind h
  cases hb
  by_cases cd : src.drawnAll k₁ = true
  rotate_left
  · rw [ite_eq_left (by simpa using cd)] at h
    obtain ⟨_, _, h0, _⟩ := Flock.RecAlgebra.cm_bind h
    cases h0
  rw [ite_eq_right (by simpa using cd)] at h
  dsimp only at h
  by_cases cm : st.msg = M.msgs.size ∧ st.ood = M.ood.size
  rotate_left
  · rw [ite_eq_left (by omega)] at h
    obtain ⟨_, _, h0, _⟩ := Flock.RecAlgebra.cm_bind h
    cases h0
  rw [ite_eq_right (by omega)] at h
  obtain ⟨res, k₆, h6, h⟩ := Flock.RecAlgebra.cm_bind h
  obtain ⟨res', hfw, h6⟩ := Flock.RecAlgebra.liftE_ok h6
  cases h6
  by_cases cy : M.yr.size = 2 <<< ext
  rotate_left
  · rw [ite_eq_left cy] at h
    obtain ⟨_, _, h0, _⟩ := Flock.RecAlgebra.cm_bind h
    cases h0
  rw [ite_eq_right (fun hn => hn cy)] at h
  cases Flock.RecAlgebra.cm_pure h
  refine ⟨_, _, _, _, ?_, h1, ?_⟩ <;> rfl

/-- **The verifier rows query level `ℓ` at the positions of the rounds' own coins at `lgStart + qOff sch ℓ`**: at any
run of `verifierRows` on rounds with no empty round, at `m ≥ 14` and a schedule of the seams, context `ℓ`'s positions
are those of `q` coins of that round. -/
theorem rows_pos {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin} {kappa : Array (Array (Array F128))}
    (hne : Flock.RecAlgebra.NoEmpty rounds) (hm : 14 ≤ sh.m) (hdim : DimSeam sh.sched)
    (hlv : ∀ i < sh.sched.levels.size, LevelSeam (sh.sched.levels.getD i default))
    (h : Flock.RecAlgebra.verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    ∀ ℓ < sh.sched.levels.size, (info.lg.conss[ℓ]!).pos = positions (sh.sched.levels[ℓ]!)
      ((Array.range (sh.sched.levels[ℓ]!).q).map (cfOf rounds (lgStart sh extras.size + qOff sh.sched ℓ))) := by
  have : Fact (Flock.RecAlgebra.NoEmpty rounds) := ⟨hne⟩
  obtain ⟨lins, k, hr⟩ := FlockVBridge.Runs.run_of_rows h
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, h0, h1, h2, h3, -⟩ := Flock.RecAlgebra.run_stages hr
  obtain ⟨-, -, -, -, -, -, -, -, -, hk1, -⟩ := Flock.RecAlgebra.zerocheck_coins hm h0
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, hk2, -⟩ := Flock.RecAlgebra.lincheck_coins h1
  have hcs : (Flock.RecAlgebra.claimsOf sh.m sh.kLog info.zc info.lc extras).size = 2 + extras.size := by
    simp [Flock.RecAlgebra.claimsOf]
  obtain ⟨_, -, -, -, hk3, -, -, -⟩ := Flock.RecAlgebra.ringSwitch_spec (by omega) h2
  have hzc : (Flock.RecAlgebra.symbolic sh).zc.size = sh.m - 6 := by simp [Flock.RecAlgebra.symbolic]
  have hlc : (Flock.RecAlgebra.symbolic sh).lc.size = sh.kLog - 6 := by simp [Flock.RecAlgebra.symbolic]
  rw [hzc] at hk1
  rw [hlc] at hk2
  rw [hcs] at hk3
  have hks : k3 = lgStart sh extras.size := by
    unfold lgStart
    omega
  have h3' := Flock.RecAlgebra.ligerito_sim (Flock.RecAlgebra.Lin.eval_hom fun _ => 0) _ _ _ _ _ _ _ _ h3
  obtain ⟨st₀, out, stV, k₄, hst0, hrun, hcc⟩ := ligerito_run_conss h3'
  have hc : info.lg.conss = stV.conss := hcc
  obtain ⟨hpos, hsz⟩ := ligeritoRun_pos hst0 hdim hlv hrun
  intro ℓ hℓ
  have hlen : stV.conss.size = sh.sched.levels.size := by simpa using congrArg List.length hpos
  have e := List.getElem_of_eq hpos (i := ℓ) (by simp [hlen, hℓ])
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList] at e
  have hsz' : (rounds[lgStart sh extras.size + qOff sh.sched ℓ]!).size = (sh.sched.levels[ℓ]!).q := by
    rw [← hks, ← Conss.getD_default]
    exact hsz ℓ hℓ
  rw [range_cfOf hsz', hc, getElem!_pos stV.conss ℓ (by omega), e, hks, Conss.getD_default]

end FlockVBridge.Family

namespace FlockComplete.Replay

open Flock Transcript

/-- **`MerkleOk` reads its coins only at the queried rounds**: coins that agree with `cf`'s there, from another start
round, pass the same checks. -/
theorem Lg.merkleOk_congr {hs : Hashes V} {ms : MerkleScheme} {sch : Schedule} {cap : Array ByteArray}
    {p : LigeritoProof} {cf cf' : ℕ → ℕ → F128} {k k' : ℕ} (h : Lg.MerkleOk hs ms sch cap p cf k)
    (hcf : ∀ j < sch.levels.size, ∀ t < (sch.levels.getD j default).q, cf (k + Lg.qOff sch j) t = cf' (k' + Lg.qOff sch j) t) :
    Lg.MerkleOk hs ms sch cap p cf' k' := by
  unfold Lg.MerkleOk at h ⊢
  intro j hj
  have e : (Array.range (sch.levels.getD j default).q).map (cf' (k' + Lg.qOff sch j)) =
      (Array.range (sch.levels.getD j default).q).map (cf (k + Lg.qOff sch j)) := by
    refine Array.ext (by simp) fun t h1 h2 => ?_
    simp only [Array.getElem_map, Array.getElem_range]
    exact (hcf j hj t (by simpa using h1)).symm
  rw [e]
  exact h j hj

/-- **A returning honest rep passes every level's Merkle check** at its replies' coins from Ligerito's first round:
the honest rep `repP` on a `commitP` of its table, run from a transcript at round `0` with `l` bodies sent, at hashes
whose digests have the scheme's width (`repP_opens`, `merkleOk_of_opens`; `Run.lgMerkle_of` without the run). -/
theorem merkleOk_of_repP (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tr : Trace) (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) {m : ℕ} (hfast : fast100 m = some sch) (hm : 14 ≤ st.m)
    (hn : ∀ a b v, (st.merkle.opsWith (Hashes.ofSha512 h : Hashes V)).node a b = .ok v → v.size = st.merkle.digestLen)
    (hlf : ∀ row s v, (st.merkle.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row s = .ok v →
      v.size = st.merkle.digestLen)
    {s0 : ℕ → ByteArray}
    (h0 : commitP (Hashes.ofSha512 h : Hashes V) st.merkle (sch.levels.getD 0 default) tbl s0 = .ok cm0)
    (t : Transcript) (ht : t.k = 0) (l : Array Claims.Msg) {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run t).go l =
      (.ok (p, t'), l')) :
    Lg.MerkleOk (Hashes.ofSha512 h : Hashes V) st.merkle sch p.cap p.ligerito (Lg.cfOf resp l.size)
      (3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + (2 + extras.size) + 1) := by
  obtain ⟨hn', hL, -, hq, -⟩ := fast100_ligerito hfast
  have hseam := FlockSoundness.Refine.fast100_level _ _ hfast
  have hcp := Lg.fast100_cap hfast
  obtain ⟨full, hfold⟩ := repP_fold resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  obtain ⟨hcap, hop⟩ := repP_opens resp h _ _ _ tbl cm0 _ salt nonce hm hn' hL hq _ ht l hfold hrun
  have hM := Lg.merkleOk_of_opens hn hlf (fun j hj => hcp _ (Lg.getD_mem _ hj))
    (fun j hj coins => Lg.positions_lt_d (hseam _ (Lg.getD_mem _ hj)) coins) h0 hop
  rw [hcap]
  convert hM using 1
  simp only [Rs.tEnd_k, Lc.tEnd_k, Zc.tEnd_k, ht, Array.size_append]
  simp [abcClaims]

end FlockComplete.Replay

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP)
open FlockVBridge.Family (plain cfOf lgStart)
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockComplete.Replay.Lg (qOff)
open FlockComplete (repP asksS commitP Commit Trace)

/-- **At the deployed hashes the plain scheme is its own closures** (`levels_of_merkleOk`'s `hops`). -/
theorem plain_ops_std : plain.opsWith (Hashes.ofSha512 Sha512.hash : Hashes Flock.V) = MerkleOps.ofScheme plain := rfl

/-- **The plain scheme's digests are 64 bytes at the deployed hashes**, at its nodes and leaves. -/
theorem plain_widths_std :
    (∀ a b v, (plain.opsWith (Hashes.ofSha512 Sha512.hash : Hashes Flock.V)).node a b = .ok v →
      v.size = plain.digestLen) ∧
    ∀ row s v, (plain.opsWith (Hashes.ofSha512 Sha512.hash : Hashes Flock.V)).leaf row s = .ok v →
      v.size = plain.digestLen := by
  rw [plain_ops_std]
  refine ⟨fun a b v e => ?_, fun row s v e => ?_⟩
  · simp only [MerkleOps.ofScheme, pure, Except.pure, Except.ok.injEq] at e
    rw [← e]
    exact Sha512.hash_size _
  · simp only [MerkleOps.ofScheme, pure, Except.pure, Except.ok.injEq] at e
    rw [← e]
    exact Sha512.hash_size _

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128)

/-- **Row D's coin tie** (`levels_of_merkleOk`'s `htie`): at row G, any run of rep `r`'s verifier rows queries level
`ℓ` at the positions of the rep's recorded coins at round `lgStart + qOff sch ℓ` (`rows_pos`). -/
theorem htie {i : ℕ} (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (r : Fin 2) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, (info.lg.conss[ℓ]!).pos = positions ((D.files cls i).sch.levels[ℓ]!)
        ((Array.range ((D.files cls i).sch.levels[ℓ]!).q).map
          (cfOf (D.rep cls acc i r).rounds
            (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch ℓ))) := by
  intro v info kappa hv
  obtain ⟨h22, -⟩ := FlockVBridge.Family.fast100_range hG.fast
  have hdim := (FlockSoundness.Refine.fast100_dims _ _ hG.fast).1
  have hlv : ∀ j < (D.files cls i).sch.levels.size,
      FlockSoundness.Refine.LevelSeam ((D.files cls i).sch.levels.getD j default) :=
    fun j hj => FlockSoundness.Refine.fast100_level _ _ hG.fast _
      (by unfold Array.getD; rw [dite_eq_left hj]; exact Array.getElem_mem hj)
  exact FlockVBridge.Family.rows_pos (D.noEmpty cls acc hti r) (by show 14 ≤ (D.files cls i).st.m; omega) hdim hlv hv

/-- **Row D's `hlev` from `MerkleOk` at coins that agree with the rep's recorded rounds** at the queried rounds:
`levels_of_merkleOk` at `htie`, through `merkleOk_congr`. -/
theorem hlev_of_merkleOk {i : ℕ} (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (r : Fin 2)
    {hs : Hashes Flock.V} (hops : plain.opsWith hs = MerkleOps.ofScheme plain) {cf : ℕ → ℕ → F128} {k : ℕ}
    (h : FlockComplete.Replay.Lg.MerkleOk hs plain (D.files cls i).sch ((D.files cls i).proof r).cap
      ((D.files cls i).proof r).ligerito cf k)
    (hcf : ∀ j < (D.files cls i).sch.levels.size, ∀ t < ((D.files cls i).sch.levels.getD j default).q,
      cf (k + qOff (D.files cls i).sch j) t =
        cfOf (D.rep cls acc i r).rounds (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) t) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () :=
  D.levels_of_merkleOk cls acc r hops hG.fast (FlockComplete.Replay.Lg.merkleOk_congr h hcf) (D.htie cls acc hG hti r)

/-- **Row D's `hlev` at an honest rep**: where rep `r`'s decided proof is a returning honest rep's (`repP` on a
`commitP` of a table, at table `i`'s setup, schedule and extra claims, at hashes whose digests have the scheme's
width), and the replies' coins at the queried rounds are the rep's recorded ones (`merkleOk_of_repP`). -/
theorem hlev_of_repP {i : ℕ} (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (r : Fin 2)
    (h : ByteArray → ByteArray) (hops : plain.opsWith (Hashes.ofSha512 h : Hashes Flock.V) = MerkleOps.ofScheme plain)
    (resp : ℕ → Claims.Msg) (tr : Trace) (tbl : Array F128) (cm0 : Commit) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64)
    (hn : ∀ a b v, ((D.files cls i).st.merkle.opsWith (Hashes.ofSha512 h : Hashes Flock.V)).node a b = .ok v →
      v.size = (D.files cls i).st.merkle.digestLen)
    (hlf : ∀ row s v, ((D.files cls i).st.merkle.opsWith (Hashes.ofSha512 h : Hashes Flock.V)).leaf row s = .ok v →
      v.size = (D.files cls i).st.merkle.digestLen)
    {s0 : ℕ → ByteArray}
    (h0 : commitP (Hashes.ofSha512 h : Hashes Flock.V) (D.files cls i).st.merkle
      ((D.files cls i).sch.levels.getD 0 default) tbl s0 = .ok cm0)
    (t : Transcript) (ht : t.k = 0) (l : Array Claims.Msg) {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) (D.files cls i).st
      (D.files cls i).sch tr tbl cm0 (D.files cls i).extras salt nonce).run t).go l =
        (.ok ((D.files cls i).proof r, t'), l'))
    (hcoin : ∀ j < (D.files cls i).sch.levels.size, ∀ t < ((D.files cls i).sch.levels.getD j default).q,
      FlockComplete.Replay.Lg.cfOf resp l.size
          (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) t =
        cfOf (D.rep cls acc i r).rounds (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) t) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () := by
  obtain ⟨h22, -⟩ := FlockVBridge.Family.fast100_range hG.fast
  have hM := FlockComplete.Replay.merkleOk_of_repP resp h _ _ tr tbl cm0 _ salt nonce hG.fast (by omega) hn hlf h0
    t ht l hrun
  have hN : 3 + ((D.files cls i).st.m - 6) + 3 + ((D.files cls i).st.fold.kLog - 6) +
      (2 + (D.files cls i).extras.size) + 1 = lgStart (D.sh cls i) (D.files cls i).extras.size := by
    unfold lgStart
    rw [hG.foldKLog]
    rfl
  rw [hN, hG.merkle] at hM
  exact D.hlev_of_merkleOk cls acc hG hti r hops hM hcoin

/-- **Row D's `hlev` at the deployed hashes**: `hlev_of_repP` at `Hashes.ofSha512 Sha512.hash`, where the plain
scheme is its own closures (`plain_ops_std`) and its digests are 64 bytes (`plain_widths_std`). What is left are the
honest rep's equations: its decided proof is `repP`'s output (`hrun`) on a `commitP` of its table (`h0`), and its
replies' coins at the queried rounds are its recorded ones (`hcoin`). -/
theorem hlev_deployed {i : ℕ} (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (r : Fin 2)
    (resp : ℕ → Claims.Msg) (tr : Trace) (tbl : Array F128) (cm0 : Commit) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) {s0 : ℕ → ByteArray}
    (h0 : commitP (Hashes.ofSha512 Sha512.hash : Hashes Flock.V) (D.files cls i).st.merkle
      ((D.files cls i).sch.levels.getD 0 default) tbl s0 = .ok cm0)
    (t : Transcript) (ht : t.k = 0) (l : Array Claims.Msg) {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 Sha512.hash) (D.files cls i).st
      (D.files cls i).sch tr tbl cm0 (D.files cls i).extras salt nonce).run t).go l =
        (.ok ((D.files cls i).proof r, t'), l'))
    (hcoin : ∀ j < (D.files cls i).sch.levels.size, ∀ t < ((D.files cls i).sch.levels.getD j default).q,
      FlockComplete.Replay.Lg.cfOf resp l.size
          (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) t =
        cfOf (D.rep cls acc i r).rounds (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) t) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () :=
  D.hlev_of_repP cls acc hG hti r Sha512.hash plain_ops_std resp tr tbl cm0 salt nonce
    (by rw [hG.merkle]; exact plain_widths_std.1) (by rw [hG.merkle]; exact plain_widths_std.2) h0 t ht l hrun hcoin

end DecAt

end RecDeploy.Rows
