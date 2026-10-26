import Proofs.Flock.Claims.Complete.Zk.Level0Lanes
import Proofs.Flock.Claims.Complete.Zk.ProverOk
import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsCarries

/-!
# The masked Ligerito prover's proof lengths

`ligeritoPZk_shape`: a masked Ligerito prover (`ligeritoPZk`) that returns, at a schedule the walk can follow
(`LigSched`), on a level-0 commitment of rows `2^lanes + EXTRA_LANES` wide and a table of `2^n` entries, has every
length `Zk.decode` checks of its proof (`ZkTables.Level0.LgShaped`) and `t_pad` padding coefficients.

* **Level 0's masked commitment by its shape** (`CommitOkW`, `commitRowsP_shapeW`, `openP_okW`): `CommitOk` and
  `openP_ok` at a row width `w`; `commitPZk`'s rows are `2^lanes + EXTRA_LANES` wide (`commitPZk_shape`).
* **The `--zk` schedule** is one the walk follows (`zkSchedule_lig`): fast100's dimensions and levels above 0
  (`zk_schedule_fast100`, `fast100_lig`), every level's seam (`zk_schedule_seams`), level 0's cap within its depth
  (`zkSchedule_cap0`).
* **The lane phase** is walked on the run (`lforIn_invS`), keeping its lengths (`meas`) and, at its last coin, the
  padding's `t_pad` coefficients. **The rest** is plain's code from the split table on, walked on `Total.H` as
  `ligeritoP_shape` walks it and read off the run by determinism (`H_run`).
-/

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg JsonRT FlockSoundness.Discharge

/-! ## Level 0's masked commitment, by its shape -/

/-- `CommitOk` at a row width `w`. -/
def CommitOkW (hl sb w : ℕ) (lv : Level) (cm : Commit) : Prop :=
  cm.rows.size = 2 ^ lv.d ∧ (∀ r ∈ cm.rows, r.size = w) ∧ cm.salts.size = 2 ^ lv.d ∧
    (∀ x ∈ cm.salts, x.size = sb) ∧
    (∀ h < lv.d - lv.cap, ∃ hh : h < cm.layers.size, cm.layers[h].size = 2 ^ (lv.d - h) ∧
      ∀ x ∈ cm.layers[h], x.size = hl) ∧
    cm.cap.size = 2 ^ lv.cap ∧ ∀ x ∈ cm.cap, x.size = hl

/-- **A commitment of `2^d` rows `w` wide, by its shape** (`commitP_shape` at given rows). -/
theorem commitRowsP_shapeW {ms : MerkleScheme} {hs : Hashes V} {hl w : ℕ}
    (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = hl)
    (hlf : ∀ row salt v, (ms.opsWith hs).leaf row salt = .ok v → v.size = hl) {lv : Level} (hcap : lv.cap ≤ lv.d)
    {rows : Array (Array F128)} (hrs : rows.size = 2 ^ lv.d) (hrw : ∀ r ∈ rows, r.size = w)
    {salt : ℕ → ByteArray} {cm : Commit} (h : commitRowsP hs ms lv rows salt = .ok cm) :
    CommitOkW hl ms.saltBytes w lv cm := by
  unfold commitRowsP at h
  obtain ⟨leaves, hlv, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨hls, hld⟩ := mapM_V (fun _ _ e => hlf _ _ _ e) hlv
  obtain ⟨r, hr, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  have hI := forIn_upto_V (lv.d - lv.cap) (#[leaves], leaves)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => p.1.size = i + 1 ∧
      (∀ j ≤ i, ∃ hj : j < p.1.size, p.1[j].size = 2 ^ (lv.d - j) ∧ ∀ x ∈ p.1[j], x.size = hl) ∧
      p.2.size = 2 ^ (lv.d - i) ∧ ∀ x ∈ p.2, x.size = hl)
    ⟨rfl, fun j hj => by obtain rfl : j = 0 := by omega
                         exact ⟨by simp, by simpa using hls, hld⟩, by simpa using hls, hld⟩
    (fun i hi p r' hp e => by
      obtain ⟨cur', hc, e⟩ := FlockSoundness.Refine.ebind_ok e
      cases e
      obtain ⟨hcs, hcd⟩ := mapM_V (fun _ _ e => hn _ _ _ e) hc
      obtain ⟨h1, h2, h3, h4⟩ := hp
      have hsz : cur'.size = 2 ^ (lv.d - (i + 1)) := by
        rw [hcs, Array.size_range, h3, show lv.d - i = lv.d - (i + 1) + 1 by omega, Nat.pow_succ,
          Nat.mul_div_cancel _ (by norm_num)]
      refine ⟨_, rfl, by simp [h1], fun j hj => ?_, hsz, hcd⟩
      rcases Nat.lt_or_eq_of_le hj with hj | rfl
      · obtain ⟨hj', e1, e2⟩ := h2 j (by omega)
        refine ⟨by simp; omega, ?_⟩
        simp only [Array.getElem_push, hj', ↓reduceDIte]
        exact ⟨e1, e2⟩
      · refine ⟨by simp [h1], ?_⟩
        simp only [Array.getElem_push, h1, lt_self_iff_false, ↓reduceDIte]
        exact ⟨hsz, hcd⟩) hr
  obtain ⟨-, hL, hC, hCd⟩ := hI
  refine ⟨hrs, hrw, by simp, fun x hx => ?_, fun h hh => hL h hh.le, by rw [hC]; congr 1; omega, hCd⟩
  simp only [Array.mem_map, Array.mem_range] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact fitBytes_size _ _

theorem encodeRowsZk_shape (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) :
    (encodeRowsZk lv tPad g extra lanePad).size = 2 ^ lv.d ∧
      ∀ r ∈ encodeRowsZk lv tPad g extra lanePad, r.size = 2 ^ lv.lanes + Zk.EXTRA_LANES := by
  refine ⟨by simp [encodeRowsZk], fun r hr => ?_⟩
  simp only [encodeRowsZk, Array.mem_map, Array.mem_range] at hr
  obtain ⟨_, -, rfl⟩ := hr
  simp

/-- **Level 0's masked commitment, by its shape**: rows `2^lanes + EXTRA_LANES` wide. -/
theorem commitPZk_shape {ms : MerkleScheme} {hs : Hashes V} {hl : ℕ}
    (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = hl)
    (hlf : ∀ row salt v, (ms.opsWith hs).leaf row salt = .ok v → v.size = hl) {lv : Level} (hcap : lv.cap ≤ lv.d)
    {tPad : ℕ} {g : Array F128} {extra lanePad : ℕ → ℕ → F128} {salt : ℕ → ByteArray} {cm : Commit}
    (h : commitPZk hs ms lv tPad g extra lanePad salt = .ok cm) :
    CommitOkW hl ms.saltBytes (2 ^ lv.lanes + Zk.EXTRA_LANES) lv cm :=
  commitRowsP_shapeW hn hlf hcap (encodeRowsZk_shape lv tPad g extra lanePad).1
    (encodeRowsZk_shape lv tPad g extra lanePad).2 h

/-- **An opening of a commitment `w` wide, by its shape** (`openP_ok` at a row width). -/
theorem openP_okW {sh : Shape} {ms : MerkleScheme} {lv : Level} {w : ℕ} {cm : Commit} {pos : Array ℕ}
    (hc : CommitOkW sh.hashLen ms.saltBytes w lv cm) (hq : pos.size = lv.q) (hp : ∀ p ∈ pos, p < 2 ^ lv.d)
    (hsl : (sh.saltLen = none ∧ ms.saltBytes = 0) ∨ sh.saltLen = some ms.saltBytes) :
    ZkView.LevelOk sh lv w (openP ms cm lv pos) := by
  obtain ⟨hr, hrw, hs, hsw, hL, -, -⟩ := hc
  have hsib := siblings_facts hL pos.toList #[] (fun p h => hp p (Array.mem_toList_iff.1 h)) (by simp)
  rw [Array.foldl_toList] at hsib
  refine ⟨by simp [openP, hq], ?_, ?_, hsib.2, ?_⟩
  · intro r h
    simp only [openP, Array.mem_map] at h
    obtain ⟨p, hpm, rfl⟩ := h
    exact hrw _ (getElem!_mem (by rw [hr]; exact hp p hpm))
  · have e := hsib.1
    rw [Array.length_toList, Array.size_empty, Nat.zero_add] at e
    rw [← hq]
    exact e
  · unfold ZkView.SaltsOk
    rcases hsl with ⟨e, e0⟩ | e
    · rw [e]
      simp [openP, e0]
    · rw [e]
      by_cases e0 : ms.saltBytes = 0
      · simp [openP, e0]
      · obtain ⟨s, hs'⟩ : ∃ s, ms.saltBytes = s + 1 := ⟨ms.saltBytes - 1, by omega⟩
        simp only [openP, hs', Nat.add_one_ne_zero, ↓reduceIte, Array.size_map, hq, Array.mem_map, true_and]
        rintro _ ⟨p, hpm, rfl⟩
        rw [← hs']
        exact hsw _ (getElem!_mem (by rw [hs]; exact hp p hpm))

/-! ## The `--zk` schedule -/

/-- **The walk follows the `--zk` schedule**: fast100's dimensions and levels above 0, every level's seam, and level
0's cap within its depth. -/
theorem zkSchedule_lig {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    LigSched v.1 := by
  obtain ⟨-, -, hm, hk0, -, ⟨hnm, -, -, -, hl4, -, -⟩, hseam, -, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams m v.1 v.2 h
  obtain ⟨s₀, hf, ek, ee, es, el⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_fast100 h
  obtain ⟨⟨-, h2', hdim, hcap, -, -⟩, hm₀⟩ := fast100_lig m s₀ hf
  have hn₀ := (FlockSoundness.Refine.fast100_dims m s₀ hf).1.1
  refine ⟨hk0, by omega, by omega, fun i hi => ?_, hseam, hl4⟩
  rcases Nat.eq_zero_or_pos i with rfl | hi0
  · exact (zkSchedule_cap0 h1 h2 h).1
  · rw [el i hi0]
    exact hcap i (by omega)

/-! ## The walk -/

/-- **A `Total.H` walk, read off a run**: a computation that returns from every state `pre` holds at, with `post`,
has `post` at whatever it returned. -/
theorem H_run {α : Type} {y : LP PS α} {pre : LigState → Prop} {post : α → LigState → Prop}
    (hy : totT.H y pre post) {st : LigState} (hp : pre st) {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    {st' : LigState} (h : ((y.run st).run t).go l = (.ok ((a, st'), t'), l')) : post a st' := by
  obtain ⟨p, l₁, e, hq⟩ := hy st hp t l
  rw [show ((y.run st).run t) = y st t from rfl, e] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨rfl, -⟩ := h
  exact hq

open ZkTables.Level0 in
/-- **The masked Ligerito prover's proof has its lengths**: a masked Ligerito prover that returns, at a schedule the
walk follows, on a level-0 commitment of rows `2^lanes + EXTRA_LANES` wide and a table of `2^n` entries, has
`LgShaped`'s lengths and `t_pad` padding coefficients. -/
theorem ligeritoPZk_shape (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) {sh : Shape} {ms : MerkleScheme}
    (hn : ∀ l r v, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).node l r = .ok v → v.size = sh.hashLen)
    (hlf : ∀ row salt v, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok v → v.size = sh.hashLen)
    (hsl : (sh.saltLen = none ∧ ms.saltBytes = 0) ∨ sh.saltLen = some ms.saltBytes) {sch : Schedule}
    (hsh : sh.sched = sch) (hs : LigSched sch) (T : F128) {cm0 : Commit}
    (hc0 : CommitOkW sh.hashLen ms.saltBytes (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES)
      (sch.levels.getD 0 default) cm0) {tb : Array F128} (ht : tb.size = 2 ^ sch.n) (b : Array F128)
    (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {t t' : Transcript} {l l' : Array Claims.Msg}
    {P : LigeritoProof} {zl : Zk.Level0Msgs} {rc : Zk.L0Rec}
    (hrun : ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b lay hv rep tPad extra lanePad salt
      nonce).run t).go l = (.ok ((P, zl, rc), t'), l')) :
    LgShaped sh P ∧ zl.ybar.size = tPad := by
  obtain ⟨hk, hR, hdim, hcap, hseam, hlanes⟩ := hs
  have hn0 : sch.n ≠ 0 := by omega
  have hc : ∀ i < sch.levels.size, ∀ g sl,
      totS.T (commitP (Hashes.ofSha512 h : Hashes PS) ms (sch.levels.getD i default) g sl)
        fun cm => CommitOk sh.hashLen ms.saltBytes (sch.levels.getD i default) cm := fun i hi g sl =>
    commitP_shapeS h ms _ g sl hn hlf (hcap i hi)
  have hop : ∀ i < sch.levels.size, ∀ cm coins, CommitOk sh.hashLen ms.saltBytes (sch.levels.getD i default) cm →
      ZkView.LevelOk sh (sch.levels.getD i default) (2 ^ (sch.levels.getD i default).lanes)
        (openP ms cm (sch.levels.getD i default) (positions (sch.levels.getD i default) coins)) :=
    fun i hi cm coins hcm =>
    openP_ok hcm (positions_facts _ (hseam i hi) coins).1 (positions_facts _ (hseam i hi) coins).2 hsl
  have hopZ : ∀ coins, ZkView.LevelOk sh (sch.levels.getD 0 default)
      (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES)
      (openP ms cm0 (sch.levels.getD 0 default) (positions (sch.levels.getD 0 default) coins)) := fun coins =>
    openP_okW hc0 (positions_facts _ (hseam 0 (by omega)) coins).1 (positions_facts _ (hseam 0 (by omega)) coins).2
      hsl
  have hL : 2 ≤ sch.levels.size := by omega
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0, zl', initCh, rho, b0', y'⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨rfl, hP₂⟩ := Prod.mk.inj hP
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hP₂
  obtain ⟨_, _, -, h1⟩ := lsampleVecP_bind resp hn0 h1
  have h1 := lmodify_bind h1
  obtain ⟨_, _, h1⟩ := lliftTM_bind h1
  obtain ⟨_, h1⟩ := lpowB_bind nonce h1
  obtain ⟨_, _, -, h1⟩ := lsampleP_bind resp h1
  have h1 := lmodify_bind h1
  have h1 := lget_bind h1
  obtain ⟨_, h1⟩ := lsendMsg_bind _ h1
  obtain ⟨sL, stC, tC, lC, hLoop, h1⟩ := lp_bind_ok h1
  have hI := lforIn_invS _ (fun j (s : Array F256 × (F256 × F256) × Zk.Level0Msgs) st (_ : Array Claims.Msg) =>
      meas st = ⟨j + 1, 0, 0, 1, 1, if j < sch.k0 then 2 ^ (sch.n - j) else 2 ^ (sch.n - sch.k0 + 1), true⟩ ∧
        (j = sch.k0 → s.2.2.ybar.size = tPad)) sch.k0 ?step ?init hLoop
  case init =>
    refine ⟨?_, fun h => absurd h (by omega)⟩
    simp [meas, JsonRT.embedF_size, ht, show 0 < sch.k0 by omega, Array.all_push, Array.all_empty]
  case step =>
    rintro j hj s st t₀ l₀ r st' t₂ l₂ ⟨hm, -⟩ hb
    simp only [meas, Ms.mk.injEq, hj, ↓reduceIte] at hm
    obtain ⟨hm1, hm2, hm3, hm4, hm5, hm6, hm7⟩ := hm
    obtain ⟨_, _, -, hb⟩ := lsampleKP_bind resp hb
    have hb := lmodify_bind hb
    by_cases hj1 : j + 1 = sch.k0
    · rw [ite_eq_left (by simp [hj1])] at hb
      have hb := lget_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨_, _, -, hb⟩ := lsampleKP_bind resp hb
      obtain ⟨_, _, -, hb⟩ := lsampleKP_bind resp hb
      split at hb
      · obtain ⟨_, _, _, _, h0, _⟩ := lp_bind_ok hb
        cases h0
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      obtain ⟨_, _, hb⟩ := lliftTM_bind hb
      have hb := lmodify_bind hb
      have hb := lget_bind hb
      obtain ⟨_, hb⟩ := lsendMsg_bind _ hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      refine ⟨_, rfl, ?_, fun _ => by simp⟩
      simp only [meas]
      refine Ms.eq' hm7 ?_ ?_ ?_ ?_ ?_ ?_ <;>
        simp only [Array.size_push, JsonRT.embedF_size, JsonRT.splitK_size, Array.size_map, Array.size_range,
          JsonRT.foldK_size, hm1, hm2, hm3, hm4, hm5, hm6, show ¬ j + 1 < sch.k0 by omega, ↓reduceIte]
      rw [pow_half (by omega), pow_dbl (by omega)]
      congr 1
      omega
    · rw [ite_eq_right (by simp [hj1])] at hb
      have hb := lget_bind hb
      obtain ⟨_, hb⟩ := lsendMsg_bind _ hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      refine ⟨_, rfl, ?_, fun h => absurd h hj1⟩
      simp only [meas]
      refine Ms.eq' hm7 ?_ ?_ ?_ ?_ ?_ ?_ <;>
        simp only [Array.size_push, JsonRT.foldK_size, hm1, hm2, hm3, hm4, hm5, hm6, show j + 1 < sch.k0 by omega,
          ↓reduceIte]
      rw [pow_half (by omega), Nat.sub_sub]
  obtain ⟨hmC, hyb⟩ := hI
  have hpre : meas stC = ⟨0 + 1 + sch.k0, 0, 0, 0 + 1, 0 + 1, 2 ^ (sch.n - sch.k0 + 1), true⟩ := by
    rw [hmC]
    simp only [lt_self_iff_false, ↓reduceIte]
    exact Ms.eq (by omega) rfl rfl rfl rfl rfl
  have hsuf := H_run (show totT.H _
      (fun st => meas st = ⟨0 + 1 + sch.k0, 0, 0, 0 + 1, 0 + 1, 2 ^ (sch.n - sch.k0 + 1), true⟩)
      (fun (r : Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof × Zk.Level0Msgs ×
          Array F256 × (F256 × F256) × F128 × F128) st =>
        ZkView.LevelOk sh (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES)
          r.2.2.2.1 ∧
        r.1.size = sch.levels.size - 1 ∧
        (∀ i (h : i < r.1.size), ZkView.CapOk sh (sch.levels.getD (i + 1) default) r.1[i]) ∧
        r.2.1.size = sch.levels.size - 1 - 1 ∧
        (∀ i (h : i < r.2.1.size), ZkView.LevelOk sh (sch.levels.getD (i + 1) default) 16 r.2.1[i]) ∧
        r.2.2.1.yr.size = 2 ^ (sch.ext + 1) ∧
        ZkView.LevelOk sh (sch.levels.getD (sch.levels.size - 1) default) 16
          ⟨r.2.2.1.rows, r.2.2.1.siblings, r.2.2.1.salts⟩ ∧
        r.2.2.2.2.1 = sL.2.2 ∧
        ∃ tb, meas st = ⟨sch.k0 + 8 + 7 * (sch.levels.size - 1 - 1), sch.levels.size - 1 + 1,
          sch.levels.size - 1 + 1, 3 * (sch.levels.size - 1) + 2, 1 + 2 * (sch.levels.size - 1), tb, true⟩) from by
    refine Total.H.bind Total.H.get fun dl => ?_
    refine Total.H.bind (H_liftTQ (Total.lift_stQ _ (hc 1 (by omega) _ (salt 1)))) fun cm1 => ?_
    refine Total.H.fact fun hcm1 => ?_
    refine Total.H.pre
      (pre := fun s => meas s = ⟨0 + 1 + sch.k0, 0, 0, 0 + 1, 0 + 1, 2 ^ (sch.n - sch.k0 + 1), true⟩) ?_
      fun s hs => hs.2
    refine Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
    refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
    refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
    refine Total.H.bind (queryP_H resp nonce ms (sch.levels.getD 0 default) cm0 _) fun x0 => ?_
    refine Total.H.fact fun ⟨c0, hx0⟩ => ?_
    have hop0 : ZkView.LevelOk sh (sch.levels.getD 0 default)
        (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) x0.1 := hx0 ▸ hopZ c0
    refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
    refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
    refine Total.H.bind (H_liftT (sampleP_tot resp)) fun _ => ?_
    refine Total.H.bind (H_modify _ id (by intro s; rfl) _) fun _ => ?_
    refine Total.H.bind (Total.H.pre (Total.H.forIn_upto (sch.levels.size - 1) _
      (fun i (q : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof) s =>
        (i < sch.levels.size - 1 ∧ q.1 = sch.ext + 3 * (sch.levels.size - 1 - i) + 1 ∧
          meas s = ⟨sch.k0 + 4 + 7 * i, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i, 2 ^ q.1, true⟩ ∧
          CommitOk sh.hashLen ms.saltBytes (sch.levels.getD (i + 1) default) q.2.1 ∧ q.2.2.1.size = i + 1 ∧
          (∀ j (h : j < q.2.2.1.size), ZkView.CapOk sh (sch.levels.getD (j + 1) default) q.2.2.1[j]) ∧
          q.2.2.2.1.size = i ∧
          (∀ j (h : j < q.2.2.2.1.size), ZkView.LevelOk sh (sch.levels.getD (j + 1) default) 16 q.2.2.2.1[j])) ∨
        (i = sch.levels.size - 1 ∧ q.2.2.1.size = sch.levels.size - 1 ∧
          (∀ j (h : j < q.2.2.1.size), ZkView.CapOk sh (sch.levels.getD (j + 1) default) q.2.2.1[j]) ∧
          q.2.2.2.1.size = sch.levels.size - 1 - 1 ∧
          (∀ j (h : j < q.2.2.2.1.size), ZkView.LevelOk sh (sch.levels.getD (j + 1) default) 16 q.2.2.2.1[j]) ∧
          q.2.2.2.2.yr.size = 2 ^ (sch.ext + 1) ∧
          ZkView.LevelOk sh (sch.levels.getD (sch.levels.size - 1) default) 16
            ⟨q.2.2.2.2.rows, q.2.2.2.2.siblings, q.2.2.2.2.salts⟩ ∧
          ∃ tb, meas s = ⟨sch.k0 + 8 + 7 * (sch.levels.size - 1 - 1), sch.levels.size - 1 + 1,
            sch.levels.size - 1 + 1, 3 * (sch.levels.size - 1) + 2, 1 + 2 * (sch.levels.size - 1), tb, true⟩)) ?_)
        ?_)
      fun q => ?_
    · intro i hi q
      refine Total.H.intro fun s0 hI => ?_
      rcases hI with ⟨-, hsd, -, hcm, hcs, hca, hrs, hrp⟩ | ⟨hiR, -⟩
      swap
      · omega
      refine Total.H.pre (pre := fun s =>
          meas s = ⟨sch.k0 + 4 + 7 * i, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i, 2 ^ q.1, true⟩) ?_ fun s hs => by
        rcases hs with h | h
        · exact h.2.2.1
        · exact absurd h.1 (by omega)
      have hsd4 : 4 ≤ q.1 := by omega
      refine Total.H.bind (Total.H.pre (Total.H.forIn_upto 4 _
        (fun j (f : Array F256) s => (j = 4 → f.size = 2 ^ (q.1 - 4)) ∧
          meas s = ⟨sch.k0 + 4 + 7 * i + j, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i,
            if j < 4 then 2 ^ (q.1 - j) else 2 ^ (q.1 - 3), true⟩) ?_) ?_) fun f => ?_
      · intro j hj g
        refine Total.H.fact_l fun _ => ?_
        split
        · next hj3 =>
          have e3 : j = 3 := by simpa using hj3
          subst e3
          refine Total.H.bind (H_liftT (sampleKP_tot resp)) fun r => ?_
          refine Total.H.bind (H_modify _ (fun v => { v with tab := v.tab / 2 })
            (fun s => by simp [meas, JsonRT.foldK_size]) _) fun _ => ?_
          refine Total.H.bind Total.H.get fun dl => ?_
          refine Total.H.intro fun s hs => ?_
          have hdl : dl.tab.size = 2 ^ (q.1 - 4) := by
            obtain ⟨rfl, e⟩ := hs
            have := congrArg Ms.tab e
            simp only [meas] at this
            rw [this, pow_half (by omega), Nat.sub_sub]
          refine Total.H.pre (Total.H.bind (H_modify _ (fun v => { v with tab := 2 * v.tab })
            (fun s => by simp [meas, JsonRT.embedF_size, JsonRT.splitK_size]) _) fun _ => ?_) fun s hs => hs.2
          refine Total.H.get_bind fun st => ?_
          refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
          refine Total.H.pure _ fun s hs => ⟨_, rfl, fun _ => hdl, ?_⟩
          rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only)
          simp only [↓reduceIte, lt_self_iff_false]
          rw [pow_half (by omega), pow_dbl (by omega)]
        · next hj3 =>
          have hj3' : j + 1 ≠ 4 := by simpa using hj3
          refine Total.H.bind (challengeP_H resp false _) fun _ => ?_
          refine Total.H.pure _ fun s hs => ⟨_, rfl, fun h => absurd h hj3', ?_⟩
          rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only) <;> (try omega)
          simp only [show j + 1 < 4 by omega, ↓reduceIte, Bool.false_eq_true]
          rw [pow_half (by omega), Nat.sub_sub]
      · intro s hs
        exact ⟨fun h => absurd h (by omega), by rw [hs]; simp⟩
      refine Total.H.fact_l fun hf => ?_
      have hf := hf rfl
      refine Total.H.pre (pre := fun s =>
          meas s = ⟨sch.k0 + 8 + 7 * i, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i, 2 ^ (q.1 - 3), true⟩) ?_ fun s hs => by
        rw [hs]
        simp only [Ms.mk.injEq, lt_self_iff_false, ↓reduceIte, and_true]
        omega
      split
      · next hlast =>
        have hiR : i + 1 ≠ sch.levels.size - 1 := by simpa using hlast
        refine Total.H.bind (H_liftTQ (Total.lift_stQ _ (hc (i + 2) (by omega) _ (salt (i + 2))))) fun cmN => ?_
        refine Total.H.fact fun hcmN => ?_
        refine Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
        refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
        refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
        refine Total.H.bind (queryP_H resp nonce ms (sch.levels.getD (i + 1) default) _ _) fun x => ?_
        refine Total.H.fact fun ⟨cs, hx⟩ => ?_
        have hopx : ZkView.LevelOk sh (sch.levels.getD (i + 1) default) 16 x.1 := by
          have e : (sch.levels.getD (i + 1) default).lanes = 4 := hlanes (i + 1) (by omega) (by omega)
          have := hop (i + 1) (by omega) q.2.1 cs hcm
          rw [e] at this
          rw [hx]
          exact this
        refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
        refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
        refine Total.H.bind (H_liftT (sampleP_tot resp)) fun _ => ?_
        refine Total.H.bind (H_modify _ id (by intro s; rfl) _) fun _ => ?_
        refine Total.H.pure _ fun s hs => ⟨_, rfl, Or.inl ⟨by omega, by omega, ?_, hcmN, by simp [hcs], ?_,
          by simp [hrs], ?_⟩⟩
        · rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only) <;> (try omega)
          congr 1
          omega
        · intro j hj
          rw [Array.getElem_push]
          split
          · exact hca j ‹_›
          · obtain rfl : j = i + 1 := by simp at hj; omega
            exact ⟨hcmN.2.2.2.2.2.1, hcmN.2.2.2.2.2.2⟩
        · intro j hj
          rw [Array.getElem_push]
          split
          · exact hrp j ‹_›
          · obtain rfl : j = i := by simp at hj; omega
            exact hopx
      · next hlast =>
        have hiR : i + 1 = sch.levels.size - 1 := by simpa using hlast
        refine Total.H.bind (Total.H.forIn_array_keep _ _ fun _ _ =>
          Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => Total.H.pure _ fun _ h => h) fun _ => ?_
        refine Total.H.bind (queryP_H resp nonce ms (sch.levels.getD (sch.levels.size - 1) default) _ _)
          fun x => ?_
        refine Total.H.fact fun ⟨cs, hx⟩ => ?_
        have hopx : ZkView.LevelOk sh (sch.levels.getD (sch.levels.size - 1) default) 16 x.1 := by
          have e : (sch.levels.getD (sch.levels.size - 1) default).lanes = 4 :=
            hlanes (sch.levels.size - 1) (by omega) (by omega)
          have := hop (sch.levels.size - 1) (by omega) q.2.1 cs (by rw [← hiR]; exact hcm)
          rw [e] at this
          rw [hx]
          exact this
        refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
        refine Total.H.bind (H_liftT (sampleP_tot resp)) fun _ => ?_
        refine Total.H.pure _ fun s hs => ⟨_, rfl, Or.inr ⟨by omega, by dsimp only; omega, hca,
          by dsimp only; omega, hrp, ?_, hopx,
          2 ^ (q.1 - 3), ?_⟩⟩
        · show (splitK f).size = _
          rw [JsonRT.splitK_size, hf, ← Nat.pow_succ']
          congr 1
          omega
        · rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only) <;> omega
    · intro s hs
      refine Or.inl ⟨by omega, by dsimp only; omega, ?_, hcm1, rfl, ?_, rfl,
        fun j hj => absurd hj (by simp)⟩
      · rw [hs]
        refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only)
        omega
      · intro j hj
        obtain rfl : j = 0 := by simpa using hj
        exact ⟨hcm1.2.2.2.2.2.1, hcm1.2.2.2.2.2.2⟩
    · refine Total.H.pure _ fun s hs => ?_
      rcases hs with ⟨hlt, -⟩ | ⟨-, e1, e2, e3, e4, e5, e6, e7⟩
      · omega
      · exact ⟨hop0, e1, e2, e3, e4, e5, e6, rfl, e7⟩) hpre h1
  obtain ⟨h0, e1, e2, e3, e4, e5, e6, hz, tbS, h7⟩ := hsuf
  have h0s := JsonRT.zeros_of_meas (congrArg Ms.z h7)
  simp only [meas, Ms.mk.injEq] at h7
  obtain ⟨m1, m2, m3, m4, m5, -⟩ := h7
  have hz : zl' = sL.2.2 := hz
  subst hsh
  refine ⟨⟨⟨hc0.2.2.2.2.2.1, hc0.2.2.2.2.2.2⟩, h0, e1, e2, e3, e4, e5, e6, m1, m2, m5, m4, m3, h0s⟩, ?_⟩
  rw [hz]
  exact hyb rfl

end FlockComplete.L0
