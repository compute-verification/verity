module

public import Proofs.Flock.SessionLoop.LiveOwn
public import Proofs.Flock.SessionLoop.RoundOff
public import Proofs.Flock.SessionLoop.RoundPiop
public import Proofs.Flock.SessionLoop.InnerSlot
public import Proofs.Flock.SessionLoop.SwCausalExe

/-!
# An honest worker's served values meet `RowsOK`

`LeanRelease.liveRoundEqOwn_vsOf` takes one condition on the worker beyond the statement and the loop's inputs:
`Loop.RowsOK (Loop.vsServed bx) x w`, that the rows hold at the values the algebraic-phase worker `bx` serves. This
file derives it for a worker that serves honestly, one whose every freed word is the table's own value
(`Loop.TakesVals` at `honestVals`):

* `Loop.freedAt_round`: a round that frees pad `i` of rep stream `2j + r` after `repBase + jj` openings is PIOP round
  `jj`, and `i` is at a place of the absorbed order below the values of PIOP rounds `0` to `jj` (`piopLen`): the binding
  round frees none (`zcBind_frame`), PIOP round `jj` frees the values it carries (`piop_frame`, `roundFree_vals`), and
  past the PIOP every place is below them (`piopLen_full`).
* `Loop.vals_freed`: the table's own value at such a pad is one at two coin vectors agreeing below `repBase + jj`; the
  honest sources' masked values there agree (`repOk`), so the table's values do (`vals_of_masked`).
* `Loop.vsOfWord_of_takes`: where the worker takes the honest values, `vsOfWord` is the honest values: the word a round
  frees is the table's value along the coins the round opened, which is its value along the full coins
  (`vals_freed`), and where no round frees a pad `vsOfWord` is the table's own value.
* **`Loop.rowsOK_vsServed_of_takes`**: `RowsOK (vsServed bx) x w` at such a worker and a witness under the relation:
  `vsServed bx` is the honest values at every index, `final_c`'s the c_eval of the round-1c values as the honest values'
  is (`finalCFixed_honestVals`), and the rows hold there (`rowsOK_honestVals`).
* **`LeanRelease.liveRoundEqOwn_vsOf_takes`**: `liveRoundEqOwn_vsOf` with its `RowsOK` premise discharged so.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Claims.ZKServer

open _root_.Flock (V F128 Stmt SessionSpec)
open Claims Claims.ZKMal
open FlockSoundness.Discharge
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock.Firewall.Finish (Tab)
open _root_.Flock.Firewall.Frame (Cfg Masker absorbed nLig roundFree)

namespace Loop

/-! A place below the values of PIOP rounds `0` to `jj` is in a round no later than `jj`: round 2 for round 1's values,
residual round `3 + a` for its pair `a`, `final_a`/`final_b`'s round, lincheck round `5 + (m - 6) + a`, `z_partial`'s. -/

theorem round_r1 {m k jj P : ℕ} (h : P < piopLen m k (jj + 1)) : 2 ≤ jj := by
  unfold piopLen at h
  split_ifs at h <;> omega

theorem round_ml {m k jj P : ℕ} (h : P < piopLen m k (jj + 1)) (h1 : 128 ≤ P) (h2 : P < 128 + 2 * (m - 6)) :
    3 + (P - 128) / 2 ≤ jj := by
  unfold piopLen at h
  split_ifs at h <;> omega

theorem round_f {m k jj P : ℕ} (h : P < piopLen m k (jj + 1)) (h1 : 128 + 2 * (m - 6) ≤ P) : 3 + (m - 6) ≤ jj := by
  unfold piopLen at h
  split_ifs at h <;> omega

theorem round_lc {m k jj P : ℕ} (h : P < piopLen m k (jj + 1)) (h1 : 130 + 2 * (m - 6) ≤ P)
    (h2 : P < 130 + 2 * (m - 6) + 2 * (k - 6)) : 5 + (m - 6) + (P - 130 - 2 * (m - 6)) / 2 ≤ jj := by
  unfold piopLen at h
  split_ifs at h <;> omega

theorem round_zp {m k jj P : ℕ} (h : P < piopLen m k (jj + 1)) (h1 : 130 + 2 * (m - 6) + 2 * (k - 6) ≤ P) :
    5 + (m - 6) + (k - 6) ≤ jj := by
  unfold piopLen at h
  split_ifs at h <;> omega

/-- **`vsWord` at the honest values is the table's own value** at the tape's base, below `n_E`. -/
theorem vsWord_honestVals_tab {x : Msg} {w : Flock.Wit} {ν : CoinLayer.Nonce} {c : Fin (refK x) → FlockDeploy.Coin}
    {τ : UE.Tape} {j : Fin (jOf x)} {r : Fin 2} {k : ℕ} (hk : k < (tabC PE x j).z.nE) :
    vsWord (CoinLayer.session loopNr Masked.dataE).honestVals x w ν c τ j r k =
      FlockLevel3.GF128.val (((tabC PE x j).T (czExe PE x c j)).tab.vals r (wtOf PE x w j)
        ((UE.tape x τ).1 j).1 ((UE.tape x τ).1 j).2.1 ⟨k, hk⟩) := by
  unfold vsWord
  rw [dite_eq_left hk]
  rfl

set_option maxHeartbeats 1000000 in
/-- **The table's own value at a pad is one along coins agreeing up to the PIOP round that absorbs it**: at a place `P`
of the absorbed order before the Ligerito values, below the values of PIOP rounds `0` to `jj`, that holds pad `i`, the
honest sources' masked values at `i` agree at two coin vectors that agree below `repBase + jj` (`repOk`, the field at
`P`'s round: `round_r1` to `round_zp`), so the table's own values do (`vals_of_masked`). -/
theorem vals_freed {x : Msg} (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m)
    (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray) {c c' : Fin (refK x) → FlockDeploy.Coin}
    (j : Fin (jOf x)) (r : Fin 2) {jj P i : ℕ}
    (hcc : ∀ k : Fin (refK x), k.val < repBase x (2 * j.val + r.val) + jj → c k = c' k)
    (hPs : P < (absorbed (layE x j)).size - nLig (layE x j)) (hPi : (absorbed (layE x j))[P]! = i)
    (hPj : P < piopLen (stAt x j).m (stAt x j).c.kLog (jj + 1)) (hi : i < (tabC PE x j).z.nE) :
    ((tabC PE x j).T (czExe PE x c j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 ⟨i, hi⟩ =
      ((tabC PE x j).T (czExe PE x c' j)).tab.vals r (wt j) (ω j).1 (ω j).2.1 ⟨i, hi⟩ := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  have hA := repOk x (Masked.live_of_lt j) wt ω p c c' _ hcc s0 spec hsx j r
    (Array.replicate ((stAt x j).regions.size * spec.points) default) Array.size_replicate
  refine vals_of_masked wt ω p j r hi ?_
  have hm14 : 14 ≤ (stAt x j).m := (schOk_of (zkTab x j)).m
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hbase := Masked.repBase_eq hsx (2 * j.val + r.val)
  have hs := absorbed_size (layE x j)
  have hfa : (layE x j).fa = 128 + 2 * ((stAt x j).m - 6) := rfl
  have hlc : (layE x j).nLc = (stAt x j).c.kLog - 6 := rfl
  rw [absorbed_getElem! (layE x j) (p := P) (by omega)] at hPi
  have hM : ∀ {k : ℕ}, k = i →
      (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r).masked k =
        (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r).masked k →
      (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r).masked i =
        (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r).masked i := fun hk h => by
    rw [← hk]
    exact h
  split_ifs at hPi with hP
  · by_cases h1 : P < 64
    · have := round_r1 hPj
      refine hM ?_ (hA.ab ?_ P h1).1
      all_goals try simp only [FlockComplete.CutEq.layOf, pl_r1ab, ofCircuit_m, ofCircuit_kLog]
      all_goals omega
    by_cases h2 : P < 128
    · have := round_r1 hPj
      refine hM ?_ (hA.ab ?_ (P - 64) (by omega)).2
      all_goals try simp only [FlockComplete.CutEq.layOf, pl_r1c, ofCircuit_m, ofCircuit_kLog]
      all_goals omega
    by_cases h3 : P < 128 + 2 * ((stAt x j).m - 6)
    · have := round_ml hPj (by omega) h3
      rcases Nat.mod_two_eq_zero_or_one (P - 128) with hp | hp
      · refine hM ?_ (hA.ml ((P - 128) / 2) ?_ ?_).1
        all_goals try simp only [FlockComplete.CutEq.layOf, pl_ml, ofCircuit_m, ofCircuit_kLog]
        all_goals omega
      · refine hM ?_ (hA.ml ((P - 128) / 2) ?_ ?_).2
        all_goals try simp only [FlockComplete.CutEq.layOf, pl_ml, ofCircuit_m, ofCircuit_kLog]
        all_goals omega
    have := round_f hPj (by omega)
    by_cases h4 : P < 129 + 2 * ((stAt x j).m - 6)
    · refine hM ?_ (hA.f ?_).1
      all_goals try simp only [FlockComplete.CutEq.layOf, pl_fa, ofCircuit_m, ofCircuit_kLog]
      all_goals omega
    · refine hM ?_ (hA.f ?_).2.1
      all_goals try simp only [FlockComplete.CutEq.layOf, pl_fb, ofCircuit_m, ofCircuit_kLog]
      all_goals omega
  · by_cases h5 : P < 130 + 2 * ((stAt x j).m - 6) + 2 * ((stAt x j).c.kLog - 6)
    · have := round_lc hPj (by omega) h5
      rcases Nat.mod_two_eq_zero_or_one (P - 130 - 2 * ((stAt x j).m - 6)) with hp | hp
      · refine hM ?_ (hA.lc ((P - 130 - 2 * ((stAt x j).m - 6)) / 2) ?_ ?_).1
        all_goals try simp only [FlockComplete.CutEq.layOf, pl_lc, ofCircuit_m, ofCircuit_kLog, ofCircuit_fold]
        all_goals omega
      · refine hM ?_ (hA.lc ((P - 130 - 2 * ((stAt x j).m - 6)) / 2) ?_ ?_).2
        all_goals try simp only [FlockComplete.CutEq.layOf, pl_lc, ofCircuit_m, ofCircuit_kLog, ofCircuit_fold]
        all_goals omega
    · have := round_zp hPj (by omega)
      refine hM ?_ (hA.zp ?_ (P - (130 + 2 * ((stAt x j).m - 6) + 2 * ((stAt x j).c.kLog - 6))) ?_)
      all_goals try simp only [FlockComplete.CutEq.layOf, pl_zp, ofCircuit_m, ofCircuit_kLog, ofCircuit_fold]
      all_goals omega

section Takes

variable (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))

set_option maxHeartbeats 2000000 in
/-- **A pad is freed in the PIOP round whose values hold it**: where a round frees pad `i` of rep stream `2j + r`, its
history opens `repBase + jj` coins and `i` is at a place of the absorbed order before the Ligerito values and below the
values of PIOP rounds `0` to `jj`. The round is the rep's `jj`-th (`locate_inv`); the binding round, `jj = 0`, frees no
word (`zcBind_frame`); PIOP round `jj` frees from its state's position (`piop_frame`) the values it carries
(`roundFree_vals`), the last of which is below `piopLen (jj + 1)` (`piopV_length_eq`); past the PIOP every place before
the Ligerito values is below it (`piopLen_full`). -/
theorem freedAt_round (hreal : real = realOf src) {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {ν : CoinLayer.Nonce}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {cs : List FlockDeploy.Coin}
    {j : Fin (jOf x)} {r : Fin 2} {i : ℕ} {a : F128}
    (hF : FreedAt (LeanRelease.ofLoopLink L₀ src real) wk x w ν s cs j r i a) :
    ∃ jj P, cs.length = repBase x (2 * j.val + r.val) + jj ∧
      P < (absorbed (layE x j)).size - nLig (layE x j) ∧ (absorbed (layE x j))[P]! = i ∧
      P < piopLen (stAt x j).m (stAt x j).c.kLog (jj + 1) := by
  obtain ⟨t, κ, m₀, ρ₀, ops, v, m, out, s', fr, k, -, hd, hpos, hlt, hl, -, hv, ha, hf, hrep, hk, hi, -⟩ := hF
  have hcl : cs.length = ops.length := by
    subst hd
    exact List.length_map ..
  have hx : Masked.Live x := Masked.live_of_lt j
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  -- the place the round frees `i` at, in the rep's layout
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hst := start_of_answered Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real x
    (ν, t) (m₀, ρ₀) ops ha hsx hq
  have hk2 : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 = 2 * j.val + r.val := by
    rw [← LeanRelease.repOf_add_two hsx, show q + 2 = ops.length + 1 by omega]
    exact hrep
  rw [hk2, show (2 * j.val + r.val) / 2 = j.val by omega] at hst
  obtain ⟨lay, mask, hst⟩ := hst
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real j r hst
  have hord := cfg_order_of_rep (LeanRelease.ofLoopLink L₀ src real) (ρ := (ν, t)) (h := (m₀, ρ₀) :: ops)
    (k := frepOf j.val r.val) hrep
  rw [hlay] at hord
  have hP0 : m.pos + k < ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop :=
    (Free.roundFree_pos hf).2 k hk
  rw [hord.2] at hP0
  rw [hord.1] at hi
  -- the round is the rep's `jj`-th, after `repBase + jj` openings
  have hsum : (roundsOf spec (Masked.stsOf x)).sum + 1 = refK x := by
    rw [Masked.refK_eq hsx, ← FlockComplete.ZkTables.repStart_last, ← hrsz]
    unfold FlockComplete.repStartZk roundsOf
    omega
  have hinv := (locate_inv (roundsOf spec (Masked.stsOf x)) 0 q (by omega)).1
  simp only [Nat.sub_zero] at hinv
  rw [hk2] at hinv
  have hbase : repBase x (2 * j.val + r.val) =
      1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨jj, hjj⟩ : ∃ jj, (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2 = jj := ⟨_, rfl⟩
  rw [hjj] at hinv
  have hn : ops.length = repBase x (2 * j.val + r.val) + jj := by omega
  refine ⟨jj, m.pos + k, by rw [hcl]; exact hn, hP0, hi, ?_⟩
  have hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)) := by
    rw [hreal]
    exact realTabs_of src x w (ν, t) hsx
  have hm25 : 25 ≤ (stAt x j).m := (htabs j.val hj).1
  have hm35 : (stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hs := absorbed_size (layE x j)
  have hfa : (layE x j).fa = 128 + 2 * ((stAt x j).m - 6) := rfl
  have hlc : (layE x j).nLc = (stAt x j).c.kLog - 6 := rfl
  rcases Nat.eq_zero_or_pos jj with h0 | h0
  · -- the binding round frees no word
    obtain ⟨-, -, -, -, -, hfree, -⟩ := LeanRelease.zcBind_frame
      (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops hRT hlt hl hv j r
      (by omega)
    rw [hfree (m₁ := m) (out₁ := out) (s₁ := s') (fr₁ := fr) ha hf] at hk
    exact absurd hk (Nat.not_lt_zero k)
  by_cases hge : piopNR (stAt x j).m (stAt x j).c.kLog ≤ jj
  · -- past the PIOP, every place before the Ligerito values is below its values
    have e := piopLen_mono (stAt x j).m (stAt x j).c.kLog
      (show piopNR (stAt x j).m (stAt x j).c.kLog ≤ jj + 1 by omega)
    rw [piopLen_full] at e
    omega
  have hjjN : jj < piopNR (stAt x j).m (stAt x j).c.kLog := by omega
  -- PIOP round `jj` frees the values it carries
  obtain ⟨m', G, hA, hon, hin, hposm, hAG, hmid, hplain, hvG, hcoins, -, -⟩ :=
    LeanRelease.piop_frame (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀
      ops hRT hlt hl hv j r hn (by omega) hjjN
  have hA' : (LeanRelease.ofLoopLink L₀ src real).answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m' := hA
  have hmm : m' = m := Except.ok.inj (hA'.symm.trans ha)
  rw [hmm] at hon hin hposm
  have hcoins' : (LeanRelease.ofLoopLink L₀ src real).coins x (ops.length + 1) = piopN (stAt x j).m jj := hcoins
  have hc : ∀ p, p < ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size →
      ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[p]! <
        ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.size :=
    Claims.ZKServer.ofDraws_order _ _ _
  have hNO : ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop ≤
      ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size := by
    rw [hord.2, hord.1]
    omega
  have hlenF : m.pos + (fvals G).length = (piopV (stAt x j).m (stAt x j).c.kLog ByteArray.empty
      ByteArray.empty ByteArray.empty (layE x j) (fun _ => 0) (jj + 1)).length := by
    rw [hposm, fvals_length_alike hAG, piopV_succ, List.length_append]
  have hV1 : (piopV (stAt x j).m (stAt x j).c.kLog ByteArray.empty ByteArray.empty ByteArray.empty
      (layE x j) (fun _ => 0) (jj + 1)).length ≤ (layE x j).fa + 2 + (2 * (layE x j).nLc + 64) :=
    piopV_length_le (layE x j) (by omega) hk6 rfl rfl (by omega)
  have hbP : m.pos + nAbs G ≤ ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop := by
    rw [nAbs_fvals, hlenF, hord.2, absorbed_size]
    omega
  have hKN := piopKN_ok (m := (stAt x j).m) (jj := jj) (by omega) (lt_of_le_of_lt hm35 (by norm_num)) (by omega)
  obtain ⟨out', fr', hrf, hfr⟩ := roundFree_vals ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    { m with stopped := false } G (piopK jj) (piopN (stAt x j).m jj) hc hNO hon hin hmid hplain hKN.1
    hKN.2.1 hKN.2.2 hbP
  rw [← hvG, ← hcoins'] at hrf
  have hfr' : fr = fr' := (Prod.mk.inj (Prod.mk.inj (Except.ok.inj (hf.symm.trans hrf))).2).2
  have e := congrArg List.length hfr
  simp only [List.length_map, Array.length_toList] at e
  rw [← hfr'] at e
  have hL := piopV_length_eq (stAt x j).m (stAt x j).c.kLog ByteArray.empty ByteArray.empty ByteArray.empty
    (layE x j) (fun _ => 0) (jj + 1) hjjN
  omega

set_option maxHeartbeats 1000000 in
/-- **Where the worker takes the honest values, `vsOfWord` is the honest values**: where a round frees pad `i`, the
word is the table's value along the coins the round opened (`TakesVals` at `honestVals`, the round's worker the
pad-blind one: `phased_alg`), the first `repBase + jj` coins of round `jj` (`freedAt_round`), along which the table's
value is its value along the full coins (`vals_freed`); where none frees it, `vsOfWord` is the table's own value. -/
theorem vsOfWord_of_takes (hreal : real = realOf src) {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real)) (post : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker)
    (hT : TakesVals (CoinLayer.session loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
      (phased _ bx post) x w)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {i : ℕ}
    (hi : i < (tabC PE x j).z.nE) :
    vsOfWord _ bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 i =
      vsWord (CoinLayer.session loopNr Masked.dataE).honestVals x w ν c τ j r i := by
  have hi' : i < (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.nE := hi
  rcases hfw : freeWord _ bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r i with _ | a
  · refine ((vsOfWord_of_none _ bx _ _ _ hfw).trans (dite_eq_left hi')).trans ?_
    unfold vsWord
    rw [dite_eq_left hi]
    rfl
  refine (vsOfWord_of_free _ bx _ _ _ hfw).trans ?_
  -- the round that frees it
  have hex : ∃ n a, n < refK x ∧ FreedAt (LeanRelease.ofLoopLink L₀ src real) (blind _ (bwOf _ bx)) x w ν
      ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) ((List.ofFn c).take n) j r i a := by
    by_contra hne
    have e : freeWord _ bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r i = none := by
      unfold freeWord
      exact dite_eq_right hne
    rw [e] at hfw
    cases hfw
  have e : freeWord _ bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r i =
      some (Classical.choose (Classical.choose_spec hex)) := by
    unfold freeWord
    exact dite_eq_left hex
  obtain ⟨hn, hF⟩ := Classical.choose_spec (Classical.choose_spec hex)
  rw [Option.some.inj (e.symm.trans hfw)] at hF
  obtain ⟨n, hnn⟩ : ∃ n, Classical.choose hex = n := ⟨_, rfl⟩
  rw [hnn] at hn hF
  obtain ⟨jj, P, hlen, hPs, hPi, hPj⟩ := freedAt_round L₀ src real hreal hSv _ hF
  rw [List.length_take, List.length_ofFn, Nat.min_eq_left hn.le] at hlen
  obtain ⟨t, κ, m₀, ρ₀, ops, v, m, out, s', fr, k, hs, hd, hpos, hlt, hl, hok, hv, ha, hf, hrep, hk, hik, hwd⟩ := hF
  -- the round's worker is the pad-blind one, so the worker takes the honest values there
  have hle : ((m₀, ρ₀) :: ops).length ≤ refK x := by
    rw [List.length_cons]
    omega
  have he : ∀ h' : List (Msg × Msg), h'.length ≤ ((m₀, ρ₀) :: ops).length →
      phased _ bx post x w (ν, t) h' = blind _ (bwOf _ bx) x w (ν, t) h' :=
    fun h' hh => phased_alg _ bx post x w (ν, t) h' (hh.trans hle)
  have hlP := (ZkServer.live_congr _ κ he).trans hl
  have hvP := (congrFun (he ((m₀, ρ₀) :: ops) le_rfl) 0).trans hv
  have hw := hT ν t κ m₀ ρ₀ ops v m out s' fr j r hpos hlt hlP hok hvP ha hf hrep k hk
  rw [hik, hwd, hd] at hw
  have hcc : ∀ k' : Fin (refK x), k'.val < repBase x (2 * j.val + r.val) + jj →
      c k' = (CoinLayer.session loopNr Masked.dataE).pad ((List.ofFn c).take n) k' :=
    fun k' hk' => (RefSession.pad_take (CoinLayer.session loopNr Masked.dataE) c n k' (by omega)).symm
  have h1 : ((UE.tape x t).1 j).1 = ((UE.tape x τ).1 j).1 :=
    congrArg (fun s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z => (s j).1) hs
  have h2 : ((UE.tape x t).1 j).2.1 = ((UE.tape x τ).1 j).2.1 :=
    congrArg (fun s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z => (s j).2.1) hs
  have hv := vals_freed (wtOf PE x w) (UE.tape x τ).1 (fun _ => (0, ByteArray.empty)) j r hcc hPs hPi hPj hi
  refine hw.trans ((vsWord_honestVals_tab hi).trans
    ((congrArg FlockLevel3.GF128.val ?_).trans (vsWord_honestVals_tab hi).symm))
  rw [h1, h2]
  exact hv.symm

/-- **An honest worker's served values meet `RowsOK`**: at a witness under the relation, where the algebraic-phase
worker `bx` takes the honest values (`TakesVals` at `honestVals`, the worker `phased bx post`), `vsServed bx` is the
honest values at every index: `vsOfWord_of_takes` off `final_c`'s, and at `final_c`'s the c_eval of the round-1c values,
which is the honest values' `final_c` (`finalCFixed_honestVals`). So its rows hold where the honest values' do
(`rowsOK_honestVals`). -/
theorem rowsOK_vsServed_of_takes (hreal : real = realOf src) {x : Msg} {w : Flock.Wit} (hSv : Serves x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (bx : AlgWorker (LeanRelease.ofLoopLink L₀ src real)) (post : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker)
    (hT : TakesVals (CoinLayer.session loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
      (phased _ bx post) x w) :
    RowsOK (vsServed _ bx) x w := by
  intro ν c τ j hout
  have hE : (tabC PE x j).z.nE = (layE x j).hab := by
    show (layE x j).len - 4 = (layE x j).hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlt : ∀ i < 64, (layE x j).r1c i < (layE x j).fc := fun i _ => by
    simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.r1c, _root_.Flock.Zk.PadLayout.fc,
      _root_.Flock.Zk.PadLayout.fa]
    omega
  have hr1c : ∀ i < 64, (layE x j).r1c i < (tabC PE x j).z.nE := fun i hi => by
    rw [hE]
    exact lt_hab (layE x j) (hlt i hi).le
  have hw : ∀ (r : Fin 2) {i : ℕ} (hi : i < (tabC PE x j).z.nE),
      vsOfWord _ bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 i =
        vsWord (CoinLayer.session loopNr Masked.dataE).honestVals x w ν c τ j r i :=
    fun r _ hi => vsOfWord_of_takes L₀ src real hreal hSv bx post hT ν c τ j r hi
  refine (FlockSoundness.ZK.Table.onVals_rowsAt_congr _ _ _ _ _ _ fun r => ?_).2
    (rowsOK_honestVals hR ν c τ j hout)
  funext k
  obtain ⟨i, hi⟩ := k
  have hiz : i < (tabC PE x j).z.nE := hi
  by_cases hfi : i = (layE x j).fc
  · subst hfi
    have e0 : vsServed _ bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 ⟨(layE x j).fc, hi⟩ =
        (_root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
          ((Array.range 64).map fun k => vsOfWord _ bx x w ν c
            ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 ((layE x j).r1c k)) :
          FlockLevel3.GF128) :=
      ite_eq_left rfl
    have e2 : ((Array.range 64).map fun k => vsOfWord _ bx x w ν c
          ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 ((layE x j).r1c k)) =
        (Array.range 64).map fun k => vsWord (CoinLayer.session loopNr Masked.dataE).honestVals x w ν c τ j r
          ((layE x j).r1c k) :=
      Array.ext (by simp) fun k h1 _ => by
        simp only [Array.getElem_map, Array.getElem_range]
        exact hw r (hr1c k (by simpa using h1))
    have e3 := finalCFixed_honestVals x w ν c τ j hout r
    rw [e0, e2, ← e3]
    unfold vsWord
    rw [dite_eq_left hiz]
    rfl
  · refine (ite_eq_right hfi).trans ((hw r hiz).trans ?_)
    unfold vsWord
    rw [dite_eq_left hiz]
    rfl

end Takes

end Loop

namespace LeanRelease

/-- **At an honestly serving algebraic-phase worker, through `vsOf`**: `liveRoundEqOwn_vsOf` at a worker
`Loop.phased bx post` that takes the honest values (`Loop.TakesVals` at `honestVals`), whose served values' rows then
hold (`Loop.rowsOK_vsServed_of_takes`). -/
theorem liveRoundEqOwn_vsOf_takes (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables) (hTape : Loop.finS.TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (bx : Loop.AlgWorker (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
    (post : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hT : Loop.TakesVals Loop.finS.honestVals
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) (Loop.phased _ bx post) x w) :
    LiveRoundEqOwn Loop.finS (Loop.vsOf _ bx)
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) CoinOpen.shaH
      (Loop.phased _ bx post) x w :=
  liveRoundEqOwn_vsOf L₁ src real hreal x w hTab hTape hR hSv hsc hSrc bx
    (Loop.rowsOK_vsServed_of_takes _ src real hreal hSv hR bx post hT) post

end LeanRelease

end Claims.ZKServer
