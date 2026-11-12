module

public import Proofs.Flock.SessionLoop.VsOfTakes
public import Proofs.Flock.SessionLoop.CutPiopLive
public import Proofs.Flock.SessionLoop.PiopState
public import Proofs.Flock.SessionLoop.RepSrc

/-!
# A pad is freed no later than the PIOP round that absorbs it

* `piopV_length_eq`: the values the PIOP's first `a ≤ piopNR` rounds carry number `piopLen m kLog a`, whatever the
  binding, layout and masked values.
* `LeanRelease.piop_pos`: where the frame takes the coins after a history whose next round is rep stream `2j + r`'s
  PIOP round `jj`, `1 ≤ jj ≤ piopNR`, that round is located there, and the frame's state is at the values before it
  (`piopV … jj`), whatever the worker sends.
* **`Loop.freedAt_le_body`**: a pad of rep stream `2j + r` at a place of the absorbed order below the values of PIOP rounds
  `0` to `jj` is freed (`FreedAt`) only after a history of at most `repBase + jj` openings. Past it, the next round is PIOP
  round `jj + 1`, whose state is already past that place (`piop_pos`), and the rep's later rounds only move the state on
  (`Loop.progOf_pos_mono`), so no later round frees it.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Claims.ZKServer

open _root_.Flock (V F128)

section PiopLen

/-- **The values the PIOP's first `a` rounds carry, counted**: none through `r_outer`'s round, `round1`'s 128, two each
residual round and `final_a`/`final_b`'s, none at `β`'s round, two each lincheck round, and `z_partial`'s 64. -/
def piopLen (m kLog a : ℕ) : ℕ :=
  if a ≤ 2 then 0
  else if a ≤ 3 + (m - 6) then 128 + 2 * (a - 3)
  else if a ≤ 5 + (m - 6) then 128 + 2 * (m - 6) + 2
  else if a ≤ 5 + (m - 6) + (kLog - 6) then 128 + 2 * (m - 6) + 2 + 2 * (a - 5 - (m - 6))
  else 128 + 2 * (m - 6) + 2 + 2 * (kLog - 6) + 64

/-- The values the PIOP's round `jj` carries, counted. -/
def piopFrLen (m kLog jj : ℕ) : ℕ :=
  if jj = 0 then 0 else if jj = 1 then 0 else if jj = 2 then 128 else if jj < 3 + (m - 6) then 2
  else if jj = 3 + (m - 6) then 2 else if jj = 4 + (m - 6) then 0
  else if jj < 5 + (m - 6) + (kLog - 6) then 2 else 64

theorem fvals_piopFr_len (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) (jj : ℕ) : (fvals (piopFr m kLog digest cap root lay mk jj)).length = piopFrLen m kLog jj := by
  unfold piopFr piopFrLen
  split_ifs <;> simp [fvals]

theorem piopLen_succ (m kLog a : ℕ) (h : a < piopNR m kLog) :
    piopLen m kLog (a + 1) = piopLen m kLog a + piopFrLen m kLog a := by
  unfold piopNR at h
  unfold piopFrLen
  split_ifs with h0 h1 h2
  · subst h0; rfl
  · subst h1; rfl
  · subst h2; unfold piopLen; split_ifs <;> omega
  all_goals unfold piopLen; split_ifs <;> omega

theorem piopLen_mono (m kLog : ℕ) {a b : ℕ} (h : a ≤ b) : piopLen m kLog a ≤ piopLen m kLog b := by
  unfold piopLen
  split_ifs <;> omega

theorem piopLen_full (m kLog : ℕ) :
    piopLen m kLog (piopNR m kLog) = 128 + 2 * (m - 6) + 2 + (2 * (kLog - 6) + 64) := by
  unfold piopLen piopNR
  split_ifs <;> omega

/-- Through `z_partial`'s round, the PIOP's values are at most the absorbed order's before the Ligerito values. -/
theorem piopLen_le (m kLog : ℕ) {a : ℕ} (h : a ≤ piopNR m kLog) :
    piopLen m kLog a ≤ 128 + 2 * (m - 6) + 2 + (2 * (kLog - 6) + 64) :=
  le_trans (piopLen_mono m kLog h) (piopLen_full m kLog).le

/-- **The values the PIOP's first `a ≤ piopNR` rounds carry number `piopLen m kLog a`**, at any binding, layout and
masked values. -/
theorem piopV_length_eq (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) : ∀ a, a ≤ piopNR m kLog → (piopV m kLog digest cap root lay mk a).length = piopLen m kLog a
  | 0, _ => by simp [piopV, piopLen]
  | a + 1, h => by
    rw [piopV_succ, List.length_append, piopV_length_eq m kLog digest cap root lay mk a (by omega),
      fvals_piopFr_len, piopLen_succ m kLog a h]

end PiopLen

section PiopPos

open FlockSoundness.Refine (Frame encs)
open Claims Claims.ZKMal
open _root_.Flock (SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

open Loop in
set_option maxHeartbeats 2000000 in
/-- **The frame's state at a rep's PIOP round past its first**: where the frame takes the coins after a history whose
next round is rep stream `2j + r`'s PIOP round `jj`, `1 ≤ jj ≤ piopNR`, the round is located there and the frame's
state is at the values of the rounds before it (`repState_piop`), at any tape and whatever the worker sends. -/
theorem piop_pos (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) {x : Msg} (hSv : Loop.Serves x)
    (ρ : Loop.Tp) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) {m : _root_.Flock.Firewall.Frame.Masker}
    (hA : (ofLoop Masked.dataE L₁ real).answered x ρ ((m₀, ρ₀) :: ops) = .ok m) (j : Fin (Masked.jOf x))
    (r : Fin 2) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {jj : ℕ}
    (hn : ops.length = Masked.repBase x (2 * j.val + r.val) + jj) (h0 : jj ≠ 0)
    (hjj : jj ≤ piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog) :
    ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (ops.length - 1) = (2 * j.val + r.val, jj) ∧
      m.pos = (piopV (Masked.stAt x j).m (Masked.stAt x j).c.kLog ByteArray.empty ByteArray.empty ByteArray.empty
        (Masked.layE x j) (fun _ => 0) jj).length := by
  have hx := Masked.live_of_lt j
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hR0 : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog <
      (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hst]
    have hA0 : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
        (Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6) := by
      unfold piopNR
      omega
    refine Nat.lt_of_le_of_lt hA0 ?_
    show _ < (Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6) +
      (2 + spec.points * (Masked.stAt x j).regions.size + 1) +
      ((Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) + 4
    omega
  obtain ⟨a, rfl⟩ : ∃ a, jj = a + 1 := ⟨jj - 1, by omega⟩
  have hbase : Masked.repBase x (2 * j.val + r.val) =
      1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨i, hp⟩ : ∃ i, ops.length = i + 2 := ⟨ops.length - 2, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1) = (2 * j.val + r.val, a + 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) (a + 1)
      (by rw [hrsl]; exact hq2) (lt_of_le_of_lt hjj hR0)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 = a + 1 := by rw [hloc]
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ real x ρ (m₀, ρ₀) ops hsx (q := i + 1) (by omega) hj1 ho ht] at hA'
  replace hA' := (bind_rankStop_ok hA').1
  have hcs : (ops.map fun p => ZKMal.CoinLayer.rd p.2).length = i + 2 := by rw [List.length_map]; omega
  have ho' : (progOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun h => answer_owed_none _ h hA'
  obtain ⟨c', tb, dom, xs, his, hph0⟩ := progOf_placeholder x _ hsx hcs ho'
  rw [locate_pred _ 0 i a hj1, hloc] at hph0
  dsimp only at hph0
  have hph : placeholder spec (Masked.stsOf x) (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).1
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', tb, dom, xs, his) := by
    rw [hloc]
    exact hph0
  rw [progOf_at x _ hsx (q := i + 1) (by rw [List.length_map]; omega) hj1 hph, hloc] at hA'
  dsimp only at hA'
  obtain ⟨alg, sched, hal, hs⟩ := repState_ok hA'
  obtain ⟨hts', -, -⟩ := Count.placeholder_tab hph0
  rw [hst] at hts'
  have htm : tb.st.m = (Masked.stAt x j).m := by rw [hts']; rfl
  have htk' : tb.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hts']; rfl
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  obtain ⟨sch', tp', hsch', hlay⟩ := start_lay hds
  rw [hz] at hsch'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hsch'
  obtain ⟨rfl, rfl⟩ := hsch'
  rw [hst] at hlay
  have hlayS : lay = Masked.layE x j := hlay
  have hnM : lay.nMlv = tb.st.m - 6 := by
    rw [hlayS, htm]
    try rfl
  have hnL : lay.nLc = tb.st.fold.kLog - 6 := by
    rw [hlayS, htk']
    try rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hjjT : a + 1 ≤ piopNR tb.st.m tb.st.fold.kLog := by rw [htm, htk']; exact hjj
  have hjjT' : a < piopNR tb.st.m tb.st.fold.kLog := by omega
  have hb : (piopV tb.st.m tb.st.fold.kLog tb.st.digest (_root_.Flock.Firewall.Finish.flat tb.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0) (fun _ => 0) (a + 1)).length ≤ c'.order.size := by
    rw [hc', Loop.ofDraws_order, absorbed_size, piopV_length_eq _ _ _ _ _ _ _ _ hjjT]
    have e := piopLen_le tb.st.m tb.st.fold.kLog hjjT
    simp only [_root_.Flock.Zk.PadLayout.fa]
    omega
  have hdg : tb.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tb.st.m < 2 ^ 64 := by rw [htm]; exact lt_of_le_of_lt hm35 (by norm_num)
  obtain ⟨-, hdom', -, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨-, -, -, hpos⟩ := repState_piop hcc hal hs (by omega) hmb hdg (placeholder_cap hph0)
    (hdm ▸ hdom') hjjT' hb hA'
  refine ⟨?_, ?_⟩
  · rw [show ops.length - 1 = i + 1 by omega]
    exact hloc
  · rw [hpos, htm, htk', piopV_length_eq _ _ _ _ _ _ _ _ hjj, piopV_length_eq _ _ _ _ _ _ _ _ hjj]

end LeanRelease

end PiopPos

section FreedBody

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)
open _root_.Flock.Firewall.Frame (Cfg Masker absorbed nLig)

namespace Loop

/-- **A pad is freed no later than the PIOP round that absorbs it**: where pad `i` of rep stream `2j + r` is at place
`P` of the absorbed order, below the values of PIOP rounds `0` to `jj < piopNR`, every round that frees it (`FreedAt`) is
after a history of at most `repBase + jj` openings. After one of more, the next round is PIOP round `jj + 1`, at whose
state the frame is past `P` (`LeanRelease.piop_pos`); the rep's later rounds only move it on
(`progOf_pos_mono`, `answered_pos_same`), and a round frees only places from its own state on (`Free.roundFree_pos`,
`absorbed_inj`). -/
theorem freedAt_le_body
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hSv : Serves x) (hP : (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w)
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) {ν : CoinLayer.Nonce}
    {s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z} {cs : List FlockDeploy.Coin}
    {j : Fin (jOf x)} {r : Fin 2} {i : ℕ} {a : F128}
    (hF : FreedAt (LeanRelease.ofLoopLink L₀ src real) wk x w ν s cs j r i a) {P jj : ℕ}
    (hPs : P < (absorbed (layE x j)).size) (hPi : (absorbed (layE x j))[P]! = i)
    (hjj : jj < piopNR (stAt x j).m (stAt x j).c.kLog)
    (hPv : P < (piopV (stAt x j).m (stAt x j).c.kLog ByteArray.empty ByteArray.empty ByteArray.empty (layE x j)
      (fun _ => 0) (jj + 1)).length) :
    cs.length ≤ repBase x (2 * j.val + r.val) + jj := by
  obtain ⟨t, κ, m₀, ρ₀, ops, v, m, out, s', fr, k, -, hd, hpos, hlt, hl, -, -, ha, hf, hrep, hk, hi, -⟩ := hF
  have hcl : cs.length = ops.length := by
    subst hd
    exact List.length_map ..
  rw [hcl]
  by_contra hgt
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  -- the rep's layout, and the place the round frees `i` at
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hst := start_of_answered Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real x
    (ν, t) (m₀, ρ₀) ops ha hx hq
  have hk2 : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 = 2 * j.val + r.val := by
    rw [← LeanRelease.repOf_add_two hx, show q + 2 = ops.length + 1 by omega]
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
  have hE : m.pos + k = P := absorbed_inj _ (by omega) hPs (hi.trans hPi.symm)
  -- PIOP round `jj + 1`, the round after the first `n₁` openings
  obtain ⟨n₁, hn₁⟩ : ∃ n, n = repBase x (2 * j.val + r.val) + jj + 1 := ⟨_, rfl⟩
  have hlen : ((m₀, ρ₀) :: ops.take n₁).length = n₁ + 1 := by
    rw [List.length_cons, List.length_take, Nat.min_eq_left (by omega)]
  have hlast : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  have hlq := LeanRelease.live_take (LeanRelease.ofLoopLink L₀ src real) wk x w (ν, t) κ _ hl (n₁ + 1)
  rw [List.take_succ_cons] at hlq
  have hkq : (LeanRelease.ofLoopLink L₀ src real).kindAt x ((m₀, ρ₀) :: ops.take n₁).length = .masked :=
    LeanRelease.kindAt_masked_of _ (by rw [hlen]; omega) (by rw [hlen, hlast]; omega)
  obtain ⟨v₁, hv₁, -⟩ := LeanRelease.live_some (LeanRelease.ofLoopLink L₀ src real) wk x w (ν, t) κ _ hkq hlq
  obtain ⟨M₁, o₁, s₁, hM₁, -⟩ :=
    LeanRelease.check_of_live (LeanRelease.ofLoopLink L₀ src real) wk x w (ν, t) κ _ v₁ hlq hkq hv₁
  have hn₁l : (ops.take n₁).length = n₁ := by rw [List.length_take, Nat.min_eq_left (by omega)]
  obtain ⟨hloc₁, hM₁pos⟩ := LeanRelease.piop_pos (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real hSv
    (ν, t) m₀ ρ₀ (ops.take n₁) hM₁ j r hx (jj := jj + 1) (by omega) (by omega) (by omega)
  rw [hn₁l] at hloc₁
  have hloc₁f : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (n₁ - 1)).1 = 2 * j.val + r.val := by
    rw [hloc₁]
  have hloc₁s : (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (n₁ - 1)).2 = jj + 1 := by rw [hloc₁]
  -- the rounds from `n₁` on are the rep's rounds past its first
  have hin : ∀ p, n₁ ≤ p → p ≤ ops.length →
      ∃ j', (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (p - 1)).2 = j' + 1 := by
    intro p h1 h2
    by_cases hp : p = n₁
    · subst hp
      exact ⟨jj, hloc₁s⟩
    · have e1 := locate_fst_mono (roundsOf spec (Masked.stsOf x)) 0 (n₁ - 1) (p - 1) (by omega)
      have e2 := locate_fst_mono (roundsOf spec (Masked.stsOf x)) 0 (p - 1) q (by omega)
      have hne := locate_snd_ne_zero (roundsOf spec (Masked.stsOf x)) 0 (n₁ - 1) (p - 1) (by omega) (by omega)
      exact ⟨(CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (p - 1)).2 - 1, by omega⟩
  have hm₁ := answered_pos_same Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real x
    (ν, t) (m₀, ρ₀) (ops.take n₁) hM₁ hx (q := n₁ - 1) (by omega) hloc₁s
  have hw := progOf_pos_mono (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real hP wk (ν, t) κ m₀ ρ₀
    ops hl hlt hx (n₁ := n₁) (n₂ := ops.length) (by omega) (by omega) le_rfl (fun p h1 h2 => hin p h1 h2.le)
  rw [List.take_length] at hw
  obtain ⟨jq, hjq⟩ := hin ops.length (by omega) le_rfl
  have hm := answered_pos_same Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real x
    (ν, t) (m₀, ρ₀) ops ha hx (q := ops.length - 1) (by omega) hjq
  omega

end Loop

end FreedBody

end Claims.ZKServer
