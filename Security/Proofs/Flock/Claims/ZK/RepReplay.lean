import Proofs.Flock.Claims.ZK.RepFree
import Proofs.Flock.Claims.ZK.SimPiopGo
import Proofs.Flock.Claims.ZK.FinishSwPts
import Proofs.Flock.Claims.Complete.ZkTables.ReplayGo
import Proofs.Flock.Soundness.Discharge.Composed.Inner.LincheckReplay

/-!
# The cut rep's replay returns on its coins alone

The replay re-derives each of the rep's coins from the rounds the rep recorded, so it returns where the rep's own
zerocheck and lincheck absorbed what the replay absorbs. The cut's zerocheck and lincheck at masked values `mk` are the
masked prover's at the pads `mk` and the empty trace, whose plain values are all `0`: so their proofs and transcripts
are `ZcZ`'s and `LcZ`'s, and `replayWith_ok` runs the replay to its end wherever the circuit fold at the coins returns
a vector of width `64`. That fold reads only coins, so it is the firewall's.

* **The empty trace** (`tr0`): every plain zerocheck and lincheck value is `0` (`zc_msg_nil`, `lc_msg_nil`, …).
* **Closed forms** (`zerocheckCut_closed`, `lincheckCut_closed`): answered by `asksCut` at replies with its coins, the
  cut's zerocheck and lincheck at `mk` return `ZcZ`'s and `LcZ`'s proofs and transcripts at the pads `padsOf lay mk`.
* **`replay_cut_ok`**: on rounds that begin with the cut's, the replay of a proof whose zerocheck and lincheck are the
  cut's returns, wherever the fold at the coins returns a vector of width `64`.
* **`Lincheck.replay_fold_w`**: a returned replay's fold at its rounds' coins returned a vector of width `64`.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.CutFree

open Flock Transcript Replay

/-! ## Words -/

section Arith
open FlockSoundness.Refine

theorem f_mul_zero (a : F128) : a * 0 = 0 := fOf_injective (by rw [fOf_mul, fOf_zero, mul_zero])
theorem f_zero_mul (a : F128) : 0 * a = 0 := fOf_injective (by rw [fOf_mul, fOf_zero, zero_mul])
theorem f_add_zero (a : F128) : a + 0 = a := fOf_injective (by rw [fOf_add, fOf_zero, add_zero])
theorem f_zero_add (a : F128) : 0 + a = a := fOf_injective (by rw [fOf_add, fOf_zero, zero_add])

theorem dot_zero_right (a b : Array F128) (hb : ∀ i : ℕ, b[i]! = 0) : dot a b = 0 :=
  fOf_injective (by
    rw [fOf_dot a b a.size rfl, fOf_zero]
    exact Finset.sum_eq_zero fun i _ => by rw [hb, fOf_zero, mul_zero])

end Arith

theorem get_map_range_zero {n : ℕ} {f : ℕ → F128} (h : ∀ i < n, f i = 0) (i : ℕ) :
    ((Array.range n).map f)[i]! = 0 := by
  by_cases hi : i < n
  · rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range]
    exact h i hi
  · rw [getElem!_neg _ i (by simpa using hi)]
    rfl

theorem sumF_eq_zero {f : ℕ → F128} : ∀ {n : ℕ}, (∀ i < n, f i = 0) → sumF n f = 0
  | 0, _ => rfl
  | n + 1, h => by
    have ih := sumF_eq_zero (n := n) fun i hi => h i (by omega)
    unfold sumF at ih ⊢
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, ih, h n (by omega), f_add_zero]

theorem sumF_zero (n : ℕ) : sumF n (fun _ => 0) = 0 := sumF_eq_zero fun _ _ => rfl

/-! ## The empty trace -/

/-- **The empty trace**: no bits, so every plain value is `0`. -/
abbrev tr0 : Trace := ⟨#[], #[], #[]⟩

theorem skipHat_nil (w : Array F128) (x : ℕ) : skipHat #[] w x = 0 := by
  unfold skipHat
  exact sumF_eq_zero fun s _ => by
    rw [getElem!_neg (#[] : Array Bool) _ (by simp)]
    rfl

theorem zc_tab_nil (n : ℕ) (z : F128) (ρ : ℕ → F128) : ∀ i k : ℕ, (Zc.tab n z ρ #[] i)[k]! = 0
  | 0, k => by
    show ((Array.range (2 ^ n)).map (skipHat #[] (skipWeights z)))[k]! = 0
    exact get_map_range_zero (fun x _ => skipHat_nil _ x) k
  | i + 1, k => by
    show (foldLow (ρ i) (Zc.tab n z ρ #[] i))[k]! = 0
    unfold foldLow
    exact get_map_range_zero (fun x _ => by simp only [zc_tab_nil n z ρ i, f_add_zero, f_mul_zero]) k

theorem zc_r1ab_nil (n : ℕ) (r : Array F128) (i : ℕ) : (Zc.r1ab tr0 n r)[i]! = 0 := by
  unfold Zc.r1ab
  exact get_map_range_zero (fun j _ => sumF_eq_zero fun x _ => by simp only [tr0, skipHat_nil, f_mul_zero]) i

theorem zc_r1c_nil (n : ℕ) (r : Array F128) (i : ℕ) : (Zc.r1c tr0 n r)[i]! = 0 := by
  unfold Zc.r1c
  exact get_map_range_zero (fun j _ => sumF_eq_zero fun x _ => by simp only [tr0, skipHat_nil, f_mul_zero]) i

theorem zc_msg_nil (n : ℕ) (r : Array F128) (z : F128) (ρ : ℕ → F128) (i : ℕ) :
    Zc.msg tr0 n r z ρ i = (0, 0) := by
  simp only [Zc.msg, tr0, zc_tab_nil, f_add_zero, f_mul_zero, sumF_zero]

theorem lc_zr0_nil (c : CircuitFold) (m : ℕ) (zc : ZcOut) (k : ℕ) : (Lc.zr0 c m tr0 zc)[k]! = 0 := by
  unfold Lc.zr0
  exact get_map_range_zero (fun col _ => sumF_eq_zero fun o _ => by
    simp only [tr0]
    rw [getElem!_neg (#[] : Array Bool) _ (by simp)]
    rfl) k

theorem lc_zrT_nil (c : CircuitFold) (m : ℕ) (zc : ZcOut) (cs : Lc.Challenges) :
    ∀ i k : ℕ, (Lc.zrT c m tr0 zc cs i)[k]! = 0
  | 0, k => by
    show (Lc.zr0 c m tr0 zc)[k]! = 0
    exact lc_zr0_nil c m zc k
  | i + 1, k => by
    show (halve (cs.τ i) (Lc.zrT c m tr0 zc cs i))[k]! = 0
    unfold halve
    exact get_map_range_zero (fun x _ => by simp only [lc_zrT_nil c m zc cs i, f_add_zero, f_mul_zero]) k

theorem lc_msg_nil (c : CircuitFold) (m : ℕ) (zc : ZcOut) (full : Array F128) (cs : Lc.Challenges) (j : ℕ) :
    Lc.msg c m tr0 zc full cs j = (0, 0) := by
  simp only [Lc.msg, lc_zrT_nil, f_add_zero, f_mul_zero, sumF_zero]

/-- **The masked values as pads**, on the layout's length. -/
def padsOf (lay : Zk.PadLayout) (mk : ℕ → F128) : Array F128 := (Array.range lay.len).map mk

theorem padsOf_get {lay : Zk.PadLayout} {mk : ℕ → F128} {i : ℕ} (hi : i < lay.len) : (padsOf lay mk)[i]! = mk i := by
  rw [padsOf, getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range]

/-- Pad indices below the layout's length. -/
theorem lay_lt_len (lay : Zk.PadLayout) :
    (∀ i < 64, lay.r1ab i < lay.len) ∧ (∀ i < 64, lay.r1c i < lay.len) ∧
      (∀ j < lay.nMlv, ∀ t < 2, lay.ml j t < lay.len) ∧ lay.fa < lay.len ∧ lay.fb < lay.len ∧ lay.fc < lay.len ∧
      (∀ j < lay.nLc, ∀ t < 2, lay.lc j t < lay.len) ∧ (∀ i < 64, lay.zp i < lay.len) := by
  unfold Zk.PadLayout.len Zk.PadLayout.tw Zk.PadLayout.hab Zk.PadLayout.zp Zk.PadLayout.lc Zk.PadLayout.fc
    Zk.PadLayout.fb Zk.PadLayout.fa Zk.PadLayout.ml Zk.PadLayout.r1c Zk.PadLayout.r1ab
  refine ⟨fun _ _ => ?_, fun _ _ => ?_, fun _ _ _ _ => ?_, ?_, ?_, ?_, fun _ _ _ _ => ?_, fun _ _ => ?_⟩ <;> omega

/-! ## Closed forms -/

/-- **The cut's zerocheck is the masked prover's at the pads `mk` and the empty trace**: answered by `asksCut` at
replies with its coins, where it returns from `l`, its proof and transcript are `ZcZ`'s at the coins from `l.size`. -/
theorem zerocheckCut_closed (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg}
    (hr : ∀ k, coinOf (resp k) = coin k) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray) {lay : Zk.PadLayout}
    (hn : m - 6 ≤ lay.nMlv) (mk : ℕ → F128) {t t₁ : Transcript} {l l₁ : Array Claims.Msg} {zp : ZerocheckProof}
    {zc : ZcOut}
    (h : ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l = (.ok ((zp, zc), t₁), l₁)) :
    zp = ZcZ.proof lay (padsOf lay mk) m (zcCoins coin l.size) tr0 ∧
      t₁ = ZcZ.tEnd lay (padsOf lay mk) m digest cap root (zcCoins coin l.size) tr0 t := by
  obtain ⟨hLab, hLc, hLml, hLfa, hLfb, hLfc, -, -⟩ := lay_lt_len lay
  have hA := zerocheckCut_alike coin n hm digest cap root lay (padsOf lay mk) tr0 mk t l
    (fun i hi => by rw [zc_r1ab_nil, f_zero_add, padsOf_get (hLab i hi)])
    (fun i hi => by rw [zc_r1c_nil, f_zero_add, padsOf_get (hLc i hi)])
    (fun i hi => by simp only [zc_msg_nil, f_zero_add]; rw [padsOf_get (hLml i (by omega) 0 (by omega))])
    (fun i hi => by simp only [zc_msg_nil, f_zero_add]; rw [padsOf_get (hLml i (by omega) 1 (by omega))])
    (by simp only [tr0, zc_tab_nil, f_zero_add]; rw [padsOf_get hLfa])
    (by simp only [tr0, zc_tab_nil, f_zero_add]; rw [padsOf_get hLfb])
    (by rw [dot_zero_right _ _ (zc_r1c_nil _ _), f_zero_add, padsOf_get hLfc])
  rw [h] at hA
  rcases hA with ⟨e, L, he, -⟩ | ⟨a, b, t', L, he, hy, hab, -⟩
  · simp only [Prod.mk.injEq, reduceCtorEq, false_and] at he
  simp only [Prod.mk.injEq, Except.ok.injEq] at he
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := he
  have hy' := Claims.ZKMal.Masked.FinRep.zerocheckPZk_cut (n := n) hr m digest cap root lay (padsOf lay mk) tr0 t l
    _ _ hy
  obtain ⟨l', -, hgo⟩ := zerocheckPZk_go resp hm digest cap root lay (padsOf lay mk) tr0 t l
  have e := hgo.symm.trans hy'
  simp only [Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨⟨rfl, rfl⟩, -⟩ := e
  have hc : Zc.coinsAt resp l.size = zcCoins coin l.size := by simp only [Zc.coinsAt, zcCoins, hr]
  rw [hc] at hab
  exact ⟨hab, by rw [hc]⟩

/-- **The cut's lincheck is the masked prover's at the pads `mk` and the empty trace**: answered by `asksCut` at replies
with its coins, from a zerocheck output with the `z` and `ρ` of `zcs`, where it returns from `l`, its fold at `α`
returned some `full`, and its proof and transcript are `LcZ`'s at the coins from `l.size`. -/
theorem lincheckCut_closed (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg}
    (hr : ∀ k, coinOf (resp k) = coin k) (c : CircuitFold) (m : ℕ) {lay : Zk.PadLayout} (hn' : c.kLog - 6 ≤ lay.nLc)
    (mk : ℕ → F128) (zcs : Zc.Challenges) {zc : ZcOut} (hz : zc.z = (Zc.out m zcs tr0).z)
    (hρ : zc.rho = (Zc.out m zcs tr0).rho) {t t₂ : Transcript} {l l₂ : Array Claims.Msg} {lp : LincheckProof}
    {lc : LcOut} (h : ((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l = (.ok ((lp, lc), t₂), l₂)) :
    ∃ full, c.fold (coin l.size 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok full ∧
      lp = LcZ.proof lay (padsOf lay mk) c m tr0 (Zc.out m zcs tr0) full (lcCoins coin l.size (c.kLog - 6)) ∧
      t₂ = LcZ.tEnd lay (padsOf lay mk) c m tr0 (Zc.out m zcs tr0) full (lcCoins coin l.size (c.kLog - 6)) t := by
  obtain ⟨full, hfull⟩ := RankAt.lincheckCut_fold coin n h
  obtain ⟨-, -, -, -, -, -, hLlc, hLzp⟩ := lay_lt_len lay
  have hA := lincheckCut_alike coin n c m lay (padsOf lay mk) tr0 mk (Zc.out m zcs tr0) hz hρ rfl rfl t l
    (fun full _ j hj => ⟨by
        simp only [LcZ.msg, lc_msg_nil, f_zero_add]; rw [padsOf_get (hLlc j (by omega) 0 (by omega))],
      by simp only [LcZ.msg, lc_msg_nil, f_zero_add]; rw [padsOf_get (hLlc j (by omega) 1 (by omega))]⟩)
    (fun i hi => by rw [lc_zrT_nil, f_zero_add, padsOf_get (hLzp i hi)])
  rw [h] at hA
  rcases hA with ⟨e, L, he, -⟩ | ⟨a, b, t', L, he, hy, hab, -⟩
  · simp only [Prod.mk.injEq, reduceCtorEq, false_and] at he
  simp only [Prod.mk.injEq, Except.ok.injEq] at he
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := he
  have hy' := Claims.ZKMal.Masked.FinRep.lincheckPZk_cut (n := n) hr c m lay (padsOf lay mk) tr0 (Zc.out m zcs tr0)
    t l _ _ hy
  have hfold : c.fold (coinOf (resp l.size) 0) (skipWeights (Zc.out m zcs tr0).z)
      ((Zc.out m zcs tr0).rho.extract 0 (c.kLog - 6)) = .ok full := by
    rw [hr, ← hz, ← hρ]
    exact hfull
  obtain ⟨l', -, hgo⟩ := lincheckPZk_go resp c m lay (padsOf lay mk) tr0 (Zc.out m zcs tr0) full t l hfold
  have e := hgo.symm.trans hy'
  simp only [Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨⟨rfl, rfl⟩, -⟩ := e
  have hc : Lc.coinsAt resp l.size (c.kLog - 6) = lcCoins coin l.size (c.kLog - 6) := by
    simp only [Lc.coinsAt, lcCoins, hr]
  rw [hc] at hab
  exact ⟨full, hfull, hab, by rw [hc]⟩

/-! ## The replay -/

/-- **The cut rep's replay returns**: answered by `asksCut` at replies with its coins, where the cut's zerocheck and
lincheck at any masked values `mk` returned from the empty transcript of domain `d` and the log `l`, the replay of a
proof whose zerocheck and lincheck they returned, on rounds that begin with the lincheck's, returns, wherever the
circuit fold at the coins from `l.size` returns a vector of width `64`. -/
theorem replay_cut_ok (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    {st : Setup} (hm : 14 ≤ st.m) (hkk : st.kLog = st.fold.kLog) {lay : Zk.PadLayout} (hn : st.m - 6 ≤ lay.nMlv)
    (hn' : st.fold.kLog - 6 ≤ lay.nLc) (mk : ℕ → F128) (p : Proof) (root d : ByteArray)
    {l l₁ l₂ : Array Claims.Msg} {zc : ZcOut} {t₁ t₂ : Transcript} {lc : LcOut}
    (h₁ : ((zerocheckCut (asksCut coin n) st.m st.digest (flat p.cap) root lay mk).run (Transcript.init #[] d)).go l =
      (.ok ((p.zerocheck, zc), t₁), l₁))
    (h₂ : ((lincheckCut (asksCut coin n) st.fold st.m lay mk zc).run t₁).go l₁ = (.ok ((p.lincheck, lc), t₂), l₂))
    {R : Array Round} (hR : Replay.Prefix t₂.rounds R)
    (hL : ∃ comb, st.fold.folded ((lcCoins coin (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6)).ca 0)
      ((lcCoins coin (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6)).cb 0) (skipWeights (zcCoins coin l.size).z)
      ((pushes (zcCoins coin l.size).ρ (st.m - 6)).extract 0 (st.fold.kLog - 6))
      (pushes (lcCoins coin (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6)).τ (st.fold.kLog - 6)) = .ok comb ∧
        comb.size = 64) :
    ∃ q, (Zk.replay st lay p root).run (Transcript.init R d) = .ok q := by
  obtain ⟨hzp, ht₁⟩ := zerocheckCut_closed coin n hr hm _ _ _ hn mk h₁
  obtain ⟨hz, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  subst ht₁
  obtain ⟨full, -, hlp, ht₂⟩ := lincheckCut_closed coin n hr st.fold st.m hn' mk (zcCoins coin l.size) hz.1 hz.2.1 h₂
  rw [hl₁] at hlp ht₂
  subst ht₂
  obtain ⟨q, hq⟩ := replayWith_ok With.std st hm lay (padsOf lay mk) p root (zcCoins coin l.size)
    (lcCoins coin (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6)) full tr0 (Transcript.init #[] d) rfl
    ((Agrees.self R 0).of_prefix hR) hzp hlp hkk hL
  rw [With.zk_replayWith_std] at hq
  exact ⟨q, hq⟩

end FlockComplete.CutFree

namespace FlockSoundness.Discharge.Composed.Lincheck

open Flock FlockLevel3 Model Refine

/-- **A returned replay's fold returned a vector of width `64`**, at the lincheck's `α`, `β`, the skip weights at `z`,
`ρ[.. k_log − 6]` and the round coins of the rounds it consumed (`replay_lincheck`'s, with the width check after it). -/
theorem replay_fold_w (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) (hm : 14 ≤ st.m)
    {t t' : Flock.Transcript} {rp : Zk.Replayed} (h : Zk.replay st lay p root t = .ok (rp, t')) :
    ∃ combv : Array F128,
      st.fold.folded (coinAt t (3 + (st.m - 6))) (coinAt t (4 + (st.m - 6))) (skipWeights (coinAt t 2))
          ((rhoOf st.m t).extract 0 (st.kLog - 6)) (tsOf st.m st.kLog t) = .ok combv ∧ combv.size = 64 ∧
      ∀ i < 5 + (st.m - 6) + (st.kLog - 6), (t.rounds[t.k + i]!).n ≠ 0 := by
  unfold Zk.replay at h
  dsimp only at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, n₀, t₇, r₇, kk₇, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨-, n₁, t₈, r₈, kk₈, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₂, t₁₁, r₁₁, kk₁₁, h⟩ := sample_bind h
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  obtain ⟨t₁₃, k₁₃, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, k₁₄, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, k₁₅, h⟩ := label_bind h
  obtain ⟨-, nα, t₁₆, r₁₆, kk₁₆, h⟩ := sample_bind h
  obtain ⟨-, nβ, t₁₇, r₁₇, kk₁₇, h⟩ := sample_bind h
  obtain ⟨s', t₁₈, hloop', h⟩ := bind_ok h
  rcases hf : st.fold.folded ((t₁₅.rounds[t₁₅.k]!).coins.getD 0 0) ((t₁₆.rounds[t₁₆.k]!).coins.getD 0 0)
      (skipWeights ((t₁₀.rounds[t₁₀.k]!).coins.getD 0 0)) (s.2.extract 0 (st.kLog - 6)) s'.2 with err | v
  · rw [hf] at h
    obtain ⟨_, _, h₀, _⟩ := bind_ok h
    cases h₀
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨hw, -⟩ := check_bind h
  rw [forIn_range_eq] at hloop hloop'
  obtain ⟨-, -, nl, hs⟩ := ZkExec.forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (st.m - 6) 0 _ t₁₁ t₁₂ s hloop
  obtain ⟨-, -, nl', hs'⟩ := ZkExec.forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (st.kLog - 6) 0 _ t₁₇ t₁₈ s' hloop'
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, k₃.2, k₂.2, k₁.2]
  have hk₆ : t₆.k = t.k := by rw [k₆.1, k₅.1, k₄.1, k₃.1, k₂.1, k₁.1]
  have hr₁₀ : t₁₀.rounds = t.rounds := by rw [k₁₀.2, k₉.2, r₈, r₇, hr₆]
  have hk₁₀ : t₁₀.k = t.k + 2 := by rw [k₁₀.1, k₉.1, kk₈, kk₇, hk₆]
  have hr₁₁ : t₁₁.rounds = t.rounds := by rw [r₁₁, hr₁₀]
  have hk₁₁ : t₁₁.k = t.k + 3 := by rw [kk₁₁, hk₁₀]
  have hr₁₂ : t₁₂.rounds = t.rounds ∧ t₁₂.k = t.k + 3 + (st.m - 6) := by
    obtain ⟨a, b, -, -⟩ := ZkExec.forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (st.m - 6) 0 _ t₁₁ t₁₂ s hloop
    exact ⟨a.trans hr₁₁, by rw [b, hk₁₁]⟩
  have hr₁₅ : t₁₅.rounds = t.rounds := by rw [k₁₅.2, k₁₄.2, k₁₃.2, hr₁₂.1]
  have hk₁₅ : t₁₅.k = t.k + 3 + (st.m - 6) := by rw [k₁₅.1, k₁₄.1, k₁₃.1, hr₁₂.2]
  have hr₁₆ : t₁₆.rounds = t.rounds := by rw [r₁₆, hr₁₅]
  have hk₁₆ : t₁₆.k = t.k + 3 + (st.m - 6) + 1 := by rw [kk₁₆, hk₁₅]
  have hr₁₇ : t₁₇.rounds = t.rounds := by rw [r₁₇, hr₁₆]
  have hk₁₇ : t₁₇.k = t.k + 3 + (st.m - 6) + 2 := by rw [kk₁₇, hk₁₆]
  have hz : (t₁₀.rounds[t₁₀.k]!).coins.getD 0 0 = coinAt t 2 := by rw [hr₁₀, hk₁₀]
  have hα : (t₁₅.rounds[t₁₅.k]!).coins.getD 0 0 = coinAt t (3 + (st.m - 6)) := by
    rw [hr₁₅, hk₁₅, Nat.add_assoc]
  have hβ : (t₁₆.rounds[t₁₆.k]!).coins.getD 0 0 = coinAt t (4 + (st.m - 6)) := by
    rw [hr₁₆, hk₁₆, show t.k + 3 + (st.m - 6) + 1 = t.k + (4 + (st.m - 6)) by omega]
  obtain ⟨hssz, hsget⟩ := ZkExec.push_get s.2 (st.m - 6) _ hs
  obtain ⟨hs'sz, hs'get⟩ := ZkExec.push_get s'.2 (st.kLog - 6) _ hs'
  have hrho : s.2 = rhoOf st.m t := by
    refine Array.ext (by simp [rhoOf, hssz]) fun i h₁ h₂ => ?_
    rw [← getElem!_pos s.2 i h₁, hsget i (by omega), hr₁₁, hk₁₁]
    simp [rhoOf, Nat.add_assoc]
  have htsg : ∀ j < st.kLog - 6, s'.2[j]! = coinAt t (5 + (st.m - 6) + j) := fun j hj => by
    rw [hs'get j hj, hr₁₇, hk₁₇, show t.k + 3 + (st.m - 6) + 2 + j = t.k + (5 + (st.m - 6) + j) by omega]
  have hts : s'.2 = tsOf st.m st.kLog t := by
    refine Array.ext (by simp [tsOf, hs'sz]) fun i h₁ h₂ => ?_
    rw [← getElem!_pos s'.2 i h₁, htsg i (by omega)]
    simp [tsOf]
  have hnz : ∀ i < 5 + (st.m - 6) + (st.kLog - 6), (t.rounds[t.k + i]!).n ≠ 0 := by
    intro i hi
    by_cases i0 : i = 0
    · subst i0
      rw [Nat.add_zero, ← hr₆, ← hk₆, n₀]
      decide
    by_cases i1 : i = 1
    · subst i1
      rw [← hr₆, ← r₇, ← hk₆, ← kk₇, n₁]
      omega
    by_cases i2 : i = 2
    · subst i2
      rw [← hr₁₀, ← hk₁₀, n₂]
      decide
    by_cases i3 : i < 3 + (st.m - 6)
    · rw [show t.k + i = t₁₁.k + (i - 3) by omega, ← hr₁₁, nl (i - 3) (by omega)]
      decide
    by_cases i4 : i = 3 + (st.m - 6)
    · subst i4
      rw [show t.k + (3 + (st.m - 6)) = t₁₅.k by omega, ← hr₁₅, nα]
      decide
    by_cases i5 : i = 4 + (st.m - 6)
    · subst i5
      rw [show t.k + (4 + (st.m - 6)) = t₁₆.k by omega, ← hr₁₆, nβ]
      decide
    rw [show t.k + i = t₁₇.k + (i - (5 + (st.m - 6))) by omega, ← hr₁₇, nl' _ (by omega)]
    decide
  rw [hz, hα, hβ, hrho, hts] at hf
  exact ⟨v, hf, by simpa using hw, hnz⟩

end FlockSoundness.Discharge.Composed.Lincheck

namespace FlockComplete.CutFree

open Flock Transcript Replay

/-- **A returned replay's fold at coins from `L`**: on rounds whose round `j` is answered by `coin (L + j)`, a returned
replay's fold at the lincheck's `α`, `β`, the skip weights at `z`, `ρ[.. k_log − 6]` and the round coins, read at the
coins from `L`, returned a vector of width `64`: `replay_cut_ok`'s `hL`. -/
theorem replay_fold_coins (coin : ℕ → ℕ → F128) {L : ℕ} {R : Array Round} (hR : RankAt.CoinsAt coin L R)
    {st : Setup} (hm : 14 ≤ st.m) (hkk : st.kLog = st.fold.kLog) (lay : Zk.PadLayout) (p : Proof) (root d : ByteArray)
    {rp : Zk.Replayed} {t' : Transcript} (h : (Zk.replay st lay p root).run (Transcript.init R d) = .ok (rp, t')) :
    ∃ comb, st.fold.folded ((lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)).ca 0)
      ((lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)).cb 0) (skipWeights (zcCoins coin L).z)
      ((pushes (zcCoins coin L).ρ (st.m - 6)).extract 0 (st.fold.kLog - 6))
      (pushes (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)).τ (st.fold.kLog - 6)) = .ok comb ∧
        comb.size = 64 := by
  obtain ⟨comb, hf, hw, hnz⟩ := FlockSoundness.Discharge.Composed.Lincheck.replay_fold_w st lay p root hm h
  have hc : ∀ i < 5 + (st.m - 6) + (st.kLog - 6),
      FlockSoundness.Discharge.Composed.Lincheck.coinAt (Transcript.init R d) i = coin (L + i) 0 := by
    intro i hi
    have hn := hnz i hi
    change (R[0 + i]!).n ≠ 0 at hn
    show (R[0 + i]!).coins.getD 0 0 = _
    rw [Nat.zero_add] at hn ⊢
    by_cases hi' : i < R.size
    · obtain ⟨hpos, hcs⟩ := hR i hi'
      rw [hcs]
      simp [Array.getD, hpos]
    · rw [getElem!_neg R i hi'] at hn
      exact absurd rfl hn
  have hz : (zcCoins coin L).z = FlockSoundness.Discharge.Composed.Lincheck.coinAt (Transcript.init R d) 2 :=
    (hc 2 (by omega)).symm
  have hα : (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)).ca 0 =
      FlockSoundness.Discharge.Composed.Lincheck.coinAt (Transcript.init R d) (3 + (st.m - 6)) := by
    rw [hc _ (by omega)]
    show coin (L + 3 + (st.m - 6)) 0 = _
    rw [Nat.add_assoc]
  have hβ : (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)).cb 0 =
      FlockSoundness.Discharge.Composed.Lincheck.coinAt (Transcript.init R d) (4 + (st.m - 6)) := by
    rw [hc _ (by omega)]
    show coin (L + 3 + (st.m - 6) + 1) 0 = _
    congr 1
    omega
  have hρ : pushes (zcCoins coin L).ρ (st.m - 6) =
      FlockSoundness.Discharge.Composed.Lincheck.rhoOf st.m (Transcript.init R d) := by
    refine Array.ext (by rw [pushes_size]; simp [FlockSoundness.Discharge.Composed.Lincheck.rhoOf]) fun i h₁ h₂ => ?_
    rw [pushes_size] at h₁
    rw [← getElem!_pos _ i (by rw [pushes_size]; exact h₁), pushes_get _ _ i h₁]
    simp only [FlockSoundness.Discharge.Composed.Lincheck.rhoOf, Array.getElem_ofFn]
    rw [hc (3 + i) (by omega)]
    show coin (L + 3 + i) 0 = _
    rw [Nat.add_assoc]
  have hts : pushes (lcCoins coin (L + 3 + (st.m - 6)) (st.kLog - 6)).τ (st.kLog - 6) =
      FlockSoundness.Discharge.Composed.Lincheck.tsOf st.m st.kLog (Transcript.init R d) := by
    refine Array.ext (by rw [pushes_size]; simp [FlockSoundness.Discharge.Composed.Lincheck.tsOf]) fun i h₁ h₂ => ?_
    rw [pushes_size] at h₁
    rw [← getElem!_pos _ i (by rw [pushes_size]; exact h₁), pushes_get _ _ i h₁]
    simp only [FlockSoundness.Discharge.Composed.Lincheck.tsOf, Array.getElem_ofFn]
    rw [hc (5 + (st.m - 6) + i) (by omega)]
    show coin (L + 3 + (st.m - 6) + 2 + i) 0 = _
    congr 1
    omega
  rw [hz, hα, hβ, hρ, ← hkk, hts]
  exact ⟨comb, hf, hw⟩

end FlockComplete.CutFree
