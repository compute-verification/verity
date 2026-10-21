import Proofs.Flock.Claims.ZK.FinExe
import Proofs.Flock.Claims.ZK.CausalExe
import Proofs.Flock.Claims.ZK.FrameRounds

/-!
# A rep's rank check from the source, at coins every table admits

`Flock.Firewall.Finish.repCut_fw` asks (`hpiop`) that the rank check pass at the rounds the source's zerocheck and lincheck
leave, answered by `asksCut`. Those rounds carry the coins of the sends they made, whatever the masked values, so the
check reads the claim points at the coins from the rep's first send; where every table admits the opened coins, no
refusal shows (`not_refused_of_admits`), so the check passes there.

* **The rounds' coins** (`CoinsAt`, `growS`): under `asksCut coin n`, a squeeze from `l` records a round answered by
  `coin l.size`, and nothing else adds rounds or sends, so every round from the rep's first is answered by the coin at
  its send, and the rounds only grow (`Replay.Prefix`): `zerocheckCut_grows`, `lincheckCut_grows`.
* **Round `1`** (`zerocheckCut_round1`): the zerocheck's second squeeze records `m − 13` outer coins, and the rounds after
  it only grow.
* **The claim points** (`roundPoints_cut`): the zerocheck's and lincheck's `m + k_log − 6` rounds (`CutEq`'s counts) read as
  `claimPoints` at the coins from the rep's first send.
* **At admitted coins** (`Claims.ZKMal.Masked.rankOk_of_admits`, **`rank_at_admits`**): table `j`'s rep `r`'s check, after
  rep `0`'s points at rep `1` (`priorE`), passes at the opened coins, so at any coins that are the opened ones through
  the rep's rank check.
-/

namespace FlockComplete.RankAt

open Flock Transcript FlockComplete FlockDeploy With

/-! ## The rounds' coins -/

/-- **Round `j` of `R` is answered by `coin (L + j)`**, with at least one coin. -/
def CoinsAt (coin : ℕ → ℕ → F128) (L : ℕ) (R : Array Round) : Prop :=
  ∀ j < R.size, 0 < R[j]!.n ∧ R[j]!.coins = (Array.range R[j]!.n).map (coin (L + j))

theorem coinsAt_push {coin : ℕ → ℕ → F128} {L : ℕ} {R : Array Round} (hR : CoinsAt coin L R) (t : Transcript)
    (kind : UInt8) {n k : ℕ} (hn : n ≠ 0) (hk : k = L + R.size) :
    CoinsAt coin L (R.push (Replay.sent t kind n (coin k))) := by
  subst hk
  intro j hj
  rw [Array.size_push] at hj
  rw [getElem!_pos (R.push (Replay.sent t kind n (coin (L + R.size)))) j (by rw [Array.size_push]; exact hj),
    Array.getElem_push]
  by_cases h : j < R.size
  · rw [dif_pos h]
    have := hR j h
    rwa [getElem!_pos R j h] at this
  · rw [dif_neg h]
    have hj' : j = R.size := by omega
    subst hj'
    exact ⟨Nat.pos_of_ne_zero hn, rfl⟩

/-- The rounds from the log's `L`th send on are answered by their sends' coins, one round to each send. -/
def Inv (coin : ℕ → ℕ → F128) (L : ℕ) (t : Transcript) (l : Array Claims.Msg) : Prop :=
  CoinsAt coin L t.rounds ∧ l.size = L + t.rounds.size

/-- **A transcript computation that only adds rounds**, each answered by its send's coin. -/
def Grows (coin : ℕ → ℕ → F128) (L : ℕ) {α : Type} (y : TMW PS α) : Prop :=
  ∀ (t : Transcript) (l : Array Claims.Msg) (a : α) (t' : Transcript) (l' : Array Claims.Msg),
    (y.run t).go l = (.ok (a, t'), l') → Replay.Prefix t.rounds t'.rounds ∧ (Inv coin L t l → Inv coin L t' l')

/-- **The simulation of `Grows`**: a transcript computation of the pure session is related to itself when it grows. -/
def growS (coin : ℕ → ℕ → F128) (L : ℕ) : With.Sim (TMW PS) (TMW PS) where
  R _ y := Grows coin L y
  pure a t l a' t' l' h := by
    simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    exact ⟨Replay.Prefix.refl _, id⟩
  bind {α β x y f g} hxy hfg t l b t'' l'' h := by
    rw [StateT.run_bind] at h
    obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := JsonRT.go_bind_ok h
    obtain ⟨p1, i1⟩ := hxy t l a t' l' h1
    obtain ⟨p2, i2⟩ := hfg a t' l' b t'' l'' h2
    exact ⟨p1.trans p2, fun hi => i2 (i1 hi)⟩
  throw e t l a t' l' h := by
    change ((throw e : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨_, _, h1, -⟩ := JsonRT.go_bind_ok h
    cases h1
  lift v t l a t' l' h := by
    change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨b, l₁, h1, h2⟩ := JsonRT.go_bind_ok h
    cases v with
    | error _ => cases h1
    | ok c =>
      cases h1
      cases h2
      exact ⟨Replay.Prefix.refl _, id⟩

section Asks

variable (coin : ℕ → ℕ → F128) (n L : ℕ)

/-- A verifier action that keeps the rounds. -/
theorem liftTM_grows {α : Type} (x : TM α) (hx : ∀ t a t', x t = .ok (a, t') → t'.rounds = t.rounds) :
    (growS coin L).R (liftTM x : TMW PS α) (liftTM x) := by
  intro t l a t' l' h
  change (monadLift (x t) : PS _).go l = _ at h
  rcases e : x t with _ | ⟨a', t''⟩
  · rw [e] at h; cases h
  · rw [e] at h
    cases h
    have hr := hx t _ _ e
    refine ⟨by rw [hr]; exact Replay.Prefix.refl _, fun hi => ?_⟩
    unfold Inv at hi ⊢
    rw [hr]
    exact hi

theorem label_grows (s : String) : (growS coin L).R (liftTM (label s) : TMW PS Unit) (liftTM (label s)) :=
  liftTM_grows coin L _ fun _ _ _ h => by cases h; rfl

theorem absorbF_grows (v : F128) : (growS coin L).R (liftTM (absorbF v) : TMW PS Unit) (liftTM (absorbF v)) :=
  liftTM_grows coin L _ fun _ _ _ h => by cases h; rfl

theorem absorbSlice_grows (vs : Array F128) :
    (growS coin L).R (liftTM (absorbSlice vs) : TMW PS Unit) (liftTM (absorbSlice vs)) :=
  liftTM_grows coin L _ fun _ _ _ h => by cases h; rfl

theorem absorbBytes_grows (b : ByteArray) :
    (growS coin L).R (liftTM (absorbBytes b) : TMW PS Unit) (liftTM (absorbBytes b)) :=
  liftTM_grows coin L _ fun _ _ _ h => by cases h; rfl

/-- **`asksCut`'s squeeze** records, where it returns, one round answered by the coin at its send. -/
theorem squeezeP_grows (kind : UInt8) (k : ℕ) :
    (growS coin L).R (squeezeP (asksCut coin n) kind k) (squeezeP (asksCut coin n) kind k) := by
  intro t l a t' l' h
  by_cases hk : k = 0
  · change (squeezeP (asksCut coin n) kind k t).go l = _ at h
    unfold squeezeP at h
    split at h
    · simp only [JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
      obtain ⟨⟨-, rfl⟩, rfl⟩ := h
      exact ⟨Replay.Prefix.refl _, id⟩
    · contradiction
  · obtain ⟨-, -, rfl, rfl⟩ := squeezeP_cut_inv coin kind hk h
    refine ⟨Replay.Prefix.push _ _, fun hi => ⟨coinsAt_push hi.1 t kind hk hi.2, ?_⟩⟩
    have hs := hi.2
    simp only [Replay.afterSq, Array.size_push]
    omega

theorem sampleP_grows : (growS coin L).R (sampleP (asksCut coin n)) (sampleP (asksCut coin n)) := by
  unfold sampleP
  sim_walk [squeezeP_grows coin n L]

theorem zerocheckCut_grows (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : ℕ → F128) :
    (growS coin L).R (zerocheckCut (asksCut coin n) m digest cap root lay mk)
      (zerocheckCut (asksCut coin n) m digest cap root lay mk) := by
  unfold zerocheckCut sampleVecP
  sim_walk [squeezeP_grows coin n L, sampleP_grows coin n L, label_grows coin L, absorbBytes_grows coin L,
    absorbSlice_grows coin L, absorbF_grows coin L]

theorem lincheckCut_grows (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) :
    (growS coin L).R (lincheckCut (asksCut coin n) c m lay mk zc) (lincheckCut (asksCut coin n) c m lay mk zc) := by
  unfold lincheckCut
  sim_walk [sampleP_grows coin n L, label_grows coin L, absorbSlice_grows coin L, absorbF_grows coin L]

theorem run_bind_ok {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t : Transcript} {l : Array Claims.Msg} {b : β}
    {t'' : Transcript} {l'' : Array Claims.Msg} (h : ((x >>= f).run t).go l = (.ok (b, t''), l'')) :
    ∃ a t' l', (x.run t).go l = (.ok (a, t'), l') ∧ ((f a).run t').go l' = (.ok (b, t''), l'') := by
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := JsonRT.go_bind_ok h
  exact ⟨a, t', l', h1, h2⟩

theorem prefix_of_grows {α : Type} {y : TMW PS α} (hy : (growS coin L).R y y) {t t' : Transcript}
    {l l' : Array Claims.Msg} {a : α} (h : (y.run t).go l = (.ok (a, t'), l')) : Replay.Prefix t.rounds t'.rounds :=
  (hy t l a t' l' h).1

/-- **Round `1` of a zerocheck that returned** holds its `m − 13` outer coins, and its first send was before round
`n`. -/
theorem zerocheckCut_round1 {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (mk : ℕ → F128) {t t' : Transcript} {l l' : Array Claims.Msg} {res : ZerocheckProof × ZcOut}
    (h : ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l = (.ok (res, t'), l')) :
    l.size < n ∧ t'.rounds[t.rounds.size + 1]!.n = m - 13 := by
  unfold zerocheckCut at h
  rw [Replay.bind_go (Replay.liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)] at h
  rw [Replay.bind_go (Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)] at h
  rw [Replay.bind_go (Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)] at h
  rw [Replay.bind_go (Replay.liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)] at h
  rw [Replay.bind_go (Replay.liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)] at h
  rw [Replay.bind_go (Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)] at h
  obtain ⟨a₁, t₁, l₁, h1, h2⟩ := run_bind_ok h
  obtain ⟨hl, -, rfl, rfl⟩ := sampleVecP_cut_inv coin (by decide) h1
  obtain ⟨a₂, t₂, l₂, h3, h4⟩ := run_bind_ok h2
  obtain ⟨-, -, rfl, rfl⟩ := sampleVecP_cut_inv coin (show m - 13 ≠ 0 by omega) h3
  refine ⟨hl, ?_⟩
  obtain ⟨C, hC⟩ := prefix_of_grows coin 0 (by
    sim_walk [squeezeP_grows coin n 0, sampleP_grows coin n 0, label_grows coin 0, absorbBytes_grows coin 0,
      absorbSlice_grows coin 0, absorbF_grows coin 0]) h4
  rw [hC, getElem!_pos _ (t.rounds.size + 1)
      (by simp only [Replay.afterSq, Array.size_append, Array.size_push]; omega),
    Array.getElem_append_left (by simp only [Replay.afterSq, Array.size_push]; omega)]
  simp [Replay.afterSq, Array.getElem_push, Replay.sent]

/-- **The claim points at the rounds a rep's zerocheck and lincheck leave** from no rounds: the coins from the rep's
first send `l.size`. -/
theorem roundPoints_cut {m kLog : ℕ} (hm : 14 ≤ m) (hk : 6 ≤ kLog) (digest cap root : ByteArray)
    (lay : Zk.PadLayout) (mk : ℕ → F128) (fold : CircuitFold) (hfk : fold.kLog = kLog) {t₀ : Transcript}
    (ht₀ : t₀.rounds = #[]) {l : Array Claims.Msg} {zp : ZerocheckProof} {zc : ZcOut} {t₁ : Transcript}
    {l₁ : Array Claims.Msg} {lp : LincheckProof} {lc : LcOut} {t₂ : Transcript} {l₂ : Array Claims.Msg}
    (h₁ : ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t₀).go l = (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((lincheckCut (asksCut coin n) fold m lay mk zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂)) :
    MaskRank.roundPoints m kLog t₂.rounds = MaskRank.claimPoints m kLog fun j e => coin (l.size + j) e := by
  have i0 : Inv coin l.size t₀ l := ⟨fun j hj => absurd hj (by simp [ht₀]), by simp [ht₀]⟩
  obtain ⟨-, hi1⟩ := zerocheckCut_grows coin n l.size m digest cap root lay mk t₀ l _ t₁ l₁ h₁
  obtain ⟨p2, hi2⟩ := lincheckCut_grows coin n l.size fold m lay mk zc t₁ l₁ _ t₂ l₂ h₂
  obtain ⟨-, hs1⟩ := hi1 i0
  obtain ⟨hc2, hs2⟩ := hi2 (hi1 i0)
  obtain ⟨hl, hz1⟩ := zerocheckCut_round1 coin n hm digest cap root lay mk h₁
  obtain ⟨e1, e1n, -⟩ := (CutEq.zerocheckCut_eqAt coin n hm digest cap root lay mk mk t₀ (l := l) rfl (by omega)
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
  obtain ⟨e2, -, -⟩ := (CutEq.lincheckCut_eqAt coin n fold m lay mk mk zc t₁ (l := l₁) rfl (by omega)
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
  subst hfk
  have hsz : t₂.rounds.size = m + fold.kLog - 6 := by omega
  have hr1 : t₂.rounds[1]!.n = m - 13 := by
    obtain ⟨C, hC⟩ := p2
    have h1s : 1 < t₁.rounds.size := by omega
    rw [hC, getElem!_pos _ 1 (by rw [Array.size_append]; omega), Array.getElem_append_left h1s,
      ← getElem!_pos t₁.rounds 1 h1s]
    simpa [ht₀] using hz1
  have key : ∀ j < t₂.rounds.size, ∀ e, e < t₂.rounds[j]!.n →
      (t₂.rounds[j]!.coins).getD e 0 = coin (l.size + j) e := by
    intro j hj e he
    rw [(hc2 j hj).2, Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range]
    simp [he]
  unfold MaskRank.roundPoints
  exact ZkTables.claimPoints_congr_lt (by omega) hk (fun j hj => key j (by omega) 0 (hc2 j (by omega)).1)
    (fun e he => key 1 (by omega) e (by omega))

end Asks

end FlockComplete.RankAt

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem tab_bounds {x : Msg} (hx : Live x) {j : ℕ} (hj : j < (stsOf x).size) :
    14 ≤ (stsOf x)[j]!.m ∧ 6 ≤ (stsOf x)[j]!.c.kLog := by
  obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hjJ : j < jOf x := hJ ▸ hj
  have hst : (stsOf x)[j]! = stAt x ⟨j, hjJ⟩ := getElem!_pos (stsOf x) j hj
  refine ⟨?_, ?_⟩
  · rw [hst]
    exact le_trans (by norm_num) (hall j hj).1
  · rw [hst]
    exact (zkTab x ⟨j, hjJ⟩).wf.1.hk6

/-- **Where every table admits the opened coins, every rank check passes there**: table `j`'s rep `r`'s, after rep `0`'s
points at rep `1`. -/
theorem rankOk_of_admits {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) {j r : ℕ} (hj : j < (stsOf x).size) (hr : r < 2) :
    FlockComplete.MaskRank.maskRankOk (priorE c j r ++ ptsOpen c (stsOf x)[j]! (2 * j + r))
      (Composed.maskWordsOf (stsOf x)[j]!) = true := by
  obtain ⟨hm, hk6⟩ := tab_bounds hx hj
  by_contra hb
  exact not_refused_of_admits hx hA (rankSlot x j 0 + rankSlot x j r)
    (.inr ⟨j, hj, r, hr, .inl ⟨hm, hk6, by omega, by omega, Bool.eq_false_iff.2 hb⟩⟩)

/-- **A rep's rank check from the source, at coins every table admits** (`repCut_fw`'s `hpiop`, its first half): table
`j`'s rep `r`, its zerocheck and lincheck run from no rounds at its first send `repBase x (2j + r)` and answered by
coins that are the opened ones through its rank check, leaves rounds at which the check, after rep `0`'s points at rep
`1` (`priorE`), passes over the table's mask words. -/
theorem rank_at_admits {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) {j r : ℕ} (hj : j < (stsOf x).size) (hr : r < 2)
    (coin : ℕ → ℕ → _root_.Flock.F128) (n : ℕ)
    (hcoin : ∀ k < rankSlot x j r, ∀ e, coin k e = slotAt c k e) (st : _root_.Flock.Setup)
    (hsm : st.m = (stsOf x)[j]!.m) (hsk : st.kLog = (stsOf x)[j]!.c.kLog) (hfk : st.fold.kLog = st.kLog)
    (cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → _root_.Flock.F128)
    {t₀ : _root_.Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Msg} (hl : l.size = repBase x (2 * j + r))
    {zp : _root_.Flock.ZerocheckProof} {zc : _root_.Flock.ZcOut} {t₁ : _root_.Flock.Transcript} {l₁ : Array Msg}
    {lp : _root_.Flock.LincheckProof} {lc : _root_.Flock.LcOut} {t₂ : _root_.Flock.Transcript} {l₂ : Array Msg}
    (h₁ : ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest cap root lay mk).run t₀).go l =
      (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m lay mk zc).run t₁).go l₁ =
      (.ok ((lp, lc), t₂), l₂)) :
    FlockComplete.MaskRank.maskRankOk (priorE c j r ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds)
      (Composed.maskWordsOf (stsOf x)[j]!) = true := by
  obtain ⟨hm, hk6⟩ := tab_bounds hx hj
  rw [FlockComplete.RankAt.roundPoints_cut coin n (m := st.m) (kLog := st.kLog) (by omega) (by omega) st.digest cap
    root lay mk st.fold hfk ht₀ h₁ h₂]
  have hpts : FlockComplete.MaskRank.claimPoints st.m st.kLog (fun j' e => coin (l.size + j') e) =
      ptsOpen c (stsOf x)[j]! (2 * j + r) := by
    unfold ptsOpen
    rw [← hsm, ← hsk]
    refine FlockComplete.ZkTables.claimPoints_congr_lt (by omega) (by omega) (fun j' hj' => ?_) (fun e he => ?_)
    · rw [hl]
      exact hcoin _ (by unfold rankSlot; omega) 0
    · rw [hl]
      exact hcoin _ (by unfold rankSlot; omega) e
  rw [hpts]
  exact rankOk_of_admits hx hA hj hr

end Masked

end Claims.ZKMal
