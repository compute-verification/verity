import Proofs.Flock.SessionLoop.FinRepFinal
import Proofs.Flock.SessionLoop.FinRepXs
import Proofs.Flock.SessionLoop.Tables
import Proofs.Flock.Claims.ZK.FinishRank
import Proofs.Flock.Claims.ZK.FinishSwPts
import Proofs.Flock.Claims.ZK.LinkFft

/-!
# (c.3)'s `FinRepL0`: level 0's opened rows and salts

`Loop.FinRepL0` asks that the run from the views and the session's rep open level 0 at the same rows and salts. Both are
answered by the opened coins (`asksCut`) from a fresh transcript at rep `k`'s first round, and both send one body for
each round they take through the PIOP (`RankAt.Paces`, `RankAt.piopCut_counts`), `piopZk` of them, so both open level 0
at the coins of Ligerito's round `k₀ + 8` after it (`FinRep.repCut_opens0`, `FinRep.repFwC_open0`), the slot `repSlot`
names: at `q0At`'s positions. Each of those is opened (`FinRep.q0At_opened`), and there the two commitments agree:

* the views' commitment (`commitsView`'s `k / 2`-th) holds the views' level-0 rows and salts
  (`FlockComplete.commit0Cut_rows`). The switched views' level-0 rows and salts are the honest views'
  (`Coined.swExe_l0`, `Loop.finSrc_l0`): the switched prover moves the pads alone. At an opened position those are
  `encodeRowsZk`'s row and the tape's fitted level-0 salt (`FinRep.srcV_open0`);
* the session's commitment is the loop's table's (`tb.cm0`), `commitOf` of the `Level0` the loop drew, whose rows are
  `Link.rows` of `Masked.linkIn` and salts its fitted salts (`Loop.Level0Src.realOf_rows0`), the same row
  (`Link.rows_eq`) and salt.

**`Loop.finRepL0`**: at a statement in the masked scope with a witness, where the loop draws the tape's level 0
(`Loop.Level0Src`), `FinRepL0` holds. It reads no worker's values.
-/

namespace FlockComplete.RankAt

open Flock Transcript FlockComplete FlockDeploy With

/-! ## Bodies and rounds -/

/-- **A transcript computation that sends one body for each round it takes**, where it returns. -/
def Paces {α : Type} (y : TMW PS α) : Prop :=
  ∀ (t : Transcript) (l : Array Claims.Msg) (a : α) (t' : Transcript) (l' : Array Claims.Msg),
    (y.run t).go l = (.ok (a, t'), l') → l'.size + t.k = l.size + t'.k

/-- **The simulation of `Paces`**: a transcript computation of the pure session is related to itself when it paces. -/
def paceS : With.Sim (TMW PS) (TMW PS) where
  R _ y := Paces y
  pure a t l a' t' l' h := by
    simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    rfl
  bind {α β x y f g} hxy hfg t l b t'' l'' h := by
    rw [StateT.run_bind] at h
    obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := JsonRT.go_bind_ok h
    have p1 := hxy t l a t' l' h1
    have p2 := hfg a t' l' b t'' l'' h2
    omega
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
      rfl

/-- A verifier action that keeps the round count. -/
theorem liftTM_paces {α : Type} (x : TM α) (hx : ∀ t a t', x t = .ok (a, t') → t'.k = t.k) :
    paceS.R (liftTM x : TMW PS α) (liftTM x) := by
  intro t l a t' l' h
  change (monadLift (x t) : PS _).go l = _ at h
  rcases e : x t with _ | ⟨a', t''⟩
  · rw [e] at h; cases h
  · rw [e] at h
    cases h
    have hr := hx t _ _ e
    omega

theorem label_paces (s : String) : paceS.R (liftTM (label s) : TMW PS Unit) (liftTM (label s)) :=
  liftTM_paces _ fun _ _ _ h => by cases h; rfl

theorem absorbF_paces (v : F128) : paceS.R (liftTM (absorbF v) : TMW PS Unit) (liftTM (absorbF v)) :=
  liftTM_paces _ fun _ _ _ h => by cases h; rfl

theorem absorbSlice_paces (vs : Array F128) :
    paceS.R (liftTM (absorbSlice vs) : TMW PS Unit) (liftTM (absorbSlice vs)) :=
  liftTM_paces _ fun _ _ _ h => by cases h; rfl

theorem absorbBytes_paces (b : ByteArray) :
    paceS.R (liftTM (absorbBytes b) : TMW PS Unit) (liftTM (absorbBytes b)) :=
  liftTM_paces _ fun _ _ _ h => by cases h; rfl

section Asks

variable (coin : ℕ → ℕ → F128) (n : ℕ)

/-- **`asksCut`'s squeeze** sends one body and takes one round, where it returns. -/
theorem squeezeP_paces (kind : UInt8) (k : ℕ) :
    paceS.R (squeezeP (asksCut coin n) kind k) (squeezeP (asksCut coin n) kind k) := by
  intro t l a t' l' h
  by_cases hk : k = 0
  · change (squeezeP (asksCut coin n) kind k t).go l = _ at h
    unfold squeezeP at h
    split at h
    · simp only [JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
      obtain ⟨⟨-, rfl⟩, rfl⟩ := h
      rfl
    · contradiction
  · obtain ⟨-, -, rfl, rfl⟩ := squeezeP_cut_inv coin kind hk h
    simp only [Replay.afterSq, Array.size_push]
    all_goals omega

theorem sampleP_paces : paceS.R (sampleP (asksCut coin n)) (sampleP (asksCut coin n)) := by
  unfold sampleP
  sim_walk [squeezeP_paces coin n]

theorem zerocheckCut_paces (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : ℕ → F128) :
    paceS.R (zerocheckCut (asksCut coin n) m digest cap root lay mk)
      (zerocheckCut (asksCut coin n) m digest cap root lay mk) := by
  unfold zerocheckCut sampleVecP
  sim_walk [squeezeP_paces coin n, sampleP_paces coin n, label_paces, absorbBytes_paces, absorbSlice_paces,
    absorbF_paces]

theorem lincheckCut_paces (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) :
    paceS.R (lincheckCut (asksCut coin n) c m lay mk zc) (lincheckCut (asksCut coin n) c m lay mk zc) := by
  unfold lincheckCut
  sim_walk [sampleP_paces coin n, label_paces, absorbSlice_paces, absorbF_paces]

theorem ringSwitchCut_paces (sv : ℕ → Array F128) (claims : Array Claim) :
    paceS.R (ringSwitchCut (asksCut coin n) sv claims) (ringSwitchCut (asksCut coin n) sv claims) := by
  unfold ringSwitchCut sampleVecP
  sim_walk [squeezeP_paces coin n, label_paces, absorbSlice_paces]

/-- **The cut's PIOP sends `piopZk` bodies and takes as many rounds**, where it returns, whatever its source: the
zerocheck's `3 + (m − 6)` (`zerocheckCut_at`), the lincheck's `3 + (k_log − 6)` (`lincheckCut_at`), none for the rank
check (`rankCheckP_go`) and the ring switch's one per claim and one (`ringSwitchCut_sws`), each a round. -/
theorem piopCut_counts (st : Setup) (lay : Zk.PadLayout) (src : RepSrc) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) {t t' : Transcript}
    {l l' : Array Claims.Msg} {o : ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched}
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t).go l = (.ok (o, t'), l')) :
    l'.size = l.size + piopZk st extras ∧ l'.size + t.k = l.size + t'.k := by
  unfold piopCut at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, hA⟩ := run_bind_ok h
  try dsimp only at hA
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, hB⟩ := run_bind_ok hA
  try dsimp only at hB
  obtain ⟨u, t₃, l₃, h₃, hC⟩ := run_bind_ok hB
  obtain ⟨rfl, rfl⟩ := Replay.rankCheckP_go h₃
  try dsimp only at hC
  obtain ⟨⟨ss, T, sws⟩, t₄, l₄, h₄, hD⟩ := run_bind_ok hC
  obtain ⟨hE, rfl⟩ := Prod.mk.inj hD
  obtain ⟨-, rfl⟩ := Prod.mk.inj (Except.ok.inj hE)
  have s1 := (zerocheckCut_at coin n hm _ _ _ _ _ h₁).2
  have s2 := (lincheckCut_at coin n _ _ _ _ _ h₂).2
  have s3 := (ringSwitchCut_sws coin n _ _ (by simp [abcClaims]) h₄).2.1
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    simp [abcClaims, Nat.add_comm]
  have p1 := zerocheckCut_paces coin n _ _ _ _ _ _ _ _ _ _ _ h₁
  have p2 := lincheckCut_paces coin n _ _ _ _ _ _ _ _ _ _ h₂
  have p3 := ringSwitchCut_paces coin n _ _ _ _ _ _ _ h₄
  unfold piopZk
  constructor <;> omega

end Asks

end FlockComplete.RankAt

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

namespace FinRep

open FlockComplete FlockComplete.Replay FlockComplete.Replay.Lg FlockComplete.L0 Flock.With

/-- **The cut's rep opens level 0 at round `k₀ + 8` of its Ligerito**, whatever its source: answered by `asksCut` from a
fresh transcript, its PIOP takes `piopZk` rounds with as many bodies (`RankAt.piopCut_counts`), and its Ligerito opens
its level-0 commitment `cm0` there (`ligeritoCut_open0`). -/
theorem repCut_opens0 {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    (hsh : ByteArray → ByteArray) (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (rep : ℕ) (dom : ByteArray)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (ic : InnerCommit) (src : RepSrc)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ)
    (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (hq : (sch.levels.getD 0 default).q ≠ 0)
    {t t' : _root_.Flock.Transcript} (ht : t.k = 0) {l l' : Array Msg} {P : _root_.Flock.Zk.ZkProof}
    (hrun : ((repCut (asksCut coin n) (_root_.Flock.Hashes.ofSha512 hsh) st sch rep dom cm0 extras ic src prior
      words).run t).go l = (.ok (P, t'), l')) :
    P.proof.ligerito.initialProof = openP st.merkle cm0 (sch.levels.getD 0 default)
      (_root_.Flock.positions (sch.levels.getD 0 default) ((Array.range (sch.levels.getD 0 default).q).map
        (coinOf (resp (l.size + (piopZk st extras + sch.k0 + 8)))))) := by
  unfold repCut at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨zp, lp, ss, T, sws⟩, t₃⟩, l₃, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨hl₃, ht₃⟩ := FlockComplete.RankAt.piopCut_counts coin n st _ src _ _ extras prior words hm h1
  dsimp only at h2
  rw [StateT.run_bind] at h2
  obtain ⟨⟨⟨lg, zl, rec⟩, t₄⟩, l₄, h3, h4⟩ := JsonRT.go_bind_ok h2
  have h0 := ligeritoCut_open0 resp hsh st.merkle sch _ cm0 _ _ src l.size hn hk0 hq
    (by omega) (ligeritoCut_cut (n := n) hr _ _ _ _ _ _ _ _ t₃ l₃ _ l₄ h3)
  dsimp only at h4
  rw [bind_go (show ((get : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Transcript).run t₄).go l₄ =
    (.ok (t₄, t₄), l₄) from rfl)] at h4
  obtain ⟨⟨rp, x⟩, -, h4⟩ := liftV_bind_ok h4
  dsimp only at h4
  obtain ⟨lgc, -, h4⟩ := liftV_bind_ok h4
  rw [StateT.run_bind] at h4
  obtain ⟨⟨inner, t₅⟩, l₅, -, h4⟩ := JsonRT.go_bind_ok h4
  obtain ⟨h5, -⟩ := Prod.mk.inj h4
  obtain ⟨hP, -⟩ := Prod.mk.inj (Except.ok.inj h5)
  subst hP
  refine h0.trans ?_
  rw [show t₃.k = piopZk st extras by omega]

/-- **The firewall's rep opens level 0 at round `k₀ + 8` of its Ligerito**: answered by `asksCut` from a fresh
transcript, its zerocheck, lincheck and ring switch take `piopZk` rounds with as many bodies (`zerocheckCut_at`,
`lincheckCut_at`, `ringSwitchCut_sws`, `RankAt.Paces`), and its Ligerito opens `cm0` at the query of the round
`k₀ + 6` after its pair's coins (`L0.ligeritoPZk_opens`, through `ligeritoPZk_cut`); the replay, the constraints and
the inner proof leave the Ligerito proof as it is. -/
theorem repFwC_open0 {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk)
    (ic : InnerCommit) (vals : Array _root_.Flock.F128) (extra lanePad : ℕ → ℕ → _root_.Flock.F128)
    (salt : ℕ → ℕ → ByteArray) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t t' : _root_.Flock.Transcript} (ht : t.k = 0)
    {l l' : Array Msg} {P : _root_.Flock.Zk.ZkProof}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (asksCut coin n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st sch tPad rep dom g cm0 extras pads ic vals extra
      lanePad salt).run t).go l = (.ok (P, t'), l')) :
    P.proof.ligerito.initialProof = openP _root_.Flock.MerkleScheme.hm96Sha512 cm0 (sch.levels.getD 0 default)
      (_root_.Flock.positions (sch.levels.getD 0 default) ((Array.range (sch.levels.getD 0 default).q).map
        (coinOf (resp (l.size + (piopZk st extras + sch.k0 + 8)))))) := by
  dsimp only [_root_.Flock.Firewall.Finish.repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  rw [_root_.Flock.Firewall.Finish.get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hr₆⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hg
  obtain ⟨-, ht₆, hl₆⟩ := _root_.Flock.Firewall.Finish.liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hr₆
  obtain ⟨lgc, t₇, l₇, h₇, hc⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hr₆
  obtain ⟨-, ht₇, hl₇⟩ := _root_.Flock.Firewall.Finish.liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hc
  split at hc
  · obtain ⟨_, _, _, h', _⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hc
    rw [_root_.Flock.Firewall.Finish.throw_run_go] at h'
    cases h'
  have hpl : P.proof.ligerito = lg ∧ P.level0 = zl := by
    obtain ⟨_, _, _, -, hp⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hc
    simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hp
    obtain ⟨⟨rfl, -⟩, -⟩ := hp
    exact ⟨rfl, rfl⟩
  have s1 := (zerocheckCut_at coin n hm _ _ _ _ _ h₁).2
  have s2 := (lincheckCut_at coin n _ _ _ _ _ h₂).2
  have s3 := (ringSwitchCut_sws coin n _ _ (by simp [_root_.Flock.abcClaims]) h₃).2.1
  have hsz : (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    simp [_root_.Flock.abcClaims, Nat.add_comm]
  have p1 := FlockComplete.RankAt.zerocheckCut_paces coin n _ _ _ _ _ _ _ _ _ _ _ h₁
  have p2 := FlockComplete.RankAt.lincheckCut_paces coin n _ _ _ _ _ _ _ _ _ _ h₂
  have p3 := FlockComplete.RankAt.ringSwitchCut_paces coin n _ _ _ _ _ _ _ h₃
  have h0 := (L0.ligeritoPZk_opens resp _root_.Flock.Sha512.hash _root_.Flock.MerkleScheme.hm96Sha512 sch _ cm0 _ _ _ _
    rep tPad extra lanePad salt (fun _ _ => 0) l.size hn hk0 hL hq (by omega)
    (Masked.FinRep.ligeritoPZk_cut (n := n) hr _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ t₃ l₃ _ l₄ h₄)).1
  rw [hpl.1]
  refine h0.trans ?_
  have e : t₃.k + 2 + qOff sch 0 = piopZk st extras + sch.k0 + 8 := by
    unfold qOff
    rw [ite_eq_left (by omega)]
    unfold piopZk
    omega
  simp only [posAt, cfOf, e]

end FinRep

end Masked

end Claims.ZKMal

namespace FlockComplete

open Flock

/-- **Level 0's commitment from a source holds the source's rows and salts.** -/
theorem commit0Cut_rows {ms : MerkleScheme} {lv : Level} {src : TabSrc} {cm : Commit}
    (h : (commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes V) ms lv src : V Commit) = .ok cm) :
    cm.rows = (Array.range (2 ^ lv.d)).map src.row0 ∧ cm.salts = (Array.range (2 ^ lv.d)).map src.salt0 := by
  have e := commit0Cut_leaf0 (M := V) (Hashes.ofSha512 Sha512.hash) ms lv (src := src) (src' := src) rfl
  rw [h] at e
  have e2 : (Except.ok cm : V Commit) = Except.ok { cm with
      rows := (Array.range (2 ^ lv.d)).map src.row0
      salts := (Array.range (2 ^ lv.d)).map src.salt0 } := e
  have e3 := Except.ok.inj e2
  exact ⟨congrArg Commit.rows e3, congrArg Commit.salts e3⟩

end FlockComplete

namespace FlockSoundness.ZK.Table

section

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

theorem toBase_l0 (v : T.onVals.View) :
    ((T.toBase v).rows, (T.toBase v).xrows, (T.toBase v).salts0) = (v.rows, v.xrows, v.salts0) := rfl

end

section

variable {F K W Salt B D Hi VL C : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
  {T : Table F K W Salt (B × Salt) D Hi VL}

/-- **A real view's level-0 rows and salts read the witness alone**, not the clear values. -/
theorem Hm96.onVals_viewR_l0 (M : T.Hm96 C) (w : W) (v v' : T.Vals) (ω : T.Rand) (ys : T.Hid → Salt) :
    ((M.onVals.viewR (w, v) ω ys).rows, (M.onVals.viewR (w, v) ω ys).xrows, (M.onVals.viewR (w, v) ω ys).salts0) =
      ((M.onVals.viewR (w, v') ω ys).rows, (M.onVals.viewR (w, v') ω ys).xrows,
        (M.onVals.viewR (w, v') ω ys).salts0) := by
  unfold FlockSoundness.ZK.Table.Hm96.viewR FlockSoundness.ZK.Table.viewWith FlockSoundness.ZK.Table.view
  try dsimp only
  try rfl

/-- **Moving the pads leaves a real view's level-0 rows and salts**: `Table.shift` moves the pads alone. -/
theorem Hm96.viewR_shift_l0 (M : T.Hm96 C) (p q r : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    ((M.viewR r (T.shift p q ω) ys).rows, (M.viewR r (T.shift p q ω) ys).xrows,
        (M.viewR r (T.shift p q ω) ys).salts0) =
      ((M.viewR r ω ys).rows, (M.viewR r ω ys).xrows, (M.viewR r ω ys).salts0) := by
  unfold FlockSoundness.ZK.Table.Hm96.viewR FlockSoundness.ZK.Table.viewWith FlockSoundness.ZK.Table.view
    FlockSoundness.ZK.Table.shift
  try dsimp only
  try rfl

end

end FlockSoundness.ZK.Table

/-- **A switched view's level-0 rows and salts are the honest view's**: it is the view at the workers' values or the
view with the pads moved (`Hm96.swViewR_eq`), and level 0's rows and salts read the witness, the level-0 randomness and
the seeds alone, whatever the values and the pads. -/
theorem FlockSoundness.Discharge.ZkView.Coined.swExe_l0 {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (ω : T.tab.Rand) (ys : T.tab.Hid → FlockSoundness.Discharge.ZkView.Salt) :
    (T.swExe w v ω ys).rows = (T.viewExe w ω ys).rows ∧ (T.swExe w v ω ys).xrows = (T.viewExe w ω ys).xrows ∧
      (T.swExe w v ω ys).salts0 = (T.viewExe w ω ys).salts0 := by
  have e1 := congrArg (fun V => ((T.tab.toBase V).rows, (T.tab.toBase V).xrows, (T.tab.toBase V).salts0))
    (FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, v) (w, T.tab.vals) ω ys)
  have e2 := congrArg (fun V : T.tab.View => (V.rows, V.xrows, V.salts0))
    ((T.hm96.toBase_viewR w ω ys).trans (FlockSoundness.Discharge.ZkView.Coined.viewExe_eq T w ω ys).symm)
  have key : ((T.swExe w v ω ys).rows, (T.swExe w v ω ys).xrows, (T.swExe w v ω ys).salts0) =
      ((T.viewExe w ω ys).rows, (T.viewExe w ω ys).xrows, (T.viewExe w ω ys).salts0) := by
    refine e1.trans (Eq.trans ?_ e2)
    split_ifs
    · exact (FlockSoundness.ZK.Table.toBase_l0 T.tab _).trans
        ((FlockSoundness.ZK.Table.Hm96.onVals_viewR_l0 T.hm96 w v T.tab.vals ω ys).trans
          (FlockSoundness.ZK.Table.toBase_l0 T.tab _).symm)
    · exact (FlockSoundness.ZK.Table.toBase_l0 T.tab _).trans
        ((FlockSoundness.ZK.Table.Hm96.viewR_shift_l0 T.hm96.onVals (w, v) (w, T.tab.vals) (w, T.tab.vals) ω
          ys).trans (FlockSoundness.ZK.Table.toBase_l0 T.tab _).symm)
  exact ⟨congrArg (·.1) key, congrArg (·.2.1) key, congrArg (·.2.2) key⟩

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge CoinLayer

/-- **A source reads its view's level-0 rows and salts**: views with the same level-0 rows and salts at table `i` give
sources with the same level-0 rows and salts there. -/
theorem tabSrcView_l0 (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (v v' : ∀ j, ((tabC P x j).T (cz j)).tab.View) {i : ℕ} (hi : i < jOf x)
    (h1 : (v ⟨i, hi⟩).rows = (v' ⟨i, hi⟩).rows) (h2 : (v ⟨i, hi⟩).xrows = (v' ⟨i, hi⟩).xrows)
    (h3 : (v ⟨i, hi⟩).salts0 = (v' ⟨i, hi⟩).salts0) :
    (tabSrcView P x cz v i).row0 = (tabSrcView P x cz v' i).row0 ∧
      (tabSrcView P x cz v i).salt0 = (tabSrcView P x cz v' i).salt0 := by
  unfold tabSrcView
  simp only [dite_eq_left hi, h1, h2, h3, and_self]

end Claims.ZKMal.Masked

namespace Flock.Firewall.Finish

/-- **A table from a level 0 keeps `commitOf` of it** as its level-0 commitment. -/
theorem tabOfLevel0_commitOf {st : _root_.Flock.Setup} {l : _root_.Flock.Firewall.Level0.Level0}
    {t : _root_.Flock.Firewall.Finish.Tab} (h : _root_.Flock.Firewall.Finish.tabOfLevel0 st l = .ok t) :
    t.cm0 = commitOf l := by
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0 at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  split at h
  · cases h
  rcases hs : _root_.Flock.Zk.schedule st.m with e | ⟨sch, tPad⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    simp only at h
    split at h
    · cases h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

end Flock.Firewall.Finish

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **The loop's table keeps `commitOf` of the tape's level 0**: where `src` draws the tape's level 0, at rep stream
`2j + r` the loop's table's level-0 commitment is `commitOf` of a `Level0` on `Masked.linkIn`, at level 0 of table
`j`'s `--zk` schedule and its `t_pad`, whose rows are `Link.rows` of its own inputs (`Level0.ofDraws_ok`). -/
theorem Loop.Level0Src.realOf_rows0 {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.Level0Src src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    ∃ l : _root_.Flock.Firewall.Level0.Level0, l.toIn = Masked.linkIn x w t j ∧
      tb.cm0 = _root_.Flock.Firewall.Finish.commitOf l ∧ l.lv = Masked.lv0 x j ∧ l.tPad = (Masked.zkTab x j).tPad ∧
      l.rows = _root_.Flock.Firewall.Link.rows l.lv l.tPad l.toIn.g l.toIn.extra l.toIn.lanePad := by
  obtain ⟨l, hl, hin⟩ := hS ν t j r
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  unfold Loop.realOf at h
  rw [hsx] at h
  dsimp only at h
  unfold Loop.level0At at hl
  rcases hs : src x w (ν, t) (2 * j.val + r.val) with e | ⟨g, d, hi'⟩
  · rw [hs] at hl
    cases hl
  · rw [hs] at h hl
    simp only [bind, Except.bind] at h hl
    rw [hl] at h
    simp only at h
    split at h
    · cases h
    · rename_i tb' htb
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      obtain ⟨hm, -, ⟨s, hz, hlv⟩, hrows, -, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl
      have hm' : l.m = (Masked.stAt x j).m := by
        rw [show l.m = l.toIn.m from rfl, hin]
        rfl
      rw [hm] at hm'
      rw [hm'] at hz
      obtain ⟨hs', ht'⟩ := Prod.mk.inj (Except.ok.inj (hz.symm.trans (Masked.zkTab x j).hz))
      have hlv' : l.lv = Masked.lv0 x j := by rw [hlv, hs']; rfl
      exact ⟨l, hin, _root_.Flock.Firewall.Finish.tabOfLevel0_commitOf htb, hlv', ht', hrows⟩

namespace Loop

theorem l0_go_of_fst {α β : Type} {z : α × β} {a : α} (h : z.1 = a) : z = (a, z.2) := Prod.ext h rfl

open FlockSoundness.Discharge in
/-- **The views' level-0 rows and salts are the honest views'**: table `i`'s source from the switched views reads the
switched view's level-0 rows and salts (`Masked.tabSrcView_l0`), which are the honest view's (`Coined.swExe_l0`). -/
theorem finSrc_l0 (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) {i : ℕ} (hi : i < Masked.jOf x) :
    (finSrc vs x w ν t ops i).row0 = (Masked.srcV x (finCoins x ops) w t i).row0 ∧
      (finSrc vs x w ν t ops i).salt0 = (Masked.srcV x (finCoins x ops) w t i).salt0 := by
  obtain ⟨h1, h2, h3⟩ := ZkView.Coined.swExe_l0 ((finS.C x ⟨i, hi⟩).T (finS.cz x (finCoins x ops) ⟨i, hi⟩))
    (finS.wt x w ⟨i, hi⟩) (vs x w ν (finCoins x ops) ⟨i, hi⟩) (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).1
    (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).2
  exact Masked.tabSrcView_l0 Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
    (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)
    (Masked.viewsE x (finCoins x ops) w t) hi h1 h2 h3

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinRepL0` holds where the loop draws the tape's level 0**: at a statement in the masked scope with a witness
(`Loop.Level0Src`), the run from the views and the session's rep open level 0 at `q0At`'s positions
(`FinRep.repCut_opens0`, `FinRep.repFwC_open0`, at `repSlot`'s slot), where the views' commitment
(`commit0Cut_rows`, `finSrc_l0`, `FinRep.srcV_open0`) and the loop's table's (`Level0Src.realOf_rows0`) hold the same
row and salt. -/
theorem finRepL0 (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hS : Level0Src src x w) : FinRepL0 vs L₀ src real wk x w := by
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hRep ic' p tr hic'
    hrun p' l' hv _
  show p'.proof.ligerito.initialProof.rows = p.proof.ligerito.initialProof.rows ∧
    p'.proof.ligerito.initialProof.salts = p.proof.ligerito.initialProof.salts
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    rw [ite_eq_left hx]
  have hj : k / 2 < Masked.jOf x := by omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by decide)
  have hi' : k / 2 < (Masked.stsOf x).size := Masked.lt_of_jOf ⟨k / 2, hj⟩
  have hjT : k / 2 < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hi'
  have hst : (Masked.stsOf x)[k / 2]! = Masked.stAt x ⟨k / 2, hj⟩ := getElem!_pos (Masked.stsOf x) (k / 2) hi'
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! =
      _root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) hjT, getElem!_pos (Masked.stsOf x) (k / 2) hi']
    simp [FlockDeploy.tabsOf]
  -- the table, its schedule and its bounds
  obtain ⟨hst2, hsch2⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs hk hRep
  have hmsF := (finRepPads L₀ src real wk x w hsc ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q f hRep).1
  have hmsV : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hst2]
    exact hmsF
  obtain ⟨schs', hs', hall⟩ := Masked.schedsPZk_live hx
  have hss : schs' = schs := Except.ok.inj (hs'.symm.trans hschs)
  have hsj : schs[k / 2]! = ((Masked.zkTab x ⟨k / 2, hj⟩).sch, (Masked.zkTab x ⟨k / 2, hj⟩).tPad) := by
    rw [← hss]
    exact hall ⟨k / 2, hj⟩
  have hsj1 : schs[k / 2]!.1 = (Masked.zkTab x ⟨k / 2, hj⟩).sch := congrArg Prod.fst hsj
  have hsc2 : tb.sch = (Masked.zkTab x ⟨k / 2, hj⟩).sch := congrArg Prod.fst (hsch2.symm.trans hsj)
  obtain ⟨hb1, hb2⟩ := Masked.tab_bounds hx hi'
  have hm14S : 14 ≤ (Masked.stAt x ⟨k / 2, hj⟩).m := by rw [← hst]; exact hb1
  have hm14V : 14 ≤ (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m := by rw [htab]; exact hb1
  have hm14F : 14 ≤ tb.st.m := by rw [← hst2]; exact hm14V
  obtain ⟨-, -, hsm, hk0, -, hdim, hlv, -, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (Masked.zkTab x ⟨k / 2, hj⟩).hz
  obtain ⟨hn7, hL2, -, -, -, -, -⟩ := hdim
  have hn : (Masked.zkTab x ⟨k / 2, hj⟩).sch.n ≠ 0 := by rw [hn7, hsm]; omega
  have hq0 : ((Masked.zkTab x ⟨k / 2, hj⟩).sch.levels.getD 0 default).q ≠ 0 := (hlv 0 (by omega)).q_ne_zero
  -- the extra claims and the slot
  have hxs := finRepXs L₀ src real wk x w hsc hR ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk extras hxe tb hi q f hRep
  have hpz : FlockComplete.piopZk tb.st (_root_.Flock.Firewall.Finish.claimsOf q.xs) =
      FlockComplete.piopZk tb.st extras := by
    unfold FlockComplete.piopZk
    rw [← hxs]
    simp [_root_.Flock.Firewall.Finish.claimsOf]
  have hes : extras.size = spec.points * (Masked.stsOf x)[k / 2]!.regions.size := by
    have hex := hxe
    rw [htab] at hex
    rw [FlockComplete.JsonRT.extraClaims_size hex]
    simp [Nat.mul_comm, viewPoints]
  have hslot : l.size + (FlockComplete.piopZk (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! extras +
      (Masked.zkTab x ⟨k / 2, hj⟩).sch.k0 + 8) = Masked.repSlot x (2 * (k / 2) + k % 2) (fun sch => sch.k0 + 8) := by
    have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * (k / 2) + k % 2) / 2]!.m =
        .ok ((Masked.zkTab x ⟨k / 2, hj⟩).sch, (Masked.zkTab x ⟨k / 2, hj⟩).tPad) := by
      rw [show (2 * (k / 2) + k % 2) / 2 = k / 2 by omega, hst]
      exact (Masked.zkTab x ⟨k / 2, hj⟩).hz
    rw [Masked.repSlot_eq hsx hz, show 2 * (k / 2) + k % 2 = k by omega, hl, Masked.repBase_eq hsx, htab]
    have e6 : (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!).fold.kLog = (Masked.stsOf x)[k / 2]!.c.kLog :=
      rfl
    have e7 : (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!).m = (Masked.stsOf x)[k / 2]!.m := rfl
    unfold FlockComplete.piopZk FlockComplete.piopRoundsZk
    rw [e6, e7, hes]
    omega
  -- both runs open level 0 at the slot's coins
  have hv' := hv
  unfold viewRep at hv'
  have hP := Masked.FinRep.repCut_opens0 (coin := fun k i => Masked.slotAt (finCoins x ops) k i) (n := refK x)
    (resp := Masked.FinRep.respOf (finCoins x ops)) (fun k => Masked.FinRep.coinOf_respOf _ k)
    _ _ _ _ _ _ _ _ _ _ _ hm14V (by rw [hsj1]; exact hn) (by rw [hsj1]; exact hk0) (by rw [hsj1]; exact hq0) rfl hv'
  have hco : ∀ N, FlockComplete.coinOf (Masked.FinRep.respOf (finCoins x ops) N) =
      fun i => Masked.slotAt (finCoins x ops) N i := fun N => Masked.FinRep.coinOf_respOf (finCoins x ops) N
  rw [hsj1, hmsV, hco, hslot] at hP
  have hQ := Masked.FinRep.repFwC_open0 (coin := fun k i => Masked.slotAt (finCoins x ops) k i) (n := refK x)
    (resp := Masked.FinRep.respOf (finCoins x ops)) (fun k => Masked.FinRep.coinOf_respOf _ k)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ hm14F (by rw [hsc2]; exact hn) (by rw [hsc2]; exact hk0)
    (by rw [hsc2]; exact hL2) (by rw [hsc2]; exact fun i hi => (hlv i hi).q_ne_zero) rfl (l0_go_of_fst hrun)
  rw [hpz, ← hst2, hsc2, hco, hslot] at hQ
  -- the views' commitment
  have hq' : FlockComplete.quietS.R
      ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
        FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
          (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
          (finSrc vs x w ν t ops i))
      (commitsView vs x w ν t ops schs) :=
    _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
      FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
  have hv0 := hcms #[]
  rw [hq' #[]] at hv0
  obtain ⟨hcsz, hall'⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok (Prod.mk.inj hv0).1
  have hk2' : k / 2 < cms.size := by rw [hcsz]; exact hjT
  have hc := hall' (k / 2) hk2'
  rw [← getElem!_pos cms (k / 2) hk2', hsj1] at hc
  obtain ⟨hcr, hcs⟩ := FlockComplete.commit0Cut_rows hc
  -- the loop's table's commitment
  have hk2 : k = 2 * (⟨k / 2, hj⟩ : Fin (Masked.jOf x)).val + (⟨k % 2, hr⟩ : Fin 2).val := by
    show k = 2 * (k / 2) + k % 2
    omega
  have hreal := hRep.1
  rw [hk2] at hreal
  obtain ⟨l0, hin0, hcm0, hlv0, htp0, hrows0⟩ := Level0Src.realOf_rows0 hS ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hreal
  have hgk : (Masked.linkIn x w t ⟨k / 2, hj⟩).g = (FlockComplete.tablesP (FlockDeploy.tabsOf (Masked.stsOf x))
      (FlockComplete.ZkTables.traces (Masked.stsOf x)
        (FlockComplete.ZkTables.maskWOf x w (Masked.UE.tapeZk t).mask)))[k / 2]! := rfl
  have key : ∀ p ∈ Masked.q0At (finCoins x ops) ⟨k / 2, hj⟩ ⟨k % 2, hr⟩,
      cms[k / 2]!.rows[p]! = tb.cm0.toC.rows[p]! ∧ cms[k / 2]!.salts[p]! = tb.cm0.toC.salts[p]! := by
    intro p hp
    obtain ⟨hpd, a, ha⟩ := Masked.FinRep.q0At_opened (finCoins x ops) ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hp
    have hpd' : p < 2 ^ ((Masked.zkTab x ⟨k / 2, hj⟩).sch.levels.getD 0 default).d := hpd
    obtain ⟨hrow, hsalt⟩ := Masked.FinRep.srcV_open0 x (finCoins x ops) w t hj a
    rw [ha] at hrow hsalt
    obtain ⟨hf1, hf2⟩ := finSrc_l0 vs x w ν t ops hj
    refine ⟨?_, ?_⟩
    · rw [hcr, FlockSoundness.Refine.getElem!_range_map _ hpd', hf1, hrow, hcm0]
      show _ = l0.rows[p]!
      rw [hrows0, hlv0, htp0, _root_.Flock.Firewall.Link.rows_eq, hin0, hgk, FlockComplete.tablesP,
        FlockSoundness.Refine.getElem!_range_map _ hjT]
      all_goals rfl
    · rw [hcs, FlockSoundness.Refine.getElem!_range_map _ hpd', hf2, hsalt, hcm0]
      show _ = ((Array.range (2 ^ l0.lv.d)).map fun q =>
        _root_.Flock.Firewall.Finish.fitBytes 192 (l0.salt q))[p]!
      rw [hlv0, FlockSoundness.Refine.getElem!_range_map _ hpd]
      have hsl : l0.salt p = (Masked.linkIn x w t ⟨k / 2, hj⟩).salt p :=
        congrFun (congrArg _root_.Flock.Firewall.Link.In.salt hin0) p
      rw [hsl]
      all_goals rfl
  rw [hP, hQ]
  unfold FlockComplete.openP
  refine ⟨Array.map_congr_left fun p hp => (key p hp).1, ?_⟩
  dsimp only
  split_ifs
  · rfl
  · exact Array.map_congr_left fun p hp => (key p hp).2

end Loop

end Claims.ZKServer
