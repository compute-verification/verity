module

public import Proofs.Flock.SessionLoop.FinInnerRec
public import Proofs.Flock.SessionLoop.FinRepTau
public import Proofs.Flock.Claims.ZK.FinishTrunc

/-! The finish's live inner proof at the session's cut, and its `τ` commitment. -/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Flock.ZkServer.Finish

open Flock With Transcript

section

variable {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M]

def repConsC (io : FlockComplete.Asks M) (st : Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : FlockComplete.Commit) (extras : Array Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) : TMW M (Proof × Zk.Level0Msgs × Array (String × Zk.Lin)) := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let mk : ℕ → F128 := fun i => vals[i]! + pads.h[i]!
  let (zp, zc) ← FlockComplete.zerocheckCut io st.m st.digest (FlockComplete.flat cm0.cap) ic.root lay mk
  let (lp, lc) ← FlockComplete.lincheckCut io st.fold st.m lay mk zc
  let claims := abcClaims st.m st.kLog zc lc ++ extras
  let (ss, T, sws) ← FlockComplete.ringSwitchCut io (fun k => switchVec g claims[k]!) claims
  let (lg, zl, rec) ← FlockComplete.ligeritoPZk io (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 sch T cm0 g
    (FlockComplete.basisVals sws (st.m - 7)) lay pads.h rep tPad extra lanePad salt (fun _ _ => 0)
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    M (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : M (Array (String × Zk.Lin)))
  return (p, zl, rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++
    lgc)

theorem repCons_eqC (io : Asks M) (st : Setup) (hm : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk)
    (ic : InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    repCons io (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt =
      repConsC io.toC st sch tPad rep dom g cm0.toC extras pads.toC ic.toC vals extra lanePad salt := by
  unfold repCons repPiop repOpening repConsC
  simp only [hm, zerocheckFw_eq, lincheckFw_eq, ringSwitchP_eq, ligeritoPZk_fw, basisVals_eq, PadsZk.toC_h,
    InnerCommit.toC_root, Commit.toC_cap, flat_eq, bind_assoc]
  all_goals refine bind_congr fun ⟨zp, zc⟩ => ?_
  all_goals try simp only [bind_assoc]
  all_goals refine bind_congr fun ⟨lp, lc⟩ => ?_
  all_goals try simp only [pure_bind, bind_assoc]

end

section

variable (coins : Array (Array F128)) (base N : ℕ) (coin : ℕ → ℕ → F128) (hn : base + coins.size ≤ N)
  (hc : ∀ k < coins.size, ∀ i < (coins[k]!).size, coin (base + k) i = (coins[k]!)[i]!)

include hn hc

theorem repConsC_br (st : Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128)
    (cm0 : FlockComplete.Commit) (extras : Array Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (hm : st.m - 13 ≤ 65536) (hx : extras.size + 2 < 2 ^ 64)
    (hN : sch.n + 1 ≤ 65536) (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536) :
    ((bridge base).st Transcript).R
      (repConsC (asksOf coins).toC st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt)
      (repConsC (FlockComplete.asksCut coin N) st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt) := by
  unfold repConsC
  refine With.Sim.bind _ (zerocheckCut_br coins base N coin hn hc _ _ _ _ _ _ hm) fun ⟨zp, zc⟩ => ?_
  refine With.Sim.bind _ (lincheckCut_br coins base N coin hn hc _ _ _ _ _) fun ⟨lp, lc⟩ => ?_
  refine With.Sim.bind _ (ringSwitchCut_br coins base N coin hn hc _ _ ?_) fun ⟨ss, T, sws⟩ => ?_
  · rw [Array.size_append, show (abcClaims st.m st.kLog zc lc).size = 2 from rfl, Nat.add_comm]
    exact hx
  refine With.Sim.bind _ (ligeritoPZk_br coins base N coin hn hc _ sch _ _ _ _ _ _ _ _ _ _ _ _ hN hq)
    fun ⟨lg, zl, rec⟩ => ?_
  refine With.Sim.bind _ (With.Sim.get _) fun tr' => ?_
  refine bridgeT_liftV_bind _ fun ⟨rp, t''⟩ _ => ?_
  refine bridgeT_liftV_bind _ fun lgc _ => ?_
  exact With.Sim.pure _ _

theorem repCons_cut (st : Setup) (hms : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (hm : st.m - 13 ≤ 65536)
    (hx : extras.size + 2 < 2 ^ 64) (hN : sch.n + 1 ≤ 65536) (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536)
    (tr : Transcript) {r : Proof × Zk.Level0Msgs × Array (String × Zk.Lin)} {t : Transcript} {k' : ℕ}
    (h : ((repCons (asksOf coins) (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((r, t), k'))
    (l : Array Claims.Msg) (hl : l.size = base) :
    (((repConsC (FlockComplete.asksCut coin N) st sch tPad rep dom g cm0.toC extras pads.toC ic.toC vals extra lanePad
        salt).run tr).go l).1 = .ok (r, t) ∧
      (((repConsC (FlockComplete.asksCut coin N) st sch tPad rep dom g cm0.toC extras pads.toC ic.toC vals extra
        lanePad salt).run tr).go l).2.size = base + k' := by
  rw [repCons_eqC (asksOf coins) st hms sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt] at h
  exact repConsC_br coins base N coin hn hc st sch tPad rep dom g cm0.toC extras pads.toC ic.toC vals extra lanePad salt
    hm hx hN hq tr 0 l (by rw [hl]; rfl) (r, t) k' h

end

end Flock.ZkServer.Finish

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 TMW Transcript Round liftTM)
open _root_.Flock.ZkServer.Finish (Asks)

namespace Loop

theorem asksPad_round_size (C : Array (Array F128)) (c : ByteArray) {k : ℕ} {coin : ℕ → F128} {k' : ℕ}
    (h : ((asksPad C).round c).run k = .ok (coin, k')) {cs : Array F128} (hk : C[k]? = some cs) :
    cs.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat := by
  simp only [asksPad, StateT.run_bind, StateT.run_get, pure_bind] at h
  rw [hk] at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hne
    simpa using hne

theorem asksPad_relOf (C : Array (Array F128)) :
    _root_.Flock.ZkServer.Finish.AsksRel (padSim C.size) (asksPad C).toC
      (_root_.Flock.ZkServer.Finish.asksOf C).toC where
  link _ _ _ _ _ h := by cases h
  round c k a k' h := by
    change ((asksPad C).round c).run k = _ at h
    obtain ⟨ha, hk'⟩ := asksPad_round C c h
    refine ⟨by omega, fun hle => ?_⟩
    have hkC : k < C.size := by omega
    have hk : C[k]? = some C[k] := getElem?_pos C k hkC
    have hsz := asksPad_round_size C c h hk
    have hn' : (C[k].size != (_root_.Flock.u64OfLE c (c.size - 8)).toNat) = false := by simp [hsz]
    have hpc : padCoin C k = fun i => C[k][i]! := by
      unfold padCoin
      rw [hk]
    show ((_root_.Flock.ZkServer.Finish.asksOf C).round c).run k = _
    rw [ha, hpc, hk']
    simp only [_root_.Flock.ZkServer.Finish.asksOf, StateT.run_bind, StateT.run_get, pure_bind, hk, hn',
      Bool.false_eq_true, ↓reduceIte, StateT.run_set, StateT.run_pure]
    all_goals rfl

theorem repCons_ofPad (C : Array (Array F128)) (st : _root_.Flock.Setup)
    (hm : st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (sch : _root_.Flock.Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (g : Array F128) (cm0 : _) (extras : Array _root_.Flock.Claim) (pads : _) (ic : _)
    (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (tr : _root_.Flock.Transcript)
    {r : _} {tr' : _root_.Flock.Transcript} {k' : ℕ}
    (h : ((_root_.Flock.ZkServer.Finish.repCons (asksPad C)
      (fun lv g s => pure (_root_.Flock.ZkServer.Finish.commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((r, tr'), k'))
    (hk : k' ≤ C.size) :
    ((_root_.Flock.ZkServer.Finish.repCons (_root_.Flock.ZkServer.Finish.asksOf C)
      (fun lv g s => pure (_root_.Flock.ZkServer.Finish.commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((r, tr'), k') :=
  ((_root_.Flock.ZkServer.Finish.repCons_rel (padSim C.size) (asksPad_relOf C) st hm sch tPad rep dom g cm0 extras
    pads ic vals extra lanePad salt tr 0 (r, tr') k' h).2 hk)

theorem liftTM_k {α : Type} {x : _root_.Flock.Transcript.TM α} {t : Flock.Transcript} {k : ℕ} {a : α} {t' : Flock.Transcript}
    {k' : ℕ} (h : ((liftTM x : TMW (StateT ℕ V) α).run t).run k = .ok ((a, t'), k')) : k' = k := by
  change (monadLift (x t) : StateT ℕ V _).run k = _ at h
  rcases e : x t with _ | ⟨a', t''⟩
  · rw [e] at h; cases h
  · rw [e] at h
    cases h
    rfl

theorem squeezeP_pad (C : Array (Array F128)) (kind : UInt8) {n : ℕ} (hn : n ≠ 0) (hn64 : n < 2 ^ 64)
    {t : Flock.Transcript} {k : ℕ} {v : Array F128} {t' : Flock.Transcript} {k' : ℕ}
    (h : ((_root_.Flock.ZkServer.Finish.squeezeP (asksPad C) kind n).run t).run k = .ok ((v, t'), k')) :
    v = (Array.range n).map (padCoin C k) ∧ k' = k + 1 ∧ ∀ cs, C[k]? = some cs → cs.size = n := by
  change ((_root_.Flock.ZkServer.Finish.squeezeP (asksPad C) kind n t).run k) = _ at h
  unfold _root_.Flock.ZkServer.Finish.squeezeP at h
  dsimp only at h
  rw [if_neg hn] at h
  obtain ⟨coin, k₁, h1, h2⟩ := _root_.Flock.ZkServer.Finish.sv_bind_ok h
  obtain ⟨rfl, rfl⟩ := asksPad_round C _ h1
  cases h2
  refine ⟨rfl, rfl, fun cs hcs => ?_⟩
  rw [asksPad_round_size C _ h1 hcs]
  exact Claims.ZKServer.Count.framed_count t kind hn64

theorem sampleP_pad (C : Array (Array F128)) {t : Flock.Transcript} {k : ℕ} {a : F128} {t' : Flock.Transcript} {k' : ℕ}
    (h : ((_root_.Flock.ZkServer.Finish.sampleP (asksPad C)).run t).run k = .ok ((a, t'), k')) :
    a = padCoin C k 0 ∧ k' = k + 1 ∧ ∀ cs, C[k]? = some cs → cs.size = 1 := by
  unfold _root_.Flock.ZkServer.Finish.sampleP at h
  obtain ⟨c, t₁, k₁, h1, h2⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  obtain ⟨rfl, rfl, hs⟩ := squeezeP_pad C 1 (by decide) (by norm_num) h1
  cases h2
  exact ⟨by simp [Array.getD_eq_getD_getElem?], rfl, hs⟩

theorem tauCommitmentWith_sv (hh : ByteArray → ByteArray) (tau : F128) (salt : ByteArray) :
    _root_.Flock.Zk.tauCommitmentWith (_root_.Flock.Hashes.ofSha512 hh : _root_.Flock.Hashes (StateT ℕ V)) tau salt =
      pure (FlockComplete.Replay.InZ.tcOf hh tau salt) := rfl

theorem tauLift_ok {hh : ByteArray → ByteArray} {tau : F128} {salt : ByteArray} {t : Flock.Transcript} {k : ℕ}
    {tc : ByteArray} {t' : Flock.Transcript} {k' : ℕ}
    (h : ((StateT.lift (_root_.Flock.Zk.tauCommitmentWith
      (_root_.Flock.Hashes.ofSha512 hh : _root_.Flock.Hashes (StateT ℕ V)) tau salt) :
      TMW (StateT ℕ V) ByteArray).run t).run k = .ok ((tc, t'), k')) :
    tc = FlockComplete.Replay.InZ.tcOf hh tau salt ∧ t' = t ∧ k' = k := by
  have e : ((StateT.lift (_root_.Flock.Zk.tauCommitmentWith
      (_root_.Flock.Hashes.ofSha512 hh : _root_.Flock.Hashes (StateT ℕ V)) tau salt) :
      TMW (StateT ℕ V) ByteArray).run t).run k = .ok ((FlockComplete.Replay.InZ.tcOf hh tau salt, t), k) := by
    rw [tauCommitmentWith_sv]
    rfl
  have e' := h.symm.trans e
  simp only [Except.ok.injEq, Prod.mk.injEq] at e'
  obtain ⟨⟨h1, h2⟩, h3⟩ := e'
  exact ⟨h1, h2, h3⟩

theorem innerP_pad_tau (C : Array (Array F128)) (lay : _root_.Flock.Zk.PadLayout)
    (p : _root_.Flock.ZkServer.Finish.PadsZk) (ic : _root_.Flock.ZkServer.Finish.InnerCommit) (finalC : F128)
    (cons : Array (String × _root_.Flock.Zk.Lin)) (hcs : cons.size + 3 < 2 ^ 64) {t : Flock.Transcript} {k : ℕ}
    {ip : _root_.Flock.Zk.InnerProof} {t' : Flock.Transcript} {k' : ℕ}
    (h : ((_root_.Flock.ZkServer.Finish.innerP (asksPad C) (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) lay
      p ic finalC cons).run t).run k = .ok ((ip, t'), k')) :
    (∀ cs, C[k]? = some cs → cs.size = 1) ∧ (∀ cs, C[k + 1]? = some cs → cs.size = cons.size + 3) ∧
      ip.tauCommit = FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash
        (FlockComplete.Replay.InZ.tau lay p.toC cons (FlockComplete.SimInner.cutCoins (padCoin C) k)) p.tauSalt := by
  have hsz : ∀ e r s, (cons ++ _root_.Flock.Zk.triple lay e r s).size = cons.size + 3 := fun _ _ _ => by
    simp [_root_.Flock.Zk.triple]
  have key : ∀ {c : ByteArray} {x : TMW (StateT ℕ V) _root_.Flock.Zk.InnerProof} {tr : Flock.Transcript} {k₀ : ℕ}
      {a : _root_.Flock.Zk.InnerProof} {tr' : Flock.Transcript} {k₀' : ℕ},
      (x.run tr).run k₀ = .ok ((a, tr'), k₀') → Rets (fun ip : _root_.Flock.Zk.InnerProof => ip.tauCommit = c) x →
        a.tauCommit = c :=
    fun e hx => hx _ _ _ _ _ e
  unfold _root_.Flock.ZkServer.Finish.innerP at h
  try dsimp only at h
  obtain ⟨_, t₁, k₁, h1, h⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  rw [liftTM_k h1] at h
  obtain ⟨_, t₂, k₂, h2, h⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  rw [liftTM_k h2] at h
  obtain ⟨eps, t₃, k₃, h3, h⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  obtain ⟨he, hk3, hs1⟩ := sampleP_pad C h3
  subst he hk3
  try dsimp only at h
  obtain ⟨_, t₄, k₄, h4, h⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  rw [liftTM_k h4] at h
  obtain ⟨gammas, t₅, k₅, h5, h⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  unfold _root_.Flock.ZkServer.Finish.sampleVecP at h5
  obtain ⟨hg, hk5, hs2⟩ := squeezeP_pad C 2 (by rw [hsz]; omega) (by rw [hsz]; exact hcs) h5
  subst hg hk5
  try dsimp only at h
  obtain ⟨tc, t₆, k₆, h6, h⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  obtain ⟨htc, -, -⟩ := tauLift_ok h6
  refine ⟨hs1, fun cs hc => (hs2 cs hc).trans (hsz _ _ _), ?_⟩
  rw [key (c := tc) h (by
    repeat (first | refine rets_bind fun _ => ?_ | exact rets_pure rfl | exact rets_throw _ | split | dsimp only)),
    htc]
  rfl

theorem tau_congr (lay : _root_.Flock.Zk.PadLayout) (p : FlockComplete.PadsZk)
    (cons : Array (String × _root_.Flock.Zk.Lin)) {cs cs' : FlockComplete.Replay.InZ.Challenges}
    (he : cs.ce 0 = cs'.ce 0) (hg : ∀ i < cons.size + 3, cs.cg i = cs'.cg i) :
    FlockComplete.Replay.InZ.tau lay p cons cs = FlockComplete.Replay.InZ.tau lay p cons cs' := by
  have hall : FlockComplete.Replay.InZ.all lay p cons cs = FlockComplete.Replay.InZ.all lay p cons cs' := by
    unfold FlockComplete.Replay.InZ.all FlockComplete.Replay.InZ.rho
    rw [he]
  have hsz : (FlockComplete.Replay.InZ.all lay p cons cs').size = cons.size + 3 := by
    simp [FlockComplete.Replay.InZ.all, _root_.Flock.Zk.triple]
  unfold FlockComplete.Replay.InZ.tau FlockComplete.Replay.InZ.lin
  rw [hall]
  have hgs : (Array.range (FlockComplete.Replay.InZ.all lay p cons cs').size).map cs.cg =
      (Array.range (FlockComplete.Replay.InZ.all lay p cons cs').size).map cs'.cg :=
    Array.map_congr_left fun i hi => hg i (by rw [Array.mem_range, hsz] at hi; exact hi)
  rw [hgs]

theorem extract_get! {C : Array (Array F128)} {N k : ℕ} (hk : k < N) (hkC : k < C.size) :
    (C.extract 0 N)[k]! = C[k]! := by
  rw [getElem!_pos (C.extract 0 N) k (by simp only [Array.size_extract]; omega), getElem!_pos C k hkC]
  simp

end Loop

end Claims.ZKServer

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf gOf)

namespace Masked

theorem repConsC_rows (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1)
    (xs : Array _root_.Flock.Claim) (hxs : xs.size = spec.points * (stAt x j).regions.size)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (tPad : ℕ) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit)
    (vals : Array _root_.Flock.F128) (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray)
    {t₀ : _root_.Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val))
    {res : _root_.Flock.Proof × _root_.Flock.Zk.Level0Msgs × Array (String × _root_.Flock.Zk.Lin)}
    {t : _root_.Flock.Transcript} {l' : Array Msg}
    (h : ((_root_.Flock.ZkServer.Finish.repConsC (FlockComplete.asksCut (ligCoin c) n) st (zkTab x j).sch tPad r.val
      dom g cm0 xs pads ic vals extra lanePad salt).run t₀).go l = (.ok (res, t), l'))
    (hmk : ∀ k < (Composed.layOf (zkTab x j).table (stAt x j)).zp 64,
      ((swSrcE x c w v τ j.val).rep r.val).masked k = vals[k]! + pads.h[k]!) :
    SwRows x c w v τ j r res.2.2 := by
  subst hg
  obtain ⟨h25, -, hsm, hk0, -, ⟨hn7, h2, -⟩, hlv, -, -⟩ := ZkExec.zk_schedule_seams _ _ _ _ (zkTab x j).hz
  have hk6' : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hstj : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  have hSm : st.m = (stAt x j).m := by
    subst hst
    rfl
  have hSk : st.kLog = (stAt x j).c.kLog := by
    subst hst
    rfl
  have hkL : st.fold.kLog = st.kLog := by
    subst hst
    rfl
  have hm14 : 14 ≤ st.m := by
    rw [hSm]
    omega
  have hk6 : 6 ≤ st.kLog := by
    rw [hSk]
    exact hk6'
  have hlay : Composed.layOf (zkTab x j).table (stAt x j) =
      _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 := by
    subst hst
    rfl
  have hlen : (Composed.layOf (zkTab x j).table (stAt x j)).len = (tabC PE x j).z.nE + 4 := by
    have h := innerLayE_len x j
    rw [innerLayE_layE] at h
    exact h
  have ez : ligCoin c (l.size + 2) 0 = Composed.zcF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    rw [hl]
    rfl
  have er : _root_.Flock.pinned ++ (Array.range (st.m - 13)).map (ligCoin c (l.size + 1)) =
      Composed.rF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    subst hst
    rw [rF_czExe c hs j r, hl]
    rfl
  have eρ : Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r =
      FlockComplete.Replay.pushes (fun k => ligCoin c (l.size + (3 + k)) 0) (st.m - 6) := by
    subst hst
    rw [hl]
    refine arr_eq_pushes (by unfold Composed.rhoF; exact Array.size_ofFn) fun k hk => ?_
    unfold Composed.rhoF
    rw [getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn]
    rfl
  have eα : ligCoin c (l.size + (3 + (st.m - 6))) 0 = (czExe PE x c j).rest.lcAlpha r := by
    subst hst
    rw [hl]
    rfl
  have eβ : ligCoin c (l.size + (4 + (st.m - 6))) 0 = (czExe PE x c j).rest.lcBeta r := by
    subst hst
    rw [hl]
    rfl
  have eτ : Composed.tsF (zkTab x j).table (stAt x j) (czExe PE x c j) r =
      FlockComplete.Replay.pushes (fun k => ligCoin c (l.size + (5 + (st.m - 6) + k)) 0) (st.kLog - 6) := by
    subst hst
    rw [hl]
    refine arr_eq_pushes (by unfold Composed.tsF; exact Array.size_ofFn) fun k hk => ?_
    unfold Composed.tsF
    rw [getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn]
    rfl
  have er' : ligCoin c (l.size + (5 + (st.m - 6) + (st.kLog - 6))) 0 = (czExe PE x c j).rest.rPrime r := by
    subst hst
    rw [hl]
    rfl
  have hmk' : ∀ k < (Composed.layOf (zkTab x j).table (stAt x j)).zp 64,
      (Composed.msgAt ((swViewE x c w v τ j).pre.2.1 r) k).val = vals[k]! + pads.h[k]! := fun k hk => by
    rw [swViewMasked x c w v τ j r hrows k]
    exact hmk k hk
  have hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => vals[i]! + pads.h[i]!).run t₀).go l = (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut (ligCoin c) n)
          (fun k => _root_.Flock.ZkServer.Finish.switchVec (ligTab x w τ j)
            (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!)
          (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val) ∧
        ss[1]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + 1) := by
    intro zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ h₁ h₂ h₃
    have hl0 := (FlockComplete.RankAt.zerocheckCut_round1 (ligCoin c) n hm14 _ _ _ _ _ h₁).1
    obtain ⟨hl1, hl1n, -⟩ := (FlockComplete.CutEq.zerocheckCut_eqAt (ligCoin c) n hm14 st.digest
      (FlockComplete.flat cm0.cap) ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
      (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) t₀ (l := l) rfl hl0.le
      (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
    obtain ⟨hl2, hl2n, -⟩ := (FlockComplete.CutEq.lincheckCut_eqAt (ligCoin c) n st.fold st.m
      (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
      (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) zc t₁ (l := l₁) rfl
      (by omega) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
    have hrank : rankSlot x j.val r.val ≤ n := by
      unfold rankSlot
      rw [hstj]
      omega
    have hsv : ∀ k < 2,
        ((swSrcE x c w v τ j.val).rep r.val).sv k =
          _root_.Flock.ZkServer.Finish.switchVec (ligTab x w τ j)
            (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]! := by
      intro k hk2
      obtain ⟨hzc, hl₁⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
      obtain ⟨hlc, -⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
      subst hst
      set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
      have hzE : FlockComplete.zcCoins (ligCoin c) l.size = zcE c j r := by
        rw [hl]
      have hlE : FlockComplete.lcCoins (ligCoin c) l₁.size
          ((_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog - 6) = lcE c j r := by
        rw [hl₁, hl]
        rfl
      rw [hzE] at hzc
      rw [hlE] at hlc
      have hab' := FlockComplete.Replay.abc_x (m := (_root_.Flock.Setup.ofCircuit (stAt x j)).m)
        (kLog := (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog) (z₁ := zc) (l₁ := lc)
        (z₂ := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
        (l₂ := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) hzc.2.1 hzc.2.2 hlc.2
      have hxk := FlockComplete.Replay.claims_x (e := xs) (FlockComplete.abcClaims_size _ _ _ _)
        (FlockComplete.abcClaims_size _ _ _ _) hab'.1 hab'.2 k
      have hA2 : (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
          (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r))).size = 2 :=
        FlockComplete.abcClaims_size _ _ _ _
      have hx6E : ((_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
          (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) ++ xs)[k]!).x.size =
          (stAt x j).m - 6 := by
        rw [FlockComplete.Replay.get_left (lt_of_lt_of_eq hk2 hA2.symm)]
        interval_cases k
        · show ((FlockComplete.Replay.pushes (lcE c j r).τ ((stAt x j).circuitFold.kLog - 6)).reverse ++
            (FlockComplete.Replay.pushes (zcE c j r).ρ ((stAt x j).m - 6)).extract ((stAt x j).c.kLog - 6)
              ((stAt x j).m - 6)).size = (stAt x j).m - 6
          rw [Array.size_append, Array.size_reverse, FlockComplete.Replay.pushes_size, Array.size_extract,
            FlockComplete.Replay.pushes_size, Nat.min_self]
          omega
        · exact FlockComplete.Replay.Zc.r_size (stAt x j).m (by omega) (zcE c j r)
      have hx6 : ((_root_.Flock.abcClaims (_root_.Flock.Setup.ofCircuit (stAt x j)).m
          (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog zc lc ++ xs)[k]!).x.size = (stAt x j).m - 6 := by
        rw [hxk]
        exact hx6E
      have hpk : ligTab x w τ j = FlockComplete.packed tr (stAt x j).m := by
        show FlockComplete.packed tr (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = _
        rw [tabsOf_stAt]
        rfl
      rw [hpk, FlockComplete.switchVec_packed (by omega) tr _ hx6, FlockComplete.Replay.switchVec_x hxk, swSrcE_sv]
      have hl₃ := (FlockComplete.ringSwitchCut_sws (ligCoin c) n _ _
        (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) h₃).2
      rw [Array.size_append, FlockComplete.abcClaims_size] at hl₃
      exact src_sv_abc c hw τ j r hx hnr hs xs k hk2 (by rw [hfk]; omega)
    have hss := FlockComplete.ringSwitchCut_ss (ligCoin c) n _ _ h₃
    have hsz : 2 ≤ (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size := by
      rw [Array.size_append, FlockComplete.abcClaims_size]
      omega
    have e : ∀ k < 2, ss[k]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + k) := by
      intro k hk
      rw [hss]
      refine (FlockComplete.Replay.pushes_get _ _ k (by omega)).trans ?_
      refine (hsv k (by omega)).symm.trans ?_
      exact swSrcE_sv_rank x hx c w v τ hnr hs j r hrank k hk
    have e0 := e 0 (by norm_num)
    rw [Nat.add_zero] at e0
    exact ⟨e0, e 1 (by norm_num)⟩
  dsimp only [_root_.Flock.ZkServer.Finish.repConsC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := _root_.Flock.ZkServer.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := _root_.Flock.ZkServer.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := _root_.Flock.ZkServer.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := _root_.Flock.ZkServer.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := _root_.Flock.ZkServer.Finish.run_bind_ok h
  rw [_root_.Flock.ZkServer.Finish.get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hr₆⟩ := _root_.Flock.ZkServer.Finish.run_bind_ok hg
  obtain ⟨hrp, ht₆, hl₆⟩ := _root_.Flock.ZkServer.Finish.liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hr₆
  obtain ⟨lgc, t₇, l₇, h₇, hc⟩ := _root_.Flock.ZkServer.Finish.run_bind_ok hr₆
  obtain ⟨hlgc, ht₇, hl₇⟩ := _root_.Flock.ZkServer.Finish.liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hc
  simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hc
  obtain ⟨⟨rfl, -⟩, -⟩ := hc
  have hl0 := (FlockComplete.RankAt.zerocheckCut_round1 (ligCoin c) n hm14 _ _ _ _ _ h₁).1
  obtain ⟨hl1, hl1n, -⟩ := (FlockComplete.CutEq.zerocheckCut_eqAt (ligCoin c) n hm14 st.digest
    (FlockComplete.flat cm0.cap) ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) t₀ (l := l) rfl hl0.le
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
  obtain ⟨hl2, hl2n, -⟩ := (FlockComplete.CutEq.lincheckCut_eqAt (ligCoin c) n st.fold st.m
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) zc t₁ (l := l₁) rfl
    (by omega) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
  obtain ⟨hl3, -, -⟩ := (FlockComplete.CutEq.ringSwitchCut_eqAt (ligCoin c) n
    (fun k => _root_.Flock.ZkServer.Finish.switchVec (ligTab x w τ j)
      (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!)
    (fun k => _root_.Flock.ZkServer.Finish.switchVec (ligTab x w τ j)
      (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!)
    (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)
    (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) t₂ (l := l₂) rfl (by omega)
    (fun _ _ _ => rfl)).2 _ _ h₃
  rw [Array.size_append, FlockComplete.abcClaims_size] at hl3
  have hS4 := FlockComplete.SimInner.ligeritoPZk_ok (ligCoin c) n (respE c) (coinOf_respE c) _ _ _ _ _ _ _ _ _ _ _ _ _
    _ _ t₃ l₃ _ _ h₄
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.wideLeaf (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hslot : ∀ f, repSlot x (2 * j.val + r.val) f = l₃.size + f (zkTab x j).sch := by
    intro f
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val), ← hl]
    unfold FlockComplete.piopRoundsZk
    omega
  have hrec := (FlockComplete.L0.ligeritoPZk_lanes (respE c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (by omega) hk0 hS4).record
  have hLC : LaneCoins (zkTab x j).table (stAt x j) (czExe PE x c j) r rec := by
    refine ⟨fun i => ?_, fun i hi => ?_, ?_⟩
    · rw [hrec]
      show fOf (FlockComplete.coinOf (respE c (l₃.size + 1)) 0) =
        fOf (slotAt c (repSlot x (2 * j.val + r.val) (fun _ => 1)) 0)
      rw [coinOf_respE, hslot]
    · rw [hrec]
      show (FlockComplete.ZkTables.Level0.laneCh (respE c) (zkTab x j).sch.k0 l₃.size).toArray[i]! =
        kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + i)
      unfold FlockComplete.ZkTables.Level0.laneCh
      rw [getElem!_pos _ i (by simp only [List.size_toArray, List.length_map, List.length_range]; exact hi)]
      simp only [List.getElem_toArray, List.getElem_map, List.getElem_range, FlockComplete.coinKOf, coinOf_respE, kAt,
        hslot, Nat.add_assoc]
    · rw [hrec]
      show FlockComplete.ZkTables.Level0.laneRho (respE c) (zkTab x j).sch.k0 l₃.size = rhoAt c j.val r.val
      unfold FlockComplete.ZkTables.Level0.laneRho rhoAt rhoSlot
      simp only [FlockComplete.coinKOf, coinOf_respE, hslot]
      rw [show l₃.size + 2 + (zkTab x j).sch.k0 = l₃.size + ((zkTab x j).sch.k0 + 2) by omega,
        show l₃.size + 3 + (zkTab x j).sch.k0 = l₃.size + ((zkTab x j).sch.k0 + 2) + 1 by omega]
  obtain ⟨hlsz, hlasz, hrow⟩ := Composed.tauFw_at_coef (zkTab x j).table (stAt x j) (czExe PE x c j) r
    ((swViewE x c w v τ j).pre.2.1 r) (swViewE x c w v τ j).pre.1
    ((swViewE x c w v τ j).pre.2.2.2.2.2 r).1 ((swViewE x c w v τ j).pre.2.2.2.2.1 r)
    (ligCoin c) n st (zkTab x j).sch tPad r.val dom (ligTab x w τ j) cm0 xs pads ic
    vals extra lanePad salt ht₀ hm14 hk6 hSm hSk hkL hlay hmk' ez er eρ eα eβ eτ er'
    (fun combv hcomb => by
      subst hst
      have hkm := (zkTab x j).table.S.hk
      have hS : (zkTab x j).table.S = FlockSoundness.Refine.stmtOf (stAt x j) (zkTab x j).wf.1
          (FlockSoundness.Refine.regionsOf (stAt x j)) := rfl
      refine folded_unique (zkTab x j).table.S _ (Composed.modelFold (zkTab x j).table.S)
        (hS ▸ FlockSoundness.Refine.stmtOf_fold (stAt x j) (zkTab x j).wf.1
          (FlockSoundness.Refine.regionsOf (stAt x j)))
        (Composed.modelFold_realizes _) _ _ _ _ _ _ _ (FlockSoundness.Refine.skipWeights_size _) ?_ hcomb rfl
      simp only [Composed.rhoF, Array.size_extract, Array.size_ofFn]
      show min ((zkTab x j).table.S.kLog - 6) ((zkTab x j).table.S.m - 6) - 0 = (zkTab x j).table.S.kLog - 6
      omega)
    hsw hlen h₁ h₂ h₃ h₄ hrp hlgc hLC
  have hLy : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = innerLayE x j := by
    subst hst
    exact (innerLayE_eq x j).symm
  have hrp' := hrp
  rw [hLy] at hrp'
  exact swRows_coef x c w v τ j r hm14 hrp' ss ⟨hlsz, hlasz⟩ hrow

end Masked
end Claims.ZKMal
