module

public import Proofs.Flock.SessionLoop.FinInner

/-! The finish's transcript at a rep answered by the request's coins records each round it takes. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 TMW Transcript Round liftTM)
open _root_.Flock.ZkServer.Finish (Asks)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

def padCoin (C : Array (Array F128)) (k : ℕ) : ℕ → F128 :=
  match C[k]? with
  | some cs => fun i => cs[i]!
  | none => fun _ => 1

theorem padCoin_extract {C : Array (Array F128)} {N k : ℕ} (hk : k < N) (hkC : k < C.size) :
    padCoin (C.extract 0 N) k = fun i => (C[k]!)[i]! := by
  have h : (C.extract 0 N)[k]? = some (C[k]'hkC) := by
    rw [Array.getElem?_extract_of_lt (by omega)]
    simp
  unfold padCoin
  rw [h]
  funext i
  simp only [getElem!_pos C k hkC]

theorem getElem!_map_range {f : ℕ → F128} {n i : ℕ} (hi : i < n) : ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using hi)]
  simp

theorem getD_map_range {f : ℕ → F128} {n : ℕ} (hn : 0 < n) : ((Array.range n).map f).getD 0 0 = f 0 := by
  simp [Array.getD_eq_getD_getElem?, hn]

def RecStep (C : Array (Array F128)) (t : Flock.Transcript) (k : ℕ) (t' : Flock.Transcript) (k' : ℕ) : Prop :=
  ∃ rs : List Round, t'.rounds.toList = t.rounds.toList ++ rs ∧ k' = k + rs.length ∧
    ∀ i (hi : i < rs.length), 0 < rs[i].n ∧ rs[i].coins = (Array.range rs[i].n).map (padCoin C (k + i))

theorem RecStep.same {C : Array (Array F128)} {t t' : Flock.Transcript} {k k' : ℕ} (ht : t'.rounds = t.rounds)
    (hk : k' = k) : RecStep C t k t' k' :=
  ⟨[], by simp [ht], by simp [hk], fun i hi => absurd hi (Nat.not_lt_zero i)⟩

theorem RecStep.trans {C : Array (Array F128)} {t₀ t₁ t₂ : Flock.Transcript} {k₀ k₁ k₂ : ℕ}
    (h1 : RecStep C t₀ k₀ t₁ k₁) (h2 : RecStep C t₁ k₁ t₂ k₂) : RecStep C t₀ k₀ t₂ k₂ := by
  obtain ⟨rs, e1, ek1, hr1⟩ := h1
  obtain ⟨rs', e1', ek1', hr1'⟩ := h2
  refine ⟨rs ++ rs', by rw [e1', e1, List.append_assoc], by rw [ek1', ek1, List.length_append, Nat.add_assoc],
    fun i hi => ?_⟩
  by_cases hil : i < rs.length
  · rw [List.getElem_append_left hil]
    exact hr1 i hil
  · have hle : rs.length ≤ i := Nat.le_of_not_lt hil
    rw [List.getElem_append_right hle]
    have hi' : i - rs.length < rs'.length := by rw [List.length_append] at hi; omega
    rw [show k₀ + i = k₁ + (i - rs.length) by omega]
    exact hr1' _ hi'

def RecP (C : Array (Array F128)) {α : Type} (y : TMW (StateT ℕ V) α) : Prop :=
  ∀ (t : Flock.Transcript) (k : ℕ) (a : α) (t' : Flock.Transcript) (k' : ℕ), (y.run t).run k = .ok ((a, t'), k') →
    RecStep C t k t' k'

def recP (C : Array (Array F128)) : _root_.Flock.With.Sim (TMW (StateT ℕ V)) (TMW (StateT ℕ V)) where
  R _ y := RecP C y
  pure _ t k _ t' k' h := by
    cases h
    exact RecStep.same rfl rfl
  bind {α β x y f g} hxy hfg t k b t'' k'' h := by
    obtain ⟨a, t₁, k₁, h1, h2⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
    exact (hxy t k a t₁ k₁ h1).trans (hfg a t₁ k₁ b t'' k'' h2)
  throw _ _ _ _ _ _ h := by cases h
  lift v t k a t' k' h := by
    cases v with
    | error e => cases h
    | ok a0 =>
      cases h
      exact RecStep.same rfl rfl

theorem asksPad_round (C : Array (Array F128)) (c : ByteArray) {k : ℕ} {coin : ℕ → F128} {k' : ℕ}
    (h : ((asksPad C).round c).run k = .ok (coin, k')) : coin = padCoin C k ∧ k' = k + 1 := by
  simp only [asksPad, StateT.run_bind, StateT.run_get, pure_bind] at h
  unfold padCoin
  rcases hk : C[k]? with _ | cs
  · rw [hk] at h
    cases h
    exact ⟨rfl, rfl⟩
  · rw [hk] at h
    dsimp only at h
    split at h
    · cases h
    · cases h
      exact ⟨rfl, rfl⟩

theorem squeezeP_recP (C : Array (Array F128)) (kind : UInt8) (n : ℕ) :
    (recP C).R (_root_.Flock.ZkServer.Finish.squeezeP (asksPad C) kind n)
      (_root_.Flock.ZkServer.Finish.squeezeP (asksPad C) kind n) := by
  intro t k a t' k' h
  change ((_root_.Flock.ZkServer.Finish.squeezeP (asksPad C) kind n t).run k) = _ at h
  unfold _root_.Flock.ZkServer.Finish.squeezeP at h
  dsimp only at h
  split at h
  · cases h
    exact RecStep.same rfl rfl
  · rename_i hn
    obtain ⟨coin, k₁, h1, h2⟩ := _root_.Flock.ZkServer.Finish.sv_bind_ok h
    obtain ⟨rfl, rfl⟩ := asksPad_round C _ h1
    cases h2
    refine ⟨[⟨n, ByteArray.empty, (Array.range n).map (padCoin C k), some (t.framed kind n)⟩], by simp, by simp,
      fun i hi => ?_⟩
    obtain rfl : i = 0 := by simpa using hi
    exact ⟨Nat.pos_of_ne_zero hn, rfl⟩

theorem liftTM_recP (C : Array (Array F128)) {α : Type} (x : _root_.Flock.Transcript.TM α)
    (hx : ∀ t a t', x t = .ok (a, t') → t'.rounds = t.rounds) :
    (recP C).R (liftTM x : TMW (StateT ℕ V) α) (liftTM x) := by
  intro t k a t' k' h
  change (monadLift (x t) : StateT ℕ V _).run k = _ at h
  rcases e : x t with _ | ⟨a', t''⟩
  · rw [e] at h; cases h
  · rw [e] at h
    cases h
    exact RecStep.same (hx t _ _ e) rfl

theorem label_recP (C : Array (Array F128)) (s : String) :
    (recP C).R (liftTM (_root_.Flock.Transcript.label s) : TMW (StateT ℕ V) Unit)
      (liftTM (_root_.Flock.Transcript.label s)) :=
  liftTM_recP C _ fun _ _ _ h => by cases h; rfl

theorem absorbF_recP (C : Array (Array F128)) (v : F128) :
    (recP C).R (liftTM (_root_.Flock.Transcript.absorbF v) : TMW (StateT ℕ V) Unit)
      (liftTM (_root_.Flock.Transcript.absorbF v)) :=
  liftTM_recP C _ fun _ _ _ h => by cases h; rfl

theorem absorbSlice_recP (C : Array (Array F128)) (vs : Array F128) :
    (recP C).R (liftTM (_root_.Flock.Transcript.absorbSlice vs) : TMW (StateT ℕ V) Unit)
      (liftTM (_root_.Flock.Transcript.absorbSlice vs)) :=
  liftTM_recP C _ fun _ _ _ h => by cases h; rfl

theorem absorbK_recP (C : Array (Array F128)) (v : _root_.Flock.F256) :
    (recP C).R (liftTM (_root_.Flock.Transcript.absorbK v) : TMW (StateT ℕ V) Unit)
      (liftTM (_root_.Flock.Transcript.absorbK v)) :=
  liftTM_recP C _ fun _ _ _ h => by cases h; rfl

theorem absorbBytes_recP (C : Array (Array F128)) (b : ByteArray) :
    (recP C).R (liftTM (_root_.Flock.Transcript.absorbBytes b) : TMW (StateT ℕ V) Unit)
      (liftTM (_root_.Flock.Transcript.absorbBytes b)) :=
  liftTM_recP C _ fun _ _ _ h => by cases h; rfl

theorem pow_recP (C : Array (Array F128)) (v : UInt64) :
    (recP C).R (liftTM (_root_.Flock.Transcript.pow v) : TMW (StateT ℕ V) Unit)
      (liftTM (_root_.Flock.Transcript.pow v)) :=
  liftTM_recP C _ fun _ _ _ h => by cases h; rfl

theorem liftK_recP (C : Array (Array F128)) {α : Type} (x : StateT ℕ V α)
    (hx : ∀ k a k', x.run k = .ok (a, k') → k' = k) :
    (recP C).R (StateT.lift x : TMW (StateT ℕ V) α) (StateT.lift x) := by
  intro t k a t' k' h
  rw [StateT.run_lift] at h
  obtain ⟨a', k₁, h1, h2⟩ := _root_.Flock.ZkServer.Finish.sv_bind_ok h
  cases h2
  exact RecStep.same rfl (hx k _ _ h1)

theorem liftPure_recP (C : Array (Array F128)) {α : Type} (a : α) :
    (recP C).R (StateT.lift (pure a : StateT ℕ V α) : TMW (StateT ℕ V) α) (StateT.lift (pure a)) :=
  liftK_recP C _ fun _ _ _ h => by cases h; rfl

theorem get_recP (C : Array (Array F128)) :
    (recP C).R (get : TMW (StateT ℕ V) Flock.Transcript) get := by
  intro t k a t' k' h
  cases h
  exact RecStep.same rfl rfl

theorem liftT_recP (C : Array (Array F128)) {α : Type} {x y : TMW (StateT ℕ V) α} (h : (recP C).R x y) :
    ((recP C).st _root_.Flock.ZkServer.Finish.LigState).R (_root_.Flock.ZkServer.Finish.liftT x)
      (_root_.Flock.ZkServer.Finish.liftT y) :=
  _root_.Flock.With.Sim.lift_st _ h

theorem sampleP_recP (C : Array (Array F128)) :
    (recP C).R (_root_.Flock.ZkServer.Finish.sampleP (asksPad C)) (_root_.Flock.ZkServer.Finish.sampleP (asksPad C)) := by
  unfold _root_.Flock.ZkServer.Finish.sampleP
  sim_walk [squeezeP_recP C]

theorem sampleKP_recP (C : Array (Array F128)) :
    (recP C).R (_root_.Flock.ZkServer.Finish.sampleKP (asksPad C))
      (_root_.Flock.ZkServer.Finish.sampleKP (asksPad C)) := by
  unfold _root_.Flock.ZkServer.Finish.sampleKP
  sim_walk [squeezeP_recP C]

theorem sendMsg_recP (C : Array (Array F128)) (m : _root_.Flock.F256 × _root_.Flock.F256) :
    ((recP C).st _root_.Flock.ZkServer.Finish.LigState).R
      (_root_.Flock.ZkServer.Finish.sendMsg m : _root_.Flock.ZkServer.Finish.LP (StateT ℕ V) Unit)
      (_root_.Flock.ZkServer.Finish.sendMsg m) := by
  unfold _root_.Flock.ZkServer.Finish.sendMsg
  sim_walk [liftT_recP C, absorbK_recP C]

theorem powP_recP (C : Array (Array F128)) (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((recP C).st _root_.Flock.ZkServer.Finish.LigState).R
      (_root_.Flock.ZkServer.Finish.powP nonce kind : _root_.Flock.ZkServer.Finish.LP (StateT ℕ V) Unit)
      (_root_.Flock.ZkServer.Finish.powP nonce kind) := by
  unfold _root_.Flock.ZkServer.Finish.powP
  sim_walk [liftT_recP C, pow_recP C, label_recP C, absorbF_recP C, absorbSlice_recP C, absorbK_recP C,
    absorbBytes_recP C, get_recP C]

theorem challengeP_recP (C : Array (Array F128)) (split : Bool) :
    ((recP C).st _root_.Flock.ZkServer.Finish.LigState).R
      (_root_.Flock.ZkServer.Finish.challengeP (asksPad C) split)
      (_root_.Flock.ZkServer.Finish.challengeP (asksPad C) split) := by
  unfold _root_.Flock.ZkServer.Finish.challengeP
  sim_walk [liftT_recP C, sampleKP_recP C, sendMsg_recP C, label_recP C, absorbF_recP C, absorbSlice_recP C,
    absorbK_recP C, absorbBytes_recP C, pow_recP C, get_recP C]

theorem oodIntroP_recP (C : Array (Array F128)) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ((recP C).st _root_.Flock.ZkServer.Finish.LigState).R
      (_root_.Flock.ZkServer.Finish.oodIntroP (asksPad C) nonce g dim)
      (_root_.Flock.ZkServer.Finish.oodIntroP (asksPad C) nonce g dim) := by
  unfold _root_.Flock.ZkServer.Finish.oodIntroP _root_.Flock.ZkServer.Finish.sampleVecP
  sim_walk [liftT_recP C, squeezeP_recP C, sampleP_recP C, sendMsg_recP C, powP_recP C, label_recP C,
    absorbF_recP C, absorbSlice_recP C, absorbK_recP C, absorbBytes_recP C, pow_recP C, get_recP C]

theorem queryP_recP (C : Array (Array F128)) (nonce : ℕ → ℕ → UInt64) (ms : _root_.Flock.MerkleScheme)
    (lv : _root_.Flock.Level) (cm : _root_.Flock.ZkServer.Finish.Commit) :
    ((recP C).st _root_.Flock.ZkServer.Finish.LigState).R
      (_root_.Flock.ZkServer.Finish.queryP (asksPad C) nonce ms lv cm)
      (_root_.Flock.ZkServer.Finish.queryP (asksPad C) nonce ms lv cm) := by
  unfold _root_.Flock.ZkServer.Finish.queryP _root_.Flock.ZkServer.Finish.sampleVecP
  sim_walk [liftT_recP C, squeezeP_recP C, powP_recP C, label_recP C, absorbF_recP C, absorbSlice_recP C,
    absorbK_recP C, absorbBytes_recP C, pow_recP C, get_recP C]

theorem zerocheckFw_recP (C : Array (Array F128)) (m : ℕ) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) :
    (recP C).R (_root_.Flock.ZkServer.Finish.zerocheckFw (asksPad C) m digest cap root lay mk)
      (_root_.Flock.ZkServer.Finish.zerocheckFw (asksPad C) m digest cap root lay mk) := by
  unfold _root_.Flock.ZkServer.Finish.zerocheckFw _root_.Flock.ZkServer.Finish.sampleVecP
  sim_walk [squeezeP_recP C, sampleP_recP C, label_recP C, absorbF_recP C, absorbSlice_recP C, absorbK_recP C,
    absorbBytes_recP C, pow_recP C, get_recP C]

theorem lincheckFw_recP (C : Array (Array F128)) (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) (zc : _root_.Flock.ZcOut) :
    (recP C).R (_root_.Flock.ZkServer.Finish.lincheckFw (asksPad C) c lay mk zc)
      (_root_.Flock.ZkServer.Finish.lincheckFw (asksPad C) c lay mk zc) := by
  unfold _root_.Flock.ZkServer.Finish.lincheckFw
  sim_walk [sampleP_recP C, label_recP C, absorbF_recP C, absorbSlice_recP C, absorbK_recP C, absorbBytes_recP C,
    pow_recP C, get_recP C]

theorem ringSwitchP_recP (C : Array (Array F128)) (sv : ℕ → Array F128) (claims : Array _root_.Flock.Claim) :
    (recP C).R (_root_.Flock.ZkServer.Finish.ringSwitchP (asksPad C) sv claims)
      (_root_.Flock.ZkServer.Finish.ringSwitchP (asksPad C) sv claims) := by
  unfold _root_.Flock.ZkServer.Finish.ringSwitchP _root_.Flock.ZkServer.Finish.sampleVecP
  sim_walk [squeezeP_recP C, label_recP C, absorbF_recP C, absorbSlice_recP C, absorbK_recP C, absorbBytes_recP C,
    pow_recP C, get_recP C]

theorem ligeritoPZk_recP (C : Array (Array F128))
    (commit : _root_.Flock.Level → Array F128 → (ℕ → ByteArray) → StateT ℕ V _root_.Flock.ZkServer.Finish.Commit)
    (hcommit : ∀ lv g s, (recP C).R (StateT.lift (commit lv g s) : TMW (StateT ℕ V) _) (StateT.lift (commit lv g s)))
    (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : F128)
    (cm0 : _root_.Flock.ZkServer.Finish.Commit) (t b : Array F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (recP C).R
      (_root_.Flock.ZkServer.Finish.ligeritoPZk (asksPad C) commit ms sch T cm0 t b lay h rep tPad extra lanePad salt
        nonce)
      (_root_.Flock.ZkServer.Finish.ligeritoPZk (asksPad C) commit ms sch T cm0 t b lay h rep tPad extra lanePad salt
        nonce) := by
  unfold _root_.Flock.ZkServer.Finish.ligeritoPZk _root_.Flock.ZkServer.Finish.sampleVecP
  sim_walk [liftT_recP C, squeezeP_recP C, sampleP_recP C, sampleKP_recP C, sendMsg_recP C, powP_recP C,
    challengeP_recP C, oodIntroP_recP C, queryP_recP C, hcommit, label_recP C, absorbF_recP C, absorbBytes_recP C,
    absorbK_recP C, absorbSlice_recP C, pow_recP C, get_recP C]

theorem repPiop_recP (C : Array (Array F128)) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout)
    (cap root : ByteArray) (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) :
    (recP C).R (_root_.Flock.ZkServer.Finish.repPiop (asksPad C) st lay cap root extras mk)
      (_root_.Flock.ZkServer.Finish.repPiop (asksPad C) st lay cap root extras mk) := by
  unfold _root_.Flock.ZkServer.Finish.repPiop
  sim_walk [zerocheckFw_recP C, lincheckFw_recP C]

theorem repOpening_recP (C : Array (Array F128)) (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule)
    (lay : _root_.Flock.Zk.PadLayout) (tPad rep : ℕ) (g : Array F128) (cm0 : _root_.Flock.ZkServer.Finish.Commit)
    (claims : Array _root_.Flock.Claim) (h : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) :
    (recP C).R
      (_root_.Flock.ZkServer.Finish.repOpening (asksPad C)
        (fun lv g s => pure (_root_.Flock.ZkServer.Finish.commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra
        lanePad salt)
      (_root_.Flock.ZkServer.Finish.repOpening (asksPad C)
        (fun lv g s => pure (_root_.Flock.ZkServer.Finish.commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra
        lanePad salt) := by
  unfold _root_.Flock.ZkServer.Finish.repOpening
  sim_walk [ringSwitchP_recP C,
    ligeritoPZk_recP C _ (fun lv g s => liftPure_recP C (_root_.Flock.ZkServer.Finish.commitFw lv g s))]

def Rets {α : Type} (P : α → Prop) (x : TMW (StateT ℕ V) α) : Prop :=
  ∀ tr k a tr' k', (x.run tr).run k = .ok ((a, tr'), k') → P a

theorem rets_pure {α : Type} {P : α → Prop} {a : α} (h : P a) : Rets P (pure a) := by
  intro tr k b tr' k' e
  cases e
  exact h

theorem rets_throw {α : Type} {P : α → Prop} (e : String) : Rets P (throw e : TMW (StateT ℕ V) α) := by
  intro tr k b tr' k' h
  cases h

theorem rets_bind {α β : Type} {P : β → Prop} {x : TMW (StateT ℕ V) α} {f : α → TMW (StateT ℕ V) β}
    (h : ∀ a, Rets P (f a)) : Rets P (x >>= f) := by
  intro tr k b tr' k' e
  obtain ⟨a, tr₁, k₁, -, e2⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok e
  exact h a tr₁ k₁ b tr' k' e2

theorem rets_bind' {α β : Type} {P : α → Prop} {Q : β → Prop} {x : TMW (StateT ℕ V) α}
    {f : α → TMW (StateT ℕ V) β} (hx : Rets P x) (h : ∀ a, P a → Rets Q (f a)) : Rets Q (x >>= f) := by
  intro tr k b tr' k' e
  obtain ⟨a, tr₁, k₁, e1, e2⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok e
  exact h a (hx tr k a tr₁ k₁ e1) tr₁ k₁ b tr' k' e2

theorem zerocheckFw_rets (io : Asks (StateT ℕ V)) (m : ℕ) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) :
    Rets (fun r => r.1.round1c = (Array.range 64).map (fun i => mk (lay.r1c i)) ∧ r.1.finalC = mk lay.fc)
      (_root_.Flock.ZkServer.Finish.zerocheckFw io m digest cap root lay mk) := by
  unfold _root_.Flock.ZkServer.Finish.zerocheckFw
  repeat (first | refine rets_bind fun _ => ?_ | exact rets_pure ⟨rfl, rfl⟩ | split | dsimp only)

theorem repPiop_rets (io : Asks (StateT ℕ V)) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout)
    (cap root : ByteArray) (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) :
    Rets (fun r => r.1.round1c = (Array.range 64).map (fun i => mk (lay.r1c i)) ∧ r.1.finalC = mk lay.fc)
      (_root_.Flock.ZkServer.Finish.repPiop io st lay cap root extras mk) := by
  unfold _root_.Flock.ZkServer.Finish.repPiop
  refine rets_bind' (zerocheckFw_rets io st.m st.digest cap root lay mk) ?_
  rintro ⟨zp, zc⟩ hr
  refine rets_bind ?_
  rintro ⟨lp, lc⟩
  exact rets_pure hr

theorem repCons_rets (io : Asks (StateT ℕ V))
    (commit : _root_.Flock.Level → Array F128 → (ℕ → ByteArray) → StateT ℕ V _root_.Flock.ZkServer.Finish.Commit)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128)
    (cm0 : _root_.Flock.ZkServer.Finish.Commit) (extras : Array _root_.Flock.Claim)
    (pads : _root_.Flock.ZkServer.Finish.PadsZk) (ic : _root_.Flock.ZkServer.Finish.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    Rets (fun r => r.1.zerocheck.finalC = vals[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).fc]! +
        pads.h[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).fc]!)
      (_root_.Flock.ZkServer.Finish.repCons io commit st sch tPad rep dom g cm0 extras pads ic vals extra lanePad
        salt) := by
  unfold _root_.Flock.ZkServer.Finish.repCons
  refine rets_bind' (repPiop_rets io st _ _ _ extras _) ?_
  rintro ⟨zp, lp, claims⟩ hr
  refine rets_bind ?_
  rintro ⟨ss, lg, zl, rec⟩
  refine rets_bind fun _ => ?_
  refine rets_bind ?_
  rintro ⟨rp, _⟩
  refine rets_bind fun _ => ?_
  exact rets_pure hr.2

theorem liftV_ok {α : Type} {v : V α} {t : Flock.Transcript} {k : ℕ} {a : α} {t' : Flock.Transcript} {k' : ℕ}
    (h : ((StateT.lift (monadLift v : StateT ℕ V α) : TMW (StateT ℕ V) α).run t).run k = .ok ((a, t'), k')) :
    v = .ok a ∧ t' = t ∧ k' = k := by
  cases v with
  | error e => cases h
  | ok a0 =>
    cases h
    exact ⟨rfl, rfl, rfl⟩

theorem repCons_parts {C : Array (Array F128)} {st : _root_.Flock.Setup} {sch : _root_.Flock.Schedule}
    {tPad rep : ℕ} {dom : ByteArray} {g : Array F128} {cm0 : _root_.Flock.ZkServer.Finish.Commit}
    {extras : Array _root_.Flock.Claim} {pads : _root_.Flock.ZkServer.Finish.PadsZk}
    {ic : _root_.Flock.ZkServer.Finish.InnerCommit} {vals : Array F128} {extra lanePad : ℕ → ℕ → F128}
    {salt : ℕ → ℕ → ByteArray} {p : _root_.Flock.Proof} {zl : _root_.Flock.Zk.Level0Msgs}
    {cons : Array (String × _root_.Flock.Zk.Lin)} {tF : Flock.Transcript} {kF : ℕ}
    (h : ((_root_.Flock.ZkServer.Finish.repCons (asksPad C)
      (fun lv g s => pure (_root_.Flock.ZkServer.Finish.commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run (Flock.Transcript.init #[] dom)).run 0 = .ok (((p, zl, cons), tF), kF)) :
    ∃ (rs : List Round) (rp : _root_.Flock.Zk.Replayed) (tq : Flock.Transcript) (suf : Array (String × _root_.Flock.Zk.Lin)),
      tF.rounds.toList = rs ∧ kF = rs.length ∧
      (∀ i (hi : i < rs.length), 0 < rs[i].n ∧ rs[i].coins = (Array.range rs[i].n).map (padCoin C i)) ∧
      _root_.Flock.Zk.replay st (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) p ic.root
        (Flock.Transcript.init tF.rounds dom) = .ok (rp, tq) ∧
      cons = rp.cons ++ suf ∧
      p.zerocheck.round1c = (Array.range 64).map (fun i =>
        vals[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).r1c i]! +
          pads.h[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).r1c i]!) ∧
      p.zerocheck.finalC = vals[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).fc]! +
        pads.h[(_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).fc]! := by
  unfold _root_.Flock.ZkServer.Finish.repCons at h
  try dsimp only at h
  obtain ⟨⟨zp, lp, claims⟩, t₁, k₁, h1, hA⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok h
  clear h
  have hz := repPiop_rets (asksPad C) st _ _ _ extras _ _ _ _ _ _ h1
  try dsimp only at hA
  obtain ⟨⟨ss, lg, zl', rec⟩, t₂, k₂, h2, hB⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok hA
  clear hA
  have hR := (repPiop_recP C st _ _ _ extras _ _ _ _ _ _ h1).trans
    (repOpening_recP C st sch _ tPad rep g cm0 claims pads.h extra lanePad salt _ _ _ _ _ h2)
  try dsimp only at hB
  obtain ⟨tr', t₃, k₃, h3, hC⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok hB
  clear hB
  cases h3
  try dsimp only at hC
  obtain ⟨⟨rp, tq⟩, t₄, k₄, h4, hD⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok hC
  clear hC
  obtain ⟨hrp, rfl, rfl⟩ := liftV_ok h4
  try dsimp only at hD
  obtain ⟨lgc, t₅, k₅, h5, hE⟩ := _root_.Flock.ZkServer.Finish.tmw_bind_ok hD
  clear hD
  obtain ⟨-, rfl, rfl⟩ := liftV_ok h5
  cases hE
  obtain ⟨rs, e1, ek1, hr1⟩ := hR
  refine ⟨rs, rp, tq, _, e1.trans (List.nil_append rs), ek1.trans (Nat.zero_add _), fun i hi => ?_, hrp,
    Array.append_assoc, hz.1, hz.2⟩
  have e := hr1 i hi
  rwa [Nat.zero_add] at e

end Loop

end Claims.ZKServer
