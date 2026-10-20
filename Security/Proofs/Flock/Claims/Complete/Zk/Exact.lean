import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.LigeritoProverPos
import Proofs.Flock.Claims.Complete.Zk.InnerGo

/-!
# A returning masked rep's sends, and the refusals it passed

Read forward from a run that returns (`LigeritoProverPos`'s steps), each prover step of a masked rep sends a fixed
number of bodies, and each of its refusals that can fire did not:

* **Ligerito** (`ligeritoPZk_exact`): `k_0 + 7 + 11 R` sends over `R + 1` levels (`k_0 + 11` before the recursive
  levels, `11` at each but the last and `7` at the last; `oodIntroP_two`, `queryP_two`, `challengeP_one`), and level 0's
  pair `ρ₀, ρ₁`, the replies to sends `|l| + 2 + k_0` and `|l| + 3 + k_0`, are independent.
* **The inner proof** (`innerP_exact`): `4` sends, and `β`, the reply to send `|l| + 2`, is not `0`.
* **A rep** (`repPZk_exact`): the zerocheck's `3 + (m − 6)`, the lincheck's `3 + (k_log − 6)`, the ring switch's
  `claims + 1` (`repPZk_rs`'s closed forms), then Ligerito and the inner proof; its rank check passed on its
  transcript after the lincheck.
-/

namespace FlockComplete

open Flock Transcript Replay Replay.Lg

variable {α β : Type}

/-! ## Ligerito's steps -/

theorem lsilent_pure (a : α) : LSilent (Pure.pure a : LP PS α) := by
  intro st t l b st' t' l' h
  exact (lpure_go_ok h).2.symm

theorem lsilent_bind' {x : LP PS α} {f : α → LP PS β} (hx : LSilent x) (hf : ∀ a, LSilent (f a)) :
    LSilent (x >>= f) := by
  intro st t l b st' t' l' h
  obtain ⟨a, st₁, t₁, l₁, h₁, h₂⟩ := lp_bind_ok h
  rw [hf a st₁ t₁ l₁ b st' t' l' h₂, hx st t l a st₁ t₁ l₁ h₁]

theorem lsilent_set (s : LigState) : LSilent (set s : LP PS PUnit) := by
  intro st t l a st' t' l' h
  exact (ok_inj (show (Except.ok (((), s), t), l) = _ from h)).2.2.2.symm

theorem lsilent_powP (nonce : ℕ → ℕ → UInt64) (kind : ℕ) : LSilent (powP nonce kind : LP PS Unit) := by
  unfold powP
  exact lsilent_bind' lsilent_get fun _ => lsilent_bind' (lsilent_set _) fun _ => lsilent_liftTM _

theorem lsilent_forIn_list {γ σ : Type} (f : γ → σ → LP PS (ForInStep σ)) (hf : ∀ x s, LSilent (f x s)) :
    ∀ (xs : List γ) (s : σ), LSilent (forIn xs s f)
  | [], s => by rw [List.forIn_nil]; exact lsilent_pure s
  | x :: xs, s => by
    rw [List.forIn_cons]
    refine lsilent_bind' (hf x s) fun r => ?_
    cases r with
    | done s' => exact lsilent_pure s'
    | yield s' => exact lsilent_forIn_list f hf xs s'

theorem lsilent_forIn_array {γ σ : Type} (xs : Array γ) (s : σ) (f : γ → σ → LP PS (ForInStep σ))
    (hf : ∀ x s, LSilent (f x s)) : LSilent (forIn xs s f) := by
  rw [← Array.forIn_toList]
  exact lsilent_forIn_list f hf xs.toList s

/-- A quiet computation of the pure session, lifted into Ligerito's, sends nothing. -/
theorem lsilent_quiet {v : V α} {y : PS α} (hq : quietS.R v y) : LSilent (liftT (StateT.lift y) : LP PS α) := by
  intro st t l a st' t' l' e
  change (((StateT.lift y : TMW PS α) >>= fun a => Pure.pure (a, st)).run t).go l = _ at e
  rw [StateT.run_bind] at e
  obtain ⟨⟨a₁, t₁⟩, l₁, e1, e2⟩ := JsonRT.go_bind_ok e
  change (y >>= fun a => Pure.pure (a, t)).go l = _ at e1
  obtain ⟨b, l₂, e3, e4⟩ := JsonRT.go_bind_ok e1
  rw [hq l] at e3
  obtain ⟨-, rfl⟩ := Prod.mk.inj e3
  obtain ⟨-, rfl⟩ := Prod.mk.inj e4
  exact (Prod.mk.inj e2).2.symm

theorem challengeP_one (resp : ℕ → Claims.Msg) (split : Bool) {st st' : LigState} {t t' : Transcript}
    {l l' : Array Claims.Msg} {a : F256}
    (h : (((challengeP (asksS resp) split).run st).run t).go l = (.ok ((a, st'), t'), l')) :
    l'.size = l.size + 1 := by
  unfold challengeP at h
  obtain ⟨_, l₁, hl₁, h⟩ := lsampleKP_bind resp h
  obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_modify _) h
  split at h
  · obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_modify _) h
    obtain ⟨_, _, _, h⟩ := lsilent_bind lsilent_get h
    obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_sendMsg _) h
    obtain ⟨-, rfl⟩ := lpure_go_ok h
    exact hl₁
  · obtain ⟨_, _, _, h⟩ := lsilent_bind lsilent_get h
    obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_sendMsg _) h
    obtain ⟨-, rfl⟩ := lpure_go_ok h
    exact hl₁

theorem oodIntroP_two (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ} (hdim : dim ≠ 0)
    {st st' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} {u : Unit}
    (h : (((oodIntroP (asksS resp) nonce g dim).run st).run t).go l = (.ok ((u, st'), t'), l')) :
    l'.size = l.size + 2 := by
  unfold oodIntroP at h
  obtain ⟨_, l₁, hl₁, h⟩ := lsampleVecP_bind resp hdim h
  obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_modify _) h
  obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_liftTM _) h
  obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_sendMsg _) h
  obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_powB nonce) h
  obtain ⟨_, l₂, hl₂, h⟩ := lsampleP_bind resp h
  rw [lsilent_modify _ _ _ _ _ _ _ _ h]
  omega

theorem queryP_two (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit)
    (hq : lv.q ≠ 0) {st st' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg} {a : LevelProof × Array F128}
    (h : (((queryP (asksS resp) nonce ms lv cm).run st).run t).go l = (.ok ((a, st'), t'), l')) :
    l'.size = l.size + 2 := by
  unfold queryP at h
  obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_powP nonce 0) h
  obtain ⟨_, l₁, hl₁, h⟩ := lsampleVecP_bind resp hq h
  obtain ⟨_, _, _, h⟩ := lsilent_bind (lsilent_powP nonce 1) h
  obtain ⟨_, l₂, hl₂, h⟩ := lsampleVecP_bind resp (Nat.succ_ne_zero _) h
  obtain ⟨-, rfl⟩ := lpure_go_ok h
  omega

/-! ## Ligerito -/

/-- **A masked Ligerito prover that returns sent `k_0 + 7 + 11 R` bodies** over its `R + 1` levels, and level 0's pair
coins, the replies to sends `|l| + 2 + k_0` and `|l| + 3 + k_0`, are independent. -/
theorem ligeritoPZk_exact (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t t' : Transcript} {l l' : Array Claims.Msg}
    {r : LigeritoProof × Zk.Level0Msgs × Zk.L0Rec}
    (hrun : ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b lay hv rep tPad extra lanePad salt
      nonce).run t).go l = (.ok (r, t'), l')) :
    l'.size = l.size + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) ∧
      Zk.rhoIndependent (coinKOf (resp (l.size + 2 + sch.k0)), coinKOf (resp (l.size + 3 + sch.k0))) = true := by
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨R, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨-, rfl⟩ := Prod.mk.inj h2
  obtain ⟨tA, lA, hlA, h1⟩ := lsampleVecP_bind resp hn h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_modify _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_liftTM _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_powB nonce) h1
  obtain ⟨tB, lB, hlB, h1⟩ := lsampleP_bind resp h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_modify _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind lsilent_get h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_sendMsg _) h1
  obtain ⟨s, stC, tC, lC, hLoop, h1⟩ := lp_bind_ok h1
  have hI := lforIn_inv _ (fun j (_ : Array F256 × (F256 × F256) × Zk.Level0Msgs) (l' : Array Claims.Msg) =>
      l'.size = l.size + 2 + j + (if j = sch.k0 then 2 else 0) ∧
      (j = sch.k0 → Zk.rhoIndependent (coinKOf (resp (l.size + 2 + sch.k0)),
        coinKOf (resp (l.size + 3 + sch.k0))) = true)) sch.k0
    ?step ⟨by rw [ite_eq_right (by omega)]; omega, fun h => absurd h (by omega)⟩ hLoop
  case step =>
    rintro j hj s st t l₀ r st' t' l₂ ⟨hl₀, -⟩ hb
    rw [ite_eq_right (by omega), Nat.add_zero] at hl₀
    obtain ⟨t₃, l₃, hl₃, hb⟩ := lsampleKP_bind resp hb
    obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
    by_cases hj1 : j + 1 = sch.k0
    · rw [ite_eq_left (by simp [hj1])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨t₄, l₄, hl₄, hb⟩ := lsampleKP_bind resp hb
      obtain ⟨t₅, l₅, hl₅, hb⟩ := lsampleKP_bind resp hb
      split at hb
      · obtain ⟨_, _, _, _, h0, _⟩ := lp_bind_ok hb
        cases h0
      rename_i hind
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      refine ⟨_, rfl, by rw [ite_eq_left hj1]; omega, fun _ => ?_⟩
      rw [show l₃.size = l.size + 2 + sch.k0 by omega, show l₄.size = l.size + 3 + sch.k0 by omega] at hind
      simpa using hind
    · rw [ite_eq_right (by simp [hj1])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      exact ⟨_, rfl, by rw [ite_eq_right hj1]; omega, fun h => absurd h hj1⟩
  obtain ⟨hlC, hrho⟩ := hI
  rw [ite_eq_left rfl] at hlC
  refine ⟨?_, hrho rfl⟩
  obtain ⟨_, _, _, h1⟩ := lsilent_bind lsilent_get h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_quiet (commitP_quiet h _ _ _ _)) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_liftTM _) h1
  obtain ⟨_, stD, tD, lD, hO1, h1⟩ := lp_bind_ok h1
  have hlD := oodIntroP_two resp nonce _ (Nat.succ_ne_zero _) hO1
  obtain ⟨_, stE, tE, lE, hO2, h1⟩ := lp_bind_ok h1
  have hlE := oodIntroP_two resp nonce _ (Nat.succ_ne_zero _) hO2
  obtain ⟨⟨op0, v0⟩, stF', tF, lF, hQ, h1⟩ := lp_bind_ok h1
  have hlF := queryP_two resp nonce ms _ cm0 (hq 0 (by omega)) hQ
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_sendMsg _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_powB nonce) h1
  obtain ⟨tG, lG, hlG, h1⟩ := lsampleP_bind resp h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_modify _) h1
  obtain ⟨s2, stH, tH, lH, hLoop2, h1⟩ := lp_bind_ok h1
  obtain ⟨-, rfl⟩ := lpure_go_ok h1
  have hI2 := lforIn_inv _ (fun i _ (l'' : Array Claims.Msg) =>
      l''.size + (if i = sch.levels.size - 1 then 4 else 0) = lG.size + 11 * i) (sch.levels.size - 1)
    ?step2 (by rw [ite_eq_right (by omega)]) hLoop2
  case step2 =>
    rintro i hi ⟨sd, pc, cps, rps, fn⟩ st t l₀ r st' t' l₂ hl₀ hb
    rw [ite_eq_right (by omega)] at hl₀
    obtain ⟨f, st₁, t₁, l₁, hIn, hb⟩ := lp_bind_ok hb
    have hl₁ := lforIn_inv _ (fun j _ (l'' : Array Claims.Msg) => l''.size = l₀.size + j) 4 ?inner rfl hIn
    case inner =>
      rintro j hj f st t l₃ r st' t' l₄ hl₃ hb
      split at hb
      · obtain ⟨_, l₅, hl₅, hb⟩ := lsampleKP_bind resp hb
        obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
        obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
        obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
        obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
        obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
        obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
        exact ⟨_, rfl, by omega⟩
      · obtain ⟨_, _, _, l₅, hc, hb⟩ := lp_bind_ok hb
        have hl₅ := challengeP_one resp false hc
        obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
        exact ⟨_, rfl, by omega⟩
    by_cases hlast : i + 1 = sch.levels.size - 1
    · rw [ite_eq_right (by simp [hlast])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_forIn_array _ _ _ fun x s =>
        lsilent_bind' (lsilent_liftTM _) fun _ => lsilent_pure _) hb
      obtain ⟨⟨op, v⟩, _, _, l₅, hQ, hb⟩ := lp_bind_ok hb
      have hl₅ := queryP_two resp nonce ms _ _ (hq _ (by omega)) hQ
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_powB nonce) hb
      obtain ⟨_, l₆, hl₆, hb⟩ := lsampleP_bind resp hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      exact ⟨_, rfl, by rw [ite_eq_left hlast]; omega⟩
    · rw [ite_eq_left (by simp [hlast])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_quiet (commitP_quiet h _ _ _ _)) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, l₅, hO1, hb⟩ := lp_bind_ok hb
      have hl₅ := oodIntroP_two resp nonce _ (Nat.succ_ne_zero _) hO1
      obtain ⟨_, _, _, l₆, hO2, hb⟩ := lp_bind_ok hb
      have hl₆ := oodIntroP_two resp nonce _ (Nat.succ_ne_zero _) hO2
      obtain ⟨⟨op, v⟩, _, _, l₇, hQ, hb⟩ := lp_bind_ok hb
      have hl₇ := queryP_two resp nonce ms _ _ (hq _ (by omega)) hQ
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_powB nonce) hb
      obtain ⟨_, l₈, hl₈, hb⟩ := lsampleP_bind resp hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      exact ⟨_, rfl, by rw [ite_eq_right hlast]; omega⟩
  rw [ite_eq_left rfl] at hI2
  omega

/-! ## The inner proof -/

/-- **A masked inner proof that returns sent `4` bodies**, and `β`, the reply to send `|l| + 2`, is not `0`. -/
theorem innerP_exact (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) {t t' : Transcript}
    {l l' : Array Claims.Msg} {r : Zk.InnerProof}
    (hrun : ((innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons).run t).go l = (.ok (r, t'), l')) :
    l'.size = l.size + 4 ∧ (coinOf (resp (l.size + 2)) 0).isZero = false := by
  unfold innerP at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _),
    bind_go (sampleP_go resp _ _)] at hrun
  dsimp only at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _),
    bind_go (sampleVecP_go resp (by simp [Zk.triple]) _ _)] at hrun
  split at hrun
  · rw [StateT.run_bind] at hrun
    obtain ⟨_, _, h0, _⟩ := JsonRT.go_bind_ok hrun
    cases h0
  rw [tauCommitmentWith_ofSha512, bind_go (x := StateT.lift (pure _)) rfl,
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _), bind_go (sampleP_go resp _ _)] at hrun
  split at hrun
  · rw [StateT.run_bind] at hrun
    obtain ⟨_, _, h0, _⟩ := JsonRT.go_bind_ok hrun
    cases h0
  rename_i hb
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _), bind_go (sampleVecP_go resp (by decide) _ _),
    pure_go] at hrun
  obtain ⟨-, rfl⟩ := Prod.mk.inj hrun
  refine ⟨by simp, ?_⟩
  simpa using hb

/-! ## A rep -/

/-- **The rank check, where it returns, passed** on the transcript it ran at. -/
theorem rankCheckP_ok {m kLog : ℕ} {prior : Array (List F128)} {words : List ℕ} {t t' : Transcript}
    {l l' : Array Claims.Msg} {u : Unit}
    (h : ((rankCheckP m kLog prior words : TMW PS Unit).run t).go l = (.ok (u, t'), l')) :
    MaskRank.maskRankOk (prior ++ MaskRank.roundPoints m kLog t.rounds) words = true := by
  unfold rankCheckP at h
  simp only [StateT.run_bind, StateT.run_get, pure_bind] at h
  split at h
  · assumption
  · change ((throw _ : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨_, _, h1, -⟩ := JsonRT.go_bind_ok h
    cases h1

/-- A lifted check of the verifier's monad that returns sends nothing. -/
theorem liftV_bind_ok {v : V α} {f : α → TMW PS β} {t : Transcript} {l l'' : Array Claims.Msg}
    {r : β × Transcript}
    (h : (((StateT.lift (monadLift v : PS α) : TMW PS α) >>= f).run t).go l = (.ok r, l'')) :
    ∃ a, v = .ok a ∧ ((f a).run t).go l = (.ok r, l'') := by
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, t₁⟩, l₁, e1, e2⟩ := JsonRT.go_bind_ok h
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at e1
  obtain ⟨b, l₂, e3, e4⟩ := JsonRT.go_bind_ok e1
  rw [JsonRT.monadLift_go] at e3
  simp only [Prod.mk.injEq] at e3
  obtain ⟨hv, rfl⟩ := e3
  cases e4
  exact ⟨_, hv, e2⟩

/-- A masked rep's sends before Ligerito: the zerocheck's `3 + (m − 6)`, the lincheck's `3 + (k_log − 6)` and the ring
switch's `claims + 1`, at its extra claims. -/
def piopZk (st : Setup) (extras : Array Claim) : ℕ := 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + (3 + extras.size)

/-- **A masked rep that returns sent its rounds**, `piopZk` and then Ligerito's `k_0 + 7 + 11 R` and the inner proof's
`4`; level 0's pair coins, Ligerito's sends `k_0 + 2` and `k_0 + 3`, are independent; the inner proof's `β`, its send
`2`, is not `0`; and the rank check passed on the transcript after the lincheck. -/
theorem repPZk_exact (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t t' : Transcript} {l l' : Array Claims.Msg}
    {P : Zk.ZkProof}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run t).go l = (.ok (P, t'), l')) :
    l'.size = l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 4 ∧
      Zk.rhoIndependent (coinKOf (resp (l.size + piopZk st extras + 2 + sch.k0)),
        coinKOf (resp (l.size + piopZk st extras + 3 + sch.k0))) = true ∧
      (coinOf (resp (l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 2)) 0).isZero =
        false ∧
      ∃ full, MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog
        (LcZ.tEnd (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.fold st.m tr
          (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
          (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
          (ZcZ.tEnd (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.m st.digest (flat cm0.cap) ic.root
            (Zc.coinsAt resp l.size) tr t)).rounds) words = true := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest (flat cm0.cap) ic.root
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr t l
  unfold repPZk at h
  rw [bind_go hz] at h
  dsimp only at h
  have h' := h
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ hfold
  rw [bind_go hl] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨u, t₃⟩, l₃', hk, h⟩ := JsonRT.go_bind_ok h
  have hrk := rankCheckP_ok hk
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras) (by simp [abcClaims]) _ l₂
  rw [bind_go hrs] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨lg, zl, rec⟩, t₄⟩, l₄, hlg, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨hl₄, hrho⟩ := ligeritoPZk_exact resp hsh st.merkle sch _ cm0 _ _ _ _ rep tPad extra lanePad salt nonce hn
    hk0 hL hq hlg
  rw [bind_go (show ((get : TMW PS Transcript).run t₄).go l₄ = (.ok (t₄, t₄), l₄) from rfl)] at h
  dsimp only at h
  obtain ⟨⟨rp, x⟩, -, h⟩ := liftV_bind_ok h
  dsimp only at h
  obtain ⟨lgc, -, h⟩ := liftV_bind_ok h
  rw [StateT.run_bind] at h
  obtain ⟨⟨inner, t₅⟩, l₅, hin, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨hl₅, hb⟩ := innerP_exact resp hsh _ _ _ _ _ hin
  obtain ⟨-, rfl⟩ := Prod.mk.inj h
  have hsz : (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size = 2 + extras.size := by simp [abcClaims, Nat.add_comm]
  have e4 : l₄.size = l.size + piopZk st extras + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := by
    unfold piopZk; omega
  have e3 : l₃.size = l.size + piopZk st extras := by unfold piopZk; omega
  refine ⟨by omega, by rw [← e3]; exact hrho, by rw [← e4]; exact hb, full, ?_⟩
  rw [← hl₁]
  exact hrk

end FlockComplete
