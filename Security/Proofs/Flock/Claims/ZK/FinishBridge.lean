import Proofs.Flock.Claims.ZK.FinishRep

/-!
# The firewall's rounds as the session's

`flock-firewall finish` runs a rep in `StateT Nat V`, answering the rep's round `k`, counted from `0`, with the rep's
coins `coins[k]` (`asksOf`). The session runs it in `FlockComplete.PS`, logging every send and answering the session's
round `j`, counted over the whole session, with `coin j` (`FlockComplete.asksCut`). Here one is held to the other:

* **The prover's phases under any simulation** (`squeezeP_rel`, …, `repFwC_rel`): `Flock.With.Sim` relates the runs of
  each of `FlockComplete`'s prover functions that `repFwC` calls in two monads, wherever it relates their sends
  (`AsksRel`) and hashes. Lean has no parametricity theorem, so each is walked once (`sim_walk`), as the honest prover's
  are under `simS`.
* **The state map** (`bridge base`): a port run from round `k` that returns relates to a session run from a log of
  `base + k` sends, which returns the same with a log of `base + k'` sends, `k'` the port's final round.
* **The sends** (`asksOf_rel`): the port's coins, read from the session's round `base` on (`coinsFrom`), are what
  `asksCut` answers, so the port's round `k` is the session's round `base + k` while the cut is past the rep's last.
* **`repFw_cut`**: where `flock-firewall finish`'s rep returns a proof after `k'` rounds, `repFwC` at the session's sends
  from round `base` returns that proof, with `base + k'` sends logged.

The coins are the port's own (`coinsFrom`); that they are the opened coins at the rep's rounds is the transport's, as
is each round's coin count, which the port reads back from the round's header.
-/

namespace Flock.Firewall.Finish

open Flock With Transcript

/-! ## The prover's phases under any simulation -/

section Generic

variable {M N : Type → Type} [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N] [MonadLiftT V M]
  [MonadLiftT V N]

/-- **Two senders a simulation relates**: each send of the same message. -/
structure AsksRel (S : Sim M N) (io : FlockComplete.Asks M) (io' : FlockComplete.Asks N) : Prop where
  link : ∀ a b, S.R (io.link a b) (io'.link a b)
  round : ∀ c, S.R (io.round c) (io'.round c)

theorem ofSha512_rel (S : Sim M N) (h : ByteArray → ByteArray) : S.Hashes (Hashes.ofSha512 h) (Hashes.ofSha512 h) where
  sha512 _ := S.pure _
  sha256 _ := S.pure _
  blake3 _ := S.pure _
  blake3Keyed _ _ := S.pure _

variable (S : Sim M N) {io : FlockComplete.Asks M} {io' : FlockComplete.Asks N}

theorem squeezeP_rel (hio : AsksRel S io io') (kind : UInt8) (n : ℕ) :
    (S.st Transcript).R (FlockComplete.squeezeP io kind n) (FlockComplete.squeezeP io' kind n) := by
  intro t
  unfold FlockComplete.squeezeP
  sim_walk [hio.round]

theorem sampleP_rel (hio : AsksRel S io io') :
    (S.st Transcript).R (FlockComplete.sampleP io) (FlockComplete.sampleP io') := by
  unfold FlockComplete.sampleP
  sim_walk [squeezeP_rel S hio]

theorem sampleKP_rel (hio : AsksRel S io io') :
    (S.st Transcript).R (FlockComplete.sampleKP io) (FlockComplete.sampleKP io') := by
  unfold FlockComplete.sampleKP
  sim_walk [squeezeP_rel S hio]

theorem liftT_rel {α : Type} {x : TMW M α} {y : TMW N α} (h : (S.st Transcript).R x y) :
    ((S.st Transcript).st FlockComplete.LigState).R (FlockComplete.liftT x) (FlockComplete.liftT y) :=
  Sim.lift_st _ h

theorem sendMsg_rel (m : F256 × F256) :
    ((S.st Transcript).st FlockComplete.LigState).R (FlockComplete.sendMsg m : FlockComplete.LP M Unit)
      (FlockComplete.sendMsg m) := by
  unfold FlockComplete.sendMsg
  sim_walk [liftT_rel S]

theorem powP_rel (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((S.st Transcript).st FlockComplete.LigState).R (FlockComplete.powP nonce kind : FlockComplete.LP M Unit)
      (FlockComplete.powP nonce kind) := by
  unfold FlockComplete.powP
  sim_walk [liftT_rel S]

theorem challengeP_rel (hio : AsksRel S io io') (split : Bool) :
    ((S.st Transcript).st FlockComplete.LigState).R (FlockComplete.challengeP io split)
      (FlockComplete.challengeP io' split) := by
  unfold FlockComplete.challengeP
  sim_walk [liftT_rel S, sampleKP_rel S hio, sendMsg_rel S]

theorem oodIntroP_rel (hio : AsksRel S io io') (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ((S.st Transcript).st FlockComplete.LigState).R (FlockComplete.oodIntroP io nonce g dim)
      (FlockComplete.oodIntroP io' nonce g dim) := by
  unfold FlockComplete.oodIntroP FlockComplete.sampleVecP
  sim_walk [liftT_rel S, squeezeP_rel S hio, sampleP_rel S hio, sendMsg_rel S, powP_rel S]

theorem queryP_rel (hio : AsksRel S io io') (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level)
    (cm : FlockComplete.Commit) :
    ((S.st Transcript).st FlockComplete.LigState).R (FlockComplete.queryP io nonce ms lv cm)
      (FlockComplete.queryP io' nonce ms lv cm) := by
  unfold FlockComplete.queryP FlockComplete.sampleVecP
  sim_walk [liftT_rel S, squeezeP_rel S hio, powP_rel S]

theorem commitP_rel [LawfulMonad M] [LawfulMonad N] {hs : Hashes M} {hs' : Hashes N} (hh : S.Hashes hs hs')
    (ms : MerkleScheme) (lv : Level) (g : Array F128) (salt : ℕ → ByteArray) :
    S.R (FlockComplete.commitP hs ms lv g salt) (FlockComplete.commitP hs' ms lv g salt) := by
  have ho := opsWith_sim S hh ms
  unfold FlockComplete.commitP
  sim_walk [ho.leaf, ho.node]

theorem ligeritoPZk_rel [LawfulMonad M] [LawfulMonad N] (hio : AsksRel S io io') {hs : Hashes M} {hs' : Hashes N} (hh : S.Hashes hs hs')
    (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cm0 : FlockComplete.Commit) (t b : Array F128)
    (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) :
    (S.st Transcript).R (FlockComplete.ligeritoPZk io hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (FlockComplete.ligeritoPZk io' hs' ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce) := by
  unfold FlockComplete.ligeritoPZk FlockComplete.sampleVecP
  sim_walk [liftT_rel S, squeezeP_rel S hio, sampleP_rel S hio, sampleKP_rel S hio, sendMsg_rel S, powP_rel S,
    challengeP_rel S hio, oodIntroP_rel S hio, queryP_rel S hio, commitP_rel S hh]

theorem innerP_rel (hio : AsksRel S io io') {hs : Hashes M} {hs' : Hashes N} (hh : S.Hashes hs hs')
    (lay : Zk.PadLayout) (p : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (finalC : F128)
    (cons : Array (String × Zk.Lin)) :
    (S.st Transcript).R (FlockComplete.innerP io hs lay p ic finalC cons)
      (FlockComplete.innerP io' hs' lay p ic finalC cons) := by
  unfold FlockComplete.innerP FlockComplete.sampleVecP
  sim_walk [sampleP_rel S hio, squeezeP_rel S hio, zk_tauCommitmentWith_sim S hh]

theorem zerocheckCut_rel (hio : AsksRel S io io') (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (mk : ℕ → F128) :
    (S.st Transcript).R (FlockComplete.zerocheckCut io m digest cap root lay mk)
      (FlockComplete.zerocheckCut io' m digest cap root lay mk) := by
  unfold FlockComplete.zerocheckCut FlockComplete.sampleVecP
  sim_walk [squeezeP_rel S hio, sampleP_rel S hio]

theorem lincheckCut_rel (hio : AsksRel S io io') (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128)
    (zc : ZcOut) :
    (S.st Transcript).R (FlockComplete.lincheckCut io c m lay mk zc) (FlockComplete.lincheckCut io' c m lay mk zc) := by
  unfold FlockComplete.lincheckCut
  sim_walk [sampleP_rel S hio]

theorem ringSwitchCut_rel (hio : AsksRel S io io') (sv : ℕ → Array F128) (claims : Array Claim) :
    (S.st Transcript).R (FlockComplete.ringSwitchCut io sv claims) (FlockComplete.ringSwitchCut io' sv claims) := by
  unfold FlockComplete.ringSwitchCut FlockComplete.sampleVecP
  sim_walk [squeezeP_rel S hio]

/-- **The firewall's rep under a simulation** that relates its sends and hashes. -/
theorem repFwC_rel [LawfulMonad M] [LawfulMonad N] (hio : AsksRel S io io') {hs : Hashes M} {hs' : Hashes N}
    (hh : S.Hashes hs hs') (st : Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128)
    (cm0 : FlockComplete.Commit) (extras : Array Claim) (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit)
    (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    (S.st Transcript).R (repFwC io hs st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt)
      (repFwC io' hs' st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt) := by
  have h5 := ofSha512_rel S Sha512.hash
  unfold repFwC
  sim_walk [zerocheckCut_rel S hio, lincheckCut_rel S hio, ringSwitchCut_rel S hio, ligeritoPZk_rel S hio h5,
    innerP_rel S hio hh]

end Generic

/-! ## The state map -/

/-- **The port's rounds as the session's from round `base`**: a port run from round `k` that returns `a` at round `k'`
relates to a session run from a log of `base + k` sends, which returns `a` with a log of `base + k'`. -/
def bridge (base : ℕ) : Sim (StateT ℕ V) FlockComplete.PS where
  R x y := ∀ k (l : Array Claims.Msg), l.size = base + k → ∀ a k', x.run k = .ok (a, k') →
    (y.go l).1 = .ok a ∧ (y.go l).2.size = base + k'
  pure _ _ _ hl _ _ h := by
    cases h
    exact ⟨rfl, hl⟩
  bind {α β x y f g} hxy hfg k l hl b k'' h := by
    rw [StateT.run_bind] at h
    rcases hx : x.run k with _ | ⟨a, k1⟩
    · rw [hx] at h; cases h
    · rw [hx] at h
      obtain ⟨h1, h2⟩ := hxy k l hl a k1 hx
      simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind] at h1 h2 ⊢
      rcases hr : y.run.run l with ⟨r, l1⟩
      rw [hr] at h1 h2
      dsimp only at h1 h2
      subst h1
      exact hfg a k1 l1 h2 b k'' h
  throw _ _ _ _ _ _ h := by cases h
  lift v _ _ hl _ _ h := by
    cases v with
    | error e => cases h
    | ok a0 =>
      cases h
      exact ⟨rfl, hl⟩

/-- **The port's coins as the session's** from round `base` on: the session's round `j` is the rep's round `j - base`. -/
def coinsFrom (base : ℕ) (coins : Array (Array F128)) (j i : ℕ) : F128 := (coins[j - base]!)[i]!

theorem asksCut_round_fw (coin : ℕ → ℕ → F128) (n : ℕ) (c : ByteArray) (l : Array Claims.Msg) :
    ((FlockComplete.asksCut coin n).round c).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size), l.push (Claims.toMsg c)) := by
  have e : ((FlockComplete.asksCut coin n).round c).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : FlockComplete.PS _)).go
        (l.push (Claims.toMsg c)) := rfl
  rw [e]
  split_ifs <;> rfl

/-- **The port's sends are the cut's** from round `base`, while the cut is past the rep's last round. -/
theorem asksOf_rel (coins : Array (Array F128)) (base n : ℕ) (hn : base + coins.size ≤ n) :
    AsksRel (bridge base) (asksOf coins).toC (FlockComplete.asksCut (coinsFrom base coins) n) where
  link _ _ _ _ _ _ _ h := by cases h
  round c k l hl a k' h := by
    change ((asksOf coins).round c).run k = _ at h
    dsimp only [asksOf] at h
    rw [StateT.run_bind, StateT.run_get, pure_bind] at h
    rcases hk : coins[k]? with _ | cs
    · rw [hk] at h; cases h
    · rw [hk] at h
      dsimp only at h
      have hk' : k < coins.size := (Array.getElem?_eq_some_iff.1 hk).1
      have hcs : coins[k]! = cs := by
        rw [getElem!_pos coins k hk']
        exact (Array.getElem?_eq_some_iff.1 hk).2
      split at h
      · rw [StateT.run_bind] at h
        cases h
      · rw [StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure] at h
        cases h
        have hlt : ¬ n ≤ l.size := by omega
        rw [asksCut_round_fw, if_neg hlt]
        refine ⟨?_, ?_⟩
        · show Except.ok (coinsFrom base coins l.size) = Except.ok (fun i => cs[i]!)
          congr 1
          funext i
          simp only [coinsFrom, hl, Nat.add_sub_cancel_left, hcs]
        · show (l.push (Claims.toMsg c)).size = base + (k + 1)
          rw [Array.size_push, hl]
          rfl

/-- **The firewall's rep is the session's rounds from `base`**: where `flock-firewall finish`'s rep (`repFw` at
`commitFw`, answered by its coins from round `0`, in `StateT Nat V`) returns a proof and transcript at round `k'`,
`repFwC` answered by `asksCut` at those coins from the session's round `base`, from a log of `base` sends, returns the
same with a log of `base + k'`, wherever the cut is past the rep's last round. -/
theorem repFw_cut (coins : Array (Array F128)) (base n : ℕ) (hn : base + coins.size ≤ n) (st : Setup)
    (hm : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (hps : pads.h.size = (Zk.PadLayout.new st.m st.kLog sch.k0).len)
    (hab : pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).hab]! =
      pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fa]! * pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fb]!)
    (htw : pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).tw]! =
      pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).tx]! * pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).ty]!)
    (tr : Transcript) {p : Zk.ZkProof} {t : Transcript} {k' : ℕ}
    (h : ((repFw (asksOf coins) (Hashes.ofSha512 Sha512.hash) (fun lv g s => pure (commitFw lv g s)) st sch tPad rep
      dom g cm0 extras pads ic vals extra lanePad salt).run tr).run 0 = .ok ((p, t), k'))
    (l : Array Claims.Msg) (hl : l.size = base) :
    (((repFwC (FlockComplete.asksCut (coinsFrom base coins) n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g
        cm0.toC extras pads.toC ic.toC vals extra lanePad salt).run tr).go l).1 = .ok (p, t) ∧
      (((repFwC (FlockComplete.asksCut (coinsFrom base coins) n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g
        cm0.toC extras pads.toC ic.toC vals extra lanePad salt).run tr).go l).2.size = base + k' := by
  rw [repFw_eq (asksOf coins) _ st hm sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt hps hab htw
    (fun e f => throw_bind_sv e f)] at h
  exact repFwC_rel (bridge base) (asksOf_rel coins base n hn) (ofSha512_rel _ Sha512.hash) st sch tPad rep dom g
    cm0.toC extras pads.toC ic.toC vals extra lanePad salt tr 0 l (by rw [hl]; rfl) (p, t) k' h

end Flock.Firewall.Finish
