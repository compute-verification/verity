import Proofs.Flock.Claims.ZK.FinishBridge
import Proofs.Flock.Claims.ZK.FrameAsks
import Proofs.Flock.Claims.Complete.Zk.RepRounds

/-!
# The firewall's rounds as the session's, at coins that agree below each round's count

`repFw_cut` answers the session's rounds with the port's own coins (`coinsFrom`), which are `0` past each round's count.
The session's opened coins are not: an opening carries its round's `n` coins and then more bytes, all of which its coin
reads. A squeeze reads only its first `n` coins, and the port checks `n` against its coins' count (`asksOf`, at the round
header's `u64`), so the two runs agree wherever every squeeze is below `2^64`:

* **A squeeze** (`squeezeP_br`): where the port's round returns, it read coins of the squeeze's count (`asksOf_round_ok`,
  `Count.framed_count`), and the session's coin agrees with them there.
* **Each phase of the rep, given its squeezes' sizes** (`sampleP_br`, …, `innerP_br`). The recursive levels'
  out-of-domain sizes ride Ligerito's loop (`loopG_bridge`), whose invariant holds of what the port's step returns
  (`PRet`), with no walk of the step's calls.
* **`repFwC_br`**: `repFwC` at the port's coins and at any coins that agree with them below each round's count, from the
  session's round `base`, given `m − 13`, the claims, `n + 1` and every level's `q + 1` bounded.
-/

namespace Flock.Firewall.Finish

open Flock With Transcript

theorem lt64 {a : ℕ} (h : a ≤ 65536) : a < 2 ^ 64 := lt_of_le_of_lt h (by norm_num)

theorem cons_size_le {α : Type} {a c : Array α} {x y : α} (ha : a.size = 3) (hc : c.size = 2) :
    (a ++ #[x, y] ++ c).size + 3 ≤ 65536 := by
  have hxy : (#[x, y] : Array α).size = 2 := rfl
  rw [Array.size_append, Array.size_append, ha, hxy, hc] <;> omega

/-- A `forIn` over a range in any monad, named so that a walk stops at it. -/
def loopG {m : Type → Type} [Monad m] {β : Type} (r : Std.Legacy.Range) (b : β) (f : ℕ → β → m (ForInStep β)) : m β :=
  forIn r b f

theorem forIn_loopG {m : Type → Type} [Monad m] (r : Std.Legacy.Range)
    (b : ℕ × FlockComplete.Commit × Array (Array ByteArray) × Array LevelProof × FinalProof)
    (f : ℕ → ℕ × FlockComplete.Commit × Array (Array ByteArray) × Array LevelProof × FinalProof →
      m (ForInStep (ℕ × FlockComplete.Commit × Array (Array ByteArray) × Array LevelProof × FinalProof))) :
    forIn r b f = loopG r b f := rfl

theorem pure_st {M N : Type → Type} [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N]
    [MonadLiftT V M] [MonadLiftT V N] (S : Sim M N) {σ α : Type} (a : α) (s : σ) :
    S.R ((Pure.pure a : StateT σ M α) s) ((Pure.pure a : StateT σ N α) s) :=
  S.pure (a, s)

/-! ## The bridge, with what the port's run returns -/

theorem bridge_bind_ok {base : ℕ} {α β : Type} {x : StateT ℕ V α} {y : FlockComplete.PS α} {f : α → StateT ℕ V β}
    {g : α → FlockComplete.PS β} (Q : α → Prop) (hxy : (bridge base).R x y)
    (hQ : ∀ k a k', x.run k = .ok (a, k') → Q a) (hfg : ∀ a, Q a → (bridge base).R (f a) (g a)) :
    (bridge base).R (x >>= f) (y >>= g) := by
  intro k l hl b k'' h
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
    exact hfg a (hQ k a k1 hx) k1 l1 h2 b k'' h

theorem bridgeT_bind_ok {base : ℕ} {α β : Type} {x : TMW (StateT ℕ V) α} {y : TMW FlockComplete.PS α}
    {f : α → TMW (StateT ℕ V) β} {g : α → TMW FlockComplete.PS β} (Q : α → Prop)
    (hxy : ((bridge base).st Transcript).R x y)
    (hQ : ∀ t k a t' k', (x.run t).run k = .ok ((a, t'), k') → Q a)
    (hfg : ∀ a, Q a → ((bridge base).st Transcript).R (f a) (g a)) :
    ((bridge base).st Transcript).R (x >>= f) (y >>= g) := fun t =>
  bridge_bind_ok (fun p => Q p.1) (hxy t) (fun k p k' h => hQ t k p.1 p.2 k' h) fun p hp => hfg p.1 hp p.2

theorem bridgeL_bind_ok {base : ℕ} {α β : Type} {x : FlockComplete.LP (StateT ℕ V) α}
    {y : FlockComplete.LP FlockComplete.PS α} {f : α → FlockComplete.LP (StateT ℕ V) β}
    {g : α → FlockComplete.LP FlockComplete.PS β} (Q : α → Prop)
    (hxy : (((bridge base).st Transcript).st FlockComplete.LigState).R x y)
    (hQ : ∀ s t k a s' t' k', ((x.run s).run t).run k = .ok (((a, s'), t'), k') → Q a)
    (hfg : ∀ a, Q a → (((bridge base).st Transcript).st FlockComplete.LigState).R (f a) (g a)) :
    (((bridge base).st Transcript).st FlockComplete.LigState).R (x >>= f) (y >>= g) := fun s =>
  bridgeT_bind_ok (fun p => Q p.1) (hxy s) (fun t k p t' k' h => hQ s t k p.1 p.2 t' k' h) fun p hp =>
    hfg p.1 hp p.2

/-- A lifted verdict's value: what follows it is related at the value it returns. -/
theorem bridgeT_liftV_bind {base : ℕ} {α β : Type} (v : V α) {f : α → TMW (StateT ℕ V) β}
    {g : α → TMW FlockComplete.PS β} (hfg : ∀ a, v = .ok a → ((bridge base).st Transcript).R (f a) (g a)) :
    ((bridge base).st Transcript).R (StateT.lift (monadLift v : StateT ℕ V α) >>= f)
      (StateT.lift (monadLift v : FlockComplete.PS α) >>= g) := by
  refine bridgeT_bind_ok (fun a => v = .ok a) (With.Sim.lift_st _ ((bridge base).lift v)) ?_ hfg
  intro t k a t' k' e
  revert e
  cases v with
  | error e0 => intro e; cases e
  | ok b => intro e; cases e; rfl

/-- **What a Ligerito step of the port returns**, wherever it returns. -/
def PRet {α : Type} (x : FlockComplete.LP (StateT ℕ V) α) (Q : α → Prop) : Prop :=
  ∀ s t k a s' t' k', ((x.run s).run t).run k = .ok (((a, s'), t'), k') → Q a

theorem PRet.pure {α : Type} {a : α} {Q : α → Prop} (h : Q a) : PRet (Pure.pure a) Q := by
  intro s t k a' s' t' k' e
  cases e
  exact h

theorem PRet.bind {α β : Type} {x : FlockComplete.LP (StateT ℕ V) α} {f : α → FlockComplete.LP (StateT ℕ V) β}
    {Q : β → Prop} (hf : ∀ a, PRet (f a) Q) : PRet (x >>= f) Q := by
  have bok : ∀ {γ δ : Type} {y : V γ} {g : γ → V δ} {d : δ}, y >>= g = .ok d → ∃ c, y = .ok c ∧ g c = .ok d := by
    intro γ δ y g d h
    cases y with
    | error e => cases h
    | ok c => exact ⟨c, rfl, h⟩
  intro s t k b s'' t'' k'' e
  simp only [StateT.run_bind] at e
  obtain ⟨⟨⟨⟨a, s'⟩, t'⟩, k'⟩, -, e⟩ := bok e
  exact hf a s' t' k' b s'' t'' k'' e

/-- **A loop whose port steps keep an invariant of its state is related**, its steps related at the invariant. -/
theorem loopG_bridge {base : ℕ} {β : Type} (I : β → Prop) (r : Std.Legacy.Range) (b : β)
    (f : ℕ → β → FlockComplete.LP (StateT ℕ V) (ForInStep β))
    (g : ℕ → β → FlockComplete.LP FlockComplete.PS (ForInStep β)) (hb : I b)
    (hI : ∀ i b, I b → PRet (f i b) fun s => I s.value)
    (hfg : ∀ i b, I b → (((bridge base).st Transcript).st FlockComplete.LigState).R (f i b) (g i b)) :
    (((bridge base).st Transcript).st FlockComplete.LigState).R (loopG r b f) (loopG r b g) := by
  unfold loopG
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  generalize List.range' r.start r.size r.step = xs
  induction xs generalizing b with
  | nil => exact With.Sim.pure _ b
  | cons a xs ih =>
    rw [List.forIn_cons, List.forIn_cons]
    refine bridgeL_bind_ok (fun s => I s.value) (hfg a b hb)
      (fun s t k st s' t' k' e => hI a b hb s t k st s' t' k' e) fun st hst => ?_
    cases st with
    | done b' => exact With.Sim.pure _ b'
    | yield b' => exact ih b' hst

/-! ## A squeeze -/

/-- **A round of the port that returns** read the rep's coins at its round, of the header's count. -/
theorem asksOf_round_ok {coins : Array (Array F128)} {c : ByteArray} {k k1 : ℕ} {co : ℕ → F128}
    (h : ((asksOf coins).round c).run k = .ok (co, k1)) :
    ∃ cs, coins[k]? = some cs ∧ cs.size = (u64OfLE c (c.size - 8)).toNat ∧ co = (fun i => cs[i]!) ∧ k1 = k + 1 := by
  dsimp only [asksOf] at h
  rw [StateT.run_bind, StateT.run_get, pure_bind] at h
  rcases hk : coins[k]? with _ | cs
  · rw [hk] at h; cases h
  · rw [hk] at h
    dsimp only at h
    split at h
    · rw [StateT.run_bind] at h
      cases h
    · rename_i hne
      rw [StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure] at h
      cases h
      refine ⟨cs, rfl, ?_, rfl, rfl⟩
      simpa using hne

section

variable (coins : Array (Array F128)) (base N : ℕ) (coin : ℕ → ℕ → F128) (hn : base + coins.size ≤ N)
  (hc : ∀ k < coins.size, ∀ i < (coins[k]!).size, coin (base + k) i = (coins[k]!)[i]!)

include hn hc

/-- **A squeeze below `2^64` coins is related**: where the port's round returns, it read the rep's coins at its round, of
the squeeze's count, and the session's coin agrees with them there. -/
theorem squeezeP_br (kind : UInt8) {m : ℕ} (hm : m < 2 ^ 64) :
    ((bridge base).st Transcript).R (FlockComplete.squeezeP (asksOf coins).toC kind m)
      (FlockComplete.squeezeP (FlockComplete.asksCut coin N) kind m) := by
  intro t
  unfold FlockComplete.squeezeP
  dsimp only
  split
  · exact (bridge base).pure _
  · intro k l hl a k' h
    rw [StateT.run_bind] at h
    rcases hr : ((asksOf coins).toC.round (Transcript.framed t kind m)).run k with _ | ⟨co, k1⟩
    · rw [hr] at h; cases h
    · rw [hr] at h
      cases h
      obtain ⟨cs, hk, hsz, rfl, rfl⟩ := asksOf_round_ok (coins := coins) (c := Transcript.framed t kind m) hr
      rw [Claims.ZKServer.Count.framed_count t kind hm] at hsz
      have hk' : k < coins.size := (Array.getElem?_eq_some_iff.1 hk).1
      have hcs : coins[k]! = cs := by
        rw [getElem!_pos coins k hk']
        exact (Array.getElem?_eq_some_iff.1 hk).2
      have hlt : ¬ N ≤ l.size := by omega
      have hv : (Array.range m).map (coin l.size) = (Array.range m).map (fun i => cs[i]!) :=
        Array.map_congr_left fun i hi => by
          rw [Array.mem_range] at hi
          rw [hl, hc k hk' i (by rw [hcs]; omega), hcs]
      simp only [FlockComplete.go_bind, asksCut_round_fw, if_neg hlt, hv]
      refine ⟨rfl, ?_⟩
      change (Array.push l _).size = _
      rw [Array.size_push, hl] <;> omega

theorem sampleP_br :
    ((bridge base).st Transcript).R (FlockComplete.sampleP (asksOf coins).toC)
      (FlockComplete.sampleP (FlockComplete.asksCut coin N)) := by
  unfold FlockComplete.sampleP
  sim_walk [squeezeP_br coins base N coin hn hc _ (show 1 < 2 ^ 64 by norm_num)]

theorem sampleKP_br :
    ((bridge base).st Transcript).R (FlockComplete.sampleKP (asksOf coins).toC)
      (FlockComplete.sampleKP (FlockComplete.asksCut coin N)) := by
  unfold FlockComplete.sampleKP
  sim_walk [squeezeP_br coins base N coin hn hc _ (show 2 < 2 ^ 64 by norm_num)]

theorem challengeP_br (split : Bool) :
    (((bridge base).st Transcript).st FlockComplete.LigState).R (FlockComplete.challengeP (asksOf coins).toC split)
      (FlockComplete.challengeP (FlockComplete.asksCut coin N) split) := by
  unfold FlockComplete.challengeP
  sim_walk [liftT_rel (bridge base), sampleKP_br coins base N coin hn hc, sendMsg_rel (bridge base)]

theorem oodIntroP_br (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) (hd : dim ≤ 65536) :
    (((bridge base).st Transcript).st FlockComplete.LigState).R
      (FlockComplete.oodIntroP (asksOf coins).toC nonce g dim)
      (FlockComplete.oodIntroP (FlockComplete.asksCut coin N) nonce g dim) := by
  unfold FlockComplete.oodIntroP FlockComplete.sampleVecP
  sim_walk [liftT_rel (bridge base), squeezeP_br coins base N coin hn hc _ (lt64 hd),
    sampleP_br coins base N coin hn hc, sendMsg_rel (bridge base), powP_rel (bridge base)]

theorem oodIntroP_br_loop (nonce : ℕ → ℕ → UInt64) (M : ℕ) (hM : M ≤ 65536) (g : Array F128)
    (b : ℕ × FlockComplete.Commit × Array (Array ByteArray) × Array LevelProof × FinalProof) (hb : b.1 ≤ M) :
    (((bridge base).st Transcript).st FlockComplete.LigState).R
      (FlockComplete.oodIntroP (asksOf coins).toC nonce g (b.1 - 4 + 1))
      (FlockComplete.oodIntroP (FlockComplete.asksCut coin N) nonce g (b.1 - 4 + 1)) :=
  oodIntroP_br coins base N coin hn hc nonce g _ (by omega)

theorem queryP_br (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : FlockComplete.Commit)
    (hq : lv.q ≤ 65536) (hl : Nat.log2 (lv.q - 1) + 1 ≤ 65536) :
    (((bridge base).st Transcript).st FlockComplete.LigState).R
      (FlockComplete.queryP (asksOf coins).toC nonce ms lv cm)
      (FlockComplete.queryP (FlockComplete.asksCut coin N) nonce ms lv cm) := by
  unfold FlockComplete.queryP FlockComplete.sampleVecP
  sim_walk [liftT_rel (bridge base), squeezeP_br coins base N coin hn hc _ (lt64 hq),
    squeezeP_br coins base N coin hn hc _ (lt64 hl), powP_rel (bridge base)]

theorem queryP_br_lv (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (sch : Flock.Schedule)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536) (j : ℕ) (cm : FlockComplete.Commit) :
    (((bridge base).st Transcript).st FlockComplete.LigState).R
      (FlockComplete.queryP (asksOf coins).toC nonce ms (sch.levels.getD j default) cm)
      (FlockComplete.queryP (FlockComplete.asksCut coin N) nonce ms (sch.levels.getD j default) cm) :=
  queryP_br coins base N coin hn hc nonce ms _ cm (by have := hq j; omega)
    (by have := hq j; have := Nat.log2_le_self ((sch.levels.getD j default).q - 1); omega)

/-- **The masked Ligerito is related**, given `n + 1` and every level's `q + 1` at most `65536`. -/
theorem ligeritoPZk_br (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cm0 : FlockComplete.Commit)
    (t b : Array F128) (lay : Zk.PadLayout) (hp : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hN : sch.n + 1 ≤ 65536)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536) :
    ((bridge base).st Transcript).R
      (FlockComplete.ligeritoPZk (asksOf coins).toC (Hashes.ofSha512 Sha512.hash) ms sch T cm0 t b lay hp rep tPad
        extra lanePad salt nonce)
      (FlockComplete.ligeritoPZk (FlockComplete.asksCut coin N) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 t b lay
        hp rep tPad extra lanePad salt nonce) := by
  have h5 := ofSha512_rel (bridge base) Sha512.hash
  unfold FlockComplete.ligeritoPZk FlockComplete.sampleVecP
  simp (config := { zeta := false }) only [forIn_loopG]
  sim_walk [liftT_rel (bridge base), squeezeP_br coins base N coin hn hc _ (lt64 (show sch.n ≤ 65536 by omega)),
    sampleP_br coins base N coin hn hc, sampleKP_br coins base N coin hn hc, sendMsg_rel (bridge base),
    powP_rel (bridge base), challengeP_br coins base N coin hn hc,
    oodIntroP_br coins base N coin hn hc nonce _ (sch.n - sch.k0 + 1) (by omega),
    queryP_br_lv coins base N coin hn hc nonce ms sch hq, commitP_rel (bridge base) h5,
    pure_st (bridge base), pure_st ((bridge base).st Transcript),
    oodIntroP_br_loop coins base N coin hn hc nonce (sch.n + 1) hN,
    loopG_bridge (fun s : ℕ × FlockComplete.Commit × Array (Array ByteArray) × Array LevelProof × FinalProof =>
      s.1 ≤ sch.n + 1)]
  -- `sim_walk` stops at the first goal it can't walk (the loop's start bound, its `PRet` invariant), leaving the loop's
  -- step and what follows it to the rounds below
  iterate 3
    all_goals first
      | exact fun _ => ((bridge base).st Transcript).pure _
      | (simp (config := { zetaDelta := true }) only []
         omega)
      | (repeat (first
           | extract_lets
           | (refine PRet.bind fun _ => ?_)
           | (refine PRet.pure ?_
              simp (config := { zetaDelta := true }) only [ForInStep.value]
              omega)
           | split
           | (intro _))
         done)
      | sim_walk [liftT_rel (bridge base), squeezeP_br coins base N coin hn hc _ (lt64 (show sch.n ≤ 65536 by omega)),
      sampleP_br coins base N coin hn hc, sampleKP_br coins base N coin hn hc, sendMsg_rel (bridge base),
      powP_rel (bridge base), challengeP_br coins base N coin hn hc,
      oodIntroP_br coins base N coin hn hc nonce _ (sch.n - sch.k0 + 1) (by omega),
      queryP_br_lv coins base N coin hn hc nonce ms sch hq, commitP_rel (bridge base) h5,
      pure_st (bridge base), pure_st ((bridge base).st Transcript),
      oodIntroP_br_loop coins base N coin hn hc nonce (sch.n + 1) hN,
      loopG_bridge (fun s : ℕ × FlockComplete.Commit × Array (Array ByteArray) × Array LevelProof × FinalProof =>
        s.1 ≤ sch.n + 1)]

theorem zerocheckCut_br (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : ℕ → F128)
    (hm : m - 13 ≤ 65536) :
    ((bridge base).st Transcript).R (FlockComplete.zerocheckCut (asksOf coins).toC m digest cap root lay mk)
      (FlockComplete.zerocheckCut (FlockComplete.asksCut coin N) m digest cap root lay mk) := by
  unfold FlockComplete.zerocheckCut FlockComplete.sampleVecP
  sim_walk [squeezeP_br coins base N coin hn hc _ (show 6 < 2 ^ 64 by norm_num),
    squeezeP_br coins base N coin hn hc _ (lt64 hm), sampleP_br coins base N coin hn hc]

theorem lincheckCut_br (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) :
    ((bridge base).st Transcript).R (FlockComplete.lincheckCut (asksOf coins).toC c m lay mk zc)
      (FlockComplete.lincheckCut (FlockComplete.asksCut coin N) c m lay mk zc) := by
  unfold FlockComplete.lincheckCut
  sim_walk [sampleP_br coins base N coin hn hc]

theorem ringSwitchCut_br (sv : ℕ → Array F128) (claims : Array Claim) (hcl : claims.size < 2 ^ 64) :
    ((bridge base).st Transcript).R (FlockComplete.ringSwitchCut (asksOf coins).toC sv claims)
      (FlockComplete.ringSwitchCut (FlockComplete.asksCut coin N) sv claims) := by
  unfold FlockComplete.ringSwitchCut FlockComplete.sampleVecP
  sim_walk [squeezeP_br coins base N coin hn hc _ (show 7 < 2 ^ 64 by norm_num),
    squeezeP_br coins base N coin hn hc _ hcl]

theorem squeezeP_br_cons (lay : Zk.PadLayout) (cons : Array (String × Zk.Lin)) (hcs : cons.size + 3 ≤ 65536)
    (e r s : F128) :
    ((bridge base).st Transcript).R (FlockComplete.squeezeP (asksOf coins).toC 2 (cons ++ Zk.triple lay e r s).size)
      (FlockComplete.squeezeP (FlockComplete.asksCut coin N) 2 (cons ++ Zk.triple lay e r s).size) :=
  squeezeP_br coins base N coin hn hc 2 (lt64 (by simp [Zk.triple]; omega))

theorem innerP_br (lay : Zk.PadLayout) (p : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (finalC : F128)
    (cons : Array (String × Zk.Lin)) (hcs : cons.size + 3 ≤ 65536) :
    ((bridge base).st Transcript).R
      (FlockComplete.innerP (asksOf coins).toC (Hashes.ofSha512 Sha512.hash) lay p ic finalC cons)
      (FlockComplete.innerP (FlockComplete.asksCut coin N) (Hashes.ofSha512 Sha512.hash) lay p ic finalC cons) := by
  have hh := ofSha512_rel (bridge base) Sha512.hash
  unfold FlockComplete.innerP FlockComplete.sampleVecP
  sim_walk [sampleP_br coins base N coin hn hc,
    squeezeP_br coins base N coin hn hc _ (lt64 (show Zk.Q ≤ 65536 by decide)),
    squeezeP_br_cons coins base N coin hn hc lay cons hcs, zk_tauCommitmentWith_sim (bridge base) hh]

/-- **The firewall's rep is related**, at the port's coins and at the session's coins from round `base`, given
`m − 13`, `n + 1` and every level's `q + 1` at most `65536` and the claims below `2^64`. -/
theorem repFwC_br (st : Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128)
    (cm0 : FlockComplete.Commit) (extras : Array Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (hm : st.m - 13 ≤ 65536) (hx : extras.size + 2 < 2 ^ 64)
    (hN : sch.n + 1 ≤ 65536) (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536) :
    ((bridge base).st Transcript).R
      (repFwC (asksOf coins).toC (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g cm0 extras pads ic vals extra
        lanePad salt)
      (repFwC (FlockComplete.asksCut coin N) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g cm0 extras pads ic
        vals extra lanePad salt) := by
  unfold repFwC
  refine With.Sim.bind _ (zerocheckCut_br coins base N coin hn hc _ _ _ _ _ _ hm) fun ⟨zp, zc⟩ => ?_
  refine With.Sim.bind _ (lincheckCut_br coins base N coin hn hc _ _ _ _ _) fun ⟨lp, lc⟩ => ?_
  refine With.Sim.bind _ (ringSwitchCut_br coins base N coin hn hc _ _ ?_) fun ⟨ss, T, sws⟩ => ?_
  · rw [Array.size_append, show (abcClaims st.m st.kLog zc lc).size = 2 from rfl, Nat.add_comm]
    exact hx
  refine With.Sim.bind _ (ligeritoPZk_br coins base N coin hn hc _ sch _ _ _ _ _ _ _ _ _ _ _ _ hN hq)
    fun ⟨lg, zl, rec⟩ => ?_
  refine With.Sim.bind _ (With.Sim.get _) fun tr' => ?_
  refine bridgeT_liftV_bind _ fun ⟨rp, t''⟩ hrp => ?_
  refine bridgeT_liftV_bind _ fun lgc hlgc => ?_
  have h3 := FlockSoundness.Discharge.ZkExec.replay_cons_size _ _ _ _ hrp
  have h2 := FlockSoundness.Discharge.ZkExec.ligeritoConstraints_size hlgc
  sim_walk [innerP_br coins base N coin hn hc, cons_size_le]
  iterate 2
    all_goals first
      | exact cons_size_le h3 h2
      | omega
      | sim_walk [innerP_br coins base N coin hn hc, cons_size_le]

end

end Flock.Firewall.Finish
