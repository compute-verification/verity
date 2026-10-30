import Proofs.Flock.Claims.Complete.Zk.LigeritoProverPos
import Proofs.Flock.Claims.Complete.Reps
import Proofs.Flock.Claims.Complete.Zk.Coins
import Proofs.Flock.Claims.Complete.Json.Goals

/-!
# The masked Ligerito prover returns where its pair's coins are independent

`ligeritoPZk_ret`: the masked Ligerito prover (`ligeritoPZk`), run in the pure session from the bodies `l`, returns
whenever the replies to sends `|l| + 2 + k_0` and `|l| + 3 + k_0`, its pair's coins `ρ` (`ligeritoPZk_pos`), are
independent over `F` (`Zk.rhoIndependent`): that refusal is the one step of the prover that a coin can make throw.

* **Returning from `n` sends** (`LRet`): a Ligerito step returns from every state, transcript and `n` sends, with a
  property of its value and of the sends after. Squeezes are answered by the reply to the send they make
  (`LRet.sampleKP`, `LRet.sampleP`, `LRet.sampleVecP`), the transcript's and the state's steps send nothing
  (`LRet.silent`), and a loop over `[0:n]` keeps an invariant of its state and its sends (`LRet.forIn_upto`).
* After the lane phase the sends no longer matter: the rest returns from any sends (`LRet.ofTot`, Reps' `totL`).
-/

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg JsonRT

variable {α β : Type}

/-- **A Ligerito prover step returns** from every state, transcript and `n` sends, with `P` of its value and of the
number of sends after. -/
def LRet (y : LP PS α) (n : ℕ) (P : α → ℕ → Prop) : Prop :=
  ∀ (st : LigState) (t : Transcript) (l : Array Claims.Msg), l.size = n →
    ∃ a st' t' l', ((y.run st).run t).go l = (.ok ((a, st'), t'), l') ∧ P a l'.size

theorem LRet.pure {a : α} {n : ℕ} {P : α → ℕ → Prop} (h : P a n) : LRet (Pure.pure a : LP PS α) n P :=
  fun st t l hl => ⟨a, st, t, l, rfl, hl ▸ h⟩

theorem LRet.bind {x : LP PS α} {f : α → LP PS β} {n : ℕ} {Q : α → ℕ → Prop} {P : β → ℕ → Prop}
    (hx : LRet x n Q) (hf : ∀ a m, Q a m → LRet (f a) m P) : LRet (x >>= f) n P := by
  intro st t l hl
  obtain ⟨a, st₁, t₁, l₁, h1, hq⟩ := hx st t l hl
  obtain ⟨b, st₂, t₂, l₂, h2, hp⟩ := hf a l₁.size hq st₁ t₁ l₁ rfl
  refine ⟨b, st₂, t₂, l₂, ?_, hp⟩
  rw [StateT.run_bind, StateT.run_bind, go_bind, h1]
  exact h2

theorem LRet.mono {y : LP PS α} {n : ℕ} {Q P : α → ℕ → Prop} (h : LRet y n Q) (hq : ∀ a m, Q a m → P a m) :
    LRet y n P := fun st t l hl =>
  let ⟨a, st', t', l', e, q⟩ := h st t l hl
  ⟨a, st', t', l', e, hq a _ q⟩

/-- A step that returns and sends nothing. -/
theorem LRet.silent {x : LP PS α}
    (hx : ∀ st t l, ∃ a st' t', ((x.run st).run t).go l = (.ok ((a, st'), t'), l)) (n : ℕ) :
    LRet x n fun _ m => m = n := fun st t l hl =>
  let ⟨a, st', t', e⟩ := hx st t l
  ⟨a, st', t', l, e, hl⟩

theorem LRet.liftTM {x : TM α} (hx : ∀ t, ∃ p, x t = .ok p) (n : ℕ) :
    LRet (liftT (liftTM x) : LP PS α) n fun _ m => m = n :=
  LRet.silent (fun st t l => let ⟨p, hp⟩ := hx t; ⟨p.1, st, p.2, liftT_go (liftTM_ok hp l) st⟩) n

theorem LRet.modify (g : LigState → LigState) (n : ℕ) : LRet (modify g : LP PS PUnit) n fun _ m => m = n :=
  LRet.silent (fun st t _ => ⟨(), g st, t, rfl⟩) n

theorem LRet.get (n : ℕ) : LRet (MonadState.get : LP PS LigState) n fun _ m => m = n :=
  LRet.silent (fun st t _ => ⟨st, st, t, rfl⟩) n

theorem LRet.sendMsg (msg : F256 × F256) (n : ℕ) : LRet (sendMsg msg : LP PS Unit) n fun _ m => m = n :=
  LRet.silent (fun st t _ => ⟨(), { st with msgs := st.msgs.push msg }, ap (pK msg) t, rfl⟩) n

theorem LRet.powB (nonce : ℕ → ℕ → UInt64) (n : ℕ) : LRet (powP nonce 2 : LP PS Unit) n fun _ m => m = n :=
  LRet.silent (fun st t _ => ⟨(), { st with cbn := st.cbn.push 0 },
    ap (pN 0) t, rfl⟩) n

theorem LRet.sampleKP (resp : ℕ → Claims.Msg) (n : ℕ) :
    LRet (liftT (sampleKP (asksS resp))) n fun a m => a = coinKOf (resp n) ∧ m = n + 1 := fun st t l hl =>
  ⟨_, st, _, _, liftT_go (sampleKP_go resp t l) st, by rw [hl]; exact ⟨rfl, by simp [hl]⟩⟩

theorem LRet.sampleP (resp : ℕ → Claims.Msg) (n : ℕ) :
    LRet (liftT (sampleP (asksS resp))) n fun _ m => m = n + 1 := fun st t l hl =>
  ⟨_, st, _, _, liftT_go (sampleP_go resp t l) st, by simp [hl]⟩

theorem LRet.sampleVecP (resp : ℕ → Claims.Msg) {k : ℕ} (hk : k ≠ 0) (n : ℕ) :
    LRet (liftT (sampleVecP (asksS resp) k)) n fun _ m => m = n + 1 := fun st t l hl =>
  ⟨_, st, _, _, liftT_go (sampleVecP_go resp hk t l) st, by simp [hl]⟩

theorem LRet.ite {c : Prop} [Decidable c] {y z : LP PS α} {n : ℕ} {P : α → ℕ → Prop} (hy : c → LRet y n P)
    (hz : ¬c → LRet z n P) : LRet (if c then y else z) n P := by
  by_cases h : c
  · rw [ite_eq_left h]; exact hy h
  · rw [ite_eq_right h]; exact hz h

/-- From the totality of Reps (`totL`): the step returns from any sends. -/
theorem LRet.ofTot {y : LP PS α} {P : α → Prop} (h : totL.T y P) (n : ℕ) : LRet y n fun a _ => P a :=
  fun st t l _ => let ⟨a, l', e, hp⟩ := h st t l; ⟨a.1.1, a.1.2, a.2, l', e, hp⟩

/-- **A loop over `[0:N]`** that keeps an invariant `I` of its state and its sends. -/
theorem LRet.forIn_upto {σ : Type} (N : ℕ) (b : σ) (f : ℕ → σ → LP PS (ForInStep σ)) (I : ℕ → σ → ℕ → Prop)
    {n : ℕ} (h0 : I 0 b n)
    (h : ∀ i < N, ∀ s m, I i s m → LRet (f i s) m fun r m' => ∃ s', r = .yield s' ∧ I (i + 1) s' m') :
    LRet (forIn [0:N] b f) n fun s m => I N s m := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (s : σ) m, a + k = N → I a s m → LRet (forIn (List.range' a k 1) s f) m fun s m => I N s m from
    key N 0 b n (by omega) h0
  intro k
  induction k with
  | zero =>
    intro a s m ha hI
    rw [List.range'_zero, List.forIn_nil]
    exact LRet.pure (by rw [← ha]; exact hI)
  | succ k ih =>
    intro a s m ha hI
    rw [List.range'_succ, List.forIn_cons]
    refine LRet.bind (h a (by omega) s m hI) fun r m' hr => ?_
    obtain ⟨s', rfl, hI'⟩ := hr
    exact ih (a + 1) s' m' (by omega) hI'

/-- A Ligerito program run from a state, then a step that returns, returns. -/
theorem LRet.run_bind {γ : Type} {y : LP PS α} {n : ℕ} {P : α → ℕ → Prop} (h : LRet y n P) (st : LigState)
    {g : α × LigState → TMW PS γ} (hg : ∀ p t l, ∃ c l', ((g p).run t).go l = (.ok c, l')) (t : Transcript)
    (l : Array Claims.Msg) (hl : l.size = n) : ∃ c l', ((y.run st >>= g).run t).go l = (.ok c, l') := by
  obtain ⟨a, st', t', l', e, -⟩ := h st t l hl
  obtain ⟨c, l'', e'⟩ := hg (a, st') t' l'
  exact ⟨c, l'', by rw [bind_go e]; exact e'⟩

theorem lsilentT {x : TM α} (hx : ∀ t, ∃ p, x t = .ok p) {n : ℕ} {P : β → ℕ → Prop} {f : α → LP PS β}
    (hf : ∀ a, LRet (f a) n P) : LRet (liftT (liftTM x) >>= f) n P :=
  LRet.bind (LRet.liftTM hx n) fun a _ hm => hm ▸ hf a

/-- **The masked Ligerito prover returns** from the bodies `l` where its pair's coins, the replies to sends
`|l| + 2 + k_0` and `|l| + 3 + k_0` (`ligeritoPZk_pos`), are independent over `F`. -/
theorem ligeritoPZk_ret (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (t : Transcript) (l : Array Claims.Msg)
    (hrho : Zk.rhoIndependent (coinKOf (resp (l.size + 2 + sch.k0)), coinKOf (resp (l.size + 3 + sch.k0))) = true) :
    ∃ a l', ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h : Hashes PS) ms sch T cm0 tb b lay hv rep tPad extra lanePad
      salt nonce).run t).go l = (.ok a, l') := by
  unfold ligeritoPZk
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
  dsimp only
  refine LRet.run_bind (n := l.size) (P := fun _ _ => True) ?_ _ (fun _ _ _ => ⟨_, _, rfl⟩) _ _ rfl
  refine LRet.bind (LRet.sampleVecP resp hn _) fun _ m hm => ?_
  subst hm
  refine LRet.bind (LRet.modify _ _) fun _ m hm => ?_
  subst hm
  refine lsilentT (fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine LRet.bind (LRet.powB nonce _) fun _ m hm => ?_
  subst hm
  refine LRet.bind (LRet.sampleP resp _) fun _ m hm => ?_
  subst hm
  refine LRet.bind (LRet.modify _ _) fun _ m hm => ?_
  subst hm
  refine LRet.bind (LRet.get _) fun _ m hm => ?_
  subst hm
  refine LRet.bind (LRet.sendMsg _ _) fun _ m hm => ?_
  subst hm
  refine LRet.bind (LRet.forIn_upto sch.k0 _ _ (fun j (_ : Array F256 × (F256 × F256) × Zk.Level0Msgs) m =>
      m = l.size + 2 + j + (if j = sch.k0 then 2 else 0)) (by rw [ite_eq_right (by omega)]) fun j hj s m hm => ?_)
    fun _ _ _ => LRet.ofTot ?_ _
  · rw [ite_eq_right (by omega), Nat.add_zero] at hm
    subst hm
    refine LRet.bind (LRet.sampleKP resp _) fun r m hr => ?_
    obtain ⟨rfl, rfl⟩ := hr
    refine LRet.bind (LRet.modify _ _) fun _ m hm => ?_
    subst hm
    refine LRet.ite (fun hj1 => ?_) fun hj1 => ?_
    · have hj1' : j + 1 = sch.k0 := beq_iff_eq.mp hj1
      refine LRet.bind (LRet.get _) fun _ m hm => ?_
      subst hm
      refine lsilentT (fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine lsilentT (fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine LRet.bind (LRet.sampleKP resp _) fun r0 m hr => ?_
      obtain ⟨rfl, rfl⟩ := hr
      refine LRet.bind (LRet.sampleKP resp _) fun r1 m hr => ?_
      obtain ⟨rfl, rfl⟩ := hr
      refine LRet.ite (fun hc => absurd hc ?_) fun _ => ?_
      · rw [show l.size + 2 + j + 1 = l.size + 2 + sch.k0 by omega,
          show l.size + 2 + sch.k0 + 1 = l.size + 3 + sch.k0 by omega, hrho]
        decide
      refine lsilentT (fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine lsilentT (fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine lsilentT (fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine LRet.bind (LRet.modify _ _) fun _ m hm => ?_
      subst hm
      refine LRet.bind (LRet.get _) fun _ m hm => ?_
      subst hm
      refine LRet.bind (LRet.sendMsg _ _) fun _ m hm => ?_
      subst hm
      refine LRet.pure ⟨_, rfl, ?_⟩
      rw [ite_eq_left (by omega)]
      omega
    · have hj1' : j + 1 ≠ sch.k0 := fun e => hj1 (beq_iff_eq.mpr e)
      refine LRet.bind (LRet.get _) fun _ m hm => ?_
      subst hm
      refine LRet.bind (LRet.sendMsg _ _) fun _ m hm => ?_
      subst hm
      refine LRet.pure ⟨_, rfl, ?_⟩
      rw [ite_eq_right hj1']
      omega
  · repeat (first
      | extract_lets
      | (with_reducible apply Total.bind')
      | (with_reducible intro _)
      | (with_reducible apply Total.pure')
      | (with_reducible apply Total.get)
      | (with_reducible apply Total.modify)
      | (with_reducible apply sendMsg_tot)
      | (with_reducible apply powP_tot)
      | (with_reducible apply challengeP_tot)
      | (with_reducible apply oodIntroP_tot)
      | (with_reducible apply queryP_tot)
      | (with_reducible apply liftT_tot)
      | (with_reducible apply Total.lift_st)
      | (with_reducible apply commitP_tot)
      | (with_reducible apply sampleP_tot)
      | (with_reducible apply sampleVecP_tot)
      | (with_reducible apply sampleKP_tot)
      | (with_reducible apply Total.forIn_range)
      | (with_reducible apply Total.forIn_array)
      | (with_reducible apply Total.ite)
      | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩)
      | split)

/-- `CoinsT`'s pair condition is `rhoIndependent` at the coins' replies. -/
theorem rhoIndependent_of {k : ℕ} (r : Fin k → FlockDeploy.Coin) {a : ℕ} (ha : a + 1 < k) (h : ¬ rhoDep r a) :
    Zk.rhoIndependent (coinKOf (Claims.coinSeq r a).bytes, coinKOf (Claims.coinSeq r (a + 1)).bytes) = true := by
  have hz : ¬ ((Claims.coinSeq r a).at 0 * (Claims.coinSeq r (a + 1)).at 1 +
      (Claims.coinSeq r a).at 1 * (Claims.coinSeq r (a + 1)).at 0).isZero = true :=
    fun hz => h ⟨ha, FlockLevel3.F128.isZero_iff _ hz⟩
  simpa [Zk.rhoIndependent, coinKOf, coinOf_bytes] using hz

/-- **The masked Ligerito prover returns on good coins**: answered by the coins `r` from the bodies `l`, where the pair
`ρ` at sends `|l| + 2 + k_0` and `|l| + 3 + k_0` is not dependent (`CoinsT`'s `¬ rhoDep`). -/
theorem ligeritoPZk_ret_coins {k : ℕ} (r : Fin k → FlockDeploy.Coin) (h : ByteArray → ByteArray) (ms : MerkleScheme)
    (sch : Schedule) (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (t : Transcript) (l : Array Claims.Msg) (hk : l.size + 3 + sch.k0 < k)
    (hr : ¬ rhoDep r (l.size + 2 + sch.k0)) :
    ∃ a l', ((ligeritoPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 h : Hashes PS) ms sch T cm0 tb
      b lay hv rep tPad extra lanePad salt nonce).run t).go l = (.ok a, l') := by
  refine ligeritoPZk_ret _ h ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce hn hk0 t l ?_
  rw [show l.size + 3 + sch.k0 = l.size + 2 + sch.k0 + 1 by omega]
  exact rhoIndependent_of r (by omega) hr

end FlockComplete.L0
