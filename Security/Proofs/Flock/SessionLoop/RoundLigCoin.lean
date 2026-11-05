module

public import Proofs.Flock.Claims.ZK.CutPrefix
public import Proofs.Flock.Claims.ZK.SimLig

/-!
# Ligerito past level 0 reads its coins only below each round's count (`ligeritoPZk_coin`)

`ligeritoPZk` answered by `asksCut coin N` reads, at a round sent at log index `p < N` as a squeeze of `k` coins, only
`coin p i` for `i < k`. Two coin functions that agree there (`Good`) give the same run: the same result, the same
transcript and the same log.

* `CR Q x y`: the coin side `x` shifts with its history (`CutPrefix.Shift`), ends in `Q` where it returns, and equals the
  other side `y` from every history whose log after `x` is `Good` from its length on; `CRT` and `CRL` lift it through the
  transcript and the prover's state.
* Steps: binds, pure, throws, the coin-free steps (`crl_refl`), the squeezes (`crt_squeeze`, below `2 ^ 64` coins, the
  header's count), loops with an invariant (`crl_forIn`), and the opening's blocks.
* `ligeritoPZk_coin`: the opening's walk, its split dimension bounded by `sch.n + 1` through the levels.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.LigCoin

open Flock Transcript Replay

/-! ## Two coin functions agreeing on what a log's rounds read -/

/-- **`coin` and `coin'` agree on the rounds of `L` from index `P` to before `N`**: at a round framed as a squeeze of
`k < 2 ^ 64` coins, on the coins below `k`. -/
def Good (coin coin' : ℕ → ℕ → F128) (N P : ℕ) (L : Array Claims.Msg) : Prop :=
  ∀ p, P ≤ p → p < N → ∀ (t : Transcript) (kind : UInt8) (k : ℕ), k < 2 ^ 64 →
    L[p]? = some (Claims.toMsg (framed t kind k)) → ∀ i < k, coin p i = coin' p i

variable {coin coin' : ℕ → ℕ → F128} {N : ℕ}

theorem Good.mono {P P' : ℕ} {L : Array Claims.Msg} (h : Good coin coin' N P L) (hP : P ≤ P') :
    Good coin coin' N P' L :=
  fun p hp => h p (le_trans hP hp)

theorem Good.of_append {P : ℕ} {L bs : Array Claims.Msg} (h : Good coin coin' N P (L ++ bs)) :
    Good coin coin' N P L := by
  intro p hp hpN t kind k hk hL
  by_cases hs : p < L.size
  · exact h p hp hpN t kind k hk (by rw [Array.getElem?_append_left hs]; exact hL)
  · have hn : L[p]? = none := Array.getElem?_eq_none (by omega)
    rw [hn] at hL
    cases hL

/-! ## Runs equal under two coin functions -/

variable {α β σ : Type}

/-- **`x` with `coin`, `y` with `coin'`**: `x` shifts with its history and ends in `Q` where it returns, and the two
runs are equal from any history whose log after `x` is `Good` from its length on. -/
structure CR (coin coin' : ℕ → ℕ → F128) (N : ℕ) (Q : α → Prop) (x y : PS α) : Prop where
  shift : CutPrefix.Shift x x
  same : ∀ l, Good coin coin' N l.size (x.go l).2 → x.go l = y.go l
  post : ∀ l a L, x.go l = (.ok a, L) → Q a

theorem cr_refl {x : PS α} (h : CutPrefix.Shift x x) : CR coin coin' N (fun _ => True) x x :=
  ⟨h, fun _ _ => rfl, fun _ _ _ _ => trivial⟩

theorem cr_pure {Q : α → Prop} {a : α} (h : Q a) : CR coin coin' N Q (pure a) (pure a) where
  shift := CutPrefix.Shift.still (v := .ok a) fun _ => rfl
  same _ _ := rfl
  post l a' L e := by
    have e' : ((.ok a : Except String α), l) = (.ok a', L) := e
    cases e'
    exact h

theorem cr_bind {Q : α → Prop} {R : β → Prop} {x y : PS α} {f g : α → PS β} (hx : CR coin coin' N Q x y)
    (hf : ∀ a, Q a → CR coin coin' N R (f a) (g a)) : CR coin coin' N R (x >>= f) (y >>= g) where
  shift := by
    intro l l' hs
    obtain ⟨r, bs, h1, h2⟩ := hx.shift l l' hs
    rw [go_bind, go_bind, h1, h2]
    cases r with
    | error e => exact ⟨.error e, bs, rfl, rfl⟩
    | ok a =>
      obtain ⟨r', bs', h1', h2'⟩ := (hf a (hx.post l a _ h1)).shift (l ++ bs) (l' ++ bs) (by simp [hs])
      refine ⟨r', bs ++ bs', ?_, ?_⟩
      · dsimp only
        rw [h1', Array.append_assoc]
      · dsimp only
        rw [h2', Array.append_assoc]
  same l hG := by
    obtain ⟨r, bs, h1, -⟩ := hx.shift l l rfl
    rw [go_bind, h1] at hG
    have hG1 : Good coin coin' N l.size (x.go l).2 := by
      rw [h1]
      cases r with
      | error e => exact hG
      | ok a =>
        dsimp only at hG ⊢
        obtain ⟨r', bs', h1', -⟩ := (hf a (hx.post l a _ h1)).shift (l ++ bs) (l ++ bs) rfl
        rw [h1'] at hG
        exact Good.of_append hG
    have e1 := hx.same l hG1
    rw [go_bind, go_bind, ← e1, h1]
    cases r with
    | error e => rfl
    | ok a =>
      dsimp only at hG ⊢
      exact (hf a (hx.post l a _ h1)).same _ (hG.mono (by simp))
  post l b L h := by
    obtain ⟨a, l', h1, h2⟩ := JsonRT.go_bind_ok h
    exact (hf a (hx.post l a l' h1)).post l' b L h2

/-! ## Through the transcript -/

/-- `CR` from every transcript. -/
def CRT (coin coin' : ℕ → ℕ → F128) (N : ℕ) (Q : α → Prop) (x y : TMW PS α) : Prop :=
  ∀ t, CR coin coin' N (fun r => Q r.1) (x.run t) (y.run t)

theorem crt_bind {Q : α → Prop} {R : β → Prop} {x y : TMW PS α} {f g : α → TMW PS β}
    (hx : CRT coin coin' N Q x y) (hf : ∀ a, Q a → CRT coin coin' N R (f a) (g a)) :
    CRT coin coin' N R (x >>= f) (y >>= g) := fun t => by
  rw [StateT.run_bind, StateT.run_bind]
  exact cr_bind (hx t) fun p hp => hf p.1 hp p.2

theorem crt_pure {Q : α → Prop} {a : α} (h : Q a) : CRT coin coin' N Q (pure a) (pure a) := fun t => by
  rw [StateT.run_pure]
  exact cr_pure h

theorem crt_refl {x : TMW PS α} (h : (CutPrefix.shiftS.st Transcript).R x x) :
    CRT coin coin' N (fun _ => True) x x :=
  fun t => cr_refl (h t)

theorem crt_liftTM (x : Transcript.TM α) : CRT coin coin' N (fun _ => True) (liftTM x) (liftTM x) :=
  crt_refl (With.Sim.liftTM CutPrefix.shiftS x)

/-- **Two squeezes of `k < 2 ^ 64` coins**: equal where the log's round is `Good`. -/
theorem crt_squeeze (coin coin' : ℕ → ℕ → F128) (N : ℕ) (kind : UInt8) {k : ℕ} (hk : k < 2 ^ 64) :
    CRT coin coin' N (fun _ => True) (squeezeP (asksCut coin N) kind k) (squeezeP (asksCut coin' N) kind k) := by
  intro t
  refine ⟨CutPrefix.squeezeP_shift coin N t, fun l hG => ?_, fun _ _ _ _ => trivial⟩
  by_cases h0 : k = 0
  · subst h0
    rfl
  by_cases hn : N ≤ l.size
  · rw [LigLock.squeezeP_cut_err coin kind h0 t hn, LigLock.squeezeP_cut_err coin' kind h0 t hn]
  · have hl : l.size < N := by omega
    rw [LigLock.squeezeP_cut_ok coin kind h0 t hl] at hG ⊢
    rw [LigLock.squeezeP_cut_ok coin' kind h0 t hl]
    have hc : ∀ i < k, coin l.size i = coin' l.size i :=
      hG l.size le_rfl hl t kind k hk Array.getElem?_push_size
    have hm : (Array.range k).map (coin l.size) = (Array.range k).map (coin' l.size) := by
      apply Array.ext (by simp)
      intro i h1 _
      simp only [Array.getElem_map, Array.getElem_range]
      exact hc i (by simpa using h1)
    simp only [afterSq, sent, hm]

theorem crt_sampleP (coin coin' : ℕ → ℕ → F128) (N : ℕ) :
    CRT coin coin' N (fun _ => True) (sampleP (asksCut coin N)) (sampleP (asksCut coin' N)) := by
  unfold sampleP
  exact crt_bind (crt_squeeze coin coin' N 1 (by decide)) fun _ _ => crt_pure trivial

theorem crt_sampleVecP (coin coin' : ℕ → ℕ → F128) (N : ℕ) (k : ℕ) (hk : k < 2 ^ 64) :
    CRT coin coin' N (fun _ => True) (sampleVecP (asksCut coin N) k) (sampleVecP (asksCut coin' N) k) := by
  unfold sampleVecP
  exact crt_squeeze coin coin' N 2 hk

theorem crt_sampleKP (coin coin' : ℕ → ℕ → F128) (N : ℕ) :
    CRT coin coin' N (fun _ => True) (sampleKP (asksCut coin N)) (sampleKP (asksCut coin' N)) := by
  unfold sampleKP
  exact crt_bind (crt_squeeze coin coin' N 2 (by decide)) fun _ _ => crt_pure trivial

/-! ## Through the prover's state -/

/-- `CRT` from every prover state. -/
def CRL (coin coin' : ℕ → ℕ → F128) (N : ℕ) (Q : α → Prop) (x y : LP PS α) : Prop :=
  ∀ s, CRT coin coin' N (fun r => Q r.1) (x.run s) (y.run s)

theorem crl_bind {Q : α → Prop} {R : β → Prop} {x y : LP PS α} {f g : α → LP PS β}
    (hx : CRL coin coin' N Q x y) (hf : ∀ a, Q a → CRL coin coin' N R (f a) (g a)) :
    CRL coin coin' N R (x >>= f) (y >>= g) := fun s => by
  rw [StateT.run_bind, StateT.run_bind]
  exact crt_bind (hx s) fun p hp => hf p.1 hp p.2

theorem crl_pure {Q : α → Prop} {a : α} (h : Q a) : CRL coin coin' N Q (pure a) (pure a) := fun s => by
  rw [StateT.run_pure]
  exact crt_pure h

theorem crl_liftT {Q : α → Prop} {x y : TMW PS α} (h : CRT coin coin' N Q x y) :
    CRL coin coin' N Q (liftT x) (liftT y) := fun s => by
  unfold liftT
  rw [StateT.run_lift, StateT.run_lift]
  exact crt_bind h fun _ ha => crt_pure ha

theorem crl_refl {x : LP PS α} (h : ((CutPrefix.shiftS.st Transcript).st LigState).R x x) :
    CRL coin coin' N (fun _ => True) x x :=
  fun s t => cr_refl (h s t)

theorem crl_get : CRL coin coin' N (fun _ => True) (get : LP PS LigState) get :=
  crl_refl (With.Sim.get _)

theorem crl_set (s : LigState) : CRL coin coin' N (fun _ => True) (set s : LP PS PUnit) (set s) :=
  crl_refl (With.Sim.set _ s)

theorem crl_modify (F : LigState → LigState) : CRL coin coin' N (fun _ => True) (modify F : LP PS PUnit) (modify F) :=
  crl_refl (With.Sim.modify _ F)

theorem crl_lTM (x : Transcript.TM α) :
    CRL coin coin' N (fun _ => True) (liftT (liftTM x) : LP PS α) (liftT (liftTM x)) :=
  crl_refl (CutPrefix.liftT_shift (With.Sim.liftTM CutPrefix.shiftS x))

theorem crl_sendMsg (m : F256 × F256) : CRL coin coin' N (fun _ => True) (sendMsg m : LP PS Unit) (sendMsg m) :=
  crl_refl (CutPrefix.sendMsg_shift m)

theorem crl_powP (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    CRL coin coin' N (fun _ => True) (powP nonce kind : LP PS Unit) (powP nonce kind) :=
  crl_refl (CutPrefix.powP_shift nonce kind)

theorem crl_commit (H : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) :
    CRL coin coin' N (fun _ => True)
      (liftT (StateT.lift (commitP (Hashes.ofSha512 H : Hashes PS) ms lv g salt)) : LP PS Commit)
      (liftT (StateT.lift (commitP (Hashes.ofSha512 H : Hashes PS) ms lv g salt))) :=
  crl_refl (CutPrefix.liftT_shift (With.Sim.lift_st CutPrefix.shiftS (CutPrefix.commitP_shift H ms lv g salt)))

theorem crl_ite {Q : α → Prop} (c : Prop) [Decidable c] {x x' y y' : LP PS α} (h1 : c → CRL coin coin' N Q x y)
    (h2 : ¬ c → CRL coin coin' N Q x' y') :
    CRL coin coin' N Q (if c then x else x') (if c then y else y') := by
  by_cases hc : c
  · rw [ite_eq_left hc, ite_eq_left hc]
    exact h1 hc
  · rw [ite_eq_right hc, ite_eq_right hc]
    exact h2 hc

theorem crl_throw {Q : α → Prop} (e : String) : CRL coin coin' N Q (throw e : LP PS α) (throw e) := fun s t =>
  ⟨((CutPrefix.shiftS.st Transcript).st LigState).throw e s t, fun _ _ => rfl, fun l a L h => by
    have h' : (((throw e : LP PS α).run s).run t).go l = (.error e, l) := rfl
    rw [h'] at h
    cases h⟩

theorem crl_throw_bind {R : β → Prop} (e : String) {f g : α → LP PS β} :
    CRL coin coin' N R (throw e >>= f) (throw e >>= g) :=
  crl_bind (Q := fun _ => False) (crl_throw e) fun _ h => h.elim

/-- **A loop over `[0:n]`**, step `i` from a value in `J i` yielding one in `J (i + 1)`. -/
theorem crl_forIn (n : ℕ) (J : ℕ → σ → Prop) (body body' : ℕ → σ → LP PS (ForInStep σ))
    (hb : ∀ i < n, ∀ u, J i u →
      CRL coin coin' N (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u) (body' i u))
    (u : σ) (h0 : J 0 u) : CRL coin coin' N (J n) (forIn [:n] u body) (forIn [:n] u body') := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → J a u →
      CRL coin coin' N (J n) (forIn (List.range' a k) u body) (forIn (List.range' a k) u body') from
    key n 0 u (by omega) h0
  intro k
  induction k with
  | zero =>
    intro a u ha hJ
    have e : a = n := by omega
    exact crl_pure (e ▸ hJ)
  | succ k ih =>
    intro a u ha hJ
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine crl_bind (hb a (by omega) u hJ) fun r hr => ?_
    obtain ⟨v, rfl, hv⟩ := hr
    exact ih (a + 1) v (by omega) hv

theorem crl_forIn_bind {n : ℕ} (J : ℕ → σ → Prop) {R : β → Prop} {body body' : ℕ → σ → LP PS (ForInStep σ)}
    {u : σ} {f g : σ → LP PS β} (h0 : J 0 u)
    (hb : ∀ i < n, ∀ u, J i u →
      CRL coin coin' N (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u) (body' i u))
    (hf : ∀ r, J n r → CRL coin coin' N R (f r) (g r)) :
    CRL coin coin' N R (forIn [:n] u body >>= f) (forIn [:n] u body' >>= g) :=
  crl_bind (crl_forIn n J body body' hb u h0) hf

theorem crl_forIn_list {τ : Type} (l : List τ) (body body' : τ → σ → LP PS (ForInStep σ))
    (hb : ∀ x u, CRL coin coin' N (fun r => ∃ v, r = .yield v) (body x u) (body' x u)) (u : σ) :
    CRL coin coin' N (fun _ => True) (forIn l u body) (forIn l u body') := by
  induction l generalizing u with
  | nil => exact crl_pure trivial
  | cons x l ih =>
    rw [List.forIn_cons, List.forIn_cons]
    refine crl_bind (hb x u) fun r hr => ?_
    obtain ⟨v, rfl⟩ := hr
    exact ih v

theorem crl_forIn_array {τ : Type} (l : Array τ) (body body' : τ → σ → LP PS (ForInStep σ))
    (hb : ∀ x u, CRL coin coin' N (fun r => ∃ v, r = .yield v) (body x u) (body' x u)) (u : σ) :
    CRL coin coin' N (fun _ => True) (forIn l u body) (forIn l u body') := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact crl_forIn_list l.toList body body' hb u

/-! ## The opening's blocks -/

theorem crl_challengeP (coin coin' : ℕ → ℕ → F128) (N : ℕ) (split : Bool) :
    CRL coin coin' N (fun _ => True) (challengeP (asksCut coin N) split) (challengeP (asksCut coin' N) split) := by
  unfold challengeP
  apply crl_bind (crl_liftT (crt_sampleKP coin coin' N)); intro _ _
  apply crl_bind (crl_modify _); intro _ _
  dsimp only
  apply crl_ite
  · intro _
    apply crl_bind (crl_modify _); intro _ _
    apply crl_bind crl_get; intro _ _
    apply crl_bind (crl_sendMsg _); intro _ _
    exact crl_pure trivial
  · intro _
    apply crl_bind crl_get; intro _ _
    apply crl_bind (crl_sendMsg _); intro _ _
    exact crl_pure trivial

theorem crl_oodIntroP (coin coin' : ℕ → ℕ → F128) (N : ℕ) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ)
    (hd : dim < 2 ^ 64) :
    CRL coin coin' N (fun _ => True) (oodIntroP (asksCut coin N) nonce g dim)
      (oodIntroP (asksCut coin' N) nonce g dim) := by
  unfold oodIntroP
  exact crl_bind (crl_liftT (crt_sampleVecP coin coin' N dim hd)) fun _ _ => crl_bind (crl_modify _) fun _ _ =>
    crl_bind (crl_lTM _) fun _ _ => crl_bind (crl_sendMsg _) fun _ _ =>
    crl_bind (crl_powP nonce 2) fun _ _ => crl_bind (crl_liftT (crt_sampleP coin coin' N)) fun _ _ => crl_modify _

theorem crl_queryP (coin coin' : ℕ → ℕ → F128) (N : ℕ) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level)
    (cm : Commit) (hq : lv.q < 2 ^ 63) :
    CRL coin coin' N (fun _ => True) (queryP (asksCut coin N) nonce ms lv cm)
      (queryP (asksCut coin' N) nonce ms lv cm) := by
  unfold queryP
  have hl : Nat.log2 (lv.q - 1) + 1 < 2 ^ 64 := by
    have := Nat.log2_le_self (lv.q - 1)
    omega
  exact crl_bind (crl_powP nonce 0) fun _ _ =>
    crl_bind (crl_liftT (crt_sampleVecP coin coin' N lv.q (by omega))) fun _ _ =>
    crl_bind (crl_powP nonce 1) fun _ _ => crl_bind (crl_liftT (crt_sampleVecP coin coin' N _ hl)) fun _ _ =>
    crl_pure trivial

theorem crt_run_bind {Q : α → Prop} {R : β → Prop} {p p' : LP PS α} {st : LigState} {f f' : α × LigState → TMW PS β}
    (hp : CRL coin coin' N Q p p') (hf : ∀ a, Q a.1 → CRT coin coin' N R (f a) (f' a)) :
    CRT coin coin' N R (p.run st >>= f) (p'.run st >>= f') :=
  crt_bind (hp st) hf

/-! ## The opening -/

set_option maxHeartbeats 4000000 in
/-- **Ligerito's opening under two coin functions**: with the schedule's sizes below the headers' range, equal from
any history whose log after it is `Good`. -/
theorem ligeritoPZk_coin (coin coin' : ℕ → ℕ → F128) (N : ℕ) (H : ByteArray → ByteArray) (ms : MerkleScheme)
    (sch : Schedule) (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hn : sch.n < 2 ^ 62) (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 63) :
    CRT coin coin' N (fun _ => True)
      (ligeritoPZk (asksCut coin N) (Hashes.ofSha512 H) ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksCut coin' N) (Hashes.ofSha512 H) ms sch T cm0 t b lay h rep tPad extra lanePad salt
        nonce) := by
  unfold ligeritoPZk
  apply crt_bind (crt_liftTM _); intro _ _
  apply crt_bind (crt_liftTM _); intro _ _
  apply crt_bind (crt_liftTM _); intro _ _
  apply crt_bind (crt_liftTM _); intro _ _
  dsimp only
  apply crt_run_bind (Q := fun _ => True)
  · apply crl_bind (crl_liftT (crt_sampleVecP coin coin' N sch.n (by omega))); intro _ _
    apply crl_bind (crl_modify _); intro _ _
    apply crl_bind (crl_lTM _); intro _ _
    apply crl_bind (crl_powP nonce 2); intro _ _
    apply crl_bind (crl_liftT (crt_sampleP coin coin' N)); intro _ _
    apply crl_bind (crl_modify _); intro _ _
    apply crl_bind crl_get; intro _ _
    apply crl_bind (crl_sendMsg _); intro _ _
    apply crl_forIn_bind (fun _ _ => True) trivial
    · intro j _ u _
      apply crl_bind (crl_liftT (crt_sampleKP coin coin' N)); intro _ _
      apply crl_bind (crl_modify _); intro _ _
      apply crl_ite
      · intro _
        apply crl_bind crl_get; intro _ _
        apply crl_bind (crl_lTM _); intro _ _
        apply crl_bind (crl_lTM _); intro _ _
        apply crl_bind (crl_liftT (crt_sampleKP coin coin' N)); intro _ _
        apply crl_bind (crl_liftT (crt_sampleKP coin coin' N)); intro _ _
        apply crl_ite
        · intro _
          exact crl_throw_bind _
        · intro _
          apply crl_bind (crl_lTM _); intro _ _
          apply crl_bind (crl_lTM _); intro _ _
          apply crl_bind (crl_lTM _); intro _ _
          apply crl_bind (crl_modify _); intro _ _
          apply crl_bind crl_get; intro _ _
          apply crl_bind (crl_sendMsg _); intro _ _
          exact crl_pure ⟨_, rfl, trivial⟩
      · intro _
        apply crl_bind crl_get; intro _ _
        apply crl_bind (crl_sendMsg _); intro _ _
        exact crl_pure ⟨_, rfl, trivial⟩
    · intro _ _
      apply crl_bind crl_get; intro _ _
      apply crl_bind (crl_commit H ms _ _ _); intro _ _
      apply crl_bind (crl_lTM _); intro _ _
      apply crl_bind (crl_oodIntroP coin coin' N nonce _ _ (by omega)); intro _ _
      apply crl_bind (crl_oodIntroP coin coin' N nonce _ _ (by omega)); intro _ _
      apply crl_bind (crl_queryP coin coin' N nonce ms _ cm0 (hq 0)); intro _ _
      apply crl_bind (crl_sendMsg _); intro _ _
      apply crl_bind (crl_powP nonce 2); intro _ _
      apply crl_bind (crl_liftT (crt_sampleP coin coin' N)); intro _ _
      apply crl_bind (crl_modify _); intro _ _
      apply crl_forIn_bind (fun _ u => u.1 ≤ sch.n + 1) (by dsimp only; omega)
      · intro i _ u hJ
        obtain ⟨sd, pc, cs, rps, fn⟩ := u
        dsimp only at hJ
        apply crl_forIn_bind (fun _ _ => True) trivial
        · intro j _ v _
          apply crl_ite
          · intro _
            apply crl_bind (crl_liftT (crt_sampleKP coin coin' N)); intro _ _
            apply crl_bind (crl_modify _); intro _ _
            apply crl_bind crl_get; intro _ _
            apply crl_bind (crl_modify _); intro _ _
            apply crl_bind crl_get; intro _ _
            apply crl_bind (crl_sendMsg _); intro _ _
            exact crl_pure ⟨_, rfl, trivial⟩
          · intro _
            apply crl_bind (crl_challengeP coin coin' N false); intro _ _
            exact crl_pure ⟨_, rfl, trivial⟩
        · intro _ _
          apply crl_ite
          · intro _
            apply crl_bind (crl_commit H ms _ _ _); intro _ _
            apply crl_bind (crl_lTM _); intro _ _
            apply crl_bind (crl_oodIntroP coin coin' N nonce _ (sd - 4 + 1) (by omega)); intro _ _
            apply crl_bind (crl_oodIntroP coin coin' N nonce _ (sd - 4 + 1) (by omega)); intro _ _
            apply crl_bind (crl_queryP coin coin' N nonce ms _ _ (hq (i + 1))); intro _ _
            apply crl_bind (crl_sendMsg _); intro _ _
            apply crl_bind (crl_powP nonce 2); intro _ _
            apply crl_bind (crl_liftT (crt_sampleP coin coin' N)); intro _ _
            apply crl_bind (crl_modify _); intro _ _
            exact crl_pure ⟨_, rfl, by dsimp only; omega⟩
          · intro _
            refine crl_bind (crl_forIn_array _ _ _ (fun _ _ => crl_bind (crl_lTM _) fun _ _ =>
              crl_pure ⟨_, rfl⟩) _) fun _ _ => ?_
            apply crl_bind (crl_queryP coin coin' N nonce ms _ _ (hq _)); intro _ _
            apply crl_bind (crl_powP nonce 2); intro _ _
            apply crl_bind (crl_liftT (crt_sampleP coin coin' N)); intro _ _
            exact crl_pure ⟨_, rfl, hJ⟩
      · intro _ _
        exact crl_pure trivial
  · intro _ _
    exact crt_pure trivial

/-- **The opening's run under two coin functions** is the same from any history whose log after it is `Good` from its
length on. -/
theorem ligeritoPZk_go_coin (coin coin' : ℕ → ℕ → F128) (N : ℕ) (H : ByteArray → ByteArray) (ms : MerkleScheme)
    (sch : Schedule) (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hn : sch.n < 2 ^ 62) (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 63) (tr : Transcript)
    (l : Array Claims.Msg)
    (hG : Good coin coin' N l.size
      (((ligeritoPZk (asksCut coin N) (Hashes.ofSha512 H) ms sch T cm0 t b lay h rep tPad extra lanePad salt
        nonce).run tr).go l).2) :
    ((ligeritoPZk (asksCut coin N) (Hashes.ofSha512 H) ms sch T cm0 t b lay h rep tPad extra lanePad salt
        nonce).run tr).go l =
      ((ligeritoPZk (asksCut coin' N) (Hashes.ofSha512 H) ms sch T cm0 t b lay h rep tPad extra lanePad salt
        nonce).run tr).go l :=
  (ligeritoPZk_coin coin coin' N H ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce hn hq tr).same l hG

end FlockComplete.LigCoin
