import Proofs.Flock.Claims.ZK.Split

/-!
# The session from the cut sends the session's bodies

`SimSession` from `SimCommit` and the hybrids `HybridStep 0, 1, 2` (`simSession_of`).

* **The opened coins as replies** (`respOf`, `cutS`): a run answered by `asksCut` at the opened coins that returns is
  the run answered by `asksS` at their bytes (`repPZk_cut`), so a masked rep of the session stopped after round `n`
  that returns sent its rounds (`repPZk_exact`) and its rounds carry its claim points (`repPZk_points`).
* **Two runs in step** (`Both`): the same bodies, both stopped or both returned with related values; through binds
  and the rep streams' loop (`forIn_both`).
* **The rep from the cut is the first hybrid** (`repCut_mix0`): at the erased commitments, `repCut` is `repMix 0` but
  what only the final message reads. **The last hybrid is the session's rep** (`repMix3_eq`): neither the replay nor
  level 0's constraints read level 0's opened rows and salts.
* **The session** (`simSession_of`): the schedules, the commitments from the cut's leaves (`SimCommit`), the link and
  each rep stream in turn, from its first round (`repBase`) and with rep `1`'s prior points rep `0`'s (`priorE`).
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

open FlockComplete Flock.With

/-! ## The opened coins as replies -/

/-- **A run that returns** from the bodies `l` returns the same on the right. -/
def cutS : Sim FlockComplete.PS FlockComplete.PS where
  R x y := ∀ l a l', x.go l = (.ok a, l') → y.go l = (.ok a, l')
  pure _ _ _ _ h := h
  bind {α β x y f g} hxy hfg l b l'' h := by
    obtain ⟨a, l', h1, h2⟩ := JsonRT.go_bind_ok h
    rw [go_bind, hxy l a l' h1]
    exact hfg a l' b l'' h2
  throw _ _ _ _ h := by cases h
  lift _ _ _ _ h := h

theorem cutS_refl {α : Type} (x : FlockComplete.PS α) : cutS.R x x := fun _ _ _ h => h

theorem cutS_reflT {α : Type} (x : _root_.Flock.TMW FlockComplete.PS α) : (cutS.st _root_.Flock.Transcript).R x x :=
  fun _ => cutS_refl _

theorem cutS_reflL {α : Type} (x : LP FlockComplete.PS α) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R x x :=
  fun _ => cutS_reflT _

theorem asksCut_link_go (coin : ℕ → ℕ → _root_.Flock.F128) (n : ℕ) (a b : Array ByteArray) (l : Array Msg) :
    ((asksCut coin n).link a b).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size),
        l.push (FlockDeploy.linkBody a.toList b.toList)) := by
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte]
    rfl
  · simp only [hn, ↓reduceIte]
    rfl

theorem asksCut_round_go (coin : ℕ → ℕ → _root_.Flock.F128) (n : ℕ) (c : ByteArray) (l : Array Msg) :
    ((asksCut coin n).round c).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size), l.push (Claims.toMsg c)) := by
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte]
    rfl
  · simp only [hn, ↓reduceIte]
    rfl

theorem cutS_link {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    (a b : Array ByteArray) : cutS.R ((asksCut coin n).link a b) ((asksS resp).link a b) := by
  intro l v l' e
  rw [asksCut_link_go] at e
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte, Prod.mk.injEq, reduceCtorEq, false_and] at e
  · simp only [hn, ↓reduceIte, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    rw [JsonRT.link_go, hr]

theorem cutS_round {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    (c : ByteArray) : cutS.R ((asksCut coin n).round c) ((asksS resp).round c) := by
  intro l v l' e
  rw [asksCut_round_go] at e
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte, Prod.mk.injEq, reduceCtorEq, false_and] at e
  · simp only [hn, ↓reduceIte, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    show (Except.ok (coinOf (resp l.size)), l.push (Claims.toMsg c)) = _
    rw [hr]

section Bridge

variable {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg}

theorem squeezeP_cut (hr : ∀ k, coinOf (resp k) = coin k) (kind : UInt8) (m : ℕ) :
    (cutS.st _root_.Flock.Transcript).R (squeezeP (asksCut coin n) kind m) (squeezeP (asksS resp) kind m) := by
  intro t
  unfold squeezeP
  sim_walk [cutS_round hr]

theorem sampleP_cut (hr : ∀ k, coinOf (resp k) = coin k) :
    (cutS.st _root_.Flock.Transcript).R (sampleP (asksCut coin n)) (sampleP (asksS resp)) := by
  unfold sampleP
  sim_walk [squeezeP_cut hr]

theorem sampleKP_cut (hr : ∀ k, coinOf (resp k) = coin k) :
    (cutS.st _root_.Flock.Transcript).R (sampleKP (asksCut coin n)) (sampleKP (asksS resp)) := by
  unfold sampleKP
  sim_walk [squeezeP_cut hr]

theorem liftT_cut {α : Type} {x y : _root_.Flock.TMW FlockComplete.PS α} (h : (cutS.st _root_.Flock.Transcript).R x y) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (liftT x) (liftT y) :=
  Sim.lift_st _ h

theorem zerocheckPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (m : ℕ) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (tr : Trace) :
    (cutS.st _root_.Flock.Transcript).R (zerocheckPZk (asksCut coin n) m digest cap root lay h tr)
      (zerocheckPZk (asksS resp) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_cut hr, sampleP_cut hr]

theorem lincheckPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (c : _root_.Flock.CircuitFold) (m : ℕ)
    (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (tr : Trace) (zc : ZcOut) :
    (cutS.st _root_.Flock.Transcript).R (lincheckPZk (asksCut coin n) c m lay h tr zc)
      (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_cut hr]

theorem ringSwitchP_cut (hr : ∀ k, coinOf (resp k) = coin k) (tr : Trace) (claims : Array _root_.Flock.Claim) :
    (cutS.st _root_.Flock.Transcript).R (ringSwitchP (asksCut coin n) tr claims) (ringSwitchP (asksS resp) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_cut hr]

theorem challengeP_cut (hr : ∀ k, coinOf (resp k) = coin k) (split : Bool) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (challengeP (asksCut coin n) split)
      (challengeP (asksS resp) split) := by
  unfold challengeP
  sim_walk [liftT_cut, sampleKP_cut hr, cutS_reflL]

theorem oodIntroP_cut (hr : ∀ k, coinOf (resp k) = coin k) (nonce : ℕ → ℕ → UInt64) (g : Array _root_.Flock.F128)
    (dim : ℕ) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (oodIntroP (asksCut coin n) nonce g dim)
      (oodIntroP (asksS resp) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_cut, squeezeP_cut hr, sampleP_cut hr, cutS_reflL]

theorem queryP_cut (hr : ∀ k, coinOf (resp k) = coin k) (nonce : ℕ → ℕ → UInt64) (ms : _root_.Flock.MerkleScheme)
    (lv : _root_.Flock.Level) (cm : Commit) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (queryP (asksCut coin n) nonce ms lv cm)
      (queryP (asksS resp) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_cut, squeezeP_cut hr, cutS_reflL]

theorem ligeritoPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : _root_.Flock.F128) (cm0 : Commit)
    (t b : Array _root_.Flock.F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (cutS.st _root_.Flock.Transcript).R
      (ligeritoPZk (asksCut coin n) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksS resp) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce) := by
  unfold ligeritoPZk sampleVecP
  sim_walk [liftT_cut, squeezeP_cut hr, sampleP_cut hr, sampleKP_cut hr, challengeP_cut hr, oodIntroP_cut hr,
    queryP_cut hr, cutS_reflL, cutS_reflT, cutS_refl]

theorem innerP_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (lay : _root_.Flock.Zk.PadLayout) (p : PadsZk) (ic : InnerCommit) (finalC : _root_.Flock.F128)
    (cons : Array (String × _root_.Flock.Zk.Lin)) :
    (cutS.st _root_.Flock.Transcript).R (innerP (asksCut coin n) hs lay p ic finalC cons)
      (innerP (asksS resp) hs lay p ic finalC cons) := by
  unfold innerP sampleVecP
  sim_walk [sampleP_cut hr, squeezeP_cut hr, cutS_reflT, cutS_refl]

/-- **A masked rep answered by `asksCut` that returns** is the rep answered by `asksS` at the same coins. -/
theorem repPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (t : Array _root_.Flock.F128)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R
      (repPZk (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words)
      (repPZk (asksS resp) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words) := by
  unfold repPZk rankCheckP
  sim_walk [zerocheckPZk_cut hr, lincheckPZk_cut hr, ringSwitchP_cut hr, ligeritoPZk_cut hr, innerP_cut hr,
    cutS_reflT, cutS_refl]

end Bridge

/-- **The opened coins' replies**: round `k`'s coin's bytes, the zero coin's past the pre-final rounds. -/
def respOf {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) : Msg :=
  (if h : k < refK x then c ⟨k, h⟩ else default : FlockDeploy.Coin).bytes

theorem coinOf_respOf {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) :
    coinOf (respOf c k) = fun i => slotAt c k i := by
  funext i
  unfold respOf slotAt
  rw [JsonRT.coinOf_bytes]
  split_ifs with hk
  · rfl
  · unfold FlockDeploy.Coin.at
    split_ifs <;> rfl

/-! ## Two runs in step -/

/-- **Two runs in step**: they end at the same bodies, and both stopped, or both returned with `P` of their values and
the bodies. -/
def Both {α β : Type} (P : α → β → Array Msg → Prop) (x : Except String α × Array Msg)
    (y : Except String β × Array Msg) : Prop :=
  x.2 = y.2 ∧ match x.1, y.1 with
    | .ok a, .ok b => P a b x.2
    | .error _, .error _ => True
    | _, _ => False

theorem Both.sends {α β : Type} {P : α → β → Array Msg → Prop} {x : Except String α × Array Msg}
    {y : Except String β × Array Msg} (h : Both P x y) : x.2 = y.2 := h.1

theorem both_bind {α β γ δ : Type} {P : α → β → Array Msg → Prop} {Q : γ → δ → Array Msg → Prop}
    {x : FlockComplete.PS α} {y : FlockComplete.PS β} {f : α → FlockComplete.PS γ} {g : β → FlockComplete.PS δ}
    {l : Array Msg} (h : Both P (x.go l) (y.go l)) (hfg : ∀ a b l', P a b l' → Both Q ((f a).go l') ((g b).go l')) :
    Both Q ((x >>= f).go l) ((y >>= g).go l) := by
  rw [go_bind, go_bind]
  obtain ⟨hl, hr⟩ := h
  rcases ex : x.go l with ⟨rx, lx⟩
  rcases ey : y.go l with ⟨ry, ly⟩
  rw [ex, ey] at hl hr
  dsimp only at hl hr ⊢
  subst hl
  cases rx <;> cases ry
  · exact ⟨rfl, trivial⟩
  · exact hr.elim
  · exact hr.elim
  · exact hfg _ _ _ hr

theorem both_sends_bind {α β γ δ : Type} {P : α → β → Array Msg → Prop} {x : FlockComplete.PS α}
    {y : FlockComplete.PS β} {f : α → FlockComplete.PS γ} {g : β → FlockComplete.PS δ} {l : Array Msg}
    (h : Both P (x.go l) (y.go l)) (hfg : ∀ a b l', P a b l' → ((f a).go l').2 = ((g b).go l').2) :
    ((x >>= f).go l).2 = ((y >>= g).go l).2 := by
  rw [go_bind, go_bind]
  obtain ⟨hl, hr⟩ := h
  rcases ex : x.go l with ⟨rx, lx⟩
  rcases ey : y.go l with ⟨ry, ly⟩
  rw [ex, ey] at hl hr
  dsimp only at hl hr ⊢
  subst hl
  cases rx <;> cases ry
  · rfl
  · exact hr.elim
  · exact hr.elim
  · exact hfg _ _ _ hr

/-- One computation run twice is in step with itself. -/
theorem both_rfl {α : Type} (x : FlockComplete.PS α) (l : Array Msg) :
    Both (fun a b l' => a = b ∧ x.go l = (.ok a, l')) (x.go l) (x.go l) := by
  refine ⟨rfl, ?_⟩
  rcases e : x.go l with ⟨r, l'⟩
  cases r
  · trivial
  · exact ⟨rfl, rfl⟩

/-- Two runs that end alike are in step, the right one's run recorded. -/
theorem both_of_eq {α : Type} {x y : Except String α × Array Msg} (h : x = y) :
    Both (fun a b l' => a = b ∧ y = (.ok b, l')) x y := by
  subst h
  refine ⟨rfl, ?_⟩
  rcases x with ⟨r, l'⟩
  cases r
  · trivial
  · exact ⟨rfl, rfl⟩

theorem both_mono {α β : Type} {P Q : α → β → Array Msg → Prop} {x : Except String α × Array Msg}
    {y : Except String β × Array Msg} (h : Both P x y) (hPQ : ∀ a b l', P a b l' → Q a b l') : Both Q x y := by
  refine ⟨h.1, ?_⟩
  obtain ⟨hl, hr⟩ := h
  rcases x with ⟨rx, lx⟩
  rcases y with ⟨ry, ly⟩
  dsimp only at hl hr ⊢
  cases rx <;> cases ry
  · trivial
  · exact hr.elim
  · exact hr.elim
  · exact hPQ _ _ _ hr

/-- **Two `for` loops in step over one list**, from position `k` of `s`, with an invariant of the position, the two
loops' states and the bodies; each step sees the element at its position. -/
theorem forIn_list_both {α β γ : Type} (f : α → β → FlockComplete.PS (ForInStep β))
    (g : α → γ → FlockComplete.PS (ForInStep γ)) (inv : ℕ → β → γ → Array Msg → Prop) (s : List α)
    (hs : ∀ k a b c l, s[k]? = some a → inv k b c l →
      Both (fun r r' l' => ∃ b' c', r = .yield b' ∧ r' = .yield c' ∧ inv (k + 1) b' c' l') ((f a b).go l)
        ((g a c).go l)) :
    ∀ (xs : List α) (k : ℕ), s.drop k = xs → ∀ b c l, inv k b c l →
      Both (fun b' c' l' => inv (k + xs.length) b' c' l') ((forIn xs b f).go l) ((forIn xs c g).go l)
  | [], k, _, b, c, l, h => by
    simp only [List.forIn_nil, JsonRT.go_pure, List.length_nil, Nat.add_zero]
    exact ⟨rfl, h⟩
  | a :: xs, k, hd, b, c, l, h => by
    rw [List.forIn_cons, List.forIn_cons]
    have ha : s[k]? = some a := by
      have := List.getElem?_drop (xs := s) (i := k) (j := 0)
      rw [hd, Nat.add_zero] at this
      rw [← this]
      rfl
    have hd' : s.drop (k + 1) = xs := by
      rw [← List.drop_drop, hd]
      rfl
    refine both_bind (hs k a b c l ha h) ?_
    rintro r r' l' ⟨b', c', rfl, rfl, h'⟩
    have := forIn_list_both f g inv s hs xs (k + 1) hd' b' c' l' h'
    rw [List.length_cons, show k + (xs.length + 1) = k + 1 + xs.length by omega]
    exact this

/-- **Two `for` loops in step over one array**, with an invariant of the steps taken. -/
theorem forIn_both {α β γ : Type} (xs : Array α) (f : α → β → FlockComplete.PS (ForInStep β))
    (g : α → γ → FlockComplete.PS (ForInStep γ)) (inv : ℕ → β → γ → Array Msg → Prop)
    (hs : ∀ k a b c l, xs[k]? = some a → inv k b c l →
      Both (fun r r' l' => ∃ b' c', r = .yield b' ∧ r' = .yield c' ∧ inv (k + 1) b' c' l') ((f a b).go l)
        ((g a c).go l))
    {b : β} {c : γ} {l : Array Msg} (h0 : inv 0 b c l) :
    Both (fun b' c' l' => inv xs.size b' c' l') ((forIn xs b f).go l) ((forIn xs c g).go l) := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  have := forIn_list_both f g inv xs.toList (fun k a b c l ha => hs k a b c l (by simpa using ha)) xs.toList 0
    (by simp) b c l h0
  simpa using this

/-! ## A rep's run up to its proof -/

/-- **A rep's run with its proof forgotten**: the verdict's transcript and the bodies. -/
def nf (z : Except String (_root_.Flock.Zk.ZkProof × _root_.Flock.Transcript) × Array Msg) :
    Except String _root_.Flock.Transcript × Array Msg :=
  (Except.map Prod.snd z.1, z.2)

theorem go_map {α β : Type} (f : α → β) (y : FlockComplete.PS α) (l : Array Msg) :
    (f <$> y).go l = (Except.map f (y.go l).1, (y.go l).2) := by
  rw [map_eq_pure_bind, go_bind]
  rcases y.go l with ⟨r, l'⟩
  cases r <;> rfl

theorem nf_map (f : _root_.Flock.Zk.ZkProof → _root_.Flock.Zk.ZkProof)
    (X : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof) (t : _root_.Flock.Transcript) (l : Array Msg) :
    nf (((f <$> X).run t).go l) = nf ((X.run t).go l) := by
  rw [StateT.run_map, go_map]
  unfold nf
  rcases (X.run t).go l with ⟨r, l'⟩
  cases r <;> rfl

/-- Runs alike but their proofs are in step, the right one's run recorded. -/
theorem both_of_nf {x y : Except String (_root_.Flock.Zk.ZkProof × _root_.Flock.Transcript) × Array Msg}
    (h : nf x = nf y) : Both (fun a b l' => a.2 = b.2 ∧ y = (.ok b, l')) x y := by
  rcases x with ⟨rx, lx⟩
  rcases y with ⟨ry, ly⟩
  unfold nf at h
  simp only [Prod.mk.injEq] at h
  obtain ⟨h1, rfl⟩ := h
  refine ⟨rfl, ?_⟩
  cases rx <;> cases ry
  · trivial
  · cases h1
  · cases h1
  · dsimp only
    exact ⟨by simpa [Except.map] using h1, rfl⟩

end Masked

end Claims.ZKMal
