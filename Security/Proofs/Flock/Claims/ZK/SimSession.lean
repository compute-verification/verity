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
  have e : ((asksCut coin n).link a b).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : FlockComplete.PS _)).go
        (l.push (FlockDeploy.linkBody a.toList b.toList)) := rfl
  rw [e]
  split_ifs <;> rfl

theorem asksCut_round_go (coin : ℕ → ℕ → _root_.Flock.F128) (n : ℕ) (c : ByteArray) (l : Array Msg) :
    ((asksCut coin n).round c).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size), l.push (Claims.toMsg c)) := by
  have e : ((asksCut coin n).round c).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : FlockComplete.PS _)).go
        (l.push (Claims.toMsg c)) := rfl
  rw [e]
  split_ifs <;> rfl

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
    (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace) :
    (cutS.st _root_.Flock.Transcript).R (zerocheckPZk (asksCut coin n) m digest cap root lay h tr)
      (zerocheckPZk (asksS resp) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_cut hr, sampleP_cut hr]

theorem lincheckPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (c : _root_.Flock.CircuitFold) (m : ℕ)
    (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace) (zc : _root_.Flock.ZcOut) :
    (cutS.st _root_.Flock.Transcript).R (lincheckPZk (asksCut coin n) c m lay h tr zc)
      (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_cut hr]

theorem ringSwitchP_cut (hr : ∀ k, coinOf (resp k) = coin k) (tr : FlockComplete.Trace) (claims : Array _root_.Flock.Claim) :
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
theorem repPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace) (t : Array _root_.Flock.F128)
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

theorem go_map {α β : Type} (f : α → β) (y : FlockComplete.PS α) (l : Array Msg) :
    (f <$> y).go l = (Except.map f (y.go l).1, (y.go l).2) := by
  rw [map_eq_pure_bind, go_bind]
  rcases y.go l with ⟨r, l'⟩
  cases r <;> rfl

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

/-- Two runs in step, the right one's run recorded. -/
theorem both_right {α β : Type} {P : α → β → Array Msg → Prop} {x : Except String α × Array Msg}
    {y : Except String β × Array Msg} (h : Both P x y) : Both (fun a b l' => P a b l' ∧ y = (.ok b, l')) x y := by
  obtain ⟨hl, hr⟩ := h
  rcases x with ⟨rx, lx⟩
  rcases y with ⟨ry, ly⟩
  dsimp only at hl hr ⊢
  refine ⟨hl, ?_⟩
  cases rx <;> cases ry
  · trivial
  · exact hr.elim
  · exact hr.elim
  · exact ⟨hr, by rw [hl]⟩

/-- A run that ends as another's mapped by `e` is in step with it. -/
theorem both_of_map_eq {α β : Type} {x : FlockComplete.PS α} {y : FlockComplete.PS β} {e : β → α} {l : Array Msg}
    (h : x.go l = (e <$> y).go l) : Both (fun a b _ => a = e b) (x.go l) (y.go l) := by
  rw [h, go_map]
  rcases y.go l with ⟨r, l'⟩
  cases r
  · exact ⟨rfl, trivial⟩
  · exact ⟨rfl, rfl⟩

theorem both_map {α β γ δ : Type} {P : α → β → Array Msg → Prop} {Q : γ → δ → Array Msg → Prop}
    {x : FlockComplete.PS α} {y : FlockComplete.PS β} (F : α → γ) (G : β → δ) {l : Array Msg}
    (h : Both P (x.go l) (y.go l)) (hPQ : ∀ a b l', P a b l' → Q (F a) (G b) l') :
    Both Q ((F <$> x).go l) ((G <$> y).go l) := by
  rw [go_map, go_map]
  rcases ex : x.go l with ⟨rx, lx⟩
  rcases ey : y.go l with ⟨ry, ly⟩
  rw [ex, ey] at h
  obtain ⟨hl, hr⟩ := h
  dsimp only at hl hr ⊢
  refine ⟨hl, ?_⟩
  cases rx <;> cases ry
  · trivial
  · exact hr.elim
  · exact hr.elim
  · exact hPQ _ _ _ hr

/-- **Two `mapM`s in step**, each element's run on the left the right one's mapped by `e`. -/
theorem both_mapM_list {α β γ : Type} (f : α → FlockComplete.PS β) (g : α → FlockComplete.PS γ) (e : γ → β) :
    ∀ (xs : List α), (∀ a ∈ xs, ∀ l, (f a).go l = (e <$> g a).go l) → ∀ l : Array Msg,
      Both (fun bs cs _ => bs = cs.map e) ((xs.mapM f).go l) ((xs.mapM g).go l)
  | [], _, l => by
    simp only [List.mapM_nil, JsonRT.go_pure]
    exact ⟨rfl, rfl⟩
  | a :: xs, h, l => by
    rw [List.mapM_cons, List.mapM_cons]
    refine both_bind (both_of_map_eq (h a List.mem_cons_self l)) fun b c l' hb => ?_
    refine both_bind (both_mapM_list f g e xs (fun a' ha' => h a' (List.mem_cons_of_mem _ ha')) l')
      fun bs cs l'' hbs => ?_
    simp only [JsonRT.go_pure]
    refine ⟨rfl, ?_⟩
    show b :: bs = (c :: cs).map e
    rw [hb, hbs, List.map_cons]

theorem both_mapM_array {α β γ : Type} (f : α → FlockComplete.PS β) (g : α → FlockComplete.PS γ) (e : γ → β)
    (xs : Array α) (h : ∀ a ∈ xs, ∀ l, (f a).go l = (e <$> g a).go l) (l : Array Msg) :
    Both (fun bs cs _ => bs = cs.map e) ((xs.mapM f).go l) ((xs.mapM g).go l) := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
  refine both_map _ _ (both_mapM_list f g e xs.toList (fun a ha => h a (Array.mem_toList_iff.1 ha)) l) ?_
  rintro bs cs _ rfl
  simp

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

theorem nf_map (f : _root_.Flock.Zk.ZkProof → _root_.Flock.Zk.ZkProof)
    (prog : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof) (t : _root_.Flock.Transcript) (l : Array Msg) :
    nf (((f <$> prog).run t).go l) = nf ((prog.run t).go l) := by
  rw [StateT.run_map, go_map]
  unfold nf
  rcases (prog.run t).go l with ⟨r, l'⟩
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

/-! ## The rep from the cut is the first hybrid -/

theorem eraseC_cap (cm : Commit) : (eraseC cm).cap = cm.cap := rfl

theorem eraseIC_root (ic : InnerCommit) : (eraseIC ic).root = ic.root := rfl

/-- **The inner proof from the source reads the pads commitment's columns and salts only where the final message
does.** -/
theorem innerCut_eraseIC (io : Asks FlockComplete.PS) (lay : _root_.Flock.Zk.PadLayout) (ic : InnerCommit)
    (finalC : _root_.Flock.F128) (src : RepSrc) (cons : Array (String × _root_.Flock.Zk.Lin)) :
    eraseIP <$> innerCut io lay (eraseIC ic) finalC src cons = eraseIP <$> innerCut io lay ic finalC src cons := by
  unfold innerCut
  simp only [map_bind, map_pure, apply_ite (Functor.map (f := _root_.Flock.TMW FlockComplete.PS) eraseIP),
    eraseIP, eraseIC, Array.size_map, Array.map_map, Function.comp_def]

/-- A rep's proof forgotten reads its inner proof forgotten. -/
theorem bind_eraseZ {xa xb : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.InnerProof}
    (h : eraseIP <$> xa = eraseIP <$> xb) (p : _root_.Flock.Proof) (zl : _root_.Flock.Zk.Level0Msgs) :
    (xa >>= fun inner => pure (eraseZ { proof := p, level0 := zl, inner })) =
      (xb >>= fun inner => pure (eraseZ { proof := p, level0 := zl, inner })) := by
  have e : ∀ xc : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.InnerProof,
      (xc >>= fun inner => pure (eraseZ { proof := p, level0 := zl, inner })) =
        (fun ip => ({ proof := { p with ligerito := eraseL p.ligerito }, level0 := zl, inner := ip } :
          _root_.Flock.Zk.ZkProof)) <$> (eraseIP <$> xc) := fun xc => by
    rw [← comp_map, ← bind_pure_comp]
    rfl
  rw [e, e, h]

/-- **The rep from the cut at the erased commitments is the first hybrid** but what only the final message reads. -/
theorem repCut_mix0 (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace) (t : Array _root_.Flock.F128)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) (src : RepSrc) :
    repMix 0 io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words src =
      eraseZ <$> repCut io hs st sch rep dom (eraseC cm0) extras (eraseIC ic) src prior words := by
  unfold repMix repCut
  simp only [↓reduceIte, Nat.zero_le, eraseC_cap, eraseIC_root]
  rw [map_bind]; refine bind_congr fun ⟨zp, lp, ss, T, sws⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun ⟨lg, zl, rec⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun tr' => ?_
  rw [map_bind]; refine bind_congr fun ⟨rp, _⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun lgc => ?_
  rw [map_bind]
  simp only [map_pure]
  exact bind_eraseZ (innerCut_eraseIC _ _ _ _ _ _).symm _ _

/-! ## The last hybrid is the session's rep -/

/-- The replay reads the proof's cap, zerocheck and lincheck only. -/
theorem replay_eraseL (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap : Array ByteArray)
    (params : _root_.Flock.PcsParams) (zp : _root_.Flock.ZerocheckProof) (lp : _root_.Flock.LincheckProof)
    (ss : Array (Array _root_.Flock.F128)) (lg : _root_.Flock.LigeritoProof) (root : ByteArray) :
    _root_.Flock.Zk.replay st lay
        { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := eraseL lg } root =
      _root_.Flock.Zk.replay st lay
        { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := lg } root := by
  unfold _root_.Flock.Zk.replay
  dsimp only

/-- Level 0's constraints read the opening's messages and out-of-domain values only. -/
theorem ligeritoConstraints_eraseL (lay : _root_.Flock.Zk.PadLayout) (rec : _root_.Flock.Zk.L0Rec)
    (lg : _root_.Flock.LigeritoProof) (zl : _root_.Flock.Zk.Level0Msgs) :
    _root_.Flock.Zk.ligeritoConstraints lay rec (eraseL lg) zl = _root_.Flock.Zk.ligeritoConstraints lay rec lg zl := by
  unfold _root_.Flock.Zk.ligeritoConstraints eraseL
  dsimp only

theorem eraseL_eraseL (lg : _root_.Flock.LigeritoProof) : eraseL (eraseL lg) = eraseL lg := by
  simp only [eraseL, eraseLP, Array.map_map, Function.comp_def]

/-- **The last hybrid is the session's rep** but what only the final message reads. -/
theorem repMix3_eq (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace) (t : Array _root_.Flock.F128)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) (src : RepSrc) :
    repMix 3 io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words src =
      eraseZ <$> repPZk io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words := by
  unfold repMix repPZk piopPZk
  simp only [show ¬((3 : ℕ) = 0) by decide, show ¬((3 : ℕ) ≤ 1) by decide, show ¬((3 : ℕ) ≤ 2) by decide,
    ↓reduceIte]
  rw [bind_assoc, map_bind]; refine bind_congr fun ⟨zp, zc⟩ => ?_; dsimp only
  rw [bind_assoc, map_bind]; refine bind_congr fun ⟨lp, lc⟩ => ?_; dsimp only
  rw [bind_assoc, map_bind]; refine bind_congr fun _ => ?_
  rw [bind_assoc, map_bind]; refine bind_congr fun ⟨ss, T, sws⟩ => ?_; dsimp only
  rw [pure_bind]; dsimp only
  rw [map_bind]; refine bind_congr fun ⟨lg, zl, rec⟩ => ?_; dsimp only
  rw [pure_bind]; dsimp only
  simp only [replay_eraseL, ligeritoConstraints_eraseL]
  rw [map_bind]; refine bind_congr fun tr' => ?_
  rw [map_bind]; refine bind_congr fun ⟨rp, _⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun lgc => ?_
  rw [map_bind]; refine bind_congr fun inner => ?_
  rw [map_pure]
  simp only [eraseZ, eraseL_eraseL]

/-! ## A rep from the cut against the session's -/

/-- **Table `j`'s rep `r` from the cut sends what the session's rep sends**, to the same transcript: from its first
round, at the erased commitments, through the hybrids. -/
theorem repCut_nf (h0 : HybridStep 0) (h1 : HybridStep 1) (h2 : HybridStep 2) {x : Msg} (hx : Live x)
    {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ)
    (hn : ¬ Refused x c n) {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray)
    (cm0 : Commit) (ic : InnerCommit) (extras : Array _root_.Flock.Claim)
    (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras) (l : Array Msg)
    (hl : l.size = repBase x (2 * j.val + r.val)) :
    nf (((repCut (ioE c n) hsE (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val dom (eraseC cm0) extras
        (eraseIC ic) ((srcE x c w τ j.val).rep r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run (_root_.Flock.Transcript.init #[] dom)).go l) =
      nf (((repPZk (ioE c n) hsE (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch (zkTab x j).tPad r.val dom
        (trsE x w (UE.tapeZk τ))[j.val]!
        (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m) cm0 extras
        (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)) ic
        ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
        ((UE.tapeZk τ).nonce j.val r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run (_root_.Flock.Transcript.init #[] dom)).go l) := by
  rw [← nf_map eraseZ, ← repCut_mix0 (tPad := (zkTab x j).tPad) (tr := (trsE x w (UE.tapeZk τ))[j.val]!)
    (t := FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m)
    (pads := FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0))
    (extra := (UE.tapeZk τ).extra j.val) (lanePad := (UE.tapeZk τ).lanePad j.val)
    (salt := (UE.tapeZk τ).salt j.val r.val) (nonce := (UE.tapeZk τ).nonce j.val r.val)]
  change nf (((mixE 0 x w τ c n j r dom cm0 extras ic).run _).go l) = _
  rw [h0 x hx w hw τ c n hn s spec hsx j r dom cm0 ic extras he l hl,
    h1 x hx w hw τ c n hn s spec hsx j r dom cm0 ic extras he l hl,
    h2 x hx w hw τ c n hn s spec hsx j r dom cm0 ic extras he l hl]
  unfold mixE
  rw [repMix3_eq, nf_map]

/-! ## The session -/

/-- **The record sends nothing.** -/
theorem recordJson_still (spec : _root_.Flock.SessionSpec) (rootB : Array ByteArray)
    (points : Array (Array _root_.Flock.F128)) (streams : Array (Array _root_.Flock.Round))
    (proofs : Array ByteArray) : JsonRT.Still (recordJson hsE spec rootB points streams proofs) := by
  have hq : quietS.R (recordJson (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) spec rootB points streams
      proofs : _root_.Flock.V _) (recordJson hsE spec rootB points streams proofs) := by
    have hh := quietS_hashes _root_.Flock.Sha512.hash
    unfold recordJson _root_.Flock.SessionSpec.dgWith
    sim_walk [roundJ_quiet _, Sim.mapM_list]
  intro l
  rw [hq l]

theorem getElem!_range_map {α : Type} [Inhabited α] (f : ℕ → α) {N i : ℕ} (hi : i < N) :
    ((Array.range N).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using hi)]
  simp

open JsonRT in
theorem simSession_of (hc : SimCommit) (h0 : HybridStep 0) (h1 : HybridStep 1) (h2 : HybridStep 2) :
    SimSession := by
  intro x hx w hw τ c n hn s spec hsx
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    simp only [hx, ↓reduceIte]
  have htabs : (FlockDeploy.tabsOf (stsOf x)).size = (stsOf x).size := FlockComplete.ZkTables.tabsOf_size _
  unfold runCut runExe sessionCut sessionPZk
  refine both_sends_bind (both_rfl _ _) ?_
  rintro schs _ l1 ⟨rfl, hA⟩
  obtain ⟨hssz, hsf⟩ := Ret.go (Ret.mapM_array _ (fun st' v => _root_.Flock.Zk.schedule st'.m = .ok v) _
    fun a _ => Ret.monadLift (_root_.Flock.Zk.schedule a.m)) hA
  have hl1 := Still.go (Still.mapM_array (fun a l => by rw [monadLift_go]) _) hA
  subst hl1
  have hsch : ∀ i (hi : i < jOf x), schs[i]! = ((zkTab x ⟨i, hi⟩).sch, (zkTab x ⟨i, hi⟩).tPad) := fun i hi => by
    have h1 := hsf i (by omega) (by omega)
    have h2 := (zkTab x ⟨i, hi⟩).hz
    have e : (FlockDeploy.tabsOf (stsOf x))[i].m = (stAt x ⟨i, hi⟩).m := by
      unfold stAt
      simp [FlockDeploy.tabsOf]
      rfl
    rw [e, h2] at h1
    rw [getElem!_pos schs i (by omega)]
    exact (Except.ok.inj h1).symm
  dsimp only
  refine both_sends_bind (both_right (both_mapM_array _ _ eraseC _ ?_ _)) ?_
  · intro i hi l
    rw [Array.mem_range] at hi
    have hij : i < jOf x := by omega
    have := (hc x hx w hw τ c ⟨i, hij⟩ l).1
    rw [hsch i hij, getElem!_range_map _ hi]
    exact this
  rintro _ cms l2 ⟨rfl, hB⟩
  have hl2 := Still.go (Still.mapM_array (fun i => Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨hcsz, -⟩ := Ret.go (Ret.mapM_array _ (fun _ _ => True) _ fun _ _ _ _ _ _ => trivial) hB
  have hroot : (cms.map eraseC).map (fun c => flat c.cap) = cms.map fun c => flat c.cap := by
    rw [Array.map_map]
    rfl
  simp only [hroot]
  refine both_sends_bind (both_rfl _ _) ?_
  rintro c1 _ l3 ⟨rfl, hL⟩
  rw [asksCut_link_go] at hL
  have hn0 : ¬ n ≤ (#[] : Array Msg).size := by
    intro h
    simp only [h, ↓reduceIte] at hL
    cases (Prod.mk.inj hL).1
  simp only [hn0, ↓reduceIte] at hL
  obtain ⟨hc1, rfl⟩ := Prod.mk.inj hL
  cases hc1
  refine both_sends_bind (forIn_both _ _ _ (fun k bC bP l => bC.1 = bP.1 ∧ l.size = repBase x k ∧
      (k % 2 = 1 → MaskRank.roundPoints (stsOf x)[k / 2]!.m (stsOf x)[k / 2]!.c.kLog bP.1.back! =
        ptsOpen c (stsOf x)[k / 2]! (k - 1))) ?_ ?_) ?_
  · rintro k a ⟨sC, pC⟩ ⟨sP, pP⟩ l ha ⟨hs, hl, hback⟩
    dsimp only at hs
    subst hs
    obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
    have hk' : k < 2 * (stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec] at hk; exact hk
    have hi : k / 2 < jOf x := by omega
    have hr2 : k % 2 < 2 := Nat.mod_lt _ (by omega)
    have hi' : k / 2 < (stsOf x).size := by omega
    have hst : (stsOf x)[k / 2]! = stAt x ⟨k / 2, hi⟩ := by
      unfold stAt
      exact getElem!_pos _ _ _
    have hv : _root_.Flock.Zk.schedule (stsOf x)[k / 2]!.m =
        .ok ((zkTab x ⟨k / 2, hi⟩).sch, (zkTab x ⟨k / 2, hi⟩).tPad) := by
      rw [hst]
      exact (zkTab x ⟨k / 2, hi⟩).hz
    have hk6 : 6 ≤ (stsOf x)[k / 2]!.c.kLog := by
      rw [hst]
      exact (zkTab x ⟨k / 2, hi⟩).wf.1.hk6
    have hm : 14 ≤ (stsOf x)[k / 2]!.m := by
      have := (hall (k / 2) hi').1
      rw [getElem!_pos (stsOf x) (k / 2) hi']
      omega
    rw [FlockDeploy.repStreams_getElem _ hk]
    dsimp only
    rw [hsch (k / 2) hi]
    dsimp only
    refine both_bind (both_rfl _ _) ?_
    rintro extras _ l1 ⟨rfl, e1⟩
    rw [monadLift_go] at e1
    obtain ⟨hex, rfl⟩ := Prod.mk.inj e1
    refine both_bind (both_right (both_of_map_eq ((hc x hx w hw τ c ⟨k / 2, hi⟩ l).2 ⟨k % 2, hr2⟩))) ?_
    rintro _ ic l2 ⟨rfl, e2⟩
    have hl2 := (Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e2).symm
    subst hl2
    have hcms : (cms.map eraseC)[k / 2]! = eraseC cms[k / 2]! := by
      rw [getElem!_pos _ (k / 2) (by simp only [Array.size_map, hcsz, Array.size_range]; omega),
        getElem!_pos cms (k / 2) (by simp only [hcsz, Array.size_range]; omega), Array.getElem_map]
    have htb : ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).map fun i =>
        packed (trsE x w (UE.tapeZk τ))[i]! (FlockDeploy.tabsOf (stsOf x))[i]!.m)[k / 2]! =
          packed (trsE x w (UE.tapeZk τ))[k / 2]! (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m :=
      getElem!_range_map _ (by omega)
    have htab : (FlockDeploy.tabsOf (stsOf x))[k / 2]! = _root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]! := by
      rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) (k / 2) (by omega), getElem!_pos (stsOf x) (k / 2) hi']
      simp [FlockDeploy.tabsOf]
    have hpr : priorOf (FlockDeploy.tabsOf (stsOf x))[k / 2]! sC (k % 2) = priorE c (k / 2) (k % 2) := by
      unfold priorOf priorE
      by_cases h1 : k % 2 = 1
      · simp only [h1, ↓reduceIte]
        rw [htab, show 2 * (k / 2) = k - 1 by omega]
        exact hback h1
      · simp only [h1, ↓reduceIte]
    rw [hcms, htb, hpr, show asksCut (fun k i => slotAt c k i) n = ioE c n from rfl]
    refine both_bind (both_of_nf (repCut_nf h0 h1 h2 hx hw τ c n hn hsx ⟨k / 2, hi⟩ ⟨k % 2, hr2⟩ _ cms[k / 2]! ic
      extras hex l (by rw [hl]; congr 1; show k = 2 * (k / 2) + k % 2; omega))) ?_
    rintro ⟨pfC, tC⟩ ⟨pfP, tP⟩ l3 ⟨htr, e3⟩
    dsimp only at htr
    subst htr
    have e3' := repPZk_cut (coin := fun k i => slotAt c k i) (n := n) (resp := respOf c)
      (fun k => coinOf_respOf c k) hsE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (_root_.Flock.Transcript.init #[] _) l _ l3 e3
    rw [htab] at e3' hex
    obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ :=
      FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ hv
    obtain ⟨hn7, hL2, -, -, -, -, -⟩ := hdim
    obtain ⟨hl3, -, -, -⟩ := repPZk_exact (hm := hm) (hn := by rw [hn7, hsm]; omega) (hk0 := hk0) (hL := hL2)
      (hq := fun i hi => (hlv i hi).q_ne_zero) (h := e3')
    have hx' : extras.size = spec.points * (stsOf x)[k / 2]!.regions.size := by
      rw [JsonRT.extraClaims_size hex]
      simp [Nat.mul_comm]
    have e6 : (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!).fold.kLog = (stsOf x)[k / 2]!.c.kLog := rfl
    have e7 : (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!).m = (stsOf x)[k / 2]!.m := rfl
    have e8 : (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!).kLog = (stsOf x)[k / 2]!.c.kLog := rfl
    simp only [JsonRT.go_pure]
    refine ⟨rfl, ?_⟩
    dsimp only
    refine ⟨_, _, rfl, rfl, rfl, ?_, fun hk1 => ?_⟩
    · rw [repBase_eq hsx, FlockComplete.ZkTables.repStart_succ, ← FlockComplete.ZkTables.repRoundsT_eq hv hx', hl3,
        hl, repBase_eq hsx]
      unfold FlockComplete.ZkTables.repRoundsT piopZk
      rw [e6, e7, hx']
      dsimp only
      omega
    · rw [show (k + 1) / 2 = k / 2 by omega, Array.back!_push, show k + 1 - 1 = k by omega]
      have hpts := FlockComplete.ZkTables.repPZk_points (hm := hm) (hk := hk6) (h := e3') (hsz := by
        rw [hl3]
        unfold piopZk
        rw [e6, e7, e8]
        omega)
      simp only [coinOf_respOf] at hpts
      unfold ptsOpen
      rw [← hl]
      exact hpts
  · refine ⟨rfl, ?_, fun h => absurd h (by decide)⟩
    rw [repBase_eq hsx]
    simp [FlockComplete.repStartZk]
  · rintro ⟨sC, pC⟩ ⟨sP, pP⟩ l4 -
    dsimp only
    exact (Still.bind (recordJson_still _ _ _ _ _) (fun _ => Still.pure _) l4).trans
      (Still.bind (recordJson_still _ _ _ _ _) (fun _ => Still.pure _) l4).symm

end Masked

end Claims.ZKMal
