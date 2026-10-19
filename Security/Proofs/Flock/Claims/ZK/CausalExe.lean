import Proofs.Flock.Claims.ZK.Split

/-!
# The deployed round plan reads the coins before each round only (`CausalExe`)

`QE`'s round `n ≥ 1` is body `n` of `sessionCut` on the cut's sources at the coins `c`, answered by `asksCut`, which
returns coin `k` right after body `k` and throws once body `n` is sent. So body `n` reads the coins before `n` through
the sends (`asksCut_congr`), and the sources through what the session reads before it sends body `n`. Every source
value the session reads while it has sent `s` bodies reads only the coins before `s` (its fields at `restAt`'s slots,
and the guards before them), and `s ≤ n` until the throw, so body `n` is the same at any coins agreeing before `n`.

* **Equal runs** (`EqAt`): two runs from the same bodies return the same, and where they return they sent a known
  number of bodies, at most `n`. Binds (`eqAt_bindT`, `eqAt_bindL`), loops (`eqAt_forIn`, `eqAt_forInL`,
  `eqAt_forIn_array`) and `asksCut`'s squeezes.
* **The phases at sources that agree where read** (`RepAgree`): each phase of `repCut` from the bodies it starts at,
  then `sessionCut` over the rep streams from `repStartZk`.
* **The cut's sources agree where read** (`repAgree_czExe`): at coins agreeing before `n`, each field `repSrcCut` reads
  is the same wherever the session reads it before body `n`.
-/

namespace FlockComplete.CutEq

open Flock Transcript Replay

variable {α β σ : Type}

/-! ## Equal runs -/

/-- **Equal runs that send `N` bodies**: the same result, and where it returns, `N ≤ n` bodies sent in all and a value
with `P`. -/
def EqAt (n N : ℕ) (P : α → Prop) (x y : Except String α × Array Claims.Msg) : Prop :=
  x = y ∧ ∀ a L, x = (.ok a, L) → L.size = N ∧ N ≤ n ∧ P a

theorem eqAt_refl {n N : ℕ} {P : α → Prop} {x : Except String α × Array Claims.Msg}
    (h : ∀ a L, x = (.ok a, L) → L.size = N ∧ N ≤ n ∧ P a) : EqAt n N P x x := ⟨rfl, h⟩

theorem EqAt.mono {n N : ℕ} {P Q : α → Prop} {x y : Except String α × Array Claims.Msg} (h : EqAt n N P x y)
    (hPQ : ∀ a, P a → Q a) : EqAt n N Q x y :=
  ⟨h.1, fun a L e => ⟨(h.2 a L e).1, (h.2 a L e).2.1, hPQ a (h.2 a L e).2.2⟩⟩

theorem EqAt.cast {n N N' : ℕ} {P : α → Prop} {x y : Except String α × Array Claims.Msg} (h : EqAt n N P x y)
    (e : N = N') : EqAt n N' P x y := e ▸ h

theorem bind_go_err {x : TMW PS α} {f : α → TMW PS β} {t : Transcript} {l l' : Array Claims.Msg} {e : String}
    (h : (x.run t).go l = (.error e, l')) : ((x >>= f).run t).go l = (.error e, l') := by
  rw [StateT.run_bind, go_bind, h]

theorem eqAt_bind {n N M : ℕ} {P : α → Prop} {Q : β → Prop} {x y : PS α} {f g : α → PS β} {l : Array Claims.Msg}
    (hxy : EqAt n N P (x.go l) (y.go l))
    (hfg : ∀ a l', x.go l = (.ok a, l') → l'.size = N → N ≤ n → P a → EqAt n M Q ((f a).go l') ((g a).go l')) :
    EqAt n M Q ((x >>= f).go l) ((y >>= g).go l) := by
  obtain ⟨e, hok⟩ := hxy
  rw [go_bind, go_bind, ← e]
  rcases hx : x.go l with ⟨r, l'⟩
  cases r with
  | error err => exact ⟨rfl, fun a L h => by simp at h⟩
  | ok a =>
    obtain ⟨h1, h2, h3⟩ := hok a l' hx
    exact hfg a l' hx h1 h2 h3

/-- **A bind of equal runs**, the continuations equal from each return. -/
theorem eqAt_bindT {n N M : ℕ} {P : α × Transcript → Prop} {Q : β × Transcript → Prop} {x y : TMW PS α}
    {f g : α → TMW PS β} {t : Transcript} {l : Array Claims.Msg} (hxy : EqAt n N P ((x.run t).go l) ((y.run t).go l))
    (hfg : ∀ a t' l', l'.size = N → N ≤ n → P (a, t') →
      EqAt n M Q (((f a).run t').go l') (((g a).run t').go l')) :
    EqAt n M Q (((x >>= f).run t).go l) (((y >>= g).run t).go l) := by
  rw [StateT.run_bind, StateT.run_bind]
  exact eqAt_bind hxy fun p l' _ h1 h2 h3 => hfg p.1 p.2 l' h1 h2 h3

/-- **A Ligerito bind of equal runs.** -/
theorem eqAt_bindL {n N M : ℕ} {P : (α × LigState) × Transcript → Prop} {Q : (β × LigState) × Transcript → Prop}
    {x y : LP PS α} {f g : α → LP PS β} {st : LigState} {t : Transcript} {l : Array Claims.Msg}
    (hxy : EqAt n N P (((x.run st).run t).go l) (((y.run st).run t).go l))
    (hfg : ∀ a st' t' l', l'.size = N → N ≤ n →
      P ((a, st'), t') → EqAt n M Q ((((f a).run st').run t').go l') ((((g a).run st').run t').go l')) :
    EqAt n M Q ((((x >>= f).run st).run t).go l) ((((y >>= g).run st).run t).go l) := by
  rw [StateT.run_bind, StateT.run_bind]
  exact eqAt_bindT hxy fun p t' l' h1 h2 h3 => hfg p.1 p.2 t' l' h1 h2 h3

theorem eqAt_pureT (a : α) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) (((pure a : TMW PS α).run t).go l) (((pure a : TMW PS α).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨-, rfl⟩ := Prod.mk.inj h
    exact ⟨hl, hN, trivial⟩

theorem eqAt_pureP {P : α × Transcript → Prop} (a : α) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ}
    (hl : l.size = N) (hN : N ≤ n) (hP : P (a, t)) :
    EqAt n N P (((pure a : TMW PS α).run t).go l) (((pure a : TMW PS α).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨e, rfl⟩ := Prod.mk.inj h
    obtain rfl := Except.ok.inj e
    exact ⟨hl, hN, hP⟩

theorem eqAt_purePL {P : (α × LigState) × Transcript → Prop} (a : α) (st : LigState) (t : Transcript)
    {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n) (hP : P ((a, st), t)) :
    EqAt n N P ((((pure a : LP PS α).run st).run t).go l) ((((pure a : LP PS α).run st).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨e, rfl⟩ := Prod.mk.inj h
    obtain rfl := Except.ok.inj e
    exact ⟨hl, hN, hP⟩

theorem eqAt_pureL (a : α) (st : LigState) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N)
    (hN : N ≤ n) :
    EqAt n N (fun _ => True) ((((pure a : LP PS α).run st).run t).go l) ((((pure a : LP PS α).run st).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨-, rfl⟩ := Prod.mk.inj h
    exact ⟨hl, hN, trivial⟩

/-- **A step that sends nothing**, the same on both sides. -/
theorem eqAt_silentT {x : TMW PS α} {t : Transcript} {l : Array Claims.Msg} {n N : ℕ}
    (hx : ∀ a t' l', (x.run t).go l = (.ok (a, t'), l') → l' = l) (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) ((x.run t).go l) ((x.run t).go l) :=
  eqAt_refl fun p L h => by
    obtain rfl := hx p.1 p.2 L h
    exact ⟨hl, hN, trivial⟩

theorem eqAt_liftTM (x : TM α) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) (((liftTM x : TMW PS α).run t).go l) (((liftTM x : TMW PS α).run t).go l) :=
  eqAt_silentT (fun _ _ _ h => (congrArg Prod.snd ((liftTM_go x t l).symm.trans h)).symm) hl hN

theorem eqAt_liftV (v : V α) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) (((StateT.lift (monadLift v : PS α) : TMW PS α).run t).go l)
      (((StateT.lift (monadLift v : PS α) : TMW PS α).run t).go l) := by
  refine eqAt_silentT (fun a t' l' h => ?_) hl hN
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
  obtain ⟨b, l₂, e3, e4⟩ := JsonRT.go_bind_ok h
  rw [JsonRT.monadLift_go] at e3
  obtain ⟨-, rfl⟩ := Prod.mk.inj e3
  exact (Prod.mk.inj e4).2.symm

theorem eqAt_getT (t : Transcript) {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) (((get : TMW PS Transcript).run t).go l) (((get : TMW PS Transcript).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨-, rfl⟩ := Prod.mk.inj h
    exact ⟨hl, hN, trivial⟩

/-- **A Ligerito step of the session's monad**, lifted. -/
theorem eqAt_liftT {x : TMW PS α} {st : LigState} {t : Transcript} {l : Array Claims.Msg} {n N : ℕ}
    (h : EqAt n N (fun _ => True) ((x.run t).go l) ((x.run t).go l)) :
    EqAt n N (fun _ => True) ((((liftT x : LP PS α).run st).run t).go l) ((((liftT x : LP PS α).run st).run t).go l) := by
  refine eqAt_refl fun p L e => ?_
  change ((x >>= fun a => Pure.pure (a, st)).run t).go l = _ at e
  rw [StateT.run_bind] at e
  obtain ⟨⟨a, t₁⟩, l₁, e1, e2⟩ := JsonRT.go_bind_ok e
  obtain ⟨-, rfl⟩ := Prod.mk.inj e2
  exact ⟨(h.2 _ _ e1).1, (h.2 _ _ e1).2.1, trivial⟩

/-- **A Ligerito step that sends nothing.** -/
theorem eqAt_lsilent {x : LP PS α} (hx : LSilent x) (st : LigState) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ}
    (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) (((x.run st).run t).go l) (((x.run st).run t).go l) :=
  eqAt_refl fun p L h => by
    obtain rfl := hx st t l p.1.1 p.1.2 p.2 L h
    exact ⟨hl, hN, trivial⟩

/-- A step of the session's monad that sends nothing. -/
def TSilent (x : TMW PS α) : Prop :=
  ∀ t l a t' l', (x.run t).go l = (.ok (a, t'), l') → l' = l

theorem tsilent_liftTM (x : TM α) : TSilent (liftTM x : TMW PS α) :=
  fun t l _ _ _ h => (congrArg Prod.snd ((liftTM_go x t l).symm.trans h)).symm

/-- **A bind after a step that sends nothing**: the continuations from the same bodies. -/
theorem eqAt_tsilent_bind {x : TMW PS α} (hx : TSilent x) {f g : α → TMW PS β} {t : Transcript}
    {l : Array Claims.Msg} {n M : ℕ} {Q : β × Transcript → Prop}
    (hfg : ∀ a t', EqAt n M Q (((f a).run t').go l) (((g a).run t').go l)) :
    EqAt n M Q (((x >>= f).run t).go l) (((x >>= g).run t).go l) := by
  rw [StateT.run_bind, StateT.run_bind, go_bind, go_bind]
  rcases hxr : (x.run t).go l with ⟨r, l'⟩
  cases r with
  | error e => exact ⟨rfl, fun a L h => by simp at h⟩
  | ok p =>
    obtain ⟨a, t'⟩ := p
    obtain rfl := hx t l a t' l' hxr
    exact hfg a t'

/-- **A Ligerito bind after a step that sends nothing.** -/
theorem eqAt_lsilent_bind {x : LP PS α} (hx : LSilent x) {f g : α → LP PS β} {st : LigState} {t : Transcript}
    {l : Array Claims.Msg} {n M : ℕ} {Q : (β × LigState) × Transcript → Prop}
    (hfg : ∀ a st' t', EqAt n M Q ((((f a).run st').run t').go l) ((((g a).run st').run t').go l)) :
    EqAt n M Q ((((x >>= f).run st).run t).go l) ((((x >>= g).run st).run t).go l) := by
  rw [StateT.run_bind, StateT.run_bind, StateT.run_bind, StateT.run_bind, go_bind, go_bind]
  rcases hxr : ((x.run st).run t).go l with ⟨r, l'⟩
  cases r with
  | error e => exact ⟨rfl, fun a L h => by simp at h⟩
  | ok p =>
    obtain ⟨⟨a, st'⟩, t'⟩ := p
    obtain rfl := hx st t l a st' t' l' hxr
    exact hfg a st' t'

/-! ## Loops -/

/-- **A loop of equal steps**: step `i`, from `L + g i ≤ n` bodies, yields after `L + g (i + 1)`; the loops over
`[0:k]` end after `L + g k`. -/
theorem eqAt_forIn {n : ℕ} (body body' : ℕ → σ → TMW PS (ForInStep σ)) (L : ℕ) (g : ℕ → ℕ) (k : ℕ)
    (hb : ∀ i < k, ∀ (s : σ) (t : Transcript) (l : Array Claims.Msg), l.size = L + g i → L + g i ≤ n →
      EqAt n (L + g (i + 1)) (fun p => ∃ u, p.1 = .yield u) (((body i s).run t).go l) (((body' i s).run t).go l))
    (s : σ) (t : Transcript) (l : Array Claims.Msg) (hl : l.size = L + g 0) (hL : L + g 0 ≤ n) :
    EqAt n (L + g k) (fun _ => True) (((forIn [:k] s body).run t).go l) (((forIn [:k] s body').run t).go l) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ j a (s : σ) (t : Transcript) (l : Array Claims.Msg), a + j = k → l.size = L + g a →
      L + g a ≤ n → EqAt n (L + g k) (fun _ => True)
        (((forIn (List.range' a j) s body).run t).go l) (((forIn (List.range' a j) s body').run t).go l) from
    key k 0 s t l (by omega) hl hL
  intro j
  induction j with
  | zero =>
    intro a s t l ha hl hL
    obtain rfl : a = k := by omega
    exact eqAt_pureT s t hl hL
  | succ j ih =>
    intro a s t l ha hl hL
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine eqAt_bindT (hb a (by omega) s t l hl hL) fun r t₁ l₁ hl₁ hL₁ hr => ?_
    obtain ⟨u, hu⟩ := hr
    dsimp only at hu
    subst hu
    exact ih (a + 1) u t₁ l₁ (by omega) hl₁ hL₁

/-- **A Ligerito loop of equal steps.** -/
theorem eqAt_forInL {n : ℕ} (body body' : ℕ → σ → LP PS (ForInStep σ)) (L : ℕ) (g : ℕ → ℕ) (k : ℕ)
    (hb : ∀ i < k, ∀ (s : σ) (st : LigState) (t : Transcript) (l : Array Claims.Msg), l.size = L + g i →
      L + g i ≤ n → EqAt n (L + g (i + 1)) (fun p => ∃ u, p.1.1 = .yield u)
        ((((body i s).run st).run t).go l) ((((body' i s).run st).run t).go l))
    (s : σ) (st : LigState) (t : Transcript) (l : Array Claims.Msg) (hl : l.size = L + g 0) (hL : L + g 0 ≤ n) :
    EqAt n (L + g k) (fun _ => True) ((((forIn [:k] s body).run st).run t).go l)
      ((((forIn [:k] s body').run st).run t).go l) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ j a (s : σ) (st : LigState) (t : Transcript) (l : Array Claims.Msg), a + j = k →
      l.size = L + g a → L + g a ≤ n → EqAt n (L + g k) (fun _ => True)
        ((((forIn (List.range' a j) s body).run st).run t).go l)
        ((((forIn (List.range' a j) s body').run st).run t).go l) from
    key k 0 s st t l (by omega) hl hL
  intro j
  induction j with
  | zero =>
    intro a s st t l ha hl hL
    obtain rfl : a = k := by omega
    exact eqAt_pureL s st t hl hL
  | succ j ih =>
    intro a s st t l ha hl hL
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine eqAt_bindL (hb a (by omega) s st t l hl hL) fun r st₁ t₁ l₁ hl₁ hL₁ hr => ?_
    obtain ⟨u, hu⟩ := hr
    dsimp only at hu
    subst hu
    exact ih (a + 1) u st₁ t₁ l₁ (by omega) hl₁ hL₁

/-- **A loop over an array of equal steps**: step `k`, from `g k ≤ n` bodies, yields after `g (k + 1)`. -/
theorem eqAt_forIn_array {n : ℕ} (xs : Array α) (body body' : α → σ → PS (ForInStep σ)) (g : ℕ → ℕ)
    (hb : ∀ k (hk : k < xs.size), ∀ (s : σ) (l : Array Claims.Msg), l.size = g k → g k ≤ n →
      EqAt n (g (k + 1)) (fun r => ∃ u, r = .yield u) ((body xs[k] s).go l) ((body' xs[k] s).go l))
    (s : σ) (l : Array Claims.Msg) (hl : l.size = g 0) (hL : g 0 ≤ n) :
    EqAt n (g xs.size) (fun _ => True) ((forIn xs s body).go l) ((forIn xs s body').go l) := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  suffices key : ∀ (ys : List α) (j : ℕ) (s : σ) (l : Array Claims.Msg), xs.toList.drop j = ys →
      j ≤ xs.size → l.size = g j → g j ≤ n →
      EqAt n (g xs.size) (fun _ => True) ((forIn ys s body).go l) ((forIn ys s body').go l) from
    key xs.toList 0 s l (by simp) (Nat.zero_le _) hl hL
  intro ys
  induction ys with
  | nil =>
    intro j s l hd hj hl hL
    have hj' : j = xs.size := by
      have := congrArg List.length hd
      simp only [List.length_drop, Array.length_toList, List.length_nil] at this
      omega
    subst hj'
    refine eqAt_refl fun _ L h => ?_
    rw [List.forIn_nil, JsonRT.go_pure] at h
    obtain ⟨-, rfl⟩ := Prod.mk.inj h
    exact ⟨hl, hL, trivial⟩
  | cons a ys ih =>
    intro j s l hd _ hl hL
    have hget : xs.toList[j]? = some a := by rw [← List.head?_drop, hd]; rfl
    rw [Array.getElem?_toList] at hget
    obtain ⟨hjx, hxa⟩ := Array.getElem?_eq_some_iff.1 hget
    rw [List.forIn_cons, List.forIn_cons]
    have hb' := hb j hjx s l hl hL
    rw [hxa] at hb'
    refine eqAt_bind hb' fun r l' _ hl' hL' hr => ?_
    obtain ⟨u, rfl⟩ := hr
    exact ih (j + 1) u l' (by rw [← List.drop_drop, hd]; rfl) hjx hl' hL'

/-! ## `asksCut`'s sends -/

section Asks

variable (coin : ℕ → ℕ → F128) (n : ℕ)

theorem asksCut_round_go (c : ByteArray) (l : Array Claims.Msg) :
    ((asksCut coin n).round c).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : PS (ℕ → F128)).go
        (l.push (Claims.toMsg c))) := rfl

theorem squeezeP_cut_ok (kind : UInt8) {k : ℕ} (hk : k ≠ 0) (t : Transcript) {l : Array Claims.Msg}
    (hl : l.size < n) :
    ((squeezeP (asksCut coin n) kind k).run t).go l =
      (.ok ((Array.range k).map (coin l.size), afterSq t kind k (coin l.size)),
        l.push (Claims.toMsg (framed t kind k))) := by
  simp only [squeezeP, StateT.run, hk, ite_false]
  rw [go_bind, asksCut_round_go, ite_eq_right (show ¬ n ≤ l.size by omega)]
  rfl

theorem squeezeP_cut_err (kind : UInt8) {k : ℕ} (hk : k ≠ 0) (t : Transcript) {l : Array Claims.Msg}
    (hl : n ≤ l.size) :
    ((squeezeP (asksCut coin n) kind k).run t).go l =
      (.error "past the openings", l.push (Claims.toMsg (framed t kind k))) := by
  simp only [squeezeP, StateT.run, hk, ite_false]
  rw [go_bind, asksCut_round_go, ite_eq_left hl]
  rfl

/-- **A squeeze sends one body**, and returns only before round `n`. -/
theorem eqAt_squeezeP (kind : UInt8) {k : ℕ} (hk : k ≠ 0) (t : Transcript) {l : Array Claims.Msg} {N : ℕ}
    (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) (((squeezeP (asksCut coin n) kind k).run t).go l)
      (((squeezeP (asksCut coin n) kind k).run t).go l) := by
  refine eqAt_refl fun a L h => ?_
  by_cases hn : n ≤ l.size
  · rw [squeezeP_cut_err coin n kind hk t hn] at h
    simp at h
  · rw [squeezeP_cut_ok coin n kind hk t (by omega)] at h
    obtain ⟨-, rfl⟩ := Prod.mk.inj h
    exact ⟨by simp [hl], by omega, trivial⟩

theorem eqAt_sampleVecP {k : ℕ} (hk : k ≠ 0) (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) (((sampleVecP (asksCut coin n) k).run t).go l)
      (((sampleVecP (asksCut coin n) k).run t).go l) :=
  eqAt_squeezeP coin n 2 hk t hl

theorem eqAt_sampleP (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) (((sampleP (asksCut coin n)).run t).go l)
      (((sampleP (asksCut coin n)).run t).go l) := by
  unfold sampleP
  exact eqAt_bindT (eqAt_squeezeP coin n 1 one_ne_zero t hl) fun _ t' _ h1 h2 _ => eqAt_pureT _ t' h1 h2

theorem eqAt_sampleKP (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) (((sampleKP (asksCut coin n)).run t).go l)
      (((sampleKP (asksCut coin n)).run t).go l) := by
  unfold sampleKP
  exact eqAt_bindT (eqAt_squeezeP coin n 2 two_ne_zero t hl) fun _ t' _ h1 h2 _ => eqAt_pureT _ t' h1 h2

theorem eqAt_link (rootB publics : Array ByteArray) (hn : 1 ≤ n) :
    EqAt n 1 (fun _ => True) (((asksCut coin n).link rootB publics).go #[])
      (((asksCut coin n).link rootB publics).go #[]) := by
  refine eqAt_refl fun a L h => ?_
  have e : ((asksCut coin n).link rootB publics).go #[] =
      (.ok (coin 0), (#[] : Array Claims.Msg).push (FlockDeploy.linkBody rootB.toList publics.toList)) := by
    show ((if n ≤ 0 then throw "past the openings" else pure (coin 0) : PS (ℕ → F128)).go _) = _
    rw [ite_eq_right (show ¬ n ≤ 0 by omega)]
    rfl
  rw [e] at h
  obtain ⟨-, rfl⟩ := Prod.mk.inj h
  exact ⟨rfl, hn, trivial⟩

/-! ### Ligerito's sends -/

theorem eqAt_lsampleKP (st : LigState) (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) ((((liftT (sampleKP (asksCut coin n)) : LP PS F256).run st).run t).go l)
      ((((liftT (sampleKP (asksCut coin n)) : LP PS F256).run st).run t).go l) :=
  eqAt_liftT (eqAt_sampleKP coin n t hl)

theorem eqAt_lsampleP (st : LigState) (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) ((((liftT (sampleP (asksCut coin n)) : LP PS F128).run st).run t).go l)
      ((((liftT (sampleP (asksCut coin n)) : LP PS F128).run st).run t).go l) :=
  eqAt_liftT (eqAt_sampleP coin n t hl)

theorem eqAt_lsampleVecP {k : ℕ} (hk : k ≠ 0) (st : LigState) (t : Transcript) {l : Array Claims.Msg} {N : ℕ}
    (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True)
      ((((liftT (sampleVecP (asksCut coin n) k) : LP PS (Array F128)).run st).run t).go l)
      ((((liftT (sampleVecP (asksCut coin n) k) : LP PS (Array F128)).run st).run t).go l) :=
  eqAt_liftT (eqAt_sampleVecP coin n hk t hl)

theorem eqAt_liftTML (x : TM α) (st : LigState) (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N)
    (hN : N ≤ n) :
    EqAt n N (fun _ => True) ((((liftT (liftTM x) : LP PS α).run st).run t).go l)
      ((((liftT (liftTM x) : LP PS α).run st).run t).go l) :=
  eqAt_liftT (eqAt_liftTM x t hl hN)

theorem eqAt_challengeP (split : Bool) (st : LigState) (t : Transcript) {l : Array Claims.Msg} {N : ℕ}
    (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) ((((challengeP (asksCut coin n) split).run st).run t).go l)
      ((((challengeP (asksCut coin n) split).run st).run t).go l) := by
  unfold challengeP
  refine eqAt_bindL (eqAt_lsampleKP coin n st t hl) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_modify _) st₁ t₁ h1 h2) fun _ st₂ t₂ l₂ h1 h2 _ => ?_
  split
  · refine eqAt_bindL (eqAt_lsilent (lsilent_modify _) st₂ t₂ h1 h2) fun _ st₃ t₃ l₃ h1 h2 _ => ?_
    refine eqAt_bindL (eqAt_lsilent lsilent_get st₃ t₃ h1 h2) fun _ st₄ t₄ l₄ h1 h2 _ => ?_
    refine eqAt_bindL (eqAt_lsilent (lsilent_sendMsg _) st₄ t₄ h1 h2) fun _ st₅ t₅ l₅ h1 h2 _ => ?_
    exact eqAt_pureL _ st₅ t₅ h1 h2
  · refine eqAt_bindL (eqAt_lsilent lsilent_get st₂ t₂ h1 h2) fun _ st₄ t₄ l₄ h1 h2 _ => ?_
    refine eqAt_bindL (eqAt_lsilent (lsilent_sendMsg _) st₄ t₄ h1 h2) fun _ st₅ t₅ l₅ h1 h2 _ => ?_
    exact eqAt_pureL _ st₅ t₅ h1 h2

theorem eqAt_oodIntroP (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ} (hdim : dim ≠ 0) (st : LigState)
    (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 2) (fun _ => True) ((((oodIntroP (asksCut coin n) nonce g dim).run st).run t).go l)
      ((((oodIntroP (asksCut coin n) nonce g dim).run st).run t).go l) := by
  unfold oodIntroP
  refine eqAt_bindL (eqAt_lsampleVecP coin n hdim st t hl) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_modify _) st₁ t₁ h1 h2) fun _ st₂ t₂ l₂ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_liftTML n _ st₂ t₂ h1 h2) fun _ st₃ t₃ l₃ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_sendMsg _) st₃ t₃ h1 h2) fun _ st₄ t₄ l₄ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_powB nonce) st₄ t₄ h1 h2) fun _ st₅ t₅ l₅ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsampleP coin n st₅ t₅ h1) fun _ st₆ t₆ l₆ h1 h2 _ => ?_
  exact eqAt_lsilent (lsilent_modify _) st₆ t₆ h1 h2

theorem eqAt_queryP (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) (hq : lv.q ≠ 0)
    (st : LigState) (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n (N + 2) (fun _ => True) ((((queryP (asksCut coin n) nonce ms lv cm).run st).run t).go l)
      ((((queryP (asksCut coin n) nonce ms lv cm).run st).run t).go l) := by
  unfold queryP
  refine eqAt_bindL (eqAt_lsilent (lsilent_powP nonce 0) st t hl hN) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsampleVecP coin n hq st₁ t₁ h1) fun _ st₂ t₂ l₂ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_powP nonce 1) st₂ t₂ h1 h2) fun _ st₃ t₃ l₃ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsampleVecP coin n (Nat.succ_ne_zero _) st₃ t₃ h1) fun _ st₄ t₄ l₄ h1 h2 _ => ?_
  exact eqAt_pureL _ st₄ t₄ h1 h2

end Asks

/-! ## A rep's phases at sources that agree where read -/

section Phases

variable (coin : ℕ → ℕ → F128) (n : ℕ)

/-- A refusal, whatever follows it. -/
theorem eqAt_throwT (err : String) {f g : α → TMW PS β} {Q : β × Transcript → Prop} (t : Transcript)
    (l : Array Claims.Msg) {M : ℕ} :
    EqAt n M Q ((((throw err : TMW PS α) >>= f).run t).go l) ((((throw err : TMW PS α) >>= g).run t).go l) := by
  have e : ((throw err : TMW PS α).run t).go l = (.error err, l) := rfl
  rw [bind_go_err e, bind_go_err e]
  exact eqAt_refl fun _ _ h => by simp at h

theorem eqAt_rankCheckP (m kLog : ℕ) (prior : Array (List F128)) (words : List ℕ) (t : Transcript)
    {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) (((rankCheckP m kLog prior words : TMW PS Unit).run t).go l)
      (((rankCheckP m kLog prior words : TMW PS Unit).run t).go l) := by
  refine eqAt_silentT (fun a t' l' h => ?_) hl hN
  unfold rankCheckP at h
  simp only [StateT.run_bind, StateT.run_get, pure_bind] at h
  split at h
  · exact (Prod.mk.inj h).2.symm
  · change ((throw _ : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨_, _, h1, -⟩ := JsonRT.go_bind_ok h
    cases h1

/-- **The cut's zerocheck at masked values agreeing where sent**: `round1_ab`'s and `round1_c`'s after its `2` sends,
residual round `i`'s after `3 + i`, the finals after `3 + (m − 6)`, each where that many bodies are at most `n`. -/
theorem zerocheckCut_eqAt {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (mk mk' : ℕ → F128) (t : Transcript) {l : Array Claims.Msg} {L : ℕ} (hl : l.size = L) (hL : L ≤ n)
    (hab : L + 2 ≤ n → ∀ i < 64, mk (lay.r1ab i) = mk' (lay.r1ab i) ∧ mk (lay.r1c i) = mk' (lay.r1c i))
    (hml : ∀ i < m - 6, L + 3 + i ≤ n → mk (lay.ml i 0) = mk' (lay.ml i 0) ∧ mk (lay.ml i 1) = mk' (lay.ml i 1))
    (hf : L + 3 + (m - 6) ≤ n → mk lay.fa = mk' lay.fa ∧ mk lay.fb = mk' lay.fb ∧ mk lay.fc = mk' lay.fc) :
    EqAt n (L + 3 + (m - 6)) (fun _ => True)
      (((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l)
      (((zerocheckCut (asksCut coin n) m digest cap root lay mk').run t).go l) := by
  unfold zerocheckCut
  refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleVecP coin n (by norm_num) t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleVecP coin n (by omega) t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
  dsimp only
  have e1 : ((Array.range 64).map fun i => mk (lay.r1ab i)) = (Array.range 64).map fun i => mk' (lay.r1ab i) :=
    Array.map_congr_left fun i hi => ((hab (by omega)) i (by simpa using hi)).1
  have e2 : ((Array.range 64).map fun i => mk (lay.r1c i)) = (Array.range 64).map fun i => mk' (lay.r1c i) :=
    Array.map_congr_left fun i hi => ((hab (by omega)) i (by simpa using hi)).2
  rw [e1, e2]
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_forIn _ _ (L + 3) (fun i => i) (m - 6) (fun i hi s t l hl hL => ?_) _ t₁ l₁
    (by omega) (by omega)) fun r t₁ l₁ h1 h2 _ => ?_
  · rw [(hml i hi (by omega)).1, (hml i hi (by omega)).2]
    refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
    exact eqAt_pureP _ t₁ (by omega) (by omega) ⟨_, rfl⟩
  · obtain ⟨ea, eb, ec⟩ := hf (by omega)
    rw [ea, eb, ec]
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
    exact eqAt_pureT _ t₁ h1 h2

/-- **The cut's lincheck at masked values agreeing where sent**: lincheck round `j`'s after its `2 + j` sends, `z_partial`
after `2 + (k_log − 6)`. -/
theorem lincheckCut_eqAt (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk mk' : ℕ → F128) (zc : ZcOut)
    (t : Transcript) {l : Array Claims.Msg} {L : ℕ} (hl : l.size = L) (hL : L ≤ n)
    (hlc : ∀ j < c.kLog - 6, L + 2 + j ≤ n → mk (lay.lc j 0) = mk' (lay.lc j 0) ∧ mk (lay.lc j 1) = mk' (lay.lc j 1))
    (hzp : L + 2 + (c.kLog - 6) ≤ n → ∀ i < 64, mk (lay.zp i) = mk' (lay.zp i)) :
    EqAt n (L + 3 + (c.kLog - 6)) (fun _ => True)
      (((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l)
      (((lincheckCut (asksCut coin n) c m lay mk' zc).run t).go l) := by
  unfold lincheckCut
  dsimp only
  refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun alpha t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
  rcases hfold : c.fold alpha (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) with err | v
  · exact eqAt_throwT n err t₁ l₁
  refine eqAt_bindT (eqAt_pureT v t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_forIn _ _ (L + 2) (fun i => i) (c.kLog - 6) (fun i hi s t l hl hL => ?_) _ t₁ l₁
    (by omega) (by omega)) fun r t₁ l₁ h1 h2 _ => ?_
  · rw [(hlc i hi (by omega)).1, (hlc i hi (by omega)).2]
    refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
    exact eqAt_pureP _ t₁ (by omega) (by omega) ⟨_, rfl⟩
  · have e : ((Array.range 64).map fun i => mk (lay.zp i)) = (Array.range 64).map fun i => mk' (lay.zp i) :=
      Array.map_congr_left fun i hi => hzp (by omega) i (by simpa using hi)
    rw [e]
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
    exact eqAt_pureT _ t₁ (by omega) (by omega)

/-- **The ring switch at switched vectors agreeing where sent**: claim `k`'s after `k` sends. -/
theorem ringSwitchCut_eqAt (sv sv' : ℕ → Array F128) (claims : Array Claim) (hc : claims.size ≠ 0) (t : Transcript)
    {l : Array Claims.Msg} {L : ℕ} (hl : l.size = L) (hL : L ≤ n)
    (hsv : ∀ k < claims.size, L + k ≤ n → sv k = sv' k) :
    EqAt n (L + claims.size + 1) (fun _ => True)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l)
      (((ringSwitchCut (asksCut coin n) sv' claims).run t).go l) := by
  unfold ringSwitchCut
  dsimp only
  refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_forIn _ _ L (fun i => i) claims.size (fun i hi s t l hl hL => ?_) _ t₁ l₁
    (by omega) (by omega)) fun r t₁ l₁ h1 h2 _ => ?_
  · rw [hsv i hi (by omega)]
    refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleVecP coin n (by norm_num) t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
    exact eqAt_pureP _ t₁ (by omega) (by omega) ⟨_, rfl⟩
  · refine eqAt_bindT (eqAt_sampleVecP coin n hc t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
    exact eqAt_pureT _ t₁ h1 h2

/-- **The cut's PIOP at sources agreeing where sent**: the zerocheck's, the lincheck's and the ring switch's reads, each
at its send count. -/
theorem piopCut_eqAt (st : Setup) (hm : 14 ≤ st.m) (lay : Zk.PadLayout) (src src' : RepSrc) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) (t : Transcript) {l : Array Claims.Msg}
    {L : ℕ} (hl : l.size = L) (hL : L ≤ n)
    (hab : L + 2 ≤ n → ∀ i < 64, src.masked (lay.r1ab i) = src'.masked (lay.r1ab i) ∧
      src.masked (lay.r1c i) = src'.masked (lay.r1c i))
    (hml : ∀ i < st.m - 6, L + 3 + i ≤ n →
      src.masked (lay.ml i 0) = src'.masked (lay.ml i 0) ∧ src.masked (lay.ml i 1) = src'.masked (lay.ml i 1))
    (hf : L + 3 + (st.m - 6) ≤ n → src.masked lay.fa = src'.masked lay.fa ∧ src.masked lay.fb = src'.masked lay.fb ∧
      src.masked lay.fc = src'.masked lay.fc)
    (hlc : ∀ j < st.fold.kLog - 6, L + 3 + (st.m - 6) + 2 + j ≤ n →
      src.masked (lay.lc j 0) = src'.masked (lay.lc j 0) ∧ src.masked (lay.lc j 1) = src'.masked (lay.lc j 1))
    (hzp : L + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) ≤ n → ∀ i < 64, src.masked (lay.zp i) = src'.masked (lay.zp i))
    (hsv : ∀ k < 2 + extras.size, L + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + k ≤ n → src.sv k = src'.sv k) :
    EqAt n (L + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + (2 + extras.size) + 1) (fun _ => True)
      (((piopCut (asksCut coin n) st lay src cap root extras prior words).run t).go l)
      (((piopCut (asksCut coin n) st lay src' cap root extras prior words).run t).go l) := by
  unfold piopCut
  refine eqAt_bindT (zerocheckCut_eqAt coin n hm _ _ _ lay _ _ t hl hL hab hml hf) fun zz t₁ l₁ h1 h2 _ => ?_
  obtain ⟨zp, zc⟩ := zz
  dsimp only
  refine eqAt_bindT (lincheckCut_eqAt coin n _ _ lay _ _ zc t₁ h1 h2 hlc hzp) fun ll t₁ l₁ h1 h2 _ => ?_
  obtain ⟨lp, lc⟩ := ll
  dsimp only
  refine eqAt_bindT (eqAt_rankCheckP n _ _ prior words t₁ h1 h2) fun _ t₁ l₁ h1 h2 _ => ?_
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    simp [abcClaims, Nat.add_comm]
  refine eqAt_bindT (ringSwitchCut_eqAt coin n _ _ _ (by omega) t₁ h1 h2 fun k hk hkn => hsv k (by omega)
    (by omega)) fun rr t₁ l₁ h1 h2 _ => ?_
  obtain ⟨ss, T, sws⟩ := rr
  exact eqAt_pureT _ t₁ (by rw [h1, hsz]; omega) (by rw [← hsz]; omega)

/-- A Ligerito refusal, whatever follows it. -/
theorem eqAt_throwL (err : String) {f g : α → LP PS β} {Q : (β × LigState) × Transcript → Prop} (st : LigState)
    (t : Transcript) (l : Array Claims.Msg) {M : ℕ} :
    EqAt n M Q (((((throw err : LP PS α) >>= f).run st).run t).go l)
      (((((throw err : LP PS α) >>= g).run st).run t).go l) := by
  have e : (((((throw err : LP PS α) >>= f).run st).run t).go l) = (.error err, l) := rfl
  have e' : (((((throw err : LP PS α) >>= g).run st).run t).go l) = (.error err, l) := rfl
  rw [e, e']
  exact eqAt_refl fun _ _ h => by simp at h

/-- **The cut's Ligerito opening at sources agreeing where sent**: level 0's out-of-domain value after `1` send, lane
message `j`'s after `2 + j`, the pair's sums after `2 + k_0`, and the restart's `y₁'`, `ȳ` and the salts of levels ≥ 1
after `k_0 + 4`; it sends `k_0 + 7 + 11 R` bodies. -/
theorem ligeritoCut_eqAt (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit) (b : Array F128)
    (lay : Zk.PadLayout) (src src' : RepSrc) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (hR : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) (t : Transcript) {l : Array Claims.Msg} {L : ℕ}
    (hl : l.size = L) (hL : L ≤ n)
    (hood : L + 1 ≤ n → src.masked (lay.lgOod 0) = src'.masked (lay.lgOod 0))
    (hmsg : ∀ j < sch.k0, L + 2 + j ≤ n → ∀ i < 4, src.masked (lay.lgMsg j i) = src'.masked (lay.lgMsg j i))
    (hE : L + 2 + sch.k0 ≤ n → ∀ a < 2, ∀ i < 2, src.masked (lay.lgE a i) = src'.masked (lay.lgE a i))
    (hy : L + sch.k0 + 4 ≤ n → src.y1 = src'.y1 ∧ src.ybar = src'.ybar ∧ src.hiSalt = src'.hiSalt) :
    EqAt n (L + (sch.k0 + 7 + 11 * (sch.levels.size - 1))) (fun _ => True)
      (((ligeritoCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src).run t).go l)
      (((ligeritoCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src').run t).go l) := by
  unfold ligeritoCut
  dsimp only
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_bindT (N := L + (sch.k0 + 7 + 11 * (sch.levels.size - 1))) (P := fun _ => True) ?_
    fun r t₁ l₁ h1 h2 _ => ?_
  rotate_left
  · obtain ⟨⟨_, _, _, _, _, _, _, _, _⟩, _⟩ := r
    exact eqAt_pureT _ t₁ h1 h2
  refine eqAt_bindL (eqAt_lsampleVecP coin n hn _ t₁ hl) fun z st₁ t₁ l₁ h1 h2 _ => ?_
  rw [hood (by omega)]
  refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
  refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
  refine eqAt_lsilent_bind (lsilent_powB _) fun _ st₁ t₁ => ?_
  refine eqAt_bindL (eqAt_lsampleP coin n st₁ t₁ h1) fun b0 st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
  have hm0 := hmsg 0 (by omega) (by omega)
  rw [hm0 0 (by norm_num), hm0 1 (by norm_num), hm0 2 (by norm_num), hm0 3 (by norm_num)]
  refine eqAt_lsilent_bind (lsilent_sendMsg _) fun _ st₁ t₁ => ?_
  refine eqAt_bindL (eqAt_forInL _ _ (L + 2) (fun j => j + 2 * (j + 1 - sch.k0)) sch.k0
    (fun i hi s st t l hl hL => ?_) _ st₁ t₁ l₁ (by omega) (by omega)) fun s st₁ t₁ l₁ h1 h2 _ => ?_
  · have hg : i + 2 * (i + 1 - sch.k0) = i := by omega
    have hg' : i + 1 + 2 * (i + 1 + 1 - sch.k0) = if i + 1 = sch.k0 then sch.k0 + 2 else i + 1 := by
      split_ifs <;> omega
    rw [hg] at hl hL
    rw [hg']
    clear hg hg'
    refine eqAt_bindL (eqAt_lsampleKP coin n st t hl) fun r st₁ t₁ l₁ h1 h2 _ => ?_
    refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
    by_cases hj1 : (i + 1 == sch.k0) = true
    · have hik : i + 1 = sch.k0 := by simpa using hj1
      rw [ite_eq_left hj1, ite_eq_left hj1, ite_eq_left hik]
      have hE' := hE (by omega)
      rw [hE' 0 (by norm_num) 0 (by norm_num), hE' 0 (by norm_num) 1 (by norm_num), hE' 1 (by norm_num) 0 (by norm_num),
        hE' 1 (by norm_num) 1 (by norm_num)]
      clear hE'
      refine eqAt_lsilent_bind lsilent_get fun _ st₁ t₁ => ?_
      refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
      refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
      refine eqAt_bindL (eqAt_lsampleKP coin n st₁ t₁ h1) fun rh0 st₁ t₁ l₁ h1 h2 _ => ?_
      refine eqAt_bindL (eqAt_lsampleKP coin n st₁ t₁ h1) fun rh1 st₁ t₁ l₁ h1 h2 _ => ?_
      obtain ⟨ey, eyb, -⟩ := hy (by omega)
      rw [ey, eyb]
      by_cases hc : (!Zk.rhoIndependent (rh0, rh1)) = true
      · rw [ite_eq_left hc]
        exact eqAt_throwL n _ st₁ t₁ l₁
      · rw [ite_eq_right hc]
        refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind lsilent_get fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind (lsilent_sendMsg _) fun _ st₁ t₁ => ?_
        exact eqAt_purePL _ st₁ t₁ (by omega) (by omega) ⟨_, rfl⟩
    · have hi1 : i + 1 ≠ sch.k0 := by simpa using hj1
      rw [ite_eq_right hj1, ite_eq_right hj1, ite_eq_right hi1]
      have hm := hmsg (i + 1) (by omega) (by omega)
      rw [hm 0 (by norm_num), hm 1 (by norm_num), hm 2 (by norm_num), hm 3 (by norm_num)]
      refine eqAt_lsilent_bind (lsilent_sendMsg _) fun _ st₁ t₁ => ?_
      exact eqAt_purePL _ st₁ t₁ (by omega) (by omega) ⟨_, rfl⟩
  have h2' : L + sch.k0 + 4 ≤ n := by omega
  have h1' : l₁.size = L + sch.k0 + 4 := by omega
  clear h1 h2
  obtain ⟨-, -, ehs⟩ := hy h2'
  rw [ehs]
  refine eqAt_lsilent_bind lsilent_get fun _ st₁ t₁ => ?_
  refine eqAt_lsilent_bind (lsilent_quiet (commitP_quiet Sha512.hash _ _ _ _)) fun _ st₁ t₁ => ?_
  refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
  refine eqAt_bindL (eqAt_oodIntroP coin n _ _ (Nat.succ_ne_zero _) st₁ t₁ h1') fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_oodIntroP coin n _ _ (Nat.succ_ne_zero _) st₁ t₁ h1) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_queryP coin n _ ms _ cm0 (hq 0 (by omega)) st₁ t₁ h1 h2) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_lsilent_bind (lsilent_sendMsg _) fun _ st₁ t₁ => ?_
  refine eqAt_lsilent_bind (lsilent_powB _) fun _ st₁ t₁ => ?_
  refine eqAt_bindL (eqAt_lsampleP coin n st₁ t₁ h1) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
  have hb1 : l₁.size = L + sch.k0 + 11 := by omega
  have hb2 : L + sch.k0 + 11 ≤ n := by omega
  clear h1 h2 h1' h2'
  refine eqAt_bindL (eqAt_forInL _ _ (L + sch.k0 + 11) (fun i => 11 * i - 4 * (i + 1 - (sch.levels.size - 1)))
    (sch.levels.size - 1) (fun i hi s st t l hl hL => ?_) _ st₁ t₁ l₁ (by omega) (by omega))
    fun _ st₁ t₁ l₁ h1 h2 _ => ?_
  · have hg : 11 * i - 4 * (i + 1 - (sch.levels.size - 1)) = 11 * i := by omega
    rw [hg] at hl hL
    clear hg
    refine eqAt_bindL (eqAt_forInL _ _ (L + sch.k0 + 11 + 11 * i) (fun j => j) 4 (fun j hj f st t l hl hL => ?_) _ st t
      l (by omega) (by omega)) fun f st₁ t₁ l₁ h1 h2 _ => ?_
    · by_cases h4 : (j + 1 == 4) = true
      · rw [ite_eq_left h4]
        refine eqAt_bindL (eqAt_lsampleKP coin n st t hl) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
        refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind lsilent_get fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind lsilent_get fun _ st₁ t₁ => ?_
        refine eqAt_lsilent_bind (lsilent_sendMsg _) fun _ st₁ t₁ => ?_
        exact eqAt_purePL _ st₁ t₁ (by omega) (by omega) ⟨_, rfl⟩
      · rw [ite_eq_right h4]
        refine eqAt_bindL (eqAt_challengeP coin n false st t hl) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
        exact eqAt_purePL _ st₁ t₁ (by omega) (by omega) ⟨_, rfl⟩
    by_cases hlast : i + 1 = sch.levels.size - 1
    · rw [ite_eq_right (by simp [hlast])]
      refine eqAt_lsilent_bind (lsilent_forIn_array _ _ _ fun x s =>
        lsilent_bind' (lsilent_liftTM _) fun _ => lsilent_pure _) fun _ st₁ t₁ => ?_
      refine eqAt_bindL (eqAt_queryP coin n _ ms _ _ (hq _ (by omega)) st₁ t₁ h1 h2) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
      refine eqAt_lsilent_bind (lsilent_powB _) fun _ st₁ t₁ => ?_
      refine eqAt_bindL (eqAt_lsampleP coin n st₁ t₁ h1) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
      exact eqAt_purePL _ st₁ t₁ (by omega) (by omega) ⟨_, rfl⟩
    · rw [ite_eq_left (by simp [hlast])]
      refine eqAt_lsilent_bind (lsilent_quiet (commitP_quiet Sha512.hash _ _ _ _)) fun _ st₁ t₁ => ?_
      refine eqAt_lsilent_bind (lsilent_liftTM _) fun _ st₁ t₁ => ?_
      refine eqAt_bindL (eqAt_oodIntroP coin n _ _ (Nat.succ_ne_zero _) st₁ t₁ h1) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
      refine eqAt_bindL (eqAt_oodIntroP coin n _ _ (Nat.succ_ne_zero _) st₁ t₁ h1) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
      refine eqAt_bindL (eqAt_queryP coin n _ ms _ _ (hq _ (by omega)) st₁ t₁ h1 h2) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
      refine eqAt_lsilent_bind (lsilent_sendMsg _) fun _ st₁ t₁ => ?_
      refine eqAt_lsilent_bind (lsilent_powB _) fun _ st₁ t₁ => ?_
      refine eqAt_bindL (eqAt_lsampleP coin n st₁ t₁ h1) fun _ st₁ t₁ l₁ h1 h2 _ => ?_
      refine eqAt_lsilent_bind (lsilent_modify _) fun _ st₁ t₁ => ?_
      have hc : 11 * (i + 1) - 4 * (i + 1 + 1 - (sch.levels.size - 1)) = 11 * i + 11 := by
        clear * - hlast hi; omega
      rw [hc]
      exact eqAt_purePL _ st₁ t₁ (by omega) (by omega) ⟨_, rfl⟩
  have hc : 11 * (sch.levels.size - 1) - 4 * (sch.levels.size - 1 + 1 - (sch.levels.size - 1)) + sch.k0 + 11 =
      sch.k0 + 7 + 11 * (sch.levels.size - 1) := by
    clear * - hR; omega
  exact eqAt_pureL _ st₁ t₁ (by omega) (by omega)

theorem tsilent_get : TSilent (get : TMW PS Transcript) :=
  fun _ _ _ _ _ h => (Prod.mk.inj h).2.symm

theorem tsilent_liftV (v : V α) : TSilent (StateT.lift (monadLift v : PS α) : TMW PS α) := by
  intro t l a t' l' h
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
  obtain ⟨b, l₂, e3, e4⟩ := JsonRT.go_bind_ok h
  rw [JsonRT.monadLift_go] at e3
  obtain ⟨-, rfl⟩ := Prod.mk.inj e3
  exact (Prod.mk.inj e4).2.symm

/-- **The cut's inner proof at sources agreeing where sent**: `ρ`, `σ` after `1` send, `τ`'s commitment after `2`, `Y`
after `3`, `τ` and its salt after `4`; it sends `4` bodies. -/
theorem innerCut_eqAt (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128) (src src' : RepSrc)
    (cons : Array (String × Zk.Lin)) (t : Transcript) {l : Array Claims.Msg} {L : ℕ} (hl : l.size = L) (hL : L ≤ n)
    (hrs : L + 1 ≤ n → src.rhoIn = src'.rhoIn ∧ src.sigmaIn = src'.sigmaIn)
    (htc : L + 2 ≤ n → src.tauCom = src'.tauCom) (hy : L + 3 ≤ n → src.y = src'.y)
    (hT : L + 4 ≤ n → src.tau = src'.tau ∧ src.tauSalt = src'.tauSalt) :
    EqAt n (L + 4) (fun _ => True)
      (((innerCut (asksCut coin n) lay ic finalC src cons).run t).go l)
      (((innerCut (asksCut coin n) lay ic finalC src' cons).run t).go l) := by
  unfold innerCut
  dsimp only
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ hl) fun eps t₁ l₁ h1 h2 _ => ?_
  obtain ⟨er, es⟩ := hrs (by omega)
  rw [er, es]
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_bindT (eqAt_sampleVecP coin n (by simp [Zk.triple]) t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
  rw [htc (by omega)]
  refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun beta t₁ l₁ h1 h2 _ => ?_
  rw [hy (by omega)]
  by_cases hb : beta.isZero = true
  · rw [ite_eq_left hb, ite_eq_left hb]
    exact eqAt_throwT n _ t₁ l₁
  · rw [ite_eq_right hb, ite_eq_right hb]
    refine eqAt_tsilent_bind (tsilent_liftTM _) fun _ t₁ => ?_
    refine eqAt_bindT (eqAt_sampleVecP coin n (by simp [Zk.Q]) t₁ h1) fun _ t₁ l₁ h1 h2 _ => ?_
    obtain ⟨e1, e2⟩ := hT (by omega)
    rw [e1, e2]
    exact eqAt_pureT _ t₁ (by omega) (by omega)

end Phases

/-! ## A rep's reads, by the bodies sent before each -/

/-- A rep's pad layout. -/
abbrev layOf (st : Setup) (sch : Schedule) : Zk.PadLayout := Zk.PadLayout.new st.m st.kLog sch.k0

/-- The bodies a rep from `B` has sent when its Ligerito opening starts. -/
abbrev ligAt (B : ℕ) (st : Setup) (e : Array Claim) : ℕ :=
  B + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + (2 + e.size) + 1

/-- The bodies a rep from `B` has sent when its inner proof starts. -/
abbrev innerAt (B : ℕ) (st : Setup) (sch : Schedule) (e : Array Claim) : ℕ :=
  ligAt B st e + (sch.k0 + 7 + 11 * (sch.levels.size - 1))

/-- **Two sources agree on what a rep from body `B` reads**: each field the same wherever the rep reads it with at most
`n` bodies sent. -/
structure RepAgree (n B : ℕ) (st : Setup) (sch : Schedule) (e : Array Claim) (src src' : RepSrc) : Prop where
  ab : B + 2 ≤ n → ∀ i < 64, src.masked ((layOf st sch).r1ab i) = src'.masked ((layOf st sch).r1ab i) ∧
    src.masked ((layOf st sch).r1c i) = src'.masked ((layOf st sch).r1c i)
  ml : ∀ i < st.m - 6, B + 3 + i ≤ n → src.masked ((layOf st sch).ml i 0) = src'.masked ((layOf st sch).ml i 0) ∧
    src.masked ((layOf st sch).ml i 1) = src'.masked ((layOf st sch).ml i 1)
  f : B + 3 + (st.m - 6) ≤ n → src.masked (layOf st sch).fa = src'.masked (layOf st sch).fa ∧
    src.masked (layOf st sch).fb = src'.masked (layOf st sch).fb ∧ src.masked (layOf st sch).fc = src'.masked (layOf st sch).fc
  lc : ∀ j < st.fold.kLog - 6, B + 3 + (st.m - 6) + 2 + j ≤ n →
    src.masked ((layOf st sch).lc j 0) = src'.masked ((layOf st sch).lc j 0) ∧
    src.masked ((layOf st sch).lc j 1) = src'.masked ((layOf st sch).lc j 1)
  zp : B + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) ≤ n → ∀ i < 64,
    src.masked ((layOf st sch).zp i) = src'.masked ((layOf st sch).zp i)
  sv : ∀ k < 2 + e.size, B + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + k ≤ n → src.sv k = src'.sv k
  ood : ligAt B st e + 1 ≤ n → src.masked ((layOf st sch).lgOod 0) = src'.masked ((layOf st sch).lgOod 0)
  msg : ∀ j < sch.k0, ligAt B st e + 2 + j ≤ n → ∀ i < 4,
    src.masked ((layOf st sch).lgMsg j i) = src'.masked ((layOf st sch).lgMsg j i)
  E : ligAt B st e + 2 + sch.k0 ≤ n → ∀ a < 2, ∀ i < 2,
    src.masked ((layOf st sch).lgE a i) = src'.masked ((layOf st sch).lgE a i)
  y1 : ligAt B st e + sch.k0 + 4 ≤ n → src.y1 = src'.y1 ∧ src.ybar = src'.ybar ∧ src.hiSalt = src'.hiSalt
  rs : innerAt B st sch e + 1 ≤ n → src.rhoIn = src'.rhoIn ∧ src.sigmaIn = src'.sigmaIn
  tc : innerAt B st sch e + 2 ≤ n → src.tauCom = src'.tauCom
  y : innerAt B st sch e + 3 ≤ n → src.y = src'.y
  tau : innerAt B st sch e + 4 ≤ n → src.tau = src'.tau ∧ src.tauSalt = src'.tauSalt

/-- A rep's sends. -/
abbrev repSends (st : Setup) (sch : Schedule) (e : Array Claim) : ℕ := innerAt 0 st sch e + 4

/-- What a rep's walk needs of its table and schedule: the zerocheck's `m ≥ 14`, a non-empty opening point, `k_0 ≥ 1`, at
least two levels, and non-empty query sets. -/
structure SchOk (st : Setup) (sch : Schedule) : Prop where
  m : 14 ≤ st.m
  n : sch.n ≠ 0
  k0 : 1 ≤ sch.k0
  levels : 2 ≤ sch.levels.size
  q : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0

section Rep

variable (coin : ℕ → ℕ → F128) (n : ℕ)

/-- **A cut rep at sources agreeing where read**: the PIOP, the Ligerito opening and the inner proof in turn; it sends
`repSends` bodies. -/
theorem repCut_eqAt (st : Setup) (sch : Schedule) (hs : SchOk st sch) (rep : ℕ) (dom : ByteArray) (cm0 : Commit)
    (extras : Array Claim) (ic : InnerCommit) (src src' : RepSrc) (prior : Array (List F128)) (words : List ℕ)
    (t : Transcript) {l : Array Claims.Msg} {B : ℕ} (hl : l.size = B) (hB : B ≤ n)
    (hA : RepAgree n B st sch extras src src') :
    EqAt n (B + repSends st sch extras) (fun _ => True)
      (((repCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras ic src prior words).run t).go l)
      (((repCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras ic src' prior words).run t).go
        l) := by
  unfold repCut
  dsimp only
  refine eqAt_bindT (piopCut_eqAt coin n st hs.m (layOf st sch) src src' _ _ extras prior words t hl hB hA.ab hA.ml
    hA.f hA.lc hA.zp hA.sv) fun r t₁ l₁ h1 h2 _ => ?_
  obtain ⟨zp, lp, ss, T, sws⟩ := r
  refine eqAt_bindT (ligeritoCut_eqAt coin n st.merkle sch T cm0 _ (layOf st sch) src src' hs.n hs.k0 hs.levels hs.q t₁
    h1 h2 hA.ood hA.msg hA.E hA.y1) fun r t₁ l₁ h1 h2 _ => ?_
  obtain ⟨lg, zl, rec⟩ := r
  refine eqAt_tsilent_bind tsilent_get fun _ t₁ => ?_
  refine eqAt_tsilent_bind (tsilent_liftV _) fun rp t₁ => ?_
  obtain ⟨rp, _⟩ := rp
  refine eqAt_tsilent_bind (tsilent_liftV _) fun lgc t₁ => ?_
  refine eqAt_bindT (innerCut_eqAt coin n (layOf st sch) ic _ src src' _ t₁ h1 h2 hA.rs hA.tc hA.y hA.tau)
    fun _ t₁ l₁ h1 h2 _ => ?_
  simp only [repSends, innerAt, ligAt] at h1 h2 ⊢
  exact eqAt_pureT _ t₁ (by omega) (by omega)

end Rep

/-! ## The session -/

section Session

open With

theorem commit0Cut_quiet (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (src : TabSrc) :
    quietS.R (commit0Cut (Hashes.ofSha512 h : Hashes V) ms lv src) (commit0Cut (Hashes.ofSha512 h : Hashes PS) ms lv src) := by
  have ho := opsWith_sim quietS (quietS_hashes h) ms
  unfold commit0Cut
  sim_walk [ho.node]

theorem padsTreeCut_quiet (h : ByteArray → ByteArray) (N : ℕ) (src : RepSrc) :
    quietS.R (padsTreeCut (Hashes.ofSha512 h : Hashes V) N src) (padsTreeCut (Hashes.ofSha512 h : Hashes PS) N src) := by
  have hh := quietS_hashes h
  unfold padsTreeCut MerkleOps.hm96With
  sim_walk [leafWith_sim quietS hh]

theorem eqAt_quiet_bind {n M : ℕ} {Q : β → Prop} {v : V α} {x : PS α} (hq : quietS.R v x) {f g : α → PS β}
    {l : Array Claims.Msg} (hfg : ∀ a, v = .ok a → EqAt n M Q ((f a).go l) ((g a).go l)) :
    EqAt n M Q ((x >>= f).go l) ((x >>= g).go l) := by
  rw [go_bind, go_bind, hq l]
  cases v with
  | error e => exact ⟨rfl, fun a L h => by simp at h⟩
  | ok a => exact hfg a rfl

theorem eq_quiet_bind {v : V α} {x : PS α} (hq : quietS.R v x) {f g : α → PS β} {l : Array Claims.Msg}
    (hfg : ∀ a, v = .ok a → (f a).go l = (g a).go l) : (x >>= f).go l = (x >>= g).go l := by
  rw [go_bind, go_bind, hq l]
  cases v with
  | error e => rfl
  | ok a => exact hfg a rfl

theorem eq_of_eqAt_bind {n N : ℕ} {P : α → Prop} {x y : PS α} {f g : α → PS β} {l : Array Claims.Msg}
    (hxy : EqAt n N P (x.go l) (y.go l)) (hfg : ∀ a l', x.go l = (.ok a, l') → (f a).go l' = (g a).go l') :
    (x >>= f).go l = (y >>= g).go l := by
  rw [go_bind, go_bind, ← hxy.1]
  rcases hx : x.go l with ⟨r, l'⟩
  cases r with
  | error err => rfl
  | ok a => exact hfg a l' hx

theorem eqAt_purePS {P : α → Prop} (a : α) {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n)
    (hP : P a) : EqAt n N P ((pure a : PS α).go l) ((pure a : PS α).go l) := by
  refine eqAt_refl fun b L h => ?_
  rw [JsonRT.go_pure] at h
  obtain ⟨e, rfl⟩ := Prod.mk.inj h
  obtain rfl := Except.ok.inj e
  exact ⟨hl, hN, hP⟩

theorem link_go (coin : ℕ → ℕ → F128) {n : ℕ} (rootB publics : Array ByteArray) (hn : 1 ≤ n) :
    ((asksCut coin n).link rootB publics).go #[] =
      (.ok (coin 0), #[FlockDeploy.linkBody rootB.toList publics.toList]) := by
  show ((if n ≤ 0 then throw "past the openings" else pure (coin 0) : PS (ℕ → F128)).go _) = _
  rw [ite_eq_right (show ¬ n ≤ 0 by omega)]
  rfl

variable (coin : ℕ → ℕ → F128) (n : ℕ)

/-- The link points at the link's coin `coin 0`. -/
abbrev ptsOf (spec : SessionSpec) : Array (Array F128) :=
  (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => coin 0 (p * spec.mPts + j)

/-- **The session at sources that agree where read**: level 0's and the pads' commitments the same, and each rep stream
`k`'s source agreeing where read from its first body `G k` (`G 0 = 1`, the link's). -/
theorem sessionCut_eq (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (src src' : ℕ → TabSrc)
    (hn : 1 ≤ n)
    (hcm : ∀ ms lv i, commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes PS) ms lv (src i) =
      commit0Cut (Hashes.ofSha512 Sha512.hash) ms lv (src' i))
    (hic : ∀ N i r, padsTreeCut (Hashes.ofSha512 Sha512.hash : Hashes PS) N ((src i).rep r) =
      padsTreeCut (Hashes.ofSha512 Sha512.hash) N ((src' i).rep r))
    (G : ℕ → ℕ) (hG0 : G 0 = 1)
    (hR : ∀ schs, (tabs.mapM fun st => (monadLift (Zk.schedule st.m) : V (Schedule × ℕ))) = .ok schs →
      ∀ k (hk : k < spec.repStreams.size) i r w, spec.repStreams[k] = (i, r, w) → ∀ extras,
        tabs[i]!.extra (ptsOf coin spec) = .ok extras →
        SchOk tabs[i]! schs[i]!.1 ∧ G (k + 1) = G k + repSends tabs[i]! schs[i]!.1 extras ∧
          RepAgree n (G k) tabs[i]! schs[i]!.1 extras ((src i).rep r) ((src' i).rep r)) :
    (sessionCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) spec tabs words src).go #[] =
      (sessionCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) spec tabs words src').go #[] := by
  unfold sessionCut
  simp only [hcm, hic]
  have hq1 : quietS.R (tabs.mapM fun st => (monadLift (Zk.schedule st.m) : V (Schedule × ℕ)))
      (tabs.mapM fun st => (monadLift (Zk.schedule st.m) : PS (Schedule × ℕ))) := by
    sim_walk
  refine eq_quiet_bind hq1 fun schs hschs => ?_
  have hq2 : quietS.R ((Array.range tabs.size).mapM fun i =>
        commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes V) tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src' i))
      ((Array.range tabs.size).mapM fun i =>
        commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes PS) tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default)
          (src' i)) := by
    sim_walk [commit0Cut_quiet]
  refine eq_quiet_bind hq2 fun cms _ => ?_
  rw [go_bind, link_go coin _ _ hn, go_bind, link_go coin _ _ hn]
  dsimp only
  refine eq_of_eqAt_bind (eqAt_forIn_array (n := n) spec.repStreams _ _ G ?_ _ _ (by simp [hG0]) (by omega))
    fun _ _ _ => rfl
  intro k hk s l hl hL
  rcases hx : spec.repStreams[k] with ⟨i, r, w⟩
  dsimp only
  refine eqAt_quiet_bind (quietS.lift _) fun extras hex => ?_
  obtain ⟨hS, hG, hA⟩ := hR schs hschs k hk i r w hx extras hex
  refine eqAt_quiet_bind (padsTreeCut_quiet _ _ _) fun ic _ => ?_
  refine eqAt_bind (repCut_eqAt coin n _ _ hS r _ _ extras ic _ _ _ _ _ hl hL hA) fun p l' _ h1 h2 _ => ?_
  obtain ⟨p, t⟩ := p
  exact eqAt_purePS _ (by omega) (by omega) ⟨_, rfl⟩

end Session

end FlockComplete.CutEq


/-! ## The pads trees by tree position

ZKR1's `Round1Pads` (on `cursor/claims-zk-prover-faithful-c7a9`), here under `Cx`: a rep's pads leaf at tree position `p`
commits a column that reads no coin (`Cx.digOf_pads`), and a cut's pads leaf at `q` is the leaf at `q`
(`Cx.leafAtP_cutAt`). `treeEquiv` is irreducible here: unfolding it decides its `∃ e, …` over the tree's size.
-/

namespace FlockSoundness.Discharge.Composed.Cx

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL

local notation "𝔽" => FlockLevel3.GF128

variable (P : LigProver) {mPts : ℕ}

attribute [local irreducible] treeEquiv

/-- **Each rep's pads tree places every position at its index** at every coin vector of a `--zk` table whose pads code
has `Q ≤ n ≤ 2^63` positions: the opened ones at `padPos`, the others at `unPos`. -/
theorem coinedOf_sP_val (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (hQ : Zk.Q ≤ padNOf Tz st) (hN : padNOf Tz st ≤ 2 ^ 63) (r : Fin 2)
    (x : Fin (coinedOf P Tz st).z.nQ ⊕ Fin (coinedOf P Tz st).z.nPU) :
    ((((coinedOf P Tz st).T c).sP r) x).val =
      Sum.elim (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest))
        (fun u => unPos (padNOf Tz st)
          (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val) x := by
  have hL : max (min (padNOf Tz st) (2 ^ 63)) Zk.Q = padNOf Tz st := by
    rw [min_eq_left hN, max_eq_left hQ]
  have hQ' : Zk.Q ≤ max (min (padNOf Tz st) (2 ^ 63)) Zk.Q := le_max_right _ _
  have hsP : ((coinedOf P Tz st).T c).sP r = treeEquiv (a := (coinedOf P Tz st).z.nQ) (b := (coinedOf P Tz st).z.nPU)
      (Sum.elim (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest))
        (fun u => unPos (padNOf Tz st)
          (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val)) := rfl
  rw [hsP]
  exact treeEquiv_val_pos (t := Zk.Q) (N := padNOf Tz st) _ (padPos_injective _ hQ')
    (fun i => lt_of_lt_of_eq (padPos_lt _ hQ' i) hL) hQ x

/-- **A `--zk` table's exposed pad `h_fa`** (`coinedOf`'s `ia`), the same at every coin vector. -/
theorem coinedOf_ia (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ia = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩ :=
  rfl

/-- **A `--zk` table's exposed pad `h_fb`** (`coinedOf`'s `ib`), the same at every coin vector. -/
theorem coinedOf_ib (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ib = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ :=
  rfl

/-- **A rep's pads at its `h`** (`ZK.Table.hFull` at `ia`, `ib`): `h`'s entries, then `h_x`, `h_y`, `h_ia·h_ib`,
`h_x·h_y`. -/
noncomputable def padH {nE : ℕ} (ia ib : Fin nE) (h : (Fin nE → 𝔽) × 𝔽 × 𝔽) : ZK.PadIdx nE → 𝔽
  | .inl i => h.1 i
  | .inr j => ![h.2.1, h.2.2, h.1 ia * h.1 ib, h.2.1 * h.2.2] j

/-- **Rep `r`'s pads column at the code symbol of position `p`**: `(C_h, C_μ)` at `X = p + 1` (`innerPt`), pad `i`
(`padH` at `ia`, `ib`) at `X^(padIdx i)` and padding symbol `j` at `X^(len + j)`. -/
noncomputable def padColAt {Hi : Type} {z : ZkView.Sizes} (ia ib : Fin z.nE) (padLen : ℕ) (b : ZkView.Base z Hi)
    (r : Fin 2) (p : ℕ) : 𝔽 × 𝔽 :=
  (∑ i, innerPt p ^ padIdx i * padH ia ib (b.2.2.1 r) i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.1 r j,
    ∑ i, innerPt p ^ padIdx i * b.2.2.2.2.2.1 r i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.2.1 r j)

/-- **Rep `r`'s pads column at tree position `p` is `padColAt` at `X = p + 1`**, at every coin vector of a `--zk` table
whose pads code has `Q ≤ n ≤ 2^63` positions (`coinedOf_sP_val`): an opened column and a hidden one are both the code
there. -/
theorem coinedOf_colPAt (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (hQ : Zk.Q ≤ padNOf Tz st) (hN : padNOf Tz st ≤ 2 ^ 63)
    (w : Model.Witness (coinedOf P Tz st).S.m) (b : ZkView.Base (coinedOf P Tz st).z (coinedOf P Tz st).Hi) (r : Fin 2)
    (p : Fin ((coinedOf P Tz st).z.nQ + (coinedOf P Tz st).z.nPU)) :
    ((coinedOf P Tz st).T c).colPAt w b r p =
      padColAt (z := (coinedOf P Tz st).z)
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ (padLenOf Tz st) b r p.val := by
  have hs := coinedOf_sP_val P Tz st c hQ hN r
  have hH : ∀ (h : (Fin (coinedOf P Tz st).z.nE → 𝔽) × 𝔽 × 𝔽) i, ((coinedOf P Tz st).T c).tab.hFull h i =
      padH (nE := (coinedOf P Tz st).z.nE)
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ h i := fun h i => by
    rcases i with i | i <;> rfl
  have hH' := funext (hH (b.2.2.1 r))
  unfold ZkView.Coined.colPAt
  rcases h : (((coinedOf P Tz st).T c).sP r).symm p with k | u
  · have hk : padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k = p.val := by
      have e : ((coinedOf P Tz st).T c).sP r (.inl k) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs (.inl k)
      rw [e] at this
      exact this.symm
    have hC : ∀ i, ((coinedOf P Tz st).T c).Ch r k i = innerPt p.val ^ padIdx i := fun i => by
      show fOf ⟨(padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k + 1).toUInt64,
        0⟩ ^ padIdx i = _
      rw [hk]
      rfl
    have hP : ∀ j, ((coinedOf P Tz st).T c).Cp r k j = innerPt p.val ^ (padLenOf Tz st + j.val) := fun j => by
      show fOf ⟨(padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k + 1).toUInt64,
        0⟩ ^ (padLenOf Tz st + j.val) = _
      rw [hk]
      rfl
    refine Prod.ext ?_ ?_
    · show ∑ i, ((coinedOf P Tz st).T c).Ch r k i * ((coinedOf P Tz st).T c).tab.hFull (b.2.2.1 r) i +
          ∑ j, ((coinedOf P Tz st).T c).Cp r k j * b.2.2.2.1 r j = _
      simp only [hC, hP, hH']
      rfl
    · show ∑ i, ((coinedOf P Tz st).T c).Ch r k i * b.2.2.2.2.2.1 r i +
          ∑ j, ((coinedOf P Tz st).T c).Cp r k j * b.2.2.2.2.1 r j = _
      simp only [hC, hP]
      rfl
  · have hu : unPos (padNOf Tz st)
        (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val = p.val := by
      have e : ((coinedOf P Tz st).T c).sP r (.inr u) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs (.inr u)
      rw [e] at this
      exact this.symm
    have hC : ∀ i, ((coinedOf P Tz st).T c).ChU r u i = innerPt p.val ^ padIdx i := fun i => by
      show fOf ⟨(unPos (padNOf Tz st) (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q
        ((bodyOf Tz st).padQueries r c.rest)) u.val + 1).toUInt64, 0⟩ ^ padIdx i = _
      rw [hu]
      rfl
    have hP : ∀ j, ((coinedOf P Tz st).T c).CpU r u j = innerPt p.val ^ (padLenOf Tz st + j.val) := fun j => by
      show fOf ⟨(unPos (padNOf Tz st) (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q
        ((bodyOf Tz st).padQueries r c.rest)) u.val + 1).toUInt64, 0⟩ ^ (padLenOf Tz st + j.val) = _
      rw [hu]
      rfl
    refine Prod.ext ?_ ?_
    · show ∑ i, ((coinedOf P Tz st).T c).ChU r u i * ((coinedOf P Tz st).T c).tab.hFull (b.2.2.1 r) i +
          ∑ j, ((coinedOf P Tz st).T c).CpU r u j * b.2.2.2.1 r j = _
      simp only [hC, hP, hH']
      rfl
    · show ∑ i, ((coinedOf P Tz st).T c).ChU r u i * b.2.2.2.2.2.1 r i +
          ∑ j, ((coinedOf P Tz st).T c).CpU r u j * b.2.2.2.2.1 r j = _
      simp only [hC, hP]
      rfl

end FlockSoundness.Discharge.Composed.Cx

namespace Claims.ZKMal.Masked.Cx

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge

attribute [local irreducible] Composed.treeEquiv

/-- **A live table's pads code has `Q ≤ n ≤ 2^63` positions**: `n = innerN`, a power of two below `2^64` holding the
`len + K_PAD ≥ Q` symbols. -/
theorem padN_bounds (x : Msg) (j : Fin (jOf x)) :
    _root_.Flock.Zk.Q ≤ Composed.padNOf (zkTab x j).table (stAt x j) ∧
      Composed.padNOf (zkTab x j).table (stAt x j) ≤ 2 ^ 63 := by
  have hN : Composed.padNOf (zkTab x j).table (stAt x j) = ZkExec.innerN (ZkLink.layOf (stAt x j) (zkTab x j).sch) := rfl
  have hL := (zkTab x j).hL
  have h64 := (zkTab x j).hN
  rw [hN]
  refine ⟨le_trans (by simp [_root_.Flock.Zk.Q, _root_.Flock.Zk.K_PAD]) hL, ?_⟩
  have hp := FlockComplete.innerCols_pow ((ZkLink.layOf (stAt x j) (zkTab x j).sch).len + _root_.Flock.Zk.K_PAD)
  unfold ZkExec.innerN at h64 ⊢
  rw [hp] at h64 ⊢
  exact Nat.pow_le_pow_right (by norm_num) (by
    have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).1 h64
    omega)

/-- **Table `j`'s rep `r` pads column at tree position `p`** is `padColAt` at `X = p + 1`, at every coin vector. -/
theorem colPAt_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest)
    (w : Model.Witness (tabC P x j).S.m) (b : ZkView.Base (tabC P x j).z (tabC P x j).Hi) (r : Fin 2)
    (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    ((tabC P x j).T c).colPAt w b r p =
      Composed.Cx.padColAt (z := (tabC P x j).z)
        ⟨(_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).table.sch.k0).fa,
          Nat.lt_of_succ_lt (Composed.fa_lt (zkTab x j).table (stAt x j))⟩
        ⟨(_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).table.sch.k0).fb,
          Composed.fa_lt (zkTab x j).table (stAt x j)⟩
        (Composed.padLenOf (zkTab x j).table (stAt x j)) b r p.val :=
  Composed.Cx.coinedOf_colPAt P (zkTab x j).table (stAt x j) c (padN_bounds x j).1 (padN_bounds x j).2 w b r p

/-- **A cut's pads leaf at tree position `q` is the leaf at `q`**, whatever the coins' tree. -/
theorem padLeafCut_cutAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC P x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2) {q : ℕ}
    (hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU) :
    padLeafCut P x cz (cutAt (tabC P x) cz wt ω lv) j r q = lv ⟨j, .inr (.inl (r, ⟨q, hq⟩))⟩ := by
  unfold padLeafCut
  rw [dite_eq_left_of_eq_true (eq_true hq)]
  show (match (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with
    | .inl a => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inl a)))⟩
    | .inr u => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inr u)))⟩) = _
  rcases h : (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with a | u
  · have e : ((tabC P x j).T (cz j)).sP r (.inl a) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]
  · have e : ((tabC P x j).T (cz j)).sP r (.inr u) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]

/-- **A pads leaf's digest is the same at every coin vector** (`colPAt_tabC`). -/
theorem digOf_pads (P : Composed.LigProver) (x : Msg) (cz cz' : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi) (j : Fin (jOf x))
    (r : Fin 2) (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    digOf (tabC P x) cz wt ω ⟨j, .inr (.inl (r, p))⟩ = digOf (tabC P x) cz' wt ω ⟨j, .inr (.inl (r, p))⟩ := by
  unfold digOf
  simp only [ZkView.Coined.content]
  rw [colPAt_tabC, colPAt_tabC]

end Claims.ZKMal.Masked.Cx

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.ZK FlockSoundness.Discharge CoinLayer
open FlockComplete.CutEq

/-- **The cut at the coins `c`** on the simulator's leaves `H(dig ⊕ b, c)` (`RoundPlan.Causal`'s). -/
noncomputable abbrev cutC (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m)
    (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray) (c : Fin (refK x) → FlockDeploy.Coin) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Cut :=
  cutAt (tabC PE x) (czExe PE x c) wt ω fun l =>
    ZkView.leafH (digOf (tabC PE x) (czExe PE x c) wt ω l + (p l).1, (p l).2)

/-- What the session reads from `cutC`. -/
noncomputable abbrev srcC (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m)
    (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray) (c : Fin (refK x) → FlockDeploy.Coin) :
    ℕ → FlockComplete.TabSrc :=
  tabSrcCut PE x (czExe PE x c) (cutC x wt ω p c)

/-- **The sources agree where the session reads them**: at a live statement and coins agreeing before `n ≥ 1`, level 0's
and the pads' commitments are the same, and each rep stream's source agrees where read from its first body. -/
def SrcAgree : Prop :=
  ∀ x, Live x → ∀ wt ω p (c c' : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), (∀ j : Fin (refK x), j.val < n → c j = c' j) →
    1 ≤ n → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (∀ ms lv i, FlockComplete.commit0Cut hsE ms lv (srcC x wt ω p c i) =
        FlockComplete.commit0Cut hsE ms lv (srcC x wt ω p c' i)) ∧
      (∀ N i r, FlockComplete.padsTreeCut hsE N ((srcC x wt ω p c i).rep r) =
        FlockComplete.padsTreeCut hsE N ((srcC x wt ω p c' i).rep r)) ∧
      ∃ G : ℕ → ℕ, G 0 = 1 ∧ ∀ schs, ((FlockDeploy.tabsOf (stsOf x)).mapM fun st =>
          (monadLift (_root_.Flock.Zk.schedule st.m) : _root_.Flock.V (_root_.Flock.Schedule × ℕ))) = .ok schs →
        ∀ k (hk : k < spec.repStreams.size) i r w, spec.repStreams[k] = (i, r, w) → ∀ extras,
          (FlockDeploy.tabsOf (stsOf x))[i]!.extra (ptsOf (fun k i => slotAt c k i) spec) = .ok extras →
          SchOk (FlockDeploy.tabsOf (stsOf x))[i]! schs[i]!.1 ∧
            G (k + 1) = G k + repSends (FlockDeploy.tabsOf (stsOf x))[i]! schs[i]!.1 extras ∧
            RepAgree n (G k) (FlockDeploy.tabsOf (stsOf x))[i]! schs[i]!.1 extras ((srcC x wt ω p c i).rep r)
              ((srcC x wt ω p c' i).rep r)

theorem stsOf_of {x : Msg} {s spec sts} (h : FlockDeploy.setupOf true x = some (s, spec, sts)) : stsOf x = sts := by
  unfold stsOf
  rw [h]

theorem srcC_dead {x : Msg} (hx : ¬ Live x) (wt ω p) (c : Fin (refK x) → FlockDeploy.Coin) :
    srcC x wt ω p c = fun _ => default := by
  have hj : jOf x = 0 := by
    unfold jOf
    simp [hx]
  funext i
  unfold srcC tabSrcCut
  rw [dif_neg (by omega)]

/-- **The deployed plan is causal given the sources agree where read.** -/
theorem causalExe_of (hB : SrcAgree) : CausalExe := by
  intro x wt ω p c c' n hcc
  show encExe PE x c n (cutC x wt ω p c) = encExe PE x c' n (cutC x wt ω p c')
  unfold encExe
  by_cases hn : n = 0
  · simp only [hn, ↓reduceIte]
    exact enc0_causal PE x _ _ wt ω p
  rw [if_neg hn, if_neg hn]
  rcases hsx : FlockDeploy.setupOf true x with _ | ⟨s, spec, sts⟩
  · rfl
  dsimp only
  have hs : stsOf x = sts := stsOf_of hsx
  have hrun : runCut x c n spec (srcC x wt ω p c) = runCut x c' n spec (srcC x wt ω p c') := by
    unfold runCut
    rw [FlockComplete.asksCut_congr (coin := fun k i => slotAt c' k i) (coin' := fun k i => slotAt c k i)
      (fun k hk => funext fun e => (slotAt_congr hcc hk e).symm)]
    by_cases hx : Live x
    · obtain ⟨hcm, hic, G, hG0, hR⟩ := hB x hx wt ω p c c' n hcc (by omega) s spec (by rw [hs]; exact hsx)
      exact sessionCut_eq _ n spec _ _ _ _ (by omega) hcm hic G hG0 hR
    · rw [srcC_dead hx, srcC_dead hx]
  rw [hrun]

/-! ## The commitments read no coin -/

theorem commit0Cut_congr (ms : _root_.Flock.MerkleScheme) (lv : _root_.Flock.Level) {src src' : FlockComplete.TabSrc}
    (h0 : src.leaf0 = src'.leaf0) (h1 : src.row0 = src'.row0) (h2 : src.salt0 = src'.salt0) :
    FlockComplete.commit0Cut hsE ms lv src = FlockComplete.commit0Cut hsE ms lv src' := by
  unfold FlockComplete.commit0Cut
  rw [h0, h1, h2]

theorem padsTreeCut_congr (N : ℕ) {src src' : FlockComplete.RepSrc} (h0 : src.padLeaf = src'.padLeaf)
    (h1 : src.ch = src'.ch) (h2 : src.cm = src'.cm) (h3 : src.padSalt = src'.padSalt) :
    FlockComplete.padsTreeCut hsE N src = FlockComplete.padsTreeCut hsE N src' := by
  unfold FlockComplete.padsTreeCut
  rw [h0, h1, h2, h3]

section Proj

variable {x : Msg} (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest) (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut)

theorem tabSrcCut_live {i : ℕ} (h : i < jOf x) : tabSrcCut PE x cz cut i =
    { leaf0 := leafAt PE x cz cut ⟨i, h⟩, row0 := fun _ => #[], salt0 := fun _ => ByteArray.empty,
      rep := fun r => if hr : r < 2 then repSrcCut PE x cz cut ⟨i, h⟩ ⟨r, hr⟩ else default } := by
  unfold tabSrcCut
  rw [dif_pos h]

theorem tabSrcCut_leaf0 {i : ℕ} (h : i < jOf x) : (tabSrcCut PE x cz cut i).leaf0 = leafAt PE x cz cut ⟨i, h⟩ := by
  unfold tabSrcCut
  rw [dif_pos h]

theorem tabSrcCut_rep {i r : ℕ} (h : i < jOf x) (hr : r < 2) :
    (tabSrcCut PE x cz cut i).rep r = repSrcCut PE x cz cut ⟨i, h⟩ ⟨r, hr⟩ := by
  unfold tabSrcCut
  rw [dif_pos h]
  exact dif_pos hr

theorem tabSrcCut_rep_dead {i r : ℕ} (h : i < jOf x) (hr : ¬ r < 2) : (tabSrcCut PE x cz cut i).rep r = default := by
  unfold tabSrcCut
  rw [dif_pos h]
  exact dif_neg hr

theorem tabSrcCut_dead {i : ℕ} (h : ¬ i < jOf x) : tabSrcCut PE x cz cut i = default := by
  unfold tabSrcCut
  rw [dif_neg h]

theorem repSrcCut_padLeaf (j : Fin (jOf x)) (r : Fin 2) :
    (repSrcCut PE x cz cut j r).padLeaf = padLeafCut PE x cz cut j r := by
  simp only [repSrcCut]

end Proj

section Agree

variable (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m)
  (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
  (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)

theorem leaf0_agree (c c' : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) :
    leafAt PE x (czExe PE x c) (cutC x wt ω p c) j = leafAt PE x (czExe PE x c') (cutC x wt ω p c') j :=
  funext fun q => by
    by_cases hq : q < (tabC PE x j).z.nS + (tabC PE x j).z.nU
    · rw [leafAt_cutAt PE x _ wt ω _ j hq, leafAt_cutAt PE x _ wt ω _ j hq,
        digOf_inl PE x (czExe PE x c) (czExe PE x c') wt ω j]
    · simp only [leafAt, hq, ↓reduceDIte]

theorem padLeaf_agree (c c' : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) :
    padLeafCut PE x (czExe PE x c) (cutC x wt ω p c) j r = padLeafCut PE x (czExe PE x c') (cutC x wt ω p c') j r :=
  funext fun q => by
    by_cases hq : q < (tabC PE x j).z.nQ + (tabC PE x j).z.nPU
    · rw [Cx.padLeafCut_cutAt PE x _ wt ω _ j r hq, Cx.padLeafCut_cutAt PE x _ wt ω _ j r hq,
        Cx.digOf_pads PE x (czExe PE x c) (czExe PE x c') wt ω j r]
    · simp only [padLeafCut, hq, ↓reduceDIte]

theorem commits_agree (c c' : Fin (refK x) → FlockDeploy.Coin) :
    (∀ ms lv i, FlockComplete.commit0Cut hsE ms lv (srcC x wt ω p c i) =
        FlockComplete.commit0Cut hsE ms lv (srcC x wt ω p c' i)) ∧
      (∀ N i r, FlockComplete.padsTreeCut hsE N ((srcC x wt ω p c i).rep r) =
        FlockComplete.padsTreeCut hsE N ((srcC x wt ω p c' i).rep r)) := by
  refine ⟨fun ms lv i => ?_, fun N i r => ?_⟩
  · by_cases h : i < jOf x
    · rw [srcC, srcC, tabSrcCut_live _ _ h, tabSrcCut_live _ _ h]
      exact commit0Cut_congr ms lv (leaf0_agree x wt ω p c c' ⟨i, h⟩) rfl rfl
    · rw [srcC, srcC, tabSrcCut_dead _ _ h, tabSrcCut_dead _ _ h]
  · by_cases h : i < jOf x
    · by_cases hr : r < 2
      · rw [srcC, srcC, tabSrcCut_rep _ _ h hr, tabSrcCut_rep _ _ h hr]
        refine padsTreeCut_congr N ?_ ?_ ?_ ?_
        · rw [repSrcCut_padLeaf, repSrcCut_padLeaf]
          exact padLeaf_agree x wt ω p c c' ⟨i, h⟩ ⟨r, hr⟩
        all_goals simp only [repSrcCut]
      · rw [srcC, srcC, tabSrcCut_rep_dead _ _ h hr, tabSrcCut_rep_dead _ _ h hr]
    · rw [srcC, srcC, tabSrcCut_dead _ _ h, tabSrcCut_dead _ _ h]

end Agree

/-! ## The reps' bookkeeping -/

/-- **Each rep's sources agree where read**, at coins agreeing below `n`: table `j`'s rep `r`, from its first body
`repStartZk (2j + r)`, with its region claims. -/
def RepOk : Prop :=
  ∀ x, Live x → ∀ wt ω p (c c' : Fin (refK x) → FlockDeploy.Coin) (n : ℕ),
    (∀ j : Fin (refK x), j.val < n → c j = c' j) → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      ∀ (j : Fin (jOf x)) (r : Fin 2) (e : Array _root_.Flock.Claim), e.size = (stAt x j).regions.size * spec.points →
        RepAgree n (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val)) (_root_.Flock.Setup.ofCircuit (stAt x j))
          (zkTab x j).sch e (repSrcCut PE x (czExe PE x c) (cutC x wt ω p c) j r)
          (repSrcCut PE x (czExe PE x c') (cutC x wt ω p c') j r)

theorem schOk_of {st : _root_.Flock.Stmt} (y : ZkHidden.ZkTab st) : SchOk (_root_.Flock.Setup.ofCircuit st) y.sch := by
  obtain ⟨h1, -, h3, h4, -, ⟨hn, hL, -⟩, hlv, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ y.hz
  exact ⟨show 14 ≤ st.m by omega, by omega, h4, hL, fun j hj => (hlv j hj).q_ne_zero⟩

theorem srcAgree_of (h : RepOk) : SrcAgree := by
  intro x hx wt ω p c c' n hcc hn s spec hsx
  obtain ⟨hcm, hic⟩ := commits_agree x wt ω p c c'
  refine ⟨hcm, hic, FlockComplete.repStartZk spec (stsOf x), by simp [FlockComplete.repStartZk], ?_⟩
  intro schs hschs k hk i r w hkw extras hext
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  rw [FlockDeploy.repStreams_getElem spec hk] at hkw
  simp only [Prod.mk.injEq] at hkw
  obtain ⟨rfl, rfl, -⟩ := hkw
  rw [FlockDeploy.repStreams_size] at hk
  have hj : k / 2 < jOf x := by unfold jOf; rw [if_pos hx]; omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hst : (stsOf x)[k / 2]! = stAt x ⟨k / 2, hj⟩ := getElem!_pos (stsOf x) (k / 2) (lt_of_jOf ⟨k / 2, hj⟩)
  have htab : (FlockDeploy.tabsOf (stsOf x))[k / 2]! = _root_.Flock.Setup.ofCircuit (stAt x ⟨k / 2, hj⟩) := by
    rw [← hst]
    unfold FlockDeploy.tabsOf
    rw [getElem!_pos _ (k / 2) (by rw [Array.size_map]; exact lt_of_jOf ⟨k / 2, hj⟩),
      getElem!_pos _ (k / 2) (lt_of_jOf ⟨k / 2, hj⟩), Array.getElem_map]
  have hsch : schs[k / 2]!.1 = (zkTab x ⟨k / 2, hj⟩).sch := by
    obtain ⟨hsz, hget⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hschs
    have hlt : k / 2 < (FlockDeploy.tabsOf (stsOf x)).size := by
      rw [FlockDeploy.tabsOf, Array.size_map]; exact lt_of_jOf ⟨k / 2, hj⟩
    have h1 := hget (k / 2) hlt (hsz ▸ hlt)
    rw [← getElem!_pos (FlockDeploy.tabsOf (stsOf x)) (k / 2) hlt, htab] at h1
    have h2 : (_root_.Flock.Zk.schedule (stAt x ⟨k / 2, hj⟩).m : _root_.Flock.V _) = .ok schs[k / 2] := h1
    rw [(zkTab x ⟨k / 2, hj⟩).hz] at h2
    rw [getElem!_pos schs (k / 2) (hsz ▸ hlt), ← Except.ok.inj h2]
  rw [htab] at hext
  have hes : extras.size = (stAt x ⟨k / 2, hj⟩).regions.size * spec.points := by
    rw [FlockDeploy.extraClaims_size (st := stAt x ⟨k / 2, hj⟩) hext]
    simp
  rw [htab, hsch, srcC, srcC, tabSrcCut_rep _ _ hj hr, tabSrcCut_rep _ _ hj hr]
  refine ⟨schOk_of _, ?_, ?_⟩
  · have hk6 := (zkTab x ⟨k / 2, hj⟩).wf.1.hk6
    have hm := (FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x ⟨k / 2, hj⟩).hz).1
    rw [FlockComplete.ZkTables.repStart_succ, hst]
    unfold FlockDeploy.repRounds
    simp only [↓reduceIte, (zkTab x ⟨k / 2, hj⟩).hz]
    simp only [repSends, innerAt]
    show _ = _ + (3 + ((stAt x ⟨k / 2, hj⟩).m - 6) + (3 + ((stAt x ⟨k / 2, hj⟩).c.kLog - 6)) + _ + 1 + _ + 4)
    rw [hes.trans (Nat.mul_comm _ _)]
    omega
  · have := h x hx wt ω p c c' n hcc s spec hsx ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ extras hes
    rwa [show 2 * (⟨k / 2, hj⟩ : Fin (jOf x)).val + (⟨k % 2, hr⟩ : Fin 2).val = k by simp only; omega] at this

theorem causalExe_of_rep (h : RepOk) : CausalExe := causalExe_of (srcAgree_of h)

end Claims.ZKMal.Masked
