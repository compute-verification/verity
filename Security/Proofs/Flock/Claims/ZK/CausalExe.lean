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
(`Cx.padLeafCut_cutAt`). `treeEquiv` is irreducible here: unfolding it decides its `∃ e, …` over the tree's size.
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

/-! ## The honest clear values read their coins in order -/

namespace FlockSoundness.Discharge.Composed.Cx

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem foldN_congr {ρ ρ' : ℕ → 𝔽} (u : ℕ → 𝔽) :
    ∀ {j : ℕ}, (∀ i < j, ρ i = ρ' i) → Zerocheck.foldN ρ j u = Zerocheck.foldN ρ' j u
  | 0, _ => rfl
  | j + 1, h => by
    simp only [Zerocheck.foldN]
    rw [foldN_congr u fun i hi => h i (by omega), h j (by omega)]

theorem ofFn_get!_congr {α : Type} [Inhabited α] {n : ℕ} {f f' : Fin n → α} {i : ℕ}
    (h : ∀ a : Fin n, a.val < i → f a = f' a) : ∀ i' < i, (Array.ofFn f)[i']! = (Array.ofFn f')[i']! := by
  intro i' hi'
  by_cases hn : i' < n
  · rw [getElem!_pos _ i' (by simpa using hn), getElem!_pos _ i' (by simpa using hn), Array.getElem_ofFn,
      Array.getElem_ofFn]
    exact h ⟨i', hn⟩ hi'
  · rw [getElem!_neg _ i' (by simpa using hn), getElem!_neg _ i' (by simpa using hn)]

theorem range_map_get! {α : Type} [Inhabited α] (n i : ℕ) {g g' : ℕ → α} (h : i < n → g i = g' i) :
    ((Array.range n).map g)[i]! = ((Array.range n).map g')[i]! := by
  by_cases hi : i < n
  · rw [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi)]
    simp [h hi]
  · rw [getElem!_neg _ i (by simpa using hi), getElem!_neg _ i (by simpa using hi)]

section Honest

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c c' : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m) {L : ℕ} (R : Fin 2 → Fin L → 𝔽)

theorem zcH_r1 (hO : c.rest.rOuter r = c'.rest.rOuter r) :
    (zcH Tz st c r zw).round1ab = (zcH Tz st c' r zw).round1ab ∧
      (zcH Tz st c r zw).round1c = (zcH Tz st c' r zw).round1c := by
  simp only [zcH, Zerocheck.honest, rF, hO, and_self]

theorem zcH_round (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r) {i : ℕ}
    (hρ : ∀ a : Fin (st.m - 6), a.val < i → c.rest.rho r a = c'.rest.rho r a) :
    (zcH Tz st c r zw).rounds[i]! = (zcH Tz st c' r zw).rounds[i]! := by
  simp only [zcH, Zerocheck.honest, rF, zcF, hO, hz]
  refine range_map_get! _ _ fun _ => ?_
  have h := ofFn_get!_congr hρ
  have hc : ∀ i' < i, Zerocheck.coin (rhoF Tz st c r) i' = Zerocheck.coin (rhoF Tz st c' r) i' :=
    fun i' hi' => congrArg FlockLevel3.GF128.of (h i' hi')
  simp only [Zerocheck.roundMsg, foldN_congr _ hc]

theorem zcH_final (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r)
    (hρ : c.rest.rho r = c'.rest.rho r) :
    (zcH Tz st c r zw).finalA = (zcH Tz st c' r zw).finalA ∧ (zcH Tz st c r zw).finalB = (zcH Tz st c' r zw).finalB ∧
      (zcH Tz st c r zw).finalC = (zcH Tz st c' r zw).finalC := by
  simp only [zcH, Zerocheck.honest, rF, zcF, rhoF, hO, hz, hρ, and_self]

theorem take_ofFn_congr {n : ℕ} {f f' : Fin n → F128} (g : F128 → 𝔽) {j : ℕ}
    (h : ∀ a : Fin n, a.val < j → f a = f' a) :
    ((Array.ofFn f).toList.map g).take j = ((Array.ofFn f').toList.map g).take j := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_take, List.getElem_map, Array.getElem_toList, Array.getElem_ofFn]
  simp only [List.length_take, List.length_map, Array.length_toList, Array.size_ofFn] at h1
  rw [h ⟨i, by omega⟩ (by simp only; omega)]

theorem lay_fa : (layOf Tz st).fa = 128 + 2 * (st.m - 6) := rfl
theorem lay_fb : (layOf Tz st).fb = 129 + 2 * (st.m - 6) := by
  show 128 + 2 * (st.m - 6) + 1 = _
  omega
theorem lay_fc : (layOf Tz st).fc = 130 + 2 * (st.m - 6) := by
  show 128 + 2 * (st.m - 6) + 2 = _
  omega
theorem lay_zp (i : ℕ) : (layOf Tz st).zp i = 131 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + i := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + i = _
  omega
theorem lay_lgMsg (j t : ℕ) :
    (layOf Tz st).lgMsg j t = 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * j + t := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + 64 + 1 + 4 * j + t = _
  omega
theorem lay_lgE (i t : ℕ) :
    (layOf Tz st).lgE i t = 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 + 2 * i + t := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + 64 + 1 + 4 * Tz.sch.k0 + 2 * i + t = _
  omega
theorem lay_hab : (layOf Tz st).hab = 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 := by
  show 128 + 2 * (st.m - 6) + 2 + 1 + 2 * (st.c.kLog - 6) + 0 + 64 + 1 + 4 * Tz.sch.k0 + 4 = _
  omega

theorem honestAt_r1 (hO : c.rest.rOuter r = c'.rest.rOuter r) {k : ℕ} (hk : k < 128) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  obtain ⟨h1, h2⟩ := zcH_r1 Tz st c c' r zw hO
  unfold honestAt
  by_cases h64 : k < 64
  · simp only [ite_eq_left h64, h1]
  · simp only [ite_eq_right h64, ite_eq_left hk, h2]

theorem honestAt_ml (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r) {i : ℕ}
    (hρ : ∀ a : Fin (st.m - 6), a.val < i → c.rest.rho r a = c'.rest.rho r a) (hi : i < st.m - 6) {k : ℕ}
    (hk1 : 128 + 2 * i ≤ k) (hk2 : k < 130 + 2 * i) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h := zcH_round Tz st c c' r zw hO hz hρ
  have e : (k - 128) / 2 = i := by omega
  have := lay_fa Tz st
  unfold honestAt
  simp only [ite_eq_right (show ¬ k < 64 by omega), ite_eq_right (show ¬ k < 128 by omega),
    ite_eq_left (show k < (layOf Tz st).fa by omega), e, h]

theorem honestAt_f (hO : c.rest.rOuter r = c'.rest.rOuter r) (hz : c.rest.zc r = c'.rest.zc r)
    (hρ : c.rest.rho r = c'.rest.rho r) {k : ℕ} (hk1 : 128 + 2 * (st.m - 6) ≤ k) (hk2 : k < 131 + 2 * (st.m - 6)) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  obtain ⟨h1, h2, h3⟩ := zcH_final Tz st c c' r zw hO hz hρ
  have := lay_fa Tz st
  have := lay_fb Tz st
  have := lay_fc Tz st
  unfold honestAt
  simp only [ite_eq_right (show ¬ k < 64 by omega), ite_eq_right (show ¬ k < 128 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).fa by omega)]
  by_cases ha : k = (layOf Tz st).fa
  · simp only [ite_eq_left ha, h1]
  by_cases hb : k = (layOf Tz st).fb
  · simp only [ite_eq_right ha, ite_eq_left hb, h2]
  simp only [ite_eq_right ha, ite_eq_right hb, ite_eq_left (show k = (layOf Tz st).fc by omega), h3]

/-- Past the zerocheck: the lincheck's and later regions. -/
theorem honestAt_past (k : ℕ) (hk : 131 + 2 * (st.m - 6) ≤ k) :
    honestAt Tz st c r zw R k =
      let lay := layOf Tz st
      if k < lay.zp 0 then
        (if (k - lay.fc - 1) % 2 = 0 then (lcH Tz st c r zw ((k - lay.fc - 1) / 2)).1
          else (lcH Tz st c r zw ((k - lay.fc - 1) / 2)).2)
      else if k < lay.zp 64 then zpH Tz st c r zw (k - lay.zp 0)
      else if k < lay.lgMsg 0 0 then
        ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), eqAt (oodPoint Tz st c r (k - lay.lgOod 0)) y * wordOf zw y
      else if k < lay.lgE 0 0 then
        let j := (k - lay.lgMsg 0 0) / 4
        let t := (k - lay.lgMsg 0 0) % 4
        let m := laneMsg Tz st c r zw j
        let x := if t < 2 then m.1 else m.2
        if t % 2 = 0 then (coords execArith x).1 else (coords execArith x).2
      else if k < lay.hab then
        let i := (k - lay.lgE 0 0) / 2
        let e := ∑ col : Fin L, execArith.embed (seqOf R i col) * bFold Tz st c r col.val
        if (k - lay.lgE 0 0) % 2 = 0 then (coords execArith e).1 else (coords execArith e).2
      else 0 := by
  have := lay_fa Tz st
  have := lay_fb Tz st
  have := lay_fc Tz st
  unfold honestAt
  simp only [ite_eq_right (show ¬ k < 64 by omega), ite_eq_right (show ¬ k < 128 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).fa by omega), ite_eq_right (show ¬ k = (layOf Tz st).fa by omega),
    ite_eq_right (show ¬ k = (layOf Tz st).fb by omega), ite_eq_right (show ¬ k = (layOf Tz st).fc by omega)]

theorem honestAt_lc (hz : c.rest.zc r = c'.rest.zc r) (ha : c.rest.lcAlpha r = c'.rest.lcAlpha r)
    (hb : c.rest.lcBeta r = c'.rest.lcBeta r) (hρ : c.rest.rho r = c'.rest.rho r) {j : ℕ}
    (ht : ∀ a : Fin (st.c.kLog - 6), a.val < j → c.rest.ts r a = c'.rest.ts r a) (hj : j < st.c.kLog - 6) {k : ℕ}
    (hk1 : 131 + 2 * (st.m - 6) + 2 * j ≤ k) (hk2 : k < 133 + 2 * (st.m - 6) + 2 * j) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h : lcH Tz st c r zw j = lcH Tz st c' r zw j := by
    simp only [lcH, Lincheck.lcMsg, zcF, rhoF, hz, ha, hb, hρ, tsF, take_ofFn_congr fOf ht]
  have := lay_fc Tz st
  have := lay_zp Tz st 0
  have e : (k - (layOf Tz st).fc - 1) / 2 = j := by omega
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_left (show k < (layOf Tz st).zp 0 by omega), e, h]

theorem honestAt_zp (hρ : c.rest.rho r = c'.rest.rho r) (ht : c.rest.ts r = c'.rest.ts r) {k : ℕ}
    (hk1 : 131 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) ≤ k) (hk2 : k < 195 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h : zpH Tz st c r zw = zpH Tz st c' r zw := funext fun i => by simp only [zpH, tsF, rhoF, hρ, ht]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_left (show k < (layOf Tz st).zp 64 by omega), h]

theorem honestAt_ood (hp : c.rest.oodPt r = c'.rest.oodPt r) {k : ℕ}
    (hk : k = 195 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have h : ∀ i, oodPoint Tz st c r i = oodPoint Tz st c' r i := fun i => by unfold oodPoint; rw [hp]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_left (show k < (layOf Tz st).lgMsg 0 0 by omega), h]

/-- What level 0's basis reads: the ring switch's coins, the claims' points and the out-of-domain sample. -/
structure B0Agree : Prop where
  link : c.rest.link = c'.rest.link
  rOuter : c.rest.rOuter r = c'.rest.rOuter r
  rho : c.rest.rho r = c'.rest.rho r
  ts : c.rest.ts r = c'.rest.ts r
  rpp : c.rest.rpp r = c'.rest.rpp r
  rsGamma : c.rest.rsGamma r = c'.rest.rsGamma r
  oodPt : c.rest.oodPt r = c'.rest.oodPt r
  ood : c.rest.ood r = c'.rest.ood r

theorem laneB0_congr (h : B0Agree Tz st c c' r) : laneB0 Tz st c r = laneB0 Tz st c' r := by
  have hx : ∀ k, claimX Tz st c r k = claimX Tz st c' r k := fun k => by
    simp only [claimX, abX, lcT, zcRho, zcReq, h.ts, h.rho, h.rOuter, h.link]
  have hp : ∀ i, oodPoint Tz st c r i = oodPoint Tz st c' r i := fun i => by unfold oodPoint; rw [h.oodPt]
  funext y
  simp only [laneB0, openB, hx, hp, h.rpp, h.rsGamma, h.ood]

theorem laneF_congr {j : ℕ} (hl : ∀ i < j, laneCh Tz st c r i = laneCh Tz st c' r i) :
    laneF Tz st c r zw j = laneF Tz st c' r zw j := by
  induction j with
  | zero => rfl
  | succ j ih => simp only [laneF, ih fun i hi => hl i (by omega), hl j (by omega)]

theorem laneB_congr (h0 : laneB0 Tz st c r = laneB0 Tz st c' r) {j : ℕ}
    (hl : ∀ i < j, laneCh Tz st c r i = laneCh Tz st c' r i) : laneB Tz st c r j = laneB Tz st c' r j := by
  induction j with
  | zero => exact h0
  | succ j ih => simp only [laneB, ih fun i hi => hl i (by omega), hl j (by omega)]

theorem laneCh_congr {j : ℕ} (hl : ∀ a : Fin Tz.sch.k0, a.val < j → c.rest.lane r a = c'.rest.lane r a) :
    ∀ i < j, laneCh Tz st c r i = laneCh Tz st c' r i := fun i hi => by
  unfold laneCh seqOf
  split_ifs with h
  · rw [hl ⟨i, h⟩ hi]
  · rfl

theorem honestAt_msg (h0 : B0Agree Tz st c c' r) {j : ℕ}
    (hl : ∀ a : Fin Tz.sch.k0, a.val < j → c.rest.lane r a = c'.rest.lane r a) (hj : j < Tz.sch.k0) {k : ℕ}
    (hk1 : 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * j ≤ k)
    (hk2 : k < 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * j) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have hc := laneCh_congr Tz st c c' r hl
  have h : laneMsg Tz st c r zw j = laneMsg Tz st c' r zw j := by
    simp only [laneMsg, laneF_congr Tz st c c' r zw hc, laneB_congr Tz st c c' r (laneB0_congr Tz st c c' r h0) hc]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  have := lay_lgE Tz st 0 0
  have e : (k - (layOf Tz st).lgMsg 0 0) / 4 = j := by omega
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).lgMsg 0 0 by omega), ite_eq_left (show k < (layOf Tz st).lgE 0 0 by omega), e, h]

theorem honestAt_E (h0 : B0Agree Tz st c c' r) (hl : c.rest.lane r = c'.rest.lane r) {k : ℕ}
    (hk1 : 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 ≤ k)
    (hk2 : k < 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  have hc := laneCh_congr Tz st c c' r (j := Tz.sch.k0) fun a _ => congrFun hl a
  have h : bFold Tz st c r = bFold Tz st c' r := by
    funext col
    simp only [bFold, laneB_congr Tz st c c' r (laneB0_congr Tz st c c' r h0) hc]
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  have := lay_lgE Tz st 0 0
  have := lay_hab Tz st
  rw [honestAt_past Tz st c r zw R k (by omega), honestAt_past Tz st c' r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).lgMsg 0 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).lgE 0 0 by omega),
    ite_eq_left (show k < (layOf Tz st).hab by omega), h]

theorem bFold_congr (h0 : B0Agree Tz st c c' r) (hl : c.rest.lane r = c'.rest.lane r) :
    bFold Tz st c r = bFold Tz st c' r := by
  have hc := laneCh_congr Tz st c c' r (j := Tz.sch.k0) fun a _ => congrFun hl a
  funext col
  simp only [bFold, laneB_congr Tz st c c' r (laneB0_congr Tz st c c' r h0) hc]

theorem honestAt_end {k : ℕ} (hk : 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0 ≤ k) :
    honestAt Tz st c r zw R k = 0 := by
  have := lay_zp Tz st 0
  have := lay_zp Tz st 64
  have := lay_lgMsg Tz st 0 0
  have := lay_lgE Tz st 0 0
  have := lay_hab Tz st
  rw [honestAt_past Tz st c r zw R k (by omega)]
  simp only [ite_eq_right (show ¬ k < (layOf Tz st).zp 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).zp 64 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).lgMsg 0 0 by omega), ite_eq_right (show ¬ k < (layOf Tz st).lgE 0 0 by omega),
    ite_eq_right (show ¬ k < (layOf Tz st).hab by omega)]

/-- **Every masked value of rep `r` reads only the coins of its PIOP and level 0's lane phase.** -/
theorem honestAt_all (h0 : B0Agree Tz st c c' r) (hz : c.rest.zc r = c'.rest.zc r)
    (ha : c.rest.lcAlpha r = c'.rest.lcAlpha r) (hb : c.rest.lcBeta r = c'.rest.lcBeta r)
    (hl : c.rest.lane r = c'.rest.lane r) (k : ℕ) :
    honestAt Tz st c r zw R k = honestAt Tz st c' r zw R k := by
  by_cases h1 : k < 128
  · exact honestAt_r1 Tz st c c' r zw R h0.rOuter h1
  by_cases h2 : k < 128 + 2 * (st.m - 6)
  · exact honestAt_ml Tz st c c' r zw R h0.rOuter hz (i := (k - 128) / 2) (fun a _ => congrFun h0.rho a)
      (by omega) (by omega) (by omega)
  by_cases h3 : k < 131 + 2 * (st.m - 6)
  · exact honestAt_f Tz st c c' r zw R h0.rOuter hz h0.rho (by omega) h3
  by_cases h4 : k < 131 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)
  · exact honestAt_lc Tz st c c' r zw R hz ha hb h0.rho (j := (k - (131 + 2 * (st.m - 6))) / 2)
      (fun a _ => congrFun h0.ts a) (by omega) (by omega) (by omega)
  by_cases h5 : k < 195 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)
  · exact honestAt_zp Tz st c c' r zw R h0.rho h0.ts (by omega) h5
  by_cases h6 : k < 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6)
  · exact honestAt_ood Tz st c c' r zw R h0.oodPt (by omega)
  by_cases h7 : k < 196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0
  · exact honestAt_msg Tz st c c' r zw R h0 (j := (k - (196 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6))) / 4)
      (fun a _ => congrFun hl a) (by omega) (by omega) (by omega)
  by_cases h8 : k < 200 + 2 * (st.m - 6) + 2 * (st.c.kLog - 6) + 4 * Tz.sch.k0
  · exact honestAt_E Tz st c c' r zw R h0 hl (by omega) h8
  · exact (honestAt_end Tz st c r zw R (by omega)).trans (honestAt_end Tz st c' r zw R (by omega)).symm

/-- What rep `r`'s inner proof reads of its coins: level 0's basis, the PIOP's, the lanes, the restart's pair, `ε`
and the `γ`s. -/
structure InAgree extends B0Agree Tz st c c' r : Prop where
  zc : c.rest.zc r = c'.rest.zc r
  lcAlpha : c.rest.lcAlpha r = c'.rest.lcAlpha r
  lcBeta : c.rest.lcBeta r = c'.rest.lcBeta r
  rPrime : c.rest.rPrime r = c'.rest.rPrime r
  lane : c.rest.lane r = c'.rest.lane r
  rhoP : c.rho r = c'.rho r
  zeta : c.rest.zeta r = c'.rest.zeta r
  gammas : c.rest.gammas r = c'.rest.gammas r

/-- **Rep `r`'s ten constraints read its coins and its two masked claims' `ŝ`.** -/
theorem rowsOf_congr {nE nC : ℕ} (v : Fin nE → 𝔽) (sh sh' : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (hA : InAgree Tz st c c' r) (hs0 : sHatAt sh (2 * r.val) = sHatAt sh' (2 * r.val))
    (hs1 : sHatAt sh (2 * r.val + 1) = sHatAt sh' (2 * r.val + 1)) :
    rowsOf Tz st c r v sh T' ρσ = rowsOf Tz st c' r v sh' T' ρσ := by
  have e0 : shArr sh (2 * r.val) = shArr sh' (2 * r.val) := by unfold shArr; rw [hs0]
  have e1 : shArr sh (2 * r.val + 1) = shArr sh' (2 * r.val + 1) := by unfold shArr; rw [hs1]
  have hsw : ∀ k, swAt Tz st c r sh k = swAt Tz st c' r sh' k := fun k => by
    unfold swAt
    rw [hs0, hs1, hA.link]
  have hT : openT Tz st c r sh = openT Tz st c' r sh' := by
    unfold openT
    simp only [hsw, hA.rsGamma, hA.rpp]
  have hrun : ∀ j, laneRun Tz st c r v sh j = laneRun Tz st c' r v sh' j := by
    intro j
    induction j with
    | zero => simp only [laneRun, hT, hA.ood]
    | succ j ih => simp only [laneRun, ih, laneCh, hA.lane]
  simp only [rowsOf, rowCEval, rowAB, rowLC, rowRsAb, rowRsC, laneRow, abClaim, cClaim, combvOf, zcF, rF, rhoF, tsF,
    innerEps, rhoK, e0, e1, hrun, hA.zc, hA.rOuter, hA.rho, hA.lcAlpha, hA.lcBeta, hA.ts, hA.rPrime, hA.rhoP, hA.zeta]

/-- **The inner proof's batched coefficients read rep `r`'s masked values, its two masked claims' `ŝ`, the restart's
`T'`, the triple's `(ρ, σ)` and its coins.** -/
theorem lamOf_congr {z : Sizes} (pre pre' : PreOf z) (hA : InAgree Tz st c c' r) (hv : pre.2.1 r = pre'.2.1 r)
    (hs0 : sHatAt pre.1 (2 * r.val) = sHatAt pre'.1 (2 * r.val))
    (hs1 : sHatAt pre.1 (2 * r.val + 1) = sHatAt pre'.1 (2 * r.val + 1))
    (hT : (pre.2.2.2.2.2 r).1 = (pre'.2.2.2.2.2 r).1) (hρσ : pre.2.2.2.2.1 r = pre'.2.2.2.2.1 r) :
    lamOf Tz st z c r pre = lamOf Tz st z c' r pre' := by
  funext i
  unfold lamOf
  have hg : innerGamma Tz st c r = innerGamma Tz st c' r := funext fun k => by simp only [innerGamma, hA.gammas]
  rw [hv, hT, hρσ, rowsOf_congr Tz st c c' r _ _ _ _ _ hA hs0 hs1, hg]

end Honest

section Fields

variable (P : LigProver) {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st))

theorem coinedOf_nu (w q v) : ((coinedOf P Tz st).T c).tab.nu w q v = ∑ y : Fin (2 ^ (Tz.S.m - 7)),
    if y.val ∈ maskWordsOf st then 0
    else fOf (eqWord (pointsOf c.rest q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val) := rfl

theorem coinedOf_eqw (q y) :
    ((coinedOf P Tz st).T c).tab.eqw q y = fOf (eqWord (pointsOf c.rest q) ((maskWordsOf st).getD y.val 0)) := rfl

theorem coinedOf_sReg (w q v) : ((coinedOf P Tz st).T c).tab.sReg w q v = ∑ y : Fin (2 ^ (Tz.S.m - 7)),
    fOf (eqWord (regionPtOf st c.rest.link q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val) := rfl

theorem coinedOf_rho (r i) : ((coinedOf P Tz st).T c).tab.ρ r i =
    if i = 0 then FlockLevel3.GF256.of (c.rho r).1 else FlockLevel3.GF256.of (c.rho r).2 := rfl

theorem coinedOf_beta (r) : ((coinedOf P Tz st).T c).tab.β r = fOf (c.beta r) := rfl

theorem coinedOf_rep (k) : ((coinedOf P Tz st).T c).rep k = if k.val < 2 then 0 else 1 := rfl

theorem coinedOf_embed : ((coinedOf P Tz st).T c).tab.embed = FlockLevel3.embed := rfl

theorem coined_embed : ((coinedOf P Tz st).T c).embed = FlockLevel3.embed := rfl

variable (c' : CoinsZK (RestOf Tz st))

theorem claimsOf_congr (r : Fin 2) (h1 : c.rest.zc r = c'.rest.zc r) (h2 : c.rest.rho r = c'.rest.rho r)
    (h3 : c.rest.rOuter r = c'.rest.rOuter r) (h4 : c.rest.rPrime r = c'.rest.rPrime r)
    (h5 : c.rest.ts r = c'.rest.ts r) : claimsOf c.rest r = claimsOf c'.rest r := by
  simp only [claimsOf, h1, h2, h3, h4, h5]

theorem coined_rankOn {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T T' : ZkView.Coined K W Hi VL z)
    (Pr : Fin z.nC → Prop) (h : ∀ k, Pr k → T.eqw k = T'.eqw k) : T.tab.RankOn Pr ↔ T'.tab.RankOn Pr := by
  unfold ZK.Table.RankOn
  have e : (fun (u : Fin z.nMask → ZMod 2) (k : {k // Pr k}) => ∑ y, T.eqw k.1 y * ZK.Table.castF (F := 𝔽) (u y)) =
      fun u k => ∑ y, T'.eqw k.1 y * ZK.Table.castF (F := 𝔽) (u y) := by
    funext u k
    rw [h k.1 k.2]
  exact Iff.of_eq (congrArg Function.Surjective e)

theorem eqw_congr (q) (h : pointsOf c.rest q = pointsOf c'.rest q) :
    ((coinedOf P Tz st).T c).eqw q = ((coinedOf P Tz st).T c').eqw q := by
  funext y
  show fOf (eqWord (pointsOf c.rest q) ((maskWordsOf st).getD y.val 0)) =
    fOf (eqWord (pointsOf c'.rest q) ((maskWordsOf st).getD y.val 0))
  rw [h]

theorem nu_congr (w) (q : Fin (coinedOf P Tz st).z.nC) (h : pointsOf c.rest q = pointsOf c'.rest q) :
    ((coinedOf P Tz st).T c).nu w q = ((coinedOf P Tz st).T c').nu w q := by
  funext v
  show (∑ y : Fin (2 ^ (Tz.S.m - 7)), if y.val ∈ maskWordsOf st then 0
      else fOf (eqWord (pointsOf c.rest q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)) =
    ∑ y : Fin (2 ^ (Tz.S.m - 7)), if y.val ∈ maskWordsOf st then 0
      else fOf (eqWord (pointsOf c'.rest q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)
  rw [h]

theorem sReg_congr (w) (q : Fin (coinedOf P Tz st).z.nR) (h : c.rest.link = c'.rest.link) :
    ((coinedOf P Tz st).T c).sReg w q = ((coinedOf P Tz st).T c').sReg w q := by
  funext v
  show (∑ y : Fin (2 ^ (Tz.S.m - 7)),
      fOf (eqWord (regionPtOf st c.rest.link q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)) =
    ∑ y : Fin (2 ^ (Tz.S.m - 7)),
      fOf (eqWord (regionPtOf st c'.rest.link q) y.val) * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)
  rw [h]

theorem coined_val_congr {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T T' : ZkView.Coined K W Hi VL z) (w : W)
    (q : Fin z.nC) (u : Fin z.nMask → Fin 128 → ZMod 2) (h1 : T.nu w q = T'.nu w q) (h2 : T.eqw q = T'.eqw q) :
    (fun v => T.tab.nu w q v + ∑ y, T.tab.eqw q y * ZK.Table.castF (F := 𝔽) (u y v)) =
      fun v => T'.tab.nu w q v + ∑ y, T'.tab.eqw q y * ZK.Table.castF (F := 𝔽) (u y v) := by
  show (fun v => T.nu w q v + ∑ y, T.eqw q y * ZK.Table.castF (F := 𝔽) (u y v)) =
    fun v => T'.nu w q v + ∑ y, T'.eqw q y * ZK.Table.castF (F := 𝔽) (u y v)
  rw [h1, h2]

/-- **`τ` reads its batched coefficients** (and the base's `μ`). -/
theorem coined_tau_congr {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T T' : ZkView.Coined K W Hi VL z) (w : W)
    (b : ZkView.Base z Hi) (r : Fin 2)
    (h : T.lam r (T.tab.view w (T.randOf b 0)).pre = T'.lam r (T'.tab.view w (T'.randOf b 0)).pre) :
    (T.tab.view w (T.randOf b 0)).tau r = (T'.tab.view w (T'.randOf b 0)).tau r := by
  show ∑ i, T.lam r (T.tab.view w (T.randOf b 0)).pre i * b.2.2.2.2.2.1 r i =
    ∑ i, T'.lam r (T'.tab.view w (T'.randOf b 0)).pre i * b.2.2.2.2.2.1 r i
  rw [h]

/-- **The rank check reads its claims' points** (of the reps up to `r`). -/
theorem rankOn_congr (r : Fin 2)
    (h : ∀ k, ((coinedOf P Tz st).T c).rep k ≤ r → pointsOf c.rest k = pointsOf c'.rest k) :
    ((coinedOf P Tz st).T c).tab.RankOn (fun k => ((coinedOf P Tz st).T c).rep k ≤ r) ↔
      ((coinedOf P Tz st).T c').tab.RankOn (fun k => ((coinedOf P Tz st).T c').rep k ≤ r) := by
  show _ ↔ ((coinedOf P Tz st).T c').tab.RankOn (fun k => ((coinedOf P Tz st).T c).rep k ≤ r)
  exact coined_rankOn _ _ _ fun k hk => eqw_congr P Tz st c c' k (h k hk)

/-- **The restart's check reads its pair coins.** -/
theorem wr_congr (r : Fin 2) (h : c.rho r = c'.rho r) :
    Function.Bijective (fun p : 𝔽 × 𝔽 => ((coinedOf P Tz st).T c).tab.Wr r p.1 p.2) ↔
      Function.Bijective (fun p : 𝔽 × 𝔽 => ((coinedOf P Tz st).T c').tab.Wr r p.1 p.2) := by
  have e : (fun p : 𝔽 × 𝔽 => ((coinedOf P Tz st).T c).tab.Wr r p.1 p.2) =
      fun p => ((coinedOf P Tz st).T c').tab.Wr r p.1 p.2 := by
    funext p
    simp only [ZK.Table.Wr, coinedOf_rho, coinedOf_embed, h]
  rw [e]

/-- **The inner proof's check reads its `β`.** -/
theorem beta_congr (r : Fin 2) (h : c.beta r = c'.beta r) :
    ((coinedOf P Tz st).T c).tab.β r ≠ 0 ↔ ((coinedOf P Tz st).T c').tab.β r ≠ 0 := by
  rw [coinedOf_beta, coinedOf_beta, h]

theorem rbar_congr (r : Fin 2) (h : c.rest.lane r = c'.rest.lane r) :
    ((coinedOf P Tz st).T c).rbar r = ((coinedOf P Tz st).T c').rbar r := by
  funext l
  show eqLane (c.rest.lane r) l.val = eqLane (c'.rest.lane r) l.val
  rw [h]

theorem ρ_congr (r : Fin 2) (h : c.rho r = c'.rho r) :
    ((coinedOf P Tz st).T c).ρ r = ((coinedOf P Tz st).T c').ρ r := by
  funext i
  show (if i = 0 then FlockLevel3.GF256.of (c.rho r).1 else FlockLevel3.GF256.of (c.rho r).2) =
    if i = 0 then FlockLevel3.GF256.of (c'.rho r).1 else FlockLevel3.GF256.of (c'.rho r).2
  rw [h]

theorem embed_congr : ((coinedOf P Tz st).T c).embed = ((coinedOf P Tz st).T c').embed :=
  (coined_embed P Tz st c).trans (coined_embed P Tz st c').symm

theorem lanes_congr : ((coinedOf P Tz st).T c).lanes = ((coinedOf P Tz st).T c').lanes := by
  funext w u l col
  show wordOf (zWith st w u) (l.val + (coinedOf P Tz st).z.nL * col.val) =
    wordOf (zWith st w u) (l.val + (coinedOf P Tz st).z.nL * col.val)
  rfl

theorem β_congr (r : Fin 2) (h : c.beta r = c'.beta r) :
    ((coinedOf P Tz st).T c).β r = ((coinedOf P Tz st).T c').β r := by
  show fOf (c.beta r) = fOf (c'.beta r)
  rw [h]

theorem ζ_congr (r : Fin 2) (h : c.rest.zeta r = c'.rest.zeta r) :
    ((coinedOf P Tz st).T c).ζ r = ((coinedOf P Tz st).T c').ζ r := by
  show fOf (c.rest.zeta r) = fOf (c'.rest.zeta r)
  rw [h]

theorem ia_congr : ((coinedOf P Tz st).T c).ia = ((coinedOf P Tz st).T c').ia :=
  Fin.ext (show (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa = (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa from rfl)

theorem ib_congr : ((coinedOf P Tz st).T c).ib = ((coinedOf P Tz st).T c').ib :=
  Fin.ext (show (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb = (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb from rfl)

end Fields

end FlockSoundness.Discharge.Composed.Cx

namespace FlockComplete.ZkTables

open FlockDeploy FlockSoundness FlockSoundness.Discharge

/-- **A masked session in scope links at two points.** -/
theorem points_of_setup {x : Claims.Msg} {s : Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hscope : Claims.Flock.Scope (deployZk _root_.Flock.Readable.relation) x)
    (hsetup : setupOf true x = some (s, spec, sts)) : spec.points = 2 := by
  obtain ⟨p, hp, hsc⟩ := (_root_.Flock.Readable.scope_iff (deployZk _root_.Flock.Readable.relation)
    _root_.Flock.Readable.relation (readOf true) rfl (fun _ _ h => readOf_zk h) x).1 hscope
  have hi := (setupOf_some hsetup).1
  unfold _root_.Flock.Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := _root_.Flock.Readable.toRead_of_parse (r := s.read) hp
  obtain ⟨hc, -⟩ := (_root_.Flock.Readable.scope_ok_iff p).1 hsc
  have hR := ((_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some hsetup).2
  rw [← ht] at hset
  unfold _root_.Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  exact setupTables_points hset

end FlockComplete.ZkTables

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.ZK FlockSoundness.Discharge CoinLayer
open FlockComplete.CutEq

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

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

/-! ## The pads layout's positions -/

section Layout

variable (m kl k0 : ℕ)

theorem pl_r1ab (i : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).r1ab i = i := rfl
theorem pl_r1c (i : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).r1c i = 64 + i := rfl
theorem pl_ml (i t : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).ml i t = 128 + 2 * i + t := rfl
theorem pl_fa : (_root_.Flock.Zk.PadLayout.new m kl k0).fa = 128 + 2 * (m - 6) := rfl
theorem pl_fb : (_root_.Flock.Zk.PadLayout.new m kl k0).fb = 129 + 2 * (m - 6) := by
  show 128 + 2 * (m - 6) + 1 = _
  omega
theorem pl_fc : (_root_.Flock.Zk.PadLayout.new m kl k0).fc = 130 + 2 * (m - 6) := by
  show 128 + 2 * (m - 6) + 2 = _
  omega
theorem pl_lc (i t : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).lc i t = 131 + 2 * (m - 6) + 2 * i + t := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * i + t = _
  omega
theorem pl_zp (i : ℕ) : (_root_.Flock.Zk.PadLayout.new m kl k0).zp i = 131 + 2 * (m - 6) + 2 * (kl - 6) + i := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + i = _
  omega
theorem pl_lgOod (i : ℕ) :
    (_root_.Flock.Zk.PadLayout.new m kl k0).lgOod i = 195 + 2 * (m - 6) + 2 * (kl - 6) + i := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + 64 + i = _
  omega
theorem pl_lgMsg (i t : ℕ) :
    (_root_.Flock.Zk.PadLayout.new m kl k0).lgMsg i t = 196 + 2 * (m - 6) + 2 * (kl - 6) + 4 * i + t := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + 64 + 1 + 4 * i + t = _
  omega
theorem pl_lgE (i t : ℕ) :
    (_root_.Flock.Zk.PadLayout.new m kl k0).lgE i t = 196 + 2 * (m - 6) + 2 * (kl - 6) + 4 * k0 + 2 * i + t := by
  show 128 + 2 * (m - 6) + 2 + 1 + 2 * (kl - 6) + 0 + 64 + 1 + 4 * k0 + 2 * i + t = _
  omega

end Layout

/-! ## The cut's fields -/

/-- A rep's sent `y₁` (`repSrcCut`'s `y1`). -/
def y1Arr {L : ℕ} : Option (Fin L → 𝕂) → Array _root_.Flock.F256
  | some f => Array.ofFn fun c => kVal (f c)
  | none => #[]

/-- A rep's sent `Y` (`repSrcCut`'s `y`). -/
def yArr {nE nPad : ℕ} (Y2 : Fin nPad → 𝔽) : Option (PadIdx nE → 𝔽) → Array _root_.Flock.F128
  | some Y1 => Array.ofFn (yAt Y1 Y2)
  | none => #[]

section CutFields

variable {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
  (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
  (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) (j : Fin J) (r : Fin 2)

theorem cutAt_masked (i) : (cutAt C cz wt ω lv j).masked r i =
    ((C j).T (cz j)).tab.vals r (wt j) (ω j).1 (ω j).2.1.2.1 i + ((ω j).2.2.1 r).1 i := rfl

theorem cutAt_sReg : (cutAt C cz wt ω lv j).sReg = ((C j).T (cz j)).tab.sReg (wt j) := rfl

open Classical in
theorem cutAt_Y1 : (cutAt C cz wt ω lv j).Y1 r =
    ((if (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).fin r then
      some (fun i : PadIdx (C j).z.nE =>
        ((C j).T (cz j)).tab.hFull ((ω j).2.2.1 r) i + ((C j).T (cz j)).β r * (ω j).2.2.2.2.2.1 r i)
    else none) : Option (PadIdx (C j).z.nE → 𝔽)) := rfl

theorem cutAt_Y2 : (cutAt C cz wt ω lv j).Y2 r =
    fun i : Fin (C j).z.nPad => (ω j).2.2.2.1 r i + ((C j).T (cz j)).β r * (ω j).2.2.2.2.1 r i := rfl

open Classical in
theorem cutAt_y1 : (cutAt C cz wt ω lv j).y1 r =
    if (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).y1 r then
      some (((C j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r)
    else none := rfl

theorem cutAt_ybar : (cutAt C cz wt ω lv j).ybar r = ((C j).T (cz j)).tab.ybar (ω j).2.1.1 (ω j).2.1.2.2 r := rfl

open Classical in
theorem cutAt_lig : (cutAt C cz wt ω lv j).lig r =
    if (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).y1 r then
      some (((C j).T (cz j)).tab.lig r (((C j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r)
        ((ω j).2.2.2.2.2.2.2.2.2 r))
    else none := rfl

variable (cz' : ∀ j, Composed.CoinsZK (C j).Rest) (lv' : ZkView.Leaves (fun j => (C j).z) → ByteArray)

theorem cutAt_inner_congr (hζ : ((C j).T (cz j)).ζ r = ((C j).T (cz' j)).ζ r)
    (ha : ((C j).T (cz j)).ia = ((C j).T (cz' j)).ia) (hb : ((C j).T (cz j)).ib = ((C j).T (cz' j)).ib) :
    (cutAt C cz wt ω lv j).inner r = (cutAt C cz' wt ω lv' j).inner r := by
  show (((C j).T (cz j)).ζ r * ((ω j).2.2.1 r).1 ((C j).T (cz j)).ia + ((ω j).2.2.1 r).2.1,
      ((ω j).2.2.1 r).1 ((C j).T (cz j)).ib + ((ω j).2.2.1 r).2.2) =
    (((C j).T (cz' j)).ζ r * ((ω j).2.2.1 r).1 ((C j).T (cz' j)).ia + ((ω j).2.2.1 r).2.1,
      ((ω j).2.2.1 r).1 ((C j).T (cz' j)).ib + ((ω j).2.2.1 r).2.2)
  rw [hζ, ha, hb]

theorem hFull_congr (ha : ((C j).T (cz j)).ia = ((C j).T (cz' j)).ia)
    (hb : ((C j).T (cz j)).ib = ((C j).T (cz' j)).ib) (h : (Fin (C j).z.nE → 𝔽) × 𝔽 × 𝔽) :
    ((C j).T (cz j)).tab.hFull h = ((C j).T (cz' j)).tab.hFull h := by
  funext i
  rcases i with i | i
  · rfl
  · show ![h.2.1, h.2.2, h.1 ((C j).T (cz j)).ia * h.1 ((C j).T (cz j)).ib, h.2.1 * h.2.2] i =
      ![h.2.1, h.2.2, h.1 ((C j).T (cz' j)).ia * h.1 ((C j).T (cz' j)).ib, h.2.1 * h.2.2] i
    rw [ha, hb]

theorem cutAt_y_congr
    (hfin : (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).fin r ↔
      (famSent (fun j => ((C j).T (cz' j)).tab) (fun j => ((C j).T (cz' j)).rep) j).fin r)
    (hβ : ((C j).T (cz j)).β r = ((C j).T (cz' j)).β r)
    (ha : ((C j).T (cz j)).ia = ((C j).T (cz' j)).ia) (hb : ((C j).T (cz j)).ib = ((C j).T (cz' j)).ib) :
    yArr (nE := (C j).z.nE) (nPad := (C j).z.nPad) ((cutAt C cz wt ω lv j).Y2 r) ((cutAt C cz wt ω lv j).Y1 r) =
      yArr (nE := (C j).z.nE) (nPad := (C j).z.nPad) ((cutAt C cz' wt ω lv' j).Y2 r)
        ((cutAt C cz' wt ω lv' j).Y1 r) := by
  rw [cutAt_Y1, cutAt_Y1, cutAt_Y2, cutAt_Y2]
  simp only [hβ]
  split_ifs with h1 h2 h2
  · rw [hFull_congr (C := C) (cz := cz) (cz' := cz') (j := j) ha hb]
  · exact absurd (hfin.1 h1) h2
  · exact absurd (hfin.2 h2) h1
  · rfl

theorem y1'_congr (hrbar : ((C j).T (cz j)).rbar r = ((C j).T (cz' j)).rbar r)
    (hρ : ((C j).T (cz j)).ρ r = ((C j).T (cz' j)).ρ r) (he : ((C j).T (cz j)).embed = ((C j).T (cz' j)).embed)
    (hl : ((C j).T (cz j)).lanes = ((C j).T (cz' j)).lanes) :
    ((C j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r = ((C j).T (cz' j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r := by
  funext col
  show (∑ l, ((C j).T (cz j)).rbar r l * ((C j).T (cz j)).embed (((C j).T (cz j)).lanes (wt j) (ω j).1 l col)) +
      (((C j).T (cz j)).ρ r 0 * ((C j).T (cz j)).embed ((ω j).2.1.2.1 r 0 col) +
        ((C j).T (cz j)).ρ r 1 * ((C j).T (cz j)).embed ((ω j).2.1.2.1 r 1 col)) =
    (∑ l, ((C j).T (cz' j)).rbar r l * ((C j).T (cz' j)).embed (((C j).T (cz' j)).lanes (wt j) (ω j).1 l col)) +
      (((C j).T (cz' j)).ρ r 0 * ((C j).T (cz' j)).embed ((ω j).2.1.2.1 r 0 col) +
        ((C j).T (cz' j)).ρ r 1 * ((C j).T (cz' j)).embed ((ω j).2.1.2.1 r 1 col))
  rw [hrbar, hρ, he, hl]

theorem cutAt_y1_congr
    (hy : (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j).y1 r ↔
      (famSent (fun j => ((C j).T (cz' j)).tab) (fun j => ((C j).T (cz' j)).rep) j).y1 r)
    (hrbar : ((C j).T (cz j)).rbar r = ((C j).T (cz' j)).rbar r)
    (hρ : ((C j).T (cz j)).ρ r = ((C j).T (cz' j)).ρ r) (he : ((C j).T (cz j)).embed = ((C j).T (cz' j)).embed)
    (hl : ((C j).T (cz j)).lanes = ((C j).T (cz' j)).lanes) :
    y1Arr (L := (C j).z.L) ((cutAt C cz wt ω lv j).y1 r) = y1Arr (L := (C j).z.L) ((cutAt C cz' wt ω lv' j).y1 r) := by
  rw [cutAt_y1, cutAt_y1]
  split_ifs with h1 h2 h2
  · rw [y1'_congr C cz wt ω j r cz' hrbar hρ he hl]
    exact rfl
  · exact absurd (hy.1 h1) h2
  · exact absurd (hy.2 h2) h1
  · rfl

theorem cutAt_ybar_congr (hrbar : ((C j).T (cz j)).rbar r = ((C j).T (cz' j)).rbar r)
    (hρ : ((C j).T (cz j)).ρ r = ((C j).T (cz' j)).ρ r) (he : ((C j).T (cz j)).embed = ((C j).T (cz' j)).embed) :
    (Array.ofFn fun i => kVal ((cutAt C cz wt ω lv j).ybar r i)) =
      Array.ofFn fun i => kVal ((cutAt C cz' wt ω lv' j).ybar r i) := by
  rw [cutAt_ybar, cutAt_ybar]
  show (Array.ofFn fun i => kVal ((∑ l, ((C j).T (cz j)).rbar r l * ((C j).T (cz j)).embed ((ω j).2.1.1 l i)) +
      (((C j).T (cz j)).ρ r 0 * ((C j).T (cz j)).embed ((ω j).2.1.2.2 r 0 i) +
        ((C j).T (cz j)).ρ r 1 * ((C j).T (cz j)).embed ((ω j).2.1.2.2 r 1 i)))) =
    Array.ofFn fun i => kVal ((∑ l, ((C j).T (cz' j)).rbar r l * ((C j).T (cz' j)).embed ((ω j).2.1.1 l i)) +
      (((C j).T (cz' j)).ρ r 0 * ((C j).T (cz' j)).embed ((ω j).2.1.2.2 r 0 i) +
        ((C j).T (cz' j)).ρ r 1 * ((C j).T (cz' j)).embed ((ω j).2.1.2.2 r 1 i)))
  rw [hrbar, hρ, he]

/-- **A rep's `τ` commitment reads its `τ`** (at the executable's leaves). -/
theorem cutC_tauCom_congr (p : ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ByteArray)
    (htau : (((C j).T (cz j)).tab.view (wt j) (((C j).T (cz j)).randOf (ω j) 0)).tau r =
      (((C j).T (cz' j)).tab.view (wt j) (((C j).T (cz' j)).randOf (ω j) 0)).tau r) :
    (cutAt C cz wt ω (fun l => ZkView.leafH (digOf C cz wt ω l + (p l).1, (p l).2)) j).tauCom r =
      (cutAt C cz' wt ω (fun l => ZkView.leafH (digOf C cz' wt ω l + (p l).1, (p l).2)) j).tauCom r := by
  show ZkView.leafH (ZkView.bitsOf (_root_.Flock.Sha512.hash (_root_.Flock.Zk.TAU_TAG ++ _root_.Flock.F128.toBytes
        ((((C j).T (cz j)).tab.view (wt j) (((C j).T (cz j)).randOf (ω j) 0)).tau r))) + (p ⟨j, .inr (.inr r)⟩).1,
      (p ⟨j, .inr (.inr r)⟩).2) =
    ZkView.leafH (ZkView.bitsOf (_root_.Flock.Sha512.hash (_root_.Flock.Zk.TAU_TAG ++ _root_.Flock.F128.toBytes
        ((((C j).T (cz' j)).tab.view (wt j) (((C j).T (cz' j)).randOf (ω j) 0)).tau r))) + (p ⟨j, .inr (.inr r)⟩).1,
      (p ⟨j, .inr (.inr r)⟩).2)
  rw [htau]

end CutFields

section Masked

variable (x : Msg) (cz cz' : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
  (wt : ∀ j, FlockSoundness.Model.Witness (tabC PE x j).S.m) (ω : ∀ j, ZkView.Base (tabC PE x j).z (tabC PE x j).Hi)
  (lv lv' : ZkView.Leaves (fun j => (tabC PE x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2)

theorem repSrcCut_masked (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) (k : ℕ) :
    (repSrcCut PE x cz cut j r).masked k =
      if h : k < (tabC PE x j).z.nE then FlockLevel3.GF128.val ((cut j).masked r ⟨k, h⟩) else 0 := by
  simp only [repSrcCut]
  rfl

theorem tabC_vals (c : Composed.CoinsZK (tabC PE x j).Rest) (w u R i) :
    ((tabC PE x j).T c).tab.vals r w u R i =
      Composed.honestAt (zkTab x j).table (stAt x j) c r (Composed.zWith (stAt x j) w u) (R r) i.val := rfl

theorem cutTab_masked (i) : (cutAt (tabC PE x) cz wt ω lv j).masked r i =
    Composed.honestAt (zkTab x j).table (stAt x j) (cz j) r (Composed.zWith (stAt x j) (wt j) (ω j).1)
      ((ω j).2.1.2.1 r) i.val + ((ω j).2.2.1 r).1 i :=
  (cutAt_masked _ _ _ _ _ _ _ _).trans (congrArg (· + _) (tabC_vals x j r _ _ _ _ _))

theorem masked_congr (k : ℕ)
    (h : ∀ (zw : FlockSoundness.Model.Witness (zkTab x j).table.S.m) (R : Fin 2 → Fin (tabC PE x j).z.L → 𝔽),
      Composed.honestAt (zkTab x j).table (stAt x j) (cz j) r zw R k =
        Composed.honestAt (zkTab x j).table (stAt x j) (cz' j) r zw R k) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).masked k =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).masked k := by
  rw [repSrcCut_masked, repSrcCut_masked]
  by_cases hk : k < (tabC PE x j).z.nE
  · rw [dite_eq_left hk, dite_eq_left hk, cutTab_masked, cutTab_masked, h]
    rfl
  · rw [dite_eq_right hk, dite_eq_right hk]

open Classical in
theorem cutTab_sHat (q) : (cutAt (tabC PE x) cz wt ω lv j).sHat q =
    if (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).cl q then
      some (fun v => ((tabC PE x j).T (cz j)).tab.nu (wt j) q v +
        ∑ y, ((tabC PE x j).T (cz j)).tab.eqw q y * FlockSoundness.ZK.Table.castF ((ω j).1 y v))
    else none := rfl


theorem sv_congr (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) (cut' : ∀ j, ((tabC PE x j).T (cz' j)).tab.Cut)
    (k : ℕ) (h1 : k < 2 → ∀ h : 2 * r.val + k < (tabC PE x j).z.nC,
      (cut j).sHat ⟨2 * r.val + k, h⟩ = (cut' j).sHat ⟨2 * r.val + k, h⟩)
    (h2 : ¬ k < 2 → ∀ h : k - 2 < (tabC PE x j).z.nR, (cut j).sReg ⟨k - 2, h⟩ = (cut' j).sReg ⟨k - 2, h⟩) :
    (repSrcCut PE x cz cut j r).sv k = (repSrcCut PE x cz' cut' j r).sv k := by
  simp only [repSrcCut]
  by_cases hk : k < 2
  · rw [ite_eq_left hk, ite_eq_left hk]
    by_cases hc : 2 * r.val + k < (tabC PE x j).z.nC
    · rw [dite_eq_left (show 2 * r.val + k < ((tabC PE x j).T (cz j)).tab.nC from hc),
        dite_eq_left (show 2 * r.val + k < ((tabC PE x j).T (cz' j)).tab.nC from hc), h1 hk hc]
    · rw [dite_eq_right (show ¬ 2 * r.val + k < ((tabC PE x j).T (cz j)).tab.nC from hc),
        dite_eq_right (show ¬ 2 * r.val + k < ((tabC PE x j).T (cz' j)).tab.nC from hc)]
  · rw [ite_eq_right hk, ite_eq_right hk]
    by_cases hc : k - 2 < (tabC PE x j).z.nR
    · rw [dite_eq_left (show k - 2 < ((tabC PE x j).T (cz j)).tab.nR from hc),
        dite_eq_left (show k - 2 < ((tabC PE x j).T (cz' j)).tab.nR from hc), h2 hk hc]
    · rw [dite_eq_right (show ¬ k - 2 < ((tabC PE x j).T (cz j)).tab.nR from hc),
        dite_eq_right (show ¬ k - 2 < ((tabC PE x j).T (cz' j)).tab.nR from hc)]

theorem tabC_lig_snd (c : Composed.CoinsZK (tabC PE x j).Rest) (y hi) :
    (((tabC PE x j).T c).tab.lig r y hi : Composed.VLOf).2 =
      saltBlob (zkTab x j).table.sch.levels (hiSalt (zkTab x j).table.sch.levels hi) := rfl

theorem hiSalt_congr (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) (cut' : ∀ j, ((tabC PE x j).T (cz' j)).tab.Cut)
    (h : (((cut j).lig r : Option Composed.VLOf)).map Prod.snd = (((cut' j).lig r : Option Composed.VLOf)).map Prod.snd) :
    (repSrcCut PE x cz cut j r).hiSalt = (repSrcCut PE x cz' cut' j r).hiSalt := by
  show blobSalt (zkTab x j).table.sch.levels
      ((((cut j).lig r : Option Composed.VLOf)).map Prod.snd |>.getD ByteArray.empty) =
    blobSalt (zkTab x j).table.sch.levels ((((cut' j).lig r : Option Composed.VLOf)).map Prod.snd |>.getD ByteArray.empty)
  rw [h]

theorem cutTab_lig_congr
    (hy : (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).y1 r ↔
      (famSent (fun j => ((tabC PE x j).T (cz' j)).tab) (fun j => ((tabC PE x j).T (cz' j)).rep) j).y1 r) :
    (((cutAt (tabC PE x) cz wt ω lv j).lig r : Option Composed.VLOf)).map Prod.snd =
      (((cutAt (tabC PE x) cz' wt ω lv' j).lig r : Option Composed.VLOf)).map Prod.snd := by
  rw [cutAt_lig, cutAt_lig]
  split_ifs with h1 h2 h2
  · exact congrArg some ((tabC_lig_snd x j r _ _ _).trans (tabC_lig_snd x j r _ _ _).symm)
  · exact absurd (hy.1 h1) h2
  · exact absurd (hy.2 h2) h1
  · rfl

theorem repSrcCut_tau (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) :
    (repSrcCut PE x cz cut j r).tau = 0 ∧ (repSrcCut PE x cz cut j r).tauSalt = ByteArray.empty := ⟨rfl, rfl⟩

theorem repSrcCut_y1 (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) :
    (repSrcCut PE x cz cut j r).y1 = y1Arr (L := (tabC PE x j).z.L) ((cut j).y1 r) := by
  simp only [repSrcCut]
  cases (cut j).y1 r <;> rfl

theorem repSrcCut_y (cut : ∀ j, ((tabC PE x j).T (cz j)).tab.Cut) :
    (repSrcCut PE x cz cut j r).y =
      yArr (nE := (tabC PE x j).z.nE) (nPad := (tabC PE x j).z.nPad) ((cut j).Y2 r) ((cut j).Y1 r) := by
  simp only [repSrcCut]
  cases (cut j).Y1 r <;> rfl

theorem rep_y1_congr
    (hy : (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).y1 r ↔
      (famSent (fun j => ((tabC PE x j).T (cz' j)).tab) (fun j => ((tabC PE x j).T (cz' j)).rep) j).y1 r)
    (hrbar : ((tabC PE x j).T (cz j)).rbar r = ((tabC PE x j).T (cz' j)).rbar r)
    (hρ : ((tabC PE x j).T (cz j)).ρ r = ((tabC PE x j).T (cz' j)).ρ r)
    (he : ((tabC PE x j).T (cz j)).embed = ((tabC PE x j).T (cz' j)).embed)
    (hl : ((tabC PE x j).T (cz j)).lanes = ((tabC PE x j).T (cz' j)).lanes) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).y1 =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).y1 :=
  (repSrcCut_y1 x cz j r _).trans
    ((cutAt_y1_congr (tabC PE x) cz wt ω lv j r cz' lv' hy hrbar hρ he hl).trans (repSrcCut_y1 x cz' j r _).symm)

theorem rep_ybar_congr (hrbar : ((tabC PE x j).T (cz j)).rbar r = ((tabC PE x j).T (cz' j)).rbar r)
    (hρ : ((tabC PE x j).T (cz j)).ρ r = ((tabC PE x j).T (cz' j)).ρ r)
    (he : ((tabC PE x j).T (cz j)).embed = ((tabC PE x j).T (cz' j)).embed) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).ybar =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).ybar :=
  cutAt_ybar_congr (tabC PE x) cz wt ω lv j r cz' lv' hrbar hρ he

theorem rep_inner_congr (hζ : ((tabC PE x j).T (cz j)).ζ r = ((tabC PE x j).T (cz' j)).ζ r)
    (ha : ((tabC PE x j).T (cz j)).ia = ((tabC PE x j).T (cz' j)).ia)
    (hb : ((tabC PE x j).T (cz j)).ib = ((tabC PE x j).T (cz' j)).ib) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).rhoIn =
        (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).rhoIn ∧
      (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).sigmaIn =
        (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).sigmaIn := by
  have h := cutAt_inner_congr (tabC PE x) cz wt ω lv j r cz' lv' hζ ha hb
  constructor <;> (simp only [repSrcCut]; rw [h])

theorem rep_y_congr
    (hfin : (famSent (fun j => ((tabC PE x j).T (cz j)).tab) (fun j => ((tabC PE x j).T (cz j)).rep) j).fin r ↔
      (famSent (fun j => ((tabC PE x j).T (cz' j)).tab) (fun j => ((tabC PE x j).T (cz' j)).rep) j).fin r)
    (hβ : ((tabC PE x j).T (cz j)).β r = ((tabC PE x j).T (cz' j)).β r)
    (ha : ((tabC PE x j).T (cz j)).ia = ((tabC PE x j).T (cz' j)).ia)
    (hb : ((tabC PE x j).T (cz j)).ib = ((tabC PE x j).T (cz' j)).ib) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω lv) j r).y =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω lv') j r).y :=
  (repSrcCut_y x cz j r _).trans
    ((cutAt_y_congr (tabC PE x) cz wt ω lv j r cz' lv' hfin hβ ha hb).trans (repSrcCut_y x cz' j r _).symm)

/-- **Rep `r`'s `τ` reads only its coins below the inner proof's `β`** and its own masked claims' points. -/
theorem view_tau_congr (hA : Composed.Cx.InAgree (zkTab x j).table (stAt x j) (cz j) (cz' j) r)
    (hpt : ∀ k (h : k < (tabC PE x j).z.nC), (k = 2 * r.val ∨ k = 2 * r.val + 1) →
      Composed.pointsOf ((cz j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩ =
        Composed.pointsOf ((cz' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) ⟨k, h⟩) :
    (((tabC PE x j).T (cz j)).tab.view (wt j) (((tabC PE x j).T (cz j)).randOf (ω j) 0)).tau r =
      (((tabC PE x j).T (cz' j)).tab.view (wt j) (((tabC PE x j).T (cz' j)).randOf (ω j) 0)).tau r := by
  have hrbar := Composed.Cx.rbar_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.lane
  have hρ := Composed.Cx.ρ_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.rhoP
  have he := Composed.Cx.embed_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hla := Composed.Cx.lanes_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hζ : ((tabC PE x j).T (cz j)).ζ r = ((tabC PE x j).T (cz' j)).ζ r :=
    Composed.Cx.ζ_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j) r hA.zeta
  have hia : ((tabC PE x j).T (cz j)).ia = ((tabC PE x j).T (cz' j)).ia :=
    Composed.Cx.ia_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  have hib : ((tabC PE x j).T (cz j)).ib = ((tabC PE x j).T (cz' j)).ib :=
    Composed.Cx.ib_congr PE (zkTab x j).table (stAt x j) (cz j) (cz' j)
  refine Composed.Cx.coined_tau_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j) (ω j) r
    (Composed.Cx.lamOf_congr (zkTab x j).table (stAt x j) (cz j) (cz' j) r _ _ hA ?_ ?_ ?_ ?_ ?_)
  · funext i
    show Composed.honestAt (zkTab x j).table (stAt x j) (cz j) r (Composed.zWith (stAt x j) (wt j) (ω j).1)
        ((ω j).2.1.2.1 r) i.val + ((ω j).2.2.1 r).1 i =
      Composed.honestAt (zkTab x j).table (stAt x j) (cz' j) r (Composed.zWith (stAt x j) (wt j) (ω j).1)
        ((ω j).2.1.2.1 r) i.val + ((ω j).2.2.1 r).1 i
    rw [Composed.Cx.honestAt_all _ _ _ _ r _ _ hA.toB0Agree hA.zc hA.lcAlpha hA.lcBeta hA.lane]
  · unfold Composed.sHatAt
    split_ifs with h
    · exact Composed.Cx.coined_val_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j)
        ⟨2 * r.val, h⟩ (ω j).1 (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val, h⟩ (hpt _ h (.inl rfl)))
        (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val, h⟩ (hpt _ h (.inl rfl)))
    · rfl
  · unfold Composed.sHatAt
    split_ifs with h
    · exact Composed.Cx.coined_val_congr ((tabC PE x j).T (cz j)) ((tabC PE x j).T (cz' j)) (wt j)
        ⟨2 * r.val + 1, h⟩ (ω j).1 (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val + 1, h⟩ (hpt _ h (.inr rfl)))
        (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val + 1, h⟩ (hpt _ h (.inr rfl)))
    · rfl
  · show ∑ col : Fin (tabC PE x j).z.L, ((tabC PE x j).T (cz j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (cz j) r col.val =
      ∑ col : Fin (tabC PE x j).z.L, ((tabC PE x j).T (cz' j)).tab.y1' (wt j) (ω j).1 (ω j).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (cz' j) r col.val
    rw [y1'_congr (tabC PE x) cz wt ω j r cz' hrbar hρ he hla,
      Composed.Cx.bFold_congr _ _ _ _ r hA.toB0Agree hA.lane]
  · show (((tabC PE x j).T (cz j)).ζ r * ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz j)).ia + ((ω j).2.2.1 r).2.1,
        ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz j)).ib + ((ω j).2.2.1 r).2.2) =
      (((tabC PE x j).T (cz' j)).ζ r * ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz' j)).ia + ((ω j).2.2.1 r).2.1,
        ((ω j).2.2.1 r).1 ((tabC PE x j).T (cz' j)).ib + ((ω j).2.2.1 r).2.2)
    rw [hζ, hia, hib]

theorem rep_tauCom_congr (p : ZkView.Leaves (fun j => (tabC PE x j).z) → ZkView.Bits × ByteArray)
    (htau : (((tabC PE x j).T (cz j)).tab.view (wt j) (((tabC PE x j).T (cz j)).randOf (ω j) 0)).tau r =
      (((tabC PE x j).T (cz' j)).tab.view (wt j) (((tabC PE x j).T (cz' j)).randOf (ω j) 0)).tau r) :
    (repSrcCut PE x cz (cutAt (tabC PE x) cz wt ω fun l =>
        ZkView.leafH (digOf (tabC PE x) cz wt ω l + (p l).1, (p l).2)) j r).tauCom =
      (repSrcCut PE x cz' (cutAt (tabC PE x) cz' wt ω fun l =>
        ZkView.leafH (digOf (tabC PE x) cz' wt ω l + (p l).1, (p l).2)) j r).tauCom :=
  cutC_tauCom_congr (tabC PE x) cz wt ω j r cz' p htau

end Masked

/-! ## Each coin at its round -/

/-- **The rounds a check reads below**: table `j`'s rep `r`'s rank check, after its lincheck's last coin; its restart,
after the lane phase's pair; its inner check, after `β`. -/
noncomputable def gateAt (spec : _root_.Flock.SessionSpec) (x : Msg) (j : Fin (jOf x)) (r : Fin 2) :
    FlockSoundness.ZK.Session.Phase → ℕ
  | .opening => FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + 6 + ((stAt x j).m - 6) +
      ((stAt x j).c.kLog - 6)
  | .restart => FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stAt x j) + (zkTab x j).sch.k0 + 4
  | .inner => FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 2 (zkTab x j).sch + 1

theorem rep_two (r : Fin 2) {k : ℕ} (hk : k < 2) : (if 2 * r.val + k < 2 then (0 : Fin 2) else 1) = r := by
  rcases r with ⟨_ | _ | _, hr⟩
  · simp only [Nat.mul_zero, Nat.zero_add, ite_eq_left hk]; rfl
  · simp only [ite_eq_right (show ¬ 2 + k < 2 by omega)]; rfl
  · omega

section Slots

variable {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
  (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)

theorem cxStsOf_half : (stsOf x)[(2 * j.val + r.val) / 2]! = stAt x j := by
  rw [show (2 * j.val + r.val) / 2 = j.val by omega]
  exact getElem!_pos (stsOf x) j.val (lt_of_jOf j)

theorem repStart_next : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val + 1) =
    FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (stAt x j) +
      innerOff 0 (zkTab x j).sch + 4 := by
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  rw [FlockComplete.ZkTables.repStart_succ, cxStsOf_half]
  unfold FlockDeploy.repRounds FlockComplete.piopRoundsZk innerOff
  simp only [↓reduceIte, (zkTab x j).hz]
  omega

theorem gate_le_next (p : FlockSoundness.ZK.Session.Phase) :
    gateAt spec x j r p ≤ FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val + 1) := by
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hm : 14 ≤ (stAt x j).m := (schOk_of (zkTab x j)).m
  rw [repStart_next]
  cases p <;> simp only [gateAt] <;> unfold FlockComplete.piopRoundsZk innerOff <;> omega

theorem gate_le_start (p : FlockSoundness.ZK.Session.Phase) :
    FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) ≤ gateAt spec x j r p := by
  cases p <;> simp only [gateAt] <;> omega

theorem gate_mono {j' : Fin (jOf x)} {r' : Fin 2} {p p' : FlockSoundness.ZK.Session.Phase}
    (h : FlockSoundness.ZK.Session.pos j' r' p' ≤ FlockSoundness.ZK.Session.pos j r p) :
    gateAt spec x j' r' p' ≤ gateAt spec x j r p := by
  unfold FlockSoundness.ZK.Session.pos at h
  have hi : ∀ p : FlockSoundness.ZK.Session.Phase, p.idx ≤ 2 := fun p => by
    cases p <;> simp only [FlockSoundness.ZK.Session.Phase.idx] <;> omega
  by_cases hq : 2 * j'.val + r'.val < 2 * j.val + r.val
  · calc gateAt spec x j' r' p' ≤ FlockComplete.repStartZk spec (stsOf x) (2 * j'.val + r'.val + 1) :=
          gate_le_next j' r' p'
      _ ≤ FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) :=
          FlockComplete.ZkTables.repStart_mono spec (stsOf x) (by omega)
      _ ≤ gateAt spec x j r p := gate_le_start j r p
  · have h1 := hi p
    have h2 := hi p'
    have hj : j' = j := Fin.ext (by omega)
    have hr : r' = r := Fin.ext (by omega)
    subst hj hr
    have hk6 : 6 ≤ (stAt x j').c.kLog := (zkTab x j').wf.1.hk6
    have hm : 14 ≤ (stAt x j').m := (schOk_of (zkTab x j')).m
    cases p <;> cases p' <;> simp only [FlockSoundness.ZK.Session.Phase.idx] at h <;>
      simp only [gateAt, FlockComplete.piopRoundsZk, innerOff] <;> omega

include hsx in
theorem repSlot_at (f : _root_.Flock.Schedule → ℕ) :
    repSlot x (2 * j.val + r.val) f = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stAt x j) + f (zkTab x j).sch := by
  rw [repSlot_eq hsx (sch := (zkTab x j).sch) (tPad := (zkTab x j).tPad)
    (by rw [cxStsOf_half]; exact (zkTab x j).hz) f, cxStsOf_half]

include hsx in
theorem outerSlot_at : outerSlot x j.val r.val = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + 1 := by
  rw [outerSlot_eq hsx, repBase_eq hsx]

variable {c c' : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hcc : ∀ k : Fin (refK x), k.val < n → c k = c' k)

include hcc

theorem outerAt_agree {j r : ℕ} (h : outerSlot x j r < n) (i : ℕ) : outerAt c j r i = outerAt c' j r i := by
  unfold outerAt
  split_ifs with hK
  · rw [hcc ⟨_, hK⟩ h]
  · rfl

theorem link_agree (h : 0 < n) : ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link =
    ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link :=
  funext fun _ => funext fun _ => slotAt_congr hcc h _

theorem rOuter_agree (h : outerSlot x j.val r.val < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r :=
  funext fun _ => outerAt_agree hcc h _

theorem zc_agree (h : repBase x (2 * j.val + r.val) + 2 < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r :=
  slotAt_congr hcc h _

theorem rho_agree_lt {i : ℕ} (h : repBase x (2 * j.val + r.val) + 3 + i ≤ n) :
    ∀ a : Fin ((stAt x j).m - 6), a.val < i →
      ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r a =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r a :=
  fun _ ha => slotAt_congr hcc (by omega) _

theorem rho_agree (h : repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) ≤ n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rho r :=
  funext fun a => rho_agree_lt j r hcc h a a.isLt

theorem lcAlpha_agree (h : repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcAlpha r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcAlpha r :=
  slotAt_congr hcc (by omega) _

theorem lcBeta_agree (h : repBase x (2 * j.val + r.val) + 4 + ((stAt x j).m - 6) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcBeta r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lcBeta r :=
  slotAt_congr hcc (by omega) _

theorem ts_agree_lt {i : ℕ} (h : repBase x (2 * j.val + r.val) + 5 + ((stAt x j).m - 6) + i ≤ n) :
    ∀ a : Fin ((stAt x j).c.kLog - 6), a.val < i →
      ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r a =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r a :=
  fun _ ha => slotAt_congr hcc (by omega) _

theorem ts_agree (h : repBase x (2 * j.val + r.val) + 5 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) ≤ n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ts r :=
  funext fun a => ts_agree_lt j r hcc h a a.isLt

theorem rPrime_agree (h : repBase x (2 * j.val + r.val) + 5 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rPrime r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rPrime r :=
  slotAt_congr hcc (by omega) _

theorem rpp_agree (h : repBase x (2 * j.val + r.val) + 6 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) +
    (2 + 2 * (stAt x j).regions.size) ≤ n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rpp r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rpp r :=
  funext fun k => funext fun _ => slotAt_congr hcc (by have := k.isLt; omega) _

theorem kAt_agree {k : ℕ} (h : k < n) : kAt c k = kAt c' k := by
  unfold kAt
  rw [slotAt_congr hcc h, slotAt_congr hcc h]

include hsx

theorem rsGamma_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) - 1 < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rsGamma r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rsGamma r :=
  funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem oodPt_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).oodPt r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).oodPt r :=
  funext fun _ => funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem ood_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + 1 < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ood r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).ood r :=
  funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem lane_agree_lt {i : ℕ} (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + 2 + i ≤ n) :
    ∀ a : Fin (zkTab x j).sch.k0, a.val < i →
      ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r a =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r a :=
  fun a ha => kAt_agree hcc (by rw [repSlot_at hsx]; omega)

theorem rhoPair_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + (zkTab x j).sch.k0 + 3 < n) :
    (czExe PE x c j).rho r = (czExe PE x c' j).rho r := by
  have h0 : rhoSlot x j.val r.val < n := by unfold rhoSlot; rw [repSlot_at hsx]; omega
  show rhoAt c j.val r.val = rhoAt c' j.val r.val
  unfold rhoAt
  rw [slotAt_congr hcc h0, slotAt_congr hcc h0, slotAt_congr hcc (show rhoSlot x j.val r.val + 1 < n by
    unfold rhoSlot; rw [repSlot_at hsx]; omega), slotAt_congr hcc (show rhoSlot x j.val r.val + 1 < n by
    unfold rhoSlot; rw [repSlot_at hsx]; omega)]

theorem zeta_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 0 (zkTab x j).sch < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zeta r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zeta r :=
  slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem gammas_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 1 (zkTab x j).sch < n) :
    ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).gammas r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).gammas r :=
  funext fun _ => slotAt_congr hcc (by rw [repSlot_at hsx]; exact h) _

theorem beta_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
    FlockComplete.piopRoundsZk spec (stAt x j) + innerOff 2 (zkTab x j).sch < n) :
    (czExe PE x c j).beta r = (czExe PE x c' j).beta r :=
  slotAt_congr hcc (by unfold betaSlot; rw [repSlot_at hsx]; unfold innerOff at h; omega) _

theorem claims_agree (h : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) + 6 + ((stAt x j).m - 6) +
      ((stAt x j).c.kLog - 6) ≤ n) :
    Composed.claimsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) r =
      Composed.claimsOf ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) r := by
  have hb := repBase_eq hsx (2 * j.val + r.val)
  have ho := outerSlot_at hsx j r
  exact Composed.Cx.claimsOf_congr (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
    (zc_agree j r hcc (by omega)) (rho_agree j r hcc (by omega)) (rOuter_agree j r hcc (by omega))
    (rPrime_agree j r hcc (by omega)) (ts_agree j r hcc (by omega))

theorem points_agree (k : Fin _root_.Flock.Zk.RANK_POINTS) (hk : (if k.val < 2 then (0 : Fin 2) else 1) ≤ r)
    (h : gateAt spec x j r .opening ≤ n) :
    Composed.pointsOf ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) k =
      Composed.pointsOf ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) k := by
  have hle : (if k.val < 2 then (0 : Fin 2) else 1).val ≤ r.val := hk
  have hm := FlockComplete.ZkTables.repStart_mono spec (stsOf x)
    (show 2 * j.val + (if k.val < 2 then (0 : Fin 2) else 1).val ≤ 2 * j.val + r.val by omega)
  simp only [gateAt] at h
  unfold Composed.pointsOf
  rw [claims_agree hsx j _ hcc (by omega)]

theorem guard_agree (p : Session.Phase) (h : gateAt spec x j r p ≤ n) :
    famGuard (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
        j r p ↔
      famGuard (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep)
        j r p := by
  cases p with
  | opening =>
    exact Composed.Cx.rankOn_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      fun k hk => points_agree hsx j r hcc k hk h
  | restart =>
    simp only [gateAt] at h
    exact Composed.Cx.wr_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      (rhoPair_agree hsx j r hcc (by omega))
  | inner =>
    simp only [gateAt] at h
    exact Composed.Cx.beta_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      (beta_agree hsx j r hcc (by omega))

/-- **Whether the session reaches a check reads only the coins below its gate.** -/
theorem passes_agree (p : Session.Phase) (h : gateAt spec x j r p ≤ n) :
    famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
        j r p ↔
      famPasses (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep)
        j r p :=
  forall_congr' fun j' => forall_congr' fun r' => forall_congr' fun p' => imp_congr_right fun hp =>
    guard_agree hsx j' r' hcc p' (le_trans (gate_mono j r hp) h)

theorem cl_agree (q : Fin (tabC PE x j).z.nC) (hq : (if q.val < 2 then (0 : Fin 2) else 1) = r)
    (h : gateAt spec x j r .opening ≤ n) :
    (famSent (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
        j).cl q ↔
      (famSent (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab)
        (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep) j).cl q :=
  (Iff.of_eq (congrArg (fun r' => famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab)
      (fun j => ((tabC PE x j).T (czExe PE x c j)).rep) j r' .opening) hq)).trans
    ((passes_agree hsx j r hcc .opening h).trans
      (Iff.of_eq (congrArg (fun r' => famPasses (fun j => ((tabC PE x j).T (czExe PE x c' j)).tab)
        (fun j => ((tabC PE x j).T (czExe PE x c' j)).rep) j r' .opening) hq)).symm)

end Slots

theorem repOk : RepOk := by
  intro x hx wt ω p c c' n hcc s spec hsx j r e he
  have hp2 := FlockComplete.ZkTables.points_of_setup hx.1 hsx
  have hm14 : 14 ≤ (stAt x j).m := (schOk_of (zkTab x j)).m
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hbase := repBase_eq hsx (2 * j.val + r.val)
  have hout := outerSlot_at hsx j r
  have hlig : FlockComplete.CutEq.ligAt (FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val))
      (_root_.Flock.Setup.ofCircuit (stAt x j)) e = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) +
        FlockComplete.piopRoundsZk spec (stAt x j) := by
    unfold FlockComplete.CutEq.ligAt FlockComplete.piopRoundsZk
    rw [he, hp2]
    show _ + 3 + ((stAt x j).m - 6) + (3 + ((stAt x j).c.kLog - 6)) + _ + 1 = _
    omega
  generalize hB : FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) = B at hbase hout hlig ⊢
  have em : (_root_.Flock.Setup.ofCircuit (stAt x j)).m = (stAt x j).m := rfl
  have ef : (_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog = (stAt x j).c.kLog := rfl
  have ek : (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog = (stAt x j).c.kLog := rfl
  have hO : B + 2 ≤ n → ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r := fun hn =>
    rOuter_agree j r hcc (by omega)
  have hz : B + 3 ≤ n → ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r =
      ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).zc r := fun hn =>
    zc_agree j r hcc (by omega)
  have hpiop : FlockComplete.piopRoundsZk spec (stAt x j) =
      (stAt x j).m + (stAt x j).c.kLog - 6 + (2 + 2 * (stAt x j).regions.size + 1) := by
    unfold FlockComplete.piopRoundsZk
    rw [hp2]
  have h0 : B + FlockComplete.piopRoundsZk spec (stAt x j) + 2 ≤ n →
      Composed.Cx.B0Agree (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r := fun hn =>
    ⟨link_agree j hcc (by omega), hO (by omega), rho_agree j r hcc (by omega), ts_agree j r hcc (by omega),
      rpp_agree j r hcc (by omega), rsGamma_agree hsx j r hcc (by omega), oodPt_agree hsx j r hcc (by omega),
      ood_agree hsx j r hcc (by omega)⟩
  refine ⟨fun hn i hi => ?_, fun i hi hn => ?_, fun hn => ?_, fun i hi hn => ?_, fun hn i hi => ?_, ?_,
    fun hn => ?_, fun i hi hn t ht => ?_, fun hn a ha t ht => ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [layOf, pl_r1ab, pl_r1c]
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_r1 _ _ _ _ r zw R (hO hn) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_r1 _ _ _ _ r zw R (hO hn) (by omega)⟩
  · simp only [layOf, pl_ml, em] at hi ⊢
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_ml _ _ _ _ r zw R (hO (by omega)) (hz (by omega))
          (rho_agree_lt j r hcc (by omega)) hi (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_ml _ _ _ _ r zw R (hO (by omega)) (hz (by omega))
          (rho_agree_lt j r hcc (by omega)) hi (by omega) (by omega)⟩
  · simp only [layOf, pl_fa, pl_fb, pl_fc, em] at hn ⊢
    have hρ := rho_agree j r hcc (by omega)
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_f _ _ _ _ r zw R (hO (by omega)) (hz (by omega)) hρ (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_f _ _ _ _ r zw R (hO (by omega)) (hz (by omega)) hρ (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_f _ _ _ _ r zw R (hO (by omega)) (hz (by omega)) hρ (by omega) (by omega)⟩
  · simp only [layOf, pl_lc, em, ek, ef] at hi hn ⊢
    have hρ := rho_agree j r hcc (by omega)
    have ha := lcAlpha_agree j r hcc (by omega)
    have hb := lcBeta_agree j r hcc (by omega)
    exact ⟨masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_lc _ _ _ _ r zw R (hz (by omega)) ha hb hρ (ts_agree_lt j r hcc (by omega)) hi
          (by omega) (by omega),
      masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
        Composed.Cx.honestAt_lc _ _ _ _ r zw R (hz (by omega)) ha hb hρ (ts_agree_lt j r hcc (by omega)) hi
          (by omega) (by omega)⟩
  · simp only [layOf, pl_zp, em, ek, ef] at hn ⊢
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_zp _ _ _ _ r zw R (rho_agree j r hcc (by omega)) (ts_agree j r hcc (by omega))
        (by omega) (by omega)
  · intro k hk hn
    simp only [em, ef] at hn
    have hg : gateAt spec x j r .opening ≤ n := by simp only [gateAt]; rw [hB]; omega
    refine sv_congr x _ _ j r _ _ k (fun hk2 hc => ?_) (fun hk2 hc => ?_)
    · have hq := rep_two r hk2
      have hpt := points_agree hsx j r hcc ⟨2 * r.val + k, hc⟩ (le_of_eq hq) hg
      have hcl := cl_agree hsx j r hcc ⟨2 * r.val + k, hc⟩ hq hg
      rw [cutTab_sHat, cutTab_sHat]
      split_ifs with h1 h2 h2
      · exact congrArg some (Composed.Cx.coined_val_congr _ _ (wt j) ⟨2 * r.val + k, hc⟩ (ω j).1
          (Composed.Cx.nu_congr PE _ _ _ _ (wt j) ⟨2 * r.val + k, hc⟩ hpt)
          (Composed.Cx.eqw_congr PE _ _ _ _ ⟨2 * r.val + k, hc⟩ hpt))
      · exact absurd (hcl.1 h1) h2
      · exact absurd (hcl.2 h2) h1
      · rfl
    · rw [cutAt_sReg, cutAt_sReg]
      exact Composed.Cx.sReg_congr PE _ _ _ _ (wt j) ⟨k - 2, hc⟩ (link_agree j hcc (by omega))
  · simp only [layOf, pl_lgOod, em, ek] at hn ⊢
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_ood _ _ _ _ r zw R (oodPt_agree hsx j r hcc (by omega)) (by omega)
  · simp only [layOf, pl_lgMsg, em, ek] at hn ⊢
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_msg _ _ _ _ r zw R (h0 (by omega)) (lane_agree_lt hsx j r hcc (by omega)) hi
        (by omega) (by omega)
  · simp only [layOf, pl_lgE, em, ek] at hn ⊢
    have ek0 : (zkTab x j).table.sch.k0 = (zkTab x j).sch.k0 := rfl
    exact masked_congr x _ _ wt ω _ _ j r _ fun zw R =>
      Composed.Cx.honestAt_E _ _ _ _ r zw R (h0 (by omega))
        (funext fun a => lane_agree_lt hsx j r hcc (i := (zkTab x j).sch.k0) (by omega) a a.isLt) (by omega)
        (by omega)
  · intro hn
    have ek0 : (zkTab x j).table.sch.k0 = (zkTab x j).sch.k0 := rfl
    have hg : gateAt spec x j r .restart ≤ n := by simp only [gateAt]; rw [hB]; omega
    have hy := passes_agree hsx j r hcc .restart hg
    have hl : ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r :=
      funext fun a => lane_agree_lt hsx j r hcc (i := (zkTab x j).sch.k0) (by omega) a a.isLt
    have hrbar := Composed.Cx.rbar_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r hl
    have hρ := Composed.Cx.ρ_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
      (rhoPair_agree hsx j r hcc (by omega))
    have he := Composed.Cx.embed_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j)
    have hla := Composed.Cx.lanes_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j)
    exact ⟨rep_y1_congr x _ _ wt ω _ _ j r hy hrbar hρ he hla, rep_ybar_congr x _ _ wt ω _ _ j r hrbar hρ he,
      hiSalt_congr x _ _ j r _ _ (cutTab_lig_congr x _ _ wt ω _ _ j r hy)⟩
  · intro hn
    simp only [innerAt] at hn
    have hζ := zeta_agree hsx j r hcc (by simp only [innerOff]; omega)
    exact rep_inner_congr x _ _ wt ω _ _ j r
      (Composed.Cx.ζ_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r hζ)
      (Composed.Cx.ia_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
      (Composed.Cx.ib_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
  · intro hn
    simp only [innerAt] at hn
    have ek0 : (zkTab x j).table.sch.k0 = (zkTab x j).sch.k0 := rfl
    have hg : gateAt spec x j r .opening ≤ n := by simp only [gateAt]; rw [hB]; omega
    have hl : ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r =
        ((czExe PE x c' j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).lane r :=
      funext fun a => lane_agree_lt hsx j r hcc (i := (zkTab x j).sch.k0) (by omega) a a.isLt
    have hA : Composed.Cx.InAgree (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r :=
      { toB0Agree := h0 (by omega)
        zc := hz (by omega)
        lcAlpha := lcAlpha_agree j r hcc (by omega)
        lcBeta := lcBeta_agree j r hcc (by omega)
        rPrime := rPrime_agree j r hcc (by omega)
        lane := hl
        rhoP := rhoPair_agree hsx j r hcc (by omega)
        zeta := zeta_agree hsx j r hcc (by simp only [innerOff]; omega)
        gammas := gammas_agree hsx j r hcc (by simp only [innerOff]; omega) }
    have hrep : ∀ k, (k = 2 * r.val ∨ k = 2 * r.val + 1) → (if k < 2 then (0 : Fin 2) else 1) ≤ r := by
      intro k hk
      rcases hk with rfl | rfl
      · exact le_of_eq (rep_two r (k := 0) (by omega))
      · exact le_of_eq (rep_two r (k := 1) (by omega))
    exact rep_tauCom_congr x _ _ wt ω j r p (view_tau_congr x _ _ wt ω j r hA fun k hk hk2 =>
      points_agree hsx j r hcc ⟨k, hk⟩ (hrep k hk2) hg)
  · intro hn
    simp only [innerAt] at hn
    have hg : gateAt spec x j r .inner ≤ n := by simp only [gateAt, innerOff]; rw [hB]; omega
    exact rep_y_congr x _ _ wt ω _ _ j r (passes_agree hsx j r hcc .inner hg)
      (Composed.Cx.β_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j) r
        (beta_agree hsx j r hcc (by simp only [innerOff]; omega)))
      (Composed.Cx.ia_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
      (Composed.Cx.ib_congr PE (zkTab x j).table (stAt x j) (czExe PE x c j) (czExe PE x c' j))
  · exact fun _ => ⟨(repSrcCut_tau x _ j r _).1.trans (repSrcCut_tau x _ j r _).1.symm,
      (repSrcCut_tau x _ j r _).2.trans (repSrcCut_tau x _ j r _).2.symm⟩

/-- **Sent's `hc`**: every value the executable sends reads only the coins drawn before it. -/
theorem causalExe : CausalExe := causalExe_of_rep repOk

end Claims.ZKMal.Masked
