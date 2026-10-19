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
    (hfg : ∀ a t' l', (x.run t).go l = (.ok (a, t'), l') → l'.size = N → N ≤ n → P (a, t') →
      EqAt n M Q (((f a).run t').go l') (((g a).run t').go l')) :
    EqAt n M Q (((x >>= f).run t).go l) (((y >>= g).run t).go l) := by
  rw [StateT.run_bind, StateT.run_bind]
  exact eqAt_bind hxy fun p l' h h1 h2 h3 => hfg p.1 p.2 l' h h1 h2 h3

/-- **A Ligerito bind of equal runs.** -/
theorem eqAt_bindL {n N M : ℕ} {P : (α × LigState) × Transcript → Prop} {Q : (β × LigState) × Transcript → Prop}
    {x y : LP PS α} {f g : α → LP PS β} {st : LigState} {t : Transcript} {l : Array Claims.Msg}
    (hxy : EqAt n N P (((x.run st).run t).go l) (((y.run st).run t).go l))
    (hfg : ∀ a st' t' l', ((x.run st).run t).go l = (.ok ((a, st'), t'), l') → l'.size = N → N ≤ n →
      P ((a, st'), t') → EqAt n M Q ((((f a).run st').run t').go l') ((((g a).run st').run t').go l')) :
    EqAt n M Q ((((x >>= f).run st).run t).go l) ((((y >>= g).run st).run t).go l) := by
  rw [StateT.run_bind, StateT.run_bind]
  exact eqAt_bindT hxy fun p t' l' h h1 h2 h3 => hfg p.1 p.2 t' l' h h1 h2 h3

theorem eqAt_pureT (a : α) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n N (fun _ => True) (((pure a : TMW PS α).run t).go l) (((pure a : TMW PS α).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨-, rfl⟩ := Prod.mk.inj h
    exact ⟨hl, hN, trivial⟩

theorem eqAt_pureP {P : α × Transcript → Prop} (a : α) (t : Transcript) {l : Array Claims.Msg} {n N : ℕ}
    (hl : l.size = N) (hN : N ≤ n) (hP : P (a, t)) :
    EqAt n N P (((pure a : TMW PS α).run t).go l) (((pure a : TMW PS α).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    exact ⟨hl, hN, hP⟩

theorem eqAt_purePL {P : (α × LigState) × Transcript → Prop} (a : α) (st : LigState) (t : Transcript)
    {l : Array Claims.Msg} {n N : ℕ} (hl : l.size = N) (hN : N ≤ n) (hP : P ((a, st), t)) :
    EqAt n N P ((((pure a : LP PS α).run st).run t).go l) ((((pure a : LP PS α).run st).run t).go l) :=
  eqAt_refl fun _ L h => by
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
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
    refine eqAt_bindT (hb a (by omega) s t l hl hL) fun r t₁ l₁ _ hl₁ hL₁ hr => ?_
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
    refine eqAt_bindL (hb a (by omega) s st t l hl hL) fun r st₁ t₁ l₁ _ hl₁ hL₁ hr => ?_
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
      l.size = g j → g j ≤ n →
      EqAt n (g xs.size) (fun _ => True) ((forIn ys s body).go l) ((forIn ys s body').go l) from
    key xs.toList 0 s l (by simp) hl hL
  intro ys
  induction ys with
  | nil =>
    intro j s l hd hl hL
    have hj : xs.size ≤ j := by
      have := congrArg List.length hd
      simp at this
      omega
    have hj' : j = xs.size := by
      by_contra hne
      have hlt : j < xs.size := by omega
      have := congrArg List.length hd
      simp at this
      omega
    subst hj'
    refine eqAt_refl fun _ L h => ?_
    rw [List.forIn_nil, go_pure] at h
    obtain ⟨-, rfl⟩ := Prod.mk.inj h
    exact ⟨hl, hL, trivial⟩
  | cons a ys ih =>
    intro j s l hd hl hL
    have hget : xs.toList[j]? = some a := by rw [← List.head?_drop, hd]; rfl
    rw [Array.getElem?_toList] at hget
    obtain ⟨hjx, hxa⟩ := Array.getElem?_eq_some_iff.1 hget
    rw [List.forIn_cons, List.forIn_cons]
    have hb' := hb j hjx s l hl hL
    rw [hxa] at hb'
    refine eqAt_bind hb' fun r l' _ hl' hL' hr => ?_
    obtain ⟨u, rfl⟩ := hr
    exact ih (j + 1) u l' (by rw [← List.drop_drop, hd]; rfl) hl' hL'

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
  exact eqAt_bindT (eqAt_squeezeP coin n 1 one_ne_zero t hl) fun _ t' _ _ h1 h2 _ => eqAt_pureT _ t' h1 h2

theorem eqAt_sampleKP (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 1) (fun _ => True) (((sampleKP (asksCut coin n)).run t).go l)
      (((sampleKP (asksCut coin n)).run t).go l) := by
  unfold sampleKP
  exact eqAt_bindT (eqAt_squeezeP coin n 2 two_ne_zero t hl) fun _ t' _ _ h1 h2 _ => eqAt_pureT _ t' h1 h2

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
  refine eqAt_bindL (eqAt_lsampleKP coin n st t hl) fun _ st₁ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_modify _) st₁ t₁ h1 h2) fun _ st₂ t₂ l₂ _ h1 h2 _ => ?_
  split
  · refine eqAt_bindL (eqAt_lsilent (lsilent_modify _) st₂ t₂ h1 h2) fun _ st₃ t₃ l₃ _ h1 h2 _ => ?_
    refine eqAt_bindL (eqAt_lsilent lsilent_get st₃ t₃ h1 h2) fun _ st₄ t₄ l₄ _ h1 h2 _ => ?_
    refine eqAt_bindL (eqAt_lsilent (lsilent_sendMsg _) st₄ t₄ h1 h2) fun _ st₅ t₅ l₅ _ h1 h2 _ => ?_
    exact eqAt_pureL _ st₅ t₅ h1 h2
  · refine eqAt_bindL (eqAt_lsilent lsilent_get st₂ t₂ h1 h2) fun _ st₄ t₄ l₄ _ h1 h2 _ => ?_
    refine eqAt_bindL (eqAt_lsilent (lsilent_sendMsg _) st₄ t₄ h1 h2) fun _ st₅ t₅ l₅ _ h1 h2 _ => ?_
    exact eqAt_pureL _ st₅ t₅ h1 h2

theorem eqAt_oodIntroP (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ} (hdim : dim ≠ 0) (st : LigState)
    (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) :
    EqAt n (N + 2) (fun _ => True) ((((oodIntroP (asksCut coin n) nonce g dim).run st).run t).go l)
      ((((oodIntroP (asksCut coin n) nonce g dim).run st).run t).go l) := by
  unfold oodIntroP
  refine eqAt_bindL (eqAt_lsampleVecP coin n hdim st t hl) fun _ st₁ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_modify _) st₁ t₁ h1 h2) fun _ st₂ t₂ l₂ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_liftTML coin n _ st₂ t₂ h1 h2) fun _ st₃ t₃ l₃ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_sendMsg _) st₃ t₃ h1 h2) fun _ st₄ t₄ l₄ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_powB nonce) st₄ t₄ h1 h2) fun _ st₅ t₅ l₅ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsampleP coin n st₅ t₅ h1) fun _ st₆ t₆ l₆ _ h1 h2 _ => ?_
  exact eqAt_lsilent (lsilent_modify _) st₆ t₆ h1 h2

theorem eqAt_queryP (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) (hq : lv.q ≠ 0)
    (st : LigState) (t : Transcript) {l : Array Claims.Msg} {N : ℕ} (hl : l.size = N) (hN : N ≤ n) :
    EqAt n (N + 2) (fun _ => True) ((((queryP (asksCut coin n) nonce ms lv cm).run st).run t).go l)
      ((((queryP (asksCut coin n) nonce ms lv cm).run st).run t).go l) := by
  unfold queryP
  refine eqAt_bindL (eqAt_lsilent (lsilent_powP nonce 0) st t hl hN) fun _ st₁ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsampleVecP coin n hq st₁ t₁ h1) fun _ st₂ t₂ l₂ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsilent (lsilent_powP nonce 1) st₂ t₂ h1 h2) fun _ st₃ t₃ l₃ _ h1 h2 _ => ?_
  refine eqAt_bindL (eqAt_lsampleVecP coin n (Nat.succ_ne_zero _) st₃ t₃ h1) fun _ st₄ t₄ l₄ _ h1 h2 _ => ?_
  exact eqAt_pureL _ st₄ t₄ h1 h2

end Asks

/-! ## A rep's phases at sources that agree where read -/

section Phases

variable (coin : ℕ → ℕ → F128) (n : ℕ)

/-- A check that throws or returns, sending nothing. -/
theorem eqAt_except (r : Except String α) {f g : α → TMW PS β} {Q : β × Transcript → Prop} (t : Transcript)
    {l : Array Claims.Msg} {N M : ℕ}
    (hfg : ∀ v t' l', r = .ok v → l'.size = N → N ≤ n → EqAt n M Q (((f v).run t').go l') (((g v).run t').go l'))
    (hl : l.size = N) (hN : N ≤ n) :
    EqAt n M Q ((((match r with | .ok v => pure v | .error err => throw err : TMW PS α) >>= f).run t).go l)
      ((((match r with | .ok v => pure v | .error err => throw err : TMW PS α) >>= g).run t).go l) := by
  cases r with
  | ok v =>
    refine eqAt_bindT (eqAt_pureT v t hl hN) fun a t' l' h h1 h2 _ => ?_
    have e : ((Except.ok (v, t), l) : Except String (α × Transcript) × Array Claims.Msg) = (.ok (a, t'), l') := h
    simp only [Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e
    exact hfg v t l rfl h1 h2
  | error err =>
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
  refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleVecP coin n (by norm_num) t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleVecP coin n (by omega) t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
  dsimp only
  have e1 : ((Array.range 64).map fun i => mk (lay.r1ab i)) = (Array.range 64).map fun i => mk' (lay.r1ab i) :=
    Array.map_congr_left fun i hi => ((hab (by omega)) i (by simpa using hi)).1
  have e2 : ((Array.range 64).map fun i => mk (lay.r1c i)) = (Array.range 64).map fun i => mk' (lay.r1c i) :=
    Array.map_congr_left fun i hi => ((hab (by omega)) i (by simpa using hi)).2
  rw [e1, e2]
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_forIn _ _ (L + 3) (fun i => i) (m - 6) (fun i hi s t l hl hL => ?_) _ t₁ l₁
    (by simpa using h1) (by simpa using h2)) fun r t₁ l₁ _ h1 h2 _ => ?_
  · dsimp only
    rw [(hml i hi (by omega)).1, (hml i hi (by omega)).2]
    refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
    exact eqAt_pureP _ t₁ (by rw [h1]) (by omega) ⟨_, rfl⟩
  · dsimp only
    obtain ⟨ea, eb, ec⟩ := hf (by omega)
    rw [ea, eb, ec]
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
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
  refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun alpha t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_except n _ t₁ (fun _ t₁ l₁ _ h1 h2 => ?_) h1 h2
  refine eqAt_bindT (eqAt_forIn _ _ (L + 2) (fun i => i) (c.kLog - 6) (fun i hi s t l hl hL => ?_) _ t₁ l₁
    (by simpa using h1) (by simpa using h2)) fun r t₁ l₁ _ h1 h2 _ => ?_
  · dsimp only
    rw [(hlc i hi (by omega)).1, (hlc i hi (by omega)).2]
    refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
    exact eqAt_pureP _ t₁ (by rw [h1]) (by omega) ⟨_, rfl⟩
  · dsimp only
    have e : ((Array.range 64).map fun i => mk (lay.zp i)) = (Array.range 64).map fun i => mk' (lay.zp i) :=
      Array.map_congr_left fun i hi => hzp (by omega) i (by simpa using hi)
    rw [e]
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleP coin n t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
    exact eqAt_pureT _ t₁ (by rw [h1]) (by omega)

/-- **The ring switch at switched vectors agreeing where sent**: claim `k`'s after `k` sends. -/
theorem ringSwitchCut_eqAt (sv sv' : ℕ → Array F128) (claims : Array Claim) (hc : claims.size ≠ 0) (t : Transcript)
    {l : Array Claims.Msg} {L : ℕ} (hl : l.size = L) (hL : L ≤ n)
    (hsv : ∀ k < claims.size, L + k ≤ n → sv k = sv' k) :
    EqAt n (L + claims.size + 1) (fun _ => True)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l)
      (((ringSwitchCut (asksCut coin n) sv' claims).run t).go l) := by
  unfold ringSwitchCut
  dsimp only
  refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ _ h1 h2 _ => ?_
  refine eqAt_bindT (eqAt_forIn _ _ L (fun i => i) claims.size (fun i hi s t l hl hL => ?_) _ t₁ l₁
    (by simpa using h1) (by simpa using h2)) fun r t₁ l₁ _ h1 h2 _ => ?_
  · dsimp only
    rw [hsv i hi (by omega)]
    refine eqAt_bindT (eqAt_liftTM _ t hl hL) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_liftTM _ t₁ h1 h2) fun _ t₁ l₁ _ h1 h2 _ => ?_
    refine eqAt_bindT (eqAt_sampleVecP coin n (by norm_num) t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
    exact eqAt_pureP _ t₁ (by rw [h1]) (by omega) ⟨_, rfl⟩
  · dsimp only
    refine eqAt_bindT (eqAt_sampleVecP coin n hc t₁ h1) fun _ t₁ l₁ _ h1 h2 _ => ?_
    exact eqAt_pureT _ t₁ h1 h2

end Phases

end FlockComplete.CutEq
