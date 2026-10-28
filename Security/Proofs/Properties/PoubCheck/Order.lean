import Proofs.Properties.PoubCheck.Frozen

/-!
# `Live` under the release order (`LiveUnder`)

`liveUnder`: on coins that agree below `W + t`, and events that the release order (`ReleaseOrder`) fixes up to window
`t`'s draw, two sessions draw window `t` at the same event, from states that differ only in their coins
(`OpeningsUnreadUntilDrawn`), and that event is `close t`, which keeps the window's record and the appends
(`draw_keeps`). After it, `RecordKeptAfterDraw` keeps the record and the openable receipts on each side.

The first event that draws window `t` on the first coins (`exists_first`) is the same on the second (`take_agree`, by
induction with `ReleaseOrder`), and the second draws it there too: were it undrawn, `OpeningsUnreadUntilDrawn` back
from the second coins would leave it undrawn on the first. Nor does the second draw it earlier, by the same lemma
from the first coins. Runs are taken over lists (`runL`), split by `runL_append`.
-/

open PoubCheck PoubCheck.Session

namespace Properties.PoubCheck

/-! ## Runs over lists -/

/-- One event of `run`'s fold. -/
abbrev stepAcc (acc : Array Result × State) (e : Event) : Except String (Array Result × State) := do
  let (r, s') ← step acc.2 e
  pure (acc.1.push r, s')

/-- `run` over a list of events. -/
def runL (s : State) (l : List Event) : Except String (Array Result × State) := l.foldlM stepAcc (#[], s)

theorem run_runL (s : State) (events : Array Event) : run s events = runL s events.toList := by
  unfold run runL
  rw [← Array.foldlM_toList]

theorem stepAcc_ok {acc : Array Result × State} {e : Event} {r : Result} {s' : State}
    (he : step acc.2 e = .ok (r, s')) :
    stepAcc acc e = .ok (acc.1.push r, s') := by
  simp only [stepAcc, he]; rfl

theorem stepAcc_error {acc : Array Result × State} {e : Event} {err : String} (he : step acc.2 e = .error err) :
    stepAcc acc e = .error err := by
  simp only [stepAcc, he]; rfl

theorem fold_state : ∀ (l : List Event) (acc out : Array Result × State) (b : Array Result),
    l.foldlM stepAcc acc = .ok out → ∃ rs, l.foldlM stepAcc (b, acc.2) = .ok (rs, out.2) := by
  intro l
  induction l with
  | nil => intro acc out b h; simp only [List.foldlM_nil] at h; cases h; exact ⟨b, rfl⟩
  | cons e l ih =>
    intro acc out b h
    rw [List.foldlM_cons] at h ⊢
    cases he : step acc.2 e with
    | error err => rw [stepAcc_error he] at h; cases h
    | ok p =>
      obtain ⟨r, s'⟩ := p
      rw [stepAcc_ok he] at h
      obtain ⟨rs, hrs⟩ := ih (acc.1.push r, s') out (b.push r) h
      exact ⟨rs, by rw [stepAcc_ok (acc := (b, acc.2)) he]; exact hrs⟩

theorem runL_append {s s' : State} {xs ys : List Event} {rs : Array Result} (h : runL s (xs ++ ys) = .ok (rs, s')) :
    ∃ rs₁ s₁ rs₂, runL s xs = .ok (rs₁, s₁) ∧ runL s₁ ys = .ok (rs₂, s') := by
  unfold runL at h ⊢
  rw [List.foldlM_append] at h
  cases h₁ : xs.foldlM stepAcc (#[], s) with
  | error err => rw [h₁] at h; cases h
  | ok mid =>
    rw [h₁] at h
    obtain ⟨rs₂, h₂⟩ := fold_state ys mid (rs, s') #[] h
    exact ⟨mid.1, mid.2, rs₂, rfl, h₂⟩

theorem runL_snoc {s s' : State} {xs : List Event} {e : Event} {rs : Array Result}
    (h : runL s (xs ++ [e]) = .ok (rs, s')) :
    ∃ rs₁ s₁ r, runL s xs = .ok (rs₁, s₁) ∧ step s₁ e = .ok (r, s') := by
  obtain ⟨rs₁, s₁, rs₂, h₁, h₂⟩ := runL_append h
  refine ⟨rs₁, s₁, ?_⟩
  unfold runL at h₂
  rw [List.foldlM_cons] at h₂
  cases he : step s₁ e with
  | error err => rw [stepAcc_error (acc := (#[], s₁)) he] at h₂; cases h₂
  | ok p =>
    obtain ⟨r, s₂⟩ := p
    rw [stepAcc_ok (acc := (#[], s₁)) he] at h₂
    have h₃ : (Except.ok (#[].push r, s₂) : Except String (Array Result × State)) = .ok (rs₂, s') := h₂
    injection h₃ with h₃
    injection h₃ with _ h₃
    exact ⟨r, h₁, by rw [h₃]⟩

theorem runL_det {s : State} {l : List Event} {a b : Array Result × State} (h₁ : runL s l = .ok a)
    (h₂ : runL s l = .ok b) : a = b := by
  rw [h₁] at h₂; injection h₂

theorem runL_prefix {s s' : State} {l : List Event} {rs : Array Result} (h : runL s l = .ok (rs, s')) (k : Nat) :
    ∃ rs₁ s₁, runL s (l.take k) = .ok (rs₁, s₁) := by
  rw [← List.take_append_drop k l] at h
  obtain ⟨rs₁, s₁, -, h₁, -⟩ := runL_append h
  exact ⟨rs₁, s₁, h₁⟩

theorem runL_drawn {s s' : State} {l : List Event} {rs : Array Result} (h : runL s l = .ok (rs, s')) {t : Nat}
    {d : Array (Nat × Nat)} (hd : drawnOf s t = some d) : drawnOf s' t = some d :=
  fold_inv (fun x => drawnOf x t = some d) (fun _ _ _ _ hs he => (adv_step he).2.1 t d hs) l (#[], s) (rs, s') hd h

theorem extract_toList (a : Array Event) (k : Nat) : (a.extract 0 k).toList = a.toList.take k := by
  simp [Array.toList_extract]

theorem run_extract (s : State) (a : Array Event) (k : Nat) : run s (a.extract 0 k) = runL s (a.toList.take k) := by
  rw [run_runL, extract_toList]

theorem run_toArray (s : State) (l : List Event) : run s l.toArray = runL s l := by
  rw [run_runL, List.toList_toArray]

/-! ## The step that draws a window -/

theorem no_draw {ws ws' : Array Win} {t : Nat} (k : Keep ws ws') (h0 : dr ws t = none) (h1 : (dr ws' t).isSome) :
    False := by
  rcases k.2 t with e | ⟨-, e⟩ <;> rw [e] at h1
  · rw [h0] at h1; cases h1
  · cases h1

/-- What a window's record reads of it. -/
def SameW (w w' : Win) : Prop :=
  w'.challenge = w.challenge ∧ w'.roots = w.roots ∧ w'.plans = w.plans ∧ w'.heads = w.heads

theorem sameW_modify (ws : Array Win) (u : Nat) (f : Win → Win) (hf : ∀ w, SameW w (f w)) (v : Nat) :
    SameW ws[v]! (ws.modify u f)[v]! := by
  rw [getElem!_modify']
  split
  · exact hf _
  · exact ⟨rfl, rfl, rfl, rfl⟩

theorem sameW_trans {a b c : Win} (h₁ : SameW a b) (h₂ : SameW b c) : SameW a c :=
  ⟨h₂.1.trans h₁.1, h₂.2.1.trans h₁.2.1, h₂.2.2.1.trans h₁.2.2.1, h₂.2.2.2.trans h₁.2.2.2⟩

theorem record_of {s s' : State} (hd : s'.digest = s.digest) (h : ∀ u : Nat, SameW s.windows[u]! s'.windows[u]!)
    (t : Nat) : s'.record t = s.record t := by
  obtain ⟨a, b, c, d⟩ := h t
  obtain ⟨-, b', -, -⟩ := h (t - 1)
  simp only [State.record, a, b, c, d, b', hd]

/-- `close` keeps the record of the window it draws, and the appends. -/
theorem close_keeps {s : State} {t' : Int} {now : Nat} {r : Result} {s₁ : State} (he : close s t' now = .ok (r, s₁))
    {t : Nat} (h0 : dr s.windows t = none) (h1 : (dr s₁.windows t).isSome) :
    s₁.record t = s.record t ∧ s₁.appends = s.appends := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals try
    (exfalso
     refine no_draw ?_ h0 h1
     first
      | exact keep_refl _
      | (apply keep_trans _ (keep_fail _ _ _ _); refine keep_modify _ _ _ ?_; intro; rfl))
  all_goals
    have _ := ‹draws _ _ _ _ _ = some _›
    refine ⟨record_of (s := s) ?_ ?_ t, rfl⟩
    · rfl
    · intro v
      show SameW s.windows[v]! ((s.windows.modify t'.toNat _).modify t'.toNat _)[v]!
      exact sameW_trans (sameW_modify _ _ _ (by intro; exact ⟨rfl, rfl, rfl, rfl⟩) v)
        (sameW_modify _ _ _ (by intro; exact ⟨rfl, rfl, rfl, rfl⟩) v)

/-- The event that draws window `t` is `close t`, so it keeps the window's record and the appends. -/
theorem draw_keeps {s : State} {e : Event} {r : Result} {s₁ : State} (he : step s e = .ok (r, s₁)) {t : Nat}
    (h0 : drawnOf s t = none) (h1 : (drawnOf s₁ t).isSome) :
    s₁.record t = s.record t ∧ s₁.appends = s.appends := by
  cases e with
  | start now => injection he with he; exact (no_draw (keep_start he) h0 h1).elim
  | boundary now => injection he with he; exact (no_draw (keep_boundary he) h0 h1).elim
  | append root count => exact (no_draw (keep_append he) h0 h1).elim
  | release runs withdrawn => injection he with he; exact (no_draw (keep_release he) h0 h1).elim
  | receive t₁ i root runs change st now => injection he with he; exact (no_draw (keep_receive he) h0 h1).elim
  | close t' now => exact close_keeps he h0 h1
  | verify t' openings => injection he with he; exact (no_draw (keep_verify he) h0 h1).elim

/-! ## Two sessions up to the draw -/

theorem take_split (l : List Event) {a b : Nat} (hab : a ≤ b) : l.take b = l.take a ++ (l.take b).drop a := by
  have := List.take_append_drop a (l.take b)
  rw [List.take_take, Nat.min_eq_left hab] at this
  exact this.symm

/-- The first `m + 1` at which `Q` holds, if it holds somewhere and not at `0`. -/
theorem exists_first {Q : Nat → Prop} (h0 : ¬Q 0) :
    ∀ k, (∃ j, j ≤ k ∧ Q j) → ∃ m, m + 1 ≤ k ∧ Q (m + 1) ∧ ∀ i, i ≤ m → ¬Q i := by
  intro k
  induction k with
  | zero =>
    rintro ⟨j, hj, hq⟩
    rw [Nat.le_zero.1 hj] at hq
    exact (h0 hq).elim
  | succ k ih =>
    intro hk
    rcases Classical.em (∃ j, j ≤ k ∧ Q j) with hq | hq
    · obtain ⟨m, hm, hq, hmin⟩ := ih hq
      exact ⟨m, by omega, hq, hmin⟩
    · obtain ⟨j, hj, hqj⟩ := hk
      rcases Nat.lt_or_ge j (k + 1) with hl | hl
      · exact (hq ⟨j, by omega, hqj⟩).elim
      · rw [show j = k + 1 by omega] at hqj
        exact ⟨k, Nat.le_refl _, hqj, fun i hi hqi => hq ⟨i, hi, hqi⟩⟩

/-- Up to window `t`'s draw on `coins`, the events on `coins'` are the same. -/
theorem take_agree (params : Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Event) (hD : ReleaseOrder params reg appendCoins notBeforeNs D)
    {coins coins' : Array ByteArray} {t : Nat} (hag : ∀ j, j < params.windows + t → coins'[j]! = coins[j]!) :
    ∀ (k : Nat) (rs : Array Result) (s : State),
      runL (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) →
      drawnOf s t = none → (D coins').toList.take k = (D coins).toList.take k := by
  intro k
  induction k with
  | zero => intros; simp
  | succ k ih =>
    intro rs s h hnd
    rw [List.take_add_one] at h
    obtain ⟨rs₁, s₁, rs₂, h₁, h₂⟩ := runL_append h
    have hnd₁ : drawnOf s₁ t = none := by
      cases hx : drawnOf s₁ t with
      | none => rfl
      | some d => rw [runL_drawn h₂ hx] at hnd; cases hnd
    have e₂ := hD coins coins' t k rs₁ s₁ hag (by rw [run_extract]; exact h₁) hnd₁
    rw [List.take_add_one, List.take_add_one, ih rs₁ s₁ h₁ hnd₁, Array.getElem?_toList, Array.getElem?_toList,
      e₂]

theorem liveUnder (params : Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Event) (hW : 1 ≤ params.windows) (hm : 1 ≤ params.minPasses)
    (hD : ReleaseOrder params reg appendCoins notBeforeNs D) : LiveUnder params reg appendCoins notBeforeNs D := by
  intro coins coins' t k₁ k₂ rs₁ rs₂ s₁ s₂ hag h₁ h₂ hd₁ hd₂
  rw [run_extract] at h₁ h₂
  have hag' : ∀ j, j < params.windows + t → coins[j]! = coins'[j]! := fun j hj => (hag j hj).symm
  have T2 := openingsUnreadUntilDrawn
  have T3 := recordKeptAfterDraw
  obtain ⟨m, hk₁, ⟨rsA, sA, hA, hdA⟩, hmin'⟩ := exists_first (Q := fun k => ∃ rs s,
      runL (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) ∧
        (drawnOf s t).isSome)
    (by
      rintro ⟨rs0, s0, h0, hd0⟩
      rw [List.take_zero] at h0
      have e := runL_det h0 rfl
      injection e with _ e
      subst e
      simp [drawnOf, State.init] at hd0)
    k₁ ⟨k₁, Nat.le_refl _, rs₁, s₁, h₁, hd₁⟩
  have hmin : ∀ k, k ≤ m → ∀ rs s,
      runL (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) →
      drawnOf s t = none := by
    intro k hk rs s h
    cases hx : drawnOf s t with
    | none => rfl
    | some d => exact (hmin' k hk ⟨rs, s, h, by rw [hx]; rfl⟩).elim
  have hmL : m < (D coins).toList.length := by
    rcases Nat.lt_or_ge m (D coins).toList.length with hc | hc
    · exact hc
    have e : (D coins).toList.take (m + 1) = (D coins).toList.take m := by
      rw [List.take_of_length_le (by omega), List.take_of_length_le (by omega)]
    rw [e] at hA
    rw [hmin m (Nat.le_refl _) _ _ hA] at hdA
    cases hdA
  obtain ⟨e, he⟩ : ∃ e, (D coins).toList[m]? = some e := ⟨_, List.getElem?_eq_getElem hmL⟩
  have htake : (D coins).toList.take (m + 1) = (D coins).toList.take m ++ [e] := by rw [List.take_add_one, he]; rfl
  rw [htake] at hA
  obtain ⟨rsm, sm, r, hsm, hstep⟩ := runL_snoc hA
  have hndm : drawnOf sm t = none := hmin m (Nat.le_refl _) _ _ hsm
  have hL'm := take_agree params reg appendCoins notBeforeNs D hD hag m rsm sm hsm hndm
  have hL'e : (D coins').toList[m]? = (D coins).toList[m]? := by
    rw [Array.getElem?_toList, Array.getElem?_toList]
    exact hD coins coins' t m rsm sm hag (by rw [run_extract]; exact hsm) hndm
  have htake' : (D coins').toList.take (m + 1) = (D coins).toList.take m ++ [e] := by
    rw [List.take_add_one, hL'm, hL'e, he]; rfl
  have hT2 : ∀ k, k ≤ m → ∀ rs s,
      runL (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) →
      runL (State.init params reg coins' appendCoins notBeforeNs) ((D coins).toList.take k) =
        .ok (rs, { s with coins := coins' }) := by
    intro k hk rs s h
    have := T2 params reg coins appendCoins notBeforeNs ((D coins).toList.take k).toArray rs s
      (by rw [run_toArray]; exact h) t coins' hW (hmin k hk rs s h) hag
    rwa [run_toArray] at this
  have hk₂ : m + 1 ≤ k₂ := by
    rcases Nat.lt_or_ge m k₂ with hc | hle
    · exact hc
    obtain ⟨rsk, sk, hsk⟩ := runL_prefix hsm k₂
    rw [List.take_take, Nat.min_eq_left hle] at hsk
    have hndk := hmin k₂ hle _ _ hsk
    rw [take_agree params reg appendCoins notBeforeNs D hD hag k₂ rsk sk hsk hndk] at h₂
    have e := runL_det h₂ (hT2 k₂ hle rsk sk hsk)
    injection e with _ e
    rw [e] at hd₂
    rw [show drawnOf { sk with coins := coins' } t = drawnOf sk t from rfl, hndk] at hd₂
    cases hd₂
  rw [take_split _ hk₂, htake'] at h₂
  obtain ⟨rsB, sB, rsB₂, hB, hB₂⟩ := runL_append h₂
  obtain ⟨rsy, y, r', hy, hstep'⟩ := runL_snoc hB
  have hy' := runL_det hy (hT2 m (Nat.le_refl _) rsm sm hsm)
  injection hy' with _ hy'
  subst hy'
  have hdB : (drawnOf sB t).isSome := by
    cases hx : drawnOf sB t with
    | some d => rfl
    | none =>
      exfalso
      have := T2 params reg coins' appendCoins notBeforeNs ((D coins).toList.take m ++ [e]).toArray rsB sB
        (by rw [run_toArray]; exact hB) t coins hW hx hag'
      rw [run_toArray] at this
      have e := runL_det hA this
      injection e with _ e
      rw [e, show drawnOf { sB with coins := coins } t = drawnOf sB t from rfl, hx] at hdA
      cases hdA
  obtain ⟨hrA, haA⟩ := draw_keeps hstep hndm hdA
  obtain ⟨hrB, haB⟩ := draw_keeps hstep' (s := { sm with coins := coins' }) hndm hdB
  rw [take_split _ hk₁, htake] at h₁
  obtain ⟨rsA', sA', rsA₂, hA', hA₂⟩ := runL_append h₁
  have eA := runL_det hA' hA
  injection eA with _ eA
  subst eA
  obtain ⟨dA, hdA'⟩ := Option.isSome_iff_exists.1 hdA
  obtain ⟨dB, hdB'⟩ := Option.isSome_iff_exists.1 hdB
  obtain ⟨-, r₁, o₁⟩ := T3 params reg coins appendCoins notBeforeNs ((D coins).toList.take m ++ [e]).toArray
    (((D coins).toList.take k₁).drop (m + 1)).toArray rsA' rsA₂ sA' s₁ (by rw [run_toArray]; exact hA')
    (by rw [run_toArray]; exact hA₂) t dA hm hdA'
  obtain ⟨-, r₂, o₂⟩ := T3 params reg coins' appendCoins notBeforeNs ((D coins).toList.take m ++ [e]).toArray
    (((D coins').toList.take k₂).drop (m + 1)).toArray rsB rsB₂ sB s₂ (by rw [run_toArray]; exact hB)
    (by rw [run_toArray]; exact hB₂) t dB hm hdB'
  have hrec : s₂.record t = s₁.record t := by rw [r₂, r₁, hrA, hrB]; rfl
  refine ⟨hrec, fun u => ?_⟩
  rw [o₂ u, o₁ u, hrB, haB, hrA, haA]
  exact Iff.rfl

end Properties.PoubCheck
