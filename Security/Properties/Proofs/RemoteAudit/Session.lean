import Properties.RemoteAudit
import Properties.Proofs.RemoteAudit.Plan

/-!
# The session's bookkeeping (`AcceptedMinPasses`, `PassesStepped`, `PassesReadDistinct`, `AppendsInOrder`,
`PassesChained`)

One invariant of every state a run reaches from `State.init` (`Inv`, on the fields it reads), kept by every event
(`step_inv`): passes arrive only in the window being filled, each with its plan, step and head; a window is drawn only
with `min_passes` passes and accepted only once drawn; the steps strictly increase in arrival order below `2^64`; every
plan reads distinct blocks of the store; the appends are the store's next indices, each with the next append coin; and
the session's head, and each pass's, is the chain over a prefix of the appends. A rejection only marks its window's
verdict, so it keeps the invariant too.
-/

open RemoteAudit RemoteAudit.Session

namespace Properties.RemoteAudit

/-! ## Runs -/

theorem run_inv (Inv : State → Prop) (hstep : ∀ s e r s', Inv s → step s e = .ok (r, s') → Inv s') :
    ∀ (s : State) (events : Array Event) (rs : Array Result) (s' : State), Inv s → run s events = .ok (rs, s') →
      Inv s' := by
  intro s events rs s' hs h
  unfold run at h
  rw [← Array.foldlM_toList] at h
  have key : ∀ (l : List Event) (acc : Array Result × State), Inv acc.2 →
      ∀ out, List.foldlM (fun (acc : Array Result × State) e => do
        let (r, s') ← step acc.2 e
        pure (acc.1.push r, s')) acc l = .ok out → Inv out.2 := by
    intro l
    induction l with
    | nil => intro acc hacc out h; simp only [List.foldlM_nil] at h; cases h; exact hacc
    | cons e l ih =>
      intro acc hacc out h
      rw [List.foldlM_cons] at h
      cases he : step acc.2 e with
      | error err => rw [he] at h; cases h
      | ok rs' =>
        rw [he] at h
        obtain ⟨r, s''⟩ := rs'
        exact ih _ (hstep _ _ _ _ hacc he) out h
  exact key _ (#[], s) hs _ h

/-! ## The invariant -/

/-- What every reached state keeps, on the fields it reads. -/
structure InvOn (P : Params) (R : Registration) (AC : Array ByteArray) (ws : Array Win) (filling : Nat) (stp : Int)
    (extent : Nat) (appends : Array Append) (drawnN : Nat) (head : Option ByteArray) : Prop where
  sizes : ∀ t (ht : t < ws.size), ws[t].plans.size = ws[t].roots.size ∧ ws[t].steps.size = ws[t].roots.size ∧
    ws[t].heads.size = ws[t].roots.size
  few : ∀ t (ht : t < ws.size), ws[t].drawn.isSome → P.minPasses ≤ ws[t].roots.size
  drawn : ∀ t (ht : t < ws.size), ws[t].verdict = some .accept → ws[t].drawn.isSome
  beyond : ∀ t (ht : t < ws.size), filling < t → ws[t].roots.size = 0
  below : ∀ t (ht : t < ws.size) i (hi : i < ws[t].steps.size), (ws[t].steps[i] : Int) ≤ stp
  limit : stp < STEP_LIMIT
  inc : ∀ t₁ t₂ (h₁ : t₁ < ws.size) (h₂ : t₂ < ws.size) i₁ i₂ (hi₁ : i₁ < ws[t₁].steps.size)
    (hi₂ : i₂ < ws[t₂].steps.size), (t₁ < t₂ ∨ (t₁ = t₂ ∧ i₁ < i₂)) → ws[t₁].steps[i₁] < ws[t₂].steps[i₂]
  reads : ∀ t (ht : t < ws.size) i (hi : i < ws[t].plans.size), ReadsDistinct ws[t].plans[i] extent
  store : R.blocks ≤ extent
  drawnN : appends.size = drawnN ∧ drawnN ≤ AC.size
  coins : ∀ k (hk : k < appends.size), appends[k].coin = AC[k]!
  extentSum : extent = R.blocks + (appends.toList.map (·.count)).sum
  firsts : ∀ k (hk : k < appends.size), appends[k].first = R.blocks + ((appends.toList.take k).map (·.count)).sum
  chain : head = chainOf appends
  heads : ∀ t (ht : t < ws.size) i (hi : i < ws[t].heads.size),
    ∃ m, m ≤ appends.size ∧ ws[t].heads[i] = chainOf (appends.extract 0 m)

def Inv (s : State) : Prop :=
  InvOn s.params s.reg s.appendCoins s.windows s.filling s.step s.extent s.appends s.appendDrawn s.head

theorem inv_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Inv (State.init params reg coins appendCoins notBeforeNs) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [State.init, STEP_LIMIT, chainOf]

/-! ## Window updates -/

/-- Windows that agree with `ws` on everything the invariant reads, and accept no window `ws` doesn't. -/
def SamePasses (ws ws' : Array Win) : Prop :=
  ws'.size = ws.size ∧ ∀ t (h : t < ws.size) (h' : t < ws'.size), ws'[t].roots = ws[t].roots ∧
    ws'[t].plans = ws[t].plans ∧ ws'[t].steps = ws[t].steps ∧ ws'[t].heads = ws[t].heads ∧
    ws'[t].drawn = ws[t].drawn ∧ (ws'[t].verdict = some .accept → ws[t].verdict = some .accept)

theorem invOn_same {P R AC ws filling stp extent appends drawnN head} {ws' : Array Win}
    (h : InvOn P R AC ws filling stp extent appends drawnN head) (hs : SamePasses ws ws') :
    InvOn P R AC ws' filling stp extent appends drawnN head := by
  obtain ⟨hsz, hw⟩ := hs
  have e : ∀ t (h' : t < ws'.size), ws'[t].roots = ws[t].roots ∧ ws'[t].plans = ws[t].plans ∧
      ws'[t].steps = ws[t].steps ∧ ws'[t].heads = ws[t].heads ∧ ws'[t].drawn = ws[t].drawn ∧
      (ws'[t].verdict = some .accept → ws[t].verdict = some .accept) := fun t h' => hw t (by omega) h'
  refine ⟨fun t ht => ?_, fun t ht hd => ?_, fun t ht hv => ?_, fun t ht hf => ?_, fun t ht i hi => ?_, h.limit,
    fun t₁ t₂ h₁ h₂ i₁ i₂ hi₁ hi₂ hlt => ?_, fun t ht i hi => ?_, h.store, h.drawnN, h.coins, h.extentSum, h.firsts,
    h.chain, fun t ht i hi => ?_⟩
  · obtain ⟨e1, e2, e3, e4, -⟩ := e t ht
    rw [e1, e2, e3, e4]; exact h.sizes t (by omega)
  · obtain ⟨e1, -, -, -, e5, -⟩ := e t ht
    rw [e1]; rw [e5] at hd; exact h.few t (by omega) hd
  · obtain ⟨-, -, -, -, e5, e6⟩ := e t ht
    rw [e5]; exact h.drawn t (by omega) (e6 hv)
  · obtain ⟨e1, -⟩ := e t ht
    rw [e1]; exact h.beyond t (by omega) hf
  · obtain ⟨-, -, e3, -⟩ := e t ht
    have hi' : i < ws[t].steps.size := by rw [← e3]; exact hi
    have := h.below t (by omega) i hi'
    simp only [e3]; exact this
  · obtain ⟨-, -, e3, -⟩ := e t₁ h₁
    obtain ⟨-, -, e3', -⟩ := e t₂ h₂
    have hi₁' : i₁ < ws[t₁].steps.size := by rw [← e3]; exact hi₁
    have hi₂' : i₂ < ws[t₂].steps.size := by rw [← e3']; exact hi₂
    have := h.inc t₁ t₂ (by omega) (by omega) i₁ i₂ hi₁' hi₂' hlt
    simp only [e3, e3']; exact this
  · obtain ⟨-, e2, -⟩ := e t ht
    have hi' : i < ws[t].plans.size := by rw [← e2]; exact hi
    have := h.reads t (by omega) i hi'
    simp only [e2]; exact this
  · obtain ⟨-, -, -, e4, -⟩ := e t ht
    have hi' : i < ws[t].heads.size := by rw [← e4]; exact hi
    have := h.heads t (by omega) i hi'
    simp only [e4]; exact this

theorem samePasses_modify (ws : Array Win) (u : Nat) (f : Win → Win)
    (hf : ∀ w, (f w).roots = w.roots ∧ (f w).plans = w.plans ∧ (f w).steps = w.steps ∧ (f w).heads = w.heads ∧
      (f w).drawn = w.drawn ∧ ((f w).verdict = some .accept → w.verdict = some .accept)) :
    SamePasses ws (ws.modify u f) := by
  refine ⟨Array.size_modify, fun t h h' => ?_⟩
  rw [Array.getElem_modify]
  split
  · exact hf _
  · exact ⟨rfl, rfl, rfl, rfl, rfl, id⟩

theorem inv_fail {s : State} (h : Inv s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) : Inv (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact invOn_same h (samePasses_modify _ _ _ fun w => ⟨rfl, rfl, rfl, rfl, rfl, fun hv => by cases hv⟩)
  · exact h

theorem inv_final {s : State} (h : Inv s) (u : Nat) :
    Inv { s with windows := s.windows.modify u ({· with final := true}) } :=
  invOn_same h (samePasses_modify _ _ _ fun _ => ⟨rfl, rfl, rfl, rfl, rfl, id⟩)

/-- A fresh window: no passes, not drawn, no verdict. -/
theorem invOn_push {P R AC ws filling stp extent appends drawnN head} (h : InvOn P R AC ws filling stp extent appends drawnN head)
    (w : Win) (hr : w.roots = #[]) (hp : w.plans = #[]) (hs : w.steps = #[]) (hh : w.heads = #[])
    (hd : w.drawn = none) (hv : w.verdict = none) :
    InvOn P R AC (ws.push w) filling stp extent appends drawnN head := by
  have g : ∀ t (ht : t < (ws.push w).size), ¬ t < ws.size → (ws.push w)[t] = w := by
    intro t ht hn; simp [Array.getElem_push, hn]
  have k : ∀ t (ht : t < (ws.push w).size) (hl : t < ws.size), (ws.push w)[t] = ws[t] := by
    intro t ht hl; simp [Array.getElem_push, hl]
  refine ⟨fun t ht => ?_, fun t ht hdr => ?_, fun t ht hvr => ?_, fun t ht hf => ?_, fun t ht i hi => ?_, h.limit,
    fun t₁ t₂ h₁ h₂ i₁ i₂ hi₁ hi₂ hlt => ?_, fun t ht i hi => ?_, h.store, h.drawnN, h.coins, h.extentSum, h.firsts,
    h.chain, fun t ht i hi => ?_⟩
  · by_cases hl : t < ws.size
    · simp only [k t ht hl]; exact h.sizes t hl
    · simp only [g t ht hl, hr, hp, hs, hh]; simp
  · by_cases hl : t < ws.size
    · simp only [k t ht hl] at hdr ⊢; exact h.few t hl hdr
    · simp only [g t ht hl, hd] at hdr; cases hdr
  · by_cases hl : t < ws.size
    · simp only [k t ht hl] at hvr ⊢; exact h.drawn t hl hvr
    · simp only [g t ht hl, hv] at hvr; cases hvr
  · by_cases hl : t < ws.size
    · simp only [k t ht hl]; exact h.beyond t hl hf
    · simp only [g t ht hl, hr]; simp
  · by_cases hl : t < ws.size
    · simp only [k t ht hl] at hi ⊢; exact h.below t hl i hi
    · simp only [g t ht hl, hs] at hi; simp at hi
  · by_cases hl₁ : t₁ < ws.size
    · by_cases hl₂ : t₂ < ws.size
      · simp only [k t₁ h₁ hl₁, k t₂ h₂ hl₂] at hi₁ hi₂ ⊢
        exact h.inc t₁ t₂ hl₁ hl₂ i₁ i₂ hi₁ hi₂ hlt
      · simp only [g t₂ h₂ hl₂, hs] at hi₂; simp at hi₂
    · simp only [g t₁ h₁ hl₁, hs] at hi₁; simp at hi₁
  · by_cases hl : t < ws.size
    · simp only [k t ht hl] at hi ⊢; exact h.reads t hl i hi
    · simp only [g t ht hl, hp] at hi; simp at hi
  · by_cases hl : t < ws.size
    · simp only [k t ht hl] at hi ⊢; exact h.heads t hl i hi
    · simp only [g t ht hl, hh] at hi; simp at hi

theorem inv_boundaryNs {s : State} (h : Inv s) (u now : Nat) :
    Inv { s with windows := s.windows.modify u ({· with boundaryNs := some now}) } :=
  invOn_same h (samePasses_modify _ _ _ fun _ => ⟨rfl, rfl, rfl, rfl, rfl, id⟩)

theorem inv_releaseCoin {s : State} (h : Inv s) (t now : Nat) : Inv (s.releaseCoin t now) :=
  invOn_push h _ rfl rfl rfl rfl rfl rfl

theorem readsDistinct_mono {pl : Plan} {B B' : Nat} (h : ReadsDistinct pl B) (hB : B ≤ B') : ReadsDistinct pl B' :=
  ⟨fun p hp => Nat.lt_of_lt_of_le (h.1 p hp) hB, h.2⟩

theorem chainOf_push (as : Array Append) (a : Append) :
    chainOf (as.push a) = some (appendHead (chainOf as) a.first a.count a.root) := by
  unfold chainOf
  rw [Array.foldl_push]

theorem extract_push_le (as : Array Append) (a : Append) (m : Nat) (hm : m ≤ as.size) :
    (as.push a).extract 0 m = as.extract 0 m := by
  rw [Array.extract_push, ite_eq_left hm]

/-- An append: the store's next `n` indices, the next coin, the chain extended. -/
theorem invOn_append {P R AC ws filling stp extent appends drawnN head}
    (h : InvOn P R AC ws filling stp extent appends drawnN head) (n : Nat) (root coin : ByteArray)
    (hc : AC[drawnN]? = some coin) :
    InvOn P R AC ws filling stp (extent + n) (appends.push ⟨extent, n, root, coin⟩) (drawnN + 1)
      (some (appendHead head extent n root)) := by
  have hlt : drawnN < AC.size := by
    rcases Nat.lt_or_ge drawnN AC.size with hl | hl
    · exact hl
    · rw [Array.getElem?_eq_none hl] at hc; cases hc
  have hcoin : AC[drawnN]! = coin := by
    rw [getElem!_pos AC drawnN hlt]; rw [Array.getElem?_eq_getElem hlt] at hc; exact Option.some.inj hc
  have hsz := h.drawnN.1
  refine ⟨h.sizes, h.few, h.drawn, h.beyond, h.below, h.limit, h.inc,
    fun t ht i hi => readsDistinct_mono (h.reads t ht i hi) (by omega), by have := h.store; omega,
    ⟨by simp [hsz], by omega⟩, fun k hk => ?_, ?_, fun k hk => ?_, ?_, fun t ht i hi => ?_⟩
  · rw [Array.getElem_push]
    split
    · exact h.coins k (by assumption)
    · rename_i hn
      simp only [Array.size_push] at hk
      have : k = drawnN := by omega
      subst this; exact hcoin.symm
  · rw [h.extentSum]; simp [Nat.add_assoc]
  · rw [Array.getElem_push]
    split
    · rename_i hl
      rw [h.firsts k hl]
      simp only [Array.toList_push]
      rw [List.take_append_of_le_length (by simp; omega)]
    · rename_i hn
      simp only [Array.size_push] at hk
      have hk' : k = appends.size := by omega
      subst hk'
      simp only [Array.toList_push]
      rw [List.take_append_of_le_length (by simp), List.take_of_length_le (by simp)]
      exact h.extentSum
  · rw [chainOf_push, ← h.chain]
  · obtain ⟨m, hm, he⟩ := h.heads t ht i hi
    exact ⟨m, by simp; omega, by rw [extract_push_le _ _ _ hm]; exact he⟩

/-! ## The events -/

theorem inv_start {s : State} {now : Nat} {r : Result} {s' : State} (h : Inv s) (he : start s now = (r, s')) :
    Inv s' := by
  unfold start at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  · cases he; exact inv_releaseCoin h _ _

theorem inv_boundary {s : State} {now : Nat} {r : Result} {s' : State} (h : Inv s) (he : boundary s now = (r, s')) :
    Inv s' := by
  unfold boundary at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact inv_releaseCoin (inv_boundaryNs h _ _) _ _
    | exact inv_boundaryNs h _ _

end Properties.RemoteAudit
