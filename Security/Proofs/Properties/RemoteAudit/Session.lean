import Properties.RemoteAudit
import Proofs.Properties.RemoteAudit.Plan

/-!
# The session's bookkeeping (`AcceptedMinPasses`, `PassesStepped`, `PassesReadDistinct`, `ReceiptsBelowLimit`,
`AppendsInOrder`, `PassesChained`)

One invariant of every state a run reaches from `State.init` (`Inv`, on the fields it reads), kept by every event
(`step_inv`): passes arrive only in the window being filled, each with its plan, step and head; a window is drawn only
with `min_passes` passes and accepted only once drawn; the steps strictly increase in arrival order below `2^64`; every
plan reads distinct blocks of the store; each window's receipts stay below `RECEIPT_LIMIT`; the appends are the store's
next indices, each with the next append coin; and the session's head, and each pass's, is the chain over a prefix of the
appends, past which the pass's plan reads nothing. A rejection only marks its window's verdict, so it keeps the
invariant too.
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
  nonneg : -1 ≤ stp
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
    ∃ m, m ≤ appends.size ∧ ws[t].heads[i] = chainOf (appends.extract 0 m) ∧
      ∀ p, p < ws[t].plans[i]!.length → ws[t].plans[i]!.block p < R.blocks + ((appends.toList.take m).map (·.count)).sum
  receipts : ∀ t (ht : t < ws.size), ws[t].receipts < RECEIPT_LIMIT

def Inv (s : State) : Prop :=
  InvOn s.params s.reg s.appendCoins s.windows s.filling s.step s.extent s.appends s.appendDrawn s.head

theorem inv_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Inv (State.init params reg coins appendCoins notBeforeNs) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [State.init, STEP_LIMIT, chainOf]

/-! ## Window updates -/

/-- `w'` keeps `w`'s passes, is drawn only if `w` was or has `min_passes` passes, stays drawn, and is accepted only if
`w` was or it is drawn. -/
def Keeps (P : Params) (w w' : Win) : Prop :=
  w'.roots = w.roots ∧ w'.plans = w.plans ∧ w'.steps = w.steps ∧ w'.heads = w.heads ∧
    (w'.drawn.isSome → w.drawn.isSome ∨ P.minPasses ≤ w.roots.size) ∧ (w.drawn.isSome → w'.drawn.isSome) ∧
    (w'.verdict = some .accept → w.verdict = some .accept ∨ w'.drawn.isSome)

theorem keeps_refl (P : Params) (w : Win) : Keeps P w w := ⟨rfl, rfl, rfl, rfl, Or.inl, id, Or.inl⟩

/-- Windows that keep `ws`'s, one for one. -/
def SamePasses (P : Params) (ws ws' : Array Win) : Prop :=
  ws'.size = ws.size ∧ ∀ t (h : t < ws.size) (h' : t < ws'.size), Keeps P ws[t] ws'[t]

theorem invOn_same {P R AC ws filling stp extent appends drawnN head} {ws' : Array Win}
    (h : InvOn P R AC ws filling stp extent appends drawnN head) (hs : SamePasses P ws ws') :
    InvOn P R AC ws' filling stp extent appends drawnN head := by
  obtain ⟨hsz, hw⟩ := hs
  have e : ∀ t (h' : t < ws'.size), Keeps P ws[t] ws'[t] := fun t h' => hw t (by omega) h'
  refine ⟨fun t ht => ?_, fun t ht hd => ?_, fun t ht hv => ?_, fun t ht hf => ?_, fun t ht i hi => ?_, h.limit, h.nonneg,
    fun t₁ t₂ h₁ h₂ i₁ i₂ hi₁ hi₂ hlt => ?_, fun t ht i hi => ?_, h.store, h.drawnN, h.coins, h.extentSum, h.firsts,
    h.chain, fun t ht i hi => ?_, fun t ht => ?_⟩
  · obtain ⟨e1, e2, e3, e4, -⟩ := e t ht
    rw [e1, e2, e3, e4]; exact h.sizes t (by omega)
  · obtain ⟨e1, -, -, -, e5, -⟩ := e t ht
    rw [e1]
    rcases e5 hd with hd' | hd'
    · exact h.few t (by omega) hd'
    · exact hd'
  · obtain ⟨-, -, -, -, -, e6, e7⟩ := e t ht
    rcases e7 hv with hv' | hv'
    · exact e6 (h.drawn t (by omega) hv')
    · exact hv'
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
  · obtain ⟨-, e2, -, e4, -⟩ := e t ht
    have hi' : i < ws[t].heads.size := by rw [← e4]; exact hi
    have := h.heads t (by omega) i hi'
    simp only [e2, e4]; exact this
  · obtain ⟨-, e2, -⟩ := e t ht
    have := h.receipts t (by omega)
    unfold Win.receipts at this ⊢
    rw [e2]; exact this

theorem samePasses_modify (P : Params) (ws : Array Win) (u : Nat) (f : Win → Win)
    (hf : ∀ hu : u < ws.size, Keeps P ws[u] (f ws[u])) : SamePasses P ws (ws.modify u f) := by
  refine ⟨Array.size_modify, fun t h h' => ?_⟩
  rw [Array.getElem_modify]
  split
  · rename_i hu; subst hu; exact hf h
  · exact keeps_refl P _

/-- A change of fields the invariant doesn't read, or a rejection. -/
theorem keeps_of (P : Params) (w w' : Win) (e1 : w'.roots = w.roots) (e2 : w'.plans = w.plans) (e3 : w'.steps = w.steps)
    (e4 : w'.heads = w.heads) (e5 : w'.drawn = w.drawn) (e6 : w'.verdict = some .accept → w.verdict = some .accept) :
    Keeps P w w' :=
  ⟨e1, e2, e3, e4, fun hd => Or.inl (e5 ▸ hd), fun hd => e5 ▸ hd, fun hv => Or.inl (e6 hv)⟩

theorem inv_fail {s : State} (h : Inv s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) : Inv (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact invOn_same h (samePasses_modify _ _ _ _ fun _ => keeps_of _ _ _ rfl rfl rfl rfl rfl fun hv => by cases hv)
  · exact h

theorem inv_final {s : State} (h : Inv s) (u : Nat) :
    Inv { s with windows := s.windows.modify u ({· with final := true}) } :=
  invOn_same h (samePasses_modify _ _ _ _ fun _ => keeps_refl _ _)

/-- A fresh window: no passes, not drawn, no verdict. -/
theorem invOn_push {P R AC ws filling stp extent appends drawnN head} (h : InvOn P R AC ws filling stp extent appends drawnN head)
    (w : Win) (hr : w.roots = #[]) (hp : w.plans = #[]) (hs : w.steps = #[]) (hh : w.heads = #[])
    (hd : w.drawn = none) (hv : w.verdict = none) :
    InvOn P R AC (ws.push w) filling stp extent appends drawnN head := by
  have g : ∀ t (ht : t < (ws.push w).size), ¬ t < ws.size → (ws.push w)[t] = w := by
    intro t ht hn; simp [Array.getElem_push, hn]
  have k : ∀ t (ht : t < (ws.push w).size) (hl : t < ws.size), (ws.push w)[t] = ws[t] := by
    intro t ht hl; simp [Array.getElem_push, hl]
  refine ⟨fun t ht => ?_, fun t ht hdr => ?_, fun t ht hvr => ?_, fun t ht hf => ?_, fun t ht i hi => ?_, h.limit, h.nonneg,
    fun t₁ t₂ h₁ h₂ i₁ i₂ hi₁ hi₂ hlt => ?_, fun t ht i hi => ?_, h.store, h.drawnN, h.coins, h.extentSum, h.firsts,
    h.chain, fun t ht i hi => ?_, fun t ht => ?_⟩
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
  · by_cases hl : t < ws.size
    · simp only [k t ht hl]; exact h.receipts t hl
    · simp only [g t ht hl, Win.receipts, hp]; simp [RECEIPT_LIMIT]

theorem inv_boundaryNs {s : State} (h : Inv s) (u now : Nat) :
    Inv { s with windows := s.windows.modify u ({· with boundaryNs := some now}) } :=
  invOn_same h (samePasses_modify _ _ _ _ fun _ => keeps_refl _ _)

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
  refine ⟨h.sizes, h.few, h.drawn, h.beyond, h.below, h.limit, h.nonneg, h.inc,
    fun t ht i hi => readsDistinct_mono (h.reads t ht i hi) (by omega), by have := h.store; omega,
    ⟨by simp [hsz], by omega⟩, fun k hk => ?_, ?_, fun k hk => ?_, ?_, fun t ht i hi => ?_, h.receipts⟩
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
  · obtain ⟨m, hm, he, hr⟩ := h.heads t ht i hi
    refine ⟨m, by simp; omega, by rw [extract_push_le _ _ _ hm]; exact he, ?_⟩
    simp only [Array.toList_push]
    rw [List.take_append_of_le_length (by simp; omega)]
    exact hr

/-! ## The events -/

/-- An event leaves the session's parameters, registration and append coins as they were. -/
abbrev Frame (s s' : State) : Prop := s'.params = s.params ∧ s'.reg = s.reg ∧ s'.appendCoins = s.appendCoins

theorem inv_start {s : State} {now : Nat} {r : Result} {s' : State} (h : Inv s) (he : start s now = (r, s')) :
    Inv s' ∧ Frame s s' := by
  unfold start at he
  split at he
  · cases he; exact ⟨h, rfl, rfl, rfl⟩
  split at he
  · cases he; exact ⟨h, rfl, rfl, rfl⟩
  · cases he; exact ⟨inv_releaseCoin h _ _, rfl, rfl, rfl⟩

theorem inv_boundary {s : State} {now : Nat} {r : Result} {s' : State} (h : Inv s) (he : boundary s now = (r, s')) :
    Inv s' ∧ Frame s s' := by
  unfold boundary at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact ⟨h, rfl, rfl, rfl⟩
    | exact ⟨inv_releaseCoin (inv_boundaryNs h _ _) _ _, rfl, rfl, rfl⟩
    | exact ⟨inv_boundaryNs h _ _, rfl, rfl, rfl⟩

theorem inv_append {s : State} {root : ByteArray} {count : Int} {r : Result} {s' : State} (h : Inv s)
    (he : append s root count = .ok (r, s')) : Inv s' ∧ Frame s s' := by
  unfold append at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨h, rfl, rfl, rfl⟩
    | exact ⟨inv_fail h _ _ _, rfl, rfl, rfl⟩
    | exact ⟨invOn_append h _ _ _ (by assumption), rfl, rfl, rfl⟩

theorem inv_draw {s : State} (h : Inv s) (u : Nat) (d : Array (Nat × Nat)) (now : Nat)
    (hm : ∀ hu : u < s.windows.size, s.params.minPasses ≤ s.windows[u].roots.size) :
    Inv { s with windows := s.windows.modify u ({· with drawn := some d, openedNs := some now}) } :=
  invOn_same h (samePasses_modify _ _ _ _ fun hu =>
    ⟨rfl, rfl, rfl, rfl, fun _ => Or.inr (hm hu), fun _ => rfl, fun _ => Or.inr rfl⟩)

theorem inv_accept {s : State} (h : Inv s) (u : Nat) (hd : ∀ hu : u < s.windows.size, s.windows[u].drawn.isSome) :
    Inv { s with windows := s.windows.modify u ({· with verdict := some .accept}) } :=
  invOn_same h (samePasses_modify _ _ _ _ fun hu =>
    ⟨rfl, rfl, rfl, rfl, Or.inl, id, fun _ => Or.inr (hd hu)⟩)

theorem roots_final (ws : Array Win) (u : Nat) (hu : u < (ws.modify u ({· with final := true})).size) :
    (ws.modify u ({· with final := true}))[u].roots.size = ws[u]!.roots.size := by
  rw [Array.getElem_modify, ite_eq_left rfl, getElem!_pos ws u (by simpa using hu)]

theorem drawn_of_get! (ws : Array Win) (u : Nat) (d : Array (Nat × Nat)) (h : ws[u]!.drawn = some d) :
    ∀ hu : u < ws.size, ws[u].drawn.isSome := by
  intro hu
  rw [getElem!_pos ws u hu] at h
  rw [h]; rfl

theorem push_get! (xs : Array Plan) (x : Plan) (i : Nat) (hi : i < xs.size) : (xs.push x)[i]! = xs[i]! := by
  rw [getElem!_pos (xs.push x) i (by simp; omega), getElem!_pos xs i hi, Array.getElem_push_lt]

/-- A pass of window `f`: its root, plan, step, the session's head, its stale bound and its arrival pushed. -/
def addPass (root : ByteArray) (plan : Plan) (post : Nat) (st : Int) (head : Option ByteArray) (below now : Nat)
    (w : Win) : Win :=
  { w with roots := w.roots.push root, plans := w.plans.push plan, post := w.post.push post,
           steps := w.steps.push st.toNat, heads := w.heads.push head, stale := w.stale.push below,
           arrived := w.arrived.push now }

theorem receipts_addPass (root : ByteArray) (plan : Plan) (post : Nat) (st : Int) (head : Option ByteArray)
    (below now : Nat) (w : Win) :
    (addPass root plan post st head below now w).receipts = w.receipts + plan.length := by
  simp [addPass, Win.receipts]

theorem invOn_add {P R AC ws filling stp extent appends drawnN head} {below now : Nat}
    (h : InvOn P R AC ws filling stp extent appends drawnN head) (root : ByteArray) (plan : Plan) (post : Nat)
    (st : Int) (hst : stp < st) (hlim : st < STEP_LIMIT) (hplan : ReadsDistinct plan extent)
    (hrec : ws[filling]!.receipts + plan.length < RECEIPT_LIMIT) :
    InvOn P R AC (ws.modify filling (addPass root plan post st head below now)) filling st extent appends drawnN
      head := by
  have hst0 : ((st.toNat : Nat) : Int) = st := Int.toNat_of_nonneg (by have := h.nonneg; omega)
  have get : ∀ t (ht : t < (ws.modify filling (addPass root plan post st head below now)).size),
      (ws.modify filling (addPass root plan post st head below now))[t] =
        if filling = t then addPass root plan post st head below now (ws[t]'(by simpa using ht)) else ws[t]'(by simpa using ht) :=
    fun t ht => Array.getElem_modify ht
  have hsz : (ws.modify filling (addPass root plan post st head below now)).size = ws.size := Array.size_modify
  refine ⟨fun t ht => ?_, fun t ht hd => ?_, fun t ht hv => ?_, fun t ht hf => ?_, fun t ht i hi => ?_, hlim,
    by have := h.nonneg; omega, fun t₁ t₂ h₁ h₂ i₁ i₂ hi₁ hi₂ hlt => ?_, fun t ht i hi => ?_, h.store, h.drawnN, h.coins, h.extentSum, h.firsts,
    h.chain, fun t ht i hi => ?_, fun t ht => ?_⟩
  · rw [get t ht]
    have := h.sizes t (by omega)
    split
    · simp only [addPass, Array.size_push]; omega
    · exact this
  · rw [get t ht] at hd ⊢
    split
    · rename_i he; rw [ite_eq_left he] at hd
      have := h.few t (by omega) hd
      simp only [addPass, Array.size_push]; omega
    · rename_i he; rw [ite_eq_right he] at hd; exact h.few t (by omega) hd
  · rw [get t ht] at hv ⊢
    split
    · rename_i he; rw [ite_eq_left he] at hv; exact h.drawn t (by omega) hv
    · rename_i he; rw [ite_eq_right he] at hv; exact h.drawn t (by omega) hv
  · rw [get t ht, ite_eq_right (by omega)]; exact h.beyond t (by omega) hf
  · have hb := h.below t (by omega)
    revert hi
    rw [get t ht]
    split
    · intro hi
      simp only [addPass, Array.getElem_push]
      split
      · have := hb i (by assumption); omega
      · omega
    · intro hi; have := hb i hi; omega
  · have hsz₁ := (h.sizes t₁ (by omega)).2.1
    have hsz₂ := (h.sizes t₂ (by omega)).2.1
    revert hi₁ hi₂
    rw [get t₁ h₁, get t₂ h₂]
    intro hi₁ hi₂
    by_cases e₁ : filling = t₁ <;> by_cases e₂ : filling = t₂
    · subst e₁; subst e₂
      simp only [ite_true, addPass, Array.getElem_push, Array.size_push] at hi₁ hi₂ ⊢
      rcases hlt with hlt | ⟨-, hlt⟩
      · omega
      split
      · split
        · exact h.inc _ _ _ _ _ _ _ _ (Or.inr ⟨rfl, hlt⟩)
        · have := h.below filling (by omega) i₁ (by assumption); omega
      · split
        · omega
        · omega
    · subst e₁
      simp only [ite_true, ite_false, e₂, addPass, Array.getElem_push, Array.size_push] at hi₁ hi₂ ⊢
      split
      · exact h.inc _ _ _ _ _ _ _ _ hlt
      · exfalso
        rcases hlt with hlt | ⟨hlt, -⟩
        · have := h.beyond t₂ (by omega) hlt; omega
        · exact e₂ hlt
    · subst e₂
      simp only [ite_true, ite_false, e₁, addPass, Array.getElem_push, Array.size_push] at hi₁ hi₂ ⊢
      split
      · exact h.inc _ _ _ _ _ _ _ _ hlt
      · have := h.below t₁ (by omega) i₁ hi₁; omega
    · simp only [ite_false, e₁, e₂] at hi₁ hi₂ ⊢
      exact h.inc _ _ _ _ _ _ _ _ hlt
  · revert hi
    rw [get t ht]
    split
    · intro hi
      simp only [addPass, Array.getElem_push]
      split
      · exact h.reads t (by omega) i (by assumption)
      · exact hplan
    · intro hi; exact h.reads t (by omega) i hi
  · have hsz₀ := h.sizes t (by omega)
    revert hi
    rw [get t ht]
    split
    · intro hi
      simp only [addPass, Array.getElem_push]
      split
      · rename_i hl
        obtain ⟨m, hm, e, hr⟩ := h.heads t (by omega) i hl
        refine ⟨m, hm, e, ?_⟩
        rw [push_get! _ _ _ (by omega)]
        exact hr
      · rename_i hl
        simp only [addPass, Array.size_push] at hi
        have hg : (ws[t].plans.push plan)[i]! = plan := by
          rw [getElem!_pos (ws[t].plans.push plan) i (by simp; omega), Array.getElem_push]
          split
          · omega
          · rfl
        rw [hg]
        refine ⟨appends.size, Nat.le_refl _, by rw [h.chain, Array.extract_size], fun p hp => ?_⟩
        rw [List.take_of_length_le (by simp), ← h.extentSum]
        exact hplan.1 p hp
    · intro hi; exact h.heads t (by omega) i hi
  · rw [get t ht]
    split
    · rename_i he
      subst he
      rw [receipts_addPass]
      rw [getElem!_pos ws filling (by omega)] at hrec
      exact hrec
    · exact h.receipts t (by omega)

theorem inv_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State} (h : Inv s)
    (he : close s t now = .ok (r, s')) : Inv s' ∧ Frame s s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨h, rfl, rfl, rfl⟩
    | exact ⟨inv_fail (inv_final h _) _ _ _, rfl, rfl, rfl⟩
    | exact ⟨inv_draw (inv_final h _) _ _ _ (fun hu => by dsimp only at hu ⊢; rw [roots_final]; omega), rfl, rfl, rfl⟩

theorem inv_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State} (h : Inv s)
    (he : verify s t openings = (r, s')) : Inv s' ∧ Frame s s' := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨h, rfl, rfl, rfl⟩
    | exact ⟨inv_fail h _ _ _, rfl, rfl, rfl⟩
    | exact ⟨inv_accept h _ (drawn_of_get! _ _ _ (by assumption)), rfl, rfl, rfl⟩

theorem inv_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State} (h : Inv s)
    (he : release s runs withdrawn = (r, s')) : Inv s' ∧ Frame s s' := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨h, rfl, rfl, rfl⟩
    | exact ⟨inv_fail h _ _ _, rfl, rfl, rfl⟩

theorem invOn_filling {P R AC ws filling stp extent appends drawnN head}
    (h : InvOn P R AC ws filling stp extent appends drawnN head) (f' : Nat) (hf : filling ≤ f') :
    InvOn P R AC ws f' stp extent appends drawnN head :=
  { h with beyond := fun t ht hl => h.beyond t ht (by omega) }

theorem inv_advance {s : State} (h : Inv s) {t i : Int}
    (hc : (t == ↑s.filling + 1 && i == 0 && decide (t < ↑s.windows.size)) = true) :
    Inv { s with windows := s.windows.modify s.filling ({· with final := true}), filling := t.toNat } :=
  invOn_filling (inv_final h s.filling) _ (by show s.filling ≤ t.toNat; simp at hc; omega)

theorem step_ok {a b c : Int} (h : ¬(!(decide (a < b) && decide (b < c))) = true) : a < b ∧ b < c := by
  simpa using h

theorem wholeReads (N : Nat) : ReadsDistinct (Plan.whole N) N :=
  readsDistinct_of N N #[(0, N)] #[N] ⟨rfl, fun i hi => by simp at hi; subst hi; simp⟩ (by simp) (by simp)

theorem inv_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {step : Option Int} {now : Nat} {r : Result}
    {s' : State} (h : Inv s) (he : receive s t i root runs change step now = (r, s')) : Inv s' ∧ Frame s s' := by
  unfold receive at he
  split at he
  · cases he; exact ⟨h, rfl, rfl, rfl⟩
  split at he
  · cases he; exact ⟨h, rfl, rfl, rfl⟩
  split at he
  · cases he; exact ⟨inv_fail h _ _ _, rfl, rfl, rfl⟩
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Inv s₂ ∧ Frame s s₂ → jp () s₂ = (r, s') → Inv s' ∧ Frame s s' := by
    intro s₂ ⟨h₂, f₂⟩ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have hP : ∀ plan, plan₂ = some plan → ReadsDistinct plan s₂.extent := by
      intro plan hp
      dsimp only [plan₂, ext₂, N₂] at hp
      split at hp
      · cases hp
      · exact changedReadsDistinct _ _ _ _ _ hp
      · cases hp; exact readsDistinct_mono (wholeReads _) h₂.store
      · exact planReadsDistinct _ _ _ hp
    have key₂ : jp₂ () = (r, s') → Inv s' ∧ Frame s s' := by
      intro hj
      dsimp only [jp₂] at hj
      generalize hq : plan₂ = q at hj
      repeat' split at hj
      all_goals (try dsimp only at hj)
      all_goals cases hj
      all_goals first
        | exact ⟨inv_fail h₂ _ _ _, f₂⟩
        | exact ⟨invOn_add h₂ _ _ _ _ (step_ok ‹_›).1 (step_ok ‹_›).2 (hP _ (by first | exact hq | exact hq.trans ‹_›))
            (Nat.lt_of_not_le ‹_›), f₂⟩
    split at hj
    · split at hj
      · cases hj; exact ⟨inv_fail h₂ _ _ _, f₂⟩
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · exact key s₁ ⟨inv_advance h ‹_›, rfl, rfl, rfl⟩ he
  · split at he
    · cases he; exact ⟨inv_fail h _ _ _, rfl, rfl, rfl⟩
    · exact key s₀ ⟨h, rfl, rfl, rfl⟩ he

theorem step_inv {s : State} {e : Event} {r : Result} {s' : State} (h : Inv s) (he : step s e = .ok (r, s')) :
    Inv s' ∧ Frame s s' := by
  cases e with
  | start now => injection he with he; exact inv_start h he
  | boundary now => injection he with he; exact inv_boundary h he
  | append root count => exact inv_append h he
  | release runs withdrawn => injection he with he; exact inv_release h he
  | receive t i root runs change st now => injection he with he; exact inv_receive h he
  | close t now => exact inv_close h he
  | verify t openings => injection he with he; exact inv_verify h he

/-- Every state a run reaches from `State.init` keeps the invariant, with the parameters, registration and append
coins it started with. -/
theorem inv_run {params : Params} {reg : Registration} {coins appendCoins : Array ByteArray} {notBeforeNs : Nat}
    {events : Array Event} {rs : Array Result} {s : State}
    (h : run (State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s)) :
    Inv s ∧ s.params = params ∧ s.reg = reg ∧ s.appendCoins = appendCoins :=
  run_inv (fun s => Inv s ∧ s.params = params ∧ s.reg = reg ∧ s.appendCoins = appendCoins)
    (fun _ _ _ _ ⟨hs, hp, hr, ha⟩ he =>
      have ⟨hs', fp, fr, fa⟩ := step_inv hs he
      ⟨hs', fp.trans hp, fr.trans hr, fa.trans ha⟩)
    _ _ _ _ ⟨inv_init .., rfl, rfl, rfl⟩ h

/-! ## The statements -/

theorem verdict_beq {v : Option Verdict} (h : (v == some .accept) = true) : v = some .accept := by
  rcases v with _ | _ | ⟨f, a⟩
  · cases h
  · rfl
  · cases h

theorem acceptedMinPasses : AcceptedMinPasses := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun hout
  obtain ⟨h, hp, -, -⟩ := inv_run hrun
  unfold State.outcome at hout
  split at hout
  · rename_i f _
    have := Option.some.inj hout
    cases hf : f.failure <;> simp [hf, Failure.name] at this
  · split at hout
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true] at hc
      obtain ⟨hsz, hall⟩ := hc
      refine ⟨by rw [← hp]; exact hsz, fun t ht => ?_⟩
      rw [← hp]
      exact h.few t ht (h.drawn t ht (verdict_beq (hall t ht)))
    · cases hout

theorem passesStepped : PassesStepped := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun
  have h := (inv_run hrun).1
  refine ⟨fun t ht => ⟨(h.sizes t ht).1, (h.sizes t ht).2.1, (h.sizes t ht).2.2, fun i hi => ?_⟩, h.inc⟩
  have := h.below t ht i hi
  have := h.limit
  omega

theorem passesReadDistinct : PassesReadDistinct :=
  fun _ _ _ _ _ _ _ _ hrun => (inv_run hrun).1.reads

theorem receiptsBelowLimit : ReceiptsBelowLimit :=
  fun _ _ _ _ _ _ _ _ hrun => (inv_run hrun).1.receipts

theorem appendsInOrder : AppendsInOrder := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun
  obtain ⟨h, -, hr, ha⟩ := inv_run hrun
  refine ⟨by rw [← hr]; exact h.extentSum, fun k hk => by rw [← hr]; exact h.firsts k hk, ?_,
    fun k hk => by rw [← ha]; exact h.coins k hk⟩
  have := h.drawnN
  rw [← ha]
  omega

theorem passesChained : PassesChained := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun
  obtain ⟨h, -, hr, -⟩ := inv_run hrun
  rw [← hr]
  exact ⟨h.chain, h.heads⟩

end Properties.RemoteAudit
