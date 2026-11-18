module

public import Proofs.Properties.Poub.Remote.Accept
public import Properties.Poub.Remote.Refresh
public import Proofs.Properties.Poub.Remote.Released
public import Batteries.Tactic.OpenPrivate

/-!
# Proofs of refresh's code-side properties

`Properties.Poub.Remote.Refresh`'s `OpensOrigin`: an accepted window passes `checkOpening` on each sampled opening
(`checkWindow_accept`), and `checkOpening_origin` reads the origin's checks off its early returns.

`RegisteredCapped`, as `LifetimeCapped` is proved (`Lifetime`): the invariant `Cap` says that with refresh every block
below `N` a recorded pass of window `t` reads has `t < lifetime`; `receive` adds a pass only past `refreshRefusal`,
whose `lifetime` arm refuses a plan of window `t ≥ lifetime` with a run starting below `N` (`cap_check`), and a block a
plan reads lies in one of its runs (`block_mem`).

`ProbationKept` the same way, over the invariant `Prob`: every append holding a block a recorded pass of window `t`
reads is `Kept` at `t`. A new pass is past `refreshRefusal`'s `probation` arm (`prob_check`), which reads each append's
birth: defined, since every append's slot is in the schedule and so is a re-key's origin slot, which `State.slotOk`
finds an append holding (`Slots`). An append comes after every block a recorded pass reads (`InvOn.reads`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open Poub Poub.Session

open private Poub.Session.live from Poub.Session

namespace Properties.Poub

/-- An opening `checkOpening` passes, of a block `j` of an append `a` with an origin leaf `(q, oroot, odom)`, names
`q`, and its block decodes under `a`'s coin to a payload whose leaf is leaf `q` under `oroot`. -/
theorem checkOpening_origin {reg : Registration} {digest : ByteArray} {appends : Array Append} {w : WindowRecord}
    {r : Receipter} {i p : Nat} {o : Opening} {a : Append} {q : Nat} {oroot : ByteArray}
    {odom : Flock.FV3.Sha512.Domain}
    (h : checkOpening reg digest appends w r i p o = none) (hj : reg.blocks ≤ w.passes[i]!.plan.block p)
    (ha : appendOf appends (w.passes[i]!.plan.block p) = some a)
    (hq : originLeaf reg digest appends a (w.passes[i]!.plan.block p) = some (q, oroot, odom)) :
    o.originPosition = some q ∧ ∃ payload,
      Codec.decode o.block (w.passes[i]!.plan.block p) reg.key (some (o.digest, some a.coin)) = .ok (some payload) ∧
      odom.verifyOpening oroot odom.count q (hidden payload o.payloadSalt) SCHEMA o.payloadPath = true := by
  unfold checkOpening at h
  simp only [Id.run] at h
  generalize w.passes[i]!.plan.block p = j at h hj ha hq ⊢
  have hb : ¬ j < reg.blocks := Nat.not_lt.2 hj
  split at h
  · simp at h
  simp only [ha, hq, Option.bind, Option.map] at h
  split at h
  · simp at h
  split at h
  · simp at h
  split at h
  · simp at h
  rename_i hpos
  refine ⟨by simpa using hpos, ?_⟩
  split at h
  · split at h
    · simp at h
    split at h
    · rename_i payload hdec
      split at h
      · simp at h
      rename_i h4
      exact ⟨payload, hdec, by simpa using h4⟩
    · simp at h
  · simp at h

theorem opensOrigin : OpensOrigin := by
  intro reg appends w coin samples openings h
  obtain ⟨sampled, hd, hc⟩ := verifyWindow_accept h
  obtain ⟨hsize, hk⟩ := checkWindow_accept hc
  refine ⟨sampled, hd, hsize, ?_⟩
  intro k hk' a q oroot odom
  dsimp only
  intro hj ha hq
  exact checkOpening_origin (hk k hk') hj ha hq

/-! ## `RegisteredCapped` -/

/-- `Q t b` of every block `b` a recorded pass of window `t` reads. -/
def PlansOn (Q : Nat → Nat → Prop) (ws : Array Win) : Prop :=
  ∀ t (ht : t < ws.size) i (hi : i < ws[t].plans.size) p, p < ws[t].plans[i].length → Q t (ws[t].plans[i].block p)

/-- With refresh, every block below `N` a recorded pass of window `t` reads has `t < L`. -/
def Cap (s : State) : Prop :=
  s.params.refresh = true → PlansOn (fun t b => b < s.reg.blocks → t < s.params.lifetime) s.windows

theorem cap_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Cap (State.init params reg coins appendCoins notBeforeNs) :=
  fun _ t ht => by simp [State.init] at ht

theorem plansOn_same {Q ws} {ws' : Array Win} (h : PlansOn Q ws) (hs : SamePlans ws ws') : PlansOn Q ws' := by
  obtain ⟨hsz, hw⟩ := hs
  have e : ∀ t (h' : t < ws'.size), ws'[t].plans = ws[t].plans := fun t h' => hw t (by omega) h'
  intro t ht i hi p hp
  have hi' : i < ws[t].plans.size := by rw [← e t ht]; exact hi
  simp only [e t ht] at hp ⊢
  exact h t (by omega) i hi' p hp

theorem plansOn_push {Q ws} (h : PlansOn Q ws) (w : Win) (hp : w.plans = #[]) : PlansOn Q (ws.push w) := by
  intro t ht i hi
  by_cases hl : t < ws.size
  · have k : (ws.push w)[t] = ws[t] := by simp [Array.getElem_push, hl]
    simp only [k] at hi ⊢
    exact h t hl i hi
  · have g : (ws.push w)[t] = w := by simp [Array.getElem_push, hl]
    simp only [g, hp] at hi; simp at hi

theorem plansOn_add {Q ws filling} {below now : Nat} {pr : Array Nat → Array Nat} (h : PlansOn Q ws)
    (root : ByteArray) (plan : Plan) (post : Nat) (st : Int) (head : Option ByteArray)
    (hav : ∀ p, p < plan.length → Q filling (plan.block p)) :
    PlansOn Q (ws.modify filling (addPass root plan post st head below now pr)) := by
  have get : ∀ t (ht : t < (ws.modify filling (addPass root plan post st head below now pr)).size),
      (ws.modify filling (addPass root plan post st head below now pr))[t] =
        if filling = t then addPass root plan post st head below now pr (ws[t]'(by simpa using ht)) else ws[t]'(by simpa using ht) :=
    fun t ht => Array.getElem_modify ht
  have hsz : (ws.modify filling (addPass root plan post st head below now pr)).size = ws.size := Array.size_modify
  intro t ht i
  rw [get t ht]
  split
  · rename_i he
    subst he
    intro hi
    simp only [addPass, Array.getElem_push]
    split
    · exact h filling (by omega) i (by assumption)
    · exact hav
  · intro hi; exact h t (by omega) i hi

theorem cap_same {s : State} (h : Cap s) {ws' : Array Win} (hs : SamePlans s.windows ws') :
    Cap { s with windows := ws' } :=
  fun hr => plansOn_same (h hr) hs

theorem cap_fail {s : State} (h : Cap s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) :
    Cap (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact cap_same h (samePlans_modify _ _ _ fun _ => rfl)
  · exact h

theorem cap_final {s : State} (h : Cap s) (u : Nat) :
    Cap { s with windows := s.windows.modify u ({· with final := true}) } :=
  cap_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem cap_boundaryNs {s : State} (h : Cap s) (u now : Nat) :
    Cap { s with windows := s.windows.modify u ({· with boundaryNs := some now}) } :=
  cap_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem cap_sample {s : State} (h : Cap s) (u : Nat) (d : Array (Nat × Nat)) (now : Nat) :
    Cap { s with windows := s.windows.modify u ({· with drawn := some d, openedNs := some now}) } :=
  cap_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem cap_accept {s : State} (h : Cap s) (u : Nat) :
    Cap { s with windows := s.windows.modify u ({· with verdict := some .accept}) } :=
  cap_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem cap_releaseCoin {s : State} (h : Cap s) (t now : Nat) : Cap (s.releaseCoin t now) :=
  fun hr => plansOn_push (h hr) _ rfl

/-- A plan `receive` lets past `refreshRefusal` in window `t`, with refresh, reads a block below `N` only if
`t < lifetime`. -/
theorem cap_check {s : State} {t : Nat} {plan : Plan} (hP : EndsOk plan)
    (hchk : ∀ f, s.refreshRefusal t plan = some f → False) (hr : s.params.refresh = true) :
    ∀ p, p < plan.length → plan.block p < s.reg.blocks → t < s.params.lifetime := by
  intro p hp hN
  replace hchk : s.refreshRefusal t plan = none := Option.eq_none_iff_forall_ne_some.2 hchk
  refine Nat.lt_of_not_le fun hL => ?_
  obtain ⟨⟨f, c⟩, hrun, hr1, -⟩ := block_mem hP hp
  dsimp only at hr1
  unfold State.refreshRefusal at hchk
  dsimp only at hchk
  split at hchk
  · rename_i hn; simp [hr] at hn
  split at hchk
  · cases hchk
  rename_i hno
  apply hno
  rw [Bool.and_eq_true]
  exact ⟨decide_eq_true hL, Array.any_eq_true'.2 ⟨(f, c), hrun, by simp only [decide_eq_true_eq]; omega⟩⟩

theorem cap_start {s : State} {now : Nat} {r : Result} {s' : State} (h : Cap s)
    (he : start s now = (r, s')) : Cap s' := by
  unfold start at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact cap_releaseCoin h _ _

theorem cap_boundary {s : State} {now : Nat} {r : Result} {s' : State} (h : Cap s)
    (he : boundary s now = (r, s')) : Cap s' := by
  unfold boundary at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact cap_releaseCoin (cap_boundaryNs h _ _) _ _
    | exact cap_boundaryNs h _ _

theorem cap_append {s : State} {root : ByteArray} {count : Int} {origin : _root_.Poub.Origin} {slot : Option Nat}
    {r : Result} {s' : State} (h : Cap s) (he : append s root count origin slot = .ok (r, s')) : Cap s' := by
  unfold append at he
  repeat' (first | split at he | (dsimp only at he; split at he))
  all_goals (try simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at he)
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact cap_fail h _ _ _

theorem cap_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State} (h : Cap s)
    (he : release s runs withdrawn = (r, s')) : Cap s' := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact cap_fail h _ _ _

theorem cap_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State} (h : Cap s)
    (he : close s t now = .ok (r, s')) : Cap s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact cap_fail (cap_final h _) _ _ _
    | exact cap_sample (cap_final h _) _ _ _

theorem cap_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (h : Cap s) (he : verify s t openings = (r, s')) : Cap s' := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact cap_fail h _ _ _
    | exact cap_accept h _

theorem cap_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {step : Option Int} {now : Nat} {r : Result}
    {s' : State} (h : Cap s) (he : receive s t i root runs change step now = (r, s')) : Cap s' := by
  unfold receive at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  split at he
  · cases he; exact cap_fail h _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Cap s₂ → jp () s₂ = (r, s') → Cap s' := by
    intro s₂ h₂ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have hW : ∀ plan, plan₂ = some plan → EndsOk plan := by
      intro plan hp
      dsimp only [plan₂, ext₂, N₂] at hp
      split at hp
      · cases hp
      · exact endsOk_changed hp
      · cases hp; exact endsOk_whole _
      · exact (make_spec hp).1
    have h₃ : Cap s₃ := h₂
    have key₂ : jp₂ () = (r, s') → Cap s' := by
      intro hj
      dsimp only [jp₂] at hj
      generalize hq : plan₂ = q at hj
      repeat' split at hj
      all_goals (try dsimp only at hj)
      all_goals cases hj
      all_goals first
        | exact cap_fail h₂ _ _ _
        | exact cap_fail h₃ _ _ _
        | exact fun hr => plansOn_add (h₃ hr) _ _ _ _ _
            (cap_check (s := s₃) (hW _ (by first | exact hq | exact hq.trans ‹_›)) ‹_› hr)
    split at hj
    · split at hj
      · cases hj; exact cap_fail h₂ _ _ _
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · exact key s₁ (cap_final h _) he
  · split at he
    · cases he; exact cap_fail h _ _ _
    · exact key s₀ h he

theorem step_cap {s : State} {e : Event} {r : Result} {s' : State} (h : Cap s)
    (he : step s e = .ok (r, s')) : Cap s' := by
  cases e with
  | start now => injection he with he; exact cap_start h he
  | boundary now => injection he with he; exact cap_boundary h he
  | append root count origin slot => exact cap_append h he
  | release runs withdrawn => injection he with he; exact cap_release h he
  | receive t i root runs change st now => injection he with he; exact cap_receive h he
  | close t now => exact cap_close h he
  | verify t openings => injection he with he; exact cap_verify h he

theorem registeredCapped : RegisteredCapped := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun hr
  have hc := run_inv Cap (fun _ _ _ _ hs he => step_cap hs he) _ _ _ _ (cap_init _ _ _ _ _) hrun
  obtain ⟨-, hp, hreg, -⟩ := inv_run hrun
  intro t ht i hi p hpl hN
  have := hc (by rw [hp]; exact hr) t ht i hi p hpl (by rw [hreg]; exact hN)
  rwa [hp] at this

/-! ## `ProbationKept` -/

/-- Append `x`'s payload is seasoned at window `t`: registered, or born (by `sw`, the window of its fresh slot) at a
window `b` with `b + L ≤ t`. -/
def Kept (sw : Nat → Option Nat) (L t : Nat) (x : Append) : Prop :=
  match x.origin with
  | .registered _ => True
  | .fresh => ∃ b, sw x.slot = some b ∧ b + L ≤ t
  | .rekey k => ∃ b, sw k = some b ∧ b + L ≤ t

/-- Every append holding a block a recorded pass of window `t` reads is `Kept` at `t`. -/
def ProbOn (sw : Nat → Option Nat) (L : Nat) (ws : Array Win) (as : Array Append) : Prop :=
  PlansOn (fun t b => ∀ a (ha : a < as.size), as[a].first ≤ b → b < as[a].first + as[a].count → Kept sw L t as[a]) ws

/-- With refresh, every append's slot is in the schedule, and so is a re-key's origin slot. -/
def Slots (s : State) : Prop :=
  s.params.refresh = true → ∀ k (hk : k < s.appends.size), (s.slotWindow s.appends[k].slot).isSome ∧
    ∀ m, s.appends[k].origin = .rekey m → (s.slotWindow m).isSome

/-- With refresh, `Slots` and `ProbOn`. -/
def Prob (s : State) : Prop :=
  Slots s ∧ (s.params.refresh = true → ProbOn s.slotWindow s.params.lifetime s.windows s.appends)

/-- What the probation check reads of the appends: `Slots`, and each append at or above `N`. -/
def Held (s : State) : Prop :=
  Slots s ∧ ∀ k (hk : k < s.appends.size), s.reg.blocks ≤ s.appends[k].first

theorem held_of {s : State} (hi : Inv s) (h : Prob s) : Held s :=
  ⟨h.1, fun k hk => by rw [hi.firsts k hk]; exact Nat.le_add_right _ _⟩

theorem prob_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Prob (State.init params reg coins appendCoins notBeforeNs) :=
  ⟨fun _ k hk => by simp [State.init] at hk, fun _ t ht => by simp [State.init] at ht⟩

theorem probOn_push {sw L ws as} (h : ProbOn sw L ws as) (x : Append) (hx : PlansOn (fun _ b => b < x.first) ws) :
    ProbOn sw L ws (as.push x) := by
  intro t ht i hi p hp a ha h1 h2
  have hb := hx t ht i hi p hp
  by_cases hl : a < as.size
  · simp only [Array.getElem_push_lt hl] at h1 h2 ⊢
    exact h t ht i hi p hp a hl h1 h2
  · have : a = as.size := by simp only [Array.size_push] at ha; omega
    subst this
    simp only [Array.getElem_push_eq] at h1
    omega

theorem prob_same {s : State} (h : Prob s) {ws' : Array Win} (hs : SamePlans s.windows ws') :
    Prob { s with windows := ws' } :=
  ⟨h.1, fun hr => plansOn_same (h.2 hr) hs⟩

theorem prob_fail {s : State} (h : Prob s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) :
    Prob (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact prob_same h (samePlans_modify _ _ _ fun _ => rfl)
  · exact h

theorem prob_final {s : State} (h : Prob s) (u : Nat) :
    Prob { s with windows := s.windows.modify u ({· with final := true}) } :=
  prob_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem prob_boundaryNs {s : State} (h : Prob s) (u now : Nat) :
    Prob { s with windows := s.windows.modify u ({· with boundaryNs := some now}) } :=
  prob_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem prob_sample {s : State} (h : Prob s) (u : Nat) (d : Array (Nat × Nat)) (now : Nat) :
    Prob { s with windows := s.windows.modify u ({· with drawn := some d, openedNs := some now}) } :=
  prob_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem prob_accept {s : State} (h : Prob s) (u : Nat) :
    Prob { s with windows := s.windows.modify u ({· with verdict := some .accept}) } :=
  prob_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem prob_releaseCoin {s : State} (h : Prob s) (t now : Nat) : Prob (s.releaseCoin t now) :=
  ⟨h.1, fun hr => plansOn_push (h.2 hr) _ rfl⟩

/-- With refresh, `append` fills only a slot of the schedule, and a re-key slot only re-keys a slot an append holds
(`State.slotOk`). -/
theorem slot_held {s : State} {count : Nat} {origin : _root_.Poub.Origin} {slot : Option Nat} (hS : Slots s)
    (h2 : ¬(s.params.refresh && !(slot.any fun k => s.slotOk k count origin)) = true) (hr : s.params.refresh = true) :
    (s.slotWindow (slot.getD s.appends.size)).isSome ∧ ∀ m, origin = .rekey m → (s.slotWindow m).isSome := by
  have hS := hS hr
  rw [hr] at h2
  cases slot with
  | none => simp at h2
  | some k =>
    have h3 : s.slotOk k count origin = true := by simpa using h2
    unfold State.slotOk at h3
    rw [Bool.and_eq_true] at h3
    obtain ⟨-, h3⟩ := h3
    simp only [Option.getD_some]
    split at h3
    · rename_i hk
      exact ⟨by simp [State.slotWindow, hk], fun m hm => by cases hm⟩
    · rename_i hk
      exact ⟨by simp [State.slotWindow, hk], fun m hm => by cases hm⟩
    · rename_i k₀ k' hk
      refine ⟨by simp [State.slotWindow, hk], fun m hm => ?_⟩
      cases hm
      simp only [Bool.and_eq_true, beq_iff_eq, Option.any_eq_true] at h3
      obtain ⟨hkk, y, hy, -⟩ := h3
      have hys := Array.find?_some (p := fun x : Append => x.slot == k₀) hy
      obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_of_find?_eq_some hy)
      simp only [beq_iff_eq] at hys
      rw [← hkk, ← hys]
      exact (hS j hj).1
    · simp at h3

theorem prob_push {s : State} (hi : Inv s) (h : Prob s) {count : Nat} {origin : _root_.Poub.Origin}
    {slot : Option Nat} (h2 : ¬(s.params.refresh && !(slot.any fun k => s.slotOk k count origin)) = true)
    (x : Append) (hx : x.first = s.extent) (hxs : x.slot = slot.getD s.appends.size) (hxo : x.origin = origin)
    (hd : Option ByteArray) (n : Nat) :
    Prob { s with appends := s.appends.push x, head := hd, extent := n } := by
  refine ⟨fun hr k hk => ?_, fun hr => ?_⟩
  · show (s.slotWindow (s.appends.push x)[k].slot).isSome ∧
      ∀ m', (s.appends.push x)[k].origin = .rekey m' → (s.slotWindow m').isSome
    rw [Array.getElem_push]
    split
    · exact h.1 hr k ‹_›
    · rw [hxs, hxo]; exact slot_held h.1 h2 hr
  · show ProbOn s.slotWindow s.params.lifetime s.windows (s.appends.push x)
    exact probOn_push (h.2 hr) x fun t ht i hi' p hp => by rw [hx]; exact (hi.reads t ht i hi').1 p hp

/-- A plan past `refreshRefusal` in window `t`, with refresh: every append holding a block it reads is `Kept` at `t`.
Its `probation` arm refuses a run holding a block of an append whose birth `b` has `t < b + lifetime`, and with every
slot in the schedule each append's birth is defined. -/
theorem prob_check {s : State} {t : Nat} {plan : Plan} (hP : EndsOk plan)
    (hchk : ∀ f, s.refreshRefusal t plan = some f → False) (hr : s.params.refresh = true) (hH : Held s) :
    ∀ p, p < plan.length → ∀ a (ha : a < s.appends.size), s.appends[a].first ≤ plan.block p →
      plan.block p < s.appends[a].first + s.appends[a].count → Kept s.slotWindow s.params.lifetime t s.appends[a] := by
  intro p hp a ha h1 h2
  obtain ⟨hS, hN⟩ := hH
  replace hchk : s.refreshRefusal t plan = none := Option.eq_none_iff_forall_ne_some.2 hchk
  obtain ⟨⟨f, c⟩, hrun, hr1, hr2⟩ := block_mem hP hp
  dsimp only at hr1 hr2
  have hNa := hN a ha
  have hb : ¬(s.born a).any (t < · + s.params.lifetime) = true := by
    intro hb
    unfold State.refreshRefusal at hchk
    dsimp only at hchk
    split at hchk
    · rename_i hn; simp [hr] at hn
    split at hchk
    · cases hchk
    split at hchk
    · cases hchk
    rename_i hno
    apply hno
    refine Array.any_eq_true'.2 ⟨(f, c), hrun, ?_⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨by omega, ?_⟩
    unfold State.probation
    refine Array.any_eq_true'.2 ⟨a, Array.mem_range.2 ha, ?_⟩
    simp only [getElem!_pos s.appends a ha, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨⟨by omega, by omega⟩, hb⟩
  unfold Kept
  cases hor : s.appends[a].origin with
  | registered x => trivial
  | fresh =>
    obtain ⟨b, hbw⟩ := Option.isSome_iff_exists.1 (hS hr a ha).1
    simp [State.born, Array.getElem?_eq_getElem ha, hor, hbw] at hb
    exact ⟨b, hbw, by omega⟩
  | rekey k =>
    obtain ⟨b, hbw⟩ := Option.isSome_iff_exists.1 ((hS hr a ha).2 k hor)
    simp [State.born, Array.getElem?_eq_getElem ha, hor, hbw] at hb
    exact ⟨b, hbw, by omega⟩

theorem prob_start {s : State} {now : Nat} {r : Result} {s' : State} (h : Prob s)
    (he : start s now = (r, s')) : Prob s' := by
  unfold start at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact prob_releaseCoin h _ _

theorem prob_boundary {s : State} {now : Nat} {r : Result} {s' : State} (h : Prob s)
    (he : boundary s now = (r, s')) : Prob s' := by
  unfold boundary at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact h
    | exact prob_releaseCoin (prob_boundaryNs h _ _) _ _
    | exact prob_boundaryNs h _ _

theorem prob_append {s : State} {root : ByteArray} {count : Int} {origin : _root_.Poub.Origin} {slot : Option Nat}
    {r : Result} {s' : State} (hi : Inv s) (h : Prob s) (he : append s root count origin slot = .ok (r, s')) :
    Prob s' := by
  unfold append at he
  repeat' (first | split at he | (dsimp only at he; split at he))
  all_goals (try simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at he)
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact prob_fail h _ _ _
    | exact prob_push hi h ‹_› _ rfl rfl rfl _ _

theorem prob_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State} (h : Prob s)
    (he : release s runs withdrawn = (r, s')) : Prob s' := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact prob_fail h _ _ _

theorem prob_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State} (h : Prob s)
    (he : close s t now = .ok (r, s')) : Prob s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact prob_fail (prob_final h _) _ _ _
    | exact prob_sample (prob_final h _) _ _ _

theorem prob_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (h : Prob s) (he : verify s t openings = (r, s')) : Prob s' := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact prob_fail h _ _ _
    | exact prob_accept h _

theorem prob_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {step : Option Int} {now : Nat} {r : Result}
    {s' : State} (hi : Inv s) (h : Prob s) (he : receive s t i root runs change step now = (r, s')) : Prob s' := by
  have hH := held_of hi h
  unfold receive at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  split at he
  · cases he; exact prob_fail h _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Prob s₂ → Held s₂ → jp () s₂ = (r, s') → Prob s' := by
    intro s₂ h₂ hH₂ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have hW : ∀ plan, plan₂ = some plan → EndsOk plan := by
      intro plan hp
      dsimp only [plan₂, ext₂, N₂] at hp
      split at hp
      · cases hp
      · exact endsOk_changed hp
      · cases hp; exact endsOk_whole _
      · exact (make_spec hp).1
    have h₃ : Prob s₃ := h₂
    have hH₃ : Held s₃ := hH₂
    have key₂ : jp₂ () = (r, s') → Prob s' := by
      intro hj
      dsimp only [jp₂] at hj
      generalize hq : plan₂ = q at hj
      repeat' split at hj
      all_goals (try dsimp only at hj)
      all_goals cases hj
      all_goals first
        | exact prob_fail h₂ _ _ _
        | exact prob_fail h₃ _ _ _
        | exact ⟨h₃.1, fun hr => plansOn_add (h₃.2 hr) _ _ _ _ _
            (prob_check (s := s₃) (hW _ (by first | exact hq | exact hq.trans ‹_›)) ‹_› hr hH₃)⟩
    split at hj
    · split at hj
      · cases hj; exact prob_fail h₂ _ _ _
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · exact key s₁ (prob_final h _) hH he
  · split at he
    · cases he; exact prob_fail h _ _ _
    · exact key s₀ h hH he

theorem step_prob {s : State} {e : Event} {r : Result} {s' : State} (hi : Inv s) (h : Prob s)
    (he : step s e = .ok (r, s')) : Prob s' := by
  cases e with
  | start now => injection he with he; exact prob_start h he
  | boundary now => injection he with he; exact prob_boundary h he
  | append root count origin slot => exact prob_append hi h he
  | release runs withdrawn => injection he with he; exact prob_release h he
  | receive t i root runs change st now => injection he with he; exact prob_receive hi h he
  | close t now => exact prob_close h he
  | verify t openings => injection he with he; exact prob_verify h he

theorem probationKept : ProbationKept := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun hr
  have hc := run_inv (fun s => Inv s ∧ Prob s) (fun _ _ _ _ hs he => ⟨(step_inv hs.1 he).1, step_prob hs.1 hs.2 he⟩)
    _ _ _ _ ⟨inv_init _ _ _ _ _, prob_init _ _ _ _ _⟩ hrun
  obtain ⟨-, hp, -, -⟩ := inv_run hrun
  intro t ht i hi p hpl a ha h1 h2
  have := hc.2.2 (by rw [hp]; exact hr) t ht i hi p hpl a ha h1 h2
  rw [hp] at this
  revert this
  unfold Kept
  cases s.appends[a].origin <;> exact id

/-! ## `CoinAfterAppend` and `CoinBySlot` -/

/-- An event that leaves the appends as they were and answers with no `appended`. -/
def NoAppend (s s' : State) (r : Result) : Prop := s'.appends = s.appends ∧ ∀ first c, r ≠ .appended first c

theorem live_refused {s : State} {r : Result} (h : Poub.Session.live s = some r) :
    ∀ first c, r ≠ .appended first c := by
  unfold Poub.Session.live at h
  cases hf : s.failed with
  | none => rw [hf] at h; cases h
  | some f => rw [hf] at h; cases h; intro _ _ h; cases h

theorem quiet_start {s : State} {now : Nat} {r : Result} {s' : State} (he : start s now = (r, s')) : NoAppend s s' r := by
  unfold start State.releaseCoin at he
  repeat' split at he
  all_goals cases he
  all_goals exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_boundary {s : State} {now : Nat} {r : Result} {s' : State} (he : boundary s now = (r, s')) :
    NoAppend s s' r := by
  unfold boundary State.releaseCoin at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State}
    (he : release s runs withdrawn = (r, s')) : NoAppend s s' r := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (he : close s t now = .ok (r, s')) : NoAppend s s' r := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (he : verify s t openings = (r, s')) : NoAppend s s' r := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact ⟨rfl, live_refused ‹_›⟩
    | exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | exact ⟨rfl, fun _ _ h => by cases h⟩

theorem quiet_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {st : Option Int} {now : Nat} {r : Result}
    {s' : State} (he : receive s t i root runs change st now = (r, s')) : NoAppend s s' r := by
  unfold receive at he
  split at he
  · cases he; exact ⟨rfl, live_refused ‹_›⟩
  split at he
  · cases he; exact ⟨rfl, fun _ _ h => by cases h⟩
  split at he
  · cases he; exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, s₂.appends = s.appends → jp () s₂ = (r, s') → NoAppend s s' r := by
    intro s₂ e₂ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have key₂ : jp₂ () = (r, s') → NoAppend s s' r := by
      intro hj
      dsimp only [jp₂] at hj
      generalize plan₂ = q at hj
      repeat' split at hj
      all_goals (try dsimp only at hj)
      all_goals cases hj
      all_goals first
        | exact ⟨e₂, fun _ _ h => by cases h⟩
        | exact ⟨e₂, fun _ _ h => by simp [State.fail] at h⟩
    split at hj
    · split at hj
      · cases hj; exact ⟨e₂, fun _ _ h => by simp [State.fail] at h⟩
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · exact key s₁ rfl he
  · split at he
    · cases he; exact ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    · exact key s₀ rfl he

/-- An accepted append pushes one append, at the store's extent, in the slot it names (the appends before it without
one), with that slot's append coin, and answers with it; with refresh it names a slot. -/
theorem append_step {s : State} {root : ByteArray} {count : Int} {origin : _root_.Poub.Origin} {slot : Option Nat}
    {r : Result} {s' : State} (he : append s root count origin slot = .ok (r, s')) :
    NoAppend s s' r ∨ ∃ c, 1 ≤ count ∧ r = .appended s.extent c ∧
      s.appendCoins[slot.getD s.appends.size]? = some c ∧ (s.params.refresh = true → slot.isSome) ∧
      s'.appends = s.appends.push
        { first := s.extent, count := count.toNat, root, coin := c, origin, slot := slot.getD s.appends.size } := by
  unfold append at he
  repeat' (first | split at he | (dsimp only at he; split at he))
  all_goals (try simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at he)
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact Or.inl ⟨rfl, live_refused ‹_›⟩
    | exact Or.inl ⟨rfl, fun _ _ h => by simp [State.fail] at h⟩
    | (refine Or.inr ⟨_, ?_, rfl, ‹_›, ?_, rfl⟩
       · by_cases h : 1 ≤ count
         · exact h
         · exfalso; simp_all
       · intro hr
         cases slot with
         | none => exfalso; simp_all
         | some _ => rfl)

theorem step_appends {s : State} {e : Event} {r : Result} {s' : State} (he : step s e = .ok (r, s')) :
    NoAppend s s' r ∨ ∃ root count origin slot c, e = .append root count origin slot ∧ 1 ≤ count ∧
      r = .appended s.extent c ∧ s.appendCoins[slot.getD s.appends.size]? = some c ∧
      (s.params.refresh = true → slot.isSome) ∧
      s'.appends = s.appends.push
        { first := s.extent, count := count.toNat, root, coin := c, origin, slot := slot.getD s.appends.size } := by
  cases e with
  | start now => injection he with he; exact Or.inl (quiet_start he)
  | boundary now => injection he with he; exact Or.inl (quiet_boundary he)
  | append root count origin slot =>
    rcases append_step he with h | ⟨c, h1, h2, h3, h4, h5⟩
    · exact Or.inl h
    · exact Or.inr ⟨root, count, origin, slot, c, rfl, h1, h2, h3, h4, h5⟩
  | release runs withdrawn => injection he with he; exact Or.inl (quiet_release he)
  | receive t i root runs change st now => injection he with he; exact Or.inl (quiet_receive he)
  | close t now => exact Or.inl (quiet_close he)
  | verify t openings => injection he with he; exact Or.inl (quiet_verify he)

theorem appendedOf_push (rs : Array Result) (r : Result) :
    appendedOf (rs.push r) = appendedOf rs ++ appendedOf #[r] := by
  unfold appendedOf
  rw [← Array.toList_inj]
  simp [List.filterMap_append]

theorem appendedOf_quiet {r : Result} (h : ∀ first c, r ≠ .appended first c) : appendedOf #[r] = #[] := by
  unfold appendedOf
  rw [← Array.toList_inj]
  cases r <;> simp_all

theorem appendedOf_one (first : Nat) (c : ByteArray) : appendedOf #[.appended first c] = #[(first, c)] := by
  unfold appendedOf
  rw [← Array.toList_inj]
  simp

theorem extract_push_le' (rs : Array Result) (r : Result) {e : Nat} (he : e ≤ rs.size) :
    (rs.push r).extract 0 e = rs.extract 0 e := by
  rw [Array.extract_push, ite_eq_left he]

/-- What the results `rs` of the events `l`, reaching `s` with refresh `R`, on the append coins `AC`, say of each
`appended` result: its event is an append, recorded as the `k`-th (`k` the `appended` results before it), in the slot it
names (else `k`), with that slot's coin. -/
def Answered (R : Bool) (AC : Array ByteArray) (l : List Event) (rs : Array Result) (s : State) : Prop :=
  rs.size = l.length ∧ (appendedOf rs).size = s.appends.size ∧
  ∀ e first c, rs[e]? = some (.appended first c) →
    let k := (appendedOf (rs.extract 0 e)).size
    ∃ root count origin slot, l[e]? = some (.append root count origin slot) ∧ (R = true → slot.isSome) ∧
      ∃ hk : k < s.appends.size,
        s.appends[k].root = root ∧ (s.appends[k].count : Int) = count ∧ s.appends[k].origin = origin ∧
        s.appends[k].slot = slot.getD k ∧ s.appends[k].first = first ∧ s.appends[k].coin = c ∧
        AC[slot.getD k]? = some c

theorem answered_step {R : Bool} {AC : Array ByteArray} {l : List Event} {rs : Array Result} {s : State} {e : Event}
    {r : Result} {s' : State} (hR : s.params.refresh = R) (hAC : s.appendCoins = AC) (h : Answered R AC l rs s)
    (he : step s e = .ok (r, s')) : Answered R AC (l ++ [e]) (rs.push r) s' := by
  obtain ⟨hsz, hcnt, hold⟩ := h
  have hlen : (rs.push r).size = (l ++ [e]).length := by simp [hsz]
  rcases step_appends he with ⟨ha, hq⟩ | ⟨root, count, origin, slot, c, rfl, h1, rfl, hc, hsl, ha⟩
  · refine ⟨hlen, by rw [appendedOf_push, appendedOf_quiet hq, Array.append_empty, ha, hcnt], ?_⟩
    intro e' first c' he'
    rw [Array.getElem?_push] at he'
    split at he'
    · cases he'; exact absurd rfl (hq first c')
    · have hlt : e' < rs.size := by
        rcases Nat.lt_or_ge e' rs.size with hl | hl
        · exact hl
        · rw [Array.getElem?_eq_none (by omega)] at he'; cases he'
      obtain ⟨root, count, origin, slot, hl, hs, hk, h2⟩ := hold e' first c' he'
      dsimp only
      rw [extract_push_le' rs r (Nat.le_of_lt hlt), ha]
      exact ⟨root, count, origin, slot, by rw [List.getElem?_append_left (by omega)]; exact hl, hs, hk, h2⟩
  · have hk0 : (appendedOf rs).size = s.appends.size := hcnt
    refine ⟨hlen, by rw [appendedOf_push, appendedOf_one, ha]; simp [hcnt], ?_⟩
    intro e' first c' he'
    rw [Array.getElem?_push] at he'
    split at he'
    · rename_i hE
      subst hE
      cases he'
      dsimp only
      rw [extract_push_le' rs _ (Nat.le_refl _), Array.extract_size, ha]
      refine ⟨root, count, origin, slot, by rw [List.getElem?_append_right (by omega)]; simp [hsz],
        fun hr => hsl (hR.trans hr), by simp only [Array.size_push]; omega, ?_⟩
      simp only [hk0, Array.getElem_push_eq]
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      all_goals first | trivial | rfl | omega | (rw [← hAC]; exact hc) | (simp only [Int.toNat_of_nonneg (by omega : (0 : Int) ≤ count)])
    · have hlt : e' < rs.size := by
        rcases Nat.lt_or_ge e' rs.size with hl | hl
        · exact hl
        · rw [Array.getElem?_eq_none (by omega)] at he'; cases he'
      obtain ⟨root', count', origin', slot', hl, hs, hk, h2⟩ := hold e' first c' he'
      dsimp only
      rw [extract_push_le' rs _ (Nat.le_of_lt hlt), ha]
      refine ⟨root', count', origin', slot', by rw [List.getElem?_append_left (by omega)]; exact hl, hs,
        by simp only [Array.size_push]; omega, ?_⟩
      simp only [Array.getElem_push_lt hk]
      exact h2

theorem answered_run {R : Bool} {AC : Array ByteArray} :
    ∀ (l pre : List Event) (acc : Array Result × State), Inv acc.2 → acc.2.params.refresh = R →
      acc.2.appendCoins = AC → Answered R AC pre acc.1 acc.2 →
      ∀ out, List.foldlM (fun (acc : Array Result × State) e => do
        let (r, s') ← step acc.2 e
        pure (acc.1.push r, s')) acc l = .ok out → Answered R AC (pre ++ l) out.1 out.2 := by
  intro l
  induction l with
  | nil => intro pre acc _ _ _ h out ho; simp only [List.foldlM_nil] at ho; cases ho; simpa using h
  | cons e l ih =>
    intro pre acc hi hR hAC h out ho
    rw [List.foldlM_cons] at ho
    cases he : step acc.2 e with
    | error err => rw [he] at ho; cases ho
    | ok p =>
      rw [he] at ho
      obtain ⟨r, s''⟩ := p
      obtain ⟨hi', fp, -, fa⟩ := step_inv hi he
      have := ih (pre ++ [e]) (acc.1.push r, s'') hi' (by rw [fp]; exact hR) (fa.trans hAC) (answered_step hR hAC h he) out ho
      simpa using this

theorem answered_of_run {params : Params} {reg : Registration} {coins appendCoins : Array ByteArray}
    {notBeforeNs : Nat} {events : Array Event} {rs : Array Result} {s : State}
    (hrun : run (State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s)) :
    Answered params.refresh appendCoins events.toList rs s := by
  unfold run at hrun
  rw [← Array.foldlM_toList] at hrun
  simpa using answered_run (R := params.refresh) (AC := appendCoins) events.toList []
    (#[], State.init params reg coins appendCoins notBeforeNs) (inv_init _ _ _ _ _) rfl rfl
    ⟨rfl, by simp [appendedOf, State.init], fun e _ _ h => by simp at h⟩ (rs, s) hrun

theorem coinAfterAppend : CoinAfterAppend := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun
  obtain ⟨hsz, -, hold⟩ := answered_of_run hrun
  refine ⟨by simpa using hsz, fun e first c he => ?_⟩
  obtain ⟨root, count, origin, slot, hl, -, hk, h2⟩ := hold e first c he
  exact ⟨root, count, origin, slot, by simpa using hl, hk, h2⟩

theorem coinBySlot : CoinBySlot := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun hr e first c he
  obtain ⟨-, -, hold⟩ := answered_of_run hrun
  obtain ⟨root, count, origin, slot, hl, hs, hk, h1, h2, h3, h4, h5, h6, h7⟩ := hold e first c he
  obtain ⟨σ, rfl⟩ := Option.isSome_iff_exists.1 (hs hr)
  exact ⟨root, count, origin, σ, by simpa using hl, hk, h4, h5, h6, h7⟩

/-! ## `OriginsOnRota` -/

/-- `Params.slotAt`'s loop over the windows `ws`, its first slot `start`. -/
def slotScan (p : Params) (N k : Nat) : List Nat → Nat → Option (Nat × Slot)
  | [], _ => none
  | w :: ws, start =>
    if k < start + p.windowSize N w then some (w, p.slotIn N w (k - start))
    else slotScan p N k ws (start + p.windowSize N w)

/-- Where `Params.slotAt`'s loop stops: the first slot of the window holding `k`, or past the last. -/
def scanStop (p : Params) (N k : Nat) : List Nat → Nat → Nat
  | [], start => start
  | w :: ws, start => if k < start + p.windowSize N w then start else scanStop p N k ws (start + p.windowSize N w)

theorem forIn_scan (p : Params) (N k : Nat) : ∀ (l : List Nat) (start : Nat),
    forIn (m := Id) l ((none : Option (Option (Nat × Slot))), start) (fun w (r : Option (Option (Nat × Slot)) × Nat) =>
      if k < r.2 + p.windowSize N w then pure (ForInStep.done (some (some (w, p.slotIn N w (k - r.2))), r.2))
      else pure (ForInStep.yield (none, r.2 + p.windowSize N w))) =
      ((slotScan p N k l start).map some, scanStop p N k l start)
  | [], start => rfl
  | w :: ws, start => by
    by_cases h : k < start + p.windowSize N w
    · simp [List.forIn_cons, slotScan, scanStop, h]; exact rfl
    · simp only [List.forIn_cons, slotScan, scanStop, h, ite_false]
      exact forIn_scan p N k ws _

theorem slotAt_eq (p : Params) (N k : Nat) : p.slotAt N k = slotScan p N k (List.range' 0 p.windows) 0 := by
  unfold Params.slotAt
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel,
    Nat.div_one, forIn_scan]
  cases slotScan p N k (List.range' 0 p.windows) 0 <;> rfl

theorem windowStart_succ (p : Params) (N w : Nat) :
    p.windowStart N (w + 1) = p.windowStart N w + p.windowSize N w := by
  simp [Params.windowStart, List.range_succ, List.map_append, List.sum_append]

theorem windowStart_mono (p : Params) (N : Nat) {a b : Nat} (h : a ≤ b) : p.windowStart N a ≤ p.windowStart N b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => rw [windowStart_succ]; omega

theorem slotScan_range (p : Params) (N k w : Nat) (x : Slot) : ∀ n a, p.windowStart N a ≤ k →
    (slotScan p N k (List.range' a n) (p.windowStart N a) = some (w, x) ↔
      a ≤ w ∧ w < a + n ∧ p.windowStart N w ≤ k ∧ k < p.windowStart N (w + 1) ∧
        x = p.slotIn N w (k - p.windowStart N w)) := by
  intro n
  induction n with
  | zero => intro a _; simp [slotScan]; omega
  | succ n ih =>
    intro a ha
    rw [List.range'_succ]
    simp only [slotScan]
    by_cases hk : k < p.windowStart N a + p.windowSize N a
    · rw [if_pos hk]
      constructor
      · intro h
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨Nat.le_refl _, by omega, ha, by rw [windowStart_succ]; exact hk, rfl⟩
      · rintro ⟨h1, -, h3, -, rfl⟩
        have : w = a := by
          rcases Nat.eq_or_lt_of_le h1 with h | h
          · exact h.symm
          · have := windowStart_mono p N (Nat.succ_le_of_lt h)
            rw [windowStart_succ] at this
            omega
        subst this
        rfl
    · rw [if_neg hk, ← windowStart_succ]
      rw [ih (a + 1) (by rw [windowStart_succ]; omega)]
      constructor
      · rintro ⟨h1, h2, h3⟩; exact ⟨by omega, by omega, h3⟩
      · rintro ⟨h1, h2, h3, h4, h5⟩
        refine ⟨?_, by omega, h3, h4, h5⟩
        rcases Nat.eq_or_lt_of_le h1 with h | h
        · subst h; rw [windowStart_succ] at h4; omega
        · omega

/-- Slot `k` of the schedule is in window `w` (`w < windows`, `k` among its slots), of the kind `slotIn` gives it. -/
theorem slotAt_iff (p : Params) (N k w : Nat) (x : Slot) :
    p.slotAt N k = some (w, x) ↔ w < p.windows ∧ p.windowStart N w ≤ k ∧ k < p.windowStart N (w + 1) ∧
      x = p.slotIn N w (k - p.windowStart N w) := by
  rw [slotAt_eq]
  have h0 : p.windowStart N 0 = 0 := by simp [Params.windowStart]
  have := slotScan_range p N k w x p.windows 0 (by rw [h0]; exact Nat.zero_le _)
  rw [h0] at this
  rw [this]
  constructor
  · rintro ⟨-, h2, h3⟩; exact ⟨by omega, h3⟩
  · rintro ⟨h2, h3⟩; exact ⟨Nat.zero_le _, by omega, h3⟩

/-- A re-key slot's origin is a fresh slot of the schedule: re-key slot `rekey k` of window `w` names fresh slot
`q mod freshSlots` of window `w − n·L`, `n ≥ 1`. -/
theorem slotAt_rekey {p : Params} {N σ w k : Nat} (h : p.slotAt N σ = some (w, .rekey k)) :
    ∃ b, p.slotAt N k = some (b, .fresh) := by
  obtain ⟨hw, h1, h2, h3⟩ := (slotAt_iff p N σ w _).1 h
  rw [windowStart_succ] at h2
  unfold Params.slotIn at h3
  dsimp only at h3
  generalize hr : (if p.hasRegistered N w then 1 else 0) = r at h3
  generalize hm : σ - p.windowStart N w = m at h3
  by_cases c1 : m < r
  · rw [ite_cond_eq_true _ _ (eq_true c1)] at h3
    generalize p.classRun N (w % p.lifetime) = c at h3
    cases c
    cases h3
  rw [ite_cond_eq_false _ _ (eq_false c1)] at h3
  by_cases c2 : m < r + p.freshSlots * (w / p.lifetime)
  · rw [ite_cond_eq_true _ _ (eq_true c2)] at h3
    injection h3 with h3
    subst h3
    have hF : 0 < p.freshSlots := by
      rcases Nat.eq_zero_or_pos p.freshSlots with h0 | h0
      · rw [h0, Nat.zero_mul] at c2; omega
      · exact h0
    generalize hb : w - (w / p.lifetime - (m - r) / p.freshSlots) * p.lifetime = b
    generalize hi : (m - r) % p.freshSlots = i
    have hiF : i < p.freshSlots := by rw [← hi]; exact Nat.mod_lt _ hF
    have hbw : b < p.windows := by omega
    have hsz : p.freshSlots ≤ p.windowSize N b := by unfold Params.windowSize; omega
    refine ⟨b, ?_⟩
    rw [slotAt_iff]
    unfold Params.freshSlot
    refine ⟨hbw, by omega, by rw [windowStart_succ]; omega, ?_⟩
    unfold Params.slotIn Params.windowSize
    dsimp only
    generalize (if p.hasRegistered N b then 1 else 0) = rb
    rw [ite_cond_eq_false _ _ (eq_false (by omega)), ite_cond_eq_false _ _ (eq_false (by omega))]
  · rw [ite_cond_eq_false _ _ (eq_false c2)] at h3; cases h3

/-- What `OriginsOnRota` reads of every reached state: with refresh each append is `OnRota` and each joined window is
its slot's; no more windows joined than appends; a drawn window is final; at most `max 1 windows` windows, each but the
last ended; and once the last window is drawn, every append has joined. -/
structure RotaOn (P : Params) (N : Nat) (as : Array Append) (js : Array Nat) (ws : Array Win) : Prop where
  onRota : P.refresh = true → ∀ a, a < as.size → OnRota P N as a
  le : js.size ≤ as.size
  win : P.refresh = true → ∀ a, a < js.size → js[a]? = (P.slotAt N (as[a]! : Append).slot).map (·.1)
  final : ∀ u, (ws[u]! : Win).drawn.isSome → (ws[u]! : Win).final = true
  size : ws.size ≤ max 1 P.windows
  bnd : ∀ u, u + 1 < ws.size → (ws[u]! : Win).boundaryNs.isSome
  last : ∀ u, u + 1 = P.windows → (ws[u]! : Win).drawn.isSome → js.size = as.size

def Rota (s : State) : Prop := RotaOn s.params s.reg.blocks s.appends s.joined s.windows

theorem drawn_past (ws : Array Win) (u : Nat) (h : ¬u < ws.size) : ws[u]!.drawn = none := by
  rw [getElem!_neg ws u h]; rfl

theorem rota_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Rota (State.init params reg coins appendCoins notBeforeNs) := by
  refine ⟨fun _ a ha => by simp [State.init] at ha, by simp [State.init], fun _ a ha => by simp [State.init] at ha,
    fun u hu => ?_, by simp [State.init], fun u hu => by simp [State.init] at hu, fun u _ hu => ?_⟩
  · rw [drawn_past _ _ (by simp [State.init])] at hu; cases hu
  · rw [drawn_past _ _ (by simp [State.init])] at hu; cases hu

/-- `ws'` is `ws` with each window as drawn as it was, final if it was, and ended if it was. -/
def WK (ws ws' : Array Win) : Prop :=
  ws'.size = ws.size ∧ ∀ u, (ws'[u]! : Win).drawn = (ws[u]! : Win).drawn ∧
    ((ws[u]! : Win).final = true → (ws'[u]! : Win).final = true) ∧
    ((ws[u]! : Win).boundaryNs.isSome → (ws'[u]! : Win).boundaryNs.isSome)

theorem rotaOn_wk {P N as js ws ws'} (h : RotaOn P N as js ws) (k : WK ws ws') : RotaOn P N as js ws' :=
  ⟨h.onRota, h.le, h.win, fun u hu => (k.2 u).2.1 (h.final u (by rw [← (k.2 u).1]; exact hu)),
    by rw [k.1]; exact h.size, fun u hu => (k.2 u).2.2 (h.bnd u (by rw [← k.1]; exact hu)),
    fun u hw hu => h.last u hw (by rw [← (k.2 u).1]; exact hu)⟩

theorem rota_getElem!_modify (ws : Array Win) (u t : Nat) (f : Win → Win) :
    (ws.modify u f)[t]! = if u = t ∧ t < ws.size then f ws[t]! else ws[t]! := by
  by_cases ht : t < ws.size
  · rw [getElem!_pos (ws.modify u f) t (by simpa using ht), getElem!_pos ws t ht, Array.getElem_modify]
    by_cases hu : u = t <;> simp [hu, ht]
  · rw [getElem!_neg (ws.modify u f) t (by simpa using ht), getElem!_neg ws t ht]
    simp [ht]

theorem wk_modify (ws : Array Win) (u : Nat) (f : Win → Win)
    (hf : ∀ w : Win, (f w).drawn = w.drawn ∧ (w.final = true → (f w).final = true) ∧
      (w.boundaryNs.isSome → (f w).boundaryNs.isSome)) : WK ws (ws.modify u f) := by
  refine ⟨by simp, fun t => ?_⟩
  rw [rota_getElem!_modify]
  split
  · exact hf _
  · exact ⟨rfl, id, id⟩

theorem rota_fail {s : State} (h : Rota s) (t : Nat) (f : Failure) (at? : Option (Nat × Nat)) :
    Rota (s.fail t f at?).2 := by
  unfold State.fail
  dsimp only
  split
  · exact rotaOn_wk h (wk_modify _ _ _ fun _ => ⟨rfl, id, id⟩)
  · exact h

theorem rota_final {s : State} (h : Rota s) (u : Nat) :
    Rota { s with windows := s.windows.modify u ({· with final := true}) } :=
  rotaOn_wk h (wk_modify _ _ _ fun _ => ⟨rfl, fun _ => rfl, id⟩)

/-- A new window, not drawn, past windows that have all ended, within `max 1 windows`. -/
theorem rotaOn_push {P N as js ws} (h : RotaOn P N as js ws) (w : Win) (hd : w.drawn = none)
    (hs : ws.size + 1 ≤ max 1 P.windows) (hb : ∀ u, u < ws.size → ws[u]!.boundaryNs.isSome) :
    RotaOn P N as js (ws.push w) := by
  have g : ∀ u, (ws.push w)[u]!.drawn = ws[u]!.drawn := by
    intro u
    by_cases hl : u < ws.size
    · rw [getElem!_pos (ws.push w) u (by simp only [Array.size_push]; omega), getElem!_pos ws u hl,
        Array.getElem_push_lt hl]
    · rw [drawn_past ws u hl]
      by_cases he : u = ws.size
      · subst he; rw [getElem!_pos (ws.push w) ws.size (by simp), Array.getElem_push_eq, hd]
      · exact drawn_past _ _ (by simp only [Array.size_push]; omega)
  refine ⟨h.onRota, h.le, h.win, fun u hu => ?_, by simpa using hs, fun u hu => ?_,
    fun u hw hu => h.last u hw (by rw [← g u]; exact hu)⟩
  · have hu' := hu
    rw [g u] at hu'
    have hl : u < ws.size := Decidable.byContradiction fun hl => by rw [drawn_past ws u hl] at hu'; cases hu'
    have := h.final u hu'
    rw [getElem!_pos ws u hl] at this
    rw [getElem!_pos (ws.push w) u (by simp only [Array.size_push]; omega), Array.getElem_push_lt hl]
    exact this
  · simp only [Array.size_push] at hu
    have hl : u < ws.size := by omega
    have := hb u hl
    rw [getElem!_pos ws u hl] at this
    rw [getElem!_pos (ws.push w) u (by simp only [Array.size_push]; omega), Array.getElem_push_lt hl]
    exact this

/-- `close`'s draw of window `t`, made final first; the last window only once every append has joined. -/
theorem rotaOn_sample {P N as js ws} (h : RotaOn P N as js ws) (t : Nat) (d : Array (Nat × Nat)) (now : Nat)
    (hl : t + 1 = P.windows → as.size ≤ js.size) :
    RotaOn P N as js ((ws.modify t ({· with final := true})).modify t
      ({· with drawn := some d, openedNs := some now})) := by
  have e : ∀ u, ((ws.modify t ({· with final := true})).modify t
      ({· with drawn := some d, openedNs := some now}))[u]! =
      if t = u ∧ u < ws.size then { ({ ws[u]! with final := true } : Win) with drawn := some d, openedNs := some now }
      else ws[u]! := by
    intro u
    rw [rota_getElem!_modify, Array.size_modify]
    by_cases hc : t = u ∧ u < ws.size
    · rw [if_pos hc, if_pos hc, rota_getElem!_modify, if_pos hc]
    · rw [if_neg hc, if_neg hc, rota_getElem!_modify, if_neg hc]
  refine ⟨h.onRota, h.le, h.win, fun u hu => ?_, by simp only [Array.size_modify]; exact h.size, fun u hu => ?_,
    fun u hw hu => ?_⟩
  · rw [e u] at hu ⊢
    by_cases hc : t = u ∧ u < ws.size
    · rw [if_pos hc]
    · rw [if_neg hc] at hu ⊢; exact h.final u hu
  · simp only [Array.size_modify] at hu
    rw [e u]
    by_cases hc : t = u ∧ u < ws.size
    · rw [if_pos hc]; exact h.bnd u hu
    · rw [if_neg hc]; exact h.bnd u hu
  · by_cases hc : t = u ∧ u < ws.size
    · obtain ⟨rfl, -⟩ := hc; exact Nat.le_antisymm h.le (hl hw)
    · rw [e u, if_neg hc] at hu; exact h.last u hw hu

theorem close_last {a b c d : Nat} (h : ¬(a + 1 == b && decide (c < d)) = true) : a + 1 = b → d ≤ c := by
  intro e; simp [e] at h; omega

theorem onRota_push {P : Params} {N : Nat} {as : Array Append} {a : Nat} (x : Append) (h : OnRota P N as a) :
    OnRota P N (as.push x) a := by
  unfold OnRota at h ⊢
  rcases Nat.lt_or_ge a as.size with ha | ha
  · rw [Array.getElem?_push, if_neg (by omega)]
    cases hy : as[a]? with
    | none => rw [hy] at h; exact h.elim
    | some y =>
      rw [hy] at h
      dsimp only at h ⊢
      rcases hs : P.slotAt N y.slot with _ | ⟨w, _ | ⟨lo, n⟩ | k⟩ <;> cases ho : y.origin <;>
        simp only [hs, ho] at h ⊢
      all_goals first
        | exact h
        | (obtain ⟨hk, b, hb, z, hz, rest⟩ := h
           exact ⟨hk, b, hb, z, by rw [Array.getElem?_push, if_neg (by omega)]; exact hz, rest⟩)
  · rw [Array.getElem?_eq_none ha] at h; exact h.elim

/-- With refresh, the append `append` takes in slot `k` is `OnRota`: `State.slotOk` checks its slot's kind, and a re-key
slot's origin is a fresh slot (`slotAt_rekey`), so the append `find?` gives there is fresh by its own `OnRota`. -/
theorem onRota_new {s : State}
    (hR : s.params.refresh = true → ∀ a, a < s.appends.size → OnRota s.params s.reg.blocks s.appends a)
    (hr : s.params.refresh = true) {k n : Nat} {origin : _root_.Poub.Origin} (hk : s.slotOk k n origin = true)
    (x : Append) (hxs : x.slot = k) (hxc : x.count = n) (hxo : x.origin = origin) :
    OnRota s.params s.reg.blocks (s.appends.push x) s.appends.size := by
  unfold State.slotOk at hk
  rw [Bool.and_eq_true] at hk
  obtain ⟨-, hk⟩ := hk
  unfold OnRota
  rw [Array.getElem?_push, if_pos rfl]
  dsimp only
  rw [hxs, hxo, hxc]
  rcases hs : s.params.slotAt s.reg.blocks k with _ | ⟨w, _ | ⟨lo, m⟩ | k₀⟩ <;> cases origin <;>
    simp only [hs] at hk ⊢
  all_goals first
    | (simp at hk; done)
    | skip
  · simpa using hk
  · simpa using hk
  · rename_i k'
    simp only [Bool.and_eq_true, beq_iff_eq, Option.any_eq_true] at hk
    obtain ⟨hkk, y, hy, hyc⟩ := hk
    have hys := Array.find?_some (p := fun x : Append => x.slot == k₀) hy
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_of_find?_eq_some hy)
    simp only [beq_iff_eq] at hys hyc
    obtain ⟨b', hb'⟩ := slotAt_rekey hs
    have hj' := hR hr j hj
    unfold OnRota at hj'
    rw [Array.getElem?_eq_getElem hj] at hj'
    dsimp only at hj'
    rw [hys, hb'] at hj'
    have ho : s.appends[j].origin = .fresh := by
      revert hj'; cases s.appends[j].origin <;> simp
    exact ⟨hkk.symm, j, hj, s.appends[j], by rw [Array.getElem?_push, if_neg (by omega), Array.getElem?_eq_getElem hj],
      hys, ho, hyc⟩

/-- An append `append` takes: `OnRota` with refresh, and never once the last window is drawn, since then the windows
are all released and the last is final, where `append` refuses with `coin`. -/
theorem rota_push {s : State} (h : Rota s) {A : Bool} {count : Nat} {origin : _root_.Poub.Origin}
    {slot : Option Nat} (h2 : ¬(s.params.refresh && !(slot.any fun k => s.slotOk k count origin)) = true)
    (h3 : ¬(A || (s.windows.size == s.params.windows && (s.windows.back?.map (·.final)).getD false)) = true)
    (x : Append) (hxs : x.slot = slot.getD s.appends.size) (hxc : x.count = count) (hxo : x.origin = origin)
    (hd : Option ByteArray) (n : Nat) :
    Rota { s with appends := s.appends.push x, head := hd, extent := n } := by
  have hno : ∀ u, u + 1 = s.params.windows → s.windows[u]!.drawn.isSome → False := by
    intro u hw hu
    have hl : u < s.windows.size := Decidable.byContradiction fun hl => by rw [drawn_past _ _ hl] at hu; cases hu
    have hf := h.final u hu
    have hsz := h.size
    have he : s.windows.size = s.params.windows := by omega
    have hb : s.windows.back? = some s.windows[u]! := by
      rw [Array.back?_eq_getElem?, show s.windows.size - 1 = u by omega, getElem!_pos s.windows u hl,
        Array.getElem?_eq_getElem hl]
    apply h3
    simp [he, hb, hf]
  have hle := h.le
  refine ⟨fun hr a ha => ?_, ?_, fun hr a ha => ?_, h.final, h.size, h.bnd, fun u hw hu => (hno u hw hu).elim⟩
  · change a < (s.appends.push x).size at ha
    simp only [Array.size_push] at ha
    by_cases hl : a < s.appends.size
    · exact onRota_push x (h.onRota hr a hl)
    · have : a = s.appends.size := by omega
      subst this
      rw [hr] at h2
      cases slot with
      | none => simp at h2
      | some k =>
        have hk : s.slotOk k count origin = true := by simpa using h2
        exact onRota_new h.onRota hr hk x (by simpa using hxs) hxc hxo
  · show s.joined.size ≤ (s.appends.push x).size
    simp only [Array.size_push]; omega
  · show s.joined[a]? = (s.params.slotAt s.reg.blocks (s.appends.push x)[a]!.slot).map (·.1)
    have hl : a < s.appends.size := Nat.lt_of_lt_of_le ha hle
    rw [h.win hr a ha, getElem!_pos (s.appends.push x) a (by simp only [Array.size_push]; omega),
      getElem!_pos s.appends a hl, Array.getElem_push_lt hl]

/-- `receive`'s join, past `joinsAnother`: every append not joined yet joins window `filling`, its slot's. -/
theorem rota_join {s : State} (h : Rota s) (hj : ¬s.joinsAnother s.filling = true) :
    Rota { s with joined := s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling } := by
  have hle := h.le
  refine ⟨h.onRota, ?_, fun hr a ha => ?_, h.final, h.size, h.bnd, fun _ _ _ => ?_⟩
  · show (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size ≤ s.appends.size
    simp only [Array.size_append, Array.size_replicate]; omega
  · show (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling)[a]? =
      (s.params.slotAt s.reg.blocks s.appends[a]!.slot).map (·.1)
    change a < (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size at ha
    simp only [Array.size_append, Array.size_replicate] at ha
    by_cases hl : a < s.joined.size
    · rw [Array.getElem?_append_left hl]; exact h.win hr a hl
    · rw [Array.getElem?_append_right (by omega), Array.getElem?_replicate, if_pos (by omega)]
      unfold State.joinsAnother at hj
      rw [hr] at hj
      simp only [Bool.true_and, List.any_eq_true, not_exists, not_and] at hj
      have := hj a (List.mem_range'_1.2 ⟨by omega, by omega⟩)
      have e : (s.params.slotAt s.reg.blocks s.appends[a]!.slot).map (·.1) = some s.filling := by
        simpa [State.slotWindow] using this
      exact e.symm
  · show (s.joined ++ Array.replicate (s.appends.size - s.joined.size) s.filling).size = s.appends.size
    simp only [Array.size_append, Array.size_replicate]; omega

theorem rota_start {s : State} {now : Nat} {r : Result} {s' : State} (h : Rota s)
    (he : start s now = (r, s')) : Rota s' := by
  unfold start at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  · rename_i h1 _
    cases he
    have he : s.windows.size = 0 := by simpa using h1
    exact rotaOn_push h _ rfl (by rw [he]; omega) (fun u hu => by omega)

theorem rota_boundary {s : State} {now : Nat} {r : Result} {s' : State} (h : Rota s)
    (he : boundary s now = (r, s')) : Rota s' := by
  unfold boundary at he
  simp only [Id.run] at he
  split at he
  · cases he; exact h
  split at he
  · rename_i t hf
    have hmem := Array.mem_of_find?_eq_some hf
    have hnone := Array.find?_some hf
    rw [Array.mem_range] at hmem
    have hlast : t + 1 = s.windows.size := Decidable.byContradiction fun hne => by
      have := h.bnd t (by omega)
      revert this hnone
      try simp only []
      cases s.windows[t]!.boundaryNs <;> simp
    have hb' : ∀ u, u < s.windows.size →
        (s.windows.modify t ({· with boundaryNs := some now}))[u]!.boundaryNs.isSome := by
      intro u hu
      rw [rota_getElem!_modify]
      split
      · rfl
      · exact h.bnd u (by omega)
    split at he
    · cases he; exact h
    split at he
    · rename_i hc
      have hc' : t + 1 < s.params.windows := hc
      cases he
      exact rotaOn_push (rotaOn_wk h (wk_modify _ _ _ fun _ => ⟨rfl, id, fun _ => rfl⟩)) _ rfl
        (by simp only [Array.size_modify]; omega) (by simpa only [Array.size_modify] using hb')
    · cases he; exact rotaOn_wk h (wk_modify _ _ _ fun _ => ⟨rfl, id, fun _ => rfl⟩)
  · cases he; exact h

theorem rota_append {s : State} {root : ByteArray} {count : Int} {origin : _root_.Poub.Origin} {slot : Option Nat}
    {r : Result} {s' : State} (h : Rota s) (he : append s root count origin slot = .ok (r, s')) : Rota s' := by
  unfold append at he
  repeat' (first | split at he | (dsimp only at he; split at he))
  all_goals (try simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at he)
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rota_fail h _ _ _
    | exact rota_push h ‹_› ‹_› _ rfl rfl rfl _ _

theorem rota_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State} (h : Rota s)
    (he : release s runs withdrawn = (r, s')) : Rota s' := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rota_fail h _ _ _

theorem rota_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State} (h : Rota s)
    (he : close s t now = .ok (r, s')) : Rota s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rota_fail (rota_final h _) _ _ _
    | exact rotaOn_sample h _ _ _ (close_last ‹_›)

theorem rota_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (h : Rota s) (he : verify s t openings = (r, s')) : Rota s' := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact h
    | exact rota_fail h _ _ _
    | exact rotaOn_wk h (wk_modify _ _ _ fun _ => ⟨rfl, id, id⟩)

theorem rota_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {step : Option Int} {now : Nat} {r : Result}
    {s' : State} (h : Rota s) (he : receive s t i root runs change step now = (r, s')) : Rota s' := by
  unfold receive at he
  split at he
  · cases he; exact h
  split at he
  · cases he; exact h
  split at he
  · cases he; exact rota_fail h _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Rota s₂ → jp () s₂ = (r, s') → Rota s' := by
    intro s₂ h₂ hj
    dsimp -zeta only [jp] at hj
    extract_lets t₂ N₂ ext₂ plan₂ s₃ below₂ head₂ jp₂ at hj
    have key₂ : jp₂ () = (r, s') → Rota s' := by
      intro hj
      dsimp only [jp₂] at hj
      generalize hq : plan₂ = q at hj
      repeat' split at hj
      all_goals (try dsimp only at hj)
      all_goals cases hj
      all_goals first
        | exact rota_fail h₂ _ _ _
        | exact rota_fail (rota_join h₂ ‹_›) _ _ _
        | exact rotaOn_wk (rota_join h₂ ‹_›) (wk_modify _ _ _ fun _ => ⟨rfl, id, id⟩)
    split at hj
    · split at hj
      · cases hj; exact rota_fail h₂ _ _ _
      · exact key₂ hj
    · exact key₂ hj
  split at he
  · exact key s₁ (rotaOn_wk h (wk_modify _ _ _ fun _ => ⟨rfl, fun _ => rfl, id⟩)) he
  · split at he
    · cases he; exact rota_fail h _ _ _
    · exact key s₀ h he

theorem step_rota {s : State} {e : Event} {r : Result} {s' : State} (h : Rota s)
    (he : step s e = .ok (r, s')) : Rota s' := by
  cases e with
  | start now => injection he with he; exact rota_start h he
  | boundary now => injection he with he; exact rota_boundary h he
  | append root count origin slot => exact rota_append h he
  | release runs withdrawn => injection he with he; exact rota_release h he
  | receive t i root runs change st now => injection he with he; exact rota_receive h he
  | close t now => exact rota_close h he
  | verify t openings => injection he with he; exact rota_verify h he

theorem slots_ordered {as : Array Append} (h : ∀ k (hk : k + 1 < as.size), as[k].slot < as[k + 1].slot) :
    ∀ a b (hb : b < as.size) (hab : a < b), as[a].slot < as[b].slot := by
  intro a b hb hab
  induction b with
  | zero => omega
  | succ b ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 hab with hl | hl
    · exact Nat.lt_trans (ih (by omega) hl) (h b hb)
    · subst hl; exact h a hb

theorem originsOnRota : OriginsOnRota := by
  intro params reg coins appendCoins notBeforeNs events rs s hrun hr hout a ha
  have hR : Rota s := run_inv Rota (fun _ _ _ _ hs he => step_rota hs he) _ _ _ _ (rota_init _ _ _ _ _) hrun
  obtain ⟨hI, hp, hg, -⟩ := inv_run hrun
  subst hp hg
  unfold State.outcome at hout
  split at hout
  · rename_i f _
    have := Option.some.inj hout
    cases hf : f.failure <;> simp [hf, Failure.name] at this
  split at hout
  swap
  · cases hout
  rename_i hc
  simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true] at hc
  obtain ⟨hsz, hall⟩ := hc
  have hon := hR.onRota hr a ha
  have hw : 0 < s.params.windows := by
    have hon' := hon
    unfold OnRota at hon'
    rw [Array.getElem?_eq_getElem ha] at hon'
    dsimp only at hon'
    revert hon'
    cases hs : s.params.slotAt s.reg.blocks s.appends[a].slot with
    | none => cases s.appends[a].origin <;> simp
    | some p => intro _; obtain ⟨w, x⟩ := p; exact Nat.lt_of_le_of_lt (Nat.zero_le _) ((slotAt_iff _ _ _ _ _).1 hs).1
  have hu : s.params.windows - 1 < s.windows.size := by omega
  have hd := hI.drawn _ hu (verdict_beq (hall _ hu))
  have hj := hR.last (s.params.windows - 1) (by omega) (by rw [getElem!_pos s.windows _ hu]; exact hd)
  have hle := hR.le
  refine ⟨hon, ?_, fun b hb hab => slots_ordered hI.ordered a b hb hab⟩
  unfold State.slotWindow
  rw [hR.win hr a (by omega), getElem!_pos s.appends a ha]

end Properties.Poub
