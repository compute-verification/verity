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

end Properties.Poub
