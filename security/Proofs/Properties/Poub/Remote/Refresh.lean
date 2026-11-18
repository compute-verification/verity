module

public import Proofs.Properties.Poub.Remote.Accept
public import Properties.Poub.Remote.Refresh
import Proofs.Properties.Poub.Remote.Released

/-!
# Proofs of refresh's code-side properties

`Properties.Poub.Remote.Refresh`'s `OpensOrigin`: an accepted window passes `checkOpening` on each sampled opening
(`checkWindow_accept`), and `checkOpening_origin` reads the origin's checks off its early returns.

`RegisteredCapped`, as `LifetimeCapped` is proved (`Lifetime`): the invariant `Cap` says that with refresh every block
below `N` a recorded pass of window `t` reads has `t < lifetime`; `receive` adds a pass only past `refreshRefusal`,
whose `lifetime` arm refuses a plan of window `t ≥ lifetime` with a run starting below `N` (`cap_check`), and a block a
plan reads lies in one of its runs (`block_mem`).
-/

@[expose] public section

open Poub Poub.Session

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

/-- Every block below `N` a recorded pass of window `t` reads has `t < L`. -/
def CapOn (N L : Nat) (ws : Array Win) : Prop :=
  ∀ t (ht : t < ws.size) i (hi : i < ws[t].plans.size) p, p < ws[t].plans[i].length →
    ws[t].plans[i].block p < N → t < L

def Cap (s : State) : Prop := s.params.refresh = true → CapOn s.reg.blocks s.params.lifetime s.windows

theorem cap_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    Cap (State.init params reg coins appendCoins notBeforeNs) :=
  fun _ t ht => by simp [State.init] at ht

theorem capOn_same {N L ws} {ws' : Array Win} (h : CapOn N L ws) (hs : SamePlans ws ws') : CapOn N L ws' := by
  obtain ⟨hsz, hw⟩ := hs
  have e : ∀ t (h' : t < ws'.size), ws'[t].plans = ws[t].plans := fun t h' => hw t (by omega) h'
  intro t ht i hi p hp hN
  have hi' : i < ws[t].plans.size := by rw [← e t ht]; exact hi
  simp only [e t ht] at hp hN
  exact h t (by omega) i hi' p hp hN

theorem capOn_push {N L ws} (h : CapOn N L ws) (w : Win) (hp : w.plans = #[]) : CapOn N L (ws.push w) := by
  intro t ht i hi
  by_cases hl : t < ws.size
  · have k : (ws.push w)[t] = ws[t] := by simp [Array.getElem_push, hl]
    simp only [k] at hi ⊢
    exact h t hl i hi
  · have g : (ws.push w)[t] = w := by simp [Array.getElem_push, hl]
    simp only [g, hp] at hi; simp at hi

theorem capOn_add {N L ws filling} {below now : Nat} {pr : Array Nat → Array Nat} (h : CapOn N L ws)
    (root : ByteArray) (plan : Plan) (post : Nat) (st : Int) (head : Option ByteArray)
    (hav : ∀ p, p < plan.length → plan.block p < N → filling < L) :
    CapOn N L (ws.modify filling (addPass root plan post st head below now pr)) := by
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
  fun hr => capOn_same (h hr) hs

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

theorem cap_releaseCoin {s : State} (h : Cap s) (t now : Nat) : Cap (s.releaseCoin t now) :=
  fun hr => capOn_push (h hr) _ rfl

/-- A plan `receive` lets past `refreshRefusal` in window `t`, with refresh, reads a block below `N` only if
`t < lifetime`. -/
theorem cap_check {s : State} {t : Nat} {plan : Plan} (hP : EndsOk plan) (hchk : s.refreshRefusal t plan = none)
    (hr : s.params.refresh = true) : ∀ p, p < plan.length → plan.block p < s.reg.blocks → t < s.params.lifetime := by
  intro p hp hN
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
    | exact cap_releaseCoin (cap_same h (samePlans_modify _ _ _ fun _ => rfl)) _ _
    | exact cap_same h (samePlans_modify _ _ _ fun _ => rfl)

theorem cap_append {s : State} {root : ByteArray} {count : Int} {origin : _root_.Poub.Origin} {slot : Option Nat}
    {r : Result} {s' : State} (h : Cap s) (he : append s root count origin slot = .ok (r, s')) : Cap s' := by
  unfold append at he
  repeat' split at he
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
    | exact cap_same (cap_final h _) (samePlans_modify _ _ _ fun _ => rfl)

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
    | exact cap_same h (samePlans_modify _ _ _ fun _ => rfl)

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
        | exact fun hr => capOn_add (h₃ hr) _ _ _ _ _
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
  have hc := run_inv Cap (fun _ _ _ _ hs he => step_cap hs he) _ _ _ _ (cap_init ..) hrun
  obtain ⟨-, hp, hreg, -⟩ := inv_run hrun
  intro t ht i hi p hpl hN
  have := hc (by rw [hp]; exact hr) t ht i hi p hpl (by rw [hreg]; exact hN)
  rwa [hp] at this

end Properties.Poub
