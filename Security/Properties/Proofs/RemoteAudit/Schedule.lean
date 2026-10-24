import Properties.RemoteAudit.Schedule
import Models.Accounting.Assumptions.Stacking
import Mathlib.Tactic.Choose
import Batteries.Lean.Except

/-!
# The schedule check fixes each credited pass's plan (`ScheduleChecked`, `ScheduleFixesLayout`)

`check` throws unless `malformed`, `appendsDiffer` and `passesDiffer` each find nothing (`check_ok`). `appendsDiffer`
finding nothing makes the appends the schedule's, entry by entry (`appends_of`), and `passesDiffer` finding nothing puts
each pass at the schedule's step with the schedule's plan there (`passes_of`). Two sessions accepted on one schedule
then have the same appends and, pass by pass, the same steps and so the same plans (`scheduleFixesLayout`).

From the code to the model, under `ScheduleLayoutGap`: a layout the check accepted at `b ≤ a` bits is the verifier's
plan of a well-formed schedule of at most `a` bits (`scheduleDeterministic_of_checked`), and a layout that is
`ScheduleDeterministic` at `a` is `SteeredLayout` at `a + 1`: its summary is the schedule's bits padded to `a + 1` with
a one and zeros (`pad`, one-to-one on strings of at most `a` bits), read back by the schedule they pad
(`steered_of_scheduleDeterministic`).

The collapse it rules out: `coinLayout` puts a pass's one block where a coin bit says. It is `SteeredLayout` at one bit
(`coinLayout_steered`), as every layout is at `a = |Y|`, and `ScheduleDeterministic` at no `a` under a schedule that
fixes the layout (`coinLayout_not_scheduled`). In the code: the published schedule `honest` keeps block 12 resident
from step 3, so a session whose third pass's plan drops it, as a coin might say, is refused, and the honest one passes
at 200 bits (the examples at the end; `test_schedule.py` plays the same session in Python).
-/

open RemoteAudit

namespace Properties.RemoteAudit

theorem check_ok {sc : Schedule.Schedule} {blocks : Nat} {passes : Array (Nat × Array (Nat × Nat))}
    {appends : Array (Nat × Nat)} {b : Nat} (h : Schedule.check sc blocks passes appends = .ok b) :
    Schedule.malformed sc blocks = none ∧ Schedule.appendsDiffer sc blocks appends = none ∧
      Schedule.passesDiffer sc blocks passes = none ∧ b = Schedule.bits sc := by
  unfold Schedule.check at h
  cases hm : Schedule.malformed sc blocks <;> cases ha : Schedule.appendsDiffer sc blocks appends <;>
    cases hp : Schedule.passesDiffer sc blocks passes <;>
    simp_all [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]

theorem appends_of {sc : Schedule.Schedule} {blocks : Nat} {appends : Array (Nat × Nat)}
    (h : Schedule.appendsDiffer sc blocks appends = none) : appends = Schedule.appendRuns sc blocks := by
  unfold Schedule.appendsDiffer at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hs
    have hs : appends.size = (Schedule.appendRuns sc blocks).size := Classical.not_not.mp hs
    rw [List.findSome?_eq_none_iff] at h
    apply Array.ext hs
    intro i h1 h2
    have := h i (List.mem_range.mpr h1)
    split at this
    · rename_i he
      rwa [getElem!_pos appends i h1, getElem!_pos _ i h2] at he
    · cases this

theorem passes_of {sc : Schedule.Schedule} {blocks : Nat} {passes : Array (Nat × Array (Nat × Nat))}
    (h : Schedule.passesDiffer sc blocks passes = none) :
    passes.map Prod.fst = sc.steps ∧ ∀ p ∈ passes, p.2 = Schedule.plan sc blocks p.1 := by
  unfold Schedule.passesDiffer at h
  split at h
  · cases h
  · rename_i hs
    have hs : passes.size = sc.steps.size := Classical.not_not.mp hs
    rw [List.findSome?_eq_none_iff] at h
    have each : ∀ k (hk : k < passes.size),
        passes[k].1 = sc.steps[k]'(hs ▸ hk) ∧ passes[k].2 = Schedule.plan sc blocks passes[k].1 := by
      intro k hk
      have := h k (List.mem_range.mpr hk)
      rw [getElem!_pos passes k hk, getElem!_pos sc.steps k (hs ▸ hk)] at this
      split at this
      · cases this
      · split at this
        · cases this
        · rename_i h1 h2
          exact ⟨Classical.not_not.mp h1, Classical.not_not.mp h2⟩
    refine ⟨Array.ext (by simp [hs]) fun i h1 h2 => ?_, fun p hp => ?_⟩
    · simp only [Array.getElem_map]
      exact (each i (by simpa using h1)).1
    · obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.mp hp
      exact (each k hk).2

theorem scheduleChecked : ScheduleChecked := by
  intro sc blocks passes appends b h
  obtain ⟨hm, ha, hp, hb⟩ := check_ok h
  obtain ⟨h1, h2⟩ := passes_of hp
  exact ⟨hm, h1, h2, appends_of ha, hb⟩

theorem scheduleFixesLayout : ScheduleFixesLayout := by
  intro sc blocks passes passes' appends appends' b b' h h'
  obtain ⟨_, ha, hp, _⟩ := check_ok h
  obtain ⟨_, ha', hp', _⟩ := check_ok h'
  refine ⟨?_, (appends_of ha).trans (appends_of ha').symm⟩
  obtain ⟨f1, s1⟩ := passes_of hp
  obtain ⟨f2, s2⟩ := passes_of hp'
  have e := f1.trans f2.symm
  have hsz : passes.size = passes'.size := by simpa using congrArg Array.size e
  refine Array.ext hsz fun i h1 h2 => ?_
  have e1 : passes[i].1 = passes'[i].1 := by
    have := congrArg (fun a : Array Nat => a[i]?) e
    simpa [h1, h2] using this
  exact Prod.ext e1 (by rw [s1 _ (Array.getElem_mem h1), s2 _ (Array.getElem_mem h2), e1])

theorem scheduleBits_length (sc : Schedule.Schedule) : (scheduleBits sc).length = Schedule.bits sc := by
  unfold scheduleBits Schedule.bits
  induction Schedule.encode sc with
  | nil => rfl
  | cons b l ih => simp only [List.flatMap_cons, List.length_append, List.length_ofFn, ih, List.length_cons]; omega

/-- A string of at most `a` bits as `a + 1` bits: the string, a one, then zeros. -/
def pad (a : ℕ) (l : List Bool) : Pous.Bits (a + 1) := fun i =>
  if h : i.val < l.length then l[i.val] else decide (i.val = l.length)

theorem pad_short {a : ℕ} {l l' : List Bool} (hl' : l'.length ≤ a) (h : pad a l = pad a l') :
    ¬ l.length < l'.length := by
  intro hlt
  have := congrFun h ⟨l'.length, by omega⟩
  simp [pad, Nat.not_lt.mpr (Nat.le_of_lt hlt), (Nat.ne_of_lt hlt).symm] at this

theorem pad_injective {a : ℕ} {l l' : List Bool} (hl : l.length ≤ a) (hl' : l'.length ≤ a)
    (h : pad a l = pad a l') : l = l' := by
  have hlen : l.length = l'.length := by
    rcases Nat.lt_trichotomy l.length l'.length with h1 | h1 | h1
    · exact absurd h1 (pad_short hl' h)
    · exact h1
    · exact absurd h1 (pad_short hl h.symm)
  refine List.ext_getElem hlen fun i h1 h2 => ?_
  have := congrFun h ⟨i, by omega⟩
  simpa [pad, h1, h2] using this

/-- A layout that is the plan of a schedule of at most `a` bits is steered by `a + 1` bits: the schedule's, padded. -/
theorem steered_of_scheduleDeterministic {Pfx Y S : Type} [Nonempty S] {N T M : ℕ}
    {canon : Pfx → S → Fin N ↪ Fin T × Fin M} {enc : S → List Bool} {a : ℕ}
    {layout : Pfx → Y → Fin N ↪ Fin T × Fin M} (h : Accounting.ScheduleDeterministic canon enc a layout) :
    ∃ steer : Pfx → Y → Pous.Bits (a + 1), Accounting.Assumptions.SteeredLayout layout steer := by
  classical
  obtain ⟨inj, each⟩ := h
  choose s hs using each
  refine ⟨fun c y => pad a (enc (s c y)), fun c b => canon c
    (if h : ∃ t, (enc t).length ≤ a ∧ pad a (enc t) = b then h.choose else Classical.arbitrary S), fun c y => ?_⟩
  have hex : ∃ t, (enc t).length ≤ a ∧ pad a (enc t) = pad a (enc (s c y)) := ⟨s c y, (hs c y).1, rfl⟩
  simp only [hex, ↓reduceDIte]
  rw [(hs c y).2, inj (pad_injective hex.choose_spec.1 (hs c y).1 hex.choose_spec.2)]

/-- **The check's acceptance gives `ScheduleDeterministic`, under `ScheduleLayoutGap`.** If at every prefix `c` and
coins `y` the check accepts some schedule at `b ≤ a` bits for a session whose appends are `appended c y` and whose pass
`idx c` sent the runs `sent c y`, the layout is the verifier's plan of a well-formed schedule (its appends, and its
plan at its own step `idx c`) whose bits are at most `a`. -/
theorem scheduleDeterministic_of_checked {Pfx Y : Type} {N T M : ℕ} {blocks a : ℕ} {idx : Pfx → ℕ}
    {pos : Pfx → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M} {appended sent : Pfx → Y → Array (ℕ × ℕ)}
    {layout : Pfx → Y → Fin N ↪ Fin T × Fin M} (gap : ScheduleLayoutGap blocks pos appended sent layout)
    (checked : ∀ c y, ∃ sc passes b, Schedule.check sc blocks passes (appended c y) = .ok b ∧ b ≤ a ∧
      (passes[idx c]?).map Prod.snd = some (sent c y)) :
    Accounting.ScheduleDeterministic (S := {sc : Schedule.Schedule // Schedule.malformed sc blocks = none})
      (fun c sc => pos c (Schedule.appendRuns sc.1 blocks) (Schedule.plan sc.1 blocks sc.1.steps[idx c]!))
      (fun sc => scheduleBits sc.1) a layout := by
  refine ⟨fun s s' h => Subtype.ext (gap.1 s.1 s'.1 s.2 s'.2 h), fun c y => ?_⟩
  obtain ⟨sc, passes, b, h, hb, hsent⟩ := checked c y
  obtain ⟨hm, hsteps, hplan, happ, rfl⟩ := scheduleChecked sc blocks passes (appended c y) b h
  obtain ⟨p, hp, hp2⟩ : ∃ p, passes[idx c]? = some p ∧ p.2 = sent c y := by
    cases e : passes[idx c]? with
    | none => simp [e] at hsent
    | some p => exact ⟨p, rfl, by simpa [e] using hsent⟩
  have hstep : sc.steps[idx c]! = p.1 := by
    have := congrArg (fun a : Array Nat => a[idx c]?) hsteps
    simp only [Array.getElem?_map, hp, Option.map_some] at this
    simp [getElem!_def, ← this]
  refine ⟨⟨sc, hm⟩, Nat.le_trans (Nat.le_of_eq (scheduleBits_length sc)) hb, ?_⟩
  rw [gap.2 c y, ← hp2, happ, hplan p (Array.mem_of_getElem? hp)]
  dsimp only
  rw [hstep]

/-- **The check's acceptance gives `SteeredLayout` at `a + 1` bits, under `ScheduleLayoutGap`.** -/
theorem steered_of_checked {Pfx Y : Type} {N T M : ℕ} {blocks a : ℕ} {idx : Pfx → ℕ}
    {pos : Pfx → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M} {appended sent : Pfx → Y → Array (ℕ × ℕ)}
    {layout : Pfx → Y → Fin N ↪ Fin T × Fin M} (gap : ScheduleLayoutGap blocks pos appended sent layout)
    (checked : ∀ c y, ∃ sc passes b, Schedule.check sc blocks passes (appended c y) = .ok b ∧ b ≤ a ∧
      (passes[idx c]?).map Prod.snd = some (sent c y)) :
    ∃ steer : Pfx → Y → Pous.Bits (a + 1), Accounting.Assumptions.SteeredLayout layout steer :=
  have : Nonempty {sc : Schedule.Schedule // Schedule.malformed sc blocks = none} := ⟨⟨⟨#[], #[], #[]⟩, rfl⟩⟩
  steered_of_scheduleDeterministic (scheduleDeterministic_of_checked (idx := idx) gap checked)

/-- One block, put at position `y` of one append: the coin bit `y` steers it. -/
def coinLayout : Unit → Fin 2 → Fin 1 ↪ Fin 1 × Fin 2 :=
  fun _ y => ⟨fun _ => (0, y), fun _ _ _ => Subsingleton.elim _ _⟩

/-- `coinLayout` is `SteeredLayout` at one bit, the coin's. -/
theorem coinLayout_steered :
    Accounting.Assumptions.SteeredLayout coinLayout (fun _ y => (fun _ => decide (y = 1) : Pous.Bits 1)) :=
  ⟨fun _ b => coinLayout () (if b 0 then 1 else 0), fun _ y => by match y with | 0 => rfl | 1 => rfl⟩

/-- The one schedule's plan: the block at position `0`. -/
def storeOrder : Unit → Unit → Fin 1 ↪ Fin 1 × Fin 2 := fun _ _ => coinLayout () 0

/-- `coinLayout` is `ScheduleDeterministic` at no `a` under a schedule that fixes the block. -/
theorem coinLayout_not_scheduled (enc : Unit → List Bool) (a : ℕ) :
    ¬ Accounting.ScheduleDeterministic storeOrder enc a coinLayout := by
  rintro ⟨_, h⟩
  obtain ⟨s, _, hs⟩ := h () 1
  have : (1 : Fin 2) = 0 := congrArg (fun f => (f 0).2) hs
  exact absurd this (by decide)

/-- A store of 8 registered blocks, block 2 freed at step 1, and two appends: blocks 8–11 written at step 0 (block 9
freed at step 1), blocks 12–13 written at step 2; passes at steps 0 to 4. -/
def honest : Schedule.Schedule :=
  ⟨#[(2, 1, 1)], #[#[some (0, none), some (0, some 1), some (0, none), some (0, none)], #[some (2, none), some (2, none)]],
    #[0, 1, 2, 3, 4]⟩

/-- The session's passes, the third's plan as a coin `keep12` says: `honest`'s at step 3 when it is set, block 12
dropped when not. -/
def coinPasses (keep12 : Bool) : Array (ℕ × Array (ℕ × ℕ)) :=
  #[(0, #[(0, 8)]), (1, #[(0, 12)]), (2, #[(0, 2), (3, 6), (10, 2)]),
    (3, if keep12 then #[(0, 2), (3, 6), (10, 4)] else #[(0, 2), (3, 6), (10, 2), (13, 1)]), (4, #[(0, 2), (3, 6), (10, 4)])]

example : Schedule.check honest 8 (coinPasses false) #[(8, 4), (12, 2)] =
    .error "pass 3's plan is not the schedule's at step 3" := by
  decide +kernel

example : Schedule.check honest 8 (coinPasses true) #[(8, 4), (12, 2)] = .ok 200 := by
  decide +kernel

end Properties.RemoteAudit
