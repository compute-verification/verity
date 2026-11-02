import Properties.RemoteAudit.Schedule
import Definitions.Accounting.Assumptions.Stacking
import Mathlib.Tactic.Choose
import Batteries.Lean.Except

/-!
# The schedule check fixes each credited pass's plan (`ScheduleChecked`, `ScheduleFixesLayout`)

`check` throws unless `malformed`, `appendsDiffer` and `passesDiffer` each find nothing and the schedule's bits are at
most the cap (`check_ok`). `appendsDiffer`
finding nothing makes the appends the schedule's, entry by entry (`appends_of`), and `passesDiffer` finding nothing puts
each pass at the schedule's step with the schedule's plan there (`passes_of`). Two sessions accepted on one schedule
then have the same appends and, pass by pass, the same steps and so the same plans (`scheduleFixesLayout`).

The schedule's bits are one-to-one on well-formed schedules (`ScheduleCode.scheduleBits_injective`): bits to bytes
(eight a byte, `flatMap_byteBits_injective`), bytes to integers (LEB128 is prefix-free, `varint_prefix`), and integers
to the schedule (each list after its length, a block's tag before its write's step, `numbers_injective`). That reads
two facts off `malformed` finding nothing (`wf_of_malformed`): a freed block's free comes after its write, so its tag
`f − w + 1` is at least 2, and the steps increase, so their rises give them back.

From the code to the model, under `ScheduleLayoutGap`: a layout the check accepted at the cap `a` the session committed
before its coins is the verifier's plan of a well-formed schedule of at most `a` bits
(`scheduleDeterministic_of_checked`; the cap is one `a` at every coin, where the bits of each coin's own schedule
would not be), and a layout that is
`ScheduleDeterministic` at `a` is `SteeredLayout` at `a + 1`: its summary is the schedule's bits padded to `a + 1` with
a one and zeros (`pad`, one-to-one on strings of at most `a` bits), read back by the schedule they pad
(`steered_of_scheduleDeterministic`).

The collapse it rules out: `coinLayout` puts a pass's one block where a coin bit says. It is `SteeredLayout` at one bit
(`coinLayout_steered`), as every layout is at `a = |Y|`, and `ScheduleDeterministic` at no `a` under a schedule that
fixes the layout (`coinLayout_not_scheduled`). In the code: the published schedule `honest` keeps block 12 resident
from step 3, so a session whose third pass's plan drops it, as a coin might say, is refused, and the honest one passes
at 200 bits under a cap of 200 and is refused under 199 (the examples at the end; `test_schedule.py` plays the same
session in Python).
-/

open RemoteAudit

namespace Properties.RemoteAudit

theorem check_ok {sc : Schedule.Schedule} {blocks cap : Nat} {passes : Array (Nat × Array (Nat × Nat))}
    {appends : Array (Nat × Nat)} {b : Nat} (h : Schedule.check sc blocks cap passes appends = .ok b) :
    Schedule.malformed sc blocks = none ∧ Schedule.appendsDiffer sc blocks appends = none ∧
      Schedule.passesDiffer sc blocks passes = none ∧ b = Schedule.bits sc ∧ b ≤ cap := by
  unfold Schedule.check at h
  by_cases hc : cap < Schedule.bits sc
  · cases hm : Schedule.malformed sc blocks <;>
      simp_all [bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
  · have hle : Schedule.bits sc ≤ cap := Nat.le_of_not_lt hc
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
  intro sc blocks cap passes appends b h
  obtain ⟨hm, ha, hp, hb, hcap⟩ := check_ok h
  obtain ⟨h1, h2⟩ := passes_of hp
  exact ⟨hm, h1, h2, appends_of ha, hb, hcap⟩

theorem scheduleFixesLayout : ScheduleFixesLayout := by
  intro sc blocks cap cap' passes passes' appends appends' b b' h h'
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

namespace ScheduleCode

theorem varint_lt {n : Nat} (h : n < 128) : Schedule.varint n = [n.toUInt8] := by
  rw [Schedule.varint]; simp [h]

theorem varint_ge {n : Nat} (h : ¬ n < 128) :
    Schedule.varint n = (n % 128 + 128).toUInt8 :: Schedule.varint (n / 128) := by
  rw [Schedule.varint]; simp [h]

theorem varint_ne_nil (n : Nat) : Schedule.varint n ≠ [] := by
  by_cases h : n < 128
  · simp [varint_lt h]
  · simp [varint_ge h]

theorem toUInt8_toNat {n : Nat} (h : n < 256) : n.toUInt8.toNat = n := by
  simp [Nat.toUInt8, Nat.mod_eq_of_lt h]

theorem varint_prefix (n m : Nat) (r r' : List UInt8) (h : Schedule.varint n ++ r = Schedule.varint m ++ r') :
    n = m ∧ r = r' := by
  by_cases hn : n < 128 <;> by_cases hm : m < 128
  · rw [varint_lt hn, varint_lt hm] at h
    simp only [List.singleton_append, List.cons.injEq] at h
    have := congrArg UInt8.toNat h.1
    rw [toUInt8_toNat (by omega), toUInt8_toNat (by omega)] at this
    exact ⟨this, h.2⟩
  · rw [varint_lt hn, varint_ge hm] at h
    simp only [List.cons_append, List.cons.injEq] at h
    have := congrArg UInt8.toNat h.1
    rw [toUInt8_toNat (by omega), toUInt8_toNat (by omega)] at this
    omega
  · rw [varint_ge hn, varint_lt hm] at h
    simp only [List.cons_append, List.cons.injEq] at h
    have := congrArg UInt8.toNat h.1
    rw [toUInt8_toNat (by omega), toUInt8_toNat (by omega)] at this
    omega
  · rw [varint_ge hn, varint_ge hm] at h
    simp only [List.cons_append, List.cons.injEq] at h
    have e1 := congrArg UInt8.toNat h.1
    rw [toUInt8_toNat (by omega), toUInt8_toNat (by omega)] at e1
    obtain ⟨e2, e3⟩ := varint_prefix (n / 128) (m / 128) r r' h.2
    exact ⟨by omega, e3⟩
termination_by n
decreasing_by omega

theorem flatMap_varint_injective :
    ∀ (xs ys : List Nat), xs.flatMap Schedule.varint = ys.flatMap Schedule.varint → xs = ys
  | [], [], _ => rfl
  | [], y :: ys, h => by
    simp only [List.flatMap_nil, List.flatMap_cons] at h
    exact absurd (List.append_eq_nil_iff.mp h.symm).1 (varint_ne_nil y)
  | x :: xs, [], h => by
    simp only [List.flatMap_nil, List.flatMap_cons] at h
    exact absurd (List.append_eq_nil_iff.mp h).1 (varint_ne_nil x)
  | x :: xs, y :: ys, h => by
    simp only [List.flatMap_cons] at h
    obtain ⟨rfl, h'⟩ := varint_prefix x y _ _ h
    rw [flatMap_varint_injective xs ys h']

/-- A code that is prefix-free on the values `P` holds of decodes a list of known length, and the rest after it. -/
theorem flatMap_prefix {α β : Type} (enc : α → List β) (P : α → Prop)
    (hp : ∀ x y r r', P x → P y → enc x ++ r = enc y ++ r' → x = y ∧ r = r') :
    ∀ (xs ys : List α) (r r' : List β), xs.length = ys.length → (∀ x ∈ xs, P x) → (∀ y ∈ ys, P y) →
      xs.flatMap enc ++ r = ys.flatMap enc ++ r' → xs = ys ∧ r = r'
  | [], [], _, _, _, _, _, h => ⟨rfl, by simpa using h⟩
  | [], _ :: _, _, _, hl, _, _, _ => by simp at hl
  | _ :: _, [], _, _, hl, _, _, _ => by simp at hl
  | x :: xs, y :: ys, r, r', hl, px, py, h => by
    simp only [List.flatMap_cons, List.append_assoc] at h
    obtain ⟨rfl, h'⟩ := hp x y _ _ (px x (by simp)) (py y (by simp)) h
    obtain ⟨rfl, rfl⟩ := flatMap_prefix enc P hp xs ys r r' (by simpa using hl)
      (fun x hx => px x (by simp [hx])) (fun y hy => py y (by simp [hy])) h'
    exact ⟨rfl, rfl⟩

def byteBits (b : UInt8) : List Bool := List.ofFn fun i : Fin 8 => b.toNat.testBit i

theorem byteBits_injective {b b' : UInt8} (h : byteBits b = byteBits b') : b = b' := by
  apply UInt8.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 8
  · have := List.getElem_of_eq h (i := i) (by simp [byteBits, hi])
    simpa only [byteBits, List.getElem_ofFn] using this
  · have h1 : b.toNat < 2 ^ i := Nat.lt_of_lt_of_le b.toNat_lt (Nat.pow_le_pow_right (by omega) (by omega))
    have h2 : b'.toNat < 2 ^ i := Nat.lt_of_lt_of_le b'.toNat_lt (Nat.pow_le_pow_right (by omega) (by omega))
    rw [Nat.testBit_lt_two_pow h1, Nat.testBit_lt_two_pow h2]

theorem length_flatMap_byteBits (l : List UInt8) : (l.flatMap byteBits).length = 8 * l.length := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    simp only [List.flatMap_cons, List.length_append, byteBits, List.length_ofFn, ih, List.length_cons]
    omega

theorem flatMap_byteBits_injective {l l' : List UInt8} (h : l.flatMap byteBits = l'.flatMap byteBits) : l = l' := by
  have hlen : l.length = l'.length := by
    have := congrArg List.length h
    rw [length_flatMap_byteBits, length_flatMap_byteBits] at this
    omega
  refine (flatMap_prefix byteBits (fun _ => True) (fun x y r r' _ _ he => ?_) l l' [] [] hlen
    (fun _ _ => trivial) (fun _ _ => trivial) (by simpa using h)).1
  have := List.append_inj he (by simp [byteBits])
  exact ⟨byteBits_injective this.1, this.2⟩

/-- A block a well-formed schedule may hold: freed only after its write. -/
def BlockOk : Schedule.Chunk → Prop
  | some (w, some f) => w < f
  | _ => True

theorem blockCode_prefix (x y : Schedule.Chunk) (r r' : List Nat) (hx : BlockOk x) (hy : BlockOk y)
    (h : Schedule.blockCode x ++ r = Schedule.blockCode y ++ r') : x = y ∧ r = r' := by
  rcases x with _ | ⟨w, _ | f⟩ <;> rcases y with _ | ⟨w', _ | f'⟩ <;>
    simp only [Schedule.blockCode, BlockOk, List.cons_append, List.nil_append, List.cons.injEq] at h hx hy
  · exact ⟨rfl, h.2⟩
  · omega
  · omega
  · omega
  · exact ⟨by rw [h.2.1], h.2.2⟩
  · omega
  · omega
  · omega
  · refine ⟨?_, h.2.2⟩
    have : f = f' := by omega
    rw [h.2.1, this]

def regCode (x : Nat × Nat × Nat) : List Nat := [x.1, x.2.1, x.2.2]

def appendCode (a : Array Schedule.Chunk) : List Nat := a.size :: a.toList.flatMap Schedule.blockCode

theorem regCode_prefix (x y : Nat × Nat × Nat) (r r' : List Nat) (_ : True) (_ : True)
    (h : regCode x ++ r = regCode y ++ r') : x = y ∧ r = r' := by
  obtain ⟨a, b, c⟩ := x
  obtain ⟨a', b', c'⟩ := y
  simp only [regCode, List.cons_append, List.nil_append, List.cons.injEq] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := h
  exact ⟨rfl, rfl⟩

theorem appendCode_prefix (x y : Array Schedule.Chunk) (r r' : List Nat) (hx : ∀ b ∈ x.toList, BlockOk b)
    (hy : ∀ b ∈ y.toList, BlockOk b) (h : appendCode x ++ r = appendCode y ++ r') : x = y ∧ r = r' := by
  simp only [appendCode, List.cons_append, List.cons.injEq] at h
  obtain ⟨hs, h⟩ := h
  obtain ⟨e, rfl⟩ := flatMap_prefix Schedule.blockCode BlockOk blockCode_prefix x.toList y.toList r r'
    (by simpa using hs) hx hy h
  exact ⟨Array.toList_inj.mp e, rfl⟩

/-- Each at least the one before it, the first at least `p`. -/
def Mono : Nat → List Nat → Prop
  | _, [] => True
  | p, x :: xs => p ≤ x ∧ Mono x xs

theorem rises_injective : ∀ (l l' : List Nat) (p : Nat), Mono p l → Mono p l' →
    Schedule.rises l p = Schedule.rises l' p → l = l'
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, _, h => by simp [Schedule.rises] at h
  | _ :: _, [], _, _, _, h => by simp [Schedule.rises] at h
  | x :: xs, y :: ys, p, ⟨hx, hm⟩, ⟨hy, hm'⟩, h => by
    simp only [Schedule.rises, List.cons.injEq] at h
    have : x = y := by omega
    subst this
    rw [rises_injective xs ys x hm hm' h.2]

theorem mono_of_le : ∀ (l : List Nat) (p : Nat), (∀ h : 0 < l.length, p ≤ l[0]) →
    (∀ k (hk : k + 1 < l.length), l[k] ≤ l[k + 1]) → Mono p l
  | [], _, _, _ => trivial
  | x :: xs, p, h0, hs => ⟨h0 (by simp), mono_of_le xs x (fun h => hs 0 (by simpa using h))
      (fun k hk => hs (k + 1) (by simpa using hk))⟩

theorem wf_of_malformed {sc : Schedule.Schedule} {blocks : Nat} (h : Schedule.malformed sc blocks = none) :
    (∀ a ∈ sc.appends.toList, ∀ b ∈ a.toList, BlockOk b) ∧ Mono 0 sc.steps.toList := by
  unfold Schedule.malformed at h
  dsimp only at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hbad
      split at h
      · cases h
      · rename_i hsteps
        refine ⟨fun a ha b hb => ?_, ?_⟩
        · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp ha
          obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.mp hb
          rw [List.findSome?_eq_none_iff] at hbad
          have := hbad i (List.mem_range.mpr (by simpa using hi))
          simp only [Array.getElem_toList] at hk ⊢
          rw [getElem!_pos sc.appends i (by simpa using hi)] at this
          split at this
          · cases this
          · rw [List.findSome?_eq_none_iff] at this
            have := this k (List.mem_range.mpr (by simpa using hk))
            rw [getElem!_pos _ k (by simpa using hk)] at this
            revert this
            rcases sc.appends[i][k] with _ | ⟨w, _ | f⟩ <;> simp [BlockOk]
            intro hc
            split at hc
            · cases hc
            · split at hc
              · cases hc
              · omega
        · simp only [List.any_eq_true, List.mem_range, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true,
            not_exists, not_and] at hsteps
          refine mono_of_le _ 0 (fun _ => Nat.zero_le _) fun k hk => ?_
          have := hsteps (k + 1) (by simpa using hk)
          simp only [Array.getElem_toList]
          rw [getElem!_pos sc.steps (k + 1) (by simpa using hk), Nat.add_sub_cancel,
            getElem!_pos sc.steps k (by simp at hk; omega)] at this
          omega

theorem numbers_eq (sc : Schedule.Schedule) : Schedule.numbers sc = sc.registered.size ::
    (sc.registered.toList.flatMap regCode ++ (sc.appends.size :: (sc.appends.toList.flatMap appendCode ++
      (sc.steps.size :: Schedule.rises sc.steps.toList 0)))) := by
  simp only [Schedule.numbers, List.append_assoc, List.cons_append]
  rfl

theorem numbers_injective {sc sc' : Schedule.Schedule} {blocks : Nat} (hm : Schedule.malformed sc blocks = none)
    (hm' : Schedule.malformed sc' blocks = none) (h : Schedule.numbers sc = Schedule.numbers sc') : sc = sc' := by
  obtain ⟨ha, hs⟩ := wf_of_malformed hm
  obtain ⟨ha', hs'⟩ := wf_of_malformed hm'
  rw [numbers_eq, numbers_eq, List.cons.injEq] at h
  obtain ⟨hr, h⟩ := h
  obtain ⟨hR, h⟩ := flatMap_prefix regCode (fun _ => True) regCode_prefix _ _ _ _ (by simpa using hr)
    (fun _ _ => trivial) (fun _ _ => trivial) h
  rw [List.cons.injEq] at h
  obtain ⟨hn, h⟩ := h
  obtain ⟨hA, h⟩ := flatMap_prefix appendCode _ appendCode_prefix _ _ _ _ (by simpa using hn) ha ha' h
  rw [List.cons.injEq] at h
  have hS := rises_injective _ _ 0 hs hs' h.2
  cases sc
  cases sc'
  simp only [Array.toList_inj] at hR hA hS
  rw [hR, hA, hS]

/-- **The schedule's bits are one-to-one on well-formed schedules.** -/
theorem scheduleBits_injective {sc sc' : Schedule.Schedule} {blocks : Nat} (hm : Schedule.malformed sc blocks = none)
    (hm' : Schedule.malformed sc' blocks = none) (h : scheduleBits sc = scheduleBits sc') : sc = sc' :=
  numbers_injective hm hm' (flatMap_varint_injective _ _ (flatMap_byteBits_injective h))

end ScheduleCode

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

/-- **The check's acceptance at the committed cap gives `ScheduleDeterministic`, under `ScheduleLayoutGap`.** If at
every prefix `c` and coins `y` the check at the cap `a` the session committed before its coins accepts some schedule
for a session whose appends are `appended c y` and whose pass `idx c` sent the runs `sent c y`, the layout is the
verifier's plan of a well-formed schedule (its appends, and its plan at its own step `idx c`) whose bits are at most
`a`. The cap is the check's argument, one `a` at every coin, not the bits of each coin's own schedule. -/
theorem scheduleDeterministic_of_checked {Pfx Y : Type} {N T M : ℕ} {blocks a : ℕ} {idx : Pfx → ℕ}
    {pos : Pfx → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M} {appended sent : Pfx → Y → Array (ℕ × ℕ)}
    {layout : Pfx → Y → Fin N ↪ Fin T × Fin M} (gap : ScheduleLayoutGap pos appended sent layout)
    (checked : ∀ c y, ∃ sc passes b, Schedule.check sc blocks a passes (appended c y) = .ok b ∧
      (passes[idx c]?).map Prod.snd = some (sent c y)) :
    Accounting.ScheduleDeterministic (S := {sc : Schedule.Schedule // Schedule.malformed sc blocks = none})
      (fun c sc => pos c (Schedule.appendRuns sc.1 blocks) (Schedule.plan sc.1 blocks sc.1.steps[idx c]!))
      (fun sc => scheduleBits sc.1) a layout := by
  refine ⟨fun s s' h => Subtype.ext (ScheduleCode.scheduleBits_injective s.2 s'.2 h), fun c y => ?_⟩
  obtain ⟨sc, passes, b, h, hsent⟩ := checked c y
  obtain ⟨hm, hsteps, hplan, happ, rfl, hb⟩ := scheduleChecked sc blocks a passes (appended c y) b h
  obtain ⟨p, hp, hp2⟩ : ∃ p, passes[idx c]? = some p ∧ p.2 = sent c y := by
    cases e : passes[idx c]? with
    | none => simp [e] at hsent
    | some p => exact ⟨p, rfl, by simpa [e] using hsent⟩
  have hstep : sc.steps[idx c]! = p.1 := by
    have := congrArg (fun a : Array Nat => a[idx c]?) hsteps
    simp only [Array.getElem?_map, hp, Option.map_some] at this
    simp [getElem!_def, ← this]
  refine ⟨⟨sc, hm⟩, Nat.le_trans (Nat.le_of_eq (scheduleBits_length sc)) hb, ?_⟩
  rw [gap c y, ← hp2, happ, hplan p (Array.mem_of_getElem? hp)]
  dsimp only
  rw [hstep]

/-- **The check's acceptance at the committed cap `a` gives `SteeredLayout` at `a + 1` bits, under
`ScheduleLayoutGap`.** `a` is the cap the session's parameters committed before its first coin
(`audit.Params.schedule_cap`, `audit.Ledger.steering`), so it is the same at every prefix and coin. -/
theorem steered_of_checked {Pfx Y : Type} {N T M : ℕ} {blocks a : ℕ} {idx : Pfx → ℕ}
    {pos : Pfx → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M} {appended sent : Pfx → Y → Array (ℕ × ℕ)}
    {layout : Pfx → Y → Fin N ↪ Fin T × Fin M} (gap : ScheduleLayoutGap pos appended sent layout)
    (checked : ∀ c y, ∃ sc passes b, Schedule.check sc blocks a passes (appended c y) = .ok b ∧
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
  ⟨#[(2, 1, 1)],
    #[#[some (0, none), some (0, some 1), some (0, none), some (0, none)], #[some (2, none), some (2, none)]],
    #[0, 1, 2, 3, 4]⟩

/-- The session's passes, the third's plan as a coin `keep12` says: `honest`'s at step 3 when it is set, block 12
dropped when not. -/
def coinPasses (keep12 : Bool) : Array (ℕ × Array (ℕ × ℕ)) :=
  #[(0, #[(0, 8)]), (1, #[(0, 12)]), (2, #[(0, 2), (3, 6), (10, 2)]),
    (3, if keep12 then #[(0, 2), (3, 6), (10, 4)] else #[(0, 2), (3, 6), (10, 2), (13, 1)]),
    (4, #[(0, 2), (3, 6), (10, 4)])]

example : Schedule.check honest 8 200 (coinPasses false) #[(8, 4), (12, 2)] =
    .error "pass 3's plan is not the schedule's at step 3" := by
  decide +kernel

example : Schedule.check honest 8 200 (coinPasses true) #[(8, 4), (12, 2)] = .ok 200 := by
  decide +kernel

example : Schedule.check honest 8 199 (coinPasses true) #[(8, 4), (12, 2)] =
    .error "the schedule is 200 bits, over the cap of 199 committed before the coins" := by
  decide +kernel

end Properties.RemoteAudit
