import Batteries.Tactic.OpenPrivate
import Properties.RemoteAudit.Schedule
import Proofs.Properties.RemoteAudit.Plan
import Proofs.Properties.RemoteAudit.Window
import Definitions.Accounting.Assumptions.Stacking
import Mathlib.Tactic.Choose
import Mathlib.Data.Fintype.Pigeonhole
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

From the code to the model, at the verifier's layout (`sessionLayout`, where `ScheduleLayoutGap` holds by definition,
`scheduleLayoutGap_code`): a layout the check accepted at the cap `a` the session committed before its coins is the
verifier's plan of a well-formed schedule of at most `a` bits
(`scheduleDeterministic_of_checked`; the cap is one `a` at every coin, where the bits of each coin's own schedule
would not be), and a layout that is
`ScheduleDeterministic` at `a` is `SteeredLayout` at `a + 1`: its summary is the schedule's bits padded to `a + 1` with
a one and zeros (`pad`, one-to-one on strings of at most `a` bits), read back by the schedule they pad
(`steered_of_scheduleDeterministic`).

That layout is where the verifier checks. A plan's position `p` is entry `p` of its runs' blocks (`planBlocks`: the
running ends' bisection finds the run, `block_spec`, and the offset into it is the entry, `getElem?_flatMap_range'`). A
plan the verifier makes reads distinct blocks (`planReadsDistinct`), and a tile's position's second coordinate is its
block (`snd_of_tileAt`), so a plan of `N` blocks with positions in range is one embedding, which `codeLayout` picks
(`codeLayoutTiles`). An accepted opening passed `checkOpening`'s checks (`checkOpening_none`), whose tree is the
registration's below `N` and `appendOf`'s above, the tree `position` names (`opensAtPosition`).

The collapse it rules out: `coinLayout` puts a pass's one block where a coin bit says. It is `SteeredLayout` at one bit
(`coinLayout_steered`), as every layout is at `a = |Y|`, and `ScheduleDeterministic` at no `a` under a schedule that
fixes the layout (`coinLayout_not_scheduled`). In the code: the published schedule `honest` keeps block 12 resident
from step 3, so a session whose third pass's plan drops it, as a coin might say, is refused, and the honest one passes
at 200 bits under a cap of 200 and is refused under 199 (`coinPasses_refused`, `coinPasses_accepted`,
`coinPasses_overCap`; `test_schedule.py` plays the same session in Python).

Each guarantee's premises hold at the deployed shape, one pass over all 130023424 blocks (the `_honest` lemmas),
and each conclusion is false for some input a verifier that accepted everything would take (the `_nontrivial`
lemmas): the lock's `witnesses`.
-/

open RemoteAudit

open private RemoteAudit.endsOf from RemoteAudit.Plan

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

/-- **`ScheduleLayoutGap` holds of the verifier's layout**: `sessionLayout` is `codeLayout` of the session's appends and
the runs the pass sent, at every prefix and coin. -/
theorem scheduleLayoutGap_code {Pfx Y : Type} {N T M : ℕ} (blocks : ℕ) (dflt : Pfx → Fin N ↪ Fin T × Fin M)
    (appended sent : Pfx → Y → Array (ℕ × ℕ)) :
    ScheduleLayoutGap (fun c => codeLayout blocks (dflt c)) appended sent (sessionLayout blocks dflt appended sent) :=
  fun _ _ => rfl

/-- **The check's acceptance at the committed cap `a` gives `SteeredLayout` at `a + 1` bits, at the verifier's own
layout** (`sessionLayout`, where `ScheduleLayoutGap` holds by definition, `scheduleLayoutGap_code`). `a` is the cap
the session's parameters committed before its first coin (`audit.Params.schedule_cap`, `audit.Ledger.steering`), so it
is the same at every prefix and coin. -/
theorem steered_of_checked {Pfx Y : Type} {N T M : ℕ} {blocks a : ℕ} {idx : Pfx → ℕ}
    {dflt : Pfx → Fin N ↪ Fin T × Fin M} {appended sent : Pfx → Y → Array (ℕ × ℕ)}
    (checked : ∀ c y, ∃ sc passes b, Schedule.check sc blocks a passes (appended c y) = .ok b ∧
      (passes[idx c]?).map Prod.snd = some (sent c y)) :
    ∃ steer : Pfx → Y → Pous.Bits (a + 1),
      Accounting.Assumptions.SteeredLayout (sessionLayout blocks dflt appended sent) steer :=
  have : Nonempty {sc : Schedule.Schedule // Schedule.malformed sc blocks = none} := ⟨⟨⟨#[], #[], #[]⟩, rfl⟩⟩
  steered_of_scheduleDeterministic
    (scheduleDeterministic_of_checked (idx := idx) (scheduleLayoutGap_code blocks dflt appended sent) checked)

/-! ## The verifier's layout is where it checks -/

theorem make_ends {runs : Array (Int × Int)} {blocks : ℕ} {pl : Plan} (h : Plan.make? runs blocks = some pl) :
    pl.ends = RemoteAudit.endsOf pl.runs := by
  unfold Plan.make? at h
  simp only [Bool.not_eq_true', Option.bind_eq_bind, Option.bind_none] at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  cases h
  rfl

theorem length_runBlocks (rs : Array (ℕ × ℕ)) : (runBlocks rs).length = (rs.toList.map (·.2)).sum := by
  simp [runBlocks, List.length_flatMap]

/-- Entry `p` of runs' blocks, `p` in run `r`: the run's first block, `p`'s offset into it on. -/
theorem getElem?_flatMap_range' : ∀ (l : List (ℕ × ℕ)) (r p : ℕ) (hr : r < l.length),
    ((l.map (·.2)).take r).sum ≤ p → p < ((l.map (·.2)).take r).sum + l[r].2 →
      (l.flatMap fun x => List.range' x.1 x.2)[p]? = some (l[r].1 + (p - ((l.map (·.2)).take r).sum))
  | [], _, _, hr, _, _ => absurd hr (by simp)
  | x :: xs, 0, p, _, _, h2 => by
    simp only [List.take_zero, List.sum_nil, Nat.zero_add, List.getElem_cons_zero, Nat.sub_zero] at h2 ⊢
    rw [List.flatMap_cons, List.getElem?_append_left (by simpa using h2), List.getElem?_range' h2, Nat.one_mul]
  | x :: xs, r + 1, p, hr, h1, h2 => by
    simp only [List.map_cons, List.take_succ_cons, List.sum_cons, List.getElem_cons_succ] at h1 h2 ⊢
    rw [List.flatMap_cons, List.getElem?_append_right (by simp only [List.length_range']; omega),
      List.length_range', getElem?_flatMap_range' xs r (p - x.2) (by simpa using hr) (by omega) (by omega)]
    congr 2
    omega

theorem planBlocks_of (b : ℕ) (rs : Array (ℕ × ℕ)) (ends : Array ℕ) (hP : PrefixSums ends (rs.toList.map (·.2))) :
    Plan.length ⟨b, rs, ends⟩ = (runBlocks rs).length ∧
      ∀ p < Plan.length ⟨b, rs, ends⟩, (runBlocks rs)[p]? = some (Plan.block ⟨b, rs, ends⟩ p) := by
  have hlen : Plan.length ⟨b, rs, ends⟩ = (rs.toList.map (·.2)).sum := back_getD_of_prefix hP
  refine ⟨hlen.trans (length_runBlocks rs).symm, fun p hp => ?_⟩
  rw [hlen] at hp
  obtain ⟨hr, hlo, hhi, hb⟩ := block_spec b rs ends hP p hp
  rw [hb, getElem!_pos rs _ hr]
  rw [getElem!_pos rs _ hr] at hhi
  have := getElem?_flatMap_range' rs.toList (bisectRight ends p) p (by simpa using hr) hlo
    (by rw [Array.getElem_toList]; exact hhi)
  rw [Array.getElem_toList] at this
  exact this

theorem whole_prefix (blocks : ℕ) : PrefixSums #[blocks] ((#[(0, blocks)] : Array (ℕ × ℕ)).toList.map (·.2)) := by
  refine ⟨rfl, fun i hi => ?_⟩
  obtain rfl : i = 0 := by simpa using hi
  rfl

theorem planBlocks : PlanBlocks := by
  refine ⟨fun runs blocks pl h => ?_, fun blocks => planBlocks_of blocks #[(0, blocks)] #[blocks] (whole_prefix blocks)⟩
  obtain ⟨b, rs, ends⟩ := pl
  have he : ends = RemoteAudit.endsOf rs := make_ends h
  subst he
  exact planBlocks_of b rs _ (endsOf_prefix rs)

/-- A tile's position's second coordinate is its block. -/
theorem snd_of_tileAt {blocks : ℕ} {appends runs : Array (ℕ × ℕ)} {i t m : ℕ}
    (h : tileAt blocks appends runs i = some (t, m)) : (runBlocks runs)[i]? = some m := by
  unfold tileAt at h
  cases hj : (runBlocks runs)[i]? with
  | none => simp [hj] at h
  | some j =>
    rw [hj] at h
    change position blocks appends j = some (t, m) at h
    unfold position at h
    split at h
    · exact congrArg some (Prod.mk.inj (Option.some.inj h)).2
    · dsimp only at h
      split at h
      · cases h
      · exact congrArg some (Prod.mk.inj (Option.some.inj h)).2

theorem codeLayoutTiles : CodeLayoutTiles := by
  intro N T M blocks extent appends runs pl hmake dflt hN hpos i
  obtain ⟨hlen, hget⟩ := planBlocks.1 runs extent pl hmake
  have hdist := (planReadsDistinct runs extent pl hmake).2
  choose t m hspec using hpos
  let f : Fin N → Fin T × Fin M := fun k => (⟨t k k.2, (hspec k k.2).2.1⟩, ⟨m k k.2, (hspec k k.2).2.2⟩)
  have hf : Function.Injective f := by
    intro a b hab
    have hm : m a a.2 = m b b.2 := congrArg (fun x => (x.2 : ℕ)) hab
    have ha := snd_of_tileAt (hspec a a.2).1
    have hb := snd_of_tileAt (hspec b b.2).1
    have hpa : (a : ℕ) < pl.length := by have := a.2; omega
    have hpb : (b : ℕ) < pl.length := by have := b.2; omega
    rw [hget a hpa, hm] at ha
    rw [hget b hpb] at hb
    exact Fin.ext (hdist a b hpa hpb (Option.some.inj (ha.trans hb.symm)))
  have hex : ∃ e : Fin N ↪ Fin T × Fin M, (runBlocks pl.runs).length = N ∧
      ∀ k : Fin N, tileAt blocks appends pl.runs k = some (((e k).1 : ℕ), ((e k).2 : ℕ)) :=
    ⟨⟨f, hf⟩, hN, fun k => (hspec k k.2).1⟩
  have hc : codeLayout blocks dflt appends pl.runs = hex.choose := by
    unfold codeLayout
    split
    · rfl
    · rename_i hn
      exact absurd hex hn
  rw [hc]
  exact hex.choose_spec.2 i

theorem opensAtPosition : OpensAtPosition := by
  intro reg appends w i p o h
  have hp := checkOpening_none h
  unfold OpeningPasses at hp
  dsimp only at hp ⊢
  generalize w.passes[i]!.plan.block p = j at hp ⊢
  obtain ⟨-, -, -, -, -, -, -, -, -, _, -, -, ⟨payload, ⟨hj, hdec, hver⟩ | ⟨hj, a, ha, hdec, hver⟩⟩, -⟩ := hp
  · exact ⟨0, by simp [position, hj], payload, Or.inl ⟨rfl, hdec, hver⟩⟩
  · unfold appendOf at ha
    dsimp only at ha
    split at ha
    · cases ha
    · rename_i hk
      have hmap : (appends.map fun a => (a.first, a.count)).map Prod.fst = appends.map (·.first) := by
        simp [Array.map_map, Function.comp_def]
      refine ⟨bisectRight (appends.map (·.first)) j, ?_, payload,
        Or.inr ⟨Nat.pos_of_ne_zero hk, a, ha, hdec, hver⟩⟩
      unfold position
      rw [hmap]
      simp [Nat.not_lt.mpr hj, hk]

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

theorem coinPasses_refused : Schedule.check honest 8 200 (coinPasses false) #[(8, 4), (12, 2)] =
    .error "pass 3's plan is not the schedule's at step 3" := by
  decide +kernel

theorem coinPasses_accepted : Schedule.check honest 8 200 (coinPasses true) #[(8, 4), (12, 2)] = .ok 200 := by
  decide +kernel

theorem coinPasses_overCap : Schedule.check honest 8 199 (coinPasses true) #[(8, 4), (12, 2)] =
    .error "the schedule is 200 bits, over the cap of 199 committed before the coins" := by
  decide +kernel

/-! ## Honest and nontrivial lemmas

Each property's honest lemma (`_honest`) shows its premises hold at the deployed instance, the session of
`benchmarks/remote_session/tests/fixtures/gate_run_7.json.gz`: a store of 130023424 blocks of 4096 bytes under
p1024/k3/masked with sketch-sha512 receipts, each window one pass over the whole store (`Plan.whole`), with no appends
and no schedule cap committed. Its nontrivial lemma (`_nontrivial`) shows its conclusion fails where the code's
check is dropped, for a verifier that accepts everything.

The lock lists both as the property's `witnesses`, so each builds its instance: no binder ranges over it (a
nontrivial lemma's one `∀` stands for its property's `∃`), and its statement reads only the spec and the code, its
instance written out rather than named by a definition of this file. -/

theorem size_qsort {α : Type} (as : Array α) (lt : α → α → Bool) (lo hi : ℕ) :
    (as.qsort lt lo hi).size = as.size := by
  unfold Array.qsort
  split
  · rfl
  · simp

/-- **`PlanBlocks` meets the deployed plan.** The plan of the deployed passes, the whole store as one run, is one
`Plan.make?` makes, and it is `Plan.whole`, the second conjunct's plan. -/
theorem planBlocks_honest : Plan.make? #[((0 : ℤ), (130023424 : ℤ))] 130023424 = some (Plan.whole 130023424) := by
  have h : (Plan.make? #[((0 : ℤ), (130023424 : ℤ))] 130023424).map (fun pl => (pl.blocks, pl.runs, pl.ends)) =
      some (130023424, #[(0, 130023424)], #[130023424]) := by
    simp only [Plan.make?, size_qsort]
    decide +kernel
  cases e : Plan.make? #[((0 : ℤ), (130023424 : ℤ))] 130023424 with
  | none => rw [e] at h; cases h
  | some pl =>
    rw [e] at h
    obtain ⟨b, rs, ends⟩ := pl
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    rfl

/-- **`PlanBlocks` fails for a plan the verifier did not make**: a verifier that accepted every plan would take one
of one position over no runs, which has more positions than its runs have blocks. -/
theorem planBlocks_nontrivial :
    ¬ ((⟨1, #[], #[1]⟩ : Plan).length = (runBlocks (⟨1, #[], #[1]⟩ : Plan).runs).length ∧
      ∀ p < (⟨1, #[], #[1]⟩ : Plan).length,
        (runBlocks (⟨1, #[], #[1]⟩ : Plan).runs)[p]? = some ((⟨1, #[], #[1]⟩ : Plan).block p)) :=
  fun h => absurd h.1 (by decide)

theorem runBlocks_whole (B : ℕ) : runBlocks (Plan.whole B).runs = List.range' 0 B := by
  simp [runBlocks, Plan.whole]

theorem tileAt_whole {B i : ℕ} (hi : i < B) : tileAt B #[] (Plan.whole B).runs i = some (0, i) := by
  rw [tileAt, runBlocks_whole, List.getElem?_range' hi]
  simp [position, hi]

/-- **`CodeLayoutTiles` meets the deployed pass.** At the deployed plan, made by `Plan.make?`, its runs hold the
store's 130023424 blocks, tile `i` at `(0, i)`, a position below `(1, 130023424)`: the registration's tree, at its own
index. -/
theorem codeLayoutTiles_honest :
    Plan.make? #[((0 : ℤ), (130023424 : ℤ))] 130023424 = some (Plan.whole 130023424) ∧
      (runBlocks (Plan.whole 130023424).runs).length = 130023424 ∧
      ∀ i < 130023424, tileAt 130023424 #[] (Plan.whole 130023424).runs i = some (0, i) :=
  ⟨planBlocks_honest, by rw [runBlocks_whole, List.length_range'], fun _ hi => tileAt_whole hi⟩

/-- An embedding that puts each tile at `tileAt` of the runs is the verifier's layout. -/
theorem codeLayout_eq {N T M blocks : ℕ} {dflt : Fin N ↪ Fin T × Fin M} {appends runs : Array (ℕ × ℕ)}
    (e : Fin N ↪ Fin T × Fin M) (hN : (runBlocks runs).length = N)
    (he : ∀ i : Fin N, tileAt blocks appends runs i = some (((e i).1 : ℕ), ((e i).2 : ℕ))) :
    codeLayout blocks dflt appends runs = e := by
  have hex : ∃ e : Fin N ↪ Fin T × Fin M, (runBlocks runs).length = N ∧
      ∀ i : Fin N, tileAt blocks appends runs i = some (((e i).1 : ℕ), ((e i).2 : ℕ)) := ⟨e, hN, he⟩
  have hc : codeLayout blocks dflt appends runs = hex.choose := by
    unfold codeLayout
    split
    · rfl
    · rename_i hn
      exact absurd hex hn
  rw [hc]
  refine Function.Embedding.ext fun i => ?_
  obtain ⟨h1, h2⟩ := Prod.mk.inj (Option.some.inj ((hex.choose_spec.2 i).symm.trans (he i)))
  exact Prod.ext (Fin.ext h1) (Fin.ext h2)

/-- Store order: tile `i` at block `i` of the registration's tree. -/
def wholeLayout (B : ℕ) : Fin B ↪ Fin 1 × Fin B := ⟨fun i => (0, i), fun _ _ h => (Prod.mk.inj h).2⟩

/-- At the deployed plan the verifier's layout is store order. -/
theorem codeLayout_whole (B : ℕ) (dflt : Fin B ↪ Fin 1 × Fin B) :
    codeLayout B dflt #[] (Plan.whole B).runs = wholeLayout B :=
  codeLayout_eq (wholeLayout B) (by rw [runBlocks_whole, List.length_range']) fun i => tileAt_whole i.2

/-- Runs that repeat a block put both tiles at one position, which no embedding does, so the verifier's layout of them
is the default. -/
theorem codeLayout_repeat (dflt : Fin 2 ↪ Fin 1 × Fin 2) : codeLayout 1 dflt #[] #[(0, 1), (0, 1)] = dflt := by
  have ht : ∀ i < 2, tileAt 1 #[] #[(0, 1), (0, 1)] i = some (0, 0) := by decide
  unfold codeLayout
  split
  · rename_i h
    exfalso
    obtain ⟨e, -, he⟩ := h
    have e0 := (Prod.mk.inj (Option.some.inj ((ht 0 (by decide)).symm.trans (he 0)))).2
    have e1 := (Prod.mk.inj (Option.some.inj ((ht 1 (by decide)).symm.trans (he 1)))).2
    have : e 0 = e 1 := Prod.ext (Subsingleton.elim _ _) (Fin.ext (e0.symm.trans e1))
    exact absurd (e.injective this) (by decide)
  · rfl

/-- **`CodeLayoutTiles` fails for a plan the verifier did not make**: runs that repeat a block, which `Plan.make?`
refuses and a verifier that accepted every plan would take, hold two blocks with positions in range, both `(0, 0)`.
No embedding puts both tiles there, so the verifier's layout is the default, store order, and tile 1 is not at its
block's position. -/
theorem codeLayoutTiles_nontrivial :
    ¬ (tileAt 1 #[] #[(0, 1), (0, 1)] ((1 : Fin 2) : ℕ) =
      some (((codeLayout 1 (⟨fun i => (0, i), fun _ _ h => (Prod.mk.inj h).2⟩ : Fin 2 ↪ Fin 1 × Fin 2) #[]
          #[(0, 1), (0, 1)] 1).1 : ℕ),
        ((codeLayout 1 (⟨fun i => (0, i), fun _ _ h => (Prod.mk.inj h).2⟩ : Fin 2 ↪ Fin 1 × Fin 2) #[]
          #[(0, 1), (0, 1)] 1).2 : ℕ))) := by
  rw [codeLayout_repeat]
  decide

/-! ### An opening the verifier accepts at the deployed shape

A tree whose siblings are all 64 zero bytes (`zeroPath`) has the root its leaf climbs to (`zeroRoot`), and its
opening verifies (`verifyOpening_zeroRoot`). The zero block decodes under every key (its words are 0, below `p`), to a
payload `zeroPayload`; a registration whose payload tree holds that payload at block 0, and a pass over the whole store
whose receipt tree holds the zero block's receipt at position 0, accept the opening of position 0
(`opensAtPosition_honest`). The payload is the zero block's decoding under the codec key, which an honest prover stores
after the key is drawn; at `Registration.register` the root comes before the seed, so this exhibits acceptance at the
deployed registration (`Registration.make?`), parameters and plan, not a prover that registered first. -/

theorem size_sha512 (b : ByteArray) : (RemoteAudit.sha512 b).size = 64 := by
  unfold RemoteAudit.sha512 Flock.Sha512.hash
  exact Array.size_ofFn

theorem size_frameHash (tag : String) (parts : List ByteArray) : (Flock.FV3.Sha512.hash tag parts).size = 64 := by
  unfold Flock.FV3.Sha512.hash Flock.FV3.hash
  exact size_sha512 _

theorem size_node (d : Flock.FV3.Sha512.Domain) (l i : ℕ) (a b : ByteArray) : (d.node l i a b).size = 64 :=
  size_frameHash _ _

theorem size_leaf (d : Flock.FV3.Sha512.Domain) (r q : ℕ) (s : String) (v : ByteArray) :
    (d.leaf r q s v).size = 64 :=
  size_frameHash _ _

theorem zeros_size (n : ℕ) : (zeros n).size = n :=
  Array.size_replicate

theorem receipt_size (r : Receipter) (j : ℕ) (b : ByteArray) : (r.receipt j b).size = 64 := by
  unfold Receipter.receipt
  split <;> exact size_sha512 _

/-- A digest of 64 bytes climbs, through any siblings, to a node or itself: 64 bytes. -/
theorem climbL_size (dom : ByteArray) : ∀ (cur : ByteArray) (idx l : ℕ) (sibs : List ByteArray), cur.size = 64 →
    (Flock.FV3.Sha512.scheme.climbL dom cur idx l sibs).size = 64
  | _, _, _, [], h => h
  | cur, idx, l, sib :: rest, _ => by
    simp only [Flock.FV3.Scheme.climbL]
    exact climbL_size dom _ _ _ rest (by split <;> exact size_frameHash _ _)

/-- A root climbed to is a node or the leaf: 64 bytes. -/
theorem climb_size (d : Flock.FV3.Sha512.Domain) (pos : ℕ) (schema : String) (value : ByteArray)
    (path : Array ByteArray) : (d.climb pos schema value path).size = 64 := by
  simp only [Flock.FV3.Domain.climb]
  exact climbL_size d.id _ _ _ _ (size_leaf _ _ _ _ _)

theorem scheme_bytes : Flock.FV3.Sha512.scheme.bytes = 64 := rfl

/-- A path of `log2ceil count` siblings, each 64 zero bytes. -/
def zeroPath (d : Flock.FV3.Sha512.Domain) : Array ByteArray := Array.replicate (Flock.log2ceil d.count) (zeros 64)

/-- The root `value` at `position` climbs to along `zeroPath`. -/
def zeroRoot (d : Flock.FV3.Sha512.Domain) (position : ℕ) (value : ByteArray) : ByteArray :=
  d.climb position SCHEMA value (zeroPath d)

theorem verifyOpening_zeroRoot (d : Flock.FV3.Sha512.Domain) (pos : ℕ) (v : ByteArray) (hs : d.start ≤ pos)
    (hp : pos < d.start + d.count) : d.verifyOpening (zeroRoot d pos v) d.count pos v SCHEMA (zeroPath d) = true := by
  have hall : (zeroPath d).all (·.size == 64) = true := by
    rw [Array.all_eq_true]
    intro i _
    simp [zeroPath, zeros_size]
  have hlen : (zeroPath d).size = Flock.log2ceil d.count := Array.size_replicate
  unfold Flock.FV3.Domain.verifyOpening
  simp only [scheme_bytes]
  rw [hall]
  simp [zeroRoot, climb_size, hlen, hs, hp]

theorem leafReceipt_first {R : ByteArray} (hR : R.size = 64) : leafReceipt (R ++ zeros 960) 0 130023424 = some R := by
  have hs : (R ++ zeros 960).size = 1024 := by rw [ByteArray.size_append, hR, zeros_size]
  unfold leafReceipt
  dsimp only
  split
  · rename_i h
    rw [hs] at h
    exact absurd h (by decide)
  · show some ((R ++ zeros 960).extract 0 64) = some R
    rw [ByteArray.extract_append_eq_left hR.symm]

theorem bytes_toList_loop (b : ByteArray) (i : ℕ) (r : List UInt8) :
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i := by
  fun_induction ByteArray.toList.loop b i r with
  | case1 i r h ih =>
    rw [ih, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 1
    conv_rhs => rw [List.drop_eq_getElem_cons (by simpa using h)]
    simp [ByteArray.get!, h]
  | case2 i r h =>
    rw [List.drop_eq_nil_of_le (by simp; simpa using h), List.append_nil]

theorem bytes_toList (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [bytes_toList_loop]
  simp

theorem foldr_zero : ∀ l : List UInt8, (∀ x ∈ l, x = 0) → l.foldr (fun x acc => acc * 256 + x.toNat) 0 = 0
  | [], _ => rfl
  | x :: l, h => by
    have hx := h x List.mem_cons_self
    subst hx
    simp [foldr_zero l fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem leNat_zeros (n a c : ℕ) : leNat ((zeros n).extract a c) = 0 := by
  unfold leNat
  rw [bytes_toList]
  refine foldr_zero _ fun x hx => ?_
  rw [Array.mem_toList_iff] at hx
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
  simp [ByteArray.data_extract, zeros]

/-- The zero block decodes, under any key of a codec whose stores hold a block. -/
theorem decode_zeros (k : Codec.Key) (hj : 0 < k.codec.maxBlocks) (hp : 0 < k.codec.field.p) :
    ∃ payload, Codec.decode (zeros Codec.BLOCK_BYTES) 0 k none = .ok (some payload) := by
  have hw : ∀ x ∈ Codec.words (zeros Codec.BLOCK_BYTES) k.codec.field, x = 0 := by
    intro x hx
    simp only [Codec.words, Array.mem_map] at hx
    obtain ⟨w, -, rfl⟩ := hx
    exact leNat_zeros _ _ _
  have hany : ((Codec.words (zeros Codec.BLOCK_BYTES) k.codec.field).any fun x => decide (x ≥ k.codec.field.p)) =
      false := by
    rw [Array.any_eq_false]
    intro i hi
    rw [hw _ (Array.getElem_mem hi)]
    simp only [ge_iff_le, decide_eq_true_eq]
    omega
  unfold Codec.decode
  dsimp only
  rw [hany]
  simp [Nat.not_le.mpr hj, zeros_size]
  try exact ⟨_, rfl⟩

/-- The zero block's decoding under `k`. -/
def zeroPayload (k : Codec.Key) : ByteArray :=
  match Codec.decode (zeros Codec.BLOCK_BYTES) 0 k none with
  | .ok (some p) => p
  | _ => .empty

theorem decode_zeroPayload (k : Codec.Key) (hj : 0 < k.codec.maxBlocks) (hp : 0 < k.codec.field.p) :
    Codec.decode (zeros Codec.BLOCK_BYTES) 0 k none = .ok (some (zeroPayload k)) := by
  obtain ⟨payload, h⟩ := decode_zeros k hj hp
  unfold zeroPayload
  rw [h]
  try rfl

/-- `checkOpening` accepts an opening of a registered block that passes checks 1–5. -/
theorem checkOpening_of {reg : Registration} {digest : ByteArray} {appends : Array Append} {w : WindowRecord}
    {R : Receipter} {i p : ℕ} {o : Opening} {pr : PassRecord} {j : ℕ} {y payload : ByteArray}
    (hpr : w.passes[i]! = pr) (hpass : o.pass = i) (hpos : o.position = p)
    (hblock : o.block.size = Codec.BLOCK_BYTES) (htile : o.tile.size = 0) (hls : o.leafSalt.size = SALT_BYTES)
    (hps : o.payloadSalt.size = SALT_BYTES) (hpaths : ∀ x ∈ o.receiptPath ++ o.payloadPath, x.size = DIGEST_BYTES)
    (hjb : pr.plan.block p = j) (hj : j < reg.blocks) (hdig : o.digest.size = 0)
    (hy : leafReceipt o.leaf p pr.plan.length = some y)
    (hleaf : verifyLeaf (w.seedOf i) pr.plan pr.root p o.leaf o.leafSalt o.receiptPath = true)
    (hdec : Codec.decode o.block j reg.key none = .ok (some payload))
    (hver : reg.payloadDomain.verifyOpening reg.payloadRoot reg.payloadDomain.count j (hidden payload o.payloadSalt)
      SCHEMA o.payloadPath = true)
    (hrec : R.receipt j o.block = y) :
    checkOpening reg digest appends w R i p o = none := by
  have hany : ((o.receiptPath ++ o.payloadPath).any fun x => decide (x.size ≠ DIGEST_BYTES)) = false := by
    rw [Array.any_eq_false]
    intro k hk
    rw [hpaths _ (Array.getElem_mem hk)]
    simp
  unfold checkOpening
  simp only [Id.run, hpr, hjb]
  rw [hany]
  simp [hpass, hpos, hblock, htile, hls, hps, hj, hdig, hy, hleaf, hdec, hver, hrec]
  rfl

/-- The deployed codec: p1024, three rounds, masked. -/
def honestCodec : Codec.Codec := ⟨Codec.P1024, 3, true⟩

/-- The deployed registration of node `node` under codec seed `codecSeed`: 130023424 blocks, sketch-sha512 receipts
of `rows` rows, and a payload tree holding `zeroPayload` at block 0. -/
def honestReg (node codecSeed : ByteArray) (rows : ℕ) : Registration :=
  let key := Codec.key codecSeed honestCodec
  let pd := Flock.FV3.Sha512.Domain.make (RemoteAudit.sha512 (PAYLOAD_DOMAIN ++ frameBytes node)) 0 0 130023424
  { node, blocks := 130023424, payloadRoot := zeroRoot pd 0 (hidden (zeroPayload key) (zeros SALT_BYTES)),
    codecSeed, codec := honestCodec, receipt := .sketch, rows, key }

/-- The receipts of position 0's leaf: the zero block's receipt under seed `s`, then the 15 others' bytes. -/
def honestLeaf (reg : Registration) (s : ByteArray) : ByteArray :=
  (receipter reg.receipt s reg.rows).receipt 0 (zeros Codec.BLOCK_BYTES) ++ zeros 960

/-- A pass over the whole store under seed `s` whose receipt tree holds `honestLeaf` at leaf 0. -/
def honestPass (reg : Registration) (s : ByteArray) : PassRecord :=
  ⟨zeroRoot (treeDomain s (Plan.whole 130023424)) 0 (hidden (honestLeaf reg s) (zeros SALT_BYTES)),
    Plan.whole 130023424, none⟩

/-- Window 0 of one pass, as the deployed session's windows are. -/
def honestWindow (reg : Registration) (challenge previous : ByteArray) : WindowRecord :=
  ⟨0, challenge, previous, #[honestPass reg (seed challenge 0 previous)]⟩

/-- The opening of position 0 of pass 0. -/
def honestOpening (reg : Registration) (challenge previous : ByteArray) : Opening :=
  let s := seed challenge 0 previous
  ⟨0, 0, zeros Codec.BLOCK_BYTES, honestLeaf reg s, zeros SALT_BYTES, zeroPath (treeDomain s (Plan.whole 130023424)),
    zeros SALT_BYTES, zeroPath reg.payloadDomain, .empty, .empty⟩

theorem honest_checks (reg : Registration) (challenge previous : ByteArray) (hB : reg.blocks = 130023424)
    (hk : reg.key.codec = honestCodec)
    (hroot : reg.payloadRoot = zeroRoot reg.payloadDomain 0 (hidden (zeroPayload reg.key) (zeros SALT_BYTES))) :
    checkOpening reg reg.digest #[] (honestWindow reg challenge previous)
      (receipterOf reg (honestWindow reg challenge previous) 0) 0 0 (honestOpening reg challenge previous) = none := by
  have hs : (treeDomain (seed challenge 0 previous) (Plan.whole 130023424)).start ≤ 0 := Nat.le_refl 0
  have hp : 0 < (treeDomain (seed challenge 0 previous) (Plan.whole 130023424)).start +
      (treeDomain (seed challenge 0 previous) (Plan.whole 130023424)).count := by
    show 0 < 0 + leaves (Plan.whole 130023424).length
    decide
  have hs' : reg.payloadDomain.start ≤ 0 := Nat.le_refl 0
  have hp' : 0 < reg.payloadDomain.start + reg.payloadDomain.count := by
    show 0 < 0 + reg.blocks
    rw [hB]
    decide
  have hb0 : (Plan.whole 130023424).block 0 = 0 := by
    first
      | rfl
      | decide
      | decide +kernel
  have hj : 0 < reg.key.codec.maxBlocks := by
    rw [hk]
    decide
  have hlt : honestCodec.field.c < 2 ^ honestCodec.field.n :=
    Nat.lt_of_lt_of_le (show honestCodec.field.c < 2 ^ 7 by decide)
      (Nat.pow_le_pow_right (by norm_num) (show 7 ≤ honestCodec.field.n by decide))
  have hfp : 0 < reg.key.codec.field.p := by
    rw [hk]
    exact Nat.sub_pos_of_lt hlt
  refine checkOpening_of (pr := honestPass reg (seed challenge 0 previous)) (j := 0)
    (y := (receipter reg.receipt (seed challenge 0 previous) reg.rows).receipt 0 (zeros Codec.BLOCK_BYTES))
    (payload := zeroPayload reg.key) rfl rfl rfl (zeros_size _) rfl (zeros_size _) (zeros_size _) ?_ hb0
    (by rw [hB]; decide) rfl (leafReceipt_first (receipt_size _ _ _)) ?_ (decode_zeroPayload reg.key hj hfp) ?_ rfl
  · intro x hx
    simp only [honestOpening, zeroPath, Array.mem_append, Array.mem_replicate] at hx
    rcases hx with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> exact zeros_size 64
  · unfold verifyLeaf
    exact verifyOpening_zeroRoot (treeDomain (seed challenge 0 previous) (Plan.whole 130023424)) 0
      (hidden (honestLeaf reg (seed challenge 0 previous)) (zeros SALT_BYTES)) hs hp
  · rw [hroot]
    exact verifyOpening_zeroRoot reg.payloadDomain 0 (hidden (zeroPayload reg.key) (zeros SALT_BYTES)) hs' hp'

theorem honestReg_make (node codecSeed : ByteArray) (rows : ℕ) (hnode : 1 ≤ node.size ∧ node.size ≤ 64)
    (hseed : codecSeed.size = Codec.SEED_BYTES) (hrows : rows = 16 ∨ rows = 64) :
    Registration.make? node 130023424 (honestReg node codecSeed rows).payloadRoot codecSeed honestCodec .sketch false
      LAYOUT rows = .ok (honestReg node codecSeed rows) := by
  have hs1 : storeBlocks 130023424 = true := by decide
  have hs2 : 130023424 ≤ honestCodec.maxBlocks := by decide
  have hroot : (honestReg node codecSeed rows).payloadRoot.size = 64 := by
    simp only [honestReg, zeroRoot, climb_size]
  have hrow : Kind.rowsOk .sketch rows = true := by
    rcases hrows with rfl | rfl <;> rfl
  unfold Registration.make?
  simp [hnode.1, hnode.2, hs1, hs2, hroot, hseed, hrow]
  try rfl

/-- **`OpensAtPosition` meets the deployed instance.** The deployed registration (130023424 blocks, p1024/k3/masked,
sketch-sha512 receipts of 16 rows), here of a one-byte node id under the zero codec seed, is one `Registration.make?`
accepts, and in a window of one pass over the whole store, as the deployed session's are, `checkOpening` accepts an
opening of position 0: `OpensAtPosition`'s premise holds. `honestReg_make` and `honest_checks` give the same at any
node id and codec seed of the sizes `Registration.make?` takes, either row count, and any challenge. -/
theorem opensAtPosition_honest :
    ∃ (reg : Registration) (w : WindowRecord) (o : Opening),
      Registration.make? (zeros 1) 130023424 reg.payloadRoot (zeros Codec.SEED_BYTES) ⟨Codec.P1024, 3, true⟩ .sketch
          false LAYOUT 16 = .ok reg ∧
        w.passes.size = 1 ∧ w.passes[0]!.plan = Plan.whole 130023424 ∧
        checkOpening reg reg.digest #[] w (receipterOf reg w 0) 0 0 o = none :=
  ⟨honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16,
    honestWindow (honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16) (zeros 64) (zeros 64),
    honestOpening (honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16) (zeros 64) (zeros 64),
    honestReg_make (zeros 1) (zeros Codec.SEED_BYTES) 16 (by simp [zeros_size]) (zeros_size _) (Or.inl rfl), rfl, rfl,
    honest_checks _ _ _ rfl rfl rfl⟩

/-- **`OpensAtPosition` fails without `checkOpening`.** Its conclusion's first part, that the opened block has a
position, fails at the deployed store for a pass whose plan reads past the store, where no append is: the block at
position 130023424 of a plan of 130023425 blocks has none. A verifier that accepted every opening would accept it. -/
theorem opensAtPosition_nontrivial :
    ∀ t, ¬ (position 130023424 #[] ((Plan.whole 130023425).block 130023424) =
      some (t, (Plan.whole 130023425).block 130023424)) := by
  intro t h
  have hn : position 130023424 #[] ((Plan.whole 130023425).block 130023424) = none := by
    decide +kernel
  exact Option.some_ne_none _ (h.symm.trans hn)

/-! ### The scheduled layout -/

/-- **The schedule check meets the deployed session.** The deployed session commits no schedule cap and appends
nothing, so its ledger asks no schedule; had it committed one, its T = 1 s session's 96 passes over the whole store are
the plans of the schedule with no frees and no appends at steps 0 to 95, which the check accepts at its 792 bits (99
one-byte integers). The premises of `Properties.DeclaredCircuit.ScheduledRequirement` besides the check are `G`'s,
whose honest lemma is `G`'s; at the deployed plan the verifier's layout is store order (`codeLayout_whole`). -/
theorem scheduled_meets : Schedule.check ⟨#[], #[], (List.range 96).toArray⟩ 130023424 792
    ((List.range 96).map fun s => (s, (Plan.whole 130023424).runs)).toArray #[] = .ok 792 := by
  decide +kernel

/-- **`steered_of_checked`'s premise holds at the deployed session's shape.** Over the 130023424-block store with no
appends, at the cap of 792 bits, a schedule's check accepts passes whose pass at each of the T = 1 s session's 96 steps
sent the whole store's runs (`scheduled_meets`): `checked`, with the pass at prefix `c` the `c`-th and every coin's
plan the whole store. -/
theorem steered_of_checked_honest :
    ∃ (sc : Schedule.Schedule) (passes : Array (ℕ × Array (ℕ × ℕ))) (b : ℕ),
      Schedule.check sc 130023424 792 passes #[] = .ok b ∧
        ∀ s < 96, (passes[s]?).map Prod.snd = some (Plan.whole 130023424).runs :=
  ⟨_, _, _, scheduled_meets, by first | decide +kernel | (intro s hs; simp [hs])⟩

/-- No appends. -/
def noAppends : Unit → Fin 3 → Array (ℕ × ℕ) := fun _ _ => #[]

/-- A pass of one block that reads block `y` of a store of three. -/
def coinRun : Unit → Fin 3 → Array (ℕ × ℕ) := fun _ y => #[((y : ℕ), 1)]

/-- One tile at block `y` of the registration's tree. -/
def threeLayout (y : Fin 3) : Fin 1 ↪ Fin 1 × Fin 3 := ⟨fun _ => (0, y), fun _ _ _ => Subsingleton.elim _ _⟩

theorem runBlocks_coinRun : ∀ y : Fin 3, (runBlocks (coinRun () y)).length = 1 := by decide

theorem tileAt_coinRun : ∀ (y : Fin 3) (i : Fin 1),
    tileAt 3 #[] (coinRun () y) i = some (((threeLayout y i).1 : ℕ), ((threeLayout y i).2 : ℕ)) := by
  decide

theorem sessionLayout_coinRun (dflt : Unit → Fin 1 ↪ Fin 1 × Fin 3) (y : Fin 3) :
    sessionLayout 3 dflt noAppends coinRun () y = threeLayout y :=
  codeLayout_eq (threeLayout y) (runBlocks_coinRun y) (tileAt_coinRun y)

/-- **`steered_of_checked` fails without the schedule check.** A verifier that accepted every session would meet its
premise for the pass that reads the block a coin `y ∈ {0, 1, 2}` names, at cap 0; the verifier's layout there takes
three values, which no one bit steers. The real check refuses every schedule at cap 0 and a coin-steered plan at any
cap (`coinPasses_refused`). Stated at the default layout of block 0; the layout there is the plan's whatever the
default (`sessionLayout_coinRun`). -/
theorem steered_of_checked_nontrivial :
    ∀ steer : Unit → Fin 3 → Pous.Bits (0 + 1),
      ¬ Accounting.Assumptions.SteeredLayout
        (sessionLayout 3 (fun _ : Unit => (⟨fun _ => (0, 0), fun _ _ _ => Subsingleton.elim _ _⟩ : Fin 1 ↪ Fin 1 × Fin 3))
          (fun (_ : Unit) (_ : Fin 3) => (#[] : Array (ℕ × ℕ))) (fun (_ : Unit) (y : Fin 3) => #[((y : ℕ), 1)]))
        steer := by
  intro steer hs
  change Accounting.Assumptions.SteeredLayout (sessionLayout 3 (fun _ => threeLayout 0) noAppends coinRun) steer at hs
  obtain ⟨slot, h⟩ := hs
  have key : ∀ y y' : Fin 3, steer () y 0 = steer () y' 0 → y = y' := by
    intro y y' hs
    have hs' : steer () y = steer () y' := funext fun i => by
      obtain ⟨i, hi⟩ := i
      obtain rfl : i = 0 := by omega
      exact hs
    exact congrArg (fun e : Fin 1 ↪ Fin 1 × Fin 3 => (e 0).2)
      ((sessionLayout_coinRun (fun _ => threeLayout 0) y).symm.trans ((h () y).trans
        ((congrArg (slot ()) hs').trans ((h () y').symm.trans (sessionLayout_coinRun (fun _ => threeLayout 0) y')))))
  obtain ⟨y, y', hne, he⟩ := Fintype.exists_ne_map_eq_of_card_lt (fun y : Fin 3 => steer () y 0) (by simp)
  exact hne (key y y' he)

end Properties.RemoteAudit
