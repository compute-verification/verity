import FlockSoundness.Refine.Regions
import FlockSoundness.Refine.Walk
import Flock.HmRow
import FlockSoundness.Protocol.Refine.Setup

/-!
# The statement decoding, part 3: the setup's statements are well formed (PROTOCOL.md §16.1, §16.2)

`StmtWF` and `RegionsWF` are the facts `ofCircuit_fold` and `ofCircuit_extra` take. `setupH_wf` establishes them for
`Stmt.setupH`.
* **The slots.** `HmRow.parse` ends with `HmRow.check`, which refuses a `slot_log` outside 7..k_log (`hm_check_slots`).
* **The pin.** `HmRow.pin` refuses a column outside the block (`hm_pin_ok`).
* **The regions.** Every region is `mkRegion`'s, and `mkRegion_ok` gives distinct free bits below `k_log`, at most
  `PT_LOCAL` of them (`hm_regions_wf`, `outRegion_wf`).
* `m = k_log + nbl` and `m_pts = PT_LOCAL + nbl` are the setup's own.

`stmtOf_linkLayout` gives `stmtOf st`'s `LinkLayout`, which the table-soundness theorems take.
-/

set_option Elab.async false

namespace FlockSoundness.Refine

open Flock FlockLevel3

/-! ### Loops -/

theorem ite_throw_ok {α γ : Type} {c : Prop} [Decidable c] {e : String} {f : γ → V α} {x : V α} {v : α} :
    ((if c then ((throw e : V γ) >>= f) else x) = Except.ok v) ↔ (¬c ∧ x = .ok v) := by
  by_cases hc : c
  · simp only [hc, ite_true, not_true_eq_false, false_and, iff_false]
    intro h; cases h
  · simp only [hc, ite_false, not_false_eq_true, true_and]

theorem ite_throw_ok' {α : Type} {c : Prop} [Decidable c] {e : String} {x : V α} {v : α} :
    ((if c then (throw e : V α) else x) = .ok v) ↔ (¬c ∧ x = .ok v) := by
  by_cases hc : c
  · simp only [hc, ite_true, not_true_eq_false, false_and, iff_false]; exact throw_ne_ok e v
  · simp only [hc, ite_false, not_false_eq_true, true_and]

/-! ### Regions -/

theorem outRegion_wf {c : Circuit} {pub : Public} {r : Region} (h : c.outRegion pub = .ok r) : RegionWF c.kLog r := by
  unfold Circuit.outRegion at h
  exact mkRegion_wf h

/-! ### The pin -/

/-! ### The slots -/

/-! ### The regions the setup builds -/

/-! ### The setup -/

/-- `HmRow.check`'s mask range: unpacked, one slot of at least a word, inside the block. -/
theorem hm_check_mask {c : Flock.Circuit} {tags : Tags} {ls : Lean.Json} (h : HmRow.check c tags ls = .ok ()) :
    ∀ r ∈ c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ c.kLog := by
  unfold HmRow.check at h
  dsimp only at h
  obtain ⟨_, hloop, -⟩ := ebind_ok h
  clear h
  rw [← Array.forIn_toList] at hloop
  refine list_forIn_each _ (fun r => r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ c.kLog)
    ?_ _ _ _ hloop
  clear hloop
  intro r s y hy
  have hp : 1 ≤ nextPow2 r.count := Nat.one_le_two_pow
  repeat' (first | (obtain ⟨_, hx, hy⟩ := ebind_ok hy; try (cases hx)) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    refine ⟨fun hk => ?_, _, rfl⟩
    simp only [hk, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, not_or, not_lt, not_and] at *
    first
    | omega
    | (constructor; omega; nlinarith [Nat.one_le_two_pow (n := r.slotLog)])
    | simp_all

set_option maxHeartbeats 2000000 in
/-- **`Stmt.setupH`'s mask slot is a slot of at least a word inside the block** (`HmRow.check`). -/
theorem setupH_mask {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st) :
    (∀ r ∈ st.c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog) ∧
      st.m = st.c.kLog + st.nbl := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨_, hc⟩ : ∃ ls, HmRow.check _ tags ls = .ok () := by
      first | exact hm_parse_check (by assumption) | exact hm_parseTyped_check (by assumption)
    exact ⟨hm_check_mask hc, rfl⟩

end FlockSoundness.Refine
