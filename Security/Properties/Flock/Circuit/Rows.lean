import Mathlib.Tactic.Ring
import Definitions.Flock.Lowering
import Flock.Net

/-!
# The rows the verifier parses, as the audit's `Rows`

`Net.checkOrder`'s accept as `RowOrder` (`checkOrder_ok`), the audit's `Rows` from ordered rows (`Rows.ofOrdered`)
and from a net `Net.ofRows` built (`Built`, `Rows.ofNet`), and a template instance's rows: `k` copies of a
template's, stacked (`Rows.stack`). Moved from `Proofs.Flock.Soundness.ExecRows` unchanged, with the lemmas these
definitions' bodies use; the other lemmas stay there.
-/

namespace FlockSoundness

open Flock

/-! ## Loops that only continue or throw -/

theorem forIn_list_yield_ok {α ε : Type} {l : List α} {f : α → PUnit → Except ε (ForInStep PUnit)}
    (hf : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn l PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ l, f a PUnit.unit = .ok (.yield PUnit.unit) := by
  induction l with
  | nil => intro a ha; cases ha
  | cons x l ih =>
    rw [List.forIn_cons] at h
    rcases hx : f x PUnit.unit with e | s
    · rw [hx] at h; cases h
    · have hs := hf x s hx
      subst hs
      rw [hx] at h
      intro a ha
      rcases List.mem_cons.1 ha with rfl | ha
      · exact hx
      · exact ih h a ha

theorem forIn_array_yield_ok {α ε : Type} {xs : Array α} {f : α → PUnit → Except ε (ForInStep PUnit)}
    (hf : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn xs PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ xs.toList, f a PUnit.unit = .ok (.yield PUnit.unit) :=
  forIn_list_yield_ok hf (by rw [Array.forIn_toList]; exact h)

theorem forIn_range_yield_ok {ε : Type} {lo hi : ℕ} {f : ℕ → PUnit → Except ε (ForInStep PUnit)}
    (hf : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn [lo:hi] PUnit.unit f = .ok PUnit.unit) :
    ∀ r, lo ≤ r → r < hi → f r PUnit.unit = .ok (.yield PUnit.unit) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  intro r hlo hhi
  refine forIn_list_yield_ok hf h r ?_
  simp only [Std.Legacy.Range.size, List.mem_range'_1]
  omega

/-! ## The order `Net.checkOrder` checks -/

/-- Column `c` is **input padding**: an input column whose rows are empty in `A` and in `B`. -/
def InputPadding (inCols : ℕ) (ra rb : Array (Array ℕ)) (c : ℕ) : Prop :=
  c < inCols ∧ (ra.getD c #[]).isEmpty = true ∧ (rb.getD c #[]).isEmpty = true

/-- **Row order** (`Rows.topo`, as the parser checks it): every computed row
`inCols ≤ r < constPos` reads, in `A` and in `B`, only the constant or an earlier column that is not
input padding. -/
def RowOrder (inCols constPos : ℕ) (ra rb : Array (Array ℕ)) : Prop :=
  ∀ r, inCols ≤ r → r < constPos → ∀ c ∈ (ra.getD r #[]).toList ++ (rb.getD r #[]).toList,
    c = constPos ∨ (c < r ∧ ¬ InputPadding inCols ra rb c)

/-- **The parser's order check, accepted, is `RowOrder`.** -/
theorem checkOrder_ok {name : String} {inCols constPos : ℕ} {ra rb : Array (Array ℕ)}
    (h : Net.checkOrder name inCols constPos ra rb = .ok ()) : RowOrder inCols constPos ra rb := by
  unfold Net.checkOrder at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i v hloop
    cases v
    have hout := forIn_range_yield_ok (fun a s hs => by
      split at hs
      · cases hs
      · cases hs; rfl) hloop
    intro r hlo hhi c hc
    have hr := hout r hlo hhi
    split at hr
    · cases hr
    · rename_i w hinner
      cases w
      have hin := forIn_array_yield_ok (fun a s hs => by
        split at hs
        · cases hs; rfl
        · split at hs
          · simp [throw, throwThe, MonadExceptOf.throw] at hs
          · split at hs
            · simp [throw, throwThe, MonadExceptOf.throw] at hs
            · cases hs; rfl) hinner c (by rw [Array.toList_append]; exact hc)
      by_cases hk : c = constPos
      · exact Or.inl hk
      · refine Or.inr ?_
        simp only [hk, beq_iff_eq, ite_false] at hin
        split at hin
        · simp [throw, throwThe, MonadExceptOf.throw] at hin
        · rename_i hlt
          split at hin
          · simp [throw, throwThe, MonadExceptOf.throw] at hin
          · rename_i hpad
            refine ⟨by omega, fun ⟨h1, h2, h3⟩ => hpad ?_⟩
            rw [h2, h3]
            simp [h1]

/-! ## The audit's `Rows` from ordered rows -/

/-- A column a computed row reads is the constant or an earlier column, so below `constPos + 1`. -/
theorem RowOrder.lt {inCols constPos : ℕ} {ra rb : Array (Array ℕ)} (h : RowOrder inCols constPos ra rb)
    {r : ℕ} (hlo : inCols ≤ r) (hhi : r < constPos) {c : ℕ}
    (hc : c ∈ (ra.getD r #[]).toList ++ (rb.getD r #[]).toList) : c = constPos ∨ c < r :=
  (h r hlo hhi c hc).imp id And.left

/-- **The template's rows, from rows the parser ordered**: columns below `min inCols constPos` are
its inputs, rows from there to `constPos − 1` are computed, and column `constPos` is its constant.
(A net whose constant sits among its input rows has no computed rows.) -/
def Rows.ofOrdered (inCols constPos : ℕ) (ra rb : Array (Array ℕ)) (h : RowOrder inCols constPos ra rb) :
    Rows where
  nIn := min inCols constPos
  nComp := constPos - min inCols constPos
  a i := (ra.getD (min inCols constPos + i.val) #[]).toList.pmap (fun c hc => ⟨c, hc⟩) fun c hc => by
    have hi := i.isLt
    rcases h.lt (r := min inCols constPos + i.val) (by omega) (by omega) (List.mem_append_left _ hc) with
      hc' | hc' <;> omega
  b i := (rb.getD (min inCols constPos + i.val) #[]).toList.pmap (fun c hc => ⟨c, hc⟩) fun c hc => by
    have hi := i.isLt
    rcases h.lt (r := min inCols constPos + i.val) (by omega) (by omega) (List.mem_append_right _ hc) with
      hc' | hc' <;> omega
  topo i c hc := by
    have hi := i.isLt
    have hlo : inCols ≤ min inCols constPos + i.val := by omega
    have hhi : min inCols constPos + i.val < constPos := by omega
    have hone : min inCols constPos + (constPos - min inCols constPos) = constPos := by omega
    rcases List.mem_append.1 hc with hc | hc <;>
    · obtain ⟨c₀, hc₀, rfl⟩ := List.mem_pmap.1 hc
      rcases h.lt hlo hhi
          (by first | exact List.mem_append_left _ hc₀ | exact List.mem_append_right _ hc₀) with
        h' | h'
      · exact Or.inr (by show c₀ = _; rw [hone]; exact h')
      · exact Or.inl h'

/-- **A net `Net.ofRows` built**: the parser's (`parse_built`), or a typed class's from `derive`'s rows
(`Typed.netOfD`). -/
def Built (net : Net) : Prop :=
  ∃ name useful constPos ig og ra rb, Net.ofRows name useful constPos ig og ra rb = .ok net

/-- **What `Net.ofRows` built**: its order check passed on the very rows it stored as `A` and `B`. -/
theorem built_checkOrder {net : Net} (h : Built net) :
    ∃ ra rb, Net.checkOrder "flock-ir-unit/v2" (net.inWords * WORD) net.constPos ra rb = .ok () ∧
      Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b := by
  obtain ⟨_, _, _, _, _, ra, rb, ho⟩ := h
  obtain ⟨-, ha, hb, -, hc, -, -, -, -, -, hco, -, -, -⟩ := Net.ofRows_ok ho
  exact ⟨ra, rb, by rw [hc]; exact hco, ha, hb⟩

/-- **Every net `Net.ofRows` builds is ordered**: the rows it stored as `A` and `B` have `RowOrder`. -/
theorem built_rowOrder {net : Net} (h : Built net) :
    ∃ ra rb, RowOrder (net.inWords * WORD) net.constPos ra rb ∧
      Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b :=
  let ⟨ra, rb, hco, ha, hb⟩ := built_checkOrder h
  ⟨ra, rb, checkOrder_ok hco, ha, hb⟩

/-- **The rows of a net `Net.ofRows` built** (the parser's, or a typed class's), as the audit's `Rows`: its computed rows
are `net.a.row r`, `net.b.row r` (`Rows.ofNet_a`, `Rows.ofNet_b`), and `topo` holds because `Net.ofRows` checked it
(`built_rowOrder`). -/
noncomputable def Rows.ofNet {net : Net} (h : Built net) : Rows :=
  Rows.ofOrdered (net.inWords * WORD) net.constPos (Classical.choose (built_rowOrder h))
    (Classical.choose (Classical.choose_spec (built_rowOrder h)))
    (Classical.choose_spec (Classical.choose_spec (built_rowOrder h))).1

/-! ## An instance's rows: `k` copies of the template's, stacked -/

namespace Rows

variable (R : Rows) (k : ℕ)

theorem mul_add_lt {u k n c : ℕ} (hu : u < k) (hc : c < n) : u * n + c < k * n := by
  have : u * n + n ≤ k * n := by
    calc u * n + n = (u + 1) * n := by ring
      _ ≤ k * n := Nat.mul_le_mul_right _ hu
  omega

/-- The columns of `k` stacked copies. -/
abbrev stackW : ℕ := k * R.nIn + k * (R.nComp + 1) + 1

/-- **Copy `u`'s column `c`, in the stacked order**: its input `c` is input `u·nIn + c`; its computed
row `c − nIn` follows the copy's constant row; its constant is the copy's constant row. -/
def stackCol {u : ℕ} (hu : u < k) (c : Fin R.w) : Fin (R.stackW k) :=
  if hc : c.val < R.nIn then ⟨u * R.nIn + c.val, by
    have := mul_add_lt hu hc; show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩
  else if hc' : c.val < R.nIn + R.nComp then
    ⟨k * R.nIn + (u * (R.nComp + 1) + (1 + (c.val - R.nIn))), by
      have := mul_add_lt hu (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
      show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩
  else ⟨k * R.nIn + (u * (R.nComp + 1) + 0), by
      have := mul_add_lt hu (show 0 < R.nComp + 1 by omega)
      show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩

theorem stack_copy_lt {i : ℕ} (hi : i < k * (R.nComp + 1)) : i / (R.nComp + 1) < k :=
  (Nat.div_lt_iff_lt_mul (by omega)).2 hi

/-- Row `i` of the stack, from a copy's rows `f`: the copy's constant row reads the shared constant,
and the copy's computed row `j` reads its own columns. -/
def stackRow (f : Fin R.nComp → List (Fin R.w)) (i : Fin (k * (R.nComp + 1))) : List (Fin (R.stackW k)) :=
  if hj : i.val % (R.nComp + 1) = 0 then [⟨k * R.nIn + k * (R.nComp + 1), by
    show _ < k * R.nIn + k * (R.nComp + 1) + 1; omega⟩]
  else (f ⟨i.val % (R.nComp + 1) - 1, by have := Nat.mod_lt i.val (show 0 < R.nComp + 1 by omega); omega⟩).map
    (R.stackCol k (R.stack_copy_lt k i.isLt))

theorem stackRow_topo (f : Fin R.nComp → List (Fin R.w))
    (hf : ∀ j, ∀ c ∈ f j, c.val < R.nIn + j.val ∨ c.val = R.nIn + R.nComp)
    (i : Fin (k * (R.nComp + 1))) (c : Fin (R.stackW k)) (hc : c ∈ R.stackRow k f i) :
    c.val < k * R.nIn + i.val ∨ c.val = k * R.nIn + k * (R.nComp + 1) := by
  have hdm := Nat.div_add_mod' i.val (R.nComp + 1)
  have hmod := Nat.mod_lt i.val (show 0 < R.nComp + 1 by omega)
  unfold stackRow at hc
  split at hc
  · simp only [List.mem_singleton] at hc
    exact Or.inr (by rw [hc])
  · rename_i hj
    obtain ⟨c₀, hc₀, rfl⟩ := List.mem_map.1 hc
    refine Or.inl ?_
    have ht := hf _ c₀ hc₀
    simp only at ht
    unfold stackCol
    split
    · rename_i h1
      have := mul_add_lt (R.stack_copy_lt k i.isLt) h1
      simp only
      omega
    · split
      · simp only
        omega
      · simp only
        omega

/-- **An instance's rows**: `k` copies of the template's rows `R`, in the order `Rows` needs. First
every copy's inputs; then, copy by copy, the copy's constant (a computed row reading the shared
constant) and its computed rows; then the shared constant. -/
def stack : Rows where
  nIn := k * R.nIn
  nComp := k * (R.nComp + 1)
  a := R.stackRow k R.a
  b := R.stackRow k R.b
  topo i c hc := by
    rcases List.mem_append.1 hc with hc | hc
    · exact R.stackRow_topo k R.a (fun j c hc => R.topo j c (List.mem_append_left _ hc)) i c hc
    · exact R.stackRow_topo k R.b (fun j c hc => R.topo j c (List.mem_append_right _ hc)) i c hc

end Rows

end FlockSoundness
