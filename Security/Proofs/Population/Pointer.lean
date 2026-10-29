import Proofs.Population.Egress
import Mathlib.Tactic.DeriveFintype

/-!
# Value-dependent reads: the VM pointer

A machine trace is a `DynPopulation`: a step reads memory at an address it holds, so its read set is a function of
values. Two things are true of it.

* `DynPopulation.card_egress_le_of_timeOrdered`, **the count still holds**: if every step reads, in every assignment,
  only inputs and slots written at earlier steps, then with the inputs fixed the egress takes at most
  `|V| ^ |writes B|` values at the wrong set `B`, as for a static population. A program's trace bounds the prover's
  choice of egress as well as a circuit does.
* `vm_leak`, **the reach does not**: in the pointer machine `vm N`, every assignment's dependency graph is acyclic
  (`vm_acyclicAt`), the writes are single assignment (`vm_singleAssignment`), and in every honest run the egress reads
  no secret (`vm_honest_up`); yet one faulty proof block, the pointer step writing one bit, makes the egress copy all
  `N` secret bits, so over the secrets the egress takes `2 ^ N` values. `vm_count` is the same machine under the count:
  with the secrets fixed (as inputs), at most `2 ^ 1`.

The same computation declared statically (`vmStatic`, each copy step reading the pointer, its zero slot and its secret
slot) is a checked population (`vmStatic_checked`) whose egress has every secret upstream (`vmStatic_secret_up`):
the checker sees the secret in the egress's cone and refuses to certify it, which is what `card_egress_le` needs. A
circuit does not stop a value from selecting what is read; its public wiring makes the selection visible.
-/

namespace Population

open Finset

namespace DynPopulation

variable {U S V : Type} (D : DynPopulation U S V)

theorem correct_of_not_mem_wrong [Fintype U] {σ : S → V} {u : U} (h : u ∉ D.wrong σ) : D.Correct σ u := by
  classical
  by_contra hc
  exact h (by simp [wrong, hc])

/-- **Time-ordered determinacy**: an assignment of a time-ordered population is determined by its inputs and the
writes of a set containing its faulty proof blocks. -/
theorem eq_of_timeOrdered [Fintype U] (hc : D.Covered) {step : U → ℕ} (ht : D.TimeOrdered step) {B : Finset U}
    {σ τ : S → V} (hin : ∀ s ∈ D.input, σ s = τ s) (hσ : D.wrong σ ⊆ B) (hτ : D.wrong τ ⊆ B)
    (hB : ∀ u ∈ B, ∀ s ∈ D.writes u, σ s = τ s) : σ = τ := by
  have key : ∀ u, ∀ s ∈ D.writes u, σ s = τ s := by
    intro u
    induction u using (measure step).wf.induction with
    | _ u ih =>
      by_cases huB : u ∈ B
      · exact hB u huB
      · intro s hs
        have cσ := D.correct_of_not_mem_wrong (σ := σ) fun h => huB (hσ h)
        have cτ := D.correct_of_not_mem_wrong (σ := τ) fun h => huB (hτ h)
        rw [cσ s hs, cτ s hs]
        refine D.reads_only u σ τ (fun r hr => ?_) s hs
        rcases ht u σ r hr with hri | ⟨v, hv, hlt⟩
        · exact hin r hri
        · exact ih v hlt r hv
  funext s
  by_cases hs : s ∈ D.input
  · exact hin s hs
  · obtain ⟨v, hv⟩ := hc s hs
    exact key v s hv

/-- **The count for a time-ordered trace**: with every input fixed, at most `|V| ^ |writes B|` egress values. -/
theorem card_egress_le_of_timeOrdered [Fintype U] [Fintype S] [Fintype V] (hc : D.Covered) {step : U → ℕ}
    (ht : D.TimeOrdered step) (a : S → V) (O : Finset S) (B : Finset U) :
    (D.egress a ∅ O B).card ≤ Fintype.card V ^ (D.writesOf B).card := by
  classical
  have key : ∀ e ∈ D.egress a ∅ O B,
      ∃ σ : S → V, (D.Holds a ∅ σ ∧ D.wrong σ ⊆ B) ∧ (fun o : O => σ o) = e := by
    intro e he
    simpa only [egress, mem_image, mem_filter, mem_univ, true_and] using he
  let g : D.egress a ∅ O B → (D.writesOf B → V) := fun e s => Classical.choose (key e.1 e.2) s
  have hg : Function.Injective g := by
    intro e e' h
    obtain ⟨⟨hσa, hσw⟩, hσe⟩ := Classical.choose_spec (key e.1 e.2)
    obtain ⟨⟨hτa, hτw⟩, hτe⟩ := Classical.choose_spec (key e'.1 e'.2)
    apply Subtype.ext
    rw [← hσe, ← hτe]
    have := D.eq_of_timeOrdered hc ht (fun s hs => (hσa s hs (notMem_empty s)).trans
      (hτa s hs (notMem_empty s)).symm) hσw hτw fun u hu s hs => by
        have hsB : s ∈ D.writesOf B := by
          simp only [writesOf, mem_biUnion]
          exact ⟨u, hu, hs⟩
        exact congrFun h ⟨s, hsB⟩
    rw [this]
  calc (D.egress a ∅ O B).card = Fintype.card (D.egress a ∅ O B) := (Fintype.card_coe _).symm
    _ ≤ Fintype.card (D.writesOf B → V) := Fintype.card_le_of_injective g hg
    _ = Fintype.card V ^ (D.writesOf B).card := by rw [Fintype.card_fun, Fintype.card_coe]

end DynPopulation

/-! ## The pointer machine -/

/-- The pointer machine's memory: the pointer, a public zero region, a secret region, and the output region. -/
inductive Slot (N : ℕ)
  | ptr
  | zero (i : Fin N)
  | secret (i : Fin N)
  | out (i : Fin N)
  deriving DecidableEq, Fintype

/-- The memory the machine is given: both regions. -/
def Slot.isInput {N : ℕ} : Slot N → Bool
  | .zero _ => true
  | .secret _ => true
  | _ => false

/-- The secret region. -/
def Slot.isSecret {N : ℕ} : Slot N → Bool
  | .secret _ => true
  | _ => false

/-- The output region, which is the egress. -/
def Slot.isOut {N : ℕ} : Slot N → Bool
  | .out _ => true
  | _ => false

@[simp] theorem Slot.isInput_ptr {N : ℕ} : (Slot.ptr : Slot N).isInput = false := rfl
@[simp] theorem Slot.isInput_zero {N : ℕ} (i : Fin N) : (Slot.zero i).isInput = true := rfl
@[simp] theorem Slot.isInput_secret {N : ℕ} (i : Fin N) : (Slot.secret i).isInput = true := rfl
@[simp] theorem Slot.isInput_out {N : ℕ} (i : Fin N) : (Slot.out i).isInput = false := rfl
@[simp] theorem Slot.isSecret_ptr {N : ℕ} : (Slot.ptr : Slot N).isSecret = false := rfl
@[simp] theorem Slot.isSecret_zero {N : ℕ} (i : Fin N) : (Slot.zero i).isSecret = false := rfl
@[simp] theorem Slot.isSecret_secret {N : ℕ} (i : Fin N) : (Slot.secret i).isSecret = true := rfl
@[simp] theorem Slot.isSecret_out {N : ℕ} (i : Fin N) : (Slot.out i).isSecret = false := rfl
@[simp] theorem Slot.isOut_ptr {N : ℕ} : (Slot.ptr : Slot N).isOut = false := rfl
@[simp] theorem Slot.isOut_zero {N : ℕ} (i : Fin N) : (Slot.zero i).isOut = false := rfl
@[simp] theorem Slot.isOut_secret {N : ℕ} (i : Fin N) : (Slot.secret i).isOut = false := rfl
@[simp] theorem Slot.isOut_out {N : ℕ} (i : Fin N) : (Slot.out i).isOut = true := rfl

/-- Where copy step `i` reads, at pointer value `p`: the secret region when `p` is set, the zero region otherwise. -/
def src {N : ℕ} (p : Bool) (i : Fin N) : Slot N := if p then .secret i else .zero i

/-- **The pointer machine**: step `none` sets the pointer (honestly to the zero region), and copy step `some i` copies
word `i` of the region the pointer names to output word `i`. Copy step `i` reads the pointer and the one word it names. -/
def vm (N : ℕ) : DynPopulation (Option (Fin N)) (Slot N) Bool where
  input := univ.filter fun s => s.isInput
  writes
    | none => {.ptr}
    | some i => {.out i}
  reads
    | none => fun _ => ∅
    | some i => fun σ => {.ptr, src (σ .ptr) i}
  f
    | none => fun _ _ => false
    | some i => fun σ _ => σ (src (σ .ptr) i)
  reads_only := by
    intro u σ τ h s _
    cases u with
    | none => rfl
    | some i =>
      have hp : σ .ptr = τ .ptr := h .ptr (by simp)
      have hs : σ (src (σ .ptr) i) = τ (src (σ .ptr) i) := h _ (by simp)
      show σ (src (σ .ptr) i) = τ (src (τ .ptr) i)
      rw [← hp, hs]

theorem vm_singleAssignment (N : ℕ) : (vm N).SingleAssignment := by
  intro s
  cases s with
  | ptr => exact ⟨fun h => by simp [vm] at h, fun _ => ⟨none, by simp [vm], fun u hu => by
      cases u <;> simp_all [vm]⟩⟩
  | zero i => exact ⟨fun _ u => by cases u <;> simp [vm], fun h => by simp [vm] at h⟩
  | secret i => exact ⟨fun _ u => by cases u <;> simp [vm], fun h => by simp [vm] at h⟩
  | out i => exact ⟨fun h => by simp [vm] at h, fun _ => ⟨some i, by simp [vm], fun u hu => by
      cases u <;> simp_all [vm]⟩⟩

/-- Every assignment's dependency graph is acyclic: the pointer step reads nothing, and no step reads an output. -/
theorem vm_acyclicAt (N : ℕ) (σ : Slot N → Bool) : (vm N).AcyclicAt σ := by
  intro u h
  obtain ⟨b, -, hbu⟩ := Relation.TransGen.tail'_iff.1 h
  obtain ⟨c, huc, -⟩ := Relation.TransGen.head'_iff.1 h
  obtain ⟨s, hs, hsr⟩ := hbu
  obtain ⟨t, ht, htr⟩ := huc
  cases u with
  | none => simp [vm] at hsr
  | some i =>
    have : t = .out i := by simpa [vm] using ht
    subst this
    cases c with
    | none => simp [vm] at htr
    | some j => simp [vm, src] at htr; split at htr <;> simp at htr

theorem vm_timeOrdered (N : ℕ) : (vm N).TimeOrdered fun u => if u.isSome then 1 else 0 := by
  intro u σ s hs
  cases u with
  | none => simp [vm] at hs
  | some i =>
    simp only [vm, mem_insert, mem_singleton] at hs
    rcases hs with rfl | rfl
    · exact Or.inr ⟨none, by simp [vm]⟩
    · left
      simp [vm, src]
      split <;> simp []

theorem vm_covered (N : ℕ) : (vm N).Covered := fun s hs => (((vm_singleAssignment N) s).2 hs).exists

/-- **The honest run never reads a secret**: in every assignment where every step is correct, the pointer names the
zero region and no secret word is upstream of the egress. -/
theorem vm_honest_up (N : ℕ) {σ : Slot N → Bool} (hσ : ∀ u, (vm N).Correct σ u) (i : Fin N) :
    Slot.secret i ∉ (vm N).up σ (univ.filter fun s => s.isOut) := by
  have hp : σ .ptr = false := hσ none .ptr (by simp [vm])
  rintro ⟨o, ho, hro⟩
  rcases Relation.ReflTransGen.cases_head hro with e | ⟨c, ⟨u, hu, -⟩, -⟩
  · subst e
    simp [] at ho
  · cases u with
    | none => simp [vm] at hu
    | some j => simp [vm, src, hp] at hu

/-- **One wrong one-bit unit reroutes all `N` secret words**: over the secrets, with the public inputs fixed and only
the pointer step wrong, the egress takes `2 ^ N` values. -/
theorem vm_leak (N : ℕ) :
    2 ^ N ≤ ((vm N).egress (fun _ => false) (univ.filter fun s => s.isSecret) (univ.filter fun s => s.isOut)
      {none}).card := by
  classical
  let O : Finset (Slot N) := univ.filter fun s => s.isOut
  let σ : (Fin N → Bool) → Slot N → Bool := fun x s => match s with
    | .ptr => true
    | .zero _ => false
    | .secret i => x i
    | .out i => x i
  have hmem : ∀ x, (fun o : O => σ x o) ∈ (vm N).egress (fun _ => false) (univ.filter fun s => s.isSecret) O
      {none} := by
    intro x
    simp only [DynPopulation.egress, mem_image, mem_filter, mem_univ, true_and]
    refine ⟨σ x, ⟨fun s hs hsX => ?_, fun u hu => ?_⟩, rfl⟩
    · cases s <;> simp_all [vm, σ]
    · simp only [DynPopulation.wrong, mem_filter, mem_univ, true_and] at hu
      cases u with
      | none => simp
      | some i =>
        exfalso
        refine hu fun s hs => ?_
        have : s = .out i := by simpa [vm] using hs
        subst this
        simp [vm, src, σ]
  have hinj : Function.Injective fun x (o : O) => σ x o := by
    intro x y h
    funext i
    have := congrFun h ⟨.out i, by simp [O]⟩
    simpa [σ] using this
  have := card_le_card_of_injOn _ (s := univ) (fun x _ => hmem x) hinj.injOn
  simpa using this

/-- **The count still holds for the pointer machine**: with the secrets fixed as inputs, the same faulty proof block
gives at most `2 ^ 1` egress values. -/
theorem vm_count (N : ℕ) (a : Slot N → Bool) :
    ((vm N).egress a ∅ (univ.filter fun s => s.isOut) {none}).card ≤ 2 ^ 1 := by
  classical
  have := (vm N).card_egress_le_of_timeOrdered (vm_covered N) (vm_timeOrdered N) a
    (univ.filter fun s => s.isOut) {none}
  have hw : ((vm N).writesOf {none}).card = 1 := by
    simp [DynPopulation.writesOf, vm]
  rw [hw, Fintype.card_bool] at this
  exact this

/-! ## The same computation, declared statically -/

/-- The pointer machine with public read sets: copy step `i` reads the pointer, its zero word and its secret word, and
selects between them inside the unit. -/
def vmStatic (N : ℕ) : Dataflow (Option (Fin N)) (Slot N) where
  input := univ.filter fun s => s.isInput
  reads
    | none => ∅
    | some i => {.ptr, .zero i, .secret i}
  writes
    | none => {.ptr}
    | some i => {.out i}

theorem vmStatic_checked (N : ℕ) : (vmStatic N).Checked := by
  refine ⟨fun s => ?_, fun u h => ?_⟩
  · cases s with
    | ptr => exact ⟨fun h => by simp [vmStatic] at h, fun _ => ⟨none, by simp [vmStatic],
        fun u hu => by cases u <;> simp_all [vmStatic]⟩⟩
    | zero i => exact ⟨fun _ u => by cases u <;> simp [vmStatic], fun h => by simp [vmStatic] at h⟩
    | secret i => exact ⟨fun _ u => by cases u <;> simp [vmStatic], fun h => by simp [vmStatic] at h⟩
    | out i => exact ⟨fun h => by simp [vmStatic] at h, fun _ => ⟨some i, by simp [vmStatic],
        fun u hu => by cases u <;> simp_all [vmStatic]⟩⟩
  · obtain ⟨b, -, hbu⟩ := Relation.TransGen.tail'_iff.1 h
    obtain ⟨c, huc, -⟩ := Relation.TransGen.head'_iff.1 h
    obtain ⟨s, hs, hsr⟩ := hbu
    obtain ⟨t, ht, htr⟩ := huc
    cases u with
    | none => simp [vmStatic] at hsr
    | some i =>
      have : t = .out i := by simpa [vmStatic] using ht
      subst this
      cases c <;> simp [vmStatic] at htr

/-- **The checker sees the secret**: in the static declaration every secret word is upstream of the egress. -/
theorem vmStatic_secret_up (N : ℕ) (i : Fin N) :
    Slot.secret i ∈ (vmStatic N).up (univ.filter fun s => s.isOut) :=
  ⟨.out i, by simp [], .single ⟨some i, by simp [vmStatic], by simp [vmStatic]⟩⟩

end Population
