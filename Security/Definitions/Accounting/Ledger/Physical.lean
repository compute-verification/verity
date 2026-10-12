import Definitions.Accounting.Ledger.Basic

/-!
# Physical assumptions over the ledger (trusted)

Physical assumptions are about every actual circuit the declared hardware can run. Here they are about a family of
actual circuits `A : X → Annotation Λ G`, indexed by whatever an experiment ranges over (adversaries and coins). Each is
a named shape over its parameters; an assumption is an instance a guarantee takes as a hypothesis.

* **Capacity** (`Capacity`): a part's ledger over one tick is at most `c`, entry by entry, in every actual circuit. Over
  a region of `n` ticks it is then at most `n • c` (`Accounting.Ledger.ledger_le_card_smul`). These are the numbers a
  datasheet or a benchmark can speak to. A part's crossing capacity is a cut bound: which links join which parts never
  enters.
* **Completeness** (`Complete`): a part's boundary crossings are every way it affects the rest. What the rest sees is a
  function of the rest's own inputs and the content that crossed, and that content is carried by the bits the ledger
  counts as crossing. With a capacity it bounds the content of a crossing with no protocol at all
  (`Accounting.Ledger.contentAtMost_of_capacity`).
* **Identity** (`Identity`): the verifier's transcript was produced by the declared part: it is a function of what the
  part held when the window opened and what crossed into it.
* **Latency** (`Latency`): a chain of `k` dependent operations completed within a window of `n` ticks has `k·τ ≤ n`.

The annotation's counts carry neither content nor dependencies, so completeness, identity and latency take them as
parameters (`crossed`, `inside`, `chain`): they read the actual circuit beyond its ledger.
-/

namespace Accounting.Ledger

variable {Λ G X : Type} [Fintype Λ] [DecidableEq Λ]

/-- **Capacity** `c` of part `p`: in every actual circuit `A x`, the part's ledger over any one tick is at most `c`,
entry by entry. -/
def Capacity (A : X → Annotation Λ G) (p : Finset Λ) (c : Ledger G) : Prop :=
  ∀ x t, (A x).ledger ⟨p, {t}⟩ ≤ c

/-- `f` is a function of `g`: two points that agree on `g` agree on `f`. -/
def Determines {V W : Type} (g : X → W) (f : X → V) : Prop := ∀ x y, g x = g y → f x = f y

/-- **Completeness** of region `ρ`'s boundary, outward: what the rest of the actual circuit sees (`view`) is a function
of the rest's own inputs (`rest`) and the content that crossed `ρ`'s boundary outward (`crossed`), and that content is
no more bits than the ledger counts as crossing outward. -/
def Complete {V R : Type} (A : X → Annotation Λ G) (ρ : Region Λ) (view : X → V) (rest : X → R)
    (crossed : X → List Bool) : Prop :=
  Determines (fun x => (rest x, crossed x)) view ∧ ∀ x, (crossed x).length ≤ (A x).ledger ρ .crossOut

/-- **Identity** of the declared part: the verifier's transcript (`transcript`) is a function of what the part held
when the region opened and the content that crossed into it (`inside`), so the part answered, not another. -/
def Identity {T I : Type} (transcript : X → T) (inside : X → I) : Prop := Determines inside transcript

/-- **Latency** `τ`: a chain of dependent operations the actual circuit completes within a window (`chain x w` of them)
takes at least `τ` ticks each. -/
def Latency (chain : X → Window → ℕ) (τ : ℕ) : Prop := ∀ x w, τ * chain x w ≤ w.len

end Accounting.Ledger
