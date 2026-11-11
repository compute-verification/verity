module

public import Proofs.Properties.Nci.Record

/-!
# What a served record's circuit is tied to, and the tie it lacks

A scratch file, never merged.

* **`Served` reads the layout alone.** A circuit a record lays out is laid out again under any registration
  (`served_registration_free`: a one-entry log whose builder is the constant compiler of the circuit's own output,
  with no advice) and with every operation and output replaced (`served_ops_free`). So the circuits `Complies`
  speaks of are not tied to a deployment's log, builder or advice.
* **The tie it lacks** (`DeclaresBy`, `Declares`): the record lays out the declaration's circuit, under some division
  into blocks, with every row the check holds among the declaration's registered inputs. With it, an accepted record
  gives both bounds on each of those blocks at the declaration's own registered inputs (`meets_of_declaresBy`), so
  the rate rule on the declaration itself (`compliesRate_of_declares`), whatever the declaration's own blocks are: on
  the declaration over a circuit with each computed gate its own block (`compliesRate_ofCircuit`).
* **One size for every block** (`counts_one_size`): where the summary counts every block, any two blocks of a served
  circuit that hold a gate hold the same number of gates.
* **A collision is a collision** (`collision_of_collides`): `Collides` gives two different byte strings with one
  SHA-512, whatever the descriptors and the key.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)

/-! ### What `Served` does not read -/

/-- **The registration is free.** -/
theorem served_registration_free {F : Form} {c : F.Case} {d : Verity.RegisteredCircuit} (h : Served F c d) :
    Served F c (Verity.RegisteredCircuit.ofPartition d.D d.θ d.θ_input d.n d.𝓘) := by
  obtain ⟨P, doc, W, S, rs, at_, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h
  exact ⟨P, doc, W, S, rs, at_, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩

/-- A circuit with every operation the constant zero and no output gate. -/
abbrev zeroOps (C : Verity.Circuit) : Verity.Circuit :=
  { C with op := fun _ _ => 0, op_congr := fun _ _ _ _ => rfl, outputs := ∅ }

/-- **The operations and the outputs are free.** -/
theorem served_ops_free {F : Form} {c : F.Case} {d : Verity.RegisteredCircuit} (h : Served F c d) :
    Served F c (Verity.RegisteredCircuit.ofPartition (zeroOps d.D) d.θ d.θ_input d.n
      ⟨d.𝓘.unit, d.𝓘.input_iff⟩) := by
  obtain ⟨P, doc, W, S, rs, at_, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h
  exact ⟨P, doc, W, S, rs, at_, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩

/-! ### The tie to a declaration -/

/-- **The record `c` lays out the declaration `d₀` in the blocks `I`**: `d₀`'s circuit, divided by `I`, is the
record's, and every row the check holds is one of `d₀`'s registered inputs. -/
def DeclaresBy (F : Form) (c : F.Case) (d₀ : Verity.RegisteredCircuit) {n : ℕ} (I : Verity.Partition d₀.D n) : Prop :=
  ∃ (θ' : Finset (Fin d₀.D.N)) (h : ∀ g ∈ θ', d₀.D.isInput g = true),
    θ' ⊆ d₀.θ ∧ Served F c (Verity.RegisteredCircuit.ofPartition d₀.D θ' h n I)

/-- **The record `c` lays out the declaration `d₀`**, in some division of its circuit into blocks. -/
def Declares (F : Form) (c : F.Case) (d₀ : Verity.RegisteredCircuit) : Prop :=
  ∃ (n : ℕ) (I : Verity.Partition d₀.D n), DeclaresBy F c d₀ I

/-- Fewer registered inputs, more fresh ones. -/
theorem novelReads_anti (d₀ : Verity.RegisteredCircuit) {n : ℕ} (I : Verity.Partition d₀.D n)
    (θ' : Finset (Fin d₀.D.N)) (h : ∀ g ∈ θ', d₀.D.isInput g = true) (hsub : θ' ⊆ d₀.θ)
    (R : Finset (Fin d₀.D.N)) :
    Nci.novelReads d₀ R ≤ Nci.novelReads (Verity.RegisteredCircuit.ofPartition d₀.D θ' h n I) R := by
  show d₀.D.width (d₀.D.into R \ d₀.θ) ≤ d₀.D.width (d₀.D.into R \ θ')
  unfold Verity.Circuit.width
  exact Finset.sum_le_sum_of_subset (Finset.sdiff_subset_sdiff (Finset.Subset.refl _) hsub)

/-- **An accepted record gives both bounds on every block of a declaration it lays out**, at the declaration's own
registered inputs. -/
theorem meets_of_declaresBy {F : Form} (hc : Complies F) {j tr : Json} {c : F.Case} (hacc : Accepts F j tr c)
    {d₀ : Verity.RegisteredCircuit} {n : ℕ} {I : Verity.Partition d₀.D n} (hd : DeclaresBy F c d₀ I) (u : Fin n) :
    Meets d₀ I (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) u := by
  obtain ⟨θ', h, hsub, hs⟩ := hd
  obtain ⟨-, -, hall⟩ := hc j tr c hacc
  obtain ⟨hmeets, -⟩ := hall _ hs
  have hm : Meets (Verity.RegisteredCircuit.ofPartition d₀.D θ' h n I) I (F.inst c).gamma
      (((F.inst c).num : ℝ) / (F.inst c).den) u := hmeets u
  obtain ⟨hsize, hnov⟩ := hm
  refine ⟨hsize, le_trans ?_ hnov⟩
  exact_mod_cast novelReads_anti d₀ I θ' h hsub (I.gates u)

/-- **An accepted record gives the rate rule on a declaration it lays out.** -/
theorem compliesRate_of_declares {F : Form} (hc : Complies F) {j tr : Json} {c : F.Case} (hacc : Accepts F j tr c)
    {d₀ : Verity.RegisteredCircuit} (hd : Declares F c d₀) :
    Nci.CompliesRate (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) d₀ := by
  obtain ⟨n, I, hI⟩ := hd
  unfold Nci.CompliesRate
  exact ⟨n, I, fun u => meets_of_declaresBy hc hacc hI u⟩

/-- The same on the declaration over a circuit `C` with registered inputs `θ`, each computed gate its own block. -/
theorem compliesRate_ofCircuit {F : Form} (hc : Complies F) {j tr : Json} {c : F.Case} (hacc : Accepts F j tr c)
    (C : Verity.Circuit) (θ : Finset (Fin C.N)) (hθ : ∀ g ∈ θ, C.isInput g = true) {n : ℕ}
    (I : Verity.Partition C n) (θ' : Finset (Fin C.N)) (h : ∀ g ∈ θ', C.isInput g = true) (hsub : θ' ⊆ θ)
    (hs : Served F c (Verity.RegisteredCircuit.ofPartition C θ' h n I)) :
    Nci.CompliesRate (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den)
      (Verity.RegisteredCircuit.ofCircuit C θ hθ) :=
  compliesRate_of_declares hc hacc (d₀ := Verity.RegisteredCircuit.ofCircuit C θ hθ) ⟨n, I, θ', h, hsub, hs⟩

/-- **`Complies` at the registration's own**: every circuit a record lays out is one a compiler of no advice and a log
of one entry registers, so `Complies` holds of it whatever the deployment registered. -/
theorem complies_reads_no_registration {F : Form} (hc : Complies F) {j tr : Json} {c : F.Case}
    (hacc : Accepts F j tr c) : ∃ d, Served F c d ∧ d.a = [] := by
  obtain ⟨-, ⟨d, hd⟩, -⟩ := hc j tr c hacc
  exact ⟨_, served_registration_free hd, rfl⟩

/-! ### The summary's one size -/

/-- **Every block that holds a gate has one size**, where the summary counts every block. -/
theorem counts_one_size {F : Form} (hc : Counts F) {j tr : Json} {c : F.Case} (hr : F.read j tr = .ok c)
    {d : Verity.RegisteredCircuit} (hd : Served F c d) (u v : Fin d.n) (hu : d.𝓘.gates u ≠ ∅)
    (hv : d.𝓘.gates v ≠ ∅) : d.D.size (d.𝓘.gates u) = d.D.size (d.𝓘.gates v) := by
  obtain ⟨-, h1 | h1⟩ := hc j tr c hr d hd u
  · obtain ⟨-, h2 | h2⟩ := hc j tr c hr d hd v
    · exact h1.trans h2.symm
    · exact absurd h2 hv
  · exact absurd h1 hu

/-! ### The collision -/

/-- **`Collides` is a SHA-512 collision.** -/
theorem collision_of_collides {j j' : Json} {key : String} (h : Collides j j' key) :
    ∃ x x' : ByteArray, x ≠ x' ∧ Flock.Sha512.hash x = Flock.Sha512.hash x' := by
  obtain ⟨x, x', -, -, hne, heq⟩ := h
  exact ⟨x, x', hne, heq⟩

/-- The key reaches itself in any descriptor, a Definition of it or not: the strings a closure digest hashes are
never none. -/
theorem reaches_self (j : Json) (key : String) : Reaches j key key := Reaches.self

end NciCheck.Properties.Record.Attack
