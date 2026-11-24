module

public import Proofs.Slowdown.Secure

/-!
# The three sets meet two by two

`secure` says no computation is feasible, accounted for and prohibited. Here its hypotheses hold together and each
pair of its sets it can meet does, at one frame and one recipe, so the theorem is not true for want of a computation:

* `served_mem`: a computation in `H ∩ A`, at amounts that are the whole of the server's memory and of its links;
* `trained_mem`: a computation in `H ∩ B`, holding a reference training algorithm of one counted multiplication;
* `params`: the conditions `secure` puts on the amounts and the recipe, at the count `Y = 0`.

The frame is one server of one leaf, carrying one bit and joined to the world outside by one link of one bit a tick,
over one tick and one window, which the audits cover. The run is four one-bit gates: `0`, an input the server holds;
`1`, an input outside; `2`, computed outside from `1`; and `3`, computed on the server from `0` and `2`. So the server
carries one bit into the window and reads one bit from outside, which is all its memory and its link allow.

The two computations are that one run, and differ only in the declared circuit's gates. In `served` every gate is the
declared circuit's: the audits account for the bit carried and the bit read. In `trained` none is, and gate `3` is the
one counted multiplication of a reference training algorithm, of the weight `0` by the activation `2`, which reads
two fresh bits for the one weight it uses (`b_w = 2`). `trained_not_accounted` is `secure` at it.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs.SecureWitness

open Verity ResourceAccounting ResourceAccounting.SecurityProofs.Actual Finset Slowdown.Security

/-- The run's circuit: the inputs `0` and `1`, gate `2` reading `1`, and gate `3` reading `0` and `2`. -/
abbrev circuit : Circuit where
  N := 4
  w _ := 1
  isInput g := decide (g.val < 2)
  reads g := if g.val = 2 then {1} else if g.val = 3 then {0, 2} else ∅
  reads_lt := by decide +kernel
  input_reads := by decide +kernel
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {3}

/-- The run: gates `0` and `3` on the server (`true`), gates `1` and `2` outside (`false`), all at tick 0. -/
abbrev run : Placed Bool where
  toCircuit := circuit
  x g := decide (g.val = 0 ∨ g.val = 3)
  t _ := 0
  cell g := g.val
  cell_inj := by decide +kernel

/-- Each gate uses one unit of its leaf's compute, and every wire runs on the one link. -/
abbrev usage : Usage run Unit := ⟨fun _ => 1, fun _ _ => ()⟩

/-- The frame: the server `{true}`, each leaf doing one unit of work a tick and carrying one bit, one link of one bit
a tick between the server and the world outside, and one window of one tick, covered. -/
abbrev frame : Frame Bool Unit where
  hw := Hardware.ofLeaves (fun _ => 1) (fun _ => 1) (fun _ => (false, true)) (fun _ => 1)
  n := 1
  server _ := {true}
  disjoint i j h := absurd (Subsingleton.elim i j) h
  unit _ := Set.mem_univ _
  t := 1
  cuts := ⟨1, fun j => j.val, fun _ _ h => h, rfl, rfl⟩
  missed := ∅

/-- The run with every gate the declared circuit's. -/
abbrev served : Computation Bool Unit := ⟨run, usage, ∅, fun _ h => absurd h (notMem_empty _), univ⟩

/-- The run with no gate the declared circuit's. -/
abbrev trained : Computation Bool Unit := ⟨run, usage, ∅, fun _ h => absurd h (notMem_empty _), ∅⟩

/-- The recipe: one weight reaches the target, a window reads two bits for each weight it uses and one for each
thread it enters, a thread uses a weight once, and nothing is shared with the declared circuit. -/
noncomputable abbrev recipe : Recipe := ⟨1, 2, 1, 1, 0, 0, 0⟩

theorem region_empty (X : Finset Bool) {I : Set ℕ} (h : 0 ∉ I) : run.region X I = ∅ := by
  ext g
  simp [mem_region run, h]

theorem region_eq (X : Finset Bool) {I : Set ℕ} (h : 0 ∈ I) :
    run.region X I = univ.filter fun g : Fin 4 => 2 ≤ g.val ∧ run.x g ∈ X := by
  ext g
  simp only [mem_region run, h, and_true, mem_filter, mem_univ, true_and]
  exact and_congr_left fun _ => by simp; omega

theorem feasible : Feasible frame.hw run usage := by
  refine (feasible_ofLeaves_iff run _ _ _ _ usage).2
    ⟨fun _ _ _ => le_rfl, fun p a b => ?_, fun X t => ?_, ⟨by decide, ?_⟩⟩
  · by_cases h : (0 : ℕ) ∈ Set.Ico a b
    · rw [region_eq _ h]
      have hc : ∀ p : Bool, run.work usage.π (univ.filter fun g : Fin 4 => 2 ≤ g.val ∧ run.x g ∈ ({p} : Finset Bool)) ≤ 1 := by
        decide
      have hab : 1 ≤ b - a := by
        have := h.1
        have := h.2
        omega
      exact (hc p).trans (by simpa using hab)
    · rw [region_empty _ h]
      simp [Placed.work]
  · by_cases h : (0 : ℕ) ∈ Set.Ici t
    · unfold Placed.state
      rw [region_eq _ h, Finset.filter_congr_decidable]
      revert X
      decide
    · unfold Placed.state
      rw [region_empty _ h]
      simp [Circuit.into, Circuit.width]
  · intro e y a b
    by_cases h : (0 : ℕ) ∈ Set.Ico a b
    · have key : ∀ g k : Fin 4, g ∈ run.reads k → run.x g ≠ run.x k → g ∈ ({2} : Finset (Fin 4)) := by decide
      have hsub : run.readOn usage.link e y (Set.Ico a b) ⊆ {2} := by
        intro g hg
        simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hg
        obtain ⟨k, hk, hx, -⟩ := hg
        exact key g k hk hx
      have hab : 1 ≤ b - a := by
        have := h.1
        have := h.2
        omega
      exact (width_mono run hsub).trans (by simpa [Circuit.width] using hab)
    · have he : run.readOn usage.link e y (Set.Ico a b) = ∅ := by
        ext g
        simp [Placed.readOn, h]
      rw [he]
      simp [Circuit.width]

theorem window_eq (i : Fin frame.n) (j : Fin frame.cuts.J) : run.region (frame.server i) (frame.span j) = {3} := by
  have hj : j = (0 : Fin 1) := Subsingleton.elim _ _
  subst hj
  change run.region {true} (Set.Ico 0 1) = {3}
  rw [region_eq _ (show (0 : ℕ) ∈ Set.Ico 0 1 from ⟨le_rfl, Nat.one_pos⟩)]
  decide

/-- **A computation that is feasible and accounted for**, at the amounts `1` and `1`: the declared circuit's gates
hold the one bit the server carries and the window reads the one bit its link delivers. -/
theorem served_mem : served ∈ H frame ∧ served ∈ A frame 1 1 0 := by
  refine ⟨feasible, fun i j => ?_⟩
  have hj : j = (0 : Fin 1) := Subsingleton.elim _ _
  subst hj
  refine ⟨fun _ => ⟨?_, ?_⟩, fun _ => Nat.zero_le _⟩
  · change 1 ≤ run.stateIn {true} 0 univ
    unfold Placed.stateIn
    rw [region_eq _ (Set.mem_Ici.2 (le_refl 0)), Finset.filter_congr_decidable]
    decide
  · change 1 ≤ run.width (((run.into (run.region (frame.server i) (frame.span 0))).filter
      fun g => run.x g ∉ ({true} : Finset Bool)) ∩ univ)
    rw [window_eq]
    decide

theorem unaccounted_eq : frame.unaccounted 1 1 recipe = 0 := by
  have hm : frame.m = 1 := by decide
  have hω : frame.ω = 1 := by decide
  have hb : frame.b = 1 := by
    simp [Frame.b, linkCapacity, boundary, Hardware.ofLeaves]
  simp [Frame.unaccounted, hm, hω, hb]

/-- **`secure`'s conditions on the amounts and the recipe hold** at the amounts `1`, `1` and `0` and the count
`Y = 0`: the audits account for all the server's memory and all its link carries, so a covered window can read no bit
of a reference training algorithm, and the fit bound is `0`. -/
theorem params : 0 ≤ recipe.b_w ∧ 0 < recipe.w_e ∧
    (frame.unaccounted 1 1 recipe : ℝ) < recipe.Nstar * recipe.b_w ∧ frame.bound 1 1 0 recipe ≤ 0 := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  · rw [unaccounted_eq]
    norm_num
  · unfold Frame.bound
    rw [unaccounted_eq]
    simp

/-- The reference training algorithm: the weight `0`, the activation `2` and their counted multiplication `3`. -/
abbrev ref : Circuit where
  N := 4
  w _ := 1
  isInput g := decide (g.val < 3)
  reads g := if g.val = 3 then {0, 2} else ∅
  reads_lt := by decide +kernel
  input_reads := by decide +kernel
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {3}

abbrev decl : RegisteredCircuit := .ofCircuit ref ∅ (by decide +kernel)

/-- One thread, of the activation `2`; the counted multiplication `3` uses the weight `0` once. -/
abbrev threading : Threading decl where
  m := 1
  W := ({0} : Finset (Fin 4))
  thread g := if g.val = 2 then some 0 else none
  thread_W := by decide +kernel
  M := ({3} : Finset (Fin 4))
  M_computed := by decide +kernel
  weight _ := (0 : Fin 4)
  weight_W := by decide +kernel
  weight_reads := by decide +kernel
  act _ := (2 : Fin 4)
  act_reads := by decide +kernel
  act_thread := by decide +kernel
  reuse := 1
  reuse_le := by decide +kernel

theorem developer_region : run.region (univ.biUnion frame.server) (Set.Iio frame.t) = {3} := by
  rw [region_eq _ (show (0 : ℕ) ∈ Set.Iio 1 from Nat.one_pos)]
  decide

/-- The reference training algorithm is what the server did, gate for gate. -/
noncomputable def emb : Containment decl.D (trained.circuit frame).D where
  emb g := g
  emb_inj _ _ h := h
  emb_w _ := rfl
  isInput_emb g := by
    change (run.cut (run.region (univ.biUnion frame.server) (Set.Iio frame.t))).isInput g = ref.isInput g
    rw [developer_region]
    have key : ∀ g : Fin 4, (if g ∈ ({3} : Finset (Fin 4)) then run.isInput g else true) = ref.isInput g := by
      decide
    exact key g
  reads_emb g hg := by
    change (run.cut (run.region (univ.biUnion frame.server) (Set.Iio frame.t))).reads g = (ref.reads g).map _
    rw [developer_region]
    have key : ∀ g : Fin 4, ref.isInput g = false →
        (if g ∈ ({3} : Finset (Fin 4)) then run.reads g else ∅) =
          (ref.reads g).map ⟨fun g => g, fun _ _ h => h⟩ := by decide
    exact key g hg
  op_emb g hg v := by
    change BitVec.cast _ ((run.cut (run.region (univ.biUnion frame.server) (Set.Iio frame.t))).op g v) = 0
    simp only [Placed.cut]
    split_ifs <;> rfl
  outputs_emb o ho := by
    change o ∈ run.region (univ.biUnion frame.server) (Set.Iio frame.t)
    rw [developer_region]
    exact ho

/-- The one window starts the one thread. -/
abbrev start : Fin threading.m → Fin (frame.n * frame.cuts.J) := fun _ => ⟨0, by decide⟩

theorem gates_eq {Q : Partition decl.D (frame.n * frame.cuts.J)} (hQ : Divides frame trained emb Q)
    (j : Fin (frame.n * frame.cuts.J)) : Q.gates j = {3} := by
  have hj : j = finProdFinEquiv ((0 : Fin 1), (0 : Fin 1)) := Subsingleton.elim (α := Fin 1) _ _
  ext g
  rw [hj, hQ 0 0 g]
  change g ∈ run.region (frame.server 0) (frame.span 0) ↔ _
  rw [window_eq]

theorem trains : Trains frame trained recipe decl emb threading start where
  reuse := le_rfl
  load _ _ := le_rfl
  reach Q hQ u _ _ := by
    have hu : ∀ u, 1 ≤ (threading.threadWeights univ u).card := by decide +kernel
    have hf : (univ.filter fun g => ∀ j ∈ frame.missed, g ∉ Q.gates j) = univ := by simp
    rw [hf]
    exact hu u
  fits Q hQ j _ := by
    have hw : (threading.weightsUsed ({3} : Finset (Fin 4))).card = 1 := by decide +kernel
    have hf : freshInput decl ({3} : Finset (Fin 4)) = 2 := by decide +kernel
    have he : threading.entered Q start j = ∅ := by
      ext t
      simp only [Threading.entered, mem_filter, mem_univ, true_and, notMem_empty, iff_false, not_and, not_not]
      exact fun _ => Subsingleton.elim (α := Fin 1) _ _
    rw [gates_eq hQ, hw, hf, he]
    norm_num
  shared_carried i j _ := by simp [Computation.carries, Placed.stateIn, Circuit.width]
  shared_read i j _ := by simp [Computation.reads, Circuit.width]
  shared_work i j h := absurd h (notMem_empty _)

/-- **A computation that is feasible and prohibited**, at the count `0`: the run holds a reference training algorithm
of one counted multiplication that meets the recipe's premises. -/
theorem trained_mem : trained ∈ H frame ∧ trained ∈ B frame recipe 0 :=
  ⟨feasible, decl, emb, threading, start, trains, by
    have h : threading.M.card = 1 := by decide +kernel
    rw [h]
    norm_num⟩

/-- `secure` at the witness: the run with no gate the declared circuit's is not accounted for. -/
theorem trained_not_accounted : trained ∉ A frame 1 1 0 := fun hA =>
  secure frame 1 1 0 recipe 0 params.1 params.2.1 params.2.2.1 params.2.2.2 trained trained_mem.1 hA trained_mem.2

/-- `secure` at the witness: the run with every gate the declared circuit's is not prohibited. -/
theorem served_not_prohibited : served ∉ B frame recipe 0 :=
  secure frame 1 1 0 recipe 0 params.1 params.2.1 params.2.2.1 params.2.2.2 served served_mem.1 served_mem.2

end Slowdown.SecurityProofs.SecureWitness
