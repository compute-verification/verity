module

public import Proofs.Slowdown.Secure

/-!
# The three sets meet two by two

`secure` says no computation is feasible, accounted for and prohibited. Here its hypotheses hold together and each
pair of its sets it can meet does, at one recipe and at a frame whose one window is covered or missed, so the theorem
is not true for want of a computation, on either of its branches:

* `served_mem`: a computation in `H ∩ A`, at amounts that are the whole of the server's memory, of its link and of
  its work;
* `trained_mem`: a computation in `H ∩ B`, holding a reference training algorithm of one counted multiplication;
* `params`: the conditions `secure` puts on the amounts and the recipe, at the count `Y = 0`.

The frame is one server of one leaf, doing one unit of work a tick, carrying one bit and joined to the world outside
by one link of one bit a tick, over one tick and one window. The run is four one-bit gates: `0`, an input the server
holds; `1`, an input outside; `2`, computed outside from `1`; and `3`, computed on the server from `0` and `2`. So the
server carries one bit into the window, reads one bit from outside and does one unit of work, which is all its memory,
its link and its compute allow.

The hardware's three limits are parameters (`hw c m β`), 1 each above.

The two computations are that one run, and differ only in the declared circuit's gates. In `served` every gate is the
declared circuit's: the audits account for the bit carried and the bit read. In `trained` none is, and gate `3` is the
one counted multiplication of a reference training algorithm, of the weight `0` by the activation `2`, which reads
two fresh bits for the one weight it uses (`b_w = 2`). `trained_not_accounted` is `secure` at it.

**Each clause of the feasible set is needed.** At limits per leaf the feasible set is four clauses
(`feasible_iff`). With one limit set to 0 the run meets the other three clauses, and `trained` is accounted for and
prohibited at the fit bound (`Escapes`): `memory_needed` (the server carries nothing), `bandwidth_needed` (the link
carries nothing) and `work_needed` (the server does no work, the window missed). For the order clause the run is
another, `late`: the counted multiplication at tick 0 reads its weight and its activation from gates the server
computes at tick 5, so the server carries nothing into the window and reads nothing from outside, and `late` meets
the work, memory and bandwidth clauses on a server that carries nothing over a link that carries nothing
(`order_needed`). So `secure` is false with any one clause dropped from `H`.
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

/-- The hardware: each leaf does `c` units of work a tick and carries `m` bits, and one link of `β` bits a tick joins
the server and the world outside. -/
abbrev hw (c m β : ℕ) : Hardware Bool Unit :=
  Hardware.ofLeaves (fun _ => c) (fun _ => m) (fun _ => (false, true)) (fun _ => β)

/-- One window of one tick. -/
abbrev cuts : Cuts 1 := ⟨1, fun j => j.val, fun _ _ h => h, rfl, rfl⟩

variable (c m β : ℕ) (I : Finset (Fin (1 * cuts.J)))

/-- The frame: the server `{true}` over one window, missed when it is in `I`. -/
abbrev frame : Frame Bool Unit := Frame.single (hw c m β) {true} (Set.mem_univ _) 1 cuts I

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

/-- The order clause: a gate reads only gates at its tick or before. -/
def Order (M : Computation Bool Unit) : Prop := M.run.Causal fun _ _ => 0

/-- The work clause: a leaf does at most `c` units of work a tick. -/
def Work (M : Computation Bool Unit) : Prop :=
  ∀ p a b, M.run.work M.usage.π (M.run.region {p} (Set.Ico a b)) ≤ c * (b - a)

/-- The memory clause: a set of leaves carries at most `m` bits a leaf across a tick. -/
def Memory (M : Computation Bool Unit) : Prop := ∀ X t, M.run.state X t ≤ ∑ _p ∈ X, m

/-- The bandwidth clause: the link carries at most `β` bits a tick. -/
def Bandwidth (M : Computation Bool Unit) : Prop :=
  M.run.Links (fun _ : Unit => (false, true)) (fun _ => β) M.usage.link

/-- The feasible set at the hardware `hw c m β` is the four clauses. -/
theorem feasible_iff (M : Computation Bool Unit) :
    Feasible (hw c m β) M.run M.usage ↔ Order M ∧ Work c M ∧ Memory m M ∧ Bandwidth β M :=
  feasible_ofLeaves_iff M.run _ _ _ _ M.usage

theorem order : Order trained := fun _ _ _ => le_rfl

theorem work (hc : 1 ≤ c) : Work c trained := by
  intro p a b
  change run.work usage.π (run.region {p} (Set.Ico a b)) ≤ c * (b - a)
  by_cases h : (0 : ℕ) ∈ Set.Ico a b
  · rw [region_eq _ h]
    have h1 : ∀ p : Bool, run.work usage.π (univ.filter fun g : Fin 4 => 2 ≤ g.val ∧ run.x g ∈ ({p} : Finset Bool)) ≤ 1 := by
      decide
    have hab : 1 ≤ b - a := by
      have := h.1
      have := h.2
      omega
    exact (h1 p).trans (Nat.mul_pos hc hab)
  · rw [region_empty _ h]
    simp [Placed.work]

theorem memory (hm : 1 ≤ m) : Memory m trained := by
  intro X t
  change run.state X t ≤ ∑ _p ∈ X, m
  have h1 : run.state X t ≤ ∑ _p ∈ X, 1 := by
    by_cases h : (0 : ℕ) ∈ Set.Ici t
    · unfold Placed.state
      rw [region_eq _ h, Finset.filter_congr_decidable]
      revert X
      decide
    · unfold Placed.state
      rw [region_empty _ h]
      simp [Circuit.into, Circuit.width]
  exact h1.trans (sum_le_sum fun _ _ => hm)

theorem bandwidth (hβ : 1 ≤ β) : Bandwidth β trained := by
  change run.Links (fun _ : Unit => (false, true)) (fun _ => β) usage.link
  refine ⟨by decide, fun e y a b => ?_⟩
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
    exact (width_mono run hsub).trans ((show run.width {2} ≤ 1 by decide).trans (Nat.mul_pos hβ hab))
  · have he : run.readOn usage.link e y (Set.Ico a b) = ∅ := by
      ext g
      simp [Placed.readOn, h]
    rw [he]
    simp [Circuit.width]

theorem feasible : Feasible (hw 1 1 1) run usage :=
  (feasible_iff 1 1 1 trained).2 ⟨order, work 1 le_rfl, memory 1 le_rfl, bandwidth 1 le_rfl⟩

theorem window_eq (i : Fin (frame c m β I).n) (j : Fin (frame c m β I).cuts.J) :
    run.region ((frame c m β I).server i) ((frame c m β I).span j) = {3} := by
  have hj : j = (0 : Fin 1) := Subsingleton.elim _ _
  subst hj
  change run.region {true} (Set.Ico 0 1) = {3}
  rw [region_eq _ (show (0 : ℕ) ∈ Set.Ico 0 1 from ⟨le_rfl, Nat.one_pos⟩)]
  decide

theorem ingress_le (i : Fin (frame c m β I).n) (j : Fin (frame c m β I).cuts.J) :
    run.ingress ((frame c m β I).server i) ((frame c m β I).span j) ≤ 1 := by
  unfold Placed.ingress
  rw [window_eq, Finset.filter_congr_decidable]
  change run.width ((run.into {3}).filter fun g => run.x g ∉ ({true} : Finset Bool)) ≤ 1
  decide

/-- **A computation that is feasible and accounted for**, at the amounts `1`, `1`, `1` and `1`: the declared
circuit's gates hold the one bit the server carries, the window reads the one bit its link delivers and no other, and
they do the one unit of work the server does. -/
theorem served_mem : served ∈ H (frame 1 1 1 I) ∧ served ∈ A (frame 1 1 1 I) 1 1 1 1 := by
  refine ⟨feasible, fun i j => ?_⟩
  have hj : j = (0 : Fin 1) := Subsingleton.elim _ _
  subst hj
  refine ⟨fun _ => ⟨?_, ?_, ingress_le 1 1 1 I i 0⟩, fun _ => ?_⟩
  · change 1 ≤ run.stateIn {true} 0 univ
    unfold Placed.stateIn
    rw [region_eq _ (Set.mem_Ici.2 (le_refl 0)), Finset.filter_congr_decidable]
    decide
  · change 1 ≤ run.width (((run.into (run.region ((frame 1 1 1 I).server i) ((frame 1 1 1 I).span 0))).filter
      fun g => run.x g ∉ ({true} : Finset Bool)) ∩ univ)
    rw [window_eq]
    decide
  · change 1 ≤ run.work usage.π (run.region ((frame 1 1 1 I).server i) ((frame 1 1 1 I).span 0) ∩ univ)
    rw [window_eq]
    decide

theorem ω_eq : (frame c m β I).ω = 1 := by
  change (frame 0 0 0 ∅).ω = 1
  decide

theorem m_eq : (frame c m β I).m = m :=
  (single_m _ _ _ 1 cuts I).trans (by simp [Hardware.ofLeaves])

theorem b_eq : (frame c m β I).b = β :=
  (single_b _ _ _ 1 cuts I).trans (by simp [linkCapacity, boundary, Hardware.ofLeaves])

theorem Φ_eq : (frame c m β I).Φ = c :=
  (single_Φ _ _ _ 1 cuts I).trans (by simp [Hardware.ofLeaves])

theorem unaccounted_eq : (frame 1 1 1 I).unaccounted 1 1 1 recipe = 0 := by
  simp [Frame.unaccounted, m_eq, ω_eq, b_eq]

/-- **`secure`'s conditions on the amounts and the recipe hold** at the amounts `1`, `1`, `1` and `1` and the count
`Y = 0`: the audits account for all the server's memory, all its link carries and all its work, so a covered window
can read no bit of a reference training algorithm, a missed one has no work left for it, and the fit bound is `0`. -/
theorem params : 0 ≤ recipe.b_w ∧ 0 < recipe.w_e ∧
    ((frame 1 1 1 I).unaccounted 1 1 1 recipe : ℝ) < recipe.Nstar * recipe.b_w ∧
      (frame 1 1 1 I).bound 1 1 1 1 recipe ≤ 0 := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  · rw [unaccounted_eq]
    norm_num
  · unfold Frame.bound
    rw [unaccounted_eq, Φ_eq, ω_eq]
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

theorem developer_region : run.region (univ.biUnion (frame c m β I).server) (Set.Iio (frame c m β I).t) = {3} := by
  change run.region (univ.biUnion fun _ : Fin 1 => ({true} : Finset Bool)) (Set.Iio 1) = {3}
  rw [region_eq _ (show (0 : ℕ) ∈ Set.Iio 1 from Nat.one_pos)]
  decide

/-- The reference training algorithm is what the server did, gate for gate. -/
noncomputable def emb : Containment decl.D (trained.circuit (frame c m β I)).D where
  emb g := g
  emb_inj _ _ h := h
  emb_w _ := rfl
  isInput_emb g := by
    change (run.cut (run.region (univ.biUnion (frame c m β I).server) (Set.Iio (frame c m β I).t))).isInput g = ref.isInput g
    rw [developer_region]
    have key : ∀ g : Fin 4, (if g ∈ ({3} : Finset (Fin 4)) then run.isInput g else true) = ref.isInput g := by
      decide
    exact key g
  reads_emb g hg := by
    change (run.cut (run.region (univ.biUnion (frame c m β I).server) (Set.Iio (frame c m β I).t))).reads g = (ref.reads g).map _
    rw [developer_region]
    have key : ∀ g : Fin 4, ref.isInput g = false →
        (if g ∈ ({3} : Finset (Fin 4)) then run.reads g else ∅) =
          (ref.reads g).map ⟨fun g => g, fun _ _ h => h⟩ := by decide
    exact key g hg
  op_emb g hg v := by
    change BitVec.cast _ ((run.cut (run.region (univ.biUnion (frame c m β I).server) (Set.Iio (frame c m β I).t))).op g v) = 0
    simp only [Placed.cut]
    split_ifs <;> rfl
  outputs_emb o ho := by
    change o ∈ run.region (univ.biUnion (frame c m β I).server) (Set.Iio (frame c m β I).t)
    rw [developer_region]
    exact ho

/-- The one window starts the one thread. -/
abbrev start : Fin threading.m → Fin ((frame c m β I).n * (frame c m β I).cuts.J) := fun _ => ⟨0, Nat.one_pos⟩

theorem gates_eq {Q : Partition decl.D ((frame c m β I).n * (frame c m β I).cuts.J)} (hQ : Divides (frame c m β I) trained (emb c m β I) Q)
    (j : Fin ((frame c m β I).n * (frame c m β I).cuts.J)) : Q.gates j = {3} := by
  have hj : j = finProdFinEquiv ((0 : Fin 1), (0 : Fin 1)) := Subsingleton.elim (α := Fin 1) _ _
  ext g
  rw [hj, hQ 0 0 g]
  change g ∈ run.region ((frame c m β I).server 0) ((frame c m β I).span 0) ↔ _
  rw [window_eq]

/-- The reference training algorithm trains inside a computation with no gate the declared circuit's whose one window
holds the algorithm's counted multiplication, each such gate using a unit of compute. -/
theorem trains_of (M : Computation Bool Unit) (hd : M.declared = ∅)
    (e : Containment decl.D (M.circuit (frame c m β I)).D) (hload : ∀ g ∈ threading.M, 1 ≤ M.usage.π (e.emb g))
    (hgates : ∀ Q : Partition decl.D ((frame c m β I).n * (frame c m β I).cuts.J),
      Divides (frame c m β I) M e Q → ∀ j, Q.gates j = {3}) :
    Trains (frame c m β I) M recipe decl e threading (start c m β I) where
  reuse := le_rfl
  load := hload
  reach Q hQ u hu _ := by
    have hcard : ∀ u, 1 ≤ (threading.threadWeights univ u).card := by decide +kernel
    have hf : (univ.filter fun g => ∀ j ∈ (frame c m β I).missed, g ∉ Q.gates j) = univ :=
      filter_true_of_mem fun g _ j hj => absurd (Subsingleton.elim (α := Fin 1) j (start c m β I u) ▸ hj) hu
    rw [hf]
    exact hcard u
  fits Q hQ j _ := by
    have hw : (threading.weightsUsed ({3} : Finset (Fin 4))).card = 1 := by decide +kernel
    have hf : freshInput decl ({3} : Finset (Fin 4)) = 2 := by decide +kernel
    have he : threading.entered Q (start c m β I) j = ∅ := by
      ext t
      simp only [Threading.entered, mem_filter, mem_univ, true_and, notMem_empty, iff_false, not_and, not_not]
      exact fun _ => Subsingleton.elim (α := Fin 1) _ _
    rw [hgates Q hQ, hw, hf, he]
    norm_num
  shared_carried i j _ := by simp [Computation.carries, Placed.stateIn, Circuit.width, hd]
  shared_read i j _ := by simp [Computation.reads, Circuit.width, hd]
  shared_work i j _ := by
    classical
    refine Nat.le_of_eq (card_eq_zero.2 (eq_empty_of_forall_notMem fun g hg => ?_))
    rw [mem_filter] at hg
    exact notMem_empty _ (hd ▸ (mem_inter.1 hg.2).2)

theorem trains : Trains (frame c m β I) trained recipe decl (emb c m β I) threading (start c m β I) :=
  trains_of c m β I trained rfl (emb c m β I) (fun _ _ => le_rfl) fun _ hQ j => gates_eq c m β I hQ j

theorem prohibited_of (M : Computation Bool Unit) (e : Containment decl.D (M.circuit (frame c m β I)).D)
    (hT : Trains (frame c m β I) M recipe decl e threading (start c m β I)) {Y : ℝ} (hY : Y < 1) :
    M ∈ B (frame c m β I) recipe Y :=
  ⟨decl, e, threading, start c m β I, hT, by
    have h : threading.M.card = 1 := by decide +kernel
    rw [h]
    exact_mod_cast hY⟩

/-- The run with no gate the declared circuit's holds a reference training algorithm of one counted multiplication
that meets the recipe's premises, whatever the hardware's limits. -/
theorem trained_prohibited {Y : ℝ} (hY : Y < 1) : trained ∈ B (frame c m β I) recipe Y :=
  prohibited_of c m β I trained (emb c m β I) (trains c m β I) hY

/-- **A computation that is feasible and prohibited**, at the count `0`. -/
theorem trained_mem : trained ∈ H (frame 1 1 1 I) ∧ trained ∈ B (frame 1 1 1 I) recipe 0 :=
  ⟨feasible, trained_prohibited 1 1 1 I (by norm_num)⟩

/-- `secure` at the witness: the run with no gate the declared circuit's is not accounted for. -/
theorem trained_not_accounted : trained ∉ A (frame 1 1 1 I) 1 1 1 1 := fun hA =>
  secure (frame 1 1 1 I) 1 1 1 1 recipe 0 (params I).1 (params I).2.1 (params I).2.2.1 (params I).2.2.2 trained
    (trained_mem I).1 hA (trained_mem I).2

/-- `secure` at the witness: the run with every gate the declared circuit's is not prohibited. -/
theorem served_not_prohibited : served ∉ B (frame 1 1 1 I) recipe 0 :=
  secure (frame 1 1 1 I) 1 1 1 1 recipe 0 (params I).1 (params I).2.1 (params I).2.2.1 (params I).2.2.2 served
    (served_mem I).1 (served_mem I).2

/-- **`secure`'s conclusion fails of `M` at the frame `F`**: its conditions on the amounts and the count hold, and
`M` is accounted for and prohibited. By `secure`, `M` is then not feasible on `F`'s hardware. -/
def Escapes (M : Computation Bool Unit) (F : Frame Bool Unit) (x_s x_b x_w y_b : ℕ) (Y : ℝ) : Prop :=
  (F.unaccounted x_s x_b y_b recipe : ℝ) < recipe.Nstar * recipe.b_w ∧ F.bound x_s x_b x_w y_b recipe ≤ Y ∧
    M ∈ A F x_s x_b x_w y_b ∧ M ∈ B F recipe Y

theorem accounted_covered : trained ∈ A (frame c m β ∅) 0 0 0 1 := fun i j =>
  ⟨fun _ => ⟨Nat.zero_le _, Nat.zero_le _, ingress_le c m β ∅ i j⟩, fun h => absurd h (notMem_empty _)⟩

theorem accounted_missed : trained ∈ A (frame c m β univ) 1 1 0 1 := fun _ _ =>
  ⟨fun h => absurd (mem_univ _) h, fun _ => Nat.zero_le _⟩

theorem escapes_covered (h : m + min β 1 = 1) : Escapes trained (frame c m β ∅) 0 0 0 1 (1 / 4) := by
  have hu : (frame c m β ∅).unaccounted 0 0 1 recipe = 1 := by
    simpa [Frame.unaccounted, m_eq, b_eq, ω_eq] using h
  refine ⟨?_, ?_, accounted_covered c m β, trained_prohibited c m β ∅ (by norm_num)⟩
  · rw [hu]
    norm_num
  · unfold Frame.bound
    rw [hu]
    norm_num

/-- **The memory clause is needed.** On a server that carries nothing, the run meets the order, work and bandwidth
clauses, and with no gate the declared circuit's it is accounted for and prohibited at the fit bound. -/
theorem memory_needed :
    Order trained ∧ Work 1 trained ∧ Bandwidth 1 trained ∧ Escapes trained (frame 1 0 1 ∅) 0 0 0 1 (1 / 4) :=
  ⟨order, work 1 le_rfl, bandwidth 1 le_rfl, escapes_covered 1 0 1 rfl⟩

/-- **The bandwidth clause is needed.** On a link that carries nothing, the run meets the order, work and memory
clauses, and with no gate the declared circuit's it is accounted for and prohibited at the fit bound. -/
theorem bandwidth_needed :
    Order trained ∧ Work 1 trained ∧ Memory 1 trained ∧ Escapes trained (frame 1 1 0 ∅) 0 0 0 1 (1 / 4) :=
  ⟨order, work 1 le_rfl, memory 1 le_rfl, escapes_covered 1 1 0 rfl⟩

/-- **The work clause is needed.** On a server that does no work, with the window missed, the run meets the order,
memory and bandwidth clauses, and with no gate the declared circuit's it is accounted for and prohibited at the fit
bound. -/
theorem work_needed :
    Order trained ∧ Memory 1 trained ∧ Bandwidth 1 trained ∧ Escapes trained (frame 0 1 1 univ) 1 1 0 1 0 := by
  have hu : (frame 0 1 1 univ).unaccounted 1 1 1 recipe = 0 := by
    simp [Frame.unaccounted, m_eq, b_eq, ω_eq]
  refine ⟨order, memory 1 le_rfl, bandwidth 1 le_rfl, ?_, ?_, accounted_missed 0 1 1,
    trained_prohibited 0 1 1 univ (by norm_num)⟩
  · rw [hu]
    norm_num
  · unfold Frame.bound
    rw [hu, Φ_eq, ω_eq]
    simp

/-- The circuit of a run out of order: three computed gates that read nothing, and gate `3` reading `0` and `2`. -/
abbrev lateCircuit : Circuit where
  N := 4
  w _ := 1
  isInput _ := false
  reads g := if g.val = 3 then {0, 2} else ∅
  reads_lt := by decide +kernel
  input_reads := by decide +kernel
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {3}

/-- The run out of order: every gate on the server, gate `3` at tick 0 and the gates it reads at tick 5. -/
abbrev lateRun : Placed Bool where
  toCircuit := lateCircuit
  x _ := true
  t g := if g.val = 3 then 0 else 5
  cell g := g.val
  cell_inj := by decide +kernel

/-- Gate `3` uses one unit of the server's compute and the others none. -/
abbrev lateUsage : Usage lateRun Unit := ⟨fun g => if g.val = 3 then 1 else 0, fun _ _ => ()⟩

/-- The run out of order, with no gate the declared circuit's. -/
abbrev late : Computation Bool Unit := ⟨lateRun, lateUsage, ∅, fun _ h => absurd h (notMem_empty _), ∅⟩

theorem late_region {J : Set ℕ} (h0 : 0 ∈ J) (h5 : 5 ∉ J) : lateRun.region {true} J = {3} := by
  ext g
  have key : ∀ g : Fin 4, g ≠ 3 → lateRun.t g = 5 := by decide
  rw [mem_region lateRun]
  constructor
  · rintro ⟨-, -, ht⟩
    by_contra hne
    exact h5 (key g (fun h => hne (mem_singleton.2 h)) ▸ ht)
  · intro hg
    obtain rfl := mem_singleton.1 hg
    exact ⟨rfl, by decide, h0⟩

theorem late_window (i : Fin (frame c m β I).n) (j : Fin (frame c m β I).cuts.J) :
    lateRun.region ((frame c m β I).server i) ((frame c m β I).span j) = {3} := by
  have hj : j = (0 : Fin 1) := Subsingleton.elim _ _
  subst hj
  change lateRun.region {true} (Set.Ico 0 1) = {3}
  exact late_region ⟨le_rfl, Nat.one_pos⟩ fun h => absurd h.2 (by decide)

theorem late_developer :
    lateRun.region (univ.biUnion (frame c m β I).server) (Set.Iio (frame c m β I).t) = {3} := by
  change lateRun.region (univ.biUnion fun _ : Fin 1 => ({true} : Finset Bool)) (Set.Iio 1) = {3}
  have h : (univ.biUnion fun _ : Fin 1 => ({true} : Finset Bool)) = {true} := by decide
  rw [h]
  exact late_region Nat.one_pos fun h => absurd (Set.mem_Iio.1 h) (by decide)

/-- The reference training algorithm is what the server did in the frame's ticks, gate for gate. -/
noncomputable def lateEmb : Containment decl.D (late.circuit (frame c m β I)).D where
  emb g := g
  emb_inj _ _ h := h
  emb_w _ := rfl
  isInput_emb g := by
    change (lateRun.cut (lateRun.region (univ.biUnion (frame c m β I).server)
      (Set.Iio (frame c m β I).t))).isInput g = ref.isInput g
    rw [late_developer]
    have key : ∀ g : Fin 4, (if g ∈ ({3} : Finset (Fin 4)) then lateRun.isInput g else true) = ref.isInput g := by
      decide
    exact key g
  reads_emb g hg := by
    change (lateRun.cut (lateRun.region (univ.biUnion (frame c m β I).server)
      (Set.Iio (frame c m β I).t))).reads g = (ref.reads g).map _
    rw [late_developer]
    have key : ∀ g : Fin 4, ref.isInput g = false →
        (if g ∈ ({3} : Finset (Fin 4)) then lateRun.reads g else ∅) =
          (ref.reads g).map ⟨fun g => g, fun _ _ h => h⟩ := by decide
    exact key g hg
  op_emb g hg v := by
    change BitVec.cast _ ((lateRun.cut (lateRun.region (univ.biUnion (frame c m β I).server)
      (Set.Iio (frame c m β I).t))).op g v) = 0
    simp only [Placed.cut]
    split_ifs <;> rfl
  outputs_emb o ho := by
    change o ∈ lateRun.region (univ.biUnion (frame c m β I).server) (Set.Iio (frame c m β I).t)
    rw [late_developer]
    exact ho

theorem late_trains : Trains (frame c m β I) late recipe decl (lateEmb c m β I) threading (start c m β I) :=
  trains_of c m β I late rfl (lateEmb c m β I)
    (fun g hg => (by decide +kernel : ∀ g ∈ threading.M, 1 ≤ lateUsage.π g) g hg) fun Q hQ j => by
    have hj : j = finProdFinEquiv ((0 : Fin 1), (0 : Fin 1)) := Subsingleton.elim (α := Fin 1) _ _
    ext g
    rw [hj, hQ 0 0 g]
    change g ∈ lateRun.region ((frame c m β I).server 0) ((frame c m β I).span 0) ↔ _
    rw [late_window]

theorem late_work : Work 1 late := by
  intro p a b
  change lateRun.work lateUsage.π (lateRun.region {p} (Set.Ico a b)) ≤ 1 * (b - a)
  unfold Placed.work
  by_cases h3 : (3 : Fin 4) ∈ lateRun.region {p} (Set.Ico a b)
  · have ht : (0 : ℕ) ∈ Set.Ico a b := ((mem_region lateRun).1 h3).2.2
    have hab : 1 ≤ b - a := by
      have := ht.1
      have := ht.2
      omega
    have h1 : ∑ g : Fin 4, lateUsage.π g ≤ 1 := by decide
    exact ((sum_le_sum_of_subset (subset_univ _)).trans h1).trans (by simpa using hab)
  · have key : ∀ g : Fin 4, g ≠ 3 → lateUsage.π g = 0 := by decide
    exact (Nat.le_of_eq (sum_eq_zero fun g hg => key g fun h => h3 (h ▸ hg))).trans (Nat.zero_le _)

theorem late_memory : Memory 0 late := by
  intro X t
  change lateRun.state X t ≤ ∑ _p ∈ X, 0
  have key : ∀ g a : Fin 4, a ∈ lateRun.reads g →
      lateRun.isInput a = false ∧ lateRun.x a = lateRun.x g ∧ lateRun.t g ≤ lateRun.t a := by decide
  have h : lateRun.into (lateRun.region X (Set.Ici t)) = ∅ := by
    refine eq_empty_of_forall_notMem fun a ha => ?_
    obtain ⟨hna, g, hg, hr⟩ := (ResourceAccounting.SecurityProofs.Actual.mem_into lateRun).1 ha
    obtain ⟨-, hx, ht⟩ := (mem_region lateRun).1 hg
    obtain ⟨k1, k2, k3⟩ := key g a hr
    exact hna ((mem_region lateRun).2 ⟨k1, k2 ▸ hx, Set.mem_Ici.2 ((Set.mem_Ici.1 ht).trans k3)⟩)
  unfold Placed.state
  rw [h]
  simp [Circuit.width]

theorem late_bandwidth : Bandwidth 0 late := by
  change lateRun.Links (fun _ : Unit => (false, true)) (fun _ => 0) lateUsage.link
  have key : ∀ g k : Fin 4, g ∈ lateRun.reads k → lateRun.x g = lateRun.x k := by decide
  refine ⟨fun g a ha hne => absurd (key a g ha) hne, fun e y a b => ?_⟩
  have he : lateRun.readOn lateUsage.link e y (Set.Ico a b) = ∅ := by
    refine eq_empty_of_forall_notMem fun g hg => ?_
    simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hg
    obtain ⟨k, hk, hx, -⟩ := hg
    exact hx (key g k hk)
  rw [he]
  simp [Circuit.width]

/-- **The order clause is needed.** On a server that carries nothing, over a link that carries nothing, the run out
of order meets the work, memory and bandwidth clauses, and it is accounted for and prohibited at the fit bound. -/
theorem order_needed :
    Work 1 late ∧ Memory 0 late ∧ Bandwidth 0 late ∧ Escapes late (frame 1 0 0 ∅) 0 0 0 0 0 := by
  have hu : (frame 1 0 0 ∅).unaccounted 0 0 0 recipe = 0 := by
    simp [Frame.unaccounted, m_eq, b_eq, ω_eq]
  refine ⟨late_work, late_memory, late_bandwidth, ?_, ?_,
    fun i j => ⟨fun _ => ⟨Nat.zero_le _, Nat.zero_le _, ?_⟩, fun h => absurd h (notMem_empty _)⟩,
    prohibited_of 1 0 0 ∅ late _ (late_trains 1 0 0 ∅) (by norm_num)⟩
  · rw [hu]
    norm_num
  · unfold Frame.bound
    rw [hu]
    simp
  · change lateRun.ingress ((frame 1 0 0 ∅).server i) ((frame 1 0 0 ∅).span j) ≤ 0
    unfold Placed.ingress
    rw [late_window, Finset.filter_congr_decidable]
    change lateRun.width ((lateRun.into {3}).filter fun g => lateRun.x g ∉ ({true} : Finset Bool)) ≤ 0
    decide

end Slowdown.SecurityProofs.SecureWitness
