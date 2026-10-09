import Proofs.Flock.Soundness.Discharge.PrivateCircuit.ReadsHidden
import Mathlib.Algebra.MvPolynomial.SchwartzZippel

/-!
# Route P: hidden wiring over a public table of gates

Route P hides a circuit whose proof units all differ, with no recursion. Every proof unit runs the same public unit circuit,
a table of gate rows (each row one AND and one XOR gate in a fixed layout) over `nc` local cells; what differs between
units, and is hidden, is each unit's **wiring** `σ_u : Fin nc → Fin nc`, whose cycles are the unit's nets (one source cell
and its consumers). The wiring is committed at registration in the unit's topology entry, with the store cells the unit
reads (`hiddenPop` at `routeCircuits`). Each audit checks, by a grand-product argument over the drawn units' cells, that
the committed cell values follow the wiring: cells on one cycle carry one value.

* **`WiringSound`** (`WireId`, `Wired`, `wirePoly`): over any field `F` and any finite set `ι` of cells, tagged by an
  injective `lab : ι → F`, the identity `∏ᵢ (wᵢ + β·lab i + γ) = ∏ᵢ (wᵢ + β·lab (σ i) + γ)` holds as a polynomial in
  `(β, γ)` iff `w ∘ σ = w` (`wirePoly_eq_zero_iff`, for a permutation; `wired_of_wirePoly_eq_zero` for any function,
  which is what a check of a committed `σ` sees), and `w ∘ σ = w` iff `w` is constant on `σ`'s cycles
  (`wired_iff_cycles`). `wiring_sound`: when `w ∘ σ ≠ w`, the identity holds at uniform `(β, γ)` with probability at most
  `|ι|/|F|` (Schwartz–Zippel, total degree `|ι|`). `wiring_escape` is the form the protocol uses: whatever the prover
  fixes before the verifier draws `(β, γ)`, accepting the check while its values do not follow its wiring has probability
  at most `|ι|/|F|`. `wired_diag`: the batched check over several units' cells, tagged by (unit, local cell), checks
  each unit's wiring.
* **Acyclicity is local, against one global order**, the public unit index and, within a unit, its cell index. Each unit
  proves two things in `RouteRel`, its own unit relation:
  - **external**: each read points below its own unit's index (`Entry.Acyclic`, as in `ClassRel`);
  - **internal**: its wiring `σ_u` is ordered (`Ordered` against the public gate rows `GateRows`: every consumer cell is
    wired from a cell below it, and the input heads sit at their fixed cells). An ordered wiring determines its outputs
    (`ordered_determines`): cells on which the gate rows hold and which follow `σ_u` are `runOrd σ_u` of the unit's inputs,
    the reference semantics that evaluates the cells in increasing order (`evalOrd`).

  Both are checked only at the drawn units, which is all soundness needs: a unit whose registered wiring is not ordered
  fails its unit relation, so if it is drawn it is wrong in the committed values (`ε_ks + δ_link`), and if it is not drawn
  it is at most a unit that may be wrong against the store (`bad`), which the draw misses at the law's rate like any other.
  The undrawn units' wiring is irrelevant: the bound reads no property of the registered table. In the implementation a
  unit's internal proof is done once, the first time it is drawn, and later audits reuse it through the binding of
  `com_σ_u`; here it is part of each drawn unit's relation.
* **`route_p_sound`**: protocol 2 with route P's units. It is `private_circuit_sound_reads` at `routeCircuits`, with
  the block conjunct of the class relation replaced by route P's unit check (`RouteRel`: the public gate rows hold on the
  unit's committed cells, which carry its reads and writes, and its wiring is ordered) and one more term, the wiring
  check's escape `εW` (`hW`,
  `wiring_escape` at the session's coins over the drawn units' cells). A wrong owed output means a drawn unit wrong in
  the committed values (`ε_ks + δ_link`), the draw missing the units wrong against the store (the law's escape), a drawn
  unit whose cells do not follow its wiring with the check passed (`εW`), or a SHA-512 collision on the store or the
  table. The draw is over units and reads only the verifier's coins and the public class: one size class, one public
  unit circuit, so the law has no strata by circuit (`s.Lm` is any law over the units).
* What `σ` computes is the unit circuit's semantics `runP σ`, which `route_p_sound` ties to `runOrd` only at ordered
  wirings (`hrun`; `runP := runOrd R outC` meets it by `rfl`). A cyclic wiring can leave the cells under-determined
  (`Determines` fails), and at one the owed outputs are judged against whatever `runP` gives.

Route P's privacy, without recursion, rests on the registered commitment's hiding over the openings one registration
allows; it is not stated here.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

namespace RouteP

open Game
open scoped ENNReal

section Wiring

variable {F : Type} [_root_.Field F] {ι : Type} [Fintype ι] [DecidableEq ι]

/-- **The grand-product identity** at the verifier's coins `β, γ`: the cells' values `w` tagged by their own labels and by
their wiring's labels give one product. -/
def WireId (lab : ι → F) (w : ι → F) (σ : ι → ι) (β γ : F) : Prop :=
  ∏ i, (w i + β * lab i + γ) = ∏ i, (w i + β * lab (σ i) + γ)

/-- **The values follow the wiring**: every cell carries the value of the cell its wiring sends it to. -/
def Wired {α : Type} (w : ι → α) (σ : ι → ι) : Prop :=
  ∀ i, w (σ i) = w i

/-- **The identity as a polynomial**, in `γ = X 0` and `β = X 1`: the two products' difference. -/
noncomputable def wirePoly (lab w : ι → F) (σ : ι → ι) : MvPolynomial (Fin 2) F :=
  ∏ i, (MvPolynomial.X 0 + MvPolynomial.C (w i) + MvPolynomial.C (lab i) * MvPolynomial.X 1) -
    ∏ i, (MvPolynomial.X 0 + MvPolynomial.C (w i) + MvPolynomial.C (lab (σ i)) * MvPolynomial.X 1)

/-- **Following a permutation's wiring is being constant on its cycles.** -/
theorem wired_iff_cycles {α : Type} (w : ι → α) (σ : Equiv.Perm ι) :
    Wired w σ ↔ ∀ (i : ι) (k : ℕ), w ((σ ^ k) i) = w i := by
  constructor
  · intro h i k
    induction k with
    | zero => simp
    | succ k ih => rw [pow_succ', Equiv.Perm.mul_apply, h]; exact ih
  · intro h i
    simpa using h i 1

/-- The polynomial vanishes at `(γ, β)` exactly when the identity holds there. -/
theorem wirePoly_eval (lab w : ι → F) (σ : ι → ι) (β γ : F) :
    MvPolynomial.eval ![γ, β] (wirePoly lab w σ) = 0 ↔ WireId lab w σ β γ := by
  have e : ∀ a b : F, γ + a + b * β = a + β * b + γ := fun a b => by ring
  simp only [wirePoly, WireId, map_sub, map_prod, map_add, map_mul, MvPolynomial.eval_X, MvPolynomial.eval_C,
    sub_eq_zero, Matrix.cons_val_zero, Matrix.cons_val_one, e]

/-- **The identity as a polynomial forces the wiring**, for any committed `σ`, with injective labels. -/
theorem wired_of_wirePoly_eq_zero (lab w : ι → F) (σ : ι → ι) (hlab : Function.Injective lab)
    (h : wirePoly lab w σ = 0) : Wired w σ := by
  classical
  let e := MvPolynomial.finSuccEquiv F 1
  let a : ι → MvPolynomial (Fin 1) F := fun i => MvPolynomial.C (w i) + MvPolynomial.C (lab i) * MvPolynomial.X 0
  let b : ι → MvPolynomial (Fin 1) F := fun i => MvPolynomial.C (w i) + MvPolynomial.C (lab (σ i)) * MvPolynomial.X 0
  have eC : ∀ x : F, e (MvPolynomial.C x) = Polynomial.C (MvPolynomial.C x) := fun x => by
    rw [← MvPolynomial.algebraMap_eq, AlgEquiv.commutes, Polynomial.algebraMap_apply, MvPolynomial.algebraMap_eq]
  have e1 : e (MvPolynomial.X 1) = Polynomial.C (MvPolynomial.X 0) :=
    MvPolynomial.finSuccEquiv_X_succ (j := (0 : Fin 1))
  have ef : ∀ x l : F, e (MvPolynomial.X 0 + MvPolynomial.C x + MvPolynomial.C l * MvPolynomial.X 1) =
      Polynomial.X - Polynomial.C (-(MvPolynomial.C x + MvPolynomial.C l * MvPolynomial.X 0)) := fun x l => by
    rw [map_add, map_add, map_mul, MvPolynomial.finSuccEquiv_X_zero, eC, eC, e1, map_neg, sub_neg_eq_add, map_add,
      map_mul, add_assoc]
  have hprod : ∀ c : ι → MvPolynomial (Fin 1) F,
      ∏ i, (Polynomial.X - Polynomial.C (-c i)) =
        ((Finset.univ.val.map fun i => -c i).map fun x => Polynomial.X - Polynomial.C x).prod := fun c => by
    rw [Multiset.map_map, Finset.prod_eq_multiset_prod]
    rfl
  have heq := congrArg e (sub_eq_zero.1 h)
  simp only [map_prod, ef] at heq
  rw [hprod a, hprod b] at heq
  have hroots := congrArg Polynomial.roots heq
  rw [Polynomial.roots_multiset_prod_X_sub_C, Polynomial.roots_multiset_prod_X_sub_C] at hroots
  intro i
  have hmem : -b i ∈ Finset.univ.val.map fun i => -a i := by
    rw [hroots]
    exact Multiset.mem_map_of_mem _ (Finset.mem_univ_val i)
  obtain ⟨k, -, hk⟩ := Multiset.mem_map.1 hmem
  have hk' : a k = b i := neg_inj.1 hk
  have h0 := congrArg (fun p : MvPolynomial (Fin 1) F => p.coeff 0) hk'
  have h1 := congrArg (fun p : MvPolynomial (Fin 1) F => p.coeff (Finsupp.single 0 1)) hk'
  have hs : (0 : Fin 1 →₀ ℕ) ≠ Finsupp.single 0 1 := (Finsupp.single_ne_zero.2 one_ne_zero).symm
  simp [a, b, MvPolynomial.coeff_X, hs] at h0 h1
  rw [← hlab h1]
  exact h0

/-- **At a permutation, the identity holds as a polynomial iff the values follow the wiring.** -/
theorem wirePoly_eq_zero_iff (lab w : ι → F) (σ : Equiv.Perm ι) (hlab : Function.Injective lab) :
    wirePoly lab w σ = 0 ↔ Wired w σ := by
  refine ⟨wired_of_wirePoly_eq_zero lab w σ hlab, fun h => ?_⟩
  rw [wirePoly, sub_eq_zero]
  have : ∀ i, MvPolynomial.X 0 + MvPolynomial.C (w i) + MvPolynomial.C (lab (σ i)) * MvPolynomial.X 1 =
      (fun j => MvPolynomial.X 0 + MvPolynomial.C (w j) + MvPolynomial.C (lab j) * (MvPolynomial.X 1 :
        MvPolynomial (Fin 2) F)) (σ i) := fun i => by simp only [h i]
  rw [Finset.prod_congr rfl (fun i _ => this i)]
  exact (Equiv.prod_comp σ _).symm

/-- **Completeness**: values that follow a permutation's wiring pass at every coin. -/
theorem wireId_of_wired (lab w : ι → F) (σ : Equiv.Perm ι) (h : Wired w σ) (β γ : F) : WireId lab w σ β γ := by
  unfold WireId
  have : ∀ i, w i + β * lab (σ i) + γ = (fun j => w j + β * lab j + γ) (σ i) := fun i => by simp only [h i]
  rw [Finset.prod_congr rfl (fun i _ => this i), Equiv.prod_comp σ (fun j => w j + β * lab j + γ)]

/-- **The batched check's wiring is each unit's**: on cells tagged (unit, local cell), the block-diagonal wiring
`(u, i) ↦ (u, σ u i)` is followed iff every unit's own wiring is. -/
theorem wired_diag {U κ α : Type} (c : U → κ → α) (σ : U → κ → κ) :
    Wired (fun p : U × κ => c p.1 p.2) (fun p => (p.1, σ p.1 p.2)) ↔ ∀ u, Wired (c u) (σ u) := by
  constructor
  · intro h u i
    exact h (u, i)
  · rintro h ⟨u, i⟩
    exact h u i

variable [Fintype F] [DecidableEq F]

/-- **`WiringSound`**: values that do not follow the wiring pass the identity at uniform coins `(β, γ)` with probability
at most `|ι|/|F|`, for any committed `σ` and injective labels. -/
theorem wiring_sound (lab w : ι → F) (σ : ι → ι) (hlab : Function.Injective lab) (h : ¬ Wired w σ) :
    prCoin (fun c : F × F => WireId lab w σ c.1 c.2) ≤ (Fintype.card ι : ℝ≥0∞) / Fintype.card F := by
  classical
  have hP : wirePoly lab w σ ≠ 0 := fun h0 => h (wired_of_wirePoly_eq_zero lab w σ hlab h0)
  have hdeg : (wirePoly lab w σ).totalDegree ≤ Fintype.card ι := by
    have hf : ∀ l : ι → F, (∏ i, (MvPolynomial.X 0 + MvPolynomial.C (w i) + MvPolynomial.C (l i) * MvPolynomial.X 1 :
        MvPolynomial (Fin 2) F)).totalDegree ≤ Fintype.card ι := fun l => by
      refine (MvPolynomial.totalDegree_finsetProd _ _).trans ?_
      have h1 : ∀ i, (MvPolynomial.X 0 + MvPolynomial.C (w i) + MvPolynomial.C (l i) * MvPolynomial.X 1 :
          MvPolynomial (Fin 2) F).totalDegree ≤ 1 := fun i => by
        refine (MvPolynomial.totalDegree_add _ _).trans (max_le ((MvPolynomial.totalDegree_add _ _).trans
          (max_le ?_ ?_)) ?_)
        · rw [MvPolynomial.totalDegree_X]
        · rw [MvPolynomial.totalDegree_C]; exact zero_le_one
        · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
          rw [MvPolynomial.totalDegree_C, MvPolynomial.totalDegree_X]
      calc ∑ i, _ ≤ ∑ _i : ι, 1 := Finset.sum_le_sum fun i _ => h1 i
        _ = Fintype.card ι := by simp
    exact (MvPolynomial.totalDegree_sub _ _).trans (max_le (hf lab) (hf fun i => lab (σ i)))
  have hsz := MvPolynomial.schwartz_zippel_totalDegree hP (Finset.univ : Finset F)
  have hcard : (Finset.univ.filter fun c : F × F => WireId lab w σ c.1 c.2).card =
      (Finset.univ.filter fun f : Fin 2 → F => MvPolynomial.eval f (wirePoly lab w σ) = 0).card := by
    rw [← Fintype.card_subtype, ← Fintype.card_subtype]
    refine Fintype.card_congr (Equiv.subtypeEquiv ((Equiv.prodComm F F).trans (finTwoArrowEquiv F).symm) fun c => ?_)
    simp [wirePoly_eval]
  have hq : (0 : ℚ≥0) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  have hnat : (Finset.univ.filter fun f : Fin 2 → F => MvPolynomial.eval f (wirePoly lab w σ) = 0).card ≤
      Fintype.card ι * Fintype.card F := by
    have h2 : ((Finset.univ.filter fun f : Fin 2 → F => MvPolynomial.eval f (wirePoly lab w σ) = 0).card : ℚ≥0) /
        (Fintype.card F : ℚ≥0) ^ 2 ≤ (Fintype.card ι : ℚ≥0) / Fintype.card F := by
      have hdq : ((wirePoly lab w σ).totalDegree : ℚ≥0) ≤ Fintype.card ι := by exact_mod_cast hdeg
      refine le_trans ?_ (div_le_div_of_nonneg_right hdq hq.le)
      simpa [Fintype.piFinset_univ, Finset.card_univ] using hsz
    rw [div_le_div_iff₀ (pow_pos hq 2) hq, pow_two, ← mul_assoc] at h2
    exact_mod_cast le_of_mul_le_mul_right h2 hq
  rw [prCoin_eq_card, hcard, Fintype.card_prod]
  refine (ENNReal.div_le_div_right (Nat.cast_le.2 hnat) _).trans (le_of_eq ?_)
  push_cast
  exact ENNReal.mul_div_mul_right _ _ (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- **The wiring check's escape over the verifier's coins**: in any game `g` after which the verifier draws `(β, γ)`,
and for any prover, the outcome's values `w` and wiring `σ` (whatever `g` fixed) not following each other while the
identity holds at the drawn coins has probability at most `|ι|/|F|`. -/
theorem wiring_escape {α : Type} (lab : ι → F) (hlab : Function.Injective lab) (w : α → ι → F) (σ : α → ι → ι)
    (g : Game α) (s : Strategy (g.bind fun a => Game.coin (F × F) fun c => .ret (a, c))) :
    prob (fun x : α × (F × F) => ¬ Wired (w x.1) (σ x.1) ∧ WireId lab (w x.1) (σ x.1) x.2.1 x.2.2)
        (g.bind fun a => Game.coin (F × F) fun c => .ret (a, c)) s
      ≤ (Fintype.card ι : ℝ≥0∞) / Fintype.card F := by
  classical
  have hf : ∀ a, True → ∀ t, prob (fun x : α × (F × F) => ¬ Wired (w x.1) (σ x.1) ∧
      WireId lab (w x.1) (σ x.1) x.2.1 x.2.2) (Game.coin (F × F) fun c => .ret (a, c)) t
        ≤ (Fintype.card ι : ℝ≥0∞) / Fintype.card F := by
    intro a _ t
    rw [prob_coin]
    by_cases hw : Wired (w a) (σ a)
    · refine le_trans (le_of_eq ?_) zero_le
      simp [prob, hw, avg]
    · calc avg (fun c => prob (fun x : α × (F × F) => ¬ Wired (w x.1) (σ x.1) ∧
            WireId lab (w x.1) (σ x.1) x.2.1 x.2.2) (Game.ret (a, c)) (t c))
            = prCoin (fun c : F × F => WireId lab (w a) (σ a) c.1 c.2) := by
              unfold prCoin
              congr 1
              funext c
              simp [prob, hw]
        _ ≤ _ := wiring_sound lab (w a) (σ a) hlab hw
  refine (prob_bind_le _ _ (fun _ => True) _ hf g s).trans ?_
  have h0 : prob (fun _ : α => ¬ True) g (Strategy.ofBind _ g s) = 0 := by
    simpa using prob_false g (Strategy.ofBind _ g s)
  rw [h0, zero_add]

end Wiring

end RouteP

namespace Reads

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit RouteP

/-- **Route P's unit circuits**: a unit circuit is a wiring of `nc` local cells over the public gate table, run by its
reference semantics `runP`. -/
abbrev routeCircuits (F : Type) (nc nIn nOut : ℕ) (runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F) :
    UnitCircuits F nIn nOut where
  UC := Fin nc → Fin nc
  gates _ := nc
  run := runP

/-- **A wiring determines its outputs**: any cells on which the public gate rows hold (`gok`) and which follow the wiring
`σ` carry `runP σ` of their input cells at their output cells. -/
def Determines {F : Type} {nc nIn nOut : ℕ} (gok : (Fin nc → F) → Prop) (inC : Fin nIn → Fin nc) (outC : Fin nOut → Fin nc)
    (runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F) (σ : Fin nc → Fin nc) : Prop :=
  ∀ c : Fin nc → F, gok c → Wired c σ → ∀ l, c (outC l) = runP σ (fun k => c (inC k)) l

/-- **A cell of route P's gate table**: a gate cell (a source carrying its gate's function of lower cells; a constant
cell is a gate of no inputs), the head of input `k` (a source carrying that input), or a consumer (a gate's input cell or
any other cell that reads its net's source). -/
inductive CellKind (nIn : ℕ) where
  | gate
  | head (k : Fin nIn)
  | consumer

/-- **Route P's public gate rows** over `nc` cells: each cell's kind, and at each gate cell `j` its gate's function `fn j`
of the cells, which reads only cells below `j` (`fn_local`: in each row the gate's input cells precede its output cells,
as `a_g = l_g·r_g` and `x_g = l_g + r_g` follow `l_g, r_g`). -/
structure GateRows (F : Type) (nc nIn : ℕ) where
  kind : Fin nc → CellKind nIn
  fn : Fin nc → (Fin nc → F) → F
  fn_local : ∀ (j : Fin nc) (c c' : Fin nc → F), (∀ i, i < j → c i = c' i) → fn j c = fn j c'

/-- **The gate rows hold on the cells `c`**: every gate cell carries its gate's function of the cells. -/
def GateRows.Holds {F : Type} {nc nIn : ℕ} (R : GateRows F nc nIn) (c : Fin nc → F) : Prop :=
  ∀ j, R.kind j = .gate → c j = R.fn j c

/-- **The wiring `σ` is ordered** against the gate rows `R`, with input `k` at `inC k`, in the unit's own cell order:
every consumer cell `j` is the wiring's image of a cell below it (`σ i = j`, `i < j`), and the input heads are exactly
the cells `inC k`, each the head of its own input. A canonical `σ` (each net's cycle in increasing cell order from its
source) with `σ⁻¹ j < j` at every consumer `j` is ordered (`ordered_of_perm`). -/
def Ordered {F : Type} {nc nIn : ℕ} (R : GateRows F nc nIn) (inC : Fin nIn → Fin nc) (σ : Fin nc → Fin nc) : Prop :=
  (∀ j, R.kind j = .consumer → ∃ i, σ i = j ∧ i < j) ∧ ∀ j k, R.kind j = .head k ↔ j = inC k

/-- **The cells in increasing order** at the wiring `σ` and the inputs `x`: a gate cell is its gate's function of the cells
below it, the head of input `k` is `x k`, and a consumer `j` is the cell below it that `σ` sends to `j` (`default` when
there is none). -/
noncomputable def evalOrd {F : Type} [Inhabited F] {nc nIn : ℕ} (R : GateRows F nc nIn) (σ : Fin nc → Fin nc)
    (x : Fin nIn → F) (j : Fin nc) : F :=
  match R.kind j with
  | .gate => R.fn j fun i => if _h : i < j then evalOrd R σ x i else default
  | .head k => x k
  | .consumer => if h : ∃ i, σ i = j ∧ i < j then evalOrd R σ x (Classical.choose h) else default
termination_by j.val
decreasing_by
  all_goals first
    | assumption
    | exact (Classical.choose_spec h).2

/-- **Route P's reference semantics**: the output cells of `evalOrd`. -/
noncomputable def runOrd {F : Type} [Inhabited F] {nc nIn nOut : ℕ} (R : GateRows F nc nIn) (outC : Fin nOut → Fin nc)
    (σ : Fin nc → Fin nc) (x : Fin nIn → F) (l : Fin nOut) : F :=
  evalOrd R σ x (outC l)

/-- At an ordered wiring, cells on which the gate rows hold and which follow the wiring are `evalOrd` of their input
heads. -/
theorem evalOrd_eq {F : Type} [Inhabited F] {nc nIn : ℕ} (R : GateRows F nc nIn) (inC : Fin nIn → Fin nc)
    (σ : Fin nc → Fin nc) (ho : Ordered R inC σ) (c : Fin nc → F) (hc : R.Holds c) (hw : Wired c σ) (j : Fin nc) :
    c j = evalOrd R σ (fun k => c (inC k)) j := by
  suffices ∀ n (j : Fin nc), j.val = n → c j = evalOrd R σ (fun k => c (inC k)) j from this _ j rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  rintro j rfl
  rw [evalOrd]
  split
  next hk =>
    refine (hc j hk).trans (R.fn_local j _ _ fun i hi => ?_)
    rw [dite_eq_left hi]
    exact ih i.val hi i rfl
  next k hk =>
    exact congrArg c ((ho.2 j k).1 hk)
  next hk =>
    have h := ho.1 j hk
    rw [dite_eq_left h]
    obtain ⟨hs, hlt⟩ := Classical.choose_spec h
    rw [← ih _ hlt _ rfl]
    conv_lhs => rw [← hs]
    exact hw _

/-- **An ordered wiring determines its outputs**, at `runOrd`: for gate rows `R` that every `gok` assignment satisfies,
any cells on which `gok` holds and which follow an ordered `σ` carry `runOrd R outC σ` of their input cells at their
output cells. -/
theorem ordered_determines {F : Type} [Inhabited F] {nc nIn nOut : ℕ} (R : GateRows F nc nIn)
    {gok : (Fin nc → F) → Prop} (hR : ∀ c, gok c → R.Holds c) (inC : Fin nIn → Fin nc) (outC : Fin nOut → Fin nc)
    (σ : Fin nc → Fin nc) (ho : Ordered R inC σ) : Determines gok inC outC (runOrd R outC) σ :=
  fun c hg hw l => evalOrd_eq R inC σ ho c (hR c hg) hw (outC l)

/-- **A permutation whose inverse sends every consumer below itself is ordered**, with the input heads at `inC`. -/
theorem ordered_of_perm {F : Type} {nc nIn : ℕ} (R : GateRows F nc nIn) (inC : Fin nIn → Fin nc)
    (σ : Equiv.Perm (Fin nc)) (hπ : ∀ j, R.kind j = .consumer → σ.symm j < j)
    (hh : ∀ j k, R.kind j = .head k ↔ j = inC k) : Ordered R inC σ :=
  ⟨fun j hj => ⟨σ.symm j, σ.apply_symm_apply j, hπ j hj⟩, hh⟩

/-- **Route P's class relation at unit `u`**: `ClassRel`'s reads (the entry read opens the table root at `u` and is acyclic,
each input read opens the store root at the cell the entry names, each output write at `u`'s output row), the entry's
wiring ordered against the public gate rows `R` (`Ordered`), and the block conjunct replaced by the unit check: the public
gate rows hold on `u`'s committed cells `cl u` (`gok`), whose input cells carry its input reads and whose output cells its
output writes. The wiring is not checked here: the batched check over the drawn units' cells does that
(`wiring_escape`). -/
def RouteRel {F : Type} {nc nIn nOut N G M : ℕ} {runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F}
    (L : ReadLayout (hiddenPop (routeCircuits F nc nIn nOut runP) N) G)
    (lfT : ExplicitLeaf (Entry (hiddenPop (routeCircuits F nc nIn nOut runP) N))) (lfS : ExplicitLeaf F)
    (idx : Cell → Fin M) (R : GateRows F nc nIn) (gok : (Fin nc → F) → Prop) (inC : Fin nIn → Fin nc)
    (outC : Fin nOut → Fin nc) (cl : Fin N → Fin nc → Reads.Field G F) (rtT rtS : Flock.Registered.Port)
    (v : Fin G → Bool) (u : Fin N) : Prop :=
  ReadOk lfT rtT u ((L.ent u).read v) ∧ (L.tabOf v u).Acyclic u ∧ Ordered R inC (L.tabOf v u).dims ∧
    (∀ k, ReadOk lfS rtS (idx ((L.tabOf v u).src k)) ((L.inp u k).read v)) ∧
    (∀ l : Fin nOut, ReadOk lfS rtS (idx (.out u l)) ((L.out u l).read v)) ∧
    gok (fun i => (cl u i).read v) ∧ (∀ k, (cl u (inC k)).read v = L.xOf v u k) ∧
    ∀ l, (cl u (outC l)).read v = L.yOf v u l

end Reads

section Sound

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding Audit Reads
  RouteP
open scoped ENNReal

variable {F : Type} [Inhabited F] {nc nIn nOut N : ℕ} {runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F}
  {C : Audit.Circuit} {Pt : Audit.Partition C N} {Reg : Type} {M : ℕ}

open Classical in
/-- **Route P is sound at the drawn units.** Protocol 2 at `hiddenPop (routeCircuits F nc nIn nOut runP) N`: each unit's
topology entry holds its wiring `σ_u` and the store cells it reads, committed under the table root; the class net computes
route P's unit relation at each unit's check gate (`hcomp`: reads, the entry's wiring ordered against the public gate rows
`R`, public gate rows on the unit's committed cells `cl u`, and those cells carrying its reads and writes); every
assignment `gok` holds of satisfies `R` (`hR`); the unit semantics `runP` is `runOrd` at ordered wirings (`hrun`); and the wiring
check over the drawn units' cells escapes with at most `εW` (`hW`: accepting while a drawn unit not wrong in the committed
values has cells not following its entry's wiring). The rest is `private_circuit_sound_reads`'s. Then, with `T₀`, `w` the
registered table and store: (1) a delivered value off the committed one is at most `q_S²/2^513`; (2) if the registered
circuit's owed outputs are wrong, accepting is at most `escape(bad T₀ w) + ε_ks + δ_link + εW + q_S²/2^513 +
q_T²/2^513`; (3) a delivered public input off the committed cell or a delivered output off the registered circuit's end
to end on the committed cells is at most `q_S²/2^513` plus, when the owed outputs are wrong,
`escape(bad T₀ w) + ε_ks + δ_link + εW + q_T²/2^513`. -/
theorem route_p_sound
    (L : ReadLayout (hiddenPop (routeCircuits F nc nIn nOut runP) N) C.N)
    (lfT : ExplicitLeaf (Entry (hiddenPop (routeCircuits F nc nIn nOut runP) N))) (lfS : ExplicitLeaf F)
    (idx : Cell → Fin M) (R : GateRows F nc nIn) (gok : (Fin nc → F) → Prop) (hR : ∀ c, gok c → R.Holds c)
    (inC : Fin nIn → Fin nc) (outC : Fin nOut → Fin nc)
    (hrun : ∀ σ, Ordered R inC σ → runP σ = runOrd R outC σ)
    (cl : Fin N → Fin nc → Reads.Field C.N F)
    (rootT rootS : Reg → Flock.Registered.Port) (good : Reg → (Fin C.N → Bool) → Prop) (ok : Fin N → Fin C.N)
    (hcomp : ClassComputes Pt good (fun Rg v u => RouteRel L lfT lfS idx R gok inC outC cl (rootT Rg) (rootS Rg) v u)
      ok)
    (rtT₀ rtS₀ : Flock.Registered.Port)
    (tr : Fin N → ReadVal (Entry (hiddenPop (routeCircuits F nc nIn nOut runP) N))) (wo : Fin M → ReadVal F)
    (htr : ∀ j : Fin N, ReadOk lfT rtT₀ j (tr j)) (hwo : ∀ i : Fin M, ReadOk lfS rtS₀ i (wo i))
    (s : UnitSession Pt Reg) (drawn : s.Out → Finset (Fin N)) (hD : ZkSessionDrawn s drawn)
    (hE : DrawEscapes s drawn) (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (τ : Strategy s.G) (hCR : s.CR τ) (hgood : ∀ o, s.Accepts o → good (s.regOf o) (s.vals τ o))
    (εW : ℝ≥0∞)
    (hW : prob (fun o => s.Accepts o ∧ ∃ u ∈ drawn o, u ∉ Pt.wrong (s.vals τ o) ∧
        ¬ Wired (fun i => (cl u i).read (s.vals τ o)) (L.tabOf (s.vals τ o) u).dims) s.G τ ≤ εW)
    (owed : Finset (Fin N × Fin nOut)) (pubIn : Finset ℕ)
    (yD : s.Out → Fin N × Fin nOut → ReadVal F) (xD : s.Out → ℕ → ReadVal F)
    (hdel : ∀ o, s.Accepts o →
      (∀ p ∈ owed, ReadOk lfS (rootS (s.regOf o)) (idx (.out p.1 p.2)) (yD o p)) ∧
        ∀ m ∈ pubIn, ReadOk lfS (rootS (s.regOf o)) (idx (.reg m)) (xD o m))
    (costS costT : s.Out → ℝ) (qS qT : ℝ) (hcS : ∀ o, costS o ≤ qS) (hcT : ∀ o, costT o ≤ qT)
    (hS : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some (readPick lfS rtS₀ Fin.val wo (storeReads L idx s τ drawn owed pubIn yD xD) o)) costS qS)
    (hT : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some ((regCommit lfT N).ext (rtT₀, tabFill L s τ drawn tr o) (rtT₀, tr))) costT qT) :
    let w : Cell → F := fun c => (wo (idx c)).val
    let T₀ : Table (hiddenPop (routeCircuits F nc nIn nOut runP) N) := fun j => (tr j).val
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨ ∃ p ∈ owed, (yD o p).val ≠ w (.out p.1 p.2))) s.G τ
      ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) ∧
    (OwedWrongR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w owed →
      prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀) s.G τ
        ≤ s.Lm.escape (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w) + (s.εks τ + s.δlink τ) + εW +
          ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))) ∧
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨
          ∃ p ∈ owed, (yD o p).val ≠ evalR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ (fun m => w (.reg m))
            p.1 p.2)) s.G τ
      ≤ (if OwedWrongR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w owed then
          s.Lm.escape (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w) + (s.εks τ + s.δlink τ) + εW +
            ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))
          else 0) +
        ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
  intro w T₀
  classical
  have hwr := classWrongOn_of_computes hcomp
  have hRB := reads_bind_le s.G τ lfS rtS₀ Fin.val wo hwo (storeReads L idx s τ drawn owed pubIn yD xD) costS qS hcS hS
  have hrel : ∀ o, s.Accepts o → ∀ u, u ∉ Pt.wrong (s.vals τ o) →
      RouteRel L lfT lfS idx R gok inC outC cl (rootT (s.regOf o)) (rootS (s.regOf o)) (s.vals τ o) u :=
    fun o ha u hu => by
      by_contra hr
      exact hu ((hwr _ _ (hgood o ha) u).2 hr)
  have h1 : prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
      ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨ ∃ p ∈ owed, (yD o p).val ≠ w (.out p.1 p.2))) s.G τ
      ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
    refine le_trans (prob_mono (fun o ⟨ha, _, hS', hq⟩ => ?_) _ _) hRB
    have hd := hdel o ha
    rw [hS'] at hd
    rcases hq with ⟨m, hm, hne⟩ | ⟨p, hp, hne⟩
    · exact ⟨idx (.reg m), xD o m, Or.inr (Or.inr ⟨m, hm, rfl, rfl⟩), hd.2 m hm, hne⟩
    · exact ⟨idx (.out p.1 p.2), yD o p, Or.inr (Or.inl ⟨p, hp, rfl, rfl⟩), hd.1 p hp, hne⟩
  have h2 : OwedWrongR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w owed →
      prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀) s.G τ
        ≤ s.Lm.escape (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w) + (s.εks τ + s.δlink τ) + εW +
          ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) := by
    intro _
    let acc : s.Out → Prop := fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀
    let Q : s.Out → Prop := fun o => (∃ u ∈ drawn o, u ∈ Pt.wrong (s.vals τ o)) ∨
      (∃ u ∈ drawn o, u ∉ Pt.wrong (s.vals τ o) ∧
        ¬ Wired (fun i => (cl u i).read (s.vals τ o)) (L.tabOf (s.vals τ o) u).dims) ∨
      (∃ i x, storeReads L idx s τ drawn owed pubIn yD xD o i x ∧ ReadOk lfS rtS₀ (↑i) x ∧ x.val ≠ (wo i).val) ∨
        Disjoint (drawn o) (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w)
    have hQ : prob (fun o => acc o ∧ Q o) s.G τ
        ≤ s.Lm.escape (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w) + (s.εks τ + s.δlink τ) + εW +
          ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
      calc _ ≤ prob (fun o => (s.Accepts o ∧ ∃ u ∈ drawn o, u ∈ Pt.wrong (s.vals τ o)) ∨
            ((s.Accepts o ∧ ∃ u ∈ drawn o, u ∉ Pt.wrong (s.vals τ o) ∧
                ¬ Wired (fun i => (cl u i).read (s.vals τ o)) (L.tabOf (s.vals τ o) u).dims) ∨
              ((∃ i x, storeReads L idx s τ drawn owed pubIn yD xD o i x ∧ ReadOk lfS rtS₀ (↑i) x ∧
                  x.val ≠ (wo i).val) ∨
                (s.Accepts o ∧ Disjoint (drawn o)
                  ((fun _ : Reg => bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w) (s.regOf o)))))) s.G τ := by
            refine prob_mono (fun o ⟨⟨ha, _⟩, hq⟩ => ?_) _ _
            rcases hq with h | h | h | h
            · exact Or.inl ⟨ha, h⟩
            · exact Or.inr (Or.inl ⟨ha, h⟩)
            · exact Or.inr (Or.inr (Or.inl h))
            · exact Or.inr (Or.inr (Or.inr ⟨ha, h⟩))
        _ ≤ _ := prob_or_le _ _ _ _
        _ ≤ _ := add_le_add le_rfl (prob_or_le _ _ _ _)
        _ ≤ _ := add_le_add le_rfl (add_le_add le_rfl (prob_or_le _ _ _ _))
        _ ≤ (s.εks τ + s.δlink τ) + (εW + (ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) +
              s.Lm.escape (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w))) :=
            add_le_add (hD hA3 hA6 τ hCR)
              (add_le_add hW (add_le_add hRB ((hE τ _).trans (iSup_le fun _ => le_rfl))))
        _ = _ := by ring
    have hOpens : ∀ o, acc o → (regCommit lfT N).Opens rtT₀ (fun j => (tabFill L s τ drawn tr o j).val)
        (rtT₀, tabFill L s τ drawn tr o) := by
      rintro o ⟨ha, hT', -⟩
      refine show _ ∧ _ from ⟨rfl, fun j => ⟨rfl, ?_⟩⟩
      by_cases hj : j ∈ drawn o ∧ j ∉ Pt.wrong (s.vals τ o)
      · have := (hrel o ha j hj.2).1
        rw [hT'] at this
        simpa only [tabFill, ite_eq_left hj] using this
      · simpa only [tabFill, ite_eq_right hj] using htr j
    have hRR := ran_registered (regCommit lfT N) (regCommit_binding lfT N) rtT₀ T₀ (rtT₀, tr)
      (show _ ∧ _ from ⟨rfl, fun j => ⟨rfl, htr j⟩⟩) s.G τ acc Q
      (fun o j => (tabFill L s τ drawn tr o j).val) (fun o => (rtT₀, tabFill L s τ drawn tr o)) hOpens _ hQ
      costT qT hcT hT
    refine le_trans (prob_mono (fun o ha => ⟨ha, ?_⟩) _ _) hRR
    obtain ⟨ha, hT', hS'⟩ := ha
    by_cases hdis : Disjoint (drawn o) (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w)
    · exact Or.inl (Or.inr (Or.inr (Or.inr hdis)))
    obtain ⟨u, hu, hbad⟩ := Finset.not_disjoint_iff.1 hdis
    by_cases hwu : u ∈ Pt.wrong (s.vals τ o)
    · exact Or.inl (Or.inl ⟨u, hu, hwu⟩)
    have hr := hrel o ha u hwu
    by_cases hWd : Wired (fun i => (cl u i).read (s.vals τ o)) (L.tabOf (s.vals τ o) u).dims
    swap
    · exact Or.inl (Or.inr (Or.inl ⟨u, hu, hwu, hWd⟩))
    by_cases hT : L.tabOf (s.vals τ o) u = T₀ u
    · obtain ⟨hr1, hr2, hro, hr3, hr4, hg, hin, hout⟩ := hr
      have hcr : ClassRel L lfT lfS idx (rootT (s.regOf o)) (rootS (s.regOf o)) (s.vals τ o) u := by
        refine ⟨hr1, hr2, hr3, hr4, fun l => ?_⟩
        have hW' : Wired (fun i => (cl u i).read (s.vals τ o)) (tr u).val.dims := by
          rw [hT] at hWd
          exact hWd
        have hd : (cl u (outC l)).read (s.vals τ o) =
            runP (tr u).val.dims (fun k => (cl u (inC k)).read (s.vals τ o)) l := by
          have ho : Ordered R inC (tr u).val.dims := by
            rw [hT] at hro
            exact hro
          rw [hrun _ ho]
          exact ordered_determines R hR inC outC _ ho _ hg hW' l
        have hx : (fun k => (cl u (inC k)).read (s.vals τ o)) = L.xOf (s.vals τ o) u := funext hin
        show runP (L.tabOf (s.vals τ o) u).dims (L.xOf (s.vals τ o) u) l = L.yOf (s.vals τ o) u l
        rw [← hout l, hd, hx, hT]
      rcases classRel_bad L lfT lfS idx _ _ _ u hcr T₀ w hT hbad with ⟨k, hk⟩ | ⟨l, hl⟩
      · refine Or.inl (Or.inr (Or.inr (Or.inl ⟨idx ((L.tabOf (s.vals τ o) u).src k),
          (L.inp u k).read (s.vals τ o), Or.inl ⟨u, hu, hwu, Or.inl ⟨k, rfl, rfl⟩⟩, ?_, ?_⟩)))
        · have := hr3 k
          rw [hS'] at this
          exact this
        · show L.xOf (s.vals τ o) u k ≠ (wo (idx ((L.tabOf (s.vals τ o) u).src k))).val
          rw [hT]
          exact hk
      · refine Or.inl (Or.inr (Or.inr (Or.inl ⟨idx (.out u l), (L.out u l).read (s.vals τ o),
          Or.inl ⟨u, hu, hwu, Or.inr ⟨l, rfl, rfl⟩⟩, ?_, hl⟩)))
        have := hr4 l
        rw [hS'] at this
        exact this
    · refine Or.inr fun heq => hT ?_
      have := congrFun heq u
      unfold tabFill at this
      rw [ite_eq_left ⟨hu, hwu⟩] at this
      exact this
  refine ⟨h1, h2, ?_⟩
  by_cases hOW : OwedWrongR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w owed
  · rw [ite_eq_left hOW]
    refine le_trans (prob_mono (fun o ⟨ha, hT', hS', _⟩ => ⟨ha, hT', hS'⟩) _ _) ((h2 hOW).trans (le_of_eq ?_))
    ring
  · rw [ite_eq_right hOW, zero_add]
    refine le_trans (prob_mono (fun o ⟨ha, hT', hS', hq⟩ => ⟨ha, hT', hS', ?_⟩) _ _) h1
    rcases hq with h | ⟨p, hp, hne⟩
    · exact Or.inl h
    · refine Or.inr ⟨p, hp, fun heq => hne (heq.trans ?_)⟩
      by_contra hne'
      exact hOW ⟨p, hp, hne'⟩

end Sound

end FlockSoundness.Discharge.PrivateCircuit
