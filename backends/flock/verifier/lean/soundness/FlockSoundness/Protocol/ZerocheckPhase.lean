import FlockSoundness.Protocol.LincheckPhase
import FlockSoundness.Protocol.ListSize
import FlockSoundness.Protocol.OpeningPhase

/-!
# The zerocheck phase (PROTOCOL.md §9.2)

For one codeword near the committed table whose witness makes the region claims true, the doomed
start says the block R1CS or the pin fails. The zerocheck then ends with the `a`, `b`, `c` claims
and the pin's extension all true with small probability:

* **R1CS.** A nonzero error bit at skip index `s₀` stays visible after the eq-reduction over the
  other address bits (`m − 13` random coordinates: the pinned coordinates' weights are independent
  over GF(2)). `Pcomb` vanishes on the skip nodes by construction, so either the prover's `P_c` is
  wrong at `z` or its `P_ab` is (degree 127). The residual rounds keep a false claim false
  (`2/|F|` each, given every `r_eq ≠ 1`, one event for all codewords), and a false final claim
  makes the `a` or the `b` claim false.
* **Pin.** A block whose pin bit is 0 makes the pin's extension at `ρ_out` a nonzero polynomial of
  degree `m − k_log` in the rounds' coins.

`zerocheck_phase_of_card` takes the union over a list of at most `L` codewords.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Averages and coins -/

/-! ## Folding the lowest bit, and the equality weights -/

/-- Sums over `2^(n+1)` indices by pairs `(2y, 2y+1)`. -/
theorem sum_range_pairs {M : Type} [AddCommMonoid M] (n : ℕ) (f : ℕ → M) :
    ∑ x ∈ Finset.range (2 ^ (n + 1)), f x =
      ∑ y ∈ Finset.range (2 ^ n), (f (2 * y) + f (2 * y + 1)) := by
  rw [← Fin.sum_univ_eq_sum_range, ← Fin.sum_univ_eq_sum_range (fun y => f (2 * y) + f (2 * y + 1))]
  have h : 2 ^ n * 2 = 2 ^ (n + 1) := (pow_succ 2 n).symm
  rw [← (finCongr h).sum_comp, sum_fin_mul]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [Fin.sum_univ_two]
  simp only [finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val, Fin.val_zero, Fin.val_one]
  congr 2 <;> ring

/-- Fold the lowest bit at `t`: `u'(y) = u(2y) + t·(u(2y+1) + u(2y))`. -/
def foldLo (t : F) (u : ℕ → F) : ℕ → F := fun y => u (2 * y) + t * (u (2 * y + 1) + u (2 * y))

/-! ## The residual rounds (Z6) -/

/-! ## The nodes, and polynomials through them -/

/-- The skip nodes `S` (§9). -/
noncomputable def nodeS (A : Arith F K) (hA : A.Correct) : Fin 64 → F := hA.nodes.choose

/-- The output coset `Λ` (§9.2). -/
noncomputable def nodeΛ (A : Arith F K) (hA : A.Correct) : Fin 64 → F :=
  hA.nodes.choose_spec.choose

/-! ## The witness by skip index -/

/-! ## The true polynomials of the zerocheck -/

/-! ## The three coin bounds -/

/-- `Pc(z)` as the verifier computes it from `round1_c` (Z5). -/
def pcOf (A : Arith F K) (msg : (Fin 64 → F) × (Fin 64 → F)) (z : F) : F :=
  ∑ i, A.lagΛ z i * msg.2 i

/-- The zerocheck's starting claim `Pcomb(z) + Pc(z)` (Z5). -/
def claim0 (A : Arith F K) (msg : (Fin 64 → F) × (Fin 64 → F)) (z : F) : F :=
  ∑ i, A.combW z i * (msg.1 i + msg.2 i) + pcOf A msg z

/-! ## The game -/

/-! ## The phase -/

end FlockSoundness
