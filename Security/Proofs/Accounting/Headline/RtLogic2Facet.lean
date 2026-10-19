import Proofs.Accounting.Headline.Sm120Measured

/-!
# RT2-LOGIC-9: VF3r-16's π* row is not a ceiling on sm_120 (red-team witness, not for landing)

Builds on #2482's head (`cursor/lean-headline-sm120-ea24` @ a495f9c5dd), not on main.

`Sm120MeasuredHeadline` takes `FacetCeilings Sm120Measured.machine` on every run, and its π* is VF3r-16's joint issue
row JI3s_0 (`Sm120Measured.piStar = 86`). RT-FACET's probe `jm_i16832_fmuli_q120_p0_a3u20_w8` (IMMA.16832 beside
FMUL, unrolled 20 times; r20261017-192516-dd11 on vy-nebius-2 at 2092 MHz, SASS-gated) ran at 0.053850189564545986
IMMA16832 and 0.8077528434681898 FFMA warp instructions a SMSP-cycle (`out/loads.jsonl`'s `x`, counting only the
gated opcodes):

* `probe_load`: those rates load JI3s_0 at more than one cycle a cycle (1.0107);
* `not_facetCeilings`: so any run whose counts are at least those rates times its cycles fails `FacetCeilings` at π*.
  Every coefficient is `≥ 0`, so the loop's uncounted instructions only add to the load;
* `not_premise`: a play with one such path fails the measured headline's (I1) premise.
-/

namespace Accounting.SecurityProofs.TS.Headline.RtLogic2Facet

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Accounting.SecurityProofs.TS.Headline
  Pouw.RelativeSecurity

theorem piStar_name : Sm120Measured.rowNames.getD Sm120Measured.piStar.val "" = "JI3s_0" := rfl

theorem class_names : Sm120.classes.getD 27 "" = "IMMA16832" ∧ Sm120.classes.getD 13 "" = "FFMA" := ⟨rfl, rfl⟩

/-- The probe's measured rates: warp instructions of IMMA16832 (27) and FFMA (13) a SMSP-cycle. -/
def probe : Sparse := [(27, 53850189564545986, 10 ^ 18), (13, 8077528434681898, 10 ^ 16)]

theorem probe_load : 1 < dot (vfRow Sm120Measured.piStar) 122 probe := by decide +kernel

theorem probe_load_hi : dot (vfRow Sm120Measured.piStar) 122 probe < 10108 / 10000 := by decide +kernel

theorem machine_a : ∀ r k, Sm120Measured.machine.a r k = (vfRow r k.val : ℝ) := fun _ _ => rfl

theorem not_facetCeilings (P : Run (Fin 122)) (hc : 0 < P.cycles)
    (hn : ∀ k, Sparse.counts probe k * P.cycles ≤ P.n k) : ¬ FacetCeilings Sm120Measured.machine P := by
  rintro ⟨-, -, -, -, h⟩
  have hr := (h Sm120Measured.piStar).1 rfl
  have hge : Sm120Measured.machine.load (Sparse.counts probe) Sm120Measured.piStar * P.cycles ≤
      Sm120Measured.machine.load P.n Sm120Measured.piStar := by
    unfold Machine.load
    rw [Finset.sum_mul]
    refine Finset.sum_le_sum fun k _ => ?_
    rw [mul_assoc]
    exact mul_le_mul_of_nonneg_left (hn k) (by rw [machine_a]; exact Rat.cast_nonneg.mpr (coeff_nonneg _ _))
  rw [load_counts _ vfRow machine_a] at hge
  have h1 : ((1 : ℚ) : ℝ) < (dot (vfRow Sm120Measured.piStar) 122 probe : ℝ) := Rat.cast_lt.mpr probe_load
  rw [Rat.cast_one] at h1
  have h2 := mul_lt_mul_of_pos_right h1 hc
  linarith

theorem not_premise {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty)
    (x : Paths Ω Sm120.checks) (hc : 0 < (Pl.run x).run.cycles)
    (hn : ∀ k, Sparse.counts probe k * (Pl.run x).run.cycles ≤ (Pl.run x).run.n k) :
    ¬ ∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run :=
  fun h => not_facetCeilings _ hc hn (h x)

end Accounting.SecurityProofs.TS.Headline.RtLogic2Facet

#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2Facet.probe_load
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2Facet.probe_load_hi
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2Facet.not_facetCeilings
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2Facet.not_premise
