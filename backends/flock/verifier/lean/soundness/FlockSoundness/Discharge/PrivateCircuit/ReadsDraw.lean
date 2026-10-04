import FlockSoundness.Discharge.PrivateCircuit.ReadsDefs
import FlockSoundness.Discharge.PrivateCircuit.InstanceReal

/-!
# The draw over the padded population, set by set

The verifier draws over the public padded population, with the block classes as strata (`Law.stratified`). The set the
sampling term is about is the registered model's units wrong against the store, `bad T w`: the registration fixes it
before the draw, so the useful bound is the law's escape at that one set, not its miss at a count (`miss(1)` is near 1
under sparse sampling).

* `prob_auditLive_escape_set`: in the live audit (`auditLive`: the prover sends its registration, then the verifier draws),
  accepting while the draw misses a set `B R` that reads only the registration `R` has probability at most the largest
  escape of the law at `B R`.
* `drawEscapes_realR`: so the `--zk --registered` session's draw escapes every such set with at most `Law.stratified`'s
  escape (`DrawEscapes`), under A3 and the strata's agreement with the live draw's (`Law.ExecStrata`).
* `realR_escape_stratified`: at a fixed set `B`, that escape in closed form, `∏ₛ C(nₛ − |B ∩ stratum s|, kₛ)/C(nₛ, kₛ)`.
* `escape_bad_le_cone`: the escape at the wrong units is at most the escape at those of one cone.
-/

namespace FlockSoundness.Discharge.PrivateCircuit.Reads

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg Binding
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **The live audit misses a set fixed by the registration with at most the law's escape there.** In
`auditLive L Reg live`, played by any `τ`, accepting (`acc`, any event) while the draw misses `B R` for the registration
`R` the outcome records has probability at most `⨆ R, L.escape (B R)`. -/
theorem prob_auditLive_escape_set (L : Law n) (live : Finset (Fin n) → Reg → Game Exec.Transcript)
    (τ : Strategy (auditLive L Reg live)) (acc : Reg × Finset (Fin n) × Exec.Transcript → Prop)
    (B : Reg → Finset (Fin n)) :
    prob (fun o => acc o ∧ Disjoint o.2.1 (B o.1)) (auditLive L Reg live) τ ≤ ⨆ R, L.escape (B R) := by
  classical
  refine le_trans ?_ (le_iSup (fun R => L.escape (B R)) τ.1)
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => acc o ∧ Disjoint o.2.1 (B o.1)) ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t))
        (τ.2 ω) ≤ _
  unfold Law.escape prCoin avg
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  rw [prob_map]
  by_cases hd : Disjoint (L.draw ω) (B τ.1)
  · exact (prob_le_one _ _ _).trans (ite_eq_left hd).symm.le
  · exact (prob_mono (fun _ h => hd h.2) _ _).trans ((prob_false _ _).trans (ite_eq_right hd).symm).le

/-- **The escape at the wrong units is at most the escape at those in one cone.** -/
theorem escape_bad_le_cone {V : Type} {P : Pop V} (L : Law P.N) (T : Table P) (w : Cell → V) (i : Fin P.N) :
    L.escape (bad P T w) ≤ L.escape (cone P T i ∩ bad P T w) :=
  Law.escape_anti L Finset.inter_subset_right

section Session

variable (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (dg : ByteArray → ByteArray) (N : ℕ)
  {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
  {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hscope : Layout.scopeOk c = true)
  {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
  (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
  {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
  (k Rw M : ℕ) (ρ t' qF qS : ℝ)

/-- **The `--zk --registered` session's draw escapes a set fixed by the registration with at most `Law.stratified`'s
escape**, under A3 and the strata's agreement with the live draw's (`Law.ExecStrata`). -/
theorem drawEscapes_realR (hS : Law.ExecStrata σs ks St) (hA3 : Assumptions.UniformRandomBytes Lb src) :
    DrawEscapes (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
      (fun o => o.2.1) := by
  intro τ B
  change prob (fun o => LiveAcceptsZKR I own o ∧ Disjoint o.2.1 (B o.1))
      (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ
    ≤ ⨆ R, (Law.stratified σs ks hks).escape (B R)
  exact (prob_auditLive_escape_set _ _ τ _ B).trans
    (iSup_mono fun R => Law.execOS_escape_le σs ks hks St hS Kd pop Lb src hA3 (B R))

/-- **The miss at one set, in closed form**: in the `--zk --registered` session, under A3 and `Law.ExecStrata`, accepting
while the draw misses every unit of `B` has probability at most `∏ₛ C(nₛ − |B ∩ stratum s|, kₛ)/C(nₛ, kₛ)`, with `nₛ` the
size of stratum `s` and `kₛ` its draw. -/
theorem realR_escape_stratified (hS : Law.ExecStrata σs ks St) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (τ : Strategy (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (B : Finset (Fin n)) :
    prob (fun o => LiveAcceptsZKR I own o ∧ Disjoint o.2.1 B)
        (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      ≤ ∏ s, (((Law.stratum σs s).card - (B.filter (σs · = s)).card).choose (ks s) : ℝ≥0∞) /
          ((Law.stratum σs s).card.choose (ks s) : ℝ≥0∞) := by
  rw [← Law.stratified_escape σs ks hks B]
  exact (drawEscapes_realR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS hS hA3
    τ fun _ => B).trans (iSup_le fun _ => le_rfl)

end Session

end FlockSoundness.Discharge.PrivateCircuit.Reads
