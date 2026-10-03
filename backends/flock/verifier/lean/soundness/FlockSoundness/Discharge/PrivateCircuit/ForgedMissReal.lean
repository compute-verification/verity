import FlockSoundness.Discharge.PrivateCircuit.InstanceReal

/-!
# The law's miss at the forged units, at the universal unit's `--zk --registered` session

`ForgedMiss` asks the law to miss the forged units `F o`. In the live audit (`auditLive`) the prover sends its
registration, then the verifier draws its coin, so a set `F` that reads only the registration is fixed before the draw,
and missing it is the law's escape at it (`prob_auditLive_escape_le`). At `realSessionR` the live draw escapes no more
often than `Law.stratified` under A3 (`Law.execOS_escape_le`), so `ForgedMiss` holds for every such `F`
(`forgedMiss_realR`). The committed values `s.vals τ` do not read the outcome, so the forged units of the committed
program are such an `F`, and `private_circuit_sound_realF` is `private_circuit_sound_real` without `ForgedMiss`.

The session enters only through its game (`auditLive` at `Law.execOS`) and its law (`Law.stratified`), which the
hidden-outputs session shares.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg Binding
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **The live audit misses a set fixed by the registration with at most the law's miss.** In `auditLive L Reg live`,
played by any `τ`, accepting (`acc`, any event) while the draw misses `F R` for the registration `R` the outcome
records, with `F R` of at least `K` units, has probability at most `L.miss K`. -/
theorem prob_auditLive_escape_le (L : Law n) (live : Finset (Fin n) → Reg → Game Exec.Transcript)
    (τ : Strategy (auditLive L Reg live)) (acc : Reg × Finset (Fin n) × Exec.Transcript → Prop)
    (F : Reg → Finset (Fin n)) (K : ℕ) :
    prob (fun o => acc o ∧ (∀ u ∈ o.2.1, u ∉ F o.1) ∧ K ≤ (F o.1).card) (auditLive L Reg live) τ ≤ L.miss K := by
  sorry

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

/-- **`ForgedMiss` at the universal unit's `--zk --registered` session**, under A3 and the strata's agreement with the
live draw's (`Law.ExecStrata`): for every live prover `τ`, every `F` that reads only the outcome's registration (`hF`:
fixed before the draw) and every `K`, accepting while the audit's draw `o.2.1` misses `F o` and `F o` has at least `K`
units has probability at most `Law.stratified`'s `miss K`. -/
theorem forgedMiss_realR (hS : Law.ExecStrata σs ks St) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (τ : Strategy (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (F : Reg × Finset (Fin n) × Exec.Transcript → Finset (Fin n))
    (hF : ∀ o o' : Reg × Finset (Fin n) × Exec.Transcript, o.1 = o'.1 → F o = F o') (K : ℕ) :
    ForgedMiss (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ
      (fun o => o.2.1) F K := by
  sorry

end Session

/-- **Protocol 2 is sound at the universal unit's `--zk --registered` session, with the law's miss at the forged units
discharged.** `private_circuit_sound_real`, whose second bound no longer takes `ForgedMiss`: under the same hypotheses,
accepting with at least `K₀ + K₁` units of `C` wrong at the committed values' claimed inputs and outputs has probability
at most `miss(K₀) + miss(K₁) + ε_ks + δ_link + q²/2^513`, for every `K₁`. The step to the program at the drawn units
(`LeafBindsOn`) stays a hypothesis. -/
theorem private_circuit_sound_realF {cls : Universal.Cls}
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin (Universal.units cls)) → Lean.Json) {S₀ : Finset (Fin (Universal.units cls))}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin (Universal.units cls)) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
    {m : ℕ} (σs : Fin (Universal.units cls) → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src) (hA6 : A6Z I dj wr St Kd pop Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') {qF qS : ℝ}
    (C : Universal.Circ) (hC : Universal.fits C cls)
    (progOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      cls.Prog)
    (xOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → ℕ → Bool)
    (yOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → List Bool)
    (hunit : UnitWrong ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      progOf xOf yOf)
    (rd : Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsSh ht dj y₀ hc hU hpub hscope))
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (row : cls.Prog → Fin L → (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
    (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C cls) i)
    (τ : Strategy (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (hCR : (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).CR τ)
    (valPair : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ progOf
      (Universal.pad C cls) (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => o.2.1) valPair)
    (K₀ : ℕ) (cost : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      (fun o => some (progPairOn ps
        (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
        (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => regPaths c pub o.2.2 rd.port) valPair o)) cost q) :
    let s := realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS
    prob (fun o => LiveAcceptsZKR I own o ∧
        (K₀ ≤ wrongAgree C cls (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ∨
          (∃ u ∈ o.2.1, progOf (s.vals τ o) u ≠ Universal.pad C cls u) ∨
          ∃ i, ownAt own c pub o.2.2 rd.port rd.readAt (ps.site i) ∧
            (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).regLeaf o.1 (ps.site i) ≠
              (ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf
                (row (Universal.pad C cls) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    ∀ K₁, prob (fun o => LiveAcceptsZKR I own o ∧
          K₀ + K₁ ≤ wrong (Universal.eval C) (Universal.size C) (xOf (s.vals τ o)) (yOf (s.vals τ o))) s.G τ
        ≤ s.Lm.miss K₀ + s.Lm.miss K₁ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
