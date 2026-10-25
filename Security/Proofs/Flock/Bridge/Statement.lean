import Proofs.Flock.EndToEndHidden.Statement
import Proofs.Flock.Bridge.Defs

/-!
# The sampled audit's knowledge soundness, live (stated, not proved)

`SampledKSLive` is `EndToEndHidden`'s setting with its event read in the sampling claims' relation: for every live
prover, the probability that flock-verify accepts and the committed transcript fixed at registration
(`XplurZC`), read as a sampling witness (`Bridge.witOf`) of the session's statement (`Bridge.stmtOf`, `Bridge.partOf`)
under the registration's commitment (`Bridge.commitOf`), is outside `RelS` at
`δ = C(n − K₀, kd)/C(n, kd)` is at most `δ + ε_ks + δ_link`, `EndToEndHidden`'s terms. Its named assumptions are
`EndToEndHidden`'s: A3, `live-verifier`, `cr/sha-512` and `ecr/sha-512`.

`SampledKSLiveStated` freezes the statement and the definitions it reads; no proof of `SampledKSLive` exists yet.
`SampledKSLive` holds at `kd = 0` (`δ = 1`), and elsewhere follows from `EndToEndHidden` at
`min K₀ (n − kd + 1)` by `Bridge.FaultyRightTarget` and `Bridge.CountTarget` (`Bridge.AssemblyTarget`).
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **The sampled audit's knowledge soundness, live.** `EndToEndHidden`'s setting: for every live prover `τ`, every
hash `H`, descriptor bytes `desc`, query `q` and roots `roots` (none of which `RelS` reads at `commitOf`),
$$\Pr[\text{flock-verify accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witOf}(X))] \le
  \delta + \varepsilon_{ks} + \delta_{link}, \qquad \delta = \binom{n-K_0}{kd} / \binom{n}{kd},$$
`X` the committed transcript fixed at registration (`XplurZC`), `ε_ks` and `δ_link` `EndToEndHidden`'s. -/
def SampledKSLive : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read at `commitOf`
    (H : Claims.Msg → Claims.Msg) (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let X := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS H (Bridge.commitOf vb.pos vb.bit vb.val vb.com (vb.cs (reg σ))) δ
          (Bridge.stmtOf P kd desc q roots) (Bridge.partOf P) (Bridge.witOf _ X))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

/-- **`SampledKSLive` is stated**: it, and the definitions it reads, equal to itself. -/
def SampledKSLiveStated : Prop :=
  let s := SampledKSLive
  s = s

end Flock.Guarantees
