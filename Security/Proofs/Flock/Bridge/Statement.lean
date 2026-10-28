import Proofs.Flock.EndToEndHidden.Statement
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Bridge.Deployed

/-!
# The sampled audit's knowledge soundness, live (stated, not proved)

`SampledKSLive` is `EndToEndHidden`'s setting with its event read in the sampling claims' relation, at the deployed
trace commitment: for every live prover, the probability that flock-verify accepts and the committed transcript fixed
at registration (`XplurZC`), read as the sampling witness `Bridge.witDep` of the session's statement (`Bridge.stmtOf`
at the circuit's constants `Bridge.constsOf`, `Bridge.partOf`), is outside `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` is
at most `δ + ε_ks + δ_link`, `EndToEndHidden`'s terms. `RelS` is read at SHA-512 (`Claims.Sampling.sha512Msg`), under the registration's frame-v3 roots over the public
file's rows (`Bridge.rootsOf`), and at the deployed commitment `Bridge.commitDep`, whose layout is the game's rows read
off the circuit file (`HiddenExec.rowOfH`); nothing of the commitment, its roots or its hash is a free parameter. Its
named assumptions are `EndToEndHidden`'s: A3 and `live-verifier`. Its knowledge and link terms are `EndToEndHidden`'s
additive ones, its finders' probabilities of outputting a SHA-512 collision; `cr/sha-512` and `ecr/sha-512` enter
only outside it, bounding those.

**Input rows.** `Bridge.partOf P κ` puts each input row in the slot of the unit that owns it, its first reader
(`Bridge.slotOf`), and has no input class: that is what the deployed verifier checks, whose session (`HmRow.drawn`) opens
only the rows its drawn units read, while registration opens none. So a unit whose owned input row doesn't open is a
faulty slot, and `δ` bounds those slots with the rest. An input no unit reads is in no slot. The deployed
`Slots.layout` still lays a separate input class, which the claims' design reads `all`, so `partOf` is not yet its
partition.

`SampledKSLiveStated` freezes the statement and the definitions it reads; no proof of `SampledKSLive` exists yet.
`SampledKSLive` holds at `kd = 0` (`δ = 1`), and elsewhere follows from `EndToEndHidden` at
`min K₀ (n − kd + 1)` by `Bridge.TransportsLive`, `Bridge.FaultyRightTarget` and `Bridge.CountTarget`
(`Bridge.AssemblyTarget`).
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **The circuit's constants**: `keyProg`'s constant wire `oneG` is `1` and its zero `zeroG` is `0`, public circuit
structure, so `progOf` lowers them to constants (`.const`), which a proof unit reads with no opening. -/
noncomputable def _root_.Flock.Bridge.constsOf (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)
    (g : Fin (Discharge.Layout.keyN c hU n)) : Option Bool :=
  if g = Discharge.Layout.oneG then some true else if g = Discharge.Layout.zeroG then some false else none

open Classical in
/-- **The sampled audit's knowledge soundness, live.** `EndToEndHidden`'s setting: for every live prover `τ`, descriptor
bytes `desc` and query `q` (which `RelS` doesn't read), at SHA-512, the registration's roots and the deployed commitment,
$$\Pr[\text{flock-verify accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witDep}(X))] \le
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
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- what the sampling statement carries and `RelS` doesn't read
    (desc : Claims.Msg) (q : Claims.Sampling.Query),
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
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    -- the deployed commitment's layout: the game's rows, read off the circuit file
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- the registration's frame-v3 roots, over the public file's rows
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    -- the circuit's constants, public
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

open Classical in
/-- **The registered openings open the deployed commitment** (the bridge's `Transports`, at `SampledKSLive`'s
commitment, roots, hash and registration): in `EndToEndHidden`'s setting, at every unit `u`, every committed wire `g` of
`u` and every opening `o` of `g`'s registered commit string at the prover's registration, some data opens `g`'s claims
position to `o`'s bit there, under the registration's roots at SHA-512. With `tr i o := ⟨i, vb.bit (vb.val o) (φ i), d⟩`,
`d` chosen by it, it is `WrongSet.Transports execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb sha512Msg Cd
(stmtOf P κ kd desc q roots) (reg σ) u φ tr` at every `u`, `kd`, `desc`, `q` and every `φ` that maps into `P.io u` only
committed positions, its wire's (`φ i ∈ P.io u → posOf Cc (φ i) = i`). -/
def _root_.Flock.Bridge.TransportsLive : Prop :=
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
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    -- the deployed commitment's layout: the game's rows, read off the circuit file
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- the registration's frame-v3 roots, over the public file's rows
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    ∀ u, ∀ g ∈ P.io u, ∀ o : vb.Opening, vb.com o = vb.cs (reg σ) (vb.pos g) →
      ∃ d : Claims.Msg, Cd.Opens Claims.Sampling.sha512Msg roots ⟨Bridge.posOf Cc g, vb.bit (vb.val o) g, d⟩ = true

/-- **`SampledKSLive` is stated**: it, and the definitions it reads, equal to itself. -/
def SampledKSLiveStated : Prop :=
  let s := SampledKSLive
  s = s

end Flock.Guarantees
