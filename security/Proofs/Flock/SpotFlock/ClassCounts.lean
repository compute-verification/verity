module

public import Proofs.Flock.SpotFlock.Counts
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassCount

/-!
# The by-class first-hit bound's counts at SPOT[Flock]'s code

`CompiledTwo.eventTwoX_le_cls_at_budget` reads three expectations over the coins: the sessions of a run's sample
(`N_s`) and, under `cls`, the classes of the gates the sample needs (`K_s`) and the classes' representatives its
sessions need, session by session (`K_m`). At SPOT[Flock]'s compiled verifier:

* `N_s` is the count of `code_hNs` as it stands;
* `K_s` and `K_m` are at most the counts by gates where each proof block needs the representatives of its gates'
  classes (`Compiled.clsNeeds`; `Compiled.ex_cls_le`, and `Compiled.ex_sessCountIn_cls_le` where a representative
  represents itself), so `code_hKs` and `code_hKm` at those needs bound them by template (`code_hKs_cls`,
  `code_hKm_cls`): `Σ_always |clsNeeds u| + K · Σ_t k_t · M_t + Pr[refused] · Σ_{u<N} |clsNeeds u|`, with `M_t` the
  most classes a block of template `t` needs (`|clsNeeds u|` is the classes among the gates `u` needs:
  `Compiled.card_clsNeeds`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)

/-- **`K_s` by class at SPOT[Flock]'s code**: with every block of template `t` needing at most `M t` classes under
`cls`, the classes of the gates a run's sample needs average at most the always-checked blocks' classes below `N` plus
`K · Σ_t k_t · M t`, plus the refusal's probability times every block's classes below `N`. -/
theorem code_hKs_cls (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) →
      Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x)) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls
        needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      ((((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r)).image cls).card : ℝ)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum : ℝ) :=
  (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.ex_cls_le x needs cls
    (code_hKs h F x (Claims.SamplingGeneric.Compiled.clsNeeds
      ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs) M hM hK)

/-- **`K_m` by class at SPOT[Flock]'s code**: with a representative representing itself under `cls` and every block of
template `t` needing at most `M t` classes, the classes' representatives a run's sessions need, session by session,
average at most the always-checked blocks' classes below `N` plus `K · Σ_t k_t · M t`, plus the refusal's
probability times every block's classes below `N`. -/
theorem code_hKm_cls (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) →
      Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x)) (hidem : ∀ g, cls (cls g) = cls g)
    (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls
        needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn x needs
        (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs l).image cls)
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r) : ℝ)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum : ℝ) :=
  (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.ex_sessCountIn_cls_le x needs cls hidem
    (code_hKm h F x (Claims.SamplingGeneric.Compiled.clsNeeds
      ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs) M hM hK)

end SpotFlock
