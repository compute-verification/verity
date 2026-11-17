module

public import Proofs.Flock.SpotFlock.Counts
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassCount

/-! The by-class first-hit bound's counts `N_s`, `K_s` and `K_m` at SPOT[Flock]'s code. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)

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
