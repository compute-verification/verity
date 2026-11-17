module

public import Proofs.Flock.SpotFlock.FirstHitClassAt
public import Proofs.Flock.SpotFlock.ClassCounts

/-! The by-class first-hit bound at a budget at SPOT[Flock]'s code, with its three counts in closed form at caps by
template. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- What it still takes: the caps `M t` and the first stage's count `K₀`, read off the registration, the budget `T` a
run, and the linear expected-time bound at `B_conf,X` and at the two by-class finders. -/
theorem code_eventTwoX_le_cls_at_caps (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ}
    (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P L.V x ω r).pLog.length +
      (run FlockDeploy.sha512 P L.V x ω r).vLog.length ≤ T) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs (h.reg.blockOf ρ t i)).card ≤ M t)
    {K₀ : ℕ} (hK : firstK h.reg.first = some K₀)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      L.εclsAt FlockDeploy.sha512 x P T
        ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).length +
            K₀ * h.reg.ks.toList.sum : ℕ) +
          Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
            (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
            (SamplingDeploy.units ((base h).atOrder (order h)) x : ℝ))
        (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum +
            K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) +
          Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
            (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
            (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum : ℝ))
        (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum +
            K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) +
          Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
            (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
            (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum : ℝ)) := by
  subst hL
  exact code_eventTwoX_le_cls_at_budget h hB hS rfl hx hP hd hin P hneeds hcls T hT
    (code_hNs h (FlockDeploy.deployZk _root_.Flock.Readable.relation) x hK)
    (code_hKs_cls h (FlockDeploy.deployZk _root_.Flock.Readable.relation) x needs cls M hM hK)
    (code_hKm_cls h (FlockDeploy.deployZk _root_.Flock.Readable.relation) x needs cls
      (by subst hcls; exact Claims.SamplingGeneric.Compiled.clsOf_idem _) M hM hK)
    hBc hX hI

theorem code_eventTwoX_le_cls_at_cap (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ}
    (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P L.V x ω r).pLog.length +
      (run FlockDeploy.sha512 P L.V x ω r).vLog.length ≤ T) (M : ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs (h.reg.blockOf ρ t i)).card ≤ M)
    {K₀ : ℕ} (hK : firstK h.reg.first = some K₀)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      L.εclsAt FlockDeploy.sha512 x P T
        ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).length +
            K₀ * h.reg.ks.toList.sum : ℕ) +
          Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
            (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
            (SamplingDeploy.units ((base h).atOrder (order h)) x : ℝ))
        (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum +
            K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M).sum : ℕ) +
          Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
            (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
            (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum : ℝ))
        (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum +
            K₀ * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M).sum : ℕ) +
          Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
            (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
            (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
              fun u => (Claims.SamplingGeneric.Compiled.clsNeeds (L.nG x) cls needs u).card).sum : ℝ)) :=
  code_eventTwoX_le_cls_at_caps h hB hS hL hx hP hd hin P hneeds hcls T hT (fun _ => M) hM hK hBc hX hI

end SpotFlock
