module

public import Proofs.Flock.SpotFlock.FirstHitClassAt
public import Proofs.Flock.SpotFlock.ClassCounts

/-!
# The by-class first-hit bound at a budget, at SPOT[Flock]'s code, at caps by template

`code_eventTwoX_le_cls_at_budget` takes three counts over the coins as hypotheses: the sessions of a run's sample
(`N_s`), the classes of the gates the sample needs (`K_s`) and the classes its sessions need, session by session
(`K_m`). At the code each is a closed expression (`code_hNs`, `code_hKs_cls`, `code_hKm_cls`), so the bound holds with
those expressions in their place (`code_eventTwoX_le_cls_at_caps`):

* `N_s = |always| + K₀ · Σ_t k_t + Pr[refused] · N`;
* `K_s = K_m = Σ_always |clsNeeds u| + K₀ · Σ_t k_t · M_t + Pr[refused] · Σ_{u<N} |clsNeeds u|`,

with `K₀` the first stage's count (`hK`), `k_t` the second stage's draws of template `t`, `M_t` a cap on the classes
a block of template `t` needs (`hM`), and `Pr[refused]` the probability that the second stage's selection refuses,
where every block below `N` is checked. A class's representative represents itself (`Compiled.clsOf_idem`), which
`code_hKm_cls` asks of the classes.

What the theorem still takes: the caps `M_t` and the first stage's count `K₀`, read off the registration, the budget
`T` a run, and the linear expected-time bound at `B_conf,X` and at the two by-class finders.
`code_eventTwoX_le_cls_at_cap` is the case of one cap `M` for every template.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The by-class first-hit bound at a budget, at the code, at caps by template**: `code_eventTwoX_le_cls_at_budget`
with its three counts in closed form. Under `Built` and `Addresses` at a statement in the two-stage scope, where every
run calls SHA-512 at most `T` times, both parties' calls counted (`hT`), every block of template `t` the second stage
can draw needs at most `M t` classes (`hM`), the first stage's count is `K₀` (`hK`), and `B_conf,X` and the two
by-class finders of every cap meet the linear expected-time bound, an accepted run samples a proof block of
`FregTwoX` with probability at most `εclsAt` at `N_s = |always| + K₀ · Σ_t k_t + Pr[refused] · N` and
`K_s = K_m = Σ_always |clsNeeds u| + K₀ · Σ_t k_t · M t + Pr[refused] · Σ_{u<N} |clsNeeds u|`. -/
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

/-- **The by-class first-hit bound at a budget, at the code, at one cap**: `code_eventTwoX_le_cls_at_caps` where every
block the second stage can draw needs at most `M` classes, whatever its template. -/
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
