module

public import Proofs.Flock.SpotFlock.Counts
public import Proofs.Core.SamplingGeneric.FirstHitClass

/-!
# The first-hit bound's counts by class at SPOT[Flock]'s code

`CompiledTwo.eventTwoX_le_cls_at_budget` reads two of its three counts by class: under `cls`, the classes of the gates
a run's sample needs (`K_s`) and the classes of the sample's gates that its sessions need, session by session (`K_m`).
Each is at most a weight summed over the sample's proof blocks: the classes of the gates a block needs (`clsCount`).

* `Compiled.card_image_sampleNeeds_le`: `K_s` on a sample `l` is at most `Σ_{u ∈ l} clsCount u`, for any `cls`.
* `Compiled.sessCountIn_image_le`: so is `K_m`, for a `cls` that keeps every class's representative
  (`cls (cls g) = cls g`, which `clsOf` of any key does: `clsOf_clsOf`).

Both are `Compiled.card_filter_needs_le` and `Compiled.sum_first_sessNeeds_le` at the class needs, where each block
needs the representatives of its gates' classes. At SPOT[Flock]'s code, `code_ex_le` with one cap `M` on every block's
`clsCount` turns them into the bound's `hKs` and `hKm` (`code_hKs_cls`, `code_hKm_cls`): each at most
`Σ_always clsCount u + K · Σ_t k_t · M + Pr[refused] · Σ_{u<N} clsCount u`. The third count, `N_s`, is the same by
class as by gates, `code_hNs`.
-/

@[expose] public section

namespace Verity.FirstHit

/-- A class's representative is its own class's representative. -/
theorem clsOf_clsOf {n : ℕ} {Key : Type} [DecidableEq Key] (key : Fin n → Key) (g : Fin n) :
    clsOf key (clsOf key g) = clsOf key g :=
  (clsOf_eq_iff key _ _).2 (key_clsOf key g)

end Verity.FirstHit

namespace Claims.SamplingGeneric

/-- **The classes a proof block needs**: under `cls`, the classes of the gates below `N` that block `u` needs. -/
def clsCount {N : ℕ} (cls : Fin N → Fin N) (needs : ℕ → Finset ℕ) (u : ℕ) : ℕ :=
  ((Finset.univ.filter fun g : Fin N => g.val ∈ needs u).image cls).card

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`K_s` by class on a sample**: the classes of the gates the sample `l` needs are at most the classes each of its
blocks needs, summed over the sample. -/
theorem card_image_sampleNeeds_le (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) (l : List ℕ) :
    ((L.sampleNeeds x needs l).image cls).card ≤ (l.map (clsCount cls needs)).sum := by
  have hsub : (L.sampleNeeds x needs l).image cls ⊆
      Finset.univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ l,
        g.val ∈ ((Finset.univ.filter fun g' : Fin (L.nG x) => g'.val ∈ needs u).image cls).map Fin.valEmbedding) := by
    intro c hc
    obtain ⟨g, hg, rfl⟩ := Finset.mem_image.1 hc
    obtain ⟨u, hu, hgu⟩ := of_decide_eq_true (Finset.mem_filter.1 hg).2
    exact Finset.mem_filter.2 ⟨Finset.mem_univ _, decide_eq_true ⟨u, hu, Finset.mem_map.2
      ⟨cls g, Finset.mem_image.2 ⟨g, Finset.mem_filter.2 ⟨Finset.mem_univ _, hgu⟩, rfl⟩, rfl⟩⟩⟩
  have h := card_filter_needs_le l (fun u => ((Finset.univ.filter fun g : Fin (L.nG x) => g.val ∈ needs u).image
    cls).map Fin.valEmbedding) (L.nG x)
  simp only [Finset.card_map] at h
  exact (Finset.card_le_card hsub).trans h

/-- **`K_m` by class on a sample**: with `cls` keeping every class's representative, the classes of the sample's gates
that its sessions need, session by session, are at most the classes each of its blocks needs, summed over the sample. -/
theorem sessCountIn_image_le (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) (hcls : ∀ g, cls (cls g) = cls g) (l : List ℕ) :
    L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) l ≤ (l.map (clsCount cls needs)).sum := by
  have hsub : ∀ i : Fin (L.nPU x),
      ((L.sampleNeeds x needs l).image cls).filter (fun g : Fin (L.nG x) =>
          decide (∃ u ∈ L.sessBlocks x l i.val, g.val ∈ needs u)) ⊆
        Finset.univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.sessBlocks x l i.val,
          g.val ∈ ((Finset.univ.filter fun g' : Fin (L.nG x) => g'.val ∈ needs u).image cls).map Fin.valEmbedding) := by
    intro i c hc
    obtain ⟨hc1, hc2⟩ := Finset.mem_filter.1 hc
    obtain ⟨g, -, rfl⟩ := Finset.mem_image.1 hc1
    obtain ⟨u, hu, hgu⟩ := of_decide_eq_true hc2
    exact Finset.mem_filter.2 ⟨Finset.mem_univ _, decide_eq_true ⟨u, hu, Finset.mem_map.2
      ⟨cls g, Finset.mem_image.2 ⟨cls g, Finset.mem_filter.2 ⟨Finset.mem_univ _, hgu⟩, hcls g⟩, rfl⟩⟩⟩
  have h := L.sum_first_sessNeeds_le x l (fun u => ((Finset.univ.filter fun g : Fin (L.nG x) => g.val ∈ needs u).image
    cls).map Fin.valEmbedding) (L.nPU x) (L.nG x)
  simp only [Finset.card_map] at h
  unfold sessCountIn firstSlots
  exact (Finset.sum_le_sum fun i _ => Finset.card_le_card (hsub i)).trans h

end Compiled

end Claims.SamplingGeneric

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)

/-- **`K_s` by class at SPOT[Flock]'s code**: with every block needing at most `M` classes under `cls`, the classes of
the gates a run's sample needs average at most the always-checked blocks' classes below `N` plus `K · Σ_t k_t · M`,
plus the refusal's probability times every block's classes below `N`. -/
theorem code_hKs_cls (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) →
      Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x))
    (M : ℕ) (hM : ∀ u, Claims.SamplingGeneric.clsCount cls needs u ≤ M) {K : ℕ}
    (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      ((((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r)).image cls).card : ℝ)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            (Claims.SamplingGeneric.clsCount cls needs)).sum + K * h.reg.ks.toList.sum * M : ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (src (r (order h x).second))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            (Claims.SamplingGeneric.clsCount cls needs)).sum : ℝ) :=
  code_ex_le h x (fun l => ((((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).image cls).card : ℝ))
    (Claims.SamplingGeneric.clsCount cls needs) M hM
    (fun l => by
      have hc := (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.card_image_sampleNeeds_le x needs
        cls (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
      exact_mod_cast hc.trans (sum_map_take_le l _ (Claims.SamplingGeneric.clsCount cls needs))) hK

/-- **`K_m` by class at SPOT[Flock]'s code**: with `cls` keeping every class's representative and every block needing
at most `M` classes under it, the classes of a run's sample's gates that its sessions need, session by session,
average at most the always-checked blocks' classes below `N` plus `K · Σ_t k_t · M`, plus the refusal's probability
times every block's classes below `N`. -/
theorem code_hKm_cls (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) →
      Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x))
    (hcls : ∀ g, cls (cls g) = cls g) (M : ℕ) (hM : ∀ u, Claims.SamplingGeneric.clsCount cls needs u ≤ M) {K : ℕ}
    (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn x needs
        (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs l).image cls)
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r) : ℝ)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            (Claims.SamplingGeneric.clsCount cls needs)).sum + K * h.reg.ks.toList.sum * M : ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (src (r (order h x).second))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            (Claims.SamplingGeneric.clsCount cls needs)).sum : ℝ) :=
  code_ex_le h x (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn x needs
      (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs l).image cls)
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ))
    (Claims.SamplingGeneric.clsCount cls needs) M hM
    (fun l => by
      have hc := (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn_image_le x needs cls
        hcls (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
      exact_mod_cast hc.trans (sum_map_take_le l _ (Claims.SamplingGeneric.clsCount cls needs))) hK

end SpotFlock
