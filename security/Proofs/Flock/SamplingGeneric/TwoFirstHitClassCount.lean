module

public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
import Proofs.Flock.Claims.Prob

/-!
# The by-class first-hit bound's two counts, as counts of gates

`CompiledTwo.eventTwoX_le_cls_at_budget` reads two expectations over the coins by class of gates: the classes of the
gates a run's sample needs (`K_s`) and the classes' representatives its sessions need, session by session (`K_m`,
`Compiled.sessCountIn`). Both are the by-gate counts (`Compiled.sampleNeeds`, `Compiled.sessCount`) of other needs:
each proof block needing, in place of its gates, the representatives of their classes (`Compiled.clsNeeds`).

* The classes of the gates a sample needs are among the gates it needs under `clsNeeds`
  (`Compiled.image_sampleNeeds_subset`), and where a representative represents itself, the representatives a session
  needs are among the gates it needs under `clsNeeds` (`Compiled.sessCountIn_cls_le`).
* So a bound on the two by-gate counts at `clsNeeds` is a bound on the two by-class counts (`Compiled.ex_cls_le`,
  `Compiled.ex_sessCountIn_cls_le`), and a proof block counts for the classes among its gates
  (`Compiled.card_clsNeeds`), never more than its gates (`Compiled.card_clsNeeds_le`).
* The least gate with a gate's key represents itself (`Compiled.clsOf_idem`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The classes of the gates a proof block needs**, each by its representative's position: the needs of proof
block `u` with every gate replaced by its class's representative. -/
def clsNeeds (N : ℕ) (cls : Fin N → Fin N) (needs : ℕ → Finset ℕ) (u : ℕ) : Finset ℕ :=
  ((Finset.univ.filter fun g : Fin N => g.val ∈ needs u).image cls).image Fin.val

/-- The representative of a gate a proof block needs is among the block's needs by class. -/
theorem mem_clsNeeds {N : ℕ} {cls : Fin N → Fin N} {needs : ℕ → Finset ℕ} {u : ℕ} {g : Fin N}
    (hg : g.val ∈ needs u) : (cls g).val ∈ clsNeeds N cls needs u :=
  Finset.mem_image.2 ⟨cls g, Finset.mem_image.2 ⟨g, Finset.mem_filter.2 ⟨Finset.mem_univ _, hg⟩, rfl⟩, rfl⟩

/-- A proof block's needs by class are as many as the classes among the gates it needs. -/
theorem card_clsNeeds {N : ℕ} (cls : Fin N → Fin N) (needs : ℕ → Finset ℕ) (u : ℕ) :
    (clsNeeds N cls needs u).card = ((Finset.univ.filter fun g : Fin N => g.val ∈ needs u).image cls).card :=
  Finset.card_image_of_injective _ Fin.val_injective

/-- A proof block's needs by class are at most the gates it needs. -/
theorem card_clsNeeds_le {N : ℕ} (cls : Fin N → Fin N) (needs : ℕ → Finset ℕ) (u : ℕ) :
    (clsNeeds N cls needs u).card ≤ (needs u).card := by
  rw [card_clsNeeds]
  refine Finset.card_image_le.trans ?_
  refine Finset.card_le_card_of_injOn Fin.val (fun g hg => (Finset.mem_filter.1 hg).2) ?_
  exact fun a _ b _ hab => Fin.val_injective hab

/-- **The classes of the gates a sample needs are among the gates it needs by class.** -/
theorem image_sampleNeeds_subset (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) (l : List ℕ) :
    (L.sampleNeeds x needs l).image cls ⊆ L.sampleNeeds x (clsNeeds (L.nG x) cls needs) l := by
  intro q hq
  obtain ⟨g, hg, rfl⟩ := Finset.mem_image.1 hq
  obtain ⟨u, hu, hgu⟩ := of_decide_eq_true (Finset.mem_filter.1 hg).2
  exact Finset.mem_filter.2 ⟨Finset.mem_univ _, decide_eq_true ⟨u, hu, mem_clsNeeds hgu⟩⟩

/-- **The classes' representatives a sample's sessions need, session by session, are at most the gates they need by
class**, where a representative represents itself (`hidem`). -/
theorem sessCountIn_cls_le (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) (hidem : ∀ g, cls (cls g) = cls g) (l : List ℕ) :
    L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) l ≤
      L.sessCount x (clsNeeds (L.nG x) cls needs) l := by
  unfold sessCountIn sessCount sessNeeds
  refine Finset.sum_le_sum fun i _ => Finset.card_le_card fun q hq => ?_
  obtain ⟨hq1, hq2⟩ := Finset.mem_filter.1 hq
  obtain ⟨g, -, rfl⟩ := Finset.mem_image.1 hq1
  obtain ⟨u, hu, hgu⟩ := of_decide_eq_true hq2
  refine Finset.mem_filter.2 ⟨Finset.mem_univ _, decide_eq_true ⟨u, hu, ?_⟩⟩
  have h := mem_clsNeeds (cls := cls) hgu
  rwa [hidem] at h

/-- **`K_s` by class from a count of gates**: a bound on the expected number of gates a run's sample needs by class
bounds the expected number of classes of the gates it needs. -/
theorem ex_cls_le (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ) (cls : Fin (L.nG x) → Fin (L.nG x))
    {Ks : ℝ}
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      ((L.sampleNeeds x (clsNeeds (L.nG x) cls needs) (L.checked x r)).card : ℝ)) ≤ Ks) :
    Ex (fun r : Fin (L.V.k x) → Coin => (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks :=
  (Ex_mono fun r => Nat.cast_le.2
    (Finset.card_le_card (L.image_sampleNeeds_subset x needs cls (L.checked x r)))).trans hKs

/-- **`K_m` by class from a count of gates**: where a representative represents itself, a bound on the expected
number of gates a run's sessions need by class, session by session, bounds the expected number of the classes'
representatives they need. -/
theorem ex_sessCountIn_cls_le (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) (hidem : ∀ g, cls (cls g) = cls g) {Km : ℝ}
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.sessCount x (clsNeeds (L.nG x) cls needs) (L.checked x r) : ℝ)) ≤ Km) :
    Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ)) ≤ Km :=
  (Ex_mono fun r => Nat.cast_le.2 (L.sessCountIn_cls_le x needs cls hidem (L.checked x r))).trans hKm

/-- The least gate with a gate's key represents itself. -/
theorem clsOf_idem {N : ℕ} {Key : Type} [DecidableEq Key] (key : Fin N → Key) (g : Fin N) :
    Verity.FirstHit.clsOf key (Verity.FirstHit.clsOf key g) = Verity.FirstHit.clsOf key g :=
  (Verity.FirstHit.clsOf_eq_iff key _ g).2 (Verity.FirstHit.key_clsOf key g)

end Compiled

end Claims.SamplingGeneric
