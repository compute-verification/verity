import Properties.Verity.Soundness
import Proofs.Properties.Composition
import Proofs.Core.Game.Prob

/-!
# Soundness at the verifier of record, from its subprotocols' targets

Onsite's acceptance is NCI's on the registration sampled proofs committed to, and every component's on its own case
(`accepts_registered`, `accepts_components`, from `Properties.Composition`'s `answerAccepts` at onsite and
`acceptRegistration`). So the event `Verity.Soundness` bounds lies inside the union of the subprotocols' events, each at
most its target's error at the strategy (`prob_exists_finset_le`).

Beside it, the two checks every guarantee carries: `soundness_of_subprotocols_honest`, its hypotheses hold wherever
each claim holds on every play, so none is false by its form; and `soundness_of_subprotocols_nontrivial`, its
conclusion fails for a verifier that accepts everything, wherever a claim fails on every play of a strategy whose files
pass the commit phase's check and the errors sum below one. `Vocab.holds_sampled_of_opens` is the honest commitment's
case of sampled proofs' claim: a correct transcript that the commitment opens everywhere, the constants by their
constants, has no unbound unit.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

theorem component_mem (a : Audit) : component a ∈ Onsite.components := by
  cases a <;> simp [component, Onsite.components]

/-- **What the verifier of record accepts, NCI accepted on the registration sampled proofs committed to.** -/
theorem accepts_registered {s : Sent} (h : Accepts s) : RegistrationAccepts s := by
  obtain ⟨⟨n, ps⟩, ⟨b, d⟩, ⟨k, ev⟩, hp, hc, he, pub, prof, hch⟩ := Properties.Composition.answerAccepts _ _ _ _ h
  exact ⟨n, ps, b, d, k, ev, ⟨hp, hc, he⟩, pub, prof,
    Properties.Composition.acceptRegistration "verity/onsite/v0" NciCheck.Protocol.protocol
      SampledProofs.OneStage.protocol.id Onsite.components n ps b d k ev pub prof hch⟩

/-- **What the verifier of record accepts, every component accepts on its own case.** -/
theorem accepts_components {s : Sent} (h : Accepts s) (a : Audit) : ComponentAccepts a s := by
  obtain ⟨n, ps, b, d, k, ev, hdec, pub, prof, hreg⟩ := accepts_registered h
  obtain ⟨-, -, -, -, -, -, ⟨-, -, hfst, -⟩, hcases, -⟩ := hreg
  have hm : component a ∈ (Properties.Composition.triples Onsite.components ps ev).map (·.1) := by
    rw [hfst]
    exact component_mem a
  obtain ⟨x, hx, hx1⟩ := List.mem_map.mp hm
  exact ⟨n, ps, b, d, k, ev, hdec, x, hx, hx1, hcases x hx⟩

/-- The union bound over a finite family of events, for one strategy. -/
theorem prob_exists_finset_le {β κ : Type} (S : Finset κ) (p : κ → β → Prop) (g : Game β)
    (σ : Game.Strategy g) : Game.prob (fun b => ∃ i ∈ S, p i b) g σ ≤ ∑ i ∈ S, Game.prob (p i) g σ := by
  classical
  induction S using Finset.induction_on with
  | empty => simpa using (Game.prob_false g σ).le
  | insert j S hj ih =>
      rw [Finset.sum_insert hj]
      calc Game.prob (fun b => ∃ i ∈ insert j S, p i b) g σ
          ≤ Game.prob (fun b => p j b ∨ ∃ i ∈ S, p i b) g σ :=
            Game.prob_mono (fun b ⟨i, hi, hpi⟩ => by
              rcases Finset.mem_insert.1 hi with rfl | hi
              · exact Or.inl hpi
              · exact Or.inr ⟨i, hi, hpi⟩) g σ
        _ ≤ Game.prob (p j) g σ + Game.prob (fun b => ∃ i ∈ S, p i b) g σ := Game.prob_or_le _ _ g σ
        _ ≤ Game.prob (p j) g σ + ∑ i ∈ S, Game.prob (p i) g σ := add_le_add le_rfl ih

theorem soundness_of_subprotocols : Soundness := by
  intro ι α _ d V hS hW hP hB hN θ F c σ
  have hT : ∀ a, V.Target a := by
    intro a
    cases a with
    | sampled => exact hS
    | pouw => exact hW
    | pous => exact hP
    | poub => exact hB
    | network => exact hN
  calc Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧ ∃ a, ¬ V.Holds θ F c σ a o)
          (V.game θ F c) σ
      ≤ Game.prob (fun o => ∃ a ∈ Finset.univ, V.Verdict a o ∧ ¬ V.Holds θ F c σ a o) (V.game θ F c) σ :=
        Game.prob_mono (fun o ⟨hacc, hreg, a, hfail⟩ =>
          ⟨a, Finset.mem_univ a, ⟨hreg, accepts_registered hacc, accepts_components hacc a⟩, hfail⟩) _ σ
    _ ≤ ∑ a, Game.prob (fun o => V.Verdict a o ∧ ¬ V.Holds θ F c σ a o) (V.game θ F c) σ :=
        prob_exists_finset_le _ _ _ σ
    _ ≤ ∑ a, V.error a θ F c σ := Finset.sum_le_sum fun a _ => hT a θ F c σ

/-- An event every play after `θ`, `F` and `c` meets has probability one. -/
theorem prob_play_eq_one (ns : List ℕ) (θ F c : ByteArray) (p : Outcome → Prop)
    (hp : ∀ log e, p ⟨θ, F, c, log, e⟩) (σ : Game.Strategy (play ns θ F c)) : Game.prob p (play ns θ F c) σ = 1 := by
  refine (Game.prob_bind_of_const (fun log => Game.send ByteArray fun e => Game.ret (⟨θ, F, c, log, e⟩ : Outcome)) p
    (fun _ => True) (fun log t => ?_) (releases ns) σ).trans (Game.prob_true _ _)
  rw [Game.prob_true]
  simp only [Game.prob]
  rw [if_pos (hp log t.1)]

/-- **Its hypotheses hold where every claim holds on every play**: each subprotocol's target, at any errors. -/
theorem soundness_of_subprotocols_honest {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (h : ∀ a θ F c σ o, V.Holds θ F c σ a o) :
    V.SampledProofsSound ∧ V.PouwSound ∧ V.PousSound ∧ V.BandwidthAuditSound ∧ V.NetworkCertifierSound := by
  have hT : ∀ a, V.Target a := fun a θ F c σ =>
    (Game.prob_mono (p := fun o => V.Verdict a o ∧ ¬ V.Holds θ F c σ a o) (q := fun _ => False)
      (fun o ho => ho.2 (h a θ F c σ o)) _ σ).trans
      ((Game.prob_false _ σ).le.trans (zero_le _))
  exact ⟨hT _, hT _, hT _, hT _, hT _⟩

/-- **Its conclusion fails for a verifier that accepts everything**: at files that pass the commit phase's check, a
strategy on whose every play some claim fails, and errors summing below one. -/
theorem soundness_of_subprotocols_nontrivial {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (hreg : V.Registered θ F) (a : Audit)
    (hfail : ∀ o, ¬ V.Holds θ F c σ a o) (hlt : ∑ a, V.error a θ F c σ < 1) :
    ¬ V.SoundnessFor fun _ => True := by
  intro h
  have h1 := prob_play_eq_one (V.rounds θ) θ F c
    (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.file ∧ ∃ a, ¬ V.Holds θ F c σ a o)
    (fun _ _ => ⟨trivial, hreg, a, hfail _⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

/-- **The honest commitment's case of sampled proofs' claim**: when the extractor's transcript is correct on every unit
and the commitment opens every gate to it (a registered constant to its constant), no unit is unbound, and the claim
holds at any `ε ≥ 0`. -/
theorem holds_sampled_of_opens (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (o : Outcome)
    (hc : ∀ u, (V.partsOf F).Correct (V.extract θ F c σ) u) (ho : ∀ g, V.Opens θ F c g (V.extract θ F c σ g))
    (hε : 0 ≤ (V.certOf θ).ε) : V.Holds θ F c σ .sampled o := by
  have hu : V.unbound θ F c (V.extract θ F c σ) = ∅ := by
    classical
    refine Finset.filter_eq_empty_iff.2 fun u _ => ?_
    rintro (hw | ⟨g, -, hg⟩)
    · exact hw (hc u)
    · exact hg (ho g)
  show ((V.unbound θ F c (V.extract θ F c σ)).card : ℝ) ≤ (V.certOf θ).ε * V.units F
  rw [hu, Finset.card_empty, Nat.cast_zero]
  exact mul_nonneg hε (Nat.cast_nonneg _)

/-- A registered constant opens to exactly the values holding its constant. -/
theorem opens_const {θ F c : ByteArray} {g : Fin (V.circuitOf F).N} {b : Bool} (hg : (V.read F).consts g = some b)
    (v : BitVec ((V.circuitOf F).w g)) :
    V.Opens θ F c g v ↔ ∀ i : Fin ((V.circuitOf F).w g), v.getLsbD i.val = b := by
  unfold Opens
  rw [hg]

/-- **The file finder outputs a collision** when the committed blob is not the registered one and has its SHA-512. -/
theorem fileFinder_collides {θ F : ByteArray} (hne : F ≠ V.registered θ)
    (hd : sha512 F.data.toList = sha512 (V.registered θ).data.toList) : (V.fileFinder θ F).collides = 1 := by
  have hl : F.data.toList ≠ (V.registered θ).data.toList := fun e =>
    hne (ByteArray.ext (Array.ext' e))
  unfold Finder.collides fileFinder
  simp only [Game.prob, if_neg hne]
  rw [if_pos ⟨_, _, rfl, hl, hd⟩]

end Vocab

end Verity
