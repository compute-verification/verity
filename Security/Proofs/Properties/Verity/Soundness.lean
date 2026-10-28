import Properties.Verity.Soundness
import Proofs.Properties.Composition
import Proofs.Core.Game.Prob

/-!
# Soundness at the verifier of record, from its subprotocols' targets

Onsite's acceptance is NCI's on the registration sampled proofs committed to, and every component's on its own case
(`accepts_registered`, `accepts_components`, from `Properties.Composition`'s `answerAccepts` at onsite and
`acceptRegistration`). So the event `Verity.Soundness` bounds lies inside the union of the subprotocols' events, each at
most its target's error at the strategy (`prob_exists_finset_le`).

Beside it, the two checks every guarantee carries: `soundness_of_subprotocols_honest`, at the honest deployment (the
deployment's honest prover at registered files, each subprotocol's honest lemma `Vocab.HonestHolds`) every target's
event and the theorem's own have probability `0`; and `soundness_of_subprotocols_nontrivial`, its
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
theorem prob_play_eq_one (ns : List ℕ) (θ F c : ByteArray) (p : Ending → Prop)
    (hp : ∀ log e, p ⟨θ, F, c, log, e⟩) (σ : Game.Strategy (play ns θ F c)) : Game.prob p (play ns θ F c) σ = 1 := by
  refine (Game.prob_bind_of_const (fun log => Game.send ByteArray fun e => Game.ret (⟨θ, F, c, log, e⟩ : Ending)) p
    (fun _ => True) (fun log t => ?_) (releases ns) σ).trans (Game.prob_true _ _)
  simp [Game.prob, hp]

/-- **Its hypotheses hold at the honest deployment**: at registered files, where each subprotocol's honest lemma holds
(`Vocab.HonestHolds`), the deployment's honest prover meets every target with probability `0` on its event, and the
event the theorem bounds has probability `0`. -/
theorem soundness_of_subprotocols_honest {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (h : ∀ a, V.HonestHolds a) (θ F c : ByteArray) (hreg : V.Registered θ F) :
    (∀ a, Game.prob (fun o => V.Verdict a o ∧ ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c)
        ≤ V.error a θ F c (V.honest θ F c)) ∧
      Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧ ∃ a, ¬ V.Holds θ F c (V.honest θ F c) a o)
        (V.game θ F c) (V.honest θ F c) = 0 := by
  have h0 : ∀ a, Game.prob (fun o => ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) = 0 :=
    fun a => h a θ F c hreg
  refine ⟨fun a => ?_, le_antisymm ?_ zero_le⟩
  · refine le_trans (Game.prob_mono (p := fun o => V.Verdict a o ∧ ¬ V.Holds θ F c (V.honest θ F c) a o)
      (q := fun o => ¬ V.Holds θ F c (V.honest θ F c) a o) (fun o ho => ho.2) _ _) ?_
    rw [h0 a]
    exact zero_le
  · calc Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧
            ∃ a, ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c)
        ≤ Game.prob (fun o => ∃ a ∈ Finset.univ, ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c)
            (V.honest θ F c) :=
          Game.prob_mono (fun o ⟨_, _, a, ha⟩ => ⟨a, Finset.mem_univ a, ha⟩) _ _
      _ ≤ ∑ a, Game.prob (fun o => ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) :=
          prob_exists_finset_le _ _ _ _
      _ = 0 := by simp only [h0, Finset.sum_const_zero]

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
theorem holds_sampled_of_opens (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (o : Ending)
    (hc : ∀ u, (V.partsOf F).Correct (V.extract θ F c σ) u) (ho : ∀ g, V.Opens θ F c g (V.extract θ F c σ g))
    (hε : 0 ≤ (V.certOf θ).ε) : V.Holds θ F c σ .sampled o := by
  have hu : Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ) = ∅ := by
    classical
    refine Finset.filter_eq_empty_iff.2 fun u _ => ?_
    rintro (hw | ⟨g, -, hg⟩)
    · exact hw (hc u)
    · exact hg (ho g)
  show ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
    (V.certOf θ).ε * V.units F
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
    (hd : sha512Msg F.data.toList = sha512Msg (V.registered θ).data.toList) :
    (V.fileFinder θ F).Pr sha512Msg = 1 := by
  have hl : F.data.toList ≠ (V.registered θ).data.toList := fun e =>
    hne (ByteArray.ext (Array.ext' e))
  have hs : ∀ s : (V.fileFinder θ F).S,
      ∃ u v, (V.fileFinder θ F).out s = some (u, v) ∧ Claims.Collision sha512Msg u v :=
    fun _ => ⟨_, _, by simp only [fileFinder, if_neg hne], hl, hd⟩
  classical
  unfold Claims.Finder.Pr Claims.Pr
  rw [Finset.filter_true_of_mem fun s _ => hs s, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)

/-- **The file finder's SHA-512 calls**: two on its one coin, at most `0 · calls(τ) + 2`. -/
theorem fileFinder_calls (θ F : ByteArray) (x : (V.fileFinder θ F).S) (n : ℕ) :
    (V.fileFinder θ F).cost x ≤ 0 * n + 2 := by
  simp [fileFinder]

end Vocab

end Verity
