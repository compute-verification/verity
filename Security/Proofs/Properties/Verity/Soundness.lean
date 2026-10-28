import Properties.Verity.Soundness
import Properties.Verity.Idle
import Proofs.Properties.Composition
import Proofs.Core.Game.Prob

/-!
# Soundness at the verifier of record, from its subprotocols' targets

A composite's acceptance is its acceptance as a `same` composition and every component's on its own case
(`accepts_same`, `accepts_components`, from `Properties.Composition`'s `answerAccepts` at the composite, `acceptTied` at
onsite's ties and `acceptSame`). So the event `Verity.SoundnessOf` bounds at a composite lies inside the union of its
subprotocols' events, each at most its target's error at the strategy: the union bound over the composite's
subprotocols in one game (`soundness_at_of_targets`). `soundness_of_legs` is it at onsite's seven, and
`remote_soundness_of_legs` at remote's three, which are a subset of onsite's.

Beside it, the two checks every guarantee carries, for any names `V`: `soundness_honest_of`, at the honest deployment
(the deployment's honest prover at registered files, each subprotocol's honest lemma `Vocab.HonestHolds`) every target's
event and the theorem's own have probability `0`; and `soundness_nontrivial_of`, its conclusion fails for a verifier
that accepts everything, wherever a claim fails on every play of a strategy whose files pass the commit phase's check and
the errors sum below one. `Vocab.holds_sampled_of_opens` is the honest commitment's case of sampled proofs' claim: a
correct transcript that the commitment opens everywhere, the constants by their constants, has no unbound unit.

The witnesses of each of the three theorems are at the idle deployment (`Properties.Verity.Idle`), at the compiled
composites: its hypotheses hold at `Idle.unbounded` (`soundness_at_of_targets_honest`, `soundness_of_legs_honest`,
`remote_soundness_of_legs_honest`), and its conclusion fails at `Idle.exact` for a verifier that accepts everything
(`soundness_at_of_targets_nontrivial`, `soundness_of_legs_nontrivial`, `remote_soundness_of_legs_nontrivial`): there
every error is `0` and PoUW's claim fails on every play of the files that pass the commit phase's check.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

theorem component_mem {k : Verity.Registration.Composite} {a : Subprotocol} {P : BlobProtocol}
    (h : component k a = some P) : P ∈ componentsOf k := by
  cases k <;> cases a <;> simp [component] at h <;> subst h <;>
    simp [componentsOf, Onsite.components, Remote.components]

/-- **Each composite's verifier of record is the one the composites' table lists for it**, with its subprotocols'
ids. -/
theorem verifier_mem (k : Verity.Registration.Composite) :
    (verifier k, (componentsOf k).map (·.id)) ∈ Composites.composites := by
  cases k
  · exact List.mem_cons.2 (Or.inl rfl)
  · exact List.mem_cons.2 (Or.inr (List.mem_cons.2 (Or.inl rfl)))

/-- **What a composite's verifier of record accepts, it accepted as a `same` composition**: an accepted composition of
its components whose record subprotocols committed to one record, and at onsite PoUS and PoUB-P2 on one store. -/
theorem accepts_same {k : Verity.Registration.Composite} {s : Sent} (h : Accepts k s) : SameAccepts k s := by
  cases k
  · obtain ⟨ps, c, ev, hp, hc, he, pub, prof, hch⟩ :=
      Properties.Composition.answerAccepts (Onsite.protocol.on s.blobs) s.params s.commitment s.evidence h
    obtain ⟨hh, hties⟩ := Properties.Composition.acceptTied [Onsite.store] ev
      (Composition.checkSame "verity/onsite/v0" Onsite.tied (s.components .onsite) ps c ev) pub prof hch
    exact ⟨ps, c, ev, ⟨hp, hc, he⟩, pub, prof,
      Properties.Composition.acceptSame _ _ _ _ _ _ _ _ hh, hties⟩
  · obtain ⟨ps, c, ev, hp, hc, he, pub, prof, hch⟩ :=
      Properties.Composition.answerAccepts (Remote.protocol.on s.blobs) s.params s.commitment s.evidence h
    have hch' : Composition.checkSame "verity/remote/v0" (tied .remote) (s.components .remote) ps c ev =
        .accept pub prof := hch
    exact ⟨ps, c, ev, ⟨hp, hc, he⟩, pub, prof,
      Properties.Composition.acceptSame _ _ _ _ _ _ _ _ hch', fun t ht => absurd ht List.not_mem_nil⟩

/-- **What a composite's verifier of record accepts, each of its subprotocols' components accepts on its own
case.** -/
theorem accepts_components {k : Verity.Registration.Composite} {s : Sent} (h : Accepts k s) {a : Subprotocol}
    (ha : a ∈ subprotocols k) : ComponentAccepts k a s := by
  obtain ⟨ps, c, ev, hdec, pub, prof, ⟨⟨⟨-, -, hfst, -⟩, hcases, -⟩, -⟩, -⟩ := accepts_same h
  obtain ⟨P, hP⟩ : ∃ P, component k a = some P := Option.isSome_iff_exists.1 (Finset.mem_filter.1 ha).2
  have hm : P.onBlobs (Blobs.ofArray s.blobs) ∈ (Properties.Composition.triples (s.components k) ps ev).map (·.1) := by
    rw [hfst]
    exact List.mem_map.2 ⟨P, component_mem hP, rfl⟩
  obtain ⟨x, hx, hx1⟩ := List.mem_map.mp hm
  exact ⟨P, hP, ps, c, ev, hdec, x, hx, hx1, hcases x hx⟩

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

/-- **Soundness at every composite, from its subprotocols' targets**: the union bound over the composite's
subprotocols, each event inside its target's. -/
theorem soundness_at_of_targets : SoundnessOf := by
  intro ι α _ d V k hT θ F c σ
  calc Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.file ∧
          ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ
      ≤ Game.prob (fun o => ∃ a ∈ subprotocols k, V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ :=
        Game.prob_mono (fun o ⟨hacc, hreg, a, ha, hfail⟩ =>
          ⟨a, ha, ⟨hreg, accepts_same hacc, accepts_components hacc ha⟩, hfail⟩) _ σ
    _ ≤ ∑ a ∈ subprotocols k,
          Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ :=
        prob_exists_finset_le _ _ _ _
    _ ≤ ∑ a ∈ subprotocols k, V.error a θ F c σ := Finset.sum_le_sum fun a ha => hT a ha θ F c σ

theorem soundness_of_legs : Soundness := by
  intro ι α _ d V hS hW hI hP hB hN hX
  refine soundness_at_of_targets ι α d V .onsite fun a _ => ?_
  cases a
  exacts [hS, hW, hI, hP, hB, hN, hX]

theorem remote_soundness_of_legs : RemoteSoundness := by
  intro ι α _ d V hS hW hI
  refine soundness_at_of_targets ι α d V .remote fun a ha => ?_
  cases a
  · exact hS
  · exact hW
  · exact hI
  all_goals exact absurd (Finset.mem_filter.1 ha).2 Bool.false_ne_true

/-- An event every play after `θ`, `F` and `c` meets has probability one. -/
theorem prob_play_eq_one (ns : List ℕ) (θ F c : ByteArray) (p : Exchange → Prop)
    (hp : ∀ log e, p ⟨θ, F, c, log, e⟩) (σ : Game.Strategy (play ns θ F c)) : Game.prob p (play ns θ F c) σ = 1 := by
  refine (Game.prob_bind_of_const (fun log => Game.send ByteArray fun e => Game.ret (⟨θ, F, c, log, e⟩ : Exchange)) p
    (fun _ => True) (fun log t => ?_) (releases ns) σ).trans (Game.prob_true _ _)
  simp [Game.prob, hp]

/-- **Its hypotheses hold at the honest deployment**: at registered files, where each of composite `k`'s subprotocols'
honest lemma holds (`Vocab.HonestHolds`), the deployment's honest prover meets every target with probability `0` on its
event, and the event the theorem bounds has probability `0`. -/
theorem soundness_honest_of {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (k : Verity.Registration.Composite) (h : ∀ a ∈ subprotocols k, V.HonestHolds k a) (θ F c : ByteArray)
    (hreg : V.Registered θ F) :
    (∀ a ∈ subprotocols k, Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error a θ F c (V.honest θ F c)) ∧
      Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.file ∧
          ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c (V.honest θ F c) a o)
        (V.game θ F c) (V.honest θ F c) = 0 := by
  have h0 : ∀ a ∈ subprotocols k,
      Game.prob (fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) = 0 :=
    fun a ha => h a ha θ F c hreg
  refine ⟨fun a ha => ?_, le_antisymm ?_ zero_le⟩
  · refine le_trans (Game.prob_mono (p := fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
      (q := fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (fun o ho => ho.2) _ _) ?_
    rw [h0 a ha]
    exact zero_le
  · calc Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.file ∧
            ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c)
        ≤ Game.prob (fun o => ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c)
            (V.honest θ F c) :=
          Game.prob_mono (fun o ⟨_, _, ha⟩ => ha) _ _
      _ ≤ ∑ a ∈ subprotocols k,
            Game.prob (fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) :=
          prob_exists_finset_le _ _ _ _
      _ = 0 := Finset.sum_eq_zero h0

/-- **Its conclusion fails for a verifier that accepts everything**: at files that pass the commit phase's check, a
strategy on whose every play the claim of one of composite `k`'s subprotocols fails, and errors summing below one. -/
theorem soundness_nontrivial_of {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (k : Verity.Registration.Composite) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hreg : V.Registered θ F) (a : Subprotocol) (ha : a ∈ subprotocols k) (hfail : ∀ o, ¬ V.Holds k θ F c σ a o)
    (hlt : ∑ a ∈ subprotocols k, V.error a θ F c σ < 1) :
    ¬ V.SoundnessFor k fun _ => True := by
  intro h
  have h1 := prob_play_eq_one (V.rounds θ) θ F c
    (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.file ∧ ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o)
    (fun _ _ => ⟨trivial, hreg, a, ha, hfail _⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

/-- An event no play after `θ`, `F` and `c` meets has probability zero. -/
theorem prob_play_eq_zero (ns : List ℕ) (θ F c : ByteArray) (p : Exchange → Prop)
    (hp : ∀ log e, ¬ p ⟨θ, F, c, log, e⟩) (σ : Game.Strategy (play ns θ F c)) : Game.prob p (play ns θ F c) σ = 0 := by
  refine (Game.prob_bind_of_const (fun log => Game.send ByteArray fun e => Game.ret (⟨θ, F, c, log, e⟩ : Exchange)) p
    (fun _ => False) (fun log t => ?_) (releases ns) σ).trans (Game.prob_false _ _)
  simp [Game.prob, hp]

/-- **One subprotocol's failing instance is the theorem's**: at files that pass the commit phase's check, a strategy at
which the claim of one of composite `k`'s subprotocols fails with probability above the errors summed makes the
conclusion fail for a verifier that accepts everything. So the one model-level failing instance (through that
subprotocol's `_nontrivial`) gives the theorem's, with no instance of its own. -/
theorem soundness_nontrivial_of_lt {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (k : Verity.Registration.Composite) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hreg : V.Registered θ F) (a : Subprotocol) (ha : a ∈ subprotocols k)
    (hlt : ∑ b ∈ subprotocols k, V.error b θ F c σ < Game.prob (fun o => ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ) :
    ¬ V.SoundnessFor k fun _ => True := by
  intro h
  have hz : Game.prob (fun o => ¬ V.Registered o.params o.file) (V.game θ F c) σ = 0 :=
    prob_play_eq_zero (V.rounds θ) θ F c _ (fun _ _ hn => hn hreg) σ
  have hle : Game.prob (fun o => ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ ≤
      Game.prob (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.file ∧
        ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ :=
    calc Game.prob (fun o => ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ
        ≤ Game.prob (fun o => ((fun _ => True) o.sent ∧ V.Registered o.params o.file ∧
            ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o) ∨ ¬ V.Registered o.params o.file) (V.game θ F c) σ :=
          Game.prob_mono (fun o ho => by
            by_cases hr : V.Registered o.params o.file
            · exact Or.inl ⟨trivial, hr, a, ha, ho⟩
            · exact Or.inr hr) _ σ
      _ ≤ _ := Game.prob_or_le _ _ _ σ
      _ = _ := by rw [hz, add_zero]
  exact lt_irrefl _ ((hle.trans (h θ F c σ)).trans_lt hlt)

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

/-- **A target of a subprotocol that decides an `Audit` is the per-period guarantee** `Accounting.PeriodLive.Sound` at
the period `live`, read in the one vocabulary. -/
theorem target_audit (k : Verity.Registration.Composite) (b : Audit) :
    V.Target k (.ofAudit b) ↔ ∀ θ F c, (V.live k θ F c).Sound (V.certOf θ) b (V.error (.ofAudit b) θ F c) := by
  cases b <;> exact Iff.rfl

/-- **The honest commitment's case of sampled proofs' claim**: when the extractor's transcript is correct on every unit
and the commitment opens every gate to it (a registered constant to its constant), no unit is unbound, and the claim
holds at any `ε ≥ 0`. -/
theorem holds_sampled_of_opens (k : Verity.Registration.Composite) (θ F c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (o : Exchange)
    (hc : ∀ u, (V.partsOf F).Correct (V.extract θ F c σ) u) (ho : ∀ g, V.Opens θ F c g (V.extract θ F c σ g))
    (hε : 0 ≤ (V.certOf θ).ε) : V.Holds k θ F c σ .sampled o := by
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

namespace Idle

/-- At `unbounded`, every subprotocol's error is `⊤`. -/
theorem error_unbounded (a : Subprotocol) (θ F c : ByteArray) (σ : Game.Strategy (unbounded.game θ F c)) :
    unbounded.error a θ F c σ = ⊤ := by
  unfold Vocab.error Accounting.condError
  split_ifs <;> simp [unbounded, vocab]

/-- At `unbounded`, every subprotocol meets its target at every composite. -/
theorem target_unbounded (k : Verity.Registration.Composite) (a : Subprotocol) : unbounded.Target k a := by
  intro θ F c σ
  rw [error_unbounded]
  exact le_top

/-- The idle finder outputs no collision. -/
theorem finder_pr (H : Claims.Msg → Claims.Msg) : finder.Pr H = 0 := by
  unfold Claims.Finder.Pr Claims.Pr
  simp [finder]

/-- The idle forger forges nothing. -/
theorem forger_pr (verify : Hash.Msg → Hash.Msg → Hash.Msg → Bool) : forger.Pr verify = 0 := by
  unfold Hash.Assumptions.LmsForger.Pr Hash.Assumptions.LmsForger.Forges
  simp [forger]

/-- At `exact`, every subprotocol's error is `0`: its finder and forger meet their premises, and every term is `0`. -/
theorem error_exact (a : Subprotocol) (θ F c : ByteArray) (σ : Game.Strategy (exact.game θ F c)) :
    exact.error a θ F c σ = 0 := by
  unfold Vocab.error Accounting.condError
  rw [if_pos ⟨le_of_eq (finder_pr _), fun _ _ => le_of_eq (forger_pr _)⟩]
  simp [exact, vocab]

/-- At `exact`, PoUW's claim fails on every play: it certifies one gate computed, and no play computes one. -/
theorem pouw_fails (k : Verity.Registration.Composite) (θ F c : ByteArray) (σ : Game.Strategy (exact.game θ F c))
    (o : Exchange) : ¬ exact.Holds k θ F c σ .pouw o := by
  intro h
  change (1 : ℝ) ≤ ((Finset.card (∅ : Finset (Fin (exact.circuitOf F).N)) : ℕ) : ℝ) at h
  simp at h

/-- At `exact`, the conclusion at composite `k` fails for a verifier that accepts everything, at the files that are the
parameters (`ByteArray.empty`), for the honest prover. -/
theorem nontrivial_at (k : Verity.Registration.Composite) (hk : Subprotocol.pouw ∈ subprotocols k) :
    ¬ exact.SoundnessFor k fun _ => True := by
  refine soundness_nontrivial_of exact k .empty .empty .empty (exact.honest .empty .empty .empty) rfl .pouw hk
    (pouw_fails k _ _ _ _) ?_
  simp only [error_exact, Finset.sum_const_zero]
  exact zero_lt_one

end Idle

/-- **`SoundnessOf`'s honest lemma**: its hypotheses hold at the idle deployment's compiled composites, every
subprotocol meeting its target at onsite and at remote. -/
theorem soundness_at_of_targets_honest :
    Idle.unbounded.Target .onsite .sampled ∧ Idle.unbounded.Target .onsite .pouw ∧
      Idle.unbounded.Target .onsite .inferenceOnly ∧ Idle.unbounded.Target .onsite .pous ∧
      Idle.unbounded.Target .onsite .poub ∧ Idle.unbounded.Target .onsite .network ∧
      Idle.unbounded.Target .onsite .wipe ∧ Idle.unbounded.Target .remote .sampled ∧
      Idle.unbounded.Target .remote .pouw ∧ Idle.unbounded.Target .remote .inferenceOnly :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _,
    Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _,
    Idle.target_unbounded _ _, Idle.target_unbounded _ _⟩

/-- **`SoundnessOf`'s nontrivial lemma**: its conclusion fails at onsite for a verifier that accepts everything, at the
idle deployment whose errors are all `0`. -/
theorem soundness_at_of_targets_nontrivial : ¬ Idle.exact.SoundnessFor .onsite fun _ => True :=
  Idle.nontrivial_at .onsite (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

/-- **`Soundness`'s honest lemma**: its seven hypotheses hold at the idle deployment's compiled onsite composite. -/
theorem soundness_of_legs_honest :
    Idle.unbounded.SampledProofsSound .onsite ∧ Idle.unbounded.PouwSound .onsite ∧
      Idle.unbounded.InferenceOnlySound .onsite ∧ Idle.unbounded.PousSound .onsite ∧
      Idle.unbounded.BandwidthAuditSound .onsite ∧ Idle.unbounded.NetworkCertifierSound .onsite ∧
      Idle.unbounded.WipeSound .onsite :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _,
    Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _⟩

/-- **`Soundness`'s nontrivial lemma**: its conclusion fails at onsite for a verifier that accepts everything, at the
idle deployment whose errors are all `0`. -/
theorem soundness_of_legs_nontrivial : ¬ Idle.exact.SoundnessFor .onsite fun _ => True :=
  Idle.nontrivial_at .onsite (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

/-- **`RemoteSoundness`'s honest lemma**: its three hypotheses hold at the idle deployment's compiled remote
composite. -/
theorem remote_soundness_of_legs_honest :
    Idle.unbounded.SampledProofsSound .remote ∧ Idle.unbounded.PouwSound .remote ∧
      Idle.unbounded.InferenceOnlySound .remote :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _⟩

/-- **`RemoteSoundness`'s nontrivial lemma**: its conclusion fails at remote for a verifier that accepts everything, at
the idle deployment whose errors are all `0`. -/
theorem remote_soundness_of_legs_nontrivial : ¬ Idle.exact.SoundnessFor .remote fun _ => True :=
  Idle.nontrivial_at .remote (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

end Verity
