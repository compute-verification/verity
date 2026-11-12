module

public import Properties.Verity.Soundness
public import Properties.Verity.Idle
public import Proofs.Properties.Composition
public import Proofs.Core.Game.Prob
public import Proofs.Core.Game.Seq
public import Proofs.Flock.Claims.Prob

/-!
# Soundness at the verifier, from its subprotocols' targets

A composite's acceptance is its acceptance as a `same` composition and every component's on its own case
(`accepts_same`, `accepts_components`, from `Properties.Composition`'s `answerAccepts` at the composite, `acceptTied` at
its ties and `acceptSame`). So the event `Verity.SoundnessOf` bounds at a composite lies inside the union of its
subprotocols' events, each at most its target's error at the strategy: the union bound over the composite's
subprotocols in one game (`soundness_at_of_targets`). `soundness_of_subprotocols` is it at onsite's seven, and
`remote_soundness_of_subprotocols` at remote's three, which are a subset of onsite's.

Beside it, the two checks every guarantee carries, for any names `V`: `soundness_honest_of`, at the honest deployment
(the deployment's honest prover at registered files, each subprotocol's honest lemma `Vocab.HonestHolds`) every target's
event and the theorem's own have probability `0`; and `soundness_nontrivial_of`, its conclusion fails for a verifier
that accepts everything, wherever a claim fails on every play of a strategy whose files pass the commit phase's check and
the errors sum below one. `Vocab.holds_sampled_of_opens` is the honest commitment's case of sampled proofs' claim: a
correct transcript that the commitment opens everywhere, the constants by their constants, has no unbound unit.

The event isn't restricted to some input shape: the prover sends the committed files before any coin and the
evidence's blobs with the evidence, and the composite reads both, so every input whose blobs begin with the committed
files is what some strategy sends (`sent_reachable`), and every acceptance of the compiled verifier on such an input
at files that pass the commit phase's check lies in the event with probability one (`Vocab.accepts_reachable`).

The premises at a subprotocol's further finders (`Vocab.finderCR`) are what each finder's named claim gives it: under
`ecr/sha-512` its calls are within its budget on every coin, so where it meets `Claims.ExpectedCR` it finds a collision
with probability at most its budget over `2^256`, and under `cr/sha-512` the premise is `Claims.StrictBirthday` itself
(`finderCROf_of_expectedCR`).

The witnesses of each of the three theorems are at the idle deployment (`Properties.Verity.Idle`), at the compiled
composites: its hypotheses hold at `Idle.unbounded` (`soundness_at_of_targets_honest`, `soundness_of_subprotocols_honest`,
`remote_soundness_of_subprotocols_honest`), and its conclusion fails at `Idle.exact` for a verifier that accepts everything
(`soundness_at_of_targets_nontrivial`, `soundness_of_subprotocols_nontrivial`, `remote_soundness_of_subprotocols_nontrivial`): there
every error is `0` and PoUW's claim fails on every play of the files that pass the commit phase's check.
-/

@[expose] public section

namespace Verity

open Accounting Core
open scoped ENNReal

theorem component_mem {k : Verity.Registration.Composite} {a : Subprotocol} {P : BlobProtocol}
    (h : component k a = some P) : P ∈ componentsOf k := by
  cases k <;> cases a <;> simp [component] at h <;> subst h <;>
    simp [componentsOf, Onsite.components, Remote.components]

/-- **Each composite's verifier is the one the composites' table lists for it**, with its subprotocols'
ids. -/
theorem verifier_mem (k : Verity.Registration.Composite) :
    (verifier k, (componentsOf k).map (·.id)) ∈ Composites.composites := by
  cases k
  · exact List.mem_cons.2 (Or.inl rfl)
  · exact List.mem_cons.2 (Or.inr (List.mem_cons.2 (Or.inl rfl)))

/-- **What a composite's verifier accepts, it accepted as a `same` composition**: an accepted composition of
its components whose record subprotocols committed to one record, every window of PoUW's period on the policy's record,
and at onsite PoUS and PoUB-P2 on one store. -/
theorem accepts_same {k : Verity.Registration.Composite} {s : Sent} (h : Accepts k s) : SameAccepts k s := by
  cases k
  · obtain ⟨ps, c, ev, hp, hc, he, pub, prof, hch⟩ :=
      Properties.Composition.answerAccepts (Onsite.protocol.on s.blobs) s.params s.commitment s.evidence h
    obtain ⟨hh, hties⟩ := Properties.Composition.acceptTied Onsite.ties ev
      (Composition.checkSame "verity/onsite/v0" Onsite.tied (s.components .onsite) ps c ev) pub prof hch
    exact ⟨ps, c, ev, ⟨hp, hc, he⟩, pub, prof,
      Properties.Composition.acceptSame _ _ _ _ _ _ _ _ hh, hties⟩
  · obtain ⟨ps, c, ev, hp, hc, he, pub, prof, hch⟩ :=
      Properties.Composition.answerAccepts (Remote.protocol.on s.blobs) s.params s.commitment s.evidence h
    obtain ⟨hh, hties⟩ := Properties.Composition.acceptTied [Remote.windows] ev
      (Composition.checkSame "verity/remote/v0" Remote.tied (s.components .remote) ps c ev) pub prof hch
    exact ⟨ps, c, ev, ⟨hp, hc, he⟩, pub, prof,
      Properties.Composition.acceptSame _ _ _ _ _ _ _ _ hh, hties⟩

/-- **What a composite's verifier accepts, each of its subprotocols' components accepts on its own
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
  calc Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.files ∧
          ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ
      ≤ Game.prob (fun o => ∃ a ∈ subprotocols k, V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ :=
        Game.prob_mono (fun o ⟨hacc, hreg, a, ha, hfail⟩ =>
          ⟨a, ha, ⟨hreg, accepts_same hacc, accepts_components hacc ha⟩, hfail⟩) _ σ
    _ ≤ ∑ a ∈ subprotocols k,
          Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ :=
        prob_exists_finset_le _ _ _ _
    _ ≤ ∑ a ∈ subprotocols k, V.error a θ F c σ := Finset.sum_le_sum fun a ha => hT a ha θ F c σ

theorem soundness_of_subprotocols : Soundness := by
  intro ι α _ d V hS hW hI hP hB hN hX
  refine soundness_at_of_targets ι α d V .onsite fun a _ => ?_
  cases a
  exacts [hS, hW, hI, hP, hB, hN, hX]

theorem remote_soundness_of_subprotocols : RemoteSoundness := by
  intro ι α _ d V hS hW hI
  refine soundness_at_of_targets ι α d V .remote fun a ha => ?_
  cases a
  · exact hS
  · exact hW
  · exact hI
  all_goals exact absurd (Finset.mem_filter.1 ha).2 Bool.false_ne_true

open Classical in
/-- **The close's sample**: from the prover's close, an event's probability is its probability over the device's sample, at
the bytes the close leaves to the device and the evidence the strategy sends after each sample. -/
theorem prob_finish (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c a : ByteArray) (log : List Release)
    (p : Exchange → Prop) (t : Game.Strategy (finish D θ F c a log)) :
    Game.prob p (finish D θ F c a log) t = Game.avg fun ρ : Fin D.draws → Bool =>
      if p ⟨θ, F, c, a, log, t.1, List.ofFn ρ, D.signatures a log t.1 (List.ofFn ρ), (t.2 ρ).1.1, (t.2 ρ).1.2⟩
      then 1 else 0 := rfl

/-- An event every play after `θ`, `F` and `c` meets has probability one. -/
theorem prob_play_eq_one (fs : List (String × ℕ)) {D : Signer} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (p : Exchange → Prop) (hp : ∀ o : Exchange, o.params = θ ∧ o.files = F ∧ o.commitment = c → p o)
    (σ : Game.Strategy (play fs D θ F c)) : Game.prob p (play fs D θ F c) σ = 1 := by
  refine (Game.prob_send p ByteArray (fun a => (releases (frames fs) fs []).bind (finish D θ F c a)) σ).trans <|
    (Game.prob_bind_of_const (finish D θ F c σ.1) p (fun _ => True) (fun log t => ?_)
      (releases (frames fs) fs []) σ.2).trans (Game.prob_true _ _)
  rw [prob_finish]
  refine ((congrArg Game.avg (funext fun ρ => ite_eq_left ?_)).trans (Game.avg_const 1)).trans
    (Game.prob_true _ _).symm
  exact hp _ ⟨rfl, rfl, rfl⟩

/-- **Its hypotheses hold at the honest deployment**: at registered files, where each of composite `k`'s subprotocols'
honest lemma holds (`Vocab.HonestHolds`), the deployment's honest prover meets every target with probability `0` on its
event, and the event the theorem bounds has probability `0`. -/
theorem soundness_honest_of {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (k : Verity.Registration.Composite) (h : ∀ a ∈ subprotocols k, V.HonestHolds k a) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (hreg : V.Registered θ F) :
    (∀ a ∈ subprotocols k, Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c (V.honest θ F c) a o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error a θ F c (V.honest θ F c)) ∧
      Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.files ∧
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
  · calc Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.files ∧
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
    (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hreg : V.Registered θ F) (a : Subprotocol) (ha : a ∈ subprotocols k) (hfail : ∀ o, ¬ V.Holds k θ F c σ a o)
    (hlt : ∑ a ∈ subprotocols k, V.error a θ F c σ < 1) :
    ¬ V.SoundnessFor k fun _ => True := by
  intro h
  have h1 := prob_play_eq_one (V.families θ) θ F c
    (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.files ∧ ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o)
    (fun o h3 => ⟨trivial, h3.1 ▸ h3.2.1 ▸ hreg, a, ha, hfail o⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

/-- An event no play after `θ`, `F` and `c` meets has probability zero. -/
theorem prob_play_eq_zero (fs : List (String × ℕ)) {D : Signer} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (p : Exchange → Prop) (hp : ∀ o : Exchange, o.params = θ ∧ o.files = F ∧ o.commitment = c → ¬ p o)
    (σ : Game.Strategy (play fs D θ F c)) : Game.prob p (play fs D θ F c) σ = 0 := by
  refine (Game.prob_send p ByteArray (fun a => (releases (frames fs) fs []).bind (finish D θ F c a)) σ).trans <|
    (Game.prob_bind_of_const (finish D θ F c σ.1) p (fun _ => False) (fun log t => ?_)
      (releases (frames fs) fs []) σ.2).trans (Game.prob_false _ _)
  rw [prob_finish]
  refine ((congrArg Game.avg (funext fun ρ => ite_eq_right ?_)).trans (Game.avg_const 0)).trans
    (Game.prob_false _ _).symm
  exact hp _ ⟨rfl, rfl, rfl⟩

/-- **One subprotocol's failing instance is the theorem's**: at files that pass the commit phase's check, a strategy at
which the claim of one of composite `k`'s subprotocols fails with probability above the errors summed makes the
conclusion fail for a verifier that accepts everything. So the one model-level failing instance (through that
subprotocol's `_nontrivial`) gives the theorem's, with no instance of its own. -/
theorem soundness_nontrivial_of_lt {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (hreg : V.Registered θ F) (a : Subprotocol) (ha : a ∈ subprotocols k)
    (hlt : ∑ b ∈ subprotocols k, V.error b θ F c σ < Game.prob (fun o => ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ) :
    ¬ V.SoundnessFor k fun _ => True := by
  intro h
  have hz : Game.prob (fun o => ¬ V.Registered o.params o.files) (V.game θ F c) σ = 0 :=
    prob_play_eq_zero (V.families θ) (D := V.signer θ) θ F c _ (fun _ h3 hn => hn (h3.1 ▸ h3.2.1 ▸ hreg)) σ
  have hle : Game.prob (fun o => ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ ≤
      Game.prob (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.files ∧
        ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ :=
    calc Game.prob (fun o => ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ
        ≤ Game.prob (fun o => ((fun _ => True) o.sent ∧ V.Registered o.params o.files ∧
            ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o) ∨ ¬ V.Registered o.params o.files) (V.game θ F c) σ :=
          Game.prob_mono (fun o ho => by
            by_cases hr : V.Registered o.params o.files
            · exact Or.inl ⟨trivial, hr, a, ha, ho⟩
            · exact Or.inr hr) _ σ
      _ ≤ _ := Game.prob_or_le _ _ _ σ
      _ = _ := by rw [hz, add_zero]
  exact lt_irrefl _ ((hle.trans (h θ F c σ)).trans_lt hlt)

/-- **An opening opens its stream's next unopened round**: the round `OpenTree.next` gives for stream `s` is the number of
the stream's rounds already opened. -/
theorem OpenTree.next_round {T : OpenTree} {s n : ℕ} {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)}
    (h : T.next s n = some p) : p.1.val = s ∧ p.2.val = T.opened p.1 := by
  unfold OpenTree.next at h
  split at h
  · split at h
    · cases h
      exact ⟨rfl, rfl⟩
    · cases h
  · cases h

/-- **Opening a round counts it**: once stream `s`'s next round is opened, its count is one more and every other stream's
is unchanged, so with `OpenTree.next_round` a stream's rounds are opened in order, each at most once. -/
theorem OpenTree.bump_opened (T : OpenTree) (s j : Fin T.spec.streams.size) :
    (T.bump s).opened j = if j = s then T.opened s + 1 else T.opened j := rfl

/-- The device's releases have a strategy from any coins and coin trees left: the prover's close. -/
theorem releases_nonempty : (k : ℕ) → (fs : List (String × ℕ)) → (ts : List OpenTree) →
    Nonempty (Game.Strategy (releases k fs ts))
  | 0, _, _ => ⟨PUnit.unit⟩
  | _ + 1, _, _ => ⟨⟨none, PUnit.unit⟩⟩

/-- Playing `s` in `g` and then `t b` in `f b`, where each `t b` meets `p` surely, meets `p` surely. -/
theorem prob_bind_strategy {β γ : Type} (f : β → Game γ) (p : γ → Prop) (t : (b : β) → Game.Strategy (f b))
    (ht : ∀ b, Game.prob p (f b) (t b) = 1) :
    ∀ (g : Game β) (s : Game.Strategy g),
      Game.prob p (g.bind f) (Game.Strategy.bind g f s t) = Game.prob (fun _ => True) g s
  | .ret b, _ => by
      show Game.prob p (f b) (t b) = _
      rw [ht b]
      simp [Game.prob]
  | .send _ k, s => prob_bind_strategy f p t ht (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg Game.avg (funext fun c => prob_bind_strategy f p t ht (k c) (s c))

/-- **Every input whose blobs begin with the committed files is what some play sends**: at any parameters `θ`,
committed files `F` and commitment `c`, and any evidence `e` and evidence blobs `B`, the strategy that sends an empty
request, closes the device at once with an empty close and then sends `e` with `B` ends every play with the composite
reading `⟨θ, c, e, F ++ B⟩`. So the game's event admits every input of that shape the compiled verifier accepts. -/
theorem sent_reachable (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c e : ByteArray)
    (B : Array ByteArray) :
    ∃ σ : Game.Strategy (play fs D θ F c),
      Game.prob (fun o => o.params = θ ∧ o.files = F ∧ o.sent = ⟨θ, c, e, F ++ B⟩) (play fs D θ F c) σ = 1 := by
  let t : (log : List Release) → Game.Strategy (finish D θ F c .empty log) :=
    fun _ => ⟨.empty, fun _ => ⟨(e, B), PUnit.unit⟩⟩
  obtain ⟨s⟩ := releases_nonempty (frames fs) fs []
  refine ⟨⟨.empty, Game.Strategy.bind (releases (frames fs) fs []) (finish D θ F c .empty) s t⟩, ?_⟩
  refine (Game.prob_send _ ByteArray (fun a => (releases (frames fs) fs []).bind (finish D θ F c a)) _).trans <|
    (prob_bind_strategy (finish D θ F c .empty) _ t (fun log => ?_) (releases (frames fs) fs []) s).trans
      (Game.prob_true _ _)
  rw [prob_finish]
  refine (congrArg Game.avg (funext fun ρ => ite_eq_left ?_)).trans (Game.avg_const 1)
  exact ⟨rfl, rfl, rfl⟩

/-- **The event of `Vocab.SoundnessAt` holds every acceptance of the compiled verifier**: if composite `k`'s compiled
verifier accepts an input whose blobs are committed files that pass the commit phase's check followed by any evidence
blobs, then in the game after those parameters, files and commitment some strategy makes the acceptance at files
that pass the check hold with probability one. -/
theorem Vocab.accepts_reachable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)
    (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c e : ByteArray) (B : Array ByteArray)
    (hacc : Accepts k ⟨θ, c, e, F ++ B⟩) (hreg : V.Registered θ F) :
    ∃ σ : Game.Strategy (V.game θ F c),
      Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.files) (V.game θ F c) σ = 1 := by
  obtain ⟨σ, hσ⟩ := sent_reachable (V.families θ) (V.signer θ) θ F c e B
  refine ⟨σ, le_antisymm (Game.prob_le_one _ _ _) (hσ.symm.le.trans (Game.prob_mono ?_ _ σ))⟩
  rintro o ⟨hθ, hF, hs⟩
  refine ⟨?_, ?_⟩
  · rw [hs]
    exact hacc
  · rw [hθ, hF]
    exact hreg

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
theorem holds_sampled_of_opens (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (o : Exchange)
    (hc : ∀ u, (V.partsOf θ F).Correct (V.extract θ F c σ) u) (ho : ∀ g, V.Opens θ F c g (V.extract θ F c σ g))
    (hε : 0 ≤ (V.certOf θ).ε) : V.Holds k θ F c σ .sampled o := by
  have hu : Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) = ∅ := by
    classical
    refine Finset.filter_eq_empty_iff.2 fun u _ => ?_
    rintro (hw | ⟨g, -, hg⟩)
    · exact hw (hc u)
    · exact hg (ho g)
  show ((Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
    (V.certOf θ).ε * V.units θ F
  rw [hu, Finset.card_empty, Nat.cast_zero]
  exact mul_nonneg hε (Nat.cast_nonneg _)

/-- A registered constant opens to exactly the values holding its constant. -/
theorem opens_const {θ c : ByteArray} {F : Array ByteArray} {g : Fin (V.circuitOf θ F).N} {b : Bool}
    (hg : (V.read θ F).consts g = some b) (v : BitVec ((V.circuitOf θ F).w g)) :
    V.Opens θ F c g v ↔ ∀ i : Fin ((V.circuitOf θ F).w g), v.getLsbD i.val = b := by
  unfold Opens
  rw [hg]

end Vocab

/-- **A file finder outputs a collision** when the committed file the digest `h` names is not the registered one and
has its SHA-512: the circuit file's (`Vocab.fileFinder`) and the public file's (`Vocab.publicFinder`). -/
theorem namedFinder_collides {h : Option Hash.Digest} {F R : Array ByteArray} (hne : namedFile h F ≠ namedFile h R)
    (hd : Verity.sha512Msg (namedFile h F).data.toList = Verity.sha512Msg (namedFile h R).data.toList) :
    (namedFinder h F R).Pr Verity.sha512Msg = 1 := by
  have hl : (namedFile h F).data.toList ≠ (namedFile h R).data.toList := fun e =>
    hne (ByteArray.ext (Array.ext' e))
  have hs : ∀ s : (namedFinder h F R).S,
      ∃ u v, (namedFinder h F R).out s = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
    fun _ => ⟨_, _, by simp only [namedFinder, if_neg hne], hl, hd⟩
  classical
  unfold Claims.Finder.Pr Claims.Pr Claims.prUniform
  rw [Finset.filter_true_of_mem fun s _ => hs s, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)

/-- **A file finder's SHA-512 calls**: two on its one coin, at most `0 · calls(τ) + 2`. -/
theorem namedFinder_calls (h : Option Hash.Digest) (F R : Array ByteArray) (x : (namedFinder h F R).S) (n : ℕ) :
    (namedFinder h F R).cost x ≤ 0 * n + 2 := by
  simp [namedFinder]

/-- **`ecr/sha-512` at a further finder asked under it gives its premise**: its calls are within its budget on every
coin (`FinderSpec.calls`), so where it meets `Claims.ExpectedCR` it finds a collision with probability at most its
budget over `2^256`. -/
theorem FinderSpec.cr_of_expectedCR {G : Game Exchange} {cost : Game.Strategy G → ℕ} (f : FinderSpec G cost)
    (hc : f.claim = .ecr) (σ : Game.Strategy G) (h : Claims.ExpectedCR Verity.sha512Msg (f.finder σ)) : f.CR σ := by
  unfold FinderSpec.CR
  rw [hc]
  unfold Claims.ExpectedCR at h
  have hex : Claims.Ex (fun s => ((f.finder σ).cost s : ℝ)) ≤ (f.budget (cost σ) : ℝ) :=
    calc Claims.Ex (fun s => ((f.finder σ).cost s : ℝ))
        ≤ Claims.Ex (fun _ : (f.finder σ).S => (f.budget (cost σ) : ℝ)) :=
          Claims.Ex_mono fun s => Nat.cast_le.2 (f.calls σ s)
      _ = (f.budget (cost σ) : ℝ) := Claims.Ex_const _
  exact h.trans (div_le_div_of_nonneg_right hex (Real.rpow_nonneg (by norm_num) _))

/-- **Each further finder's named claim gives their premises** (`finderCROf`): `ecr/sha-512` at each asked under it,
and `cr/sha-512` at each asked under the strict birthday bound. -/
theorem finderCROf_of_expectedCR {G : Game Exchange} {cost : Game.Strategy G → ℕ} (fs : List (FinderSpec G cost))
    (σ : Game.Strategy G)
    (h : ∀ f ∈ fs, (f.claim = .ecr ∧ Claims.ExpectedCR Verity.sha512Msg (f.finder σ)) ∨
      (f.claim = .cr ∧ Claims.StrictBirthday Verity.sha512Msg (f.finder σ))) : finderCROf fs σ := by
  unfold finderCROf
  refine List.forall_iff_forall_mem.2 fun f hf => ?_
  rcases h f hf with ⟨hc, he⟩ | ⟨hc, hs⟩
  · exact f.cr_of_expectedCR hc σ he
  · unfold FinderSpec.CR
    rw [hc]
    exact hs

/-- **`B^τ`'s collision premise gives its bound**: where subprotocol `a`'s finder meets `Claims.ExpectedCR`
(`Vocab.FinderCR`), its calls within its budget (`Vocab.finderSpec`) put its collision probability at most `crBound`
at the strategy's calls. -/
theorem Vocab.finderPr_le_crBound {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) {a : Subprotocol}
    {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (h : V.FinderCR a θ F c σ) : (V.finder a θ F c σ).Pr Verity.sha512Msg ≤ V.crBound θ a (V.cost θ F c σ) :=
  (V.finderSpec a θ F c).cr_of_expectedCR rfl σ h

namespace Idle

/-- At `unbounded`, every subprotocol's error is `⊤`. -/
theorem error_unbounded (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (unbounded.game θ F c)) :
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
  unfold Claims.Finder.Pr Claims.Pr Claims.prUniform
  simp [finder]
  exact Or.inl rfl

/-- The idle forger forges nothing. -/
theorem forger_pr (verify : Hash.Msg → Hash.Msg → Hash.Msg → Bool) : forger.Pr verify = 0 := by
  unfold Hash.Assumptions.LmsForger.Pr Hash.Assumptions.LmsForger.Forges
  simp [forger]
  exact Or.inl rfl

/-- At `exact`, every subprotocol's error is `0`: its finder and forger meet their premises, and every term is `0`. -/
theorem error_exact (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (exact.game θ F c)) :
    exact.error a θ F c σ = 0 := by
  unfold Vocab.error Accounting.condError
  rw [if_pos ⟨trivial, trivial, (finder_pr _).trans_le
    (div_nonneg (Claims.Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)), fun _ _ => le_of_eq (forger_pr _)⟩]
  simp [exact, vocab]

/-- At `exact`, PoUW's claim fails on every play: it certifies one gate computed, and no play computes one. -/
theorem pouw_fails (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (exact.game θ F c))
    (o : Exchange) : ¬ exact.Holds k θ F c σ .pouw o := by
  intro h
  change (1 : ℝ) ≤ ((Finset.card (∅ : Finset (Fin (exact.circuitOf θ F).N)) : ℕ) : ℝ) at h
  simp at h
  norm_num at h

/-- At `exact`, the conclusion at composite `k` fails for a verifier that accepts everything, at the files that are the
parameters (`ByteArray.empty`, the one committed file), for the honest prover. -/
theorem nontrivial_at (k : Verity.Registration.Composite) (hk : Subprotocol.pouw ∈ subprotocols k) :
    ¬ exact.SoundnessFor k fun _ => True := by
  have h0 : ∑ a ∈ subprotocols k, exact.error a .empty #[.empty] .empty (exact.honest .empty #[.empty] .empty) = 0 :=
    Finset.sum_eq_zero fun a _ => error_exact a _ _ _ _
  exact soundness_nontrivial_of exact k .empty #[.empty] .empty (exact.honest .empty #[.empty] .empty)
    ⟨⟨.empty, by simp, rfl⟩, ⟨.empty, by simp, rfl⟩⟩ .pouw hk (pouw_fails k _ _ _ _) (lt_of_eq_of_lt h0 zero_lt_one)

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
theorem soundness_of_subprotocols_honest :
    Idle.unbounded.SpotSound .onsite ∧ Idle.unbounded.PouwSound .onsite ∧
      Idle.unbounded.InferenceOnlySound .onsite ∧ Idle.unbounded.PousSound .onsite ∧
      Idle.unbounded.BandwidthAuditSound .onsite ∧ Idle.unbounded.NetworkCertifierSound .onsite ∧
      Idle.unbounded.WipeSound .onsite :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _,
    Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _⟩

/-- **`Soundness`'s nontrivial lemma**: its conclusion fails at onsite for a verifier that accepts everything, at the
idle deployment whose errors are all `0`. -/
theorem soundness_of_subprotocols_nontrivial : ¬ Idle.exact.SoundnessFor .onsite fun _ => True :=
  Idle.nontrivial_at .onsite (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

/-- **`RemoteSoundness`'s honest lemma**: its three hypotheses hold at the idle deployment's compiled remote
composite. -/
theorem remote_soundness_of_subprotocols_honest :
    Idle.unbounded.SpotSound .remote ∧ Idle.unbounded.PouwSound .remote ∧
      Idle.unbounded.InferenceOnlySound .remote :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _⟩

/-- **`RemoteSoundness`'s nontrivial lemma**: its conclusion fails at remote for a verifier that accepts everything, at
the idle deployment whose errors are all `0`. -/
theorem remote_soundness_of_subprotocols_nontrivial : ¬ Idle.exact.SoundnessFor .remote fun _ => True :=
  Idle.nontrivial_at .remote (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

end Verity
