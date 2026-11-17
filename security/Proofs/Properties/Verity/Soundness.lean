module

public import Properties.Verity.Soundness
public import Properties.Verity.Idle
public import Proofs.Properties.Composition
public import Proofs.Properties.Verity.Soundness.Composite
public import Proofs.Core.Game.Prob
public import Proofs.Core.Game.Seq
public import Proofs.Flock.Claims.Prob

/-! Soundness at the verifier from its subprotocols' targets, by the union bound over a composite's components. -/

@[expose] public section

namespace Verity

open ResourceAccounting Core
open scoped ENNReal

theorem accepts_same {k : Verity.Registration.Composite} {s : Sent} (h : Accepts k s) : SameAccepts k s := by
  obtain ⟨ps, c, ev, hp, hc, he, pub, prof, hch⟩ :=
    Properties.Composition.answerAccepts ((verifier k).on s.blobs) s.params s.commitment s.evidence h
  obtain ⟨hh, hties⟩ := Properties.Composition.acceptTied (ties k) ev
    (Composition.checkSame (compositeId k) (tied k) (s.components k) ps c ev) pub prof hch
  exact ⟨ps, c, ev, ⟨hp, hc, he⟩, pub, prof,
    Properties.Composition.acceptSame _ _ _ _ _ _ _ _ hh, hties⟩

theorem accepts_components {k : Verity.Registration.Composite} {s : Sent} (h : Accepts k s) {a : Subprotocol}
    (ha : a ∈ subprotocols k) : ComponentAccepts k a s := by
  obtain ⟨ps, c, ev, hdec, pub, prof, ⟨⟨⟨-, -, hfst, -⟩, hcases, -⟩, -⟩, -⟩ := accepts_same h
  obtain ⟨P, hP⟩ : ∃ P, component k a = some P := Option.isSome_iff_exists.1 (Finset.mem_filter.1 ha).2
  have hm : P.onBlobs (Blobs.ofArray s.blobs) ∈ (Properties.Composition.triples (s.components k) ps ev).map (·.1) := by
    rw [hfst]
    exact List.mem_map.2 ⟨P, component_mem hP, rfl⟩
  obtain ⟨x, hx, hx1⟩ := List.mem_map.mp hm
  exact ⟨P, hP, ps, c, ev, hdec, x, hx, hx1, hcases x hx⟩

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

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι}

theorem targets_onsite (V : Vocab d α) (hS : V.SpotSound .onsite) (hW : V.PouwSound .onsite)
    (hI : V.InferenceOnlySound .onsite) (hP : V.PousSound .onsite) (hB : V.BandwidthAuditSound .onsite)
    (hN : V.NetworkAttesterSound .onsite) (hX : V.WipeSound .onsite) :
    ∀ a ∈ subprotocols .onsite, V.Target .onsite a := by
  intro a _
  cases a
  exacts [hS, hW, hI, hP, hB, hN, hX]

theorem targets_remote (V : Vocab d α) (hS : V.SpotSound .remote) (hW : V.PouwSound .remote)
    (hI : V.InferenceOnlySound .remote) : ∀ a ∈ subprotocols .remote, V.Target .remote a := by
  intro a ha
  have hm := mem_subprotocols.1 ha
  cases a
  · exact hS
  · exact hW
  · exact hI
  all_goals exact absurd hm (by decide)

theorem targets_plain (V : Vocab d α) (hS : V.SpotSound .plain) (hW : V.PouwSound .plain)
    (hI : V.InferenceOnlySound .plain) : ∀ a ∈ subprotocols .plain, V.Target .plain a := by
  intro a ha
  have hm := mem_subprotocols.1 ha
  cases a
  · exact hS
  · exact hW
  · exact hI
  all_goals exact absurd hm (by decide)

end Vocab

theorem soundness_of_subprotocols : Soundness :=
  fun ι α _ d V hS hW hI hP hB hN hX =>
    soundness_at_of_targets ι α d V .onsite (V.targets_onsite hS hW hI hP hB hN hX)

theorem plain_soundness_of_subprotocols : PlainSoundness :=
  fun ι α _ d V hS hW hI => soundness_at_of_targets ι α d V .plain (V.targets_plain hS hW hI)

open Classical in
theorem prob_finish (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c a : ByteArray) (log : List Release)
    (p : Exchange → Prop) (t : Game.Strategy (finish D θ F c a log)) :
    Game.prob p (finish D θ F c a log) t = Game.avg fun ρ : Fin D.draws → Bool =>
      if p ⟨θ, F, c, a, log, t.1, List.ofFn ρ, D.signatures a log t.1 (List.ofFn ρ), (t.2 ρ).1.1, (t.2 ρ).1.2⟩
      then 1 else 0 := rfl

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
    (hreg : V.Registered θ F) (a : Subprotocol) (ha : a ∈ subprotocols k)
    (hfail : ∀ o : Exchange, o.params = θ ∧ o.files = F ∧ o.commitment = c → ¬ V.Holds k θ F c σ a o)
    (hlt : ∑ a ∈ subprotocols k, V.error a θ F c σ < 1) :
    ¬ V.SoundnessFor k fun _ => True := by
  intro h
  have h1 := prob_play_eq_one (V.families θ) θ F c
    (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.files ∧ ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o)
    (fun o h3 => ⟨trivial, h3.1 ▸ h3.2.1 ▸ hreg, a, ha, hfail o h3⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

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
which one of composite `k`'s subprotocols' claims fails with probability above the errors summed makes the conclusion
fail for a verifier that accepts everything. -/
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

theorem OpenTree.next_round {T : OpenTree} {s n : ℕ} {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)}
    (h : T.next s n = some p) : p.1.val = s ∧ p.2.val = T.opened p.1 := by
  unfold OpenTree.next at h
  split at h
  · split at h
    · cases h
      exact ⟨rfl, rfl⟩
    · cases h
  · cases h

theorem OpenTree.bump_opened (T : OpenTree) (s j : Fin T.spec.streams.size) :
    (T.bump s).opened j = if j = s then T.opened s + 1 else T.opened j := rfl

theorem releases_nonempty : (k : ℕ) → (fs : List (String × ℕ)) → (ts : List OpenTree) →
    Nonempty (Game.Strategy (releases k fs ts))
  | 0, _, _ => ⟨PUnit.unit⟩
  | _ + 1, _, _ => ⟨⟨none, PUnit.unit⟩⟩

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

theorem holds_pouw_of_mem {k : Verity.Registration.Composite} (hk : PouwCheck.PearlPlus.Protocol.protocolId ∈ k)
    {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {o : Exchange} :
    V.Holds k θ F c σ .pouw o ↔ ∃ w, PearlPlusCredits k o.sent w ∧
      (1 - (w.rho : ℝ)) * (1 - (w.epsilon : ℝ)) * w.macs ≤ V.macs θ F c σ o := by
  simp only [Holds, if_pos hk]

theorem holds_pouw_of_not_mem {k : Verity.Registration.Composite} (hk : PouwCheck.PearlPlus.Protocol.protocolId ∉ k)
    {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {o : Exchange} :
    V.Holds k θ F c σ .pouw o ↔ (V.live k θ F c).Holds (V.certOf θ) σ .pouw o := by
  simp only [Holds, if_neg hk]

theorem target_audit (k : Verity.Registration.Composite) (b : Audit) (hb : k = .onsite ∨ b ≠ .pouw) :
    V.Target k (.ofAudit b) ↔ ∀ θ F c, (V.live k θ F c).Sound (V.certOf θ) b (V.error (.ofAudit b) θ F c) := by
  cases b
  case pouw =>
    have hk : PouwCheck.PearlPlus.Protocol.protocolId ∉ k := by
      rcases hb with rfl | h
      · decide
      · exact absurd rfl h
    simp only [Target, Subprotocol.ofAudit, V.holds_pouw_of_not_mem hk]
    exact Iff.rfl
  all_goals exact Iff.rfl

theorem holds_sampled_of_opens (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (o : Exchange)
    (hc : ∀ u, (V.partsOf θ F).Correct (V.extract θ F c σ) u) (ho : ∀ g, V.Opens θ F c g (V.extract θ F c σ g))
    (hε : 0 ≤ (V.certOf θ).ε) : V.Holds k θ F c σ .sampled o := by
  have hu : ResourceAccounting.unbound (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) = ∅ := by
    classical
    refine Finset.filter_eq_empty_iff.2 fun u _ => ?_
    rintro (hw | ⟨g, -, hg⟩)
    · exact hw (hc u)
    · exact hg (ho g)
  show ((ResourceAccounting.unbound (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
    (V.certOf θ).ε * V.units θ F
  rw [hu, Finset.card_empty, Nat.cast_zero]
  exact mul_nonneg hε (Nat.cast_nonneg _)

theorem opens_const {θ c : ByteArray} {F : Array ByteArray} {g : Fin (V.circuitOf θ F).N} {b : Bool}
    (hg : (V.read θ F).consts g = some b) (v : BitVec ((V.circuitOf θ F).w g)) :
    V.Opens θ F c g v ↔ ∀ i : Fin ((V.circuitOf θ F).w g), v.getLsbD i.val = b := by
  unfold Opens
  rw [hg]

end Vocab

theorem namedFinder_collides {h : Option Hash.Digest} {F R : Array ByteArray} (hne : namedFile h F ≠ namedFile h R)
    (hd : sha512Msg (namedFile h F).data.toList = sha512Msg (namedFile h R).data.toList) :
    (namedFinder h F R).Pr sha512Msg = 1 := by
  have hl : (namedFile h F).data.toList ≠ (namedFile h R).data.toList := fun e =>
    hne (ByteArray.ext (Array.ext' e))
  have hs : ∀ s : (namedFinder h F R).S,
      ∃ u v, (namedFinder h F R).out s = some (u, v) ∧ Claims.Collision sha512Msg u v :=
    fun _ => ⟨_, _, by simp only [namedFinder, if_neg hne], hl, hd⟩
  classical
  unfold Claims.Finder.Pr Claims.Pr Claims.prUniform
  rw [Finset.filter_true_of_mem fun s _ => hs s, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)

theorem namedFinder_calls (h : Option Hash.Digest) (F R : Array ByteArray) (x : (namedFinder h F R).S) (n : ℕ) :
    (namedFinder h F R).cost x ≤ 0 * n + 2 := by
  simp [namedFinder]

theorem FinderSpec.cr_of_expectedCR {G : Game Exchange} {cost : Game.Strategy G → ℕ} (f : FinderSpec G cost)
    (hc : f.claim = .ecr) (σ : Game.Strategy G) (h : Claims.ExpectedCR sha512Msg (f.finder σ)) : f.CR σ := by
  unfold FinderSpec.CR
  rw [hc]
  unfold Claims.ExpectedCR at h
  have hex : Claims.Ex (fun s => ((f.finder σ).cost s : ℝ)) ≤ (f.budget (cost σ) : ℝ) :=
    calc Claims.Ex (fun s => ((f.finder σ).cost s : ℝ))
        ≤ Claims.Ex (fun _ : (f.finder σ).S => (f.budget (cost σ) : ℝ)) :=
          Claims.Ex_mono fun s => Nat.cast_le.2 (f.calls σ s)
      _ = (f.budget (cost σ) : ℝ) := Claims.Ex_const _
  exact h.trans (div_le_div_of_nonneg_right hex (Real.rpow_nonneg (by norm_num) _))

theorem finderCROf_of_expectedCR {G : Game Exchange} {cost : Game.Strategy G → ℕ} (fs : List (FinderSpec G cost))
    (σ : Game.Strategy G)
    (h : ∀ f ∈ fs, (f.claim = .ecr ∧ Claims.ExpectedCR sha512Msg (f.finder σ)) ∨
      (f.claim = .cr ∧ Claims.StrictBirthday sha512Msg (f.finder σ))) : finderCROf fs σ := by
  unfold finderCROf
  refine List.forall_iff_forall_mem.2 fun f hf => ?_
  rcases h f hf with ⟨hc, he⟩ | ⟨hc, hs⟩
  · exact f.cr_of_expectedCR hc σ he
  · unfold FinderSpec.CR
    rw [hc]
    exact hs

theorem Vocab.finderPr_le_crBound {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) {a : Subprotocol}
    {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (h : V.FinderCR a θ F c σ) : (V.finder a θ F c σ).Pr sha512Msg ≤ V.crBound θ a (V.cost θ F c σ) :=
  (V.finderSpec a θ F c).cr_of_expectedCR rfl σ h

namespace Idle

theorem error_unbounded (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (unbounded.game θ F c)) :
    unbounded.error a θ F c σ = ⊤ := by
  unfold Vocab.error ResourceAccounting.condError
  split_ifs <;> simp [unbounded, vocab]

theorem target_unbounded (k : Verity.Registration.Composite) (a : Subprotocol) : unbounded.Target k a := by
  intro θ F c σ
  rw [error_unbounded]
  exact le_top

theorem wiresSound_unbounded (k : Verity.Registration.Composite) : unbounded.WiresSound k := by
  intro θ F c σ
  rw [error_unbounded, ENNReal.mul_top (by norm_num)]
  exact le_top

theorem finder_pr (H : Claims.Msg → Claims.Msg) : finder.Pr H = 0 := by
  unfold Claims.Finder.Pr Claims.Pr Claims.prUniform
  simp [finder]
  exact Or.inl rfl

theorem forger_pr (verify : Hash.Msg → Hash.Msg → Hash.Msg → Bool) : forger.Pr verify = 0 := by
  unfold Hash.Assumptions.LmsForger.Pr Hash.Assumptions.LmsForger.Forges
  simp [forger]
  exact Or.inl rfl

theorem error_exact (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (exact.game θ F c)) :
    exact.error a θ F c σ = 0 := by
  unfold Vocab.error ResourceAccounting.condError
  rw [if_pos ⟨trivial, trivial, (finder_pr _).trans_le
    (div_nonneg (Claims.Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)), fun _ _ => le_of_eq (forger_pr _)⟩]
  simp [exact, vocab]

/-- At `exact`, PoUW's claim fails on every play after the empty commitment: where the composite doesn't list Pearl+'s
id (onsite) it certifies one gate computed, which no play computes, and where it does (remote) no credit is published,
because the empty commitment is no binding of the components' digests. -/
theorem pouw_fails (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (σ : Game.Strategy (exact.game θ F .empty)) (o : Exchange)
    (ho : o.params = θ ∧ o.files = F ∧ o.commitment = .empty) : ¬ exact.Holds k θ F .empty σ .pouw o := by
  intro h
  by_cases hk : PouwCheck.PearlPlus.Protocol.protocolId ∈ k
  · obtain ⟨_, ⟨_, c', _, ⟨_, hc, _⟩, _⟩, _⟩ := (exact.holds_pouw_of_mem hk).1 h
    have h64 : ¬ ByteArray.empty.size = 64 := by decide
    change Composition.decodeCommitment o.commitment = .ok c' at hc
    rw [ho.2.2, Composition.decodeCommitment, if_neg h64] at hc
    cases hc
  · have h := (exact.holds_pouw_of_not_mem hk).1 h
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
    ⟨⟨.empty, by simp, rfl⟩, ⟨.empty, by simp, rfl⟩⟩ .pouw hk (pouw_fails k _ _ _) (lt_of_eq_of_lt h0 zero_lt_one)

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
  Idle.nontrivial_at .onsite (mem_subprotocols.2 (by decide))

/-- **`Soundness`'s honest lemma**: its seven hypotheses hold at the idle deployment's compiled onsite composite. -/
theorem soundness_of_subprotocols_honest :
    Idle.unbounded.SpotSound .onsite ∧ Idle.unbounded.PouwSound .onsite ∧
      Idle.unbounded.InferenceOnlySound .onsite ∧ Idle.unbounded.PousSound .onsite ∧
      Idle.unbounded.BandwidthAuditSound .onsite ∧ Idle.unbounded.NetworkAttesterSound .onsite ∧
      Idle.unbounded.WipeSound .onsite :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _,
    Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _⟩

/-- **`Soundness`'s nontrivial lemma**: its conclusion fails at onsite for a verifier that accepts everything, at the
idle deployment whose errors are all `0`. -/
theorem soundness_of_subprotocols_nontrivial : ¬ Idle.exact.SoundnessFor .onsite fun _ => True :=
  Idle.nontrivial_at .onsite (mem_subprotocols.2 (by decide))

/-- **`PlainSoundness`'s honest lemma**: its three targets hold at the idle deployment, at the plain sibling. -/
theorem plain_soundness_of_subprotocols_honest :
    Idle.unbounded.SpotSound .plain ∧ Idle.unbounded.PouwSound .plain ∧
      Idle.unbounded.InferenceOnlySound .plain :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.target_unbounded _ _⟩

/-- **`PlainSoundness`'s nontrivial lemma**: its conclusion fails at the plain sibling for a verifier that accepts
everything, at the idle deployment whose errors are all `0`. -/
theorem plain_soundness_of_subprotocols_nontrivial : ¬ Idle.exact.SoundnessFor .plain fun _ => True :=
  Idle.nontrivial_at .plain (mem_subprotocols.2 (by decide))

end Verity
