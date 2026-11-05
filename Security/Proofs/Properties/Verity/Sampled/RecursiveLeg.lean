module

public import Proofs.Properties.Verity.Sampled.Leg
public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.Verity.Recursive.Audit

/-!
# Sampled proofs' recursive form at the deployment, from the targets still open

`RecursiveForm V k`, the leg's target at parameters that don't name the one-stage form (`ParamsAt`), from
`Flock.SecurityProofs.RecursiveKS` through the recursive configuration's target
(`Recursive.sampledProofsRecursive_of_ks`):

* **the verdict** (`recVerdict_of_verdict`): off the one-stage form, the sampled verdict is the recursive
  configuration's (`Recursive.RecVerdict`), and its parameters name the recursive form (`RecParamsAt`). Sampled proofs'
  case is the one the composite's parameters name under its id, since the components' ids are distinct, so a case at
  the one-stage form would be `ParamsAt`'s; parameters that name neither form make the component refuse.
* **the session** (`RecSession`, at each strategy where the parameters name the recursive form and the files are
  registered): a deployment `RecursiveKS` is stated at (the deployed weights, decoders meeting the list-decoding spec),
  the recursive session inside the causal game (`Recursive.SessionIn`, the coupling), the deployment's reading of a
  witness and its collision from two (the recursive bridge), the sample's miss and `live-verifier` (`LiveCoins`); and
  the vocabulary tied to it as `Deployed` ties it to v2: the extractor is the session's, `error`'s condition holds the
  session's per-finder premises (`Recursive.Dep.FinderCR`), and the published `δ` is at least the session's
  (`Recursive.Dep.delta`).
* `recursiveForm_of_deployed`: the leg's target, the recursive configuration's bound (`RecSession.bound`) under the
  vocabulary's error (`RecSession.errAt_le`).

The published `δ` dominates the session's `δs + E[ksAt] + √(min(W, 2n) · E[bindBudget] / 2^256)`; route 2C's
`2^{-167.7} + t' · 2^{-174.95}` is only its knowledge-soundness term.
-/

@[expose] public section

namespace Verity.Sampled

open scoped ENNReal
open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **The parameters `θ` name the recursive form**: the components' parameters decode, and sampled proofs' parameters
in them, the first under its id as `ParamsAt` finds them, decode to the recursive form. -/
def RecParamsAt (θ : ByteArray) : Prop :=
  ∃ ps, (_root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams) = .ok ps ∧
    ∃ x, ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x ∧
      ∃ p, (_root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inr p)

/-- **Sampled proofs' case is the one the composite's parameters name under its id**, in a composition composite `k`
accepted, since the components' ids are distinct. -/
theorem find_case {k : Verity.Registration.Composite} {s : Verity.Sent} (hS : Verity.SameAccepts k s)
    {ps : List _root_.Core.Composition.Named} {c₀ : ByteArray} {evs : List _root_.Core.Composition.Case}
    (hd : Verity.Decoded s ps c₀ evs)
    {x : _root_.Core.Protocol × _root_.Core.Composition.Named × _root_.Core.Composition.Case}
    (hx : x ∈ Properties.Composition.triples (s.components k) ps evs)
    (hx1 : x.1 = VStar.Recursive.Sampled.protocol.onBlobs (_root_.Core.Blobs.ofArray s.blobs)) :
    ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x.2.1 := by
  obtain ⟨hps, -, hevs⟩ := hd
  obtain ⟨ps', c₀', evs', ⟨hps', -, hevs'⟩, pub, prof, ⟨⟨⟨hfp, hfe, -, -⟩, -⟩, -⟩, -⟩ := hS
  rw [Except.ok.inj (hps'.symm.trans hps)] at hfp
  rw [Except.ok.inj (hevs'.symm.trans hevs)] at hfe
  have hid : x.1.id = VStar.Recursive.Sampled.PROTOCOL := (congrArg (·.id) hx1).trans rfl
  have hpid : x.2.1.id = x.1.id := (ids_of_zip hfp hfe hx).1
  have hm := List.of_mem_zip (List.of_mem_zip hx).2
  rw [← hid, ← hpid]
  exact find?_nodup (by rw [hfp]; exact components_nodup k s) hm.1

/-- **Off the one-stage form, the sampled verdict is the recursive configuration's**: where composite `k`'s sampled
verdict holds of what the prover sent and its parameters don't name the one-stage form, sampled proofs' case names the
recursive form (`Recursive.RecVerdict`), and so do the parameters (`RecParamsAt`). -/
theorem recVerdict_of_verdict (V : Vocab d α) (k : Verity.Registration.Composite) {o : Verity.Exchange}
    (hv : V.Verdict k .sampled o) (hP : ¬ ∃ p, ParamsAt o.params p) :
    Recursive.RecVerdict V k o ∧ RecParamsAt o.params := by
  obtain ⟨hreg, hS, P, hP', ps, c₀, evs, hd, x, hx, hx1, hacc⟩ := hv
  have hPeq : P = VStar.Recursive.Sampled.protocol := by
    cases k <;> exact (Option.some.inj hP').symm
  subst hPeq
  have hfind := find_case hS hd hx hx1
  have hacc' : Properties.Composition.DecodedAccepted
      (VStar.Recursive.Sampled.protocol.onBlobs (_root_.Core.Blobs.ofArray o.sent.blobs))
      x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence := hx1 ▸ hacc
  obtain ⟨p, cm, e, hp, hc, he, hA⟩ := hacc'
  rcases p with p₀ | p₁
  · exact absurd ⟨p₀, ps, hd.1, x.2.1, hfind, hp⟩ hP
  · obtain ⟨j, hj, hpj⟩ : ∃ j, _root_.Core.canonicalJson x.2.1.params.toUTF8 = .ok j ∧
        VStar.Recursive.Sampled.paramsOfJson j = .ok (.inr p₁) := by
      cases hcj : _root_.Core.canonicalJson x.2.1.params.toUTF8 with
      | error err => rw [hcj] at hp; cases hp
      | ok j => rw [hcj] at hp; exact ⟨j, rfl, hp⟩
    exact ⟨⟨hreg, hS, x.2.1.params.toUTF8, x.2.2.commitment, x.2.2.evidence,
      ⟨ps, c₀, evs, hd, x, hx, hx1, rfl, rfl, rfl⟩, ⟨_, cm, e, hp, hc, he, hA⟩, j, p₁, hj, hpj⟩,
      ps, hd.1, x.2.1, hfind, p₁, hp⟩

/-- **An event implied on every play after `θ`**: every play after `θ`, `F` and `c` has the parameters `θ`, so an event
that implies another wherever the parameters are `θ` is at most as likely. -/
theorem prob_le_on_play (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    (σ : Verity.Game.Strategy (V.game θ F c)) {p q : Verity.Exchange → Prop} (h : ∀ o, o.params = θ → p o → q o) :
    Verity.Game.prob p (V.game θ F c) σ ≤ Verity.Game.prob q (V.game θ F c) σ := by
  have h0 : Verity.Game.prob (fun o => o.params ≠ θ) (V.game θ F c) σ = 0 :=
    prob_play_eq_zero (V.families θ) (V.signer θ) θ F c _ (fun _ hθ _ _ hn => hn hθ) σ
  calc Verity.Game.prob p (V.game θ F c) σ
      ≤ Verity.Game.prob (fun o => q o ∨ o.params ≠ θ) (V.game θ F c) σ :=
        Verity.Game.prob_mono (fun o ho => (em (o.params = θ)).elim (fun hθ => .inl (h o hθ ho)) .inr) _ σ
    _ ≤ Verity.Game.prob q (V.game θ F c) σ + Verity.Game.prob (fun o => o.params ≠ θ) (V.game θ F c) σ :=
        Verity.Game.prob_or_le _ _ _ σ
    _ = Verity.Game.prob q (V.game θ F c) σ := by rw [h0, add_zero]

/-- **The deployment at the deployed weights is `Recursive.Dep.deployed` at its own brings, record and decoders.** -/
theorem dep_deployed_self (q : Recursive.Dep) (hwE : q.wE = RecDeploy.DeployedR.wER)
    (hw : q.w = RecDeploy.DeployedR.wR) :
    Recursive.Dep.deployed q.b q.aux q.nOf q.hd q.recRest q.decI q.decO = q := by
  cases q
  dsimp only at hwE hw
  subst hwE hw
  rfl

/-- **The recursive form's session at a strategy**, at composite `k`, the parameters `θ`, the committed files `F`, the
commitment `c` and the strategy `σ`: what `Recursive.SampledProofsRecursiveAt` reads, at a deployment `RecursiveKS`
is stated at, and the vocabulary tied to it.
* `q` is at the deployed weights, with decoders meeting the list-decoding spec (`decOK`);
* `S` is the recursive session inside the causal game (the coupling); `readX` reads a witness in the recursive relation
  as the committed circuit's transcript, correct at and opened on every unit the sample takes (`read`), and `collOf`
  turns two readings that differ at a wire of units both samples take into a SHA-512 collision (`coll`);
* `K₀` covers the claim's `ε` share (`cover`) and the sample misses any `K₀` units with probability at most `δs`
  (`miss`); the records hold the run's live coins (`live`, `live-verifier`);
* the vocabulary's extractor is the session's (`extract`), `error`'s condition holds the session's per-finder premises
  (`finderCR`), and the published `δ` at the strategy's calls is at least the session's (`delta`). -/
structure RecSession (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) where
  q : Recursive.Dep
  wE : q.wE = RecDeploy.DeployedR.wER
  w : q.w = RecDeploy.DeployedR.wR
  decI : Claims.Recursion.decOK q.Bd.deploy q.decI
  decO : Claims.Recursion.decOK (RecDeploy.deployZkRec _root_.Flock.Readable.relation) q.decO
  S : Recursive.SessionIn V q k θ F c σ
  readX : Claims.Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript
  collOf : Claims.Msg → Claims.Flock.Wit → Claims.Msg → Claims.Flock.Wit → Option (Claims.Msg × Claims.Msg)
  dColl : ℕ
  K₀ : ℕ
  δs : ℝ
  cover : ∀ m : ℕ, (V.certOf θ).ε * V.units θ F < m → K₀ ≤ m
  miss : ∀ U : Finset (Fin (V.units θ F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs
  extract : V.extract θ F c σ = q.extract V S readX
  read : ∀ ξ wt, V.Registered θ F → q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
    (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧ ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g)
  coll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
    (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
    readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
    ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m'
  live : ∀ ξ ω r, Recursive.LiveCoins q.Bd.IVStar q.st (S.x ξ)
    (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen
  finderCR : V.finderCR .sampled θ F c σ ∧ V.FinderCR .sampled θ F c σ ∧ V.Unforgeable .sampled θ F c σ →
    q.FinderCR V S collOf dColl Recursive.nI₀ Recursive.vS₀
  delta : q.delta V S dColl δs 135.7 ≤ V.delta θ .sampled (V.cost θ F c σ)

/-- **The deployment's recursive sessions**: at parameters that name the recursive form, registered files, every
commitment and every strategy, a recursive session (`RecSession`). -/
def RecDeployed (V : Vocab d α) (k : Verity.Registration.Composite) : Prop :=
  ∀ θ, RecParamsAt θ → ∀ F, V.Registered θ F → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    Nonempty (RecSession V k θ F c σ)

namespace RecSession

variable {V : Vocab d α} {k : Verity.Registration.Composite} {θ c : ByteArray} {F : Array ByteArray}
  {σ : Verity.Game.Strategy (V.game θ F c)}

/-- **The recursive configuration's bound at the session**: `RecursiveKS` through
`Recursive.sampledProofsRecursive_of_ks`, at the session's deployment, with the call bounds `nI₀`, `vS₀` and
`λ = 135.7`. -/
theorem bound (R : RecSession V k θ F c σ) :
    Verity.Game.prob (fun o => Recursive.RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      R.q.errAt V R.S R.collOf R.dColl Recursive.nI₀ Recursive.vS₀ R.δs 135.7 := by
  have h := Recursive.sampledProofsRecursive_of_ks Flock.SecurityProofs.RecursiveKS R.q.b R.q.aux R.q.nOf R.q.hd
    R.q.recRest R.q.decI R.q.decO
  rw [dep_deployed_self R.q R.wE R.w] at h
  exact h R.decI R.decO V k θ F c σ R.S R.readX R.collOf R.dColl R.K₀ R.δs R.cover R.miss R.extract R.read R.coll
    R.live

/-- **The session's error is at most the vocabulary's**: where the vocabulary's premises hold, so do the session's
(`finderCR`), and the session's `δ` is at most the published one (`delta`), to which the vocabulary's error adds the
forgery and loss terms; elsewhere the vocabulary's error is `⊤`. -/
theorem errAt_le (R : RecSession V k θ F c σ) :
    R.q.errAt V R.S R.collOf R.dColl Recursive.nI₀ Recursive.vS₀ R.δs 135.7 ≤ V.error .sampled θ F c σ := by
  classical
  unfold Vocab.error Accounting.condError Recursive.Dep.errAt
  beta_reduce
  by_cases hV : V.finderCR .sampled θ F c σ ∧ V.FinderCR .sampled θ F c σ ∧ V.Unforgeable .sampled θ F c σ
  · rw [ite_eq_left hV, ite_eq_left (R.finderCR hV)]
    exact R.delta.trans (le_self_add.trans le_self_add)
  · rw [ite_eq_right hV]
    exact le_top

end RecSession

/-- **The recursive form's target from the deployment's recursive sessions**: at parameters that don't name the
one-stage form, a play the sampled verdict accepts names the recursive form (`recVerdict_of_verdict`); at registered
files there the session bounds it by the recursive configuration's error (`RecSession.bound`), which is at most the
vocabulary's (`RecSession.errAt_le`). Elsewhere no play is accepted. -/
theorem recursiveForm_of_deployed (V : Vocab d α) (k : Verity.Registration.Composite) (hR : RecDeployed V k) :
    RecursiveForm V k := by
  intro θ hP F c σ
  by_cases hreg : V.Registered θ F
  swap
  · rw [show Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ = 0
      from prob_play_eq_zero (V.families θ) (V.signer θ) θ F c _ (fun _ hθ hF _ h => hreg (hθ ▸ hF ▸ h.1.1)) σ]
    exact zero_le
  by_cases hrp : RecParamsAt θ
  swap
  · rw [show Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ = 0
      from prob_play_eq_zero (V.families θ) (V.signer θ) θ F c _
        (fun o hθ _ _ h => hrp (hθ ▸ (recVerdict_of_verdict V k h.1 (hθ ▸ hP)).2)) σ]
    exact zero_le
  obtain ⟨R⟩ := hR θ hrp F hreg c σ
  calc Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ
      ≤ Verity.Game.prob (fun o => Recursive.RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ :=
        prob_le_on_play V σ fun o hθ ho => ⟨(recVerdict_of_verdict V k ho.1 (hθ ▸ hP)).1, ho.2⟩
    _ ≤ _ := R.bound
    _ ≤ _ := R.errAt_le

/-- **Its conclusion fails for a verifier that accepts everything**: at a strategy whose extracted transcript fails
the claim, where the vocabulary's error is below one. -/
theorem recursiveForm_nontrivial (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hbad : ¬ V.Holds k θ F c σ .sampled ⟨θ, F, c, .empty, [], .empty, [], [], .empty, #[]⟩)
    (hlt : V.error .sampled θ F c σ < 1) :
    ¬ Verity.Game.prob (fun o => True ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      V.error .sampled θ F c σ := by
  intro h
  have h1 : Verity.Game.prob (fun o => True ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ = 1 :=
    prob_play_eq_one (V.families θ) (V.signer θ) θ F c (fun o => True ∧ ¬ V.Holds k θ F c σ .sampled o)
      (fun _ _ _ _ => ⟨trivial, hbad⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le h).trans_lt hlt)

/-- **Its hypotheses hold at the honest deployment where they are choices of the deployment**: `K₀ = ⌊ε n⌋ + 1`
covers the claim's `ε` share (`RecSession.cover`) wherever `ε n ≥ 0`; and where the extracted transcript is correct on
every unit and the commitment opens it (the honest commitment's case), the target's event has probability zero. -/
theorem recursiveForm_honest (V : Vocab d α) (k : Verity.Registration.Composite) :
    (∀ θ (F : Array ByteArray), 0 ≤ (V.certOf θ).ε * V.units θ F → ∀ m : ℕ, (V.certOf θ).ε * V.units θ F < m →
      ⌊(V.certOf θ).ε * V.units θ F⌋₊ + 1 ≤ m) ∧
    ∀ θ F c (σ : Verity.Game.Strategy (V.game θ F c)), (∀ u, (V.partsOf θ F).Correct (V.extract θ F c σ) u) →
      (∀ g, V.Opens θ F c g (V.extract θ F c σ g)) → 0 ≤ (V.certOf θ).ε →
      Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ = 0 := by
  refine ⟨(Recursive.sampledProofsRecursive_honest V k).1, fun θ F c σ hc ho hε => ?_⟩
  refine le_antisymm ((Verity.Game.prob_mono (q := fun _ => False)
    (fun o h => h.2 (V.holds_sampled_of_opens k θ F c σ o hc ho hε)) _ σ).trans ?_) zero_le
  exact (Verity.Game.prob_false _ σ).le

end Verity.Sampled
