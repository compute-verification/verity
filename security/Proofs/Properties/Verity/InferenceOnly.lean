module

public import Properties.Verity.Game
public import Proofs.Properties.Remote
public import Proofs.Properties.Verity.Soundness
public import Proofs.Core.Game.Prob

/-!
# The inference-only policy's target, at every composite

The policy's claim is on the Program its check judges (`Vocab.Holds` at `.inferenceOnly`): the policy's case in the
play's messages, decoded, meets `Properties.Remote.PolicyHeld`. A composite's acceptance holds each of its cases to its
check's, and the three byte strings decode to one composition, so the case the claim reads is the case the policy's
check accepted (`policyHeld_of_same`), and what that check accepts is `PolicyHeld`
(`Properties.Remote.SecurityProofs.policyHeld`). The target's event, the composite accepting while the claim fails, is
therefore empty, and the target holds of every vocabulary, device and strategy with no premise
(`Vocab.inferenceOnlySound`). It reads no coin and no committed file.

The claim is not empty of content on an accepted play: the policy's component accepting gives the case the claim reads
(`policyCase_of_component`), so there `PolicyHeld` holds of a case that exists.

This is the clause of the case as sent. The link from that case to the committed files is two checks the verifier
lacks (the case's wires are the committed record's wires, and the case's unit is the committed `Γ`'s unit), which
`Vocab.Holds` names with what else the claim leaves unsaid. Onsite's soundness keeps this target as a hypothesis.
Remote's (`remote_soundness_of_subprotocols`, here) applies it, and takes in its place the two statements those checks
would establish (`Vocab.WiresSound`, `GateARow`), which no clause of its conclusion reads and nothing proves: the
union bound at remote's three subprotocols, sampled proofs' and PoUW's targets as given and the policy's as proved.

Its witnesses are at the idle deployment (`Properties.Verity.Idle`): the two targets and the wires' statement hold at
`Idle.unbounded`, where every error is `⊤` (`remote_soundness_of_subprotocols_honest`), and its conclusion fails at
`Idle.exact` for a verifier that accepts everything (`remote_soundness_of_subprotocols_nontrivial`). `GateARow` reads
no deployment, so no instance concludes it: it holds or fails once, for every file.
-/

@[expose] public section

namespace Verity

open Accounting Core

/-- **A composite's acceptance holds the policy's case to its check's**: on what composite `k` accepted as a `same`
composition, every decoding of the policy's case meets `PolicyHeld`. -/
theorem policyHeld_of_same {k : Verity.Registration.Composite} {s : Sent} (h : SameAccepts k s)
    {p : NciCheck.Protocol.Params} {r : NciCheck.Protocol.Commitment} {e : NciCheck.Protocol.Evidence}
    (hc : PolicyCase k s p r e) : Properties.Remote.PolicyHeld p r e := by
  obtain ⟨ps, c, ev, ⟨h1, h2, h3⟩, pub, prof, ⟨⟨-, hcases, -⟩, -⟩, -⟩ := h
  obtain ⟨ps', c', ev', ⟨h1', h2', h3'⟩, ⟨P, n, cs⟩, hx, hx1, hp, hr, he⟩ := hc
  obtain rfl : ps' = ps := Except.ok.inj (h1'.symm.trans h1)
  obtain rfl : ev' = ev := Except.ok.inj (h3'.symm.trans h3)
  have hx1' : P = NciCheck.Protocol.protocol := hx1
  subst hx1'
  obtain ⟨p', r', e', hp', hr', he', pub', prof', hacc⟩ := hcases _ hx
  have hp'' : (canonicalJson n.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok p' := hp'
  have hr'' : NciCheck.Protocol.commitmentOf cs.commitment = .ok r' := hr'
  have he'' : NciCheck.Protocol.evidenceOf cs.evidence = .ok e' := he'
  obtain rfl : p' = p := Except.ok.inj (hp''.symm.trans hp)
  obtain rfl : r' = r := Except.ok.inj (hr''.symm.trans hr)
  obtain rfl : e' = e := Except.ok.inj (he''.symm.trans he)
  exact Properties.Remote.SecurityProofs.policyHeld (pub := pub') (prof := prof') hacc

/-- **The claim reads a case that exists**: where the policy's component accepts in what the prover sent composite `k`,
the policy's case decodes. -/
theorem policyCase_of_component {k : Verity.Registration.Composite} {s : Sent}
    (h : ComponentAccepts k .inferenceOnly s) : ∃ p r e, PolicyCase k s p r e := by
  obtain ⟨Q, hQ, ps, c, ev, hdec, ⟨P, n, cs⟩, hx, hx1, hacc⟩ := h
  have hQ' : Q = NciCheck.Protocol.protocol := by
    cases k <;> exact (Option.some.inj hQ).symm
  subst hQ'
  have hx1' : P = NciCheck.Protocol.protocol := hx1
  subst hx1'
  obtain ⟨p, r, e, hp, hr, he, -⟩ := hacc
  exact ⟨p, r, e, ps, c, ev, hdec, _, hx, rfl, hp, hr, he⟩

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

/-- **The inference-only policy's target, at every composite**: for any names `V` and composite `k`, the probability
that `k`'s verifier accepts registered files while the policy's case fails `PolicyHeld` is `0`, so it is within the
policy's error. -/
theorem inferenceOnlySound (k : Verity.Registration.Composite) : V.InferenceOnlySound k := by
  intro θ F c σ
  calc Game.prob (fun o => V.Verdict k .inferenceOnly o ∧ ¬ V.Holds k θ F c σ .inferenceOnly o) (V.game θ F c) σ
      ≤ Game.prob (fun _ => False) (V.game θ F c) σ :=
        Game.prob_mono (fun _ ⟨⟨_, hsame, _⟩, hfail⟩ => hfail fun _ _ _ hc => policyHeld_of_same hsame hc) _ σ
    _ = 0 := Game.prob_false _ σ
    _ ≤ V.error .inferenceOnly θ F c σ := zero_le

end Vocab

theorem remote_soundness_of_subprotocols : RemoteSoundness := by
  intro ι α _ d V hS hW _ _
  refine soundness_at_of_targets ι α d V .remote fun a ha => ?_
  cases a
  · exact hS
  · exact hW
  · exact V.inferenceOnlySound .remote
  all_goals exact absurd (Finset.mem_filter.1 ha).2 Bool.false_ne_true

/-- **`RemoteSoundness`'s honest lemma**: its two targets and the wires' statement hold at the idle deployment's
compiled remote composite. -/
theorem remote_soundness_of_subprotocols_honest :
    Idle.unbounded.SpotSound .remote ∧ Idle.unbounded.PouwSound .remote ∧ Idle.unbounded.WiresSound .remote :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.wiresSound_unbounded _⟩

/-- **`RemoteSoundness`'s nontrivial lemma**: its conclusion fails at remote for a verifier that accepts everything, at
the idle deployment whose errors are all `0`. -/
theorem remote_soundness_of_subprotocols_nontrivial : ¬ Idle.exact.SoundnessFor .remote fun _ => True :=
  Idle.nontrivial_at .remote (Finset.mem_filter.2 ⟨Finset.mem_univ _, rfl⟩)

end Verity
