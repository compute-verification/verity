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

What the claim leaves unsaid is listed at `Vocab.Holds`.
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

end Verity
