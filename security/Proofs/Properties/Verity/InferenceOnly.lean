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
therefore empty, and the target holds of every vocabulary, device and strategy with no premise, at every composite
that lists v0 (`Vocab.inferenceOnlySound`). It reads no coin and no committed file.

Remote's plain sibling lists no v0 (`v0_not_mem_plain`, `v0_not_mem_componentsOf_plain`): there v0's clause would read
no case and hold of every play, so the policy's claim is `rowsHeld` (`Vocab.holds_plain`), which nothing states, and
`InferenceOnlySound .plain` stays the hypothesis `Verity.PlainSoundness` takes until the rows check's guarantee.

The claim is not empty of content on an accepted play: the policy's component accepting gives the case the claim reads
(`policyCase_of_component`), so there `PolicyHeld` holds of a case that exists.

This is the clause of the case as sent. The link from that case to the committed files is two checks the verifier
lacks (the case's wires are the committed record's wires, and the case's unit is the committed `Γ`'s unit), which
`Vocab.Holds` names with what else the claim leaves unsaid; until both exist, the composites' soundness keeps this
target in its trust gap.
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
  have hQ' : Q = NciCheck.Protocol.protocol := (component_eq_some.1 hQ).2
  subst hQ'
  have hx1' : P = NciCheck.Protocol.protocol := hx1
  subst hx1'
  obtain ⟨p, r, e, hp, hr, he, -⟩ := hacc
  exact ⟨p, r, e, ps, c, ev, hdec, _, hx, rfl, hp, hr, he⟩

/-- **Remote's plain sibling lists no v0**, so `Vocab.inferenceOnlySound`'s hypothesis fails there. -/
theorem v0_not_mem_plain : NciCheck.Protocol.protocolId ∉ Verity.Registration.Composite.plain := by decide

/-- **Remote's plain sibling has no v0 component**: no case of its play is v0's. -/
theorem v0_not_mem_componentsOf_plain : NciCheck.Protocol.protocol ∉ componentsOf .plain := by
  intro h
  have h' := List.mem_map_of_mem (f := BlobProtocol.id) h
  rw [componentsOf_plain] at h'
  revert h'
  decide

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

/-- **The inference-only policy's target, at every composite that lists v0**: for any names `V` and composite `k` that
lists v0, the probability that `k`'s verifier accepts registered files while the policy's case fails `PolicyHeld` is
`0`, so it is within the policy's error. -/
theorem inferenceOnlySound (k : Verity.Registration.Composite) (hk : NciCheck.Protocol.protocolId ∈ k) :
    V.InferenceOnlySound k := by
  intro θ F c σ
  calc Game.prob (fun o => V.Verdict k .inferenceOnly o ∧ ¬ V.Holds k θ F c σ .inferenceOnly o) (V.game θ F c) σ
      ≤ Game.prob (fun _ => False) (V.game θ F c) σ :=
        Game.prob_mono (fun _ ⟨⟨_, hsame, _⟩, hfail⟩ => hfail <| by
          simp only [Holds, if_pos hk]
          exact fun _ _ _ hc => policyHeld_of_same hsame hc) _ σ
    _ = 0 := Game.prob_false _ σ
    _ ≤ V.error .inferenceOnly θ F c σ := zero_le

/-- **The policy's claim at remote's plain sibling is `rowsHeld`'s**, not v0's clause, which `inferenceOnlySound`
reads. -/
theorem holds_plain (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (o : Exchange) : V.Holds .plain θ F c σ .inferenceOnly o = rowsHeld .plain o.sent := by
  simp only [Holds, if_neg v0_not_mem_plain]

end Vocab

end Verity
