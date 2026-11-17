module

public import Properties.Verity.Game
public import Proofs.Properties.InferenceOnly.RecordRowsComplies
public import Proofs.Properties.Verity.Soundness.Composite
public import Proofs.Core.Game.Prob

/-!
# The inference-only policy's target where the rows check decides it

A composite's acceptance holds each of its cases to its check's, and the three byte strings decode to one composition,
so a rows case the clause reads is a case the rows check accepted (`rowsHeld_of_same`), and what that check accepts is
`Properties.Remote.RowsHeld` (`Properties.Remote.SecurityProofs.rowsHeld`, from `RowsComplies`). The policy's target is
therefore proved, with no premise, at every composite where v0 doesn't decide the policy (`Vocab.inferenceOnlySound_rows`),
remote's plain sibling among them. Where the composite lists the rows check, the policy's component accepting gives the
case the clause reads (`rowsCase_of_component`).
-/

@[expose] public section

namespace Properties.Remote.SecurityProofs

/-- **What the rows check accepts is `RowsHeld`.** -/
theorem rowsHeld {p : InferenceOnly.RowsProtocol.Params} {c : InferenceOnlyVerifier.Protocol.Commitment}
    {e : InferenceOnly.RowsProtocol.Evidence} {bs : Core.Blobs} {pub : Unit} {prof : Lean.Json}
    (h : InferenceOnly.RowsProtocol.checkBlobs p c e bs = .accept pub prof) : RowsHeld p c e bs :=
  InferenceOnlyVerifier.Properties.Record.SecurityProofs.RowsComplies p c e bs pub prof h

end Properties.Remote.SecurityProofs

namespace Verity

open ResourceAccounting Core

/-- **A composite's acceptance holds the rows check's case to its check's**: on what composite `k` accepted as a `same`
composition, every decoding of the rows check's case is `RowsHeld` on the blobs sent. -/
theorem rowsHeld_of_same {k : Verity.Registration.Composite} {s : Sent} (h : SameAccepts k s)
    {p : InferenceOnly.RowsProtocol.Params} {r : InferenceOnlyVerifier.Protocol.Commitment}
    {e : InferenceOnly.RowsProtocol.Evidence} (hc : RowsCase k s p r e) :
    Properties.Remote.RowsHeld p r e (Blobs.ofArray s.blobs) := by
  obtain ⟨ps, c, ev, ⟨h1, h2, h3⟩, pub, prof, ⟨⟨-, hcases, -⟩, -⟩, -⟩ := h
  obtain ⟨ps', c', ev', ⟨h1', h2', h3'⟩, ⟨P, n, cs⟩, hx, hx1, hp, hr, he⟩ := hc
  obtain rfl : ps' = ps := Except.ok.inj (h1'.symm.trans h1)
  obtain rfl : ev' = ev := Except.ok.inj (h3'.symm.trans h3)
  have hx1' : P = InferenceOnly.RowsProtocol.protocol.onBlobs (Blobs.ofArray s.blobs) := hx1
  subst hx1'
  obtain ⟨p', r', e', hp', hr', he', pub', prof', hacc⟩ := hcases _ hx
  have hp'' : (canonicalJson n.params.toUTF8 >>= InferenceOnly.RowsProtocol.paramsOf) = .ok p' := hp'
  have hr'' : InferenceOnlyVerifier.Protocol.commitmentOf cs.commitment = .ok r' := hr'
  have he'' : InferenceOnly.RowsProtocol.evidenceOf cs.evidence = .ok e' := he'
  obtain rfl : p' = p := Except.ok.inj (hp''.symm.trans hp)
  obtain rfl : r' = r := Except.ok.inj (hr''.symm.trans hr)
  obtain rfl : e' = e := Except.ok.inj (he''.symm.trans he)
  exact Properties.Remote.SecurityProofs.rowsHeld (pub := pub') (prof := prof') hacc

/-- **The clause reads a case that exists**: where the policy's component accepts in what the prover sent composite
`k`, which lists the rows check, the rows check's case decodes. -/
theorem rowsCase_of_component {k : Verity.Registration.Composite} {s : Sent} (hk : rowsId ∈ k) (h : ComponentAccepts k .inferenceOnly s) :
    ∃ p r e, RowsCase k s p r e := by
  obtain ⟨Q, hQ, ps, c, ev, hdec, ⟨P, n, cs⟩, hx, hx1, hacc⟩ := h
  have hQ' : Q = InferenceOnly.RowsProtocol.protocol := by
    rw [(component_eq_some.1 hQ).2]
    simp only [Subprotocol.protocolAt, hk, ↓reduceIte, rowsCheck]
  subst hQ'
  have hx1' : P = InferenceOnly.RowsProtocol.protocol.onBlobs (Blobs.ofArray s.blobs) := hx1
  subst hx1'
  obtain ⟨p, r, e, hp, hr, he, -⟩ := hacc
  exact ⟨p, r, e, ps, c, ev, hdec, _, hx, rfl, hp, hr, he⟩

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

/-- **The inference-only policy's target, at every composite where v0 doesn't decide it**: for any names `V` and
composite `k` that lists the rows check or no v0, the probability that `k`'s verifier accepts registered files while a
rows case in what was sent fails `RowsHeld` is `0`, so it is within the policy's error. -/
theorem inferenceOnlySound_rows (k : Verity.Registration.Composite)
    (h : ¬ (InferenceOnlyVerifier.Protocol.protocolId ∈ k ∧ rowsId ∉ k)) : V.InferenceOnlySound k := by
  intro θ F c σ
  calc Game.prob (fun o => V.Verdict k .inferenceOnly o ∧ ¬ V.Holds k θ F c σ .inferenceOnly o) (V.game θ F c) σ
      ≤ Game.prob (fun _ => False) (V.game θ F c) σ :=
        Game.prob_mono (fun _ ⟨⟨_, hsame, _⟩, hfail⟩ => hfail <| by
          simp only [Holds, h, ↓reduceIte]
          exact fun _ _ _ hc => rowsHeld_of_same hsame hc) _ σ
    _ = 0 := Game.prob_false _ σ
    _ ≤ V.error .inferenceOnly θ F c σ := zero_le

end Vocab

end Verity
