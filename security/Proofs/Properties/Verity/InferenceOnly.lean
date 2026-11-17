module

public import Properties.Verity.Game
public import Proofs.Properties.Remote
public import Proofs.Properties.Verity.Soundness
public import Proofs.Core.Game.Prob

/-! The inference-only policy's target at every composite where v0 decides it; at remote's plain sibling
`InferenceOnlySound .plain` stays open. -/

@[expose] public section

namespace Verity

open ResourceAccounting Core

theorem policyHeld_of_same {k : Verity.Registration.Composite} {s : Sent} (h : SameAccepts k s)
    {p : InferenceOnlyCheck.Protocol.Params} {r : InferenceOnlyCheck.Protocol.Commitment} {e : InferenceOnlyCheck.Protocol.Evidence}
    (hc : PolicyCase k s p r e) : Properties.Remote.PolicyHeld p r e := by
  obtain ⟨ps, c, ev, ⟨h1, h2, h3⟩, pub, prof, ⟨⟨-, hcases, -⟩, -⟩, -⟩ := h
  obtain ⟨ps', c', ev', ⟨h1', h2', h3'⟩, ⟨P, n, cs⟩, hx, hx1, hp, hr, he⟩ := hc
  obtain rfl : ps' = ps := Except.ok.inj (h1'.symm.trans h1)
  obtain rfl : ev' = ev := Except.ok.inj (h3'.symm.trans h3)
  have hx1' : P = InferenceOnlyCheck.Protocol.protocol := hx1
  subst hx1'
  obtain ⟨p', r', e', hp', hr', he', pub', prof', hacc⟩ := hcases _ hx
  have hp'' : (canonicalJson n.params.toUTF8 >>= InferenceOnlyCheck.Protocol.paramsOf) = .ok p' := hp'
  have hr'' : InferenceOnlyCheck.Protocol.commitmentOf cs.commitment = .ok r' := hr'
  have he'' : InferenceOnlyCheck.Protocol.evidenceOf cs.evidence = .ok e' := he'
  obtain rfl : p' = p := Except.ok.inj (hp''.symm.trans hp)
  obtain rfl : r' = r := Except.ok.inj (hr''.symm.trans hr)
  obtain rfl : e' = e := Except.ok.inj (he''.symm.trans he)
  exact Properties.Remote.SecurityProofs.policyHeld (pub := pub') (prof := prof') hacc

theorem policyCase_of_component {k : Verity.Registration.Composite} {s : Sent} (hk : rowsId ∉ k)
    (h : ComponentAccepts k .inferenceOnly s) : ∃ p r e, PolicyCase k s p r e := by
  obtain ⟨Q, hQ, ps, c, ev, hdec, ⟨P, n, cs⟩, hx, hx1, hacc⟩ := h
  have hQ' : Q = InferenceOnlyCheck.Protocol.protocol := by
    rw [(component_eq_some.1 hQ).2]
    simp only [Subprotocol.protocolAt, hk, ↓reduceIte]
  subst hQ'
  have hx1' : P = InferenceOnlyCheck.Protocol.protocol := hx1
  subst hx1'
  obtain ⟨p, r, e, hp, hr, he, -⟩ := hacc
  exact ⟨p, r, e, ps, c, ev, hdec, _, hx, rfl, hp, hr, he⟩

theorem v0_not_mem_plain : InferenceOnlyCheck.Protocol.protocolId ∉ Verity.Registration.Composite.plain := by decide

theorem v0_not_mem_componentsOf_plain : ∀ P ∈ componentsOf .plain, P.id ≠ InferenceOnlyCheck.Protocol.protocolId := by
  rw [componentsOf_plain]
  decide

/-- **`Vocab.inferenceOnlySound` doesn't reach remote's plain sibling**: its hypotheses fail there, so
`InferenceOnlySound .plain` stays open. -/
theorem not_inferenceOnlySound_hyp_plain :
    ¬ (InferenceOnlyCheck.Protocol.protocolId ∈ Verity.Registration.Composite.plain ∧
      rowsId ∉ Verity.Registration.Composite.plain) :=
  fun h => v0_not_mem_plain h.1

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

theorem inferenceOnlySound (k : Verity.Registration.Composite) (hk : InferenceOnlyCheck.Protocol.protocolId ∈ k)
    (hr : rowsId ∉ k) : V.InferenceOnlySound k := by
  intro θ F c σ
  calc Game.prob (fun o => V.Verdict k .inferenceOnly o ∧ ¬ V.Holds k θ F c σ .inferenceOnly o) (V.game θ F c) σ
      ≤ Game.prob (fun _ => False) (V.game θ F c) σ :=
        Game.prob_mono (fun _ ⟨⟨_, hsame, _⟩, hfail⟩ => hfail <| by
          simp only [Holds, hk, hr, not_false_eq_true, and_self, ↓reduceIte]
          exact fun _ _ _ hc => policyHeld_of_same hsame hc) _ σ
    _ = 0 := Game.prob_false _ σ
    _ ≤ V.error .inferenceOnly θ F c σ := zero_le

theorem holds_plain (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (o : Exchange) : V.Holds .plain θ F c σ .inferenceOnly o = rowsHeld .plain o.sent := by
  simp only [Holds, v0_not_mem_plain, false_and, ↓reduceIte]

end Vocab

theorem remote_soundness_of_subprotocols : RemoteSoundness :=
  fun ι α _ d V hS hW _ _ =>
    soundness_at_of_targets ι α d V .remote (V.targets_remote hS hW (V.inferenceOnlySound .remote (by decide) (by decide)))

/-- **`RemoteSoundness`'s honest lemma**: its two targets and the wires' statement hold at the idle deployment's
compiled remote composite. -/
theorem remote_soundness_of_subprotocols_honest :
    Idle.unbounded.SpotSound .remote ∧ Idle.unbounded.PouwSound .remote ∧ Idle.unbounded.WiresSound .remote :=
  ⟨Idle.target_unbounded _ _, Idle.target_unbounded _ _, Idle.wiresSound_unbounded _⟩

/-- **`RemoteSoundness`'s nontrivial lemma**: its conclusion fails at remote for a verifier that accepts everything, at
the idle deployment whose errors are all `0`. -/
theorem remote_soundness_of_subprotocols_nontrivial : ¬ Idle.exact.SoundnessFor .remote fun _ => True :=
  Idle.nontrivial_at .remote (mem_subprotocols.2 (by decide))

end Verity
