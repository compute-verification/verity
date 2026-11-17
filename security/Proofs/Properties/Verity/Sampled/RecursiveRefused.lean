module

public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.Verity.Soundness

/-! SPOT refuses the recursive form and accepts no two-stage audit, so off the one-stage form it accepts
nothing. -/

@[expose] public section

namespace Verity.Sampled

theorem twoStage_refused (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (e : VStar.Recursive.Sampled.Evidence) (bs : _root_.Core.Blobs) (pub : Spot.OneStage.Public)
    (prof : _root_.Core.Integrity) : VStar.Recursive.Sampled.checkBlobs (.inr p) c e bs ≠ .accept pub prof := by
  rcases e with e | e
  · simp [VStar.Recursive.Sampled.checkBlobs]
  · exact VStar.Recursive.SampledTwo.checkBlobs_refuses VStar.Recursive.SampledTwo.form_off p c e bs pub prof

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

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem paramsAt_of_verdict (V : Vocab d α) (k : Verity.Registration.Composite) {o : Verity.Exchange}
    (hv : V.Verdict k .sampled o) : ∃ p, ParamsAt o.params p := by
  obtain ⟨-, hS, P, hP, ps, c₀, evs, hd, x, hx, hx1, hacc⟩ := hv
  have hPeq : P = VStar.Recursive.Sampled.protocol := (Verity.component_eq_some.1 hP).2
  subst hPeq
  have hfind := find_case hS hd hx hx1
  have hacc' : Properties.Composition.DecodedAccepted
      (VStar.Recursive.Sampled.protocol.onBlobs (_root_.Core.Blobs.ofArray o.sent.blobs))
      x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence := hx1 ▸ hacc
  obtain ⟨p, c, e, hp, -, -, pub, prof, hchk⟩ := hacc'
  rcases p with p₀ | p₂
  · exact ⟨p₀, ps, hd.1, x.2.1, hfind, hp⟩
  · exact absurd hchk (twoStage_refused p₂ c e (_root_.Core.Blobs.ofArray o.sent.blobs) pub prof)

theorem recursiveForm_refused (V : Vocab d α) (k : Verity.Registration.Composite) :
    ∀ θ, (¬ ∃ p, ParamsAt θ p) → ∀ (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
      Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
        V.error .sampled θ F c σ := by
  intro θ hθ F c σ
  have h0 : Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o)
      (V.game θ F c) σ = 0 :=
    prob_play_eq_zero (V.families θ) θ F c (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o)
      (fun o ho h => hθ (by rw [← ho.1]; exact paramsAt_of_verdict V k h.1)) σ
  rw [h0]
  exact zero_le

end Verity.Sampled
