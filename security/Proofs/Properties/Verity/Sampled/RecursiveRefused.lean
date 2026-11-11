module

public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.Verity.Soundness

/-!
# SPOT refuses the recursive form, so off the one-stage form it accepts nothing

SPOT's compiled component (`VStar.Recursive.Sampled.protocol`) refuses parameters naming the recursive form by name
(`VStar.Recursive.Sampled.RECURSIVE_REFUSED`), before their rest is read, so its parameters never decode to that form
(`paramsOfJson_ne_inr`).

* **A sampled verdict names the one-stage form** (`paramsAt_of_verdict`): wherever composite `k`'s sampled verdict holds
  of what the prover sent, SPOT's case decodes in the one-stage form, under the parameters `ParamsAt` reads,
  since the components' ids are distinct (`find_case`).
* **Off the one-stage form, the event has probability zero** (`recursiveForm_refused`): at parameters `θ` that name no
  one-stage parameters, no play after `θ` meets the sampled verdict, so the probability that it holds while its claim
  fails is `0`, at most the error at any strategy, for every vocabulary and composite.
-/

@[expose] public section

namespace Verity.Sampled

/-- **SPOT's parameters never decode to the recursive form**: `paramsOfJson` refuses it by name
(`VStar.Recursive.Sampled.RECURSIVE_REFUSED`). -/
theorem paramsOfJson_ne_inr (j : Lean.Json) (p : VStar.Recursive.Params) :
    VStar.Recursive.Sampled.paramsOfJson j ≠ .ok (.inr p) := by
  intro h
  unfold VStar.Recursive.Sampled.paramsOfJson at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals cases h

/-- **SPOT's case is the one its id names**: in a composition composite `k` accepted, the parameters the composition
names for SPOT's id are those of SPOT's case, since the components' ids are distinct. -/
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

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **A sampled verdict names the one-stage form**: wherever composite `k`'s sampled verdict holds of what the prover
sent, its parameters name SPOT's one-stage form (`ParamsAt`), since its component refuses every other
(`paramsOfJson_ne_inr`). -/
theorem paramsAt_of_verdict (V : Vocab d α) (k : Verity.Registration.Composite) {o : Verity.Exchange}
    (hv : V.Verdict k .sampled o) : ∃ p, ParamsAt o.params p := by
  obtain ⟨-, hS, P, hP, ps, c₀, evs, hd, x, hx, hx1, hacc⟩ := hv
  have hPeq : P = VStar.Recursive.Sampled.protocol := by
    cases k <;> exact (Option.some.inj hP).symm
  subst hPeq
  have hfind := find_case hS hd hx hx1
  have hacc' : Properties.Composition.DecodedAccepted
      (VStar.Recursive.Sampled.protocol.onBlobs (_root_.Core.Blobs.ofArray o.sent.blobs))
      x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence := hx1 ▸ hacc
  obtain ⟨p, -, -, hp, -, -, -⟩ := hacc'
  rcases p with p₀ | p₁
  · exact ⟨p₀, ps, hd.1, x.2.1, hfind, hp⟩
  · exfalso
    cases hcj : _root_.Core.canonicalJson x.2.1.params.toUTF8 with
    | error err => rw [hcj] at hp; cases hp
    | ok j => rw [hcj] at hp; exact paramsOfJson_ne_inr j p₁ hp

/-- **Off the one-stage form, SPOT accepts nothing**: at parameters `θ` that name no one-stage parameters
(`ParamsAt`), the probability that composite `k`'s sampled verdict holds while its claim fails is `0`, so at most the
error at any strategy, since no play after `θ` meets the verdict (`paramsAt_of_verdict`). -/
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
