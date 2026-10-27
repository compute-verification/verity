import Proofs.Properties.Verity.SampledAccept
import Proofs.Properties.Challenger

/-!
# v2's accept runs the round coins under the registered key

Sampled proofs' compiled component, the one-stage audit v2, accepts only where `roundsWhy` passes
(`OneStageZk.roundsWhy_of_checkBlobs`), and `roundsWhy` passes only where v2's parameters name a challenger key and a
window root and `roundCoins` returns under that key on that window (`OneStageZk.roundCoins_of_roundsWhy`).

v2's parameters are the registration's entry for v2's id: onsite composes one component of that id, the parameters name
the components in order (`Properties.Composition.Fits`), so the entry sampled proofs' case reads is the one `find?`
names (`OnsiteIds.find_of_unique`), and its challenger is the key v2's compiled check reads from the registered
parameters (`Properties.Challenger.keyOfParams`), the device's registered key at `θ` (`registeredKey`). So sampled proofs' verdict
runs `roundCoins` under the device's registered key (`Vocab.roundCoins_at_verdict`).
-/

open Lean (Json)

namespace SampledProofs.OneStageZk

/-- **`roundsWhy` passes only on a run of the round coins**: where it names no reason, v2's parameters name a
challenger key and label and its verifier's window root, and `roundCoins` returns under that key on that window. -/
theorem roundCoins_of_roundsWhy {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    (h : roundsWhy p c e bs = none) :
    ∃ key label w us order groups coins ξs, p.base.challenger = some (key, label) ∧
      p.base.verifier.windowRoot = some w ∧ Challenger.roundCoins key w us order groups coins = .ok ξs := by
  unfold roundsWhy at h
  split at h
  · rename_i key label w hc hw
    dsimp only at h
    split at h
    · rename_i hr
      simp only [bind, Except.bind, pure, Except.pure] at hr
      repeat' split at hr
      all_goals first
        | (cases hr; done)
        | (rename_i hq _ _ _; exact ⟨key, label, w, _, _, _, _, _, hc, hw, hq⟩)
    · cases h
  · cases h

/-- **v2's accept passes `roundsWhy`**: `checkBlobs` accepts only where `roundsWhy` names no reason. -/
theorem roundsWhy_of_checkBlobs {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : checkBlobs p c e bs = .accept pub prof) :
    roundsWhy p c e bs = none := by
  unfold checkBlobs at h
  repeat' split at h
  all_goals first | assumption | (cases h; done) | (rename_i hn; exact absurd h (hn _ _))

end SampledProofs.OneStageZk

namespace Verity.OnsiteIds

/-- In a composition whose parameters name its components in order, each triple's parameters carry its component's id. -/
theorem id_of_triples : ∀ {cs : List Core.Protocol} {ps : List Core.Composition.Named}
    {ev : List Core.Composition.Case}, ps.map (·.id) = cs.map (·.id) →
    ∀ x ∈ Properties.Composition.triples cs ps ev, x.2.1.id = x.1.id
  | [], _, _, _, x, hx => by simp [Properties.Composition.triples] at hx
  | _ :: _, [], _, _, x, hx => by simp [Properties.Composition.triples] at hx
  | _ :: _, _ :: _, [], _, x, hx => by simp [Properties.Composition.triples] at hx
  | P :: cs, n :: ps, k :: ev, h, x, hx => by
    simp only [List.map_cons, List.cons.injEq] at h
    simp only [Properties.Composition.triples, List.zip_cons_cons, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact h.1
    · exact id_of_triples h.2 x hx

/-- The entry `n` of `ps` is the one `find?` names for its id when no other entry carries that id. -/
theorem find_of_unique {ps : List Core.Composition.Named} {n : Core.Composition.Named} {i : String} (hn : n ∈ ps) (hi : n.id = i)
    (hc : (ps.map (·.id)).count i = 1) : ps.find? (·.id == i) = some n := by
  obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1
    (List.find?_isSome (p := fun x : Core.Composition.Named => x.id == i) |>.2 ⟨n, hn, beq_iff_eq.2 hi⟩)
  have hym := List.mem_of_find?_eq_some hy
  have hyp : y.id = i := beq_iff_eq.1 (List.find?_some (p := fun x : Core.Composition.Named => x.id == i) hy)
  rw [List.count, List.countP_map, List.countP_eq_length_filter] at hc
  obtain ⟨z, hz⟩ := List.length_eq_one_iff.1 hc
  have h1 : n ∈ ps.filter ((· == i) ∘ (·.id)) := List.mem_filter.2 ⟨hn, beq_iff_eq.2 hi⟩
  have h2 : y ∈ ps.filter ((· == i) ∘ (·.id)) := List.mem_filter.2 ⟨hym, beq_iff_eq.2 hyp⟩
  rw [hz, List.mem_singleton] at h1 h2
  rw [hy, h1, h2]

/-- Onsite composes one component of v2's id. -/
theorem onsite_count (bs : Core.Blobs) :
    ((Core.Onsite.components.map (·.onBlobs bs)).map (·.id)).count SampledProofs.OneStageZk.PROTOCOL = 1 := by
  rw [List.map_map]
  have : ((fun x : Core.Protocol => x.id) ∘ fun x : Core.BlobProtocol => x.onBlobs bs) = (·.id) := rfl
  rw [this]
  decide

end Verity.OnsiteIds

namespace Verity

open Accounting Core

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **v2's accept runs the round coins under the registered key**: where sampled proofs' verdict accepts the play that
ended with `o` after `θ`, v2 accepts its case, whose parameters name a challenger key, with the LMS parameters and
public key of the device's registered key at `θ` (`registeredKey`), and a window root, and `roundCoins` returns under
that key on that window. -/
theorem Vocab.roundCoins_at_verdict (V : Vocab d α) {θ : ByteArray} {o : Outcome} (hθ : o.params = θ)
    (hv : V.Verdict .sampled o) :
    ∃ p c e pub prof, SampledCase o.sent p c e ∧
      SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof ∧
      ∃ key label w us order groups coins ξs, p.base.challenger = some (key, label) ∧
        p.base.verifier.windowRoot = some w ∧ key.params = (registeredKey θ).1 ∧ key.pub = (registeredKey θ).2 ∧
        SampledProofs.Challenger.roundCoins key w us order groups coins = .ok ξs := by
  obtain ⟨-, hra, hca⟩ := hv
  obtain ⟨p, c, e, pub, prof, hcase, hacc⟩ := componentAccepts_sampled hca
  refine ⟨p, c, e, pub, prof, hcase, hacc, ?_⟩
  obtain ⟨key, label, w, us, order, groups, coins, ξs, hc, hw, hr⟩ :=
    SampledProofs.OneStageZk.roundCoins_of_roundsWhy (SampledProofs.OneStageZk.roundsWhy_of_checkBlobs hacc)
  obtain ⟨n, ps, b, dg, k, ev, hD, nm, cs, hx, hp, -, -⟩ := hcase
  obtain ⟨n', ps', b', d', k', ev', hD', _, _, hreg⟩ := hra
  have hps : ps' = ps := congrArg Prod.snd (Except.ok.inj (hD'.1.symm.trans hD.1))
  obtain ⟨hfit, -⟩ := hreg.2.2.2.2.2.2
  rw [hps] at hfit
  have hn : nm ∈ ps := (List.of_mem_zip (List.of_mem_zip hx).2).1
  have hid : nm.id = SampledProofs.OneStageZk.PROTOCOL := OnsiteIds.id_of_triples hfit.1 _ hx
  have hfind := OnsiteIds.find_of_unique hn hid (by rw [hfit.1]; exact OnsiteIds.onsite_count _)
  have hkey : Properties.Challenger.keyOfParams ps = some key :=
    Properties.Challenger.keyOfParams_eq_some.2 ⟨nm, p, label, hfind, hp, hc⟩
  have hd : (Core.canonicalJson θ >>= Core.Composition.decodeRegisteredParams) = .ok (n, ps) := by
    rw [← hθ]
    exact hD.1
  have hk : registeredKey θ = Properties.Challenger.lmsKey (some key) := by
    unfold registeredKey
    rw [hd, ← hkey]
    rfl
  exact ⟨key, label, w, us, order, groups, coins, ξs, hc, hw, by simp only [hk, Properties.Challenger.lmsKey],
    by simp only [hk, Properties.Challenger.lmsKey], hr⟩

end Verity
