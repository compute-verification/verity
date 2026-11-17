module

public import Proofs.Properties.Verity.Sampled.Record

/-! The sampling record's facts at one-stage v2's session. -/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling

variable (A : Live) (o : Own) (e : Ext)

theorem record_parse_eq (x : Msg) :
    (record A o e).parse x = if x = xOf A o then some (stmtA A) else none := rfl

theorem record_parse : (record A o e).parse (xOf A o) = some (stmtA A) :=
  (record_parse_eq A o e (xOf A o)).trans (ite_eq_left rfl)

theorem sampleFixed : SampleFixed (record A o e) := by
  refine sampleFixed_of_first _ (fun _ _ _ _ => subset_refl _) fun x r r' h => ?_
  have hc : @coinsAt ((V A o).k x) r = @coinsAt ((V A o).k x) r' := coinsAt_congr fun t ht => h t ht
  refine (record_checked A o e x r).trans ?_
  rw [hc]
  exact (record_checked A o e x r').symm

theorem interiorFixed : InteriorFixed sha512Msg (record A o e) (fun _ _ _ => A.roots) :=
  fun _ _ _ _ _ _ _ => rfl

theorem refuses_of_inScope (H : Msg → Msg) (h : InScope (partA A) (stmtA A)) : Refuses H (record A o e) := by
  intro x Ω P ω r hacc
  obtain ⟨as, cs, has⟩ := Claims.Sampling.run_acc_next H P (record A o e).V x ω r hacc
  by_cases hx : x = xOf A o
  · subst hx
    exact ⟨stmtA A, partA A, record_parse A o e, rfl, h⟩
  · have hn : (record A o e).V.next x as cs = .ret .rej := ite_eq_right hx
    rw [hn] at has
    cases has

end Verity.Sampled.Record
