import Proofs.Properties.Verity.Sampled.Record

/-!
# The sampling record's facts at one-stage v2's session

* `drawFixed`: the draw precedes the session (`DrawFixed`): every unit's session rounds are the second stage, and the
  checked units read only the draw rounds (`Claims.Sampling.drawFixed_of_first`).
* `interiorFixed`: the interior values are committed under the session's roots, which the statement fixes, so the roots
  a run shows for them are the same at fresh second-stage coins (`InteriorFixed` at `fun _ _ _ => A.roots`).
-/

namespace Verity.Sampled.Record

open Claims Claims.Sampling

variable (A : Live) (o : Own) (e : Ext)

/-- **The draw precedes the session.** -/
theorem drawFixed : DrawFixed (record A o e) := by
  refine drawFixed_of_first _ (fun _ _ _ _ => subset_refl _) fun x r r' h => ?_
  have hc : @coinsAt ((V A o).k x) r = @coinsAt ((V A o).k x) r' := coinsAt_congr fun t ht => h t ht
  refine (record_checked A o e x r).trans ?_
  rw [hc]
  exact (record_checked A o e x r').symm

/-- **The interior roots are the statement's**, at every second stage. -/
theorem interiorFixed : InteriorFixed sha512Msg (record A o e) (fun _ _ _ => A.roots) :=
  fun _ _ _ _ _ _ _ => rfl

end Verity.Sampled.Record
