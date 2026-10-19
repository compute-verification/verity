import Proofs.Flock.Claims.KS.ZkAlign
import Proofs.Flock.Claims.KS.ZkForkClash
import Proofs.Flock.Claims.KS.ZkPadsSlot
import Proofs.Flock.Claims.KS.ZkTauPair

/-!
# Masked KS from the `τ` alignment

`zk_KS_of_slot` takes five premises at every message the masked verifier accepts at. Four of them hold at every
message: the caps' alignment (`zkSlotAlign`) and clash (`zkForkClash`), the pads' fork bound (`zkSimForkLePads`) and
the pads' view clash (`zkViewClashQP`). The fifth, the fork bound at `τ`, rests on `τ`'s alignment alone
(`zkSimForkLeTau_of_align`), so masked KS has the one premise `ZkTauAlign` (`zk_KS_of_tauAlign`).
-/

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **The fork bound at the `τ` events**, from `ZkTauAlign` (`zkSimForkLeTau_of_align`). -/
theorem zkSimForkLeTauZ_of_align {x : Claims.Msg} (ha : ZkTauAlign x) : ZkSimForkLeOn x fun _ i => IsTauZ i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with ((_ | (_ | r)) | ⟨⟩)
  · exact False.elim hi
  · exact False.elim hi
  · exact zkSimForkLeTau_of_align ha s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ ⟨r, rfl⟩
  · exact False.elim hi

end Gen

/-- **The fork bound at the `τ` events**, from `ZkTauAlign` (`zkSimForkLeTau_of_align`). -/
theorem zkSimForkLeTauZ_of_align {x : Claims.Msg} (ha : ZkTauAlign x) : ZkSimForkLeOn x fun _ i => IsTauZ i :=
  Gen.zkSimForkLeTauZ_of_align (S := S₀) (x := x) (ha := ha)

namespace Gen

variable [S : ZkSetup]

/-- **The deployed masked verifier: KS from the `τ` alignment**, at SHA-512 at the trimmed record at the deployed
relation: `zk_KS_of_slot` at `zkSlotAlign`, `zkForkClash`, `zkSimForkLePads` and `zkViewClashQP`. -/
theorem zk_KS_of_tauAlign (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkTauAlign x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_slot (fun x _ => zkSlotAlign x) (fun x _ => zkForkClash x)
    (fun x hno => zkSimForkLeTauZ_of_align (ha x hno)) (fun x _ => zkSimForkLePads x) (fun x _ => zkViewClashQP x)

end Gen

/-- **The deployed masked verifier: KS from the `τ` alignment**, at SHA-512 at the trimmed record at the deployed
relation: `zk_KS_of_slot` at `zkSlotAlign`, `zkForkClash`, `zkSimForkLePads` and `zkViewClashQP`. -/
theorem zk_KS_of_tauAlign (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkTauAlign x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_of_tauAlign (S := S₀) (ha := ha)

end FlockDeploy.KS
