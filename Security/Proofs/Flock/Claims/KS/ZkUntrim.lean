import Proofs.Flock.Claims.KS.TrimBridge
import Proofs.Flock.Claims.KS.ZkTauSlot

/-!
# KS at the untrimmed masked instance from `ExtendJ` alone

The trimmed record's KS holds with no premise (`zk_KS`), and does not give the untrimmed instance's
(`Flock.TrimBridge.hole_agree`). That one takes `zk_KS_of_kinds`'s premises and `ExtendJ` at full-width columns
(`TrimBridge.zk_KS_untrim_of_kinds`), and the premises hold at every message:

* `zkSimForkLeCapTau`: the fork bound at the caps and `τ`, from the caps' alignment and clash (`zkSlotAlign`,
  `zkForkClash`) and `τ`'s bound (`zkSimForkLeTau`);
* `zk_KS_untrim_of_extendJ`: with the pads' fork bound and view clash (`zkSimForkLePads`, `zkViewClashQP`), KS at the
  untrimmed instance at SHA-512 at the deployed relation from `ExtendJ` alone, which `ZkExtend` proves
  (`zk_extendJ`, `zk_KS_untrim`).
-/

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **The fork bound at the caps and `τ`, with no hypothesis.** -/
theorem zkSimForkLeCapTau (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsCapTauZ i :=
  zkSimForkLeOn_capTau (zkSimForkLeCap_of (zkForkAlign_of_slot (zkSlotAlign x)) (zkForkClash x)) (zkSimForkLeTau x)

end Gen

/-- **The fork bound at the caps and `τ`, with no hypothesis.** -/
theorem zkSimForkLeCapTau (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsCapTauZ i :=
  Gen.zkSimForkLeCapTau (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **KS at the untrimmed masked instance**, at SHA-512 at the deployed relation, from `ExtendJ` at full-width columns
alone (`TrimBridge.zk_KS_untrim_of_kinds`). -/
theorem zk_KS_untrim_of_extendJ
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  TrimBridge.zk_KS_untrim_of_kinds (fun x _ => zkSimForkLeCapTau x) (fun x _ => zkSimForkLePads x)
    (fun x _ => zkViewClashQP x) hext

end Gen

/-- **KS at the untrimmed masked instance**, at SHA-512 at the deployed relation, from `ExtendJ` at full-width columns
alone (`TrimBridge.zk_KS_untrim_of_kinds`). -/
theorem zk_KS_untrim_of_extendJ
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  Gen.zk_KS_untrim_of_extendJ (S := S₀) (hext := hext)

end FlockDeploy.KS
