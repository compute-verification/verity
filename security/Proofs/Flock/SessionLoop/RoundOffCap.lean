module

public import Proofs.Flock.SessionLoop.FinRepCm

/-!
# The loop's level-0 cap at a rep

* **`Loop.cap_zcBind`**: on the loop's table at rep stream `2j + r` (`Level0Src.realOf`), the level-0 cap is the switched
  source's: the tree of its level-0 leaves (`Masked.leafAt_linkIn`).
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

namespace Loop

/-- **The loop's level-0 cap at rep stream `2j + r` is the switched source's**: the recheck's tree of `linkIn`
(`Level0Src.realOf_cm0`), whose leaves are the switched cut's level-0 leaves (`Masked.leafAt_linkIn`). -/
theorem cap_zcBind {src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg}
    {w : Flock.Wit} (hS : Level0Src src x w) (ν : finS.Nonce) (t : finS.Tape) (C : Fin (refK x) → finS.Coin)
    (v : ∀ j, ((Masked.tabC Masked.PE x j).T (Masked.czExe Masked.PE x C j)).tab.Vals) (j : Fin (Masked.jOf x))
    (r : Fin 2) {tb : Tab} {hi : Array ByteArray} (h : realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    tb.cm0.cap = FlockComplete.layerOf (fun a b => _root_.Flock.Sha512.hash (a ++ b)) (Masked.lv0 x j).d
      (Masked.swSrcE x C w v t j.val).leaf0 ((Masked.lv0 x j).d - (Masked.lv0 x j).cap) := by
  rw [(Level0Src.realOf_cm0 hS ν t j r h).2, _root_.Flock.Firewall.Link.layer_eq]
  refine _root_.Flock.Firewall.Link.layerOf_congr _ _ (fun q hq => ?_) _
  rw [show Masked.swSrcE x C w v t j.val =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x C) (Masked.swCutE x C w v t) j.val from rfl,
    Masked.tabSrcCut_leaf0 _ _ j.isLt]
  exact (Masked.leafAt_linkIn x w t (Masked.czExe Masked.PE x C) v j hq).symm

end Loop

end Claims.ZKServer
