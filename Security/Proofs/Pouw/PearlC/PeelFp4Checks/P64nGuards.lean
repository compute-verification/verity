import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `prod-64-nvfp4`: factor chains by evaluation (`#guard`, not a proof) (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

/-! `prod-64-nvfp4`'s factor chains by evaluation (`#guard`, not a proof): its selected rows of `A′·F_B` and `B̃·F_A`, and its whole `G`. -/

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 0).1 (colF (basis p64n.k p64n.fb) q)
    (rowA sm120Nvf4 p64n 0).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.af.getD 0 [])

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 1).1 (colF (basis p64n.k p64n.fb) q)
    (rowA sm120Nvf4 p64n 1).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.af.getD 1 [])

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 2).1 (colF (basis p64n.k p64n.fb) q)
    (rowA sm120Nvf4 p64n 2).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.af.getD 2 [])

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 3).1 (colF (basis p64n.k p64n.fb) q)
    (rowA sm120Nvf4 p64n 3).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.af.getD 3 [])

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 p64n 0).1 (colF (basis p64n.k p64n.fa) q)
    (rowB sm120Nvf4 p64n 0).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.bf.getD 0 [])

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 p64n 1).1 (colF (basis p64n.k p64n.fa) q)
    (rowB sm120Nvf4 p64n 1).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.bf.getD 1 [])

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 p64n 2).1 (colF (basis p64n.k p64n.fa) q)
    (rowB sm120Nvf4 p64n 2).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.bf.getD 2 [])

#guard (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 p64n 3).1 (colF (basis p64n.k p64n.fa) q)
    (rowB sm120Nvf4 p64n 3).2 (onesScales sm120Nvf4 p64n.k)).mapM id = some (p64n.bf.getD 3 [])

#guard (List.ofFn fun q => (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis p64n.k p64n.fb) q) (colF (basis p64n.k p64n.fa) q')
    (onesScales sm120Nvf4 p64n.k) (onesScales sm120Nvf4 p64n.k)).mapM id).mapM id = some p64n.g

end Pouw.PearlC.PeelFp4Checks
