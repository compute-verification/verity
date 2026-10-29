/-
`Flock.Registered.domain`, which `RecVStar.topEntry` derives R's registered domain with, is `SampledProofs.Registered.domain`:
the same function of the program, the value, its words, leaves and word width.
-/
import Flock.Registered
import SampledProofs.Registered

namespace SampledProofs.DomainAgrees

theorem domain_agrees : @Flock.Registered.domain = @SampledProofs.Registered.domain := rfl

private def same (program value : String) (words leaves wordBits : Nat) : Bool :=
  match Flock.Registered.domain program value words leaves wordBits,
      SampledProofs.Registered.domain program value words leaves wordBits with
  | .ok a, .ok b => a == b
  | .error a, .error b => a == b
  | _, _ => false

private def PROGRAM : String :=
  "21aaf3e23665ab7d98ad7485fe35db6179187966e431b9713ef921d2008074c503865a8558f74cb1eaa7aaa20da9b62e7ec6d5f861c913da44e01c2c3dd2514c"

#guard same PROGRAM "stacked" 64 128 16
#guard same PROGRAM "weights" 4 4096 32
#guard same ("ab".pushn 'a' 126) "stacked" 64 7 16
#guard same PROGRAM "stacked" 64 128 8

end SampledProofs.DomainAgrees
