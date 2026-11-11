module

public import Definitions.Accounting.Device
public import Definitions.Pouw.PearlC.W1

/-!
# The device model's price at W1's prices (trusted)

`Accounting.Price` prices the device model's three op classes; W1 (`Pouw.PearlC.W1.W1Prices`) prices each tensor-core
format's multiply-add and each CUDA-core instruction, in one dense E4M3 multiply-add. `w1Price pr f` reads the first
off the second for a run whose tensor-core multiply-adds are of format `f`. A card's price is it at that card's W1
prices and a format, stated beside the proof that reads it.
-/

@[expose] public section

namespace Accounting

open Pouw.PearlC.W1

/-- **The device model's price at W1's prices `pr`**, for tensor-core multiply-adds of format `f`: a tensor-core
multiply-add at `f`'s W1 price, a CUDA-core op at the cheaper of W1's FP32 and INT32 instructions, and an oracle query
at one SHA-512 compression, each rounded up to a whole unit (one dense E4M3 multiply-add). -/
def w1Price (pr : W1Prices) (f : Fmt) : Price := fun o =>
  match o with
  | .tc => ⌈pr.mac f⌉₊
  | .cuda => ⌈min pr.fp32 pr.int32⌉₊
  | .oracle => ⌈pr.sha512⌉₊

end Accounting
