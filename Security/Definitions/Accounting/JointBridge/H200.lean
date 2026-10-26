import Definitions.Accounting.Device
import Definitions.Pouw.PearlC.W1

/-!
# The device model's price at W1's prices, and the H200's with INT8 (trusted)

`Accounting.Price` prices the device model's three op classes; W1 (`Pouw.PearlC.W1.W1Prices`) prices each tensor-core
format's multiply-add and each CUDA-core instruction, in one dense E4M3 multiply-add. `w1Price pr f` reads the first
off the second for a run whose tensor-core multiply-adds are of format `f`, and `h200S8Price` is the H200's with INT8,
the format of an H1 + fx1 tile's products (`GemmAccumulatorS8_v2`).
-/

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

/-- **The H200's price with INT8**: `W1Device.s8H100`'s prices (`W1Prices.h100`, the H100's and the H200's, one GH100)
for S8 multiply-adds: an INT8 tensor-core multiply-add at W1's S8 price, `1`; a CUDA-core op at FP32's `32` (INT32's
is `64`); an oracle query at `sha` rounded up. -/
def h200S8Price (sha fetch : ℚ) : Price := w1Price (W1Device.s8H100 sha fetch).prices .s8

end Accounting
