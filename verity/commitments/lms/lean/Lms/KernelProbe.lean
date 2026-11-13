import Lms.Verify

/-!
# A measurement: one LMS verification in the kernel at the deployed parameters

Scratch, on a branch that does not land. Each theorem is proved by the kernel's evaluation alone (`decide +kernel`),
and the lines `#eval IO.monoMsNow` between them print the clock, so the difference of two is one theorem's time:

* `hash87`, `hash1024`, `hash4096`, `hash33430`: one SHA-512 each, of 1, 9, 33 and 262 blocks. A chain step of the
  verification hashes 87 bytes, and its one-time public key 33,430.
* `verifies`: `Lms.verify` at `Lms.deployed` (N = 64, W = 1, H = 20) accepts a signature of 522 chain values and 20
  path nodes, of which 257 chains take their one step: 563 blocks in all.

The signature's values are a fixed pattern (`pat`), and the public key is the identifier and the root that signature
gives, computed by the Python `verity/commitments/crypto/lms.py`'s `root_of`: the verification does the work it does
on a certificate's signature, whatever key made it.
-/

namespace Lms.KernelProbe

/-- `n` bytes, byte `j` being `(7k + 13j + 5) mod 256`. -/
def pat (n k : Nat) : ByteArray := ⟨((List.range n).map fun j => ((k * 7 + j * 13 + 5) % 256).toUInt8).toArray⟩

def identifier : ByteArray := pat 16 0
def message : ByteArray := pat 200 999

def signature : Lms.Signature :=
  ⟨123456, pat 64 1, ((List.range 522).map fun i => pat 64 (i + 2)).toArray,
    ((List.range 20).map fun i => pat 64 (i + 600)).toArray⟩

def root : ByteArray := ⟨#[190, 139, 216, 1, 6, 136, 251, 37, 119, 245, 88, 5, 163, 159, 247, 197, 112, 183, 82, 60, 50,
  250, 54, 236, 159, 234, 132, 152, 69, 241, 14, 250, 107, 11, 99, 197, 65, 103, 192, 80, 144, 164, 40, 193, 65, 231,
  129, 123, 253, 188, 0, 11, 185, 237, 173, 52, 185, 110, 154, 206, 87, 7, 21, 230]⟩

def publicKey : ByteArray := identifier ++ root

def digest87 : ByteArray := ⟨#[212, 246, 142, 180, 68, 190, 23, 193, 6, 229, 15, 73, 143, 161, 110, 125, 143, 205, 125,
  149, 224, 227, 110, 1, 239, 38, 154, 253, 104, 182, 189, 219, 161, 9, 248, 125, 234, 226, 93, 93, 111, 186, 58, 78,
  91, 123, 228, 115, 17, 251, 250, 120, 96, 196, 79, 11, 62, 119, 81, 150, 229, 213, 250, 235]⟩

def digest1024 : ByteArray := ⟨#[153, 99, 105, 235, 209, 158, 61, 62, 234, 226, 77, 233, 70, 154, 207, 66, 210, 74, 226,
  194, 11, 249, 192, 185, 17, 109, 203, 163, 159, 221, 5, 79, 73, 245, 163, 170, 113, 3, 242, 183, 144, 120, 9, 180,
  229, 204, 201, 205, 161, 246, 6, 239, 141, 107, 210, 75, 41, 7, 65, 156, 119, 154, 160, 91]⟩

def digest4096 : ByteArray := ⟨#[37, 215, 120, 63, 222, 168, 239, 51, 109, 47, 25, 32, 138, 212, 234, 219, 187, 156, 36,
  77, 203, 122, 24, 137, 207, 224, 153, 220, 149, 166, 5, 210, 83, 32, 102, 213, 192, 145, 106, 97, 42, 158, 33, 206,
  112, 240, 234, 146, 128, 237, 215, 66, 129, 232, 1, 128, 37, 90, 128, 228, 123, 225, 172, 109]⟩

def digest33430 : ByteArray := ⟨#[87, 124, 105, 236, 233, 78, 62, 211, 5, 57, 37, 117, 176, 17, 237, 146, 137, 116, 21,
  93, 76, 223, 68, 178, 91, 209, 140, 244, 252, 171, 196, 53, 172, 162, 61, 222, 43, 12, 125, 29, 254, 50, 131, 254,
  161, 102, 222, 198, 62, 250, 188, 103, 5, 201, 166, 24, 50, 152, 34, 255, 201, 41, 17, 63]⟩

#eval IO.monoMsNow

set_option maxHeartbeats 0 in
theorem hash87 : (Flock.Sha512.hash (pat 87 3) == digest87) = true := by decide +kernel

#eval IO.monoMsNow

set_option maxHeartbeats 0 in
theorem hash1024 : (Flock.Sha512.hash (pat 1024 5) == digest1024) = true := by decide +kernel

#eval IO.monoMsNow

set_option maxHeartbeats 0 in
theorem hash4096 : (Flock.Sha512.hash (pat 4096 6) == digest4096) = true := by decide +kernel

#eval IO.monoMsNow

set_option maxHeartbeats 0 in
theorem hash33430 : (Flock.Sha512.hash (pat 33430 4) == digest33430) = true := by decide +kernel

#eval IO.monoMsNow

set_option maxHeartbeats 0 in
theorem verifies :
    (match Lms.verify Lms.deployed publicKey message signature with
      | .ok accepted => accepted
      | .error _ => false) = true := by
  decide +kernel

#eval IO.monoMsNow

end Lms.KernelProbe
