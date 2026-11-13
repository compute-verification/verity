import Lms.Verify

/-!
# A measurement: one LMS verification in the kernel at the deployed parameters

Scratch, on a branch that does not land. Each theorem is proved by the kernel's evaluation alone (`decide +kernel`),
smallest first, and `mark` after each writes one line at once: the clock, and the process's peak resident memory since
the line before. The difference of two clocks is one theorem's time, and a line's peak is its theorem's.

* `compress1`: one SHA-512 compression, of a literal block.
* `hash87`, `hash200`, `hash300`: one SHA-512 each, of 1, 2 and 3 blocks. A chain step of the verification hashes 87
  bytes.
* `verifies`: `Lms.verify` at `Lms.deployed` (N = 64, W = 1, H = 20) accepts a signature of 522 chain values and 20
  path nodes, of which 257 chains take their one step, and whose one-time public key is hashed over 33,430 bytes: 563
  blocks in all.

The signature's values are a fixed pattern (`pat`), and the public key is the identifier and the root that signature
gives, computed by the Python `verity/commitments/crypto/lms.py`'s `root_of`: the verification does the work it does
on a certificate's signature, whatever key made it.

The job that elaborates this file limits its own address space, so a theorem that needs more than the limit ends the
process there, after the lines of the theorems before it.
-/

namespace Lms.KernelProbe

/-- One line, written to the process's standard error and flushed at once (lean holds its own output back while its
standard output is a pipe): the clock in milliseconds, and the process's peak and present resident memory as
`/proc/self/status` gives them. The peak is then reset, so the next line's peak is the next theorem's. -/
def mark (label : String) : IO Unit := do
  let ms ← IO.monoMsNow
  let status ← try IO.FS.readFile "/proc/self/status" catch _ => pure ""
  let field (k : String) : String :=
    match (status.splitOn "\n").find? (·.startsWith k) with
    | some l => String.intercalate " " ((l.splitOn "\t").drop 1 |>.map (·.trimAscii.toString))
    | none => "not read"
  let line := s!"kernel-probe: {label}: clock {ms} ms, peak {field "VmHWM"}, now {field "VmRSS"}"
  try
    let h ← IO.FS.Handle.mk "/proc/self/fd/2" .append
    h.putStrLn line
    h.flush
  catch _ => pure ()
  try IO.FS.writeFile "/proc/self/clear_refs" "5" catch _ => pure ()
  IO.println line

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

/-- `pat 87 3` padded as SHA-512 pads it: one block. -/
def block87 : ByteArray := ⟨#[26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 208, 221, 234, 247, 4,
  17, 30, 43, 56, 69, 82, 95, 108, 121, 134, 147, 160, 173, 186, 199, 212, 225, 238, 251, 8, 21, 34, 47, 60, 73, 86,
  99, 112, 125, 138, 151, 164, 177, 190, 203, 216, 229, 242, 255, 12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155,
  168, 181, 194, 207, 220, 233, 246, 3, 16, 29, 42, 55, 68, 81, 94, 107, 120, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 184]⟩

/-- The eight words `block87` compresses to from the initial value: `digest87`'s. -/
def words87 : Array UInt64 := #[0xd4f68eb444be17c1, 0x06e50f498fa16e7d, 0x8fcd7d95e0e36e01, 0xef269afd68b6bddb,
  0xa109f87deae25d5d, 0x6fba3a4e5b7be473, 0x11fbfa7860c44f0b, 0x3e775196e5d5faeb]

def digest200 : ByteArray := ⟨#[58, 149, 64, 243, 169, 194, 145, 173, 10, 104, 166, 172, 225, 71, 131, 138, 25, 50,
  179, 210, 16, 9, 184, 48, 53, 247, 207, 7, 18, 197, 20, 100, 213, 137, 129, 55, 93, 44, 247, 128, 251, 146, 129, 26,
  27, 118, 26, 2, 168, 27, 114, 173, 173, 4, 214, 50, 62, 153, 234, 9, 215, 193, 71, 26]⟩

def digest300 : ByteArray := ⟨#[195, 149, 247, 250, 91, 205, 44, 165, 150, 153, 41, 201, 136, 19, 233, 70, 130, 230,
  145, 166, 230, 151, 140, 238, 60, 202, 130, 3, 241, 92, 153, 118, 162, 213, 60, 192, 147, 37, 190, 100, 19, 69, 107,
  19, 165, 110, 22, 128, 20, 61, 99, 112, 150, 146, 52, 3, 225, 29, 204, 26, 253, 169, 4, 94]⟩

#eval mark "start"

set_option maxHeartbeats 0 in
theorem compress1 : (Flock.Sha512.compress Flock.Sha512.iv block87 0 == words87) = true := by decide +kernel

#eval mark "compress1, one compression"

set_option maxHeartbeats 0 in
theorem hash87 : (Flock.Sha512.hash (pat 87 3) == digest87) = true := by decide +kernel

#eval mark "hash87, 1 block"

set_option maxHeartbeats 0 in
theorem hash200 : (Flock.Sha512.hash (pat 200 7) == digest200) = true := by decide +kernel

#eval mark "hash200, 2 blocks"

set_option maxHeartbeats 0 in
theorem hash300 : (Flock.Sha512.hash (pat 300 8) == digest300) = true := by decide +kernel

#eval mark "hash300, 3 blocks"

set_option maxHeartbeats 0 in
theorem verifies :
    (match Lms.verify Lms.deployed publicKey message signature with
      | .ok accepted => accepted
      | .error _ => false) = true := by
  decide +kernel

#eval mark "verifies, 563 blocks"

end Lms.KernelProbe
