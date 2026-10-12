/-
The library (Project doc `constant-api-public.md` §2.9; core's `verity.ml.library`, version 1): the tables core owns as
hardware semantics, public in both tracks. A table is `library` exactly when its SHA-512 (of its raw little-endian u32
words, the archive key) is listed here, and `program` otherwise. `backends/flock/tests/test_table_library.py` checks this
list against core's.
-/
namespace Flock

structure LibraryTable where
  name : String
  sha512 : String
  indexBits : Nat
  valueBits : Nat
  deriving Inhabited, Repr

def LIBRARY : String := "library"
def PROGRAM : String := "program"

def LIBRARY_TABLES : List LibraryTable :=
  [⟨"rsq", "39f7b912bc25b913618d926725a18974af9eb246f6f36ce06ef63ffdb293d80bc222623c6d182d2c23e16f5ada2b8db9e7e6a53b3e42046763f7794ac3cef378", 24, 31⟩,
   ⟨"sqrt", "33d076106efaa955e6452200d45651a799628e76d1d1cc682168f464a03100b937bee08533761d9fa8f8a114cee2a32d322888d5f08fd320bfeb17805a5c2536", 24, 31⟩,
   ⟨"rcp", "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc", 23, 31⟩,
   ⟨"ex2", "5cb768c1725369ad905e725437463729d5b6054ca9adc498e755247e2ba848b510e3325e1c3fecad1242832aca5fa0afd463b16002a7f34050a7b377ed107877", 23, 31⟩,
   ⟨"tanh_mufu", "f83e87737a231de2ec109f5a7ea9f8dac78089ef9978ac19ce540c03cefd356ce542948ba8ffb11952b68a2d0f63fe4c6e2e84d218739a262a9b4bb07dd66985", 27, 31⟩]

def libraryTable? (sha512 : String) : Option LibraryTable := LIBRARY_TABLES.find? (·.sha512 == sha512)

end Flock
