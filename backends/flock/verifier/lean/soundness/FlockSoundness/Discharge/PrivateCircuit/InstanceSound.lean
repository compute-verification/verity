import FlockSoundness.Discharge.PrivateCircuit.InstanceDefs
import FlockSoundness.Discharge.PrivateCircuit.Soundness
import FlockSoundness.Discharge.PrivateCircuit.InstanceCommit
import FlockSoundness.Discharge.PrivateCircuit.InterpPad

/-!
# Protocol 2 is sound at the universal unit

`private_circuit_sound`, at `UniversalUnit_v1`'s interpreter (`Universal`): accepting with at least `K₀` wrong units of
the circuit `C` itself, or with program leaves other than the registered program `pad C c`'s, is at most
`miss(K₀) + ε_ks + δ_link + q²/2^513`. It counts `C`'s units directly, so no citation composes it with anything else.

The steps, each a named hypothesis or a proved fact:

* **(i) the leaves.** The session opens the program commitment of the registered root (`progCommit`) at its leaves
  (`Reads.Checked`, `hchk`), and the registrant at `pad C c`'s: leaves other than the registrant's are a SHA-512
  collision (`progCommit_binding`).
* **The program.** Where the leaves are the registrant's, the program the committed values hold at the registered sites is
  `pad C c`, or SHA-512 collides (`LeafBinds`: (i)'s leaf equality and the layout's binding from leaves to values).
* **The units.** Unit `i` of the universal unit's partition is wrong at the committed values exactly when the universal
  unit running the program they hold, on unit `i`'s claimed inputs, misses its claimed outputs (`UnitWrong`), so the
  partition's wrong units are the padded program's (`UnitWrong.card`).
* **The padding.** The padded program's wrong units are at least `C`'s (`pad_wrong` at `pad_correct`, proved).
* Protocol 1's headline at the universal unit (`ZkSessionSound`) bounds the partition's wrong units, and one finder
  (`progPair`) outputs both collisions.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

/-- **The universal unit's partition is the interpreter's, unit by unit**: at every assignment `v` of the universal
circuit's wires, unit `i` of `P` is wrong exactly when the universal unit running the program `v` holds at the registered
sites (`progOf v`), on unit `i`'s claimed inputs (`xOf v i`), outputs other than its claimed outputs (`yOf v i`). A fact
about the circuit the universal unit compiles to (`UniversalUnit_v1`), discharged where that circuit is built. -/
def UnitWrong {C : Audit.Circuit} {c : Universal.Cls} (P : Audit.Partition C (Universal.units c))
    (progOf : (Fin C.N → Bool) → c.Prog) (xOf : (Fin C.N → Bool) → ℕ → ℕ → Bool)
    (yOf : (Fin C.N → Bool) → ℕ → List Bool) : Prop :=
  ∀ v (i : Fin (Universal.units c)), i ∈ P.wrong v ↔ Universal.run c (progOf v) i (xOf v i) ≠ yOf v i

/-- Under `UnitWrong`, the partition's wrong units are the padded program's: as many as the units, of `units c`, where
the universal unit running `progOf v` misses the claimed outputs. -/
theorem UnitWrong.card {C : Audit.Circuit} {c : Universal.Cls} {P : Audit.Partition C (Universal.units c)}
    {progOf : (Fin C.N → Bool) → c.Prog} {xOf : (Fin C.N → Bool) → ℕ → ℕ → Bool}
    {yOf : (Fin C.N → Bool) → ℕ → List Bool} (h : UnitWrong P progOf xOf yOf) (v : Fin C.N → Bool) :
    (P.wrong v).card = wrong (Universal.run c (progOf v)) (Universal.units c) (xOf v) (yOf v) := by
  sorry

section Sites

variable {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay}
  {rd : Registered.Reads lay rs} {rt : Flock.Registered.Port} {L : ℕ}

/-- **The step to the program**: on every accepted outcome of the session `s` against `τ` whose leaves at the program's
sites are the registrant's (`ps.leaves`, `ps.regLeaves`: (i)'s leaf equality), the program the committed values hold at
the registered sites (`progOf (s.vals τ o)`) is `p`, or `valPair o` is a SHA-512 collision. It is the layout's binding
from leaves to values (`Reads.registered_weights`) read at the program's sites, discharged where `s.vals` is the
session's extracted values. -/
def LeafBinds (ps : ProgSites rd rt L) (s : UnitSession P Reg) (τ : Strategy s.G) {Prg : Type}
    (progOf : (Fin C.N → Bool) → Prg) (p : Prg) (valPair : s.Out → List UInt8 × List UInt8) : Prop :=
  ∀ o, s.Accepts o → ps.leaves (s.regOf o) = ps.regLeaves → progOf (s.vals τ o) = p ∨ Binding.Collides (valPair o)

open Classical in
/-- **The binding finder's pair** at an outcome `o`: the step to the program's (`valPair o`) where the session's leaves
are the registrant's, else the program commitment's extractor on the session's opening and the registrant's. -/
noncomputable def progPair (ps : ProgSites rd rt L) (s : UnitSession P Reg) (path : s.Out → lay.Pos → ByteArray)
    (valPair : s.Out → List UInt8 × List UInt8) (o : s.Out) : List UInt8 × List UInt8 :=
  if ps.leaves (s.regOf o) = ps.regLeaves then valPair o
  else (progCommit L).ext (ps.opening (s.regOf o) (path o)) ps.regOpening

end Sites

/-- **Protocol 2 is sound at the universal unit.** A circuit `C` fits the class `c`, and `pad C c` is registered: the
registrant's rows at the program's sites (`ps`, `L` registered reads of the root `rt` in the universal unit's layout
`lay` over its partition `UP c`, of `units c` units) are `pad C c`'s rows (`row`). The partition is the interpreter's
(`UnitWrong`: unit `i` wrong at `v` exactly when running the program `progOf v` on `xOf v i` misses `yOf v i`). The
universal unit's session `s` meets protocol 1's soundness headline (`ZkSessionSound`, under A3 and A6); on every
accepted outcome the registered check holds at the outcome's registration (`hchk`, `Flock.Registered.check_ok` read at
the model's leaves); and where the session's leaves are the registrant's, the committed values hold `pad C c` at the
registered sites or `valPair` collides (`LeafBinds`).

Then for every live prover `τ` whose knowledge finders meet `cr/sha-512` (`hCR`), accepting with at least `K₀` units of
`C` wrong at the committed values' claimed inputs and outputs (`wrong (eval C) (size C)`), or with program leaves other
than `pad C c`'s registered ones, has probability at most `miss(K₀) + ε_ks + δ_link + q²/2^513`, the last under
`SHA512CRStrict` (`cr/sha-512`) for the finder that plays the session once and outputs `progPair`, at most `q` SHA-512
evaluations on every outcome (`hBind`). `miss` is the law's over the padded `units c` units, dummies included. -/
theorem private_circuit_sound
    {UC : Universal.Cls → Audit.Circuit} (UP : (c : Universal.Cls) → Audit.Partition (UC c) (Universal.units c))
    (c : Universal.Cls) (C : Universal.Circ) (hC : Universal.fits C c)
    (progOf : (Fin (UC c).N → Bool) → c.Prog) (xOf : (Fin (UC c).N → Bool) → ℕ → ℕ → Bool)
    (yOf : (Fin (UC c).N → Bool) → ℕ → List Bool) (hunit : UnitWrong (UP c) progOf xOf yOf)
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (Universal.units c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (Universal.units c))) → (R : Reg) → Fin (Universal.units c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UC c) (Reg := Reg) plan tab}
    {lay : Binding.Layout (UP c) plan tab decode} {rs : Binding.HmRows lay} (rd : Registered.Reads lay rs)
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (row : c.Prog → Fin L → lay.Val) (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C c) i)
    (s : UnitSession (UP c) Reg) (hS : ZkSessionSound s)
    (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (path : s.Out → lay.Pos → ByteArray) (hchk : ∀ o, s.Accepts o → rd.Checked (s.regOf o) (path o))
    (τ : Strategy s.G) (hCR : s.CR τ) (valPair : s.Out → List UInt8 × List UInt8)
    (hstep : LeafBinds ps s τ progOf (Universal.pad C c) valPair)
    (K₀ : ℕ) (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ (fun o => some (progPair ps s path valPair o)) cost q) :
    prob (fun o => s.Accepts o ∧
        (K₀ ≤ wrong (Universal.eval C) (Universal.size C) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ∨
          ∃ i, lay.regLeaf (s.regOf o) (ps.site i) ≠
            (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (row (Universal.pad C c) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
