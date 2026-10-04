import FlockSoundness.Discharge.PrivateCircuit.ReadsSound

/-!
# Hidden reads at hidden unit circuits: the size-only class list

No part of the full circuit is public. The full circuit is cut into proof units; each proof unit runs a unit circuit (a
template) that is itself a committed witness, so the verifier knows only its size bounds. What is public is the
**size-only class list** (`SizeClass`, `SizeList`): per class, the gate bound of its unit circuit, its input and output
widths and its padded unit count. The number of classes is public; which unit circuit a class holds is not.

* **Committed unit circuits** (`UnitCircuits`, `hiddenPop`, `TableFits`): a unit circuit is a value of `H.UC`, with its
  gate count `H.gates` and its evaluation `H.run` on `nIn` input words, giving `nOut` output words. The topology entry of
  unit `j` holds its unit circuit and the store cells its read slots read (`Entry (hiddenPop H N)`), so the unit circuit
  is committed under the table root with the rest of the entry, and unit `j`'s block is `H.run` of the unit circuit its
  entry holds. `TableFits` is the size check: every unit circuit within its class's gate bound. A realization that
  commits each template's unit circuit once, with each entry naming its index, refines this: the entry's unit circuit is
  the template table's at that index.
* `private_circuit_sound_hidden`: the read binding at a committed unit circuit. It is `private_circuit_sound_reads` at
  `hiddenPop H N`: the outer verifier circuit's net (V*, the class circuit of the recursion) computes `ClassRel`, whose
  block conjunct reads the unit circuit from the opened table entry (`ClassComputes`), and the bounds are the same.
  `ClassComputes` at V*'s net is what V*'s relation, under the inner proof's soundness, has to give.
* `private_circuit_private_hidden`, `private_circuit_indist_hidden`: privacy at a size-only class list `sl`. The program
  is the table of `sl.units` entries, unit circuits included, each registered as its hm96 commit string
  (`hiddenScheme`); the simulator reads `sl` and the owed outputs `o` only, and a dummy table (`p₀`) whose unit circuits
  are any of the class's. The size list is public, so circuits whose size lists differ can be told apart; at one size
  list the view hides every committed entry: the unit circuits, their dimensions and the cells each unit reads.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding Audit Reads
open scoped ENNReal

namespace Reads

/-- **A class's public sizes**: the gate bound of its unit circuit, its input and output widths, and its padded unit
count. Nothing names the unit circuit. -/
structure SizeClass where
  gates : ℕ
  inWidth : ℕ
  outWidth : ℕ
  padded : ℕ
  deriving DecidableEq

/-- **The size-only class list**, one `SizeClass` per class: the public part of the full circuit. -/
abbrev SizeList := List SizeClass

/-- The table's length at a size list: every class's padded unit count. -/
def SizeList.units (sl : SizeList) : ℕ := (sl.map SizeClass.padded).sum

/-- **Unit circuits as committed values**: a type `UC` of unit circuits, each with its gate count and its evaluation on
`nIn` input words of `V`, giving `nOut` output words. -/
structure UnitCircuits (V : Type) (nIn nOut : ℕ) where
  UC : Type
  gates : UC → ℕ
  run : UC → (Fin nIn → V) → Fin nOut → V

/-- **The population at committed unit circuits**: `N` units of `nIn` read slots and `nOut` output rows, whose topology
entry holds the unit circuit (`Dm := H.UC`), and whose block is `H.run` of it. -/
abbrev hiddenPop {V : Type} {nIn nOut : ℕ} (H : UnitCircuits V nIn nOut) (N : ℕ) : Pop V where
  N := N
  r := nIn
  o := nOut
  Dm := H.UC
  blk _ c := H.run c

/-- **The table's unit circuits fit their classes**: unit `j`'s unit circuit is within the gate bound of its public class
`cls j`. -/
def TableFits {V : Type} {nIn nOut N : ℕ} (H : UnitCircuits V nIn nOut) (cls : Fin N → SizeClass)
    (T : Table (hiddenPop H N)) : Prop :=
  ∀ j, H.gates (T j).dims ≤ (cls j).gates

end Reads

section Sound

variable {V : Type} [Inhabited V] {nIn nOut N : ℕ} {H : UnitCircuits V nIn nOut} {C : Audit.Circuit}
  {Pt : Audit.Partition C N} {Reg : Type} {M : ℕ}

open Classical in
/-- **The read binding at committed unit circuits.** `private_circuit_sound_reads` at `hiddenPop H N`: the topology entry
of each unit holds its unit circuit, committed under the table root `rtT₀` with the cells it reads, and the class net
(V*'s) computes `ClassRel` at each unit's check gate (`hcomp`), whose block conjunct runs the unit circuit the opened
entry holds. The three bounds are `private_circuit_sound_reads`'s, with `T₀`'s unit circuits the registered ones. -/
theorem private_circuit_sound_hidden
    (L : ReadLayout (hiddenPop H N) C.N) (lfT : ExplicitLeaf (Entry (hiddenPop H N))) (lfS : ExplicitLeaf V)
    (idx : Cell → Fin M) (rootT rootS : Reg → Flock.Registered.Port) (good : Reg → (Fin C.N → Bool) → Prop)
    (ok : Fin N → Fin C.N)
    (hcomp : ClassComputes Pt good (fun R v u => ClassRel L lfT lfS idx (rootT R) (rootS R) v u) ok)
    (rtT₀ rtS₀ : Flock.Registered.Port) (tr : Fin N → ReadVal (Entry (hiddenPop H N))) (wo : Fin M → ReadVal V)
    (htr : ∀ j : Fin N, ReadOk lfT rtT₀ j (tr j)) (hwo : ∀ i : Fin M, ReadOk lfS rtS₀ i (wo i))
    (s : UnitSession Pt Reg) (drawn : s.Out → Finset (Fin N)) (hD : ZkSessionDrawn s drawn)
    (hE : DrawEscapes s drawn) (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (τ : Strategy s.G) (hCR : s.CR τ) (hgood : ∀ o, s.Accepts o → good (s.regOf o) (s.vals τ o))
    (owed : Finset (Fin N × Fin nOut)) (pubIn : Finset ℕ)
    (yD : s.Out → Fin N × Fin nOut → ReadVal V) (xD : s.Out → ℕ → ReadVal V)
    (hdel : ∀ o, s.Accepts o →
      (∀ p ∈ owed, ReadOk lfS (rootS (s.regOf o)) (idx (.out p.1 p.2)) (yD o p)) ∧
        ∀ m ∈ pubIn, ReadOk lfS (rootS (s.regOf o)) (idx (.reg m)) (xD o m))
    (costS costT : s.Out → ℝ) (qS qT : ℝ) (hcS : ∀ o, costS o ≤ qS) (hcT : ∀ o, costT o ≤ qT)
    (hS : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some (readPick lfS rtS₀ Fin.val wo (storeReads L idx s τ drawn owed pubIn yD xD) o)) costS qS)
    (hT : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some ((regCommit lfT N).ext (rtT₀, tabFill L s τ drawn tr o) (rtT₀, tr))) costT qT) :
    let w : Cell → V := fun c => (wo (idx c)).val
    let T₀ : Table (hiddenPop H N) := fun j => (tr j).val
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨ ∃ p ∈ owed, (yD o p).val ≠ w (.out p.1 p.2))) s.G τ
      ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) ∧
    (OwedWrongR (hiddenPop H N) T₀ w owed →
      prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀) s.G τ
        ≤ s.Lm.escape (bad (hiddenPop H N) T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) +
          ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))) ∧
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨
          ∃ p ∈ owed, (yD o p).val ≠ evalR (hiddenPop H N) T₀ (fun m => w (.reg m)) p.1 p.2)) s.G τ
      ≤ (if OwedWrongR (hiddenPop H N) T₀ w owed then
          s.Lm.escape (bad (hiddenPop H N) T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))
          else 0) +
        ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) :=
  private_circuit_sound_reads L lfT lfS idx rootT rootS good ok hcomp rtT₀ rtS₀ tr wo htr hwo s drawn hD hE hA3 hA6 τ
    hCR hgood owed pubIn yD xD hdel costS costT qS qT hcS hcT hS hT

end Sound

/-- **The program commitment at a size-only class list**: `hm96Scheme` with the size list as the class and the topology
table of `sl.units` entries, unit circuits included, as the program, each entry registered as its hm96 commit string
(`encE`). -/
abbrev hiddenScheme {V Circ Out : Type} {nIn nOut : ℕ} {K W Hi VL : Type} [Field K] (H : UnitCircuits V nIn nOut)
    (fits : Circ → SizeList → Prop) (tab : Circ → (sl : SizeList) → Table (hiddenPop H sl.units))
    (encE : (sl : SizeList) → Entry (hiddenPop H sl.units) → List UInt8) (J : SizeList → ℕ)
    (zs : (sl : SizeList) → Fin (J sl) → Sizes)
    (U : (sl : SizeList) → Out → (Fin sl.units → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs sl j)) :=
  hm96Scheme fits tab SizeList.units (fun sl T j => encE sl (T j)) J zs U

/-- **Protocol 2 at hidden unit circuits is private at a size-only class list.** The public view is the size list `sl`
(gate bound, input and output widths and padded unit count per class) and the owed outputs `o`; the program is the
topology table of `sl.units` entries (`tab C sl`), each holding its unit circuit and the cells it reads, registered as
its hm96 commit string (`encE`) under the registrant's uniform salts (`hiddenScheme`); the outer verifier circuit's
`--zk` tables `U sl o v` read the size list, the owed outputs and that registration `v` only. Under `hash-derived-key` at
`δ₁` (`hKey`), where the tables at every registration of `tab C sl` meet `ZkWireFixed`'s conditions (`hZk`) and the
honest witness satisfies them (`hSat`): for every dummy table `p₀` and dummy witness `w₀`, every encoding of the views
and every event of the registration's view and that encoding, the real view and the simulator's, which reads `sl` and
`o` and nothing of `C`, are within `2·sl.units·δ₁ + zkGap (zs sl)`, both ways. `circuit_private` at `hiddenScheme`. -/
theorem private_circuit_private_hidden
    {V Circ Out : Type} {nIn nOut : ℕ} {K W Hi VL : Type} [Field K] [Fintype K] [DecidableEq K]
    [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] (H : UnitCircuits V nIn nOut) (fits : Circ → SizeList → Prop)
    (tab : Circ → (sl : SizeList) → Table (hiddenPop H sl.units))
    (encE : (sl : SizeList) → Entry (hiddenPop H sl.units) → List UInt8) (J : SizeList → ℕ)
    (zs : (sl : SizeList) → Fin (J sl) → Sizes)
    (U : (sl : SizeList) → Out → (Fin sl.units → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs sl j))
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (sl : SizeList) (o : Out) (C : Circ) (w : (Fin sl.units → Salt) → Fin (J sl) → W)
    (hZk : ∀ r, ZkOk (U sl o (hm96Reg (fun T j => rowBits (encE sl (T j))) (tab C sl) r)))
    (hSat : ∀ r, Sat (U sl o (hm96Reg (fun T j => rowBits (encE sl (T j))) (tab C sl) r)) (w r))
    (p₀ : Table (hiddenPop H sl.units)) (w₀ : Fin (J sl) → W) {Obs : Type}
    (encV : ∀ v, (∀ j, (U sl o v j).tab.View) → Obs) (E : (Fin sl.units → Bits × ByteArray) → Obs → Prop) :
    realPr (hiddenScheme H fits tab encE J zs U) sl o (tab C sl) w encV E
        ≤ simPr (hiddenScheme H fits tab encE J zs U) sl o p₀ w₀ encV E + (2 * sl.units * δ₁ + zkGap (zs sl)) ∧
      simPr (hiddenScheme H fits tab encE J zs U) sl o p₀ w₀ encV E
        ≤ realPr (hiddenScheme H fits tab encE J zs U) sl o (tab C sl) w encV E + (2 * sl.units * δ₁ + zkGap (zs sl)) :=
  circuit_private (hiddenScheme H fits tab encE J zs U) sl (2 * sl.units * δ₁) (hm96Reg_hides _ hKey) o C w hZk hSat
    p₀ w₀ encV E

/-- **At hidden unit circuits the view is a function of the size-only class list and the owed outputs**: any two full
circuits `C`, `C'` at one size list `sl`, with honest witnesses for the same owed outputs `o`, give every event of the
verifier's view probabilities within `2·(2·sl.units·δ₁ + zkGap (zs sl))`, under `hash-derived-key` at `δ₁`, whatever
their unit circuits, dimensions and reads. Nothing is claimed for circuits of different size lists: the size list is
public. `circuit_indist` at `hiddenScheme`. -/
theorem private_circuit_indist_hidden
    {V Circ Out : Type} {nIn nOut : ℕ} {K W Hi VL : Type} [Field K] [Fintype K] [DecidableEq K]
    [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] (H : UnitCircuits V nIn nOut) (fits : Circ → SizeList → Prop)
    (tab : Circ → (sl : SizeList) → Table (hiddenPop H sl.units))
    (encE : (sl : SizeList) → Entry (hiddenPop H sl.units) → List UInt8) (J : SizeList → ℕ)
    (zs : (sl : SizeList) → Fin (J sl) → Sizes)
    (U : (sl : SizeList) → Out → (Fin sl.units → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs sl j))
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (sl : SizeList) (o : Out) (C C' : Circ) (w w' : (Fin sl.units → Salt) → Fin (J sl) → W)
    (hZk : ∀ r, ZkOk (U sl o (hm96Reg (fun T j => rowBits (encE sl (T j))) (tab C sl) r)))
    (hZk' : ∀ r, ZkOk (U sl o (hm96Reg (fun T j => rowBits (encE sl (T j))) (tab C' sl) r)))
    (hSat : ∀ r, Sat (U sl o (hm96Reg (fun T j => rowBits (encE sl (T j))) (tab C sl) r)) (w r))
    (hSat' : ∀ r, Sat (U sl o (hm96Reg (fun T j => rowBits (encE sl (T j))) (tab C' sl) r)) (w' r)) {Obs : Type}
    (encV : ∀ v, (∀ j, (U sl o v j).tab.View) → Obs) (E : (Fin sl.units → Bits × ByteArray) → Obs → Prop) :
    realPr (hiddenScheme H fits tab encE J zs U) sl o (tab C sl) w encV E
      ≤ realPr (hiddenScheme H fits tab encE J zs U) sl o (tab C' sl) w' encV E +
        2 * (2 * sl.units * δ₁ + zkGap (zs sl)) :=
  circuit_indist (hiddenScheme H fits tab encE J zs U) sl (2 * sl.units * δ₁) (hm96Reg_hides _ hKey) o C C' w w' hZk
    hZk' hSat hSat' encV E

end FlockSoundness.Discharge.PrivateCircuit
