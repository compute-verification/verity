import FlockSoundness.Discharge.PrivateCircuit.ReadsBind
import FlockSoundness.Discharge.PrivateCircuit.ReadsCompose
import FlockSoundness.Discharge.PrivateCircuit.Soundness
import FlockSoundness.Discharge.PrivateCircuit.Privacy
import FlockSoundness.Discharge.PrivateCircuit.InstanceHides

/-!
# Protocol 2 with hidden reads, end to end

`U cls` is the class circuit (a library block plus its read gadget, `ClassRel`), and the program is the topology table,
each unit's entry its dimensions and the store cells it reads. The registrant registers the table under the root `rtT₀`
and the row store under `rtS₀`, and holds an opening of every index of each (`tr`, `wo`). The registered model is the
table `T₀ j = (tr j).val` and the store `w c = (wo (idx c)).val`; its registered cells `w (.reg m)` are the committed
inputs and weights.

* `private_circuit_sound_reads`, soundness at the drawn units, for every live prover of a session whose registration
  carries the registrant's roots:
  1. the delivered values are the committed ones: a delivered public input other than the store's registered cell, or a
     delivered output other than the store's output row, is a read off the store, at most `q_S²/2^513`;
  2. if the registered model's owed outputs are wrong (`OwedWrongR`), accepting is at most the law's escape at the units
     wrong against the store (`bad T₀ w`, fixed before the draw) plus `ε_ks + δ_link` plus `q_S²/2^513` plus `q_T²/2^513`:
     a drawn unit wrong against the store is wrong in the committed values, or its gadget's reads are not the
     registered ones (a collision with the registrant's opening: the store's finder, `readPick`; the table's, through
     `ran_registered` at `regCommit`);
  3. together: accepting with a delivered public input other than the committed cell, or a delivered output other than
     the registered model's end to end on the committed cells, is at most `q_S²/2^513` when the registered model's owed
     outputs are right, and part 2's bound otherwise.
  The inputs condition is explicit: the model runs on the committed cells, and the delivered public inputs are bound to
  them (`pubIn`); every other registered cell (the weights, any private input) is the registrant's.
* `private_circuit_private_reads`: `circuit_private` at the program commitment `hm96Scheme` with the topology table's
  entries as its rows: the verifier's view at the registered table and the simulator's, which reads the class and the
  owed outputs only, are within `2·N·δ₁ + zkGap`.

Protocol 1's drawn bound (`ZkSessionDrawn`, `zk_session_drawnR`'s form) and the class circuit's relation
(`ClassWrongOn`, the gadget's lowering) are hypotheses, as `ZkSessionSound` and `UnitWrong` were.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding Audit Reads
open scoped ENNReal

section Sound

variable {V : Type} [Inhabited V] {P : Pop V} {C : Audit.Circuit} {Pt : Audit.Partition C P.N} {Reg : Type} {M : ℕ}

open Classical in
/-- **The store reads an outcome makes**, at index `i` with opening `x`: an input read or an output write of a drawn unit
that is not wrong in the committed values, at the index of the cell its entry read names or of its output row; a
delivered output, at its output row's index; or a delivered public input, at its cell's index. -/
def storeReads (L : ReadLayout P C.N) (idx : Cell → Fin M) (s : UnitSession Pt Reg) (τ : Strategy s.G)
    (drawn : s.Out → Finset (Fin P.N)) (owed : Finset (Fin P.N × Fin P.o)) (pubIn : Finset ℕ)
    (yD : s.Out → Fin P.N × Fin P.o → ReadVal V) (xD : s.Out → ℕ → ReadVal V) (o : s.Out) (i : Fin M)
    (x : ReadVal V) : Prop :=
  (∃ u ∈ drawn o, u ∉ Pt.wrong (s.vals τ o) ∧
      ((∃ k, idx ((L.tabOf (s.vals τ o) u).src k) = i ∧ x = (L.inp u k).read (s.vals τ o)) ∨
        ∃ l : Fin P.o, idx (.out u l) = i ∧ x = (L.out u l).read (s.vals τ o))) ∨
    (∃ p ∈ owed, idx (.out p.1 p.2) = i ∧ x = yD o p) ∨ ∃ m ∈ pubIn, idx (.reg m) = i ∧ x = xD o m

open Classical in
/-- **The table opening an outcome shows**: at a drawn unit not wrong in the committed values, its gadget's entry read;
at every other unit, the registrant's opening. -/
noncomputable def tabFill (L : ReadLayout P C.N) (s : UnitSession Pt Reg) (τ : Strategy s.G)
    (drawn : s.Out → Finset (Fin P.N)) (tr : Fin P.N → ReadVal (Entry P)) (o : s.Out) (j : Fin P.N) :
    ReadVal (Entry P) :=
  if j ∈ drawn o ∧ j ∉ Pt.wrong (s.vals τ o) then (L.ent j).read (s.vals τ o) else tr j

open Classical in
/-- **Protocol 2 with hidden reads is sound at the drawn units.** The class circuit's wrong units are its relation's
(`hwr`, at the good assignments, which the committed values of an accepted outcome are, `hgood`); the registrant's
openings of the table and the store check against its roots (`htr`, `hwo`); the session meets protocol 1's drawn bound
(`hD`) and escapes sets fixed before the draw with at most the law's escape (`hE`); the verifier checks the delivered
outputs and public inputs against the registration's store root (`hdel`); and the store's and the table's finders meet
`cr/sha-512` at budgets `qS`, `qT` (`hS`, `hT`). Then, at sessions whose registration carries the registrant's roots, with
`T₀`, `w` the registered table and store: (1) a delivered value off the committed one is at most `q_S²/2^513`; (2) if the
registered model's owed outputs are wrong, accepting is at most `escape(bad T₀ w) + ε_ks + δ_link + q_S²/2^513 +
q_T²/2^513`; (3) a delivered public input off the committed cell or a delivered output off the registered model's end to
end on the committed cells is at most `q_S²/2^513` plus, when the registered model's owed outputs are wrong,
`escape(bad T₀ w) + ε_ks + δ_link + q_T²/2^513`. -/
theorem private_circuit_sound_reads
    (L : ReadLayout P C.N) (lfT : ExplicitLeaf (Entry P)) (lfS : ExplicitLeaf V) (idx : Cell → Fin M)
    (rootT rootS : Reg → Flock.Registered.Port) (good : Reg → (Fin C.N → Bool) → Prop)
    (hwr : ClassWrongOn Pt good fun R v u => ClassRel L lfT lfS idx (rootT R) (rootS R) v u)
    (rtT₀ rtS₀ : Flock.Registered.Port) (tr : Fin P.N → ReadVal (Entry P)) (wo : Fin M → ReadVal V)
    (htr : ∀ j : Fin P.N, ReadOk lfT rtT₀ j (tr j)) (hwo : ∀ i : Fin M, ReadOk lfS rtS₀ i (wo i))
    (s : UnitSession Pt Reg) (drawn : s.Out → Finset (Fin P.N)) (hD : ZkSessionDrawn s drawn)
    (hE : DrawEscapes s drawn) (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (τ : Strategy s.G) (hCR : s.CR τ) (hgood : ∀ o, s.Accepts o → good (s.regOf o) (s.vals τ o))
    (owed : Finset (Fin P.N × Fin P.o)) (pubIn : Finset ℕ)
    (yD : s.Out → Fin P.N × Fin P.o → ReadVal V) (xD : s.Out → ℕ → ReadVal V)
    (hdel : ∀ o, s.Accepts o →
      (∀ p ∈ owed, ReadOk lfS (rootS (s.regOf o)) (idx (.out p.1 p.2)) (yD o p)) ∧
        ∀ m ∈ pubIn, ReadOk lfS (rootS (s.regOf o)) (idx (.reg m)) (xD o m))
    (costS costT : s.Out → ℝ) (qS qT : ℝ) (hcS : ∀ o, costS o ≤ qS) (hcT : ∀ o, costT o ≤ qT)
    (hS : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some (readPick lfS rtS₀ Fin.val wo (storeReads L idx s τ drawn owed pubIn yD xD) o)) costS qS)
    (hT : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some ((regCommit lfT P.N).ext (rtT₀, tabFill L s τ drawn tr o) (rtT₀, tr))) costT qT) :
    let w : Cell → V := fun c => (wo (idx c)).val
    let T₀ : Table P := fun j => (tr j).val
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨ ∃ p ∈ owed, (yD o p).val ≠ w (.out p.1 p.2))) s.G τ
      ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) ∧
    (OwedWrongR P T₀ w owed →
      prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀) s.G τ
        ≤ s.Lm.escape (bad P T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) +
          ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))) ∧
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨
          ∃ p ∈ owed, (yD o p).val ≠ evalR P T₀ (fun m => w (.reg m)) p.1 p.2)) s.G τ
      ≤ (if OwedWrongR P T₀ w owed then
          s.Lm.escape (bad P T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) else 0) +
        ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end Sound

/-- **Protocol 2 with hidden reads is circuit-private.** The program of class `c` is its topology table, `Nc c` entries
(`tab C c` the table circuit `C` becomes at class `c`, `encE c e` an entry's bytes), registered as their hm96 commit strings
under the registrant's uniform salts (`hm96Scheme`); the class circuit's `--zk` tables `U c o v` read the class, the owed
outputs `o` and that registration `v` only. Under `hash-derived-key` at `δ₁` (`hKey`), where the tables at every
registration of `tab C c` meet `ZkWireFixed`'s conditions (`hZk`) and the honest witness satisfies them (`hSat`): for every
dummy table `p₀` and dummy witness `w₀`, every encoding of the session's views and every event of the registration's view
and that encoding, the real view and the simulator's are within `2·Nc c·δ₁ + zkGap (zs c)`, both ways. `circuit_private`
at that scheme. -/
theorem private_circuit_private_reads
    {Cls Circ Out : Type} {Ent : Cls → Type} {K W Hi VL : Type} [Field K] [Fintype K] [DecidableEq K]
    [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] (fits : Circ → Cls → Prop) (Nc : Cls → ℕ)
    (tab : Circ → (c : Cls) → Fin (Nc c) → Ent c) (encE : (c : Cls) → Ent c → List UInt8) (J : Cls → ℕ)
    (zs : (c : Cls) → Fin (J c) → Sizes)
    (U : (c : Cls) → Out → (Fin (Nc c) → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs c j))
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (c : Cls) (o : Out) (C : Circ) (w : (Fin (Nc c) → Salt) → Fin (J c) → W)
    (hZk : ∀ r, ZkOk (U c o (hm96Reg (fun T j => rowBits (encE c (T j))) (tab C c) r)))
    (hSat : ∀ r, Sat (U c o (hm96Reg (fun T j => rowBits (encE c (T j))) (tab C c) r)) (w r))
    (p₀ : Fin (Nc c) → Ent c) (w₀ : Fin (J c) → W) {Obs : Type}
    (encV : ∀ v, (∀ j, (U c o v j).tab.View) → Obs) (E : (Fin (Nc c) → Bits × ByteArray) → Obs → Prop) :
    realPr (hm96Scheme fits tab Nc (fun c T j => encE c (T j)) J zs U) c o (tab C c) w encV E
        ≤ simPr (hm96Scheme fits tab Nc (fun c T j => encE c (T j)) J zs U) c o p₀ w₀ encV E +
          (2 * Nc c * δ₁ + zkGap (zs c)) ∧
      simPr (hm96Scheme fits tab Nc (fun c T j => encE c (T j)) J zs U) c o p₀ w₀ encV E
        ≤ realPr (hm96Scheme fits tab Nc (fun c T j => encE c (T j)) J zs U) c o (tab C c) w encV E +
          (2 * Nc c * δ₁ + zkGap (zs c)) := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
