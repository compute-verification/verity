import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-!
# The binding half's sites at the statement circuit (PRF-133 G-B)

The binding half (`SoundBind.lean`'s `sitesShH` … `rsZH`) reads each site's circuit through `AcceptedH.c_eq`:
`HmRow.parse` of the site's files is the site's committed circuit. At a templated file the binding is to the statement
circuit `c = HmOut.unrolled st.c`, so what a site gives the key program is `HmOut.unrolled (si.acc …).st.c = c`:

* `TableH.stmt_eq`: every table's session commits the circuit the statement circuit unrolls (`stmt_of_setupHidden`).
* `AcceptedH.stmt_of_parse`, `SitesH.stmt_of_parse`: a flat site's `HmRow.parse` gives the same fact, so a lemma
  stated over it keeps its flat corollary.
* `tableShH_stmt`: the fact at every `--zk` hidden-output table's public twin, from `HmOut.parseStmt` at the `--zk` tags.
* `unrolled_ports` …: `HmOut.unrolled` changes only the unit net and `tmpl`, so every other field the key program
  reads (ports, leaves, `upv`, output rows) is the committed circuit's.
-/

namespace FlockSoundness

open Flock

section unrolled

variable (c : Circuit)

theorem unrolled_ports : (HmOut.unrolled c).ports = c.ports := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_outPorts : (HmOut.unrolled c).outPorts = c.outPorts := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_outRowPorts : (HmOut.unrolled c).outRowPorts = c.outRowPorts := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_leavesIn : (HmOut.unrolled c).leavesIn = c.leavesIn := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_leavesOut : (HmOut.unrolled c).leavesOut = c.leavesOut := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_upv : (HmOut.unrolled c).upv = c.upv := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_unitNet : (HmOut.unrolled c).unitNet = c.unitNet := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_leafCuts : (HmOut.unrolled c).leafCuts = c.leafCuts := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

end unrolled

end FlockSoundness

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit
open FlockSoundness.Discharge.Exec (Inputs)

/-- **A table's session commits the circuit the statement circuit unrolls**, on untyped tags. -/
theorem TableH.stmt_eq {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {c : Flock.Circuit}
    (ht : I.tags.typed = false) (hc : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c)
    (x : TableH I mPts S) : HmOut.unrolled x.st.c = c :=
  (stmt_of_setupHidden ht hc x.h).symm

/-- A site whose files `HmRow.parse` reads as `c` commits `c` itself, flat. -/
theorem AcceptedH.stmt_of_parse {T : Model.Statement} (a : AcceptedH T) {c : Flock.Circuit}
    (hp : HmRow.parse a.tags a.cf a.tables = .ok c) : HmOut.unrolled a.st.c = c := by
  rw [a.c_eq hp]
  exact unrolled_flat (parse_facts hp).tmpl

/-- `AcceptedH.stmt_of_parse` at every site. -/
theorem SitesH.stmt_of_parse {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {si : SitesH plan tab}
    {c : Flock.Circuit}
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c) :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), HmOut.unrolled (si.acc S R u h).st.c = c :=
  fun S R u h => (si.acc S R u h).stmt_of_parse (hparse S R u h)

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.ZkHidden

open Flock ZkExec ZkBind HiddenExec
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))

/-- **Every `--zk` table's public twin commits the circuit the statement circuit unrolls.** -/
theorem tableShH_stmt (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (S : Finset (Fin n)) :
    HmOut.unrolled (tableShH dj y₀ S).st.c = c :=
  TableH.stmt_eq (zkTyped ht) hc _

end FlockSoundness.Discharge.ZkHidden
