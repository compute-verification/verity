import FlockSoundness.Discharge.PrivateCircuit.InstanceDefs
import FlockSoundness.Discharge.PrivateCircuit.Interp

/-!
# Protocol 2 at the universal unit: the program's rows, one set per Call

The program is registered per Call: `L = N · L_u` rows, row `(j, r)` being row `r` of unit `j`'s program `p j`, each
under its own salt. Rows shared across units would publish which units are equal.

The instance's objects already take that layout, because they are generic in the row count and the rows:

* **Soundness** (`ProgSites rd rt L`, `row : c.Prog → Fin L → Val`, `hreg`). Each of the `L` sites is its own commit
  string of the layout, with its own salt `rd.ws (ps.site i)`, and `row` is any map from the program to its rows. The
  per-Call layout is `L := units c * L_u` and `row := callRow rowU`, and `hreg_call` gives `hreg` from the registration of
  each unit's rows. In this layout a unit's rows are its own sites, so the checked sites of the drawn units bind the
  drawn units' programs only: that is why a forged program at undrawn units needs the law's miss (`ForgedMiss`).
* **Hiding** (`hm96Scheme fits pad L enc …`, `private_circuit_private`). The registration's view is `hm96Reg` of the
  `L c` rows under `L c` independent uniform salts, so the per-Call layout is `L c := units c * L_u c` and
  `enc := callEnc encU`, and the hiding term is `2 · N · L_u · δ₁`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Registered Binding Audit

/-- **The per-Call rows of a program**: row `(j, r)` (`finProdFinEquiv`) is row `r` of unit `j`'s program. -/
def callRow {c : Universal.Cls} {Lu : ℕ} {Val : Type} (rowU : c.Unit → Fin Lu → Val) (p : c.Prog) :
    Fin (Universal.units c * Lu) → Val :=
  fun i => rowU (p (finProdFinEquiv.symm i).1) (finProdFinEquiv.symm i).2

/-- **The per-Call row bytes of a class's programs**, for `hm96Scheme`: row `(j, r)` is row `r` of unit `j`'s bytes. -/
def callEnc (Lu : Universal.Cls → ℕ) (encU : (c : Universal.Cls) → c.Unit → Fin (Lu c) → List UInt8) :
    (c : Universal.Cls) → c.Prog → Fin (Universal.units c * Lu c) → List UInt8 :=
  fun c p => callRow (encU c) p

section Sites

variable {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay}

/-- **The per-Call registration gives `hreg`**: if the committed value at the program's site `(j, r)` is row `r` of
unit `j`'s program `p j`, for every unit `j` and row `r`, then every site holds the per-Call row of `p`. -/
theorem hreg_call {c : Universal.Cls} {Lu : ℕ} {rd : Registered.Reads lay rs} {rt : Flock.Registered.Port}
    (ps : ProgSites rd rt (Universal.units c * Lu)) (rowU : c.Unit → Fin Lu → lay.Val) (p : c.Prog)
    (h : ∀ (j : Fin (Universal.units c)) (r : Fin Lu), rd.wv (ps.site (finProdFinEquiv (j, r))) = rowU (p j) r) :
    ∀ i, rd.wv (ps.site i) = callRow rowU p i := by
  intro i
  have := h (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
  rw [Prod.mk.eta, Equiv.apply_symm_apply] at this
  exact this

end Sites

end FlockSoundness.Discharge.PrivateCircuit
