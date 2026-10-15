import Definitions.Nci
import NciCheck.Check

/-!
# The NCI check's properties, stated against the Lean function

`NciCheck.evaluate` (`verity/core/protocols/network/nci/lean/NciCheck/Check.lean`, package `nci_check`) is
the verifier's check of a declared division against the NCI policy, `Nci.Complies Γ α` (`Definitions/Nci.lean`), and
`NciCheck.check` is `evaluate` with a refusal written `{"error": code}`. The difftest holds it to the Python reference on
random instances; the statements here say what it computes on every instance. Each is a named `Prop`, proved as
`NciCheck.Properties.SecurityProofs.X : NciCheck.Properties.X` in `Properties.Proofs.Nci`, and a guarantee of this
package (`lean-audit.json`).

Each statement has one shape (the lean-proofs skill, "One shape"): for every instance, if the check gives a verdict (or
doesn't refuse with a code), its named conclusion holds. The converses, that the check refuses exactly the malformed
and gives a verdict on every well-formed instance, are completeness, in `Properties.Proofs.Nci`, each with an instance.

**Refusals: a malformed instance is refused with its code.**

* `RejectsExactly`: if `evaluate j` doesn't refuse with code `k`, then not `Malformed j k` (the first condition `j`
  fails, in the order the check reads them, is `k`'s): so a malformed instance is refused, with that condition's code.
  The conditions are written over `j`'s JSON and what C-Flock's decoder returns for the program (`Flock.Program.decode`,
  then `Prog.gate` per gate), never over the check's own state.
* `AcceptsExactly`: `evaluate j` gives a verdict only when `WellFormed j`.
* `CheckRefuses`, `CheckAccepts`: the same of `NciCheck.check`, whose `error` field is the refusal's code and which has
  none on a verdict.

**A verdict is the definition's.** The decoded Program is the definition's circuit as the check's docstring maps
descriptor v1 to it (`LaysOut`: gate `g` of the canonical layout is gate `g`, of its primitive's width `Value<w>`, an input
gate when its primitive is an input node, reading the gates its operands name). `LaysOut` fixes a circuit's gates,
widths, input gates and reads, which is all `size` and `Nci.novelReads` read, and leaves its operations and outputs free.

* `Decodes`: on every instance with a verdict, the Program lays out a circuit (`Verity.Circuit`, its gates in
  topological order and its input gates reading nothing), `theta` names a baseline of it (`Verity.Declaration`), and the
  owners are a partition of its computed gates into IUs `0 … n - 1`, none empty (`Verity.Partition`, `Divides`).
* `Reports`: for every declaration `d` laid out by the Program with the baseline `theta` names, and the partition `U`
  the owners are, the verdict's `gates` is `d`'s number of gates, its `ius` lists per IU `u` its `size` and
  `Nci.novelReads` exactly, its `complies` is `true` exactly when every IU has at most `Γ` gates and NCI at most
  `α = num / den` (`Meets`), and its `failure` is the first IU that doesn't, with the rule it breaks first.
* `Witnesses`: a verdict whose `complies` is `true` means the declaration complies, `Nci.Complies Γ (num / den) d`,
  with the owners as the witness.

No statement takes an assumption. What is not proved about the decoder (its `Prog.gate` is `partial`) is not needed:
the statements read the decoder only through what it returns.
-/

namespace NciCheck.Properties

open Lean (Json toJson)
open Flock.Program (Prog)

/-! ## The instance -/

/-- `j` has the five fields, `program` a string and `theta` and `owner` arrays. -/
def Shaped (j : Json) : Prop :=
  (∃ s, j.getObjVal? "program" = .ok (.str s)) ∧ (∃ a, j.getObjVal? "theta" = .ok (.arr a)) ∧
    (∃ a, j.getObjVal? "owner" = .ok (.arr a)) ∧ (∃ v, j.getObjVal? "gamma" = .ok v) ∧
    ∃ v, j.getObjVal? "alpha" = .ok v

/-- `j`'s fields are `i`'s: the program's text, θ and the owners as arrays, `Γ` a natural, and `α` a pair of naturals
`[num, den]` with `den > 0`. -/
def Parses (j : Json) (i : Instance) : Prop :=
  j.getObjVal? "program" = .ok (.str i.program) ∧ j.getObjVal? "theta" = .ok (.arr i.theta) ∧
    j.getObjVal? "owner" = .ok (.arr i.owner) ∧ (∃ v, j.getObjVal? "gamma" = .ok v ∧ natOf? v = some i.gamma) ∧
    (∃ a b, j.getObjVal? "alpha" = .ok (.arr #[a, b]) ∧ natOf? a = some i.num ∧ natOf? b = some i.den) ∧ 0 < i.den

/-! ## The Program's gates, as the decoder returns them -/

/-- The number of gates of `P`'s canonical layout. -/
def gates (P : Prog) : ℕ := (P.defs[P.root]!).gates

/-- Gate `g`'s primitive `p` and operands `ops` are located, `p` returns one `Value<w>`, and `p` is an input node only
with no operands. -/
def Located (P : Prog) (g p : ℕ) (ops : Array ℕ) (w : ℕ) : Prop :=
  (∃ sc, P.gate g = .ok (p, ops, sc)) ∧ (P.defs[p]!).ret = .value w ∧
    (Flock.Program.isInputId (P.defs[p]!).key = true → ops = #[])

/-- Gate `g` is as the definition's circuit needs: located, and reading only earlier gates. -/
def GateOk (P : Prog) (g : ℕ) : Prop :=
  ∃ p ops w, Located P g p ops w ∧ ∀ a ∈ ops, a < g

/-- Gate `g`'s refusal `k`: `program` where it is not located, else `not-topological` where it reads itself or a later
gate. -/
def GateFault (P : Prog) (g : ℕ) (k : String) : Prop :=
  (k = "program" ∧ ¬ ∃ p ops w, Located P g p ops w) ∨
    (k = "not-topological" ∧ ∃ p ops w, Located P g p ops w ∧ ∃ a ∈ ops, g ≤ a)

/-- Gate `g` is an input gate: its primitive is an input node (`Input<w>_v1`, `RegisteredInput<w>_v1`). -/
def IsInput (P : Prog) (g : ℕ) : Prop :=
  ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ Flock.Program.isInputId (P.defs[p]!).key = true

/-! ## θ and the owners -/

/-- Entry `t` of θ names an input gate that no earlier entry names. -/
def ThetaOk (P : Prog) (theta : Array Json) (t : ℕ) : Prop :=
  ∃ g, natOf? theta[t]! = some g ∧ g < gates P ∧ IsInput P g ∧ ∀ t' < t, natOf? theta[t']! ≠ some g

/-- Entry `t` of θ's refusal `k`: `theta-unknown` where it is not a gate id, else `theta-not-input` where its gate is
not an input gate, else `theta-duplicate` where an earlier entry names the same gate. -/
def ThetaFault (P : Prog) (theta : Array Json) (t : ℕ) (k : String) : Prop :=
  (k = "theta-unknown" ∧ ∀ g, natOf? theta[t]! = some g → gates P ≤ g) ∨
    (k = "theta-not-input" ∧ ∃ g, natOf? theta[t]! = some g ∧ g < gates P ∧ ¬ IsInput P g) ∨
    (k = "theta-duplicate" ∧ ∃ g, natOf? theta[t]! = some g ∧ g < gates P ∧ IsInput P g ∧
      ∃ t' < t, natOf? theta[t']! = some g)

/-- Gate `g`'s owner is `"input"` and `g` an input gate, or an IU and `g` a computed gate. -/
def OwnerOk (P : Prog) (owner : Array Json) (g : ℕ) : Prop :=
  (owner[g]! = .str "input" ∧ IsInput P g) ∨ ((∃ u, natOf? owner[g]! = some u) ∧ ¬ IsInput P g)

/-- Gate `g`'s owner's refusal `k`: `owner-entry` where it is neither `"input"` nor a natural, `input-in-iu` where it
puts an input gate in an IU, `gate-in-no-iu` where it puts a computed gate in none. -/
def OwnerFault (P : Prog) (owner : Array Json) (g : ℕ) (k : String) : Prop :=
  (k = "owner-entry" ∧ owner[g]! ≠ .str "input" ∧ natOf? owner[g]! = none) ∨
    (k = "input-in-iu" ∧ (∃ u, natOf? owner[g]! = some u) ∧ IsInput P g) ∨
    (k = "gate-in-no-iu" ∧ owner[g]! = .str "input" ∧ ¬ IsInput P g)

/-- An IU below one that an owner names holds no gate. -/
def EmptyIU (owner : Array Json) : Prop :=
  ∃ g u, g < owner.size ∧ natOf? owner[g]! = some u ∧ ∃ u' < u, ∀ g' < owner.size, natOf? owner[g']! ≠ some u'

/-- The first of steps `0 … n - 1` that is not `ok` fails with `k`. -/
def FirstFault (ok : ℕ → Prop) (fault : ℕ → String → Prop) (n : ℕ) (k : String) : Prop :=
  ∃ i < n, (∀ i' < i, ok i') ∧ fault i k

/-! ## (1) Refusals -/

/-- `j` is refused with `k`: the first condition it fails, in the order the check reads them, is `k`'s. First the
fields (`instance`), `Γ` and `α` (`params`) and the program's decoding (`program`), then gate by gate (`GateFault`), then
θ entry by entry (`ThetaFault`), then one owner per gate (`owner-length`), then gate by gate (`OwnerFault`), then the IUs
(`empty-iu`). -/
def Malformed (j : Json) (k : String) : Prop :=
  (¬ Shaped j ∧ k = "instance") ∨
  (Shaped j ∧ (¬ ∃ i, Parses j i) ∧ k = "params") ∨
  ∃ i, Parses j i ∧
    (((∃ e, Flock.Program.decode i.program.toUTF8 = .error e) ∧ k = "program") ∨
     ∃ P, Flock.Program.decode i.program.toUTF8 = .ok P ∧
      (FirstFault (GateOk P) (GateFault P) (gates P) k ∨
       (∀ g < gates P, GateOk P g) ∧
        (FirstFault (ThetaOk P i.theta) (ThetaFault P i.theta) i.theta.size k ∨
         (∀ t < i.theta.size, ThetaOk P i.theta t) ∧
          ((i.owner.size ≠ gates P ∧ k = "owner-length") ∨
           i.owner.size = gates P ∧
            (FirstFault (OwnerOk P i.owner) (OwnerFault P i.owner) (gates P) k ∨
             (∀ g < gates P, OwnerOk P i.owner g) ∧ EmptyIU i.owner ∧ k = "empty-iu")))))

/-- `j` is well-formed: its fields parse, its program decodes, and every gate, every entry of θ and every owner is as
stated, with no IU empty. -/
def WellFormed (j : Json) : Prop :=
  ∃ i P, Parses j i ∧ Flock.Program.decode i.program.toUTF8 = .ok P ∧ (∀ g < gates P, GateOk P g) ∧
    (∀ t < i.theta.size, ThetaOk P i.theta t) ∧ i.owner.size = gates P ∧ (∀ g < gates P, OwnerOk P i.owner g) ∧
    ¬ EmptyIU i.owner

/-- **A malformed instance is refused with its condition's code**, stated as its contrapositive: whatever `evaluate j`
doesn't refuse with `k` is not `Malformed j k`. (Its converse, that a refusal with `k` is `Malformed j k`, is
completeness: `malformed_of_refused`.) -/
def RejectsExactly : Prop :=
  ∀ (j : Json) (k : String), evaluate j ≠ .error k → ¬ Malformed j k

/-- **A verdict only on a well-formed instance.** (Its converse, that a well-formed instance gets a verdict, is
completeness: `verdict_of_wellFormed`.) -/
def AcceptsExactly : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → WellFormed j

/-- **`check` refuses a malformed instance with its condition's code**, stated as its contrapositive: whatever `check j`
doesn't give the `error` field `k` is not `Malformed j k`. (Its converse is completeness: `check_malformed_of_refused`.) -/
def CheckRefuses : Prop :=
  ∀ (j : Json) (k : String), (check j).getObjVal? "error" ≠ .ok (.str k) → ¬ Malformed j k

/-- **`check` gives a verdict, one with no `error` field, only on a well-formed instance.** (Its converse is
completeness: `check_verdict_of_wellFormed`.) -/
def CheckAccepts : Prop :=
  ∀ j : Json, (∀ k, (check j).getObjVal? "error" ≠ .ok (.str k)) → WellFormed j

/-! ## (2) Verdicts -/

/-- `D` is the circuit Program `P` lays out: `P`'s gates, gate `g` of width `w` where its primitive returns `Value<w>`,
an input gate where its primitive is an input node, and reading the gates its operands name. -/
def LaysOut (P : Prog) (D : Verity.Circuit) : Prop :=
  D.N = gates P ∧ ∀ g : Fin D.N, ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value (D.w g) ∧
    D.isInput g = Flock.Program.isInputId (P.defs[p]!).key ∧ ∀ a : Fin D.N, a ∈ D.reads g ↔ a.val ∈ ops

/-- θ is the set of gates the entries of `theta` name. -/
def Names {D : Verity.Circuit} (theta : Array Json) (θ : Finset (Fin D.N)) : Prop :=
  ∀ g : Fin D.N, g ∈ θ ↔ ∃ t ∈ theta, natOf? t = some g.val

/-- The owners are the partition `U`, one per gate: `"input"` for a gate in no unit, `u` for a gate of unit `u`, and
no unit empty. -/
def Divides {D : Verity.Circuit} {n : ℕ} (owner : Array Json) (U : Verity.Partition D n) : Prop :=
  owner.size = D.N ∧
    (∀ g : Fin D.N, (U.unit g = none ↔ owner[g.val]! = .str "input") ∧
      ∀ u : Fin n, U.unit g = some u ↔ natOf? owner[g.val]! = some u.val) ∧
    ∀ u, (U.gates u).Nonempty

/-- IU `u` of `U` has at most `Γ` gates and NCI at most `α`, the policy's two bounds as `Nci.Complies` states them. -/
def Meets (d : Verity.Declaration) {n : ℕ} (U : Verity.Partition d.D n) (Γ : ℕ) (α : ℝ) (u : Fin n) : Prop :=
  d.D.size (U.gates u) ≤ Γ ∧ (Nci.novelReads d (U.gates u) : ℝ) ≤ α * d.D.size (U.gates u)

/-- **What a verdict is about.** On every instance with a verdict, the Program lays out a circuit of the definition's,
the entries of `theta` name a baseline of it, and the owners are a partition of its computed gates. -/
def Decodes : Prop :=
  ∀ (j out : Json), evaluate j = .ok out →
    ∃ i P, Parses j i ∧ Flock.Program.decode i.program.toUTF8 = .ok P ∧
      ∃ D : Verity.Circuit, LaysOut P D ∧ (∃ θ : Finset (Fin D.N), Names i.theta θ ∧ ∀ g ∈ θ, D.isInput g = true) ∧
        ∃ (n : ℕ) (U : Verity.Partition D n), Divides i.owner U

/-- The verdict `out` on `j` is the definition's: for every declaration `d` the Program lays out with the baseline
`theta` names, and the partition `U` the owners are, at `Γ = gamma` and `α = num / den`: `gates` is `d`'s number of
gates; `ius` lists per IU its `size` and `novelReads`; `complies` is `true` exactly when every IU meets both bounds; and
`failure` is `null` when every IU does, and otherwise the first IU `u` that doesn't, with `size` when it has more than `Γ`
gates and else `nci`. -/
def Reported (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration) (n : ℕ) (U : Verity.Partition d.D n),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    Divides i.owner U →
    let α : ℝ := (i.num : ℝ) / i.den
    out.getObjVal? "gates" = .ok (toJson d.D.N) ∧
    out.getObjVal? "ius" = .ok (Json.arr (Array.ofFn fun u : Fin n =>
      Json.arr #[toJson (d.D.size (U.gates u)), toJson (Nci.novelReads d (U.gates u))])) ∧
    (out.getObjVal? "complies" = .ok (.bool true) ↔ ∀ u, Meets d U i.gamma α u) ∧
    (out.getObjVal? "complies" = .ok (.bool false) ↔ ¬ ∀ u, Meets d U i.gamma α u) ∧
    (out.getObjVal? "failure" = .ok .null ↔ ∀ u, Meets d U i.gamma α u) ∧
    ∀ (u : Fin n) (rule : String),
      out.getObjVal? "failure" = .ok (Json.arr #[toJson u.val, .str rule]) ↔
        (∀ u' < u, Meets d U i.gamma α u') ∧
          ((rule = "size" ∧ i.gamma < d.D.size (U.gates u)) ∨
           (rule = "nci" ∧ d.D.size (U.gates u) ≤ i.gamma ∧
             α * d.D.size (U.gates u) < (Nci.novelReads d (U.gates u) : ℝ)))

/-- **The verdict is the definition's.** On every instance with a verdict, the verdict is `Reported`. -/
def Reports : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → Reported j out

/-- When the verdict `out` on `j` has `complies` `true`, every declaration the Program lays out with the baseline
`theta` names complies with `Nci.Complies Γ (num / den)`. -/
def Witnessed (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) → Nci.Complies i.gamma ((i.num : ℝ) / i.den) d

/-- **A verdict of compliance witnesses the policy.** On every instance with a verdict, the verdict is `Witnessed`. -/
def Witnesses : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → Witnessed j out

end NciCheck.Properties
