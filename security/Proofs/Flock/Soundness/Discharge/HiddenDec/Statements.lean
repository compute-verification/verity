module

public import Proofs.Flock.Soundness.Discharge.DecodeZL.OP
public import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Tables
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# The decoder at the hidden-output headline's samples: statements

`flock_headline_exec_hidden`'s open guarantee is `hdec : DecodesOn …` at the hidden tables `tablesAtH`.
`DecodesOnExec` (`Discharge/DecodeZL`) is the same guarantee at the plain headline's tables, under `Stmt.setupH`; the
decoder `decAt`, the number of extra claims `nX`, `RecordsLive` and the chunk lemmas (`zl_carries`, `op_carries` at
`wOP`, `lig_carries` at `wLIG`) are generic in the table or the setup and are DecodeZL's. What changes is where the sample
is set up:

* `TagsOkH`: DecodeZL's `TagsOk` with hidden outputs (`hiddenOutputs = true`); `TagsOkHL` also admits plain SHA-512
  leaves.
* `DrawOkH`: DecodeZL's `DrawOk` with `Stmt.setupHidden` for `Stmt.setupH`.
* `opnAtH`: DecodeZL's `opnAt` at `Stmt.setupHidden`'s statement.
* `DecodesOnExecH`: `DecodesOn` at the hidden tables and the tags' leaf (`leafOfId`), the decoder `decAt` at
  DecodeZL's wires `wOP` and `wLIG`, `one := true`, no final messages, `capOfRoot`. `DecodesOnExecHSt` is it at the
  statement circuit (`HmOut.parseStmt`, a templated file's template unrolled).
-/

@[expose] public section

namespace FlockSoundness.Discharge.HiddenDec

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.HiddenExec

local notation "𝔽" => FlockLevel3.GF128

/-- **The verify command's tags under hidden outputs**: hidden outputs, retained rounds, `hm96-sha512/v1` leaves, one
table. -/
def TagsOkH (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1" ∧ I.count ≤ 1

/-- **The verify command's tags under hidden outputs, without `--zk`**: `TagsOkH` with `hm96-sha512/v1` or plain
SHA-512 leaves (`flock-leaf/sha512-unsalted`). -/
def TagsOkHL (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧
    (I.tags.merkleLeaf = "hm96-sha512/v1" ∨ I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") ∧ I.count ≤ 1

theorem TagsOkH.toL {I : Inputs} (h : TagsOkH I) : TagsOkHL I :=
  ⟨h.1, h.2.1, Or.inl h.2.2.1, h.2.2.2⟩

variable {n mPts : ℕ} {Reg : Type}

/-- **The challenger's sample at `S`, under hidden outputs**: it proves `S`, and setupHidden sets up at it with `mPts`
link points. -/
def DrawOkH (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts

/-- **The openings at sample `S`, under hidden outputs**: `opnOf` at the statement setupHidden sets up at the record's
sample `dj S` and its `fast100` schedule, on the first two proof files. -/
noncomputable def opnAtH (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) : Opens 𝔽 Dig × Salts (List UInt8) :=
  match Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program with
  | .ok st => match Flock.fast100 st.m with
    | some sch => opnOf (Setup.ofCircuit st) sch (nX T S R) #[proofs[0]!, proofs[1]!]
    | none => default
  | .error _ => default

/-- **The decoder is right at the hidden-output headline's samples** (the decoder premise): given the tags, `wr` the challenger's
record and the challenger's samples proving their units at `mPts` link points under setupHidden, `DecodesOn` holds at the
hidden tables, at the model's leaf of the tags' leaf scheme (`leafOfId`), with the decoder `decAt` at DecodeZL's wires,
single coins as field elements, no final messages and `capOfRoot`. -/
def DecodesOnExecH : Prop :=
  ∀ {c : Flock.Circuit} {n : ℕ} {I : Inputs} (_ht : I.tags.typed = false)
    (_hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (L : Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    TagsOkHL I → RecordsLive I dj wr → (∀ ω, DrawOkH I dj mPts (L.draw ω)) →
    DecodesOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH (Reg := Reg) dj x₀) I wr
      (opnAtH (tablesAtH dj x₀) I dj)
      (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L

/-- **`DecodesOnExecH` at the statement circuit**: on every circuit `HmOut.parseStmt` reads (a templated file's template
unrolled) whose unit returns the outputs. -/
def DecodesOnExecHSt : Prop :=
  ∀ {c : Flock.Circuit} {n : ℕ} {I : Inputs} (_ht : I.tags.typed = false)
    (_hc : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (L : Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    TagsOkHL I → RecordsLive I dj wr → (∀ ω, DrawOkH I dj mPts (L.draw ω)) →
    DecodesOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH (Reg := Reg) dj x₀) I wr
      (opnAtH (tablesAtH dj x₀) I dj)
      (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L

end FlockSoundness.Discharge.HiddenDec
