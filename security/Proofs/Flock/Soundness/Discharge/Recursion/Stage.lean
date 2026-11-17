module

public import Proofs.Flock.Soundness.Assumptions.CustodyZk
public import Proofs.Flock.Soundness.Discharge.Composed.Sound
public import Proofs.Flock.Soundness.Discharge.Recursion.Defs
public import Proofs.Flock.Soundness.Discharge.Recursion.Flock
public import Proofs.Flock.Soundness.Discharge.Recursion.Fork
public import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
import Proofs.Flock.Soundness.Game
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
import Mathlib.Tactic.ReduceModChar

/-! # What V*'s bridge obligation reads of `Recursive.Stage`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg

open ZkHidden (planZHJ tabZHJ)

open scoped ENNReal

section Fixed

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- **V*'s commit strings**: unit `u`'s row of port `q`, or a single gate (`Readers.ofRows`' positions at
`HiddenExec.keyReadersH`). -/
abbrev PosK : Type := (Fin n × Fin c.ports.size) ⊕ Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N

/-- **Their values**: bit strings of at most the widest row's `16 · words` bits, with their width. -/
abbrev ValK : Type := Layout.BitVal (Finset.univ.sup (Layout.portRowW c) + 1)

/-- **Each commit string's leaf scheme**: `hm96X` of the bytes its port's hash reads (`HiddenExec.encKeyH`:
`sha512/row/v1`'s, `/row/v2`'s or `/row-seg/v1`'s, `HiddenExec.encPortH`) at a row, `SHA-512(bitCode v)` at a single
gate. -/
noncomputable def schemeK (p : PosK c hU n) : Binding.ExplicitLeaf (ValK c) :=
  if p.isLeft = true then
    Binding.ExplicitLeaf.hm96X (HiddenExec.encKeyH c p) (HiddenExec.extKeyH c p) (HiddenExec.extKeyH_spec c p)
  else Binding.ExplicitLeaf.plain Layout.bitCode Layout.bitCode_inj

/-- **V*'s openings**: a commit string, a value and a salt, opening the commit string's leaf of the value under the
salt; two openings of one commit string's leaf to different values are a SHA-512 collision (`ExplicitLeaf.pair`). -/
noncomputable def obK : Openings H512 (ValK c) where
  Op := PosK c hU n × ValK c × List UInt8
  Dg := PosK c hU n × Dig
  com o := (o.1, (schemeK c hU n o.1).leaf o.2.1 o.2.2)
  val o := o.2.1
  collide o o' := (schemeK c hU n o.1).pair o.2.1 o.2.2 o'.2.1 o'.2.2
  collide_spec o o' hc hv := by
    obtain ⟨p, v, y⟩ := o
    obtain ⟨p', v', y'⟩ := o'
    simp only [Prod.mk.injEq] at hc
    obtain ⟨rfl, hl⟩ := hc
    exact (schemeK c hU n p).pair_spec v y v' y' hv hl

/-- The commit string holding wire `g` (`Readers.ofRows`' `pos`): its row when `HiddenExec.rowOfH` has one (an input
row's bit, or a returned output's bit of its output row), else its own. -/
noncomputable def posK (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N) : PosK c hU n :=
  match HiddenExec.rowOfH c hU n (Layout.outsOf c hU n) g with
  | some (u, ⟨q, _⟩) => .inl (u, q)
  | none => .inr g

/-- Wire `g`'s place in its commit string's value (`Readers.ofRows`' `idx`): bit `k` of its row, or a single gate's
only bit. -/
noncomputable def idxK (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N) :
    Fin (Finset.univ.sup (Layout.portRowW c) + 1) :=
  match HiddenExec.rowOfH c hU n (Layout.outsOf c hU n) g with
  | some (_, ⟨q, k⟩) => Fin.castLE ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) k
  | none => ⟨0, Nat.succ_pos _⟩

/-- Wire `g`'s bit in the value `v` of its commit string. -/
noncomputable def bitK (v : ValK c) (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N) : Bool :=
  v.2 (idxK c hU n g)

end Fixed

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

variable {Reg₂ : Type}

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **V*'s public files register the stages' commitments**: at every history `t` and registration `R₂`, where `t`'s
stage-`i` commitment is `e.1` and session `j` accepts a transcript it writes at a registration `R'` (`ZkOuter.Writes`,
an outcome its game reaches), the digest session `j`'s value binding registers there at each commit string of
`Sk i j` (`cs`, the public file's leaf there) is `cmt j e.1`'s. A session that accepts nothing it writes may be another
history's, registering that history's commitments; the transfer reads only the registrations where a session accepts
an outcome of its game. -/
def VCommits (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg) : Prop :=
  ∀ i R ω t (e : Cm × I.Coin), t[i]? = some e → ∀ R₂ j R' tr, (zs R ω t R₂ j).Writes R' tr →
    AcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own tr → ∀ pos ∈ Sk i j, (zs R ω t R₂ j).vb.cs R' pos = cmt j e.1 pos

variable (Vs) in
/-- **Wire `g` of session `j` is read**: a committed wire of some unit. -/
def ReadW (j : Fin S) (g : Fin ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).circuit
    (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).N) : Prop :=
  ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u

/-- **`VDecodes` at the read wires**: layers whose bits on the read wires of each `Sk i j` are those of the values `v j`
hold a round-`i` message the inner verifier reads as `msg i v`, at any coins. It asks more than `VDecodes` (the message
must be fixed by the read wires alone), and it is true at V* when every wire `wmsg` depends on is some unit's `io`,
which is what VBridge G's `vbridgeRead_assembled` builds. -/
def VDecodesR (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  ∀ i < I.r, ∀ (ρs : List I.Coin) (Xt : (j : Fin S) → (Vs j).Bits)
    (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c),
    (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j → ReadW Vs j g →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (v j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) →
    Option.Rel (same i ρs) (wmsg Xt)[i]? (some (msg i v))

/-- **`VFull` at the read wires**: each read wire of `Sk i j` (`i < r`) is a wire of some unit, which holds of every read
wire (`vFullR_of_read`). -/
def VFullR (r : ℕ) (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) : Prop :=
  ∀ i < r, ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j → ReadW Vs j g →
    ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u

theorem vFullR_of_read (r : ℕ) (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) : VFullR r Sk :=
  fun _ _ _ _ _ h => h

end FlockSoundness.Discharge.Recursion
