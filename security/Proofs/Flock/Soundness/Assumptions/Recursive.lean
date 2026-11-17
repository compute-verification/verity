module

public import Proofs.Flock.Recursive.Registrants
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

/-! V*'s bridge: the recursive audit's open obligation, a fact about V*'s circuits and the verifier rather than a premise. -/

@[expose] public section

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

/-- Obligation: discharged by VBridge G (`Proofs.Flock.VBridge`), with `VCommits` derived there from acceptance
(`PubCommits`).

**`v-bridge`: V*'s circuits compute the inner verifier** (V*'s session `j` is of the circuit `(Vs j).c`, pinned by
its file's digest through `ZkOuter.hc`, under the sampling design `(Vs j).samplingDesign`; `k, Rw` the count curve's). Four facts about V*'s
circuits, sampling designs and inputs, each read at every history `(R, ω, t)` (the registration, the sample of the hidden circuit's
units, the rounds' commitments and coins) and every registration `R₂` of the outer proof (`outerV`: the values V*'s
sessions read that the history does not fix, its running sums `rec-acc` and its opened rows and their siblings
`rec-rows-L<level>` and `rec-aux-L<level>`, sent after the last inner coin and before any session):

* **they accept only an accepted inner transcript** (`VBridgeOff RegColl`): where no two of V*'s sessions' registrant
  records clash (`RegColl`: two registered reads whose openings or hm96 leaves are a SHA-512 collision, which the
  recursive guarantees answer with a registrant fork, `regFinder`), at all strategies of V*'s sessions there and at
  every record that is a genuine session-game outcome (each session's record is one its game reaches against its
  strategy, `Game.Outcome`; an accepted record is always such an outcome, since the audit only ever accepts what its
  sessions play), when every session's outcome `verity-verify-proof verify --zk --registered` accepts and every session's
  value layer is `Good` (no incorrect proof block, and at every unit each registered read carries the registrant's value: the
  hidden circuit's rows as registered, `R₂`'s values, and the values V* registers itself, the coins and the comb
  `coef`), the messages `wmsg` reads from the layers hold an inner transcript the inner verifier accepts on the hidden
  circuit's statement `x R ω` at the rounds' coins;
* **their public files register the ZK server's commitments** (`VCommits`): the digest session `j`'s value binding
  registers at each commit string of stage `i`'s `Sk i j` is the ZK server's stage-`i` commitment's, `cmt j`, at every
  `R₂` and registration `R'` where session `j` accepts a transcript it writes at `R'` (`ZkOuter.Writes`, an outcome
  its game reaches; the verifier ties a public file's rows to its registration only on an accepted run);
* **they decode each round's message from those commit strings** (`VDecodesR`, at equality): layers whose bits on the
  wires of each `Sk i j` that some unit reads (`ReadW`) are those of the values `v j` hold `msg i v` as their round-`i`
  message (true at V* when every wire `wmsg` depends on is some unit's `io`, which is what VBridge G's
  `vbridgeRead_assembled` builds);
* **their sampling designs sample every such wire** (`VFullR`): each read wire of `Sk i j` is a wire of a unit session `j`'s sampling design
  samples. Every unit is sampled under the full sample, so this holds of every `Sk` (`vFullR_of_read`).

Both read only the wires some unit reads: V*'s rows pad a port to whole words, and no unit reads the padding.

The first is the soundness direction of "V* accepts exactly when the Lean verifier accepts the inner session the
commitments commit to"; the recursive guarantees read no other. -/
def VBridge {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  VBridgeOff (RegColl zs) zs dg N k Rw x wmsg ∧ VCommits zs Sk cmt ∧ VDecodesR wmsg (fun _ _ a b => a = b) Sk msg ∧
    VFullR I.r Sk

/-- Obligation, as `VBridge`, for a protocol whose last round is not committed (candidate 3's).

**`v-bridge` at the committed rounds**: `VBridge` field for field, with the stages' commit strings only at the rounds
before the last (`i + 1 < r`: no commit string of the last round, so no commitment registered or decoded there), and
at the last round the decoder holding some message, the one the recursive prover's own leaf sends (`VDecodesR` there
reads no wire and asks no equality). `vbridgeC_of_vbridge` gives it from `VBridge` at a decoder that holds a last
message. -/
def VBridgeC {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  VBridgeOff (RegColl zs) zs dg N k Rw x wmsg ∧ VCommits zs (fun i j => if i + 1 < I.r then Sk i j else ∅) cmt ∧
    VDecodesR wmsg (fun i _ a b => i + 1 < I.r → a = b) (fun i j => if i + 1 < I.r then Sk i j else ∅) msg ∧
    VFullR I.r (fun i j => if i + 1 < I.r then Sk i j else ∅)

theorem vCommits_masked {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg} (r : ℕ)
    (h : VCommits zs Sk cmt) : VCommits zs (fun i j => if i + 1 < r then Sk i j else ∅) cmt := by
  intro i R ω t e he R₂ j R' tr hw hacc pos hp
  by_cases hc : i + 1 < r
  · exact h i R ω t e he R₂ j R' tr hw hacc pos (by simpa [hc] using hp)
  · exact absurd hp (by simp [hc])

/-- This direction only. -/
theorem vbridgeC_of_vbridge {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {x : Reg → L.Ω → I.Stmt}
    {wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg}
    (hb : VBridge zs dg N k Rw x wmsg Sk cmt msg) (hlast : ∀ Xt, I.r - 1 < (wmsg Xt).length) :
    VBridgeC zs dg N k Rw x wmsg Sk cmt msg := by
  refine ⟨hb.1, vCommits_masked I.r hb.2.1, fun i hi ρs Xt v hX => ?_, vFullR_of_read _ _⟩
  by_cases hc : i + 1 < I.r
  · have h := hb.2.2.1 i hi ρs Xt v fun j g hg hr => hX j g (by simpa [hc] using hg) hr
    revert h
    generalize (wmsg Xt)[i]? = o
    intro h
    cases h with
    | some h => exact Option.Rel.some (by intro _; exact h)
  · rw [List.getElem?_eq_getElem (by have := hlast Xt; omega)]
    exact Option.Rel.some (by intro h; exact absurd h hc)

/-- Obligation, as `VBridgeC`, off `RegColl` and one more bad event `Bad`. `Bad` is a parameter because the
module of `Recursion.ValueColl`, the event it is at the verifier's rows, imports this one; `RegColl` is not widened. -/
def VBridgeCOff {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type}
    (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ →
      ((j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × FlockSoundness.Discharge.Exec.Transcript) → Prop)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  VBridgeOffOut (fun R ω t R₂ o => RegColl zs R ω t R₂ ∨ Bad R ω t R₂ o) zs dg N k Rw x wmsg ∧
    VCommits zs (fun i j => if i + 1 < I.r then Sk i j else ∅) cmt ∧
    VDecodesR wmsg (fun i _ a b => i + 1 < I.r → a = b) (fun i j => if i + 1 < I.r then Sk i j else ∅) msg ∧
    VFullR I.r (fun i j => if i + 1 < I.r then Sk i j else ∅)

theorem vbridgeCOff_of_vbridgeC {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ →
      ((j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × FlockSoundness.Discharge.Exec.Transcript) → Prop)
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {x : Reg → L.Ω → I.Stmt}
    {wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg}
    (hb : VBridgeC zs dg N k Rw x wmsg Sk cmt msg) : VBridgeCOff Bad zs dg N k Rw x wmsg Sk cmt msg :=
  ⟨fun R ω t R₂ τ o hout hn => hb.1 R ω t R₂ (fun h => hn (Or.inl h)) τ o hout, hb.2⟩

end Flock.Assumptions
