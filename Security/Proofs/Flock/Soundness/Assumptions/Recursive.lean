import Proofs.Flock.Recursive.Registrants

/-!
# V*'s bridge: the recursive audit's open obligation

Obligation: `VBridge` is a fact about our own code (V*'s circuits and the verifier), not about the world, so it is an
obligation, not a premise. VBridge G (`Proofs.Flock.VBridge`) discharges it, and it stays a hypothesis of
`RecursiveSound`, and so in this module, only until that lands.

In the recursive audit (`Proofs.Flock.Recursive.Defs`: `recGame`), the developer's firewall sends the auditor a salted
SHA-512 commitment to each inner round message, relays the auditor's live coin for the round, and afterwards proves, in
C-Flock `--zk --registered` sessions of the public verifier circuits V* (`Vs j`, one session each, in a row), that
openings of those commitments exist on which the inner verifier accepts. `VBridge` is what V*'s circuits must compute
for those sessions to mean it: V* accepts at its public inputs (the firewall's commitments registered in its public
files) and at the values it registers itself (the rounds' coins and the comb `coef`, in its record `ZkOuter.rd`) only
when the inner verifier accepts the inner transcript those commitments open to.

C-Flock's definitions are still in its proofs package (`Proofs.Flock.Soundness`), and so is the recursion's model
(`Proofs.Flock.Recursive`), so this module imports `Proofs.Flock.Recursive.Registrants`, and `Models.lean`, which the `security`
package builds without `security_proofs`, does not import it: `Proofs.Flock.Recursive` does.
-/

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

/-- Obligation: discharged by VBridge G (`Proofs.Flock.VBridge`), with `VCommits` derived there from acceptance
(`PubCommits`).

**`v-bridge`: V*'s circuits compute the inner verifier** (V*'s session `j` is of the circuit `(Vs j).c`, pinned by
its file's digest through `ZkOuter.hc`, under the law `(Vs j).law`; `k, Rw` the count curve's). Four facts about V*'s
circuits, laws and inputs, each read at every history `(R, ω, t)` (the registration, the draw of the hidden circuit's
units, the rounds' commitments and coins) and every registration `R₂` of the outer proof (`outerV`: the values V*'s
sessions read that the history does not fix, its running sums `rec-acc` and with `link_rows` its opened rows
`rec-rows-L<level>`, sent after the last inner coin and before any session):

* **they accept only an accepted inner transcript** (`VBridgeOff RegColl`): where no two of V*'s sessions' registrant
  records clash (`RegColl`: two registered reads whose openings or hm96 leaves are a SHA-512 collision, which the
  recursive guarantees answer with a registrant fork, `regFinder`), at all strategies of V*'s sessions there, when
  every session's outcome `flock-verify verify --zk --registered` accepts and every session's value layer is
  `Good` (no wrong unit, and at every unit each registered read carries the registrant's value: the hidden circuit's
  rows as registered, `R₂`'s values, and the values V* registers itself, the coins and the comb `coef`), the messages
  `wmsg` reads from the layers hold an inner transcript the inner verifier accepts on the hidden circuit's statement
  `x R ω` at the rounds' coins;
* **their public files register the firewall's commitments** (`VCommits`): the digest session `j`'s value binding
  registers at each commit string of stage `i`'s `Sk i j` is the firewall's stage-`i` commitment's, `cmt j`, at every
  `R₂` where session `j` accepts some transcript (the verifier ties a public file's rows to its registration only on an
  accepted run);
* **they decode each round's message from those commit strings** (`VDecodesR`, at equality): layers whose bits on the
  wires of each `Sk i j` that some unit reads (`ReadW`) are those of the values `v j` hold `msg i v` as their round-`i`
  message (true at V* when every wire `wmsg` depends on is some unit's `io`, which is what VBridge G's
  `vbridgeRead_assembled` builds);
* **their laws draw every such wire** (`VFullR`): each read wire of `Sk i j` is a wire of a unit session `j`'s law
  draws. Every unit is drawn under the full draw, so this holds of every `Sk` (`vFullR_of_read`).

Both read only the wires some unit reads: V*'s rows pad a port to whole words, and no unit reads the padding.

The first is the soundness direction of "V* accepts exactly when the Lean verifier accepts the inner session the
commitments commit to"; the recursive guarantees read no other. -/
def VBridge {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
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
def VBridgeC {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  VBridgeOff (RegColl zs) zs dg N k Rw x wmsg ∧ VCommits zs (fun i j => if i + 1 < I.r then Sk i j else ∅) cmt ∧
    VDecodesR wmsg (fun i _ a b => i + 1 < I.r → a = b) (fun i j => if i + 1 < I.r then Sk i j else ∅) msg ∧
    VFullR I.r (fun i j => if i + 1 < I.r then Sk i j else ∅)

/-- `VCommits` at the commit strings of the rounds before the last of `r`. -/
theorem vCommits_masked {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg} (r : ℕ)
    (h : VCommits zs Sk cmt) : VCommits zs (fun i j => if i + 1 < r then Sk i j else ∅) cmt := by
  intro i R ω t e he R₂ j hacc pos hp R'
  by_cases hc : i + 1 < r
  · exact h i R ω t e he R₂ j hacc pos (by simpa [hc] using hp) R'
  · exact absurd hp (by simp [hc])

/-- **`VBridge` gives `VBridgeC`** (this direction only), at a decoder that holds a message at the last round, as
`wmsgOf`'s does at every round: the committed rounds' fields are `VBridge`'s, and at the last round `VBridgeC` asks only
that message. -/
theorem vbridgeC_of_vbridge {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
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

/-- Obligation, as `VBridgeC`, off one more bad event `Bad`.

**`v-bridge` at the committed rounds, off `RegColl` and `Bad`**: `VBridgeC` field for field, its first read only where
neither two sessions' registrant records clash (`RegColl`, `regFinder`'s) nor `Bad` holds. At candidate 3, `Bad` is
`Recursion.ValueColl` at the verifier's rows (`valueSrcAll`): a registrant's row off a value the verifier registers itself
(`coef`, `dirs`, `v`), by a SHA-512 collision `valueFinder` outputs, the case `FlockC3.RegLevels.value_tie` leaves. This
module cannot name it (its module imports this one), so `Bad` is a parameter; RegColl is not widened. `VBridgeC` gives it at
every `Bad` (`vbridgeCOff_of_vbridgeC`). -/
def VBridgeCOff {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  VBridgeOff (fun R ω t R₂ => RegColl zs R ω t R₂ ∨ Bad R ω t R₂) zs dg N k Rw x wmsg ∧
    VCommits zs (fun i j => if i + 1 < I.r then Sk i j else ∅) cmt ∧
    VDecodesR wmsg (fun i _ a b => i + 1 < I.r → a = b) (fun i j => if i + 1 < I.r then Sk i j else ∅) msg ∧
    VFullR I.r (fun i j => if i + 1 < I.r then Sk i j else ∅)

/-- **`VBridgeC` gives `VBridgeCOff` at every `Bad`**: off more, the bridge asks less. -/
theorem vbridgeCOff_of_vbridgeC {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop)
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {x : Reg → L.Ω → I.Stmt}
    {wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg}
    (hb : VBridgeC zs dg N k Rw x wmsg Sk cmt msg) : VBridgeCOff Bad zs dg N k Rw x wmsg Sk cmt msg :=
  ⟨fun R ω t R₂ hn => hb.1 R ω t R₂ fun h => hn (Or.inl h), hb.2⟩

end Flock.Assumptions
