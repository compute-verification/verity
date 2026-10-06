import Proofs.Flock.Recursive.Stage

/-!
# The recursive audit's one new assumption: V*'s bridge (trusted)

In the recursive audit (`Proofs.Flock.Recursive.Defs`: `recGame`), the developer's firewall sends the auditor a salted
SHA-512 commitment to each inner round message, relays the auditor's live coin for the round, and afterwards proves, in
C-Flock `--zk --registered` sessions of the public verifier circuits V* (`Vs j`, one session each, in a row), that
openings of those commitments exist on which the inner verifier accepts. `VBridge` is what V*'s circuits must compute
for those sessions to mean it: V* accepts at its public inputs (the firewall's commitments registered in its public
files) and at the values it registers itself (the rounds' coins and the comb `coef`, in its record `ZkOuter.rd`) only
when the inner verifier accepts the inner transcript those commitments open to.

C-Flock's definitions are still in its proofs package (`Proofs.Flock.Soundness`), and so is the recursion's model
(`Proofs.Flock.Recursive`), so this module imports `Proofs.Flock.Recursive.Stage`, and `Specs.lean`, which the `security`
package builds without `security_proofs`, does not import it: `Proofs.Flock.Recursive` does.
-/

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

/-- **`v-bridge`: V*'s circuits compute the inner verifier** (V*'s session `j` is of the circuit `(Vs j).c`, pinned by
its file's digest through `ZkOuter.hc`, under the law `(Vs j).law`; `k, Rw` the count curve's). Four facts about V*'s
circuits, laws and inputs, each read at every history `(R, ω, t)` (the registration, the draw of the hidden circuit's
units, the rounds' commitments and coins):

* **they accept only an accepted inner transcript** (`Recursion.VBridge`): at all strategies of V*'s sessions there,
  when every session's outcome `flock-verify verify --zk --registered` accepts and every session's value layer is
  `Good` (no wrong unit, and at every unit each registered read carries the registrant's value: the hidden circuit's
  rows as registered, and the values V* registers itself, the coins and the comb `coef`), the messages `wmsg` reads
  from the layers hold an inner transcript the inner verifier accepts on the hidden circuit's statement `x R ω` at the
  rounds' coins;
* **their public files register the firewall's commitments** (`VCommits`): the digest session `j`'s value binding
  registers at each commit string of stage `i`'s `Sk i j` is the firewall's stage-`i` commitment's, `cmt j`;
* **they decode each round's message from those commit strings** (`VDecodes`, at equality): layers whose bits on the
  wires of each `Sk i j` are those of the values `v j` hold `msg i v` as their round-`i` message;
* **their laws draw every such wire** (`VFull`): each wire of `Sk i j` is a wire of a unit whose stratum session `j`'s
  law draws in full.

The first is the soundness direction of "V* accepts exactly when the Lean verifier accepts the inner session the
commitments commit to"; the recursive guarantees read no other. -/
def VBridge {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}
    (zs : Reg → L.Ω → List (Cm × I.Coin) → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  Recursion.VBridge zs dg N k Rw x wmsg ∧ VCommits zs Sk cmt ∧ VDecodes wmsg (fun _ _ a b => a = b) Sk msg ∧
    VFull I.r Sk

end Flock.Assumptions
