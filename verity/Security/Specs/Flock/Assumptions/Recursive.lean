import Proofs.Flock.Recursive.Stage

/-!
# The recursive audit's one new assumption: V*'s bridge (trusted)

In the recursive audit (`Proofs.Flock.Recursive.Defs`: `recGame`), the developer's gateway sends the auditor a salted
SHA-512 commitment to each inner round message, relays the auditor's live coin for the round, and afterwards proves, in
one C-Flock `--zk --registered` session of the public verifier circuit V*, that openings of those commitments exist on
which the inner verifier accepts. `VBridge` is what V*'s circuit must compute for that session to mean it: V* accepts at
its public inputs (the gateway's commitments registered in its public file, the rounds' coins, the comb) only when the
inner verifier accepts the inner transcript those commitments open to.

C-Flock's definitions are still in its proofs package (`Proofs.Flock.Soundness`), and so is the recursion's model
(`Proofs.Flock.Recursive`), so this module imports `Proofs.Flock.Recursive.Stage`, and `Specs.lean`, which the `security`
package builds without `security_proofs`, does not import it: `Proofs.Flock.Recursive` does.
-/

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

/-- **`v-bridge`: V*'s circuit computes the inner verifier** (V* is the circuit `c`, pinned by its file's digest through
`ZkOuter.hc`; `σs`, `ks` its law's strata and their draw counts; `k, Rw` the count curve's). Four facts about V* and its
public inputs, each read at every history `(R, ω, t)` (the registration, the draw of the hidden circuit's units, the
rounds' commitments and coins):

* **it accepts only an accepted inner transcript** (`Recursion.VBridge`): at every strategy of V*'s session there, every
  outcome `flock-verify verify --zk --registered` accepts at which V*'s value layer has no wrong unit and agrees with the
  registrant at the outcome's drawn units holds, in the messages `wmsg` reads from the layer, an inner transcript the
  inner verifier accepts on the hidden circuit's statement `x R ω` at the rounds' coins;
* **its public file registers the gateway's commitments** (`VCommits`): the digest V*'s value binding registers at each
  commit string of stage `i`'s `Sk i` is the gateway's stage-`i` commitment's, `cmt`;
* **it decodes each round's message from those commit strings** (`VDecodes`, at equality): a layer whose bits on the
  wires of `Sk i` are those of the values `v` holds `msg i v` as its round-`i` message;
* **its law draws every such wire**: each wire of `Sk i` is a wire of a unit whose stratum V*'s law draws in full.

The first is the soundness direction of "V* accepts exactly when the Lean verifier accepts the inner session the
commitments commit to"; the recursive guarantees read no other. -/
def VBridge {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {mPts nv : ℕ} {Reg' : Type} {c : Flock.Circuit}
    {hU : Built c.unit} {hscope : Layout.scopeOk c = true} {Lv : Law nv} {m : ℕ} (σs : Fin nv → Fin m) (ks : Fin m → ℕ)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → ZkOuter mPts nv Reg' c hU hscope Lv) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : (Fin (Layout.keyN c hU nv) → Bool) → List I.Msg)
    (Sk : ℕ → Finset (PosK c hU nv)) (cmt : Cm → PosK c hU nv → (obK c hU nv).Dg)
    (msg : ℕ → (PosK c hU nv → ValK c) → I.Msg) : Prop :=
  Recursion.VBridge zs dg N k Rw x wmsg ∧ VCommits zs Sk cmt ∧ VDecodes wmsg (fun _ _ a b => a = b) Sk msg ∧
    ∀ i < I.r, ∀ g, posK c hU nv g ∈ Sk i →
      ∃ u, g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u ∧ (Law.stratum σs (σs u)).card ≤ ks (σs u)

end Flock.Assumptions
