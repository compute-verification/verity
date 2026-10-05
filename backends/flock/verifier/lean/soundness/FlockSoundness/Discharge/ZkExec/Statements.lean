import FlockSoundness.Discharge.ZkExec.Live
import FlockSoundness.Discharge.Exec.Setup
import FlockSoundness.Discharge.Exec.Retained
import FlockSoundness.Protocol.Discharge.ZkExec.Statements

/-!
# The target statements of flock-verify `--zk`'s acceptance and its refinement

Each statement is a `Prop` here; its theorem (`theorem … : G`) is in the file that proves it.

**Acceptance** (what `AcceptsZK` gives, peeled check by check, at any tag and any table count `J`):
* `ZkVerifyOk` (`Verify.lean`): `Zk.verify` accepts only on the keyed coin tree, a record that decodes, one statement
  per table, one proof file per rep stream, and at every table both reps' checks accepting with one cap.
* `ZkRepOk` (`Verify.lean`): a rep's check accepts only on its stream and proof file, the pinned PCS parameters, a
  64-byte pads root, the transcript program `repProg` running on the stream's rounds and consuming every round, and
  the proof's cap equal to the table's `root_B`.
* `ZkInnerOk` (`Inner.lean`): the inner proof accepts only with its pinned sizes, `β ≠ 0`, `τ` opening its
  `hm96-sha512/v1` commitment, the batched constraint `⟨c, Y⟩ = t + β·τ`, and at each of the `Q` queried columns `Y`'s
  code equal to `a + β·b` and the column's `hm96` leaf climbing to the pads' root.

**Acceptance and the statement** (under tags without hidden outputs, `hh`, as `Exec`'s; with them `setupTables` calls
`setupHidden`, [wave2-hidden-exec]'s):
* `AcceptsZKSetupH`: every table is `setupH`'s statement under the `--zk` tags (`tagsAt (zkTags I) J j`) and the coin
  tree, at some draw (the record's, for one table), and passes `Zk.shapeOk`.
* `AcceptsZKOne`: a one-table acceptance is one `setupH` statement's, with `Zk.verify` on `#[Setup.ofCircuit st]`.
* `AcceptsZKRetained`: under tags that retain rounds, every stream of the accepted record retained every round's bytes.

**Refinement** (one table, against the compiled `--zk` table `G, out, acc` named in `Live.lean`):
* `ExecRefinesZKOne`: R11 at `--zk`: against any live prover, flock-verify `--zk`'s acceptance is no likelier than the
  compiled table's acceptance against the simulated prover `simAuditZK … τ`, with `bad = ⊥`, given `DecodesZK`.
* `ZkHeadlineExecOne`: the `--zk` headline at flock-verify's acceptance, carried from any bound `B` on the compiled
  audit's `Pr[accept ∧ Q]` at the simulated prover, for any event `Q` of that prover (at least `K` wrong units of its
  committed values): `Pr[flock-verify --zk accepts ∧ Q] ≤ B`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Lean (Json)
open Game Audit Exec Refine Flock
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## Acceptance -/

/-! ## Acceptance and the statement -/

/-- **Each table of a `--zk` acceptance is `setupH`'s statement** under the `--zk` tags and the coin tree, at some
draw (the record's, for one table), and passes the `--zk` shape check. -/
def AcceptsZKSetupH : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = false → AcceptsZK I t →
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧ ∃ d,
        Stmt.setupH (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

/-- **A `--zk` acceptance under retaining tags has every round's bytes**: the record decodes against the session's spec,
and every stream of it retained every round's bytes (`AllRetained`). -/
def AcceptsZKRetained : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = false → I.tags.retainRounds = true → AcceptsZK I t →
    ∃ spec sts sess, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds

/-! ## Refinement, one table -/

section Refine

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {P Cp Op Out : Finset (Fin n) → Reg → Type}
  (G : ∀ S R, (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (P S R))
  (out : ∀ S R, Cp S R → P S R × P S R → Op S R → Out S R) (acc : ∀ S R, Out S R → Bool)
  [∀ S R, Nonempty (Op S R)] (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ S R, Array ByteArray → Op S R) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ S R, ByteArray → Cp S R) (N : ℕ) (L : Law n)

/-- **R11 at `--zk`, one table**: against any live prover `τ` of the live audit (`liveTable` written out by `wr`),
flock-verify `--zk`'s acceptance is no likelier than the compiled `--zk` table's acceptance against the simulated
prover `simAuditZK … τ`, with no live event added, whenever every rep's game has a strategy (`hinh`) and the decoder is
right on accepted sessions (`DecodesZK`). -/
def ExecRefinesZKOne : Prop :=
  ∀ (hinh : ∀ S R pts r, AllInh (G S R pts r)) (hdec : DecodesZK G out acc I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))),
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAcceptsZK I) (fun _ => False)
      (audit L Reg (sessionZC G out acc)) (simAuditZK G out acc I wr opn dg dec one fin capOf N hinh hdec τ)

/-- **The `--zk` headline at flock-verify's acceptance, one table**, from a bound on the compiled audit: for any event
`Q` of the simulated prover (at least `K` wrong units of its committed values, say), if the compiled `--zk` audit
accepts with `Q` with probability at most `B` against `simAuditZK … τ`, then flock-verify `--zk` accepts the live
audit's transcript with `Q` with probability at most `B`. -/
def ZkHeadlineExecOne : Prop :=
  ∀ (hinh : ∀ S R pts r, AllInh (G S R pts r)) (hdec : DecodesZK G out acc I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (Q : Prop) (B : ℝ≥0∞),
    prob (fun o => o.1 = true ∧ Q) (audit L Reg (sessionZC G out acc))
        (simAuditZK G out acc I wr opn dg dec one fin capOf N hinh hdec τ) ≤ B →
      prob (fun o => LiveAcceptsZK I o ∧ Q) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤ B

end Refine

end FlockSoundness.Discharge.ZkExec
