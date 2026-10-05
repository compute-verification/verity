import FlockSoundness.Discharge.ZkSession.Session
import FlockSoundness.CompiledSound
import FlockSoundness.ExtractAcc
import FlockSoundness.Protocol.Discharge.ZkLink.Compiled

/-!
# The compiled `--zk` table and session (live/PROTOCOL.md §1, §3, §5, §7; PROTOCOL.md §5, §14)

`sessionZK` (`Discharge/ZkSession/Session.lean`) is the protocol as Lean defines it at the interactive-oracle layer:
`Commit` sends each table's level-0 oracle, each rep receives its pads as two words, level 1's oracle as a table, and
the inner proof's `τ` in the clear. Here is the protocol as run, with every oracle a commitment:

* `Commit` sends each table's **witness cap**, the 68-lane level-0 tree (`2^k₀` witness lanes and the reps' `2e` extra
  lanes, one tree);
* each rep (`repZC`) starts with the **pads root** (`recv D`, §3), runs the masked algebraic phase as in `repMaskedZK`,
  and level 0 masked with Ligerito's levels compiled (`ligeritoPadZC`: level 1 onward commit by caps, and level 0's
  consistency value is computed from the opened 68-lane rows, `padColumnO`), then `final_c'`;
* the **inner proof** (`innerGameC`, §5) sends `τ`'s commitment (`recv D`) where the interactive-oracle layer sends `τ`;
* after the last coin of both reps, one message (`OpensZ`) carries every opening: the queried rows of every level with
  their salts and paths, and per rep the queried pads pairs `(C_h[j], C_mu[j])` with their salts and paths, and `τ`
  with its salt.

The table accepts (`tabAfterZC`) when both reps' Ligerito verdicts accept on the opened rows, both inner checks hold on
the opened pads and `τ` (`innerCheckC`: `τ` opens its commitment, the batched constraint with the opened `τ`, the code
identity at every query, and every pads path to the pads root), and every row path verifies from its row's leaf under
its salt, level 0's rows being 68 lanes wide (`OpensOKZ`).

`sessionZG` is `Exec.sessionG` over `--zk` tables: `sessionZK` is its instance at `TabZK.after` (`sessionZK_eq`), and the
compiled session `sessionZC` its instance at `tabAfterZC`. `batchedSessionZC` is the compiled session on a draw, as
`batchedSessionZK` is the interactive-oracle one.

**Extraction** (as `SessionBatchAcc.lean` at M0): table `j`'s extractor fills its level-0 oracle from the verifying
level-0 openings of `k` reruns of everything after `Commit` in which the session accepts (`extractTableZ`), and
`extractProbAccZ` is the probability of an event of that oracle over those reruns.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Level 0, compiled -/

/-! ## The inner proof, compiled -/

/-! ## The table -/

/-! ## The session -/

omit [Fintype K] in
/-- The interactive-oracle session is `sessionZG` at `TabZK.after`. -/
theorem sessionZK_eq (A : Arith F K) (ts : List (TabZK F mPts)) :
    sessionZK A ts = sessionZG (fun T C pts => T.after A C pts) ts := by
  unfold sessionZK sessionZG
  simp only [List.map_ofFn]
  rfl

omit [Fintype K] in
/-- The compiled session is `sessionZG` at `tabAfterZC`, as `sessionZK` is at `TabZK.after`. -/
theorem sessionZC_eq (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts)) :
    sessionZC A Hs ts = sessionZG (tabAfterZC A Hs) ts :=
  rfl

section Batched

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) {n : ℕ} {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))

end Batched

/-! ## Extraction from accepted reruns -/

section Acc

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))

/-- The probability that the compiled session accepts, after `Commit`. -/
noncomputable def accProbZC (τ : Strategy (sessionZC A Hs ts)) : ℝ :=
  expect (ind (acceptedZC ts)) (sessAfterZC (gamesZC A Hs ts τ.1)) τ.2

open Classical in
/-- **An event of table `j`'s extraction, from accepted reruns**: the probability, over `k` reruns of everything after
`Commit` in which the compiled session accepts, that the level-0 oracle the extractor fills from table `j`'s openings
satisfies `G`. -/
noncomputable def extractProbAccZ (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts))
    (G : Oracle F → Prop) : ℝ :=
  Extract.expectAcc (sessAfterZC (gamesZC A Hs ts τ.1)) τ.2 (acceptedZC ts) k
    (ind fun bs => G (extractTableZ Hs (ts.get j) j (τ.1 j) bs))

/-- **Table `j`'s extraction from accepted reruns fails**: no padded witness near the extracted oracle satisfies the
table's statement. -/
noncomputable def failProbAccZ (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts)) : ℝ :=
  extractProbAccZ A Hs ts j k τ fun C => ¬ (ts.get j).Committed A C

end Acc

end FlockSoundness.Discharge.ZkLink
