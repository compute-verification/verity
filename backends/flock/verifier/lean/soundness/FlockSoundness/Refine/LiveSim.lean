import FlockSoundness.Refine.Live
import FlockSoundness.Refine.Plays
import FlockSoundness.Model.Compiled
import FlockSoundness.Protocol.Refine.LiveSim

/-!
# The table's simulation (the transfer, R11c)

A live prover for one table's session (`liveTable`) is simulated in the model's table (`modelTable`, which `tableC` is,
`tableC_eq_modelTable`). The simulated prover sends the cap `root_B` decodes to, shares the link points, and follows the
rounds: for a round on rep `r`'s stream, the model's rep `r` takes the messages decoded from the round's bytes, then
draws its coins with the live round's. At the end it takes the reps' final messages and the openings, decoded from the
proof files. Wherever the model's game doesn't fit what was decoded, no accepted live run goes on, and the model plays
any strategy.

* `PlaysT g ts a`: `g`, played with the typed steps `ts` (messages and coins, in order), ends with `a`.
* `Decodes`: on every accepted live outcome, each rep's model game runs on the tape decoded from its stream and its
  proof file (`repTape`), and the model accepts those runs. This is what the refinement and framing (R11b) provide.
* `live_le`: against any live prover, the live verdict's probability is at most the model's against the simulated
  prover.
-/

namespace FlockSoundness.Game

variable {α β A B : Type}

/-! ## Runs on one typed tape -/

/-! ## Every message type inhabited -/

/-! ## The model's moves in an interleave -/

namespace Sim

variable {pL : β → Prop} {pg : α → Prop}

variable {L : Game β} {K : A × B → Game α}

end Sim

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game

local notation "𝔽" => FlockLevel3.GF128

section Table

variable {P OutT Cp Op : Type} {mPts : ℕ}

variable (dec : ℕ → ByteArray → List Msg) (one : ℕ → Bool) (fin : Fin 2 → Array ByteArray → List Msg)

end Table

section Sim

variable {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Cp → P × P → Op → OutT) (pL : LiveOut mPts → Prop)
  (pg : OutT → Prop) (opn : Array ByteArray → Op)

variable {out pL pg opn}

variable (out pL pg opn)

variable (dg : ByteArray → ByteArray) (dec : ℕ → ByteArray → List Msg) (one : ℕ → Bool)
  (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Cp)

/-- **The transfer, for one table.** Against any live prover, the executable's verdict on the live session holds with
probability at most the model table's verdict against the simulated prover, whenever the decoder is right on accepted
sessions. -/
theorem live_le (N : ℕ) (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (hinh : ∀ pts r, AllInh (G pts r))
    (hdec : Decodes out pL pg opn dec one fin capOf G) (Pr : Strategy (liveTable dg mPts N)) :
    prob pL (liveTable dg mPts N) Pr ≤
      prob pg (modelTable G out) ((simTable out pL pg opn dg dec one fin capOf N G hinh hdec).strategy Pr) :=
  Sim.prob_le _ Pr

end Sim

open Classical in
/-- `tableC` is the model table with the compiled reps and its verdict. -/
theorem tableC_eq_modelTable {K D : Type} [Field K] [Fintype K] [DecidableEq K] (A : Model.Arith 𝔽 K)
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (S : Model.Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) :
    Model.tableC A H E S sch hm ptLocal mPts =
      modelTable (fun pts (r : Fin 2) => Model.repC A r.val S sch hm (Model.extraClaims S ptLocal mPts pts))
        (fun cap pend o => (⟨pend.1.verdict o && pend.2.verdict o &&
          decide (Model.OpensOK H E cap 0 pend.1 o ∧ Model.OpensOK H E cap 1 pend.2 o), cap, pend, o⟩ : Model.OutC 𝔽 D)) :=
  rfl

end FlockSoundness.Refine
