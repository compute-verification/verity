import FlockSoundness.Protocol.Discharge.ZkExec.Event
import FlockSoundness.Protocol.Discharge.Exec.Live

/-!
# The live `--zk` audit, and the compiled `--zk` table it is simulated in

The live audit is `Exec.auditLive` with the live coin server's one table (`liveOf dg mPts N wr`: `liveTable`, every
round's coins fresh and drawn after its bytes, written out by `wr`), as at M0. Under `--zk` the coins come from the
keyed coin tree committed before the prover's first message (M2), so this is the session with each round's coins
fresh: the coin tree's step from that to M2's is [wave2-zk-coins]'s, an open guarantee here.

The target is a compiled `--zk` table on one table, named by its pieces in `modelTable`'s shape (the live coin server's
simulation, `Refine/LiveSim.lean`, reads exactly these), at each draw `S` and registration `R`:

* `Cp S R`, the level-0 commitment the prover sends first (`root_B`);
* `G S R pts r : Game (P S R)`, rep `r`'s game at link points `pts`, from its pads' commitment to its inner proof's last
  coin, every commitment a digest message and every coin a draw;
* `Op S R`, the openings sent after the last coin of both reps (level 0's rows, the pads' columns, `τ`'s opening);
* `out S R` the table's output, and `acc S R` its verdict.

`sessionZC G out acc S R` is that table's game, `(modelTable (G S R) (out S R)).map (acc S R)`. The compiled `--zk`
session with salted commitments, its instance of these pieces and its bound against `ZkSession.sessionZK` are
[wave2-zk-link]'s.

`DecodesZK` is R11b at `--zk` (`Exec.DecodesAll`'s form): on every live outcome flock-verify `--zk` accepts, each rep's
game runs on the tape decoded from its stream and its proof file, and the table's verdict on those runs, with the cap
`root_B` decodes to and the openings the proof files decode to, holds. `simAuditZK` is the audit strategy the simulation
builds from a live prover.
-/

namespace FlockSoundness.Discharge.ZkExec

open Game Audit Exec Refine Flock

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {P Cp Op Out : Finset (Fin n) → Reg → Type}
  (G : ∀ S R, (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (P S R))
  (out : ∀ S R, Cp S R → P S R × P S R → Op S R → Out S R) (acc : ∀ S R, Out S R → Bool)

/-- **The compiled `--zk` table's game at a draw**: the level-0 cap, the link points, the two reps concurrently, the
openings, and the table's verdict. -/
def sessionZC (S : Finset (Fin n)) (R : Reg) : Game Bool := (modelTable (G S R) (out S R)).map (acc S R)

variable [∀ S R, Nonempty (Op S R)] (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ S R, Array ByteArray → Op S R) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ S R, ByteArray → Cp S R) (N : ℕ)

variable {L : Law n}

end FlockSoundness.Discharge.ZkExec
