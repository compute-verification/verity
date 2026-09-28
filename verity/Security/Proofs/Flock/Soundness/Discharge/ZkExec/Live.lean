import Proofs.Flock.Soundness.Discharge.ZkExec.Event
import Proofs.Flock.Soundness.Discharge.Exec.Live

/-!
# The live `--zk` audit, and the compiled `--zk` table it is simulated in

The live audit is `Exec.auditLive` with the challenger's one table (`liveOf dg mPts N wr`: `liveTable`, every
round's coins fresh and drawn after its bytes, written out by `wr`), as without `--zk`. Under `--zk` the coins come from
the keyed coin tree committed before the prover's first message (C-Flock ZK), so this is the session with each round's
coins fresh: the coin tree's step from that to C-Flock ZK's is [wave2-zk-coins]'s, an open guarantee here.

The target is a compiled `--zk` table on one table, named by its pieces in `modelTable`'s shape (the challenger's
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

local notation "𝔽" => FlockLevel3.GF128

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

/-- **The decoder is right on the sessions flock-verify `--zk` accepts** (R11b at `--zk`, named): at every draw and
registration, on every live outcome whose session directory `AcceptsZK` holds of, each rep's game runs on the tape
decoded from its stream and its proof file, and the compiled table's verdict on those runs, with the cap `root_B`
decodes to and the openings the proof files decode to, holds. -/
def DecodesZK : Prop :=
  ∀ S R, Decodes (out S R) (fun o => AcceptsZK I (wr S R o)) (fun x => acc S R x = true) (opn S R) (dec S R)
    (one S R) (fin S R) (capOf S R) (G S R)

variable {L : Law n}

/-- **The simulated audit strategy**: the live prover's registration, and at each draw the simulated table strategy
(`simTable`) against its live session. -/
noncomputable def simAuditZK (hinh : ∀ S R pts r, AllInh (G S R pts r))
    (hdec : DecodesZK G out acc I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) : Strategy (audit L Reg (sessionZC G out acc)) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    ((simTable (out (L.draw ω) τ.1) (fun o => AcceptsZK I (wr (L.draw ω) τ.1 o))
      (fun x => acc (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1)
      (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (G (L.draw ω) τ.1) (hinh (L.draw ω) τ.1)
      (hdec (L.draw ω) τ.1)).strategy
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

end FlockSoundness.Discharge.ZkExec
