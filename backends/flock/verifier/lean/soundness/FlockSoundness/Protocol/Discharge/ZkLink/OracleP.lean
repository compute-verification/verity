import FlockSoundness.Protocol.Discharge.ZkLink.Push

/-!
# The `--zk` table with every committed object an oracle

The interactive-oracle `--zk` table (`tableAfterZK`) receives each rep's pads as two words and `τ` in the clear. The
primed table (`tabOrcP`) receives both as oracles (`Oracle F`), the pads on lanes `0` and `1` (`padWords`) and `τ` at
`(0, 0)`, so that every table node of the compiled table (level caps, pads roots, `τ`'s commitments) faces an oracle.
Each oracle the primed prover sends gives words or a `τ` the interactive-oracle prover can send, so the primed table is
no easier (`value_repZP_le`) and keeps `TabZK.after_sound`'s bound (`tabOrcP_sound`). Inside a session, the primed
table at its position accepts uncommitted no more often for an optimal prover (`orcSessZ_value_le`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-- The pads words an oracle carries: `C_h` on lane `0`, `C_μ` on lane `1`. -/
def padWords {N : ℕ} (O : Oracle F) : (Fin N → F) × (Fin N → F) :=
  (fun p => O 0 p.val, fun p => O 1 p.val)

/-- **The inner proof with `τ` an oracle**, read at `(0, 0)`. -/
def innerGameP {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s)
    (tr : Triple s) (Ch Cμ : Fin N → F) : Game Bool := do
  let ε ← draw F
  let ρσ ← recv (F × F)
  let γ ← draw (Fin (nA + 3) → F)
  let Oτ ← recv (Oracle F)
  let β ← draw F
  let y ← recv (Fin (s + nP) → F)
  let w ← draw (Fin q → F)
  pure (decide (innerCheck dom pos rows tr Ch Cμ ε ρσ γ (Oτ 0 0) β y w))

/-- **A `--zk` rep with its pads an oracle** (`repZK` with `padWords` and `innerGameP`). -/
def repZP {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ} (tr : Triple s)
    (phase : Game ((Fin nA → Lin F s) × Bool)) : Game Bool := do
  let O ← recv (Oracle F)
  let o ← phase
  let b ← innerGameP dom s nP q pos o.1 tr (padWords O).1 (padWords O).2
  pure (o.2 && b)

/-! ## Reordered rows -/

/-! ## The primed table -/

/-- **Rep `r` of the primed table** (`repZP` over the masked rep, its rows in flock-verify's order, as
`repMaskedZC`'s). -/
noncomputable def repZPAt (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (r : ℕ)
    (extra : List (Claim F T.S.m)) : Game Bool :=
  repZP T.dom T.np T.nP T.q T.pos T.tr
    ((repMaskedZK A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r C₀ extra).map fun o => (o.1 ∘ rowOrder, o.2))

/-- **The primed table after its link points**, with the dummy last message (the compiled table's openings). -/
noncomputable def tabOrcP (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) :
    Game (Bool × Oracle F) :=
  (interleave (repZPAt A T C₀ 0 (extraClaims T.S T.ptLocal mPts pts))
    (repZPAt A T C₀ 1 (extraClaims T.S T.ptLocal mPts pts))).bind fun acc =>
      Game.send (OpensZ F D Sl) fun _ => Game.ret (acc.1 && acc.2, C₀)

/-! ## The primed table in a session -/

open Classical in
/-- The primed table's output as a compiled output: accepted when both reps accept and the table is not committed. -/
noncomputable def orcOutZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (T : TabZK F mPts) (cap₀ : Cap D)
    (a : Bool × Oracle F) : OutZC F D Sl :=
  ⟨a.1 && decide (¬ T.Committed A a.2), cap₀, (PendZ.reject, PendZ.reject), default⟩

/-- **The oracle session**: the primed table in the compiled table's place. -/
noncomputable def orcSessZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (C₀ : Oracle F) (cap₀ : Cap D) :
    Game (List (OutZC F D Sl)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    batch (pre.map (· pts) ++ (tabOrcP (D := D) (Sl := Sl) A T C₀ pts).map (orcOutZ A T cap₀) :: post.map (· pts))

end FlockSoundness.Discharge.ZkLink
