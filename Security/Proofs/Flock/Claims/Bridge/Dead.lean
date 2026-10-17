import Proofs.Flock.Soundness.Discharge.Tables.Sim

/-!
# A simulation's dead branches, answered with a fixed message

A simulation (`Refine.Sim`) plays, past a live branch on which no live run is accepted, any model strategy (`Sim.dead`).
`Sim.mapDead f` replaces each such strategy `σ` by `f σ`, and the transfer holds of the result (`Sim.prob_le_mapDead`),
since a dead branch's live probability is `0` whatever the model plays there. `answerWith nul` is the transformer that
answers every message of `nul`'s type with `nul` and plays `σ` elsewhere: at the salted session, empty openings, so a
dead branch opens nothing.
-/

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Refine

open Classical in
/-- **Answer every message of `Op` with `nul`**: at a node sending `Op`, `nul` when the game goes on from it, and `σ`'s
message elsewhere. -/
noncomputable def answerWith {α Op : Type} (nul : Op) : (g : Game α) → Strategy g → Strategy g
  | .ret _, σ => σ
  | .send M k, σ =>
    if h : ∃ e : Op = M, Nonempty (Strategy (k (cast e nul))) then
      ⟨cast h.fst nul, answerWith nul (k (cast h.fst nul)) (Classical.choice h.snd)⟩
    else ⟨σ.1, answerWith nul (k σ.1) σ.2⟩
  | @Game.coin _ _ _ _ k, σ => fun c => answerWith nul (k c) (σ c)

/-- **The dead branches' strategies replaced** by `f`'s. -/
def Sim.mapDead {β α : Type} {pL : β → Prop} {pg : α → Prop} (f : (g : Game α) → Strategy g → Strategy g) :
    {L : Game β} → {g : Game α} → Sim pL pg L g → Sim pL pg L g
  | _, _, .ret h => .ret h
  | _, _, .model m d => .model m (Sim.mapDead f d)
  | _, _, .live k => .live fun m => Sim.mapDead f (k m)
  | _, _, .coin e k => .coin e fun c => Sim.mapDead f (k c)
  | _, _, .dead h σ => .dead h (f _ σ)

/-- **The transfer with the dead branches replaced**: the live verdict is no likelier than the model's against the
simulated strategy. -/
theorem Sim.prob_le_mapDead {β α : Type} {pL : β → Prop} {pg : α → Prop} (f : (g : Game α) → Strategy g → Strategy g)
    {L : Game β} {g : Game α} (d : Sim pL pg L g) (s : Strategy L) :
    prob pL L s ≤ prob pg g ((Sim.mapDead f d).strategy s) :=
  Sim.prob_le _ s

end Claims.Bridge
