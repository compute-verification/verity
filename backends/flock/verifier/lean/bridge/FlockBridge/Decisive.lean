import FlockBridge.Game
import FlockSoundness.Discharge.Exec.SessionLDecisive
import Verity.Protocol.Game.Decisive

/-!
# C-Flock's decisive games as core's

C-Flock states decisiveness over its own games (`FlockSoundness.Discharge.Exec.SessionLDecisive`: `Productive`,
`Reaches`, `Decisive`), clause for clause core's definitions over core's games (`Verity.Game`). `toCore`
reads a C-Flock game as the same tree, so it keeps each of them (`productive_toCore`, `reaches_toCore`,
`decisive_toCore`).

A tile check's game relabelled to its verdict is the one-stage audit's verdict, `(toCore (audit …)).map Prod.fst`
(PoUW's `HiddenSystem.game` relabels by the proof system's `accept`). It is decisive for every one-table plan
(`audit_tile_verdict_decisive`): C-Flock's `audit_tile_decisive_verdict` read through `toCore`, which commutes with
`map` (`toCore_map`). Decisiveness is existential (`Reaches` picks the messages and the coins), not C-Flock's
completeness.
-/

namespace FlockBridge

namespace Game

open FlockSoundness.Discharge.Exec

variable {α : Type}

/-- **`toCore` keeps productivity.** -/
theorem productive_toCore : ∀ g : FlockSoundness.Game α, SessionLDecisive.Productive g →
    Verity.Game.Productive (toCore g)
  | .ret _, _ => trivial
  | .send _ k, ⟨hM, h⟩ => ⟨hM, fun m => productive_toCore (k m) (h m)⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, h => fun c => productive_toCore (k c) (h c)

/-- **`toCore` keeps every reachable output.** -/
theorem reaches_toCore : ∀ (g : FlockSoundness.Game α) (a : α), SessionLDecisive.Reaches g a →
    Verity.Game.Reaches (toCore g) a
  | .ret _, _, h => h
  | .send _ k, a, ⟨m, h⟩ => ⟨m, reaches_toCore (k m) a h⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, a, ⟨c, h⟩ => ⟨c, reaches_toCore (k c) a h⟩

/-- **C-Flock's decisive games are decisive in core.** -/
theorem decisive_toCore {g : FlockSoundness.Game α} (h : SessionLDecisive.Decisive g) :
    Verity.Game.Decisive (toCore g) := by
  obtain ⟨hP, a, b, hab, ha, hb⟩ := h
  exact ⟨productive_toCore g hP, a, b, hab, reaches_toCore g a ha, reaches_toCore g b hb⟩

/-- **A tile's audit verdict is decisive in core**: the verdict of the one-stage audit over the salted batched session
of a one-table plan, read as a core game, when registrations exist. Some run accepts and some run rejects, on every
registration and draw; this is the tile check's part of PoUW's `hDec`. -/
theorem audit_tile_verdict_decisive {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    [Nonempty Reg] (L : FlockSoundness.Audit.Law n) (H : List UInt8 → D)
    (E : FlockSoundness.Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : FlockSoundness.Merkle.Leaf (List FlockLevel3.GF128) Sl D H)
    (T : Finset (Fin n) → Reg → FlockSoundness.TabSpec mPts) :
    Verity.Game.Decisive
      ((toCore (FlockSoundness.Audit.audit L Reg (batchedSessionL H E Lf fun S R => [T S R]))).map Prod.fst) := by
  rw [← toCore_map]
  exact decisive_toCore (SessionLDecisive.audit_tile_decisive_verdict L H E Lf T)

end Game

end FlockBridge
