import FlockBridge.Game
import FlockSoundness.Discharge.Exec.SessionLComplete
import Pouw.Protocol.Draft.Decisive

/-!
# C-Flock's decisive games as PoUW's

C-Flock states decisiveness over its own games (`FlockSoundness.Discharge.Exec.SessionLComplete`: `Productive`,
`Reaches`, `Decisive`), clause for clause the definitions PoUW states over core's games (`Pouw.Draft.Game`). `toCore`
reads a C-Flock game as the same tree, so it keeps each of them (`productive_toCore`, `reaches_toCore`,
`decisive_toCore`).

A tile check's game relabelled to its verdict is the one-stage audit's verdict, `(toCore (audit …)).map Prod.fst`
(PoUW's `HiddenSystem.game` relabels by the proof system's `accept`). It is decisive for every one-table plan
(`audit_tile_verdict_decisive`): the audit reaches `(true, S)` and `(false, S)` on one draw `S`, whose verdicts differ
(`decisive_map`). C-Flock's `audit_tile_decisive` alone does not give it, since its two outputs may differ only in the
draw. Decisiveness is existential (`Reaches` picks the messages and the coins), not C-Flock's completeness.
-/

namespace FlockBridge

namespace Game

open FlockSoundness.Discharge.Exec

variable {α β : Type}

/-- **`toCore` keeps productivity.** -/
theorem productive_toCore : ∀ g : FlockSoundness.Game α, SessionLComplete.Productive g →
    Pouw.Draft.Game.Productive (toCore g)
  | .ret _, _ => trivial
  | .send _ k, ⟨hM, h⟩ => ⟨hM, fun m => productive_toCore (k m) (h m)⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, h => fun c => productive_toCore (k c) (h c)

/-- **`toCore` keeps every reachable output.** -/
theorem reaches_toCore : ∀ (g : FlockSoundness.Game α) (a : α), SessionLComplete.Reaches g a →
    Pouw.Draft.Game.Reaches (toCore g) a
  | .ret _, _, h => h
  | .send _ k, a, ⟨m, h⟩ => ⟨m, reaches_toCore (k m) a h⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, a, ⟨c, h⟩ => ⟨c, reaches_toCore (k c) a h⟩

/-- **C-Flock's decisive games are decisive in core.** -/
theorem decisive_toCore {g : FlockSoundness.Game α} (h : SessionLComplete.Decisive g) :
    Pouw.Draft.Game.Decisive (toCore g) := by
  obtain ⟨hP, a, b, hab, ha, hb⟩ := h
  exact ⟨productive_toCore g hP, a, b, hab, reaches_toCore g a ha, reaches_toCore g b hb⟩

/-- **Relabelled, a productive game is decisive** when it reaches two outputs whose labels differ. -/
theorem decisive_map (f : α → β) {g : FlockSoundness.Game α} {a b : α} (hP : SessionLComplete.Productive g)
    (ha : SessionLComplete.Reaches g a) (hb : SessionLComplete.Reaches g b) (hab : f a ≠ f b) :
    Pouw.Draft.Game.Decisive ((toCore g).map f) := by
  rw [← toCore_map]
  exact decisive_toCore ⟨SessionLComplete.productive_map f hP, f a, f b, hab, SessionLComplete.reaches_map f ha,
    SessionLComplete.reaches_map f hb⟩

/-- **A tile's audit verdict is decisive in core**: the verdict of the one-stage audit over the salted batched session
of a one-table plan, read as a core game, when registrations exist. Some run accepts and some run rejects, on every
registration and draw; this is the tile check's part of PoUW's `hDec`. -/
theorem audit_tile_verdict_decisive {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    [Nonempty Reg] (L : FlockSoundness.Audit.Law n) (H : List UInt8 → D)
    (E : FlockSoundness.Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : FlockSoundness.Merkle.Leaf (List FlockLevel3.GF128) Sl D H)
    (T : Finset (Fin n) → Reg → FlockSoundness.TabSpec mPts) :
    Pouw.Draft.Game.Decisive
      ((toCore (FlockSoundness.Audit.audit L Reg (batchedSessionL H E Lf fun S R => [T S R]))).map Prod.fst) := by
  obtain ⟨R⟩ := ‹Nonempty Reg›
  obtain ⟨ω⟩ := L.nonempty
  exact decisive_map Prod.fst (SessionLComplete.audit_productive L H E Lf _)
    (SessionLComplete.audit_reaches_accept L H E Lf _ R ω)
    (SessionLComplete.audit_reaches_reject L H E Lf _ R ω (List.cons_ne_nil _ _)) Bool.noConfusion

end Game

end FlockBridge
