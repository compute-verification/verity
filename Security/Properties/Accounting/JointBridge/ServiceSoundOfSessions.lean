import Definitions.Accounting.Assumptions.TS
import Properties.Pouw.Period

/-!
# PoUW's proof system soundness from its proof layer

`Properties.Pouw.Period.ServiceSound` bounds, in each window, the event that PoUW's deployed check
(`PouwCheck.protocol.check`) accepts the window and some unit it publishes as drawn (`Public.drawn`) has an opening that
fails its relation. `Accounting.TS.ProofLayer.SoundAt` bounds each proof of a window's proof layer the same way, and
`Accounting.Guarantees.TS.SessionSoundAt` concludes it of a layer whose proofs are C-Flock sessions. This module states
the step between the two:

* `Sessions.layer`: a window verdict `acc` with its drawn units (`drawn`), each unit `u` read at its key `key u` of the
  window's openings, as a proof layer of `n_π` proofs a window. Proof `i` is one session: `sessOf x w u` is the session
  that proves drawn unit `u`, and the session's statement is its drawn units' relations. Every proof of a window is
  accepted where the verdict accepts the window;
* `Sessions.checkLayer`: that layer at PoUW's check, its verdict the check's acceptance and its drawn units the check's
  `Public.drawn`;
* `CheckProofSystemSound`: the check layer's soundness at `δ` (the named premise
  `Accounting.Assumptions.TS.ProofSoundAt`) gives `ServiceSound` at `n_π · δ`, a union bound over the window's
  sessions. With one session a window (`n_π = 1`, the window's staged statement in one session) the bound is `δ`;
* `CheckProofSystemSoundHonest`, `CheckProofSystemSoundNontrivial`: some period of the deployed check meets the
  premise with the check accepting on every path, and some period breaks the conclusion at `0`.
-/

namespace Properties.Accounting.JointBridge

open Pouw.RelativeSecurity (Paths)

namespace Sessions

/-- **A window verdict with its drawn units, as a proof layer of `n_π` sessions a window.** On path `x`, every proof of
window `w` is accepted where the verdict accepts the window (`acc x w`), and opens the window's openings
(`opened x w`, unit by key); proof `i`'s relation is that every drawn unit `u` of the window (`drawn x w u`) whose
session is `i` (`sessOf x w u = i`) meets its relation at its key's opening. -/
def layer {Ω U V : Type} {n nπ : ℕ} (acc : Paths Ω n → Fin n → Prop)
    (drawn : Paths Ω n → Fin n → U → Prop) (key : U → ℕ) (opened : Paths Ω n → Fin n → ℕ → ℕ → V)
    (rel : Paths Ω n → Fin n → U → (ℕ → V) → Prop) (sessOf : Paths Ω n → Fin n → U → Fin nπ) :
    Accounting.TS.ProofLayer Ω n nπ (ℕ → ℕ → V) where
  accepted x w _ := acc x w
  opened x w _ := opened x w
  rel x w i v := ∀ u, drawn x w u → sessOf x w u = i → rel x w u (v (key u))
  right _ _ := True

open PouwCheck Properties.Pouw.Period in
/-- **PoUW's check as a drawn layer**: window `w` is accepted where PoUW's deployed check accepts it, and its drawn
units are the units the check publishes as drawn (`Public.drawn`), each read at its index. -/
def checkLayer {C Ξ V : Type} {m Np n nπ : ℕ} (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (sessOf : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → Fin nπ) :
    Accounting.TS.ProofLayer (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n nπ (ℕ → ℕ → V) :=
  layer (fun x w => Accepts p (ws.reg x w) (ws.ev x w))
    (fun x w u => ∃ (pub : Public) (prof : Core.Integrity),
      protocol.check p (ws.reg x w) (ws.ev x w) = .accept pub prof ∧ u.val ∈ pub.drawn.toList)
    Fin.val opened rel sessOf

end Sessions

/-- **PoUW's proof system is sound against its check, from its sessions' soundness.** For every period of `n` windows
of PoUW's deployed check (`PouwCheck.protocol.check` at parameters `p`, each window's registration and answer any
function of the period's coins, `ws`), openings `opened` and relations `rel` of the drawn units, and every assignment
`sessOf` of each window's drawn units to `n_π` sessions: if each session of the check's layer is accepted with a
statement that fails at its openings with probability at most `δ` (`Accounting.Assumptions.TS.ProofSoundAt`, which
`Accounting.Guarantees.TS.SessionSoundAt` concludes of C-Flock sessions), then in every window the check accepts and
some unit it publishes as drawn fails its relation with probability at most `n_π · δ` (`ServiceSound`). -/
def CheckProofSystemSound : Prop :=
  ∀ {C Ξ V : Type} [Fintype C] [Fintype Ξ] {m Np n nπ : ℕ} (p : PouwCheck.Params)
    (ws : Properties.Pouw.Period.Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (sessOf : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → Fin nπ) (δ : ℝ),
    Accounting.Assumptions.TS.ProofSoundAt (Sessions.checkLayer p ws opened rel sessOf) δ →
      Properties.Pouw.Period.ServiceSound p ws opened rel (nπ * δ)

/-- **`CheckProofSystemSound`'s premise holds where the deployed check accepts**: some period of one window, at a
population of 144 units and one session a window, has PoUW's deployed check accepting the window on every path and its
layer, under relations every opening meets, sound at `0`. -/
def CheckProofSystemSoundHonest : Prop :=
  ∃ (m : ℕ) (p : PouwCheck.Params)
    (ws : Properties.Pouw.Period.Executions (Paths Bool m × (Equiv.Perm (Fin 144) × Unit)) 1),
    (∀ x w, Properties.Pouw.Period.Accepts p (ws.reg x w) (ws.ev x w)) ∧
      Accounting.Assumptions.TS.ProofSoundAt
        (Sessions.checkLayer p ws (fun _ _ _ _ => ()) (fun _ _ _ _ => True) (fun _ _ _ => (0 : Fin 1))) 0

/-- **`CheckProofSystemSound`'s conclusion can fail at `0`**: some period of one window, at a population of 144 units,
has PoUW's deployed check accepting and publishing a drawn unit on every path, so with relations every opening fails
`ServiceSound` fails there at `0`. -/
def CheckProofSystemSoundNontrivial : Prop :=
  ∃ (m : ℕ) (p : PouwCheck.Params)
    (ws : Properties.Pouw.Period.Executions (Paths Bool m × (Equiv.Perm (Fin 144) × Unit)) 1),
    ¬ Properties.Pouw.Period.ServiceSound p ws (fun _ _ _ _ => ()) (fun _ _ _ _ => False) 0

end Properties.Accounting.JointBridge
