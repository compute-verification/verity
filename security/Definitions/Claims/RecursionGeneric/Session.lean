module

public import Definitions.Core.Interaction

/-!
# One session of a compiled verifier, as a prover and coins of its own

A compiled verifier runs Π_out's sessions inside its own run. Session `s` is then a run of Π_out's verifier whose prover
is the compiled run's prover after the messages before the session (`sessProver`), and whose coins are the compiled
run's from the session's first round on (`sessCoins`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Recursion

open Claims

/-- What a verifier's answer tells the prover: the message, or nothing once it decided. -/
def outMsg : Out → Msg
  | .msg m => m
  | _ => []

/-- **The prover of one session**: `P` at tape `ω`, after the messages `pre` of the rounds before the session. -/
def sessProver {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (pre : List (Msg × Out)) : Prover Unit :=
  ⟨fun _ hist _ => P.next x (pre.map (fun p => (p.1, outMsg p.2)) ++ hist) ω⟩

/-- The `n` coins from round `o` on. -/
def sessCoins {Coin : Type} [Inhabited Coin] {k : ℕ} (o : ℕ) (d : Fin k → Coin) (n : ℕ) : Fin n → Coin :=
  fun i => coinSeq d (o + i.val)

end Claims.Recursion
