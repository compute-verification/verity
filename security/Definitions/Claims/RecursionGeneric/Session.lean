module

public import Definitions.Core.Interaction

/-! One session of a compiled verifier, as a prover and coins of its own. -/

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
