module

public import Definitions.Core.Challenger

/-! The challenger's premises, each a named `Prop` about one run of a challenger that flips `n` coins. -/

@[expose] public section

namespace Verity.Assumptions

open Verity.Challenger

/-- **`challenger-logs-coins`**: the challenger logs every coin it releases, and each is its own flip: on every run,
the coins its log shows released (`Log.coins`) are its flips `ω 0, ω 1, …`, in order, as far as the run went. -/
def ChallengerLogsCoins {R C : Type} {n : Nat} (log : (Fin n → C) → Log R C) : Prop :=
  ∀ ω, (log ω).coins <+: List.ofFn ω

/-- **`flips-after-recorded`**: the challenger flips coin `i` only after everything its log holds before that coin
(`Log.before`: the commitments coin `i` samples, and the coins released before it) is recorded. So that part of the
log is fixed before coin `i` exists: on two runs whose flips agree before `i`, it is the same. -/
def FlipsAfterRecorded {R C : Type} {n : Nat} (log : (Fin n → C) → Log R C) : Prop :=
  ∀ (i : Nat) (ω ω' : Fin n → C), (∀ j : Fin n, j.val < i → ω j = ω' j) → (log ω).before i = (log ω').before i

/-- **`verifier-reads-log`**: the verifier reads its coins only from the challenger's log: on every run, the coins it
reads (`coins ω`) are the ones the log released (`Log.coins`), in order. -/
def VerifierReadsLog {R C : Type} {n : Nat} (coins : (Fin n → C) → List C) (log : (Fin n → C) → Log R C) : Prop :=
  ∀ ω, coins ω = (log ω).coins

end Verity.Assumptions
