import Definitions.Core.Challenger

/-!
# Where coins come from: the challenger's premises

What a guarantee takes of the coins it reads, each a named `Prop` about one run of a challenger
(`Definitions.Core.Challenger`) that flips `n` coins of `C`: its flips `ω : Fin n → C`, and `log ω`, its log on those
flips against a fixed prover strategy (a randomized prover is a mixture of fixed ones). Each statement that takes them
weighs the flips evenly, as `Game.coin` weighs a live coin, and nothing else is random in it: the verifier has no
randomness of its own.

* `ChallengerLogsCoins`: the coins the log shows released are the challenger's flips, in the order it flipped them;
* `FlipsAfterRecorded`: the challenger flips each coin only after what its log holds before that coin is recorded, so
  nothing there depends on that coin or a later one;
* `VerifierReadsLog`: the verifier's coins are the ones it reads from the challenger's log.

The challenger is an auditor device on a host the prover doesn't control (`trusted-host`,
`Verity.Assumptions.TrustedHost`), which is why we believe the first two. The third is the verifier's: it reads coins
from nowhere else. The two cases where randomness is derived from a coin by hashing are named in
`Definitions.Core.Challenger`; they read their coin through these premises.
-/

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
