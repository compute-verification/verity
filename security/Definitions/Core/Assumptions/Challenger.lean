module

public import Definitions.Core.Challenger

/-!
# Where coins come from: the challenger's premises

What a guarantee takes of the coins it reads, each a named `Prop` about one run of a challenger
(`Definitions.Core.Challenger`) that flips `n` coins, coin `i` of its own type `C i`: its flips `ω : (i : Fin n) → C i`,
and `log ω`, its log on those flips against a fixed prover strategy (a randomized prover is a mixture of fixed ones),
which shows each coin `i` it releases as `tag i (ω i)`, of the log's one coin type `T`. Coins of one type are the
constant family, logged as themselves (`tag i c = c`); coins of several types are logged with their family, the code's
`(label, index)`, or `⟨i, c⟩` in `Σ i, C i`. Each statement that takes them weighs the flips evenly, as `Game.coin`
weighs a live coin, and nothing else is random in it: the verifier has no randomness of its own.

* `ChallengerLogsCoins`: the coins the log shows released are the challenger's flips, in the order it flipped them;
* `FlipsAfterRecorded`: the challenger flips each coin only after what its log holds before that coin is recorded, so
  nothing there depends on that coin or a later one;
* `VerifierReadsLog`: the verifier's coins are the ones it reads from the challenger's log.

The challenger is an auditor device on a host the prover doesn't control (`trusted-device`,
`Verity.Assumptions.TrustedDevice`), which is why we believe the first two. The third is the verifier's: it reads coins
from nowhere else. The two cases where randomness is derived from a coin by hashing are named in
`Definitions.Core.Challenger`; they read their coin through these premises.
-/

@[expose] public section

namespace Verity.Assumptions

open Verity.Challenger

/-- **`challenger-logs-coins`**: the challenger logs every coin it releases, and each is its own flip: on every run,
the coins its log shows released (`Log.coins`) are its flips `ω 0, ω 1, …` as the log shows them (`tag`), in order, as
far as the run went. -/
def ChallengerLogsCoins {R T : Type} {n : Nat} {C : Fin n → Type} (tag : (i : Fin n) → C i → T)
    (log : ((i : Fin n) → C i) → Log R T) : Prop :=
  ∀ ω, (log ω).coins <+: List.ofFn fun i => tag i (ω i)

/-- **`flips-after-recorded`**: the challenger flips coin `i` only after everything its log holds before that coin
(`Log.before`: the commitments coin `i` samples, and the coins released before it) is recorded. So that part of the
log is fixed before coin `i` exists: on two runs whose flips agree before `i`, it is the same. -/
def FlipsAfterRecorded {R T : Type} {n : Nat} {C : Fin n → Type} (log : ((i : Fin n) → C i) → Log R T) : Prop :=
  ∀ (i : Nat) (ω ω' : (j : Fin n) → C j), (∀ j : Fin n, j.val < i → ω j = ω' j) → (log ω).before i = (log ω').before i

/-- **`verifier-reads-log`**: the verifier reads its coins only from the challenger's log: on every run, the coins it
reads (`coins ω`) are the ones the log released (`Log.coins`), in order. -/
def VerifierReadsLog {R T : Type} {n : Nat} {C : Fin n → Type} (coins : ((i : Fin n) → C i) → List T)
    (log : ((i : Fin n) → C i) → Log R T) : Prop :=
  ∀ ω, coins ω = (log ω).coins

end Verity.Assumptions
