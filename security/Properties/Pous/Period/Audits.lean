module

public import Properties.Pous.EndToEnd
public import Definitions.Pous.P2.Clip

/-!
# What the verifier reads of one PoUS audit, and a prover strategy in the ring model

The vocabulary PoUS's per-run guarantees and its instantiation (`Properties.Assumptions.Pous`) are stated over: what
the verifier reads of one audit (`Audits`), with its acceptance, its break, what an accepted audit certifies and the
memory challenger's on-time record; a prover strategy in the ring model (`Strategy`), with its trace and its answers;
and the memory challenger's sample as the ring game reads it. It reads the deployed check and the ring game
(`Properties.Pous.EndToEnd`), never the joint run, so the assumptions home imports it without the device model.
-/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **What the verifier reads of a period's PoUS audit**, at the joint run's coins: the verifier's list of
memory-challenger keys and the session id the auditor designated, fixed for the period; the server's commitment and
salt, its `C*` (`store`), setup's public outputs, the memory challenger's table of row digests and the payload as the
ring game's bits (`W`), each fixed before the capacity coins are flipped, so a function of the compute and bandwidth audits'
coins alone; and the memory challenger's signed session and its device record (`MemoryChallenger.Properties.Challengers`), a
function of all the run's coins. -/
structure Audits (ΩW ΩQ : Type) where
  /-- the verifier's list of memory-challenger keys (never the run's record) -/
  trusted : List ByteArray
  /-- the session id the auditor designated -/
  sid : ByteArray
  /-- the server's commitment and salt -/
  commitment : ΩW × ΩQ → SetupProtocol.Commitment
  /-- `C*`, the blocks setup checks -/
  store : ΩW × ΩQ → Array (Array ByteArray)
  /-- setup's public outputs -/
  pub : ΩW × ΩQ → SetupProtocol.Public
  /-- the memory challenger's table of `C*`'s row digests -/
  digests : ΩW × ΩQ → ByteArray
  /-- the payload `C*` decodes to, as the ring game's bits -/
  W : ΩW × ΩQ → Bits sch.n
  /-- the memory challenger's signed session -/
  session : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
    PousCheck.Run.ChallengePhaseRecord
  /-- the memory challenger's record of its sessions -/
  device : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) →
    MemoryChallenger.Properties.Challengers

/-- **A prover strategy in the ring model**, at the other audits' coins, with no bound on its state or its time: what
survives setup is any list of bits computed from the payload `W`, the public parameters, `C` and the whole primitive
(perfect isolation: nothing else survives); round `j`'s answer to index `i` is any program over the ring, from the
state the previous round left, returning the block and the next state. The memory challenger's coins reach it only as
the challenger releases them: each index as the previous answer returns, and the coin commitment only once the session
is signed, after the last answer. -/
structure Strategy (ΩW ΩQ : Type) where
  /-- what survives setup -/
  pre : ΩW × ΩQ → Bits sch.n → Bits sch.t → sch.Code → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω → List Bool
  /-- round `j`'s answer to index `i` from the state `σ` -/
  step : ΩW × ΩQ → Bits sch.n → Bits sch.t → ℕ → List Bool → Fin sch.B →
    OracleAlgorithm (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)) (Bits sch.ℓ × List Bool)

namespace Audits

variable {ΩW ΩQ : Type} (P : Audits ΩW ΩQ)

/-- The verifier's whole decision on the period's audit at the coins `ω` (`Pous.EndToEnd.Accepted`): setup accepts
`C*` and publishes `pub`, and the composed check accepts the session. -/
def Accepts (ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop :=
  Accepted P.trusted (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1))
    (P.session ω) P.sid

/-- The period's own data holds a break at the coins `ω`: a SHA-512 collision among the strings the session and `C*`
hash or an LMS forgery under a key of the verifier's list (`RunBreak`), or a SHA-512 collision in the table's and `C*`'s
trees (`TableBreak`). -/
def Break (ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop :=
  RunBreak P.trusted (P.session ω).signed (P.device ω) (P.store (ω.1, ω.2.1)) ∨
    TableBreak (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1))

/-- What an accepted period certifies outside a break (`Pous.EndToEnd.RunCertifies`'s conclusion): `C*` decodes to a
payload with the server's commitment, setup published its root, the session audits that setup at the deployed point
under the designated id and is signed under a key of the verifier's list, and that key's holder ran a session with the
accepted commitment in which every recorded answer is the challenged block of `C*`, on time on its own clock. -/
def Certified (ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop :=
  DecodesToCommitment q (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) ∧
    Published q (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1)) ∧
    AuditsSetup (P.pub (ω.1, ω.2.1)) (P.session ω).commitment ∧ (P.session ω).commitment.session = P.sid ∧
    (P.session ω).signed.publicKey ∈ P.trusted ∧
    ∃ run ∈ (P.device ω).sessions (P.session ω).signed.publicKey,
      run.commitment = (P.session ω).commitment ∧ AnsweredOnTime run (P.store (ω.1, ω.2.1))

/-- **The memory challenger's signed record has every answer on time**: the session key's holder ran a session with the
signed commitment, at a limit within `Δ + RTT`, in which every one of the `k` recorded answers is the challenged block
of `C*` and arrived within the limit on the challenger's own clock (`AnsweredOnTime`). It reads the challenger's record
only, never what the prover reports. -/
def OnTime (ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : Prop :=
  ∃ run ∈ (P.device ω).sessions (P.session ω).signed.publicKey,
    run.commitment = (P.session ω).commitment ∧ run.limitNs ≤ LIMIT_NS ∧ AnsweredOnTime run (P.store (ω.1, ω.2.1))

/-- **The device's trace** of the period's session in the ring model, for the strategy `τ` at the other audits' coins
`o`, the capacity coins `x` and the memory challenger's indices `I`: for each of the 105 rounds, the state `τ` starts
it from and the program it runs on index `I j`, the first round starting from what survived setup. The coin
commitment is not among its inputs: the challenger releases it after the last answer. -/
noncomputable def trace (τ : Strategy ΩW ΩQ) (o : ΩW × ΩQ)
    (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) (I : Fin 105 → Fin (2 ^ 19)) :
    List (List Bool × OracleAlgorithm (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)) (Bits sch.ℓ × List Bool)) :=
  seqRounds ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1)
    (τ.step o (P.W o) (sch.enc (P.W o) x.1 ()).1) 0 (List.ofFn I)
    (τ.pre o (P.W o) (sch.enc (P.W o) x.1 ()).1 (sch.enc (P.W o) x.1 ()).2 x.1)

/-- The blocks `τ` returns in its trace's rounds, on the period's primitive. -/
noncomputable def answers (τ : Strategy ΩW ΩQ) (o : ΩW × ΩQ)
    (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) (I : Fin 105 → Fin (2 ^ 19)) :
    List (Bits sch.ℓ) :=
  (P.trace τ o x I).map fun e => (e.2.run ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1)).1

end Audits

/-- **The memory challenger's sample, as the ring game reads it** (`Passes`'s): the code's sampler reads index `I j` from
stream `j` at `2^19` blocks, and the code builds the coin tree over those indices under the session id `sid`,
publishing `v` (`coinView`). -/
def Drawn (sid : ByteArray) (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (I : Fin 105 → Fin (2 ^ 19)) (v : ByteArray × ByteArray) : Prop :=
  (∀ j, FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf (x.2.2 j)) = some (I j).val) ∧
    coinView sid x.2.1 I = some v

end Pous.Period
