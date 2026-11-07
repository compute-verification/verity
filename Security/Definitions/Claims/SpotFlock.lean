module

public import Definitions.Claims.SamplingTwo

/-!
# SPOT-Flock's layers and the interfaces between them

SPOT-Flock is four layers, each with its own guarantee, meeting at two interfaces.

* **L1, Flock**: a proof system for proof blocks (`Claims.SamplingGeneric.ZkArg`), knowledge sound (`Arg.KS`) and zero
  knowledge with a public outcome (`ZkArg.ZK`) at one hash: `Exports`.
* **L2, the sampled compiler** (`Claims.SamplingGeneric`, one and two stages): from a proof system's guarantee and a
  compiled verifier's contract, the compiled verifier's (`CompilerKS`, `CompilerKSReg`, `CompilerKSCount`,
  `CompilerBits`, `CompilerZK`, `CompilerKSTwo`, `CompilerExfilTwo`). It reads nothing of Flock or of the deployment.
* **L3, the deployment**: the deployed sampled verifier with its Flock sessions as a compiled verifier over L1's
  proof system, and the compiler's contract at it: `Refines` (one stage) and `RefinesTwo` (two stages). Each field is a
  fact of the deployed code, proved at the deployed instance.
* **L4, recursion**, imports L1 to L3; none of them imports it.

A deployed guarantee is L2's theorem at L1's export and L3's (`Composes`, `ComposesTwo`). Its
premises are the trust list's, each applied outside the layers: SHA-512's collision resistance to the collision
finders a bound names; uniform random bytes and the honest verifier server to the law of the live coins; for zero
knowledge, the leaf commitment's hiding (the hiding gap a bound names) and the release's sources.
-/

@[expose] public section

namespace Claims.SpotFlock

open Claims Claims.SamplingGeneric

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **L1's export**: the proof system for proof blocks `A` is knowledge sound and zero knowledge with a public outcome,
at the hash `H`. -/
structure Exports (H : Msg → Msg) (A : ZkArg Coin) : Prop where
  ks : A.toArg.KS H
  zk : A.ZK H

/-- **L3's export, one stage**: at the hash `H`, the masked compiled verifier `L` over `A` commits with a binding
commitment, and meets the compiler's knowledge-soundness contract (`Compiled.Sessions`), its opening checks' count
(`Compiled.OpenCallsLe`) and its zero-knowledge contract (`ZkCompiled.ZkSessions`). -/
structure Refines (H : Msg → Msg) {A : ZkArg Coin} (L : ZkCompiled Coin A) : Prop where
  binds : L.commit.Binds H
  sessions : L.toCompiled.Sessions H
  openCalls : L.toCompiled.OpenCallsLe H
  zkSessions : L.ZkSessions H

/-- **L3's export, two stages**: at the hash `H`, the two-stage compiled verifier `L` over `A` commits with a binding
commitment and meets the two-stage contract (`CompiledTwo.Sessions`). It has no zero-knowledge contract: the
two-stage view includes the interior registration, which commits under the ZK server's salts and so reads the witness
and the tape, and two-stage zero knowledge is the sampled audit's over that view. -/
structure RefinesTwo (H : Msg → Msg) {A : Arg Coin} (L : CompiledTwo Coin A) : Prop where
  binds : L.commit.Binds H
  sessions : L.Sessions H

/-- **SPOT-Flock's one-stage composition**: L2's theorems at L1's export and L3's give the compiled verifier's
knowledge soundness in each form, its bits, and its zero knowledge. -/
def Composes : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : ZkArg Coin) (L : ZkCompiled Coin A)
    (H : Msg → Msg), Exports H A → Refines H L →
      L.toCompiled.KS H ∧ L.toCompiled.KSReg H ∧ L.toCompiled.KSCount H ∧ L.toCompiled.Bits H ∧ L.ZK H

/-- **SPOT-Flock's two-stage composition**: L2's two-stage theorems at L1's export and L3's give the two-stage compiled
verifier's knowledge soundness and exfiltration bound. -/
def ComposesTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : ZkArg Coin) (L : CompiledTwo Coin A.toArg)
    (H : Msg → Msg), Exports H A → RefinesTwo H L → L.KS H ∧ L.Exfil H

end Claims.SpotFlock
