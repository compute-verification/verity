import Proofs.Flock.Claims.Sampling.ZK

/-!
# The sampled masked protocol, every coin committed at `Hello`

The sampled verifier's coins are live: the challenger commits to all of them, the selection draw's and each checked
unit's session's, under one hiding coin tree (`coin-tree/hm96-sha512/v2`, the tree C-Flock's masked sessions commit) in
its answer to the prover's `Hello`, and opens each at its round, after the prover's message of that round; the prover
checks each opening against the root with the coin-tree walk before it reads the coin, and from the first that fails
sends `abort`. `Sampled c` is that protocol's data at the code `c`, and `Sampled.leaky` the protocol, as a
`Claims.Sampling.Leaky`:

* **Rounds**: `Hello`, then the sampled verifier's rounds after its first, each answered with the opening of that
  round's coin, the last with the decision. The sampled verifier's first round is the `Hello` round (`HelloRounds`), so
  the protocol's `K + 2` rounds are `c.V.k` (`Sampled.rounds`), and the coins it opens are `c.V`'s at rounds `1` to
  `c.V.k − 2`, in the deployed order: the first stage's draw, then the rounds `c.stage2` holds, each checked unit's
  session's at its `c.rounds`.
* **The opening check**: the coin-tree walk `treeChk` and the opening's coins as its coin (`decOf`), as ZKS's reference
  session's (`RefSession.refP`), so `CoinCheck` holds by definition (`Sampled.coinCheck`).
* **The messages**: after the coins `cs`, the sampled prover in public-coin form (`P w`) at the answers the public-coin
  verifier sends at `cs` (`ans`), its own earlier messages replayed on its tape (`replay`); the dummy-witness messages
  are the same at the dummy witness of the leaf layer `lx` (`dw x lx`), and the SHVZK simulator is given `lx` (`sim`).

`zk_sampled` is `ZKAt` at the deployed record and `P = (Sampled.leaky S).G.prover`, at the key's hiding gap, from
`HelloRounds` and the four models' Props at each leaf layer; `zk_sampled_mask` is `Claims.Sampling.ZK` at the record
masked at `tapeOf`, where the code's reference prover on the masked tape is that protocol's (`Honest`); and
`sampling_deployed_sampled` is `sampling_deployed` with it.
-/

namespace SamplingDeploy

open Claims Claims.Sampling Claims.ZKMal

instance instDecidableEqCoin : DecidableEq FlockDeploy.Coin := inferInstanceAs (DecidableEq (Fin 65536 → Flock.F128))

/-- **A public-coin prover's message after the answers `bs`**, from the history `hist`: its own messages in order, each
paired with its answer, then its message at the whole history. -/
def replay (nxt : List (Msg × Msg) → OC Msg) : List (Msg × Msg) → List Msg → OC Msg
  | hist, [] => nxt hist
  | hist, b :: bs => ZKMal.ocBind (nxt hist) fun a => replay nxt (hist ++ [(a, b)]) bs

/-- **The sampled verifier's first round is the `Hello` round**: at every statement that parses it has that round and
its last. -/
def HelloRounds (c : Code) : Prop := ∀ x s, c.parse x = some s → 2 ≤ c.V.k x

/-- **The sampled masked protocol's data** at the code `c`: the prover's nonce (its `Hello`) and tape; the coin tree the
challenger commits at `Hello` and how an opening is read (`CoinCheck`'s data, ZKS's `RefSession`'s); the `Hello` and
`abort` messages; the answer the public-coin verifier sends at a coin (`ans`); the sampled prover in public-coin form
(`P`); the dummy witness at a statement and leaf layer (`dw`); and the SHVZK simulator given the leaf layer (`sim`). -/
structure Sampled (c : Code) where
  Nonce : Type
  [finNonce : Fintype Nonce]
  [neNonce : Nonempty Nonce]
  Tape : Type
  [finTape : Fintype Tape]
  [neTape : Nonempty Tape]
  CoinT : Type
  CSalt : Type
  Tr : Msg → Msg → FlockSoundness.ZK.GK.CoinTree Dg CoinT CSalt
  nb : Nonce → List UInt8
  root : Msg → Msg → Dg
  parse : Msg → Option (FlockSoundness.ZK.GK.CoinOpening Dg CoinT CSalt)
  pos : Msg → Msg → ℕ → List UInt8
  idx : Msg → Msg → ℕ → ℕ
  len : Msg → Msg → ℕ → ℕ
  depth : Msg → Msg → ℕ → ℕ
  dc : List CoinT → FlockDeploy.Coin
  hello : Msg → Nonce → Msg
  abort : Msg
  ans : FlockDeploy.Coin → Msg
  P : Wit → Prover Tape
  dw : Msg → List Msg → Wit
  SimRand : Msg → Type
  [finSim : ∀ x, Fintype (SimRand x)]
  [neSim : ∀ x, Nonempty (SimRand x)]
  sim : List Msg → (x : Msg) → Nonce → (Fin (c.V.k x - 2) → FlockDeploy.Coin) → SimRand x → OC (List Msg)

attribute [instance] Sampled.finNonce Sampled.neNonce Sampled.finTape Sampled.neTape Sampled.finSim Sampled.neSim

namespace Sampled

variable {c : Code} (S : Sampled c)

/-- What an opening reads as: its coins, or the default where it does not parse. -/
def decOf (m : Msg) : FlockDeploy.Coin :=
  match S.parse m with
  | some op => S.dc op.cs
  | none => default

/-- **The honest message after the coins `cs`**: `P w` on the tape `t`, at the answers the public-coin verifier sends
at `cs`, its earlier messages replayed. -/
def msgAt (x : Msg) (w : Wit) (t : S.Tape) (cs : List FlockDeploy.Coin) : OC Msg :=
  replay (fun hist => (S.P w).next x hist t) [] (cs.map S.ans)

/-- **The sampled masked protocol**: `Hello`, then `c.V.k − 2` coins, each opened and checked by the coin-tree walk,
then the last message; its messages `msgAt`, on the dummy witness of the empty leaf layer for its own `dmsg`. -/
def proto : Proto Wit where
  Coin := FlockDeploy.Coin
  Nonce := S.Nonce
  Rand := S.Tape
  SimRand := S.SimRand
  K x := c.V.k x - 2
  hello := S.hello
  abort := S.abort
  dec := S.decOf
  chk x ν ρ j m :=
    treeChk (S.Tr x ρ) (S.nb ν) (S.root x ρ) (S.pos x ρ j) (S.idx x ρ j) (S.len x ρ j) (S.depth x ρ j) (S.parse m)
  msg x w _ t cs := S.msgAt x w t cs
  dmsg x _ t cs := S.msgAt x (S.dw x []) t cs
  sim := S.sim []

/-- **`L`, the sampled masked protocol with the leaf layer as its leakage**: at `lx`, the dummy-witness messages are
`msgAt` at the dummy witness of `lx`, and the SHVZK simulator is `sim lx`. -/
def leaky : Leaky Wit where
  G := S.proto
  dmsg lx x _ t cs := S.msgAt x (S.dw x lx) t cs
  sim := S.sim

/-- **Its rounds are the sampled verifier's**, `K + 2 = c.V.k`, at every statement that parses, when the sampled
verifier's first round is the `Hello` round. -/
theorem rounds (h : HelloRounds c) : ∀ x s, c.parse x = some s → S.leaky.G.K x + 2 = c.V.k x := fun x s hs => by
  have := h x s hs
  show c.V.k x - 2 + 2 = c.V.k x
  omega

/-- **`CoinCheck` holds of `L`**: its check is the coin-tree walk and its reading the opening's coins, by definition. -/
theorem coinCheck : S.leaky.G.CoinCheck :=
  ⟨S.CoinT, S.CSalt, S.Tr, S.nb, S.root, S.parse, S.pos, S.idx, S.len, S.depth, S.dc, fun _ _ _ _ _ => rfl,
    fun m op h => by
      show S.decOf m = _
      simp only [decOf, h]⟩

end Sampled

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

/-- **Sampling's zero knowledge at the deployed record, of the sampled masked protocol**: `zk_deploy_of_models_gap` at
`L = Sampled.leaky S`, its rounds from `HelloRounds` and its coin check by definition; what remains is the four models'
Props at each leaf layer. It is `ZKAt`, of `L`'s prover. -/
theorem zk_sampled (H : Msg → Msg) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t) {Salt B Cc : Type}
    [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
    [DecidableEq Cc] (My : Salt → B) (cs : Salt → Cc)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).CoinModel x Salt B Cc)
    (nL nB : Msg → ℕ) (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x)
    (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (S.leaky.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (S.leaky.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (S.leaky.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (S.leaky.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) :
    ZKAt H (deploy rel c) S.leaky.G.prover (S.leaky.simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  zk_deploy_of_models_gap rel c H S.leaky t ht (S.rounds hR) My cs Mp Mc nL nB hnL hnB hL hS hN hI S.coinCheck

/-- **Sampling's zero knowledge at the deployed record masked at the sampled masked protocol's tapes**:
`zk_deploy_mask_of_models_gap` at `L = Sampled.leaky S`, where the code's reference prover on the masked tape `tapeOf ω`
is `L`'s (`Honest`); what remains is that and the four models' Props at each leaf layer. -/
theorem zk_sampled_mask (H : Msg → Msg) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] [DecidableEq Cc] (My : Salt → B) (cs : Salt → Cc)
    (tapeOf : S.leaky.G.Nonce × S.leaky.G.Rand → c.Tape) (hP : Honest c H S.leaky.G tapeOf)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).CoinModel x Salt B Cc)
    (nL nB : Msg → ℕ) (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x)
    (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (S.leaky.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (S.leaky.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (S.leaky.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (S.leaky.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) :
    ZK H ((deploy rel c).mask S.leaky.G.Nonce S.leaky.G.Rand tapeOf (S.leaky.simLeakE t (t * 3 ^ t)))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  zk_deploy_mask_of_models_gap rel c H S.leaky t ht (S.rounds hR) My cs tapeOf hP Mp Mc nL nB hnL hnB hL hS hN hI
    S.coinCheck

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record, with zero knowledge of the sampled masked protocol**:
`sampling_deployed_models` at `L = Sampled.leaky S`, its rounds from `HelloRounds` and its coin check by definition, at
the record masked at `tapeOf` where the code's reference prover is `L`'s (`Honest`). -/
theorem sampling_deployed_sampled (H : Msg → Msg) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (hfixed : DrawFixed (deploy rel c)) (hsessions : SessionRuns H (deploy rel c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy rel c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens H (deploy rel c) iroots) (hfix : InteriorFixed H (deploy rel c) iroots)
    (hprog : ProgramScope (deploy rel c)) (hchecks : ChecksScope H (deploy rel c)) (hreveals : RevealsLe c)
    (hComplete : Complete H (deploy rel c)) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] [DecidableEq Cc] (My : Salt → B) (cs : Salt → Cc)
    (tapeOf : S.leaky.G.Nonce × S.leaky.G.Rand → c.Tape) (hP : Honest c H S.leaky.G tapeOf)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).CoinModel x Salt B Cc)
    (nL nB : Msg → ℕ) (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x)
    (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (S.leaky.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (S.leaky.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (S.leaky.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (S.leaky.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      KS H (deploy rel c) ∧ KSCount H (deploy rel c) ∧ Bits H (deploy rel c) ∧
      Efficient H (deploy rel c) FlockDeploy.zkStepPoly ∧ Complete H (deploy rel c) ∧
      ZK H ((deploy rel c).mask S.leaky.G.Nonce S.leaky.G.Rand tapeOf (S.leaky.simLeakE t (t * 3 ^ t)))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
        fun _ => 4 * t * 3 ^ t :=
  sampling_deployed_models rel c H hKS hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks hreveals
    hComplete S.leaky t ht (S.rounds hR) My cs tapeOf hP Mp Mc nL nB hnL hnB hL hS hN hI S.coinCheck

end SamplingDeploy
