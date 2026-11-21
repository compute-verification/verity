module

public import Proofs.Flock.Soundness.Assumptions.Recursive
public import Proofs.Flock.Soundness.Assumptions.Zk
public import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Defs
public import Proofs.Flock.Soundness.ZK.HidingAdaptive
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# The recursive audit's guarantees (trusted)

The recursive audit (`Proofs.Flock.Recursive.Defs`: `recGame`): the developer registers `R` (the hidden circuit's
commitment among it); the auditor samples `ω`, the hidden circuit's units to prove; in each of the inner protocol's `I.r`
rounds the developer's ZK server sends the auditor a salted SHA-512 commitment to the round's inner message (Flock run
with `--zk` off on the workers), and the auditor answers with a live coin; then the developer registers `R₂`, the
values V*'s sessions read that the rounds do not fix (V*'s running sums `rec-acc`, and its opened rows and their siblings),
and Flock `--zk --registered` sessions of the public verifier circuits V* (`zs R ω t R₂ j`, `ZkOuter`, one per circuit
`(Vs j).c`, in a row: `outerV`) prove that openings of those commitments exist on which the inner verifier accepts. Each guarantee is a `def G : Prop` in the
form of `Pouw.Guarantees`; its proof is `Flock.SecurityProofs.G : Flock.Guarantees.G`
(`Proofs.Flock.Recursive.Guarantees`).

* `RecursiveSound`, for the auditor, in fork form (not a guarantee: the lock retired it, since its event is
  possibilistic and nearly empty; the recursion's guarantee is `Flock.Guarantees.RecursiveKS`): either a run of the prover's own strategy finds a SHA-512 collision
  (between two of V*'s sessions' registrant records, at a round's commitment, or in one of V*'s sessions), or the auditor accepts every one of V*'s sessions on a false
  hidden statement with probability at most V*'s sessions' statistical errors, plus each round's transfer slack, plus
  the inner protocol's soundness error. It assumes no hash. Its one new assumption is `Flock.Assumptions.VBridge`.
* `RecursiveRanRegistered`, for the auditor, in `RecursiveSound`'s fork form: on its hypotheses and a registration
  binding at each row of the private circuit, a round or session fork, or a run the prover reaches whose sessions all accept
  having read a row other than the registered one and whose opening of it collides with the developer's, or the auditor
  accepts every session while the registered private circuit's hidden statement fails with probability at most
  `RecursiveSound`'s bound. `RanRegisteredRun` is the one-session form, `PrivateCircuit.RanRegistered` with its
  `cr/sha-512` hypothesis replaced by the same kind of break.
* `commitBytes`, `zkServerPairs`, `zkServerCommits`: the ZK server's commit strings and commitments when the inner coins
  are live, message `i` reading the commit strings before it, which (ii) of `RecursiveAudit` at the ZK server reads
  (`Proofs.Flock.Recursive.ZkServer`); `zkServerLeaf` and `idealLeaf` are a message's commitment and its ideal
  counterpart.
* `OuterViewZK`: the outer session's zero knowledge, an obligation on the view and simulator it ranges over.
* `regLeaves`: a registration by hm96 leaves of a private circuit's rows.

Flock's definitions and the recursion's model are still in the proofs package, so `Definitions.lean` does not import
this module: `Proofs.Flock.Recursive` does.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Spot.Sampling Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The recursive audit is sound.** V*'s sessions are of the circuits `(Vs j).c`, with hidden outputs and `(Vs j).J`
tables, under their sampling designs `(Vs j).samplingDesign`, each the full sample (`--draw subset:<units>`) on the operating system's bytes (A3,
`uniform/io-getrandombytes`); at each history `(R, ω, t)` and registration `R₂` of the outer proof, session `j` is
`zs R ω t R₂ j`, whose fields carry `verity-verify-proof verify --zk --registered`'s setup, the `live-verifier` record custody
(`ZkOuter.hRec`) and the registrant's record of the registered reads (`ZkOuter.rd`: the hidden circuit's rows, `R₂`'s
values, and the verifier's own). For every prover `σ` of the recursive audit, given V*'s bridge
(`Flock.Assumptions.VBridge`) and the inner protocol's soundness at `ε_in`, one of four holds. It needs `VBridge` in
full, which a protocol whose last round isn't committed (candidate 3 today) doesn't give.

* A registrant fork (`regFinder`): one run of `σ` reaches an outcome at whose history and registration two of V*'s
  sessions' registrant records clash (`RegColl`: two registered reads whose openings or hm96 leaves are a SHA-512
  collision), and outputs that collision. V*'s bridge reads off such outcomes (`VBridgeOff`).
* A round fork (`RoundFork`): for some round `i < r`, sample `ω` and commit string `pos`, round `i`'s fork finder
  (`forkAt`: two continuations of `σ` after its round-`i` commitment, each running the trial for `pos`) outputs a
  SHA-512 collision at an outcome it reaches.
* A session fork (`SessionFork`): at some leaf `σ` reaches, some outcome of V*'s sessions its strategy reaches and some
  session `j`, a finder of that session run from the strategy `σ` names there (its tables' knowledge finders, its
  registered-value finders, the run that opens a sampled row's registered leaf, or its link finder: `ZkOuter.Fork`)
  outputs a SHA-512 collision at an outcome it reaches.
* $$\Pr[\text{every session of V* accepts} \wedge \neg\,\text{the hidden statement holds}] \le
    \mathbb E\Big[\sum_j \text{outer bound}_j\Big] + r\cdot\mathbb E\Big[\sum_j \text{stage slack}_j\Big] +
    \varepsilon_{in},$$
  the expectations over the sample, the rounds' coins at the prover's leaves (`expAt`) and the sessions (`sumV`), each
  term at the registration `R₂` the prover sends and the strategy it names for its session. A session's outer bound
  (`ZkOuter.boundCR`) is its sampling design's miss at one bad unit plus the sampled proof blocks' statistical errors: the tables'
  (`tableErrZ`: each sampled table's `bound` over `1 − N₀/(ek)`), the link's (`linkErrZC`: `Q_s/(1 − ρ)·(1/(eM) +
  k/(eR_w))`) and the waits' (`waitErrZC`: `Q_s/(1 − ρ)·k/(eR_w)`); its stage slack (`ZkOuter.slackCR`) is the same
  without the miss.

Every `∃` in a fork ranges over `σ`'s own runs and what they output, never over arbitrary strings. The bound is exact
for a prover that never collides on a fork, and says nothing for one that collides on a single pair: no term counts
SHA-512 evaluations.

**A nearly empty event**: Its event asks that the inner statement have no witness (`¬
holds`, with `holds` = `∃ z, Satisfies z`). The inner statement pins each row only by its hm96 commit string `b ‖ c`,
with the salt a free witness wire, so a commitment to a wrong row under a fresh salt also opens to the right row except
with probability at most `4·hidingGap` (`VacuityProbes.Hm96Opens.opens_to_any_row`, `shift_reachable`, probes
outside this tree). For any cheater the event is then nearly empty, and the bound bounds it, not a wrong
computation. The fix is the knowledge form the claims API states recursion in; `EndToEndHidden` avoids this by
extraction. The knowledge form, `Claims.Recursion.KS`, is at the recursion's level `Claims.Recursion.lam`, which no
theorem evaluates at the deployment. Route 2C's bound λ₂ (`lamLin2`) evaluates to at least 135.7 at the deployed
records (`RecDeploy.lamLin2_deployed_ge_dep_tight`), and `KSLin2` at the deployment's dense statements (`KSLin2On` at
`RecDeploy.DenseAt`) is proved (`Flock.SecurityProofs.RecursiveKS`), Holo statements being candidate 3's own model, and
`BitsLin2` at those statements follows from it (`bitsLin2On_of_ksLin2On`). This fork form states no bits.

**Retired from the lock**: its break is possibilistic, so the lock lists no theorem of it; the recursion's
guarantee is the knowledge form at the deployed reader, `Flock.Guarantees.RecursiveKS` (`Proofs.Flock.Recursive.KSStatement`). -/
def RecursiveSound : Prop :=
  ∀ {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    -- V*'s sampling designs: A3
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).samplingDesign.Lb (Vs j).samplingDesign.src)
    -- V*'s sessions at each history and registration `R₂`
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, the rounds' commit strings, the ZK server's
    -- commitments and the messages they open to, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    -- the inner protocol
    {εin : ℝ≥0∞} (_hin : InnerSound I εin),
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin

/-- **`RecursiveSound` with no A3**: `RecursiveSound` without its A3 premise, at the same bound. V*'s sampling designs are the full
sample (`--draw subset:<units>`), which samples every unit at every value of the operating system's bytes
(`execOS_subset_full_mem`), so no session's miss reads those bytes. -/
def RecursiveSoundNoA3 : Prop :=
  ∀ {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    -- V*'s sessions at each history and registration `R₂`
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, the rounds' commit strings, the ZK server's
    -- commitments and the messages they open to, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    -- the inner protocol
    {εin : ℝ≥0∞} (_hin : InnerSound I εin),
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin

/-- **The ZK server's commitment** to one message `msg` (an inner round message or a Merkle tree top) under the salt `y`:
the hm96-sha512 leaf of the message's SHA-512 digest, `H(SHA-512(msg) ⊕ M·y, c(y))` (`ZkView.leafH`, `ZkView.My`,
`ZkView.saltHash` at the pinned key; `ZkView.leafSplit` states that this is the executable's `Hm96.Default512.leaf`). An
honest ZK server sends the auditor this, under a fresh uniform salt, for every message, and nothing else of them. The salt
is 192 bytes from the operating system, uniform under A3 (`uniform/io-getrandombytes`); at those bytes `b` the
executable's leaf is `zkServerLeaf msg (ZkView.saltOfBytes b)` (`Flock.SecurityProofs.zkServerLeaf_os`). -/
def zkServerLeaf (msg : ByteArray) (y : ZkView.Salt) : ByteArray :=
  ZkView.leafH (ZkView.bitsOf (Sha512.hash msg) + ZkView.My y, ZkView.saltHash y)

/-- **An ideal commitment**: the leaf of a uniform `u` in place of `SHA-512(msg) ⊕ M·y`, beside the salt's own hash; it
reads no message. -/
def idealLeaf (u : ZkView.Bits × ZkView.Salt) : ByteArray :=
  ZkView.leafH (u.1, ZkView.saltHash u.2)

/-- **An hm96-sha512 commit string from its two halves**, `b ‖ c`: the 128 bytes a ZK server leaf hashes
(`ZkView.leafH` is `Default512.treeLeaf` of them). -/
def commitBytes (u : ZkView.Bits × ByteArray) : ByteArray := ZkView.bytesOf (k := 64) u.1 ++ u.2

/-- **The ZK server's commit strings under live inner coins**, as their two halves: string `i` is
`(SHA-512(m_i) ⊕ M·y_i, c(y_i))`, a `zkServerLeaf`'s before its leaf hash, where the message `m_i = msgs i l` reads the
commit strings `l` before it (`commitBytes` of their halves). The auditor has seen those strings when it flips the coins
message `i` answers, so the message may read them: each inner coin is live, sampled after the commitment before it. They
are `ZK.adaptiveLeaves` at the identity encoding: pair `i` is message `i`'s digest at the pairs before it, plus
`(M·y_i, c(y_i))`. -/
def zkServerPairs {nh : ℕ} (msgs : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (y : Fin nh → ZkView.Salt) :
    Fin nh → ZkView.Bits × ByteArray :=
  FlockSoundness.ZK.adaptiveLeaves (id : ZkView.Bits × ByteArray → ZkView.Bits × ByteArray)
    (fun i l => ZkView.bitsOf (Flock.Sha512.hash (msgs i fun j => commitBytes (l j))))
    (fun i => (ZkView.My (y i), ZkView.saltHash (y i)))

/-- **The ZK server's commitments under live inner coins**: commitment `i` is the hm96-sha512 leaf of message `i`'s
SHA-512 under its own fresh salt `y i`, `zkServerLeaf (msgs i l) (y i)`, where message `i` reads the commit strings `l`
before it (`zkServerPairs`). A message that reads none of them is the committed-coin case,
`zkServerLeaf (msgs i) (y i)`. -/
def zkServerCommits {nh : ℕ} (msgs : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (y : Fin nh → ZkView.Salt)
    (i : Fin nh) : ByteArray :=
  ZkView.leafH (zkServerPairs msgs y i)

/-- Obligation (the outer session is zero knowledge): at every auditor's tape `a`,
developer's randomness `p` and ZK server salts `y`, the auditor's view of the outer session, `view a h p y` at the
ZK server's commitments `h` to the messages `msgs a p` (`zkServerLeaf`), and the simulator `sim a h`, which reads the
auditor's tape and the commitments only, give every event probabilities within `ε_o`, over the session's randomness and
the simulator's. -/
def OuterViewZK {Ωa Ωp Ωo Ωs V : Type} [Fintype Ωo] [Fintype Ωs] (nh : ℕ) (msgs : Ωa → Ωp → Fin nh → ByteArray)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) (εo : ℝ≥0∞) : Prop :=
  ∀ a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
    prCoin (fun r : Ωo => E (view a (fun i => zkServerLeaf (msgs a p i) (y i)) p y r))
      ≤ prCoin (fun s : Ωs => E (sim a (fun i => zkServerLeaf (msgs a p i) (y i)) s)) + εo

/-- **A registration by hm96 leaves**: row `i` of the private circuit `P` (`rows P i`) committed as the ZK server commits a
message, `zkServerLeaf (rows P i) (r i)` under its own fresh uniform salt `r i`; the auditor sees `pub` of the leaves
(their Merkle root, and whatever else the registration record publishes of them) and nothing else of the rows or salts.
The registrant's opening of a `--zk --registered` session's registered read is such a row and salt (`ZkReg.regOpenZ`). -/
def regLeaves {Prog Rv : Type} {nr : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (P : Prog)
    (r : Fin nr → ZkView.Salt) : Rv :=
  pub fun i => zkServerLeaf (rows P i) (r i)

/-- **A session ran the registered program**, with its break tied to the run: `PrivateCircuit.RanRegistered` without its
`cr/sha-512` hypothesis. The root `rt` commits to a registered program `p` (for a private circuit, `pad C c`) through
the registrant's opening `op₀`, and the commitment binds (`PrivateCircuit.Commit.Binding`: two openings of one root to
two different programs give the extractor's pair `com.ext`, two different strings SHA-512 maps to one digest). Take a
session, a game `G` against any prover `σ`, whose acceptance `acc` shows that the program of its committed values
(`prog b`) opens `rt` (through `op b`), and suppose protocol 1's soundness bounds accepting with its bad event `Q` (at
least `K` incorrect proof blocks of the committed values, say) by `B`. Then either some outcome `σ` reaches accepts a program other
than `p`, and its opening and the registrant's collide (`Binding.Collides (com.ext (op b) op₀)`), or accepting with `Q`,
or with a program other than `p`, has probability at most `B`. `RanRegistered`'s finder plays the session once and
outputs the extractor's pair, so the first is a run of that finder finding a collision; nothing is assumed of SHA-512,
and the bound has no `q²/2^513`. -/
def RanRegisteredRun : Prop :=
  ∀ {Root P Op : Type} (com : PrivateCircuit.Commit Root P Op), com.Binding →
    ∀ (rt : Root) (p : P) (op₀ : Op), com.Opens rt p op₀ →
    ∀ {β : Type} (G : Game β) (σ : Strategy G) (acc Q : β → Prop) (prog : β → P) (op : β → Op),
      (∀ b, acc b → com.Opens rt (prog b) (op b)) →
    ∀ (B : ℝ≥0∞), prob (fun b => acc b ∧ Q b) G σ ≤ B →
      (∃ b, FlockSoundness.Game.Outcome G σ b ∧ acc b ∧ prog b ≠ p ∧
          FlockSoundness.Binding.Collides (com.ext (op b) op₀)) ∨
        prob (fun b => acc b ∧ (Q b ∨ prog b ≠ p)) G σ ≤ B

/-- **The recursive audit ran the registered private circuit**, in fork form. On `RecursiveSound`'s hypotheses (the same prover
`σ`, V*'s sampling designs and sessions, the bridge `Flock.Assumptions.VBridge` and `ε_in`), let the registration commit the
private circuit `p`, row `p i` at each position `i`, at the root `rt`; `com i` is the commitment at position `i`, binding
(`PrivateCircuit.Commit.Binding`: two openings of `rt` at `i` to two different rows give the extractor's pair `com.ext`,
two different strings SHA-512 maps to one digest), and the developer's opening of row `i` is `op₀ i`. V*'s sessions
read some of the rows, never necessarily all of them: on a run `o` whose sessions all accept, `reads o i` says the run
read row `i`, and `row o i` and `op o i` are the row and the opening of `rt` at `i` the run carries (`_hop`; the run's
output holds the strategy the prover names for each session, and with it the registered reads), and the hidden
statement the inner verifier ran on is the statement at the sample of every private circuit that has the rows the run read
(`_hx`: `x R ω = stmt q ω` for every such `q`). Then one of five holds. It needs `VBridge` in full, which a protocol
whose last round isn't committed (candidate 3 today) doesn't give.

* `RecursiveSound`'s registrant fork (`regFinder`), round fork (`RoundFork`) or session fork (`SessionFork`).
* A registration break: some outcome `o` that `σ` reaches has all of V*'s sessions accepting, having read at some
  position `i` a row other than the registered `p i`, and its opening of `rt` at `i` and the developer's collide
  (`Binding.Collides ((com i).ext (op o i) (op₀ i))`).
* $$\Pr[\text{every session of V* accepts} \wedge \neg\,\text{the statement of } p \text{ holds}]$$
  is at most `RecursiveSound`'s bound, unchanged.

Every `∃` ranges over `σ`'s own runs and what they output; nothing is assumed of SHA-512.

**A nearly empty event**: its event, as `RecursiveSound`'s, asks that the inner
statement have no witness (`¬ holds`, with `holds` = `∃ z, Satisfies z`). The inner statement pins each row only by its
hm96 commit string `b ‖ c`, with the salt a free witness wire, so a commitment to a wrong row under a fresh salt also
opens to the right row except with probability at most `4·hidingGap` (`VacuityProbes.Hm96Opens.opens_to_any_row`,
`shift_reachable`, probes outside this tree). For any cheater the event is then nearly empty, and the bound bounds
it, not a wrong computation. The fix is the knowledge form the claims API states recursion in;
`EndToEndHidden` avoids this by extraction. The knowledge form, `Claims.Recursion.KS`, is at the recursion's level
`Claims.Recursion.lam`, which no theorem evaluates at the deployment. Route 2C's bound λ₂ (`lamLin2`) evaluates to at
least 135.7 at the deployed records (`RecDeploy.lamLin2_deployed_ge_dep_tight`), and `KSLin2` at the deployment's dense
statements (`KSLin2On` at `RecDeploy.DenseAt`) is proved (`Flock.SecurityProofs.RecursiveKS`), Holo statements being
candidate 3's own model, and `BitsLin2` at those statements follows from it (`bitsLin2On_of_ksLin2On`). This fork form
states no bits. -/
def RecursiveRanRegistered : Prop :=
  ∀ {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).samplingDesign.Lb (Vs j).samplingDesign.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    {εin : ℝ≥0∞} (_hin : InnerSound I εin)
    -- the registration's commitment at each row position, the registered private circuit, and the developer's openings
    {Root Pos Row Op : Type} (com : Pos → PrivateCircuit.Commit Root Row Op) (_hbind : ∀ i, (com i).Binding)
    (rt : Root) (p : Pos → Row) (op₀ : Pos → Op) (_hreg : ∀ i, (com i).Opens rt (p i) (op₀ i))
    -- a private circuit's statement at a sample; the rows an accepted run read, and the rows and openings of `rt` it carries
    (stmt : (Pos → Row) → L.Ω → I.Stmt)
    (reads : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Pos → Prop)
    (row : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Pos → Row)
    (op : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Pos → Op)
    (_hop : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ i, reads o i → (com i).Opens rt (row o i) (op o i))
    (_hx : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ q : Pos → Row, (∀ i, reads o i → q i = row o i) →
      x o.1 o.2.1 = stmt q o.2.1),
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      (∃ o, FlockSoundness.Game.Outcome (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
        AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, reads o i ∧ row o i ≠ p i ∧ FlockSoundness.Binding.Collides ((com i).ext (op o i) (op₀ i))) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (stmt p o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin

/-- **`RecursiveRanRegistered` with no A3**: `RecursiveRanRegistered` without its A3 premise, at the same bound. V*'s
sampling designs are the full sample (`--draw subset:<units>`), which samples every unit at every value of the operating system's bytes
(`execOS_subset_full_mem`), so no session's miss reads those bytes. -/
def RecursiveRanRegisteredNoA3 : Prop :=
  ∀ {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    {εin : ℝ≥0∞} (_hin : InnerSound I εin)
    -- the registration's commitment at each row position, the registered private circuit, and the developer's openings
    {Root Pos Row Op : Type} (com : Pos → PrivateCircuit.Commit Root Row Op) (_hbind : ∀ i, (com i).Binding)
    (rt : Root) (p : Pos → Row) (op₀ : Pos → Op) (_hreg : ∀ i, (com i).Opens rt (p i) (op₀ i))
    -- a private circuit's statement at a sample; the rows an accepted run read, and the rows and openings of `rt` it carries
    (stmt : (Pos → Row) → L.Ω → I.Stmt)
    (reads : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Pos → Prop)
    (row : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Pos → Row)
    (op : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Pos → Op)
    (_hop : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ i, reads o i → (com i).Opens rt (row o i) (op o i))
    (_hx : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ q : Pos → Row, (∀ i, reads o i → q i = row o i) →
      x o.1 o.2.1 = stmt q o.2.1),
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      (∃ o, FlockSoundness.Game.Outcome (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
        AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, reads o i ∧ row o i ≠ p i ∧ FlockSoundness.Binding.Collides ((com i).ext (op o i) (op₀ i))) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (stmt p o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin

end Flock.Guarantees
