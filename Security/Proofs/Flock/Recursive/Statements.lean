import Proofs.Flock.Soundness.Assumptions.Recursive
import Proofs.Flock.Soundness.Assumptions.Zk
import Proofs.Flock.Soundness.Discharge.ZkView.Defs
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Defs
import Proofs.Flock.Soundness.ZK.HidingAdaptive

/-!
# The recursive audit's guarantees (trusted)

The recursive audit (`Proofs.Flock.Recursive.Defs`: `recGame`): the developer registers `R` (the hidden circuit's
commitment among it); the auditor draws `ω`, the hidden circuit's units to prove; in each of the inner protocol's `I.r`
rounds the developer's firewall sends the auditor a salted SHA-512 commitment to the round's inner message (C-Flock run
with `--zk` off on the workers), and the auditor answers with a live coin; then the developer registers `R₂`, the
values V*'s sessions read that the rounds do not fix (V*'s running sums `rec-acc`, and with `link_rows` its opened rows),
and C-Flock `--zk --registered` sessions of the public verifier circuits V* (`zs R ω t R₂ j`, `ZkOuter`, one per circuit
`(Vs j).c`, in a row: `outerV`) prove that openings of those commitments exist on which the inner verifier accepts. Each guarantee is a `def G : Prop` in the
form of `Pouw.Guarantees`; its proof is `Flock.SecurityProofs.G : Flock.Guarantees.G`
(`Proofs.Flock.Recursive.Guarantees`).

* `RecursiveSound`, for the auditor, in fork form: either a run of the prover's own strategy finds a SHA-512 collision
  (between two of V*'s sessions' registrant records, at a round's commitment, or in one of V*'s sessions), or the auditor accepts every one of V*'s sessions on a false
  hidden statement with probability at most V*'s sessions' statistical errors, plus each round's transfer slack, plus
  the inner protocol's soundness error. It assumes no hash. Its one new assumption is `Flock.Assumptions.VBridge`.
* `RecursiveZK`, for the developer: the auditor's whole view, the firewall's commitments (`firewallLeaf`) to every inner
  message and tree top and then its view of the outer session, has a simulator that reads public inputs only, within
  the commitments' hiding gap at hm96-sha512's pinned key plus the outer session's zero-knowledge gap. It assumes nothing
  of the key or of SHA-512.
* `RecursiveCircuitPrivate`, for the developer of a hidden circuit: the private circuit is a witness value of the class's
  universal unit `UniversalUnit_v1{G, N_IN, N_OUT}`, and the auditor's whole view, the private circuit's registration, the
  firewall's commitments and V*'s sessions, has a simulator that reads only the class and the owed outputs, within the
  registration's hiding gap plus `RecursiveZK`'s. `RecursiveCircuitIndist` is its two-circuit form, and
  `RecursiveCircuitPrivateLeaves` the registration as hm96 leaves of the private circuit's rows, its hiding gap hm96's at
  the pinned key. They assume nothing of the key or of SHA-512.
* `RecursiveRanRegistered`, for the auditor, in `RecursiveSound`'s fork form: on its hypotheses and a registration
  binding at each row of the private circuit, a round or session fork, or a run the prover reaches whose sessions all accept
  having read a row other than the registered one and whose opening of it collides with the developer's, or the auditor
  accepts every session while the registered private circuit's hidden statement fails with probability at most
  `RecursiveSound`'s bound. `RanRegisteredRun` is the one-session form, `PrivateCircuit.RanRegistered` with its
  `cr/sha-512` hypothesis replaced by the same kind of break.
* `commitBytes`, `firewallPairs`, `firewallCommits`: the firewall's commit strings and commitments when the inner coins
  are live, message `i` reading the commit strings before it, which (ii) of `RecursiveAudit` at the firewall reads
  (`Proofs.Flock.Recursive.Firewall`). `RecursiveZK` and the circuit-privacy forms above are the committed-coin case,
  each message fixed by the auditor's tape before any commitment.

C-Flock's definitions and the recursion's model are still in the proofs package, so `Models.lean` does not import this
module: `Proofs.Flock.Recursive` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The recursive audit is sound.** V*'s sessions are of the circuits `(Vs j).c`, with hidden outputs and `(Vs j).J`
tables, under their laws `(Vs j).law`, each the full draw (`--draw subset:<units>`) on the operating system's bytes (A3,
`uniform/io-getrandombytes`); at each history `(R, ω, t)` and registration `R₂` of the outer proof, session `j` is
`zs R ω t R₂ j`, whose fields carry `flock-verify verify --zk --registered`'s setup, the `live-verifier` record custody
(`ZkOuter.hRec`) and the registrant's record of the registered reads (`ZkOuter.rd`: the hidden circuit's rows, `R₂`'s
values, and the verifier's own). For every prover `σ` of the recursive audit, given V*'s bridge
(`Flock.Assumptions.VBridge`) and the inner protocol's soundness at `ε_in`, one of four holds.

* A registrant fork (`regFinder`): one run of `σ` reaches an outcome at whose history and registration two of V*'s
  sessions' registrant records clash (`RegColl`: two registered reads whose openings or hm96 leaves are a SHA-512
  collision), and outputs that collision. V*'s bridge reads off such outcomes (`VBridgeOff`).
* A round fork (`RoundFork`): for some round `i < r`, draw `ω` and commit string `pos`, round `i`'s fork finder
  (`forkAt`: two continuations of `σ` after its round-`i` commitment, each running the trial for `pos`) outputs a
  SHA-512 collision at an outcome it reaches.
* A session fork (`SessionFork`): at some leaf `σ` reaches, some outcome of V*'s sessions its strategy reaches and some
  session `j`, a finder of that session run from the strategy `σ` names there (its tables' knowledge finders, its
  registered-value finders, the run that opens a drawn row's registered leaf, or its link finder: `ZkOuter.Fork`)
  outputs a SHA-512 collision at an outcome it reaches.
* $$\Pr[\text{every session of V* accepts} \wedge \neg\,\text{the hidden statement holds}] \le
    \mathbb E\Big[\sum_j \text{outer bound}_j\Big] + r\cdot\mathbb E\Big[\sum_j \text{stage slack}_j\Big] +
    \varepsilon_{in},$$
  the expectations over the draw, the rounds' coins at the prover's leaves (`expAt`) and the sessions (`sumV`), each
  term at the registration `R₂` the prover sends and the strategy it names for its session. A session's outer bound
  (`ZkOuter.boundCR`) is its law's miss at one bad unit plus the drawn units' statistical errors: the tables'
  (`tableErrZ`: each drawn table's `bound` over `1 − N₀/(ek)`), the link's (`linkErrZC`: `Q_s/(1 − ρ)·(1/(eM) +
  k/(eR_w))`) and the waits' (`waitErrZC`: `Q_s/(1 − ρ)·k/(eR_w)`); its stage slack (`ZkOuter.slackCR`) is the same
  without the miss.

Every `∃` in a fork ranges over `σ`'s own runs and what they output, never over arbitrary strings. The bound is exact
for a prover that never collides on a fork, and says nothing for one that collides on a single pair: no term counts
SHA-512 evaluations. -/
def RecursiveSound : Prop :=
  ∀ {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    -- V*'s laws: A3
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration `R₂`
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, the rounds' commit strings, the firewall's
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
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin

/-- **The firewall's commitment** to one message `msg` (an inner round message or a Merkle tree top) under the salt `y`:
the hm96-sha512 leaf of the message's SHA-512 digest, `H(SHA-512(msg) ⊕ M·y, c(y))` (`ZkView.leafH`, `ZkView.My`,
`ZkView.saltHash` at the pinned key; `ZkView.leafSplit` states that this is the executable's `Hm96.Default512.leaf`). An
honest firewall sends the auditor this, under a fresh uniform salt, for every message, and nothing else of them. The salt
is 192 bytes from the operating system, uniform under A3 (`uniform/io-getrandombytes`); at those bytes `b` the
executable's leaf is `firewallLeaf msg (ZkView.saltOfBytes b)` (`Flock.SecurityProofs.firewallLeaf_os`). -/
def firewallLeaf (msg : ByteArray) (y : ZkView.Salt) : ByteArray :=
  ZkView.leafH (ZkView.bitsOf (Sha512.hash msg) + ZkView.My y, ZkView.saltHash y)

/-- **An ideal commitment**: the leaf of a uniform `u` in place of `SHA-512(msg) ⊕ M·y`, beside the salt's own hash; it
reads no message. -/
def idealLeaf (u : ZkView.Bits × ZkView.Salt) : ByteArray :=
  ZkView.leafH (u.1, ZkView.saltHash u.2)

/-- **An hm96-sha512 commit string from its two halves**, `b ‖ c`: the 128 bytes a firewall leaf hashes
(`ZkView.leafH` is `Default512.treeLeaf` of them). -/
def commitBytes (u : ZkView.Bits × ByteArray) : ByteArray := ZkView.bytesOf (k := 64) u.1 ++ u.2

/-- **The firewall's commit strings under live inner coins**, as their two halves: string `i` is
`(SHA-512(m_i) ⊕ M·y_i, c(y_i))`, a `firewallLeaf`'s before its leaf hash, where the message `m_i = msgs i l` reads the
commit strings `l` before it (`commitBytes` of their halves). The auditor has seen those strings when it draws the coins
message `i` answers, so the message may read them: each inner coin is live, drawn after the commitment before it. They
are `ZK.adaptiveLeaves` at the identity encoding: pair `i` is message `i`'s digest at the pairs before it, plus
`(M·y_i, c(y_i))`. -/
def firewallPairs {nh : ℕ} (msgs : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (y : Fin nh → ZkView.Salt) :
    Fin nh → ZkView.Bits × ByteArray :=
  FlockSoundness.ZK.adaptiveLeaves (id : ZkView.Bits × ByteArray → ZkView.Bits × ByteArray)
    (fun i l => ZkView.bitsOf (Flock.Sha512.hash (msgs i fun j => commitBytes (l j))))
    (fun i => (ZkView.My (y i), ZkView.saltHash (y i)))

/-- **The firewall's commitments under live inner coins**: commitment `i` is the hm96-sha512 leaf of message `i`'s
SHA-512 under its own fresh salt `y i`, `firewallLeaf (msgs i l) (y i)`, where message `i` reads the commit strings `l`
before it (`firewallPairs`). A message that reads none of them is the committed-coin case,
`firewallLeaf (msgs i) (y i)`. -/
def firewallCommits {nh : ℕ} (msgs : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (y : Fin nh → ZkView.Salt)
    (i : Fin nh) : ByteArray :=
  ZkView.leafH (firewallPairs msgs y i)

/-- **The recursive audit is zero knowledge.** The auditor's tape is `Ωa` and the developer's randomness `Ωp`. The
firewall sends the `nh` commitments `firewallLeaf (msgs a p i) (y i)`, to the inner messages and tree tops `msgs a p`, each
under its own fresh uniform salt `y i`; the inner coins the messages answer are the auditor's, fixed by its tape before
the first commitment (Goldreich–Kahan's committed coins, or a beacon's). Then the outer session runs at the public input
`h` (the commitments), with the inner proof and the salts as its witness: `view a h p y r` is the auditor's view of it on
the developer's randomness `r`, against the auditor of tape `a`, which may read `h`. When the outer session has a
simulator `sim a h` (reading the auditor and the public input only) within `ε_o` of `view` at every tape, witness and
event, the auditor's whole view (its tape, the commitments, the outer view) is within
$$\varepsilon_o + n_h \cdot \delta_{K^*}$$
of the simulator that draws `nh` ideal commitments (`idealLeaf`) and runs `sim` on them: it reads the auditor and public
inputs only, nothing of the inner proof. One-sided at every event, so both ways (an event's complement).

`δ_{K*}` is `hidingGap My saltHash`, the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`,
`ZkView.saltHash`): the statistical distance of `(M(K*)·y, H(sp ‖ y))` from `(U, H(sp ‖ y))` for a uniform salt `y`.
Nothing here proves it small for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and
that `K*` is not one of the others is unproved. No premise is taken on the key or on SHA-512.

The salts are drawn uniformly here; the firewall reads them from the operating system, and under A3
(`uniform/io-getrandombytes`) the same bound holds at those bytes (`Flock.SecurityProofs.recursiveZK_os_gap`), so no
premise on a stream generator enters. -/
def RecursiveZK : Prop :=
  ∀ {Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo]
    [Fintype Ωs] [Nonempty Ωs] (nh : ℕ) (msgs : Ωa → Ωp → Fin nh → ByteArray)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞},
    (∀ a p y (E : V → Prop),
      prCoin (fun r : Ωo => E (view a (fun i => firewallLeaf (msgs a p i) (y i)) p y r))
        ≤ prCoin (fun s : Ωs => E (sim a (fun i => firewallLeaf (msgs a p i) (y i)) s)) + εo) →
    ∀ E : Ωa × (Fin nh → ByteArray) × V → Prop,
      prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
          E (w.1.1.1, (fun i => firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)),
            view w.1.1.1 (fun i => firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)) w.1.1.2 w.1.2 w.2))
        ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
            E (w.1.1, (fun i => idealLeaf (w.1.2 i)), sim w.1.1 (fun i => idealLeaf (w.1.2 i)) w.2))
          + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)

/-- **The recursive audit's real view of a private circuit**, the probability of the event `E`. The developer registers
the private circuit `P` under the registration's randomness `r`, and the auditor sees `reg P r`; the firewall sends the `nh`
commitments `firewallLeaf (msgs P w r a p i) (y i)` to the inner messages and tree tops, each under its own fresh
uniform salt `y i`; and V*'s sessions give the auditor `view P w r a h p y q` at those commitments `h`. The average is over
`r`, the auditor's tape `a`, the developer's randomness `p`, the salts `y` and V*'s sessions' randomness `q`, uniform and
independent; `E` reads the registration's view, the tape, the commitments and V*'s sessions' view. -/
noncomputable def recRealPr {Prog Wit Rr Rv Ωa Ωp Ωo V : Type} [Fintype Rr] [Fintype Ωa] [Fintype Ωp] [Fintype Ωo]
    (reg : Prog → Rr → Rv) (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (P : Prog) (w : Wit) (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) : ℝ≥0∞ :=
  avg fun r : Rr => prCoin fun z : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
    E (reg P r, z.1.1.1, (fun i => firewallLeaf (msgs P w r z.1.1.1 z.1.1.2 i) (z.1.2 i)),
      view P w r z.1.1.1 (fun i => firewallLeaf (msgs P w r z.1.1.1 z.1.1.2 i) (z.1.2 i)) z.1.1.2 z.1.2 z.2)

/-- **The simulated recursive view**, the probability of the event `E`: the simulator registers the fixed private circuit `p₀`
of the class (`reg p₀ r` at a fresh `r`), draws `nh` ideal commitments (`idealLeaf`, which read no message), and runs
V*'s sessions' simulator `sim o v a h` at the owed outputs `o`, the registration's view `v`, the auditor's tape `a` and
those commitments `h`. It reads no private circuit but `p₀`, no witness, and none of the developer's randomness or salts. -/
noncomputable def recSimPr {Out Prog Rr Rv Ωa Ωs V : Type} [Fintype Rr] [Fintype Ωa] [Fintype Ωs]
    (reg : Prog → Rr → Rv) (nh : ℕ) (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) (o : Out) (p₀ : Prog)
    (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) : ℝ≥0∞ :=
  avg fun r : Rr => prCoin fun z : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
    E (reg p₀ r, z.1.1, (fun i => idealLeaf (z.1.2 i)), sim o (reg p₀ r) z.1.1 (fun i => idealLeaf (z.1.2 i)) z.2)

/-- **The recursive audit is circuit private.** Everything here is at one size class of the universal unit
`UniversalUnit_v1{G, N_IN, N_OUT}`, the circuit V* and the inner verifier run, public: `Prog` is the class's private circuits,
each a witness value of it; `reg` is the class's registration of a private circuit (`Rr` its randomness, `Rv` what the auditor
sees of it); `nh` is the number of the firewall's commitments; and `msgs` and `view` are the inner prover's messages and
the auditor's view of V*'s `--zk` sessions (`recRealPr`), functions of the private circuit `P`, the developer's witness `w` (its
data), the registration's randomness, the auditor's tape (the inner coins are on it, fixed before the first
commitment, as in `RecursiveZK`) and the developer's randomness. `honest o P w` says that `w` is an honest witness on
which `P` computes the owed outputs `o`.

The view is the auditor's whole view: the registration's, the firewall's commitments to every inner message and tree
top (`firewallLeaf`), and V*'s sessions'. Assumed:
* the registration hides the private circuit up to `ε` (`PrivateCircuit.Hides reg ε`), for every pair of the class's private circuits
  and every event (`RecursiveCircuitPrivateLeaves` discharges it for a registration by hm96 leaves);
* V*'s sessions are zero knowledge within `ε_o` at every honest witness, registration randomness, tape and salts, with a
  simulator `sim o v a h` that reads the owed outputs, the registration's view, the auditor's tape and the commitments
  (V*'s public inputs), and nothing of the private circuit, the witness, the salts or the developer's randomness (`_hOuter`).

Then for every honest witness `w` of every private circuit `P` of the class, and every private circuit `p₀` of the class, the real view
and the simulated one (`recSimPr`: `p₀` registered, ideal commitments, `sim`), which reads only the class (`p₀` and the
class's functions) and the owed outputs `o`, give every event probabilities within
$$\varepsilon + \varepsilon_o + n_h \cdot \delta_{K^*}$$
of each other, both ways.

The audits it speaks of are those whose inner session's circuits are private: the verifier's own `--hide` sets the inner
shape's `RecVStar.Inner.holo`, and every `VERDICT` line verified with `--rec-inner` carries it as `"private_circuit"`. A
verdict with `"private_circuit": false` claims no circuit privacy. `RecVStar.statements` refuses a holo inner shape
until V*'s circuit writer writes holo parts (`statements_holo`), so no audit the verifier accepts says true yet.

`δ_{K*}` is `hidingGap My saltHash`, the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`,
`ZkView.saltHash`): the statistical distance of `(M(K*)·y, H(sp ‖ y))` from `(U, H(sp ‖ y))` for a uniform salt `y`.
Nothing here proves it small for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and
that `K*` is not one of the others is unproved. No premise is taken on the key or on SHA-512. -/
def RecursiveCircuitPrivate : Prop :=
  ∀ {Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Rr] [Nonempty Rr] [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp]
    [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    -- the class's registration of a private circuit, hiding up to `ε`
    (reg : Prog → Rr → Rv) {ε : ℝ≥0∞} (_hreg : PrivateCircuit.Hides reg ε)
    -- the honest witnesses: `P` computes the owed outputs `o` on `w`
    (honest : Out → Prog → Wit → Prop)
    -- the firewall's messages, the auditor's view of V*'s sessions, and its simulator within `ε_o`
    (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (_hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (fun i => firewallLeaf (msgs P w r a p i) (y i)) p y q))
        ≤ prCoin (fun s : Ωs => E (sim o (reg P r) a (fun i => firewallLeaf (msgs P w r a p i) (y i)) s)) + εo)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit), honest o P w →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr reg nh msgs view P w E ≤ recSimPr reg nh sim o p₀ E +
          (ε + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) ∧
        recSimPr reg nh sim o p₀ E ≤ recRealPr reg nh msgs view P w E +
          (ε + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))

/-- **Two private circuits of a class are indistinguishable in the recursive audit.** On `RecursiveCircuitPrivate`'s
hypotheses, for honest witnesses `w` and `w'` of private circuits `P` and `P'` of the class that owe the same outputs `o`, the
real views (`recRealPr`) give every event probabilities within
$$2\,(\varepsilon + \varepsilon_o + n_h \cdot \delta_{K^*});$$
exchanging `P` and `P'` gives the other way.

`δ_{K*}` is `hidingGap My saltHash`, the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`,
`ZkView.saltHash`): the statistical distance of `(M(K*)·y, H(sp ‖ y))` from `(U, H(sp ‖ y))` for a uniform salt `y`.
Nothing here proves it small for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and
that `K*` is not one of the others is unproved. No premise is taken on the key or on SHA-512. -/
def RecursiveCircuitIndist : Prop :=
  ∀ {Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Rr] [Nonempty Rr] [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp]
    [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    (reg : Prog → Rr → Rv) {ε : ℝ≥0∞} (_hreg : PrivateCircuit.Hides reg ε)
    (honest : Out → Prog → Wit → Prop)
    (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (_hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (fun i => firewallLeaf (msgs P w r a p i) (y i)) p y q))
        ≤ prCoin (fun s : Ωs => E (sim o (reg P r) a (fun i => firewallLeaf (msgs P w r a p i) (y i)) s)) + εo)
    (o : Out) (P P' : Prog) (w w' : Wit), honest o P w → honest o P' w' →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr reg nh msgs view P w E ≤ recRealPr reg nh msgs view P' w' E +
        2 * (ε + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))

/-- **A registration by hm96 leaves**: row `i` of the private circuit `P` (`rows P i`) committed as the firewall commits a
message, `firewallLeaf (rows P i) (r i)` under its own fresh uniform salt `r i`; the auditor sees `pub` of the leaves
(their Merkle root, and whatever else the registration record publishes of them) and nothing else of the rows or salts.
The registrant's opening of a `--zk --registered` session's registered read is such a row and salt (`ZkReg.regOpenZ`). -/
def regLeaves {Prog Rv : Type} {nr : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (P : Prog)
    (r : Fin nr → ZkView.Salt) : Rv :=
  pub fun i => firewallLeaf (rows P i) (r i)

/-- **The recursive audit is circuit private, the private circuit registered by hm96 leaves.** `RecursiveCircuitPrivate` with
the registration `regLeaves rows pub` (the private circuit's `nr` rows, each under its own salt, the auditor seeing `pub` of the
leaves): its hiding is hm96's at the pinned key twice per row, `ε = 2·n_r·δ_{K*}`, so the one hypothesis left is V*'s
sessions' zero knowledge (`_hOuter`). The salts of the registration may be read by the inner messages and V*'s sessions
(the registered reads open them), never by the simulator. For every honest witness `w` of every private circuit `P` of the
class and every private circuit `p₀` of it, the real view and the simulated one give every event probabilities within
$$2 n_r \cdot \delta_{K^*} + \varepsilon_o + n_h \cdot \delta_{K^*}$$
both ways.

`δ_{K*}` is `hidingGap My saltHash`, the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`,
`ZkView.saltHash`): the statistical distance of `(M(K*)·y, H(sp ‖ y))` from `(U, H(sp ‖ y))` for a uniform salt `y`.
Nothing here proves it small for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and
that `K*` is not one of the others is unproved. No premise is taken on the key or on SHA-512. -/
def RecursiveCircuitPrivateLeaves : Prop :=
  ∀ {Out Prog Wit Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo]
    [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    -- the registration: the private circuit's rows, and what the auditor sees of their leaves
    (nr : ℕ) (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop)
    (nh : ℕ) (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (_hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (fun i => firewallLeaf (msgs P w r a p i) (y i)) p y q))
        ≤ prCoin (fun s : Ωs =>
            E (sim o (regLeaves rows pub P r) a (fun i => firewallLeaf (msgs P w r a p i) (y i)) s)) + εo)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit), honest o P w →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr (regLeaves rows pub) nh msgs view P w E ≤ recSimPr (regLeaves rows pub) nh sim o p₀ E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) + εo +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) ∧
        recSimPr (regLeaves rows pub) nh sim o p₀ E ≤ recRealPr (regLeaves rows pub) nh msgs view P w E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) + εo +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))

/-- **A session ran the registered program**, with its break tied to the run: `PrivateCircuit.RanRegistered` without its
`cr/sha-512` hypothesis. The root `rt` commits to a registered program `p` (for a private circuit, `pad C c`) through
the registrant's opening `op₀`, and the commitment binds (`PrivateCircuit.Commit.Binding`: two openings of one root to
two different programs give the extractor's pair `com.ext`, two different strings SHA-512 maps to one digest). Take a
session, a game `G` against any prover `σ`, whose acceptance `acc` shows that the program of its committed values
(`prog b`) opens `rt` (through `op b`), and suppose protocol 1's soundness bounds accepting with its bad event `Q` (at
least `K` wrong units of the committed values, say) by `B`. Then either some outcome `σ` reaches accepts a program other
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
`σ`, V*'s laws and sessions, the bridge `Flock.Assumptions.VBridge` and `ε_in`), let the registration commit the
private circuit `p`, row `p i` at each position `i`, at the root `rt`; `com i` is the commitment at position `i`, binding
(`PrivateCircuit.Commit.Binding`: two openings of `rt` at `i` to two different rows give the extractor's pair `com.ext`,
two different strings SHA-512 maps to one digest), and the developer's opening of row `i` is `op₀ i`. V*'s sessions
read some of the rows, never necessarily all of them: on a run `o` whose sessions all accept, `reads o i` says the run
read row `i`, and `row o i` and `op o i` are the row and the opening of `rt` at `i` the run carries (`_hop`; the run's
output holds the strategy the prover names for each session, and with it the registered reads), and the hidden
statement the inner verifier ran on is the statement at the draw of every private circuit that has the rows the run read
(`_hx`: `x R ω = stmt q ω` for every such `q`). Then one of five holds.

* `RecursiveSound`'s registrant fork (`regFinder`), round fork (`RoundFork`) or session fork (`SessionFork`).
* A registration break: some outcome `o` that `σ` reaches has all of V*'s sessions accepting, having read at some
  position `i` a row other than the registered `p i`, and its opening of `rt` at `i` and the developer's collide
  (`Binding.Collides ((com i).ext (op o i) (op₀ i))`).
* $$\Pr[\text{every session of V* accepts} \wedge \neg\,\text{the statement of } p \text{ holds}]$$
  is at most `RecursiveSound`'s bound, unchanged.

Every `∃` ranges over `σ`'s own runs and what they output; nothing is assumed of SHA-512. -/
def RecursiveRanRegistered : Prop :=
  ∀ {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
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
    -- a private circuit's statement at a draw; the rows an accepted run read, and the rows and openings of `rt` it carries
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
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin

end Flock.Guarantees
