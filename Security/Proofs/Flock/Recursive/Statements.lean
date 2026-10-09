import Proofs.Flock.Soundness.Assumptions.Recursive
import Proofs.Flock.Soundness.Assumptions.Zk
import Proofs.Flock.Soundness.Discharge.ZkView.Defs
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Defs

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
  (at a round's commitment, or in one of V*'s sessions), or the auditor accepts every one of V*'s sessions on a false
  hidden statement with probability at most V*'s sessions' statistical errors, plus each round's transfer slack, plus
  the inner protocol's soundness error. It assumes no hash. Its one new assumption is `Flock.Assumptions.VBridge`.
* `RecursiveZK`, for the developer: the auditor's whole view, the firewall's commitments (`firewallLeaf`) to every inner
  message and tree top and then its view of the outer session, has a simulator that reads public inputs only, within
  the commitments' hiding gap (`hash-derived-key`) plus the outer session's zero-knowledge gap.
* `RecursiveCircuitPrivate`, for the developer of a hidden circuit: the netlist is a witness value of the class's
  universal unit `UniversalUnit_v1{G, N_IN, N_OUT}`, and the auditor's whole view, the netlist's registration, the
  firewall's commitments and V*'s sessions, has a simulator that reads only the class and the owed outputs, within the
  registration's hiding gap plus `RecursiveZK`'s. `RecursiveCircuitIndist` is its two-netlist form, and
  `RecursiveCircuitPrivateLeaves` the registration as hm96 leaves of the netlist's rows, its hiding gap from
  `hash-derived-key`.
* `RecursiveRanRegistered`, for the auditor: on `RecursiveSound`'s hypotheses and a binding program commitment, when
  every one of V*'s sessions accepts, the hidden statement holds of the registered netlist, or the run's opening of the
  registration and the developer's give a SHA-512 collision along their paths, except with `RecursiveSound`'s bound.

C-Flock's definitions and the recursion's model are still in the proofs package, so `Models.lean` does not import this
module: `Proofs.Flock.Recursive` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The recursive audit is sound.** V*'s sessions are of the circuits `(Vs j).c` under their laws `(Vs j).law`, each
the live stratified law on the operating system's bytes (A3, `uniform/os-random`); at each history `(R, ω, t)` and
registration `R₂` of the outer proof, session `j` is `zs R ω t R₂ j`, whose fields carry `flock-verify verify --zk
--registered`'s setup, the `live-verifier` record custody (`ZkOuter.hRec`) and the registrant's record of the registered
reads (`ZkOuter.rd`: the hidden circuit's rows, `R₂`'s values, and the verifier's own). For every prover `σ` of the
recursive audit, given V*'s bridge (`Flock.Assumptions.VBridge`) and the inner protocol's soundness at `ε_in`, one of
three holds.

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
    -- V*'s laws: their strata and A3
    (_hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration `R₂`
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
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
    RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin

/-- **The firewall's commitment** to one message `msg` (an inner round message or a Merkle tree top) under the salt `y`:
the hm96-sha512 leaf of the message's SHA-512 digest, `H(SHA-512(msg) ⊕ M·y, c(y))` (`ZkView.leafH`, `ZkView.My`,
`ZkView.saltHash` at the pinned key; `ZkView.leafSplit` states that this is the executable's `Hm96.Default512.leaf`). An
honest firewall sends the auditor this, under a fresh uniform salt, for every message, and nothing else of them. -/
def firewallLeaf (msg : ByteArray) (y : ZkView.Salt) : ByteArray :=
  ZkView.leafH (ZkView.bitsOf (Sha512.hash msg) + ZkView.My y, ZkView.saltHash y)

/-- **An ideal commitment**: the leaf of a uniform `u` in place of `SHA-512(msg) ⊕ M·y`, beside the salt's own hash; it
reads no message. -/
def idealLeaf (u : ZkView.Bits × ZkView.Salt) : ByteArray :=
  ZkView.leafH (u.1, ZkView.saltHash u.2)

/-- **The recursive audit is zero knowledge.** The auditor's tape is `Ωa` and the developer's randomness `Ωp`. The
firewall sends the `nh` commitments `firewallLeaf (msgs a p i) (y i)`, to the inner messages and tree tops `msgs a p`, each
under its own fresh uniform salt `y i`; the inner coins the messages answer are the auditor's, fixed by its tape before
the first commitment (Goldreich–Kahan's committed coins, or a beacon's). Then the outer session runs at the public input
`h` (the commitments), with the inner proof and the salts as its witness: `view a h p y r` is the auditor's view of it on
the developer's randomness `r`, against the auditor of tape `a`, which may read `h`. When the outer session has a
simulator `sim a h` (reading the auditor and the public input only) within `ε_o` of `view` at every tape, witness and
event, and under `hash-derived-key`, the auditor's whole view (its tape, the commitments, the outer view) is within
$$\varepsilon_o + n_h \cdot 2^{-193}$$
of the simulator that draws `nh` ideal commitments (`idealLeaf`) and runs `sim` on them: it reads the auditor and public
inputs only, nothing of the inner proof. One-sided at every event, so both ways (an event's complement). -/
def RecursiveZK : Prop :=
  ∀ {Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo]
    [Fintype Ωs] [Nonempty Ωs] (nh : ℕ) (msgs : Ωa → Ωp → Fin nh → ByteArray)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞},
    (∀ a p y (E : V → Prop),
      prCoin (fun r : Ωo => E (view a (fun i => firewallLeaf (msgs a p i) (y i)) p y r))
        ≤ prCoin (fun s : Ωs => E (sim a (fun i => firewallLeaf (msgs a p i) (y i)) s)) + εo) →
    FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96 →
    ∀ E : Ωa × (Fin nh → ByteArray) × V → Prop,
      prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
          E (w.1.1.1, (fun i => firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)),
            view w.1.1.1 (fun i => firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)) w.1.1.2 w.1.2 w.2))
        ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
            E (w.1.1, (fun i => idealLeaf (w.1.2 i)), sim w.1.1 (fun i => idealLeaf (w.1.2 i)) w.2))
          + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)

/-- **The recursive audit's real view of a hidden netlist**, the probability of the event `E`. The developer registers
the netlist `P` under the registration's randomness `r`, and the auditor sees `reg P r`; the firewall sends the `nh`
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

/-- **The simulated recursive view**, the probability of the event `E`: the simulator registers the fixed netlist `p₀`
of the class (`reg p₀ r` at a fresh `r`), draws `nh` ideal commitments (`idealLeaf`, which read no message), and runs
V*'s sessions' simulator `sim o v a h` at the owed outputs `o`, the registration's view `v`, the auditor's tape `a` and
those commitments `h`. It reads no netlist but `p₀`, no witness, and none of the developer's randomness or salts. -/
noncomputable def recSimPr {Out Prog Rr Rv Ωa Ωs V : Type} [Fintype Rr] [Fintype Ωa] [Fintype Ωs]
    (reg : Prog → Rr → Rv) (nh : ℕ) (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) (o : Out) (p₀ : Prog)
    (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) : ℝ≥0∞ :=
  avg fun r : Rr => prCoin fun z : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
    E (reg p₀ r, z.1.1, (fun i => idealLeaf (z.1.2 i)), sim o (reg p₀ r) z.1.1 (fun i => idealLeaf (z.1.2 i)) z.2)

/-- **The recursive audit is circuit private.** Everything here is at one size class of the universal unit
`UniversalUnit_v1{G, N_IN, N_OUT}`, the circuit V* and the inner verifier run, public: `Prog` is the class's netlists,
each a witness value of it; `reg` is the class's registration of a netlist (`Rr` its randomness, `Rv` what the auditor
sees of it); `nh` is the number of the firewall's commitments; and `msgs` and `view` are the inner prover's messages and
the auditor's view of V*'s `--zk` sessions (`recRealPr`), functions of the netlist `P`, the developer's witness `w` (its
data), the registration's randomness, the auditor's tape (the inner coins are on it, fixed before the first
commitment, as in `RecursiveZK`) and the developer's randomness. `honest o P w` says that `w` is an honest witness on
which `P` computes the owed outputs `o`.

The view is the auditor's whole view: the registration's, the firewall's commitments to every inner message and tree
top (`firewallLeaf`), and V*'s sessions'. Assumed:
* the registration hides the netlist up to `ε` (`PrivateCircuit.Hides reg ε`), for every pair of the class's netlists
  and every event (`RecursiveCircuitPrivateLeaves` discharges it for a registration by hm96 leaves);
* V*'s sessions are zero knowledge within `ε_o` at every honest witness, registration randomness, tape and salts, with a
  simulator `sim o v a h` that reads the owed outputs, the registration's view, the auditor's tape and the commitments
  (V*'s public inputs), and nothing of the netlist, the witness, the salts or the developer's randomness (`_hOuter`);
* `hash-derived-key` (`Assumptions.Zk.HashDerivedKeyHm96`).

Then for every honest witness `w` of every netlist `P` of the class, and every netlist `p₀` of the class, the real view
and the simulated one (`recSimPr`: `p₀` registered, ideal commitments, `sim`), which reads only the class (`p₀` and the
class's functions) and the owed outputs `o`, give every event probabilities within
$$\varepsilon + \varepsilon_o + n_h \cdot 2^{-193}$$
of each other, both ways. -/
def RecursiveCircuitPrivate : Prop :=
  ∀ {Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Rr] [Nonempty Rr] [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp]
    [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    -- the class's registration of a netlist, hiding up to `ε`
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
    -- hash-derived-key
    (_hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit), honest o P w →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr reg nh msgs view P w E ≤ recSimPr reg nh sim o p₀ E + (ε + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
        recSimPr reg nh sim o p₀ E ≤ recRealPr reg nh msgs view P w E + (ε + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ))

/-- **Two netlists of a class are indistinguishable in the recursive audit.** On `RecursiveCircuitPrivate`'s
hypotheses, for honest witnesses `w` and `w'` of netlists `P` and `P'` of the class that owe the same outputs `o`, the
real views (`recRealPr`) give every event probabilities within
$$2\,(\varepsilon + \varepsilon_o + n_h \cdot 2^{-193});$$
exchanging `P` and `P'` gives the other way. -/
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
    (_hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96)
    (o : Out) (P P' : Prog) (w w' : Wit), honest o P w → honest o P' w' →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr reg nh msgs view P w E ≤
        recRealPr reg nh msgs view P' w' E + 2 * (ε + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ))

/-- **A registration by hm96 leaves**: row `i` of the netlist `P` (`rows P i`) committed as the firewall commits a
message, `firewallLeaf (rows P i) (r i)` under its own fresh uniform salt `r i`; the auditor sees `pub` of the leaves
(their Merkle root, and whatever else the registration record publishes of them) and nothing else of the rows or salts.
The registrant's opening of a `--zk --registered` session's registered read is such a row and salt (`ZkReg.regOpenZ`). -/
def regLeaves {Prog Rv : Type} {nr : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (P : Prog)
    (r : Fin nr → ZkView.Salt) : Rv :=
  pub fun i => firewallLeaf (rows P i) (r i)

/-- **The recursive audit is circuit private, the netlist registered by hm96 leaves.** `RecursiveCircuitPrivate` with
the registration `regLeaves rows pub` (the netlist's `nr` rows, each under its own salt, the auditor seeing `pub` of the
leaves): its hiding is `hash-derived-key` twice per row, `ε = 2·n_r·2^-193`, so the one hypothesis left beside
`hash-derived-key` is V*'s sessions' zero knowledge (`_hOuter`). The salts of the registration may be read by the inner
messages and V*'s sessions (the registered reads open them), never by the simulator. For every honest witness `w` of
every netlist `P` of the class and every netlist `p₀` of it, the real view and the simulated one give every event
probabilities within
$$2 n_r \cdot 2^{-193} + \varepsilon_o + n_h \cdot 2^{-193}$$
both ways. -/
def RecursiveCircuitPrivateLeaves : Prop :=
  ∀ {Out Prog Wit Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo]
    [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    -- the registration: the netlist's rows, and what the auditor sees of their leaves
    (nr : ℕ) (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop)
    (nh : ℕ) (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (_hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (fun i => firewallLeaf (msgs P w r a p i) (y i)) p y q))
        ≤ prCoin (fun s : Ωs =>
            E (sim o (regLeaves rows pub P r) a (fun i => firewallLeaf (msgs P w r a p i) (y i)) s)) + εo)
    (_hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit), honest o P w →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr (regLeaves rows pub) nh msgs view P w E ≤ recSimPr (regLeaves rows pub) nh sim o p₀ E +
          (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
        recSimPr (regLeaves rows pub) nh sim o p₀ E ≤ recRealPr (regLeaves rows pub) nh msgs view P w E +
          (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ))

/-- **The recursive audit ran the registered netlist.** On `RecursiveSound`'s hypotheses (the same prover `σ`, V*'s laws
and sessions, the bridge `Flock.Assumptions.VBridge`, the fork finders' `cr/sha-512` and `ε_in`), let the registration
commit the netlist `p` at the root `rt` of a binding program commitment `com` (`PrivateCircuit.Commit.Binding`: two
openings of one root to different netlists give a SHA-512 collision between inputs hashed along their paths,
`com.ext`), the developer's opening of it being `op₀`. On a run `o` whose sessions all accept, `prog o` and `op o` are a
netlist and an opening of `rt` to it that the run carries (`_hop`; the run's output holds the strategy the prover names
for each session), and the hidden statement the inner verifier ran on is that netlist's at the draw (`_hx`:
`x R ω = stmt (prog o) ω`). Then
$$\Pr[\text{every session of V* accepts} \wedge \neg\,\text{the statement of } p \text{ holds} \wedge
  \neg\,\text{the openings } op(o), op_0 \text{ collide}]$$
is at most `RecursiveSound`'s bound. The break is tied to the run: two inputs that SHA-512 hashes along the run's
opening of `rt` and along the registration's (`Binding.Collides (com.ext (op o) op₀)`); nothing is assumed of SHA-512
for it, and it adds no term. -/
def RecursiveRanRegistered : Prop :=
  ∀ {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (_hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (_ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    (q : ℕ → ℝ)
    (_hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    {εin : ℝ≥0∞} (_hin : InnerSound I εin)
    -- the registration's program commitment, the registered netlist, and the developer's opening of it
    {Root Prog Op : Type} (com : PrivateCircuit.Commit Root Prog Op) (_hbind : com.Binding)
    (rt : Root) (p : Prog) (op₀ : Op) (_hreg : com.Opens rt p op₀)
    -- a netlist's statement at a draw, and the netlist and opening of `rt` an accepted run carries
    (stmt : Prog → L.Ω → I.Stmt)
    (prog : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Prog)
    (op : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Op)
    (_hop : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → com.Opens rt (prog o) (op o))
    (_hx : ∀ o, AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → x o.1 o.2.1 = stmt (prog o) o.2.1),
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (stmt p o.2.1) ∧
        ¬ FlockSoundness.Binding.Collides (com.ext (op o) op₀))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin

end Flock.Guarantees
