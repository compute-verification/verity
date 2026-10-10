import Proofs.Flock.Soundness.Assumptions.Recursive
import Proofs.Flock.Soundness.Assumptions.Zk
import Proofs.Flock.Soundness.Discharge.ZkView.Defs

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

* `RecursiveSound`, for the auditor: when the auditor accepts every one of V*'s sessions, the hidden circuit's statement
  holds, except with probability at most the sum of V*'s sessions' soundness errors, plus each round's transfer slack
  and rewinding loss (the binding of the firewall's commitments under `cr/sha-512`), plus the inner protocol's soundness
  error. Its one new assumption is `Flock.Assumptions.VBridge`.
* `RecursiveZK`, for the developer: the auditor's whole view, the firewall's commitments (`firewallLeaf`) to every inner
  message and tree top and then its view of the outer session, has a simulator that reads public inputs only, within
  the commitments' hiding gap (`hash-derived-key`) plus the outer session's zero-knowledge gap.

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
recursive audit, given V*'s bridge (`Flock.Assumptions.VBridge`), `cr/sha-512` for the fork finders of each round
(`hCR`) and for the registrant records' finder (`regFinder`, at `qX`), and the inner protocol's soundness at `ε_in`:
$$\Pr[\text{every session of V* accepts} \wedge \neg\,\text{the hidden statement holds}] \le
  \mathbb E\Big[\sum_j \text{outer bound}_j\Big] + q_X^2/2^{513} + \sum_{i<r}\Big(\mathbb E\Big[\sum_j \text{stage slack}_j\Big] +
  \big(\textstyle\sum_{pos}\kappa_i(pos)\big)\sqrt{q_i^2/2^{513}}\Big) + \varepsilon_{in},$$
the expectations over the draw, the rounds' coins at the prover's leaves (`expAt`) and the sessions (`sumV`), each
term at the registration `R₂` the prover sends and the strategy it names for its session. A session's outer bound
(`ZkOuter.boundCR`) is, where its
finders satisfy `cr/sha-512` at budgets `qF, qS, qW, qR` (`ZkOuter.CR`), its law's miss at one bad unit plus
`zk_session_soundR`'s and `zk_session_regValsR`'s terms at the drawn units and `qR²/2^513` for its registered paths
(`ZkOuter.bound`), and `1` elsewhere; its stage slack (`ZkOuter.slackCR`) is likewise `ZkOuter.stageSlack` or `1`. The
weights `κ_i = weightS` and `√(q²/2^513)` are the binding of round `i`'s commitments (`stage_bind`); `qX²/2^513`, once
per audit, prices two sessions' registrant records clashing (`RegColl`, `prob_regColl_le`), off which V*'s bridge
reads. -/
def RecursiveSound : Prop :=
  ∀ {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    -- V*'s laws: their strata and A3
    (_hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration `R₂`
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (_ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, the rounds' commit strings, the firewall's
    -- commitments and the messages they open to, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (_hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    -- cr/sha-512 for the registrant records' finder, and for the fork finders of each round
    (qX : ℝ) (_hX : (regFinder zs dg N σ).CR H512 qX)
    (q : ℕ → ℝ)
    (_hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (_hin : InnerSound I εin),
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ENNReal.ofReal (qX ^ 2 / 2 ^ (513 : ℝ)) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin

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

end Flock.Guarantees
