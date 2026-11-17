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

/-! The recursive audit's guarantees, each a `def G : Prop` whose proof is `Flock.SecurityProofs.G` (`Proofs.Flock.Recursive.Guarantees`) (trusted). -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- Needs `VBridge` in full, which an uncommitted last round doesn't give; no term counts SHA-512 evaluations.
Its event (`¬ holds`) is nearly empty: a row is pinned only by its hm96 commit string, the salt a free witness wire,
so it bounds no wrong computation and states no bits. The lock lists no theorem of it: the recursion's is `RecursiveKS`. -/
def RecursiveSound : Prop :=
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

/-- Obligation (the outer session is zero knowledge, `RecursiveZK`'s `_hOuter`): at every auditor's tape `a`,
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

/-- Obligation (V*'s sessions are zero knowledge, the circuit-privacy forms' `_hOuter`): at every honest witness `w` of
a private circuit `P` owing `o` and every registration randomness `r`, `OuterViewZK` of the messages `msgs P w r` and
the view `view P w r`, with a simulator `sim o (reg P r)` that reads the owed outputs, the registration's view, the
auditor's tape and the commitments, and nothing of the private circuit, the witness, the salts or the developer's
randomness. At V*'s `--zk` sessions run by the honest prover's code it holds within V*'s sessions' error at the pinned
key's hiding gap, on V*'s staging (`SessionsPublic`, `SessionsCoined`, `SessionsHonest`:
`OuterZK.recursiveZK_of_sessionZK`). -/
def OuterSessionsZK {Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Ωo] [Fintype Ωs] (reg : Prog → Rr → Rv)
    (honest : Out → Prog → Wit → Prop) (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) (εo : ℝ≥0∞) : Prop :=
  ∀ o P w, honest o P w → ∀ r, OuterViewZK nh (msgs P w r) (view P w r) (sim o (reg P r)) εo

/-- **The auditor's whole view of the recursive audit** at `u = (((a, p), y), r)`: its tape `a`, the ZK server's
commitments to `msgs a p` under the salts `y` (`zkServerLeaf`), and the outer session's view at them on the
developer's randomness `r`. -/
def recView {Ωa Ωp Ωo V : Type} (nh : ℕ) (msgs : Ωa → Ωp → Fin nh → ByteArray)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (u : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo) : Ωa × (Fin nh → ByteArray) × V :=
  (u.1.1.1, (fun i => zkServerLeaf (msgs u.1.1.1 u.1.1.2 i) (u.1.2 i)),
    view u.1.1.1 (fun i => zkServerLeaf (msgs u.1.1.1 u.1.1.2 i) (u.1.2 i)) u.1.1.2 u.1.2 u.2)

/-- **The simulated view** at `u = ((a, l), s)`: the tape `a`, `nh` ideal commitments (`idealLeaf`) of `l`, and `sim`
on them with its coins `s`. -/
def recIdeal {Ωa Ωs V : Type} (nh : ℕ) (sim : Ωa → (Fin nh → ByteArray) → Ωs → V)
    (u : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) : Ωa × (Fin nh → ByteArray) × V :=
  (u.1.1, (fun i => idealLeaf (u.1.2 i)), sim u.1.1 (fun i => idealLeaf (u.1.2 i)) u.2)

/-- **The recursive audit is zero knowledge.** The auditor's tape is `Ωa` and the developer's randomness `Ωp`. The
ZK server sends the `nh` commitments `zkServerLeaf (msgs a p i) (y i)`, to the inner messages and tree tops `msgs a p`, each
under its own fresh uniform salt `y i`; the inner coins the messages answer are the auditor's, fixed by its tape before
the first commitment (Goldreich–Kahan's committed coins, or a beacon's). Then the outer session runs at the public input
`h` (the commitments), with the inner proof and the salts as its witness: `view a h p y r` is the auditor's view of it on
the developer's randomness `r`, against the auditor of tape `a`, which may read `h`. When the outer session has a
simulator `sim a h` (reading the auditor and the public input only) within `ε_o` of `view` at every tape, witness and
event (`OuterViewZK`), the auditor's whole view (its tape, the commitments, the outer view: `recView`) is within
$$\varepsilon_o + n_h \cdot \delta_{K^*}$$
of the simulator that samples `nh` ideal commitments (`idealLeaf`) and runs `sim` on them (`recIdeal`): it reads the
auditor and public inputs only, nothing of the inner proof. One-sided at every event, so both ways (an event's
complement).

**Where the audit stops.** What the auditor is released is the whole view cut at a stop: `cut s`, for a stop `s` of a
public family `St`, reads only the stop and the whole view. The audit stops at `stop u`, which may depend on anything:
the coins, any further coins `Ωk`, and the witness. Every event of the released view
`cut (stop u) (recView nh msgs view u.1)` is at most `|St|` times as likely as at the simulator's view cut at a uniform
stop, plus the same `ε_o + n_h·δ_{K*}`, added once: the stop leaks at most `log₂ |St|` bits. At `St = Ωk = Unit` and
`cut` the identity, this is the whole view's bound above (`Flock.SecurityProofs.recursiveZK_whole`).

`δ_{K*}` is `hidingGap My saltHash`, the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`,
`ZkView.saltHash`): the statistical distance of `(M(K*)·y, H(sp ‖ y))` from `(U, H(sp ‖ y))` for a uniform salt `y`.
Nothing here proves it small for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and
that `K*` is not one of the others is unproved. No premise is taken on the key or on SHA-512.

The salts are sampled uniformly here; the ZK server reads them from the operating system, and under A3 per test
(`UniformRandomBytesTest`, `uniform/io-getrandombytes`) the same bound plus the source's slack `η` on the event's test
holds at those bytes (`Flock.SecurityProofs.recursiveZK_os_gap`), so no premise on a stream generator enters. -/
def RecursiveZK : Prop :=
  ∀ {Ωa Ωp Ωo Ωs V St Ωk W : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo]
    [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] [Fintype St] [Nonempty St] [Fintype Ωk] [Nonempty Ωk]
    (nh : ℕ) (msgs : Ωa → Ωp → Fin nh → ByteArray)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞},
    OuterViewZK nh msgs view sim εo →
    ∀ (cut : St → Ωa × (Fin nh → ByteArray) × V → W)
      (stop : (((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo) × Ωk → St) (E : W → Prop),
      prCoin (fun u => E (cut (stop u) (recView nh msgs view u.1))) ≤
        (Fintype.card St : ℝ≥0∞) * prCoin (fun u : St × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
            E (cut u.1 (recIdeal nh sim u.2))) +
          εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)

/-- **The recursive audit's real view of a private circuit**, the probability of the event `E`. The developer registers
the private circuit `P` under the registration's randomness `r`, and the auditor sees `reg P r`; the ZK server sends the `nh`
commitments `zkServerLeaf (msgs P w r a p i) (y i)` to the inner messages and tree tops, each under its own fresh
uniform salt `y i`; and V*'s sessions give the auditor `view P w r a h p y q` at those commitments `h`. The average is over
`r`, the auditor's tape `a`, the developer's randomness `p`, the salts `y` and V*'s sessions' randomness `q`, uniform and
independent; `E` reads the registration's view, the tape, the commitments and V*'s sessions' view. -/
noncomputable def recRealPr {Prog Wit Rr Rv Ωa Ωp Ωo V : Type} [Fintype Rr] [Fintype Ωa] [Fintype Ωp] [Fintype Ωo]
    (reg : Prog → Rr → Rv) (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (P : Prog) (w : Wit) (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) : ℝ≥0∞ :=
  avg fun r : Rr => prCoin fun z : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
    E (reg P r, z.1.1.1, (fun i => zkServerLeaf (msgs P w r z.1.1.1 z.1.1.2 i) (z.1.2 i)),
      view P w r z.1.1.1 (fun i => zkServerLeaf (msgs P w r z.1.1.1 z.1.1.2 i) (z.1.2 i)) z.1.1.2 z.1.2 z.2)

/-- **The simulated recursive view**, the probability of the event `E`: the simulator registers the fixed private circuit `p₀`
of the class (`reg p₀ r` at a fresh `r`), samples `nh` ideal commitments (`idealLeaf`, which read no message), and runs
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
sees of it); `nh` is the number of the ZK server's commitments; and `msgs` and `view` are the inner prover's messages and
the auditor's view of V*'s `--zk` sessions (`recRealPr`), functions of the private circuit `P`, the developer's witness `w` (its
data), the registration's randomness, the auditor's tape (the inner coins are on it, fixed before the first
commitment, as in `RecursiveZK`) and the developer's randomness. `honest o P w` says that `w` is an honest witness on
which `P` computes the owed outputs `o`.

The view is the auditor's whole view: the registration's, the ZK server's commitments to every inner message and tree
top (`zkServerLeaf`), and V*'s sessions'. Assumed:
* the registration hides the private circuit up to `ε` (`PrivateCircuit.Hides reg ε`), for every pair of the class's private circuits
  and every event (`RecursiveCircuitPrivateLeaves` discharges it for a registration by hm96 leaves);
* V*'s sessions are zero knowledge within `ε_o` at every honest witness, registration randomness, tape and salts, with a
  simulator `sim o v a h` that reads the owed outputs, the registration's view, the auditor's tape and the commitments
  (V*'s public inputs), and nothing of the private circuit, the witness, the salts or the developer's randomness
  (`_hOuter`, `OuterSessionsZK`).

Then for every honest witness `w` of every private circuit `P` of the class, and every private circuit `p₀` of the class, the real view
and the simulated one (`recSimPr`: `p₀` registered, ideal commitments, `sim`), which reads only the class (`p₀` and the
class's functions) and the owed outputs `o`, give every event probabilities within
$$\varepsilon + \varepsilon_o + n_h \cdot \delta_{K^*}$$
of each other, both ways.

The audits it speaks of are those whose inner session's circuits are private: the verifier's own `--hide` sets the inner
shape's `RecVStar.Inner.holo`, and every `VERDICT` line verified with `--rec-inner` carries it as `"private_circuit"`. A
verdict with `"private_circuit": false` claims no circuit privacy. At a holo inner shape `RecVStar.statements` lists
V*'s holo statements (`FlockC3.VStarList.statements_holo`), whose parts' files carry META `holo`.

`δ_{K*}` is `hidingGap My saltHash`, the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`,
`ZkView.saltHash`): the statistical distance of `(M(K*)·y, H(sp ‖ y))` from `(U, H(sp ‖ y))` for a uniform salt `y`.
Nothing here proves it small for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and
that `K*` is not one of the others is unproved. No premise is taken on the key or on SHA-512. -/
def RecursiveCircuitPrivate : Prop :=
  ∀ {Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Rr] [Nonempty Rr] [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp]
    [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    (reg : Prog → Rr → Rv) {ε : ℝ≥0∞} (_hreg : PrivateCircuit.Hides reg ε)
    (honest : Out → Prog → Wit → Prop)
    (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (_hOuter : OuterSessionsZK reg honest nh msgs view sim εo)
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
    (_hOuter : OuterSessionsZK reg honest nh msgs view sim εo)
    (o : Out) (P P' : Prog) (w w' : Wit), honest o P w → honest o P' w' →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr reg nh msgs view P w E ≤ recRealPr reg nh msgs view P' w' E +
        2 * (ε + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))

/-- **A registration by hm96 leaves**: row `i` of the private circuit `P` (`rows P i`) committed as the ZK server commits a
message, `zkServerLeaf (rows P i) (r i)` under its own fresh uniform salt `r i`; the auditor sees `pub` of the leaves
(their Merkle root, and whatever else the registration record publishes of them) and nothing else of the rows or salts.
The registrant's opening of a `--zk --registered` session's registered read is such a row and salt (`ZkReg.regOpenZ`). -/
def regLeaves {Prog Rv : Type} {nr : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (P : Prog)
    (r : Fin nr → ZkView.Salt) : Rv :=
  pub fun i => zkServerLeaf (rows P i) (r i)

/-- **The recursive audit is circuit private, the private circuit registered by hm96 leaves.** `RecursiveCircuitPrivate` with
the registration `regLeaves rows pub` (the private circuit's `nr` rows, each under its own salt, the auditor seeing `pub` of the
leaves): its hiding is hm96's at the pinned key twice per row, `ε = 2·n_r·δ_{K*}`, so the one hypothesis left is V*'s
sessions' zero knowledge (`_hOuter`, `OuterSessionsZK`). The salts of the registration may be read by the inner messages and V*'s sessions
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
    (nr : ℕ) (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop)
    (nh : ℕ) (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (_hOuter : OuterSessionsZK (regLeaves rows pub) honest nh msgs view sim εo)
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
    {Root Pos Row Op : Type} (com : Pos → PrivateCircuit.Commit Root Row Op) (_hbind : ∀ i, (com i).Binding)
    (rt : Root) (p : Pos → Row) (op₀ : Pos → Op) (_hreg : ∀ i, (com i).Opens rt (p i) (op₀ i))
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
    {Root Pos Row Op : Type} (com : Pos → PrivateCircuit.Commit Root Row Op) (_hbind : ∀ i, (com i).Binding)
    (rt : Root) (p : Pos → Row) (op₀ : Pos → Op) (_hreg : ∀ i, (com i).Opens rt (p i) (op₀ i))
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
