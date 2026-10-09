import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.Statements
import Proofs.Flock.Recursive.Premises
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# The recursive audit's property (trusted)

One statement, `RecursiveAudit`, of the recursive audit as the verifier of record runs it: V*'s outer sessions verified
by `flock-verify verify --zk --registered` (each session's verdict `Flock.Zk.verify` and the statement check
`Flock.ProvedScope.check`), the inner protocol C-Flock's compiled table (`flockInnerCL` at `execArith`), and the hidden
netlist registered by hm96 leaves. It is a conjunction:

* (i) soundness in fork form: a round fork, a session fork or an inner fork of the prover's own runs, a run of the
  prover's whose sessions all verify having read a commit string other than the registered one at a row whose opening
  collides with the developer's, or the verifier of record accepts every session while the registered netlist's hidden
  statement fails with probability at most `RecursiveSound`'s bound with the compiled table's error as its inner term;
* (ii) zero knowledge with circuit privacy, `RecursiveCircuitPrivateLeaves`' form: a simulator that reads only the
  class and the owed outputs.

Its premises are the list in `Proofs.Flock.Recursive.Premises`. Its proof (`Flock.SecurityProofs.RecursiveAudit`,
`Proofs.Flock.Recursive.Gaps`) rests on the open lemmas that file and `Compile` name.

* `regBcCommit`: the registration's commitment at a row position as V* checks a registered read, the commit string
  `b ‖ c` that `Flock.Registered.opens` opens; the netlist's rows are what those strings commit to, so a row's hm96
  binding is in what the hidden statement of a list of commit strings means (`stmt`).
* `VerifiesV`: the verifier of record's verdict on every one of V*'s sessions.

C-Flock's definitions and the recursion's model are still in the proofs package, so the statement is here rather than
in `Security/Properties/`.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

open Classical in
/-- **The registration's commitment at row position `i`, as V* checks a registered read**: an opening
`(reg, bc, path)` of the port `reg` to the commit string `bc` is the verifier's `Flock.Registered.opens reg bc i path`.
The extractor returns the two leaves' SHA-512 inputs when the two strings' tree leaves agree, and the registered path's
pair (`Flock.Registered.openPair`) otherwise. -/
noncomputable def regBcCommit (i : ℕ) :
    PrivateCircuit.Commit Flock.Registered.Port ByteArray (Flock.Registered.Port × ByteArray × ByteArray) where
  Opens reg bc op := op.1 = reg ∧ op.2.1 = bc ∧ Flock.Registered.opens reg bc i op.2.2 = true
  ext op op' :=
    if Hm96.Default512.treeLeaf op.2.1 = Hm96.Default512.treeLeaf op'.2.1 then
      ((Hm96.Default512.leafPrefix ++ op.2.1).data.toList, (Hm96.Default512.leafPrefix ++ op'.2.1).data.toList)
    else
      ((Flock.Registered.openPair op.1 i (Hm96.Default512.treeLeaf op.2.1) op.2.2
          (Hm96.Default512.treeLeaf op'.2.1) op'.2.2).1.data.toList,
        (Flock.Registered.openPair op.1 i (Hm96.Default512.treeLeaf op.2.1) op.2.2
          (Hm96.Default512.treeLeaf op'.2.1) op'.2.2).2.data.toList)

/-- **The verifier of record accepts every one of V*'s sessions**: each session's proof (`AcceptsV`: setup, registered
reads, `Flock.Zk.shapeOk`, `Flock.Zk.verify`) and its statement's form (`FailClosed.ScopeZ`: `Flock.ProvedScope.check`),
at the registration the outcome records. -/
def VerifiesV {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsV zs o ∧ ∀ j, FailClosed.ScopeZ (zs o.1 j).I

/-- **C-Flock's compiled inner protocol as the executable runs it**: `flockInnerCL` at the executable's arithmetic
(`execArith`) and its hm96-sha512 Merkle scheme (`Refine.enc512`, `Refine.leaf512` under `Refine.H512`), on the class
`cls`, its regions and the schedule `sch`. -/
noncomputable abbrev innerExec {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Accounting.Schedule) : Inner :=
  flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 cls regs sch

/-- **The recursive audit, as the verifier of record runs it.**

(i) For every prover `σ` of the recursive audit with C-Flock's compiled table as its inner protocol (`innerExec`): V*'s
sessions
`zs R ω t R₂ j` under their laws, the hidden statement `x`, the messages V*'s layers hold and the rounds' commitments
(`wmsg`, `Sk`, `cmt`, `msg`, as in `RecursiveSound`), and the registration's root `rt`. On an outcome `o` whose sessions
all verify, `reads o i` says a session read row `i`, `row o i` is the commit string it read there and `op o i` its
opening of `rt`; `stmt q ω` is the hidden statement at the draw `ω` of the netlist whose rows the commit strings `q`
commit to. Taken: the schedule and layout of the class (`_hsch`, `_hlay`), the strata of V*'s laws (`_hS`), the
statement check accepting V*'s statements (`_hck`), A3 (`_hA3`), every read string opening `rt` at an accepted outcome
(`_hop`), and the run's hidden statement being the read strings' (`_hx`). Then for the registered commit strings `p`
(the developer's openings `op₀`), for every count curve `(k, R_w, M, ρ)` that bounds every session's rate, given V*'s
bridge (VBridge): a round fork, a session fork, an inner fork of `σ`'s own runs whose decoded transcripts play the compiled table to
outcomes that clash; or a run `σ` reaches whose sessions all verify having read at some row a commit string other than
`p i`, whose opening and the developer's collide; or
$$\Pr[\text{every session verifies} \wedge \neg\,\text{the statement of } p \text{ holds}]$$
is at most V*'s sessions' expected statistical errors, `r` times their expected stage slack, and the compiled table's
error.

(ii) `RecursiveCircuitPrivateLeaves`: for the class's netlists registered by hm96 leaves, when V*'s sessions are zero
knowledge within `ε_o` with a simulator reading the owed outputs, the registration's view, the auditor's tape and the
commitments, under `hash-derived-key`, the auditor's whole view and the simulated one (reading only the class and the
owed outputs) give every event probabilities within `2 n_r·2^-193 + ε_o + n_h·2^-193` both ways. -/
def RecursiveAudit : Prop :=
  (∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Inhabited (CT k)]
    {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    -- the class of the compiled table, its regions and schedule
    (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Accounting.Schedule)
    -- V*'s sessions at each history and registration `R₂`
    (zs : Reg → L.Ω → List (Cm × (innerExec T CT cls regs sch).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg (innerExec T CT cls regs sch) Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden statement, the messages V*'s layers hold, the rounds' commit strings and commitments
    (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (innerExec T CT cls regs sch).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (innerExec T CT cls regs sch).Msg)
    -- the registration's root; a netlist's statement by its rows' commit strings; the rows a run read, and the commit
    -- strings and openings of `rt` it carries
    (rt : Flock.Registered.Port) (stmt : (ℕ → ByteArray) → L.Ω → Hidden cls)
    (reads : Reg × L.Ω × List (Cm × (innerExec T CT cls regs sch).Coin) ×
      (Reg₂ × VRec Vs Reg' dg N) → ℕ → Prop)
    (row : Reg × L.Ω × List (Cm × (innerExec T CT cls regs sch).Coin) ×
      (Reg₂ × VRec Vs Reg' dg N) → ℕ → ByteArray)
    (op : Reg × L.Ω × List (Cm × (innerExec T CT cls regs sch).Coin) ×
      (Reg₂ × VRec Vs Reg' dg N) → ℕ → Flock.Registered.Port × ByteArray × ByteArray)
    -- the class's schedule and layout, and V*'s laws' strata
    (_hsch : Accounting.fast100 cls.m = some sch)
    (_hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts)
    (_hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    -- the verifier of record's statement check accepts V*'s statements
    (_hck : ∀ R ω t R₂ j, FailClosed.ScopeZ (zs R ω t R₂ j).I)
    -- A3
    (_hA3 : ∀ j, Flock.Premises.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- at an outcome whose sessions all verify, every read commit string opens `rt`, and the hidden statement is theirs
    (_hop : ∀ o, VerifiesV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ i, reads o i →
      (regBcCommit i).Opens rt (row o i) (op o i))
    (_hx : ∀ o, VerifiesV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ q : ℕ → ByteArray, (∀ i, reads o i → q i = row o i) →
      x o.1 o.2.1 = stmt q o.2.1)
    -- the registered commit strings and the developer's openings
    (p : ℕ → ByteArray) (op₀ : ℕ → Flock.Registered.Port × ByteArray × ByteArray)
    (_hreg : ∀ i, (regBcCommit i).Opens rt (p i) (op₀ i))
    -- the count curve, bounding every session's rate
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    -- V*'s bridge
    (_hbr : Flock.Premises.VBridge zs dg N k Rw x wmsg Sk cmt msg),
    RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg Refine.H512
        (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 cls regs sch) ∨
      (∃ o, FlockSoundness.Game.Outcome (recGame L Reg (innerExec T CT cls regs sch) Cm
          fun R ω t => outerV (zs R ω t) dg N) σ o ∧
        VerifiesV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, reads o i ∧ row o i ≠ p i ∧ FlockSoundness.Binding.Collides ((regBcCommit i).ext (op o i) (op₀ i))) ∨
      prob (fun o => VerifiesV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (innerExec T CT cls regs sch).holds (stmt p o.2.1))
          (recGame L Reg (innerExec T CT cls regs sch) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((innerExec T CT cls regs sch).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError sch ⟨cls.kLog, regs.length, cls.mPts⟩)) ∧
  (∀ {Out Prog Wit Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Inhabited Ωa] [Fintype Ωp] [Inhabited Ωp] [Fintype Ωo]
    [Inhabited Ωo] [Fintype Ωs] [Inhabited Ωs]
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
    (_hKey : Flock.Premises.HashDerivedKeyHm96)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit), honest o P w →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr (regLeaves rows pub) nh msgs view P w E ≤ recSimPr (regLeaves rows pub) nh sim o p₀ E +
          (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
        recSimPr (regLeaves rows pub) nh sim o p₀ E ≤ recRealPr (regLeaves rows pub) nh msgs view P w E +
          (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)))

end Flock.Guarantees
