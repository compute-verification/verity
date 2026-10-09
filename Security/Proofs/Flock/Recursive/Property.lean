import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.Statements
import Proofs.Flock.Recursive.Premises
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Recursive.Reads
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Flock.RecResiduals

/-!
# The recursive audit's property (trusted)

One statement, `RecursiveAudit`, of the recursive audit as the verifier of record runs it: V*'s outer sessions verified
by `flock-verify verify --zk --registered` (each session's verdict `Flock.Zk.verify` and the statement check
`Flock.ProvedScope.check`), the inner protocol C-Flock's compiled table (`flockInnerCL` at `execArith`), and the
private circuit registered by hm96 leaves. It is a conjunction:

* (i) soundness in fork form, at an inner statement `st` whose class, regions and schedule are `execClassAt st`: a
  round fork, a session fork or an inner fork of the prover's own runs, a run of the prover's whose sessions all verify
  having read a commit string other than the registered one at a row whose opening collides with the developer's, two
  such runs at one registration and draw having read different commit strings at a row whose openings collide, or
  Lean's verifier accepts `st` and the verifier of record every session while the registered private circuit's hidden
  statement fails with probability at most `RecursiveSound`'s bound with the compiled table's error as its inner term;
* (ii) zero knowledge with circuit privacy, `RecursiveCircuitPrivateLeaves`' form: a simulator that reads only the
  class and the owed outputs.

Its premises are the list in `Proofs.Flock.Recursive.Premises`. Its proof (`Flock.SecurityProofs.RecursiveAudit`,
`Proofs.Flock.Recursive.Audit`) rests on the open lemmas that file names.

The class is a total function of the statement (`Recursion.execClassAt`, `Proofs.Flock.Recursive.ExecClass`): when
the executable's checks of the statement's shape pass, `HClass.ofStmt` at no hidden entries (the universal unit's one
public statement), its regions and its `fast100` schedule; the constant `ExecClass.fixed` otherwise. It reads only the
statement's shape (`Recursion.shapeOf`: no region's bytes, which are instance data). An accepted statement
(`Recursion.AcceptsStmt`) passes the checks (`Recursion.classOk_of_accepts`), so the fixed class adds no accepted
case: at a statement Lean's verifier refuses, (i)'s event is empty.

* `regBcCommit`: the registration's commitment at a row position as V* checks a registered read, the commit string
  `b ‖ c` that `Flock.Registered.opens` opens; the private circuit's rows are what those strings commit to, so a row's
  hm96 binding is in what the hidden statement of a list of commit strings means (`stmt`).
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
    [∀ k, Nonempty (CT k)] (cls : HClass) (regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog)))
    (sch : Accounting.Schedule) : Inner :=
  flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 cls regs sch

/-- **The number of V*'s outer statements** over an inner session of a table of `m` variables, `k_log` and `claims`
claims, as V*'s staging lays them out (`rec_vstage.stage`, `VStar.statements`) and the verifier of record groups the
algebra's residuals (`Flock.RecResiduals.groups`, `rec_residuals.parts`): one `RecOpen` per Ligerito level of
`fast100 m`, then one `InnerRepCheck` per part of the rep's algebra, each residual in S's order in the first part that
stays within `Flock.RecResiduals.MAX_SLOTS` sha512x3 slots, `MAX_REGIONS` regions and 16 residuals. Zero at a shape the
verifier refuses, where V* stages nothing. -/
def outerParts (m kLog claims : ℕ) : ℕ :=
  match Flock.RecResiduals.shape? m kLog claims with
  | none => 0
  | some sh =>
    match Flock.RecAlgebra.structureOf sh >>= fun S =>
        Flock.RecResiduals.groups S (Flock.RecResiduals.portSizes sh) with
    | .ok gs => sh.sched.levels.size + gs.size
    | .error _ => 0

/-- **The number of V*'s outer statements over the class `cls` and its regions `regs`**: `outerParts` at the class's
`m` and `k_log`, and the inner session's claims, two of the protocol's and two a region (`Flock.Stmt.extraClaims`). -/
abbrev outerStatements (cls : HClass) (regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog))) : ℕ :=
  outerParts cls.m cls.kLog (2 + 2 * regs.length)

/-- **C-Flock's compiled inner protocol at the inner statement `st`**: `innerExec` at its class, regions and schedule
(`execClassAt st`). -/
noncomputable abbrev innerAt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) : Inner :=
  innerExec T CT (execClassAt st).cls (execClassAt st).regs (execClassAt st).sch

/-- **The number of V*'s outer statements over the inner statement `st`**: `outerStatements` at its class and
regions. -/
noncomputable abbrev outerStatementsAt (st : Flock.Stmt) : ℕ :=
  outerStatements (execClassAt st).cls (execClassAt st).regs

/-- **V*'s sessions, their records' custody given** (`Flock.Premises.RecordCustody`): each session's data and checks
`zs₀ R ω t R₂ j`, with the record (`RecordsLiveZK`) its custody gives. -/
abbrev custodied {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zs₀ : A → B → C → D → (j : Fin S) → (Vs j).OuterD Reg') (h : Flock.Premises.RecordCustody zs₀) :
    A → B → C → D → (j : Fin S) → (Vs j).Outer Reg' :=
  fun R ω t R₂ j => { toZkOuterD := zs₀ R ω t R₂ j, hRec := And.left (h R ω t R₂ j) }

/-- **The recursive audit, as the verifier of record runs it.**

(i) For every inner statement `st`, with its class, regions and schedule `execClassAt st`, and every prover `σ` of the
recursive audit with C-Flock's compiled table there as its inner protocol (`innerAt`): V*'s statements `Vs`, as many
as V*'s staging lays out over the class and its regions (`outerStatementsAt`), each with its model read off its
executable strata (`execStmts Vs`: `VStmt.exec`, which keeps its live law and so its sessions), and V*'s sessions
`zs R ω t R₂ j` under their laws (`custodied zs₀ _hCust`: their data and checks `zs₀`, and their records' custody
`_hCust`), the hidden statement `x`, the messages V*'s layers hold and the rounds' commitments (`wmsg`, `Sk`, `cmt`,
`msg`, as in `RecursiveSound`), and the registration's root `rt`. At an outcome `o`, `readsZ zs rt o i` says a session
whose verifier holds `rt` read row `i` of it, `rowZ zs rt o i` is the commit string it read there and `opZ zs rt o i`
its opening of `rt` (`Proofs.Flock.Recursive.Reads`; at an outcome whose sessions all verify, each opens `rt`:
`zk_reads_open`); `stmt q ω` is the hidden statement at the draw `ω` of the private circuit whose rows the commit
strings `q` commit to. Taken: the statement check accepting V*'s statements (`_hck`), A3 (`_hA3`), and the hidden
statement of a run `σ` reaches whose sessions all verify being the read strings' (`_hx`), unless another such run at its
registration and draw read another commit string at a row both read, the two openings colliding (two distinct SHA-512
preimages with one hash, `(regBcCommit i).ext`). Then for the registered commit strings `p` (the developer's openings
`op₀`), for every count curve `(k, R_w, M, ρ)` that bounds every session's rate, given V*'s
bridge (VBridge): a round fork, a session fork, an inner fork of `σ`'s own runs whose decoded transcripts play the
compiled table to outcomes that clash; or a run `σ` reaches whose sessions all verify having read at some row a commit
string other than `p i`, whose opening and the developer's collide; or two such runs at one registration and draw that
read different commit strings at a row both read, whose openings collide; or
$$\Pr[\text{Lean's verifier accepts } st \wedge \text{every session verifies} \wedge
  \neg\,\text{the statement of } p \text{ holds}]$$
is at most V*'s sessions' expected statistical errors under the models read off their executable strata, `r` times
their expected stage slack, and the compiled table's error.

(ii) `RecursiveCircuitPrivateLeaves`: for the class's private circuits registered by hm96 leaves, when V*'s sessions
are zero knowledge within `ε_o` with a simulator reading the owed outputs, the registration's view, the auditor's tape and the
commitments, under `hash-derived-key`, the auditor's whole view and the simulated one (reading only the class and the
owed outputs) give every event probabilities within `2 n_r·2^-193 + ε_o + n_h·2^-193` both ways. -/
def RecursiveAudit : Prop :=
  (∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Inhabited (CT k)]
    {n : ℕ} {L : Law n} {Reg Cm : Type} {Reg' Reg₂ : Type}
    -- the inner statement; the compiled table's class, regions and schedule are `execClassAt st`
    (st : Flock.Stmt)
    -- V*'s statements over the class, each with its model read off its executable strata (`execStmts`), and V*'s
    -- sessions' data at each history and registration `R₂`
    {Vs : Fin (outerStatementsAt st) → VStmt}
    (zs₀ : Reg → L.Ω → List (Cm × (innerAt T CT st).Coin) → Reg₂ →
      (j : Fin (outerStatementsAt st)) → (execStmts Vs j).OuterD Reg')
    -- the records' custody
    (_hCust : Flock.Premises.RecordCustody zs₀)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg (innerAt T CT st) Cm fun R ω t => outerV (custodied zs₀ _hCust R ω t) dg N))
    -- the hidden statement, the messages V*'s layers hold, the rounds' commit strings and commitments
    (x : Reg → L.Ω → Hidden (execClassAt st).cls)
    (wmsg : ((j : Fin (outerStatementsAt st)) → (execStmts Vs j).Bits) → List (innerAt T CT st).Msg)
    (Sk : ℕ → (j : Fin (outerStatementsAt st)) →
      Finset (PosK (execStmts Vs j).c (execStmts Vs j).hU (execStmts Vs j).nv))
    (cmt : (j : Fin (outerStatementsAt st)) → Cm →
      PosK (execStmts Vs j).c (execStmts Vs j).hU (execStmts Vs j).nv →
      (obK (execStmts Vs j).c (execStmts Vs j).hU (execStmts Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (outerStatementsAt st)) →
      PosK (execStmts Vs j).c (execStmts Vs j).hU (execStmts Vs j).nv → ValK (execStmts Vs j).c) →
      (innerAt T CT st).Msg)
    -- the registration's root, and a private circuit's statement by its rows' commit strings
    (rt : Flock.Registered.Port) (stmt : (ℕ → ByteArray) → L.Ω → Hidden (execClassAt st).cls)
    -- the verifier of record's statement check accepts V*'s statements
    (_hck : ∀ R ω t R₂ j, FailClosed.ScopeZ (custodied zs₀ _hCust R ω t R₂ j).I)
    -- A3
    (_hA3 : Flock.Premises.UniformRandomBytes (execStmts Vs))
    -- at an outcome `σ` reaches whose sessions all verify, the hidden statement is the read commit strings', unless
    -- another such outcome at its registration and draw read another string at a row both read, the openings colliding
    (_hx : ∀ o : Reg × L.Ω × List (Cm × (innerAt T CT st).Coin) × (Reg₂ × VRec (execStmts Vs) Reg' dg N),
      FlockSoundness.Game.Outcome (recGame L Reg (innerAt T CT st) Cm
        fun R ω t => outerV (custodied zs₀ _hCust R ω t) dg N) σ o →
      VerifiesV (custodied zs₀ _hCust o.1 o.2.1 o.2.2.1) o.2.2.2 → ∀ q : ℕ → ByteArray,
      (∀ i, readsZ (custodied zs₀ _hCust) rt o i → q i = rowZ (custodied zs₀ _hCust) rt o i) →
        x o.1 o.2.1 = stmt q o.2.1 ∨
        ∃ o', FlockSoundness.Game.Outcome (recGame L Reg (innerAt T CT st) Cm
            fun R ω t => outerV (custodied zs₀ _hCust R ω t) dg N) σ o' ∧ o'.1 = o.1 ∧ o'.2.1 = o.2.1 ∧
          VerifiesV (custodied zs₀ _hCust o'.1 o'.2.1 o'.2.2.1) o'.2.2.2 ∧
          ∃ i, readsZ (custodied zs₀ _hCust) rt o i ∧ readsZ (custodied zs₀ _hCust) rt o' i ∧
            rowZ (custodied zs₀ _hCust) rt o i ≠ rowZ (custodied zs₀ _hCust) rt o' i ∧
            FlockSoundness.Binding.Collides ((regBcCommit i).ext (opZ (custodied zs₀ _hCust) rt o i)
              (opZ (custodied zs₀ _hCust) rt o' i)))
    -- the registered commit strings and the developer's openings
    (p : ℕ → ByteArray) (op₀ : ℕ → Flock.Registered.Port × ByteArray × ByteArray)
    (_hreg : ∀ i, (regBcCommit i).Opens rt (p i) (op₀ i))
    -- the count curve, bounding every session's rate
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (custodied zs₀ _hCust R ω t R₂ j).dj
      (custodied zs₀ _hCust R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    -- V*'s bridge
    (_hbr : Flock.Premises.VBridge (custodied zs₀ _hCust) dg N k Rw x wmsg Sk cmt msg),
    RoundFork (custodied zs₀ _hCust) dg N k Rw σ cmt ∨ SessionFork (custodied zs₀ _hCust) dg N k Rw M σ ∨
      InnerForkRec (custodied zs₀ _hCust) dg N k Rw σ x wmsg Refine.H512
        (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 (execClassAt st).cls
          (execClassAt st).regs (execClassAt st).sch) ∨
      (∃ o, FlockSoundness.Game.Outcome (recGame L Reg (innerAt T CT st) Cm
          fun R ω t => outerV (custodied zs₀ _hCust R ω t) dg N) σ o ∧
        VerifiesV (custodied zs₀ _hCust o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, readsZ (custodied zs₀ _hCust) rt o i ∧ rowZ (custodied zs₀ _hCust) rt o i ≠ p i ∧
          FlockSoundness.Binding.Collides ((regBcCommit i).ext (opZ (custodied zs₀ _hCust) rt o i) (op₀ i))) ∨
      (∃ o o', FlockSoundness.Game.Outcome (recGame L Reg (innerAt T CT st) Cm
          fun R ω t => outerV (custodied zs₀ _hCust R ω t) dg N) σ o ∧
        FlockSoundness.Game.Outcome (recGame L Reg (innerAt T CT st) Cm
          fun R ω t => outerV (custodied zs₀ _hCust R ω t) dg N) σ o' ∧ o'.1 = o.1 ∧ o'.2.1 = o.2.1 ∧
        VerifiesV (custodied zs₀ _hCust o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        VerifiesV (custodied zs₀ _hCust o'.1 o'.2.1 o'.2.2.1) o'.2.2.2 ∧
        ∃ i, readsZ (custodied zs₀ _hCust) rt o i ∧ readsZ (custodied zs₀ _hCust) rt o' i ∧
          rowZ (custodied zs₀ _hCust) rt o i ≠ rowZ (custodied zs₀ _hCust) rt o' i ∧
          FlockSoundness.Binding.Collides ((regBcCommit i).ext (opZ (custodied zs₀ _hCust) rt o i)
            (opZ (custodied zs₀ _hCust) rt o' i))) ∨
      prob (fun o => AcceptsStmt st ∧ VerifiesV (custodied zs₀ _hCust o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (innerAt T CT st).holds (stmt p o.2.1))
          (recGame L Reg (innerAt T CT st) Cm fun R ω t => outerV (custodied zs₀ _hCust R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (custodied zs₀ _hCust σ.1 ω t) dg N
            (fun R₂ j τ =>
              (custodied zs₀ _hCust σ.1 ω t R₂ j).boundCR dg N (execStmts Vs j).law.model k Rw M ρ τ) s) +
          ((innerAt T CT st).r : ℝ≥0∞) *
            expAt σ (fun ω t s => sumV (custodied zs₀ _hCust σ.1 ω t) dg N
              (fun R₂ j τ => (custodied zs₀ _hCust σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError (execClassAt st).sch
            ⟨(execClassAt st).cls.kLog, (execClassAt st).regs.length, (execClassAt st).cls.mPts⟩)) ∧
  (∀ {Out Prog Wit Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Inhabited Ωa] [Fintype Ωp] [Inhabited Ωp] [Fintype Ωo]
    [Inhabited Ωo] [Fintype Ωs] [Inhabited Ωs]
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
    (_hKey : Flock.Premises.HashDerivedKeyHm96)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit), honest o P w →
    ∀ E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop,
      recRealPr (regLeaves rows pub) nh msgs view P w E ≤ recSimPr (regLeaves rows pub) nh sim o p₀ E +
          (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
        recSimPr (regLeaves rows pub) nh sim o p₀ E ≤ recRealPr (regLeaves rows pub) nh msgs view P w E +
          (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)))

end Flock.Guarantees
