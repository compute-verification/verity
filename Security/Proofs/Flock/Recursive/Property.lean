import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.Statements
import Proofs.Flock.Recursive.Premises
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Recursive.Firewall
import Proofs.Flock.Recursive.Description
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Flock.RecResiduals

/-!
# The recursive audit's property (trusted)

One statement, `RecursiveAudit`, of the recursive audit as the verifier of record runs it: V*'s outer sessions verified
by `flock-verify verify --zk --registered` (each session's verdict `Flock.Zk.verify` and the statement check
`Flock.ProvedScope.check`), the inner protocol C-Flock's compiled table (`flockInnerCL` at `execArith`), and the
private circuit registered by hm96 leaves. It is a conjunction:

* (i) soundness in fork form, at an inner statement `st` whose class, regions and schedule are `execClassAt st`, over
  V*'s sessions as the prover supplies them, with the hidden statement read from `st`'s own description rows: the
  event is empty, or at a row `st`'s commit string is not the registered one and their openings collide, or the
  sessions type and there is a registrant fork (`regFinder`), a round fork, a session fork or an inner fork of the
  prover's own runs, or Lean's verifier accepts `st` and its description and the verifier of record every session
  while the registered private circuit's hidden statement fails with probability at most `RecursiveSound`'s bound with
  the compiled table's error as its inner term;
* (ii) zero knowledge with circuit privacy, stated about the firewall's code (`Flock.Firewall.run`) and V*'s outer
  sessions run by the honest prover's code (`Recursion.OuterZK.proverExe`), `RecursiveFirewallPrivate`'s form
  (`Proofs.Flock.Recursive.Firewall`), the inner coins live: a simulator that reads only the class, the schedule's
  transcript and the owed outputs, at the pinned key's hiding gap with no premise, and what it takes of the code each an
  obligation.

Its premises are the list in `Proofs.Flock.Recursive.Premises`. Its proof (`Flock.SecurityProofs.RecursiveAudit`,
`Proofs.Flock.Recursive.Audit`) names what else it takes.

The class is a total function of the statement (`Recursion.execClassAt`, `Proofs.Flock.Recursive.ExecClass`): when
the executable's checks of the statement's shape pass, `HClass.ofStmt` at no hidden entries (the universal unit's one
public statement), its regions and its `fast100` schedule; the constant `ExecClass.fixed` otherwise. It reads only the
statement's shape (`Recursion.shapeOf`: no region's bytes, which are instance data). An accepted statement
(`Recursion.AcceptsStmt`) passes the checks (`Recursion.classOk_of_accepts`), so the fixed class adds no accepted
case: at a statement Lean's verifier refuses, (i)'s event is empty.

* `stmtAt`, `hiddenAt`: the private circuit's statement at a list of commit strings `b ‖ c`, one for each of the
  registration's rows (what `Flock.Registered.opens` opens; the rows are what those strings commit to, so a row's hm96
  binding is in what the statement means), and the run's hidden statement, at `st`'s own description rows.
* `VerifiesV`, `VerifiesVR`, `VerifiesVStar`: the verifier of record's verdict on every one of V*'s sessions, typed, as
  the prover supplies them, and as the prover supplies them with the verifier's check that each is V*'s statement of
  its index (`RecVStar.checkStmt`). (i)'s event reads the last, so nothing is assumed of what a session's inputs parse
  to.

C-Flock's definitions and the recursion's model are still in the proofs package, so the statement is here rather than
in `Security/Properties/`.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The verifier of record accepts every one of V*'s sessions**: each session's proof (`AcceptsV`: setup, registered
reads, `Flock.Zk.shapeOk`, `Flock.Zk.verify`) and its statement's form as `--registered` checks it
(`FailClosed.ScopeZR`: `Flock.ProvedScope.check` with the registered reads), at the registration the outcome records. -/
def VerifiesV {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsV zs o ∧ ∀ j, FailClosed.ScopeZR (zs o.1 j).I

/-- **The verifier of record accepts every one of V*'s sessions as the prover supplies them** (`VerifiesV` at
`ZkOuterR`): each session's proof (`AcceptsVR`) and its statement's form (`FailClosed.ScopeZR`), at the registration
the outcome records. -/
def VerifiesVR {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsVR zr o ∧ ∀ j, FailClosed.ScopeZR (zr o.1 j).I

/-- **The verifier of record accepts every one of V*'s sessions as V*'s statement of its index**, as the prover
supplies them: each session's proof and the check that it is V*'s statement `j` (`AcceptsVRV`: flock-verify with
`--rec-inner` at `i`, `--rec-part j`, `--rec-bindings` at `bind j` and `--rec-rows` at `rows`, `RecVStar.checkStmt`),
and its statement's form (`FailClosed.ScopeZR`), at the registration the outcome records. -/
def VerifiesVStar {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (i : RecVStar.Inner) (bind : Fin S → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port)) (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsVRV i bind rows zr o ∧ ∀ j, FailClosed.ScopeZR (zr o.1 j).I

/-- **C-Flock's compiled inner protocol as the executable runs it**: `flockInnerCL` at the executable's arithmetic
(`execArith`) and the inner statement's Merkle scheme, SHA-512 with plain leaves (`Refine.enc512`, `Refine.leafPlain`
under `Refine.H512`: `flock-leaf/sha512-unsalted`, which `RecVStar.innerTags` names), on the class `cls`, its regions
and the schedule `sch`. -/
noncomputable abbrev innerExec {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (cls : HClass) (regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog)))
    (sch : Accounting.Schedule) : Inner :=
  flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain cls regs sch

/-- **The number of V*'s outer statements** over an inner session of a table of `m` variables, `k_log` and `claims`
claims, as V*'s staging lays them out (`rec_vstage.stage`, `VStar.statements`) and the verifier of record groups the
algebra's residuals (`Flock.RecResiduals.groups`, `rec_residuals.parts`): one `RecOpen` per Ligerito level of
`fast100 m`, then one `InnerRepCheck` per part of the rep's algebra, each residual in S's order in the first part that
stays within `Flock.RecResiduals.MAX_SLOTS` sha512x3 slots, `MAX_REGIONS` regions and 16 residuals, then one `RecCap` a
level: at each later level `l` over both reps' `2^cap_l` nodes of its tops (`VStar.levelCapStatements`,
`RecVStar.checkTopsListed`), and last over the `2^cap₀` nodes of `root_B`'s level-0 cap (`VStar.capStatement`,
`RecVStar.checkCapListed`). Zero at a shape the verifier refuses, where V* stages nothing. -/
def outerParts (m kLog claims : ℕ) : ℕ :=
  match Flock.RecResiduals.shape? m kLog claims with
  | none => 0
  | some sh =>
    match Flock.RecAlgebra.structureOf sh >>= fun S =>
        Flock.RecResiduals.groups S (Flock.RecResiduals.portSizes sh) with
    | .ok gs => sh.sched.levels.size + gs.size + sh.sched.levels.size
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

/-- **V*'s typed sessions** (`Recursion.zsOf`), at the records' custody (`Flock.Premises.RecordCustody`) and a typing
of every index (`Recursion.Typed`): each session `zr R ω t R₂ j` that types, with the record (`RecordsLiveZKJ`) its
custody gives, and at one that doesn't, a session of its index that types, refusing everything. -/
noncomputable abbrev custodied {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') (hC : Flock.Premises.RecordCustody zr) (h : Typed zr) :
    A → B → C → D → (j : Fin S) → (Vs j).Outer Reg' :=
  zsOf zr (fun R ω t R₂ j => And.left (hC R ω t R₂ j)) h

/-- **The prover's strategy against the typed sessions** (`Recursion.recT`): its strategy against the sessions as it
supplies them, with the same registrations and rounds. -/
noncomputable abbrev transported {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hC : Flock.Premises.RecordCustody zr) (h : Typed zr)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N)) :
    Strategy (recGame L Reg I Cm fun R ω t => outerV (custodied zr hC h R ω t) dg N) :=
  recT zr (fun R ω t R₂ j => And.left (hC R ω t R₂ j)) h σ

/-- **A private circuit's statement at a list of commit strings**: `stmt` at the list's first `rt.leaves` strings, one
for each of the registration's rows; it reads nothing past them. -/
abbrev stmtAt {Ω H : Type} (rt : Flock.Registered.Port) (stmt : (Fin rt.leaves → ByteArray) → Ω → H) :
    (ℕ → ByteArray) → Ω → H :=
  fun q ω => stmt (fun r => q r) ω

/-- **The hidden statement read from the inner statement's own rows**: the private circuit's statement `stmt` at `st`'s
description rows (`Flock.Registered.rowsOf st.c st.pub rt`, the commit strings Lean's verifier opens against `rt`), at
every registration. -/
abbrev hiddenAt {Ω H : Type} (st : Flock.Stmt) (rt : Flock.Registered.Port)
    (stmt : (Fin rt.leaves → ByteArray) → Ω → H) {Reg : Type} : Reg → Ω → H :=
  fun _ ω => stmtAt rt stmt (Flock.Registered.rowsOf st.c st.pub rt) ω

/-- **The law of a finite space of draws with an element**: `Law` at `Ω` and the units each draw takes, nonempty by
the element. Every law is one (`Classical.inhabited_of_nonempty`); (i) binds the space, its element and the draw rather
than a law, whose `nonempty` field would be a hypothesis of the statement. -/
abbrev drawLaw {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] (draw : Ω → Finset (Fin n)) : Law n := ⟨Ω, draw⟩

/-- **The recursive audit, as the verifier of record runs it.**

(i) For every inner statement `st`, with its class, regions and schedule `execClassAt st`, and every prover `σ` of the
recursive audit with C-Flock's compiled table there as its inner protocol (`innerAt`): V*'s statements `Vs`, as many
as V*'s staging lays out over the class and its regions (`outerStatementsAt`), each drawing all its units (`VLaw`:
the full draw on the operating system's bytes, `--draw subset:<units>`), V*'s sessions `zr R ω t R₂ j` under
their laws as the prover supplies them (`ZkOuterR`: inputs, own ports, draw file and record writer, with nothing
assumed of what the inputs parse to), their records' custody `_hCust`, the verifier of record's own inputs to them
(the inner shape `ish`, V*'s bindings `bind` and the shared rows `rows`; session `j` runs as V*'s statement `j`), at
which V*'s statements' files parse to `Vs` (`VStarParses`, an obligation until gap 4), the messages V*'s layers hold
and the rounds' commitments (`wmsg`, `Sk`, `cmt`, `msg`, as in `RecursiveSound`), and the registration's root
`rt`. Every session's inputs carry the tags their name names (`TagsFound`, `_hF`): an obligation on our code, not a
premise, which with refusal (5)'s name equation makes them V*'s. Where every index has a session that types
(`Typed zr`: its inputs pass the statement check, parse to V*'s circuit, have V*'s units and set up at some draw), `custodied zr _hCust h` are the typed sessions (`zsOf`: each
session that types, and at one that doesn't, a session of its index that types, refusing everything) and
`transported zr _hCust h σ` is `σ` against them (`recT`). `stmt q ω` is the hidden statement at the draw `ω` of the
private circuit whose `rt.leaves` rows the commit strings `q` commit to (`stmtAt` reads it at a list's first
`rt.leaves`), and the run's hidden statement is `stmt` at `st`'s own description rows (`hiddenAt`: the commit strings
`Flock.Registered.rowsOf st.c st.pub rt` that Lean's verifier opens against `rt`, `AcceptsDescription`, PROTOCOL.md
§16.16), whichever rows V*'s sessions read. Taken: A3 (`_hA3`). Then for the registered commit strings `p`, each
opening `rt` at its row `r < rt.leaves` through the developer's path `path₀ r`, for every count curve `(k, R_w, M, ρ)`
that bounds every session's rate at each setup of its inputs, given V*'s bridge at the committed rounds (VBridgeC, which
VBridge gives) at the typed sessions and that hidden statement: the event below is empty; or at some row `st`'s commit
string is not `p r`, and its opening of `rt` and the developer's give an explicit SHA-512 collision
(`Flock.Registered.bcPair`); or the sessions type and there is a registrant fork (one run of the transported prover
reaching two sessions whose registrant records clash, `regFinder`), a round fork, a session fork, or an inner fork of
the transported prover's own runs whose decoded transcripts play the compiled table to outcomes that clash; or
$$\Pr[\text{Lean's verifier accepts } st \text{ and its description} \wedge \text{every session verifies as V*'s}
  \text{ statement of its index} \wedge
  \neg\,\text{the statement of } p \text{ holds}]$$
is at most V*'s sessions' expected statistical errors under the full draw's model, `r` times their expected stage
slack, and the compiled table's error.

(ii) `RecursiveFirewallPrivate`, with no premise: for the class's private circuits registered by hm96 leaves
(`regLeaves rows pub`), the auditor's whole view (`fwRealPr`: the registration's, what the firewall sends,
`Flock.Firewall.run` at the pinned schedule `s` on the honest worker's requests `reqs` and `nh` salts of 192 uniform
bytes, and V*'s outer sessions `V` run by the honest prover's code `OuterZK.proverExe`) and the simulated one
(`fwSimPr`: `p₀` registered, the schedule's transcript `T` on ideal commit strings, V*'s simulator at the public
context `pubS` reads), which reads only the class, `T` and the owed outputs, give every event probabilities within
`2 n_r·δ_{K*} + ε_V(δ_{K*}) + n_h·δ_{K*}` both ways, `δ_{K*}` the hiding gap of hm96-sha512's pinned key
(`hidingGap My saltHash`) and `ε_V(δ)` V*'s sessions' error at `δ` (`OuterZK.Sessions.errAt`). The inner coins are
live: the auditor draws each after the commit strings before it, and each message `msgs … i` and the requests `reqs`
read those strings (`firewallStrings`, `firewallCommits`). What it takes of the code is each an obligation: the requests run through the firewall to `T` with the messages' salted commit strings in
it (`FirewallSends`), and V*'s staging (`SessionsPublic`, `SessionsCoined`, `SessionsHonest`, and `OuterZK.Sessions`'
`hz`, `hR`, `hS`).

(i) is in fork form and states no bits. The recursion's knowledge form, `Claims.Recursion.KS`, is at its level
`Claims.Recursion.lam`: about 65 bits per SHA-512 call at the deployed shape, below the 100-bit floor and the `2^-128`
target under every weight. Route 2C's bound λ₂ (`lamLin2`) evaluates to at least 100 at the deployed records
(`RecDeploy.lamLin2_deployed_ge_dep`), but `KSLin2` at the deployment's dense statements (`KSLin2On` at
`RecDeploy.DenseAt`) only follows from premises still open (`RecDeploy.ksLin2_deployed_open`), Holo statements being
candidate 3's own model, and `BitsLin2` at those statements follows from it (`bitsLin2On_of_ksLin2On`). -/
def RecursiveAudit : Prop :=
  (∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Inhabited (CT k)]
    -- the registration's draws, a finite space with an element, and the units each draws
    {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] {draw : Ω → Finset (Fin n)} {Reg Cm : Type} {Reg' Reg₂ : Type}
    -- the inner statement; the compiled table's class, regions and schedule are `execClassAt st`
    (st : Flock.Stmt)
    -- V*'s statements over the class, each drawing all its units, and V*'s sessions at each history and
    -- registration `R₂`, as the prover supplies them
    {Vs : Fin (outerStatementsAt st) → VStmt}
    (zr : Reg → (drawLaw draw).Ω → List (Cm × (innerAt T CT st).Coin) → Reg₂ →
      (j : Fin (outerStatementsAt st)) → (Vs j).OuterR Reg')
    -- the records' custody
    (_hCust : Flock.Premises.RecordCustody zr)
    -- the verifier of record's own inputs to V*'s sessions: the inner shape (`--rec-inner`), V*'s bindings for each
    -- statement (`--rec-bindings`) and the shared rows (`--rec-rows`); session `j` runs with `--rec-part j`
    (ish : RecVStar.Inner) (bind : Fin (outerStatementsAt st) → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port))
    -- obligation (gap 4): V*'s statement `j`'s file under `bind j` parses to `Vs j`'s circuit
    (_hob : VStarParses Flock.Tags.circuit ish bind Vs)
    -- obligation (our code): every session's inputs carry the tags their name names, as `verifyCmd` forms them
    (_hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover
    (σ : Strategy (recGame (drawLaw draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N))
    -- the messages V*'s layers hold, the rounds' commit strings and commitments
    (wmsg : ((j : Fin (outerStatementsAt st)) → (Vs j).Bits) → List (innerAt T CT st).Msg)
    (Sk : ℕ → (j : Fin (outerStatementsAt st)) →
      Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin (outerStatementsAt st)) → Cm →
      PosK (Vs j).c (Vs j).hU (Vs j).nv →
      (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (outerStatementsAt st)) →
      PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (innerAt T CT st).Msg)
    -- the registration's root, and the private circuit's statement by the commit strings of its rows
    (rt : Flock.Registered.Port) (stmt : (Fin rt.leaves → ByteArray) → (drawLaw draw).Ω → Hidden (execClassAt st).cls)
    -- A3
    (_hA3 : Flock.Premises.UniformRandomBytes Vs)
    -- the registered commit strings, each opening `rt` at its row through the developer's path
    (p path₀ : ℕ → ByteArray) (_hreg : ∀ r < rt.leaves, Flock.Registered.opens rt (p r) r (path₀ r) = true)
    -- the count curve, bounding every session's rate
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S₀
      (d₀ : Lean.Json) (y₀ : ZkHidden.DrawSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    -- V*'s bridge at the committed rounds, at the typed sessions
    (_hbr : ∀ h : Typed zr,
      Flock.Premises.VBridgeC (custodied zr _hCust h) dg N k Rw (hiddenAt st rt stmt) wmsg Sk cmt msg),
    prob (fun o => AcceptsStmt st ∧ AcceptsDescription rt st ∧
          VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ (innerAt T CT st).holds (stmtAt rt stmt p o.2.1))
        (recGame (drawLaw draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N) σ = 0 ∨
    (∃ r < rt.leaves, Flock.Registered.rowsOf st.c st.pub rt r ≠ p r ∧
      Flock.Registered.Collides (Flock.Registered.bcPair rt r (Flock.Registered.rowsOf st.c st.pub rt r)
        (Flock.Registered.pathOf st.pub rt r) (p r) (path₀ r))) ∨
    ∃ h : Typed zr,
      (regFinder (custodied zr _hCust h) dg N (transported zr _hCust h σ)).Finds Refine.H512 ∨
      RoundFork (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) cmt ∨
      SessionFork (custodied zr _hCust h) dg N k Rw M (transported zr _hCust h σ) ∨
      InnerForkRec (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) (hiddenAt st rt stmt) wmsg
        Refine.H512 (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (execClassAt st).cls
          (execClassAt st).regs (execClassAt st).sch) ∨
      prob (fun o => AcceptsStmt st ∧ AcceptsDescription rt st ∧
            VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (innerAt T CT st).holds (stmtAt rt stmt p o.2.1))
          (recGame (drawLaw draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N) σ ≤
        expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
            (fun R₂ j τ =>
              (custodied zr _hCust h σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((innerAt T CT st).r : ℝ≥0∞) *
            expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
              (fun R₂ j τ => (custodied zr _hCust h σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError (execClassAt st).sch
            ⟨(execClassAt st).cls.kLog, (execClassAt st).regs.length, (execClassAt st).cls.mPts⟩)) ∧
  (∀ {Out Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Inhabited Ωa] [Fintype Ωp] [Inhabited Ωp]
    -- the firewall's pinned schedule
    (s : Flock.Firewall.Schedule)
    -- the registration: the private circuit's rows, and what the auditor sees of their leaves
    (nr : ℕ) (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop)
    -- the honest worker's requests to the firewall and the messages they commit, each reading the commit strings the
    -- firewall sent before it (the inner coins live), and the schedule's transcript
    (nh : ℕ) (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent)
    -- V*'s outer sessions, the public context their simulator reads, and what a stopped run sends
    (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub)
    (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub) (stopped : OuterZK.Stopped)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit)
    -- what it takes of the code, each an obligation
    (_hF : FirewallSends s reqs msgs T) (_hpub : SessionsPublic rows pub honest msgs V pubS) (_hC : SessionsCoined V)
    (_hH : SessionsHonest honest msgs V), honest o P w →
    ∀ E : Rv × Ωa × (Array Flock.Firewall.Sent × Bool) × V.View → Prop,
      fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E ≤ fwSimPr rows pub T V pubS o p₀ E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) ∧
        fwSimPr rows pub T V pubS o p₀ E ≤ fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)))

end Flock.Guarantees
