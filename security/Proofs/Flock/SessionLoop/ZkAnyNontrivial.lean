module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.SessionLoop.Leak
public import Proofs.Flock.SessionLoop.Concrete
public import Proofs.Flock.SessionLoop.Concrete.Stmt
public import Proofs.Properties.Flock.Nets
public import Proofs.Flock.Claims.Prob

/-!
# `SessionLoopZKAnyDeployed`'s conclusion can fail

`Flock.Guarantees.SessionLoopZKAnyDeployed` concludes `Loop.ZkAnyAt`: the session loop's release meets `Claims.ZKOut`
within `Loop.anyErr`. That conclusion fails where the release's link carries the witness (`Loop.leakL`), at the concrete
statement and two witnesses whose hidden values differ (`Loop.not_zkOut_leakL`), wherever the error against
`Loop.helloRej` sums below `1` over the two.

* `Loop.noTables`: a session without tables, so without leaves (`Loop.nLeaves_noTables`), and so no hiding term.
* `Loop.pLog_unP_helloRej`, `Loop.fork_unP_helloRej`: against `helloRej` the unswitched prover sends its `Hello` and its
  first message without a SHA-512 call before `helloRej` rejects, so its collision finder finds nothing.
* `Flock.SecurityProofs.SessionLoopZKAnyDeployed_nontrivial`: there, at `t = 2`, the error at each witness is
  `2·3^{-2}`, and `ZkAnyAt` fails.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal

/-- **A session without tables**: no table at any statement, one tape. -/
def noTables : CoinLayer.TableData where
  Tape := Unit
  tapeZk _ := default
  J _ := 0
  C _ j := Fin.elim0 j
  cz _ _ j := Fin.elim0 j
  wt _ _ j := Fin.elim0 j
  wt₀ _ j := Fin.elim0 j
  enc _ _ _ _ := []
  fin _ _ _ := []
  off _ _ _ := []
  tape _ _ := (fun j => Fin.elim0 j, fun l => Fin.elim0 l.1, fun l => Fin.elim0 l.1)

/-- The tape of `noTables`'s session. -/
abbrev NT : Type :=
  (CoinLayer.session loopNr noTables).refP.Nonce × (CoinLayer.session loopNr noTables).refP.Rand

/-- **A release of constant fields**, at any tape. -/
def constRel (R : Type) : LeanRelease Flock.Wit R where
  rep _ _ := 0
  lay _ _ := default
  slot _ _ := none
  draws _ _ := .empty
  prog _ _ := {}
  coinsOf _ _ := #[]
  fresh _ _ := .error ""
  device _ _ _ _ _ := []
  computed _ _ _ _ := default
  cross _ _ _ _ _ _ := false
  scheduled _ _ _ _ _ := []
  coins _ _ := 0
  sched _ _ := 0
  stop := []
  hello _ _ := []
  helloChk _ _ := .error ""
  req _ _ := ("", 0)
  link _ _ _ _ := []
  files _ _ _ _ := ("", #[])
  finStops _ _ _ _ := false
  last _ := 0

/-- Tables that never compute. -/
def noReal :
    Msg → Flock.Wit → NT → ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray) :=
  fun _ _ _ _ => .error ""

/-- A worker that never answers. -/
def noWk : (LeanRelease.ofLoop noTables (leakL (constRel NT)) noReal).toSrv.Worker := fun _ _ _ _ _ => none

/-- The values of a session without tables. -/
def noVals : (CoinLayer.session loopNr noTables).SwVals := fun _ _ _ _ _ j => Fin.elim0 (j : Fin 0)

theorem nLeaves_noTables (x : Msg) : (CoinLayer.session loopNr noTables).nLeaves x = 0 :=
  Fintype.card_eq_zero_iff.2 ⟨fun l => Fin.elim0 (l.1 : Fin 0)⟩

/-- **The unswitched prover makes no SHA-512 call against `helloRej`**: it sends its `Hello`, then its first message
on no opening, and `helloRej` rejects the second. -/
theorem pLog_unP_helloRej (S : RefSession) (vs : S.SwVals) (w : Flock.Wit) (x : Msg)
    (ω : (S.unP vs).Nonce × (S.unP vs).Rand) :
    (runStar CoinOpen.shaH ((S.unP vs).prover w) helloRej x ω ()).pLog = [] := by
  simp [runStar, helloRej, stepWith, Proto.prover, RefSession.unP, strat, allOk, ocBind, OC.eval]

/-- **The unswitched prover's collision finder against `helloRej` finds nothing.** -/
theorem fork_unP_helloRej (S : RefSession) (vs : S.SwVals) (w : Flock.Wit) (x : Msg) :
    (Fork CoinOpen.shaH ((S.unP vs).prover w) helloRej () x).Pr CoinOpen.shaH = 0 := by
  have h : ∀ s, (Fork CoinOpen.shaH ((S.unP vs).prover w) helloRej () x).out s = none := fun s => by
    show findColl ((runStar CoinOpen.shaH ((S.unP vs).prover w) helloRej x (s.1, s.2.1) ()).pLog ++
      (runStar CoinOpen.shaH ((S.unP vs).prover w) helloRej x (s.1, s.2.2) ()).pLog) = none
    rw [pLog_unP_helloRej, pLog_unP_helloRej]
    rfl
  unfold Finder.Pr
  rw [Claims.Pr_congr (q := fun _ => False) fun s => by simp [h s]]
  exact Claims.Pr_false

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer Claims.ZKServer.Loop

/-- **`SessionLoopZKAnyDeployed`'s conclusion fails where the link carries the witness**: at a session without tables,
the release `leakL` of constant fields, the concrete statement under the relation of every pair and two witnesses whose
hidden values differ, `ZkAnyAt` fails at `t = 2`, its error against `helloRej` being `2·3^{-2}` at each witness. -/
theorem SessionLoopZKAnyDeployed_nontrivial :
    ¬ Loop.ZkAnyAt Loop.noTables Loop.noVals (Loop.leakL (Loop.constRel Loop.NT)) Loop.noReal Loop.noWk
      (fun _ _ => True) 2 := by
  intro h
  obtain ⟨h1, -⟩ := h
  have hs : FlockDeploy.setupOf true FlockDeploy.concreteStmt =
      some (FlockDeploy.concreteIns, FlockDeploy.concreteSetup.1, FlockDeploy.concreteSetup.2) :=
    FlockDeploy.concreteStmt_setupOf _root_.Properties.Flock.Nets.fixedNetsBuild
  have hF : ∀ w, (Fork CoinOpen.shaH
      (((Loop.anyStops Loop.noTables (Loop.leakL (Loop.constRel Loop.NT)) Loop.noReal).unP Loop.noVals).prover w)
      Loop.helloRej () FlockDeploy.concreteStmt).Pr CoinOpen.shaH = 0 :=
    fun w => Loop.fork_unP_helloRej _ Loop.noVals w _
  have hL : (CoinLayer.session Loop.loopNr Loop.noTables).nLeaves FlockDeploy.concreteStmt = 0 :=
    Loop.nLeaves_noTables _
  refine Loop.not_zkOut_leakL Loop.noTables (Loop.constRel Loop.NT) Loop.noReal Loop.noWk hs (refK_pos hs)
    (w₁ := ⟨[]⟩) (w₂ := ⟨[true]⟩) (by decide) (fun _ _ => True) trivial trivial _ _ ?_ h1
  simp only [Loop.anyErr, Bool.false_eq_true, ↓reduceIte, hF, hL]
  norm_num

end Flock.SecurityProofs
