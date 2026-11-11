module

public import Proofs.Flock.Recursive.LoopAny
public import Proofs.Flock.Recursive.FirewallAccepts
public import Proofs.Flock.Recursive.FirewallFixed
public import Proofs.Flock.Claims.Recursion.BaseParses
public import Proofs.Flock.SessionLoop.TablesSide
public import Proofs.Flock.Soundness.Audit.OneStage
import all Init.Control.Except
import all Init.Data.Array.Basic
import all Init.Data.Int.Repr
import all Init.Data.List.Basic
import all Init.Data.List.Control
import all Init.Data.Ord.String
import all Init.Data.Range.Basic
import all Init.Data.Repr
import all Init.Data.String.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Legacy
import all Init.Data.ToString.Basic
import all Lean.Data.Json.Basic

/-!
# `RecursiveZKLoop` and `RecursiveZKLoopAnySum` say something

Each guarantee's honest lemma concludes its premises at one instance, and its nontrivial lemma states that its bound
fails at another.

The instance:
* the ZK server's schedule `sched`: one root table `t` of one node, no rounds, a Hello that links nothing. It is
  canonical (`sched_canonical`), and the honest worker at it, `hon`, fits it (`hon_fits`), so the ZK server sends its
  transcript on the worker's requests and finishes (`sends`: `FirewallSends`, from `honestReqs_accepts`);
* the same requests as a worker of the adaptive firewall (`wAny`): it finishes at every salt (`wAny_finishes`), by
  one kernel evaluation at the zero salt (`hon_finishes0`) and `liveFinished_chopReqs`;
* one statement of V* (`m = 1`): the honest fixture's (`RecDeploy.BaseParses.fixture_stmt`), whose statement and
  witness are in the claims' relation at the deployed masked parser, `deployZk Readable.relation`;
* the deployed tables, through `TablesSide`'s `_dataE` theorems, with `SwCausalOn (Loop.SwOk Masked.dataE)` at the
  honest values (`swCausalOn_honestVals_dataE`); and the deployed release, which reads no stop
  (`LoopAny.releaseReadsNoStop_loopRelease`).

The nontrivial lemmas take no request and no statement of V* (`m = 0`, so the error terms are `0`), so the real view
never finishes, and a release that reads whether the inner session finished. The bound fails where the ZK server never
finishes (`RecursiveZKLoop`: the simulated view always finishes), and where the release reads the stop of a session
that didn't finish (`RecursiveZKLoopAnySum`, at no holes: the real view is released unfinished at the stop `true`, and
the simulated view is unfinished only at the stop `false`).

* `Flock.SecurityProofs.RecursiveZKLoop_honest`, `Flock.SecurityProofs.RecursiveZKLoop_nontrivial`.
* `Flock.SecurityProofs.RecursiveZKLoopAnySum_honest`, `Flock.SecurityProofs.RecursiveZKLoopAnySum_nontrivial`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.LoopWitness

open Lean (Json)
open Flock.Firewall FirewallRun FirewallAny FirewallAdaptive
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## The schedule and its honest worker -/

/-- **One root table, `t`, of one node, and no rounds**: Hello links nothing, and there is no link `y` and no coin. -/
def sched : Schedule where
  hello := Json.obj ⟨⟨Std.DTreeMap.Internal.Impl.inner 1 "link" (Json.obj ⟨⟨.leaf⟩⟩) .leaf .leaf⟩⟩
  roots := Json.obj ⟨⟨Std.DTreeMap.Internal.Impl.inner 1 "t" (jnat 1) .leaf .leaf⟩⟩
  y := 0
  m := 0
  streams := []
  rounds := #[]

theorem roundSpec_null : Flock.Firewall.roundSpec Json.null = none := (rfl)

/-- One 64-byte node of zeros. -/
def top64 : ByteArray := ⟨Array.replicate 64 0⟩

/-- **The honest worker at `sched`**: table `t`'s top is `top64`, with no root frame, label, opening or round. -/
def hon : Flock.Firewall.Honest where
  sigma := ""
  rootF := .empty
  roots := [("t", top64)]
  label _ := .empty
  opens _ := []
  round _ := {}

theorem hon_fits : FirewallHonest.Fits sched hon where
  pings := by intro b hb; cases hb
  sigma := by
    show plainStr "" = true
    simp [plainStr]
  hello := by
    show (Flock.Firewall.printJson (Flock.Firewall.helloWith sched.hello "")).toUTF8.size < 2 ^ 32
    decide +kernel
  commit := by
    simp only [FirewallWire.ReqFits, FirewallWire.PairsFit]
    decide +kernel
  tables := by decide +kernel
  roots := by decide +kernel
  caps := by decide +kernel
  link := by decide
  opensFit := by
    intro j r hr
    simp [Flock.Firewall.honestOpens, hon] at hr
  opened := by decide +kernel
  rounds := by
    intro i hi
    simp [sched] at hi
  roundsFit := by
    intro i r hr
    have h0 : Flock.Firewall.honestRound sched hon i = [] := by
      unfold Flock.Firewall.honestRound
      have hg : sched.rounds.getD i Json.null = Json.null := rfl
      rw [hg, roundSpec_null]
    rw [h0] at hr
    cases hr
  proofs := by intro p hp; cases hp

/-- **Hello is plain**: its one key, `link`, a plain string, and its value an empty object. `plainStr` reads its string
through `String.all`, which the kernel doesn't evaluate, so `simp` decides it, and the kernel the rest. -/
theorem plain_hello : plain sched.hello = true := by
  have hs : plainStr "link" = true := by simp [plainStr]
  show (plainObj .leaf && plainStr "link" && plain (Json.obj ⟨⟨.leaf⟩⟩) && plainObj .leaf) &&
    ascending (Std.DTreeMap.Internal.Impl.inner 1 "link" (Json.obj ⟨⟨.leaf⟩⟩) .leaf .leaf) = true
  rw [hs]
  decide +kernel

/-- **The ZK server holds a session to `sched`**: `Schedule.canonical`, by evaluation once Hello is plain. -/
theorem sched_canonical : FirewallHonest.Canonical sched := by
  unfold FirewallHonest.Canonical Schedule.canonical
  rw [plain_hello]
  decide +kernel

/-- The schedule's holes: its one row. -/
abbrev nhS : ℕ := FirewallHonest.rowCount sched

/-- The honest worker at every program, witness, randomness, tape and commit strings. -/
def honD : Unit → Unit → Unit → Unit → Unit → (Fin nhS → ByteArray) → Flock.Firewall.Honest :=
  fun _ _ _ _ _ _ => hon

/-- **The ZK server sends `sched`'s transcript on the honest worker's requests, and finishes.** -/
theorem sends : Flock.Guarantees.FirewallSends (Prog := Unit) (Wit := Unit) (Rr := Unit) sched
    (fun _ _ _ => fun (_ : Unit) (_ : Unit) (_ : Fin nhS → ByteArray) => Flock.Firewall.honestReqs sched hon)
    (fun _ _ _ => FirewallHonest.honestMsgs sched honD () () ())
    (Flock.Guarantees.firewallT sched sched.tables) :=
  Flock.SecurityProofs.firewallSends_of_accepts sched _ _ _
    (Flock.SecurityProofs.honestReqs_accepts sched sched_canonical honD (fun _ _ _ _ _ _ => hon_fits)
      (by unfold FirewallHonest.Causal; intros; rfl))

/-! ## The adaptive firewall finishes at every salt -/

theorem fwHoles_sched : Flock.Guarantees.fwHoles sched = 1 := by decide +kernel

/-- **At the zero salt, the live session on the honest worker's requests finishes**, by evaluation. -/
theorem hon_finishes0 :
    liveFinished sched (chopReqs (Flock.Firewall.honestReqs sched hon) [Flock.Guarantees.zeroSalt]) = true := by
  decide +kernel

/-- **The honest worker as a worker of the adaptive firewall**: its `i`-th request is the `i`-th honest one, whatever
the ZK server answered. -/
def wAny : Unit → Unit → Worker := fun _ _ h => (Flock.Firewall.honestReqs sched hon)[h.length]?

/-- Its request count. -/
abbrev nA : ℕ := (Flock.Firewall.honestReqs sched hon).length

/-- **It finishes at every salt.** -/
theorem wAny_finishes (b : Fin (Flock.Guarantees.fwHoles sched + 0) → Fin 192 → Fin 256) :
    adaptFinished sched (wAny () ()) nA (Flock.Guarantees.saltsOf b) = true := by
  have hl : (Flock.Guarantees.saltsOf b).length = [Flock.Guarantees.zeroSalt].length := by
    rw [Flock.Guarantees.saltsOf, List.length_ofFn, fwHoles_sched]
    rfl
  show adaptFinished sched (fun h => (Flock.Firewall.honestReqs sched hon)[h.length]?)
    (Flock.Firewall.honestReqs sched hon).length (Flock.Guarantees.saltsOf b) = true
  rw [adaptFinished_fixed, liveFinished_chopReqs _ _ hl]
  exact hon_finishes0

/-! ## The nontrivial lemmas' instance -/

/-- A release that releases nothing. -/
def rel0 : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp where
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

/-- Frames that refuse. -/
def src0 : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
    _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray) :=
  fun _ _ _ _ => .error ""

/-- An auditor that rejects at once. -/
def vs0 : Claims.VStar Unit := ⟨fun _ => 0, fun _ _ _ => .ret .rej⟩

/-- **`sched` without its root**: no hole. -/
def sched0 : Schedule where
  hello := Json.obj ⟨⟨Std.DTreeMap.Internal.Impl.inner 1 "link" (Json.obj ⟨⟨.leaf⟩⟩) .leaf .leaf⟩⟩
  roots := Json.obj ⟨⟨.leaf⟩⟩
  y := 0
  m := 0
  streams := []
  rounds := #[]

theorem fwHoles_sched0 : Flock.Guarantees.fwHoles sched0 = 0 := by decide +kernel

theorem not_le_of {a b c d k : ℝ≥0∞} (ha : a = 1) (hb : b = 0) (hc : c = 0) (hd : d = 0) :
    ¬ a ≤ k * b + c + d := by
  rw [ha, hb, hc, hd]
  simp

end FlockSoundness.Discharge.Recursion.LoopWitness

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Flock.Firewall FirewallAdaptive LoopWitness
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **`RecursiveZKLoop`'s premises hold at one instance**: the deployed tables (with `SwCausalOn` at the honest
values), the ZK server sending `sched`'s transcript on the honest worker's requests, and one statement of V*, the honest
fixture's, in the claims' relation at the deployed masked parser at every context, its simulator's coins its own. -/
theorem RecursiveZKLoop_honest (h : RecDeploy.HonestFixture) :
    ∃ (y : Claims.Msg) (w : Claims.Flock.Wit),
      Claims.ZKServer.Loop.finS.Tables ∧
      Claims.ZKServer.Loop.finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) ∧
      Claims.ZKServer.Loop.finS.DummyReg ∧ Claims.ZKServer.Loop.finS.Causal ∧
      Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE)
        Claims.ZKServer.Loop.finS.honestVals ∧
      Claims.ZKServer.Loop.finS.TapeOK ∧
      Flock.Guarantees.FirewallSends (Prog := Unit) (Wit := Unit) (Rr := Unit) sched
        (fun _ _ _ => fun (_ : Unit) (_ : Unit) (_ : Fin nhS → ByteArray) => Flock.Firewall.honestReqs sched hon)
        (fun _ _ _ => FirewallHonest.honestMsgs sched honD () () ())
        (Flock.Guarantees.firewallT sched sched.tables) ∧
      (OuterZK.LoopSessions.atWorker 1 (fun (_ : ℕ) (_ : Unit × (Fin nhS → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit nhS) => w)).Honest
        (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
        (fun u : (Unit × Unit) × (Fin nhS → ZkView.Salt) =>
          OuterZK.wctxAt (FirewallHonest.honestMsgs sched honD () () ()) u.1.1 u.1.2 u.2) ∧
      ∀ (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
        (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
          _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)) (t : ℕ),
        (OuterZK.LoopSessions.atWorker 1 (fun (_ : ℕ) (_ : Unit × (Fin nhS → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit nhS) => w)).Sized (OuterZK.loopSim L₀ src t)
          (fun _ => (OuterZK.loopSim L₀ src t).C y) := by
  obtain ⟨y, w, hR⟩ := RecDeploy.BaseParses.fixture_stmt h
  exact ⟨y, w, Claims.ZKServer.Loop.tables_dataE, Claims.ZKServer.Loop.complete_dataE,
    Claims.ZKServer.Loop.dummyReg_dataE, Claims.ZKServer.Loop.causal_dataE,
    Claims.ZKServer.Loop.swCausalOn_honestVals_dataE, Claims.ZKServer.Loop.tapeOK_dataE, sends,
    fun _ _ _ => hR, fun _ _ _ _ _ _ => ⟨Equiv.refl _⟩⟩

/-- **`RecursiveZKLoop`'s bound fails where the ZK server never finishes**: with no request and no statement of V*, the
real view never finishes and the simulated one always does, so an event of not finishing has probability `1` against a
bound of `0`. -/
theorem RecursiveZKLoop_nontrivial :
    ∃ (cut : Unit → Unit × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → Bool)
      (stop : (((Unit × Unit) × (Fin 0 → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
        ((OuterZK.loopRef rel0 src0).swP Claims.ZKServer.Loop.finS.honestVals).Nonce
        ((OuterZK.loopRef rel0 src0).swP Claims.ZKServer.Loop.finS.honestVals).Rand 0) × Unit → Unit)
      (E : Bool → Prop),
      ¬ Verity.Game.prCoin (fun u => E (cut (stop u) (Flock.Guarantees.loopReal Claims.ZKServer.Loop.finS.honestVals
            rel0 src0 sched (fun (_ : Unit) (_ : Unit) (_ : Fin 0 → ByteArray) => [])
            (fun (_ : Unit) (_ : Unit) (i : Fin 0) _ => i.elim0) 0 (fun _ _ => [])
            (fun _ _ => (⟨[]⟩ : Claims.Flock.Wit)) vs0 (fun _ _ => ()) u.1))) ≤
        (Fintype.card Unit : ℝ≥0∞) * Verity.Game.prCoin (fun c : Unit × ((Unit × (Fin 0 → ZkView.Bits × ZkView.Salt)) ×
            OuterZK.LoopSessions.SimCoins (fun _ => Unit) 0) =>
          E (cut c.1 (Flock.Guarantees.loopIdeal 1 rel0 src0 (fun _ => #[]) 0 (fun _ _ => []) (fun _ => Unit) vs0
            (fun _ _ => ()) c.2))) +
        (OuterZK.LoopSessions.atWorker (Ωa := Unit) (Ωp := Unit) (nh := 0) 0 (fun _ _ => [])
            (fun _ _ => (⟨[]⟩ : Claims.Flock.Wit))).err
            Claims.ZKMal.CoinOpen.shaH OuterZK.loopK
            ((OuterZK.loopRef rel0 src0).swP Claims.ZKServer.Loop.finS.honestVals).prover
            (OuterZK.loopEps rel0 src0 1) (4 * 1 * 3 ^ 1) vs0
            (fun x => (fun (_ : Unit) (_ : Fin 0 → ByteArray) => ()) x.1 x.2.1) +
        ((0 : ℕ) : ℝ≥0∞) * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) := by
  refine ⟨fun _ v => v.2.1.2, fun _ => (), fun b => b = false, ?_⟩
  refine not_le_of (FlockSoundness.Audit.prCoin_eq_one_of fun _ => rfl)
    (le_antisymm ((FlockSoundness.Game.prCoin_mono fun _ hc => nomatch hc).trans
      FlockSoundness.Game.prCoin_false.le) zero_le) ?_ (by simp)
  refine ENNReal.iSup_eq_zero.2 fun _ => ?_
  simp [OuterZK.LoopSessions.atWorker]

/-- **`RecursiveZKLoopAnySum`'s premises hold at one instance**: the deployed tables (with `SwCausalOn` at the honest
values); the honest worker's requests, as a worker of the adaptive firewall, finishing at every salt, so its contexts
are all of them; one statement of V*, the honest fixture's, in the claims' relation at the deployed masked parser at
every context, its simulator's coins its own; and the deployed release reading no stop. -/
theorem RecursiveZKLoopAnySum_honest (h : RecDeploy.HonestFixture) :
    ∃ (y : Claims.Msg) (w : Claims.Flock.Wit),
      Claims.ZKServer.Loop.finS.Tables ∧
      Claims.ZKServer.Loop.finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) ∧
      Claims.ZKServer.Loop.finS.DummyReg ∧ Claims.ZKServer.Loop.finS.Causal ∧
      Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE)
        Claims.ZKServer.Loop.finS.honestVals ∧
      Claims.ZKServer.Loop.finS.TapeOK ∧
      (∀ u : (Unit × Unit) × (Fin (Flock.Guarantees.fwHoles sched + 0) → Fin 192 → Fin 256),
        adaptFinished sched (wAny u.1.1 u.1.2) nA (Flock.Guarantees.saltsOf u.2) = true) ∧
      (OuterZK.LoopSessions.atWorker 1
          (fun (_ : ℕ) (_ : Unit × (Fin (Flock.Guarantees.fwHoles sched) → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit (Flock.Guarantees.fwHoles sched)) => w)).Honest
        (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
        (fun u : {u : (Unit × Unit) × (Fin (Flock.Guarantees.fwHoles sched + 0) → Fin 192 → Fin 256) //
            adaptFinished sched (wAny u.1.1 u.1.2) nA (Flock.Guarantees.saltsOf u.2) = true} =>
          Flock.Guarantees.anyCtx sched wAny nA 0 u.1.1.1 u.1.1.2 u.1.2) ∧
      (∀ (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
        (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
          _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)) (t : ℕ),
        (OuterZK.LoopSessions.atWorker 1
          (fun (_ : ℕ) (_ : Unit × (Fin (Flock.Guarantees.fwHoles sched) → ByteArray)) => y)
          (fun (_ : ℕ) (_ : OuterZK.WCtx Unit Unit (Flock.Guarantees.fwHoles sched)) => w)).Sized
          (OuterZK.loopSim L₀ src t) (fun _ => (OuterZK.loopSim L₀ src t).C y)) ∧
      ∀ (n : ℕ) (M : Fin n → ℕ),
        Flock.Guarantees.ReleaseReadsNoStop (Flock.Guarantees.loopRelease (Ωa := Unit) M) none := by
  obtain ⟨y, w, hR⟩ := RecDeploy.BaseParses.fixture_stmt h
  exact ⟨y, w, Claims.ZKServer.Loop.tables_dataE, Claims.ZKServer.Loop.complete_dataE,
    Claims.ZKServer.Loop.dummyReg_dataE, Claims.ZKServer.Loop.causal_dataE,
    Claims.ZKServer.Loop.swCausalOn_honestVals_dataE, Claims.ZKServer.Loop.tapeOK_dataE, fun u => wAny_finishes u.2,
    fun _ _ _ => hR, fun _ _ _ _ _ _ => ⟨Equiv.refl _⟩, fun _ M => LoopAny.releaseReadsNoStop_loopRelease M⟩

/-- **`RecursiveZKLoopAnySum`'s bound fails where the release reads the stop of an unfinished session**: with no
request, no hole and no statement of V*, the real view never finishes. Released with the stop `true` beside whether it
finished, it reads `(true, false)` with probability `1`; the simulated view, cut before its end, is released at the
stop `false` unfinished, and at its end finished, so it never reads `(true, false)`, against a bound of `0`. -/
theorem RecursiveZKLoopAnySum_nontrivial :
    ∃ (cut : Bool → Unit × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → Bool × Bool)
      (s₀ : Bool)
      (stop : (((Unit × Unit) × (Fin (Flock.Guarantees.fwHoles sched0 + 0) → Fin 192 → Fin 256)) ×
        Claims.Sampling.unitTape.at 0
          ((OuterZK.loopRef rel0 src0).swP Claims.ZKServer.Loop.finS.honestVals).Nonce
          ((OuterZK.loopRef rel0 src0).swP Claims.ZKServer.Loop.finS.honestVals).Rand 0) × Unit → Bool)
      (E : Bool × Bool → Prop),
      ¬ Verity.Game.prCoin (fun u => E (cut (stop u) (Flock.Guarantees.anyReal Claims.ZKServer.Loop.finS.honestVals
            rel0 src0 sched0 (fun (_ : Unit) (_ : Unit) (_ : History) => none) 0 0 0 (fun _ _ => [])
            (fun _ _ => (⟨[]⟩ : Claims.Flock.Wit)) vs0 (fun _ _ => ()) u.1))) ≤
        (Fintype.card (Fin ((FirewallTemplate.tmpl sched0 sched0.tables).length + 1) ⊕ Bool) : ℝ≥0∞) *
          Verity.Game.prCoin (fun c : (Fin ((FirewallTemplate.tmpl sched0 sched0.tables).length + 1) ⊕ Bool) ×
              ((Unit × (Fin (Flock.Guarantees.fwHoles sched0) → ZkView.Bits × ZkView.Salt)) ×
                OuterZK.LoopSessions.SimCoins (fun _ => Unit) 0) =>
            E (Flock.Guarantees.sumCut s₀ cut c.1 (Flock.Guarantees.loopIdeal 1 rel0 src0
              (Flock.Guarantees.fwFilled sched0) 0 (fun _ _ => []) (fun _ => Unit) vs0 (fun _ _ => ()) c.2))) +
        (OuterZK.LoopSessions.atWorker (Ωa := Unit) (Ωp := Unit) (nh := Flock.Guarantees.fwHoles sched0) 0
            (fun _ _ => []) (fun _ _ => (⟨[]⟩ : Claims.Flock.Wit))).err
            Claims.ZKMal.CoinOpen.shaH OuterZK.loopK
            ((OuterZK.loopRef rel0 src0).swP Claims.ZKServer.Loop.finS.honestVals).prover
            (OuterZK.loopEps rel0 src0 1) (4 * 1 * 3 ^ 1) vs0
            (fun x => (fun (_ : Unit) (_ : Fin (Flock.Guarantees.fwHoles sched0) → ByteArray) => ()) x.1 x.2.1) +
        (Flock.Guarantees.fwHoles sched0 : ℝ≥0∞) * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) := by
  refine ⟨fun c v => (c, v.2.1.2), false, fun _ => true, fun p => p.1 = true ∧ p.2 = false, ?_⟩
  refine not_le_of (FlockSoundness.Audit.prCoin_eq_one_of fun _ => ⟨rfl, rfl⟩)
    (le_antisymm ((FlockSoundness.Game.prCoin_mono fun c hc => ?_).trans FlockSoundness.Game.prCoin_false.le)
      zero_le) ?_ (by rw [fwHoles_sched0]; simp)
  · rcases c with ⟨k | c', x⟩
    · exact Bool.false_ne_true hc.1
    · exact Bool.false_ne_true hc.2.symm
  · refine ENNReal.iSup_eq_zero.2 fun _ => ?_
    simp [OuterZK.LoopSessions.atWorker]

end Flock.SecurityProofs
