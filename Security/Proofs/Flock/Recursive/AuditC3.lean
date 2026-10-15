import Proofs.Flock.Recursive.PublicAudit
import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.HoloCoins
import Proofs.Flock.Recursive.HoloMap
import Proofs.Flock.Recursive.TopFinder
import Proofs.Flock.Recursive.ValueSrc
import Proofs.Flock.Recursive.LeafFinder

/-!
# The recursive audit at candidate 3 (trusted)

`RecursiveAuditC3` is `PublicAudit` at candidate 3's inner protocol, the design that runs: the inner session's
circuits have hidden unit circuits, whose templates are the registered table R's, opened by the sparse phase and R's
registered opening (`Model.holoTableCL`, `Recursion.holoInnerAt`). It is a conjunction:

* (i) soundness in fork form, at an inner statement `st` that Lean's verifier sets up from its own partition object and
  program (`AcceptsPublic`), the sparse phase's parameters `hp` (the verifier's own) and R's registration: the event is
  empty, or the sessions type and there is a registrant fork, a round fork, a session fork or an inner fork of the
  prover's own runs whose decoded transcripts play candidate 3's table to outcomes that clash (`clashPickHL`), or
  Lean's verifier accepts `st`, builds V*'s statement list and accepts every session as V*'s statement of its index while
  no hidden types make a table near the word the prover's own runs open at R's level 0 (`regStmt`) satisfiable
  (`HoloSatSome`: the slots' types are the partition's, private, so the statement binds them existentially — for some
  hidden types — below `2^tLog`, the range the zero outputs pin), with
  probability at most `RecursiveSound`'s bound with candidate 3's error (`holoError`) as its inner term;
* (ii) zero knowledge: `RecursiveFirewallPrivate`, as in `PublicAudit`.

What it takes, against `PublicAudit`:

* the inner protocol `holoInnerAt (holoTys st hp) (holoCTs st hp) st hp` in place of `innerAt T CT st`: `holoTableCL` at
  `holoAt st hp`, the class, regions and schedule of `execClassAt st` with the sparse parameters `hp`, R's registered
  schedule `fast100`'s at `mReg` and R opened at `STACKED_CLAIMS = 2` claims, the two reps' sparse claims (a constant of
  the model, which `regPhase` opens, and of the error, `regOpenError`; no parameter the prover picks), its messages
  typed by the stages' shapes (`holoTys`, a function of `st` and `hp`) in place of a palette `T` of the prover's, and
  its coins by their coin kinds in the run's order (`holoCTs`, the link points and then `holoKinds`'s coins) in place
  of coin types `CT` of the prover's;
* R's registration, which says which table the audit is about, as `RecursiveAudit`'s `p`, `path₀` and `_hreg` say which
  private circuit: the record `reg`, the verifier of record's `--rec-inner-registration`, which `st`'s META binds by
  its digest (`metaRegistration st`, as the verifier reads it off `--rec-inner-circuit`, the file `st` is set up from),
  and R's cap `cap`, the record's `stacked` entry at explicit strings `bcs'` under `salt` (`_hstk`, `StackedTopAt`), a
  relation between public objects a user can check but for `salt`, the salts the prover's workers committed R's level-0
  top rows under, which they hold (`registrant-holds-openings`: the guarantee is applied at the held salts), so `st`
  fixes the cap up to `topFinder`'s collision. No registrant's
  openings are taken: the word is the one the prover's
  own runs open at R's level 0 against `cap` at each draw (`regWord`, the knowledge form of `note:41519fce`), on which
  two runs disagree only by a collision, read as an inner fork. The word's closeness to a codeword is no hypothesis: the
  statement holds only when some table within the unique-decoding radius of the word has a satisfiable hidden
  statement, so a far word is a false statement that the bound covers;
* the coin server's records of the inner session as a function of the history (`wr`: the inner transcript and session
  record it writes at each history, `--rec-inner-transcript` and `--rec-inner-session`), in place of fixed records: the
  event reads the records written at the outcome's history, and their custody (`_hIC`,
  `Flock.Premises.InnerRecordCustody`, `live-verifier` at the inner session) says they hold that history's commitments
  and coins where the verifier's map says each round's go (`holoMap`, from the session's schedule at the verifier's
  inner shape `ish`: a stream round's coins at the last history round it takes, read off the history's prefix), at the
  table's prefix in the inner transcript `tbl` and the `b ‖ c` a round's commitment gives a value's row `bc`;
* V*'s statements indexed by the verifier's own list (`vstarCount ish`, `RecVStar.statements`; gap 4(c)), in place of
  the formula `outerStatementsAt st`; the event adds the verifier's checks at candidate 3 of its own inputs
  (`AcceptsC3`): its list builds, it holds the inner transcript against `reg`, the covers check passes there (all of
  `RecVStar.checkCovers`: the sparse-presence, stacked-presence and stacked-shape refusals and each stream's rounds),
  and check 5's sparse refusals pass, so `ℓ` is in `[6, k_log]` and the unit range's slots are inside the block
  (`SparseRangeOk`). These are not the whole compiled acceptance (`AcceptsPublic` and `ChecksVR` carry the rest). The
  verifier's acceptance gives all of `AcceptsC3` (`FlockC3.VStarList.acceptsC3_of_verify`);
* R's `m`: Lean's verifier never compares META's `stacked.m` with the bucket's `table_m`; it holds R's stream and each
  `rec-opening` unit to META's shape, and the unit names come from the outer program it compiles (the list). The model
  takes R's `m` from the class (`mReg`), and VBridge (`_hbr`, V*'s circuits compute this inner verifier at the list's
  statements) carries the transitive link, which piece (h) derives from the list (`stackedShape_of_list`): derived, not
  a hypothesis of this statement;
* the inner fork's pick `clashPickHL` and the error `holoError (holoAt st hp)` in place of `clashPickCL` and
  `tableError`;
* the verifier's checks at candidate 3 of each session it accepts, in the event (`ChecksVR`): it holds the session's
  statements to V*'s statement of its index with its own reads (`RecVStar.checkStmt`, which runs `checkOwn`), their
  verifier-registered values to check 5's source at the session record (`coins`, `RecCoins.check`) and their public rows
  to what the inner transcript fixes (`RecVStar.checkPubRows`), both at the records written at the history; and
  VBridge, the forks and the bound at the typed sessions kept at a whole history where those checks pass (`checkedC3`,
  each refusing everything elsewhere, so VBridge is read at the histories `recGame` plays), whose games are the typed
  sessions';
* V*'s files parse to `Vs` only where the verifier's acceptance parses them into scope (`VStarParsesScoped`; #1847's
  `scopeOk` refusal), so `_hob` asks nothing of the writer's round trip.

Its proof is `Flock.SecurityProofs.RecursiveAuditC3` (`Proofs.Flock.Recursive.AuditC3Proof`), from gap 2 at the
word the runs open (`Recursive/RegExtract.lean`), which reads piece (a)'s runs (`FlockC3.Fork.holo_runs_sound`). Its
non-vacuity needs VBridgeCOff at candidate 3 (piece (b), `FlockC3.Compose`) and
V*'s statements over the verifier's list (piece (h)), which at candidate 3's inner shape is circuits' gap 2's eleven:
`rec-open`, `rec-residuals`, `rec-sparse`, R's levels (`rec-open-stacked`) and `rec-opening` at META's `stacked`.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Check 5's inputs at candidate 3**, as the verifier forms them (`RecSetup.c3`): the inner shape's RecSparse
shape (`--sparse`), R's shape, the inner shape's or else the one the inner transcript held R's stream to, and the
verifier's hold of the inner transcript `cs`. -/
def c3Of (ish : RecVStar.Inner) (cs : Option Flock.RecVStar.InnerCommits) : Option Flock.RecCoins.C3 := do
  let c ← cs
  let sp ← ish.sparse
  pure { sparse := sp, stacked := ish.stacked.or c.stacked, commits := c }

/-- **The verifier's hold of the inner transcript at a history**: `innerCommitsAt` against R's registration `reg`, at
the inner transcript and session record the coin server writes at the history (`wr`). -/
def innerHeld {Cm Coin : Type} (wr : List (Cm × Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (t : List (Cm × Coin)) : Option Flock.RecVStar.InnerCommits :=
  (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption

/-- **The verifier's checks at candidate 3 of one session**, after `buildSession` (`FlockVerify.lean`'s `setupOf`): each
statement the session sets up from `zkInputs I` and the record's draw is V*'s statement `j` under the bindings `f` at
the inner shape `ish`, with the verifier's own reads `own` and shared rows `rows` (`RecVStar.checkStmt`, which runs
`checkOwn`); its verifier-registered values are the ones check 5's source `coins` gives at the inner commits `cs`
(`RecCoins.check` at `c3Of`); its public rows are the ones the inner transcript fixes (`RecVStar.checkPubRows`); and
each table's statement labels its instances with the units its part of the record's draw proves
(`RecVStar.checkLabels`). `checkCovers` and `Zk.shapeOk` are not read here. -/
def ChecksC3 (ish : RecVStar.Inner) (cs : Option Flock.RecVStar.InnerCommits) (coins : Option Flock.RecCoins.Source)
    (j : ℕ) (f : VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (I : FlockSoundness.Discharge.Exec.Inputs) (own : Array Flock.Registered.Port)
    (t : FlockSoundness.Discharge.Exec.Transcript) : Prop :=
  ∃ spec sts, FlockSoundness.Discharge.Exec.setupOf (FlockSoundness.Discharge.ZkExec.zkInputs I) t.record =
      .ok (spec, sts) ∧
    sts.forM (Flock.RecVStar.checkStmt (some ish) I.tags I.tables · I.circuitFile I.partition I.program (some own)
      (some j) (some f) rows) = .ok () ∧
    sts.forM (Flock.RecCoins.check · (some own) coins (c3Of ish cs)) = .ok () ∧
    sts.forM (Flock.RecVStar.checkPubRows · (some own) cs ish.sparse) = .ok () ∧
    Flock.RecVStar.checkLabels (FlockSoundness.Discharge.Exec.recordDraw t.record) I.count sts = .ok ()

/-- **The verifier's checks at candidate 3 of every one of V*'s sessions**, as the prover supplies them: session `j` as
V*'s statement `j` under `bind j` (`ChecksC3`), at the registration the outcome records. -/
def ChecksVR {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (ish : RecVStar.Inner)
    (cs : Option Flock.RecVStar.InnerCommits) (coins : Option Flock.RecCoins.Source) (bind : Fin S → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port)) (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  ∀ j : Fin S, ChecksC3 ish cs coins j (bind j) rows (zr o.1 j).I (zr o.1 j).own (o.2 j).2.2.2

open Classical in
/-- **A session kept where an accepted transcript passes `P`**: `z` where some transcript `z` accepts passes `P` at its
inputs and own list, and elsewhere `z` with the own list `freshOwn`, which accepts nothing (`ZkOuter.refuse`'s pattern,
`not_acceptsZKR_fresh`). Its game is `z`'s. -/
noncomputable def keepOwn {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg')
    (P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop) : Q.Outer Reg' :=
  { z with
    own := if ∃ t, AcceptsZKR z.I z.own t ∧ P z.I z.own t then z.own else freshOwn Q.c
    hown := fun ω R' o h => by
      by_cases hp : ∃ t, AcceptsZKR z.I z.own t ∧ P z.I z.own t
      · rw [ite_eq_left hp] at h ⊢
        exact z.hown ω R' o h
      · rw [ite_eq_right hp] at h
        exact (not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH _ h).elim }

/-- **V*'s checked sessions**: the typed sessions (`custodied`), each kept at a whole history `t` (a coin for each of the
inner protocol's rounds, the only histories `recGame` plays) where an accepted transcript passes the verifier's checks
at candidate 3 as V*'s statement of its index (`ChecksC3`) at its hold of the inner transcript and check 5's source at
`t` (`cs t`, `coins t`), and refusing everything elsewhere (`keepOwn`). Their games are the typed sessions', so the
transported prover plays them (`checkedT`). -/
noncomputable abbrev checkedC3 {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B D : Type} {I : Inner} {Cm : Type}
    (ish : RecVStar.Inner) (cs : List (Cm × I.Coin) → Option Flock.RecVStar.InnerCommits)
    (coins : List (Cm × I.Coin) → Option Flock.RecCoins.Source)
    (bind : Fin S → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (zr : A → B → List (Cm × I.Coin) → D → (j : Fin S) → (Vs j).OuterR Reg') (hC : Flock.Premises.RecordCustody zr)
    (h : Typed zr) : A → B → List (Cm × I.Coin) → D → (j : Fin S) → (Vs j).Outer Reg' :=
  fun R ω t R₂ j => keepOwn (custodied zr hC h R ω t R₂ j) fun J own tr =>
    I.r ≤ t.length ∧ ChecksC3 ish (cs t) (coins t) j (bind j) rows J own tr

/-- **The prover's strategy against the checked sessions**: its strategy against the typed sessions (`transported`),
whose games the checked sessions' are. -/
noncomputable def checkedT {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ} (ish : RecVStar.Inner)
    (cs : List (Cm × I.Coin) → Option Flock.RecVStar.InnerCommits)
    (coins : List (Cm × I.Coin) → Option Flock.RecCoins.Source) (bind : Fin S → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port))
    (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hC : Flock.Premises.RecordCustody zr) (h : Typed zr)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N)) :
    Strategy (recGame L Reg I Cm fun R ω t => outerV (checkedC3 ish cs coins bind rows zr hC h R ω t) dg N) :=
  transported zr hC h σ

/-- **A run of the prover opens a position of R's level 0**: at the draw `ω`, some outcome `σ` reaches (an inner
transcript and V*'s outer proof at its leaf) decodes (`decodedV`) to an inner transcript whose steps play candidate 3's
table at the cap `cap` (`outHL`, which reads only its statement's cap) to an outcome with an opening of R's level-0
position `pos` to the row `col` that verifies against `cap`. -/
def RunOpensReg {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    (st : Flock.Stmt) (hp : HoloParams) {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) (pos : ℕ) (col : List FlockLevel3.GF128) : Prop :=
  ∃ o : Reg × L.Ω × List (Cm × (holoInnerAt T CT st hp).Coin) × (Reg₂ × VRec Vs Reg' dg N),
    Game.Outcome (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
    o.2.1 = ω ∧ ∃ out : OutHL FlockLevel3.GF128 Refine.Dig (List UInt8),
      outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)
          (HoloReg.mk (fun _ _ => 0) cap) (decodedV zs dg N k Rw wmsg o.1 o.2.1 o.2.2.1 o.2.2.2) = some out ∧
        ∃ y ∈ slotOpensL (out.reg cap) 0 0 (holoAt st hp).rsch.sch.level₀.queries, y.1 = pos ∧
          VerifiedL Refine.H512 Refine.enc512 Refine.leafPlain cap (holoAt st hp).rsch.sch.level₀.lanes
            (pathLen (holoAt st hp).rsch.sch.level₀) y ∧ y.2.1 = col

open Classical in
/-- **The word the prover's runs open at R's level 0** at the draw `ω`: at each position, lane by lane, the row some
run opens there against `cap` (`RunOpensReg`), one row for all lanes, and zero where no run opens the position. Two
runs that open one position to different rows against one cap are a collision of SHA-512, so every run opens this
word, up to an inner fork. -/
noncomputable def regWord {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) : Oracle FlockLevel3.GF128 := fun ℓ pos =>
  if h : ∃ col, RunOpensReg T CT st hp zs dg N k Rw wmsg σ cap ω pos col then h.choose.getD ℓ 0 else 0

/-- **The inner statement the prover's runs make**: at each draw, the word they open at R's level 0 (`regWord`) and
R's cap. -/
noncomputable def regStmt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) : Reg → L.Ω → HoloReg FlockLevel3.GF128 Refine.Dig :=
  fun _ ω => HoloReg.mk (regWord T CT st hp zs dg N k Rw wmsg σ cap ω) cap

/-- **`RunOpensReg` at the typing `τv`**: the same opening fact with the runs decoded against candidate 3's table at
`τv` (`holoAtτ`, whose every field but the typing is `holoAt st hp`'s own term), and the sessions, the decoder and the
prover typed at that table's protocol (`holoInnerAtτ`). The guarantee reads it only inside its existential over `τv`,
so no statement of it ever compares two typings' protocols. -/
def RunOpensRegτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) {n : ℕ} {L : Law n}
    {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAtτ T CT st hp τv).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) (pos : ℕ) (col : List FlockLevel3.GF128) : Prop :=
  ∃ o : Reg × L.Ω × List (Cm × (holoInnerAtτ T CT st hp τv).Coin) × (Reg₂ × VRec Vs Reg' dg N),
    Game.Outcome (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
    o.2.1 = ω ∧ ∃ out : OutHL FlockLevel3.GF128 Refine.Dig (List UInt8),
      outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAtτ st hp τv)
          (HoloReg.mk (fun _ _ => 0) cap) (decodedV zs dg N k Rw wmsg o.1 o.2.1 o.2.2.1 o.2.2.2) = some out ∧
        ∃ y ∈ slotOpensL (out.reg cap) 0 0 (holoAtτ st hp τv).rsch.sch.level₀.queries, y.1 = pos ∧
          VerifiedL Refine.H512 Refine.enc512 Refine.leafPlain cap (holoAtτ st hp τv).rsch.sch.level₀.lanes
            (pathLen (holoAtτ st hp τv).rsch.sch.level₀) y ∧ y.2.1 = col

open Classical in
/-- **`regWord` at the typing `τv`**: the word the runs open with their outcomes decoded at `τv`'s table
(`RunOpensRegτ`). -/
noncomputable def regWordτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) {n : ℕ}
    {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAtτ T CT st hp τv).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) : Oracle FlockLevel3.GF128 := fun ℓ pos =>
  if h : ∃ col, RunOpensRegτ T CT st hp τv zs dg N k Rw wmsg σ cap ω pos col then h.choose.getD ℓ 0 else 0

/-- **`regStmt` at the typing `τv`**: at each draw, the word the runs open at `τv`'s table (`regWordτ`) and R's cap. -/
noncomputable def regStmtτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) {n : ℕ}
    {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAtτ T CT st hp τv).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) : Reg → L.Ω → HoloReg FlockLevel3.GF128 Refine.Dig :=
  fun _ ω => HoloReg.mk (regWordτ T CT st hp τv zs dg N k Rw wmsg σ cap ω) cap

/-- **The inner protocol's message types at candidate 3**, a message's index naming its type: those of the stages'
shapes at `F = GF128`, `K = GF256` and R's digests, which read `st` and `hp` only through the two Ligerito schedules'
`ext` and the sparse shape. In order: the level-0 cap, the accept bit, and the openings with their salts; the
zerocheck's `(round1_ab, round1_c)`, its and the lincheck's pairs, and the lincheck's `Fin 64` vectors; the ring
switches'; Ligerito's single elements and `K` pairs, and its final messages at the circuit table's schedule and at R's;
the sparse phase's rounds at degrees `1`, `2ℓ` and `2ℓ + 1`, and its `2^pLog` and `2^tLog` values. -/
noncomputable def holoTys (st : Flock.Stmt) (hp : HoloParams) : Fin 16 → Type :=
  ![FlockSoundness.Model.Cap Refine.Dig, Bool,
    FlockSoundness.Model.Opens FlockLevel3.GF128 Refine.Dig × FlockSoundness.Model.Salts (List UInt8),
    (Fin 64 → FlockLevel3.GF128) × (Fin 64 → FlockLevel3.GF128), FlockLevel3.GF128 × FlockLevel3.GF128,
    Fin 64 → FlockLevel3.GF128, Fin 128 → FlockLevel3.GF128, FlockLevel3.GF128,
    FlockLevel3.GF256 × FlockLevel3.GF256, Fin (2 ^ ((holoAt st hp).sch.ext + 1)) → FlockLevel3.GF128,
    Fin (2 ^ ((holoAt st hp).rsch.sch.ext + 1)) → FlockLevel3.GF128,
    FlockLevel3.GF128 × (Fin 1 → FlockLevel3.GF128),
    FlockLevel3.GF128 × (Fin (2 * (holoAt st hp).cls.shape.ell) → FlockLevel3.GF128),
    FlockLevel3.GF128 × (Fin (2 * (holoAt st hp).cls.shape.ell + 1) → FlockLevel3.GF128),
    Fin (2 ^ (holoAt st hp).cls.shape.pLog) → FlockLevel3.GF128,
    Fin (2 ^ (holoAt st hp).cls.shape.tLog) → FlockLevel3.GF128]

/-- **Some hidden types make a table near the word satisfiable**: `holoInnerCL`'s `holds` with the slots' types
existentially bound — for some hidden types — in place of the parameter `hp.τ`, which the verifier does not hold
(partitions are private). The types' bound is the binder's range `Fin (2^tLog)`, which the run carries through the
zero outputs: the sparse unit lays each slot's type out as 16 output bits with the bits from `tLog` up pinned to zero
(`RecSparse.steps`' stacked tail, check 5's `rec-sparse-zero` rows), so the typing any accepted run registers is
below `2^tLog`. -/
def HoloSatSome (st : Flock.Stmt) (hp : HoloParams) (x : HoloReg FlockLevel3.GF128 Refine.Dig) : Prop :=
  ∃ (τ : Fin (2 ^ (holoAt st hp).cls.sLog) → Fin (2 ^ (holoAt st hp).cls.tLog))
    (R : RegTable (holoAt st hp).cls),
    (holoAt st hp).rsch.Close execArith x.w R ∧ ∃ z, (R.toStatement τ (holoAt st hp).regs).Satisfies z

/- `holoAt` stays applied, never unfolded, while the statement elaborates: the statement's binders are typed at
`schedInner`'s parameter-level `Coin` and `Msg` and the game at `rounds` literally, and every application inside the
existential over `τv` is explicitly at that typing's protocol (`holoInnerAtτ`), so every defeq the statement forces is
one protocol against a literal — never two typings' protocols against each other, which would make the unifier and the
kernel unfold `holoTableCL`'s body at every failed congruence. The attribute makes any accidental crossing fail fast. -/
attribute [local irreducible] holoAt

/-- **The recursive audit at candidate 3's soundness half, at coin types `CT` and the record map `im`**
(`RecursiveAuditC3` reads it at the inner session's coin kinds, `holoCTs`, and the verifier's map, `holoMap`).

For every partition object `obj` and program `prog` of the verifier's own, every inner statement `st` and sparse
parameters `hp`, R's registration (the verifier of record's record `reg`, which `st`'s META binds by its digest, and R's
cap `cap`, the record's `stacked` entry `_hstk` under `salt`, the salts the prover's workers committed R's level-0 top
rows under, which they hold: `registrant-holds-openings`, and the guarantee is applied at the held salts), the coin
server's records of the inner session at each history (`wr`), check 5's source at the session record (`coins`) and
the inner shape `ish`, and every prover `σ` of the recursive audit
with candidate 3's compiled table as its inner protocol (`holoInnerAt`, its messages of the stages' types `holoTys`
and its coins of types `CT st hp`),
at the law of the registration's draws
(`drawLaw`): V*'s statements `Vs`, one for each of the verifier's list (`vstarCount ish`), V*'s sessions `zr R ω t R₂ j`
as the prover supplies them (`ZkOuterR`), their records' custody `_hCust`, their registrants' records covering the
rows their reads files name `_hRC`, the table's prefix `tbl` and the `b ‖ c`
`bc` at which the map `im` says where each round's commitment and coins go in the inner session's records, and those
records' custody `_hIC` at `im`; check 5's source at each record is that record's inner-session files (`_hcoins`), the
verifier of record's own inputs (V*'s bindings
`bind` and the shared rows `rows`; session `j` runs as V*'s statement `j`), at which V*'s statements' files, where they
parse into scope as the verifier's acceptance parses them, parse to `Vs` (`VStarParsesScoped`), every session's inputs
carrying the tags their name names (`TagsFound`), the messages V*'s layers hold and the rounds' commitments (`wmsg`,
`Sk`, `cmt`, `msg`). Taken: A3 (`_hA3`).
Then for every count curve `(k, R_w, M, ρ)` that bounds every session's rate at each setup of its inputs, given that
at some typing `τv` of the slots (`_hbr`'s existential, below `2^tLog` — the discharge takes the registered one, which
the zero outputs pin to that range; `holoAtτ`)
V*'s circuits compute candidate 3's inner verifier at that typing at the checked sessions (`checkedC3`: the typed
sessions at a whole
history where the verifier's checks at candidate 3 pass) and the word the transported prover's runs open at R's level 0
against `cap` (`regStmt`), at the rounds before the last, off a registrant's row off a value the verifier registers
itself at its rows `valueSrcAll` (`ValueColl`), off a session's registered read whose leaf is not the table's by a
collision (`LeafColl`) and off a session's read of R's level-0 top other than its row under the held salts by a
collision (`SaltColl`) (VBridge at the committed rounds, `Flock.Premises.VBridgeCOff`):
Lean's verifier, at the records written at the outcome's history, never builds `st`, accepts its inputs at candidate 3
and accepts every session as V*'s statement of its index with its checks at candidate 3;
or the sessions type and there is, at some typing `τv`, a registrant fork, a held top other than `bcs'` (`topFinder`),
a registrant's row
off a value the verifier registers itself, at its rows and salts (`valueFinder` at `valueSrcAll`), a session's
registered read off the table's leaf (`leafFinder`), a session's read of R's level-0 top other than its row under the
held salts (`saltFinder`), a round fork, a session fork, or an inner fork of the transported prover's own runs at the
checked sessions whose decoded transcripts play candidate 3's table at `τv` to outcomes that clash against each other
(`clashPickHL`); or, with `w` the word the transported prover's runs open at the draw (`regWord`, at `τv`),
$$\Pr[\text{Lean's verifier builds } st \wedge \text{it accepts its inputs at candidate 3} \wedge \text{every session
  verifies as V*'s statement of its index and passes its checks at candidate 3} \wedge \neg\,\exists \tau\,
  \exists R \text{ near } w,\ R\text{'s hidden statement at } \tau \text{ holds}]$$
is at most the checked sessions' expected statistical errors under the full draw's model, `r` times their expected stage
slack, and candidate 3's error.

At a private inner shape (`ish.holo`), `RecVStar.statements` refuses (`statements_holo`), so no run the verifier accepts
has a private inner session, and the guarantee says nothing about one until V*'s holo parts land (#1896). -/
def RecursiveAuditC3At {r : Flock.Stmt → HoloParams → ℕ}
    (CT : (st : Flock.Stmt) → (hp : HoloParams) → Fin (r st hp) → Type) [∀ st hp k, Fintype (CT st hp k)]
    [∀ st hp k, Inhabited (CT st hp k)]
    (im : (st : Flock.Stmt) → (hp : HoloParams) → (Cm : Type) → RecVStar.Inner → String →
      (Cm → String → Nat → List UInt8) →
        FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin (r st hp)) → CT st hp k)) :
    Prop :=
  ∀
    -- the registration's draws, a finite space with an element, and the units each draws
    {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] {draw : Ω → Finset (Fin n)} {Reg Cm : Type} {Reg' Reg₂ : Type}
    -- the verifier's own partition object and program, the inner statement and the sparse phase's parameters
    (obj : Lean.Json) (prog : ByteArray) (st : Flock.Stmt) (hp : HoloParams)
    -- the verifier of record's own inputs: R's registration (`--rec-inner-registration`, the record `st`'s META binds),
    -- and the coin server's records of V*'s inner session at each history (`--rec-inner-transcript` and
    -- `--rec-inner-session`)
    (reg : Lean.Json) (wr : List (Cm × ((k : Fin (r st hp)) → CT st hp k)) → Lean.Json × String)
    -- check 5's source at the inner session's record (`--rec-inner-coins` with it, or the inner session's files), the
    -- verifier's own
    (coins : String → Option Flock.RecCoins.Source)
    -- R's cap, the salts the prover's workers committed R's level-0 top rows under, which they hold
    -- (`registrant-holds-openings`: the guarantee is applied at the held salts), and the explicit commit strings of the
    -- record's `stacked` entry
    (cap : Cap Refine.Dig) (salt : ℕ → ByteArray) (bcs' : Array ByteArray)
    (_hstk : StackedTopAt (holoAt st hp).rsch.sch.level₀ st reg cap salt bcs')
    -- the verifier of record's inner shape (`--rec-inner`), and V*'s statements over its list, as the prover supplies
    -- their sessions
    (ish : RecVStar.Inner) {Vs : Fin (vstarCount ish) → VStmt}
    (zr : Reg → (drawLaw draw).Ω → List (Cm × ((k : Fin (r st hp)) → CT st hp k)) → Reg₂ →
      (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg')
    -- the records' custody
    (_hCust : Flock.Premises.RecordCustody zr)
    -- the registrant's records cover the rows the sessions' reads files name
    (_hRC : Flock.Premises.RegistrantCovers zr)
    -- the table's prefix in the inner transcript and the `b ‖ c` a round's commitment gives a value's row, and the
    -- custody of the inner session's records where the map says each round's commitment and coins go
    (tbl : String) (bc : Cm → String → Nat → List UInt8)
    (_hIC : Flock.Premises.InnerRecordCustody wr (fun w => innerCommitsAt w.1 w.2 st reg) ish.sparse
      (im st hp Cm ish tbl bc))
    (_hcoins : ∀ r src, coins r = some src → ∃ f, src = .files f ∧ f.record = r)
    -- the verifier of record's own inputs: V*'s bindings for each statement and the shared rows
    (bind : Fin (vstarCount ish) → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    -- obligation (gap 4(c)): where V*'s statement `j`'s file under `bind j` parses into scope, as the verifier's
    -- acceptance parses it, it parses to `Vs j`'s circuit
    (_hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    -- obligation (our code): every session's inputs carry the tags their name names
    (_hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover, at the audit's game written out (`recGame` at the inner protocol's parameter-level coin
    -- type and round count), so its type reads no typing's protocol
    (σ : Strategy (Game.send Reg fun R => Game.coin (drawLaw draw).Ω fun ω =>
      rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
        (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)))
    -- the messages V*'s layers hold, the rounds' commit strings and commitments
    (wmsg : ((j : Fin (vstarCount ish)) → (Vs j).Bits) → List (List (SMsg (holoTys st hp))))
    (Sk : ℕ → (j : Fin (vstarCount ish)) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv →
      (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (vstarCount ish)) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      List (SMsg (holoTys st hp)))
    -- A3
    (_hA3 : Flock.Premises.UniformRandomBytes Vs)
    -- the count curve, bounding every session's rate
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S₀
      (d₀ : Lean.Json) (y₀ : ZkHidden.DrawSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    -- VBridge at the committed rounds, a premise: at some typing `τv` of the slots — its discharge takes the
    -- registered one, which the zero outputs pin below `2^tLog` — V*'s circuits compute candidate 3's inner verifier
    -- at that typing at the rounds before the last, which it does not commit, at the checked sessions and the word
    -- the prover's runs open, off a registrant's row off a value the verifier registers itself, at the verifier's
    -- rows (`valueSrcAll`), off a read's leaf off the table's, and off a read of R's level-0 top off its node's row
    -- under the held salts
    (_hbr : ∀ h : Typed zr, ∃ τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog),
      Flock.Premises.VBridgeCOff
        (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv)
        (fun R ω t R₂ =>
          ValueColl (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h)
            (valueSrcAll (fun t => coins (wr t).2) fun t => c3Of ish (innerHeld wr st reg t)) R ω t R₂ ∨
          LeafColl (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) R ω t R₂ ∨
          SaltColl (holoAt st hp).rsch.sch.level₀ cap salt
            (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h) R ω t R₂)
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg
        N k Rw (regStmtτ (holoTys st hp) (CT st hp) st hp τv
          (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
          (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap)
        wmsg Sk cmt msg),
    prob (fun o => (AcceptsPublic obj prog st ∧ AcceptsC3 ish st (wr o.2.2.1).1 (wr o.2.2.1).2 reg) ∧
          VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ChecksVR ish (innerHeld wr st reg o.2.2.1) (coins (wr o.2.2.1).2) bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2)
        (Game.send Reg fun R => Game.coin (drawLaw draw).Ω fun ω =>
          rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
            (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)) σ = 0 ∨
    ∃ (h : Typed zr) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)),
      (regFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (topFinder wr st reg bcs' (holoAt st hp).rsch.sch.level₀ cap salt _hstk
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (valueFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (valueSrcAll (fun t => coins (wr t).2) fun t => c3Of ish (innerHeld wr st reg t)) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (leafFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (saltFinder (holoAt st hp).rsch.sch.level₀ cap salt
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      RoundFork (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ) cmt ∨
      SessionFork (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw M
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ) ∨
      InnerForkRec (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)
        (regStmtτ (holoTys st hp) (CT st hp) st hp τv
          (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
          (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap)
        wmsg Refine.H512
        (clashPickHL (holoTys st hp) (CT st hp) execArith Refine.H512 Refine.enc512 Refine.leafPlain
          (holoAtτ st hp τv)) ∨
      prob (fun o => (AcceptsPublic obj prog st ∧ AcceptsC3 ish st (wr o.2.2.1).1 (wr o.2.2.1).2 reg) ∧
            VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ChecksVR ish (innerHeld wr st reg o.2.2.1) (coins (wr o.2.2.1).2) bind rows (zr o.1 o.2.1 o.2.2.1)
              o.2.2.2 ∧
            ¬ HoloSatSome st hp
              (regStmtτ (holoTys st hp) (CT st hp) st hp τv
                (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
                (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap o.1 o.2.1))
          (Game.send Reg fun R => Game.coin (drawLaw draw).Ω fun ω =>
            rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
              (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)) σ ≤
        expAt (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) (fun ω t s =>
            sumV (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t) dg N
            (fun R₂ j τ =>
              (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t R₂
                j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((r st hp + 1 : ℕ) : ℝ≥0∞) *
            expAt (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ) (fun ω t s =>
              sumV (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t) dg N
              (fun R₂ j τ =>
                (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t R₂
                  j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp))

/-- **The recursive audit at candidate 3**: (i) soundness in fork form, `RecursiveAuditC3At` at the inner session's
coins of their coin kinds in the run's order (`holoCTs`: the link points, then the coins of the stages' shapes,
`holoKinds`) and the verifier's map of its records (`holoMap`: the session's schedule at the verifier's inner shape,
each stream round's coins read off the history's prefix at the last round it takes), and (ii) zero knowledge,
`RecursiveFirewallPrivate`, as in `PublicAudit`. -/
def RecursiveAuditC3 : Prop :=
  RecursiveAuditC3At holoCTs holoMap ∧ RecursiveFirewallPrivate

end Flock.Guarantees
