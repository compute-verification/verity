import Proofs.Flock.Claims.Recursion.VStarBound
import Proofs.Flock.Claims.Recursion.DeployPremises

/-!
# `KSBound` at the deployment (option (d))

`Claims.Recursion.KSBound` takes the compiled record as a function of the inner verifier its inner view runs and the
bound record (`Claims.Recursion.Bound`). Here both are pinned to the deployment:

* **The compiled record** is `deployRec IV ivScope b` at each inner verifier `IV`, its inner view running `IV`
  (`recBoundAt`, a `Claims.Recursion.RecAt`), applied at `IVStar` (`innerView_ksBound`). Its inner scope check is the
  deployed plain parser's, and its readers are V*'s bound layout's (`LayoutB.readers`: every round off the parts,
  `VStarB.HeldB`). A compiled record that ignores the inner verifier it is given is no `RecAt` (`RecAt.not_const`).
* **The element-level inner Flock `DE`** is the deployed plain record at plain leaves as the claims read it,
  `deployPlainTrim rel` (columns and code trimmed of trailing zeros), with only three fields replaced: its verifier's
  steps (at the deployed plain verifier's rounds, `FlockDeploy.V false`'s `k`), its columns and its weight (`deE`). Its
  parser, scope, tables, code, decoder and numbers are `deployPlainTrim`'s, so the inner decoder `KSBound` quantifies
  over meets the spec where `deployPlainTrim`'s does. At the deployed plain verifier's own steps, columns and weight,
  `DE` is `deployPlainTrim rel` (`deE_plainTrim`), so (P1) there is `plain_KSAbort`'s statement (`plainKSAbort_bd`).
* **The bound record** (`bdDeployed`) answers a round with its coin's bytes as `FlockDeploy.next` does, counts the
  schedule's commit strings (`nAtSched`, off `rec_live.rep_rounds`), and opens at the pinned hm96 key
  (`Hm96.Default512`, as `deployRec`'s). Its rows' digest input, how its last message reads and the element-level
  verifier's steps stay parameters: they are the code side's, which has no element-level verdict yet.

**The guarantee** is `KSBoundDeployed`, and its ∀-form is `KSBoundRoute` (proved: `RecDeploy.ksBoundRoute`): at every
choice the code side makes, the premises P1–P4 at the deployed bound record, Flock's KS at the outer record, and the
code facts of the staging, V*'s parts and the readers at the bound layout (`VStar.RunPairs`; `VStarB.HeldB`,
`VStarB.VstarShowsB`, `VStarB.PartChecksB`, `VStarB.PartOpensB`, `VStarB.DecompB`; `ReadersTrue`, `ReadersLogColl` and
`ReadersRunPlayed` at `LayoutB.readers`) give `KSBoundDeployed`.

**What reads the last message.** `IVStar` and the opened view read it only through `final`
(`Bound.ivstar_congr_final`, `Bound.openView_congr_final`), so (P4, `Bound.LastBinds`) is stated modulo `final`.

Statements only, with those three lemmas and (P1) at the deployed plain verifier's own steps (`plainKSAbort_bd`).
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **V\*'s bound layout as the compiled verifier's readers**: the inner messages and each part's calls, off the parts'
witnesses. -/
def _root_.Claims.Recursion.VStarB.LayoutB.readers (L : Claims.Recursion.VStarB.LayoutB) : RecDeploy.Readers :=
  ⟨L.msgs, L.calls⟩

/-! ## The schedule's commit strings -/

/-- A rep's schedule at a table of the session (`rec_live.rep_rounds`, `Flock.RecVStar.repRounds`): at the table's `m`
and `k_log` and the session's claims, `2 + points · regions` as `FlockDeploy.repRounds` counts them; empty off
`fast100`. -/
def repSchedAt (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) : List _root_.Flock.RecVStar.SchedRound :=
  ((_root_.Flock.RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size)).map Array.toList).getD []

/-- A table's level-0 top's nodes, `2^{cap₀}` at `fast100`'s first level (`2^7 = 128` at its 218 queries). -/
def level0Nodes (st : _root_.Flock.Stmt) : ℕ :=
  2 ^ ((_root_.Flock.fast100 st.m).map fun s => (s.levels.getD 0 default).cap).getD 0

/-- **The schedule's commit strings at each inner round before the last** (`Bound.nAt` at the deployment), at an inner
statement the deployed plain verifier sets up. Round 0, the link, carries each table's level-0 top, one string a cap
node. Round `1 + i`, the `i`-th recorded round (each table's two reps in turn, each rep's rounds in order), carries the
salted tops it commits, one a cap node, then one row where it absorbs a message. A rep's copies of level 0's top are
round 0's strings again, not new ones. None at a statement that doesn't set up. -/
def nAtSched (x : Msg) (j : ℕ) : ℕ :=
  match FlockDeploy.setupOf false x with
  | none => 0
  | some (_, spec, sts) =>
    if j = 0 then (sts.toList.map level0Nodes).sum
    else
      match (sts.toList.flatMap fun st => repSchedAt spec st ++ repSchedAt spec st)[j - 1]? with
      | some r => r.2.1.toList.sum + if r.2.2.2 then 1 else 0
      | none => 0

/-! ## The pinned records -/

/-- **The element-level inner Flock, pinned to the deployed plain record the claims read**: `deployPlainTrim rel` (the
trimmed record `plain_KSAbort` and `ksA_deployed` take) with only its verifier, its columns and its weight replaced.
The verifier keeps the deployed plain verifier's rounds (`FlockDeploy.V false`'s `k`); its steps (`nextE`) read
element-level messages. At the deployed plain verifier's own steps, columns and weight it is `deployPlainTrim rel`
(`deE_plainTrim`). -/
noncomputable def deE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  { FlockDeploy.deployPlainTrim rel with V := ⟨(FlockDeploy.V false).k, nextE⟩, cols := colsE, weight := wE }

/-- **The deployed bound record**: `DE` pinned (`deE`), each round before the last answered by its coin's bytes, the
schedule's commit strings (`nAtSched`) and the pinned hm96 key's salts, salt prefix and salt mask. The rows' digest input
(`rowIn`), how the last message reads (`final`), how a round's rows join (`join`) and `γ`'s weight at `IVStar` (`w`)
are the code side's. -/
noncomputable def bdDeployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace where
  DE := deE rel nextE colsE wE
  ans c := c.bytes
  join := join
  nAt := nAtSched
  saltLen := _root_.Flock.Hm96.sha512.saltBytes
  saltPrefix := Claims.toMsg _root_.Flock.Hm96.Default512.saltPrefix
  mask y i := (_root_.Flock.Hm96.Default512.mask ⟨Array.ofFn y⟩).get! i.val
  rowIn := rowIn
  final := final
  weight := w

/-- **The compiled record at an inner verifier, at the deployment**: `deployRec` with the deployed plain parser's
scope check for the inner statement and V*'s bound layout's readers. -/
noncomputable def recBound (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) : Claims.Recursion.Deploy Coin :=
  deployRec IV (FlockDeploy.deployPlain rel).scopeCheck { b with rd := L.readers }

/-- **The compiled record at each inner verifier, at the deployment** (`KSBound`'s `RecAt`): `recBound`, whose inner
view runs the verifier it is given on V*'s bound layout's messages at the inner statement. -/
noncomputable def recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) : Claims.Recursion.RecAt Coin where
  record := recBound rel b L
  msgs := L.msgs
  xI := xIOf
  runs _ _ _ _ _ := rfl

/-- **`KSBound` at the deployment**, the guarantee option (d) states: at SHA-512, the compiled record `deployRec` whose
inner view runs `IVStar` at V*'s bound layout (`recBoundAt`), the deployed bound record (`bdDeployed`) and the deployed
outer `--zk` record (`deployZkRec`). At `γ`'s weight `wCover` at `IVStar` it is `KSBoundCover`. Its level
(`Claims.Recursion.lam`) is about 65 bits per SHA-512 call at the deployed shape (one inner table at `m = 35`, `γ`'s
weight at `IVStar` at `coverW`), below the 100-bit floor and the `2^-128` target under every weight. Route 2C's
bound λ₂ (`lamLin2`) evaluates to at least 100 at the deployed records (`RecDeploy.lamLin2_deployed_ge_dep`), but
`KSLin2` at the deployment's dense statements (`KSLin2On` at `RecDeploy.DenseAt`) only follows from premises still open
(`RecDeploy.ksLin2_deployed_open`), Holo statements being candidate 3's own model, and `BitsLin2` at those statements
follows from it (`bitsLin2On_of_ksLin2On`). -/
def KSBoundDeployed (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSBound FlockDeploy.sha512 (recBoundAt rel b L) (bdDeployed rel nextE colsE wE join rowIn final w)
    (deployZkRec relO)

/-- **The private variant at the deployment**, at the same records. -/
def KSPrivBoundDeployed (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSPrivBound FlockDeploy.sha512 (recBoundAt rel b L)
    (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO)

/-- **The compiled record's inner view runs `IVStar`**, at V*'s bound layout's messages and the run's first coins. -/
theorem innerView_ksBound (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128)
      Claims.Flock.PlainTrace) (H : Msg → Msg) (x : Msg) (r : Fin ((recBound rel b L Bd.IVStar).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    (recBound rel b L Bd.IVStar).innerView H x r ws =
      (Claims.run H (Claims.replay (L.msgs x ws)) Bd.IVStar (xIOf x) () fun i => Claims.coinSeq r i.val).seen :=
  rfl

/-- **`DE`'s rounds are the deployed plain verifier's.** -/
theorem deE_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    (deE rel nextE colsE wE).V.k = (FlockDeploy.V false).k :=
  rfl

/-- **The inner decoder meets the spec at the pinned `DE`** wherever it does at the deployed plain record the claims
read: `decOK` reads the code (`enc`, `len`, `θ`), which `deE` keeps. -/
theorem decOK_deE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    Claims.Recursion.decOK (deE rel nextE colsE wE) decI ↔
      Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI :=
  Iff.rfl

/-- **At the deployed plain verifier's own steps, columns and weight, `DE` is the trimmed plain record.** -/
theorem deE_plainTrim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    deE rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols
      (FlockDeploy.deployPlainTrim rel).weight = FlockDeploy.deployPlainTrim rel :=
  rfl

/-- **(P1) at the deployed plain verifier's own steps, columns and weight is `plain_KSAbort`'s statement**: from the
plain record's table bound with escapes on accepted runs (`TableBoundθOnAcc` at `deployPlain rel`), whatever the bound
record's other choices. -/
theorem plainKSAbort_bd (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (join : List Msg → Msg) (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols
      (FlockDeploy.deployPlainTrim rel).weight join rowIn final w).PlainKSAbort FlockDeploy.sha512 :=
  fun decI hdec => FlockDeploy.KS.plain_KSAbort rel htab decI hdec

/-! ## V*'s readers at the bound layout -/

/-- **The readers' rounds before the last are the run's** (a code fact about V*'s circuits and the readers, option (d)'s
`ReadersOpenPlayed`): at outer witnesses in the outer relation at the V* statements staged at inner and staging messages
`as` and coins `cs`, each part's staged as V\* stages it (`StagedPart`), whose recorded calls are true evaluations, at
each round before `IV`'s last that `IV`'s run of the readers' messages plays, the readers' message is the run's
(`as[j]`, the coin server's commit strings), or two recorded calls collide. `ReadersOpenPlayed` can't hold at
`IVStar`'s last round: its message, the openings, is never sent in the run, and `IVStar` checks the openings itself
(`Bound.opens`). -/
def ReadersRunPlayed {Col T : Type} (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
        (∀ j < (innerRunAt H IV b.rd x ws cs).msgs.length, j + 1 < IV.k (xIOf x) →
          ((innerRunAt H IV b.rd x ws cs).msgs.map Prod.fst).getD j [] = as.getD j []) ∨
        ∃ u v, Claims.findColl (callsOf b.rd x ws) = some (u, v) ∧ Claims.Collision H u v

/-! ## The guarantee's ∀-form -/

/-- **`KSBoundDeployed`'s ∀-form** (option (d), proved: `ksBoundRoute`): at the deployment's brings `b` and V*'s bound
layout `L`, at every choice the code side makes (the element-level verifier's steps `nextE`, its columns `colsE` and
weight `wE`, how a round's rows join, the rows' digest input, how the last message reads, `γ`'s weight at `IVStar`), the
deployed bound record `Bd` meets `KSBoundDeployed` given:

* (P1) to (P4) at `Bd`: `Bound.PlainKSAbort`, `Bound.Binds`, `Bound.WeightCovers`, `Bound.LastBinds`. (P1) stays a
  premise here: at an accept-all `nextE` it is false. At the deployed plain verifier's own steps, columns and weight it
  is `plain_KSAbort`'s statement (`plainKSAbort_bd`);
* Flock's KS at the deployed outer `--zk` record;
* the staging off the run: an accepted run's verifier messages show its coins, read by some `decC` (`VStar.RunPairs`);
* V*'s parts at `L`, at each of the three forms the compiled verifier takes (dense, holo, sparse), for some inner check
  `chk` each part computes, the coins each statement shows (`shows`) and the commit strings it shows (`cmt`):
  `VStarB.HeldB`, `VStarB.VstarShowsB` (the staging shows the coins `decC` reads and the run's commit strings),
  `VStarB.PartChecksB` and `VStarB.PartOpensB` at every part, and `VStarB.DecompB` at `IVStar`;
* V*'s readers at `L.readers`: `ReadersTrue`, `ReadersLogColl` (so the collisions P4, `DecompB` and the inner `B`
  produce are among the run's own recorded calls, which `Be` searches) and `ReadersRunPlayed`.

`VstarShowsB` with `RunPairs` ties `shows` to the run: without them a `shows` that holds nowhere makes `PartChecksB` and
`DecompB` hold at a `chk` that holds nowhere, at every inner verifier, even one that rejects everything, where `E*` never
extracts (note:proofs/20261019T0540Z-report-rec1-ksboundroute-counterexample). `PartOpensB` makes every holder of a
round hold the run's commit string there, which the readers read, so each part's check reaches the readers' messages.

`Bridge` is a lemma on the way, not a premise. At the deployed bound record (`bdCover`, `γ`'s weight `wCover` at
`IVStar`, ASK-72) (P3) holds and the route is `ksBoundRoute_cover`. Its level (`Claims.Recursion.lam`) is about 65 bits
per SHA-512 call at the deployed shape (one inner table at `m = 35`, `γ`'s weight at `IVStar` at `coverW`), below the
100-bit floor and the `2^-128` target under every weight (ASK-73). Route 2C's bound λ₂ (`lamLin2`) evaluates to at least
100 at the deployed records (`RecDeploy.lamLin2_deployed_ge_dep`), but `KSLin2` at the deployment's dense statements
(`KSLin2On` at `RecDeploy.DenseAt`) only follows from premises still open (`RecDeploy.ksLin2_deployed_open`), Holo
statements being candidate 3's own model, and `BitsLin2` at those statements follows from it (`bitsLin2On_of_ksLin2On`).
-/
def KSBoundRoute (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) : Prop :=
  ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
    let Bd := bdDeployed rel nextE colsE wE join rowIn final w
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.WeightCovers → Bd.LastBinds FlockDeploy.sha512 →
    Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    KSBoundDeployed rel relO b L nextE colsE wE join rowIn final w

end RecDeploy

namespace Claims.Recursion

/-- **`IVStar` reads the last message only through `final`**: two last messages `final` reads alike give one decision
at the same coins, with the same calls. -/
theorem Bound.ivstar_congr_final {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] (Bd : Bound Coin Col T) (x : Claims.Msg) (pre : List Claims.Msg) (a a' : Claims.Msg)
    (cs : List Coin) (hfin : Bd.final a = Bd.final a') :
    Bd.IVStar.next x (pre ++ [a]) cs = Bd.IVStar.next x (pre ++ [a']) cs := by
  simp [Bound.IVStar, hfin]

/-- **The opened view, whose columns `Bound.deploy` takes, reads the last message only through `final`.** -/
theorem Bound.openView_congr_final {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] (Bd : Bound Coin Col T) (ms : List (Claims.Msg × Claims.Out)) (a a' : Claims.Msg)
    (o out : Claims.Out) (p : List (List (Claims.Msg × (Fin Bd.saltLen → UInt8))) × Claims.Msg)
    (hfin : Bd.final a = some p) (hfin' : Bd.final a' = some p) :
    Bd.openView ⟨ms ++ [(a, o)], out⟩ = Bd.openView ⟨ms ++ [(a', o)], out⟩ := by
  simp [Bound.openView, hfin, hfin']

/-- **No `RecAt` ignores the inner verifier it is given**: its record at an accept-all verifier and at a reject-all one
differ in their inner views' decisions. So `KSBound` is never `KSFlock` at a compiled record fixed in advance, where its
inner record's verifier would be read only for its rounds. -/
theorem RecAt.not_const {Coin : Type} [Inhabited Coin] (D : RecAt Coin) (D₀ : Claims.Recursion.Deploy Coin) :
    ¬ ∀ IV, D.record IV = D₀ := by
  intro h
  have key : ∀ (IV : Claims.Verifier Coin) (r : Fin (D₀.V.k []) → Coin),
      D₀.innerView id [] r [] =
        (Claims.run id (Claims.replay (D.msgs [] [])) IV (D.xI []) () fun i => Claims.coinSeq r i.val).seen := by
    intro IV
    have hr := D.runs IV
    rw [h IV] at hr
    exact fun r => hr id [] r []
  let r : Fin (D₀.V.k []) → Coin := fun _ => default
  have h1 := congrArg Claims.Seen.out (key ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩ r)
  have h2 := congrArg Claims.Seen.out (key ⟨fun _ => 1, fun _ _ _ => .ret .rej⟩ r)
  have e1 : (Claims.run id (Claims.replay (D.msgs [] [])) ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩ (D.xI []) ()
      fun i => Claims.coinSeq r i.val).seen.out = Claims.Out.acc := rfl
  have e2 : (Claims.run id (Claims.replay (D.msgs [] [])) ⟨fun _ => 1, fun _ _ _ => .ret .rej⟩ (D.xI []) ()
      fun i => Claims.coinSeq r i.val).seen.out = Claims.Out.rej := rfl
  rw [e1] at h1
  rw [e2] at h2
  exact absurd (h1.symm.trans h2) (by decide)

end Claims.Recursion
