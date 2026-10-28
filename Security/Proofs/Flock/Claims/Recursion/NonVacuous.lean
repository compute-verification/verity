import Proofs.Flock.Claims.Recursion.VerdictAssembly
import Proofs.Flock.Claims.Recursion.VerdictLoop
import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.VerdictRun
import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.LayoutBTops
import Proofs.Flock.Claims.Recursion.DeployOuter
import Proofs.Flock.Claims.Recursion.BindsDeployed
import Proofs.Flock.Claims.KS.PlainR

/-!
# `RecursiveKS`'s premise at `nextR`, from the verdict's rows

The recursion's top statement. `RecursiveKS` takes one premise, that V\*'s parts decompose the inner
verifier (`DecompB` at `PartAccepts`). At the deployed reader `nextR` that premise is `RecKSDecompR` below, and
`recKSDecompR_of_targets` proves it from named targets, one per open obligation. Each target is a `def … : Prop` with a
fixed statement: a row's owner proves `theorem … : Target…` against it, and the integration replaces the argument with
that lemma. When no argument is left, `RecKSDecompR` holds outright and `RecursiveKS` at `nextR` is not vacuous.

**Where the rows are asked** (`At rd P x ms cs`): at an inner statement that a compiled statement read at the dense form
has (`DenseInner`), where the last step reads (`ReadOkR`), the four base checks hold (`setup`, `shaped`, `names`,
`after`) and the verdict's pieces hold (`P`, the parts' shares: whatever the pieces are, every target is monotone in
`P`). Each table's targets range over every decision `D : DecAt rd x ms cs`, every class `cls` and table `i`.

**The targets** (`acc` is rep `k`'s four accumulators, which the integration chooses once for every row):

* `TargetB` (row B): each rep's stream rounds hold its frames at `w`'s `T` (`FramedAt`).
* `TargetL` (row L): `RowL` at each rep.
* `TargetD` (row D): `rec-open` pieces, their sides and positions, holding, tied to the decision's side (`hside`) and
  the verifier rows' positions (`hpos`), and covering every query.
* `TargetR` (row R): S, its groups, the rounds' sizes, `w`'s size and every part's residuals vanishing.
* `TargetE` (row E): each rep's accumulator chain and position, with `UnitSums` and `ReadsEnds`.
* `TargetDecode` (the record): at a decision, the record value the verdict reads decodes (S1–S14).
* `TargetPublics` (the record): the session's public words are empty.

Rows G and B's `finds`, `roundsWF`, `noEmpty` are proved (`runs_assembled`), and so are C's sizes, shapes and points:
each rep's proof index is in range (`decR_fits`), shaped for its file (the `shaped` base check), and the session's link
points are the first coin's (`DecAt.points_eq`). Row A's step 3 is proved (`DecAt.verdict_of_runs`): the decision sets
up one table, which the spec names (`DecAt.tables_of_dense`), and every table's runs with row L make the verdict on the
decoded proofs accept.

**The pieces** (`Piece Q`): the parts' pieces `Q`, and one more, `dense`, which says the inner statement is a dense one.
Its owner's lowering fact (`hlowP`) holds at every dense compiled statement, and with it the verdict is only ever asked
at a dense inner statement.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR holdsR tabsOf pointsOf shapeAt)
open FlockVBridge.Family (FramedAt ProofShaped roundSizes)
open FlockVBridge.AccChain (UnitSums ReadsEnds)
open Flock.RecAlgebra (verifierRows valOf structureOf)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)
open Claims.Recursion.VStarB
open RecDeploy.Rows (DecAt)

noncomputable section

namespace Top

/-! ## Where the rows are asked -/

/-- **A dense inner statement**: the inner statement of a compiled statement read at the dense form. -/
def DenseInner (x : Msg) : Prop :=
  ∃ (xo : Msg) (sto : Claims.Recursion.Stmt), parse xo = some sto ∧ Claims.Recursion.Form.Dense sto.form ∧
    x = xIOf xo

/-- **Where the rows are asked**: at a dense inner statement, where the last step reads, the four base checks hold and
the verdict's pieces hold (`P`). -/
def At (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop) (x : Msg) (ms : List Msg)
    (cs : List Coin) : Prop :=
  DenseInner x ∧ ReadOkR rd x ms cs ∧ (∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) ∧ P x ms cs

/-! ## The targets -/

section Targets

variable (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop)
  (acc : Msg → List Msg → List Coin → ℕ → Array F128)

/-- **Row B's target, the frames**: each rep's stream rounds hold its frames at `w`'s batched value `T`. -/
def TargetB : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size → ∀ r : Fin 2,
      FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st (D.files cls i).sch
        ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
        (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
        (valOf (D.sh cls i) (D.rep cls (acc x ms cs) i r).w (D.rep cls (acc x ms cs) i r).comb
          (D.rep cls (acc x ms cs) i r).cons .T))

/-- **Row L's target**: the fields the reader fixes, at each rep. -/
def TargetL : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size → ∀ r : Fin 2, D.RowL cls (acc x ms cs) i r

/-- **Row D's target, the openings**: `rec-open` pieces that hold, whose sides are the decision's (`hside`), whose
positions are the verifier rows' (`hpos`), and which cover every query of every level of each rep. -/
def TargetD : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size →
      ∃ (pieces : List OpenPiece) (sideAt : Msg → List Msg → OpenPiece → OpenSide)
        (posAt : Msg → List Coin → OpenPiece → ℕ),
        (∀ q ∈ pieces, openHolds sideAt posAt q x ms cs) ∧
        (∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r) ∧
        (∀ r : Fin 2, ∀ v info kappa,
          verifierRows (D.sh cls i) (D.rep cls (acc x ms cs) i r).rounds (D.rep cls (acc x ms cs) i r).extras
            (D.rep cls (acc x ms cs) i r).comb = .ok (v, info, kappa) →
          ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!) ∧
        (∀ r : Fin 2, ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
          ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j)

/-- **Row R's target, the residuals**: the residuals' structure and groups at the table's shape, each rep's rounds at
the shape's sizes, `w` at the structure's size, and every part's residuals vanishing. -/
def TargetR : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size →
      ∃ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
        structureOf (D.sh cls i) = .ok S ∧ groups S (portSizes (D.sh cls i)) = .ok gs ∧
        (∀ r : Fin 2, (D.rep cls (acc x ms cs) i r).rounds.toList.map Array.size = roundSizes (D.sh cls i)) ∧
        (∀ r : Fin 2, (D.rep cls (acc x ms cs) i r).w.size = S.nw) ∧
        ∀ r : Fin 2, D.PartsZero cls (acc x ms cs) i r gs

/-- **Row E's target, the accumulators**: at each rep, an accumulator chain and position whose units sum
(`UnitSums`) and whose ends the algebra reads (`ReadsEnds`). -/
def TargetE : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size →
      ∃ (accC : Fin 2 → ℕ → Array F128) (p0 : Fin 2 → ℕ),
        (∀ r : Fin 2, UnitSums (D.sh cls i) (accC r) (D.opens cls (acc x ms cs) i r (p0 r))) ∧
        ∀ r : Fin 2, ReadsEnds (D.sh cls i) (D.rep cls (acc x ms cs) i r) (accC r)
          (D.opens cls (acc x ms cs) i r (p0 r))

/-- **The record's target, the decode**: at a decision, the record value the verdict reads decodes at the setup
SHA-512 gives. -/
def TargetDecode : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ d : FlockDeploy.DecR,
    FlockDeploy.decR rd x ms cs = some d → ∀ (spec : SessionSpec) (sts : Array Stmt),
    FlockDeploy.setupWith (Claims.hOf FlockDeploy.sha512) d.s.read = .ok (spec, sts) →
    ∃ sess, Record.decodeJWith (Flock.Hashes.ofSha512 (Claims.hOf FlockDeploy.sha512) : Flock.Hashes Flock.V) spec
      (FlockDeploy.jAt FlockDeploy.sha512 spec sts d.c₁ d.mids d.f) = .ok sess

/-- **The record's target, the public words**: the session's public words are empty. -/
def TargetPublics : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ D : DecAt rd x ms cs,
    (D.sess.publics.all fun p => p.size == 0) = true

end Targets

/-! ## What the base checks and the reading already give -/

section Given

variable {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop} {x : Msg} {ms : List Msg}
  {cs : List Coin}

/-- At a decision, the last step decides on `decR`'s reading. -/
theorem actR_of_dec (hr : ReadOkR rd x ms cs) (D : DecAt rd x ms cs) :
    FlockDeploy.actR rd x ms cs =
      .decide D.d.s D.d.spec D.d.sts D.d.link D.d.c₁ D.d.mids D.d.f := by
  unfold ReadOkR at hr
  revert hr
  cases h : FlockDeploy.actR rd x ms cs with
  | rej => intro hr; exact (hr : False).elim
  | coin c => intro hr; exact (hr : False).elim
  | decide s spec sts link c₁ mids f =>
    intro _
    rw [← Option.some.inj ((FlockDeploy.actR_decide h).symm.trans D.dec)]

/-- **Row C's sizes**: each rep's proof index is in range (`decR_fits`). -/
theorem idx_lt (D : DecAt rd x ms cs) {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) : 2 * i + r < D.d.f.ps.size := by
  obtain ⟨-, -, -, -, hsz, -⟩ := FlockDeploy.decR_fits D.dec
  rw [hsz, ← (D.setup_eq).2]
  omega

/-- **Row C's shapes**: each rep's proof is shaped for its file (the `shaped` base check). -/
theorem shaped_of_base (hr : ReadOkR rd x ms cs) (hb : ∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs)
    (D : DecAt rd x ms cs) (k : ℕ) (hk : k < D.d.f.ps.size) :
    ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) k) D.d.f.ps[k] := by
  have h := hb .shaped (by simp [VerdictP.bases])
  unfold holdsR at h
  rw [actR_of_dec hr D] at h
  exact h D.spec D.sts D.setup k hk

end Given

/-! ## The top statement at a table -/

section Runs

variable (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop)
  (acc : Msg → List Msg → List Coin → ℕ → Array F128)

/-- **Every table's reps run where the rows are asked**: `runs_assembled`, each row's hypothesis its target. -/
theorem runs_of_targets (hB : TargetB rd P acc) (hL : TargetL rd P acc) (hD : TargetD rd P acc)
    (hR : TargetR rd P acc) (hE : TargetE rd P acc) :
    ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size → (D.files cls i).Runs := by
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, hr, hb, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  obtain ⟨pieces, sideAt, posAt, hholds, hside, hpos, hcover⟩ := hD x ms cs hA D cls i hi
  obtain ⟨S, gs, hS, hg, hsz, hw, hparts⟩ := hR x ms cs hA D cls i hi
  obtain ⟨accC, p0, hu, hrd⟩ := hE x ms cs hA D cls i hi
  exact D.runs_assembled cls (acc x ms cs) hst hf hx hi hti (hB x ms cs hA D cls i hi) D.points_eq
    (fun r => idx_lt D hi r) (fun r => shaped_of_base hr hb D _ (idx_lt D hi r)) (hL x ms cs hA D cls i hi)
    pieces sideAt posAt hholds hside hpos hcover hS hg hsz hw hparts accC p0 hu hrd

/-- A class to read the tables' files at: their runs and row L do not depend on it. -/
def cls₀ : HClass where
  m := 13
  kLog := 6
  hk := by omega
  hk6 := le_refl _
  hm := le_refl _
  pin := ⟨0, by positivity⟩
  A₀pub := 0
  B₀pub := 0
  slotLog := 0
  units := ∅
  ptLocal := 0
  mPts := 0
  nnz := 0

/-- **The verdict where the rows are asked**: at a decision, the record decodes (`TargetDecode`), and row A's step 3
(`DecAt.verdict_of_runs`) at row G (`DecAt.rowG`), `runs_of_targets` and row L. -/
theorem verdict_of_targets (hB : TargetB rd P acc) (hL : TargetL rd P acc) (hD : TargetD rd P acc)
    (hR : TargetR rd P acc) (hE : TargetE rd P acc) (hDec : TargetDecode rd P) (hPub : TargetPublics rd P) :
    ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → holdsR FlockDeploy.sha512 rd .verdict x ms cs := by
  intro x ms cs hA
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  unfold holdsR
  cases h : FlockDeploy.actR rd x ms cs with
  | rej => trivial
  | coin c => trivial
  | decide s spec sts link c₁ mids f =>
    intro spec' sts' hs
    have hd := FlockDeploy.actR_decide h
    obtain ⟨sess, hsess⟩ := hDec x ms cs hA _ hd spec' sts' hs
    let D : DecAt rd x ms cs := ⟨_, hd, spec', sts', hs, sess, hsess⟩
    obtain ⟨h1, ht1⟩ := D.tables_of_dense hst hf hx
    exact D.verdict_of_runs cls₀ (acc x ms cs) (h1.trans ht1.symm) (by omega) (hPub x ms cs hA D)
      (fun i hi => D.rowG cls₀ (acc x ms cs) hst hf hx hi)
      (fun i hi => ⟨runs_of_targets rd P acc hB hL hD hR hE x ms cs hA D cls₀ i hi, hL x ms cs hA D cls₀ i hi⟩)

end Runs

/-! ## The pieces, with the dense one -/

/-- **The verdict's pieces**: the parts' pieces `Q`, and `dense`, that the inner statement is a dense one. -/
inductive Piece (Q : Type) where
  | dense
  | row (q : Q)

/-- The pieces at `x`: `dense`, then the parts'. -/
def pieces {Q : Type} (rows : Msg → List Q) (x : Msg) : List (Piece Q) := .dense :: (rows x).map .row

/-- What a piece says: `dense` that `x` is a dense inner statement; a part's piece what `pholds` says. -/
def pholds {Q : Type} (holds : Q → Msg → List Msg → List Coin → Prop) : Piece Q → Msg → List Msg → List Coin → Prop
  | .dense, x, _, _ => DenseInner x
  | .row q, x, ms, cs => holds q x ms cs

/-- The parts' pieces all hold. -/
def RowsHold {Q : Type} (rows : Msg → List Q) (holds : Q → Msg → List Msg → List Coin → Prop) (x : Msg)
    (ms : List Msg) (cs : List Coin) : Prop :=
  ∀ q ∈ rows x, holds q x ms cs

/-- **Every target**, at the parts' pieces `rows` (what `holds` says of each) and the reps' accumulators `acc`. -/
structure Targets (rd : Msg → Option FinalP) {Q : Type} (rows : Msg → List Q)
    (holds : Q → Msg → List Msg → List Coin → Prop) (acc : Msg → List Msg → List Coin → ℕ → Array F128) : Prop where
  b : TargetB rd (RowsHold rows holds) acc
  l : TargetL rd (RowsHold rows holds) acc
  d : TargetD rd (RowsHold rows holds) acc
  r : TargetR rd (RowsHold rows holds) acc
  e : TargetE rd (RowsHold rows holds) acc
  decode : TargetDecode rd (RowsHold rows holds)
  publics : TargetPublics rd (RowsHold rows holds)

/-- **`hpieces` at the pieces, from the targets.** -/
theorem hpieces_of_targets (rd : Msg → Option FinalP) {Q : Type} (rows : Msg → List Q)
    (holds : Q → Msg → List Msg → List Coin → Prop) (acc : Msg → List Msg → List Coin → ℕ → Array F128)
    (h : Targets rd rows holds acc) :
    ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ReadOkR rd x ms cs →
      (∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) →
      (∀ q ∈ pieces rows x, pholds holds q x ms cs) → holdsR FlockDeploy.sha512 rd .verdict x ms cs := by
  intro x ms cs hr hb hp
  refine verdict_of_targets rd (RowsHold rows holds) acc h.b h.l h.d h.r h.e h.decode h.publics x ms cs
    ⟨hp .dense List.mem_cons_self, hr, hb, fun q hq => hp (.row q) ?_⟩
  exact List.mem_cons_of_mem _ (List.mem_map.2 ⟨q, hq, rfl⟩)

end Top

/-! ## The premise of `RecursiveKS` at `nextR` -/

section Premise

/-- The deployed bound record at `nextR rd`, with the columns the reader opens, trimmed (`deployPlainTrimR`). -/
abbrev bdR (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :=
  bdDeployed _root_.Flock.Readable.relation (nextR rd)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR rowInDeployed final w

/-- V\*'s bound layout at the last round's pieces `fin`. -/
abbrev layR (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : LayoutB :=
  layoutBTops (finOpen fin)

/-- The compiled record at the coin server's transcript, whose inner view runs `IVStar` at `nextR rd`. -/
abbrev recR (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :=
  recBound _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd rest } (layR fin) (bdR rd wE final w).IVStar

/-- What each part's circuit accepts, at the deployed outer `--zk` record and the coin server's record's reader. -/
abbrev acceptsR (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :=
  PartAccepts FlockDeploy.sha512 (recR rd b aux nOf hd rest fin wE final w) (deployZkRec _root_.Flock.Readable.relation)
    (layR fin) Claims.Recursion.Form.Dense (showsPart recRead { b with tr := recTr aux nOf hd rest }) (cmtTr recRead)

/-- **`RecursiveKS`'s premise at the deployed reader**: `Flock.Assumptions.RecKSDecomp` with `IVStar`'s last
step `nextR rd` and its rows joined by `joinR`. At the brings whose transcript is the coin server's record, V\*'s bound
layout and the deployed outer `--zk` record, wherever every part's circuit accepts at a dense compiled statement,
`IVStar` accepts or two of the parts' recorded calls collide. -/
def RecKSDecompR (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) : Prop :=
  DecompB FlockDeploy.sha512 (recR rd b aux nOf hd rest fin wE final w) (bdR rd wE final w).IVStar (layR fin)
    Claims.Recursion.Form.Dense (acceptsR rd b aux nOf hd rest fin wE final w)

variable (rd : Msg → Option FinalP) {Q : Type} (rows : Msg → List Q) (holds : Q → Msg → List Msg → List Coin → Prop)
  (acc : Msg → List Msg → List Coin → ℕ → Array F128) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
  (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
  (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)

/-- The split verdict at the pieces, where the targets hold. -/
abbrev splitR (h : Top.Targets rd rows holds acc) :=
  lastSplitP FlockDeploy.sha512 rd (Top.pieces rows) (Top.pholds holds) (Top.hpieces_of_targets rd rows holds acc h)

/-- The inventory's holders at the split verdict. -/
abbrev holdersR (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Top.Piece Q → Option ℕ) :=
  ownHolders (recR rd b aux nOf hd rest fin wE final w) (layR fin)
    (ownDeploy (ownJ (recR rd b aux nOf hd rest fin wE final w) (layR fin) (ownP ownB ownQ)))

/-- **The top statement: `RecursiveKS`'s premise at `nextR`, from the targets.** At the deployed reader `rd`, the
parts' pieces `rows` (what `holds` says of each) and the reps' accumulators `acc`: where every target holds (`h`), the
owners are the statement's parts (`hR`), the checks no part holds hold off any witnesses (`hrest`), and each part's
circuit accepting implies its share of the other checks (`hlowR`) and of the pieces, `dense` among them (`hlowP`),
`RecKSDecompR` holds. -/
theorem recKSDecompR_of_targets (h : Top.Targets rd rows holds acc) (ownB : Msg → VerdictR → Option ℕ)
    (ownQ : Msg → Top.Piece Q → Option ℕ)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recR rd b aux nOf hd rest fin wE final w).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ r ∈ (splitR rd rows holds acc h).checks (Claims.Recursion.xI (recR rd b aux nOf hd rest fin wE final w) x),
        ∀ s, ownP ownB ownQ x r = some s → s < (recR rd b aux nOf hd rest fin wE final w).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recR rd b aux nOf hd rest fin wE final w).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ q ∈ ivChecks (bdR rd wE final w)
          (deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR rowInDeployed
            final w (nextR_coin rd FlockDeploy.sha512) (splitR rd rows holds acc h).toLast)
          (Claims.Recursion.xI (recR rd b aux nOf hd rest fin wE final w) x),
        holdersR rd b aux nOf hd rest fin wE final w ownB ownQ x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recR rd b aux nOf hd rest fin wE final w).nS x →
          (∀ e ∈ (layR fin).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
          q.Holds FlockDeploy.sha512 (bdR rd wE final w)
            (deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
              (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR rowInDeployed
              final w (nextR_coin rd FlockDeploy.sha512) (splitR rd rows holds acc h).toLast)
            (Claims.Recursion.xI (recR rd b aux nOf hd rest fin wE final w) x) c ((layR fin).msgs x ws))
    (hlowR : ∀ s x c ms, acceptsR rd b aux nOf hd rest fin wE final w s x c ms →
      RestComputes FlockDeploy.sha512 (recR rd b aux nOf hd rest fin wE final w) (bdR rd wE final w)
        (deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR rowInDeployed
          final w (nextR_coin rd FlockDeploy.sha512) (splitR rd rows holds acc h).toLast)
        (holdersR rd b aux nOf hd rest fin wE final w ownB ownQ) s x c ms)
    (hlowP : ∀ s x c ms, acceptsR rd b aux nOf hd rest fin wE final w s x c ms →
      PiecesComputes (bdR rd wE final w) (recR rd b aux nOf hd rest fin wE final w) (Top.pieces rows)
        (Top.pholds holds) ownQ s x c ms) :
    RecKSDecompR rd b aux nOf hd rest fin wE final w :=
  decompB_lastSplitP_shares FlockDeploy.sha512 rd (Top.pieces rows) (Top.pholds holds)
    (Top.hpieces_of_targets rd rows holds acc h) _root_.Flock.Readable.relation
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE rowInDeployed final w ownB ownQ
    _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd rest } (layR fin) (bdR rd wE final w).IVStar
    Claims.Recursion.Form.Dense (deployZkRec _root_.Flock.Readable.relation)
    (showsPart recRead { b with tr := recTr aux nOf hd rest }) (cmtTr recRead) hR hrest hlowR hlowP

end Premise

end

end RecDeploy
