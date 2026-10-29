import Proofs.Flock.Claims.Recursion.VStarFinal
import Proofs.Flock.Claims.Recursion.HpEarlyStaged
import Proofs.Flock.Claims.Recursion.HhHeld
import Proofs.Flock.Claims.Recursion.HmsgRun
import Proofs.Flock.Claims.Recursion.HonestRun
import Proofs.Flock.Claims.Recursion.MoreSlotsAll
import Proofs.Flock.Claims.Recursion.HonestPartsLayR
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.VStarDense
import Proofs.Flock.Claims.Recursion.HonestWitR
import Proofs.Flock.Claims.Recursion.NumsFixedR
import Proofs.Flock.Claims.Recursion.VStarTopsOk
import Proofs.Flock.Claims.Recursion.RoundWords
import Proofs.Flock.Claims.Recursion.VStarPortNames

/-!
# The honest half of `RecursiveKSAt`'s non-vacuity at the deployed V\* (`honest_half`)

`RecursiveKS` reads, at the brings `deployedB b` for every `b`, the premise `DecompB` at `PartAccepts`: wherever every V\*
part's circuit accepts at a dense compiled statement, `IVStar` accepts or two recorded calls collide. It is non-vacuous
where some run has every part accepting. `honest_half` gives that run: the compiled verifier's run of the reference
prover at `deployedB` of the deployed brings at `x` (`bringsAt`: the deployed assigner `A0` and the fitted records), and
every part `s < nS x` accepts on it (`PartAccepts` at `DeployedR.layR OpenRows.elemR`, `RecursiveKSAt`'s layout).

`honest_partAccepts_final` with its pieces discharged:

* `hA`, the run lemma (`run_msg_inner`);
* `hpE`, hp before the last round (`hpEarly_staged`, every part staged, given `RowsFit`);
* `hhFit`, hh from the fitting (`hh_A0` at the dense schedule `schedMatches_dense`, the trace `traceAt_staged` and the
  messages `hmsg_run`);
* `hTm`, the extra calls `layR` records (`moreSlotsTrue_deployed` at the run's staged part, whose outer witness
  `Flock.check` accepts: `vstar_complete_run_setup_at`);
* the schedule (`schedMatches_dense`) and each part's staging (`stagedPart_run`, from the run accepting).

The run accepting is one named premise, `HonestRunAccepted`: the statement is in the relation, parses and is dense, and
the compiled verifier accepts the reference prover's run. V\*'s gate-level completeness at the run is
`HonestWitSatisfies`, and the inner prover's openings at the run are `IrShaped` (`Ir` is not fixed in Lean: the
theorem holds for every inner prover `Ir` with `IrShaped` at the run, the facts of `Ir.rest`, which the deployed
builder's `rec_vstage`, `rec_vstar` and `verifier_rows` build, that the stager's checks read). These three are
witnessed outside Lean at the deployed builder, all on one run, r20261027-014625-06c6 (`HonestFixture`): there the
compiled `flock-verify` accepted every V\* part's session of the honest deployed recursive run, each part on its own
(outer tree art:886efa1d995543a70ba46e6ef713a2afe89a7022b7fc4988c9cc2dbd629a43a8), and CAPLOWER's checks of `IrShaped` and
`HonestWitSatisfies` passed on that tree (r20261027-025051-8aa6). No collision among a part's recorded calls is
`NoCollAt`, SHA-512's collision event, which passed on the same tree (r20261027-033500-eeb1).

Discharged in Lean: each part's staging setting up (`stagedSetsUp_accepted`, from the run accepting), the stager's
numbers (`numsFixed`), the stager accepting every part (`hok_honest`: the level tops and the caps from `hTops`, at the
history `hist.take (kI x - 1)`, which holds the schedule's `2R` rounds since `kI x = 2 + 2R`; the inner parts' port
names from `VStar.parts`, `innerPorts_named`), and the holders of one round's row agreeing on its port's words
(`heldRows_words`, the verifier's `checkRoundWords`). The rest are hypotheses: the fitted rows (`RowsFit`) and the tape
exception (`hTops`).
`bringsAt_deployedB`: the brings here are `RecursiveKS`'s `deployedB` at `bringsAt mask Ir b0 x`.
-/

namespace RecDeploy.HonestHalf

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest

/-- **The brings `honest_half` runs at are `RecursiveKS`'s**: `deployedB` of the deployed brings at `x`. -/
theorem bringsAt_deployedB (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) :
    bringsAt mask Ir (deployedB b0) x = deployedB (bringsAt mask Ir b0 x) := rfl

/-- A part with a statement and a circuit is one of `x`'s parts. -/
theorem lt_nS_of_partAt {x : Msg} {s : ℕ} {sv : _root_.VStar.Stmt} {c : Circuit}
    (h : partAt x s = some (sv, c)) : s < nS x := by
  cases hr : readX x with
  | none => simp [partAt, hr] at h
  | some r =>
    simp only [nS, hr]
    by_contra hs
    have hn : r.w.parts[s]? = none := by simp; omega
    simp [partAt, hr, hn] at h

/-! ## The run and its named premises -/

section Premises

variable (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
  (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
  (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)

/-- The deployed brings at `x` under `deployedB`, with the coin server's record and V\*'s layout's readers. -/
noncomputable abbrev bLAt : Brings :=
  { { bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } with
    rd := (DeployedR.layR OpenRows.elemR).readers }

/-- The inner verifier the compiled record runs. -/
noncomputable abbrev IAt : Claims.Recursion.Inner Coin Claims.Flock.Wit := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI

/-- `RecursiveKSAt`'s compiled record at the deployed brings at `x`. -/
noncomputable abbrev DAt : Claims.Recursion.Deploy Coin :=
  recBound _root_.Flock.Readable.relation
    { bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } (DeployedR.layR OpenRows.elemR)
    DeployedR.bdR.IVStar

/-- **The verifier accepts the reference prover's run, at an instance that exists** (witnessed outside Lean). The
statement `x` with witness `w` is in the compiled record's relation, parses to `st` at the dense form, and the compiled
verifier accepts the reference prover's run on tape `ω` and coins `r`. A concrete statement can't be evaluated in Lean
(its circuit is 22 MB and `Json.parse` is not kernel-evaluable), so this is witnessed outside Lean: every V\* part's
session of the honest deployed recursive run is accepted by the compiled `flock-verify` (composed per part; no single
compiled verdict exists).

The witness is run r20261027-014625-06c6 (`benchmarks/recursion_e2e --e2e --cpu`, K = 64, N = 16, under the firewall,
at f1asm 16a44e3b86). The compiled verifier printed `VERDICT {"accepted":true}` on each of the 10 V\* parts (L0–L3,
alg-p0, alg-p1, cap, cap-L1–L3), and the inner session was accepted. With a challenge or query coin flipped, the parts
that read it refused. The outer tree is art:886efa1d995543a70ba46e6ef713a2afe89a7022b7fc4988c9cc2dbd629a43a8.

Only Python joins the 10 per-part verdicts, so what the run witnesses is that every part's session accepted, not that
one compiled verifier accepted the whole outer statement. -/
def HonestRunAccepted (w : Claims.Flock.Wit) (st : Claims.Recursion.Stmt)
    (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
    (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin) : Prop :=
  Claims.Recursion.R (DAt mask Ir b0 x aux nOf hd recRest) (IAt decI) x w ∧
    (DAt mask Ir b0 x aux nOf hd recRest).parse x = some st ∧ Claims.Recursion.Form.Dense st.form ∧
    (Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
      (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).out = .acc

/-- **No collision among a part's recorded calls** on the reference prover's run: at each part `s < nS x`, the hm96
calls `layR` records at its outer witness and the calls the compiled verifier makes at its V\* statement hold no two
distinct inputs with one SHA-512 output (`findColl`). At SHA-512 modelled as a random function with 512-bit outputs,
each pair of distinct inputs collides with probability `2^-512`, so the run has a collision with probability at most
`C²·2^-513` for `C` calls; this is the run's collision event, not a fact of the honest prover. -/
def NoCollAt (w : Claims.Flock.Wit) (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
    (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin) : Prop :=
  ∀ s < nS x, (Claims.findColl ((DeployedR.layR OpenRows.elemR).calls x s
    ((bLAt mask Ir b0 x aux nOf hd recRest).ref.wit w ω.1
      (((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
        (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.take ((IAt decI).V.k (xIOf x))).map histOf) s) ++
    Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 (DAt mask Ir b0 x aux nOf hd recRest)
      ((Claims.Recursion.runVstar FlockDeploy.sha512 (DAt mask Ir b0 x aux nOf hd recRest) (IAt decI)
        (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w) x ω r).getD s []))).isSome = false

/-- **The honest run's staging sets up** (`StagedSetsUp` at the honest run's statement `x`), from
`HonestRunAccepted`: the compiled verifier accepting the reference prover's run means each part's recursive setup passed
(`stagedSetsUp_run`). -/
theorem stagedSetsUp_accepted {w : Claims.Flock.Wit} {st : Claims.Recursion.Stmt}
    {ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest)} {r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin}
    (h : HonestRunAccepted mask Ir b0 x aux nOf hd recRest decI w st ω r) :
    ∀ s < nS x, (setupO (staged (bLAt mask Ir b0 x aux nOf hd recRest).st x
      ((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
        (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.take ((IAt decI).V.k (xIOf x) + kS x)) s)).isSome :=
  stagedSetsUp_run FlockDeploy.sha512 (IAt decI).V _ x _ ω r h.2.2.2

end Premises

/-- **If the verifier accepts the reference prover's run (witnessed by the fixture), every V\* part accepts**, at
`RecursiveKSAt`'s record: the honest half of its non-vacuity. For every inner prover `Ir` with `IrShaped` at the run,
at the deployed brings at `x` under `deployedB` (`bringsAt_deployedB`), V\*'s layout of record `layR elemR` and the
deployed bound record, where the compiled verifier accepts the reference prover's run (`HonestRunAccepted`), each part
`s < nS x`'s circuit accepts. `HonestRunAccepted`, `IrShaped` and `HonestWitSatisfies` are witnessed outside Lean at the
deployed builder, on one run, r20261027-014625-06c6 (`HonestFixture`): each of its 10 V\* parts' sessions was
accepted by the compiled `flock-verify`, and only Python joins those verdicts (outer tree
art:886efa1d995543a70ba46e6ef713a2afe89a7022b7fc4988c9cc2dbd629a43a8). `IrShaped` passed on that tree in
r20261027-025051-8aa6, and so did `HonestWitSatisfies`; `NoCollAt` passed there in r20261027-033500-eeb1.

Besides those it takes no collision among each part's recorded calls (`NoCollAt`, SHA-512's collision event), the
fitted rows (`RowsFit`) and the tape exception (`hTops`: the reps' tops differ at every level above 0). The stager
accepts every part by `hok_honest` (the level tops and the caps from `hTops`, the inner parts' port names by
`innerPorts_named`), and the holders of one round's row agree on its port's words by the verifier's check
(`heldRows_words`). -/
theorem honest_half (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let bR : Brings := { bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR OpenRows.elemR
    let bL := bLAt mask Ir b0 x aux nOf hd recRest
    let I := IAt decI
    let D := DAt mask Ir b0 x aux nOf hd recRest
    ∀ {w : Claims.Flock.Wit} {st : Claims.Recursion.Stmt} (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
    HonestRunAccepted mask Ir b0 x aux nOf hd recRest decI w st ω r →
    HonestWitSatisfies FlockDeploy.sha512 I.V bL x w ω r →
    NoCollAt mask Ir b0 x aux nOf hd recRest decI w ω r →
    let hist := ((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf
    let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
    IrShaped Ir x w ω.1 (hist.take (RestRead.kI x - 1)) →
    (∀ i < RestRead.kI x - 1, HpEarly.RowsFit (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) i) →
    (∀ src, srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) →
    ∀ s < nS x,
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro bR L bL I D w st ω r hRun hW hnc hist ms hIr hfit hTops s hs
  have hN := numsFixed bL.st
  obtain ⟨hR, hx, hf, hacc⟩ := hRun
  obtain ⟨xr, hxr⟩ := readX_of_lt_nS hs
  have hk : I.V.k (xIOf x) = RestRead.kI x := rfl
  have hSall : ∀ s' sv c, partAt x s' = some (sv, c) → ∃ xO, StagedPart bL.st s' x xO := fun s' _ _ hp =>
    ⟨_, stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc (lt_nS_of_partAt hp)⟩
  have hTm : ∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
      FlockDeploy.sha512 (p.1 ++ readForms (bL.ref.wit w ω.1 hist s) p.2.1) =
        readForms (bL.ref.wit w ω.1 hist s) p.2.2 := by
    obtain ⟨-, -, -, -, hrel, -⟩ := vstar_complete_run_setup_at FlockDeploy.sha512 I
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN hR ω r hacc hW hs
    have hck := check_of_R hrel
    have hsg := stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs
    have heq := runVstar_getD_staged FlockDeploy.sha512 I
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN (refP I.V bL w) ω r hacc hs
    exact fun xr' hxr' => MoreSlots.moreSlotsTrue_deployed bL.st x _ s _ xr' hsg (by rw [← heq]; exact hck) hxr'
  have hA : ∀ i < I.V.k (xIOf x), ms.getD i [] =
      (((bL.ref.PI w).next (xIOf x) (hist.take i) ω.1).eval FlockDeploy.sha512).1 :=
    run_msg_inner FlockDeploy.sha512 I.V bL w hxr ω r
  have hpE : ∀ i < RestRead.kI x - 1, ∀ o m, (o, m) ∈ L.piecesOf x s (bL.ref.wit w ω.1 hist s) i →
      ((ms.getD i []).drop o).take m.length = m := fun i hi =>
    HpEarly.hpEarly_staged mask (Ir.fit x) I.V bL w hxr hk ω r hs hi (hfit i hi) hSall
  have hmsg : ∀ i < RestRead.kI x - 1, ms.getD i [] = (roundRows (Ir.fit x) w ω.1 hist i).flatMap HpEarly.rowCommit :=
    fun i hi => hmsg_run (A0 mask) (Ir.fit x) I.V bL w hxr hk ω r hi
  obtain ⟨xr₁, rs, hxr₁, hrs, hkR⟩ := schedMatches_dense x st hx hf
  have hH : ∀ r' rs', readX x = some r' → repSched r'.shape = some rs' →
      2 * rs'.size ≤ (hist.take (RestRead.kI x - 1)).length := by
    intro r' rs' hr' hrs'
    have hrr : rs' = rs := by
      rw [hxr₁] at hr'
      rw [← Option.some.inj hr', hrs] at hrs'
      exact (Option.some.inj hrs').symm
    have e : (hist.take (RestRead.kI x - 1)).length = RestRead.kI x - 1 := by
      rw [List.length_take, Nat.min_eq_left]
      show RestRead.kI x - 1 ≤
        (((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf).length
      rw [List.length_map, List.length_take]
      have hlen : I.V.k (xIOf x) + kS x ≤ (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.length :=
        run_length_ge FlockDeploy.sha512 I.V bL.st hxr (refP I.V bL w) ω r
      have hk2 : I.V.k (xIOf x) = RestRead.kI x := rfl
      omega
    have hkI : RestRead.kI x = 2 + 2 * rs.size := hkR
    rw [e, hrr]
    omega
  exact honest_partAccepts_final mask Ir (deployedB b0) x aux nOf hd recRest decI hN hR hx hf ω r hacc hW s hs
    hTm (hnc s hs) hA hpE ⟨xr₁, rs, hxr₁, hrs⟩
    ⟨bL.st, _, stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs⟩ hTops
    (fun _ => hok_honest hx hf hIr (fun _ _ _ _ hp _ _ hr hrs => innerPorts_named hp hr hrs) hH hTops)
    (fun _ hxr' => heldRows_words hxr')
    (fun hfitH => HhHeld.hh_A0 (mask := mask) (I := Ir.fit x) (x := x) (w := w) (ρ := ω.1) (hist := hist) (s := s)
      (ms := ms) (schedMatches_dense x st hx hf)
      (fun sv c src pin hp hsrc hpi => VStarTraceAt.traceAt_staged hSall mask s sv c src pin hp hsrc hpi) hfitH hmsg)

end RecDeploy.HonestHalf
