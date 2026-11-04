import Proofs.Flock.Claims.Recursion.KSLin2R
import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Claims.Recursion.LamDeployed
import Proofs.Flock.Claims.Recursion.PartAcceptsPin

/-!
# `KSLin2` and `BitsLin2` at the layout of record (`DeployedR.layR`)

`DeployedR.layR elem` is `layoutBTops (finOpen (restR elem))` with every part's other hm96 calls recorded too
(`DeployedR.moreCalls`: the output rows', check 5's and the `rec-acc` chain's). Its rounds, held messages, pieces and
readers' messages are `layoutBTops`', so of the route's premises only the two that read the recorded calls move:

* `ReadersTrue` (`readersTrue_layR`): `layoutBTops`' calls are true off `HmSlotsCommit` (`readersTrue_layoutBTops`),
  and the extra ones off `MoreSlotsTrue`, the same fact at `moreCalls`' slots: at a part's staged statement, a witness
  in the outer relation evaluates each of them truly.
* `PartOpensB` (`partOpensB_layR`): a part's calls true at `layR` are true at `layoutBTops`, and a collision among
  `layoutBTops`' calls is one among `layR`'s.

`ksLin2_deployedR_layR` and `bitsLin2_deployedR_layR` are `ksLin2_deployedR_partAccepts` and
`bitsLin2_deployedR_partAccepts` at `layR`, reopened on `PlainRdInR`, `MoreSlotsTrue` and `DecompB` at `PartAccepts`
there. `ksLin2_deployedR_layRP` asks the pinned form instead, `DecompBP` at `PartAcceptsP` (each part's check at its
own witness `ws[s]`), through `ksLin2_route_stagedP`; `DecompB` at `PartAccepts` gives it (`decompBP_of_decompB`).

**`λ₂` does not read the layout's calls.** `lamLin2`'s `c_S = c_Be + |S(x_I)| + 2(n + v)` reads the parts' numbers,
the inner setup, the inner verifier's calls `n` and the compiled verifier's own `v`; the recorded calls are found, not
evaluated, by the finders. `lamLin2_layR_ge` charges the extra calls anyway, `e` more per draw beside `v`: `λ₂ ≥ 100`
holds at `layR` for every `v, e ≤ 2^54` (`lamLin2_deployed_ge_dep`, whose six terms are each at least `103` up to
`c_S ≤ 2^59`).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextR)

/-! ## The extra calls' slots -/

/-- **The extra calls are true at a witness in the outer relation** (`HmSlotsCommit` at `DeployedR.moreCalls`): at a
part's staged statement, a witness `Flock.check` passes evaluates each of the part's output rows', check 5's and
`rec-acc` chain's hm96 calls truly. -/
def MoreSlotsTrue {Col T : Type} (DO : Claims.Flock.Deploy Coin Col T) (st : Stage) : Prop :=
  ∀ (x : Msg) (ms : List (Msg × Out)) (s : ℕ) (w : Claims.Flock.Wit) (r : XR),
    StagedPart st s x (staged st x ms s) → Claims.Flock.check DO.parse (staged st x ms s) w = true → readX x = some r →
      ∀ p ∈ DeployedR.moreCalls r s,
        FlockDeploy.sha512 (p.1 ++ Claims.Recursion.VStarB.readForms w p.2.1) =
          Claims.Recursion.VStarB.readForms w p.2.2

/-- **`ReadersTrue` at the layout of record**: `layoutBTops`' calls off `HmSlotsCommit` (`readersTrue_layoutBTops`),
the extra ones off `MoreSlotsTrue`. -/
theorem readersTrue_layR {Col T : Type} (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) (h : HmSlotsCommit DO b.st)
    (hM : MoreSlotsTrue DO b.st) :
    ReadersTrue FlockDeploy.sha512 IV { b with rd := (DeployedR.layR elem).readers } DO := by
  intro x as cs ws hla hlc hsp hO e he
  obtain ⟨s, hs, he⟩ := Claims.Recursion.VStar.part_of_mem_calls he
  have hsn : s < nS x := hO.1 ▸ hs
  have hck : Claims.Flock.check DO.parse (staged b.st x (roundsOf as cs) s) ws[s] = true := by
    have h2 := hO.2 s hs
    rwa [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hsn, Option.map_some,
      Option.getD_some] at h2
  have hsg : StagedPart b.st s x (staged b.st x (roundsOf as cs) s) := by
    have h1 := hsp s hsn
    rwa [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hsn, Option.map_some,
      Option.getD_some] at h1
  change e ∈ (DeployedR.layR elem).calls x s ws[s] at he
  rw [DeployedR.layR_calls] at he
  rcases List.mem_append.1 he with he | he
  · exact readersTrue_layoutBTops (finOpen (DeployedR.restR elem)) IV b DO h x as cs ws hla hlc hsp hO e
      (Claims.Recursion.VStar.mem_calls_of_part hs he)
  · cases hr : readX x with
    | none => rw [hr] at he; simp at he
    | some r =>
      rw [hr, Option.map_some, Option.getD_some] at he
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
      exact hM x _ s ws[s] r hsg hck hr p hp

/-! ## The route's premises at `layR` -/

section Transfer

variable (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

theorem heldWhole_layR : (DeployedR.layR elem).HeldWhole :=
  heldWhole_layoutBTops (finOpen (DeployedR.restR elem))

theorem heldB_layR {W : Type} {D : Claims.Recursion.Deploy Coin} {I : Claims.Recursion.Inner Coin W}
    {shape : Claims.Recursion.Form → Prop}
    (h : Claims.Recursion.VStarB.HeldB D I (layoutBTops (finOpen (DeployedR.restR elem))) shape) :
    Claims.Recursion.VStarB.HeldB D I (DeployedR.layR elem) shape :=
  h

/-- **`PartOpensB` at `layR` from `layoutBTops`'**: a part's calls true at `layR` are true at `layoutBTops`
(`layoutBTops_sub_layR`), and a collision among `layoutBTops`' is one among `layR`'s. -/
theorem partOpensB_layR {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {shape : Claims.Recursion.Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    (h : Claims.Recursion.VStarB.PartOpensB H D DO (layoutBTops (finOpen (DeployedR.restR elem))) shape shows cmt s) :
    Claims.Recursion.VStarB.PartOpensB H D DO (DeployedR.layR elem) shape shows cmt s := by
  intro x st hx hf xO w hck hT
  rcases h x st hx hf xO w hck (fun e he => hT e (DeployedR.layoutBTops_sub_layR elem _ x s w e he)) with h | h
  · exact Or.inl h
  · exact Or.inr (Claims.Recursion.findColl_isSome_of_sub H hT
      (fun e he => DeployedR.layoutBTops_sub_layR elem _ x s w e he) h)

theorem readsRows_layR : (DeployedR.layR elem).ReadsRows FlockDeploy.sha512 :=
  Claims.Recursion.VStarB.LayoutB.readsRows_of_rowIn fun _ _ => rfl

/-- **`HeldB` at `layR`, at the dense form and the deployed inner verifier.** -/
theorem heldB_dense_layR (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (bR : Brings) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    Claims.Recursion.VStarB.HeldB
      (recBound _root_.Flock.Readable.relation bR (DeployedR.layR elem)
        (bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w).IVStar)
      (Claims.Recursion.Inner.ofFlock
        (bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w).deploy decI)
      (DeployedR.layR elem) Claims.Recursion.Form.Dense :=
  heldB_layR elem (heldB_layoutBTops _ _ _ _ (fun _ => rfl) (finOpen (DeployedR.restR elem)) _ schedMatches_dense
    (capPorts _) (rowsCover _) (finPieces_finOpen (DeployedR.restR elem) _))

/-- **`PartOpensB` at `layR`, at the dense form and the coin server's reader.** -/
theorem partOpensB_dense_layR (bR : Brings) (IV : Claims.Verifier Coin) (s : ℕ) :
    Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation bR (DeployedR.layR elem) IV)
      (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR elem) Claims.Recursion.Form.Dense
      (showsPart recRead bR) (cmtTr recRead) s :=
  partOpensB_layR elem (partOpensB_recRead (finOpen (DeployedR.restR elem)) bR (fun _ _ h => h) schedMatches_dense)

end Transfer

/-! ## `KSLin2` at any layout meeting the route's layout premises -/

/-- **`KSLin2` at the deployment's dense statements, the coin server's record and any steps and columns meeting
`StepsBaseC`, at any layout `L` whose rows, holdings, part openings and readers meet the route's premises, from
`DecompB` at `PartAccepts`**: `ksLin2_route_staged`, with `StagedB` from `stagedB_deployRec_part` at `recRead`
(`readsBack_recTr`) and the inner check from `exists_chk_iff`. The layout stays a variable here, so no step unfolds
it. -/
theorem ksLin2_stepsC_at (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ReadersTrue FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (deployZkRec _root_.Flock.Readable.relation) →
    (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L
      Claims.Recursion.Form.Dense) →
    (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
      Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s) →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR Bd D hRT hHe hPO hD
  obtain ⟨hKS, hEC, hCF⟩ := hB
  have hrb : ReadsBackB FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (fun _ => True) recRead :=
    readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl
  refine ksLin2_route_staged _root_.Flock.Readable.relation _root_.Flock.Readable.relation bR L hRows hHW
    FlockDeploy.KS.CpDep nextE colsE wE join rowInDeployed final w hKS (binds_deployed _ _ _ wE join final w) hEC hCF
    FlockDeploy.KS.zk_KS_outer ?_ hRT
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨chk, hPC, hDc⟩ := (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _ _).2 (hD _ hsh)
  exact ⟨chk, showsPart recRead bR, cmtTr recRead, hHe,
    fun decI => stagedB_deployRec_part FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } recRead L _
      (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    hPC, hPO, hDc⟩

/-- **`ksLin2_stepsC_at` from `DecompBP` at `PartAcceptsP`**, through `ksLin2_route_stagedP`. -/
theorem ksLin2_stepsC_atP (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ReadersTrue FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (deployZkRec _root_.Flock.Readable.relation) →
    (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L
      Claims.Recursion.Form.Dense) →
    (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
      Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s) →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBP FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsP FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR Bd D hRT hHe hPO hD
  obtain ⟨hKS, hEC, hCF⟩ := hB
  have hrb : ReadsBackB FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (fun _ => True) recRead :=
    readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl
  refine ksLin2_route_stagedP _root_.Flock.Readable.relation _root_.Flock.Readable.relation bR L hRows hHW
    FlockDeploy.KS.CpDep nextE colsE wE join rowInDeployed final w hKS (binds_deployed _ _ _ wE join final w) hEC hCF
    FlockDeploy.KS.zk_KS_outer ?_ hRT
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  exact ⟨showsPart recRead bR, cmtTr recRead, hHe,
    fun decI => stagedB_deployRec_part FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } recRead L _
      (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    hPO, hD _ hsh⟩

/-! ## `KSLin2` and `BitsLin2` at `layR` -/

/-- **`KSLin2` at the deployment's dense statements, at the layout of record `DeployedR.layR elem`, the coin server's
record (`recTr`), and any steps and columns meeting `StepsBaseC`, from `DecompB`**: `ksLin2_stepsC_at` at
`layR`, with `ReadersTrue` from `readersTrue_layR` (`HmSlotsCommit` and `MoreSlotsTrue`) and `PartOpensB` from
`partOpensB_layR`. -/
theorem ksLin2_stepsC_layR (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  fun hD => ksLin2_stepsC_at nextE colsE b aux nOf hd recRest (DeployedR.layR elem) (readsRows_layR elem)
    (heldWhole_layR elem) wE join final w hB (readersTrue_layR elem _ _ _ hmSlotsCommit_of_full hM)
    (heldB_dense_layR elem nextE colsE _ wE join final w) (partOpensB_dense_layR elem _ _) hD

/-- **`BitsLin2` at the same choice**: `ksLin2_stepsC_layR` through `bitsLin2On_of_ksLin2On` (`sessionAccepts`). -/
theorem bitsLin2_stepsC_layR (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D hD
  exact Claims.Recursion.bitsLin2On_of_ksLin2On
    (sessionAccepts Bd.IVStar (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck
      { bR with rd := L.readers })
    (ksLin2_stepsC_layR nextE colsE b aux nOf hd recRest elem wE join final w hB hM hD)

/-- **`KSLin2` at the deployment's dense statements, at the layout of record, the coin server's record and the
deployed reader's steps and trimmed columns**: `ksLin2_stepsC_layR` at `stepsBase_R`. What stays open: `DecompB` at
`PartAccepts`, `PlainRdInR` and `MoreSlotsTrue`. -/
theorem ksLin2_deployedR_layR (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation (nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_stepsC_layR _ _ b aux nOf hd recRest elem wE join final w (stepsBase_R rd wE join final w hIn) hM

/-- **`BitsLin2` at the same choice**: `bitsLin2_stepsC_layR` at `stepsBase_R`. It reads `StrictBirthday` and
`ExpectedCR`, the named SHA-512 collision assumptions, and `ksLin2_deployedR_layR`'s open premises: a bound on a
number under named premises, not a proved security level. -/
theorem bitsLin2_deployedR_layR (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation (nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  bitsLin2_stepsC_layR _ _ b aux nOf hd recRest elem wE join final w (stepsBase_R rd wE join final w hIn) hM

/-! ## The pinned form: each part's check at its own witness -/

/-- **`ksLin2_stepsC_layR` from `DecompBP` at `PartAcceptsP`**: `ksLin2_stepsC_atP` at `layR`. -/
theorem ksLin2_stepsC_layRP (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBP FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsP FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  fun hD => ksLin2_stepsC_atP nextE colsE b aux nOf hd recRest (DeployedR.layR elem) (readsRows_layR elem)
    (heldWhole_layR elem) wE join final w hB (readersTrue_layR elem _ _ _ hmSlotsCommit_of_full hM)
    (heldB_dense_layR elem nextE colsE _ wE join final w) (partOpensB_dense_layR elem _ _) hD

/-- **`KSLin2` at the deployed reader's steps, at the layout of record, from the pinned form**:
`ksLin2_stepsC_layRP` at `stepsBase_R`. What stays open: `DecompBP` at `PartAcceptsP`, `PlainRdInR` and
`MoreSlotsTrue`. -/
theorem ksLin2_deployedR_layRP (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation (nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBP FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsP FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_stepsC_layRP _ _ b aux nOf hd recRest elem wE join final w (stepsBase_R rd wE join final w hIn) hM

/-! ## `λ₂` at `layR`, charged the extra calls -/

/-- **`λ₂ ≥ 100` at the layout of record, with `c_S` charged `e` extra calls per draw** beside the compiled verifier's
own `v` (`c_S = c_Be + |S(x_I)| + 2(n + v + e)`): `lamLin2_deployed_ge_dep` at `layR` and `v + e ≤ 2^55`. -/
theorem lamLin2_layR_ge (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) {v e : ℕ}
    (hV : v ≤ 2 ^ 54) (hE : e ≤ 2 ^ 54) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel b (DeployedR.layR elem)).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (v + e) :=
  lamLin2_deployed_ge_dep rel b (DeployedR.layR elem) nextE colsE wE join rowIn final w decI hin hO hn hS hU
    ((Nat.add_le_add hV hE).trans (by norm_num))

end RecDeploy
