import Proofs.Flock.Claims.Recursion.DeployPremises
import Proofs.Flock.Claims.Recursion.VStarKS

/-!
# The recursion's premises at `deployRec`, with registered ties

`DeployPremises.lean` reduces the V* facts at `deployRec IV ivScope b` to facts about V*'s circuits and the readers
(`ReadersLogColl`, `ReadersOpenPlayed`, `ReadersAccept`), each falling back only on a collision among the recorded
calls, and proves the recursion's KS over the deployed plain inner record from them (`ksA_deployed`). A V* part may also
fall back on a tie between a recorded call and an evaluation the V* setup makes outside the oracle at its statement
(`Brings.setup`, which `deployRec` takes for `Deploy.setup`), which `Br` finds. This file is that reduction with the
registered ties:

* **The readers at a layout** (`readersOf L`, `⟨L.msgs, L.calls⟩`): `deployRec` at them reads through the layout's
  readers (`readsThrough_deployRec`), and its V* statements number `nS` (`vstarLen_deployRec`), which is what
  `VStar.codeFacts_of_parts` reads besides V*'s parts.
* **The readers' facts with registered ties** (`ReadersLogCollR`, `ReadersOpenPlayedR`, `ReadersAcceptR`): each falls
  back on `ReadersColl`, a collision among the recorded calls or a registered tie at the V* statements staged at the
  run's inner and staging rounds (`regCalls`). Each of `DeployPremises`' readers' facts gives its registered form
  (`ReadersLogColl.toR`, `ReadersOpenPlayed.toR`, `ReadersAccept.toR`). They give `VStar.CallsLogColl`,
  `VStar.CallsOpenPlayed` and `VStar.InnerAcceptsR` at `deployRec` (`callsLogCollR`, `callsOpenPlayedR`,
  `innerAcceptsR`), as their originals give `CallsLogColl`, `CallsOpenPlayed` and `InnerAcceptsColl`.
* **The recursion's KS with registered ties** (`ksA_deployed_reg`, `ksFlock_deployed_reg`): `ksA_deployed` and
  `ksFlock_deployed` at `VStar.KS_of_flockKS_played`, whose bound charges the forks' registered ties to `Br`.
  `ksA_deployed_parts` is the same from V*'s parts at a layout the readers are (`VStar.codeFacts_of_parts`).
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay)
open FlockDeploy (Coin)

/-! ## The readers at a layout -/

/-- **The deployed readers at a layout**: the inner messages each off the first part that holds it, and each part's
SHA-512 evaluations, read at the hidden values the layout names (`VStar.Layout.msgs`, `VStar.Layout.calls`). -/
def readersOf (L : Claims.Recursion.VStar.Layout) : Readers := ⟨L.msgs, L.calls⟩

/-- **`deployRec` reads through the layout's readers** wherever it brings them. -/
theorem readsThrough_deployRec (H : Msg → Msg) (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (L : Claims.Recursion.VStar.Layout) (hb : b.rd = readersOf L) :
    Claims.Recursion.VStar.ReadsThrough H (deployRec IV ivScope b) IV L.msgs L.calls := by
  refine ⟨fun x r ws => ?_, fun x ws => ?_⟩
  · show innerView H IV b.rd x r ws = _
    rw [hb]
    rfl
  · show (ws.mapIdx fun s w => b.rd.calls x s w).flatten = _
    rw [hb]
    rfl

/-- **`deployRec`'s V\* statements number `nS`**: `vstar` stages one per part. -/
theorem vstarLen_deployRec (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings) :
    Claims.Recursion.VStar.VstarLen (deployRec IV ivScope b) := fun x ms => by
  show nS x ≤ (vstar b.st x ms).length
  simp [vstar]

/-! ## The readers' facts with registered ties -/

/-- **The V\* setup's evaluations at the V\* statements `xs`**: each input `b.setup` lists at each, with its hash. -/
def regCalls (H : Msg → Msg) (b : Brings) (xs : List Msg) : List (Msg × Msg) :=
  xs.flatMap fun xO => (b.setup xO).map fun u => (u, H u)

theorem setupCalls_deployRec (H : Msg → Msg) (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (xs : List Msg) : Claims.Recursion.setupCalls H (deployRec IV ivScope b) xs = regCalls H b xs := rfl

/-- **A collision the finders charge, at the readers**: two calls the witnesses record collide (`Be`'s), or one of the
V* setup's evaluations at the V* statements `xs` ties with another or with a recorded call (`Br`'s). -/
def ReadersColl (H : Msg → Msg) (b : Brings) (x : Msg) (xs : List Msg) (ws : List Claims.Flock.Wit) : Prop :=
  (∃ u v, Claims.findColl (callsOf b.rd x ws) = some (u, v) ∧ Claims.Collision H u v) ∨
    (Claims.Recursion.findCollIn (callsOf b.rd x ws) (regCalls H b xs)).isSome = true

theorem callsColl_deployRec {H : Msg → Msg} {IV : Claims.Verifier Coin} {ivScope : Msg → Bool} {b : Brings}
    {x : Msg} {xs : List Msg} {ws : List Claims.Flock.Wit} (h : ReadersColl H b x xs ws) :
    Claims.Recursion.VStar.CallsColl H (deployRec IV ivScope b) x xs ws := h

/-- **The readers record the inner verifier's calls, off a collision or a registered tie** (`ReadersLogColl` with
`ReadersColl` for its collision, at the V* statements staged at `as` and `cs`). -/
def ReadersLogCollR {Col T : Type} (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
        (∀ e ∈ (innerRunAt H IV b.rd x ws cs).vLog, e ∈ callsOf b.rd x ws) ∨
          ReadersColl H b x (stagedAll b.st x as cs) ws

/-- **The readers record each played inner message's opening, off a collision or a registered tie**
(`ReadersOpenPlayed` with `ReadersColl` to fall back on, at recorded calls that are true evaluations). -/
def ReadersOpenPlayedR {Col T : Type} (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col T) (opening : Claims.Recursion.Opening) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
        ∀ j < (innerRunAt H IV b.rd x ws cs).msgs.length, (∃ es, es ⊆ callsOf b.rd x ws ∧
          opening.opens (as.getD j []) (((innerRunAt H IV b.rd x ws cs).msgs.map Prod.fst).getD j []) es) ∨
            ReadersColl H b x (stagedAll b.st x as cs) ws

/-- **V\*'s circuits compute the inner verifier at the shape `shape`, off a collision or a registered tie**
(`ReadersAccept` with `ReadersColl` for its collision). -/
def ReadersAcceptR {Col T : Type} (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col T) (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ (x : Msg) (s : Claims.Recursion.Stmt), parse x = some s → shape s.form →
    ∀ (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
      cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
        Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
        (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
          (innerRunAt H IV b.rd x ws cs).out = .acc ∨ ReadersColl H b x (stagedAll b.st x as cs) ws

theorem ReadersLogColl.toR {Col T : Type} {H : Msg → Msg} {IV : Claims.Verifier Coin} {b : Brings}
    {DO : Claims.Flock.Deploy Coin Col T} (h : ReadersLogColl H IV b DO) : ReadersLogCollR H IV b DO :=
  fun x as cs ws hl hc hs ho ht => (h x as cs ws hl hc hs ho ht).imp_right Or.inl

theorem ReadersOpenPlayed.toR {Col T : Type} {H : Msg → Msg} {IV : Claims.Verifier Coin} {b : Brings}
    {DO : Claims.Flock.Deploy Coin Col T} {opening : Claims.Recursion.Opening}
    (h : ReadersOpenPlayed H IV b DO opening) : ReadersOpenPlayedR H IV b DO opening :=
  fun x as cs ws hl hc hs ho _ j hj => Or.inl (h x as cs ws hl hc hs ho j hj)

theorem ReadersAccept.toR {Col T : Type} {H : Msg → Msg} {IV : Claims.Verifier Coin} {b : Brings}
    {DO : Claims.Flock.Deploy Coin Col T} {shape : Claims.Recursion.Form → Prop} (h : ReadersAccept H IV b DO shape) :
    ReadersAcceptR H IV b DO shape :=
  fun x s hs hf as cs ws hl hc hsp ho ht => (h x s hs hf as cs ws hl hc hsp ho ht).imp_right Or.inl

/-! ## The V\* facts at `deployRec`, with registered ties -/

/-- **`VStar.CallsLogColl` at `deployRec`** from the readers' facts. -/
theorem callsLogCollR {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) (hT : ReadersTrue H I.V b DO)
    (hL : ReadersLogCollR H I.V b DO) : Claims.Recursion.VStar.CallsLogColl H (deployRec I.V ivScope b) I DO := by
  intro decO x Ω P ω r c ws hg
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho ⊢
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hrun : Claims.Recursion.innerRun H (deployRec I.V ivScope b) I x r ws =
      innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r) := by
    show run H (replay ((innerView H I.V b.rd x r ws).msgs.map Prod.fst)) I.V (xIOf x) ()
      (fun i => coinSeq r i.val) = _
    unfold innerView
    rw [run_replay_seen, coins_getD r (show I.V.k (xIOf x) ≤ I.V.k (xIOf x) + kS x by omega)]
    rfl
  rw [hrun]
  have hsp := stagedAll_run H I.V b x P ω r hg.1
  exact (hL x _ _ ws hl hc hsp ho (hT x _ _ ws hl hc hsp ho)).imp_left fun h e he => Or.inl (h e he)

/-- **`VStar.CallsOpenPlayed` at `deployRec`** from the readers' facts, with no fixed or public round: the inner view
is the inner verifier's run of the readers' messages, which plays at most the inner rounds (`run_length_le`). -/
theorem callsOpenPlayedR {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) (opening : Claims.Recursion.Opening)
    (hT : ReadersTrue H I.V b DO) (hO : ReadersOpenPlayedR H I.V b DO opening) :
    Claims.Recursion.VStar.CallsOpenPlayed H (deployRec I.V ivScope b) I DO opening (fun _ _ => none)
      (fun _ _ => false) := by
  intro decO x Ω P ω r c ws hg j hj
  refine ⟨fun m hm => by simp at hm, fun hp => by simp at hp, fun _ _ => ?_⟩
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho ⊢
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hv : (deployRec I.V ivScope b).innerView H x r ws =
      (innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r)).seen :=
    innerView_eq H I.V b.rd x r ws (Nat.le_add_right _ _)
  rw [hv] at hj ⊢
  have hsp := stagedAll_run H I.V b x P ω r hg.1
  rcases hO x _ _ ws hl hc hsp ho (hT x _ _ ws hl hc hsp ho) j hj with ⟨es, hes, hop⟩ | hcoll
  · refine Or.inl ⟨es, hes, ?_⟩
    have hjk : j < I.V.k (xIOf x) :=
      Nat.lt_of_lt_of_le hj (Claims.Recursion.run_length_le H _ I.V (xIOf x) () _)
    have hjn : j < I.V.k (xIOf x) + kS x := Nat.lt_of_lt_of_le hjk (Nat.le_add_right _ _)
    have ha : ((run H P (deployRec I.V ivScope b).V x ω r).msgs.map Prod.fst).getD j [] =
        (runAs (I.V.k (xIOf x) + kS x) (run H P (V I.V b.st) x ω r)).getD j [] := by
      show ((run H P (V I.V b.st) x ω r).msgs.map Prod.fst).getD j [] = _
      simp only [runAs, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_take_of_lt hjn]
    rw [ha]
    exact hop
  · exact Or.inr hcoll

/-- **`VStar.VStarAcceptsR shape` at `deployRec`** from V*'s circuits and the readers at that shape. -/
theorem vstarAcceptsR {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) (shape : Claims.Recursion.Form → Prop)
    (hA : ReadersAcceptR H I.V b DO shape) :
    Claims.Recursion.VStar.VStarAcceptsR H (deployRec I.V ivScope b) I DO shape := by
  intro x s hs hf Ω P ω r hacc ws ho ht
  rw [runVstar_staged H I ivScope b P ω r hacc] at ho ⊢
  have hl := runAs_length H I.V b x P ω r hacc
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hv : (deployRec I.V ivScope b).innerView H x r ws =
      (innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r)).seen :=
    innerView_eq H I.V b.rd x r ws (Nat.le_add_right _ _)
  rw [hv]
  exact hA x s hs hf _ _ ws hl hc (stagedAll_run H I.V b x P ω r hacc) ho ht

/-- **`VStar.InnerAcceptsR` at `deployRec`** from the readers' facts and V*'s circuits at the dense, holo and sparse
shapes, with `refuses`. -/
theorem innerAcceptsR {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) (hT : ReadersTrue H I.V b DO)
    (hD : ReadersAcceptR H I.V b DO Claims.Recursion.Form.Dense)
    (hH : ReadersAcceptR H I.V b DO Claims.Recursion.Form.Holo)
    (hS : ReadersAcceptR H I.V b DO Claims.Recursion.Form.Sparse) :
    Claims.Recursion.VStar.InnerAcceptsR H (deployRec I.V ivScope b) I DO :=
  Claims.Recursion.VStar.innerAcceptsR_of_vstar (refuses I.V ivScope b H) (callsTrue H I ivScope b DO hT)
    (vstarAcceptsR H I ivScope b DO _ hD) (vstarAcceptsR H I ivScope b DO _ hH)
    (vstarAcceptsR H I ivScope b DO _ hS)

/-! ## The recursion's KS over the deployed plain inner record, with registered ties -/

/-- **The recursion's KS at `deployRec` over the deployed plain inner record, with registered ties**: `ksA_deployed`
at `VStar.KS_of_flockKS_played`, from the readers' facts with registered ties (`ReadersLogCollR`,
`ReadersOpenPlayedR`, `ReadersAcceptR` at the three shapes), which `DeployPremises`' readers' facts give
(`ReadersLogColl.toR`, `ReadersOpenPlayed.toR`, `ReadersAccept.toR`). -/
theorem ksA_deployed_reg (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hL : ReadersLogCollR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hO : ReadersOpenPlayedR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) opening)
    (hD : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Dense)
    (hH : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Holo)
    (hS : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO)
      Claims.Recursion.Form.Sparse)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) :=
  let I := Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI
  Claims.Recursion.VStar.KS_of_flockKS_played FlockDeploy.sha512 (deployRec I.V ivScope b) I (deployZkRec relO)
    opening FlockDeploy.decCoin hF (FlockDeploy.KS.plain_KSAbort rel htab decI hdec)
    (refuses I.V ivScope b FlockDeploy.sha512) (innerRounds I ivScope b) (sessionNums I.V ivScope b relO)
    (tableErrNonneg I.V ivScope b) (sessionRounds I ivScope b relO) (sessionAccepts I.V ivScope b)
    (sessionRuns_sha512 I ivScope b relO) (innerViewRun FlockDeploy.sha512 I ivScope b (deployZkRec relO))
    (callsTrue FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT)
    (callsLogCollR FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT hL)
    (callsOpenPlayedR FlockDeploy.sha512 I ivScope b (deployZkRec relO) opening hT hO)
    (innerAcceptsR FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT hD hH hS)
    (roundsMargin_deployRec_V false FlockDeploy.sha512 I rfl ivScope b)
    (FlockDeploy.coinsShown_V FlockDeploy.sha512 false) (callsNil I.V ivScope b FlockDeploy.sha512)

/-- **`KSFlock` at `deployRec`, with registered ties**: `ksA_deployed_reg` at every inner decoder that meets the spec, as
`KSA` is `KS`. Its level (`Claims.Recursion.lam`) is about 65 bits per SHA-512 call at the deployed shape (one inner
table at `m = 35`, `γ`'s weight at `IVStar` at `coverW`), below the 100-bit floor and the `2^-128` target under every
weight. Route 2C's bound λ₂ (`lamLin2`) evaluates to at least 100 at the deployed records
(`RecDeploy.lamLin2_deployed_ge_dep`), but `KSLin2` at the deployment's dense statements (`KSLin2On` at
`RecDeploy.DenseAt`) only follows from premises still open (`RecDeploy.ksLin2_deployed_open`), Holo statements being
candidate 3's own model, and `BitsLin2` at those statements follows from it (`bitsLin2On_of_ksLin2On`). -/
theorem ksFlock_deployed_reg (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hL : ReadersLogCollR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hO : ReadersOpenPlayedR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) opening)
    (hD : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Dense)
    (hH : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Holo)
    (hS : ReadersAcceptR FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO)
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec relO) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec =>
    ksA_deployed_reg rel relO ivScope b opening hF htab hT hL hO hD hH hS decI hdec

/-- **The recursion's KS at `deployRec` from V\*'s parts at a layout**: `ksA_deployed_reg`'s chain with every code fact
about the readers and V* from V*'s parts (`VStar.codeFacts_of_parts`), where `deployRec` brings the layout's readers
(`readsThrough_deployRec`), its V* statements number `nS` (`vstarLen_deployRec`), every part's SHA-512 gadgets compute
SHA-512 (`PartCallsTrue`), the readers read the layout's public rounds off the run (`ReadsPub`), the last inner round
opens (`FinalOpens`), and V*'s parts meet `VStarParts` at the dense, holo and sparse shapes. -/
theorem ksA_deployed_parts (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (L : Claims.Recursion.VStar.Layout) (hb : b.rd = readersOf L) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI)
    (hT : ∀ s, Claims.Recursion.VStar.PartCallsTrue FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (deployZkRec relO) L s)
    (hPub : Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) L)
    (hFin : Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) opening)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) L opening
      Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) L opening
      Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) L opening
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) :=
  let I := Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI
  have hR := readsThrough_deployRec FlockDeploy.sha512 I.V ivScope b L hb
  have hcf := Claims.Recursion.VStar.codeFacts_of_parts (refuses I.V ivScope b FlockDeploy.sha512) hR
    (vstarLen_deployRec I.V ivScope b) hT hPub hFin hD hH hS
  Claims.Recursion.VStar.KS_of_flockKS_played FlockDeploy.sha512 (deployRec I.V ivScope b) I (deployZkRec relO)
    opening FlockDeploy.decCoin hF (FlockDeploy.KS.plain_KSAbort rel htab decI hdec)
    (refuses I.V ivScope b FlockDeploy.sha512) (innerRounds I ivScope b) (sessionNums I.V ivScope b relO)
    (tableErrNonneg I.V ivScope b) (sessionRounds I ivScope b relO) (sessionAccepts I.V ivScope b)
    (sessionRuns_sha512 I ivScope b relO) hcf.1 hcf.2.2.1 hcf.2.2.2.2.1 hcf.2.2.2.1 hcf.2.2.2.2.2
    (roundsMargin_deployRec_V false FlockDeploy.sha512 I rfl ivScope b)
    (FlockDeploy.coinsShown_V FlockDeploy.sha512 false) hcf.2.1

end RecDeploy
