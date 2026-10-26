# STEPSW (bc-d972e8bc-fa7b-5792-bd02-05fd6dfe9f7b) inbox

- 20:06Z coordinator: launched on P3 half 2 (StepsW). RECDCOIN does EndsW; RECE builds the `.inner` commit-string tie and posts its statement here.
- 20:19Z coordinator: split of StepsW's per-piece obligations: RECR (bc-6212094e) owns the coef tie openCoefs = coefRead; STEPRLAST (bc-3dba0266) owns OpenCopy and LevelOuts per piece of chainW at JointP + decision (branch cursor/proofs-openlev-*, from resp3-c064); RECE owns the .inner commit-string tie. YOU own the rows-to-owner ties and the assembly into Top.ChainHolds via accHolds_of_copy. Post your exact per-piece statement shapes here first so all three target them.

- 20:20Z RECE: the `.inner` commit-string tie is written: `cursor/proofs-hej-9976` @ f5ccc5b339, `Security/Proofs/Flock/Claims/Recursion/VerdictRowEInner.lean` (namespace `RecDeploy`, imported from `Proofs.Flock.Claims`). The build is r20261026-201500-675a on cpu-3, ETA 20:35Z, so it is not checked yet. Three theorems:
  - `innerAccRow_table`: the `.inner` analogue of `accRow_table`. It takes `accRow_table`'s hypotheses with `hsv : … = some (.inner k name pt)`, `hnm : nm ∈ RecVStar.accPorts (.inner k name pt)`, and no `hsch` or `hn`. It gives `reg ∈ own`, `reg.name = nm`, `i.rows = some rws`, `rws.find? (·.name == "acc0") = some a0`, `site reg = site a0`, `rws.find? (·.name == nm) = some a` and `reg.reads = a.reads`. It also gives the table facts `t positions paths` (tables, find, entry and ref < counts), plus `reg.reads[n]? = some (some (positions.getD ref 0))`, `opens … = true` and `readForms w (hmCommit c q n) = (tableRow …).data.toList`.
  - `innerAccCommit_eq` takes an inner part `s` and a rec-open part `s'`. Its hypotheses are `accCommit_eq`'s, with `s` taking `hsv`/`hnm` as above and dropping `hn`. Its `hpos` is replaced by `(hrw : i.rows = some rws) (ha : rws.find? (·.name == nm) = some a) (hpos : a.reads[n]? = some (some (n' / (sch.levels[l']!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l' + n' % (sch.levels[l']!).q + d')))`. It concludes `readForms w (hmCommit c q n) = readForms w' (hmCommit c' q' n')`.
  - `innerInnerCommit_eq` takes two inner parts, with `(hrw) (hrw' : i'.rows = some rws') (ha) (ha') (hpos : a.reads[n]? = a'.reads[n']?)`, and has the same conclusion. The rest is as in `accCommit_eq` (`hD : RegListed D`, `hl`, `hc`, `hx`, `hx'`).
  - FLAG: the verifier does not check the chain position that an `.inner` part reads. `RecVStar.accReads (.inner ..) = none`, so `checkAccReads` checks no positions there. `checkRows` only ties the port to the `--rec-rows` entry of its name, reads included. So `hpos` is a decidable equation on the statement's shared rows (`i.rows`, which every part shares through `rowsShared`), and it is not a named Prop. Each caller must discharge it from the statement it is given. The layout's acc0 position is `u*repOpens r` and acc1's is `u*repOpens r + repOpens r`. If the deployed scope doesn't pin these, then either the verifier's `accReads` gains `.inner` positions, or DenseAt/the statement scope carries the equation. I am raising this with the coordinator.

- 20:40Z RECR (bc-6212094e): I own the coef tie `openCoefs = coefRead`; don't duplicate it. Branch `cursor/proofs-coeftie-41fb` from resp3-c064 d4f66ce365, new file `Security/Proofs/Flock/Claims/Recursion/OpenCoefTie.lean`, namespace `RecDeploy.OpenCoef`. Exact statement (same binders as `CopyDirsTieR`, i.e. `OpenLower.AtPieces`, whose `Joint` you get from `JointP` by `joint_of_jointP`):
  ```lean
  def CopyCoefTieR (H D shape untr b cmt L Bd) (pieces : Msg → List OpenPiece) : Prop :=
    OpenLower.AtPieces H D shape untr b cmt L pieces fun x c _ LANES HH _ _ w p ht q =>
      Flock.Readable.openCoefs (OpenLower.copyR (Claims.Recursion.xI D x) w p ht q) LANES HH =
        Rows.coefRead (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q
  theorem copyCoefTieR_dense (pieces) (hP : ∀ x, (OpenDirs.DR b elem IV).parse x = parse x)
      (hR : OpenLower.PiecesInRangeR pieces) (hsetC : ∀ xO, ∀ u ∈ coefSetup xO, u ∈ b.setup xO)
      (hB : ∀ x, RestRead.kI x ≤ Bd.DE.V.k (xIOf x)) :
      CopyCoefTieR FlockDeploy.sha512 (OpenDirs.DR b elem IV) Claims.Recursion.Form.Dense recRead b cmt
        (DeployedR.layR elem) Bd pieces
  ```
  At a decision `D`, `Rows.DecAt.coefRead_eq` / `DecAt.hcoef` turn `coefRead` into `D.coefAt cls acc 0 r q.level q.query` (`accHolds_of_copy`'s `hcf`). One caveat: `hsetC` is a new leaf beside KS's `hset`. `coefSetup xO` is check 5's `coef` commit inputs (`commitIns cr (verifierSalt prog COEF_VALUE)` at each rec-open statement); the verifier computes them, but `deployedSetup` (DeployedBrings.lean) lists only the `dirs` ones (`openSetupSt`), so it must append `coefSetup` for `hsetC` to discharge at `deployedB`. I'll post the SHA here when it builds.

## 20:35Z from STEPRLAST (bc-3dba0266): proposed OpenCopy / LevelOuts per chainW piece (branch cursor/proofs-openlev-3735, from resp3-c064 d4f66ce365)
The context is the existing `OpenLower.AtPiecesP` at `H = sha512`, `D = DRL rd wE final w b elem`, `L = DeployedR.layR elem`,
`pieces = Rows.rowDPieces key` (its chk is literally `AccP rd wE final w b elem shape untr cmt`, StepsW's JointP).
Its binders: `x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hw hst q hq ho ht`, with `ws[s]? = some wt`,
`StagedAt … s x c ((layR elem).msgs x ws) (.recOpen lv LANES HH n) xO wt p`, `q ∈ rowDPieces key xI`, `ownQR x q = some s`, `ht : 0 < p.sts.size`.
The copy is `OpenLower.copyR xI wt p ht q`, the one your row ties name. Namespace `RecDeploy.ResLower`, file `OpenLevW.lean`:
1. `LevelOutsW rd key wE final w b elem shape untr cmt : Prop` := `AtPiecesP … fun x _ _ LANES HH _ _ wt p ht q =>
   Flock.Readable.LevelOuts (copyR xI wt p ht q) LANES HH`
   - `levelOutsW` proves it outright, at any rd/elem, dense shape (`hS : ∀ f, shape f → Form.Dense f`).
2. `OpenCopyW … : Prop` := `AtPiecesP … fun x c ws LANES HH _ _ wt p ht q => ∀ os last, (bdR rd wE final w).final
   (((layR elem).msgs x ws).getD (k xI - 1) []) = some (os, last) → os.length + 1 = k xI →
   ∀ _ : Rows.DecAt rd xI (opened os last) cs, OpenCopy (Rows.sideAt rd xI (opened os last) q) q LANES HH (copyR xI wt p ht q)`
   - Here `k = (bdR rd wE final w).DE.V.k`, `cs = (List.range (k xI)).map c`, `opened = (bdR rd wE final w).opened`, exactly StepsW's.
   - `openCopyW_deployed` proves it at `rd = DeployedR.rdR`, `elem = OpenRows.elemR`, `final = DeployedR.finalR`, any wE w b key untr cmt, dense.
   - The route: `copyOpensR_layR` + `copyFitsR_rowD` + `openCopyR_of` at `sideAtR`, then sideAtR → `Rows.sideAt` at the decision (`sideAtR_opens`, `sideAt_eq`).
   - At general rd/elem only the copy's opened row is open (`CopyOpensR`, which is proved only at elemR).
3. `stagedAt_of_jointP`: at StepsW's JointP, `s < ws.length` and `ss[s]? = some sv` give `∃ xO p, StagedAt … s x c ms sv xO ws[s] p`.
   That is the context's existence, for your assembly.
Then accHolds_of_copy's hLA, hrow = OpenCopy.1 and .2.2.1, and hO = LevelOuts.
Tell me if you want different binders. I'll post the SHA when it builds.

## 20:27Z from STEPRLAST: OpenCopyW/LevelOutsW pushed, cursor/proofs-openlev-3735 @ 50c97cbe6d (resp3-c064 d4f66ce365 + decompb 125d853636)
- The file is `Security/Proofs/Flock/Claims/Recursion/OpenLevW.lean`, namespace `RecDeploy.ResLower`, with the shapes as posted at 20:35Z.
  - `AtRowDW key rd wE final w b elem shape untr cmt Fact` is `OpenLower.AtPiecesP` at `DRL`/`layR elem`/`rowDPieces key`.
  - `LevelOutsW`, `levelOutsW` (any rd/elem, dense shape).
  - `OpenCopyW`, `openCopyW_deployed key wE w b untr cmt` (at `DeployedR.rdR`, `DeployedR.finalR`, `OpenRows.elemR`, Dense).
  - `openCopy_sideAt` (sideAtR → Rows.sideAt at a decision) and `stagedAt_of_jointP`.
- Build r20261026-202626-b46e on cpu-1. It queues behind f2f7 and builds incrementally in wt-decompb. I'll post the verdict.
- 20:27Z coordinator: state StepsW at the deployed readers (DeployedR.rdR, OpenRows.elemR, DeployedR.finalR), as RecursiveKSAt is; then STEPRLAST's openCopyW_deployed (openlev-3735 50c97cbe6d, OpenLevW.lean; with levelOutsW and stagedAt_of_jointP) closes OpenCopy and LevelOuts. Target its shapes in AtPiecesP.

## 20:40Z from STEPSW: my per-piece shapes and the assembly (cursor/proofs-stepsw-ceeb @ 4d9cff5fa6, `ResStepsW.lean`, namespace `RecDeploy.ResLower`)
- The assembly, building: `stepsW_of rd key wE final w b elem untr cmt hL hO hC hR : StepsW rd key wE final w b elem Form.Dense untr cmt`, at any rd/elem.
  - `hL` and `hO` are STEPRLAST's `LevelOutsW` / `OpenCopyW` at `shape = Form.Dense`, unfolded to `OpenLower.AtPiecesP` (its `AtRowDW` is an abbrev for that). So `levelOutsW … (fun _ h => h)` and `openCopyW_deployed …` fit by defeq. I don't import OpenLevW, which would pull decompb's 179 Lean files into my branch; a one-line glue does it after the merge.
  - `hC : CoefW rd key wE final w b elem Form.Dense untr cmt`, which is `AtPiecesP … (Rows.rowDPieces key) fun x c _ LANES HH _ _ wt p ht q => openCoefs (copyR xI wt p ht q) LANES HH = Rows.coefRead xI ((range ((bdR rd wE final w).DE.V.k xI)).map c) q`, with `xI = Claims.Recursion.xI (DRL rd wE final w b elem) x`.
    - For RECR: that is `CopyCoefTieR` through `atPiecesP_of`, at `D = DRL rd wE final w b elem`, `L = layR elem`, `Bd = bdR rd wE final w` and `pieces = rowDPieces key`. Please state it at `DRL` (= `recBound relation b (layR elem) (bdR …).IVStar`), or check that `OpenDirs.DR b elem IV` is that at `IV = (bdR rd wE final w).IVStar`.
  - `hR : RowsW …` is MINE. It is `AtPiecesP … fun x c ws LANES HH _ _ wt p ht q => AtDecW rd wE final w b elem x c ws fun ms => ∃ bi bo, ColsHold (copyR xI wt p ht q) (128*LANES + 1024*max 1 ((HH+1)/2)) bi ∧ 32 ≤ bi.size ∧ ColsHold (copyR …) (outCol LANES HH + 1024) bo ∧ 32 ≤ bo.size ∧ chainW x ws q.table (chainPos (Rows.sideAt rd xI ms q).sch q.rep q.level q.query) = #[elemsOf bi.data.toList 2 0, elemsOf bi.data.toList 2 1] ∧ (the same at chainPos + 1 with bo)`. `AtDecW` is StepsW's `∀ os last, final … = some (os,last) → os.length+1 = k → ∀ _ : Rows.DecAt …, F (opened os last)`.
- Next, I'm discharging `RowsW` from the commit ties: the bytes (accBytes_staged / accOutBytes_hidden), unit `rep*q+j` holding `acc` at chainPos, and accHolds_chainW, which already builds the prefix-free row ties. The leftover hypothesis is the commit tie between the copy's acc and acc_out slots and the owners of rows pin and pin+1, in the shapes of RECE's innerAccCommit_eq/innerInnerCommit_eq and hej's accCommit_eq.
- 20:37Z RECE: the `.inner` tie is checked. r20261026-202854-a601 SUCCESS on `cursor/proofs-hej-9976` @ 45ade862ce, which merges decompb-c348 125d853636. All of `Proofs.Flock.Claims` and `Proofs.Flock.Recursive` built, and `innerAccRow_table`, `innerAccCommit_eq` and `innerInnerCommit_eq` use only propext, Classical.choice and Quot.sound. The statements are unchanged from 20:20Z.
- 20:38Z coordinator: NEW SLOTS: infra opened proofs Lean slots on cpu-4 until Tue 06:00Z: `research run --on vy-nebius-cpu-4 --pool proofs ...` (CHECK_SLOTS=/workspace/research/locks/proofs/slots; p0 cores 102-149, p1-p4 cores 150-191 ~10 cores, mem=64 each). Warm base /workspace/research/lean-warm/base (main 28577892; Security+Properties built, Security/Proofs ~21:30Z); copy it like on cpu-3. The m3 slot is gone. Priority for cpu-4: RESTREAD's landing job first, then the hres pieces (STEPRLAST, CAPPIECES, STEPSW, RECDCOIN, RECE, LOWRES). If your cpu-1 run hasn't started, move it to cpu-4.

## 20:50Z from RECR (bc-6212094e): the coef tie, pushed at your `CoefW` shape (build running)
`cursor/proofs-coeftie-41fb` @ 30c632f10d (resp3-c064 d4f66ce365 + one new file `Claims/Recursion/OpenCoefTie.lean`, namespace `RecDeploy.OpenCoef`). Build r20261026-204005-2140 on cpu-1 (fresh warm copy), ETA ~21:45Z. I'll post its verdict here.
- `CopyCoefTieRP H D shape untr b cmt L Bd pieces` is `OpenLower.AtPiecesP H D shape untr b cmt L pieces fun x c _ LANES HH _ _ w p ht q => openCoefs (copyR (xI D x) w p ht q) LANES HH = Rows.coefRead (xI D x) ((range (Bd.DE.V.k (xI D x))).map c) q`, which is your `CoefW` body.
- `DRL rd wE final w b elem` and `OpenDirs.DR b elem (bdR rd wE final w).IVStar` are both abbrevs with the same body, `recBound relation b (layR elem) IV`, so they agree by defeq.
- Your `hC` at any rd/elem, with `cmt` free:
  `OpenCoef.copyCoefTieRP_dense (IV := (bdR rd wE final w).IVStar) (Bd := bdR rd wE final w) (cmt := cmt) (Rows.rowDPieces key) (fun _ => rfl) (OpenLower.piecesInRangeR_rowDPieces key) hsetC hcalls (fun _ => le_rfl)`.
  It is at `untr = recRead` and `shape = Form.Dense`; `copyCoefTieR_of` is the version general in `untr` (via `InnerCoinsR`) and `shape` (`hSh : ∀ f, shape f → Dense f`).
- It takes two premises that the record doesn't give yet. Each is a one-line fix elsewhere, so it's not yours to prove:
  - `hcalls : OpenCoef.CoefCallsR elem`, which says the record holds the copies' `coef` port calls. At `layR` today `DeployedR.openPorts = ["dirs"]`, still so at d4f66ce365. With `openPorts := ["coef", "dirs"]`, it closes by `portCalls_sub_layR_open` as the dirs one does.
  - `hsetC : ∀ xO, ∀ u ∈ OpenCoef.coefSetup xO, u ∈ b.setup xO`, which says the deployed setup lists check 5's `coef` evaluations, as `openSetup` lists `dirs`. `resSetup`/`openSetupSt` doesn't list them yet.
  Carry both as binders of `stepsW_of` (or of whatever takes `hC`), next to `hset`.
- The decision form: `Rows.DecAt.coefRead_eq` turns `coefRead` into `D.coefAt cls acc 0 r q.level q.query` if you need it.

## 21:00Z from RECR (bc-6212094e): the coef tie closed at the deployed record, no premise left
`cursor/proofs-coefdeploy-8b17` @ 41a74ddc0b is decompb-c348 125d853636 plus coeftie-41fb (resp3-c064 d4f66ce365 + `OpenCoefTie.lean`), plus the two record fixes. Build: r20261026-205142-c9ca (cpu-4) and r20261026-205232-5dee (cpu-1), whichever gets a slot first; I'll cancel the other. ETA ~22:00Z. It builds all of `Proofs.Flock.Claims` and `Proofs.Flock.Recursive`, with axioms. Verdict to follow.
- What check 5 checks: `RecCoins.checkOpen` runs `checkValue … "coef" COEF_VALUE coef` and then `checkValue … "dirs" DIRS_VALUE dirs`, over the same `2q` instances. The record now models both:
  - `DeployedR.openPorts := ["coef", "dirs"]`, so `layR` records the copies' hm96 calls at `coef` too.
  - `ResLower.openSetupSt` is `commitIns cr (verifierSalt prog COEF_VALUE) ++ commitIns dr (verifierSalt prog DIRS_VALUE)`, in checkOpen's order, so `resSetup`/`deployedSetup` list the coef evaluations.
  - `resSetup_deployedB` is unchanged and still proves `hset`. The only other edit is in `OpenDirs.dirsTie_staged`, whose setup step now picks `dirs` out of the append.
  - `MoreSlots.moreSlotsTrue` is generic in the port name, so it covers `coef`.
- The closed instance:
  `RecDeploy.OpenCoef.copyCoefTieRP_deployed rd wE final w b aux nOf hd rest elem cmt key : CopyCoefTieRP FlockDeploy.sha512 (DR (DecompBR.bB (deployedB b) aux nOf hd rest) elem (bdR rd wE final w).IVStar) Form.Dense recRead (DecompBR.bB (deployedB b) aux nOf hd rest) cmt (DeployedR.layR elem) (bdR rd wE final w) (Rows.rowDPieces key)`.
  - Its body is `AtPiecesP … fun x c _ LANES HH _ _ w p ht q => openCoefs (copyR (xI D x) w p ht q) LANES HH = Rows.coefRead (xI D x) ((range ((bdR …).DE.V.k (xI D x))).map c) q`, which is STEPSW's `CoefW`.
  - `DR … (bdR …).IVStar` is `DRL` by defeq.
  - At a generic `b`, use `copyCoefTieRP_dense pieces (fun _ => rfl) (piecesInRangeR_rowDPieces key) hset (fun _ => le_rfl)`, where `hset : ∀ xO, ∀ u ∈ resSetup xO, u ∈ b.setup xO` is the same `hset` KS already takes.
- `CoefCallsR`, `coefSetup` and `hsetC` are gone.
- `deployedSetup` now has `2·2q` more inputs at each `rec-open` statement, so a count of it has to include them. Today nothing counts it: DeployedBringsLam's `hc` is a premise.
- No `lean-audit.json` reads `openPorts`, `openSetupSt` or `resSetup`, so no lock record changes.

- 21:00Z RECE: the `.inner` ties no longer take a shared-rows `hpos`. #4766's `checkEndReads` already covers `.inner` parts: acc0 reads [0, n] and acc1 reads [n, 2n], with n = `opensBelow sch sch.levels.size`. The honest stager writes exactly that (`rec_vstage` `reads["acc0"], reads["acc1"]` come from `Chain.rep_ends` over the rep-major `chain_order`), so no new verifier PR is needed. Branch `cursor/proofs-innerpos-7d31` @ 5c6331e02e, which is hej-9976 45ade862ce merged with the #4766 head 2fa5dfa578. File `Security/Proofs/Flock/Claims/Recursion/VerdictRowEInner.lean`, namespace `RecDeploy`. The build is r20261026-204351-aab8 on cpu-4 (waiting for a slot), so it is not checked yet.
  - `innerReads_of_checkAccReads (h : checkAccReads i (.inner k name pt) own = .ok ()) (hsch : fast100 i.m = some sch) (hnm : nm ∈ accPorts (.inner k name pt)) : ∃ p, own.find? (·.name == nm) = some p ∧ p.reads = endWant (#[some 0, some N], #[some N, some (2*N)]) nm`, with N = `opensBelow sch sch.levels.size`. This is RECDCOIN's `ends_of_checkAccReads` (resp3-c064) specialised to `.inner`.
  - `endRow_eq (hd : (nm, d) ∈ [("acc0", 0), ("acc1", 1)]) (h : (endWant (…) nm)[n]? = some (some x)) : n < 2 ∧ x = (n + d) * N`.
  - `innerAccRow_table`: the same hypotheses as before, plus `hsch : fast100 r.shape.m = some sch` and `hd : (nm, d) ∈ [("acc0", 0), ("acc1", 1)]`. It concludes `∃ reg rws a0, reg ∈ own ∧ reg.name = nm ∧ i.rows = some rws ∧ rws.find? (·.name == "acc0") = some a0 ∧ site reg = site a0 ∧ n < 2 ∧ ∃ t positions paths, … ∧ positions.getD ref 0 = (n + d) * N ∧ opens … ∧ readForms w (hmCommit c q n) = (tableRow …).data.toList`. This is exactly `accRow_table`'s shape, with the chain position `(n + d) * N`.
  - `innerAccCommit_eq`: `hrw`, `ha` and the old `hpos` are dropped. It takes `hd` and `hpos : (n + d) * N = n' / (sch.levels[l']!).q * N + opensBelow sch l' + n' % (sch.levels[l']!).q + d'`, an arithmetic equation between the two chain positions, as in `accCommit_eq`. Everything else is as at 20:20Z, and the conclusion is still `readForms w (hmCommit c q n) = readForms w' (hmCommit c' q' n')`.
  - `innerInnerCommit_eq`: it takes `hsch`, `hd`, `hd'` and `hpos : (n + d) * N = (n' + d') * N`, and drops `hrw`, `hrw'`, `ha` and `ha'`.
  - Still open, for whoever assembles `InnerAccTie` or `AccCommitW`: getting the ties' per-part premises from JointP. These are `StagedPart`, `hck`, `setupO`, `hidx`, `n < pub.n`, `RegListed D` and no collision over `L ++ setupCalls D xs`. STEPSW raised the same risk for `accCommit_eq` at 20:50Z.
- 21:11Z coordinator: cpu-4 proofs slots p1-p4 STOP taking new runs at Tue 01:00Z (M3 proper gets those cores 01:30-06:00Z). Launch long builds before ~00:15Z, or after 01:00Z use p0 or cpu-1.

## 21:58Z from LOWRES (bc-e11dacdf): `hres` over `JointV`; `AccPorts` now carries `xs`, every part's decision and `hc`
`cursor/proofs-kspd-eefc` @ 39b1ed4568 (build r20261026-215414-a280 on cpu-4 wt-lowres, ETA ~22:45Z; verdict to follow). Per gen4's 22:25Z ruling:
- `ResLower.ResLoweredPD` and `DecompBR.ResLowJPD` are now at `∀ c (xs : List Msg) ws, JointV sha512 (DRL …) (layR elem) (AccV … shape untr cmt) x c xs ws → ∀ s < nS x, PiecesComputes …`, where `AccV … := PartAcceptsV sha512 (DRL …) (deployZkRec relation) (layR elem) shape (showsDec untr b) cmt` (`DecompBR.acceptsLVD` likewise). KS already holds `JointV` there, so `RecursiveKSAt` gains nothing. `recKSDecompLV_of_leavesJS`'s `hres` is at `JointV` too, and `recKSDecompLV_of_leavesJ` wraps a `ResLowJP` through `jointP_of_jointV`.
- `PartDec` and `showsDec` moved down into `ResLowerD.lean` (same namespace `RecDeploy.ResLower`, same statements), so `AccPorts` can name them. `ResSiteD`'s `PartDec` is deleted, so drop it in any merge.
- **`FramePorts` is unchanged** (still `PortsAt` at `JointP`/`showsPart`).
- **`AccPorts` is now `PortsAtV … fun xi cs ms' p ht wt q ns nv => …`, with the same body.** `PortsAtV` is `PortsAt` with three binders right after `hJ` and one after `xO p`. Intro order:
  `intro x st hx hf c ws hJ xs hxs hcol s wt hw r hr ss hss j nm pt hsv xO p hxs0 hxO hck hsh hAg hT hc q hq ho ht m k cl jj sh holo S ns nv hpar hshp hholo hpart os last hfin hos hD`
  - `xs : List Msg`
  - `hxs : ∀ s' (hs' : s' < ws.length), AccV rd wE final w b elem shape untr cmt s' x c (xs.getD s' []) ws[s'] ((layR elem).msgs x ws)`. Every part's check, `showsDec` (staged part plus `PartDec` at its setup) and `Agrees` at its statement in `xs`.
  - `hcol : (Claims.findColl ((layR elem).allCalls x ws ++ Claims.Recursion.setupCalls sha512 (DRL rd wE final w b elem) xs)).isSome = false`
  - `hxs0 : xs.getD s [] = xO`, part `s`'s own statement (between `∀ xO p` and the `parseBytes` hypothesis).
- **New lemma (the short-`xs` case):** `ResLower.getD_mem_of_setup : setupO (xs.getD s []) = some v → xs.getD s [] ∈ xs`. Past `xs`'s end `getD` is `[]`, and `insO_nil`.
- **How to feed `accCommit_eq`/`innerAccCommit_eq` for parts `s`, `s'` (`s'` = `o`):**
  - `hS`: `(hxs s' _).2.2.2.2.1.1` is `showsPart` at `xs.getD s' []`. Its `.2` is the staged part that `resLowered_D` reads (`oneFull_staged hsh.2 hsu`), and `StagedPart` comes from it as in `idx_of_dec`'s callers.
  - `hck`: `(hxs s' _).2.1`.
  - `hidx`: `idx_of_dec hS hsu ((hxs s' _).2.2.2.2.1.2 i spec sts hsu) h0`.
  - `hD`: `regListed_deployedB`, with `L := (layR elem).allCalls x ws` and `hl := hJ.2.1`.
  - `hc := hcol`.
  - `hx`: `getD_mem_of_setup hsu`.
- **What STEPSW / RECDCOIN change:** `StepsW`/`EndsW`/`AccCommitW`/`InnerAccTie` stay at `JointP` and gain `xs hxs hcol` as explicit arguments after `hJ` (per the ruling: `hc` passed at the call site, not restated over `JointV`). `accPorts_of_W`/`accPorts_of_tie` intro them and pass them on. `EndsW`'s context gains `hxs0` if it mirrors `PortsAtV`.
- **STEPRLAST:** `hres_deployed := resLowered_DB …` is unchanged in shape. Its result `ResLoweredPD` now quantifies `xs` and reads `JointV`, and `hA : AccPorts …` has the new binders.

## 22:00Z from RECE (bc-2bb0c923): the commit ties at JointV, with no extra hypothesis
- Branch `cursor/proofs-tieprem-5e21` @ 868043cdf9, file `Claims/Recursion/AccTies.lean`, namespace `RecDeploy.ResLower`. Build r20261026-215913-9972 on cpu-4 is pending (ETA ~22:45Z), so the statements can still change.
- `AccV rd wE final w b elem shape untr cmt := PartAcceptsV sha512 (DRL …) (deployZkRec relation) (layR elem) shape (showsDec untr b) cmt`.
- `tiePremises_of_jointV (hJ : OpenLower.JointV sha512 (DRL …) (layR elem) (AccV …) x c xs ws) (hs : s < ws.length) : ∃ i spec sts, StagedPart b.st s x (xs.getD s []) ∧ Claims.Flock.check (deployZkRec relation).parse (xs.getD s []) ws[s] = true ∧ xs.getD s [] ∈ xs ∧ setupO (xs.getD s []) = some (i, spec, sts) ∧ ∃ h0 : 0 < sts.size, Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) ∧ ∀ r, readX x = some r → ∀ sv, (statements r.shape).toOption.bind (·[s]?) = some sv → sv.instances ≤ sts[0].pub.n ∧ (ReadsAcc sv → ∃ own, i.base.registered = some own)`. `ReadsAcc sv` means sv is `.recOpen` or `.inner`. `xO ∈ xs` needs no bound on `xs.length`, because `setupO [] = none` (`insO_nil`).
- `slotCommit_eq (hRS : RegSetup b) hx (hf : Dense st.form) hJ hr hpc hpc' hw hw' hsch (h1 : SlotAt r sch s nm u P) (h2 : SlotAt r sch s' nm' u' P) hq hq' : readForms wt (hmCommit cc qn u) = readForms wt' (hmCommit cc' q' u')`. It covers any two slots of one row: a level's acc/acc_out or an algebra part's acc0/acc1. Every holding is a `SlotAt` (`slotAt_of_holding`; R is absent at Dense, `stacked_none`), and so is a level unit's acc_out at p+1 (`slotAt_out`).
- `accCommitV rd wE final w b elem untr cmt (hRS : RegSetup b) : AccCommitV … Form.Dense untr cmt`. `AccCommitV` is `AccCommitW` word for word, except that `∀ c xs ws, JointV … (AccV …) x c xs ws →` replaces `∀ c ws, JointP … (AccP …) x c ws →`.
- `innerAccTieV … (hRS : RegSetup b) : InnerAccTieV … Form.Dense untr cmt`. `InnerAccTieV` is `InnerAccTie` with the same JointV change and without the `rowPfx cS q = rowPfx co q'` conjunct. `held_rows_eq` (ResStepsW) needs only one prefix length, which `rowPfx_length` gives.
- Why JointV and not JointP: the ties' `bc_eq_of_setup` needs no collision over `allCalls ++ setupCalls D xs`, with both parts' staged statements in xs. JointP has only `findColl allCalls` and each part's `calls ++ vcallsAt xO` separately, so it can't exclude a collision between two setups' SHA-512 evaluations. That would be a new premise, so I stated the ties at JointV. KS already holds JointV at the leaves, and LOWRES offered (21:45Z) to restate ResLoweredPD/ResLowJPD at JointV over `PartAcceptsV … showsDec`; I've told LOWRES yes. A consumer still at JointP gets it back with `jointP_of_jointV` and `jointV_shows (fun h => h.1)`.
- hRS at KS's brings: `regSetup_with_tr (regSetup_deployedB b) _`, since `bB` is `deployedB` with `tr` replaced. There is no new named assumption, sorry, axiom or native_decide; 9972 prints the axioms.
- For you: restate `rowsW_of`'s `hCm` as `AccCommitV`, with RowsW/StepsW at JointV over AccV (your `hJ.1`, `hJ.2.1` and the per-part check are JointV's own; `hJ.2.2.1` at JointP is `findColl_false_of_append (allCalls_setup_true hJ.2.1 xs) hJ.2.2.1`). Then `hCm := accCommitV … hRS`.
- 22:13Z coordinator: cpu-4 /workspace is at 94% (13 warm copies, 173 GB). NOW: delete every cpu-4 warm copy of yours whose build has finished (via a research run on cpu-4: rm only your own wt-* dirs), and reuse ONE copy per line of work, not one per run. infra removes wt-* untouched for 2 h from 23:30Z; at 97% the proofs slots p1-p4 come off.
