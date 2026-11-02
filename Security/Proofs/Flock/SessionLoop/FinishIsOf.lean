import Proofs.Flock.SessionLoop.FinPiopRet
import Proofs.Flock.SessionLoop.FinHmkFw
import Proofs.Flock.SessionLoop.FinRepRows
import Proofs.Flock.SessionLoop.FinRepTau

/-!
# `FinishIs` from the rows' consumed facts and level 0's openings

`finishIs_of_abc` takes (a), (b) and (c); (a) and (b) hold (`finCommits_of`, `finPoints`), and (c) is
`finRepProofs_of_view` at `FinRepErased`, `FinRepFinal` and `FinRepXs`. Here every part of (c) comes from
`FinRepRows`'s four consumed facts (`finRepRows`) and `FinRepL0`:

* **`Loop.finFc_of_rows`**: `FinFc` where every table's rows hold at its base: `swSrcE_hmk_fw` at `final_c`'s pad, its
  `hxs` from `finRepXs`, its `hl0` from `swSrcE_l0_views`, the firewall's level-0 commitment from `cms_cm0` at the
  switched cut's (`cmsCut_ok`), its pads commitment from `padsView_ic` at the switched cut's (`icCut_ok`), and its rep
  from `proofFile_ok` and `finRepCut`.
* **`Loop.finRepRounds_of_erased`**: (d) from `FinRepErased` alone, by strong induction on the stream
  (`viewRep_rounds`, `viewRep_size`), so `finRepOpen` reads no (c).
* **`Loop.finishIs_of_consumed`**: at SessionLoopZKDeployed's premises, `FinRepMsgs`, `FinRepLanes`, `FinRepTarget`,
  `FinRepRestart` and `FinRepL0` give `FinishIs`.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 4000000 in
/-- **`FinFc` where the rows hold**: at `FinFc`'s premises, `swSrcE_hmk_fw` at `final_c`'s pad, with the firewall's
level-0 and pads commitments matched to the switched cut's through the views' (`cms_cm0` and `cmsCut_ok`;
`padsView_ic`, which reads the rows, and `icCut_ok`). -/
theorem finFc_of_rows (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) (hRows : FinRepRows vs L₀ src (realOf src) wk x w) : FinFc vs L₀ src wk x w := by
  intro ν t κ m₀ ρ₀ ops hF s0 spec hsx schs hs cms hcms j r hexts hprev tb hi q f hRep extras hext ic hic l hl
    res tr₁ l₁ hpiop hcap hroot
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    rw [hspec]
    omega
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
  obtain ⟨schs', hs', hschZ⟩ := Masked.schedsPZk_live hx
  have hsj : schs[j.val]! = ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [← Except.ok.inj (hs'.symm.trans hs)]
    exact hschZ j
  -- the views' commitments and pads
  obtain ⟨schsV, hsV, cmsV, hcmsV, -, hpads⟩ :=
    finCommits_of vs L₀ src (realOf src) wk x w hsc hR hS0 ν t κ m₀ ρ₀ ops hF s0 spec hsx
  have e0 : schsV = schs := Except.ok.inj (hsV.symm.trans hs)
  rw [e0] at hcmsV hpads
  -- level 0's commitment: the firewall's is the views', which is the switched cut's
  obtain ⟨cmsC, hcmsC, -, -, -, hcm⟩ := cmsCut_ok vs w ν t hA schs hcmsV
  have hCe : cmsC = cms := Except.ok.inj (Prod.mk.inj ((hcmsC #[]).symm.trans (hcms #[]))).1
  obtain ⟨hcap0, hlay0⟩ := cms_cm0 vs src x w hsc hR hS0 hsx hs hcmsV hk hRep
  rw [hk2] at hcap0 hlay0
  have hcm0 : (FlockComplete.eraseC tb.cm0.toC).cap = cms[j.val]!.cap ∧
      (FlockComplete.eraseC tb.cm0.toC).layers = cms[j.val]!.layers := by
    rw [← hCe]
    exact ⟨hcap0.symm.trans (hcm j.val).1.symm, hlay0.symm.trans (hcm j.val).2.symm⟩
  -- the firewall's rep at the opened coins
  obtain ⟨ic', p, tr, hic', hfw, -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hRep.2.2
  have hrun := finRepCut L₀ src (realOf src) wk x w hsc ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hk tb hi q _ hRep ic' p tr
    hic' hfw l hl
  -- the pads commitment: the firewall's is the views', which is the switched cut's
  obtain ⟨icV, hicV⟩ := hpads (2 * j.val + r.val) hk l
  obtain ⟨hn₁, hroot₁, hlay₁⟩ :=
    padsView_ic vs L₀ src x w hsc hR hS (hRows ν t κ m₀ ρ₀ ops hF) hsx hs hk hicV hRep hic'
  have hic2 := padsView_all vs w ν t hicV
  obtain ⟨icC, hicC, hn₂, hroot₂, hlay₂⟩ := icCut_ok vs w ν t hA (i := j.val) (r := r.val) j.isLt r.isLt _
    (fun l => by
      have e := hic2 l
      unfold padsView at e
      rw [hk2, hkr] at e
      exact e)
  have hIe : icC = ic := by
    have e' := hicC l
    rw [hsj] at e'
    exact Except.ok.inj (Prod.mk.inj (e'.symm.trans (hic l))).1
  rw [hIe] at hn₂ hroot₂ hlay₂
  have hicE : ic'.toC.ch.size = ic.ch.size ∧ ic'.toC.root = ic.root ∧ ic'.toC.layers = ic.layers :=
    ⟨hn₁.symm.trans hn₂, hroot₁.symm.trans hroot₂, hlay₁.symm.trans hlay₂⟩
  -- the extra claims' coordinates, and level 0's masked values
  have hxs := finRepXs L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hk extras
    (by rw [hk2]; exact hext) tb hi q f hRep
  obtain ⟨-, -, -, hg⟩ := hS0.realOf ν t j r hRep.1
  obtain ⟨hexT, -⟩ := hS0.realOf_lanes ν t j r hRep.1
  have hlt : (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc <
      (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64 := by
    unfold _root_.Flock.Zk.PadLayout.zp _root_.Flock.Zk.PadLayout.lc
    omega
  exact swSrcE_hmk_fw L₀ src wk x w vs hsc hR hServe hS hS0 hE hc hF hsx hs hcms j r hexts hprev hRep hext hic hl
    hpiop hcap hroot hxs
    (fun _ _ _ _ i hlo hhi => by
      have e := swSrcE_l0_views L₀ src wk x w vs hsc hR hServe hL0 hE hc hF hsx hs hcms j r hexts hprev hRep hext
        hic hl hlo hhi
      rw [hg, hexT] at e
      exact e)
    hcm0 hicE hrun _ hlt

set_option maxHeartbeats 1000000 in
/-- **(d) from `FinRepErased`**: where the run from the views is the firewall's up to `eraseZ` at every rep stream, by
strong induction on the stream each returns with the auditor's rounds (`viewRep_rounds`, the firewall's rep from
`proofsOf_at`, `proofFile_ok` and `finRepCut`), so every run from the views has them (`viewRep_rounds`) and ends at the
next rep's first round (`viewRep_size`). -/
theorem finRepRounds_of_erased (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS0 : Level0Src src x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) (hEr : FinRepErased vs L₀ src (realOf src) wk x w) :
    FinRepRounds vs L₀ src (realOf src) wk x w := by
  intro ν t κ m₀ ρ₀ ops hF s0 spec hsx schs hs cms hcms
  obtain ⟨schs', hs', cms', -, -, hpads⟩ :=
    finCommits_of vs L₀ src (realOf src) wk x w hsc hR hS0 ν t κ m₀ ρ₀ ops hF s0 spec hsx
  have e0 : schs' = schs := Except.ok.inj (hs'.symm.trans hs)
  rw [e0] at hpads
  have hext := (finPoints L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx).2
  obtain ⟨files, hp⟩ := proofsOf_ok_of_ctx L₀ src (realOf src) wk x w hF hsx
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  have hAll : ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
        ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p, tr), l') ∧
          tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k ∧ l'.size = Masked.repBase x (k + 1) := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih =>
    intro hk extras hex ic l hic hl
    obtain ⟨tb, hi, q, hRep⟩ := proofsOf_at hp (by rw [hrs]; exact hk)
    obtain ⟨ic', p, tr, hic', hfw, -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hRep.2.2
    have hcut := finRepCut L₀ src (realOf src) wk x w hsc ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk tb hi q _ hRep ic' p tr
      hic' hfw l hl
    obtain ⟨p', l', hv, -⟩ := hEr ν t κ m₀ ρ₀ ops hF s0 spec hsx schs hs cms hcms k hk extras hex ic l hic hl tb hi q _
      hRep ic' p tr hic' hcut
    exact ⟨p', tr, l', hv,
      viewRep_rounds vs L₀ src (realOf src) wk hsc hR hServe hE hc hF hsx hs hcms hext hpads hk
        (fun k' hk' => ih k' hk' (by omega)) hex hic hl hv,
      viewRep_size vs hsc hR ν t m₀ ρ₀ ops hsx hs cms hk hex ic hl hv⟩
  intro k hk extras hex ic l hic hl p tr l' hv
  exact ⟨viewRep_rounds vs L₀ src (realOf src) wk hsc hR hServe hE hc hF hsx hs hcms hext hpads hk
      (fun k' hk' => hAll k' (by omega)) hex hic hl hv,
    viewRep_size vs hsc hR ν t m₀ ρ₀ ops hsx hs cms hk hex ic hl hv⟩

set_option maxHeartbeats 1000000 in
/-- **`FinishIs` from the rows' consumed facts and level 0's openings**: at SessionLoopZKDeployed's premises, the four
facts `finRepRows` consumes give every table's rows at its base, so `FinFc` holds (`finFc_of_rows`) and with it
`FinRepErased` (`finRepErased_of_fc`), (d) (`finRepRounds_of_erased`), `τ` (`finRepTau`), its salt (`finRepTauSalt`)
and the pads' openings (`finRepOpen`); with level 0's openings (`FinRepL0`) they give `FinRepFinal`, so (c)
(`finRepProofs_of_view`, `finRepXs`), and (a), (b) and (c) give `FinishIs` (`finishIs_of_abc`). -/
theorem finishIs_of_consumed (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w) (hSrc : Loop.Src L₀ src x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    (hM : FinRepMsgs vs L₀ src (realOf src) wk x w) (hV : FinRepLanes vs L₀ src (realOf src) wk x w)
    (hT : FinRepTarget vs L₀ src (realOf src) wk x w) (hTP : FinRepRestart vs L₀ src (realOf src) wk x w)
    (hL : FinRepL0 vs L₀ src (realOf src) wk x w) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w := by
  obtain ⟨hS0, hS⟩ := hSrc
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hRows := finRepRows vs L₀ src (realOf src) wk x w hsc hR hServe hS0 hM hV hT hTP
  have hEr := finRepErased_of_fc L₀ src wk x w vs hsc hR hServe hS hS0 hL0 hE hc hRows
    (finFc_of_rows vs L₀ src wk x w hsc hR hServe hS hS0 hL0 hE hc hRows)
  have hD := finRepRounds_of_erased vs L₀ src wk x w hsc hR hServe hS0 hE hc hEr
  have hFin := finRepFinal_of vs L₀ src (realOf src) wk x w
    ⟨hL, finRepTau vs L₀ src (realOf src) wk x w hsc hR hServe hS0 hS hRows,
      finRepTauSalt vs L₀ src (realOf src) wk x w hx hS,
      finRepOpen vs L₀ src (realOf src) wk x w hx hS hRows hD⟩
  exact finishIs_of_abc vs L₀ src (realOf src) wk hsc hR hServe hE hc
    (finCommits_of vs L₀ src (realOf src) wk x w hsc hR hS0) (finPoints L₀ src (realOf src) wk x w hsc hR)
    (finRepProofs_of_view vs L₀ src (realOf src) wk x w hsc
      ⟨hEr, hFin, finRepXs L₀ src (realOf src) wk x w hsc hR⟩)

end Loop

end Claims.ZKServer
