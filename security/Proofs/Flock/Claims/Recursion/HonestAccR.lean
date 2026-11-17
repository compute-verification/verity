module

public import Proofs.Flock.Claims.Recursion.HonestIVStarR
public import Proofs.Flock.Claims.Recursion.HonestVerdictR
public import Proofs.Flock.Claims.Recursion.RestVerdictCustody
public import Proofs.Flock.Claims.Complete.ZkTables.Stmt
public import Proofs.Flock.Claims.JsonNorm
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
public import Proofs.Flock.Claims.Recursion.VerdictRowDCover
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Claims.Recursion.HonestRepR
import Proofs.Flock.Claims.Recursion.HonestStreams
import Proofs.Flock.Claims.Recursion.HonestVerdictFinal
import Proofs.Flock.Claims.Recursion.NextRCoins
import Proofs.Flock.Claims.Recursion.NextRRows
import Proofs.Flock.Claims.Recursion.NextRSetup
import Proofs.Flock.Claims.Recursion.RestLive
import Proofs.Flock.Claims.Recursion.RestVerdictCustody
import Proofs.Flock.Soundness.Refine.Seam

/-! The deployed verifier accepts the honest recursive prover, the honest half of `RecursiveKSAt`'s non-vacuity
(`honest_accepts_nextR`). -/

@[expose] public section

namespace FlockComplete.Run

open Flock FlockDeploy

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

def recCoins (o : Run H rel) : List Coin := List.ofFn o.r

def recLast (o : Run H rel) : Claims.Msg := RecDeploy.DeployedR.encLast o.x o.recPs.toList

theorem recCoins_length (o : Run H rel) : o.recCoins.length = (V false).k o.x := List.length_ofFn

theorem coinsAt_recCoins (o : Run H rel) : o.CoinsAt o.recCoins := by
  refine ⟨fun i hi => ?_, hlen_of_k o.setup (fun i hi => (o.recTabsOk i hi).1) o.recCoins_length⟩
  have hi' : i < (V false).k o.x := by rw [← o.recCoins_length]; exact hi
  show (List.ofFn o.r).getD i default = Claims.coinSeq o.r i
  rw [List.getD_eq_getElem _ _ (by simpa using hi'), List.getElem_ofFn, Claims.coinSeq, dif_pos hi']

noncomputable def recAsR (o : Run sha512 Readable.relation) (ys : ℕ → ℕ → RecDeploy.Salt) : List Claims.Msg :=
  KS.honestAsR Readable.relation RecDeploy.DeployedR.rdR
    (deployPlainTrimR Readable.relation RecDeploy.DeployedR.rdR).cols RecDeploy.DeployedR.wER RecDeploy.rowInDeployed
    RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wR o.recCaps
    (RecDeploy.DeployedR.filesReps o.spec o.sts o.recC₁ o.proofs o.recCoins o.recTs) ys
    (RecDeploy.DeployedR.lastOR (RecDeploy.DeployedR.filesOs o.spec o.sts o.recC₁ o.proofs o.recCoins o.recTs ys)
      o.recLast)

end FlockComplete.Run

namespace FlockDeploy

open Flock FlockComplete

theorem scheme_bounds {ms : MerkleScheme} (h : ms ∈ MerkleScheme.all) : ms.digestLen ≤ 64 ∧ ms.saltBytes ≤ 192 := by
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl <;> exact ⟨by decide, by decide⟩

theorem fast100_level_bounds (m i : ℕ) {lv : Flock.Level} (h : lv = ((fast100 m).getD default).levels.getD i default) :
    lv.q < 2 ^ 64 ∧ lv.q * (lv.d - lv.cap) < 2 ^ 64 ∧ lv.lanes < 64 := by
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, (s.levels.all fun lv =>
      decide (lv.q < 2 ^ 64 ∧ lv.q * (lv.d - lv.cap) < 2 ^ 64 ∧ lv.lanes < 64)) = true := by
    decide +kernel
  have hdef : (default : Flock.Level).q < 2 ^ 64 ∧
      (default : Flock.Level).q * ((default : Flock.Level).d - (default : Flock.Level).cap) < 2 ^ 64 ∧
      (default : Flock.Level).lanes < 64 := by decide
  subst h
  generalize hs : (fast100 m).getD default = sch
  by_cases hi : i < sch.levels.size
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
    cases hf : fast100 m with
    | none =>
      rw [hf] at hs
      subst hs
      exact absurd hi (Nat.not_lt_zero _)
    | some s =>
      rw [hf, Option.getD_some] at hs
      subst hs
      have hm : m < 40 := by
        rcases Nat.lt_or_ge m 40 with hm | hm
        · exact hm
        · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at hf
          cases hf
      have h := Array.all_eq_true.1 (key ⟨m, hm⟩ s hf) i hi
      simpa using h
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none]
    exact hdef

theorem fast100_levels_le (m : ℕ) : ((fast100 m).getD default).levels.size ≤ 7 := by
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide (s.levels.size ≤ 7) = true := by
    decide +kernel
  cases hf : fast100 m with
  | none => exact Nat.zero_le _
  | some s =>
    have hm : m < 40 := by
      rcases Nat.lt_or_ge m 40 with hm | hm
      · exact hm
      · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at hf
        cases hf
    simpa using key ⟨m, hm⟩ s hf

theorem levelFits_of {sh : Flock.Shape} {l : Flock.Level} {w : ℕ} {p : Flock.LevelProof}
    (h : FlockSoundness.Discharge.ZkView.LevelOk sh l w p) (hq : l.q < 2 ^ 64) (hs : l.q * (l.d - l.cap) < 2 ^ 64)
    (hw : w < 2 ^ 64) (hh : sh.hashLen < 2 ^ 64) (hsl : ∀ s, sh.saltLen = some s → s < 2 ^ 64) :
    RecDeploy.DeployedR.LevelFits p.rows p.siblings p.salts := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  refine ⟨by rw [h1]; exact hq, fun r hr => ?_, by rw [h3]; exact hs, fun b hb => ?_, ?_⟩
  · rw [h2 r (Array.mem_toList_iff.1 hr)]
    exact hw
  · rw [h4 b (Array.mem_toList_iff.1 hb)]
    exact hh
  · unfold FlockSoundness.Discharge.ZkView.SaltsOk at h5
    split at h5
    · rw [h5]; simp
    · rw [h5]; simp
    · obtain ⟨hsz, hall⟩ := h5
      exact ⟨by rw [hsz]; exact hq, fun b hb => by rw [hall b (Array.mem_toList_iff.1 hb)]; exact hsl _ ‹_›⟩

theorem opensFit_of_shaped {tabs : Array Flock.Setup} {pts : Array (Array F128)} {k : ℕ} {p : Flock.Proof}
    (hms : tabs[k / 2]!.merkle ∈ MerkleScheme.all)
    (hp : FlockVBridge.Family.ProofShaped (shapeAt tabs pts k) p) : RecDeploy.DeployedR.OpensFit p := by
  obtain ⟨hd, hsb⟩ := scheme_bounds hms
  have hh : (shapeAt tabs pts k).hashLen < 2 ^ 64 := lt_of_le_of_lt hd (by norm_num)
  have hsl : ∀ s, (shapeAt tabs pts k).saltLen = some s → s < 2 ^ 64 := by
    intro s hs
    have e : s = tabs[k / 2]!.merkle.saltBytes := by
      unfold shapeAt Flock.repFileShape at hs
      dsimp only at hs
      split at hs
      · cases hs
      · exact (Option.some.inj hs).symm
    rw [e]
    exact lt_of_le_of_lt hsb (by norm_num)
  have hL : ∀ i, ((shapeAt tabs pts k).sched.levels.getD i default).q < 2 ^ 64 ∧
      ((shapeAt tabs pts k).sched.levels.getD i default).q *
        (((shapeAt tabs pts k).sched.levels.getD i default).d -
          ((shapeAt tabs pts k).sched.levels.getD i default).cap) < 2 ^ 64 ∧
      ((shapeAt tabs pts k).sched.levels.getD i default).lanes < 64 :=
    fun i => fast100_level_bounds tabs[k / 2]!.m i rfl
  have hsz : (shapeAt tabs pts k).sched.levels.size ≤ 7 := fast100_levels_le tabs[k / 2]!.m
  obtain ⟨-, -, -, L⟩ := hp
  obtain ⟨-, -, -, -, -, -, -, -, -, h0, -, -, hrs, hr, -, hf, -⟩ := L
  refine ⟨levelFits_of h0 (hL 0).1 (hL 0).2.1 (Nat.pow_lt_pow_right (by norm_num) (hL 0).2.2) hh hsl, ?_,
    fun q hq => ?_, levelFits_of hf (hL _).1 (hL _).2.1 (by norm_num) hh hsl⟩
  · rw [hrs]
    exact lt_of_le_of_lt (by omega) (show (7 : ℕ) < 2 ^ 64 by norm_num)
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hq)
    exact levelFits_of (hr i hi) (hL _).1 (hL _).2.1 (by norm_num) hh hsl

theorem toArray_map_toList {α β : Type} (f : α → β) (xs : Array α) : (xs.toList.map f).toArray = xs.map f := by
  cases xs; simp

theorem honest_rdR (o : Run sha512 Readable.relation) :
    RecDeploy.DeployedR.rdR o.recLast = some ⟨o.x, Lean.Json.null, o.recPs.map RecDeploy.DeployedR.opensOf⟩ := by
  obtain ⟨-, hts⟩ := setup_tables o.setup
  have hsz := (sessionS_reps out64_sha512 (rfl : o.resp 0 = o.recC₁.bytes) hts.symm o.recTabsOk o.ran [] []).1
  have hps : o.recPs.size = 2 * o.sts.size := by
    unfold Run.recPs
    rw [filePs_size, hsz]
  obtain ⟨st, hst, -⟩ := o.tableOk
  have h1 : o.sts.size = 1 := by rw [hst]; rfl
  have hl : o.recPs.toList.length < 2 ^ 64 := by
    rw [Array.length_toList, hps, h1]
    norm_num
  have hp : ∀ p ∈ o.recPs.toList, RecDeploy.DeployedR.OpensFit p := by
    intro p hp
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hp)
    have hk2 : k / 2 < o.sts.size := by omega
    refine opensFit_of_shaped ?_ (recShaped out64_sha512 o k hk)
    rw [tabsOf_get! hk2]
    exact (o.recTabsOk (k / 2) hk2).1.scheme
  rw [Run.recLast, RecDeploy.DeployedR.rdR_encLast o.ready_parts.2.1 hl hp, toArray_map_toList]

open Rows in
theorem honest_len (o : Run sha512 Readable.relation) {cs : List Coin} (hcs : o.CoinsAt cs)
    (hk : cs.length = (V false).k o.x) (last : Claims.Msg) : (o.recMsgs cs last).length = cs.length := by
  have h64 := out64_sha512
  have hx := o.setup
  have htab := o.recTabsOk
  have hresp : ∀ i < cs.length, o.resp i = (cs.getD i default).bytes := by
    intro i hi
    simp only [Case.resp, hcs.1 i hi]
  have hc : o.resp 0 = o.recC₁.bytes := rfl
  have hfc := filesFinalC (resp := o.resp) h64 hc (setup_tables hx).2.symm htab o.ran hresp hcs.2
  obtain ⟨-, hts⟩ := setup_tables hx
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts.symm htab o.ran (capRows o.recCaps) cs.tail.dropLast
  have hcaps : ∀ i < o.recCaps.size, o.recCaps[i]!.size = RecDeploy.level0Nodes o.sts[i]! := by
    intro i hi
    unfold Run.recCaps at hi ⊢
    rw [fileCaps_size] at hi
    have h := (hall (2 * i) (by omega)).2.2.2.2.1
    rw [show 2 * i / 2 = i by omega] at h
    simp only [fileCaps, JsonRT.range_map_getElem! _ hi]
    exact h
  have hrep : ∀ k < o.recPs.size, SchedOf (o.recCtx cs k) ∧ HonestRep (o.recCtx cs k) o.recPs[k]! ∧
      o.recPs[k]!.cap = o.recCaps[k / 2]! := by
    intro k hk
    unfold Run.recPs at hk
    rw [filePs_size] at hk
    have hf := hfc k hk
    rw [hsz] at hk
    obtain ⟨-, h2, h3, h4, -, -⟩ := hall k hk
    refine ⟨h2, h3 hf, ?_⟩
    unfold Run.recCaps
    simp only [fileCaps, JsonRT.range_map_getElem! _ (show k / 2 < o.sts.size by omega)]
    exact h4
  obtain ⟨hl, -⟩ := KS.honestRounds_nAt (Ts := o.recTs) (c₁ := o.recC₁) (cs := cs.tail.dropLast)
    (caps := o.recCaps) (ps := o.recPs) hx (fileCaps_size ..) hcaps
    (by unfold Run.recPs; rw [filePs_size, hsz]) hrep
  have hlen : (o.recMsgs cs last).length = (1 + (repLens o.spec o.sts).sum) + 1 := by
    rw [← hl]
    unfold Run.recMsgs Run.recCtx
    rw [KS.honestMsgs_rounds]
    simp
  rw [hlen, repLens_sum (fun i hi => (htab i hi).1), hk]
  simp only [V, hx]
  omega

theorem honest_sampleTied (o : Run sha512 Readable.relation) :
    sampleEq (RecDeploy.sampleOfU o.spec.unitSample) o.s.read.draw = true := by
  obtain ⟨-, -, hDO, hU⟩ := JsonRT.stmtOk (H := sha512) (rel := Readable.relation) o.toCase
  have hne : ∀ d, o.s.read.draw = some d → d ≠ Lean.Json.null := by
    intro d hd
    unfold JsonRT.SampleOk at hDO
    rw [hU, hd] at hDO
    exact hDO.1
  rw [hU]
  cases hdr : o.s.read.draw with
  | none => rfl
  | some d =>
    have hn := hne d hdr
    obtain ⟨w, -, hw⟩ := JsonRT.insOf_false (setupOf_some o.setup).1
    obtain ⟨dr, -, hdr'⟩ := (FlockComplete.JsonRT.Raw.ins_read hw).2.2
    rw [hdr] at hdr'
    obtain ⟨d₀, -, rfl⟩ := Option.map_eq_some_iff.1 hdr'.symm
    have hnn := norm_norm d₀
    have e : RecDeploy.sampleOfU (some (Flock.norm d₀)) = some (Flock.norm d₀) := by
      cases h : Flock.norm d₀ with
      | null => exact absurd h hn
      | _ =>
        rw [h] at hnn
        exact congrArg some hnn
    rw [e]
    exact sampleEq_refl _

theorem honest_sampleJ (o : Run sha512 Readable.relation)
    (cs : List Coin) : sampleEq (sampleOfJ (o.recDec cs).f.j) o.s.read.draw = true := by
  show sampleEq (sampleOfJ (recJ o.spec o.sts o.recLink o.recC₁ (o.recFrames cs) cs.tail.dropLast o.recPs))
    o.s.read.draw = true
  rw [recJ, RecDeploy.sampleOfJ_recordObj]
  exact RecDeploy.sampleEq_map_norm
    (fun e he => (FlockDeploy.insOf_sample_norm (FlockDeploy.setupOf_some o.setup).1 he).1) (honest_sampleTied o)

theorem honest_decR_deployed (o : Run sha512 Readable.relation) :
    decR RecDeploy.DeployedR.rdR o.x (o.recMsgs o.recCoins o.recLast) o.recCoins = some (o.recDec o.recCoins) :=
  honest_decR out64_sha512 o o.coinsAt_recCoins (honest_rdR o) rfl rfl (recShaped out64_sha512 o)

theorem honest_actR_deployed (o : Run sha512 Readable.relation) :
    actR RecDeploy.DeployedR.rdR o.x (o.recMsgs o.recCoins o.recLast) o.recCoins =
      .decide (o.recDec o.recCoins).s (o.recDec o.recCoins).spec (o.recDec o.recCoins).sts
        (o.recDec o.recCoins).link (o.recDec o.recCoins).c₁ (o.recDec o.recCoins).mids (o.recDec o.recCoins).f :=
  honest_actR out64_sha512 o o.coinsAt_recCoins (honest_rdR o) (RecDeploy.DeployedR.encLast_head o.x o.recPs.toList)
    rfl rfl

theorem honest_readOkR (o : Run sha512 Readable.relation) :
    ReadOkR RecDeploy.DeployedR.rdR o.x (o.recMsgs o.recCoins o.recLast) o.recCoins := by
  unfold ReadOkR
  rw [honest_actR_deployed o]
  exact RecDeploy.RestRead.decLive RecDeploy.DeployedR.rdR o.x _ _ _
    (honest_len o o.coinsAt_recCoins o.recCoins_length _) (honest_decR_deployed o) o.scope_ok

theorem honest_holdsR (o : Run sha512 Readable.relation) :
    ∀ q : VerdictR, holdsR sha512 RecDeploy.DeployedR.rdR q o.x (o.recMsgs o.recCoins o.recLast) o.recCoins := by
  have hcs := o.coinsAt_recCoins
  intro q
  unfold holdsR
  rw [honest_actR_deployed o]
  cases q
  · exact honestSetup o _ hcs
  · exact honestShaped out64_sha512 o _ hcs
  · exact honestNames o _ hcs
  · exact honestVerdict o _ hcs
  · intro spec' sts' hs'
    change setupWith (Claims.hOf sha512) o.s.read = _ at hs'
    rw [o.recSetup] at hs'
    cases hs'
    obtain ⟨lv, hL, hC⟩ := RecDeploy.custodyRecJ RecDeploy.DeployedR.rdR o.x _ _ _
      (honest_len o hcs o.recCoins_length _) (honest_decR_deployed o) o.scope_ok
    exact RecDeploy.checksAtJ_ok_of (f := ⟨o.x, (o.recDec o.recCoins).f.j, #[]⟩) rfl (honest_sampleJ o _) hL hC
      o.scope_ok o.ready_parts.1

end FlockDeploy

namespace RecDeploy

open Flock
open FlockComplete (Run)

theorem honest_accepts_nextR (o : Run FlockDeploy.sha512 Readable.relation) (ys : ℕ → ℕ → Salt) :
    FlockDeploy.ReadOkR DeployedR.rdR o.x (o.recMsgs o.recCoins o.recLast) o.recCoins ∧
      (∀ q : FlockDeploy.VerdictR,
        FlockDeploy.holdsR FlockDeploy.sha512 DeployedR.rdR q o.x (o.recMsgs o.recCoins o.recLast) o.recCoins) ∧
      ((FlockDeploy.nextR DeployedR.rdR o.x (o.recMsgs o.recCoins o.recLast) o.recCoins).eval
        FlockDeploy.sha512).1 = .acc ∧
      (Claims.run FlockDeploy.sha512 (Claims.replay (o.recAsR ys)) DeployedR.bdR.IVStar o.x ()
        fun i => o.recCoins.getD i.val default).out = .acc ∧
      ∀ (D : Rows.DecAt DeployedR.rdR o.x (o.recMsgs o.recCoins o.recLast) o.recCoins)
        (cls : FlockSoundness.Discharge.Recursion.HClass) (acc : ℕ → Array F128) {i : ℕ}, D.RowG cls acc i →
        i < D.spec.tables.size → ∀ (r : Fin 2) v info kappa,
        Flock.RecAlgebra.verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
          (D.rep cls acc i r).comb = .ok (v, info, kappa) →
        ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
          Rows.levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
            (info.lg.conss[ℓ]!).pos = .ok () := by
  have hcs := o.coinsAt_recCoins
  have hresp : ∀ i < o.recCoins.length, o.resp i = (o.recCoins.getD i default).bytes := by
    intro i hi
    simp only [FlockComplete.Case.resp, hcs.1 i hi]
  have hR := FlockDeploy.honest_readOkR o
  have hH := FlockDeploy.honest_holdsR o
  have hacc := FlockDeploy.nextR_sound _ _ _ _ _ hR hH
  refine ⟨hR, hH, hacc, ?_, ?_⟩
  · exact DeployedR.ivstar_honest_filesR Readable.relation DeployedR.rdR
      (FlockDeploy.deployPlainTrimR Readable.relation DeployedR.rdR).cols DeployedR.wER rowInDeployed DeployedR.wR
      (resp := o.resp) (Ts := o.recTs) (ys := ys) (last := o.recLast) o.setup o.recTabsOk hcs.head hresp
      o.recCoins_length o.ran hacc
  · intro D cls acc i hG hti r
    have hDd : D.d = o.recDec o.recCoins := Option.some.inj (D.dec.symm.trans (FlockDeploy.honest_decR_deployed o))
    have hsD : FlockDeploy.setupWith (Claims.hOf FlockDeploy.sha512) o.s.read = .ok (D.spec, D.sts) := by
      have h := D.setup
      rw [hDd] at h
      exact h
    rw [o.recSetup] at hsD
    obtain ⟨hspec, -⟩ := Prod.mk.inj (Except.ok.inj hsD)
    have hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup := by
      rw [← hspec]
      exact FlockDeploy.nodup_names
        (FlockDeploy.namesDistinct_of_setup (FlockDeploy.setupOf_some o.setup).2 o.recPublicInputs)
    exact Rows.DecAt.hlev_honest_deployed cls acc (Ts := o.recTs) (last := o.recLast) o.setup hcs.head hresp hcs.2
      o.recTabsOk o.ran (FlockDeploy.honest_rdR o) rfl rfl D hG hti r hnd

end RecDeploy
