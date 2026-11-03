import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.Recursion.LastSharedR
import Proofs.Flock.Claims.Recursion.NextRFits

/-!
# The plain simulator's round counts and rep schedules at a parsed record

`decodeWithK_n` and `liveOf_nK` (`PlainStep`) at the record's parsed value (`Record.decodeJWithK`, `verifyJWithMK`,
`liveOfJ`): every round the decoder reads, and every round of an accepted record's live streams, has `1` to `65536`
coins. And every rep of an accepted record is set up (`SchedOf (repCtx …)`, `schedOf_repCtx_of_checksAtJ`), from the
verdict and the link points the checks pin to the first coin.
-/

namespace FlockDeploy.KS

open Flock Transcript
open FlockSoundness.Refine (ebind_ok)
open Lean (Json)

/-- **The verifier's decoder reads only rounds of `1` to `65536` coins** (S12) at a parsed record, at any `keep`. -/
theorem decodeJWithK_n (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : RunRecord} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∀ x ∈ sess.streams.toList, ∀ rd ∈ x.rounds.toList, 1 ≤ rd.n ∧ rd.n ≤ 65536 := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine forall_of_forall₂ (forall₂_of_array_mapM
      (P := fun _ (b : StreamRec) => ∀ rd ∈ b.rounds.toList, 1 ≤ rd.n ∧ rd.n ≤ 65536) (fun s x hx => ?_)
      (by assumption))
    obtain ⟨name, -, hx⟩ := ebind_ok hx
    obtain ⟨rs, -, hx⟩ := ebind_ok hx
    obtain ⟨rounds, hrounds, hx⟩ := ebind_ok hx
    repeat' obtain ⟨_, _, hx⟩ := ebind_ok hx
    cases hx
    refine forall_of_forall₂ (forall₂_of_mapIdxM
      (P := fun _ (rd : Round) => 1 ≤ rd.n ∧ rd.n ≤ 65536) (fun k r rd hr => ?_) hrounds)
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨n, -, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, hc, hr⟩ := ebind_ok hr
    have hb := check_true hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
    repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (split at hr))
    all_goals
      cases hr
      exact ⟨hb.1.1.1, hb.1.1.2⟩

/-- **An accepted parsed record's live rounds have `1` to `65536` coins**: `liveOfJ`'s rounds are the decoder's, by
coin count (`liveOfJ_decodeJWithK`). At any `keep` (`verifyJWithMK`). -/
theorem liveOfJ_nK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {j : Json} {proofs : Array ByteArray} {lv : Live} (hv : verifyJWithMK keep hs spec tabs j proofs = .ok ())
    (hl : FlockDeploy.liveOfJ spec j = .ok lv) (k : ℕ) (hk : k < lv.streams.size) :
    ∀ rd ∈ lv.streams[k].toList, 1 ≤ rd.n ∧ rd.n ≤ 65536 := by
  obtain ⟨sess, -, hdec, -⟩ := verifyJWithMK_tables keep hs hv
  obtain ⟨-, -, hsz, hall⟩ := liveOfJ_decodeJWithK keep hs hl hdec
  obtain ⟨stream, hfind, hfa⟩ := hall k (hsz ▸ hk) hk
  have hmem : stream ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hfind)
  have hn := map_n_of_forall₂ hfa
  intro rd hrd
  have : rd.n ∈ stream.rounds.toList.map Round.n := hn ▸ List.mem_map_of_mem hrd
  obtain ⟨rd', hrd', e⟩ := List.mem_map.1 this
  rw [← e]
  exact decodeJWithK_n keep hs hdec stream hmem rd' hrd'

/-! ## The reps' schedules -/

/-- **An accepted parsed record's reps are set up** (`SchedOf`): where the plain verdict accepts at the tables and
`liveOfJ` reads link points that are `c₁`'s, every rep `k` has its table's `fast100` schedule and extra claims at
`c₁`'s points (`rep_plainJK`, `verifyRepWith_readsN`), so `schedOf_repCtx` holds, at any `r₀` and `cs` (`SchedOf`
reads neither). At any `keep`. -/
theorem schedOf_repCtx_of_verify (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec}
    {sts : Array Stmt} {j : Json} {proofs : Array ByteArray} {lv : Live} {c₁ : Coin}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv)
    (hpts : lv.points = pointsOf spec c₁) (r₀ : List ByteArray) (cs : List Coin) {k : ℕ} (hk : k < 2 * sts.size) :
    SchedOf (repCtx spec sts c₁ r₀ cs k) := by
  obtain ⟨hsz, hall⟩ := rep_plainJK keep hs hv hl
  obtain ⟨sess, sch, -, c, hsp, -, hf, hrep, -, -⟩ := hall k (by rw [hsz]; exact hk)
  have hk2 : k / 2 < sts.size := by omega
  rw [getElem!_pos sts (k / 2) hk2] at hf hrep
  obtain ⟨-, extras, -, he, -⟩ := verifyRepWith_readsN (st := Setup.ofCircuit sts[k / 2]) hs rfl hf hrep
  rw [hsp, hpts] at he
  exact schedOf_repCtx hk2 (by rw [hf]; rfl) he

/-- `schedOf_repCtx_of_verify` at a decision's checks: `checksAtJ` reads the value's live session, whose link points
are the first coin's (`checksAtJ_ok`, `custody_ok`). -/
theorem schedOf_repCtx_of_checksAtJ (keep : Bool → Bool) (hs : Hashes Flock.V) {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {x : Claims.Msg} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalJ} {v : Json} {proofs : Array ByteArray}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) v proofs = .ok ())
    (hc : checksAtJ s spec x link c₁ mids f v = .ok ()) (r₀ : List ByteArray) (cs : List Coin) {k : ℕ}
    (hk : k < 2 * sts.size) : SchedOf (repCtx spec sts c₁ r₀ cs k) := by
  obtain ⟨-, lv, hl, hcus⟩ := Claims.Bridge.Custody.checksAtJ_ok hc
  exact schedOf_repCtx_of_verify keep hs hv hl (Claims.Bridge.Custody.custody_ok hcus).2.2.1 r₀ cs hk

end FlockDeploy.KS
