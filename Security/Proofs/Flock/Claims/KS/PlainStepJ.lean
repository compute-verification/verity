import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.Recursion.LastSharedR

/-!
# The plain simulator's round counts at a parsed record

`decodeWithK_n` and `liveOf_nK` (`PlainStep`) at the record's parsed value (`Record.decodeJWithK`, `verifyJWithMK`,
`liveOfJ`): every round the decoder reads, and every round of an accepted record's live streams, has `1` to `65536`
coins.
-/

namespace FlockDeploy.KS

open Flock Transcript
open FlockSoundness.Refine (ebind_ok)
open Lean (Json)

/-- **The verifier's decoder reads only rounds of `1` to `65536` coins** (S12) at a parsed record, at any `keep`. -/
theorem decodeJWithK_n (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : Session} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
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

end FlockDeploy.KS
