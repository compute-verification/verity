module

public import Proofs.Properties.Verity.SampledTwo.PrefixFixed
public import Proofs.Properties.Verity.SampledTwo.SessionPrefix
public import Proofs.Properties.Verity.SampledTwo.SessionExtra

/-! A release put on a session's round is of a round the session's record gives; how many rounds a record
gives a stream is not established here. -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)

section Recorded

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Release} {trees : Array Spot.Challenger.LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)} {tree : ℕ → ℕ}

theorem MembersHeld.recorded (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log)
    (hM : Spot.Challenger.membersTreeReason trees lines records = none) {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String}
    (hs : ∀ t, off (t + 1) = off t + K t) {pre : List Release} {r : Release} {post : List Release}
    (he : log = pre ++ r :: post) {t j : ℕ} (hj : j < K t)
    (hloc : sessLoc off K M tab pre r.frame = some (off t + j)) :
    ∃ x, records[t]? = some x ∧ (slotStream (tab t) (repsOf (K t)) j).2 <
      Spot.Challenger.recordRounds x.2.2.1 (slotStream (tab t) (repsOf (K t)) j).1 := by
  obtain ⟨t', j', s, c, ht, hf, hj', hn, hname, hk⟩ := H.key hT hR he hloc
  obtain ⟨rfl, rfl⟩ := off_sep hs hj hj' hn
  obtain ⟨x, hx⟩ : ∃ x, records[t]? = some x := ⟨_, Array.getElem?_eq_getElem ht⟩
  refine ⟨x, hx, ?_⟩
  obtain ⟨l, ρ, name, k, hl, hro, hkey, hnm, hkk⟩ := Verity.Sampled.readLog_round hT hR he hf
  have en : name = (slotStream (tab t) (repsOf (K t)) j).1 := Option.some.inj (hnm.symm.trans hname)
  have ek : k = (slotStream (tab t) (repsOf (K t)) j).2 := hkk.trans hk
  obtain ⟨-, sp, hsp, -⟩ := hkey
  obtain ⟨u, hui, hur, -⟩ := heldTrees_getElem? hsp
  have hum : u ∈ trees := Array.mem_iff_getElem?.2 ⟨_, hui⟩
  have hnd := (Verity.Sampled.Lines.trees_log hT).nodup
  have hmem : (name, k) ∈ lines.filterMap (Verity.Sampled.lineOpening u.root) :=
    List.mem_filterMap.2 ⟨l, hl, by rw [hur]; exact Verity.Sampled.OpensRound.lineOpening hro.1⟩
  rw [← (Verity.Sampled.coinTreesOf_listed hT).held (tree t) u hui (Verity.Sampled.findIdx?_root hnd hui)] at hmem
  obtain ⟨o, ho, hoe⟩ := List.mem_map.1 hmem
  obtain ⟨ρx, hρx, hux⟩ := H.roots t x hx
  have eρ : ρ = ρx := Verity.Sampled.MemberForm.findIdx_fst (H.idx t ρx hux) hsp
  have hroot : Spot.getD (Spot.getD x.2.2.1 "coin_tree") "root" = Json.str u.root := by rw [hρx, hur, eρ]
  have hne : (o.stream, o.round) ≠ (Spot.Challenger.TREE_SESSION, 1) := by
    intro hEq
    obtain ⟨hq1, hq2⟩ := Prod.mk.inj (hoe.symm.trans hEq)
    exact slotStream_ne_closing (tab t) (repsOf (K t)) j (Prod.ext (en.symm.trans hq1) (ek.symm.trans hq2))
  have hlt := membersTree_recorded_at hM hnd hx hum hroot (Array.mem_def.2 ho) hne
  rw [(Prod.mk.inj hoe).1, (Prod.mk.inj hoe).2, en, ek] at hlt
  exact hlt

end Recorded

def LocatedRecorded (A : Live₂) (C : CaseOf) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ a r b, l = a ++ r :: b →
    ∀ t k, k < SpotFlock.k₀ A.h t → locOf A tab a r.frame = some (SpotFlock.off A.h t + k) →
      ∃ recs rec, sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
        (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).2 <
          Spot.Challenger.recordRounds rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).1

theorem located_recorded (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) : LocatedRecorded A C tab := by
  intro o hacc l hl a r b hsplit t k hk hloc
  obtain ⟨label, after, ξ, lines, trees, records, hT, hR, hMR, hS⟩ :=
    membersAccepted (membersNamed C hsz) o hacc l hl
  obtain ⟨tree, H⟩ := membersHeld_of hT hR hMR
  unfold locOf at hloc
  obtain ⟨x, hx, hlt⟩ := H.recorded hT hR hMR (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h)
    (SpotFlock.off_succ A.h) hsplit hk hloc
  refine ⟨_, x.2.2.1, hS, ?_, hlt⟩
  rw [List.getElem?_map, Array.getElem?_toList, hx]
  rfl

def LogRecorded (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (L : CoinMap.Locate) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ t k, k < SpotFlock.k₀ A.h t →
    CoinMap.locAt L a rl.frame rl.answers = some (SpotFlock.off A.h t + k) →
      ∃ recs rec, sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
        (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).2 <
          Spot.Challenger.recordRounds rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).1

theorem logRecorded_of {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {L : CoinMap.Locate} {tab : ℕ → String}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hsz : HeldSize) (hI : InteriorLogged C Good) :
    LogRecorded A C Good L tab := by
  intro o hacc hgood a rl b hlog t k hk hloc
  obtain ⟨l, hl⟩ := interiorReads C hsz o hacc
  obtain ⟨q, l', hlog', hq, hF⟩ := hI o hacc hgood l hl
  have h1 : locOf A tab a rl.frame = some (SpotFlock.off A.h t + k) :=
    (hL a rl.frame rl.answers _ (groupOf_off A t k)).1 hloc
  obtain ⟨i, hi⟩ := openTree_of_locOf h1
  rw [hlog'] at hlog
  obtain ⟨a', rfl, hl''⟩ := split_past hlog (fun x hx => (hq x hx).1) hi
  subst hl''
  obtain ⟨la, x, lrest, rfl, hla, hx, -⟩ := forall₂_split_right (frames_of_read hF)
  have hxf : x.frame = rl.frame := hx
  have e1 : locOf A tab la x.frame = some (SpotFlock.off A.h t + k) := by
    rw [locOf_logged A tab hq hla hxf]
    exact h1
  exact located_recorded A C tab hsz o hacc _ hl la x lrest rfl t k hk e1

theorem logRecorded_run (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hsz : HeldSize) (hI : InteriorLogged C Good) : LogRecorded A C Good (runLocate A Ls tab) tab :=
  logRecorded_of (locates_run A Ls tab) hsz hI

end Verity.SampledTwo.SessionCoins
