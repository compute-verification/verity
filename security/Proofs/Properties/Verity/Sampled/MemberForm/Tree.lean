module

public import Proofs.Properties.Verity.Sampled.MemberForm.Order
public import Proofs.Properties.Verity.Sampled.MemberForm.Named
public import Proofs.Properties.Verity.Sampled.MovesReach

/-! A member's session's record, held to its own tree of a log of several. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger
open Properties.Challenger.CanonRead (sorted_getD)

variable {t : LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}

theorem tree_coinTree (h : sessionTreeReason #[t.unclosed] tables record = none) :
    recordTree record = t.root ∧ jeq (getD record "coin_tree") t.record = true := by
  obtain ⟨t', ht', hroot, hj⟩ := Trees.coinTree_held h
  obtain rfl := mem_one ht'
  exact ⟨hroot, hj⟩

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Verity.Release} {trees : Array LoggedTree}

theorem tree_open (hT : coinTreesOf lines = .ok trees) (ht : t ∈ trees)
    (h : sessionTreeReason #[t.unclosed] tables record = none) {stream : String} {k : ℕ} {cs : ByteArray}
    (hr : recordRound record stream k = some cs) :
    ∃ l ∈ lines, ∃ n after, Trees.OpenLine l t.root ⟨stream, k, n, after, sha512Hex cs⟩ := by
  obtain ⟨-, o, ho, hc⟩ := tree_round h hr
  have hp := Array.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  obtain ⟨l, hl, hol⟩ := Trees.opened_in hT t ht o (Array.mem_of_find?_eq_some ho)
  refine ⟨l, hl, o.n, o.after, ?_⟩
  rw [← hc, ← hp.1, ← hp.2]
  exact hol

theorem tree_keyAt (hT : coinTreesOf lines = .ok trees) (ht : t ∈ trees)
    (h : sessionTreeReason #[t.unclosed] tables record = none) {name : String}
    (hpos : 0 < recordRounds record name) : ∃ i si, KeyAt (heldTrees trees) i si t.root name := by
  obtain ⟨cs, hcs⟩ := Option.isSome_iff_exists.1 (recordRound_isSome h name 0 hpos)
  obtain ⟨l, hl, _, _, htl, d, hev, hTE, hroot, hs, -⟩ := tree_open hT ht h hcs
  exact treeFold_keyAt (records := []) lines [] _ (coinTreesOf_fold [] hT) l hl htl d hev hTE hroot hs

theorem tree_rounds_le (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    (ht : t ∈ trees) (h : sessionTreeReason #[t.unclosed] tables record = none) {name : String} {i si : ℕ}
    (hk : KeyAt (heldTrees trees) i si t.root name) : recordRounds record name ≤ openedOf log i si := by
  rcases Nat.eq_zero_or_pos (recordRounds record name) with h0 | hpos
  · rw [h0]
    exact Nat.zero_le _
  obtain ⟨cs, hcs⟩ :=
    Option.isSome_iff_exists.1 (recordRound_isSome h name _ (Nat.sub_lt hpos Nat.one_pos))
  obtain ⟨-, o, ho, -⟩ := tree_round h hcs
  have := opening_lt_openedOf hT hR ht ho hk
  omega

theorem helloSpec_of_tree (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    {Lg : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) log Lg) (ht : t ∈ trees)
    (h : sessionTreeReason #[t.unclosed] tables record = none) (hrs : sortedKeys record = true)
    {sp : Flock.CoinTree.Spec} {nonce key : String}
    (hC : Flock.CoinTree.checkRecord sp nonce (getD record "coin_tree") = .ok key) {i si : ℕ} {name : String}
    (hk : KeyAt (heldTrees trees) i si t.root name) : (helloSpecs Lg)[i]? = some sp := by
  obtain ⟨-, hj⟩ := tree_coinTree h
  have hcs := sorted_getD hrs "coin_tree"
  have hts := trees_sorted hT t ht
  obtain ⟨s, hs1, hs2⟩ := checkRecord_spec hC
  have hsp : Lines.SpecValue t.spec sp :=
    specValue_of_equiv hts (Device.printed_spec sp).1
      (Eq.trans (Eq.symm (spec_equiv hcs hts hj))
        (Properties.Challenger.CanonLine.canon_inj (sorted_getD hcs "spec") (Device.printed_spec sp).1 hs1 hs2))
      (specValue_toJson sp)
  obtain ⟨-, sj, hg, -⟩ := hk
  have hg' : (trees.toList.map fun t => (t.root, t.spec))[i]? = some (t.root, sj) := hg
  rw [List.getElem?_map, Array.getElem?_toList] at hg'
  cases hti : trees[i]? with
  | none => simp [hti] at hg'
  | some t' =>
    rw [hti] at hg'
    obtain ⟨hr', hs'⟩ := Prod.mk.inj (Option.some.inj hg')
    have e : t' = t := List.inj_on_of_nodup_map (Lines.trees_log hT).nodup
      (Array.mem_toList_iff.2 (Array.mem_iff_getElem?.2 ⟨i, hti⟩)) (Array.mem_toList_iff.2 ht) hr'
    obtain ⟨sp', hsp', hv⟩ := forall₂_getElem? (readLog_specs hR (coinTreesOf_fold recs hT)) hg
    have hv' : specOf sj = some sp' := hv
    rw [← hs', e, Lines.specOf_value hsp] at hv'
    rw [← helloSpecs_congr hl, hsp']
    exact hv'.symm

theorem helloSpec_of_text {spec : Flock.SessionSpec} {text : String} {sess : Flock.RunRecord}
    {sp : Flock.CoinTree.Spec} (hd : Flock.Record.decode spec text = .ok sess) (hc : spec.coinTree = some sp)
    (hJ : Flock.JsonParse.parse text = .ok record) (hT : coinTreesOf lines = .ok trees)
    (hR : readLog label after recs ξ lines = some log) {Lg : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) log Lg) (ht : t ∈ trees)
    (h : sessionTreeReason #[t.unclosed] tables record = none) {i si : ℕ} {name : String}
    (hk : KeyAt (heldTrees trees) i si t.root name) : (helloSpecs Lg)[i]? = some sp := by
  obtain ⟨j, c, nonce, key, hp, hcj, hC⟩ := decode_coinTree hd hc
  have hjr : j = record := Except.ok.inj (hp.symm.trans hJ)
  rw [hjr] at hcj
  rw [← getD_of_fld hcj] at hC
  exact helloSpec_of_tree hT hR hl ht h
    (Properties.Challenger.CanonRead.sortedKeys_of_wf record (FlockComplete.JsonRT.parse_wf text record hJ)) hC hk

end Verity.Sampled.MemberForm
