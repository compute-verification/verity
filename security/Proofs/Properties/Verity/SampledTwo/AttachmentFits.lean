module

public import Proofs.Properties.Verity.SampledTwo.SessionAgrees
public import Proofs.Properties.Verity.Sampled.RoundsRead

/-!
# Each opening the check read is of at most one coin's elements

The coin map reads a session's round as one coin of at most 65536 elements of GF(2^128). The device's game alone does
not bound an opening so: it samples a tree whose spec's coins are at most `Verity.maxTreeCoins` (`Verity.TreeSpecOk`)
and opens a round for as many coins as the tree's block. The check does: `Spot.Challenger.coinTreesOf` accepts a
tree's line only where its spec's block is at most `Spot.Challenger.MAX_BLOCK` (`Spot.Challenger.isTreeSpec`), and a
line opening a round only where its `n` is at most its tree's block.

* `coinTreesOf_fits`: where `coinTreesOf` accepts the release log, each of its lines that opens a round names an `n` of
  at most `MAX_BLOCK` (`OpensBounded`, kept line by line).
* `readFits`: on a play the two-stage form accepts, each opening in the log the check read is of at most 65536 coins
  (`ReadFits`): the opening is a line of the log `coinTreesOf` accepted, with that line's `n`
  (`Verity.Sampled.readLog_round`).
* `sessionsFit_of`: on a play the form accepts whose log the check read is the device's, the releases the
  session-round locator puts on a session's round carry at most 65536 elements (`SessionsFit`): it puts at most one
  release of the log the check read on each round (`located_one_accepted`), and that one is an opening `readFits`
  bounds. The check bounds an opening, not the device's game.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Spot Spot.Challenger

/-! ## What `coinTreesOf` holds of an opening's count -/

/-- **What `coinTreesOf` holds of the trees `trees` it read from the lines `pre`, on blocks**: each tree's spec is one
`isTreeSpec` accepts, and each line of `pre` opening a round names at most `MAX_BLOCK` coins. -/
structure OpensBounded (pre : List String) (trees : Array LoggedTree) : Prop where
  spec : ∀ t ∈ trees, isTreeSpec t.spec = true
  fits : ∀ l ∈ pre, treeLine l = true → ∀ d, event l = .ok d → jsonStrIs (getD d "event") TREE_EVENT = false →
    ∀ n, nat? (getD d "n") = some n → n ≤ MAX_BLOCK

/-- A spec `isTreeSpec` accepts names a block of at most `MAX_BLOCK` coins. -/
theorem block_le {spec : Json} (h : isTreeSpec spec = true) : (nat? (getD spec "block")).getD 0 ≤ MAX_BLOCK := by
  unfold isTreeSpec at h
  simp only [Bool.and_eq_true] at h
  have hb := h.1.1.2
  cases hn : nat? (getD spec "block") with
  | none =>
    rw [hn] at hb
    cases hb
  | some b =>
    rw [hn] at hb
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
    exact hb.2

/-- **Each line `coinTreesOf` reads keeps `OpensBounded`.** -/
theorem coinTreesOf_fits {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    OpensBounded log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Verity.Sampled.Trees.forIn_inv_run (I := OpensBounded) ?_
    (by exact ⟨fun _ h => absurd h (by simp), fun _ h => absurd h (by simp)⟩) h
  intro pre line ts r hI hr
  have mem : ∀ {l : String}, l ∈ pre ++ [line] → l ∈ pre ∨ l = line := fun h => by simpa using h
  split at hr
  case isFalse htl =>
    cases hr
    refine ⟨ts, rfl, hI.spec, ?_⟩
    intro l hl ht d hev hT n hn
    rcases mem hl with hl | rfl
    · exact hI.fits l hl ht d hev hT n hn
    · exact absurd ht htl
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, Verity.Sampled.Trees.ok_bind] at hr
    split at hr
    · rename_i htree
      split at hr
      case isFalse => cases hr
      rename_i hcond
      split at hr
      case isTrue => cases hr
      cases hr
      simp only [Bool.and_eq_true] at hcond
      refine ⟨_, rfl, ?_, ?_⟩
      · intro t ht
        rcases Array.mem_push.1 ht with ht | rfl
        · exact hI.spec t ht
        · exact hcond.2
      · intro l hl ht d' hev' hT n hn
        rcases mem hl with hl | rfl
        · exact hI.fits l hl ht d' hev' hT n hn
        · obtain rfl : d' = d := Except.ok.inj (hev'.symm.trans hev)
          rw [htree] at hT
          cases hT
    · split at hr
      case isFalse => cases hr
      split at hr
      case h_2 => cases hr
      split at hr
      case h_2 => cases hr
      rename_i i hidx
      split at hr
      case h_2 => cases hr
      rename_i t hget
      split at hr
      case h_2 => cases hr
      split at hr
      case h_2 => cases hr
      rename_i k n hkn
      obtain ⟨hi, -, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
      have hti : ts[i] = t := by
        rw [Array.getElem?_eq_getElem hi] at hget
        exact Option.some.inj hget
      have htmem : t ∈ ts := hti ▸ Array.getElem_mem hi
      simp only [Prod.mk.injEq] at hkn
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          have hn : n ≤ (nat? (getD t.spec "block")).getD 0 := of_decide_eq_true hcond.1.1.2
          cases hr
          refine ⟨_, rfl, ?_, ?_⟩
          · intro t' ht'
            rcases Verity.Sampled.Trees.mem_modify ht' with ht' | ⟨a, ha, rfl⟩
            · exact hI.spec t' ht'
            · obtain rfl : a = t := Option.some.inj (ha.symm.trans hget)
              exact hI.spec a htmem
          · intro l hl ht d' hev' hT n' hn'
            rcases mem hl with hl | rfl
            · exact hI.fits l hl ht d' hev' hT n' hn'
            · obtain rfl : d' = d := Except.ok.inj (hev'.symm.trans hev)
              obtain rfl : n = n' := Option.some.inj (hkn.2.symm.trans hn')
              exact Nat.le_trans hn (block_le (hI.spec t htmem))
        · cases hr

/-! ## The log the check read -/

/-- **Every opening the check read is of at most one coin's elements**: on a play the form accepts, an opening of `n`
coins in the log the check read has `n ≤ 65536`. It is stated of the read, at the form's acceptance, since the device's
game alone bounds an opening only by its tree's block, and a tree's block only by `Verity.maxTreeCoins`. -/
def ReadFits (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l →
    ∀ r ∈ l, ∀ i s n, r.frame = .opening i s n → n ≤ 65536

/-- **`ReadFits`, from the members' check on the log the check read**: each opening the read shows is a line of the
log `coinTreesOf` accepted, with that line's `n` (`Verity.Sampled.readLog_round`), which is at most `MAX_BLOCK`
(`coinTreesOf_fits`). -/
theorem readFits_of {C : CaseOf} (h : MembersNamed C) : ReadFits C := by
  intro o hacc l hl r hr i s n hfr
  obtain ⟨-, -, label, after, ξ, lines, trees, records, -, -, hT, hR, -, -⟩ := h o hacc l hl
  obtain ⟨pre, post, he⟩ := List.append_of_mem hr
  obtain ⟨l₀, ρ, name, k, hl₀, ⟨⟨htl, d, hev, hTE, -⟩, -, d', hev', hn, -⟩, -⟩ :=
    Verity.Sampled.readLog_round hT hR he hfr
  obtain rfl : d = d' := Except.ok.inj (hev.symm.trans hev')
  have hle := (coinTreesOf_fits hT).fits l₀ hl₀ htl d hev hTE n (hn i s n hfr)
  unfold MAX_BLOCK at hle
  exact Nat.le_trans hle (by norm_num)

/-- **`ReadFits` at the form's acceptance** (`membersNamed`). -/
theorem readFits (C : CaseOf) (hsz : HeldSize) : ReadFits C :=
  readFits_of (membersNamed C hsz)

/-! ## The releases on a session's round -/

/-- **The releases on each session's round fit in one coin**: on an accepted play whose log is the device's, the
releases the session-round locator puts on a session's round carry at most 65536 elements. -/
def SessionsFit (A : Live₂) (tab : ℕ → String) (Acc Good : Exchange → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∀ n, groupOf A n = .sessions →
    ((located (locOf A tab) n [] o.log).map fun y => widthOf y.1).sum ≤ 65536

/-- **`SessionsFit` at the form's acceptance**: the session-round locator puts at most one release of the log the
check read on each round (`located_one_accepted`), the play's log puts releases of the same frames there
(`located_widths`), and the read's one is an opening of at most one coin's elements (`readFits`). -/
theorem sessionsFit_of (A : Live₂) (C : CaseOf) (tab : ℕ → String) (Good : Exchange → Prop) (hsz : HeldSize)
    (hlog : InteriorLogged C Good) : SessionsFit A tab (FormAccepts C) Good := by
  intro o hA hG n hg
  obtain ⟨l, hl⟩ := interiorReads C hsz o hA
  obtain ⟨q, l', hlog', hq, hR⟩ := hlog o hA hG l hl
  have hF : List.Forall₂ (fun r r' => r.frame = r'.frame) l l' :=
    (List.forall₂_map_left_iff.1 hR).imp fun _ _ h => (of_ofV2 h).1
  have hS : Same [] q := by simpa using same_append (pre := []) hq
  have e : ((located (locOf A tab) n [] o.log).map fun y => widthOf y.1).sum =
      ((located (locOf A tab) n [] l).map fun y => widthOf y.1).sum := by
    unfold locOf
    rw [hlog', located_append, located_none q (fun y hy => (hq y hy).1) []]
    simp only [List.nil_append]
    exact located_widths hF hS
  rw [e]
  have h1 := located_one_accepted A C tab hsz o hA l hl n
  cases hx : located (locOf A tab) n [] l with
  | nil => simp
  | cons x t =>
    cases t with
    | cons y t =>
      rw [hx] at h1
      simp at h1
    | nil =>
      obtain ⟨a, r, b, he, hloc, rfl⟩ := mem_located (show x ∈ located (locOf A tab) n [] l by rw [hx]; simp)
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
      cases hfr : r.frame with
      | opening i s c => exact readFits C hsz o hA l hl r (by rw [he]; simp) i s c hfr
      | fresh _ => rw [hfr] at hloc; simp [locOf, sessLoc] at hloc
      | hello _ => rw [hfr] at hloc; simp [locOf, sessLoc] at hloc

end Verity.SampledTwo.SessionCoins
