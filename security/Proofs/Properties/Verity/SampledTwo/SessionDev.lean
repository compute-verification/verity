module

public import Proofs.Properties.Verity.SampledTwo.SessionHeld
public import Proofs.Properties.Verity.SampledTwo.CoinStages

/-!
# The coin map's release at the second stage's sessions

The coin map reads the device's release for a run's round through a locator (`CoinMap.Locate`, `CoinMap.dev`). The
two-stage audit's locator for the stages (`CoinMap.stageLocate`) puts the stages' families' fresh coins on the first
and second stages and nothing else. The run's locator (`runLocate`) adds the second stage's sessions: an opening goes
where the session-round locator puts it (`locOf`), a fresh coin where the stages' locator puts it, up to the second
stage's last round. At it:

* `locates_run`: it agrees with the session-round locator at every session's round (`LocatesSessions`);
* `devLocal_of`: at any locator that does, the coin map's release for a session's round is a function of the releases
  the session-round locator puts there (`DevLocal`), the elements they carry and their bits in order
  (`usedOf_located`, `devBits_located`);
* `devElems_of`: the coin map's agreement (`CoinMap.ElemAgrees`) gives the sessions' (`DevElems`), where the releases
  on each session's round fit in one coin (`SessionsFit`): below 65536 elements the coin map's count of the releases
  that fit (`CoinMap.fitOf`) is the count of all of them (`fitOf_eq_usedOf`);
* `sessionsFit_of`: on a play the form accepts, the session-round locator puts at most one release on each round
  (`located_one_accepted`), so `SessionsFit` holds where every opening the device releases is of at most one coin's
  elements (`OpensFit`).

At the stages' rounds the run's locator puts what the stages' locator puts, so `CoinMap.devFirst_stage` and
`CoinMap.devSecond_stage` read the same release there (`dev_congr`).
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

/-! ## The run's locator -/

/-- **The run's locator** at the first stage's family `ℓ₁`, the second stage's `ℓ₂` and the sessions' tables `tab`:
an opening on the session's round the session-round locator gives, a fresh coin on the stage the stages' locator gives
where that is at most the second stage's last, nothing else. -/
def runLocate (A : Live₂) (ℓ₁ ℓ₂ : String) (tab : ℕ → String) : CoinMap.Locate := fun pre fr b =>
  match fr with
  | .opening _ _ _ => locOf A tab pre fr
  | _ => (CoinMap.stageLocate ℓ₁ ℓ₂ pre fr b).filter fun n => decide (n ≤ A.h.stage)

theorem stage_lt_of_sessions {A : Live₂} {n : ℕ} (hg : groupOf A n = .sessions) : A.h.stage + 1 < n := by
  unfold groupOf at hg
  split_ifs at hg
  omega

/-- **The run's locator agrees with the session-round locator at the sessions' rounds.** -/
theorem locates_run (A : Live₂) (ℓ₁ ℓ₂ : String) (tab : ℕ → String) :
    LocatesSessions A tab (CoinMap.locAt (runLocate A ℓ₁ ℓ₂ tab)) := by
  intro pre fr b n hg
  have hn := stage_lt_of_sessions hg
  cases fr with
  | opening i s c => exact Iff.rfl
  | hello sp => simp [CoinMap.locAt, locOf, sessLoc]
  | fresh l =>
    simp only [CoinMap.locAt, runLocate, locOf, sessLoc, reduceCtorEq, iff_false]
    intro h
    obtain ⟨-, hle⟩ := Option.filter_eq_some_iff.1 h
    have := of_decide_eq_true hle
    omega

/-! ## The release a locator reads, from the releases it puts on the round -/

section Located

variable {L : CoinMap.Locate} {loc : List Release → Verity.Frame → Option ℕ} {n : ℕ}

/-- **The elements the coin map counts on round `n` are those the releases `loc` puts there carry**, where the coin
map's locator puts a release on round `n` just where `loc` does, and gives each such release `loc`'s width. -/
theorem usedOf_located (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ loc pre fr = some n)
    (hw : ∀ pre fr, loc pre fr = some n → CoinMap.widthOf fr = widthOf fr) :
    ∀ pre log : List Release,
      CoinMap.usedOf L n pre log = ((located loc n pre log).map fun y => widthOf y.1).sum
  | _, [] => rfl
  | pre, r :: log => by
    simp only [CoinMap.usedOf, located, List.map_append, List.sum_append, usedOf_located h hw (pre ++ [r]) log]
    by_cases hl : loc pre r.frame = some n
    · rw [if_pos ((h pre r.frame r.answers).2 hl), if_pos hl, hw pre r.frame hl]
      simp
    · rw [if_neg (mt (h pre r.frame r.answers).1 hl), if_neg hl]
      simp

/-- **The bits the coin map reads for round `n` are those the releases `loc` puts there carry**, in order. -/
theorem devBits_located (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ loc pre fr = some n) :
    ∀ pre log : List Release, CoinMap.devBits L n pre log = (located loc n pre log).flatMap (·.2)
  | _, [] => rfl
  | pre, r :: log => by
    simp only [CoinMap.devBits, located, List.flatMap_append, devBits_located h (pre ++ [r]) log]
    by_cases hl : loc pre r.frame = some n
    · rw [if_pos ((h pre r.frame r.answers).2 hl), if_pos hl]
      simp
    · rw [if_neg (mt (h pre r.frame r.answers).1 hl), if_neg hl]
      simp

/-- **The coin map's release for round `n`, from the releases `loc` puts there.** -/
theorem dev_located (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ loc pre fr = some n)
    (hw : ∀ pre fr, loc pre fr = some n → CoinMap.widthOf fr = widthOf fr) (log : List Release) :
    CoinMap.dev L log n =
      if ((located loc n [] log).map fun y => widthOf y.1).sum = 0 then none
      else some fun j => CoinMap.elemOfBits fun b => ((located loc n [] log).flatMap (·.2)).getD (128 * j.val + b) false := by
  unfold CoinMap.dev
  rw [usedOf_located h hw, devBits_located h]
  rfl

end Located

/-- **Two locators that put the same releases on round `n` read the same release there.** -/
theorem dev_congr {L L' : CoinMap.Locate} {n : ℕ}
    (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n) (log : List Release) :
    CoinMap.dev L log n = CoinMap.dev L' log n := by
  have hu : ∀ pre log : List Release, CoinMap.usedOf L n pre log = CoinMap.usedOf L' n pre log := by
    intro pre log
    induction log generalizing pre with
    | nil => rfl
    | cons r log ih =>
      simp only [CoinMap.usedOf, ih (pre ++ [r])]
      by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
      · rw [if_pos hl, if_pos ((h pre r.frame r.answers).1 hl)]
      · rw [if_neg hl, if_neg (mt (h pre r.frame r.answers).2 hl)]
  have hb : ∀ pre log : List Release, CoinMap.devBits L n pre log = CoinMap.devBits L' n pre log := by
    intro pre log
    induction log generalizing pre with
    | nil => rfl
    | cons r log ih =>
      simp only [CoinMap.devBits, ih (pre ++ [r])]
      by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
      · rw [if_pos hl, if_pos ((h pre r.frame r.answers).1 hl)]
      · rw [if_neg hl, if_neg (mt (h pre r.frame r.answers).2 hl)]
  unfold CoinMap.dev
  rw [hu, hb]

/-- The session-round locator puts only openings on a round, and the coin map gives an opening its count of coins. -/
theorem widthOf_locOf {A : Live₂} {tab : ℕ → String} {n : ℕ} :
    ∀ pre fr, locOf A tab pre fr = some n → CoinMap.widthOf fr = widthOf fr
  | _, .opening _ _ _, _ => rfl
  | _, .fresh _, h => by simp [locOf, sessLoc] at h
  | _, .hello _, h => by simp [locOf, sessLoc] at h

/-- **`DevLocal` at the coin map's release**, at any locator that agrees with the session-round locator at the
sessions' rounds. -/
theorem devLocal_of (A : Live₂) (tab : ℕ → String) (L : CoinMap.Locate)
    (hL : LocatesSessions A tab (CoinMap.locAt L)) : DevLocal A (locOf A tab) (CoinMap.dev L) := by
  intro n hg log log' he
  rw [dev_located (fun pre fr b => hL pre fr b n hg) widthOf_locOf log,
    dev_located (fun pre fr b => hL pre fr b n hg) widthOf_locOf log', he]

/-! ## The coin map's agreement at the sessions' rounds -/

theorem usedOf_append (L : CoinMap.Locate) (n : ℕ) :
    ∀ p a b : List Release, CoinMap.usedOf L n p (a ++ b) = CoinMap.usedOf L n p a + CoinMap.usedOf L n (p ++ a) b
  | p, [], b => by simp [CoinMap.usedOf]
  | p, r :: a, b => by
    simp only [List.cons_append, CoinMap.usedOf, usedOf_append L n (p ++ [r]) a b, List.append_assoc,
      List.nil_append]
    omega

/-- **Below one coin's elements, the releases that fit are all of them.** -/
theorem fitOf_eq_usedOf (L : CoinMap.Locate) (n : ℕ) :
    ∀ pre log : List Release, CoinMap.usedOf L n [] pre + CoinMap.usedOf L n pre log ≤ 65536 →
      CoinMap.fitOf L n pre log = CoinMap.usedOf L n pre log
  | _, [], _ => rfl
  | pre, r :: log, h => by
    have hpre : CoinMap.usedOf L n [] (pre ++ [r]) = CoinMap.usedOf L n [] pre +
        (if CoinMap.locAt L pre r.frame r.answers = some n then CoinMap.widthOf r.frame else 0) := by
      rw [usedOf_append]
      simp [CoinMap.usedOf]
    simp only [CoinMap.usedOf] at h
    simp only [CoinMap.fitOf, CoinMap.usedOf]
    rw [fitOf_eq_usedOf L n (pre ++ [r]) log (by rw [hpre]; omega)]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
    · rw [if_pos hl] at h
      rw [if_pos ⟨hl, by omega⟩, if_pos hl]
    · rw [if_neg (fun h' => hl h'.1), if_neg hl]

/-- **The releases on each session's round fit in one coin**: on an accepted play whose log is the device's, the
releases the session-round locator puts on a session's round carry at most 65536 elements. -/
def SessionsFit (A : Live₂) (tab : ℕ → String) (Acc Good : Exchange → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∀ n, groupOf A n = .sessions →
    ((located (locOf A tab) n [] o.log).map fun y => widthOf y.1).sum ≤ 65536

/-- **`DevElems` from the coin map's agreement**: at a locator that agrees with the session-round locator at the
sessions' rounds, reading every round the run's verifier reads, where the releases on each session's round fit in one
coin. -/
theorem devElems_of (A : Live₂) (x : Claims.Msg) (tab : ℕ → String) (L : CoinMap.Locate) (reads : ℕ → Prop)
    (Acc Good : Exchange → Prop) (hL : LocatesSessions A tab (CoinMap.locAt L))
    (hreads : ∀ i, readsAt A.h i → reads i) (hfit : SessionsFit A tab Acc Good) :
    DevElems A x (locOf A tab) (CoinMap.dev L) Acc Good (CoinMap.ElemAgrees L A x reads) := by
  intro o hA hG r hAg i hr hg c hc e he
  have hu := usedOf_located (fun pre fr b => hL pre fr b i.val hg) widthOf_locOf [] o.log
  refine hAg i (hreads _ hr) c hc e ?_
  rw [fitOf_eq_usedOf L i.val [] o.log (by rw [hu, show CoinMap.usedOf L i.val [] [] = 0 from rfl, Nat.zero_add]; exact hfit o hA hG i.val hg), hu]
  exact he

theorem frame_mem {l l' : List Release} (h : List.Forall₂ (fun r r' => r.frame = r'.frame) l l') :
    ∀ r ∈ l, ∃ r' ∈ l', r.frame = r'.frame := by
  induction h with
  | nil => intro r hr; exact absurd hr List.not_mem_nil
  | cons hab _ ih =>
    intro r hr
    rcases List.mem_cons.1 hr with rfl | hr
    · exact ⟨_, by simp, hab⟩
    · obtain ⟨r', hr', e⟩ := ih r hr
      exact ⟨r', List.mem_cons_of_mem _ hr', e⟩

/-- **Every opening the device releases is of at most one coin's elements**: on a play whose log is the device's, an
opening of `n` coins has `n ≤ 65536`. -/
def OpensFit (Good : Exchange → Prop) : Prop :=
  ∀ o, Good o → ∀ r ∈ o.log, ∀ i s n, r.frame = .opening i s n → n ≤ 65536

/-- **`SessionsFit` at the form's acceptance**: the session-round locator puts at most one release of the log the
check read on each round, the play's log puts releases of the same frames there, and each is an opening of at most one
coin's elements. -/
theorem sessionsFit_of (A : Live₂) (C : CaseOf) (tab : ℕ → String) (Good : Exchange → Prop) (hsz : HeldSize)
    (hlog : InteriorLogged C Good) (hfit : OpensFit Good) : SessionsFit A tab (FormAccepts C) Good := by
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
      obtain ⟨r', hr', hrr'⟩ := frame_mem hF r (by rw [he]; simp)
      have hr'o : r' ∈ o.log := by rw [hlog']; exact List.mem_append_right _ hr'
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
      cases hfr : r.frame with
      | opening i s c =>
        rw [hfr] at hrr'
        exact hfit o hG r' hr'o i s c hrr'.symm
      | fresh _ => rw [hfr] at hloc; simp [locOf, sessLoc] at hloc
      | hello _ => rw [hfr] at hloc; simp [locOf, sessLoc] at hloc

end Verity.SampledTwo.SessionCoins
