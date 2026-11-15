module

public import Proofs.Properties.Verity.SampledTwo.SessionAgrees
public import Proofs.Properties.Verity.SampledTwo.AttachmentFits
public import Proofs.Properties.Verity.SampledTwo.SessionRecord
public import Proofs.Properties.Verity.SampledTwo.CoinMap

/-!
# The coin map's release at the second stage's sessions

The coin map reads the device's release for a run's round through a locator (`CoinMap.Locate`, `CoinMap.dev`). The
run's locator (`runLocate`) is a locator for the stages, `Ls`, with the second stage's sessions added: an opening goes
where the session-round locator puts it (`locOf`), a fresh coin where `Ls` puts it, up to the second stage's last
round. The two-stage audit's `Ls` is the stages' locator, which puts the stages' families' fresh coins on the first
and second stages and nothing else. At `runLocate`, for any `Ls`:

* `locates_run`: it agrees with the session-round locator at every session's round (`LocatesSessions`);
* `devLocal_of`: at any locator that does, the coin map's release for a session's round is a function of the releases
  the session-round locator puts there (`DevLocal`), the elements they carry and their bits in order
  (`usedOf_located`, `devBits_located`);
* `devElems_of`: the coin map's agreement (`CoinMap.ElemAgrees`) gives the sessions' (`DevElems`), where the releases
  on each session's round fit in one coin (`SessionsFit`, which `sessionsFit_of` proves at the form's acceptance):
  below 65536 elements the coin map's count of the releases that fit (`CoinMap.fitOf`) is the count of all of them
  (`fitOf_eq_usedOf`).

* `coinsAgree_run`, `sessionsAgree_run`: the piece at the coin map's release: `CoinsAgree` at the sessions' rounds
  and `SessionsAgree`, at the run's locator and the coin map's agreement there, from `InteriorLogged` and `HeldSize`.
* `devReads_run`: the coin map's release there reads a session's round's releases as `coin_record` takes it
  (`DevReads`, at the coin map's `elemOfBits`), so the coin the check read for a session's round is its record's.

At a stage's round the run's locator puts what `Ls` puts, where `Ls` puts no opening on a round
(`locAt_run_stage`; the session-round locator puts a release only past the statements' round, `locOf_past`), so the
coin map reads the same release there at either (`dev_congr`, `dev_run_stage`), counts the same elements that fit
in the round's coin (`fitOf_congr`, `fitOf_run_stage`), and its agreement at the run's locator is its agreement at
`Ls` on those rounds (`elemAgrees_run_stage`).
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

/-! ## The run's locator -/

/-- **The run's locator** at the stages' locator `Ls` and the sessions' tables `tab`: an opening on the session's
round the session-round locator gives, any other frame on the stage `Ls` gives where that is at most the second
stage's last, nothing else. -/
def runLocate (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) : CoinMap.Locate := fun pre fr b =>
  match fr with
  | .opening _ _ _ => locOf A tab pre fr
  | _ => (Ls pre fr b).filter fun n => decide (n ≤ A.h.stage)

theorem stage_lt_of_sessions {A : Live₂} {n : ℕ} (hg : groupOf A n = .sessions) : A.h.stage + 1 < n := by
  unfold groupOf at hg
  split_ifs at hg
  omega

/-- **The run's locator agrees with the session-round locator at the sessions' rounds.** -/
theorem locates_run (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) :
    LocatesSessions A tab (CoinMap.locAt (runLocate A Ls tab)) := by
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

/-- The session-round locator puts a release only on a round past the statements'. -/
theorem locOf_past {A : Live₂} {tab : ℕ → String} {pre : List Release} {fr : Verity.Frame} {n : ℕ}
    (h : locOf A tab pre fr = some n) : A.h.stage + 1 < n := by
  cases fr with
  | fresh _ => simp [locOf, sessLoc] at h
  | hello _ => simp [locOf, sessLoc] at h
  | opening i s c =>
    simp only [locOf, sessLoc] at h
    obtain ⟨j, -, hif⟩ := Option.bind_eq_some_iff.1 h
    have hc : (treesOpened pre).idxOf i < A.h.ms.length ∧ j < SpotFlock.k₀ A.h ((treesOpened pre).idxOf i) := by
      by_contra hc
      rw [if_neg hc] at hif
      cases hif
    rw [if_pos hc] at hif
    have hle := SpotFlock.stage_le_off A.h ((treesOpened pre).idxOf i)
    have hn := Option.some.inj hif
    omega

/-- **At a stage's round the run's locator puts what the stages' locator puts**, where the stages' locator puts no
opening on a round: at a round `n` at most the second stage's last. -/
theorem locAt_run_stage (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hLs : ∀ pre i s c b, Ls pre (.opening i s c) b = none) {n : ℕ} (hn : n ≤ A.h.stage) (pre : List Release)
    (fr : Verity.Frame) (b : ByteArray) :
    CoinMap.locAt (runLocate A Ls tab) pre fr b = some n ↔ CoinMap.locAt Ls pre fr b = some n := by
  cases fr with
  | hello sp => exact Iff.rfl
  | opening i s c =>
    constructor
    · intro h
      have := locOf_past (show locOf A tab pre (.opening i s c) = some n from h)
      omega
    · intro h
      have h' : Ls pre (.opening i s c) b = some n := h
      rw [hLs] at h'
      cases h'
  | fresh l =>
    show (Ls pre (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage)) = some n ↔ Ls pre (.fresh l) b = some n
    rw [Option.filter_eq_some_iff]
    exact ⟨fun h => h.1, fun h => ⟨h, decide_eq_true hn⟩⟩

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
      else some fun j => CoinMap.elemOfBits fun b =>
        ((located loc n [] log).flatMap (·.2)).getD (128 * j.val + b) false := by
  unfold CoinMap.dev
  rw [usedOf_located h hw, devBits_located h]
  rfl

end Located

section Congr

variable {L L' : CoinMap.Locate} {n : ℕ}

/-- **Two locators that put the same releases on round `n` count the same elements there.** -/
theorem usedOf_congr (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n) :
    ∀ pre log : List Release, CoinMap.usedOf L n pre log = CoinMap.usedOf L' n pre log
  | _, [] => rfl
  | pre, r :: log => by
    simp only [CoinMap.usedOf, usedOf_congr h (pre ++ [r]) log]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
    · rw [if_pos hl, if_pos ((h pre r.frame r.answers).1 hl)]
    · rw [if_neg hl, if_neg (mt (h pre r.frame r.answers).2 hl)]

/-- **Two locators that put the same releases on round `n` read the same bits for it.** -/
theorem devBits_congr (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n) :
    ∀ pre log : List Release, CoinMap.devBits L n pre log = CoinMap.devBits L' n pre log
  | _, [] => rfl
  | pre, r :: log => by
    simp only [CoinMap.devBits, devBits_congr h (pre ++ [r]) log]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
    · rw [if_pos hl, if_pos ((h pre r.frame r.answers).1 hl)]
    · rw [if_neg hl, if_neg (mt (h pre r.frame r.answers).2 hl)]

/-- **Two locators that put the same releases on round `n` count the same elements that fit in its coin.** -/
theorem fitOf_congr (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n) :
    ∀ pre log : List Release, CoinMap.fitOf L n pre log = CoinMap.fitOf L' n pre log
  | _, [] => rfl
  | pre, r :: log => by
    simp only [CoinMap.fitOf, fitOf_congr h (pre ++ [r]) log, usedOf_congr h [] pre]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
    · have hl' := (h pre r.frame r.answers).1 hl
      by_cases hw : CoinMap.usedOf L' n [] pre + CoinMap.widthOf r.frame ≤ 65536
      · rw [if_pos ⟨hl, hw⟩, if_pos ⟨hl', hw⟩]
      · rw [if_neg (fun c => hw c.2), if_neg (fun c => hw c.2)]
    · rw [if_neg (fun c => hl c.1), if_neg (fun c => hl ((h pre r.frame r.answers).2 c.1))]

/-- **Two locators that put the same releases on round `n` read the same release there.** -/
theorem dev_congr (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n)
    (log : List Release) : CoinMap.dev L log n = CoinMap.dev L' log n := by
  unfold CoinMap.dev
  rw [usedOf_congr h, devBits_congr h]

end Congr

/-- **The coin map's release at a stage's round is the same at the run's locator as at the stages'**, where the
stages' locator puts no opening on a round. -/
theorem dev_run_stage (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hLs : ∀ pre i s c b, Ls pre (.opening i s c) b = none) {n : ℕ} (hn : n ≤ A.h.stage) (log : List Release) :
    CoinMap.dev (runLocate A Ls tab) log n = CoinMap.dev Ls log n :=
  dev_congr (fun pre fr b => locAt_run_stage A Ls tab hLs hn pre fr b) log

/-- **The elements that fit in a stage's round's coin are counted the same at the run's locator as at the
stages'**, where the stages' locator puts no opening on a round. -/
theorem fitOf_run_stage (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hLs : ∀ pre i s c b, Ls pre (.opening i s c) b = none) {n : ℕ} (hn : n ≤ A.h.stage) (pre log : List Release) :
    CoinMap.fitOf (runLocate A Ls tab) n pre log = CoinMap.fitOf Ls n pre log :=
  fitOf_congr (fun pre fr b => locAt_run_stage A Ls tab hLs hn pre fr b) pre log

/-- **The coin map's agreement at the run's locator is its agreement at the stages', at the stages' rounds**: where
the run's coins have the elements the releases the run's locator puts on each read round carry, they have those the
stages' locator's releases carry, at each read round at most the second stage's last. -/
theorem elemAgrees_run_stage (A : Live₂) (x : Claims.Msg) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hLs : ∀ pre i s c b, Ls pre (.opening i s c) b = none) (reads : ℕ → Prop) {o : Exchange} {r : Coins A x}
    (h : CoinMap.ElemAgrees (runLocate A Ls tab) A x reads o r) :
    CoinMap.ElemAgrees Ls A x (fun i => reads i ∧ i ≤ A.h.stage) o r := by
  intro i hi c hc j hj
  refine h i hi.1 c ?_ j ?_
  · rw [dev_run_stage A Ls tab hLs hi.2]
    exact hc
  · rw [fitOf_run_stage A Ls tab hLs hi.2]
    exact hj

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
  have hle : CoinMap.usedOf L i.val [] [] + CoinMap.usedOf L i.val [] o.log ≤ 65536 := by
    rw [hu, show CoinMap.usedOf L i.val [] [] = 0 from rfl, Nat.zero_add]
    exact hfit o hA hG i.val hg
  rw [fitOf_eq_usedOf L i.val [] o.log hle, hu]
  exact he

/-! ## The piece at the coin map's release -/

/-- **The sessions' coins agree as the device's releases do, at the coin map's release** (`CoinsAgree` at the
sessions' rounds, as `ProjectionOf` takes it): at the run's locator over any stages' locator, from
`InteriorLogged`. -/
theorem coinsAgree_run (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String) (Ls : CoinMap.Locate)
    (Good : Exchange → Prop) (hlog : InteriorLogged C Good) :
    CoinsAgree A (formReading A C (withCoins A C tab (CoinMap.dev (runLocate A Ls tab)) Ms)
      (CoinMap.dev (runLocate A Ls tab))) (FormAccepts C) Good .sessions :=
  coinsAgree_sessions A C Ms tab _ Good (devLocal_of A tab _ (locates_run A Ls tab)) hlog

/-- **The sessions' releases agree with the run's coins, at the coin map's release and its agreement**
(`SessionsAgree`, as `devAgrees_formReading` takes it): at the run's locator over any stages' locator and the coin
map's agreement there on the rounds the run's verifier reads, from `InteriorLogged` and `HeldSize`. -/
theorem sessionsAgree_run (A : Live₂) (x : Claims.Msg) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (Ls : CoinMap.Locate) (Good : Exchange → Prop) (hsz : HeldSize) (hlog : InteriorLogged C Good) :
    SessionsAgree A x (withCoins A C tab (CoinMap.dev (runLocate A Ls tab)) Ms) (CoinMap.dev (runLocate A Ls tab))
      (FormAccepts C) Good (CoinMap.ElemAgrees (runLocate A Ls tab) A x (readsAt A.h)) :=
  sessionsAgree_accepted A x C Ms tab _ Good _ hsz hlog
    (devElems_of A x tab _ (readsAt A.h) (FormAccepts C) Good (locates_run A Ls tab) (fun _ h => h)
      (sessionsFit_of A C tab Good hsz hlog))

/-- **The coin map's release at the run's locator reads a session's round's releases** (`DevReads`, as `coin_record`
and `agree_record` take it): at a session's round it is the coin of the bits the releases the session-round locator
puts there carry, in order. -/
theorem devReads_run (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) :
    DevReads A (locOf A tab) (CoinMap.dev (runLocate A Ls tab)) CoinMap.elemOfBits := fun n hg log =>
  dev_located (fun pre fr b => locates_run A Ls tab pre fr b n hg) widthOf_locOf log

end Verity.SampledTwo.SessionCoins
