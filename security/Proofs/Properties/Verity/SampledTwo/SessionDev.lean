module

public import Proofs.Properties.Verity.SampledTwo.SessionAgrees
public import Proofs.Properties.Verity.SampledTwo.AttachmentFits
public import Proofs.Properties.Verity.SampledTwo.SessionRecord
public import Proofs.Properties.Verity.SampledTwo.CoinMap

/-! The run's locator (`runLocate`), the stages' locator with the second stage's sessions added, and the coin
map's release at the sessions' rounds. -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

def runLocate (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) : CoinMap.Locate := fun pre fr b =>
  match fr with
  | .opening _ _ _ => locOf A tab pre fr
  | _ => (Ls pre fr b).filter fun n => decide (n ≤ A.h.stage)

theorem stage_lt_of_sessions {A : Live₂} {n : ℕ} (hg : groupOf A n = .sessions) : A.h.stage + 1 < n := by
  unfold groupOf at hg
  split_ifs at hg
  omega

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

section Located

variable {L : CoinMap.Locate} {loc : List Release → Verity.Frame → Option ℕ} {n : ℕ}

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

theorem usedOf_congr (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n) :
    ∀ pre log : List Release, CoinMap.usedOf L n pre log = CoinMap.usedOf L' n pre log
  | _, [] => rfl
  | pre, r :: log => by
    simp only [CoinMap.usedOf, usedOf_congr h (pre ++ [r]) log]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
    · rw [if_pos hl, if_pos ((h pre r.frame r.answers).1 hl)]
    · rw [if_neg hl, if_neg (mt (h pre r.frame r.answers).2 hl)]

theorem devBits_congr (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n) :
    ∀ pre log : List Release, CoinMap.devBits L n pre log = CoinMap.devBits L' n pre log
  | _, [] => rfl
  | pre, r :: log => by
    simp only [CoinMap.devBits, devBits_congr h (pre ++ [r]) log]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
    · rw [if_pos hl, if_pos ((h pre r.frame r.answers).1 hl)]
    · rw [if_neg hl, if_neg (mt (h pre r.frame r.answers).2 hl)]

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

theorem dev_congr (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n)
    (log : List Release) : CoinMap.dev L log n = CoinMap.dev L' log n := by
  unfold CoinMap.dev
  rw [usedOf_congr h, devBits_congr h]

end Congr

theorem dev_run_stage (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hLs : ∀ pre i s c b, Ls pre (.opening i s c) b = none) {n : ℕ} (hn : n ≤ A.h.stage) (log : List Release) :
    CoinMap.dev (runLocate A Ls tab) log n = CoinMap.dev Ls log n :=
  dev_congr (fun pre fr b => locAt_run_stage A Ls tab hLs hn pre fr b) log

theorem fitOf_run_stage (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hLs : ∀ pre i s c b, Ls pre (.opening i s c) b = none) {n : ℕ} (hn : n ≤ A.h.stage) (pre log : List Release) :
    CoinMap.fitOf (runLocate A Ls tab) n pre log = CoinMap.fitOf Ls n pre log :=
  fitOf_congr (fun pre fr b => locAt_run_stage A Ls tab hLs hn pre fr b) pre log

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

theorem widthOf_locOf {A : Live₂} {tab : ℕ → String} {n : ℕ} :
    ∀ pre fr, locOf A tab pre fr = some n → CoinMap.widthOf fr = widthOf fr
  | _, .opening _ _ _, _ => rfl
  | _, .fresh _, h => by simp [locOf, sessLoc] at h
  | _, .hello _, h => by simp [locOf, sessLoc] at h

theorem devLocal_of (A : Live₂) (tab : ℕ → String) (L : CoinMap.Locate)
    (hL : LocatesSessions A tab (CoinMap.locAt L)) : DevLocal A (locOf A tab) (CoinMap.dev L) := by
  intro n hg log log' he
  rw [dev_located (fun pre fr b => hL pre fr b n hg) widthOf_locOf log,
    dev_located (fun pre fr b => hL pre fr b n hg) widthOf_locOf log', he]

theorem usedOf_append (L : CoinMap.Locate) (n : ℕ) :
    ∀ p a b : List Release, CoinMap.usedOf L n p (a ++ b) = CoinMap.usedOf L n p a + CoinMap.usedOf L n (p ++ a) b
  | p, [], b => by simp [CoinMap.usedOf]
  | p, r :: a, b => by
    simp only [List.cons_append, CoinMap.usedOf, usedOf_append L n (p ++ [r]) a b, List.append_assoc,
      List.nil_append]
    omega

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

theorem coinsAgree_run (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String) (Ls : CoinMap.Locate)
    (Good : Exchange → Prop) (hlog : InteriorLogged C Good) :
    CoinsAgree A (formReading A C (withCoins A C tab (CoinMap.dev (runLocate A Ls tab)) Ms)
      (CoinMap.dev (runLocate A Ls tab))) (FormAccepts C) Good .sessions :=
  coinsAgree_sessions A C Ms tab _ Good (devLocal_of A tab _ (locates_run A Ls tab)) hlog

theorem sessionsAgree_run (A : Live₂) (x : Claims.Msg) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (Ls : CoinMap.Locate) (Good : Exchange → Prop) (hsz : HeldSize) (hlog : InteriorLogged C Good) :
    SessionsAgree A x (withCoins A C tab (CoinMap.dev (runLocate A Ls tab)) Ms) (CoinMap.dev (runLocate A Ls tab))
      (FormAccepts C) Good (CoinMap.ElemAgrees (runLocate A Ls tab) A x (readsAt A.h)) :=
  sessionsAgree_accepted A x C Ms tab _ Good _ hsz hlog
    (devElems_of A x tab _ (readsAt A.h) (FormAccepts C) Good (locates_run A Ls tab) (fun _ h => h)
      (sessionsFit_of A C tab Good hsz hlog))

theorem devReads_run (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) :
    DevReads A (locOf A tab) (CoinMap.dev (runLocate A Ls tab)) CoinMap.elemOfBits := fun n hg log =>
  dev_located (fun pre fr b => locates_run A Ls tab pre fr b n hg) widthOf_locOf log

end Verity.SampledTwo.SessionCoins
