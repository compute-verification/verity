module

public import Proofs.Properties.Verity.SampledTwo.SessionDev

/-! The session-round locator reads only the frames before a release; that the log before a round's opening
is the same at two challenge vectors agreeing on the earlier rounds is not established here. -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Verity.Sampled.Binding (Read)

theorem openTrees_frames :
    ∀ {l l' : List Release}, List.Forall₂ (fun a b => a.frame = b.frame) l l' →
      (l.filterMap fun r => openTree r.frame) = l'.filterMap fun r => openTree r.frame
  | _, _, .nil => rfl
  | x :: _, y :: _, .cons h t => by
    have e := openTrees_frames t
    have hf : x.frame = y.frame := h
    simp only [List.filterMap_cons, hf, e]

theorem opened_frames {i s : ℕ} :
    ∀ {l l' : List Release}, List.Forall₂ (fun a b => a.frame = b.frame) l l' → openedOf l i s = openedOf l' i s
  | _, _, .nil => rfl
  | _ :: _, _ :: _, .cons h t => by
    have e := opened_frames (i := i) (s := s) t
    simp only [openedOf, List.countP_cons, h] at e ⊢
    rw [e]

theorem same_of_frames {pre pre' : List Release} (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') :
    Same pre pre' := by
  refine ⟨?_, Verity.Sampled.helloSpecs_frames h, fun _ _ => opened_frames h⟩
  unfold treesOpened
  rw [openTrees_frames h]

theorem locOf_frames (A : Live₂) (tab : ℕ → String) {pre pre' : List Release}
    (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') (fr : Verity.Frame) :
    locOf A tab pre fr = locOf A tab pre' fr := by
  unfold locOf
  exact sessLoc_congr (same_of_frames h) fr

theorem locOf_after (A : Live₂) (tab : ℕ → String) {q q' : List Release} {rl rl' : Release}
    (hq : List.Forall₂ (fun a b => a.frame = b.frame) q q') (hf : rl.frame = rl'.frame) :
    locOf A tab q rl.frame = locOf A tab q' rl'.frame :=
  (locOf_frames A tab hq rl.frame).trans (congrArg (locOf A tab q') hf)

theorem locates_frames {A : Live₂} {tab : ℕ → String}
    {L : List Release → Verity.Frame → ByteArray → Option ℕ} (hL : LocatesSessions A tab L) {n : ℕ}
    (hg : groupOf A n = .sessions) {pre pre' : List Release}
    (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') (fr : Verity.Frame) (b b' : ByteArray) :
    L pre fr b = some n ↔ L pre' fr b' = some n :=
  have e := locOf_frames A tab h fr
  ⟨fun hn => (hL pre' fr b' n hg).2 (e.symm.trans ((hL pre fr b n hg).1 hn)),
    fun hn => (hL pre fr b n hg).2 (e.trans ((hL pre' fr b' n hg).1 hn))⟩

theorem locates_after {A : Live₂} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) {n : ℕ} (hg : groupOf A n = .sessions)
    {q q' : List Release} {rl rl' : Release}
    (hq : List.Forall₂ (fun a b => a.frame = b.frame) q q') (hf : rl.frame = rl'.frame) :
    CoinMap.locAt L q rl.frame rl.answers = some n ↔ CoinMap.locAt L q' rl'.frame rl'.answers = some n := by
  rw [← hf]
  exact locates_frames hL hg hq rl.frame rl.answers rl'.answers

theorem same_append_of_frames {a a' : List Release} (ha : List.Forall₂ (fun x y => x.frame = y.frame) a a') :
    ∀ {pre pre' : List Release}, Same pre pre' → Same (pre ++ a) (pre' ++ a') := by
  induction ha with
  | nil =>
    intro pre pre' h
    simpa using h
  | @cons x y a a' hxy _ ih =>
    intro pre pre' h
    simpa using ih (h.snoc hxy)

theorem frames_of_read {l l' : List Release} (hR : List.Forall₂ Read.Of (l.map Read.ofV2) l') :
    List.Forall₂ (fun x y => x.frame = y.frame) l l' :=
  (List.forall₂_map_left_iff.1 hR).imp fun _ _ h => (of_ofV2 h).1

theorem locOf_logged (A : Live₂) (tab : ℕ → String) {q a a' : List Release} {r r' : Release}
    (hq : ∀ x ∈ q, openTree x.frame = none ∧ ∀ sp, x.frame ≠ .hello sp)
    (ha : List.Forall₂ (fun x y => x.frame = y.frame) a a') (hf : r.frame = r'.frame) :
    locOf A tab a r.frame = locOf A tab (q ++ a') r'.frame := by
  have hS : Same [] q := by simpa using same_append (pre := []) hq
  have hS' : Same a (q ++ a') := by simpa using same_append_of_frames ha hS
  rw [← hf]
  unfold locOf
  exact sessLoc_congr hS' r.frame

theorem locates_logged {A : Live₂} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) {n : ℕ} (hg : groupOf A n = .sessions)
    {q a a' : List Release} {r r' : Release}
    (hq : ∀ x ∈ q, openTree x.frame = none ∧ ∀ sp, x.frame ≠ .hello sp)
    (ha : List.Forall₂ (fun x y => x.frame = y.frame) a a') (hf : r.frame = r'.frame) :
    CoinMap.locAt L (q ++ a') r'.frame r'.answers = some n ↔ locOf A tab a r.frame = some n :=
  have e := locOf_logged A tab hq ha hf
  ⟨fun h => e.trans ((hL (q ++ a') r'.frame r'.answers n hg).1 h),
    fun h => (hL (q ++ a') r'.frame r'.answers n hg).2 (e.symm.trans h)⟩

def FramesOnly (L : CoinMap.Locate) : Prop :=
  ∀ (pre pre' : List Release) (fr : Verity.Frame) (b : ByteArray),
    List.Forall₂ (fun x y => x.frame = y.frame) pre pre' → L pre fr b = L pre' fr b

theorem framesOnly_run (A : Live₂) {Ls : CoinMap.Locate} (tab : ℕ → String) (hLs : FramesOnly Ls) :
    FramesOnly (runLocate A Ls tab) := by
  intro pre pre' fr b h
  cases fr with
  | opening i s c =>
    show locOf A tab pre (.opening i s c) = locOf A tab pre' (.opening i s c)
    exact locOf_frames A tab h _
  | hello sp =>
    show (Ls pre (.hello sp) b).filter (fun n => decide (n ≤ A.h.stage)) =
      (Ls pre' (.hello sp) b).filter (fun n => decide (n ≤ A.h.stage))
    rw [hLs pre pre' (.hello sp) b h]
  | fresh l =>
    show (Ls pre (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage)) =
      (Ls pre' (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage))
    rw [hLs pre pre' (.fresh l) b h]

theorem locAt_frames {L : CoinMap.Locate} (hL : FramesOnly L) {pre pre' : List Release}
    (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') (fr : Verity.Frame) (b : ByteArray) :
    CoinMap.locAt L pre fr b = CoinMap.locAt L pre' fr b := by
  cases fr with
  | hello _ => rfl
  | fresh l => exact hL pre pre' (.fresh l) b h
  | opening i s c => exact hL pre pre' (.opening i s c) b h

end Verity.SampledTwo.SessionCoins
