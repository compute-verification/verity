module

public import Proofs.Properties.Verity.SampledTwo.CutRead
public import Proofs.Properties.Challenger.Coins

/-!
# The play's log shows each stage's coins after one root

On a play whose logs the check read and found the device's (`ReadIsDevice`), every fresh coin the play's log shows of
the first stage's family is released after the boundary registration's digest, and every one of the second stage's
family after the interior registration's (`oneRoot_draws`, at the cut's two samples; `oneRoot_second`, the second
stage's in the case's words; `OneRootIn`).

A session's read shows a fresh coin of a family only after a root its log's lines show of that family
(`readLog_fresh`), and those are the roots `releasedRootsOf` lists (`Sampled.releasedRootsOf_ok`). The challenger's
stream for a sample refuses a session that shows a root of the sample's own family other than its digest
(`Coins.stream_ok`), so each session's read shows its own stage's family after its own digest alone (`first_oneRoot`,
`second_oneRoot`).

Where the two sessions are on one window, each read is the whole log, and that is all it takes. On two windows the
log is the first session's releases then the interior's, and the statement also needs each session to show the other
stage's family after that stage's digest alone: `OtherRoots`, a named `Prop` taken as a hypothesis. The form's accept
does not give it: on two windows it refuses a first session that shows a coin of the interior's family after the
interior's digest, which says nothing of that family's other roots there, and it holds the interior session's roots of
the boundary's family to nothing. `OtherRoots` holds wherever neither session's roots show the other stage's family at
all.

As in `Cut`, at a history holding one play's signatures (`SignsPlay`) a genuine signature of an earlier period that the
evidence carries is none the history signed, so it is the forger's, and the forger's term bounds nothing there.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private pairs from Spot.ChallengerSession

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Binding (Read)

/-! ## The statement -/

/-- **The log shows family `ℓ` after one root**: every fresh coin of family `ℓ` in `log` is released after `after`. -/
def OneRootIn (ℓ : String) (after : ByteArray) (log : List Release) : Prop :=
  ∀ rl ∈ log, rl.frame = .fresh ℓ → rl.answers = after

theorem oneRootIn_append {ℓ : String} {after : ByteArray} {a b : List Release} (ha : OneRootIn ℓ after a)
    (hb : OneRootIn ℓ after b) : OneRootIn ℓ after (a ++ b) :=
  fun rl hrl hfr => (List.mem_append.1 hrl).elim (fun h => ha rl h hfr) (fun h => hb rl h hfr)

/-- **A log the read is of shows what the read shows**: a read's frames and answers are the log's. -/
theorem oneRootIn_of_read {ℓ : String} {after : ByteArray} :
    ∀ {l log : List Release}, List.Forall₂ Read.Of (l.map Verity.Sampled.Binding.Read.ofV2) log →
      OneRootIn ℓ after l → OneRootIn ℓ after log
  | [], _, h, _ => by
    cases h
    exact fun _ hrl _ => absurd hrl List.not_mem_nil
  | x :: xs, _, h, hl => by
    cases h with
    | cons hxy hs =>
      intro rl hrl hfr
      rcases List.mem_cons.1 hrl with rfl | hrl
      · obtain ⟨hf, ha, -⟩ := hxy
        exact ha.symm.trans (hl x (List.mem_cons_self ..) (hf.trans hfr))
      · exact oneRootIn_of_read hs (fun y hy => hl y (List.mem_cons_of_mem _ hy)) rl hrl hfr

/-! ## A read's fresh coins are after roots its lines show -/

private theorem bind_some {α β : Type} {x : Option α} {f : α → Option β} {b : β} (h : (x >>= f) = some b) :
    ∃ a, x = some a ∧ f a = some b := by
  cases x with
  | none => cases h
  | some a => exact ⟨a, rfl, h⟩

theorem sel_cons_mem {L : String} {p : String × String} {ps : List (String × String)} {x : ByteArray}
    (h : x ∈ Verity.Sampled.sel L ps) : x ∈ Verity.Sampled.sel L (p :: ps) := by
  unfold Verity.Sampled.sel at h ⊢
  obtain ⟨y, hy, hf⟩ := List.mem_filterMap.1 h
  exact List.mem_filterMap.2 ⟨y, List.mem_cons_of_mem _ hy, hf⟩

theorem sel_cons_head {L a b : String} {ps : List (String × String)} {root : ByteArray}
    (h : Verity.Sampled.rootOf a = some (L, root)) : root ∈ Verity.Sampled.sel L ((a, b) :: ps) := by
  unfold Verity.Sampled.sel
  refine List.mem_filterMap.2 ⟨(a, b), List.mem_cons_self .., ?_⟩
  dsimp only
  rw [h]
  exact ite_eq_left rfl

/-- **A fresh coin the lines' read shows is after a root the lines show of its family**: the read places a fresh
release at each pair of lines that are no coin tree's, of the family and after the root the pair's first line names. -/
theorem readTail_fresh (label : String) (after : ByteArray) (records : List Json) :
    ∀ (ls : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      {out : List Release}, Verity.Sampled.readTail label after records ls ts pending ξ = some out →
      ∀ rl ∈ out, ∀ L, rl.frame = .fresh L →
        rl.answers ∈ Verity.Sampled.sel L (pairs (pending.toList ++ ls.filter fun l => !treeLine l))
  | [], _, none, _, _, h, rl, hrl, _, _ => by
    simp only [Verity.Sampled.readTail, Option.some.injEq] at h
    subst h
    cases hrl
  | [], _, some _, _, _, h, _, _, _, _ => by simp [Verity.Sampled.readTail] at h
  | l :: ls, ts, pending, ξ, out, h, rl, hrl, L, hL => by
    unfold Verity.Sampled.readTail at h
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at h
      obtain ⟨⟨r, ts'⟩, hX, h⟩ := bind_some h
      obtain ⟨j, -, hj⟩ := bind_some hX
      dsimp only at h
      obtain ⟨rest, hR, h⟩ := bind_some h
      have h' : r :: rest = out := Option.some.inj h
      subst h'
      rw [List.filter_cons_of_neg (by simp [ht])]
      rcases List.mem_cons.1 hrl with rfl | hrl
      · exact absurd hL (Verity.Sampled.treeStep_frame hj L)
      · exact readTail_fresh label after records ls ts' pending ξ hR rl hrl L hL
    · rw [ite_eq_right ht] at h
      cases pending with
      | none =>
        have hp : (none : Option String).toList ++ (l :: ls).filter (fun l => !treeLine l) =
            (some l).toList ++ ls.filter fun l => !treeLine l := by
          rw [List.filter_cons_of_pos (by simp [ht])]
          rfl
        rw [hp]
        exact readTail_fresh label after records ls ts (some l) ξ h rl hrl L hL
      | some a =>
        have hp : pairs ((some a).toList ++ (l :: ls).filter fun l => !treeLine l) =
            (a, l) :: pairs ((none : Option String).toList ++ ls.filter fun l => !treeLine l) := by
          rw [List.filter_cons_of_pos (by simp [ht])]
          rfl
        rw [hp]
        obtain ⟨⟨lab, root⟩, hra, h⟩ := bind_some h
        dsimp only at h
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at h
          obtain ⟨rest, hR, h⟩ := bind_some h
          have h' : _ :: rest = out := Option.some.inj h
          subst h'
          rcases List.mem_cons.1 hrl with rfl | hrl
          · have e : label = L := Verity.Frame.fresh.inj hL
            subst e
            exact sel_cons_head (by rw [hra, hm.1, hm.2])
          · exact sel_cons_mem (readTail_fresh label after records ls ts none _ hR rl hrl L hL)
        · rw [ite_eq_right hm] at h
          obtain ⟨rest, hR, h⟩ := bind_some h
          have h' : _ :: rest = out := Option.some.inj h
          subst h'
          rcases List.mem_cons.1 hrl with rfl | hrl
          · have e : lab = L := Verity.Frame.fresh.inj hL
            subst e
            exact sel_cons_head hra
          · exact sel_cons_mem (readTail_fresh label after records ls ts none ξ hR rl hrl L hL)

/-- **A fresh coin a log's read shows is after a root the log shows of its family**: one of the roots the root lines of
the log's pairs name for that family (`Sampled.sel`), the log's head and its coin trees' lines apart. -/
theorem readLog_fresh {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {out : List Release}
    (h : Verity.Sampled.readLog label after records ξ lines = some out) :
    ∀ rl ∈ out, ∀ L, rl.frame = .fresh L →
      rl.answers ∈ Verity.Sampled.sel L (pairs (lines.tail.filter fun l => !treeLine l)) := by
  intro rl hrl L hL
  cases lines with
  | nil =>
    simp only [Verity.Sampled.readLog, Option.some.injEq] at h
    subst h
    cases hrl
  | cons l ls =>
    simp only [Verity.Sampled.readLog] at h
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at h
      obtain ⟨⟨r, ts⟩, hX, h⟩ := bind_some h
      obtain ⟨j, -, hj⟩ := bind_some hX
      dsimp only at h
      obtain ⟨rest, hR, h⟩ := bind_some h
      have h' : r :: rest = out := Option.some.inj h
      subst h'
      rcases List.mem_cons.1 hrl with rfl | hrl
      · exact absurd hL (Verity.Sampled.treeStep_frame hj L)
      · exact readTail_fresh label after records ls ts none ξ hR rl hrl L hL
    · rw [ite_eq_right ht] at h
      exact readTail_fresh label after records ls [] none ξ h rl hrl L hL

/-- **A read shows a family after one root where its lines do.** -/
theorem read_oneRoot {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {out : List Release}
    (h : Verity.Sampled.readLog label after records ξ lines = some out) {L : String} {a : ByteArray}
    (hall : ∀ r ∈ Verity.Sampled.sel L (pairs (lines.tail.filter fun l => !treeLine l)), r = a) :
    OneRootIn L a out :=
  fun rl hrl hfr => hall _ (readLog_fresh h rl hrl L hfr)

/-! ## A sample's session shows its own family after its digest alone -/

/-- **The session a sample's coins are read from**: its commitment and the roots it shows, its log's lines, every root
it shows of the sample's own family the sample's digest (the challenger's stream refuses any other), and the roots it shows
of each family those its lines' pairs name. -/
theorem drawCoins_roots {key : ChallengerKey} {w label : String} {after : ByteArray} {coins : Json} {ξ : ByteArray}
    (h : drawCoins key w label after coins = .ok ξ) :
    ∃ cm roots, ∃ (ls : Array Json) (lines : List String),
      trustedSessionOf (Spot.getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
        (Spot.getD coins "pins") (Spot.getD coins "main") (Spot.getD coins "session") = .ok (cm, roots) ∧
      Spot.getD (Spot.getD coins "session") "log" = .arr ls ∧ ls.toList = lines.map Json.str ∧
      (∀ r ∈ Verity.Sampled.look roots label, r = after) ∧
      ∀ L, Verity.Sampled.look roots L =
        Verity.Sampled.sel L (pairs (lines.tail.filter fun l => !treeLine l)) := by
  obtain ⟨cm, roots, hts, hst⟩ := Verity.Sampled.drawCoins_ok h
  obtain ⟨_, hsn⟩ := Verity.Sampled.trustedSessionOf_ok hts
  obtain ⟨_, hso⟩ := Verity.Sampled.sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := Verity.Sampled.sessionOf_ok hso
  obtain ⟨_, _, _, _, -, -, -, -, -, -, hall⟩ := Properties.Challenger.Coins.stream_ok hst
  exact ⟨cm, roots, ls, lines, hts, hlog, hls, hall, Verity.Sampled.releasedRootsOf_ok hrr⟩

/-- **The first session's read shows a family after one root where its roots do**, and its roots show the first
sample's family after the sample's digest alone. -/
theorem first_oneRoot {key : ChallengerKey} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {d : Draws} {ξ : ByteArray} {r : List Release}
    (hd : firstDraws key p c e = some d)
    (hξ : drawCoins d.key d.window d.label d.after (firstCoins e) = .ok ξ) (hr : firstRead key p c e ξ = some r) :
    ∃ cm roots,
      trustedSessionOf (Spot.getD (firstCoins e) "ledger") (Json.mkObj [("root", Json.str d.window)]) d.key
        (Spot.getD (firstCoins e) "pins") (Spot.getD (firstCoins e) "main") (Spot.getD (firstCoins e) "session") =
          .ok (cm, roots) ∧
      (∀ x ∈ Verity.Sampled.look roots d.label, x = d.after) ∧
      ∀ L a, (∀ x ∈ Verity.Sampled.look roots L, x = a) → OneRootIn L a r := by
  obtain ⟨cm, roots, ls, lines, hts, hlog, hls, hown, hsel⟩ := drawCoins_roots hξ
  have hl : firstLines e = .ok lines := firstLines_eq hlog hls
  have he : firstRead key p c e ξ = Verity.Sampled.readLog d.label d.after [] ξ lines := by
    simp [firstRead, hd, hl, Except.toOption]
  rw [he] at hr
  refine ⟨cm, roots, hts, hown, fun L a hall => read_oneRoot hr ?_⟩
  rw [← hsel L]
  exact hall

/-- **The interior session's read shows a family after one root where its roots do**, and its roots show the second
sample's family after the sample's digest alone. -/
theorem second_oneRoot {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {bs : _root_.Core.Blobs} {c₂ : Spot.OneStage.Commitment} {d : Draws} {ξ : ByteArray} {r : List Release}
    (hd' : oneStageZkSource p.audit c₂ = some d) (hd : secondDraws p e = some d)
    (hξ : Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ) (hr : secondRead p e bs ξ = some r) :
    ∃ cm roots,
      trustedSessionOf (Spot.getD e.interior.base.coins "ledger") (Json.mkObj [("root", Json.str d.window)]) d.key
        (Spot.getD e.interior.base.coins "pins") (Spot.getD e.interior.base.coins "main")
        (Spot.getD e.interior.base.coins "session") = .ok (cm, roots) ∧
      (∀ x ∈ Verity.Sampled.look roots d.label, x = d.after) ∧
      ∀ L a, (∀ x ∈ Verity.Sampled.look roots L, x = a) → OneRootIn L a r := by
  obtain ⟨raw, hraw, -⟩ := oneStageZk_reads p.audit c₂ d hd' e.interior ξ hξ
  obtain ⟨cm, roots, ls, lines, hts, hlog, hls, hown, hsel⟩ := drawCoins_roots hraw
  have hl : Verity.Sampled.linesOf e.interior = .ok lines := Verity.Sampled.Binding.linesOf_eq hlog hls
  obtain ⟨recs, hrecs⟩ : ∃ recs, secondRecords bs e = some recs := by
    unfold secondRead at hr
    obtain ⟨_, -, hr⟩ := bind_some hr
    obtain ⟨recs, hrecs, -⟩ := bind_some hr
    exact ⟨recs, hrecs⟩
  have he : secondRead p e bs ξ = Verity.Sampled.readLog d.label d.after recs ξ lines := by
    simp [secondRead, hd, hrecs, hl, Except.toOption]
  rw [he] at hr
  refine ⟨cm, roots, hts, hown, fun L a hall => read_oneRoot hr ?_⟩
  rw [← hsel L]
  exact hall

/-! ## The premise on two windows -/

/-- **On two windows each session shows the other stage's family after that stage's digest alone**: where the case
reads as the two-stage form's, the parameters name a key and the two samples are on two windows, the roots the first
session shows of the second sample's family are all the second sample's digest, and the roots the interior session shows
of the first sample's family are all the first sample's digest. It holds wherever neither session's roots show the other
sample's family. The form's accept does not give it. -/
def OtherRoots (s : Verity.Sent) : Prop :=
  ∀ p c e key d₁ d₂, caseOf s = .ok (p, c, e) → keyOf s.params = some key → firstDraws key p c e = some d₁ →
    secondDraws p e = some d₂ → d₁.window ≠ d₂.window →
    (∀ cm roots,
      trustedSessionOf (Spot.getD (firstCoins e) "ledger") (Json.mkObj [("root", Json.str d₁.window)]) d₁.key
        (Spot.getD (firstCoins e) "pins") (Spot.getD (firstCoins e) "main") (Spot.getD (firstCoins e) "session") =
          .ok (cm, roots) →
      ∀ x ∈ Verity.Sampled.look roots d₂.label, x = d₂.after) ∧
    ∀ cm roots,
      trustedSessionOf (Spot.getD e.interior.base.coins "ledger") (Json.mkObj [("root", Json.str d₂.window)]) d₂.key
        (Spot.getD e.interior.base.coins "pins") (Spot.getD e.interior.base.coins "main")
        (Spot.getD e.interior.base.coins "session") = .ok (cm, roots) →
      ∀ x ∈ Verity.Sampled.look roots d₁.label, x = d₁.after

/-! ## The play's log -/

/-- **The play's log shows each sample's family after its digest alone**, on a play whose logs the check read and found
the device's (`ReadIsDevice`), with every step the check read on the way: on one window each session's read is the
whole log; on two, the log is the first session's releases then the interior's, each showing the other sample's family
after that sample's digest alone (`OtherRoots`). -/
theorem oneRoot_draws {o : Exchange} (hg : ReadIsDevice o) (hx : OtherRoots o.sent) :
    ∃ p c e key d₁ c₂ d₂ ξ, caseOf o.sent = .ok (p, c, e) ∧ keyOf o.sent.params = some key ∧
      firstDraws key p c e = some d₁ ∧ drawCoins d₁.key d₁.window d₁.label d₁.after (firstCoins e) = .ok ξ ∧
      interiorOf e = .ok c₂ ∧ secondDraws p e = some d₂ ∧
      OneRootIn d₁.label d₁.after o.log ∧ OneRootIn d₂.label d₂.after o.log := by
  obtain ⟨hread, hdev⟩ := hg
  cases hR : readsLogs o.sent with
  | nil => exact absurd hR hread
  | cons l₀ ls₀ =>
    obtain ⟨p, c, e, key, d₁, ξ, c₂, d₂, ξ₂, r₁, r₂, h1, h2, h3, h4, h5, h6, h7, h8, h9, hE⟩ :=
      readsLogs_cases (s := o.sent) (l := l₀) (by rw [hR]; exact List.mem_cons_self ..)
    obtain ⟨cm₁, roots₁, ht₁, ho₁, hall₁⟩ := first_oneRoot h3 h4 h8
    obtain ⟨cm₂, roots₂, ht₂, ho₂, hall₂⟩ := second_oneRoot (secondDraws_eq h5 h6) h6 h7 h9
    refine ⟨p, c, e, key, d₁, c₂, d₂, ξ, h1, h2, h3, h4, h5, h6, ?_⟩
    by_cases hw : d₁.window = d₂.window
    · have hl₁ : r₁ ∈ readsLogs o.sent := by
        rw [hE]
        unfold logsOf
        rw [ite_eq_left hw]
        exact List.mem_cons_self ..
      have hl₂ : r₂ ∈ readsLogs o.sent := by
        rw [hE]
        unfold logsOf
        rw [ite_eq_left hw]
        exact List.mem_cons_of_mem _ (List.mem_cons_self ..)
      exact ⟨oneRootIn_of_read (hdev r₁ hl₁) (hall₁ _ _ ho₁), oneRootIn_of_read (hdev r₂ hl₂) (hall₂ _ _ ho₂)⟩
    · have hl : r₁ ++ r₂ ∈ readsLogs o.sent := by
        rw [hE]
        unfold logsOf
        rw [ite_eq_right hw]
        exact List.mem_cons_self ..
      obtain ⟨hx₁, hx₂⟩ := hx p c e key d₁ d₂ h1 h2 h3 h6 hw
      exact ⟨oneRootIn_of_read (hdev _ hl) (oneRootIn_append (hall₁ _ _ ho₁) (hall₂ _ _ (hx₂ cm₂ roots₂ ht₂))),
        oneRootIn_of_read (hdev _ hl) (oneRootIn_append (hall₁ _ _ (hx₁ cm₁ roots₁ ht₁)) (hall₂ _ _ ho₂))⟩

/-! ## The second stage, in the case's words -/

theorem source_draws {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment} {d : Draws} {key : ChallengerKey}
    {label : String} (h : oneStageZkSource p c = some d) (hch : p.base.challenger = some (key, label)) :
    d.label = label ∧ d.after = c.digest := by
  unfold oneStageZkSource oneStageSource at h
  rcases hw : p.base.verifier.windowRoot with _ | w <;> simp only [hch, hw] at h <;> cases h
  exact ⟨rfl, rfl⟩

/-- **The play's log shows the second stage's family after the interior registration's digest alone**: the family the
interior audit's parameters name for its challenger, on a play whose logs the check read and found the device's
(`ReadIsDevice`), where on two windows each session shows the other stage's family after that stage's digest alone
(`OtherRoots`). -/
theorem oneRoot_second {o : Exchange} (hg : ReadIsDevice o) (hx : OtherRoots o.sent)
    {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence}
    {c₂ : Spot.OneStage.Commitment} {k : ChallengerKey} {label : String} (hcase : caseOf o.sent = .ok (p, c, e))
    (h5 : interiorOf e = .ok c₂) (hch : p.audit.base.challenger = some (k, label)) :
    OneRootIn label c₂.digest o.log := by
  obtain ⟨p', c', e', key, d₁, c₂', d₂, ξ, h1, -, -, -, h5', h6, -, ho⟩ := oneRoot_draws hg hx
  have hpce := Except.ok.inj (h1.symm.trans hcase)
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  obtain rfl : c₂' = c₂ := Except.ok.inj (h5'.symm.trans h5)
  obtain ⟨hlab, haft⟩ := source_draws (secondDraws_eq h5' h6) hch
  rw [← hlab, ← haft]
  exact ho

end Verity.SampledTwo
