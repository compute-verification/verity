module

public import Proofs.Properties.Verity.SampledTwo.Cut
public import Proofs.Properties.Verity.Sampled.Lines
public import Proofs.Properties.Challenger.Drawn

/-!
# The cut's binding where the device printed the play's log in its two sessions

`Cut` reduces the cut to two premises, `Binds` and `SignsPlay`. This module proves `Binds` from what the device's
history printed (`PrintsTwo`) and from the interior's records being held to its session's coin trees (`InteriorHeld`,
which the form's per-member check holds: `OneStageZk.membersRun`'s `membersTreesHeld`).

* **One session** (`read_on`): where `drawCoins` accepts a sample's coin evidence, it read the coins a run of the device
  released and that run's lines (`Drawn.drawn_log`), or the finder outputs a break. The runs on the sample's window
  print the play's log (`PrintsOn`), so the history keeping only those runs, with its ledger, prints it in the sense
  `Lines.PrintsLog` states (`printsLog_onWindow`), and `Lines.read_run` reads the session's releases as that log but for
  openings' coins of one SHA-512. `Lines.read_run` reads the history only at the run it reads, so no other run on
  another window need print this log.
* **The first session** (`first_session`): `readsLogs` itself flips its coins under the parameters' key, so nothing of
  the form's check is read; its read takes no records.
* **The interior session** (`second_session`): the coins the second stage read are what `drawCoins` reads from the
  interior audit's coin evidence (`oneStageZk_reads`), and its records are held to the trees its lines hold
  (`InteriorHeld`).
* **Both** (`binds_of_prints`): on one window the sessions are one, and each read is the play's log; on two, the
  first's releases then the interior's are. A read that is the log but for an opening's coins of one SHA-512, whose
  coins aren't all the device's, gives an opening collision (`Lines.open_collision`) in the matching read without the
  samples' coins (`opensReads`).

The history at each play may hold earlier runs and lines: `read_on` reads only the runs the sample names and the
ledger. `PrintsTwo` is a premise at the history the leg's `dev` gives; at a one-play history it is the step
`Device.lean`'s printing proves.
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Binding (Read)

/-! ## One session -/

/-- **The device's history kept to the runs its ledger opens withheld on window `w`**, with all of its ledger. -/
noncomputable def onWindow (dev : History) (w : String) : History :=
  ⟨fun pk => (dev.runs pk).filter fun r => @decide (OpensOn dev pk w r.id) (Classical.propDecidable _), dev.ledger⟩

theorem mem_onWindow {dev : History} {w : String} {pk : ByteArray} {r : Properties.Challenger.Run}
    (hr : r ∈ dev.runs pk) (ho : OpensOn dev pk w r.id) : r ∈ (onWindow dev w).runs pk :=
  List.mem_filter.2 ⟨hr, @decide_eq_true _ (Classical.propDecidable _) ho⟩

theorem opensOn_onWindow {dev : History} {w w' : String} {pk : ByteArray} {id : String}
    (ho : OpensOn dev pk w' id) : OpensOn (onWindow dev w) pk w' id := ho

/-- The runs on window `w` printing `log` is the history kept to them printing it. -/
theorem printsLog_onWindow {dev : History} {w : String} {log : List Release} (h : PrintsOn dev w log) :
    Verity.Sampled.Lines.PrintsLog (onWindow dev w) log := by
  intro pk w' r hr _
  obtain ⟨hr, ho⟩ := List.mem_filter.1 hr
  obtain ⟨h1, -, h3⟩ := h pk r hr (@of_decide_eq_true _ (Classical.propDecidable _) ho)
  exact ⟨h1, h3⟩

/-- **`Lines.read_run` on lines a round's coins are read against** (`Lines.TreeLines`), however the records were held
to them: `read_run`'s own proof, which reads its records only through `Lines.tree_lines`. -/
theorem read_run_lines {dev : History} {log : List Release} (hP : Verity.Sampled.Lines.PrintsLog dev log)
    {pk : ByteArray} {w label : String} {after : ByteArray} {recs : List Json} {r : Properties.Challenger.Run}
    (hr : r ∈ dev.runs pk) (ho : OpensOn dev pk w r.id)
    (hcr : ∃ c roots, r.commitment = some c ∧ releasedRootsOf c r.log = .ok roots) {ts : Array LoggedTree}
    (hts : coinTreesOf r.log = .ok ts) (hL : Verity.Sampled.Lines.TreeLines recs r.log) {out : List Release}
    (h : Verity.Sampled.readLog label after recs (r.released label after) r.log = some out) :
    List.Forall₂ Verity.Sampled.Lines.ReadOf out log := by
  obtain ⟨hp, hξ⟩ := hP pk w r hr ho
  obtain ⟨c, roots, -, hroots⟩ := id hcr
  rw [Verity.Sampled.Lines.readLog_head hroots] at h
  have hT := Verity.Sampled.Lines.trees_log hts
  have hn : (r.log.filterMap Verity.Sampled.Lines.helloRoot).Nodup := hT.roots ▸ hT.nodup
  refine Verity.Sampled.Lines.read_prints hp .nil (hL.sub fun _ hl => List.mem_of_mem_tail hl) ?_
    (fun _ h => absurd h List.not_mem_nil) (hξ hcr label after) h
  simpa using hn.sublist ((List.tail_sublist r.log).filterMap _)

/-- **A session's read, where `drawCoins` accepts its coin evidence and the runs on its window print `log`, is `log`
but for openings' coins of one SHA-512**, or the finder outputs a break: the lines it read are the sampled run's
(`hl`), and its records give each round opened the coins those lines show (`hL`). -/
theorem read_on {dev : History} {key : ChallengerKey} {w label : String} {after ξ : ByteArray} {coins : Json}
    {log : List Release} {lines : List String} {recs : List Json} {out : List Release}
    (hP : PrintsOn dev w log) (hξ : drawCoins key w label after coins = .ok ξ)
    (hl : ∀ (ls : Array Json) (l : List String), Spot.getD (Spot.getD coins "session") "log" = .arr ls →
      ls.toList = l.map Json.str → lines = l)
    (hL : ∀ ts, coinTreesOf lines = .ok ts → Verity.Sampled.Lines.TreeLines recs lines)
    (h : Verity.Sampled.readLog label after recs ξ lines = some out) :
    List.Forall₂ Verity.Sampled.Lines.ReadOf out log ∨ Break dev key label coins := by
  rcases Drawn.drawn_log (dev := dev) hξ with ⟨s, hs, ho, hξs, hcr, ls, hls, hlog⟩ | b
  · obtain rfl : lines = s.log := hl ls s.log hls hlog
    obtain ⟨-, ⟨ts, hts⟩, -⟩ := hP key.pub s hs ho
    rw [hξs] at h
    exact .inl (read_run_lines (printsLog_onWindow hP) (mem_onWindow hs ho) (opensOn_onWindow ho) hcr hts
      (hL ts hts) h)
  · exact .inr b

/-! ## A member's tree less its closing round -/

theorem find?_dropLast {α : Type} {p : α → Bool} {x : α} :
    ∀ l : List α, l.dropLast.find? p = some x → l.find? p = some x
  | [], h => by simp at h
  | [_], h => by simp at h
  | a :: b :: l, h => by
    have e : (a :: b :: l).dropLast = a :: (b :: l).dropLast := rfl
    rw [e] at h
    cases hp : p a
    · simp only [List.find?_cons, hp] at h ⊢
      exact find?_dropLast (b :: l) h
    · simp only [List.find?_cons, hp] at h ⊢
      exact h

/-- A round the tree less its closing round shows opened is one the tree shows opened. -/
theorem opening_unclosed {t : LoggedTree} {S : String} {k : ℕ} {o : TreeOpening}
    (h : t.unclosed.opening S k = some o) : t.opening S k = some o := by
  unfold LoggedTree.opening LoggedTree.unclosed at h
  unfold LoggedTree.opening
  rw [← Array.find?_toList] at h ⊢
  rw [Array.toList_pop] at h
  exact find?_dropLast _ h

/-- **Records held to the trees less their closing rounds give each round opened the coins its line shows**, as
`Lines.tree_lines` gives it for records held to the trees: a round the tree less its closing round shows opened is one
the tree shows (`opening_unclosed`). -/
theorem tree_lines_unclosed {lines : List String} {ts : Array LoggedTree} {recs : List Json}
    (h : coinTreesOf lines = .ok ts)
    (hrecs : ∀ rec ∈ recs, ∃ t ∈ ts, ∃ tabs, sessionTreeReason #[t.unclosed] tabs rec = none) :
    Verity.Sampled.Lines.TreeLines recs lines := by
  have hT := Verity.Sampled.Lines.trees_log h
  refine ⟨hT.spec, fun l hl j hev hnt hop R S k C hR hS hk hC rec hrec htree cs hcs => ?_⟩
  obtain ⟨t, ht, htr, o, ho, hoc⟩ := hT.opened l hl j hev hnt hop R S k C hR hS hk hC
  obtain ⟨t₀, ht₀, tabs, htabs⟩ := hrecs rec hrec
  obtain ⟨t', ht', htr', o', ho', hoc'⟩ := Verity.Sampled.Trees.round_held htabs hcs
  obtain rfl : t' = t₀.unclosed := by simpa using ht'
  have e : t₀ = t := List.inj_on_of_nodup_map hT.nodup (Array.mem_toList_iff.2 ht₀) (Array.mem_toList_iff.2 ht)
    (htr'.symm.trans (htree.trans htr.symm))
  subst e
  have ho'' := opening_unclosed ho'
  rw [ho] at ho''
  cases ho''
  rw [← hoc, hoc']

/-! ## The first session -/

/-- The first session's lines, read as `firstLines` reads them, are the lines its coin evidence's log holds. -/
theorem firstLines_eq {e : VStar.Recursive.SampledTwo.Evidence} {ls : Array Json} {l : List String}
    (h : Spot.getD (Spot.getD (firstCoins e) "session") "log" = .arr ls) (hl : ls.toList = l.map Json.str) :
    firstLines e = .ok l := by
  unfold firstLines
  rw [h]
  show ls.toList.mapM _ = _
  rw [hl]
  exact Verity.Sampled.Binding.mapM_map_str (fun _ => rfl) l

/-- **The first session's read is the log its window's runs print**, but for openings' coins of one SHA-512, with
the same read without the samples' coins agreeing with it; or its record carries a break. -/
theorem first_session {dev : History} {key : ChallengerKey} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence} {d : Draws}
    (hd : firstDraws key p c e = some d) {ξ : ByteArray}
    (hξ : drawCoins d.key d.window d.label d.after (firstCoins e) = .ok ξ) {log : List Release}
    (hP : PrintsOn dev d.window log) {out : List Release} (h : firstRead key p c e ξ = some out) :
    (List.Forall₂ Verity.Sampled.Lines.ReadOf out log ∧
      ∃ out', firstRead key p c e .empty = some out' ∧ List.Forall₂ Verity.Sampled.Lines.Agree out out') ∨
      Break dev d.key d.label (firstCoins e) := by
  cases hl : firstLines e with
  | error _ => simp [firstRead, hd, hl, Except.toOption] at h
  | ok lines =>
    have h' : Verity.Sampled.readLog d.label d.after [] ξ lines = some out := by
      simpa [firstRead, hd, hl, Except.toOption] using h
    have he : firstRead key p c e .empty = Verity.Sampled.readLog d.label d.after [] .empty lines := by
      simp [firstRead, hd, hl, Except.toOption]
    have hlines : ∀ (ls : Array Json) (l : List String),
        Spot.getD (Spot.getD (firstCoins e) "session") "log" = .arr ls → ls.toList = l.map Json.str → lines = l :=
      fun _ _ hls hl' => Except.ok.inj (hl.symm.trans (firstLines_eq hls hl'))
    obtain ⟨out', h₀, hag⟩ := Verity.Sampled.Lines.readLog_agree .empty h'
    rcases read_on hP hξ hlines (fun _ hts => Verity.Sampled.Lines.tree_lines hts fun _ hm => by cases hm) h' with
      hr | b
    · exact .inl ⟨hr, out', he.trans h₀, hag⟩
    · exact .inr b

/-! ## The interior session -/

/-- **The interior's records are held to the coin trees its session's lines hold**, as the form's per-member check
holds them (`OneStageZk.membersRun`'s `membersTreesHeld`, `memberTreeReason`): every record `secondRecords` reads
names a tree `coinTreesOf` reads from the lines `linesOf` reads, and passes it less its closing round. -/
def InteriorHeld (s : Verity.Sent) : Prop :=
  ∀ p c e, caseOf s = .ok (p, c, e) → ∀ recs, secondRecords (_root_.Core.Blobs.ofArray s.blobs) e = some recs →
    ∀ lines, Verity.Sampled.linesOf e.interior = .ok lines → ∀ ts, coinTreesOf lines = .ok ts →
      ∀ rec ∈ recs, ∃ t ∈ ts, ∃ tabs, sessionTreeReason #[t.unclosed] tabs rec = none

/-- **The interior session's read is the log its window's runs print**, but for openings' coins of one SHA-512, with
the same read without the samples' coins agreeing with it; or its record carries a break. -/
theorem second_session {dev : History} {p : VStar.Recursive.SampledTwo.Params}
    {e : VStar.Recursive.SampledTwo.Evidence} {bs : _root_.Core.Blobs} {c₂ : Spot.OneStage.Commitment}
    (hc₂ : interiorOf e = .ok c₂) {d : Draws} (hd : secondDraws p e = some d) {ξ : ByteArray}
    (hξ : Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ) {log : List Release}
    (hP : PrintsOn dev d.window log)
    (hH : ∀ recs, secondRecords bs e = some recs → ∀ lines, Verity.Sampled.linesOf e.interior = .ok lines →
      ∀ ts, coinTreesOf lines = .ok ts →
        ∀ rec ∈ recs, ∃ t ∈ ts, ∃ tabs, sessionTreeReason #[t.unclosed] tabs rec = none)
    {out : List Release} (h : secondRead p e bs ξ = some out) :
    (List.Forall₂ Verity.Sampled.Lines.ReadOf out log ∧
      ∃ out', secondRead p e bs .empty = some out' ∧ List.Forall₂ Verity.Sampled.Lines.Agree out out') ∨
      Break dev d.key d.label e.interior.base.coins := by
  have hd' : oneStageZkSource p.audit c₂ = some d := by
    unfold secondDraws at hd
    rw [hc₂] at hd
    exact hd
  obtain ⟨raw, hraw, hrx⟩ := oneStageZk_reads p.audit c₂ d hd' e.interior ξ hξ
  rw [hrx] at hraw
  cases hr : secondRecords bs e with
  | none => simp [secondRead, hd, hr] at h
  | some recs =>
    cases hl : Verity.Sampled.linesOf e.interior with
    | error _ => simp [secondRead, hd, hr, hl, Except.toOption] at h
    | ok lines =>
      have h' : Verity.Sampled.readLog d.label d.after recs ξ lines = some out := by
        simpa [secondRead, hd, hr, hl, Except.toOption] using h
      have he : secondRead p e bs .empty = Verity.Sampled.readLog d.label d.after recs .empty lines := by
        simp [secondRead, hd, hr, hl, Except.toOption]
      have hlines : ∀ (ls : Array Json) (l : List String),
          Spot.getD (Spot.getD e.interior.base.coins "session") "log" = .arr ls → ls.toList = l.map Json.str →
            lines = l :=
        fun _ _ hls hl' => Except.ok.inj (hl.symm.trans (Verity.Sampled.Binding.linesOf_eq hls hl'))
      obtain ⟨out', h₀, hag⟩ := Verity.Sampled.Lines.readLog_agree .empty h'
      rcases read_on hP hraw hlines (fun ts hts => tree_lines_unclosed hts (hH recs hr lines hl ts hts)) h' with
        hr' | b
      · exact .inl ⟨hr', out', he.trans h₀, hag⟩
      · exact .inr b

/-! ## What the case names -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error _ => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- The two-stage case names the sampled component's parameters in the parameters sent. -/
theorem paramsAt_of_caseOf {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} (h : caseOf s = .ok (p, c, e)) : paramsAt s.params = .ok p := by
  unfold caseOf at h
  obtain ⟨p', hp, h⟩ := bind_ok h
  obtain ⟨ce, -, h⟩ := bind_ok h
  have hpe : p' = p := by
    first
      | exact congrArg Prod.fst (Except.ok.inj h)
      | (simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h; exact h.1)
  rw [hp, hpe]

theorem firstDraws_window {key : ChallengerKey} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence} {d : Draws}
    (h : firstDraws key p c e = some d) :
    ∃ bp, boundaryOf p = .ok bp ∧ bp.verifier.windowRoot = some d.window ∧ d.key = key := by
  unfold firstDraws at h
  cases hb : boundaryOf p with
  | error _ => simp [hb, Except.toOption] at h
  | ok bp =>
    cases hw : bp.verifier.windowRoot with
    | none => simp [hb, hw, Except.toOption] at h
    | some w =>
      cases hl : firstLabel p with
      | none => simp [hb, hw, hl, Except.toOption] at h
      | some lab =>
        have h' : some (⟨key, w, lab, c.digest⟩ : Draws) = some d := by
          rw [← h]
          simp [hb, hw, hl, Except.toOption]
        obtain rfl := Option.some.inj h'
        first
          | exact ⟨bp, rfl, hw, rfl⟩
          | exact ⟨bp, hb, hw, rfl⟩

theorem secondDraws_eq {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {c₂ : Spot.OneStage.Commitment} {d : Draws} (h5 : interiorOf e = .ok c₂) (h : secondDraws p e = some d) :
    oneStageZkSource p.audit c₂ = some d := by
  unfold secondDraws at h
  rw [h5] at h
  exact h

theorem source_window {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment} {d : Draws}
    (h : oneStageZkSource p c = some d) : p.base.verifier.windowRoot = some d.window := by
  unfold oneStageZkSource oneStageSource at h
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h <;> cases h
  first | rfl | simp [hw]

/-- **The two sessions' windows are the ones the parameters name**: the first's sample's and the interior's. -/
theorem windowsOf_eq {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {key : ChallengerKey} {d₁ d₂ : Draws} {c₂ : Spot.OneStage.Commitment}
    (h1 : caseOf s = .ok (p, c, e)) (h3 : firstDraws key p c e = some d₁) (h5 : interiorOf e = .ok c₂)
    (h6 : secondDraws p e = some d₂) : windowsOf s.params = some (d₁.window, d₂.window) := by
  obtain ⟨bp, hb, hw₁, -⟩ := firstDraws_window h3
  have hw₂ := source_window (secondDraws_eq h5 h6)
  unfold windowsOf
  rw [paramsAt_of_caseOf h1]
  simp [Except.toOption, hb, hw₁, hw₂]

/-- **The interior's sample is under the parameters' key.** -/
theorem key_second {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {key : ChallengerKey} {d₂ : Draws} {c₂ : Spot.OneStage.Commitment}
    (h1 : caseOf s = .ok (p, c, e)) (h2 : keyOf s.params = some key) (h5 : interiorOf e = .ok c₂)
    (h6 : secondDraws p e = some d₂) : d₂.key = key := by
  have hk := Verity.Sampled.Binding.key_of_source (secondDraws_eq h5 h6)
  have hK : keyOf s.params = (some d₂.key).filter atDeployed := by
    unfold keyOf
    rw [paramsAt_of_caseOf h1]
    exact congrArg (Option.filter atDeployed) hk
  rw [hK] at h2
  exact Option.some.inj (Option.filter_eq_some_iff.1 h2).1

theorem records_eq {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {key : ChallengerKey} (h1 : caseOf s = .ok (p, c, e))
    (h2 : keyOf s.params = some key) :
    records s = ((firstDraws key p c e).map fun d => (d, firstCoins e)).toList ++
      ((secondDraws p e).map fun d => (d, e.interior.base.coins)).toList := by
  first
    | (simp only [records, h1, h2]; done)
    | (unfold records; rw [h1, h2])

/-- **Where the form's check read both sessions, the logs it made of them**, with every step it read on the way. -/
theorem readsLogs_cases {s : Verity.Sent} {l : List Release} (hm : l ∈ readsLogs s) :
    ∃ p c e key d₁ ξ c₂ d₂ ξ₂ r₁ r₂, caseOf s = .ok (p, c, e) ∧ keyOf s.params = some key ∧
      firstDraws key p c e = some d₁ ∧ drawCoins d₁.key d₁.window d₁.label d₁.after (firstCoins e) = .ok ξ ∧
      interiorOf e = .ok c₂ ∧ secondDraws p e = some d₂ ∧
      Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧ firstRead key p c e ξ = some r₁ ∧
      secondRead p e (_root_.Core.Blobs.ofArray s.blobs) ξ₂ = some r₂ ∧
      readsLogs s = logsOf d₁.window d₂.window r₁ r₂ := by
  unfold readsLogs at hm
  cases h1 : caseOf s with
  | error _ => simp [h1, Except.toOption] at hm
  | ok pce =>
  obtain ⟨p, c, e⟩ := pce
  cases h2 : keyOf s.params with
  | none => simp [h1, h2, Except.toOption] at hm
  | some key =>
  cases h3 : firstDraws key p c e with
  | none => simp [h1, h2, h3, Except.toOption] at hm
  | some d₁ =>
  cases h4 : drawCoins d₁.key d₁.window d₁.label d₁.after (firstCoins e) with
  | error _ => simp [h1, h2, h3, h4, Except.toOption] at hm
  | ok ξ =>
  cases h5 : interiorOf e with
  | error _ => simp [h1, h2, h3, h4, h5, Except.toOption] at hm
  | ok c₂ =>
  cases h6 : secondDraws p e with
  | none => simp [h1, h2, h3, h4, h5, h6, Except.toOption] at hm
  | some d₂ =>
  cases h7 : Spot.OneStageZk.coinsOf p.audit c₂ e.interior with
  | error _ => simp [h1, h2, h3, h4, h5, h6, h7, Except.toOption] at hm
  | ok ξ₂ =>
  cases h8 : firstRead key p c e ξ with
  | none => simp [h1, h2, h3, h4, h5, h6, h7, h8, Except.toOption] at hm
  | some r₁ =>
  cases h9 : secondRead p e (_root_.Core.Blobs.ofArray s.blobs) ξ₂ with
  | none => simp [h1, h2, h3, h4, h5, h6, h7, h8, h9, Except.toOption] at hm
  | some r₂ =>
  refine ⟨p, c, e, key, d₁, ξ, c₂, d₂, ξ₂, r₁, r₂, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals first
    | assumption
    | rfl
    | (unfold readsLogs; simp [h1, h2, h3, h4, h5, h6, h7, h8, h9, Except.toOption])

theorem opensReads_eq {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {key : ChallengerKey} {d₁ d₂ : Draws} {o₁ o₂ : List Release}
    (h1 : caseOf s = .ok (p, c, e)) (h2 : keyOf s.params = some key) (h3 : firstDraws key p c e = some d₁)
    (h6 : secondDraws p e = some d₂) (h8 : firstRead key p c e .empty = some o₁)
    (h9 : secondRead p e (_root_.Core.Blobs.ofArray s.blobs) .empty = some o₂) :
    opensReads s = logsOf d₁.window d₂.window o₁ o₂ := by
  unfold opensReads
  simp [h1, h2, h3, h6, h8, h9, Except.toOption]

theorem forall₂_append {α β : Type} {R : α → β → Prop} {a₁ a₂ : List α} {b₁ b₂ : List β}
    (h₁ : List.Forall₂ R a₁ b₁) (h₂ : List.Forall₂ R a₂ b₂) : List.Forall₂ R (a₁ ++ a₂) (b₁ ++ b₂) := by
  induction h₁ with
  | nil => exact h₂
  | cons r _ ih => exact .cons r ih

/-! ## Both sessions -/

/-- **`Binds` where the device printed each play's log in its two sessions and the form held the interior's records
to its trees.** On an accepted play whose log the check read isn't the device's, each session's read is its window's
runs' log but for openings' coins of one SHA-512, or its record carries a break under the parameters' key
(`first_session`, `second_session`); where both read so, the logs the check made of them (`logsOf`) are the play's
log but for those coins, and one of them, whose coins aren't all the device's, gives an opening collision in its
read without the samples' coins (`Lines.open_collision`). -/
theorem binds_of_prints (dev : Exchange → History) (θ : ByteArray)
    (hI : ∀ s : Verity.Sent, s.params = θ → Accepts s → InteriorHeld s) : BindsOfPrints dev θ := by
  intro hP o hθ ha hl
  cases hR : readsLogs o.sent with
  | nil => exact (hl fun l hm => by rw [hR] at hm; cases hm).elim
  | cons l₀ ls₀ =>
  obtain ⟨p, c, e, key, d₁, ξ, c₂, d₂, ξ₂, r₁, r₂, h1, h2, h3, h4, h5, h6, h7, h8, h9, hE⟩ :=
    readsLogs_cases (s := o.sent) (l := l₀) (by rw [hR]; simp)
  obtain ⟨-, -, -, hk₁⟩ := firstDraws_window h3
  have hk₂ : d₂.key = key := key_second h1 h2 h5 h6
  have hm₁ : (d₁, firstCoins e) ∈ records o.sent := by rw [records_eq h1 h2, h3]; simp
  have hm₂ : (d₂, e.interior.base.coins) ∈ records o.sent := by rw [records_eq h1 h2, h6]; simp
  have hK : keyOf θ = some key := by rw [← hθ]; exact h2
  have hw : windowsOf θ = some (d₁.window, d₂.window) := by rw [← hθ]; exact windowsOf_eq h1 h3 h5 h6
  have hPT := hP o hθ d₁.window d₂.window hw
  have hH := hI o.sent hθ ha p c e h1
  unfold PrintsTwo at hPT
  by_cases hww : d₁.window = d₂.window
  · rw [if_pos hww] at hPT
    rcases first_session h3 h4 hPT h8 with ⟨hr₁, o₁, ho₁, ha₁⟩ | b₁
    · rcases second_session (log := o.log) h5 h6 h7 (by rw [← hww]; exact hPT) hH h9 with
        ⟨hr₂, o₂, ho₂, ha₂⟩ | b₂
      · have hO := opensReads_eq h1 h2 h3 h6 ho₁ ho₂
        unfold logsOf at hE hO
        rw [if_pos hww] at hE hO
        refine .inr ?_
        by_cases hc₁ : List.Forall₂ (fun y x => y.coins ≠ [] → y.coins = x.coins) r₁ o.log
        · have hc₂ : ¬ List.Forall₂ (fun y x => y.coins ≠ [] → y.coins = x.coins) r₂ o.log := by
            intro hc₂
            apply hl
            intro l hm
            rw [hE] at hm
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
            rcases hm with rfl | rfl
            · exact Verity.Sampled.Binding.logged_of_reads hr₁ hc₁
            · exact Verity.Sampled.Binding.logged_of_reads hr₂ hc₂
          obtain ⟨u, v, hu, hne, hh⟩ := Verity.Sampled.Lines.open_collision hr₂ ha₂ hc₂
          exact ⟨o₂, by rw [hO]; simp, u, v, hu, hne, hh⟩
        · obtain ⟨u, v, hu, hne, hh⟩ := Verity.Sampled.Lines.open_collision hr₁ ha₁ hc₁
          exact ⟨o₁, by rw [hO]; simp, u, v, hu, hne, hh⟩
      · exact .inl ⟨_, hm₂, by rw [hK, ← hk₂], b₂⟩
    · exact .inl ⟨_, hm₁, by rw [hK, ← hk₁], b₁⟩
  · rw [if_neg hww] at hPT
    obtain ⟨log₁, log₂, hlog, hP₁, hP₂⟩ := hPT
    rcases first_session h3 h4 hP₁ h8 with ⟨hr₁, o₁, ho₁, ha₁⟩ | b₁
    · rcases second_session h5 h6 h7 hP₂ hH h9 with ⟨hr₂, o₂, ho₂, ha₂⟩ | b₂
      · have hO := opensReads_eq h1 h2 h3 h6 ho₁ ho₂
        unfold logsOf at hE hO
        rw [if_neg hww] at hE hO
        refine .inr ?_
        have hr := forall₂_append hr₁ hr₂
        have hag := forall₂_append ha₁ ha₂
        rw [← hlog] at hr
        have hc : ¬ List.Forall₂ (fun y x => y.coins ≠ [] → y.coins = x.coins) (r₁ ++ r₂) o.log := by
          intro hc
          apply hl
          intro l hm
          rw [hE] at hm
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
          subst hm
          exact Verity.Sampled.Binding.logged_of_reads hr hc
        obtain ⟨u, v, hu, hne, hh⟩ := Verity.Sampled.Lines.open_collision hr hag hc
        exact ⟨o₁ ++ o₂, by rw [hO]; simp, u, v, hu, hne, hh⟩
      · exact .inl ⟨_, hm₂, by rw [hK, ← hk₂], b₂⟩
    · exact .inl ⟨_, hm₁, by rw [hK, ← hk₁], b₁⟩

end Verity.SampledTwo
