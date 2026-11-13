module

public import Proofs.Properties.Verity.SampledTwo.CutHeld
public import Proofs.Properties.Verity.SampledRead

/-!
# The cut where the check read a log

`LogIsDevice` speaks of every log the two-stage form's check read (`readsLogs`); on a play where it read none, it
holds of any log. This module states the cut with the read in its event (`ReadIsDevice`: the check read a log of both
sessions, `LogRead`, and every log it read is the device's), at the same terms (`cutBound_read`).

That an accepted play is one whose logs the cut reads does not follow from the form's acceptance alone:

* **The first stage's coins under the parameters' key** (`FirstReads`, a premise here). The run of record's coin check
  (`Spot.Run.coinsCode`) reads the first stage's coin evidence with `drawCoins` under the parameters' key, the boundary
  registration's window and family, after its digest, as the cut does (`keyOf`, `firstDraws`), and takes `null`
  evidence where the first stage read no byte. `CutFirst` derives `FirstReads` from the form's acceptance on every
  play whose first stage's coin evidence is not `null` (`FirstCoinsShown`); of a two-stage audit whose first stage
  reads no coin, the cut says nothing.
* **The first session's log read as coin trees.** The form reads the first session's log only through `drawCoins`,
  which reads no coin tree. The device's run prints its log as coin trees (`PrintsOn`), and the log `drawCoins` read
  is that run's or the record carries a break (`Drawn.drawn_log`); so the first read exists, or a break
  (`first_reads`).

The interior session's read exists on every accepted play (`second_reads`): the form's per-member check read its coin
trees (`members_trees`) and every statement's session's record on the blobs (`members_records`), and the second
stage's sample read its log's pairs (`oneStageZk_reads`, `Sampled.releasedRootsOf_pairsOk`), which is what the read
needs (`Sampled.readLog_isSome`).

So an accepted play is one whose logs the cut reads, or one of its two records carries a break (`reads_or_break`), and
the cut's bound holds with the read in its event (`cutBound_of`, `cutBound_read`).

The terms are `Cut`'s, at any history. As there, at a history holding one play's signatures (`SignsPlay`) a genuine
signature of an earlier period that the evidence carries is none the history signed, so it is the forger's, and the
forger's term bounds nothing there.
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open scoped ENNReal

/-! ## The event and the premise -/

/-- **The check read a log of both sessions**: `readsLogs` reads the first stage's session and the interior's. -/
def LogRead (o : Exchange) : Prop := readsLogs o.sent ≠ []

/-- **The check read a log of both sessions, and every log it read is the device's** (`LogRead`, `LogIsDevice`). -/
def ReadIsDevice (o : Exchange) : Prop := LogRead o ∧ LogIsDevice o

/-- **The first stage's coins are read under the parameters' key**, on every accepted play after the parameters `θ`:
they name a challenger key at the deployed LMS parameters (`keyOf`), the case names the first session's window and
family (`firstDraws`), and `drawCoins` under that key reads the first stage's coin evidence. -/
def FirstReads (θ : ByteArray) : Prop :=
  ∀ s : Verity.Sent, s.params = θ → Accepts s → ∀ p c e, caseOf s = .ok (p, c, e) →
    ∃ key d ξ, keyOf θ = some key ∧ firstDraws key p c e = some d ∧
      drawCoins d.key d.window d.label d.after (firstCoins e) = .ok ξ

theorem logsOf_ne_nil (w₁ w₂ : String) (r₁ r₂ : List Release) : logsOf w₁ w₂ r₁ r₂ ≠ [] := by
  unfold logsOf
  by_cases h : w₁ = w₂
  · rw [if_pos h]
    exact List.cons_ne_nil _ _
  · rw [if_neg h]
    exact List.cons_ne_nil _ _

theorem mapM_isSome_of {α β : Type} (f : α → Option β) :
    ∀ l : List α, (∀ x ∈ l, (f x).isSome) → (l.mapM f).isSome := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons x l ih =>
    intro hl
    rw [List.mapM_cons]
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (hl x (List.mem_cons_self ..))
    obtain ⟨r, hr⟩ := Option.isSome_iff_exists.1 (ih fun y hy => hl y (List.mem_cons_of_mem _ hy))
    rw [hb, hr]
    rfl

/-! ## The first session's read -/

/-- **The first session's read exists where the runs on its window print a log**, or its record carries a break: the
lines `drawCoins` read are a run's of the device (`Drawn.drawn_log`), which read as coin trees (`PrintsOn`) and as its
coin commitment's pairs. -/
theorem first_reads {dev : History} {key : ChallengerKey} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence} {d : Draws}
    (hd : firstDraws key p c e = some d) {ξ : ByteArray}
    (hξ : drawCoins d.key d.window d.label d.after (firstCoins e) = .ok ξ) {log : List Release}
    (hP : PrintsOn dev d.window log) :
    (∃ out, firstRead key p c e ξ = some out) ∨ Break dev d.key d.label (firstCoins e) := by
  rcases Drawn.drawn_log (dev := dev) hξ with ⟨s, hs, ho, -, hcr, ls, hls, hlog⟩ | b
  · obtain ⟨-, ⟨ts, hts⟩, -⟩ := hP d.key.pub s hs ho
    obtain ⟨cm, roots, -, hroots⟩ := hcr
    have hl : firstLines e = .ok s.log := firstLines_eq hls hlog
    have hT : (Verity.Sampled.treeFold [] [] s.log).isSome := by
      rw [Verity.Sampled.coinTreesOf_fold [] hts]
      rfl
    obtain ⟨out, hout⟩ := Option.isSome_iff_exists.1
      (Verity.Sampled.readLog_isSome d.label d.after [] ξ hT (Verity.Sampled.releasedRootsOf_pairsOk hroots))
    have he : firstRead key p c e ξ = Verity.Sampled.readLog d.label d.after [] ξ s.log := by
      simp [firstRead, hd, hl, Except.toOption]
    exact .inl ⟨out, he.trans hout⟩
  · exact .inr b

/-! ## The interior session's read -/

/-- **`membersTreesHeld` read every statement's session's record on the blobs**, as UTF-8 (`Sampled.recordOn`). -/
theorem members_records {ts : Array LoggedTree} {L : List String}
    {sts : Array (Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member)} {bs : _root_.Core.Blobs}
    (h : Spot.OneStageZk.membersTreesHeld ts L sts bs = .ok ()) :
    ∀ x ∈ sts.toList, ∀ s ∈ x.1.sessions.toList, (Verity.Sampled.recordOn bs s).isSome := by
  unfold Spot.OneStageZk.membersTreesHeld at h
  dsimp only at h
  obtain ⟨named, hn, -⟩ := Verity.Sampled.Trees.bind_ok h
  rw [← Array.forIn_toList] at hn
  have hI := Verity.Sampled.Trees.forIn_inv
    (I := fun pre (_ : Array (String × Array (String × Option ℕ) × Json × String)) =>
      ∀ x ∈ pre, ∀ s ∈ x.1.sessions.toList, (Verity.Sampled.recordOn bs s).isSome)
    ?_ sts.toList [] #[] named (by simp) hn
  · simpa using hI
  · rintro pre ⟨st, m⟩ named' res hI' hres
    dsimp only at hres
    obtain ⟨named'', hin, hres⟩ := Verity.Sampled.Trees.bind_ok hres
    obtain rfl : ForInStep.yield named'' = res := Except.ok.inj hres
    rw [← Array.forIn_toList] at hin
    have hJ := Verity.Sampled.Trees.forIn_inv
      (I := fun pre' (_ : Array (String × Array (String × Option ℕ) × Json × String)) =>
        ∀ s ∈ pre', (Verity.Sampled.recordOn bs s).isSome)
      ?_ st.sessions.toList [] named' named'' (by simp) hin
    · refine ⟨named'', rfl, fun x hx s hs => ?_⟩
      rcases List.mem_append.1 hx with hx | hx
      · exact hI' x hx s hs
      · rw [List.mem_singleton] at hx
        subst hx
        exact hJ s (by simpa using hs)
    · intro pre' s n res' hJ' hres'
      obtain ⟨b', hb', hres'⟩ := Verity.Sampled.Trees.bind_ok hres'
      split at hres'
      case h_2 => first | cases hres' | simp [throw, throwThe, MonadExceptOf.throw] at hres'
      rename_i record hrec
      obtain ⟨ev, -, hres'⟩ := Verity.Sampled.Trees.bind_ok hres'
      obtain rfl := Except.ok.inj hres'
      refine ⟨_, rfl, fun s' hs' => ?_⟩
      rcases List.mem_append.1 hs' with hs' | hs'
      · exact hJ' s' hs'
      · rw [List.mem_singleton] at hs'
        subst hs'
        first
          | (simp [Verity.Sampled.recordOn, hb', hrec, Except.toOption]; done)
          | (unfold Verity.Sampled.recordOn
             rw [Verity.Sampled.Trees.get_ok _ hb']
             simp [hrec])

/-- **The interior session's read exists** where the second stage's sample read its coins, the form read the
challenger's trees from the interior's coin evidence (`treesOf`), and every statement's session's record is on the
blobs: its lines read as coin trees (`Binding.trees_of`, `Sampled.coinTreesOf_fold`) and as its coin commitment's
pairs (`Sampled.releasedRootsOf_pairsOk`). -/
theorem second_reads {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {bs : _root_.Core.Blobs} {c₂ : Spot.OneStage.Commitment} {d : Draws}
    (hd' : oneStageZkSource p.audit c₂ = some d) (hd : secondDraws p e = some d) {ξ : ByteArray}
    (hξ : Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ) {ts : Array LoggedTree}
    (hts : Spot.OneStageZk.treesOf p.audit.base e.interior.base = .ok ts)
    (hrec : ∀ rf ∈ e.statements.toList.flatMap (·.sessions.toList), (Verity.Sampled.recordOn bs rf).isSome) :
    ∃ out, secondRead p e bs ξ = some out := by
  obtain ⟨raw, hraw, -⟩ := oneStageZk_reads p.audit c₂ d hd' e.interior ξ hξ
  obtain ⟨cm, roots, htsn, -⟩ := Verity.Sampled.drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := Verity.Sampled.trustedSessionOf_ok htsn
  obtain ⟨_, hso⟩ := Verity.Sampled.sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := Verity.Sampled.sessionOf_ok hso
  have hl : Verity.Sampled.linesOf e.interior = .ok lines := Verity.Sampled.Binding.linesOf_eq hlog hls
  have hct : coinTreesOf lines = .ok ts := Verity.Sampled.Binding.trees_of hts hlog hls
  obtain ⟨recs, hrecs⟩ := Option.isSome_iff_exists.1 (mapM_isSome_of (Verity.Sampled.recordOn bs) _ hrec)
  have hr : secondRecords bs e = some recs := hrecs
  have hT : (Verity.Sampled.treeFold recs [] lines).isSome := by
    rw [Verity.Sampled.coinTreesOf_fold recs hct]
    rfl
  obtain ⟨out, hout⟩ := Option.isSome_iff_exists.1
    (Verity.Sampled.readLog_isSome d.label d.after recs ξ hT (Verity.Sampled.releasedRootsOf_pairsOk hrr))
  have he : secondRead p e bs ξ = Verity.Sampled.readLog d.label d.after recs ξ lines := by
    simp [secondRead, hd, hr, hl, Except.toOption]
  exact ⟨out, he.trans hout⟩

/-! ## An accepted play's logs are read, or a break -/

/-- **On an accepted play the cut reads a log of both sessions, or the first session's record carries a break**,
where the first stage's coins are read under the parameters' key (`FirstReads`) and the device's history at the
play prints the play's log in its two sessions (`PrintsTwo`). -/
theorem reads_or_break (dev : Exchange → History) (θ : ByteArray) (hF : FirstReads θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → PrintsTwo (dev o) w₁ w₂ o.log)
    {o : Exchange} (hθ : o.params = θ) (ha : Accepts o.sent) :
    LogRead o ∨ ∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2 := by
  have ha' := ha
  obtain ⟨p, c, e, pub, prof, h1, hacc⟩ := ha'
  obtain ⟨key, d₁, ξ, hK, h3, h4⟩ := hF o.sent hθ ha p c e h1
  obtain ⟨-, c₂, ξ₂, r, i, _run, hin, -, -, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨⟨sj, hcanon, hcom⟩, h7, hrun, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have h5 : interiorOf e = .ok c₂ := by
    unfold interiorOf
    rw [hcanon]
    first
      | exact hcom
      | simpa [bind, Except.bind] using hcom
  obtain ⟨d₂, hd₂⟩ := oneStageZk_source h7
  have h6 : secondDraws p e = some d₂ := by
    unfold secondDraws
    rw [h5]
    first
      | exact hd₂
      | simpa [Except.toOption] using hd₂
  have h2 : keyOf o.sent.params = some key := by
    have h := hK
    rw [← hθ] at h
    exact h
  obtain ⟨-, -, -, hk₁⟩ := firstDraws_window h3
  have hm₁ : (d₁, firstCoins e) ∈ records o.sent := by rw [records_eq h1 h2, h3]; simp
  have hw : windowsOf θ = some (d₁.window, d₂.window) := by rw [← hθ]; exact windowsOf_eq h1 h3 h5 h6
  have hPT := hP o hθ d₁.window d₂.window hw
  have hP₁ : ∃ log₁, PrintsOn (dev o) d₁.window log₁ := by
    unfold PrintsTwo at hPT
    by_cases hww : d₁.window = d₂.window
    · rw [if_pos hww] at hPT
      exact ⟨_, hPT⟩
    · rw [if_neg hww] at hPT
      obtain ⟨log₁, log₂, -, hP₁, -⟩ := hPT
      exact ⟨log₁, hP₁⟩
  obtain ⟨log₁, hP₁⟩ := hP₁
  rcases first_reads h3 h4 hP₁ with ⟨r₁, h8⟩ | b
  · left
    obtain ⟨held, ts, L, hheld, hts, hM⟩ := members_trees hrun
    have hall := members_records hM
    have hrec : ∀ rf ∈ e.statements.toList.flatMap (·.sessions.toList),
        (Verity.Sampled.recordOn (_root_.Core.Blobs.ofArray o.sent.blobs) rf).isSome := by
      intro rf hrf
      obtain ⟨st, hst, hrf⟩ := List.mem_flatMap.1 hrf
      obtain ⟨m, hm⟩ := mem_zip_left e.statements.toList held.toList (by simp [held_size hheld]) hst
      have hzip : (st, m) ∈ (e.statements.zip held).toList := by
        first
          | (rw [Array.toList_zip]; exact hm)
          | simpa using hm
      exact hall (st, m) hzip rf hrf
    obtain ⟨r₂, h9⟩ := second_reads hd₂ h6 h7 hts hrec
    have hE : readsLogs o.sent = logsOf d₁.window d₂.window r₁ r₂ := by
      unfold readsLogs
      simp [h1, h2, h3, h4, h5, h6, h7, h8, h9, Except.toOption]
    unfold LogRead
    rw [hE]
    exact logsOf_ne_nil _ _ _ _
  · exact .inr ⟨(d₁, firstCoins e), hm₁, by rw [hK, ← hk₁], b⟩

/-! ## The cut with the read in its event -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The cut at an event `Good`**: where the sampled verdict accepts only where the form does (`Wired`), every
accepted play where `Good` fails carries a break in one of the two sessions' records under the parameters' key or an
opening collision in a read without the samples' coins, and the device's history at each play signed what the play's
signatures are on (`SignsPlay`), an accepted play where `Good` fails is at most as likely as the read finder's
collision plus the forger's forgery (`terms`). `cutBound` is this at `LogIsDevice`. -/
theorem cutBound_of (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Good : Exchange → Prop)
    (hw : Wired V k θ)
    (hb : ∀ o : Exchange, o.params = θ → Accepts o.sent → ¬ Good o →
      (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
        ∃ l ∈ opensReads o.sent, ∃ u v, Verity.Sampled.Lines.openCollision l o.log = some (u, v) ∧ u ≠ v ∧
          Flock.Sha512.hash u = Flock.Sha512.hash v)
    (hs : SignsPlay V dev θ F c) : CutBound V k θ F c σ Good (terms V dev θ F c σ) := by
  unfold CutBound terms
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) ∧
          ¬ Good (Verity.Game.Tape.out _ σ t))
      ≤ Verity.Game.prCoin (fun t =>
          (∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
            (forger V dev θ F c σ).Forges Verity.lmsVerify t) := by
        refine Verity.Game.prCoin_mono fun t ⟨hv, hl⟩ => ?_
        have hθ := (Verity.Sampled.Binding.play_reaches (Verity.Game.Tape.reaches_out _ σ t)).1
        rcases hb _ hθ (hw _ hθ hv) hl with ⟨x, hx, hk, hbr⟩ | ⟨l, hl', u, v, huv, hne, hh⟩
        · exact finders_of_break V dev θ F c σ t hs hx hk hbr
        · exact .inl (finder_of_open V dev θ F c σ t hl' huv ⟨hne, hh⟩)
    _ ≤ _ := Verity.Game.prCoin_or_le _ _
    _ = _ := congrArg₂ (· + ·) (Verity.Game.prCoin_eq_ofReal_pr _) (Verity.Game.prCoin_eq_ofReal_pr _)

/-- **The cut with the read in its event**: where the sampled verdict accepts only where the form does (`Wired`), the
first stage's coins are read under the parameters' key (`FirstReads`), the device's history at each play prints
the play's log in its two sessions (`PrintsTwo`) and signed what the play's signatures are on (`SignsPlay`), an
accepted play whose logs the check didn't read, or read other than the device's, is at most as likely as the read
finder's collision plus the forger's forgery (`terms`), over both sessions. -/
theorem cutBound_read (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hw : Wired V k θ) (hF : FirstReads θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → PrintsTwo (dev o) w₁ w₂ o.log)
    (hs : SignsPlay V dev θ F c) : CutBound V k θ F c σ ReadIsDevice (terms V dev θ F c σ) := by
  refine cutBound_of V k dev θ F c σ ReadIsDevice hw (fun o hθ ha hg => ?_) hs
  rcases reads_or_break dev θ hF hP hθ ha with hr | hb
  · exact binds_of_accepts dev θ hP o hθ ha fun hl => hg ⟨hr, hl⟩
  · exact .inl hb

end Verity.SampledTwo
