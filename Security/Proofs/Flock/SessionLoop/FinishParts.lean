import Proofs.Flock.SessionLoop.Statement
import Proofs.Flock.Claims.ZK.FinExe
import Proofs.Flock.Claims.ZK.PrefixLive

/-!
# `FinishIs` at the loop's own `Finish`, in parts

At the deployed tables the session's last message is `finExe`, `finMsg` of `sessionCut` read from the tables' switched
views (`Masked.finExe_eq`). So `FinishIs` at the loop's release (`LeanRelease.ofLoopLink`, whose files are `Loop.filesE`)
asks that `sessionCut` on the views' source (`Loop.finSrc`), answered by the opened coins, return the loop's record and
proof files. `sessionCut` runs the schedules, level 0's commitments, the link, then each rep stream, then the record
(`FlockComplete.finSessionCut_eq`, `afterLinkCut`). Each of its parts is a named statement here, none a premise of a
guarantee, each at a `Finish` `FinishIs` reads (`Loop.FinCtx`):

* **(a) `Loop.FinCommits`**: the schedules return, level 0's commitments from the views' leaves send nothing and carry the
  roots the loop drew (`Loop.rootsOf`), and each rep's pads commitment sends nothing.
* **(b) `Loop.FinPoints`**: the link points at the opened coins are the ones the auditor saw (`Loop.pointsOf`), and each
  table's extra claims at them return.
* **(c) `Loop.FinRepProofs`**: rep stream `k`'s run from the views (`Loop.viewRep`) returns a proof whose file
  (`encZk` at `shapeZk`) is the loop's (`Loop.proofsOf`'s `k`-th, `Finish.proofFile`'s). The proof file's bytes are
  part of this statement: `proofFile` is `encZk` of its rep's proof at the same shape, so they follow from the two proofs
  being one.
* **(d) `Loop.FinRepRounds`**: where rep stream `k`'s run returns, its rounds are the ones the auditor saw
  (`Loop.streamOf`) and the history is at rep `k + 1`'s first round (`Masked.repBase`).

Each rep's run reads the streams before it as the auditor saw them (`FlockComplete.priorOf` of `Loop.streamOf`), which
(d) makes the session's. **`Loop.finishIs_of_parts`**: (a) to (d) give `FinishIs` at `ofLoopLink`.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- The session the loop's release is at: rep rounds of `loopNr` coins, at the deployed tables. -/
noncomputable abbrev finS : RefSession := CoinLayer.session Loop.loopNr Masked.dataE

/-- The release's tape: the session's nonce and tape. -/
abbrev Tp : Type := finS.refP.Nonce × finS.refP.Rand

/-- **A `Finish` `FinishIs` reads** at the release `L`: after `Hello` `(m₀, ρ₀)` and `refK x` openings `ops`, the run is
live, the openings pass, no coin is refused, the loop's `Finish` does not stop, and the history's messages are the
release's own (`ZkServer.OwnHist`). -/
def FinCtx (L : LeanRelease Flock.Wit Tp) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce)
    (t : finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) : Prop :=
  ops.length = refK x ∧ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true ∧
    ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true ∧
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false ∧ L.finStops x w (ν, t) ((m₀, ρ₀) :: ops) = false ∧
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops)

/-- The coins the openings `ops` open. -/
noncomputable def finCoins (x : Msg) (ops : List (Msg × Msg)) : Fin (refK x) → finS.Coin :=
  finS.pad (ops.map fun p => finS.decOf p.2)

/-- **The views' source**: what `sessionCut` reads from the tables' switched views at the opened coins, the workers'
values `vs` and the tape `t` (`Masked.tabSrcView`, as `finExe` reads them). -/
noncomputable def finSrc (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) : ℕ → FlockComplete.TabSrc :=
  Masked.tabSrcView Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
    (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)

/-- **Level 0's commitments from the views' leaves**, at the schedules `schs` (`sessionCut`'s). -/
noncomputable def commitsView (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) (schs : Array (_root_.Flock.Schedule × ℕ)) : FlockComplete.PS (Array FlockComplete.Commit) :=
  (Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
    FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (finSrc vs x w ν t ops i)

/-- **Rep stream `k`'s pads commitment from the views' leaves** (table `k / 2`'s rep `k % 2`), at the schedules `schs`. -/
noncomputable def padsView (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) (schs : Array (_root_.Flock.Schedule × ℕ)) (k : ℕ) :
    FlockComplete.PS FlockComplete.InnerCommit :=
  FlockComplete.padsTreeCut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
    (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
      2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ((finSrc vs x w ν t ops (k / 2)).rep (k % 2))

/-- **The link points at the opened coins**: `sessionCut`'s, from the link's coin (round `0`'s). -/
noncomputable def viewPoints (x : Msg) (ops : List (Msg × Msg)) (spec : SessionSpec) : Array (Array F128) :=
  (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j =>
    Masked.slotAt (finCoins x ops) 0 (p * spec.mPts + j)

/-- **Rep stream `k`'s run from the views** (`afterLinkCut`'s `repCut`): table `k / 2`'s rep `k % 2`, answered by the
opened coins, at the schedules `schs`, level 0's commitments `cms`, the extra claims `extras` and the pads commitment
`ic`, from the streams before it as the auditor saw them (`streamOf`). -/
noncomputable def viewRep (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (spec : SessionSpec) (schs : Array (_root_.Flock.Schedule × ℕ))
    (cms : Array FlockComplete.Commit) (k : ℕ) (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) :
    FlockComplete.PS (_root_.Flock.Zk.ZkProof × _root_.Flock.Transcript) :=
  (FlockComplete.repCut (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!
    schs[k / 2]!.1 (k % 2) (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic
    ((finSrc vs x w ν t ops (k / 2)).rep (k % 2))
    (FlockComplete.priorOf (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!
      ((Array.range k).map (streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x))) (k % 2))
    (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
    (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **(a) The commitments**: at a `Finish` `FinishIs` reads, the schedules return; level 0's commitments from the views'
leaves send nothing, and their roots are the ones the loop drew (`rootsOf`, as its `Commit` carries them); and each rep
stream's pads commitment from the views' leaves sends nothing. -/
def FinCommits : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
      ∃ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs ∧
        ∃ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) ∧
          cms.map (fun c => FlockComplete.flat c.cap) = rootsOf src x w (ν, t) (Masked.stsOf x) ∧
          ∀ k < 2 * spec.tables.size, ∀ l, ∃ ic, (padsView vs x w ν t ops schs k).go l = (.ok ic, l)

/-- **(b) The link points**: at a `Finish` `FinishIs` reads, the link points at the opened coins are the ones the auditor
saw (`pointsOf`, the `Commit`'s coins), and each table's extra claims at them return. -/
def FinPoints : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
      viewPoints x ops spec = pointsOf x ((m₀, ρ₀) :: ops) spec ∧
        ∀ k < 2 * spec.tables.size, ∃ extras,
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras

/-- **(c) The reps' proof files**: at a `Finish` `FinishIs` reads, at the schedules and level 0's commitments (a) gives,
where the loop's proof files are `files`, rep stream `k`'s run from the views, at extra claims and a pads commitment the
session computes and from the history at its first round, returns a proof whose file is the loop's `k`-th. -/
def FinRepProofs : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs →
    ∀ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) →
    ∀ files, proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) = .ok files →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
      ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p, tr), l') ∧
        FlockSoundness.Discharge.ZkView.encZk
          (FlockComplete.shapeZk (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 extras.size) p = files[k]!

/-- **(d) The reps' rounds**: at a `Finish` `FinishIs` reads, at the schedules and level 0's commitments (a) gives, where
rep stream `k`'s run from the views returns from the history at its first round, its rounds are the ones the auditor saw
(`streamOf`) and the history is at rep `k + 1`'s first round. -/
def FinRepRounds : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs →
    ∀ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
    ∀ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p, tr), l') →
      tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k ∧ l'.size = Masked.repBase x (k + 1)

/-- **A loop that accumulates**: where each body, from the state `B k` and a history the invariant `I k` holds at,
yields `B (k + 1)` at a history `I (k + 1)` holds at, the loop from `B 0` returns `B xs.length`. -/
theorem forIn_go_acc {α β : Type} (xs : List α) (f : α → β → FlockComplete.PS (ForInStep β)) (B : ℕ → β)
    (I : ℕ → Array Msg → Prop)
    (h : ∀ k (hk : k < xs.length) l, I k l →
      ∃ l', (f xs[k] (B k)).go l = (.ok (.yield (B (k + 1))), l') ∧ I (k + 1) l')
    (l : Array Msg) (h0 : I 0 l) :
    ∃ l', (forIn xs (B 0) f).go l = (.ok (B xs.length), l') ∧ I xs.length l' := by
  induction xs generalizing B I l with
  | nil => exact ⟨l, rfl, h0⟩
  | cons a as ih =>
    obtain ⟨l₁, e₁, h₁⟩ := h 0 (by simp) l h0
    have e₁' : (f a (B 0)).go l = (.ok (.yield (B 1)), l₁) := e₁
    rw [List.forIn_cons, FlockComplete.go_bind, e₁']
    exact ih (fun k => B (k + 1)) (fun k => I (k + 1)) (fun k hk l hI => h (k + 1) (by simpa using hk) l hI) l₁ h₁

/-- **A loop that accumulates, then the rest**: `forIn_go_acc`, and the rest from `B xs.length` at every history
`I xs.length` holds at. -/
theorem forIn_bind_acc {α β γ : Type} {xs : List α} {b : β} {f : α → β → FlockComplete.PS (ForInStep β)}
    {g : β → FlockComplete.PS γ} {l : Array Msg} {R : Except String γ} (B : ℕ → β) (I : ℕ → Array Msg → Prop)
    (h : ∀ k (hk : k < xs.length) l, I k l →
      ∃ l', (f xs[k] (B k)).go l = (.ok (.yield (B (k + 1))), l') ∧ I (k + 1) l')
    (hb : B 0 = b) (h0 : I 0 l) (hg : ∀ l', I xs.length l' → ((g (B xs.length)).go l').1 = R) :
    ((forIn xs b f >>= g).go l).1 = R := by
  subst hb
  obtain ⟨l', e, hI⟩ := forIn_go_acc xs f B I h l h0
  rw [FlockComplete.go_bind, e]
  exact hg l' hI

theorem map_range_succ {β : Type} (f : ℕ → β) (k : ℕ) :
    (Array.range (k + 1)).map f = ((Array.range k).map f).push (f k) := by
  simp [Array.range_succ]

/-- **(a) to (d) give `FinishIs` at the loop's own `Finish`.** Where the loop's `Finish` does not stop, `x` sets up and
the loop's files are `(rec, files)`; the session read from the views runs part by part (`finSessionCut_eq`): the
schedules and level 0's commitments (a), the link and its points (b), each rep stream (c, d), accumulating the streams
the auditor saw and the loop's proof files (`forIn_bind_acc`), then the same record (`recordJson_quietZ`). -/
theorem finishIs_of_parts
    (h : FinCommits vs L₀ src real wk x w ∧ FinPoints L₀ src real wk x w ∧ FinRepProofs vs L₀ src real wk x w ∧
      FinRepRounds vs L₀ src real wk x w) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  obtain ⟨hA, hB, hC, hD⟩ := h
  intro ν t κ m₀ ρ₀ ops hl hlive hok hr hs hown
  have hctx : FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops := ⟨hl, hlive, hok, hr, hs, hown⟩
  have hs' : (filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk = true := by
    change (!(filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk) = false at hs
    simpa using hs
  cases hfe : filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) with
  | error e => rw [hfe] at hs'; cases hs'
  | ok fp =>
  have hF : (LeanRelease.ofLoopLink L₀ src real).files x w (ν, t) ((m₀, ρ₀) :: ops) = fp := by
    change (filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).toOption.getD ("", #[]) = fp
    rw [hfe]
    rfl
  rw [hF]
  unfold filesE at hfe
  cases hsx0 : FlockDeploy.setupOf true x with
  | none => rw [hsx0] at hfe; cases hfe
  | some v =>
  obtain ⟨s0, spec, sts⟩ := v
  have hst : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hsx0]
  subst hst
  rw [hsx0] at hfe
  dsimp only at hfe
  cases hp : proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) with
  | error e => rw [hp] at hfe; cases hfe
  | ok files =>
  rw [hp] at hfe
  change (recordOf src x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) files >>= fun r =>
    (pure (r, files) : V _)) = _ at hfe
  cases hrc : recordOf src x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) files with
  | error e => rw [hrc] at hfe; cases hfe
  | ok rec =>
  rw [hrc] at hfe
  have hfp : fp = (rec, files) := (Except.ok.inj hfe).symm
  subst hfp
  obtain ⟨schs, hschs, cms, hcms, hroots, hpads⟩ := hA ν t κ m₀ ρ₀ ops hctx s0 spec hsx0
  obtain ⟨hpts, hext⟩ := hB ν t κ m₀ ρ₀ ops hctx s0 spec hsx0
  have hK := Masked.refK_eq hsx0
  have hrs : spec.repStreams.size = 2 * spec.tables.size := by
    have e := congrArg List.length (FlockDeploy.repStreams_toList spec)
    simpa using e
  have hfs : files.size = spec.repStreams.size := by
    unfold proofsOf at hp
    exact (FlockSoundness.Discharge.Exec.range_mapM_ok hp).1
  have hP : (Array.range (2 * spec.tables.size)).map (fun q => files[q]!) = files := by
    refine Array.ext (by simp [hfs, hrs]) fun i h1 h2 => ?_
    simp only [Array.getElem_map, Array.getElem_range]
    exact getElem!_pos files i h2
  have hrun : (Masked.runCut x (finCoins x ops) (refK x) spec (finSrc vs x w ν t ops)).1 = .ok (rec, files) := by
    have hcms' := hcms
    unfold commitsView at hcms'
    unfold Masked.runCut
    rw [FlockComplete.finSessionCut_eq, FlockComplete.go_bind, FlockComplete.schedsPZk_quiet, hschs]
    dsimp only
    rw [FlockComplete.go_bind, hcms']
    dsimp only
    rw [FlockComplete.go_bind, FlockComplete.finAsksCut_link_go,
      ite_eq_right_of_eq_false _ _ (eq_false (by rw [hK]; simp))]
    try dsimp only
    unfold FlockComplete.afterLinkCut
    try simp only [Array.size_empty]
    try dsimp only
    rw [← Array.forIn_toList, FlockDeploy.repStreams_toList]
    refine forIn_bind_acc
      (B := fun k => ((Array.range k).map (streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x)),
        (Array.range k).map fun q => files[q]!))
      (I := fun k l => l.size = Masked.repBase x k) ?_ (by simp) ?_ ?_
    · intro k hk l hI
      simp only [List.length_map, List.length_range] at hk
      have hI' : l.size = Masked.repBase x k := hI
      obtain ⟨extras, hxe⟩ := hext k hk
      obtain ⟨ic, hic⟩ := hpads k hk l
      obtain ⟨p, tr, l', hrep, hfile⟩ := hC ν t κ m₀ ρ₀ ops hctx s0 spec hsx0 schs hschs cms hcms files hp k hk
        extras hxe ic l hic hI'
      obtain ⟨hrounds, hsz⟩ := hD ν t κ m₀ ρ₀ ops hctx s0 spec hsx0 schs hschs cms hcms k hk extras hxe ic l hic hI'
        p tr l' hrep
      unfold viewPoints at hxe
      unfold padsView at hic
      unfold viewRep at hrep
      refine ⟨l', ?_, hsz⟩
      simp only [List.getElem_map, List.getElem_range]
      try dsimp only
      rw [FlockComplete.go_bind, FlockComplete.go_liftV, hxe]
      try dsimp only
      rw [FlockComplete.go_bind, hic]
      try dsimp only
      rw [FlockComplete.go_bind, hrep]
      try dsimp only
      rw [hrounds, hfile, map_range_succ, map_range_succ]
      rfl
    · show (_ : Array Msg).size = Masked.repBase x 0
      rw [Masked.repBase_eq hsx0]
      simp [FlockComplete.repStartZk]
    · intro l' _
      simp only [List.length_map, List.length_range]
      try dsimp only
      unfold viewPoints at hpts
      unfold recordOf at hrc
      rw [hrs] at hrc
      rw [FlockComplete.go_bind, FlockComplete.PrefixAgree.recordJson_quietZ, hroots, hpts, hP, hrc]
      rfl
  have e := Masked.finExe_eq hsx0 (finCoins x ops)
    (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)
  have e2 : FlockComplete.ZkTables.finMsg x
      (Masked.runCut x (finCoins x ops) (refK x) spec (finSrc vs x w ν t ops)).1 =
      FlockDeploy.Final.encode ⟨x, rec, files⟩ := by
    rw [hrun]
    rfl
  exact e2.symm.trans e.symm

end Loop

end Claims.ZKServer
