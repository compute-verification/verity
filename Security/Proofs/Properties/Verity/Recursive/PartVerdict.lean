import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Flock.Claims.Complete.Encode
import VStarRecursive

/-!
# A V\* part's compiled verdict is the model's decision at its session (piece E of `acc_of_accepts`)

The compiled recursive component runs, per V\* part `j`, `partVerdict`: part `j`'s setup at its record's draw
(`buildSessionOf` and the recursive setup's checks), then `Zk.verify` on the record and the rep files the part names
(`partFiles`), the scope check, the tables' sizes, the draw's law, the full-draw and inner-draw refusals, the program and
the record's canonical text. The model verifier (`RecDeploy.V`) decides session `s` with `RecDeploy.nextO` at the
session's V\* statement `xO`: at SHA-512 it accepts exactly when `Zk.verify` at `xO`'s recursive setup accepts the last
message's record and proof files, `FlockDeploy.checks` pass and so do the verdict's last refusals
(`RecDeploy.nextO_decide_sha512`).

`partVerdict_ok` unfolds the compiled verdict, and `decide_of_partVerdict` derives the model's three conditions from it.
What the compiled part can't see is a hypothesis, each named by the piece of the bridge's split that supplies it:

* **D** (`hD`): `xO`'s session is what `buildSessionOf` sets up at `xO`'s reading of the part's inputs.
* **B/C** (`hfind` … `hdraw`): `xO`'s reading (`s`, its `Ins`, and `innerOfO`, `srcOfO`) holds the compiled part's
  inputs: the statement's tags, the circuit and public file it reads off the blobs, the session's tags, one table, no
  lookup tables or public inputs, the registered reads, the verifier's inner shape and the inner session's record, and the
  record's draw.
* **`Shows`** (`hstmt`, `hrec`, `hreps`, `hnp`): the last message's statement is `xO`, its record is the part's record
  blob and its proof files are the part's rep files, by `repName`, in `spec.repStreams` order. `hnp` (the last message
  carries exactly one proof file per rep stream) is not in `Shows` yet: `Zk.verify` refuses any other count, so
  without it the model can refuse what the compiled part accepts.
* **`LiveCoins`** (`hlive`): the record's live session is the session's link message, first coin and round messages
  with their coins (`FlockDeploy.custody`).

`nextO_acc_of_partVerdict`, `decisionAt_of_partVerdict` and `next_of_partVerdict` restate the conclusion as the outer
verifier's acceptance, as `FlockDeploy.decisionAt`'s verdict at SHA-512, and as the model verifier's answer at the
session's last round.
-/

namespace Verity.Recursive

open Lean (Json)
open Flock (V SessionSpec Stmt Setup)
open FlockDeploy (Coin Final Ins sha512 tabsOf checks)
open RecDeploy (actO nextO setupO fullDrawO innerOfO srcOfO VO)
open VStar.Recursive (InnerHeld PartOwn PartProof STATEMENT registeredOf partVerdict)

/-- **Part `j`'s rep files** at the session `spec`: for each rep stream, the file the part names under the stream's
name, off the blobs (`partVerdict`'s `files`). -/
def partFiles (spec : SessionSpec) (s : PartProof) (bs : Core.Blobs) : V (Array ByteArray) :=
  spec.repStreams.mapM fun (_, _, w) => do
    let n := s!"{w.replace "/" "."}.bin"
    let some (_, d) := s.reps.find? (·.1 == n) | throw s!"session files: the session names no rep file {n}"
    (bs.get d s!"session files: {n}")

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => first | cases h | simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

private theorem ok_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {a : α} {b : β} (hx : x = .ok a)
    (hf : f a = .ok b) : (x >>= f) = .ok b := by
  rw [hx]
  exact hf

private theorem map_ok {ε α β : Type} {x : Except ε α} {f : α → β} {b : β} (h : Except.map f x = .ok b) :
    ∃ a, x = .ok a ∧ f a = b := by
  cases x with
  | error e => first | cases h | simp [Except.map] at h
  | ok a => exact ⟨a, rfl, Except.ok.inj h⟩

private theorem throw_ne {α : Type} {e : String} {a : α} : (throw e : V α) ≠ .ok a := by
  intro h
  first | cases h | simp [throw, throwThe, MonadExceptOf.throw] at h

/-- A blob's bytes don't depend on the name a refusal would give it. -/
theorem get_what {bs : Core.Blobs} {d w w' : String} {b : ByteArray} (h : bs.get d w = .ok b) :
    bs.get d w' = .ok b := by
  unfold Core.Blobs.get at h ⊢
  obtain ⟨a, hd, h⟩ := bind_ok h
  refine ok_bind hd ?_
  try dsimp only at h ⊢
  by_cases hc : (a.size != 64 || Flock.hexOf a != d) = true
  · rw [if_pos hc] at h
    first | cases h | simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · rw [if_neg hc] at h ⊢
    cases hf : bs.entries.find? (·.1 == a) with
    | none =>
      rw [hf] at h
      first | cases h | simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    | some e =>
      rw [hf] at h
      obtain ⟨e1, e2⟩ := e
      first | exact h | (simp only at h ⊢; exact h)

private theorem mapM_ok_of_getElem {α β ε : Type} (g : α → Except ε β) :
    ∀ (l : List α) (l' : List β), l.length = l'.length →
      (∀ i (hi : i < l.length) (hi' : i < l'.length), g l[i] = .ok l'[i]) → l.mapM g = .ok l'
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => absurd hl (by simp)
  | _ :: _, [], hl, _ => absurd hl (by simp)
  | a :: l, b :: l', hl, hg => by
    have h0 : g a = .ok b := hg 0 (by simp) (by simp)
    have ht : l.mapM g = .ok l' := mapM_ok_of_getElem g l l' (by simpa using hl)
      (fun i hi hi' => by
        have h := hg (i + 1) (by simp [hi]) (by simp [hi'])
        rwa [List.getElem_cons_succ, List.getElem_cons_succ] at h)
    simp [List.mapM_cons, h0, ht, bind, Except.bind, pure, Except.pure]

/-- **The part's rep files are the last message's proof files** where each stream's file is that proof file and there
are as many proof files as streams. -/
theorem partFiles_eq {spec : SessionSpec} {s : PartProof} {bs : Core.Blobs} {proofs : Array ByteArray}
    (hreps : ∀ k (hk : k < spec.repStreams.size), ∃ d,
      (s.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < proofs.size, bs.get d "session files" = .ok proofs[k])
    (hnp : proofs.size = spec.repStreams.size) :
    partFiles spec s bs = .ok proofs := by
  unfold partFiles
  rw [Array.mapM_eq_mapM_toList]
  rw [mapM_ok_of_getElem _ spec.repStreams.toList proofs.toList (by simp [hnp]) (fun k hk hk' => ?_)]
  · simp [Functor.map, Except.map]
  · have hk2 : k < spec.repStreams.size := by simpa using hk
    obtain ⟨d, hd, hk3, hget⟩ := hreps k hk2
    simp only [Array.getElem_toList]
    rcases hx : spec.repStreams[k] with ⟨a, b, w⟩
    rw [hx] at hd
    obtain ⟨⟨n', d'⟩, hfind, hdd⟩ := Option.map_eq_some_iff.1 hd
    simp only at hdd
    subst hdd
    simp only [hfind]
    exact get_what hget

private theorem fullDraw_some (i : Flock.RecVStar.Inner) (b : Bool) (d : Option Json) (r : String) :
    Flock.RecSetup.fullDraw (some i) b d r = Flock.RecSetup.fullDraw (some i) true d r := by
  cases b <;> first | rfl | simp [Flock.RecSetup.fullDraw]

private theorem innerDrawn_record {s s' : Flock.RecCoins.Source} (h : s.record = s'.record) :
    Flock.RecSetup.innerDrawn (some s) = Flock.RecSetup.innerDrawn (some s') := by
  simp only [Flock.RecSetup.innerDrawn, h]

private theorem recordDraw_eq (r : String) :
    FlockSoundness.Discharge.Exec.recordDraw r = Flock.Witness.recordDraw r := rfl

/-- **The compiled part's verdict, unfolded**: where part `j`'s verdict is `.ok ()`, it read the statement's tags, the
circuit, public file and registered reads, and a UTF-8 record with a draw off the blobs; `buildSessionOf` at the part's
tags, coins and the record's draw set up a session `(spec, sts)`; the part's rep files are `.ok`; and `Zk.verify` at that
session, the scope check at the session's tags, the full-draw and inner-draw refusals and the record's canonical text
pass. -/
theorem partVerdict_ok {h : InnerHeld} {j : Nat} {p : PartOwn} {s : PartProof} {bs : Core.Blobs}
    (hpart : partVerdict h j p s bs = .ok (.ok ())) :
    ∃ (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (registered : Option (Array Flock.Registered.Port))
      (rB : ByteArray) (record : String) (spec : SessionSpec) (sts : Array Stmt) (proofs : Array ByteArray),
      Flock.Tags.find? STATEMENT = some tags0 ∧
      bs.get p.circuit s!"part {j}'s circuit" = .ok circuitB ∧
      bs.get s.pub s!"part {j}'s public file" = .ok pubB ∧
      p.registered.bind (fun r => (registeredOf r).toOption.map (·.1)) = registered ∧
      bs.get s.record s!"part {j}'s record" = .ok rB ∧ String.fromUTF8? rB = some record ∧
      (Flock.Witness.recordDraw record).isSome = true ∧
      SampledProofs.OneStageZk.buildSessionOf
          (Flock.ProvedScope.circuitOf
            (Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
              (Flock.Zk.maskLogOf circuitB) 1) circuitB [])
          (Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
            (Flock.Zk.maskLogOf circuitB) 1)
          circuitB pubB (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) [] 1
          (Flock.Witness.recordDraw record) (some p.partition) (some p.program) registered = .ok (spec, sts) ∧
      partFiles spec s bs = .ok proofs ∧
      Flock.Zk.verify spec (tabsOf sts) record proofs = .ok () ∧
      Flock.ProvedScope.check
          (Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
            (Flock.Zk.maskLogOf circuitB) 1)
          true circuitB pubB [] 1 false registered.isSome = .ok () ∧
      Flock.RecSetup.fullDraw (some h.shape) true (Flock.Witness.recordDraw record) record = .ok () ∧
      Flock.RecSetup.innerDrawn (some h.src) = .ok () ∧
      Flock.RecordText.check record = .ok () := by
  unfold partVerdict at hpart
  dsimp only at hpart
  rcases hT : Flock.Tags.find? STATEMENT with _ | tags0
  · simp only [hT] at hpart
    try exact absurd hpart throw_ne
  simp only [hT] at hpart
  obtain ⟨_, -, hpart⟩ := bind_ok hpart
  obtain ⟨circuitB, hc, hpart⟩ := bind_ok hpart
  obtain ⟨pubB, hp, hpart⟩ := bind_ok hpart
  obtain ⟨_, -, hpart⟩ := bind_ok hpart
  rcases hpr : p.registered with _ | rj <;> simp only [hpr] at hpart
  all_goals obtain ⟨⟨registered, complete, known, regParams⟩, hr, hpart⟩ := bind_ok hpart
  all_goals
    have hreg : p.registered.bind (fun r => (registeredOf r).toOption.map (·.1)) = registered := by
      first
        | (rw [hpr]
           have hr := Except.ok.inj hr
           simp only [Prod.mk.injEq] at hr
           exact hr.1)
        | (obtain ⟨⟨own, c, k, ps⟩, hro, hr⟩ := map_ok hr
           simp only [Prod.mk.injEq] at hr
           rw [hpr, ← hr.1]
           change (registeredOf rj).toOption.map (·.1) = some own
           rw [hro]
           all_goals rfl)
    rw [hpr] at hreg
    clear hr
  all_goals
    try dsimp only at hpart
    obtain ⟨bindings, -, hpart⟩ := bind_ok hpart
    obtain ⟨rB, hrB, hpart⟩ := bind_ok hpart
    rcases hu : String.fromUTF8? rB with _ | record
    · simp only [hu] at hpart
      try exact absurd hpart throw_ne
    simp only [hu] at hpart
    generalize hT' : Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
      (Flock.Zk.maskLogOf circuitB) 1 = T at hpart
    generalize hD : (if T.hm96Rows = true then Flock.Witness.recordDraw record else none) = D at hpart
    have hv := Except.ok.inj hpart
    clear hpart
    obtain ⟨proofs, hfiles, hv⟩ := bind_ok hv
    obtain ⟨⟨spec, tabs⟩, hst, hv⟩ := bind_ok hv
    obtain ⟨⟨spec2, tabs2⟩, hst2, hf⟩ := bind_ok hfiles
    rw [hst] at hst2
    have hst2 := Except.ok.inj hst2
    simp only [Prod.mk.injEq] at hst2
    obtain ⟨rfl, rfl⟩ := hst2
    try dsimp only at hv hf
    obtain ⟨⟨⟩, hzv, hv⟩ := bind_ok hv
    obtain ⟨⟨⟩, hsc, hv⟩ := bind_ok hv
    obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
    rcases D with _ | d
    · simp only [Option.isNone_none, ↓reduceIte] at hv
      obtain ⟨⟨⟩, hdl, -⟩ := bind_ok hv
      first | cases hdl | simp [Flock.ProvedScope.drawLaw] at hdl
    simp only [Option.isNone_some, Bool.false_eq_true, ↓reduceIte] at hv
    obtain ⟨⟨⟩, hfd, hv⟩ := bind_ok hv
    obtain ⟨⟨⟩, hid, hv⟩ := bind_ok hv
    obtain ⟨⟨⟩, -, hrt⟩ := bind_ok hv
    have hm : T.hm96Rows = true := by
      cases hh : T.hm96Rows
      · simp [hh] at hD
      · rfl
    rw [if_pos hm] at hD
    have hds : (Flock.Witness.recordDraw record).isSome = true := by
      simp [hD]
    split at hst
    rotate_left
    · cases hst
    rename_i spec' sts' hsu
    obtain ⟨_, -, hsu⟩ := bind_ok hsu
    obtain ⟨_, -, hsu⟩ := bind_ok hsu
    obtain ⟨_, -, hsu⟩ := bind_ok hsu
    obtain ⟨_, -, hsu⟩ := bind_ok hsu
    obtain ⟨r, hbs, hsu⟩ := bind_ok hsu
    have hr' : r = (spec', sts') := by
      try dsimp only at hsu
      split at hsu <;> (repeat (obtain ⟨_, -, hsu⟩ := bind_ok hsu)) <;> exact Except.ok.inj hsu
    subst hr'
    obtain ⟨_, -, hpair⟩ := map_ok hst
    simp only [Prod.mk.injEq] at hpair
    obtain ⟨rfl, rfl⟩ := hpair
    rw [← hD] at hbs hfd
    subst hT'
    exact ⟨tags0, circuitB, pubB, registered, rB, record, _, _, proofs, rfl, hc, hp, hreg, hrB, hu, hds, hbs,
      hf, hzv, hsc, hfd, hid, hrt⟩

/-- **Piece E: a V\* part's compiled verdict is the model's decision at its session.** Where part `j`'s compiled
verdict is `.ok ()` and the model decides at `xO` on the last message `f` with `xO`'s reading `s` and session
`(spec, sts)`, the deployed `Zk.verify` at that session accepts `f`'s record and proof files, the custody checks pass and
so do the verdict's last refusals, given:
* D: `(spec, sts)` is what `buildSessionOf` sets up at `s`'s reading of the part's inputs (`hD`);
* B/C: `s`, `innerOfO xO` and `srcOfO xO` hold the compiled part's inputs (`hfind` … `hsrc`), and `s`'s draw is the
  record's (`hdraw`);
* `Shows`: `f` is the part's last message at `xO` (`hstmt`), its record the part's record blob (`hrec`) and its proof
  files the part's rep files (`hreps`), one per rep stream (`hnp`);
* `LiveCoins`: the record's live session is the session's messages and coins (`hlive`). -/
theorem decide_of_partVerdict {h : InnerHeld} {j : Nat} {p : PartOwn} {q : PartProof} {bs : Core.Blobs}
    (hpart : partVerdict h j p q bs = .ok (.ok ()))
    {xO : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hfind : Flock.Tags.find? STATEMENT = some s.stmtTags)
    (hcirc : bs.get p.circuit s!"part {j}'s circuit" = .ok s.read.I.circuitFile)
    (hpub : bs.get q.pub "a part's public file" = .ok s.read.I.publicFile)
    (htags : s.read.tags = Flock.Zk.tags (s.stmtTags.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec
      (s.stmtTags.tableNames 1)))) (Flock.Zk.maskLogOf s.read.I.circuitFile) 1)
    (hzk : s.read.zk = true) (htables : s.read.I.tables = []) (hcount : s.read.I.count = 1)
    (hpi : s.read.publicInputs = none)
    (hreg : p.registered.bind (fun r => (registeredOf r).toOption.map (·.1)) = s.registered)
    (hreadReg : s.read.registered = s.registered.isSome)
    (hinner : innerOfO xO = some h.shape)
    (hsrc : (srcOfO xO).map (·.record) = some h.src.record)
    (hdraw : s.read.draw = Flock.Witness.recordDraw f.record)
    (hD : SampledProofs.OneStageZk.buildSessionOf
        (Flock.ProvedScope.circuitOf s.read.tags s.read.I.circuitFile []) s.read.tags s.read.I.circuitFile
        s.read.I.publicFile (Flock.Coins.tree (Flock.Zk.coinSpec (s.stmtTags.tableNames 1))) [] 1 s.read.draw
        (some p.partition) (some p.program) s.registered = .ok (spec, sts))
    (hstmt : f.stmt = xO)
    (hrec : bs.get q.record "a part's record" = .ok f.record.toUTF8)
    (hreps : ∀ k (hk : k < spec.repStreams.size), ∃ d,
      (q.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k])
    (hnp : f.proofs.size = spec.repStreams.size)
    (hlive : (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()) :
    Flock.Zk.verify spec (tabsOf sts) f.record f.proofs = .ok () ∧ checks true s spec xO link c₁ mids f = .ok () ∧
      fullDrawO xO f = true := by
  obtain ⟨tags0, circuitB, pubB, registered, rB, record, spec', sts', proofs, hT, hc, hp, hr, hrB, hu, hds, hbs, hpf,
    hzv, hsc, hfd, hid, hrt⟩ := partVerdict_ok hpart
  rw [hfind] at hT
  cases hT
  rw [hcirc] at hc
  cases hc
  rw [get_what hpub] at hp
  cases hp
  rw [hreg] at hr
  subst hr
  rw [get_what hrec] at hrB
  cases hrB
  rw [FlockComplete.fromUTF8?_toUTF8] at hu
  cases hu
  rw [htags, hdraw] at hD
  rw [hD] at hbs
  cases hbs
  rw [partFiles_eq hreps hnp] at hpf
  cases hpf
  refine ⟨hzv, ?_, ?_⟩
  · obtain ⟨lv, hlv, hcu⟩ := bind_ok hlive
    have hcheck : s.read.check = .ok () := by
      unfold Flock.Readable.Read.check
      rw [htags, hzk, htables, hcount, hpi, hreadReg]
      exact hsc
    have hdl : (if s.read.draw.isNone then Flock.ProvedScope.drawLaw true s.read.tags false none else pure ()) =
        (.ok () : V Unit) := by
      obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 hds
      rw [hdraw, hd]
      all_goals rfl
    unfold FlockDeploy.checks
    refine ok_bind (a := ()) ?_ ?_
    · simp [Flock.Record.check, hstmt]
      all_goals rfl
    refine ok_bind (a := ()) ?_ ?_
    · rw [recordDraw_eq, hdraw, FlockDeploy.drawEq_refl]
      all_goals rfl
    refine ok_bind hlv ?_
    refine ok_bind hcu ?_
    refine ok_bind hcheck ?_
    refine ok_bind hdl ?_
    exact hrt
  · obtain ⟨src, hsrcEq, hrecEq⟩ := Option.map_eq_some_iff.1 hsrc
    unfold fullDrawO
    rw [hinner, hsrcEq, fullDraw_some, hfd, innerDrawn_record hrecEq, hid]
    all_goals rfl

/-- **The outer verifier accepts the session** where the compiled part does, under the hypotheses of
`decide_of_partVerdict` (`RecDeploy.nextO_decide_sha512`). -/
theorem nextO_acc_of_partVerdict {h : InnerHeld} {j : Nat} {p : PartOwn} {q : PartProof} {bs : Core.Blobs}
    (hpart : partVerdict h j p q bs = .ok (.ok ()))
    {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (ha : actO xO as' cs' = .decide s spec sts link c₁ mids f)
    (hfind : Flock.Tags.find? STATEMENT = some s.stmtTags)
    (hcirc : bs.get p.circuit s!"part {j}'s circuit" = .ok s.read.I.circuitFile)
    (hpub : bs.get q.pub "a part's public file" = .ok s.read.I.publicFile)
    (htags : s.read.tags = Flock.Zk.tags (s.stmtTags.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec
      (s.stmtTags.tableNames 1)))) (Flock.Zk.maskLogOf s.read.I.circuitFile) 1)
    (hzk : s.read.zk = true) (htables : s.read.I.tables = []) (hcount : s.read.I.count = 1)
    (hpi : s.read.publicInputs = none)
    (hreg : p.registered.bind (fun r => (registeredOf r).toOption.map (·.1)) = s.registered)
    (hreadReg : s.read.registered = s.registered.isSome)
    (hinner : innerOfO xO = some h.shape)
    (hsrc : (srcOfO xO).map (·.record) = some h.src.record)
    (hdraw : s.read.draw = Flock.Witness.recordDraw f.record)
    (hD : SampledProofs.OneStageZk.buildSessionOf
        (Flock.ProvedScope.circuitOf s.read.tags s.read.I.circuitFile []) s.read.tags s.read.I.circuitFile
        s.read.I.publicFile (Flock.Coins.tree (Flock.Zk.coinSpec (s.stmtTags.tableNames 1))) [] 1 s.read.draw
        (some p.partition) (some p.program) s.registered = .ok (spec, sts))
    (hstmt : f.stmt = xO)
    (hrec : bs.get q.record "a part's record" = .ok f.record.toUTF8)
    (hreps : ∀ k (hk : k < spec.repStreams.size), ∃ d,
      (q.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k])
    (hnp : f.proofs.size = spec.repStreams.size)
    (hlive : (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()) :
    ((nextO xO as' cs').eval sha512).1 = .acc :=
  (RecDeploy.nextO_decide_sha512 ha).2
    (decide_of_partVerdict hpart hfind hcirc hpub htags hzk htables hcount hpi hreg hreadReg hinner hsrc hdraw hD hstmt
      hrec hreps hnp hlive)

/-- **`FlockDeploy.decisionAt`'s verdict at SHA-512 is `.ok ()`** at the session where the compiled part's is, under the
hypotheses of `decide_of_partVerdict` (`FlockDeploy.decisionAt_eval`, `RecDeploy.verdictWithO_sha512`). -/
theorem decisionAt_of_partVerdict {h : InnerHeld} {j : Nat} {p : PartOwn} {q : PartProof} {bs : Core.Blobs}
    (hpart : partVerdict h j p q bs = .ok (.ok ()))
    {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (ha : actO xO as' cs' = .decide s spec sts link c₁ mids f)
    (hfind : Flock.Tags.find? STATEMENT = some s.stmtTags)
    (hcirc : bs.get p.circuit s!"part {j}'s circuit" = .ok s.read.I.circuitFile)
    (hpub : bs.get q.pub "a part's public file" = .ok s.read.I.publicFile)
    (htags : s.read.tags = Flock.Zk.tags (s.stmtTags.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec
      (s.stmtTags.tableNames 1)))) (Flock.Zk.maskLogOf s.read.I.circuitFile) 1)
    (hzk : s.read.zk = true) (htables : s.read.I.tables = []) (hcount : s.read.I.count = 1)
    (hpi : s.read.publicInputs = none)
    (hreg : p.registered.bind (fun r => (registeredOf r).toOption.map (·.1)) = s.registered)
    (hreadReg : s.read.registered = s.registered.isSome)
    (hinner : innerOfO xO = some h.shape)
    (hsrc : (srcOfO xO).map (·.record) = some h.src.record)
    (hdraw : s.read.draw = Flock.Witness.recordDraw f.record)
    (hD : SampledProofs.OneStageZk.buildSessionOf
        (Flock.ProvedScope.circuitOf s.read.tags s.read.I.circuitFile []) s.read.tags s.read.I.circuitFile
        s.read.I.publicFile (Flock.Coins.tree (Flock.Zk.coinSpec (s.stmtTags.tableNames 1))) [] 1 s.read.draw
        (some p.partition) (some p.program) s.registered = .ok (spec, sts))
    (hstmt : f.stmt = xO)
    (hrec : bs.get q.record "a part's record" = .ok f.record.toUTF8)
    (hreps : ∀ k (hk : k < spec.repStreams.size), ∃ d,
      (q.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k])
    (hnp : f.proofs.size = spec.repStreams.size)
    (hlive : (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()) :
    ((FlockDeploy.decisionAt true s spec sts xO link c₁ mids f).run.eval sha512).1 = .ok () := by
  obtain ⟨hv, hc, -⟩ := decide_of_partVerdict hpart hfind hcirc hpub htags hzk htables hcount hpi hreg hreadReg hinner
    hsrc hdraw hD hstmt hrec hreps hnp hlive
  obtain ⟨i, hi, -⟩ := RecDeploy.actO_decide ha
  rw [(FlockDeploy.decisionAt_eval sha512 true s spec sts xO link c₁ mids f).1, RecDeploy.verdictWithO_sha512 hi, hv]
  all_goals exact hc

/-- **The model verifier's answer at the session's last round** is the compiled part's: `.acc` at the last session,
`.msg []` before it, where the session is the part's staged one (`ract`), this is its last round, and the hypotheses of
`decide_of_partVerdict` hold (`RecDeploy.next_decide_sha512`). -/
theorem next_of_partVerdict {IV : Claims.Verifier Coin} {st : RecDeploy.Stage} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {i j' : ℕ} {h : InnerHeld} {j : Nat} {p : PartOwn} {q : PartProof} {bs : Core.Blobs}
    (hpart : partVerdict h j p q bs = .ok (.ok ()))
    {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hs : RecDeploy.ract IV st x as cs = .sess i j' xO as' cs') (hlast : j' + 1 = VO.k xO)
    (ha : actO xO as' cs' = .decide s spec sts link c₁ mids f)
    (hfind : Flock.Tags.find? STATEMENT = some s.stmtTags)
    (hcirc : bs.get p.circuit s!"part {j}'s circuit" = .ok s.read.I.circuitFile)
    (hpub : bs.get q.pub "a part's public file" = .ok s.read.I.publicFile)
    (htags : s.read.tags = Flock.Zk.tags (s.stmtTags.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec
      (s.stmtTags.tableNames 1)))) (Flock.Zk.maskLogOf s.read.I.circuitFile) 1)
    (hzk : s.read.zk = true) (htables : s.read.I.tables = []) (hcount : s.read.I.count = 1)
    (hpi : s.read.publicInputs = none)
    (hreg : p.registered.bind (fun r => (registeredOf r).toOption.map (·.1)) = s.registered)
    (hreadReg : s.read.registered = s.registered.isSome)
    (hinner : innerOfO xO = some h.shape)
    (hsrc : (srcOfO xO).map (·.record) = some h.src.record)
    (hdraw : s.read.draw = Flock.Witness.recordDraw f.record)
    (hD : SampledProofs.OneStageZk.buildSessionOf
        (Flock.ProvedScope.circuitOf s.read.tags s.read.I.circuitFile []) s.read.tags s.read.I.circuitFile
        s.read.I.publicFile (Flock.Coins.tree (Flock.Zk.coinSpec (s.stmtTags.tableNames 1))) [] 1 s.read.draw
        (some p.partition) (some p.program) s.registered = .ok (spec, sts))
    (hstmt : f.stmt = xO)
    (hrec : bs.get q.record "a part's record" = .ok f.record.toUTF8)
    (hreps : ∀ k (hk : k < spec.repStreams.size), ∃ d,
      (q.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k])
    (hnp : f.proofs.size = spec.repStreams.size)
    (hlive : (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()) :
    ((RecDeploy.next IV st x as cs).eval sha512).1 = (if i + 1 = RecDeploy.nS x then .acc else .msg []) :=
  (RecDeploy.next_decide_sha512 hs ha).2
    ⟨hlast, decide_of_partVerdict hpart hfind hcirc hpub htags hzk htables hcount hpi hreg hreadReg hinner hsrc hdraw hD
      hstmt hrec hreps hnp hlive⟩

end Verity.Recursive
