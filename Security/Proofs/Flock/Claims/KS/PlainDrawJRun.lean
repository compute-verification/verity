import Proofs.Flock.Claims.KS.PlainDrawJMove
import Proofs.Flock.Claims.Recursion.NextJ
import Proofs.Flock.Claims.Recursion.CallsBridgeE

/-!
# The plain verdict at the draw from an accepting `nextJ` step

`plainVerdictAtJK` from the recursion's last step at the record's parsed value (`FlockDeploy.nextJ`). Its decision
runs the plain verdict and custody's reading at the coin-substituted value `coinSubJ spec c₁ mids f.j`
(`decisionAtJ`), so the facts here are custody's at a parsed value (`liveOfJ`), not at a record text (`liveOf`):

* `liveOfJ_ok`, `checksAtJ_ok`, `checksAtJ_scope`: what `liveOfJ` reads of the parsed value and what `checksAtJ`
  establishes (`liveOf_ok`, `checks_ok`, `checks_scope` without the parse). `liveDJ`, `viewOfJ`: the step's live
  outcome at the parsed value (`liveD`, `viewOf`).
* `decodeJWithK_ok`, `decodeJWithK_live`: the element-level decoder at a parsed value reads the same `root_B`, link
  points and rep-stream rounds, up to their digests, as `liveOfJ` (`decodeWithK_ok`, `decodeWithK_live`).
* `nextJ_acc`, `decideJ_sha512`: an accepting `nextJ` step at SHA-512 is a statement that sets up plain and without
  `--public-inputs`, `verifyJWithMK id` accepting the substituted value and the proof files, and `checksAtJ` passing
  (`nextJ_decide`, `decisionAtJ`, through `decisionJF_eval`).
* `recordsLiveAtDJK_of_checksJ`: custody at the substituted value gives `RecordsLiveAtDJK` of `viewOfJ`.
* `plainVerdictAtJK_of_nextJ`: on an accepting `nextJ` step whose statement carries the draw `d`, where the deployed
  plain reader looks up no column off `C` in the step's own parsed session (`TxNoOffAtJK` at
  `⟨coinSubJ spec c₁ mids f.j, f.proofs⟩`), `plainVerdictAtJK id` holds at the step's live outcome and its link's root.
  The reader's premise moves to what `wrPDJK` writes by `txNoOffAtJK_of_records`.
-/

namespace Claims.Bridge.Custody

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge
open FlockSoundness.Refine (ebind_ok ptsOf)
open Lean (Json)

/-- `roundJ`'s fields: the record round's coin count, coins and retained bytes, with no digest. -/
theorem roundJ_fields {r : Json} {rd : Round} (h : FlockDeploy.roundJ r = .ok rd) :
    RoundFields r rd.n rd.coins rd.bytes ∧ rd.sha = ByteArray.empty := by
  unfold FlockDeploy.roundJ at h
  obtain ⟨n, hn, h⟩ := ebind_ok h
  obtain ⟨cs, hcs, h⟩ := ebind_ok h
  obtain ⟨coins, hcoins, h⟩ := ebind_ok h
  by_cases hnull : Record.isNull r "msg" = true
  · rw [ite_eq_left hnull] at h
    obtain ⟨b, hb, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at hb h
    subst hb h
    exact ⟨⟨hn, ⟨cs, hcs, hcoins⟩, fun _ => rfl, fun h' => by simp [hnull] at h'⟩, rfl⟩
  · rw [ite_eq_right hnull] at h
    obtain ⟨s', hs', h⟩ := ebind_ok h
    obtain ⟨b, hb, h⟩ := ebind_ok h
    cases hu : unhex s' with
    | error e => rw [hu] at hb; cases hb
    | ok u =>
      rw [hu] at hb
      simp only [Functor.map, Except.map, pure, Except.pure, Except.ok.injEq] at hb h
      subst hb h
      exact ⟨⟨hn, ⟨cs, hcs, hcoins⟩, fun h' => absurd h' hnull, fun _ => ⟨s', u, hs', hu, rfl⟩⟩, rfl⟩

/-- `streamJ`'s stream: the first of the name, and its rounds' fields. -/
theorem streamJ_ok {recs : Array Json} {name : String} {rds : Array Round}
    (h : FlockDeploy.streamJ recs name = .ok rds) :
    ∃ x, recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) = some x ∧
      ∃ rs, Record.arr x "rounds" = .ok rs ∧
        List.Forall₂ (fun r rd => RoundFields r rd.n rd.coins rd.bytes ∧ rd.sha = ByteArray.empty) rs.toList
          rds.toList := by
  unfold FlockDeploy.streamJ at h
  split at h
  · rename_i x hx
    obtain ⟨rs, hrs, h⟩ := ebind_ok h
    exact ⟨x, hx, rs, hrs, List.Forall₂.imp (fun _ _ h => roundJ_fields h) (mapM_forall₂ h)⟩
  · cases h

/-- **What `liveOfJ` reads** of the parsed value: its link, its streams, each table's `root_B`, the link points and
each rep stream's rounds, field by field (`liveOf_ok` without the parse). -/
theorem liveOfJ_ok {spec : SessionSpec} {j : Json} {lv : Live} (h : FlockDeploy.liveOfJ spec j = .ok lv) :
    ∃ l recs, Record.fld j "link" = .ok l ∧ Record.arr j "streams" = .ok recs ∧
      List.Forall₂ (RootField l) spec.tables.toList lv.rootB.toList ∧
      (∃ ps, Record.arr l "points" = .ok ps ∧ List.Forall₂ PointField ps.toList lv.points.toList) ∧
      List.Forall₂ (LiveStream recs) spec.repStreams.toList lv.streams.toList := by
  unfold FlockDeploy.liveOfJ at h
  obtain ⟨l, hl, h⟩ := ebind_ok h
  obtain ⟨rootB, hrootB, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨ps, hps, h⟩ := ebind_ok h
  obtain ⟨points, hpoints, h⟩ := ebind_ok h
  obtain ⟨recs, hrecs, h⟩ := ebind_ok h
  obtain ⟨streams, hstreams, h⟩ := ebind_ok h
  cases h
  refine ⟨l, recs, hl, hrecs, ?_, ⟨ps, hps, ?_⟩, ?_⟩
  · refine List.Forall₂.imp (fun t rb h => ?_) (mapM_forall₂ hrootB)
    beta_reduce at h
    obtain ⟨v, h1, h⟩ := ebind_ok h
    obtain ⟨s, h2, h3⟩ := ebind_ok h
    exact ⟨v, s, h1, h2, h3⟩
  · refine List.Forall₂.imp (fun p pt h => ?_) (mapM_forall₂ hpoints)
    beta_reduce at h
    split at h
    · rename_i a ha
      exact ⟨a, ha, h⟩
    · cases h
  · refine List.Forall₂.imp (fun w rds h => ?_) (mapM_forall₂ hstreams)
    obtain ⟨_, _, name⟩ := w
    exact streamJ_ok (recs := recs) (name := name) h

/-- **What `checksAtJ` establishes**: the last message's statement is `x`, the value `v`'s live session reads
(`liveOfJ`), and its custody passes (`checks_ok` at a parsed value). -/
theorem checksAtJ_ok {s : Ins} {spec : SessionSpec} {x : Claims.Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {v : Json}
    (h : checksAtJ s spec x link c₁ mids f v = .ok ()) :
    f.stmt = x ∧ ∃ lv, FlockDeploy.liveOfJ spec v = .ok lv ∧ custody spec lv link c₁ mids = .ok () := by
  unfold checksAtJ at h
  obtain ⟨_, h1, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨lv, h3, h⟩ := ebind_ok h
  obtain ⟨_, h4, -⟩ := ebind_ok h
  exact ⟨beq_iff_eq.1 (FlockSoundness.Refine.check_ok_iff.1 h1), lv, h3, h4⟩

/-- **`checksAtJ` runs the verifier's scope check** (`checks_scope` at a parsed value). -/
theorem checksAtJ_scope {s : Ins} {spec : SessionSpec} {x : Claims.Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {v : Json}
    (h : checksAtJ s spec x link c₁ mids f v = .ok ()) : s.read.check = .ok () := by
  unfold checksAtJ at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, h5, -⟩ := ebind_ok h
  exact h5

/-- The parsed value's live session (`liveOfJ`), empty where it does not read (`liveD` at a parsed value). -/
def liveDJ (spec : SessionSpec) (v : Json) : Live :=
  match FlockDeploy.liveOfJ spec v with
  | .ok lv => lv
  | .error _ => ⟨#[], #[], #[], #[]⟩

/-- **The live outcome of a `nextJ` decision** at the parsed value `v` (`viewOf` with `liveDJ spec v` for
`liveD spec f.record`): the link message's `root_B`, the first coin's two points, each rep stream's rounds as `v`
lists them, with their bytes' digests, and the proof files. -/
def viewOfJ (J mPts : ℕ) (spec : SessionSpec) (link : List ByteArray × List ByteArray) (c₁ : Coin) (v : Json)
    (proofs : Array ByteArray) : Tables.LiveOutJ J mPts where
  rootB := link.1.toArray
  pts := ptsOf mPts (pointsOf spec c₁)
  reps i r := ((liveDJ spec v).streams[2 * i.val + r.val]!).toList.map (withSha spec)
  proofs := proofs

end Claims.Bridge.Custody

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine
open Claims.Bridge.Custody (RootField PointField RoundFields)

/-- **What `Record.decodeJWithK` reads** of the parsed value, whatever the hashes and `keep` (`decodeWithK_ok` without
the parse). -/
theorem decodeJWithK_ok (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Lean.Json}
    {sess : Session} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∃ l recs, Record.fld j "link" = .ok l ∧ Record.arr j "streams" = .ok recs ∧
      List.Forall₂ (RootField l) spec.tables.toList sess.rootB.toList ∧
      (∃ ps, Record.arr l "points" = .ok ps ∧ List.Forall₂ PointField ps.toList sess.points.toList) ∧
      List.Forall₂ DecStreamK recs.toList sess.streams.toList := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, by assumption, by assumption, ?_, ⟨_, by assumption, ?_⟩, ?_⟩
    · refine List.Forall₂.imp (fun t rb h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
      beta_reduce at h
      obtain ⟨v, h1, h⟩ := ebind_ok h
      obtain ⟨s, h2, h⟩ := ebind_ok h
      obtain ⟨rb', h3, h⟩ := ebind_ok h
      repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨v, s, h1, h2, h3⟩
    · refine List.Forall₂.imp (fun p pt h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
      beta_reduce at h
      split at h
      · rename_i a ha
        exact ⟨a, ha, h⟩
      · cases h
    · refine List.Forall₂.imp (fun x y h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
      beta_reduce at h
      obtain ⟨name, hname, h⟩ := ebind_ok h
      obtain ⟨rs, hrs, h⟩ := ebind_ok h
      obtain ⟨rounds, hrounds, h⟩ := ebind_ok h
      repeat' obtain ⟨_, _, h⟩ := ebind_ok h
      cases h
      refine ⟨hname, rs, hrs, forall₂_of_mapIdxM (fun k r rd hr => ?_) hrounds⟩
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨n, hn, hr⟩ := ebind_ok hr
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨_, _, hr⟩ := ebind_ok hr
      obtain ⟨cs, hcs, hr⟩ := ebind_ok hr
      obtain ⟨coins, hcoins, hr⟩ := ebind_ok hr
      repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (simp only [] at hr))
      by_cases hb : Record.isNull r "msg" = true
      · rw [ite_eq_left hb] at hr
        obtain ⟨w, hw, hr⟩ := ebind_ok hr
        simp only [pure, Except.pure, Except.ok.injEq] at hw hr
        subst hw hr
        exact ⟨hn, ⟨cs, hcs, hcoins⟩, fun _ => rfl, fun h' => by rw [hb] at h'; cases h'⟩
      · rw [ite_eq_right hb] at hr
        obtain ⟨s', hs', hr⟩ := ebind_ok hr
        obtain ⟨b, hb', hr⟩ := ebind_ok hr
        obtain ⟨_, _, hr⟩ := ebind_ok hr
        obtain ⟨_, _, hr⟩ := ebind_ok hr
        obtain ⟨_, _, hr⟩ := ebind_ok hr
        obtain ⟨w, hw, hr⟩ := ebind_ok hr
        simp only [pure, Except.pure, Except.ok.injEq] at hw hr
        subst hw hr
        exact ⟨hn, ⟨cs, hcs, hcoins⟩, fun h' => absurd h' hb, fun _ => ⟨s', b, hs', hb', rfl⟩⟩

/-- **The element-level decoder's live part at a parsed value is `liveOfJ`'s**, whatever the hashes and `keep`
(`decodeWithK_live` without the parse): the same `root_B` and link points, and at each rep stream the stream the
decoder finds by its name, whose rounds are `liveOfJ`'s up to their recorded digests. -/
theorem decodeJWithK_live (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Lean.Json}
    {sess : Session} {lv : Live} (hd : Record.decodeJWithK keep hs spec j = .ok sess)
    (hl : FlockDeploy.liveOfJ spec j = .ok lv) :
    sess.rootB = lv.rootB ∧ sess.points = lv.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ q (hq : q < spec.repStreams.size), ∃ y, sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y ∧
        List.Forall₂ (fun rd rd' => rd.n = rd'.n ∧ rd.coins = rd'.coins ∧ rd.bytes = rd'.bytes) y.rounds.toList
          (lv.streams[q]!).toList := by
  obtain ⟨l, recs, hl1, hr, hroot, ⟨ps, hps, hpts⟩, hstr⟩ := decodeJWithK_ok keep hs hd
  obtain ⟨l', recs', hl1', hr', hroot', ⟨ps', hps', hpts'⟩, hstr'⟩ := Claims.Bridge.Custody.liveOfJ_ok hl
  rw [hl1] at hl1'
  cases hl1'
  rw [hr] at hr'
  cases hr'
  rw [hps] at hps'
  cases hps'
  refine ⟨Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := RootField l)
      (fun _ _ _ => Claims.Bridge.Custody.rootField_unique) hroot hroot'),
    Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := PointField)
      (fun _ _ _ => Claims.Bridge.Custody.pointField_unique) hpts hpts'),
    by simpa using hstr'.length_eq.symm,
    fun q hq => ?_⟩
  obtain ⟨x, hx, rs, hrs, hrds⟩ := Claims.Bridge.Custody.forall₂_getElem hstr' hq
  rw [← Array.find?_toList] at hx
  obtain ⟨y, ⟨-, rs', hrs', hdr⟩, hy⟩ := Claims.Bridge.Custody.forall₂_find? (q := (·.name == spec.repStreams[q].2.2))
    (fun a b hab => (Claims.Bridge.Custody.stream_pred hab.1).symm) hstr hx
  rw [hrs] at hrs'
  cases hrs'
  refine ⟨y, by rw [← Array.find?_toList]; exact hy,
    (Claims.Bridge.Custody.forall₂_left hdr hrds).imp fun rd rd' h => ?_⟩
  obtain ⟨r, h1, h3, -⟩ := h
  exact Claims.Bridge.Custody.roundFields_unique h1 h3

end FlockDeploy.KS

namespace Claims

open _root_.Flock

/-- **The verdict at a parsed value over `OC`, against `H`, is `verifyJWithM` with SHA-512 replaced by `hOf H`**
(`verifyOC_eval` at a parsed value). -/
theorem verifyJOC_eval (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (j : Lean.Json)
    (proofs : Array ByteArray) :
    ((verifyJWithM Hashes.oc spec tabs j proofs).run.eval H).1 =
      verifyJWithM (_root_.Flock.Hashes.ofSha512 (hOf H)) spec tabs j proofs := by
  rw [← (With.verifyJWithM_sim (ocLog H) (ocLog_hashes H) spec tabs j proofs #[]).1]
  exact With.verifyJWithM_sim (With.Sim.forget _) (With.Sim.forget_logged _) spec tabs j proofs #[]

end Claims

namespace FlockDeploy

open Flock

/-- The decision at a parsed value against `H`: the verdict with `hOf H` at the substituted value, then `checksAtJ`
there (`decisionAtK_eval` at a parsed value). -/
theorem decisionAtJ_eval (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : FinalJ) :
    ((decisionAtJ s spec sts x link c₁ mids f).run.eval H).1 =
      (do verifyJWithM (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs
          checksAtJ s spec x link c₁ mids f (coinSubJ spec c₁ mids f.j)) := by
  have he := Claims.verifyJOC_eval H spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs
  simp only [decisionAtJ, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs).run.eval H
    with ⟨r, l⟩
  rw [hv] at he
  simp only at he
  subst he
  cases verifyJWithM (M := Flock.V) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts) (coinSubJ spec c₁ mids f.j)
    f.proofs <;> rfl

/-- The decision at a parsed value against `H`: the setup with `hOf H`, then at its session `decisionAtJ_eval`'s
verdict and checks (`decisionK_eval` at a parsed value). -/
theorem decisionJF_eval (H : Claims.Msg → Claims.Msg) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalJ) :
    ((decisionJF s x link c₁ mids f).run.eval H).1 =
      (do let (spec, sts) ← setupWith (Claims.hOf H) s.read
          verifyJWithM (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs
          checksAtJ s spec x link c₁ mids f (coinSubJ spec c₁ mids f.j)) := by
  obtain ⟨he, -⟩ := setupM_eval H s.read
  simp only [decisionJF, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at he
  simp only at he
  subst he
  cases setupWith (Claims.hOf H) s.read with
  | error e => rfl
  | ok p =>
    obtain ⟨spec, sts⟩ := p
    exact decisionAtJ_eval H s spec sts x link c₁ mids f

/-- **`nextJ`'s decision against `H`**: it accepts exactly when the setup with `hOf H` is `.ok`, and at its session the
verdict with `hOf H` at the substituted value and `checksAtJ` there are `.ok` (`nextJ_decide`, `nextK_decide` at a
parsed value). -/
theorem nextJ_acc (H : Claims.Msg → Claims.Msg) {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ}
    (h : actJ rd x as cs = .decide s spec sts link c₁ mids f) :
    ((nextJ rd x as cs).eval H).1 = .acc ↔
      ∃ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        verifyJWithM (M := Flock.V) (Hashes.ofSha512 (Claims.hOf H)) spec' (tabsOf sts') (coinSubJ spec' c₁ mids f.j)
          f.proofs = .ok () ∧
        checksAtJ s spec' x link c₁ mids f (coinSubJ spec' c₁ mids f.j) = .ok () := by
  have he := decisionJF_eval H s x link c₁ mids f
  rw [nextJ_decide h]
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval]
  rw [he]
  cases hs : setupWith (Claims.hOf H) s.read with
  | error e => simp [bind, Except.bind]
  | ok p =>
    obtain ⟨spec', sts'⟩ := p
    cases hv : verifyJWithM (M := Flock.V) (Hashes.ofSha512 (Claims.hOf H)) spec' (tabsOf sts')
        (coinSubJ spec' c₁ mids f.j) f.proofs with
    | error e => simp [bind, Except.bind, hv]
    | ok u =>
      cases hc : checksAtJ s spec' x link c₁ mids f (coinSubJ spec' c₁ mids f.j) with
      | error e => simp [bind, Except.bind, hv, hc]
      | ok u' => simp [bind, Except.bind, hv, hc]

/-- `nextJ` decides only on a statement that sets up plain, at its setup (`act_decide`). -/
theorem actJ_decide {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalJ} (h : actJ rd x as cs = .decide s spec sts link c₁ mids f) :
    setupOf false x = some (s, spec, sts) := by
  unfold actJ at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

end FlockDeploy

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockDeploy.KS (decodeJWithK_live)
open Lean (Json)

/-- **An accepting `nextJ` step at SHA-512**: the statement sets up at `(spec, sts)`, plain and without
`--public-inputs`, `verifyJWithMK id` at SHA-512 accepts the substituted value and the proof files there, and
`checksAtJ` passes there (`nextJ_acc` at the setup `actJ` read; `decideK_sha512` at a parsed value). -/
theorem decideJ_sha512 {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalJ} (ha : actJ rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextJ rd x as cs).eval sha512).1 = .acc) :
    verifyJWithMK id With.std spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs = .ok () ∧
      checksAtJ s spec x link c₁ mids f (coinSubJ spec c₁ mids f.j) = .ok () ∧ s.setup = .ok (spec, sts) ∧
      s.read.zk = false ∧ s.read.publicInputs = none := by
  obtain ⟨spec', sts', hsu, hv, hc⟩ := (nextJ_acc sha512 ha).1 hacc
  obtain ⟨hins, hs⟩ := setupOf_some (actJ_decide ha)
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at hsu
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hsu)
  rw [hOf_sha512, verifyJWithM_eq_verifyJWithMK] at hv
  exact ⟨hv, hc, hs, readOf_zk (x := x) (by simp [readOf, hins]), publicInputs_none_of_check (checksAtJ_scope hc)⟩

theorem map_congr_of_forall₂ {α β γ : Type} {R : α → β → Prop} {f : α → γ} {g : β → γ} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → (∀ a b, b ∈ l' → R a b → f a = g b) → l.map f = l'.map g
  | _, _, .nil, _ => rfl
  | _, _, .cons hab hl, H => by
    rw [List.map_cons, List.map_cons, H _ _ (by simp) hab,
      map_congr_of_forall₂ hl fun a b hb h => H a b (List.mem_cons_of_mem _ hb) h]

/-- **Custody at a parsed value gives `RecordsLiveAtDJK`**: where `checksAtJ` passes at `v` on a plain statement that
sets up at its draw `d` with no `--public-inputs`, `v` and the proof files are a record of the step's live outcome
(`viewOfJ`) at the setup at `d` through every `keep`: the decoder's rounds are `liveOfJ`'s in coin count, coins and
bytes (`decodeJWithK_live`), and custody retains every live round's bytes, so they agree after `Round.erase`. -/
theorem recordsLiveAtDJK_of_checksJ (keep : Bool → Bool) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {x : Claims.Msg} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : FinalJ} {v : Json} {J mPts : ℕ} {d : Json} (hs : s.setup = .ok (spec, sts)) (hzk : s.read.zk = false)
    (hpi : s.read.publicInputs = none) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size)
    (hc : checksAtJ s spec x link c₁ mids f v = .ok ()) :
    RecordsLiveAtDJK keep s.read.I d (viewOfJ J mPts spec link c₁ v f.proofs) ⟨v, f.proofs⟩ := by
  intro _
  refine ⟨rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [setupAtD_plainInputs (Ins.setup_read hs) hzk hpi hd] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checksAtJ_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decodeJWithK_live keep With.std hdec hlv
  have hD : liveDJ spec v = lv := by simp [liveDJ, hlv]
  refine ⟨by simp only [viewOfJ]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList.map Round.erase =
      (((liveDJ spec v).streams[2 * j.val + r.val]!).toList.map (withSha spec)).map Round.erase
    rw [hD, List.map_map]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine map_congr_of_forall₂ hyr fun rd rd' hmem ⟨hn, hcn, hb⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hbytes
    subst hn hcn hb hbytes
    rfl

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **An accepting `nextJ` step gives the verdict at the draw**: at SHA-512, on a step whose statement carries the
draw `d`, if the deployed plain reader looks up no column off `C` in the step's own parsed session at the setup at `d`
(`TxNoOffAtJK id` at `⟨coinSubJ spec c₁ mids f.j, f.proofs⟩`), `plainVerdictAtJK id` holds at the step's live outcome
(`viewOfJ`, at any `mPts`) and its link's root. `verifyJWithMK id` accepts the step's session at the setup at `d`
(`decideJ_sha512`, `setupAtD_plainInputs`), which records the outcome (`recordsLiveAtDJK_of_checksJ`), so it accepts
what `wrPDJK` writes, and the reader's premise moves there (`txNoOffAtJK_of_records`). -/
theorem plainVerdictAtJK_of_nextJ (mPts : ℕ) {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {d : Json}
    {N : ℕ} (C : Fin N → List 𝔽) (ha : actJ rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextJ rd x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d)
    (hno : TxNoOffAtJK id s.read.I d C ⟨coinSubJ spec c₁ mids f.j, f.proofs⟩) :
    plainVerdictAtJK id s.read.I d C link.1.toArray
      (viewOfJ 1 mPts spec link c₁ (coinSubJ spec c₁ mids f.j) f.proofs) := by
  obtain ⟨hv, hc, hsu, hzk, hpi⟩ := decideJ_sha512 ha hacc
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verifyJWithMK_tables_cap id With.std hv
  have hJ : 1 ≤ spec.tables.size := by rw [← htsz]; exact hpos
  have hA₀ : AcceptsAtDJK id s.read.I (some d) ⟨coinSubJ spec c₁ mids f.j, f.proofs⟩ :=
    ⟨spec, sts, setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd, hv⟩
  have hR₀ := recordsLiveAtDJK_of_checksJ id (J := 1) (mPts := mPts) hsu hzk hpi hd hJ hc
  have hA := acceptsAtDJK_wrPDJK hA₀ hR₀
  exact ⟨hA, rfl, txNoOffAtJK_of_records hA₀ hR₀ hA (recordsLiveAtDJK_wrPDJK _ _ _ _) hno⟩

end FlockDeploy.KS
