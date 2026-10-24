import Proofs.Flock.Claims.Recursion.CallsBridge
import Proofs.Flock.Claims.Recursion.ConflictTiedE
import Proofs.Flock.Claims.Recursion.RepErase

/-!
# The calls half where a proof file differs, at the element-level verifier

`plain_calls_tied` at `VE`. Where a proof file differs, custody pins both records' rounds to the same round messages
at the counts the statement fixes, so the two sessions' rounds on every rep stream agree as replay reads them
(`sess_rounds_eqK`: the element-level decoder drops S12's digest of a retained round's bytes, so the rounds' recorded
digests may differ, `Round.erase`). A rep reads its rounds only as replay does (`Flock.With.verifyRepWith_erase`), so
the two files are accepted as one rep on one session, and `rep_bind_proof` gives one proof or two openings that
collide among the calls.

Here first: what the element-level decoder reads of the record (`decodeWithK_ok`, `decode_ok`'s without the digests),
and the plain verdict's round counts at any `keep` (`streams_ns_plainK`, `rounds_ns_plainK`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine
open Claims.Bridge.Custody (RootField PointField RoundFields)

/-! ## What the element-level decoder reads -/

/-- A stream as `Record.decodeWithK` reads it, whatever `keep`: its name, and its rounds' fields. -/
def DecStreamK (x : Lean.Json) (y : StreamRec) : Prop :=
  Record.str x "stream" = .ok y.name ∧ ∃ rs, Record.arr x "rounds" = .ok rs ∧
    List.Forall₂ (fun r rd => RoundFields r rd.n rd.coins rd.bytes) rs.toList y.rounds.toList

/-- **What `Record.decodeWithK` reads**, whatever the hashes and `keep`: `decode_ok`'s fields, without S12's digests. -/
theorem decodeWithK_ok (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : Session} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∃ j l recs, Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr j "streams" = .ok recs ∧
      List.Forall₂ (RootField l) spec.tables.toList sess.rootB.toList ∧
      (∃ ps, Record.arr l "points" = .ok ps ∧ List.Forall₂ PointField ps.toList sess.points.toList) ∧
      List.Forall₂ DecStreamK recs.toList sess.streams.toList := by
  unfold Record.decodeWithK at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, _, by assumption, by assumption, by assumption, ?_, ⟨_, by assumption, ?_⟩, ?_⟩
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
      repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (split at hr))
      all_goals
        cases hr
        refine ⟨hn, ⟨cs, hcs, hcoins⟩, fun hnull => ?_, fun hnull => ?_⟩
        all_goals first
          | rfl
          | exact ⟨_, _, by assumption, by assumption, rfl⟩
          | exact absurd hnull (by assumption)
          | (rw [‹Record.isNull r "msg" = true›] at hnull; cases hnull)

/-- **The element-level decoder's live part is `liveOf`'s**, whatever the hashes and `keep`: the same `root_B` and
link points, and at each rep stream the stream the decoder finds by its name, whose rounds are `liveOf`'s up to their
recorded digests (`decode_live`'s, without S12's digests). -/
theorem decodeWithK_live (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : Session} {lv : Live} (hd : Record.decodeWithK keep hs spec text = .ok sess)
    (hl : liveOf spec text = .ok lv) :
    sess.rootB = lv.rootB ∧ sess.points = lv.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ q (hq : q < spec.repStreams.size), ∃ y, sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y ∧
        List.Forall₂ (fun rd rd' => rd.n = rd'.n ∧ rd.coins = rd'.coins ∧ rd.bytes = rd'.bytes) y.rounds.toList
          (lv.streams[q]!).toList := by
  obtain ⟨j, l, recs, hp, hl1, hr, hroot, ⟨ps, hps, hpts⟩, hstr⟩ := decodeWithK_ok keep hs hd
  obtain ⟨j', l', recs', hp', hl1', hr', hroot', ⟨ps', hps', hpts'⟩, hstr'⟩ := Claims.Bridge.Custody.liveOf_ok hl
  rw [hp] at hp'
  cases hp'
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

/-! ## The plain verdict's round counts, at any `keep` -/

/-- `streams_ns_plain` at `verifyWithMK keep`. -/
theorem streams_ns_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].toList.map Round.n = plainRepNs spec sts[k / 2]! := by
  obtain ⟨hsz, hall⟩ := rep_plainK keep hs hv hl
  refine ⟨hsz, fun k hk => ?_⟩
  obtain ⟨sess, sch, stream, c, -, hpn, hf, hrep, hfind, hfa⟩ := hall k hk
  obtain ⟨stream', extras, hfind', hex, hns⟩ := verifyRepWith_readsN hs rfl hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := extraClaims_size hex
  have hsch : Claims.Flock.plainSched sts[k / 2]!.m = sch := by
    simp only [Claims.Flock.plainSched, hf, Option.getD_some]
  rw [map_n_of_forall₂ hfa, hns, hx, hpn, plainRepNs, hsch, Nat.mul_comm]
  rfl

/-- `rounds_ns_plain` at `verifyWithMK keep`. -/
theorem rounds_ns_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv) :
    (lv.streams.toList.flatMap (·.toList)).map Round.n = plainNs spec sts := by
  obtain ⟨hsz, hall⟩ := streams_ns_plainK keep hs hv hl
  have e : lv.streams.toList = (List.range lv.streams.size).map fun k => lv.streams[k]! := by
    refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
    simp only [Array.length_toList] at h₁
    simp [getElem!_pos lv.streams i h₁]
  rw [List.map_flatMap, e, List.flatMap_map, plainNs, ← hsz, List.flatMap_def, List.flatMap_def]
  refine congrArg List.flatten (List.map_congr_left fun k hk => ?_)
  have hk' : k < lv.streams.size := List.mem_range.1 hk
  rw [getElem!_pos lv.streams k hk']
  exact hall k hk'

/-- `streams_sizes_plain` at `verifyWithMK keep`. -/
theorem streams_sizes_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv) :
    lv.streams.toList.map (fun s => s.size) =
      (List.range (2 * sts.size)).map fun k => (plainRepNs spec sts[k / 2]!).length := by
  obtain ⟨hsz, hall⟩ := streams_ns_plainK keep hs hv hl
  refine List.ext_getElem (by simp [hsz]) fun i h₁ h₂ => ?_
  have hi : i < lv.streams.size := by simpa using h₁
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  rw [← hall i hi]
  simp only [List.length_map, Array.length_toList]

/-! ## Two records under one custody have one session's rounds, as replay reads them -/

/-- `live_keys_eq` at `verifyWithMK keep` and `keep'`. -/
theorem live_keys_eqK (keep keep' : Bool → Bool) (hs hs' : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record record' : String} {proofs proofs' : Array ByteArray} {lv lv' : Live}
    {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin} {mids : List (Claims.Msg × Coin)}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv)
    (hv' : verifyWithMK keep' hs' spec (tabsOf sts) record' proofs' = .ok ()) (hl' : liveOf spec record' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ()) :
    lv.streams.toList.map (fun s => s.toList.map roundKey) =
      lv'.streams.toList.map (fun s => s.toList.map roundKey) := by
  refine streams_eq_of_flat roundKey
    ((streams_sizes_plainK keep hs hv hl).trans (streams_sizes_plainK keep' hs' hv' hl').symm) ?_
  rw [keys_of_forall₂ (Claims.Bridge.Custody.custody_ok hc).2.2.2,
    keys_of_forall₂ (Claims.Bridge.Custody.custody_ok hc').2.2.2, rounds_ns_plainK keep hs hv hl,
    rounds_ns_plainK keep' hs' hv' hl']

/-- How the element-level decoder reads a round that `liveOf` reads (`decodeWithK_live`). -/
abbrev DecRoundK (rd rd' : Round) : Prop := rd.n = rd'.n ∧ rd.coins = rd'.coins ∧ rd.bytes = rd'.bytes

theorem erase_eq_of {x y : Round} {B : ByteArray} (hn : x.n = y.n) (hc : x.coins = y.coins)
    (hx : x.bytes = some B) (hy : y.bytes = some B) : x.erase = y.erase := by
  rcases x with ⟨n, s, c, b⟩
  rcases y with ⟨n', s', c', b'⟩
  simp only at hn hc hx hy
  subst hn hc hx hy
  rfl

theorem retained_ofK : ∀ {a c : List Round}, List.Forall₂ DecRoundK a c →
    (∀ rd ∈ c, ∃ B, rd.bytes = some B) → ∀ rd ∈ a, ∃ B, rd.bytes = some B
  | _, _, .nil, _, _, h => by simp at h
  | _, _, .cons hab h, hr, rd, hm => by
    rcases List.mem_cons.1 hm with rfl | hm
    · obtain ⟨B, hB⟩ := hr _ (List.mem_cons_self ..)
      exact ⟨B, hab.2.2.trans hB⟩
    · exact retained_ofK h (fun x hx => hr x (List.mem_cons_of_mem _ hx)) rd hm

/-- Two decoded round lists over retained rounds with the same counts, coins and bytes are one list as replay reads
it. -/
theorem rounds_erase_of_keys : ∀ {a b c d : List Round},
    List.Forall₂ DecRoundK a c → List.Forall₂ DecRoundK b d → c.map roundKey = d.map roundKey →
    (∀ rd ∈ c, ∃ B, rd.bytes = some B) → a.map Round.erase = b.map Round.erase
  | _, _, _, _, .nil, .nil, _, _ => rfl
  | _, _, _, _, .nil, .cons _ _, hk, _ => by simp at hk
  | _, _, _, _, .cons _ _, .nil, hk, _ => by simp at hk
  | _ :: _, _ :: _, u :: _, _ :: _, .cons hx ha, .cons hy hb, hk, hret => by
    simp only [List.map_cons, List.cons.injEq] at hk ⊢
    obtain ⟨hk1, hk2⟩ := hk
    obtain ⟨B, hB⟩ := hret u (List.mem_cons_self ..)
    obtain ⟨hn₁, hc₁, hb₁⟩ := hx
    obtain ⟨hn₂, hc₂, hb₂⟩ := hy
    simp only [roundKey, Prod.mk.injEq] at hk1
    obtain ⟨kn, kc, kb⟩ := hk1
    exact ⟨erase_eq_of (hn₁.trans (kn.trans hn₂.symm)) (hc₁.trans (kc.trans hc₂.symm)) (hb₁.trans hB)
      (hb₂.trans (kb.symm.trans hB)), rounds_erase_of_keys ha hb hk2 (fun rd h => hret rd (List.mem_cons_of_mem _ h))⟩

/-- **Two records the plain verdict accepts at any `keep`s, under custody of the same link and round messages,
decode to sessions with the same rounds on every rep stream as replay reads them**, all retained (`sess_rounds_eq`,
with the rounds compared after `Round.erase`). -/
theorem sess_rounds_eqK (keep keep' : Bool → Bool) (hs hs' : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record record' : String} {proofs proofs' : Array ByteArray} {lv lv' : Live}
    {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin} {mids : List (Claims.Msg × Coin)}
    {sess sess' : Session}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv)
    (hv' : verifyWithMK keep' hs' spec (tabsOf sts) record' proofs' = .ok ()) (hl' : liveOf spec record' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ())
    (hd : Record.decodeWithK keep hs spec record = .ok sess)
    (hd' : Record.decodeWithK keep' hs' spec record' = .ok sess') {q : ℕ} (hq : q < spec.repStreams.size) :
    ∃ y y' : StreamRec, sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y ∧
      sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' ∧
      y.rounds.map Round.erase = y'.rounds.map Round.erase ∧ AllRetained y.rounds := by
  obtain ⟨-, -, -, hst⟩ := decodeWithK_live keep hs hd hl
  obtain ⟨-, -, -, hst'⟩ := decodeWithK_live keep' hs' hd' hl'
  obtain ⟨y, hy, hfa⟩ := hst q hq
  obtain ⟨y', hy', hfa'⟩ := hst' q hq
  have hk := keys_at (live_keys_eqK keep keep' hs hs' hv hl hv' hl' hc hc') q
  have hr := live_retained hc q
  have e : y.rounds.toList.map Round.erase = y'.rounds.toList.map Round.erase := rounds_erase_of_keys hfa hfa' hk hr
  refine ⟨y, y', hy, hy', Array.ext' (by simpa only [Array.toList_map] using e), fun j hj => ?_⟩
  rw [getElem!_pos y.rounds j hj]
  exact retained_ofK hfa hr _ (Array.mem_toList_iff.2 (Array.getElem_mem hj))

end FlockDeploy.KS
