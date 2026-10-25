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
and the plain verdict's round counts at any `keep` (`rounds_ns_plainK`, from `streams_ns_plainK`).
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
  unfold Record.decodeWithK Record.decodeJWithK at h
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

/-! ## The element-level verdict's reps -/

/-- `verifyWithM_proofs_size` at `verifyWithMK keep`. -/
theorem verifyWithMK_proofs_size (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec}
    {tabs : Array Setup} {record : String} {proofs : Array ByteArray}
    (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) : proofs.size = spec.repStreams.size := by
  unfold verifyWithMK at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rename_i hne
  simpa [bne_iff_ne] using hne

/-- `verifyWithM_rep` at `verifyWithMK keep`: table `k / 2`'s rep `k % 2` passes on proof file `k`, on the session the
element-level decoder reads. -/
theorem verifyWithMK_rep (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {record : String} {proofs : Array ByteArray} (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) :
    ∃ (sess : Session) (schs : Array Schedule), Record.decodeWithK keep hs spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧
      ∀ (k : ℕ) (hk : k / 2 < tabs.size), fast100 tabs[k / 2].m = some schs[k / 2]! ∧
        ∃ c, verifyRepWith hs spec tabs[k / 2] schs[k / 2]! sess (k / 2) (k % 2) proofs[k]! = .ok c := by
  obtain ⟨sess, schs, hdec, hsz, hss, hall⟩ := verifyWithMK_tables keep hs h
  refine ⟨sess, schs, hdec, hsz, fun k hk => ?_⟩
  obtain ⟨hk', c₀, c₁, hf, h₀, h₁⟩ := hall (k / 2) hk
  rw [getElem!_pos schs (k / 2) hk']
  refine ⟨hf, ?_⟩
  rcases Nat.mod_two_eq_zero_or_one k with e | e
  · have e' : proofs[k]! = proofs[2 * (k / 2)]! := by rw [show 2 * (k / 2) = k by omega]
    rw [e, e']
    exact ⟨c₀, h₀⟩
  · have e' : proofs[k]! = proofs[2 * (k / 2) + 1]! := by rw [show 2 * (k / 2) + 1 = k by omega]
    rw [e, e']
    exact ⟨c₁, h₁⟩

/-- `next_acc_decide` at `nextE`. -/
theorem nextE_acc_decide {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((nextE x as cs).eval H).1 = .acc) :
    ∃ s spec sts link c₁ mids f, act false x as cs = .decide s spec sts link c₁ mids f ∧
      Runs H (decisionE s x link c₁ mids f) () (((nextE x as cs).eval H).2.map Prod.fst) := by
  cases hact : act false x as cs with
  | rej => rw [nextE_rej H hact] at h; cases h
  | coin c => rw [nextE_coin H hact] at h; cases h
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_⟩
    have ha : ((outOf (decisionE s x link c₁ mids f)).eval H).1 = .acc := by
      simp only [nextE, hact] at h
      exact h
    have hl : ((nextE x as cs).eval H).2 = ((decisionE s x link c₁ mids f).run.eval H).2 := by
      simp only [nextE, hact, outOf_log]
    rw [hl]
    exact Runs.self H (outOf_acc H _ ha)

/-! ## The calls half where a proof file differs, at `VE` -/

/-- **Two last messages the element-level verifier accepts after the same rounds and coins** have the same proof
files, or two of their openings collide among the two decisions' calls, or two of their proof files at one position
differ and decode alike (`plain_calls_tied`'s, at `nextE`). The second file's rep is moved onto the first record's
session (`Flock.With.verifyRepWith_erase`), where both are accepted as one rep on one stream. -/
theorem plainE_calls_tied {x : Claims.Msg} {ms : List Claims.Msg} {l l' : Claims.Msg} {cs : List Coin}
    (hc : cs.length = ms.length + 1)
    (hacc : ((nextE x (ms ++ [l]) cs).eval sha512).1 = .acc)
    (hacc' : ((nextE x (ms ++ [l']) cs).eval sha512).1 = .acc) :
    ∃ f f' : Final, finalOf l = some f ∧ finalOf l' = some f' ∧
      (f.proofs = f'.proofs ∨
        Claims.TiedColl sha512 (((nextE x (ms ++ [l]) cs).eval sha512).2 ++
          ((nextE x (ms ++ [l']) cs).eval sha512).2) ∨
        RecDeploy.ProofSlack l l') := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := nextE_acc_decide hacc
  obtain ⟨s', spec', sts', link', c₁', mids', f', hact', hrun'⟩ := nextE_acc_decide hacc'
  have hs := act_decide hact
  have hs' := act_decide hact'
  rw [hs] at hs'
  cases hs'
  have hlen := act_decide_length (by simp [hc]) hact
  obtain ⟨b, ms'', rfl⟩ : ∃ b ms'', ms = b :: ms'' := by
    cases ms with
    | nil =>
      simp only [List.nil_append, List.length_singleton] at hlen
      omega
    | cons b ms'' => exact ⟨b, ms'', rfl⟩
  obtain ⟨a₁, rest, crest, has, hcs, hlink, hmids⟩ := act_decide_parts hact
  obtain ⟨a₁', rest', crest', has', hcs', hlink', hmids'⟩ := act_decide_parts hact'
  simp only [List.cons_append, List.cons.injEq] at has has'
  obtain ⟨rfl, rfl⟩ := has
  obtain ⟨rfl, rfl⟩ := has'
  obtain ⟨rfl, rfl⟩ : c₁' = c₁ ∧ crest' = crest := by
    rw [hcs] at hcs'
    simp only [List.cons.injEq] at hcs'
    exact ⟨hcs'.1.symm, hcs'.2.symm⟩
  rw [hlink] at hlink'
  cases hlink'
  have hm : mids' = mids := by rw [hmids, hmids', List.dropLast_concat, List.dropLast_concat]
  subst hm
  have hfl : finalOf l = some f := by
    have e := act_decide_final hact
    rw [List.getLast?_concat] at e
    exact e
  have hfl' : finalOf l' = some f' := by
    have e := act_decide_final hact'
    rw [List.getLast?_concat] at e
    exact e
  refine ⟨f, f', hfl, hfl', ?_⟩
  by_cases hpf : f.proofs = f'.proofs
  · exact Or.inl hpf
  right
  have hsu := (setupOf_some hs).2
  obtain ⟨hvr, hck⟩ := decisionE_runs hsu hrun
  obtain ⟨hvr', hck'⟩ := decisionE_runs hsu hrun'
  generalize hL₁ : ((nextE x (b :: ms'' ++ [l]) cs).eval sha512).2 = L₁ at hvr ⊢
  generalize hL₂ : ((nextE x (b :: ms'' ++ [l']) cs).eval sha512).2 = L₂ at hvr' ⊢
  have hV : verifyWithMK (fun _ => true) (hashesIn (L₁.map Prod.fst)) spec (tabsOf sts) f.record f.proofs =
      .ok () := verdictE_within hvr
  have hV' : verifyWithMK (fun _ => true) (hashesIn (L₂.map Prod.fst)) spec (tabsOf sts) f'.record f'.proofs =
      .ok () := verdictE_within hvr'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checks_ok hck'
  have hsz := verifyWithMK_proofs_size _ _ hV
  have hsz' := verifyWithMK_proofs_size _ _ hV'
  obtain ⟨k, hk, hk', hne⟩ : ∃ (k : ℕ) (hk : k < f.proofs.size) (hk' : k < f'.proofs.size),
      f.proofs[k] ≠ f'.proofs[k] := by
    by_contra hcon
    push Not at hcon
    exact hpf (Array.ext (hsz.trans hsz'.symm) hcon)
  obtain ⟨sess, schs, hdec, htsz, hrep⟩ := verifyWithMK_rep _ _ hV
  obtain ⟨sess', schs', hdec', -, hrep'⟩ := verifyWithMK_rep _ _ hV'
  have hq : k < spec.repStreams.size := hsz ▸ hk
  have hi : k / 2 < (tabsOf sts).size := by
    have := repStreams_size spec
    omega
  obtain ⟨hsch, c, hr₁⟩ := hrep k hi
  obtain ⟨hsch', c', hr₂⟩ := hrep' k hi
  rw [hsch] at hsch'
  have hss : schs'[k / 2]! = schs[k / 2]! := (Option.some.inj hsch').symm
  rw [hss] at hr₂
  generalize hst : (tabsOf sts)[k / 2] = st at hsch hr₁ hr₂
  generalize hsc : schs[k / 2]! = sch at hsch hr₁ hr₂
  have hfold := tabsOf_fold sts st (hst ▸ Array.getElem_mem hi)
  have hsts : k / 2 < sts.size := by simpa [tabsOf] using hi
  have hms : st.merkle = MerkleScheme.hm96Sha512 ∨
      st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rw [← hst, tabsOf_getElem hi]
    rcases checks_leaf hck hsu (k / 2) hsts with e | ⟨-, e⟩
    · exact Or.inl e
    · exact Or.inr e
  have named := named_of hms
  obtain ⟨y, y', hy, hy', hyy, hret⟩ := sess_rounds_eqK _ _ _ _ hV hlv hV' hlv' hcu hcu' hdec hdec' hq
  simp only [repStreams_getElem spec hq] at hy hy'
  obtain ⟨hrB, hpts, -⟩ := decodeWithK_live _ _ hdec hlv
  obtain ⟨hrB', hpts', -⟩ := decodeWithK_live _ _ hdec' hlv'
  obtain ⟨hlB, -, hlP, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  obtain ⟨hlB', -, hlP', -⟩ := Claims.Bridge.Custody.custody_ok hcu'
  have hP : sess.points = sess'.points := by rw [hpts, hpts', hlP, hlP']
  have hB : sess.rootB = sess'.rootB := by
    rw [hrB, hrB']
    exact Array.ext' (hlB.trans hlB'.symm)
  have hr₂s : verifyRepWith (hashesIn (L₂.map Prod.fst)) spec st sch sess (k / 2) (k % 2) f'.proofs[k]! =
      .ok c' := by
    rw [With.verifyRepWith_erase With.Sim.liftMap_V _ spec st sch _ hy hy' hyy hP (by rw [hB])]
    exact hr₂
  have hpure : ∀ {L : List Claims.Msg} {sess₀ : Session} {bs c₀ : ByteArray},
      verifyRepWith (hashesIn L) spec st sch sess₀ (k / 2) (k % 2) bs = .ok c₀ →
        verifyRep spec st sch sess₀ (k / 2) (k % 2) bs = .ok c₀ := by
    intro L sess₀ bs c₀ h
    rw [← With.verifyRepWith_std _ _ named]
    exact (With.verifyRepWith_sim sub (sub_hashes L) spec st sch sess₀ _ _ bs) c₀ h
  have hret' : ∀ stream : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[k / 2]! ++ toString "/rep" ++ toString (k % 2)) =
        some stream → AllRetained stream.rounds := by
    intro stream hs₀
    have e : stream = y := Option.some.inj (hs₀.symm.trans hy)
    rw [e]
    exact hret
  obtain ⟨stream, extras, p, p', hstr, hex, hp, hp', hor⟩ :=
    rep_bind_proof spec st sch sess (k / 2) (k % 2) _ _ c c' hsch hfold.1 hfold.2 hms (hpure hr₁) (hpure hr₂s) hret'
  have hb : f.proofs[k]! = f.proofs[k] := getElem!_pos f.proofs k hk
  have hb' : f'.proofs[k]! = f'.proofs[k] := getElem!_pos f'.proofs k hk'
  rcases hor with rfl | ⟨ℓ, hℓ, j, hj, hcoll⟩
  · right
    refine ⟨f, f', k, f.proofs[k], f'.proofs[k], hfl, hfl', Array.getElem?_eq_getElem hk,
      Array.getElem?_eq_getElem hk', hne, shapeOf st sch extras.size, p, ?_, ?_⟩
    · rw [← hb]; exact hp
    · rw [← hb']; exact hp'
  · left
    obtain ⟨s₁, e₁, q₁, hs₁, he₁, hq₁, hlog₁⟩ := rep_level_logs hsch hfold.1 hfold.2 hms hr₁
    obtain ⟨s₂, e₂, q₂, hs₂, he₂, hq₂, hlog₂⟩ := rep_level_logs hsch hfold.1 hfold.2 hms hr₂s
    have es₁ : s₁ = stream := Option.some.inj (hs₁.symm.trans hstr)
    have es₂ : s₂ = stream := Option.some.inj (hs₂.symm.trans hstr)
    have ee₁ : e₁ = extras := Except.ok.inj (he₁.symm.trans hex)
    have ee₂ : e₂ = extras := Except.ok.inj (he₂.symm.trans hex)
    subst s₁ s₂ e₁ e₂
    have eq₁ : q₁ = p := Except.ok.inj (hq₁.symm.trans hp)
    have eq₂ : q₂ = p' := Except.ok.inj (hq₂.symm.trans hp')
    subst q₁ q₂
    refine CollIn.tied hcoll (fun u hu => ?_) (fun u hu => ?_)
    · rw [List.map_append]
      exact List.mem_append_left _ (hlog₁ ℓ hℓ j hj u hu)
    · rw [List.map_append]
      exact List.mem_append_right _ (hlog₂ ℓ hℓ j hj u hu)

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out OC)
open FlockDeploy (Coin)

/-- `PlainCallsOfProofs` at the element-level verifier: two last messages `VE` accepts after the same rounds and
coins, where it answered each earlier round with that round's coin, with the same proof files, make the same set of
calls. -/
def PlainCallsOfProofsE (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin) (f f' : FlockDeploy.Final),
    ms.length + 1 = FlockDeploy.VE.k x → cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.nextE x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).1 = .acc →
    FlockDeploy.finalOf l = some f → FlockDeploy.finalOf l' = some f' → f.proofs = f'.proofs →
    ((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2.toFinset =
      ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2.toFinset

/-- `PlainCallsBindOrSlack` at the element-level verifier. -/
def PlainCallsBindOrSlackE (H : Msg → Msg) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (l l' : Msg) (cs : List Coin), ms.length + 1 = FlockDeploy.VE.k x →
    cs.length = ms.length + 1 →
    (∀ j < ms.length, ((FlockDeploy.nextE x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
      .msg (cs.getD j default).bytes) →
    ((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).1 = .acc →
    ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).1 = .acc →
    ((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2.toFinset ≠
      ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2.toFinset →
    (∃ q ∈ (((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2).map Prod.fst,
      ∃ q' ∈ (((FlockDeploy.nextE x (ms ++ [l]) cs).eval H).2 ++
        ((FlockDeploy.nextE x (ms ++ [l']) cs).eval H).2).map Prod.fst, Claims.Collision H q q') ∨
    ProofSlack l l'

/-- **The calls half at `VE` from the case where the proof files agree**, at SHA-512
(`plainCallsBindOrSlack_of_proofs`'s, through `FlockDeploy.KS.plainE_calls_tied`). -/
theorem plainCallsBindOrSlackE_of_proofs (hR : PlainCallsOfProofsE FlockDeploy.sha512) :
    PlainCallsBindOrSlackE FlockDeploy.sha512 := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  obtain ⟨f, f', hfl, hfl', hpf | ⟨u, v, hcol, hu, hv⟩ | hsl⟩ := FlockDeploy.KS.plainE_calls_tied hc hacc hacc'
  · exact absurd (hR x ms l l' cs f f' hk hc hearly hacc hacc' hfl hfl' hpf) hne
  · exact Or.inl ⟨u, hu, v, hv, hcol⟩
  · exact Or.inr hsl

end RecDeploy
