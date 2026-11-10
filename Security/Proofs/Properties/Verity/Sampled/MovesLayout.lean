module

public import Proofs.Properties.Verity.Sampled.RootB
public import Proofs.Properties.Verity.Sampled.LinkFirst
public import Proofs.Properties.Verity.Sampled.Held

/-!
# `MovesReach` at the session's layout

At the readings and the decoders at the session's layout (`Live.layout`, `Live.tables`) and the sample-coupled coin map
under `fieldEnc`, every clause of `MovesOf` holds at one witness of the joint law's support off the link's and the
rounds' empty classes (`ReachesOff`): one widened sample, one set of link points and one set of rep rounds, since the
three sit together inside the writer's output.

* **The digest each opening answers** (`rep_answers`): each opening v2's read shows of a rep stream answers the
  digest the record's round gives, the `after` of the one line opening that round (`opensRound_unique`,
  `opening_after`), which `Record.decode` reads as the round's digest under SHA-512 digests (`decode_sha`).
* **`MovesOf` at a witness** (`ReachesOff.movesOf`): the rep rounds (`firstStop_reps_verdict`) fix the outcome as the
  writer's; the trace's first stop fixes a cut (`firstStop_fixes`), the link points are released after a log `L₀` in
  order (`firstStop_order`), and that release is the log's first opening (`firstOpened_link`). Then the record decodes
  to the session's sample file at the sample (`recordDraw_firstStop`), the roots decoded at the first stop are the
  record's `root_B` (`firstStop_rootB`), the link points are the record's (`link_pts`), and the proof files are v2's
  (`proofs_of_sent`).
  The order the record's sample and the first stop take, that the sample's coins precede every opening and that the first
  opening is the session stream's, comes from v2's order refusal, which the sampled verdict passed
  (`freshBefore_verdict`, `linkFirst_verdict`).
* **`MovesReach`** (`movesReach_layout`): `ReachesOff.movesOf` at every outcome, at the coupling's cut (v2's read of
  the log is the device's, and v2 samples at the digest `after`, `SampledDigest`), with what it takes of each accepted
  play the cut holds of stated for every tape.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Claims.Bridge.Custody (forall₂_getElem mapM_forall₂ mapIdxM_forall₂)
open FlockSoundness.Refine (ebind_ok Ret)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The digest each opening of a rep stream answers -/

/-- **The line opening a round a tree shows opened is after the opening's bytes**: where `coinTreesOf` accepts the
release log `lines`, a line of it opening round `k` of stream `name` of a tree `t` of its trees (`OpensRound`) is the
one line opening that round (`opensRound_unique`), so its `after` is that of the opening `t` shows (`opened_after`). -/
theorem opening_after {lines : List String} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    {t : LoggedTree} (ht : t ∈ trees) {name : String} {k : ℕ} {o : TreeOpening} (ho : t.opening name k = some o)
    {l : String} (hl : l ∈ lines) (hr : OpensRound l t.root name k) {d : Json} (hd : event l = .ok d) :
    _root_.Spot.getD d "after" = .str o.after := by
  have hp := Array.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  obtain ⟨l', hl', hr', d', hd', ha, -⟩ := opened_after hT t ht o (Array.mem_of_find?_eq_some ho)
  rw [hp.1, hp.2] at hr'
  obtain rfl := opensRound_unique hT hl hl' hr hr'
  rw [hd] at hd'
  cases hd'
  exact ha

/-- Of one tree, a stream of a tree read is of that tree. -/
theorem root_of_keyAt {trees : Array LoggedTree} (h1 : trees.size = 1) {t : LoggedTree} (ht : t ∈ trees)
    {i si : ℕ} {ρ name : String} (hk : KeyAt (heldTrees trees) i si ρ name) : ρ = t.root := by
  obtain ⟨-, sp, hg, -⟩ := hk
  unfold heldTrees at hg
  rw [List.getElem?_map, Array.getElem?_toList] at hg
  obtain ⟨t', ht', he⟩ := Option.map_eq_some_iff.1 hg
  obtain ⟨hi, -⟩ := Array.getElem?_eq_some_iff.1 ht'
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 ht
  obtain rfl : i = j := by omega
  rw [Array.getElem?_eq_getElem hj] at ht'
  rw [← (Prod.mk.inj he).1, ← Option.some.inj ht']

/-- **What `Record.decode` reads of each round's digest**: the bytes of the round's `msg_{dk}` (S12). -/
theorem decode_sha {spec : Flock.SessionSpec} {text : String} {sess : Flock.RunRecord}
    (h : Flock.Record.decode spec text = .ok sess) :
    ∃ j recs, Flock.JsonParse.parse text = .ok j ∧ Flock.Record.arr j "streams" = .ok recs ∧
      List.Forall₂ (fun x (y : Flock.StreamRec) => Flock.Record.str x "stream" = .ok y.name ∧
        ∃ rs, Flock.Record.arr x "rounds" = .ok rs ∧ List.Forall₂ (fun r (rd : Flock.Round) =>
          ∃ s, Flock.Record.str r s!"msg_{spec.dk}" = .ok s ∧ Flock.unhex s = .ok rd.sha) rs.toList y.rounds.toList)
        recs.toList sess.streams.toList := by
  revert sess
  show Ret (Flock.Record.decode spec text) _
  rw [Flock.Record.decode.eq_def]
  cases hp : Flock.JsonParse.parse text with
  | error e => intro sess h; cases h
  | ok j =>
    simp only []
    rw [pure_bind]
    ret_walk [Flock.Record.fld, Flock.Record.arr, Array.mapM]
    all_goals
      refine ⟨j, _, rfl, by assumption, List.Forall₂.imp (fun x y h => ?_) (mapM_forall₂ (by assumption))⟩
      beta_reduce at h
      repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      refine ⟨by assumption, _, by assumption,
        List.Forall₂.imp (fun r rd h => ?_) (mapIdxM_forall₂ (by assumption))⟩
      obtain ⟨k, h⟩ := h
      repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
      by_cases hnull : Flock.Record.isNull r "msg" = true
      · rw [ite_eq_left hnull] at h
        obtain ⟨b, hb, h⟩ := ebind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at hb h
        subst hb h
        exact ⟨_, by assumption, by assumption⟩
      · rw [ite_eq_right hnull] at h
        obtain ⟨s', hs', h⟩ := ebind_ok h
        obtain ⟨b, hb, h⟩ := ebind_ok h
        obtain ⟨len, hlen, h⟩ := ebind_ok h
        obtain ⟨u, hck, h⟩ := ebind_ok h
        obtain ⟨b', hb', h⟩ := ebind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at hb' h
        subst hb' h
        exact ⟨_, by assumption, by assumption⟩

/-- **Each opening v2's read shows of a record's stream answers the record's round's digest** (`rep_rounds`'s
`hans`): where v2 read the release log `lines` of one coin tree as the releases `pre ++ y :: post`, `y` opening tree
`i`'s stream `s`, named `name` in the `Hello`s before it, the line `y` reads opens round `openedOf pre i s` of that
stream (`readLog_round`) after the bytes `y` answers; the record `sessionTreeReason` accepts is of that tree
(`root_of_keyAt`) and gives that round's `msg_sha512` as the `after` of the opening the tree shows (`rounds_held`),
which is that line's (`opening_after`); and under SHA-512 digests `Record.decode` reads the round's digest as those
bytes (`decode_sha`). -/
theorem rep_answers {label : String} {after : ByteArray} {J : Json} {ξ : ByteArray} {lines : List String}
    {pre post : List Release} {y : Release} {trees : Array LoggedTree} {i s n : ℕ} {name : String}
    {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {st : Flock.StreamRec}
    (hR : readLog label after [J] ξ lines = some (pre ++ y :: post)) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) {tabsB : Array (String × Option ℕ)} (hJt : sessionTreeReason trees tabsB J = none)
    (hy : y.frame = .opening i s n) (hO : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name)
    (hd : Flock.Record.decode spec record = .ok sess) (hJ : Flock.JsonParse.parse record = .ok J)
    (h512 : spec.sha512Digests = true) (hst : sess.streams.find? (·.name == name) = some st)
    (hk : openedOf pre i s < st.rounds.size) : y.answers = st.rounds[openedOf pre i s]!.sha := by
  obtain ⟨l, ρ, name', k, hl, hro, hkey, hname, rfl⟩ := readLog_round hct hR rfl hy
  have hnn : name' = name := Option.some.inj (hname.symm.trans hO)
  subst name'
  obtain ⟨t, htm, -, streams, hstr, hall⟩ := rounds_held hJt
  obtain rfl := root_of_keyAt h1 htm hkey
  obtain ⟨j, recs, hp, hrecs, hdec⟩ := decode_sha hd
  rw [hJ] at hp
  obtain rfl : J = j := Except.ok.inj hp
  have hsr := arr_getD hrecs
  rw [hstr] at hsr
  obtain rfl : streams = recs := Lean.Json.arr.inj hsr
  have hsn : st.name = name := by simpa using Array.find?_some hst
  obtain ⟨x, hx, hxs, rs, hrs, hrds⟩ := Properties.Spot.SecurityProofs.forall₂_mem_right hdec st
    (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hst))
  have hkr : openedOf pre i s < rs.size := by
    have e := hrds.length_eq
    simp only [Array.length_toList] at e
    omega
  obtain ⟨sx, hsx, hux⟩ := forall₂_getElem hrds hkr
  have hdk : spec.dk = "sha512" := by
    unfold Flock.SessionSpec.dk
    simp [h512]
  rw [show s!"msg_{spec.dk}" = "msg_sha512" by simp [hdk, toString]] at hsx
  obtain ⟨-, o, ho, hmsg⟩ := hall x (Array.mem_toList_iff.1 hx) name rs (by rw [← hsn]; exact str_getD hxs)
    (arr_getD hrs) (rs[openedOf pre i s], openedOf pre i s)
    (Array.mem_zipIdx_iff_getElem?.2 (Array.getElem?_eq_getElem hkr))
  obtain rfl : sx = o.after := Lean.Json.str.inj ((str_getD hsx).symm.trans hmsg)
  obtain ⟨hor, -, d, hev, -, aft, haft, hya⟩ := hro
  have ha := opening_after hct htm ho hl hor hev
  rw [haft] at ha
  obtain rfl : aft = o.after := Lean.Json.str.inj ha
  exact Except.ok.inj (hya.symm.trans hux)

/-! ## `MovesOf` at a witness -/

/-- **A session v2 sets up is tied to v2's verifier** (`SetsUp`'s `params`). -/
theorem SetsUp.popTied {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A) :
    PopTied θ A := by
  obtain ⟨p, hp, -, -, -, hdesc, hq, -⟩ := hA.params
  unfold PopTied
  exact ⟨p, hp, hdesc, hq⟩

-- `decodersAt`'s roots search `List.finRange A.rounds`, which unification would otherwise unfold at `A.rounds`'s value
attribute [local irreducible] Live.rounds Live.N in
/-- **`MovesOf` at a witness of the joint law's support, at the session's layout**: at the readings and the decoders at
`Live.layout` and `Live.tables`, and at the sample-coupled coin map under `fieldEnc`, an outcome off the link's and the
rounds' empty classes, beside a play that the sampled verdict accepts, that misses nowhere and whose log v2's check read
is the device's, is the writer's at the widened sample, the decoded registration and an outcome whose moves are those
v2's files decode to (`MovesOf`). It takes, for the record's sample (`recordDraw_firstStop`, at the session tied to v2's
verifier, `SetsUp.popTied`, at v2's log at its coins, `drawReadAll_coinsRead`, and at v2's hold on the record's sample,
`drawHeld`), that v2 reads the sample's coins of the family `label` answering `after` (`hN`), and, for the rep rounds
(`firstStop_reps_verdict`), that the log opens only the record's rounds. v2's order refusal, which the sampled verdict
passed, gives that every release of that family precedes every opening (`freshBefore_verdict`) and that the device's
log opens the session stream first (`linkFirst_verdict`). That the log has an opening and that its
trees name their streams distinctly come from the verdict (`verdict_opens`, `verdict_nodup`), that each opening of a rep
stream answers the record's round's digest from v2's read and the record (`rep_answers`), and the session's one link
round and its `N` live rounds from `Live.rounds_eq`. The roots (`firstStop_rootB`) and the link points (`link_pts`) read
the same first opening. -/
theorem ReachesOff.movesOf {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (reg : Option Claims.Msg → A.Reg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {rd : Reading A.rounds}
    (hrd : rd = readingAt A.tables A.layout record recordDigest) {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {label : String} {after : ByteArray} {m : ℕ}
    {κ : A.Ω → Fin m → Fin 512 → Bool} (hm : _root_.Spot.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (κ ω) 0))
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ)
    {o : A.Out × CoinTape (V.game θ F c) σ}
    (h : ReachesOff V θ F c A rd σ D M (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
      (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) o)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2)) (hn : NoMiss V θ F c rd σ o.2)
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (hN : SampleNamed k (CoinTape.run (V.game θ F c) σ o.2).sent label after)
    (hO : ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text →
      Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log) :
    ∃ (S : Finset (Fin A.n)) (R : A.Reg) (out : LiveOutJ (A.J S R) A.mP), o.1 = (R, S, A.wr S R out) ∧
      MovesOf A (CoinTape.run (V.game θ F c) σ o.2).sent S R out := by
  subst hrd hD hM
  obtain ⟨ω, pts, hL0, hr⟩ := h
  have hθ := (play_ends V θ F c σ o.2).1
  have hfirst := linkFirst_verdict hA hθ hv hl
  have hP := freshBefore_verdict hA hθ hv hl hN
  have h0r := A.rounds_pos
  obtain ⟨reps, ho, hreps⟩ := firstStop_reps_verdict hA hDg reg record recordDigest rfl rfl rfl ω pts hL0 hr
    hv hn hl hfirst hO
    (fun hR hct h1 => fun hJt hy hO' _ hd hJ h512 hst hk => rep_answers hR hct h1 hJt hy hO' hd hJ h512 hst hk)
  obtain ⟨_, _, _, _, hrec₀, -⟩ := accepts_sent hA hθ hv (filesOn_play hA hDg hv)
  have hop := verdict_opens hA hθ hv hl hrec₀
  have hnd := verdict_nodup hA hθ hv hl hrec₀
  have hjl := A.layout_link h0r
  have hK := firstStop_fixes V θ F c A (readingAt A.tables A.layout record recordDigest) σ
    (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout record recordDigest) σ label after κ)
    (readingAt_picks A.tables A.layout record recordDigest) A.tables ⟨0, h0r⟩
    (fun pre fr h => by
      show slotRound A.tables (A.layout ⟨0, h0r⟩) pre fr
      rw [hjl]
      exact h) ω _ hr hop
  obtain ⟨L₀, as, ms, ⟨hLA, hc⟩, -, -, -, hgt, -, hcov⟩ := firstStop_order V θ F c A A.layout_injective
    A.tables_nodup (hA.layout_lt _ _) record recordDigest reg rfl rfl σ
    (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout record recordDigest) σ label after κ)
    (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (B := roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
    (linkBits fieldEnc) (fun _ _ => rfl) (fun _ _ h => h) (fun _ x => roundBits fieldEnc x) (fun _ _ _ => rfl)
    (fun _ _ _ h => h) ω pts hK hL0 hr rfl (picksOnce_readingAt A.tables A.layout record recordDigest hnd)
    A.rounds_eq.le
  obtain ⟨_, i, s₀, n₀, b₀, L', -, -, -, hL₀⟩ := id hLA
  have h₀ := firstOpened_link (logServed_run V θ F c σ o.2) hjl hcov hgt hL₀ rfl
  have hno := noOpening_of_firstOpened h₀
  refine ⟨_, _, _, ho, fun text proofs spec sts sess hrec hpr hs hd =>
    ⟨?_, ?_, ?_, hreps text proofs spec sts sess hrec hpr hs hd, proofs_of_sent reg A.layout hpr⟩⟩
  · rw [sample_extend]
    exact recordDraw_firstStop fieldEnc hA hA.popTied (drawReadAll_coinsRead k) (drawHeld k)
      (readingAt_picks A.tables A.layout record recordDigest) hm hsrc ω
      (firstStop_playReaches V θ F c A _ σ _ ω hr) hv hl hN hP hrec
  · exact firstStop_rootB hA rfl rfl hr hn hv hl hfirst hLA hc h₀ hrec hs hd
  · exact (link_pts hA hθ hv hl hL₀ rfl hno (hfirst L₀ _ L' hL₀ i s₀ n₀ rfl hno) rfl hrec hs hd).symm

/-- **`MovesReach` at the session's layout** (`couplingTarget_assembled`'s `hM` at these readings, decoders and coin
map, with `after` the strategy's `afterAt`): `ReachesOff.movesOf` at every outcome, at the cut that v2's check read the
device's log and that v2 samples at `after` (`SampledDigest`), where what it takes of a play holds of every tape whose
play the sampled verdict accepts and whose log v2's check read is the device's, and v2's family and digest (`hN`) of
those whose sample answers `after`. -/
theorem movesReach_layout {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (reg : Option Claims.Msg → A.Reg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {rd : Reading A.rounds}
    (hrd : rd = readingAt A.tables A.layout record recordDigest) {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {label : String} {after : ByteArray} {m : ℕ}
    {κ : A.Ω → Fin m → Fin 512 → Bool} (hm : _root_.Spot.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (κ ω) 0))
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ)
    (hN : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      SampledDigest k (CoinTape.run (V.game θ F c) σ x).sent after →
        SampleNamed k (CoinTape.run (V.game θ F c) σ x).sent label after)
    (hO : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log) :
    MovesReach V θ F c k (fun o => Binding.Logged Binding.readsV2 o ∧ SampledDigest k o.sent after) A rd σ D M
      (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) :=
  fun _ h ⟨hv, hn, hl, hd⟩ => h.movesOf hA hDg reg record recordDigest hrd hD hm hsrc hM hv hn hl
    (hN _ hv hl hd) (hO _ hv hl)

end Verity.Sampled
