module

public import Proofs.Properties.Verity.Sampled.RoundsRead
public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Verity.Sampled.LinkPts
public import Proofs.Properties.Verity.Sampled.SpecTie

/-!
# `MovesReach`, clause by clause

On the joint law's support off the held-off events (`ReachesOff`), each outcome is the writer's at the widened sample's
sample, the decoded registration and a live session's outcome (`reachesOff_out`). What v2's files decode to in the play on
the tape beside it, clause by clause of `MovesOf`:

* **The play from the first stop** (`ReachesOff.out_reaches`): the play beside an outcome is one the first stop
  reaches (`firstStop_playReaches`), at the widened sample and the link points the outcome is the writer's at.
* **The record's sample** (`ReachesOff.recordDraw`, clause (i)): at the sample-coupled coin map, where the play's sampled
  verdict accepts, the log v2 read is the device's, v2 reads the sample's coins of the family `label` answering `after`,
  and every release of that family precedes every opening, the record decodes to the session's sample file at the
  outcome's sample (`recordDraw_firstStop`, `sample_extend`).
* **An answered stop's round** (`stop_round`, toward clause (iv)): a stop the trace answers on the first table's rep
  `r` stream, at that stream's `k`-th opening, submits the record's round `k` of that stream, as `roundOf` makes it of
  the stop's message and the coins the opening released; the record's stream has that many rounds
  (`recordRounds_decode`).
* **The rep rounds** (`rep_rounds`, clause (iv) at the first table): the rounds the answered stops submit on the first
  table's rep `r`, in order, are the record's rounds of that stream (`repRounds`), by `stream_rounds` at the stream's
  openings, `stop_round` at each and `openedOf_eq_recordRounds` for their count. It takes, about v2's read and the
  record, that each round the record gives a stream gives coins (`hsome`), that each stream it gives rounds of is a
  stream of the tree read (`hkey`), and that each opening answers its round's digest (`hans`).
* **The rep rounds at a `ReachesOff` witness** (`firstStop_reps`, clause (iv)): at a layout with a link slot and one
  table, an outcome reached at a widened sample and link points off the held-off events is the writer's there at rep
  rounds that are the record's, where the log's first opening is the session stream's: the first stop fixes a cut
  (`firstStop_fixes`), the trace's answered stops are in order in the log (`firstStop_order`), the link's opening is the
  log's first (`firstOpened_link`), and `rep_rounds` at each rep. It is stated at the witness, so that `MovesReach`
  reads every clause at one sample, one set of link points and one set of rep rounds.
* **The rounds a record `sessionTreeReason` accepts** (`recordRound_isSome`, `keyAt_of_record`): each round it gives a
  stream gives coins (`rep_rounds`'s `hsome`), and each stream it gives rounds of is a stream of the tree v2's read
  holds (`hkey`), the line opening its round 0 naming that stream of that tree.
* **v2's read on an accepted play** (`verdict_read`, `verdict_nodup`, `verdict_opens`): where the sampled verdict
  accepts and the log v2's check read is the device's, v2 read one coin tree's release log at the record it parsed,
  which `sessionTreeReason` holds to it; the log's trees name their streams distinctly, and it has an opening, of the
  record's link points.
* **Clause (iv) at the session's layout, from the verdict** (`firstStop_reps_verdict`): at `Live.layout` and
  `Live.tables`, the rep rounds at a `ReachesOff` witness are those of the session v2's record decodes to at the
  session `--zk` sets up at `A`, for every record and session `MovesOf` reads; v2's read, the spec's tables, digests
  and retained rounds come from the verdict (`verdict_read`, `SetsUp.setup_one`, `setup_sha512Digests`), and the
  layout's slot for each round the log opens of a stream of the record's tree from the session's coin tree
  (`SetsUp.covers_logged`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger
open Claims.Bridge.Custody (decode_ok DecStream forall₂_find? forall₂_mem_left)
open FlockSoundness.Refine (roundOf)
open FlockSoundness.Discharge.Tables (LiveOutJ SubmitJ addRoundJ)
open Claims.Bridge.CustodyZkP (repRounds)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An outcome on the joint law's support off `BL` and `B` is the writer's, beside a play the first stop reaches**
(`reachesOff_out`, `firstStop_playReaches`). -/
theorem ReachesOff.out_reaches {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live}
    {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {BL : LinkHold A.mP} {B : RoundHold}
    {o : A.Out × CoinTape (V.game θ F c) σ} (h : ReachesOff V θ F c A rd σ D M BL B o) :
    ∃ (ω : (A.extend M.widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽)
      (reps : Fin (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) → Fin 2 → List Flock.Round),
      ¬ BL (firstStop V θ F c A rd σ M ω).g' pts ∧
      o.1 = (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)
          ⟨D.root (firstStop V θ F c A rd σ M ω).modes, pts, reps, D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩) ∧
      PlayReaches (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s'
        (CoinTape.run (V.game θ F c) σ o.2) := by
  obtain ⟨ω, pts, hL, hr⟩ := h
  obtain ⟨_, h1, -⟩ := hr.out
  exact ⟨ω, pts, _, hL, h1, firstStop_playReaches V θ F c A rd σ M ω hr⟩

/-- **The record's sample on the joint law's support** (`MovesOf`'s clause (i)): at the sample-coupled coin map, an
outcome on the joint law's support off `BL` and `B` is the writer's at the widened sample `ω`, and where the play beside
it ends where the sampled verdict accepts, the log v2 read is the device's, v2 reads the sample's coins of the family
`label` answering `after`, and every release of that family precedes every opening, its record decodes to the session's
sample file at `ω`'s sample (`ReachesOff.out_reaches`, `recordDraw_firstStop`). -/
theorem ReachesOff.recordDraw (e : 𝔽 ≃ (Fin 128 → Bool)) {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {k : Verity.Registration.Composite} {A : Live} {rd : Reading A.rounds}
    {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg} {BL : LinkHold A.mP} {B : RoundHold}
    {label : String} {after : ByteArray} {m : ℕ} {κ : A.Ω → Fin m → Fin 512 → Bool}
    (hA : SetsUp V θ F A) (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) (hH : DrawHeld k)
    (hrd : PicksOpenings rd) (hm : _root_.Spot.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (κ ω) 0))
    {o : A.Out × CoinTape (V.game θ F c) σ}
    (h : ReachesOff V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ) BL B o)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2))
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (hN : SampleNamed k (CoinTape.run (V.game θ F c) σ o.2).sent label after)
    (hP : ∀ pre r post, (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: post → ∀ i s n,
      r.frame = .opening i s n → ∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after)) :
    ∃ (ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽)
      (reps : Fin (A.J ((A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω)
        (regOf V θ F c A rd σ D)) → Fin 2 → List Flock.Round),
      ¬ BL (firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).g' pts ∧
      o.1 = (regOf V θ F c A rd σ D, (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω,
        A.wr ((A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω) (regOf V θ F c A rd σ D)
          ⟨D.root (firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).modes, pts, reps,
            D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩) ∧
      ∀ record, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some record →
        FlockSoundness.Discharge.Exec.recordDraw record =
          some (A.dj ((A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω)) := by
  obtain ⟨ω, pts, reps, hL, h1, hreach⟩ := h.out_reaches
  refine ⟨ω, pts, reps, hL, h1, fun record hrec => ?_⟩
  rw [sample_extend]
  exact recordDraw_firstStop e hA ht hR hH hrd hm hsrc ω hreach hv hl hN hP hrec

/-! ## The rep rounds, stop by stop -/

/-- **The rounds the record gives a decoded stream are its rounds**: where `record` decodes to `sess` and parses as `J`,
`recordRounds` reads of `J`'s stream `name`, not the session stream, the count of rounds of `sess`'s stream so named. -/
theorem recordRounds_decode {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {J : Json}
    {name : String} {y : Flock.StreamRec} (hd : Flock.Record.decode spec record = .ok sess)
    (hJ : Flock.JsonParse.parse record = .ok J)
    (hn : (name == _root_.Spot.Challenger.TREE_SESSION) = false)
    (hy : sess.streams.find? (·.name == name) = some y) : recordRounds J name = y.rounds.size := by
  obtain ⟨j, l, recs, hp, -, hrecs, -, -, hstr⟩ := decode_ok hd
  rw [hJ] at hp
  obtain rfl := Except.ok.inj hp
  have hy' : sess.streams.toList.find? (·.name == name) = some y := by rw [Array.find?_toList]; exact hy
  obtain ⟨x, ⟨-, rs, hrs, hrds⟩, hx⟩ := forall₂_find? (R := fun y x => DecStream spec x y)
    (p := fun y => y.name == name) (q := fun x => (Flock.Record.str x "stream").toOption == some name)
    (fun a b hab => by rw [hab.1]; rfl) (forall₂_swap hstr) hy'
  have hlen : rs.size = y.rounds.size := by simpa using hrds.length_eq
  have hx' : recs.find? (fun x => (Flock.Record.str x "stream").toOption == some name) = some x := by
    rw [← Array.find?_toList]; exact hx
  simp only [recordRounds, hn, Bool.false_eq_true, ↓reduceIte, arr_getD hrecs]
  rw [find?_congr (q' := fun x => (Flock.Record.str x "stream").toOption == some name) ?_, hx']
  · simp only [arr_getD hrs]
    exact hlen
  · intro a ha
    obtain ⟨y', -, hy''⟩ := forall₂_mem_left hstr a (Array.mem_toList_iff.2 ha)
    rw [str_getD hy''.1, hy''.1]
    rfl

/-- **An answered stop's round is the record's**, at the readings at a layout `L` whose round `j` is round `k` of the
first table's rep `r` stream: where the stop `q`, keeping the invariant at the rounds' cuts, fixes round `j`'s cut at
the most likely message `w`, the rest of the trace runs from it off every miss, and the device's log has the coins `cs`
of `w`'s count released at the opening of tree `i`'s stream `s`, named that stream, after the log `m` at which the cuts
the stop fixes are at the bytes that opening answers and which opened that stream `k` times: where v2 read the log of
one tree (`readLog`, `coinTreesOf`) of the record `sessionTreeReason` accepts, `Flock.Zk.verify` accepts the record
under retained rounds and SHA-512 digests at `A`'s digest, the log opens only the record's rounds (`OpenedRecorded`),
each round the record gives a stream gives coins (`hsome`) and each opening v2 read answers its round's digest
(`hans`), the record's stream so named has round `k`, and it is the round `roundOf` makes of `w` and `cs`. -/
theorem stop_round {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live}
    {tables : Array String} {L : Fin A.rounds → Slot} {record : Exchange → Option Claims.Msg}
    {recordDigest : Claims.Msg} {rd : Reading A.rounds} (hrd : rd = readingAt tables L record recordDigest)
    {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP}
    {B : RoundHold} {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin A.mP → 𝔽} {rs : Fin J → Fin 2 → List Flock.Round}
    {q : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds)} (hq : q.Inv (roundCuts V θ F c rd))
    {φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs (q.toStop.rewrite φ).consume o)
    (hn : NoMiss V θ F c rd σ o.2) {j : Fin A.rounds} {r : Fin 2} {k : ℕ} (hLj : L j = .rep 0 r k)
    (hj : (q.toStop.Ks j).isHere = true) {w : Claims.Msg} (hw : q.toStop.modes j = some w)
    {cs : Fin (tailCount w) → 𝔽} {m L' : List Release} {i s n : ℕ} {b : ByteArray}
    (hlog : (CoinTape.run (V.game θ F c) σ o.2).log = m ++ ⟨.opening i s n, b, roundBits fieldEnc cs⟩ :: L')
    (hcut : CutsAt rd.reveal q.toStop m (CoinTape.run (V.game θ F c) σ o.2).log) (hk : openedOf m i s = k)
    (hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = tables[0]?.map (repStreamName · r.val))
    {label : String} {after ξ : ByteArray} {lines : List String} {lr : List Release} {trees : Array LoggedTree}
    {Jr : Json}
    (hF : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (CoinTape.run (V.game θ F c) σ o.2).log)
    (hread : readLog label after [Jr] ξ lines = some lr) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) {tabsB : Array (String × Option ℕ)}
    (hJt : sessionTreeReason trees tabsB Jr = none) {text : String} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {proofs : Array ByteArray} {sess : Flock.RunRecord}
    (hrec : sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text)
    (hJ : Flock.JsonParse.parse text = .ok Jr) (hd : Flock.Record.decode spec text = .ok sess)
    (hv : Flock.Zk.verify spec tabs text proofs = .ok ()) (hret : spec.retainRounds = true)
    (h512 : spec.sha512Digests = true) (hdg : spec.dg = A.dg) (htab : tables = spec.tables)
    (h0 : 0 < spec.tables.size) (hO : OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log)
    (hsome : ∀ name, ∀ k < recordRounds Jr name, (recordRound Jr name k).isSome)
    (hans : ∀ {pre post : List Release} {y : Release} {i s n : ℕ} {name : String} {st : Flock.StreamRec},
      lr = pre ++ y :: post → y.frame = .opening i s n →
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == _root_.Spot.Challenger.TREE_SESSION) = false →
      sess.streams.find? (·.name == name) = some st → openedOf pre i s < st.rounds.size →
      y.answers = st.rounds[openedOf pre i s]!.sha) :
    ∃ st, sess.streams.find? (·.name == repStreamName spec.tables[0]! r.val) = some st ∧
      ∃ hk : k < st.rounds.size, roundOf A.dg ⟨w.toArray⟩ (tailCount w) cs = st.rounds[k] := by
  subst hrd htab
  have ht0 : spec.tables[0]? = some spec.tables[0]! := by
    rw [Array.getElem?_eq_getElem h0, getElem!_pos spec.tables 0 h0]
  have hname' : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) =
      some (repStreamName spec.tables[0]! r.val) :=
    hname.trans (congrArg (Option.map (repStreamName · r.val)) ht0)
  have hnT := repStreamName_ne spec.tables[0]! r.val
  obtain ⟨h0t, sch, tPad, cap, -, hrep⟩ := verifyRep_of_accepted hv hd r
  obtain ⟨st, _, _, _, hst, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrep
  have hst' : sess.streams.find? (·.name == repStreamName spec.tables[0]! r.val) = some st := hst
  have hrr : recordRounds Jr (repStreamName spec.tables[0]! r.val) = st.rounds.size :=
    recordRounds_decode hd hJ hnT hst'
  have hkR : k < st.rounds.size := by
    have h := hO m _ L' hlog i s n rfl _ hname'
    rwa [hk, hrr] at h
  rw [hlog] at hF
  obtain ⟨pre, y, post, hlr, hpre, hy, -⟩ := forall₂_split (List.forall₂_map_left_iff.1 hF)
  have hy' : (Binding.Read.ofV2 y).Of ⟨.opening i s n, b, roundBits fieldEnc cs⟩ := hy
  have hyf : y.frame = .opening i s n := hy'.1
  have hya : y.answers = b := hy'.2.1
  have hpre' : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) pre m :=
    hpre.imp fun _ _ h => ⟨h.1, h.2.1⟩
  have hHe : helloSpecs pre = helloSpecs m := helloSpecs_congr hpre'
  have hOp : openedOf pre i s = k := (openedOf_congr hpre').trans hk
  obtain ⟨l, ρ, name, k', -, hro, hkey, hnm, hk'⟩ := readLog_round hct hread hlr hyf
  rw [hHe, hname'] at hnm
  obtain rfl : repStreamName spec.tables[0]! r.val = name := Option.some.inj hnm
  rw [hOp] at hk'
  rw [hk'] at hro
  obtain ⟨b', hb'⟩ := Option.isSome_iff_exists.1
    (hsome (repStreamName spec.tables[0]! r.val) k (by rw [hrr]; exact hkR))
  obtain ⟨t₁, ht₁, hroot, -⟩ := Trees.round_held hJt hb'
  have hρ : ρ = recordTree Jr := by
    obtain ⟨-, sp, hg, -⟩ := hkey
    unfold heldTrees at hg
    rw [List.getElem?_map] at hg
    obtain ⟨t₂, ht₂, he₂⟩ := Option.map_eq_some_iff.1 hg
    rw [hroot, LinkTrees.eq_of_size_one h1 ht₁ (Array.mem_toList_iff.1 (List.mem_of_getElem? ht₂))]
    exact (Prod.mk.inj he₂).1.symm
  have hyc : y.coins = Verity.bitsOfBytes b' := by
    rw [hro.2.1, hρ]
    simp [Lines.openCoins, hb']
  obtain ⟨cs', hrr', hf', hsha⟩ := recordRound_decode hd hJ hnT hst' hkR
  obtain ⟨bb, hbb, hmsg⟩ := roundMsg_decode hd hret hst' hkR
  have hdg512 : spec.dg bb = Flock.Sha512.hash bb := by simp [Flock.SessionSpec.dg, h512]
  have hyans : y.answers = st.rounds[k].sha := by
    have h := hans hlr hyf (by rw [hHe]; exact hname') hnT hst' (by rw [hOp]; exact hkR)
    rwa [hOp, getElem!_pos st.rounds k hkR] at h
  have hrev : (readingAt spec.tables L record recordDigest).reveal j (CoinTape.run (V.game θ F c) σ o.2) =
      some bb.data.toList := by
    change slotReveal spec.tables (L j) _ = _
    rw [hLj]
    simp only [slotReveal, roundIn, ht0, Option.bind_some, hrec, hmsg]
  have hmode := RoundsReach.stop_modes_reveal hq hr hn hcut hlog hj hrev (by
    change Verity.sha512Msg bb.data.toList = b.data.toList
    rw [← hya, hyans, hsha bb hbb, hdg512]
    all_goals rfl)
  have hwb : w = bb.data.toList := Option.some.inj (hw.symm.trans hmode)
  subst hwb
  have hbits : Verity.bitsOfBytes b' = roundBits fieldEnc cs := by
    by_cases hc : y.coins = []
    · have hrc : _root_.Spot.Challenger.recordCoins? (.arr cs') = some b' := hrr'.symm.trans hb'
      have hlen := congrArg List.length (bits_of_coins hf' hrc)
      rw [← hyc, hc, length_chunks, List.length_nil, Array.length_toList] at hlen
      have hmem : st ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hst')
      have hwf := FlockSoundness.Refine.decode_roundsWF hd st hmem k hkR
      rw [getElem!_pos st.rounds k hkR] at hwf
      have htc := rep_tailCount_accepted hv hd hret r hst' hkR hmsg
      have hz : (roundBits fieldEnc cs).length = 0 := by
        rw [length_roundBits, htc, ← hwf]
        omega
      rw [← hyc, hc, List.length_eq_zero_iff.1 hz]
    · rw [← hyc]
      exact hy'.2.2 y.coins (by simp [Binding.Read.ofV2, hc])
  obtain ⟨hB', hS', hN', hsz, hC'⟩ := answered_of_record hd hJ hret hnT hst' hkR hmsg hb' hbits
  have hcoins : (Array.ofFn fun i => (cs i).val) = st.rounds[k].coins := by
    refine Array.ext (by rw [Array.size_ofFn, hsz]) fun x h₁ h₂ => ?_
    have hx : x < tailCount bb.data.toList := by simpa using h₁
    rw [Array.getElem_ofFn, hC' ⟨x, hx⟩, getElem!_pos _ x h₂]
  refine ⟨st, hst', hkR, ?_⟩
  have e : st.rounds[k] =
      ({ n := st.rounds[k].n, sha := st.rounds[k].sha, coins := st.rounds[k].coins, bytes := st.rounds[k].bytes } :
        Flock.Round) := rfl
  rw [e, hB', hS', hN', ← hcoins, ← hdg]
  all_goals rfl

/-- A stream of the trees a read of one tree names is of the first tree. -/
theorem KeyAt.eq_zero {trees : Array LoggedTree} (h1 : trees.size = 1) {i si : ℕ} {ρ name : String}
    (h : KeyAt (heldTrees trees) i si ρ name) : i = 0 := by
  have hlt : i < (heldTrees trees).length := (List.findIdx?_eq_some_iff_getElem.1 h.1).1
  rw [heldTrees, List.length_map, Array.length_toList, h1] at hlt
  omega

theorem map_range_getElem! {γ : Type} [Inhabited γ] (xs : Array γ) :
    (List.range xs.size).map (fun k => xs[k]!) = xs.toList := by
  refine List.ext_getElem (by simp) fun n h₁ h₂ => ?_
  rw [List.getElem_map, List.getElem_range, Array.getElem_toList]
  exact getElem!_pos xs n (by simpa using h₂)

/-- **The trace's rounds on the first table's rep `r` stream are the record's** (`MovesOf`'s clause (iv) at a rep):
at the readings and the decoders at a layout `L`, injective, of distinct tables, where the trace's answered stops `as`
each release their round's coins after the logs `ms` (`firstStop_order`, at the coin map that writes a round's coins
as `roundBits fieldEnc`), the device's log `Lg` opens the session stream first, the layout has a slot for each round
`Lg` opens of a stream of the record's tree (`hcovL`), and v2's read and the record are as `stop_round` takes them,
with each stream the record gives
rounds of a stream of the tree read (`hkey`), the rounds the stops on that stream submit, in order, are the record's
rounds of it (`repRounds`): by `stream_rounds` at the stream's openings, `stop_round` at each, and
`openedOf_eq_recordRounds` for their count; a stream the record gives no rounds has no stop on it. -/
theorem rep_rounds {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live}
    (reg : Option Claims.Msg → A.Reg) {tables : Array String} {L : Fin A.rounds → Slot}
    (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {B : RoundHold} {J : ℕ} {β : Type}
    {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {o : β × CoinTape (V.game θ F c) σ} (hn : NoMiss V θ F c (readingAt tables L record recordDigest) σ o.2)
    {as : List (Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) × ((s : SubmitJ J) × (Fin s.n → 𝔽)))}
    {ms : List (List Release)} {L₀ L' : List Release}
    (hF : List.Forall₂ (fun a m => (decodersAt reg L).sub J a.1.toStop.modes = some a.2.1 ∧
      LoggedAfter (readingAt tables L record recordDigest).round a.1.toStop m (roundBits fieldEnc a.2.2)
        (CoinTape.run (V.game θ F c) σ o.2).log ∧
      CutsAt (readingAt tables L record recordDigest).reveal a.1.toStop m (CoinTape.run (V.game θ F c) σ o.2).log)
      as ms)
    (hI : ∀ a ∈ as, a.1.Inv (roundCuts V θ F c (readingAt tables L record recordDigest)) ∧ ∃ N' rs',
      RoundsReach D M A.dg J B fin rootB pts N' rs' (a.1.toStop.rewrite (M.round a.1.toStop J a.2.1 a.2.2)).consume o)
    (hgt : ∀ m ∈ ms, L₀.length < m.length) (hpw : ms.Pairwise (fun m m' => m.length < m'.length))
    (hcov : ∀ j L₁, PicksIn (readingAt tables L record recordDigest).round (CoinTape.run (V.game θ F c) σ o.2).log j
      L₁ → L₁ = L₀ ∨ L₁ ∈ ms)
    (hS : LogServed (CoinTape.run (V.game θ F c) σ o.2).log)
    (hnd : ∀ sp ∈ helloSpecs (CoinTape.run (V.game θ F c) σ o.2).log, sp.streams.toList.Nodup) {r₀ : Release}
    {i s₀ n₀ : ℕ} (hL₀ : (CoinTape.run (V.game θ F c) σ o.2).log = L₀ ++ r₀ :: L')
    (hr₀ : r₀.frame = .opening i s₀ n₀) (h₀ : firstOpened L₀ = none)
    (hses : ((helloSpecs L₀)[i]?.bind fun sp => sp.streams[s₀]?) = some TREE_SESSION) (t : Fin J) (htz : t.val = 0)
    (r : Fin 2)
    {label : String} {after ξ : ByteArray} {lines : List String} {lr : List Release} {trees : Array LoggedTree}
    {Jr : Json}
    (hV : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (CoinTape.run (V.game θ F c) σ o.2).log)
    (hread : readLog label after [Jr] ξ lines = some lr) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) {tabsB : Array (String × Option ℕ)}
    (hJt : sessionTreeReason trees tabsB Jr = none)
    (hcovL : ∀ {i si : ℕ} {name : String}, KeyAt (heldTrees trees) i si (recordTree Jr) name → ∀ s,
      ∀ k < openedOf (CoinTape.run (V.game θ F c) σ o.2).log i s, ∃ j, L j = .rep t.val r k)
    {text : String} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {proofs : Array ByteArray} {sess : Flock.RunRecord}
    (hrec : sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text)
    (hJ : Flock.JsonParse.parse text = .ok Jr) (hd : Flock.Record.decode spec text = .ok sess)
    (hv : Flock.Zk.verify spec tabs text proofs = .ok ()) (hret : spec.retainRounds = true)
    (h512 : spec.sha512Digests = true) (hdg : spec.dg = A.dg) (htab : tables = spec.tables)
    (h0 : 0 < spec.tables.size) (hO : OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log)
    (hsome : ∀ name, ∀ k < recordRounds Jr name, (recordRound Jr name k).isSome)
    (hkey : ∀ name, 0 < recordRounds Jr name → ∃ i si, KeyAt (heldTrees trees) i si (recordTree Jr) name)
    (hans : ∀ {pre post : List Release} {y : Release} {i s n : ℕ} {name : String} {st : Flock.StreamRec},
      lr = pre ++ y :: post → y.frame = .opening i s n →
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == _root_.Spot.Challenger.TREE_SESSION) = false →
      sess.streams.find? (·.name == name) = some st → openedOf pre i s < st.rounds.size →
      y.answers = st.rounds[openedOf pre i s]!.sha) :
    as.foldl (fun rs a => addRoundJ rs a.2.1.tab a.2.1.rep (roundOf A.dg a.2.1.bytes a.2.1.n a.2.2))
      (fun _ _ => []) t r = repRounds spec sess t.val r.val := by
  subst htab
  obtain ⟨tv, htv⟩ := t
  obtain rfl : tv = 0 := htz
  show _ = repRounds spec sess 0 r.val
  have hlg : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) lr
      (CoinTape.run (V.game θ F c) σ o.2).log :=
    (List.forall₂_map_left_iff.1 hV).imp fun _ _ h => ⟨h.1, h.2.1⟩
  have ht0 : spec.tables[0]? = some spec.tables[0]! := by
    rw [Array.getElem?_eq_getElem h0, getElem!_pos spec.tables 0 h0]
  have hnT := repStreamName_ne spec.tables[0]! r.val
  obtain ⟨h0t, sch, tPad, cap, -, hrep⟩ := verifyRep_of_accepted hv hd r
  obtain ⟨st, _, _, _, hst, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrep
  have hst' : sess.streams.find? (·.name == repStreamName spec.tables[0]! r.val) = some st := hst
  have hrr := recordRounds_decode hd hJ hnT hst'
  have hR : repRounds spec sess 0 r.val = st.rounds.toList := by
    show ((sess.streams.find? (·.name == repStreamName spec.tables[0]! r.val)).map (·.rounds.toList)).getD [] = _
    rw [hst']
    all_goals rfl
  by_cases hpos : 0 < st.rounds.size
  · obtain ⟨i', si, hk⟩ := hkey (repStreamName spec.tables[0]! r.val) (by rw [hrr]; exact hpos)
    have hlg₀ := hlg
    rw [hL₀] at hlg₀
    obtain ⟨pre₀, y₀, post₀, hlr₀, -, hy₀, -⟩ := forall₂_split hlg₀
    obtain ⟨_, ρ₀, _, _, -, -, hk₀, -, -⟩ := readLog_round hct hread hlr₀ (hy₀.1.trans hr₀)
    rw [(KeyAt.eq_zero h1 hk).trans (KeyAt.eq_zero h1 hk₀).symm] at hk
    have hnm := readLog_named hread (coinTreesOf_fold [Jr] hct) hk
    rw [helloSpecs_congr hlg] at hnm
    have hs : ((helloSpecs (CoinTape.run (V.game θ F c) σ o.2).log)[i]?.bind fun sp => sp.streams[si]?) =
        spec.tables[0]?.map (repStreamName · r.val) := by
      rw [hnm, ht0]
      all_goals rfl
    have hf : ∀ a m, (a, m) ∈ as.zip ms → a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = r →
        roundOf A.dg a.2.1.bytes a.2.1.n a.2.2 = st.rounds[openedOf m i si]! := by
      rintro ⟨q, sub, cs⟩ m hx htr
      obtain ⟨hsub0, hlog0, hcut0⟩ := List.forall₂_zip hF hx
      have hsub : (decodersAt reg L).sub J q.toStop.modes = some sub := hsub0
      have hlog : LoggedAfter (readingAt spec.tables L record recordDigest).round q.toStop m (roundBits fieldEnc cs)
          (CoinTape.run (V.game θ F c) σ o.2).log := hlog0
      have hcut : CutsAt (readingAt spec.tables L record recordDigest).reveal q.toStop m
          (CoinTape.run (V.game θ F c) σ o.2).log := hcut0
      obtain ⟨hinv, N', rs', hreach⟩ := hI _ (List.of_mem_zip hx).1
      obtain ⟨k, w, j, n, b, L'', hLj, hj, hw, hsa, hname, hk', he⟩ := LoggedAfter.at_stream reg hL ht record
        recordDigest hsub hlog hS hnd hL₀ hr₀ h₀ (hgt m (List.of_mem_zip hx).2) ⟨0, htv⟩ r hs htr
      subst hsa
      obtain ⟨st', hst'', hkR, hround⟩ := stop_round rfl hinv hreach hn hLj hj hw he hcut hk' hname hV hread hct h1
        hJt hrec hJ hd hv hret h512 hdg rfl h0 hO hsome hans
      have hss := Option.some.inj (hst''.symm.trans hst')
      subst hss
      rw [hk', getElem!_pos _ k hkR]
      exact hround
    have hfold := stream_rounds reg hL ht record recordDigest (fun _ x => roundBits fieldEnc x) A.dg
      (hF.imp fun _ _ h => ⟨h.1, h.2.1⟩) hgt hpw hcov hS hnd hL₀ hr₀ h₀ hses ⟨0, htv⟩ r hs (hcovL hk si)
      (fun k => st.rounds[k]!) hf
    have hlen : openedOf (CoinTape.run (V.game θ F c) σ o.2).log i si = st.rounds.size := by
      rw [openedOf_eq_recordRounds hct hread hlg hJt (hsome _) hO hS hk, hrr]
    rw [hfold, hlen, map_range_getElem!, hR]
  · have hz : st.rounds.size = 0 := by omega
    have hnone : ∀ a ∈ as, ¬ (a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = r) := by
      intro a ha htr
      have ha' : a ∈ (as.zip ms).map Prod.fst := by
        rw [List.map_fst_zip hF.length_eq.le]
        exact ha
      obtain ⟨⟨⟨q, sub, cs⟩, m⟩, hx, rfl⟩ := List.mem_map.1 ha'
      obtain ⟨hsub0, hlog0, hcut0⟩ := List.forall₂_zip hF hx
      have hsub : (decodersAt reg L).sub J q.toStop.modes = some sub := hsub0
      have hlog : LoggedAfter (readingAt spec.tables L record recordDigest).round q.toStop m (roundBits fieldEnc cs)
          (CoinTape.run (V.game θ F c) σ o.2).log := hlog0
      have hcut : CutsAt (readingAt spec.tables L record recordDigest).reveal q.toStop m
          (CoinTape.run (V.game θ F c) σ o.2).log := hcut0
      obtain ⟨hinv, N', rs', hreach⟩ := hI _ (List.of_mem_zip hx).1
      obtain ⟨t', ht', r', k, w, j, i', s', n', b, L'', hLj, hj, hw, hsa, -, hname, hk', he⟩ :=
        LoggedAfter.slot reg hL ht record recordDigest hsub hlog
      subst hsa
      obtain ⟨ht'0, hr'r⟩ : t' = 0 ∧ r' = r := ⟨congrArg Fin.val htr.1, htr.2⟩
      subst ht'0 hr'r
      obtain ⟨st', hst'', hkR, -⟩ := stop_round rfl hinv hreach hn hLj hj hw he hcut hk' hname hV hread hct h1 hJt
        hrec hJ hd hv hret h512 hdg rfl h0 hO hsome hans
      have hss := Option.some.inj (hst''.symm.trans hst')
      subst hss
      omega
    have hfil : as.filter (fun a => decide (a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = r)) = [] :=
      List.filter_eq_nil_iff.2 fun a ha h => hnone a ha (of_decide_eq_true h)
    have hnil : st.rounds.toList = [] := List.eq_nil_of_length_eq_zero (by simpa using hz)
    refine (foldl_addRoundJ (fun a => a.2.1.tab) (fun a => a.2.1.rep)
      (fun a => roundOf A.dg a.2.1.bytes a.2.1.n a.2.2) as (fun _ _ => []) ⟨0, htv⟩ r).trans ?_
    rw [hfil, hR, hnil]
    all_goals rfl

/-- **The rep rounds at a sample the joint law's support is reached at** (`MovesOf`'s clause (iv) at a `ReachesOff`
witness): at the readings and the decoders at a layout `L`, injective, of distinct tables, whose rep slots are of a
plan's one table and which has a link slot, and at the sample-coupled coin map under `fieldEnc`, an outcome with a
widened sample `ω` and link points `pts` off the link's empty class, with a trace off the rounds' empty class from the
first stop ending on it, is the writer's at `ω`, `pts` and some rep rounds; where the tape beside it misses nowhere, its
log has an opening, its trees name their streams distinctly, its first opening is the session stream's
(`SampleLinkFirst`'s second conjunct), the layout has a slot for each round the log opens of a stream of the record's
tree, and v2's read and the record are as `rep_rounds` takes them, those rep rounds are the record's (`repRounds`), at
every table and rep
(`firstStop_fixes`, `firstStop_order`, `firstOpened_link`, `rep_rounds`). It takes the witness, not `ReachesOff`, since
the rep rounds sit inside the writer's output with the roots and the link points, which the other clauses read at the
same witness. -/
theorem firstStop_reps {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live}
    (reg : Option Claims.Msg → A.Reg) {tables : Array String} {L : Fin A.rounds → Slot}
    (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) (hLJ : ∀ S R j t r k, L j = .rep t r k → t < A.J S R)
    (hJ1 : ∀ S R, A.J S R = 1) {jl : Fin A.rounds} (hjl : L jl = .link) (hRN : A.rounds ≤ A.N + 1)
    {rd : Reading A.rounds} (hrd : rd = readingAt tables L record recordDigest) {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg L) {label : String} {after : ByteArray} {m : ℕ}
    {κ : A.Ω → Fin m → Fin 512 → Bool} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP}
    (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ) (ω : (A.extend M.widening).L.Ω)
    (pts : Fin 2 → Fin A.mP → 𝔽) {o : A.Out × CoinTape (V.game θ F c) σ}
    (hL0 : ¬ linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1) (firstStop V θ F c A rd σ M ω).g' pts)
    (hr : RoundsReach D M A.dg (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D))
      (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
      (fun t => (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D) t))
      (D.root (firstStop V θ F c A rd σ M ω).modes) pts A.N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o)
    (hn : NoMiss V θ F c rd σ o.2)
    (hop : ∃ L₁ r L₂ i s n, (CoinTape.run (V.game θ F c) σ o.2).log = L₁ ++ r :: L₂ ∧ r.frame = .opening i s n)
    (hnd : ∀ sp ∈ helloSpecs (CoinTape.run (V.game θ F c) σ o.2).log, sp.streams.toList.Nodup)
    (hfirst : ∀ pre r post, (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: post → ∀ i s n,
      r.frame = .opening i s n → (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some _root_.Spot.Challenger.TREE_SESSION)
    {label' : String} {after' ξ : ByteArray} {lines : List String} {lr : List Release} {trees : Array LoggedTree}
    {Jr : Json}
    (hV : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (CoinTape.run (V.game θ F c) σ o.2).log)
    (hread : readLog label' after' [Jr] ξ lines = some lr) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) {tabsB : Array (String × Option ℕ)}
    (hJt : sessionTreeReason trees tabsB Jr = none)
    (hcovL : ∀ {i si : ℕ} {name : String}, KeyAt (heldTrees trees) i si (recordTree Jr) name → ∀ s (r : Fin 2),
      ∀ k < openedOf (CoinTape.run (V.game θ F c) σ o.2).log i s, ∃ j, L j = .rep 0 r k)
    {text : String} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {proofs : Array ByteArray} {sess : Flock.RunRecord}
    (hrec : sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text)
    (hJ : Flock.JsonParse.parse text = .ok Jr) (hd : Flock.Record.decode spec text = .ok sess)
    (hv : Flock.Zk.verify spec tabs text proofs = .ok ()) (hret : spec.retainRounds = true)
    (h512 : spec.sha512Digests = true) (hdg : spec.dg = A.dg) (htab : tables = spec.tables)
    (h0 : 0 < spec.tables.size) (hO : OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log)
    (hsome : ∀ name, ∀ k < recordRounds Jr name, (recordRound Jr name k).isSome)
    (hkey : ∀ name, 0 < recordRounds Jr name → ∃ i si, KeyAt (heldTrees trees) i si (recordTree Jr) name)
    (hans : ∀ {pre post : List Release} {y : Release} {i s n : ℕ} {name : String} {st : Flock.StreamRec},
      lr = pre ++ y :: post → y.frame = .opening i s n →
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == _root_.Spot.Challenger.TREE_SESSION) = false →
      sess.streams.find? (·.name == name) = some st → openedOf pre i s < st.rounds.size →
      y.answers = st.rounds[openedOf pre i s]!.sha) :
    ∃ reps : Fin (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) → Fin 2 → List Flock.Round,
      o.1 = (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)
          ⟨D.root (firstStop V θ F c A rd σ M ω).modes, pts, reps, D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩) ∧
      ∀ j r, reps j r = repRounds spec sess j.val r.val := by
  subst hrd hD hM
  have hK := firstStop_fixes V θ F c A (readingAt tables L record recordDigest) σ
    (sampleMap fieldEnc V θ F c A (readingAt tables L record recordDigest) σ label after κ)
    (readingAt_picks tables L record recordDigest) tables jl
    (fun pre fr h => by
      show slotRound tables (L jl) pre fr
      rw [hjl]
      exact h) ω _ hr hop
  obtain ⟨L₀, as, ms, ⟨hlog₀, -⟩, ho, hF, hI, hgt, hpw, hcov⟩ := firstStop_order V θ F c A hL ht (hLJ _ _)
    record recordDigest reg rfl rfl σ
    (sampleMap fieldEnc V θ F c A (readingAt tables L record recordDigest) σ label after κ)
    (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (B := roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
    (linkBits fieldEnc) (fun _ _ => rfl) (fun _ _ h => h) (fun _ x => roundBits fieldEnc x) (fun _ _ _ => rfl)
    (fun _ _ _ h => h) ω pts hK hL0 hr rfl (picksOnce_readingAt tables L record recordDigest hnd) hRN
  refine ⟨_, ho, fun j r => ?_⟩
  obtain ⟨_, i, s₀, n₀, b₀, L', -, -, -, hL₀⟩ := hlog₀
  have hS := logServed_run V θ F c σ o.2
  have h₀ := firstOpened_link hS hjl hcov hgt hL₀ rfl
  have hses := hfirst L₀ _ L' hL₀ i s₀ n₀ rfl fun q hq i' s' n' hqf => by
    have h' := List.findSome?_eq_none_iff.1 h₀ q hq
    simp [hqf] at h'
  have htz : j.val = 0 := Nat.lt_one_iff.1 (lt_of_lt_of_eq j.isLt (hJ1 _ _))
  exact rep_rounds reg hL ht record recordDigest hn hF hI hgt hpw hcov hS hnd hL₀ rfl h₀ hses j htz r hV hread hct h1
    hJt (fun hk s k hk' => by
      rw [htz]
      exact hcovL hk s r k hk') hrec hJ hd hv hret h512 hdg htab h0 hO hsome hkey hans

/-! ## The rounds a record `sessionTreeReason` accepts -/

/-- **Each round of each stream a record `sessionTreeReason` accepts gives, gives coins, and is opened after its
digest**: the record's coin tree is a tree `t` of the log's, the record gives its streams as an array, and each round
of a stream of it named `name` gives coins (`recordCoins?`) and is a round `t` shows opened (`LoggedTree.opening`)
after the round's `msg_sha512`. -/
theorem rounds_held {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, recordTree record = t.root ∧ ∃ streams, _root_.Spot.getD record "streams" = .arr streams ∧
      ∀ s ∈ streams, ∀ name rounds, _root_.Spot.getD s "stream" = .str name →
        _root_.Spot.getD s "rounds" = .arr rounds → ∀ x ∈ rounds.zipIdx,
          (recordCoins? (_root_.Spot.getD x.1 "coins")).isSome ∧
            ∃ o, t.opening name x.2 = some o ∧ _root_.Spot.getD x.1 "msg_sha512" = .str o.after := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  rename_i t ht
  refine ⟨t, Array.mem_of_find?_eq_some ht, ?_, ?_⟩
  · have hp := Array.find?_some ht
    unfold recordTree
    rw [Trees.strIs hp]
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rename_i rootF points streams htri
  simp only [Prod.mk.injEq] at htri
  refine ⟨streams, htri.2.2, ?_⟩
  rcases hlo : t.opening _root_.Spot.Challenger.TREE_SESSION 0 with _ | o
  · simp only [hlo, Bool.false_eq_true, ↓reduceIte] at h
    all_goals (split at h <;> (try split at h) <;> cases h)
  simp only [hlo] at h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  refine Trees.ret_held (P := fun s => ∀ name rounds, _root_.Spot.getD s "stream" = .str name →
    _root_.Spot.getD s "rounds" = .arr rounds → ∀ x ∈ rounds.zipIdx,
      (recordCoins? (_root_.Spot.getD x.1 "coins")).isSome ∧
        ∃ o, t.opening name x.2 = some o ∧ _root_.Spot.getD x.1 "msg_sha512" = .str o.after) ?_
    (fun w _ hw => by cases hw) h
  intro s u res hres
  split at hres
  case h_2 => exact .inl ⟨_, _, hres.symm⟩
  rename_i name rounds hsr
  refine Or.imp id ?_ (Trees.step_held (P := fun x : Json × ℕ =>
    (recordCoins? (_root_.Spot.getD x.1 "coins")).isSome ∧
      ∃ o, t.opening name x.2 = some o ∧ _root_.Spot.getD x.1 "msg_sha512" = .str o.after) ?_ ?_ ?_ hres)
  · rintro ⟨u', hy, hrs⟩
    refine ⟨u', hy, fun name' rounds' hn hr' => ?_⟩
    simp only [Prod.mk.injEq] at hsr
    cases hn.symm.trans hsr.1
    cases hr'.symm.trans hsr.2
    exact hrs
  · intro x u res' hres'
    split at hres'
    case h_2 => exact .inl ⟨_, _, hres'.symm⟩
    rename_i coins n hcn
    rcases ho : t.opening name x.2 with _ | o
    · simp only [ho, Bool.false_eq_true, ↓reduceIte] at hres'
      all_goals (split at hres' <;> exact .inl ⟨_, _, hres'.symm⟩)
    simp only [ho] at hres'
    split at hres'
    case isFalse => exact .inl ⟨_, _, hres'.symm⟩
    split at hres'
    case isFalse => exact .inl ⟨_, _, hres'.symm⟩
    rename_i hok
    simp only [Prod.mk.injEq] at hcn
    simp only [Bool.and_eq_true, beq_iff_eq] at hok
    exact .inr ⟨_, hres'.symm, Option.isSome_iff_exists.2 ⟨coins, hcn.1⟩, o, rfl, Trees.strIs hok.1.2⟩
  · exact fun _ _ => ⟨(), rfl⟩
  · exact fun _ => ⟨(), rfl⟩

/-- A record's stream named `name`, as `recordRound` and `recordRounds` find it. -/
def streamNamed (name : String) (x : Json) : Bool :=
  match _root_.Spot.getD x "stream" with
  | .str n => n == name
  | _ => false

/-- **Each round a record `sessionTreeReason` accepts gives a stream gives coins** (`recordRound`), below the rounds it
gives that stream (`recordRounds`): the session stream's round 0 its link points' (`LinkTrees.link_some`), and each
other stream's its round's (`rounds_held`). -/
theorem recordRound_isSome {trees : Array LoggedTree} {tabsB : Array (String × Option ℕ)} {J : Json}
    (h : sessionTreeReason trees tabsB J = none) (name : String) :
    ∀ k < recordRounds J name, (recordRound J name k).isSome := by
  intro k hk
  by_cases hn : (name == _root_.Spot.Challenger.TREE_SESSION) = true
  · have h1 : recordRounds J name = 1 := by rw [recordRounds, ite_eq_left hn]
    rw [h1] at hk
    obtain rfl := Nat.lt_one_iff.1 hk
    obtain rfl := beq_iff_eq.1 hn
    obtain ⟨cs, hcs⟩ := LinkTrees.link_some h
    rw [hcs]
    rfl
  have hf := Bool.eq_false_iff.2 hn
  obtain ⟨_, _, _, streams, hst, hall⟩ := rounds_held h
  simp only [recordRounds, hf, Bool.false_eq_true, ↓reduceIte, hst] at hk
  simp only [recordRound, hf, Bool.false_and, Bool.false_eq_true, ↓reduceIte, hst]
  rw [find?_congr (q' := streamNamed name) ?_] at hk
  on_goal 2 => intro a _; unfold streamNamed; rfl
  rw [find?_congr (q' := streamNamed name) ?_]
  on_goal 2 => intro a _; unfold streamNamed; rfl
  split at hk
  case h_2 => exact absurd hk (Nat.not_lt_zero _)
  rename_i x hx
  have hname := Array.find?_some hx
  unfold streamNamed at hname
  split at hname
  case h_2 => cases hname
  rename_i n hxn
  cases beq_iff_eq.1 hname
  split at hk
  case h_2 => exact absurd hk (Nat.not_lt_zero _)
  rename_i rs hrs
  have hrk : rs[k]? = some rs[k] := Array.getElem?_eq_getElem hk
  simp only [hx, Option.bind_some, hrs, hrk]
  exact (hall x (Array.mem_of_find?_eq_some hx) _ rs hxn hrs (rs[k], k) (Array.mem_zipIdx_iff_getElem?.2 hrk)).1

/-- **Each stream a record `sessionTreeReason` accepts gives rounds of is a stream of the tree read**: where
`coinTreesOf` reads the release log `lines` as `trees`, its round 0 (`recordRound_isSome`) is opened on a line of the
log naming the record's tree and that stream (`Trees.open_held`), which names that stream of that tree among the trees
v2's read holds (`heldTrees`, `coinTreesOf_fold`, `treeFold_keyAt`). -/
theorem keyAt_of_record {lines : List String} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    {tabsB : Array (String × Option ℕ)} {J : Json} (h : sessionTreeReason trees tabsB J = none) {name : String}
    (hpos : 0 < recordRounds J name) : ∃ i si, KeyAt (heldTrees trees) i si (recordTree J) name := by
  obtain ⟨cs, hcs⟩ := Option.isSome_iff_exists.1 (recordRound_isSome h name 0 hpos)
  obtain ⟨l, hl, _, _, htl, d, hev, hTE, hroot, hs, -⟩ := Trees.open_held hT h hcs
  exact treeFold_keyAt (records := []) lines [] _ (coinTreesOf_fold [] hT) l hl htl d hev hTE hroot hs

/-! ## At the session's layout, from the verdict -/

/-- **v2's read on an accepted play**: where `A` is the session v2 sets up on the parameters, the sampled verdict
accepts and the log v2's check read is the device's, v2 read the release log `lines` of one coin tree
(`coinTreesOf`) as the releases `lr` (`readLog`) whose reads are the device's log's, at the record it parsed as `Jr`,
which `sessionTreeReason` holds to that tree at some tables (`Trees.records_held`). -/
theorem verdict_read {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} {x : Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ) (hv : V.Verdict k .sampled x)
    (hl : Binding.Logged Binding.readsV2 x) {text : String} (hrec : sentRecord x.sent = some text) :
    ∃ (label : String) (after ξ : ByteArray) (lines : List String) (lr : List Release) (trees : Array LoggedTree)
      (Jr : Json) (tabsB : Array (String × Option ℕ)),
      List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) x.log ∧ readLog label after [Jr] ξ lines = some lr ∧
      coinTreesOf lines = .ok trees ∧ trees.size = 1 ∧ sessionTreeReason trees tabsB Jr = none ∧
      Flock.JsonParse.parse text = .ok Jr := by
  subst hθ
  obtain ⟨p', hp', -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  obtain ⟨lr, hlr⟩ := Option.isSome_iff_exists.1 (logRead_isSome hv.2.1 hcase hacc)
  obtain ⟨record', _, _, _, _, hcs, hrec', -, hver⟩ := sent_verified hv.2.1 hcase hacc
  have hrr : record' = text := Option.some.inj (hrec'.symm.trans hrec)
  rw [hrr] at hver
  obtain ⟨J, hJ, hrs⟩ := sent_records hver
  obtain ⟨rs, _, _, _, _, he, -, -, -, -, hsr, -⟩ := hver
  obtain ⟨-, -, -, hξ, hdr, _, _, hck⟩ := hacc
  obtain ⟨key, hch, -⟩ := source_of hdr
  obtain ⟨ts, htr, hth⟩ := trees_of_checkBlobsAt hck
  obtain ⟨_, _, htt⟩ := treesOf_ok htr
  obtain ⟨ls, lines, hls0, hls, hct⟩ := trustedTreesOf_ok htt
  have hread : readLog dr.label c.digest [J] ξ lines = some lr := by
    rw [← hlr, logRead_eq hcs hch hξ (linesOf_of hls0 hls), hrs, Option.bind_some]
  have hlog := hl (lr.map Binding.Read.ofV2) (by
    show lr.map Binding.Read.ofV2 ∈ (logRead x.sent).map (·.map Binding.Read.ofV2)
    rw [hlr]
    exact Option.mem_def.2 rfl)
  have h1 : ts.size = 1 := by
    rw [LinkTrees.treesHeld_size hth, he]
    try rfl
  obtain ⟨gb, hgb, hgu⟩ := sessionRecord_blob he hsr
  have hgb' : ((_root_.Core.Blobs.ofArray x.sent.blobs).get rs.record s!"session {rs.name}'s record").toOption =
      some gb := by
    rw [hgb]
    try rfl
  have hJp : (Flock.JsonParse.parse text).toOption.getD .null = J := by
    rw [hJ]
    try rfl
  obtain ⟨tabsB, hJt⟩ := Trees.records_held hth (r := rs) (by simp [he]) hgb' hgu hJp
  exact ⟨dr.label, c.digest, ξ, lines, lr, ts, J, tabsB, hlog, hread, hct, h1, hJt, hJ⟩

/-- **An accepted play's trees name their streams distinctly**: where the sampled verdict accepts and the log v2's
check read is the device's, each `Hello` of the device's log names distinct streams, as v2's read of it does
(`verdict_read`, `readLog_nodup`). -/
theorem verdict_nodup {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} {x : Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ) (hv : V.Verdict k .sampled x)
    (hl : Binding.Logged Binding.readsV2 x) {text : String} (hrec : sentRecord x.sent = some text) :
    ∀ sp ∈ helloSpecs x.log, sp.streams.toList.Nodup := by
  obtain ⟨_, _, _, _, lr, _, _, _, hV, hread, hct, -⟩ := verdict_read hA hθ hv hl hrec
  have hlg : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) lr x.log :=
    (List.forall₂_map_left_iff.1 hV).imp fun _ _ h => ⟨h.1, h.2.1⟩
  rw [← helloSpecs_congr hlg]
  exact readLog_nodup hct hread

/-- **An accepted play's log has an opening**: where the sampled verdict accepts and the log v2's check read is the
device's, the record v2 parsed gives its link points' coins (`LinkTrees.link_some`), which the release log opens on a
line (`Trees.open_held`) that v2's read reads as an opening (`readLog_opens`), whose frame the device's log has. -/
theorem verdict_opens {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} {x : Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ) (hv : V.Verdict k .sampled x)
    (hl : Binding.Logged Binding.readsV2 x) {text : String} (hrec : sentRecord x.sent = some text) :
    ∃ L₁ r L₂ i s n, x.log = L₁ ++ r :: L₂ ∧ r.frame = .opening i s n := by
  obtain ⟨_, _, _, _, lr, _, _, _, hV, hread, hct, -, hJt, -⟩ := verdict_read hA hθ hv hl hrec
  have hlg : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) lr x.log :=
    (List.forall₂_map_left_iff.1 hV).imp fun _ _ h => ⟨h.1, h.2.1⟩
  obtain ⟨cs, hcs⟩ := LinkTrees.link_some hJt
  obtain ⟨l, hlm, _, _, htl, _, hev, hTE, -⟩ := Trees.open_held hct hJt hcs
  obtain ⟨y, hy, i, s, n, hyo⟩ := readLog_opens hread hlm htl hev hTE
  obtain ⟨q, hq, hyq⟩ := forall₂_mem_left hlg y hy
  obtain ⟨L₁, L₂, he⟩ := List.append_of_mem hq
  exact ⟨L₁, q, L₂, i, s, n, he, hyq.1.symm.trans hyo⟩

/-- **The session reads a round**: its link's (`Live.rounds_eq`). -/
theorem Live.rounds_pos (A : Live) : 0 < A.rounds := by
  rw [A.rounds_eq]
  exact Nat.succ_pos _

/-- **The rep rounds at a `ReachesOff` witness are the record's, at the session's layout** (`MovesOf`'s clause (iv) as
`MovesReach` reads it): at the readings and the decoders at the session's layout (`Live.layout`, `Live.tables`), and
at the sample-coupled coin map under `fieldEnc`, an outcome
reached at a widened sample `ω` and link points `pts` off the link's and the rounds' empty classes, where the play
beside it is accepted by the sampled verdict, misses nowhere, and the log v2's check read is the device's, is the
writer's at `ω`, `pts` and rep rounds that are those of the session v2's record decodes to (`repRounds`) at the session
`--zk` sets up at `A`, at every table and rep. It takes, of the device's log, that its first opening is the session
stream's and that it opens only the record's rounds (`OpenedRecorded`); and of v2's read and the record, the
statement `rep_rounds`'s premise `hans` is an instance of (`verdict_read`, `SetsUp.setup_one`, `setup_sha512Digests`,
`firstStop_reps`). That the log has an opening and that its trees name their streams distinctly come from the verdict
(`verdict_opens`, `verdict_nodup`), its premises `hsome` and `hkey` from the record `sessionTreeReason` accepts
(`recordRound_isSome`, `keyAt_of_record`), and the layout's slot for each round the log opens of a stream of the
record's tree from the session's coin tree, which `Record.decode` holds the record's to (`SetsUp.covers_logged`). -/
theorem firstStop_reps_verdict {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (reg : Option Claims.Msg → A.Reg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading A.rounds} (hrd : rd = readingAt A.tables A.layout record recordDigest)
    {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {label : String}
    {after : ByteArray} {m : ℕ} {κ : A.Ω → Fin m → Fin 512 → Bool} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP}
    (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ) (ω : (A.extend M.widening).L.Ω)
    (pts : Fin 2 → Fin A.mP → 𝔽) {o : A.Out × CoinTape (V.game θ F c) σ}
    (hL0 : ¬ linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1) (firstStop V θ F c A rd σ M ω).g' pts)
    (hr : RoundsReach D M A.dg (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D))
      (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
      (fun t => (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D) t))
      (D.root (firstStop V θ F c A rd σ M ω).modes) pts A.N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2)) (hn : NoMiss V θ F c rd σ o.2)
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (ht : TablesSent (CoinTape.run (V.game θ F c) σ o.2).sent)
    (hfirst : ∀ pre r post, (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: post → ∀ i s n,
      r.frame = .opening i s n → (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some _root_.Spot.Challenger.TREE_SESSION)
    (hO : ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text →
      Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log)
    (hans : ∀ {label : String} {after : ByteArray} {J : Json} {ξ : ByteArray} {lines : List String}
      {pre post : List Release} {y : Release} {trees : Array LoggedTree} {i s n : ℕ} {name : String}
      {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {st : Flock.StreamRec},
      readLog label after [J] ξ lines = some (pre ++ y :: post) → coinTreesOf lines = .ok trees →
      trees.size = 1 → ∀ {tabsB : Array (String × Option ℕ)}, sessionTreeReason trees tabsB J = none →
      y.frame = .opening i s n → ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == _root_.Spot.Challenger.TREE_SESSION) = false →
      Flock.Record.decode spec record = .ok sess → Flock.JsonParse.parse record = .ok J →
      spec.sha512Digests = true → sess.streams.find? (·.name == name) = some st →
      openedOf pre i s < st.rounds.size → y.answers = st.rounds[openedOf pre i s]!.sha) :
    ∃ reps : Fin (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) → Fin 2 → List Flock.Round,
      o.1 = (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)
          ⟨D.root (firstStop V θ F c A rd σ M ω).modes, pts, reps, D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩) ∧
      ∀ text proofs spec sts sess, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text →
        sentProofs (CoinTape.run (V.game θ F c) σ o.2).sent = some proofs →
        FlockSoundness.Discharge.Exec.setupOfP (FlockSoundness.Discharge.ZkExec.zkInputs A.I) A.own text =
          .ok (spec, sts) →
        Flock.Record.decode spec text = .ok sess → ∀ j r, reps j r = repRounds spec sess j.val r.val := by
  have hθ := (play_ends V θ F c σ o.2).1
  obtain ⟨text₀, proofs₀, spec₀, sts₀, hrec₀, -, hs₀, spec', sts', hs', -, hver⟩ :=
    accepts_sent hA hθ hv (filesOn_play hA hDg hv ht)
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hs₀.symm.trans hs'))
  obtain ⟨sess₀, hd₀⟩ := zkVerify_decode hver
  obtain ⟨label', after', ξ, lines, lr, trees, Jr, tabsB, hV, hread, hct, h1, hJt, hJ⟩ :=
    verdict_read hA hθ hv hl hrec₀
  obtain ⟨htab, -, hret, -, hdg⟩ := hA.setup_one hs₀
  have h512 := setup_sha512Digests A.hTags.1 (statement_hm96Rows hA.tags) hA.count hs₀
  have h0 : 0 < spec₀.tables.size := by
    rw [htab]
    exact Nat.one_pos
  have hlg : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) lr
      (CoinTape.run (V.game θ F c) σ o.2).log :=
    (List.forall₂_map_left_iff.1 hV).imp fun _ _ h => ⟨h.1, h.2.1⟩
  have ht0 : 0 < A.J ∅ ByteArray.empty := by
    rw [hA.J_eq]
    exact Nat.one_pos
  obtain ⟨reps, ho, hreps⟩ := firstStop_reps reg A.layout_injective A.tables_nodup record recordDigest hA.layout_lt
    hA.J_eq (A.layout_link A.rounds_pos) A.rounds_eq.le hrd hD hM ω pts hL0 hr hn (verdict_opens hA hθ hv hl hrec₀)
    (verdict_nodup hA hθ hv hl hrec₀) hfirst hV hread hct h1 hJt
    (fun hk s r => hA.covers_logged hs₀ hd₀ hJ hct hread hlg (logServed_run V θ F c σ o.2) hJt hk ⟨0, ht0⟩ r s)
    hrec₀ hJ hd₀ hver hret h512 hdg htab.symm h0 (hO _ _ hrec₀ hJ) (recordRound_isSome hJt)
    (fun _ hpos => keyAt_of_record hct hJt hpos)
    (fun hlr hy hO' hn' hst hk => hans (hlr ▸ hread) hct h1 hJt hy hO' hn' hd₀ hJ h512 hst hk)
  refine ⟨reps, ho, fun text proofs spec sts sess hrec _ hs hd => ?_⟩
  obtain rfl : text = text₀ := Option.some.inj (hrec.symm.trans hrec₀)
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hs.symm.trans hs₀))
  obtain rfl : sess = sess₀ := Except.ok.inj (hd.symm.trans hd₀)
  exact hreps

end Verity.Sampled
