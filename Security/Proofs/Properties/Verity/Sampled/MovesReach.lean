module

public import Proofs.Properties.Verity.Sampled.RoundsRead
public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Verity.Sampled.LinkPts

/-!
# `MovesReach`, clause by clause

On the joint law's support off the held-off events (`ReachesOff`), each outcome is the writer's at the widened sample's
draw, the decoded registration and a live session's outcome (`reachesOff_out`). What v2's files decode to in the play on
the tape beside it, clause by clause of `MovesOf`:

* **The play from the first stop** (`ReachesOff.out_reaches`): the play beside an outcome is one the first stop
  reaches (`firstStop_playReaches`), at the widened sample and the link points the outcome is the writer's at.
* **The record's draw** (`ReachesOff.recordDraw`, clause (i)): at the sample-coupled coin map, where the play's sampled
  verdict accepts, the log v2 read is the device's, v2 reads the sample's coins of the family `label` answering `after`,
  and every release of that family precedes every opening, the record decodes to the session's draw file at the
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
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open SampledProofs.Challenger
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

/-- **The record's draw on the joint law's support** (`MovesOf`'s clause (i)): at the sample-coupled coin map, an
outcome on the joint law's support off `BL` and `B` is the writer's at the widened sample `ω`, and where the play beside
it ends where the sampled verdict accepts, the log v2 read is the device's, v2 reads the sample's coins of the family
`label` answering `after`, and every release of that family precedes every opening, its record decodes to the session's
draw file at `ω`'s sample (`ReachesOff.out_reaches`, `recordDraw_firstStop`). -/
theorem ReachesOff.recordDraw (e : 𝔽 ≃ (Fin 128 → Bool)) {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {k : Verity.Registration.Composite} {A : Live} {rd : Reading A.rounds}
    {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg} {BL : LinkHold A.mP} {B : RoundHold}
    {label : String} {after : ByteArray} {m : ℕ} {κ : A.Ω → Fin m → Fin 512 → Bool}
    (hA : SetsUp V θ F A) (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) (hH : DrawHeld k)
    (hrd : PicksOpenings rd) (hm : _root_.SampledProofs.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 (κ ω) 0))
    (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) {o : A.Out × CoinTape (V.game θ F c) σ}
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
  exact recordDraw_firstStop e hA ht hR hH hrd hm hsrc hfix ω hreach hv hl hN hP hrec

/-! ## The rep rounds, stop by stop -/

/-- **The rounds the record gives a decoded stream are its rounds**: where `record` decodes to `sess` and parses as `J`,
`recordRounds` reads of `J`'s stream `name`, not the session stream, the count of rounds of `sess`'s stream so named. -/
theorem recordRounds_decode {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {J : Json}
    {name : String} {y : Flock.StreamRec} (hd : Flock.Record.decode spec record = .ok sess)
    (hJ : Flock.JsonParse.parse record = .ok J)
    (hn : (name == _root_.SampledProofs.Challenger.TREE_SESSION) = false)
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
    (h1 : trees.size = 1) (hJt : sessionTreeReason trees Jr = none) {text : String} {spec : Flock.SessionSpec}
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
      (name == _root_.SampledProofs.Challenger.TREE_SESSION) = false →
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
    rwa [streamOpenings_eq_openedOf, hk, hrr] at h
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
    change sha512Msg bb.data.toList = b.data.toList
    rw [← hya, hyans, hsha bb hbb, hdg512]
    all_goals rfl)
  have hwb : w = bb.data.toList := Option.some.inj (hw.symm.trans hmode)
  subst hwb
  have hbits : Verity.bitsOfBytes b' = roundBits fieldEnc cs := by
    by_cases hc : y.coins = []
    · have hrc : _root_.SampledProofs.Challenger.recordCoins? (.arr cs') = some b' := hrr'.symm.trans hb'
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
as `roundBits fieldEnc`), the device's log `Lg` opens the session stream first and has a slot for each opening of a
rep stream (`hcovL`), and v2's read and the record are as `stop_round` takes them, with each stream the record gives
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
    (hcovL : ∀ s, ((helloSpecs (CoinTape.run (V.game θ F c) σ o.2).log)[i]?.bind fun sp => sp.streams[s]?) =
        tables[t.val]?.map (repStreamName · r.val) →
      ∀ k < openedOf (CoinTape.run (V.game θ F c) σ o.2).log i s, ∃ j, L j = .rep t.val r k)
    {label : String} {after ξ : ByteArray} {lines : List String} {lr : List Release} {trees : Array LoggedTree}
    {Jr : Json}
    (hV : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (CoinTape.run (V.game θ F c) σ o.2).log)
    (hread : readLog label after [Jr] ξ lines = some lr) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) (hJt : sessionTreeReason trees Jr = none) {text : String} {spec : Flock.SessionSpec}
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
      (name == _root_.SampledProofs.Challenger.TREE_SESSION) = false →
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
      (hF.imp fun _ _ h => ⟨h.1, h.2.1⟩) hgt hpw hcov hS hnd hL₀ hr₀ h₀ hses ⟨0, htv⟩ r hs (hcovL si hs)
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
(`SampleLinkFirst`'s second conjunct), the layout has a slot for each opening of a rep stream, and v2's read and the
record are as `rep_rounds` takes them, those rep rounds are the record's (`repRounds`), at every table and rep
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
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some _root_.SampledProofs.Challenger.TREE_SESSION)
    (hcovL : ∀ i s (r : Fin 2),
      ((helloSpecs (CoinTape.run (V.game θ F c) σ o.2).log)[i]?.bind fun sp => sp.streams[s]?) =
        tables[0]?.map (repStreamName · r.val) →
      ∀ k < openedOf (CoinTape.run (V.game θ F c) σ o.2).log i s, ∃ j, L j = .rep 0 r k)
    {label' : String} {after' ξ : ByteArray} {lines : List String} {lr : List Release} {trees : Array LoggedTree}
    {Jr : Json}
    (hV : List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) (CoinTape.run (V.game θ F c) σ o.2).log)
    (hread : readLog label' after' [Jr] ξ lines = some lr) (hct : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) (hJt : sessionTreeReason trees Jr = none) {text : String} {spec : Flock.SessionSpec}
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
      (name == _root_.SampledProofs.Challenger.TREE_SESSION) = false →
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
  exact rep_rounds reg hL ht record recordDigest hn hF hI hgt hpw hcov hS hnd hL₀ rfl h₀ hses j htz r
    (fun s hs k hk => by
      rw [htz] at hs ⊢
      exact hcovL i s r hs k hk) hV hread hct h1 hJt hrec hJ hd hv hret h512 hdg htab h0 hO hsome hkey hans

end Verity.Sampled
