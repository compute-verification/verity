module

public import Proofs.Flock.Claims.Complete.ZkTables.RepBytes
public import Proofs.Flock.Claims.Complete.Zk.Walk
public import Proofs.Flock.Claims.Complete.Zk.Json.RecordSize
public import Proofs.Flock.Claims.Complete.Zk.Sim
public import Proofs.Flock.Claims.Complete.Ready
import Proofs.Flock.Claims.Complete.Zk.RepRounds
import Proofs.Flock.Claims.Complete.Zk.ProverOk
import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.Claims.Complete.Json.Size
import Proofs.Flock.Claims.Complete.ZkTables.View

/-!
# The honest masked record's length, from its statement

At one table whose statement has `25 ≤ m ≤ 35`, the `hm96` leaf, `kLog ≤ m`, at most `2^10` regions and at most two
points, the honest masked session's printed record weighs at most `recBound` of its spec's text, whatever the witness,
tape and coins.

* `repOk_at`: each rep's rounds are nonempty and of at most `2^16` coins (`sessionPZk_walk`'s `hR`), from the
  statement alone.
* `streamsLe_at`: the two streams the walk returns have `repRoundsT` rounds (`rep_rounds_of`) of at most `|d| + 2^16`
  bytes (`rep_bytes`), `d` the stream's domain.
* `record_size_le`: the record weighs at most `recBound` (`jsz_recordObjZk_le`, `printJ_size_le`).
* `recordFitsZk_of`: `RecordFitsZkAt` at one table wherever `recBound` is under `2^64` (`sessionPZk_sim`).
-/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy With FlockComplete.Replay
open Lean (Json)

variable {H : Claims.Msg → Claims.Msg}

/-- The weight `record_size_le` bounds the honest masked record by: its hello `hz`, unit sample, coin tree, `σ`,
table name `t`, `Q` coordinates a point, domains of at most `D` bytes and stream names of at most `L`. -/
def recBound (spec : SessionSpec) (hz t : String) (ct : CoinTree.Spec) (Q D L : ℕ) : ℕ :=
  18 * hz.length + jsz (spec.unitSample.getD Json.null) + jsz ct.toJson + 72 * spec.sigma.size + 36 * 16384 +
    36 * t.length + spec.points * (579 * Q + 2 + 1) + 579 * spec.points +
    spec.repStreams.size * (18 * L + 5000 * (579 * 65536 + 36 * (D + 65536) + 4747) + 4500 + 1) +
    spec.repStreams.size * (18 * L + 2900 + 1) + 30000

/-- **Each masked rep's rounds are nonempty and of at most `2^16` coins**, at a statement with `25 ≤ m ≤ 35`, at most
`2^10` regions and at most two points: `sessionPZk_walk`'s `hR`. -/
theorem repOk_at (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (st : Flock.Stmt)
    (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hreg : st.regions.size ≤ 1024) (hpt : spec.points ≤ 2) :
    ∀ v, Zk.schedule (Setup.ofCircuit st).wideLeaf (Setup.ofCircuit st).m = .ok v →
      ∀ rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce prior ws,
      (Setup.ofCircuit st).extra (JsonRT.ptsOf spec (coinOf (resp 0))) = .ok extras →
      Ret ((repPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2 rep dom tr tb cm₀
        extras pads ic extra lanePad salt nonce prior ws).run (Transcript.init #[] dom))
        fun pt => JsonRT.RepOk pt.2.rounds := by
  intro v hv rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce prior ws hex
  have hx : extras.size = st.regions.size * spec.points := by
    rw [JsonRT.extraClaims_size hex]
    simp [JsonRT.ptsOf]
  have hrp : st.regions.size * spec.points ≤ 2048 := Nat.mul_le_mul hreg hpt
  have hm : (Setup.ofCircuit st).m = st.m := rfl
  rw [hm] at hv
  obtain ⟨hn, hq⟩ := JsonRT.zkSchedule_small h25 h35 hv
  rintro l ⟨p, t'⟩ l' hrun
  obtain ⟨⟨hs, hsz⟩, -⟩ := JsonRT.repPZk_grows resp _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (B := 65536) (by decide)
    (by simp [Setup.ofCircuit]; omega) (by omega) hn hq _ l _ _ _ hrun
  exact ⟨show 0 < t'.rounds.size by simp [Transcript.init] at hsz; omega, hs (by simp [CoinsLe, Transcript.init])⟩

/-- **The masked session's two streams fit `StreamsLe`**: rep `j`'s stream is its rounds (`RepRunZk`), `repRoundsT`
of them (`rep_rounds_of`), each of at most `2^16` coins (`RoundsOk`) and `|d| + 2^16` bytes (`rep_bytes`). -/
theorem streamsLe_at (h64 : Claims.Out64 H) {spec : SessionSpec} {t : String} (ht : spec.tables = #[t])
    (st : Flock.Stmt) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hms : st.merkle = MerkleScheme.hm96Sha512)
    (hkm : st.c.kLog ≤ st.m) (hpm : spec.mPts = st.mPts) {D : ℕ} (hD : ∀ rep < 2, (spec.domain t rep).toUTF8.size ≤ D)
    (hDl : D < 10 ^ 19) {k : ℕ} (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit) (e : Array Claim) (tr : Trace)
    (tape : TapeZk) (streams : Array (Array Round))
    (hv : Zk.schedule (Setup.ofCircuit st).wideLeaf (Setup.ofCircuit st).m = .ok v)
    (hcm : commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
      (v.1.levels.getD 0 default) v.2 (packed tr (Setup.ofCircuit st).m) (tape.extra 0) (tape.lanePad 0)
      (tape.salt 0 0 0) = .ok cm)
    (he : (Setup.ofCircuit st).extra (JsonRT.ptsOf spec (coinOf (Claims.coinSeq r 0).bytes)) = .ok e)
    (hss : streams.size = 2) (hrs : ∀ rs ∈ streams, JsonRT.RoundsOk rs)
    (hreps : ∀ j < 2, ∃ q, JsonRT.RepRunZk (fun j => (Claims.coinSeq r j).bytes) (Claims.hOf H) spec
      (Setup.ofCircuit st) v tr cm e tape j streams[j]! q) :
    StreamsLe streams (repRoundsT st v e) (579 * 65536 + 36 * (D + 65536)) := by
  intro rs hmem
  obtain ⟨j, hj, rfl⟩ : ∃ j, j < 2 ∧ streams[j]! = rs := by
    obtain ⟨j, hj, hje⟩ := Array.mem_iff_getElem.1 hmem
    exact ⟨j, hss ▸ hj, by rw [getElem!_pos streams j hj]; exact hje⟩
  obtain ⟨q, p, ic, l, t', l', prior, ws, hic, -, hrst, hgo⟩ := hreps j hj
  have hin : RepIn H st tr tape 0 v cm (JsonRT.ptsOf spec (coinOf (Claims.coinSeq r 0).bytes)) e j ic := by
    refine ⟨hv, hcm, fun a ha => ?_, he, hic⟩
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 ha
    simp [hpm]
  have ht0 : spec.tables[0]! = t := by rw [ht]; rfl
  have hd : (spec.domain spec.tables[0]! j).toUTF8.size ≤ D := by rw [ht0]; exact hD j hj
  have hR := rep_rounds_of L0.verifier h64 st tr h25 h35 hms hkm tape 0 (spec.domain spec.tables[0]! j).toUTF8 r v
    cm _ e j ic hin hj prior ws l p t' l' hgo
  have hB := rep_bytes h64 st tr h25 h35 hms hkm tape 0 (spec.domain spec.tables[0]! j).toUTF8 r v cm _ e j ic hin hj
    prior ws l p t' l' hgo
  obtain ⟨-, hok⟩ := hrs _ hmem
  rw [hrst] at hok ⊢
  refine ⟨hR.le, fun x hx => ?_⟩
  obtain ⟨h1, h2, h3⟩ := hok x hx
  have hb := hB x hx
  refine ⟨by omega, by omega, ?_⟩
  rw [h3]
  omega

/-- **The honest masked record weighs at most `recBound`** at one table whose statement has `25 ≤ m ≤ 35`, the `hm96`
leaf, `kLog ≤ m`, at most `2^10` regions and at most two points of its `mPts` coordinates. -/
theorem record_size_le (h64 : Claims.Out64 H) {spec : SessionSpec} {t : String} (ht : spec.tables = #[t])
    {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json}
    (hown : Flock.JsonParse.parse spec.hello = .ok own) {hz : String} (hc : canon (JsonRT.helloJ own) = .ok hz)
    (st : Flock.Stmt) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hms : st.merkle = MerkleScheme.hm96Sha512)
    (hkm : st.c.kLog ≤ st.m) (hreg : st.regions.size ≤ 1024) (hpm : spec.mPts = st.mPts) (hpt : spec.points ≤ 2)
    {D L : ℕ} (hD : ∀ rep < 2, (spec.domain t rep).toUTF8.size ≤ D) (hDl : D < 10 ^ 19) (hL : NamesLe spec L)
    (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk) {k : ℕ} (r : Fin k → Coin)
    {rec : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H)) spec
      #[Setup.ofCircuit st] words trs tape).go #[] = (.ok rec, bodies)) :
    rec.1.utf8ByteSize ≤ recBound spec hz t ct st.mPts D L := by
  have hmerk : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := hms
  obtain ⟨v, cm, e, streams, hv, hcm, he, hr1, -, -, hss, -, hrs, -, hreps⟩ :=
    JsonRT.sessionPZk_walk H _ ht hct hown hc (Setup.ofCircuit st) words trs tape
      (fun v hv g extra lanePad salt cm hcm => by
        rw [hmerk] at hcm
        exact JsonRT.commitPZk_cap (JsonRT.hOf_size h64) h25 h35 hv hcm)
      (repOk_at H _ spec st h25 h35 hreg hpt) hrun
  have hs := streamsLe_at h64 ht st h25 h35 hms hkm hpm hD hDl r v cm e trs[0]! tape streams hv hcm he hss hrs
    fun j hj => ⟨_, hreps j hj⟩
  have hm : (Setup.ofCircuit st).m = st.m := rfl
  have hR : repRoundsT st v e ≤ 5000 := by
    rw [hm] at hv
    obtain ⟨-, -, hk0, -⟩ := L0.zkSchedule_sizes h25 h35 hv
    have hkn := (L0.zkSch_facts h25 h35 hv).2.2.2.1
    have hn := ZkTables.zkSchedule_n hv
    have hx : e.size = st.regions.size * spec.points := by
      rw [JsonRT.extraClaims_size he]
      simp [JsonRT.ptsOf]
    have hrp : st.regions.size * spec.points ≤ 2048 := Nat.mul_le_mul hreg hpt
    unfold repRoundsT
    omega
  have hs' : StreamsLe streams 5000 (579 * 65536 + 36 * (D + 65536)) := fun rs h =>
    ⟨(hs rs h).1.trans hR, (hs rs h).2⟩
  have hp : ∀ p ∈ JsonRT.ptsOf spec (coinOf (Claims.coinSeq r 0).bytes), p.size ≤ st.mPts := by
    intro a ha
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 ha
    simp [hpm]
  have hJ := jsz_recordObjZk_le (JsonRT.hOf_size h64) ht hz ct (flat cm.cap) hp (by decide) hs' hL rec.2
  have hps : (JsonRT.ptsOf spec (coinOf (Claims.coinSeq r 0).bytes)).size = spec.points := by
    simp [JsonRT.ptsOf]
  rw [hps] at hJ
  rw [hmerk] at hcm
  have hcap : (flat cm.cap).size ≤ 16384 := capBytes_le h64 h25 h35 hv hcm
  have hP := printJ_size_le (JsonRT.recordObjZk spec (Claims.hOf H) hz ct #[flat cm.cap]
    (JsonRT.ptsOf spec (coinOf (Claims.coinSeq r 0).bytes)) streams rec.2)
  rw [hr1]
  unfold recBound
  omega

/-- **The honest masked record fits its length word** at one table whose statement has `25 ≤ m ≤ 35`, the `hm96` leaf,
`kLog ≤ m`, at most `2^10` regions and at most two points, wherever `recBound` is under `2^64`. -/
theorem recordFitsZk_of (h64 : Claims.Out64 H) {x : Claims.Msg} {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json}
    (hown : Flock.JsonParse.parse spec.hello = .ok own) {hz : String} (hc : canon (JsonRT.helloJ own) = .ok hz)
    (st : Flock.Stmt) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hms : st.merkle = MerkleScheme.hm96Sha512)
    (hkm : st.c.kLog ≤ st.m) (hreg : st.regions.size ≤ 1024) (hpm : spec.mPts = st.mPts) (hpt : spec.points ≤ 2)
    {D L : ℕ} (hD : ∀ rep < 2, (spec.domain t rep).toUTF8.size ≤ D) (hDl : D < 10 ^ 19) (hL : NamesLe spec L)
    (hfit : recBound spec hz t ct st.mPts D L < 2 ^ 64) :
    RecordFitsZkAt H x spec #[st] := by
  intro w ω r record proofs hsim
  obtain ⟨bs, h1, -⟩ := sessionPZk_sim H (fun j => (Claims.coinSeq r j).bytes) spec (tabsOf #[st])
    (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω #[]
  have h2 := congrArg Prod.snd h1
  simp only [List.size_toArray, List.length_nil, Array.size_empty] at h2
  rw [hsim] at h2
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  have hrun : (sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H)) spec
      #[Setup.ofCircuit st] (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω).go #[] =
      (.ok (record, proofs), ((sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H))
        spec #[Setup.ofCircuit st] (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω).go #[]).2) := by
    rw [← htabs, h2]
  rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]
  exact lt_of_le_of_lt (record_size_le h64 ht hct hown hc st h25 h35 hms hkm hreg hpm hpt hD hDl hL _ _ ω r hrun) hfit

end FlockComplete.ZkTables
