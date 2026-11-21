module

public import Properties.Poub.Remote.Line.Record
public import Properties.Poub.Remote.Shared
public import Proofs.Properties.Poub.Remote.Schedule
public import Proofs.Properties.Poub.Remote.Window
public import Proofs.Properties.Poub.Remote.SubprotocolWitnesses
public import Proofs.Flock.Bridge.QsRows.FirstReads
import all Init.Data.Array.Basic
import all Init.Data.Array.QSort.Basic
import all Init.Data.Range.Basic

/-!
# An accepted session of the remote audit, without evaluating SHA-512

`record_verified`: the Lean verifier accepts a record (`Properties.Poub.Line.Verified`). The node stores one
registered block under the deployed codec, and runs one window of one pass under one sample, at one window a sample. The
session starts, receives the pass's root over the whole store, ends the window, samples its one receipt and verifies its
opening. The sample of one receipt among one is position 0 on a sample's coins of 128 zero bytes, `Session.sampleBytes 1 1`
(`receiptSample_one`). The opening is
`Schedule`'s zero-block opening at one block: its leaf is the block's receipt alone, and the receipt and payload trees
each have one leaf, so their roots are the leaves' digests (`zeroRoot`). The session's steps read the roots and coins
only through their sizes (`run_accepts`, over any registration of one block, any first coin and any root of 64 bytes), so no
SHA-512 is evaluated.

`lineHeld_accepted`: `Properties.Poub.Line.LineHeld`'s premises hold together with an accepted play. The record's
one window is one group, its coin released at 0 and its root arriving at 1 (`groups_one`), so pass 0 runs over `[0, 1)`
inside its span (`honestSchedule_one`), and `Subprotocol.lineHeldAtSubprotocol_honest` gives every premise. Stated with
`Shared.run` as `Record.run` applies it, that is `LineHeld`'s honest witness
(`Properties.Poub.Line.SecurityProofs.lineHeld_honest`).
The same session's window, sample and run are the honest witnesses of the window's properties
(`AcceptChecksOpenings`, `AcceptOpensSamples`, `AcceptSamplesOpenable`), the sample's (`SampleIsSubset`,
`Shared.AcceptSamplesOpenable`) and `AppendsInOrder`, beside their nontrivial witnesses (the last section).
-/

@[expose] public section

open _root_.Poub

open private Poub.Session.live Poub.Session.cover from Poub.Session

namespace Properties.Poub.Line.Honest

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sample
  ResourceAccounting.Remote.Sample.Assumptions

attribute [local irreducible] zeroRoot

/-! ## The sample of one receipt among one -/

theorem beNat_lt_aux (b : ByteArray) (off : ℕ) : ∀ (l : List ℕ) (acc k : ℕ), acc < 256 ^ k →
    l.foldl (fun acc i => acc * 256 + (b.get! (off + i)).toNat) acc < 256 ^ (k + l.length)
  | [], acc, k, h => by simpa using h
  | i :: l, acc, k, h => by
    rw [List.foldl_cons, List.length_cons, show k + (l.length + 1) = (k + 1) + l.length by omega]
    apply beNat_lt_aux b off l
    have hb : (b.get! (off + i)).toNat < 256 := (b.get! (off + i)).toBitVec.isLt
    rw [Nat.pow_succ]
    omega

theorem beNat_lt (b : ByteArray) (off len : ℕ) : Flock.Draw.beNat b off len < 256 ^ len := by
  have h := beNat_lt_aux b off (List.range len) 0 0 (by decide)
  simpa [Flock.Draw.beNat] using h

theorem width_one : Flock.Draw.width 1 = 9 := by
  have h : Nat.log2 1 = 0 := (Nat.log2_eq_iff (by decide)).2 ⟨by decide, by decide⟩
  simp [Flock.Draw.width, h]

/-- One uniform draw in `[0, 1)` from a stream of 82 bytes is 0, at its first 9-byte word, below `2^72`. -/
theorem uniform_one (s : ByteArray) (hs : s.size = 82) : Flock.Draw.uniform 1 s (s.size + 1) 0 = some (0, 9) := by
  have hx : Flock.Draw.beNat s 0 9 < 4722366482869645213696 := by
    have h := beNat_lt s 0 9
    norm_num at h
    exact h
  rw [hs, Flock.Draw.uniform]
  simp only [width_one, hs]
  simp [hx, Nat.mod_one]

theorem qsort_single (a : ℕ) (lt : ℕ → ℕ → Bool) : #[a].qsort lt 0 0 = #[a] := by
  unfold Array.qsort
  simp only [List.size_toArray, List.length_singleton, Nat.one_ne_zero, ↓reduceDIte, Nat.sub_self, Nat.min_self,
    Nat.max_self]
  rw [Array.qsort.sort]
  simp

theorem qsort_map_zero (F : ℕ → ℕ) (lt : ℕ → ℕ → Bool) (h : F 0 = 0) : (#[0].map F).qsort lt 0 0 = #[0] := by
  rw [Array.map_singleton, h, qsort_single]

theorem subset_one (s : ByteArray) {p : ℕ} (h : Flock.Draw.uniform 1 s (s.size + 1) 0 = some (0, p)) :
    Flock.Draw.subset 1 1 s = some #[0] := by
  unfold Flock.Draw.subset
  simp [Id.run, h, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  rw [show Array.range 1 = #[0] from rfl, qsort_map_zero]
  · rfl
  · simp [Std.HashMap.getD_insert_self]

/-- The subset draw of one unit among one, on 128 zero bytes, reads its first stream's `2 · width 1 + 64 = 82` bytes
and samples unit 0. -/
theorem sampleUnits_one : Spot.Draw.sampleUnits (.subset 1) 1 (zeros 128) = .ok (#[0], 82) := by
  have hz : ((zeros 128).extract 0 (0 + 82)).size = 82 := by simp [ByteArray.size_extract, zeros_size]
  have h := Properties.Spot.SecurityProofs.onWith_take_first (zeros 128) 82 (Flock.Draw.subset 1 1) 0 #[0]
    (by rw [zeros_size]; decide) (subset_one _ (by rw [uniform_one _ hz]))
  unfold Spot.Draw.sampleUnits Flock.Draw.drawWith
  simp only [width_one]
  simp [h]
  rfl

theorem bisectRight_one : bisectRight #[1] 0 = 0 := by
  first
    | rfl
    | decide
    | decide +kernel

/-- A window of one pass of one receipt samples that receipt on a sample's coins of 128 zero bytes, the session's
`sampleBytes 1 1`. -/
theorem receiptSample_one : receiptSample 1 (zeros 128) #[1] = some #[(0, 0)] := by
  unfold receiptSample receiptSampleRead
  simp [sampleUnits_one, bisectRight_one]
  exact ⟨82, rfl⟩

/-! ## The window: one block, one pass, its opening -/

theorem plan_length : (Plan.whole 1).length = 1 := rfl

theorem leafReceipt_one {R : ByteArray} (hR : R.size = 64) : leafReceipt R 0 1 = some R := by
  unfold leafReceipt
  dsimp only
  split
  · rename_i h
    rw [hR] at h
    exact absurd h (by decide)
  · show some (R.extract 0 64) = some R
    rw [← hR, ByteArray.extract_zero_size]

/-- The opening of position 0 of pass 0 under challenge `c`. -/
def opening1 (reg : Registration) (c : ByteArray) : Opening :=
  let s := seed c 0 reg.digest
  ⟨0, 0, zeros Codec.BLOCK_BYTES, leaf1 reg s, zeros SALT_BYTES, zeroPath (treeDomain s (Plan.whole 1)),
    zeros SALT_BYTES, zeroPath reg.payloadDomain, .empty, .empty, none⟩

theorem checks_one (reg : Registration) (c : ByteArray) (hB : reg.blocks = 1) (hk : reg.key.codec = honestCodec)
    (hroot : reg.payloadRoot = zeroRoot reg.payloadDomain 0 (hidden (zeroPayload reg.key) (zeros SALT_BYTES))) :
    checkOpening reg reg.digest #[] (window1 reg c (root1 reg (seed c 0 reg.digest)))
      (receipterOf reg (window1 reg c (root1 reg (seed c 0 reg.digest))) 0) 0 0 (opening1 reg c) = none := by
  have hs : (treeDomain (seed c 0 reg.digest) (Plan.whole 1)).start ≤ 0 := Nat.le_refl 0
  have hp : 0 < (treeDomain (seed c 0 reg.digest) (Plan.whole 1)).start +
      (treeDomain (seed c 0 reg.digest) (Plan.whole 1)).count := by
    show 0 < 0 + leaves (Plan.whole 1).length
    rw [plan_length]
    decide
  have hs' : reg.payloadDomain.start ≤ 0 := Nat.le_refl 0
  have hp' : 0 < reg.payloadDomain.start + reg.payloadDomain.count := by
    show 0 < 0 + reg.blocks
    rw [hB]
    decide
  have hb0 : (Plan.whole 1).block 0 = 0 := by
    first
      | rfl
      | decide
      | decide +kernel
  have hj : 0 < reg.key.codec.maxBlocks := by
    rw [hk]
    decide
  have hlt : honestCodec.field.c < 2 ^ honestCodec.field.n :=
    Nat.lt_of_lt_of_le (show honestCodec.field.c < 2 ^ 7 by decide)
      (Nat.pow_le_pow_right (by norm_num) (show 7 ≤ honestCodec.field.n by decide))
  have hfp : 0 < reg.key.codec.field.p := by
    rw [hk]
    exact Nat.sub_pos_of_lt hlt
  refine checkOpening_of (pr := ⟨root1 reg (seed c 0 reg.digest), Plan.whole 1, none⟩) (j := 0)
    (y := leaf1 reg (seed c 0 reg.digest)) (payload := zeroPayload reg.key) rfl rfl rfl (zeros_size _) rfl
    (zeros_size _) (zeros_size _) ?_ hb0 (by rw [hB]; decide) rfl ?_ ?_ (decode_zeroPayload reg.key hj hfp) ?_ rfl rfl
  · intro x hx
    simp only [opening1, zeroPath, Array.mem_append, Array.mem_replicate] at hx
    rcases hx with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> exact zeros_size 64
  · rw [plan_length]
    exact leafReceipt_one (receipt_size _ _ _)
  · unfold verifyLeaf
    exact verifyOpening_zeroRoot (treeDomain (seed c 0 reg.digest) (Plan.whole 1)) 0
      (hidden (leaf1 reg (seed c 0 reg.digest)) (zeros SALT_BYTES)) hs hp
  · rw [hroot]
    exact verifyOpening_zeroRoot reg.payloadDomain 0 (hidden (zeroPayload reg.key) (zeros SALT_BYTES)) hs' hp'

theorem checkWindow_one (reg : Registration) (c : ByteArray) (hB : reg.blocks = 1) (hk : reg.key.codec = honestCodec)
    (hroot : reg.payloadRoot = zeroRoot reg.payloadDomain 0 (hidden (zeroPayload reg.key) (zeros SALT_BYTES))) :
    checkWindow reg reg.digest #[] (window1 reg c (root1 reg (seed c 0 reg.digest))) #[(0, 0)] #[opening1 reg c] =
      .accept := by
  rw [checkWindow_eq]
  simp [rejectAt, checks_one reg c hB hk hroot]

/-! ## The session -/

/-- Start at 0; receive window 0's one root, over the whole store at step 0, at 1; end window 0 at 1; sample it at 2;
verify its opening. -/
def events1 (root : ByteArray) (o : Opening) : Array _root_.Poub.Session.Event :=
  #[.start 0, .receive 0 0 root none none (some 0) 1, .boundary 1, .close 0 2, .verify 0 #[o]]

/-- The session's run accepts, over any registration of one block, any first coin and any root of 64 bytes whose
window's openings `checkWindow` accepts, and its record has one window, of one pass of one block, which drew one
receipt, its sample's coins 128 zero bytes. -/
theorem run_accepts (reg : Registration) (hB : reg.blocks = 1) (c0 root : ByteArray) (hroot : root.size = 64)
    (o : Opening) (hcheck : checkWindow reg reg.digest #[] (window1 reg c0 root) #[(0, 0)] #[o] = .accept) :
    ∃ rs s, _root_.Poub.Session.run (_root_.Poub.Session.State.init params1 reg #[c0, zeros 128] #[] 0)
      (events1 root o) = .ok (rs, s) ∧ s.outcome = some "accepted" ∧
      s.summary.map (fun sm => sm.windows.map fun w => (w.lengths, w.drawn)) = some #[(#[1], 1)] := by
  have hdraw := receiptSample_one
  have hr1 : Array.range 1 = #[0] := rfl
  simp only [window1] at hcheck
  simp [hr1, _root_.Poub.Session.run, events1, _root_.Poub.Session.step, _root_.Poub.Session.start,
    _root_.Poub.Session.boundary, _root_.Poub.Session.receive, _root_.Poub.Session.close,
    _root_.Poub.Session.verify, _root_.Poub.Session.State.init,
    _root_.Poub.Session.State.releaseCoin, _root_.Poub.Session.State.deadline,
    _root_.Poub.Session.State.staleBelow, _root_.Poub.Session.State.record,
    _root_.Poub.Session.State.outcome, _root_.Poub.Session.Win.receipts, Poub.Session.live,
    Poub.Session.cover, _root_.Poub.Session.State.joinsAnother, _root_.Poub.Session.State.refreshRefusal,
    _root_.Poub.Session.State.probationPush, params1, hB,
    hroot, plan_length, hdraw, hcheck, _root_.Poub.Session.RECEIPT_LIMIT,
    _root_.Poub.Session.STEP_LIMIT, DIGEST_BYTES]
  refine ⟨_, _, rfl, ?_, ?_⟩
  · simp
    rfl
  · simp [_root_.Poub.Session.State.summary, _root_.Poub.Session.State.outcome, hr1, plan_length]
    rfl

/-! ## The instance -/

/-- The node: `params1` at one window a sample over `reg1`; the line's other fields are arbitrary. -/
def node1 : Node :=
  { params := params1, groupsPerCoin := 1, reg := reg1, per := 1, blockBytes := 1, chipBytes := 0, nodeBytes := 0,
    rates := ⟨0, 0, 0, 0, 0, 0, 0⟩, delta := 0 }

/-- What the auditor saw of window 0: its coin released at 0 and its last root at 1. -/
def seen1 : Array _root_.Poub.Line.Seen := #[⟨0, 1, 1, 0, 1, 0⟩]

/-- The record: its challenge and its sample's coins zero, no appends, `events1` with `root1` and `opening1`, and
`seen1`. -/
def record1 : Record :=
  let c := zeros 64
  { coins := #[c, zeros 128], appendCoins := #[], notBeforeNs := 0,
    events := events1 (root1 reg1 (seed c 0 reg1.digest)) (opening1 reg1 c), seen := seen1 }

theorem reg1_make : Registration.make? (zeros 1) 1 reg1.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false
    LAYOUT 16 = .ok reg1 := by
  have hs1 : storeBlocks 1 = true := by decide
  have hs2 : 1 ≤ honestCodec.maxBlocks := by decide
  have hroot : reg1.payloadRoot.size = 64 := by
    simp only [reg1, zeroRoot, climb_size]
  have hrow : Kind.rowsOk .sketch 16 = true := rfl
  have hnode : (zeros 1).size = 1 := zeros_size 1
  have hseed : (zeros Codec.SEED_BYTES).size = Codec.SEED_BYTES := zeros_size _
  unfold Registration.make?
  simp [hnode, hs1, hs2, hroot, hseed, hrow]
  try rfl

/-- **An accepted session of the remote audit.** The Lean verifier accepts `record1` on `node1`: a node of one
registered block under the deployed codec, which `Registration.make?` accepts (`reg1_make`), one window of one pass
under one sample. -/
theorem record_run : ∃ rs s, record1.run node1 = .ok (rs, s) ∧ s.outcome = some "accepted" ∧
    s.summary.map (fun sm => sm.windows.map fun w => (w.lengths, w.drawn)) = some #[(#[1], 1)] := by
  have hroot : (root1 reg1 (seed (zeros 64) 0 reg1.digest)).size = 64 := by
    simp only [root1, zeroRoot, climb_size]
  have hpay : reg1.payloadRoot = zeroRoot reg1.payloadDomain 0 (hidden (zeroPayload reg1.key) (zeros SALT_BYTES)) :=
    rfl
  have hcheck := checkWindow_one reg1 (zeros 64) rfl rfl hpay
  simp only [Record.run, node1, record1, _root_.Poub.Shared.run_one]
  exact run_accepts reg1 rfl (zeros 64) _ hroot _ hcheck

theorem record_verified : Verified node1 record1 := by
  obtain ⟨rs, s, hr, ho, -⟩ := record_run
  exact ⟨rs, s, hr, ho⟩


/-! ## `LineHeld`'s premises at the record -/

/-- Window 0 as one group: coin at 0, last root at 1, one pass. -/
def group1 : _root_.Poub.Line.Group := ⟨0, 1, 1, 1, 0, 1, 0⟩

theorem ofSummary_one (sm : _root_.Poub.Session.Summary)
    (h : sm.windows.map (fun w => (w.lengths, w.drawn)) = #[(#[1], 1)]) :
    _root_.Poub.Line.ofSummaryAt 1 sm 1 seen1 = .ok (#[group1], #[(1, 1)]) := by
  obtain ⟨⟨ws⟩, blocks, appended, released, lifetime, pace, appends⟩ := sm
  rcases ws with _ | ⟨w, _ | ⟨w', ws⟩⟩
  · simp at h
  · simp only [List.map_toArray, List.map_cons, List.map_nil, Array.mk.injEq, List.cons.injEq, Prod.mk.injEq,
      and_true] at h
    obtain ⟨hl, hd⟩ := h
    simp [_root_.Poub.Line.ofSummaryAt, _root_.Poub.Line.ofSummary, _root_.Poub.Line.receipts,
      hl, hd, seen1, group1]
    refine ⟨0, rfl, rfl, rfl, ?_, rfl, rfl, rfl, rfl⟩
    simp [hl]
  · simp at h

theorem groups_one : record1.groups node1 = #[group1] := by
  obtain ⟨rs, s, hr, -, hsm⟩ := record_run
  obtain ⟨sm, hs, hw⟩ := Option.map_eq_some_iff.1 hsm
  have hsum : record1.summary node1 = some sm := by
    simp only [Record.summary, hr, hs]
  simp only [Record.groups, hsum]
  rw [show node1.groupsPerCoin = 1 from rfl, show node1.params.samples = 1 from rfl,
    show record1.seen = seen1 from rfl, ofSummary_one sm hw]

/-- The play: `record1` at every commitment and every coin. -/
abbrev recs1 : Unit → SampleCoins node1.samples 0 Unit → Record := fun _ _ => record1

/-- **`record1`'s schedule is honest**: one window, one pass over `[0, 1)`, the sample's receipts the same at every
coin, and the record's one group, of one pass from coin 0 to last root 1, spans it. -/
theorem honestSchedule_one :
    Subprotocol.HonestSchedule node1 1 (fun _ p => p) recs1 (fun _ => ()) := by
  refine ⟨by decide, fun _ p => Nat.lt_succ_self p, fun _ _ _ _ _ => rfl, fun ω _ gs hgs => ?_⟩
  have hgs' : gs = #[group1] := hgs.symm.trans groups_one
  subst hgs'
  have hw : node1.windows = 1 := rfl
  rw [hw]
  refine ⟨rfl, fun g hg => ?_, fun g h _ hh => ?_, by decide, fun g hg p h1 h2 => ?_⟩
  · obtain rfl : g = 0 := by omega
    decide
  · obtain rfl : h = 0 := by omega
    obtain rfl : g = 0 := by omega
    exact le_rfl
  · obtain rfl : g = 0 := by omega
    have hf0 : firstPass #[group1] 0 = 0 := rfl
    have hf1 : firstPass #[group1] (0 + 1) = 1 := by decide
    obtain rfl : p = 0 := by omega
    show released #[group1] 1 0 ≤ 0 ∧ 1 ≤ ended #[group1] 0
    decide

/-- Constant receipts: `LineHeld`'s premises read them only through `Refused`'s `Moved`, which the honest sample never
reaches. -/
def receipts1 : Receipts node1.samples 0 Unit :=
  ⟨fun _ _ => 0, fun _ _ _ => ⟨⟨fun _ => 0, 0, fun _ => 0⟩, [], .v2⟩, fun _ _ _ _ => 0, fun _ _ _ => 0,
    fun _ _ _ => ⟨[], [], []⟩⟩

/-- **`LineHeld`'s premises hold together at an accepted play.** At `node1` (one registered block under the deployed
codec, one window of one pass, one sample), the play `record1` at every commitment and coin, the honest trace on
`[p, p + 1)` for pass `p` (the whole store in node memory, every receipt correct), no outside device, nothing but the
verifier's messages entering it as its certifiers count, every receipt correct and the sampler's fallbacks 0, every
premise of `Properties.Poub.Line.LineHeld` holds at `δs = δnet = 0`, and the Lean verifier accepts the play at
every coin. -/
theorem lineHeld_accepted :
    ∃ (node : Node) (κ : ℕ) («X» Cm : Type) (_ : Fintype «X»)
      (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
      (rest : SampleCoins node.samples κ «X» → Fin node.samples → ℕ → ℕ)
      (recs : Cm → SampleCoins node.samples κ «X» → Record)
      (correct : Cm → Fin node.samples → SampleCoins node.samples κ «X» → Finset ℕ)
      (com : SampleCoins node.samples κ «X» → Cm) (T : Trace (SampleCoins node.samples κ «X»))
      (R : Receipts node.samples κ «X») (δs δnet : ℝ)
      (O : Outside (ofRecord T node (played recs com))) (I : Ingress (SampleCoins node.samples κ «X»)),
      Paced (ofRecord T node (played recs com)) (recordPace node (played recs com)) ∧
      Covers (ofRecord T node (played recs com)) ∧
      SketchBoundAt (ofRecord T node (played recs com)) O δs ∧
      Covered (ofRecord T node (played recs com)) O I ∧
      IngressCounted (ofRecord T node (played recs com)) I δnet ∧
      (∀ c, Live ((recordCommitted node recs correct).view c)) ∧
      Refused ((recordCommitted node recs correct).play com) prf rest (ofRecord T node (played recs com)) R ∧
      Counted ((recordCommitted node recs correct).play com) (ofRecord T node (played recs com)) ∧
      ∃ ω, Verified node (played recs com ω) := by
  obtain ⟨hP, hCv, hSk, hCov, hIc, hLive, hRef, hCnt, -, -⟩ :=
    Subprotocol.lineHeldAtSubprotocol_honest node1 0 Unit Unit (fun _ _ _ _ => false) recs1 (fun _ => ()) receipts1 1
      (fun _ p => p) honestSchedule_one
  exact ⟨node1, 0, Unit, Unit, inferInstance, fun _ _ _ _ => false, fun _ _ _ => 0, recs1,
    fun _ _ _ => Finset.range (receiptMax + 1), fun _ => (), _, receipts1, 0, 0, _, _, hP, hCv, hSk, hCov, hIc,
    hLive, hRef, hCnt, ⟨(), fun _ _ => false⟩, record_verified⟩

/-- **`LineHeld`'s honest witness, with no premise.** At `lineHeld_accepted`'s play every premise of
`Properties.Poub.Line.LineHeld` holds, and `Shared.run` runs the played record's session, at the node's parameters,
windows a sample and registration, to an accepted state. -/
theorem _root_.Properties.Poub.Line.SecurityProofs.lineHeld_honest :
    ∃ (node : Node) (κ : ℕ) («X» Cm : Type) (_ : Fintype «X»)
      (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
      (rest : SampleCoins node.samples κ «X» → Fin node.samples → ℕ → ℕ)
      (recs : Cm → SampleCoins node.samples κ «X» → Record)
      (correct : Cm → Fin node.samples → SampleCoins node.samples κ «X» → Finset ℕ)
      (com : SampleCoins node.samples κ «X» → Cm) (T : Trace (SampleCoins node.samples κ «X»))
      (R : Receipts node.samples κ «X») (δs δnet : ℝ)
      (O : Outside (ofRecord T node (played recs com))) (I : Ingress (SampleCoins node.samples κ «X»)),
      Paced (ofRecord T node (played recs com)) (recordPace node (played recs com)) ∧
      Covers (ofRecord T node (played recs com)) ∧
      SketchBoundAt (ofRecord T node (played recs com)) O δs ∧
      Covered (ofRecord T node (played recs com)) O I ∧
      IngressCounted (ofRecord T node (played recs com)) I δnet ∧
      (∀ c, Live ((recordCommitted node recs correct).view c)) ∧
      Refused ((recordCommitted node recs correct).play com) prf rest (ofRecord T node (played recs com)) R ∧
      Counted ((recordCommitted node recs correct).play com) (ofRecord T node (played recs com)) ∧
      ∃ ω rs s, _root_.Poub.Shared.run node.groupsPerCoin
          (_root_.Poub.Session.State.init node.params node.reg (played recs com ω).coins
            (played recs com ω).appendCoins (played recs com ω).notBeforeNs) (played recs com ω).events =
          .ok (rs, s) ∧ s.outcome = some "accepted" := by
  obtain ⟨node, κ, «X», Cm, hX, prf, rest, recs, correct, com, T, R, δs, δnet, O, I, hP, hCv, hSk, hCov, hIc, hLive,
    hRef, hCnt, ω, rs, s, hr, ho⟩ := lineHeld_accepted
  exact ⟨node, κ, «X», Cm, hX, prf, rest, recs, correct, com, T, R, δs, δnet, O, I, hP, hCv, hSk, hCov, hIc, hLive,
    hRef, hCnt, ω, rs, s, hr, ho⟩

end Properties.Poub.Line.Honest

/-! ## Witnesses of the window's, the sample's and the appends' properties

The honest witnesses are at this session: the one-block registration `reg1`, which `Registration.make?` accepts at the
deployed codec, sketch-sha512 receipts of 16 rows (`reg1_make`), and its window 0 of one pass over the whole store under
challenge `zeros 64`, whose one receipt a sample of one draws on 128 zero bytes (`receiptSample_one`) and whose opening
`checkWindow` accepts (`checkWindow_one`).

* `verifyWindow_one`: `verifyWindow` accepts that window, its one opening and its sample's coins: the premise of
  `AcceptChecksOpenings`, `AcceptOpensSamples` and `AcceptSamplesOpenable`.
* `sampleIsSubset_honest`, `Shared.acceptSamplesOpenable_honest`: the sample of that window, alone or as the one window
  of its sample, is drawn.
* `appendsInOrder_honest`: the session's run, which accepts (`run_accepts`).

The nontrivial witnesses: on a sample's coins of no bytes the draw of one receipt among one runs out on its first read
(`sampleUnits_empty`), so no sample is drawn (`receiptSample_empty`) and no values are the draw's
(`sampleValues_empty`), while a verifier that accepted every window would accept it (also the shared sample's
`SampleIsSubset`, `RunSamplesRecords` and `AcceptedSamplesOpenable`, whose honest witnesses need a run's state at a
sample's close); and a state whose store does not end where its registration and appends end fails `AppendsInOrder`'s
first conjunct.
-/

namespace Properties.Poub

open Properties.Poub.Line.Honest

/-- The draw of one unit among one on no bytes throws `RAN_OUT` at its first read, before its loop. -/
theorem sampleUnits_empty : Spot.Draw.sampleUnits (.subset 1) 1 ByteArray.empty = .error Spot.Draw.RAN_OUT := by
  unfold Spot.Draw.sampleUnits Flock.Draw.drawWith Flock.Draw.onWith
  simp only [width_one]
  rfl

theorem receiptSample_empty : receiptSample 1 ByteArray.empty #[1] = none := by
  unfold receiptSample receiptSampleRead
  simp [sampleUnits_empty]
  rfl

theorem sampleValues_empty (us : Array ℕ) : ¬ SampleValues 1 1 ByteArray.empty us := by
  rintro (⟨h, -⟩ | ⟨-, read, h⟩)
  · exact absurd h (by decide)
  · rw [show min 1 1 = 1 from rfl, sampleUnits_empty] at h
    cases h

theorem window0_lengths : window0.passes.map (·.plan.length) = #[1] := by
  simp [window0, window1, plan_length]

theorem verifyWindow_one : verifyWindow reg1 #[] window0 (zeros 128) 1 #[opening1 reg1 (zeros 64)] = some .accept := by
  have hs : receiptSample 1 (zeros 128) (window0.passes.map (·.plan.length)) = some #[(0, 0)] := by
    rw [window0_lengths]
    exact receiptSample_one
  have hc := checkWindow_one reg1 (zeros 64) rfl rfl rfl
  unfold verifyWindow
  rw [hs]
  exact congrArg some hc

/-- **`AcceptChecksOpenings`' premise holds at the one-block registration.** `verifyWindow` accepts window 0 of one
pass over the whole store of a registration `Registration.make?` accepts, on a sample of one receipt and its opening. -/
theorem acceptChecksOpenings_honest :
    ∃ (reg : Registration) (w : WindowRecord) (coin : ByteArray) (openings : Array Opening),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        verifyWindow reg #[] w coin 1 openings = some .accept :=
  ⟨reg1, window0, zeros 128, #[opening1 reg1 (zeros 64)], reg1_make, verifyWindow_one⟩

/-- **`AcceptChecksOpenings` fails without the draw.** On a sample's coins of no bytes no sample is drawn over the
window's one receipt, which a verifier that accepted every window would accept. -/
theorem acceptChecksOpenings_nontrivial : ∀ sampled, ¬ receiptSample 1 ByteArray.empty #[1] = some sampled := by
  intro sampled h
  rw [receiptSample_empty] at h
  cases h

/-- **`AcceptOpensSamples`' premise holds at the one-block registration** (`acceptChecksOpenings_honest`). -/
theorem acceptOpensSamples_honest :
    ∃ (reg : Registration) (w : WindowRecord) (coin : ByteArray) (openings : Array Opening),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        verifyWindow reg #[] w coin 1 openings = some .accept :=
  acceptChecksOpenings_honest

/-- **`AcceptOpensSamples` fails without the draw** (`acceptChecksOpenings_nontrivial`). -/
theorem acceptOpensSamples_nontrivial : ∀ sampled, ¬ receiptSample 1 ByteArray.empty #[1] = some sampled :=
  acceptChecksOpenings_nontrivial

/-- **`AcceptSamplesOpenable`' premise holds at the one-block registration** (`acceptChecksOpenings_honest`). -/
theorem acceptSamplesOpenable_honest :
    ∃ (reg : Registration) (w : WindowRecord) (coin : ByteArray) (openings : Array Opening),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        verifyWindow reg #[] w coin 1 openings = some .accept :=
  acceptChecksOpenings_honest

/-- **`AcceptSamplesOpenable` fails without the draw.** On a sample's coins of no bytes no values among the window's one
receipt are the draw's. -/
theorem acceptSamplesOpenable_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`SampleIsSubset`'s premise holds at the one-block registration's window**: its one receipt is drawn on 128 zero
bytes. -/
theorem sampleIsSubset_honest :
    ∃ (src : ByteArray) (sampled : Array (ℕ × ℕ)), receiptSample 1 src (window0.passes.map (·.plan.length)) =
      some sampled :=
  ⟨zeros 128, #[(0, 0)], by rw [window0_lengths]; exact receiptSample_one⟩

/-- **`SampleIsSubset` fails without the draw.** On no bytes no values among one receipt are the draw's. -/
theorem sampleIsSubset_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`AppendsInOrder`'s premise holds at the one-block registration**: the session's run over window 0, its one pass,
its sample and its opening accepts (`run_accepts`). -/
theorem appendsInOrder_honest :
    ∃ (reg : Registration) (coins : Array ByteArray) (events : Array _root_.Poub.Session.Event)
      (rs : Array _root_.Poub.Session.Result) (s : _root_.Poub.Session.State),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        _root_.Poub.Session.run (_root_.Poub.Session.State.init params1 reg coins #[] 0) events = .ok (rs, s) ∧
        s.outcome = some "accepted" := by
  have hroot : (root1 reg1 (seed (zeros 64) 0 reg1.digest)).size = 64 := by
    simp only [root1, zeroRoot, climb_size]
  obtain ⟨rs, s, hr, ho, -⟩ :=
    run_accepts reg1 rfl (zeros 64) _ hroot _ (checkWindow_one reg1 (zeros 64) rfl rfl rfl)
  exact ⟨reg1, _, _, rs, s, reg1_make, hr, ho⟩

/-- **`AppendsInOrder` fails without the session's checks.** A state whose store ends past its registration's one block
with no appends, which a verifier that accepted every session would leave, has its store not ending where its
registration and appends do. -/
theorem appendsInOrder_nontrivial : ¬ overrun.extent = reg1.blocks + (overrun.appends.toList.map (·.count)).sum := by
  intro h
  have h' : (2 : ℕ) = 1 + 0 := h
  omega

namespace Shared

/-- **`Shared.AcceptSamplesOpenable`'s premise holds at the one-block registration**: a sample whose one window is its
window 0 draws that window's one receipt on 128 zero bytes. -/
theorem acceptSamplesOpenable_honest :
    ∃ (coin : ByteArray) (sampled : Array (ℕ × ℕ)), receiptSample 1 coin (lengthsOf [window0]) = some sampled :=
  ⟨zeros 128, #[(0, 0)], by
    rw [show lengthsOf [window0] = #[1] by simp [lengthsOf, window0, window1, plan_length]]
    exact receiptSample_one⟩

/-- **`Shared.AcceptSamplesOpenable` fails without the draw.** On no bytes no values among one receipt are the
draw's. -/
theorem acceptSamplesOpenable_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`Shared.SampleIsSubset` fails without the draw.** On no bytes no values among one receipt are the draw's. -/
theorem sampleIsSubset_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`Shared.RunSamplesRecords` fails without the draw.** On a sample's coins of no bytes no sample is drawn over one
receipt. -/
theorem runSamplesRecords_nontrivial : ∀ sampled, ¬ receiptSample 1 ByteArray.empty #[1] = some sampled :=
  acceptChecksOpenings_nontrivial

/-- **`Shared.AcceptedSamplesOpenable` fails without the draw.** On no bytes no values among one receipt are the
draw's. -/
theorem acceptedSamplesOpenable_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

end Shared

end Properties.Poub
