module

public import Properties.Poub.Remote.Line.Record
public import Proofs.Properties.Poub.Remote.Schedule
public import Proofs.Properties.Poub.Remote.Window
public import Proofs.Properties.Poub.Remote.SubprotocolWitnesses
import all Init.Data.Array.Basic
import all Init.Data.Array.QSort.Basic
import all Init.Data.Range.Basic

/-! An accepted session of the remote audit and `LineHeld`'s honest witness, without evaluating SHA-512, because the
session's steps read the roots and coins only through their sizes. -/

@[expose] public section

open _root_.Poub

open private Poub.Session.live Poub.Session.cover from Poub.Session

namespace Properties.Poub.Line.Honest

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sample
  ResourceAccounting.Remote.Sample.Assumptions

attribute [local irreducible] zeroRoot

theorem hash_size (m : ByteArray) : (Flock.Sha512.hash m).size = 64 := by
  unfold Flock.Sha512.hash
  exact Array.size_ofFn

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

theorem stream_size (key : ByteArray) (idx : List FlockSoundness.Randomness.Part) :
    (FlockSoundness.Randomness.stream key idx 64).size = 64 := by
  simp [FlockSoundness.Randomness.stream, FlockSoundness.Randomness.block, ByteArray.size_extract, hash_size]

theorem width_one : Flock.Draw.width 1 = 9 := by
  have h : Nat.log2 1 = 0 := (Nat.log2_eq_iff (by decide)).2 ⟨by decide, by decide⟩
  simp [Flock.Draw.width, h]

theorem uniformOn_one (s : ByteArray) (hs : s.size = 64) : FlockSoundness.Randomness.uniformOn 1 s = some 0 := by
  have hx : Flock.Draw.beNat s 0 9 < 4722366482869645213696 := by
    have h := beNat_lt s 0 9
    norm_num at h
    exact h
  unfold FlockSoundness.Randomness.uniformOn
  rw [hs]
  show (Flock.Draw.uniform 1 s (64 + 1) 0).map (·.1) = some 0
  rw [Flock.Draw.uniform]
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

theorem subsetOn_one (streams : ℕ → ByteArray) (h : FlockSoundness.Randomness.uniformOn 1 (streams 0) = some 0) :
    FlockSoundness.Randomness.subsetOn 1 1 streams = some #[0] := by
  unfold FlockSoundness.Randomness.subsetOn
  simp [Id.run, h, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  rw [show Array.range 1 = #[0] from rfl, qsort_map_zero]
  · rfl
  · simp [Std.HashMap.getD_insert_self]

theorem subsetExact_one (key : ByteArray) (idx : List FlockSoundness.Randomness.Part) :
    FlockSoundness.Randomness.subsetExact key 1 1 idx = some #[0] := by
  have h : FlockSoundness.Randomness.subset key 1 1 idx 64 = some #[0] :=
    subsetOn_one _ (uniformOn_one _ (stream_size _ _))
  unfold FlockSoundness.Randomness.subsetExact
  simp [Id.run, h, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.range']
  try rfl

theorem bisectRight_one : bisectRight #[1] 0 = 0 := by
  first
    | rfl
    | decide
    | decide +kernel

theorem receiptSample_one (digest coin : ByteArray) : receiptSample digest 0 coin 1 #[1] = some #[(0, 0)] := by
  unfold receiptSample
  simp [subsetExact_one, bisectRight_one]

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

def leaf1 (reg : Registration) (s : ByteArray) : ByteArray :=
  (receipter reg.receipt s reg.rows).receipt 0 (zeros Codec.BLOCK_BYTES)

def root1 (reg : Registration) (s : ByteArray) : ByteArray :=
  zeroRoot (treeDomain s (Plan.whole 1)) 0 (hidden (leaf1 reg s) (zeros SALT_BYTES))

def window1 (reg : Registration) (c root : ByteArray) : WindowRecord :=
  ⟨0, c, reg.digest, #[⟨root, Plan.whole 1, none⟩]⟩

def opening1 (reg : Registration) (c : ByteArray) : Opening :=
  let s := seed c 0 reg.digest
  ⟨0, 0, zeros Codec.BLOCK_BYTES, leaf1 reg s, zeros SALT_BYTES, zeroPath (treeDomain s (Plan.whole 1)),
    zeros SALT_BYTES, zeroPath reg.payloadDomain, .empty, .empty⟩

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
    (zeros_size _) (zeros_size _) ?_ hb0 (by rw [hB]; decide) rfl ?_ ?_ (decode_zeroPayload reg.key hj hfp) ?_ rfl
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

def params1 : _root_.Poub.Session.Params :=
  { deadlineNs := 0, windows := 1, samples := 1, minPasses := 1, windowNs := 0, lifetime := 1 }

def events1 (root : ByteArray) (o : Opening) : Array _root_.Poub.Session.Event :=
  #[.start 0, .receive 0 0 root none none (some 0) 1, .boundary 1, .close 0 2, .verify 0 #[o]]

theorem run_accepts (reg : Registration) (hB : reg.blocks = 1) (c0 c1 root : ByteArray) (hroot : root.size = 64)
    (o : Opening) (hcheck : checkWindow reg reg.digest #[] (window1 reg c0 root) #[(0, 0)] #[o] = .accept) :
    ∃ rs s, _root_.Poub.Session.run (_root_.Poub.Session.State.init params1 reg #[c0, c1] #[] 0)
      (events1 root o) = .ok (rs, s) ∧ s.outcome = some "accepted" ∧
      s.summary.map (fun sm => sm.windows.map fun w => (w.lengths, w.drawn)) = some #[(#[1], 1)] := by
  have hdraw := receiptSample_one reg.digest c1
  have hr1 : Array.range 1 = #[0] := rfl
  simp only [window1] at hcheck
  simp [hr1, _root_.Poub.Session.run, events1, _root_.Poub.Session.step, _root_.Poub.Session.start,
    _root_.Poub.Session.boundary, _root_.Poub.Session.receive, _root_.Poub.Session.close,
    _root_.Poub.Session.verify, _root_.Poub.Session.State.init,
    _root_.Poub.Session.State.releaseCoin, _root_.Poub.Session.State.deadline,
    _root_.Poub.Session.State.staleBelow, _root_.Poub.Session.State.record,
    _root_.Poub.Session.State.outcome, _root_.Poub.Session.Win.receipts, Poub.Session.live,
    Poub.Session.cover, params1, hB,
    hroot, plan_length, hdraw, hcheck, _root_.Poub.Session.RECEIPT_LIMIT,
    _root_.Poub.Session.STEP_LIMIT, DIGEST_BYTES]
  refine ⟨_, _, rfl, ?_, ?_⟩
  · simp
    rfl
  · simp [_root_.Poub.Session.State.summary, _root_.Poub.Session.State.outcome, hr1, plan_length]
    rfl

def reg1 : Registration :=
  let key := Codec.key (zeros Codec.SEED_BYTES) honestCodec
  let pd := Flock.FV3.Sha512.Domain.make (Poub.sha512 (PAYLOAD_DOMAIN ++ frameBytes (zeros 1))) 0 0 1
  { node := zeros 1, blocks := 1, payloadRoot := zeroRoot pd 0 (hidden (zeroPayload key) (zeros SALT_BYTES)),
    codecSeed := zeros Codec.SEED_BYTES, codec := honestCodec, receipt := .sketch, rows := 16, key }

def node1 : Node :=
  { params := params1, groupsPerCoin := 1, reg := reg1, per := 1, blockBytes := 1, chipBytes := 0, nodeBytes := 0,
    rates := ⟨0, 0, 0, 0, 0, 0, 0⟩, delta := 0 }

def seen1 : Array _root_.Poub.Line.Seen := #[⟨0, 1, 1, 0, 1, 0⟩]

def record1 : Record :=
  let c := zeros 64
  { coins := #[c, c], appendCoins := #[], notBeforeNs := 0,
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
  exact run_accepts reg1 rfl (zeros 64) (zeros 64) _ hroot _ hcheck

theorem record_verified : Verified node1 record1 := by
  obtain ⟨rs, s, hr, ho, -⟩ := record_run
  exact ⟨rs, s, hr, ho⟩

theorem record_accepts : ∃ (node : Node) (r : Record), Verified node r := ⟨node1, record1, record_verified⟩

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

abbrev recs1 : Unit → SampleCoins node1.samples 0 Unit → Record := fun _ _ => record1

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

/-- **`LineHeld`'s premises hold together at an accepted play**: at `node1`, the play `record1` at every commitment and
coin and the honest trace on `[p, p + 1)` for pass `p`, every premise holds at `δs = δnet = 0`, and the Lean verifier
accepts the play at every coin. -/
theorem lineHeld_accepted :
    ∃ (node : Node) (κ : ℕ) (X Cm : Type) (_ : Fintype X)
      (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
      (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ)
      (recs : Cm → SampleCoins node.samples κ X → Record)
      (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
      (com : SampleCoins node.samples κ X → Cm) (T : Trace (SampleCoins node.samples κ X))
      (R : Receipts node.samples κ X) (δs δnet : ℝ)
      (O : Outside (ofRecord T node (played recs com))) (I : Ingress (SampleCoins node.samples κ X)),
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
    ∃ (node : Node) (κ : ℕ) (X Cm : Type) (_ : Fintype X)
      (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
      (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ)
      (recs : Cm → SampleCoins node.samples κ X → Record)
      (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
      (com : SampleCoins node.samples κ X → Cm) (T : Trace (SampleCoins node.samples κ X))
      (R : Receipts node.samples κ X) (δs δnet : ℝ)
      (O : Outside (ofRecord T node (played recs com))) (I : Ingress (SampleCoins node.samples κ X)),
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
  obtain ⟨node, κ, X, Cm, hX, prf, rest, recs, correct, com, T, R, δs, δnet, O, I, hP, hCv, hSk, hCov, hIc, hLive,
    hRef, hCnt, ω, rs, s, hr, ho⟩ := lineHeld_accepted
  exact ⟨node, κ, X, Cm, hX, prf, rest, recs, correct, com, T, R, δs, δnet, O, I, hP, hCv, hSk, hCov, hIc, hLive,
    hRef, hCnt, ω, rs, s, hr, ho⟩

end Properties.Poub.Line.Honest
