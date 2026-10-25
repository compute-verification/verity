import Proofs.Properties.CertifierNetwork.Defs
import Proofs.Properties.CertifierDevice.Fifo
import Proofs.Properties.NetworkCertifier.Bytes

set_option maxRecDepth 20000

/-!
# The frames the shaper hands

Every frame the shaper holds pending is one `encodeFrame` made, of the egress grid's `B` bytes: an egress event is the
only handler that adds one, and a tick only removes them (`step_sized`). So every frame handed over a run has `B`
bytes (`handed_size`), and its bits (`bitsOf`) are `8 · B` of them, read back as the same bytes by the check
(`bytesOfBits_bitsOf`).
-/

namespace Properties.CertifierNetwork.Proofs

open _root_.CertifierDevice
open Properties.CertifierDevice (Packed horizon outputs after handed handedAt)
open Properties.CertifierDevice.Proofs
open NetworkCheck.Check (byteOfBits bytesOfBitsList bytesOfBits)

/-! ## Bits and bytes -/

theorem byte_aux : ∀ n < 256, byteOfBits ((List.range 8).map fun i => n.testBit (7 - i)) = n.toUInt8 := by decide

theorem toList_loop (b : ByteArray) (i : Nat) (r : List UInt8) :
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i := by
  fun_induction ByteArray.toList.loop b i r with
  | case1 i r h ih =>
    rw [ih, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 1
    conv_rhs => rw [List.drop_eq_getElem_cons (by simpa using h)]
    simp [ByteArray.get!, h]
  | case2 i r h =>
    rw [List.drop_eq_nil_of_le (by simp; simpa using h), List.append_nil]

theorem toList_eq (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [toList_loop]
  simp

theorem byte_bits (x : UInt8) : byteOfBits ((List.range 8).map fun i => x.toNat.testBit (7 - i)) = x := by
  rw [byte_aux _ x.toNat_lt]
  simp

theorem bytesOfBitsList_flat : ∀ l : List UInt8,
    bytesOfBitsList (l.flatMap fun x => (List.range 8).map fun i => x.toNat.testBit (7 - i)) = l
  | [] => rfl
  | x :: l => by
    rw [List.flatMap_cons]
    have e : ((List.range 8).map fun i => x.toNat.testBit (7 - i)) = [x.toNat.testBit 7, x.toNat.testBit 6,
        x.toNat.testBit 5, x.toNat.testBit 4, x.toNat.testBit 3, x.toNat.testBit 2, x.toNat.testBit 1,
        x.toNat.testBit 0] := rfl
    have hb := byte_bits x
    rw [e] at hb ⊢
    simp only [List.cons_append, List.nil_append, bytesOfBitsList, hb, bytesOfBitsList_flat l]

/-- The check reads a frame's bits back as its bytes. -/
theorem bytesOfBits_bitsOf (b : ByteArray) : bytesOfBits (bitsOf b) = b := by
  unfold bytesOfBits bitsOf
  rw [toList_eq, bytesOfBitsList_flat]

theorem bitsOf_length (b : ByteArray) : (bitsOf b).length = 8 * b.size := by
  have key : ∀ l : List UInt8,
      (l.flatMap fun x => (List.range 8).map fun i => x.toNat.testBit (7 - i)).length = 8 * l.length := by
    intro l
    induction l with
    | nil => rfl
    | cons x l ih =>
      rw [List.flatMap_cons, List.length_append, ih, List.length_map, List.length_range, List.length_cons]
      omega
  unfold bitsOf
  rw [toList_eq, key, Array.length_toList]
  rfl

/-! ## Pending frames -/

/-- Every frame the shaper holds pending has `B` bytes. -/
def Sized (B : Nat) (s : Shaper) : Prop := ∀ k x, s.pending.get? k = some x → x.1.bytes.size = B

theorem Sized.of_pending {B : Nat} {s s' : Shaper} (h : Sized B s) (e : s'.pending = s.pending) : Sized B s' :=
  fun k x hx => h k x (e ▸ hx)

theorem encodeFrame_size {kind s : Nat} {payload : ByteArray} {B : Nat} {bytes : ByteArray}
    (h : encodeFrame kind s payload B = some bytes) : bytes.size = B := by
  unfold encodeFrame at h
  split at h
  · cases h
  · rename_i hle
    cases h
    simp only [ByteArray.size_append, Properties.NetworkCertifier.Bytes.be32_size]
    simp only [headerSize, gt_iff_lt, Nat.not_lt, ByteArray.size] at hle
    simp only [ByteArray.size, Array.size_replicate, List.size_toArray, List.length_singleton, headerSize]
    omega

theorem compactCell_size {record : ByteArray} {B : Nat} {bytes : ByteArray} (h : compactCell record B = some bytes) :
    bytes.size = B := by
  unfold compactCell at h
  split at h
  · cases h
  · rename_i hle
    cases h
    have : (⟨Array.replicate (B - record.size) 0⟩ : ByteArray).size = B - record.size := Array.size_replicate
    rw [ByteArray.size_append]
    omega

theorem Sized.arrive {B : Nat} {s : Shaper} (h : Sized B s) (position : Nat) {frame : Frame} (hf : frame.bytes.size = B)
    (ready : Int) : Sized B (s.arrive position frame ready) := by
  unfold Shaper.arrive
  split
  · exact h.of_pending rfl
  · split
    · exact h.of_pending rfl
    · intro k x hx
      dsimp only at hx
      rw [Std.HashMap.get?_insert] at hx
      split at hx
      · cases hx
        exact hf
      · exact h k x hx

theorem Sized.handStep {B T : Nat} {s s' : Shaper} (h : Sized B s) (hs : HandStep T s s') : Sized B s' := by
  obtain ⟨frame, ready, sync, hh, hand, -, -, -, -, -, rfl⟩ := hs
  intro k x hx
  dsimp only at hx
  rw [Std.HashMap.get?_erase] at hx
  split at hx
  · cases hx
  · exact h k x hx

theorem checkAdvice_pending (s : Shaper) : s.checkAdvice.pending = s.pending := by
  rw [(checkAdvice_rel s).2]

theorem finishStage_pending (T : Nat) (s : Shaper) : (finishStage T s).1.pending = s.pending := by
  unfold finishStage
  dsimp only
  split
  · rfl
  · split
    · split
      · rw [checkAdvice_pending]; rfl
      · rw [checkAdvice_pending]
    · rw [checkAdvice_pending]

theorem Sized.tick {B : Nat} {s : Shaper} (h : Sized B s) : Sized B s.tick.1 := by
  rw [tick_eq]
  refine Sized.of_pending ?_ (finishStage_pending _ _)
  have hr := handStage_reach s.certifier.params.buckets (declareStage s)
  generalize handStage _ (declareStage s) = s' at hr
  obtain ⟨c', prover, syncs, -, -, e⟩ := declareStage_eq s
  have h0 : Sized B (declareStage s) := h.of_pending (by rw [e])
  induction hr with
  | refl => exact h0
  | tail _ hst ih => exact ih.handStep hst

/-! ## The handlers -/

theorem pump_shaper (st : State) (s : Nat) : (st.pump s).1.shaper = st.shaper := by
  unfold State.pump
  dsimp only
  split
  · rfl
  · exact Id.bind_inv (fun _ => True) (fun (r : State × Array Action) => r.1.shaper = st.shaper) _ _ trivial
      (fun _ _ => rfl)

theorem linkRead_shaper (st : State) (now : Nat) (data : ByteArray) : (st.linkRead now data).shaper = st.shaper := by
  unfold State.linkRead
  dsimp only
  split
  · rfl
  · exact Id.bind_inv (fun _ => True) (fun (r : State) => r.shaper = st.shaper) _ _ trivial (fun _ _ => rfl)

theorem upstreamEnd_pending (st : State) (s now : Nat) :
    (st.upstreamEnd s now).shaper.pending = st.shaper.pending := by
  unfold State.upstreamEnd
  dsimp only
  split <;> rfl

theorem egressFrame_config (st : State) (s position : Nat) (bytes : ByteArray) (ready : Int) :
    (st.egressFrame s position bytes ready).config = st.config := by
  unfold State.egressFrame
  dsimp only
  split <;> rfl

theorem egressEvent_config (st : State) (s : Nat) (event : ByteArray) (ready : Int) :
    (st.egressEvent s event ready).config = st.config := by
  unfold State.egressEvent
  split
  · exact egressFrame_config _ _ _ _ _
  · rfl

theorem egressRecord_config (st : State) (s position : Nat) (record : ByteArray) (ready : Int) :
    (st.egressRecord s position record ready).config = st.config := by
  unfold State.egressRecord
  split
  · exact egressFrame_config _ _ _ _ _
  · rfl

theorem egressFrame_sized {st : State} (h : Sized st.config.egress.frameBytes st.shaper) (s position : Nat)
    {bytes : ByteArray} (hb : bytes.size = st.config.egress.frameBytes) (ready : Int) :
    Sized st.config.egress.frameBytes (st.egressFrame s position bytes ready).shaper := by
  unfold State.egressFrame
  dsimp only
  split
  · exact Sized.arrive h position hb ready
  · exact h.of_pending rfl

theorem egressEvent_sized {st : State} (h : Sized st.config.egress.frameBytes st.shaper) (s : Nat) (event : ByteArray)
    (ready : Int) : Sized st.config.egress.frameBytes (st.egressEvent s event ready).shaper := by
  unfold State.egressEvent
  split
  · rename_i position bytes _ hb
    exact egressFrame_sized h s position (encodeFrame_size hb) ready
  · exact h.of_pending rfl

theorem egressRecord_sized {st : State} (h : Sized st.config.egress.frameBytes st.shaper) (s position : Nat)
    (record : ByteArray) (ready : Int) :
    Sized st.config.egress.frameBytes (st.egressRecord s position record ready).shaper := by
  unfold State.egressRecord
  split
  · rename_i bytes _ hb
    exact egressFrame_sized h s position (compactCell_size hb) ready
  · exact h.of_pending rfl

theorem upstreamRead_sized {st : State} (h : Sized st.config.egress.frameBytes st.shaper) (s now : Nat)
    (data : ByteArray) : Sized st.config.egress.frameBytes (st.upstreamRead s now data).shaper := by
  unfold State.upstreamRead
  dsimp only
  split
  · exact h.of_pending (upstreamEnd_pending _ _ _)
  · split
    · refine Id.bind_inv (fun (r : State) => r.config = st.config ∧ Sized st.config.egress.frameBytes r.shaper)
        (fun (r : State) => Sized st.config.egress.frameBytes r.shaper) _ _ ?_ (fun _ h => h.2)
      refine forIn_inv_arr (fun (r : State) => r.config = st.config ∧ Sized st.config.egress.frameBytes r.shaper)
        ⟨rfl, h⟩ ?_
      intro u _ b hb
      rcases u with ⟨position, record⟩
      have key : ∀ r, (b.egressRecord s position record r).config = st.config ∧
          Sized st.config.egress.frameBytes (b.egressRecord s position record r).shaper := fun r =>
        ⟨(egressRecord_config _ _ _ _ _).trans hb.1, by
          have := egressRecord_sized (st := b) (by rw [hb.1]; exact hb.2) s position record r
          rwa [hb.1] at this⟩
      exact key _
    · refine Id.bind_inv (fun (r : State) => r.config = st.config ∧ Sized st.config.egress.frameBytes r.shaper)
        (fun (r : State) => Sized st.config.egress.frameBytes r.shaper) _ _ ?_ (fun _ h => h.2)
      refine forIn_inv_arr (fun (r : State) => r.config = st.config ∧ Sized st.config.egress.frameBytes r.shaper)
        ⟨rfl, h⟩ ?_
      intro ev _ b hb
      have key : ∀ r, (b.egressEvent s ev r).config = st.config ∧
          Sized st.config.egress.frameBytes (b.egressEvent s ev r).shaper := fun r =>
        ⟨(egressEvent_config _ _ _ _).trans hb.1, by
          have := egressEvent_sized (st := b) (by rw [hb.1]; exact hb.2) s ev r
          rwa [hb.1] at this⟩
      exact key _

theorem ingressTail_shaper (st : State) (t : Nat) : (ingressTail st t).1.shaper = st.shaper := by
  unfold ingressTail
  dsimp only
  generalize st.bucketer.tick = x
  rcases x with ⟨b, row, record⟩
  dsimp only
  refine Id.bind_inv (fun (r : State × Array State.Started) => r.1.shaper = st.shaper)
    (fun (r : State × Array Action) => r.1.shaper = st.shaper) _ _ ?_ ?_
  · refine forIn_inv_arr (fun (r : State × Array State.Started) => r.1.shaper = st.shaper) rfl ?_
    intro slot _ y hy
    repeat' split
    all_goals first
      | exact hy
      | exact (releaseSalt_keeps _ _).2.1.trans hy
  · intro y hy
    have h0 : ∀ acts : Array Action,
        (Id.run (forIn y.2 (y.1, acts) fun x r => match x with
          | .connect s => pure (ForInStep.yield (r.1, r.2.push (.connect s)))
          | .pump s => match r.1.pump s with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more))
          | .saltConnect => pure (ForInStep.yield (r.1, r.2.push .connectSalt))
          | .saltPump => match r.1.saltPump with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more)))).1.shaper = st.shaper := by
      intro acts
      refine forIn_inv_arr (fun (r : State × Array Action) => r.1.shaper = st.shaper) hy ?_
      intro x _ r hr
      split
      · exact hr
      · rename_i s _
        have p1 := pump_shaper r.1 s
        generalize r.1.pump s = z at p1 ⊢
        rcases z with ⟨st', more⟩
        exact p1.trans hr
      · exact hr
      · have p1 := (saltPump_keeps r.1).2.1
        generalize r.1.saltPump = z at p1 ⊢
        rcases z with ⟨st', more⟩
        exact p1.trans hr
    split
    · exact h0 _
    · exact h0 _

/-- **Every step keeps the pending frames of `B` bytes.** -/
theorem step_sized {c : Config} {st : State} (hg : Good c st) (hs : Sized c.egress.frameBytes st.shaper) (ev : Event) :
    Sized c.egress.frameBytes (step st ev).1.shaper := by
  have hB : st.config.egress.frameBytes = c.egress.frameBytes := by rw [hg.config]
  have hs' : Sized st.config.egress.frameBytes st.shaper := hB ▸ hs
  by_cases hl : ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c
  · obtain ⟨now, backlog, rfl, ht⟩ := hl
    obtain ⟨l, hl⟩ := step_egress_shaper hg ht (now := now) (backlog := backlog)
    rw [hl]
    exact (hs.of_pending (egressMid_pending st now backlog)).tick.of_pending rfl
  have pump_ok : ∀ (st' : State) (s : Nat) (y : State × Array Action), st'.shaper = st.shaper → st'.pump s = y →
      Sized c.egress.frameBytes y.1.shaper := by
    intro st' s y hq hy
    subst hy
    exact hs.of_pending (by rw [pump_shaper, hq])
  have opened_ok : ∀ (s : Nat) (y : State × Array Action), st.opened s = y → Sized c.egress.frameBytes y.1.shaper := by
    intro s y hy
    subst hy
    exact hs.of_pending (by rw [opened_fst, pump_shaper]; rfl)
  have saltPump_ok : ∀ (st' : State) (y : State × Array Action), st'.shaper = st.shaper → st'.saltPump = y →
      Sized c.egress.frameBytes y.1.shaper := by
    intro st' y hq hy
    subst hy
    exact hs.of_pending (by rw [(saltPump_keeps st').2.1, hq])
  unfold step
  split
  · exact hs
  · split
    rename_i st' acts heq
    split at heq
    · rename_i now backlog
      split at heq
      · exact absurd ⟨now, backlog, rfl, by unfold State.ticks at *; rwa [hg.config] at *⟩ hl
      · cases heq; exact hs
    · split at heq
      · have e : ∀ n, (st.ingressTick n).1.shaper = st.shaper := fun n => by
          rw [ingressTick_eq, ingressTail_shaper]
          split <;> rfl
        have e2 := congrArg (fun p : State × Array Action => p.1.shaper) heq
        dsimp only at e2
        rw [e] at e2
        exact hs.of_pending (by rw [e2])
      · cases heq; exact hs
    · split at heq
      · cases heq; rw [linkRead_shaper]; exact hs
      · cases heq; exact hs
    · split at heq
      · exact opened_ok _ _ heq
      · cases heq; exact hs
    · split at heq
      · cases heq; exact hs
      · cases heq; exact hs
    · split at heq
      · cases heq; exact hB ▸ upstreamRead_sized hs' _ _ _
      · cases heq; exact hs
    · dsimp only at heq
      split at heq
      · exact pump_ok _ _ _ (by rfl) heq
      · cases heq; exact hs
    · split at heq
      · split at heq
        · refine saltPump_ok _ _ ?_ heq
          rfl
        · cases heq; exact hs
      · cases heq; exact hs
    · split at heq
      · split at heq
        · cases heq; exact hs
        · cases heq; exact hs
      · cases heq; exact hs
    · split at heq
      · split at heq
        · refine saltPump_ok _ _ ?_ heq
          rfl
        · cases heq; exact hs
      · cases heq; exact hs

theorem init_sized (c : Config) : Sized c.egress.frameBytes (init c).shaper := by
  intro k x hx
  unfold init Shaper.make at hx
  dsimp only at hx
  rw [checkAdvice_pending] at hx
  simp at hx

/-- **Every frame handed over a run has `B` bytes.** -/
theorem handed_size_from {c : Config} {st : State} (hg : Good c st) (hs : Sized c.egress.frameBytes st.shaper)
    (tr : List Event) : ∀ x ∈ handed st tr, x.1.bytes.size = c.egress.frameBytes := by
  induction tr generalizing st with
  | nil => simp [handed]
  | cons ev tr ih =>
    intro x hx
    rcases List.mem_append.1 hx with hx | hx
    · obtain ⟨f, hf, rfl⟩ := List.mem_map.1 hx
      unfold handedAt at hf
      obtain ⟨k, -, hk⟩ := List.mem_filterMap.1 hf
      obtain ⟨y, hy, rfl⟩ := Option.map_eq_some_iff.1 hk
      exact hs k y hy
    · exact ih (step_good hg ev).1 (step_sized hg hs ev) x hx

theorem handed_size (c : Config) (tr : List Event) :
    ∀ x ∈ handed (init c) tr, x.1.bytes.size = c.egress.frameBytes :=
  handed_size_from (init_good c) (init_sized c) tr

theorem openedFrames_length {c : Config} {tr : List Event} :
    ∀ f ∈ openedFrames c tr, f.payload.length = 8 * c.egress.frameBytes := by
  intro f hf
  unfold openedFrames at hf
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hf
  rw [bitsOf_length, handed_size c tr x hx]

end Properties.CertifierNetwork.Proofs
