import NetworkCheck.RecordsGrid

/-!
# The records check's byte payloads are v4's bits

`verity/network-records/v3` keeps each frame's payload as its bytes (`Records.ByteEvidence`), where
`verity/network-certificate/v4` reads it as bits (`Check.evidenceOf`, `NetworkCertifier.LogicalFrame`). The two readings
of the evidence agree, and so do the grids, the clock and the frame-size check the verdicts read from them:

- `bytesOfBits_bitsOfBytes`: a payload's bits, read back as bytes, are the payload;
- `unhex_eq`: the records check's hex decoder, which reads a string's characters into an array with no list between, is
  C-Flock's `Flock.unhex`;
- `evidenceOf_toEvidence`: `Check.evidenceOf` is `Records.evidenceOf` with each payload read as its bits;
- `expectedGridsOfBytes_eq`: window `w`'s grid of `expectedGridsOfBytes` is `Check.expectedGrid`'s of the link as v4 reads it
  (through `expectedGrids` and `RecordsGrid.expectedGrids_eq`);
- `unitSync_toEvidence` and `frameBytes_toEvidence`: the unit's clock and the frame-size refusal are v4's.
-/

namespace NetworkCheck.RecordsBytes

open NetworkCheck.Records
open NetworkCheck.Check (bitsOfBytes bytesOfBits bytesOfBitsList byteOfBits)
open NetworkCertifier (ClockSync LogicalFrame)

/-- The eight bits of `n`, most significant first, as `bitsOfBytes` writes a byte. -/
private def bits8 (n : Nat) : List Bool := (List.range 8).map fun i => (n >>> (7 - i)) % 2 == 1

private theorem bits8_eq (n : Nat) : bits8 n = [(n >>> 7) % 2 == 1, (n >>> 6) % 2 == 1, (n >>> 5) % 2 == 1,
    (n >>> 4) % 2 == 1, (n >>> 3) % 2 == 1, (n >>> 2) % 2 == 1, (n >>> 1) % 2 == 1, (n >>> 0) % 2 == 1] := rfl

private theorem toNat_bit (m : Nat) : (m % 2 == 1).toNat = m % 2 := by
  cases Nat.mod_two_eq_zero_or_one m <;> simp [*]

private theorem fold_bits8 (n : Nat) (h : n < 256) :
    (bits8 n).foldl (fun (a : Nat) (b : Bool) => 2 * a + b.toNat) 0 = n := by
  simp only [bits8_eq, List.foldl_cons, List.foldl_nil, toNat_bit, Nat.shiftRight_eq_div_pow]
  omega

private theorem byteOfBits_bits8 (x : UInt8) : byteOfBits (bits8 x.toNat) = x := by
  unfold byteOfBits
  rw [fold_bits8 _ x.toNat_lt]
  exact UInt8.ofNat_toNat

private theorem bytesOfBitsList_flatMap (xs : List UInt8) :
    bytesOfBitsList (xs.flatMap fun x => bits8 x.toNat) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.flatMap_cons, bits8_eq]
    simp only [List.cons_append, List.nil_append, bytesOfBitsList, ih]
    rw [← bits8_eq, byteOfBits_bits8]

/-- A payload's bits, read back as bytes, are the payload. -/
theorem bytesOfBits_bitsOfBytes (b : ByteArray) : bytesOfBits (bitsOfBytes b) = b := by
  show (⟨(bytesOfBitsList (b.data.toList.flatMap fun x => bits8 x.toNat)).toArray⟩ : ByteArray) = b
  rw [bytesOfBitsList_flatMap]

theorem length_bitsOfBytes (b : ByteArray) : (bitsOfBytes b).length = 8 * b.size := by
  show (b.data.toList.flatMap fun x => bits8 x.toNat).length = 8 * b.size
  have h8 : ∀ x : UInt8, (bits8 x.toNat).length = 8 := fun x => by rw [bits8_eq]; rfl
  rw [List.length_flatMap]
  simp only [h8, List.map_const', ByteArray.size, Array.length_toList]
  rw [List.sum_replicate_nat, Nat.mul_comm]

private theorem listMapM_map {ε α β γ} (f : α → Except ε β) (g : β → γ) (xs : List α) :
    (xs.mapM fun x => g <$> f x) = (List.map g) <$> xs.mapM f := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.mapM_cons, ih]
    cases f x <;> cases xs.mapM f <;> rfl

private theorem arrayMapM_map {ε α β γ} (f : α → Except ε β) (g : β → γ) (xs : Array α) :
    (xs.mapM fun x => g <$> f x) = (Array.map g) <$> xs.mapM f := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList, listMapM_map]
  cases xs.toList.mapM f <;> simp [Functor.map, Except.map]

private theorem unhex_flock (s : String) : Flock.unhex s = unhexChars s.toList.toArray := rfl

private theorem foldl_push (l : List Char) (a : Array Char) : l.foldl Array.push a = a ++ l.toArray := by
  induction l generalizing a with
  | nil => simp
  | cons c l ih => simp [ih]

/-- The records check's hex decoder is C-Flock's: the same bytes and the same refusals. -/
theorem unhex_eq (s : String) : unhex s = Flock.unhex s := by
  rw [unhex_flock, unhex, String.foldl_eq_foldl_toList, foldl_push, Array.emptyWithCapacity_eq, Array.empty_append]

/-- `Check.frameOf` is `byteFrameOf` with the payload read as its bits. -/
theorem frameOf_toLogical (j : Lean.Json) : Check.frameOf j = ByteFrame.toLogical <$> byteFrameOf j := by
  unfold Check.frameOf byteFrameOf
  cases Flock.J.asArr j "a frame" with
  | error e => rfl
  | ok a =>
    rcases a with ⟨_ | ⟨p, _ | ⟨q, _ | ⟨r, _ | ⟨s, t⟩⟩⟩⟩⟩
    · rfl
    · rfl
    · rfl
    · show (Except.bind (Flock.J.asStr p "a frame's payload") fun x => Except.bind (Flock.unhex x) fun x =>
          Except.bind (Flock.J.asNat q "an anchor") fun y => Except.bind (Flock.J.asNat r "an index") fun z =>
          Except.pure (⟨Check.bitsOfBytes x, y, z⟩ : NetworkCertifier.LogicalFrame)) =
        ByteFrame.toLogical <$> (Except.bind (Flock.J.asStr p "a frame's payload") fun x =>
          Except.bind (unhex x) fun x => Except.bind (Flock.J.asNat q "an anchor") fun y =>
          Except.bind (Flock.J.asNat r "an index") fun z => Except.pure (⟨x, y, z⟩ : ByteFrame))
      simp only [unhex_eq]
      rcases Flock.J.asStr p "a frame's payload" with _ | x <;> try rfl
      dsimp only [Except.bind]
      rcases Flock.unhex x with _ | b <;> try rfl
      dsimp only
      rcases Flock.J.asNat q "an anchor" with _ | y <;> try rfl
      rcases Flock.J.asNat r "an index" with _ | z <;> rfl
    · show (Except.error "a frame is [payload, anchor, index]" : Flock.V NetworkCertifier.LogicalFrame) =
        ByteFrame.toLogical <$> (Except.error "a frame is [payload, anchor, index]" : Flock.V ByteFrame)
      rfl

/-- `Check.evidenceOf`'s link, its do block written out. -/
private def linkOfL (l : Lean.Json) : Flock.V Check.LinkOpening :=
  Except.bind (Flock.J.arr l "frames" "a link") fun fs => Except.bind (fs.mapM Check.frameOf) fun fr =>
    Except.bind (Flock.J.arr l "syncs" "a link") fun ss => Except.bind (ss.mapM Check.syncOf) fun sy =>
    Except.pure { frames := fr.toList, syncs := sy }

/-- `Records.evidenceOf`'s link, its do block written out. -/
private def linkOfB (l : Lean.Json) : Flock.V ByteLink :=
  Except.bind (Flock.J.arr l "frames" "a link") fun fs => Except.bind (fs.mapM byteFrameOf) fun fr =>
    Except.bind (Flock.J.arr l "syncs" "a link") fun ss => Except.bind (ss.mapM Check.syncOf) fun sy =>
    Except.pure { frames := fr.toList, syncs := sy }

private theorem linkOf_toLink (l : Lean.Json) : linkOfL l = ByteLink.toLink <$> linkOfB l := by
  unfold linkOfL linkOfB
  rcases Flock.J.arr l "frames" "a link" with _ | fs <;> try rfl
  dsimp only [Except.bind]
  rw [show Check.frameOf = fun x => ByteFrame.toLogical <$> byteFrameOf x from funext frameOf_toLogical,
    arrayMapM_map]
  rcases fs.mapM byteFrameOf with _ | fr <;> try rfl
  dsimp only [Functor.map, Except.map]
  rcases Flock.J.arr l "syncs" "a link" with _ | ss <;> try rfl
  dsimp only [Except.bind]
  rcases ss.mapM Check.syncOf with _ | sy <;> try rfl
  simp [Except.pure, ByteLink.toLink]

/-- `Check.evidenceOf` is `Records.evidenceOf` with each payload read as its bits: the same JSON, the same refusals. -/
theorem evidenceOf_toEvidence (j : Lean.Json) : Check.evidenceOf j = ByteEvidence.toEvidence <$> evidenceOf j := by
  show (Except.bind (Flock.J.arr j "links" "the object") fun ls => Except.bind (ls.mapM linkOfL) fun links =>
      Except.pure (⟨links⟩ : Check.Evidence)) =
    ByteEvidence.toEvidence <$> (Except.bind (Flock.J.arr j "links" "the object") fun ls =>
      Except.bind (ls.mapM linkOfB) fun links => Except.pure (⟨links⟩ : ByteEvidence))
  rcases Flock.J.arr j "links" "the object" with _ | ls <;> try rfl
  dsimp only [Except.bind]
  rw [show linkOfL = fun x => ByteLink.toLink <$> linkOfB x from funext linkOf_toLink, arrayMapM_map]
  rcases ls.mapM linkOfB with _ | links <;> rfl

/-- The hand times read only the frames' anchors and indices. -/
theorem handingRev_timing (T W : Nat) (syncs : Fin W → ClockSync) (last : Nat) (fs : List ByteFrame) (acc : List Nat) :
    handingRev T W syncs last (fs.map ByteFrame.toLogical) acc = handingRev T W syncs last (fs.map ByteFrame.timing) acc := by
  induction fs generalizing last acc with
  | nil => rfl
  | cons f fs ih => simp only [List.map_cons, handingRev, NetworkCertifier.eligible, ByteFrame.toLogical, ByteFrame.timing, ih]

theorem payloads_bytes (fs : List ByteFrame) :
    ((fs.map ByteFrame.toLogical).toArray.map fun f => Check.bytesOfBits f.payload) = fs.toArray.map ByteFrame.payload := by
  simp [ByteFrame.toLogical, bytesOfBits_bitsOfBytes]

/-- `expectedGridsOfBytes` is `expectedGrids` of the link as v4 reads it. -/
theorem expectedGridsOfBytes_eq_expectedGrids (g : Check.Grid) (W : Nat) (l : ByteLink) :
    expectedGridsOfBytes g W l = expectedGrids g W l.toLink := by
  simp only [expectedGridsOfBytes, expectedGrids, ByteLink.toLink, handingRev_timing, payloads_bytes]

/-- Window `w`'s grid of `expectedGridsOfBytes` is `Check.expectedGrid`'s of the link as v4 reads it, slot for slot:
`expectedGridsOfBytes_eq_expectedGrids` composed with `RecordsGrid.expectedGrids_eq`. -/
theorem expectedGridsOfBytes_eq (g : Check.Grid) (W : Nat) (l : ByteLink) (w : Fin W) :
    (expectedGridsOfBytes g W l)[w.val]? = some (Check.expectedGrid g W l.toLink w.val) := by
  rw [expectedGridsOfBytes_eq_expectedGrids]
  exact RecordsGrid.expectedGrids_eq g W l.toLink w

/-- The unit's clock is v4's. -/
theorem unitSync_toEvidence (e : ByteEvidence) (w : Nat) : unitSync e w = Check.unitSync e.toEvidence w := by
  simp only [unitSync, Check.unitSync, ByteEvidence.toEvidence, Array.getElem?_map]
  cases e.links[0]? <;> rfl

/-- The frame-size refusal is v4's: a payload of other than `B` bytes is one of other than `8 · B` bits. -/
theorem frameBytes_toEvidence (e : ByteEvidence) (B : Nat) :
    (e.links.any fun l => l.frames.any (·.payload.size ≠ B)) =
      (e.toEvidence.links.any fun l => l.frames.any (·.payload.length ≠ 8 * B)) := by
  simp only [ByteEvidence.toEvidence, Array.any_map]
  congr 1; funext l
  simp only [Function.comp, ByteLink.toLink, List.any_map]
  congr 1; funext f
  simp only [Function.comp, ByteFrame.toLogical, length_bitsOfBytes]
  by_cases h : f.payload.size = B <;> simp [h] <;> omega

end NetworkCheck.RecordsBytes
