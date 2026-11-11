module

public import Proofs.Flock.SessionLoop.Honest
public import Proofs.Flock.SessionLoop.Level0SrcSat

/-!
# The loop's rep samples are satisfiable, and so is `Loop.Src`

`Loop.RepSrc`, the rep-draw half of `Loop.Src` (`flock-firewall-os-src`), holds at every statement and witness, at a
release whose frames sample the tape's pads and pads-tree salts and a source that hands each rep the tape's salts for
levels ≥ 1, each written in the layout the host reads it in: `Frame.Draws.ofBytes`'s for the frame's samples and
`Finish.hiSaltsOf`'s, `192·2^d` bytes a level, for the salts.

* `Frame.tapeBytes`: table `i`'s rep `r` samples from a tape at layout `l`, `drawBytes l` bytes: `h`, `μ`, the rows'
  padding (each word `lo ‖ hi`, little-endian), `τ`'s salt fitted to `TAU_SALT_BYTES` and each leaf's salt fitted to
  `192`. `Draws.ofBytes` reads back the tape's pads (`Frame.padsOf_tapeBytes`) and leaf salts
  (`Frame.innerSalt_tapeBytes`).
* `Loop.hiBytes`: a rep's salts for levels ≥ 1 from a tape, each level's `2^d` salts in order; `Finish.saltAt` reads
  back each one of `192` bytes (`Loop.saltAt_hiBytes`), as the tape's are on a table's schedule
  (`Masked.tapeSalt_size`).
* `Loop.tapeDraws`, `Loop.tapeSrcHi`: at statement `x`, the frames' samples, and `Loop.tapeSrc` with those salts.
* **`Loop.src_tape`**: `Loop.Src` holds at every `x` and `w`, at every release whose samples are `Loop.tapeDraws x` and
  the source `Loop.tapeSrcHi` (`Loop.level0Src_tapeSrcHi`, `Loop.repSrc_tape`).
* **`LeanRelease.trustsStop_tape`**: so `trustsStop_finHonest` holds with no premise on the loop at such a release, at
  every relation over the one statement `x`. The samples read no statement, so one release serves one statement's
  layouts.
-/

@[expose] public section

namespace Flock.Firewall.Frame

open Claims.ZKServer (get!_list size_list)
open Level0 (wordByte wordAt_of extract_of_get)

/-- Byte `p` of table `i`'s rep `r` samples from the tape `T` at layout `l`, in `Draws.ofBytes`'s layout: `h`, `μ` and the
rows' padding, each word `lo ‖ hi` little-endian, then `τ`'s salt fitted to `TAU_SALT_BYTES` and each leaf's salt fitted
to `192`. -/
def tapeByte (T : FlockComplete.TapeZk) (i r : Nat) (l : Zk.PadLayout) (p : Nat) : UInt8 :=
  if p < 16 * l.len then wordByte (T.h i r (p / 16)) (p % 16)
  else if p < 32 * l.len then wordByte (T.mu i r ((p - 16 * l.len) / 16)) ((p - 16 * l.len) % 16)
  else if p < 32 * l.len + 16 * Zk.K_PAD then
    wordByte (T.padH i r ((p - 32 * l.len) / 16)) ((p - 32 * l.len) % 16)
  else if p < 32 * l.len + 32 * Zk.K_PAD then
    wordByte (T.padMu i r ((p - (32 * l.len + 16 * Zk.K_PAD)) / 16)) ((p - (32 * l.len + 16 * Zk.K_PAD)) % 16)
  else if p < 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES then
    (FlockComplete.fitBytes Zk.TAU_SALT_BYTES (T.tauSalt i r)).get! (p - (32 * l.len + 32 * Zk.K_PAD))
  else
    (FlockComplete.fitBytes 192
        (T.innerSalt i r ((p - (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES)) / 192))).get!
      ((p - (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES)) % 192)

/-- **Table `i`'s rep `r` samples from the tape `T`** at layout `l`: `drawBytes l` bytes in `Draws.ofBytes`'s layout. -/
def tapeBytes (T : FlockComplete.TapeZk) (i r : Nat) (l : Zk.PadLayout) : ByteArray :=
  ByteArray.mk ((List.range (drawBytes l)).map (tapeByte T i r l)).toArray

theorem drawBytes_eq (l : Zk.PadLayout) :
    drawBytes l = 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * codeLen l := by
  show 16 * (2 * l.len + 2 * Zk.K_PAD) + Zk.TAU_SALT_BYTES + 192 * codeLen l = _
  omega

theorem tapeBytes_size (T : FlockComplete.TapeZk) (i r : Nat) (l : Zk.PadLayout) :
    (tapeBytes T i r l).size = drawBytes l := by
  unfold tapeBytes
  rw [size_list]
  simp

theorem tapeBytes_get (T : FlockComplete.TapeZk) (i r : Nat) (l : Zk.PadLayout) {p : Nat} (hp : p < drawBytes l) :
    (tapeBytes T i r l).get! p = tapeByte T i r l p := by
  unfold tapeBytes
  rw [get!_list, getElem!_pos _ p (by simpa using hp)]
  simp

section Bytes

variable (T : FlockComplete.TapeZk) (i r : Nat) (l : Zk.PadLayout) {c k : Nat}

theorem tapeByte_h (hc : c < l.len) (hk : k < 16) : tapeByte T i r l (0 + 16 * c + k) = wordByte (T.h i r c) k := by
  unfold tapeByte
  rw [ite_eq_left (show 0 + 16 * c + k < 16 * l.len by omega), show (0 + 16 * c + k) / 16 = c by omega,
    show (0 + 16 * c + k) % 16 = k by omega]

theorem tapeByte_mu (hc : c < l.len) (hk : k < 16) :
    tapeByte T i r l (16 * l.len + 16 * c + k) = wordByte (T.mu i r c) k := by
  unfold tapeByte
  rw [ite_eq_right (show ¬ 16 * l.len + 16 * c + k < 16 * l.len by omega),
    ite_eq_left (show 16 * l.len + 16 * c + k < 32 * l.len by omega),
    show (16 * l.len + 16 * c + k - 16 * l.len) / 16 = c by omega,
    show (16 * l.len + 16 * c + k - 16 * l.len) % 16 = k by omega]

theorem tapeByte_padH (hc : c < Zk.K_PAD) (hk : k < 16) :
    tapeByte T i r l (32 * l.len + 16 * c + k) = wordByte (T.padH i r c) k := by
  unfold tapeByte
  rw [ite_eq_right (show ¬ 32 * l.len + 16 * c + k < 16 * l.len by omega),
    ite_eq_right (show ¬ 32 * l.len + 16 * c + k < 32 * l.len by omega),
    ite_eq_left (show 32 * l.len + 16 * c + k < 32 * l.len + 16 * Zk.K_PAD by omega),
    show (32 * l.len + 16 * c + k - 32 * l.len) / 16 = c by omega,
    show (32 * l.len + 16 * c + k - 32 * l.len) % 16 = k by omega]

theorem tapeByte_padMu (hc : c < Zk.K_PAD) (hk : k < 16) :
    tapeByte T i r l (32 * l.len + 16 * Zk.K_PAD + 16 * c + k) = wordByte (T.padMu i r c) k := by
  unfold tapeByte
  rw [ite_eq_right (show ¬ 32 * l.len + 16 * Zk.K_PAD + 16 * c + k < 16 * l.len by omega),
    ite_eq_right (show ¬ 32 * l.len + 16 * Zk.K_PAD + 16 * c + k < 32 * l.len by omega),
    ite_eq_right (show ¬ 32 * l.len + 16 * Zk.K_PAD + 16 * c + k < 32 * l.len + 16 * Zk.K_PAD by omega),
    ite_eq_left (show 32 * l.len + 16 * Zk.K_PAD + 16 * c + k < 32 * l.len + 32 * Zk.K_PAD by omega),
    show (32 * l.len + 16 * Zk.K_PAD + 16 * c + k - (32 * l.len + 16 * Zk.K_PAD)) / 16 = c by omega,
    show (32 * l.len + 16 * Zk.K_PAD + 16 * c + k - (32 * l.len + 16 * Zk.K_PAD)) % 16 = k by omega]

theorem tapeByte_tau (hk : k < Zk.TAU_SALT_BYTES) :
    tapeByte T i r l (32 * l.len + 32 * Zk.K_PAD + k) =
      (FlockComplete.fitBytes Zk.TAU_SALT_BYTES (T.tauSalt i r)).get! k := by
  unfold tapeByte
  rw [ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + k < 16 * l.len by omega),
    ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + k < 32 * l.len by omega),
    ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + k < 32 * l.len + 16 * Zk.K_PAD by omega),
    ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + k < 32 * l.len + 32 * Zk.K_PAD by omega),
    ite_eq_left (show 32 * l.len + 32 * Zk.K_PAD + k < 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES by omega),
    Nat.add_sub_cancel_left]

theorem tapeByte_salt {q : Nat} (hk : k < 192) :
    tapeByte T i r l (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k) =
      (FlockComplete.fitBytes 192 (T.innerSalt i r q)).get! k := by
  unfold tapeByte
  rw [ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k < 16 * l.len by omega),
    ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k < 32 * l.len by omega),
    ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k <
      32 * l.len + 16 * Zk.K_PAD by omega),
    ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k <
      32 * l.len + 32 * Zk.K_PAD by omega),
    ite_eq_right (show ¬ 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k <
      32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES by omega),
    show (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k -
      (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES)) / 192 = q by omega,
    show (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + k -
      (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES)) % 192 = k by omega]

end Bytes

/-- `words` reads back `n` words written from byte `o`. -/
theorem words_of (b : ByteArray) (o n : Nat) (f : Nat → F128)
    (h : ∀ c < n, ∀ k < 16, b.get! (o + 16 * c + k) = wordByte (f c) k) : words b o n = (Array.range n).map f := by
  unfold words
  exact Array.map_congr_left fun c hc => wordAt_of b (o + 16 * c) (f c) (h c (Array.mem_range.1 hc))

/-- **The tape's samples read back as its pads**: `Draws.ofBytes` on `tapeBytes` has the tape's `h`, `μ`, rows' padding
and `τ`'s salt, with `h_ab` and `w` set as `padsOf` sets them. -/
theorem padsOf_tapeBytes (T : FlockComplete.TapeZk) (i r : Nat) (l : Zk.PadLayout) :
    FlockComplete.padsOf T i r l = Claims.ZKServer.drawsPads (Draws.ofBytes l (tapeBytes T i r l)) := by
  have hD := drawBytes_eq l
  have hH : words (tapeBytes T i r l) 0 l.len = (Array.range l.len).map (T.h i r) :=
    words_of _ _ _ _ fun c hc k hk => by
      rw [tapeBytes_get T i r l (by rw [hD]; omega)]
      exact tapeByte_h T i r l hc hk
  have hM : words (tapeBytes T i r l) (16 * l.len) l.len = (Array.range l.len).map (T.mu i r) :=
    words_of _ _ _ _ fun c hc k hk => by
      rw [tapeBytes_get T i r l (by rw [hD]; omega)]
      exact tapeByte_mu T i r l hc hk
  have hPH : words (tapeBytes T i r l) (32 * l.len) Zk.K_PAD = (Array.range Zk.K_PAD).map (T.padH i r) :=
    words_of _ _ _ _ fun c hc k hk => by
      rw [tapeBytes_get T i r l (by rw [hD]; omega)]
      exact tapeByte_padH T i r l hc hk
  have hPM : words (tapeBytes T i r l) (32 * l.len + 16 * Zk.K_PAD) Zk.K_PAD =
      (Array.range Zk.K_PAD).map (T.padMu i r) :=
    words_of _ _ _ _ fun c hc k hk => by
      rw [tapeBytes_get T i r l (by rw [hD]; omega)]
      exact tapeByte_padMu T i r l hc hk
  have hT : (tapeBytes T i r l).extract (32 * l.len + 32 * Zk.K_PAD)
      (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES) =
      FlockComplete.fitBytes Zk.TAU_SALT_BYTES (T.tauSalt i r) := by
    have e := extract_of_get (tapeBytes T i r l) (32 * l.len + 32 * Zk.K_PAD)
      (FlockComplete.fitBytes Zk.TAU_SALT_BYTES (T.tauSalt i r))
      (by rw [FlockComplete.JsonRT.fitBytes_size, tapeBytes_size, hD]; omega) fun k hk => by
        rw [FlockComplete.JsonRT.fitBytes_size] at hk
        rw [tapeBytes_get T i r l (by rw [hD]; omega)]
        exact tapeByte_tau T i r l hk
    rwa [FlockComplete.JsonRT.fitBytes_size] at e
  unfold FlockComplete.padsOf Claims.ZKServer.drawsPads Draws.ofBytes
  simp only [hH, hM, hPH, hPM, hT]

/-- **The tape's samples read back as its leaf salts**: each of the `codeLen` leaves' `192` bytes of `Draws.ofBytes`'s
salts on `tapeBytes` is the tape's leaf salt, fitted. -/
theorem innerSalt_tapeBytes (T : FlockComplete.TapeZk) (i r : Nat) (l : Zk.PadLayout) {q : Nat} (hq : q < codeLen l) :
    (Draws.ofBytes l (tapeBytes T i r l)).salts.extract (SALT * q) (SALT * q + SALT) =
      FlockComplete.fitBytes 192 (T.innerSalt i r q) := by
  have hD := drawBytes_eq l
  have hS : SALT = 192 := rfl
  have hq' : 192 * (q + 1) ≤ 192 * codeLen l := Nat.mul_le_mul (le_refl _) (Nat.succ_le_of_lt hq)
  show ((tapeBytes T i r l).extract (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES)
      (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + SALT * codeLen l)).extract (SALT * q) (SALT * q + SALT) = _
  rw [ByteArray.extract_extract, hS, Nat.min_eq_left (by omega),
    show 32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + (192 * q + 192) =
      32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q + 192 by omega]
  have e := extract_of_get (tapeBytes T i r l) (32 * l.len + 32 * Zk.K_PAD + Zk.TAU_SALT_BYTES + 192 * q)
    (FlockComplete.fitBytes 192 (T.innerSalt i r q))
    (by rw [FlockComplete.JsonRT.fitBytes_size, tapeBytes_size, hD]; omega) fun k hk => by
      rw [FlockComplete.JsonRT.fitBytes_size] at hk
      rw [tapeBytes_get T i r l (by rw [hD]; omega)]
      exact tapeByte_salt T i r l hk
  rwa [FlockComplete.JsonRT.fitBytes_size] at e

end Flock.Firewall.Frame

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.ZK FlockSoundness.Discharge CoinLayer

/-- **The tape's salts on a table's levels ≥ 1 are `192` bytes** at the executable's layout: at every level `1 ≤ l` of
table `j`'s schedule and position `p` of its tree, rep `r`'s salt is a cell's (`hiSalt_exe`). -/
theorem tapeSalt_size (x : Msg) (t : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {l p : ℕ} (h1 : 1 ≤ l)
    (h2 : l < (zkTab x j).sch.levels.size) (hp : p < 2 ^ ((zkTab x j).sch.levels.getD l default).d) :
    ((UE.tapeZk t).salt j.val r.val l p).size = 192 := by
  obtain ⟨-, -, -, -, -, -, hseam, -, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hat : HiAt (zkTab x j).table.sch.levels l p := by
    have e : (zkTab x j).table.sch.levels.getD l default =
        FlockSoundness.Refine.levelOf ((zkTab x j).sch.levels.getD l default) := by
      show ((zkTab x j).sch.levels.toList.map FlockSoundness.Refine.levelOf).getD l default = _
      simp [List.getD_eq_getElem?_getD, Array.getD_eq_getD_getElem?, h2]
    refine ⟨h1, ?_, ?_⟩
    · show l < ((zkTab x j).sch.levels.toList.map FlockSoundness.Refine.levelOf).length
      simpa using h2
    · rw [e, ← (hseam l h2).2.2.2.2]
      exact hp
  have hk : hiOff (zkTab x j).table.sch.levels l + p < Composed.hiCount (zkTab x j).table := by
    have := hiOff_add_lt hat
    unfold Composed.hiCount
    omega
  show ((exeTape t).salt j.val r.val l p).size = 192
  rw [← hiSalt_exe PE x t j r hat]
  unfold hiSalt
  rw [ite_eq_left hat, fAt_lt _ _ hk]
  exact saltBytes_size _

end Claims.ZKMal.Masked

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- `n` salts of `192` bytes in order, salt `p` the bytes of `f p`. -/
def Loop.saltBlocks (f : ℕ → ByteArray) (n : ℕ) : ByteArray :=
  ByteArray.mk ((List.range (192 * n)).map fun b => (f (b / 192)).get! (b % 192)).toArray

theorem Loop.saltAt_saltBlocks (f : ℕ → ByteArray) {n p : ℕ} (hp : p < n) (hs : (f p).size = 192) :
    _root_.Flock.Firewall.Finish.saltAt (Loop.saltBlocks f n) p = f p := by
  unfold _root_.Flock.Firewall.Finish.saltAt Loop.saltBlocks
  have h192 : 192 * (p + 1) ≤ 192 * n := Nat.mul_le_mul (le_refl _) (Nat.succ_le_of_lt hp)
  have e := _root_.Flock.Firewall.Level0.extract_of_get
    (ByteArray.mk ((List.range (192 * n)).map fun b => (f (b / 192)).get! (b % 192)).toArray) (192 * p) (f p)
    (by rw [hs, size_list, List.length_map, List.length_range]; omega) fun k hk => by
      rw [hs] at hk
      rw [get!_list, getElem!_pos _ (192 * p + k) (by simp only [List.length_map, List.length_range]; omega)]
      simp only [List.getElem_map, List.getElem_range]
      rw [show (192 * p + k) / 192 = p by omega, show (192 * p + k) % 192 = k by omega]
  rwa [hs] at e

/-- **A rep's salts for levels ≥ 1 from the tape `T`**: table `i`'s rep `r` at the schedule `sch`, the `2^d` salts of
each level `1` to `R` in order (`192·2^d` bytes a level where each is `192`, as `Finish.hiSaltsOf` reads them). -/
def Loop.hiBytes (T : FlockComplete.TapeZk) (i r : ℕ) (sch : _root_.Flock.Schedule) : Array ByteArray :=
  (Array.range (sch.levels.size - 1)).map fun k =>
    Loop.saltBlocks (T.salt i r (k + 1)) (2 ^ (sch.levels.getD (k + 1) default).d)

theorem Loop.saltAt_hiBytes (T : FlockComplete.TapeZk) (i r : ℕ) (sch : _root_.Flock.Schedule) {l p : ℕ}
    (h1 : 1 ≤ l) (h2 : l < sch.levels.size) (hp : p < 2 ^ (sch.levels.getD l default).d)
    (hs : (T.salt i r l p).size = 192) :
    _root_.Flock.Firewall.Finish.saltAt ((Loop.hiBytes T i r sch).getD (l - 1) ByteArray.empty) p = T.salt i r l p := by
  have hl : l - 1 < sch.levels.size - 1 := by omega
  have hg : (Loop.hiBytes T i r sch).getD (l - 1) ByteArray.empty =
      Loop.saltBlocks (T.salt i r l) (2 ^ (sch.levels.getD l default).d) := by
    unfold Loop.hiBytes
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, hl, ↓reduceIte,
      Option.map_some, Option.getD_some]
    rw [show l - 1 + 1 = l by omega]
  rw [hg]
  exact Loop.saltAt_saltBlocks _ hp hs

/-- **The frames' samples at statement `x`**: rep stream `k`'s, of table `k / 2`, the tape's (`Frame.tapeBytes`) at its
layout. -/
noncomputable def Loop.tapeDraws (x : Msg)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (k : ℕ) : ByteArray :=
  if h : k / 2 < Masked.jOf x then
    _root_.Flock.Firewall.Frame.tapeBytes (Masked.UE.tapeZk ρ.2) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, h⟩)
  else ByteArray.empty

/-- **A source that samples the tape's level 0 and levels ≥ 1's salts**: `Loop.tapeSrc`, with rep stream `k`'s salts for
levels ≥ 1 the tape's (`Loop.hiBytes`). -/
noncomputable def Loop.tapeSrcHi (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (k : ℕ) : V (Array F128 × ByteArray × Array ByteArray) :=
  if h : k / 2 < Masked.jOf x then
    .ok ((Masked.linkIn x w ρ.2 ⟨k / 2, h⟩).g, _root_.Flock.Firewall.Level0.inBytes (Masked.lv0 x ⟨k / 2, h⟩)
      (Masked.zkTab x ⟨k / 2, h⟩).tPad (Masked.linkIn x w ρ.2 ⟨k / 2, h⟩),
      Loop.hiBytes (Masked.UE.tapeZk ρ.2) (k / 2) (k % 2) (Masked.zkTab x ⟨k / 2, h⟩).sch)
  else .error "no table"

theorem Loop.level0At_tapeSrcHi {x : Msg} {w : Flock.Wit}
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (sts : Array _root_.Flock.Stmt) (k : ℕ) :
    Loop.level0At Loop.tapeSrcHi x w ρ sts k = Loop.level0At Loop.tapeSrc x w ρ sts k := by
  unfold Loop.level0At Loop.tapeSrcHi Loop.tapeSrc
  by_cases h : k / 2 < Masked.jOf x
  · rw [dite_eq_left h, dite_eq_left h]
    all_goals rfl
  · rw [dite_eq_right h, dite_eq_right h]

/-- **`Loop.tapeSrcHi` meets `Loop.Level0Src`** at every `x` and `w`: its level 0 is `Loop.tapeSrc`'s
(`Loop.level0Src_tapeSrc`). -/
theorem Loop.level0Src_tapeSrcHi (x : Msg) (w : Flock.Wit) : Loop.Level0Src Loop.tapeSrcHi x w := fun ν t j r => by
  obtain ⟨l, hl, hin⟩ := Loop.level0Src_tapeSrc x w ν t j r
  exact ⟨l, (Loop.level0At_tapeSrcHi (ν, t) (Masked.stsOf x) _).trans hl, hin⟩

section Rep

variable (x : Msg) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
  (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)

theorem Loop.tapeDraws_rep :
    Loop.tapeDraws x (ν, t) (2 * j.val + r.val) =
      _root_.Flock.Firewall.Frame.tapeBytes (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) := by
  have hk : (2 * j.val + r.val) / 2 = j.val := by omega
  have hm : (2 * j.val + r.val) % 2 = r.val := by omega
  have hj : (2 * j.val + r.val) / 2 < Masked.jOf x := by
    have := j.isLt
    omega
  have hJ : (⟨(2 * j.val + r.val) / 2, hj⟩ : Fin (Masked.jOf x)) = j := Fin.ext hk
  unfold Loop.tapeDraws
  rw [dite_eq_left hj, hJ, hk, hm]

theorem Loop.tapeSrcHi_hi (w : Flock.Wit) {g : Array F128} {d : ByteArray} {hi : Array ByteArray}
    (h : Loop.tapeSrcHi x w (ν, t) (2 * j.val + r.val) = .ok (g, d, hi)) :
    hi = Loop.hiBytes (Masked.UE.tapeZk t) j.val r.val (Masked.zkTab x j).sch := by
  have hk : (2 * j.val + r.val) / 2 = j.val := by omega
  have hm : (2 * j.val + r.val) % 2 = r.val := by omega
  have hj : (2 * j.val + r.val) / 2 < Masked.jOf x := by
    have := j.isLt
    omega
  have hJ : (⟨(2 * j.val + r.val) / 2, hj⟩ : Fin (Masked.jOf x)) = j := Fin.ext hk
  unfold Loop.tapeSrcHi at h
  rw [dite_eq_left hj, hJ, hk, hm] at h
  exact (Prod.mk.inj (Prod.mk.inj (Except.ok.inj h)).2).2.symm

end Rep

/-- **`Loop.RepSrc` is satisfiable**: at every statement `x` and witness `w`, a release whose frames sample
`Loop.tapeDraws x` and the source `Loop.tapeSrcHi` meet it. The samples are `drawBytes` long and read back as the tape's
pads and leaf salts (`Frame.padsOf_tapeBytes`, `Frame.innerSalt_tapeBytes`), and the source's salts for levels ≥ 1 read
back as the tape's (`Loop.saltAt_hiBytes`, each `192` bytes: `Masked.tapeSalt_size`). -/
theorem Loop.repSrc_tape
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit) (hd : L.draws = Loop.tapeDraws x) : Loop.RepSrc L Loop.tapeSrcHi x w := by
  intro ν t j r
  rw [hd, Loop.tapeDraws_rep x ν t j r]
  refine ⟨_root_.Flock.Firewall.Frame.tapeBytes_size _ _ _ _, _root_.Flock.Firewall.Frame.padsOf_tapeBytes _ _ _ _,
    fun q hq => (_root_.Flock.Firewall.Frame.innerSalt_tapeBytes _ _ _ _ hq).symm, ?_⟩
  intro g d hi hs l h1 h2 p hp
  rw [Loop.tapeSrcHi_hi x ν t j r w hs]
  exact Loop.saltAt_hiBytes _ _ _ _ h1 h2 hp (Masked.tapeSalt_size x t j r h1 h2 hp)

/-- **`Loop.Src` is satisfiable**: at every statement `x` and witness `w`, both halves hold at the source
`Loop.tapeSrcHi` and every release whose frames sample `Loop.tapeDraws x`. -/
theorem Loop.src_tape
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (x : Msg) (w : Flock.Wit) (hd : L.draws = Loop.tapeDraws x) : Loop.Src L Loop.tapeSrcHi x w :=
  ⟨Loop.level0Src_tapeSrcHi x w, Loop.repSrc_tape L x w hd⟩

namespace LeanRelease

open Loop

/-- **The transport's trusts at the honest session loop, with no premise on the loop**: `trustsStop_finHonest` at a
release `L₁` whose frames sample the tape's samples at `x₀` (`Loop.tapeDraws`) and the source `Loop.tapeSrcHi`, which meet
`Loop.Src` there (`Loop.src_tape`), at every relation over `x₀` alone. -/
theorem trustsStop_tape (L₁ : LeanRelease Flock.Wit Tp) (x₀ : Msg) (hd : L₁.draws = Loop.tapeDraws x₀)
    (Rl : Rel Flock.Wit)
    (hRl : ∀ x w, Rl x w → Serves x ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x ∧
      Masked.RZk x w)
    (hx : ∀ x w, Rl x w → x = x₀) :
    TrustsStop finS finS.honestVals (ofLoopLink (withInner Masked.dataE (acceptCross L₁) (realOf Loop.tapeSrcHi)
        (finHonest (ofLoopLink (acceptCross L₁) Loop.tapeSrcHi (realOf Loop.tapeSrcHi)))) Loop.tapeSrcHi
        (realOf Loop.tapeSrcHi)) CoinOpen.shaH
      (valsWk finS.honestVals (ofLoopLink (withInner Masked.dataE (acceptCross L₁) (realOf Loop.tapeSrcHi)
        (finHonest (ofLoopLink (acceptCross L₁) Loop.tapeSrcHi (realOf Loop.tapeSrcHi)))) Loop.tapeSrcHi
        (realOf Loop.tapeSrcHi))) Rl :=
  trustsStop_finHonest L₁ Loop.tapeSrcHi Rl hRl fun x w h => by
    rw [hx x w h]
    exact Loop.src_tape _ x₀ w (by exact hd)

end LeanRelease

end Claims.ZKServer
