module

public import Proofs.Flock.SessionLoop.Tables
public import Proofs.Flock.Claims.ZK.FrameParse

/-!
# The loop's level-0 source is satisfiable

`Loop.Level0Src`, the level-0 half of `Loop.Src` (`flock-firewall-os-src`), holds at every statement and witness for
some source: the one that hands each rep stream its table's packed witness at the tape's mask words and the tape's
level-0 samples, written in `Level0.ofDraws`'s layout.

* `Level0.inBytes`: a `Link.In`'s extra lanes, lane paddings and salts at level `lv` with `t_pad` paddings, as `ofDraws`
  reads them: each extra lane's `2^cols` words, each of the `2^lanes + 4` lanes' `t_pad` words, every word `lo ‖ hi`
  little-endian, then `192` bytes of salt for each of the `2^d` tree positions.
* **`Level0.ofDraws_inBytes`**: where a `Link.In`'s packed table has `2^(lanes + cols)` words, its extra lanes and lane
  paddings are `0` past those sizes and its salts are `192` bytes at the tree positions and empty past them, `ofDraws`
  on its packed table and `inBytes` returns a level 0 whose inputs (`toIn`) are it.
* `Masked.linkIn` has that shape at every table, so **`Loop.level0Src_tapeSrc`**: the source `Loop.tapeSrc` meets
  `Loop.Level0Src` at every `x` and `w`.
-/

@[expose] public section

namespace Flock.Firewall.Level0

open Claims.ZKServer (get!_list size_list u64OfLE_of_get)

/-- Byte `b < 16` of the word `a` as `Link.wordAt` reads it: `lo ‖ hi`, each little-endian. -/
def wordByte (a : F128) (b : Nat) : UInt8 :=
  if b < 8 then (a.lo >>> (8 * b).toUInt64).toUInt8 else (a.hi >>> (8 * (b - 8)).toUInt64).toUInt8

/-- Byte `p` of `i`'s level-0 samples at level `lv` with `t_pad` paddings, in `ofDraws`'s layout: the four extra lanes'
`2^cols` words, every lane's `t_pad` padding words, then a `192`-byte salt per tree position. -/
def inByte (lv : Level) (tPad : Nat) (i : Link.In) (p : Nat) : UInt8 :=
  if p < 16 * Zk.EXTRA_LANES * 2 ^ lv.cols then
    wordByte (i.extra (p / (16 * 2 ^ lv.cols)) (p % (16 * 2 ^ lv.cols) / 16)) (p % 16)
  else if p < 16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad then
    wordByte (i.lanePad ((p - 16 * Zk.EXTRA_LANES * 2 ^ lv.cols) / (16 * tPad))
      ((p - 16 * Zk.EXTRA_LANES * 2 ^ lv.cols) % (16 * tPad) / 16)) ((p - 16 * Zk.EXTRA_LANES * 2 ^ lv.cols) % 16)
  else
    (i.salt ((p - (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad)) / 192)).get!
      ((p - (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad)) % 192)

/-- **`i`'s level-0 samples** at level `lv` with `t_pad` paddings: `drawSize lv tPad` bytes in `ofDraws`'s layout. -/
def inBytes (lv : Level) (tPad : Nat) (i : Link.In) : ByteArray :=
  ByteArray.mk ((List.range (drawSize lv tPad)).map (inByte lv tPad i)).toArray

/-- The extra lanes `ofDraws` reads from the samples `d` at level `lv`. -/
def drawnExtra (lv : Level) (d : ByteArray) : Array (Array F128) :=
  (Array.range Zk.EXTRA_LANES).map fun e => Link.wordsAt d (16 * 2 ^ lv.cols * e) (2 ^ lv.cols)

/-- The lane paddings `ofDraws` reads from the samples `d` at level `lv` with `t_pad` paddings. -/
def drawnPads (lv : Level) (tPad : Nat) (d : ByteArray) : Array (Array F128) :=
  (Array.range (2 ^ lv.lanes + Zk.EXTRA_LANES)).map fun l =>
    Link.wordsAt d (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * tPad * l) tPad

/-- The salts `ofDraws` reads from the samples `d` at level `lv` with `t_pad` paddings. -/
def drawnSalts (lv : Level) (tPad : Nat) (d : ByteArray) : ByteArray :=
  d.extract (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad) d.size

theorem inBytes_size (lv : Level) (tPad : Nat) (i : Link.In) : (inBytes lv tPad i).size = drawSize lv tPad := by
  unfold inBytes
  rw [size_list]
  simp

theorem inBytes_get (lv : Level) (tPad : Nat) (i : Link.In) {p : Nat} (hp : p < drawSize lv tPad) :
    (inBytes lv tPad i).get! p = inByte lv tPad i p := by
  unfold inBytes
  rw [get!_list, getElem!_pos _ p (by simpa using hp)]
  simp

theorem wordAt_of (b : ByteArray) (o : Nat) (a : F128) (h : ∀ k < 16, b.get! (o + k) = wordByte a k) :
    Link.wordAt b o = a := by
  obtain ⟨lo, hi⟩ := a
  unfold Link.wordAt
  congr 1
  · exact u64OfLE_of_get b o lo fun k hk => by
      rw [h k (by omega)]
      unfold wordByte
      rw [if_pos hk]
  · exact u64OfLE_of_get b (o + 8) hi fun k hk => by
      rw [Nat.add_assoc, h (8 + k) (by omega)]
      unfold wordByte
      rw [if_neg (show ¬ 8 + k < 8 by omega), Nat.add_sub_cancel_left]

theorem getElem!_default (c : Nat) : (default : Array F128)[c]! = 0 := by
  rw [getElem!_neg (default : Array F128) c (Nat.not_lt_zero c)]
  rfl

theorem rangeMap_get! {α : Type} [Inhabited α] (f : Nat → α) {n i : Nat} (h : i < n) :
    ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using h), Array.getElem_map, Array.getElem_range]

theorem wordsAt_out (b : ByteArray) (o n c : Nat) (hc : ¬ c < n) : (Link.wordsAt b o n)[c]! = 0 := by
  unfold Link.wordsAt
  rw [getElem!_neg _ _ (by simpa using hc)]
  rfl

theorem wordsAt_inBytes (lv : Level) (tPad : Nat) (i : Link.In) {o n c : Nat} (a : F128) (hc : c < n)
    (hlt : o + 16 * c + 16 ≤ drawSize lv tPad) (hb : ∀ b < 16, inByte lv tPad i (o + 16 * c + b) = wordByte a b) :
    (Link.wordsAt (inBytes lv tPad i) o n)[c]! = a := by
  unfold Link.wordsAt
  rw [rangeMap_get! _ hc]
  exact wordAt_of _ _ _ fun b hb' => by rw [inBytes_get _ _ _ (by omega), hb b hb']

theorem drawPos_lt {A B b nA nB : Nat} (hA : A < nA) (hB : B < nB) (hb : b < 16) :
    16 * nB * A + 16 * B + b < 16 * nA * nB := by
  have h1 : 16 * nB * (A + 1) ≤ 16 * nB * nA := Nat.mul_le_mul (le_refl _) (Nat.succ_le_of_lt hA)
  have h2 : 16 * nB * nA = 16 * nA * nB := by ring
  rw [Nat.mul_add, Nat.mul_one] at h1
  omega

theorem drawDivmod {A B b nB : Nat} (hB : B < nB) (hb : b < 16) :
    (16 * nB * A + 16 * B + b) / (16 * nB) = A ∧ (16 * nB * A + 16 * B + b) % (16 * nB) / 16 = B ∧
      (16 * nB * A + 16 * B + b) % 16 = b := by
  have h0 : 0 < 16 * nB := by omega
  have hl : 16 * B + b < 16 * nB := by omega
  have e1 : 16 * nB * A + 16 * B + b = (16 * B + b) + 16 * nB * A := by ring
  refine ⟨?_, ?_, ?_⟩
  · rw [e1, Nat.add_mul_div_left _ _ h0, Nat.div_eq_of_lt hl, Nat.zero_add]
  · rw [e1, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hl, Nat.add_comm,
      Nat.add_mul_div_left _ _ (by norm_num : 0 < 16), Nat.div_eq_of_lt hb, Nat.zero_add]
  · rw [show 16 * nB * A + 16 * B + b = b + 16 * (nB * A + B) by ring, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt hb]

theorem extract_of_get (d : ByteArray) (o : Nat) (y : ByteArray) (hn : o + y.size ≤ d.size)
    (h : ∀ k < y.size, d.get! (o + k) = y.get! k) : d.extract o (o + y.size) = y := by
  apply ByteArray.ext
  rw [ByteArray.data_extract]
  apply Array.ext
  · rw [Array.size_extract, ByteArray.size_data, ByteArray.size_data]
    omega
  · intro k h1 h2
    have h2' : k < y.size := h2
    have hk := h k h2'
    rw [FlockSoundness.Discharge.ZkView.get!_eq, FlockSoundness.Discharge.ZkView.get!_eq,
      getElem!_pos d.data (o + k) (by rw [ByteArray.size_data]; omega), getElem!_pos y.data k h2] at hk
    rw [Array.getElem_extract]
    exact hk

theorem rows_size (lv : Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) :
    (Link.rows lv tPad g extra lanePad).size = 2 ^ lv.d := by
  unfold Link.rows Link.rowsFft Link.rowsDirect
  split <;> simp

theorem In_ext {a b : Link.In} (h1 : a.m = b.m) (h2 : a.g = b.g) (h3 : a.extra = b.extra)
    (h4 : a.lanePad = b.lanePad) (h5 : a.salt = b.salt) : a = b := by
  cases a
  cases b
  simp only [Link.In.mk.injEq]
  exact ⟨h1, h2, h3, h4, h5⟩

set_option maxHeartbeats 1000000 in
/-- **`ofDraws` on a table's inputs written in its layout gives them back**: where `i`'s packed table has
`2^(lanes + cols)` words, its extra lanes and lane paddings are `0` past level 0's sizes, and its salts are `192` bytes at
the `2^d` tree positions and empty past them, `ofDraws` on `i.g` and `inBytes` returns a level 0 whose inputs are `i`. -/
theorem ofDraws_inBytes {m : Nat} {s : _root_.Flock.Schedule} {tPad : Nat} {lv : Level}
    (hz : Zk.schedule m = .ok (s, tPad))
    (hlv : s.levels.getD 0 default = lv) (i : Link.In) (hm : i.m = m) (hg : i.g.size = 2 ^ (lv.lanes + lv.cols))
    (he : ∀ e c, ¬ (e < Zk.EXTRA_LANES ∧ c < 2 ^ lv.cols) → i.extra e c = 0)
    (hp : ∀ l k, ¬ (l < 2 ^ lv.lanes + Zk.EXTRA_LANES ∧ k < tPad) → i.lanePad l k = 0)
    (hs : ∀ q, q < 2 ^ lv.d → (i.salt q).size = 192) (hs' : ∀ q, ¬ q < 2 ^ lv.d → i.salt q = .empty) :
    ∃ l, ofDraws m i.g (inBytes lv tPad i) = .ok l ∧ l.toIn = i := by
  have h1 : (i.g.size != 2 ^ (lv.lanes + lv.cols)) = false := by rw [hg]; simp
  have h2 : ((inBytes lv tPad i).size != drawSize lv tPad) = false := by rw [inBytes_size]; simp
  have h3 : ∀ g E P S, ((build m lv tPad g E P S).rows.size != 2 ^ lv.d) = false := fun g E P S => by
    have hr : (build m lv tPad g E P S).rows.size = 2 ^ lv.d := by
      first
        | exact rows_size _ _ _ _ _
        | exact (congrArg Array.size (build_rows rfl)).trans (rows_size _ _ _ _ _)
    rw [hr]
    simp
  have hok : ofDraws m i.g (inBytes lv tPad i) = .ok (build m lv tPad i.g (drawnExtra lv (inBytes lv tPad i))
      (drawnPads lv tPad (inBytes lv tPad i)) (drawnSalts lv tPad (inBytes lv tPad i))) := by
    subst hlv
    unfold ofDraws
    rw [hz]
    simp only [bind, Except.bind, pure, Except.pure, h1, h2, h3, Bool.false_eq_true, ↓reduceIte]
    all_goals rfl
  have hE : ∀ e c, ((drawnExtra lv (inBytes lv tPad i))[e]!)[c]! = i.extra e c := by
    intro e c
    unfold drawnExtra
    by_cases he4 : e < Zk.EXTRA_LANES
    · rw [rangeMap_get! _ he4]
      by_cases hc : c < 2 ^ lv.cols
      · refine wordsAt_inBytes lv tPad i _ hc ?_ fun b hb => ?_
        · have := drawPos_lt (b := 15) he4 hc (by norm_num)
          unfold drawSize
          omega
        · obtain ⟨d1, d2, d3⟩ := drawDivmod (A := e) hc hb
          unfold inByte
          rw [if_pos (drawPos_lt he4 hc hb), d1, d2, d3]
      · rw [wordsAt_out _ _ _ _ hc]
        exact (he e c fun h => hc h.2).symm
    · rw [getElem!_neg ((Array.range Zk.EXTRA_LANES).map _) e (by simpa using he4), getElem!_default]
      exact (he e c fun h => he4 h.1).symm
  have hP : ∀ l k, ((drawnPads lv tPad (inBytes lv tPad i))[l]!)[k]! = i.lanePad l k := by
    intro l k
    unfold drawnPads
    by_cases hl : l < 2 ^ lv.lanes + Zk.EXTRA_LANES
    · rw [rangeMap_get! _ hl]
      by_cases hk : k < tPad
      · refine wordsAt_inBytes lv tPad i _ hk ?_ fun b hb => ?_
        · have := drawPos_lt (b := 15) hl hk (by norm_num)
          unfold drawSize
          omega
        · have hlt := drawPos_lt hl hk hb
          obtain ⟨d1, d2, d3⟩ := drawDivmod (A := l) hk hb
          have hn1 : ¬ (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * tPad * l + 16 * k + b <
              16 * Zk.EXTRA_LANES * 2 ^ lv.cols) := by omega
          have hn2 : 16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * tPad * l + 16 * k + b <
              16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad := by omega
          have hsub : 16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * tPad * l + 16 * k + b -
              16 * Zk.EXTRA_LANES * 2 ^ lv.cols = 16 * tPad * l + 16 * k + b := by omega
          unfold inByte
          rw [if_neg hn1, if_pos hn2, hsub, d1, d2, d3]
      · rw [wordsAt_out _ _ _ _ hk]
        exact (hp l k fun h => hk h.2).symm
    · rw [getElem!_neg ((Array.range (2 ^ lv.lanes + Zk.EXTRA_LANES)).map _) l (by simpa using hl),
        getElem!_default]
      exact (hp l k fun h => hl h.1).symm
  have hS : ∀ q, (if q < 2 ^ lv.d then (drawnSalts lv tPad (inBytes lv tPad i)).extract (192 * q) (192 * (q + 1))
      else ByteArray.empty) = i.salt q := by
    intro q
    by_cases hq : q < 2 ^ lv.d
    · rw [if_pos hq]
      unfold drawnSalts
      rw [ByteArray.extract_extract]
      have h192 : 192 * (q + 1) ≤ 192 * 2 ^ lv.d := Nat.mul_le_mul (le_refl _) (Nat.succ_le_of_lt hq)
      have hsq := hs q hq
      have hmin : min (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad +
          192 * (q + 1)) (inBytes lv tPad i).size =
          16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad + 192 * q +
            (i.salt q).size := by
        rw [inBytes_size, hsq]
        unfold drawSize
        omega
      rw [hmin]
      refine extract_of_get _ _ _ (by rw [inBytes_size, hsq]; unfold drawSize; omega) fun k hk => ?_
      rw [hsq] at hk
      have hn1 : ¬ (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad + 192 * q + k <
          16 * Zk.EXTRA_LANES * 2 ^ lv.cols) := by omega
      have hn2 : ¬ (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad + 192 * q + k <
          16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad) := by omega
      have hsub : 16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad + 192 * q + k -
          (16 * Zk.EXTRA_LANES * 2 ^ lv.cols + 16 * (2 ^ lv.lanes + Zk.EXTRA_LANES) * tPad) = 192 * q + k := by omega
      rw [inBytes_get _ _ _ (by unfold drawSize; omega)]
      unfold inByte
      rw [if_neg hn1, if_neg hn2, hsub, show (192 * q + k) / 192 = q by omega, show (192 * q + k) % 192 = k by omega]
    · rw [if_neg hq]
      exact (hs' q hq).symm
  exact ⟨_, hok, In_ext hm.symm rfl (funext fun e => funext fun c => hE e c) (funext fun l => funext fun k => hP l k)
    (funext fun q => hS q)⟩

end Flock.Firewall.Level0

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **A level-0 source that samples the tape's level 0**: at rep stream `k`, of table `k / 2`, the table's packed witness at
the tape's mask words (`Masked.linkIn`'s) and the tape's extra lanes, lane paddings and level-0 salts at level 0's sizes,
in `Level0.ofDraws`'s layout (`Level0.inBytes`); no salts for levels ≥ 1. -/
noncomputable def Loop.tapeSrc (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (k : ℕ) : V (Array F128 × ByteArray × Array ByteArray) :=
  if h : k / 2 < Masked.jOf x then
    .ok ((Masked.linkIn x w ρ.2 ⟨k / 2, h⟩).g, _root_.Flock.Firewall.Level0.inBytes (Masked.lv0 x ⟨k / 2, h⟩)
      (Masked.zkTab x ⟨k / 2, h⟩).tPad (Masked.linkIn x w ρ.2 ⟨k / 2, h⟩), #[])
  else .error "no table"

set_option maxHeartbeats 1000000 in
/-- **`Loop.Level0Src` is satisfiable**: at every statement `x` and witness `w`, the source `Loop.tapeSrc` meets it, so
at every table, nonce, tape and rep the level 0 the loop samples has `Masked.linkIn`'s inputs. -/
theorem Loop.level0Src_tapeSrc (x : Msg) (w : Flock.Wit) : Loop.Level0Src Loop.tapeSrc x w := by
  intro ν t j r
  have hk : (2 * j.val + r.val) / 2 = j.val := by omega
  have hj : (2 * j.val + r.val) / 2 < Masked.jOf x := by
    have := j.isLt
    omega
  have hJ : (⟨(2 * j.val + r.val) / 2, hj⟩ : Fin (Masked.jOf x)) = j := Fin.ext hk
  have hsrc : Loop.tapeSrc x w (ν, t) (2 * j.val + r.val) = .ok ((Masked.linkIn x w t j).g,
      _root_.Flock.Firewall.Level0.inBytes (Masked.lv0 x j) (Masked.zkTab x j).tPad (Masked.linkIn x w t j), #[]) := by
    unfold Loop.tapeSrc
    rw [dif_pos hj, hJ]
    all_goals rfl
  obtain ⟨-, -, -, -, h5, hm⟩ := Masked.tabC_z Masked.PE x j
  have hz := Masked.fits_zkTab Masked.PE (Masked.zkTab x j)
  have hg : (Masked.linkIn x w t j).g.size = 2 ^ ((Masked.lv0 x j).lanes + (Masked.lv0 x j).cols) := by
    have hmL : (Masked.ligSt x j).m = (Masked.tabC Masked.PE x j).S.m := by
      unfold Masked.ligSt
      rw [Masked.tabsOf_stAt]
      rfl
    rw [Masked.linkIn_g x w t j]
    unfold Masked.ligTab
    rw [FlockComplete.JsonRT.packed_size, hmL, ← hm, Nat.add_sub_cancel]
  have he : ∀ e c, ¬ (e < _root_.Flock.Zk.EXTRA_LANES ∧ c < 2 ^ (Masked.lv0 x j).cols) →
      (Masked.linkIn x w t j).extra e c = 0 := fun e c h => by
    show (if c < 2 ^ (Masked.lv0 x j).cols then (Masked.exeTape t).extra j.val e c else (0 : F128)) = 0
    by_cases hc : c < 2 ^ (Masked.lv0 x j).cols
    · have he4 : ¬ e < 4 := fun he4 => h ⟨he4, hc⟩
      rw [if_pos hc]
      show Masked.ext4 (fun r i' => Masked.fAt (0 : F128) ((Masked.cellAt t j.val).1.2.1.2 r i') c) (0 : F128) e = 0
      unfold Masked.ext4
      rw [if_neg (show ¬ e = 0 by omega), if_neg (show ¬ e = 1 by omega), if_neg (show ¬ e = 2 by omega),
        if_neg (show ¬ e = 3 by omega)]
    · rw [if_neg hc]
  have hp : ∀ l k, ¬ (l < 2 ^ (Masked.lv0 x j).lanes + _root_.Flock.Zk.EXTRA_LANES ∧ k < (Masked.zkTab x j).tPad) →
      (Masked.linkIn x w t j).lanePad l k = 0 := fun l k h => by
    show (if l < 2 ^ (Masked.lv0 x j).lanes + _root_.Flock.Zk.EXTRA_LANES ∧ k < (Masked.zkTab x j).tPad then
      (Masked.exeTape t).lanePad j.val l k else (0 : F128)) = 0
    rw [if_neg h]
  have hs : ∀ q, q < 2 ^ (Masked.lv0 x j).d → ((Masked.linkIn x w t j).salt q).size = 192 := fun q hq => by
    rw [Masked.salt_linkIn x w t j hq]
    have hq' : q < Masked.schMax + Masked.schMax :=
      lt_of_lt_of_le (hq.trans_eq h5.symm) (Nat.add_le_add hz.nS hz.nU)
    show (Masked.saltAt (Masked.cellAt t j.val) 0 0 q).size = 192
    unfold Masked.saltAt
    rw [if_pos rfl, if_pos rfl, Masked.fAt_lt _ _ hq']
    exact Masked.saltBytes_size _
  have hs' : ∀ q, ¬ q < 2 ^ (Masked.lv0 x j).d → (Masked.linkIn x w t j).salt q = ByteArray.empty := fun q hq => by
    show (if q < 2 ^ (Masked.lv0 x j).d then (Masked.exeTape t).toTape.salt j.val 0 0 q else ByteArray.empty) =
      ByteArray.empty
    rw [if_neg hq]
  unfold Loop.level0At
  rw [hsrc]
  simp only [bind, Except.bind]
  rw [Masked.stsOf_half x j r]
  exact _root_.Flock.Firewall.Level0.ofDraws_inBytes (m := (Masked.stAt x j).m) (s := (Masked.zkTab x j).sch)
    (tPad := (Masked.zkTab x j).tPad) (lv := Masked.lv0 x j) (Masked.zkTab x j).hz rfl (Masked.linkIn x w t j) rfl hg
    he hp hs hs'

end Claims.ZKServer
