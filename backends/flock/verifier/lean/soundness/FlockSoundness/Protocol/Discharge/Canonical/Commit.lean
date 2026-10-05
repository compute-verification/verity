import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.Canonical.Chain
import FlockSoundness.Discharge.Canonical.OutCopy
import FlockSoundness.Discharge.Hidden.Wide.Upv
import FlockSoundness.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# A canonical row's `b ‖ c` at its port

`Hidden.row_bc_tgt` and `HiddenExec.bc_site_commitH` for canonical rows: the `b ‖ c` at row `p`'s `hm96` output is the
commit string of `P(ℓ)` and `packBits` of the `ℓ` bits the witness holds at the row's message columns
(`Typed.msgCol`, bit `i` at byte `i / 8`'s bit `i % 8`), for any `ℓ`. `row_bc_tgt` takes `bits = 16 · words` and
`2 · words = 128 · chunks` (whole blocks); `bc_site_commitH` takes `sha512/row/v1` rows and `outNet = unitNet`.

* `rowBits`: the `ℓ` bits at row `p`'s message columns;
* `row_bc_C`: `row_bc_tgt` at any unsegmented bit row of the flat statement whose compressions are the ones its length
  needs (`chunks + 1 = ⌈(⌈ℓ / 8⌉ + 17) / 128⌉`, `HmRow.rowPorts`' count, `v2_of_parse`);
* `bc_site_commitC`: `bc_site_commitH` at a canonical circuit's accepted hidden-output statement.

`row_bc_C`: `row_copies_tgt`'s copies into the chain hold unchanged (`inBit_copyAll`); a message bit `HmRow.msgBit`
calls `.const v` (`msg_filterConstC`, `pad_filterAll`) holds `v`, which is `Chain.rowTail`'s bit there
(`constBit_rowC`: the fill bits of the last data byte are `packBits`' zeros, the rest `padTail(128 + B)`), and its
`.row i` bits are the message columns themselves (`msgCol_eq`, `msgCol_pad`), so `Chain.row_commitC` hashes
`P(ℓ) ‖ packBits (rowBits …)` (`port_commit_rowsC`). `bc_site_commitC`: the hidden statement's satisfying witness
satisfies its flat twin (`OutCopy`'s `sat_flatC`, at `outRowsC_of`'s output rows), then `row_bc_C`.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3 Model
open Flock.HmNets (wordBit)
open FlockSoundness.Discharge.Hm (blockBit bytesOf bcCol saltCol)
open FlockSoundness.Discharge.HiddenExec (AcceptedH siteAddrH)

/-- **The bits at row `p`'s message columns** at VU `g`, under the bits `w`: bit `i < ℓ` is `w` at byte `i / 8`'s
bit `i % 8` (`Typed.msgCol`). -/
def rowBits (w : ℕ → Bool) (c : Circuit) (g p : ℕ) : List Bool :=
  List.ofFn fun i : Fin c.ports[p]!.bits => w (Typed.msgCol c g p (i.val / 8) (i.val % 8))

theorem rowBits_length (w : ℕ → Bool) (c : Circuit) (g p : ℕ) : (rowBits w c g p).length = c.ports[p]!.bits :=
  List.length_ofFn

theorem rowBits_getD (w : ℕ → Bool) (c : Circuit) (g p : ℕ) {x : ℕ} (hx : x < c.ports[p]!.bits) :
    (rowBits w c g p).getD x false = w (Typed.msgCol c g p (x / 8) (x % 8)) := by
  rw [List.getD_eq_getElem?_getD, rowBits, List.getElem?_ofFn]
  simp only [hx, ↓reduceDIte, Option.getD_some]

/-! ## A bit row's bytes and its message after the prefix -/

open Hm in
theorem testBit_bitsVal : ∀ (bs : List Bool) (i : ℕ), (HmRow.bitsVal bs).testBit i = bs.getD i false
  | [], i => by simp [HmRow.bitsVal]
  | b :: bs, 0 => by cases b <;> simp [HmRow.bitsVal, Nat.testBit_zero]
  | b :: bs, i + 1 => by
      rw [Nat.testBit_succ, HmRow.bitsVal,
        show ((if b = true then 1 else 0) + 2 * HmRow.bitsVal bs) / 2 = HmRow.bitsVal bs by split <;> omega,
        testBit_bitsVal bs i]
      rfl

open Hm in
/-- `packBits` holds bit `i` at byte `i / 8`'s bit `i % 8`, and zeros past the bits. -/
theorem bitL_packBits (bs : List Bool) (i : ℕ) : bitL (HmRow.packBits bs) i = bs.getD i false := by
  rw [bitL]
  by_cases hi : i / 8 < (bs.length + 7) / 8
  · rw [HmRow.packBits, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi, Option.map_some,
      Option.getD_some]
    have hlt : HmRow.bitsVal ((bs.drop (8 * (i / 8))).take 8) < 256 := by
      have h2 : 2 ^ ((bs.drop (8 * (i / 8))).take 8).length ≤ 2 ^ 8 :=
        Nat.pow_le_pow_right (by norm_num) (by simp only [List.length_take]; omega)
      exact lt_of_lt_of_le (RowV2.bitsVal_lt _) (le_trans h2 (by norm_num))
    rw [UInt8.toNat_ofNat_of_lt' hlt, testBit_bitsVal, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
      List.getElem?_take, ite_eq_left (Nat.mod_lt i (by norm_num : 8 > 0)), List.getElem?_drop,
      Nat.div_add_mod]
  · rw [List.getD_eq_default _ _ (by rw [RowV2.packBits_length]; omega), List.getD_eq_default _ _ (by omega)]
    simp

open Hm in
theorem blockBit_wordBit {o t : ℕ} (ho : o < 128) (ht : t < 8) : HmRow.blockBit (wordBit (8 * o + t)) = (o, t) := by
  unfold HmRow.blockBit wordBit
  refine Prod.ext ?_ ?_ <;> dsimp only <;> omega

open Hm in
/-- **An unsegmented bit row's constant message bits are `Chain.rowTail`'s**: past the row's `ℓ` bits, block `k`'s bit
`wordBit (8 o + t)` is `rowTail (packBits bs)`'s bit `8 (128 k + o) + t` (a fill bit of the last byte, zero, or
`padTail(128 + B)`'s). -/
theorem constBit_rowC {p : Port} (hv : p.v2 = true) (hs : p.segs = none) {bs : List Bool} (hl : bs.length = p.bits)
    (k : ℕ) {o t : ℕ} (ho : o < 128) (ht : t < 8) (h : p.bits ≤ 8 * (128 * k + o) + t) :
    HmRow.constBit p (HmRow.tailOf p) k (wordBit (8 * o + t)) =
      some (bitL (rowTail (HmRow.packBits bs)) (8 * (128 * k + o) + t)) := by
  have hby : p.bytes = (p.bits + 7) / 8 := by unfold Port.bytes; rw [hv]; rfl
  have hlen : (HmRow.packBits bs).length = p.bytes := by rw [RowV2.packBits_length, hl, hby]
  simp only [HmRow.constBit, HmRow.tailOf, blockBit_wordBit ho ht, hs, show ¬(8 * (128 * k + o) + t < p.bits) by omega,
    ↓reduceIte]
  congr 1
  rw [rowTail, hlen, bitL, show (8 * (128 * k + o) + t) / 8 = 128 * k + o by omega,
    show (8 * (128 * k + o) + t) % 8 = t by omega]
  by_cases hj : p.bytes ≤ 128 * k + o
  · rw [List.getD_append_right _ _ _ _ (by rw [hlen]; exact hj), hlen, decide_eq_true hj, Bool.true_and, get!_eq_getD]
    generalize ((HmRow.padTail (128 + p.bytes)).data.toList.getD (128 * k + o - p.bytes) 0).toNat = x
    rw [Nat.testBit, Nat.one_and_eq_mod_two]
    rcases Nat.mod_two_eq_zero_or_one (x >>> t) with h | h <;> simp [h]
  · rw [List.getD_append _ _ _ _ (by rw [hlen]; omega), decide_eq_false hj, Bool.false_and]
    have e := bitL_packBits bs (8 * (128 * k + o) + t)
    rw [bitL, show (8 * (128 * k + o) + t) / 8 = 128 * k + o by omega,
      show (8 * (128 * k + o) + t) % 8 = t by omega,
      List.getD_eq_default bs false (n := 8 * (128 * k + o) + t) (by omega)] at e
    exact e.symm

open Hm in
/-- `rowTail (packBits bs)`'s bits below `bs`'s length are `bs`'s. -/
theorem bitL_rowTail_lt {bs : List Bool} {x : ℕ} (hx : x < bs.length) :
    bitL (rowTail (HmRow.packBits bs)) x = bs.getD x false := by
  rw [rowTail, bitL, List.getD_append _ _ _ _ (by rw [RowV2.packBits_length]; omega), ← bitL, bitL_packBits]

open Hm in
/-- The row's byte `128 · chunks + o` (past its compressions), bit `t`, is its `hm96` slot's `pad` bit
`wordBit (8 o + t)`. -/
theorem msgCol_pad (c : Circuit) (g p : ℕ) {o t : ℕ} (hs : c.ports[p]!.segs = none) (ho : o < 128) (ht : t < 8) :
    Typed.msgCol c g p (128 * ((HmRow.rowSlots c.ports).2[p]!).size + o) t = padCol c g p + wordBit (8 * o + t) := by
  have hk' : (128 * ((HmRow.rowSlots c.ports).2[p]!).size + o) / 128 = ((HmRow.rowSlots c.ports).2[p]!).size := by
    omega
  have ho' : (128 * ((HmRow.rowSlots c.ports).2[p]!).size + o) % 128 = o := by omega
  simp only [Typed.msgCol, Typed.byteAt_none hs, hk', ho', lt_irrefl, ↓reduceIte, padCol, wordBit]
  omega

/-- **An unsegmented `sha512/row/v2` row's compressions are the ones its length needs**: `HmRow.rowPorts`' count. -/
theorem v2_of_parse {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parse tags cf tables = .ok c) {p : ℕ} (hp : p < c.ports.size) (hv : c.ports[p]!.v2 = true)
    (hs : c.ports[p]!.segs = none) :
    c.ports[p]!.words = (c.ports[p]!.bits + 15) / 16 ∧
      c.ports[p]!.chunks + 1 = ((c.ports[p]!.bits + 7) / 8 + 17 + 127) / 128 := by
  obtain ⟨⟨mj, typed, hr⟩, -⟩ := Hidden.parse_rowPorts h
  obtain ⟨hw, hc, -⟩ := Hidden.rowPorts_v2 hr _ (getElem_mem_toList' hp) hv
  have := hc hs
  exact ⟨hw, by omega⟩

/-! ## Δ's copies at a bit row's message -/

section Msg

open Hm Hidden

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

/-- **Δ's copy into compression `k`'s message bit `t₀`**, for any rows: the constant `HmRow.constBit` gives there, none
at a row bit. -/
theorem msg_filterConstC {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ht₀ : t₀ < 1024) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = msgColOf c g p k + t₀) =
      bitCopy c pin (msgColOf c g p k + t₀) (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t₀) := by
  have hx := msgColOf_pos R hu (g := g) hp hk t₀
  rw [filter_rows R hu hg hp (Or.inl ⟨k, 512 + t₀, hk, by omega, hx⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = msgColOf c g p k + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases wireFAll_mem R hu hp he with ⟨k', t', -, hk', ht', rfl⟩ | ⟨-, t', ht', rfl⟩
    · intro h
      have := (shaPos_inj R hu hg hp hk' (by omega) hg hp hk (by omega) h).2.2.2
      omega
    · exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
  have hcv : (cvCopies c pin g p).filter (fun e => e.1 = msgColOf c g p k + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases cvCopies_fst R hu hp he with ⟨h0, t', ht', he1⟩ | ⟨h, -⟩
    · rw [he1]
      intro h
      have := (shaPos_inj R hu hg hp h0 (by omega) hg hp hk (by omega) h).2.2.2
      omega
    · omega
  have hpad : (padCopies c pin g p).filter (fun e => e.1 = msgColOf c g p k + t₀) = [] :=
    filter_nil_of _ _ fun e he => by
      rw [hx]
      obtain ⟨t', ht', he1⟩ := padCopies_mem R hu he
      rw [he1]
      exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
  have hmsg : (msgCopies c pin g p).filter (fun e => e.1 = msgColOf c g p k + t₀) =
      bitCopy c pin (msgColOf c g p k + t₀) (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t₀) := by
    by_cases hkb : c.ports[p]!.bits / 1024 ≤ k
    · unfold msgCopies
      rw [filter_flatMap_single _ List.nodup_range' _ _ k (by rw [List.mem_range'_1, rowSlots_size hp]; omega)
        fun k' hk' hne => filter_nil_of _ _ fun e he => ?_]
      · rw [filter_flatMap_single _ List.nodup_range _ _ t₀ (List.mem_range.2 ht₀) fun t _ hne =>
          filter_nil_of _ _ fun e he => by rw [bitCopy_fst he]; omega]
        exact filter_bitCopy _ _ _ _
      · rw [List.mem_range'_1, rowSlots_size hp] at hk'
        simp only [List.mem_flatMap, List.mem_range] at he
        obtain ⟨t, ht, he⟩ := he
        rw [bitCopy_fst he, msgColOf_pos R hu hp (show k' < c.ports[p]!.chunks by omega), hx]
        intro h
        exact hne (shaPos_inj R hu hg hp (show k' < c.ports[p]!.chunks by omega) (by omega) hg hp hk (by omega) h).2.2.1
    · have hnone : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t₀ = none := by
        have hb1 : (HmRow.blockBit t₀).1 = 8 * (t₀ / 64) + 7 - t₀ % 64 / 8 := rfl
        have hb2 : (HmRow.blockBit t₀).2 = t₀ % 8 := rfl
        simp only [HmRow.constBit, hb1, hb2]
        exact ite_eq_left (by omega)
      rw [hnone]
      refine filter_nil_of _ _ fun e he => ?_
      obtain ⟨k', t', hk1, hk', ht', he1⟩ := msgCopies_mem R hu hp he
      rw [he1, hx]
      intro h
      have := (shaPos_inj R hu hg hp hk' (by omega) hg hp hk (by omega) h).2.2.1
      omega
  rw [hw, List.append_nil]
  unfold rowCopiesOf
  rw [List.filter_append, List.filter_append, hcv, hpad, hmsg, List.nil_append, List.nil_append]

end Msg

/-! ## The port -/

section Port

open Flock.HmNets hiding and
open Hm Hidden

/-- **`Hidden.port_commit_rowsP` at any length**: a row of `B` data bytes in `n + 1 = ⌈(B + 17) / 128⌉` blocks after
the prefix `pre`, whose `n` `sha512x3` slots and `hm96` slot hold their nets' rows, whose chain holds Δ's copies (the
first `cv` the prefix's midstate, or at `n = 0` the `hm96` slot's), and whose message bits, the `hm96` slot's `pad`
the last block's, are `Chain.rowTail data`'s: the bytes at `bcCol` are the commit string of `pre ‖ data` under the salt
at `saltCol`. -/
theorem port_commit_rowsC (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128)
    {data : List UInt8} {n : ℕ} (hn : n + 1 = (data.length + 17 + 127) / 128) (hck : HmNets.check c = .ok ())
    (hsub : ∀ k < n, shaSub c p k < 3)
    (hsha : ∀ k < n, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcv0 : 0 < n → ∀ t < 512, w (shaBase c g p 0 + shaIn c (2 * shaSub c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hchain : ∀ k, k + 1 < n → ∀ t < 512, w (shaBase c g p (k + 1) + shaIn c (2 * shaSub c p (k + 1)) + t) =
      w (shaBase c g p k + shaOut c (shaSub c p k) + t))
    (hcvh0 : n = 0 → ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hcvh : 0 < n → ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      w (shaBase c g p (n - 1) + shaOut c (shaSub c p (n - 1)) + t))
    (hmsg : ∀ k < n, ∀ o < 128, ∀ t < 8, w (shaBase c g p k + shaIn c (2 * shaSub c p k + 1) + wordBit (8 * o + t)) =
      bitL (rowTail data) (8 * (128 * k + o) + t))
    (hpad : ∀ o < 128, ∀ t < 8, w (c.slotCol c.hmNet g p + hmIn c 1 + wordBit (8 * o + t)) =
      bitL (rowTail data) (8 * (128 * n + o) + t)) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString (pre.data.toList ++ data) (bytesOf w 192 (saltCol c g p)) := by
  have hsin := shaIn_eq hck
  have hhin := hmIn_eq hck
  have hshaL : ∀ k < n, LaidRows (layout (compression3.run St.empty).2 SHA_IN SHA_OUT
      (compression3.run St.empty).1.2.flatten.toArray) (inRowsOf SHA_IN) fun j => w (shaBase c g p k + j) := by
    intro k hk
    have := hsha k hk
    rw [shaLaid_eq] at this
    rw [inRows_SHA_IN]
    exact this
  have hgs : Good (SHA_IN.foldl (· + ·) 0) (compression3.run St.empty).2 := by rw [sum_SHA_IN]; exact good_compression3
  let zs : ℕ → Var → Bool := fun k x => w (shaBase c g p k + col (SHA_IN.foldl (· + ·) 0) (inRowsOf SHA_IN) x)
  have hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2 := fun k hk => laid_rows _ _ _ _ _ hgs (hshaL k hk)
  have hzs : ∀ k < n, ∀ v : ℕ, v < 4608 → zs k v = w (shaBase c g p k + v) := by
    intro k _ v hv
    show w (shaBase c g p k + col _ _ v) = _
    rw [col_input (by rw [sum_SHA_IN]; exact hv)]
  have hout : ∀ k < n, ∀ t < 512, w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t := by
    intro k hk t ht
    have hs := hsub k hk
    obtain ⟨-, hl3, -⟩ := compression3_spec (zs k) (hrows k hk)
    have hlens := compression3_lens (zs k) (hrows k hk)
    have hflat : (compression3.run St.empty).1.2.flatten.length = 1536 := by
      rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
      rfl
    have hj : 512 * shaSub c p k + t < (compression3.run St.empty).1.2.flatten.toArray.size := by
      rw [List.size_toArray, hflat]; omega
    have := laid_outs (compression3.run St.empty).2 SHA_IN SHA_OUT _ (fun j => w (shaBase c g p k + j))
      (by rw [List.size_toArray, hflat]; decide) (hshaL k hk) _ hj
    rw [shaOut_eq hck hs, Nat.add_assoc, Nat.add_assoc, this, List.getElem_toArray,
      ← bit_of_lt (by rw [hflat]; omega), bit_flatten_of_len (zs k) hlens ht]
  have hzh := hm_zh_in c w g p
  beta_reduce at hzh
  have hrow := row_commitC hpre hn zs (shaSub c p) hsub hrows
    (fun h0 q hq => by
      rw [hzs 0 h0 _ (by have := hsub 0 h0; omega), ← hcv0 h0 q hq, (hsin _ (hsub 0 h0)).1]
      exact congrArg w (by omega))
    (fun k hk q hq => by
      rw [hzs (k + 1) hk _ (by have := hsub (k + 1) hk; omega), ← hout k (by omega) q hq, ← hchain k hk q hq,
        (hsin _ (hsub (k + 1) hk)).1]
      exact congrArg w (by omega))
    (fun k hk o ho t ht => by
      have hwb : wordBit (8 * o + t) < 1024 := by unfold wordBit; omega
      rw [hzs k hk _ (by have := hsub k hk; omega), ← hmsg k hk o ho t ht, (hsin _ (hsub k hk)).2]
      exact congrArg w (by omega))
    _ (hm_zh c w g p hhm)
    (fun h0 q hq => by rw [hzh q (by omega), ← hcvh0 h0 q hq, hhin.1, Nat.add_zero])
    (fun h0 q hq => by
      rw [hzh q (by omega), ← hout (n - 1) (by omega) q hq, ← hcvh h0 q hq, hhin.1, Nat.add_zero])
    (fun o ho t ht => by
      have hwb : wordBit (8 * o + t) < 1024 := by unfold wordBit; omega
      rw [hzh _ (by omega), ← hpad o ho t ht, hhin.2.1]
      exact congrArg w (by omega))
    (length_bytesOf w 192 (saltCol c g p))
    (fun q hq => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, ← hrow t ht, ← hm_bc c w g p hck hhm t ht, bcCol]
  exact congrArg w (by omega)

end Port

/-! ## The flat statement -/

theorem blockBit_ite {S : Model.Statement} {z : Model.Witness S.m} {o : Fin (2 ^ S.nbl)} {c : Circuit} {pin : ℕ}
    (hpin : blockBit S z o pin = true) (hzero : blockBit S z o (Hm.zeroCol c) = false) (v : Bool) :
    blockBit S z o (if v then pin else Hm.zeroCol c) = v := by
  cases v <;> simp [hpin, hzero]

section Bc

open Hm Hidden

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`Hidden.row_bc_tgt` at any bit row**: in every block of a satisfying witness of the untyped (flat) statement, at
an unsegmented `sha512/row/v2` row `p` of any `ℓ` bits in the compressions its length needs (`hn`, `v2_of_parse`), the
`b ‖ c` at its `hm96` output is the commit string of `P(ℓ)` and `packBits` of the bits at its message columns, under
the salt its `hm96` slot reads.

*Route*: the module docstring's. -/
theorem row_bc_C (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hv : st.c.ports[p]!.v2 = true)
    (hn : st.c.ports[p]!.chunks + 1 = ((st.c.ports[p]!.bits + 7) / 8 + 17 + 127) / 128) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefixV2 st.c.ports[p]!.bits).data.toList ++
        HmRow.packBits (rowBits (blockBit (L.model regs) z o) st.c g p))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  have R := RowRanges.ofL L F.parse F.check
  have hrw : RowWires st.c := F.parse.rowWires
  have hs := hu _ (getElem_mem_toList' hp)
  have hpin := pin_bit L hz o
  have hzero := zero_bit L F hupv (by omega) hz o
  have hhm := hmSlot_flat L F hz o hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))
  have hmid : HmRow.midstateOf st.c.ports[p]! = Sha512.compress Sha512.iv (HmRow.rowPrefixV2 st.c.ports[p]!.bits) 0 := by
    simp only [HmRow.midstateOf, HmRow.prefixOf, hs, hv, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
  have hbl := rowBits_length (blockBit (L.model regs) z o) st.c g p
  have hn' : st.c.ports[p]!.chunks + 1 =
      ((HmRow.packBits (rowBits (blockBit (L.model regs) z o) st.c g p)).length + 17 + 127) / 128 := by
    rw [RowV2.packBits_length, hbl]; exact hn
  have hsz := rowSlots_size hp
  have hrowbit : ∀ x, x < st.c.ports[p]!.bits →
      bitL (rowTail (HmRow.packBits (rowBits (blockBit (L.model regs) z o) st.c g p))) x =
        blockBit (L.model regs) z o (Typed.msgCol st.c g p (x / 8) (x % 8)) := fun x hx => by
    rw [bitL_rowTail_lt (by rw [hbl]; exact hx), rowBits_getD _ _ _ _ hx]
  refine port_commit_rowsC st.c _ g p (size_rowPrefixV2 _) hn' R.rowNets (fun k hk => (rowSlot_lt hp hk).2)
    (fun k hk => shaSlot_flat L F hz o hg (by rw [R.perVu_sha]; exact (rowSlot_lt hp hk).1)) hhm
    (fun h0 t ht => ?_) (fun k hk t ht => ?_) (fun h0 t ht => ?_) (fun h0 t ht => ?_) (fun k hk o' ho' t ht => ?_)
    (fun o' ho' t ht => ?_)
  · have hs' : shaSub st.c p 0 < 3 := (rowSlot_lt hp h0).2
    rw [shaIn_pos R hu hp h0]
    have hf := cv0_filterAll R hu (pin := st.pin) hg hp h0 ht
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inl rfl) (shaPos_at R hu hg hp h0 (off := t) (by omega))
      (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf,
      blockBit_constSrc _ _ hpin hzero, hmid]
  · have hs' : shaSub st.c p (k + 1) < 3 := (rowSlot_lt hp hk).2
    have hf := chain_filterAll R hu (pin := st.pin) (k := k + 1) hg hp (by omega) hk ht
    rw [chainDst_eq R.rowNets hs'] at hf
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inl rfl) (R.shaIn_at hg hp hk ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    rfl
  · rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
    have hf := hmcv0_filterAll R hu (pin := st.pin) hg hp h0 ht
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega)) (hmNet_ofRows R.rowNets)
      (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf, blockBit_constSrc _ _ hpin hzero, hmid]
  · rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
    have hf := hmcv_filterAll R hu (pin := st.pin) hg hp h0 ht
    have hx : Flat.wireDst st.c g (hmWire st.c.ports st.c.shaNet st.c.hmNet p) = st.c.slotCol st.c.hmNet g p :=
      hmWireDst_eq R.rowNets g p
    rw [hx] at hf
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega)) (hmNet_ofRows R.rowNets)
      (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    unfold Flat.wireSrc hmWire shaBase shaOut shaSub
    simp only [rowSlots_size hp]
  · have hwb : wordBit (8 * o' + t) < 1024 := by unfold wordBit; omega
    have hs' : shaSub st.c p k < 3 := (rowSlot_lt hp hk).2
    by_cases hbit : 8 * (128 * k + o') + t < st.c.ports[p]!.bits
    · rw [hrowbit _ hbit, show (8 * (128 * k + o') + t) / 8 = 128 * k + o' by omega,
        show (8 * (128 * k + o') + t) % 8 = t by omega, msgCol_eq st.c g p k hs (by rw [hsz]; exact hk) ho' ht]
    · have hf := msg_filterConstC R hu (pin := st.pin) hg hp hk hwb
      rw [constBit_rowC hv hs hbl k ho' ht (by omega)] at hf
      simp only [bitCopy] at hf
      have hat := shaPos_at R hu hg hp hk (off := 512 + wordBit (8 * o' + t)) (by omega)
      rw [← msgColOf_pos R hu hp hk] at hat
      rw [show shaBase st.c g p k + shaIn st.c (2 * shaSub st.c p k + 1) + wordBit (8 * o' + t) =
          msgColOf st.c g p k + wordBit (8 * o' + t) from rfl,
        inBit_copyAll L F hda hrw hu hz o (Or.inl rfl) hat (shaNet_ofRows R.rowNets)
          (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
      exact blockBit_ite hpin hzero _
  · have hwb : wordBit (8 * o' + t) < 1024 := by unfold wordBit; omega
    have hx : st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + wordBit (8 * o' + t) = padCol st.c g p + wordBit (8 * o' + t) := by
      rw [(hmIn_eq R.rowNets).2.1, padCol_eq R.rowNets]
    rw [hx]
    by_cases hbit : 8 * (128 * st.c.ports[p]!.chunks + o') + t < st.c.ports[p]!.bits
    · rw [hrowbit _ hbit, show (8 * (128 * st.c.ports[p]!.chunks + o') + t) / 8 = 128 * st.c.ports[p]!.chunks + o' by
          omega, show (8 * (128 * st.c.ports[p]!.chunks + o') + t) % 8 = t by omega, ← hsz,
        msgCol_pad st.c g p hs ho' ht]
    · have hf := pad_filterAll R hu (pin := st.pin) hg hp hwb
      rw [hsz, constBit_rowC hv hs hbl _ ho' ht (by omega)] at hf
      simp only [bitCopy] at hf
      have hx' : padCol st.c g p + wordBit (8 * o' + t) = st.c.slotCol st.c.hmNet g p + (512 + wordBit (8 * o' + t)) := by
        rw [padCol_eq R.rowNets, Nat.add_assoc]
      rw [hx'] at hf ⊢
      rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := 512 + wordBit (8 * o' + t)) (by omega))
        (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
      exact blockBit_ite hpin hzero _

end Bc

variable {T : Model.Statement}

/-- **`row_bc_C` in an accepted hidden-output statement on a canonical circuit**, through its flat twin
(`sat_flatC`). -/
theorem AcceptedH.row_bcC (a : AcceptedH T) (hfmt : formatOk a.st.c = true) {z : Witness T.m} (hz : T.Satisfies z)
    (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g) (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      ((HmRow.rowPrefixV2 a.st.c.ports[p]!.bits).data.toList ++ HmRow.packBits (rowBits (blockBit T z o) a.st.c g p))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  have hpa := a.parse
  have hcr := a.checkRows
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hfmt hg hp hpa hcr ⊢
  have s := Hidden.setupHidden_spec' h
  have L0 := HiddenExec.flat_layout ht h
  have F0 := HiddenExec.flat_unitFacts ht h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  obtain ⟨hrow, -, -, -, -⟩ := formatOk_spec L.unit_built hfmt
  have hz0 := sat_flatC L L0 F0 hda (outRowsC_of L.unit_built hpa hcr hfmt).1 hz
  have hupv : 0 < st.c.upv := Hidden.setupHidden_upv_pos ht h
  have hpr := hrow _ (getElem_mem_toList' hp)
  exact row_bc_C L0 F0 rfl hupv (fun q hq => (hrow q hq).2) hz0 o hg hp hpr.1 (v2_of_parse hpa hp hpr.1 hpr.2).2

/-- **A canonical row's `b ‖ c` at a drawn unit's site** (`bc_site_commitH` for canonical circuits): in a satisfying
witness of an accepted hidden-output statement on a canonical circuit, at block `o`, VU `g` and row `p`, input or
output, the bytes at the address the regions pin (`Layout.bcAddr`) are the commit string of `P(ℓ_p)` and `packBits` of
the `ℓ_p` bits at the row's message columns, under the salt at `saltCol`. No condition on the out net.

*Route*: the module docstring's. -/
theorem bc_site_commitC (a : AcceptedH T) (hfmt : formatOk a.st.c = true) {z : Witness T.m} (hz : T.Satisfies z)
    (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g) (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefixV2 a.st.c.ports[p]!.bits).data.toList ++
        HmRow.packBits (rowBits (fun x => FlockSoundness.bit (z (siteAddrH a o x))) a.st.c g p))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have R := a.rowRanges
  have hbc : (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => siteAddrH a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (Hm.perVu_hm_pow2 R)]
    rfl
  have hrb : rowBits (fun x => FlockSoundness.bit (z (siteAddrH a o x))) a.st.c g p =
      rowBits (blockBit T z o) a.st.c g p := by
    unfold rowBits
    congr 1
    funext i
    exact HiddenExec.bit_siteAddrH a z o
      (Layout.msgCol_lt_of_parse a.parse hg hp _ _ (Nat.mod_lt _ (by norm_num)))
  rw [hbc, HiddenExec.bytesAt_siteH a z o 128 (bcCol a.st.c g p) (fun j hj t ht => Hm.bcCol_lt R hg hp hj ht),
    HiddenExec.bytesAt_siteH a z o 192 (saltCol a.st.c g p) (fun j hj t ht => Hm.saltCol_lt R hg hp hj ht), hrb]
  exact AcceptedH.row_bcC a hfmt hz o hg hp

end FlockSoundness.Discharge.Canonical
