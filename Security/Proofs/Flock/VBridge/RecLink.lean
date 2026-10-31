import Proofs.Flock.VBridge.OfVStar
import Proofs.Flock.VBridge.AccUnits
import Proofs.Flock.VBridge.RecLinkGadgets
import Proofs.Flock.Soundness.Discharge.Hm.Salt
import Proofs.Flock.Soundness.Discharge.Hm.Commit
import VStar.Link
import Flock.RecLinkRef

/-!
# V*'s `RecLink{UB, PB}` unit computes its reference

`VStar.linkUnit UB PB` is the netlist V* writes for one row of a recursive session's link table. This file proves that,
wherever its rows hold, its outputs are the bytes `Flock.RecLinkRef` says: each opened row's hm96 tree leaf
(`RecLinkTree.leaf` of `Hm96.Default512.commitString`) climbed by its siblings (`Flock.climb` under `RecLinkRef.scheme`) to
the position the table row names (`RecLinkRef.posOf`), then 64 zero bytes, when the row passes `RecLinkRef.okRef`, and
128 zero bytes when it doesn't (`rootB`, `linkB`).

The gadgets and the check are in `Proofs.Flock.VBridge.RecLinkGadgets`.

* The gadgets. `rec_link._all` (`sound_allGo`, `sound_allForms`): 1 exactly when every form is 1; `_none`
  (`sound_noneForms`): 1 exactly when every form is 0 (`AllZero`); `orForm`.
* The check. `sound_okForms`: on forms that carry a 128-byte table row and two 64-byte digests (`Carries`), `okForms` is
  `okRef`: a field's 32 forms are its little-endian u32's bits (`bit_linkField`), its high bits zero exactly when it fits
  (`allZero_dropField`), the row's bytes past the fields zero exactly when `tailZero` (`tailZero_iff`), and the digests'
  XOR zero exactly when they are equal (`allZero_xor`).
* The leaf. `sound_hm96Leaf`: the commit string `b ‖ c` (`b = x ⊕ M·y` word by word, `c` the salt's SHA-512 from its
  prefix's midstate) and its SHA-512 from the leaf prefix's midstate are `RecLinkTree.leaf (commitString x y)`.
* The steps. `sound_side`: each side's leaf, climbed by its siblings and ANDed with `ok` (`carries_gate`), carries
  `SideP`; `sound_linkSteps`: the climbs' direction bits are the bits of `dirIdx`, which is `posOf` wherever the row
  passes `okRef` (`dirIdx_eq`).
* The unit. `linkUnit_ok` (its builder is `linkRun`, the steps on the seven ports' input bits), `laid_linkUnit` (the
  laid-out unit's output columns hold `linkB` wherever its rows and input columns hold) and `unit_linkUnit` (a copy of it
  in a unit of any partition).
-/

namespace FlockVBridge.RecLink

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## The hm96 leaf -/

theorem wordBit_wordBit {t : ℕ} (ht : t < 512) : wordBit (wordBit t) = t := by
  unfold wordBit
  omega

theorem size_leafPrefix : Hm96.Default512.leafPrefix.size = 128 := by
  rw [Hm96.Default512.leafPrefix, Hm96.leafPrefix, Hm96.pad, ByteArray.size_append, ByteArray.size_append,
    show Hm96.sha512.hash = Sha512.hash from rfl, Sha512.hash_size]
  have ht : Hm96.sha512.leafTag.size = 27 := by decide +kernel
  rw [ht]
  rfl

theorem preCv_one (p : List UInt8) :
    preCv p 1 = (List.range 8).map fun i => (Sha512.compress Sha512.iv ⟨p.toArray⟩ 0)[i]!.toBitVec := by
  rw [preCv, chainV, chain_succ] <;> rfl

theorem saltMid_eq : saltMid = preCv Hm96.Default512.saltPrefix.data.toList 1 := by
  rw [preCv_one]
  rfl

theorem midstate_eq {p : ByteArray} (hp : p.size = 128) : VStar.midstate p = preCv p.data.toList 1 := by
  rw [preCv_one, map_range_eq_toBV _ (size_compress _ _ _)]
  have hb : (⟨p.data.toList.toArray⟩ : ByteArray) = p := ByteArray.ext (by simp)
  rw [hb]
  unfold VStar.midstate toBV
  rw [hp, show (128 : ℕ) / 128 = 1 from rfl, List.range_one, List.foldl_cons, List.foldl_nil, Nat.mul_zero] <;> rfl

theorem size_leaf (b : ByteArray) : (RecLinkTree.leaf b).size = 64 := Sha512.hash_size _

theorem catBytes_eq (a b : ByteArray) : catBytes a.data.toList b.data.toList = a ++ b :=
  ByteArray.ext (Array.toList_inj.1 (by rw [data_catBytes, ByteArray.toList_data_append]))

/-- `commitString`'s first 64 bytes, `x ⊕ M·y`. -/
def xorPart (Dg Sl : ByteArray) : ByteArray :=
  ⟨(Array.range 64).map fun i => Dg.get! i ^^^ (Hm96.Default512.mask Sl).get! i⟩

theorem commitString_eq (Dg Sl : ByteArray) :
    Hm96.Default512.commitString Dg Sl = xorPart Dg Sl ++ Sha512.hash (Hm96.Default512.saltPrefix ++ Sl) := rfl

theorem size_xorPart (Dg Sl : ByteArray) : (xorPart Dg Sl).size = 64 := by
  show ((Array.range 64).map fun i => Dg.get! i ^^^ (Hm96.Default512.mask Sl).get! i).size = 64
  rw [Array.size_map, Array.size_range]

theorem size_commitString (Dg Sl : ByteArray) : (Hm96.Default512.commitString Dg Sl).size = 128 := by
  rw [commitString_eq, ByteArray.size_append, size_xorPart, Sha512.hash_size]

theorem bitL_xorPart (Dg Sl : ByteArray) {q : ℕ} (hq : q < 512) :
    bitL (xorPart Dg Sl).data.toList q = (bitL Dg.data.toList q ^^ maskV Hm96.Default512.key (bitL Sl.data.toList) q) := by
  have hm : bitL (Hm96.Default512.mask Sl).data.toList q = maskV Hm96.Default512.key (bitL Sl.data.toList) q :=
    bitL_mask Sl.data.toList hq
  have hb : (xorPart Dg Sl).data.toList.getD (q / 8) 0 = Dg.get! (q / 8) ^^^ (Hm96.Default512.mask Sl).get! (q / 8) := by
    show (((Array.range 64).map fun i => Dg.get! i ^^^ (Hm96.Default512.mask Sl).get! i).toList).getD (q / 8) 0 = _
    rw [Array.toList_map, Array.toList_range, List.getD_eq_getElem _ _ (by simp; omega), List.getElem_map,
      List.getElem_range]
  rw [bitL, hb, UInt8.toNat_xor, Nat.testBit_xor, ← hm, bitL, bitL, get!_eq_getD, get!_eq_getD]

theorem carries_xorPart {z : Var → Bool} {bw x y : List Lin} {Dg Sl : ByteArray} (hx : Carries z x Dg)
    (hy : Carries z y Sl)
    (hbw : bw.length = (List.range 512).length ∧ ∀ i (h : i < (List.range 512).length), bit z bw i =
      (ev z (x.getD (wordBit (List.range 512)[i]) Lin.zero) ^^
        maskV Hm96.Default512.key (bit z y) (wordBit (List.range 512)[i]))) :
    Carries z ((List.range 512).map fun t => bw.toArray.getD (wordBit t) Lin.zero) (xorPart Dg Sl) := by
  refine ⟨by rw [List.length_map, List.length_range, size_xorPart], fun q hq => ?_⟩
  have hq' : q < 512 := by simpa using hq
  rw [bit_of_lt hq, List.getElem_map, List.getElem_range, OfVStar.getD_toArray, ev_getD,
    hbw.2 (wordBit q) (by simpa using wordBit_lt hq'), List.getElem_range, wordBit_wordBit hq', ev_getD,
    bit_eq_bitL hx, bit_eq_bitL hy, bitL_xorPart Dg Sl hq']

/-- **V*'s hm96 leaf gadget is the tree's leaf**: whenever its rows hold, on forms that carry a 64-byte digest `Dg` and a
192-byte salt `Sl`, its output carries `RecLinkTree.leaf (commitString Dg Sl)`. -/
theorem sound_hm96Leaf (x y : List Lin) : Sound (VStar.hm96Leaf x y) fun z out => ∀ Dg Sl : ByteArray,
    Dg.size = 64 → Sl.size = 192 → Carries z x Dg → Carries z y Sl →
    Carries z out (RecLinkTree.leaf (Hm96.Default512.commitString Dg Sl)) := by
  unfold VStar.hm96Leaf
  simp only [OfVStar.shaFrom_eq]
  rw [show Hm96.Default512.saltPrefix.size = 128 * 1 from size_saltPrefix,
    show Hm96.Default512.leafPrefix.size = 128 * 1 from size_leafPrefix]
  refine Sound.bind' (sound_mapM _ (fun z p => ev z (x.getD (wordBit p) Lin.zero) ^^
      maskV Hm96.Default512.key (bit z y) (wordBit p))
    (fun p => Sound.mono (sound_commit _) fun z r hr => by rw [hr, eval_maskBit, OfVStar.getD_toArray]) _) fun bw => ?_
  refine Sound.bind' (sound_shaFrom saltMid y 1) fun c => ?_
  refine Sound.mono (sound_shaFrom (VStar.midstate Hm96.Default512.leafPrefix) _ 1)
    fun z out hout hc hbw Dg Sl hDg hSl hx hy => ?_
  obtain ⟨hcl, hcb⟩ := hc Hm96.Default512.saltPrefix.data.toList Sl.data.toList (by rw [length_saltPrefix])
    (by rw [Array.length_toList]; exact hy.1) hy.2 saltMid_eq
  rw [catBytes_eq] at hcb
  have hcc : Carries z c (Sha512.hash (Hm96.Default512.saltPrefix ++ Sl)) :=
    ⟨by rw [hcl, Sha512.hash_size], fun t ht => hcb t (by omega)⟩
  have hbc := carries_append (carries_xorPart hx hy hbw) hcc
  rw [← commitString_eq] at hbc
  obtain ⟨hol, hob⟩ := hout Hm96.Default512.leafPrefix.data.toList (Hm96.Default512.commitString Dg Sl).data.toList
    (by rw [Array.length_toList, ByteArray.size_data, size_leafPrefix])
    (hbc.1.trans (by rw [Array.length_toList, ByteArray.size_data])) hbc.2 (midstate_eq size_leafPrefix)
  rw [catBytes_eq] at hob
  exact ⟨by rw [hol, size_leaf], fun t ht => hob t (by omega)⟩

/-! ## The steps -/

def zeroB (n : ℕ) : ByteArray := ⟨Array.replicate n 0⟩

theorem size_zeroB (n : ℕ) : (zeroB n).size = n := by
  show (Array.replicate n (0 : UInt8)).size = n
  exact Array.size_replicate

theorem bitL_zeroB (n t : ℕ) : bitL (zeroB n).data.toList t = false := by
  have h : (zeroB n).data.toList.getD (t / 8) 0 = 0 := by
    show (Array.replicate n (0 : UInt8)).toList.getD (t / 8) 0 = 0
    rw [Array.toList_replicate, List.getD_eq_getElem?_getD, List.getElem?_replicate]
    split_ifs <;> rfl
  rw [bitL, h]
  simp

theorem carries_zeros (z : Var → Bool) (n : ℕ) : Carries z (List.replicate (8 * n) Lin.zero) (zeroB n) :=
  ⟨by rw [List.length_replicate, size_zeroB], fun t ht => by
    rw [bit_of_lt ht, List.getElem_replicate, ev_zero, bitL_zeroB]⟩

/-- A climb's root and 64 zero bytes when `ok`, else 128 zero bytes: `rec_link.expected` of the root, or `zeros`. -/
def rootB (ok : Bool) (r : ByteArray) : ByteArray := (if ok then r else zeroB 64) ++ zeroB 64

theorem size_rootB (ok : Bool) {r : ByteArray} (hr : r.size = 64) : (rootB ok r).size = 128 := by
  unfold rootB
  split <;> rw [ByteArray.size_append, size_zeroB] <;> simp [hr, size_zeroB]

theorem rootB_eq {ok : Bool} {r r' : ByteArray} (h : ok = true → r = r') :
    rootB ok r = (if ok = true then r' else zeroB 64) ++ zeroB 64 := by
  unfold rootB
  by_cases hk : ok = true
  · rw [if_pos hk, if_pos hk, h hk]
  · rw [if_neg hk, if_neg hk]

/-- The unit's outputs `(rin, rout)`: the reader's and the writer's climbs to the positions the row names, each through
`rootB` at `okRef`. -/
def linkB (UB PB : ℕ) (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray) : ByteArray × ByteArray :=
  (rootB (RecLinkRef.okRef UB PB T Xr Xw)
      (Flock.climb RecLinkRef.scheme (RecLinkTree.leaf (Hm96.Default512.commitString Xr Yr))
        (RecLinkRef.posOf T 1 2 PB (UB + PB)) sr),
    rootB (RecLinkRef.okRef UB PB T Xr Xw)
      (Flock.climb RecLinkRef.scheme (RecLinkTree.leaf (Hm96.Default512.commitString Xw Yw))
        (RecLinkRef.posOf T 3 4 PB (UB + PB)) sw))

theorem size_linkB (UB PB : ℕ) (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray) :
    (linkB UB PB T Xr Yr Xw Yw sr sw).1.size = 128 ∧ (linkB UB PB T Xr Yr Xw Yw sr sw).2.size = 128 := by
  dsimp only [linkB]
  exact ⟨size_rootB _ (AccChain.climb_size_of RecLinkRef.scheme (fun _ _ => rfl) _ _ _ (size_leaf _)),
    size_rootB _ (AccChain.climb_size_of RecLinkRef.scheme (fun _ _ => rfl) _ _ _ (size_leaf _))⟩

theorem carries_rootB {z : Var → Bool} {w : List Lin} {ok : Bool} {r r' : ByteArray} (h : ok = true → r = r')
    (hc : Carries z w ((if ok then r' else zeroB 64) ++ zeroB 64)) : Carries z w (rootB ok r) := by
  rw [rootB_eq h]
  exact hc

/-- The climb's direction bits as a number: the row field `r`'s low `PB` bits, the unit field `u`'s low `UB` above. -/
def dirIdx (T : ByteArray) (u r UB PB : ℕ) : ℕ :=
  2 ^ PB * (RecLinkRef.field T u % 2 ^ UB) + RecLinkRef.field T r % 2 ^ PB

theorem dirIdx_eq {T : ByteArray} {u r UB PB : ℕ} (hu : RecLinkRef.field T u < 2 ^ UB)
    (hr : RecLinkRef.field T r < 2 ^ PB) : dirIdx T u r UB PB = RecLinkRef.posOf T u r PB (UB + PB) := by
  rw [dirIdx, Nat.mod_eq_of_lt hu, Nat.mod_eq_of_lt hr, RecLinkRef.posOf_fits hu hr, RecLinkTree.index, Nat.mul_comm]

/-- **The climb's direction forms are `dirIdx`'s bits.** -/
theorem dirs_ok {z : Var → Bool} {row : List Lin} {T : ByteArray} (hc : Carries z row T) (hT : T.size = 128)
    {UB PB : ℕ} (hUB : UB ≤ 32) (hPB : PB ≤ 32) (u r : ℕ) (hu : u < 32) (hr : r < 32) (j : ℕ) (hj : j < UB + PB) :
    ev z (((VStar.linkField row r).take PB ++ (VStar.linkField row u).take UB).getD (0 + j) Lin.zero) =
      (dirIdx T u r UB PB).testBit j := by
  have hl : ((VStar.linkField row r).take PB).length = PB := by
    rw [List.length_take, length_linkField hc hT r hr]
    omega
  rw [Nat.zero_add, ev_getD, dirIdx, Nat.testBit_two_pow_mul_add _ (Nat.mod_lt _ (Nat.two_pow_pos PB))]
  split_ifs with h
  · rw [bit_append_left (show j < ((VStar.linkField row r).take PB).length by omega), bit_take h,
      bit_linkField hc hT r j hr (by omega), Nat.testBit_mod_two_pow, decide_eq_true h, Bool.true_and]
  · rw [bit_append_right (show ((VStar.linkField row r).take PB).length ≤ j by omega), hl,
      bit_take (show j - PB < UB by omega), bit_linkField hc hT u (j - PB) hu (by omega), Nat.testBit_mod_two_pow,
      decide_eq_true (show j - PB < UB by omega), Bool.true_and]

/-- A digest's forms ANDed with `ok` carry the digest when `ok` holds and 64 zero bytes when it doesn't. -/
theorem carries_gate {z : Var → Bool} {ok : Lin} {cr rin : List Lin} {C : ByteArray} (hC : Carries z cr C)
    (hCs : C.size = 64)
    (hr : rin.length = cr.length ∧ ∀ i (h : i < cr.length), bit z rin i = (ev z ok && ev z cr[i])) :
    Carries z rin (if ev z ok then C else zeroB 64) := by
  refine ⟨?_, fun t ht => ?_⟩
  · rw [hr.1, hC.1]
    split_ifs
    · rfl
    · rw [size_zeroB, hCs]
  · have ht' : t < cr.length := by rw [← hr.1]; exact ht
    rw [hr.2 t ht', ← bit_of_lt ht', hC.2 t ht']
    split_ifs with h
    · simp [h]
    · simp [h, bitL_zeroB]

theorem carries_linkZeros (z : Var → Bool) :
    Carries z (List.replicate (16 * (VStar.LINK_OUT_WORDS - VStar.NODE_WORDS)) Lin.zero) (zeroB 64) :=
  carries_zeros z 64

theorem length_sibForms (aux : List Lin) (D : ℕ) : (VStar.sibForms aux D).length = D := by
  simp [VStar.sibForms]

/-- What one side of the steps carries (`sound_side`): on forms that carry a 64-byte digest `Dg`, a 192-byte salt `Sl`,
siblings `ss` and the bits of `idx`, the climb of `Dg, Sl`'s leaf when `ok` holds and 64 zero bytes when it doesn't. -/
def SideP (ok : Lin) (d x y aux : List Lin) (D : ℕ) (z : Var → Bool) (g : List Lin) : Prop :=
  ∀ (Dg Sl : ByteArray) (idx : ℕ) (ss : List ByteArray), Dg.size = 64 → Sl.size = 192 → Carries z x Dg →
    Carries z y Sl → List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (VStar.sibForms aux D) ss →
    (∀ j < D, ev z (d.getD (0 + j) Lin.zero) = idx.testBit j) →
    Carries z g (if ev z ok then
      Flock.climb RecLinkRef.scheme (RecLinkTree.leaf (Hm96.Default512.commitString Dg Sl)) idx ss else zeroB 64)

/-- **One side of the steps**: the leaf of `x, y`, its climb by the siblings in `aux` along `d`, ANDed with `ok`
(`carries_gate`), carries `SideP` into the rest `k`. -/
theorem sound_side {ok : Lin} {d x y aux : List Lin} {D : ℕ} {β : Type} {k : List Lin → M β}
    {Q : (Var → Bool) → β → Prop} (hk : ∀ g, Sound (k g) fun z b => SideP ok d x y aux D z g → Q z b) :
    Sound (VStar.hm96Leaf x y >>= fun l => climbFrom d 0 l (VStar.sibForms aux D) >>= fun c =>
      c.mapM (Flock.HmNets.and ok) >>= k) Q :=
  Sound.bind' (sound_hm96Leaf x y) fun l =>
    Sound.bind' (sound_climbFrom RecLinkRef.scheme (fun _ _ => rfl) d 0 l _) fun c =>
    Sound.bind' (sound_mapM _ (fun z w => ev z ok && ev z w) (fun w => sound_and ok w) c) fun g =>
    Sound.mono (hk g) fun z _ hb hg hc hl => hb fun Dg Sl idx ss hDg hSl hx hy hss hdir =>
      carries_gate (hc (RecLinkTree.leaf (Hm96.Default512.commitString Dg Sl)) idx ss (size_leaf _)
          (hl Dg Sl hDg hSl hx hy) hss fun j hj => hdir j (by rwa [length_sibForms] at hj))
        (AccChain.climb_size_of RecLinkRef.scheme (fun _ _ => rfl) _ _ _ (size_leaf _)) hg

/-- **V*'s link steps compute the unit's reference outputs**: whenever their rows hold, on forms that carry a 128-byte
table row `T`, the reader's digest and salt `Xr, Yr` and siblings `sr`, and the writer's `Xw, Yw` and `sw`, the outputs
carry `linkB`: each opened row's leaf climbed to the position the row names, gated by `okRef`, then 64 zero bytes. -/
theorem sound_linkSteps (UB PB : ℕ) (hUB : UB ≤ 32) (hPB : PB ≤ 32) (trow xr yr raux xw yw waux : List Lin) :
    Sound (VStar.linkSteps UB PB trow xr yr raux xw yw waux) fun z o =>
      ∀ (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray),
        T.size = 128 → Xr.size = 64 → Yr.size = 192 → Xw.size = 64 → Yw.size = 192 →
        Carries z trow T → Carries z xr Xr → Carries z yr Yr → Carries z xw Xw → Carries z yw Yw →
        List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (VStar.sibForms raux (UB + PB)) sr →
        List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (VStar.sibForms waux (UB + PB)) sw →
        Carries z o.1 (linkB UB PB T Xr Yr Xw Yw sr sw).1 ∧ Carries z o.2 (linkB UB PB T Xr Yr Xw Yw sr sw).2 := by
  unfold VStar.linkSteps
  simp only [OfVStar.climbFrom_eq]
  refine Sound.bind' (sound_okForms UB PB hUB hPB trow xr xw) fun _ =>
    sound_side fun _ => sound_side fun _ =>
    Sound.pure' fun z hW hR hok T Xr Yr Xw Yw sr sw hT hXr hYr hXw hYw htrow hxr hyr hxw hyw hsr hsw => ?_
  have hok' := hok T Xr Xw hT hXr hXw htrow hxr hxw
  have hR' := carries_append (hR Xr Yr (dirIdx T 1 2 UB PB) sr hXr hYr hxr hyr hsr fun j hj =>
    dirs_ok htrow hT hUB hPB 1 2 (by norm_num) (by norm_num) j hj) (carries_linkZeros z)
  have hW' := carries_append (hW Xw Yw (dirIdx T 3 4 UB PB) sw hXw hYw hxw hyw hsw fun j hj =>
    dirs_ok htrow hT hUB hPB 3 4 (by norm_num) (by norm_num) j hj) (carries_linkZeros z)
  rw [hok'] at hR' hW'
  refine ⟨?_, ?_⟩ <;> dsimp only [linkB]
  · exact carries_rootB (fun hk => by
      rw [dirIdx_eq (RecLinkRef.okRef_fits hk).2.1 (RecLinkRef.okRef_fits hk).2.2.2.1]) hR'
  · exact carries_rootB (fun hk => by
      rw [dirIdx_eq (RecLinkRef.okRef_fits hk).2.2.1 (RecLinkRef.okRef_fits hk).2.2.2.2]) hW'

/-! ## Freshness -/

theorem fresh_hm96Leaf (x y : List Lin) : Fresh (VStar.hm96Leaf x y) := by
  unfold VStar.hm96Leaf
  simp only [OfVStar.shaFrom_eq]
  exact Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_commit _) _) fun _ =>
    Fresh.bind' (fresh_shaFrom _ _ _) fun _ => fresh_shaFrom _ _ _

theorem fresh_linkSteps (UB PB : ℕ) (trow xr yr raux xw yw waux : List Lin) :
    Fresh (VStar.linkSteps UB PB trow xr yr raux xw yw waux) := by
  unfold VStar.linkSteps
  simp only [OfVStar.climbFrom_eq]
  exact Fresh.bind' (fresh_okForms _ _ _ _ _) fun _ => Fresh.bind' (fresh_hm96Leaf _ _) fun _ =>
    Fresh.bind' (fresh_climbFrom _ _ _ _) fun _ => Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_and _ _) _) fun _ =>
    Fresh.bind' (fresh_hm96Leaf _ _) fun _ => Fresh.bind' (fresh_climbFrom _ _ _ _) fun _ =>
    Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_and _ _) _) fun _ => Fresh.pure' _

/-! ## V*'s `RecLink` unit -/

/-- An `aux` port's bits: `16 A(D)`. -/
abbrev auxBits (UB PB : ℕ) : ℕ := 16 * VStar.auxWords (UB + PB)

/-- The unit's input bits: its seven ports' words, 16 bits each. -/
def linkIn (UB PB : ℕ) : ℕ := 16 * (VStar.linkPorts UB PB).sum

theorem linkIn_eq (UB PB : ℕ) : linkIn UB PB = 5120 + 2 * auxBits UB PB := by
  simp only [linkIn, VStar.linkPorts, List.sum_cons, List.sum_nil, VStar.LINK_ROW_WORDS, VStar.LINK_X_WORDS,
    VStar.LINK_Y_WORDS, auxBits]
  omega

/-- The unit's builder after its input: the steps on the ports `trow, xr, yr, raux, xw, yw, waux` in that order. -/
def linkRun (UB PB : ℕ) : M (List Lin × List Lin) :=
  VStar.linkSteps UB PB (vars 0 1024) (vars 1024 512) (vars 1536 1536) (vars 3072 (auxBits UB PB))
    (vars (3072 + auxBits UB PB) 512) (vars (3584 + auxBits UB PB) 1536) (vars (5120 + auxBits UB PB) (auxBits UB PB))

theorem cons_vars {k k' n n' : ℕ} {l l' : List (List Lin)} (hk : k = k') (hn : n = n') (hl : l = l') :
    vars k n :: l = vars k' n' :: l' := by
  subst hk hn hl
  rfl

theorem varSlices_link (a : ℕ) : OfVStar.varSlices 0 [64, 32, 96, a, 32, 96, a] =
    [vars 0 1024, vars 1024 512, vars 1536 1536, vars 3072 (16 * a), vars (3072 + 16 * a) 512,
      vars (3584 + 16 * a) 1536, vars (5120 + 16 * a) (16 * a)] := by
  refine cons_vars rfl rfl (cons_vars rfl rfl (cons_vars rfl rfl (cons_vars rfl rfl (cons_vars ?_ rfl
    (cons_vars ?_ rfl (cons_vars ?_ rfl rfl)))))) <;> omega

theorem varSlices_linkPorts (UB PB : ℕ) : OfVStar.varSlices 0 (VStar.linkPorts UB PB) =
    [vars 0 1024, vars 1024 512, vars 1536 1536, vars 3072 (auxBits UB PB), vars (3072 + auxBits UB PB) 512,
      vars (3584 + auxBits UB PB) 1536, vars (5120 + auxBits UB PB) (auxBits UB PB)] :=
  varSlices_link (VStar.auxWords (UB + PB))

set_option maxHeartbeats 1000000 in -- the kernel's check of the builder's `run_input_bind` rewrite needs about 5x
/-- **V*'s unit is `linkRun`**: its builder's state and outputs are `linkRun`'s from its input bits, its input rows
`VStar.linkRows` (over `VStar.linkPorts`' words) and its output ports `rin`, `rout`. -/
theorem linkUnit_ok {UB PB : ℕ} {U : VStar.UnitC} (h : VStar.linkUnit UB PB = .ok U) :
    U.st = ((linkRun UB PB).run ⟨linkIn UB PB, []⟩).2 ∧
      U.outs = (((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.1 ++ ((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.2).toArray ∧
      U.inPorts = VStar.linkRows UB PB ∧ U.outPorts = [VStar.LINK_OUT_WORDS, VStar.LINK_OUT_WORDS] := by
  unfold VStar.linkUnit at h
  unfold linkRun linkIn
  -- Opaque, so that none of the kernel's checks below can evaluate the builder.
  generalize VStar.linkSteps UB PB = S at h ⊢
  cases hc : RecLinkTree.checkShape UB PB with
  | error e => simp [hc, bind, Except.bind] at h
  | ok _ =>
    simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hn : 16 * (VStar.linkPorts UB PB).foldl (· + ·) 0 = 16 * (VStar.linkPorts UB PB).sum + 0 := by
      simp [VStar.linkPorts]
      omega
    simp only [OfVStar.run_input_bind]
    rw [hn, OfVStar.slices_vars, varSlices_linkPorts, Nat.add_zero]
    refine ⟨?_, ?_, ?_, ?_⟩ <;> first | rfl | trivial

theorem checkShape_ok {UB PB : ℕ} {x : Unit} (h : RecLinkTree.checkShape UB PB = .ok x) : UB ≤ 32 ∧ PB ≤ 32 := by
  unfold RecLinkTree.checkShape at h
  split at h
  · rename_i hc
    rw [Bool.and_eq_true, Bool.and_eq_true] at hc
    have h1 : UB ≤ 30 := of_decide_eq_true hc.1.1
    have h2 : PB ≤ 16 := of_decide_eq_true hc.1.2
    omega
  · simp at h

theorem linkUnit_shape {UB PB : ℕ} {U : VStar.UnitC} (h : VStar.linkUnit UB PB = .ok U) : UB ≤ 32 ∧ PB ≤ 32 := by
  unfold VStar.linkUnit at h
  cases hc : RecLinkTree.checkShape UB PB with
  | error e => simp [hc, bind, Except.bind] at h
  | ok x => exact checkShape_ok hc

theorem aux_ge (D : ℕ) : 512 * D ≤ 16 * VStar.auxWords D := by
  unfold VStar.auxWords VStar.NODE_WORDS VStar.ROW_BLOCK
  have := Nat.le_max_right 1 ((32 * D + 64 - 1) / 64)
  omega

/-- **V*'s laid-out `RecLink` unit computes its reference outputs**: whenever its rows hold at `v`, if its input columns
hold the table row `T` (from column 0), the reader's digest `Xr` (1024) and salt `Yr` (1536), its 64-byte siblings
`sr` (from 3072, 512 columns each), the writer's `Xw` and `Yw` (past the reader's `aux` port) and its siblings `sw`
(past the writer's salt), then its output columns hold `linkB`'s `rin ‖ rout`. -/
theorem laid_linkUnit {UB PB : ℕ} {U : VStar.UnitC} (hU : VStar.linkUnit UB PB = .ok U) {l : Laid}
    (hl : U.laid = .ok l) (v : ℕ → Bool) (h : LaidRows l (inRowsOf (Array.replicate U.inWords 16)) v)
    (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray)
    (hT : T.size = 128) (hXr : Xr.size = 64) (hYr : Yr.size = 192) (hXw : Xw.size = 64) (hYw : Yw.size = 192)
    (hsr : sr.length = UB + PB) (hsw : sw.length = UB + PB)
    (htrow : ColsHold v 0 T) (hxr : ColsHold v 1024 Xr) (hyr : ColsHold v 1536 Yr)
    (hraux : ∀ i (hi : i < sr.length), sr[i].size = 64 ∧ ColsHold v (3072 + 512 * i) sr[i])
    (hxw : ColsHold v (3072 + auxBits UB PB) Xw) (hyw : ColsHold v (3584 + auxBits UB PB) Yw)
    (hwaux : ∀ i (hi : i < sw.length), sw[i].size = 64 ∧ ColsHold v (5120 + auxBits UB PB + 512 * i) sw[i]) :
    ColsHold v (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16))
      ((linkB UB PB T Xr Yr Xw Yw sr sw).1 ++ (linkB UB PB T Xr Yr Xw Yw sr sw).2) := by
  obtain ⟨hst, houts, hin, hop⟩ := linkUnit_ok hU
  obtain ⟨hUB, hPB⟩ := linkUnit_shape hU
  have hN : (Array.replicate U.inWords 16).foldl (· + ·) 0 = 5120 + 2 * auxBits UB PB := by
    rw [OfVStar.foldl_replicate, VStar.UnitC.inWords, hin]
    simp only [VStar.linkRows, List.foldl_cons, List.foldl_nil, VStar.LINK_ROW_WORDS, VStar.LINK_X_WORDS,
      VStar.LINK_Y_WORDS, auxBits]
    omega
  have hA := aux_ge (UB + PB)
  have hle : ∀ m, m ≤ 5120 + 2 * auxBits UB PB → m ≤ (Array.replicate U.inWords 16).foldl (· + ·) 0 :=
    fun m hm => by rw [hN]; exact hm
  have hg : Good ((Array.replicate U.inWords 16).foldl (· + ·) 0) U.st := by
    rw [hst, hN, ← linkIn_eq]
    exact fresh_linkSteps UB PB _ _ _ _ _ _ _ (linkIn UB PB) ⟨linkIn UB PB, []⟩ ⟨by simp, fun k r h => by simp at h⟩
  rw [OfVStar.laid_ok hl] at h
  have hz := laid_rows U.st _ _ U.outs v hg h
  rw [hst] at hz
  have hS : Sound (linkRun UB PB) _ := sound_linkSteps UB PB hUB hPB (vars 0 1024) (vars 1024 512) (vars 1536 1536)
    (vars 3072 (auxBits UB PB)) (vars (3072 + auxBits UB PB) 512) (vars (3584 + auxBits UB PB) 1536)
    (vars (5120 + auxBits UB PB) (auxBits UB PB))
  obtain ⟨h1, h2⟩ := (hS ⟨linkIn UB PB, []⟩).2 _ hz T Xr Yr Xw Yw sr sw hT hXr hYr hXw hYw
    (carries_vars (by omega) (hle _ (by omega)) htrow) (carries_vars (by omega) (hle _ (by omega)) hxr)
    (carries_vars (by omega) (hle _ (by omega)) hyr) (carries_vars (by omega) (hle _ (by omega)) hxw)
    (carries_vars (by omega) (hle _ (by omega)) hyw) (forall₂_sibs sr hsr hA (hle _ (by omega)) hraux)
    (forall₂_sibs sw hsw hA (hle _ (by omega)) hwaux)
  have hw : (Array.replicate U.outWords 16).foldl (· + ·) 0 = 2048 := by
    rw [OfVStar.foldl_replicate, VStar.UnitC.outWords, hop]
    rfl
  have hsz := size_linkB UB PB T Xr Yr Xw Yw sr sw
  have hlen : (((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.1 ++ ((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.2).length ≤
      (Array.replicate U.outWords 16).foldl (· + ·) 0 :=
    calc _ = 8 * 128 + 8 * 128 := by rw [List.length_append, h1.1, h2.1, hsz.1, hsz.2]
      _ ≤ 2048 := by norm_num
      _ = _ := hw.symm
  rw [houts] at h
  exact colsHold_outs (st := U.st) hlen h (carries_append h1 h2)

section Unit

open FlockSoundness FlockSoundness.Audit

/-- **A copy of V*'s laid-out `RecLink` unit, in a unit of any partition, computes its reference outputs**
(`laid_linkUnit`): let the verifier build `net` from the layout of `VStar.linkUnit UB PB` (`netOf`), and let unit `u` be
an instance of `k` stacked copies of the net's rows, the transcript `tr` carrying 1 at the stack's constant. Then at copy
`j`'s column values (`copyVal`), `laid_linkUnit` holds. -/
theorem unit_linkUnit {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {net : Flock.Net} (hB : Built net)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (tr : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : tr (I.wire ((Rows.ofNet hB).stack k).one) = true) {UB PB : ℕ} {U : VStar.UnitC}
    (hU : VStar.linkUnit UB PB = .ok U) {l : Laid} (hl : U.laid = .ok l) {name : String}
    (hn : Flock.HmNets.netOf name l = .ok net) (hu : net.unitLog ≤ 32)
    (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray)
    (hT : T.size = 128) (hXr : Xr.size = 64) (hYr : Yr.size = 192) (hXw : Xw.size = 64) (hYw : Yw.size = 192)
    (hsr : sr.length = UB + PB) (hsw : sw.length = UB + PB)
    (htrow : ColsHold (copyVal I tr hj) 0 T) (hxr : ColsHold (copyVal I tr hj) 1024 Xr)
    (hyr : ColsHold (copyVal I tr hj) 1536 Yr)
    (hraux : ∀ i (hi : i < sr.length), sr[i].size = 64 ∧ ColsHold (copyVal I tr hj) (3072 + 512 * i) sr[i])
    (hxw : ColsHold (copyVal I tr hj) (3072 + auxBits UB PB) Xw)
    (hyw : ColsHold (copyVal I tr hj) (3584 + auxBits UB PB) Yw)
    (hwaux : ∀ i (hi : i < sw.length), sw[i].size = 64 ∧
      ColsHold (copyVal I tr hj) (5120 + auxBits UB PB + 512 * i) sw[i]) :
    ColsHold (copyVal I tr hj) (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16))
      ((linkB UB PB T Xr Yr Xw Yw sr sw).1 ++ (linkB UB PB T Xr Yr Xw Yw sr sw).2) := by
  obtain ⟨hs, h1⟩ := sat_copy I tr hj
  have hL := laidRows_ofNet hB hn hu hs (h1.trans hone)
  rw [OfVStar.inWords_laid hl hn] at hL
  exact laid_linkUnit hU hl _ hL T Xr Yr Xw Yw sr sw hT hXr hYr hXw hYw hsr hsw htrow hxr hyr hraux hxw hyw hwaux

end Unit

/-! ## At the reference's u16 words -/

theorem bytesOf_append (a b : List ℕ) :
    RecLinkRef.bytesOf (a ++ b) = RecLinkRef.bytesOf a ++ RecLinkRef.bytesOf b := by
  apply ByteArray.ext
  simp [RecLinkRef.bytesOf, List.flatMap_append]

theorem flatMap_zeros : ∀ n : ℕ,
    (List.replicate n 0).flatMap (fun w : ℕ => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]) =
      List.replicate (2 * n) (0 : UInt8)
  | 0 => rfl
  | n + 1 => by
    rw [List.replicate_succ, List.flatMap_cons, flatMap_zeros n, show 2 * (n + 1) = 2 * n + 1 + 1 by omega,
      List.replicate_succ, List.replicate_succ]
    rfl

theorem bytesOf_replicate (n : ℕ) : RecLinkRef.bytesOf (List.replicate n 0) = zeroB (2 * n) := by
  unfold RecLinkRef.bytesOf zeroB
  rw [flatMap_zeros]
  rfl

theorem zeroB_add (m n : ℕ) : zeroB (m + n) = zeroB m ++ zeroB n := by
  apply ByteArray.ext
  simp [zeroB]

theorem toUInt8_toNat (a : UInt8) : a.toNat.toUInt8 = a := by
  apply UInt8.toNat_inj.1
  simp

theorem pair_lo (a b : UInt8) : ((a.toNat + 256 * b.toNat) % 256).toUInt8 = a := by
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (UInt8.toNat_lt a), toUInt8_toNat]

theorem pair_hi (a b : UInt8) : ((a.toNat + 256 * b.toNat) / 256 % 256).toUInt8 = b := by
  rw [show a.toNat + 256 * b.toNat = a.toNat + b.toNat * 256 by omega, Nat.add_mul_div_right _ _ (by omega),
    Nat.div_eq_of_lt (UInt8.toNat_lt a), Nat.zero_add, Nat.mod_eq_of_lt (UInt8.toNat_lt b), toUInt8_toNat]

theorem get!_eq (b : ByteArray) (i : ℕ) : b.get! i = b.data.toList.getD i 0 := by
  cases b with
  | mk bs =>
    show bs[i]! = _
    rw [Array.getElem!_eq_getD]
    unfold Array.getD
    split <;> simp_all <;> rfl

theorem pairs_eq : ∀ (l : List UInt8) (m : ℕ), l.length = 2 * m →
    ((List.range m).map fun i => (l.getD (2 * i) 0).toNat + 256 * (l.getD (2 * i + 1) 0).toNat).flatMap
      (fun w : ℕ => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]) = l
  | [], 0, _ => rfl
  | a :: b :: r, m + 1, h => by
    rw [List.range_succ_eq_map, List.map_cons, List.flatMap_cons, List.map_map]
    have hr : r.length = 2 * m := by rw [List.length_cons, List.length_cons] at h; omega
    have := pairs_eq r m hr
    simp only [Nat.mul_zero, Nat.zero_add, List.getD_cons_zero, List.getD_cons_succ] at *
    rw [pair_lo, pair_hi]
    show a :: b :: _ = a :: b :: r
    congr 2
  | [], m + 1, h => by rw [List.length_nil] at h; omega
  | [_], _, h => by rw [List.length_singleton] at h; omega
  | _ :: _ :: _, 0, h => by rw [List.length_cons, List.length_cons] at h; omega

/-- `bytes_of(words_of(b))` is `b` for `b` of even size. -/
theorem bytesOf_wordsOf (b : ByteArray) (h : b.size % 2 = 0) : RecLinkRef.bytesOf (RecLinkRef.wordsOf b) = b := by
  apply ByteArray.ext
  apply Array.toList_inj.1
  unfold RecLinkRef.bytesOf RecLinkRef.wordsOf
  simp only [get!_eq]
  have hl : b.data.toList.length = 2 * (b.size / 2) := by
    rw [Array.length_toList, ByteArray.size_data]
    omega
  rw [List.toList_toArray]
  exact pairs_eq _ _ hl

theorem bytesOf_expected {r : ByteArray} (hr : r.size = 64) :
    RecLinkRef.bytesOf (RecLinkRef.expected r) = r ++ zeroB 64 :=
  (bytesOf_append _ _).trans (congr (congrArg (· ++ ·) (bytesOf_wordsOf r (by omega))) (bytesOf_replicate 32))

theorem bytesOf_zeros : RecLinkRef.bytesOf RecLinkRef.zeros = zeroB 64 ++ zeroB 64 :=
  (bytesOf_replicate 64).trans (zeroB_add 64 64)

/-- **The unit's outputs are the reference's `linkOut`**: on ports that are the bytes of u16 words (`bytesOf`), with
the siblings `sibsOf` reads off `raux` and `waux`, `linkB`'s `(rin, rout)` are the bytes of `linkOut`'s. -/
theorem linkB_words (UB PB : ℕ) (trow xr yr raux xw yw waux : List ℕ) :
    linkB UB PB (RecLinkRef.bytesOf trow) (RecLinkRef.bytesOf xr) (RecLinkRef.bytesOf yr) (RecLinkRef.bytesOf xw)
        (RecLinkRef.bytesOf yw) (RecLinkRef.sibsOf raux (UB + PB)) (RecLinkRef.sibsOf waux (UB + PB)) =
      (RecLinkRef.bytesOf (RecLinkRef.linkOut UB PB trow xr yr raux xw yw waux).1,
        RecLinkRef.bytesOf (RecLinkRef.linkOut UB PB trow xr yr raux xw yw waux).2) := by
  have hls : ∀ Dg Sl : ByteArray, (RecLinkTree.leaf (Hm96.Default512.commitString Dg Sl)).size = 64 :=
    fun _ _ => Sha512.hash_size _
  have h1 := AccChain.climb_size_of RecLinkRef.scheme (fun _ _ => rfl) (RecLinkRef.sibsOf raux (UB + PB))
    (RecLinkRef.posOf (RecLinkRef.bytesOf trow) 1 2 PB (UB + PB)) _
    (hls (RecLinkRef.bytesOf xr) (RecLinkRef.bytesOf yr))
  have h2 := AccChain.climb_size_of RecLinkRef.scheme (fun _ _ => rfl) (RecLinkRef.sibsOf waux (UB + PB))
    (RecLinkRef.posOf (RecLinkRef.bytesOf trow) 3 4 PB (UB + PB)) _
    (hls (RecLinkRef.bytesOf xw) (RecLinkRef.bytesOf yw))
  unfold linkB RecLinkRef.linkOut
  cases RecLinkRef.okRef UB PB (RecLinkRef.bytesOf trow) (RecLinkRef.bytesOf xr) (RecLinkRef.bytesOf xw)
  · exact Prod.ext bytesOf_zeros.symm bytesOf_zeros.symm
  · exact Prod.ext (bytesOf_expected h1).symm (bytesOf_expected h2).symm

end FlockVBridge.RecLink
