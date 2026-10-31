import Proofs.Flock.Recursive.LinksRef
import Proofs.Flock.VBridge.RecLink
import Proofs.Flock.Claims.Recursion.VerdictRowEBytes
import Proofs.Flock.Claims.Recursion.MoreSlots
import Proofs.Flock.Claims.Recursion.VStarParts

/-!
# A link part's unit at each instance: the reference step

`LinkRefStaged`: at a staged link part (V\*'s statement `.link UB PB N`), instance `n < N`'s reference link check
(`linkRefAt`) outputs `(expected R, expected W)`, or two of `refMsgs` collide.

* **The inputs.** The unit's copy `n` (`copyOf`) holds its laid-out rows (`laidRows_of_satisfies` at the net the
  setup accepted, `RecLinkPart.unitNet`), and its input columns hold the seven ports' words: port `q` of V\*'s rows
  (`linkRows_eq`) from column `16 Σ_{q' < q} words(q')` (`colsHold_inRow`, `inPort`). Those are the u16 words
  `inWordsAt` reads, so the reference check accepts them (`linkRef_of_shape`).
* **The outputs.** V\*'s laid-out `RecLink{UB, PB}` computes `linkOut` at its output columns (`laid_linkUnit`,
  `linkB_words`, `unit_out`), which are the bits `outRowForms` reads at the `rin` and `rout` rows (`colsHold_linkOut`,
  `colsHold_outRow`'s argument at `RecLinkPart.outsOk`'s ports).
* **The rows.** The `rin` (`rout`) slot commits that row (`slotOut_stagedG`) and check 5 holds it to `expected R`
  (`expected W`, `linkValue_slot`), so the two rows are equal or SHA-512 collides (`linkOut_bind`, `out_or_coll`).
  Two u16 lists with one byte string are equal (`bytesOf_inj`).
-/

namespace RecDeploy.Links

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Hm (bitL)
open FlockVBridge (ColsHold)

namespace RefStep

/-! ## Columns and bytes -/

theorem colsHold_split {v : ℕ → Bool} {base : ℕ} {a b : ByteArray} (h : ColsHold v base (a ++ b)) :
    ColsHold v base a ∧ ColsHold v (base + 8 * a.size) b := by
  have hl : a.data.toList.length = a.size := Array.length_toList
  refine ⟨fun t ht => ?_, fun t ht => ?_⟩
  · rw [← FlockVBridge.bitL_append_left b.data.toList (a := a.data.toList) (t := t) (by omega),
      ← ByteArray.toList_data_append]
    exact h t (by rw [ByteArray.size_append]; omega)
  · have e := h (8 * a.size + t) (by rw [ByteArray.size_append]; omega)
    rw [ByteArray.toList_data_append, FlockVBridge.bitL_append_right _ (by omega), hl, Nat.add_sub_cancel_left,
      ← Nat.add_assoc] at e
    exact e

theorem size_bytesOf (ws : List ℕ) : (RecLinkRef.bytesOf ws).size = 2 * ws.length := by
  show (List.flatMap _ ws).toArray.size = _
  rw [List.size_toArray]
  induction ws with
  | nil => rfl
  | cons a l ih =>
    rw [List.flatMap_cons, List.length_append, ih, List.length_cons]
    simp only [List.length_cons, List.length_nil]
    omega

theorem flatMap_take (g : ℕ → List UInt8) (hg : ∀ x, (g x).length = 2) :
    ∀ (ws : List ℕ) (m : ℕ), (ws.take m).flatMap g = (ws.flatMap g).take (2 * m)
  | [], m => by simp
  | a :: l, 0 => by simp
  | a :: l, m + 1 => by
    rw [List.take_succ_cons, List.flatMap_cons, List.flatMap_cons, flatMap_take g hg l m,
      show 2 * (m + 1) = (g a).length + 2 * m by rw [hg]; omega, List.take_length_add_append]

theorem flatMap_drop (g : ℕ → List UInt8) (hg : ∀ x, (g x).length = 2) :
    ∀ (ws : List ℕ) (j : ℕ), (ws.drop j).flatMap g = (ws.flatMap g).drop (2 * j)
  | [], j => by simp
  | a :: l, 0 => by simp
  | a :: l, j + 1 => by
    rw [List.drop_succ_cons, List.flatMap_cons, flatMap_drop g hg l j,
      show 2 * (j + 1) = (g a).length + 2 * j by rw [hg]; omega, List.drop_length_add_append]

theorem bytesOf_slice (ws : List ℕ) (j m : ℕ) :
    (RecLinkRef.bytesOf ((ws.drop j).take m)).data.toList =
      ((RecLinkRef.bytesOf ws).data.toList.drop (2 * j)).take (2 * m) := by
  show List.flatMap _ _ = ((List.flatMap _ _).drop _).take _
  rw [flatMap_take _ (fun _ => rfl), flatMap_drop _ (fun _ => rfl)]

theorem colsHold_slice {v : ℕ → Bool} {base : ℕ} {B : ByteArray} (h : ColsHold v base B) (j m : ℕ)
    (hjm : j + m ≤ B.size) : ColsHold v (base + 8 * j) ⟨((B.data.toList.drop j).take m).toArray⟩ := by
  have hl : B.data.toList.length = B.size := Array.length_toList
  intro t ht
  have hs : (⟨((B.data.toList.drop j).take m).toArray⟩ : ByteArray).size = m := by
    show ((B.data.toList.drop j).take m).toArray.size = m
    rw [List.size_toArray, List.length_take, List.length_drop]
    omega
  rw [hs] at ht
  have e := h (8 * j + t) (by omega)
  rw [← Nat.add_assoc] at e
  rw [e]
  show bitL B.data.toList (8 * j + t) = bitL ((B.data.toList.drop j).take m) t
  unfold bitL
  rw [show (8 * j + t) / 8 = j + t / 8 by omega, show (8 * j + t) % 8 = t % 8 by omega,
    List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt (by omega),
    List.getElem?_drop]

/-- **Columns that hold a word list hold each run of its words**, `16` columns a word. -/
theorem colsHold_sliceW {v : ℕ → Bool} {base : ℕ} {ws : List ℕ} (h : ColsHold v base (RecLinkRef.bytesOf ws))
    (j m : ℕ) (hjm : j + m ≤ ws.length) {b' : ℕ} (hb : base + 16 * j = b') :
    ColsHold v b' (RecLinkRef.bytesOf ((ws.drop j).take m)) := by
  subst hb
  have e : RecLinkRef.bytesOf ((ws.drop j).take m) =
      ⟨(((RecLinkRef.bytesOf ws).data.toList.drop (2 * j)).take (2 * m)).toArray⟩ := by
    rw [← bytesOf_slice]
  rw [e, show base + 16 * j = base + 8 * (2 * j) by omega]
  exact colsHold_slice h _ _ (by rw [size_bytesOf]; omega)

theorem colsHold_takeW {v : ℕ → Bool} {base : ℕ} {ws : List ℕ} (h : ColsHold v base (RecLinkRef.bytesOf ws))
    (m : ℕ) (hm : m ≤ ws.length) : ColsHold v base (RecLinkRef.bytesOf (ws.take m)) := by
  have e := colsHold_sliceW h 0 m (by omega) (b' := base) (by omega)
  rwa [List.drop_zero] at e

theorem colsHold_dropW {v : ℕ → Bool} {base : ℕ} {ws : List ℕ} (h : ColsHold v base (RecLinkRef.bytesOf ws))
    (j : ℕ) (hj : j ≤ ws.length) {b' : ℕ} (hb : base + 16 * j = b') :
    ColsHold v b' (RecLinkRef.bytesOf (ws.drop j)) := by
  have e := colsHold_sliceW h j (ws.length - j) (by omega) hb
  have et : (ws.drop j).take (ws.length - j) = ws.drop j := List.take_of_length_le (by simp)
  rw [et] at e
  exact e

theorem colsHold_eq {v : ℕ → Bool} {base : ℕ} {a b : ByteArray} (ha : ColsHold v base a) (hb : ColsHold v base b)
    (hs : a.size = b.size) : a = b := by
  have hla : a.data.toList.length = a.size := Array.length_toList
  have hlb : b.data.toList.length = b.size := Array.length_toList
  apply ByteArray.ext
  apply Array.toList_inj.1
  refine FlockSoundness.Discharge.Hm.eq_of_bitL (by omega) fun t ht => ?_
  rw [← ha t (by omega), hb t (by omega)]

/-- **Two 128-byte strings the same columns hold are equal.** -/
theorem bytes_of_cols {v : ℕ → Bool} {b1 b2 : ℕ} {l : List UInt8} {b : ByteArray} (hl : ColsHold v b1 ⟨l.toArray⟩)
    (hb : ColsHold v b2 b) (hbb : b1 = b2) (hlen : l.length = 128) (hs : b.size = 128) : l = b.data.toList := by
  subst hbb
  have e := colsHold_eq hl hb (by show l.toArray.size = b.size; rw [List.size_toArray]; omega)
  subst e
  rfl

/-! ## u16 words -/

theorem length_wordsOf (b : ByteArray) : (RecLinkRef.wordsOf b).length = b.size / 2 := by
  simp only [RecLinkRef.wordsOf, List.length_map, List.length_range]

theorem wordsOf_u16 (b : ByteArray) : ∀ x ∈ RecLinkRef.wordsOf b, x < 65536 := by
  intro x hx
  simp only [RecLinkRef.wordsOf, List.mem_map] at hx
  obtain ⟨i, -, rfl⟩ := hx
  have h1 : (b.get! (2 * i)).toNat < 256 := UInt8.toNat_lt _
  have h2 : (b.get! (2 * i + 1)).toNat < 256 := UInt8.toNat_lt _
  omega

theorem expected_u16 (r : ByteArray) : ∀ x ∈ RecLinkRef.expected r, x < 65536 := by
  intro x hx
  rcases List.mem_append.1 hx with h | h
  · exact wordsOf_u16 r x h
  · rw [(List.mem_replicate.1 h).2]; decide

theorem zeros_u16 : ∀ x ∈ RecLinkRef.zeros, x < 65536 := by
  intro x hx
  rw [RecLinkRef.zeros, List.mem_replicate] at hx
  rw [hx.2]; decide

theorem linkOut_u16 (UB PB : ℕ) (trow xr yr raux xw yw waux : List ℕ) :
    (∀ x ∈ (RecLinkRef.linkOut UB PB trow xr yr raux xw yw waux).1, x < 65536) ∧
      ∀ x ∈ (RecLinkRef.linkOut UB PB trow xr yr raux xw yw waux).2, x < 65536 := by
  unfold RecLinkRef.linkOut
  split
  · exact ⟨expected_u16 _, expected_u16 _⟩
  · exact ⟨zeros_u16, zeros_u16⟩

theorem size_expected {r : ByteArray} (hr : r.size = 64) : (RecLinkRef.expected r).toArray.size = 64 := by
  rw [List.size_toArray, RecLinkRef.expected, List.length_append, length_wordsOf, List.length_replicate, hr]
  rfl

/-- **u16 words are their bytes**: two lists of u16s with one `bytesOf` are equal. -/
theorem bytesOf_inj : ∀ {x y : List ℕ}, (∀ a ∈ x, a < 65536) → (∀ a ∈ y, a < 65536) →
    RecLinkRef.bytesOf x = RecLinkRef.bytesOf y → x = y
  | [], [], _, _, _ => rfl
  | [], b :: y, _, _, h => by
    have := congrArg ByteArray.size h
    rw [size_bytesOf, size_bytesOf] at this
    simp at this
  | a :: x, [], _, _, h => by
    have := congrArg ByteArray.size h
    rw [size_bytesOf, size_bytesOf] at this
    simp at this
  | a :: x, b :: y, hx, hy, h => by
    have hd := congrArg (fun z : ByteArray => z.data.toList) h
    simp only [RecLinkRef.bytesOf, List.flatMap_cons, List.cons_append, List.nil_append, List.cons.injEq] at hd
    obtain ⟨h1, h2, h3⟩ := hd
    have hl : x = y := bytesOf_inj (fun c hc => hx c (List.mem_cons_of_mem _ hc))
      (fun c hc => hy c (List.mem_cons_of_mem _ hc)) (by
        show (⟨(x.flatMap _).toArray⟩ : ByteArray) = ⟨(y.flatMap _).toArray⟩
        rw [h3])
    have ha := hx a (List.mem_cons_self ..)
    have hb := hy b (List.mem_cons_self ..)
    have e1 := congrArg UInt8.toNat h1
    have e2 := congrArg UInt8.toNat h2
    simp only [Nat.toUInt8, UInt8.toNat_ofNat', Nat.reducePow] at e1 e2
    rw [hl, show a = b by omega]

theorem wordBytes_eq (l : List ℕ) : ResLower.wordBytes l.toArray = (RecLinkRef.bytesOf l).data.toList := rfl

theorem eq_of_wordBytes {xs ys : List ℕ} (h : (RecLinkRef.bytesOf xs).data.toList = ResLower.wordBytes ys.toArray)
    (hxs : ∀ x ∈ xs, x < 65536) (hys : ∀ x ∈ ys, x < 65536) : xs = ys :=
  bytesOf_inj hxs hys (bytes_eq_of_toList (h.trans (wordBytes_eq ys)))

theorem ports_u16 (b0 b1 b2 b3 b4 : ByteArray) :
    ∀ l ∈ [RecLinkRef.wordsOf b0, (RecLinkRef.wordsOf b1).take 32, (RecLinkRef.wordsOf b1).drop 32,
      RecLinkRef.wordsOf b2, (RecLinkRef.wordsOf b3).take 32, (RecLinkRef.wordsOf b3).drop 32,
      RecLinkRef.wordsOf b4], ∀ x ∈ l, x < 65536 := by
  intro l hl x hx
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hl
  rcases hl with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact wordsOf_u16 _ x hx
  · exact wordsOf_u16 _ x (List.mem_of_mem_take hx)
  · exact wordsOf_u16 _ x (List.mem_of_mem_drop hx)
  · exact wordsOf_u16 _ x hx
  · exact wordsOf_u16 _ x (List.mem_of_mem_take hx)
  · exact wordsOf_u16 _ x (List.mem_of_mem_drop hx)
  · exact wordsOf_u16 _ x hx

theorem auxWords_eq (D : ℕ) : RecLinkRef.auxWords D = VStar.auxWords D := rfl

theorem padTo_64 {E : Array ℕ} (h : E.size = 64) : RecCoins.padTo E (Registered.rowWords E.size 16) = E := by
  unfold RecCoins.padTo Registered.rowWords
  rw [h]
  simp

/-! ## The reference check and the unit -/

/-- **The reference link check accepts ports of its words, u16s**, at a shape `checkShape` accepts. -/
theorem linkRef_of_shape {UB PB : ℕ} {trow xr yr raux xw yw waux : List ℕ}
    (hs : RecLinkTree.checkShape UB PB = .ok ()) (h0 : trow.length = 64) (h1 : xr.length = 32)
    (h2 : yr.length = 96) (h3 : raux.length = RecLinkRef.auxWords (UB + PB)) (h4 : xw.length = 32)
    (h5 : yw.length = 96) (h6 : waux.length = RecLinkRef.auxWords (UB + PB))
    (hb : ∀ l ∈ [trow, xr, yr, raux, xw, yw, waux], ∀ x ∈ l, x < 65536) :
    RecLinkRef.linkRef UB PB trow xr yr raux xw yw waux = .ok (RecLinkRef.linkOut UB PB trow xr yr raux xw yw waux) := by
  have hg : [trow, xr, yr, raux, xw, yw, waux].map (·.length) = RecLinkRef.portWords UB PB := by
    simp only [List.map_cons, List.map_nil, h0, h1, h2, h3, h4, h5, h6]
    rfl
  have ha : [trow, xr, yr, raux, xw, yw, waux].all (·.all (· < 65536)) = true :=
    List.all_eq_true.2 fun l hl => List.all_eq_true.2 fun x hx => decide_eq_true (hb l hl x hx)
  unfold RecLinkRef.linkRef
  simp only [hs, hg, ha, beq_self_eq_true, bind, Except.bind, pure, Except.pure, ↓reduceIte]

theorem checkShape_of_unit {UB PB : ℕ} {u : VStar.UnitC} (hu : VStar.linkUnit UB PB = .ok u) :
    RecLinkTree.checkShape UB PB = .ok () := by
  unfold VStar.linkUnit at hu
  cases hc : RecLinkTree.checkShape UB PB with
  | error e =>
    rw [hc] at hu
    exact absurd hu (by intro h; cases h)
  | ok x => rfl

theorem unitNet_laid {UB PB : ℕ} {u : VStar.UnitC} (hu : VStar.linkUnit UB PB = .ok u) {net : Flock.Net}
    (h : RecLinkPart.unitNet UB PB = .ok net) :
    ∃ l, u.laid = .ok l ∧ Flock.HmNets.netOf (VStar.linkName UB PB) l = .ok net := by
  unfold RecLinkPart.unitNet at h
  cases hl : u.laid with
  | error e => simp [hu, hl, bind, Except.bind] at h
  | ok l => exact ⟨l, rfl, by simpa [hu, hl, bind, Except.bind] using h⟩

theorem inCol_laid {U : VStar.UnitC} {l : Flock.HmNets.Laid} (hl : U.laid = .ok l) {name : String}
    {net : Flock.Net} (hn : Flock.HmNets.netOf name l = .ok net) : (net.inGroups.getD 0 default).col = 0 := by
  unfold Flock.HmNets.netOf at hn
  rw [(Flock.Net.ofRows_ok hn).2.2.2.2.2.1, FlockVBridge.OfVStar.laid_ok hl,
    FlockSoundness.Discharge.Hm.layout_inGroups]
  rfl

/-- **V\*'s link part's row ports**, by index: name, words and bits, the earlier rows' words. -/
theorem port_facts {ps : Array Port} {L : List (String × ℕ × ℕ)}
    (hr : ps.toList.map (fun q => (q.name, q.words, q.bits)) = L) {k : ℕ} {nm : String} {wd bt : ℕ}
    (hk : L[k]? = some (nm, wd, bt)) (hf : L.findIdx? (·.1 == nm) = some k) :
    ∃ hks : k < ps.size, ps.findIdx? (·.name == nm) = some k ∧ ps[k].words = wd ∧
      ps[k].bits = bt ∧ (ps.extract 0 k).foldl (· + ·.words) 0 = ((L.take k).map (·.2.1)).sum := by
  subst hr
  have hks : k < ps.size := by
    have := (List.getElem?_eq_some_iff.1 hk).1
    rw [List.length_map, Array.length_toList] at this
    exact this
  rw [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hks, Option.map_some,
    Option.some.injEq, Prod.mk.injEq, Prod.mk.injEq] at hk
  refine ⟨hks, ?_, hk.2.1, hk.2.2, ?_⟩
  · rw [List.findIdx?_map] at hf
    exact (List.findIdx?_toArray _ ps.toList).trans hf
  · rw [extract_words Port.words]
    simp only [List.map_take, List.map_map]
    rfl

/-- **A copy's columns hold an input port's words**: where the unit reads its `N` input words in order from column 0,
port `q` (`nm`), after `S` words of earlier ports and `W` words of its own, is the u16 words `inWordsAt` reads, from
column `16 S`. -/
theorem inPort {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (h0 : 0 < p.sts.size) {c : Circuit}
    (hc0 : p.sts[0].c = c) {tags : Flock.Tags} (P : FlockSoundness.ParseFacts tags c) {N : ℕ}
    (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) (hcol : (c.unit.inGroups.getD 0 default).col = 0)
    (hN0 : 0 < N) (w : Claims.Flock.Wit) (n : ℕ) {q : ℕ} {nm : String}
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hqs : q < c.ports.size) {S W : ℕ}
    (hx : (c.ports.extract 0 q).foldl (· + ·.words) 0 = S) (hwq : c.ports[q].words = W)
    (hbq : c.ports[q].bits = 16 * W) (hfit : S + W ≤ N) {base : ℕ} (hbase : 16 * S = base) :
    ColsHold (Flock.Readable.copyOf p 0 h0 w (n / c.g) (n % c.g) 0) base
        (RecLinkRef.bytesOf (inWordsAt c w nm n)) ∧ (inWordsAt c w nm n).length = W := by
  have hC := leaves_le_constPos P hl hN0
  obtain ⟨hcs, hlen⟩ := colsHold_inRow h1 h0 hc0 w n hl hq hqs (by rw [hbq, hwq]) (by rw [hx, hbq]; omega) hC
    (base := base) (by rw [hcol, hx, Nat.zero_mul, Nat.zero_add]; exact hbase)
  have e : (⟨(readSegs w (accSegs c (nm, n))).toArray⟩ : ByteArray) = inBytesAt c w nm n := by
    rw [readSegs_accSegs]
    rfl
  have hsz : (inBytesAt c w nm n).size = 2 * W := by
    rw [← e]
    show (readSegs w (accSegs c (nm, n))).toArray.size = _
    rw [List.size_toArray, hlen, hwq]
  have hb := FlockVBridge.RecLink.bytesOf_wordsOf (inBytesAt c w nm n) (by rw [hsz]; omega)
  refine ⟨?_, ?_⟩
  · show ColsHold _ base (RecLinkRef.bytesOf (RecLinkRef.wordsOf (inBytesAt c w nm n)))
    rw [hb, ← e]
    exact hcs
  · show (RecLinkRef.wordsOf (inBytesAt c w nm n)).length = W
    rw [length_wordsOf, hsz]
    omega

theorem sibs_cols {v : ℕ → Bool} {D : ℕ} {A : List ℕ} {base : ℕ} (h : ColsHold v base (RecLinkRef.bytesOf A))
    (hl : 32 * D ≤ A.length) : ∀ i (hi : i < (RecLinkRef.sibsOf A D).length),
      (RecLinkRef.sibsOf A D)[i].size = 64 ∧ ColsHold v (base + 512 * i) (RecLinkRef.sibsOf A D)[i] := by
  intro i hi
  have hi' : i < D := by simpa only [RecLinkRef.sibsOf, List.length_map, List.length_range] using hi
  have e : (RecLinkRef.sibsOf A D)[i] = RecLinkRef.bytesOf ((A.drop (32 * i)).take 32) := by
    simp only [RecLinkRef.sibsOf, List.getElem_map, List.getElem_range, RecLinkRef.NODE_WORDS]
  rw [e]
  exact ⟨by rw [size_bytesOf, List.length_take, List.length_drop]; omega,
    colsHold_sliceW h (32 * i) 32 (by omega) (by omega)⟩

/-- **V\*'s laid-out `RecLink{UB, PB}` outputs `linkOut` of its ports' words**: where its rows hold at `v` and its
input columns hold the words `trow`, `xyr`, `raux`, `xyw`, `waux` (from columns `0`, `1024`, `3072`, `3072 + 16 a`,
`5120 + 16 a`, `a` the `aux` ports' words), its output columns hold `linkOut`'s two rows of 128 bytes
(`laid_linkUnit`, `linkB_words`). -/
theorem unit_out {UB PB : ℕ} {U : VStar.UnitC} (hU : VStar.linkUnit UB PB = .ok U) {l : Flock.HmNets.Laid}
    (hl : U.laid = .ok l) {v : ℕ → Bool}
    (h : FlockSoundness.Discharge.Hm.LaidRows l
      (FlockSoundness.Discharge.Hm.inRowsOf (Array.replicate U.inWords 16)) v)
    {TR XY RA XW WA : List ℕ} (h0 : TR.length = 64) (h1 : XY.length = 128)
    (h2 : RA.length = VStar.auxWords (UB + PB)) (h3 : XW.length = 128) (h4 : WA.length = VStar.auxWords (UB + PB))
    (c0 : ColsHold v 0 (RecLinkRef.bytesOf TR)) (c1 : ColsHold v 1024 (RecLinkRef.bytesOf XY))
    (c2 : ColsHold v 3072 (RecLinkRef.bytesOf RA))
    (c3 : ColsHold v (3072 + 16 * VStar.auxWords (UB + PB)) (RecLinkRef.bytesOf XW))
    (c4 : ColsHold v (5120 + 16 * VStar.auxWords (UB + PB)) (RecLinkRef.bytesOf WA)) :
    ColsHold v (FlockSoundness.Discharge.Hm.outStart U.st (Array.replicate U.inWords 16)
        (Array.replicate U.outWords 16))
      (RecLinkRef.bytesOf (RecLinkRef.linkOut UB PB TR (XY.take 32) (XY.drop 32) RA (XW.take 32) (XW.drop 32) WA).1 ++
        RecLinkRef.bytesOf (RecLinkRef.linkOut UB PB TR (XY.take 32) (XY.drop 32) RA (XW.take 32) (XW.drop 32) WA).2) ∧
      (RecLinkRef.bytesOf
        (RecLinkRef.linkOut UB PB TR (XY.take 32) (XY.drop 32) RA (XW.take 32) (XW.drop 32) WA).1).size = 128 ∧
      (RecLinkRef.bytesOf
        (RecLinkRef.linkOut UB PB TR (XY.take 32) (XY.drop 32) RA (XW.take 32) (XW.drop 32) WA).2).size = 128 := by
  have hA := FlockVBridge.RecLink.aux_ge (UB + PB)
  have hab : FlockVBridge.RecLink.auxBits UB PB = 16 * VStar.auxWords (UB + PB) := rfl
  have hO := FlockVBridge.RecLink.laid_linkUnit hU hl v h (RecLinkRef.bytesOf TR) (RecLinkRef.bytesOf (XY.take 32))
    (RecLinkRef.bytesOf (XY.drop 32)) (RecLinkRef.bytesOf (XW.take 32)) (RecLinkRef.bytesOf (XW.drop 32))
    (RecLinkRef.sibsOf RA (UB + PB)) (RecLinkRef.sibsOf WA (UB + PB))
    (by rw [size_bytesOf]; omega) (by rw [size_bytesOf, List.length_take]; omega)
    (by rw [size_bytesOf, List.length_drop]; omega) (by rw [size_bytesOf, List.length_take]; omega)
    (by rw [size_bytesOf, List.length_drop]; omega)
    (by simp only [RecLinkRef.sibsOf, List.length_map, List.length_range])
    (by simp only [RecLinkRef.sibsOf, List.length_map, List.length_range])
    c0 (colsHold_takeW c1 32 (by omega)) (colsHold_dropW c1 32 (by omega) (by omega))
    (sibs_cols c2 (by omega)) (colsHold_takeW c3 32 (by omega))
    (colsHold_dropW c3 32 (by omega) (by omega)) (sibs_cols c4 (by omega))
  have hsz := FlockVBridge.RecLink.size_linkB UB PB (RecLinkRef.bytesOf TR) (RecLinkRef.bytesOf (XY.take 32))
    (RecLinkRef.bytesOf (XY.drop 32)) (RecLinkRef.bytesOf (XW.take 32)) (RecLinkRef.bytesOf (XW.drop 32))
    (RecLinkRef.sibsOf RA (UB + PB)) (RecLinkRef.sibsOf WA (UB + PB))
  rw [FlockVBridge.RecLink.linkB_words] at hO hsz
  exact ⟨hO, hsz⟩

/-- **A copy's columns hold its unit's output row `jo`**, `colsHold_outRow` at a link part's output layout
(`RecLinkPart.outsOk`: output ports `rin` and `rout`, one output port per output row, 64 words each), which names
them as `OutsLaid` doesn't. -/
theorem colsHold_linkOut {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (h0 : 0 < p.sts.size) {c : Circuit}
    (hc0 : p.sts[0].c = c) {tags : Flock.Tags} (P : FlockSoundness.ParseFacts tags c) (ho : c.outNet = c.unitNet)
    (hop : c.outPorts = #[("rin", 64), ("rout", 64)]) (hlo : c.leavesOut.getD 0 #[] = Array.range 128)
    (hw : ∀ k, HmOut.nIn c ≤ k → k < HmOut.nIn c + 2 → (c.ports.getD k default).words = 64)
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) {B : ℕ}
    (hB : (c.unit.outGroups.getD 0 default).col * Flock.WORD = B) (hE : B + 2048 ≤ c.unit.constPos)
    (w : Claims.Flock.Wit) (n : ℕ) {jo : ℕ} (hjo : jo < 2) {q : ℕ} (hq : HmOut.nIn c + jo = q) :
    ColsHold (Flock.Readable.copyOf p 0 h0 w (n / c.g) (n % c.g) 0) (B + 1024 * jo)
        ⟨(readForms w (DeployedR.outRowForms c q n)).toArray⟩ ∧
      (readForms w (DeployedR.outRowForms c q n)).length = 128 := by
  subst hq
  have hwj : (c.ports.getD (HmOut.nIn c + jo) default).words = 64 := hw _ (by omega) (by omega)
  have hjo' : jo < c.outPorts.size := by rw [hop]; exact hjo
  have hgrp : HmOut.outGroup c jo = (jo, 1, 64 * jo, 64) := by
    rw [FlockSoundness.Discharge.Hidden.outGroup_single hr hjo']
    unfold FlockSoundness.Discharge.Hidden.outAt
    rw [hop]
    rcases (by omega : jo = 0 ∨ jo = 1) with rfl | rfl <;> rfl
  have hlen : (DeployedR.outRowForms c (HmOut.nIn c + jo) n).length = 1024 := by
    simp only [DeployedR.outRowForms, List.length_map, List.length_range, hwj]
  have hf : ∀ t (ht : t < (DeployedR.outRowForms c (HmOut.nIn c + jo) n).length),
      (DeployedR.outRowForms c (HmOut.nIn c + jo) n)[t] = HmNets.Lin.var (n / c.g * 2 ^ c.kLog +
        (HmOut.outCol c (n % c.g) (64 * jo + t / 16) + t % 16)) := fun t ht => by
    rw [hlen] at ht
    simp only [DeployedR.outRowForms, List.getElem_map, List.getElem_range, Nat.add_sub_cancel_left, hgrp]
    simp only [show t / 16 < 64 by omega, ↓reduceIte]
  have hcol : ∀ k < 128, HmOut.outCol c (n % c.g) k = c.unitRange.slot (n % c.g * c.upv) + (B + 16 * k) := by
    intro k hk
    have h := outCol_unit P ho (n % c.g) (j := k) (by rw [hlo, Array.size_range]; exact hk)
    simp only [hlo, Array.getElem_range] at h
    rw [h, hB]
  have hR : readForms w (DeployedR.outRowForms c (HmOut.nIn c + jo) n) =
      Claims.Recursion.VStar.bytesOf ((DeployedR.outRowForms c (HmOut.nIn c + jo) n).map (formBit w)) := rfl
  rw [hR]
  refine ⟨colsHold_bytesOf (by rw [List.length_map, hlen]) fun t ht => ?_, by
    rw [length_bytesOf, List.length_map, hlen]⟩
  have ht' : t < 1024 := by simpa only [List.length_map, hlen] using ht
  have hj : B + 1024 * jo + t ≤ p.sts[0].c.unit.constPos := by rw [hc0]; omega
  have e : ∀ S : ℕ, S + (B + 16 * (64 * jo + t / 16)) + t % 16 = S + (B + 1024 * jo + t) := fun S => by omega
  rw [List.getElem_map, hf t (by rw [hlen]; exact ht'), formBit_var, hcol (64 * jo + t / 16) (by omega), e,
    Claims.Recursion.VStar.copyOf_eq hj]
  simp only [Claims.Recursion.VStar.copyIdx, hc0, h1, Nat.mul_one, Nat.add_zero]

/-- **An output slot's row, or a collision among its messages** (`outMsgs`): `linkOut_bind`'s three cases at port
`nm`'s index and the program `c`'s META names. -/
theorem out_or_coll {c : Circuit} {w : Claims.Flock.Wit} {n q : ℕ} {nm value prog : String} {E : Array ℕ}
    (hq : portIdx c nm = q) (hp : progOf c = prog)
    (h : rowPfx c q ++ readForms w (DeployedR.outRowForms c q n) =
        ResLower.rowInput (RecCoins.padTo E (Registered.rowWords E.size 16)) ∨
      Claims.Collision FlockDeploy.sha512 (rowPfx c q ++ readForms w (DeployedR.outRowForms c q n))
        (ResLower.rowInput (RecCoins.padTo E (Registered.rowWords E.size 16))) ∨
      Claims.Collision FlockDeploy.sha512 (FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n))
        (FlockSoundness.Refine.K512.saltPrefix ++ (RecCoins.verifierSalt prog value n).data.toList)) :
    rowPfx c q ++ readForms w (DeployedR.outRowForms c q n) =
        ResLower.rowInput (RecCoins.padTo E (Registered.rowWords E.size 16)) ∨
      ∃ a ∈ outMsgs c w n nm value E, ∃ b ∈ outMsgs c w n nm value E, Claims.Collision FlockDeploy.sha512 a b := by
  subst hq hp
  rcases h with h | h | h
  · exact .inl h
  · exact .inr ⟨_, by simp [outMsgs], _, by simp [outMsgs], h⟩
  · exact .inr ⟨_, by simp [outMsgs], _, by simp [outMsgs], h⟩

end RefStep

/-- **A staged link part's unit gives the verifier's rows** (`LinkRefStaged`): instance `n`'s ports, read off copy
`n`'s input columns, pass the reference check (`linkRef_of_shape`), whose output the unit computes at its output
columns (`unit_out`), which are the `rin` and `rout` rows (`colsHold_linkOut`); check 5 holds those to `expected R`
and `expected W`, or two of the slots' messages collide (`out_or_coll`). -/
theorem linkRef_staged : LinkRefStaged := by
  unfold LinkRefStaged
  intro st s x xO w hS hck r hr ss hss UB PB N hsv c hpc i spec sts hsu h0 own hown hidx hNn hnn rowsR rowsW R W hR hW
    wi wo hrin hrout n hn
  -- the part's one table: circuit `c`, V*'s `RecLink{UB, PB}` laid out at `l`
  obtain ⟨p, _, _, hx, ht, hpi, _, -, h0p, -⟩ := stagedTables hS
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  obtain ⟨h1, hc0, ⟨tags, P⟩, hname, ho, ⟨u, hu, hupv1, hl, hrowsV⟩, hv2, -, -, -⟩ :=
    stagedLink hS hx h0p hr hss hsv hpc
  have hsc : sts[0].c = c := by rw [← e0]; exact hc0
  have hnm : sts[0].c.unit.name = VStar.linkName UB PB := by rw [hsc]; exact hname
  obtain ⟨-, hnet, hok⟩ := linkCheck_staged hsu h0 hnm
  rw [hsc] at hnet hok
  obtain ⟨l, hlaid, hnl⟩ := RefStep.unitNet_laid hu hnet
  -- copy `n` holds the unit's laid-out rows
  obtain ⟨L, hz⟩ := layout_of_check_regs hck hx ht hpi h0p
  have hfl := Flock.Readable.flat_of_check hck hx ht hpi h0p
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨_, hset0⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset0
  rw [hpi] at hset0
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset0
  obtain ⟨_, hd, -⟩ := hall 0 h0p
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0p
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  have hupv0 : 0 < p.sts[0].c.upv := by rw [hc0]; omega
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd hupv0 hv1 L _ hz
  have hnN : n < p.sts[0].n := by rw [e0]; omega
  have hdl := div_lt_blocks hg0 hnN hnbl
  rw [hc0] at hdl hg0
  have hob : n / c.g < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) := by
    rw [L.m, Nat.add_sub_cancel_left]
    exact hdl
  have hgb : n % c.g < p.sts[0].c.g := by rw [hc0]; exact Nat.mod_lt _ hg0
  have hnetp : Flock.HmNets.netOf (VStar.linkName UB PB) l = .ok p.sts[0].c.unit := by rw [hc0]; exact hnl
  have hLR := Flock.Readable.laidRows_of_satisfies L hfl _ hz hob hgb hupv0 hnetp
  rw [FlockVBridge.OfVStar.inWords_laid hlaid hnetp] at hLR
  -- the seven row ports; the first five are the unit's input words, in order from column 0
  have hrows7 := hrowsV
  rw [linkRows_eq hu] at hrows7
  obtain ⟨hs0, hq0, hw0, hb0, hx0⟩ :=
    RefStep.port_facts hrows7 (k := 0) (nm := "trow") (wd := 64) (bt := 1024) (by rfl) (by rfl)
  obtain ⟨hs1, hq1, hw1, hb1, hx1⟩ :=
    RefStep.port_facts hrows7 (k := 1) (nm := "xyr") (wd := 128) (bt := 2048) (by rfl) (by rfl)
  obtain ⟨hs2, hq2, hw2, hb2, hx2⟩ := RefStep.port_facts hrows7 (k := 2) (nm := "raux")
    (wd := VStar.auxWords (UB + PB)) (bt := 16 * VStar.auxWords (UB + PB)) (by rfl) (by rfl)
  obtain ⟨hs3, hq3, hw3, hb3, hx3⟩ :=
    RefStep.port_facts hrows7 (k := 3) (nm := "xyw") (wd := 128) (bt := 2048) (by rfl) (by rfl)
  obtain ⟨hs4, hq4, hw4, hb4, hx4⟩ := RefStep.port_facts hrows7 (k := 4) (nm := "waux")
    (wd := VStar.auxWords (UB + PB)) (bt := 16 * VStar.auxWords (UB + PB)) (by rfl) (by rfl)
  obtain ⟨hs5, hq5, hw5, -, -⟩ :=
    RefStep.port_facts hrows7 (k := 5) (nm := "rin") (wd := 64) (bt := 1024) (by rfl) (by rfl)
  obtain ⟨hs6, hq6, hw6, -, -⟩ :=
    RefStep.port_facts hrows7 (k := 6) (nm := "rout") (wd := 64) (bt := 1024) (by rfl) (by rfl)
  have hP0 : (c.ports.extract 0 0).foldl (· + ·.words) 0 = 0 := by
    rw [hx0]
    simp
  have hP1 : (c.ports.extract 0 1).foldl (· + ·.words) 0 = 64 := by
    rw [hx1]
    simp
  have hP2 : (c.ports.extract 0 2).foldl (· + ·.words) 0 = 192 := by
    rw [hx2]
    simp
  have hP3 : (c.ports.extract 0 3).foldl (· + ·.words) 0 = 192 + VStar.auxWords (UB + PB) := by
    rw [hx3]
    simp
    omega
  have hP4 : (c.ports.extract 0 4).foldl (· + ·.words) 0 = 320 + VStar.auxWords (UB + PB) := by
    rw [hx4]
    simp
    omega
  have hIW : u.inWords = 320 + 2 * VStar.auxWords (UB + PB) := by
    obtain ⟨-, -, hin, -⟩ := FlockVBridge.RecLink.linkUnit_ok hu
    rw [VStar.UnitC.inWords, hin]
    simp only [VStar.linkRows, List.foldl_cons, List.foldl_nil, VStar.LINK_ROW_WORDS, VStar.LINK_X_WORDS,
      VStar.LINK_Y_WORDS] <;> omega
  have hN0 : 0 < u.inWords := by omega
  have hcol := RefStep.inCol_laid hlaid hnl
  obtain ⟨c0, l0⟩ := RefStep.inPort h1 h0p hc0 P hl hcol hN0 w n hq0 hs0 hP0 hw0 (by rw [hb0])
    (by omega) (base := 0) (by omega)
  obtain ⟨c1, l1⟩ := RefStep.inPort h1 h0p hc0 P hl hcol hN0 w n hq1 hs1 hP1 hw1 (by rw [hb1])
    (by omega) (base := 1024) (by omega)
  obtain ⟨c2, l2⟩ := RefStep.inPort h1 h0p hc0 P hl hcol hN0 w n hq2 hs2 hP2 hw2 (by rw [hb2])
    (by omega) (base := 3072) (by omega)
  obtain ⟨c3, l3⟩ := RefStep.inPort h1 h0p hc0 P hl hcol hN0 w n hq3 hs3 hP3 hw3 (by rw [hb3])
    (by omega) (base := 3072 + 16 * VStar.auxWords (UB + PB)) (by omega)
  obtain ⟨c4, l4⟩ := RefStep.inPort h1 h0p hc0 P hl hcol hN0 w n hq4 hs4 hP4 hw4 (by rw [hb4])
    (by omega) (base := 5120 + 16 * VStar.auxWords (UB + PB)) (by omega)
  -- the unit's output columns hold `linkOut`'s two rows
  obtain ⟨hO, hsz1, hsz2⟩ := RefStep.unit_out hu hlaid
    (v := Flock.Readable.copyOf p 0 h0p w (n / c.g) (n % c.g) 0) hLR l0 l1 l2 l3 l4 c0 c1 c2 c3 c4
  obtain ⟨hO1, hO2⟩ := RefStep.colsHold_split hO
  rw [hsz1] at hO2
  -- which are the `rin` and `rout` rows
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  have hU := FlockSoundness.unit_built P C
  unfold RecLinkPart.outsOk at hok
  rw [Bool.and_eq_true, Bool.and_eq_true] at hok
  obtain ⟨⟨hop, hallo⟩, -⟩ := hok
  have hop' : c.outPorts = #[("rin", 64), ("rout", 64)] := of_decide_eq_true hop
  have hlo0 : 0 < c.leavesOut.size := by
    rw [(FlockSoundness.Discharge.Layout.check_out P.typed ho hls).2.1]
    exact FlockSoundness.Discharge.Hm.upv_pos C ho
  have hlo : c.leavesOut.getD 0 #[] = Array.range 128 := by
    have h := Array.all_eq_true.1 hallo 0 hlo0
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlo0, Option.getD_some]
    first | exact of_decide_eq_true h | simpa using h
  have hh := ResLower.staged_hidden hS hx h0p
  have hor := Flock.Readable.outRowPorts_hidden hx ht hpi hh h0p (by rw [hc0]; exact hv2)
  rw [hc0] at hor
  have h7 : c.ports.size = 7 := by
    have hlen := congrArg List.length hrows7
    rw [List.length_map, Array.length_toList] at hlen
    exact hlen.trans rfl
  have hnr : c.outRowPorts.size = 2 := by rw [hor, Array.size_replicate, hop'] <;> rfl
  have hn5 : HmOut.nIn c = 5 := by unfold HmOut.nIn; omega
  have hw56 : ∀ k, HmOut.nIn c ≤ k → k < HmOut.nIn c + 2 → (c.ports.getD k default).words = 64 := by
    intro k hk1 hk2
    rw [hn5] at hk1 hk2
    rcases (by omega : k = 5 ∨ k = 6) with rfl | rfl
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs5, Option.getD_some]; exact hw5
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs6, Option.getD_some]; exact hw6
  have hnl' := hnl
  rw [FlockVBridge.OfVStar.laid_ok hlaid] at hnl'
  obtain ⟨hB, -, hE⟩ := FlockC3.RegLevels.outCols_of_laid (hU := hU) hnl'
  have hopU : u.outWords = 128 := by
    obtain ⟨-, -, -, hop2⟩ := FlockVBridge.RecLink.linkUnit_ok hu
    rw [VStar.UnitC.outWords, hop2] <;> rfl
  rw [FlockVBridge.OfVStar.foldl_replicate, show (FlockSoundness.Rows.ofNet hU).nIn +
    (FlockSoundness.Rows.ofNet hU).nComp = c.unit.constPos from FlockSoundness.Rows.ofNet_one hU] at hE
  have hE' : FlockSoundness.Discharge.Hm.outStart u.st (Array.replicate u.inWords 16)
      (Array.replicate u.outWords 16) + 2048 ≤ c.unit.constPos := by omega
  obtain ⟨hR0, hL0⟩ := RefStep.colsHold_linkOut h1 h0p hc0 P ho hop' hlo hw56 hor hB (by omega) w n
    (jo := 0) (by omega) (q := 5) (by omega)
  obtain ⟨hR1, hL1⟩ := RefStep.colsHold_linkOut h1 h0p hc0 P ho hop' hlo hw56 hor hB (by omega) w n
    (jo := 1) (by omega) (q := 6) (by omega)
  have hr0 := RefStep.bytes_of_cols hR0 hO1 (by omega) hL0 hsz1
  have hr1 := RefStep.bytes_of_cols hR1 hO2 (by omega) hL1 hsz2
  -- the slots commit those rows, and check 5 holds them to `expected R` and `expected W`
  have hout5 := MoreSlots.slotOut_stagedG hS hck hr hss hsv hpc hs5 (by omega) (.inl hn)
  have hout6 := MoreSlots.slotOut_stagedG hS hck hr hss hsv hpc hs6 (by omega) (.inl hn)
  obtain ⟨prog, hprog, hv5⟩ := linkValue_slot hS hck hr hpc hsu h0 hidx hown hrin hq5 hn hNn hnn
  obtain ⟨prog', hprog', hv6⟩ := linkValue_slot hS hck hr hpc hsu h0 hidx hown hrout hq6 hn hNn hnn
  have hpg : progOf c = prog := by unfold progOf; rw [← hsc, hprog] <;> rfl
  have hpg' : progOf c = prog' := by unfold progOf; rw [← hsc, hprog'] <;> rfl
  have hpi5 : portIdx c "rin" = 5 := by unfold portIdx; rw [hq5] <;> rfl
  have hpi6 : portIdx c "rout" = 6 := by unfold portIdx; rw [hq6] <;> rfl
  have hE5 := RefStep.size_expected (Flock.RecLinkBind.root_size hR)
  have hE6 := RefStep.size_expected (Flock.RecLinkBind.root_size hW)
  rcases RefStep.out_or_coll hpi5 hpg (linkOut_bind hout5 hv5) with hin5 | ⟨a, ha, b, hb, hcl⟩
  swap
  · exact .inr ⟨a, List.mem_append_left _ ha, b, List.mem_append_left _ hb, hcl⟩
  rcases RefStep.out_or_coll hpi6 hpg' (linkOut_bind hout6 hv6) with hin6 | ⟨a, ha, b, hb, hcl⟩
  swap
  · exact .inr ⟨a, List.mem_append_right _ ha, b, List.mem_append_right _ hb, hcl⟩
  rw [RefStep.padTo_64 hE5] at hin5
  rw [RefStep.padTo_64 hE6] at hin6
  have hrd5 := ResLower.readForms_of_rowInput (Array.getElem?_eq_getElem hs5) (by rw [hE5]; exact hw5) hin5
  have hrd6 := ResLower.readForms_of_rowInput (Array.getElem?_eq_getElem hs6) (by rw [hE6]; exact hw6) hin6
  have hlo1 := RefStep.eq_of_wordBytes (hr0.symm.trans hrd5) (RefStep.linkOut_u16 UB PB _ _ _ _ _ _ _).1
    (RefStep.expected_u16 R)
  have hlo2 := RefStep.eq_of_wordBytes (hr1.symm.trans hrd6) (RefStep.linkOut_u16 UB PB _ _ _ _ _ _ _).2
    (RefStep.expected_u16 W)
  -- the reference check accepts the ports and outputs those rows
  refine .inl ?_
  have L1 : ((inWordsAt c w "xyr" n).take 32).length = 32 := by rw [List.length_take, l1] <;> omega
  have L2 : ((inWordsAt c w "xyr" n).drop 32).length = 96 := by rw [List.length_drop, l1] <;> omega
  have L4 : ((inWordsAt c w "xyw" n).take 32).length = 32 := by rw [List.length_take, l3] <;> omega
  have L5 : ((inWordsAt c w "xyw" n).drop 32).length = 96 := by rw [List.length_drop, l3] <;> omega
  unfold linkRefAt
  rw [RefStep.linkRef_of_shape (RefStep.checkShape_of_unit hu) l0 L1 L2 (l2.trans (RefStep.auxWords_eq _).symm)
    L4 L5 (l4.trans (RefStep.auxWords_eq _).symm)
    (RefStep.ports_u16 (inBytesAt c w "trow" n) (inBytesAt c w "xyr" n) (inBytesAt c w "raux" n)
      (inBytesAt c w "xyw" n) (inBytesAt c w "waux" n))]
  exact congrArg Except.ok (Prod.ext hlo1 hlo2)

end RecDeploy.Links
