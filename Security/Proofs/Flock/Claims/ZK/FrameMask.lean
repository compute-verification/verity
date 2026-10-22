import Proofs.Flock.Claims.ZK.FrameParse

/-!
# What the frame releases of a round: the device's frames, masked

The frame releases each device round as the round's frames (`FlockSoundness.Refine.Frame`), domains dropped, with each
value absorbed in a masked phase sent as `v + h_i` under the next pad of the layout's absorbed order (`Masker.absorb`),
or as the pad alone in a stopped rep's algebraic phase. This file states that release at the frames `roundFree_encs`
folds over:

* **The absorbed order** (`absorbed_toList`, `absorbed_getElem!`): the frame masks under every pad below `h_ab` but
  `final_c`'s, ascending, so absorbed value `p` takes pad `p` up to `final_b`'s and pad `p + 1` after, the indices the
  masked prover pads its sent values under (`zerocheckPZk`, `lincheckPZk`: `r1ab i = i`, `r1c i = 64 + i`, …). Every
  such pad is among a rep's draws (`ofDraws_order`), and a rep's pad there is its draws' (`padAt_ofDraws`).
* **The release in a phase** (`maskFrames`, `maskVal`): the frames with each absorbed value through `maskVal` from the
  state's position on, or unchanged outside a masked phase; labels that neither end nor start a phase (`plainLabel`)
  go through as they came (`label_plain`).
* **The folds** (`foldlM_stepAt_on`, `foldlM_stepAt_off`) and **the rounds** (`roundFree_on`, `roundFree_off`): a round
  inside a masked phase, or outside one, leaves as its masked frames and its squeeze. **The zerocheck's first round**
  (`roundFree_zc`, `label_zc`): its label starts the algebraic phase and is followed by the pads' label and their root,
  as `zerocheckPZk` absorbs them.
* **An honest device's round** (`roundFree_honest`): the masked prover sends each value as `v' = v + h_i`, so the
  device's plain value is `v' + h_i` (`unmaskFrames`), and the frame's release of the device's round is the masked
  prover's round (`maskFrames_unmask`, characteristic 2) without its domain frames (`dropDoms`): a rep's first round
  carries its stream's domain, which the masked prover's ask sends and the frame never releases. Its pads are the
  masked prover's: the frame's pad at pad `j`'s absorbed position (`absPos`) is the draws' pad `j`
  (`padAt_ofDraws_absPos`), which on `tapeOfDraws` is the masked prover's (`padsOf_tapeOfDraws`).
-/

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header padded)
open _root_.Flock.Firewall.Frame
  (Cfg Masker absorbed nLig isLabel labelFrame bytesFrame roundFree LABEL_PCS LABEL_ZC SKIP)
open FlockSoundness.Refine (Frame encs colBytes)

/-! ## The absorbed order -/

theorem foldl_push_toList {α : Type} (f : ℕ → α) (L : List ℕ) (v : Array α) :
    (L.foldl (fun b a => b.push (f a)) v).toList = v.toList ++ L.map f := by
  induction L generalizing v with
  | nil => simp
  | cons a L ih => simp [List.foldl_cons]

theorem foldl_nest_toList {α : Type} (f : ℕ → ℕ → α) (L L' : List ℕ) (v : Array α) :
    (L.foldl (fun b j => L'.foldl (fun b t => b.push (f j t)) b) v).toList =
      v.toList ++ L.flatMap fun j => L'.map (f j) := by
  induction L generalizing v with
  | nil => simp
  | cons a L ih => rw [List.foldl_cons, ih, foldl_push_toList, List.flatMap_cons, List.append_assoc]

theorem foldl_push2_toList {α : Type} (f g : ℕ → α) (L : List ℕ) (v : Array α) :
    (L.foldl (fun b a => (b.push (f a)).push (g a)) v).toList = v.toList ++ L.flatMap fun a => [f a, g a] := by
  induction L generalizing v with
  | nil => simp
  | cons a L ih => simp [List.foldl_cons, ih]

theorem range'_map_add (b n : ℕ) : (List.range' 0 n).map (fun i => b + i) = List.range' b n := by
  rw [List.map_add_range']; simp

theorem range'_flatMap2 (b : ℕ) : ∀ n : ℕ,
    (List.range' 0 n).flatMap (fun a => [b + 2 * a, b + 2 * a + 1]) = List.range' b (2 * n)
  | 0 => rfl
  | n + 1 => by
    rw [List.range'_1_concat, List.flatMap_append, range'_flatMap2 b n, show 2 * (n + 1) = 2 * n + 2 by omega,
      ← List.range'_append_1]
    simp [List.range'_succ]

theorem range'_flatMap4 (b : ℕ) : ∀ n : ℕ,
    (List.range' 0 n).flatMap (fun j => (List.range' 0 4).map fun t => b + 4 * j + t) = List.range' b (4 * n)
  | 0 => rfl
  | n + 1 => by
    rw [show List.range' 0 (n + 1) = List.range' 0 n ++ [n] by simp [List.range'_1_concat], List.flatMap_append,
      range'_flatMap4 b n, show 4 * (n + 1) = 4 * n + 4 by omega, ← List.range'_append_1]
    simp [List.range'_succ]

theorem range'_join (s m n t : ℕ) (ht : s + m = t) (R : List ℕ) :
    List.range' s m ++ (List.range' t n ++ R) = List.range' s (m + n) ++ R := by
  subst ht; rw [← List.append_assoc, List.range'_append_1]

theorem range'_join' (s m n t : ℕ) (ht : s + m = t) :
    List.range' s m ++ List.range' t n = List.range' s (m + n) := by
  subst ht; rw [List.range'_append_1]

/-- The zerocheck's, lincheck's and `z_partial`'s pads in the frame's order, as ranges of pad indices. -/
theorem absorbed_piop (l : _root_.Flock.Zk.PadLayout) (R : List ℕ) :
    List.map l.r1ab (List.range' 0 64) ++
      (List.map l.r1c (List.range' 0 64) ++
        (List.flatMap (fun a => [l.ml a 0, l.ml a 1]) (List.range' 0 l.nMlv) ++
          l.fa :: l.fb :: (List.flatMap (fun a => [l.lc a 0, l.lc a 1]) (List.range' 0 l.nLc) ++
            (List.map l.zp (List.range' 0 64) ++ R)))) =
      List.range (l.fa + 2) ++ (List.range' (l.fa + 3) (2 * l.nLc + 64) ++ R) := by
  have h1 : List.map l.r1ab (List.range' 0 64) = List.range' 0 64 := by
    show List.map (fun i => i) (List.range' 0 64) = _
    simp
  have h2 : List.map l.r1c (List.range' 0 64) = List.range' 64 64 := range'_map_add 64 64
  have h3 : List.flatMap (fun a => [l.ml a 0, l.ml a 1]) (List.range' 0 l.nMlv) = List.range' 128 (2 * l.nMlv) :=
    range'_flatMap2 128 l.nMlv
  have h4 : List.flatMap (fun a => [l.lc a 0, l.lc a 1]) (List.range' 0 l.nLc) = List.range' (l.fa + 3) (2 * l.nLc) :=
    range'_flatMap2 (l.fa + 3) l.nLc
  have h5 : List.map l.zp (List.range' 0 64) = List.range' (l.fa + 3 + 2 * l.nLc) 64 :=
    range'_map_add (l.fa + 3 + 2 * l.nLc) 64
  have h6 : l.fa :: l.fb :: (List.range' (l.fa + 3) (2 * l.nLc) ++ (List.range' (l.fa + 3 + 2 * l.nLc) 64 ++ R)) =
      List.range' l.fa 2 ++ (List.range' (l.fa + 3) (2 * l.nLc + 64) ++ R) := by
    rw [← List.append_assoc, range'_join' _ _ _ _ rfl]
    simp [List.range'_succ, _root_.Flock.Zk.PadLayout.fb]
  rw [h1, h2, h3, h4, h5, h6, range'_join 128 (2 * l.nMlv) 2 l.fa rfl, range'_join 64 64 _ 128 rfl,
    range'_join 0 64 _ 64 rfl, List.range_eq_range']
  congr 2
  simp only [_root_.Flock.Zk.PadLayout.fa]
  omega

/-- **The frame's absorbed order is every pad below `h_ab` but `final_c`'s, ascending**: the zerocheck's `64 + 64`
skip values, its multilinear rounds and `final_a`, `final_b` (pads `0, …, fb`), then the lincheck's rounds,
`z_partial` and, with a masked level 0, its OOD values, lane-round messages and the pair's sums (pads `fc + 1, …`). -/
theorem absorbed_toList (l : _root_.Flock.Zk.PadLayout) :
    (absorbed l).toList = List.range (l.fa + 2) ++ List.range' (l.fa + 3) (2 * l.nLc + 64 + nLig l) := by
  unfold absorbed
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.forIn_pure_yield_eq_foldl,
    SKIP, pure_bind, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  by_cases hk : l.k0 > 0
  · simp only [hk, ↓reduceIte, Id.run_pure, Array.toList_append, foldl_push_toList, foldl_push2_toList,
      foldl_nest_toList, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    have hz : l.zp 64 = l.fa + 3 + (2 * l.nLc + 64) := by
      simp only [_root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc, _root_.Flock.Zk.PadLayout.fc]; omega
    have h7 : List.map l.lgOod (List.range' 0 l.nOod0) = List.range' (l.zp 64) l.nOod0 :=
      range'_map_add (l.zp 64) l.nOod0
    have h8 : List.flatMap (fun j => List.map (l.lgMsg j) (List.range' 0 4)) (List.range' 0 l.k0) =
        List.range' (l.zp 64 + l.nOod0) (4 * l.k0) := range'_flatMap4 (l.zp 64 + l.nOod0) l.k0
    have h9 : [l.lgE 0 0, l.lgE 0 1, l.lgE 1 0, l.lgE 1 1] = List.range' (l.zp 64 + l.nOod0 + 4 * l.k0) 4 := by
      simp only [_root_.Flock.Zk.PadLayout.lgE, List.range'_succ]
      simp
    rw [absorbed_piop, h7, h8, h9, range'_join' _ _ _ _ rfl, range'_join' _ _ _ _ rfl, hz,
      range'_join' _ _ _ _ rfl]
    simp only [nLig, hk, ↓reduceIte]
    congr 2
  · simp only [hk, ↓reduceIte, Id.run_pure, foldl_push_toList, foldl_push2_toList,
      Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rw [show List.map l.zp (List.range' 0 64) = List.map l.zp (List.range' 0 64) ++ [] by simp, absorbed_piop]
    simp [nLig, hk]

theorem absorbed_size (l : _root_.Flock.Zk.PadLayout) :
    (absorbed l).size = l.fa + 2 + (2 * l.nLc + 64 + nLig l) := by
  rw [← Array.length_toList, absorbed_toList]; simp

/-- **The pad of absorbed value `p`**: pad `p` up to `final_b`'s, then pad `p + 1`, `final_c`'s skipped. -/
theorem absorbed_getElem? (l : _root_.Flock.Zk.PadLayout) (p : ℕ) :
    (absorbed l)[p]? =
      if p < (absorbed l).size then some (if p < l.fa + 2 then p else p + 1) else none := by
  rw [absorbed_size, ← Array.getElem?_toList, absorbed_toList]
  by_cases h1 : p < l.fa + 2
  · rw [List.getElem?_append_left (by simpa using h1), List.getElem?_range h1]
    simp [h1]; omega
  · rw [List.getElem?_append_right (by simp; omega)]
    simp only [List.length_range, h1, ↓reduceIte]
    by_cases h2 : p < l.fa + 2 + (2 * l.nLc + 64 + nLig l)
    · rw [ite_eq_left h2, List.getElem?_range' (by omega)]
      congr 1; omega
    · rw [ite_eq_right h2]
      exact List.getElem?_eq_none (by simp; omega)

theorem absorbed_getElem! (l : _root_.Flock.Zk.PadLayout) {p : ℕ} (hp : p < (absorbed l).size) :
    (absorbed l)[p]! = if p < l.fa + 2 then p else p + 1 := by
  rw [getElem!_def, absorbed_getElem?, ite_eq_left hp]

/-- Every pad the frame absorbs under is one of the layout's. -/
theorem absorbed_lt_len (l : _root_.Flock.Zk.PadLayout) {p : ℕ} (hp : p < (absorbed l).size) :
    (absorbed l)[p]! < l.len := by
  rw [absorbed_getElem! l hp]
  rw [absorbed_size] at hp
  simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw, _root_.Flock.Zk.PadLayout.hab,
    _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc, _root_.Flock.Zk.PadLayout.fc, nLig] at hp ⊢
  split <;> split at hp <;> omega

/-- The draws' pads are the layout's `len`. -/
theorem ofBytes_h_size (l : _root_.Flock.Zk.PadLayout) (b : ByteArray) :
    (_root_.Flock.Firewall.Frame.Draws.ofBytes l b).h.size = l.len := by
  simp [_root_.Flock.Firewall.Frame.Draws.ofBytes, _root_.Flock.Firewall.Frame.words]

/-- **A rep's configuration masks under its draws' pads**: each pad of its order is among its draws'. -/
theorem ofDraws_order (l : _root_.Flock.Zk.PadLayout) (b : ByteArray)
    (m : Option _root_.Flock.Firewall.Frame.MaskSlot) :
    ∀ p, p < (Cfg.ofDraws l (_root_.Flock.Firewall.Frame.Draws.ofBytes l b) m).order.size →
      (Cfg.ofDraws l (_root_.Flock.Firewall.Frame.Draws.ofBytes l b) m).order[p]! <
        (Cfg.ofDraws l (_root_.Flock.Firewall.Frame.Draws.ofBytes l b) m).h.size := by
  intro p hp
  show (absorbed l)[p]! < (_root_.Flock.Firewall.Frame.Draws.ofBytes l b).h.size
  rw [ofBytes_h_size]
  exact absorbed_lt_len l hp

/-! ## The masked phase's release -/

/-- The pad of absorbed value `p`. -/
def padAt (c : Cfg) (p : ℕ) : F128 := c.h[c.order[p]!]!

/-- Absorbed value `p`, `v`, as the frame releases it: `v + h_i`, or the pad alone in a stopped rep's algebraic
phase. -/
def maskVal (c : Cfg) (st : Bool) (p : ℕ) (v : F128) : F128 :=
  (if (st && decide (p < c.nPiop)) = true then 0 else v) + padAt c p

theorem padAt_ofDraws (l : _root_.Flock.Zk.PadLayout) (d : _root_.Flock.Firewall.Frame.Draws)
    (m : Option _root_.Flock.Firewall.Frame.MaskSlot) {p : ℕ} (hp : p < (absorbed l).size) :
    padAt (Cfg.ofDraws l d m) p = d.h[if p < l.fa + 2 then p else p + 1]! := by
  show d.h[(absorbed l)[p]!]! = _
  rw [absorbed_getElem! l hp]

/-- The values `vs` from absorbed value `p` on, each through `g` at its position. -/
def maskVals (g : ℕ → F128 → F128) : ℕ → List F128 → List F128
  | _, [] => []
  | p, v :: vs => g p v :: maskVals g (p + 1) vs

theorem maskVals_length (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (vs : List F128), (maskVals g p vs).length = vs.length
  | _, [] => rfl
  | p, _ :: vs => by simp [maskVals, maskVals_length g (p + 1) vs]

/-- How many values the frames absorb. -/
def nAbs : List Frame → ℕ
  | [] => 0
  | .dom _ :: fs => nAbs fs
  | .label _ :: fs => nAbs fs
  | .bytes _ :: fs => nAbs fs
  | .one _ :: fs => nAbs fs + 1
  | .vec vs :: fs => nAbs fs + vs.length
  | .sq _ _ :: fs => nAbs fs

/-- **The frames as the frame releases them**, from absorbed value `p` on: domains dropped, and each absorbed value
through `g` at its position. -/
def maskFrames (g : ℕ → F128 → F128) : ℕ → List Frame → List Frame
  | _, [] => []
  | p, .dom _ :: fs => maskFrames g p fs
  | p, .label b :: fs => .label b :: maskFrames g p fs
  | p, .bytes b :: fs => .bytes b :: maskFrames g p fs
  | p, .one v :: fs => .one (g p v) :: maskFrames g (p + 1) fs
  | p, .vec vs :: fs => .vec (maskVals g p vs) :: maskFrames g (p + vs.length) fs
  | p, .sq k n :: fs => .sq k n :: maskFrames g p fs

/-- A label that neither ends nor starts a phase: none of the opening's, level 0's two, the inner proof's or the
zerocheck's. -/
def plainLabel (b : ByteArray) : Prop :=
  isLabel b LABEL_PCS = false ∧ isLabel b _root_.Flock.Zk.LABEL_CLOSE = false ∧
    isLabel b _root_.Flock.Zk.LABEL_INNER = false ∧ isLabel b LABEL_ZC = false ∧
    isLabel b _root_.Flock.Zk.LABEL_OPEN = false

/-- A frame inside a phase: a plain label, or any frame but a squeeze of coins. -/
def plainFrame : Frame → Prop
  | .label b => plainLabel b
  | .sq k n => k = 2 ∧ n = 0
  | _ => True

theorem label_plain (c : Cfg) (s : Masker) (b : ByteArray) (h : plainLabel b) :
    s.label c b = .ok (s, labelFrame b) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  simp [Masker.label, h1, h2, h3, h4, h5]
  rfl

theorem labelFrame_data (b : ByteArray) : (labelFrame b).data.toList = (Frame.label b).enc := by
  simp [labelFrame, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

theorem bytesFrame_data (b : ByteArray) : (bytesFrame b).data.toList = (Frame.bytes b).enc := by
  simp [bytesFrame, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

/-- **A masked value**: in a masked phase, at an absorbed position of the order, `v` leaves as `maskVal`. -/
theorem absorb_on (c : Cfg) (s : Masker) (v : F128) (hon : s.on = true) (hp : s.pos < c.order.size)
    (hc : c.order[s.pos]! < c.h.size) :
    s.absorb c v = .ok ({ s with pos := s.pos + 1 }, maskVal c s.stopped s.pos v) := by
  rw [getElem!_pos c.order s.pos hp] at hc
  simp [Masker.absorb, hon, Array.getElem?_eq_getElem hp, Array.getElem?_eq_getElem hc, maskVal, padAt,
    getElem!_pos c.order s.pos hp, getElem!_pos c.h c.order[s.pos] hc]
  rfl

theorem absorb_off (c : Cfg) (s : Masker) (v : F128) (hon : s.on = false) : s.absorb c v = .ok (s, v) := by
  simp [Masker.absorb, hon]
  rfl

theorem data_append' (a b : ByteArray) : (a ++ b).data.toList = a.data.toList ++ b.data.toList := by simp

theorem pushBytes_data (o : ByteArray) (v : F128) :
    (F128.pushBytes o v).data.toList = o.data.toList ++ v.toBytes.data.toList := by
  simp [F128.pushBytes]

theorem foldlM_absorb1_on (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) :
    ∀ (vs : List F128) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ),
      s.on = true → s.pos + vs.length ≤ c.order.size →
      ∃ r, vs.foldlM (absorb1 c) (s, o, fr, q) = .ok r ∧ r.1 = { s with pos := s.pos + vs.length } ∧
        r.2.1.data.toList = o.data.toList ++ colBytes (maskVals (maskVal c s.stopped) s.pos vs)
  | [], s, o, fr, q, _, _ => ⟨(s, o, fr, q), rfl, by simp, by simp [maskVals, colBytes]⟩
  | v :: vs, s, o, fr, q, hon, hp => by
    simp only [List.length_cons] at hp
    rw [List.foldlM_cons]
    have ha : absorb1 c (s, o, fr, q) v = .ok ({ s with pos := s.pos + 1 },
        F128.pushBytes o (maskVal c s.stopped s.pos v),
        if (s.on && decide (s.pos < c.nPiop)) = true then fr.push q else fr, q + 16) := by
      unfold absorb1
      rw [absorb_on c s v hon (by omega) (hc _ (by omega))]
      rfl
    rw [ha]
    obtain ⟨r, hr, h1, h2⟩ :=
      foldlM_absorb1_on c hc vs { s with pos := s.pos + 1 } _ _ (q + 16) hon (by simp; omega)
    refine ⟨r, hr, ?_, ?_⟩
    · rw [h1]; simp only [List.length_cons]; congr 1; omega
    · rw [h2, pushBytes_data]
      simp [maskVals, colBytes]

theorem foldlM_absorb1_off (c : Cfg) :
    ∀ (vs : List F128) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ), s.on = false →
      ∃ r, vs.foldlM (absorb1 c) (s, o, fr, q) = .ok r ∧ r.1 = s ∧
        r.2.1.data.toList = o.data.toList ++ colBytes vs
  | [], s, o, fr, q, _ => ⟨(s, o, fr, q), rfl, rfl, by simp [colBytes]⟩
  | v :: vs, s, o, fr, q, hon => by
    rw [List.foldlM_cons]
    have ha : absorb1 c (s, o, fr, q) v = .ok (s, F128.pushBytes o v,
        if (s.on && decide (s.pos < c.nPiop)) = true then fr.push q else fr, q + 16) := by
      unfold absorb1
      rw [absorb_off c s v hon]
      rfl
    rw [ha]
    obtain ⟨r, hr, h1, h2⟩ := foldlM_absorb1_off c vs s _ _ (q + 16) hon
    refine ⟨r, hr, h1, ?_⟩
    rw [h2, pushBytes_data]
    simp [colBytes]

theorem sq_enc (k : UInt8) (n : ℕ) : (Frame.sq k n).enc = (header 4 k n).data.toList := by
  simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

/-- **A masked phase's frames leave masked**: from a state in a masked phase, the fold of the frame's step over frames
inside the phase releases the frames with each absorbed value `v` as `maskVal`, in the order's positions from the
state's on. -/
theorem foldlM_stepAt_on (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) :
    ∀ (fs : List Frame) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ),
      s.on = true → (∀ f ∈ fs, plainFrame f) → s.pos + nAbs fs ≤ c.order.size →
      ∃ r, fs.foldlM (stepAt c) (s, o, fr, q) = .ok r ∧ r.1 = { s with pos := s.pos + nAbs fs } ∧
        r.2.1.data.toList = o.data.toList ++ encs (maskFrames (maskVal c s.stopped) s.pos fs)
  | [], s, o, fr, q, _, _, _ => ⟨(s, o, fr, q), rfl, by simp [nAbs], by simp [maskFrames, encs]⟩
  | f :: fs, s, o, fr, q, hon, hp, hb => by
    rw [List.foldlM_cons]
    have hp' : ∀ g ∈ fs, plainFrame g := fun g hg => hp g (List.mem_cons_of_mem _ hg)
    cases f with
    | dom b =>
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs s o fr _ hon hp' (by simpa [nAbs] using hb)
      exact ⟨r, hr, by rw [h1]; rfl, by rw [h2]; rfl⟩
    | label b =>
      have hl := label_plain c s b (hp _ List.mem_cons_self)
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs s (o ++ labelFrame b) fr (q + 16 + (b.size + 15) / 16 * 16)
        hon hp' (by simpa [nAbs] using hb)
      refine ⟨r, ?_, by rw [h1]; rfl, ?_⟩
      · simp only [stepAt, hl]
        exact hr
      · rw [h2, data_append', labelFrame_data]
        simp [maskFrames, encs]
    | bytes b =>
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs s (o ++ bytesFrame b) fr (q + 16 + (b.size + 15) / 16 * 16)
        hon hp' (by simpa [nAbs] using hb)
      refine ⟨r, hr, by rw [h1]; rfl, ?_⟩
      rw [h2, data_append', bytesFrame_data]
      simp [maskFrames, encs]
    | one v =>
      simp only [nAbs] at hb
      obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_absorb1_on c hc [v] s (o ++ header 3 1 1) fr (q + 16) hon
        (by simp; omega)
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ h2₁
      subst h1₁
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs { s with pos := s.pos + 1 } o₁ fr₁ q₁ hon hp'
        (by simp; omega)
      refine ⟨r, ?_, ?_, ?_⟩
      · simp only [stepAt]
        rw [hr₁]
        exact hr
      · rw [h1]; simp only [nAbs]; congr 1; omega
      · rw [h2, h2₁]
        simp [maskFrames, encs, maskVals, colBytes, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]
    | vec vs =>
      simp only [nAbs] at hb
      obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_absorb1_on c hc vs s (o ++ header 3 2 vs.length) fr (q + 16) hon
        (by omega)
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ h2₁
      subst h1₁
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs { s with pos := s.pos + vs.length } o₁ fr₁ q₁ hon hp'
        (by simp; omega)
      refine ⟨r, ?_, ?_, ?_⟩
      · simp only [stepAt]
        rw [hr₁]
        exact hr
      · rw [h1]; simp only [nAbs]; congr 1; omega
      · rw [h2, h2₁]
        simp [maskFrames, encs, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay, maskVals_length]
    | sq k n =>
      obtain ⟨rfl, rfl⟩ : k = 2 ∧ n = 0 := hp _ List.mem_cons_self
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs s (o ++ header 4 2 0) fr (q + 16) hon hp'
        (by simpa [nAbs] using hb)
      refine ⟨r, hr, by rw [h1]; rfl, ?_⟩
      rw [h2, data_append', ← sq_enc]
      simp [maskFrames, encs]

theorem maskVals_id : ∀ (p : ℕ) (vs : List F128), maskVals (fun _ v => v) p vs = vs
  | _, [] => rfl
  | p, v :: vs => by rw [maskVals, maskVals_id (p + 1) vs]

/-- **Outside a masked phase the frames leave as they came**, domains dropped. -/
theorem foldlM_stepAt_off (c : Cfg) :
    ∀ (fs : List Frame) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q p : ℕ),
      s.on = false → (∀ f ∈ fs, plainFrame f) →
      ∃ r, fs.foldlM (stepAt c) (s, o, fr, q) = .ok r ∧ r.1 = s ∧
        r.2.1.data.toList = o.data.toList ++ encs (maskFrames (fun _ v => v) p fs)
  | [], s, o, fr, q, _, _, _ => ⟨(s, o, fr, q), rfl, rfl, by simp [maskFrames, encs]⟩
  | f :: fs, s, o, fr, q, p, hon, hp => by
    rw [List.foldlM_cons]
    have hp' : ∀ g ∈ fs, plainFrame g := fun g hg => hp g (List.mem_cons_of_mem _ hg)
    cases f with
    | dom b =>
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs s o fr _ p hon hp'
      exact ⟨r, hr, h1, by rw [h2]; rfl⟩
    | label b =>
      have hl := label_plain c s b (hp _ List.mem_cons_self)
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs s (o ++ labelFrame b) fr (q + 16 + (b.size + 15) / 16 * 16)
        p hon hp'
      refine ⟨r, ?_, h1, ?_⟩
      · simp only [stepAt, hl]
        exact hr
      · rw [h2, data_append', labelFrame_data]
        simp [maskFrames, encs]
    | bytes b =>
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs s (o ++ bytesFrame b) fr (q + 16 + (b.size + 15) / 16 * 16)
        p hon hp'
      refine ⟨r, hr, h1, ?_⟩
      rw [h2, data_append', bytesFrame_data]
      simp [maskFrames, encs]
    | one v =>
      obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_absorb1_off c [v] s (o ++ header 3 1 1) fr (q + 16) hon
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ h2₁
      subst h1₁
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs m o₁ fr₁ q₁ (p + 1) hon hp'
      refine ⟨r, ?_, h1, ?_⟩
      · simp only [stepAt]
        rw [hr₁]
        exact hr
      · rw [h2, h2₁]
        simp [maskFrames, encs, colBytes, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]
    | vec vs =>
      obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_absorb1_off c vs s (o ++ header 3 2 vs.length) fr (q + 16) hon
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ h2₁
      subst h1₁
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs m o₁ fr₁ q₁ (p + vs.length) hon hp'
      refine ⟨r, ?_, h1, ?_⟩
      · simp only [stepAt]
        rw [hr₁]
        exact hr
      · rw [h2, h2₁]
        simp [maskFrames, encs, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay, maskVals_id]
    | sq k n =>
      obtain ⟨rfl, rfl⟩ : k = 2 ∧ n = 0 := hp _ List.mem_cons_self
      obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs s (o ++ header 4 2 0) fr (q + 16) p hon hp'
      refine ⟨r, hr, h1, ?_⟩
      rw [h2, data_append', ← sq_enc]
      simp [maskFrames, encs]

/-! ### No free values past the algebraic phase

`roundFree`'s free offsets are the values it absorbs in the algebraic phase. Outside a masked phase, and in a masked
phase past the algebraic phase's values, a round's frames add none. -/

theorem foldlM_absorb1_on_fr (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) :
    ∀ (vs : List F128) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ) (r : Masker × ByteArray × Array ℕ × ℕ),
      s.on = true → s.pos + vs.length ≤ c.order.size → c.nPiop ≤ s.pos →
      vs.foldlM (absorb1 c) (s, o, fr, q) = .ok r → r.2.2.1 = fr
  | [], s, o, fr, q, r, _, _, _, h => by cases h; rfl
  | v :: vs, s, o, fr, q, r, hon, hp, hpi, h => by
    simp only [List.length_cons] at hp
    rw [List.foldlM_cons] at h
    have ha : absorb1 c (s, o, fr, q) v = .ok ({ s with pos := s.pos + 1 },
        F128.pushBytes o (maskVal c s.stopped s.pos v),
        if (s.on && decide (s.pos < c.nPiop)) = true then fr.push q else fr, q + 16) := by
      unfold absorb1
      rw [absorb_on c s v hon (by omega) (hc _ (by omega))]
      rfl
    simp only [show decide (s.pos < c.nPiop) = false from decide_eq_false (by omega), Bool.and_false,
      Bool.false_eq_true, ↓reduceIte] at ha
    rw [ha] at h
    exact foldlM_absorb1_on_fr c hc vs { s with pos := s.pos + 1 } _ fr (q + 16) r hon
      (by show s.pos + 1 + vs.length ≤ _; omega) (by show c.nPiop ≤ s.pos + 1; omega) h

theorem foldlM_absorb1_off_fr (c : Cfg) :
    ∀ (vs : List F128) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ) (r : Masker × ByteArray × Array ℕ × ℕ),
      s.on = false → vs.foldlM (absorb1 c) (s, o, fr, q) = .ok r → r.2.2.1 = fr
  | [], s, o, fr, q, r, _, h => by cases h; rfl
  | v :: vs, s, o, fr, q, r, hon, h => by
    rw [List.foldlM_cons] at h
    have ha : absorb1 c (s, o, fr, q) v = .ok (s, F128.pushBytes o v,
        if (s.on && decide (s.pos < c.nPiop)) = true then fr.push q else fr, q + 16) := by
      unfold absorb1
      rw [absorb_off c s v hon]
      rfl
    simp only [hon, Bool.false_and, Bool.false_eq_true, ↓reduceIte] at ha
    rw [ha] at h
    exact foldlM_absorb1_off_fr c vs s _ fr (q + 16) r hon h

theorem foldlM_stepAt_on_fr (c : Cfg) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) :
    ∀ (fs : List Frame) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ) (r : Masker × ByteArray × Array ℕ × ℕ),
      s.on = true → (∀ f ∈ fs, plainFrame f) → s.pos + nAbs fs ≤ c.order.size → c.nPiop ≤ s.pos →
      fs.foldlM (stepAt c) (s, o, fr, q) = .ok r → r.2.2.1 = fr
  | [], s, o, fr, q, r, _, _, _, _, h => by cases h; rfl
  | f :: fs, s, o, fr, q, r, hon, hp, hb, hpi, h => by
    rw [List.foldlM_cons] at h
    have hp' : ∀ g ∈ fs, plainFrame g := fun g hg => hp g (List.mem_cons_of_mem _ hg)
    cases f with
    | dom b => exact foldlM_stepAt_on_fr c hc fs s o fr _ r hon hp' (by simpa [nAbs] using hb) hpi h
    | label b =>
      have hl := label_plain c s b (hp _ List.mem_cons_self)
      simp only [stepAt, hl] at h
      exact foldlM_stepAt_on_fr c hc fs s (o ++ labelFrame b) fr _ r hon hp' (by simpa [nAbs] using hb) hpi h
    | bytes b =>
      exact foldlM_stepAt_on_fr c hc fs s (o ++ bytesFrame b) fr _ r hon hp' (by simpa [nAbs] using hb) hpi h
    | one v =>
      simp only [nAbs] at hb
      obtain ⟨r₁, hr₁, h1₁, -⟩ := foldlM_absorb1_on c hc [v] s (o ++ header 3 1 1) fr (q + 16) hon
        (by simp; omega)
      have hf₁ := foldlM_absorb1_on_fr c hc [v] s (o ++ header 3 1 1) fr (q + 16) r₁ hon (by simp; omega) hpi hr₁
      simp only [stepAt] at h
      rw [hr₁] at h
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ hf₁
      subst h1₁ hf₁
      exact foldlM_stepAt_on_fr c hc fs { s with pos := s.pos + 1 } o₁ _ q₁ r hon hp' (by simp; omega)
        (by simp; omega) h
    | vec vs =>
      simp only [nAbs] at hb
      obtain ⟨r₁, hr₁, h1₁, -⟩ := foldlM_absorb1_on c hc vs s (o ++ header 3 2 vs.length) fr (q + 16) hon
        (by omega)
      have hf₁ := foldlM_absorb1_on_fr c hc vs s (o ++ header 3 2 vs.length) fr (q + 16) r₁ hon (by omega) hpi hr₁
      simp only [stepAt] at h
      rw [hr₁] at h
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ hf₁
      subst h1₁ hf₁
      exact foldlM_stepAt_on_fr c hc fs { s with pos := s.pos + vs.length } o₁ _ q₁ r hon hp' (by simp; omega)
        (by simp; omega) h
    | sq k n =>
      obtain ⟨rfl, rfl⟩ : k = 2 ∧ n = 0 := hp _ List.mem_cons_self
      exact foldlM_stepAt_on_fr c hc fs s (o ++ header 4 2 0) fr (q + 16) r hon hp' (by simpa [nAbs] using hb) hpi h

theorem foldlM_stepAt_off_fr (c : Cfg) :
    ∀ (fs : List Frame) (s : Masker) (o : ByteArray) (fr : Array ℕ) (q : ℕ) (r : Masker × ByteArray × Array ℕ × ℕ),
      s.on = false → (∀ f ∈ fs, plainFrame f) → fs.foldlM (stepAt c) (s, o, fr, q) = .ok r → r.2.2.1 = fr
  | [], s, o, fr, q, r, _, _, h => by cases h; rfl
  | f :: fs, s, o, fr, q, r, hon, hp, h => by
    rw [List.foldlM_cons] at h
    have hp' : ∀ g ∈ fs, plainFrame g := fun g hg => hp g (List.mem_cons_of_mem _ hg)
    cases f with
    | dom b => exact foldlM_stepAt_off_fr c fs s o fr _ r hon hp' h
    | label b =>
      have hl := label_plain c s b (hp _ List.mem_cons_self)
      simp only [stepAt, hl] at h
      exact foldlM_stepAt_off_fr c fs s (o ++ labelFrame b) fr _ r hon hp' h
    | bytes b => exact foldlM_stepAt_off_fr c fs s (o ++ bytesFrame b) fr _ r hon hp' h
    | one v =>
      obtain ⟨r₁, hr₁, h1₁, -⟩ := foldlM_absorb1_off c [v] s (o ++ header 3 1 1) fr (q + 16) hon
      have hf₁ := foldlM_absorb1_off_fr c [v] s (o ++ header 3 1 1) fr (q + 16) r₁ hon hr₁
      simp only [stepAt] at h
      rw [hr₁] at h
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ hf₁
      subst h1₁ hf₁
      exact foldlM_stepAt_off_fr c fs _ o₁ _ q₁ r hon hp' h
    | vec vs =>
      obtain ⟨r₁, hr₁, h1₁, -⟩ := foldlM_absorb1_off c vs s (o ++ header 3 2 vs.length) fr (q + 16) hon
      have hf₁ := foldlM_absorb1_off_fr c vs s (o ++ header 3 2 vs.length) fr (q + 16) r₁ hon hr₁
      simp only [stepAt] at h
      rw [hr₁] at h
      obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
      simp only at h1₁ hf₁
      subst h1₁ hf₁
      exact foldlM_stepAt_off_fr c fs _ o₁ _ q₁ r hon hp' h
    | sq k n =>
      obtain ⟨rfl, rfl⟩ : k = 2 ∧ n = 0 := hp _ List.mem_cons_self
      exact foldlM_stepAt_off_fr c fs s (o ++ header 4 2 0) fr (q + 16) r hon hp' h

/-- **The zerocheck's label starts the algebraic phase**: outside a masked phase and before the opening, the frame
releases it, then the pads' label and their root. -/
theorem label_zc (c : Cfg) (s : Masker) (hon : s.on = false) (hd : s.done = false) :
    s.label c LABEL_ZC.toUTF8 = .ok ({ s with on := true },
      labelFrame LABEL_ZC.toUTF8 ++ labelFrame _root_.Flock.Zk.LABEL_PADS.toUTF8 ++ bytesFrame c.root) := by
  simp [Masker.label, hon, hd]
  rfl

/-- **A round inside a masked phase**: at the bytes of frames `fs` inside the phase, then the squeeze of `n` coins of
kind `k`, the frame releases the frames with each absorbed value masked (`maskVal`) from the state's position on, then
the squeeze, and owes the squeeze's coins. -/
theorem roundFree_on (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = true) (hi : s.inner = none)
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hb : s.pos + nAbs fs ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, { s with pos := s.pos + nAbs fs, owed := some n }, fr) ∧
      out.data.toList = encs (maskFrames (maskVal c s.stopped) s.pos fs) ++ (Frame.sq k n).enc := by
  obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs s ByteArray.empty #[] 0 hon hp hb
  rw [roundFree_encs c s fs k n (by rw [hi]; simp) hm hk h0 hn, hr]
  obtain ⟨m, o, fr, q⟩ := r
  simp only at h1 h2
  subst h1
  refine ⟨o ++ header 4 k n, fr, ?_, ?_⟩
  · simp only [bind, Except.bind, hi]
    rfl
  · rw [data_append', h2, sq_enc]; simp

/-- **A round inside a masked phase past the algebraic phase's values carries no free value**: `roundFree_on`, with no
free offset. -/
theorem roundFree_on_fr (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = true) (hi : s.inner = none)
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hb : s.pos + nAbs fs ≤ c.order.size) (hpi : c.nPiop ≤ s.pos) :
    ∃ out, roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, { s with pos := s.pos + nAbs fs, owed := some n }, #[]) ∧
      out.data.toList = encs (maskFrames (maskVal c s.stopped) s.pos fs) ++ (Frame.sq k n).enc := by
  obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_on c hc fs s ByteArray.empty #[] 0 hon hp hb
  have h3 := foldlM_stepAt_on_fr c hc fs s ByteArray.empty #[] 0 r hon hp hb hpi hr
  rw [roundFree_encs c s fs k n (by rw [hi]; simp) hm hk h0 hn, hr]
  obtain ⟨m, o, fr, q⟩ := r
  simp only at h1 h2 h3
  subst h1 h3
  refine ⟨o ++ header 4 k n, ?_, ?_⟩
  · simp only [bind, Except.bind, hi]
    rfl
  · rw [data_append', h2, sq_enc]; simp

/-- **A round outside a masked phase**: the frames leave as they came, domains dropped, then the squeeze. -/
theorem roundFree_off (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hon : s.on = false) (hi : s.inner = none)
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, { s with owed := some n }, fr) ∧
      out.data.toList = encs (maskFrames (fun _ v => v) 0 fs) ++ (Frame.sq k n).enc := by
  obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs s ByteArray.empty #[] 0 0 hon hp
  rw [roundFree_encs c s fs k n (by rw [hi]; simp) hm hk h0 hn, hr]
  obtain ⟨m, o, fr, q⟩ := r
  simp only at h1 h2
  subst h1
  refine ⟨o ++ header 4 k n, fr, ?_, ?_⟩
  · simp only [bind, Except.bind, hi]
    rfl
  · rw [data_append', h2, sq_enc]; simp

/-- **The zerocheck's first round**: at frames `pre` outside a masked phase, the zerocheck's label, then frames `post`
inside the algebraic phase and the squeeze, the frame releases `pre` as they came, the zerocheck's label followed by
the pads' label and root, `post` with each absorbed value masked from the state's position on, then the squeeze. -/
theorem roundFree_zc (c : Cfg) (s : Masker) (pre post : List Frame) (k : UInt8) (n : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = false) (hd : s.done = false)
    (hi : s.inner = none) (hm : ∀ f ∈ pre ++ Frame.label LABEL_ZC.toUTF8 :: post, midFrame f)
    (hpre : ∀ f ∈ pre, plainFrame f) (hpost : ∀ f ∈ post, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hb : s.pos + nAbs post ≤ c.order.size) :
    ∃ out fr, roundFree c s
        (ByteArray.mk (encs (pre ++ Frame.label LABEL_ZC.toUTF8 :: post) ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, { s with on := true, pos := s.pos + nAbs post, owed := some n }, fr) ∧
      out.data.toList = encs (maskFrames (fun _ v => v) 0 pre) ++
        encs [.label LABEL_ZC.toUTF8, .label _root_.Flock.Zk.LABEL_PADS.toUTF8, .bytes c.root] ++
        encs (maskFrames (maskVal c s.stopped) s.pos post) ++ (Frame.sq k n).enc := by
  obtain ⟨r₁, hr₁, h1₁, h2₁⟩ := foldlM_stepAt_off c pre s ByteArray.empty #[] 0 0 hon hpre
  obtain ⟨m, o₁, fr₁, q₁⟩ := r₁
  simp only at h1₁ h2₁
  subst h1₁
  obtain ⟨r₂, hr₂, h1₂, h2₂⟩ := foldlM_stepAt_on c hc post { m with on := true }
    (o₁ ++ (labelFrame LABEL_ZC.toUTF8 ++ labelFrame _root_.Flock.Zk.LABEL_PADS.toUTF8 ++ bytesFrame c.root)) fr₁
    (q₁ + 16 + (LABEL_ZC.toUTF8.size + 15) / 16 * 16) rfl hpost hb
  have h3 : (Frame.label LABEL_ZC.toUTF8 :: post).foldlM (stepAt c) (m, o₁, fr₁, q₁) = .ok r₂ := by
    rw [List.foldlM_cons]
    simp only [stepAt, label_zc c m hon hd]
    exact hr₂
  have hf : (pre ++ Frame.label LABEL_ZC.toUTF8 :: post).foldlM (stepAt c) (m, ByteArray.empty, #[], 0) = .ok r₂ := by
    rw [List.foldlM_append, hr₁]
    exact h3
  rw [roundFree_encs c m _ k n (by rw [hi]; simp) hm hk h0 hn, hf]
  obtain ⟨m₂, o₂, fr₂, q₂⟩ := r₂
  simp only at h1₂ h2₂
  subst h1₂
  refine ⟨o₂ ++ header 4 k n, fr₂, ?_, ?_⟩
  · simp only [bind, Except.bind, hi]
    rfl
  · rw [data_append', h2₂, sq_enc, data_append', h2₁]
    simp [labelFrame_data, bytesFrame_data, encs]

/-! ## An honest device's round -/

/-- **A pad added twice cancels** (GF(2^128) has characteristic 2). -/
theorem add_pad_pad (v x : F128) : v + x + x = v := by
  obtain ⟨a, b⟩ := v
  obtain ⟨c, d⟩ := x
  show F128.mk (a ^^^ c ^^^ c) (b ^^^ d ^^^ d) = F128.mk a b
  rw [UInt64.xor_assoc, UInt64.xor_self, UInt64.xor_zero, UInt64.xor_assoc, UInt64.xor_self, UInt64.xor_zero]

/-- The frames but their domains: what the frame releases of them (the domain frame is the stream's own). -/
def dropDoms : List Frame → List Frame
  | [] => []
  | .dom _ :: fs => dropDoms fs
  | f :: fs => f :: dropDoms fs

/-- **The device's frames of a masked prover's round** `F`: each value of the masked phase with its pad added back, the
plain value the masked prover padded (`zerocheckPZk`'s `v' = v + h_i`, so `v = v' + h_i`); every other frame, domains
included, as it is. -/
def unmaskFrames (c : Cfg) : ℕ → List Frame → List Frame
  | _, [] => []
  | p, .dom b :: fs => .dom b :: unmaskFrames c p fs
  | p, .label b :: fs => .label b :: unmaskFrames c p fs
  | p, .bytes b :: fs => .bytes b :: unmaskFrames c p fs
  | p, .one v :: fs => .one (v + padAt c p) :: unmaskFrames c (p + 1) fs
  | p, .vec vs :: fs => .vec (maskVals (fun q v => v + padAt c q) p vs) :: unmaskFrames c (p + vs.length) fs
  | p, .sq k n :: fs => .sq k n :: unmaskFrames c p fs

theorem maskVals_unmask (c : Cfg) : ∀ (p : ℕ) (vs : List F128),
    maskVals (maskVal c false) p (maskVals (fun q v => v + padAt c q) p vs) = vs
  | _, [] => rfl
  | p, v :: vs => by
    simp only [maskVals, maskVal, Bool.false_and, Bool.false_eq_true, ↓reduceIte, add_pad_pad,
      maskVals_unmask c (p + 1) vs]

/-- **Masking the device's frames gives the masked prover's back**, but its domains. -/
theorem maskFrames_unmask (c : Cfg) : ∀ (p : ℕ) (F : List Frame),
    maskFrames (maskVal c false) p (unmaskFrames c p F) = dropDoms F
  | _, [] => rfl
  | p, f :: F => by
    cases f with
    | dom b => exact maskFrames_unmask c p F
    | label b => exact congrArg (List.cons _) (maskFrames_unmask c p F)
    | bytes b => exact congrArg (List.cons _) (maskFrames_unmask c p F)
    | one v =>
      simp only [unmaskFrames, maskFrames, maskVal, Bool.false_and, Bool.false_eq_true, ↓reduceIte, add_pad_pad]
      exact congrArg (List.cons _) (maskFrames_unmask c (p + 1) F)
    | vec vs =>
      simp only [unmaskFrames, maskFrames, maskVals_length]
      rw [maskVals_unmask]
      exact congrArg (List.cons _) (maskFrames_unmask c (p + vs.length) F)
    | sq k n => exact congrArg (List.cons _) (maskFrames_unmask c p F)

theorem nAbs_unmask (c : Cfg) : ∀ (p : ℕ) (F : List Frame), nAbs (unmaskFrames c p F) = nAbs F
  | _, [] => rfl
  | p, f :: F => by
    cases f <;> simp only [unmaskFrames, nAbs, maskVals_length, nAbs_unmask c _ F]

theorem unmask_mid (c : Cfg) : ∀ (p : ℕ) (F : List Frame), (∀ f ∈ F, midFrame f) →
    ∀ f ∈ unmaskFrames c p F, midFrame f
  | _, [], _ => by simp [unmaskFrames]
  | p, f :: F, hm => by
    have hm' : ∀ g ∈ F, midFrame g := fun g hg => hm g (List.mem_cons_of_mem _ hg)
    have h0 := hm f List.mem_cons_self
    intro x hx
    cases f with
    | vec vs =>
      simp only [unmaskFrames, List.mem_cons] at hx
      rcases hx with rfl | hx
      · show (maskVals _ p vs).length < 2 ^ 64; rw [maskVals_length]; exact h0
      · exact unmask_mid c _ F hm' x hx
    | one v =>
      simp only [unmaskFrames, List.mem_cons] at hx
      rcases hx with rfl | hx
      · trivial
      · exact unmask_mid c _ F hm' x hx
    | dom b | label b | bytes b | sq k n =>
      simp only [unmaskFrames, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact h0
      · exact unmask_mid c _ F hm' x hx

theorem unmask_plain (c : Cfg) : ∀ (p : ℕ) (F : List Frame), (∀ f ∈ F, plainFrame f) →
    ∀ f ∈ unmaskFrames c p F, plainFrame f
  | _, [], _ => by simp [unmaskFrames]
  | p, f :: F, hm => by
    have hm' : ∀ g ∈ F, plainFrame g := fun g hg => hm g (List.mem_cons_of_mem _ hg)
    have h0 := hm f List.mem_cons_self
    intro x hx
    cases f with
    | vec vs =>
      simp only [unmaskFrames, List.mem_cons] at hx
      rcases hx with rfl | hx
      · trivial
      · exact unmask_plain c _ F hm' x hx
    | one v =>
      simp only [unmaskFrames, List.mem_cons] at hx
      rcases hx with rfl | hx
      · trivial
      · exact unmask_plain c _ F hm' x hx
    | dom b | label b | bytes b | sq k n =>
      simp only [unmaskFrames, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact h0
      · exact unmask_plain c _ F hm' x hx

/-- **An honest device's round inside the algebraic phase leaves as the masked prover's, but its domains**: where the
device sends the frames `F` of the masked prover's round with each value's pad added back (`unmaskFrames`, the plain
values), a rep that has not stopped releases `F` without its domain frames, then the squeeze. A rep's first round
carries its stream's domain frame (`Transcript.init`), which the masked prover's ask (`squeezeP`'s `framed`) sends and
the frame keeps for its stream. -/
theorem roundFree_honest (c : Cfg) (s : Masker) (F : List Frame) (k : UInt8) (n : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = true) (hst : s.stopped = false)
    (hi : s.inner = none) (hm : ∀ f ∈ F, midFrame f) (hp : ∀ f ∈ F, plainFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64)
    (hb : s.pos + nAbs F ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs (unmaskFrames c s.pos F) ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, { s with pos := s.pos + nAbs F, owed := some n }, fr) ∧
      out.data.toList = encs (dropDoms F) ++ (Frame.sq k n).enc := by
  have hN : nAbs (unmaskFrames c s.pos F) = nAbs F := nAbs_unmask c _ F
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on c s (unmaskFrames c s.pos F) k n hc hon hi
    (unmask_mid c _ F hm) (unmask_plain c _ F hp) hk h0 hn (by rw [hN]; exact hb)
  rw [hN] at hr
  refine ⟨out, fr, hr, ?_⟩
  rw [ho, hst, maskFrames_unmask c s.pos F]

/-! ## The masked prover's pads in the frame's order -/

/-- The absorbed position of pad `j`: `j` up to `final_b`'s, then `j − 1` (`final_c`'s is never absorbed). -/
def absPos (l : _root_.Flock.Zk.PadLayout) (j : ℕ) : ℕ := if j < l.fa + 2 then j else j - 1

/-- **The frame's pad at pad `j`'s absorbed position is the draws' pad `j`**, for each pad the frame absorbs under:
any below `h_ab` but `final_c`'s. So the pad the masked prover adds to a sent value at its layout index (`h[lay.r1ab
i]`, `h[lay.lc j t]`, …) is the one the frame adds at that value's place in the absorbed order. -/
theorem padAt_ofDraws_absPos (l : _root_.Flock.Zk.PadLayout) (d : _root_.Flock.Firewall.Frame.Draws)
    (m : Option _root_.Flock.Firewall.Frame.MaskSlot) {j : ℕ} (hj : j ≠ l.fc)
    (hlt : j < l.fa + 3 + (2 * l.nLc + 64 + nLig l)) :
    absPos l j < (absorbed l).size ∧ padAt (Cfg.ofDraws l d m) (absPos l j) = d.h[j]! := by
  have hfc : l.fc = l.fa + 2 := rfl
  have hs : absPos l j < (absorbed l).size := by
    rw [absorbed_size]; unfold absPos; split <;> omega
  refine ⟨hs, ?_⟩
  rw [padAt_ofDraws l d m hs]
  unfold absPos
  by_cases h : j < l.fa + 2
  · simp [h]
  · rw [ite_eq_right h, ite_eq_right (show ¬ j - 1 < l.fa + 2 by omega)]
    congr 1; omega

end Claims.ZKServer
