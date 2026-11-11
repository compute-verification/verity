module

public import Proofs.Flock.SessionLoop.SchedAt

/-!
# A rep's PIOP round past its first, in frames

At a live round of rep stream `2j + r`'s PIOP past its first round, the frame checks the worker's round against the
loop's own scheduled round, the PIOP's frames at zero values (`Sched.rounds_piop`), and the switched session sends the
PIOP's frames at its masked values (`Masked.runCut_piopBody`). This file shows the two are one round in frames
(`LeanRelease.RoundFrames`), so the frame releases the session's message:

* **Alike frames** (`alike_cases`, `plainFrames_of_alike`, `noDoms_of_alike`, `fvals_length_alike`): frames alike to
  ones inside a phase are inside it, and carry as many values. `eq_of_alike_fvals`: alike frames with no domain or
  bytes frame (`fixedFrame`) and the same values are the same.
* **The release's values** (`alike_maskFrames`, `fvals_maskFrames`, `maskVals_getElem?`): the release of frames is
  alike to them, and its values are theirs through the mask, each at its position.
* **The PIOP's frames** (`piopFr_alike`, `fvals_piopFr_getElem?`, `piopFr_eq_maskFrames`): a PIOP round's frames past
  round `0` are alike at any values, round `jj`'s value `k` is the masked value at the absorbed order's position
  `P + k`, `P` the values before it, and so the round's frames are the release of any alike frames whose values, through
  the mask at `P`, are the masked values there.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open _root_.Flock (V F128)
open FlockSoundness.Refine (Frame encs)

/-! ## Alike frames -/

theorem alike_cases : ∀ {f f' : Frame}, Alike f f' →
    (∃ b b', f = .dom b ∧ f' = .dom b') ∨ (∃ b, f = .label b ∧ f' = .label b) ∨
    (∃ b b', f = .bytes b ∧ f' = .bytes b') ∨ (∃ v v', f = .one v ∧ f' = .one v') ∨
    (∃ vs vs', f = .vec vs ∧ f' = .vec vs' ∧ vs.length = vs'.length) ∨ (∃ k n, f = .sq k n ∧ f' = .sq k n)
  | .dom _, .dom _, _ => .inl ⟨_, _, rfl, rfl⟩
  | .label _, .label _, h => by
    simp only [Alike] at h
    subst h
    exact .inr (.inl ⟨_, rfl, rfl⟩)
  | .bytes _, .bytes _, _ => .inr (.inr (.inl ⟨_, _, rfl, rfl⟩))
  | .one _, .one _, _ => .inr (.inr (.inr (.inl ⟨_, _, rfl, rfl⟩)))
  | .vec _, .vec _, h => .inr (.inr (.inr (.inr (.inl ⟨_, _, rfl, rfl, h⟩))))
  | .sq _ _, .sq _ _, h => by
    simp only [Alike] at h
    obtain ⟨rfl, rfl⟩ := h
    exact .inr (.inr (.inr (.inr (.inr ⟨_, _, rfl, rfl⟩))))

theorem plainFrame_of_alike {f f' : Frame} (h : Alike f f') (hp : plainFrame f) : plainFrame f' := by
  rcases alike_cases h with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
    ⟨vs, vs', rfl, rfl, -⟩ | ⟨k, n, rfl, rfl⟩
  · exact True.intro
  · exact hp
  · exact True.intro
  · exact True.intro
  · exact True.intro
  · exact hp

theorem noDom_of_alike {f f' : Frame} (h : Alike f f') (hn : noDom f) : noDom f' := by
  rcases alike_cases h with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
    ⟨vs, vs', rfl, rfl, -⟩ | ⟨k, n, rfl, rfl⟩
  · exact hn
  all_goals exact True.intro

theorem plainFrames_of_alike {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') (hp : ∀ f ∈ fs, plainFrame f) :
    ∀ f ∈ fs', plainFrame f := by
  induction h with
  | nil => intro f hf; cases hf
  | cons ha _ ih =>
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · exact plainFrame_of_alike ha (hp _ (List.mem_cons_self ..))
    · exact ih (fun g hg => hp g (List.mem_cons_of_mem _ hg)) f hf

theorem noDoms_of_alike {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') (hn : ∀ f ∈ fs, noDom f) :
    ∀ f ∈ fs', noDom f := by
  induction h with
  | nil => intro f hf; cases hf
  | cons ha _ ih =>
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · exact noDom_of_alike ha (hn _ (List.mem_cons_self ..))
    · exact ih (fun g hg => hn g (List.mem_cons_of_mem _ hg)) f hf

theorem fvals_length_alike {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') :
    (fvals fs').length = (fvals fs).length := by
  induction h with
  | nil => rfl
  | cons ha _ ih =>
    rcases alike_cases ha with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
      ⟨vs, vs', rfl, rfl, hl⟩ | ⟨k, n, rfl, rfl⟩ <;>
      (simp only [fvals, List.length_cons, List.length_append, ih] <;> omega)

/-- A frame its values fix among the frames alike to it: neither a domain frame nor a bytes frame. -/
def fixedFrame : Frame → Prop
  | .dom _ => False
  | .bytes _ => False
  | _ => True

/-- **Alike frames with the same values are the same**, where neither has a domain or bytes frame. -/
theorem eq_of_alike_fvals {A B : List Frame} (h : List.Forall₂ Alike A B) :
    (∀ f ∈ A, fixedFrame f) → fvals A = fvals B → A = B := by
  induction h with
  | nil => intro _ _; rfl
  | @cons a b l₁ l₂ ha hr ih =>
    intro hf hv
    have hfA : ∀ f ∈ l₁, fixedFrame f := fun g hg => hf g (List.mem_cons_of_mem _ hg)
    have ha' := hf a (List.mem_cons_self ..)
    rcases alike_cases ha with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
      ⟨vs, vs', rfl, rfl, hl⟩ | ⟨k, n, rfl, rfl⟩
    · exact (ha' : False).elim
    · simp only [fvals] at hv
      rw [ih hfA hv]
    · exact (ha' : False).elim
    · simp only [fvals, List.cons.injEq] at hv
      rw [hv.1, ih hfA hv.2]
    · simp only [fvals] at hv
      obtain ⟨h1, h2⟩ := List.append_inj hv hl
      rw [h1, ih hfA h2]
    · simp only [fvals] at hv
      rw [ih hfA hv]

/-! ## The release's values -/

theorem alike_maskFrames (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (G : List Frame), (∀ f ∈ G, noDom f) → List.Forall₂ Alike G (maskFrames g p G)
  | _, [], _ => .nil
  | p, f :: G, h => by
    have hG : ∀ f ∈ G, noDom f := fun y hy => h y (List.mem_cons_of_mem _ hy)
    cases f with
    | dom b => exact (h _ (List.mem_cons_self ..) : False).elim
    | label b =>
      simp only [maskFrames]
      exact .cons rfl (alike_maskFrames g p G hG)
    | bytes b =>
      simp only [maskFrames]
      exact .cons rfl (alike_maskFrames g p G hG)
    | one v =>
      simp only [maskFrames]
      exact .cons True.intro (alike_maskFrames g (p + 1) G hG)
    | vec vs =>
      simp only [maskFrames]
      exact .cons (maskVals_length g p vs).symm (alike_maskFrames g (p + vs.length) G hG)
    | sq k n =>
      simp only [maskFrames]
      exact .cons ⟨rfl, rfl⟩ (alike_maskFrames g p G hG)

theorem maskVals_append (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (a b : List F128), maskVals g p (a ++ b) = maskVals g p a ++ maskVals g (p + a.length) b
  | p, [], b => by simp [maskVals]
  | p, v :: a, b => by
    simp only [maskVals, List.cons_append, List.length_cons]
    rw [maskVals_append g (p + 1) a b, show p + 1 + a.length = p + (a.length + 1) by omega]

theorem fvals_maskFrames (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (G : List Frame), fvals (maskFrames g p G) = maskVals g p (fvals G)
  | _, [] => rfl
  | p, .dom _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .label _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .bytes _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .sq _ _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .one v :: G => by
    simp only [maskFrames, fvals, maskVals]
    rw [fvals_maskFrames g (p + 1) G]
  | p, .vec vs :: G => by
    simp only [maskFrames, fvals]
    rw [fvals_maskFrames g (p + vs.length) G, maskVals_append]

theorem maskVals_getElem? (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (vs : List F128) (i : ℕ), (maskVals g p vs)[i]? = vs[i]?.map (g (p + i))
  | _, [], _ => by simp [maskVals]
  | p, v :: vs, 0 => by simp [maskVals]
  | p, v :: vs, i + 1 => by
    simp only [maskVals, List.getElem?_cons_succ]
    rw [maskVals_getElem? g (p + 1) vs i, show p + 1 + i = p + (i + 1) by omega]

/-! ## The PIOP's frames -/

/-- A PIOP round's frames past round `0` are alike at any values and layout. -/
theorem piopFr_alike (m kLog : ℕ) {digest cap root digest' cap' root' : ByteArray}
    {lay lay' : _root_.Flock.Zk.PadLayout} (mk mk' : ℕ → F128) {jj : ℕ} (h0 : jj ≠ 0) :
    List.Forall₂ Alike (piopFr m kLog digest cap root lay mk jj) (piopFr m kLog digest' cap' root' lay' mk' jj) := by
  unfold piopFr
  rw [ite_eq_right h0, ite_eq_right h0]
  split_ifs <;> simp [Alike]

theorem piopFr_fixed {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout} {mk : ℕ → F128}
    {jj : ℕ} (h : jj ≠ 0) : ∀ f ∈ piopFr m kLog digest cap root lay mk jj, fixedFrame f := by
  intro f hf
  unfold piopFr at hf
  rw [ite_eq_right h] at hf
  split_ifs at hf <;> simp only [List.mem_cons, List.not_mem_nil, or_false] at hf <;>
    rcases hf with rfl | rfl | rfl <;> exact True.intro

theorem fvals_piopFr_length (m kLog : ℕ) {digest cap root digest' cap' root' : ByteArray}
    {lay lay' : _root_.Flock.Zk.PadLayout} (mk mk' : ℕ → F128) (jj : ℕ) :
    (fvals (piopFr m kLog digest cap root lay mk jj)).length =
      (fvals (piopFr m kLog digest' cap' root' lay' mk' jj)).length := by
  unfold piopFr
  split_ifs <;> simp [fvals]

theorem piopV_length_congr (m kLog : ℕ) {digest cap root digest' cap' root' : ByteArray}
    {lay lay' : _root_.Flock.Zk.PadLayout} (mk mk' : ℕ → F128) :
    ∀ a, (piopV m kLog digest cap root lay mk a).length = (piopV m kLog digest' cap' root' lay' mk' a).length
  | 0 => by simp [piopV]
  | a + 1 => by
    rw [piopV_succ, piopV_succ, List.length_append, List.length_append]
    exact congrArg₂ (· + ·) (piopV_length_congr m kLog mk mk' a) (fvals_piopFr_length m kLog mk mk' a)

theorem piopV_prefix (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    {a b : ℕ} (h : a ≤ b) :
    ∃ rest, piopV m kLog digest cap root lay mk b = piopV m kLog digest cap root lay mk a ++ rest := by
  induction b with
  | zero =>
    obtain rfl : a = 0 := by omega
    exact ⟨[], (List.append_nil _).symm⟩
  | succ b ih =>
    rcases Nat.eq_or_lt_of_le h with rfl | h'
    · exact ⟨[], (List.append_nil _).symm⟩
    · obtain ⟨rest, hr⟩ := ih (by omega)
      exact ⟨rest ++ fvals (piopFr m kLog digest cap root lay mk b), by rw [piopV_succ, hr, List.append_assoc]⟩

/-- **A PIOP round's value `k`** is the masked value at the absorbed order's pad of position `P + k`, `P` the values the
rounds before it carry. -/
theorem fvals_piopFr_getElem? {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout}
    {mk : ℕ → F128} (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hn : lay.nMlv = m - 6) (hl : lay.nLc = kLog - 6) {jj P k : ℕ}
    (hj : jj < piopNR m kLog) (hP : (piopV m kLog digest cap root lay mk jj).length = P)
    (hk' : k < (fvals (piopFr m kLog digest cap root lay mk jj)).length) :
    P + k < lay.fa + 2 + (2 * lay.nLc + 64) ∧
      (fvals (piopFr m kLog digest cap root lay mk jj))[k]? =
        some (mk (if P + k < lay.fa + 2 then P + k else P + k + 1)) := by
  obtain ⟨rest, hr⟩ := piopV_prefix m kLog digest cap root lay mk (show jj + 1 ≤ piopNR m kLog by omega)
  rw [piopV_all m kLog digest cap root lay mk hm hk hn hl, piopV_succ, List.append_assoc] at hr
  have hlen := congrArg List.length hr
  simp only [List.length_map, List.length_append, List.length_range, List.length_range', hP] at hlen
  have hlt : P + k < lay.fa + 2 + (2 * lay.nLc + 64) := by omega
  refine ⟨hlt, ?_⟩
  have e := congrArg (·[P + k]?) hr
  rw [List.getElem?_append_right (by omega), hP, Nat.add_sub_cancel_left, List.getElem?_append_left hk'] at e
  rw [← e, List.getElem?_map]
  split_ifs with h1
  · rw [List.getElem?_append_left (by simpa using h1), List.getElem?_range h1]
    rfl
  · rw [List.getElem?_append_right (by simp only [List.length_range]; omega), List.length_range,
      List.range'_eq_map_range, List.getElem?_map, List.getElem?_range (by omega)]
    exact congrArg (fun v => some (mk v)) (show lay.fa + 3 + (P + k - (lay.fa + 2)) = P + k + 1 by omega)

/-- **A PIOP round's frames are the release of alike frames** whose values, through the mask `g` from position `P` on,
are the masked values at the absorbed order's pads. -/
theorem piopFr_eq_maskFrames {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout}
    {mk : ℕ → F128} (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hn : lay.nMlv = m - 6) (hl : lay.nLc = kLog - 6) {jj P : ℕ}
    (h0 : jj ≠ 0) (hj : jj < piopNR m kLog) (hP : (piopV m kLog digest cap root lay mk jj).length = P)
    (g : ℕ → F128 → F128) {G : List Frame} (hG : List.Forall₂ Alike (piopFr m kLog digest cap root lay mk jj) G)
    (hv : ∀ k (hkG : k < (fvals G).length),
      mk (if P + k < lay.fa + 2 then P + k else P + k + 1) = g (P + k) ((fvals G)[k]'hkG)) :
    piopFr m kLog digest cap root lay mk jj = maskFrames g P G := by
  have hnd := noDoms_of_alike hG (piopFr_noDom h0)
  have hA := Sched.forall₂_alike_trans hG (alike_maskFrames g P G hnd)
  refine eq_of_alike_fvals hA (piopFr_fixed h0) ?_
  rw [fvals_maskFrames]
  apply List.ext_getElem?
  intro k
  rw [maskVals_getElem?]
  have hlen := fvals_length_alike hG
  rcases Nat.lt_or_ge k (fvals G).length with hk' | hk'
  · obtain ⟨-, e⟩ := fvals_piopFr_getElem? hm hk hn hl hj hP (k := k) (by omega)
    rw [e, List.getElem?_eq_getElem hk', Option.map_some, hv k hk']
  · rw [List.getElem?_eq_none hk', List.getElem?_eq_none (by omega)]
    rfl

theorem piopV_length_le {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout}
    {mk : ℕ → F128} (lay' : _root_.Flock.Zk.PadLayout) (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hn : lay'.nMlv = m - 6)
    (hl : lay'.nLc = kLog - 6) {a : ℕ} (ha : a ≤ piopNR m kLog) :
    (piopV m kLog digest cap root lay mk a).length ≤ lay'.fa + 2 + (2 * lay'.nLc + 64) := by
  rw [piopV_length_congr m kLog (digest' := digest) (cap' := cap) (root' := root) (lay' := lay') mk mk a]
  refine le_trans (piopV_mono m kLog digest cap root lay' mk ha) ?_
  rw [piopV_all m kLog digest cap root lay' mk hm hk hn hl]
  simp

/-- A PIOP round's squeeze past round `0` is one the frame takes: two words, or one word of one coin. -/
theorem piopKN_ok {m jj : ℕ} (hm : 14 ≤ m) (hm' : m < 2 ^ 64) (h0 : jj ≠ 0) :
    (piopK jj = 2 ∨ (piopK jj = 1 ∧ piopN m jj = 1)) ∧ ¬ (piopN m jj = 0 ∧ piopK jj = 2) ∧ piopN m jj < 2 ^ 64 := by
  unfold piopK piopN
  rw [ite_eq_right h0]
  by_cases h1 : jj = 1
  · rw [ite_eq_left h1, ite_eq_left (show jj ≤ 1 by omega)]
    exact ⟨.inl rfl, fun h => by have := h.1; omega, lt_of_le_of_lt (Nat.sub_le m 13) hm'⟩
  · rw [ite_eq_right h1, ite_eq_right (show ¬ jj ≤ 1 by omega)]
    exact ⟨.inr ⟨rfl, rfl⟩, fun h => absurd h.1 one_ne_zero, by norm_num⟩

/-! ## The round in frames -/

section RoundFramesPiop

open Claims Claims.ZKMal
open _root_.Flock (SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **The switched session's PIOP round `jj ≥ 1` of rep stream `2j + r`**: where the coins show no refusal by it or by
the rep's first round, the session sends the PIOP's frames at the switched source's masked values, at some binding, and
its squeeze (`Masked.runCut_piopBody`: the views' level-0 commitments and the pads commitment return quietly). -/
theorem enc_piop (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (C : Fin (refK x) → finS.Coin) (j : Fin (Masked.jOf x)) (r : Fin 2) {jj : ℕ} (h0 : jj ≠ 0)
    (hjj : jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog)
    (hN : ¬ Masked.Refused x C (Masked.repBase x (2 * j.val + r.val) + jj))
    (hN0 : ¬ Masked.Refused x C (Masked.repBase x (2 * j.val + r.val))) :
    ∃ dg cp rt : ByteArray, finS.enc x C (Masked.repBase x (2 * j.val + r.val) + jj)
        (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) =
      0 :: (encs (piopFr (Masked.stAt x j).m (Masked.stAt x j).c.kLog dg cp rt (Masked.layE x j)
        ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val).masked jj) ++
        (Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc) := by
  have hj := Masked.lt_of_jOf j
  obtain ⟨schs, hs, -⟩ := Masked.schedsPZk_live hx
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hms : ∀ i < (FlockDeploy.tabsOf (Masked.stsOf x)).size,
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (Masked.stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  let cmOf : ℕ → FlockComplete.Commit := fun i =>
    Classical.choose (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
  have hcm : ∀ i, (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
      _root_.Flock.Hashes V) _root_.Flock.MerkleScheme.hm96Sha512 (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i) : V _) = .ok (cmOf i) := fun i =>
    (Classical.choose_spec (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))).1
  have hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)).go l =
      (.ok ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf), l) := by
    intro l
    have hq : FlockComplete.quietS.R
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
              _root_.Flock.Hashes FlockComplete.PS)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)) :=
      _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
        FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
    rw [hq l, FlockComplete.mapM_ok_mem cmOf _ fun i hi => ?_]
    rw [hms i (by simpa using hi)]
    exact hcm i
  obtain ⟨extras, hext⟩ := Masked.extra_ok hx hsx C hj
  obtain ⟨ic, hic0⟩ := FlockComplete.padsTreeCut_ok
    (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
      _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
    ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val)
  have hrc := Masked.runCut_piopBody hx C (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t) hsx hs hcms j r hN0 hext (ic := ic)
    (fun l => by rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic0]) (Masked.tabsOf_stAt x j) (jj := jj) hjj
  have hsrc : Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x C) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) :=
    rfl
  rw [enc_finS_of C (by omega) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) hsx hN
    (by rw [← hsrc]; exact hrc.1), ← hsrc, hrc.2]
  simp only [piopBody, h0, ↓reduceIte, List.nil_append]
  exact ⟨_, _, _, rfl⟩

/-- **`enc_piop` at the session's message after the openings `cs`**: `swMsgAt` sends round `cs.length`. -/
theorem swMsgAt_piop (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (cs : List finS.Coin) (hlt : cs.length < refK x) (j : Fin (Masked.jOf x)) (r : Fin 2) {jj : ℕ}
    (hn : cs.length = Masked.repBase x (2 * j.val + r.val) + jj) (h0 : jj ≠ 0)
    (hjj : jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog)
    (hN : ¬ Masked.Refused x (finS.pad (K := refK x) cs) (Masked.repBase x (2 * j.val + r.val) + jj))
    (hN0 : ¬ Masked.Refused x (finS.pad (K := refK x) cs) (Masked.repBase x (2 * j.val + r.val))) :
    ∃ dg cp rt : ByteArray, finS.swMsgAt x (finS.wt x w) (vs x w ν) t cs =
      0 :: (encs (piopFr (Masked.stAt x j).m (Masked.stAt x j).c.kLog dg cp rt (Masked.layE x j)
        ((Masked.swSrcE x (finS.pad (K := refK x) cs) w (vs x w ν (finS.pad (K := refK x) cs) (finS.seenOf x t)) t j.val).rep
          r.val).masked jj) ++
        (Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc) := by
  rw [swMsgAt_enc vs w ν t cs hlt, hn]
  exact enc_piop vs w hx hsx ν t (finS.pad (K := refK x) cs) j r h0 hjj hN hN0

end Loop

namespace LeanRelease

open Loop in
set_option maxHeartbeats 2000000 in
/-- **The frame at a rep's live PIOP round past its first**: at a live history whose next round is rep stream
`2j + r`'s PIOP round `jj ≥ 1`, the frame takes the round's coins at a state inside the masked phase, at the values
before round `jj`, and the worker's round is the PIOP's frames there at some values (`Alike`) and the round's squeeze
(`Frame.passes_frames` against the loop's own round, `Sched.rounds_piop`). -/
theorem piop_frame (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoop Masked.dataE L₁ real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Loop.Serves x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoop Masked.dataE L₁ real).toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)))
    (hlt : ops.length < refK x)
    (hl : (ofLoop Masked.dataE L₁ real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) {v : Msg}
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {jj : ℕ}
    (hn : ops.length = Masked.repBase x (2 * j.val + r.val) + jj) (h0 : jj ≠ 0)
    (hjj : jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog) :
    ∃ (m : _root_.Flock.Firewall.Frame.Masker) (G : List Frame),
      (ofLoop Masked.dataE L₁ real).answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m ∧
      m.on = true ∧ m.inner = none ∧
      m.pos = (piopV (Masked.stAt x j).m (Masked.stAt x j).c.kLog ByteArray.empty ByteArray.empty ByteArray.empty
        (Masked.layE x j) (fun _ => 0) jj).length ∧
      List.Forall₂ Alike (piopFr (Masked.stAt x j).m (Masked.stAt x j).c.kLog ByteArray.empty ByteArray.empty
        ByteArray.empty (Masked.layE x j) (fun _ => 0) jj) G ∧
      (∀ f ∈ G, midFrame f) ∧ (∀ f ∈ G, plainFrame f) ∧
      v = encs G ++ (Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc ∧
      (ofLoop Masked.dataE L₁ real).coins x (ops.length + 1) = piopN (Masked.stAt x j).m jj ∧
      (ofLoop Masked.dataE L₁ real).rep x (ops.length + 1) = 2 * j.val + r.val ∧
      ∃ lay mask, _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[j.val]! = .ok (lay, mask) := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjh : (2 * j.val + r.val) / 2 = j.val := by omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hR0 : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
      (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hst]
    have hA : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
        (Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6) := by
      unfold piopNR
      omega
    exact le_trans (le_trans (le_trans hA (Nat.le_add_right _ _)) (Nat.le_add_right _ _)) (Nat.le_add_right _ _)
  obtain ⟨a, rfl⟩ : ∃ a, jj = a + 1 := ⟨jj - 1, by omega⟩
  have hbase : Masked.repBase x (2 * j.val + r.val) = 1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨i, hp⟩ : ∃ i, ops.length = i + 2 := ⟨ops.length - 2, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1) = (2 * j.val + r.val, a + 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) (a + 1)
      (by rw [hrsl]; exact hq2) (lt_of_lt_of_le hjj hR0)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 = a + 1 := by rw [hloc]
  -- the frame's check
  have hlast : (ofLoop Masked.dataE L₁ real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ real) x
  have hk : (ofLoop Masked.dataE L₁ real).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, out, s', hA, hck⟩ :=
    check_of_live (ofLoop Masked.dataE L₁ real) wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hk hv
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ real x (ν, t) (m₀, ρ₀) ops hsx (q := i + 1) (by omega) hj1 ho ht] at hA'
  replace hA' := (bind_rankStop_ok hA').1
  -- the loop's placeholder table and its state there
  have hcs : (ops.map fun p => ZKMal.CoinLayer.rd p.2).length = i + 2 := by rw [List.length_map]; omega
  have ho' : (progOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun h => answer_owed_none _ h hA'
  obtain ⟨c', tb, dom, xs, his, hph0⟩ := progOf_placeholder x _ hsx hcs ho'
  rw [locate_pred _ 0 i a hj1, hloc] at hph0
  dsimp only at hph0
  have hph : placeholder spec (Masked.stsOf x) (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).1
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', tb, dom, xs, his) := by
    rw [hloc]
    exact hph0
  rw [progOf_at x _ hsx (q := i + 1) (by rw [List.length_map]; omega) hj1 hph, hloc] at hA'
  dsimp only at hA'
  obtain ⟨alg, sched, hal, hs⟩ := repState_ok hA'
  obtain ⟨hts', -, -⟩ := Count.placeholder_tab hph0
  rw [hst] at hts'
  have htm : tb.st.m = (Masked.stAt x j).m := by rw [hts']; rfl
  have htk' : tb.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hts']; rfl
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  obtain ⟨sch', tp', hsch', hlay⟩ := start_lay hds
  rw [hz] at hsch'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hsch'
  obtain ⟨rfl, rfl⟩ := hsch'
  rw [hst] at hlay
  have hlayS : lay = Masked.layE x j := hlay
  have hnM : lay.nMlv = tb.st.m - 6 := by
    rw [hlayS, htm]
    try rfl
  have hnL : lay.nLc = tb.st.fold.kLog - 6 := by
    rw [hlayS, htk']
    try rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hjjT : a + 1 < piopNR tb.st.m tb.st.fold.kLog := by rw [htm, htk']; exact hjj
  have hjjT' : a < piopNR tb.st.m tb.st.fold.kLog := by omega
  have hb : (piopV tb.st.m tb.st.fold.kLog tb.st.digest (_root_.Flock.Firewall.Finish.flat tb.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0) (fun _ => 0) (a + 1)).length ≤ c'.order.size := by
    rw [hc', Loop.ofDraws_order, absorbed_size]
    exact le_trans (piopV_length_le lay (by omega) (by omega) hnM hnL hjjT.le) (by omega)
  have hdg : tb.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tb.st.m < 2 ^ 64 := by rw [htm]; exact lt_of_le_of_lt hm35 (by norm_num)
  obtain ⟨-, hdom', -, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨hon, hin, -, hpos⟩ := repState_piop hcc hal hs (by omega) hmb hdg (placeholder_cap hph0)
    (hdm ▸ hdom') hjjT' hb hA'
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_live Masked.dataE L₁ real wk x w (ν, t) κ (m₀, ρ₀) ops hl hk
    hsx (q := i + 1) (by omega)
  have hre' : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, -, -⟩ := Count.real_tabAlike hRT hph0 hre'
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, hts']; rfl
  have htrk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1, hts']; rfl
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hmT' : tr.st.m < 2 ^ 64 := by rw [htrm]; exact lt_of_le_of_lt hm35 (by norm_num)
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨hsz', hpi'⟩ := Sched.rounds_piop hmT hmT' hal'
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal' hs'
  have hjjR : a + 1 < piopNR tr.st.m tr.st.fold.kLog := by rw [htrm, htrk]; exact hjj
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 < sched'.size := by
    rw [hloc, hrest, Array.size_append, hsz']
    dsimp only
    omega
  have hscheq : (sched'.getD (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2
      (.empty, 0)).1.data.toList =
      encs (piopFr tr.st.m tr.st.fold.kLog tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap) Sched.root0
        (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0) (fun _ => 0) (a + 1)) ++
        (Frame.sq (piopK (a + 1)) (piopN tr.st.m (a + 1))).enc := by
    rw [hloc]
    dsimp only
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left (by rw [hsz']; exact hjjR),
      hpi' (a + 1) (by omega) hjjR]
    simp only [Option.getD_some, List.toList_toArray]
  -- the loop's coin counts there
  have hpl := Count.piop_list (kLog := (Masked.stAt x j).c.kLog) (by omega : 14 ≤ (Masked.stAt x j).m)
  have hpL := congrArg List.length hpl
  simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil] at hpL
  have hNR : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      loopNr x (Masked.repBase x (2 * j.val + r.val) + jj) = piopN (Masked.stAt x j).m jj := by
    intro jj hjj'
    unfold piopNR at hjj'
    rw [Masked.repBase_eq hsx, loopNr_plan hsx hq2 hz (by rw [hst]; exact hk6) (i := jj)
      (by unfold FlockComplete.piopRoundsZk; rw [hst]; omega),
      Count.plan_get hz (by rw [List.length_append, hst, hpL]; omega), hst,
      List.getElem?_append_left (by rw [hpL]; omega), hpl]
    exact piopN_list (by omega) hk6 (by unfold piopNR; omega)
  have hcoins : (ofLoop Masked.dataE L₁ real).coins x ((m₀, ρ₀) :: ops).length = piopN tr.st.m (a + 1) := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hn, hNR (a + 1) hjj, htrm]
  rw [ofLoop_scheduled Masked.dataE L₁ real x w (ν, t) (m₀, ρ₀) ops _ hsx (q := i + 1) (by omega) hre hs' hlt',
    hscheq, hcoins] at hck
  -- the worker's round is the loop's round at some values
  have hF0 : ∀ f ∈ piopFr tr.st.m tr.st.fold.kLog tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap)
      Sched.root0 (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0) (fun _ => 0) (a + 1),
      midFrame f ∧ plainFrame f := by
    intro f hf
    rcases Nat.eq_zero_or_pos a with rfl | ha
    · simp [piopFr] at hf
    · obtain ⟨h1, h2⟩ := piopFr_frames _ _ _ _ _ _ _ (show 2 ≤ a + 1 by omega)
      exact ⟨h1 f hf, h2 f hf⟩
  have hKN := piopKN_ok (m := tr.st.m) (jj := a + 1) hmT hmT' (by omega)
  obtain ⟨G, hAG, hmidG, hvG⟩ := passes_frames _ _ _ _ _ (by simp [hin]) (fun f hf => (hF0 f hf).1) hKN.1
    hKN.2.1 hKN.2.2 _ (check_none_passes hck)
  have hv' : v = encs G ++ (Frame.sq (piopK (a + 1)) (piopN (Masked.stAt x j).m (a + 1))).enc := by
    have e := congrArg (fun b : ByteArray => b.data.toList) hvG
    simp only at e
    rw [e, htrm]
  refine ⟨m, G, hA, hon, hin, ?_, ?_, hmidG, plainFrames_of_alike hAG (fun f hf => (hF0 f hf).2), hv', ?_, ?_, lay,
    mask, by rw [hjh] at hds; exact hds⟩
  · rw [hpos, htm, htk']
    exact piopV_length_congr _ _ _ _ _
  · rw [htrm, htrk] at hAG
    exact Sched.forall₂_alike_trans (piopFr_alike (jj := a + 1) _ _ _ _ (by omega)) hAG
  · show loopNr x (ops.length + 1 - 1) = _
    rw [Nat.add_sub_cancel, hn, hNR (a + 1) hjj]
  · show repOf x (ops.length + 1) = _
    rw [show ops.length + 1 = i + 1 + 2 by omega, repOf_add_two hsx, hloc]

open Loop in
set_option maxHeartbeats 2000000 in
/-- **A rep's live PIOP round past its first is one round in frames** (`RoundFrames`), at `SessionLoopZKDeployed`'s
release and its premises on the statement and the loop's inputs, after a history at which the worker takes `vs`'s
values (`Loop.TakesValsAt`): the frame takes the round's coins inside the masked phase (`piop_frame`); the worker's
round is the PIOP's frames at the values it frees, which are `vs`'s; and the session sends the PIOP's frames at the
switched source's masked values (`Loop.swMsgAt_piop`, the coins showing no refusal there, `unrefused_prefix`), which
are `vs`'s values under the frame's pads (`Loop.swSrcE_masked_vsWord`, `Loop.RepSrc.pads`), so the release of the
worker's frames (`piopFr_eq_maskFrames`). -/
theorem roundFrames_piop (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {jj : ℕ}
    (hn : ops.length = Masked.repBase x (2 * j.val + r.val) + jj) (h0 : jj ≠ 0)
    (hjj : jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog) :
    (ofLoopLink L₀ src real).RoundFrames x (ν, t) ((m₀, ρ₀) :: ops) false v
      (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, -, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hSc hsx
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  obtain ⟨m, G, hA, hon, hin, hposm, hAG, hmid, hplain, hvG, hcoins, hrep, lay, mask, hds⟩ :=
    piop_frame (withFiles (withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops
      (by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hsx) hlt hl hv j r hn h0 hjj
  have hA' : (ofLoopLink L₀ src real).answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m := hA
  have hcoins' : (ofLoopLink L₀ src real).coins x ((m₀, ρ₀) :: ops).length = piopN (Masked.stAt x j).m jj :=
    hcoins
  have hcoins'' : (ofLoopLink L₀ src real).coins x (ops.length + 1) = piopN (Masked.stAt x j).m jj := hcoins
  have hrep' : (ofLoopLink L₀ src real).rep x ((m₀, ρ₀) :: ops).length = Loop.frepOf j.val r.val := hrep
  have hrep'' : (ofLoopLink L₀ src real).rep x (ops.length + 1) = 2 * j.val + r.val := hrep
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real j r hds
  -- the frame's configuration: its order and pads
  have hord : ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order =
      _root_.Flock.Firewall.Frame.absorbed (Masked.layE x j) := by
    show _root_.Flock.Firewall.Frame.absorbed ((ofLoopLink L₀ src real).lay x
      ((ofLoopLink L₀ src real).rep x ((m₀, ρ₀) :: ops).length)) = _
    rw [hrep', hlay]
  have hnP : ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop =
      (_root_.Flock.Firewall.Frame.absorbed (Masked.layE x j)).size -
        _root_.Flock.Firewall.Frame.nLig (Masked.layE x j) := by
    show (_root_.Flock.Firewall.Frame.absorbed ((ofLoopLink L₀ src real).lay x
      ((ofLoopLink L₀ src real).rep x ((m₀, ρ₀) :: ops).length))).size -
        _root_.Flock.Firewall.Frame.nLig ((ofLoopLink L₀ src real).lay x
          ((ofLoopLink L₀ src real).rep x ((m₀, ρ₀) :: ops).length)) = _
    rw [hrep', hlay]
  have hc : ∀ p, p < ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size →
      ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[p]! <
        ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.size :=
    Claims.ZKServer.ofDraws_order _ _ _
  have hNO : ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop ≤
      ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size := by
    rw [hnP, hord]
    omega
  -- the worker's values fit before the PIOP's end
  have hlenF : m.pos + (fvals G).length = (piopV (Masked.stAt x j).m (Masked.stAt x j).c.kLog ByteArray.empty
      ByteArray.empty ByteArray.empty (Masked.layE x j) (fun _ => 0) (jj + 1)).length := by
    rw [hposm, fvals_length_alike hAG, piopV_succ, List.length_append]
  have hV1 : (piopV (Masked.stAt x j).m (Masked.stAt x j).c.kLog ByteArray.empty ByteArray.empty ByteArray.empty
      (Masked.layE x j) (fun _ => 0) (jj + 1)).length ≤ (Masked.layE x j).fa + 2 + (2 * (Masked.layE x j).nLc + 64) :=
    piopV_length_le (Masked.layE x j) (by omega) hk6 rfl rfl (by omega)
  have hbP : m.pos + nAbs G ≤ ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop := by
    rw [nAbs_fvals, hlenF, hnP, absorbed_size]
    omega
  have hbO : m.pos + nAbs G ≤ ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size := by
    rw [nAbs_fvals, hlenF, hord, absorbed_size]
    omega
  have hKN := piopKN_ok (m := (Masked.stAt x j).m) (jj := jj) (by omega) (lt_of_le_of_lt hm35 (by norm_num)) h0
  -- the words the frame frees are `vs`'s values
  obtain ⟨out, fr, hrf, hfr⟩ := roundFree_vals ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    { m with stopped := false } G (piopK jj) (piopN (Masked.stAt x j).m jj) hc hNO hon hin hmid hplain hKN.1
    hKN.2.1 hKN.2.2 hbP
  have hrf' := hrf
  rw [← hvG, ← hcoins''] at hrf'
  have hRS : Loop.RepSrc L₀ src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).2
  have hz := Loop.RepSrc.size hRS (ofLoopLink L₀ src real) (LeanRelease.draws_ofLoopLink src real) ν t j r hlay
  have hH := (cfg_tapeOfDraws (ofLoopLink L₀ src real) x (ν, t) ((m₀, ρ₀) :: ops) (Masked.UE.tapeZk t) Loop.frepOf
    j.val r.val hrep' hz).1
  rw [hlay, ← Loop.RepSrc.pads hRS (ofLoopLink L₀ src real) (LeanRelease.draws_ofLoopLink src real) ν t j r hlay
    (Masked.UE.tapeZk t)] at hH
  have hhab : (Masked.layE x j).hab = (Masked.layE x j).fa + 3 + 2 * (Masked.layE x j).nLc + 64 +
      (Masked.layE x j).nOod0 + 4 * (Masked.layE x j).k0 + 4 := rfl
  -- `Masked.swSrcE` at `finS.pad (ops.map …)` itself exhausts `whnf`; at a variable list it elaborates
  generalize hcs : (ops.map fun p => Loop.finS.decOf p.2) = cs
  have hvals : ∀ k (hkG : k < (fvals G).length),
      ((Masked.swSrcE x (Loop.finS.pad (K := refK x) cs) w (vs x w ν (Loop.finS.pad (K := refK x) cs) (Loop.finS.seenOf x t)) t
          j.val).rep r.val).masked
          (if m.pos + k < (Masked.layE x j).fa + 2 then m.pos + k else m.pos + k + 1) =
        maskVal ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) false (m.pos + k) ((fvals G)[k]'hkG) := by
    intro k hkG
    have hks : k < fr.size := by
      have e := congrArg List.length hfr
      simp only [List.length_map, Array.length_toList] at e
      omega
    have hpk : m.pos + k < (Masked.layE x j).fa + 2 + (2 * (Masked.layE x j).nLc + 64) := by omega
    have hidx : ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]! =
        if m.pos + k < (Masked.layE x j).fa + 2 then m.pos + k else m.pos + k + 1 := by
      rw [hord, absorbed_getElem! _ (by rw [absorbed_size]; omega)]
    have hE : (if m.pos + k < (Masked.layE x j).fa + 2 then m.pos + k else m.pos + k + 1) <
        (Masked.tabC Masked.PE x j).z.nE := by
      rw [Masked.nE_eq_hab, hhab]
      split_ifs <;> omega
    have hw := hTV v m out _ fr j r hpos hlt hl hok hv hA' hrf' hrep'' k hks
    rw [hidx, getElem!_pos fr k hks, hvG, hcs] at hw
    have e : (fvals G)[k]'hkG = wd (encs G ++ (Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc) (fr[k]'hks) := by
      rw [List.getElem_of_eq hfr.symm hkG, List.getElem_map, Array.getElem_toList]
    rw [Loop.swSrcE_masked_vsWord vs x w ν (Loop.finS.pad (K := refK x) cs) t j r hE, maskVal_false]
    unfold padAt
    rw [hH, hidx, e]
    exact congrArg (· + _) hw.symm
  -- the session's round
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hRk : Masked.rankSlot x ((2 * j.val + r.val) / 2) ((2 * j.val + r.val) % 2) =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, show (2 * j.val + r.val) % 2 = r.val by omega,
      show Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) +
        ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) from rfl, hstj]
  have hNRv : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog =
      6 + ((Masked.stAt x j).m - 6) + ((Masked.stAt x j).c.kLog - 6) := rfl
  have hN := unrefused_prefix Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops
    hlt.le hl href (N := ops.length) le_rfl (q := 2 * j.val + r.val) (by omega) (by omega) (by rw [hRk]; omega)
  have hN0 := unrefused_prefix Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops
    hlt.le hl href (N := Masked.repBase x (2 * j.val + r.val)) (by omega) (q := 2 * j.val + r.val) (by omega) le_rfl
    (by rw [hRk]; omega)
  rw [hcs] at hN hN0
  obtain ⟨dg, cp, rt, hmsg⟩ := Loop.swMsgAt_piop vs w hx hsx ν t cs
    (by rw [← hcs, List.length_map]; exact hlt) j r (by rw [← hcs, List.length_map]; exact hn) h0 hjj
    (by rw [← hn]; exact hN) hN0
  have hfr0 := piopFr_eq_maskFrames (m := (Masked.stAt x j).m) (kLog := (Masked.stAt x j).c.kLog) (digest := dg)
    (cap := cp) (root := rt) (lay := Masked.layE x j)
    (mk := ((Masked.swSrcE x (Loop.finS.pad (K := refK x) cs) w
      (vs x w ν (Loop.finS.pad (K := refK x) cs) (Loop.finS.seenOf x t)) t j.val).rep r.val).masked)
    (by omega) hk6 rfl rfl h0 hjj (P := m.pos) (by rw [hposm]; exact piopV_length_congr _ _ _ _ _)
    (maskVal ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) false)
    (Sched.forall₂_alike_trans (piopFr_alike _ _ _ _ h0) hAG) hvals
  unfold LeanRelease.RoundFrames
  refine ⟨m, G, [], piopK jj, hA', hc, hon, hin, hmid, hplain, ?_, ?_, ?_, hbO, .inl rfl, ?_, ?_⟩
  · rw [hcoins']
    exact hKN.1
  · rw [hcoins']
    exact hKN.2.1
  · rw [hcoins']
    exact hKN.2.2
  · rw [hcoins']
    exact hvG
  · rw [hmsg, hcoins', List.nil_append, hfr0]

end LeanRelease

end RoundFramesPiop

end Claims.ZKServer
