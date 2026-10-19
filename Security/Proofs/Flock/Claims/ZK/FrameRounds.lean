import Proofs.Flock.Claims.ZK.SimZerocheck
import Proofs.Flock.Claims.ZK.FrameMask

/-!
# The cut's zerocheck, round by round, and the frame's release of the device's

The cut session's zerocheck (`FlockComplete.zerocheckCut`) sends the masked values `mk` at the layout's pad indices, and
the frame releases the device's plain values each plus its pad. This file states both sides at the zerocheck's rounds
and matches them:

* **The cut's bodies** (`zerocheckCut_bodies`, `zcBodies`): answered by `asksCut` with room for its `3 + (m − 6)` asks,
  the cut sends, after the pending bytes, the binding (`zcBind`: the statement's label, digest and capacity, the
  zerocheck's label, the pads' label and root) and `r_skip`'s squeeze; `r_outer`'s squeeze; `round1_ab`'s and
  `round1_c`'s masked values and `z`'s squeeze; then each residual round's two masked values and its coin's squeeze
  (`zcLoopB_enc`), and leaves the finals' two values pending. The loop is `Replay.forIn_go` from any transcript, each
  step's hypothesis below the openings only (`forIn_go_lt`, `forIn_go_from`).
* **The frame's rounds** (`roundFree_zcBind`, `roundFree_zcOuter`, `roundFree_zcR1`, `roundFree_zcMl`): at the device's
  binding (no pads' frames: the frame inserts them) the frame starts the algebraic phase and releases the cut's binding;
  `r_outer`'s squeeze leaves as it came; `round1`'s values (`zcR1`, absorbed positions `0 … 127`) and residual round
  `i`'s (`zcMl`, positions `128 + 2i` and `128 + 2i + 1`) leave each plus its pad (`maskVal`).
* **The match** (`zcBodies_frame`): where each masked value the cut sends below `final_c`'s position is the device's
  value there plus the frame's pad, the cut's bodies from empty pending bytes are the frame's releases of the device's
  rounds at a rep that has not stopped. A rep's first round also carries its stream's domain, pending before the
  binding: the cut sends it and the frame never releases it (`dropDoms`), so the match is at empty pending bytes.
-/

namespace Claims.ZKServer

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Transcript (framed)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree LABEL_ZC)
open FlockSoundness.Refine (Frame encs label_enc absorbBytes_enc absorbSlice_enc absorbF_enc framed_enc)
open FlockComplete (PS asksCut zerocheckCut sampleVecP squeezeP_cut_ok sampleP_cut_ok)
open FlockComplete.Replay (pF pushes afterSq bind_go liftTM_ok pure_go)

/-! ## The cut's zerocheck, round by round -/

/-- **The zerocheck's binding**: the frames `zerocheckCut` (and `zerocheckPZk`) absorbs before its first squeeze, the
pads' label and root after the zerocheck's label. -/
def zcBind (digest cap root : ByteArray) : List Frame :=
  [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label "flock-zerocheck-v0".toUTF8,
    .label _root_.Flock.Zk.LABEL_PADS.toUTF8, .bytes root]

/-- **The cut's zerocheck bodies** after the pending bytes `P`: the binding and `r_skip`'s squeeze; `r_outer`'s squeeze
alone; `round1_ab`'s and `round1_c`'s masked values and `z`'s squeeze; then each residual round's two masked values and
its coin's squeeze. -/
def zcBodies (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (P : List UInt8) : List Claims.Msg :=
  (P ++ encs (zcBind digest cap root) ++ (Frame.sq 2 6).enc) :: (Frame.sq 2 (m - 13)).enc ::
    (encs [.vec ((List.range 64).map fun i => mk (lay.r1ab i)), .vec ((List.range 64).map fun i => mk (lay.r1c i))] ++
      (Frame.sq 1 1).enc) ::
    (List.range (m - 6)).map fun i => encs [.one (mk (lay.ml i 0)), .one (mk (lay.ml i 1))] ++ (Frame.sq 1 1).enc

theorem sampleVecP_cut_ok (coin : ℕ → ℕ → F128) {n k : ℕ} (hk : k ≠ 0) (t : Transcript) {l : Array Claims.Msg}
    (hl : l.size < n) :
    ((sampleVecP (asksCut coin n) k).run t).go l =
      (.ok ((Array.range k).map (coin l.size), afterSq t 2 k (coin l.size)),
        l.push (Claims.toMsg (framed t 2 k))) :=
  squeezeP_cut_ok coin 2 hk t hl

/-- **A loop of `n` steps whose every step yields**, as `Replay.forIn_go`, with each step's hypothesis below `n` only:
under `asksCut` a step past the openings throws. -/
theorem forIn_go_lt {σ : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (S : ℕ → σ) (T : ℕ → Transcript)
    (B : ℕ → Claims.Msg) (L n : ℕ)
    (hbody : ∀ i < n, ∀ (l : Array Claims.Msg), l.size = L + i →
      ((body i (S i)).run (T i)).go l = (.ok (.yield (S (i + 1)), T (i + 1)), l.push (B i)))
    (l : Array Claims.Msg) (hl : l.size = L) :
    ((forIn [:n] (S 0) body).run (T 0)).go l = (.ok (S n, T n), l ++ ((List.range n).map B).toArray) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ k a (l : Array Claims.Msg), a + k ≤ n → l.size = L + a →
      ((forIn (List.range' a k) (S a) body).run (T a)).go l =
        (.ok (S (a + k), T (a + k)), l ++ ((List.range' a k).map B).toArray) by
    simpa [List.range_eq_range'] using h n 0 l (by omega) (by simpa using hl)
  intro k
  induction k with
  | zero =>
    intro a l _ _
    show (Except.ok (S a, T a), l) = _
    simp
  | succ k ih =>
    intro a l hk hl
    rw [List.range'_succ, List.forIn_cons, bind_go (hbody a (by omega) l hl)]
    dsimp only
    rw [ih (a + 1) (l.push (B a)) (by omega) (by simp [hl]; omega), show a + 1 + k = a + (k + 1) by omega]
    simp

/-- `forIn_go_lt` from any transcript `t`, the loop's transcripts `T t` a function of where it starts. -/
theorem forIn_go_from {σ : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (S : ℕ → σ)
    (T : Transcript → ℕ → Transcript) (B : Transcript → ℕ → Claims.Msg) (L n : ℕ) (hT : ∀ t, T t 0 = t)
    (hbody : ∀ t, ∀ i < n, ∀ (l : Array Claims.Msg), l.size = L + i →
      ((body i (S i)).run (T t i)).go l = (.ok (.yield (S (i + 1)), T t (i + 1)), l.push (B t i)))
    (t : Transcript) (l : Array Claims.Msg) (hl : l.size = L) :
    ((forIn [:n] (S 0) body).run t).go l = (.ok (S n, T t n), l ++ ((List.range n).map (B t)).toArray) := by
  have h := forIn_go_lt body S (T t) (B t) L n (hbody t) l hl
  rwa [hT] at h

/-- The transcript of the cut's residual round `i` before its squeeze: `ml i 0`'s and `ml i 1`'s masked values
absorbed. -/
def zcPre (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (i : ℕ) (t : Transcript) : Transcript :=
  ⟨t.rounds, t.k, pF (mk (lay.ml i 1)) (pF (mk (lay.ml i 0)) t.pending)⟩

/-- The cut's zerocheck transcript at its residual rounds, from `t₀`: round `i` absorbs `ml i 0`'s and `ml i 1`'s
masked values and squeezes the coin at body `L + i`. -/
def zcLoopT (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Transcript) :
    ℕ → Transcript
  | 0 => t₀
  | i + 1 => afterSq (zcPre lay mk i (zcLoopT lay mk coin L t₀ i)) 1 1 (coin (L + i))

/-- The body the cut's residual round `i` sends. -/
def zcLoopB (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Transcript)
    (i : ℕ) : Claims.Msg :=
  Claims.toMsg (framed (zcPre lay mk i (zcLoopT lay mk coin L t₀ i)) 1 1)

theorem zcLoopT_pending (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ)
    {t₀ : Transcript} (hz : t₀.pending = ByteArray.empty) :
    ∀ i, (zcLoopT lay mk coin L t₀ i).pending = ByteArray.empty
  | 0 => hz
  | _ + 1 => rfl

theorem empty_toList : ByteArray.empty.data.toList = [] := rfl

/-- Each residual round's body is its two masked values' frames and its coin's squeeze. -/
theorem zcLoopB_enc (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (coin : ℕ → ℕ → F128) (L : ℕ)
    {t₀ : Transcript} (hz : t₀.pending = ByteArray.empty) :
    zcLoopB lay mk coin L t₀ =
      fun i => encs [.one (mk (lay.ml i 0)), .one (mk (lay.ml i 1))] ++ (Frame.sq 1 1).enc := by
  funext i
  simp only [zcLoopB, Claims.toMsg, framed_enc, zcPre, pF, absorbF_enc, zcLoopT_pending lay mk coin L hz i,
    empty_toList, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]

/-- **The cut's zerocheck, answered by `asksCut` with room for its asks**: from `l`, with `3 + (m − 6)` asks before
round `n`, it returns, having sent `zcBodies` after `t`'s pending bytes, and leaves the finals' two masked values
pending. -/
theorem zerocheckCut_bodies (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (t : Transcript) (l : Array Claims.Msg)
    (hl : l.size + 3 + (m - 6) ≤ n) :
    ∃ res t' L', ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l = (.ok (res, t'), L') ∧
      L'.toList = l.toList ++ zcBodies m digest cap root lay mk t.pending.data.toList ∧
      t'.pending.data.toList = encs [.one (mk lay.fa), .one (mk lay.fb)] := by
  unfold zerocheckCut
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleVecP_cut_ok coin (by decide) _ ?h1)]
  rw [bind_go (sampleVecP_cut_ok coin (show m - 13 ≠ 0 by omega) _ ?h2)]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h3)]
  case h1 => omega
  case h2 => simp only [Array.size_push]; omega
  case h3 => simp only [Array.size_push]; omega
  erw [bind_go (forIn_go_from _
      (fun i => (pushes (fun j => (mk (lay.ml j 0), mk (lay.ml j 1))) i, pushes (fun j => coin (l.size + 3 + j) 0) i))
      (zcLoopT lay mk coin (l.size + 3)) (zcLoopB lay mk coin (l.size + 3)) (l.size + 3) (m - 6)
      (fun _ => rfl) ?hbody _ _ ?hl0)]
  case hbody =>
    intro t₀ i hi l' hl'
    dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (sampleP_cut_ok coin _ (by omega))]
    rw [hl']
    rfl
  case hl0 => simp only [Array.size_push]
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [pure_go]
  refine ⟨_, _, _, rfl, ?_, ?_⟩
  · simp only [Array.toList_append, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append,
      zcBodies, zcBind, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, Claims.toMsg, framed_enc,
      label_enc, absorbBytes_enc, absorbSlice_enc, afterSq, Array.toList_map, Array.toList_range, empty_toList,
      zcLoopB_enc]
  · simp only [absorbF_enc, zcLoopT_pending lay mk coin (l.size + 3), afterSq, empty_toList, encs, List.flatMap_cons,
      List.flatMap_nil, List.append_nil, List.nil_append]

/-! ## The frame's release of the device's zerocheck -/

/-- The device's `round1_ab` and `round1_c`, its values at absorbed positions `0 … 127`. -/
def zcR1 (dv : ℕ → F128) : List Frame :=
  [.vec ((List.range' 0 64).map dv), .vec ((List.range' 64 64).map dv)]

/-- The device's residual round `i`, its values at absorbed positions `128 + 2i` and `128 + 2i + 1`. -/
def zcMl (dv : ℕ → F128) (i : ℕ) : List Frame :=
  [.one (dv (128 + 2 * i)), .one (dv (128 + 2 * i + 1))]

/-- **The zerocheck's binding round**: outside a masked phase and before the opening, at the device's binding (the
statement's label, digest and capacity, then the zerocheck's label) and `r_skip`'s squeeze, the frame starts the
algebraic phase and releases the cut's binding, the pads' label and root inserted, then the squeeze. -/
theorem roundFree_zcBind (c : Cfg) (s : Masker) (digest cap : ByteArray)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = false) (hd : s.done = false)
    (hi : s.inner = none) (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hb : s.pos ≤ c.order.size) :
    ∃ out fr, roundFree c s
        (ByteArray.mk (encs [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++
          (Frame.sq 2 6).enc).toArray) 6 =
        .ok (out, { s with on := true, owed := some 6 }, fr) ∧
      out.data.toList = encs (zcBind digest cap c.root) ++ (Frame.sq 2 6).enc := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_zc c s [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap] [] 2 6 hc
    hon hd hi
    (by
      intro f hf
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl | rfl
      · show "flock-r1cs-v0".toUTF8.size < 2 ^ 64; decide
      · exact hdg
      · exact hcap
      · show LABEL_ZC.toUTF8.size < 2 ^ 64; decide)
    (by
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl
      · show plainLabel "flock-r1cs-v0".toUTF8; unfold plainLabel; decide
      · trivial
      · trivial)
    (by simp) (.inl rfl) (by decide) (by decide) hb
  refine ⟨out, fr, hr, ?_⟩
  rw [ho]
  simp only [maskFrames, zcBind, encs, LABEL_ZC, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.append_assoc]

/-- **`r_outer`'s round**: inside the algebraic phase, the device's squeeze alone leaves as it came. -/
theorem roundFree_zcOuter (c : Cfg) (s : Masker) (k : ℕ) (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    (hon : s.on = true) (hi : s.inner = none) (hk : k ≠ 0) (hn : k < 2 ^ 64) (hb : s.pos ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs [] ++ (Frame.sq 2 k).enc).toArray) k =
        .ok (out, { s with owed := some k }, fr) ∧ out.data.toList = (Frame.sq 2 k).enc := by
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on c s [] 2 k hc hon hi (by simp) (by simp) (.inl rfl) (by omega) hn hb
  exact ⟨out, fr, hr, by rw [ho]; rfl⟩

/-- `maskVals` over consecutive positions from where it starts. -/
theorem maskVals_range' (g : ℕ → F128 → F128) (f : ℕ → F128) :
    ∀ (k s : ℕ), maskVals g s ((List.range' s k).map f) = (List.range' s k).map fun q => g q (f q)
  | 0, _ => rfl
  | k + 1, s => by
    rw [List.range'_succ, List.map_cons, List.map_cons, maskVals, maskVals_range' g f k (s + 1)]

theorem maskVal_false (c : Cfg) (p : ℕ) (v : F128) : maskVal c false p v = v + padAt c p := by
  simp [maskVal]

/-- **`round1_ab`'s and `round1_c`'s round**: inside the algebraic phase from absorbed position `0`, the frame releases
the device's values each plus its pad, then `z`'s squeeze. -/
theorem roundFree_zcR1 (c : Cfg) (s : Masker) (dv : ℕ → F128)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = true) (hi : s.inner = none)
    (hpos : s.pos = 0) (hb : 128 ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs (zcR1 dv) ++ (Frame.sq 1 1).enc).toArray) 1 =
        .ok (out, { s with pos := 128, owed := some 1 }, fr) ∧
      out.data.toList = encs (maskFrames (maskVal c s.stopped) 0 (zcR1 dv)) ++ (Frame.sq 1 1).enc := by
  have hm : ∀ f ∈ zcR1 dv, midFrame f := by
    intro f hf
    simp only [zcR1, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> simp [midFrame]
  have hp : ∀ f ∈ zcR1 dv, plainFrame f := by
    intro f hf
    simp only [zcR1, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial
  have hN : nAbs (zcR1 dv) = 128 := by simp [zcR1, nAbs]
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on c s (zcR1 dv) 1 1 hc hon hi hm hp (.inr ⟨rfl, rfl⟩) (by decide)
    (by decide) (by rw [hN, hpos]; omega)
  rw [hN, hpos] at hr
  rw [hpos] at ho
  exact ⟨out, fr, hr, ho⟩

/-- **Residual round `i`'s round**: inside the algebraic phase from absorbed position `128 + 2i`, the frame releases
the device's two values each plus its pad, then the coin's squeeze. -/
theorem roundFree_zcMl (c : Cfg) (s : Masker) (dv : ℕ → F128) (i : ℕ)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = true) (hi : s.inner = none)
    (hpos : s.pos = 128 + 2 * i) (hb : 128 + 2 * i + 2 ≤ c.order.size) :
    ∃ out fr, roundFree c s (ByteArray.mk (encs (zcMl dv i) ++ (Frame.sq 1 1).enc).toArray) 1 =
        .ok (out, { s with pos := 128 + 2 * i + 2, owed := some 1 }, fr) ∧
      out.data.toList = encs (maskFrames (maskVal c s.stopped) (128 + 2 * i) (zcMl dv i)) ++ (Frame.sq 1 1).enc := by
  have hm : ∀ f ∈ zcMl dv i, midFrame f := by
    intro f hf
    simp only [zcMl, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial
  have hp : ∀ f ∈ zcMl dv i, plainFrame f := by
    intro f hf
    simp only [zcMl, List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl <;> trivial
  have hN : nAbs (zcMl dv i) = 2 := rfl
  obtain ⟨out, fr, hr, ho⟩ := roundFree_on c s (zcMl dv i) 1 1 hc hon hi hm hp (.inr ⟨rfl, rfl⟩) (by decide)
    (by decide) (by rw [hN, hpos]; omega)
  rw [hN, hpos] at hr
  rw [hpos] at ho
  exact ⟨out, fr, hr, ho⟩

/-- **The cut's zerocheck bodies are the frame's releases of the device's rounds**, where each masked value the cut
sends at an absorbed position below `final_c`'s is the device's value there plus the frame's pad: the binding (domains
dropped, so no pending prefix), `r_outer`'s squeeze, `round1`'s values from position `0`, and residual round `i`'s from
`128 + 2i`, each masked by a rep that has not stopped. -/
theorem zcBodies_frame (m : ℕ) (digest cap : ByteArray) (c : Cfg) (lay : _root_.Flock.Zk.PadLayout)
    (mk dv : ℕ → F128) (hn : lay.nMlv = m - 6) (hmk : ∀ p < lay.fa + 2, mk p = dv p + padAt c p) :
    zcBodies m digest cap c.root lay mk [] =
      (encs (zcBind digest cap c.root) ++ (Frame.sq 2 6).enc) :: (Frame.sq 2 (m - 13)).enc ::
        (encs (maskFrames (maskVal c false) 0 (zcR1 dv)) ++ (Frame.sq 1 1).enc) ::
        (List.range (m - 6)).map fun i =>
          encs (maskFrames (maskVal c false) (128 + 2 * i) (zcMl dv i)) ++ (Frame.sq 1 1).enc := by
  have hfa : lay.fa = 128 + 2 * (m - 6) := by rw [← hn]; rfl
  have hA : (List.range 64).map (fun i => mk (lay.r1ab i)) = (List.range' 0 64).map fun q => dv q + padAt c q := by
    rw [List.range_eq_range']
    refine List.map_congr_left fun q hq => ?_
    rw [List.mem_range'] at hq
    exact hmk q (by omega)
  have hC : (List.range 64).map (fun i => mk (lay.r1c i)) = (List.range' 64 64).map fun q => dv q + padAt c q := by
    rw [List.range'_eq_map_range, List.map_map]
    refine List.map_congr_left fun q hq => ?_
    rw [List.mem_range] at hq
    exact hmk (64 + q) (by omega)
  unfold zcBodies
  simp only [List.nil_append, zcR1, maskFrames, List.length_map, List.length_range', maskVals_range', maskVal_false,
    hA, hC, List.cons.injEq, true_and]
  refine List.map_congr_left fun i hi => ?_
  rw [List.mem_range] at hi
  simp only [zcMl, maskFrames, maskVal_false]
  rw [hmk (lay.ml i 0) (by unfold _root_.Flock.Zk.PadLayout.ml; omega),
    hmk (lay.ml i 1) (by unfold _root_.Flock.Zk.PadLayout.ml; omega)]
  rfl

end Claims.ZKServer
