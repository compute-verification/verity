module

public import Proofs.Flock.SessionLoop.RoundLigFramePast
public import Proofs.Flock.SessionLoop.LiveInner
import all Init.Data.String.Defs

/-!
# The frame past level 0's close

From the opening's first round on, the frame keeps its phase and rank (`Ranked.Good`); level 0's closing label `C`
leaves it outside every masked phase, and so does each later round of good frames (`LigOff.PastRound`). So where the
frame's check passes a device round at the close or past it, the round is the scheduled one and leaves as it came:

* **`LigOff.check_closeAt`**, **`LigOff.check_pastAt`**: the closing round, where on the frame is in level 0, and a
  later round outside every masked phase, pass only as the scheduled round, released as its frames came.
* **`LigOff.repState_close`**: the loop's state at a rep's round from the close to the opening's last, where it takes
  the coins before it, is in level 0 if anywhere, and past the close outside every masked phase.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.LigOff

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree check labelFrame isLabel LABEL_ZC LABEL_PCS)
open FlockSoundness.Refine (Frame encs)
open Loop

/-- Frames without a domain frame leave unmasked as they are. -/
theorem maskFrames_id : ∀ (p : ℕ) (fs : List Frame), (∀ f ∈ fs, noDom f) → maskFrames (fun _ v => v) p fs = fs
  | _, [], _ => rfl
  | p, f :: fs, h => by
    have h' : ∀ g ∈ fs, noDom g := fun g hg => h g (List.mem_cons_of_mem _ hg)
    cases f with
    | dom b => exact (h _ List.mem_cons_self : False).elim
    | label b => simp only [maskFrames, maskFrames_id p fs h']
    | bytes b => simp only [maskFrames, maskFrames_id p fs h']
    | one v => simp only [maskFrames, maskFrames_id (p + 1) fs h']
    | vec vs => simp only [maskFrames, maskVals_id, maskFrames_id (p + vs.length) fs h']
    | sq k n => simp only [maskFrames, maskFrames_id p fs h']

/-- Coins the frame takes keep its phase flags. -/
theorem answer_flags {cfg : Cfg} {s s' : Masker} {cs : Array F128} (h : s.answer cfg cs = .ok s') :
    s'.on = s.on ∧ s'.l0 = s.l0 := by
  cases hi : s.inner with
  | some k =>
    obtain ⟨-, a, b, -, -⟩ := Ranked.answer_proj (Or.inl (by rw [hi]; rfl)) h
    exact ⟨a, b⟩
  | none =>
    cases hl : s.l0
    · rw [answer_alg cfg s cs hi hl h]
      exact ⟨rfl, hl⟩
    · obtain ⟨-, a, b, -, -⟩ := Ranked.answer_proj (Or.inr hl) h
      exact ⟨a, b.trans hl⟩

/-- **Level 0's closing label**, where the frame takes it at a state in level 0 if in a masked phase at all, leaves
it outside every masked phase and releases its label frame. -/
theorem label_close_out {c : Cfg} {s : Masker} {q : Masker × ByteArray} (hl : s.on = true → s.l0 = true)
    (h : s.label c _root_.Flock.Zk.LABEL_CLOSE.toUTF8 = .ok q) :
    q.1.on = false ∧ q.2 = labelFrame _root_.Flock.Zk.LABEL_CLOSE.toUTF8 := by
  obtain ⟨p1, p2, p3, p4⟩ := Ranked.close_other
  cases hl0 : s.l0
  · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0] at h
    cases h
    refine ⟨?_, rfl⟩
    show s.on = false
    cases hon : s.on
    · rfl
    · have := hl hon
      rw [hl0] at this
      cases this
  · by_cases hpos : s.pos = c.order.size
    · by_cases hrho : s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
        cases h
        exact ⟨rfl, rfl⟩
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
        cases h
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0, hpos] at h
      cases h

/-- **A round outside every masked phase stays outside**: where the frame takes plain frames and the squeeze there,
past the inner proof's label or not, it is still outside. -/
theorem roundFree_offAt_on {c : Cfg} {s : Masker} {fs : List Frame} {k : UInt8} {n : ℕ} (hon : s.on = false)
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok q) :
    q.2.1.on = false := by
  have hs := roundFree_ok_inner h
  obtain ⟨r, hr, h1, -⟩ := foldlM_stepAt_off c fs s ByteArray.empty #[] 0 0 hon hp
  rw [roundFree_encs c s fs k n hs hm hk h0 hn, hr] at h
  obtain ⟨m, o, fr, q'⟩ := r
  simp only at h1
  subst h1
  obtain ⟨a, ha, h⟩ := bindV_ok h
  cases ha
  have e := exMap_ok_eq h
  subst e
  exact hon

/-- **Level 0's closing round**: where the frame takes the closing label, then plain frames and the squeeze, at a state
in level 0 if in a masked phase at all, it frees no word, leaves the frame outside every masked phase and releases the
frames as they came, then the squeeze. -/
theorem roundFree_closeAt {c : Cfg} {s : Masker} {fs : List Frame} {k : UInt8} {n : ℕ}
    (hl : s.on = true → s.l0 = true) (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s (ByteArray.mk (encs (.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs) ++
      (Frame.sq k n).enc).toArray) n = .ok q) :
    q.2.1.on = false ∧ q.2.2 = #[] ∧
      q.1.data.toList = encs (maskFrames (fun _ v => v) 0 (.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs)) ++
        (Frame.sq k n).enc := by
  have hs := roundFree_ok_inner h
  have hm' : ∀ f ∈ Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs, midFrame f := by
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · show _root_.Flock.Zk.LABEL_CLOSE.toUTF8.size < 2 ^ 64
      decide
    · exact hm f hf
  rw [roundFree_encs c s _ k n hs hm' hk h0 hn] at h
  obtain ⟨x, hx, h⟩ := bindV_ok h
  rw [List.foldlM_cons] at hx
  obtain ⟨y, hy, hx⟩ := bindV_ok hx
  obtain ⟨ql, hql, hy1, hy2, hy3⟩ : ∃ ql, s.label c _root_.Flock.Zk.LABEL_CLOSE.toUTF8 = .ok ql ∧ y.1 = ql.1 ∧
      y.2.1 = ByteArray.empty ++ ql.2 ∧ y.2.2.1 = #[] := by
    simp only [stepAt] at hy
    cases hq : s.label c _root_.Flock.Zk.LABEL_CLOSE.toUTF8 with
    | error e => rw [hq] at hy; cases hy
    | ok ql =>
      rw [hq] at hy
      cases hy
      exact ⟨ql, rfl, rfl, rfl, rfl⟩
  obtain ⟨hof, hout⟩ := label_close_out hl hql
  obtain ⟨y1, y2, y3, y4⟩ := y
  simp only at hy1 hy2 hy3
  subst hy2 hy3
  rw [← hy1] at hof
  obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs y1 (ByteArray.empty ++ ql.2) #[] y4 0 hof hp
  have hfr := foldlM_stepAt_off_fr c fs y1 (ByteArray.empty ++ ql.2) #[] y4 r hof hp hr
  have hxr : x = r := Except.ok.inj (hx.symm.trans hr)
  rw [hxr] at h
  obtain ⟨m, o, fr, q'⟩ := r
  simp only at h1 h2 hfr
  subst h1 hfr
  have e := exMap_ok_eq h
  subst e
  refine ⟨hof, rfl, ?_⟩
  dsimp only
  rw [data_append', h2, data_append', hout, labelFrame_data, sq_enc]
  simp [maskFrames, encs]

/-- **Level 0's closing round passes only as itself**: where the frame's check passes the device round `v` against
the scheduled closing round, at a state in level 0 if in a masked phase at all, `v` is that round and leaves as its
frames. -/
theorem check_closeAt {c : Cfg} {m s' : Masker} {fs : List Frame} {k : UInt8} {n : ℕ} {v out : ByteArray}
    (hl : m.on = true → m.l0 = true) (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64)
    (h : check c m v (ByteArray.mk (encs (.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs) ++
      (Frame.sq k n).enc).toArray) n = .ok (out, s', none)) :
    v = ByteArray.mk (encs (.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs) ++ (Frame.sq k n).enc).toArray ∧
      out.data.toList = encs (maskFrames (fun _ v => v) 0 (.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs)) ++
        (Frame.sq k n).enc := by
  obtain ⟨q, hq, hsz, hd⟩ := passes_none (check_none_passes h)
  rw [(roundFree_closeAt hl hm hp hk h0 hn hq).2.1] at hd
  have hv := eq_of_firstDiff hsz hd
  refine ⟨hv, ?_⟩
  obtain ⟨fr, hf⟩ := Sched.roundFree_round (check_none h)
  rw [hv] at hf
  exact (roundFree_closeAt hl hm hp hk h0 hn hf).2.2

/-- **A round of the schedule past level 0's close passes only as itself**: where the frame's check passes the device
round `v` against the scheduled round `r` (`PastRound`), at the close in level 0 if in a masked phase at all and past
it outside every masked phase, `v` is `r` and leaves as `r`'s bytes. -/
theorem check_pastAt {c : Cfg} {m s' : Masker} {C p : ℕ} {r : ByteArray × ℕ} {v out : ByteArray}
    (hCp : C ≤ p) (hr : PastRound C p r) (hl : m.on = true → m.l0 = true) (hon : C < p → m.on = false)
    (h : check c m v r.1 r.2 = .ok (out, s', none)) :
    v = r.1 ∧ out.data.toList = r.1.data.toList := by
  obtain ⟨fs, k, hb, hfs, hk, hn0, hn⟩ := hr
  have hmf : ∀ f ∈ fs, midFrame f := fun f hf => (hfs f hf).1
  have hpf : ∀ f ∈ fs, plainFrame f := fun f hf => (hfs f hf).2.1
  have hdf : ∀ f ∈ fs, noDom f := fun f hf => (hfs f hf).2.2
  rw [hb]
  rw [Flags.eq_mk_encs hb] at h ⊢
  unfold FlockComplete.LigShape.preAt at h ⊢
  by_cases hpC : p = C
  · rw [ite_eq_left hpC, List.singleton_append] at h ⊢
    obtain ⟨hv, ho⟩ := check_closeAt hl hmf hpf hk (fun h' => hn0 h'.1) hn h
    refine ⟨hv, ?_⟩
    rw [ho, maskFrames_id 0 _ (fun f hf => by
      rcases List.mem_cons.1 hf with rfl | hf
      · trivial
      · exact hdf f hf)]
  · rw [ite_eq_right hpC, List.nil_append] at h ⊢
    obtain ⟨hv, ho⟩ := check_offAt (hon (by omega)) hmf hpf hk (fun h' => hn0 h'.1) hn h
    refine ⟨hv, ?_⟩
    rw [ho, maskFrames_id 0 _ hdf]

/-- **A rep's frame from its first round keeps its phase and rank**: where its scheduled rounds up to `J > 0` run on,
the frame is `Ranked.Good`. -/
theorem run_good {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6))
    {J : ℕ} (hJ0 : 0 < J) (hJ : J < sched.size)
    (h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2 = false) :
    Ranked.Good c (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m := by
  rw [Ranked.schedReqs_pairs P sched #[] cr J hJ] at h2 ⊢
  obtain ⟨p, -, -, he⟩ := Ranked.run_prior_eq c P _ h2
  rw [he] at h2 ⊢
  have hS : 0 < sched.size := by omega
  have hq0 : Ranked.qOf sched #[] 0 = .round 6 (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes digest, .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)
      (.sched (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
        .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray)) := by
    simp only [Ranked.qOf, hS, ↓reduceIte, h0]
  rw [hq0] at h2 ⊢
  exact (Ranked.run_head hc p hdg hcap hdom (Ranked.qOf_round sched #[])
    (by simp only [Ranked.nnOf, hS, ↓reduceIte, h0]) hJ0 h2).1

/-- **A round of the schedule past the opening's first**: from a good state past the opening, outside every masked
phase from the close `C` on, a round that is `PastRound`'s from the close on leaves the frame good and past the opening,
and from the close on outside every masked phase. -/
theorem past_round {c : Cfg} {s : Masker} {C J : ℕ} {r : ByteArray × ℕ} {q : ByteArray × Masker × Array ℕ}
    (hg : Ranked.Good c s) (hd : s.done = true) (hon : C ≤ J → s.on = false)
    (hpast : C ≤ J + 1 → PastRound C (J + 1) r) (h : roundFree c s r.1 r.2 = .ok q) :
    Ranked.Good c q.2.1 ∧ q.2.1.done = true ∧ (C ≤ J + 1 → q.2.1.on = false) := by
  obtain ⟨⟨hg', -, -, hd'⟩, -, -⟩ := Ranked.roundFree_keep hg h
  refine ⟨hg', hd' hd, fun hC => ?_⟩
  obtain ⟨fs, k, hb, hfs, hk, hn0, hn⟩ := hpast hC
  rw [Flags.eq_mk_encs hb] at h
  unfold FlockComplete.LigShape.preAt at h
  by_cases hJC : J + 1 = C
  · rw [ite_eq_left hJC, List.singleton_append] at h
    exact (roundFree_closeAt (hg.2 hd).1 (fun f hf => (hfs f hf).1) (fun f hf => (hfs f hf).2.1) hk
      (fun h' => hn0 h'.1) hn h).1
  · rw [ite_eq_right hJC, List.nil_append] at h
    exact roundFree_offAt_on (hon (by omega)) (fun f hf => (hfs f hf).1) (fun f hf => (hfs f hf).2.1) hk
      (fun h' => hn0 h'.1) hn h

/-- **A rep's frame past the opening's first round**: where its scheduled rounds up to `J`, from the opening's first at
`A` on, run on, quiet before `A` and `PastRound`'s from the close `C` to `J`, the frame owes nothing or is good, past
the opening, and from the close on outside every masked phase. -/
theorem run_past {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6))
    {A C : ℕ} (hA0 : 0 < A) (hp : Flags.PcsRound (sched.getD A (.empty, 0)))
    (hq : ∀ k, 1 ≤ k → k < A → Quiet.QRound (sched.getD k (.empty, 0))) :
    ∀ J, A ≤ J → J < sched.size → (∀ k, C ≤ k → k ≤ J → PastRound C k (sched.getD k (.empty, 0))) →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2 = false →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m.owed = none ∨
        (Ranked.Good c (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m ∧
          (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m.done = true ∧
          (C ≤ J → (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1.m.on = false))
  | 0, hAJ, _, _, _ => absurd hAJ (by omega)
  | J + 1, hAJ, hJ, hP, h2 => by
    by_cases hJA : J + 1 = A
    · rcases Flags.sched_flags hc P sched cr hdg hcap hdom h0 hA0 hp (J + 1)
        (fun k h1 hk hkA => hq k h1 (by omega)) h2 with ho | ⟨-, hF⟩
      · exact Or.inl ho
      · exact Or.inr ⟨run_good hc P sched cr hdg hcap hdom h0 (by omega) hJ h2, (hF.2.2.2.2 (by omega)).1,
          fun _ => (hF.2.2.2.2 (by omega)).2.1⟩
    · have hI0 := fun hb => run_past hc P sched cr hdg hcap hdom h0 hA0 hp hq J (by omega) (by omega)
        (fun k h1 hk => hP k h1 (by omega)) hb
      rw [schedReqs_succ, Loop.run_append] at h2 ⊢
      cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).2
      · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ⊢
        have hI := hI0 hb
        generalize (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr J)).1 = S at h2 hI ⊢
        obtain ⟨s1, ha, he⟩ := run_coins_cons c S _ _ h2
        rw [he] at h2 ⊢
        rcases hI with ho | ⟨hg, hd, hon⟩
        · exact absurd ha (answer_owed_none _ ho)
        obtain ⟨hg1, -, hd1, -, -⟩ := Ranked.answer_keep hg ha
        have ho1 := answer_owed ha
        have hf1 := answer_flags ha
        rcases Loop.run_sched_cons (st := { S with m := s1 }) ho1 h2 with ⟨out, sx, why, hck, he'⟩ | he'
        · rw [he']
          show sx.owed = none ∨ (Ranked.Good c sx ∧ sx.done = true ∧ (C ≤ J + 1 → sx.on = false))
          right
          rcases Loop.check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
          · exact past_round (q := (out, sx, fr)) hg1 (hd1.trans hd) (fun h => hf1.1.trans (hon h))
              (fun h => hP (J + 1) h le_rfl) hr
          · exact past_round (q := (out, sx, fr)) (Ranked.good_stopped hg1) (hd1.trans hd)
              (fun h => hf1.1.trans (hon h)) (fun h => hP (J + 1) h le_rfl) hr
        · rw [he']
          exact Or.inl ho1
      · simp [hb] at h2

/-- **The loop's state at a rep's round from level 0's close**: where the loop's state at a rep's round from the close
`C` to the opening's last takes the coins before it, at the layout `c''`, the frame is in level 0 if in a masked phase
at all, and past the close outside every masked phase. -/
theorem repState_close {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) (hn0 : t.sch.n ≠ 0)
    (htp : 2 * t.tPad < 2 ^ 64) (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64)
    (hcr1 : alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 ≤ cr.size)
    (hcr2 : cr.size ≤ alg.size + (Count.openList (xs.size + 2) t.sch).length) {N : ℕ} (hN : cr.size = N)
    {cs : Array F128} {m : Masker} (ha : (repState c t rep dom xs hi cr0 cr ns N).answer c'' cs = .ok m) :
    (m.on = true → m.l0 = true) ∧ (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 < cr.size → m.on = false) := by
  subst hN
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hpop : cr.pop.size = cr.size - 1 := by rw [Array.size_pop]
  obtain ⟨hdrop, hnf⟩ := Bind.schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.pop.extract alg.size cr.pop.size).toList.length = cr.pop.size - alg.size := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := Bind.feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.pop.extract 0 alg.size) hdom hdg hcap hm hx hk hn hq)
    (by rw [hcsl]; omega) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = cr.pop.size + 1 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  obtain ⟨sv, hsv, hbody⟩ := Bind.openStep_switch t rep dom xs hi (cr.pop.extract 0 alg.size) hdom hdg hcap hm hmk
    (cr.pop.extract alg.size cr.pop.size).toList (by omega)
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have hN0 : xs.size + 2 ≠ 0 := by omega
  have hp : Flags.PcsRound (sched.getD alg.size (.empty, 0)) := by
    have h7 := Bind.sched_count hal hs hdom hdg hcap hm hx hk hn hq (i := 0) (by omega)
    rw [Bind.openList_lt t.sch hN0 (by omega), Nat.add_zero] at h7
    have hb := Bind.sched_bytes hdrop hbody (by omega) (i := 0) (Nat.zero_le _)
    rw [Nat.add_zero] at hb
    rw [show sched.getD alg.size (.empty, 0) = ((sched.getD alg.size (.empty, 0)).1, 7) from Prod.ext rfl h7]
    exact Switch.pcsRound_rs sv hN0 (by rw [hsv]; decide) hb
  have hq' : ∀ k, 1 ≤ k → k < alg.size → Quiet.QRound (sched.getD k (.empty, 0)) := by
    intro k h1 hkl
    rw [hg k hkl, Array.getElem?_eq_getElem hkl, Option.getD_some]
    exact Quiet.rounds_quiet hm hdg hcap hdom hal k _ h1 (Array.getElem?_eq_getElem hkl)
  have hP : ∀ k, alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 ≤ k → k ≤ cr.pop.size →
      PastRound (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4) k (sched.getD k (.empty, 0)) :=
    fun k h1 h2 => (sched_past hal hs hdom hdg hcap hm hmk hx hk hn hq hm14 hn0 htp hd (by omega) h1 h2).2
  rw [show cr.size = cr.pop.size + 1 by omega,
    repState_at c t rep dom xs hi cr0 cr ns cr.pop.size hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr cr.pop.size)).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    rcases run_past hc _ sched cr hdg hcap hdom h0 (by omega) hp hq' cr.pop.size (by omega) (by omega) hP h2 with
      ho | ⟨hg1, hd1, hon1⟩
    · exact absurd ha (answer_owed_none cs ho)
    · obtain ⟨e1, e2⟩ := answer_flags ha
      refine ⟨fun h => ?_, fun h => ?_⟩
      · rw [e2]
        exact (hg1.2 hd1).1 (e1.symm.trans h)
      · rw [e1]
        exact hon1 (by omega)
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

end Claims.ZKServer.LigOff
