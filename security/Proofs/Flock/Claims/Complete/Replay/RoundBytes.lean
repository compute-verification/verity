module

public import Proofs.Flock.Claims.Complete.Replay.RepRounds
public import Proofs.Flock.Claims.Complete.Replay.Sizes
import Proofs.Flock.Claims.Complete.Json.RecordSize

/-!
# A rep's rounds are few bytes each

A squeeze records the pending bytes and a 16-byte header (`sent_bytes_size`), then empties them (`afterSq_pending`).
So a round's bytes are what was absorbed since the last squeeze, plus 16. Every absorb of the models is of a fixed
size but four: zerocheck's round-1 vectors, lincheck's `z_partial`, the ring switch's `s_k`, and Ligerito's caps and
`y_r`. Those are bounds `V`, `C` and `NY` here.

* `Wb E B P t`: each of `t`'s rounds holds at most `B` bytes, the first at most `B + E`, and at most `P` bytes are
  pending; kept by an absorb of at most `c` bytes into `P + c` (`Wb.ap`), and by a squeeze when `P + 16 ≤ B`, after
  which nothing is pending, the first squeeze when `P + 16 ≤ B + E` (`Wb.sq0`).
* `Zc.wb_tEnd`, `Lc.wb_tEnd`, `Rs.wb_tEnd`, `Lg.wb_out`: each model keeps `Wb E B _`, at a `B` linear in the bounds.
-/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript

/-- Each of `t`'s recorded rounds holds at most `B` bytes, the first at most `B + E`, and at most `P` bytes are
pending. -/
def Wb (E B P : ℕ) (t : Transcript) : Prop := RoundsLe E B t ∧ t.pending.size ≤ P

namespace Wb

variable {E B P : ℕ}

theorem mono {t : Transcript} (h : Wb E B P t) {P' : ℕ} (hP : P ≤ P') : Wb E B P' t := ⟨h.1, h.2.trans hP⟩

theorem sq {t : Transcript} (h : Wb E B P t) (hP : P + 16 ≤ B) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) :
    Wb E B 0 (Replay.afterSq t kind n coin) :=
  ⟨h.1.afterSq (by have := h.2; omega) kind n coin, by simp [afterSq_pending]⟩

theorem sq0 {t : Transcript} (h : Wb E B P t) (h0 : t.rounds.size = 0) (hP : P + 16 ≤ B + E) (kind : UInt8) (n : ℕ)
    (coin : ℕ → F128) : Wb E B 0 (Replay.afterSq t kind n coin) :=
  ⟨h.1.afterSq0 h0 (by have := h.2; omega) kind n coin, by simp [afterSq_pending]⟩

theorem ap {t : Transcript} (h : Wb E B P t) {f : ByteArray → ByteArray} {c : ℕ} (hf : ∀ Q, (f Q).size ≤ Q.size + c) :
    Wb E B (P + c) (Lg.ap f t) := by
  refine ⟨h.1.of_rounds rfl, ?_⟩
  show (f t.pending).size ≤ P + c
  have := hf t.pending
  have := h.2
  omega

theorem init (d : ByteArray) : Wb E B (d.size + 31) (Transcript.init #[] d) :=
  ⟨RoundsLe.init #[] d (by simp), init_pending_size_le #[] d⟩

end Wb

/-! ## The absorbs' sizes -/

theorem pF_le (v : F128) : ∀ Q, (pF v Q).size ≤ Q.size + 32 := fun Q => (pF_size v Q).le

theorem pK_le (m : F256 × F256) : ∀ Q, (Lg.pK m Q).size ≤ Q.size + 96 := by
  intro Q
  unfold Lg.pK
  rw [pSlice_size, pSlice_size]
  simp only [List.size_toArray, List.length_cons, List.length_nil]
  omega

theorem pN_le (v : UInt64) : ∀ Q, (Lg.pN v Q).size ≤ Q.size + 39 := by
  intro Q
  show (pBytes (le64 v) Q).size ≤ _
  have := pBytes_size_le (le64 v) Q
  rw [le64_size] at this
  omega

theorem pBytes_le {b : ByteArray} {C : ℕ} (hb : b.size ≤ C) : ∀ Q, (pBytes b Q).size ≤ Q.size + (C + 31) := by
  intro Q
  have := pBytes_size_le b Q
  omega

theorem pLabel_le {s : String} (hs : s.length ≤ 40) : ∀ Q, (pLabel s Q).size ≤ Q.size + 191 := by
  intro Q
  have h1 := pLabel_size_le s Q
  have h2 : s.toUTF8.size = s.utf8ByteSize := by rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]
  have h3 := size_le_length s
  omega

theorem pSlice_le {vs : Array F128} {V : ℕ} (hv : vs.size ≤ V) : ∀ Q, (pSlice vs Q).size ≤ Q.size + (16 + 16 * V) := by
  intro Q
  rw [pSlice_size]
  omega

theorem foldl_pF_le {a : Array F128} {NY : ℕ} (hy : a.size ≤ NY) :
    ∀ Q, (a.foldl (fun P w => pF w P) Q).size ≤ Q.size + 32 * NY := by
  intro Q
  have key : ∀ (l : List F128) (Q : ByteArray), (l.foldl (fun P w => pF w P) Q).size = Q.size + 32 * l.length := by
    intro l
    induction l with
    | nil => intro Q; simp
    | cons x l ih =>
      intro Q
      rw [List.foldl_cons, ih, pF_size, List.length_cons]
      omega
  rw [← Array.foldl_toList, key, Array.length_toList]
  omega

/-! ## Zerocheck -/

namespace Zc

theorem wb_loopT {E B m : ℕ} {cs : Challenges} {tr : Trace} {t₀ : Transcript} (h : Wb E B 0 t₀) (hB : 80 ≤ B) :
    ∀ i, Wb E B 0 (loopT m cs tr t₀ i)
  | 0 => h
  | i + 1 => by
    simp only [loopT]
    exact (((wb_loopT h hB i).ap (pF_le _)).ap (pF_le _)).sq (by omega) _ _ _

theorem r1ab_size (tr : Trace) (n : ℕ) (r : Array F128) : (r1ab tr n r).size = 64 := by
  simp only [r1ab, Array.size_map, Array.size_range]

theorem r1c_size (tr : Trace) (n : ℕ) (r : Array F128) : (r1c tr n r).size = 64 := by
  simp only [r1c, Array.size_map, Array.size_range]

/-- Zerocheck's rounds: the binding's (the digest and the cap), round 1's two vectors, and two field elements each. -/
theorem wb_tEnd {E B P C V m : ℕ} {digest cap : ByteArray} {cs : Challenges} {tr : Trace} {t : Transcript}
    (h : Wb E B P t) (hd : digest.size ≤ C) (hc : cap.size ≤ C) (hab : (r1ab tr (m - 6) (cs.r m)).size ≤ V)
    (hcc : (r1c tr (m - 6) (cs.r m)).size ≤ V) (hP : P + 2 * C + 460 ≤ B) (hV : 32 * V + 48 ≤ B) (h80 : 80 ≤ B) :
    Wb E B 64 (tEnd m digest cap cs tr t) := by
  have h₁ : Wb E B (P + 191 + (C + 31) + (C + 31) + 191) (t₁ digest cap t) :=
    (((h.ap (pLabel_le (by decide +kernel))).ap (pBytes_le hd)).ap (pBytes_le hc)).ap
      (pLabel_le (by decide +kernel))
  have h₃ : Wb E B 0 (t₃ m digest cap cs t) := (h₁.sq (by omega) _ _ _).sq (by omega) _ _ _
  have h₅ : Wb E B 0 (t₅ m digest cap cs tr t) := ((h₃.ap (pSlice_le hab)).ap (pSlice_le hcc)).sq (by omega) _ _ _
  refine Wb.mono (P := 0 + 32 + 32) ?_ (by omega)
  exact ((wb_loopT h₅ h80 (m - 6)).ap (pF_le _)).ap (pF_le _)

end Zc

/-! ## Lincheck -/

namespace Lc

theorem wb_loopT {E B P : ℕ} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Challenges}
    {t : Transcript} (h : Wb E B P t) (hP : P + 207 ≤ B) (hB : 80 ≤ B) : ∀ i, Wb E B 0 (loopT c m tr zc full cs t i)
  | 0 => ((h.ap (pLabel_le (by decide +kernel))).sq (by omega) _ _ _).sq (by omega) _ _ _
  | i + 1 => by
    simp only [loopT]
    exact (((wb_loopT h hP hB i).ap (pF_le _)).ap (pF_le _)).sq (by omega) _ _ _

theorem zrT_size_le (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (cs : Challenges) :
    ∀ i, (zrT c m tr zc cs i).size ≤ 2 ^ c.kLog
  | 0 => by simp only [zrT, zr0, Array.size_map, Array.size_range, le_refl]
  | i + 1 => by
    simp only [zrT]
    unfold halve
    simp only [Array.size_map, Array.size_range]
    exact (Nat.div_le_self _ _).trans (zrT_size_le c m tr zc cs i)

/-- Lincheck's rounds: its label, two field elements each, and `z_partial`. -/
theorem wb_tEnd {E B P V : ℕ} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128}
    {cs : Challenges} {t : Transcript} (h : Wb E B P t) (hz : (zrT c m tr zc cs (c.kLog - 6)).size ≤ V)
    (hP : P + 207 ≤ B) (hV : 16 * V + 32 ≤ B) (hB : 80 ≤ B) : Wb E B 0 (tEnd c m tr zc full cs t) :=
  ((wb_loopT h hP hB _).ap (pSlice_le hz)).sq (by omega) _ _ _

end Lc

/-! ## The ring switch -/

namespace Rs

theorem switchVec_size (tr : Trace) (c : Claim) : (FlockComplete.switchVec tr c).size = 128 := by
  simp only [FlockComplete.switchVec, Array.size_map, Array.size_range]

theorem wb_loopT {E B P V : ℕ} {tr : Trace} {claims : Array Claim} {cs : Challenges} {t : Transcript} (h : Wb E B P t)
    (hs : ∀ k : ℕ, (switchVec tr claims[k]!).size ≤ V) (hB : P + 16 * V + 414 ≤ B) :
    ∀ k, Wb E B (P + 191) (loopT tr claims cs t k)
  | 0 => h.ap (pLabel_le (by decide +kernel))
  | k + 1 => by
    refine Wb.mono (P := 0) ?_ (by omega)
    exact (((wb_loopT h hs hB k).ap (pLabel_le (by decide +kernel))).ap (pSlice_le (hs k))).sq (by omega) _ _ _

/-- The ring switch's rounds: its label, and each claim's label and `s_k`. -/
theorem wb_tEnd {E B P V : ℕ} {tr : Trace} {claims : Array Claim} {cs : Challenges} {t : Transcript} (h : Wb E B P t)
    (hs : ∀ k : ℕ, (switchVec tr claims[k]!).size ≤ V) (hB : P + 16 * V + 414 ≤ B) :
    Wb E B 0 (tEnd tr claims cs t) :=
  (wb_loopT h hs hB _).sq (by omega) _ _ _

end Rs

/-! ## Ligerito -/

namespace Lg

theorem wb_chal {E B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {st : LState} {t : Transcript} (h : Wb E B P t)
    (hP : P + 16 ≤ B) : Wb E B 96 (chal p cf st t).2.2 := by
  refine Wb.mono (P := 0 + 96) ?_ (by omega)
  exact (h.sq hP _ _ _).ap (pK_le _)

theorem wb_chals {E B : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {acc : Array F256} {st : LState}
    {t : Transcript} (h : Wb E B 96 t) (hB : 112 ≤ B) : ∀ j, Wb E B 96 (chals p cf acc st t j).2.2
  | 0 => h
  | j + 1 => wb_chal (wb_chals h hB j) hB

theorem wb_ood {E B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {dim split : ℕ} {st : LState} {t : Transcript}
    (h : Wb E B P t) (hP : P + 16 ≤ B) (hB : 183 ≤ B) : Wb E B 0 (ood p cf dim split st t).2 :=
  ((((h.sq hP _ _ _).ap (pF_le _)).ap (pK_le _)).ap (pN_le _)).sq (by omega) _ _ _

theorem wb_query {E B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {lv : Flock.Level} {rows : Array (Array F128)}
    {fold : Array F256} {st : LState} {t : Transcript} (h : Wb E B P t) (hP : P + 55 ≤ B) (hB : 55 ≤ B) :
    Wb E B 0 (query p cf lv rows fold st t).2.2 :=
  (((h.ap (pN_le _)).sq (by omega) _ _ _).ap (pN_le _)).sq (by omega) _ _ _

theorem wb_close {E B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {cols start : ℕ}
    {r : Array ℕ × Array F128 × F256} {st : LState} {t : Transcript} (h : Wb E B P t) (hP : P + 151 ≤ B) :
    Wb E B 0 (close p cf cols start r st t).2 :=
  ((h.ap (pK_le _)).ap (pN_le _)).sq (by omega) _ _ _

theorem wb_closeLast {E B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {cols start : ℕ}
    {r : Array ℕ × Array F128 × F256} {st : LState} {t : Transcript} (h : Wb E B P t) (hP : P + 55 ≤ B) :
    Wb E B 0 (closeLast p cf cols start r st t).2 :=
  (h.ap (pN_le _)).sq (by omega) _ _ _

theorem wb_head {E B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {n : ℕ} {st : LState} {t : Transcript}
    (h : Wb E B P t) (hP : P + 16 ≤ B) (hB : 87 ≤ B) : Wb E B 96 (head p cf n st t).2 := by
  refine Wb.mono (P := 0 + 96) ?_ (by omega)
  exact ((((h.sq hP _ _ _).ap (pF_le _)).ap (pN_le _)).sq (by omega) _ _ _).ap (pK_le _)

theorem wb_lv0 {E B C : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {initCh : Array F256}
    {st : LState} {t : Transcript} (h : Wb E B 96 t) (hc : (capBytes p.recursiveCaps[0]!).size ≤ C)
    (hB : C + 247 ≤ B) : Wb E B 0 (lv0 p cf sch initCh st t).2 := by
  unfold lv0
  exact wb_close (wb_query (wb_ood (wb_ood (h.ap (pBytes_le hc)) (by omega) (by omega)) (by omega) (by omega))
    (by omega) (by omega)) (by omega)

theorem wb_stepN {E B C : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {i : ℕ}
    {x : Vars × LState × Transcript} (h : Wb E B 0 x.2.2) (hc : ∀ j : ℕ, (capBytes p.recursiveCaps[j]!).size ≤ C)
    (hB : C + 247 ≤ B) : Wb E B 0 (stepN p cf sch i x).2.2 := by
  unfold stepN
  exact wb_close (wb_query (wb_ood (wb_ood ((wb_chals (h.mono (Nat.zero_le _)) (by omega) 4).ap
    (pBytes_le (hc _))) (by omega) (by omega)) (by omega) (by omega)) (by omega) (by omega)) (by omega)

theorem wb_stepL {E B NY : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule}
    {x : Vars × LState × Transcript} (h : Wb E B 0 x.2.2) (hy : p.finalProof.yr.size ≤ NY) (hB : 32 * NY + 247 ≤ B) :
    Wb E B 0 (stepL p cf sch x).2.2 := by
  unfold stepL
  exact wb_closeLast (wb_query ((wb_chals (h.mono (Nat.zero_le _)) (by omega) 4).ap (foldl_pF_le hy)) (by omega)
    (by omega)) (by omega)

theorem wb_loop {E B C NY : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule}
    {x : Vars × LState × Transcript} (h : Wb E B 0 x.2.2) (hc : ∀ j : ℕ, (capBytes p.recursiveCaps[j]!).size ≤ C)
    (hy : p.finalProof.yr.size ≤ NY) (hB : C + 32 * NY + 256 ≤ B) : ∀ j, Wb E B 0 (loop p cf sch x j).2.2
  | 0 => h
  | j + 1 => by
    have ih := wb_loop (cf := cf) (sch := sch) h hc hy hB j
    show Wb E B 0 (step p cf sch j (loop p cf sch x j)).2.2
    unfold step
    split
    · exact wb_stepN ih hc (by omega)
    · exact wb_stepL ih hy (by omega)

/-- **Ligerito's rounds**: its label, the claim and the cap, then each level's messages, nonces, values, caps (at most
`C` bytes each) and `y_r` (at most `NY` elements). -/
theorem wb_out {E B C NY : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {T : F128}
    {cap : Array ByteArray} {t : Transcript} (h : Wb E B 0 t) (hcap : (capBytes cap).size ≤ C)
    (hc : ∀ j : ℕ, (capBytes p.recursiveCaps[j]!).size ≤ C) (hy : p.finalProof.yr.size ≤ NY)
    (hB : C + 32 * NY + 300 ≤ B) : Wb E B 0 (out p cf sch T cap t).2.2 := by
  have hpre : Wb E B 0 (pre p cf sch T cap t).2.2 := by
    unfold pre
    exact wb_lv0 (wb_chals (wb_head (((h.ap (pLabel_le (by decide +kernel))).ap (pF_le _)).ap (pBytes_le hcap))
      (by omega) (by omega)) (by omega) _) (hc 0) (by omega)
  exact wb_loop (x := x₀ p cf sch T cap t) hpre hc hy (by omega) _

end Lg

end FlockComplete.Replay
