module

public import Proofs.Flock.Claims.Complete.Replay.RepRounds
public import Proofs.Flock.Claims.Complete.Replay.Sizes

/-!
# A rep's rounds are few bytes each

A squeeze records the pending bytes and a 16-byte header (`sent_bytes_size`), then empties them (`afterSq_pending`).
So a round's bytes are what was absorbed since the last squeeze, plus 16. Every absorb of the models is of a fixed
size but four: zerocheck's round-1 vectors, lincheck's `z_partial`, the ring switch's `s_k`, and Ligerito's caps and
`y_r`. Those are bounds `V`, `C` and `Y` here.

* `Wb B P t`: each of `t`'s rounds holds at most `B` bytes, and at most `P` bytes are pending; kept by an absorb of
  at most `c` bytes into `P + c` (`Wb.ap`), and by a squeeze when `P + 16 ≤ B`, after which nothing is pending.
* `Zc.wb_tEnd`, `Lc.wb_tEnd`, `Rs.wb_tEnd`, `Lg.wb_out`: each model keeps `Wb B _`, at a `B` linear in the bounds.
-/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript

/-- Each of `t`'s recorded rounds holds at most `B` bytes, and at most `P` bytes are pending. -/
def Wb (B P : ℕ) (t : Transcript) : Prop := RoundsLe B t ∧ t.pending.size ≤ P

namespace Wb

variable {B P : ℕ}

theorem mono {t : Transcript} (h : Wb B P t) {P' : ℕ} (hP : P ≤ P') : Wb B P' t := ⟨h.1, h.2.trans hP⟩

theorem sq {t : Transcript} (h : Wb B P t) (hP : P + 16 ≤ B) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) :
    Wb B 0 (Replay.afterSq t kind n coin) :=
  ⟨h.1.afterSq (by have := h.2; omega) kind n coin, by simp [afterSq_pending]⟩

theorem ap {t : Transcript} (h : Wb B P t) {f : ByteArray → ByteArray} {c : ℕ} (hf : ∀ Q, (f Q).size ≤ Q.size + c) :
    Wb B (P + c) (Lg.ap f t) := by
  refine ⟨h.1, ?_⟩
  show (f t.pending).size ≤ P + c
  have := hf t.pending
  have := h.2
  omega

theorem init (d : ByteArray) : Wb B (d.size + 31) (Transcript.init #[] d) :=
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

theorem pLabel_le {s : String} (hs : s.toUTF8.size ≤ 64) : ∀ Q, (pLabel s Q).size ≤ Q.size + 95 := by
  intro Q
  have := pLabel_size_le s Q
  omega

theorem pSlice_le {vs : Array F128} {V : ℕ} (hv : vs.size ≤ V) : ∀ Q, (pSlice vs Q).size ≤ Q.size + (16 + 16 * V) := by
  intro Q
  rw [pSlice_size]
  omega

theorem foldl_pF_le {a : Array F128} {Y : ℕ} (hy : a.size ≤ Y) :
    ∀ Q, (a.foldl (fun P w => pF w P) Q).size ≤ Q.size + 32 * Y := by
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

theorem wb_loopT {B m : ℕ} {cs : Challenges} {tr : Trace} {t₀ : Transcript} (h : Wb B 0 t₀) (hB : 80 ≤ B) :
    ∀ i, Wb B 0 (loopT m cs tr t₀ i)
  | 0 => h
  | i + 1 => by
    simp only [loopT]
    exact (((wb_loopT h hB i).ap (pF_le _)).ap (pF_le _)).sq (by omega) _ _ _

/-- Zerocheck's rounds: the binding's (the digest and the cap), round 1's two vectors, and two field elements each. -/
theorem wb_tEnd {B P C V m : ℕ} {digest cap : ByteArray} {cs : Challenges} {tr : Trace} {t : Transcript}
    (h : Wb B P t) (hd : digest.size ≤ C) (hc : cap.size ≤ C) (hab : (r1ab tr (m - 6) (cs.r m)).size ≤ V)
    (hcc : (r1c tr (m - 6) (cs.r m)).size ≤ V) (hP : P + 2 * C + 268 ≤ B) (hV : 32 * V + 48 ≤ B) (h80 : 80 ≤ B) :
    Wb B 64 (tEnd m digest cap cs tr t) := by
  have h₁ : Wb B (P + 95 + (C + 31) + (C + 31) + 95) (t₁ digest cap t) :=
    (((h.ap (pLabel_le (by decide))).ap (pBytes_le hd)).ap (pBytes_le hc)).ap (pLabel_le (by decide))
  have h₃ : Wb B 0 (t₃ m digest cap cs t) := (h₁.sq (by omega) _ _ _).sq (by omega) _ _ _
  have h₅ : Wb B 0 (t₅ m digest cap cs tr t) := ((h₃.ap (pSlice_le hab)).ap (pSlice_le hcc)).sq (by omega) _ _ _
  refine Wb.mono (P := 0 + 32 + 32) ?_ (by omega)
  exact ((wb_loopT h₅ h80 (m - 6)).ap (pF_le _)).ap (pF_le _)

end Zc

/-! ## Lincheck -/

namespace Lc

theorem wb_loopT {B P : ℕ} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Challenges}
    {t : Transcript} (h : Wb B P t) (hP : P + 111 ≤ B) (hB : 80 ≤ B) : ∀ i, Wb B 0 (loopT c m tr zc full cs t i)
  | 0 => ((h.ap (pLabel_le (by decide))).sq (by omega) _ _ _).sq (by omega) _ _ _
  | i + 1 => by
    simp only [loopT]
    exact (((wb_loopT h hP hB i).ap (pF_le _)).ap (pF_le _)).sq (by omega) _ _ _

/-- Lincheck's rounds: its label, two field elements each, and `z_partial`. -/
theorem wb_tEnd {B P V : ℕ} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128}
    {cs : Challenges} {t : Transcript} (h : Wb B P t) (hz : (zrT c m tr zc cs (c.kLog - 6)).size ≤ V)
    (hP : P + 111 ≤ B) (hV : 16 * V + 32 ≤ B) (hB : 80 ≤ B) : Wb B 0 (tEnd c m tr zc full cs t) :=
  ((wb_loopT h hP hB _).ap (pSlice_le hz)).sq (by omega) _ _ _

end Lc

/-! ## The ring switch -/

namespace Rs

theorem wb_loopT {B P V : ℕ} {tr : Trace} {claims : Array Claim} {cs : Challenges} {t : Transcript} (h : Wb B P t)
    (hs : ∀ k, (switchVec tr claims[k]!).size ≤ V) (hB : P + 16 * V + 222 ≤ B) :
    ∀ k, Wb B (P + 95) (loopT tr claims cs t k)
  | 0 => h.ap (pLabel_le (by decide))
  | k + 1 => by
    refine Wb.mono (P := 0) ?_ (by omega)
    exact (((wb_loopT h hs hB k).ap (pLabel_le (by decide))).ap (pSlice_le (hs k))).sq (by omega) _ _ _

/-- The ring switch's rounds: its label, and each claim's label and `s_k`. -/
theorem wb_tEnd {B P V : ℕ} {tr : Trace} {claims : Array Claim} {cs : Challenges} {t : Transcript} (h : Wb B P t)
    (hs : ∀ k, (switchVec tr claims[k]!).size ≤ V) (hB : P + 16 * V + 222 ≤ B) :
    Wb B 0 (tEnd tr claims cs t) :=
  (wb_loopT h hs hB _).sq (by omega) _ _ _

end Rs

/-! ## Ligerito -/

namespace Lg

theorem wb_chal {B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {st : LState} {t : Transcript} (h : Wb B P t)
    (hP : P + 16 ≤ B) : Wb B 96 (chal p cf st t).2.2 := by
  refine Wb.mono (P := 0 + 96) ?_ (by omega)
  exact (h.sq hP _ _ _).ap (pK_le _)

theorem wb_chals {B : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {acc : Array F256} {st : LState}
    {t : Transcript} (h : Wb B 96 t) (hB : 112 ≤ B) : ∀ j, Wb B 96 (chals p cf acc st t j).2.2
  | 0 => h
  | j + 1 => wb_chal (wb_chals h hB j) hB

theorem wb_ood {B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {dim split : ℕ} {st : LState} {t : Transcript}
    (h : Wb B P t) (hP : P + 16 ≤ B) (hB : 183 ≤ B) : Wb B 0 (ood p cf dim split st t).2 :=
  ((((h.sq hP _ _ _).ap (pF_le _)).ap (pK_le _)).ap (pN_le _)).sq (by omega) _ _ _

theorem wb_query {B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {lv : Flock.Level} {rows : Array (Array F128)}
    {fold : Array F256} {st : LState} {t : Transcript} (h : Wb B P t) (hP : P + 55 ≤ B) (hB : 55 ≤ B) :
    Wb B 0 (query p cf lv rows fold st t).2.2 :=
  (((h.ap (pN_le _)).sq (by omega) _ _ _).ap (pN_le _)).sq (by omega) _ _ _

theorem wb_close {B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {cols start : ℕ}
    {r : Array ℕ × Array F128 × F256} {st : LState} {t : Transcript} (h : Wb B P t) (hP : P + 151 ≤ B) :
    Wb B 0 (close p cf cols start r st t).2 :=
  ((h.ap (pK_le _)).ap (pN_le _)).sq (by omega) _ _ _

theorem wb_closeLast {B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {cols start : ℕ}
    {r : Array ℕ × Array F128 × F256} {st : LState} {t : Transcript} (h : Wb B P t) (hP : P + 55 ≤ B) :
    Wb B 0 (closeLast p cf cols start r st t).2 :=
  (h.ap (pN_le _)).sq (by omega) _ _ _

theorem wb_head {B P : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {n : ℕ} {st : LState} {t : Transcript}
    (h : Wb B P t) (hP : P + 16 ≤ B) (hB : 87 ≤ B) : Wb B 96 (head p cf n st t).2 := by
  refine Wb.mono (P := 0 + 96) ?_ (by omega)
  exact ((((h.sq hP _ _ _).ap (pF_le _)).ap (pN_le _)).sq (by omega) _ _ _).ap (pK_le _)

theorem wb_lv0 {B C : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {initCh : Array F256}
    {st : LState} {t : Transcript} (h : Wb B 96 t) (hc : (capBytes p.recursiveCaps[0]!).size ≤ C)
    (hB : C + 247 ≤ B) : Wb B 0 (lv0 p cf sch initCh st t).2 := by
  unfold lv0
  exact wb_close (wb_query (wb_ood (wb_ood (h.ap (pBytes_le hc)) (by omega) (by omega)) (by omega) (by omega))
    (by omega) (by omega)) (by omega)

theorem wb_stepN {B C : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {i : ℕ}
    {x : Vars × LState × Transcript} (h : Wb B 0 x.2.2) (hc : ∀ j, (capBytes p.recursiveCaps[j]!).size ≤ C)
    (hB : C + 247 ≤ B) : Wb B 0 (stepN p cf sch i x).2.2 := by
  unfold stepN
  exact wb_close (wb_query (wb_ood (wb_ood ((wb_chals (h.mono (Nat.zero_le _)) (by omega) 4).ap
    (pBytes_le (hc _))) (by omega) (by omega)) (by omega) (by omega)) (by omega) (by omega)) (by omega)

theorem wb_stepL {B Y : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule}
    {x : Vars × LState × Transcript} (h : Wb B 0 x.2.2) (hy : p.finalProof.yr.size ≤ Y) (hB : 32 * Y + 247 ≤ B) :
    Wb B 0 (stepL p cf sch x).2.2 := by
  unfold stepL
  exact wb_closeLast (wb_query ((wb_chals (h.mono (Nat.zero_le _)) (by omega) 4).ap (foldl_pF_le hy)) (by omega)
    (by omega)) (by omega)

theorem wb_loop {B C Y : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule}
    {x : Vars × LState × Transcript} (h : Wb B 0 x.2.2) (hc : ∀ j, (capBytes p.recursiveCaps[j]!).size ≤ C)
    (hy : p.finalProof.yr.size ≤ Y) (hB : C + 32 * Y + 256 ≤ B) : ∀ j, Wb B 0 (loop p cf sch x j).2.2
  | 0 => h
  | j + 1 => by
    have ih := wb_loop h hc hy hB j
    show Wb B 0 (step p cf sch j (loop p cf sch x j)).2.2
    unfold step
    split
    · exact wb_stepN ih hc (by omega)
    · exact wb_stepL ih hy (by omega)

/-- **Ligerito's rounds**: its label, the claim and the cap, then each level's messages, nonces, values, caps (at most
`C` bytes each) and `y_r` (at most `Y` elements). -/
theorem wb_out {B C Y : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {T : F128}
    {cap : Array ByteArray} {t : Transcript} (h : Wb B 0 t) (hcap : (capBytes cap).size ≤ C)
    (hc : ∀ j, (capBytes p.recursiveCaps[j]!).size ≤ C) (hy : p.finalProof.yr.size ≤ Y)
    (hB : C + 32 * Y + 256 ≤ B) : Wb B 0 (out p cf sch T cap t).2.2 := by
  have hpre : Wb B 0 (pre p cf sch T cap t).2.2 := by
    unfold pre
    exact wb_lv0 (wb_chals (wb_head (((h.ap (pLabel_le (by decide))).ap (pF_le _)).ap (pBytes_le hcap))
      (by omega) (by omega)) (by omega) _) (hc 0) (by omega)
  exact wb_loop (x := x₀ p cf sch T cap t) hpre hc hy hB _

end Lg

end FlockComplete.Replay
