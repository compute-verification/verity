module

public import Proofs.Flock.Claims.Complete.Replay.RoundBytes
public import Proofs.Flock.Claims.Complete.Zk.PiopGo
public import Proofs.Flock.Claims.Complete.Zk.InnerGo
public import Proofs.Flock.Claims.Complete.Zk.Level0Model

/-!
# A masked rep's rounds are few bytes each

`Replay.RoundBytes`'s `Wb` at the masked prover's models, stage by stage: the masked zerocheck (`ZcZ.wb_tEnd`, which
also absorbs the pads' label and root), the masked lincheck (`LcZ.wb_tEnd`), the masked Ligerito (`L0.wb_outZ`: plain's
head and level loop around the lane rounds, whose last absorbs the pair's sums, the blinded claim and `ybar`) and the
inner proof (`InZ.wb_tEnd`). The ring switch is plain's (`Rs.wb_tEnd`). The masked vectors of zerocheck and lincheck
have 64 entries; the byte strings (digest, caps, root, `τ`'s commitment) are at most `C` bytes, and `y_r`, `ybar` and
the inner `y` are bounds `NY`, `NB` and `NI` here.
-/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript

/-! ## The masked zerocheck -/

namespace ZcZ

theorem r1ab_size (lay : Zk.PadLayout) (hp : Array F128) (tr : Trace) (n : ℕ) (r : Array F128) :
    (r1ab lay hp tr n r).size = 64 := by
  simp only [r1ab, Array.size_map, Array.size_range]

theorem r1c_size (lay : Zk.PadLayout) (hp : Array F128) (tr : Trace) (n : ℕ) (r : Array F128) :
    (r1c lay hp tr n r).size = 64 := by
  simp only [r1c, Array.size_map, Array.size_range]

theorem wb_loopT {E B m : ℕ} {lay : Zk.PadLayout} {hp : Array F128} {cs : Zc.Challenges} {tr : Trace}
    {t₀ : Transcript} (h : Wb E B 0 t₀) (hB : 80 ≤ B) : ∀ i, Wb E B 0 (loopT lay hp m cs tr t₀ i)
  | 0 => h
  | i + 1 => by
    simp only [loopT]
    exact (((wb_loopT h hB i).ap (pF_le _)).ap (pF_le _)).sq (by omega) _ _ _

/-- The masked zerocheck's rounds: the binding's (the digest, the cap, the pads' label and root), round 1's two padded
vectors, and two field elements each. It starts at no recorded round, so its first squeeze holds the bytes pending
before it, which `E` bounds. -/
theorem wb_tEnd {E B P C m : ℕ} {lay : Zk.PadLayout} {hp : Array F128} {digest cap root : ByteArray}
    {cs : Zc.Challenges} {tr : Trace} {t : Transcript} (h : Wb E B P t) (h0 : t.rounds.size = 0)
    (hd : digest.size ≤ C) (hc : cap.size ≤ C) (hr : root.size ≤ C) (hP : P + 3 * C + 685 ≤ B + E)
    (hB : 2096 ≤ B) : Wb E B 64 (tEnd lay hp m digest cap root cs tr t) := by
  have h₁ : Wb E B (P + 191 + (C + 31) + (C + 31) + 191 + 191 + (C + 31)) (t₁ digest cap root t) :=
    (((((h.ap (pLabel_le (by decide +kernel))).ap (pBytes_le hd)).ap (pBytes_le hc)).ap
      (pLabel_le (by decide +kernel))).ap (pLabel_le (by decide +kernel))).ap (pBytes_le hr)
  have h₃ : Wb E B 0 (t₃ m digest cap root cs t) := (h₁.sq0 h0 (by omega) _ _ _).sq (by omega) _ _ _
  have h₅ : Wb E B 0 (t₅ lay hp m digest cap root cs tr t) :=
    ((h₃.ap (pSlice_le (r1ab_size _ _ _ _ _).le)).ap (pSlice_le (r1c_size _ _ _ _ _).le)).sq (by omega) _ _ _
  refine Wb.mono (P := 0 + 32 + 32) ?_ (by omega)
  exact ((wb_loopT h₅ (by omega) (m - 6)).ap (pF_le _)).ap (pF_le _)

end ZcZ

/-! ## The masked lincheck -/

namespace LcZ

theorem zp_size (lay : Zk.PadLayout) (hp : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut)
    (cs : Lc.Challenges) : (zp lay hp c m tr zc cs).size = 64 := by
  simp only [zp, Array.size_map, Array.size_range]

theorem wb_loopT {E B P m : ℕ} {lay : Zk.PadLayout} {hp : Array F128} {c : CircuitFold} {tr : Trace} {zc : ZcOut}
    {full : Array F128} {cs : Lc.Challenges} {t : Transcript} (h : Wb E B P t) (hP : P + 207 ≤ B) (hB : 80 ≤ B) :
    ∀ i, Wb E B 0 (loopT lay hp c m tr zc full cs t i)
  | 0 => ((h.ap (pLabel_le (by decide +kernel))).sq (by omega) _ _ _).sq (by omega) _ _ _
  | i + 1 => by
    simp only [loopT]
    exact (((wb_loopT h hP hB i).ap (pF_le _)).ap (pF_le _)).sq (by omega) _ _ _

/-- The masked lincheck's rounds: its label, two field elements each, and the padded `z_partial`. -/
theorem wb_tEnd {E B P m : ℕ} {lay : Zk.PadLayout} {hp : Array F128} {c : CircuitFold} {tr : Trace} {zc : ZcOut}
    {full : Array F128} {cs : Lc.Challenges} {t : Transcript} (h : Wb E B P t) (hP : P + 207 ≤ B) (hB : 1056 ≤ B) :
    Wb E B 0 (tEnd lay hp c m tr zc full cs t) :=
  ((wb_loopT h hP (by omega) _).ap (pSlice_le (zp_size _ _ _ _ _ _ _).le)).sq (by omega) _ _ _

end LcZ

/-! ## The inner proof -/

namespace InZ

/-- The inner proof's rounds: its label and `finalC`, `ρ` and `σ`, `τ`'s commitment, and `y`. -/
theorem wb_tEnd {E B P C NI : ℕ} {lay : Zk.PadLayout} {p : PadsZk} {finalC : F128} {cons : Array (String × Zk.Lin)}
    {hsh : ByteArray → ByteArray} {cs : Challenges} {t : Transcript} (h : Wb E B P t)
    (htc : (tcOf hsh (tau lay p cons cs) p.tauSalt).size ≤ C) (hy : (y p cs).size ≤ NI) (hP : P + 239 ≤ B)
    (hC : C + 47 ≤ B) (hN : 16 * NI + 32 ≤ B) (h64 : 64 ≤ B) : Wb E B 0 (tEnd lay p finalC cons hsh cs t) :=
  ((((((((h.ap (pLabel_le (by decide +kernel))).ap (pF_le _)).sq (by omega) _ _ _).ap
    (pSlice_le (show #[rho lay p cs, sigma lay p].size ≤ 2 by simp))).sq (by omega) _ _ _).ap
    (pBytes_le htc)).sq (by omega) _ _ _).ap (pSlice_le hy)).sq (by omega) _ _ _

end InZ

end FlockComplete.Replay

/-! ## The masked Ligerito -/

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg

theorem pKv_le (v : F256) : ∀ Q, (pKv v Q).size ≤ Q.size + (16 + 16 * 2) :=
  pSlice_le (show #[v.c0, v.c1].size ≤ 2 by simp)

/-- The last lane round: the pair's sums, its coins, `LABEL_CLOSE`, the blinded claim, `ybar` and the next message. -/
theorem wb_chalZ {E B P NB : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {zm : Zk.Level0Msgs} {st : LState}
    {t : Transcript} (h : Wb E B P t) (hP : P + 16 ≤ B) (hy : (ybarFlat zm.ybar).size ≤ NB) (hB : 16 * NB + 367 ≤ B) :
    Wb E B (16 * NB + 351) (chalZ p cf zm st t).2.2.2 := by
  unfold chalZ
  dsimp only
  have h₁ : Wb E B (0 + (16 + 16 * 2) + (16 + 16 * 2))
      (ap (pKv zm.e1) (ap (pKv zm.e0) (afterSq t 2 2 (cf t.k)))) :=
    ((h.sq hP 2 2 (cf t.k)).ap (pKv_le zm.e0)).ap (pKv_le zm.e1)
  refine Wb.mono (P := 0 + 191 + (16 + 16 * 2) + (16 + 16 * NB) + 96) ?_ (by omega)
  exact (((((h₁.sq (by omega) _ _ _).sq (by omega) _ _ _).ap (pLabel_le (by decide +kernel))).ap (pKv_le _)).ap
    (pSlice_le hy)).ap (pK_le _)

/-- The lane rounds: plain's challenge, the last one `chalZ`. -/
theorem wb_laneRun {E B NB k0 : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {zm : Zk.Level0Msgs} {st : LState}
    {t : Transcript} (h : Wb E B 96 t) (hy : (ybarFlat zm.ybar).size ≤ NB) (hB : 16 * NB + 367 ≤ B) :
    ∀ j, Wb E B (16 * NB + 351) (laneRun p cf k0 zm st t j).2.2
  | 0 => h.mono (by omega)
  | j + 1 => by
    have ih := wb_laneRun (p := p) (k0 := k0) (cf := cf) (st := st) h hy hB j
    show Wb E B _ (laneStep p cf k0 zm j (laneRun p cf k0 zm st t j)).2.2
    unfold laneStep
    split
    · exact wb_chalZ ih (by omega) hy hB
    · exact (wb_chal ih (by omega)).mono (by omega)

theorem wb_lv0Z {E B P C : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {zl : Zk.Level0}
    {initCh : Array F256} {rho : F256 × F256} {st : LState} {t : Transcript} (h : Wb E B P t)
    (hc : (capBytes p.recursiveCaps[0]!).size ≤ C) (hB : P + C + 247 ≤ B) :
    Wb E B 0 (lv0Z p cf sch zl initCh rho st t).2 := by
  unfold lv0Z
  exact wb_close (wb_query (wb_ood (wb_ood (h.ap (pBytes_le hc)) (by omega) (by omega)) (by omega) (by omega))
    (by omega) (by omega)) (by omega)

/-- **The masked Ligerito's rounds**: plain's start and `LABEL_OPEN`, its head, the lane rounds, level 0, and plain's
level loop. -/
theorem wb_outZ {E B C NY NB : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {T : F128}
    {cap : Array ByteArray} {zl : Zk.Level0} {t : Transcript} (h : Wb E B 0 t) (hcap : (capBytes cap).size ≤ C)
    (hc : ∀ j : ℕ, (capBytes p.recursiveCaps[j]!).size ≤ C) (hy : p.finalProof.yr.size ≤ NY)
    (hyb : (ybarFlat zl.msgs.ybar).size ≤ NB) (hB : C + 32 * NY + 16 * NB + 900 ≤ B) :
    Wb E B 0 (outZ p cf sch T cap zl t).2.2 := by
  have hs : Wb E B (0 + 191 + 32 + (C + 31) + 191) (startZ cap T t) := by
    unfold startZ start
    exact (((h.ap (pLabel_le (by decide +kernel))).ap (pF_le _)).ap (pBytes_le hcap)).ap
      (pLabel_le (by decide +kernel))
  have hpre : Wb E B 0 (preZ p cf sch T cap zl t).2.2 := by
    unfold preZ
    exact wb_lv0Z (wb_laneRun (wb_head hs (by omega) (by omega)) hyb (by omega) _) (hc 0) (by omega)
  exact wb_loop (x := x₀Z p cf sch T cap zl t) hpre hc hy (by omega) _

end FlockComplete.L0
