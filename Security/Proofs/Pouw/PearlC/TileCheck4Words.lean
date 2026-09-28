import Proofs.Pouw.PearlC.TileCheck4
import Proofs.Pouw.PearlC.Block4Sem
import Proofs.Pouw.PearlC.TileProtocol
import Proofs.Pouw.PearlC.TileGamma

/-!
# `pc4`'s tile check computes the words (proofs)

The row units' words read in the protocol's terms (`pc4Sem`), under the named hypotheses on which the table and the
protocol's word functions agree only conditionally:

* `Pc4OpsAgree ops`: core's primitives agree with the protocol's word functions where both are defined. At core's
  instance `coreOps`, `Bf16ToF32`'s and `I32Le`'s hold by their definitions (`coreOps_bf16w`, `coreOps_le`), and all
  of them follow from core's definitions (`CoreOps.pc4OpsAgree_coreOps`);
* `Row4OK`: a row's BF16 words widen to step words of its values, and its `α` (positive) and `β` are step words. It
  is no hypothesis of L3: the widening follows from the rules' word clause (every word a finite widened BF16 word,
  `widen_of_word`), and `α` and `β` from `RowAdmit4` by magnitudes (`RowStats`, `TileCheckWords`), at a credited row
  by `Fp4Sem.RowOK`, which an accepted tile's check also asks of each of its weight rows.

On these a row unit's codes and scale bytes are `formRow`'s at `pc4Sem`'s `α` and `β` (`row4A_form`, `row4B_form`, so
`form4A`/`form4B` are `unitForm`'s `A′`/`B̃`), and its `P_A` and `P_B` rows are the FP4 peel's (`row4A_paWords`,
`row4B_pbWords`). `entry4_words` chains RowA `i` and RowB `j` into their tile (`TileCheck4`'s `chain4_*`): where the
three units are correct and the peel's rows exist, entry `(i, j)`'s `C̃` and `U` are `devFp4Sm120`'s ticket and useful
value (`tile4_devFp4`).

On the whole table a gate holds its assignment's low bits at its width (`val_in4`, `val_tile4`), so where the public
gates hold the anchors (`anchors4`) and the registered gates `rows4`'s values, an entry's registered words are the
record's (`entry4_accepts`), at two rows of finite words with `Row4OK` values where the peel's rows exist. A row's
gates hold its words' high halves, `pc4Sem`'s BF16 words up to the sign of zero (`zeroSign16_of_wordQ`), which no row
unit reads (`formRow_table`). `TileCheckWords` composes these into
L3's words for the plain `pc4` Program on a domain of whole-tile shapes (`Shape4`); D₄'s are
(`shape4_of_pearlCShapeFp4`). The peel's rows exist at two `Row4OK` rows (`PeelExists4`). The cap is not part of it.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit
  Pouw.PearlC.PeelFp4Proofs

/-! ## `pc4Sem`'s operands -/

section Sem4

variable (ops : RowOps)

/-- `pc4Sem`'s `β` of a row: `beta4`'s word read as a value, `0` where it is not finite. -/
def beta4V {k : ℕ} (X : Fin k → ℚ) : ℚ :=
  if (wordQ (beta4 ops (encRow16 k X) k)).isSome then wordVal (beta4 ops (encRow16 k X) k) else 0

/-- `pc4Sem`'s `A′`: row `i` of the activations' values formed at its `α` and `β` with `E_A`'s line `i` and `F_A`. -/
def form4A {m n k : ℕ} (z : Lines4 m n k) (act : Codes) : Op4 m := fun i =>
  formRow (wordVal (alpha4 ops (encRow16 k (wordsVal m k act i)) k)) (beta4V ops (wordsVal m k act i))
    (wordsVal m k act i) (z.EA i) z.FA

/-- `pc4Sem`'s `B̃`: row `j` of the weights' values formed with `E_B`'s line `j` and `F_B`. -/
def form4B {m n k : ℕ} (z : Lines4 m n k) (wt : Codes) : Op4 n := fun j =>
  formRow (wordVal (alpha4 ops (encRow16 k (wordsVal n k wt j)) k)) (beta4V ops (wordsVal n k wt j))
    (wordsVal n k wt j) (z.EB j) z.FB

/-- A row's `Ẽ` at `a_E = e4m3(β)`, the row `peelFp4` reads for it. -/
def be4 {k : ℕ} (X : Fin k → ℚ) (e : Fin 32 → Nib) : Fin 32 → ℕ :=
  eTilde e (e4m3 (beta4V ops X))

end Sem4

/-- Where `β`'s word is a step word, `pc4Sem`'s `β` is its value. -/
theorem beta4V_eq {ops : RowOps} {k : ℕ} {X : Fin k → ℚ} (h : AccumProofs.StepWord (beta4 ops (encRow16 k X) k)) :
    beta4V ops X = wordVal (beta4 ops (encRow16 k X) k) := by
  unfold beta4V
  rw [ite_eq_left (by rw [PeelExactProofs.wordQ_fields (AccumProofs.finite?_exp h.2.2)]; rfl)]

theorem getD_lt32 {o : Option ℕ} (h : ∀ w, o = some w → w < 2 ^ 32) : o.getD 0 < 2 ^ 32 := by
  cases o with
  | none => norm_num
  | some w => exact h w rfl

theorem ch4_lt {x y : List Nib} {sx sy : List (Fin 256)} {w : ℕ} (h : ch4 sm120Nvf4 x y sx sy = some w) :
    w < 2 ^ 32 := by
  obtain ⟨c, hc, hf⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some hf
  exact cTilde4_lt hc

theorem gram4_eq {m n k : ℕ} (z : Lines4 m n k) {s q : ℕ} (hs : s < 32) (hq : q < 32) :
    gram4 z s q = (ch4 sm120Nvf4 (colF z.FB ⟨s, hs⟩) (colF z.FA ⟨q, hq⟩) (onesScales sm120Nvf4 k)
      (onesScales sm120Nvf4 k)).getD 0 := by
  unfold gram4
  exact dite_eq_left ⟨hs, hq⟩

theorem gram4_lt {m n k : ℕ} (z : Lines4 m n k) (s q : ℕ) : gram4 z s q < 2 ^ 32 := by
  unfold gram4
  split_ifs
  · exact getD_lt32 fun w hw => ch4_lt hw
  · norm_num

theorem nib_lt4 (c : Nib) : c.val < 2 ^ 4 := lt_of_lt_of_eq c.isLt (by norm_num)

/-! ## An entry's words -/

section Entry4

variable {ops : RowOps} (hops : Pc4OpsAgree ops)
include hops

/-- **Entry `(i, j)`'s words are the record's.** Where RowA `i`, RowB `j` and their tile are correct and the inputs
hold the BF16 words of row `i` of the activations and row `j` of the weights, their noise lines, the bases and `G`:
under `Row4OK` at the two rows, and where the peel's rows exist, `C̃_ij` and `U_ij` are `devFp4Sm120`'s
ticket and useful value on `pc4Sem`'s `A′` and `B̃` (`form4A`, `form4B`), the bases and the rows' `Ẽ`. -/
theorem entry4_words {M K N TM TN nw : ℕ} (hnw : 0 < nw) (hK : K = 64 * nw)
    (hTM : TM ∣ M) (hTN : TN ∣ N) (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript)
    (z : Lines4 M N K) (act wt : Codes) (i : Fin M) (j : Fin N)
    (hXT : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_T ops M K N TM TN) (tile_index hTM hTN i.isLt j.isLt).1⟩)
    (hXA : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_A ops M K N TM TN) i.isLt⟩)
    (hXB : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_B ops M K N TM TN) j.isLt⟩)
    (hxa : ∀ l < K, ZeroSign16 (val X ((i : ℕ) * K + l)) (encRow16 K (wordsVal M K act i) l))
    (hea : ∀ l : Fin 32, val X (eaAt M K + 32 * (i : ℕ) + l) = (z.EA i l).val)
    (hxb : ∀ l < K, ZeroSign16 (val X (bAt M K + (j : ℕ) * K + l)) (encRow16 K (wordsVal N K wt j) l))
    (heb : ∀ l : Fin 32, val X (ebAt M N K + 32 * (j : ℕ) + l) = (z.EB j l).val)
    (hfa : ∀ (t : Fin K) (l : Fin 32), val X (faAt M N K + 32 * (t : ℕ) + l) = (z.FA t l).val)
    (hfb : ∀ (t : Fin K) (l : Fin 32), val X (fbAt M N K + 32 * (t : ℕ) + l) = (z.FB t l).val)
    (hg : ∀ s < 32, ∀ q < 32, val X (gAt M N K + 32 * s + q) = gram4 z s q)
    (hA : Row4OK ops K (wordsVal M K act i)) (hB : Row4OK ops K (wordsVal N K wt j))
    {c : ℕ} {pa pb : List ℕ}
    (hr : peelRows4 sm120Nvf4 (form4A ops z act) (form4B ops z wt) z.FA z.FB
      (fun i => be4 ops (wordsVal M K act i) (z.EA i)) (fun j => be4 ops (wordsVal N K wt j) (z.EB j)) i j =
        some (c, pa, pb)) :
    wordQ (val X (cGlob4 M N K TM TN i j)) = devFp4Sm120.ticket (form4A ops z act) (form4B ops z wt) i j ∧
      wordQ (val X (uGlob4 M N K TM TN i j)) = devFp4Sm120.useful (form4A ops z act) (form4B ops z wt) z.FA z.FB
        (fun i => be4 ops (wordsVal M K act i) (z.EA i)) (fun j => be4 ops (wordsVal N K wt j) (z.EB j)) i j := by
  obtain ⟨a, b, af, bf, gL, ha, hb, -, haf, hbf, hgg, hpa, hpb⟩ := peelRows4_spec hr
  have ha' := ha
  have hb' := hb
  have hpa' := hpa
  have hpb' := hpb
  simp only [form4A, form4B, be4, beta4V_eq hA.beta, beta4V_eq hB.beta] at ha' hb' hpa' hpb'
  obtain ⟨hgl, hgrow⟩ := PeelExactProofs.mapM_ofFn _ [] hgg
  have h16 : 16 ∣ K := ⟨4 * nw, by omega⟩
  have hk : 64 ∣ K := ⟨nw, hK⟩
  have h8 : 8 ≤ K := by omega
  have hinA : ∀ a < rowIn K false,
      val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) a = val X (portA8 M N K i a) :=
    fun a ha => cutA4_input ops M K N TM TN X i.isLt ha
  have hinB : ∀ a < rowIn K true,
      val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) a = val X (portB8 M N K j a) :=
    fun a ha => cutB4_input ops M K N TM TN X j.isLt ha
  have hrA := rowIn_false K
  have hrB := rowIn_true K
  have hxA : ∀ l < K, ZeroSign16 (val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) l)
      (encRow16 K (wordsVal M K act i) l) :=
    fun l hl => by rw [hinA l (by omega), portA8_x hl]; exact hxa l hl
  have heA : ∀ l : Fin 32, val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) (K + l) = (z.EA i l).val :=
    fun l => (hinA _ (by have := l.isLt; omega)).trans (by rw [portA8_e l.isLt, hea l])
  have hfA : ∀ (t : Fin K) (l : Fin 32),
      val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) (K + 32 + 32 * t + l) = (z.FA t l).val :=
    fun t l => (hinA _ (by have := t.isLt; have := l.isLt; omega)).trans
      (by rw [portA8_f t.isLt l.isLt, hfa t l])
  have hbA : ∀ (t : Fin K) (q : Fin 32),
      val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) (33 * K + 32 + 32 * t + q) = (z.FB t q).val :=
    fun t q => (hinA _ (by have := t.isLt; have := q.isLt; omega)).trans (by rw [portA8_b q.isLt, hfb t q])
  have hxB : ∀ l < K, ZeroSign16 (val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) l)
      (encRow16 K (wordsVal N K wt j) l) :=
    fun l hl => by rw [hinB l (by omega), portB8_x hl]; exact hxb l hl
  have heB : ∀ l : Fin 32, val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) (K + l) = (z.EB j l).val :=
    fun l => (hinB _ (by have := l.isLt; omega)).trans (by rw [portB8_e l.isLt, heb l])
  have hfB : ∀ (t : Fin K) (l : Fin 32),
      val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) (K + 32 + 32 * t + l) = (z.FB t l).val :=
    fun t l => (hinB _ (by have := t.isLt; have := l.isLt; omega)).trans
      (by rw [portB8_f t.isLt l.isLt, hfb t l])
  have hbB : ∀ (t : Fin K) (q : Fin 32),
      val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) (33 * K + 32 + 32 * t + q) = (z.FA t q).val :=
    fun t q => (hinB _ (by have := t.isLt; have := q.isLt; omega)).trans
      (by rw [portB8_b t.isLt q.isLt, hfa t q])
  have hgB : ∀ s < 32, ∀ q < 32, val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j)
      (65 * K + 32 + 32 * s + q) = (gL.getD s []).getD q 0 % 2 ^ 32 := fun s hs q hq => by
    have h := (PeelExactProofs.mapM_ofFn _ 0 (hgrow ⟨s, hs⟩)).2 ⟨q, hq⟩
    refine (hinB _ (by omega)).trans ?_
    rw [portB8_g hs hq, hg s hs q hq, gram4_eq z hs hq, Nat.mod_eq_of_lt (ch4_lt h), h]
    rfl
  have cA := (whole4 ops M K N TM TN).cut_correct X (blockA4 ops M K N) i
  have cB := (whole4 ops M K N TM TN).cut_correct X (blockB4 ops M K N) j
  obtain ⟨hSA, hCA⟩ := row4A_form hops h8 h16 _ cA hA hxA heA hfA ha'
  obtain ⟨hSB, hCB⟩ := row4B_form hops h8 h16 _ cB hB hxB heB hfB hb'
  have paA := row4A_paWords hops hnw hK _ cA hA hxA heA hfA hbA ha' haf hpa'
  have pbB := row4B_pbWords hops hnw hK _ cB hB hxB heB hfB hbB hb' hbf hgl hgB hpb'
  have hPA : ∀ q < 64, val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) (paOut4 K q) = pa.getD q 0 := by
    intro q hq
    unfold paOut4
    split_ifs with h
    · exact (paA q h).1
    · have := (paA (q - 32) (by omega)).2
      rwa [show 32 + (q - 32) = q by omega] at this
  have hPB : ∀ q < 64, val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) (pbOut4 K q) = pb.getD q 0 := by
    intro q hq
    unfold pbOut4
    split_ifs with h
    · exact (pbB q h).1
    · have := (pbB (q - 32) (by omega)).2
      rwa [show 32 + (q - 32) = q by omega] at this
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM i.isLt)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN j.isLt)
  obtain ⟨h1, h2⟩ := tile4_devFp4 _
    ((whole4 ops M K N TM TN).cut_correct X (blockT4 M K N TM TN) (i / TM * (N / TN) + j / TN))
    (form4A ops z act) (form4B ops z wt) z.FA z.FB (fun i => be4 ops (wordsVal M K act i) (z.EA i))
    (fun j => be4 ops (wordsVal N K wt j) (z.EB j)) i j ha hb hti htj (by omega) (formRow_length h16 ha).1
    (fun l hl => (chain4_A ops M K N TM TN X hTM hTN i.isLt j.isLt hk hXA hl).trans (hCA l hl))
    (fun l hl => (chain4_SA ops M K N TM TN X hTM hTN i.isLt j.isLt hk hXA hl).trans (hSA l hl))
    (fun l hl => (chain4_B ops M K N TM TN X hTM hTN i.isLt j.isLt hk hXB hl).trans (hCB l hl))
    (fun l hl => (chain4_SB ops M K N TM TN X hTM hTN i.isLt j.isLt hk hXB hl).trans (hSB l hl)) hr
    (fun q hq => (chain4_PA ops M K N TM TN X hTM hTN i.isLt j.isLt hk hXA hq).trans (hPA q hq))
    (fun q hq => (chain4_PB ops M K N TM TN X hTM hTN i.isLt j.isLt hk hXB hq).trans (hPB q hq))
  rw [chain4_C ops M K N TM TN X hTM hTN i.isLt j.isLt hk (by omega) hXT,
    chain4_U ops M K N TM TN X hTM hTN i.isLt j.isLt hk hXT]
  exact ⟨h1, h2⟩

end Entry4

/-! ## The anchors and the registered values, region by region -/

section Regions4

variable {m n k : ℕ} (z : Lines4 m n k)

theorem anchors4_ea {i l : ℕ} (hi : i < m) (hl : l < 32) :
    anchors4 z (eaAt m k + 32 * i + l) = (z.EA ⟨i, hi⟩ ⟨l, hl⟩).val := by
  obtain ⟨h1, -⟩ := layout8 m n k
  have hq : (eaAt m k + 32 * i + l - eaAt m k) / 32 = i := by omega
  have hr : (eaAt m k + 32 * i + l - eaAt m k) % 32 = l := by omega
  unfold anchors4
  rw [ite_eq_right (by omega), ite_eq_left (by omega)]
  simp only [hq, hr, hi, ↓reduceDIte, encNibs, hl]

theorem anchors4_eb {j l : ℕ} (hj : j < n) (hl : l < 32) :
    anchors4 z (ebAt m n k + 32 * j + l) = (z.EB ⟨j, hj⟩ ⟨l, hl⟩).val := by
  obtain ⟨h1, h2, h3, -⟩ := layout8 m n k
  have hq : (ebAt m n k + 32 * j + l - ebAt m n k) / 32 = j := by omega
  have hr : (ebAt m n k + 32 * j + l - ebAt m n k) % 32 = l := by omega
  unfold anchors4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega)]
  simp only [hq, hr, hj, ↓reduceDIte, encNibs, hl]

theorem anchors4_fa {t l : ℕ} (ht : t < k) (hl : l < 32) :
    anchors4 z (faAt m n k + 32 * t + l) = (z.FA ⟨t, ht⟩ ⟨l, hl⟩).val := by
  obtain ⟨h1, h2, h3, h4, -⟩ := layout8 m n k
  have hq : (faAt m n k + 32 * t + l - faAt m n k) / 32 = t := by omega
  have hr : (faAt m n k + 32 * t + l - faAt m n k) % 32 = l := by omega
  unfold anchors4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left (by omega)]
  simp only [hq, hr, ht, ↓reduceDIte, encNibs, hl]

theorem anchors4_fb {t l : ℕ} (ht : t < k) (hl : l < 32) :
    anchors4 z (fbAt m n k + 32 * t + l) = (z.FB ⟨t, ht⟩ ⟨l, hl⟩).val := by
  obtain ⟨h1, h2, h3, h4, h5, -⟩ := layout8 m n k
  have hq : (fbAt m n k + 32 * t + l - fbAt m n k) / 32 = t := by omega
  have hr : (fbAt m n k + 32 * t + l - fbAt m n k) % 32 = l := by omega
  unfold anchors4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_left (by omega)]
  simp only [hq, hr, ht, ↓reduceDIte, encNibs, hl]

theorem anchors4_g {s q : ℕ} (hs : s < 32) (hq : q < 32) :
    anchors4 z (gAt m n k + 32 * s + q) = gram4 z s q := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := layout8 m n k
  have hq' : (gAt m n k + 32 * s + q - gAt m n k) / 32 = s := by omega
  have hr' : (gAt m n k + 32 * s + q - gAt m n k) % 32 = q := by omega
  unfold anchors4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega), hq', hr']

end Regions4

section Rows4

variable {M N K TM TN : ℕ} (act wt tk us : Codes)

theorem rows4_a {i l : ℕ} (hi : i < M) (hl : l < K) :
    rows4 M N K TM TN act wt tk us (i * K + l) = act i l / 2 ^ 16 := by
  have hK : 0 < K := by omega
  unfold rows4
  rw [ite_eq_left (show i * K + l < eaAt M K from mul_add_lt hi hl),
    show (i * K + l) / K = i by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (i * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem rows4_b {j l : ℕ} (hj : j < N) (hl : l < K) :
    rows4 M N K TM TN act wt tk us (bAt M K + j * K + l) = wt j l / 2 ^ 16 := by
  have hK : 0 < K := by omega
  obtain ⟨h1, h2, -⟩ := layout8 M N K
  have := mul_add_lt hj hl
  unfold rows4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    show bAt M K + j * K + l - bAt M K = j * K + l by omega,
    show (j * K + l) / K = j by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (j * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

/-- **A finite word's high half is its value's BF16 word**, or `−0` (`0x8000`) where the value is `0` (`ZeroSign16`):
a finite word is its value's `f32Word` or `−0` (`zeroSign_of_wordQ`), and a gate holds the low 16 bits. -/
theorem zeroSign16_of_wordQ {w : ℕ} (h : (wordQ w).isSome) :
    ZeroSign16 (w / 2 ^ 16 % 2 ^ 16) (bf16WordOf ((wordQ w).getD 0)) := by
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  rw [← wordQ_mod32] at hx
  have hd : w / 2 ^ 16 % 2 ^ 16 = w % 2 ^ 32 / 2 ^ 16 := by
    rw [show (2 : ℕ) ^ 32 = 2 ^ 16 * 2 ^ 16 by norm_num, Nat.mod_mul_right_div_self]
  rw [hd, ← wordQ_mod32, hx, Option.getD_some, bf16WordOf]
  rcases (zeroSign_of_wordQ (Nat.mod_lt _ (by norm_num)) hx).1 with h1 | ⟨h1, h2⟩
  · exact Or.inl (by rw [h1, Nat.shiftRight_eq_div_pow])
  · exact Or.inr ⟨by rw [h1]; rfl, by rw [h2]; rfl⟩

/-- **A finite widened BF16 word widens back to itself**: its low 16 bits are zero, so its value's `f32Word` is its
value's BF16 word shifted left by 16 (`bf16w`), a step word of that value. This is `Row4OK`'s `widen` at a row that
`Fp4Sem.RowOK` passes. -/
theorem widen_of_word {ops : RowOps} (hops : Pc4OpsAgree ops) {w : ℕ} (hq : (wordQ w).isSome) (h16 : w % 2 ^ 16 = 0) :
    AccumProofs.StepWord (ops.bf16ToF32 (bf16WordOf ((wordQ w).getD 0))) ∧
      wordVal (ops.bf16ToF32 (bf16WordOf ((wordQ w).getD 0))) = (wordQ w).getD 0 := by
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hq
  rw [← wordQ_mod32] at hx
  obtain ⟨hz, hs⟩ := zeroSign_of_wordQ (Nat.mod_lt _ (by norm_num)) hx
  rw [← wordQ_mod32, hx, Option.getD_some]
  have hlow : f32Word x % 2 ^ 16 = 0 := by
    rcases hz with h1 | ⟨-, h2⟩
    · rw [← h1, Nat.mod_mod_of_dvd _ (by norm_num), h16]
    · rw [h2, Nat.zero_mod]
  have hw : ops.bf16ToF32 (bf16WordOf x) = f32Word x := by
    rw [hops.bf16w _ (bf16WordOf_lt x), bf16WordOf, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
    exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hlow)
  rw [hw]
  refine ⟨hs, ?_⟩
  rcases hz with h1 | ⟨h1, h2⟩
  · rw [← h1]; simp only [wordVal, hx]
  · rw [h1, PeelExactProofs.wordQ_negZero] at hx
    obtain rfl := (Option.some.inj hx).symm
    rw [h2]; simp [wordVal, Freeze.wordQ_zero]

/-- The registered value on a tile gate of entry `(i, j)`'s block: its ticket word at `C̃`, its useful word at `U`. -/
theorem rows4_block {i j r : ℕ} (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) (hr : r < block4Len K) :
    rows4 M N K TM TN act wt tk us
        (tile4 M N K TM TN (i / TM * (N / TN) + j / TN) (entry4Base K TM TN (i % TM) (j % TN) + r)) =
      if r = K / 64 - 1 then tk i j else if r = block4Len K - 1 then us i j else 0 := by
  obtain ⟨-, hie, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hbl := block4Len_pos K
  have hL : tile4N K TM TN - tile4In K TM TN = TM * TN * block4Len K := by unfold tile4N; omega
  have he := mul_add_lt (L := TN) hti htj
  have hb := mul_add_lt (L := block4Len K) he hr
  have hLp : 0 < TM * TN * block4Len K := by omega
  have hg : tile4 M N K TM TN (i / TM * (N / TN) + j / TN) (entry4Base K TM TN (i % TM) (j % TN) + r) =
      tiles4 M N K + ((i / TM * (N / TN) + j / TN) * (TM * TN * block4Len K) +
        ((i % TM * TN + j % TN) * block4Len K + r)) := by
    unfold tile4 tiles4 entry4Base; rw [hL]; omega
  have hq1 : ((i / TM * (N / TN) + j / TN) * (TM * TN * block4Len K) +
      ((i % TM * TN + j % TN) * block4Len K + r)) / (TM * TN * block4Len K) = i / TM * (N / TN) + j / TN := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hLp, Nat.div_eq_of_lt hb, Nat.zero_add]
  have hr1 : ((i / TM * (N / TN) + j / TN) * (TM * TN * block4Len K) +
      ((i % TM * TN + j % TN) * block4Len K + r)) % (TM * TN * block4Len K) =
        (i % TM * TN + j % TN) * block4Len K + r := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  have hq2 : ((i % TM * TN + j % TN) * block4Len K + r) / block4Len K = i % TM * TN + j % TN := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hbl, Nat.div_eq_of_lt hr, Nat.zero_add]
  have hr2 : ((i % TM * TN + j % TN) * block4Len K + r) % block4Len K = r := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]
  have htn : 0 < TN := by omega
  have hq3 : (i % TM * TN + j % TN) / TN = i % TM := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ htn, Nat.div_eq_of_lt htj, Nat.zero_add]
  have hr3 : (i % TM * TN + j % TN) % TN = j % TN := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt htj]
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := layout8 M N K
  have ht4 : in8 M N K ≤ tiles4 M N K := by unfold tiles4; omega
  unfold rows4 entry4
  rw [hg, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]
  simp only [Nat.add_sub_cancel_left, hL, hq1, hr1, hq2, hr2, hq3, hr3, hie, hje]

theorem rows4_C {i j : ℕ} (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) (hK : 0 < K / 64) :
    rows4 M N K TM TN act wt tk us (cGlob4 M N K TM TN i j) = tk i j := by
  unfold cGlob4
  rw [c4Gate_eq K TM TN hK, rows4_block act wt tk us hTM hTN hi hj (by unfold block4Len; omega), ite_eq_left rfl]

theorem rows4_U {i j : ℕ} (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) :
    rows4 M N K TM TN act wt tk us (uGlob4 M N K TM TN i j) = us i j := by
  have hbl := block4Len_pos K
  have hug : u4Gate K TM TN (i % TM) (j % TN) = entry4Base K TM TN (i % TM) (j % TN) + (block4Len K - 1) := by
    unfold u4Gate; omega
  unfold uGlob4
  rw [hug, rows4_block act wt tk us hTM hTN hi hj (by omega), ite_eq_right (by unfold block4Len; omega),
    ite_eq_left rfl]

end Rows4

/-! ## The gates' widths -/

theorem tile4Gate_w32 {K TM TN g : ℕ} (hg : tile4In K TM TN ≤ g) : (tile4Gate K TM TN g).w = 32 := by
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hs2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  have hk : (TM + TN) * row4 K ≤ tile4In K TM TN := by unfold tile4In; omega
  unfold tile4Gate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (not_lt.2 hg)]
  dsimp only
  unfold entry4Gate
  split_ifs <;> rfl

section Widths4

variable (ops : RowOps) {M K N TM TN : ℕ}

theorem inW4_a {g : ℕ} (h : g < eaAt M K) : inW4 M N K g = 16 := by
  unfold inW4; rw [ite_eq_left h]

theorem inW4_e {g : ℕ} (h1 : eaAt M K ≤ g) (h2 : g < bAt M K) : inW4 M N K g = 4 := by
  unfold inW4; rw [ite_eq_right (not_lt.2 h1), ite_eq_left h2]

theorem inW4_b {g : ℕ} (h1 : bAt M K ≤ g) (h2 : g < ebAt M N K) : inW4 M N K g = 16 := by
  obtain ⟨e1, -⟩ := layout8 M N K
  unfold inW4; rw [ite_eq_right (by omega), ite_eq_right (not_lt.2 h1), ite_eq_left h2]

theorem inW4_f {g : ℕ} (h1 : ebAt M N K ≤ g) (h2 : g < gAt M N K) : inW4 M N K g = 4 := by
  obtain ⟨e1, e2, -⟩ := layout8 M N K
  unfold inW4; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (not_lt.2 h1), ite_eq_left h2]

theorem inW4_g {g : ℕ} (h1 : gAt M N K ≤ g) : inW4 M N K g = 32 := by
  obtain ⟨e1, e2, e3, e4, e5, -⟩ := layout8 M N K
  unfold inW4; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]

theorem in4_le_N : in8 M N K ≤ (whole4 ops M K N TM TN).spec.toCircuit.N := by
  rw [whole4_N]; omega

/-- An input gate's value is its assignment's low `inW4` bits. -/
theorem val_in4 (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {g v : ℕ} (hg : g < in8 M N K)
    (hgN : g < (whole4 ops M K N TM TN).spec.toCircuit.N) (h : X ⟨g, hgN⟩ = BitVec.ofNat _ v) :
    val X g = v % 2 ^ inW4 M N K g := by
  rw [val_bv X hgN h]
  show v % 2 ^ ((whole4 ops M K N TM TN).gate g).w = _
  unfold Whole.gate
  rw [ite_eq_left (show g < (whole4 ops M K N TM TN).nIn from hg)]
  rfl

/-- A tile gate's value is its assignment's low 32 bits. -/
theorem val_tile4 (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {e a v : ℕ}
    (he : e < M / TM * (N / TN)) (ha1 : tile4In K TM TN ≤ a) (ha2 : a < tile4N K TM TN)
    (h : X ⟨_, tile4_ltN ops M K N TM TN he ha1 ha2⟩ = BitVec.ofNat _ v) :
    val X (tile4 M N K TM TN e a) = v % 2 ^ 32 := by
  rw [val_bv X _ h]
  show v % 2 ^ ((whole4 ops M K N TM TN).gate (tile4 M N K TM TN e a)).w = _
  rw [tile4_eq ops M K N TM TN, (whole4 ops M K N TM TN).gate_inst (whole4_T ops M K N TM TN) he ha1 ha2]
  show v % 2 ^ (tile4Gate K TM TN a).w = _
  rw [tile4Gate_w32 ha1]

omit ops in
theorem c4Gate_bounds {i j : ℕ} (hi : i < TM) (hj : j < TN) (hK : 0 < K / 64) :
    tile4In K TM TN ≤ c4Gate K TM TN i j ∧ c4Gate K TM TN i j < tile4N K TM TN := by
  have hb := entry4_bounds K TM TN hi hj (r := K / 64 - 1) (by unfold block4Len; omega)
  rw [c4Gate_eq K TM TN hK]; exact hb

omit ops in
theorem u4Gate_bounds {i j : ℕ} (hi : i < TM) (hj : j < TN) :
    tile4In K TM TN ≤ u4Gate K TM TN i j ∧ u4Gate K TM TN i j < tile4N K TM TN := by
  have hbl := block4Len_pos K
  have hb := entry4_bounds K TM TN hi hj (r := block4Len K - 1) (by omega)
  have hug : u4Gate K TM TN i j = entry4Base K TM TN i j + (block4Len K - 1) := by unfold u4Gate; omega
  rw [hug]; exact hb

end Widths4

/-! ## An accepted entry -/

/-- A whole-tile `pc4` shape: `64 ∣ m`, `64 ∣ n` (`pc4.Shape.check`), and `k` a positive multiple of 64. -/
def Shape4 (sh : Shape) : Prop := 64 ∣ sh.m ∧ 64 ∣ sh.n ∧ ∃ nw, 0 < nw ∧ sh.k = 64 * nw

/-- **The peel's rows exist at entry `(i, j)`** of a unit with lines `z`, activations `act` and weights `wt`
(`peelRows4`, which `devFp4Sm120`'s useful value reads; where it fails the useful value is `none`). It holds at two
`Row4OK` rows with `64 ∣ k ≤ 2^16` (`PeelExists.pc4EntryOK_of_rows`), so it is no hypothesis of L3. -/
structure Pc4EntryOK (ops : RowOps) {m n k : ℕ} (z : Lines4 m n k) (act wt : Codes) (i : Fin m) (j : Fin n) :
    Prop where
  peel : (peelRows4 sm120Nvf4 (form4A ops z act) (form4B ops z wt) z.FA z.FB
    (fun i => be4 ops (wordsVal m k act i) (z.EA i)) (fun j => be4 ops (wordsVal n k wt j) (z.EB j)) i j).isSome

section Accept4

variable {ops : RowOps} (hops : Pc4OpsAgree ops)
include hops

/-- **An accepted entry's registered words are the record's.** On `pc4`'s whole table at a whole-tile shape: where
every unit in the cone of a set `T` holding entry `(i, j)`'s `C̃` and `U` is correct, the public gates hold the anchors
of the lines `z` and the registered gates the activations `act`, the weights `wt` and the words `tk`, `us`, then at
rows `i` of `act` and `j` of `wt` of finite words whose values are `Row4OK`, where the peel's rows exist at `(i, j)`,
the registered words `tk i j` and `us i j` read as `devFp4Sm120`'s ticket and useful value on `pc4Sem`'s operands. A
row's gates hold its words' high halves, `pc4Sem`'s BF16 words up to the sign of zero (`zeroSign16_of_wordQ`). -/
theorem entry4_accepts {M K N nw : ℕ} (hnw : 0 < nw) (hK : K = 64 * nw)
    (hM : 64 ∣ M) (hN : 64 ∣ N) (X : (whole4 ops M K N 64 64).spec.toCircuit.Transcript)
    (z : Lines4 M N K) (act wt tk us : Codes) (i : Fin M) (j : Fin N)
    {T : Finset (Fin (whole4 ops M K N 64 64).spec.toCircuit.N)}
    (hT : ∀ g : Fin (whole4 ops M K N 64 64).spec.toCircuit.N,
      (g.val = cGlob4 M N K 64 64 i j ∨ g.val = uGlob4 M N K 64 64 i j) → g ∈ T)
    (hcone : ∀ v ∈ (whole4 ops M K N 64 64).part.cone T, (whole4 ops M K N 64 64).part.Correct X v)
    (hpub : ∀ g (hg : g < (whole4 ops M K N 64 64).spec.toCircuit.N),
      ((eaAt M K ≤ g ∧ g < bAt M K) ∨ (ebAt M N K ≤ g ∧ g < in8 M N K)) → X ⟨g, hg⟩ = BitVec.ofNat _ (anchors4 z g))
    (hreg : ∀ g (hg : g < (whole4 ops M K N 64 64).spec.toCircuit.N),
      (g < eaAt M K ∨ (bAt M K ≤ g ∧ g < ebAt M N K) ∨
        ∃ i < M, ∃ j < N, g = cGlob4 M N K 64 64 i j ∨ g = uGlob4 M N K 64 64 i j) →
      X ⟨g, hg⟩ = BitVec.ofNat _ (rows4 M N K 64 64 act wt tk us g))
    (hwa : ∀ l < K, (wordQ (act i l)).isSome) (hwb : ∀ l < K, (wordQ (wt j l)).isSome)
    (hA : Row4OK ops K (wordsVal M K act i)) (hB : Row4OK ops K (wordsVal N K wt j))
    (hP : (peelRows4 sm120Nvf4 (form4A ops z act) (form4B ops z wt) z.FA z.FB
      (fun i => be4 ops (wordsVal M K act i) (z.EA i)) (fun j => be4 ops (wordsVal N K wt j) (z.EB j)) i j).isSome) :
    wordQ (tk i j) = devFp4Sm120.ticket (form4A ops z act) (form4B ops z wt) i j ∧
      wordQ (us i j) = devFp4Sm120.useful (form4A ops z act) (form4B ops z wt) z.FA z.FB
        (fun i => be4 ops (wordsVal M K act i) (z.EA i)) (fun j => be4 ops (wordsVal N K wt j) (z.EB j)) i j := by
  obtain ⟨⟨c, pa, pb⟩, hr⟩ := Option.isSome_iff_exists.1 hP
  have hK0 : 0 < K / 64 := by subst hK; omega
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := layout8 M N K
  have hNin := in4_le_N ops (M := M) (K := K) (N := N) (TM := 64) (TN := 64)
  have hea0 : eaAt M K = M * K := rfl
  have hxa : ∀ l < K, ZeroSign16 (val X ((i : ℕ) * K + l)) (encRow16 K (wordsVal M K act i) l) := fun l hl => by
    have := mul_add_lt i.isLt hl
    have hg : (i : ℕ) * K + l < eaAt M K := by omega
    rw [val_in4 ops X (by omega) (by omega) (hreg _ _ (Or.inl hg)), rows4_a act wt tk us i.isLt hl, inW4_a hg,
      encRow16, dite_eq_left hl]
    exact zeroSign16_of_wordQ (hwa l hl)
  have hxb : ∀ l < K, ZeroSign16 (val X (bAt M K + (j : ℕ) * K + l)) (encRow16 K (wordsVal N K wt j) l) :=
    fun l hl => by
    have := mul_add_lt j.isLt hl
    rw [val_in4 ops X (by omega) (by omega) (hreg _ _ (Or.inr (Or.inl ⟨by omega, by omega⟩))),
      rows4_b act wt tk us j.isLt hl, inW4_b (by omega) (by omega), encRow16, dite_eq_left hl]
    exact zeroSign16_of_wordQ (hwb l hl)
  have hea : ∀ l : Fin 32, val X (eaAt M K + 32 * (i : ℕ) + l) = (z.EA i l).val := fun l => by
    have := i.isLt
    have := l.isLt
    rw [val_in4 ops X (by omega) (by omega) (hpub _ _ (Or.inl ⟨by omega, by omega⟩)), anchors4_ea z i.isLt l.isLt,
      inW4_e (by omega) (by omega), Nat.mod_eq_of_lt (nib_lt4 _)]
  have heb : ∀ l : Fin 32, val X (ebAt M N K + 32 * (j : ℕ) + l) = (z.EB j l).val := fun l => by
    have := j.isLt
    have := l.isLt
    rw [val_in4 ops X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors4_eb z j.isLt l.isLt,
      inW4_f (by omega) (by omega), Nat.mod_eq_of_lt (nib_lt4 _)]
  have hfa : ∀ (t : Fin K) (l : Fin 32), val X (faAt M N K + 32 * (t : ℕ) + l) = (z.FA t l).val := fun t l => by
    have := t.isLt
    have := l.isLt
    rw [val_in4 ops X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors4_fa z t.isLt l.isLt,
      inW4_f (by omega) (by omega), Nat.mod_eq_of_lt (nib_lt4 _)]
  have hfb : ∀ (t : Fin K) (l : Fin 32), val X (fbAt M N K + 32 * (t : ℕ) + l) = (z.FB t l).val := fun t l => by
    have := t.isLt
    have := l.isLt
    rw [val_in4 ops X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors4_fb z t.isLt l.isLt,
      inW4_f (by omega) (by omega), Nat.mod_eq_of_lt (nib_lt4 _)]
  have hg : ∀ s < 32, ∀ q < 32, val X (gAt M N K + 32 * s + q) = gram4 z s q := fun s hs q hq => by
    rw [val_in4 ops X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors4_g z hs hq,
      inW4_g (by omega), Nat.mod_eq_of_lt (gram4_lt z s q)]
  obtain ⟨w1, w2⟩ := entry4_words hops hnw hK hM hN X z act wt i j
    (hcone _ (cone4T ops M K N 64 64 hM hN i.isLt j.isLt hK0 fun g hg => hT g (Or.inl hg)))
    (hcone _ (cone4A ops M K N 64 64 hM hN i.isLt j.isLt fun g hg => hT g (Or.inr hg)))
    (hcone _ (cone4B ops M K N 64 64 hM hN i.isLt j.isLt fun g hg => hT g (Or.inr hg)))
    hxa hea hxb heb hfa hfb hg hA hB hr
  obtain ⟨he, -, -⟩ := tile_index hM hN i.isLt j.isLt
  have hti := Nat.mod_lt (i : ℕ) (show 0 < 64 by norm_num)
  have htj := Nat.mod_lt (j : ℕ) (show 0 < 64 by norm_num)
  have hc := c4Gate_bounds (K := K) hti htj hK0
  have hu := u4Gate_bounds (K := K) hti htj
  have vC : val X (cGlob4 M N K 64 64 i j) = tk i j % 2 ^ 32 := by
    unfold cGlob4
    rw [val_tile4 ops X he hc.1 hc.2 (hreg _ _ (Or.inr (Or.inr ⟨i, i.isLt, j, j.isLt, Or.inl rfl⟩)))]
    rw [show tile4 M N K 64 64 (i / 64 * (N / 64) + j / 64) (c4Gate K 64 64 (i % 64) (j % 64)) =
      cGlob4 M N K 64 64 i j from rfl, rows4_C act wt tk us hM hN i.isLt j.isLt hK0]
  have vU : val X (uGlob4 M N K 64 64 i j) = us i j % 2 ^ 32 := by
    unfold uGlob4
    rw [val_tile4 ops X he hu.1 hu.2 (hreg _ _ (Or.inr (Or.inr ⟨i, i.isLt, j, j.isLt, Or.inr rfl⟩)))]
    rw [show tile4 M N K 64 64 (i / 64 * (N / 64) + j / 64) (u4Gate K 64 64 (i % 64) (j % 64)) =
      uGlob4 M N K 64 64 i j from rfl, rows4_U act wt tk us hM hN i.isLt j.isLt]
  rw [vC, wordQ_mod32] at w1
  rw [vU, wordQ_mod32] at w2
  exact ⟨w1, w2⟩

end Accept4

/-! ## D₄ -/

/-- D₄'s shapes are whole-tile: `64 ∣ m`, `64 ∣ n`, and `k ≥ 1024` a multiple of 128. -/
theorem shape4_of_pearlCShapeFp4 {sh : Shape} (h : pearlCShapeFp4 sh) : Shape4 sh := by
  obtain ⟨-, -, hm, -, -, hn, hk1, -, ⟨c, hc⟩⟩ := h
  exact ⟨hm, hn, 2 * c, by omega, by omega⟩

end Pouw.PearlC.WholeCircuit
