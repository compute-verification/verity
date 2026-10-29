import Mathlib
import Proofs.Accounting.DeclaredCircuit.Credit
import Proofs.Accounting.DeclaredCircuit.Session

/-!
# A declared circuit's space and bandwidth from an accepted audit

`Accounting.SecurityProofs.DeclaredCircuitRequirement`, the theorem of
`Accounting.Guarantees.DeclaredCircuitRequirement`. A carrier of the declared circuit's over at most `L` of the audit's
windows carries at most `C` bits (`Carrier.card_le_census`), so every pass rebuilds at most `X_r·τ + C` bits of its
appended tiles, and its tiles of `W` are appended tiles. Each window's reads in its passes' tiles of `W`, with each
pass's tiles short of the store counted as read, against the window's correct receipts: a window short by more than a
pass's slack, `k` tiles more, has a pass unpaid on its tiles of `W` (`SketchDeployedWindow.exists_unpaid_tiles`), which
at signs of rank 12 holds a guessed tag or `k + 1` correct digests behind unobtained positions
(`Device.unpaid_tiles_split`), priced at each pass's prefix (`pr_session_le`). The draws then bound the windows' and the
session's reads so counted (`SketchDeployedSession.pr_deployed_of_unpaid_causal`), and a pass's credit is at most a
tile's bits for each of its tiles of `W` (`Requirement.credit_le_tiles`), so the shortfall against the credit is the
shortfall against the store (`Requirement.sum_lt_of_credit`, `Requirement.lt_of_credit`).
-/

namespace Accounting.RateSketch.Carrier

/-- **A carrier of a declared circuit's over at most `L` of the audit's windows carries at most `C` bits**: its inputs
at most `8·R_L` and its encodes at most `⌈1024·L·c·P·N·4096/E⌉`, the nonlinear units of each of its windows within the
caps' `c = caps.total` units a byte of the window's certified reads, at most `P·N·4096` bytes, under the encode bound
at 1024-bit words and `E` nonlinear units a word. -/
theorem card_le_census {N m P L B εnum εden E : ℕ} {ι : Type} {Wc : Gates.Dag ι} {caps : Gates.Caps}
    {Aud : Gates.Audit} (hE : Gates.EncodeBound Gates.p1024 3 B E εnum εden)
    (hE0 : 0 < E) (hprop : Gates.Property caps Aud Wc)
    (hreads : ∀ v < Aud.windows, Aud.reads v ≤ P * N * 4096)
    (K : Carrier N m) {I : Finset ℕ} (hI : I ⊆ Finset.range Aud.windows) (hcard : I.card ≤ L)
    (hK : K.provenance.OfCircuit (encodeBitsCost Gates.p1024 3 B εnum εden) Wc Aud I) :
    K.carried.card * 256 ≤ 8 * ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
        (fun I => ∑ v ∈ I, caps.runBytes v) + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊ := by
  have hb := K.card_le_of_budget hE (by decide) hE0
    fun Es => Provenance.OfCircuit.budget hK hprop hI hcard Es
  refine hb.trans (Nat.add_le_add_left ?_ _)
  rw [show 8 * Gates.p1024.bytes = 1024 by decide]
  set M := (Finset.range Aud.windows).sup (fun v =>
    (Gates.census Wc fun t => Aud.win t = v).budgeted + Gates.productsAt Wc fun t => Aud.win t = v) with hMdef
  rcases Nat.eq_zero_or_pos M with h0 | hpos
  · rw [h0]
    simp
  have hne : (Finset.range Aud.windows).Nonempty := by
    by_contra hne
    rw [Finset.not_nonempty_iff_eq_empty] at hne
    rw [hMdef, hne, Finset.sup_empty] at hpos
    exact not_lt_bot hpos
  obtain ⟨v, hv, hvM⟩ := Finset.exists_mem_eq_sup _ hne
    (fun v => (Gates.census Wc fun t => Aud.win t = v).budgeted + Gates.productsAt Wc fun t => Aud.win t = v)
  rw [← hMdef] at hvM
  have hvlt := Finset.mem_range.mp hv
  have hpv := hprop.2.1 v hvlt
  dsimp only at hpv
  obtain ⟨hr, hc, hg, hf, hn, -, hp⟩ := hpv
  have hMv : (M : ℚ) ≤ caps.total * Aud.reads v := by
    rw [hvM]
    simp only [Gates.Census.budgeted, Gates.Caps.total, hn, add_zero, add_mul]
    push_cast
    linarith
  have htot : 0 < caps.total := by
    by_contra hneg
    push Not at hneg
    have h1 : caps.total * (Aud.reads v : ℚ) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hneg (Nat.cast_nonneg _)
    have h2 : (0 : ℚ) < M := by exact_mod_cast hpos
    linarith
  have hreads' : (Aud.reads v : ℚ) ≤ ((P * N * 4096 : ℕ) : ℚ) := by exact_mod_cast hreads v hvlt
  have hMq : (M : ℚ) ≤ caps.total * ((P * N * 4096 : ℕ) : ℚ) :=
    hMv.trans (mul_le_mul_of_nonneg_left hreads' htot.le)
  have hE' : (0 : ℚ) < E := by exact_mod_cast hE0
  have hq : ((1024 * (L * M) / E : ℕ) : ℚ) ≤ (1024 * L * (P * N * 4096) : ℚ) * caps.total / E := by
    refine Nat.cast_div_le.trans ?_
    rw [div_le_div_iff_of_pos_right hE']
    have hL : (0 : ℚ) ≤ 1024 * L := by positivity
    calc ((1024 * (L * M) : ℕ) : ℚ) = (1024 * L) * (M : ℚ) := by push_cast; ring
      _ ≤ (1024 * L) * (caps.total * ((P * N * 4096 : ℕ) : ℚ)) := mul_le_mul_of_nonneg_left hMq hL
      _ = (1024 * L * (P * N * 4096) : ℚ) * caps.total := by push_cast; ring
  exact Nat.cast_le.mp (hq.trans (Nat.le_ceil _))

end Accounting.RateSketch.Carrier

namespace Accounting.RateSketch.SketchDeployedWindow

/-- Passes whose reads `r` and one slack each fall short of their correct receipts `c`, less what their tiles `u` and
credits `cr` leave of the store, have one that falls short of its correct receipts in its tiles `ci`. -/
theorem exists_lt_of_shares {P : ℕ} (r c ci u cr : Fin P → ℕ) (N slack : ℕ)
    (hc : ∀ j, c j ≤ ci j + (N - u j)) (hu : ∀ j, u j ≤ N) (hcr : ∀ j, cr j ≤ u j * 32768)
    (hx : ∑ j, r j + (P * (N * 32768) - ∑ j, cr j) + P * slack < (∑ j, c j) * 32768) :
    ∃ j, r j + slack < ci j * 32768 := by
  have hj : ∀ j, c j * 32768 + cr j ≤ ci j * 32768 + N * 32768 := fun j => by
    have := hc j
    have := hu j
    have := hcr j
    omega
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hj j
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib] at hsum
  have hcrs : ∑ j, cr j ≤ P * (N * 32768) := by
    calc ∑ j, cr j ≤ ∑ _j : Fin P, N * 32768 := Finset.sum_le_sum fun j _ =>
          (hcr j).trans (Nat.mul_le_mul_right _ (hu j))
      _ = P * (N * 32768) := by simp
  have hN : ∑ _j : Fin P, N * 32768 = P * (N * 32768) := by simp
  have hcs : (∑ j, c j) * 32768 = ∑ j, c j * 32768 := Finset.sum_mul _ _ _
  have hlt : ∑ j, (r j + slack) < ∑ j, ci j * 32768 := by
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    omega
  obtain ⟨j, -, hj⟩ := Finset.exists_lt_of_sum_lt hlt
  exact ⟨j, hj⟩

/-- **A window short on its passes' tiles has a pass unpaid on its tiles.** If the window's reads in the tiles `U j` of
each pass `j`, with each pass's store outside its credit `cr j ≤ |U j|` tiles counted as read, and one slack a pass,
fall short of its correct receipts, some pass's reads in its tiles and the slack fall short of its correct receipts
there. -/
theorem exists_unpaid_tiles {P N ρ h S L q ks G : ℕ} (A : SketchDeployedWindow P N 128 ρ h S L q ks G)
    (E : Fin P → SketchStore N 128) (σ : SketchSigns ks 128) (ω : Fin P → SketchPassCoins N ρ h ks G)
    (U : Fin P → Finset (Fin N)) (cr : Fin P → ℕ) (slack : ℕ) (hcr : ∀ j, cr j ≤ (U j).card * 32768)
    (hx : A.readsIn E σ ω U + (P * (N * 32768) - ∑ j, cr j) + P * slack < (A.correctOn E σ ω).card * 32768) :
    ∃ j, ((((A.dev j (E j) σ ω).prog σ (Data.ofParts (E j) (ω j).1)).mems (A.oracle j (E j) σ ω)
        (Data.ofParts (E j) (ω j).1)).filter fun p => p.1 ∈ U j).card * 256 + slack <
      ((A.dev j (E j) σ ω).correct sketch σ (A.oracle j (E j) σ ω) (Data.ofParts (E j) (ω j).1) ∩ U j).card *
        32768 := by
  have key : ∀ s V : Finset (Fin N), s.card ≤ (s ∩ V).card + (N - V.card) := by
    intro s V
    calc s.card ≤ ((s ∩ V) ∪ (Finset.univ \ V)).card := Finset.card_le_card fun t ht => by
          by_cases htV : t ∈ V
          · exact Finset.mem_union_left _ (Finset.mem_inter.mpr ⟨ht, htV⟩)
          · exact Finset.mem_union_right _ (Finset.mem_sdiff.mpr ⟨Finset.mem_univ t, htV⟩)
      _ ≤ (s ∩ V).card + (Finset.univ \ V).card := Finset.card_union_le _ _
      _ = (s ∩ V).card + (N - V.card) := by rw [Finset.card_univ_sdiff, Fintype.card_fin]
  have hc : (A.correctOn E σ ω).card = ∑ j,
      ((A.dev j (E j) σ ω).correct sketch σ (A.oracle j (E j) σ ω) (Data.ofParts (E j) (ω j).1)).card :=
    card_correctOn A E σ ω
  rw [hc] at hx
  exact exists_lt_of_shares
    (fun j => ((((A.dev j (E j) σ ω).prog σ (Data.ofParts (E j) (ω j).1)).mems (A.oracle j (E j) σ ω)
      (Data.ofParts (E j) (ω j).1)).filter fun p => p.1 ∈ U j).card * 256)
    (fun j => ((A.dev j (E j) σ ω).correct sketch σ (A.oracle j (E j) σ ω) (Data.ofParts (E j) (ω j).1)).card)
    (fun j => ((A.dev j (E j) σ ω).correct sketch σ (A.oracle j (E j) σ ω) (Data.ofParts (E j) (ω j).1) ∩
      U j).card)
    (fun j => (U j).card) cr N slack (fun j => key _ (U j))
    (fun j => (Finset.card_le_univ _).trans_eq (Fintype.card_fin N)) hcr hx

end Accounting.RateSketch.SketchDeployedWindow

namespace Accounting.Requirement

/-- Reads short of the credit by `g·M` over the windows, each window's read counting its store outside its credit,
at most `K` bits, are short of `(1 − g)·M` when the windows' stores total `M`. -/
theorem sum_lt_of_credit {n : ℕ} (rd ri cs : Fin n → ℕ) (K : ℕ) (g M : ℝ) (hrd : ∀ w, rd w = ri w + (K - cs w))
    (hcs : ∀ w, cs w ≤ K) (hM : (n : ℝ) * K = M)
    (h : ((∑ w, ri w : ℕ) : ℝ) < ((∑ w, cs w : ℕ) : ℝ) - g * M) :
    ((∑ w, rd w : ℕ) : ℝ) < (1 - g) * M := by
  have hnat : ∑ w, rd w + ∑ w, cs w = ∑ w, ri w + n * K := by
    have hw : ∀ w, rd w + cs w = ri w + K := fun w => by
      rw [hrd w]
      have := hcs w
      omega
    have hs : ∑ w, (rd w + cs w) = ∑ w, (ri w + K) := Finset.sum_congr rfl fun w _ => hw w
    rw [← Finset.sum_add_distrib, hs, Finset.sum_add_distrib]
    simp
  have hreal : ((∑ w, rd w : ℕ) : ℝ) + ((∑ w, cs w : ℕ) : ℝ) = ((∑ w, ri w : ℕ) : ℝ) + (n : ℝ) * K := by
    exact_mod_cast hnat
  rw [sub_mul, one_mul]
  linarith

/-- A window's reads short of its credit by `g·M`, its read counting its store outside its credit, at most `K = M`
bits, are short of `(1 − g)·M`. -/
theorem lt_of_credit (rd ri cs K : ℕ) (g M : ℝ) (hrd : rd = ri + (K - cs)) (hcs : cs ≤ K) (hM : (K : ℝ) = M)
    (h : (ri : ℝ) < (cs : ℝ) - g * M) : (rd : ℝ) < (1 - g) * M := by
  have hreal : (rd : ℝ) + cs = ri + K := by
    have : rd + cs = ri + K := by omega
    exact_mod_cast this
  rw [sub_mul, one_mul]
  linarith

end Accounting.Requirement

namespace Accounting.SecurityProofs

open Accounting.RateSketch
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_mono)

/-- **A declared circuit's held and moved bits from an accepted stacked audit, at the deployed parameters**,
`Accounting.Guarantees.DeclaredCircuitRequirement`. -/
theorem DeclaredCircuitRequirement : Accounting.Guarantees.DeclaredCircuitRequirement := by
  intro n P N ρ S q G Xr Lr qr τ k a L E es B εnum εden T M ε δ ιW Wc caps Aud Ω Ca Ch _ _ enc choose layout steer
    joined A V Kg Dv _ _ W pl ℓ grp bind mm plan addr hn hP hN hE0 hrank hKB hSq hCs hEle hε hB hCC hslot hlife _hres
    hprop hreads hx RL C γ owned credit tiles store win
  have hE : Gates.EncodeBound Gates.p1024 3 B E εnum εden :=
    SecurityProofs.Gates.encodeBound_of hSq hCs hEle hε (by decide) SecurityProofs.Gates.p1024_floor.1
      SecurityProofs.Gates.p1024_floor.2
  have hX : ∀ (c : SketchPrefix n P N 128 ρ 512 16 G 2773) (y : Ω × (Fin T → Ca)) (g : Fin G)
      (R : RateSketch.SketchStream ρ) (O : RateSketch.Oracle N ρ 512 RateSketch.Sector (Fin 16 → ZMod (2 ^ 32))),
      ((passDev A steer c y g).prog c.signs (Data.ofParts (steeredStore enc choose layout c y) R)).rebuildsIn
        (appendedTiles layout c y) O (Data.ofParts (steeredStore enc choose layout c y) R) * 256 ≤ Xr * τ + C := by
    intro c y g R O
    obtain ⟨K, I, hreb, hI, hcard, hK⟩ := hCC hlife c g y
    have h1 := hreb R O
    dsimp only at h1
    exact h1.trans (Nat.add_le_add_left (Carrier.card_le_census hE hE0 hprop hreads K hI hcard hK) _)
  have hcredit : ∀ x w j, credit x (bind x w j) ≤ (tiles x w j).card * 32768 := by
    intro x w j
    have hm := (hx x).2.2 w j
    dsimp only at hm
    obtain ⟨hS, hPl, hO, hinj, hrange⟩ := hm
    exact Requirement.credit_le_tiles W (pl x).toExecution (ℓ x) (mm x) (plan x w j) (grp x) (bind x w j) (addr x)
      (layout (SketchPrefix.of x.2 w j) x.1) hS hPl hO hinj hrange
  have htiles : ∀ x w j, tiles x w j ⊆ appendedTiles layout (SketchPrefix.of x.2 w j) x.1 := fun x w j =>
    Requirement.tiles_subset_appended layout (SketchPrefix.of x.2 w j) x.1 (addr x)
      (Requirement.blocks (ℓ x) (Requirement.creditedReads W (pl x).toExecution (ℓ x) (bind x w j)))
  have hcs : ∀ x w, ∑ j, credit x (bind x w j) ≤ P * (N * 32768) := by
    intro x w
    calc ∑ j, credit x (bind x w j) ≤ ∑ _j : Fin P, N * 32768 := Finset.sum_le_sum fun j _ =>
          (hcredit x w j).trans (Nat.mul_le_mul_right _ ((Finset.card_le_univ _).trans_eq (Fintype.card_fin N)))
      _ = P * (N * 32768) := by simp
  have hcor : ∀ (o : Ω × (Fin T → Ca)) (ws ws' : Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) (w : Fin n),
      (∀ v, v < w → ws v = ws' v) → (ws w).1 = (ws' w).1 → (ws w).2.1 = (ws' w).2.1 →
      (win (o, ws) w).correctOn (store o ws w) (ws w).1 (ws w).2.1 =
        (win (o, ws') w).correctOn (store o ws' w) (ws' w).1 (ws' w).2.1 := by
    intro o ws ws' w hlt hσ hp
    have hη : (fun v : {v : Fin n // v < w} => ws v.1) = fun v => ws' v.1 := funext fun v => hlt v.1 v.2
    have hpre : ∀ j, SketchPrefix.of ws w j = SketchPrefix.of ws' w j := fun j => by
      simp only [SketchPrefix.of]
      rw [hη, hσ, hp]
    have hwin : win (o, ws) w = win (o, ws') w := by
      show A.win w ws (fun j => steer (SketchPrefix.of ws w j) o) =
        A.win w ws' (fun j => steer (SketchPrefix.of ws' w j) o)
      simp only [hpre, SketchSummarySession.win, SketchDeployedSession.win]
      rw [hη]
    have hstore : store o ws w = store o ws' w := funext fun j => by
      show steeredStore enc choose layout (SketchPrefix.of ws w j) o =
        steeredStore enc choose layout (SketchPrefix.of ws' w j) o
      rw [hpre j]
    rw [hwin, hstore, hσ, hp]
  have hu : pr (fun z : StackedCoins Ω Ca T n P N ρ G => ∃ w,
      ((win z w).readsIn (store z.1 z.2 w) (z.2 w).1 (z.2 w).2.1 (tiles z w) +
          (P * (N * 32768) - ∑ j, credit z (bind z w j))) +
        P * (S + (Lr * τ - ρ * 256) + (Xr * τ + C + k * 32768)) <
      ((win z w).correctOn (store z.1 z.2 w) (z.2 w).1 (z.2 w).2.1).card * (128 * 256)) ≤
      (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * (2 ^ a * (((qr * τ : ℕ) : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ) := by
    refine le_trans (pr_mono fun x hx' => ?_) (pr_session_le (k := k) hrank hKB choose layout steer hslot A hB)
    obtain ⟨w, hw⟩ := hx'
    by_cases hr : RankGood 12 (x.2 w).1
    swap
    · exact ⟨w, Or.inl hr⟩
    obtain ⟨j, hj⟩ := SketchDeployedWindow.exists_unpaid_tiles (win x w) (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1
      (tiles x w) (fun j => credit x (bind x w j)) (S + (Lr * τ - ρ * 256) + (Xr * τ + C + k * 32768))
      (hcredit x w) hw
    have hXp : (((win x w).dev j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1).prog (x.2 w).1
        (Data.ofParts (store x.1 x.2 w j) ((x.2 w).2.1 j).1)).rebuildsIn (tiles x w j)
        ((win x w).oracle j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
        (Data.ofParts (store x.1 x.2 w j) ((x.2 w).2.1 j).1) * 256 ≤ Xr * τ + C :=
      le_trans (Nat.mul_le_mul_right 256 (RateSketch.Chip.rebuildsIn_mono (htiles x w j) _ _ _))
        (hX (SketchPrefix.of x.2 w j) x.1 ((win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
          ((x.2 w).2.1 j).1 (((x.2 w).2.1 j).2 ((win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)))
    rcases RateSketch.Device.unpaid_tiles_split ((win x w).dev j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
        RateSketch.sketch (x.2 w).1 ((win x w).oracle j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
        (Data.ofParts (store x.1 x.2 w j) ((x.2 w).2.1 j).1) (tiles x w j) (Xr * τ + C) k (by norm_num) hXp hj
      with ⟨t, ht⟩ | ⟨ι, hb⟩
    · exact ⟨w, Or.inr ⟨j, Or.inl ⟨(win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1, t, ht⟩⟩⟩
    · exact ⟨w, Or.inr ⟨j, Or.inr ⟨hr, (win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1, ι, hb⟩⟩⟩
  refine le_trans (pr_mono fun x hx' => ?_) ((SketchDeployedSession.pr_deployed_of_unpaid_causal
    (fun o ws w => (win (o, ws) w).correctOn (store o ws w) (ws w).1 (ws w).2.1)
    (fun o ws w => (win (o, ws) w).readsIn (store o ws w) (ws w).1 (ws w).2.1 (tiles (o, ws) w) +
      (P * (N * 32768) - ∑ j, credit (o, ws) (bind (o, ws) w j)))
    hcor (Xr * τ + C + k * 32768) hn hP hN _ hu).trans (le_of_eq ?_))
  · obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx'
    · refine ⟨hall, Or.inl ?_⟩
      exact Requirement.sum_lt_of_credit
        (fun w => (win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) +
          (P * (N * 32768) - ∑ j, credit x (bind x w j)))
        (fun w => (win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w))
        (fun w => ∑ j, credit x (bind x w j)) (P * (N * 32768)) (0.0015 + γ) (n * P * N * 32768)
        (fun w => rfl) (hcs x) (by push_cast; ring) hT
    · refine ⟨hall, Or.inr ⟨w, ?_⟩⟩
      exact Requirement.lt_of_credit _ ((win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w))
        (∑ j, credit x (bind x w j)) (P * (N * 32768)) (0.01 + γ) (P * N * 32768) rfl (hcs x w)
        (by push_cast; ring) hw
  · ring

end Accounting.SecurityProofs
