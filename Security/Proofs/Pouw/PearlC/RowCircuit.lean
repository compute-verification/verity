import Definitions.Pouw.PearlC.RowSem
import Proofs.Pouw.PearlC.TileCircuit

/-!
# The row tables compute a row's statistics, scales and forming (proofs)

For every `ops` (core's primitives as a parameter), a row table's honest values meet its gates run by run
(`Rows.holds`, `RunsHold`), the inputs being the transcript's (`Rows.val_input`), and a transcript whose one unit is
correct holds them on every output (`Spec.out_eq`). On the runs `pc8`'s two row units share (`pre8Runs`), those values
are `RowSem`'s formulas on the row's inputs:

* `s` and `ρ` (`stats_holds`): the running max by induction over the positions (`selRun_holds`), each FMA partial by
  strong induction over its accumulator (`fmaRun_holds`), the butterfly round by round (`bfly_holds`);
* `α`, `β` and the floor (`pre8_alpha`, `pre8_beta`, `pre8_floor`), the rescaled noise' codes (`pre8_eTilde`) and each
  position's code (`pre8_code`), so `row8A_code` and `row8B_code`: a transcript correct on RowA or RowB has on position
  `t`'s output `code8W` at `alpha8`'s and `beta8`'s words of its row;
* a row chain (`Chain8Holds`) is the tile check's chain on the codes it reads: each atom's word (`chain8_atom`,
  `atom8_eval`), each window (`chain8_window`), the running sum (`chain8_add`), so its last word is `chainWordP? p 4`'s
  (`chain8_word`), and RowA's `P_A` word `q` is that word cast to bf16 (`row8A_pa`);
* RowB's `P_B` word `q` is the BF16 cast of `Σ_s ẽ_s·g_{s,q}` (`pbAcc`, the FMAs from `+0` of the rescaled noise' FP32
  words, `pre8_ew`) plus its chain's word negated (`row8B_pb`), and the rescaled noise' BF16 words, `P_A`'s head and
  `P_B`'s tail negated, are `row8A_be` and `row8B_nbe`;
* on `pc4`'s shared runs, `α`, `β`, `a_E` (`pre4_alpha`) and the flag (`pre4_flag`, `flag4`), so RowA's `Ẽ` and RowB's
  `−Ẽ` (`row4A_et`, `row4B_net`) and both flags (`row4A_flag`, `row4B_flag`);
* `pc8`'s flag: the sum of squares (`pre8_sq`, `sqSum`), `θ` (`pre8_theta`), the dead count (`pre8_dead`,
  `deadCount`) and the flag (`pre8_flag`, `flag8`), so both rows' flags (`row8A_flag`, `row8B_flag`);
* words read back as values: a step's word is `f32Word` of its value (`f32Word_wordVal`), and the cast of a word that
  is neither NaN nor a signed zero is `e4m3` of its value (`e4m3_wordVal`), so a code is `e4m3` of its formed word's
  value where the cast is `e4m3Word`'s (`code8W_e4m3`), and `pc8Form` at step words' values is `form8` at the words
  (`pc8Form_value`); with the noise' and basis' codes read off their gates as they are (`gateCodes`), each code on
  RowA and RowB is `e4m3` of `pc8Form` at its row's values and codes (`codes8_sem`, `row8A_code_sem`, `row8B_code_sem`), and
  `pc8Sem`'s `α`, `β` and floor at a row of step words are the table's words read as values (`pc8Sem_scales`).
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

variable (ops : RowOps)

/-! ## Values that meet a table's gates -/

/-- The values `V` meet the run `R`'s gates from gate `o`. -/
def RunHolds (V : ℕ → ℕ) (R : Run) (o : ℕ) : Prop :=
  ∀ r < R.n, V (o + r) = (R.gate r).prim.eval ops ((R.gate r).args.map (Arg.eval ops V))

/-- The values `V` meet every run's gates, the runs laid out from gate `o`. -/
def RunsHold (V : ℕ → ℕ) : List Run → ℕ → Prop
  | [], _ => True
  | R :: L, o => RunHolds ops V R o ∧ RunsHold V L (o + R.n)

theorem runsHold_append (V : ℕ → ℕ) : ∀ {L₁ L₂ : List Run} {o : ℕ}, RunsHold ops V (L₁ ++ L₂) o →
    RunsHold ops V L₁ o ∧ RunsHold ops V L₂ (o + runsLen L₁)
  | [], _, o, h => ⟨trivial, by simpa [runsLen] using h⟩
  | R :: L₁, L₂, o, h => by
    obtain ⟨h1, h2⟩ := h
    obtain ⟨h3, h4⟩ := runsHold_append V h2
    exact ⟨⟨h1, h3⟩, by simpa [runsLen, Nat.add_assoc] using h4⟩

theorem fill_reads (V : ℕ → ℕ) : ∀ args : List Arg,
    fill ops args ((args.flatMap Arg.reads).map V) = args.map (Arg.eval ops V)
  | [] => rfl
  | .c v :: as => by simp [fill, Arg.reads, Arg.eval, fill_reads V as]
  | .g a w :: as => by simp [fill, Arg.reads, Arg.eval, fill_reads V as]

theorem runsAt_lt : ∀ (L : List Run) (o g : ℕ) (G : RGate), runsAt L o g = some G → g < o + runsLen L
  | [], _, _, _, h => by simp [runsAt] at h
  | R :: L, o, g, G, h => by
    unfold runsAt at h
    split_ifs at h with hg
    · simp only [runsLen]; omega
    · have := runsAt_lt L _ g G h; simp only [runsLen]; omega

theorem runsHold_of (V : ℕ → ℕ) : ∀ (L : List Run) (o : ℕ),
    (∀ g G, o ≤ g → runsAt L o g = some G → V g = G.prim.eval ops (G.args.map (Arg.eval ops V))) →
    RunsHold ops V L o
  | [], _, _ => trivial
  | R :: L, o, h => by
    refine ⟨fun r hr => h (o + r) (R.gate r) (by omega) ?_,
      runsHold_of V L (o + R.n) fun g G hg hG => h g G (by omega) ?_⟩
    · simp [runsAt, hr]
    · have hn : ¬ g < o + R.n := by omega
      simp only [runsAt, hn, ↓reduceIte]; exact hG

@[simp] theorem Wire.eval_id (x : ℕ) : Wire.eval ops .id x = x := rfl

theorem Prim.eval_mod (q : Prim) (l : List ℕ) : q.eval ops l % 2 ^ q.w = q.eval ops l := by
  unfold Prim.eval; exact Nat.mod_mod _ _

/-- **A table's honest values meet its gates**: the circuit's evaluation on any inputs. -/
theorem Rows.holds (T : Rows) (X : (T.spec ops).toCircuit.Transcript) :
    RunsHold ops (val ((T.spec ops).toCircuit.eval X)) T.runs T.nIn := by
  apply runsHold_of
  intro g G hg hG
  have hlt : g < (T.spec ops).N := by
    have := runsAt_lt _ _ _ _ hG
    show g < T.nIn + runsLen T.runs
    omega
  have hgate : (T.spec ops).gate g = G.toGate ops := by
    show T.gate ops g = _
    unfold Rows.gate
    have hn : ¬ g < T.nIn := by omega
    simp only [hn, ↓reduceIte, hG]
  rw [Spec.val_gate _ X hlt (by rw [hgate]; rfl), hgate]
  simp only [RGate.toGate, RGate.reads]
  rw [fill_reads, Prim.eval_mod]

/-- A table's honest values on its inputs are the transcript's. -/
theorem Rows.val_input (T : Rows) (X : (T.spec ops).toCircuit.Transcript) {l : ℕ} (hl : l < T.nIn) :
    val ((T.spec ops).toCircuit.eval X) l = val X l := by
  have hlt : l < (T.spec ops).N := by
    show l < T.nIn + runsLen T.runs
    omega
  rw [Spec.val_input _ X hlt (by show (T.gate ops l).input = true; unfold Rows.gate; simp only [hl, ↓reduceIte]),
    val_of_lt _ hlt]

/-- A transcript whose one unit is correct holds the circuit's values on every output (core's
`Verity.Partition.compose`). -/
theorem Spec.out_eq (S : Spec) (X : S.toCircuit.Transcript) (hX : S.part.Correct X 0) {g : ℕ} (hg : g ∈ S.outs)
    (hlt : g < S.N) : val X g = val (S.toCircuit.eval X) g := by
  have hw : S.part.wrong X = ∅ := by
    ext u
    simp only [Verity.Partition.mem_wrong, Finset.notMem_empty, iff_false, not_not]
    rwa [Subsingleton.elim u 0]
  have hg' : (⟨g, hlt⟩ : Fin S.toCircuit.N) ∈ S.part.committed :=
    Verity.Partition.mem_committed_of_output _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hg⟩)
  rw [val_of_lt _ hlt, val_of_lt _ hlt, S.part.compose X X (fun _ _ => rfl) hw _ hg']

/-! ## The statistics -/

section Stats

variable {ops} {V : ℕ → ℕ}

theorem selRun_holds {ab : Wire} {K b : ℕ} (h : RunHolds ops V (selRun ab K b) b) :
    ∀ t, t + 1 < K → V (b + 2 * t + 1) = rowMax ops ab V (t + 1) := by
  intro t
  induction t with
  | zero =>
    intro hK
    have h0 := h 0 (by simp [selRun]; omega)
    have h1 := h 1 (by simp [selRun]; omega)
    simp only [selRun] at h0 h1
    simp at h0 h1
    simp only [rowMax, Arg.eval, Wire.eval_id] at h0 h1 ⊢
    rw [h1, h0]
  | succ t ih =>
    intro hK
    have h0 := h (2 * t + 2) (by simp [selRun]; omega)
    have h1 := h (2 * t + 3) (by simp [selRun]; omega)
    have e0 : (2 * t + 2) % 2 = 0 := by omega
    have e1 : (2 * t + 3) % 2 = 1 := by omega
    have d0 : (2 * t + 2) / 2 + 1 = t + 2 := by omega
    have d1 : (2 * t + 3) / 2 + 1 = t + 2 := by omega
    simp only [selRun, e0, e1, d0, d1, ↓reduceIte, show 2 * t + 2 ≠ 0 by omega, show 2 * t + 3 ≠ 1 by omega,
      show (1 : ℕ) ≠ 0 by omega, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at h0 h1
    rw [show b + (2 * t + 3) = b + 2 * (t + 1) + 1 by omega] at h1
    rw [h1, show b + 2 * (t + 1) + 1 - 1 = b + (2 * t + 2) by omega, h0,
      show b + (2 * t + 2) - 1 = b + 2 * t + 1 by omega, show b + 2 * (t + 1) + 1 - 2 = b + 2 * t + 1 by omega,
      ih (by omega)]
    rfl

theorem fmaRun_holds {w : Wire} {K b : ℕ} (h : RunHolds ops V (fmaRun w K b) (fmaAt K b)) :
    ∀ i < K / 8, V (fmaAt K b + i) = fmaAcc ops w V i := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    intro hi
    have hg := h i (by simpa [fmaRun] using hi)
    simp only [fmaRun] at hg
    rw [hg, fmaAcc]
    by_cases h32 : i < 32
    · simp [h32, Arg.eval]
    · simp only [h32, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval]
      rw [show fmaAt K b + i - 32 = fmaAt K b + (i - 32) by omega, ih (i - 32) (by omega) (by omega)]

theorem bfly_congr {h n j : ℕ} {v v' : ℕ → ℕ} (hj : v j = v' j) (hh : j + h < n → v (j + h) = v' (j + h)) :
    bfly ops h n v j = bfly ops h n v' j := by
  unfold bfly
  by_cases hc : j + h < n
  · simp only [hc, ↓reduceIte, hj, hh hc]
  · simp only [hc, ↓reduceIte, hj]

theorem bfly_holds {h K pb o : ℕ} (hR : RunHolds ops V (bflyRun h K pb) o) :
    ∀ j < min h (K / 8), V (o + j) = bfly ops h (K / 8) (fun j => V (pb + j)) j := by
  intro j hj
  have hg := hR j (by simpa [bflyRun] using hj)
  simp only [bflyRun] at hg
  rw [hg, bfly]
  by_cases hc : j + h < K / 8
  · simp [hc, Arg.eval, Nat.add_assoc]
  · simp [hc, Arg.eval]

theorem accLast_lt {n j : ℕ} (hj : j < n) : accLast n j < n := by
  unfold accLast
  have := Nat.div_mul_le_self (n - 1 - j) 32
  omega

/-- **The statistics**: on values meeting `statsRuns`, the running max's last gate is `rowMax` and `ρ`'s gate is
`rowRho`, on the values of `x`'s gates. -/
theorem stats_holds {ab w : Wire} {K b : ℕ} (hK : 8 ≤ K) (h : RunsHold ops V (statsRuns ab w K b) b) :
    V (fmaAt K b - 1) = rowMax ops ab V (K - 1) ∧ V (rhoAt K b + 1) = rowRho ops w V K := by
  obtain ⟨hs, hf, h0, h8, h4, h2, h1, hr, -⟩ := h
  have hf : RunHolds ops V (fmaRun w K b) (fmaAt K b) := hf
  have h0 : RunHolds ops V (bfly0Run K b) (bf0At K b) := h0
  have h8 : RunHolds ops V (bflyRun 8 K (bf0At K b)) (bf1At K b) := h8
  have h4 : RunHolds ops V (bflyRun 4 K (bf1At K b)) (bf2At K b) := h4
  have h2 : RunHolds ops V (bflyRun 2 K (bf2At K b)) (bf3At K b) := h2
  have h1 : RunHolds ops V (bflyRun 1 K (bf3At K b)) (bf4At K b) := h1
  have hr : RunHolds ops V (rhoRun K b) (rhoAt K b) := hr
  refine ⟨?_, ?_⟩
  · rw [show fmaAt K b - 1 = b + 2 * (K - 2) + 1 by unfold fmaAt; omega,
      selRun_holds hs (K - 2) (by omega), show K - 2 + 1 = K - 1 by omega]
  have hF := fmaRun_holds hf
  set A := fun j => fmaAcc ops w V (accLast (K / 8) j) with hA
  have ha : ∀ j' < K / 8, V (lastFma K b j') = A j' := by
    intro j' hj'
    rw [show lastFma K b j' = fmaAt K b + accLast (K / 8) j' by unfold lastFma accLast; omega]
    exact hF _ (accLast_lt hj')
  have q0 : ∀ j < min 16 (K / 8), V (bf0At K b + j) = bfly ops 16 (K / 8) A j := by
    intro j hj
    have hg := h0 j (by simpa [bfly0Run] using hj)
    simp only [bfly0Run] at hg
    rw [hg, bfly]
    by_cases hc : j + 16 < K / 8
    · simp [hc, Arg.eval, ha j (by omega), ha (j + 16) hc]
    · simp [hc, Arg.eval, ha j (by omega)]
  have r1 := bfly_holds h8
  have r2 := bfly_holds h4
  have r3 := bfly_holds h2
  have r4 := bfly_holds h1
  have q1 : ∀ j < min 8 (K / 8), V (bf1At K b + j) = bfly ops 8 (K / 8) (bfly ops 16 (K / 8) A) j := by
    intro j hj
    rw [r1 j hj]
    exact bfly_congr (q0 j (by omega)) fun hc => q0 _ (by omega)
  have q2 : ∀ j < min 4 (K / 8), V (bf2At K b + j) =
      bfly ops 4 (K / 8) (bfly ops 8 (K / 8) (bfly ops 16 (K / 8) A)) j := by
    intro j hj
    rw [r2 j hj]
    exact bfly_congr (q1 j (by omega)) fun hc => q1 _ (by omega)
  have q3 : ∀ j < min 2 (K / 8), V (bf3At K b + j) =
      bfly ops 2 (K / 8) (bfly ops 4 (K / 8) (bfly ops 8 (K / 8) (bfly ops 16 (K / 8) A))) j := by
    intro j hj
    rw [r3 j hj]
    exact bfly_congr (q2 j (by omega)) fun hc => q2 _ (by omega)
  have q4 : V (bf4At K b) = sumSq ops w V (K / 8) := by
    have := r4 0 (by omega)
    rw [Nat.add_zero] at this
    rw [this]
    exact bfly_congr (q3 0 (by omega)) fun hc => q3 _ (by omega)
  have g0 := hr 0 (by simp [rhoRun, Run.ofList])
  have g1 := hr 1 (by simp [rhoRun, Run.ofList])
  simp [rhoRun, Run.ofList, Arg.eval] at g0 g1
  rw [g1, g0, show rhoAt K b - 1 = bf4At K b by unfold rhoAt; omega, q4]
  rfl

end Stats

/-! ## The formulas read only the row's inputs -/

section Congr

variable {ops}

theorem rowMax_congr {ab : Wire} {x y : ℕ → ℕ} : ∀ t, (∀ l ≤ t, x l = y l) → rowMax ops ab x t = rowMax ops ab y t
  | 0, h => by simp only [rowMax, h 0 le_rfl]
  | t + 1, h => by
    simp only [rowMax, rowMax_congr t fun l hl => h l (by omega), h (t + 1) le_rfl]

theorem fmaAcc_congr {w : Wire} {x y : ℕ → ℕ} (i : ℕ) (h : ∀ l ≤ 8 * i, x l = y l) :
    fmaAcc ops w x i = fmaAcc ops w y i := by
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    rw [fmaAcc.eq_1 ops w x i, fmaAcc.eq_1 ops w y i, h (8 * i) le_rfl]
    by_cases h32 : i < 32
    · simp only [h32, ↓reduceIte]
    · simp only [h32, ↓reduceIte, ih (i - 32) (by omega) fun l hl => h l (by omega)]

theorem bfly_agree {h n : ℕ} {v v' : ℕ → ℕ} (hv : ∀ j < n, v j = v' j) :
    ∀ j < n, bfly ops h n v j = bfly ops h n v' j :=
  fun j hj => bfly_congr (hv j hj) fun hc => hv _ hc

theorem rowRho_congr {w : Wire} {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    rowRho ops w x K = rowRho ops w y K := by
  have hA : ∀ j < K / 8, fmaAcc ops w x (accLast (K / 8) j) = fmaAcc ops w y (accLast (K / 8) j) := by
    intro j hj
    have := accLast_lt hj
    exact fmaAcc_congr _ fun l hl => h l (by omega)
  unfold rowRho sumSq
  rw [bfly_agree (bfly_agree (bfly_agree (bfly_agree (bfly_agree hA))))  0 (by omega)]

theorem alpha8_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    alpha8 ops x K = alpha8 ops y K := by
  unfold alpha8
  rw [rowRho_congr hK h, rowMax_congr (K - 1) fun l hl => h l (by omega)]

theorem beta8_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    beta8 ops x K = beta8 ops y K := by
  unfold beta8
  rw [rowRho_congr hK h, alpha8_congr hK h]

theorem floor8_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    floor8 ops x K = floor8 ops y K := by
  unfold floor8
  rw [rowRho_congr hK h, alpha8_congr hK h]

end Congr

/-! ## `pc8`'s shared runs -/

section Pre8

variable {ops} {V : ℕ → ℕ} {p : Pipeline} {K b : ℕ}

theorem pre8_runs (h : RunsHold ops V (pre8Runs p K b) b) :
    RunsHold ops V (statsRuns .fabs .id K b) b ∧ RunHolds ops V (alpha8Run K b) (al8At K b) ∧
      RunHolds ops V (ok8Run K b) (ok8At K b) ∧ RunHolds ops V (ep8Run K b) (ep8At K b) ∧
      RunHolds ops V (code8Run p K b) (code8At K b) := by
  obtain ⟨hs, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  obtain ⟨ha, -, -, -, ho, he, -, hc, -⟩ := hr
  exact ⟨hs, ha, ho, he, hc⟩

/-- **`α` and `β`** on values meeting `pc8`'s shared runs: `alpha8` and `beta8` of the values of `x`'s gates. -/
theorem pre8_alpha (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    V (al8At K b + 3) = alpha8 ops V K ∧ V (al8At K b + 5) = beta8 ops V K := by
  obtain ⟨hs, ha, -⟩ := pre8_runs h
  obtain ⟨hsm, hrho⟩ := stats_holds hK hs
  have g0 := ha 0 (by simp [alpha8Run, Run.ofList])
  have g1 := ha 1 (by simp [alpha8Run, Run.ofList])
  have g2 := ha 2 (by simp [alpha8Run, Run.ofList])
  have g3 := ha 3 (by simp [alpha8Run, Run.ofList])
  have g4 := ha 4 (by simp [alpha8Run, Run.ofList])
  have g5 := ha 5 (by simp [alpha8Run, Run.ofList])
  simp only [alpha8Run, Run.ofList, List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id, Nat.add_zero] at g0 g1 g2 g3 g4 g5
  have hα : V (al8At K b + 3) = alpha8 ops V K := by
    rw [g3, g2, g1, g0, hsm, hrho]; rfl
  refine ⟨hα, ?_⟩
  rw [g5, g4, hα, hrho]; rfl

/-- **The floor**: the flag run's first gate is `floor8`. -/
theorem pre8_floor (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) : V (ok8At K b) = floor8 ops V K := by
  obtain ⟨hs, -, ho, -⟩ := pre8_runs h
  obtain ⟨-, hrho⟩ := stats_holds hK hs
  have g0 := ho 0 (by simp [ok8Run, Run.ofList])
  simp only [ok8Run, Run.ofList, List.getD_cons_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
    Nat.add_zero] at g0
  rw [g0, hrho, (pre8_alpha hK h).1]; rfl

/-- **The rescaled noise' codes**: `eTilde8` at `β`'s word of each of `e`'s codes. -/
theorem pre8_eTilde (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    ∀ l < 32, V (ep8At K b + 3 * l + 2) = eTilde8 ops (beta8 ops V K) (V (K + l)) := by
  intro l hl
  obtain ⟨-, -, -, he, -⟩ := pre8_runs h
  have g0 := he (3 * l) (by simp [ep8Run]; omega)
  have g1 := he (3 * l + 1) (by simp [ep8Run]; omega)
  have g2 := he (3 * l + 2) (by simp [ep8Run]; omega)
  have e0 : 3 * l % 3 = 0 := by omega
  have e1 : (3 * l + 1) % 3 = 1 := by omega
  have e2 : (3 * l + 2) % 3 = 2 := by omega
  have d0 : 3 * l / 3 = l := by omega
  simp only [ep8Run, e0, e1, e2, d0, ↓reduceIte, show (2 : ℕ) ≠ 0 by omega, show (2 : ℕ) ≠ 1 by omega,
    show (1 : ℕ) ≠ 0 by omega, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g0 g1 g2
  rw [show ep8At K b + 3 * l + 2 = ep8At K b + (3 * l + 2) by omega, g2, show ep8At K b + (3 * l + 2) - 1 = ep8At K b + (3 * l + 1) by omega, g1,
    show ep8At K b + (3 * l + 1) - 1 = ep8At K b + 3 * l by omega, g0, (pre8_alpha hK h).2]
  rfl

/-- **The rescaled noise' words**: each code's FP32 word. -/
theorem pre8_ew (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    ∀ l < 32, V (ew8At K b + l) = Prim.eval ops .e4m3ToF32 [eTilde8 ops (beta8 ops V K) (V (K + l))] := by
  intro l hl
  obtain ⟨-, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  obtain ⟨-, -, -, -, -, -, hw, -⟩ := hr
  have hw' : RunHolds ops V (ew8Run K b) (ew8At K b) := hw
  have g := hw' l (by simp [ew8Run]; omega)
  simp only [ew8Run, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g
  rw [g, pre8_eTilde hK h l hl]

/-- **Position `t`'s code**: `code8W` at `α`'s word, `x_t`, the rescaled noise' codes and the form basis' row `t`. -/
theorem pre8_code (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    ∀ t < K, V (code8 K b t) = code8W ops p (alpha8 ops V K) (V t)
      (fun l => eTilde8 ops (beta8 ops V K) (V (K + l))) (fun l => V (K + 32 + 32 * t + l)) := by
  intro t ht
  obtain ⟨-, -, -, -, hc⟩ := pre8_runs h
  have g0 := hc (3 * t) (by simp [code8Run]; omega)
  have g1 := hc (3 * t + 1) (by simp [code8Run]; omega)
  have g2 := hc (3 * t + 2) (by simp [code8Run]; omega)
  have e0 : 3 * t % 3 = 0 := by omega
  have e1 : (3 * t + 1) % 3 = 1 := by omega
  have e2 : (3 * t + 2) % 3 = 2 := by omega
  have d0 : 3 * t / 3 = t := by omega
  have d1 : (3 * t + 1) / 3 = t := by omega
  simp only [code8Run, e0, e1, e2, d0, d1, ↓reduceIte, show (2 : ℕ) ≠ 0 by omega, show (2 : ℕ) ≠ 1 by omega,
    show (1 : ℕ) ≠ 0 by omega, List.map_cons, List.map_nil, List.map_append, List.map_map, Arg.eval,
    Wire.eval_id] at g0 g1 g2
  have hE : List.map ((Arg.eval ops V) ∘ fun l => Arg.g (ep8At K b + 3 * l + 2)) (List.range 32) =
      List.map (fun l => eTilde8 ops (beta8 ops V K) (V (K + l))) (List.range 32) := by
    apply List.map_congr_left
    intro l hl
    exact pre8_eTilde hK h l (List.mem_range.1 hl)
  rw [hE] at g1
  unfold code8 code8W form8
  rw [show code8At K b + 3 * t + 2 = code8At K b + (3 * t + 2) by omega, g2,
    show code8At K b + (3 * t + 2) - 1 = code8At K b + (3 * t + 1) by omega, g1,
    show code8At K b + (3 * t + 1) - 1 = code8At K b + 3 * t by omega, g0, (pre8_alpha hK h).1]
  rfl

end Pre8

/-! ## `pc8`'s row units -/

section Rows8

variable {ops}

theorem code8W_congr {p : Pipeline} {α x : ℕ} {et et' f f' : ℕ → ℕ} (he : ∀ l < 32, et l = et' l)
    (hf : ∀ l < 32, f l = f' l) : code8W ops p α x et f = code8W ops p α x et' f' := by
  unfold code8W form8
  rw [List.map_congr_left fun l hl => he l (List.mem_range.1 hl),
    List.map_congr_left fun l hl => hf l (List.mem_range.1 hl)]

/-- **A `pc8` row unit computes its codes**: a table whose runs start with `pc8`'s shared runs, in a transcript whose one
unit is correct, holds on position `t`'s code output `code8W` at `alpha8`'s and `beta8`'s words of the row's inputs `x`,
`x_t`, `e`'s codes and the form basis' row `t`. -/
theorem rows8_code (T : Rows) (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) (hb : 65 * K + 32 ≤ T.nIn) {rest : List Run}
    (hr : T.runs = pre8Runs p K T.nIn ++ rest) (X : (T.spec ops).toCircuit.Transcript)
    (hX : (T.spec ops).part.Correct X 0) {t : ℕ} (ht : t < K) (hout : code8 K T.nIn t ∈ T.outs) :
    val X (code8 K T.nIn t) = code8W ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)) := by
  have hlen := pre8Runs_len p K T.nIn
  have hlt : code8 K T.nIn t < (T.spec ops).N := by
    show code8 K T.nIn t < T.nIn + runsLen T.runs
    rw [hr, runsLen_append]
    unfold code8
    unfold post8At at hlen
    omega
  rw [Spec.out_eq (T.spec ops) X hX (List.mem_toFinset.2 hout) hlt]
  have hh := Rows.holds ops T X
  rw [hr] at hh
  have hp := (runsHold_append ops _ hh).1
  have hin : ∀ l < T.nIn, val ((T.spec ops).toCircuit.eval X) l = val X l := fun l hl => Rows.val_input ops T X hl
  rw [pre8_code hK hp t ht, hin t (by omega), alpha8_congr hK fun l hl => hin l (by omega),
    beta8_congr hK fun l hl => hin l (by omega)]
  exact code8W_congr (fun l hl => by rw [hin _ (by omega)]) fun l hl => hin _ (by omega)

/-- **RowA computes `A′`'s codes** (`rows8_code` at `Pc8RowA_v1{K}`). -/
theorem row8A_code (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) (X : ((row8A p K).spec ops).toCircuit.Transcript)
    (hX : ((row8A p K).spec ops).part.Correct X 0) {t : ℕ} (ht : t < K) :
    val X (code8 K (rowIn K false) t) = code8W ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)) :=
  rows8_code (row8A p K) p hK (by simp [row8A, rowIn]) rfl X hX ht
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨t, List.mem_range.2 ht,
      rfl⟩))))

/-- **RowB computes `B̃`'s codes** (`rows8_code` at `Pc8RowB_v1{K}`). -/
theorem row8B_code (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) (X : ((row8B p K).spec ops).toCircuit.Transcript)
    (hX : ((row8B p K).spec ops).part.Correct X 0) {t : ℕ} (ht : t < K) :
    val X (code8 K (rowIn K true) t) = code8W ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)) :=
  rows8_code (row8B p K) p hK (by simp [row8B, rowIn]) rfl X hX ht
    (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨t, List.mem_range.2 ht,
      rfl⟩))))

end Rows8

/-! ## The chains -/

section Chain8

open AccumProofs PeelExact PeelExactP

variable {ops} (p : Pipeline) {V : ℕ → ℕ} {cd cl : ℕ → ℕ} {o n : ℕ}

theorem atom8_eval (acc : ℕ) (a b : ℕ → ℕ) :
    Prim.eval ops (.atom8 p) (acc :: ((List.range 32).map a ++ (List.range 32).map b)) =
      p.step acc (List.ofFn fun l : Fin 32 => code (a l)) (List.ofFn fun l : Fin 32 => code (b l)) := by
  unfold Prim.eval Prim.w
  rw [Nat.mod_eq_of_lt (by unfold atomF; exact step_lt _ _ _ _)]
  simp only [atomF, Bool.false_eq_true, ↓reduceIte, List.headD_cons, List.tail_cons]
  congr 2 <;> funext l
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append_left, l.isLt]
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append_right]

/-- Values meeting `n` windows of a chain from gate `o` (`chain8`), the codes at `cd t` and the other side at `cl t`. -/
def Chain8Holds (V : ℕ → ℕ) (cd cl : ℕ → ℕ) (o n : ℕ) : Prop :=
  ∀ r < 5 * n, V (o + r) =
    (chain8 p cd cl (o + r) r).prim.eval ops ((chain8 p cd cl (o + r) r).args.map (Arg.eval ops V))

theorem chain8_atom (h : Chain8Holds (ops := ops) p V cd cl o n) {q r : ℕ} (hq : q < n) (hr : r < 4) :
    V (o + (5 * q + r)) = p.step (if r = 0 then 0 else V (o + (5 * q + r) - 1))
      (List.ofFn fun l : Fin 32 => code (V (cd (32 * (4 * q + r) + l))))
      (List.ofFn fun l : Fin 32 => code (V (cl (32 * (4 * q + r) + l)))) := by
  have h5 : (5 * q + r) % 5 = r := by omega
  have h5' : (5 * q + r) / 5 = q := by omega
  rw [h (5 * q + r) (by omega)]
  simp only [chain8, h5, h5', hr, ↓reduceIte, List.map_cons, List.map_append, List.map_map, List.cons_append]
  have hc : Arg.eval ops V (if r = 0 then Arg.c 0 else .g (o + (5 * q + r) - 1)) =
      if r = 0 then 0 else V (o + (5 * q + r) - 1) := by
    split_ifs <;> rfl
  rw [hc, atom8_eval]
  rfl

theorem chain8_add (h : Chain8Holds (ops := ops) p V cd cl o n) {q : ℕ} (hq : q < n) :
    V (o + (5 * q + 4)) = f32Add (if q = 0 then 0 else V (o + (5 * q + 4) - 5)) (V (o + (5 * q + 4) - 1)) % 2 ^ 32 := by
  have h5 : (5 * q + 4) % 5 = 4 := by omega
  have h5' : (5 * q + 4) / 5 = q := by omega
  rw [h (5 * q + 4) (by omega)]
  simp only [chain8, h5, h5', show ¬ (4 < 4) by omega, ↓reduceIte, List.map_cons, List.map_nil]
  have hc : Arg.eval ops V (if q = 0 then Arg.c 0 else .g (o + (5 * q + 4) - 5)) =
      if q = 0 then 0 else V (o + (5 * q + 4) - 5) := by
    split_ifs <;> rfl
  rw [hc]
  rfl

theorem chain8_inputs (A B : Fin (128 * n) → Code) (hA : ∀ l : Fin (128 * n), V (cd l) = (A l).val)
    (hB : ∀ l : Fin (128 * n), V (cl l) = (B l).val) {t : ℕ} (ht : t < (rowAtoms A B).length) :
    (List.ofFn fun l : Fin 32 => code (V (cd (32 * t + l)))) = List.ofFn (rowAtoms A B)[t].1 ∧
      (List.ofFn fun l : Fin 32 => code (V (cl (32 * t + l)))) = List.ofFn (rowAtoms A B)[t].2 := by
  have hlen : (rowAtoms A B).length = 128 * n / 32 := List.length_ofFn
  rw [hlen] at ht
  simp only [rowAtoms, List.getElem_ofFn]
  constructor <;> congr 1 <;> funext l
  · rw [hA ⟨32 * t + l, by omega⟩, code_val]
  · rw [hB ⟨32 * t + l, by omega⟩, code_val]

/-- **A window's word** in a row's chain: its four atoms write the window's word, a step word. -/
theorem chain8_window (h : Chain8Holds (ops := ops) p V cd cl o n) (A B : Fin (128 * n) → Code)
    (hA : ∀ l : Fin (128 * n), V (cd l) = (A l).val) (hB : ∀ l : Fin (128 * n), V (cl l) = (B l).val) {q : ℕ}
    (hq : q < n) {w : ℕ} (hw : windowWordP? p (((rowAtoms A B).drop (4 * q)).take 4) = some w) :
    V (o + (5 * q + 3)) = w ∧ StepWord w := by
  have hlen : (rowAtoms A B).length = 128 * n / 32 := List.length_ofFn
  have h3 : 4 * q + 3 < (rowAtoms A B).length := by rw [hlen]; omega
  rw [take_drop_four _ h3] at hw
  simp only [windowWordP?, List.foldlM_cons, List.foldlM_nil] at hw
  obtain ⟨w1, h1, hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain ⟨w2, h2, hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain ⟨w3, h3', hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain ⟨w4, h4, hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain rfl : w4 = w := Option.some.inj hw
  have ai := fun t (ht : t < (rowAtoms A B).length) => chain8_inputs A B hA hB ht
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h1
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h2
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h3'
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h4
  have e0 : V (o + (5 * q + 0)) = p.step 0 (List.ofFn (rowAtoms A B)[4 * q].1) (List.ofFn (rowAtoms A B)[4 * q].2) := by
    rw [chain8_atom p h hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  have e1 : V (o + (5 * q + 1)) = p.step (V (o + (5 * q + 0)))
      (List.ofFn (rowAtoms A B)[4 * q + 1].1) (List.ofFn (rowAtoms A B)[4 * q + 1].2) := by
    rw [chain8_atom p h hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  have e2 : V (o + (5 * q + 2)) = p.step (V (o + (5 * q + 1)))
      (List.ofFn (rowAtoms A B)[4 * q + 2].1) (List.ofFn (rowAtoms A B)[4 * q + 2].2) := by
    rw [chain8_atom p h hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  have e3 : V (o + (5 * q + 3)) = p.step (V (o + (5 * q + 2)))
      (List.ofFn (rowAtoms A B)[4 * q + 3].1) (List.ofFn (rowAtoms A B)[4 * q + 3].2) := by
    rw [chain8_atom p h hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  refine ⟨by rw [e3, e2, e1, e0], step_lt _ _ _ _, step_ne_negZero _ _ _ _, h4⟩

/-- **A row's chain**: its `n` windows' promotions write the chain's word over its codes `A` and the other side `B`
(`chainWordP? p 4`, the pinned order), on its last promotion. -/
theorem chain8_word (hn : 0 < n) (h : Chain8Holds (ops := ops) p V cd cl o n) (A B : Fin (128 * n) → Code)
    (hA : ∀ l : Fin (128 * n), V (cd l) = (A l).val) (hB : ∀ l : Fin (128 * n), V (cl l) = (B l).val) {c : ℕ}
    (hc : chainWordP? p 4 (rowAtoms A B) = some c) : V (o + (5 * (n - 1) + 4)) = c := by
  have hlen : (rowAtoms A B).length = 4 * n := by
    rw [show (rowAtoms A B).length = 128 * n / 32 from List.length_ofFn]; omega
  set G : ℕ → ℕ → Option ℕ := fun t q => (windowWordP? p (((rowAtoms A B).drop (4 * q)).take 4)).bind
    (promoteWord? t) with hG
  have hch : chainWordP? p 4 (rowAtoms A B) = (List.range n).foldlM G 0 := by
    rw [chainWordP?, Freeze.toChunks_eq (by norm_num), hlen, show (4 * n + 4 - 1) / 4 = n by omega,
      List.foldlM_map]
  have key : ∀ m ≤ n, ∀ c, (List.range m).foldlM G 0 = some c →
      StepWord c ∧ (if m = 0 then 0 else V (o + (5 * (m - 1) + 4))) = c := by
    intro m
    induction m with
    | zero =>
      intro _ c h
      simp only [List.range_zero, List.foldlM_nil, pure, Option.some.injEq] at h
      subst h
      exact ⟨⟨by norm_num, by decide, by decide⟩, rfl⟩
    | succ m ih =>
      intro hm c hf
      rw [List.range_succ, List.foldlM_append] at hf
      obtain ⟨t, ht, hf⟩ := Option.bind_eq_some_iff.1 hf
      simp only [List.foldlM_cons, List.foldlM_nil, hG] at hf
      obtain ⟨c', hc', hf⟩ := Option.bind_eq_some_iff.1 hf
      obtain rfl : c' = c := Option.some.inj hf
      obtain ⟨w, hw, hp⟩ := Option.bind_eq_some_iff.1 hc'
      obtain ⟨hts, htv⟩ := ih (by omega) t ht
      obtain ⟨hwv, -⟩ := chain8_window p h A B hA hB (by omega) hw
      obtain ⟨hadd, hcs⟩ := promote_step hts hp
      refine ⟨hcs, ?_⟩
      rw [ite_eq_right (Nat.succ_ne_zero m), show m + 1 - 1 = m by omega, chain8_add p h (by omega),
        show o + (5 * m + 4) - 1 = o + (5 * m + 3) by omega, hwv]
      have hprev : (if m = 0 then 0 else V (o + (5 * m + 4) - 5)) = t := by
        rcases Nat.eq_zero_or_pos m with rfl | hm0
        · simpa using htv
        · rw [ite_eq_right (by omega), show o + (5 * m + 4) - 5 = o + (5 * (m - 1) + 4) by omega]
          rwa [ite_eq_right (by omega)] at htv
      rw [hprev, f32Add_of_addWord hadd, Nat.mod_eq_of_lt hcs.1]
  obtain ⟨-, hv⟩ := key n le_rfl c (hch ▸ hc)
  rwa [ite_eq_right (by omega)] at hv

/-- **RowA computes `P_A`'s words**: at depth `K = 128·nw`, a transcript correct on RowA whose code outputs hold `A′`'s
codes `A` and whose chain basis' column `q` holds `B` has on `P_A`'s word `q` the BF16 cast of the chain's word over `A`
and `B` (`chainWordP? p 4`, the pinned order). -/
theorem row8A_pa {K nw : ℕ} (hnw : 0 < nw) (hK : K = 128 * nw) (X : ((row8A p K).spec ops).toCircuit.Transcript)
    (hX : ((row8A p K).spec ops).part.Correct X 0) (A B : Fin (128 * nw) → Code)
    (hA : ∀ t : Fin (128 * nw), val X (code8 K (rowIn K false) t) = (A t).val) {q : ℕ} (hq : q < 32)
    (hB : ∀ t : Fin (128 * nw), val X (33 * K + 32 + 32 * t + q) = (B t).val) {c : ℕ}
    (hc : chainWordP? p 4 (rowAtoms A B) = some c) :
    val X (post8At K (rowIn K false) + q * (5 * nw + 1) + 5 * nw) = Prim.eval ops .f2fpBf16 [c] := by
  have hKn : K / 128 = nw := by omega
  have hlen := pre8Runs_len p K (rowIn K false)
  have hN : (row8A p K).N = post8At K (rowIn K false) + 32 * (5 * nw + 1) + 32 + 1 := by
    show rowIn K false + runsLen (row8A p K).runs = _
    simp only [row8A, runsLen_append, runsLen, af8Run, Run.rep, bew8Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil, hKn]
    omega
  have hout : ∀ g ∈ (row8A p K).outs,
      val X g = val (((row8A p K).spec ops).toCircuit.eval X) g := fun g hg =>
    Spec.out_eq ((row8A p K).spec ops) X hX (List.mem_toFinset.2 hg) (by
      simp only [row8A, List.mem_append, List.mem_map, List.mem_range, List.mem_singleton] at hg
      show g < (row8A p K).N
      rw [hN]
      rcases hg with ((⟨t, ht, rfl⟩ | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩) | rfl
      · unfold code8; unfold post8At at hlen ⊢; omega
      · rw [hKn]; omega
      · rw [hKn]
        have : l * (5 * nw + 1) + 5 * nw < 32 * (5 * nw + 1) := by nlinarith
        omega
      · rw [hKn]; omega)
  have hmem : post8At K (rowIn K false) + q * (5 * nw + 1) + 5 * nw ∈ (row8A p K).outs := by
    simp only [row8A, List.mem_append, List.mem_map, List.mem_range, hKn]
    exact Or.inl (Or.inr ⟨q, hq, rfl⟩)
  rw [hout _ hmem]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row8A p K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row8A p K) X
  rw [← hV] at hh
  have hpost := (runsHold_append ops V hh).2
  rw [show (row8A p K).nIn + runsLen (pre8Runs p K (rowIn K false)) = post8At K (rowIn K false) from hlen] at hpost
  have haf : RunHolds ops V (af8Run p K (rowIn K false)) (post8At K (rowIn K false)) := hpost.1
  have hgate : ∀ s < 5 * nw + 1, V (post8At K (rowIn K false) + q * (5 * nw + 1) + s) =
      ((if s < 5 * nw then chain8 p (code8 K (rowIn K false)) (fun t => 33 * K + 32 + 32 * t + q)
          (post8At K (rowIn K false) + q * (5 * nw + 1) + s) s
        else ⟨.f2fpBf16, [.g (post8At K (rowIn K false) + q * (5 * nw + 1) + s - 1)]⟩ : RGate)).prim.eval ops
        (((if s < 5 * nw then chain8 p (code8 K (rowIn K false)) (fun t => 33 * K + 32 + 32 * t + q)
          (post8At K (rowIn K false) + q * (5 * nw + 1) + s) s
        else ⟨.f2fpBf16, [.g (post8At K (rowIn K false) + q * (5 * nw + 1) + s - 1)]⟩ : RGate)).args.map
          (Arg.eval ops V)) := by
    intro s hs
    have hr : q * (5 * nw + 1) + s < (af8Run p K (rowIn K false)).n := by
      simp only [af8Run, Run.rep, hKn]
      nlinarith
    have := haf _ hr
    have hd : (q * (5 * nw + 1) + s) / (5 * nw + 1) = q := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hs, Nat.zero_add]
    have hm : (q * (5 * nw + 1) + s) % (5 * nw + 1) = s := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]
    simp only [af8Run, Run.rep, hKn, hd, hm] at this
    rw [← Nat.add_assoc] at this
    exact this
  have hch : Chain8Holds (ops := ops) p V (code8 K (rowIn K false)) (fun t => 33 * K + 32 + 32 * t + q)
      (post8At K (rowIn K false) + q * (5 * nw + 1)) nw := by
    intro r hr
    have := hgate r (by omega)
    simp only [hr, ↓reduceIte] at this
    exact this
  have hA' : ∀ t : Fin (128 * nw), V (code8 K (rowIn K false) t) = (A t).val := by
    intro t
    have ht : code8 K (rowIn K false) t ∈ (row8A p K).outs := by
      simp only [row8A, List.mem_append, List.mem_map, List.mem_range]
      exact Or.inl (Or.inl (Or.inl ⟨t, by omega, rfl⟩))
    rw [hV, ← hout _ ht, hA t]
  have hB' : ∀ t : Fin (128 * nw), V (33 * K + 32 + 32 * t + q) = (B t).val := by
    intro t
    rw [hV, Rows.val_input ops (row8A p K) X (by show _ < rowIn K false; simp only [rowIn]; omega), hB t]
  have hw := chain8_word p hnw hch A B hA' hB' hc
  have hl := hgate (5 * nw) (by omega)
  simp only [show ¬ (5 * nw < 5 * nw) by omega, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id] at hl
  rw [hl, show post8At K (rowIn K false) + q * (5 * nw + 1) + 5 * nw - 1 =
    post8At K (rowIn K false) + q * (5 * nw + 1) + (5 * (nw - 1) + 4) by omega, hw]

end Chain8

/-! ## RowB's `P_B` -/

/-- `Σ_s ẽ_s·g_s` as RowB's FMAs accumulate it, `s` ascending from `+0`: the word after term `s`. -/
def pbAcc (ew g : ℕ → ℕ) : ℕ → ℕ
  | 0 => Prim.eval ops .fma [ew 0, g 0, 0]
  | s + 1 => Prim.eval ops .fma [ew (s + 1), g (s + 1), pbAcc ew g s]

section Mid8

open AccumProofs PeelExact PeelExactP

variable {ops} (p : Pipeline)

theorem pbAcc_congr {ew ew' g g' : ℕ → ℕ} : ∀ n, (∀ s ≤ n, ew s = ew' s ∧ g s = g' s) →
    pbAcc ops ew g n = pbAcc ops ew' g' n
  | 0, h => by simp only [pbAcc, (h 0 le_rfl).1, (h 0 le_rfl).2]
  | n + 1, h => by
    simp only [pbAcc, (h (n + 1) le_rfl).1, (h (n + 1) le_rfl).2, pbAcc_congr n fun s hs => h s (by omega)]

/-- **RowB computes `P_B`'s words**: at depth `K = 128·nw`, a transcript correct on RowB whose code outputs hold `B̃`'s
codes `A` and whose chain basis' column `q` holds `B` has on `P_B`'s word `q` the BF16 cast of `Σ_s ẽ_s·g_{s,q}` (the
FMAs from `+0` of the rescaled noise' FP32 words against `G`'s column `q`) plus the negated word of the chain over `A`
and `B` (`chainWordP? p 4`, the pinned order). -/
theorem row8B_pb {K nw : ℕ} (hnw : 0 < nw) (hK : K = 128 * nw) (X : ((row8B p K).spec ops).toCircuit.Transcript)
    (hX : ((row8B p K).spec ops).part.Correct X 0) (A B : Fin (128 * nw) → Code)
    (hA : ∀ t : Fin (128 * nw), val X (code8 K (rowIn K true) t) = (A t).val) {q : ℕ} (hq : q < 32)
    (hB : ∀ t : Fin (128 * nw), val X (33 * K + 32 + 32 * t + q) = (B t).val) {c : ℕ}
    (hc : chainWordP? p 4 (rowAtoms A B) = some c) :
    val X (post8At K (rowIn K true) + q * (5 * nw + 35) + 5 * nw + 34) =
      Prim.eval ops .f2fpBf16 [Prim.eval ops .add
        [pbAcc ops (fun s => Prim.eval ops .e4m3ToF32 [eTilde8 ops (beta8 ops (val X) K) (val X (K + s))])
          (fun s => val X (65 * K + 32 + 32 * s + q)) 31, Prim.eval ops .neg [c]]] := by
  have hKn : K / 128 = nw := by omega
  have h8 : 8 ≤ K := by omega
  have hlen := pre8Runs_len p K (rowIn K true)
  have hN : (row8B p K).N = post8At K (rowIn K true) + 32 * (5 * nw + 35) + 64 + 1 := by
    show rowIn K true + runsLen (row8B p K).runs = _
    simp only [row8B, runsLen_append, runsLen, mid8Run, Run.rep, nb8Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil, hKn]
    omega
  have hout : ∀ g ∈ (row8B p K).outs,
      val X g = val (((row8B p K).spec ops).toCircuit.eval X) g := fun g hg =>
    Spec.out_eq ((row8B p K).spec ops) X hX (List.mem_toFinset.2 hg) (by
      simp only [row8B, List.mem_append, List.mem_map, List.mem_range, List.mem_singleton] at hg
      show g < (row8B p K).N
      rw [hN]
      rcases hg with ((⟨t, ht, rfl⟩ | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩) | rfl
      · unfold code8; unfold post8At at hlen ⊢; omega
      · rw [hKn]
        have : l * (5 * nw + 35) + 5 * nw + 34 < 32 * (5 * nw + 35) := by nlinarith
        omega
      · omega
      · omega)
  have hmem : post8At K (rowIn K true) + q * (5 * nw + 35) + 5 * nw + 34 ∈ (row8B p K).outs := by
    simp only [row8B, List.mem_append, List.mem_map, List.mem_range, hKn]
    exact Or.inl (Or.inl (Or.inr ⟨q, hq, rfl⟩))
  rw [hout _ hmem]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row8B p K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row8B p K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre8Runs p K (rowIn K true)) (rowIn K true) := hsplit.1
  have hpost := hsplit.2
  rw [show (row8B p K).nIn + runsLen (pre8Runs p K (rowIn K true)) = post8At K (rowIn K true) from hlen] at hpost
  have hmid : RunHolds ops V (mid8Run p K (rowIn K true)) (post8At K (rowIn K true)) := hpost.1
  have hin : ∀ l < rowIn K true, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row8B p K) X hl
  have hgate : ∀ s < 5 * nw + 35, V (post8At K (rowIn K true) + q * (5 * nw + 35) + s) =
      (mid8 p K (rowIn K true) q s (post8At K (rowIn K true) + q * (5 * nw + 35) + s)).prim.eval ops
        ((mid8 p K (rowIn K true) q s (post8At K (rowIn K true) + q * (5 * nw + 35) + s)).args.map
          (Arg.eval ops V)) := by
    intro s hs
    have hr : q * (5 * nw + 35) + s < (mid8Run p K (rowIn K true)).n := by
      simp only [mid8Run, Run.rep, hKn]
      nlinarith
    have := hmid _ hr
    have hd : (q * (5 * nw + 35) + s) / (5 * nw + 35) = q := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hs, Nat.zero_add]
    have hm : (q * (5 * nw + 35) + s) % (5 * nw + 35) = s := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]
    simp only [mid8Run, Run.rep, hKn, hd, hm] at this
    rw [← Nat.add_assoc] at this
    exact this
  have hacc : ∀ s, s < 32 → V (post8At K (rowIn K true) + q * (5 * nw + 35) + s) =
      pbAcc ops (fun s => V (ew8At K (rowIn K true) + s)) (fun s => V (65 * K + 32 + 32 * s + q)) s := by
    intro s
    induction s with
    | zero =>
      intro _
      rw [hgate 0 (by omega)]
      simp only [mid8, show (0 : ℕ) < 32 by omega, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval,
        Wire.eval_id, pbAcc]
    | succ s ih =>
      intro hs
      rw [hgate (s + 1) (by omega)]
      simp only [mid8, hs, ↓reduceIte, Nat.add_one_ne_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
        pbAcc]
      rw [show post8At K (rowIn K true) + q * (5 * nw + 35) + (s + 1) - 1 =
        post8At K (rowIn K true) + q * (5 * nw + 35) + s by omega, ih (by omega)]
  have hch : Chain8Holds (ops := ops) p V (code8 K (rowIn K true)) (fun t => 33 * K + 32 + 32 * t + q)
      (post8At K (rowIn K true) + q * (5 * nw + 35) + 32) nw := by
    intro r hr
    have := hgate (32 + r) (by omega)
    simp only [mid8, show ¬ (32 + r < 32) by omega, show 32 + r < 32 + 5 * (K / 128) by omega, ↓reduceIte,
      show 32 + r - 32 = r by omega] at this
    rw [show post8At K (rowIn K true) + q * (5 * nw + 35) + 32 + r =
      post8At K (rowIn K true) + q * (5 * nw + 35) + (32 + r) by omega]
    exact this
  have hA' : ∀ t : Fin (128 * nw), V (code8 K (rowIn K true) t) = (A t).val := by
    intro t
    have ht : code8 K (rowIn K true) t ∈ (row8B p K).outs := by
      simp only [row8B, List.mem_append, List.mem_map, List.mem_range]
      exact Or.inl (Or.inl (Or.inl ⟨t, by omega, rfl⟩))
    rw [hV, ← hout _ ht, hA t]
  have hB' : ∀ t : Fin (128 * nw), V (33 * K + 32 + 32 * t + q) = (B t).val := by
    intro t
    rw [hin _ (by simp only [rowIn]; omega), hB t]
  have hw := chain8_word p hnw hch A B hA' hB' hc
  have hneg := hgate (32 + 5 * nw) (by omega)
  have hadd := hgate (33 + 5 * nw) (by omega)
  have hcast := hgate (34 + 5 * nw) (by omega)
  simp only [mid8, hKn] at hneg hadd hcast
  simp only [show ¬ (32 + 5 * nw < 32) by omega, show ¬ (32 + 5 * nw < 32 + 5 * nw) from Nat.lt_irrefl _,
    ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at hneg
  simp only [show ¬ (33 + 5 * nw < 32) by omega, show ¬ (33 + 5 * nw < 32 + 5 * nw) by omega,
    show ¬ (33 + 5 * nw = 32 + 5 * nw) by omega, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id] at hadd
  simp only [show ¬ (34 + 5 * nw < 32) by omega, show ¬ (34 + 5 * nw < 32 + 5 * nw) by omega,
    show ¬ (34 + 5 * nw = 32 + 5 * nw) by omega, show ¬ (34 + 5 * nw = 33 + 5 * nw) by omega, ↓reduceIte,
    List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at hcast
  rw [show post8At K (rowIn K true) + q * (5 * nw + 35) + 5 * nw + 34 =
      post8At K (rowIn K true) + q * (5 * nw + 35) + (34 + 5 * nw) by omega, hcast,
    show post8At K (rowIn K true) + q * (5 * nw + 35) + (34 + 5 * nw) - 1 =
      post8At K (rowIn K true) + q * (5 * nw + 35) + (33 + 5 * nw) by omega, hadd,
    show post8At K (rowIn K true) + q * (5 * nw + 35) + (33 + 5 * nw) - 2 - 5 * nw =
      post8At K (rowIn K true) + q * (5 * nw + 35) + 31 by omega,
    show post8At K (rowIn K true) + q * (5 * nw + 35) + (33 + 5 * nw) - 1 =
      post8At K (rowIn K true) + q * (5 * nw + 35) + (32 + 5 * nw) by omega, hneg,
    show post8At K (rowIn K true) + q * (5 * nw + 35) + (32 + 5 * nw) - 1 =
      post8At K (rowIn K true) + q * (5 * nw + 35) + 32 + (5 * (nw - 1) + 4) by omega, hw, hacc 31 (by omega)]
  have hE : pbAcc ops (fun s => V (ew8At K (rowIn K true) + s)) (fun s => V (65 * K + 32 + 32 * s + q)) 31 =
      pbAcc ops (fun s => Prim.eval ops .e4m3ToF32 [eTilde8 ops (beta8 ops (val X) K) (val X (K + s))])
        (fun s => val X (65 * K + 32 + 32 * s + q)) 31 := by
    refine pbAcc_congr 31 fun s hs => ⟨?_, hin _ (by simp only [rowIn, ↓reduceIte]; omega)⟩
    rw [pre8_ew h8 hpre s (by omega), hin _ (by simp only [rowIn]; omega),
      beta8_congr h8 fun l hl => hin l (by simp only [rowIn]; omega)]
  rw [hE]

/-- **RowA writes the rescaled noise' BF16 words** (`be_A`, `P_A`'s head in `paRow?`): word `l` is the BF16 cast of
`ẽ_l`'s FP32 word. -/
theorem row8A_be {K nw : ℕ} (hnw : 0 < nw) (hK : K = 128 * nw) (X : ((row8A p K).spec ops).toCircuit.Transcript)
    (hX : ((row8A p K).spec ops).part.Correct X 0) {l : ℕ} (hl : l < 32) :
    val X (post8At K (rowIn K false) + 32 * (5 * nw + 1) + l) =
      Prim.eval ops .f2fpBf16 [Prim.eval ops .e4m3ToF32 [eTilde8 ops (beta8 ops (val X) K) (val X (K + l))]] := by
  have hKn : K / 128 = nw := by omega
  have h8 : 8 ≤ K := by omega
  have hlen := pre8Runs_len p K (rowIn K false)
  have hmem : post8At K (rowIn K false) + 32 * (5 * nw + 1) + l ∈ (row8A p K).outs := by
    simp only [row8A, List.mem_append, List.mem_map, List.mem_range, hKn]
    exact Or.inl (Or.inl (Or.inr ⟨l, hl, rfl⟩))
  have hlt : post8At K (rowIn K false) + 32 * (5 * nw + 1) + l < ((row8A p K).spec ops).N := by
    show _ < rowIn K false + runsLen (row8A p K).runs
    simp only [row8A, runsLen_append, runsLen, af8Run, Run.rep, bew8Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil, hKn]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row8A p K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row8A p K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre8Runs p K (rowIn K false)) (rowIn K false) := hsplit.1
  have hpost := hsplit.2
  rw [show (row8A p K).nIn + runsLen (pre8Runs p K (rowIn K false)) = post8At K (rowIn K false) from hlen] at hpost
  have hin : ∀ l < rowIn K false, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row8A p K) X hl
  have g := hpost.2.1 l (by simp [bew8Run]; omega)
  simp only [bew8Run, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id, af8Run, Run.rep, hKn] at g
  rw [g, pre8_ew h8 hpre l hl, hin _ (by simp only [rowIn]; omega),
    beta8_congr h8 fun l hl => hin l (by simp only [rowIn]; omega)]

/-- **RowB writes the rescaled noise' BF16 words negated** (`P_B`'s tail in `pbRow?`): word `l` is the BF16 cast of
`ẽ_l`'s FP32 word, its sign flipped. -/
theorem row8B_nbe {K nw : ℕ} (hnw : 0 < nw) (hK : K = 128 * nw) (X : ((row8B p K).spec ops).toCircuit.Transcript)
    (hX : ((row8B p K).spec ops).part.Correct X 0) {l : ℕ} (hl : l < 32) :
    val X (post8At K (rowIn K true) + 32 * (5 * nw + 35) + 2 * l + 1) = Prim.eval ops .bf16Neg
      [Prim.eval ops .f2fpBf16 [Prim.eval ops .e4m3ToF32 [eTilde8 ops (beta8 ops (val X) K) (val X (K + l))]]] := by
  have hKn : K / 128 = nw := by omega
  have h8 : 8 ≤ K := by omega
  have hlen := pre8Runs_len p K (rowIn K true)
  have hmem : post8At K (rowIn K true) + 32 * (5 * nw + 35) + 2 * l + 1 ∈ (row8B p K).outs := by
    simp only [row8B, List.mem_append, List.mem_map, List.mem_range, hKn]
    exact Or.inl (Or.inr ⟨l, hl, rfl⟩)
  have hlt : post8At K (rowIn K true) + 32 * (5 * nw + 35) + 2 * l + 1 < ((row8B p K).spec ops).N := by
    show _ < rowIn K true + runsLen (row8B p K).runs
    simp only [row8B, runsLen_append, runsLen, mid8Run, Run.rep, nb8Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil, hKn]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row8B p K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row8B p K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre8Runs p K (rowIn K true)) (rowIn K true) := hsplit.1
  have hpost := hsplit.2
  rw [show (row8B p K).nIn + runsLen (pre8Runs p K (rowIn K true)) = post8At K (rowIn K true) from hlen] at hpost
  have hin : ∀ l < rowIn K true, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row8B p K) X hl
  have g1 := hpost.2.1 (2 * l + 1) (by simp [nb8Run]; omega)
  have g0 := hpost.2.1 (2 * l) (by simp [nb8Run]; omega)
  have e1 : (2 * l + 1) % 2 = 1 := by omega
  have e0 : 2 * l % 2 = 0 := by omega
  simp only [nb8Run, e0, e1, ↓reduceIte, show (1 : ℕ) ≠ 0 by omega, show 2 * l / 2 = l by omega, List.map_cons,
    List.map_nil, Arg.eval, Wire.eval_id, mid8Run, Run.rep, hKn] at g0 g1
  rw [show post8At K (rowIn K true) + 32 * (5 * nw + 35) + 2 * l + 1 =
      post8At K (rowIn K true) + 32 * (5 * nw + 35) + (2 * l + 1) by omega, g1,
    show post8At K (rowIn K true) + 32 * (5 * nw + 35) + (2 * l + 1) - 1 =
      post8At K (rowIn K true) + 32 * (5 * nw + 35) + 2 * l by omega, g0, pre8_ew h8 hpre l hl,
    hin _ (by simp only [rowIn]; omega), beta8_congr h8 fun l hl => hin l (by simp only [rowIn]; omega)]

end Mid8

/-! ## `pc8`'s flag -/

/-- `Σ x_{8i}²`, scaled, as the flag's exact accumulators sum it from `0`: the accumulator after position `8i`. -/
def sqSum (x : ℕ → ℕ) : ℕ → ℕ
  | 0 => Prim.eval ops .sqAcc [0, x 0]
  | i + 1 => Prim.eval ops .sqAcc [sqSum x i, x (8 * (i + 1))]

/-- The dead positions' count through position `t`: each `x_t²`'s comparison with `θ`, selected to `1` or `0`, summed
from `0`. -/
def deadCount (x : ℕ → ℕ) (θ : ℕ) : ℕ → ℕ
  | 0 => Prim.eval ops .iadd [0, Prim.eval ops .selI32 [Prim.eval ops .sqGe [x 0, θ], 1, 0]]
  | t + 1 => Prim.eval ops .iadd [deadCount x θ t, Prim.eval ops .selI32 [Prim.eval ops .sqGe [x (t + 1), θ], 1, 0]]

/-- `pc8`'s row flag as its table computes it: `s` and `ρ` finite, the floor `φ ≥ 0`, and at most `K/64` positions
dead against `θ`, the ceiling of the sum over every 8th position. -/
def flag8 (x : ℕ → ℕ) (K : ℕ) : ℕ :=
  Prim.eval ops .band
    [Prim.eval ops .band [Prim.eval ops .le [rowMax ops .fabs x (K - 1), 0x7F7FFFFF],
      Prim.eval ops .le [rowRho ops .id x K, 0x7F7FFFFF]],
    Prim.eval ops .band [Prim.eval ops .le [0, floor8 ops x K],
      Prim.eval ops .le [deadCount ops x (Prim.eval ops .sqCeil [sqSum ops x (K / 8 - 1), K / 8]) (K - 1), K / 64]]]

section Flag8

variable {ops} {V : ℕ → ℕ} {p : Pipeline} {K b : ℕ}

theorem sqSum_congr {x y : ℕ → ℕ} : ∀ i, (∀ l ≤ 8 * i, x l = y l) → sqSum ops x i = sqSum ops y i
  | 0, h => by simp only [sqSum, h 0 (by omega)]
  | i + 1, h => by simp only [sqSum, h (8 * (i + 1)) le_rfl, sqSum_congr i fun l hl => h l (by omega)]

theorem deadCount_congr {x y : ℕ → ℕ} {θ : ℕ} : ∀ t, (∀ l ≤ t, x l = y l) →
    deadCount ops x θ t = deadCount ops y θ t
  | 0, h => by simp only [deadCount, h 0 le_rfl]
  | t + 1, h => by simp only [deadCount, h (t + 1) le_rfl, deadCount_congr t fun l hl => h l (by omega)]

theorem flag8_congr {x y : ℕ → ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) : flag8 ops x K = flag8 ops y K := by
  unfold flag8
  rw [rowMax_congr (K - 1) fun l hl => h l (by omega), rowRho_congr hK h, floor8_congr hK h,
    sqSum_congr (K / 8 - 1) fun l hl => h l (by omega), deadCount_congr (K - 1) fun l hl => h l (by omega)]

/-- **The sum of squares**: the flag's accumulators are `sqSum`'s. -/
theorem pre8_sq (h : RunsHold ops V (pre8Runs p K b) b) : ∀ i, i < K / 8 → V (sq8At K b + i) = sqSum ops V i := by
  obtain ⟨-, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  obtain ⟨-, hs, -⟩ := hr
  have hs' : RunHolds ops V (sq8Run K b) (sq8At K b) := hs
  intro i
  induction i with
  | zero =>
    intro hi
    have g := hs' 0 hi
    simp only [sq8Run, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g
    rw [g]
    rfl
  | succ i ih =>
    intro hi
    have g := hs' (i + 1) hi
    simp only [sq8Run, Nat.add_one_ne_zero, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g
    rw [g, show sq8At K b + (i + 1) - 1 = sq8At K b + i by omega, ih (by omega)]
    rfl

/-- **The threshold**: `θ` is the ceiling of `sqSum`'s last accumulator over the `K/8` positions. -/
theorem pre8_theta (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) :
    V (th8At K b) = Prim.eval ops .sqCeil [sqSum ops V (K / 8 - 1), K / 8] := by
  obtain ⟨-, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  obtain ⟨-, -, hth, -⟩ := hr
  have hth' : RunHolds ops V (th8Run K b) (th8At K b) := hth
  have g := hth' 0 (by simp [th8Run, Run.ofList])
  simp only [th8Run, Run.ofList, List.getD_cons_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id,
    Nat.add_zero] at g
  rw [g, show th8At K b - 1 = sq8At K b + (K / 8 - 1) by unfold th8At; omega, pre8_sq h _ (by omega)]

/-- **The dead count**: through position `t`, `deadCount` against `θ`. -/
theorem pre8_dead (h : RunsHold ops V (pre8Runs p K b) b) :
    ∀ t, t < K → V (dead8At K b + 3 * t + 2) = deadCount ops V (V (th8At K b)) t := by
  obtain ⟨-, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  obtain ⟨-, -, -, hd, -⟩ := hr
  have hd' : RunHolds ops V (dead8Run K b) (dead8At K b) := hd
  have gate : ∀ t < K, V (dead8At K b + 3 * t + 1) =
      Prim.eval ops .selI32 [Prim.eval ops .sqGe [V t, V (th8At K b)], 1, 0] := by
    intro t ht
    have g0 := hd' (3 * t) (by simp [dead8Run]; omega)
    have g1 := hd' (3 * t + 1) (by simp [dead8Run]; omega)
    have e0 : 3 * t % 3 = 0 := by omega
    have e1 : (3 * t + 1) % 3 = 1 := by omega
    have d0 : 3 * t / 3 = t := by omega
    simp only [dead8Run, e0, e1, d0, ↓reduceIte, show (1 : ℕ) ≠ 0 by omega, List.map_cons, List.map_nil,
      Arg.eval, Wire.eval_id] at g0 g1
    rw [show dead8At K b + 3 * t + 1 = dead8At K b + (3 * t + 1) by omega, g1,
      show dead8At K b + (3 * t + 1) - 1 = dead8At K b + 3 * t by omega, g0]
  intro t
  induction t with
  | zero =>
    intro ht
    have g := hd' 2 (by simp [dead8Run]; omega)
    simp only [dead8Run, show (2 : ℕ) % 3 = 2 from rfl, show (2 : ℕ) ≠ 0 by omega, show (2 : ℕ) ≠ 1 by omega,
      ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g
    rw [show dead8At K b + 3 * 0 + 2 = dead8At K b + 2 by omega, g,
      show dead8At K b + 2 - 1 = dead8At K b + 3 * 0 + 1 by omega, gate 0 ht]
    rfl
  | succ t ih =>
    intro ht
    have g := hd' (3 * (t + 1) + 2) (by simp [dead8Run]; omega)
    have c0 : ¬ ((3 * (t + 1) + 2) % 3 = 0) := by omega
    have c1 : ¬ ((3 * (t + 1) + 2) % 3 = 1) := by omega
    have c2 : ¬ (3 * (t + 1) + 2 = 2) := by omega
    simp only [dead8Run, c0, c1, c2, ↓reduceIte, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g
    rw [show dead8At K b + 3 * (t + 1) + 2 = dead8At K b + (3 * (t + 1) + 2) by omega, g,
      show dead8At K b + (3 * (t + 1) + 2) - 3 = dead8At K b + 3 * t + 2 by omega, ih (by omega),
      show dead8At K b + (3 * (t + 1) + 2) - 1 = dead8At K b + 3 * (t + 1) + 1 by omega, gate (t + 1) ht]
    rfl

/-- **The flag** on values meeting `pc8`'s shared runs: `flag8` of the values of `x`'s gates. -/
theorem pre8_flag (hK : 8 ≤ K) (h : RunsHold ops V (pre8Runs p K b) b) : V (ok8At K b + 7) = flag8 ops V K := by
  obtain ⟨hs, -, ho, -⟩ := pre8_runs h
  obtain ⟨hsm, hrho⟩ := stats_holds hK hs
  have g1 := ho 1 (by simp [ok8Run, Run.ofList])
  have g2 := ho 2 (by simp [ok8Run, Run.ofList])
  have g3 := ho 3 (by simp [ok8Run, Run.ofList])
  have g4 := ho 4 (by simp [ok8Run, Run.ofList])
  have g5 := ho 5 (by simp [ok8Run, Run.ofList])
  have g6 := ho 6 (by simp [ok8Run, Run.ofList])
  have g7 := ho 7 (by simp [ok8Run, Run.ofList])
  simp only [ok8Run, Run.ofList, List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id] at g1 g2 g3 g4 g5 g6 g7
  rw [g7, g4, g3, g2, g6, g1, g5, pre8_floor hK h, hsm, hrho,
    show ok8At K b - 1 = dead8At K b + 3 * (K - 1) + 2 by unfold ok8At; omega, pre8_dead h (K - 1) (by omega),
    pre8_theta hK h]
  rfl

end Flag8

section Flags8

variable {ops} (p : Pipeline)

/-- **RowA writes its flag**: the BF16 select of `flag8` to `1` or `0`. -/
theorem row8A_flag {K : ℕ} (hK : 8 ≤ K) (X : ((row8A p K).spec ops).toCircuit.Transcript)
    (hX : ((row8A p K).spec ops).part.Correct X 0) :
    val X (post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + 32) =
      Prim.eval ops .selBf16 [flag8 ops (val X) K, 1, 0] := by
  have hlen := pre8Runs_len p K (rowIn K false)
  have hmem : post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + 32 ∈ (row8A p K).outs :=
    List.mem_append_right _ (List.mem_singleton_self _)
  have hlt : post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + 32 < ((row8A p K).spec ops).N := by
    show _ < rowIn K false + runsLen (row8A p K).runs
    simp only [row8A, runsLen_append, runsLen, af8Run, Run.rep, bew8Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row8A p K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row8A p K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre8Runs p K (rowIn K false)) (rowIn K false) := hsplit.1
  have hpost := hsplit.2
  rw [show (row8A p K).nIn + runsLen (pre8Runs p K (rowIn K false)) = post8At K (rowIn K false) from hlen] at hpost
  have hin : ∀ l < rowIn K false, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row8A p K) X hl
  have g := hpost.2.2.1 0 (by simp [okwRun, Run.ofList])
  simp only [okwRun, Run.ofList, List.getD_cons_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id, af8Run,
    Run.rep, bew8Run, Nat.add_zero] at g
  rw [g, pre8_flag hK hpre, flag8_congr hK fun l hl => hin l (by simp only [rowIn]; omega)]

/-- **RowB writes its flag**: the BF16 select of `flag8` to `1` or `0`. -/
theorem row8B_flag {K : ℕ} (hK : 8 ≤ K) (X : ((row8B p K).spec ops).toCircuit.Transcript)
    (hX : ((row8B p K).spec ops).part.Correct X 0) :
    val X (post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 64) =
      Prim.eval ops .selBf16 [flag8 ops (val X) K, 1, 0] := by
  have hlen := pre8Runs_len p K (rowIn K true)
  have hmem : post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 64 ∈ (row8B p K).outs :=
    List.mem_append_right _ (List.mem_singleton_self _)
  have hlt : post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 64 < ((row8B p K).spec ops).N := by
    show _ < rowIn K true + runsLen (row8B p K).runs
    simp only [row8B, runsLen_append, runsLen, mid8Run, Run.rep, nb8Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row8B p K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row8B p K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre8Runs p K (rowIn K true)) (rowIn K true) := hsplit.1
  have hpost := hsplit.2
  rw [show (row8B p K).nIn + runsLen (pre8Runs p K (rowIn K true)) = post8At K (rowIn K true) from hlen] at hpost
  have hin : ∀ l < rowIn K true, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row8B p K) X hl
  have g := hpost.2.2.1 0 (by simp [okwRun, Run.ofList])
  simp only [okwRun, Run.ofList, List.getD_cons_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id, mid8Run,
    Run.rep, nb8Run, Nat.add_zero] at g
  rw [g, pre8_flag hK hpre, flag8_congr hK fun l hl => hin l (by simp only [rowIn]; omega)]

end Flags8

/-! ## `pc4`'s row units -/

/-- `pc4`'s row flag as its table computes it: `s` finite, and `β` at least `2⁻⁷·15/8` and finite. -/
def flag4 (x : ℕ → ℕ) (K : ℕ) : ℕ :=
  Prim.eval ops .band [Prim.eval ops .le [rowMax ops .bf16Fabs x (K - 1), 0x7F7FFFFF],
    Prim.eval ops .band [Prim.eval ops .le [0x3C700000, beta4 ops x K], Prim.eval ops .le [beta4 ops x K, 0x7F7FFFFF]]]

section Rows4

variable {ops}

theorem alpha4_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    alpha4 ops x K = alpha4 ops y K := by
  unfold alpha4 drho4
  rw [rowRho_congr hK h, rowMax_congr (K - 1) fun l hl => h l (by omega)]

theorem beta4_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    beta4 ops x K = beta4 ops y K := by
  unfold beta4 drho4
  rw [rowRho_congr hK h, alpha4_congr hK h]

theorem flag4_congr {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (h : ∀ l < K, x l = y l) :
    flag4 ops x K = flag4 ops y K := by
  unfold flag4
  rw [rowMax_congr (K - 1) fun l hl => h l (by omega), beta4_congr hK h]

/-- **`α`, `β` and `a_E`** on values meeting `pc4`'s shared runs: `alpha4` and `beta4` of the values of `x`'s gates,
and `β`'s E4M3 cast. -/
theorem pre4_alpha {V : ℕ → ℕ} {K b : ℕ} (hK : 8 ≤ K) (h : RunsHold ops V (pre4Runs K b) b) :
    V (a4At K b + 4) = alpha4 ops V K ∧ V (a4At K b + 6) = beta4 ops V K ∧
      V (a4At K b + 7) = Prim.eval ops .toE4m3 [beta4 ops V K] := by
  obtain ⟨hs, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  have ha : RunHolds ops V (alpha4Run K b) (a4At K b) := hr.1
  obtain ⟨hsm, hrho⟩ := stats_holds hK hs
  have g0 := ha 0 (by simp [alpha4Run, Run.ofList])
  have g1 := ha 1 (by simp [alpha4Run, Run.ofList])
  have g2 := ha 2 (by simp [alpha4Run, Run.ofList])
  have g3 := ha 3 (by simp [alpha4Run, Run.ofList])
  have g4 := ha 4 (by simp [alpha4Run, Run.ofList])
  have g5 := ha 5 (by simp [alpha4Run, Run.ofList])
  have g6 := ha 6 (by simp [alpha4Run, Run.ofList])
  have g7 := ha 7 (by simp [alpha4Run, Run.ofList])
  simp only [alpha4Run, Run.ofList, List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id, Nat.add_zero] at g0 g1 g2 g3 g4 g5 g6 g7
  have hd : V (a4At K b) = drho4 ops V K := by rw [g0, hrho]; rfl
  have hα : V (a4At K b + 4) = alpha4 ops V K := by rw [g4, g3, g2, g1, hd, hsm]; rfl
  have hβ : V (a4At K b + 6) = beta4 ops V K := by rw [g6, g5, hd, hα]; rfl
  exact ⟨hα, hβ, by rw [g7, hβ]⟩

/-- **The flag** on values meeting `pc4`'s shared runs: `flag4` of the values of `x`'s gates. -/
theorem pre4_flag {V : ℕ → ℕ} {K b : ℕ} (hK : 8 ≤ K) (h : RunsHold ops V (pre4Runs K b) b) :
    V (a4At K b + 12) = flag4 ops V K := by
  obtain ⟨hs, hr⟩ := runsHold_append ops V h
  rw [statsRuns_len] at hr
  have ha : RunHolds ops V (alpha4Run K b) (a4At K b) := hr.1
  obtain ⟨hsm, -⟩ := stats_holds hK hs
  have g8 := ha 8 (by simp [alpha4Run, Run.ofList])
  have g9 := ha 9 (by simp [alpha4Run, Run.ofList])
  have g10 := ha 10 (by simp [alpha4Run, Run.ofList])
  have g11 := ha 11 (by simp [alpha4Run, Run.ofList])
  have g12 := ha 12 (by simp [alpha4Run, Run.ofList])
  simp only [alpha4Run, Run.ofList, List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Arg.eval,
    Wire.eval_id] at g8 g9 g10 g11 g12
  rw [g12, g11, g10, g9, g8, (pre4_alpha hK h).2.1, hsm]
  rfl

/-- **RowA writes the noise' `Ẽ`**: word `l` is `PouwETilde_v1` of `e`'s code `l` at `a_E`, `β`'s E4M3 cast. -/
theorem row4A_et {K : ℕ} (hK : 8 ≤ K) (X : ((row4A K).spec ops).toCircuit.Transcript)
    (hX : ((row4A K).spec ops).part.Correct X 0) {l : ℕ} (hl : l < 32) :
    val X (post4At K (rowIn K false) + 32 * (K / 64 + 1) + l) =
      Prim.eval ops .eTilde [val X (K + l), Prim.eval ops .toE4m3 [beta4 ops (val X) K]] := by
  have hlen := pre4Runs_len K (rowIn K false)
  have hmem : post4At K (rowIn K false) + 32 * (K / 64 + 1) + l ∈ (row4A K).outs := by
    simp only [row4A, List.mem_append, List.mem_map, List.mem_range]
    exact Or.inl (Or.inl (Or.inr ⟨l, hl, rfl⟩))
  have hlt : post4At K (rowIn K false) + 32 * (K / 64 + 1) + l < ((row4A K).spec ops).N := by
    show _ < rowIn K false + runsLen (row4A K).runs
    simp only [row4A, runsLen_append, runsLen, af4Run, Run.rep, et4Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row4A K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row4A K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre4Runs K (rowIn K false)) (rowIn K false) := hsplit.1
  have hpost := hsplit.2
  rw [show (row4A K).nIn + runsLen (pre4Runs K (rowIn K false)) = post4At K (rowIn K false) from hlen] at hpost
  have hin : ∀ l < rowIn K false, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row4A K) X hl
  have g := hpost.2.1 l (by simp [et4Run]; omega)
  simp only [et4Run, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id, af4Run, Run.rep] at g
  rw [g, (pre4_alpha hK hpre).2.2, hin _ (by simp only [rowIn]; omega),
    beta4_congr hK fun l hl => hin l (by simp only [rowIn]; omega)]

/-- **RowA writes its flag**: the BF16 select of `flag4` to `1` or `0`. -/
theorem row4A_flag {K : ℕ} (hK : 8 ≤ K) (X : ((row4A K).spec ops).toCircuit.Transcript)
    (hX : ((row4A K).spec ops).part.Correct X 0) :
    val X (post4At K (rowIn K false) + 32 * (K / 64 + 1) + 32) =
      Prim.eval ops .selBf16 [flag4 ops (val X) K, 1, 0] := by
  have hlen := pre4Runs_len K (rowIn K false)
  have hmem : post4At K (rowIn K false) + 32 * (K / 64 + 1) + 32 ∈ (row4A K).outs :=
    List.mem_append_right _ (List.mem_singleton_self _)
  have hlt : post4At K (rowIn K false) + 32 * (K / 64 + 1) + 32 < ((row4A K).spec ops).N := by
    show _ < rowIn K false + runsLen (row4A K).runs
    simp only [row4A, runsLen_append, runsLen, af4Run, Run.rep, et4Run, okwRun, Run.ofList, List.length_cons,
      List.length_nil]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row4A K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row4A K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre4Runs K (rowIn K false)) (rowIn K false) := hsplit.1
  have hpost := hsplit.2
  rw [show (row4A K).nIn + runsLen (pre4Runs K (rowIn K false)) = post4At K (rowIn K false) from hlen] at hpost
  have hin : ∀ l < rowIn K false, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row4A K) X hl
  have g := hpost.2.2.1 0 (by simp [okwRun, Run.ofList])
  simp only [okwRun, Run.ofList, List.getD_cons_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id, af4Run,
    Run.rep, et4Run, Nat.add_zero] at g
  rw [g, pre4_flag hK hpre, flag4_congr hK fun l hl => hin l (by simp only [rowIn]; omega)]

/-- **RowB writes the noise' `−Ẽ`**: word `l` is `PouwBf16Neg_v1` of `PouwETilde_v1` of `e`'s code `l` at `a_E`. -/
theorem row4B_net {K : ℕ} (hK : 8 ≤ K) (X : ((row4B K).spec ops).toCircuit.Transcript)
    (hX : ((row4B K).spec ops).part.Correct X 0) {l : ℕ} (hl : l < 32) :
    val X (post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + l) = Prim.eval ops .bf16Neg
      [Prim.eval ops .eTilde [val X (K + l), Prim.eval ops .toE4m3 [beta4 ops (val X) K]]] := by
  have hlen := pre4Runs_len K (rowIn K true)
  have hmem : post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + l ∈ (row4B K).outs := by
    simp only [row4B, List.mem_append, List.mem_map, List.mem_range]
    exact Or.inl (Or.inr ⟨l, hl, rfl⟩)
  have hlt : post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + l < ((row4B K).spec ops).N := by
    show _ < rowIn K true + runsLen (row4B K).runs
    simp only [row4B, runsLen_append, runsLen, et4Run, mid4Run, Run.rep, nb4Run, okwRun, Run.ofList,
      List.length_cons, List.length_nil]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row4B K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row4B K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre4Runs K (rowIn K true)) (rowIn K true) := hsplit.1
  have hpost := hsplit.2
  rw [show (row4B K).nIn + runsLen (pre4Runs K (rowIn K true)) = post4At K (rowIn K true) from hlen] at hpost
  have hin : ∀ l < rowIn K true, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row4B K) X hl
  have g0 := hpost.1 l (by simp [et4Run]; omega)
  have g1 := hpost.2.2.1 l (by simp [nb4Run]; omega)
  simp only [et4Run, nb4Run, mid4Run, Run.rep, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id] at g0 g1
  rw [g1, g0, (pre4_alpha hK hpre).2.2, hin _ (by simp only [rowIn]; omega),
    beta4_congr hK fun l hl => hin l (by simp only [rowIn]; omega)]

/-- **RowB writes its flag**: the BF16 select of `flag4` to `1` or `0`. -/
theorem row4B_flag {K : ℕ} (hK : 8 ≤ K) (X : ((row4B K).spec ops).toCircuit.Transcript)
    (hX : ((row4B K).spec ops).part.Correct X 0) :
    val X (post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + 32) =
      Prim.eval ops .selBf16 [flag4 ops (val X) K, 1, 0] := by
  have hlen := pre4Runs_len K (rowIn K true)
  have hmem : post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + 32 ∈ (row4B K).outs :=
    List.mem_append_right _ (List.mem_singleton_self _)
  have hlt : post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + 32 < ((row4B K).spec ops).N := by
    show _ < rowIn K true + runsLen (row4B K).runs
    simp only [row4B, runsLen_append, runsLen, et4Run, mid4Run, Run.rep, nb4Run, okwRun, Run.ofList,
      List.length_cons, List.length_nil]
    omega
  rw [Spec.out_eq _ X hX (List.mem_toFinset.2 hmem) hlt]
  obtain ⟨V, hV⟩ : ∃ V, V = val (((row4B K).spec ops).toCircuit.eval X) := ⟨_, rfl⟩
  rw [← hV]
  have hh := Rows.holds ops (row4B K) X
  rw [← hV] at hh
  have hsplit := runsHold_append ops V hh
  have hpre : RunsHold ops V (pre4Runs K (rowIn K true)) (rowIn K true) := hsplit.1
  have hpost := hsplit.2
  rw [show (row4B K).nIn + runsLen (pre4Runs K (rowIn K true)) = post4At K (rowIn K true) from hlen] at hpost
  have hin : ∀ l < rowIn K true, V l = val X l := fun l hl => by
    rw [hV]; exact Rows.val_input ops (row4B K) X hl
  have g := hpost.2.2.2.1 0 (by simp [okwRun, Run.ofList])
  simp only [okwRun, Run.ofList, List.getD_cons_zero, List.map_cons, List.map_nil, Arg.eval, Wire.eval_id, et4Run,
    mid4Run, Run.rep, nb4Run, Nat.add_zero] at g
  rw [g, pre4_flag hK hpre, flag4_congr hK fun l hl => hin l (by simp only [rowIn]; omega)]

end Rows4

/-! ## Words read back as values -/

section RoundTrip

theorem wordQ_exp_ne {w : ℕ} {y : ℚ} (h : wordQ w = some y) : (w >>> 23) % 256 ≠ 255 := by
  intro he
  simp only [wordQ, fp32Value, he, ite_true] at h
  split_ifs at h <;> simp [Value.finite?] at h

theorem sign_mul_neg_iff (b : Prop) [Decidable b] {M : ℚ} (hM : 0 < M) :
    (if b then (-1 : ℚ) else 1) * M < 0 ↔ b := by
  by_cases hb : b
  · simp only [hb, ↓reduceIte, iff_true]; linarith
  · simp only [hb, ↓reduceIte, iff_false, not_lt]; linarith

/-- A finite nonzero word's value is negative exactly when its sign bit is set. -/
theorem wordQ_neg_iff {w : ℕ} {y : ℚ} (h : wordQ w = some y) (hy : y ≠ 0) : y < 0 ↔ (w >>> 31) % 2 = 1 := by
  rw [PeelExactProofs.wordQ_fields (wordQ_exp_ne h)] at h
  obtain rfl := Option.some.inj h
  have hpos : (0 : ℚ) ≤ (((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) : ℚ) *
      (2 : ℚ) ^ (((max ((w >>> 23) % 256) 1 : ℕ) : ℤ) - 150) :=
    mul_nonneg (Nat.cast_nonneg _) (zpow_nonneg (by norm_num) _)
  have hne : (((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) : ℚ) *
      (2 : ℚ) ^ (((max ((w >>> 23) % 256) 1 : ℕ) : ℤ) - 150) ≠ 0 := by
    intro h0; apply hy; rw [h0, mul_zero]
  exact sign_mul_neg_iff _ (lt_of_le_of_ne hpos (Ne.symm hne))

/-- **A step's word is `f32Word` of its value**: below `2^32`, finite and not `−0` (`StepWord`). -/
theorem f32Word_wordVal {w : ℕ} (hw : AccumProofs.StepWord w) : f32Word (wordVal w) = w := by
  have he := AccumProofs.finite?_exp hw.2.2
  obtain ⟨y, hy⟩ : ∃ y, wordQ w = some y := ⟨_, PeelExactProofs.wordQ_fields he⟩
  have hv : wordVal w = y := by simp [wordVal, hy]
  rw [hv]
  by_cases hy0 : y = 0
  · subst hy0
    rw [AccumProofs.wordQ_zero_word hw hy]
    simp [f32Word]
  · exact AccumProofs.f32Word_word hw.1 he hy hy0

theorem e4m3Mag_480 : e4m3Mag 480 = 126 := by
  unfold e4m3Mag; norm_num

/-- **The cast of a word is `e4m3` of its value** unless the word is a NaN or a zero with its sign bit set. -/
theorem e4m3_wordVal {w : ℕ} (hnan : (w >>> 23) % 256 = 255 → w % 2 ^ 23 = 0)
    (hz : wordQ w = some 0 → (w >>> 31) % 2 = 0) : e4m3 (wordVal w) = e4m3Word w := by
  unfold e4m3Word wordVal
  rcases hq : wordQ w with _ | y
  · have he : (w >>> 23) % 256 = 255 := by
      by_contra he; rw [PeelExactProofs.wordQ_fields he] at hq; simp at hq
    simp only [hnan he, ↓reduceIte]
    split_ifs with hs
    · ext; norm_num [e4m3, e4m3Mag_480]
    · ext; norm_num [e4m3, e4m3Mag_480]
  · ext
    simp only [e4m3]
    congr 2
    by_cases hy : y = 0
    · have := hz (hq ▸ congrArg some hy)
      simp [hy, this]
    · have := wordQ_neg_iff hq hy
      by_cases hs : (w >>> 31) % 2 = 1
      · simp [hs, this.2 hs]
      · have : ¬ y < 0 := fun h => hs (this.1 h)
        simp [hs, this]

/-- **A code is `e4m3` of its formed word's value**, where the cast is `e4m3Word`'s and the formed word is neither a
NaN nor a zero with its sign bit set. -/
theorem code8W_e4m3 (hcast : ∀ u, ops.f32ToE4m3Sat u = (e4m3Word u).val) (p : Pipeline) (α x : ℕ) (et f : ℕ → ℕ)
    (hnan : (form8 ops p α x et f >>> 23) % 256 = 255 → form8 ops p α x et f % 2 ^ 23 = 0)
    (hz : wordQ (form8 ops p α x et f) = some 0 → (form8 ops p α x et f >>> 31) % 2 = 0) :
    code8W ops p α x et f = (e4m3 (wordVal (form8 ops p α x et f))).val := by
  rw [e4m3_wordVal hnan hz]
  simp only [code8W, Prim.eval, List.getD_cons_zero, hcast, Prim.w]
  exact Nat.mod_eq_of_lt (e4m3Word _).isLt

/-- **`pc8Form` at the values of step words** is the formed word `form8` at those words, read as a value. -/
theorem pc8Form_value (p : Pipeline) {a x b : ℕ} (ha : AccumProofs.StepWord a) (hx : AccumProofs.StepWord x)
    (hb : AccumProofs.StepWord b) (e f : Fin Params.pi.r → Code) :
    (pc8Form ops p).value (wordVal a) (wordVal x) (wordVal b) e f =
      wordVal (form8 ops p a x (fun l => eTilde8 ops b (encCodes e l)) (encCodes f)) := by
  simp only [pc8Form, f32Word_wordVal ha, f32Word_wordVal hx, f32Word_wordVal hb]

/-- A word the cast reads as its value (`e4m3_wordVal`): not a NaN, and not a zero with its sign bit set. -/
def CastWord (w : ℕ) : Prop := ((w >>> 23) % 256 = 255 → w % 2 ^ 23 = 0) ∧ (wordQ w = some 0 → (w >>> 31) % 2 = 0)

/-- The `r` codes on the gates from `o`. -/
def gateCodes (V : ℕ → ℕ) (o : ℕ) : Fin Params.pi.r → Code := fun l => ⟨V (o + l) % 256, Nat.mod_lt _ (by decide)⟩

theorem encCodes_gateCodes {V : ℕ → ℕ} {o : ℕ} (h : ∀ l < Params.pi.r, V (o + l) < 256) {l : ℕ}
    (hl : l < Params.pi.r) : encCodes (gateCodes V o) l = V (o + l) := by
  simp only [encCodes, hl, ↓reduceDIte, gateCodes]
  exact Nat.mod_eq_of_lt (h l hl)

theorem form8_congr (p : Pipeline) (α x : ℕ) {et et' f f' : ℕ → ℕ} (he : ∀ l < 32, et l = et' l)
    (hf : ∀ l < 32, f l = f' l) : form8 ops p α x et f = form8 ops p α x et' f' := by
  unfold form8
  rw [List.map_congr_left (fun l hl => he l (List.mem_range.1 hl)),
    List.map_congr_left (fun l hl => hf l (List.mem_range.1 hl))]

/-- **`code8W` is `pc8Sem`'s forming code** on a row's words, where the cast is `e4m3Word`'s, `α`, `x_t` and `β` are
step words, the noise' and basis' codes are below `256`, and the formed word is a `CastWord`. -/
theorem codes8_sem (hcast : ∀ u, ops.f32ToE4m3Sat u = (e4m3Word u).val) (p : Pipeline) (V : ℕ → ℕ) (K t : ℕ)
    (ha : AccumProofs.StepWord (alpha8 ops V K)) (hx : AccumProofs.StepWord (V t))
    (hb : AccumProofs.StepWord (beta8 ops V K)) (he : ∀ l < Params.pi.r, V (K + l) < 256)
    (hf : ∀ l < Params.pi.r, V (K + 32 + 32 * t + l) < 256)
    (hD : CastWord (form8 ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
      (fun l => V (K + 32 + 32 * t + l)))) :
    code8W ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
        (fun l => V (K + 32 + 32 * t + l)) =
      (e4m3 ((pc8Form ops p).value (wordVal (alpha8 ops V K)) (wordVal (V t)) (wordVal (beta8 ops V K))
        (gateCodes V K) (gateCodes V (K + 32 + 32 * t)))).val := by
  rw [code8W_e4m3 ops hcast p _ _ _ _ hD.1 hD.2, pc8Form_value ops p ha hx hb,
    form8_congr ops p _ _ (fun l hl => by rw [encCodes_gateCodes he hl]) (fun l hl => encCodes_gateCodes hf hl)]

/-- **RowA's codes are `pc8Sem`'s**: `e4m3` of `pc8Form` at `α`, `x_t` and `β` read as values and the noise line's and
`F_t`'s codes, under `codes8_sem`'s conditions. -/
theorem row8A_code_sem (hcast : ∀ u, ops.f32ToE4m3Sat u = (e4m3Word u).val) (p : Pipeline) {K : ℕ} (hK : 8 ≤ K)
    (X : ((row8A p K).spec ops).toCircuit.Transcript) (hX : ((row8A p K).spec ops).part.Correct X 0) {t : ℕ}
    (ht : t < K) (ha : AccumProofs.StepWord (alpha8 ops (val X) K)) (hx : AccumProofs.StepWord (val X t))
    (hb : AccumProofs.StepWord (beta8 ops (val X) K)) (he : ∀ l < Params.pi.r, val X (K + l) < 256)
    (hf : ∀ l < Params.pi.r, val X (K + 32 + 32 * t + l) < 256)
    (hD : CastWord (form8 ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)))) :
    val X (code8 K (rowIn K false) t) =
      (e4m3 ((pc8Form ops p).value (wordVal (alpha8 ops (val X) K)) (wordVal (val X t))
        (wordVal (beta8 ops (val X) K)) (gateCodes (val X) K) (gateCodes (val X) (K + 32 + 32 * t)))).val := by
  rw [row8A_code (ops := ops) p hK X hX ht]
  exact codes8_sem ops hcast p (val X) K t ha hx hb he hf hD

/-- **RowB's codes are `pc8Sem`'s**, as `row8A_code_sem`. -/
theorem row8B_code_sem (hcast : ∀ u, ops.f32ToE4m3Sat u = (e4m3Word u).val) (p : Pipeline) {K : ℕ} (hK : 8 ≤ K)
    (X : ((row8B p K).spec ops).toCircuit.Transcript) (hX : ((row8B p K).spec ops).part.Correct X 0) {t : ℕ}
    (ht : t < K) (ha : AccumProofs.StepWord (alpha8 ops (val X) K)) (hx : AccumProofs.StepWord (val X t))
    (hb : AccumProofs.StepWord (beta8 ops (val X) K)) (he : ∀ l < Params.pi.r, val X (K + l) < 256)
    (hf : ∀ l < Params.pi.r, val X (K + 32 + 32 * t + l) < 256)
    (hD : CastWord (form8 ops p (alpha8 ops (val X) K) (val X t)
      (fun l => eTilde8 ops (beta8 ops (val X) K) (val X (K + l))) (fun l => val X (K + 32 + 32 * t + l)))) :
    val X (code8 K (rowIn K true) t) =
      (e4m3 ((pc8Form ops p).value (wordVal (alpha8 ops (val X) K)) (wordVal (val X t))
        (wordVal (beta8 ops (val X) K)) (gateCodes (val X) K) (gateCodes (val X) (K + 32 + 32 * t)))).val := by
  rw [row8B_code (ops := ops) p hK X hX ht]
  exact codes8_sem ops hcast p (val X) K t ha hx hb he hf hD

/-- A row of words read as values. -/
def rowVals (V : ℕ → ℕ) (K : ℕ) : Fin K → ℚ := fun i => wordVal (V i)

theorem encRow_rowVals {V : ℕ → ℕ} {K : ℕ} (hV : ∀ l < K, AccumProofs.StepWord (V l)) :
    ∀ l < K, encRow K (rowVals V K) l = V l := by
  intro l hl
  simp only [encRow, hl, ↓reduceDIte, rowVals, f32Word_wordVal (hV l hl)]

/-- **`pc8Sem`'s `α`, `β` and floor at a row of step words** are the table's `alpha8`, `beta8` and `floor8` words read
as values (the floor plus one where `rowRho`'s word is finite, else `0`). -/
theorem pc8Sem_scales {Q R S : Type} (p : Pipeline)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r)
    {V : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (hV : ∀ l < K, AccumProofs.StepWord (V l)) :
    (pc8Sem ops p noise).alpha K (rowVals V K) = wordVal (alpha8 ops V K) ∧
      (pc8Sem ops p noise).beta K (rowVals V K) = wordVal (beta8 ops V K) ∧
      (pc8Sem ops p noise).floor K (rowVals V K) =
        if (wordQ (rowRho ops .id V K)).isSome then wordVal (floor8 ops V K) + 1 else 0 := by
  have h := encRow_rowVals hV
  exact ⟨by simp only [pc8Sem, alpha8_congr (ops := ops) hK h], by simp only [pc8Sem, beta8_congr (ops := ops) hK h],
    by simp only [pc8Sem, floor8_congr (ops := ops) hK h, rowRho_congr (ops := ops) hK h]⟩

end RoundTrip

/-! ## E4M3 and FP32 words are never NaN -/

section NotNaN

open Pouw.Fp8Atom.Proofs

theorem roundEven_le_of_lt {m : ℚ} {n : ℤ} (h : m < n + 1 / 2) : roundEven m ≤ n := by
  have hf : ⌊m⌋ < n + 1 := Int.floor_lt.2 (by push_cast; linarith)
  rcases (show ⌊m⌋ < n ∨ ⌊m⌋ = n by omega) with hl | he
  · have := roundEven_le m; omega
  · unfold roundEven; dsimp only
    rw [he, ite_eq_left (by linarith)]

/-- `e4m3`'s magnitude bits on a non-negative value are at most `126` (`448`). -/
theorem e4m3Mag_le {t : ℚ} (ht : 0 ≤ t) : e4m3Mag t ≤ 126 := by
  rcases ht.lt_or_eq with ht | rfl
  swap
  · simp [e4m3Mag]
  unfold e4m3Mag
  rw [ite_eq_right ht.ne']
  by_cases h464 : 464 ≤ t
  · rw [ite_eq_left h464]
  rw [ite_eq_right h464]
  rw [not_le] at h464
  dsimp only
  obtain ⟨hb1, hb2⟩ := binade_spec ht
  have hb9 : binade t < 9 := by
    have : (2 : ℚ) ^ binade t < 2 ^ (9 : ℤ) := lt_of_le_of_lt hb1 (by norm_num; linarith)
    exact (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 this
  have hE8 : max (binade t) (-6) ≤ 8 := max_le (by omega) (by norm_num)
  have hu : (0 : ℚ) < 2 ^ (max (binade t) (-6) - 3) := two_zpow_pos _
  have k16 : (2 : ℚ) ^ (max (binade t) (-6) - 3) * 16 = 2 ^ (max (binade t) (-6) + 1) := by
    rw [show (16 : ℚ) = 2 ^ (4 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero]; congr 1; ring
  have hlt : t < 2 ^ (max (binade t) (-6) + 1) :=
    lt_of_lt_of_le hb2 (zpow_le_zpow_right₀ one_le_two (by omega))
  have h16 : t / 2 ^ (max (binade t) (-6) - 3) < ((16 : ℤ) : ℚ) + 1 / 2 := by
    rw [div_lt_iff₀ hu]; push_cast; linarith
  have hR := roundEven_le_of_lt h16
  rcases (show max (binade t) (-6) ≤ 7 ∨ max (binade t) (-6) = 8 by omega) with h7 | h8
  · omega
  · have h14 : t / 2 ^ (max (binade t) (-6) - 3) < ((14 : ℤ) : ℚ) + 1 / 2 := by
      rw [h8, div_lt_iff₀ (by norm_num)]; norm_num; linarith
    have := roundEven_le_of_lt h14
    omega

/-- `f32Word` never writes a NaN. -/
theorem f32Word_notNaN (y : ℚ) : (f32Word y >>> 23) % 256 = 255 → f32Word y % 2 ^ 23 = 0 := by
  unfold f32Word
  dsimp only
  by_cases h0 : |y| = 0
  · rw [ite_eq_left h0]; simp
  rw [ite_eq_right h0]
  have hN := f32Word_N (lt_of_le_of_ne (abs_nonneg y) (Ne.symm h0))
  generalize (roundEven (|y| / 2 ^ (max (binade |y|) (-126) - 23))).toNat = N₀ at hN ⊢
  generalize max (binade |y|) (-126) = E₀
  simp only [Nat.shiftRight_eq_div_pow]
  split_ifs <;> omega

/-- `e4m3` of a value is never the E4M3 NaN. -/
theorem e4m3_notNaN (v : ℚ) : (e4m3 v).val % 128 ≠ 127 := by
  have h := e4m3Mag_le (abs_nonneg v)
  unfold e4m3
  dsimp only
  split_ifs <;> omega

end NotNaN

end Pouw.PearlC.RowCircuit
