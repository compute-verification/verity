module

public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.SamplingGeneric.TwoDeployed
public import Proofs.Flock.Claims.Sampling.Reference.DeployedCode
public import Proofs.Flock.Claims.Sampling.Select

/-!
# The two-stage sampled verifier's code from its Lean samples

`SamplingDeploy.codeTwo c reg L₁ L₂ bytes₁ bytes₂`: the record's code (`SamplingDeploy.Code`) with its checked units
the two stages the Lean sampled verifier samples (`SampledProofs.Run.run` under `Select.Rule.twoStage`), on the
registration the statement names (`reg x`, a `RegTwo`), and the rest `c`'s:

* the first stage is `Select.select` on the boundary registration (`RegTwo.first`, a `Reg` whose rule is the resolved
  two-stage rule) over `L₁ x` bytes of the coins (`bytes₁`). Its proved units are the replay sample the second stage
  reads (`RegTwo.sample`), and the replay sample of record (`replayTwo`) is its checked slots on the replay layout
  (`Reg.checked`);
* the second stage is `Select.selectSecond` on that sample, `k` of each sampled replay block's `n` proof blocks, over
  `L₂ x` bytes (`bytes₂`), interior unit `j` being proof block `blockOf sample[j / n] (j % n)` of the statement's
  partition (`RegTwo.blockAt`). The checked units are those proof blocks and the ones the registration always checks
  (`always`); every proof block is checked where either stage is refused, where the run rejects.

`RegMatchesTwo` says the registration is the statement's two-stage design: its replay layout and first-stage design are
the design's replay partition and first stage; the layout's population slot `ρ` is unit `ρ`, and that replay block's
proof blocks are `blockOf ρ` on `[0, n)`, one cell, in a stratum of which the design samples `k`; an input slot's cells
are always checked. So the deployed second stage, one uniform `k`-subset per sampled replay block, is the contract's
sample of each cell.

* `replayDraws_code`: `Code.ReplayDraws` from `RegMatchesTwo` and uniform first-stage bytes (`BytesUniform`), by
  `drawMisses`.
* `secondDraws_code`: `Code.SecondDraws` from `RegMatchesTwo`, first-stage bytes the second stage's coins don't move
  (`FirstFixed`) and second-stage bytes uniform on them (`SecondUniform`), by `secondMisses`: a replay block whose
  cells meet no proof block of `B`, or a sample entry outside the replay sample, only drops an escape at most `1`
  (`prod_second_le`).
* `drawFixed_codeTwo`: `Code.DrawFixed` from `BytesFixedTwo` (the sessions' rounds are second-stage, and neither stage's
  bytes read them).
* `ksTwo_code`: `CompiledTwo.KS` at Flock's masked record and SHA-512 (`ksTwo_masked`) with those three facts
  proved. It takes the code's refusals and interior roots, and its sessions' runs, scope and openings at a replay
  block.
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling Finset Properties.SampledProofs Properties.SampledProofs.SecurityProofs
open Lean (Json)

/-! ## The registration and its samples -/

/-- **A two-stage registration as the samples of record read it**: the boundary registration the first stage samples on
(`first`), the proof blocks in each replay block (`n`) and the second stage's count in each (`k`), the statement's
proof block at position `p` of replay block `ρ` (`blockOf ρ p`), and the proof blocks checked whatever the stages sample
(`always`). -/
structure RegTwo where
  first : Reg
  n : ℕ
  k : ℕ
  blockOf : ℕ → ℕ → ℕ
  always : ℕ → Bool

/-- **The replay sample** on the bytes `src`: the units the first stage proves (`Select.select`'s), which `Run.run`
hands the second stage; none where it is refused. -/
def RegTwo.sample (g : RegTwo) (src : ByteArray) : Array ℕ :=
  match SampledProofs.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure src with
  | .ok (_, proved, _) => proved
  | .error _ => #[]

/-- Interior unit `j`'s proof block: position `j % n` of replay block `sample[j / n]`. -/
def RegTwo.blockAt (g : RegTwo) (sample : Array ℕ) (j : ℕ) : ℕ := g.blockOf (sample.getD (j / g.n) 0) (j % g.n)

/-- **The checked proof blocks** below `N`, at first-stage bytes `src₁` and second-stage bytes `src₂`: those the second
stage proves (`Select.selectSecond` on the replay sample) and those always checked; all of them where the second stage
is refused. -/
def RegTwo.checked (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) : List ℕ :=
  match SampledProofs.Select.selectSecond (g.sample src₁) g.n g.k src₂ with
  | .ok (draw, _) => (List.range N).filter fun v =>
      g.always v || (SampledProofs.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v
  | .error _ => List.range N

/-- **The two-stage sampled verifier's code with its Lean samples**: `c` with `checked` the two stages' on the
registration `reg x`, over the bytes `bytes₁ x` and `bytes₂ x` of the coins. -/
def codeTwo (c : Code) (reg : Msg → Option RegTwo) (L₁ L₂ : Msg → ℕ)
    (bytes₁ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₁ x) → Fin 256)
    (bytes₂ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256) : Code :=
  { c with
    checked := fun x r =>
      ((reg x).map fun g => g.checked (units c x) (bytesOf (bytes₁ x r)) (bytesOf (bytes₂ x r))).getD [] }

/-- **The replay sample of record**: the first stage's checked slots on the replay layout (`Reg.checked`). -/
def replayTwo (c : Code) (reg : Msg → Option RegTwo) (L₁ : Msg → ℕ)
    (bytes₁ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₁ x) → Fin 256) (x : Msg)
    (r : Fin (c.V.k x) → FlockDeploy.Coin) : List ℕ :=
  ((reg x).map fun g => g.first.checked (bytesOf (bytes₁ x r))).getD []

theorem RegTwo.checked_length_le (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) :
    (g.checked N src₁ src₂).length ≤ N := by
  unfold RegTwo.checked
  split
  · exact List.filter_sublist.length_le.trans (by simp)
  · simp

theorem RegTwo.always_mem (g : RegTwo) {N v : ℕ} (src₁ src₂ : ByteArray) (hv : v < N) (ha : g.always v = true) :
    v ∈ g.checked N src₁ src₂ := by
  unfold RegTwo.checked
  split
  · exact List.mem_filter.2 ⟨List.mem_range.2 hv, by simp [ha]⟩
  · exact List.mem_range.2 hv

theorem RegTwo.error_mem (g : RegTwo) {N v : ℕ} {src₁ src₂ : ByteArray} {e : String}
    (h : SampledProofs.Select.selectSecond (g.sample src₁) g.n g.k src₂ = .error e) (hv : v < N) :
    v ∈ g.checked N src₁ src₂ := by
  unfold RegTwo.checked
  rw [h]
  exact List.mem_range.2 hv

theorem RegTwo.ok_mem (g : RegTwo) {N j : ℕ} {src₁ src₂ : ByteArray} {draw : Json} {read : ℕ}
    (h : SampledProofs.Select.selectSecond (g.sample src₁) g.n g.k src₂ = .ok (draw, read))
    (hj : (j : Int) ∈ SampledProofs.Draw.proved draw) (hv : g.blockAt (g.sample src₁) j < N) :
    g.blockAt (g.sample src₁) j ∈ g.checked N src₁ src₂ := by
  unfold RegTwo.checked
  rw [h]
  refine List.mem_filter.2 ⟨List.mem_range.2 hv, ?_⟩
  rw [Bool.or_eq_true, Array.any_eq_true']
  exact Or.inr ⟨j, hj, by simp⟩

/-- A nonempty replay sample is the first stage's accepted sample. -/
theorem RegTwo.sample_ok (g : RegTwo) {src : ByteArray} (h : g.sample src ≠ #[]) :
    ∃ draw read, SampledProofs.Select.select g.first.rule g.first.program g.first.query g.first.table
      g.first.closure src = .ok (draw, g.sample src, read) := by
  unfold RegTwo.sample at h ⊢
  split
  · rename_i draw proved read _
    exact ⟨draw, read, by simp_all⟩
  · rename_i hs
    rw [hs] at h
    exact absurd rfl h

theorem slots_select_of (rule : SampledProofs.Select.Rule) (p : SampledProofs.Program) (query : Json)
    (table : Option Json) (cm : Option (Array (Int × Array Int))) (src : ByteArray) {l : SampledProofs.Slots.Layout}
    {draw : Json} {proved : Array ℕ} {read : ℕ} (hl : SampledProofs.Slots.layout p query = .ok l)
    (hs : SampledProofs.Select.select rule p query table cm src = .ok (draw, proved, read)) :
    SampledProofs.Slots.select rule p query table cm src = .ok (l, SampledProofs.Slots.checked l proved) := by
  unfold SampledProofs.Slots.select
  rw [hl]
  show (SampledProofs.Select.select rule p query table cm src >>= _) = _
  rw [hs]
  rfl

/-- A checked population slot is a proved unit. -/
theorem mem_slots_checked {l : SampledProofs.Slots.Layout} {proved : Array ℕ} {ρ : ℕ}
    {b : SampledProofs.Slots.ProofBlock} (h : ρ ∈ SampledProofs.Slots.checked l proved) (hb : l.blocks[ρ]? = some b)
    (hu : b.unit = some ρ) : ρ ∈ proved := by
  unfold SampledProofs.Slots.checked at h
  simp only [Array.mem_map, Array.mem_filter] at h
  obtain ⟨⟨b', i⟩, ⟨hmem, hb'⟩, rfl⟩ := h
  rw [Array.mem_zipIdx_iff_getElem?, hb] at hmem
  obtain rfl := Option.some.inj hmem
  rw [hu] at hb'
  simpa using hb'

theorem pairwise_of_zip_tail : ∀ l : List ℕ,
    (l.zip l.tail).all (fun p => match p with | (a, b) => decide (a < b)) = true → l.Pairwise (· < ·)
  | [], _ => .nil
  | [_], _ => List.pairwise_singleton _ _
  | a :: b :: rest, h => by
    simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at h
    have ih := pairwise_of_zip_tail (b :: rest) h.2
    refine List.Pairwise.cons (fun y hy => ?_) ih
    rcases List.mem_cons.1 hy with rfl | hy
    · exact h.1
    · exact h.1.trans (List.rel_of_pairwise_cons ih hy)

/-- **An accepted second stage's sample** is one or more replay blocks, each once. -/
theorem selectSecond_sample {sample : Array ℕ} {n k : ℕ} {src : ByteArray} {r : Json × ℕ}
    (h : SampledProofs.Select.selectSecond sample n k src = .ok r) : sample ≠ #[] ∧ sample.toList.Nodup := by
  unfold SampledProofs.Select.selectSecond at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  rename_i hs
  simp only [Bool.or_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, not_or, Array.isEmpty_iff] at hs
  refine ⟨hs.1, (pairwise_of_zip_tail _ ?_).imp (fun h => (Nat.ne_of_lt h))⟩
  simpa using hs.2

/-! ## The escapes the second stage meets -/

/-- **The interior units at a proof block of `B`**: those below `Q · n` whose proof block is in `B` and below `N`. -/
def secondX (B : Finset ℕ) (N n Q : ℕ) (sample : Array ℕ) (blockOf : ℕ → ℕ → ℕ) : Finset ℕ :=
  (range (Q * n)).filter fun j => blockOf (sample.getD (j / n) 0) (j % n) ∈ B ∧
    blockOf (sample.getD (j / n) 0) (j % n) < N

/-- Sample entry `i`'s window of `secondX` is `B` on replay block `sample[i]`'s proof blocks. -/
theorem card_window_second {B : Finset ℕ} {N n Q i ρ : ℕ} {sample : Array ℕ} {blockOf : ℕ → ℕ → ℕ}
    (hn : 0 < n) (hi : i < Q) (hρ : sample.getD i 0 = ρ) (hinj : Set.InjOn (blockOf ρ) (Set.Iio n))
    (hN : ∀ p < n, blockOf ρ p < N) :
    ((secondX B N n Q sample blockOf).filter fun j => i * n ≤ j ∧ j < (i + 1) * n).card =
      (B ∩ (range n).image (blockOf ρ)).card := by
  refine card_bij (fun j _ => blockOf ρ (j - i * n)) ?_ ?_ ?_
  · intro j hj
    simp only [secondX, mem_filter, mem_range] at hj
    obtain ⟨⟨-, hX, -⟩, hlo, hhi⟩ := hj
    have hdiv : j / n = i := Nat.div_eq_of_lt_le hlo hhi
    have hmod : j % n = j - i * n := by rw [Nat.mod_eq_sub_mul_div, hdiv, mul_comm]
    rw [hdiv, hρ, hmod] at hX
    exact mem_inter.2 ⟨hX, mem_image.2 ⟨j - i * n, mem_range.2 (by rw [Nat.succ_mul] at hhi; omega), rfl⟩⟩
  · intro j₁ hj₁ j₂ hj₂ h
    simp only [secondX, mem_filter, mem_range] at hj₁ hj₂
    rw [Nat.succ_mul] at hj₁ hj₂
    have := hinj (show j₁ - i * n < n by omega) (show j₂ - i * n < n by omega) h
    omega
  · intro v hv
    obtain ⟨hvX, hvI⟩ := mem_inter.1 hv
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hvI
    rw [mem_range] at hp
    refine ⟨p + i * n, ?_, by simp⟩
    have hdiv : (p + i * n) / n = i := by rw [Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hp, zero_add]
    have hmod : (p + i * n) % n = p := by rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp]
    have hQ : (i + 1) * n ≤ Q * n := Nat.mul_le_mul_right n hi
    simp only [secondX, mem_filter, mem_range, hdiv, hmod, hρ]
    refine ⟨⟨by rw [Nat.succ_mul] at hQ; omega, hvX, hN p hp⟩, by omega, by rw [Nat.succ_mul]; omega⟩

theorem sample_getD_inj {sample : Array ℕ} (hnd : sample.toList.Nodup) {i j : ℕ} (hi : i < sample.size)
    (hj : j < sample.size) (h : sample.getD i 0 = sample.getD j 0) : i = j := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Array.getElem?_eq_getElem hj,
    Option.getD_some] at h
  rw [← Array.getElem_toList hi, ← Array.getElem_toList hj] at h
  exact (hnd.getElem_inj_iff).1 h

/-- **The product over the sample is at most the product over the replay sample**: dropping the sample's entries
outside `P` only drops escapes at most `1`, each replay block of `P` is one entry and one cell of `n` proof blocks, and
`B` meets no cell of the other replay blocks of `S`. -/
theorem prod_second_le {Pt : Claims.Sampling.Partition} {d : Claims.Sampling.Design.Two} {B : Finset ℕ} {n k : ℕ}
    {sample : Array ℕ} {blockOf : ℕ → ℕ → ℕ} {S P : Finset ℕ} (hn : 0 < n) (hnd : sample.toList.Nodup)
    (hPS : P ⊆ S)
    (hP : ∀ ρ ∈ P, ρ ∈ sample ∧ Set.InjOn (blockOf ρ) (Set.Iio n) ∧ ∃ t₀ < Pt.strata.length, d.k.getD t₀ 0 = k ∧
      ∀ t, cell Pt d.part ρ t = if t = t₀ then (range n).image (blockOf ρ) else ∅)
    (hrest : ∀ ρ ∈ S, ρ ∉ P → ∀ t ∈ range Pt.strata.length, B ∩ cell Pt d.part ρ t = ∅) :
    ∏ i ∈ range sample.size,
        escCell n k (((secondX B Pt.blocks.length n sample.size sample blockOf).filter
          fun j => i * n ≤ j ∧ j < (i + 1) * n).card) ≤
      ∏ ρ ∈ S, ∏ t ∈ range Pt.strata.length,
        escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (B ∩ cell Pt d.part ρ t).card := by
  set e : ℕ → ℝ := fun ρ => ∏ t ∈ range Pt.strata.length,
    escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (B ∩ cell Pt d.part ρ t).card with he
  set f : ℕ → ℝ := fun i => escCell n k (((secondX B Pt.blocks.length n sample.size sample blockOf).filter
    fun j => i * n ≤ j ∧ j < (i + 1) * n).card) with hf
  set I := (range sample.size).filter fun i => sample.getD i 0 ∈ P with hI
  have heP : ∀ i ∈ I, f i = e (sample.getD i 0) := by
    intro i hi
    obtain ⟨hiQ, hiP⟩ := mem_filter.1 hi
    obtain ⟨-, hinj, t₀, ht₀, hk, hcell⟩ := hP _ hiP
    have hN : ∀ p < n, blockOf (sample.getD i 0) p < Pt.blocks.length := fun p hp =>
      mem_range.1 (cell_subset Pt d.part (sample.getD i 0) t₀
        (by rw [hcell]; simp only [↓reduceIte]; exact mem_image_of_mem _ (mem_range.2 hp)))
    have hcard : ((range n).image (blockOf (sample.getD i 0))).card = n := by
      rw [card_image_of_injOn (by rw [coe_range]; exact hinj), card_range]
    simp only [hf, he]
    rw [card_window_second hn (mem_range.1 hiQ) rfl hinj hN, prod_eq_single t₀]
    · rw [hcell]; simp only [↓reduceIte]; rw [hcard, hk]
    · intro t _ ht
      rw [hcell]; simp only [ht, ↓reduceIte]; rw [card_empty, inter_empty, card_empty, escCell_zero]
    · exact fun h => absurd (mem_range.2 ht₀) h
  have he1 : ∀ ρ ∈ S, ρ ∉ P → e ρ = 1 := fun ρ hρ hρP =>
    prod_eq_one fun t ht => by rw [hrest ρ hρ hρP t ht, card_empty, escCell_zero]
  have hinj : Set.InjOn (fun i => sample.getD i 0) I := fun i hi j hj h =>
    sample_getD_inj hnd (mem_range.1 (mem_filter.1 hi).1) (mem_range.1 (mem_filter.1 hj).1) h
  have himg : I.image (fun i => sample.getD i 0) = P := by
    ext ρ
    refine ⟨fun h => ?_, fun h => ?_⟩
    · obtain ⟨i, hi, rfl⟩ := mem_image.1 h
      exact (mem_filter.1 hi).2
    · obtain ⟨i, hi, hρ⟩ := Array.mem_iff_getElem.1 (hP ρ h).1
      have hg : sample.getD i 0 = ρ := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some, hρ]
      exact mem_image.2 ⟨i, mem_filter.2 ⟨mem_range.2 hi, hg ▸ h⟩, hg⟩
  have hf0 : ∀ i, 0 ≤ f i := fun i => escCell_nonneg _ _ _
  have hf1 : ∀ i, f i ≤ 1 := fun i => escCell_le_one _ _ _
  calc ∏ i ∈ range sample.size, f i
      = (∏ i ∈ I, f i) * ∏ i ∈ (range sample.size).filter (fun i => sample.getD i 0 ∉ P), f i :=
        (prod_filter_mul_prod_filter_not _ _ _).symm
    _ ≤ ∏ i ∈ I, f i :=
        mul_le_of_le_one_right (prod_nonneg fun i _ => hf0 i) (prod_le_one₀ (fun i _ => hf0 i) fun i _ => hf1 i)
    _ = ∏ ρ ∈ I.image (fun i => sample.getD i 0), e ρ := by
        rw [prod_image hinj]
        exact prod_congr rfl heP
    _ = ∏ ρ ∈ P, e ρ := by rw [himg]
    _ = ∏ ρ ∈ S, e ρ := prod_subset hPS fun ρ hS hnP => he1 ρ hS hnP

/-! ## The stages' miss probabilities -/

/-- **The first stage misses `T` with probability at most `P_a(T)`**: the sample of record's checked slots on uniform
bytes (`drawMisses`); a refused sample checks every slot, and a well-formed design has `P_a(∅) = 1`. -/
theorem replay_le (g : Reg) (l : SampledProofs.Slots.Layout) (Dd : Claims.Sampling.Design) (L : ℕ) (T : Finset ℕ)
    (hl : SampledProofs.Slots.layout g.program g.query = .ok l)
    (hds : slotDesign g.rule g.program g.query g.table = .ok Dd) (hwf : Dd.WellFormed (slotPartition l))
    (hT : T ⊆ range (slotPartition l).blocks.length) :
    Pr (fun src : Fin L → Fin 256 => ∀ ρ ∈ g.checked (bytesOf src), ρ ∉ T) ≤ Dd.Pa (slotPartition l) T := by
  by_cases hT0 : T = ∅
  · subst hT0
    rw [Design.Pa_empty _ _ hwf]
    exact Pr_le_one _
  obtain ⟨v0, hv0⟩ := nonempty_iff_ne_empty.2 hT0
  have hlen : (slotPartition l).blocks.length = l.blocks.size := by
    rw [slotPartition_blocks, List.length_map, Array.length_toList]
  have hlay : g.layout = l := by simp only [Reg.layout, hl]
  have hb := drawMisses g.rule g.program g.query g.table g.closure L T
  simp only [missBound, hl, hds] at hb
  refine le_trans (Pr_mono fun src hm => ?_) hb
  unfold Reg.checked drawChecked at hm
  split at hm
  · rename_i l' c' hs
    refine ⟨by rw [hs]; rfl, ?_⟩
    simp only [hs, checkedOf]
    intro v hv hi
    exact hm v (Array.mem_toList_iff.2 hi) hv
  · exfalso
    rw [hlay] at hm
    have hv0l : v0 < l.blocks.size := hlen ▸ mem_range.1 (hT hv0)
    exact hm v0 (Array.mem_toList_iff.2 ((mem_allSlots _ _).2 hv0l)) hv0

/-- **The second stage misses `B` no more often than the replay sample's escapes**, at fixed first-stage bytes
`src₁` and uniform second-stage bytes: where `B` meets no cell of the replay sample `rl`, the bound is `1`; where an
always-checked proof block is in `B`, the miss probability is `0`; otherwise `secondMisses`'s product over the sample,
each entry one replay block of `rl` (`prod_second_le`). -/
theorem secondDraws_le (g : RegTwo) (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design.Two)
    (l : SampledProofs.Slots.Layout) (src₁ : ByteArray) (rl : List ℕ) (B : Finset ℕ) (L₂ : ℕ)
    (hlen : d.part.blocks.length = l.blocks.size)
    (hrl : ∀ draw proved read, SampledProofs.Select.select g.first.rule g.first.program g.first.query g.first.table
      g.first.closure src₁ = .ok (draw, proved, read) → rl = (SampledProofs.Slots.checked l proved).toList)
    (hblocks : ∀ ρ b, l.blocks[ρ]? = some b →
      (b.unit = some ρ ∧ Set.InjOn (g.blockOf ρ) (Set.Iio g.n) ∧ ∃ t₀ < Pt.strata.length, d.k.getD t₀ 0 = g.k ∧
        ∀ t, cell Pt d.part ρ t = if t = t₀ then (range g.n).image (g.blockOf ρ) else ∅) ∨
      (b.unit = none ∧ ∀ t, ∀ v ∈ cell Pt d.part ρ t, g.always v = true)) :
    Pr (fun src₂ : Fin L₂ → Fin 256 => ∀ u ∈ g.checked Pt.blocks.length src₁ (bytesOf src₂), u ∉ B) ≤
      ∏ ρ ∈ (range d.part.blocks.length).filter (· ∈ rl), ∏ t ∈ range Pt.strata.length,
        escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (B ∩ cell Pt d.part ρ t).card := by
  classical
  set N := Pt.blocks.length with hNdef
  set S := (range d.part.blocks.length).filter (· ∈ rl) with hS
  set sample := g.sample src₁ with hsample
  have hR0 : 0 ≤ ∏ ρ ∈ S, ∏ t ∈ range Pt.strata.length,
      escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (B ∩ cell Pt d.part ρ t).card :=
    prod_nonneg fun _ _ => prod_nonneg fun _ _ => escCell_nonneg _ _ _
  by_cases hA : ∃ ρ ∈ S, ∃ t ∈ range Pt.strata.length, ∃ v, v ∈ B ∩ cell Pt d.part ρ t
  swap
  · push Not at hA
    rw [prod_eq_one fun ρ hρ => prod_eq_one fun t ht => by
      rw [eq_empty_of_forall_notMem (hA ρ hρ t ht), card_empty, escCell_zero]]
    exact Pr_le_one _
  obtain ⟨ρ₀, -, t₀, -, v₀, hv₀⟩ := hA
  obtain ⟨hv₀X, hv₀c⟩ := mem_inter.1 hv₀
  have hv₀N : v₀ < N := mem_range.1 (cell_subset Pt d.part ρ₀ t₀ hv₀c)
  by_cases hB : ∃ v ∈ B, v < N ∧ g.always v = true
  · obtain ⟨v, hvX, hvN, hav⟩ := hB
    calc _ ≤ Pr (fun _ : Fin L₂ → Fin 256 => False) := Pr_mono fun src₂ hm => hm v (g.always_mem _ _ hvN hav) hvX
      _ = 0 := Pr_false
      _ ≤ _ := hR0
  push Not at hB
  let E₂ : (Fin L₂ → Fin 256) → Prop := fun src₂ =>
    (SampledProofs.Select.selectSecond sample g.n g.k (bytesOf src₂)).isOk = true ∧
      ∀ j ∈ secondX B N g.n sample.size sample g.blockOf,
        (j : Int) ∉ secondOf (SampledProofs.Select.selectSecond sample g.n g.k (bytesOf src₂))
  have hmono : Pr (fun src₂ : Fin L₂ → Fin 256 => ∀ u ∈ g.checked N src₁ (bytesOf src₂), u ∉ B) ≤ Pr E₂ := by
    refine Pr_mono fun src₂ hm => ?_
    cases hs : SampledProofs.Select.selectSecond sample g.n g.k (bytesOf src₂) with
    | error e => exact absurd hv₀X (hm v₀ (g.error_mem hs hv₀N))
    | ok p =>
      obtain ⟨draw, read⟩ := p
      refine ⟨by rw [hs]; rfl, fun j hj hjd => ?_⟩
      rw [hs] at hjd
      simp only [secondX, mem_filter] at hj
      exact hm _ (g.ok_mem hs hjd hj.2.2) hj.2.1
  by_cases hex : ∃ src₂, E₂ src₂
  swap
  · rw [Pr_congr fun src₂ => iff_false_intro fun h => hex ⟨src₂, h⟩, Pr_false] at hmono
    exact hmono.trans hR0
  obtain ⟨src0, hok0, -⟩ := hex
  obtain ⟨⟨draw0, read0⟩, h0⟩ : ∃ p, SampledProofs.Select.selectSecond sample g.n g.k (bytesOf src0) = .ok p := by
    cases h : SampledProofs.Select.selectSecond sample g.n g.k (bytesOf src0) with
    | error e => rw [h] at hok0; exact absurd hok0 (by simp [Except.isOk, Except.toBool])
    | ok p => exact ⟨p, rfl⟩
  obtain ⟨hne, hnd⟩ := selectSecond_sample h0
  have hn2 := (selectSecond_ok h0).1
  obtain ⟨draw₁, read₁, hsel⟩ := g.sample_ok (src := src₁) hne
  have hrl' := hrl _ _ _ hsel
  have hblk : ∀ ρ < l.blocks.size, ∃ b, l.blocks[ρ]? = some b := fun ρ hρ => ⟨_, Array.getElem?_eq_getElem hρ⟩
  refine hmono.trans ((secondMisses sample g.n g.k L₂ (secondX B N g.n sample.size sample g.blockOf)).trans ?_)
  refine prod_second_le (by omega) hnd (filter_subset (fun ρ => (l.blocks[ρ]?.map (·.unit)) = some (some ρ)) S) ?_ ?_
  · intro ρ hρ
    obtain ⟨hρS, hu⟩ := mem_filter.1 hρ
    obtain ⟨hρlen, hρrl⟩ := mem_filter.1 hρS
    obtain ⟨b, hb⟩ := hblk ρ (hlen ▸ mem_range.1 hρlen)
    rw [hb] at hu
    simp only [Option.map_some, Option.some.injEq] at hu
    rcases hblocks ρ b hb with ⟨-, hinj, t₀, ht₀, hk, hcell⟩ | ⟨hnone, -⟩
    · refine ⟨?_, hinj, t₀, ht₀, hk, hcell⟩
      rw [hrl'] at hρrl
      exact mem_slots_checked (Array.mem_toList_iff.1 hρrl) hb hu
    · rw [hnone] at hu; cases hu
  · intro ρ hρS hρP t _
    refine eq_empty_of_forall_notMem fun v hv => ?_
    obtain ⟨hvX, hvc⟩ := mem_inter.1 hv
    obtain ⟨b, hb⟩ := hblk ρ (hlen ▸ mem_range.1 (mem_filter.1 hρS).1)
    rcases hblocks ρ b hb with ⟨hu, -⟩ | ⟨-, hal⟩
    · exact hρP (mem_filter.2 ⟨hρS, by rw [hb, Option.map_some, hu]⟩)
    · exact hB v hvX (mem_range.1 (cell_subset Pt d.part ρ t hvc)) (hal t v hvc)

/-! ## The code's two-stage facts -/

variable (c : Code) (reg : Msg → Option RegTwo) (two : AuditStatement → Option Claims.Sampling.Design.Two)
  (L₁ L₂ : Msg → ℕ) (bytes₁ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₁ x) → Fin 256)
  (bytes₂ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256)

/-- **The registration is the statement's two-stage design**: on a statement in the two-stage scope, the statement
names a registration whose replay layout (`Slots.layout`) is the design's replay partition and whose first-stage design
(`slotDesign`) is the design's; the layout's population slot `ρ` is unit `ρ`, whose proof blocks in the statement's
partition are `blockOf ρ` on `[0, n)`, injectively, the one cell of `ρ`, in a stratum of which the design samples `k`;
and every proof block in an input slot's cells is always checked. -/
def RegMatchesTwo : Prop :=
  ∀ x s Pt d, c.parse x = some s → c.partition s = some Pt → two s = some d → InScopeTwo Pt d s →
    ∃ g l, reg x = some g ∧ SampledProofs.Slots.layout g.first.program g.first.query = .ok l ∧
      d.part = slotPartition l ∧ slotDesign g.first.rule g.first.program g.first.query g.first.table = .ok d.replay ∧
      ∀ ρ b, l.blocks[ρ]? = some b →
        (b.unit = some ρ ∧ Set.InjOn (g.blockOf ρ) (Set.Iio g.n) ∧ ∃ t₀ < Pt.strata.length, d.k.getD t₀ 0 = g.k ∧
          ∀ t, cell Pt d.part ρ t = if t = t₀ then (range g.n).image (g.blockOf ρ) else ∅) ∨
        (b.unit = none ∧ ∀ t, ∀ v ∈ cell Pt d.part ρ t, g.always v = true)

/-- **The second stage's coins don't move the first stage's bytes.** -/
def FirstFixed : Prop :=
  ∀ x r e, bytes₁ x (forkOn (c.stage2 x) r e) = bytes₁ x r

/-- **The second stage's bytes are uniform on its coins**: at any coins, fresh second-stage coins make them uniform. -/
def SecondUniform : Prop :=
  ∀ x (r : Fin (c.V.k x) → FlockDeploy.Coin) (E : (Fin (L₂ x) → Fin 256) → Prop),
    Pr (fun e : Fin (c.V.k x) → FlockDeploy.Coin => E (bytes₂ x (forkOn (c.stage2 x) r e))) = Pr E

/-- **The sessions' rounds are second-stage and neither stage's bytes read them.** -/
def BytesFixedTwo : Prop :=
  ∀ x u, c.rounds x u ⊆ c.stage2 x ∧
    ∀ r d, bytes₁ x (forkOn (c.rounds x u) r d) = bytes₁ x r ∧ bytes₂ x (forkOn (c.rounds x u) r d) = bytes₂ x r

theorem replayDraws_code (hreg : RegMatchesTwo c reg two) (hunif : BytesUniform c L₁ bytes₁) :
    (codeTwo c reg L₁ L₂ bytes₁ bytes₂).ReplayDraws two (replayTwo c reg L₁ bytes₁) := by
  intro x s Pt d hx hP h2 hin T hT
  obtain ⟨g, l, hg, hl, hpart, hds, -⟩ := hreg x s Pt d hx hP h2 hin
  have hrs : ∀ r, replayTwo c reg L₁ bytes₁ x r = g.first.checked (bytesOf (bytes₁ x r)) := fun r => by
    simp only [replayTwo, hg, Option.map_some, Option.getD_some]
  have hwf : d.replay.WellFormed (slotPartition l) := by rw [← hpart]; exact hin.2.2.2.2.1
  show Pr (fun r : Fin (c.V.k x) → FlockDeploy.Coin => ∀ ρ ∈ replayTwo c reg L₁ bytes₁ x r, ρ ∉ T) ≤
    d.replay.Pa d.part T
  simp only [hrs]
  rw [hpart] at hT ⊢
  exact (hunif x fun src => ∀ ρ ∈ g.first.checked (bytesOf src), ρ ∉ T).trans_le
    (replay_le g.first l d.replay (L₁ x) T hl hds hwf hT)

theorem secondDraws_code (hreg : RegMatchesTwo c reg two) (hfirst : FirstFixed c L₁ bytes₁)
    (hsec : SecondUniform c L₂ bytes₂) :
    (codeTwo c reg L₁ L₂ bytes₁ bytes₂).SecondDraws two (replayTwo c reg L₁ bytes₁) := by
  intro x s Pt d hx hP h2 hin r B
  obtain ⟨g, l, hg, hl, hpart, -, hblocks⟩ := hreg x s Pt d hx hP h2 hin
  have hx' : c.parse x = some s := hx
  have hP' : c.partition s = some Pt := hP
  have hN : units c x = Pt.blocks.length := by simp [units, hx', hP']
  have hchk : ∀ e : Fin (c.V.k x) → FlockDeploy.Coin,
      (((reg x).map fun g => g.checked (units c x) (bytesOf (bytes₁ x (forkOn (c.stage2 x) r e)))
        (bytesOf (bytes₂ x (forkOn (c.stage2 x) r e)))).getD []).take (units c x) =
      g.checked Pt.blocks.length (bytesOf (bytes₁ x r)) (bytesOf (bytes₂ x (forkOn (c.stage2 x) r e))) := by
    intro e
    rw [hg, Option.map_some, Option.getD_some, hfirst x r e, hN]
    exact List.take_of_length_le (g.checked_length_le _ _ _)
  have hrs : replayTwo c reg L₁ bytes₁ x r = g.first.checked (bytesOf (bytes₁ x r)) := by
    simp only [replayTwo, hg, Option.map_some, Option.getD_some]
  show Pr (fun e : Fin (c.V.k x) → FlockDeploy.Coin => ∀ u ∈ (((reg x).map fun g => g.checked (units c x)
      (bytesOf (bytes₁ x (forkOn (c.stage2 x) r e))) (bytesOf (bytes₂ x (forkOn (c.stage2 x) r e)))).getD []).take
      (units c x), u ∉ B) ≤ _
  simp only [hchk]
  refine (hsec x r fun src₂ => ∀ u ∈ g.checked Pt.blocks.length (bytesOf (bytes₁ x r)) (bytesOf src₂), u ∉ B).trans_le
    (secondDraws_le g Pt d l (bytesOf (bytes₁ x r)) _ B (L₂ x) ?_ ?_ hblocks)
  · rw [hpart, slotPartition_blocks, List.length_map, Array.length_toList]
  · intro draw proved read hs
    rw [hrs]
    unfold Reg.checked drawChecked
    simp only [slots_select_of _ _ _ _ _ _ hl hs]

theorem drawFixed_codeTwo (h : BytesFixedTwo c L₁ L₂ bytes₁ bytes₂) : (codeTwo c reg L₁ L₂ bytes₁ bytes₂).DrawFixed := by
  intro x r u _
  refine ⟨(h x u).1, fun d => ?_⟩
  show ((reg x).map fun g => g.checked (units c x) (bytesOf (bytes₁ x (forkOn (c.rounds x u) r d)))
      (bytesOf (bytes₂ x (forkOn (c.rounds x u) r d)))).getD [] =
    ((reg x).map fun g => g.checked (units c x) (bytesOf (bytes₁ x r)) (bytesOf (bytes₂ x r))).getD []
  rw [((h x u).2 r d).1, ((h x u).2 r d).2]

/-- **The deployed two-stage sampled verifier over Flock's masked record is knowledge sound at SHA-512** with its
Lean samples: `ksTwo_masked` at `codeTwo`, with its replay and second-stage samples (`replayDraws_code`,
`secondDraws_code`) and its sample's fixedness (`drawFixed_codeTwo`) proved from the registration's match and its bytes'
facts. It takes the code's refusals and interior roots, and its sessions' runs, scope and openings at a replay
block. -/
theorem ksTwo_code (iroots : Msg → Seen → ℕ → List Msg) (hreg : RegMatchesTwo c reg two)
    (hunif : BytesUniform c L₁ bytes₁) (hfirst : FirstFixed c L₁ bytes₁) (hsec : SecondUniform c L₂ bytes₂)
    (hfixed : BytesFixedTwo c L₁ L₂ bytes₁ bytes₂) (hr : c.RefusesTwo FlockDeploy.sha512 two)
    (hi : c.InteriorFixed FlockDeploy.sha512 iroots)
    (hruns : Claims.Sampling.SessionRuns FlockDeploy.sha512
      (deploy _root_.Flock.Readable.relation (codeTwo c reg L₁ L₂ bytes₁ bytes₂)))
    (hnum : SessionsIn (codeTwo c reg L₁ L₂ bytes₁ bytes₂))
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (deploy _root_.Flock.Readable.relation (codeTwo c reg L₁ L₂ bytes₁ bytes₂)) two iroots) :
    (compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeTwo c reg L₁ L₂ bytes₁ bytes₂) iroots two
      (replayTwo c reg L₁ bytes₁)).KS FlockDeploy.sha512 :=
  ksTwo_masked _ iroots two _ (replayDraws_code c reg two L₁ L₂ bytes₁ bytes₂ hreg hunif)
    (secondDraws_code c reg two L₁ L₂ bytes₁ bytes₂ hreg hfirst hsec) hr hi
    (drawFixed_codeTwo c reg L₁ L₂ bytes₁ bytes₂ hfixed) hruns hnum hopens

end SamplingDeploy
