import Proofs.Pouw.PearlPlus.Peel

/-!
# Pearl+'s bridge, part (b): a tile's words are the semantics' (the proof)

`WordsAreTheSemantics` (`Words.lean`), read through the check forward, one step at a time:

* `fixedNoise_ok`: the fixed operand's noise holds `F_A` and `F_B` as `k` rows of 32 codes, their 32 columns, and the Gram
  words as the chains of `F_B`'s and `F_A`'s columns.
* `formV1_ok`: `formV1` forms each row by `s5_stats` and its two maps, which `formRow_quant8` reads.
* A row's facts: `s5_stats` refuse an empty row (`s5Stats_pos`, its ρ is a root of a quotient by `0`, never finite), a
  chain's rows are whole atoms (`chain_size`), and a chain's word is finite (`chainWordP?_wordQ`).
* `tileWords`: over a noise whose lines are the check's, a tile entry's `C̃` and `U` are `chainWordP?` and `peelExactP`
  at the H100 on `pc8Sem`'s operands, given the peel factor rows (`PeelARow`, `PeelBRow`).
* `wordsAreTheSemantics_of`: at -h3's noise (`Noise.lean`) those are the protocol's ticket and useful words.
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

namespace SemSim

open Pouw.PearlC.RowCircuit Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactP WordsSim

/-! ## Lists, arrays and folds -/

section Lists

variable {α β : Type}

theorem ofFn_mapM {f : α → Option β} : ∀ {n : ℕ} {a : Fin n → α} {b : Fin n → β}, (∀ p, f (a p) = some (b p)) →
    (List.ofFn a).mapM f = some (List.ofFn b)
  | 0, _, _, _ => by simp only [List.ofFn_zero, List.mapM_nil] <;> rfl
  | n + 1, a, b, h => by
    have ih := ofFn_mapM (f := f) (a := fun p => a p.succ) (b := fun p => b p.succ) (fun p => h p.succ)
    rw [List.ofFn_succ (f := a), List.ofFn_succ (f := b), List.mapM_cons, h 0, ih]
    rfl

theorem foldlM_inv (P : β → Prop) (f : β → α → Option β) (hf : ∀ b a b', P b → f b a = some b' → P b') :
    ∀ (l : List α) (b b' : β), P b → l.foldlM f b = some b' → P b'
  | [], b, b', hb, h => by
    simp only [List.foldlM_nil, pure, Option.some.injEq] at h
    exact h ▸ hb
  | a :: l, b, b', hb, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.1 h
    exact foldlM_inv P f hf l s b' (hf b a s hb hs) h

theorem push_get!_lt [Inhabited α] {a : Array α} {v : α} {y : ℕ} (h : y < a.size) : (a.push v)[y]! = a[y]! := by
  have h' : y < (a.push v).size := by simp only [Array.size_push]; omega
  rw [getElem!_pos _ y h', Array.getElem_push h', dif_pos h, getElem!_pos a y h]

theorem push_get!_eq [Inhabited α] {a : Array α} {v : α} {y : ℕ} (h : a.size = y) : (a.push v)[y]! = v := by
  have h' : y < (a.push v).size := by simp only [Array.size_push]; omega
  have hn : ¬ y < a.size := by omega
  rw [getElem!_pos _ y h', Array.getElem_push h', dif_neg hn]

theorem zip_toList_get [Inhabited α] [Inhabited β] {a : Array α} {b : Array β} {j : ℕ}
    (h : j < (a.zip b).toList.length) (ha : j < a.size) (hb : j < b.size) : (a.zip b).toList[j] = (a[j]!, b[j]!) := by
  rw [Array.getElem_toList, Array.getElem_zip, getElem!_pos a j ha, getElem!_pos b j hb]

theorem lmap_get! [Inhabited α] [Inhabited β] {l : List α} {g : α → β} {x : ℕ} (h : x < l.length) :
    (l.map g)[x]! = g l[x]! := by
  rw [getElem!_pos (l.map g) x (by simpa using h), List.getElem_map, getElem!_pos l x h]

theorem zipL_get! [Inhabited α] [Inhabited β] {l : List α} {g : α → β} {x : ℕ} (h : x < l.length) :
    (l.zip (l.map g)).toArray[x]! = (l[x]!, g l[x]!) := by
  have h' : x < (l.zip (l.map g)).length := by simpa using h
  rw [List.getElem!_toArray, getElem!_pos _ x h', List.getElem_zip, List.getElem_map, getElem!_pos l x h]

theorem get!_mem [Inhabited α] {l : List α} {x : ℕ} (h : x < l.length) : l[x]! ∈ l := by
  rw [getElem!_pos l x h]
  exact List.getElem_mem h

theorem range_get! {n i : ℕ} (h : i < n) : (Array.range n)[i]! = i := by
  rw [getElem!_pos (Array.range n) i (by simpa using h), Array.getElem_range]

theorem amap_get! [Inhabited α] [Inhabited β] {a : Array α} {g : α → β} {i : ℕ} (h : i < a.size) :
    (a.map g)[i]! = g a[i]! := by
  rw [getElem!_pos (a.map g) i (by simpa using h), Array.getElem_map, getElem!_pos a i h]

end Lists

/-! ## A row's facts -/

/-- A quotient by `0` is the NaN or an infinity, whose root is never finite. -/
theorem rho_empty (a : ℕ) {n : ℕ} (hn : n = 0) :
    wordQ (Evaluate.Fp32.sqrtRn (Evaluate.Fp32.f32DivV2 a (Words.ofInt n))) = none := by
  subst hn
  rw [show Words.ofInt ((0 : ℕ) : ℤ) = 0 by decide]
  unfold Evaluate.Fp32.f32DivV2 Evaluate.Fp32.divWith
  rw [show Evaluate.Fp32.decode 0 = .fin false 0 from rfl]
  cases Evaluate.Fp32.decode a with
  | nan => decide +kernel
  | inf s => cases s <;> decide +kernel
  | fin s m =>
    by_cases hm : m = 0
    · subst hm; cases s <;> decide +kernel
    · show wordQ (Evaluate.Fp32.sqrtRn (if (0 : ℕ) = 0 then
          (if m = 0 then Evaluate.Fp32.F32_NAN_PTX else Evaluate.Fp32.infWord (s ^^ false))
        else Evaluate.Fp32.roundRn (s ^^ false) (m <<< 149) 0)) = none
      rw [if_pos rfl, if_neg hm]
      cases s <;> decide +kernel

/-- **`s5_stats` refuse an empty row**: its ρ is `√(part₀ / 0)`, which is not finite, and the stats are a value only where
ρ is (`statsRhoIsFinite`). -/
theorem s5Stats_pos {x : Array ℕ} {s ρw αw βw : ℕ} (h : Words.s5Stats x = .ok (s, ρw, αw, βw)) : 0 < x.size := by
  have hρ := statsRhoIsFinite x s ρw αw βw h
  by_contra hx0
  have hx : x.size = 0 := by omega
  unfold Words.s5Stats at h
  obtain ⟨p1, -, h⟩ := bind_ok h
  obtain ⟨p2, -, h⟩ := bind_ok h
  obtain ⟨t, -, h⟩ := bind_ok h
  obtain ⟨b, -, h⟩ := bind_ok h
  have h := pure_ok h
  simp only [Prod.mk.injEq] at h
  obtain ⟨-, hρw, -, -⟩ := h
  subst hρw
  revert hρ
  try dsimp only
  rw [rho_empty _ (n := (x.size + 7) / 8) (by rw [hx])]
  decide

/-- A chain that returns `.ok` multiplies rows of one length in whole atoms. -/
theorem chain_size {a b : Array ℕ} {c : ℕ} (h : Words.chain a b = .ok c) : a.size = b.size ∧ a.size % 32 = 0 := by
  unfold Words.chain at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := bind_ok h
    exact (throw_ok h1).elim
  rename_i hg
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hg
  exact hg

/-- A chain's word is finite: each promotion's word is `addWord?`'s, which is finite, or `+0`. -/
theorem chainWordP?_wordQ {p : Pipeline} {G : ℕ} {atoms : List AtomIn} {c : ℕ} (h : chainWordP? p G atoms = some c) :
    (wordQ c).isSome := by
  unfold chainWordP? at h
  refine foldlM_inv (fun t => (wordQ t).isSome) _ (fun t g s _ hs => ?_) _ 0 c
    (by simp only [CoreOps.wordQ_zero, Option.isSome_some]) h
  obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 hs
  unfold promoteWord? at hw
  obtain ⟨s', hs', rfl⟩ := Option.map_eq_some_iff.1 hw
  split_ifs
  · simp only [CoreOps.wordQ_zero, Option.isSome_some]
  · unfold addWord? at hs'
    split at hs'
    · obtain ⟨-, v, hv⟩ := PeelExactProofs.finite?_eq_some hs'
      obtain rfl := (PeelExactProofs.finite?_eq_some hs').1
      simp only [hv, Option.isSome_some]
    · cases hs'

/-! ## The fixed operand's noise -/

theorem basis_ok {sb : ByteArray} {side k : ℕ} {f : Array (Array ℕ)} (h : Words.basis sb side k = .ok f)
    (hs : side < 256) : f.size = k ∧ ∀ l < k, (f[l]!).size = 32 := by
  unfold Words.basis at h
  dsimp only at h
  obtain ⟨hsz, hall⟩ := mapM_ok h
  refine ⟨by simpa using hsz, fun l hl => ?_⟩
  have h1 := hall l (by simpa using hl)
  rw [range_get! hl] at h1
  exact (lineIsTheSpec sb side 1 l Words.NORM_F 16 _ hs (by norm_num) (Or.inr ⟨rfl, rfl⟩) h1).1

theorem column_size (b : Array (Array ℕ)) (q : ℕ) : (Words.column b q).size = b.size := by
  simp only [Words.column, Array.size_map]

theorem column_get {b : Array (Array ℕ)} {q l : ℕ} (hl : l < b.size) : (Words.column b q)[l]! = (b[l]!)[q]! := by
  unfold Words.column
  exact amap_get! hl

theorem cols_get (b : Array (Array ℕ)) {q : ℕ} (hq : q < 32) :
    ((Array.range R).map (Words.column b))[q]! = Words.column b q := by
  rw [amap_get! (by simpa [R] using hq), range_get! (by simpa [R] using hq)]

/-- **The fixed operand's noise**: where `fixedNoise` is a value, `F_A` and `F_B` have `k` rows of 32 codes, 32 columns
of `k` codes each, and the Gram word `(p, q)` is the chain of `F_B`'s column `p` and `F_A`'s column `q`. -/
theorem fixedNoise_ok {salt rootB : ByteArray} {k : ℕ} {wn : Words.FixedNoise}
    (h : Words.fixedNoise salt rootB k = .ok wn) :
    (∀ l < k, (wn.fa[l]!).size = 32 ∧ (wn.fb[l]!).size = 32) ∧ wn.faCols.size = 32 ∧ wn.fbCols.size = 32 ∧
      (∀ q < 32, (wn.faCols[q]!).size = k ∧ (wn.fbCols[q]!).size = k ∧
        ∀ l < k, (wn.faCols[q]!)[l]! = (wn.fa[l]!)[q]! ∧ (wn.fbCols[q]!)[l]! = (wn.fb[l]!)[q]!) ∧
      ∀ p < 32, ∀ q < 32, Words.chain wn.fbCols[p]! wn.faCols[q]! = .ok ((wn.gram[p]!)[q]!) := by
  unfold Words.fixedNoise at h
  obtain ⟨fa, hfa, h⟩ := bind_ok h
  obtain ⟨fb, hfb, h⟩ := bind_ok h
  obtain ⟨gram, hgram, h⟩ := bind_ok h
  obtain rfl := pure_ok h
  obtain ⟨hfaS, hfaR⟩ := basis_ok hfa (by norm_num)
  obtain ⟨hfbS, hfbR⟩ := basis_ok hfb (by norm_num)
  dsimp only
  have hR : ∀ b : Array (Array ℕ), ((Array.range R).map (Words.column b)).size = 32 := fun b => by
    simp only [Array.size_map, Array.size_range, R]
  obtain ⟨hgS, hgA⟩ := mapM_ok hgram
  refine ⟨fun l hl => ⟨hfaR l hl, hfbR l hl⟩, hR fa, hR fb, fun q hq => ?_, fun p hp q hq => ?_⟩
  · rw [cols_get fa hq, cols_get fb hq, column_size, column_size, hfaS, hfbS]
    exact ⟨rfl, rfl, fun l hl => ⟨column_get (by omega), column_get (by omega)⟩⟩
  · have h1 := hgA p (by rw [hR fb]; exact hp)
    obtain ⟨-, h2⟩ := mapM_ok h1
    exact h2 q (by rw [hR fa]; exact hq)

/-! ## A side's forming -/

/-- After `j` rows, `formV1`'s output holds each row's codes and `E′` as `s5_stats` and the forming's two maps give them. -/
def FormRow (rows eRows f : Array (Array ℕ)) (j : ℕ) (o : Words.Formed) : Prop :=
  o.codes.size = j ∧ o.ep.size = j ∧ ∀ y < j, ∃ s ρw αw βw : ℕ, Words.s5Stats rows[y]! = .ok (s, ρw, αw, βw) ∧
    (eRows[y]!).mapM (fun c => do Words.toFp8 (Evaluate.Fp32.f32MulV2 βw (← Words.ofFp8 c))) = .ok o.ep[y]! ∧
    (Array.range rows[y]!.size).mapM (fun t => do
      Words.toFp8 (← Words.finite (Words.atom (Evaluate.Fp32.f32MulV2 αw rows[y]![t]!) (o.ep[y]!).toList
        (f[t]!).toList))) = .ok o.codes[y]!

/-- **A side's forming**: where `formV1` is a value on rows and noise lines of one count, its codes and `E′` have a row
per row, each formed by `s5_stats` and the forming's two maps. -/
theorem formV1_ok {rows eRows f : Array (Array ℕ)} {out : Words.Formed} (h : Words.formV1 rows eRows f = .ok out)
    (hre : rows.size = eRows.size) : FormRow rows eRows f rows.size out := by
  unfold Words.formV1 at h
  obtain ⟨r, hfor, h⟩ := bind_ok h
  obtain rfl := pure_ok h
  rw [← Array.forIn_toList] at hfor
  have := StatsSim.forIn_inv (FormRow rows eRows f) _ 0
    (by exact ⟨rfl, rfl, fun y hy => absurd hy (Nat.not_lt_zero _)⟩) (fun j hj o r ho hr => by
      simp only [Nat.zero_add] at ho ⊢
      have hj' : j < min rows.size eRows.size := by simpa using hj
      have hjr : j < rows.size := by omega
      have hje : j < eRows.size := by omega
      rw [zip_toList_get hj hjr hje] at hr
      dsimp only at hr
      obtain ⟨⟨s0, rho, alpha, beta⟩, hs, hr⟩ := bind_ok hr
      dsimp only at hr
      obtain ⟨ep, hep, hr⟩ := bind_ok hr
      obtain ⟨codes, hcodes, hr⟩ := bind_ok hr
      obtain rfl := pure_ok hr
      refine ⟨_, rfl, ?_⟩
      obtain ⟨hc, he, hall⟩ := ho
      refine ⟨by simp only [Array.size_push, hc], by simp only [Array.size_push, he], fun y hy => ?_⟩
      by_cases hyj : y < j
      · obtain ⟨s', ρ', α', β', h1, h2, h3⟩ := hall y hyj
        refine ⟨s', ρ', α', β', h1, ?_, ?_⟩
        · dsimp only
          rw [push_get!_lt (show y < o.ep.size by omega)]
          exact h2
        · dsimp only
          rw [push_get!_lt (show y < o.ep.size by omega), push_get!_lt (show y < o.codes.size by omega)]
          exact h3
      · obtain rfl : y = j := by omega
        refine ⟨s0, rho, alpha, beta, hs, ?_, ?_⟩
        · dsimp only
          rw [push_get!_eq he]
          exact hep
        · dsimp only
          rw [push_get!_eq he, push_get!_eq hc]
          exact hcodes) hfor
  simpa only [Nat.zero_add, Array.length_toList, Array.size_zip, hre, Nat.min_self] using this

end SemSim

end Pouw.PearlPlus.Bridge
