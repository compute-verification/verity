module

public import Proofs.Pouw.PearlPlus.Factors

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
* `wordsAreTheSemantics`: with the peel factor rows proved (`peelARow`, `peelBRow`, `Factors.lean`).
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

namespace SemSim

open Pouw.PearlC.RowCircuit Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactP WordsSim

/-! ## Lists, arrays and folds -/

section Lists

variable {α β : Type}

theorem ofFn_mapM {f : α → Option β} : ∀ {n : ℕ} {a : Fin n → α} {b : Fin n → β}, (∀ p, f (a p) = some (b p)) →
    (List.ofFn a).mapM f = some (List.ofFn b)
  | 0, _, _, _ => by simp only [List.ofFn_zero, List.mapM_nil]; rfl
  | n + 1, a, b, h => by
    have ih := ofFn_mapM (f := f) (a := fun p => a p.succ) (b := fun p => b p.succ) (fun p => h p.succ)
    rw [List.ofFn_succ (f := a), List.ofFn_succ (f := b), List.mapM_cons, h 0, ih]
    rfl

theorem ofFn_mapM_id {n : ℕ} {a : Fin n → Option β} {b : Fin n → β} (h : ∀ p, a p = some (b p)) :
    (List.ofFn a).mapM id = some (List.ofFn b) :=
  ofFn_mapM (f := id) (a := a) (b := b) h

theorem foldlM_inv (P : β → Prop) (f : β → α → Option β) (hf : ∀ b a b', P b → f b a = some b' → P b') :
    ∀ (l : List α) (b b' : β), P b → l.foldlM f b = some b' → P b'
  | [], b, b', hb, h => by
    rw [List.foldlM_nil] at h
    obtain rfl := Option.some.inj h
    exact hb
  | a :: l, b, b', hb, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.1 h
    exact foldlM_inv P f hf l s b' (hf b a s hb hs) h

theorem push_get!_lt [Inhabited α] {a : Array α} {v : α} {y : ℕ} (h : y < a.size) : (a.push v)[y]! = a[y]! := by
  have h' : y < (a.push v).size := by simp only [Array.size_push]; omega
  rw [getElem!_pos (a.push v) y h', Array.getElem_push h', dite_eq_left h, getElem!_pos a y h]

theorem push_get!_eq [Inhabited α] {a : Array α} {v : α} {y : ℕ} (h : a.size = y) : (a.push v)[y]! = v := by
  have h' : y < (a.push v).size := by simp only [Array.size_push]; omega
  have hn : ¬ y < a.size := by omega
  rw [getElem!_pos (a.push v) y h', Array.getElem_push h', dite_eq_right hn]

theorem zip_toList_get [Inhabited α] [Inhabited β] {a : Array α} {b : Array β} {j : ℕ}
    (h : j < (a.zip b).toList.length) (ha : j < a.size) (hb : j < b.size) : (a.zip b).toList[j] = (a[j]!, b[j]!) := by
  rw [Array.getElem_toList, Array.getElem_zip, getElem!_pos a j ha, getElem!_pos b j hb]

theorem lmap_get! [Inhabited α] [Inhabited β] {l : List α} {g : α → β} {x : ℕ} (h : x < l.length) :
    (l.map g)[x]! = g l[x]! := by
  rw [getElem!_pos (l.map g) x (by simpa using h), List.getElem_map, getElem!_pos l x h]

theorem zipL_get! [Inhabited α] [Inhabited β] {l : List α} {g : α → β} {x : ℕ} (h : x < l.length) :
    (l.zip (l.map g)).toArray[x]! = (l[x]!, g l[x]!) := by
  have h' : x < (l.zip (l.map g)).length := by simpa using h
  rw [List.getElem!_toArray, getElem!_pos (l.zip (l.map g)) x h', List.getElem_zip, List.getElem_map,
    getElem!_pos l x h]

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
          (if m = 0 then Evaluate.Scalar.F32_NAN_PTX else Evaluate.Fp32.infWord (s ^^ false))
        else Evaluate.Fp32.roundRn (s ^^ false) (m <<< 149) 0)) = none
      rw [ite_eq_left rfl, ite_eq_right hm]
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
  rw [rho_empty _ (n := (x.size + 7) / 8) (by omega)]
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
  have hqR : q < R := by simpa [R] using hq
  have hqS : q < (Array.range R).size := by rw [Array.size_range]; exact hqR
  rw [amap_get! hqS, range_get! hqR]

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
      Words.toFp8 (← Words.finite (Words.atom (Evaluate.Fp32.f32MulV2 αw (rows[y]!)[t]!) (o.ep[y]!).toList
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

/-! ## A tile entry -/

open Matrix in
/-- **U from its parts**: where `C̃`, the chains of `A′·F_B`, `B̃·F_A` and `G`, the factor rows and the peel are values,
`peelExactP` is the peel's word. -/
theorem peelExactP_of {p : Pipeline} {G m n k : ℕ} {Aq : Matrix (Fin m) (Fin k) Code} {Bq : Matrix (Fin n) (Fin k) Code}
    {FA FB : Matrix (Fin k) (Fin 32) Code} {beA : Matrix (Fin m) (Fin 32) ℕ} {beB : Matrix (Fin n) (Fin 32) ℕ}
    {i : Fin m} {j : Fin n} {c u : ℕ} {af bf : Fin 32 → ℕ} {g : Fin 32 → Fin 32 → ℕ} {pa pb : List ℕ}
    (hc : chainWordP? p G (rowAtoms (Aq i) (Bq j)) = some c)
    (haf : ∀ q, chainWordP? p G (rowAtoms (Aq i) (FBᵀ q)) = some (af q))
    (hbf : ∀ q, chainWordP? p G (rowAtoms (Bq j) (FAᵀ q)) = some (bf q))
    (hg : ∀ q q', chainWordP? p G (rowAtoms (FBᵀ q) (FAᵀ q')) = some (g q q'))
    (hpa : paRow? (List.ofFn (beA i)) (List.ofFn af) = some pa)
    (hpb : pbRow? (List.ofFn (beB j)) (List.ofFn bf) (List.ofFn fun q => List.ofFn (g q)) = some pb)
    (hu : uWord? c pa pb = some u) : peelExactP p G Aq Bq FA FB beA beB i j = some u := by
  have h1 := ofFn_mapM_id haf
  have h2 := ofFn_mapM_id hbf
  have h3 := ofFn_mapM_id (b := fun q => List.ofFn (g q)) fun q => ofFn_mapM_id (b := g q) (hg q)
  simp only [peelExactP, hc, h1, h2, h3, hpa, hpb, hu, Option.bind_eq_bind, Option.bind_some]

open Matrix in
/-- **A tile entry's words**: over a noise `z` whose lines and bases are the check's (`hEA`, `hEB`, `hFA`, `hFB`), at a
tile whose rows' words are below `2^32`, where the check's fixed noise, operands and checked words are values, entry
`(x, y)` on rows that are `Row8OK` has `C̃` the H100 chain of `pc8Sem`'s operands and `U` their `peelExactP` at the H100,
given the peel factor rows. -/
theorem tileWords (hA : PeelARow) (hB : PeelBRow) {m n k u : ℕ} {act wt : Codes} {z : Noise m n k Params.pi.r}
    {salt rootB : ByteArray} {wn : Words.FixedNoise} {rows cols : List ℕ} {ops : Words.Ops} {ch : Words.Checked}
    (hkK : k ≤ 2 ^ 16) (hact : ∀ i ∈ rows, ∀ l < k, act i l < 2 ^ 32) (hwt : ∀ j ∈ cols, ∀ l < k, wt j l < 2 ^ 32)
    (hFA : ∀ l (hl : l < k) (t : Fin Params.pi.r), (wn.fa[l]!)[t.val]! = (z.FA ⟨l, hl⟩ t).val)
    (hFB : ∀ l (hl : l < k) (t : Fin Params.pi.r), (wn.fb[l]!)[t.val]! = (z.FB ⟨l, hl⟩ t).val)
    (hEA : ∀ i (hi : i < m) (e : Array ℕ),
      Words.sampleLine (Words.lineKey (Words.rowSeed salt rootB u i (rowBytesOf act k i))) 0 0 0 Words.NORM_E = .ok e →
        e.size = 32 ∧ ∀ t : Fin Params.pi.r, e[t.val]! = (z.EA ⟨i, hi⟩ t).val)
    (hEB : ∀ j (hj : j < n) (e : Array ℕ), Words.sampleLine (Words.lineKey wn.seedB) 1 0 j Words.NORM_E = .ok e →
        e.size = 32 ∧ ∀ t : Fin Params.pi.r, e[t.val]! = (z.EB ⟨j, hj⟩ t).val)
    (hwn : Words.fixedNoise salt rootB k = .ok wn)
    (henc : Words.encode salt rootB wn u k rows (rows.map (rowBytesOf act k)) cols (cols.map (rowBytesOf wt k)) = .ok ops)
    (hch : Words.checked ops = .ok ch)
    {x y : ℕ} (hx : x < rows.length) (hi : rows[x]! < m) (hRA : WholeCircuit.Row8OK coreOps k (act rows[x]!))
    (hy : y < cols.length) (hj : cols[y]! < n) (hRB : WholeCircuit.Row8OK coreOps k (wt cols[y]!)) :
    chainWordP? hopperE4m3K32 4 (rowAtoms (WholeCircuit.quant8A coreOps hopperE4m3K32 z act ⟨rows[x]!, hi⟩)
        (WholeCircuit.quant8B coreOps hopperE4m3K32 z wt ⟨cols[y]!, hj⟩)) = some ((ch.c[x]!)[y]!) ∧
      peelExactP hopperE4m3K32 4 (WholeCircuit.quant8A coreOps hopperE4m3K32 z act)
        (WholeCircuit.quant8B coreOps hopperE4m3K32 z wt) (WholeCircuit.basis8 z.FA) (WholeCircuit.basis8 z.FB)
        (fun i => WholeCircuit.be8 (WholeCircuit.betaQ8 coreOps k (act i)) (z.EA i))
        (fun j => WholeCircuit.be8 (WholeCircuit.betaQ8 coreOps k (wt j)) (z.EB j)) ⟨rows[x]!, hi⟩ ⟨cols[y]!, hj⟩ =
        some ((ch.u[x]!)[y]!) := by
  have hadd := fp32AddIsTheSpec
  obtain ⟨hfR, -, hfbC, hcolF, hgram⟩ := fixedNoise_ok hwn
  unfold Words.encode at henc
  obtain ⟨ea, hea, henc⟩ := bind_ok henc
  obtain ⟨va, hva, henc⟩ := bind_ok henc
  obtain ⟨a, ha, henc⟩ := bind_ok henc
  try dsimp only at henc
  obtain ⟨eb, heb, henc⟩ := bind_ok henc
  obtain ⟨vb, hvb, henc⟩ := bind_ok henc
  obtain ⟨b, hb, henc⟩ := bind_ok henc
  obtain ⟨pas, hpas, henc⟩ := bind_ok henc
  obtain ⟨pbs, hpbs, henc⟩ := bind_ok henc
  obtain rfl := pure_ok henc
  unfold Words.checked at hch
  try dsimp only at hch
  obtain ⟨c, hc, hch⟩ := bind_ok hch
  obtain ⟨uu, hu, hch⟩ := bind_ok hch
  obtain rfl := pure_ok hch
  try dsimp only
  -- the operands' rows, noise lines and forming
  obtain ⟨hvaS, hvaA⟩ := mapM_ok hva
  obtain ⟨heaS, heaA⟩ := mapM_ok hea
  obtain ⟨hvbS, hvbA⟩ := mapM_ok hvb
  obtain ⟨hebS, hebA⟩ := mapM_ok heb
  simp only [List.size_toArray, List.length_zip, List.length_map, Nat.min_self] at hvaS heaS hvbS hebS
  obtain ⟨haC, haE, haR⟩ := formV1_ok ha (by rw [hvaS, heaS])
  obtain ⟨hbC, hbE, hbR⟩ := formV1_ok hb (by rw [hvbS, hebS])
  obtain ⟨s, ρw, αw, βw, hs, hep, hcodes⟩ := haR x (by omega)
  obtain ⟨s', ρw', αw', βw', hs', hep', hcodes'⟩ := hbR y (by omega)
  have hvx := hvaA x (by simpa using hx)
  rw [List.getElem!_toArray, lmap_get! hx] at hvx
  obtain ⟨hvxS, hvxW⟩ := parseRow_ok (hact _ (get!_mem hx)) hvx
  have hvy := hvbA y (by simpa using hy)
  rw [List.getElem!_toArray, lmap_get! hy] at hvy
  obtain ⟨hvyS, hvyW⟩ := parseRow_ok (hwt _ (get!_mem hy)) hvy
  have hex := heaA x (by simpa using hx)
  rw [zipL_get! hx] at hex
  try dsimp only at hex
  obtain ⟨heS, heA⟩ := hEA rows[x]! hi ea[x]! hex
  have hey := hebA y (by simpa using hy)
  rw [List.getElem!_toArray] at hey
  obtain ⟨heS', heA'⟩ := hEB cols[y]! hj eb[y]! hey
  -- the depth: the chain multiplies whole atoms, and `s5_stats` refuse an empty row
  obtain ⟨hcS, hcA⟩ := mapM_ok hc
  obtain ⟨-, hcxA⟩ := mapM_ok (hcA x (by omega))
  have hcxy := hcxA y (by omega)
  have hkx : (a.codes[x]!).size = k := by rw [(mapM_ok hcodes).1, Array.size_range, hvxS]
  have hky : (b.codes[y]!).size = k := by rw [(mapM_ok hcodes').1, Array.size_range, hvyS]
  have hk32 : k % 32 = 0 := by have := (chain_size hcxy).2; omega
  have hk0 : 0 < k := by have := s5Stats_pos hs; omega
  have hk8 : 8 ∣ k := Nat.dvd_of_mod_eq_zero (by omega)
  have hk27 : k ≤ 2 ^ 27 := le_trans hkK (by norm_num)
  obtain ⟨hepS, hepA, -, hcodA⟩ := formRow_quant8 (E := z.EA ⟨rows[x]!, hi⟩) (F := z.FA) hk0 hk8 hk27 hvxS hvxW
    (hact _ (get!_mem hx)) hRA heS heA (fun t => (hfR t.val t.isLt).1) (fun t q => hFA t.val t.isLt q) hs hep hcodes
  obtain ⟨hepS', hepA', -, hcodB⟩ := formRow_quant8 (E := z.EB ⟨cols[y]!, hj⟩) (F := z.FB) hk0 hk8 hk27 hvyS hvyW
    (hwt _ (get!_mem hy)) hRB heS' heA' (fun t => (hfR t.val t.isLt).2) (fun t q => hFB t.val t.isLt q) hs' hep'
    hcodes'
  -- the chains
  have hC : chainWordP? hopperE4m3K32 4 (rowAtoms (WholeCircuit.quant8A coreOps hopperE4m3K32 z act ⟨rows[x]!, hi⟩)
      (WholeCircuit.quant8B coreOps hopperE4m3K32 z wt ⟨cols[y]!, hj⟩)) = some ((c[x]!)[y]!) :=
    chain_ok hadd hkx hky (fun t ht => hcodA ⟨t, ht⟩) (fun t ht => hcodB ⟨t, ht⟩) hcxy
  have hGq : ∀ p q : Fin 32, chainWordP? hopperE4m3K32 4 (rowAtoms ((WholeCircuit.basis8 z.FB)ᵀ p)
      ((WholeCircuit.basis8 z.FA)ᵀ q)) = some ((wn.gram[p.val]!)[q.val]!) := fun p q => by
    obtain ⟨-, hp2, hp3⟩ := hcolF p.val p.isLt
    obtain ⟨hq1, -, hq3⟩ := hcolF q.val q.isLt
    exact chain_ok hadd hp2 hq1 (fun t ht => (hp3 t ht).2.trans (hFB t ht p))
      (fun t ht => (hq3 t ht).1.trans (hFA t ht q)) (hgram p.val p.isLt q.val q.isLt)
  -- the peel factor rows
  obtain ⟨-, hpaxS, af, hafC, hpaR⟩ := hA wn a pas x
    (lineCodes (WholeCircuit.betaQ8 coreOps k (act rows[x]!)) (z.EA ⟨rows[x]!, hi⟩)) hpas (haC.trans haE.symm)
    (by omega) hfbC hepS hepA (WholeCircuit.lineCodes_notNaN _ _)
  obtain ⟨hpbS, hpbyS, bf, hbfC, hpbR⟩ := hB wn b pbs y
    (lineCodes (WholeCircuit.betaQ8 coreOps k (wt cols[y]!)) (z.EB ⟨cols[y]!, hj⟩)) (fun p q => (wn.gram[p.val]!)[q.val]!)
    hpbs (hbC.trans hbE.symm) (by omega) hepS' hepA' (WholeCircuit.lineCodes_notNaN _ _)
    (fun p q => ⟨rfl, WholeCircuit.chainWordP?_lt (hGq p q), chainWordP?_wordQ (hGq p q)⟩)
  have hAFq : ∀ q : Fin 32, chainWordP? hopperE4m3K32 4 (rowAtoms (WholeCircuit.quant8A coreOps hopperE4m3K32 z act
      ⟨rows[x]!, hi⟩) ((WholeCircuit.basis8 z.FB)ᵀ q)) = some (af q) := fun q => by
    obtain ⟨-, hq2, hq3⟩ := hcolF q.val q.isLt
    exact chain_ok hadd hkx hq2 (fun t ht => hcodA ⟨t, ht⟩) (fun t ht => (hq3 t ht).2.trans (hFB t ht q)) (hafC q)
  have hBFq : ∀ q : Fin 32, chainWordP? hopperE4m3K32 4 (rowAtoms (WholeCircuit.quant8B coreOps hopperE4m3K32 z wt
      ⟨cols[y]!, hj⟩) ((WholeCircuit.basis8 z.FA)ᵀ q)) = some (bf q) := fun q => by
    obtain ⟨hq1, -, hq3⟩ := hcolF q.val q.isLt
    exact chain_ok hadd hky hq1 (fun t ht => hcodB ⟨t, ht⟩) (fun t ht => (hq3 t ht).1.trans (hFA t ht q)) (hbfC q)
  -- the peel
  obtain ⟨-, huA⟩ := mapM_ok hu
  have hux := huA x (by rw [Array.size_range]; omega)
  rw [range_get! (show x < c.size by omega)] at hux
  obtain ⟨-, huxA⟩ := mapM_ok hux
  have huxy := huxA y (by rw [Array.size_range]; omega)
  rw [range_get! (show y < pbs.size by omega)] at huxy
  have hU := peel_uWord (WholeCircuit.chainWordP?_lt hC) (by omega) (by omega) huxy
  exact ⟨hC, peelExactP_of hC hAFq hBFq hGq hpaR hpbR hU⟩

end SemSim

open SemSim WordsSim in
/-- **(b) A tile's words are the semantics'**, given the peel factor rows: `tileWords` at -h3's noise, whose lines and
bases are the check's (`seedsAreTheSpec`, `lineIsTheSpec`, `fixedNoiseIsTheSpec`), on rows that pass the rules
(`row8OK_of_floor`), read through the H100 record's ticket and useful words. -/
theorem wordsAreTheSemantics_of (hA : PeelARow) (hB : PeelBRow) : WordsAreTheSemantics := by
  intro Q R S saltOf rootOf ρ H s U u act salt rootB wn rows cols ops ch hsalt hroot hk hrows hcols hact hwt hW hwn
    henc hch x hx hR y hy
  have him := hrows _ (get!_mem hx)
  have hjn := hcols _ (get!_mem hy)
  have hWy := hW _ (get!_mem hy)
  obtain ⟨hseed, -, -, hF⟩ := fixedNoiseIsTheSpec salt rootB _ wn hwn
  have hsd : bytesNat wn.seedB = NoiseSha512.seedB (saltOf s) (rootOf (U.layout.fixedId u)) := by
    rw [hseed, hsalt, hroot]
  obtain ⟨hC, hP⟩ := tileWords (z := (semH3 saltOf rootOf : PearlCSem Q R S).noise H s U u act)
    (wt := U.fixed (U.layout.fixedId u)) hA hB hk hact hwt
    (fun l hl t => by rw [(hF l hl t).1, hsd]; rfl) (fun l hl t => by rw [(hF l hl t).2, hsd]; rfl)
    (fun i hi e he => by
      obtain ⟨h1, h2⟩ := lineIsTheSpec _ 0 0 0 Words.NORM_E 256 e (by norm_num) (by norm_num) (Or.inl ⟨rfl, rfl⟩) he
      refine ⟨h1, fun t => ?_⟩
      rw [h2 t, (seedsAreTheSpec).2.2, hsalt, hroot, bytesNat_rowBytesOf]
      rfl)
    (fun j hj e he => by
      obtain ⟨h1, h2⟩ := lineIsTheSpec _ 1 0 j Words.NORM_E 256 e (by norm_num) (by norm_num) (Or.inl ⟨rfl, rfl⟩) he
      refine ⟨h1, fun t => ?_⟩
      rw [h2 t, hsd]
      rfl)
    hwn henc hch hx him
    (WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
      (semH3 saltOf rootOf : PearlCSem Q R S).noise him hR.1 (hR.2 him).2)
    hy hjn
    (WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
      (semH3 saltOf rootOf : PearlCSem Q R S).noise hjn hWy.1 (hWy.2 hjn).2)
  refine ⟨?_, ?_⟩
  · show _ = atNat _ rows[x]! cols[y]!
    unfold atNat
    rw [dite_eq_left (And.intro him hjn)]
    have hv := TileCircuit.chainWordP_value hopperE4m3K32 4 _ hC
    rw [Accum.chainP_hopper_four] at hv
    exact hv.symm
  · show _ = atNat _ rows[x]! cols[y]!
    unfold atNat
    rw [dite_eq_left (And.intro him hjn)]
    rw [PeelExactP.peelExactP_hopper] at hP
    have hv := congrArg (fun o => Option.bind o wordQ) hP
    simp only [Option.bind_some] at hv
    exact hv.symm

/-- **(b) A tile's words are the semantics'.** -/
theorem wordsAreTheSemantics : WordsAreTheSemantics := wordsAreTheSemantics_of peelARow peelBRow

end Pouw.PearlPlus.Bridge
