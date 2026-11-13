module

public import Proofs.Pouw.PearlPlus.Peel

/-!
# Pearl+'s bridge, part (b), the peel factor rows

`Words.peelA` writes a formed side's `P_A` rows, each row's `bf16(E′)` words then the BF16 roundings of its 32 chains
against `F_B`'s columns; `Words.peelB` its `P_B` rows, for each column `q` the BF16 rounding of an FP32 FMA chain from
`+0` over `G`'s column less the row's chain against `F_A`'s column, then `−bf16(E′)`. `peelARow` and `peelBRow` say these
are `PeelExact.paRow?` and `pbRow?` on the same words: a code's `bf16(E′)` word is `codeBf16`
(`WholeCircuit.widen8_word`), each rounding is `rnBf16?` (`toBf16_ok`), the add is `addWord?` (`fp32AddIsTheSpec`), and an
FMA of finite words whose word is finite is `fmaWord?` (`fmaWord_of`: `CoreOps.coreOps_fma`, and where the spec's rounding
overflows `coreOps_fmaOvf`, which makes the check's word that overflow).
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

namespace FactorsSim

open WordsSim

section Monad

variable {α β : Type}

/-- An `Option` `List.mapM` is `some ys` where each element maps to its entry of `ys`. -/
theorem optMapM_of {f : α → Option β} : ∀ {l : List α} {ys : List β}, l.length = ys.length →
    (∀ i (h : i < l.length) (h' : i < ys.length), f l[i] = some ys[i]) → l.mapM f = some ys
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: l, b :: ys, h, hf => by
    have h0 : f a = some b := hf 0 (by simp) (by simp)
    have ht : l.mapM f = some ys :=
      optMapM_of (by simpa using h) fun i h1 h2 => hf (i + 1) (by simpa using h1) (by simpa using h2)
    rw [List.mapM_cons, h0, ht]
    rfl

theorem bind_some_of {x : Option α} {f : α → Option β} {a : α} {b : β} (hx : x = some a) (hf : f a = some b) :
    (x >>= f) = some b := by
  rw [hx]
  exact hf

theorem zip_getElem! [Inhabited α] [Inhabited β] {a : Array α} {b : Array β} {x : ℕ} (ha : x < a.size)
    (hb : x < b.size) : (a.zip b)[x]! = (a[x]!, b[x]!) := by
  rw [getElem!_pos (a.zip b) x (by rw [Array.size_zip]; omega), Array.getElem_zip, getElem!_pos a x ha,
    getElem!_pos b x hb]

end Monad

theorem range_getElem! {n i : ℕ} (h : i < n) : (Array.range n)[i]! = i := by
  rw [getElem!_pos (Array.range n) i (by rw [Array.size_range]; exact h), Array.getElem_range]

theorem toList_eq_ofFn {a : Array ℕ} {g : Fin 32 → ℕ} (ha : a.size = 32) (h : ∀ q : Fin 32, a[q.val]! = g q) :
    a.toList = List.ofFn g := by
  apply List.ext_getElem
  · simp [ha]
  · intro l h1 _
    have hl : l < a.size := by simpa using h1
    rw [Array.getElem_toList, List.getElem_ofFn, ← getElem!_pos a l hl]
    exact h ⟨l, by omega⟩

theorem getD_ofFn {g : Fin 32 → ℕ} {i : ℕ} (hi : i < 32) : (List.ofFn g).getD i 0 = g ⟨i, hi⟩ := by
  rw [List.getD_eq_getElem _ _ (by rw [List.length_ofFn]; exact hi), List.getElem_ofFn]

/-- Column `q` of `G` by rows is `G`'s column `q`. -/
theorem col_ofFn {G : Fin 32 → Fin 32 → ℕ} {q : ℕ} (hq : q < 32) :
    PeelExact.col (List.ofFn fun p => List.ofFn (G p)) q = List.ofFn fun p => G p ⟨q, hq⟩ := by
  unfold PeelExact.col
  rw [List.map_ofFn]
  congr 1
  funext p
  show (List.ofFn (G p)).getD q 0 = G p ⟨q, hq⟩
  exact getD_ofFn hq

/-! ### The spec's rows from their entries -/

theorem paRow_of {be af fused : List ℕ} (hlen : af.length = fused.length)
    (h : ∀ i (h1 : i < af.length) (h2 : i < fused.length), PeelExact.rnBf16? af[i] = some fused[i]) :
    PeelExact.paRow? be af = some (be ++ fused) :=
  (congrArg (Option.map (be ++ ·)) (optMapM_of (f := PeelExact.rnBf16?) hlen h)).trans rfl

theorem pbRow_of {be bf : List ℕ} {g : List (List ℕ)} {mid : List ℕ} (hlen : mid.length = Params.pi.r)
    (hq : ∀ q (hq : q < Params.pi.r), ∃ acc s, PeelExact.accChain? be (PeelExact.col g q) = some acc ∧
      PeelExact.addWord? acc (PeelExact.negWord (bf.getD q 0)) = some s ∧
        PeelExact.rnBf16? s = some (mid[q]'(by omega))) :
    PeelExact.pbRow? be bf g = some (mid ++ be.map (· ^^^ 0x8000)) := by
  unfold PeelExact.pbRow?
  refine bind_some_of (optMapM_of (ys := mid) (by rw [List.length_range, hlen]) fun i h1 h2 => ?_) rfl
  have hi : i < Params.pi.r := by simpa using h1
  obtain ⟨acc, s, ha, hs, hr⟩ := hq i hi
  rw [List.getElem_range]
  exact bind_some_of ha (bind_some_of hs hr)

/-! ### `bf16(E′)` -/

/-- The check's `bf16(E′)` word of a code other than NaN is `codeBf16`. -/
theorem bf16OfCode_ok {c : Code} {o : ℕ} (hn : c.isNaN = false) (h : Words.bf16OfCode c.val = .ok o) :
    o = codeBf16 c := by
  unfold Words.bf16OfCode at h
  obtain ⟨w, hw, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := ofFp8_ok hw
  have hwid : Evaluate.Casts.e4m3ToF32 c.val = codeBf16 c <<< 16 := by
    have h1 := WholeCircuit.widen8_word CoreOps.pc8OpsAgree_coreOps hn
    rw [PeelSim.eval_e4m3ToF32, Nat.mod_eq_of_lt (PeelSim.e4m3ToF32_lt c)] at h1
    rw [show Evaluate.Casts.e4m3ToF32 = Verity.Prims.e4m3ToF32 from Properties.Evaluator.Casts.e4m3ToF32]
    exact h1
  obtain ⟨hr, -⟩ := toBf16_ok h
  rw [hwid, WholeCircuit.rnBf16_codeBf16 c hn] at hr
  exact (Option.some.inj hr).symm

/-- A row of 32 codes, none NaN, has the `bf16(E′)` words `codeBf16`. -/
theorem be_ok {ep be : Array ℕ} {E : Fin Params.pi.r → Code} (hs : ep.size = 32)
    (hE : ∀ q : Fin Params.pi.r, ep[q.val]! = (E q).val) (hn : ∀ q, (E q).isNaN = false)
    (h : ep.mapM Words.bf16OfCode = .ok be) : be.size = 32 ∧ ∀ q : Fin 32, be[q.val]! = codeBf16 (E q) := by
  obtain ⟨hbs, hbe⟩ := mapM_ok h
  refine ⟨hbs.trans hs, fun q => ?_⟩
  have h1 := hbe q.val (by rw [hs]; exact q.isLt)
  have h2 : ep[q.val]! = (E q).val := hE q
  rw [h2] at h1
  exact bf16OfCode_ok (hn q) h1

/-! ### The FMA -/

theorem roundRn_lt (s : Bool) (n d : ℕ) : Verity.Fp32.roundRn s n d < 2 ^ 32 := by
  unfold Verity.Fp32.roundRn
  dsimp only
  have key : ∀ w : ℕ, (if w ≥ 0x7F800000 then 0x7F800000 else w) < 2 ^ 32 := fun w => by split <;> omega
  exact Nat.or_lt_two_pow (by cases s <;> decide) (key _)

theorem roundSum_lt (s₁ : Bool) (n₁ : ℕ) (s₂ : Bool) (n₂ d : ℕ) : Verity.Fp32.roundSum s₁ n₁ s₂ n₂ d < 2 ^ 32 := by
  unfold Verity.Fp32.roundSum
  split_ifs <;> first | exact roundRn_lt _ _ _ | norm_num

theorem addRn_lt {nan d : ℕ} (hn : nan < 2 ^ 32) (x y : Verity.Fp32.Decoded) : x.addRn nan d y < 2 ^ 32 := by
  have hinf : ∀ s, Verity.Fp32.infWord s < 2 ^ 32 := fun s => by
    unfold Verity.Fp32.infWord
    exact Nat.or_lt_two_pow (by cases s <;> decide) (by norm_num)
  cases x <;> cases y <;> first
    | exact hn
    | exact hinf _
    | exact roundSum_lt _ _ _ _ _
    | (show (if _ = _ then _ else _) < 2 ^ 32; split <;> first | exact hinf _ | exact hn)

/-- The check's FMA writes a word below `2^32`. -/
theorem fmaV2_lt (a b c : ℕ) : Verity.Fp32.f32FmaV2 a b c < 2 ^ 32 := by
  unfold Verity.Fp32.f32FmaV2 Verity.Fp32.f32GpuNan
  split
  · decide
  · exact addRn_lt (by decide) _ _

/-- Where the spec's FMA rounding leaves the finite range, the rounded value is an overflow. -/
theorem rnWord_overflow {v : ℚ} {nz : Bool} (h : PeelExact.finite? (PeelExact.rnWord v nz) = none) :
    (f32Word v >>> 23) % 256 = 255 ∧ wordQ (f32Word v) = none := by
  have he : (PeelExact.rnWord v nz >>> 23) % 256 = 255 := by
    unfold PeelExact.finite? at h
    by_contra hne
    rw [ite_eq_right hne] at h
    cases h
  unfold PeelExact.rnWord at he
  by_cases h0 : v = 0
  · rw [ite_eq_left h0] at he
    cases nz <;> exact absurd he (by decide)
  · rw [ite_eq_right h0] at he
    refine ⟨he, ?_⟩
    cases hq : wordQ (f32Word v) with
    | none => rfl
    | some q => exact absurd he (RowCircuit.wordQ_exp_ne hq)

/-- **The check's FMA is the spec's**: on finite words below `2^32`, where `Evaluate.Fp32.f32FmaV2` is a finite word,
it is `fmaWord?`'s. -/
theorem fmaWord_of {a b c : ℕ} (ha : a < 2 ^ 32) (hb : b < 2 ^ 32) (hc : c < 2 ^ 32)
    (hxa : (wordQ a).isSome) (hxb : (wordQ b).isSome) (hxc : (wordQ c).isSome)
    (h : (Evaluate.Fp32.f32FmaV2 a b c >>> 23) % 256 ≠ 255) :
    PeelExact.fmaWord? a b c = some (Evaluate.Fp32.f32FmaV2 a b c) := by
  rw [show Evaluate.Fp32.f32FmaV2 = Verity.Fp32.f32FmaV2 from Properties.Evaluator.Fp32.f32FmaV2] at h ⊢
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hxa
  obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hxb
  obtain ⟨z, hz⟩ := Option.isSome_iff_exists.1 hxc
  have hlt := fmaV2_lt a b c
  cases hw : PeelExact.fmaWord? a b c with
  | some w =>
    have e : Verity.Fp32.f32FmaV2 a b c % 2 ^ 32 = w := CoreOps.coreOps_fma a b c w ha hb hc hw
    rw [Nat.mod_eq_of_lt hlt] at e
    rw [e]
  | none =>
    exfalso
    unfold PeelExact.fmaWord? at hw
    rw [hx, hy, hz] at hw
    dsimp only at hw
    obtain ⟨he, hn⟩ := rnWord_overflow hw
    have e : Verity.Fp32.f32FmaV2 a b c % 2 ^ 32 = f32Word (x * y + z) :=
      CoreOps.coreOps_fmaOvf a b c x y z ha hb hc hx hy hz hn
    rw [Nat.mod_eq_of_lt hlt] at e
    rw [e] at h
    exact h he

/-- **`P_B`'s FMA chain is `accChain?`'s**: the check's loop `acc ← RN(be[p]·G[p][q] + acc)` from `+0` over `p < 32`,
each word finite, is the spec's chain over the row's `bf16(E′)` words `eb` and column `q` of `G`, `gq`. -/
theorem accLoop_ok {be : Array ℕ} {gram : Array (Array ℕ)} {q : ℕ} {eb gq : Fin 32 → ℕ}
    (hbe : ∀ p (hp : p < 32), be[p]! = eb ⟨p, hp⟩)
    (heb : ∀ p, eb p <<< 16 < 2 ^ 32 ∧ (wordQ (eb p <<< 16)).isSome)
    (hg : ∀ p (hp : p < 32), (gram[p]!)[q]! = gq ⟨p, hp⟩ ∧ gq ⟨p, hp⟩ < 2 ^ 32 ∧ (wordQ (gq ⟨p, hp⟩)).isSome)
    {body : ℕ → ℕ → Except String (ForInStep ℕ)}
    (hbody : ∀ i r y, body i r = .ok y →
      ∃ v, Words.finite (Evaluate.Fp32.f32FmaV2 (be[i]! <<< 16) (gram[i]!)[q]! r) = .ok v ∧ y = .yield v)
    {acc : ℕ} (h : forIn [0:PouwCheck.PearlPlus.R] 0 body = .ok acc) :
    PeelExact.accChain? (List.ofFn eb) (List.ofFn gq) = some acc := by
  have eF : Evaluate.Fp32.f32FmaV2 = Verity.Fp32.f32FmaV2 := Properties.Evaluator.Fp32.f32FmaV2
  have hlen : ((List.ofFn eb).zip (List.ofFn gq)).length = 32 := by simp
  have hget : ∀ i (hi : i < 32), ((List.ofFn eb).zip (List.ofFn gq))[i]? = some (eb ⟨i, hi⟩, gq ⟨i, hi⟩) :=
    fun i hi => by
      rw [List.getElem?_eq_getElem (by rw [hlen]; exact hi), List.getElem_zip, List.getElem_ofFn, List.getElem_ofFn]
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have := StatsSim.forIn_inv (fun (i a : ℕ) => a < 2 ^ 32 ∧ (a >>> 23) % 256 ≠ 255 ∧
      (((List.ofFn eb).zip (List.ofFn gq)).take i).foldlM
        (fun acc x => PeelExact.fmaWord? (x.1 <<< 16) x.2 acc) 0 = some a)
    _ 0 ⟨by norm_num, by decide, rfl⟩ (fun i hi z r hz hr => by
      rw [List.getElem_range'] at hr
      simp only [Nat.zero_add, Nat.one_mul] at hz hr ⊢
      have hi32 : i < 32 := by simpa [Std.Legacy.Range.size, PouwCheck.PearlPlus.R] using hi
      obtain ⟨v, hv, rfl⟩ := hbody _ _ _ hr
      obtain ⟨rfl, hf⟩ := finite_ok hv
      obtain ⟨hz1, hz2, hz3⟩ := hz
      obtain ⟨hb1, hb2⟩ := heb ⟨i, hi32⟩
      obtain ⟨hg1, hg2, hg3⟩ := hg i hi32
      rw [hbe i hi32, hg1] at hf ⊢
      have hfma := fmaWord_of hb1 hg2 hz1 hb2 hg3 (Option.isSome_iff_exists.2 (wordQ_of hz2)) hf
      refine ⟨_, rfl, by rw [eF]; exact fmaV2_lt _ _ _, hf, ?_⟩
      rw [List.take_add_one, hget i hi32, Option.toList_some, List.foldlM_append, hz3]
      simp only [List.foldlM_cons, List.foldlM_nil, Option.bind_eq_bind, Option.bind_some, hfma,
        Option.pure_def]) h
  simp only [List.length_range', Std.Legacy.Range.size, PouwCheck.PearlPlus.R, Nat.zero_add] at this
  rw [List.take_of_length_le (le_of_eq hlen)] at this
  unfold PeelExact.accChain?
  exact this.2.2

end FactorsSim

open FactorsSim WordsSim in
/-- **`P_A`'s rows are `paRow?`'s** (`PeelARow`). -/
theorem peelARow : PeelARow := by
  intro wn side pas x E h hse hx hfb hep hE hn
  unfold Words.peelA at h
  obtain ⟨hps, hpa⟩ := mapM_ok h
  have hzs : (side.codes.zip side.ep).size = side.codes.size := by rw [Array.size_zip, ← hse, Nat.min_self]
  have hx' : x < side.ep.size := by rw [← hse]; exact hx
  have hrow := hpa x (by rw [hzs]; exact hx)
  rw [zip_getElem! hx hx'] at hrow
  dsimp only at hrow
  obtain ⟨be, hbe, hrow⟩ := bind_ok hrow
  obtain ⟨fused, hfu, hrow⟩ := bind_ok hrow
  have hpx := pure_ok hrow
  obtain ⟨hbs, hbv⟩ := be_ok hep hE hn hbe
  obtain ⟨hfs, hfa⟩ := mapM_ok hfu
  have hq : ∀ q : Fin 32, ∃ a, Words.chain side.codes[x]! wn.fbCols[q.val]! = .ok a ∧
      Words.toBf16 a = .ok fused[q.val]! := fun q => bind_ok (hfa q.val (by rw [hfb]; exact q.isLt))
  choose af haf hfz using hq
  refine ⟨by rw [hps, hzs], by rw [← hpx]; simp [hbs, hfs, hfb], af, haf, ?_⟩
  rw [← hpx, Array.toList_append, toList_eq_ofFn hbs hbv]
  exact paRow_of (by rw [List.length_ofFn, Array.length_toList, hfs, hfb]) fun i h1 _ => by
    have hi : i < 32 := by simpa using h1
    rw [List.getElem_ofFn, Array.getElem_toList, ← getElem!_pos fused i (by rw [hfs, hfb]; exact hi)]
    exact (toBf16_ok (hfz ⟨i, hi⟩)).1

open FactorsSim WordsSim in
/-- **`P_B`'s rows are `pbRow?`'s** (`PeelBRow`). -/
theorem peelBRow : PeelBRow := by
  intro wn side pbs j E G h hse hj hep hE hn hG
  unfold Words.peelB at h
  obtain ⟨hps, hpb⟩ := mapM_ok h
  have hzs : (side.codes.zip side.ep).size = side.codes.size := by rw [Array.size_zip, ← hse, Nat.min_self]
  have hj' : j < side.ep.size := by rw [← hse]; exact hj
  have hrow := hpb j (by rw [hzs]; exact hj)
  rw [zip_getElem! hj hj'] at hrow
  dsimp only at hrow
  obtain ⟨be, hbe, hrow⟩ := bind_ok hrow
  obtain ⟨mid, hmid, hrow⟩ := bind_ok hrow
  have hpj := pure_ok hrow
  obtain ⟨hbs, hbv⟩ := be_ok hep hE hn hbe
  obtain ⟨hms, hma⟩ := mapM_ok hmid
  have hq : ∀ q : Fin 32, ∃ b acc s, Words.chain side.codes[j]! wn.faCols[q.val]! = .ok b ∧
      PeelExact.accChain? (List.ofFn fun p : Fin 32 => codeBf16 (E p)) (List.ofFn fun p => G p q) = some acc ∧
      PeelExact.addWord? acc (PeelExact.negWord b) = some s ∧ PeelExact.rnBf16? s = some mid[q.val]! := by
    intro q
    have h1 := hma q.val (by rw [Array.size_range]; exact q.isLt)
    rw [range_getElem! (n := PouwCheck.PearlPlus.R) q.isLt] at h1
    obtain ⟨acc, hloop, h1⟩ := bind_ok h1
    obtain ⟨b, hb, h1⟩ := bind_ok h1
    obtain ⟨s, hs, h1⟩ := bind_ok h1
    obtain ⟨rfl, hsf⟩ := finite_ok hs
    refine ⟨b, acc, _, hb, accLoop_ok (eb := fun p => codeBf16 (E p)) (gq := fun p => G p q)
      (fun p hp => hbv ⟨p, hp⟩)
      (fun p => ⟨WholeCircuit.codeBf16_shift_lt _,
        Option.isSome_iff_exists.2 ⟨_, PeelExists.wordQ_codeBf16 _ (hn p)⟩⟩)
      (fun p hp => hG ⟨p, hp⟩ q)
      (fun i r y hy => by
        obtain ⟨v, hv, hy⟩ := bind_ok hy
        exact ⟨v, hv, (pure_ok hy).symm⟩) hloop,
      addWord_of fp32AddIsTheSpec hsf, (toBf16_ok h1).1⟩
  choose bf accs ss hbf hacc hadd hrn using hq
  refine ⟨by rw [hps, hzs], by rw [← hpj]; simp [hms, hbs, PouwCheck.PearlPlus.R], bf, hbf, ?_⟩
  have hmidL : mid.toList.length = Params.pi.r := Array.length_toList.trans (hms.trans Array.size_range)
  rw [← hpj, Array.toList_append, Array.toList_map, toList_eq_ofFn hbs hbv]
  exact pbRow_of hmidL fun q hq => by
    have hq' : q < 32 := hq
    refine ⟨accs ⟨q, hq'⟩, ss ⟨q, hq'⟩, ?_, ?_, ?_⟩
    · rw [col_ofFn hq']
      exact hacc ⟨q, hq'⟩
    · rw [getD_ofFn hq']
      exact hadd ⟨q, hq'⟩
    · rw [Array.getElem_toList, ← getElem!_pos mid q (by rw [hms, Array.size_range]; exact hq')]
      exact hrn ⟨q, hq'⟩

end Pouw.PearlPlus.Bridge
