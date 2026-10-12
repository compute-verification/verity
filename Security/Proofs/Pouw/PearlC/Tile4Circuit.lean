import Definitions.Pouw.PearlC.Tile4Circuit
import Proofs.Pouw.PearlC.TileValues
import Proofs.Pouw.PearlC.PeelFp4Proofs

/-!
# Pearl-C4's tile unit computes `C̃` and `U` (proofs)

`TileCircuit.tile4Spec K TM TN`, `pc4`'s tile template at depth `K` and tile `TM × TN`, evaluates to Pearl-C4's words:
for each entry `(i, j)`, on inputs holding rows `a` of `A′` and `b` of `B̃` (codes and scale bytes) and the factor rows
`P_A,i` and `P_B,j`, its two outputs are `C̃ = cTilde sm120Nvf4` over the rows and U's four K16 steps from it
(`tile4_words`), at every depth and tile (the deployed `64 × 64` at `K = 8192` among them). A transcript whose one unit
is correct carries them (`tile4_correct`), and they read back as the record `devFp4Sm120`'s ticket and useful value
(`tile4_values`, `tile4_devFp4`).

* **The firing** (`nvf4Word_of_step`): wherever `Fp4Atom.step sm120Nvf4` returns a word, the primitive's word
  (`nvf4Word`) is it: a firing that returns reads each of its eight scale bytes through `scaleDyadic`, so none is the
  UE4M3 NaN, and its accumulator is finite (`step_valid`). Its words are 32-bit (`step4_lt`).
* **The chain** (`tile4Val_chain`): the template chains `K/64` firings from `+0` with no promotion, as `chainN` does, so
  `C̃`'s gate is `cTilde`'s word wherever `cTilde` is `some`. Unlike Pearl-C's E4M3 chain there is no `F32Add_v2` to
  reconcile, and the ticket reads back directly (`tile4_ticket`).
* **U** (`tile4Val_uFold`): the four K16 steps from `C̃`'s word, as in `pc8`'s tile (`TileCircuit.uFold`).
-/

namespace Pouw.PearlC.TileCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact

/-! ## The firing on words -/

section Firing

theorem scaleDyadic_ue4m3 {s : Fin 256} {d : ℕ × ℤ} (h : scaleDyadic .ue4m3 s = some d) : s.val % 128 ≠ 127 := by
  simp only [scaleDyadic] at h
  split_ifs at h with hs
  exact hs

theorem groupTerm_scales {p : Fp4Atom} {a b : List Nib} {sa sb : List (Fin 256)} {j : ℕ} {t : GroupTerm}
    (h : p.groupTerm a b sa sb j = some t) :
    ∃ s s' d d', sa[j * p.group / p.block]? = some s ∧ scaleDyadic p.fmt s = some d ∧
      sb[j * p.group / p.block]? = some s' ∧ scaleDyadic p.fmt s' = some d' := by
  unfold Fp4Atom.groupTerm at h
  dsimp only at h
  rcases h1 : sa[j * p.group / p.block]? with _ | s
  · simp [h1] at h
  rcases h2 : scaleDyadic p.fmt s with _ | d
  · simp [h1, h2] at h
  rcases h3 : sb[j * p.group / p.block]? with _ | s'
  · simp [h1, h3] at h
  rcases h4 : scaleDyadic p.fmt s' with _ | d'
  · simp [h1, h3, h4] at h
  refine ⟨s, s', d, d', ?_, ?_, ?_, ?_⟩ <;> first | rfl | assumption

/-- **A firing that returns is valid**: its operands have the atom's lengths, none of its eight scale bytes is the
UE4M3 NaN, and its accumulator is finite. -/
theorem step_valid {acc w : ℕ} {a b : List Nib} {sa sb : List (Fin 256)} (h : sm120Nvf4.step acc a b sa sb = some w) :
    a.length = 64 ∧ b.length = 64 ∧ sa.length = 4 ∧ sb.length = 4 ∧ (∀ s ∈ sa ++ sb, s.val % 128 ≠ 127) ∧
      (acc >>> 23) % 256 ≠ 255 := by
  unfold Fp4Atom.step at h
  obtain ⟨ts, hts, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨cv, hc, -⟩ := Option.bind_eq_some_iff.1 h
  have hfin : (acc >>> 23) % 256 ≠ 255 := by
    intro he
    unfold fp32Value at hc
    simp only [he, ↓reduceIte] at hc
    split_ifs at hc <;> simp [Value.finite?] at hc
  unfold Fp4Atom.groupTerms at hts
  split_ifs at hts with hl
  obtain ⟨hla, hlb, hsa, hsb⟩ := hl
  have hk : sm120Nvf4.k / sm120Nvf4.group = 4 := rfl
  have hS : sm120Nvf4.scales = 4 := rfl
  rw [hS] at hsa hsb
  rw [hk, show List.range 4 = [0, 1, 2, 3] from rfl] at hts
  simp only [List.mapM_cons, List.mapM_nil, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff] at hts
  obtain ⟨t0, h0, -, ⟨t1, h1, -, ⟨t2, h2, -, ⟨t3, h3, -⟩, -⟩, -⟩, -⟩ := hts
  have hv : ∀ j < 4, (∀ s, sa[j]? = some s → s.val % 128 ≠ 127) ∧ (∀ s, sb[j]? = some s → s.val % 128 ≠ 127) := by
    intro j hj
    have hg : ∃ t, sm120Nvf4.groupTerm a b sa sb j = some t := by
      rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3) with rfl | rfl | rfl | rfl
      · exact ⟨t0, h0⟩
      · exact ⟨t1, h1⟩
      · exact ⟨t2, h2⟩
      · exact ⟨t3, h3⟩
    obtain ⟨t, ht⟩ := hg
    obtain ⟨s, s', d, d', e1, e2, e3, e4⟩ := groupTerm_scales ht
    have hj' : j * sm120Nvf4.group / sm120Nvf4.block = j := by
      show j * 16 / 16 = j
      omega
    rw [hj'] at e1 e3
    refine ⟨fun x hx => ?_, fun x hx => ?_⟩
    · rw [e1] at hx; cases hx; exact scaleDyadic_ue4m3 e2
    · rw [e3] at hx; cases hx; exact scaleDyadic_ue4m3 e4
  refine ⟨hla, hlb, hsa, hsb, fun s hs => ?_, hfin⟩
  rcases List.mem_append.1 hs with hs | hs
  · obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hs
    exact (hv j (hsa ▸ hj)).1 _ (List.getElem?_eq_getElem hj)
  · obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hs
    exact (hv j (hsb ▸ hj)).2 _ (List.getElem?_eq_getElem hj)

/-- **The primitive is the firing** wherever the firing returns a word. -/
theorem nvf4Word_of_step {acc w : ℕ} {a b : List Nib} {sa sb : List (Fin 256)}
    (h : sm120Nvf4.step acc a b sa sb = some w) : nvf4Word acc a b sa sb = w := by
  obtain ⟨-, -, -, -, hs, hf⟩ := step_valid h
  have hany : (sa ++ sb).any (fun s => s.val % 128 == 127) = false := by
    rw [List.any_eq_false]
    intro s hm
    simpa using hs s hm
  unfold nvf4Word
  rw [hany, ite_eq_right Bool.false_ne_true, ite_eq_right hf, h]
  rfl

theorem f32Bits_lt {q : ℕ} {grid : ℤ} (hg : -149 ≤ grid) (h127 : ¬127 < (bitLength q : ℤ) - 1 + grid) {s : ℕ}
    (hs : s < 2 ^ 32) :
    (if -126 ≤ (bitLength q : ℤ) - 1 + grid then
      (((bitLength q : ℤ) - 1 + grid + 127).toNat <<< 23) ||| (q &&& 0x7FFFFF) else q) ||| s < 2 ^ 32 := by
  refine Nat.or_lt_two_pow ?_ hs
  split_ifs with he
  · have h1 : ((bitLength q : ℤ) - 1 + grid + 127).toNat <<< 23 < 2 ^ 32 := by
      rw [Nat.shiftLeft_eq]
      have : ((bitLength q : ℤ) - 1 + grid + 127).toNat ≤ 254 := by omega
      calc ((bitLength q : ℤ) - 1 + grid + 127).toNat * 2 ^ 23 ≤ 254 * 2 ^ 23 := Nat.mul_le_mul_right _ this
        _ < 2 ^ 32 := by norm_num
    exact Nat.or_lt_two_pow h1 (Nat.and_lt_two_pow _ (by norm_num))
  · have hb : bitLength q ≤ 23 := by omega
    calc q < 2 ^ bitLength q := AccumProofs.bitLength_spec q
      _ ≤ 2 ^ 23 := Nat.pow_le_pow_right (by norm_num) hb
      _ < 2 ^ 32 := by norm_num

theorem toF32RZ_lt {n lsb : ℤ} {w : ℕ} (h : toF32RZ n lsb = some w) : w < 2 ^ 32 := by
  unfold toF32RZ at h
  dsimp only at h
  generalize hg : (if -126 ≤ (bitLength n.natAbs : ℤ) - 1 + lsb then (bitLength n.natAbs : ℤ) - 1 + lsb - 23
    else -149) = grid at h
  have hgrid : -149 ≤ grid := by rw [← hg]; split_ifs <;> omega
  generalize (if 0 < grid - lsb then n.natAbs >>> (grid - lsb).toNat else n.natAbs <<< (-(grid - lsb)).toNat) = q at h
  split at h
  · cases h; norm_num
  split at h
  · cases h
  next h127 =>
    obtain rfl := Option.some.inj h
    exact f32Bits_lt hgrid h127 (by split <;> norm_num)

/-- The firing's words are 32-bit. -/
theorem step4_lt {acc w : ℕ} {a b : List Nib} {sa sb : List (Fin 256)} (h : sm120Nvf4.step acc a b sa sb = some w) :
    w < 2 ^ 32 := by
  unfold Fp4Atom.step at h
  obtain ⟨ts, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨cv, -, h⟩ := Option.bind_eq_some_iff.1 h
  dsimp only at h
  generalize ts.filterMap (fun t => if t.live then some (t.unit - sm120Nvf4.fracBits) else none) ++
    (if cv.mag = 0 then [] else [cv.exp - ((sm120Nvf4.accWindow - 1 : ℕ) : ℤ)]) = L at h
  cases L with
  | nil => cases h; norm_num
  | cons x xs =>
    dsimp only at h
    split_ifs at h <;> first | exact toF32RZ_lt h | (cases h; norm_num)

end Firing

/-! ## The gates -/

section Gates

variable (K TM TN : ℕ)

theorem row4_mul (T : ℕ) : T * row4 K = T * K + T * (K / 16) := by unfold row4; ring

theorem gate4_input {g : ℕ} (hg : g < tile4In K TM TN) : (tile4Gate K TM TN g).input = true := by
  unfold tile4Gate
  split_ifs <;> first | omega | rfl

/-- Entry `(i, j)`'s block: `entry4Gate` at the rows `A′_i`, `B̃_j` (codes and scale bytes), `P_A,i` and `P_B,j`. -/
theorem gate4_entry {i j r : ℕ} (hj : j < TN) (hr : r < block4Len K) :
    tile4Gate K TM TN (entry4Base K TM TN i j + r) =
      entry4Gate K (i * K) (TM * row4 K + j * K) (TM * K + i * (K / 16)) (TM * row4 K + TN * K + j * (K / 16))
        ((TM + TN) * row4 K + 64 * i) ((TM + TN) * row4 K + 64 * TM + 64 * j) (entry4Base K TM TN i j + r) r := by
  have hL := block4Len_pos K
  have hTN : 0 < TN := by omega
  have h1 : entry4Base K TM TN i j + r - tile4In K TM TN = r + (i * TN + j) * block4Len K := by
    unfold entry4Base; omega
  have h2 : (r + (i * TN + j) * block4Len K) / block4Len K = i * TN + j := by
    rw [Nat.add_mul_div_right _ _ hL, Nat.div_eq_of_lt hr, Nat.zero_add]
  have h3 : (r + (i * TN + j) * block4Len K) % block4Len K = r := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]
  have h4 : (i * TN + j) / TN = i := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hTN, Nat.div_eq_of_lt hj, Nat.zero_add]
  have h5 : (i * TN + j) % TN = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
  have hin : tile4In K TM TN ≤ entry4Base K TM TN i j + r := by unfold entry4Base; omega
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hS' : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hS'' : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  have hk : (TM + TN) * row4 K ≤ tile4In K TM TN := by unfold tile4In; omega
  unfold tile4Gate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega)]
  dsimp only
  rw [h1, h2, h3, h4, h5]

theorem entry4_atom (a b sa sb pa pb g : ℕ) {r : ℕ} (hr : r < K / 64) :
    entry4Gate K a b sa sb pa pb g r =
      ⟨32, false, (if r = 0 then [] else [g - 1]) ++ (List.range 64).map (a + 64 * r + ·) ++
          (List.range 64).map (b + 64 * r + ·) ++ (List.range 4).map (sa + 4 * r + ·) ++
          (List.range 4).map (sb + 4 * r + ·), atom4F (r == 0)⟩ := by
  unfold entry4Gate
  rw [ite_eq_left hr]

theorem entry4_peel (a b sa sb pa pb g s : ℕ) :
    entry4Gate K a b sa sb pa pb g (K / 64 + s) =
      ⟨32, false, [g - 1] ++ (List.range 16).map (pa + 16 * s + ·) ++ (List.range 16).map (pb + 16 * s + ·),
        peelF⟩ := by
  unfold entry4Gate
  rw [ite_eq_right (by omega), show K / 64 + s - K / 64 = s by omega]

theorem getD_quad (f h u v V : ℕ → ℕ) :
    (∀ i < 64, (((List.range 64).map f ++ ((List.range 64).map h ++ ((List.range 4).map u ++
      (List.range 4).map v))).map V).getD i 0 = V (f i) ∧
      (((List.range 64).map f ++ ((List.range 64).map h ++ ((List.range 4).map u ++
        (List.range 4).map v))).map V).getD (64 + i) 0 = V (h i)) ∧
    ∀ i < 4, (((List.range 64).map f ++ ((List.range 64).map h ++ ((List.range 4).map u ++
      (List.range 4).map v))).map V).getD (128 + i) 0 = V (u i) ∧
      (((List.range 64).map f ++ ((List.range 64).map h ++ ((List.range 4).map u ++
        (List.range 4).map v))).map V).getD (132 + i) 0 = V (v i) := by
  refine ⟨fun i hi => ⟨?_, ?_⟩, fun i hi => ⟨?_, ?_⟩⟩
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append, hi]
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append, hi]
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append, show ¬128 + i < 64 by omega,
      show ¬128 + i - 64 < 64 by omega, show 128 + i - 64 - 64 = i by omega, hi]
  · simp [List.getD_eq_getElem?_getD, List.getElem?_append, show ¬132 + i < 64 by omega,
      show ¬132 + i - 64 < 64 by omega, show ¬132 + i - 64 - 64 < 4 by omega, show 132 + i - 64 - 64 - 4 = i by omega,
      hi]

/-- A firing's reads: the accumulator (`+0` for the first, else gate `g − 1`), then its 64 + 64 codes and 4 + 4 scale
bytes. -/
theorem atom4F_reads (r g : ℕ) (f h u v V : ℕ → ℕ) :
    atom4F (r == 0) (((if r = 0 then [] else [g - 1]) ++ (List.range 64).map f ++ (List.range 64).map h ++
        (List.range 4).map u ++ (List.range 4).map v).map V) =
      nvf4Word (if r = 0 then 0 else V (g - 1)) (List.ofFn fun l : Fin 64 => nib (V (f l)))
        (List.ofFn fun l : Fin 64 => nib (V (h l))) (List.ofFn fun l : Fin 4 => byte (V (u l)))
        (List.ofFn fun l : Fin 4 => byte (V (v l))) := by
  have q := getD_quad f h u v V
  simp only [List.append_assoc]
  generalize (List.range 64).map f ++ ((List.range 64).map h ++ ((List.range 4).map u ++ (List.range 4).map v)) = L
    at q ⊢
  rcases Nat.eq_zero_or_pos r with rfl | h0
  · simp only [atom4F, beq_self_eq_true, ↓reduceIte, List.nil_append]
    congr 1
    · exact congrArg List.ofFn (funext fun l => by rw [(q.1 l l.isLt).1])
    · exact congrArg List.ofFn (funext fun l => by rw [(q.1 l l.isLt).2])
    · exact congrArg List.ofFn (funext fun l => by rw [(q.2 l l.isLt).1])
    · exact congrArg List.ofFn (funext fun l => by rw [(q.2 l l.isLt).2])
  · have hb : (r == 0) = false := by simp; omega
    simp only [atom4F, hb, Bool.false_eq_true, ↓reduceIte, ite_eq_right (show r ≠ 0 by omega),
      List.cons_append, List.nil_append, List.map_cons, List.headD_cons, List.tail_cons]
    congr 1
    · exact congrArg List.ofFn (funext fun l => by rw [(q.1 l l.isLt).1])
    · exact congrArg List.ofFn (funext fun l => by rw [(q.1 l l.isLt).2])
    · exact congrArg List.ofFn (funext fun l => by rw [(q.2 l l.isLt).1])
    · exact congrArg List.ofFn (funext fun l => by rw [(q.2 l l.isLt).2])

theorem entry4_lt {i j : ℕ} (hi : i < TM) (hj : j < TN) :
    entry4Base K TM TN i j + block4Len K ≤ tile4N K TM TN := by
  have h1 : i * TN + TN ≤ TM * TN := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right TN hi
  have h2 : (i * TN + j + 1) * block4Len K ≤ TM * TN * block4Len K := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h2
  unfold entry4Base tile4N
  omega

/-- The circuit's values. -/
abbrev tile4Val (a : (tile4Spec K TM TN).toCircuit.Transcript) : ℕ → ℕ := val ((tile4Spec K TM TN).toCircuit.eval a)

theorem tile4Val_atom (a : (tile4Spec K TM TN).toCircuit.Transcript) {i j r : ℕ} (hi : i < TM) (hj : j < TN)
    (hr : r < K / 64) :
    tile4Val K TM TN a (entry4Base K TM TN i j + r) =
      nvf4Word (if r = 0 then 0 else tile4Val K TM TN a (entry4Base K TM TN i j + r - 1))
        (List.ofFn fun l : Fin 64 => nib (tile4Val K TM TN a (i * K + 64 * r + l)))
        (List.ofFn fun l : Fin 64 => nib (tile4Val K TM TN a (TM * row4 K + j * K + 64 * r + l)))
        (List.ofFn fun l : Fin 4 => byte (tile4Val K TM TN a (TM * K + i * (K / 16) + 4 * r + l)))
        (List.ofFn fun l : Fin 4 => byte (tile4Val K TM TN a (TM * row4 K + TN * K + j * (K / 16) + 4 * r + l))) %
        2 ^ 32 := by
  have hr5 : r < block4Len K := by unfold block4Len; omega
  have hN : entry4Base K TM TN i j + r < (tile4Spec K TM TN).N := by
    have := entry4_lt K TM TN hi hj; show _ < tile4N K TM TN; omega
  have hg := gate4_entry K TM TN (i := i) hj hr5
  rw [entry4_atom K _ _ _ _ _ _ _ hr] at hg
  have hin : ((tile4Spec K TM TN).gate (entry4Base K TM TN i j + r)).input = false := by
    show (tile4Gate K TM TN _).input = false
    rw [hg]
  rw [tile4Val, Spec.val_gate _ a hN hin]
  show (tile4Gate K TM TN _).f ((tile4Gate K TM TN _).rd.map _) % 2 ^ (tile4Gate K TM TN _).w = _
  rw [hg]
  dsimp only
  rw [atom4F_reads]

theorem tile4Val_input (a : (tile4Spec K TM TN).toCircuit.Transcript) {g : ℕ} (hg : g < tile4In K TM TN) :
    tile4Val K TM TN a g = val a g := by
  have hN : g < (tile4Spec K TM TN).N := by show g < tile4N K TM TN; unfold tile4N; omega
  rw [tile4Val, Spec.val_input _ a hN (gate4_input K TM TN hg), val_of_lt _ hN]

theorem tile4Val_peel (a : (tile4Spec K TM TN).toCircuit.Transcript) {i j s : ℕ} (hi : i < TM) (hj : j < TN)
    (hs : s < 4) :
    tile4Val K TM TN a (entry4Base K TM TN i j + (K / 64 + s)) =
      k16Step (tile4Val K TM TN a (entry4Base K TM TN i j + (K / 64 + s) - 1))
        (fun l => tile4Val K TM TN a ((TM + TN) * row4 K + 64 * i + 16 * s + l))
        (fun l => tile4Val K TM TN a ((TM + TN) * row4 K + 64 * TM + 64 * j + 16 * s + l)) := by
  have hr5 : K / 64 + s < block4Len K := by unfold block4Len; omega
  have hN : entry4Base K TM TN i j + (K / 64 + s) < (tile4Spec K TM TN).N := by
    have := entry4_lt K TM TN hi hj; show _ < tile4N K TM TN; omega
  have hg := gate4_entry K TM TN (i := i) hj hr5
  rw [entry4_peel K] at hg
  have hin : ((tile4Spec K TM TN).gate (entry4Base K TM TN i j + (K / 64 + s))).input = false := by
    show (tile4Gate K TM TN _).input = false
    rw [hg]
  rw [tile4Val, Spec.val_gate _ a hN hin]
  show (tile4Gate K TM TN _).f ((tile4Gate K TM TN _).rd.map _) % 2 ^ (tile4Gate K TM TN _).w = _
  rw [hg]
  dsimp only
  rw [Nat.mod_eq_of_lt (by unfold peelF k16Step; exact AccumProofs.packValue_lt _)]
  simp only [peelF, List.cons_append, List.nil_append, List.map_cons, List.headD_cons]
  congr 1 <;> funext l
  · rw [Nat.add_comm 1, List.getD_cons_succ, (getD_pair _ _ _ l.isLt).1]
  · rw [show 17 + (l : ℕ) = (16 + l) + 1 by omega, List.getD_cons_succ, (getD_pair _ _ _ l.isLt).2]

/-- **U's gate**: entry `(i, j)`'s four K16 steps from its `C̃` gate over the factor rows `P_A,i` and `P_B,j`. -/
theorem tile4Val_uFold (a : (tile4Spec K TM TN).toCircuit.Transcript) {i j : ℕ} (hi : i < TM) (hj : j < TN)
    (pa pb : List ℕ) (hpa : ∀ q < 64, val a ((TM + TN) * row4 K + 64 * i + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val a ((TM + TN) * row4 K + 64 * TM + 64 * j + q) = pb.getD q 0) :
    tile4Val K TM TN a (u4Gate K TM TN i j) = uFold pa pb (tile4Val K TM TN a (c4Gate K TM TN i j)) := by
  have h64i : 64 * i + 64 ≤ 64 * TM := by omega
  have h64j : 64 * j + 64 ≤ 64 * TN := by omega
  have hIn : tile4In K TM TN = (TM + TN) * row4 K + 64 * TM + 64 * TN := by unfold tile4In; omega
  have sl : ∀ s < 4, (fun l : Fin 16 => tile4Val K TM TN a ((TM + TN) * row4 K + 64 * i + 16 * s + l)) = slice16 pa s ∧
      (fun l : Fin 16 => tile4Val K TM TN a ((TM + TN) * row4 K + 64 * TM + 64 * j + 16 * s + l)) =
        slice16 pb s :=
    fun s hs => by
      constructor <;> funext l
      · rw [tile4Val_input K TM TN a (by omega), Nat.add_assoc, hpa _ (by omega)]; rfl
      · rw [tile4Val_input K TM TN a (by omega), Nat.add_assoc, hpb _ (by omega)]; rfl
  set E := entry4Base K TM TN i j with hE
  have e : ∀ s < 4, tile4Val K TM TN a (E + (K / 64 + s)) =
      k16Step (tile4Val K TM TN a (E + (K / 64 + s) - 1)) (slice16 pa s) (slice16 pb s) := fun s hs => by
    rw [tile4Val_peel K TM TN a hi hj hs, (sl s hs).1, (sl s hs).2]
  have e0 := e 0 (by omega)
  have e1 := e 1 (by omega)
  have e2 := e 2 (by omega)
  have e3 := e 3 (by omega)
  rw [show E + (K / 64 + 0) - 1 = c4Gate K TM TN i j by unfold c4Gate; omega] at e0
  rw [show E + (K / 64 + 1) - 1 = E + (K / 64 + 0) by omega, e0] at e1
  rw [show E + (K / 64 + 2) - 1 = E + (K / 64 + 1) by omega, e1] at e2
  rw [show E + (K / 64 + 3) - 1 = E + (K / 64 + 2) by omega, e2] at e3
  rw [show u4Gate K TM TN i j = E + (K / 64 + 3) by unfold u4Gate block4Len; omega, e3]
  rfl

end Gates

/-! ## The chain -/

section Chain

variable (K TM TN : ℕ)

theorem nib_val (c : Nib) : nib c.val = c := Fin.ext (Nat.mod_eq_of_lt c.isLt)

theorem byte_val (c : Fin 256) : byte c.val = c := Fin.ext (Nat.mod_eq_of_lt c.isLt)

theorem ofFn_getD {α : Type} (L : List α) (d : α) {s n : ℕ} (h : s + n ≤ L.length) :
    (List.ofFn fun l : Fin n => L.getD (s + l) d) = (L.drop s).take n := by
  apply List.ext_getElem
  · simp; omega
  · intro k h1 h2
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (show s + k < L.length by simp at h1; omega)]

/-- **A firing's inputs**: entry `(i, j)`'s firing `t` reads codes `64t, …` and scale bytes `4t, …` of rows `x` of `A′`
and `y` of `B̃`, as `chainN` takes them. -/
theorem fire4_inputs (a : (tile4Spec K TM TN).toCircuit.Transcript) (x y : Row4) {i j : ℕ} (hi : i < TM)
    (hj : j < TN) (hA : ∀ l < K, val a (i * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val a (TM * K + i * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val a (TM * row4 K + j * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val a (TM * row4 K + TN * K + j * (K / 16) + l) = (y.2.getD l 0).val)
    (hx : x.1.length = K) (hy : y.1.length = K) (hsx : x.2.length = K / 64 * 4) (hsy : y.2.length = K / 64 * 4)
    {t : ℕ} (ht : t < K / 64) :
    (List.ofFn fun l : Fin 64 => nib (tile4Val K TM TN a (i * K + 64 * t + l))) = (x.1.drop (64 * t)).take 64 ∧
      (List.ofFn fun l : Fin 64 => nib (tile4Val K TM TN a (TM * row4 K + j * K + 64 * t + l))) =
        (y.1.drop (64 * t)).take 64 ∧
      (List.ofFn fun l : Fin 4 => byte (tile4Val K TM TN a (TM * K + i * (K / 16) + 4 * t + l))) =
        (x.2.drop (4 * t)).take 4 ∧
      (List.ofFn fun l : Fin 4 => byte (tile4Val K TM TN a (TM * row4 K + TN * K + j * (K / 16) + 4 * t + l))) =
        (y.2.drop (4 * t)).take 4 := by
  have hA' : i * K + K ≤ TM * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hi
  have hB' : j * K + K ≤ TN * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hj
  have hSA' : i * (K / 16) + K / 16 ≤ TM * (K / 16) := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hi
  have hSB' : j * (K / 16) + K / 16 ≤ TN * (K / 16) := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hj
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hS' : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hS'' : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  have hIn : (TM + TN) * row4 K ≤ tile4In K TM TN := by unfold tile4In; omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← ofFn_getD _ _ (by omega)]
    congr 1; funext l
    rw [tile4Val_input K TM TN a (by omega), Nat.add_assoc, hA _ (by omega), nib_val]
  · rw [← ofFn_getD _ _ (by omega)]
    congr 1; funext l
    rw [tile4Val_input K TM TN a (by omega), Nat.add_assoc, hB _ (by omega), nib_val]
  · rw [← ofFn_getD _ _ (by omega)]
    congr 1; funext l
    rw [tile4Val_input K TM TN a (by omega), Nat.add_assoc, hSA _ (by omega), byte_val]
  · rw [← ofFn_getD _ _ (by omega)]
    congr 1; funext l
    rw [tile4Val_input K TM TN a (by omega), Nat.add_assoc, hSB _ (by omega), byte_val]

/-- **`C̃_ij`'s gate**: entry `(i, j)`'s `K/64` firings write `cTilde sm120Nvf4`'s word over rows `x` of `A′` and `y`
of `B̃`, wherever it is `some`. -/
theorem tile4Val_chain (a : (tile4Spec K TM TN).toCircuit.Transcript) (x y : Row4) {i j : ℕ} (hi : i < TM)
    (hj : j < TN) (hK : 0 < K) (hx : x.1.length = K) (hA : ∀ l < K, val a (i * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val a (TM * K + i * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val a (TM * row4 K + j * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val a (TM * row4 K + TN * K + j * (K / 16) + l) = (y.2.getD l 0).val) {c : ℕ}
    (hc : cTilde sm120Nvf4 x.1 y.1 x.2 y.2 = some c) : tile4Val K TM TN a (c4Gate K TM TN i j) = c := by
  have hk : sm120Nvf4.k = 64 := rfl
  have hS : sm120Nvf4.scales = 4 := rfl
  unfold cTilde Fp4Atom.chain at hc
  split_ifs at hc with hl
  obtain ⟨hyx, hmod, hsx, hsy⟩ := hl
  rw [hx, hk, hS] at hsx hsy
  rw [hx] at hyx hmod
  rw [hk] at hmod
  rw [hx, hk] at hc
  have hN : 0 < K / 64 := by omega
  set E := entry4Base K TM TN i j
  have key : ∀ n t c0 w, t + n = K / 64 → (if t = 0 then 0 else tile4Val K TM TN a (E + t - 1)) = c0 →
      sm120Nvf4.chainN n c0 (x.1.drop (64 * t)) (y.1.drop (64 * t)) (x.2.drop (4 * t)) (y.2.drop (4 * t)) = some w →
      tile4Val K TM TN a (E + K / 64 - 1) = w := by
    intro n
    induction n with
    | zero =>
      intro t c0 w ht hc0 h
      simp only [Fp4Atom.chainN, Option.some.injEq] at h
      subst h
      obtain rfl : t = K / 64 := by omega
      rwa [ite_eq_right (by omega)] at hc0
    | succ n ih =>
      intro t c0 w ht hc0 h
      simp only [Fp4Atom.chainN] at h
      obtain ⟨d, hd, h⟩ := Option.bind_eq_some_iff.1 h
      rw [hk, hS] at hd h
      obtain ⟨e1, e2, e3, e4⟩ := fire4_inputs K TM TN a x y hi hj hA hSA hB hSB hx hyx hsx hsy
        (t := t) (by omega)
      have hv : tile4Val K TM TN a (E + t) = d := by
        rw [tile4Val_atom K TM TN a hi hj (by omega), hc0, e1, e2, e3, e4, nvf4Word_of_step hd,
          Nat.mod_eq_of_lt (step4_lt hd)]
      refine ih (t + 1) d w (by omega) ?_ ?_
      · rw [ite_eq_right (by omega), show E + (t + 1) - 1 = E + t by omega, hv]
      · rw [List.drop_drop, List.drop_drop, List.drop_drop, List.drop_drop,
          show 64 * t + 64 = 64 * (t + 1) by ring, show 4 * t + 4 = 4 * (t + 1) by ring] at h
        exact h
  exact key (K / 64) 0 0 c (by omega) rfl (by simpa using hc)

/-- **The tile unit's words**: on inputs holding rows `x` of `A′` and `y` of `B̃` (codes and scale bytes) and the
factor rows `P_A,i` and `P_B,j`, entry `(i, j)`'s two outputs are `C̃_ij = cTilde sm120Nvf4` and U's four K16 steps
from it (`uFold`). -/
theorem tile4_words (a : (tile4Spec K TM TN).toCircuit.Transcript) (x y : Row4) (pa pb : List ℕ) {i j : ℕ}
    (hi : i < TM) (hj : j < TN) (hK : 0 < K) (hx : x.1.length = K)
    (hA : ∀ l < K, val a (i * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val a (TM * K + i * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val a (TM * row4 K + j * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val a (TM * row4 K + TN * K + j * (K / 16) + l) = (y.2.getD l 0).val)
    (hpa : ∀ q < 64, val a ((TM + TN) * row4 K + 64 * i + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val a ((TM + TN) * row4 K + 64 * TM + 64 * j + q) = pb.getD q 0) {c : ℕ}
    (hc : cTilde sm120Nvf4 x.1 y.1 x.2 y.2 = some c) :
    tile4Val K TM TN a (c4Gate K TM TN i j) = c ∧ tile4Val K TM TN a (u4Gate K TM TN i j) = uFold pa pb c := by
  have hC := tile4Val_chain K TM TN a x y hi hj hK hx hA hSA hB hSB hc
  exact ⟨hC, by rw [tile4Val_uFold K TM TN a hi hj pa pb hpa hpb, hC]⟩

theorem c4Gate_mem {i j : ℕ} (hi : i < TM) (hj : j < TN) : c4Gate K TM TN i j ∈ (tile4Spec K TM TN).outs :=
  Finset.mem_union_left _ (Finset.mem_image.2 ⟨(i, j), Finset.mem_product.2 ⟨Finset.mem_range.2 hi,
    Finset.mem_range.2 hj⟩, rfl⟩)

theorem u4Gate_mem {i j : ℕ} (hi : i < TM) (hj : j < TN) : u4Gate K TM TN i j ∈ (tile4Spec K TM TN).outs :=
  Finset.mem_union_right _ (Finset.mem_image.2 ⟨(i, j), Finset.mem_product.2 ⟨Finset.mem_range.2 hi,
    Finset.mem_range.2 hj⟩, rfl⟩)

/-- A transcript whose one unit is correct holds the circuit's values on every output (core's
`Verity.Partition.compose`). -/
theorem tile4_out (X : (tile4Spec K TM TN).toCircuit.Transcript) (hX : (tile4Spec K TM TN).part.Correct X 0) :
    ∀ g ∈ (tile4Spec K TM TN).outs, val X g = tile4Val K TM TN X g := by
  have hw : (tile4Spec K TM TN).part.wrong X = ∅ := by
    ext u
    simp only [Verity.Partition.mem_wrong, Finset.notMem_empty, iff_false, not_not]
    rwa [Subsingleton.elim u 0]
  intro g hg
  have hlt : g < (tile4Spec K TM TN).toCircuit.N := by
    show g < tile4N K TM TN
    simp only [tile4Spec, Finset.mem_union, Finset.mem_image, Finset.mem_product, Finset.mem_range,
      Prod.exists] at hg
    rcases hg with ⟨i, j, ⟨hi, hj⟩, rfl⟩ | ⟨i, j, ⟨hi, hj⟩, rfl⟩
    · have := entry4_lt K TM TN hi hj; unfold c4Gate; unfold block4Len at this; omega
    · have := entry4_lt K TM TN hi hj; have := block4Len_pos K; unfold u4Gate; omega
  have hg' : (⟨g, hlt⟩ : Fin (tile4Spec K TM TN).toCircuit.N) ∈ (tile4Spec K TM TN).part.committed :=
    Verity.Partition.mem_committed_of_output _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hg⟩)
  rw [tile4Val, val_of_lt _ hlt, val_of_lt _ hlt, (tile4Spec K TM TN).part.compose X X (fun _ _ => rfl) hw _ hg']

/-- **Pearl-C4's tile unit, correct**: a transcript whose one unit is correct carries `C̃_ij` and `U_ij` on entry
`(i, j)`'s outputs. -/
theorem tile4_correct (X : (tile4Spec K TM TN).toCircuit.Transcript) (hX : (tile4Spec K TM TN).part.Correct X 0)
    (x y : Row4) (pa pb : List ℕ) {i j : ℕ} (hi : i < TM) (hj : j < TN) (hK : 0 < K) (hx : x.1.length = K)
    (hA : ∀ l < K, val X (i * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val X (TM * K + i * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val X (TM * row4 K + j * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val X (TM * row4 K + TN * K + j * (K / 16) + l) = (y.2.getD l 0).val)
    (hpa : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * i + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * TM + 64 * j + q) = pb.getD q 0) {c : ℕ}
    (hc : cTilde sm120Nvf4 x.1 y.1 x.2 y.2 = some c) :
    val X (c4Gate K TM TN i j) = c ∧ val X (u4Gate K TM TN i j) = uFold pa pb c := by
  obtain ⟨h1, h2⟩ := tile4_words K TM TN X x y pa pb hi hj hK hx hA hSA hB hSB hpa hpb hc
  have hout := tile4_out K TM TN X hX
  exact ⟨(hout _ (c4Gate_mem K TM TN hi hj)).trans h1, (hout _ (u4Gate_mem K TM TN hi hj)).trans h2⟩

end Chain

/-! ## The ticket and the useful value -/

section Values

variable {K TM TN : ℕ}

/-- **`C̃_ij`'s word is the ticket**: on rows `A i = x` and `B j = y` whose chain `cTilde sm120Nvf4` is `some`, `C̃`'s
output reads back as `ticket4 sm120Nvf4`, the record `devFp4Sm120`'s ticket. -/
theorem tile4_ticket {m n : ℕ} (X : (tile4Spec K TM TN).toCircuit.Transcript)
    (hX : (tile4Spec K TM TN).part.Correct X 0) (A : Op4 m) (B : Op4 n) (i : Fin m) (j : Fin n) {x y : Row4}
    (hAi : A i = some x) (hBj : B j = some y) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN) (hK : 0 < K)
    (hx : x.1.length = K) (hA : ∀ l < K, val X (ti * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val X (TM * K + ti * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val X (TM * row4 K + tj * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val X (TM * row4 K + TN * K + tj * (K / 16) + l) = (y.2.getD l 0).val) {c : ℕ}
    (hc : cTilde sm120Nvf4 x.1 y.1 x.2 y.2 = some c) :
    wordQ (val X (c4Gate K TM TN ti tj)) = ticket4 sm120Nvf4 A B i j := by
  rw [tile4_out K TM TN X hX _ (c4Gate_mem K TM TN hti htj),
    tile4Val_chain K TM TN X x y hti htj hK hx hA hSA hB hSB hc]
  simp [ticket4, hAi, hBj, hc]

/-- **The tile unit computes Pearl-C4's ticket and useful value**: a transcript whose one unit is correct, on inputs
holding rows `A i = x` of `A′` and `B j = y` of `B̃` (codes and scale bytes) and the factor rows `P_A,i`, `P_B,j` that
U's pinned order forms from them (`PeelFp4Proofs.peelRows4`), has on entry `(ti, tj)`'s outputs `C̃`'s word of value
`ticket4 sm120Nvf4` and `U`'s of value `peelFp4 sm120Nvf4` at `(i, j)`: the record `devFp4Sm120`'s ticket and useful
value (`tile4_devFp4`). -/
theorem tile4_values {m n k : ℕ} (X : (tile4Spec K TM TN).toCircuit.Transcript)
    (hX : (tile4Spec K TM TN).part.Correct X 0) (A : Op4 m) (B : Op4 n) (FA FB : Matrix (Fin k) (Fin 32) Nib)
    (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ) (i : Fin m) (j : Fin n) {x y : Row4}
    (hAi : A i = some x) (hBj : B j = some y) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN) (hK : 0 < K)
    (hx : x.1.length = K) (hA : ∀ l < K, val X (ti * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val X (TM * K + ti * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val X (TM * row4 K + tj * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val X (TM * row4 K + TN * K + tj * (K / 16) + l) = (y.2.getD l 0).val)
    {c : ℕ} {pa pb : List ℕ} (hr : PeelFp4Proofs.peelRows4 sm120Nvf4 A B FA FB beA beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * TM + 64 * tj + q) = pb.getD q 0) :
    wordQ (val X (c4Gate K TM TN ti tj)) = ticket4 sm120Nvf4 A B i j ∧
      wordQ (val X (u4Gate K TM TN ti tj)) = (peelFp4 sm120Nvf4 A B FA FB beA beB i j).bind wordQ := by
  obtain ⟨x', y', -, -, -, hA', hB', hc, -⟩ := PeelFp4Proofs.peelRows4_spec hr
  rw [hAi] at hA'
  cases hA'
  rw [hBj] at hB'
  cases hB'
  obtain ⟨c', hc', hfin⟩ := Option.bind_eq_some_iff.1 hc
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some hfin
  obtain ⟨-, hU⟩ := tile4_correct K TM TN X hX x y pa pb hti htj hK hx hA hSA hB hSB hpa hpb hc'
  refine ⟨tile4_ticket X hX A B i j hAi hBj hti htj hK hx hA hSA hB hSB hc', ?_⟩
  rw [hU, wordQ_uFold, PeelFp4Proofs.peelFp4_eq_rows, hr, Option.bind_some]

/-- `tile4_values` at the record `devFp4Sm120`: its ticket and useful value. -/
theorem tile4_devFp4 {m n k : ℕ} (X : (tile4Spec K TM TN).toCircuit.Transcript)
    (hX : (tile4Spec K TM TN).part.Correct X 0) (A : Op4 m) (B : Op4 n) (FA FB : Matrix (Fin k) (Fin 32) Nib)
    (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ) (i : Fin m) (j : Fin n) {x y : Row4}
    (hAi : A i = some x) (hBj : B j = some y) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN) (hK : 0 < K)
    (hx : x.1.length = K) (hA : ∀ l < K, val X (ti * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val X (TM * K + ti * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val X (TM * row4 K + tj * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val X (TM * row4 K + TN * K + tj * (K / 16) + l) = (y.2.getD l 0).val)
    {c : ℕ} {pa pb : List ℕ} (hr : PeelFp4Proofs.peelRows4 sm120Nvf4 A B FA FB beA beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * TM + 64 * tj + q) = pb.getD q 0) :
    wordQ (val X (c4Gate K TM TN ti tj)) = devFp4Sm120.ticket A B i j ∧
      wordQ (val X (u4Gate K TM TN ti tj)) = devFp4Sm120.useful A B FA FB beA beB i j :=
  tile4_values X hX A B FA FB beA beB i j hAi hBj hti htj hK hx hA hSA hB hSB hr hpa hpb

end Values

end Pouw.PearlC.TileCircuit
