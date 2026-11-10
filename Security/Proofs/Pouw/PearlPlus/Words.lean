module

public import PouwPearlPlus.Words
public import Definitions.Pouw.PearlC.DeviceRev1
public import Definitions.Pouw.PearlC.RowSem
public import Definitions.Pouw.PearlC.NoiseSha512
public import Proofs.Pouw.PearlPlus.Noise
public import Proofs.Pouw.PearlPlus.Numerics
public import Proofs.Pouw.PearlPlus.Atom
public import Proofs.Pouw.PearlC.TileValues
public import Proofs.Pouw.PearlC.TileCheckWords

/-!
# Pearl+'s bridge, part (b): a tile's words under -h3's noise are the semantics'

`PouwCheck.PearlPlus.Words.encode` and `checked` form a tile's operands from its opened rows and compute its checked words
C̃ and U. The game reads the reference words of `pearlCProtocolDevRev1 devH100 sem ρ` at the deployed semantics `semH3`:
Pearl-C at `pc8`'s row arithmetic on core's primitives and the H100 atom, with -h3's noise from SHA-512
(`sha512Sem`). `WordsAreTheSemantics` says the check's words are those values on every word of a credited row.

It composes the noise (`Noise.lean`), the row statistics (`StatsAreTheRow`), the forming, the chain (the H100 chain is
`chainWordP? hopperE4m3K32 4`, whose value `TileCircuit.chainWordP_value` gives), the peel factor rows (`PeelARow`,
`PeelBRow`) and the peel (`peelExactP` at the H100). The proof is `Semantics.lean`'s, since it reads `Stats.lean` and
`Peel.lean`, which import this file. The check reads a row's raw words and the semantics their values re-encoded
(`wordsVal`, `encRow`), so a `−0` word and an A′ or B̃ code `0x80` agree with the semantics' `+0` in value only; every
statement here is on values (`wordQ`).
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- Row `i` of a matrix of FP32 words with `k` columns as the bytes a row opening carries (`NoiseSha512.rowBytes`). -/
def rowBytesOf (W : Codes) (k i : ℕ) : ByteArray := ⟨((NoiseSha512.rowBytes W k i).map Nat.toUInt8).toArray⟩

/-- **Pearl+'s semantics**: Pearl-C at `pc8`'s row arithmetic on core's primitives and the H100 atom, with -h3's noise
from SHA-512 at the salt's bytes and the weights' roots. -/
def semH3 {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) : PearlCSem Q R S :=
  sha512Sem (RowCircuit.pc8Sem RowCircuit.coreOps hopperE4m3K32
    fun _ _ _ _ _ => ⟨fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0⟩) saltOf rootOf

/-- Pearl+'s protocol at the H100 record under rev1, at the deployed semantics and cap `ρ`. -/
noncomputable def protocolH3 {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) : Pouw.PearlC.Protocol Q R S :=
  pearlCProtocolDevRev1 devH100 (semH3 saltOf rootOf) ρ

/-- Pearl+'s audit tiles at the H100 record under rev1, at the deployed semantics and cap `ρ`. -/
noncomputable def tilesH3 {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) : Pouw.PearlC.TileRules Q R S :=
  pearlCTilesDevRev1 devH100 (semH3 saltOf rootOf) ρ

/-- A row of FP32 words as values, the way `wordsVal` reads a committed row. -/
def rowVals (k : ℕ) (x : Array ℕ) : Fin k → ℚ := fun l => (wordQ x[l.val]!).getD 0

/-- **(b) The row statistics are `pc8`'s**: on a row of `k` finite FP32 words, `s5_stats`' ρ, α and β are `pc8`'s
`rowRho`, `alpha8` and `beta8` of the row's values re-encoded. `8 ∣ k`, as `s5_stats` reads `⌈k/8⌉` positions where
`rowRho` reads `⌊k/8⌋`; `k ≤ 2^27`, as `Words.ofInt` is exact only up to `2^24`. -/
def StatsAreTheRow : Prop :=
  ∀ (k : ℕ) (x : Array ℕ) (s ρw αw βw : ℕ), x.size = k → 0 < k → 8 ∣ k → k ≤ 2 ^ 27 →
    (∀ l < k, x[l]! < 2 ^ 32 ∧ (wordQ x[l]!).isSome) →
    Words.s5Stats x = .ok (s, ρw, αw, βw) →
      ρw = RowCircuit.rowRho RowCircuit.coreOps .id (RowCircuit.encRow k (rowVals k x)) k ∧
        αw = RowCircuit.alpha8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k ∧
        βw = RowCircuit.beta8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k

/-- **(b) A tile's words are the semantics'**: for a unit `u` of `U` with salt and weight root the semantics' bytes, a
depth `k` within the domain's `K_MAX` (which the run checks when it loads the pass), a tile's rows and columns in the
unit, every weight row of its columns passing the rules (as `TileGood` asks), opened rows that are the commitment's
activation rows and the weight's rows, and the check's noise, operands and checked words values (no rejection): each
word of a row that passes the rules has the reference ticket and useful values. -/
def WordsAreTheSemantics : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) (H : Q → R) (s : S) (U : Pouw.PearlC.Workload) (u : ℕ)
    (act : Codes) (salt rootB : ByteArray) (wn : Words.FixedNoise) (rows cols : List ℕ) (ops : Words.Ops)
    (ch : Words.Checked),
    bytesNat salt = saltOf s → bytesNat rootB = rootOf (U.layout.fixedId u) → (U.layout.shape u).k ≤ 2 ^ 16 →
    (∀ i ∈ rows, i < (U.layout.shape u).m) → (∀ j ∈ cols, j < (U.layout.shape u).n) →
    (∀ i ∈ rows, ∀ l < (U.layout.shape u).k, act i l < 2 ^ 32) →
    (∀ j ∈ cols, ∀ l < (U.layout.shape u).k, U.fixed (U.layout.fixedId u) j l < 2 ^ 32) →
    (∀ j ∈ cols, (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape u).n (U.layout.shape u).k
      (U.fixed (U.layout.fixedId u)) j) →
    Words.fixedNoise salt rootB (U.layout.shape u).k = .ok wn →
    Words.encode salt rootB wn u (U.layout.shape u).k rows (rows.map (rowBytesOf act (U.layout.shape u).k)) cols
      (cols.map (rowBytesOf (U.fixed (U.layout.fixedId u)) (U.layout.shape u).k)) = .ok ops →
    Words.checked ops = .ok ch →
    ∀ x < rows.length, (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape u).m (U.layout.shape u).k act
        rows[x]! →
      ∀ z < cols.length,
        wordQ ((ch.c[x]!)[z]!) = (protocolH3 saltOf rootOf ρ : Pouw.PearlC.Protocol Q R S).ticket H s U u act rows[x]! cols[z]! ∧
          wordQ ((ch.u[x]!)[z]!) = (protocolH3 saltOf rootOf ρ : Pouw.PearlC.Protocol Q R S).useful H s U u act rows[x]! cols[z]!

/-- **(b) `P_A`'s rows are `paRow?`'s**: where `peelA` is a value on a formed side whose row `x` has 32 `E′` codes `E`,
none NaN, and `F_B` has 32 columns, row `x` of `P_A` is `paRow?` of the codes' `bf16(E′)` words (`codeBf16`) and the
row's 32 `A′·F_B` chains. -/
def PeelARow : Prop :=
  ∀ (wn : Words.FixedNoise) (side : Words.Formed) (pas : Array (Array ℕ)) (x : ℕ) (E : Fin Params.pi.r → Code),
    Words.peelA wn side = .ok pas → side.codes.size = side.ep.size → x < side.codes.size → wn.fbCols.size = 32 →
    (side.ep[x]!).size = 32 → (∀ q : Fin Params.pi.r, (side.ep[x]!)[q.val]! = (E q).val) →
    (∀ q, (E q).isNaN = false) →
      pas.size = side.codes.size ∧ (pas[x]!).size = 64 ∧ ∃ af : Fin 32 → ℕ,
        (∀ q : Fin 32, Words.chain side.codes[x]! wn.fbCols[q.val]! = .ok (af q)) ∧
          PeelExact.paRow? (List.ofFn fun q : Fin 32 => codeBf16 (E q)) (List.ofFn af) = some (pas[x]!).toList

/-- **(b) `P_B`'s rows are `pbRow?`'s**: where `peelB` is a value on a formed side whose row `j` has 32 `E′` codes `E`,
none NaN, and whose Gram words `G` are finite, row `j` of `P_B` is `pbRow?` of the codes' `bf16(E′)` words, the row's 32
`B̃·F_A` chains and `G` by rows. -/
def PeelBRow : Prop :=
  ∀ (wn : Words.FixedNoise) (side : Words.Formed) (pbs : Array (Array ℕ)) (j : ℕ) (E : Fin Params.pi.r → Code)
    (G : Fin 32 → Fin 32 → ℕ),
    Words.peelB wn side = .ok pbs → side.codes.size = side.ep.size → j < side.codes.size →
    (side.ep[j]!).size = 32 → (∀ q : Fin Params.pi.r, (side.ep[j]!)[q.val]! = (E q).val) →
    (∀ q, (E q).isNaN = false) →
    (∀ p q : Fin 32, (wn.gram[p.val]!)[q.val]! = G p q ∧ G p q < 2 ^ 32 ∧ (wordQ (G p q)).isSome) →
      pbs.size = side.codes.size ∧ (pbs[j]!).size = 64 ∧ ∃ bf : Fin 32 → ℕ,
        (∀ q : Fin 32, Words.chain side.codes[j]! wn.faCols[q.val]! = .ok (bf q)) ∧
          PeelExact.pbRow? (List.ofFn fun q : Fin 32 => codeBf16 (E q)) (List.ofFn bf)
            (List.ofFn fun p => List.ofFn (G p)) = some (pbs[j]!).toList

/-! ## The check read forward

Each step of the check that returns `.ok` is the spec's step returning `some` of the same word: `Except`'s binds come
apart one at a time (`bind_ok`), a `for` loop over a range is the spec's `foldlM` over a list of its length
(`forIn_range_sim`), and an `Array.mapM` is its function at every index (`mapM_ok`). -/

namespace WordsSim

open Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactP

section Monad

variable {ε α β γ : Type}

theorem bind_ok {x : Except ε α} {f : α → Except ε β} {r : β} (h : (x >>= f) = .ok r) :
    ∃ a, x = .ok a ∧ f a = .ok r := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem map_ok {x : Except ε α} {g : α → β} {r : β} (h : (g <$> x) = .ok r) : ∃ a, x = .ok a ∧ g a = r := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, Except.ok.inj h⟩

theorem pure_ok {a b : α} (h : (pure a : Except ε α) = .ok b) : a = b := Except.ok.inj h

theorem throw_ok {e : ε} {r : α} (h : (throw e : Except ε α) = .ok r) : False := by
  cases h

theorem listMapM_ok {f : α → Except ε β} : ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys →
    ys.length = l.length ∧ ∀ i (h : i < l.length) (h' : i < ys.length), f l[i] = .ok ys[i]
  | [], ys, h => by
    rw [List.mapM_nil] at h
    obtain rfl := pure_ok h
    exact ⟨rfl, fun i h => absurd h (Nat.not_lt_zero _)⟩
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := bind_ok h
    obtain ⟨bs, hbs, h⟩ := bind_ok h
    obtain rfl := pure_ok h
    obtain ⟨hlen, hall⟩ := listMapM_ok hbs
    refine ⟨by simp [hlen], fun i hi hi' => ?_⟩
    cases i with
    | zero => exact hb
    | succ i => exact hall i (by simp at hi; omega) (by simp at hi'; omega)

theorem arrayMapM_ok {f : α → Except ε β} {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys) :
    ys.size = xs.size ∧ ∀ i (h : i < xs.size) (h' : i < ys.size), f xs[i] = .ok ys[i] := by
  rw [Array.mapM_eq_mapM_toList] at h
  obtain ⟨l, hl, rfl⟩ := map_ok h
  obtain ⟨hlen, hall⟩ := listMapM_ok hl
  exact ⟨by simpa using hlen, fun i hi hi' => hall i (by simpa using hi) (by simpa using hi')⟩

/-- An `Array.mapM` that returns `.ok` is its function at every index. -/
theorem mapM_ok [Inhabited α] [Inhabited β] {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ i < xs.size, f xs[i]! = .ok ys[i]! := by
  obtain ⟨hs, hall⟩ := arrayMapM_ok h
  refine ⟨hs, fun i hi => ?_⟩
  rw [getElem!_pos xs i hi, getElem!_pos ys i (by omega)]
  exact hall i hi (by omega)

/-- **A `for` loop that returns `.ok` is the spec's fold**, over lists of one length, where each step of the loop's
body that returns `.ok` yields the spec's `some` at the same index. -/
theorem forIn_sim {l1 : List α} {l2 : List β} {body : α → γ → Except ε (ForInStep γ)} {g : γ → β → Option γ}
    (hlen : l1.length = l2.length)
    (hstep : ∀ i (h1 : i < l1.length) (h2 : i < l2.length) x r, body l1[i] x = .ok r →
      ∃ y, r = .yield y ∧ g x l2[i] = some y) :
    ∀ {x y : γ}, forIn l1 x body = .ok y → l2.foldlM g x = some y := by
  induction l1 generalizing l2 with
  | nil =>
    intro x y h
    cases l2 with
    | nil => rw [List.forIn_nil] at h; obtain rfl := pure_ok h; rfl
    | cons => simp at hlen
  | cons a l1 ih =>
    intro x y h
    cases l2 with
    | nil => simp at hlen
    | cons b l2 =>
      rw [List.forIn_cons] at h
      obtain ⟨r, hr, h⟩ := bind_ok h
      obtain ⟨x', rfl, hg⟩ := hstep 0 (by simp) (by simp) x r hr
      have hg' : g x b = some x' := hg
      rw [List.foldlM_cons, hg']
      exact ih (by simpa using hlen)
        (fun i h1 h2 x r hr => hstep (i + 1) (by simpa using h1) (by simpa using h2) x r hr) h

/-- `forIn_sim` over a range, its `i`-th element `start + step·i`. -/
theorem forIn_range_sim {r : Std.Legacy.Range} {l2 : List β} {body : ℕ → γ → Except ε (ForInStep γ)}
    {g : γ → β → Option γ} (hlen : r.size = l2.length)
    (hstep : ∀ i (h2 : i < l2.length) (v : ℕ), v = r.start + r.step * i → ∀ x y,
      body v x = .ok y → ∃ z, y = .yield z ∧ g x l2[i] = some z) :
    ∀ {x y : γ}, forIn r x body = .ok y → l2.foldlM g x = some y := by
  intro x y h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  exact forIn_sim (by simpa using hlen) (fun i _ h2 x r hr => hstep i h2 _ (by simp) x r hr) h

end Monad

/-! ### The check's casts are the spec's roundings -/

theorem finite_ok {w v : ℕ} (h : Words.finite w = .ok v) : v = w ∧ (w >>> 23) % 256 ≠ 255 := by
  unfold Words.finite at h
  split at h
  · exact (throw_ok h).elim
  · rename_i hne
    refine ⟨(pure_ok h).symm, ?_⟩
    have e : (w >>> 23) &&& 0xFF = (w >>> 23) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
    rw [beq_iff_eq, e] at hne
    exact hne

theorem finite?_of {w : ℕ} (h : (w >>> 23) % 256 ≠ 255) : finite? w = some w := by
  unfold finite?; rw [ite_eq_right h]

theorem wordQ_of {w : ℕ} (h : (w >>> 23) % 256 ≠ 255) : ∃ x, wordQ w = some x :=
  (PeelExactProofs.finite?_eq_some (finite?_of h)).2

/-- The check's FP32 add, where its word is finite, is the spec's `addWord?`. -/
theorem addWord_of (hadd : Fp32AddIsTheSpec) {a b : ℕ} (h : (Evaluate.Fp32.f32AddV2 a b >>> 23) % 256 ≠ 255) :
    addWord? a b = some (Evaluate.Fp32.f32AddV2 a b) := by
  obtain ⟨v, hv⟩ := wordQ_of h
  obtain ⟨ha, hb, he⟩ := hadd a b (by simp [hv])
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 ha
  obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hb
  have key : addWord? a b = finite? (TileCircuit.f32Add a b) := by
    simp only [addWord?, TileCircuit.f32Add, hx, hy]
  rw [key, ← he, finite?_of h]

theorem bf16Exp_ne {o : ℕ} (h : o % 32768 < 32640) : bf16Exp o ≠ 255 := by
  unfold bf16Exp
  rw [Nat.shiftRight_eq_div_pow, show (2 : ℕ) ^ 7 = 128 from rfl, ← Nat.mod_mul_right_div_self o 128 256]
  have : o % (128 * 256) / 128 ≤ 254 := by
    show o % 32768 / 128 ≤ 254
    omega
  omega

theorem toBf16_ok {w o : ℕ} (h : Words.toBf16 w = .ok o) :
    rnBf16? w = some o ∧ o < 2 ^ 16 ∧ bf16Exp o ≠ 255 := by
  unfold Words.toBf16 at h
  obtain ⟨w', hw, h⟩ := bind_ok h
  obtain ⟨e, hf⟩ := finite_ok hw
  rw [e] at h
  dsimp only at h
  split at h
  · exact (throw_ok h).elim
  · rename_i hlt
    obtain rfl := pure_ok h
    have e1 : (w >>> 16) &&& 1 = (w >>> 16) % 2 := Nat.and_two_pow_sub_one_eq_mod _ 1
    have e2 : ∀ y : ℕ, y &&& 0xFFFF = y % 65536 := fun y => Nat.and_two_pow_sub_one_eq_mod y 16
    have e3 : ∀ y : ℕ, y &&& 0x7FFF = y % 32768 := fun y => Nat.and_two_pow_sub_one_eq_mod y 15
    rw [e1, e2, e3] at hlt
    rw [e1, e2]
    refine ⟨?_, Nat.mod_lt _ (by norm_num), ?_⟩
    · unfold rnBf16?
      rw [finite?_of hf]
      simp only [Option.bind_some]
      rw [ite_eq_right (by omega)]
    · exact bf16Exp_ne (Nat.lt_of_not_le hlt)

theorem ofFp8_ok {c w : ℕ} (h : Words.ofFp8 c = .ok w) : c % 128 ≠ 127 ∧ w = Evaluate.Casts.e4m3ToF32 c := by
  unfold Words.ofFp8 at h
  split at h
  · exact (throw_ok h).elim
  · rename_i hne
    have e : c &&& 0x7F = c % 128 := Nat.and_two_pow_sub_one_eq_mod _ 7
    rw [beq_iff_eq, e] at hne
    exact ⟨hne, (pure_ok h).symm⟩

theorem toFp8_ok {w o : ℕ} (h : Words.toFp8 w = .ok o) :
    (w >>> 23) % 256 ≠ 255 ∧ o = Evaluate.Casts.f32ToE4m3Sat w := by
  unfold Words.toFp8 at h
  obtain ⟨w', hw, h⟩ := bind_ok h
  obtain ⟨e, hf⟩ := finite_ok hw
  rw [e] at h
  exact ⟨hf, (pure_ok h).symm⟩

/-! ### The chain -/

theorem slice_code {K : ℕ} {a : Array ℕ} {A : Fin K → Code} (ha : a.size = K)
    (hA : ∀ t (ht : t < K), a[t]! = (A ⟨t, ht⟩).val) {s n : ℕ} (hs : s = 32 * n) (hn : 32 * n + 32 ≤ K) :
    (Words.slice a s 32).map TileCircuit.code =
      List.ofFn fun t : Fin 32 => A ⟨32 * n + t, by have := t.isLt; omega⟩ := by
  subst hs
  apply List.ext_getElem
  · simp only [Words.slice, List.length_map, Array.length_toList, Array.size_extract, List.length_ofFn, ha]
    omega
  · intro t _ h2
    simp only [Words.slice, List.getElem_map, Array.getElem_toList, Array.getElem_extract, List.getElem_ofFn]
    have ht : 32 * n + t < K := by simp at h2; omega
    rw [← getElem!_pos a (32 * n + t) (by omega), hA _ ht]
    exact TileCircuit.code_val _

theorem rowAtoms_chunk {K : ℕ} (A B : Fin K → Code) (i j : ℕ)
    (hi : i < ((rowAtoms A B).toChunks 4).length) (hj : j < (((rowAtoms A B).toChunks 4)[i]'hi).length) :
    ∃ hn : 32 * (4 * i + j) + 32 ≤ K, (((rowAtoms A B).toChunks 4)[i]'hi)[j]'hj =
      (fun t : Fin 32 => A ⟨32 * (4 * i + j) + t, by have := t.isLt; omega⟩,
       fun t : Fin 32 => B ⟨32 * (4 * i + j) + t, by have := t.isLt; omega⟩) := by
  have h4 : (0 : ℕ) < 4 := by norm_num
  have hra : (rowAtoms A B).length = K / 32 := by simp [rowAtoms]
  have hj' := hj
  rw [Freeze.length_getElem_toChunks h4, hra] at hj'
  refine ⟨by omega, ?_⟩
  simp only [Freeze.getElem_toChunks h4, List.getElem_take, List.getElem_drop, rowAtoms, List.getElem_ofFn]

/-- **The chain**: the check's `chain` of two rows of codes that returns `.ok` is the spec's `chainWordP?` at the
H100 (`hopperE4m3K32`, windows of 4) of the rows' atoms. -/
theorem chain_ok (hadd : Fp32AddIsTheSpec) {K : ℕ} {a b : Array ℕ} {A B : Fin K → Code} (ha : a.size = K)
    (hb : b.size = K) (hA : ∀ t (ht : t < K), a[t]! = (A ⟨t, ht⟩).val)
    (hB : ∀ t (ht : t < K), b[t]! = (B ⟨t, ht⟩).val) {c : ℕ} (h : Words.chain a b = .ok c) :
    chainWordP? hopperE4m3K32 4 (rowAtoms A B) = some c := by
  have h4 : (0 : ℕ) < 4 := by norm_num
  unfold Words.chain at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := bind_ok h
    exact (throw_ok h1).elim
  rename_i hg
  have h32 : K % 32 = 0 := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hg
    obtain ⟨-, hg⟩ := hg
    omega
  obtain ⟨c', hfor, hc⟩ := bind_ok h
  obtain rfl := pure_ok hc
  have hra : (rowAtoms A B).length = K / 32 := by simp [rowAtoms]
  unfold chainWordP?
  refine forIn_range_sim ?_ ?_ hfor
  · rw [Freeze.length_toChunks h4, hra]
    simp only [Std.Legacy.Range.size, ha, WINDOW]
    omega
  intro i h2 ws hws x r hr
  have h2' := h2
  rw [Freeze.length_toChunks h4, hra] at h2'
  dsimp only at hws hr
  simp only [WINDOW] at hws
  obtain ⟨w, hin, hr⟩ := bind_ok hr
  obtain ⟨t', ht', hr⟩ := bind_ok hr
  obtain ⟨et, hft⟩ := finite_ok ht'
  subst et
  have hwin : windowWordP? hopperE4m3K32 (((rowAtoms A B).toChunks 4)[i]'h2) = some w := by
    unfold windowWordP?
    refine forIn_range_sim ?_ ?_ hin
    · rw [Freeze.length_getElem_toChunks h4 _ _ h2, hra]
      simp only [Std.Legacy.Range.size, ha, WINDOW]
      omega
    intro j j2 s hs v r' hr'
    dsimp only at hs hr'
    obtain ⟨v', hv', hr'⟩ := bind_ok hr'
    obtain rfl := pure_ok hr'
    refine ⟨v', rfl, ?_⟩
    obtain ⟨ev, hfv⟩ := finite_ok hv'
    subst ev
    obtain ⟨hn, hx⟩ := rowAtoms_chunk A B i j h2 j2
    have hs' : s = 32 * (4 * i + j) := by omega
    have e : hopperE4m3K32.step v (List.ofFn ((((rowAtoms A B).toChunks 4)[i]'h2)[j]'j2).1)
        (List.ofFn ((((rowAtoms A B).toChunks 4)[i]'h2)[j]'j2).2) =
        Words.atom v (Words.slice a s 32) (Words.slice b s 32) := by
      rw [hx, Words.atom, atomIsTheStep v (Words.slice a s 32) (Words.slice b s 32), slice_code ha hA hs' hn,
        slice_code hb hB hs' hn]
    exact (congrArg finite? e).trans (finite?_of hfv)
  refine ⟨if Evaluate.Fp32.f32AddV2 x w = 0x80000000 then 0 else Evaluate.Fp32.f32AddV2 x w, ?_, ?_⟩
  · split at hr
    · rename_i h0
      obtain rfl := pure_ok hr
      rw [ite_eq_left (beq_iff_eq.1 h0)]
    · rename_i h0
      obtain rfl := pure_ok hr
      rw [ite_eq_right (fun e => h0 (beq_iff_eq.2 e))]
  · show (windowWordP? hopperE4m3K32 _).bind (promoteWord? x) = _
    rw [hwin]
    show promoteWord? x w = _
    unfold promoteWord?
    rw [addWord_of hadd hft]
    rfl

/-! ### A row's bytes -/

theorem rowBytes_succ (W : Codes) (k i : ℕ) :
    NoiseSha512.rowBytes W (k + 1) i = NoiseSha512.rowBytes W k i ++ Sha512.leBytes 4 (W i k) := by
  simp [NoiseSha512.rowBytes, List.range_succ, List.flatMap_append]

theorem rowBytes_length (W : Codes) (k i : ℕ) : (NoiseSha512.rowBytes W k i).length = 4 * k := by
  induction k with
  | zero => simp [NoiseSha512.rowBytes]
  | succ k ih =>
    rw [rowBytes_succ, List.length_append, ih]
    simp only [Sha512.leBytes, List.length_map, List.length_range]
    omega

theorem rowBytes_get (W : Codes) {k i l j : ℕ} (hl : l < k) (hj : j < 4) :
    (NoiseSha512.rowBytes W k i)[4 * l + j]? = some (W i l / 256 ^ j % 256) := by
  induction k with
  | zero => omega
  | succ k ih =>
    rw [rowBytes_succ]
    by_cases h : l < k
    · rw [List.getElem?_append_left (by rw [rowBytes_length]; omega), ih h]
    · obtain rfl : l = k := by omega
      rw [List.getElem?_append_right (by rw [rowBytes_length]; omega), rowBytes_length]
      simp [Sha512.leBytes, hj]

theorem bytesNat_rowBytesOf (W : Codes) (k i : ℕ) : bytesNat (rowBytesOf W k i) = NoiseSha512.rowBytes W k i := by
  show ((NoiseSha512.rowBytes W k i).map Nat.toUInt8).toArray.toList.map UInt8.toNat = _
  rw [List.toList_toArray, List.map_map]
  conv_rhs => rw [← List.map_id (NoiseSha512.rowBytes W k i)]
  apply List.map_congr_left
  intro x hx
  have hx256 : x < 256 := by
    simp only [NoiseSha512.rowBytes, List.mem_flatMap, Sha512.leBytes, List.mem_map] at hx
    obtain ⟨_, _, _, _, rfl⟩ := hx
    exact Nat.mod_lt _ (by norm_num)
  simp [Nat.mod_eq_of_lt hx256]

theorem rowBytesOf_byte {W : Codes} {k i l j n : ℕ} (hl : l < k) (hj : j < 4) (hn : n = 4 * l + j) :
    (rowBytesOf W k i)[n]!.toNat = W i l / 256 ^ j % 256 := by
  subst hn
  have hlt : 4 * l + j < (NoiseSha512.rowBytes W k i).length := by rw [rowBytes_length]; omega
  obtain ⟨_, hget⟩ := List.getElem?_eq_some_iff.1 (rowBytes_get W (i := i) hl hj)
  have hsz : 4 * l + j < (rowBytesOf W k i).size := by
    show 4 * l + j < ((NoiseSha512.rowBytes W k i).map Nat.toUInt8).toArray.size
    simpa using hlt
  rw [getElem!_pos (rowBytesOf W k i) (4 * l + j) hsz]
  show (((NoiseSha512.rowBytes W k i).map Nat.toUInt8).toArray[4 * l + j]'(by simpa using hlt)).toNat = _
  simp [hget]

theorem or4 {b0 b1 b2 b3 : ℕ} (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) :
    b0 ||| b1 <<< 8 ||| b2 <<< 16 ||| b3 <<< 24 = b0 + b1 * 256 + b2 * 65536 + b3 * 16777216 := by
  have p8 : (2 : ℕ) ^ 8 = 256 := rfl
  have p16 : (2 : ℕ) ^ 16 = 65536 := rfl
  have p24 : (2 : ℕ) ^ 24 = 16777216 := rfl
  have l0 : b0 < 2 ^ 8 := by rw [p8]; exact h0
  rw [Nat.or_comm b0, ← Nat.shiftLeft_add_eq_or_of_lt l0]
  have l1 : b1 <<< 8 + b0 < 2 ^ 16 := by rw [Nat.shiftLeft_eq, p8, p16]; omega
  rw [Nat.or_comm _ (b2 <<< 16), ← Nat.shiftLeft_add_eq_or_of_lt l1]
  have l2 : b2 <<< 16 + (b1 <<< 8 + b0) < 2 ^ 24 := by
    rw [Nat.shiftLeft_eq, Nat.shiftLeft_eq, p8, p16, p24]; omega
  rw [Nat.or_comm _ (b3 <<< 24), ← Nat.shiftLeft_add_eq_or_of_lt l2]
  simp only [Nat.shiftLeft_eq, p8, p16, p24]
  omega

/-- **The parse**: a row opening's bytes parse to the row's words. -/
theorem parseRow_ok {W : Codes} {k i : ℕ} (hW : ∀ l < k, W i l < 2 ^ 32) {v : Array ℕ}
    (h : Words.parseRow (rowBytesOf W k i) k = .ok v) : v.size = k ∧ ∀ l < k, v[l]! = W i l := by
  unfold Words.parseRow at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := bind_ok h
    exact (throw_ok h1).elim
  obtain ⟨hsz, hall⟩ := mapM_ok h
  refine ⟨by simpa using hsz, fun l hl => ?_⟩
  have hl' := hall l (by simpa using hl)
  rw [getElem!_pos (Array.range k) l (by simpa using hl), Array.getElem_range] at hl'
  obtain ⟨e, -⟩ := finite_ok hl'
  rw [e, rowBytesOf_byte (n := 4 * l) hl (by norm_num : 0 < 4) (by omega),
    rowBytesOf_byte (n := 4 * l + 1) hl (by norm_num : 1 < 4) rfl,
    rowBytesOf_byte (n := 4 * l + 2) hl (by norm_num : 2 < 4) rfl,
    rowBytesOf_byte (n := 4 * l + 3) hl (by norm_num : 3 < 4) rfl,
    or4 (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num))]
  have := hW l hl
  simp only [pow_zero, pow_one, Nat.div_one, show (256 : ℕ) ^ 2 = 65536 from rfl,
    show (256 : ℕ) ^ 3 = 16777216 from rfl]
  omega

end WordsSim

end Pouw.PearlPlus.Bridge
