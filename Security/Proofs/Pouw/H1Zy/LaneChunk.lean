import Definitions.Pouw.H1Zy.LaneChunk
import Proofs.Pouw.H1Zy.Lane

/-!
# A lane's chunks, accepted at one carry chain, check the whole lane (`laneOk_of_chunks`)

`n` chunk statements of a lane (`Pouw.H1Zy.chunkOk`, `PouwH1LaneZyChunk_v1{TILE, n, c}`), each `ok = 1`, chunk `c + 1`'s
carry in the row chunk `c`'s carry out is, check the whole lane: `laneOk` at the chain machine's words (`chainOf`) on the
lane the chunks read (`joinIn`: each row's codes the chunks' concatenated, `joinRows`, its scales chunk 0's, and the last
chunk's record, partials, public inputs and index). The chain lemma (`run_join`, `zRun_join`): the whole `K`'s segments
are the chunks' in sequence, so the chain's state and each cell's z after chunk `c` are the whole lane's after its first
`(c + 1) K_c / 32` segments. An induction over `c` (`chunkCarry_eq`) pins each carry in, through the chunk before's
carry-out check, to that state, and each chunk's scales, through its scale check, to chunk 0's. So the last chunk's
checks are the lane's on the joined rows, at `z = gemmAccS8` over the whole `K` (`zRun_zero`), which `laneOk_spec`
(with `gemmAccS8_eq`, at `32 ∣ K` and `K < 2^17`) makes the exact dot.
-/

namespace Pouw.H1Zy

/-- **A lane's whole row from its chunks' rows**: entry `k` is entry `k mod K_c` of chunk `⌊k / K_c⌋` (`SPLITK`'s chunk
row `n i + c` holding entries `[c K_c, (c + 1) K_c)` of row `i`). -/
def joinRows {n Kc : ℕ} (x : Fin n → Fin Kc → BitVec 8) : Fin (n * Kc) → BitVec 8 := fun k =>
  have hK : 0 < Kc := Nat.pos_of_ne_zero fun h => by have := k.isLt; simp [h] at this
  x ⟨k.val / Kc, (Nat.div_lt_iff_lt_mul hK).2 k.isLt⟩ ⟨k.val % Kc, Nat.mod_lt _ hK⟩

/-- The last of `n > 0` chunks. -/
def lastFin (n : ℕ) (hn : 0 < n) : Fin n := ⟨n - 1, by omega⟩

/-- **The lane `n` chunks read**: each A and B row's codes the chunks' joined (`joinRows`), its scales chunk 0's, and the
last chunk's admit bits, record, partials, public inputs and index. -/
def joinIn {n Kc nw : ℕ} {P : Type} (xs : Fin n → ChunkIn Kc nw P) (hn : 0 < n) : LaneIn (n * Kc) nw P where
  a := fun s => joinRows fun c => (xs c).a s
  sa := (xs ⟨0, hn⟩).sa
  b := fun t => joinRows fun c => (xs c).b t
  sw := (xs ⟨0, hn⟩).sw
  admit := (xs (lastFin n hn)).admit
  zr := (xs (lastFin n hn)).zr
  yr := (xs (lastFin n hn)).yr
  partials := (xs (lastFin n hn)).partials
  pub := (xs (lastFin n hn)).pub
  lane := (xs (lastFin n hn)).lane

/-! ## The chain lemma -/

/-- Segment `c K_c / 32 + j` of a joined row is segment `j` of chunk `c`'s row. -/
theorem segOf_joinRows {n Kc : ℕ} (x : Fin n → Fin Kc → BitVec 8) (hKc : 32 ∣ Kc) (c : Fin n) {j : ℕ}
    (hj : j < Kc / 32) : segOf (joinRows x) (c.val * (Kc / 32) + j) = segOf (x c) j := by
  obtain ⟨q, rfl⟩ := hKc
  have hq : 32 * q / 32 = q := by omega
  rw [hq] at hj ⊢
  funext i
  have hi := i.isLt
  have hc := c.isLt
  have hlt' : 32 * j + i.val < 32 * q := by omega
  have key : 32 * (c.val * q + j) + i.val = (32 * j + i.val) + 32 * q * c.val := by ring
  have hlt : 32 * (c.val * q + j) + i.val < n * (32 * q) := by
    rw [key]
    have : c.val + 1 ≤ n := hc
    nlinarith
  have hpos : 0 < 32 * q := by omega
  simp only [segOf, codeAt, hlt, hlt', ↓reduceDIte, joinRows]
  congr 1
  · ext
    simp only [key, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hlt', zero_add]
  · ext
    simp only [key, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt']

/-- **The chain lemma**: the chain's state after `j` segments of chunk `c`, from the whole lane's state after its first
`c K_c / 32` segments, is the whole lane's after `c K_c / 32 + j`. -/
theorem run_join {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (p : P) (l : Fin 32) (hKc : 32 ∣ Kc)
    (a : Fin n → Fin 4 → Fin Kc → BitVec 8) (b : Fin n → Fin 8 → Fin Kc → BitVec 8) (σ : ChainSt) (c : Fin n) :
    ∀ j, j ≤ Kc / 32 →
      M.run p l (c.val * (Kc / 32)) (a c) (b c)
          (M.run p l 0 (fun s => joinRows fun d => a d s) (fun t => joinRows fun d => b d t) σ (c.val * (Kc / 32))) j =
        M.run p l 0 (fun s => joinRows fun d => a d s) (fun t => joinRows fun d => b d t) σ (c.val * (Kc / 32) + j)
  | 0, _ => rfl
  | j + 1, hj => by
    rw [← Nat.add_assoc]
    simp only [ChainSteps.run]
    rw [run_join M p l hKc a b σ c j (by omega), zero_add]
    simp only [segOf_joinRows _ hKc c (by omega : j < Kc / 32)]

/-- The same for a cell's z: `j` steps of chunk `c` from the whole lane's z after `c K_c / 32` steps. -/
theorem zRun_join {n Kc : ℕ} (x w : Fin n → Fin Kc → BitVec 8) (hKc : 32 ∣ Kc) (z : BitVec 32) (c : Fin n) :
    ∀ j, j ≤ Kc / 32 →
      zRun (x c) (w c) (zRun (joinRows x) (joinRows w) z (c.val * (Kc / 32))) j =
        zRun (joinRows x) (joinRows w) z (c.val * (Kc / 32) + j)
  | 0, _ => rfl
  | j + 1, hj => by
    rw [← Nat.add_assoc]
    simp only [zRun]
    rw [zRun_join x w hKc z c j (by omega), segOf_joinRows x hKc c (by omega), segOf_joinRows w hKc c (by omega)]

/-- From zero, `zRun` is `GemmAccumulatorS8`'s steps. -/
theorem zRun_zero {K : ℕ} (x w : Fin K → BitVec 8) : ∀ m, zRun x w 0 m = gemmAccSteps x w m
  | 0 => rfl
  | m + 1 => by simp only [zRun, gemmAccSteps, zRun_zero x w m]; rfl

/-! ## One chunk's checks -/

theorem mux_false_eq_true {b t : Bool} : mux b t false = true ↔ b = false ∧ t = true := by
  cases b <;> simp [mux]

theorem ChunkIn.l_eq {Kc nw : ℕ} {P : Type} {x y : ChunkIn Kc nw P} (h : x.lane = y.lane) : x.l = y.l :=
  Fin.ext (by simp only [ChunkIn.l, ChunkIn.sel, h])

/-- **An accepted chunk's checks**: no high index bit, past chunk 0 its scales the carried ones, and before the last chunk
its carry out what it computed. -/
theorem chunkOk_parts {Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16)
    {n c : ℕ} {x : ChunkIn Kc nw P} (h : chunkOk M dqW n c x = true) :
    tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩) = false ∧
      (c ≠ 0 → x.sa = x.carryIn.sa ∧ x.sw = x.carryIn.sw) ∧
      (c + 1 ≠ n → x.carryOut = chunkCarry M c x) := by
  simp only [chunkOk] at h
  obtain ⟨hh, ht⟩ := mux_false_eq_true.1 h
  rw [tree_andBit] at ht
  refine ⟨hh, fun hc => ?_, fun hc => ?_⟩
  · simp only [hc, ↓reduceIte] at ht
    constructor
    · funext s
      have := ht (eq32 (x.sa s) (x.carryIn.sa s)) (List.mem_append_left _ (List.mem_append_left _ (List.mem_ofFn.2 ⟨s, rfl⟩)))
      simpa [eq32] using this
    · funext t
      have := ht (eq32 (x.sw t) (x.carryIn.sw t)) (List.mem_append_left _ (List.mem_append_right _ (List.mem_ofFn.2 ⟨t, rfl⟩)))
      simpa [eq32] using this
  · simp only [hc, ↓reduceIte] at ht
    simpa using ht _ (List.mem_append_right _ (List.mem_singleton_self _))

/-! ## The induction over chunks -/

/-- **Each chunk computes the whole lane's state**: at accepted chunks linked by their carries, chunk `c` computes the
chain's state and each cell's z after the whole lane's first `(c + 1) K_c / 32` segments, with chunk 0's scales. -/
theorem chunkCarry_eq {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (hn : 0 < n) (hKc : 32 ∣ Kc)
    (xs : Fin n → ChunkIn Kc nw P)
    (hpub : ∀ c, (xs c).pub = (xs (lastFin n hn)).pub) (hlane : ∀ c, (xs c).lane = (xs (lastFin n hn)).lane)
    (hcarry : ∀ (c : ℕ) (h : c + 1 < n), (xs ⟨c + 1, h⟩).carryIn = (xs ⟨c, by omega⟩).carryOut)
    (hok : ∀ c : Fin n, chunkOk M dqW n c.val (xs c) = true) :
    ∀ (c : ℕ) (hc : c < n), chunkCarry M c (xs ⟨c, hc⟩) =
      { chain := M.run (xs (lastFin n hn)).pub (xs (lastFin n hn)).l 0 (joinIn xs hn).a (joinIn xs hn).b
          (M.init (xs (lastFin n hn)).pub (xs (lastFin n hn)).l) ((c + 1) * (Kc / 32))
        z := fun s t => zRun ((joinIn xs hn).a s) ((joinIn xs hn).b t) 0 ((c + 1) * (Kc / 32))
        sa := (xs ⟨0, hn⟩).sa
        sw := (xs ⟨0, hn⟩).sw } := by
  intro c
  induction c with
  | zero =>
    intro hc
    have hp := hpub ⟨0, hc⟩
    have hl := ChunkIn.l_eq (hlane ⟨0, hc⟩)
    have hr := run_join M (xs (lastFin n hn)).pub (xs (lastFin n hn)).l hKc (fun d => (xs d).a) (fun d => (xs d).b)
      (M.init (xs (lastFin n hn)).pub (xs (lastFin n hn)).l) ⟨0, hc⟩ (Kc / 32) le_rfl
    simp only [zero_mul, ChainSteps.run, zero_add] at hr
    simp only [chunkCarry, ↓reduceIte, zero_mul, zero_add, one_mul, hp, hl, Carry.mk.injEq, and_true]
    refine ⟨hr, ?_⟩
    funext s t
    have hz := zRun_join (fun d => (xs d).a s) (fun d => (xs d).b t) hKc 0 ⟨0, hc⟩ (Kc / 32) le_rfl
    simp only [zero_mul, zRun, zero_add] at hz
    exact hz
  | succ c ih =>
    intro hc
    have hc' : c < n := by omega
    have ihc := ih hc'
    have hprev := (chunkOk_parts M dqW (hok ⟨c, hc'⟩)).2.2 (show c + 1 ≠ n by omega)
    have hin : (xs ⟨c + 1, hc⟩).carryIn = _ := (hcarry c hc).trans (hprev.trans ihc)
    have hsc := (chunkOk_parts M dqW (hok ⟨c + 1, hc⟩)).2.1 (show c + 1 ≠ 0 by omega)
    have hp := hpub ⟨c + 1, hc⟩
    have hl := ChunkIn.l_eq (hlane ⟨c + 1, hc⟩)
    have hr := run_join M (xs (lastFin n hn)).pub (xs (lastFin n hn)).l hKc (fun d => (xs d).a) (fun d => (xs d).b)
      (M.init (xs (lastFin n hn)).pub (xs (lastFin n hn)).l) ⟨c + 1, hc⟩ (Kc / 32) le_rfl
    have hk : (c + 1) * (Kc / 32) + Kc / 32 = (c + 1 + 1) * (Kc / 32) := by ring
    rw [hk] at hr
    simp only [chunkCarry, Nat.add_one_ne_zero, ↓reduceIte, hp, hl, hin, Carry.mk.injEq]
    refine ⟨hr, ?_, ?_, ?_⟩
    · funext s t
      have hz := zRun_join (fun d => (xs d).a s) (fun d => (xs d).b t) hKc 0 ⟨c + 1, hc⟩ (Kc / 32) le_rfl
      rw [hk] at hz
      exact hz
    · rw [hsc.1, hin]
    · rw [hsc.2, hin]

/-- **A lane's chunks, accepted at one carry chain, check the whole lane.** `n > 0` chunk statements of one lane, each
`ok = 1` at the chain machine `M` and the dequant `dqW`, with one public input and index, and chunk `c + 1`'s carry in
chunk `c`'s carry out, give the whole lane's `ok = 1` at `M`'s words on the lane they read (`joinIn`). -/
theorem laneOk_of_chunks {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (hn : 0 < n) (hKc : 32 ∣ Kc)
    (xs : Fin n → ChunkIn Kc nw P)
    (hpub : ∀ c, (xs c).pub = (xs (lastFin n hn)).pub) (hlane : ∀ c, (xs c).lane = (xs (lastFin n hn)).lane)
    (hcarry : ∀ (c : ℕ) (h : c + 1 < n), (xs ⟨c + 1, h⟩).carryIn = (xs ⟨c, by omega⟩).carryOut)
    (hok : ∀ c : Fin n, chunkOk M dqW n c.val (xs c) = true) :
    laneOk (chainOf M) dqW (joinIn xs hn) = true := by
  have hlast := chunkCarry_eq M dqW hn hKc xs hpub hlane hcarry hok (n - 1) (by omega)
  have hn1 : n - 1 + 1 = n := by omega
  have hks : (n * Kc) / 32 = n * (Kc / 32) := Nat.mul_div_assoc n hKc
  have h := hok (lastFin n hn)
  simp only [lastFin] at h hlast
  simp only [chunkOk, hn1, ↓reduceIte] at h
  obtain ⟨hh, ht⟩ := mux_false_eq_true.1 h
  rw [tree_andBit] at ht
  rw [hlast] at ht
  simp only [hn1] at ht
  have hsa : (xs ⟨n - 1, by omega⟩).sa = (xs ⟨0, hn⟩).sa := by
    have := congrArg Carry.sa (chunkCarry_eq M dqW hn hKc xs hpub hlane hcarry hok (n - 1) (by omega))
    simpa [chunkCarry] using this
  have hsw : (xs ⟨n - 1, by omega⟩).sw = (xs ⟨0, hn⟩).sw := by
    have := congrArg Carry.sw (chunkCarry_eq M dqW hn hKc xs hpub hlane hcarry hok (n - 1) (by omega))
    simpa [chunkCarry] using this
  rw [hsa, hsw] at ht
  simp only [laneOk, chainOf, hks]
  refine mux_false_eq_true.2 ⟨hh, ?_⟩
  rw [tree_andBit]
  intro b hb
  apply ht
  refine List.mem_append_right _ ?_
  simpa only [zRun_zero, gemmAccS8, hks, joinIn, lastFin, LaneIn.sel, ChunkIn.sel, LaneIn.l, ChunkIn.l] using hb

/-- A chunk binds where its lane does: at `n > 0` chunks of `K_c` entries, `n K_c < 2^17` gives `K_c < 2^17`, the lane
field's range at every chunk. -/
theorem chunk_lt {n Kc : ℕ} (hn : 0 < n) (hK : n * Kc < 2 ^ 17) : Kc < 2 ^ 17 :=
  lt_of_le_of_lt (Nat.le_mul_of_pos_left Kc hn) hK

/-- **What a lane's accepted chunks give** (`laneOk_spec` at `laneOk_of_chunks`), at `K = n K_c` with `32 ∣ K_c` and
`K < 2^17`: the last chunk's index has no high bit; each of the lane's cells has record z the s32 word of the exact dot
of its joined A and B rows' codes over the whole `K` and record y `dqW` of it at chunk 0's scales; and the last chunk's
partials row is the chain machine's words on the joined rows, masked by the admit bits' OR. -/
theorem chunks_spec {n Kc nw : ℕ} {P : Type} (M : ChainSteps P nw)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (hn : 0 < n) (hKc : 32 ∣ Kc) (hK : n * Kc < 2 ^ 17)
    (xs : Fin n → ChunkIn Kc nw P)
    (hpub : ∀ c, (xs c).pub = (xs (lastFin n hn)).pub) (hlane : ∀ c, (xs c).lane = (xs (lastFin n hn)).lane)
    (hcarry : ∀ (c : ℕ) (h : c + 1 < n), (xs ⟨c + 1, h⟩).carryIn = (xs ⟨c, by omega⟩).carryOut)
    (hok : ∀ c : Fin n, chunkOk M dqW n c.val (xs c) = true) :
    let y := joinIn xs hn
    (∀ t : Fin 11, y.lane ⟨5 + t.val, by omega⟩ = false) ∧
    (∀ s t, y.zr (laneRow y.l s) (laneCol y.l t) = BitVec.ofInt 32 (∑ k, (y.a s k).toInt * (y.b t k).toInt) ∧
      y.yr (laneRow y.l s) (laneCol y.l t) =
        dqW (BitVec.ofInt 32 (∑ k, (y.a s k).toInt * (y.b t k).toInt)) (y.sa s) (y.sw t)) ∧
    ∀ w, y.partials w = mask32 (chainOf M y.pub y.l y.a y.b w) (tree orBit false (List.ofFn y.admit)) :=
  laneOk_spec (chainOf M) dqW (joinIn xs hn) (Dvd.dvd.mul_left hKc n) hK
    (laneOk_of_chunks M dqW hn hKc xs hpub hlane hcarry hok)

end Pouw.H1Zy
