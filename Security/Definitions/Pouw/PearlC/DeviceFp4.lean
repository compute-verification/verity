import Definitions.Pouw.PearlC.Tile
import Definitions.Pouw.PearlC.Fp4Chain
import Definitions.Pouw.PearlC.Fp4Skip
import Definitions.Pouw.PearlC.PeelExact

/-!
# Pearl-C4's device record: NVFP4 on sm_120 (trusted; definitions only; staged, nothing pinned)

`pearl-c-nvfp4-v0` (Pearl-C4's scheme, `pearl_c4.py`), as Pearl-C4's domain theory states it (§2, §4). `PearlCDevice`'s
maps read E4M3 codes, 32-MAC atoms and promotion groups, so the FP4 line gets its own record, `Fp4Device`, with the
same fields and no promotion (`G = 0`):

* **The forming** (`Fp4Dev.formRow`, `pearl_c4.form`): per element the noise atom `Fp4.nvf4NoiseElem`, then NVFP4's block
  rule over blocks of 16 along `k`: `s_b = max(ue4m3(f32(amax_b|D|·f32(1/6))), 2⁻⁹)`, `r_b = f32(1/s_b)` and
  `code = e2m1(f32(D·r_b))`, with `e2m1` the saturating nearest-even cast (`Fp4Dev.e2m1Word`).
* **The ticket** (`Fp4Dev.ticket4`): `C̃ = Fp4.cTilde`, the atom chained from `+0` over `k/64` firings, no promotion.
* **U** (`Fp4Dev.peelFp4`, `peel_factors_a`, `peel_factors_b`, `peel`): Pearl-C4's peel factors use the FP4 atom, so U's
  twin changes in its chain words only. `C̃`, `A′·F_B`, `B̃·F_A` and the Gram matrix `F_Bᵀ·F_A` are `cTilde` chains, with
  `F`'s columns at scale `1.0`. The noise rows are `Ẽ = bf16(|E|·a_E·2⁻⁴)` with `E`'s sign (`Fp4Dev.eTilde`). `P_A`, `P_B`
  with its FP32 FMA chain, and the four BF16 K16 steps (`HOPPER_BF16_M16N8K16`, `PeelExact.k16Step`) are `PeelExact`'s
  `paRow?`, `pbRow?` and `uWord?` as they stand.
* **The flags** (`Fp4Skip.fp4PureFlags`, debit (2)): firing `t` of word `(i, j)`'s chain from `+0` is flagged when the
  chain is in the model and its lone skip leaves the word unchanged (`debit_pure` at the FP4 atom; soundness
  `Fp4Freeze.fp4PureFlags_iff`); `false` where a row's forming left the model.
* **The salt-dead predicate** (`Fp4Dev.saltDead4`): of `PearlCSem.saltDead`'s type, on the E2M1 code's value under
  `formRow`, over every noise draw. The ρ_D bound on it is `Fp4Dev.RhoDFp4`, a named `Prop` that
  `CodeProofs.rhoDFp4At_ninety` proves from `CodeProofs.PinnedScales` (argued, not proved, for `pearl_c4.stats`).
* **The row rule** (`Fp4Dev.RowAdmit4`, salt-free): D-NF (`0 ≤ β`, and `a_E = e4m3(β)` with `0x08 ≤ a_E < 0x80`, a
  normal UE4M3 with bit 7 clear), which implies D-SS's scale clauses.
  D-SS's one-in-64 dead-block cap is no admission rule: terms (3) and (4) charge a row's
  salt-dead codes per element instead. The screen `x² > 90·ρ²`, about 9.49ρ (`Fp4Dev.screenDead`, `DEAD_RHO_SQ = 90`),
  covers every salt-dead element at the pinned scales (`CodeProofs.saltDead4_screened`), and `Fp4Dev.RowRules4`, D-NF
  with the cap, is kept as `RhoDFp4`'s premise.
* **D-SK, B̃ keyed by A's salt** (`Fp4Sem.SaltKeyedB`, theory §1): `seed_B = H_"pearl-c4/v1/seed-B"(salt ‖ root_B)`
  (SHA-512 of the label block and the message, its first 32 bytes; `pearl_c4.labelled`), and `B̃ = form(B, E_B, F_B)`
  with `E_B`, `F_B` drawn from `seed_B`. The verifier forms `B̃` itself on every drawn tile and rejects when the checked
  values differ. It is a condition of the domain.
* **The debit** (`Fp4Sem.tileDebit`, §2 with §6's base-split fix, replayed over a tile's credited words, in FP4
  units). Its executable reference is `pearl_c4.tile_debit` (`9363e5012`), and each term carries the
  reference's name for it:
  - (0) D-24's 2:4 windows (`windows`), 64 per (row, aligned 128 of `k`, word the row feeds), for A′'s rows and B̃'s. A
    window is 2:4 by the hardware's pair rule: of each aligned 8-chunk's four pairs, at most two hold a nonzero code
    (`Fp4Dev.rowWin24`; sm_120's `kind::mxf4nvf4.sp` metadata picks 2 of 4 pairs). As in the reference's
    `two_four_windows`, it is charged per row window, not per 16-row span, since a cheaper computation may gather any 16
    such rows;
  - (1) **F1′** (§6.2, `split`), in place of §2's 2:4 term on salt-dead codes: per aligned row window of 128, the base
    split's expected saving less `A0`'s scale corrections, `128·min(1, ovf·min(1, E[max(0, R_24 − κ)] +
    E[max(0, R_skip − κ)]))` per word the row feeds (`Fp4Dev.f1Row`), under the exact noise law given the row's `E`
    line, with `p_t = 0` on salt-dead codes. The overfit `ovf` is 1 on A′'s rows and `F1_B_OVERFIT = 10` on the salted
    B̃'s (`Fp4Dev.f1BOverfit`);
  - (2) the flagged firings at 64 per atom-word (`identity`);
  - (3) the dead pairs (`dead_pairs`): per word and position whose A′ code and B̃ code are both salt-dead, 1;
  - (4) the forming (`forming`): the A′ elements of the rows that are salt-dead or unneeded (every B̃ row of the tile
    has a zero code at the position), at `fs` times the tile's column share;
  - (5) the sub-grid blocks (`subgrid`; D-SS's T1-a clause, recommended on every row) at 16 per (word, block): a block
    whose largest possible scaled dot lies below the grid of the replayed firing that holds it (`Fp4Dev.subgridFlags4`);
  - (6) **F2** (§6.3, `int8`): the masked int8 route's saving per chain MAC of the tile, at the largest of three readings
    of the realised scale bytes' flatness: the tile's modal byte, each row's own modal byte, and per block-column class
    (`Fp4Dev.int8Saving`, `int8_saving`), with `c_L` the fast-matmul catalogue's minimum, pinned per shape
    (`Fp4Dev.cL`, the reference's `pearl_c4_c_L`);
  - (7) **R1** (§6.8): a row whose own F1′ per word (B̃'s at its overfit) exceeds `ρ` times its cell credit is charged
    its whole cell credit.
* **Exact where the reference bounds.** Salt-deadness is `saltDead4`, over every draw, where the reference's
  `salt_dead` tries three draws and counts the rest as dead; and F1′'s modal byte and `p_t` are exact (`modalByte`,
  `leaveProb`), where its `change_probabilities` certifies the byte and takes a lower bound on `p_t`. The reference
  makes each choice on the side that charges more (its `change_probabilities_exact` and `salt_dead` docstrings), so it
  charges at least this debit once its `two_four_windows` reads the pair rule (term (0)).
* **The base-split fix** (§6.2–6.5, with its FP4 conditions). It adds terms (1), (6) and (7), and changes no forming,
  credit, `fs`, `qa` or `W_ref`. A tile whose credited rows' debit exceeds `ρ` (`1/400`, §6.6) is rejected by the cap,
  not debited: `pearlCTilesFp4`'s `capOK`, and `pearlCProtocolFp4`'s per unit.
  - *Replay (protocol side).* The verifier replays F1′, F2 and R1 from the drawn tile's own forming: its `E` rows, its
    realised scale bytes, and the clean rows and scales that the law given `E` reads. The sampled-tile budget prices
    F1′'s pass, about one chain replay per tile.
  - *Operands.* This is the plain protocol: F1′ on A′ and on the salted B̃ (D-SK), and F2 on both. The fork
    (`tt-out-aw/pearl-c-nvfp4-sm120`, §6.7) takes F1′ on A′ only, the registered B being covered at registration
    (`structure-free/rot-nvfp4` and the registration uncredit) with F2 per tile on both sides; it is not stated here.
  The unit's debit is the tile debit over all its words. None of the terms assumes the chain is exact. Any claim that
  equates `C̃` with the exact product carries a per-word exact-run condition
  (`PeelFp4Proofs.ExactRuns4`, through `Fp4Freeze.cTilde_dot`), because D₄ gives none (`Fp4Freeze.noLoss_gap`). The
  peel reads the chain's own words, so it needs nothing.
* **The accounting** (`creditFp4`, `wrefFp4`): per `m·k`, `n` (the chain, 1 per MAC) `+ 32` (`A′·F_B`, `r = 32`)
  `+ 2·bf16·32·n/k` (the BF16 peel, depth `2r`) `+ fs`. `W_ref` adds `qa` per element.
* **The protocol, domain and tiles** (`pearlCProtocolFp4`, `pearlCDomainFp4`, `pearlCTilesFp4`) are
  `pearlCProtocolDev`'s, `pearlCDomainDev`'s at `G = 0` and `pearlCTilesDev`'s shapes, on these maps. The domain is
  the reference's shapes (`PearlC4.check`) with `n ≥ 4096` in place of its `n ≥ 128`. It also requires D-SK of
  `sem`'s lines, so it is empty at a `sem` whose `B̃` is not keyed by A's salt.

`Fp4Sem` holds what is not pinned yet, as `PearlCSem` does for FP8: the noise lines from the oracle and the rows'
scales `α` and `β`. `devFp4Sm120` is the record at `Fp4.sm120Nvf4` with the §3/§4 prices (`Fp4Prices.sm120`).
-/

namespace Pouw.PearlC.Fp4Dev

open Finset Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4

/-! ## Formed operands -/

/-- One formed FP4 row: its E2M1 codes along `k` and its scale bytes, one per block. -/
abbrev Row4 : Type := List Nib × List (Fin 256)

/-- A formed operand of `p` rows: each row's codes and scale bytes, or `none` where its forming left the model. -/
abbrev Op4 (p : ℕ) : Type := Fin p → Option Row4

/-! ## NVFP4's noisy forming (`pearl_c4.form`) -/

/-- `f32(1/6)`, the word `0x3E2AAAAB` (`SIXTH`). -/
def sixth : ℚ := wordVal 0x3E2AAAAB

/-- The E2M1 grid's midpoints `0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5` (`_E2M1_MIDPOINTS`). -/
def e2m1Mids : List ℚ := [1 / 4, 3 / 4, 5 / 4, 7 / 4, 5 / 2, 7 / 2, 5]

/-- The magnitude code of `cvt.rn.satfinite.e2m1` at `v ≥ 0`: the midpoints `v` passes in order, a tie passing when
the code above it is even (nearest even); `7` (the value 6) from `5` up. -/
def e2m1Mag (v : ℚ) : ℕ :=
  ((List.range 7).takeWhile fun i =>
    decide (e2m1Mids.getD i 0 < v) || (decide (e2m1Mids.getD i 0 = v) && (i + 1) % 2 == 0)).length

/-- **`e2m1`** (`cvt.rn.satfinite.e2m1`) of a finite FP32 word: its sign bit, then `e2m1Mag` of its magnitude. So `−0`,
and a negative value that rounds to zero, are `0x8`. -/
def e2m1Word (w : ℕ) : Nib :=
  ⟨((if (w >>> 31) % 2 = 1 then 8 else 0) + e2m1Mag |wordVal w|) % 16, Nat.mod_lt _ (by decide)⟩

/-- NVFP4's block scale byte (`block_scale`): `max(ue4m3(f32(amax·f32(1/6))), 0x01)`. Here `ue4m3` is the E4M3 cast on
its sign-free half (`e4m3` of a non-negative value), and `0x01` is `2⁻⁹`. -/
def nvf4Scale (amax : ℚ) : Fin 256 := max (e4m3 (wordVal (f32Word (amax * sixth)))) 1

/-- The block's reciprocal `r_b = fp32.div(1, s_b)`: `1/s_b` rounded to FP32, nearest even. -/
def nvf4Recip (s : Fin 256) : ℚ := wordVal (f32Word (1 / codeVal s))

/-- One block's codes and scale byte from its noise-atom words `ds`: `s_b` from the block's largest magnitude, and
`code_t = e2m1(fp32.mul(D_t, r_b))`. The product of `−0` and `r_b > 0` is `−0`, so a `−0` word's code is `0x8`. -/
def nvf4Block (ds : List ℕ) : List Nib × Fin 256 :=
  let s := nvf4Scale ((ds.map fun w => |wordVal w|).foldl max 0)
  (ds.map fun w => e2m1Word (if w = 0x80000000 then w else f32Word (wordVal w * nvf4Recip s)), s)

/-- **One row's noisy NVFP4 forming** (`pearl_c4.form` on one row): element `t`'s noise atom
`nvf4NoiseElem α (x t) β e (F t)`, that is `D_t = NVF4.step(f32(α·x_t), E ‖ 0³² at e4m3(β), F_t ‖ 0³² at 2⁻⁴)`, then
`nvf4Block` on each block of 16 along `k`. It is `none` where a noise atom is outside the model. -/
def formRow {k : ℕ} (α β : ℚ) (x : Fin k → ℚ) (e : Fin 32 → Nib) (F : Fin k → Fin 32 → Nib) : Option Row4 := do
  let ds ← (List.ofFn fun t => nvf4NoiseElem α (x t) β e (F t)).mapM id
  let bs := (ds.toChunks 16).map nvf4Block
  pure ((bs.map Prod.fst).flatten, bs.map Prod.snd)

/-! ## The salt-dead predicate, the row rules and ρ_D -/

/-- **P3's salt-dead predicate at the FP4 forming**, of `PearlCSem.saltDead`'s type. Element `l` of a committed row `x`
of length `k` is salt-dead when its E2M1 code under `formRow`, at the row's scales `alpha k x` and `beta k x`, has the
same value at every noise draw. A draw is an `E` line `e` and a basis `F` (one line per position), all uniform E2M1
lines, so the support is every pair. It is on the code's value (`Fp4.decode`), so `0x0` and `0x8` count alike; a draw
whose forming leaves the model has the value `none`. -/
def saltDead4 (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) : (k : ℕ) → (Fin k → ℚ) → Fin k → Prop :=
  fun k x l => ∀ (e e' : Fin 32 → Nib) (F F' : Fin k → Fin 32 → Nib),
    (formRow (alpha k x) (beta k x) x e F).map (fun r => Fp4.decode (r.1.getD l 0)) =
      (formRow (alpha k x) (beta k x) x e' F').map (fun r => Fp4.decode (r.1.getD l 0))

/-- D-SS's dead block by the salt-free screen (`dead_blocks`): block `b` of 16 along `k` has an element with
`x² > 90·ρ²`, that is `|x| > √90·ρ ≈ 9.49ρ`, `ρ²` over every 8th position (`Stat.every8`). The constant is `90`
exactly, not `9.5² = 90.25`: at it every salt-dead element lies in a screened block at the pinned scales
(`CodeProofs.saltDead4_screened`, any screen of at most `90·ρ²`). The earlier 12ρ screen (`TAU_SQ_INV = 144`) and
10ρ are superseded. The debit doesn't read it (the reference counts screened blocks, `screened_block_pairs`, and charges
none); it is `RowRules4`'s cap. -/
def screenDead {k : ℕ} (x : Fin k → ℚ) (b : ℕ) : Prop :=
  ∃ l : Fin k, (l : ℕ) / 16 = b ∧ 90 * Stat.every8 k x < x l ^ 2

/-- **Pearl-C4's row rule** (salt-free): D-NF, `0 ≤ β` and the row's noise scale byte
`a_E = e4m3(β)` a normal UE4M3 with bit 7 clear, `0x08 ≤ a_E < 0x80` (at least `2⁻⁶`).
- *The two β clauses* (`pearl_c4.dnf` at `e7432087`). A negative β casts to a byte with bit 7 set, which `0x08 ≤ a_E`
  alone admits and sm_120 reads from its low 7 bits: at `β = −1/4096` the byte `0x80` is a zero scale, so the row's
  noise would be zero, and at `β = −1` the byte `0xB8` reads as `+1.0` (`dnf_rejects`). The pinned stats never make β
  negative, so no honest row changes verdict.
- D-SS's scale clauses follow from D-NF and the design's floor, so they add no condition. D-SS's one-in-64 dead-block
  cap is no longer an admission rule. The debit charges a row's salt-dead codes per element instead
  (`Fp4Sem.tileDebit`, terms (3) and (4)), and the tile's cap rejects a tile that this pushes past `ρ`. -/
def RowAdmit4 (beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) : Prop :=
  0 ≤ beta k x ∧ 8 ≤ (e4m3 (beta k x)).val ∧ (e4m3 (beta k x)).val < 128

open Classical in
/-- **`RhoDFp4`'s row premise** (the earlier `row_passes`): D-NF as `RowAdmit4` states it, and D-SS's cap,
at most one block in 64 dead by the screen at `90·ρ²`. It is not the admission rule (`RowAdmit4` is). It keeps the cap
so that `RhoDFp4` is `CodeProofs.RhoDFp4At 90` by `rfl`: `CodeProofs.RowRules4At` states D-NF in the same form. -/
def RowRules4 (beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) : Prop :=
  0 ≤ beta k x ∧ 8 ≤ (e4m3 (beta k x)).val ∧ (e4m3 (beta k x)).val < 128 ∧
    64 * ((range (k / 16)).filter fun b => screenDead x b).card ≤ k / 16

open Classical in
/-- **ρ_D = 1/64 at the FP4 forming**: at the row scales `alpha` and `beta`, on any operand of `p` rows of length `k`
(a multiple of 128, as in the domain) whose rows pass `RowRules4`, at most `p·k/64` elements are salt-dead under
`saltDead4`. It is a named `Prop`, taken as a hypothesis where it is used, and it holds at the pinned scales, not at
every `alpha` and `beta`.
- *What it bounds now.* Admission is `RowAdmit4`, so `RhoDFp4` bounds the salt-dead share of the rows that meet the
  old cap. The debit doesn't take it: terms (3) and (4) charge the salt-dead codes themselves.
- *Proved from `PinnedScales`.* `Fp4FormingCode` proves
  `CodeProofs.rhoDFp4At_ninety : PinnedScales alpha beta → RhoDFp4At 90 alpha beta`, and
  `CodeProofs.rhoDFp4At_ninety_eq : RhoDFp4At 90 = RhoDFp4` is `rfl`, with `screenDead` at `90·ρ²`. In fact every
  salt-dead element lies in a screened block (`CodeProofs.saltDead4_screened`).
- *The scales' bounds it reads* (`CodeProofs.PinnedScales`): on rows that pass D-NF, `|α·x_l| ≤ 2688` and
  `(128/18769)·(1 − 2⁻¹⁰)²·α²·ρ² ≤ β²`; on every row, `0 ≤ β < 464`.
- *Argued, not proved, for `pearl_c4.stats`.* That the reference's scales meet `PinnedScales` rests on the FP32 error
  bound of `s5_stats`' `ρ²`, about `(k/256 + 13)·u ≈ 270u ≈ 2^-15.9` at `k = 2^16` (`u = 2⁻²⁴`), against the `2⁻¹⁰`
  allowance, and on a numerical table of it. There is no Lean lemma
  about `s5_stats` until `s5_stats` is pinned. -/
def RhoDFp4 (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) : Prop :=
  ∀ (p k : ℕ) (X : Matrix (Fin p) (Fin k) ℚ), 128 ∣ k → (∀ i, RowRules4 beta k (X i)) →
    64 * ((univ : Finset (Fin p × Fin k)).filter fun e => saltDead4 alpha beta k (X e.1) e.2).card ≤ p * k

/-! ## The ticket and the firings -/

/-- **The ticket at an FP4 atom**: `C̃_ij = cTilde p` over rows `A i` and `B j`, as a value; `none` where a row's forming
or the chain leaves the model. -/
def ticket4 (p : Fp4Atom) {m n : ℕ} (A : Op4 m) (B : Op4 n) : Matrix (Fin m) (Fin n) (Option ℚ) :=
  fun i j => (do let a ← A i; let b ← B j; cTilde p a.1 b.1 a.2 b.2).bind wordQ

/-- One firing's operands: `k` code pairs and each operand's `k / block` scale bytes. -/
abbrev Fire4 : Type := List Nib × List Nib × List (Fin 256) × List (Fin 256)

/-- The chain's firings over two formed rows, in ascending K: firing `t` takes codes `t·k, …, t·k + k − 1` and the scale
bytes `t·(k/block), …` of each row. On rows `cTilde` accepts, their fold from `+0` is `cTilde`'s chain
(`fires_fold_eq_cTilde`). -/
def fires (p : Fp4Atom) (a b : Row4) : List Fire4 :=
  (List.range (a.1.length / p.k)).map fun t =>
    ((a.1.drop (t * p.k)).take p.k, (b.1.drop (t * p.k)).take p.k,
      (a.2.drop (t * p.scales)).take p.scales, (b.2.drop (t * p.scales)).take p.scales)

/-- One firing of `p` on an accumulator word; `none` outside the model, and after it. -/
def fireStep (p : Fp4Atom) (w : Option ℕ) (x : Fire4) : Option ℕ :=
  w.bind fun c => p.step c x.1 x.2.1 x.2.2.1 x.2.2.2

/-- The grid `2^lsb` of one firing, as `Fp4Atom.step` sets it: `lsb = max(max_g(unit_g − fracBits),
E_acc − (accWindow − 1), gridFloor)` over the live groups and a nonzero accumulator. It is `none` where the step leaves
the model or has no anchor. -/
def stepLsb (p : Fp4Atom) (acc : ℕ) (a b : List Nib) (sa sb : List (Fin 256)) : Option ℤ := do
  let ts ← p.groupTerms a b sa sb
  let c ← (fp32Value acc).finite?
  let anchors := ts.filterMap (fun t => if t.live then some (t.unit - p.fracBits) else none) ++
    (if c.mag = 0 then [] else [c.exp - ((p.accWindow - 1 : Nat) : Int)])
  match anchors with
  | [] => none
  | x :: xs => some (max (xs.foldl max x) p.gridFloor)

/-- Block `j` of a firing from the accumulator word `acc` is sub-grid (§2's term (5), D-SS's T1-a clause) when its largest
possible scaled dot, `2304·m_a·m_b·2^(e_a + e_b − 2)` (16 products of at most `12·12` quarters), lies below the
firing's grid. Such a block adds exactly `0`, and its two scale bytes show it. At NVFP4 a block is one group of 16. -/
def subgridAt (p : Fp4Atom) (acc : ℕ) (x : Fire4) (j : ℕ) : Bool :=
  match stepLsb p acc x.1 x.2.1 x.2.2.1 x.2.2.2, x.2.2.1[j]?, x.2.2.2[j]? with
  | some lsb, some sa, some sb =>
    match scaleDyadic p.fmt sa, scaleDyadic p.fmt sb with
    | some (ma, ea), some (mb, eb) => decide ((2304 * ma * mb : ℚ) * (2 : ℚ) ^ (ea + eb - 2) < (2 : ℚ) ^ lsb)
    | _, _ => false
  | _, _, _ => false

/-- **The sub-grid flags**, debit (5): block `b` along `k` of word `(i, j)`'s chain, replayed from `+0`, is sub-grid in the
firing that holds it (firing `b / (k/block)`, block `b % (k/block)` of it). They are `false` where a row's forming or
the chain left the model. -/
def subgridFlags4 (p : Fp4Atom) {m n : ℕ} (A : Op4 m) (B : Op4 n) (i : Fin m) (j : Fin n) (b : ℕ) : Bool :=
  match A i, B j with
  | some a, some b' =>
    let xs := fires p a b'
    match (xs.take (b / p.scales)).foldl (fireStep p) (some 0), xs[b / p.scales]? with
    | some acc, some x => subgridAt p acc x (b % p.scales)
    | _, _ => false
  | _, _ => false

/-! ## U at an FP4 atom (`peel_factors_a`, `peel_factors_b`, `peel`) -/

/-- Ẽ (`e_tilde`): `bf16(|E|·a_E·2⁻⁴)`, exact (an E2M1 magnitude times a UE4M3 scale has at most 6 significant bits),
with `E`'s sign bit, so code `0x8` is `−0`. -/
def eTilde (e : Fin 32 → Nib) (aE : Fin 256) : Fin 32 → ℕ := fun q =>
  bf16WordOf (|Fp4.decode (e q)| * codeVal aE / 16) ||| (if (e q).sign then 0x8000 else 0)

/-- A scale format's byte for `1.0` (`fmt.one`): UE4M3 `0x38`, UE8M0 `127`. -/
def scaleOne : ScaleFmt → Fin 256
  | .ue4m3 => 0x38
  | .ue8m0 => 127

/-- The scale bytes of a column of `F` in a chain: every block at `1.0`. -/
def onesScales (p : Fp4Atom) (k : ℕ) : List (Fin 256) := List.replicate (k / p.block) (scaleOne p.fmt)

/-- Column `q` of a basis `F` (`k` lines of 32 codes), along `k`. -/
def colF {k : ℕ} (F : Matrix (Fin k) (Fin 32) Nib) (q : Fin 32) : List Nib := List.ofFn fun t => F t q

/-- **U at an FP4 atom**, entry `(i, j)` as an FP32 word, or `none` where a forming or a step leaves the model:
- `C̃`, `A′·F_B`, `B̃·F_A` and `G = F_Bᵀ·F_A` are `cTilde p` chains, each word finite; `F`'s columns are at scale `1.0`;
- `P_A,i = [Ẽ_A,i ‖ RN_bf16(A′·F_B)]` and `P_B,j = [RN_bf16(RN_f32(acc_q − BF_jq)) ‖ −Ẽ_B,j]`, with `acc_q` the FP32 FMA
  chain, are `PeelExact.paRow?` and `pbRow?`;
- `U_ij` is four BF16 K16 steps from `C̃_ij`, `PeelExact.uWord?`. -/
def peelFp4 (p : Fp4Atom) {m n k : ℕ} (A : Op4 m) (B : Op4 n) (FA FB : Matrix (Fin k) (Fin 32) Nib)
    (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ) : Matrix (Fin m) (Fin n) (Option ℕ) :=
  fun i j => do
    let a ← A i
    let b ← B j
    let ch := fun (x y : List Nib) (sx sy : List (Fin 256)) => (cTilde p x y sx sy).bind PeelExact.finite?
    let one := onesScales p k
    let c ← ch a.1 b.1 a.2 b.2
    let af ← (List.ofFn fun q => ch a.1 (colF FB q) a.2 one).mapM id
    let bf ← (List.ofFn fun q => ch b.1 (colF FA q) b.2 one).mapM id
    let g ← (List.ofFn fun q => (List.ofFn fun q' => ch (colF FB q) (colF FA q') one one).mapM id).mapM id
    let pa ← PeelExact.paRow? (List.ofFn (beA i)) af
    let pb ← PeelExact.pbRow? (List.ofFn (beB j)) bf g
    PeelExact.uWord? c pa pb

/-! ## The debit's pieces -/

/-- **D-24's 2:4 window** (`two_four_windows`): a formed row's aligned window `w` of 128 along `k` is 2:4 when, in each
of its 16 aligned 8-chunks (codes `8q` to `8q + 7`), at most two of the four pairs `(2j, 2j + 1)` hold a nonzero code
(`code & 7 ≠ 0`); `false` where the row's forming left the model. This is the hardware's pair rule: sm_120's NVFP4
sparse MMA (`kind::mxf4nvf4.sp`, m16n8k128) gives each 8-chunk two 2-bit indices, which pick 2 of its 4 pairs. The
reference's 4-group rule (each 4-group at most two nonzero) is `kind::f8f6f4`'s pattern, which has no UE4M3 block scales
and saves nothing over dense NVFP4; neither rule implies the other (`rowWin24_groups`). -/
def rowWin24 (r : Option Row4) (w : ℕ) : Bool :=
  match r with
  | some r => (List.range 16).all fun q =>
      decide (((List.range 4).filter fun j =>
        (r.1.getD (128 * w + 8 * q + 2 * j) 0).val % 8 != 0 ||
          (r.1.getD (128 * w + 8 * q + 2 * j + 1) 0).val % 8 != 0).length ≤ 2)
  | none => false

/-! ## The base-split fix: F1′ and F2 (theory §6.2–6.5, with its FP4 conditions)

F1′ reads NVFP4's forming (`formRow`, `nvf4NoiseElem`), the forming `devFp4Sm120` runs, since its law needs each code
at a fixed scale byte. -/

/-- F1′'s patch price per changed element and MAC, `c = 16.92/128`: one FFMA per word (16.92 FP4 units) over a window's
128 MACs. -/
def f1c : ℚ := 423 / 3200

/-- The scale byte `formRow` sets on block `b` of a row at the basis `F`; `none` where the forming leaves the model. -/
def blockByte {k : ℕ} (α β : ℚ) (x : Fin k → ℚ) (e : Fin 32 → Nib) (F : Fin k → Fin 32 → Nib) (b : ℕ) :
    Option (Fin 256) :=
  (formRow α β x e F).bind fun r => r.2[b]?

/-- **Block `b`'s modal scale byte given `E_i`** (§6.2's "modal A0"): the byte `formRow` sets on the block at the most
bases `F`, every basis equally likely, so the mode of the byte's exact law given the row's `E` line `e`. Ties go to the
smaller byte. It is salt-free given `E_i`. -/
def modalByte {k : ℕ} (α β : ℚ) (x : Fin k → ℚ) (e : Fin 32 → Nib) (b : ℕ) : Fin 256 :=
  ((List.finRange 256).argmax fun v =>
    ((univ : Finset (Fin k → Fin 32 → Nib)).filter fun F => blockByte α β x e F b = some v).card).getD 0

/-- An element's code value at the scale byte `s`, from its noise atom at the basis line `f`: `e2m1(f32(D·r_s))` with
`D = nvf4NoiseElem α x β e f`, as `formRow` casts it; `none` outside the model. -/
def codeAt (α x β : ℚ) (e : Fin 32 → Nib) (s : Fin 256) (f : Fin 32 → Nib) : Option ℚ :=
  (nvf4NoiseElem α x β e f).map fun w => Fp4.decode (e2m1Word (f32Word (wordVal w * nvf4Recip s)))

/-- **`p_t` under the exact noise law** (§6.2; the FP4 condition (2)): the chance that an element's code value at
the scale byte `s` is not its modal value, the one `A0` takes. The basis line `f` is uniform over all `16^32`, and the
value is a function of `f` given `E_i`, so this is the discrete law of the row's noise `E_i·F_t` through the noise atom
itself, not the Gaussian at `σα`. -/
def leaveProb (α x β : ℚ) (e : Fin 32 → Nib) (s : Fin 256) : ℚ :=
  1 - (((univ : Finset (Fin 32 → Nib)).sup fun f₀ =>
    ((univ : Finset (Fin 32 → Nib)).filter fun f => codeAt α x β e s f = codeAt α x β e s f₀).card : ℕ) : ℚ) /
    16 ^ 32

open Classical in
/-- F1′'s `p_t` along a row (§6.2): `leaveProb` at its block's modal byte, and `0` on salt-dead codes (`sd`), as the
reference's `change_probabilities`. A `0` treats the code as never changing, the side that charges more. A screened
block's other codes are charged at their own `p_t`. Positions from `k` on read `0`. -/
noncomputable def rowLeave {k : ℕ} (α β : ℚ) (x : Fin k → ℚ) (e : Fin 32 → Nib) (sd : Fin k → Prop) (t : ℕ) : ℚ :=
  if h : t < k then
    if sd ⟨t, h⟩ then 0 else leaveProb α (x ⟨t, h⟩) β e (modalByte α β x e (t / 16))
  else 0

/-- The sorted in-block layout (§6.2): the rank of position `o < 16` of block `b` by `p`, ties by position, so rank `0`
is the most reliable code. Sorted neighbours pair up (pair `rank / 2`); a block's chunk `0` takes pairs 0, 2, 4 and 6,
and its chunk `1` pairs 1, 3, 5 and 7, so pair `q` sits in chunk `q % 2`. -/
def blockRank (p : ℕ → ℚ) (b o : ℕ) : ℕ :=
  ((range 16).filter fun o' =>
    p (16 * b + o') < p (16 * b + o) ∨ (p (16 * b + o') = p (16 * b + o) ∧ o' < o)).card

/-- Whether position `j` of a window of 128 changed, in the change pattern `z` (`false` from 128 on). -/
def chg (z : Fin 128 → Bool) (j : ℕ) : Bool := if h : j < 128 then z ⟨j, h⟩ else false

/-- **`X_w`**, the elements beyond 2:4 in window `w` at the change pattern `z` (§6.2): per block of the window and chunk
of its sorted layout, when `m > 2` of the chunk's four pairs hold a change, the changed elements of its `m − 2` pairs
with the fewest. -/
def beyond24 (p : ℕ → ℚ) (w : ℕ) (z : Fin 128 → Bool) : ℕ :=
  ∑ bl ∈ range 8, ∑ c ∈ range 2,
    let cnt := fun a : ℕ => ((range 16).filter fun o =>
      blockRank p (8 * w + bl) o / 2 = 2 * a + c ∧ chg z (16 * bl + o) = true).card
    let ex := ((range 4).filter fun a => 0 < cnt a).card - 2
    let ones := min ((range 4).filter fun a => cnt a = 1).card ex
    ones + 2 * (ex - ones)

/-- **`N_h`**, the changed codes in the aligned k64 half `h` of a window at the change pattern `z` (§6.2). -/
def halfChanged (z : Fin 128 → Bool) (h : ℕ) : ℕ := ((range 64).filter fun o => chg z (64 * h + o) = true).card

/-- **`R_24`** (the FP4 condition (1)): per MAC of window `w` at the change pattern `z`, one 2:4 pass saves
`1/2`, less the CUDA-core patch of the elements beyond 2:4, `c·X_w`. -/
def r24 (p : ℕ → ℚ) (w : ℕ) (z : Fin 128 → Bool) : ℚ := max 0 (1 / 2 - f1c * beyond24 p w z)

/-- **`R_skip`** (the FP4 condition (1)): per MAC of a window at the change pattern `z`, a skip or patch per
k64 half, each saving `1/2` less the patch of its changed codes, `c·N_h`. -/
def rSkip (z : Fin 128 → Bool) : ℚ := ∑ h ∈ range 2, max 0 (1 / 2 - f1c * halfChanged z h)

/-- The expectation of `g` over window `w`'s change patterns, the codes independent given `E_i` (§6.2; exact for one
row, since the `F_t` lines are independent): position `j` changes with chance `p (128·w + j)`. -/
def windowMean (p : ℕ → ℚ) (w : ℕ) (g : (Fin 128 → Bool) → ℚ) : ℚ :=
  ∑ z : Fin 128 → Bool, (∏ j : Fin 128, if z j then p (128 * w + (j : ℕ)) else 1 - p (128 * w + (j : ℕ))) * g z

/-- **`M_w`**, the blocks of window `w` whose realised scale byte, in the drawn tile's own forming `r`, is not their
modal byte (§6.2). It is `0` where the row's forming left the model, the side that charges more. -/
def offModal {k : ℕ} (α β : ℚ) (x : Fin k → ℚ) (e : Fin 32 → Nib) (r : Option Row4) (w : ℕ) : ℕ :=
  match r with
  | some r => ((range 8).filter fun bl => r.2.getD (8 * w + bl) 0 ≠ modalByte α β x e (8 * w + bl)).card
  | none => 0

/-- **B̃'s overfit on F1′** (`F1_B_OVERFIT = 10`; theory §6.12, measured, not derived): a layout chosen after the salt
for B̃'s residual is shared by all `n` weight rows, so B̃'s F1′ is charged at 10 times its expectation per window, capped
at the window's MACs. A′'s is charged at 1 (its layout must re-gather B̃). -/
def f1BOverfit : ℚ := 10

/-- **F1′'s charge per word of one row** (theory §6.2; the FP4 condition (1), with `κ` inside the expectation;
the reference's `split_units` and `split_saving`), in FP4 units: over each aligned window `w` of 128 (8 blocks),
`128·min(1, ovf·min(1, E[max(0, R_24 − κ)] + E[max(0, R_skip − κ)]))` with `κ = min(c·M_w, 1/8)`. The expectation is
under the exact law of the row's codes given its `E` line `e` (`rowLeave`), at §6.2's sorted in-block layout and modal
`A0`. The row is the clean `x` at the scales `α`, `β`, with salt-dead elements `sd`, and `r` is its forming in the
drawn tile. The overfit `ovf` is 1 on A′'s rows and `f1BOverfit` on B̃'s.
- The two routes: one 2:4 pass plus a CUDA-core patch, or a skip or patch per k64 half. The sum bounds the better of
  the two, and the inner `min` caps it at the most any route saves.
- `κ` is `A0`'s scale correction: each block of the window off its modal byte needs one FFMA per word, `c` per MAC,
  and a dense correction GEMM costs at least `1/8`. -/
noncomputable def f1Row {k : ℕ} (ovf α β : ℚ) (x : Fin k → ℚ) (e : Fin 32 → Nib) (sd : Fin k → Prop)
    (r : Option Row4) : ℚ :=
  ∑ w ∈ range (k / 128),
    let p := rowLeave α β x e sd
    let κ := min (f1c * offModal α β x e r w) (1 / 8)
    128 * min 1 (ovf * min 1
      (windowMean p w (fun z => max 0 (r24 p w z - κ)) + windowMean p w (fun z => max 0 (rSkip z - κ))))

/-- F2's `c_L` grid in `m` (`pearl_c4_c_L.MS`). -/
def cLMs : List ℕ := [128, 1024, 8192, 16384, 32768, 65536, 262144, 1048576, 16777216]

/-- F2's `c_L` grid in `k` (`pearl_c4_c_L.KS`). -/
def cLKs : List ℕ := [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

/-- F2's `c_L` grid in `n` (`pearl_c4_c_L.NS`). -/
def cLNs : List ℕ := [128, 1024, 8192, 16384, 32768, 65536, 262144]

/-- `cLTable[i][j][l]` is `c_L` at `m = cLMs[i]`, `k = cLKs[j]`, `n = cLNs[l]`, in units of `10⁻⁴` rounded down:
`pearl_c4_c_L.C_L` exactly (at `b3af5481`, blob `57adb6af9ccc`). -/
def cLTable : List (List (List ℕ)) :=
  [[[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 128, k = 16
    [17680, 16918, 16801, 16792, 16788, 16786, 16784],  -- k = 32
    [16880, 15477, 15275, 15261, 15253, 15250, 15247],  -- k = 64
    [16290, 14577, 14294, 14274, 14264, 14259, 14255],  -- k = 128
    [15545, 14005, 13656, 13631, 13619, 13613, 13608],  -- k = 256
    [15545, 13924, 13205, 13154, 13128, 13109, 13091],  -- k = 512
    [15545, 13650, 12398, 12287, 12232, 12232, 12232],  -- k = 1024
    [15545, 13529, 12398, 12259, 12134, 12068, 12019],  -- k = 2048
    [15545, 13529, 12308, 12052, 11904, 11829, 11773],  -- k = 4096
    [15545, 13529, 12308, 11974, 11639, 11470, 11344],  -- k = 8192
    [15545, 13529, 12308, 11974, 11591, 11320, 11082],  -- k = 16384
    [15545, 13529, 12308, 11974, 11591, 11310, 10926],  -- k = 32768
    [15545, 13529, 12308, 11974, 11591, 11310, 10845]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 1024, k = 16
    [16918, 15873, 15728, 15715, 15709, 15706, 15703],  -- k = 32
    [15477, 13633, 13042, 12984, 12955, 12941, 12930],  -- k = 64
    [14577, 12809, 11898, 11790, 11720, 11676, 11642],  -- k = 128
    [13991, 11621, 10878, 10751, 10687, 10652, 10624],  -- k = 256
    [13924, 10858,  9462,  9171,  9026,  8954,  8899],  -- k = 512
    [13525, 10684,  9175,  8839,  8579,  8450,  8353],  -- k = 1024
    [13215,  9803,  8992,  8839,  8579,  8450,  8353],  -- k = 2048
    [13215,  9634,  8877,  8636,  8503,  8437,  8353],  -- k = 4096
    [13215,  9528,  8471,  8396,  8204,  8081,  7928],  -- k = 8192
    [13215,  9528,  8471,  8396,  8137,  7886,  7498],  -- k = 16384
    [13215,  9528,  8471,  8396,  7994,  7552,  7179],  -- k = 32768
    [13215,  9528,  8471,  8396,  7756,  7323,  6791]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 8192, k = 16
    [16801, 15728, 15432, 15402, 15387, 15380, 15374],  -- k = 32
    [15275, 13042, 12319, 12258, 12219, 12199, 12184],  -- k = 64
    [14294, 11898, 10451, 10186, 10054,  9988,  9938],  -- k = 128
    [13429, 10878,  9583,  9315,  8885,  8793,  8688],  -- k = 256
    [13205,  9462,  8803,  8513,  8368,  8295,  8203],  -- k = 512
    [12398,  9175,  8175,  7781,  7522,  7393,  7296],  -- k = 1024
    [12398,  8945,  7479,  7255,  6893,  6567,  6323],  -- k = 2048
    [12071,  8806,  6781,  6582,  6479,  6194,  5324],  -- k = 4096
    [12071,  8471,  6612,  6275,  6107,  6023,  5324],  -- k = 8192
    [12071,  8471,  6595,  6275,  6103,  5928,  5324],  -- k = 16384
    [12071,  8006,  6253,  6051,  5950,  5899,  5324],  -- k = 32768
    [12071,  8006,  5962,  5789,  5651,  5562,  5213]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 16384, k = 16
    [16792, 15715, 15402, 15372, 15357, 15350, 15344],  -- k = 32
    [15261, 12984, 12258, 12184, 12131, 12089, 12056],  -- k = 64
    [14274, 11790, 10186,  9930,  9798,  9732,  9682],  -- k = 128
    [13361, 10751,  9315,  9036,  8567,  8475,  8272],  -- k = 256
    [13154,  9171,  8513,  8390,  8141,  7778,  7526],  -- k = 512
    [12287,  8839,  7781,  7706,  7447,  7317,  7220],  -- k = 1024
    [12259,  8761,  7255,  7023,  6782,  6457,  6213],  -- k = 2048
    [11800,  8627,  6582,  6376,  6273,  5996,  5073],  -- k = 4096
    [11800,  8382,  6275,  5939,  5763,  5672,  4937],  -- k = 8192
    [11800,  8354,  6275,  5790,  5495,  5347,  4937],  -- k = 16384
    [11800,  7896,  6051,  5776,  5495,  5331,  4895],  -- k = 32768
    [11800,  7896,  5789,  5473,  5261,  5005,  4814]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 32768, k = 16
    [16788, 15709, 15387, 15357, 15342, 15335, 15329],  -- k = 32
    [15253, 12955, 12219, 12131, 12046, 12003, 11947],  -- k = 64
    [14264, 11720, 10054,  9798,  9670,  9604,  9554],  -- k = 128
    [13327, 10687,  9167,  8817,  8408,  8180,  7947],  -- k = 256
    [13128,  9026,  8368,  8141,  7788,  7498,  7015],  -- k = 512
    [12232,  8579,  7522,  7447,  7313,  7107,  6736],  -- k = 1024
    [12134,  8437,  6893,  6782,  6727,  6402,  6157],  -- k = 2048
    [11655,  8437,  6479,  6273,  6148,  5896,  4948],  -- k = 4096
    [11639,  8198,  6107,  5763,  5582,  5491,  4742],  -- k = 8192
    [11591,  8137,  6056,  5495,  5199,  5044,  4721],  -- k = 16384
    [11591,  7841,  5950,  5495,  5069,  4810,  4607],  -- k = 32768
    [11591,  7756,  5651,  5261,  5034,  4779,  4412]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 65536, k = 16
    [16786, 15706, 15380, 15350, 15335, 15327, 15321],  -- k = 32
    [15250, 12941, 12199, 12089, 12003, 11922, 11848],  -- k = 64
    [14259, 11676,  9988,  9732,  9604,  9540,  9458],  -- k = 128
    [13309, 10652,  8793,  8475,  8180,  7914,  7685],  -- k = 256
    [13109,  8954,  8295,  7964,  7611,  7290,  6760],  -- k = 512
    [12232,  8450,  7393,  7317,  7107,  6755,  6407],  -- k = 1024
    [12068,  8111,  6567,  6457,  6402,  6374,  5957],  -- k = 2048
    [11582,  8111,  6194,  5996,  5896,  5845,  4885],  -- k = 4096
    [11470,  8081,  6023,  5672,  5491,  5380,  4644],  -- k = 8192
    [11320,  7886,  5928,  5347,  5044,  4885,  4593],  -- k = 16384
    [11310,  7552,  5784,  5302,  4810,  4550,  4345],  -- k = 32768
    [11310,  7323,  5458,  5005,  4474,  4083,  3790]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 262144, k = 16
    [16784, 15703, 15374, 15344, 15329, 15321, 15316],  -- k = 32
    [15247, 12930, 12184, 12056, 11947, 11848, 11773],  -- k = 64
    [14255, 11642,  9938,  9682,  9554,  9458,  9381],  -- k = 128
    [13297, 10624,  8688,  8272,  7947,  7685,  7468],  -- k = 256
    [13091,  8899,  8216,  7656,  7320,  6936,  6370],  -- k = 512
    [12232,  8353,  7296,  7220,  6736,  6407,  5874],  -- k = 1024
    [12019,  7867,  6323,  6213,  6157,  6026,  5466],  -- k = 2048
    [11527,  7867,  5324,  5073,  4948,  4885,  4838],  -- k = 4096
    [11344,  7867,  5324,  4937,  4742,  4644,  4564],  -- k = 8192
    [11082,  7562,  5324,  4937,  4721,  4593,  4352],  -- k = 16384
    [10926,  7179,  5324,  4895,  4615,  4345,  4120],  -- k = 32768
    [10826,  6791,  5213,  4809,  4181,  3790,  3477]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 1048576, k = 16
    [16784, 15703, 15373, 15343, 15328, 15320, 15314],  -- k = 32
    [15246, 12927, 12180, 12048, 11929, 11829, 11754],  -- k = 64
    [14254, 11634,  9926,  9670,  9541,  9438,  9313],  -- k = 128
    [13294, 10617,  8654,  8203,  7878,  7628,  7408],  -- k = 256
    [13087,  8886,  8144,  7467,  6956,  6616,  6107],  -- k = 512
    [12232,  8328,  7271,  7146,  6552,  6104,  5615],  -- k = 1024
    [12006,  7806,  6262,  6152,  6096,  5864,  5210],  -- k = 2048
    [11514,  7806,  5052,  4801,  4676,  4613,  4566],  -- k = 4096
    [11312,  7806,  4846,  4455,  4259,  4161,  4088],  -- k = 8192
    [11023,  7396,  4833,  4270,  3988,  3847,  3738],  -- k = 16384
    [10826,  7044,  4833,  4270,  3988,  3754,  3553],  -- k = 32768
    [10621,  6658,  4833,  4270,  3988,  3717,  3394]],  -- k = 65536
   [[20000, 20000, 20000, 20000, 20000, 20000, 20000],  -- m = 16777216, k = 16
    [16784, 15702, 15372, 15342, 15327, 15320, 15314],  -- k = 32
    [15246, 12927, 12179, 12046, 11923, 11823, 11748],  -- k = 64
    [14254, 11631,  9922,  9666,  9534,  9431,  9266],  -- k = 128
    [13293, 10615,  8644,  8181,  7856,  7610,  7373],  -- k = 256
    [13086,  8881,  8126,  7391,  6881,  6516,  6025],  -- k = 512
    [12232,  8321,  7264,  7114,  6471,  5940,  5224],  -- k = 1024
    [12003,  7787,  6243,  6132,  6077,  5667,  4845],  -- k = 2048
    [11509,  7787,  4967,  4716,  4591,  4528,  4481],  -- k = 4096
    [11302,  7787,  4695,  4304,  4108,  4010,  3937],  -- k = 8192
    [11004,  7340,  4455,  3892,  3610,  3469,  3364],  -- k = 16384
    [10795,  6963,  4455,  3811,  3373,  3154,  2990],  -- k = 32768
    [10549,  6566,  4455,  3811,  3319,  2964,  2697]]]  -- k = 65536

/-- `cLTable`'s entry `(i, j, l)`; `0` off the table. -/
def cLEntry (i j l : ℕ) : ℕ := ((cLTable[i]?.bind (·[j]?)).bind (·[l]?)).getD 0

/-- **F2's `c_L`** (theory §6.3, and the FP4 condition (3)):
the masked int8 route's least cost per honest MAC before its patch, on a GEMM `(m, k, n)` over NVFP4's blocks of 16.
- *What it is.* The fast-matmul catalogue's minimum (Perminov at `ac13ca88`), each scheme at its lower-bound add count
  (one add per new form), with leaves at W1's 2.0 FP4 slots per MAC and 4.25 FP4 slots per pre-add element-op). A
  cheater's shared subexpressions put its real add count between that bound and the achievable routes', so F2 charges at
  least its best saving.
- *How it is read.* It is the reference's pinned table (`pearl_c4_c_L` at `b3af5481`: `cLMs`, `cLKs`, `cLNs`,
  `cLTable`), at the least tabled `m`, `k` and `n` at least the shape's, as `pearl_c4.int8_depth_cost` reads it. The
  table falls in each of `m`, `k` and `n` (`cLTable_falls`), so a shape between tabled points takes the lower `c_L`
  (`cL_between`).
- *Values.* 0.6612, 0.5790, 0.5069 and 0.4083 at 8,192³, 16,384³, 32,768³ and 65,536³ (`cL_8192` to `cL_65536`), where
  Strassen–Winograd's formula gave 0.808, 0.707, 0.619 and 0.542, and 0.2697 at the table's corner (`cL_corner`).
- *Past the table* (`m > 2^24`, `k > 2^16` or `n > 2^18`) it is `0`, the most F2 can charge (`cL_beyond`), where
  `int8_depth_cost` raises. `pearlCDomainFp4` keeps every unit inside the table, and F2's class split reads it at a
  depth of at least one block. -/
def cL (m k n : ℕ) : ℚ :=
  match cLMs.findIdx? (fun v => decide (m ≤ v)), cLKs.findIdx? (fun v => decide (k ≤ v)),
      cLNs.findIdx? (fun v => decide (n ≤ v)) with
  | some i, some j, some l => (cLEntry i j l : ℚ) / 10000
  | _, _, _ => 0

/-- The most common value of `r`, ties to the smaller (`max(set(r), key=lambda s: (r.count(s), -s))`); `0` on `[]`. -/
def mostCommon (r : List ℤ) : ℤ :=
  r.foldl (fun b s => if r.count b < r.count s ∨ (r.count b = r.count s ∧ s < b) then s else b) (r.headD 0)

/-- **`_share`**: the share of the cells `c` at their most common value; `0` on `[]`. -/
def share (c : List ℤ) : ℚ := if c = [] then 0 else (((c.map c.count).foldl max 0 : ℕ) : ℚ) / c.length

/-- **`_row_share`**: the mean over the rows of each row's own `share` (flatness up to per-row scale factors, which the
route absorbs); `0` with no rows. -/
def rowShare (rows : List (List ℤ)) : ℚ := if rows = [] then 0 else (rows.map share).sum / rows.length

/-- **`_labels`**: block column `b`'s label, the most common offset of a row's byte there from that row's most common
byte. -/
def colLabel (rows : List (List ℤ)) (b : ℕ) : ℤ := mostCommon (rows.map fun r => r.getD b 0 - mostCommon r)

/-- The masked int8 route's saving per chain MAC at flatnesses `fa` and `fb` over a depth `kk` (`int8_saving`'s
`save`): `max(0, 1 − c_L(m, kk, n) − 4·(2 − fa − fb))`, the patch at 4 slots per useful MAC per side. -/
def int8Save (m n : ℕ) (fa fb : ℚ) (kk : ℕ) : ℚ := max 0 (1 - cL m kk n - 4 * (2 - fa - fb))

/-- **F2 per chain MAC** (theory §6.3; the reference's `int8_saving`) on a tile's realised scale bytes, `a` for A′'s
rows and `b` for B̃'s, `nb` blocks of 16 per row, in a unit of `m` and `n` rows: the largest of three
readings of flatness,
- the tile's modal byte (`share` of all the bytes) on each side, over `k = 16·nb`;
- each row's own modal byte (`rowShare`), over `k`;
- per block-column class: a column's class is its label on A′ and on B̃ (`colLabel`), and each class is a sub-GEMM over
  its own depth, `16·|cols|`, with row factors free, saving `int8Save` at its `rowShare`s, weighted by `|cols|/nb`. -/
def int8Saving (m n nb : ℕ) (a b : List (List ℤ)) : ℚ :=
  let key := fun c : ℕ => (colLabel a c, colLabel b c)
  let split := (((List.range nb).map key).dedup.map fun κ =>
    let cs := (List.range nb).filter fun c => key c = κ
    (cs.length : ℚ) / nb * int8Save m n (rowShare (a.map fun r => cs.map fun c => r.getD c 0))
      (rowShare (b.map fun r => cs.map fun c => r.getD c 0)) (16 * cs.length)).sum
  max (max (int8Save m n (share a.flatten) (share b.flatten) (16 * nb))
    (int8Save m n (rowShare a) (rowShare b) (16 * nb))) split

/-- The realised scale bytes of the rows of `A` in `I`, in row order, as integers. Rows whose forming left the model
are left out. The bytes `formRow` sets are UE4M3 of non-negative values, so bit 7 is clear and they are the bytes the
hardware reads (`hw_scales`). -/
def tileBytes {p : ℕ} (A : Op4 p) (I : Finset (Fin p)) : List (List ℤ) :=
  (List.finRange p).filterMap fun i => if i ∈ I then (A i).map fun r => r.2.map fun v => ((v : ℕ) : ℤ) else none

end Pouw.PearlC.Fp4Dev

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

/-! ## The record -/

/-- W1 prices in FP4 units (1 = one dense NVFP4 MAC with FP32 accumulation, W1 pricing an instruction by its whole
fragment): a BF16 MAC, the credited (salt-dependent) forming per element, and the forming per element that depends on
the activations alone. -/
structure Fp4Prices where
  bf16 : ℚ
  fs : ℚ
  qa : ℚ

/-- **sm_120's prices in FP4 units** (theory §2–§4, node 2 at locked 2,100 MHz), with an FP8 unit worth 2 FP4 units,
the block scale on `lut256` (§6.6) and FADD at the measured `1047/125`
= 8.376, as `Prices.sm120Loop`. It is `Fp4Prices.sm120At (1047/125) Fp4ScalePath.lut256`:
- a BF16 MAC is 4 (2.000 FP8 units);
- `fs = 105299/1000 = 105.299`: the noise atom 64 (a dense k64 NVFP4 `mma.sync`, 64 lanes, at full shape); the block
  scale on `lut256` at its W1 issue prices, `7.5 + FADD/8 = 8.547` (half an `F2FP`, 1.25 `PRMT` and one `LDS` per block
  of 16, and the FMUL `amax·1/6`); and the cast, FMUL plus the E2M1 cvt's F2FP alone, at `2·8.376 + 2·8.0 = 32.752`;
- `qa = 9423/500 = 18.846`: the noisy amax (one FMNMX, `2·8.376`) and ρ's squared accumulate on every 8th element
  (`2·8.376/8`), both reading the row alone.

The price rule publishes the larger γ, and γ rises with the FADD price, so the record is at the in-loop 8.376, not the
issue-bound 8.00 (`fs = 104.5`, `qa = 18`). `creditFp4` and `wrefFp4` read this one `fs`. It was `2444/25 = 97.76`, with
the block scale at 1 and FADD at 8.38. -/
def Fp4Prices.sm120 : Fp4Prices := ⟨4, 105299 / 1000, 9423 / 500⟩

/-- **An FP4 device record**: `PearlCDevice`'s fields on block-scaled FP4 operands, without promotion (`G = 0`, no FP32
add between firings). It holds the prices, the forming of one row from its values, its scales and its noise lines, the
ticket and U maps on formed operands, the flags (debit (2)), the sub-grid flags (debit (5)), and P3's salt-dead
predicate at this forming (debits (1), (3) and (4)). -/
structure Fp4Device where
  prices : Fp4Prices
  form : ∀ {k : ℕ}, ℚ → ℚ → (Fin k → ℚ) → (Fin 32 → Nib) → (Fin k → Fin 32 → Nib) → Option Row4
  ticket : ∀ {m n : ℕ}, Op4 m → Op4 n → Matrix (Fin m) (Fin n) (Option ℚ)
  useful : ∀ {m n k : ℕ}, Op4 m → Op4 n → Matrix (Fin k) (Fin 32) Nib → Matrix (Fin k) (Fin 32) Nib →
    Matrix (Fin m) (Fin 32) ℕ → Matrix (Fin n) (Fin 32) ℕ → Matrix (Fin m) (Fin n) (Option ℚ)
  flags : ∀ {m n : ℕ}, Op4 m → Op4 n → Fin m → Fin n → ℕ → Bool
  subgrid : ∀ {m n : ℕ}, Op4 m → Op4 n → Fin m → Fin n → ℕ → Bool
  saltDead : ((k : ℕ) → (Fin k → ℚ) → ℚ) → ((k : ℕ) → (Fin k → ℚ) → ℚ) → (k : ℕ) → (Fin k → ℚ) → Fin k → Prop

/-- **`devFp4Sm120`**, Pearl-C4 on sm_120 (`BLACKWELL_SM120_NVF4`, `Fp4.sm120Nvf4`: 4 groups of 16, UE4M3 per 16, 27
fraction bits, a 36-bit window, round toward zero, floor −174): NVFP4's forming, `C̃ = cTilde`, U by `peelFp4`,
`Fp4Skip`'s lone-skip flags, the sub-grid flags and `saltDead4`, at `Fp4Prices.sm120`. -/
def devFp4Sm120 : Fp4Device where
  prices := Fp4Prices.sm120
  form := fun α β x e F => formRow α β x e F
  ticket := fun A B => ticket4 sm120Nvf4 A B
  useful := fun A B FA FB beA beB i j => (peelFp4 sm120Nvf4 A B FA FB beA beB i j).bind wordQ
  flags := fun A B i j t => match A i, B j with
    | some a, some b => Fp4Skip.fp4PureFlags sm120Nvf4 0 a.1 b.1 a.2 b.2 t
    | _, _ => false
  subgrid := fun A B => subgridFlags4 sm120Nvf4 A B
  saltDead := saltDead4

/-! ## The accounting -/

/-- **The credited work of one unit at an FP4 record**: the chain `m·k·n` at 1 per MAC (no promotion adds), the fused
`A′·F_B` (`m·k·r`), the peel (depth `2r` in BF16, `2·r·m·n` MACs at the record's BF16 price), and the salt-dependent
forming (`fs·m·k`). Per `m·k` at sm_120 it is `n + 32 + 256·n/k + fs`. -/
def creditFp4 (d : Fp4Device) (s : Shape) : ℚ :=
  (s.m : ℚ) * s.k * s.n + (s.m : ℚ) * s.k * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * s.m * s.n +
    d.prices.fs * s.m * s.k

/-- The honest reference's work at an FP4 record: the credited work plus the forming that depends on the activations
alone. -/
def wrefFp4 (d : Fp4Device) (s : Shape) : ℚ := creditFp4 d s + d.prices.qa * s.m * s.k

/-! ## What is not pinned yet, and the unit's maps -/

/-- One unit's noise lines, 32 E2M1 codes each: `E_A` (row `i`), `F_A` (position `t`), `E_B` (row `j`) and `F_B`
(position `t`). -/
structure Lines4 (m n k : ℕ) where
  EA : Matrix (Fin m) (Fin 32) Nib
  FA : Matrix (Fin k) (Fin 32) Nib
  EB : Matrix (Fin n) (Fin 32) Nib
  FB : Matrix (Fin k) (Fin 32) Nib

/-- What Pearl-C4's protocol takes that is not pinned yet, as `PearlCSem` does for FP8: the unit's noise lines from the
oracle, keyed by the seed tuple (SHA-512 of the seeds under `pearl-c4/v2/line`, four lines to a block, `pearl_c4.line`);
and a row's scale `α` and noise scale `β`
as FP32 values (`pearl_c4.stats`). -/
structure Fp4Sem (Q R S : Type) where
  lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
    Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k
  alpha : (k : ℕ) → (Fin k → ℚ) → ℚ
  beta : (k : ℕ) → (Fin k → ℚ) → ℚ

variable {Q R S : Type}

/-- **D-SK: `B̃` is keyed by this job's salt** (theory §1, "D-SK"; `pearl_c4.seed_b`), a domain rule of
TT_OUT-FP4. `seed_B = H_"pearl-c4/v1/seed-B"(salt ‖ root_B)` (SHA-512, `pearl_c4.labelled`), with the salt from this
job's epoch beacon and `root_B` as committed, and `B̃ = form(B, E_B, F_B)` with `E_B`, `F_B` the lines of `seed_B`.
- `seedB s W` is the query `seed_B` at salt `s` and B's committed words `W`, which `root_B` binds. It is injective in
  the salt, so a `B̃` keyed by another salt, or by none, is excluded.
- `drawB sh r` reads `E_B`'s `n` lines and `F_B`'s `k` lines from the oracle's answer `r` at `seed_B` (SHA-512 under
  `pearl-c4/v2/line` of `seed_B ‖ role ‖ ⌊index/4⌋`, line `index` its quarter `index mod 4`, `pearl_c4.line`). So `B̃` is a function of the salt and root_B alone, never of the
  activations, and never an input fixed before the salt.
- The ticket, U and the debit read `unitForm`'s `B̃`, never a prover-supplied one, and `Correct` compares a unit's words
  with theirs. So the verifier forms `B̃` itself on every drawn tile, and a transcript whose checked values differ from
  this `B̃`'s is rejected.

A variant that fixes `B̃` before A's salt is outside D₄: it needs the per-salt split debit instead. -/
def Fp4Sem.SaltKeyedB (sem : Fp4Sem Q R S) : Prop :=
  ∃ (seedB : S → Codes → Q) (drawB : (sh : Shape) → R →
      Matrix (Fin sh.n) (Fin 32) Nib × Matrix (Fin sh.k) (Fin 32) Nib),
    (∀ W : Codes, Function.Injective fun s : S => seedB s W) ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes),
      ((sem.lines H s U u act).EB, (sem.lines H s U u act).FB) =
        drawB (U.layout.shape u) (H (seedB s (U.weight (U.layout.wid u))))

/-- A committed operand of `p` rows and `q` columns passes Pearl-C4's rules: every word a widened BF16 word (its low
16 bits zero) and finite, every row passing `RowAdmit4`. -/
def Fp4Sem.OperandOK (sem : Fp4Sem Q R S) (p q : ℕ) (W : Codes) : Prop :=
  (∀ i < p, ∀ l < q, (wordQ (W i l)).isSome ∧ W i l % 2 ^ 16 = 0) ∧
    ∀ i : Fin p, RowAdmit4 sem.beta q (wordsVal p q W i)

/-- Row `i` of a committed operand passes Pearl-C4's rules: `Fp4Sem.OperandOK` on one row. -/
def Fp4Sem.RowOK (sem : Fp4Sem Q R S) (p q : ℕ) (W : Codes) (i : ℕ) : Prop :=
  (∀ l < q, (wordQ (W i l)).isSome ∧ W i l % 2 ^ 16 = 0) ∧
    ∀ h : i < p, RowAdmit4 sem.beta q (wordsVal p q W ⟨i, h⟩)

open Classical in
/-- A tile's credited rows: those whose activation row passes Pearl-C4's rules. -/
noncomputable def Fp4Sem.passRows (sem : Fp4Sem Q R S) (sh : Shape) (act : Codes) (rows : Finset ℕ) : Finset ℕ :=
  rows.filter fun i => sem.RowOK sh.m sh.k act i

/-- The unit's formed operands `A′` and `B̃` at the record's forming. -/
noncomputable def Fp4Sem.unitForm (sem : Fp4Sem Q R S) (d : Fp4Device) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (act : Codes) : Op4 (U.layout.shape u).m × Op4 (U.layout.shape u).n :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let W := wordsVal sh.n sh.k (U.weight (U.layout.wid u))
  let z := sem.lines H s U u act
  (fun i => d.form (sem.alpha sh.k (X i)) (sem.beta sh.k (X i)) (X i) (z.EA i) z.FA,
   fun j => d.form (sem.alpha sh.k (W j)) (sem.beta sh.k (W j)) (W j) (z.EB j) z.FB)

/-- **The unit's U** at the record: `d.useful` on the formed operands, the bases `F_A` and `F_B`, and the rows' `Ẽ` at
their `a_E = e4m3(β)`. -/
noncomputable def Fp4Sem.unitUseful (sem : Fp4Sem Q R S) (d : Fp4Device) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (act : Codes) : Matrix (Fin (U.layout.shape u).m) (Fin (U.layout.shape u).n) (Option ℚ) :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let W := wordsVal sh.n sh.k (U.weight (U.layout.wid u))
  let z := sem.lines H s U u act
  let AB := sem.unitForm d H s U u act
  d.useful AB.1 AB.2 z.FA z.FB (fun i => eTilde (z.EA i) (e4m3 (sem.beta sh.k (X i))))
    (fun j => eTilde (z.EB j) (e4m3 (sem.beta sh.k (W j))))

open Classical in
/-- **A tile's replayed debit at an FP4 record** (theory §2, with §6.2–6.5's base-split fix under its FP4 conditions;
`pearl_c4.tile_debit` at `9363e5012` is the executable reference), in FP4 units, over the words `rows × cols` of unit
`u`, from the committed activations, the weights and the realized noise (the `unitForm` the ticket reads, and its lines
`E_A`, `E_B`):
- (0) D-24's 2:4 windows (`windows`): per row and aligned 128 of `k` that is 2:4 (`Fp4Dev.rowWin24`), 64 per word the
  row feeds (half the window's 128 MACs), for `A′`'s rows and `B̃`'s;
- (1) **F1′** (§6.2, `split`; in place of §2's 2:4 term on salt-dead codes): `Fp4Dev.f1Row` of each row on every word it
  feeds, for `A′`'s rows at their `E_A` lines and overfit 1, and the salted `B̃`'s at their `E_B` lines and overfit
  `f1BOverfit` (the plain protocol, condition (8));
- (2) the identity atoms (`identity`): the flagged firings `t < k/64` of each word, at 64;
- (3) the dead pairs (`dead_pairs`): the positions `l < k` of each word whose `A′` element and `B̃` element are both
  salt-dead, at 1;
- (4) the forming (`forming`): each `A′` element of the rows that is salt-dead, or that is unneeded (every formed `B̃`
  row of the tile has a zero code there, `code & 7 = 0`), at `fs` times the column share `|cols|/n`. A row's screened
  blocks are not charged as such: their salt-dead codes are, here and in (3), and F1′ charges their other codes at
  their own `p_t`. This replaces D-SS's one-in-64 row rejection;
- (5) the sub-grid blocks (`subgrid`): the blocks `b < k/16` of each word that are sub-grid in their firing, at 16;
- (6) **F2** (§6.3, `int8`; condition (3)): `Fp4Dev.int8Saving` on the tile's realised scale bytes, `A′`'s rows and
  `B̃`'s, at the unit's `m` and `n`, per chain MAC of the tile (`|rows|·|cols|·k`). Its first reading, at the tile's
  modal byte, is positive only above `f_A + f_B = 2 − (1 − c_L)/4`: 1.9153 at 8,192³ (`cL_f2_threshold`), and at least
  1.817 inside the table (its least entry, 0.2697). The other two charge rows flat up to a per-row factor, and block
  columns that fall into a few flat classes (`int8Saving_readings`);
- (7) **R1** (§6.8; condition (4)): a row of `A′` or `B̃` whose own F1′ per word (`B̃`'s at its overfit) exceeds `ρ`
  times its cell credit (`creditFp4/(m·n)` per word) is charged that whole cell credit on every word of the tile it
  feeds, so it earns nothing.
  `ρ` is the ruled cap `1/400` (§6.6), since the debit doesn't take the protocol's `ρ`. The rule has the prover declare
  such a row and leaves it out of the cap. The debit can't take a declaration, and the caps stay as the price twins read
  them, so here the charge counts toward the cap: the credit is at most the rule's, and a tile of at most 64 credited
  rows holding such a row is rejected (its charge is at least `1/64` of the tile's credit).

The cap (condition (5)) is the protocol's and the tiles' `capOK`: a tile whose debit exceeds `ρ` times its credit is
rejected, not debited. The verifier replays (1), (6) and (7) from the drawn tile's own forming, as it does (2) and (5)
(condition (7)). -/
noncomputable def Fp4Sem.tileDebit (sem : Fp4Sem Q R S) (d : Fp4Device) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let W := wordsVal sh.n sh.k (U.weight (U.layout.wid u))
  let z := sem.lines H s U u act
  let AB := sem.unitForm d H s U u act
  let fl := d.flags AB.1 AB.2
  let sdA := fun (i : Fin sh.m) (l : Fin sh.k) => d.saltDead sem.alpha sem.beta sh.k (X i) l
  let sdB := fun (j : Fin sh.n) (l : Fin sh.k) => d.saltDead sem.alpha sem.beta sh.k (W j) l
  let I := (univ : Finset (Fin sh.m)).filter fun i : Fin sh.m => (i : ℕ) ∈ rows
  let J := (univ : Finset (Fin sh.n)).filter fun j : Fin sh.n => (j : ℕ) ∈ cols
  let wins := (J.card : ℚ) * ∑ i ∈ I, ((range (sh.k / 128)).filter fun w => rowWin24 (AB.1 i) w).card +
    (I.card : ℚ) * ∑ j ∈ J, ((range (sh.k / 128)).filter fun w => rowWin24 (AB.2 j) w).card
  let f1A := fun i : Fin sh.m =>
    f1Row 1 (sem.alpha sh.k (X i)) (sem.beta sh.k (X i)) (X i) (z.EA i) (sdA i) (AB.1 i)
  let f1B := fun j : Fin sh.n =>
    f1Row f1BOverfit (sem.alpha sh.k (W j)) (sem.beta sh.k (W j)) (W j) (z.EB j) (sdB j) (AB.2 j)
  let f1 := (J.card : ℚ) * ∑ i ∈ I, f1A i + (I.card : ℚ) * ∑ j ∈ J, f1B j
  let atoms := ∑ i ∈ I, ∑ j ∈ J, ((range (sh.k / 64)).filter fun t => fl i j t = true).card
  let pairs := ∑ i ∈ I, ∑ j ∈ J, ((univ : Finset (Fin sh.k)).filter fun l => sdA i l ∧ sdB j l).card
  let subs := ∑ i ∈ I, ∑ j ∈ J, ((range (sh.k / 16)).filter fun b => d.subgrid AB.1 AB.2 i j b = true).card
  let unneeded := fun l : Fin sh.k => ∀ j ∈ J, ∀ r, AB.2 j = some r → (r.1.getD l 0).val % 8 = 0
  let elems := ((univ : Finset (Fin sh.m × Fin sh.k)).filter fun e => (e.1 : ℕ) ∈ rows ∧
    (sdA e.1 e.2 ∨ unneeded e.2)).card
  let f2 := int8Saving sh.m sh.n (sh.k / 16) (tileBytes AB.1 I) (tileBytes AB.2 J) * ((I.card : ℚ) * J.card * sh.k)
  let cell := creditFp4 d sh / (sh.m * sh.n)
  let r1 := cell * ((J.card : ℚ) * (I.filter fun i => cell / 400 < f1A i).card +
    (I.card : ℚ) * (J.filter fun j => cell / 400 < f1B j).card)
  64 * wins + f1 + 64 * (atoms : ℚ) + (pairs : ℚ) + d.prices.fs * (cols.card / sh.n) * elems + 16 * (subs : ℚ) +
    f2 + r1

/-- **The unit's replayed debit at an FP4 record**: the tile debit over all its words. -/
noncomputable def Fp4Sem.unitDebit (sem : Fp4Sem Q R S) (d : Fp4Device) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (act : Codes) : ℚ :=
  sem.tileDebit d H s U u (range (U.layout.shape u).m) (range (U.layout.shape u).n) act

/-! ## The protocol, the domain and the tiles -/

/-- **Pearl-C4's protocol at an FP4 record**, with the debit cap `ρ`: `pearlCProtocolDev`'s shape on the record's maps;
the credit is `creditFp4` less the unit's replayed debit, and a unit whose debit exceeds `ρ·creditFp4` is rejected. -/
noncomputable def pearlCProtocolFp4 (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) : Protocol Q R S where
  ticket := fun H s U u act => atNat (d.ticket (sem.unitForm d H s U u act).1 (sem.unitForm d H s U u act).2)
  useful := fun H s U u act => atNat (sem.unitUseful d H s U u act)
  actOK := fun sh act => sem.OperandOK sh.m sh.k act
  weightOK := fun sh w j => sem.RowOK sh.n sh.k w j
  credit := fun H s U u act => ((creditFp4 d (U.layout.shape u) - sem.unitDebit d H s U u act : ℚ) : ℝ)
  capOK := fun H s U u act => sem.unitDebit d H s U u act ≤ ρ * creditFp4 d (U.layout.shape u)
  Wref := fun L u => (wrefFp4 d (L.shape u) : ℝ)

/-- **D₄'s shapes**: the reference's (`PearlC4.check`) with `n ≥ 4096` in place of its
`n ≥ 128`. `0 < m ≤ 2^24` and `4096 ≤ n ≤ 2^18`, both whole numbers of 64-row tiles, and `1024 ≤ k ≤ 2^16`, a whole
number of 128-deep kernel tiles. Every such unit is inside F2's `c_L` table, and B-OVF's credit discount is `0` from
`n = 4096` on. -/
def pearlCShapeFp4 (s : Shape) : Prop :=
  0 < s.m ∧ s.m ≤ 2 ^ 24 ∧ 64 ∣ s.m ∧ 4096 ≤ s.n ∧ s.n ≤ 2 ^ 18 ∧ 64 ∣ s.n ∧ 1024 ≤ s.k ∧ s.k ≤ 2 ^ 16 ∧ 128 ∣ s.k

/-- **D₄** (theory §2) at `sem`: D-SK of `sem`'s lines (`Fp4Sem.SaltKeyedB`), and every unit of a shape D₄ admits
(`pearlCShapeFp4`); no promotion (the record has none). It is empty at a `sem` whose `B̃` is not keyed by A's salt. The
row rules are the protocol's `actOK` and `weightOK`. -/
def pearlCDomainFp4 (sem : Fp4Sem Q R S) (L : Layout) : Prop :=
  sem.SaltKeyedB ∧ ∀ u < L.N, pearlCShapeFp4 (L.shape u)

/-- The domain restricted to one shape. -/
def pearlCDomainFp4At (sem : Fp4Sem Q R S) (s : Shape) (L : Layout) : Prop :=
  pearlCDomainFp4 sem L ∧ ∀ u < L.N, L.shape u = s

/-- **Pearl-C4's audit tiles at an FP4 record**, as `pearlCTilesDev` (credit on the credited rows), with `creditFp4`,
`tileDebit` and `wrefFp4`. -/
noncomputable def pearlCTilesFp4 (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) : TileRules Q R S where
  tiling := auditTiling 64 64
  rowOK := fun sh act i => sem.RowOK sh.m sh.k act i
  credit := fun _ _ U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    (((1 - ρ) * creditFp4 d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)
  capOK := fun H s U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    sem.tileDebit d H s U (tl.unit g) (sem.passRows sh act (tl.rows g)) (tl.cols g) act ≤
      ρ * creditFp4 d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g)
  Wref := fun L g =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefFp4 d sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
  Wcred := fun L g act =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefFp4 d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)

end Pouw.PearlC
