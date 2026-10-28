import Proofs.Pouw.PearlC.PeelExists
import Proofs.Pouw.PearlC.PeelExists4
import Proofs.Pouw.PearlC.TileCheck4Words
import Proofs.Pouw.PearlC.CoreOps
import Definitions.Pouw.Guarantees.PearlCTileCheck

/-!
# The plain `pc8` and `pc4` tile checks compute the words (proofs)

L3's words for the plain `pc8` and `pc4` Programs: on tile rules whose tiles lie in their unit and a domain of
whole-tile shapes (`Shape8`, `Shape4`), an accepted tile's ticket and useful words have the reference values on every
credited row, with `Collides` never used. Each composes its table's accepted entry (`entry8_accepts`, `entry4_accepts`)
with the rows' statistics from the rules. A credited row passes `RowOK` (the tile rules' `rowOK`, `hrow`), and each of
the tile's weight rows passes it too, since the check accepts only then (`TileCheck.Accepts`, the protocol's
`weightOK`). By magnitudes (`RowStats`) these give

* `pc8`: `Row8OK` (`row8OK_of_floor`), since `pc8Sem`'s floor is at least `1` only where `ρ`'s word is finite and
  `φ ≥ 0` (`stats8OK_of_rho`);
* `pc4`: `Row4OK` (`row4OK_of_admit`): the rules' word clause asks finite widened BF16 words, which widen back to
  their values (`widen_of_word`), and `RowAdmit4` holds only where `β`'s word is finite, with `0 ≤ β` and `e4m3(β)`
  at least code `8` (`row4OK_stats_of_beta`).

`pc8`'s noise has no NaN code (`Noise8OK`) because the check refuses one (`tileCheck8`'s `anchorsOK`, as Python's
`fp8_to_f32` does), so it is no hypothesis. What stays named: `Pc8OpsAgree` and `Pc4OpsAgree` (core's primitives), and,
for `pc8`, `Pc8EntryOK` at each entry of a credited row (the peel's rows exist). `_pearlC` is each on the audit's tiles,
and `_domain` on `D_dev`'s whole-tile layouts and on D₄. At a step `⟨[32], w, z⟩` with `w ≥ 24` (`sm120E4m3`),
`D_dev`'s `k ≤ 2^16` and `Noise8OK` give `Pc8EntryOK` (`PeelExists`), so `tileCheck8_computesWords_wide` names no
hypothesis at an entry. `pc4` names none at an entry: the two rows' `Row4OK` and `k ≤ 2^16` give `Pc4EntryOK`
(`PeelExists.pc4EntryOK_of_rows`), and D₄'s shapes have `k ≤ 2^16`.

At core's instance `coreOps` both agreements follow from core's definitions (`CoreOps`), so
`tileCheck8_computesWords_core` and `tileCheck4_computesWords_core` name no hypothesis.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit

variable {Q R S : Type}

/-! ## Rows that pass the rules -/

section Rows

variable {ops : RowOps}

/-- **A `pc8` row of finite words whose floor passes is `Row8OK`**: `pc8Sem`'s floor is at least `1` only where `ρ`'s
word is finite and `φ ≥ 0`. -/
theorem row8OK_of_floor (hops : Pc8OpsAgree ops) (p : Pipeline)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r)
    {m K i : ℕ} {W : Codes} (hi : i < m) (hw : ∀ l < K, (wordQ (W i l)).isSome)
    (hf : 1 ≤ (pc8Sem ops p noise).floor K (wordsVal m K W ⟨i, hi⟩)) : Row8OK ops K (W i) := by
  refine ⟨hw, ?_⟩
  simp only [pc8Sem] at hf
  split_ifs at hf with hρ
  · have hf' : (1 : ℚ) ≤ PearlC.wordVal (floor8 ops (encRow K (rowQ K (W i))) K) + 1 := hf
    exact stats8OK_of_rho hops hw hρ (by linarith)
  · norm_num at hf

/-- **A `pc4` row of finite widened BF16 words that `RowAdmit4` admits is `Row4OK`**: its words widen back to their
values (`widen_of_word`), and `pc4Sem`'s `β` is `0` where `β`'s word is not finite, which `RowAdmit4` refuses
(`e4m3(0)` is code `0`). -/
theorem row4OK_of_admit (hops : Pc4OpsAgree ops)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k)
    {m K i : ℕ} {W : Codes} (hi : i < m) (hw : ∀ l < K, (wordQ (W i l)).isSome ∧ W i l % 2 ^ 16 = 0)
    (h : RowAdmit4 (pc4Sem ops lines).beta K (wordsVal m K W ⟨i, hi⟩)) : Row4OK ops K (wordsVal m K W ⟨i, hi⟩) := by
  obtain ⟨h0, h8, -⟩ := h
  simp only [pc4Sem] at h0 h8
  split_ifs at h0 h8 with hβ
  · obtain ⟨ha, hap, hb⟩ := row4OK_stats_of_beta hops.toPc8OpsAgree hβ h0 h8
    exact ⟨fun t => widen_of_word hops (hw t t.isLt).1 (hw t t.isLt).2, ha, hap, hb⟩
  · have h00 : (e4m3 0).val = 0 := by decide +kernel
    omega

end Rows

/-! ## `pc8`'s tile check computes the words -/

/-- **L3's words for the plain `pc8` Program** (`tileCheck8`) at the device record `devAt p 4 pr` and `pc8Sem`: on
tile rules whose tiles lie in their unit, whose credited rows pass `pc8Sem`'s rules (`hrow`) and whose domain has
whole-tile shapes (`Shape8`), an accepted tile's ticket and useful words have the reference values on every credited
row, under `Pc8OpsAgree` and the named hypothesis at each entry of a credited row of a noise without NaN codes
(`Pc8EntryOK`). The noise has none, since the check refuses one (`tileCheck8`'s `anchorsOK`, `Noise8OK`). `Collides` is
`False` and never needed. -/
theorem tileCheck8_computesWords {ops : RowOps} (hops : Pc8OpsAgree ops) (p : Pipeline) (pr : Prices) (ρ : ℚ)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r)
    (TR : TileRules Q R S) (D : UnitLayout → Prop)
    (hsh : ∀ L, D L → ∀ g < (TR.tiling L).NT, Shape8 (L.shape ((TR.tiling L).unit g)))
    (hunit : ∀ L, ∀ g < (TR.tiling L).NT, (TR.tiling L).unit g < L.N)
    (hrows : ∀ L g, ∀ i ∈ (TR.tiling L).rows g, i < (L.shape ((TR.tiling L).unit g)).m)
    (hcols : ∀ L g, ∀ j ∈ (TR.tiling L).cols g, j < (L.shape ((TR.tiling L).unit g)).n)
    (hrow : ∀ (sh : Shape) (act : Codes) (i : ℕ), TR.rowOK sh act i → (pc8Sem ops p noise).RowOK sh.m sh.k act i)
    (hent : ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) (i : Fin (U.layout.shape u).m)
      (j : Fin (U.layout.shape u).n), D U.layout → u < U.layout.N → Noise8OK (noise H s U u act) →
        TR.rowOK (U.layout.shape u) act i → Pc8EntryOK ops p (noise H s U u act) act (U.weight (U.layout.wid u)) i j) :
    (tileCheck8 ops p noise).ComputesWords (pearlCProtocolDev (devAt p 4 pr) (pc8Sem ops p noise) ρ) TR D := by
  intro U hU H s g hg c X ⟨hcone, hpub, hnoise, hreg, hwt⟩
  left
  intro i hi hok j hj
  have him := hrows U.layout g i hi
  have hjn := hcols U.layout g j hj
  obtain ⟨hM, hN, nw, hnw, hK⟩ := hsh U.layout hU.2 g hg
  have hW : (pc8Sem ops p noise).RowOK _ _ (U.weight (U.layout.wid ((TR.tiling U.layout).unit g))) j := hwt j hj
  have hR := hrow _ c.act i hok
  obtain ⟨w1, w2⟩ := entry8_accepts hops p pr hnw hK hM hN X _ c.act _ c.ticket c.useful ⟨i, him⟩ ⟨j, hjn⟩
    (fun v hv => Finset.mem_filter.2 ⟨Finset.mem_univ _, i, hi, j, hj, him, hjn, hv⟩) hcone
    (fun v hv h => hpub ⟨v, hv⟩ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
    (fun v hv h => hreg ⟨v, hv⟩ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
    (row8OK_of_floor hops p noise him hR.1 (hR.2 him).2)
    (row8OK_of_floor hops p noise hjn hW.1 (hW.2 hjn).2)
    hnoise (hent H s U _ c.act ⟨i, him⟩ ⟨j, hjn⟩ hU.2 (hunit _ g hg) hnoise hok)
  refine ⟨w1.trans ?_, w2.trans ?_⟩
  · show _ = atNat _ i j
    unfold atNat
    rw [dite_eq_left ⟨him, hjn⟩]
    rfl
  · show _ = atNat _ i j
    unfold atNat
    rw [dite_eq_left ⟨him, hjn⟩]
    rfl

/-- **L3's words for the plain `pc8` Program on the audit's tiles** (`pearlCTiles`, the `64 × 64` grid): on a domain
whose every unit has a whole-tile shape, `tileCheck8_computesWords` with the grid's tiles in their units. -/
theorem tileCheck8_computesWords_pearlC {ops : RowOps} (hops : Pc8OpsAgree ops) (p : Pipeline) (pr : Prices) (c : Costs)
    (ρ : ℚ) (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r)
    (D : UnitLayout → Prop) (hsh : ∀ L, D L → ∀ u < L.N, Shape8 (L.shape u))
    (hent : ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) (i : Fin (U.layout.shape u).m)
      (j : Fin (U.layout.shape u).n), D U.layout → u < U.layout.N → Noise8OK (noise H s U u act) →
        (pc8Sem ops p noise).RowOK (U.layout.shape u).m (U.layout.shape u).k act i →
        Pc8EntryOK ops p (noise H s U u act) act (U.weight (U.layout.wid u)) i j) :
    (tileCheck8 ops p noise).ComputesWords (pearlCProtocolDev (devAt p 4 pr) (pc8Sem ops p noise) ρ)
      (pearlCTiles (pc8Sem ops p noise) c ρ) D :=
  tileCheck8_computesWords hops p pr ρ noise _ D
    (fun L hL _ hg => hsh L hL _ (auditTiling_unit_lt 64 64 L hg)) (fun L _ hg => auditTiling_unit_lt 64 64 L hg)
    (auditTiling_rows_lt 64 64) (auditTiling_cols_lt 64 64) (fun _ _ _ h => h) hent

/-- **L3's words for the plain `pc8` Program on `D_dev`'s whole-tile layouts** and the audit's tiles, those whose every
unit's tile divides its shape (served decode's `m = 32` among them): `tileCheck8_computesWords_pearlC` with `Shape8`'s
`k` from `D_dev` (`shape8_of_pearlCDomainDev`). -/
theorem tileCheck8_computesWords_domain {ops : RowOps} (hops : Pc8OpsAgree ops) (p : Pipeline) (pr : Prices)
    (c : Costs) (ρ : ℚ) (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r)
    (hent : ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) (i : Fin (U.layout.shape u).m)
      (j : Fin (U.layout.shape u).n), pearlCDomainDev (devAt p 4 pr) U.layout → u < U.layout.N →
        Noise8OK (noise H s U u act) → (pc8Sem ops p noise).RowOK (U.layout.shape u).m (U.layout.shape u).k act i →
        Pc8EntryOK ops p (noise H s U u act) act (U.weight (U.layout.wid u)) i j) :
    (tileCheck8 ops p noise).ComputesWords (pearlCProtocolDev (devAt p 4 pr) (pc8Sem ops p noise) ρ)
      (pearlCTiles (pc8Sem ops p noise) c ρ) (fun L => pearlCDomainDev (devAt p 4 pr) L ∧
        ∀ u < L.N, tileSide (L.shape u).m ∣ (L.shape u).m ∧ tileSide (L.shape u).n ∣ (L.shape u).n) :=
  tileCheck8_computesWords_pearlC hops p pr c ρ noise _
    (fun _ hL u hu => shape8_of_pearlCDomainDev hL.1 hu (hL.2 u hu).1 (hL.2 u hu).2)
    fun H s U u act i j hD hu => hent H s U u act i j hD.1 hu

/-- **L3's words for the plain `pc8` Program at a step `⟨[32], w, z⟩` with `w ≥ 24`** (`sm120E4m3` among them), on
`D_dev`'s whole-tile layouts: `tileCheck8_computesWords_domain` with no hypothesis at an entry, since `D_dev`'s
`k ≤ 2^16` and the accepted noise's `Noise8OK` give the peel's rows (`PeelExists.pc8EntryOK_of_wide`). -/
theorem tileCheck8_computesWords_wide {ops : RowOps} (hops : Pc8OpsAgree ops) {w : ℕ} (hw : 24 ≤ w) (zf : ℤ)
    (pr : Prices) (c : Costs) (ρ : ℚ) (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r) :
    (tileCheck8 ops ⟨[32], w, zf⟩ noise).ComputesWords
      (pearlCProtocolDev (devAt ⟨[32], w, zf⟩ 4 pr) (pc8Sem ops ⟨[32], w, zf⟩ noise) ρ)
      (pearlCTiles (pc8Sem ops ⟨[32], w, zf⟩ noise) c ρ) (fun L => pearlCDomainDev (devAt ⟨[32], w, zf⟩ 4 pr) L ∧
        ∀ u < L.N, tileSide (L.shape u).m ∣ (L.shape u).m ∧ tileSide (L.shape u).n ∣ (L.shape u).n) :=
  tileCheck8_computesWords_domain hops _ pr c ρ noise fun _ _ _ u act i j hD hu hz _ =>
    PeelExists.pc8EntryOK_of_wide hw zf (hD u hu).2.2.2.2.2 hz act _ i j

/-! ## `pc4`'s tile check computes the words -/

/-- **L3's words for the plain `pc4` Program** (`tileCheck4`) at the record `devFp4Sm120` and `pc4Sem`: on tile rules
whose tiles lie in their unit, whose credited rows pass `pc4Sem`'s rules (`hrow`) and whose domain has whole-tile
shapes (`Shape4`) with `k ≤ 2^16` (`hk16`), an accepted tile's ticket and useful words have the reference values on
every credited row, under `Pc4OpsAgree`. The peel's rows exist at each entry of a credited row, since both rows are
`Row4OK` (`PeelExists.pc4EntryOK_of_rows`). `Collides` is `False` and never needed. -/
theorem tileCheck4_computesWords {ops : RowOps} (hops : Pc4OpsAgree ops) (ρ : ℚ)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k)
    (TR : TileRules Q R S) (D : UnitLayout → Prop)
    (hsh : ∀ L, D L → ∀ g < (TR.tiling L).NT, Shape4 (L.shape ((TR.tiling L).unit g)))
    (hk16 : ∀ L, D L → ∀ g < (TR.tiling L).NT, (L.shape ((TR.tiling L).unit g)).k ≤ 2 ^ 16)
    (hrows : ∀ L g, ∀ i ∈ (TR.tiling L).rows g, i < (L.shape ((TR.tiling L).unit g)).m)
    (hcols : ∀ L g, ∀ j ∈ (TR.tiling L).cols g, j < (L.shape ((TR.tiling L).unit g)).n)
    (hrow : ∀ (sh : Shape) (act : Codes) (i : ℕ), TR.rowOK sh act i → (pc4Sem ops lines).RowOK sh.m sh.k act i) :
    (tileCheck4 ops lines).ComputesWords (pearlCProtocolFp4 devFp4Sm120 (pc4Sem ops lines) ρ) TR D := by
  intro U hU H s g hg c X ⟨hcone, hpub, _, hreg, hwt⟩
  left
  intro i hi hok j hj
  have him := hrows U.layout g i hi
  have hjn := hcols U.layout g j hj
  obtain ⟨hM, hN, nw, hnw, hK⟩ := hsh U.layout hU.2 g hg
  have hW : (pc4Sem ops lines).RowOK _ _ (U.weight (U.layout.wid ((TR.tiling U.layout).unit g))) j := hwt j hj
  have hR := hrow _ c.act i hok
  have hRa := row4OK_of_admit hops lines him hR.1 (hR.2 him)
  have hRb := row4OK_of_admit hops lines hjn hW.1 (hW.2 hjn)
  have hE := PeelExists.pc4EntryOK_of_rows hops ⟨nw, hK⟩ (hk16 U.layout hU.2 g hg) (lines H s U _ c.act) c.act _
    ⟨i, him⟩ ⟨j, hjn⟩ hRa hRb
  obtain ⟨w1, w2⟩ := entry4_accepts hops hnw hK hM hN X _ c.act _ c.ticket c.useful ⟨i, him⟩ ⟨j, hjn⟩
    (fun v hv => Finset.mem_filter.2 ⟨Finset.mem_univ _, i, hi, j, hj, him, hjn, hv⟩) hcone
    (fun v hv h => hpub ⟨v, hv⟩ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
    (fun v hv h => hreg ⟨v, hv⟩ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
    (fun l hl => (hR.1 l hl).1) (fun l hl => (hW.1 l hl).1) hRa hRb hE.peel
  refine ⟨w1.trans ?_, w2.trans ?_⟩
  · show _ = atNat _ i j
    unfold atNat
    rw [dite_eq_left ⟨him, hjn⟩]
    rfl
  · show _ = atNat _ i j
    unfold atNat
    rw [dite_eq_left ⟨him, hjn⟩]
    rfl

/-- **L3's words for the plain `pc4` Program on the audit's tiles** (`pearlCTilesFp4`, the `64 × 64` grid): on a
domain whose every unit has a whole-tile shape with `k ≤ 2^16`, `tileCheck4_computesWords` with the grid's tiles in
their units. -/
theorem tileCheck4_computesWords_pearlC {ops : RowOps} (hops : Pc4OpsAgree ops) (ρ : ℚ)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k)
    (D : UnitLayout → Prop) (hsh : ∀ L, D L → ∀ u < L.N, Shape4 (L.shape u))
    (hk16 : ∀ L, D L → ∀ u < L.N, (L.shape u).k ≤ 2 ^ 16) :
    (tileCheck4 ops lines).ComputesWords (pearlCProtocolFp4 devFp4Sm120 (pc4Sem ops lines) ρ)
      (pearlCTilesFp4 devFp4Sm120 (pc4Sem ops lines) ρ) D :=
  tileCheck4_computesWords hops ρ lines _ D
    (fun L hL _ hg => hsh L hL _ (auditTiling_unit_lt 64 64 L hg))
    (fun L hL _ hg => hk16 L hL _ (auditTiling_unit_lt 64 64 L hg))
    (auditTiling_rows_lt 64 64) (auditTiling_cols_lt 64 64) (fun _ _ _ h => h)

/-- **L3's words for the plain `pc4` Program on D₄** (`pearlCDomainFp4`) and the audit's tiles:
`tileCheck4_computesWords_pearlC` with `Shape4` and `k ≤ 2^16` from D₄'s shapes, so it names no hypothesis beyond
`Pc4OpsAgree`. -/
theorem tileCheck4_computesWords_domain {ops : RowOps} (hops : Pc4OpsAgree ops) (ρ : ℚ)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k) :
    (tileCheck4 ops lines).ComputesWords (pearlCProtocolFp4 devFp4Sm120 (pc4Sem ops lines) ρ)
      (pearlCTilesFp4 devFp4Sm120 (pc4Sem ops lines) ρ) (pearlCDomainFp4 (pc4Sem ops lines)) :=
  tileCheck4_computesWords_pearlC hops ρ lines _ (fun _ hL u hu => shape4_of_pearlCShapeFp4 (hL.2 u hu))
    fun _ hL u hu => (hL.2 u hu).2.2.2.2.2.2.2.1

/-! ## At core's instance -/

/-- **L3's words for the plain `pc8` Program at core's instance** (`coreOps`), at a step `⟨[32], w, z⟩` with `w ≥ 24`
(`sm120E4m3` among them), on `D_dev`'s whole-tile layouts and the audit's tiles: `tileCheck8_computesWords_wide` with
`Pc8OpsAgree` from core's definitions (`CoreOps.pc8OpsAgree_coreOps`), so it names no hypothesis. -/
theorem tileCheck8_computesWords_core {w : ℕ} (hw : 24 ≤ w) (zf : ℤ) (pr : Prices) (c : Costs) (ρ : ℚ)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r) :
    (tileCheck8 coreOps ⟨[32], w, zf⟩ noise).ComputesWords
      (pearlCProtocolDev (devAt ⟨[32], w, zf⟩ 4 pr) (pc8Sem coreOps ⟨[32], w, zf⟩ noise) ρ)
      (pearlCTiles (pc8Sem coreOps ⟨[32], w, zf⟩ noise) c ρ) (fun L => pearlCDomainDev (devAt ⟨[32], w, zf⟩ 4 pr) L ∧
        ∀ u < L.N, tileSide (L.shape u).m ∣ (L.shape u).m ∧ tileSide (L.shape u).n ∣ (L.shape u).n) :=
  tileCheck8_computesWords_wide CoreOps.pc8OpsAgree_coreOps hw zf pr c ρ noise

/-- **L3's words for the plain `pc4` Program at core's instance** (`coreOps`) on D₄ and the audit's tiles:
`tileCheck4_computesWords_domain` with `Pc4OpsAgree` from core's definitions (`CoreOps.pc4OpsAgree_coreOps`), so it
names no hypothesis. -/
theorem tileCheck4_computesWords_core (ρ : ℚ)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k) :
    (tileCheck4 coreOps lines).ComputesWords (pearlCProtocolFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ)
      (pearlCTilesFp4 devFp4Sm120 (pc4Sem coreOps lines) ρ) (pearlCDomainFp4 (pc4Sem coreOps lines)) :=
  tileCheck4_computesWords_domain CoreOps.pc4OpsAgree_coreOps ρ lines

end Pouw.PearlC.WholeCircuit

theorem Pouw.SecurityProofs.PearlC.TileCheck8WordsCore : Pouw.Guarantees.PearlC.TileCheck8WordsCore := by
  intro Q R S p pr c ρ noise U H s g cm X hacc hgr hw hU hg
  obtain ⟨l, w, zf⟩ := p
  obtain rfl : l = [32] := hgr
  exact (Pouw.PearlC.WholeCircuit.tileCheck8_computesWords_core hw zf pr c ρ noise U hU H s g hg cm X hacc).resolve_right id

theorem Pouw.SecurityProofs.PearlC.TileCheck4WordsCore : Pouw.Guarantees.PearlC.TileCheck4WordsCore := by
  intro Q R S ρ lines U H s g cm X hacc hU hg
  exact (Pouw.PearlC.WholeCircuit.tileCheck4_computesWords_core ρ lines U hU H s g hg cm X hacc).resolve_right id
