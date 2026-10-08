import Definitions.Pouw.PearlC.WholeCircuit4
import Definitions.Pouw.PearlC.TileCheck
import Definitions.Pouw.PearlC.RowSem

/-!
# `pc4`'s tile check (definitions only)

The plain `pc4` Program as a `TileCheck`, as `TileCheck8` is `pc8`'s: at each unit shape `(m, k, n)` the circuit is
`Pc4Check{m, k, n, 64, 64}`'s table (`WholeCircuit.whole4`, the deployed tile `pc4.TILE`) and its partition the
units', with

* `out sh rows cols`: the `C̃` and `U` gates of the entries in `rows × cols`;
* `pub sh`: the noise lines `ea` and `eb`, the bases `fa` and `fb`, and `G`. Their anchors (`anchors4`) are the unit's
  lines (`lines`, the ones `pc4Sem` reads) as their E2M1 codes, and `G_{s,q}` the FP4 chain of `F_B`'s column `s`
  against `F_A`'s column `q` at scale `1.0`, the words `peelFp4` reads as `G` (`gram4`). That the verifier's public
  inputs are these is L8;
* `anchorsOK`: `True`, since E2M1 has no NaN code;
* `reg sh`: the activation rows `a`, the weight rows `b` and every entry's `C̃` and `U`. The rows are BF16 in the
  Program, so `a`'s and `b`'s gates hold the commitment's words shifted right by 16, each as the gate's low 16 bits:
  a widened BF16 word's BF16 word, `−0` included (`Fp4Sem.OperandOK` asks that every word be one), and `C̃` and `U`
  the ticket and useful words, each as the gate's low 32 bits (`rows4`);
* `Collides`: `False`, since the checked words are registered wires.

The Program needs whole tiles and `64 ∣ k` (`pc4.Shape.check`); at another shape the table is still defined, and L3 is
stated on whole-tile layouts.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit

variable {Q R S : Type}

/-- The ticket or useful word registered on a tile gate `g`: gate `g` is position `r` of entry `(ti, tj)`'s block in
tile `e = rs·(N / TN) + cs`, so of entry `(rs·TM + ti, cs·TN + tj)`; its `C̃` (the chain's last firing) is the ticket
word, its `U` the useful word, `0` elsewhere. -/
def entry4 (M N K TM TN : ℕ) (tk us : Codes) (g : ℕ) : ℕ :=
  if g < tiles4 M N K then 0
  else
    let e := (g - tiles4 M N K) / (tile4N K TM TN - tile4In K TM TN)
    let b := (g - tiles4 M N K) % (tile4N K TM TN - tile4In K TM TN)
    let i := e / (N / TN) * TM + b / block4Len K / TN
    let j := e % (N / TN) * TN + b / block4Len K % TN
    if b % block4Len K = K / 64 - 1 then tk i j
    else if b % block4Len K = block4Len K - 1 then us i j else 0

/-- The registered values on gate `g`: activation word `(i, t)`'s high half (its BF16 word) on `a`'s gates, weight word
`(j, t)`'s on `b`'s, the ticket and useful words on the tiles' `C̃` and `U` gates, `0` elsewhere. -/
def rows4 (M N K TM TN : ℕ) (act wt tk us : Codes) (g : ℕ) : ℕ :=
  if g < eaAt M K then act (g / K) (g % K) / 2 ^ 16
  else if g < bAt M K then 0
  else if g < ebAt M N K then wt ((g - bAt M K) / K) ((g - bAt M K) % K) / 2 ^ 16
  else entry4 M N K TM TN tk us g

/-- A line's E2M1 codes, `0` past its end. -/
def encNibs (e : Fin 32 → Nib) (l : ℕ) : ℕ := if h : l < 32 then (e ⟨l, h⟩).val else 0

/-- `G_{s,q}`: the FP4 chain of `F_B`'s column `s` against `F_A`'s column `q`, both at scale `1.0`, finite; `0` where it
fails. -/
def gram4 {m n k : ℕ} (z : Lines4 m n k) (s q : ℕ) : ℕ :=
  if h : s < 32 ∧ q < 32 then
    ((cTilde sm120Nvf4 (colF z.FB ⟨s, h.1⟩) (colF z.FA ⟨q, h.2⟩) (onesScales sm120Nvf4 k)
      (onesScales sm120Nvf4 k)).bind PeelExact.finite?).getD 0
  else 0

/-- The anchors on gate `g` from the unit's lines `z`: line `i`'s codes on `ea` and line `j`'s on `eb`, the bases' codes
on `fa` and `fb`, `G` on `g`, `0` elsewhere. -/
def anchors4 {m n k : ℕ} (z : Lines4 m n k) (g : ℕ) : ℕ :=
  if g < eaAt m k then 0
  else if g < bAt m k then
    if h : (g - eaAt m k) / 32 < m then encNibs (z.EA ⟨_, h⟩) ((g - eaAt m k) % 32) else 0
  else if g < ebAt m n k then 0
  else if g < faAt m n k then
    if h : (g - ebAt m n k) / 32 < n then encNibs (z.EB ⟨_, h⟩) ((g - ebAt m n k) % 32) else 0
  else if g < fbAt m n k then
    if h : (g - faAt m n k) / 32 < k then encNibs (z.FA ⟨_, h⟩) ((g - faAt m n k) % 32) else 0
  else if g < gAt m n k then
    if h : (g - fbAt m n k) / 32 < k then encNibs (z.FB ⟨_, h⟩) ((g - fbAt m n k) % 32) else 0
  else if g < in8 m n k then gram4 z ((g - gAt m n k) / 32) ((g - gAt m n k) % 32)
  else 0

/-- **`pc4`'s tile check**, with `ops` for core's primitives and the anchors from `lines`. -/
def tileCheck4 (ops : RowOps)
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k) : TileCheck Q R S where
  C sh := (whole4 ops sh.m sh.k sh.n 64 64).spec.toCircuit
  n sh := unitCount (whole4 ops sh.m sh.k sh.n 64 64).blocks
  part sh := (whole4 ops sh.m sh.k sh.n 64 64).part
  out sh rows cols := Finset.univ.filter fun g => ∃ i ∈ rows, ∃ j ∈ cols, i < sh.m ∧ j < sh.n ∧
    (g.val = cGlob4 sh.m sh.n sh.k 64 64 i j ∨ g.val = uGlob4 sh.m sh.n sh.k 64 64 i j)
  pub sh := Finset.univ.filter fun g => (eaAt sh.m sh.k ≤ g.val ∧ g.val < bAt sh.m sh.k) ∨
    (ebAt sh.m sh.n sh.k ≤ g.val ∧ g.val < in8 sh.m sh.n sh.k)
  reg sh := Finset.univ.filter fun g => g.val < eaAt sh.m sh.k ∨
    (bAt sh.m sh.k ≤ g.val ∧ g.val < ebAt sh.m sh.n sh.k) ∨
    ∃ i < sh.m, ∃ j < sh.n, g.val = cGlob4 sh.m sh.n sh.k 64 64 i j ∨ g.val = uGlob4 sh.m sh.n sh.k 64 64 i j
  anchorsOf H s U u act := fun g => BitVec.ofNat _ (anchors4 (lines H s U u act) g.val)
  anchorsOK _ _ _ _ _ := True
  rowsOf U u c := fun g => BitVec.ofNat _
    (rows4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k 64 64 c.act (U.weight (U.layout.wid u))
      c.ticket c.useful g.val)
  Collides _ _ _ _ _ _ := False

end Pouw.PearlC.WholeCircuit
