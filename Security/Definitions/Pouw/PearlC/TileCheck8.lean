import Definitions.Pouw.PearlC.WholeCircuit
import Definitions.Pouw.PearlC.TileCheck
import Definitions.Pouw.PearlC.RowSem
import Definitions.Pouw.PearlC.PeelExactP

/-!
# `pc8`'s tile check (definitions only)

The plain `pc8` Program as a `TileCheck`: at each unit shape `(m, k, n)` the circuit is `Pc8Check{m, k, n, TM, TN}`'s
table (`WholeCircuit.whole8`) at the audit grid's tile at that shape (`tileSide`): `TM = min(m, 64)` and
`TN = min(n, 64)`, so `pc8.TILE`'s `64 × 64`, or decode's `32 × 64`. Its partition is the units', with

* `out sh rows cols`: the `C̃` and `U` gates of the entries in `rows × cols`;
* `pub sh`: the noise lines `ea` and `eb`, the bases `fa` and `fb`, and `G`. Their anchors (`anchors8`) are the unit's
  noise (`noise`, the same the semantics reads) as E4M3 codes, and `G_{s,q}` the chain on the step `p` of `F_B`'s column
  `s` against `F_A`'s column `q`, the words `PeelExactP.peelExactP` reads as `G`. That the verifier's public inputs are
  these is L8;
* `anchorsOK`: the noise has no NaN code (`Noise8OK`): the verifier refuses a tile whose noise has one, as Python's
  does (`fp8_to_f32` refuses a NaN code);
* `reg sh`: the activation rows `a`, the weight rows `b` and every entry's `C̃` and `U`, at the commitment's
  activations, its unit's weights, and its ticket and useful words (`rows8`), each as the gate's low 32 bits;
* `Collides`: `False`, since the checked words are registered wires.

The Program needs whole tiles (`pc8.Shape.check`: `TM ∣ m`, `TN ∣ n`), so a shape whose last tile is narrower than
the others (`m = 80`: a 64-row tile and a 16-row one) is outside it. There the table is still defined, and L3 is
stated on shapes whose tile divides them.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit Pouw.PearlC.PeelExactP

variable {Q R S : Type}

/-- The side of the audit grid's tile along a dimension of `d` (`grid_tiles(shape, 64, 64)`): `64`, or `d` itself
below it. -/
def tileSide (d : ℕ) : ℕ := min d 64

/-- The first gate of the tiles, after the row units. -/
def tiles8 (p : Pipeline) (M N K : ℕ) : ℕ :=
  in8 M N K + N * ((row8B p K).N - rowIn K true) + M * ((row8A p K).N - rowIn K false)

/-- The ticket or useful word registered on a tile gate `g`: gate `g` is position `r` of entry `(ti, tj)`'s block in
tile `e = rs·(N / TN) + cs`, so of entry `(rs·TM + ti, cs·TN + tj)`; its `C̃` is the ticket word, its `U` the useful
word, `0` elsewhere. -/
def entry8 (p : Pipeline) (M N K TM TN : ℕ) (tk us : Codes) (g : ℕ) : ℕ :=
  if g < tiles8 p M N K then 0
  else
    let e := (g - tiles8 p M N K) / (tileN K TM TN - tileIn K TM TN)
    let b := (g - tiles8 p M N K) % (tileN K TM TN - tileIn K TM TN)
    let i := e / (N / TN) * TM + b / blockLen K / TN
    let j := e % (N / TN) * TN + b / blockLen K % TN
    if b % blockLen K = 5 * (K / 128) - 1 then tk i j
    else if b % blockLen K = blockLen K - 1 then us i j else 0

/-- The registered values on gate `g`: activation word `(i, t)` on `a`'s gates, weight word `(j, t)` on `b`'s, the
ticket and useful words on the tiles' `C̃` and `U` gates, `0` elsewhere. -/
def rows8 (p : Pipeline) (M N K TM TN : ℕ) (act wt tk us : Codes) (g : ℕ) : ℕ :=
  if g < eaAt M K then act (g / K) (g % K)
  else if g < bAt M K then 0
  else if g < ebAt M N K then wt ((g - bAt M K) / K) ((g - bAt M K) % K)
  else entry8 p M N K TM TN tk us g

/-- `G_{s,q}`: the chain on the step `p` of `F_B`'s column `s` against `F_A`'s column `q`, as E4M3 codes, `0` where it
fails. -/
def gram8 (p : Pipeline) {m n k : ℕ} (z : Noise m n k Params.pi.r) (s q : ℕ) : ℕ :=
  if h : s < 32 ∧ q < 32 then
    (chainWordP? p 4 (rowAtoms (fun l => z.FB l ⟨s, h.1⟩) (fun l => z.FA l ⟨q, h.2⟩))).getD 0
  else 0

/-- The anchors on gate `g` from the unit's noise `z`: line `i`'s codes on `ea` and line `j`'s on `eb`, the bases' codes
on `fa` and `fb`, `G` on `g`, `0` elsewhere. -/
def anchors8 (p : Pipeline) {m n k : ℕ} (z : Noise m n k Params.pi.r) (g : ℕ) : ℕ :=
  if g < eaAt m k then 0
  else if g < bAt m k then
    if h : (g - eaAt m k) / 32 < m then encCodes (z.EA ⟨_, h⟩) ((g - eaAt m k) % 32) else 0
  else if g < ebAt m n k then 0
  else if g < faAt m n k then
    if h : (g - ebAt m n k) / 32 < n then encCodes (z.EB ⟨_, h⟩) ((g - ebAt m n k) % 32) else 0
  else if g < fbAt m n k then
    if h : (g - faAt m n k) / 32 < k then encCodes (z.FA ⟨_, h⟩) ((g - faAt m n k) % 32) else 0
  else if g < gAt m n k then
    if h : (g - fbAt m n k) / 32 < k then encCodes (z.FB ⟨_, h⟩) ((g - fbAt m n k) % 32) else 0
  else if g < in8 m n k then gram8 p z ((g - gAt m n k) / 32) ((g - gAt m n k) % 32)
  else 0

/-- A noise line as the table reads it: no NaN code (`0x7F`, `0xFF`). -/
def Line8OK (e : Fin Params.pi.r → Code) : Prop := ∀ q, (e q).isNaN = false

/-- **The noise as the table reads it**: `Line8OK` at every line of its four factors. `pc8`'s check refuses the rest
(`tileCheck8`'s `anchorsOK`), as Python's verifier does: `fp8_to_f32` refuses a NaN code. Python's honest noise has
none (`sample_line`'s codes are casts that saturate at `0x7E`), which is completeness. -/
def Noise8OK {m n k : ℕ} (z : Noise m n k Params.pi.r) : Prop :=
  (∀ i, Line8OK (z.EA i)) ∧ (∀ l, Line8OK (z.FA l)) ∧ (∀ j, Line8OK (z.EB j)) ∧ ∀ l, Line8OK (z.FB l)

/-- **`pc8`'s tile check** on the step `p`, with `ops` for core's primitives and the anchors from `noise`, refusing a
noise with a NaN code. -/
def tileCheck8 (ops : RowOps) (p : Pipeline)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r) : TileCheck Q R S where
  C sh := (whole8 ops p sh.m sh.k sh.n (tileSide sh.m) (tileSide sh.n)).spec.toCircuit
  n sh := unitCount (whole8 ops p sh.m sh.k sh.n (tileSide sh.m) (tileSide sh.n)).blocks
  part sh := (whole8 ops p sh.m sh.k sh.n (tileSide sh.m) (tileSide sh.n)).part
  out sh rows cols := Finset.univ.filter fun g => ∃ i ∈ rows, ∃ j ∈ cols, i < sh.m ∧ j < sh.n ∧
    (g.val = cGlob8 p sh.m sh.n sh.k (tileSide sh.m) (tileSide sh.n) i j ∨
      g.val = uGlob8 p sh.m sh.n sh.k (tileSide sh.m) (tileSide sh.n) i j)
  pub sh := Finset.univ.filter fun g => (eaAt sh.m sh.k ≤ g.val ∧ g.val < bAt sh.m sh.k) ∨
    (ebAt sh.m sh.n sh.k ≤ g.val ∧ g.val < in8 sh.m sh.n sh.k)
  reg sh := Finset.univ.filter fun g => g.val < eaAt sh.m sh.k ∨
    (bAt sh.m sh.k ≤ g.val ∧ g.val < ebAt sh.m sh.n sh.k) ∨
    ∃ i < sh.m, ∃ j < sh.n, g.val = cGlob8 p sh.m sh.n sh.k (tileSide sh.m) (tileSide sh.n) i j ∨
      g.val = uGlob8 p sh.m sh.n sh.k (tileSide sh.m) (tileSide sh.n) i j
  anchorsOf H s U u act := fun g => BitVec.ofNat _ (anchors8 p (noise H s U u act) g.val)
  anchorsOK H s U u act := Noise8OK (noise H s U u act)
  rowsOf U u c := fun g => BitVec.ofNat _
    (rows8 p (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k (tileSide (U.layout.shape u).m)
      (tileSide (U.layout.shape u).n) c.act (U.weight (U.layout.wid u)) c.ticket c.useful g.val)
  Collides _ _ _ _ _ _ := False

end Pouw.PearlC.WholeCircuit
