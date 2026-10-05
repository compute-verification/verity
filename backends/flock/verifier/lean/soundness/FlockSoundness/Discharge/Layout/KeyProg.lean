import FlockSoundness.Discharge.Layout.Site
import FlockSoundness.Discharge.Layout.OfRows
import FlockSoundness.Discharge.Layout.QSort
import FlockSoundness.Protocol.Discharge.Layout.KeyProg
import FlockSoundness.Discharge.Layout.Pins

set_option Elab.async false

/-!
# `lay` and `rs` at the circuit's program on Δ's source keys

`Placed.keyProg c hU n` is the program of `n` units of the circuit `c`'s VU rows, each unit's input `x` on the input gate
of its Δ key (`Aliased.keyIdx`): gate `2 + u·B + b` for unit `u` and the first input `b` with `x`'s key, gate 1 for the
zero, gate 0 for the constant. Its row bits (`keyRowOf`) are the input gates whose key is message bit `bit` of byte
`byte` of a row port `q` (`Aliased.Src.msg`): bit `8·byte + bit` of unit `u`'s row `q`. Every other wire is a single
gate. At [e2e-placed]'s places (`Placed.Sites.progPlaces_keyProg`):

* `keyRowOf_wire`: a column of unit `u`'s own instance on a row bit of unit `u'` is an input column of `u = u'` with that
  message key;
* `keyProg_copies`: Δ's copies (`copy_site` at each drawn unit's VU and block; each message column is in the block by
  the parse, `msgCol_lt_of_parse`);
* `keyProg_bcPinned`: the rows' `b ‖ c` is the registration's, the population's public file (`bc_site`), at each drawn
  unit's site (`BcSite`: the site's rank is the unit, and `bc_site`'s conditions; `bcSite_keyProg` gives the VU's bound
  from the parse when the unit returns the outputs, `vu_lt_pow2`);
* `keyProg_valPinned`: the single gates hold the registration's bits: 1 at the constant, 0 at the zero, and at the units'
  committed gates the registration's bit (`hout`, the returned outputs' pin), when every input a computed row reads has
  the zero or a row bit as its key (`KeysRow`).

`keyLayout` and `keyRows` are `lay` and `rs` there.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Model Audit Binding Placed

/-! ## The program's gates -/

section uniform

variable {k : ℕ} {R : FlockSoundness.Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one}

end uniform

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

variable (rowW : Fin c.ports.size → ℕ)

variable {c hU n rowW}

/-! ## At the program's places -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : Placed.Sites plan tab)

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n)))

variable (rowW) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (popOf : Reg → Public)
  (outBit : Reg → Fin (keyN c hU n) → Bool)

variable {rowW}

include hparse in
set_option linter.unusedVariables false in
/-- **`BcSite` at a site of `keyProg`'s places** in a circuit whose unit returns the outputs: the VU is below
`2^⌊log₂ c.g⌋` by the parse (`vu_lt_pow2`), the proved units are in the population by the draw's decoding
(`proved_lt_population`), and the other conditions are `BcSite`'s. -/
theorem bcSite_keyProg (ho : c.outNet = c.unitNet) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (h : u ∈ S)
    {pop : Public} {d : Lean.Json} {D : Draw.UnitDraw} (hd : (si.acc S R u h).draw = some d)
    (hD : Draw.UnitDraw.ofJson d = .ok D)
    (hregs : ∀ r ∈ (si.acc S R u h).st.regions.toList,
      Refine.regionOf (si.acc S R u h).st.c.kLog ((si.acc S R u h).st.m - (si.acc S R u h).st.c.kLog) r ∈
        (si.acc S R u h).regs)
    (hpop : HmRow.loadPublic (si.acc S R u h).tags (si.acc S R u h).st.c (si.acc S R u h).pf = .ok pop)
    (hpt : pop.tables = none) (hu : D.proved[(si.o S R u h).val * c.g + si.vu S R u h]? = some u.val) :
    BcSite (si.acc S R u h) pop (si.o S R u h) (si.vu S R u h) u.val := by
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have hv := si.hvu S R u h
  rw [hc] at hv
  refine ⟨d, D, hd, hD, proved_lt_population hD, hregs, hpop, ?_, ?_⟩
  · rw [hc]; exact vu_lt_pow2 (hparse S R u h) ho hv
  · rw [hc]; exact hu

end places

end FlockSoundness.Discharge.Layout
