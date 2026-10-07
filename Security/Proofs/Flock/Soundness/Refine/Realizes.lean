import Proofs.Flock.Soundness.Refine.Tapes

/-!
# How the executable's statement denotes a model statement

The model's verifier is written for a `Model.Statement`: a block R1CS with matrices `A₀`, `B₀`, a pinned column and
public regions (PROTOCOL.md §16.1–§16.2). The executable's verifier never holds the matrices. The lincheck calls the
statement's circuit fold, which computes `α·A₀ᵀe + B₀ᵀe` from the statement's sparse layout (`Stmt.fold`).

`FoldRealizes c S` says the executable's fold `c` computes the model statement `S`'s. It is a fact about the statement
builder, not a verifier check. The refinement takes it as a hypothesis, and the statement decoding discharges it for
`Setup.ofCircuit` (level 3's `fold_get` describes the fold entry by entry).
-/

namespace FlockSoundness.Refine

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **The circuit fold realizes the statement** (§10, §16.1). The pinned column and `k_log` are the statement's, and on
64 skip weights `lw` and `k_log − 6` block coordinates `ρ`, a fold that succeeds has, at every column `c < 2^k_log`,
`Σ_i e_i·(α·A₀[i, c] + B₀[i, c])` for `e_i = lw[i mod 64]·eq(ρ, ⌊i/64⌋)`. A `folded` that succeeds is such a fold, of
`2^k_log` entries, with `β` added at the pin and halved at the challenges `ts` in turn. -/
structure FoldRealizes (c : CircuitFold) (S : Model.Statement) : Prop where
  kLog : c.kLog = S.kLog
  pin : c.pin = S.pin.val
  fold : ∀ (α : F128) (lw ρ v : Array F128), lw.size = 64 → ρ.size = S.kLog - 6 → c.fold α lw ρ = .ok v →
    ∀ col : Fin (2 ^ S.kLog), fOf v[col.val]! = ∑ i : Fin (2 ^ S.kLog),
      fOf lw[i.val % 64]! * Model.eqAt (vecOf (S.kLog - 6) ρ) (i.val / 64) *
        (fOf α * ((S.A₀ i col).val : 𝔽) + ((S.B₀ i col).val : 𝔽))
  folded : ∀ (α β : F128) (lw ρ ts w : Array F128), c.folded α β lw ρ ts = .ok w →
    ∃ v, c.fold α lw ρ = .ok v ∧ v.size = 2 ^ S.kLog ∧
      w = ts.foldl (fun acc t => halve t acc) (v.modify c.pin (· + β))

end FlockSoundness.Refine
