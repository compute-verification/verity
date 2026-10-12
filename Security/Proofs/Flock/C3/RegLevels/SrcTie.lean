import Proofs.Flock.C3.Compose.Decoder
import Proofs.Flock.C3.Opening.SrcOpening

/-!
# n7's levels: `RegTape` at the source, from R's Ligerito tied to an accepting run

`regLevelsRead_c3DecOf` asks for `RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR (runSteps u) u)` at the
source's run `u` at every accepted outer proof. Past R's opening a run's own steps are its Ligerito and the openings
(`tapeR_runSteps`), so `RegTape`'s tape clauses hold at `ts := u.lg` and the openings `u.os`, and the rest of it is
`LevelTie` (`regTape_of_levelTie`). R's Ligerito verifies against the registered cap, so the source's tie is at
`xR.cap` (`SrcLevelTie`), and with `SrcOK` it gives `RegLevelsRead` at the fixed decoder (`regLevelsRead_of_tie`).
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)

/-- **A run's own steps past R's opening** are its Ligerito and the openings. -/
theorem tapeR_runSteps {e : ExecHolo} (u : C3Run e) :
    tapeR (runSteps u) u = u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩] := by
  change (FlockC3.Opening.afterSparse u (runSteps u)).drop u.op.length = _
  rw [FlockC3.Opening.afterSparse_runSteps, List.drop_left]

/-- **R's Ligerito and the openings are an accepting Ligerito run's** (`RegTape`'s witnesses at `ts := u.lg` and the
openings `u.os`): at a `fast100` schedule `sch` whose model schedule is R's registered one, against a proof `p` whose
level-0 cap `cap` has the digests `cap₀`, at ring switching's `T` and switched claims `sws`, which are R's opening's
output (`OpenRel`), the executable's Ligerito accepts from the well-formed record `rt` to `rt'`; `u.lg` takes the kinds
of compiled Ligerito's shape, its messages are the proof's and its coin steps the rounds the run consumes, and the
openings' rep-2 rows and siblings are the proof's. -/
def LevelTie (cap₀ : Cap Dig) (e : ExecHolo) (u : C3Run e) : Prop :=
  ∃ (msId : String) (m : ℕ) (sch : Flock.Schedule) (Tw : Flock.F128) (sws : Array Flock.Switched) (nc : ℕ)
    (cap : Array ByteArray) (p : Flock.LigeritoProof) (rt rt' : Flock.Transcript),
    Flock.fast100 m = some sch ∧ Refine.schedOf sch = e.rsch.sch ∧ capDig cap = cap₀ ∧
      OpenRel sch.m nc Tw sws u.opened ∧ RoundsWF rt.rounds ∧
      Flock.ligerito (Flock.MerkleScheme.sha512 msId) sch Tw cap p (Flock.basis sws) rt = .ok ((), rt') ∧
      u.lg.map kindOf <+: ligShape GF128 GF256 Dig e.rsch.sch ∧
      u.lg.filterMap Sum.getLeft? = ligMsgs p capDig sch ∧
      u.lg.filterMap Sum.getRight? = (consumed rt.rounds rt.k rt'.k).map (coinArr fOf) ∧
      ∀ ℓ j, u.os.1 2 ℓ j = openRow p sch ℓ j

/-- **(n7's levels) The tie gives `RegTape` on the run's own steps**: past R's opening they are `u.lg` and then the
openings' step (`tapeR_runSteps`). -/
theorem regTape_of_levelTie {cap₀ : Cap Dig} {e : ExecHolo} {u : C3Run e} (h : LevelTie cap₀ e u) :
    RegTape cap₀ e.rsch.sch u.opened (tapeR (runSteps u) u) := by
  obtain ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', hf, hsch, hcap, hop, hwf, hlig, hk, hl, hr, ho⟩ := h
  rw [tapeR_runSteps]
  exact ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', u.lg, u.os, hf, hsch, hcap, hop, hwf, hlig,
    List.prefix_append _ _, hk, hl, hr, by rw [List.drop_left]; rfl, ho⟩

noncomputable section

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The source's levels tied to an accepting Ligerito run against `cap₀`**, at every accepted outer proof. -/
def SrcLevelTie {S : ℕ} {Vs : Fin S → VStmt}
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (cap₀ : Cap Dig) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => LevelTie cap₀ (holoAt st hp) u

/-- **`regLevelsRead_c3DecOf`'s argument**, at a source tied to accepting Ligerito runs against the registered cap. -/
theorem regTapeAt_src {S : ℕ} {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : SrcLevelTie zs dg N k Rw src xR.cap) :
    AtSrc zs dg N k Rw src fun _ u => RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR (runSteps u) u) :=
  fun R ω t R₂ τ o hacc => regTape_of_levelTie (h R ω t R₂ τ o hacc)

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {tr : Lean.Json} {record : String} {reg : Lean.Json} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)}

/-- **(n7's levels) `RegLevelsRead` at the fixed decoder**, from the source's obligations and its levels' tie. -/
theorem regLevelsRead_of_tie (hs : SrcOK zs dg N k Rw src own sel) (h : SrcLevelTie zs dg N k Rw src xR.cap) :
    RegLevelsRead (C3Family.honest (xR := xR) ish hss bind hob zs own sel bc (c3DecOf zs dg N k Rw src own sel) tr
      record reg) dg N k Rw :=
  regLevelsRead_c3DecOf hs (regTapeAt_src h)

end

end FlockC3.RegLevels
