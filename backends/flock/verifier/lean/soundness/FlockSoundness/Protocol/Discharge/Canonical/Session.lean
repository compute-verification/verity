import FlockSoundness.Protocol.Discharge.Canonical.Rows
import FlockSoundness.Protocol.Discharge.Canonical.Sites
import FlockSoundness.Assumptions.Zk

set_option Elab.async false

/-!
# One canonical hidden-output `--zk` sampled session of `J` tables

`ZkHidden/DefsJ.lean`'s shadow and live pieces, `SoundJ.lean`'s `zk_session_soundHJ`, and `CustodyJ.lean` /
`CustodyJJson.lean`'s custody at a circuit in the canonical format (`formatOk c = true`) in place of
`Layout.scopeOk c = true` and `pub.tables = none`:

* `drawSetupH_of_setupHiddenC`: `drawSetupH_of_setupHidden` without `outNet = unitNet` (`0 < c.g` is `formatOk`'s
  `vusPow2`, `formatOk_g_pos`);
* `layShC`, `rsShC`, `rsShC_computes`, `placeZC`, `decAgreeC`, `layZC`, `rsZC`, `rsZC_computes`: `keyLayoutC` and
  `keyRowsC` (`Rows.lean`) at `sitesShC` (`Sites.lean`), moved to the `--zk` tables as `layZHJ`, `rsZHJ`;
* `runsZCJH_liveC`: `runsZCJH_live` without `outNet = unitNet`;
* `zk_session_soundC`, `zk_session_soundC_custody_json`: the session's soundness, `zk_session_soundHJ`'s proof at these
  pieces, and at the record's custody and the printed draw file.

**Proof work.** None: every declaration is proved, and so is every lemma it uses from `Format`, `Chain`, `Commit`,
`Reads`, `Outputs`, `Sites` and `Rows`. `zk_session_soundC`'s bound and its assumptions (A3, `cr/sha-512` through
`TableCRZ`, the count curve's parameters) are `zk_session_soundHJ`'s; only the circuit's condition changes.
-/

namespace FlockSoundness.Discharge.Canonical

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden StrictCR
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

section Accepted

variable {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hfmt : formatOk c = true)
include hc hfmt

end Accepted

variable {n mPts : ℕ} {Reg : Type}

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hfmt : formatOk c = true) (hU : Built c.unit) {pub : Public}
  (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

/-- **`lay` at the public plan**: `keyLayoutC` at `sitesShC`. -/
noncomputable abbrev layShC :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShC_parse dj y₀ ht hc hfmt) (Layout.outsOf c hU n))) :=
  keyLayoutC (sitesShC_parse dj y₀ ht hc hfmt) (Layout.outsOf c hU n) (fun _ => pub) hfmt
    (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan**: `keyRowsC` at `sitesShC`, with `hsite` from `sitesShC_bcSite` and the keys from
`keysRowC_of_parse`. -/
noncomputable abbrev rsShC : Binding.HmRows (layShC dj y₀ ht hc hfmt hU (pub := pub) (Reg := Reg)) :=
  keyRowsC (sitesShC_parse dj y₀ ht hc hfmt) (Layout.outsOf c hU n) (fun _ => pub) hfmt
    (Layout.outsOut_outsOf c hU n) (sitesShC_bcSite dj y₀ ht hc hfmt hpub)
    (keysRowC_of_parse hU (HiddenExec.parse_untyped (zkTyped ht) hc).1 hfmt)

/-- **Each drawn unit's place in its canonical `--zk` table** (`placeZHJ` at `sitesShC`). -/
noncomputable def placeZC (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S u :=
  placeCast (planZHJ_S dj y₀ S R u)
    (keyPlaceH (sitesShC_parse dj y₀ ht hc hfmt) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree** (`decAgreeHJ`). -/
theorem decAgreeC :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
        (tabZHJ dj y₀) (placeZC dj y₀ ht hc hfmt hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShC_parse dj y₀ ht hc hfmt) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZC, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the canonical `--zk` tables** (`layZHJ`). -/
noncomputable abbrev layZC :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
      (tabZHJ dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        (placeZC dj y₀ ht hc hfmt hU)) :=
  LayoutZ.ofShadow (planZHJ_S dj y₀) (layShC dj y₀ ht hc hfmt hU (pub := pub)) (decAgreeC dj y₀ ht hc hfmt hU)

/-- **`rs` at the canonical `--zk` tables** (`rsZHJ`). -/
noncomputable abbrev rsZC : HmRowsZ (layZC dj y₀ ht hc hfmt hU (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planZHJ_S dj y₀) (decAgreeC dj y₀ ht hc hfmt hU) (rsShC dj y₀ ht hc hfmt hU hpub)

end Shadow

end FlockSoundness.Discharge.Canonical
