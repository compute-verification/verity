import Proofs.Flock.Soundness.Discharge.DecodeZL.Wire
import Proofs.Flock.Soundness.Discharge.Exec.Rounds
import Proofs.Flock.Soundness.Discharge.Exec.Verdict
import Proofs.Flock.Soundness.Discharge.Exec.LiveOn
import Proofs.Flock.Soundness.Discharge.Integrate.Headline

/-!
# The message half of R11b, and the decoder at the headline: statements

`flock_headline_exec`'s open guarantee is `hdec : DecodesOn …`: at every outcome of the live session flock-verify
accepts, each rep's tape (its rounds' messages, decoded from their bytes, and coins) plays the rep's game, and the table's
verdict accepts. The statements here:

* `ZLCarries`: the zerocheck's and lincheck's rounds carry their messages by `wZL` (`Carries`, `Wire.lean`).
* `ProofSizes`, `DecodeSizes`: the lengths `decodeProof` checks, which make the message-carrying frames well formed.
* `OPCarries`, `LIGCarries`: the same chunk lemma for the opening's ring switches and for Ligerito, at a decoded
  proof of a `fast100` statement (owners: dec-open, dec-lig).
* `TagsOk`, `RecordsLive`, `DrawOk`: what `DecodesOn` needs past the plain headline's hypotheses. `wr` is a free
  parameter of the headline, and at a `wr` constant at an accepted transcript `DecodesOn` is false (an outcome with no
  rounds can't play `repC`, which starts with a coin); `RecordsLive` says `wr` is the challenger's record of the
  outcome.
* `decAt`, `opnAt`: the decoder and the openings at draw `S`, read off the table `T S R` and setupH's statement there.
* `DecodesOnExec`: `DecodesOn` at the plain headline's tables, the decoder `decAt`, `one := true`, no final messages,
  `capOfRoot`.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Integrate

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **The zerocheck's and lincheck's rounds carry their messages**: from the domain frame through `z_partial`'s round,
the rounds read by `wZL` are the zerocheck's then the lincheck's messages, cut at `zcShape ++ lcShape`'s coins, with no
tail. -/
def ZLCarries : Prop :=
  ∀ (m kLog : ℕ) (d dig cap : ByteArray) (zp : ZerocheckProof) (lp : LincheckProof), 14 ≤ m →
    zp.round1ab.size = 64 → zp.round1c.size = 64 → lp.zPartial.size = 64 →
    Carries (wZL m kLog) (zcShape 𝔽 m ++ lcShape 𝔽 kLog) (.dom d :: (zcFrames m dig cap zp ++ lcFrames kLog lp))
      (zcMsgs m zp ++ lcMsgs kLog lp) []

/-- **The lengths a decoded proof has** (`decodeProof`'s `vecExact` checks, PROTOCOL.md §8). -/
structure ProofSizes (sh : Shape) (p : Proof) : Prop where
  r1ab : p.zerocheck.round1ab.size = 64
  r1c : p.zerocheck.round1c.size = 64
  zcRounds : p.zerocheck.rounds.size = sh.m - 6
  lcRounds : p.lincheck.rounds.size = sh.kLog - 6
  zPartial : p.lincheck.zPartial.size = 64
  rs : p.ringSwitches.size = sh.claims
  rsEach : ∀ v ∈ p.ringSwitches, v.size = 128
  cap : p.cap.size = 2 ^ (sh.sched.levels.getD 0 default).cap
  capEach : ∀ e ∈ p.cap, e.size = sh.hashLen
  rcaps : p.ligerito.recursiveCaps.size = sh.sched.levels.size - 1
  rcapsEach : ∀ (i : ℕ) (hi : i < p.ligerito.recursiveCaps.size),
    p.ligerito.recursiveCaps[i].size = 2 ^ (sh.sched.levels.getD (i + 1) default).cap ∧
      ∀ e ∈ p.ligerito.recursiveCaps[i], e.size = sh.hashLen
  yr : p.ligerito.finalProof.yr.size = 2 ^ (sh.sched.ext + 1)
  messages : p.ligerito.messages.size = sh.sched.k0 + 8 + 7 * (sh.sched.levels.size - 1 - 1)
  queryNonces : p.ligerito.queryNonces.size = sh.sched.levels.size - 1 + 1
  oodValues : p.ligerito.oodValues.size = 1 + 2 * (sh.sched.levels.size - 1)
  claimNonces : p.ligerito.claimNonces.size = 3 * (sh.sched.levels.size - 1) + 2
  consistencyNonces : p.ligerito.consistencyNonces.size = sh.sched.levels.size - 1 + 1

/-- **A decoded proof has the lengths its shape fixes.** -/
def DecodeSizes : Prop :=
  ∀ (sh : Shape) (bytes : ByteArray) (p : Proof), decodeProof sh bytes = .ok p → ProofSizes sh p

/-- **The opening's rounds carry its messages** (dec-open's chunk lemma): at a decoded proof of a `fast100` statement
with `nExtra` extra claims, the ring switches' rounds read by `wOP` are the opening's messages, cut at `opShape`'s
coins, with no tail. -/
def OPCarries (wOP : ℕ → List (List Wire)) : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (nExtra : ℕ) (bytes : ByteArray) (p : Proof), Flock.fast100 st.m = some sch →
    decodeProof (shapeOf st sch nExtra) bytes = .ok p →
    Carries (wOP (nExtra + 2)) (opShape 𝔽 (nExtra + 2)) (rsFrames (2 + nExtra) p.ringSwitches)
      (opMsgs (2 + nExtra) p.ringSwitches) []

/-- **Ligerito's rounds carry its messages** (dec-lig's chunk lemma): at a decoded proof of a `fast100` statement with
64-byte digests, Ligerito's rounds read by `wLIG` are its messages (caps as `capDig`), cut at `ligShape`'s coins, with
no tail. -/
def LIGCarries (wLIG : Accounting.Schedule → List (List Wire)) : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (nExtra : ℕ) (bytes : ByteArray) (p : Proof) (T : F128),
    Flock.fast100 st.m = some sch → st.merkle.digestLen = 64 → decodeProof (shapeOf st sch nExtra) bytes = .ok p →
    Carries (wLIG (schedOf sch)) (ligShape 𝔽 𝕂 Dig (schedOf sch)) (ligFrames sch T p.cap p.ligerito)
      (ligMsgs p.ligerito capDig sch) []

/-- **The verify command's tags**: public outputs, retained rounds, `hm96-sha512/v1` leaves, one table. -/
def TagsOk (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1" ∧ I.count ≤ 1

variable {n mPts : ℕ} {Reg : Type}

/-- **`wr` is the challenger's record of the live session**: whenever flock-verify accepts what `wr` writes for the
outcome `o` at draw `S`, the record carries the draw `dj S`, the proof files are `o`'s, and the record decodes to a
session whose table's `root_B`, link points and rep streams' rounds are `o`'s. -/
def RecordsLive (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ S R o, Accepts I (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOf I (wr S R o).record = .ok (spec, sts) → Record.decode spec (wr S R o).record = .ok sess →
      sess.rootB[0]! = o.rootB ∧ ptsOf mPts sess.points = o.pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = pick (o.rep0, o.rep1) r

/-- **The challenger's draw at `S`**: it proves `S`, and setupH sets up at it with `mPts` link points. -/
def DrawOk (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program = .ok st →
      st.mPts = mPts

/-- The number of the table's extra claims (two per region; the link points don't change it). -/
noncomputable def nX (T : Finset (Fin n) → Reg → TabSpec mPts) (S : Finset (Fin n)) (R : Reg) : ℕ :=
  (Model.extraClaims (T S R).S (T S R).ptLocal mPts (fun _ _ => (0 : 𝔽))).length

/-- **The decoder at draw `S`**: the zerocheck's and lincheck's wires, the opening's, then Ligerito's, at the table's
dimensions, number of claims and schedule. -/
noncomputable def decAt (T : Finset (Fin n) → Reg → TabSpec mPts) (wOP : ℕ → List (List Wire))
    (wLIG : Accounting.Schedule → List (List Wire)) (S : Finset (Fin n)) (R : Reg) : ℕ → ByteArray → List Msg :=
  decOf (wZL (T S R).S.m (T S R).S.kLog ++ (wOP (nX T S R + 2) ++ wLIG (T S R).sch))

/-- **The openings at draw `S`**: `opnOf` at the statement setupH sets up at the record's draw `dj S` and its `fast100`
schedule, on the first two proof files. -/
noncomputable def opnAt (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) : Opens 𝔽 Dig × Salts (List UInt8) :=
  match Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program with
  | .ok st => match Flock.fast100 st.m with
    | some sch => opnOf (Setup.ofCircuit st) sch (nX T S R) #[proofs[0]!, proofs[1]!]
    | none => default
  | .error _ => default

/-- **The decoder is right at the headline's draws** (R11b): given the opening's and Ligerito's chunk lemmas, the
tags, `wr` the challenger's record and the challenger's draws proving their units at `mPts` link points, `DecodesOn`
holds at the plain headline's tables with the decoder `decAt`, single coins as field elements, no final messages and
`capOfRoot`. -/
def DecodesOnExec : Prop :=
  ∀ {c : Flock.Circuit} {n : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    (_hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (L : Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (wOP : ℕ → List (List Wire))
    (wLIG : Accounting.Schedule → List (List Wire)),
    OPCarries wOP → LIGCarries wLIG → TagsOk I → RecordsLive I dj wr → (∀ ω, DrawOk I dj mPts (L.draw ω)) →
    DecodesOn H512 enc512 leaf512 (tablesAt (Reg := Reg) ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj)
      (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L

end FlockSoundness.Discharge.DecodeZL
