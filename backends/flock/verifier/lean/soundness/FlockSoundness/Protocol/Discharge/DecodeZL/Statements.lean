import FlockSoundness.Protocol.Discharge.DecodeZL.Wire
import FlockSoundness.Protocol.Discharge.Integrate.Headline

/-!
# The message half of R11b, and the decoder at the headline: statements

`flock_headline_exec`'s open guarantee is `hdec : DecodesOn …`: at every outcome of the live session flock-verify
accepts, each rep's tape (its rounds' messages, decoded from their bytes, and coins) plays the rep's game, and the table's
verdict accepts. The statements here:

* `ZLCarries`: the zerocheck's and lincheck's rounds carry their messages by `wZL` (`Carries`, `Wire.lean`).
* `ProofSizes`, `DecodeSizes`: the lengths `decodeProof` checks, which make the message-carrying frames well formed.
* `OPCarries`, `LIGCarries`: the same chunk lemma for the opening's ring switches and for Ligerito, at a decoded
  proof of a `fast100` statement (owners: dec-open, dec-lig).
* `TagsOk`, `RecordsLive`, `DrawOk`: what `DecodesOn` needs past #828's hypotheses. `wr` is a free parameter of the
  headline, and at a `wr` constant at an accepted transcript `DecodesOn` is false (an outcome with no rounds can't play
  `repC`, which starts with a coin); `RecordsLive` says `wr` is the coin server's record of the outcome.
* `decAt`, `opnAt`: the decoder and the openings at draw `S`, read off the table `T S R` and setupH's statement there.
* `DecodesOnExec`: `DecodesOn` at #828's tables, the decoder `decAt`, `one := true`, no final messages, `capOfRoot`.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Integrate

variable {n mPts : ℕ} {Reg : Type}

/-- **`wr` is the coin server's record of the live session**: whenever flock-verify accepts what `wr` writes for the
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

/-- **The coin server's draw at `S`**: it proves `S`, and setupH sets up at it with `mPts` link points. -/
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

end FlockSoundness.Discharge.DecodeZL
