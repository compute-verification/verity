module

public import Proofs.Flock.Claims.Complete.ZkTables.RecordBytes
import Proofs.Flock.Claims.Complete.Json.CanonSize

/-!
# The honest masked record's length, from its spec's text

`specText spec` is the text the record prints from its spec: the `hello` as the prover writes it, the unit sample, the
keyed coin tree's spec, `Σ`, the table's name and its two streams' domains. At one table, a statement of `25 ≤ m ≤ 35`
under `hm96-sha512/v1`, the honest masked session's record is at most `2^8 · specText + 2^40` bytes (`recordZk_size_of`),
whatever the witness, tape and coins, so it fits its length word where the text is under `2^55`
(`recordFitsZk_of_text`).
-/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy With FlockComplete.Replay
open Lean (Json)

variable {H : Claims.Msg → Claims.Msg}

/-- The length of the `hello` the masked prover writes: the verifier's, parsed, with the prover's nonce put in, printed
canonically; the verifier's own where the spec names no keyed coin tree. -/
def helloLen (spec : SessionSpec) : ℕ :=
  match spec.coinTree with
  | none => spec.hello.length
  | some _ =>
    match (Flock.JsonParse.parse spec.hello).toOption with
    | some own => match canon (JsonRT.helloJ own) with
      | .ok hz => hz.length
      | .error _ => 0
    | none => 0

/-- The weight of the keyed coin tree's spec the record prints. -/
def coinTreeText (spec : SessionSpec) : ℕ :=
  match spec.coinTree with
  | some ct => jsz ct.toJson
  | none => 0

/-- **The text of a session spec**: its `hello` as written, its unit sample, its coin tree's spec, `Σ`, its table's name
and the domains of the table's two streams. -/
def specText (spec : SessionSpec) : ℕ :=
  helloLen spec + jsz (spec.unitSample.getD Json.null) + coinTreeText spec + spec.sigma.size +
    (spec.tables.getD 0 "").utf8ByteSize + (spec.domain (spec.tables.getD 0 "") 0).utf8ByteSize +
    (spec.domain (spec.tables.getD 0 "") 1).utf8ByteSize

theorem length_rep (t : String) (n : ℕ) (hn : n < 2) : (s!"{t}/rep{n}").length = t.length + 5 := by
  have h : (s!"{t}/rep{n}") = t ++ "/rep" ++ toString n := rfl
  rw [h, String.length_append, String.length_append]
  have h4 : ("/rep" : String).length = 4 := by decide +kernel
  have h1 : (toString n).length = 1 := by
    interval_cases n <;> decide +kernel
  omega

/-- A one-table session's two streams are named by the table and read their domains. -/
theorem namesLe_one {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) :
    NamesLe spec (t.length + 5 + (spec.domain t 0).length + (spec.domain t 1).length) := by
  intro q hq
  rw [JsonRT.repStreams_one ht] at hq
  have ht0 : spec.tables[0]! = t := by rw [ht]; rfl
  simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hq
  rcases hq with rfl | rfl
  · show (s!"{t}/rep{(0 : Nat)}").length + (spec.domain spec.tables[0]! 0).length ≤ _
    rw [length_rep t 0 (by decide), ht0]
    omega
  · show (s!"{t}/rep{(1 : Nat)}").length + (spec.domain spec.tables[0]! 1).length ≤ _
    rw [length_rep t 1 (by decide), ht0]
    omega

theorem toUTF8_size_eq (s : String) : s.toUTF8.size = s.utf8ByteSize := by
  rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]

/-- **`recBound` is at most `2^8` the text and `2^40`**, at the domains' bytes and the names' characters. -/
theorem recBound_le_text {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) {ct : CoinTree.Spec}
    (hct : spec.coinTree = some ct) {own : Json} (hown : Flock.JsonParse.parse spec.hello = .ok own) {hz : String}
    (hc : canon (JsonRT.helloJ own) = .ok hz) {Q : ℕ} (hQ : Q ≤ 64) (hpt : spec.points ≤ 2) :
    recBound spec hz t ct Q ((spec.domain t 0).toUTF8.size + (spec.domain t 1).toUTF8.size)
      (t.length + 5 + (spec.domain t 0).length + (spec.domain t 1).length) ≤ 2 ^ 8 * specText spec + 2 ^ 40 := by
  have hrs : spec.repStreams.size = 2 := by rw [JsonRT.repStreams_one ht]; rfl
  have hT : spec.tables.getD 0 "" = t := by rw [ht]; rfl
  have hhz : helloLen spec = hz.length := by simp [helloLen, hct, hown, hc]
  have hp : spec.points * (579 * Q + 2 + 1) ≤ 2 * (579 * 64 + 2 + 1) := Nat.mul_le_mul hpt (by omega)
  have h1 := JsonRT.length_le_size t
  have h2 := JsonRT.length_le_size (spec.domain t 0)
  have h3 := JsonRT.length_le_size (spec.domain t 1)
  rw [toUTF8_size_eq, toUTF8_size_eq]
  unfold recBound
  unfold specText coinTreeText
  rw [hrs, hT, hct, hhz]
  simp only
  omega

/-- **The honest masked record weighs at most `2^8` its spec's text and `2^40`**, at one table whose statement has
`25 ≤ m ≤ 35`, the `hm96` leaf, `kLog ≤ m`, at most `2^10` regions and `m_pts ≤ 64`, with at most two link points. -/
theorem recordZk_size_of (h64 : Claims.Out64 H) {spec : SessionSpec} {t : String} (ht : spec.tables = #[t])
    {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json}
    (hown : Flock.JsonParse.parse spec.hello = .ok own) {hz : String} (hc : canon (JsonRT.helloJ own) = .ok hz)
    (st : Flock.Stmt) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hms : st.merkle = MerkleScheme.hm96Sha512)
    (hkm : st.c.kLog ≤ st.m) (hreg : st.regions.size ≤ 1024) (hpm : spec.mPts = st.mPts) (hmp : st.mPts ≤ 64)
    (hpt : spec.points ≤ 2) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk) {k : ℕ}
    (r : Fin k → Coin) {rec : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H)) spec
      #[Setup.ofCircuit st] words trs tape).go #[] = (.ok rec, bodies)) :
    rec.1.utf8ByteSize ≤ 2 ^ 8 * specText spec + 2 ^ 40 := by
  refine (record_size_le h64 ht hct hown hc st h25 h35 hms hkm hreg hpm hpt (D := (spec.domain t 0).toUTF8.size +
    (spec.domain t 1).toUTF8.size) (L := t.length + 5 + (spec.domain t 0).length + (spec.domain t 1).length)
    (fun rep hrep => by interval_cases rep <;> omega) (namesLe_one ht) words trs tape r hrun).trans ?_
  exact recBound_le_text ht hct hown hc hmp hpt

/-- **The honest masked record fits its length word** at one table whose statement has `25 ≤ m ≤ 35`, the `hm96`
leaf, `kLog ≤ m`, at most `2^10` regions and `m_pts ≤ 64`, where its spec's text is under `2^55`. -/
theorem recordFitsZk_of_text (h64 : Claims.Out64 H) {x : Claims.Msg} {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json}
    (hown : Flock.JsonParse.parse spec.hello = .ok own) {hz : String} (hc : canon (JsonRT.helloJ own) = .ok hz)
    (st : Flock.Stmt) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hms : st.merkle = MerkleScheme.hm96Sha512)
    (hkm : st.c.kLog ≤ st.m) (hreg : st.regions.size ≤ 1024) (hpm : spec.mPts = st.mPts) (hmp : st.mPts ≤ 64)
    (hpt : spec.points ≤ 2) (hfit : specText spec < 2 ^ 55) : RecordFitsZkAt H x spec #[st] := by
  refine recordFitsZk_of h64 ht hct hown hc st h25 h35 hms hkm hreg hpm hpt (D := (spec.domain t 0).toUTF8.size +
    (spec.domain t 1).toUTF8.size) (L := t.length + 5 + (spec.domain t 0).length + (spec.domain t 1).length)
    (fun rep hrep => by interval_cases rep <;> omega) (namesLe_one ht) ?_
  have := recBound_le_text ht hct hown hc hmp hpt
  omega

end FlockComplete.ZkTables
