module

public import Proofs.Flock.Claims.Complete.Json.RecordSize
public import Proofs.Flock.Claims.Complete.Zk.Json.Record

/-!
# The masked record's weight

The masked record is the plain record at `specZ` and its `coin_tree` (`recordListZk`), so it weighs what the plain
record's values do (`recordList_sum_le`), its coin tree's spec and a constant (`jsz_coinTreeJ_le`), and a field's key
each (`jsz_recordObjZk_le`).
-/

@[expose] public section

namespace FlockComplete

open Flock Transcript JsonRT
open Lean (Json)

theorem jsz_zeros (n : ℕ) : jsz (Json.str (zeros n)) = 18 * n + 2 := by
  simp [jsz_str, zeros]

theorem jsz_coinTreeJ_le (ct : CoinTree.Spec) : jsz (coinTreeJ ct) ≤ jsz ct.toJson + 14000 := by
  have hK := keys_le (l := [("key", Json.str (zeros 512)), ("nonce", Json.str (zeros 64)),
    ("root", Json.str (zeros 128)), ("spec", ct.toJson)]) (K := 16) (by
      simp only [List.map_cons, List.map_nil]
      decide +kernel)
  have e0 := jsz_mkObj_keys (coinTreeJ_list_distinct ct) hK
  unfold coinTreeJ
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons, List.length_nil,
    jsz_zeros] at e0
  omega

/-- **The masked record weighs at most** its spec's text, its root, its points and its streams, under `2^64`
whenever those are. -/
theorem jsz_recordObjZk_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) {spec : SessionSpec}
    {t : String} (ht : spec.tables = #[t]) (hz : String) (ct : CoinTree.Spec) (rb : ByteArray)
    {points : Array (Array F128)} {Q : ℕ} (hp : ∀ p ∈ points, p.size ≤ Q) {streams : Array (Array Round)}
    {R M E L : ℕ} (hR : R < 10 ^ 20) (hs : StreamsLe streams R M E) (hL : NamesLe spec L) (proofs : Array ByteArray) :
    jsz (recordObjZk spec h hz ct #[rb] points streams proofs) ≤
      18 * hz.length + jsz (spec.unitSample.getD Json.null) + jsz ct.toJson + 72 * spec.sigma.size +
        36 * rb.size + 36 * t.length + points.size * (579 * Q + 2 + 1) + 579 * spec.points +
        spec.repStreams.size * (18 * L + R * (M + 4747) + 40 * E + 4500 + 1) +
        spec.repStreams.size * (18 * L + 2900 + 1) + 30000 := by
  have hK : ∀ p ∈ recordListZk spec h hz ct #[rb] points streams proofs, p.1.length ≤ 16 :=
    keys_le (by
      rcases dk_cases (specZ spec hz) with e | e <;>
        simp only [recordListZk, recordList, e, List.map_cons, List.map_nil, List.cons_append, List.nil_append] <;>
        decide +kernel)
  have e0 := jsz_mkObj_keys (recordListZk_distinct spec h hz ct #[rb] points streams proofs) hK
  have e1 := recordList_sum_le (spec := specZ spec hz) hh ht rb hp hR hs hL proofs
  have e2 := jsz_coinTreeJ_le ct
  have e3 : (recordList (specZ spec hz) h #[rb] points streams proofs).length = 10 := rfl
  generalize hW : jsz (Json.mkObj (recordListZk spec h hz ct #[rb] points streams proofs)) = W at e0
  unfold recordObjZk
  rw [hW]
  simp only [recordListZk, List.map_append, List.sum_append, List.length_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, List.length_cons, List.length_nil, e3] at e0
  have r1 : (specZ spec hz).repStreams = spec.repStreams := rfl
  have r2 : (specZ spec hz).hello = hz := rfl
  have r3 : (specZ spec hz).unitSample = spec.unitSample := rfl
  have r4 : (specZ spec hz).sigma = spec.sigma := rfl
  have r5 : (specZ spec hz).points = spec.points := rfl
  rw [r1, r2, r3, r4, r5] at e1
  omega

end FlockComplete
