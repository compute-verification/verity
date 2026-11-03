import Proofs.Flock.Claims.Recursion.NextSha
import Proofs.Flock.Soundness.Refine.Rep

/-!
# S17 at the substituted value (`decodeJWith_shaSubJ`, `s17_shaSubJ`)

At the record's value with its proof digests written from digests `ds` (`shaSubJ`), the decoder's proof digests are
`shaArr`'s entries, whatever the rest of the value (`decodeJWith_shaSubJ`): it reads `proof_{dk}` last, and `shaSubJ`
replaces that key's value in place (`arr_mapKey_const`), each entry an object of the two distinct keys `stream` and `dk`
(`proofShaRd_entry`). At the files' digests through `h`, with as many files as rep streams and the rep streams' names
distinct (`NamesDistinct`), S17's two checks pass (`s17_canonical`): the digests list exactly the rep streams, and each
file's digest is the one its stream's entry names. `s17_shaSubJ` is the two together, at a decoded session.
-/

namespace FlockDeploy

open Flock
open Lean (Json)
open FlockComplete.JsonRT (Distinct getObjVal?_mkObj dk_cases)

/-- Rep stream `q`'s entry of `proof_{dk}` at the digest `q.2`. -/
def shaList (spec : SessionSpec) (q : (ℕ × ℕ × String) × ByteArray) : List (String × Json) :=
  [("stream", Json.str q.1.2.2), (spec.dk, FlockComplete.hexJ q.2)]

theorem shaArr_eq (spec : SessionSpec) (ds : Array ByteArray) :
    shaArr spec ds = .arr ((spec.repStreams.zip ds).map fun q => Json.mkObj (shaList spec q)) := rfl

theorem shaList_distinct (spec : SessionSpec) (q : (ℕ × ℕ × String) × ByteArray) : Distinct (shaList spec q) := by
  rcases dk_cases spec with e | e <;> simp [shaList, e]

/-- The decoder's read of one entry of `proof_{dk}` (`Record.decodeJWith`'s last step). -/
def proofShaRd (spec : SessionSpec) (p : Json) : Flock.V (String × String) := do
  pure ((← Record.str p "stream"), (← Record.str p spec.dk))

theorem str_mkObj {l : List (String × Json)} (hd : Distinct l) {k s : String} (hm : (k, Json.str s) ∈ l) :
    Record.str (Json.mkObj l) k = .ok s := by
  unfold Record.str Record.fld
  rw [getObjVal?_mkObj hd hm] <;> rfl

theorem proofShaRd_entry (spec : SessionSpec) (q : (ℕ × ℕ × String) × ByteArray) :
    proofShaRd spec (Json.mkObj (shaList spec q)) = .ok (q.1.2.2, hexOf q.2) := by
  unfold proofShaRd
  rw [str_mkObj (k := "stream") (s := q.1.2.2) (shaList_distinct spec q) (by simp [shaList]),
    str_mkObj (k := spec.dk) (s := hexOf q.2) (shaList_distinct spec q) (by simp [shaList, FlockComplete.hexJ])] <;> rfl

theorem mapM_map_ok {α β γ : Type} (xs : Array α) (h : α → β) (f : β → Flock.V γ) (g : α → γ)
    (hf : ∀ x, f (h x) = .ok (g x)) : (xs.map h).mapM f = .ok (xs.map g) := by
  have hl : ∀ l : List α, (l.map h).mapM f = .ok (l.map g) := by
    intro l
    induction l with
    | nil => rfl
    | cons x l ih =>
      rw [List.map_cons, List.mapM_cons, hf x, ih]
      try rfl
  rw [Array.mapM_eq_mapM_toList, Array.toList_map, hl]
  show Except.ok (List.toArray (xs.toList.map g)) = _
  first
  | (rw [← Array.toList_map, Array.toArray_toList])
  | (congr 1; simp)

/-- **The substituted value's digests are `shaArr`'s**: the decoder's last two reads at `shaSubJ spec ds j`. -/
theorem proofSha_read {spec : SessionSpec} {ds : Array ByteArray} {j : Json} {a : Array Json}
    {ps : Array (String × String)} (ha : Record.arr (shaSubJ spec ds j) s!"proof_{spec.dk}" = .ok a)
    (hm : a.mapM (proofShaRd spec) = .ok ps) :
    ps = (spec.repStreams.zip ds).map fun q => (q.1.2.2, hexOf q.2) := by
  have hae : a = (spec.repStreams.zip ds).map fun q => Json.mkObj (shaList spec q) := arr_mapKey_const ha
  subst hae
  rw [mapM_map_ok (spec.repStreams.zip ds) (fun q => Json.mkObj (shaList spec q)) (proofShaRd spec)
    (fun q => (q.1.2.2, hexOf q.2)) (proofShaRd_entry spec)] at hm
  cases hm
  rfl

/-- **The decoder reads `shaArr`'s digests at the substituted value**, whatever the rest of the value. -/
theorem decodeJWith_shaSubJ (hs : Flock.Hashes Flock.V) {spec : SessionSpec} {ds : Array ByteArray} {j : Json}
    {sess : RunRecord} (h : Record.decodeJWith hs spec (shaSubJ spec ds j) = .ok sess) :
    sess.proofSha = (spec.repStreams.zip ds).map fun q => (q.1.2.2, hexOf q.2) := by
  unfold Record.decodeJWith at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := FlockSoundness.Refine.ebind_ok h) | (split at h))
  all_goals
    first
    | (cases h; done)
    | (cases h; exact proofSha_read (by assumption) (by assumption))

/-- The rep streams' names are distinct. -/
def NamesDistinct (spec : SessionSpec) : Prop :=
  ∀ (i k : ℕ) (hi : i < spec.repStreams.size) (hk : k < spec.repStreams.size),
    spec.repStreams[i].2.2 = spec.repStreams[k].2.2 → i = k

/-- **S17 passes at the files' digests**: where the decoded proof digests are `shaArr`'s at the files' digests through
`h`, of as many files as rep streams with distinct names, S17's list check and every file's digest check pass (as
`FlockComplete.verifyWith_ok` takes them). -/
theorem s17_canonical (spec : SessionSpec) (h : ByteArray → ByteArray) {bs : Array ByteArray}
    {ps : Array (String × String)}
    (hps : ps = (spec.repStreams.zip (bs.map (FlockComplete.dgOf spec h))).map fun q => (q.1.2.2, hexOf q.2))
    (hsz : bs.size = spec.repStreams.size) (hnd : NamesDistinct spec) :
    (ps.size != spec.repStreams.size || !spec.repStreams.all fun a => ps.any fun x => x.1 == a.2.2) = false ∧
    ∀ q ∈ (spec.repStreams.zip bs).toList,
      ((ps.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (FlockComplete.dgOf spec h q.2))) = false := by
  have hn : ps.size = spec.repStreams.size := by
    subst hps
    simp [hsz]
  have he : ∀ (i : ℕ) (hi : i < spec.repStreams.size),
      ps[i]'(by omega) = (spec.repStreams[i].2.2, hexOf (FlockComplete.dgOf spec h (bs[i]'(by omega)))) := by
    intro i hi
    subst hps
    simp [Array.getElem_map, Array.getElem_zip]
  refine ⟨?_, ?_⟩
  · have hall : (spec.repStreams.all fun a => ps.any fun x => x.1 == a.2.2) = true := by
      rw [Array.all_eq_true]
      intro i hi
      show (ps.any fun x => x.1 == spec.repStreams[i].2.2) = true
      rw [Array.any_eq_true]
      exact ⟨i, by omega, by rw [he i hi]; exact beq_self_eq_true _⟩
    rw [hall, hn]
    simp
  · intro q hq
    rw [Array.mem_toList_iff] at hq
    obtain ⟨i, hi, hqi⟩ := Array.getElem_of_mem hq
    have hi' : i < spec.repStreams.size := by simp only [Array.size_zip] at hi; omega
    have hq' : q = (spec.repStreams[i], bs[i]'(by omega)) := by rw [← hqi, Array.getElem_zip]
    subst hq'
    dsimp only
    have hf : ps.find? (·.1 == spec.repStreams[i].2.2) = some (ps[i]'(by omega)) := by
      rw [Array.find?_eq_some_iff_getElem]
      refine ⟨by rw [he i hi']; exact beq_self_eq_true _, i, by omega, rfl, fun k hk => ?_⟩
      have hk' : k < spec.repStreams.size := by omega
      rw [he k hk']
      have hne : spec.repStreams[k].2.2 ≠ spec.repStreams[i].2.2 := fun e => by
        have := hnd k i hk' hi' e
        omega
      simpa using hne
    rw [hf, he i hi']
    simp

/-- **S17 passes at the substituted value**: a session decoded at the record's value with its proof digests the
files' through `h` passes S17's two checks, at as many files as rep streams with distinct names. -/
theorem s17_shaSubJ (h : ByteArray → ByteArray) {spec : SessionSpec} {bs : Array ByteArray} {j : Json}
    {sess : RunRecord}
    (hd : Record.decodeJWith (Flock.Hashes.ofSha512 h : Flock.Hashes Flock.V) spec
      (shaSubJ spec (bs.map (FlockComplete.dgOf spec h)) j) = .ok sess)
    (hsz : bs.size = spec.repStreams.size) (hnd : NamesDistinct spec) :
    (sess.proofSha.size != spec.repStreams.size ||
      !spec.repStreams.all fun a => sess.proofSha.any fun x => x.1 == a.2.2) = false ∧
    ∀ q ∈ (spec.repStreams.zip bs).toList,
      ((sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (FlockComplete.dgOf spec h q.2))) = false :=
  s17_canonical spec h (decodeJWith_shaSubJ _ hd) hsz hnd

end FlockDeploy
