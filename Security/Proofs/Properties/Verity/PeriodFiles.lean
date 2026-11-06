import Properties.Verity.PeriodFiles
import Proofs.Properties.Composition
import Proofs.Properties.SampledProofs
import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Verifier.Hash
import Proofs.Flock.Soundness.Discharge.Exec.ParseSha
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Hidden.ParseCommit
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.TmplUnitNet

/-!
# The circuit files among the blobs, and read by path

Where nothing among the blobs `bs ++ cs` collides under SHA-512, the blobs `bs ++ cs` hold every blob `bs` hold, at
the same digest (`BlobSup`), and each of `cs` at its own. Where the network certifier's period, with its files `cs`
read by path, accepts (`NetworkCheck.Period.readCheck`), no file of `cs` is a pinned lookup table, so the sampled
proofs find the same tables among `bs ++ cs` as among `bs`, and the period finds its listed files among `bs ++ cs` as
`cs` itself (`filesOf_eq`). The sampled proofs' check is monotone in the blobs where it accepts (`sampled_mono`):
where the blobs `bs` hold every file it reads, so do `bs ++ cs`, at the same digests. Every other component reads no
blobs. A composite whose every component accepts as much on the larger blobs answers the same (`BlobCovers`,
`tiedSameB_covers`), so at onsite, and at remote with the network certifier with PoUW or without, the verifier of
record answers on `bs ++ cs` as the executable that reads `cs` by path answers on `bs` (`agree_onsite`, `agree_rn`,
`agree_rnn`, `Verity.circuit_files_by_path_agree`). An executable that accepts everything is not one the verifier of
record agrees with (`Verity.circuit_files_by_path_agree_nontrivial`), and at a period of no link-window and no audit
the period read by path and the period among the blobs both accept (`Verity.circuit_files_by_path_agree_honest`).
-/

namespace Verity.PeriodFiles

open Core
open Lean (Json)
open Flock (V)

local instance lawfulBEqByteArrayPF : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## Results that agree where the smaller blobs give one -/

theorem ok_bind {α β : Type} (a : α) (f : α → V β) : ((.ok a : V α) >>= f) = f a := rfl

theorem bindOk_split {α β : Type} {x : V α} {f : α → V β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem mapErrorOk {α : Type} (x : V α) (g : String → String) (a : α) : x.mapError g = .ok a ↔ x = .ok a := by
  cases x <;> simp [Except.mapError]

theorem ok_of_toBool {v : V Unit} (h : v.toBool = true) : v = .ok () := by
  cases v with
  | error e => cases h
  | ok u => rfl

/-- `y` gives every result `x` gives that satisfies `P`. -/
structure ExtP {α : Type} (P : α → Prop) (x y : V α) : Prop where
  run : ∀ a, x = .ok a → P a → y = .ok a

/-- `y` gives every result `x` gives. -/
abbrev ExtT {α : Type} (x y : V α) := ExtP (fun _ => True) x y

theorem ExtP.refl {α : Type} (P : α → Prop) (x : V α) : ExtP P x x := ⟨fun _ h _ => h⟩

theorem ExtP.of_ext {α : Type} {P : α → Prop} {x y : V α} (h : ExtT x y) : ExtP P x y :=
  ⟨fun a ha _ => h.run a ha trivial⟩

theorem ExtP.bind {α β : Type} {P : β → Prop} {x x' : V α} {f f' : α → V β}
    (hx : ExtT x x') (hf : ∀ a, ExtP P (f a) (f' a)) : ExtP P (x >>= f) (x' >>= f') := by
  constructor
  intro b h hp
  obtain ⟨a, ha, hb⟩ := bindOk_split h
  rw [hx.run a ha trivial, ok_bind]
  exact (hf a).run b hb hp

theorem ExtP.bind_back {α β : Type} {P : β → Prop} {Q : α → Prop} {x x' : V α} {f f' : α → V β}
    (hx : ExtP Q x x') (hf : ∀ a, ExtP P (f a) (f' a)) (hq : ∀ a b, f a = .ok b → P b → Q a) :
    ExtP P (x >>= f) (x' >>= f') := by
  constructor
  intro b h hp
  obtain ⟨a, ha, hb⟩ := bindOk_split h
  rw [hx.run a ha (hq a b hb hp), ok_bind]
  exact (hf a).run b hb hp

theorem ExtP.map {α β : Type} {P : β → Prop} {x x' : V α} (g : α → β) (hx : ExtT x x') :
    ExtP P (g <$> x) (g <$> x') := by
  constructor
  intro a h _
  cases hx1 : x with
  | error e => rw [hx1] at h; cases h
  | ok a' => rw [hx1] at h; rw [hx.run a' hx1 trivial]; exact h

theorem ExtP.mapError {α : Type} {P : α → Prop} {x x' : V α} {g : String → String} (hx : ExtT x x') :
    ExtP P (x.mapError g) (x'.mapError g) := by
  constructor
  intro a h _
  cases hx1 : x with
  | error e => rw [hx1] at h; cases h
  | ok a' => rw [hx1] at h; rw [hx.run a' hx1 trivial]; exact h

theorem ExtT.listMapM {α β : Type} {f f' : α → V β} (hf : ∀ a, ExtT (f a) (f' a)) :
    ∀ l : List α, ExtT (l.mapM f) (l.mapM f')
  | [] => by simp only [List.mapM_nil]; exact ExtP.refl _ _
  | a :: l => by
    simp only [List.mapM_cons]
    exact ExtP.bind (hf a) fun _ => ExtP.bind (ExtT.listMapM hf l) fun _ => ExtP.refl _ _

theorem ExtP.listMapM {α β : Type} {P : List β → Prop} {f f' : α → V β} (hf : ∀ a, ExtT (f a) (f' a))
    (l : List α) : ExtP P (l.mapM f) (l.mapM f') :=
  ExtP.of_ext (ExtT.listMapM hf l)

theorem ExtP.arrayMapM {α β : Type} {P : Array β → Prop} {f f' : α → V β} (hf : ∀ a, ExtT (f a) (f' a))
    (xs : Array α) : ExtP P (xs.mapM f) (xs.mapM f') := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
  exact ExtP.map _ (ExtT.listMapM hf _)

theorem ExtP.optionMapM {α β : Type} {P : Option β → Prop} {f f' : α → V β} (hf : ∀ a, ExtT (f a) (f' a)) :
    ∀ o : Option α, ExtP P (o.mapM f) (o.mapM f')
  | none => ExtP.refl _ _
  | some a => by rw [Option.mapM_some, Option.mapM_some]; exact ExtP.map _ (hf a)

theorem ExtT.listForIn {α β : Type} {f f' : α → β → V (ForInStep β)} (hf : ∀ a b, ExtT (f a b) (f' a b)) :
    ∀ (l : List α) (b : β), ExtT (forIn l b f) (forIn l b f')
  | [], b => by simp only [List.forIn_nil]; exact ExtP.refl _ _
  | a :: l, b => by
    simp only [List.forIn_cons]
    refine ExtP.bind (hf a b) fun s => ?_
    cases s with
    | done b' => exact ExtP.refl _ _
    | yield b' => exact ExtT.listForIn hf l b'

theorem ExtP.arrayForIn {α β : Type} {P : β → Prop} {f f' : α → β → V (ForInStep β)}
    (hf : ∀ a b, ExtT (f a b) (f' a b)) (xs : Array α) (b : β) : ExtP P (forIn xs b f) (forIn xs b f') := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact ExtP.of_ext (ExtT.listForIn hf _ _)

theorem ExtP.pure_ok {v v' : V Unit} (h : ExtT v v') :
    ExtP (fun w : V Unit => w = .ok ()) (pure v : V (V Unit)) (pure v') := by
  constructor
  intro a ha hp
  have hva : v = a := Except.ok.inj ha
  have hp' : a = .ok () := hp
  have hv : v = .ok () := hva.trans hp'
  have hv' : v' = .ok () := h.run () hv trivial
  show Except.ok v' = Except.ok a
  rw [hv', ← hva, hv]

/-! ## Blobs -/

/-- No two of the byte strings `L` share a SHA-512. -/
def BlobInj (L : Array ByteArray) : Prop :=
  ∀ a ∈ L, ∀ b ∈ L, Flock.Sha512.hash a = Flock.Sha512.hash b → a = b

/-- No byte string of `cs` is a pinned lookup table, by its SHA-512. -/
def NoTable (cs : Array ByteArray) : Prop :=
  ∀ c ∈ cs, ∀ t ∈ Flock.MUFU_TABLES, (Flock.hexOf (Flock.Sha512.hash c) == t.sha512) = false

/-- The blobs `B2` give every blob `B1` gives, and the same pinned lookup tables. -/
structure BlobSup (B1 B2 : Core.Blobs) : Prop where
  get : ∀ d w b, B1.get d w = .ok b → B2.get d w = .ok b
  tables : SampledProofs.OneStageZk.tablesOf B2 = SampledProofs.OneStageZk.tablesOf B1

theorem get_ext {B1 B2 : Core.Blobs} (hB : BlobSup B1 B2) (d w : String) : ExtT (B1.get d w) (B2.get d w) :=
  ⟨fun b h _ => hB.get d w b h⟩

theorem inputsOf_tab {B1 B2 : Core.Blobs}
    (h : SampledProofs.OneStageZk.tablesOf B2 = SampledProofs.OneStageZk.tablesOf B1) (tags p c pub) :
    SampledProofs.OneStageZk.inputsOf tags p c pub B2 = SampledProofs.OneStageZk.inputsOf tags p c pub B1 := by
  simp only [SampledProofs.OneStageZk.inputsOf, h]

theorem inj_of_noCollision {L : Array ByteArray}
    (h : ¬ Hash.CollisionIn Verity.sha512 (L.toList.map (·.data.toList))) : BlobInj L := by
  intro a ha b hb hab
  by_contra hne
  apply h
  unfold Hash.CollisionIn Hash.Collision
  refine ⟨a.data.toList, List.mem_map.2 ⟨a, Array.mem_toList_iff.2 ha, rfl⟩, b.data.toList,
    List.mem_map.2 ⟨b, Array.mem_toList_iff.2 hb, rfl⟩, ?_, ?_⟩
  · intro hab'
    apply hne
    exact ByteArray.ext (Array.toList_inj.1 hab')
  · show List.ofFn (Verity.sha512 a.data.toList) = List.ofFn (Verity.sha512 b.data.toList)
    have hx : ∀ x : ByteArray, (⟨x.data.toList.toArray⟩ : ByteArray) = x := fun x => by cases x; simp
    unfold Verity.sha512
    rw [hx a, hx b, hab]

theorem get_of {B : Core.Blobs} {d w : String} {x y b : ByteArray} (hu : Flock.unhex d = .ok x) (hs : x.size = 64)
    (hh : Flock.hexOf x = d) (hf : B.entries.find? (·.1 == x) = some (y, b)) : B.get d w = .ok b := by
  have hc : (x.size != 64 || Flock.hexOf x != d) = false := by simp [hs, hh]
  unfold Core.Blobs.get
  simp only [hu, ok_bind, hc, Bool.false_eq_true, ↓reduceIte, hf]
  all_goals rfl

theorem get_ok_inv {B : Core.Blobs} {d w : String} {b : ByteArray} (h : B.get d w = .ok b) :
    ∃ x y, Flock.unhex d = .ok x ∧ x.size = 64 ∧ Flock.hexOf x = d ∧
      B.entries.find? (·.1 == x) = some (y, b) := by
  unfold Core.Blobs.get at h
  cases hu : Flock.unhex d with
  | error e => rw [hu] at h; cases h
  | ok x =>
    rw [hu, ok_bind] at h
    try dsimp only at h
    split at h
    · cases h
    next hc =>
      split at h
      · rename_i y b' heq
        cases h
        refine ⟨x, y, rfl, ?_, ?_, heq⟩
        · by_contra hne; apply hc; simp [hne]
        · by_contra hne; apply hc; simp [hne]
      · cases h

theorem get_mono {bs cs : Array ByteArray} {d w : String} {b : ByteArray}
    (h : (Core.Blobs.ofArray bs).get d w = .ok b) : (Core.Blobs.ofArray (bs ++ cs)).get d w = .ok b := by
  obtain ⟨x, y, hu, hs, hh, hf⟩ := get_ok_inv h
  refine get_of hu hs hh (y := y) ?_
  simp only [Core.Blobs.ofArray, Array.map_append, Array.find?_append] at hf ⊢
  rw [hf, Option.some_or]

theorem find_hash {L : Array ByteArray} {b : ByteArray} (hinj : BlobInj L) (hb : b ∈ L) :
    (L.map fun b => (Flock.Sha512.hash b, b)).find? (fun x => x.1 == Flock.Sha512.hash b) =
      some (Flock.Sha512.hash b, b) := by
  cases hz : (L.map fun b => (Flock.Sha512.hash b, b)).find? (fun x => x.1 == Flock.Sha512.hash b) with
  | none =>
    rw [Array.find?_eq_none] at hz
    exact absurd (hz _ (Array.mem_map.2 ⟨b, hb, rfl⟩)) (by simp)
  | some z =>
    have hz1 := Array.find?_some hz
    obtain ⟨b', hb', rfl⟩ := Array.mem_map.1 (Array.mem_of_find?_eq_some hz)
    have heq : Flock.Sha512.hash b' = Flock.Sha512.hash b := by simpa using hz1
    rw [hinj b' hb' b hb heq]

theorem get_hash {L : Array ByteArray} {b : ByteArray} (hinj : BlobInj L) (hb : b ∈ L) (w : String) :
    (Core.Blobs.ofArray L).get (Flock.hexOf (Flock.Sha512.hash b)) w = .ok b :=
  get_of (FlockComplete.JsonRT.unhex_hexOf _) (Flock.Sha512.hash_size _) rfl (find_hash hinj hb)

theorem find_none_of {cs : Array ByteArray} {q : ByteArray × ByteArray → Bool}
    (hq : ∀ c ∈ cs, q (Flock.Sha512.hash c, c) = false) :
    (cs.map fun b => (Flock.Sha512.hash b, b)).find? q = none := by
  rw [Array.find?_eq_none]
  intro x hx
  obtain ⟨c, hc, rfl⟩ := Array.mem_map.1 hx
  simp [hq c hc]

theorem filterMapCongr {α β : Type} {f g : α → Option β} :
    ∀ (l : List α), (∀ x ∈ l, f x = g x) → l.filterMap f = l.filterMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.filterMap_cons, List.filterMap_cons, h a List.mem_cons_self,
      filterMapCongr l fun x hx => h x (List.mem_cons_of_mem _ hx)]

theorem tablesOf_eq {bs cs : Array ByteArray} (hn : NoTable cs) :
    SampledProofs.OneStageZk.tablesOf (Core.Blobs.ofArray (bs ++ cs)) =
      SampledProofs.OneStageZk.tablesOf (Core.Blobs.ofArray bs) := by
  have key : ∀ q : ByteArray × ByteArray → Bool, (∀ c ∈ cs, q (Flock.Sha512.hash c, c) = false) →
      ((bs ++ cs).map fun b => (Flock.Sha512.hash b, b)).find? q =
        (bs.map fun b => (Flock.Sha512.hash b, b)).find? q := by
    intro q hq
    rw [Array.map_append, Array.find?_append, find_none_of hq, Option.or_none]
  simp only [SampledProofs.OneStageZk.tablesOf, Core.Blobs.ofArray]
  apply filterMapCongr
  intro t ht
  rw [key]
  intro c hc
  exact hn c hc t ht

theorem sup_of {bs cs : Array ByteArray} (hn : NoTable cs) :
    BlobSup (Core.Blobs.ofArray bs) (Core.Blobs.ofArray (bs ++ cs)) :=
  ⟨fun _ _ _ h => get_mono h, tablesOf_eq hn⟩

/-! ## The network certifier's period -/

theorem circuitOf_sha {b : ByteArray} {c : Flock.Circuit}
    (h : Flock.ProvedScope.circuitOf Flock.Tags.circuit b [] = .ok c) : c.sha = Flock.Sha512.hash b := by
  rw [Flock.ProvedScope.circuitOf_hidden (tags := Flock.Tags.circuit) rfl] at h
  cases ht : Flock.Tags.circuit.isTmpl
  · rw [FlockSoundness.parseCommit_flat ht] at h
    rcases (FlockSoundness.Discharge.Hidden.parse_cases h).2 with ⟨-, h1⟩ | ⟨k, b', tm, c₀, h1, rfl⟩ |
        ⟨k, u, c₀, h1, rfl⟩
    · exact FlockSoundness.Discharge.Exec.parse_sha_pre h1
    · exact FlockSoundness.Discharge.Exec.parse_sha_tmpl (c := c₀) h1
    · exact FlockSoundness.Discharge.Exec.parse_sha_pre (c := c₀) h1
  · rw [FlockSoundness.parseCommit_tmpl ht] at h
    obtain ⟨tu, c₀, h1, -, -, rfl⟩ := FlockSoundness.TmplValue.parseTmpl_cases h
    exact FlockSoundness.Discharge.Exec.parse_sha_pre (c := c₀) h1

theorem ofFile_sha {scan : Bool} {b : ByteArray} {x : ByteArray × Flock.Circuit}
    (h : NetworkCheck.Period.ofFile scan b = .ok x) : x.1 = Flock.Sha512.hash b := by
  unfold NetworkCheck.Period.ofFile NetworkCheck.CircuitFile.whole at h
  cases hc : Flock.ProvedScope.circuitOf Flock.Tags.circuit b [] with
  | error e => rw [hc] at h; cases h
  | ok c => rw [hc] at h; cases h; exact circuitOf_sha hc

theorem ofRead_ok {n : Nat} {cs : Array ByteArray} {fs : Array (ByteArray × Flock.Circuit)}
    (h : NetworkCheck.Period.ofRead n cs = .ok fs) :
    fs.size = cs.size ∧ ∀ i (h1 : i < cs.size) (h2 : i < fs.size),
      NetworkCheck.Period.ofFile (decide (cs.size - n ≤ i)) cs[i] = .ok fs[i] := by
  unfold NetworkCheck.Period.ofRead at h
  try dsimp only at h
  obtain ⟨hs, hf⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
  rw [Array.size_zipIdx] at hs
  refine ⟨hs, fun i h1 h2 => ?_⟩
  have := hf i (by simpa using h1) h2
  simp only [Array.getElem_zipIdx, Nat.zero_add] at this
  exact (mapErrorOk _ _ _).1 this

theorem fileWhy_sha {what : String} {scan : Bool} {x : NetworkCheck.Period.Listed} {k : Nat} {sha : ByteArray}
    {cf : Flock.Circuit} (h : NetworkCheck.Period.fileWhy what scan x k sha cf = none) :
    Flock.hexOf sha = x.files[k]! := by
  by_contra hne
  have hb : (Flock.hexOf sha != x.files[k]!) = true := bne_iff_ne.2 hne
  unfold NetworkCheck.Period.fileWhy at h
  simp [hb] at h

theorem circuitsWhy_none {l : NetworkCheck.Period.Listing} {fs : Array (ByteArray × Flock.Circuit)}
    (h : NetworkCheck.Period.circuitsWhy l fs = none) :
    fs.size = l.files.size ∧ ∀ i (h1 : i < l.files.size) (h2 : i < fs.size) w s x k,
      l.files[i] = (w, s, x, k) → x.files[k]! = Flock.hexOf fs[i].1 := by
  unfold NetworkCheck.Period.circuitsWhy at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hsz
    have hsz' : fs.size = l.files.size := by simpa using hsz
    refine ⟨hsz', fun i h1 h2 w s x k hf => ?_⟩
    have hmem : (l.files[i], fs[i]) ∈ l.files.zip fs := by
      have hi : i < (l.files.zip fs).size := by simp [Array.size_zip]; omega
      have := Array.getElem_mem hi
      rw [Array.getElem_zip] at this
      exact this
    have := Array.findSome?_eq_none_iff.1 h _ hmem
    rcases hg : fs[i] with ⟨sha, cf⟩
    rw [hf] at this
    (try rw [hg] at this)
    dsimp only at this
    exact (fileWhy_sha this).symm

theorem files_split (l : NetworkCheck.Period.Listing) :
    ∃ A S : Array (String × Bool × NetworkCheck.Period.Listed × Nat), l.files = A ++ S ∧
      (∀ y ∈ A, y.2.1 = false) ∧ (∀ y ∈ S, y.2.1 = true) ∧ S.size = l.scanFiles := by
  unfold NetworkCheck.Period.Listing.files NetworkCheck.Period.Listing.owners
  rw [Array.flatMap_append]
  refine ⟨_, _, rfl, ?_, ?_, ?_⟩
  · intro y hy
    obtain ⟨o, ho, hy⟩ := Array.mem_flatMap.1 hy
    obtain ⟨⟨oo, i⟩, -, ho⟩ := Array.mem_filterMap.1 ho
    cases oo with
    | none => cases ho
    | some x =>
      cases ho
      obtain ⟨k, -, rfl⟩ := Array.mem_map.1 hy
      rfl
  · intro y hy
    obtain ⟨o, ho, hy⟩ := Array.mem_flatMap.1 hy
    cases hs : l.scan with
    | none => rw [hs] at ho; simp at ho
    | some x =>
      rw [hs] at ho
      have : o = ("the scan", true, x) := by simpa using ho
      subst this
      obtain ⟨k, -, rfl⟩ := Array.mem_map.1 hy
      rfl
  · unfold NetworkCheck.Period.Listing.scanFiles
    cases h : l.scan <;> simp

theorem flag_at (l : NetworkCheck.Period.Listing) :
    ∀ i (hi : i < l.files.size), l.files[i].2.1 = decide (l.files.size - l.scanFiles ≤ i) := by
  obtain ⟨A, S, hf, hA, hS, hsz⟩ := files_split l
  rw [hf, ← hsz]
  intro j hj
  by_cases hjA : j < A.size
  · rw [Array.getElem_append_left hjA, hA _ (Array.getElem_mem hjA), eq_comm, decide_eq_false_iff_not]
    simp only [Array.size_append]
    omega
  · rw [Array.getElem_append_right (by omega), hS _ (Array.getElem_mem _), eq_comm, decide_eq_true_iff]
    simp only [Array.size_append]
    omega

theorem files_unlisted {l : NetworkCheck.Period.Listing} (hl : l.listed = false) : l.files.size = 0 := by
  unfold NetworkCheck.Period.Listing.listed at hl
  rw [Bool.or_eq_false_iff, Array.any_eq_false'] at hl
  obtain ⟨ha, hs⟩ := hl
  have hnot : ∀ y, y ∉ l.files := by
    intro y hy
    unfold NetworkCheck.Period.Listing.files NetworkCheck.Period.Listing.owners at hy
    obtain ⟨o, ho, -⟩ := Array.mem_flatMap.1 hy
    rcases Array.mem_append.1 ho with ho | ho
    · obtain ⟨⟨oo, i⟩, hm, ho⟩ := Array.mem_filterMap.1 ho
      cases oo
      · cases ho
      · exact ha _ (Array.mem_of_getElem? (Array.mem_zipIdx_iff_getElem?.1 hm)) rfl
    · cases hsc : l.scan with
      | none => rw [hsc] at ho; simp at ho
      | some x => rw [hsc] at hs; cases hs
  by_contra hne
  exact hnot _ (Array.getElem_mem (Nat.pos_of_ne_zero hne))

theorem blobsOf_of {bs cs : Array ByteArray} {l : NetworkCheck.Period.Listing} (hinj : BlobInj (bs ++ cs))
    (hsz : l.files.size = cs.size)
    (hd : ∀ i (h1 : i < l.files.size) (h2 : i < cs.size) w s x k, l.files[i] = (w, s, x, k) →
      x.files[k]! = Flock.hexOf (Flock.Sha512.hash cs[i])) :
    NetworkCheck.Period.blobsOf l (Core.Blobs.ofArray (bs ++ cs)) = .ok cs := by
  unfold NetworkCheck.Period.blobsOf
  refine FlockComplete.JsonRT.array_mapM_ok_of _ _ _ (by simp [hsz]) fun i h1 h2 => ?_
  have h1' : i < l.files.size := by simpa using h1
  simp only [Array.getElem_zipIdx, Nat.zero_add]
  rcases hf : l.files[i]'h1' with ⟨w, s, x, k⟩
  dsimp only
  rw [hd i h1' h2 w s x k hf]
  exact get_hash hinj (Array.mem_append.2 (Or.inr (Array.getElem_mem h2))) _

theorem ofFiles_of {n : Nat} {cs : Array ByteArray} {fs : Array (ByteArray × Flock.Circuit)}
    {l : NetworkCheck.Period.Listing} (hr : NetworkCheck.Period.ofRead n cs = .ok fs) (hn : l.scanFiles = n)
    (hsz : l.files.size = cs.size) : NetworkCheck.Period.ofFiles l cs = .ok fs := by
  obtain ⟨hs, hfile⟩ := ofRead_ok hr
  unfold NetworkCheck.Period.ofFiles
  refine FlockComplete.JsonRT.array_mapM_ok_of _ _ _ (by simp [hsz, hs]) fun i h1 h2 => ?_
  have h1' : i < l.files.size := by simpa [Array.size_zip, hsz] using h1
  have h1'' : i < cs.size := by omega
  simp only [Array.getElem_zipIdx, Array.getElem_zip, Nat.zero_add]
  rcases hf : l.files[i]'h1' with ⟨w, s, x, k⟩
  have hfl := flag_at l i h1'
  rw [hf, hsz, hn] at hfl
  (try dsimp only at hfl)
  dsimp only
  rw [hfl]
  exact (mapErrorOk _ _ _).2 (hfile i h1'' h2)

theorem filesOf_of {bs cs : Array ByteArray} {n : Nat} {fs : Array (ByteArray × Flock.Circuit)}
    {l : NetworkCheck.Period.Listing} (hinj : BlobInj (bs ++ cs)) (hr : NetworkCheck.Period.ofRead n cs = .ok fs)
    (hn : l.scanFiles = n) (hsz : l.files.size = cs.size)
    (hd : ∀ i (h1 : i < l.files.size) (h2 : i < cs.size) w s x k, l.files[i] = (w, s, x, k) →
      x.files[k]! = Flock.hexOf (Flock.Sha512.hash cs[i])) :
    NetworkCheck.Period.filesOf l (Core.Blobs.ofArray (bs ++ cs)) = .ok fs := by
  unfold NetworkCheck.Period.filesOf
  rw [blobsOf_of hinj hsz hd, ok_bind]
  exact ofFiles_of hr hn hsz

theorem filesOf_eq {bs cs : Array ByteArray} {n : Nat} (hinj : BlobInj (bs ++ cs))
    {p : NetworkCheck.Period.Params × NetworkCheck.Period.Listing} {c e pub prof}
    (hn : p.2.scanFiles = n) (h : NetworkCheck.Period.filesCheck (NetworkCheck.Period.ofRead n cs) p c e =
      .accept pub prof) :
    NetworkCheck.Period.filesOf p.2 (Core.Blobs.ofArray (bs ++ cs)) = NetworkCheck.Period.ofRead n cs := by
  cases hr : NetworkCheck.Period.ofRead n cs with
  | error e => rw [hr] at h; cases h
  | ok fs =>
    rw [hr] at h
    obtain ⟨hs, hfile⟩ := ofRead_ok hr
    unfold NetworkCheck.Period.filesCheck at h
    dsimp only at h
    split at h
    · cases h
    split at h
    · rename_i hcond
      have hl : p.2.listed = false := by
        cases hp : p.2.listed
        · rfl
        · rw [hp] at hcond; cases hcond
      have hfe : fs.size = 0 := by
        cases hp : fs.isEmpty
        · rw [hp, Bool.and_false] at hcond; cases hcond
        · exact Array.isEmpty_iff_size_eq_zero.1 hp
      have hcs : cs.size = 0 := by omega
      exact filesOf_of hinj hr hn (by rw [files_unlisted hl, hcs]) (fun i _ h2 => absurd h2 (by omega))
    · unfold NetworkCheck.Period.checkWith at h
      split at h
      · cases h
      split at h
      · cases h
      rename_i hw
      obtain ⟨hsz, hfs⟩ := circuitsWhy_none hw
      refine filesOf_of hinj hr hn (by omega) fun i h1 h2 w s x k hf => ?_
      have h3 : i < fs.size := by omega
      rw [hfs i h1 h3 w s x k hf, ofFile_sha (hfile i h2 h3)]

theorem net_mono {bs cs : Array ByteArray} {n : Nat} (hinj : BlobInj (bs ++ cs))
    {p : NetworkCheck.Period.Params × NetworkCheck.Period.Listing} {c e pub prof}
    (h : NetworkCheck.Period.readCheck n (NetworkCheck.Period.ofRead n cs) p c e = .accept pub prof) :
    NetworkCheck.Period.filesCheck (NetworkCheck.Period.filesOf p.2 (Core.Blobs.ofArray (bs ++ cs))) p c e =
      .accept pub prof := by
  unfold NetworkCheck.Period.readCheck at h
  split at h
  · cases h
  · rename_i hne
    have hn' : p.2.scanFiles = n := by simpa using hne
    rw [filesOf_eq hinj hn' h]
    exact h

theorem noTable_of_read {n : Nat} {cs : Array ByteArray}
    {p : NetworkCheck.Period.Params × NetworkCheck.Period.Listing} {c e pub prof}
    (h : NetworkCheck.Period.readCheck n (NetworkCheck.Period.ofRead n cs) p c e = .accept pub prof) :
    NoTable cs := by
  unfold NetworkCheck.Period.readCheck at h
  split at h
  · cases h
  unfold NetworkCheck.Period.filesCheck at h
  cases hr : NetworkCheck.Period.ofRead n cs with
  | error e => rw [hr] at h; cases h
  | ok fs =>
    rw [hr] at h
    dsimp only at h
    split at h
    · cases h
    rename_i hp
    obtain ⟨hs, hfile⟩ := ofRead_ok hr
    unfold NetworkCheck.Period.pinnedTableOf at hp
    intro b hb t ht
    obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hb
    have hi' : i < fs.size := by omega
    have hsha := ofFile_sha (hfile i hi hi')
    have hnone := Array.findSome?_eq_none_iff.1 hp (fs[i], i) (Array.mem_zipIdx_iff_getElem?.2 (by simp [hi']))
    rcases hfs : fs[i] with ⟨sha, cf⟩
    rw [hfs] at hnone hsha
    (try dsimp only at hnone)
    (try dsimp only at hsha)
    rw [Option.map_eq_none_iff, List.find?_eq_none] at hnone
    have := hnone t ht
    rw [beq_eq_false_iff_ne]
    intro heq
    apply this
    rw [hsha, heq]
    exact beq_self_eq_true _

/-! ## Compositions whose components accept as much -/

/-- `Pa` is `Pb` with a check that accepts whatever `Pb`'s accepts. -/
inductive BlobCovers (Pa Pb : Core.Protocol) : Prop
  | intro (g : Pb.Params → Pb.Commitment → Pb.Evidence → Core.Outcome Pb.Public Pb.Profile)
      (eq : Pa = { Pb with check := g })
      (acc : ∀ p c e pub prof, Pb.check p c e = .accept pub prof → g p c e = .accept pub prof)

theorem covers_onBlobs (P : Core.BlobProtocol) (B1 B2 : Core.Blobs)
    (hm : ∀ p c e pub prof, P.checkBlobs p c e B1 = .accept pub prof → P.checkBlobs p c e B2 = .accept pub prof) :
    BlobCovers (P.onBlobs B2) (P.onBlobs B1) :=
  ⟨fun p c e => P.checkBlobs p c e B2, rfl, hm⟩

theorem covers_plain (P : Core.Protocol) (B1 B2 : Core.Blobs) :
    BlobCovers (P.withoutBlobs.onBlobs B2) (P.withoutBlobs.onBlobs B1) :=
  covers_onBlobs P.withoutBlobs B1 B2 fun _ _ _ _ _ h => h

theorem covers_id {Pa Pb : Core.Protocol} (h : BlobCovers Pa Pb) : Pa.id = Pb.id := by
  obtain ⟨g, rfl, -⟩ := h
  rfl

theorem answer_eq_of (P : Core.Protocol) {a b e : ByteArray} {p : P.Params} {c : P.Commitment} {ev : P.Evidence}
    {pub : P.Public} {prof : P.Profile} (hp : (canonicalJson a >>= P.params) = .ok p)
    (hc : P.commitment b = .ok c) (he : P.evidence e = .ok ev) (hk : P.check p c ev = .accept pub prof) :
    P.answer a b e =
      { protocol := P.id, verdict := .accept, reason := none, digest := some (P.digest c),
        published := some (P.publicJson pub), profile := some (P.profileJson prof) } := by
  simp only [Core.Protocol.answer, hp, hc, he, hk]

theorem answer_of_covers {Pa Pb : Core.Protocol} (h : BlobCovers Pa Pb) {a b e : ByteArray}
    (hy : (Pb.answer a b e).verdict = .accept) : Pa.answer a b e = Pb.answer a b e := by
  obtain ⟨g, rfl, hg⟩ := h
  obtain ⟨p, c, ev, hp, hc, he, pub, prof, hk⟩ := Properties.Composition.answerAccepts Pb a b e hy
  exact (answer_eq_of { Pb with check := g } hp hc he (hg p c ev pub prof hk)).trans (answer_eq_of Pb hp hc he hk).symm

/-- Each of `xs` covers its place in `ys`. -/
inductive AllCover : List Core.Protocol → List Core.Protocol → Prop
  | nil : AllCover [] []
  | cons {Pa Pb : Core.Protocol} {xs ys : List Core.Protocol} : BlobCovers Pa Pb → AllCover xs ys →
      AllCover (Pa :: xs) (Pb :: ys)

theorem AllCover.ids {xs ys : List Core.Protocol} (h : AllCover xs ys) : xs.map (·.id) = ys.map (·.id) := by
  induction h with
  | nil => rfl
  | cons hc _ ih => simp only [List.map_cons, covers_id hc, ih]

theorem AllCover.zip_snd {xs ys : List Core.Protocol} (h : AllCover xs ys) {γ : Type} :
    ∀ (zs : List γ) (x : Core.Protocol × γ), x ∈ xs.zip zs → ∃ y, (y, x.2) ∈ ys.zip zs := by
  induction h with
  | nil => intro zs x hx; simp at hx
  | @cons Pa Pb xs ys _ _ ih =>
    intro zs x hx
    cases zs with
    | nil => simp at hx
    | cons z zs =>
      rw [List.zip_cons_cons, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact ⟨Pb, by simp⟩
      · obtain ⟨y, hy⟩ := ih zs x hx
        exact ⟨y, by simp [hy]⟩

theorem committedBy_covers {xs ys : List Core.Protocol} (h : AllCover xs ys) (s : String) :
    ∀ ev, Core.Composition.committedBy s xs ev = Core.Composition.committedBy s ys ev := by
  induction h with
  | nil => intro ev; rfl
  | @cons Pa Pb xs ys hc _ ih =>
    intro ev
    obtain ⟨g, rfl, -⟩ := hc
    cases ev with
    | nil => rfl
    | cons k ev =>
      have ih' := ih ev
      unfold Core.Composition.committedBy at ih' ⊢
      simp only [List.zip_cons_cons, List.find?_cons]
      cases hb : (Pb.id == s) <;> (try simp only) <;> first | rfl | exact ih'

theorem strays_covers {xs ys : List Core.Protocol} (h : AllCover xs ys) (ss : List String)
    (ev : List Core.Composition.Case) : Core.Composition.strays ss xs ev = Core.Composition.strays ss ys ev := by
  unfold Core.Composition.strays
  simp only [committedBy_covers h]

theorem answersOf_covers {xs ys : List Core.Protocol} (h : AllCover xs ys) :
    ∀ (ps : List Core.Composition.Named) (ev : List Core.Composition.Case),
      (∀ x ∈ Properties.Composition.triples ys ps ev,
        (x.1.answer x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence).verdict = .accept) →
      Core.Composition.answersOf xs ps ev = Core.Composition.answersOf ys ps ev := by
  induction h with
  | nil => intro ps ev _; rfl
  | @cons Pa Pb xs ys hc _ ih =>
    intro ps ev hacc
    cases ps with
    | nil => rfl
    | cons n ps =>
      cases ev with
      | nil => rfl
      | cons k ev =>
        have hY := hacc (Pb, n, k) (by simp [Properties.Composition.triples])
        have hrest := ih ps ev fun x hx => hacc x (by
          simp only [Properties.Composition.triples, List.zip_cons_cons] at hx ⊢
          exact List.mem_cons_of_mem _ hx)
        show (Pa.id, Pa.answer n.params.toUTF8 k.commitment k.evidence) :: Core.Composition.answersOf xs ps ev =
          (Pb.id, Pb.answer n.params.toUTF8 k.commitment k.evidence) :: Core.Composition.answersOf ys ps ev
        rw [answer_of_covers hc hY, covers_id hc, hrest]

theorem run_covers {xs ys : List Core.Protocol} (h : AllCover xs ys) {ps : List Core.Composition.Named}
    {ev : List Core.Composition.Case} {answers} (hr : Core.Composition.run ys ps ev = .ok answers) :
    Core.Composition.run xs ps ev = .ok (Core.Composition.answersOf xs ps ev) := by
  obtain ⟨h1, h2, h3, -⟩ := Properties.Composition.run_ok hr
  have e1 : ps.map (·.id) = xs.map (·.id) := h1.trans h.ids.symm
  have e2 : ev.map (·.id) = xs.map (·.id) := h2.trans h.ids.symm
  unfold Core.Composition.run
  simp only [e1, e2, ne_eq, not_true_eq_false, ↓reduceIte]
  split
  · rename_i P n' k' heq
    obtain ⟨y, hy⟩ := h.zip_snd _ _ (List.mem_of_find?_eq_some heq)
    have hpar : k'.params = n'.params := h3 _ hy
    have hp := List.find?_some heq
    simp [hpar] at hp
  · rfl

theorem checkSame_covers {xs ys : List Core.Protocol} (h : AllCover xs ys) {id : String} {ss : List String}
    {ps : List Core.Composition.Named} {c : ByteArray} {ev : List Core.Composition.Case} {pub prof}
    (hy : Core.Composition.checkSame id ss ys ps c ev = .accept pub prof) :
    Core.Composition.checkSame id ss xs ps c ev = .accept pub prof := by
  obtain ⟨⟨-, hacc, -⟩, -⟩ := Properties.Composition.acceptSame _ _ _ _ _ _ _ _ hy
  unfold Core.Composition.checkSame at hy ⊢
  cases hr : Core.Composition.run ys ps ev with
  | error r => rw [hr] at hy; cases hy
  | ok answers =>
    obtain ⟨-, -, -, rfl⟩ := Properties.Composition.run_ok hr
    rw [hr] at hy
    rw [run_covers h hr, strays_covers h, answersOf_covers h ps ev fun x hx =>
      (Properties.Composition.answer_accepts_iff _ _ _ _).2 (hacc x hx)]
    exact hy

theorem tiedSameB_covers {id : String} {ss : List String} {ties : List Core.Composition.Tie}
    {csX csY : List Core.BlobProtocol} {B1 B2 : Core.Blobs}
    (h : AllCover (csX.map (·.onBlobs B2)) (csY.map (·.onBlobs B1))) :
    BlobCovers ((Core.Composition.tiedSameB id ss ties csX).onBlobs B2)
      ((Core.Composition.tiedSameB id ss ties csY).onBlobs B1) := by
  refine ⟨fun ps c ev => Core.Composition.holdTies ties ev
    (Core.Composition.checkSame id ss (csX.map (·.onBlobs B2)) ps c ev), rfl, fun ps c ev pub prof hy => ?_⟩
  have hy2 : Core.Composition.holdTies ties ev
      (Core.Composition.checkSame id ss (csY.map (·.onBlobs B1)) ps c ev) = .accept pub prof := hy
  have hy' := (Properties.Composition.acceptTied ties ev _ pub prof hy2).1
  show Core.Composition.holdTies ties ev (Core.Composition.checkSame id ss (csX.map (·.onBlobs B2)) ps c ev) =
    .accept pub prof
  rw [checkSame_covers h hy']
  rw [hy'] at hy2
  exact hy2

theorem head_accepts {id : String} {ss : List String} {ties : List Core.Composition.Tie} {P0 : Core.BlobProtocol}
    {rest : List Core.BlobProtocol} {B : Core.Blobs} {a b e : ByteArray}
    (h : (((Core.Composition.tiedSameB id ss ties (P0 :: rest)).onBlobs B).answer a b e).verdict = .accept) :
    ∃ p c ev pub prof, P0.checkBlobs p c ev B = .accept pub prof := by
  obtain ⟨ps, c, ev, -, -, -, pub, prof, hk⟩ := Properties.Composition.answerAccepts _ a b e h
  have hk2 : Core.Composition.holdTies ties ev
      (Core.Composition.checkSame id ss ((P0 :: rest).map (·.onBlobs B)) ps c ev) = .accept pub prof := hk
  have hs := (Properties.Composition.acceptTied ties ev _ pub prof hk2).1
  obtain ⟨⟨⟨h1, h2, -, -⟩, hacc, -⟩, -⟩ := Properties.Composition.acceptSame _ _ _ _ _ _ _ _ hs
  cases ps with
  | nil => simp at h1
  | cons n ps =>
    cases ev with
    | nil => simp at h2
    | cons k ev =>
      obtain ⟨p, c', e', -, -, -, pub', prof', hk'⟩ :=
        hacc (P0.onBlobs B, n, k) (by simp [Properties.Composition.triples])
      exact ⟨p, c', e', pub', prof', hk'⟩

/-! ## Sampled proofs, one stage under `--zk` -/

section Zk

open SampledProofs.OneStageZk (Run RunFiles Reads)

variable {B1 B2 : Core.Blobs}

theorem push_all {I rd programSha want} {r : Run} {s : RunFiles} {record st coinKey} {v : V Unit} {r' : Run}
    (h : SampledProofs.OneStageZk.push I rd programSha want r s record st coinKey v = .ok r')
    (hr' : r'.all = true) : r.all = true := by
  cases v with
  | error e =>
    unfold SampledProofs.OneStageZk.push at h
    (try dsimp only at h)
    cases h
    simp at hr'
  | ok u =>
    cases u
    unfold SampledProofs.OneStageZk.push at h
    (try dsimp only at h)
    cases hc : ((programSha.map Json.str) != some want)
    · simp only [hc, Bool.false_eq_true, ↓reduceIte] at h
      obtain ⟨units, -, h⟩ := bindOk_split h
      cases h
      exact hr'
    · simp only [hc, ↓reduceIte] at h
      cases h

theorem push_mono {I rd programSha want} {r : Run} {s : RunFiles} {record st coinKey} {v1 v2 : V Unit} {r' : Run}
    (h : SampledProofs.OneStageZk.push I rd programSha want r s record st coinKey v1 = .ok r')
    (hr' : r'.all = true) (hv : ExtT v1 v2) :
    SampledProofs.OneStageZk.push I rd programSha want r s record st coinKey v2 = .ok r' := by
  cases v1 with
  | error e =>
    unfold SampledProofs.OneStageZk.push at h
    (try dsimp only at h)
    cases h
    simp at hr'
  | ok u => rw [hv.run u rfl trivial]; exact h

theorem verdictOf_ext {I scope ksFiles coinKeys record draw st} {f1 f2 : V (Array ByteArray)} {coinKey}
    (h : ExtT f1 f2) :
    ExtT (SampledProofs.OneStageZk.verdictOf I scope ksFiles coinKeys record draw st f1 coinKey)
      (SampledProofs.OneStageZk.verdictOf I scope ksFiles coinKeys record draw st f2 coinKey) := by
  unfold SampledProofs.OneStageZk.verdictOf
  exact ExtP.bind h fun _ => ExtP.refl _ _

theorem zk_filesOf_ext (hB : BlobSup B1 B2) (s : RunFiles) (st : SampledProofs.OneStageZk.SetUp) :
    ExtT (SampledProofs.OneStageZk.filesOf B1 s st) (SampledProofs.OneStageZk.filesOf B2 s st) := by
  unfold SampledProofs.OneStageZk.filesOf
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _) | split
    | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM | apply ExtP.arrayForIn | apply ExtP.mapError
    | apply ExtP.map | apply ExtP.bind | intro _ | dsimp only)

theorem step_mono (hB : BlobSup B1 B2) {I rd scope ksFiles programSha want} {r : Run} {s : RunFiles} {r' : Run}
    (h : SampledProofs.OneStageZk.step I rd B1 scope ksFiles programSha want r s = .ok r') (hr' : r'.all = true) :
    SampledProofs.OneStageZk.step I rd B2 scope ksFiles programSha want r s = .ok r' := by
  unfold SampledProofs.OneStageZk.step at h ⊢
  obtain ⟨x, hx, h⟩ := bindOk_split h
  rw [hB.get _ _ _ hx, ok_bind]
  cases hrec : String.fromUTF8? x with
  | none =>
    simp only [hrec] at h
    cases h
  | some record =>
    simp only [hrec] at h ⊢
    exact push_mono h hr' (verdictOf_ext (zk_filesOf_ext hB _ _))

theorem step_all {I rd scope ksFiles programSha want} {r : Run} {s : RunFiles} {r' : Run}
    (h : SampledProofs.OneStageZk.step I rd B1 scope ksFiles programSha want r s = .ok r') (hr' : r'.all = true) :
    r.all = true := by
  unfold SampledProofs.OneStageZk.step at h
  obtain ⟨x, -, h⟩ := bindOk_split h
  cases hrec : String.fromUTF8? x with
  | none =>
    simp only [hrec] at h
    cases h
  | some record =>
    simp only [hrec] at h
    exact push_all h hr'

theorem listFoldlM_mono (hB : BlobSup B1 B2) {I rd scope ksFiles programSha want} :
    ∀ (l : List RunFiles) (init r : Run),
      l.foldlM (SampledProofs.OneStageZk.step I rd B1 scope ksFiles programSha want) init = .ok r →
      r.all = true →
      l.foldlM (SampledProofs.OneStageZk.step I rd B2 scope ksFiles programSha want) init = .ok r ∧
        init.all = true
  | [], init, r, h, hr => by
    rw [List.foldlM_nil] at h ⊢
    cases h
    exact ⟨rfl, hr⟩
  | s :: l, init, r, h, hr => by
    rw [List.foldlM_cons] at h ⊢
    obtain ⟨r1, h1, h2⟩ := bindOk_split h
    obtain ⟨h3, hr1⟩ := listFoldlM_mono hB l r1 r h2 hr
    rw [step_mono hB h1 hr1, ok_bind]
    exact ⟨h3, step_all h1 hr1⟩

theorem zk_fold_ext (hB : BlobSup B1 B2) {I rd scope ksFiles programSha want} (sessions : Array RunFiles)
    (init : Run) :
    ExtP (fun r : Run => r.all = true)
      (sessions.foldlM (SampledProofs.OneStageZk.step I rd B1 scope ksFiles programSha want) init)
      (sessions.foldlM (SampledProofs.OneStageZk.step I rd B2 scope ksFiles programSha want) init) := by
  constructor
  intro r h hr
  rw [← Array.foldlM_toList] at h ⊢
  exact (listFoldlM_mono hB _ _ _ h hr).1

theorem zk_fold_bind (hB : BlobSup B1 B2) {I rd scope ksFiles programSha want} (sessions : Array RunFiles)
    (init : Run) {β : Type} {P : β → Prop} (f : Run → V β) (hf : ∀ a b, f a = .ok b → P b → a.all = true) :
    ExtP P (sessions.foldlM (SampledProofs.OneStageZk.step I rd B1 scope ksFiles programSha want) init >>= f)
      (sessions.foldlM (SampledProofs.OneStageZk.step I rd B2 scope ksFiles programSha want) init >>= f) :=
  ExtP.bind_back (zk_fold_ext hB sessions init) (fun _ => ExtP.refl _ _) hf

theorem zk_run_mono (hB : BlobSup B1 B2) (p : SampledProofs.OneStageZk.Own)
    (sessions : Array RunFiles) (pu : Option (Array Nat)) :
    ExtP (fun r : Run => r.all = true) (SampledProofs.OneStageZk.run p sessions pu B1)
      (SampledProofs.OneStageZk.run p sessions pu B2) := by
  unfold SampledProofs.OneStageZk.run
  simp only [inputsOf_tab hB.tables]
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _)
    | ((with_reducible apply zk_fold_bind hB) <;> exact fun a b hab hb => by
        (repeat' split at hab) <;> (first | assumption | (cases hab; exact hb)))
    | split | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM | apply ExtP.arrayForIn
    | apply ExtP.mapError | apply ExtP.map | apply ExtP.bind | intro _)

theorem treesHeld_ext (hB : BlobSup B1 B2) (ts : Array SampledProofs.Challenger.LoggedTree)
    (e : SampledProofs.OneStageZk.Evidence) :
    ExtT (SampledProofs.OneStageZk.treesHeld ts e B1) (SampledProofs.OneStageZk.treesHeld ts e B2) := by
  unfold SampledProofs.OneStageZk.treesHeld
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _) | split
    | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM | apply ExtP.arrayForIn | apply ExtP.mapError
    | apply ExtP.map | apply ExtP.bind | intro _)

/-- `checkBlobsAt`'s checks before v1's (the one session, the trees, the population and the public file) and its run,
on the blobs `bs`. -/
def zkPre (p : SampledProofs.OneStageZk.Params) (c : SampledProofs.OneStage.Commitment)
    (e : SampledProofs.OneStageZk.Evidence) (bs : Core.Blobs) : V Run := do
  SampledProofs.OneStageZk.oneSession e.sessions
  SampledProofs.OneStageZk.treesHeld (← SampledProofs.OneStageZk.treesOf p e) e bs
  SampledProofs.OneStageZk.popHeld p.base.verifier c.record
  SampledProofs.OneStageZk.publicHeld c.record (← bs.get p.own.pub "public file")
  SampledProofs.OneStageZk.run p.own e.sessions (some e.base.provedUnits) bs

theorem zkPre_mono (hB : BlobSup B1 B2) (p : SampledProofs.OneStageZk.Params) (c : SampledProofs.OneStage.Commitment)
    (e : SampledProofs.OneStageZk.Evidence) :
    ExtP (fun r : Run => r.all = true) (zkPre p c e B1) (zkPre p c e B2) := by
  unfold zkPre
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _)
    | exact treesHeld_ext hB _ _ | exact zk_run_mono hB _ _ _ | apply ExtP.bind | intro _)

theorem zk_at_mono (hB : BlobSup B1 B2) {p : SampledProofs.OneStageZk.Params} {c ξ}
    {e : SampledProofs.OneStageZk.Evidence} {pub prof}
    (h : SampledProofs.OneStageZk.checkBlobsAt p c ξ e B1 = .accept pub prof) :
    SampledProofs.OneStageZk.checkBlobsAt p c ξ e B2 = .accept pub prof := by
  unfold SampledProofs.OneStageZk.checkBlobsAt at h ⊢
  dsimp only at h ⊢
  split at h
  · cases h
  rename_i r hr
  have hall : r.all = true := by
    cases hk : SampledProofs.OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | accept pub' prof' =>
      rw [hk] at h
      (try dsimp only at h)
      cases hra : r.all
      · simp [hra] at h
      · rfl
    | reject s => rw [hk] at h; cases h
    | undecided s => rw [hk] at h; cases h
  have h2 : zkPre p c e B2 = .ok r := (zkPre_mono hB p c e).run r hr hall
  split
  · rename_i err herr
    have h3 : zkPre p c e B2 = .error err := herr
    rw [h2] at h3
    cases h3
  · rename_i r' hr'
    have h3 : zkPre p c e B2 = .ok r' := hr'
    rw [h2] at h3
    cases h3
    exact h

theorem zk_mono (hB : BlobSup B1 B2) {p : SampledProofs.OneStageZk.Params} {c}
    {e : SampledProofs.OneStageZk.Evidence} {pub prof}
    (h : SampledProofs.OneStageZk.checkBlobs p c e B1 = .accept pub prof) :
    SampledProofs.OneStageZk.checkBlobs p c e B2 = .accept pub prof := by
  unfold SampledProofs.OneStageZk.checkBlobs Core.atCoins at h ⊢
  split at h
  · exact zk_at_mono hB h
  · cases h

end Zk

/-! ## Sampled proofs, recursive -/

section Recursive

variable {B1 B2 : Core.Blobs}

theorem innerHeld_ext (hB : BlobSup B1 B2) (o : VStar.Recursive.Own) (pr : VStar.Recursive.Proofs) :
    ExtT (VStar.Recursive.innerHeld o pr B1) (VStar.Recursive.innerHeld o pr B2) := by
  unfold VStar.Recursive.innerHeld
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _) | split
    | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM | apply ExtP.arrayForIn | apply ExtP.mapError
    | apply ExtP.map | apply ExtP.bind | intro _)

theorem partSize_ext (hB : BlobSup B1 B2) (j : Nat) (p : VStar.Recursive.PartOwn) (s : VStar.Recursive.PartProof) :
    ExtT (VStar.Recursive.partSize j p s B1) (VStar.Recursive.partSize j p s B2) := by
  unfold VStar.Recursive.partSize
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _) | split
    | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM | apply ExtP.arrayForIn | apply ExtP.mapError
    | apply ExtP.map | apply ExtP.bind | intro _)

theorem unitsOf_ext (hB : BlobSup B1 B2) (ok : Bool) (o : VStar.Recursive.Own) (pr : VStar.Recursive.Proofs) :
    ExtT (VStar.Recursive.unitsOf ok o pr B1) (VStar.Recursive.unitsOf ok o pr B2) := by
  unfold VStar.Recursive.unitsOf
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _) | split
    | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM | apply ExtP.arrayForIn | apply ExtP.mapError
    | apply ExtP.map | apply ExtP.bind | intro _)

theorem partVerdict_ext (hB : BlobSup B1 B2) (h : VStar.Recursive.InnerHeld) (j : Nat)
    (p : VStar.Recursive.PartOwn) (s : VStar.Recursive.PartProof) :
    ExtP (fun v : V Unit => v = .ok ()) (VStar.Recursive.partVerdict h j p s B1)
      (VStar.Recursive.partVerdict h j p s B2) := by
  unfold VStar.Recursive.partVerdict
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _)
    | with_reducible apply ExtP.pure_ok | split | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM
    | apply ExtP.arrayForIn | apply ExtP.mapError | apply ExtP.map | apply ExtP.bind | intro _ | dsimp only)

theorem partsSize_ext (hB : BlobSup B1 B2) :
    ∀ (j : Nat) (ps : List VStar.Recursive.PartOwn) (ss : List VStar.Recursive.PartProof),
      ExtT (VStar.Recursive.partsSize B1 j ps ss) (VStar.Recursive.partsSize B2 j ps ss)
  | j, p :: ps, s :: ss => by
    simp only [VStar.Recursive.partsSize]
    exact ExtP.bind (partSize_ext hB j p s) fun _ => ExtP.bind (partsSize_ext hB (j + 1) ps ss) fun _ =>
      ExtP.refl _ _
  | j, [], ss => by simp only [VStar.Recursive.partsSize]; exact ExtP.refl _ _
  | j, p :: ps, [] => by simp only [VStar.Recursive.partsSize]; exact ExtP.refl _ _

theorem statementSize_ext (hB : BlobSup B1 B2) (h : VStar.Recursive.InnerHeld) (o : VStar.Recursive.Own)
    (pr : VStar.Recursive.Proofs) :
    ExtT (VStar.Recursive.statementSize h o pr B1) (VStar.Recursive.statementSize h o pr B2) := by
  unfold VStar.Recursive.statementSize
  repeat' (first
    | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (get_ext hB _ _)
    | exact partsSize_ext hB _ _ _ | split | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM
    | apply ExtP.arrayForIn | apply ExtP.mapError | apply ExtP.map | apply ExtP.bind | intro _)

theorem statementFits_ext (hB : BlobSup B1 B2) (limit : Nat) (h : VStar.Recursive.InnerHeld)
    (o : VStar.Recursive.Own) (pr : VStar.Recursive.Proofs) :
    ExtT (VStar.Recursive.statementFits limit h o pr B1) (VStar.Recursive.statementFits limit h o pr B2) := by
  unfold VStar.Recursive.statementFits
  exact ExtP.bind (statementSize_ext hB h o pr) fun _ => ExtP.refl _ _

theorem partsAt_ext (hB : BlobSup B1 B2) (h : VStar.Recursive.InnerHeld) (ss : Array VStar.Stmt)
    (o : VStar.Recursive.Own) (pr : VStar.Recursive.Proofs) :
    ∀ js : List Nat, ExtP (fun l : List (String × V Unit) => ∀ x ∈ l, x.2 = .ok ())
      (VStar.Recursive.partVerdicts.partsAt h ss o pr B1 js) (VStar.Recursive.partVerdicts.partsAt h ss o pr B2 js)
  | [] => by simp only [VStar.Recursive.partVerdicts.partsAt]; exact ExtP.refl _ _
  | j :: js => by
    simp only [VStar.Recursive.partVerdicts.partsAt]
    split
    · refine ExtP.bind_back (partVerdict_ext hB h j _ _) (fun v => ?_) (fun v b hab hb => ?_)
      · refine ExtP.bind_back (partsAt_ext hB h ss o pr js) (fun rest => ExtP.refl _ _)
          (fun rest b hab hb => ?_)
        cases hab
        exact fun x hx => hb x (List.mem_cons_of_mem _ hx)
      · obtain ⟨rest, -, hab⟩ := bindOk_split hab
        cases hab
        exact hb _ List.mem_cons_self
    · exact ExtP.refl _ _

theorem partVerdicts_ext (hB : BlobSup B1 B2) (o : VStar.Recursive.Own) (pr : VStar.Recursive.Proofs) :
    ExtP (fun x : VStar.Recursive.InnerHeld × Array (String × V Unit) => x.2.all (·.2.toBool) = true)
      (VStar.Recursive.partVerdicts o pr B1) (VStar.Recursive.partVerdicts o pr B2) := by
  unfold VStar.Recursive.partVerdicts
  refine ExtP.bind (innerHeld_ext hB o pr) fun h => ?_
  refine ExtP.bind (ExtP.refl _ _) fun ss => ?_
  refine ExtP.bind (ExtP.refl _ _) fun _ => ?_
  refine ExtP.bind (ExtP.refl _ _) fun _ => ?_
  refine ExtP.bind (ExtP.refl _ _) fun _ => ?_
  refine ExtP.bind_back (partsAt_ext hB h ss o pr _)
    (fun vs => ExtP.bind (statementFits_ext hB _ h o pr) fun _ => ExtP.refl _ _) (fun vs b hab hb => ?_)
  obtain ⟨u, -, hab⟩ := bindOk_split hab
  cases u
  cases hab
  intro x hx
  exact ok_of_toBool (Array.all_eq_true'.1 hb x (by simpa using hx))

theorem canon_bool (b : Bool) : Flock.canon (.bool b) = .ok (if b then "true" else "false") := by
  simp only [Flock.canon]
  rfl

theorem jeq_bool {b : Bool} (h : SampledProofs.jeq (.bool b) (.bool true) = true) : b = true := by
  cases b with
  | false =>
    unfold SampledProofs.jeq at h
    rw [canon_bool, canon_bool] at h
    simp at h
  | true => rfl

theorem ok_of_verdictsAccept {name : String} {ok : Bool} {session why : String} {units : Array Nat}
    {parts : Array Json} {programSha : Option String}
    (h : Properties.SampledProofs.VerdictsAccept
      #[(name, VStar.Recursive.verdictJson ok session why units parts programSha)]) : ok = true := by
  have h2 := h.1 _ (Array.mem_singleton.2 rfl) (by
    show SampledProofs.getD (VStar.Recursive.verdictJson ok session why units parts programSha) "accepted" ≠ .null
    rw [VStar.Recursive.verdictJson_accepted]
    intro h
    cases h)
  (try dsimp only at h2)
  rw [VStar.Recursive.verdictJson_accepted] at h2
  exact jeq_bool h2

theorem verdicts_ext (hB : BlobSup B1 B2) (o : VStar.Recursive.Own) (session : String)
    (pr : VStar.Recursive.Proofs) (pu : Option (Array Nat)) :
    ExtP (fun vs => Properties.SampledProofs.VerdictsAccept vs) (VStar.Recursive.verdicts o session pr pu B1)
      (VStar.Recursive.verdicts o session pr pu B2) := by
  unfold VStar.Recursive.verdicts
  refine ExtP.bind_back (partVerdicts_ext hB o pr) (fun x => ?_) (fun x b hab hb => ?_)
  · repeat' (first
      | with_reducible exact ExtP.refl _ _ | with_reducible exact ExtP.of_ext (unitsOf_ext hB _ _ _)
      | split | apply ExtP.arrayMapM | apply ExtP.listMapM | apply ExtP.optionMapM | apply ExtP.arrayForIn
      | apply ExtP.mapError | apply ExtP.map | apply ExtP.bind | intro _)
  · (try dsimp only at hab)
    obtain ⟨u1, -, hab⟩ := bindOk_split hab
    cases u1
    obtain ⟨units, -, hab⟩ := bindOk_split hab
    obtain ⟨u2, -, hab⟩ := bindOk_split hab
    cases u2
    cases hab
    exact ok_of_verdictsAccept hb

theorem rec_mono (hB : BlobSup B1 B2) {p : VStar.Recursive.Params} {c} {e : VStar.Recursive.Evidence} {pub prof}
    (h : VStar.Recursive.checkBlobs p c e B1 = .accept pub prof) :
    VStar.Recursive.checkBlobs p c e B2 = .accept pub prof := by
  unfold VStar.Recursive.checkBlobs at h ⊢
  split at h
  · cases h
  rename_i session hs
  split at h
  · cases h
  rename_i vs hv
  obtain ⟨_, _, _, -, -, -, hva, -⟩ := (Properties.SampledProofs.acceptMeans_iff _ _ _ _ _).1 h
  rw [(verdicts_ext hB p.own session e.proofs _).run vs hv hva]
  exact h

theorem sampled_mono (hB : BlobSup B1 B2) {p : VStar.Recursive.Sampled.Params} {c}
    {e : VStar.Recursive.Sampled.Evidence} {pub prof}
    (h : VStar.Recursive.Sampled.checkBlobs p c e B1 = .accept pub prof) :
    VStar.Recursive.Sampled.checkBlobs p c e B2 = .accept pub prof := by
  cases p with
  | inl p =>
    cases e with
    | inl e => exact zk_mono hB h
    | inr e => cases h
  | inr p =>
    cases e with
    | inl e => cases h
    | inr e => exact rec_mono hB h

theorem sampled_covers (hB : BlobSup B1 B2) :
    BlobCovers (VStar.Recursive.Sampled.protocol.onBlobs B2) (VStar.Recursive.Sampled.protocol.onBlobs B1) :=
  covers_onBlobs VStar.Recursive.Sampled.protocol B1 B2 fun _ _ _ _ _ h => sampled_mono hB h

end Recursive

/-! ## Remote with the network certifier and without PoUW -/

theorem net_covers {bs cs : Array ByteArray} {n : Nat} (hinj : BlobInj (bs ++ cs)) :
    BlobCovers (NetworkCheck.Period.blobProtocol.onBlobs (Core.Blobs.ofArray (bs ++ cs)))
      ((NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs.onBlobs
        (Core.Blobs.ofArray bs)) :=
  ⟨fun (p : NetworkCheck.Period.Params × NetworkCheck.Period.Listing) c e =>
      NetworkCheck.Period.filesCheck (NetworkCheck.Period.filesOf p.2 (Core.Blobs.ofArray (bs ++ cs))) p c e, rfl,
    fun p c e pub prof h => net_mono (n := n) hinj (p := p) (c := c) (e := e) (pub := pub) (prof := prof) h⟩

theorem rnn_at {p c e : ByteArray} {bs cs : Array ByteArray} {n : Nat} (hinj : BlobInj (bs ++ cs))
    (h : ((Core.RemoteNetworkNoPouw.protocolWith
      (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).answer p c e bs).verdict =
        .accept) :
    Core.RemoteNetworkNoPouw.protocol.answer p c e (bs ++ cs) =
      (Core.RemoteNetworkNoPouw.protocolWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).answer p c e bs := by
  have h' : (((Core.Composition.tiedSameB Core.RemoteNetworkNoPouw.protocolId Core.Remote.tied
      Core.RemoteNetworkNoPouw.ties (Core.RemoteNetworkNoPouw.componentsWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs)).onBlobs
          (Core.Blobs.ofArray bs)).answer p c e).verdict = .accept := h
  obtain ⟨p0, c0, e0, pub0, prof0, h0⟩ := head_accepts h'
  have h0' : NetworkCheck.Period.readCheck n (NetworkCheck.Period.ofRead n cs) p0 c0 e0 = .accept pub0 prof0 := h0
  have hsup := sup_of (bs := bs) (noTable_of_read h0')
  have hcov : AllCover
      ((Core.RemoteNetworkNoPouw.componentsWith NetworkCheck.Period.blobProtocol).map
        (·.onBlobs (Core.Blobs.ofArray (bs ++ cs))))
      ((Core.RemoteNetworkNoPouw.componentsWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).map
          (·.onBlobs (Core.Blobs.ofArray bs))) :=
    AllCover.cons (net_covers hinj)
      (AllCover.cons (covers_plain _ _ _) (AllCover.cons (sampled_covers hsup) AllCover.nil))
  exact answer_of_covers (tiedSameB_covers hcov) h'

/-- **At remote with the network certifier and without PoUW, the verifier of record agrees with the executable that
reads the period's circuit files by path**, unless SHA-512 collides among the blobs. -/
theorem agree_rnn : Verity.FilesByPathAgreeWith .remoteNetworkNoPouw (Verity.byPath .remoteNetworkNoPouw) := by
  intro p c e bs cs h
  by_cases hcol : Hash.CollisionIn Verity.sha512 ((bs ++ cs).toList.map (·.data.toList))
  · exact Or.inr hcol
  · exact Or.inl (rnn_at (n := NetworkCheck.Period.scanCount
      (Core.PeriodFiles.certifierParams ((canonicalJson p).toOption.getD .null))) (inj_of_noCollision hcol) h)

/-! ## Remote with the network certifier and PoUW -/

theorem rn_at {p c e : ByteArray} {bs cs : Array ByteArray} {n : Nat} (hinj : BlobInj (bs ++ cs))
    (h : ((Core.RemoteNetwork.protocolWith
      (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).answer p c e bs).verdict =
        .accept) :
    Core.RemoteNetwork.protocol.answer p c e (bs ++ cs) =
      (Core.RemoteNetwork.protocolWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).answer p c e bs := by
  have h' : (((Core.Composition.tiedSameB Core.RemoteNetwork.protocolId Core.Remote.tied
      Core.RemoteNetwork.ties (Core.RemoteNetwork.componentsWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs)).onBlobs
          (Core.Blobs.ofArray bs)).answer p c e).verdict = .accept := h
  obtain ⟨p0, c0, e0, pub0, prof0, h0⟩ := head_accepts h'
  have h0' : NetworkCheck.Period.readCheck n (NetworkCheck.Period.ofRead n cs) p0 c0 e0 = .accept pub0 prof0 := h0
  have hsup := sup_of (bs := bs) (noTable_of_read h0')
  have hcov : AllCover
      ((Core.RemoteNetwork.componentsWith NetworkCheck.Period.blobProtocol).map
        (·.onBlobs (Core.Blobs.ofArray (bs ++ cs))))
      ((Core.RemoteNetwork.componentsWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).map
          (·.onBlobs (Core.Blobs.ofArray bs))) :=
    AllCover.cons (net_covers hinj) (AllCover.cons (covers_plain _ _ _)
      (AllCover.cons (covers_plain _ _ _) (AllCover.cons (sampled_covers hsup) AllCover.nil)))
  exact answer_of_covers (tiedSameB_covers hcov) h'

/-- **At remote with the network certifier and PoUW, the verifier of record agrees with the executable that reads the
period's circuit files by path**, unless SHA-512 collides among the blobs. -/
theorem agree_rn : Verity.FilesByPathAgreeWith .remoteNetwork (Verity.byPath .remoteNetwork) := by
  intro p c e bs cs h
  by_cases hcol : Hash.CollisionIn Verity.sha512 ((bs ++ cs).toList.map (·.data.toList))
  · exact Or.inr hcol
  · exact Or.inl (rn_at (n := NetworkCheck.Period.scanCount
      (Core.PeriodFiles.certifierParams ((canonicalJson p).toOption.getD .null))) (inj_of_noCollision hcol) h)

/-! ## Onsite -/

theorem onsite_at {p c e : ByteArray} {bs cs : Array ByteArray} {n : Nat} (hinj : BlobInj (bs ++ cs))
    (h : ((Core.Onsite.protocolWith
      (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).answer p c e bs).verdict =
        .accept) :
    Core.Onsite.protocol.answer p c e (bs ++ cs) =
      (Core.Onsite.protocolWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).answer p c e bs := by
  have h' : (((Core.Composition.tiedSameB "verity/onsite/v0" Core.Onsite.tied [Core.Onsite.store]
      (Core.Onsite.componentsWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs)).onBlobs
          (Core.Blobs.ofArray bs)).answer p c e).verdict = .accept := h
  have hcov : AllCover
      ((Core.Onsite.componentsWith NetworkCheck.Period.blobProtocol).map
        (·.onBlobs (Core.Blobs.ofArray (bs ++ cs))))
      ((Core.Onsite.componentsWith
        (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).map
          (·.onBlobs (Core.Blobs.ofArray bs))) :=
    AllCover.cons (net_covers hinj) (AllCover.cons (covers_plain _ _ _) (AllCover.cons (covers_plain _ _ _)
      (AllCover.cons (covers_plain _ _ _) (AllCover.cons (covers_plain _ _ _) (AllCover.cons (covers_plain _ _ _)
        (AllCover.cons (covers_plain _ _ _) AllCover.nil))))))
  exact answer_of_covers (tiedSameB_covers hcov) h'

/-- **At onsite, the verifier of record agrees with the executable that reads the period's circuit files by path**,
unless SHA-512 collides among the blobs. -/
theorem agree_onsite : Verity.FilesByPathAgreeWith .onsite (Verity.byPath .onsite) := by
  intro p c e bs cs h
  by_cases hcol : Hash.CollisionIn Verity.sha512 ((bs ++ cs).toList.map (·.data.toList))
  · exact Or.inr hcol
  · exact Or.inl (onsite_at (n := NetworkCheck.Period.scanCount
      (Core.PeriodFiles.certifierParams ((canonicalJson p).toOption.getD .null))) (inj_of_noCollision hcol) h)

/-! ## An executable that accepts everything -/

theorem ff_not_ascii : (ByteArray.mk #[0xff]).data.all (· < 0x7f) = false := by
  cases h : (ByteArray.mk #[0xff]).data.all (· < 0x7f)
  · rfl
  · have h0 := Array.all_eq_true.1 h 0 (by decide)
    revert h0
    decide

theorem canonicalJson_ff : ∃ m, Core.canonicalJson (ByteArray.mk #[0xff]) = .error m := by
  unfold Core.canonicalJson
  rw [ff_not_ascii]
  exact ⟨_, rfl⟩

/-- The verifier of record refuses parameters holding a byte at or above `0x7f`, whatever else it is given. -/
theorem verifier_refuses_ff (k : Verity.Registration.Composite) (c e : ByteArray) (bs : Array ByteArray) :
    ((Verity.verifier k).answer (ByteArray.mk #[0xff]) c e bs).verdict ≠ .accept := by
  intro h
  obtain ⟨p, -, -, hp, -⟩ := Properties.Composition.answerAccepts ((Verity.verifier k).on bs) _ _ _ h
  obtain ⟨m, hm⟩ := canonicalJson_ff
  rw [hm] at hp
  cases hp

/-- **`Verity.circuit_files_by_path_agree`'s nontrivial lemma**: where an executable that accepts everything
(`Verity.acceptAllByPath`) stands for the one that reads the circuit files by path, onsite's verifier of record
disagrees with it: at parameters holding the byte `0xff` and no blobs, it refuses, and nothing collides. -/
theorem _root_.Verity.circuit_files_by_path_agree_nontrivial :
    ¬ Verity.FilesByPathAgreeWith .onsite Verity.acceptAllByPath := by
  intro h
  rcases h (ByteArray.mk #[0xff]) ByteArray.empty ByteArray.empty #[] #[] rfl with h | h
  · exact verifier_refuses_ff .onsite _ _ _ (congrArg Core.Answer.verdict h)
  · obtain ⟨x, hx, -⟩ := h
    simp at hx

/-! ## An empty period -/

theorem empty_check : NetworkCheck.Period.check emptyParams emptyCommitment emptyEvidence =
    .accept (NetworkCheck.Period.terms #[] #[] #[] 0 1 0) { link := "", windows := 0 } := by
  simp [NetworkCheck.Period.check, emptyParams, emptyCommitment, emptyEvidence, emptySync,
    NetworkCheck.Period.auditChecks, NetworkCheck.Period.wiredChecks, NetworkCheck.Period.stateChecks,
    NetworkCheck.Period.heldChecks, NetworkCheck.Period.rowsWfCheck, NetworkCheck.Period.firstFinding]

theorem unlisted_files : ({} : NetworkCheck.Period.Listing).files = #[] :=
  Array.eq_empty_of_size_eq_zero (files_unlisted (by simp [NetworkCheck.Period.Listing.listed]))

/-- **`Verity.circuit_files_by_path_agree`'s honest lemma, at the network certifier's period**, the one component the
executable that reads the circuit files by path composes otherwise than the verifier of record: at a period of no
link-window and no audit (`emptyParams`) with an empty listing, so no circuit file is read by path, the period read by
path accepts, and the verifier of record's period, finding none among no blobs, accepts with the same terms. That a
period listing a circuit file accepts with it read by path as it does with it among the blobs is checked by the
composition suite's difftest `test_the_network_certifiers_circuit_files_are_read_by_path_or_among_the_blobs`, through
the composites' Lean executable. -/
theorem _root_.Verity.circuit_files_by_path_agree_honest :
    NetworkCheck.Period.readCheck 0 (NetworkCheck.Period.ofRead 0 #[]) (emptyParams, {}) emptyCommitment
        emptyEvidence = .accept (NetworkCheck.Period.terms #[] #[] #[] 0 1 0) { link := "", windows := 0 } ∧
      NetworkCheck.Period.blobProtocol.checkBlobs (emptyParams, {}) emptyCommitment emptyEvidence
        (Core.Blobs.ofArray #[]) =
          .accept (NetworkCheck.Period.terms #[] #[] #[] 0 1 0) { link := "", windows := 0 } := by
  have hl : ({} : NetworkCheck.Period.Listing).listed = false := by
    simp [NetworkCheck.Period.Listing.listed]
  have hr : NetworkCheck.Period.ofRead 0 #[] = .ok #[] := by
    simp [NetworkCheck.Period.ofRead]
    rfl
  have hb : NetworkCheck.Period.filesOf {} (Core.Blobs.ofArray #[]) = .ok #[] := by
    simp [NetworkCheck.Period.filesOf, NetworkCheck.Period.blobsOf, NetworkCheck.Period.ofFiles, unlisted_files]
    rfl
  have hp : NetworkCheck.Period.pinnedTableOf #[] = none := by
    simp [NetworkCheck.Period.pinnedTableOf]
  have hf : NetworkCheck.Period.filesCheck (.ok #[]) (emptyParams, {}) emptyCommitment emptyEvidence =
      .accept (NetworkCheck.Period.terms #[] #[] #[] 0 1 0) { link := "", windows := 0 } := by
    simp [NetworkCheck.Period.filesCheck, hp, hl, empty_check]
  refine ⟨?_, ?_⟩
  · simp [NetworkCheck.Period.readCheck, NetworkCheck.Period.Listing.scanFiles, hr, hf]
  · show NetworkCheck.Period.filesCheck (NetworkCheck.Period.filesOf {} (Core.Blobs.ofArray #[])) (emptyParams, {})
      emptyCommitment emptyEvidence = _
    rw [hb]
    exact hf

end Verity.PeriodFiles

/-- **At every composite with the network certifier's period, the verifier of record agrees with the executable that
reads the period's circuit files by path**, unless SHA-512 collides among the blobs. -/
theorem Verity.circuit_files_by_path_agree : Verity.CircuitFilesByPathAgree :=
  ⟨Verity.PeriodFiles.agree_onsite, Verity.PeriodFiles.agree_rn, Verity.PeriodFiles.agree_rnn⟩
