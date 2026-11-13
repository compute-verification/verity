module

public import Proofs.Flock.SpotFlock.HeldOf

/-!
# The templates' reference statements, from the form's members and blobs

`HeldOf.heldOf` takes each template's reference session statement (`Unread.xs`). This module builds them from what the
two-stage form's check holds: template `t`'s member (`memberOf`: the first of the interior audit's members whose
circuit is the one the params name for `t`), its circuit and public files on the blobs, and the pinned lookup tables the
blobs hold (`OneStageZk.tablesOf`), as the per-member form's `flock-verify verify --zk` reads them
(`OneStageZk.inputsOf`: the statement `verity/flock-circuit` under `--zk`, one table, the member's partition object and
program), at no sample and with the member's registered reads (`memberRaw`). A session statement of the member is this
statement at its record's sample and its own public file and registered reads, which `pinOk` blanks.

## What this does not show

* That the session statements of the play (`SessionReading.stmts`) are `memberRaw` at the sessions' samples: whoever
  builds them should build them with `memberRaw`, so that `pinOk` compares like with like.
* That `memberRaw`'s bytes for the partition object and the registered reads (`RecordText.printT`) read back
  (`FlockDeploy.jsonOf`) as the member's: the form holds them as `Json`.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims
open Lean (Json)

/-- **A member's session statement**, as `flock-verify verify --zk` reads it in the per-member form, at the circuit
and public files `circuit` and `pub`, the lookup tables `tables`, the sample `draw` and the registered reads
`registered`. -/
def memberRaw (m : Spot.OneStageZk.Member) (circuit pub : ByteArray) (tables : List (String × ByteArray))
    (draw registered : Option ByteArray) : FlockDeploy.Raw :=
  { statement := Spot.OneStageZk.STATEMENT, circuit, publicFile := pub, tables, zk := true, osCoins := false,
    count := 1, draw, partition := some (_root_.Flock.RecordText.printT m.own.partition).toUTF8,
    program := some m.own.program, work := none, workTable := none, stratified := none, closure := none, registered,
    publicInputs := none }

/-- **Template `t`'s member**: the first of the interior audit's members whose circuit is the one the params name for
template `t`. -/
def memberOf (p : VStar.Recursive.SampledTwo.Params) (t : ℕ) : Option Spot.OneStageZk.Member :=
  (p.circuits[t]?).bind fun c => p.members.find? (·.own.circuit == c.2.1)

/-- **Template `t`'s reference statement**: its member's statement on the blobs at no sample, with the member's public
file and registered reads; none where the template has no member or a file is on no blob. -/
def refRawOf (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) (t : ℕ) : Option FlockDeploy.Raw := do
  let m ← memberOf p t
  let circuit ← (bs.get m.own.circuit "circuit").toOption
  let pub ← (bs.get m.own.pub "public file").toOption
  pure (memberRaw m circuit pub (Spot.OneStageZk.tablesOf bs) none
    (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8))

/-- **The templates' reference statements**, in the params' template order, each `refRawOf`'s bytes. -/
def refStmtsOf (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) : Option (List Msg) :=
  (List.range p.circuits.size).mapM fun t => (refRawOf p bs t).map FlockDeploy.Raw.encode

/-- **What the run does not read, from the form's members and blobs**: the reference statements `refStmtsOf`, with the
statement's descriptor and query, the frame's code and the second stage's rounds past round 1 as given. -/
def unreadOf (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) (desc : Msg) (query : Claims.Sampling.Query)
    (c₀ : SamplingDeploy.Code) (more : ℕ) : Option Unread :=
  (refStmtsOf p bs).map fun xs => ⟨desc, query, xs, c₀, more⟩

theorem length_of_mapM {α β : Type} {f : α → Option β} (l : List α) :
    ∀ {l' : List β}, l.mapM f = some l' → l'.length = l.length := by
  induction l with
  | nil =>
    intro l' h
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    rfl
  | cons a l ih =>
    intro l' h
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨b, -, bs, hbs, rfl⟩ := h
    simp [ih hbs]

theorem refStmtsOf_length {p : VStar.Recursive.SampledTwo.Params} {bs : Core.Blobs} {xs : List Msg}
    (h : refStmtsOf p bs = some xs) : xs.length = p.circuits.size := by
  unfold refStmtsOf at h
  rw [length_of_mapM _ h, List.length_range]

theorem getElem?_of_mapM {α β : Type} {f : α → Option β} (l : List α) :
    ∀ {l' : List β}, l.mapM f = some l' → ∀ {t : ℕ} (ht : t < l.length), ∃ b, f l[t] = some b ∧ l'[t]? = some b := by
  induction l with
  | nil =>
    intro l' _ t ht
    simp at ht
  | cons a l ih =>
    intro l' h t ht
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨b, hb, bs, hbs, rfl⟩ := h
    cases t with
    | zero => exact ⟨b, hb, rfl⟩
    | succ t =>
      obtain ⟨c, hc, hc'⟩ := ih hbs (Nat.lt_of_succ_lt_succ ht)
      exact ⟨c, by simpa using hc, by simpa using hc'⟩

/-- **Template `t`'s entry of the reference statements is its `refRawOf`'s bytes.** -/
theorem refStmtsOf_getElem? {p : VStar.Recursive.SampledTwo.Params} {bs : Core.Blobs} {xs : List Msg} {t : ℕ}
    (h : refStmtsOf p bs = some xs) (ht : t < p.circuits.size) :
    ∃ r, refRawOf p bs t = some r ∧ xs[t]? = some r.encode := by
  unfold refStmtsOf at h
  obtain ⟨b, hb, hx⟩ := getElem?_of_mapM _ h (t := t) (by simpa using ht)
  simp only [List.getElem_range, Option.map_eq_some_iff] at hb
  obtain ⟨r, hr, rfl⟩ := hb
  exact ⟨r, hr, hx⟩

end SpotFlock.HeldOf
