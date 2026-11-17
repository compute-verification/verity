module

public import Proofs.Flock.SpotFlock.HeldOf

/-! The templates' reference statements, built from the two-stage form's members and blobs (`refStmtsOf`). -/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims
open Lean (Json)

/-- Not shown: that the play's session statements (`SessionReading.stmts`) are `memberRaw` at the sessions' samples,
which `pinOk` needs to compare like with like, or that its bytes for the partition object and registered reads
(`RecordText.printT`) read back (`FlockDeploy.jsonOf`) as the member's: the form holds them as `Json`. -/
def memberRaw (m : Spot.OneStageZk.Member) (circuit pub : ByteArray) (tables : List (String × ByteArray))
    (draw registered : Option ByteArray) : FlockDeploy.Raw :=
  { statement := Spot.OneStageZk.STATEMENT, circuit, publicFile := pub, tables, zk := true, osCoins := false,
    count := 1, draw, partition := some (_root_.Flock.RecordText.printT m.own.partition).toUTF8,
    program := some m.own.program, work := none, workTable := none, stratified := none, closure := none, registered,
    publicInputs := none }

def memberOf (p : VStar.Recursive.SampledTwo.Params) (t : ℕ) : Option Spot.OneStageZk.Member :=
  (p.circuits[t]?).bind fun c => p.members.find? (·.own.circuit == c.2.1)

def refRawOf (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) (t : ℕ) : Option FlockDeploy.Raw := do
  let m ← memberOf p t
  let circuit ← (bs.get m.own.circuit "circuit").toOption
  let pub ← (bs.get m.own.pub "public file").toOption
  pure (memberRaw m circuit pub (Spot.OneStageZk.tablesOf bs) none
    (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8))

def refStmtsOf (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) : Option (List Msg) :=
  (List.range p.circuits.size).mapM fun t => (refRawOf p bs t).map FlockDeploy.Raw.encode

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

theorem refStmtsOf_getElem? {p : VStar.Recursive.SampledTwo.Params} {bs : Core.Blobs} {xs : List Msg} {t : ℕ}
    (h : refStmtsOf p bs = some xs) (ht : t < p.circuits.size) :
    ∃ r, refRawOf p bs t = some r ∧ xs[t]? = some r.encode := by
  unfold refStmtsOf at h
  obtain ⟨b, hb, hx⟩ := getElem?_of_mapM _ h (t := t) (by simpa using ht)
  simp only [List.getElem_range, Option.map_eq_some_iff] at hb
  obtain ⟨r, hr, rfl⟩ := hb
  exact ⟨r, hr, hx⟩

end SpotFlock.HeldOf
