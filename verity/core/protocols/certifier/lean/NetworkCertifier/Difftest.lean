import NetworkCertifier.Ingress
import NetworkCertifier.Schedule
import Lean.Data.Json

/-!
# The difftest: the certifier program against the Python audit, on random instances

`scripts/difftest_vectors.py` generates instances and runs the Python audit on them (`schedule.expected_grids`,
`Audit.check_link`, `Audit.check_ingress_link`). `NetworkCertifier/DifftestMain.lean` reads both files and fails unless `evaluate`,
which computes the same results from the program's definitions (`NetworkCertifier`), gives every one of them. The Lean
audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof. `Proofs.NetworkCertifier.Code`
(`security_proofs`) proves what it evaluates (`expectedRow`, `Accepted` of `constantRate`, `StatusComplete`, membership
in `Σ_sync`) equal to the reference definitions in `Security/Definitions/NetworkCertifier/`, which the guarantees read.

An instance is `L` egress links and `K ≥ L` boundary links (the egress links first), windows of `T` buckets, `W` of
them with advice, the allocation `S`, the sync set `Σ_sync`, and per egress link its frames in canonical order and the
sync it declares for each window. Its observation sequence is the records, one step per bucket: step `t` holds every
egress link's row `t mod T` of its record for window `t / T`, and every boundary link's status for that window. The
records stop at a window boundary, so the sequence has `T` steps per window recorded.

`evaluate` gives, from the definitions themselves:
* `rows`: `expectedRow` of each egress link's frames under the syncs that link declares, for every bucket of the
  horizon (Python's `expected_grids`);
* `verdicts`, per egress link-window, in the order `check_link` checks: `no-verdict` when the window isn't recorded,
  `status:<reason>` when one of the link's statuses in it is not complete, `advice` when the declared sync is outside
  `Σ_sync` or other than link 0's (the spatial unit's, which `Audit.syncs` keeps), `mismatch` when a recorded row is
  not the link's `expectedRow`; and `diff`, the first such bucket of a recorded window;
* `status`, per boundary link-window: `check_ingress_link`'s verdict from the statuses alone;
* for every prefix of `m ≤ W` whole windows, `accepted`: `Accepted` of the constant-rate Program on the first `m·T`
  steps, by `decide`, with link 0's syncs as the declared advice (one clock per spatial unit), and no link declaring a
  sync outside `Σ_sync` or other than link 0's in those windows: such a declaration is no value of the advice type
  `Fin W → ss.Elem`. The advice past the prefix is link 0's where it is in the set and the set's first value elsewhere,
  which the Program's rows before step `m·T` don't read; Python computes each link's grids from its own declaration;
* for every prefix, `complete`: `StatusComplete` on the first `m·T` steps, by `decide`.
-/

namespace NetworkCertifier.Difftest

open Lean

/-- Rows and their records are functions on `Fin`; Lean core decides their equality slot by slot. -/
instance decEqFinFun {n : Nat} {α : Type} [DecidableEq α] : DecidableEq (Fin n → α) := fun f g =>
  decidable_of_iff (∀ i, f i = g i) ⟨funext, fun h _ => h ▸ rfl⟩

/-- `Accepted`'s `∃ h : t < o.length, …`, decided by its premise. -/
instance decidableExistsProp {p : Prop} {q : p → Prop} [Decidable p] [∀ h, Decidable (q h)] : Decidable (∃ h, q h) :=
  if hp : p then decidable_of_iff (q hp) ⟨fun h => ⟨hp, h⟩, fun ⟨_, h⟩ => h⟩ else isFalse fun ⟨h, _⟩ => hp h

instance decidableAccepted {L K S : Nat} {In Adv : Type} (P : EgressPrediction L S In Adv)
    (op : Nat → List (Obs L K S) → In) (k : Nat) (o : List (Obs L K S)) (a : Adv) : Decidable (Accepted P op k o a) := by
  unfold Accepted; infer_instance

instance decidableStatusComplete {L K S : Nat} (k : Nat) (o : List (Obs L K S)) : Decidable (StatusComplete k o) := by
  unfold StatusComplete; infer_instance

/-- A payload's bits from its bytes in lowercase hex, eight per byte, most significant first. -/
def bitsOfHex (s : String) : Except String (List Bool) :=
  s.toList.foldrM (init := []) fun c acc => do
    let n := c.toNat
    let d ← if '0'.toNat ≤ n ∧ n ≤ '9'.toNat then pure (n - '0'.toNat)
      else if 'a'.toNat ≤ n ∧ n ≤ 'f'.toNat then pure (n - 'a'.toNat + 10)
      else throw s!"not lowercase hex: {s}"
    pure ([d / 8 % 2 == 1, d / 4 % 2 == 1, d / 2 % 2 == 1, d % 2 == 1] ++ acc)

/-- The inverse of `bitsOfHex`, four bits to a digit. -/
def hexOfBits : List Bool → String
  | b0 :: b1 :: b2 :: b3 :: bs =>
    String.ofList (Nat.toDigits 16 (8 * b0.toNat + 4 * b1.toNat + 2 * b2.toNat + b3.toNat)) ++ hexOfBits bs
  | [] => ""
  | _ => "?"

/-- Python's `Reason` value. -/
def reasonText : Reason → String
  | .monitorUnavailable => "monitor-unavailable"
  | .overflow => "overflow"
  | .malformed => "malformed"
  | .dropped => "dropped"
  | .deliveryRefused => "delivery-refused"
  | .coverageNeverEstablished => "coverage-never-established"

/-- A status from Python's `Reason` value, `null` for complete. -/
def statusOfJson : Json → Except String Status
  | .null => pure .complete
  | .str s => match [Reason.monitorUnavailable, .overflow, .malformed, .dropped, .deliveryRefused,
      .coverageNeverEstablished].find? (reasonText · == s) with
    | some r => pure (.notEstablished r)
    | none => throw s!"not a reason: {s}"
  | j => throw s!"not a status: {j.compress}"

/-- A positive rational `[num, den]`. -/
def ratOfJson (j : Json) : Except String Rat := do
  let #[n, d] ← fromJson? (α := Array Int) j | throw s!"not [num, den]: {j.compress}"
  if d ≤ 0 then throw s!"not [num, den] with den > 0: {j.compress}"
  pure (mkRat n d.toNat)

/-- A clock sync `[delta, num, den]`. -/
def syncOfJson (j : Json) : Except String (Nat × Rat) := do
  let #[d, n, m] ← fromJson? (α := Array Json) j | throw s!"not [delta, num, den]: {j.compress}"
  pure (← fromJson? d, ← ratOfJson (.arr #[n, m]))

/-- The clock sync of a declared value, whose ρ Python's `ClockSync` requires positive (`evaluate` checks it). -/
def clockOf (p : Nat × Rat) : ClockSync :=
  if h : 0 < p.2 then ⟨p.1, p.2, h⟩ else ⟨p.1, 1, by decide⟩

/-- A frame `[payload, anchor, index]`. -/
def frameOfJson (j : Json) : Except String LogicalFrame := do
  let #[p, a, i] ← fromJson? (α := Array Json) j | throw s!"not [payload, anchor, index]: {j.compress}"
  pure ⟨← bitsOfHex (← fromJson? p), ← fromJson? a, ← fromJson? i⟩

def slotOfJson : Json → Except String (Option (List Bool))
  | .null => pure none
  | j => do pure (some (← bitsOfHex (← fromJson? j)))

def slotJson : Option (List Bool) → Json
  | none => .null
  | some f => .str (hexOfBits f)

/-- An instance as read, before its sizes become types. -/
structure Raw where
  T : Nat
  W : Nat
  S : Nat
  K : Nat
  deltas : List Nat
  rhos : List Rat
  frames : Array (List LogicalFrame)
  syncs : Array (Array (Nat × Rat))
  rows : Array (Array (Array (Option (List Bool))))
  status : Array (Array Status)

def arrOf (j : Json) : Except String (Array Json) := fromJson? j

def rawOfJson (j : Json) : Except String Raw := do
  let obs ← arrOf (← j.getObjVal? "obs")
  pure {
    T := ← j.getObjValAs? Nat "T", W := ← j.getObjValAs? Nat "W", S := ← j.getObjValAs? Nat "S", K := ← j.getObjValAs? Nat "K"
    deltas := ← j.getObjValAs? (List Nat) "deltas"
    rhos := (← (← arrOf (← j.getObjVal? "rhos")).mapM ratOfJson).toList
    frames := ← (← arrOf (← j.getObjVal? "frames")).mapM fun l => do (← arrOf l).toList.mapM frameOfJson
    syncs := ← (← arrOf (← j.getObjVal? "syncs")).mapM fun l => do (← arrOf l).mapM syncOfJson
    rows := ← obs.mapM fun x => do (← arrOf (← x.getObjVal? "rows")).mapM fun r => do (← arrOf r).mapM slotOfJson
    status := ← obs.mapM fun x => do (← arrOf (← x.getObjVal? "status")).mapM statusOfJson }

/-- The first element of `l` in `s`, with its membership. -/
def pick {α : Type} [DecidableEq α] (l : List α) (s : List α) : Option {x // x ∈ s} :=
  l.findSome? fun x => if h : x ∈ s then some ⟨x, h⟩ else none

def verdictJson : Option String → Json
  | none => .null
  | some v => .str v

/-- What the program's definitions give on an instance, in the shape of the Python audit's results. -/
def evaluate (r : Raw) : Except String Json := do
  let L := r.frames.size
  let T := r.T
  let W := r.W
  let S := r.S
  let K := r.K
  if hL : ¬(0 < L ∧ L ≤ K) then throw "an instance has 1 … K egress links" else
  unless 0 < T ∧ r.syncs.size = L ∧ r.syncs.all (·.size = W) ∧ r.syncs.all (·.all (0 < ·.2)) do
    throw "an instance has T > 0 and W positive syncs per egress link"
  unless r.rows.size = r.status.size ∧ r.rows.size % T = 0 ∧ r.rows.size ≤ W * T do
    throw "an observation sequence of whole windows"
  unless r.rows.all (fun x => x.size = L ∧ x.all (·.size = S)) ∧ r.status.all (·.size = K) do
    throw "a step has L rows of S slots and K statuses"
  let ss : SyncSet ← if h : r.deltas ≠ [] ∧ r.rhos ≠ [] ∧ (∀ ρ ∈ r.rhos, 0 < ρ) ∧ r.deltas.Nodup ∧ r.rhos.Nodup then
      pure ⟨r.deltas, r.rhos, h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩
    else throw "not a sync set"
  let some d0 := pick r.deltas ss.deltas | throw "no delta"
  let some ρ0 := pick r.rhos ss.rhos | throw "no rho"
  let decl (ℓ : Fin L) (w : Fin W) : Nat × Rat := (r.syncs.getD ℓ #[]).getD w (0, 1)
  let syncs (ℓ : Fin L) : Fin W → ClockSync := fun w => clockOf (decl ℓ w)
  let unit (w : Fin W) : Nat × Rat := decl ⟨0, (Decidable.of_not_not hL).1⟩ w
  let elem (p : Nat × Rat) : ss.Elem :=
    if h : p.1 ∈ ss.deltas ∧ p.2 ∈ ss.rhos then (⟨p.1, h.1⟩, ⟨p.2, h.2⟩) else (d0, ρ0)
  let frames (ℓ : Fin L) : List LogicalFrame := r.frames.getD ℓ []
  let o : List (Obs L K S) := (List.range r.rows.size).map fun t =>
    { rows := fun ℓ s => ((r.rows.getD t #[]).getD ℓ #[]).getD s none, jitter := 0
      status := fun κ => (r.status.getD t #[]).getD κ .complete }
  let P : EgressPrediction L S Unit (Unit × (Fin W → ss.Elem)) := constantRate T W ss fun _ _ ℓ => frames ℓ
  let op : Nat → List (Obs L K S) → Unit := fun _ _ => ()
  let σ : Fin W → ss.Elem := fun w => elem (unit w)
  let recorded (w : Nat) : Bool := (w + 1) * T ≤ o.length
  let window (w : Nat) : List Nat := (List.range T).map (w * T + ·)
  -- `check_ingress_link`'s verdict for boundary link `κ` in window `w`, from its statuses there
  let statusVerdict (κ : Fin K) (w : Nat) : Option String :=
    if recorded w then
      (window w).findSome? fun t =>
        if h : t < o.length then
          match o[t].status κ with
          | .complete => none
          | .notEstablished why => some s!"status:{reasonText why}"
        else none
    else some "no-verdict"
  let rowAgrees (ℓ : Fin L) (t : Nat) : Bool :=
    if h : t < o.length then decide (o[t].rows ℓ = expectedRow S T W (syncs ℓ) (frames ℓ) t) else false
  let advice (ℓ : Fin L) (w : Fin W) : Bool :=
    decide ((decl ℓ w).1 ∈ ss.deltas ∧ (decl ℓ w).2 ∈ ss.rhos) && decl ℓ w == unit w
  let diff (ℓ : Fin L) (w : Nat) : Option Nat := if recorded w then (window w).find? (! rowAgrees ℓ ·) else none
  let verdict (ℓ : Fin L) (w : Fin W) : Option String :=
    match statusVerdict ⟨ℓ, by omega⟩ w with
    | some v => some v
    | none => if !advice ℓ w then some "advice" else (diff ℓ w).map fun _ => "mismatch"
  let accepted (m : Nat) : Bool :=
    (List.finRange W).all (fun w => decide (m ≤ w.1) || (List.finRange L).all (advice · w)) &&
      decide (Accepted P op (m * T) o ((), σ))
  let complete (m : Nat) : Bool := decide (StatusComplete (m * T) o)
  let links := (List.finRange L).toArray
  let horizon := (List.finRange W).toArray
  pure <| Json.mkObj [
    ("rows", .arr <| links.map fun ℓ => .arr <| (List.range (W * T)).toArray.map fun t =>
      .arr ((List.ofFn (expectedRow S T W (syncs ℓ) (frames ℓ) t)).map slotJson).toArray),
    ("verdicts", .arr <| links.map fun ℓ => .arr <| horizon.map (verdictJson ∘ verdict ℓ)),
    ("diff", .arr <| links.map fun ℓ => .arr <| horizon.map fun w => toJson (diff ℓ w.1)),
    ("status", .arr <| (List.finRange K).toArray.map fun κ => .arr <| horizon.map fun w =>
      verdictJson (statusVerdict κ w.1)),
    ("accepted", toJson ((List.range (W + 1)).map accepted)),
    ("complete", toJson ((List.range (W + 1)).map complete))]

/-- The results' fields, in the order they are compared. -/
def fields : List String := ["rows", "verdicts", "diff", "status", "accepted", "complete"]

/-- How the Lean side's results for instance line `i` differ from Python's, one line per field that differs. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst >>= rawOfJson >>= evaluate, Json.parse python with
  | .error e, _ => [s!"instance {i}: the Lean side rejects it: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok lean, .ok py => fields.filterMap fun k =>
    let a := lean.getObjValD k
    let b := py.getObjValD k
    if a == b then none else some s!"instance {i}: {k}: Lean {a.compress}, Python {b.compress}"

/-- `lake env lean --run NetworkCertifier/DifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES` evaluated side
by side, each compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: NetworkCertifier/DifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NetworkCertifier.Difftest
