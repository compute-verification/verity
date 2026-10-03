import FlockSoundness.Discharge.ZkPubIn.Bind
import FlockSoundness.Discharge.ZkReg.CheckedH

/-!
# The registered reads of a `verify --zk --registered R --public-inputs F` session

ZkReg's `OwnReadsH`, `regPathsH` and `checked_of_acceptsH` with the verifier's public inputs. With `own` given,
setupHidden's drawn statement is `HmIn.drawn c pi pub d` (the population's `in` words filtered to the draw), not
`HmOut.drawn c pub d`, so the registered reads are read there.

* `acceptsZKP_of_RP`: `AcceptsZKRP I reg own` is `AcceptsZKP I own` with the registered check.
* `OwnReadsHP`, `regPathsHP`: `OwnReadsH`, `regPathsH` at `HmIn.drawn c pi pub d`.
* `checked_of_acceptsHP`: the shadow layout's registered reads that are the verifier's own are `Checked` at every
  accepted outcome.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

theorem acceptsZKP_of_RP {I : Inputs} {reg : Array Registered.Port} {own : ByteArray} {t : Exec.Transcript}
    (h : AcceptsZKRP I reg own t) : AcceptsZKP I own t := by
  obtain ⟨spec, sts, hs, -, hshape, hv⟩ := h
  exact ⟨spec, sts, hs, hshape, hv⟩

variable {n : ℕ} {Gate : Type}

/-- **The registered reads are the verifier's own at the record's draw, with public inputs**: `OwnReadsH` at the drawn
statement `HmIn.drawn c pi pub d`. -/
def OwnReadsHP (reg : Array Registered.Port) (c : Circuit) (pi : HmIn.Ports) (pub : Public) (t : Exec.Transcript)
    (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop) (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port)
    (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) : Prop :=
  ∀ p, isRead p → ∃ u q d D dp idx, p = .inl (u, q) ∧ port p ∈ reg ∧
    (List.range c.ports.size).find? (fun j => c.ports[j]!.name == (port p).name) = some q.val ∧
    recordDraw t.record = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ HmIn.drawn c pi pub d = .ok dp ∧
    Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved ∧
    (port p).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some (readAt p))

/-- **The paths of the registered reads, with public inputs**: `regPathsH` at the drawn statement
`HmIn.drawn c pi pub d`. -/
def regPathsHP (c : Circuit) (pi : HmIn.Ports) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) : (Fin n × Fin c.ports.size) ⊕ Gate → ByteArray
  | .inl (u, q) => Option.getD (do
      let d ← recordDraw t.record
      let D ← (Draw.UnitDraw.ofJson d).toOption
      let dp ← (HmIn.drawn c pi pub d).toOption
      let tb ← dp.tables
      let (_, paths) ← (Registered.entry dp (port (.inl (u, q)))).toOption
      return paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty)
      ByteArray.empty
  | .inr _ => ByteArray.empty

variable {mPts : ℕ} {Reg : Type}

/-- **The registered reads are checked at every accepted outcome with public inputs** (`checked_of_acceptsH`). At an
outcome `verify --zk --registered R --public-inputs F` accepts (`reg` R's ports, `own` F's bytes), each registered read
of the shadow layout that is the verifier's own at the outcome's draw (`OwnReadsHP`) has its registered leaf open the
root `reg` holds at the position `reg` gives, through `regPathsHP`. -/
theorem checked_of_acceptsHP (I : Inputs) (ht : I.tags.typed = false) (reg : Array Registered.Port) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
    (hTags : TagsOkH I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKRP I reg own o)
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    (hown : OwnReadsHP reg c pi pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsHP c pi pub o.2.2 rd.port) := by
  sorry

end FlockSoundness.Discharge.ZkPubIn
