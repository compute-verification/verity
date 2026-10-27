import Proofs.Flock.SoundAtBridge.Served
import Proofs.Flock.C3.Compose.RowsReg

/-!
# A session's registered reads built from `own`, its registrants' openings as data

`RowsSession.rdU` is each unit's registrant's opening, a free field: which commit strings it reads, through which port
and at which position, and the rows, salts and paths it opened there. `RowsSession.canon s op` is the session `s` with
`rdU` built from the verifier's own registered reads `own` (`canonReads`): unit `u` reads its row of every circuit port
that `own` names (`CanonRead`), through `own`'s first port of that name (`ownPort`), at that port's one position
(`oneRead`), and only the rows, salts and paths come from the registrant (`Openings`, whose `committed` opens each such
read at `own`'s root). Every other field of `s` is kept.

`servedRegistered_canon`: where every port of `own` has one read (the leaf verifier's `PouwWindow.Lanes.checkOneRead`),
every unit of `canon s op` has its served roots registered (`ServedRegistered`). A read's position at any draw is that
port's position at the unit's index, and with one read only index `0` has one.
-/

namespace Flock.SoundAtBridge.RowsSession

open FlockSoundness FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg

noncomputable section

variable (s : RowsSession)

/-- The layout the session's registrants' openings are of (`rdU`'s). -/
abbrev canonLay :=
  layShHJ (Reg := s.Reg) s.dj s.y₀ s.ht (Flock.Desc.parse_flat s.hD) s.hscope s.hU (pub := s.pub)

/-- The row encodings the session's registrants' openings are of (`rdU`'s). -/
abbrev canonRs := rsShHJS (Reg := s.Reg) s.dj s.y₀ s.ht (Flock.Desc.parse_flat s.hD) s.hscope s.hU s.hpub

/-- **Unit `u`'s registered reads, built from `own`**: `u`'s row of each circuit port `q` that `own` names. -/
def CanonRead (u : Fin s.n) {G : Type} (p : (Fin s.n × Fin s.c.ports.size) ⊕ G) : Prop :=
  ∃ q : Fin s.c.ports.size, p = .inl (u, q) ∧ ∃ pt ∈ s.own, pt.name = s.c.ports[q].name

/-- The port a read of row `(u, q)` opens: `own`'s first port of `q`'s name (`ownPort`). -/
def canonPort {G : Type} : (Fin s.n × Fin s.c.ports.size) ⊕ G → _root_.Flock.Registered.Port
  | .inl (_, q) => s.ownPort q
  | .inr _ => default

/-- A port's one position: its first read's, `0` where it has none. -/
def oneRead (pt : _root_.Flock.Registered.Port) : ℕ :=
  match pt.reads[0]? with
  | some (some r) => r
  | _ => 0

/-- The position a read of row `(u, q)` opens at: `ownPort q`'s one position (`oneRead`). -/
def canonAt {G : Type} : (Fin s.n × Fin s.c.ports.size) ⊕ G → ℕ
  | .inl (_, q) => oneRead (s.ownPort q)
  | .inr _ => 0

/-- **The registrants' openings at the reads built from `own`**: at each commit string, the row, salt and path the
registrant committed (`wv`, `ws`, `wpath`), each of unit `u`'s reads (`CanonRead`) opening `own`'s root at its port and
position under hm96's leaf over that row and salt (`committed`). -/
structure Openings where
  wv : s.canonLay.Pos → s.canonLay.Val
  ws : s.canonLay.Pos → List UInt8
  wpath : s.canonLay.Pos → ByteArray
  committed : ∀ (u : Fin s.n) (p : s.canonLay.Pos), s.CanonRead u p →
    _root_.Flock.Registered.opensLeaf (s.canonPort p)
      ((Binding.ExplicitLeaf.hm96X (s.canonRs.enc p) (s.canonRs.ext p) (s.canonRs.ext_spec p)).leaf (wv p)
        (ws p)).1 (s.canonAt p) (wpath p) = true

/-- **Unit `u`'s registered reads at the openings `op`**: `CanonRead`, through `canonPort` at `canonAt`, with `op`'s
rows, salts and paths. Each is a row (the readers' rows are their `.inl` commit strings). -/
def canonReads (op : s.Openings) (u : Fin s.n) : Registered.ReadsX s.canonLay s.canonRs where
  isRead := s.CanonRead u
  read_row := by
    rintro p ⟨q, rfl, -⟩
    exact rfl
  port := s.canonPort
  readAt := s.canonAt
  wv := op.wv
  ws := op.ws
  wpath := op.wpath
  committed := op.committed u

/-- **The session at its registrants' openings `op`**: `s` with each unit's registered reads built from `own`
(`canonReads`). -/
def canon (op : s.Openings) : RowsSession := { s with rdU := s.canonReads op }

/-- **At one read per port, the reads built from `own` register the served roots**: where every port of `own` has one
read, each unit of `canon s op` reads exactly its rows of the ports `own` names, through `ownPort`, at the position
`ownPort` gives it at every draw that proves it (only index `0` has a position). -/
theorem servedRegistered_canon (op : s.Openings) (h1 : ∀ pt ∈ s.own, pt.reads.size = 1) (u : Fin s.n) :
    (s.canon op).ServedRegistered u := by
  refine ⟨fun _ => Iff.rfl, fun q hn => ⟨rfl, ?_⟩⟩
  intro d D dp idx _ _ _ _ pos hpos
  have hs : (s.ownPort q).reads.size = 1 := h1 _ (FlockC3.Compose.portOf_spec hn).1
  have hpos' : (s.ownPort q).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some pos) := hpos
  have hlt := (Array.getElem?_eq_some_iff.mp hpos').1
  have h0 : idx.getD (D.proved.idxOf u.val) 0 = 0 := by omega
  rw [h0] at hpos'
  show oneRead (s.ownPort q) = pos
  unfold oneRead
  rw [hpos']

end

end Flock.SoundAtBridge.RowsSession
