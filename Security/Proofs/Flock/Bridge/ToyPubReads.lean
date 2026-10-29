import Proofs.Flock.Bridge.Transports

/-!
# A registrant that opens port 0's rows to zeros and every other port's to ones

`lieReads`: registered reads at a C-Flock session's shadow layout (`layShHJ`, `rsShHJS`) whose registrant reads every
unit's row of every port, each through a one-leaf registered port of that port's name (`liePort`) whose root is the
leaf of the row it opened, at position 0 and through the empty path, so each read opens what was registered. The rows
it opened are all zeros at port 0 and all ones at every other port, whatever the session computed (`lieWv`).

`ownReads_lie`: at one table, under a record whose draw proves unit 0 of a population of one, with the public file's
one instance unit 0, and a circuit that names each port once, every such read is one of the verifier's own registered
reads (`OwnReadsHJ`) when `own` holds every port of the registrant's (`lieOwn`).

`bit_lie`: at a wire held by row `(u, q)`, the value binding's bit of the registrant's opening there is `q ≠ 0`.
-/

namespace Flock.Bridge.ToyPub

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.ZkExec (zkTags)
open FlockSoundness.Discharge.Placed (keyProg)

/-- A registered port of circuit port `q`'s name with one leaf, `payload`'s at position 0 under the empty domain,
which is its root; unit 0 reads position 0. -/
def liePort (c : Flock.Circuit) (q : Fin c.ports.size) (payload : ByteArray) : Flock.Registered.Port where
  name := c.ports[q].name
  value := ""
  root := Flock.Registered.leafAs HmRow.ROW_SCHEMA ByteArray.empty 0 payload
  leaves := 1
  words := none
  wordBits := 16
  domain := ByteArray.empty
  readsRel := #[some 0]

theorem opensLeaf_liePort (c : Flock.Circuit) (q : Fin c.ports.size) (payload : ByteArray) :
    Flock.Registered.opensLeaf (liePort c q payload) payload 0 ByteArray.empty = true := by
  have hl : Flock.log2ceil 1 = 0 := rfl
  simp [Flock.Registered.opensLeaf, liePort, Flock.Registered.climb, Flock.Registered.climbL, Flock.Registered.sibs,
    hl, ByteArray.size_empty]

theorem reads_liePort (c : Flock.Circuit) (q : Fin c.ports.size) (payload : ByteArray) (i : ℕ) :
    (liePort c q payload).reads[(#[0] : Array ℕ).getD i 0]? = some (some 0) := by
  have h : (#[0] : Array ℕ).getD i 0 = 0 := by
    unfold Array.getD
    split
    · rename_i h
      have : i = 0 := by simp at h; omega
      subst this
      rfl
    · rfl
  rw [h]
  rfl

/-- The commit strings a registrant reads: every unit's row of every port. -/
def LieRead {n : ℕ} {c : Flock.Circuit} {G : Type} (p : (Fin n × Fin c.ports.size) ⊕ G) : Prop :=
  ∃ u q, p = .inl (u, q)

/-- The registrant's row at a commit string: every bit of port `q`'s row is `q ≠ 0`, at the widest row's width. -/
def lieWv {n : ℕ} {c : Flock.Circuit} {G : Type} : (Fin n × Fin c.ports.size) ⊕ G → ValH c
  | .inl (_, q) => (Fin.last _, fun _ => decide (q.val ≠ 0))
  | .inr _ => (Fin.last _, fun _ => false)

/-- Bit `i` of the registrant's row at row `(u, q)`, within the widest row's width, is `q ≠ 0`. -/
theorem lieWv_get {n : ℕ} {c : Flock.Circuit} {G : Type} {p : (Fin n × Fin c.ports.size) ⊕ G} {u : Fin n}
    {q : Fin c.ports.size} (hp : p = .inl (u, q)) {i : ℕ}
    (hi : i < Finset.univ.sup (Discharge.Layout.portRowW c) + 1) :
    Discharge.Layout.BitVal.get (lieWv p) i = decide (q.val ≠ 0) := by
  subst hp
  simp [lieWv, Discharge.Layout.BitVal.get, hi]

/-- The port each read of row `(u, q)` opens: `liePort` at `q`, over the leaf `f` gives there. -/
def portOf {n : ℕ} (c : Flock.Circuit) {G : Type} (f : (Fin n × Fin c.ports.size) ⊕ G → ByteArray) :
    (Fin n × Fin c.ports.size) ⊕ G → Flock.Registered.Port
  | .inl (u, q) => liePort c q (f (.inl (u, q)))
  | .inr _ => default

section Reads

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Discharge.Layout.scopeOkH c = true)
  (hU : Built c.unit) {pub : Flock.Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

/-- The registrant's port at `p`: `liePort` over hm96's leaf of its row there (`lieWv`) under the empty salt. -/
noncomputable def liePortOf (p : PosH c hU n) : Flock.Registered.Port :=
  portOf c (fun p : PosH c hU n =>
    ((Binding.ExplicitLeaf.hm96X ((rsShHJS (Reg := Reg) dj y₀ ht hc hscope hU hpub).enc p)
      ((rsShHJS (Reg := Reg) dj y₀ ht hc hscope hU hpub).ext p)
      ((rsShHJS (Reg := Reg) dj y₀ ht hc hscope hU hpub).ext_spec p)).leaf (lieWv p) []).1) p

/-- **The registrant's reads**: every unit's row of every port (`LieRead`), port 0's opened to zeros and every
other's to ones (`lieWv`), each through `liePort` over its own leaf, at position 0 and the empty path. -/
noncomputable def lieReads :
    FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub) where
  isRead p := LieRead (c := c) (p : PosH c hU n)
  read_row := by
    rintro p ⟨u, q, rfl⟩
    exact rfl
  port := liePortOf (Reg := Reg) dj y₀ ht hc hscope hU hpub
  readAt _ := 0
  wv p := lieWv (c := c) (p : PosH c hU n)
  ws _ := []
  wpath _ := ByteArray.empty
  committed := by
    rintro p ⟨u, q, rfl⟩
    exact opensLeaf_liePort c q _

/-- The registrant's ports: each unit's port of each circuit port. -/
noncomputable def lieOwn : Array Flock.Registered.Port :=
  ((List.finRange n).flatMap fun u => (List.finRange c.ports.size).map fun q =>
    liePortOf (Reg := Reg) dj y₀ ht hc hscope hU hpub (.inl (u, q))).toArray

/-- **Every read of the registrant is one of the verifier's own**, at one table, under a record whose draw proves
unit 0 of a population of one, the public file's instance unit 0, and ports named once. -/
theorem ownReads_lie (h1 : I.count ≤ 1) (hn : n ≤ 1) {t : Exec.Transcript} {d : Lean.Json} {D : Draw.UnitDraw}
    {dp : Flock.Public} (hd : Exec.recordDraw t.record = some d) (hD : Draw.UnitDraw.ofJson d = .ok D)
    (hDp : D.proved = #[0]) (hdp : HmOut.drawn c pub d = .ok dp) (hidx : Flock.Registered.unitIndices dp = .ok #[0])
    (hnames : ∀ q : Fin c.ports.size,
      (List.range c.ports.size).find? (fun j => c.ports[j]!.name == c.ports[q].name) = some q.val) :
    OwnReadsHJ I (lieOwn (Reg := Reg) dj y₀ ht hc hscope hU hpub) c pub t
      (lieReads (Reg := Reg) dj y₀ ht hc hscope hU hpub).isRead
      (lieReads (Reg := Reg) dj y₀ ht hc hscope hU hpub).port
      (lieReads (Reg := Reg) dj y₀ ht hc hscope hU hpub).readAt := by
  rintro p ⟨u, q, rfl⟩
  have hu : u.val = 0 := by have := u.isLt; omega
  refine ⟨u, q, d, D, dp, #[0], rfl, ?_, hnames q, hd, ?_, ?_, hidx, ?_, reads_liePort c q _ _⟩
  · simp only [lieOwn, List.mem_toArray, List.mem_flatMap, List.mem_map, List.mem_finRange, true_and]
    exact ⟨u, q, rfl⟩
  · rw [drawJ_one h1]
    exact hD
  · rw [drawJ_one h1]
    exact hdp
  · rw [hDp, hu]
    simp

end Reads

/-- **The value binding's bit of the registrant's opening at a row**: at a wire `g` that row `(u, q)` holds, bit `k`
of the registrant's row there, which is `q ≠ 0`. -/
theorem bit_lie {n mPts : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json} {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} {y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀} {ht : I.tags.typed = false} {c : Flock.Circuit}
    {hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c} {hscope : Discharge.Layout.scopeOkH c = true}
    {hU : Built c.unit} {pub : Flock.Public} {hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub}
    {hpt : pub.tables = none}
    {g : Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N} {u : Fin n} {q : Fin c.ports.size}
    {k : Fin (Discharge.Layout.portRowW c q)}
    (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = some (u, ⟨q, k⟩)) (y : List UInt8) :
    (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).bit
        ((vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).val
          ((vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).pos g,
            lieWv ((vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).pos g), y)) g =
      decide (q.val ≠ 0) := by
  have hle : k.val < Finset.univ.sup (Discharge.Layout.portRowW c) + 1 :=
    lt_of_lt_of_le k.isLt ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))
  exact (vbOf_bit_row h _).trans (lieWv_get (vbOf_pos_row (dj := dj) (y₀ := y₀) (ht := ht) (hc := hc)
    (hscope := hscope) (hpub := hpub) (hpt := hpt) (Reg := Reg) h) hle)

end Flock.Bridge.ToyPub
