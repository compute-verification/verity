import Proofs.Flock.Bridge.V5.Statement
import Proofs.Flock.Bridge.TransportsRefs

/-!
# Canonical registered openings open the deployed commitment at public inputs (`Bridge.TransportsLiveV5`, proved)

`TransportsRefs.lean` at `keyProgP` and `ZkHidden.rsZHJP`:

* `Bridge.ref_lt_cntP`: `ref_lt_cnt` at `HmIn.loadPublic`, which is `HmRow.loadPublic` on the circuit whose output ports
  are the public input words (`Hidden.loadPublic_inRows`), with the same row ports;
* `Bridge.vbP` and its `pos`, `bit`, `cs` and `com` at a row (`vbS`'s at `keyReadersHP`, whose rows' `b ‖ c` is the
  population's, `fun _ => pub`);
* `Bridge.rowsOfLP`, `Bridge.transports_shapedP`: `transports_shapedS` at these;
* `Bridge.transportsLiveV5 : Bridge.TransportsLiveV5`.
-/

namespace Flock.Bridge

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open FlockSoundness.Discharge.Layout (BitVal RowShaped)

/-- **The verifier's leaf map at public inputs holds every proof block's row** (`ref_lt_cnt` at `HmIn.loadPublic`, through
`Hidden.loadPublic_inRows`: `inRows` changes only the output ports). -/
theorem ref_lt_cntP {tags : Flock.Tags} {c : Flock.Circuit} {pi : Flock.HmIn.Ports} {file : ByteArray}
    {pub : Flock.Public} (h : Flock.HmIn.loadPublic tags c pi file = .ok pub) {nl u q : ℕ} (hu : u < nl)
    (hq : q < c.ports.size) :
    (LeafMap.ofPublic c pub nl).ref q u < (LeafMap.ofPublic c pub nl).cnt q := by
  have h' := FlockSoundness.Discharge.Hidden.loadPublic_inRows h
  cases ht : pub.tables with
  | none => rw [LeafMap.ofPublic_none nl ht]; exact hu
  | some t =>
    rw [LeafMap.ofPublic_some nl ht]
    obtain ⟨-, hlt, -, -⟩ := (FlockSoundness.Discharge.Hidden.loadPublic_shared h').2 t ht
    obtain ⟨hcs, hpos⟩ := loadPublic_counts h' t ht
    have hlt' : ∀ k < t.refs.size, t.refs[k]! < t.counts[k % c.ports.size]! := hlt
    have hcs' : t.counts.size = c.ports.size := hcs
    show t.refs.getD (u * c.ports.size + q) 0 < t.counts.getD q 0
    rw [getD_zero_eq_getElem!, getD_zero_eq_getElem!]
    by_cases hk : u * c.ports.size + q < t.refs.size
    · have hmod : (u * c.ports.size + q) % c.ports.size = q := by
        rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hq]
      have := hlt' _ hk
      rwa [hmod] at this
    · rw [getElem!_neg t.refs _ hk]
      have hqc : q < t.counts.size := by rw [hcs']; exact hq
      rw [getElem!_pos t.counts q hqc]
      exact hpos _ (Array.getElem_mem hqc)

section Game

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge
  FlockSoundness.Discharge.Composed

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} {own : Option ByteArray} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀) (ht : I.tags.typed = false)
  {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Discharge.Layout.scopeOkH c = true) (hU : Built c.unit) {pi : Flock.HmIn.Ports}
  (hpi : Flock.HmIn.parse c = .ok pi) {pub : Flock.Public}
  (hpub : Flock.HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)

/-- **The game's value binding at public inputs** (`TransportsLiveV5`'s `vb`). -/
noncomputable abbrev vbP := (rsZHJP (Reg := Reg) dj y₀ ht hc hscope hU hpi hpub).bindingZ
  (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub)

/-- The session's commit strings at public inputs: proof block `u`'s row of port `q`, or a single gate. -/
abbrev PosHP (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (pi : Flock.HmIn.Ports) : Type :=
  (Fin n × Fin c.ports.size) ⊕ Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N

variable {dj y₀ ht hc hscope hU hpi hpub}
variable {g : Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N} {u : Fin n}
  {q : Fin c.ports.size} {k : Fin (Discharge.Layout.portRowW c q)}

theorem vbP_pos_row (h : HiddenExec.rowOfHP c hU n pi (Discharge.Layout.outsOfP c hU n pi) g = some (u, ⟨q, k⟩)) :
    (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).pos g = (Sum.inl (u, q) : PosHP c hU n pi) := by
  dsimp only [vbP, HmRowsZX.bindingZ, LayoutZ.bindingZ, rsZHJP, ZkHidden.layZHJP, LayoutZ.ofShadow,
    ZkHidden.layShHJP, HiddenExec.keyLayoutHP, Discharge.Layout.Readers.layout, Discharge.Layout.ofBits,
    HiddenExec.keyReadersHP]
  exact Discharge.Layout.ofRows_pos_some _ h

theorem vbP_bit_row (h : HiddenExec.rowOfHP c hU n pi (Discharge.Layout.outsOfP c hU n pi) g = some (u, ⟨q, k⟩))
    (v : ValH c) : (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).bit v g = v.get k.val := by
  have hle : Discharge.Layout.portRowW c q ≤ Finset.univ.sup (Discharge.Layout.portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  dsimp only [vbP, HmRowsZX.bindingZ, LayoutZ.bindingZ, rsZHJP, ZkHidden.layZHJP, LayoutZ.ofShadow,
    ZkHidden.layShHJP, HiddenExec.keyLayoutHP, Discharge.Layout.Readers.layout, Discharge.Layout.ofBits,
    HiddenExec.keyReadersHP, Discharge.Layout.Readers.ofRows]
  split
  · rename_i heq
    rw [h] at heq
    simp only [Option.some.injEq, Prod.mk.injEq, Sigma.mk.inj_iff] at heq
    obtain ⟨rfl, rfl, heq⟩ := heq
    rw [heq_iff_eq] at heq
    subst heq
    simp only [FlockSoundness.Discharge.Layout.BitVal.get, lt_of_lt_of_le k.isLt hle, ↓reduceDIte]
    try rfl
  · rename_i heq
    rw [h] at heq
    exact absurd heq (by simp)

theorem vbP_cs_inl (R : Reg) (u : Fin n) (q : Fin c.ports.size) :
    (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).cs R (Sum.inl (u, q) : PosHP c hU n pi) =
      ((Sum.inl (u, q) : PosHP c hU n pi),
        H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList)) :=
  rfl

theorem vbP_com_inl (u : Fin n) (q : Fin c.ports.size) (v : ValH c) (y : List UInt8) :
    (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).com ((Sum.inl (u, q) : PosHP c hU n pi), v, y) =
      ((Sum.inl (u, q) : PosHP c hU n pi), H512 (hm96Input (HiddenExec.encPortH c.ports[q.val]! v) y)) :=
  rfl

theorem vbP_com_fst (p : PosHP c hU n pi) (v : ValH c) (y : List UInt8) :
    ((vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).com (p, v, y)).1 = p :=
  rfl

theorem vbP_val (p : PosHP c hU n pi) (v : ValH c) (y : List UInt8) :
    (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).val (p, v, y) = v :=
  rfl

theorem vbP_cs_row (h : HiddenExec.rowOfHP c hU n pi (Discharge.Layout.outsOfP c hU n pi) g = some (u, ⟨q, k⟩))
    (R : Reg) :
    (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).cs R ((vbP dj y₀ ht hc hscope hU hpi hpub).pos g) =
      ((Sum.inl (u, q) : PosHP c hU n pi),
        H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList)) := by
  rw [vbP_pos_row h]
  exact vbP_cs_inl R u q

/-- **A registered opening of a row's commit string is a leaf equation** at public inputs. -/
theorem vbP_leaf_row (h : HiddenExec.rowOfHP c hU n pi (Discharge.Layout.outsOfP c hU n pi) g = some (u, ⟨q, k⟩))
    (R : Reg) (p : PosHP c hU n pi) (v : ValH c) (y : List UInt8)
    (he : (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).com (p, v, y) =
      (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).cs R ((vbP dj y₀ ht hc hscope hU hpi hpub).pos g)) :
    H512 (hm96Input (HiddenExec.encPortH c.ports[q.val]! v) y) =
      H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList) := by
  have he' := he.trans (vbP_cs_row h R)
  rcases p with ⟨u', q'⟩ | g'
  · rw [vbP_com_inl] at he'
    obtain ⟨h1, hl⟩ := Prod.mk.inj he'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Sum.inl.inj h1)
    exact hl
  · have h1 := congrArg Prod.fst he'
    rw [vbP_com_fst] at h1
    cases h1

/-- **The deployed layout's rows at a leaf map at public inputs**: `TransportsLiveV5`'s `rows`, the game's row map with
each row at its proof block's leaf `L.ref`. -/
noncomputable def rowsOfLP (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (pi : Flock.HmIn.Ports) (L : LeafMap) :
    Fin ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)).N → Option RowAt := fun g =>
  (HiddenExec.rowOfHP c hU n pi (Discharge.Layout.outsOfP c hU n pi) g).map fun r =>
    ⟨r.2.1.val, L.ref r.2.1.val r.1.val, Discharge.Layout.portRowW c r.2.1, r.2.2.val⟩

/-- **A shaped registered opening opens the deployed commitment at the public file's leaf map** at public inputs
(`transports_shapedS` at `keyProgP`). -/
theorem transports_shapedP (R : Reg) (hrows : RowsOk c)
    (hfirst : ∀ q q' : ℕ, q' < q → q < c.ports.size →
      (rootsOfL c (LeafMap.ofPublic c pub n)).getD q' [] ≠ (rootsOfL c (LeafMap.ofPublic c pub n)).getD q [])
    (hrefs : ∀ (u : Fin n) (q : Fin c.ports.size),
      (LeafMap.ofPublic c pub n).ref q.val u.val < (LeafMap.ofPublic c pub n).cnt q.val)
    {r : RowAt} (hr : rowsOfLP c hU n pi (LeafMap.ofPublic c pub n) g = some r)
    (p : PosHP c hU n pi) (v : ValH c) (y : List UInt8) (hv : RowShaped v) (hvw : v.1.val = r.width)
    (hy : y.length = saltLen)
    (he : (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).com (p, v, y) =
      (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).cs R ((vbP dj y₀ ht hc hscope hU hpi hpub).pos g)) :
    ∃ d : Claims.Msg, (commitDepL ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi))
        (rowsOfLP c hU n pi (LeafMap.ofPublic c pub n)) c (LeafMap.ofPublic c pub n)
        (rootsOfL c (LeafMap.ofPublic c pub n))).Opens sha512Msg (rootsOfL c (LeafMap.ofPublic c pub n))
      ⟨posOf ((keyProgP c pi hU n).circuit (Discharge.Layout.outsOfP c hU n pi)) g,
        (vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).bit
          ((vbP dj y₀ ht hc hscope hU hpi hpub (Reg := Reg)).val (p, v, y)) g, d⟩ = true := by
  obtain ⟨⟨u, q, k⟩, hx, rfl⟩ := Option.map_eq_some_iff.mp hr
  have hle : Discharge.Layout.portRowW c q ≤ Finset.univ.sup (Discharge.Layout.portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  have hl := vbP_leaf_row hx R p v y he
  rw [← LeafMap.dig_ref c pub n q.val u.val] at hl
  rw [vbP_val, vbP_bit_row hx v]
  exact ⟨_, opens_commitDepL _ (rowsOfLP c hU n pi (LeafMap.ofPublic c pub n)) c (LeafMap.ofPublic c pub n)
    (r := ⟨q.val, (LeafMap.ofPublic c pub n).ref q.val u.val, Discharge.Layout.portRowW c q, k.val⟩)
    (by rw [rowOfPos_posOf]; simp [rowsOfLP, hx]) q.isLt (hrefs u q) (LeafMap.le_d c pub n q.isLt)
    (rowsOk_bits hrows q) (by simp [Discharge.Layout.portRowW]) k.isLt (fun q' h => hfirst q.val q' h q.isLt) hle hv hvw
    hy hl⟩

end Game

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **`Bridge.TransportsLiveV5` holds** (`transports_shapedP`): `dupName` and `RootFixesDomainL` give distinct roots,
and the verifier's public-file check every proof block's ref below its port's count (`ref_lt_cntP`). -/
theorem _root_.Flock.Bridge.transportsLiveV5 : Bridge.TransportsLiveV5 := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope hrows hdup hroots kd _ hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags
    hCust τ P vb σ Cc Lm rows κ roots Cd u g _ _ o he hcan
  obtain ⟨⟨r, hr, hv, hvw⟩, hy⟩ := hcan
  obtain ⟨p, v, y⟩ := o
  exact Bridge.transports_shapedP (dj := dj) (y₀ := y₀) (ht := ht) (hc := Flock.Desc.parse_flat hD)
    (hscope := hscope) (hU := hU) (hpi := hpi) (hpub := hpub) _ hrows (Bridge.rootsDistinctL_of_fixes hdup hroots)
    (fun u q => Bridge.ref_lt_cntP hpub u.isLt q.isLt) hr p v y hv hvw hy he

end Flock.Guarantees
