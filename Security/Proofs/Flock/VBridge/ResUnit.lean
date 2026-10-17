import Proofs.Flock.VBridge.Decode

/-!
# The verifier's `rec-residuals/…` unit

`RecResiduals.check` accepts a `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` unit only if it is the net the verifier
builds itself (`RecResiduals.unitLaid S ns nv`) for part `j` of the shape's algebra (`RecResiduals.part`). That net is
`InnerRepCheck{S}`'s build over the verifier's `S` with row ports of `ns` elements and a `v` of `nv`, but it does not go
through `VStar.innerUnit`: its ports need no names, and its S no `checkS`. This file states the unit's theorems at that
net, at the bridge's `residualForms` (`gfsOf S`), which needs no range hypothesis.

* The ports. `partRep ns nv`: a rep with those ports, so that `InCols`, `rowCol` and `vBase` read the part's columns.
  `unitNet'_run`: the verifier's build is `partRun` (the bridge's `residualForms` on the row ports' first bits and `v`'s),
  then zero forms up to 128 words.
* The theorems. `laid_part`: whenever the laid-out net's rows hold at `v`, if the input columns hold `w` on the row ports
  and `vv` on `v`'s (`InCols`), the output columns hold each of S's residuals at `w`, `vv`, 128 bits each, then zeros
  (`PartCols`, in the form `PartS.res_partS` reads). `unit_part`: the same of a copy in a unit of any partition.
  `check_part`, `check_part_holo`: what `check` accepted, at a plain and at a holo statement. `decode_resUnit`,
  `decode_check`, `decode_check_holo`: `decode_innerUnit`, `decode_part` at the verifier's unit.
* The size. `part_small`: the verifier's part has at most 16 residuals, as `laid_part` asks: `groups` adds a residual to
  a part only where `8 (size + 1) ≤ 128` (`groups_small`), and `partS` keeps its group's residuals.
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## The part's ports -/

/-- A rep whose ports are a part's: row ports of `ns` elements, a `v` of `nv`. Only its ports, `nw` and `nv` are read. -/
def partRep (ns : List ℕ) (nv : ℕ) : VStar.RepS := ⟨⟨ns.sum, nv, [], []⟩, ns.map ("", ·)⟩

/-- The part's input ports, one 16-bit port per word. -/
def partIn (ns : List ℕ) (nv : ℕ) : Array ℕ := Array.replicate ((RecResiduals.portWords ns nv).foldl (· + ·) 0) 16

/-- The verifier's build on its input bits: the bridge's `residualForms` on `w` and `v`. -/
def partRun (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) : List Lin × St :=
  (FlockVBridge.residualForms (gfsOf S) (rowForms (partRep ns nv).ports 0)
    (vars (vBase (partRep ns nv)) (128 * nv))).run ⟨innerIn (partRep ns nv), []⟩

/-- Its outputs: the residuals' forms, then zero forms up to 128 words. -/
def partOuts (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) : List Lin :=
  (partRun S ns nv).1 ++ List.replicate (16 * RecResiduals.RES_OUT_WORDS - (partRun S ns nv).1.length) Lin.zero

/-- Where its output columns start. -/
def partBase (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) : ℕ :=
  outStart (partRun S ns nv).2 (partIn ns nv) (Array.replicate RecResiduals.RES_OUT_WORDS 16)

theorem portWordsOf_eq : RecResiduals.portWordsOf = VStar.portWordsOf := rfl

theorem slices_eq : RecResiduals.slices = VStar.slices := by
  funext bits ws
  induction ws generalizing bits with
  | nil => rfl
  | cons w ws ih => simp only [RecResiduals.slices, VStar.slices, ih]

theorem ports_map (ns : List ℕ) (nv : ℕ) :
    (partRep ns nv).ports.map (fun p => VStar.portWordsOf p.2) = ns.map VStar.portWordsOf := by
  induction ns with
  | nil => rfl
  | cons n ns ih => exact congrArg (VStar.portWordsOf n :: ·) ih

theorem ports_snd (ns : List ℕ) (nv : ℕ) : (partRep ns nv).ports.map (·.2) = ns := by
  induction ns with
  | nil => rfl
  | cons n ns ih => exact congrArg (n :: ·) ih

theorem repPorts_partRep (ns : List ℕ) (nv : ℕ) : repPorts (partRep ns nv) = RecResiduals.portWords ns nv :=
  congrArg (· ++ [VStar.portWordsOf nv]) (ports_map ns nv)

theorem innerIn_partRep (ns : List ℕ) (nv : ℕ) :
    innerIn (partRep ns nv) = 16 * (RecResiduals.portWords ns nv).foldl (· + ·) 0 := by
  unfold innerIn
  rw [repPorts_partRep]

theorem vBase_partRep (ns : List ℕ) (nv : ℕ) : vBase (partRep ns nv) = 16 * (ns.map VStar.portWordsOf).sum := by
  unfold vBase
  rw [ports_map]

theorem run_input_bind' {β : Type} (n : ℕ) (f : List Lin → M β) :
    (input n >>= f).run St.empty = (f (vars 0 n)).run ⟨n, []⟩ :=
  run_input_bind n f

/-- The verifier's `w`: each row port's slice cut to its first `128 n` bits. -/
theorem flatMap_ns (f : ℕ × List Lin → List Lin) (hf : ∀ n b, f (n, b) = b.take (128 * n)) :
    ∀ (ns : List ℕ) (k : ℕ) (rest : List (List Lin)),
      (ns.zip (varSlices k (ns.map VStar.portWordsOf) ++ rest)).flatMap f = rowForms (ns.map ("", ·)) k
  | [], _, _ => by simp [rowForms, varSlices]
  | n :: ns, k, rest => by
    simp only [List.map_cons, varSlices, List.cons_append, List.zip_cons_cons, List.flatMap_cons, hf, rowForms,
      take_vars _ _ _ (le_portWords n), flatMap_ns f hf ns]

/-- The verifier's `v`: the slice after the row ports', cut to its first `128 nv` bits. -/
theorem vPart_ns (ns : List ℕ) (nv : ℕ) :
    (((varSlices 0 (ns.map VStar.portWordsOf) ++
      varSlices (0 + 16 * (ns.map VStar.portWordsOf).sum) [VStar.portWordsOf nv]).drop ns.length).headD []).take
        (128 * nv) = vars (16 * (ns.map VStar.portWordsOf).sum) (128 * nv) := by
  rw [List.drop_left' (by rw [length_varSlices, List.length_map])]
  simp only [varSlices, List.headD_cons, Nat.zero_add]
  exact take_vars _ _ _ (le_portWords _)

/-- **The verifier's `rec-residuals` build is the bridge's `residualForms` on its ports**: its state and outputs are
`partRun`'s, then zero forms up to 128 words. -/
theorem unitNet'_run (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) :
    (RecResiduals.unitNet' S ns nv).run St.empty = (partOuts S ns nv, (partRun S ns nv).2) := by
  have hi : innerIn (partRep ns nv) =
      16 * (ns.map VStar.portWordsOf ++ [VStar.portWordsOf nv]).foldl (· + ·) 0 := by
    rw [innerIn_partRep, RecResiduals.portWords, portWordsOf_eq]
  unfold partOuts partRun
  rw [vBase_partRep, hi]
  unfold RecResiduals.unitNet'
  simp only [RecResiduals.portWords, portWordsOf_eq, slices_eq]
  rw [run_input_bind']
  simp only [slices_vars_foldl, varSlices_append]
  rw [flatMap_ns _ _, vPart_ns, RecResiduals.residualForms_eq, FlockVBridge.residualForms_eq]
  · rfl
  · intro n b
    rfl

theorem fresh_partRun (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) :
    Good (innerIn (partRep ns nv)) (partRun S ns nv).2 := by
  unfold partRun
  exact fresh_residualsGo _ _ _ _ (innerIn _) ⟨innerIn _, []⟩ ⟨by simp, fun k r h => by simp at h⟩

theorem res_gfsOf (S : Flock.RecOpen.GfS) {r : ℕ} (hr : r < S.res.length) (hr' : r < (gfsOf S).res.length) :
    (gfsOf S).res[r] = resOf S.res[r] := by
  simp [gfsOf]

/-! ## The laid-out unit -/

/-- The columns from `base` hold each residual of the verifier's part `S` at `w`, `vv`, 128 bits each, then zeros to
128 words. -/
def PartCols (S : Flock.RecOpen.GfS) (w vv : ℕ → F128) (v : ℕ → Bool) (base : ℕ) : Prop :=
  (∀ r (hr : r < S.res.length), ∀ k < 128,
    v (base + (128 * r + k)) = bitOf (resVal (gfsOf S) (termVal w vv) (resOf S.res[r])) k) ∧
    ∀ j, 128 * S.res.length ≤ j → j < 16 * RecResiduals.RES_OUT_WORDS → v (base + j) = false

/-- **The verifier's laid-out `rec-residuals` unit computes its residuals**: whenever its rows hold at `v`, if its input
columns hold `w`'s bits on the row ports and `vv`'s on `v`'s (`InCols`), `w` and `vv` zero past the row ports' elements
and `nv`, its output columns from `partBase` hold each of S's residuals (`resVal`), then zeros (`PartCols`). At an S of
at most 16 residuals, which the verifier's grouping keeps. -/
theorem laid_part (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (hres : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS)
    (v : ℕ → Bool) (h : LaidRows (RecResiduals.unitLaid S ns nv) (inRowsOf (partIn ns nv)) v) (w vv : ℕ → F128)
    (hw0 : ∀ e, ns.sum ≤ e → w e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0) (hin : InCols (partRep ns nv) w vv v) :
    PartCols S w vv v (partBase S ns nv) := by
  have hN : (partIn ns nv).foldl (· + ·) 0 = innerIn (partRep ns nv) := by
    rw [partIn, foldl_replicate, innerIn_partRep]
    exact Nat.mul_comm _ _
  have hL : LaidRows (layout (partRun S ns nv).2 (partIn ns nv) (Array.replicate RecResiduals.RES_OUT_WORDS 16)
      (partOuts S ns nv).toArray) (inRowsOf (partIn ns nv)) v := by
    have e : RecResiduals.unitLaid S ns nv = layout ((RecResiduals.unitNet' S ns nv).run St.empty).2 (partIn ns nv)
        (Array.replicate RecResiduals.RES_OUT_WORDS 16) ((RecResiduals.unitNet' S ns nv).run St.empty).1.toArray := rfl
    rw [e, unitNet'_run] at h
    exact h
  have hg : Good ((partIn ns nv).foldl (· + ·) 0) (partRun S ns nv).2 := by
    rw [hN]
    exact fresh_partRun S ns nv
  have hrows := laid_rows _ _ _ _ v hg hL
  have hP := (FlockVBridge.sound_residualForms (gfsOf S) (rowForms (partRep ns nv).ports 0)
    (vars (vBase (partRep ns nv)) (128 * nv)) ⟨innerIn (partRep ns nv), []⟩).2 _ hrows
  have hcol : ∀ x < innerIn (partRep ns nv),
      v (col ((partIn ns nv).foldl (· + ·) 0) (inRowsOf (partIn ns nv)) x) = v x := by
    intro x hx
    rw [hN]
    simp only [col, hx, ↓reduceIte]
  have hWz : ∀ q, bit (fun x => v (col ((partIn ns nv).foldl (· + ·) 0) (inRowsOf (partIn ns nv)) x))
      (rowForms (partRep ns nv).ports 0) q = wbit w q := by
    intro q
    by_cases hq : q < 128 * ns.sum
    · have hq' : q < 128 * ((partRep ns nv).ports.map (·.2)).sum := by rw [ports_snd]; exact hq
      rw [bit_rowForms _ _ _ _ hq']
      have hlt := rowCol_lt (partRep ns nv).ports 0 q hq'
      rw [hcol _ (by rw [innerIn_eq, vBase]; omega)]
      exact hin.1 q hq
    · rw [bit_of_length_le (by rw [length_rowForms, ports_snd]; omega), wbit, hw0 _ (by omega), bitOf_zero]
  have hVz : ∀ q, bit (fun x => v (col ((partIn ns nv).foldl (· + ·) 0) (inRowsOf (partIn ns nv)) x))
      (vars (vBase (partRep ns nv)) (128 * nv)) q = wbit vv q := by
    intro q
    by_cases hq : q < 128 * nv
    · rw [bit_vars _ _ _ hq]
      have hp : 128 * nv ≤ 16 * VStar.portWordsOf (partRep ns nv).gf.nv := le_portWords nv
      rw [hcol _ (by rw [innerIn_eq]; omega)]
      exact hin.2 q hq
    · rw [bit_of_length_le (by rw [length_vars]; omega), wbit, hv0 _ (by omega), bitOf_zero]
  obtain ⟨hlen, hval⟩ := hP
  change (partRun S ns nv).1.length = _ at hlen
  have hval' := hval w vv hWz hVz
  change ∀ r (hr : r < (gfsOf S).res.length), ∀ k < 128, bit _ (partRun S ns nv).1 (128 * r + k) = _ at hval'
  have hglen : (gfsOf S).res.length = S.res.length := by simp [gfsOf]
  have h16 : RecResiduals.RES_OUT_WORDS = 128 := rfl
  have hsz : (partOuts S ns nv).toArray.size = 16 * RecResiduals.RES_OUT_WORDS := by
    rw [List.size_toArray, partOuts, List.length_append, List.length_replicate, hlen, hglen]
    omega
  have hob : (Array.replicate RecResiduals.RES_OUT_WORDS 16).foldl (· + ·) 0 = 16 * RecResiduals.RES_OUT_WORDS := by
    rw [foldl_replicate]
    exact Nat.mul_comm _ _
  have hout := fun j (hj : j < (partOuts S ns nv).toArray.size) =>
    laid_outs _ _ _ _ v (by rw [hsz, hob]) hL j hj
  refine ⟨fun r hr k hk => ?_, fun j hj1 hj2 => ?_⟩
  · have hr' : r < (gfsOf S).res.length := by rw [hglen]; exact hr
    have hj : 128 * r + k < (partRun S ns nv).1.length := by rw [hlen, hglen]; omega
    unfold partBase
    rw [hout _ (by rw [hsz]; omega), ← res_gfsOf S hr hr', ← hval' r hr' k hk]
    simp only [partOuts, List.getElem_toArray, List.getElem_append_left hj, bit, List.getElem?_eq_getElem hj,
      Option.getD_some]
  · unfold partBase
    rw [hout _ (by rw [hsz]; omega)]
    simp [partOuts, List.getElem_append, show ¬ j < (partRun S ns nv).1.length by rw [hlen, hglen]; omega]

theorem inWords_unitLaid {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String} {net : Flock.Net}
    (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok net) : net.inWords * Flock.WORD = inRowsOf (partIn ns nv) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hw]
  simp only [RecResiduals.unitLaid, layout_inGroups]
  simp [inRowsOf, partIn]

/-- **What `RecResiduals.check` accepted**: a unit named `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` at a shape the
verifier accepts is the net the verifier builds for part `j` of that shape's algebra, holo as the statement's META says
(`holoOf`). -/
theorem check_part_at {c : Flock.Circuit} (h : RecResiduals.check c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) {holo : Bool} (hh : RecResiduals.holoOf c = .ok holo) :
    ∃ S ns nv, RecResiduals.part { sh with holo } j = .ok (S, ns, nv) ∧
      netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit := by
  unfold RecResiduals.check at h
  rw [hp] at h
  dsimp only at h
  rw [hsh] at h
  dsimp only at h
  split at h
  · cases h
  · rw [hh] at h
    dsimp only at h
    split at h
    · cases h
    · rename_i S ns nv hpart
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i g hg
          split at h
          · rename_i hcg
            exact ⟨S, ns, nv, hpart, by rw [hcg]; exact hg⟩
          · cases h

/-- `shape?`'s shape is plain. -/
theorem shape?_holo {m k cl : ℕ} {sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh) :
    sh.holo = false := by
  unfold RecResiduals.shape? at hsh
  cases hf : Flock.fast100 m with
  | none => rw [hf] at hsh; cases hsh
  | some s =>
    rw [hf] at hsh
    simp at hsh
    obtain ⟨-, rfl⟩ := hsh
    rfl

/-- **`check_part_at` on a plain statement**: META says no `holo`, so the unit is the plain part's net. -/
theorem check_part {c : Flock.Circuit} (h : RecResiduals.check c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf c = .ok false) :
    ∃ S ns nv, RecResiduals.part sh j = .ok (S, ns, nv) ∧
      netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit := by
  obtain ⟨S, ns, nv, hpart, hn⟩ := check_part_at h hp hsh hh
  have e : { sh with holo := false } = sh := by
    have h0 := shape?_holo hsh
    cases sh
    simp only at h0
    subst h0
    rfl
  rw [e] at hpart
  exact ⟨S, ns, nv, hpart, hn⟩

/-- **`check_part_at` on a holo statement**: META says `holo: true`, so the unit is the holo part's net, the hidden
units' residuals. -/
theorem check_part_holo {c : Flock.Circuit} (h : RecResiduals.check c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf c = .ok true) :
    ∃ S ns nv, RecResiduals.part { sh with holo := true } j = .ok (S, ns, nv) ∧
      netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit :=
  check_part_at h hp hsh hh

/-- Part `j`'s plain and holo nets differ. They need not: at (25,22,8), part 1's holo net is its plain one. -/
def NetsDiffer (m k cl j : ℕ) (sh : Flock.Shape) : Prop :=
  ∀ S ns nv S' ns' nv' g g', RecResiduals.part sh j = .ok (S, ns, nv) →
    RecResiduals.part { sh with holo := true } j = .ok (S', ns', nv') →
    netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S ns nv) = .ok g →
    netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S' ns' nv') = .ok g' → g ≠ g'

/-- **A holo statement on a plain part refuses**: where part `j`'s plain and holo nets differ, `check` fails on a unit
that is the plain net under META `holo: true`. -/
theorem check_refuses_holo_on_plain {c : Flock.Circuit} {m k cl j : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf c = .ok true) (hd : NetsDiffer m k cl j sh)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part sh j = .ok (S, ns, nv))
    (hn : netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit) :
    RecResiduals.check c ≠ .ok () := fun h => by
  obtain ⟨S', ns', nv', hpart', hn'⟩ := check_part_at h hp hsh hh
  exact hd S ns nv S' ns' nv' _ _ hpart hpart' hn hn' rfl

/-- **A plain statement on a holo part refuses**: where part `j`'s plain and holo nets differ, `check` fails on a unit
that is the holo net under META without `holo` (or `holo: false`). -/
theorem check_refuses_plain_on_holo {c : Flock.Circuit} {m k cl j : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf c = .ok false) (hd : NetsDiffer m k cl j sh)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part { sh with holo := true } j = .ok (S, ns, nv))
    (hn : netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit) :
    RecResiduals.check c ≠ .ok () := fun h => by
  obtain ⟨S', ns', nv', hpart', hn'⟩ := check_part h hp hsh hh
  exact hd S' ns' nv' S ns nv _ _ hpart' hpart hn' hn rfl

/-! ## The part's residuals fit -/

/-- **Every part `groups` returns has at most 16 residuals**: a residual joins a part, or starts one, only where `fit`
keeps `8 (size + 1) ≤ 128`. -/
theorem groups_small {S : Flock.RecAlgebra.GfS} {sizes : Array ℕ} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (h : RecResiduals.groups S sizes = .ok gs) : ∀ g ∈ gs, 8 * g.1.size ≤ RecResiduals.RES_OUT_WORDS := by
  unfold RecResiduals.groups at h
  obtain ⟨gs', hl, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · cases h
    refine FlockSoundness.Refine.forIn_range_inv _ _
      (fun gs : Array (Array ℕ × Array ℕ × Array ℕ) => ∀ g ∈ gs, 8 * g.1.size ≤ RecResiduals.RES_OUT_WORDS) ?_ _ _
      (by simp) hl
    intro i s y hs hy
    dsimp only at hy
    split at hy
    · rename_i gi hgi
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      intro g hg
      rcases Array.mem_or_eq_of_mem_setIfInBounds hg with hg | rfl
      · exact hs g hg
      · obtain ⟨hlt, hfit, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hgi
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hfit
        simp only [Array.size_push, getElem!_pos s gi hlt]
        exact hfit.2
    · split at hy
      · simp only [pure, Except.pure, Except.ok.injEq] at hy
        subst hy
        intro g hg
        rcases Array.mem_push.1 hg with hg | rfl
        · exact hs g hg
        · simp [RecResiduals.RES_OUT_WORDS]
      · cases hy

/-- **The verifier's part has at most 16 residuals** (`groups_small`): `partS` gives it one residual for each of its
group's (`PartS.res_partS`). -/
theorem part_small {sh : Flock.Shape} {j : ℕ} {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h : RecResiduals.part sh j = .ok (S, ns, nv)) : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS := by
  unfold RecResiduals.part at h
  obtain ⟨S0, -, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs, hg, h⟩ := FlockVBridge.PartS.bind_ok h
    split at h
    · rename_i R P V hj
      obtain ⟨S', hS', h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -, -⟩ := h
      obtain ⟨hn, -⟩ := FlockVBridge.PartS.res_partS hS' (w := fun _ => 0) (v := fun _ => 0) (w' := fun _ => 0)
        (v' := fun _ => 0) (fun _ _ => rfl) (fun _ _ => rfl)
      rw [hn]
      exact groups_small hg _ (Array.mem_of_getElem? hj)
    · cases h

section Unit

open FlockSoundness FlockSoundness.Audit

/-- **A copy of the verifier's laid-out `rec-residuals` unit, in a unit of any partition, computes its residuals**
(`laid_part`). Let the verifier build `net` from `RecResiduals.unitLaid S ns nv` (`netOf`), and let unit `u` be an
instance of `k` stacked copies of the net's rows, the transcript `tr` carrying 1 at the stack's constant. Then at copy
`j`'s column values (`copyVal`), `laid_part` holds. -/
theorem unit_part {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {net : Flock.Net} (hB : Built net)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (tr : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : tr (I.wire ((Rows.ofNet hB).stack k).one) = true)
    (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (hres : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS)
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok net) (hu : net.unitLog ≤ 32)
    (w vv : ℕ → F128) (hw0 : ∀ e, ns.sum ≤ e → w e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) w vv (copyVal I tr hj)) :
    PartCols S w vv (copyVal I tr hj) (partBase S ns nv) := by
  obtain ⟨hs, h1⟩ := sat_copy I tr hj
  have hL := laidRows_ofNet hB hn hu hs (h1.trans hone)
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv hres _ hL w vv hw0 hv0 hin

end Unit

section Decode

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

/-- **The verifier's `rec-residuals` unit computes its residuals in a layer agreeing with the registrant** (`unit_part`
at `fill`). Let the verifier build `c`'s unit net from `RecResiduals.unitLaid S ns nv` (`netOf`), and let the layer
`bits` agree with the registrant (`RegAgree`) and carry 1 on unit `u`'s constant gate. If copy `j`'s input columns are
registered bits of `w` on the row ports and of `vv` on `v`'s port (`RegBit`), `w`, `vv` zero past the row ports'
elements and `nv`, the copy's output columns in `bits` hold each of S's residuals (`PartCols`). -/
theorem decode_resUnit {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (hres : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS)
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit) (hu : c.unit.unitLog ≤ 32)
    (w vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → w e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hw : ∀ q < 128 * ns.sum, RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit w q))
    (hv : ∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit vv q)) :
    PartCols S w vv (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  have h1 : fill zo bits ((keyInst hU nU (Layout.outsOf c hU nU) u).wire ((Rows.ofNet hU).stack c.upv).one) = true := by
    rw [fill_one zo hA _ hj, keyInst_one]
    exact hone
  obtain ⟨hr, hz⟩ := unit_part hU (keyInst hU nU (Layout.outsOf c hU nU) u) (fill zo bits) hj h1 S ns nv hres hn hu
    w vv hw0 hv0 (inCols_fill zo bits hj (partRep ns nv) w vv hw hv)
  have hbase : (Rows.ofNet hU).nIn ≤ partBase S ns nv := by
    rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
    exact inRows_le_outStart _ _ _
  refine ⟨fun r hr' k hk => ?_, fun x hx1 hx2 => ?_⟩
  · exact (copyVal_fill zo hA _ hj (by omega)).symm.trans (hr r hr' k hk)
  · exact (copyVal_fill zo hA _ hj (by omega)).symm.trans (hz x hx1 hx2)

/-- **A part the verifier pinned computes its residuals at the whole ports, in a layer agreeing with the registrant**
(`decode_part` at the verifier's unit). Let `RecResiduals.check` accept `c`, its unit named for part `p` at a shape
`sh` the verifier accepts, part `p` being `S` with row ports of `ns` elements and a `v` of `nv`. When copy `j` of unit
`u`'s row ports are registered bits of the whole message elements `w` at the part's indices `wI`, and its `v` port
registered bits of the verifier's coefficients (`vOf`) at `V`, its output columns in `bits` hold the part's residuals at
part ports `w'`, `v'` that read `w` at `wI` and `vOf` at `V`: `PartS.res_partS`' `hw` and `hv` at `wI = wIdx sizes P`.
The part has at most 16 residuals (`part_small`). -/
theorem decode_check {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ} (hp : RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (hh : RecResiduals.holoOf c = .ok false)
    (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128)
    (w : ℕ → Flock.F128) (wI V : Array ℕ) (hwI : wI.size ≤ ns.sum) (hV : V.size ≤ nv)
    (hw : ∀ q < 128 * ns.sum, RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub wI w) q))
    (hv : ∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub V (vOf sh rounds extras comb)) q)) :
    ∃ w' v' : ℕ → Flock.F128, (∀ k (hk : k < wI.size), w' k = w wI[k]) ∧
      (∀ k (hk : k < V.size), v' k = vOf sh rounds extras comb V[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_part hc hp hsh hh
  rw [hpart] at hpart'
  cases hpart'
  exact ⟨sub wI w, sub V (vOf sh rounds extras comb), sub_lt wI w, sub_lt V _,
    decode_resUnit zo hA u hj hone S ns nv (part_small hpart) hn hu _ _ (sub_ge w hwI) (sub_ge _ hV) hw hv⟩

/-- **`decode_check` on a holo statement** (`check_part_holo`): META says `holo: true`, and `sh` is the holo form of
the shape the verifier accepts, whose part `p` is the hidden units' `S`. -/
theorem decode_check_holo {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ} (hp : RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh0 sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh0) (hho : sh = { sh0 with holo := true })
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (hh : RecResiduals.holoOf c = .ok true)
    (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128)
    (w : ℕ → Flock.F128) (wI V : Array ℕ) (hwI : wI.size ≤ ns.sum) (hV : V.size ≤ nv)
    (hw : ∀ q < 128 * ns.sum, RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub wI w) q))
    (hv : ∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub V (vOf sh rounds extras comb)) q)) :
    ∃ w' v' : ℕ → Flock.F128, (∀ k (hk : k < wI.size), w' k = w wI[k]) ∧
      (∀ k (hk : k < V.size), v' k = vOf sh rounds extras comb V[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  subst hho
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_part_holo hc hp hsh hh
  rw [hpart] at hpart'
  cases hpart'
  exact ⟨sub wI w, sub V (vOf _ rounds extras comb), sub_lt wI w, sub_lt V _,
    decode_resUnit zo hA u hj hone S ns nv (part_small hpart) hn hu _ _ (sub_ge w hwI) (sub_ge _ hV) hw hv⟩

end Decode

end FlockVBridge.OfVStar

