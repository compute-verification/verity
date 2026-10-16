import Proofs.Flock.C3.Reps.WithReads
import Proofs.Flock.C3.PubRows.Shape

/-!
# C3's rep decoder over the verifier's own list

`RepMaps` (`Maps.lean`) is generic in the list it decodes. Here it is at the verifier's own (`RecVStar.statements i`,
`listAt`), at the inner shape's form: `RecResiduals.shape?`'s `sh0` at `i.holo`, holo at a private inner shape, where
the list is V*'s holo statements (`FlockC3.VStarList.statements_holo`).

* `list_ok`: where the list builds, S's structure and groups return at that form, V*'s parts as many as the groups
  (`RecVStar.statements` computes both and compares the counts): `hStruct_of_checkH`'s structure, from the list rather
  than from one part's check.
* `inner_idx`, `recOpen_idx`: the list's statement tagged `.inner p` is its `nLev i + p`th, the one tagged `.recOpen ℓ`
  its `ℓ`th (V*'s levels, then its parts, then candidate 3's, none of them so tagged: `candidate3_kinds`).
* `RepMaps.atList`: a decoder over the list whose part and opening sessions are those (`hsess_atList`,
  `hsessO_atList`), at the units and values given.
* `RepMaps.algebraDecodes_atList`, `RepMaps.withReads_algebraDecodes_atList`: G's items 1–3 at the inner shape's form
  from each part's `ResZeroAtLH`, `RepMaps.algebraDecodes` and `withReads_algebraDecodes` with the structure and groups
  the list's.
-/

namespace FlockC3.Reps

noncomputable section

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.OfVStar (ResZeroAtLH)

theorem ok_of_mapError {ε A : Type} {x : Except ε A} {f : ε → ε} {a : A} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

/-- V*'s shape of the inner session, `RecVStar.statements`' `vsh`. -/
def vshOf (i : RecVStar.Inner) : VStar.Algebra.Shape := { m := i.m, kLog := i.kLog, claims := i.claims }

/-- V*'s level count at the inner shape, where the list's parts start. -/
def nLev (i : RecVStar.Inner) : ℕ :=
  match (vshOf i).sched with
  | .ok s => s.levels.size
  | .error _ => 0

/-- **The verifier's list**: where it builds, the inner shape is `shape?`'s `sh0`, S's structure and groups return at
`sh0`'s form at `i.holo`, V*'s parts are as many as the groups, and the list is V*'s statements at that form
(`statementsOf`), then candidate 3's. -/
theorem list_ok {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ sh0, RecResiduals.shape? i.m i.kLog i.claims = some sh0 ∧
      ∃ sp ps GS gs A C, VStar.Algebra.specOf (vshOf i) i.holo = .ok sp ∧ VStar.parts sp = .ok ps ∧
        RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS ∧
        RecResiduals.groups GS (RecResiduals.portSizes { sh0 with holo := i.holo }) = .ok gs ∧ ps.size = gs.size ∧
        VStar.statementsOf (vshOf i) i.holo = .ok A ∧ RecVStar.candidate3Statements i = .ok C ∧ ss = A ++ C := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i sh0 hsh
  split at h
  · cases h
  obtain ⟨ps, hps, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, hgs, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  rename_i hsz
  obtain ⟨A, hA, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨C, hC, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u, -, h⟩ := FlockC3.VStarList.eok_bind h
  cases h
  obtain ⟨sp, hsp, hps⟩ := FlockC3.VStarList.eok_bind (ok_of_mapError hps)
  obtain ⟨GS, hGS, hgs⟩ := FlockC3.VStarList.eok_bind (ok_of_mapError hgs)
  exact ⟨sh0, hsh, sp, ps, GS, gs, A, C, hsp, hps, hGS, hgs, by simpa using hsz, hA, hC, rfl⟩

/-- **Candidate 3's statements** are RecSparse, R's levels and the parts of R's opening. -/
theorem candidate3_kinds {i : RecVStar.Inner} {C : Array VStar.Stmt} (h : RecVStar.candidate3Statements i = .ok C) :
    ∀ s ∈ C, (∃ sp hr, s = .sparse sp hr) ∨ (∃ l L H n, s = .recOpenStacked l L H n) ∨
      ∃ m c k S ports nv, s = .opening m c k S ports nv := by
  intro s hs
  cases hsp : i.sparse with
  | none =>
    rw [(FlockC3.VStarList.candidate3_ok h).1 hsp] at hs
    simp at hs
  | some sp =>
    obtain ⟨rest, rfl, hrest⟩ := FlockC3.PubRows.candidate3_rest h hsp
    rcases Array.mem_append.1 hs with hs | hs
    · simp only [List.mem_toArray, List.mem_singleton] at hs
      exact .inl ⟨_, _, hs⟩
    · exact .inr (hrest s hs)

/-- **The list's statements by index**: past V*'s levels and parts, candidate 3's. -/
theorem getElem?_list {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ (s : Flock.Schedule) (ps : Array VStar.Part) (C : Array VStar.Stmt), nLev i = s.levels.size ∧
      RecVStar.candidate3Statements i = .ok C ∧ ∀ j,
        ss[j]? = if j < s.levels.size then some (.recOpen j (2 ^ (s.levels[j]!).lanes)
            ((s.levels[j]!).d - (s.levels[j]!).cap) (2 * (s.levels[j]!).q))
          else if j < s.levels.size + ps.size then
            some (.inner (j - s.levels.size)
              s!"rec-residuals/m{(vshOf i).m}-k{(vshOf i).kLog}-c{(vshOf i).claims}-p{j - s.levels.size}"
              ps[j - s.levels.size]!)
          else C[j - (s.levels.size + ps.size)]? := by
  obtain ⟨-, -, -, -, -, -, A, C, -, -, -, -, -, hA, hC, rfl⟩ := list_ok h
  obtain ⟨s, sp, ps, hs, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  refine ⟨s, ps, C, by simp [nLev, hs], hC, fun j => ?_⟩
  rw [← Array.getElem?_toList]
  simp only [Array.toList_append, List.append_assoc]
  split
  · rename_i h1
    rw [List.getElem?_append_left (by simpa using h1)]
    simp [h1]
  · rename_i h1
    rw [List.getElem?_append_right (by simp; omega)]
    simp only [List.length_map, List.length_range]
    split
    · rename_i h2
      rw [List.getElem?_append_left (by simp; omega)]
      simp [show j - s.levels.size < ps.size by omega]
    · rename_i h2
      rw [List.getElem?_append_right (by simp; omega), Array.getElem?_toList]
      simp only [List.length_map, List.length_range]
      congr 1
      omega

/-- **The list's statement tagged `.inner p` is its `nLev i + p`th.** -/
theorem inner_idx {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) {j p : ℕ}
    {nm : String} {pt : VStar.Part} (hj : ss[j]? = some (.inner p nm pt)) : j = nLev i + p := by
  obtain ⟨s, ps, C, hn, hC, hss⟩ := getElem?_list h
  rw [hss j] at hj
  rw [hn]
  split at hj
  · cases hj
  split at hj
  · simp only [Option.some.injEq, VStar.Stmt.inner.injEq] at hj
    omega
  · rcases candidate3_kinds hC _ (Array.mem_of_getElem? hj) with ⟨_, _, he⟩ | ⟨_, _, _, _, he⟩ |
      ⟨_, _, _, _, _, _, he⟩ <;> cases he

/-- **The list's statement tagged `.recOpen ℓ` is its `ℓ`th.** -/
theorem recOpen_idx {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) {j ℓ : ℕ}
    {LANES H m : ℕ} (hj : ss[j]? = some (.recOpen ℓ LANES H m)) : j = ℓ := by
  obtain ⟨s, ps, C, -, hC, hss⟩ := getElem?_list h
  rw [hss j] at hj
  split at hj
  · simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hj
    exact hj.1
  split at hj
  · cases hj
  · rcases candidate3_kinds hC _ (Array.mem_of_getElem? hj) with ⟨_, _, he⟩ | ⟨_, _, _, _, he⟩ |
      ⟨_, _, _, _, _, _, he⟩ <;> cases he

theorem size_list {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    vstarCount i = ss.size := by
  unfold vstarCount
  rw [h]

/-- The verifier's list over `Fin (vstarCount i)`. -/
def listAt {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (j : Fin (vstarCount i)) : VStar.Stmt :=
  ss[j.val]'(lt_of_lt_of_eq j.isLt (size_list h))

theorem listAt_getElem? {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (j : Fin (vstarCount i)) : ss[j.val]? = some (listAt h j) :=
  Array.getElem?_eq_getElem _

/-- Index `j` of the list where it has one, `0` past it (at `hS`). -/
def idxAt {i : RecVStar.Inner} (hS : 0 < vstarCount i) (j : ℕ) : Fin (vstarCount i) :=
  if h : j < vstarCount i then ⟨j, h⟩ else ⟨0, hS⟩

theorem idxAt_eq {i : RecVStar.Inner} (hS : 0 < vstarCount i) (j : Fin (vstarCount i)) : idxAt hS j.val = j := by
  simp [idxAt, j.isLt]

/-- **Part `p`'s session** (`RepMaps.hsess` at the list): the statement tagged `.inner p` is index `nLev i + p`. -/
theorem hsess_atList {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (hS : 0 < vstarCount i) (p : ℕ) (j : Fin (vstarCount i)) (hj : ∃ nm pt, listAt h j = .inner p nm pt) :
    idxAt hS (nLev i + p) = j := by
  obtain ⟨nm, pt, hj⟩ := hj
  rw [← inner_idx h ((listAt_getElem? h j).trans (congrArg some hj)), idxAt_eq]

/-- **Level `ℓ`'s session** (`RepMaps.hsessO` at the list): the statement tagged `.recOpen ℓ` is index `ℓ`. -/
theorem hsessO_atList {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (hS : 0 < vstarCount i) (ℓ : ℕ) (j : Fin (vstarCount i)) (hj : ∃ LANES H m, listAt h j = .recOpen ℓ LANES H m) :
    idxAt hS ℓ = j := by
  obtain ⟨LANES, H, m, hj⟩ := hj
  rw [← recOpen_idx h ((listAt_getElem? h j).trans (congrArg some hj)), idxAt_eq]

/-- **A decoder over the verifier's list**: part `p`'s session the list's `nLev i + p`th, level `ℓ`'s openings' its
`ℓ`th (`hsess_atList`, `hsessO_atList`), at the units `un`, `unO` and the values given. -/
def RepMaps.atList {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (hS : 0 < vstarCount i) {Vs : Fin (vstarCount i) → VStmt} {sh : Flock.Shape} {Reg Ω C : Type}
    (un : (p : ℕ) → Fin 2 → Fin (Vs (idxAt hS (nLev i + p))).nv)
    (hun : ∀ p r, r.val < (Vs (idxAt hS (nLev i + p))).nv → (un p r).val = r.val)
    (unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (idxAt hS ℓ)).nv)
    (hunO : ∀ ℓ r j, r.val * (sh.sched.levels[ℓ]!).q + j < (Vs (idxAt hS ℓ)).nv →
      (unO ℓ r j).val = r.val * (sh.sched.levels[ℓ]!).q + j)
    (rounds : Reg → Ω → List C → Fin 2 → Array (Array F128)) (extras : Reg → Ω → List C → Fin 2 → Array Flock.Claim)
    (rep₀ : ((j : Fin (vstarCount i)) → (Vs j).Bits) → Fin 2 → RepData)
    (rows : ((j : Fin (vstarCount i)) → (Vs j).Bits) → Fin 2 → ℕ → Array (Array F128))
    (alpha : Reg → Ω → List C → Fin 2 → ℕ → Array F128)
    (acc : ((j : Fin (vstarCount i)) → (Vs j).Bits) → ℕ → Array F128) :
    RepMaps (listAt h) Vs sh Reg Ω C where
  sess p := idxAt hS (nLev i + p)
  hsess p j hj := hsess_atList h hS p j hj
  un := un
  hun := hun
  sessO := idxAt hS
  hsessO ℓ j hj := hsessO_atList h hS ℓ j hj
  unO := unO
  hunO := hunO
  rounds := rounds
  extras := extras
  rep₀ := rep₀
  rows := rows
  alpha := alpha
  acc := acc

section outer

variable {n : ℕ} {L : Law n} {I : Inner} {Cm : Type} {Reg Reg' Reg₂ : Type} {i : RecVStar.Inner}
  {ss : Array VStar.Stmt}

/-- **G items 1–3 at the decoder over the verifier's list** (`RepMaps.algebraDecodes`): at the inner shape's form, the
structure and groups are the list's (`list_ok`), so each part's `ResZeroAtLH` at the groups that form computes gives
`AlgebraDecodes`. -/
theorem RepMaps.algebraDecodes_atList (h : RecVStar.statements i = .ok ss) {Vs : Fin (vstarCount i) → VStmt}
    {sh0 : Flock.Shape} (hsh : RecResiduals.shape? i.m i.kLog i.claims = some sh0)
    (M : RepMaps (listAt h) Vs { sh0 with holo := i.holo } Reg L.Ω (Cm × I.Coin))
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin (vstarCount i)) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (hvr : ∀ GS, RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS →
      ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
        RecAlgebra.verifierRows { sh0 with holo := i.holo } (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
          (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb =
            .ok (v, info, kappa) ∧ v.size = GS.nv)
    (hsz : ∀ GS, RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS →
      ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
        (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw ∧
          (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (hres : ∀ GS gs, RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS →
      RecResiduals.groups GS (RecResiduals.portSizes { sh0 with holo := i.holo }) = .ok gs →
      ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
        ResZeroAtLH (zs R ω t R₂ (M.sess p)) (layers zs dg N k Rw R ω t R₂ τ (M.sess p)) (M.un p r)
          (FlockVBridge.Family.cp p r) { sh0 with holo := i.holo } gs p
          (RecAlgebra.el (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w)
          (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
          (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras
          (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb) :
    AlgebraDecodes zs dg N k Rw { sh0 with holo := i.holo } M.rep := by
  obtain ⟨sh1, hsh1, -, -, GS, gs, -, -, -, -, hGS, hgs, -⟩ := list_ok h
  obtain rfl : sh1 = sh0 := Option.some.inj (hsh1.symm.trans hsh)
  exact M.algebraDecodes zs dg N k Rw ⟨hGS, hgs⟩ (hvr GS hGS) (hsz GS hGS) (hres GS gs hGS hgs)

end outer

section reads

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {i : RecVStar.Inner}
  {ss : Array VStar.Stmt} {Vs : Fin (vstarCount i) → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount i)) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String}
  {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {s : Fin 2 → String}
  {comb : ((j : Fin (vstarCount i)) → (Vs j).Bits) → Fin 2 → Fin 64 → F128}

/-- **G items 1–3 at the reads, over the verifier's list** (`withReads_algebraDecodes`): at the inner shape's form, holo
at a private inner shape, the structure and groups are the list's (`list_ok`), so `AlgebraDecodes` takes each part's
`ResZeroAtLH` at copy 0 at one `w` per rep and the verifier rows at the reads, at the groups that form computes. -/
theorem RepMaps.withReads_algebraDecodes_atList (h : RecVStar.statements i = .ok ss) {sh0 : Flock.Shape}
    (hsh : RecResiduals.shape? i.m i.kLog i.claims = some sh0)
    (M : RepMaps (listAt h) Vs { sh0 with holo := i.holo } Reg L.Ω (Cm × (holoInnerAt T CT st hp).Coin))
    (hrc : ResReadCol Vs { sh0 with holo := i.holo } M.sess)
    (hvr : ∀ GS, RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS →
      ∀ R ω t R₂ τ o, Accepted (I := holoInnerAt T CT st hp) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
        ∃ v info kappa, RecAlgebra.verifierRows { sh0 with holo := i.holo } (heldRounds (held (wr t)) (s ρ))
          (M.extras R ω t ρ) (Array.ofFn (comb (layers zs dg N k Rw R ω t R₂ τ) ρ)) = .ok (v, info, kappa) ∧
          v.size = GS.nv)
    (hw : ∀ GS gs, RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS →
      RecResiduals.groups GS (RecResiduals.portSizes { sh0 with holo := i.holo }) = .ok gs →
      ∀ R ω t R₂ τ o, Accepted (I := holoInnerAt T CT st hp) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
        ∃ w : ℕ → F128, ∀ p < gs.size, ResZeroAtLH (zs R ω t R₂ (M.sess p))
          (layers zs dg N k Rw R ω t R₂ τ (M.sess p)) (M.un p ρ) 0 { sh0 with holo := i.holo } gs p w
          (heldRounds (held (wr t)) (s ρ)) (M.extras R ω t ρ) (Array.ofFn (comb (layers zs dg N k Rw R ω t R₂ τ) ρ))) :
    AlgebraDecodes zs dg N k Rw { sh0 with holo := i.holo } (M.withReads wr held s comb).rep := by
  obtain ⟨sh1, hsh1, -, -, GS, gs, -, -, -, -, hGS, hgs, -⟩ := list_ok h
  obtain rfl : sh1 = sh0 := Option.some.inj (hsh1.symm.trans hsh)
  exact M.withReads_algebraDecodes ⟨hGS, hgs⟩ hrc (hvr GS hGS) (hw GS gs hGS hgs)

end reads

end

end FlockC3.Reps
