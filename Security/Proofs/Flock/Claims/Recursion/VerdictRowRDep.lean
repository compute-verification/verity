import Proofs.Flock.Claims.Recursion.DecompBDeployed
import Proofs.Flock.Claims.Recursion.VerdictRowECoin
import Proofs.Flock.Claims.Recursion.VerdictLoop

/-!
# Row R's leaf at the assembly's pieces

`Top.TargetRDep rd (PD rd resP key elemAt vvAt side posAt) acc`, the row R leaf of `recKSDecompL_of_leaves` (at
`acc := Top.accRC rd key`), from one fact about the assembly's `rec-residuals` pieces (`targetRDep_of_cover`): at every
decision `At` admits, each rep `r` and group `p` of S has a piece of `resP x` whose group is the verifier's part `p`,
whose elements at its key are the rep's `w` at the group's row ports, and whose verifier values at its key are
`verifierRows`' `v` at the group's `V`. The four accumulators are `hacc` (`accR_size`), and the pieces hold because `PD`
says so (`res_holds`). The leaf's `keyAt r p` is that piece's key.

The cover needs the key to carry the group: two groups' pieces at one key read the same elements and verifier values,
so where both cover rep `r` the groups' row ports would read the same `w` elements.

* `resVRead`: the verifier values a piece keyed `(jj, o, g, u)` reads off the statement and the coins: at a statement
  that sets up, the verifier rows at table 0's shape, its extra claims at the first coin's link points, rep `o + g`'s
  `canonRounds` and the comb table 0's fold gives at them (`combRead`, what `combOf` computes); group `jj`'s `V`
  elements of their `v`. A copy `(o, g)` of instance `o · c.g + g < 2` with `g < c.g` is instance `o + g`.
* `DecAt.resVRead_eq`: at a decision of an inner statement read at the dense form, `resVRead` at a key of group `p`
  and `o + g = r` is the decision's `v` at group `p`'s `V`, rep `r`.
* `targetRDep_resV`, `targetRDep_accRC`: the leaf at `vvAt := resVRead`, from the cover's other two halves
  (`ElemCover`): a piece of the group at a copy of the rep, whose elements are the rep's `w` at the group's row ports.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf setupOf jAt)
open Flock.RecAlgebra (verifierRows structureOf el)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (HClass)

noncomputable section

/-- The comb a table's fold gives at a shape's residual coins, as `combOf` computes it from a table's files. -/
def combRead (st : Setup) (sh : Shape) (rounds : Array (Array F128)) (extras : Array Claim) : Array F128 :=
  let c : Array F128 :=
    match Flock.RecAlgebra.residualLins sh (Flock.RecAlgebra.Coins.ofRounds rounds) extras with
    | .ok (_, info) =>
      match st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn
          info.lc.fold.ts with
      | .ok v => v
      | .error _ => #[]
    | .error _ => #[]
  if c.size = 64 then c else Array.replicate 64 0

/-- Group `p`'s `V` element `k` of the verifier rows' `v` at a shape, `0` where they or the groups don't run. -/
def resVOf (sh : Shape) (rounds : Array (Array F128)) (extras : Array Claim) (comb : Array F128) (p k : ℕ) : F128 :=
  match verifierRows sh rounds extras comb, structureOf sh with
  | .ok (v, _, _), .ok S =>
    match groups S (portSizes sh) with
    | .ok gs =>
      match gs[p]? with
      | some g => match g.2.2[k]? with
        | some j => el v j
        | none => 0
      | none => 0
    | .error _ => 0
  | _, _ => 0

/-- Table 0's extra claims at the first coin's link points. -/
def extrasRead (spec : SessionSpec) (sts : Array Stmt) (cs : List Coin) : Array Claim :=
  ((tabsOf sts)[0]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]

/-- Table 0's shape at those extra claims. -/
def shapeRead (spec : SessionSpec) (sts : Array Stmt) (cs : List Coin) : Shape :=
  shapeOf (tabsOf sts)[0]! ((fast100 (tabsOf sts)[0]!.m).getD default) (extrasRead spec sts cs).size

/-- Group `p`'s verifier value `k` at rep `r`, off the setup and the coins. -/
def resVAt (spec : SessionSpec) (sts : Array Stmt) (cs : List Coin) (r p k : ℕ) : F128 :=
  resVOf (shapeRead spec sts cs) (canonRounds spec sts cs.tail.dropLast r) (extrasRead spec sts cs)
    (combRead (tabsOf sts)[0]! (shapeRead spec sts cs) (canonRounds spec sts cs.tail.dropLast r)
      (extrasRead spec sts cs)) p k

/-- **The verifier values a `rec-residuals` piece reads off the statement and the coins**: at key `(jj, o, g, u)`,
group `jj`'s values at rep `o + g` (`resVAt`), `0` at a statement that doesn't set up. -/
def resVRead (x : Claims.Msg) (cs : List Coin) (key : ℕ × ℕ × ℕ × ℕ) (k : ℕ) : F128 :=
  match setupOf false x with
  | some v => resVAt v.2.1 v.2.2 cs (key.2.1 + key.2.2.1) key.1 k
  | none => 0

theorem resVOf_eq {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb : Array F128}
    {v : Array F128} {info : RecAlgebra.Info RecAlgebra.Lin} {kappa : Array (Array (Array F128))} {S : RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hv : verifierRows sh rounds extras comb = .ok (v, info, kappa))
    (hS : structureOf sh = .ok S) (hg : groups S (portSizes sh) = .ok gs) {p k : ℕ} (hp : p < gs.size)
    (hk : k < gs[p].2.2.size) : resVOf sh rounds extras comb p k = el v gs[p].2.2[k] := by
  unfold resVOf
  simp [hv, hS, hg, hp, hk]

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128)

/-- **A piece reads the decision's verifier values**: at a decision of an inner statement read at the dense form, a
piece keyed by group `p` at a copy `(o, g)` with `o + g = r` reads, through `resVRead`, group `p`'s `V` elements of
the `v` the verifier rows give at rep `r`'s values. -/
theorem resVRead_eq {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size) (r : Fin 2)
    {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf (D.sh cls i) = .ok S)
    (hg : groups S (portSizes (D.sh cls i)) = .ok gs) {p : ℕ} (hp : p < gs.size) {o g u : ℕ} (hog : o + g = r.val)
    (v : Array F128) (info : RecAlgebra.Info RecAlgebra.Lin) (kappa : Array (Array (Array F128)))
    (hv : verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
      (D.rep cls acc i r).comb = .ok (v, info, kappa)) (k : ℕ) (hk : k < gs[p].2.2.size) :
    resVRead x cs (p, o, g, u) k = el v gs[p].2.2[k] := by
  obtain ⟨hT1, hT2⟩ := D.tables_of_dense hst hf hx
  have hr2 := r.isLt
  obtain rfl : i = 0 := by omega
  have hnd := D.names_nodup hst hf hx
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hv₀, -, hc₁, -, -, -, hpf, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hsp' : D.spec = spec := by rw [hsp, hd]
  have hsts' : D.sts = sts := by rw [hsts, hd]
  have hdec := D.decode
  rw [hd, hsp', hsts'] at hdec
  simp only [jAt] at hdec
  have hpts := (FlockDeploy.decodeJWith_sub _ hdec).1
  have hhead : cs.headD default = c₁ := by
    cases cs with
    | nil => cases hc₁
    | cons c cs => exact Option.some.inj hc₁
  have hk2 : 2 * 0 + r.val < 2 * sts.size := by rw [← hsts']; omega
  have hti : 0 < spec.tables.size := by rw [← hsp']; omega
  rw [hsp'] at hnd
  obtain ⟨hks, e⟩ := Claims.Bridge.Custody.repStreams_getElem spec hti r.isLt
  have hrounds : (D.rep cls acc 0 r).rounds = canonRounds spec sts cs.tail.dropLast (2 * 0 + r.val) := by
    show ((D.files cls 0).stream r).rounds.map (·.coins) = _
    have hs : (D.files cls 0).stream r =
        (D.sess.streams.find? (·.name == spec.repStreams[2 * 0 + r.val].2.2)).getD default := by
      rw [e]
      show (D.sess.streams.find? fun y => y.name == toString D.spec.tables[0]! ++ toString "/rep" ++
        toString (r : ℕ)).getD default = _
      rw [hsp']
    rw [hs]
    exact coins_of_decode _ hnd (readPs_blocks hpf) hdec hk2 hks
  have hext : (D.rep cls acc 0 r).extras = extrasRead spec sts cs := by
    show ((tabsOf D.sts)[0]!.extra D.sess.points).toOption.getD #[] =
      ((tabsOf sts)[0]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]
    rw [hsts', hpts, hhead]
  have hsh : D.sh cls 0 = shapeRead spec sts cs := by
    show shapeOf (tabsOf D.sts)[0]! ((fast100 (tabsOf D.sts)[0]!.m).getD default)
      (((tabsOf D.sts)[0]!.extra D.sess.points).toOption.getD #[]).size =
      shapeOf (tabsOf sts)[0]! ((fast100 (tabsOf sts)[0]!.m).getD default)
        (((tabsOf sts)[0]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]).size
    rw [hsts', hpts, hhead]
  have hc : (D.rep cls acc 0 r).comb = combRead (D.files cls 0).st (D.sh cls 0) (D.rep cls acc 0 r).rounds
      (D.rep cls acc 0 r).extras := rfl
  have hst0 : (D.files cls 0).st = (tabsOf sts)[0]! := by
    show (tabsOf D.sts)[0]! = _
    rw [hsts']
  rw [hc, hst0, hsh, hrounds, hext] at hv
  rw [hsh] at hS hg
  unfold resVRead
  rw [hv₀]
  show resVAt spec sts cs (o + g) p k = _
  rw [hog, show r.val = 2 * 0 + r.val by omega]
  exact resVOf_eq hv hS hg hp hk

end DecAt

end

end RecDeploy.Rows

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open FlockVBridge.PartS (wIdx)
open Flock.RecAlgebra (verifierRows structureOf el)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt)

variable {resP : Msg → List ResPiece} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ}
  {elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128}
  {vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128}
  {side : Msg → List Msg → OpenPiece → OpenSide} {posAt : Msg → List Coin → OpenPiece → ℕ}

/-- **The residual pieces hold where `PD` does.** -/
theorem res_holds {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}
    (h : PD rd resP key elemAt vvAt side posAt x ms cs)
    {q : ResPiece} (hq : q ∈ resP x) : resHolds elemAt vvAt q x ms cs :=
  h (.res q) (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl (List.mem_map.2 ⟨q, hq, rfl⟩)))))

/-- **Row R's leaf from the pieces' cover**: `TargetRDep` at the assembly's pieces, where at every decision `At`
admits, each rep and group of S has a piece of `resP x` whose group is the verifier's part, whose elements at its key
are `w` at the group's row ports and whose verifier values at its key are `verifierRows`' `v` at the group's `V`. -/
theorem targetRDep_of_cover (rd : Msg → Option FinalP)
    (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128)
    (hacc : ∀ x ms cs k, (acc x ms cs k).size = 4)
    (hcov : ∀ (x : Msg) (ms : List Msg) (cs : List Coin),
      Top.At rd (PD rd resP key elemAt vvAt side posAt) x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size →
      ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
        structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
        ∀ (r : Fin 2) p (hp : p < gs.size), ∃ q ∈ resP x,
          (∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv)) ∧
          (∀ k (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size),
            elemAt x ms cs q.key k =
              el (D.rep cls (acc x ms cs) i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k]) ∧
          ∀ v info kappa, verifierRows (D.sh cls i) (D.rep cls (acc x ms cs) i r).rounds
              (D.rep cls (acc x ms cs) i r).extras (D.rep cls (acc x ms cs) i r).comb =
              .ok (v, info, kappa) →
            ∀ k (hk : k < gs[p].2.2.size), vvAt x cs q.key k = el v gs[p].2.2[k]) :
    Top.TargetRDep rd (PD rd resP key elemAt vvAt side posAt) acc := by
  classical
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, hP⟩ := id hA
  obtain ⟨S, gs, hS, hg⟩ := D.structure_ok cls hst hf hx hi D.points_eq
  have hc := hcov x ms cs hA D cls i hi S gs hS hg
  refine ⟨fun r => hacc x ms cs _, resP x, elemAt, vvAt,
    fun r p => if hp : p < gs.size then (Classical.choose (hc r p hp)).key else (0, 0, 0, 0),
    fun q hq => res_holds hP hq, ?_, ?_, ?_⟩
  · intro S' gs' hS' hg'
    rw [hS] at hS'
    cases hS'
    rw [hg] at hg'
    cases hg'
    intro r p hp
    obtain ⟨hq, hpart, -, -⟩ := Classical.choose_spec (hc r p hp)
    exact ⟨_, hq, by simp only [dif_pos hp], hpart⟩
  · intro S' gs' hS' hg'
    rw [hS] at hS'
    cases hS'
    rw [hg] at hg'
    cases hg'
    intro r p hp k hk
    simp only [dif_pos hp]
    exact (Classical.choose_spec (hc r p hp)).2.2.1 k hk
  · intro S' gs' hS' hg'
    rw [hS] at hS'
    cases hS'
    rw [hg] at hg'
    cases hg'
    intro r v info kappa hv p hp k hk
    simp only [dif_pos hp]
    exact (Classical.choose_spec (hc r p hp)).2.2.2 v info kappa hv k hk

/-- **The pieces' cover, but its verifier values** (`ElemCover`): at every decision `At` admits at `vvAt := resVRead`,
each rep `r` and group `p` of S has a piece of `resP x` of the verifier's part `p`, keyed by group `p` at a copy
`(o, g)` with `o + g = r`, whose elements at its key are the rep's `w` at the group's row ports, at the accumulators
`acc`. -/
def ElemCover (rd : Msg → Option FinalP) (resP : Msg → List ResPiece)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin),
    Top.At rd (PD rd resP key elemAt RecDeploy.Rows.resVRead side posAt) x ms cs →
    ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size →
    ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
      structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
      ∀ (r : Fin 2) p (_ : p < gs.size), ∃ q ∈ resP x,
        (∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv)) ∧
        q.key.1 = p ∧ q.key.2.1 + q.key.2.2.1 = r.val ∧
        ∀ k (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size),
          elemAt x ms cs q.key k =
            el (D.rep cls (acc x ms cs) i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k]

/-- **Row R's leaf at the coins' verifier values**: `TargetRDep` at `vvAt := resVRead`, from `ElemCover`. -/
theorem targetRDep_resV (rd : Msg → Option FinalP)
    (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128)
    (hacc : ∀ x ms cs k, (acc x ms cs k).size = 4)
    (hcov : ElemCover rd resP key elemAt side posAt acc) :
    Top.TargetRDep rd (PD rd resP key elemAt RecDeploy.Rows.resVRead side posAt) acc := by
  refine targetRDep_of_cover rd acc hacc fun x ms cs hA D cls i hi S gs hS hg r p hp => ?_
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -⟩ := id hA
  obtain ⟨q, hq, hpart, hk1, hog, hE⟩ := hcov x ms cs hA D cls i hi S gs hS hg r p hp
  refine ⟨q, hq, hpart, hE, fun v info kappa hv k hk => ?_⟩
  obtain ⟨⟨jj, o, g, u⟩, qS⟩ := q
  dsimp only at hk1 hog
  subst hk1
  exact D.resVRead_eq cls _ hst hf hx hi r hS hg hp hog v info kappa hv k hk

/-- **Row R's leaf in the assembly's form** (`DecompBDeployed`'s `hRdep`): `TargetRDep` at `vvAt := resVRead` and the
accumulators `Top.accRC rd key`, from `ElemCover` there. -/
theorem targetRDep_accRC (rd : Msg → Option FinalP)
    (hcov : ElemCover rd resP key elemAt side posAt (Top.accRC rd key)) :
    Top.TargetRDep rd (PD rd resP key elemAt RecDeploy.Rows.resVRead side posAt) (Top.accRC rd key) :=
  targetRDep_resV rd (Top.accRC rd key) (fun x ms cs k => accR_size rd _ x ms cs k) hcov

end RecDeploy.DecompBR
