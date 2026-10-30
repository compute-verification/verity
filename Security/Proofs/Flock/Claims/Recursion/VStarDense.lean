import Proofs.Flock.Claims.Recursion.VStarAccept
import Proofs.Flock.Claims.Recursion.VStarFitHeld
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.C3.Reps.AtList

/-!
# V\*'s parts at the dense form are levels, algebra parts and caps (`partAt_dense`)

At a statement the reader reads at the dense form (`Form.Dense`: no sparse phase), the verifier's own inner shape names
no RecSparse shape, so candidate 3's list is empty (`candidate3_ok`: none without a `sparse` entry, and R without one is
refused), and V\*'s list is its levels (`.recOpen`), its algebra parts (`.inner`) and its caps (`.cap`)
(`getElem?_list`). No part is `.sparse`, `.recOpenStacked` or `.opening`, the statements `partIn` has no builder for.

`hok_dense` is `hok_of_fits` with its premise split by those three kinds: a level's checks (`LevelInFits`), an algebra
part's (`AlgInFits`) and a cap's (`capIn` accepting).

`IrShaped Ir x w ρ hist` is what the stager reads of the inner prover's `rest` at one run: both reps have an opening, and
at each level part `.recOpen l L H n` the level's openings number `n`, each of `L` lanes and `H` siblings, and each one's
top node is among the level's tops wherever they are read. No Lean `InnerRec` is the deployed inner prover (its `rest` is
`rec_vstage`'s, `rec_vstar`'s and `verifier_rows`' output), so this is witnessed outside Lean, at the deployed builder's
run, as `HonestRunAccepted` is. `hok_shaped` is the stager accepting every part of a dense statement from it, given the
level tops being read (`topsAt`), each algebra port being `acc0`, `acc1` or a schedule's `m<i>`, and each cap's `capIn`.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-- **A dense statement's inner shape has no sparse phase**: the form's `sparse` is the verifier's own inner shape's. -/
theorem sparse_none_of_dense {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) : ∃ r, readX x = some r ∧ r.shape.sparse = none := by
  obtain ⟨r, hr, -, rfl⟩ := parse_read hx
  refine ⟨r, hr, ?_⟩
  have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hf.2
  cases h : r.shape.sparse with
  | none => rfl
  | some sp => rw [h] at h2; cases h2

/-- **Every part of a dense statement is a level, an algebra part or a cap**: candidate 3's list is empty without a
sparse phase, so V\*'s list is its levels, its algebra parts and its caps. -/
theorem partAt_dense {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) {s : ℕ} {sv : _root_.VStar.Stmt} {c : Circuit}
    (hp : partAt x s = some (sv, c)) :
    (∃ l L H n, sv = .recOpen l L H n) ∨ (∃ j nm p, sv = .inner j nm p) ∨ ∃ l N, sv = .cap l N := by
  obtain ⟨r, hr, hsp⟩ := sparse_none_of_dense hx hf
  obtain ⟨ss, hss, -⟩ := readX_parts hr
  have hsv : ss[s]? = some sv := by
    have h := (partAt_some hr hp).1
    rw [hss] at h
    exact h
  obtain ⟨sch, ps, C, Ks, -, hC, hK, hget⟩ := FlockC3.Reps.getElem?_list hss
  rw [hget s, (FlockC3.VStarList.candidate3_ok hC).1 hsp] at hsv
  split_ifs at hsv
  · exact .inl ⟨_, _, _, _, (Option.some.inj hsv).symm⟩
  · exact .inr (.inl ⟨_, _, _, (Option.some.inj hsv).symm⟩)
  · rw [Array.empty_append] at hsv
    exact .inr (.inr (hK _ (Array.mem_iff_getElem?.2 ⟨_, hsv⟩)))

/-- **The stager accepts at every part of a dense statement** where it accepts at each of its kinds: a level's checks
hold (`LevelInFits`), an algebra part's (`AlgInFits`), and `capIn` accepts at a cap. -/
theorem hok_dense {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)}
    {st : Claims.Recursion.CompiledStatement} {s : ℕ} (hx : parse x = some st) (hf : Claims.Recursion.Form.Dense st.form)
    (hL : ∀ l L H n c src, partAt x s = some (.recOpen l L H n, c) → srcOf I x w ρ hist = some src →
      LevelInFits src l L H n)
    (hA : ∀ (j : ℕ) (nm : String) (p : _root_.VStar.Part) c src, partAt x s = some (.inner j nm p, c) → srcOf I x w ρ hist = some src →
      AlgInFits src p.S.ports)
    (hC : ∀ l N c src, partAt x s = some (.cap l N, c) → srcOf I x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.capIn src l N = .ok pin) :
    ∀ sv c src, partAt x s = some (sv, c) → srcOf I x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin := by
  refine hok_of_fits fun sv c src hp hs => ?_
  rcases partAt_dense hx hf hp with ⟨l, L, H, n, rfl⟩ | ⟨j, nm, p, rfl⟩ | ⟨l, N, rfl⟩
  · exact hL l L H n c src hp hs
  · exact hA j nm p c src hp hs
  · exact hC l N c src hp hs

/-! ## The inner prover's openings at a run (`IrShaped`) -/

/-- **The inner prover's openings fit V\*'s levels at a run**: in `rest` at the inner witness `w`, its tape `ρ` and the
inner rounds `hist`, both reps have an opening, and at each level part `.recOpen l L H n` of `x`, level `l`'s openings
number `n`, each of `L` lanes and `H` siblings, and where the level's tops are read at the stager's inputs (`srcOf`,
`topsAt`), each opening's top node is among them. `(Ir.fit x).rest` is `Ir.rest`. Witnessed outside Lean, at the
deployed builder's run. -/
def IrShaped (Ir : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : (Ir.fit x).Tape) (hist : List (Msg × Msg)) :
    Prop :=
  (∃ o ∈ ((Ir.fit x).rest w ρ.1 hist).openings, o.rep = 0) ∧
  (∃ o ∈ ((Ir.fit x).rest w ρ.1 hist).openings, o.rep = 1) ∧
  ∀ s l L H n c, partAt x s = some (.recOpen l L H n, c) →
    (((Ir.fit x).rest w ρ.1 hist).openings.filter (·.level == l)).size = n ∧
    (∀ q ∈ ((Ir.fit x).rest w ρ.1 hist).openings, q.level = l → q.row.size = L ∧ q.sibs.size = H) ∧
    ∀ src, srcOf (Ir.fit x) x w ρ hist = some src → ∀ T, _root_.VStar.In.topsAt src l = .ok T →
      ∀ q ∈ ((Ir.fit x).rest w ρ.1 hist).openings, q.level = l → T.first[q.rep]! + q.pos / 2 ^ H < T.rows.size

/-- The stager's openings are the inner prover's `rest`'s. -/
theorem srcOf_openings {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)}
    {src : _root_.VStar.In.Src} (hs : srcOf I x w ρ hist = some src) :
    src.openings = (I.rest w ρ.1 hist).openings := by
  cases hr : readX x with
  | none => simp [srcOf, hr] at hs
  | some r =>
    cases hrs : repSched r.shape with
    | none => simp [srcOf, hr, hrs] at hs
    | some rs => rw [srcOf_eq hr hrs hs]

/-- **A level part fits** at the stager's inputs where the inner prover's openings fit (`IrShaped`) and the level's tops
are read. -/
theorem levelFits_of_shaped {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape}
    {hist : List (Msg × Msg)} (hI : IrShaped Ir x w ρ hist) {s l L H n : ℕ} {c : Circuit} {src : _root_.VStar.In.Src}
    (hp : partAt x s = some (.recOpen l L H n, c)) (hs : srcOf (Ir.fit x) x w ρ hist = some src)
    (hT : ∃ T, _root_.VStar.In.topsAt src l = .ok T) : LevelInFits src l L H n := by
  obtain ⟨hn, hsz, hpos⟩ := hI.2.2 s l L H n c hp
  have ho := srcOf_openings hs
  obtain ⟨T, hT⟩ := hT
  refine ⟨?_, fun q hq => ?_, T, hT, fun q hq => ?_⟩
  · rw [ho]; exact hn
  · rw [ho] at hq; exact hsz q hq
  · rw [ho] at hq; exact hpos src hs T hT q hq

/-- **An algebra part fits** at the stager's inputs where both reps have an opening (`IrShaped`) and each port is
`acc0`, `acc1` or an `m<i>` at one of the schedule's message rounds. -/
theorem algFits_of_shaped {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape}
    {hist : List (Msg × Msg)} (hI : IrShaped Ir x w ρ hist) {src : _root_.VStar.In.Src}
    (hs : srcOf (Ir.fit x) x w ρ hist = some src) {ports : List (String × ℕ)}
    (hN : ∀ xr rs, readX x = some xr → repSched xr.shape = some rs → ∀ nm ∈ ports.map Prod.fst,
      nm = "acc0" ∨ nm = "acc1" ∨ ∃ i, RecVStar.mIndex nm = some i ∧ i < (RecVStar.commitRounds rs).length) :
    AlgInFits src ports := by
  cases hr : readX x with
  | none => simp [srcOf, hr] at hs
  | some r =>
    cases hrs : repSched r.shape with
    | none => simp [srcOf, hr, hrs] at hs
    | some rs => exact algFits_srcOf hr hrs hs hI.1 hI.2.1 (hN r rs hr hrs)

/-- **The stager accepts every part of a dense statement** at a run where the inner prover's openings fit
(`IrShaped`), each level's tops are read (`topsAt`), each algebra part's port is `acc0`, `acc1` or a schedule's `m<i>`,
and each cap's `capIn` accepts. -/
theorem hok_shaped {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape} {hist : List (Msg × Msg)}
    {st : Claims.Recursion.CompiledStatement} {s : ℕ} (hx : parse x = some st) (hf : Claims.Recursion.Form.Dense st.form)
    (hI : IrShaped Ir x w ρ hist)
    (hT : ∀ l L H n c src, partAt x s = some (.recOpen l L H n, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ T, _root_.VStar.In.topsAt src l = .ok T)
    (hN : ∀ (j : ℕ) (nm : String) (p : _root_.VStar.Part) c, partAt x s = some (.inner j nm p, c) → ∀ xr rs, readX x = some xr →
      repSched xr.shape = some rs → ∀ pn ∈ p.S.ports.map Prod.fst,
        pn = "acc0" ∨ pn = "acc1" ∨ ∃ i, RecVStar.mIndex pn = some i ∧ i < (RecVStar.commitRounds rs).length)
    (hC : ∀ l N c src, partAt x s = some (.cap l N, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.capIn src l N = .ok pin) :
    ∀ sv c src, partAt x s = some (sv, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin :=
  hok_dense hx hf (fun l L H n c src hp hs => levelFits_of_shaped hI hp hs (hT l L H n c src hp hs))
    (fun j nm p c _ hp hs => algFits_of_shaped hI hs (hN j nm p c hp)) hC

end RecDeploy.VStarBuild
