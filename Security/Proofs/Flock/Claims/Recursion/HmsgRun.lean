import Proofs.Flock.Claims.Recursion.HonestRun
import Proofs.Flock.Claims.Recursion.VStarDeployed
import Proofs.Flock.Claims.Recursion.ResLowerDeployed

/-!
# The honest run's inner rounds before the last are their rows' commit strings (`hmsg_run`)

`hpEarly`'s `hmsg` on the compiled verifier's run of the reference prover whose pieces are `refPartsFin`: at an inner
round `i` before the last, the run's message is round `i`'s rows' commit strings in order (`roundRows`, each
`Binding.commitString` of the row's `sha512/row/v1` message under its salt, `HpEarly.rowCommit`).

* **The run** (`run_msg_inner`): round `i`'s message is the inner prover's after the run's first `i` inner rounds.
* **The inner prover before the last round** (`PFin_before`) is `InnerRec.P`, whose message is `commitThen`'s leaves
  (`commitThen_flatten`): each row's digest masked under its salt, then the salt's hash, which is the row's commit
  string (`leaf_rowCommit`, through the executable's `Hm96.Default512.commitString`, `ResLower.commitString_exec`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg Out Prover run)
open FlockDeploy (Coin)

/-- **`commitThen` at the concatenation**: the commit strings so far, then each digest masked under its salt and the
salt's hash. -/
theorem commitThen_flatten (H : Msg → Msg) :
    ∀ (ps : List ((Fin 64 → UInt8) × Salt)) (cs : List Msg),
      ((commitThen (fun cs => .ret cs.flatten) ps cs).eval H).1 =
        (cs ++ ps.map fun p => List.ofFn (fun i => p.1 i ^^^ maskOf p.2 i) ++
          H (Claims.toMsg _root_.Flock.Hm96.Default512.saltPrefix ++ List.ofFn p.2)).flatten
  | [], cs => by simp [commitThen, Claims.OC.eval]
  | (d, y) :: ps, cs => by
      simp only [commitThen, Claims.OC.eval]
      rw [commitThen_flatten H ps]
      simp [List.append_assoc]

/-- **A row's leaf is its commit string**: its digest masked under the salt, then the salt's hash, at SHA-512, is
`Binding.commitString` of the row's message under the salt's bytes. -/
theorem leaf_rowCommit (ws : Array ℕ) (y : Salt) :
    List.ofFn (fun i => digestOf ws i ^^^ maskOf y i) ++
        FlockDeploy.sha512 (Claims.toMsg _root_.Flock.Hm96.Default512.saltPrefix ++ List.ofFn y) =
      FlockSoundness.Binding.commitString (rowMsg ws) (saltBytes y).data.toList := by
  rw [← ResLower.commitString_exec, _root_.Flock.Hm96.Default512.commitString]
  simp only [ByteArray.data_append, Array.toList_append]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.ext_getElem (by simp)
    intro n h1 h2
    simp only [List.getElem_ofFn, digestOf, maskOf, saltBytes, FlockDeploy.sha512, Claims.toMsg, Claims.ofMsg,
      Array.toList_map, Array.toList_range, List.getElem_map, List.getElem_range,
      FlockSoundness.Discharge.Hm.get!_eq_getD]
  · have hs : Claims.toMsg _root_.Flock.Hm96.Default512.saltPrefix ++ List.ofFn y =
        Claims.toMsg (_root_.Flock.Hm96.Default512.saltPrefix ++ saltBytes y) := by
      rw [toMsg_append]
      simp [Claims.toMsg, saltBytes]
    rw [hs]
    simp only [FlockDeploy.sha512, Claims.ofMsg_toMsg]
    rfl

/-- **The inner prover's message is its round's rows' commit strings**, in order, each row under its round's and
row's salt. -/
theorem P_rowCommits (I : InnerRec) (w : Claims.Flock.Wit) (xI : Msg) (h : List (Msg × Msg)) (ρ : I.Tape) :
    (((I.P w).next xI h ρ).eval FlockDeploy.sha512).1 =
      ((I.rows w ρ.1 h).mapIdx fun j ws => (⟨ws, saltBytes (ρ.2 h.length j)⟩ : _root_.VStar.In.Row)).flatMap
        fun rw => FlockSoundness.Binding.commitString (rowMsg rw.words) rw.salt.data.toList := by
  rw [show (I.P w).next xI h ρ = commitThen (fun cs => .ret cs.flatten)
      ((I.rows w ρ.1 h).mapIdx fun j ws => (digestOf ws, ρ.2 h.length j)) [] from rfl,
    commitThen_flatten, List.nil_append, List.flatMap_def]
  congr 1
  apply List.ext_getElem (by simp)
  intro n h1 h2
  simp only [List.getElem_map, List.getElem_mapIdx]
  exact leaf_rowCommit _ _

/-- **The deployed inner prover before the last round sends its round's rows' commit strings**: at the inner rounds
`hist`'s first `i`, with `i` before the last round. -/
theorem PFin_rowCommits (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (xI : Msg) (hist : List (Msg × Msg)) (ρ : I.Tape) {i : ℕ} (hl : (hist.take i).length = i)
    (hi : i < RestRead.kI x - 1) :
    (((PFin A I x w).next xI (hist.take i) ρ).eval FlockDeploy.sha512).1 =
      (roundRows I w ρ hist i).flatMap fun rw =>
        FlockSoundness.Binding.commitString (rowMsg rw.words) rw.salt.data.toList := by
  rw [PFin_before A I x w xI (hist.take i) ρ (by rw [hl]; omega), P_rowCommits, hl]
  rfl

/-- **The deployed record's pieces**: `b` with the reference prover's pieces `refPartsFin`. -/
abbrev finB (b : Brings) (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg) :
    Brings :=
  { b with ref := refPartsFin A I x }

/-- **`hpEarly`'s `hmsg` on the run**: on the compiled verifier's run of `refP` at `finB` (the pieces `refPartsFin`)
at a statement that reads, at an inner round `i` before the last, the run's message is round `i`'s rows' commit strings
(`roundRows` at the run's inner rounds, `HpEarly.rowCommit`). -/
theorem hmsg_run (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec)
    (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) {x : Msg} {xr : XR} (hx : readX x = some xr)
    (hk : IV.k (xIOf x) = RestRead.kI x) (ω : Tape (finB b A I x)) (r : Fin ((V IV (finB b A I x).st).k x) → Coin)
    {i : ℕ} (hi : i < RestRead.kI x - 1) :
    ((run FlockDeploy.sha512 (refP IV (finB b A I x) w) (V IV (finB b A I x).st) x ω r).msgs.map Prod.fst).getD i
        [] =
      (roundRows I w ω.1 (((run FlockDeploy.sha512 (refP IV (finB b A I x) w) (V IV (finB b A I x).st) x ω
        r).msgs.take (IV.k (xIOf x))).map histOf) i).flatMap fun rw =>
          FlockSoundness.Binding.commitString (rowMsg rw.words) rw.salt.data.toList := by
  have hlen := run_length_ge FlockDeploy.sha512 IV (finB b A I x).st hx (refP IV (finB b A I x) w) ω r
  rw [run_msg_inner FlockDeploy.sha512 IV (finB b A I x) w hx ω r i (by omega)]
  have hl : ((((run FlockDeploy.sha512 (refP IV (finB b A I x) w) (V IV (finB b A I x).st) x ω r).msgs.take
      (IV.k (xIOf x))).map histOf).take i).length = i := by
    simp only [List.length_take, List.length_map]
    omega
  exact PFin_rowCommits A I x w _ _ _ hl hi

end RecDeploy.VStarBuild
