import Proofs.Flock.Claims.Recursion.VStarDeployed

/-!
# hp before the last round at the deployed builder (`hpEarly`)

`honest_partAccepts_layR`'s `hp` at an inner round `i < kI x - 1`, at the outer witness `witFin (VStarTrace.A mask)`
(the trace assigner's, at the history the last message reads). Round `i`'s pieces at `layR elemR` are hm96 slots'
outputs: the tops at their caps' port `top`, then the absorbed row at its holder's port (`topsPiecesAt`), each a
single segment `hmCommit c q n`. The inner prover's round `i` message is its rows' commit strings in order
(`InnerRec.P`, `commitThen`).

* **What a slot reads at the trace** (`readSegs_commit`): where `TraceOk` holds and the trace's hm96 slots commit
  (`SlotsOk`, `SlotCommits`), port `nm`'s slot at instance `n` reads the commit string of the stager's row there under
  its salt (`rowCommit`): the row at the slot's inputs and the salt at its salt inputs are `placesR_assign`'s, all of
  its words' bits (`PlacesR.take_full`), under the port's `sha512/row/v1` prefix (`TraceOk.rows`).
* **The fitted rows** (`RowsFit`, HONESTIV's): round `i`'s `k`-th piece is the hm96 slot of a port of part `p.1`'s
  circuit whose row at that instance is round `i`'s `k`-th row (`roundRows`), as many pieces as rows.
* **`hpEarly`**: where the inner prover's round `i` message is its rows' commit strings, the round assembled off the
  outer witnesses is that message (`flatMap_fit`), so `hp_of_assembled` gives `hp` there.
-/

namespace RecDeploy.HpEarly

open _root_.Flock
open Claims (Msg)
open Claims.Recursion.VStarB (LayoutB readSegs readForms pieceLen)
open RecDeploy.VStarBuild
open RecDeploy.HpLast

/-- **A row's commit string** `b ‖ c` under its salt (`Binding.commitString`), as the inner prover sends it. -/
def rowCommit (rw : _root_.VStar.In.Row) : Msg :=
  FlockSoundness.Binding.commitString (rowMsg rw.words) rw.salt.data.toList

/-- **The trace's hm96 slots commit**: at each port `nm` (index `q`) and instance `n` with a row, the slot's output is
the commit string of its row and salt off the trace (`SlotCommits`). -/
def SlotsOk (mask : ℕ → ℕ → Bool) (c : Circuit) (pin : _root_.VStar.In.PartIn) : Prop :=
  ∀ nm q n rw, c.ports.findIdx? (·.name == nm) = some q → portRow pin nm n = some rw →
    SlotCommits c nm q n (VStarTrace.A mask c pin)

/-- **The trace at every part**: at each part with a circuit and port inputs at the history `h`, `TraceOk` and
`SlotsOk`. -/
def TraceAt (mask : ℕ → ℕ → Bool) (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : I.Tape)
    (h : List (Msg × Msg)) : Prop :=
  ∀ s st c src pin, partAt x s = some (st, c) → srcOf I x w ρ h = some src →
    _root_.VStar.In.partIn src (progOf c) st = .ok pin → VStarTrace.TraceOk c pin ∧ SlotsOk mask c pin

/-- **The fitted rows at round `i`** (HONESTIV's): at the history `h`, round `i`'s pieces at `layR elemR` are as many
as its rows (`roundRows`), and the `k`-th is the hm96 slot (`hmCommit c q n`) of port `nm` (index `q`) of its part's
circuit, at an instance `n` whose row there, in the part's port inputs, is round `i`'s `k`-th row. -/
def RowsFit (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : I.Tape) (h : List (Msg × Msg)) (i : ℕ) : Prop :=
  ((DeployedR.layR OpenRows.elemR).pieceAt x i).length = (roundRows I w ρ h i).length ∧
    ∀ k (hk : k < ((DeployedR.layR OpenRows.elemR).pieceAt x i).length),
      ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn)
        (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
        partAt x ((DeployedR.layR OpenRows.elemR).pieceAt x i)[k].1 = some (st, c) ∧
        srcOf I x w ρ h = some src ∧ _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
        ((DeployedR.layR OpenRows.elemR).pieceAt x i)[k].2 = [(([] : Msg), hmCommit c q n)] ∧
        c.ports.findIdx? (·.name == nm) = some q ∧ portRow pin nm n = some rw ∧ rowHidden c nm n = some ps ∧
        (roundRows I w ρ h i)[k]? = some rw

/-- **A slot at the trace reads its row's commit string**: port `nm`'s hm96 slot at instance `n`, off the trace
assigner's witness, is the commit string of the stager's row there under its salt. -/
theorem readSegs_commit {mask : ℕ → ℕ → Bool} {c : Circuit} {pin : _root_.VStar.In.PartIn}
    (h : VStarTrace.TraceOk c pin) (hS : SlotsOk mask c pin) {nm : String} {q n : ℕ} {rw : _root_.VStar.In.Row}
    {ps : List ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) (hrw : portRow pin nm n = some rw)
    (hps : rowHidden c nm n = some ps) :
    readSegs (VStarTrace.A mask c pin) [(([] : Msg), hmCommit c q n)] = rowCommit rw := by
  have hP := VStarTrace.placesR_assign h mask
  have hc := hS nm q n rw hq hrw
  unfold SlotCommits at hc
  have hr : readForms (VStarTrace.A mask c pin) (ps.map HmNets.Lin.var) =
      Claims.Recursion.VStar.bytesOf (_root_.VStar.In.wordBits rw.words) := by
    rw [Claims.Recursion.VStarB.readForms_pos]
    unfold Claims.Recursion.VStar.readAt
    rw [hP.places.row nm n rw ps hrw hps, hP.take_full hrw hps]
  have hsl : readForms (VStarTrace.A mask c pin) (hmSalt c q n) = rw.salt.data.toList := by
    unfold Claims.Recursion.VStarB.readForms
    rw [hP.places.salt nm q n rw hq hrw, bytesOf_saltBits]
  obtain ⟨hqs, hname, -⟩ := VStarTrace.name_of_findIdx hq
  obtain ⟨hws, -⟩ := h.rows q hqs n rw (hname ▸ hrw)
  have hpfx : rowPfx c q = (HmRow.rowPrefix rw.words.size).data.toList := by
    simp [rowPfx, Array.getElem?_eq_getElem hqs, hws]
  rw [hps, Option.getD_some, hr, hsl, hpfx] at hc
  unfold Claims.Recursion.VStarB.readSegs
  simp only [List.flatMap_cons, List.flatMap_nil, List.nil_append, List.append_nil]
  rw [hc]
  rfl

theorem flatMap_fit {α β γ : Type} (f : α → List γ) (g : β → List γ) :
    ∀ (as : List α) (bs : List β), as.length = bs.length →
      (∀ k (ha : k < as.length) (hb : k < bs.length), f as[k] = g bs[k]) → as.flatMap f = bs.flatMap g
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl
  | a :: as, b :: bs, hl, h => by
    have h0 := h 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    rw [List.flatMap_cons, List.flatMap_cons, h0, flatMap_fit f g as bs (by simpa using hl) fun k ha hb => by
      have e := h (k + 1) (by simpa using ha) (by simpa using hb)
      simpa using e]

theorem roundRows_take {I : InnerRec} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {i k : ℕ}
    (hi : i ≤ k) : roundRows I w ρ (hist.take k) i = roundRows I w ρ hist i := by
  unfold roundRows
  first
  | rw [List.take_take, Nat.min_eq_left hi]
  | rw [List.take_take, Nat.min_eq_right hi]

theorem lt_nS_of_partAt {x : Msg} {s : ℕ} {st : _root_.VStar.Stmt} {c : Circuit}
    (hp : partAt x s = some (st, c)) : s < nS x := by
  cases hr : readX x with
  | none => simp [partAt, hr] at hp
  | some r =>
    have hn : nS x = r.w.parts.length := by simp [nS, hr]
    rw [hn]
    by_contra hge
    have hnone : r.w.parts[s]? = none := by simpa using (by omega : r.w.parts.length ≤ s)
    unfold partAt at hp
    rw [hr] at hp
    cases hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) with
    | none => simp [hst] at hp
    | some st' => simp [hst, hnone] at hp

/-- **hp before the last round at the deployed builder**: at an inner round `i < kI x - 1` whose rows fit the layout
(`RowsFit`), where the trace holds at every part (`TraceAt`) and the inner prover's round `i` message is its rows'
commit strings, every piece part `s` holds of round `i` at its outer witness `witFin (VStarTrace.A mask) I x w ρ hist s`
is that message's bytes at its offset. -/
theorem hpEarly {mask : ℕ → ℕ → Bool} {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape}
    {hist : List (Msg × Msg)} {s i : ℕ} (hs : s < nS x) (hi : i < RestRead.kI x - 1) {ms : List Msg}
    (hfit : RowsFit I x w ρ (hist.take (RestRead.kI x - 1)) i)
    (hT : TraceAt mask I x w ρ (hist.take (RestRead.kI x - 1)))
    (hmsg : ms.getD i [] = (roundRows I w ρ hist i).flatMap rowCommit) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s (witFin (VStarTrace.A mask) I x w ρ hist s) i →
      ((ms.getD i []).drop o).take m.length = m := by
  have hws := wOs_length (VStarTrace.A mask) I x w ρ (hist.take (RestRead.kI x - 1))
  have hs' : s < (wOs (VStarTrace.A mask) I x w ρ (hist.take (RestRead.kI x - 1))).length := by
    rw [hws]; exact hs
  have hm : ms.getD i [] = (DeployedR.layR OpenRows.elemR).assembled x
      (wOs (VStarTrace.A mask) I x w ρ (hist.take (RestRead.kI x - 1))) i := by
    rw [hmsg, ← roundRows_take (k := RestRead.kI x - 1) hi.le]
    unfold Claims.Recursion.VStarB.LayoutB.assembled
    obtain ⟨hl, hk⟩ := hfit
    generalize hP : (DeployedR.layR OpenRows.elemR).pieceAt x i = P at hl hk ⊢
    symm
    refine flatMap_fit _ _ _ _ hl fun k ha hb => ?_
    obtain ⟨st, c, src, pin, nm, q, n, ps, rw, hp, hsrc, hi', hseg, hq, hrw, hps, hrk⟩ := hk k ha
    have hlt := lt_nS_of_partAt hp
    have hw : (wOs (VStarTrace.A mask) I x w ρ (hist.take (RestRead.kI x - 1)))[P[k].1]? =
        some (vstarWit (VStarTrace.A mask) I x w ρ (hist.take (RestRead.kI x - 1)) P[k].1) := by
      simp [wOs, List.getElem?_range hlt]
    dsimp only
    rw [hw]
    first
    | rw [Option.map_some', Option.getD_some]
    | rw [Option.map_some, Option.getD_some]
    | simp only [Option.map, Option.getD]
    rw [vstarWit_eq hp hsrc hi', hseg]
    obtain ⟨hT1, hT2⟩ := hT _ st c src pin hp hsrc hi'
    rw [readSegs_commit hT1 hT2 hq hrw hps]
    rw [List.getElem?_eq_getElem hb] at hrk
    rw [Option.some.inj hrk]
  have h := hp_of_assembled (L := DeployedR.layR OpenRows.elemR) (x := x) (i := i) hs' hm
  simp only [wOs, List.getElem_map, List.getElem_range] at h
  exact h

end RecDeploy.HpEarly
