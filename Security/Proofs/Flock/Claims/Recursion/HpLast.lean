import Proofs.Flock.Claims.Recursion.VStarBuild
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed

/-!
# hp's last round at the deployed builder (`hpLast`)

`honest_partAccepts_layR`'s `hp` asks, at each inner round `i`, that every piece `(o, m)` part `s` holds of round `i`
at its outer witness `wO` (`LayoutB.piecesOf`) be round `i`'s message's bytes at offset `o`. At the layout of record
`DeployedR.layR elemR` the last round's pieces are every earlier round's openings (`openPieces`), then the
element-level last message's (`OpenRows.elemR`), then the trailer (`OpenRows.lastPieces`).

* **`hp` from the assembly** (`hp_of_assembled`, any layout and round): where round `i`'s message is the round
  assembled off the outer witnesses `ws` (`LayoutB.assembled`), part `s`'s pieces at `ws[s]` sit in it at their
  offsets (`LayoutB.mem_piecesOf`, `LayoutB.assembled_piece`).
* **The honest run's outer witnesses** (`wOs`): part `s`'s is `vstarWit A I x w ρ hist s`. `finMsg` is the last round
  assembled off them, and `hpLast` is `hp` at the last round for `wO = vstarWit A I x w ρ hist s`, wherever the inner
  prover's last message is `finMsg`.
* **What `finMsg` is** (`finMsg_eq`, at a dense compiled statement): the openings read off the outer witnesses, then
  `encLast` of the statement and each rep's proof read off the `rec-open` copies (`OpenRows.proofR`), then the trailer
  at the openings' lengths (`DeployedR.trailer`, `openLens`).
* **An opening at a trace that places its port inputs** (`readSegs_openSegs`, `readSegs_vstarWit_open`): where the
  assigner's trace `A c pin` satisfies `Places` (the port lemma, a hypothesis here), the opening of port `nm` at
  instance `n` reads the stager's row there, its words' bits as bytes, and its salt (`portRow`).

`Places` covers the port rows and salts, not the `rec-open` copies' columns `elemR` reads, so `finMsg`'s `encLast` part
stays read off the trace.
-/

namespace RecDeploy.HpLast

open _root_.Flock
open Claims (Msg)
open Claims.Recursion.VStarB (LayoutB readSegs readForms formBit pieceLen)
open RecDeploy.VStarBuild

/-! ## `hp` from the assembly -/

/-- **A round assembled off the witnesses holds each of part `s`'s pieces at its offset**: `hp` at round `i`, at
`ws[s]`, wherever round `i`'s message is the round assembled off `ws`. -/
theorem hp_of_assembled {L : LayoutB} {x : Msg} {ws : List Claims.Flock.Wit} {s i : ℕ} (hs : s < ws.length)
    {ms : List Msg} (hm : ms.getD i [] = L.assembled x ws i) :
    ∀ o m, (o, m) ∈ L.piecesOf x s ws[s] i → ((ms.getD i []).drop o).take m.length = m := by
  intro o m h
  obtain ⟨pre, p, post, hl, hp, he⟩ := Claims.Recursion.VStarB.LayoutB.mem_piecesOf h
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  rw [hm]
  exact Claims.Recursion.VStarB.LayoutB.assembled_piece hs hl hp

/-! ## The honest run's last message -/

/-- **The honest run's outer witnesses**: part `s`'s is `vstarWit`'s, at every part of `x`. -/
def wOs (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit)
    (ρ : I.Tape) (hist : List (Msg × Msg)) : List Claims.Flock.Wit :=
  (List.range (nS x)).map (vstarWit A I x w ρ hist)

theorem wOs_length (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) : (wOs A I x w ρ hist).length = nS x := by
  simp [wOs]

/-- **The honest last message**: the last round (`kI x - 1`) at the layout of record `layR elemR`, assembled off the
outer witnesses. -/
def finMsg (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit)
    (ρ : I.Tape) (hist : List (Msg × Msg)) : Msg :=
  (DeployedR.layR OpenRows.elemR).assembled x (wOs A I x w ρ hist) (RestRead.kI x - 1)

/-- **hp's last round at the deployed builder**: where the inner prover's last message is `finMsg`, every piece part
`s` holds of the last round at its outer witness `vstarWit A I x w ρ hist s` is that message's bytes at its offset. -/
theorem hpLast {A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit} {I : InnerRec} {x : Msg}
    {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {s : ℕ} (hs : s < nS x) {ms : List Msg}
    (hfin : ms.getD (RestRead.kI x - 1) [] = finMsg A I x w ρ hist) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s (vstarWit A I x w ρ hist s) (RestRead.kI x - 1) →
      ((ms.getD (RestRead.kI x - 1) []).drop o).take m.length = m := by
  have hs' : s < (wOs A I x w ρ hist).length := by rw [wOs_length]; exact hs
  have h := hp_of_assembled (L := DeployedR.layR OpenRows.elemR) (x := x) (i := RestRead.kI x - 1) hs' hfin
  simp only [wOs, List.getElem_map, List.getElem_range] at h
  exact h

/-- **What the honest last message is**, at a dense compiled statement: the openings off the outer witnesses, then
`encLast` of the statement and each rep's proof off the `rec-open` copies, then the trailer. -/
theorem finMsg_eq {A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit} {I : InnerRec} {x : Msg}
    {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} (hx : RestRead.DenseX x) :
    ∃ r, readX x = some r ∧ 0 < r.w.parts.length ∧ finMsg A I x w ρ hist =
      DeployedR.readPieces (openPieces r) (wOs A I x w ρ hist) ++
        (DeployedR.encLast (OpenRows.stmtR r)
            ((List.range (OpenRows.nProofs r)).map (OpenRows.proofR r (wOs A I x w ρ hist))) ++
          DeployedR.trailer (DeployedR.openLens r)) := by
  obtain ⟨r, hr, -, -, hp⟩ := OpenRows.lastPieces OpenRows.elemR hx
  obtain ⟨N, -, hn0⟩ := capAt_readX hr
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have h0 : 0 < r.w.parts.length := hn ▸ hn0
  have hl : (wOs A I x w ρ hist).length = r.w.parts.length := by rw [wOs_length, hn]
  refine ⟨r, hr, h0, ?_⟩
  show DeployedR.readPieces ((RestRead.layAt (DeployedR.restR OpenRows.elemR)).pieceAt x (RestRead.kI x - 1))
    (wOs A I x w ρ hist) = _
  rw [hp, OpenRows.readPieces_append, DeployedR.restR, List.map_append, OpenRows.readPieces_append,
    OpenRows.readPieces_elemR r hl h0]
  congr 2
  unfold DeployedR.trailerPiece
  rw [dif_pos h0]
  simp only [List.map_cons, List.map_nil]
  rw [OpenRows.readPieces_one (by rw [hl]; omega), OpenRows.readSegs_const]

/-! ## The openings at a trace that places its port inputs -/

theorem bytesOf_append8 {a bs : List Bool} (ha : a.length = 8) :
    Claims.Recursion.VStar.bytesOf (a ++ bs) = Claims.Recursion.VStar.byte a :: Claims.Recursion.VStar.bytesOf bs := by
  unfold Claims.Recursion.VStar.bytesOf
  have hl : ((a ++ bs).length + 7) / 8 = (bs.length + 7) / 8 + 1 := by
    rw [List.length_append, ha]; omega
  rw [hl, List.range_succ_eq_map, List.map_cons, List.map_map]
  congr 1
  · rw [Nat.mul_zero, List.drop_zero, List.take_append_of_le_length (by omega), List.take_of_length_le (by omega)]
  · apply List.map_congr_left
    intro j _
    show Claims.Recursion.VStar.byte (((a ++ bs).drop (8 * (j + 1))).take 8) =
      Claims.Recursion.VStar.byte ((bs.drop (8 * j)).take 8)
    rw [List.drop_append_eq_append_drop, List.drop_eq_nil_of_le (by rw [ha]; omega), List.nil_append, ha,
      show 8 * (j + 1) - 8 = 8 * j by omega]

/-- A salt's bits read back as its bytes. -/
theorem bytesOf_saltBits (b : ByteArray) : Claims.Recursion.VStar.bytesOf (saltBits b) = b.data.toList := by
  unfold saltBits
  generalize b.data.toList = l
  induction l with
  | nil => simp [Claims.Recursion.VStar.bytesOf]
  | cons u l ih =>
    rw [List.flatMap_cons, bytesOf_append8 (by simp), ih, Claims.Recursion.VStar.byte_testBits]

/-- **An opening at a trace that places its port inputs**: port `nm`'s (index `q`) at instance `n` reads the stager's
row there, its words' bits at the port's row width as bytes, then its salt. -/
theorem readSegs_openSegs {c : Circuit} {pin : _root_.VStar.In.PartIn} {wO : Claims.Flock.Wit} (hP : Places c pin wO)
    {nm : String} {q n : ℕ} {rw : _root_.VStar.In.Row} {ps : List ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q)
    (hrw : portRow pin nm n = some rw) (hps : rowHidden c nm n = some ps) :
    readSegs wO (openSegs c nm q n) =
      Claims.Recursion.VStar.bytesOf ((_root_.VStar.In.wordBits rw.words).take ps.length) ++ rw.salt.data.toList := by
  have hr : readForms wO (ps.map HmNets.Lin.var) =
      Claims.Recursion.VStar.bytesOf ((_root_.VStar.In.wordBits rw.words).take ps.length) := by
    rw [Claims.Recursion.VStarB.readForms_pos]
    unfold Claims.Recursion.VStar.readAt
    rw [hP.row nm n rw ps hrw hps]
  have hsl : readForms wO (hmSalt c q n) = rw.salt.data.toList := by
    unfold Claims.Recursion.VStarB.readForms
    rw [hP.salt nm q n rw hq hrw, bytesOf_saltBits]
  unfold openSegs Claims.Recursion.VStarB.readSegs
  simp only [hps, Option.getD_some, List.flatMap_cons, List.flatMap_nil, List.nil_append, List.append_nil, hr, hsl]

/-- **An opening at the deployed builder's witness**: at part `s`, where the part, the stager's inputs and its port
inputs exist and the assigner's trace there places them (`Places`, the port lemma), port `nm`'s opening at instance `n`
reads the stager's row and salt there. -/
theorem readSegs_vstarWit_open {A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit} {I : InnerRec} {x : Msg}
    {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {s : ℕ} {st : _root_.VStar.Stmt} {c : Circuit}
    {src : _root_.VStar.In.Src} {pin : _root_.VStar.In.PartIn} (hp : partAt x s = some (st, c))
    (hsrc : srcOf I x w ρ hist = some src) (hi : _root_.VStar.In.partIn src (progOf c) st = .ok pin)
    (hP : Places c pin (A c pin)) {nm : String} {q n : ℕ} {rw : _root_.VStar.In.Row} {ps : List ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hrw : portRow pin nm n = some rw)
    (hps : rowHidden c nm n = some ps) :
    readSegs (vstarWit A I x w ρ hist s) (openSegs c nm q n) =
      Claims.Recursion.VStar.bytesOf ((_root_.VStar.In.wordBits rw.words).take ps.length) ++ rw.salt.data.toList := by
  rw [vstarWit_eq hp hsrc hi]
  exact readSegs_openSegs hP hq hrw hps

end RecDeploy.HpLast
