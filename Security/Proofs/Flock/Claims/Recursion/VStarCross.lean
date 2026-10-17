import Proofs.Flock.Claims.Recursion.DeployReg
import Proofs.Flock.Claims.Recursion.DeployScopeRuns

/-!
# V*'s cross-part facts at `deployRec`

`VStarParts` (`VStar.lean`) asks, besides each part's own facts, for four facts across the parts: the staged V*
statements show the run (`Staged`), the parts' checks are the inner verifier (`Decomp`), its calls are theirs
(`LogDecomp`), and some part holds each inner round (`Held`). This file proves what of them holds at `deployRec` with no
appeal to V*'s circuits, at every shape.

* **The staged statements read back** (`staged_ne_nil`): on an accepted run every V* session ends on a decision
  (`run_decides`), which the outer verifier makes only at a statement that sets up, so every part's staged statement is
  its encoding (`staged_decode`), whose transcript is the coin server's of the run's inner rounds (`Stage.ofMsg`).
* **`RunPairs` at `deployRec`** (`runPairs_deployRec`, no hypothesis): an accepted run sends every inner and staging
  round and at least three more (`roundsMargin_V`), each inner and staging round answered by its coin's bytes
  (`run_inner`), which `FlockDeploy.decCoin` reads back.
* **`Staged` at `deployRec`** (`staged_deployRec`), at the coins and commit strings a V* statement's transcript shows
  read back through the coin server's transcript (`showsTr`, `cmtTr`): wherever that transcript determines the rounds
  (`b.tr x` injective) and the layout spans no more than the inner rounds. `VstarShows` does not hold at `deployRec`:
  a V* statement staged at rounds where its numbers aren't the part's is the one staged at `pick`, which shows `pick`'s
  coins.
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay)
open FlockDeploy (Coin)

/-! ## The staged statements on an accepted run -/

section Staging

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

/-- **Every part's staged statement reads back on an accepted run**: session `s` ends on a decision (`run_decides`)
at its V* statement, the one staged at the run's inner and staging rounds (`ract_sess_locate`, `innerPairs_run`,
`run_round`), and the outer verifier decides only at a statement that sets up (`actO_decide`), which is no empty
one. -/
theorem staged_ne_nil (hacc : (run H P (V IV st) x ω r).out = .acc) {s : ℕ} (hs : s < nS x) :
    staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s ≠ [] := by
  obtain ⟨i, f, hi, ⟨j, xO, as', cs', ins, spec, sts, link, c₁, mids, hr, hO⟩, -⟩ :=
    run_decides H IV st x P ω r hacc hs
  obtain ⟨hk, -, hxO⟩ := ract_sess_locate hr
  have hK := run_acc_length H IV st x P ω r hacc
  have hlen : (((run H P (V IV st) x ω r).msgs.take (i + 1)).map Prod.fst).length = i + 1 := by
    simp only [List.length_map, List.length_take]
    omega
  rw [hlen] at hk
  have hip := innerPairs_run H IV st x P ω r hacc (off := i + 1) hk.le (by omega) []
    (range_map_take (coinSeq r) hk.le)
  rw [List.append_nil] at hip
  rw [hip] at hxO
  obtain ⟨i', hi', -⟩ := actO_decide hO
  obtain ⟨hins, -⟩ := setupO_some hi'
  obtain ⟨hnum, -⟩ := run_round H IV st x P ω r hacc hs rfl rfl
  subst hxO
  rw [vstar_getD st x _ hs] at hins
  unfold vstarAt at hins
  rw [ite_eq_left hnum] at hins
  intro he
  rw [he, insO_nil] at hins
  cases hins

end Staging

/-! ## The run's pairs -/

/-- **`RunPairs` at `deployRec`**, with no hypothesis: an accepted run sends the inner and staging rounds and three
more (`roundsMargin_V`), and answers each inner and staging round with its coin's bytes (`run_inner`), which
`FlockDeploy.decCoin` reads back (`coin_bytes_injective`). -/
theorem runPairs_deployRec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) : Claims.Recursion.VStar.RunPairs H (deployRec I.V ivScope b) I FlockDeploy.decCoin := by
  intro x Ω P ω r hacc
  change (run H P (V I.V b.st) x ω r).out = .acc at hacc
  obtain ⟨xr, hx⟩ := run_acc_readX H I.V b.st x P ω r hacc
  have hK := run_acc_length H I.V b.st x P ω r hacc
  have hm := roundsMargin_V H I.V b.st x P ω r hacc
  refine ⟨?_, fun i hi => ?_⟩
  · show I.V.k (xIOf x) + kS x ≤ (run H P (V I.V b.st) x ω r).msgs.length
    omega
  · change i < I.V.k (xIOf x) + kS x at hi
    obtain ⟨a, ha⟩ := run_inner H I.V b.st x P ω r hx hi (by omega)
    refine ⟨(coinSeq r i).bytes, ?_, Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective _⟩
    show ((run H P (V I.V b.st) x ω r).msgs.getD i ([], .rej)).2 = _
    rw [List.getD_eq_getElem?_getD, ha]
    rfl

/-! ## The staged statements show the run -/

/-- **The rounds a V\* statement's transcript shows**: its transcript, read back through the coin server's transcript
`b.tr x`. -/
noncomputable def shownOf (b : Brings) (x xO : Msg) : Option (List (Msg × Out)) :=
  ((decodeRec xO).bind RecRaw.transcript).map (Function.invFun (b.tr x))

/-- **The coins a V\* statement shows**: its transcript's rounds are each answered by the coin's bytes. -/
def showsTr (b : Brings) (_s : ℕ) (x xO : Msg) (c : ℕ → Coin) : Prop :=
  ∃ ps, shownOf b x xO = some ps ∧ ∀ i < ps.length, (ps.getD i ([], .rej)).2 = .msg (c i).bytes

/-- **The commit string a V\* statement shows at inner round `i`**: its transcript's prover message there. -/
noncomputable def cmtTr (b : Brings) (x : Msg) (i : ℕ) (xO : Msg) : Msg :=
  ((shownOf b x xO).map fun ps => (ps.map Prod.fst).getD i []).getD []

/-- **`Staged` at `deployRec`**, at every shape: each V* statement of an accepted run is part `s`'s staged one at the
run's inner and staging rounds (`run_round`), which reads back (`staged_ne_nil`, `staged_decode`) with the coin
server's transcript of the run's inner rounds; read back through `b.tr x`, wherever it is injective, those rounds are
answered by the run's coins (`run_inner`) and their prover messages are the run's commit strings, at each round the
layout spans, which are inner rounds. -/
theorem staged_deployRec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (L : Claims.Recursion.VStar.Layout) (shape : Claims.Recursion.Form → Prop)
    (htr : ∀ x, Function.Injective (b.tr x)) (hL : ∀ x, L.rounds x ≤ I.V.k (xIOf x)) :
    Claims.Recursion.VStar.Staged H (deployRec I.V ivScope b) I L shape (showsTr b) (cmtTr b) := by
  intro x _ _ _ Ω P ω r hacc s hs
  change (run H P (V I.V b.st) x ω r).out = .acc at hacc
  change s < nS x at hs
  have hne := staged_ne_nil H I.V b.st x P ω r hacc hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  obtain ⟨xr, p, -, -, hd⟩ := staged_decode hne
  have hK := run_acc_length H I.V b.st x P ω r hacc
  have hm := roundsMargin_V H I.V b.st x P ω r hacc
  obtain ⟨xr', hx⟩ := run_acc_readX H I.V b.st x P ω r hacc
  have hv : (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
    show (vstar b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [ite_eq_left hnum]
  have hdl : ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).dropLast =
      (run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x)) := by
    apply List.ext_getElem
    · simp only [List.length_dropLast, List.length_take, kS]
      omega
    · intro i h₁ h₂
      simp only [List.getElem_dropLast, List.getElem_take]
  have hshown : shownOf b x (staged b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s) =
      some ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))) := by
    unfold shownOf
    rw [hd, Option.bind_some]
    show Option.map (Function.invFun (b.tr x))
      (some (b.tr x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).dropLast)) = _
    rw [Option.map_some, Function.leftInverse_invFun (htr x) _, hdl]
  rw [hv]
  refine ⟨⟨_, hshown, fun i hi => ?_⟩, fun i hi => ?_⟩
  · have hi' : i < I.V.k (xIOf x) := by
      simp only [List.length_take] at hi
      omega
    obtain ⟨a, ha⟩ := run_inner H I.V b.st x P ω r hx (i := i) (by simp only [kS]; omega) (by omega)
    rw [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hi', ha]
    rfl
  · have hi' : i < I.V.k (xIOf x) := lt_of_lt_of_le hi (hL x)
    unfold cmtTr
    rw [hshown]
    show ((((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map Prod.fst).getD i []) =
      (((run H P (V I.V b.st) x ω r).msgs.map Prod.fst).getD i [])
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map,
      List.getElem?_take_of_lt hi']

end RecDeploy
