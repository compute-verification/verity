import Proofs.Flock.Claims.Recursion.DeployReg
import Proofs.Flock.Claims.Recursion.DeployScopeRuns
import Proofs.Flock.Claims.Recursion.DeployLayout
import Proofs.Flock.Claims.Recursion.InnerLeaf

/-!
# V*'s cross-part facts at `deployRec`

`VStarParts` (`VStar.lean`) asks, besides each part's own facts, for four facts across the parts: the staged V*
statements show the run (`Staged`), the parts' checks are the inner verifier (`Decomp`), its calls are theirs
(`LogDecomp`), and each inner round before the last is public, fixed or held by some part (`HeldPub`). This file proves
what of them holds at `deployRec` with no appeal to V*'s circuits, at every shape.

* **The staged statements read back** (`staged_ne_nil`): on an accepted run every V* session ends on a decision
  (`run_decides`), which the outer verifier makes only at a statement that sets up, so every part's staged statement is
  its encoding (`staged_decode`), whose transcript is the coin server's of the run's inner rounds (`Stage.ofMsg`).
* **`RunPairs` at `deployRec`** (`runPairs_deployRec`, no hypothesis): an accepted run sends every inner and staging
  round and at least three more (`roundsMargin_V`), each inner and staging round answered by its coin's bytes
  (`run_inner`), which `FlockDeploy.decCoin` reads back.
* **`Staged` at `deployRec`** (`staged_deployRec`), at the coins and commit strings a V* statement's transcript shows
  read by a reader `untr` (`showsTr`, `cmtTr`): wherever, at the compiled statements of the shape, `untr` reads the coin
  server's transcript of an accepted run's inner rounds back to their prover messages and the values their coins drew
  (`ReadsBack`), and the layout spans no more than the inner rounds.
  `VstarShows` does not hold at `deployRec`:
  a V* statement staged at rounds where its numbers aren't the part's is the one staged at `pick`, which shows `pick`'s
  coins.
* **`HeldPub` at `deployRec` is the layout's alone** (`heldPub_deployRec_iff`). With `Staged`'s span it pins the
  layout's rounds to the inner verifier's (`rounds_eq_of_heldPub`).
* **`Decomp` over Flock's deployed verifier is a cut** (`decomp_iff_cut`, `decomp_deployed_iff`): Flock's deployed
  verifier accepts replayed inner messages exactly where `InnerAcc` says (`run_replay_acc`, `innerAcc_of_run`): the
  first `N` of tag `0`, the first the link message, the `N`-th `Final`, the setup, the shipped verdict and the checks
  passing on it (`InnerAccSha` at SHA-512). So `Decomp` holds at a shape exactly where the parts' inner checks together
  give that (`InnerCut`), a fact about how the compiled statement cuts the inner verifier into V*'s parts.
* **`Decomp` refuses out-of-scope inner statements** (`innerAcc_public`, `decomp_public`): the inner checks run the
  inner statement's scope check and refuse it without a draw. `readX` runs the scope check and refuses an inner
  statement without a draw (`innerOk`), so at a statement the reader parses the inner statement holds one
  (`innerDrawn_checked`, in `DeployChecked`).
* **Flock's statement check passes the inner statement only in the plain-leaves form** (`check_plain_of_decode`,
  `inner_check_plain`): `RecVStar.innerOf` reads plain leaves off the circuit file (§16.16), and
  `ProvedScope.check` passes them at hidden outputs without `--zk` (`InnerLeaf.check_of_innerOf`). The reader requires
  that check (`innerOk`), so what `Decomp` asks at the checked reader is `DeployChecked`'s.
* **`LogDecomp` over Flock's deployed verifier asks the parts for its setup's calls** (`logDecomp_setup`,
  `logDecomp_deployed_setup`): every decision runs the inner statement's setup over `OC` first, on public data (the
  inner circuit file's bytes, its sigma's), so with `PartLogs` some part's SHA-512 gadgets would have to record each of
  them. With those calls admitted as public (`VStar.LogDecompPub`, which `VStarParts` asks for with the public calls
  the V* setup's own evaluations, `VStar.PubReg`), it is the verdict's calls alone that must be the parts'
  (`VerdictCut`, `logDecompPub_of_cut`).
* **At the registered ports' layout** (`layoutOf`, `DeployLayout.lean`): `Staged` needs only `ReadsBack`
  (`staged_layoutOf`), since the layout spans exactly the inner verifier's rounds. `HeldPub` holds where the registered
  ports cover the message rounds (`heldPub_layoutOf`). The layout records no call, so `PartLogs` leaves each part no
  inner call at its own points (`partLogs_layoutOf_nil`), and with `LogDecomp` the inner setup makes none at an
  accepted inner replay there (`setupLog_nil_layoutOf`), off a collision among the V* setup's own evaluations. The
  inner setup always hashes (its digest and sigma), so off such a collision no inner replay at the parts' points accepts
  under `LogDecomp`; `VStarParts` asks for `LogDecompPub` instead, which admits those calls.
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

/-- **A reader of the coin server's transcript**: at a compiled statement `x`, the rounds a transcript's bytes say,
each as the prover's message and the coin values the round drew. -/
abbrev TrReader : Type := Msg → ByteArray → Option (List (Msg × Array Flock.F128))

/-- **The values `vs` are the coin `c`'s first ones**: what a round that drew `vs.size` values shows of its coin. The
coin server's record keeps only the values a round drew (its `n`), so a transcript shows no more of a coin. -/
def Drawn (c : Coin) (vs : Array Flock.F128) : Prop := vs = c.vals vs.size

/-- **The reader reads the coin server's transcript back at an accepted run** (a named premise): at each compiled
statement of the shape, on every accepted run of `V IV b.st`, `untr` reads `b.tr x` of the run's inner rounds back to
those rounds' prover messages, each with values its round's coin drew (`Drawn`). It asks nothing at a list of rounds
no accepted run sends, and nothing of a coin past what its round drew. -/
def ReadsBack (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (shape : Claims.Recursion.Form → Prop)
    (untr : TrReader) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ (Ω : Type) (P : Prover Ω) (ω : Ω)
    (r : Fin ((V IV b.st).k x) → Coin), (run H P (V IV b.st) x ω r).out = .acc →
      ∃ ps, untr x (b.tr x ((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x)))) = some ps ∧
        ps.map Prod.fst = ((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x))).map Prod.fst ∧
        ∀ i < ps.length, Drawn (coinSeq r i) (ps.getD i ([], #[])).2

/-- **The reader reads the coin server's transcript back before the last round** (a named premise): `ReadsBack` at
the inner rounds before the last. The coin server records no last round, the openings, so a reader of its record reads
that round back nowhere; the bound layout reads no commit string there (`VStarB.StagedB`). -/
def ReadsBackB (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (shape : Claims.Recursion.Form → Prop)
    (untr : TrReader) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ (Ω : Type) (P : Prover Ω) (ω : Ω)
    (r : Fin ((V IV b.st).k x) → Coin), (run H P (V IV b.st) x ω r).out = .acc →
      ∃ ps, untr x (b.tr x ((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x)))) = some ps ∧
        (∀ i, i + 1 < IV.k (xIOf x) →
          (ps.map Prod.fst).getD i [] = ((run H P (V IV b.st) x ω r).msgs.map Prod.fst).getD i []) ∧
        ∀ i < ps.length, Drawn (coinSeq r i) (ps.getD i ([], #[])).2

/-- `ReadsBack` reads back before the last round. -/
theorem readsBackB_of_readsBack {H : Msg → Msg} {IV : Claims.Verifier Coin} {b : Brings}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} (h : ReadsBack H IV b shape untr) :
    ReadsBackB H IV b shape untr := by
  intro x st hst hf Ω P ω r hacc
  obtain ⟨ps, hps, hfst, hdrawn⟩ := h x st hst hf Ω P ω r hacc
  refine ⟨ps, hps, fun i hi => ?_, hdrawn⟩
  rw [hfst, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map,
    List.getElem?_take_of_lt (by omega)]

/-- **The rounds a V\* statement's transcript shows**: its transcript, read by `untr`. -/
def shownOf (untr : TrReader) (x xO : Msg) : Option (List (Msg × Array Flock.F128)) :=
  ((decodeRec xO).bind RecRaw.transcript).bind (untr x)

/-- **The coins a V\* statement shows**: at each round its transcript shows, the values that round drew are the coin's
first ones (`Drawn`). It shows nothing of a coin past them, so a part's check at the coins it shows (`PartChecksB`)
holds at every coin with those values. -/
def showsTr (untr : TrReader) (_s : ℕ) (x xO : Msg) (c : ℕ → Coin) : Prop :=
  ∃ ps, shownOf untr x xO = some ps ∧ ∀ i < ps.length, Drawn (c i) (ps.getD i ([], #[])).2

/-- **The commit string a V\* statement shows at inner round `i`**: its transcript's prover message there. -/
def cmtTr (untr : TrReader) (x : Msg) (i : ℕ) (xO : Msg) : Msg :=
  ((shownOf untr x xO).map fun ps => (ps.map Prod.fst).getD i []).getD []

/-- **`Staged` at `deployRec`**, at every shape: each V* statement of an accepted run is part `s`'s staged one at the
run's inner and staging rounds (`run_round`), which reads back (`staged_ne_nil`, `staged_decode`) with the coin
server's transcript of the run's inner rounds; read by `untr`, wherever it reads that transcript back (`ReadsBack`),
each round shows values its coin drew, and the rounds' prover messages are the run's commit strings, at each round the
layout spans, which are inner rounds. -/
theorem staged_deployRec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (untr : TrReader) (L : Claims.Recursion.VStar.Layout) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBack H I.V b shape untr)
    (hL : ∀ x st, parse x = some st → shape st.form → L.rounds x ≤ I.V.k (xIOf x)) :
    Claims.Recursion.VStar.Staged H (deployRec I.V ivScope b) I L shape (showsTr untr) (cmtTr untr) := by
  intro x st hst hf Ω P ω r hacc s hs
  have hL := hL x st hst hf
  change (run H P (V I.V b.st) x ω r).out = .acc at hacc
  obtain ⟨ps, hps, hfst, hdrawn⟩ := hrb x st hst hf Ω P ω r hacc
  change s < nS x at hs
  have hne := staged_ne_nil H I.V b.st x P ω r hacc hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  obtain ⟨xr, p, -, -, hd⟩ := staged_decode hne
  have hK := run_acc_length H I.V b.st x P ω r hacc
  have hm := roundsMargin_V H I.V b.st x P ω r hacc
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
  have hshown : shownOf untr x (staged b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s) =
      some ps := by
    unfold shownOf
    rw [hd, Option.bind_some]
    show Option.bind
      (some (b.tr x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).dropLast)) (untr x) = _
    rw [Option.bind_some, hdl]
    exact hps
  rw [hv]
  refine ⟨⟨_, hshown, hdrawn⟩, fun i hi => ?_⟩
  have hi' : i < I.V.k (xIOf x) := lt_of_lt_of_le hi hL
  unfold cmtTr
  rw [hshown]
  show ((ps.map Prod.fst).getD i []) = (((run H P (V I.V b.st) x ω r).msgs.map Prod.fst).getD i [])
  rw [hfst, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map,
    List.getElem?_take_of_lt hi']

/-! ## Each inner round before the last is public, fixed or held -/

/-- **`HeldPub` at `deployRec` is a fact about the layout alone**: at each compiled statement of the shape, the layout
spans the inner verifier's rounds at its inner statement, and each round before the last is public, fixed or held by
some part. -/
theorem heldPub_deployRec_iff {W : Type} (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
    (L : Claims.Recursion.VStar.Layout) (shape : Claims.Recursion.Form → Prop) :
    Claims.Recursion.VStar.HeldPub (deployRec I.V ivScope b) I L shape ↔
      ∀ x st, parse x = some st → shape st.form →
        I.V.k (xIOf x) ≤ L.rounds x ∧ ∀ i, i + 1 < I.V.k (xIOf x) →
          L.pubAt x i = true ∨ (L.fixedAt x i).isSome ∨ ∃ s < nS x, (L.msgAt x s i).isSome :=
  Iff.rfl

/-- **`HeldPub` with `Staged`'s span pins the layout's rounds** to the inner verifier's: `staged_deployRec` asks the
layout to span no more of them, and `HeldPub` no fewer. -/
theorem rounds_eq_of_heldPub {W : Type} {I : Claims.Recursion.Inner Coin W} {ivScope : Msg → Bool} {b : Brings}
    {L : Claims.Recursion.VStar.Layout} {shape : Claims.Recursion.Form → Prop}
    (hH : Claims.Recursion.VStar.HeldPub (deployRec I.V ivScope b) I L shape)
    (hL : ∀ x st, parse x = some st → shape st.form → L.rounds x ≤ I.V.k (xIOf x))
    {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) (hf : shape st.form) :
    L.rounds x = I.V.k (xIOf x) :=
  le_antisymm (hL x st hst hf) ((heldPub_deployRec_iff I ivScope b L shape).1 hH x st hst hf).1

/-! ## The inner verifier's accepting step -/

section Accepting

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)

/-- **An accepted run accepts at a step whose calls it logs**: at some prover messages `as'` the verifier's answer is
`.acc`, and each of that answer's oracle calls is in the run's verifier log. -/
theorem stepWith_acc_step (vn : List Msg → Claims.OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Claims.Trace),
      (Claims.stepWith H P x ω vn n hist as t).out = .acc → ∃ as', ((vn as').eval H).1 = .acc ∧
        ∀ e ∈ ((vn as').eval H).2, e ∈ (Claims.stepWith H P x ω vn n hist as t).vLog := by
  intro n
  induction n with
  | zero =>
    intro hist as t h
    simp [Claims.stepWith] at h
  | succ n ih =>
    intro hist as t h
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg m =>
      rw [stepWith_succ_msg H P x ω vn n hist as t hvb] at h ⊢
      exact ih _ _ _ h
    | acc =>
      rw [stepWith_succ_acc H P x ω vn n hist as t hvb]
      refine ⟨_, by rw [hvb], fun e he => ?_⟩
      rw [hvb] at he
      exact List.mem_append_right _ he
    | rej =>
      rw [stepWith_succ_rej H P x ω vn n hist as t hvb] at h
      cases h

end Accepting

/-- **An accepted run of Flock's deployed verifier accepts at a decision**, each of whose oracle calls the run logs:
its accepting answer is neither a refusal's nor a coin's (`next_rej`, `next_coin`). -/
theorem run_acc_decide {zk : Bool} {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((FlockDeploy.V zk).k x) → Coin) (h : (run H P (FlockDeploy.V zk) x ω r).out = .acc) :
    ∃ as cs s spec sts link c₁ mids f, FlockDeploy.act zk x as cs = .decide s spec sts link c₁ mids f ∧
      ((FlockDeploy.next zk x as cs).eval H).1 = .acc ∧
        ∀ e ∈ ((FlockDeploy.next zk x as cs).eval H).2, e ∈ (run H P (FlockDeploy.V zk) x ω r).vLog := by
  obtain ⟨as, hacc, hlog⟩ := stepWith_acc_step H P x ω _ _ [] [] _ h
  change ((FlockDeploy.next zk x as ((List.range as.length).map (coinSeq r))).eval H).1 = .acc at hacc
  change ∀ e ∈ ((FlockDeploy.next zk x as ((List.range as.length).map (coinSeq r))).eval H).2,
    e ∈ (run H P (FlockDeploy.V zk) x ω r).vLog at hlog
  rcases ha : FlockDeploy.act zk x as ((List.range as.length).map (coinSeq r)) with
    _ | c | ⟨s, spec, sts, link, c₁, mids, f⟩
  · rw [FlockDeploy.next_rej H ha] at hacc
    cases hacc
  · rw [FlockDeploy.next_coin H ha] at hacc
    cases hacc
  · exact ⟨as, _, s, spec, sts, link, c₁, mids, f, ha, hacc, hlog⟩

/-- **A decision logs its statement's setup calls**: it runs the setup over `OC` first (`FlockDeploy.decision`). -/
theorem next_decide_setup {zk : Bool} (H : Msg → Msg) {x : Msg} {as : List Msg} {cs : List Coin}
    {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.Final}
    (h : FlockDeploy.act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∀ e ∈ ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).2,
      e ∈ ((FlockDeploy.next zk x as cs).eval H).2 := by
  intro e he
  simp only [FlockDeploy.next, h, FlockDeploy.outOf, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil,
    FlockDeploy.decision, ExceptT.run_bind, Claims.OC.bind_eq]
  exact List.mem_append_left _ he

/-! ## `LogDecomp` over Flock's deployed verifier -/

/-- **`LogDecomp` over Flock's deployed verifier asks the parts for its setup's calls**: wherever the inner verifier
accepts the inner messages `ms` at the coins `c`, at a compiled statement of the shape, it decided at its statement's
setup (`act_decide`), whose oracle calls, made first (`next_decide_setup`), are in the run's log (`run_acc_decide`); so
each is in some part's `logs`. Those calls are on public data (`Claims.setupOC_log`: `Stmt.setupLogged`'s inputs, the
inner circuit file's bytes first, then its sigma's), and `PartLogs` puts every call in `logs` among the calls a part's
SHA-512 gadgets record. -/
theorem logDecomp_setup {zk : Bool} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {shape : Claims.Recursion.Form → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    (hLD : Claims.Recursion.VStar.LogDecomp H D (FlockDeploy.V zk) shape logs) {x : Msg}
    {st : Claims.Recursion.Stmt} (hst : D.parse x = some st) (hf : shape st.form) (c : ℕ → Coin) (ms : List Msg)
    (hacc : (run H (replay ms) (FlockDeploy.V zk) (Claims.Recursion.xI D x) () fun i => c i.val).out = .acc) :
    ∃ s spec sts, FlockDeploy.setupOf zk (Claims.Recursion.xI D x) = some (s, spec, sts) ∧
      ∀ e ∈ ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).2, ∃ p < D.nS x, e ∈ logs p x c ms := by
  obtain ⟨as, cs, s, spec, sts, link, c₁, mids, f, ha, -, hlog⟩ := run_acc_decide H _ _ _ _ hacc
  exact ⟨s, spec, sts, FlockDeploy.act_decide ha, fun e he =>
    hLD x st hst hf c ms e (hlog e (next_decide_setup H ha e he))⟩

/-! ## `Decomp` over Flock's deployed verifier -/

/-- **Flock's deployed verifier accepts the inner messages `ms` at the coins `c`**, as its steps read them: the
statement sets up; for some `N` within its rounds, the first `N` messages are of tag `0`, the first decodes as the
link message and the `N`-th as `Final`; and the setup with `hOf H`, the shipped verdict with `hOf H` and the checks
pass on it, at the link's coin `c 0` and the round messages' coins. -/
def InnerAcc (H : Msg → Msg) (x : Msg) (c : ℕ → Coin) (ms : List Msg) : Prop :=
  ∃ (N : ℕ) (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt)
    (link : List ByteArray × List ByteArray) (F : FlockDeploy.Final),
    FlockDeploy.setupOf false x = some (s, spec, sts) ∧ N + 1 ≤ (FlockDeploy.V false).k x ∧
      (∀ j < N, ∃ t, ms.getD j [] = 0 :: t) ∧ FlockDeploy.linkOf (ms.getD 0 []) = some link ∧
      FlockDeploy.finalOf (ms.getD N []) = some F ∧
      ∃ spec' sts', FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        FlockDeploy.verdictWith false (Claims.hOf H) spec' (FlockDeploy.tabsOf sts') F.record F.proofs = .ok () ∧
        FlockDeploy.checks false s spec' x link (c 0)
          (FlockComplete.midsOf ((List.range (N + 1)).map fun j => ms.getD j []) ((List.range (N + 1)).map c))
          F = .ok ()

theorem coinSeq_val {k : ℕ} (c : ℕ → Coin) {j : ℕ} (hj : j < k) :
    coinSeq (fun i : Fin k => c i.val) j = c j := by
  unfold coinSeq
  rw [dite_eq_left hj]

/-- **The replayed messages are accepted where `InnerAcc` says**: the first `N` are each answered by their coin
(`act_coin`), and the `N`-th is decided (`act_final`), which accepts (`next_decide`). -/
theorem run_replay_acc (H : Msg → Msg) (x : Msg) (c : ℕ → Coin) (ms : List Msg) (h : InnerAcc H x c ms) :
    (run H (replay ms) (FlockDeploy.V false) x () fun i => c i.val).out = .acc := by
  obtain ⟨N, s, spec, sts, link, F, hs, hk, htag, hlink, hfin, spec', sts', hsu, hv, hc⟩ := h
  let seq : ℕ → Msg := fun j => ms.getD j []
  let resp : ℕ → Msg := fun j => (coinSeq (fun i : Fin ((FlockDeploy.V false).k x) => c i.val) j).bytes
  have hlast : ∀ j, ((List.range (j + 1)).map seq).getLast? = some (seq j) := fun j => by
    simp only [List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel,
      Option.map_some]
  unfold Claims.run
  refine FlockComplete.stepWith_acc H (replay ms) x () _ seq resp N ?_ ?_ ?_ _ hk _
  · intro j _ hist hh
    have hl : hist.length = j := by simpa using congrArg List.length hh
    show ms.getD hist.length [] = ms.getD j []
    rw [hl]
  · intro j hj
    obtain ⟨t, ht⟩ := htag j hj
    have ha : ((List.range (j + 1)).map seq).getLast? = some (0 :: t) := by
      rw [hlast j]
      exact congrArg some ht
    have hc' : ((List.range ((List.range (j + 1)).map seq).length).map
        (coinSeq (fun i : Fin ((FlockDeploy.V false).k x) => c i.val))).getLast? =
        some (coinSeq (fun i : Fin ((FlockDeploy.V false).k x) => c i.val) j) := by
      simp only [List.length_map, List.length_range, List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero,
        ite_false, Nat.add_sub_cancel, Option.map_some]
    exact (FlockDeploy.next_coin H (FlockComplete.act_coin hs ha hc')).symm ▸ rfl
  · show ((FlockDeploy.next false x ((List.range (N + 1)).map seq)
      ((List.range ((List.range (N + 1)).map seq).length).map
        (coinSeq (fun i : Fin ((FlockDeploy.V false).k x) => c i.val)))).eval H).1 = .acc
    rw [List.length_map, List.length_range]
    have hcoins : (List.range (N + 1)).map (coinSeq (fun i : Fin ((FlockDeploy.V false).k x) => c i.val)) =
        (List.range (N + 1)).map c :=
      List.map_congr_left fun j hj => coinSeq_val c (by rw [List.mem_range] at hj; omega)
    have hms : (List.range (N + 1)).map seq = seq 0 :: (List.range N).map (fun j => seq (j + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      rfl
    have hcs : (List.range (N + 1)).map c = c 0 :: (List.range N).map (fun j => c (j + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      rfl
    have hl := hlast N
    rw [hms] at hl
    rw [hcoins, hms, hcs]
    have had := FlockComplete.act_final (c₁ := c 0) (crest := (List.range N).map (fun j => c (j + 1))) hs hl hfin
      hlink
    rw [FlockDeploy.next_decide H had]
    refine ⟨spec', sts', hsu, hv, ?_⟩
    change FlockDeploy.checks false s spec' x link (c 0)
      (FlockComplete.midsOf ((List.range (N + 1)).map seq) ((List.range (N + 1)).map c)) F = .ok () at hc
    rw [hms, hcs] at hc
    exact hc

/-- **The cut of the inner verifier into V\*'s parts** (a code fact about the compiled statement at the shape): wherever
every part's inner check holds at a compiled statement of the shape, the inner verifier's acceptance condition holds
(`InnerAcc`). -/
def InnerCut (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ s < D.nS x, chk s x c ms) → InnerAcc H (Claims.Recursion.xI D x) c ms

/-- **`Decomp` over Flock's deployed verifier, from the cut** (`run_replay_acc`), at every shape. -/
theorem decomp_of_cut {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (h : InnerCut H D shape chk) :
    Claims.Recursion.VStar.Decomp H D (FlockDeploy.V false) shape chk :=
  fun x st hst hf c ms hc => run_replay_acc H _ c ms (h x st hst hf c ms hc)

/-- **`InnerAcc` reads the inner statement's public checks**: the checks after the verdict run its scope check
(`scope_of_checks`) and refuse it without a draw (`ProvedScope.drawLaw`), on what the verifier read of it. -/
theorem innerAcc_public {H : Msg → Msg} {x : Msg} {c : ℕ → Coin} {ms : List Msg} (h : InnerAcc H x c ms) :
    ∃ s spec sts, FlockDeploy.setupOf false x = some (s, spec, sts) ∧ s.read.check = .ok () ∧
      ∃ d, s.read.draw = some d := by
  obtain ⟨_, s, spec, sts, _, _, hs, -, -, -, -, _, _, -, -, hc⟩ := h
  refine ⟨s, spec, sts, hs, scope_of_checks hc, ?_⟩
  unfold FlockDeploy.checks at hc
  obtain ⟨_, -, hc⟩ := FlockSoundness.Refine.ebind_ok hc
  obtain ⟨_, -, hc⟩ := FlockSoundness.Refine.ebind_ok hc
  obtain ⟨_, -, hc⟩ := FlockSoundness.Refine.ebind_ok hc
  obtain ⟨_, -, hc⟩ := FlockSoundness.Refine.ebind_ok hc
  obtain ⟨_, -, hc⟩ := FlockSoundness.Refine.ebind_ok hc
  obtain ⟨_, hc, -⟩ := FlockSoundness.Refine.ebind_ok hc
  cases hd : s.read.draw with
  | none => simp [hd, Flock.ProvedScope.drawLaw] at hc
  | some d => exact ⟨d, rfl⟩

/-! ## The replayed run's steps -/

section Replay

variable (H : Msg → Msg) (x : Msg) (ms : List Msg) (vn : List Msg → Claims.OC Out)

/-- **A replayed run that accepts accepts at one step**: from `j` messages sent, the run sends the next of `ms` each
step; some step `N` within the rounds left is answered `.acc`, every step before it a message. -/
theorem stepWith_replay_acc :
    ∀ (n j : ℕ) (hist : List (Msg × Msg)) (t : Claims.Trace), hist.length = j →
      (Claims.stepWith H (replay ms) x () vn n hist ((List.range j).map fun i => ms.getD i []) t).out = .acc →
        ∃ N, j ≤ N ∧ N < j + n ∧
          (∀ i, j ≤ i → i < N → ∃ m, ((vn ((List.range (i + 1)).map fun i => ms.getD i [])).eval H).1 = .msg m) ∧
          ((vn ((List.range (N + 1)).map fun i => ms.getD i [])).eval H).1 = .acc := by
  intro n
  induction n with
  | zero =>
    intro j hist t _ h
    simp [Claims.stepWith] at h
  | succ n ih =>
    intro j hist t hl h
    have hs : (List.range j).map (fun i => ms.getD i []) ++ [(((replay ms).next x hist ()).eval H).1] =
        (List.range (j + 1)).map fun i => ms.getD i [] := by
      show _ ++ [ms.getD hist.length []] = _
      rw [hl, List.range_succ, List.map_append, List.map_singleton]
    rcases hvb : (vn ((List.range j).map (fun i => ms.getD i []) ++
      [(((replay ms).next x hist ()).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg m =>
      rw [stepWith_succ_msg H (replay ms) x () vn n hist _ t hvb, hs] at h
      obtain ⟨N, h1, h2, h3, h4⟩ := ih (j + 1) _ _ (by simp [hl]) h
      rw [hs] at hvb
      refine ⟨N, by omega, by omega, fun i hi hi' => ?_, h4⟩
      rcases Nat.eq_or_lt_of_le hi with rfl | hi
      · exact ⟨m, by rw [hvb]⟩
      · exact h3 i hi hi'
    | acc =>
      rw [hs] at hvb
      exact ⟨j, le_refl _, by omega, fun i hi hi' => absurd hi' (by omega), by rw [hvb]⟩
    | rej =>
      rw [stepWith_succ_rej H (replay ms) x () vn n hist _ t hvb] at h
      cases h

/-- **A replayed run logs its steps' calls**: each call in its verifier log is in the log it started with, or in the
log of the answer at some step `N` within the rounds left, every step before which is answered by a message. -/
theorem stepWith_replay_vLog :
    ∀ (n j : ℕ) (hist : List (Msg × Msg)) (t : Claims.Trace), hist.length = j →
      ∀ e ∈ (Claims.stepWith H (replay ms) x () vn n hist ((List.range j).map fun i => ms.getD i []) t).vLog,
        e ∈ t.vLog ∨ ∃ N, j ≤ N ∧ N < j + n ∧
          (∀ i, j ≤ i → i < N → ∃ m, ((vn ((List.range (i + 1)).map fun i => ms.getD i [])).eval H).1 = .msg m) ∧
          e ∈ ((vn ((List.range (N + 1)).map fun i => ms.getD i [])).eval H).2 := by
  intro n
  induction n with
  | zero =>
    intro j hist t _ e he
    exact Or.inl he
  | succ n ih =>
    intro j hist t hl e he
    have hs : (List.range j).map (fun i => ms.getD i []) ++ [(((replay ms).next x hist ()).eval H).1] =
        (List.range (j + 1)).map fun i => ms.getD i [] := by
      show _ ++ [ms.getD hist.length []] = _
      rw [hl, List.range_succ, List.map_append, List.map_singleton]
    rcases hvb : (vn ((List.range j).map (fun i => ms.getD i []) ++
      [(((replay ms).next x hist ()).eval H).1])).eval H with ⟨o, lo⟩
    have hlo : ∀ e ∈ lo, e ∈ ((vn ((List.range (j + 1)).map fun i => ms.getD i [])).eval H).2 := by
      rw [← hs, hvb]
      exact fun _ h => h
    cases o with
    | msg m =>
      rw [stepWith_succ_msg H (replay ms) x () vn n hist _ t hvb, hs] at he
      have hm : ((vn ((List.range (j + 1)).map fun i => ms.getD i [])).eval H).1 = .msg m := by rw [← hs, hvb]
      rcases ih (j + 1) _ _ (by simp [hl]) e he with he | ⟨N, h1, h2, h3, h4⟩
      · rcases List.mem_append.1 he with he | he
        · exact Or.inl he
        · exact Or.inr ⟨j, le_refl _, by omega, fun i hi hi' => absurd hi' (by omega), hlo e he⟩
      · refine Or.inr ⟨N, by omega, by omega, fun i hi hi' => ?_, h4⟩
        rcases Nat.eq_or_lt_of_le hi with rfl | hi
        · exact ⟨m, hm⟩
        · exact h3 i hi hi'
    | acc =>
      rw [stepWith_succ_acc H (replay ms) x () vn n hist _ t hvb] at he
      rcases List.mem_append.1 he with he | he
      · exact Or.inl he
      · exact Or.inr ⟨j, le_refl _, by omega, fun i hi hi' => absurd hi' (by omega), hlo e he⟩
    | rej =>
      rw [stepWith_succ_rej H (replay ms) x () vn n hist _ t hvb] at he
      rcases List.mem_append.1 he with he | he
      · exact Or.inl he
      · exact Or.inr ⟨j, le_refl _, by omega, fun i hi hi' => absurd hi' (by omega), hlo e he⟩

end Replay

/-- **Only a coin is answered by a message**: a refusal is answered `.rej`, a decision `.acc` or `.rej`. -/
theorem next_msg_coin {zk : Bool} (H : Msg → Msg) {x : Msg} {as : List Msg} {cs : List Coin} {m : Msg}
    (h : ((FlockDeploy.next zk x as cs).eval H).1 = .msg m) : ∃ c, FlockDeploy.act zk x as cs = .coin c := by
  rcases ha : FlockDeploy.act zk x as cs with _ | c | ⟨s, spec, sts, link, c₁, mids, f⟩
  · rw [FlockDeploy.next_rej H ha] at h
    cases h
  · exact ⟨c, rfl⟩
  · simp only [FlockDeploy.next, ha, FlockDeploy.outOf, Claims.OC.eval_bind, Claims.OC.eval] at h
    split at h <;> cases h

/-- **A coin answers a last message of tag `0`.** -/
theorem act_coin_tag {zk : Bool} {x : Msg} {as : List Msg} {cs : List Coin} {c : Coin}
    (h : FlockDeploy.act zk x as cs = .coin c) : ∃ t, as.getLast? = some (0 :: t) := by
  unfold FlockDeploy.act at h
  split at h
  · cases h
  · split at h
    · rename_i heq _
      exact ⟨_, heq⟩
    · repeat' split at h
      all_goals cases h
    · cases h

/-- **A decision is on the link message, its coin and the last message**, as `act_final` makes it. -/
theorem act_decide_parts {x a₁ : Msg} {rest : List Msg} {c₁' : Coin} {crest : List Coin} {s : FlockDeploy.Ins}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Msg × Coin)} {f : FlockDeploy.Final}
    (h : FlockDeploy.act false x (a₁ :: rest) (c₁' :: crest) = .decide s spec sts link c₁ mids f) :
    FlockDeploy.linkOf a₁ = some link ∧ FlockDeploy.finalOf (((a₁ :: rest).getLast?).getD []) = some f ∧
      c₁ = c₁' ∧ mids = (rest.dropLast.map List.tail).zip crest.dropLast := by
  unfold FlockDeploy.act at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i a hl _
      split at h
      · rename_i a₁' rest' c₁'' crest' hfin has hcs
        split at h
        · rename_i link' hlink
          cases h
          simp only [List.cons.injEq] at has hcs
          obtain ⟨rfl, rfl⟩ := has
          obtain ⟨rfl, rfl⟩ := hcs
          refine ⟨hlink, ?_, rfl, rfl⟩
          rw [hl]
          exact hfin
        · cases h
      · cases h
    · cases h

/-- **An accepted replay is where `InnerAcc` says**: it accepts at one step (`stepWith_replay_acc`), every step before
which is a coin's (`next_msg_coin`), so its message is of tag `0` (`act_coin_tag`); the accepting step is a decision
(`run_acc_decide`'s cases) on the link message, its coin and the last message (`act_decide_parts`), which accepts as
`next_decide` says. -/
theorem innerAcc_of_run (H : Msg → Msg) (x : Msg) (c : ℕ → Coin) (ms : List Msg)
    (h : (run H (replay ms) (FlockDeploy.V false) x () fun i => c i.val).out = .acc) : InnerAcc H x c ms := by
  let g : ℕ → Msg := fun i => ms.getD i []
  let r : Fin ((FlockDeploy.V false).k x) → Coin := fun i => c i.val
  obtain ⟨N, -, hN, hpre, hacc⟩ := stepWith_replay_acc H x ms
    (fun as => (FlockDeploy.V false).next x as ((List.range as.length).map (coinSeq r))) _ 0 [] _ rfl h
  rw [Nat.zero_add] at hN
  have hlast : ∀ j, ((List.range (j + 1)).map g).getLast? = some (g j) := fun j => by
    simp only [List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel,
      Option.map_some]
  have htag : ∀ j < N, ∃ t, ms.getD j [] = 0 :: t := by
    intro j hj
    obtain ⟨m, hm⟩ := hpre j (Nat.zero_le _) hj
    obtain ⟨c', hc'⟩ := next_msg_coin H hm
    obtain ⟨t, ht⟩ := act_coin_tag hc'
    rw [hlast j] at ht
    exact ⟨t, Option.some.inj ht⟩
  change ((FlockDeploy.next false x ((List.range (N + 1)).map g)
    ((List.range ((List.range (N + 1)).map g).length).map (coinSeq r))).eval H).1 = .acc at hacc
  rw [List.length_map, List.length_range] at hacc
  have hcoins : (List.range (N + 1)).map (coinSeq r) = (List.range (N + 1)).map c :=
    List.map_congr_left fun j hj => coinSeq_val c (by rw [List.mem_range] at hj; omega)
  have hms : (List.range (N + 1)).map g = g 0 :: (List.range N).map (fun j => g (j + 1)) := by
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    rfl
  have hcs : (List.range (N + 1)).map c = c 0 :: (List.range N).map (fun j => c (j + 1)) := by
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    rfl
  rw [hcoins, hms, hcs] at hacc
  rcases ha : FlockDeploy.act false x (g 0 :: (List.range N).map (fun j => g (j + 1)))
      (c 0 :: (List.range N).map (fun j => c (j + 1))) with _ | c' | ⟨s, spec, sts, link, c₁, mids, f⟩
  · rw [FlockDeploy.next_rej H ha] at hacc
    cases hacc
  · rw [FlockDeploy.next_coin H ha] at hacc
    cases hacc
  · obtain ⟨spec', sts', hsu, hv, hc⟩ := (FlockDeploy.next_decide H ha).1 hacc
    obtain ⟨hlink, hfin, rfl, rfl⟩ := act_decide_parts ha
    have hl := hlast N
    rw [hms] at hl
    rw [hl] at hfin
    refine ⟨N, s, spec, sts, link, f, FlockDeploy.act_decide ha, by omega, htag, hlink, hfin, spec', sts', hsu, hv,
      ?_⟩
    change FlockDeploy.checks false s spec' x link (c 0)
      (FlockComplete.midsOf ((List.range (N + 1)).map g) ((List.range (N + 1)).map c)) f = .ok ()
    rw [hms, hcs]
    exact hc

/-- **`Decomp` over Flock's deployed verifier is the cut**, at every shape: the cut gives it (`decomp_of_cut`), and an
accepted replay is where `InnerAcc` says (`innerAcc_of_run`). -/
theorem decomp_iff_cut {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} :
    Claims.Recursion.VStar.Decomp H D (FlockDeploy.V false) shape chk ↔ InnerCut H D shape chk :=
  ⟨fun h x st hst hf c ms hc => innerAcc_of_run H _ c ms (h x st hst hf c ms hc), decomp_of_cut⟩

/-- **`Decomp` asks nothing of the parts to hold where the inner statement is out of scope**: at a compiled statement
of the shape whose inner statement fails its scope check or holds no draw, the parts' inner checks hold together at no
coins and no inner messages, since the inner verifier accepts none there (`innerAcc_of_run`, `innerAcc_public`). -/
theorem decomp_public {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    (hD : Claims.Recursion.VStar.Decomp H D (FlockDeploy.V false) shape chk) {x : Msg} {st : Claims.Recursion.Stmt}
    (hst : D.parse x = some st) (hf : shape st.form)
    (hout : ¬ ∃ s spec sts, FlockDeploy.setupOf false (Claims.Recursion.xI D x) = some (s, spec, sts) ∧
      s.read.check = .ok () ∧ ∃ d, s.read.draw = some d) (c : ℕ → Coin) (ms : List Msg) :
    ¬ ∀ p < D.nS x, chk p x c ms := fun hall =>
  hout (innerAcc_public (innerAcc_of_run H _ c ms (hD x st hst hf c ms hall)))

/-! ## `LogDecomp` with the inner setup's calls admitted -/

/-- **A decision's calls are its setup's, then its verdict's** where the setup with `hOf H` is `.ok`
(`FlockDeploy.decision`, `decisionAt`; the checks make none). -/
theorem next_decide_log_mem {zk : Bool} (H : Msg → Msg) {x : Msg} {as : List Msg} {cs : List Coin}
    {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.Final}
    (h : FlockDeploy.act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∀ e ∈ ((FlockDeploy.next zk x as cs).eval H).2, e ∈ ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).2 ∨
      ∃ spec' sts', FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        e ∈ ((FlockDeploy.verdictM zk Claims.Hashes.oc spec' (FlockDeploy.tabsOf sts') f.record f.proofs).run.eval
          H).2 := by
  intro e he
  obtain ⟨hev, -⟩ := FlockDeploy.setupM_eval H s.read
  simp only [FlockDeploy.next, h, FlockDeploy.outOf, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil,
    FlockDeploy.decision, ExceptT.run_bind, Claims.OC.bind_eq] at he
  rcases hv : (FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at hev he
  simp only at hev he
  rcases List.mem_append.1 he with he | he
  · exact Or.inl he
  · subst hev
    cases hr : FlockDeploy.setupWith (Claims.hOf H) s.read with
    | error err =>
      rw [hr] at he
      cases he
    | ok p =>
      obtain ⟨spec', sts'⟩ := p
      rw [hr] at he
      refine Or.inr ⟨spec', sts', rfl, ?_⟩
      simp only [FlockDeploy.decisionAt, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind] at he
      rcases hw : (FlockDeploy.verdictM zk Claims.Hashes.oc spec' (FlockDeploy.tabsOf sts') f.record
        f.proofs).run.eval H with ⟨w, lw⟩
      rw [hw] at he
      rcases List.mem_append.1 he with he | he
      · exact he
      · cases w <;> cases he

/-- **The inner statement's setup calls** against `H`: those Flock's deployed verifier makes at its decision, its
statement's setup over `OC` (none where the statement doesn't set up). -/
def innerSetupLog (zk : Bool) (H : Msg → Msg) (x : Msg) : List (Msg × Msg) :=
  match FlockDeploy.setupOf zk x with
  | some (s, _, _) => ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).2
  | none => []

/-- **The verdict's calls are the parts'** (a code fact about the compiled statement at the shape): at a compiled
statement of the shape, wherever the inner messages `ms` reach a decision at round `N` (the first `N` of tag `0`, the
first the link message, the `N`-th `Final`), each call the shipped verdict makes there, at the setup with `hOf H`, is in
some part's `logs`. -/
def VerdictCut (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg)
    (N : ℕ) (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt)
    (link : List ByteArray × List ByteArray) (F : FlockDeploy.Final) (spec' : Flock.SessionSpec)
    (sts' : Array Flock.Stmt),
    FlockDeploy.setupOf false (Claims.Recursion.xI D x) = some (s, spec, sts) →
      N + 1 ≤ (FlockDeploy.V false).k (Claims.Recursion.xI D x) → (∀ j < N, ∃ t, ms.getD j [] = 0 :: t) →
      FlockDeploy.linkOf (ms.getD 0 []) = some link → FlockDeploy.finalOf (ms.getD N []) = some F →
      FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
      ∀ e ∈ ((FlockDeploy.verdictM false Claims.Hashes.oc spec' (FlockDeploy.tabsOf sts') F.record F.proofs).run.eval
        H).2, ∃ p < D.nS x, e ∈ logs p x c ms

/-- **`LogDecompPub` over Flock's deployed verifier, from the verdict's cut**, with the inner statement's setup calls
public: each call of a replay is at one step (`stepWith_replay_vLog`), which logs only if it is a decision
(`next_rej`, `next_coin`), whose calls are its setup's or its verdict's (`next_decide_log_mem`). -/
theorem logDecompPub_of_cut {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} (h : VerdictCut H D shape logs) :
    Claims.Recursion.VStar.LogDecompPub H D (FlockDeploy.V false) shape logs fun x =>
      innerSetupLog false H (Claims.Recursion.xI D x) := by
  intro x st hst hf c ms e he
  let x' := Claims.Recursion.xI D x
  let g : ℕ → Msg := fun i => ms.getD i []
  let r : Fin ((FlockDeploy.V false).k x') → Coin := fun i => c i.val
  rcases stepWith_replay_vLog H x' ms
      (fun as => (FlockDeploy.V false).next x' as ((List.range as.length).map (coinSeq r))) _ 0 [] _ rfl e he with
    he | ⟨N, -, hN, hpre, he⟩
  · cases he
  rw [Nat.zero_add] at hN
  have hlast : ∀ j, ((List.range (j + 1)).map g).getLast? = some (g j) := fun j => by
    simp only [List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel,
      Option.map_some]
  have htag : ∀ j < N, ∃ t, ms.getD j [] = 0 :: t := by
    intro j hj
    obtain ⟨m, hm⟩ := hpre j (Nat.zero_le _) hj
    obtain ⟨c', hc'⟩ := next_msg_coin H hm
    obtain ⟨t, ht⟩ := act_coin_tag hc'
    rw [hlast j] at ht
    exact ⟨t, Option.some.inj ht⟩
  change e ∈ ((FlockDeploy.next false x' ((List.range (N + 1)).map g)
    ((List.range ((List.range (N + 1)).map g).length).map (coinSeq r))).eval H).2 at he
  rw [List.length_map, List.length_range] at he
  have hcoins : (List.range (N + 1)).map (coinSeq r) = (List.range (N + 1)).map c :=
    List.map_congr_left fun j hj => coinSeq_val c (by
      rw [List.mem_range] at hj
      exact (show j < (FlockDeploy.V false).k (Claims.Recursion.xI D x) by omega))
  have hms : (List.range (N + 1)).map g = g 0 :: (List.range N).map (fun j => g (j + 1)) := by
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    rfl
  have hcs : (List.range (N + 1)).map c = c 0 :: (List.range N).map (fun j => c (j + 1)) := by
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    rfl
  rw [hcoins, hms, hcs] at he
  rcases ha : FlockDeploy.act false x' (g 0 :: (List.range N).map (fun j => g (j + 1)))
      (c 0 :: (List.range N).map (fun j => c (j + 1))) with _ | c' | ⟨s, spec, sts, link, c₁, mids, f⟩
  · rw [FlockDeploy.next_rej H ha] at he
    cases he
  · rw [FlockDeploy.next_coin H ha] at he
    cases he
  · have hs := FlockDeploy.act_decide ha
    rcases next_decide_log_mem H ha e he with he | ⟨spec', sts', hsu, he⟩
    · right
      show e ∈ innerSetupLog false H x'
      unfold innerSetupLog
      rw [hs]
      exact he
    · left
      obtain ⟨hlink, hfin, -, -⟩ := act_decide_parts ha
      have hl := hlast N
      rw [hms] at hl
      rw [hl] at hfin
      exact h x st hst hf c ms N s spec sts link f spec' sts' hs (by omega) htag hlink hfin hsu e he

/-- **At SHA-512 the setup with `hOf H` is the statement's**, on every statement that sets up. -/
theorem setupWith_sha512_of {zk : Bool} {x : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (h : FlockDeploy.setupOf zk x = some (s, spec, sts)) :
    FlockDeploy.setupWith (Claims.hOf FlockDeploy.sha512) s.read = .ok (spec, sts) := by
  rw [FlockDeploy.hOf_sha512, FlockDeploy.setupWith_sha512]
  exact FlockDeploy.Ins.setup_read (FlockDeploy.setupOf_some h).2

/-! ## At the deployed recursion -/

/-- **`InnerAcc` at SHA-512**: the statement's own setup, and the shipped verdict at SHA-512. -/
def InnerAccSha (x : Msg) (c : ℕ → Coin) (ms : List Msg) : Prop :=
  ∃ (N : ℕ) (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt)
    (link : List ByteArray × List ByteArray) (F : FlockDeploy.Final),
    FlockDeploy.setupOf false x = some (s, spec, sts) ∧ N + 1 ≤ (FlockDeploy.V false).k x ∧
      (∀ j < N, ∃ t, ms.getD j [] = 0 :: t) ∧ FlockDeploy.linkOf (ms.getD 0 []) = some link ∧
      FlockDeploy.finalOf (ms.getD N []) = some F ∧
      FlockDeploy.verdictWith false Flock.Sha512.hash spec (FlockDeploy.tabsOf sts) F.record F.proofs = .ok () ∧
      FlockDeploy.checks false s spec x link (c 0)
        (FlockComplete.midsOf ((List.range (N + 1)).map fun j => ms.getD j []) ((List.range (N + 1)).map c))
        F = .ok ()

theorem innerAcc_sha512_iff (x : Msg) (c : ℕ → Coin) (ms : List Msg) :
    InnerAcc FlockDeploy.sha512 x c ms ↔ InnerAccSha x c ms := by
  constructor
  · rintro ⟨N, s, spec, sts, link, F, hs, hk, htag, hlink, hfin, spec', sts', hsu, hv, hc⟩
    rw [setupWith_sha512_of hs] at hsu
    cases hsu
    rw [FlockDeploy.hOf_sha512] at hv
    exact ⟨N, s, spec, sts, link, F, hs, hk, htag, hlink, hfin, hv, hc⟩
  · rintro ⟨N, s, spec, sts, link, F, hs, hk, htag, hlink, hfin, hv, hc⟩
    refine ⟨N, s, spec, sts, link, F, hs, hk, htag, hlink, hfin, spec, sts, setupWith_sha512_of hs, ?_, hc⟩
    rw [FlockDeploy.hOf_sha512]
    exact hv

/-- **`Decomp` at the deployed recursion is the cut at SHA-512**, at every shape: wherever every part's inner check
holds at a compiled statement of the shape, its inner statement is accepted as `InnerAccSha` says (`decomp_iff_cut`,
`innerAcc_sha512_iff`). -/
theorem decomp_deployed_iff (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (ivScope : Msg → Bool) (b : Brings)
    (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) :
    Claims.Recursion.VStar.Decomp FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI).V shape chk ↔
      ∀ (x : Msg) (st : Claims.Recursion.Stmt), (deployRec (FlockDeploy.V false) ivScope b).parse x = some st →
        shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg), (∀ s < nS x, chk s x c ms) → InnerAccSha (xIOf x) c ms := by
  rw [show (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI).V = FlockDeploy.V false from rfl,
    decomp_iff_cut]
  exact forall_congr' fun x => forall_congr' fun st => forall_congr' fun _ => forall_congr' fun _ =>
    forall_congr' fun c => forall_congr' fun ms => imp_congr_right fun _ => innerAcc_sha512_iff _ c ms

/-- **`LogDecomp` at the deployed recursion asks the parts for the inner setup's calls**, at every shape
(`logDecomp_setup`): on every accepted replay of a compiled statement of the shape, each SHA-512 call of the inner
statement's setup is in some part's `logs`. -/
theorem logDecomp_deployed_setup (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (ivScope : Msg → Bool) (b : Brings)
    {shape : Claims.Recursion.Form → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    (hLD : Claims.Recursion.VStar.LogDecomp FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI).V shape logs)
    {x : Msg} {st : Claims.Recursion.Stmt} (hst : (deployRec (FlockDeploy.V false) ivScope b).parse x = some st)
    (hf : shape st.form) (c : ℕ → Coin) (ms : List Msg)
    (hacc : (run FlockDeploy.sha512 (replay ms) (FlockDeploy.V false) (xIOf x) () fun i => c i.val).out = .acc) :
    ∃ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) ∧
      ∀ e ∈ ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval FlockDeploy.sha512).2,
        ∃ p < nS x, e ∈ logs p x c ms :=
  logDecomp_setup hLD hst hf c ms hacc

/-! ## At the registered ports' layout -/

/-- **`Staged` at `layoutOf`**, at every shape, wherever the reader reads the coin server's transcript back
(`ReadsBack`): the layout spans exactly the inner verifier's rounds. -/
theorem staged_layoutOf (H : Msg → Msg) (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (ivScope : Msg → Bool) (b : Brings)
    (untr : TrReader) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBack H (FlockDeploy.V false) b.atPorts shape untr) :
    Claims.Recursion.VStar.Staged H (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) layoutOf shape (showsTr untr)
      (cmtTr untr) :=
  staged_deployRec H (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) ivScope b.atPorts untr
    layoutOf shape hrb fun _ _ _ _ => le_rfl

/-- **`PartLogs` at `layoutOf` leaves the part no inner calls**: the layout records none, so wherever the part's outer
relation holds at a V* statement showing the coins, its inner check makes no call on messages agreeing with what it
holds, or the V* setup's own evaluations at that statement collide. -/
theorem partLogs_layoutOf_nil {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {shape : Claims.Recursion.Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    (hP : Claims.Recursion.VStar.PartLogs H D DO layoutOf shape shows logs s) {x : Msg} {st : Claims.Recursion.Stmt}
    (hst : D.parse x = some st) (hf : shape st.form) {xO : Msg} {w : Claims.Flock.Wit}
    (hchk : Claims.Flock.check DO.parse xO w = true) {c : ℕ → Coin} (hc : shows s x xO c) {ms : List Msg}
    (hms : ∀ i < layoutOf.rounds x, ∀ m, layoutOf.held x s w i = some m → ms.getD i [] = m) :
    logs s x c ms = [] ∨ Claims.Recursion.SetupColl H D xO [] := by
  have hnil : layoutOf.calls x s w = [] := rfl
  rcases hP x st hst hf xO w hchk (by simp [hnil]) with h | h | h
  · exact Or.inl (List.eq_nil_iff_forall_not_mem.2 fun e he => by simpa [hnil] using h c hc ms hms e he)
  · simp [hnil, Claims.findColl] at h
  · exact Or.inr (hnil ▸ h)

/-- **`LogDecomp` with `PartLogs` at `layoutOf` leaves the inner setup no calls**: at an accepted inner replay at coins
each part's V* statement shows, on messages agreeing with what each part holds, every call of the inner statement's
setup is some part's inner check's (`logDecomp_setup`), and those are none (`partLogs_layoutOf_nil`), off a collision
among some part's V* setup evaluations. -/
theorem setupLog_nil_layoutOf {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {shape : Claims.Recursion.Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    (hLD : Claims.Recursion.VStar.LogDecomp H D (FlockDeploy.V false) shape logs)
    (hP : ∀ s, Claims.Recursion.VStar.PartLogs H D DO layoutOf shape shows logs s) {x : Msg}
    {st : Claims.Recursion.Stmt} (hst : D.parse x = some st) (hf : shape st.form) (c : ℕ → Coin) (ms : List Msg)
    (hacc : (run H (replay ms) (FlockDeploy.V false) (Claims.Recursion.xI D x) () fun i => c i.val).out = .acc)
    (hparts : ∀ p < D.nS x, ∃ (xO : Msg) (w : Claims.Flock.Wit), Claims.Flock.check DO.parse xO w = true ∧
      shows p x xO c ∧ ∀ i < layoutOf.rounds x, ∀ m, layoutOf.held x p w i = some m → ms.getD i [] = m) :
    innerSetupLog false H (Claims.Recursion.xI D x) = [] ∨
      ∃ p < D.nS x, ∃ xO, Claims.Recursion.SetupColl H D xO [] := by
  by_cases hcoll : ∃ p < D.nS x, ∃ xO, Claims.Recursion.SetupColl H D xO []
  · exact Or.inr hcoll
  left
  obtain ⟨s, spec, sts, hs, hlog⟩ := logDecomp_setup hLD hst hf c ms hacc
  unfold innerSetupLog
  rw [hs]
  refine List.eq_nil_iff_forall_not_mem.2 fun e he => ?_
  obtain ⟨p, hp, hep⟩ := hlog e he
  obtain ⟨xO, w, hchk, hc, hms⟩ := hparts p hp
  rcases partLogs_layoutOf_nil (hP p) hst hf hchk hc hms with h | h
  · rw [h] at hep
    exact List.not_mem_nil hep
  · exact hcoll ⟨p, hp, xO, h⟩

/-! ## What the reader reads of the inner statement

The inner statement as `readX` reads it, and Flock's statement check on it, which passes plain leaves at
hidden outputs without `--zk`, the one form the inner setup's circuit file can take (`InnerLeaf.check_of_innerOf`). -/

/-- What `readX` read of the inner statement: its decoding, its circuit file set up at the inner tags
(`RecVStar.innerOf`), and Flock's own setup and statement check on it, with a draw (`innerOk`). -/
theorem readX_inner {x : Msg} {r : XR} (h : readX x = some r) :
    FlockDeploy.decode r.w.inner = some r.innerRaw ∧
      Flock.RecVStar.innerOf r.innerRaw.circuit r.innerRaw.publicFile r.partition r.innerRaw.program = .ok r.dims ∧
      innerOk r.w.inner = .ok () := by
  replace h := readXPre_of_readX h
  unfold readXPre at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, -, ri, hri, sh, -, pa, -, dims, hdims, vs, -, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  obtain ⟨d, hd, hrest⟩ := FlockSoundness.Refine.ebind_ok (toOption_eq_some hdims)
  obtain ⟨u, hu, hp⟩ := FlockSoundness.Refine.ebind_ok hrest
  cases hp
  exact ⟨hri, hd, hu⟩

/-- A statement `parse` accepts is read by `readX`, and its inner bytes are the reading's. -/
theorem readX_of_parse {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) :
    ∃ r, readX x = some r ∧ xIOf x = r.w.inner := by
  have hx : xIOf x = st.inner := by unfold xIOf; rw [hst]; rfl
  obtain ⟨r', hr', rfl⟩ := Option.map_eq_some_iff.1 hst
  obtain ⟨hd, -⟩ := Option.filter_eq_some_iff.1 hr'
  obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 hd
  exact ⟨r, hr, hx⟩

theorem ins_circuitFile {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    s.read.I.circuitFile = w.circuit := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

/-- **An inner statement whose circuit file the inner setup accepts passes Flock's statement check only in the
plain-leaves form**, read by Flock's plain verifier: `RecVStar.innerOf` reads plain leaves off the file, so where
`ProvedScope.check` passes it the check ran without `--zk`, at hidden outputs and plain SHA-512 leaves
(`InnerLeaf.check_of_innerOf`), whatever `--statement` the inner statement names. -/
theorem check_plain_of_decode {inner : Msg} {ri : FlockDeploy.Raw} (hri : FlockDeploy.decode inner = some ri)
    {partition : Option Lean.Json} {dims : Flock.RecVStar.Inner}
    (hdims : Flock.RecVStar.innerOf ri.circuit ri.publicFile partition ri.program = .ok dims) {s : FlockDeploy.Ins}
    (hs : FlockDeploy.insOf false inner = some s) (hc : s.read.check = .ok ()) :
    s.read.zk = false ∧ s.read.tags.hiddenOutputs = true ∧ s.read.tags.merkleLeaf = "flock-leaf/sha512-unsalted" := by
  unfold FlockDeploy.insOf at hs
  obtain ⟨w, hw, hs⟩ := Option.bind_eq_some_iff.1 hs
  rw [hri] at hw
  obtain rfl := Option.some.inj hw
  split_ifs at hs with hz
  · cases hs
  change ri.ins.toOption = some s at hs
  have hcf := ins_circuitFile (toOption_eq_some hs)
  unfold Flock.Readable.Read.check at hc
  rw [hcf] at hc
  exact FlockSoundness.Discharge.InnerLeaf.check_of_innerOf hdims hc

/-- **The inner statement of every statement the reader parses passes Flock's statement check only in the
plain-leaves form**: `readX` sets the inner circuit file up at the inner tags (`check_plain_of_decode`). -/
theorem inner_check_plain {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) {s : FlockDeploy.Ins}
    (hs : FlockDeploy.insOf false (xIOf x) = some s) (hc : s.read.check = .ok ()) :
    s.read.zk = false ∧ s.read.tags.hiddenOutputs = true ∧ s.read.tags.merkleLeaf = "flock-leaf/sha512-unsalted" := by
  obtain ⟨r, hr, hx⟩ := readX_of_parse hst
  obtain ⟨hri, hdims, -⟩ := readX_inner hr
  rw [hx] at hs
  exact check_plain_of_decode hri hdims hs hc

end RecDeploy
