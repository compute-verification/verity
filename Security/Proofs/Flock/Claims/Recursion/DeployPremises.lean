import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.FlockKSRecord
import Proofs.Flock.Claims.Recursion.DeployScopeRuns

/-!
# The recursion's premises at `deployRec`

What the recursion's claims read of the compiled verifier, at `deployRec IV ivScope b`, beyond the code facts
`DeployFacts`, `DeploySessions` and `DeployMargin` prove there:

* **`ScopeMatches`** holds wherever the inner scope check matches the inner scope (`scopeMatches`), so with no
  hypothesis at the deployed plain inner record (`scopeMatches_deployed`): both are the deployed parser's
  (`Flock.RecScope`).
* **The V\* facts** (`CallsTrue`, `CallsLogColl`, `CallsOpenPlayed`, `VStarAccepts` at each shape) quantify over the
  compiled verifier's accepted runs. On such a run each V* statement is the one staged at the run's inner and staging
  rounds (`runVstar_staged`, from `run_round`), and the inner view is the inner verifier's run of the readers' messages
  at the run's inner coins. So each reduces to a fact about V*'s circuits and the readers alone, over the V* statements
  staged at any inner and staging messages and coins (`ReadersTrue` at the ones each part's scope check passes,
  `StagedPart`, which an accepted run gives: `stagedAll_run`), with no prover or run: `ReadersTrue` gives `CallsTrue`
  (`callsTrue`), with it `ReadersLogColl` gives `CallsLogColl` (`callsLogColl`), `ReadersOpenPlayed` gives
  `CallsOpenPlayed` (`callsOpenPlayed`), and `ReadersAccept shape` gives `VStarAccepts shape` (`vstarAccepts`); with
  `Refuses`, the three shapes' give `InnerAcceptsColl` (`innerAcceptsColl`). The readers' facts are the forms V*'s parts give: an
  opening only at the rounds the inner verifier's run plays, and its calls recorded only off a collision.
* **The recursion's KS over the deployed plain inner record** (`ksA_deployed`): `KS_of_flockKS_coll` at `deployRec`
  over Flock's plain verifier at its trimmed record (`deployPlainTrim`), every code fact of the compiled verifier
  proved or reduced as above, `Inner.KSAbort` from `plain_KSAbort` (the table bound), `PublicCoin` from
  `coinsShown_V`, and Flock's KS at the outer record (`deployZkRec`).
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay)
open FlockDeploy (Coin)

variable {W : Type}

/-! ## Scope -/

/-- **`ScopeMatches` at `deployRec`**, wherever the inner scope check `ivScope` matches the inner scope: `parse` and
`scopeCheck` are the deployed parser's (`RecScope.parseOf`, `RecScope.scopeCheckOf` at `decodeR`), so this is
`RecScope.scopeMatches`. -/
theorem scopeMatches (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
    (hI : Claims.ScopeMatches I.Scope ivScope) : Claims.Recursion.ScopeMatches (deployRec I.V ivScope b) I :=
  Flock.RecScope.scopeMatches _ I decodeR ivScope rfl rfl hI

/-- **`ScopeMatches` at `deployRec` over the deployed plain inner record**, with no hypothesis: its scope check is its
readable scope (`RecScope.scopeMatches_deployed`). -/
theorem scopeMatches_deployed (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (b : Brings) :
    Claims.Recursion.ScopeMatches (deployRec (FlockDeploy.V false) (FlockDeploy.deployPlain rel).scopeCheck b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) :=
  Flock.RecScope.scopeMatches_deployed rel _ decI decodeR rfl rfl

/-! ## The staged V* statements -/

/-- The inner and staging rounds as the compiled verifier shows them: each message `as` answered by its coin `cs`. -/
def roundsOf (as : List Msg) (cs : List Coin) : List (Msg × Out) := as.zip (cs.map fun c => .msg c.bytes)

/-- **The V\* statements staged at the inner and staging messages `as` and coins `cs`**: each part's. -/
noncomputable def stagedAll (st : Stage) (x : Msg) (as : List Msg) (cs : List Coin) : List Msg :=
  (List.range (nS x)).map (staged st x (roundsOf as cs))

/-- The calls the outer witnesses record. -/
def callsOf (rd : Readers) (x : Msg) (ws : List Claims.Flock.Wit) : List (Msg × Msg) :=
  (ws.mapIdx fun s w => rd.calls x s w).flatten

/-- **The inner verifier's run on the inner messages the outer witnesses give**, at the inner coins `cs`. -/
def innerRunAt (H : Msg → Msg) (IV : Claims.Verifier Coin) (rd : Readers) (x : Msg) (ws : List Claims.Flock.Wit)
    (cs : List Coin) : Claims.Trace :=
  run H (replay (rd.msgs x ws)) IV (xIOf x) () fun i => cs.getD i.val default

/-- A run's inner and staging messages. -/
def runAs (n : ℕ) (t : Claims.Trace) : List Msg := (t.msgs.take n).map Prod.fst

/-- A run's inner and staging coins. -/
def runCs {k : ℕ} (n : ℕ) (r : Fin k → Coin) : List Coin := (List.range n).map (coinSeq r)

theorem coins_getD {k : ℕ} (r : Fin k → Coin) {m n : ℕ} (h : m ≤ n) :
    (fun i : Fin m => coinSeq r i.val) = fun i => (runCs n r).getD i.val default := by
  funext i
  have := i.isLt
  simp [runCs, List.getD_eq_getElem?_getD, show i.val < n by omega]

/-- **The inner view is the inner verifier's run at a run's inner coins.** -/
theorem innerView_eq (H : Msg → Msg) (IV : Claims.Verifier Coin) (rd : Readers) (x : Msg) {k : ℕ}
    (r : Fin k → Coin) (ws : List Claims.Flock.Wit) {n : ℕ} (h : IV.k (xIOf x) ≤ n) :
    innerView H IV rd x r ws = (innerRunAt H IV rd x ws (runCs n r)).seen := by
  unfold innerView innerRunAt
  rw [coins_getD r h]

section Run

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV b.st).k x) → Coin)

theorem runAs_length (hacc : (run H P (V IV b.st) x ω r).out = .acc) :
    (runAs (IV.k (xIOf x) + kS x) (run H P (V IV b.st) x ω r)).length = IV.k (xIOf x) + kS x := by
  have hK := run_acc_length H IV b.st x P ω r hacc
  have hM := roundsMargin_V H IV b.st x P ω r hacc
  simp only [runAs, List.length_map, List.length_take]
  omega

theorem runCs_length : (runCs (IV.k (xIOf x) + kS x) r).length = IV.k (xIOf x) + kS x := by
  simp [runCs]

/-- **An accepted run's inner and staging rounds are its messages answered by its coins.** -/
theorem take_roundsOf (hacc : (run H P (V IV b.st) x ω r).out = .acc) :
    (run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x) =
      roundsOf (runAs (IV.k (xIOf x) + kS x) (run H P (V IV b.st) x ω r)) (runCs (IV.k (xIOf x) + kS x) r) := by
  have hl := runAs_length H IV b x P ω r hacc
  have hM := roundsMargin_V H IV b.st x P ω r hacc
  have h := innerPairs_run H IV b.st x P ω r hacc (off := IV.k (xIOf x) + kS x) le_rfl (by omega) []
    (cs := runCs (IV.k (xIOf x) + kS x) r) (List.take_of_length_le (by simp [runCs]))
  rw [List.append_nil] at h
  rw [← h]
  show innerPairs _ (runAs _ _) (runCs _ r) = _
  unfold innerPairs roundsOf
  rw [List.take_of_length_le hl.le, List.take_of_length_le (runCs_length IV b x r).le]

end Run

/-- **On an accepted run, the V\* statements are the ones staged at its inner and staging rounds**: every part's staged
numbers are its numbers (`run_round`), so `vstarAt` takes the staged statement. -/
theorem runVstar_staged (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
    {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((deployRec I.V ivScope b).V.k x) → Coin)
    (hacc : (run H P (deployRec I.V ivScope b).V x ω r).out = .acc) :
    Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r =
      stagedAll b.st x (runAs (I.V.k (xIOf x) + kS x) (run H P (V I.V b.st) x ω r))
        (runCs (I.V.k (xIOf x) + kS x) r) := by
  show vstar b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) = _
  have ht := take_roundsOf H I.V b x P ω r hacc
  rw [ht]
  unfold vstar stagedAll
  refine List.map_congr_left fun s hs => ?_
  rw [List.mem_range] at hs
  have h := (run_round H I.V b.st x P ω r hacc hs rfl rfl).1
  rw [ht] at h
  unfold vstarAt
  split_ifs
  rfl

/-! ## The staging on an accepted run, per part -/

/-- **Part `s`'s V\* statement as V\* stages it**: `xO` is `staged st x ms s` at some list of messages, its setup
decodes, and its read's check passes. -/
def StagedPart (st : Stage) (s : ℕ) (x xO : Msg) : Prop :=
  ∃ (ms : List (Msg × Out)) (i : RecIns) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt),
    xO = staged st x ms s ∧ setupO xO = some (i, spec, sts) ∧ i.base.read.check = .ok ()

section StagedRun

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

/-- **An accepted run stages every part**: session `s` decided (`run_decides`) and did not refuse, so its outer
statement's setup decoded and its read checked (`nextO_decide`, `actO_decide`, `scope_of_checks`), and that statement
is part `s`'s V\* statement at the run's inner messages (`ract_sess`, `innerPairs_run`). -/
theorem stagedPart_run (hacc : (run H P (V IV st) x ω r).out = .acc) {s : ℕ} (hs : s < nS x) :
    StagedPart st s x (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) := by
  obtain ⟨i, f, hi, ⟨j, xO, as', cs', ins, spec, sts, link, c₁, mids, hr, hO⟩, hn⟩ :=
    run_decides H IV st x P ω r hacc hs
  rw [next_sess H hr] at hn
  have hacc' : ((nextO xO as' cs').eval H).1 = .acc := by
    rcases nextO_decide_out H hO with ho | ho
    · exact ho
    · rw [ho] at hn
      exact absurd rfl hn
  obtain ⟨⟨spec', sts', -, -, hc⟩, -⟩ := (nextO_decide H hO).1 hacc'
  obtain ⟨i', hi', hib⟩ := actO_decide hO
  obtain ⟨hk, -, -⟩ := ract_sess_locate hr
  obtain ⟨-, -, hxO⟩ := ract_sess hr
  have hK := run_acc_length H IV st x P ω r hacc
  have hlen : (((run H P (V IV st) x ω r).msgs.take (i + 1)).map Prod.fst).length = i + 1 := by
    simp only [List.length_map, List.length_take]
    omega
  rw [hlen] at hk
  have hip := innerPairs_run H IV st x P ω r hacc (off := i + 1) hk.le (by omega) []
    (range_map_take (coinSeq r) hk.le)
  rw [List.append_nil] at hip
  rw [hip] at hxO
  subst hxO
  refine ⟨_, i', spec, sts, rfl, hi', ?_⟩
  rw [hib]
  exact scope_of_checks hc

end StagedRun

/-- **An accepted run stages every part, at its inner and staging rounds**: `stagedPart_run` at `stagedAll`
(`take_roundsOf`). -/
theorem stagedAll_run (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (x : Msg) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (r : Fin ((V IV b.st).k x) → Coin) (hacc : (run H P (V IV b.st) x ω r).out = .acc) :
    ∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x (runAs (IV.k (xIOf x) + kS x) (run H P (V IV b.st) x ω r))
      (runCs (IV.k (xIOf x) + kS x) r)).getD s []) := by
  intro s hs
  rw [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hs, Option.map_some,
    Option.getD_some, ← take_roundsOf H IV b x P ω r hacc]
  exact stagedPart_run H IV b.st x P ω r hacc hs

/-! ## V*'s circuits and the readers -/

section Readers

variable {Col T : Type}

/-- **The V\* witnesses record only true evaluations** (a code fact about V*'s circuits and the readers): outer
witnesses in the outer relation at the V* statements staged at inner and staging messages `as` and coins `cs`, each
part's staged as V\* stages it (`StagedPart`: its setup decodes and its read's scope check passes, as on every
accepted run, `stagedAll_run`), record only true evaluations of `H`. At SHA-512, V*'s circuits compute each recorded
call. Off `StagedPart` the staged statement's circuit and tags are the staging message's, which the parts' readers
don't read. -/
def ReadersTrue (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) :
    Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws → ∀ e ∈ callsOf b.rd x ws, H e.1 = e.2

/-- **The readers record the inner verifier's calls, off a collision** (a code fact about V*'s circuits and the
readers): at outer witnesses in the outer relation at the V* statements staged at `as` and `cs`, each part's staged as
V\* stages it (`StagedPart`, as in `ReadersTrue`), whose recorded calls are true evaluations, the inner verifier's
calls in its run of the readers' messages at the inner coins `cs` are among the recorded calls, or two recorded calls
collide. Where two parts hold different messages at one round, the run's
calls on the message the readers take aren't the other part's, so V*'s parts give this only off a collision. -/
def ReadersLogColl (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) :
    Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
        (∀ e ∈ (innerRunAt H IV b.rd x ws cs).vLog, e ∈ callsOf b.rd x ws) ∨
          ∃ u v, Claims.findColl (callsOf b.rd x ws) = some (u, v) ∧ Claims.Collision H u v

/-- **The readers record each played inner message's opening** (a code fact about V*'s circuits and the readers): at
outer witnesses in the outer relation at the V* statements staged at `as` and `cs`, each part's staged as V\* stages it
(`StagedPart`), at each round `j` the inner verifier's run of the readers' messages plays, some recorded calls open
the commit string `as[j]` to the readers' message `j` under `opening`. A run that rejects early has message `[]` at
the rounds it doesn't play, which no part opens. -/
def ReadersOpenPlayed (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T)
    (opening : Claims.Recursion.Opening) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      ∀ j < (innerRunAt H IV b.rd x ws cs).msgs.length, ∃ es, es ⊆ callsOf b.rd x ws ∧
        opening.opens (as.getD j []) (((innerRunAt H IV b.rd x ws cs).msgs.map Prod.fst).getD j []) es

/-- **V\*'s circuits compute the inner verifier at the shape `shape`, off a collision** (a code fact about V*'s circuits
and the readers): at a statement whose form has the shape, outer witnesses in the outer relation at the V* statements
staged at `as` and `cs`, each part's staged as V\* stages it (`StagedPart`), whose recorded calls are true
evaluations, give the inner verifier's acceptance of the readers' messages at the inner coins `cs`, or two of their
recorded calls collide. -/
def ReadersAccept (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ (x : Msg) (s : Claims.Recursion.CompiledStatement), parse x = some s → shape s.form →
    ∀ (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
      cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
        Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
        (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
          (innerRunAt H IV b.rd x ws cs).out = .acc ∨
            ∃ u v, Claims.findColl (callsOf b.rd x ws) = some (u, v) ∧ Claims.Collision H u v

variable (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
  (DO : Claims.Flock.Deploy Coin Col T)

/-- **`CallsTrue` at `deployRec`** from the readers' facts. -/
theorem callsTrue (hT : ReadersTrue H I.V b DO) : Claims.Recursion.CallsTrue H (deployRec I.V ivScope b) I DO := by
  intro decO x Ω P ω r c ws hg
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  exact hT x _ _ ws hl hc (stagedAll_run H I.V b x P ω r hg.1) ho

/-- **`CallsLogColl` at `deployRec`** from the readers' facts. -/
theorem callsLogColl (hT : ReadersTrue H I.V b DO) (hL : ReadersLogColl H I.V b DO) :
    Claims.Recursion.CallsLogColl H (deployRec I.V ivScope b) I DO := by
  intro decO x Ω P ω r c ws hg
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hrun : Claims.Recursion.innerRun H (deployRec I.V ivScope b) I x r ws =
      innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r) := by
    show run H (replay ((innerView H I.V b.rd x r ws).msgs.map Prod.fst)) I.V (xIOf x) ()
      (fun i => coinSeq r i.val) = _
    unfold innerView
    rw [run_replay_seen, coins_getD r (show I.V.k (xIOf x) ≤ I.V.k (xIOf x) + kS x by omega)]
    rfl
  rw [hrun]
  have hsp := stagedAll_run H I.V b x P ω r hg.1
  exact hL x _ _ ws hl hc hsp ho (hT x _ _ ws hl hc hsp ho)

/-- **`CallsOpenPlayed` at `deployRec`** from the readers' facts: the inner view is the inner verifier's run of the
readers' messages, which plays at most the inner rounds (`run_length_le`). -/
theorem callsOpenPlayed (opening : Claims.Recursion.Opening) (hO : ReadersOpenPlayed H I.V b DO opening) :
    Claims.Recursion.CallsOpenPlayed H (deployRec I.V ivScope b) I DO opening := by
  intro decO x Ω P ω r c ws hg j hj
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hv : (deployRec I.V ivScope b).innerView H x r ws =
      (innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r)).seen :=
    innerView_eq H I.V b.rd x r ws (Nat.le_add_right _ _)
  rw [hv] at hj ⊢
  obtain ⟨es, hes, hop⟩ := hO x _ _ ws hl hc (stagedAll_run H I.V b x P ω r hg.1) ho j hj
  refine ⟨es, hes, ?_⟩
  have hjk : j < I.V.k (xIOf x) :=
    Nat.lt_of_lt_of_le hj (Claims.Recursion.run_length_le H _ I.V (xIOf x) () _)
  have hjn : j < I.V.k (xIOf x) + kS x := Nat.lt_of_lt_of_le hjk (Nat.le_add_right _ _)
  have ha : ((run H P (deployRec I.V ivScope b).V x ω r).msgs.map Prod.fst).getD j [] =
      (runAs (I.V.k (xIOf x) + kS x) (run H P (V I.V b.st) x ω r)).getD j [] := by
    show ((run H P (V I.V b.st) x ω r).msgs.map Prod.fst).getD j [] = _
    simp only [runAs, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_take_of_lt hjn]
  rw [ha]
  exact hop

/-- **`VStarAccepts shape` at `deployRec`** from V*'s circuits and the readers at that shape. -/
theorem vstarAccepts (shape : Claims.Recursion.Form → Prop) (hA : ReadersAccept H I.V b DO shape) :
    Claims.Recursion.VStarAccepts H (deployRec I.V ivScope b) I DO shape := by
  intro x s hs hf Ω P ω r hacc ws ho ht
  rw [runVstar_staged H I ivScope b P ω r hacc] at ho
  have hl := runAs_length H I.V b x P ω r hacc
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hv : (deployRec I.V ivScope b).innerView H x r ws =
      (innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r)).seen :=
    innerView_eq H I.V b.rd x r ws (Nat.le_add_right _ _)
  rw [hv]
  exact hA x s hs hf _ _ ws hl hc (stagedAll_run H I.V b x P ω r hacc) ho ht

/-- **`InnerAcceptsColl` at `deployRec`** from the readers' facts and V*'s circuits at the dense, holo and sparse
shapes, with `refuses`. -/
theorem innerAcceptsColl (hT : ReadersTrue H I.V b DO)
    (hD : ReadersAccept H I.V b DO Claims.Recursion.Form.Dense)
    (hH : ReadersAccept H I.V b DO Claims.Recursion.Form.Holo)
    (hS : ReadersAccept H I.V b DO Claims.Recursion.Form.Sparse) :
    Claims.Recursion.InnerAcceptsColl H (deployRec I.V ivScope b) I DO :=
  Claims.Recursion.innerAcceptsColl_of_vstar (refuses I.V ivScope b H) (callsTrue H I ivScope b DO hT)
    (vstarAccepts H I ivScope b DO _ hD) (vstarAccepts H I ivScope b DO _ hH) (vstarAccepts H I ivScope b DO _ hS)

end Readers

/-! ## The recursion's KS over the deployed plain inner record -/

/-- **The recursion's KS at `deployRec` over the deployed plain inner record**, the trimmed one the claims read, at
SHA-512, at every inner decoder within the radius (`KSA`, at the errors the proof gives): `KS_of_flockKS_coll` with
every code fact of the compiled verifier proved at `deployRec` or reduced to V*'s circuits and the readers
(`ReadersTrue`, `ReadersLogColl`, `ReadersOpenPlayed`, `ReadersAccept` at the three shapes), `Inner.KSAbort` from the
plain record's table bound (`plain_KSAbort`), `PublicCoin` from `coinsShown_V`, and Flock's KS at the outer record. -/
theorem ksA_deployed (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Sparse)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) :=
  let I := Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI
  Claims.Recursion.KS_of_flockKS_coll FlockDeploy.sha512 (deployRec I.V ivScope b) I (deployZkRec relO) opening
    FlockDeploy.decCoin hF (FlockDeploy.KS.plain_KSAbort rel htab decI hdec)
    (refuses I.V ivScope b FlockDeploy.sha512) (innerRounds I ivScope b) (sessionNums I.V ivScope b relO)
    (tableErrNonneg I.V ivScope b) (sessionRounds I ivScope b relO) (sessionAccepts I.V ivScope b)
    (sessionRuns_sha512 I ivScope b relO) (innerViewRun FlockDeploy.sha512 I ivScope b (deployZkRec relO))
    (callsTrue FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT)
    (callsLogColl FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT hL)
    (callsOpenPlayed FlockDeploy.sha512 I ivScope b (deployZkRec relO) opening hO)
    (innerAcceptsColl FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT hD hH hS)
    (roundsMargin_deployRec_V false FlockDeploy.sha512 I rfl ivScope b)
    (FlockDeploy.coinsShown_V FlockDeploy.sha512 false) (callsNil I.V ivScope b FlockDeploy.sha512)

/-- **`KSFlock` at `deployRec`**: the recursion's KS over the deployed plain inner record (trimmed), its V* sessions at
the masked outer record (`deployZkRec`), at every inner decoder that meets the spec: `ksA_deployed`, as `KSA` is `KS`.
Its level (`Claims.Recursion.lam`) is about 65 bits per SHA-512 call at the deployed shape (one inner table at `m = 35`,
`γ`'s weight at `IVStar` at `coverW`), below the 100-bit floor and the `2^-128` target under every weight.
Route 2C's bound λ₂ (`lamLin2`) evaluates to at least 100 at the deployed records (`RecDeploy.lamLin2_deployed_ge_dep`),
but `KSLin2` at the deployment's dense statements (`KSLin2On` at `RecDeploy.DenseAt`) only follows from premises still
open (`RecDeploy.ksLin2_deployed_open`), Holo statements being candidate 3's own model, and `BitsLin2` at those
statements follows from it (`bitsLin2On_of_ksLin2On`). -/
theorem ksFlock_deployed (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec relO) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec =>
    ksA_deployed rel relO ivScope b opening hF htab hT hL hO hD hH hS decI hdec

end RecDeploy
