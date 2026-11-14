module

public import Proofs.Properties.Verity.SampledTwo.CutHeld
public import Proofs.Properties.Verity.Sampled.Device

/-!
# The device's history at a two-session play: the two runs that printed the play's log

The cut (`Cut.cutBound`) takes two facts about the device's history at each play: its runs on the parameters' two
windows printed the play's log (`PrintsTwo`, through `BindsOfPrints`), and it signed what the play's signatures are on
(`SignsPlay`). This file builds that history from the play (`dev₂`) and proves both at it.

* **The runs** (`runsAt`, `ledgerAt`): a session's run is the one-stage history's (`Sampled.Device.dev`) at that
  session's releases, kept where its release log reads as coin trees (`treesOk`), which `PrintsOn` asks of every run
  and which a prover's frame can break (an opening's `after` of another length than a digest's).
* **The history** (`logs`, `dev₂`): on one window (`oneWindow`), the two sessions are one (a ledger opens a window's
  root once), and the history is the first setup's run at the whole log. On two, the play's log is the first session's
  releases, `log.take (cut log)`, then the interior's, `log.drop (cut log)`; the history holds the first setup's run
  at the first and the second setup's at the second, with both setups' ledger lines.
* **The device printed the log** (`printsTwo_dev₂`, `binds_dev₂`): each run prints its part
  (`Sampled.Device.prints_dev`, `holds_dev`), and where the setups' ledger lines open neither session's run on the
  other's window (`Apart`), no run prints the other part's window. `binds_dev₂` is `Binds` at the history, from
  `CutHeld.binds_of_accepts`. `Apart` is taken as a hypothesis, not proved; `DeviceOrder` replaces it on the cut's
  path.
* **The history holds both runs** (`holdsTwo_dev₂`): `PrintsTwo` speaks of the runs the history holds, and holds of a
  history holding none. On two windows, where the first session's releases hold no tree's hello (`NoHello`), each part
  fits its setup's layout (`Sampled.Device.Fits`) and each printed run reads as coin trees (`TreesRead`), the history
  holds both runs. The hello condition is what lets the interior's run print from the empty state: an opening names its
  tree by its place among the trees the run sampled (`Sampled.Device.printLog`), and `PrintsOn` reads a run's lines
  from no tree (`Prints []`).
* **What the device signs** (`Deployment`, `Deployment.signer`, `signsPlay_dev₂`, `cut_dev₂`): a two-session deployment
  is two one-stage deployments, both under the parameters' key (`keyOf`). Its signer signs before any coin the ledger
  lines the request fixes, and at the close the records and closes of the runs the history holds and the ledger lines
  at the close, so at a vocabulary whose signer is it the history signed what the play's signatures are on.
  `cut_dev₂` is the cut at this history.

**Keys.** The history takes two setups whose keys may differ, so `printsTwo_dev₂` holds where each stage's command
makes its own key. `SignsPlay` asks every signature of the play to be signed under the one key `keyOf θ`, so the
deployment runs both sessions under it.

**The step that forces one play.** A history holds earlier runs and ledger lines only on other windows: `PrintsOn`
asks every run its ledger opens on a play's window to print that play's log, and `SignsPlay` reads every signature
under the key against this history alone. On the play's own two windows this history holds exactly this play's runs,
so a genuine signature from an earlier period in the evidence is the forger's (the module docstring of `Cut`). That
is the step the key pins replace.
-/

@[expose] public section

namespace Verity.SampledTwo.Sessions

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Device (RunSetup PrintState printLog runOf Fits)

/-! ## A session's run -/

/-- Whether a run's release log reads as coin trees (`coinTreesOf`). -/
def treesOk (r : Properties.Challenger.Run) : Bool :=
  match coinTreesOf r.log with
  | .ok _ => true
  | .error _ => false

theorem treesOk_ok {r : Properties.Challenger.Run} (h : treesOk r = true) : ∃ ts, coinTreesOf r.log = .ok ts := by
  unfold treesOk at h
  split at h
  · rename_i ts hts
    exact ⟨ts, hts⟩
  · cases h

/-- A play that ended with the releases `log`, its other parts empty: what the one-stage history reads of a play is
its log and the setup. -/
def ex (log : List Release) : Exchange :=
  ⟨.empty, #[], .empty, .empty, log, .empty, [], [], .empty, #[]⟩

/-- **The run of the setup `S` that released `log`, under `pk`**: the one-stage history's (`Sampled.Device.dev`), where
its release log reads as coin trees (`treesOk`). -/
noncomputable def runsAt (S : RunSetup) (log : List Release) (pk : ByteArray) : List Properties.Challenger.Run :=
  ((Verity.Sampled.Device.dev (fun _ => S) (ex log)).runs pk).filter treesOk

open Classical in
/-- The setup's ledger lines, under its key. -/
noncomputable def ledgerAt (S : RunSetup) (pk : ByteArray) : List Json := if pk = S.key then S.ledger else []

theorem mem_ledgerAt {S : RunSetup} {pk : ByteArray} {x : Json} (h : x ∈ ledgerAt S pk) : x ∈ S.ledger := by
  unfold ledgerAt at h
  split at h
  · exact h
  · cases h

theorem runsAt_id {S : RunSetup} {log : List Release} {pk : ByteArray} {r : Properties.Challenger.Run}
    (hr : r ∈ runsAt S log pk) : r.id = S.id := by
  obtain ⟨hr, -⟩ := List.mem_filter.1 hr
  simp only [Verity.Sampled.Device.dev] at hr
  split at hr
  · split at hr
    · rw [List.mem_singleton] at hr
      subst hr
      rfl
    · cases hr
  · cases hr

/-- **A run kept prints its releases** (`PrintsOn`'s three parts at one run): its lines print `log` after its head
(`Sampled.Device.prints_dev`), its release log reads as coin trees, and where v2 reads it as its coin commitment's, it
released of each family after each root the coins `log` shows (`Sampled.Device.holds_dev`). -/
theorem run_facts {S : RunSetup} {log : List Release} {pk : ByteArray} {r : Properties.Challenger.Run}
    (hr : r ∈ runsAt S log pk) :
    Verity.Sampled.Lines.Prints [] r.log.tail log ∧ (∃ ts, coinTreesOf r.log = .ok ts) ∧
      ((∃ c roots, r.commitment = some c ∧ releasedRootsOf c r.log = .ok roots) →
        ∀ label after, Verity.Sampled.Lines.Holds (r.released label after)
          (Verity.Sampled.Lines.freshOf label after log)) := by
  obtain ⟨hr, ht⟩ := List.mem_filter.1 hr
  exact ⟨Verity.Sampled.Device.prints_dev (fun _ => S) (ex log) hr, treesOk_ok ht,
    fun ⟨_, _, hc, hroots⟩ => Verity.Sampled.Device.holds_dev (fun _ => S) (ex log) hr hc hroots⟩

/-! ## The history -/

/-- Whether the parameters' two windows are one (`windowsOf`); one where they name none. -/
def oneWindow (θ : ByteArray) : Bool :=
  match windowsOf θ with
  | some (w₁, w₂) => decide (w₁ = w₂)
  | none => true

theorem oneWindow_true {θ : ByteArray} {w₁ w₂ : String} (hw : windowsOf θ = some (w₁, w₂)) (h : w₁ = w₂) :
    oneWindow θ = true := by
  simp [oneWindow, hw, h]

theorem oneWindow_false {θ : ByteArray} {w₁ w₂ : String} (hw : windowsOf θ = some (w₁, w₂)) (h : w₁ ≠ w₂) :
    oneWindow θ = false := by
  simp [oneWindow, hw, h]

/-- **Each session's releases at the parameters `θ`**: on one window, the whole log and none; on two, the first
`cut log` releases, then the rest. -/
def logs (cut : List Release → ℕ) (θ : ByteArray) (log : List Release) : List Release × List Release :=
  bif oneWindow θ then (log, []) else (log.take (cut log), log.drop (cut log))

/-- The runs the history holds under `pk`: on one window (`one`), the first setup's at the first part; on two, the
first's at the first part and the second's at the second. -/
noncomputable def runsOf (S₁ S₂ : RunSetup) (one : Bool) (log₁ log₂ : List Release) (pk : ByteArray) :
    List Properties.Challenger.Run :=
  bif one then runsAt S₁ log₁ pk else runsAt S₁ log₁ pk ++ runsAt S₂ log₂ pk

/-- The ledger lines the history holds under `pk`: the first setup's on one window, both setups' on two. -/
noncomputable def ledgerOf (S₁ S₂ : RunSetup) (one : Bool) (pk : ByteArray) : List Json :=
  bif one then ledgerAt S₁ pk else ledgerAt S₁ pk ++ ledgerAt S₂ pk

/-- **The device's history at the two-session play that ended with `o`**, of the setups `S₁ o` (the first stage's
session) and `S₂ o` (the interior's), the first session's releases the first `cut o.log` of the play's log: the runs
that released each session's releases (`runsOf`, `logs`) and the setups' ledger lines (`ledgerOf`). -/
noncomputable def dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (o : Exchange) : History :=
  ⟨runsOf (S₁ o) (S₂ o) (oneWindow o.params) (logs cut o.params o.log).1 (logs cut o.params o.log).2,
    ledgerOf (S₁ o) (S₂ o) (oneWindow o.params)⟩

/-! ## The device printed the play's log -/

/-- The device's ledger `ledger` opens window `w` for the run `id`, withheld (`OpensOn` at a ledger). -/
def OpensIn (ledger : List Json) (w id : String) : Prop :=
  OpensOn ⟨fun _ => [], fun _ => ledger⟩ ByteArray.empty w id

theorem opensIn_of_opensOn {h : History} {pk : ByteArray} {w id : String} (ho : OpensOn h pk w id) {L : List Json}
    (hL : ∀ x ∈ h.ledger pk, x ∈ L) : OpensIn L w id := by
  unfold OpensIn OpensOn at *
  obtain ⟨line, hl, rest⟩ := ho
  exact ⟨line, hL line hl, rest⟩

/-- **No session's run is opened on the other's window**: the two setups' ledger lines don't open the second setup's
run on `w₁`, nor the first's on `w₂`. It is taken as a hypothesis, not proved, and it is no lemma of a device that
opens each window as a prover's request names it: it fails where the request names the two windows the other way
round. `DeviceOrder`'s `LedgerInOrder`, which names no window, replaces it on the cut's path. -/
def Apart (S₁ S₂ : RunSetup) (w₁ w₂ : String) : Prop :=
  ¬ OpensIn (S₁.ledger ++ S₂.ledger) w₁ S₂.id ∧ ¬ OpensIn (S₁.ledger ++ S₂.ledger) w₂ S₁.id

/-- A history whose runs are a setup's run at `log` prints `log` on every window. -/
theorem printsOn_one {h : History} {S : RunSetup} {log : List Release} {w : String}
    (hr : ∀ pk r, r ∈ h.runs pk → r ∈ runsAt S log pk) : PrintsOn h w log :=
  fun pk r hr' _ => run_facts (hr pk r hr')

/-- A history whose runs are a setup's run at `log` or another's, where its ledger doesn't open the other's run on `w`,
prints `log` on `w`. -/
theorem printsOn_part {h : History} {S S' : RunSetup} {log log' : List Release} {w : String}
    (hr : ∀ pk r, r ∈ h.runs pk → r ∈ runsAt S log pk ∨ r ∈ runsAt S' log' pk)
    (hno : ∀ pk, ¬ OpensOn h pk w S'.id) : PrintsOn h w log := by
  intro pk r hr' ho
  rcases hr pk r hr' with h1 | h2
  · exact run_facts h1
  · rw [runsAt_id h2] at ho
    exact absurd ho (hno pk)

theorem logs_one {cut : List Release → ℕ} {θ : ByteArray} {log : List Release} (h : oneWindow θ = true) :
    logs cut θ log = (log, []) := by
  simp only [logs, h, Bool.cond_true]

theorem logs_two {cut : List Release → ℕ} {θ : ByteArray} {log : List Release} (h : oneWindow θ = false) :
    logs cut θ log = (log.take (cut log), log.drop (cut log)) := by
  simp only [logs, h, Bool.cond_false]

theorem runs_two (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {o : Exchange}
    (h : oneWindow o.params = false) (pk : ByteArray) :
    (dev₂ S₁ S₂ cut o).runs pk =
      runsAt (S₁ o) (o.log.take (cut o.log)) pk ++ runsAt (S₂ o) (o.log.drop (cut o.log)) pk := by
  show runsOf (S₁ o) (S₂ o) (oneWindow o.params) (logs cut o.params o.log).1 (logs cut o.params o.log).2 pk = _
  simp only [logs_two h, h, runsOf, Bool.cond_false]

theorem ledger_two (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {o : Exchange}
    (h : oneWindow o.params = false) (pk : ByteArray) :
    ∀ x ∈ (dev₂ S₁ S₂ cut o).ledger pk, x ∈ (S₁ o).ledger ++ (S₂ o).ledger := by
  intro x hx
  change x ∈ ledgerOf (S₁ o) (S₂ o) (oneWindow o.params) pk at hx
  rw [h] at hx
  rcases List.mem_append.1 hx with hx | hx
  · exact List.mem_append.2 (.inl (mem_ledgerAt hx))
  · exact List.mem_append.2 (.inr (mem_ledgerAt hx))

/-- **The device printed the play's log in its two sessions** (`PrintsTwo`), at every play: on one window, the one run
prints the whole log; on two, the first `cut o.log` releases and the rest, each printed by its setup's run, where the
ledger opens neither session's run on the other's window (`Apart`). `Apart` is taken, not proved. -/
theorem printsTwo_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (o : Exchange) {w₁ w₂ : String}
    (hw : windowsOf o.params = some (w₁, w₂)) (hA : w₁ ≠ w₂ → Apart (S₁ o) (S₂ o) w₁ w₂) :
    PrintsTwo (dev₂ S₁ S₂ cut o) w₁ w₂ o.log := by
  unfold PrintsTwo
  by_cases hww : w₁ = w₂
  · rw [if_pos hww]
    have h1 := oneWindow_true hw hww
    refine printsOn_one (S := S₁ o) fun pk r hr => ?_
    have e : (dev₂ S₁ S₂ cut o).runs pk = runsAt (S₁ o) o.log pk := by
      show runsOf (S₁ o) (S₂ o) (oneWindow o.params) (logs cut o.params o.log).1 (logs cut o.params o.log).2 pk = _
      simp only [logs_one h1, h1, runsOf, Bool.cond_true]
    rwa [e] at hr
  · rw [if_neg hww]
    have h2 := oneWindow_false hw hww
    obtain ⟨hA₁, hA₂⟩ := hA hww
    refine ⟨_, _, (List.take_append_drop (cut o.log) o.log).symm, ?_, ?_⟩
    · refine printsOn_part (S := S₁ o) (S' := S₂ o) (log' := o.log.drop (cut o.log)) (fun pk r hr => ?_)
        fun pk ho => ?_
      · rw [runs_two S₁ S₂ cut h2] at hr
        exact List.mem_append.1 hr
      · exact hA₁ (opensIn_of_opensOn ho (ledger_two S₁ S₂ cut h2 pk))
    · refine printsOn_part (S := S₂ o) (S' := S₁ o) (log' := o.log.take (cut o.log)) (fun pk r hr => ?_)
        fun pk ho => ?_
      · rw [runs_two S₁ S₂ cut h2] at hr
        exact (List.mem_append.1 hr).symm
      · exact hA₂ (opensIn_of_opensOn ho (ledger_two S₁ S₂ cut h2 pk))

/-- **`Binds` at the two-session history** (`CutHeld.binds_of_accepts`): at every play after `θ` the history printed
its log on the parameters' two windows (`printsTwo_dev₂`), where the setups' ledger lines open neither session's run
on the other's window (`Apart`). `Apart` is taken, not proved. -/
theorem binds_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (θ : ByteArray)
    (hA : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      Apart (S₁ o) (S₂ o) w₁ w₂) :
    Binds (dev₂ S₁ S₂ cut) θ :=
  binds_of_accepts (dev₂ S₁ S₂ cut) θ fun o hθ w₁ w₂ hw =>
    printsTwo_dev₂ S₁ S₂ cut o (by rw [hθ]; exact hw) (hA o hθ w₁ w₂ hw)

/-! ## The history holds both runs -/

/-- Whether a frame is a coin tree's hello. -/
def isHello : Verity.Frame → Bool
  | .hello _ => true
  | _ => false

/-- **The releases `log` hold no coin tree's hello.** -/
def NoHello (log : List Release) : Prop := ∀ x ∈ log, isHello x.frame = false

/-- **The setup's printed run at `log` reads as coin trees**, wherever the printer prints `log`. -/
def TreesRead (S : RunSetup) (log : List Release) : Prop :=
  ∀ lines, printLog S.time PrintState.init log = some lines → treesOk (runOf S log lines) = true

/-- **The printer prints a log's every prefix**, at any times and from any state with the same trees. -/
theorem printLog_prefix (t t' : ℕ → ℕ) :
    ∀ (l₁ l₂ : List Release) (st st' : PrintState) {ls : List String}, st.trees = st'.trees →
      printLog t st (l₁ ++ l₂) = some ls → (printLog t' st' l₁).isSome = true
  | [], _, _, _, _, _, _ => rfl
  | ⟨.fresh label, b, cs⟩ :: l₁, l₂, st, st', ls, htr, h => by
    simp only [List.cons_append, printLog] at h
    split at h
    · rename_i hcs
      obtain ⟨ls', hls, -⟩ := Option.map_eq_some_iff.1 h
      have ih := printLog_prefix t t' l₁ l₂ (st.coin label) (st'.coin label) htr hls
      simp only [printLog, if_pos hcs]
      simpa using ih
    · cases h
  | ⟨.hello spec, ν, cs⟩ :: l₁, l₂, st, st', ls, htr, h => by
    simp only [List.cons_append, printLog] at h
    split at h
    · rename_i hcs
      obtain ⟨ls', hls, -⟩ := Option.map_eq_some_iff.1 h
      have ih := printLog_prefix t t' l₁ l₂ _ (st'.tree ((Verity.Sampled.Lines.coinBytes cs).extract 0 64) spec)
        (by show st.trees ++ _ = st'.trees ++ _; rw [htr]) hls
      simp only [printLog, if_pos hcs]
      simpa using ih
    · cases h
  | ⟨.opening i s n, b, cs⟩ :: l₁, l₂, st, st', ls, htr, h => by
    simp only [List.cons_append, printLog] at h
    split at h
    · cases h
    · rename_i root spec hts
      split at h
      · cases h
      · rename_i stream hst
        split at h
        · rename_i hcs
          obtain ⟨ls', hls, -⟩ := Option.map_eq_some_iff.1 h
          have ih := printLog_prefix t t' l₁ l₂ (st.round i s) (st'.round i s) htr hls
          have hts' : st'.trees[i]? = some (root, spec) := htr ▸ hts
          simp only [printLog, hts', hst, if_pos hcs]
          simpa using ih
        · cases h

/-- **The printer prints what follows releases with no hello from any state with the same trees**: the releases before
it sample no tree, so its openings name the same trees. -/
theorem printLog_suffix (t t' : ℕ → ℕ) :
    ∀ (l₁ l₂ : List Release) (st st' : PrintState) {ls : List String}, NoHello l₁ → st.trees = st'.trees →
      printLog t st (l₁ ++ l₂) = some ls → (printLog t' st' l₂).isSome = true
  | [], l₂, st, st', ls, _, htr, h =>
    printLog_prefix t t' l₂ [] st st' (ls := ls) htr (by rw [List.append_nil]; exact h)
  | ⟨.fresh label, b, cs⟩ :: l₁, l₂, st, st', ls, hno, htr, h => by
    simp only [List.cons_append, printLog] at h
    split at h
    · obtain ⟨ls', hls, -⟩ := Option.map_eq_some_iff.1 h
      exact printLog_suffix t t' l₁ l₂ (st.coin label) st'
        (fun x hx => hno x (List.mem_cons_of_mem _ hx)) htr hls
    · cases h
  | ⟨.hello spec, ν, cs⟩ :: l₁, l₂, st, st', ls, hno, htr, h =>
    absurd (hno _ List.mem_cons_self) (by simp [isHello])
  | ⟨.opening i s n, b, cs⟩ :: l₁, l₂, st, st', ls, hno, htr, h => by
    simp only [List.cons_append, printLog] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · obtain ⟨ls', hls, -⟩ := Option.map_eq_some_iff.1 h
          exact printLog_suffix t t' l₁ l₂ (st.round i s) st'
            (fun x hx => hno x (List.mem_cons_of_mem _ hx)) htr hls
        · cases h

theorem mem_runsAt {S : RunSetup} {log : List Release} {lines : List String} (hf : Fits S.families log)
    (hl : printLog S.time PrintState.init log = some lines) (ht : treesOk (runOf S log lines) = true) :
    runOf S log lines ∈ runsAt S log S.key := by
  unfold runsAt
  refine List.mem_filter.2 ⟨?_, ht⟩
  have hl' : printLog S.time PrintState.init (ex log).log = some lines := hl
  simp only [Verity.Sampled.Device.dev]
  rw [if_pos (show S.key = S.key ∧ Fits S.families (ex log).log from ⟨rfl, hf⟩), hl']
  exact List.mem_singleton_self _

/-- **On two windows the history holds both runs**: at a play of the device's game whose first session's releases hold
no tree's hello, where each part fits its setup's layout and each printed run reads as coin trees, the history holds,
under each setup's key, the run that released that part. -/
theorem holdsTwo_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {fs : List (String × ℕ)}
    {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {o : Exchange}
    (h : Verity.Game.Reaches (Verity.play fs D θ F c) o)
    (h2 : oneWindow o.params = false) (hno : NoHello (o.log.take (cut o.log)))
    (hf₁ : Fits (S₁ o).families (o.log.take (cut o.log))) (hf₂ : Fits (S₂ o).families (o.log.drop (cut o.log)))
    (ht₁ : TreesRead (S₁ o) (o.log.take (cut o.log))) (ht₂ : TreesRead (S₂ o) (o.log.drop (cut o.log))) :
    ∃ lines₁ lines₂, runOf (S₁ o) (o.log.take (cut o.log)) lines₁ ∈ (dev₂ S₁ S₂ cut o).runs (S₁ o).key ∧
      runOf (S₂ o) (o.log.drop (cut o.log)) lines₂ ∈ (dev₂ S₁ S₂ cut o).runs (S₂ o).key := by
  obtain ⟨ls, hls⟩ := Verity.Sampled.Device.printable_play h (S₁ o).time
  have hls' : printLog (S₁ o).time PrintState.init (o.log.take (cut o.log) ++ o.log.drop (cut o.log)) = some ls := by
    rw [List.take_append_drop]
    exact hls
  obtain ⟨l₁, e₁⟩ := Option.isSome_iff_exists.1
    (printLog_prefix (S₁ o).time (S₁ o).time _ _ PrintState.init PrintState.init rfl hls')
  obtain ⟨l₂, e₂⟩ := Option.isSome_iff_exists.1
    (printLog_suffix (S₁ o).time (S₂ o).time _ _ PrintState.init PrintState.init hno rfl hls')
  refine ⟨l₁, l₂, ?_, ?_⟩
  · rw [runs_two S₁ S₂ cut h2]
    exact List.mem_append.2 (.inl (mem_runsAt hf₁ e₁ (ht₁ l₁ e₁)))
  · rw [runs_two S₁ S₂ cut h2]
    exact List.mem_append.2 (.inr (mem_runsAt hf₂ e₂ (ht₂ l₂ e₂)))

theorem coinsOf_append (label : String) :
    ∀ l₁ l₂ : List Release, Verity.Sampled.Device.coinsOf label (l₁ ++ l₂) =
      Verity.Sampled.Device.coinsOf label l₁ ++ Verity.Sampled.Device.coinsOf label l₂
  | [], _ => rfl
  | ⟨.fresh l, b, cs⟩ :: l₁, l₂ => by
    simp only [List.cons_append, Verity.Sampled.Device.coinsOf]
    by_cases h : l = label <;> simp [h, coinsOf_append label l₁ l₂]
  | ⟨.hello spec, ν, cs⟩ :: l₁, l₂ => by
    simp only [List.cons_append, Verity.Sampled.Device.coinsOf]
    exact coinsOf_append label l₁ l₂
  | ⟨.opening i s n, b, cs⟩ :: l₁, l₂ => by
    simp only [List.cons_append, Verity.Sampled.Device.coinsOf]
    exact coinsOf_append label l₁ l₂

/-- **Each part of a log that fits a layout fits it.** -/
theorem fits_parts {fs : List (String × ℕ)} {log : List Release} (h : Fits fs log) (n : ℕ) :
    Fits fs (log.take n) ∧ Fits fs (log.drop n) := by
  have hl : ∀ label, (Verity.Sampled.Device.coinsOf label log).length =
      (Verity.Sampled.Device.coinsOf label (log.take n)).length +
        (Verity.Sampled.Device.coinsOf label (log.drop n)).length := by
    intro label
    rw [← List.length_append, ← coinsOf_append, List.take_append_drop]
  refine ⟨fun label => ?_, fun label => ?_⟩
  · have := h label
    have := hl label
    omega
  · have := h label
    have := hl label
    omega

/-- **A play's log fits its layout** (`Sampled.Device.coins_fit`). -/
theorem fits_play {fs : List (String × ℕ)} {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {o : Exchange} (h : Verity.Game.Reaches (Verity.play fs D θ F c) o) : Fits fs o.log := by
  obtain ⟨a, ha⟩ := h
  obtain ⟨log, hlog, hf⟩ := Verity.Game.reaches_bind ha
  obtain ⟨b, ρ, e, he⟩ := hf
  subst he
  exact Verity.Sampled.Device.coins_fit _ fs [] log hlog

/-- **On two windows the history holds both runs, at setups of the play's layout**: `holdsTwo_dev₂`, each part fitting
the layout as the play's log does (`fits_play`, `fits_parts`). -/
theorem holdsTwo_play (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {fs : List (String × ℕ)}
    {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {o : Exchange}
    (h : Verity.Game.Reaches (Verity.play fs D θ F c) o) (hfs : (S₁ o).families = fs ∧ (S₂ o).families = fs)
    (h2 : oneWindow o.params = false) (hno : NoHello (o.log.take (cut o.log)))
    (ht₁ : TreesRead (S₁ o) (o.log.take (cut o.log))) (ht₂ : TreesRead (S₂ o) (o.log.drop (cut o.log))) :
    ∃ lines₁ lines₂, runOf (S₁ o) (o.log.take (cut o.log)) lines₁ ∈ (dev₂ S₁ S₂ cut o).runs (S₁ o).key ∧
      runOf (S₂ o) (o.log.drop (cut o.log)) lines₂ ∈ (dev₂ S₁ S₂ cut o).runs (S₂ o).key := by
  obtain ⟨hf₁, hf₂⟩ := fits_parts (fits_play h) (cut o.log)
  exact holdsTwo_dev₂ S₁ S₂ cut h h2 hno (hfs.1 ▸ hf₁) (hfs.2 ▸ hf₂) ht₁ ht₂

/-! ## The split at the first stage's family -/

/-- Whether a release is a fresh coin of the family `ℓ`. -/
def freshOf (ℓ : String) (x : Release) : Bool :=
  match x.frame with
  | .fresh l => l == ℓ
  | _ => false

/-- **The first session's releases at the first stage's family `ℓ`**: the log's longest prefix of fresh coins of `ℓ`,
so the first session ends at the first release that is anything else. -/
def firstLen (ℓ : String) (log : List Release) : ℕ := (log.takeWhile (freshOf ℓ)).length

theorem take_takeWhile {β : Type} (p : β → Bool) : ∀ l : List β, l.take (l.takeWhile p).length = l.takeWhile p
  | [] => rfl
  | a :: l => by
    cases ha : p a
    · simp [List.takeWhile, ha]
    · simp [List.takeWhile, ha, take_takeWhile p l]

theorem of_mem_takeWhile {β : Type} (p : β → Bool) : ∀ {l : List β} {x : β}, x ∈ l.takeWhile p → p x = true
  | [], _, h => by simp at h
  | a :: l, x, h => by
    cases ha : p a
    · simp [List.takeWhile, ha] at h
    · simp only [List.takeWhile, ha, List.mem_cons] at h
      rcases h with rfl | h
      · exact ha
      · exact of_mem_takeWhile p h

/-- **The first session at the first stage's family holds no hello**: every release it holds is a fresh coin. -/
theorem noHello_firstLen (ℓ : String) (log : List Release) : NoHello (log.take (firstLen ℓ log)) := by
  intro x hx
  rw [firstLen, take_takeWhile] at hx
  have hf := of_mem_takeWhile _ hx
  unfold freshOf at hf
  unfold isHello
  split at hf
  · rename_i l hl
    rw [hl]
  · cases hf

/-- **On two windows the history at the first stage's split holds both runs**: `holdsTwo_play` at `cut = firstLen ℓ`,
whose first session holds no hello (`noHello_firstLen`). -/
theorem holdsTwo_first (S₁ S₂ : Exchange → RunSetup) (ℓ : String) {fs : List (String × ℕ)}
    {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {o : Exchange}
    (h : Verity.Game.Reaches (Verity.play fs D θ F c) o) (hfs : (S₁ o).families = fs ∧ (S₂ o).families = fs)
    (h2 : oneWindow o.params = false)
    (ht₁ : TreesRead (S₁ o) (o.log.take (firstLen ℓ o.log)))
    (ht₂ : TreesRead (S₂ o) (o.log.drop (firstLen ℓ o.log))) :
    ∃ lines₁ lines₂, runOf (S₁ o) (o.log.take (firstLen ℓ o.log)) lines₁ ∈ (dev₂ S₁ S₂ (firstLen ℓ) o).runs (S₁ o).key ∧
      runOf (S₂ o) (o.log.drop (firstLen ℓ o.log)) lines₂ ∈ (dev₂ S₁ S₂ (firstLen ℓ) o).runs (S₂ o).key :=
  holdsTwo_play S₁ S₂ (firstLen ℓ) h hfs h2 (noHello_firstLen ℓ o.log) ht₁ ht₂

/-! ## What the device signs -/

/-- The public key of the parameters' challenger key (`keyOf`), the empty string where they name none. -/
def pubOf (θ : ByteArray) : ByteArray := ((keyOf θ).map (·.pub)).getD .empty

/-- **A two-session deployment of the device**: the first stage's session (`one`) and the interior's (`two`), each a
one-stage deployment (`Sampled.Device.Deployment`: its run's setup, its ledger lines before any coin and at the close).
The device's samples at the close and its signatures are the first's. -/
structure Deployment where
  one : Verity.Sampled.Device.Deployment
  two : Verity.Sampled.Device.Deployment

namespace Deployment

variable (P : Deployment)

/-- **A session's setup at a play**, of the one-stage deployment `Q` at its layouts `fs`, at the session's releases
`log`: `Q`'s run, under the parameters' key (`pubOf`), with the ledger lines it signs before any coin and at the
close. -/
def setupAt (Q : Verity.Sampled.Device.Deployment) (fs : ByteArray → List (String × ℕ)) (θ a : ByteArray)
    (log : List Release) (b : ByteArray) (ρ : List Bool) : RunSetup :=
  { Q.run θ a log b ρ with key := pubOf θ, families := fs θ, ledger := Q.opens θ a ++ Q.closes θ a log b ρ }

/-- The first session's setup at the play that ended with `o`, at the first session's releases. -/
def setup₁ (fs₁ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) : RunSetup :=
  setupAt P.one fs₁ o.params o.request (logs cut o.params o.log).1 o.close o.drawn

/-- The interior session's setup at the play that ended with `o`, at the interior's releases. -/
def setup₂ (fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) : RunSetup :=
  setupAt P.two fs₂ o.params o.request (logs cut o.params o.log).2 o.close o.drawn

/-- The ledger lines the device signs before any coin: the first session's on one window, both sessions' on two. -/
def opensAt (θ a : ByteArray) : List Json :=
  bif oneWindow θ then P.one.opens θ a else P.one.opens θ a ++ P.two.opens θ a

/-- The ledger lines the device signs at the close, each session's at its releases. -/
def closesAt (cut : List Release → ℕ) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    List Json :=
  bif oneWindow θ then P.one.closes θ a (logs cut θ log).1 b ρ
  else P.one.closes θ a (logs cut θ log).1 b ρ ++ P.two.closes θ a (logs cut θ log).2 b ρ

/-- **What the device holds at the close**: the runs the two-session history holds (`runsOf`, at the deployment's
setups) and the ledger lines at the close. -/
noncomputable def closing (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ a : ByteArray)
    (log : List Release) (b : ByteArray) (ρ : List Bool) : History :=
  ⟨runsOf (setupAt P.one fs₁ θ a (logs cut θ log).1 b ρ) (setupAt P.two fs₂ θ a (logs cut θ log).2 b ρ)
      (oneWindow θ) (logs cut θ log).1 (logs cut θ log).2,
    fun _ => P.closesAt cut θ a log b ρ⟩

/-- **The deployed device's signer at the parameters `θ`**: before any coin, the messages of the ledger lines the
request fixes (`opensAt`); at the close, the records and closes of the runs the history holds and the messages of the
ledger lines at the close (`closing`); each what `Properties.Challenger.signedBy` lists under the parameters' key. -/
noncomputable def signer (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) :
    Verity.Signer where
  first a := (signedBy ⟨fun _ => [], fun _ => P.opensAt θ a⟩ (pubOf θ)).map (·.data.toList)
  draws := P.one.draws θ
  last a log b ρ := (signedBy (P.closing fs₁ fs₂ cut θ a log b ρ) (pubOf θ)).map (·.data.toList)
  sign := P.one.sign θ

theorem ledger_opens (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) {x : Json}
    (hx : x ∈ P.opensAt o.params o.request) :
    x ∈ ledgerOf (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) (oneWindow o.params) (pubOf o.params) := by
  unfold opensAt at hx
  unfold ledgerOf
  simp only [ledgerAt, setup₁, setup₂, setupAt, ↓reduceIte]
  generalize oneWindow o.params = w at hx ⊢
  cases w
  · simp only [Bool.cond_false, List.mem_append] at hx ⊢
    rcases hx with hx | hx
    · exact .inl (.inl hx)
    · exact .inr (.inl hx)
  · simp only [Bool.cond_true, List.mem_append] at hx ⊢
    exact .inl hx

theorem ledger_closes (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) {x : Json}
    (hx : x ∈ P.closesAt cut o.params o.request o.log o.close o.drawn) :
    x ∈ ledgerOf (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) (oneWindow o.params) (pubOf o.params) := by
  unfold closesAt at hx
  unfold ledgerOf
  simp only [ledgerAt, setup₁, setup₂, setupAt, ↓reduceIte]
  generalize oneWindow o.params = w at hx ⊢
  cases w
  · simp only [Bool.cond_false, List.mem_append] at hx ⊢
    rcases hx with hx | hx
    · exact .inl (.inr hx)
    · exact .inr (.inr hx)
  · simp only [Bool.cond_true, List.mem_append] at hx ⊢
    exact .inr hx

theorem runsAt_length (S : RunSetup) (log : List Release) (pk : ByteArray) : (runsAt S log pk).length ≤ 1 := by
  unfold runsAt
  refine (List.length_filter_le _ _).trans ?_
  simp only [Verity.Sampled.Device.dev]
  split
  · split <;> simp
  · simp

theorem runsOf_length (S₁ S₂ : RunSetup) (one : Bool) (log₁ log₂ : List Release) (pk : ByteArray) :
    (runsOf S₁ S₂ one log₁ log₂ pk).length ≤ 2 := by
  have h1 := runsAt_length S₁ log₁ pk
  have h2 := runsAt_length S₂ log₂ pk
  unfold runsOf
  cases one
  · simp only [Bool.cond_false, List.length_append]
    omega
  · simp only [Bool.cond_true]
    omega

theorem ledger_length (cut : List Release → ℕ) (θ a : ByteArray) (log : List Release) (b : ByteArray)
    (ρ : List Bool) :
    (P.opensAt θ a).length + (P.closesAt cut θ a log b ρ).length ≤ P.one.ledgerLines θ + P.two.ledgerLines θ := by
  unfold opensAt closesAt
  have h1 := P.one.ledger_le θ a (logs cut θ log).1 b ρ
  have h2 := P.two.ledger_le θ a (logs cut θ log).2 b ρ
  generalize oneWindow θ = w
  cases w
  · simp only [Bool.cond_false, List.length_append]
    omega
  · simp only [Bool.cond_true]
    omega

/-- **`Q` at the deployment**: one message per ledger line either session signs, and each run's record and close. -/
def signatures (θ : ByteArray) : ℕ := P.one.ledgerLines θ + P.two.ledgerLines θ + 4

/-- **The deployed device signs at most `Q` messages in a play** (`Vocab.signs_le`): before any coin at most one per
line the request fixes, and at the close at most the two runs' records and closes and one per line at the close
(`Sampled.Device.signedBy_length`). -/
theorem signs_le (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ a : ByteArray)
    (log : List Release) (b : ByteArray) (ρ : List Bool) :
    ((P.signer fs₁ fs₂ cut θ).signs a log b ρ).length ≤ P.signatures θ := by
  have h1 : (signedBy ⟨fun _ => [], fun _ => P.opensAt θ a⟩ (pubOf θ)).length ≤ (P.opensAt θ a).length := by
    have := Verity.Sampled.Device.signedBy_length ⟨fun _ => [], fun _ => P.opensAt θ a⟩ (pubOf θ)
    simpa using this
  have h2 : (signedBy (P.closing fs₁ fs₂ cut θ a log b ρ) (pubOf θ)).length ≤
      2 * 2 + (P.closesAt cut θ a log b ρ).length := by
    have := Verity.Sampled.Device.signedBy_length (P.closing fs₁ fs₂ cut θ a log b ρ) (pubOf θ)
    have h3 : ((P.closing fs₁ fs₂ cut θ a log b ρ).runs (pubOf θ)).length ≤ 2 := runsOf_length _ _ _ _ _ _
    have h4 : (P.closing fs₁ fs₂ cut θ a log b ρ).ledger (pubOf θ) = P.closesAt cut θ a log b ρ := rfl
    rw [h4] at this
    omega
  have h5 := P.ledger_length cut θ a log b ρ
  show ((signedBy ⟨fun _ => [], fun _ => P.opensAt θ a⟩ (pubOf θ)).map (·.data.toList) ++
      (signedBy (P.closing fs₁ fs₂ cut θ a log b ρ) (pubOf θ)).map (·.data.toList)).length ≤ P.signatures θ
  rw [List.length_append, List.length_map, List.length_map]
  unfold signatures
  omega

end Deployment

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The two-session history at the deployment's setups.** -/
noncomputable abbrev devOf (P : Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) :
    Exchange → History :=
  dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut

/-- **The deployed device's history at each play signed what the play's signatures are on** (`SignsPlay`), at a
vocabulary whose signer at the parameters is the deployment's. Each message the play's signer signs is one a history of
the ledger lines the request fixes, or of the runs and ledger lines at the close, signs (`Deployment.signer`), and the
two-session history at the play holds those runs and lines under the parameters' key. -/
theorem signsPlay_dev₂ (V : Vocab d α) (P : Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ))
    (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    SignsPlay V (devOf P fs₁ fs₂ cut) θ F c := by
  intro o ho key hk m hm
  obtain hθ := (Verity.Sampled.Binding.play_reaches ho).1
  subst hθ
  have hpub : pubOf o.params = key.pub := by
    simp [pubOf, hk]
  have hm' : m.1 ∈ ((P.signer fs₁ fs₂ cut o.params).signatures o.request o.log o.close o.drawn).map Prod.fst := by
    rw [← hD, ← Verity.Sampled.Binding.play_signed ho]
    exact List.mem_map.2 ⟨m, hm, rfl⟩
  rw [Verity.Sampled.Device.signatures_fst] at hm'
  change m.1 ∈ (signedBy ⟨fun _ => [], fun _ => P.opensAt o.params o.request⟩ (pubOf o.params)).map
      (·.data.toList) ++ (signedBy (P.closing fs₁ fs₂ cut o.params o.request o.log o.close o.drawn)
      (pubOf o.params)).map (·.data.toList) at hm'
  rw [← List.map_append] at hm'
  obtain ⟨x, hx, hxm⟩ := List.mem_map.1 hm'
  have hmx : ({ data := m.1.toArray } : ByteArray) = x :=
    (congrArg (fun l : List UInt8 => ({ data := l.toArray } : ByteArray)) hxm).symm.trans
      (Verity.Sampled.Binding.bytes_ofList x)
  show Signed _ _ _
  rw [hmx, ← hpub]
  rcases List.mem_append.1 hx with hx | hx
  · apply Verity.Sampled.Device.signedBy_mono hx
    · intro s hs
      exact absurd hs List.not_mem_nil
    · intro l hl
      exact P.ledger_opens fs₁ fs₂ cut o hl
  · apply Verity.Sampled.Device.signedBy_mono hx
    · intro s hs
      exact hs
    · intro l hl
      exact P.ledger_closes fs₁ fs₂ cut o hl

/-- **The cut at the deployed two-session history** (`Cut.cutBound`): where the sampled verdict accepts only where the
form does (`Wired`), the ledger opens neither session's run on the other's window (`Apart`), and the vocabulary's
signer is the deployment's, an accepted play whose log the check read isn't the device's is at most as likely as the
read finder's collision plus the forger's forgery over both sessions. `Apart` is taken, not proved. -/
theorem cut_dev₂ (V : Vocab d α) (k : Verity.Registration.Composite) (P : Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ)
    (hA : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      Apart (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) w₁ w₂)
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    CutBound V k θ F c σ LogIsDevice (terms V (devOf P fs₁ fs₂ cut) θ F c σ) :=
  cutBound V k (devOf P fs₁ fs₂ cut) θ F c σ hw (binds_dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut θ hA)
    (signsPlay_dev₂ V P fs₁ fs₂ cut θ F c hD)

end Verity.SampledTwo.Sessions
