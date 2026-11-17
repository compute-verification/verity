module

public import Proofs.Properties.Verity.SampledTwo.CutHeld
public import Proofs.Properties.Verity.Sampled.Device

/-! The device's history at a two-session play (`dev₂`), with `PrintsTwo` and `SignsPlay` at it; `Apart` is
taken as a hypothesis, not proved, and the history holds exactly this play's runs on its two windows. -/

@[expose] public section

namespace Verity.SampledTwo.Sessions

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Device (RunSetup PrintState printLog runOf Fits)

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

def ex (log : List Release) : Exchange :=
  ⟨.empty, #[], .empty, .empty, log, .empty, [], [], .empty, #[]⟩

noncomputable def runsAt (S : RunSetup) (log : List Release) (pk : ByteArray) : List Properties.Challenger.Run :=
  ((Verity.Sampled.Device.dev (fun _ => S) (ex log)).runs pk).filter treesOk

open Classical in
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

theorem run_facts {S : RunSetup} {log : List Release} {pk : ByteArray} {r : Properties.Challenger.Run}
    (hr : r ∈ runsAt S log pk) :
    Verity.Sampled.Lines.Prints [] r.log.tail log ∧ (∃ ts, coinTreesOf r.log = .ok ts) ∧
      ((∃ c roots, r.commitment = some c ∧ releasedRootsOf c r.log = .ok roots) →
        ∀ label after, Verity.Sampled.Lines.Holds (r.released label after)
          (Verity.Sampled.Lines.freshOf label after log)) := by
  obtain ⟨hr, ht⟩ := List.mem_filter.1 hr
  exact ⟨Verity.Sampled.Device.prints_dev (fun _ => S) (ex log) hr, treesOk_ok ht,
    fun ⟨_, _, hc, hroots⟩ => Verity.Sampled.Device.holds_dev (fun _ => S) (ex log) hr hc hroots⟩

def oneExecutionPhase (θ : ByteArray) : Bool :=
  match executionPhasesOf θ with
  | some (w₁, w₂) => decide (w₁ = w₂)
  | none => true

theorem oneExecutionPhase_true {θ : ByteArray} {w₁ w₂ : String} (hw : executionPhasesOf θ = some (w₁, w₂)) (h : w₁ = w₂) :
    oneExecutionPhase θ = true := by
  simp [oneExecutionPhase, hw, h]

theorem oneExecutionPhase_false {θ : ByteArray} {w₁ w₂ : String} (hw : executionPhasesOf θ = some (w₁, w₂)) (h : w₁ ≠ w₂) :
    oneExecutionPhase θ = false := by
  simp [oneExecutionPhase, hw, h]

def logs (cut : List Release → ℕ) (θ : ByteArray) (log : List Release) : List Release × List Release :=
  bif oneExecutionPhase θ then (log, []) else (log.take (cut log), log.drop (cut log))

noncomputable def runsOf (S₁ S₂ : RunSetup) (one : Bool) (log₁ log₂ : List Release) (pk : ByteArray) :
    List Properties.Challenger.Run :=
  bif one then runsAt S₁ log₁ pk else runsAt S₁ log₁ pk ++ runsAt S₂ log₂ pk

noncomputable def ledgerOf (S₁ S₂ : RunSetup) (one : Bool) (pk : ByteArray) : List Json :=
  bif one then ledgerAt S₁ pk else ledgerAt S₁ pk ++ ledgerAt S₂ pk

noncomputable def dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (o : Exchange) : History :=
  ⟨runsOf (S₁ o) (S₂ o) (oneExecutionPhase o.params) (logs cut o.params o.log).1 (logs cut o.params o.log).2,
    ledgerOf (S₁ o) (S₂ o) (oneExecutionPhase o.params)⟩

def OpensIn (ledger : List Json) (w id : String) : Prop :=
  OpensOn ⟨fun _ => [], fun _ => ledger⟩ ByteArray.empty w id

theorem opensIn_of_opensOn {h : History} {pk : ByteArray} {w id : String} (ho : OpensOn h pk w id) {L : List Json}
    (hL : ∀ x ∈ h.ledger pk, x ∈ L) : OpensIn L w id := by
  unfold OpensIn OpensOn at *
  obtain ⟨line, hl, rest⟩ := ho
  exact ⟨line, hL line hl, rest⟩

/-- The two setups' ledger lines don't open the second setup's run on `w₁`, nor the first's on `w₂`; taken as
a hypothesis, not proved, and it fails where a device opens each window as a prover's request names them the other way
round. -/
def Apart (S₁ S₂ : RunSetup) (w₁ w₂ : String) : Prop :=
  ¬ OpensIn (S₁.ledger ++ S₂.ledger) w₁ S₂.id ∧ ¬ OpensIn (S₁.ledger ++ S₂.ledger) w₂ S₁.id

theorem printsOn_one {h : History} {S : RunSetup} {log : List Release} {w : String}
    (hr : ∀ pk r, r ∈ h.runs pk → r ∈ runsAt S log pk) : PrintsOn h w log :=
  fun pk r hr' _ => run_facts (hr pk r hr')

theorem printsOn_part {h : History} {S S' : RunSetup} {log log' : List Release} {w : String}
    (hr : ∀ pk r, r ∈ h.runs pk → r ∈ runsAt S log pk ∨ r ∈ runsAt S' log' pk)
    (hno : ∀ pk, ¬ OpensOn h pk w S'.id) : PrintsOn h w log := by
  intro pk r hr' ho
  rcases hr pk r hr' with h1 | h2
  · exact run_facts h1
  · rw [runsAt_id h2] at ho
    exact absurd ho (hno pk)

theorem logs_one {cut : List Release → ℕ} {θ : ByteArray} {log : List Release} (h : oneExecutionPhase θ = true) :
    logs cut θ log = (log, []) := by
  simp only [logs, h, Bool.cond_true]

theorem logs_two {cut : List Release → ℕ} {θ : ByteArray} {log : List Release} (h : oneExecutionPhase θ = false) :
    logs cut θ log = (log.take (cut log), log.drop (cut log)) := by
  simp only [logs, h, Bool.cond_false]

theorem runs_two (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {o : Exchange}
    (h : oneExecutionPhase o.params = false) (pk : ByteArray) :
    (dev₂ S₁ S₂ cut o).runs pk =
      runsAt (S₁ o) (o.log.take (cut o.log)) pk ++ runsAt (S₂ o) (o.log.drop (cut o.log)) pk := by
  show runsOf (S₁ o) (S₂ o) (oneExecutionPhase o.params) (logs cut o.params o.log).1 (logs cut o.params o.log).2 pk = _
  simp only [logs_two h, h, runsOf, Bool.cond_false]

theorem ledger_two (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {o : Exchange}
    (h : oneExecutionPhase o.params = false) (pk : ByteArray) :
    ∀ x ∈ (dev₂ S₁ S₂ cut o).ledger pk, x ∈ (S₁ o).ledger ++ (S₂ o).ledger := by
  intro x hx
  change x ∈ ledgerOf (S₁ o) (S₂ o) (oneExecutionPhase o.params) pk at hx
  rw [h] at hx
  rcases List.mem_append.1 hx with hx | hx
  · exact List.mem_append.2 (.inl (mem_ledgerAt hx))
  · exact List.mem_append.2 (.inr (mem_ledgerAt hx))

/-- `Apart` is taken, not proved. -/
theorem printsTwo_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (o : Exchange) {w₁ w₂ : String}
    (hw : executionPhasesOf o.params = some (w₁, w₂)) (hA : w₁ ≠ w₂ → Apart (S₁ o) (S₂ o) w₁ w₂) :
    PrintsTwo (dev₂ S₁ S₂ cut o) w₁ w₂ o.log := by
  unfold PrintsTwo
  by_cases hww : w₁ = w₂
  · rw [if_pos hww]
    have h1 := oneExecutionPhase_true hw hww
    refine printsOn_one (S := S₁ o) fun pk r hr => ?_
    have e : (dev₂ S₁ S₂ cut o).runs pk = runsAt (S₁ o) o.log pk := by
      show runsOf (S₁ o) (S₂ o) (oneExecutionPhase o.params) (logs cut o.params o.log).1 (logs cut o.params o.log).2 pk = _
      simp only [logs_one h1, h1, runsOf, Bool.cond_true]
    rwa [e] at hr
  · rw [if_neg hww]
    have h2 := oneExecutionPhase_false hw hww
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

/-- `Apart` is taken, not proved. -/
theorem binds_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (θ : ByteArray)
    (hA : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      Apart (S₁ o) (S₂ o) w₁ w₂) :
    Binds (dev₂ S₁ S₂ cut) θ :=
  binds_of_accepts (dev₂ S₁ S₂ cut) θ fun o hθ w₁ w₂ hw =>
    printsTwo_dev₂ S₁ S₂ cut o (by rw [hθ]; exact hw) (hA o hθ w₁ w₂ hw)

def isHello : Verity.Frame → Bool
  | .hello _ => true
  | _ => false

def NoHello (log : List Release) : Prop := ∀ x ∈ log, isHello x.frame = false

def TreesRead (S : RunSetup) (log : List Release) : Prop :=
  ∀ lines, printLog S.time PrintState.init log = some lines → treesOk (runOf S log lines) = true

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

theorem holdsTwo_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {fs : List (String × ℕ)}
    {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {o : Exchange}
    (h : Verity.Game.Reaches (Verity.play fs D θ F c) o)
    (h2 : oneExecutionPhase o.params = false) (hno : NoHello (o.log.take (cut o.log)))
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

theorem fits_play {fs : List (String × ℕ)} {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {o : Exchange} (h : Verity.Game.Reaches (Verity.play fs D θ F c) o) : Fits fs o.log := by
  obtain ⟨a, ha⟩ := h
  obtain ⟨log, hlog, hf⟩ := Verity.Game.reaches_bind ha
  obtain ⟨b, ρ, e, he⟩ := hf
  subst he
  exact Verity.Sampled.Device.coins_fit _ fs [] log hlog

theorem holdsTwo_play (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) {fs : List (String × ℕ)}
    {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {o : Exchange}
    (h : Verity.Game.Reaches (Verity.play fs D θ F c) o) (hfs : (S₁ o).families = fs ∧ (S₂ o).families = fs)
    (h2 : oneExecutionPhase o.params = false) (hno : NoHello (o.log.take (cut o.log)))
    (ht₁ : TreesRead (S₁ o) (o.log.take (cut o.log))) (ht₂ : TreesRead (S₂ o) (o.log.drop (cut o.log))) :
    ∃ lines₁ lines₂, runOf (S₁ o) (o.log.take (cut o.log)) lines₁ ∈ (dev₂ S₁ S₂ cut o).runs (S₁ o).key ∧
      runOf (S₂ o) (o.log.drop (cut o.log)) lines₂ ∈ (dev₂ S₁ S₂ cut o).runs (S₂ o).key := by
  obtain ⟨hf₁, hf₂⟩ := fits_parts (fits_play h) (cut o.log)
  exact holdsTwo_dev₂ S₁ S₂ cut h h2 hno (hfs.1 ▸ hf₁) (hfs.2 ▸ hf₂) ht₁ ht₂

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

theorem holdsTwo_first (S₁ S₂ : Exchange → RunSetup) (ℓ : String) {fs : List (String × ℕ)}
    {D : Verity.Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {o : Exchange}
    (h : Verity.Game.Reaches (Verity.play fs D θ F c) o) (hfs : (S₁ o).families = fs ∧ (S₂ o).families = fs)
    (h2 : oneExecutionPhase o.params = false)
    (ht₁ : TreesRead (S₁ o) (o.log.take (firstLen ℓ o.log)))
    (ht₂ : TreesRead (S₂ o) (o.log.drop (firstLen ℓ o.log))) :
    ∃ lines₁ lines₂, runOf (S₁ o) (o.log.take (firstLen ℓ o.log)) lines₁ ∈ (dev₂ S₁ S₂ (firstLen ℓ) o).runs (S₁ o).key ∧
      runOf (S₂ o) (o.log.drop (firstLen ℓ o.log)) lines₂ ∈ (dev₂ S₁ S₂ (firstLen ℓ) o).runs (S₂ o).key :=
  holdsTwo_play S₁ S₂ (firstLen ℓ) h hfs h2 (noHello_firstLen ℓ o.log) ht₁ ht₂

def pubOf (θ : ByteArray) : ByteArray := ((keyOf θ).map (·.pub)).getD .empty

structure Deployment where
  one : Verity.Sampled.Device.Deployment
  two : Verity.Sampled.Device.Deployment

namespace Deployment

variable (P : Deployment)

def setupAt (Q : Verity.Sampled.Device.Deployment) (fs : ByteArray → List (String × ℕ)) (θ a : ByteArray)
    (log : List Release) (b : ByteArray) (ρ : List Bool) : RunSetup :=
  { Q.run θ a log b ρ with key := pubOf θ, families := fs θ, ledger := Q.opens θ a ++ Q.closes θ a log b ρ }

def setup₁ (fs₁ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) : RunSetup :=
  setupAt P.one fs₁ o.params o.request (logs cut o.params o.log).1 o.close o.drawn

def setup₂ (fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) : RunSetup :=
  setupAt P.two fs₂ o.params o.request (logs cut o.params o.log).2 o.close o.drawn

def opensAt (θ a : ByteArray) : List Json :=
  bif oneExecutionPhase θ then P.one.opens θ a else P.one.opens θ a ++ P.two.opens θ a

def closesAt (cut : List Release → ℕ) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    List Json :=
  bif oneExecutionPhase θ then P.one.closes θ a (logs cut θ log).1 b ρ
  else P.one.closes θ a (logs cut θ log).1 b ρ ++ P.two.closes θ a (logs cut θ log).2 b ρ

noncomputable def closing (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ a : ByteArray)
    (log : List Release) (b : ByteArray) (ρ : List Bool) : History :=
  ⟨runsOf (setupAt P.one fs₁ θ a (logs cut θ log).1 b ρ) (setupAt P.two fs₂ θ a (logs cut θ log).2 b ρ)
      (oneExecutionPhase θ) (logs cut θ log).1 (logs cut θ log).2,
    fun _ => P.closesAt cut θ a log b ρ⟩

noncomputable def signer (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) :
    Verity.Signer where
  first a := (signedBy ⟨fun _ => [], fun _ => P.opensAt θ a⟩ (pubOf θ)).map (·.data.toList)
  draws := P.one.draws θ
  last a log b ρ := (signedBy (P.closing fs₁ fs₂ cut θ a log b ρ) (pubOf θ)).map (·.data.toList)
  sign := P.one.sign θ

theorem ledger_opens (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) {x : Json}
    (hx : x ∈ P.opensAt o.params o.request) :
    x ∈ ledgerOf (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) (oneExecutionPhase o.params) (pubOf o.params) := by
  unfold opensAt at hx
  unfold ledgerOf
  simp only [ledgerAt, setup₁, setup₂, setupAt, ↓reduceIte]
  generalize oneExecutionPhase o.params = w at hx ⊢
  cases w
  · simp only [Bool.cond_false, List.mem_append] at hx ⊢
    rcases hx with hx | hx
    · exact .inl (.inl hx)
    · exact .inr (.inl hx)
  · simp only [Bool.cond_true, List.mem_append] at hx ⊢
    exact .inl hx

theorem ledger_closes (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (o : Exchange) {x : Json}
    (hx : x ∈ P.closesAt cut o.params o.request o.log o.close o.drawn) :
    x ∈ ledgerOf (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) (oneExecutionPhase o.params) (pubOf o.params) := by
  unfold closesAt at hx
  unfold ledgerOf
  simp only [ledgerAt, setup₁, setup₂, setupAt, ↓reduceIte]
  generalize oneExecutionPhase o.params = w at hx ⊢
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
  generalize oneExecutionPhase θ = w
  cases w
  · simp only [Bool.cond_false, List.length_append]
    omega
  · simp only [Bool.cond_true]
    omega

def signatures (θ : ByteArray) : ℕ := P.one.ledgerLines θ + P.two.ledgerLines θ + 4

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

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

noncomputable abbrev devOf (P : Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) :
    Exchange → History :=
  dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut

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

/-- `Apart` is taken, not proved. -/
theorem cut_dev₂ (V : Vocab d α) (k : Verity.Registration.Composite) (P : Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ)
    (hA : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      Apart (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) w₁ w₂)
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    CutBound V k θ F c σ LogIsDevice (terms V (devOf P fs₁ fs₂ cut) θ F c σ) :=
  cutBound V k (devOf P fs₁ fs₂ cut) θ F c σ hw (binds_dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut θ hA)
    (signsPlay_dev₂ V P fs₁ fs₂ cut θ F c hD)

end Verity.SampledTwo.Sessions
