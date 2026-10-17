import Proofs.Flock.Claims.ZK.Reference
import Proofs.Flock.Claims.ZK.Interact
import Proofs.Flock.Claims.ZK.Rewind

/-!
# The ZK server: untrusted workers compute, the server releases

The workers are untrusted GPUs: they run the inner proof with zero knowledge off and compute the outer session's
arithmetic unmasked. The ZK server alone talks to the verifier. Each round it releases one frame (`ZkServer.frame`) of
values of four kinds (`Kind`), on a public schedule (`ZkServer.sched`):

* `fixed`: a constant of the protocol's structure; the worker's copy must match it;
* `own`: one of the server's own draws, read from its tape (salts, pads, the masks `h`, `μ`);
* `recomputed`: a value the server computes from its own copy: the witness, its tape, the history and the round's
  checked worker values (`ZkServer.copy`): every Merkle root and path, and the fold's row combination `Y = h + βμ`; the
  worker's copy must match it;
* `masked`: a worker's value under the server's mask (`ZkServer.mask`), added only once the value passed the server's
  relation check (`ZkServer.rel`), which reads the server's private check coins `Chk`.

A round in which any check fails, or a worker misses its time (`none`), releases the stop (`ZkServer.stop`), and so does
every later round (`ZkServer.live`). The worker is any function of the statement, the witness, the server's tape and
the history (`ZkServer.Worker`); it never sees the check coins. The inner phase's firewall, `Flock.Firewall.run`, is the
same shape: by `run_resalt` its stop never depends on its salts, and by `run_of_one` what it sends is a fixed template
filled with the commit strings of fixed digests under its own salts, kinds `fixed` and `recomputed` over `own`.

* `split_exact`: where every masked check along the reference prover's run is faithful (`ZkServer.Faithful`), the
  server releases the reference prover's run cut at some round `t ≤ k` and then the stop (`cutT`). It needs one
  obligation of the server's definition, `ZkServer.Ref`: at honest worker values, its frame is `P`'s message.
* `split`: for every worker, that holds but with probability `∑ ε` over the check coins, the masked checks' soundness
  `ZkServer.CheckSound ε`, a second obligation of the server's definition (exact recomputation meets it at `ε = 0`,
  `checkSound_exact`).
* `zk_server`: zero knowledge carries over from `ZKmalR` of the reference prover with a guessing loss: the released,
  scheduled transcript is within a factor `k + 1` of `leaky`'s, the simulator that guesses the stop round, plus
  `(k + 1)·(ε_zk + γ·Fork)` and `∑ ε`. `Flock.zk_server_ref` is it at `refP` and `Flock.zk_deployZk_ref`.
* The stop's leak: `Pr_choice_le`, an observation at a stop pattern the worker chooses is within the number of
  patterns of the worst fixed one; `stream_leak`, on a stream of `m` sessions with at most `b` stops each among `r`
  forms, that number is at most `∑_{i ≤ b} C(m, i)·r^i` (`card_pattern_le`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-! ## The server -/

/-- What a released value is. -/
inductive Kind
  | fixed
  | own
  | recomputed
  | masked
  deriving DecidableEq

/-- **The ZK server** of a protocol with witnesses `W`, on its tape `R`: at statement `x`, round `j` releases a frame of
`nS x j` values of the kinds `kind x j`, at time `sched x j`. The masked values' checks read the server's private check
coins `Chk`; `honest` is the honest worker's value of each slot. -/
structure ZkServer (W R : Type) where
  Chk : Type
  [chkFin : Fintype Chk]
  [chkNe : Nonempty Chk]
  Val : Type
  [valEq : DecidableEq Val]
  nS : Msg → ℕ → ℕ
  kind : Msg → ℕ → ℕ → Kind
  cst : Msg → ℕ → ℕ → Val
  own : Msg → ℕ → ℕ → R → List (Msg × Msg) → Val
  recomp : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → (ℕ → ℕ → Option Val) → Val
  mask : Msg → ℕ → ℕ → R → List (Msg × Msg) → Val → Val
  rel : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → (ℕ → ℕ → Option Val) → Chk → Val → Bool
  honest : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → Val
  frame : (x : Msg) → (j : ℕ) → (Fin (nS x j) → Val) → Msg
  stop : Msg
  sched : Msg → ℕ → ℕ

attribute [instance] ZkServer.chkFin ZkServer.chkNe ZkServer.valEq

namespace ZkServer

/-- **A worker**: any function of the statement, the witness, the server's tape and the history to each slot's value,
`none` for a missed time. It never sees the check coins. -/
abbrev Worker {W R : Type} (S : ZkServer W R) : Type := Msg → W → R → List (Msg × Msg) → ℕ → Option S.Val

/-- **The server's copy** after the history `h`: the workers' masked values of every round so far, the current one's
included. -/
def copy {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) :
    ℕ → ℕ → Option S.Val := fun i s =>
  if i ≤ h.length ∧ s < S.nS x i ∧ S.kind x i s = .masked then wk x w ρ (h.take i) s else none

/-- The copy at honest workers. -/
def hcopy {W R : Type} (S : ZkServer W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) :
    ℕ → ℕ → Option S.Val := fun i s =>
  if i ≤ h.length ∧ s < S.nS x i ∧ S.kind x i s = .masked then some (S.honest x i s w ρ (h.take i)) else none

/-- **The value released** at slot `s` of round `j`, from the copy `cp` and the worker's value `u`. -/
def value {W R : Type} (S : ZkServer W R) (x : Msg) (j s : ℕ) (w : W) (ρ : R) (h : List (Msg × Msg))
    (cp : ℕ → ℕ → Option S.Val) (u : Option S.Val) : S.Val :=
  match S.kind x j s, u with
  | .fixed, _ => S.cst x j s
  | .own, _ => S.own x j s ρ h
  | .recomputed, _ => S.recomp x j s w ρ h cp
  | .masked, some v => S.mask x j s ρ h v
  | .masked, none => S.cst x j s

/-- **The server's check** of the worker's value `u` at slot `s` of round `j`, at the check coins `κ`. -/
def check {W R : Type} (S : ZkServer W R) (x : Msg) (j s : ℕ) (w : W) (ρ : R) (h : List (Msg × Msg))
    (cp : ℕ → ℕ → Option S.Val) (κ : S.Chk) (u : Option S.Val) : Bool :=
  match S.kind x j s, u with
  | .fixed, _ => decide (u = some (S.cst x j s))
  | .own, _ => true
  | .recomputed, _ => decide (u = some (S.recomp x j s w ρ h cp))
  | .masked, some v => S.rel x j s w ρ h cp κ v
  | .masked, none => false

/-- Every slot of the round after `h` passes its check. -/
def pass {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) : Bool :=
  (List.range (S.nS x h.length)).all fun s => S.check x h.length s w ρ h (S.copy wk x w ρ h) κ (wk x w ρ h s)

/-- Every round so far, the current one's included, passed. -/
def live {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) : Bool :=
  (List.range (h.length + 1)).all fun i => S.pass wk x w ρ κ (h.take i)

/-- **What the server releases** after `h`: the round's frame, its masks added after its checks, or the stop. -/
def next {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) : Msg :=
  if S.live wk x w ρ κ h then
    S.frame x h.length fun s => S.value x h.length s w ρ h (S.copy wk x w ρ h) (wk x w ρ h s)
  else S.stop

/-- The frame at honest workers. -/
def refMsg {W R : Type} (S : ZkServer W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) : Msg :=
  S.frame x h.length fun s => S.value x h.length s w ρ h (S.hcopy x w ρ h) (some (S.honest x h.length s w ρ h))

/-- The server with the worker `wk` as a prover: its tape is the server's and the check coins. -/
def prover {W R : Type} (S : ZkServer W R) (wk : S.Worker) (w : W) : Prover (R × S.Chk) :=
  ⟨fun x h ω => .ret (S.next wk x w ω.1 ω.2 h)⟩

/-- **The released transcript, scheduled**: each message with its release time, a function of its round alone. -/
def timed {W R : Type} (S : ZkServer W R) (x : Msg) (τ : List (Msg × Out)) : List ((Msg × Out) × ℕ) :=
  τ.mapIdx fun j e => (e, S.sched x j)

/-- **The server's frame at honest workers is `P`'s message**, at `(x, w)`. -/
def Ref {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R) (x : Msg) (w : W) : Prop :=
  ∀ ρ h, (((P w).next x h ρ).eval H).1 = S.refMsg x w ρ h

/-- **The masked checks are sound**: over the check coins, a wrong value passes slot `s` of round `j` with probability
at most `ε x j s`, at every copy. -/
def CheckSound {W R : Type} (S : ZkServer W R) (ε : Msg → ℕ → ℕ → ℝ) : Prop :=
  ∀ x j s w ρ h cp v, S.kind x j s = .masked → v ≠ S.honest x j s w ρ h →
    Pr (fun κ : S.Chk => S.rel x j s w ρ h cp κ v = true) ≤ ε x j s

/-- **The round after `h` is faithful** at the check coins `κ`: every masked value that passes is the honest one. -/
def Faithful {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) : Prop :=
  ∀ s, s < S.nS x h.length → S.kind x h.length s = .masked →
    S.check x h.length s w ρ h (S.copy wk x w ρ h) κ (wk x w ρ h s) = true →
      wk x w ρ h s = some (S.honest x h.length s w ρ h)

end ZkServer

/-! ## Cut runs -/

/-- The cut prover: `τ`'s messages before round `t`, then the stop. -/
def cutMsg (stop : Msg) (t : ℕ) (τ : List (Msg × Out)) (h : List (Msg × Msg)) : Msg :=
  if h.length < t then ((τ[h.length]?).map Prod.fst).getD stop else stop

/-- **The transcript `τ` cut at round `t`**: `n` rounds of `τ`'s messages before `t`, then the stop, against `ask`. -/
def cutT (ask : List Msg → Out) (stop : Msg) (t n : ℕ) (τ : List (Msg × Out)) : List (Msg × Out) :=
  interact ask (cutMsg stop t τ) n []

/-- The prover `pm` stopped at round `t`. -/
def stopAt (pm : List (Msg × Msg) → Msg) (t : ℕ) (stop : Msg) (h : List (Msg × Msg)) : Msg :=
  if h.length < t then pm h else stop

theorem stopAt_of_lt {pm : List (Msg × Msg) → Msg} {t : ℕ} {stop : Msg} {h : List (Msg × Msg)} (hl : h.length < t) :
    stopAt pm t stop h = pm h := by
  simp [stopAt, hl]

theorem stopAt_of_ge {pm : List (Msg × Msg) → Msg} {t : ℕ} {stop : Msg} {h : List (Msg × Msg)}
    (hl : ¬ h.length < t) : stopAt pm t stop h = stop := by
  simp [stopAt, hl]

theorem cutMsg_of_lt {stop : Msg} {t : ℕ} {τ : List (Msg × Out)} {h : List (Msg × Msg)} (hl : h.length < t) :
    cutMsg stop t τ h = ((τ[h.length]?).map Prod.fst).getD stop := by
  simp [cutMsg, hl]

theorem cutMsg_of_ge {stop : Msg} {t : ℕ} {τ : List (Msg × Out)} {h : List (Msg × Msg)} (hl : ¬ h.length < t) :
    cutMsg stop t τ h = stop := by
  simp [cutMsg, hl]

theorem length_histOf_take (τ : List (Msg × Out)) (i : ℕ) : (histOf (τ.take i)).length = min i τ.length := by
  simp [histOf]

theorem histOf_take_take (τ : List (Msg × Out)) {i i' : ℕ} (h : i' ≤ i) :
    (histOf (τ.take i)).take i' = histOf (τ.take i') := by
  simp only [histOf, ← List.map_take, List.take_take, Nat.min_eq_left h]

/-- Round `i`'s message is the prover's on the history before it. -/
theorem interact_getElem_fst (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (i : ℕ), i < (interact ask pm n hist).length →
      ((interact ask pm n hist)[i]?).map Prod.fst = some (pm (hist ++ histOf ((interact ask pm n hist).take i)))
  | 0, _, _, hi => by simp [interact] at hi
  | n + 1, hist, i, hi => by
      simp only [interact] at hi ⊢
      cases hb : ask (hist.map Prod.fst ++ [pm hist]) with
      | msg m =>
          rw [hb] at hi
          cases i with
          | zero => simp [histOf]
          | succ i =>
              have := interact_getElem_fst ask pm n (hist ++ [(pm hist, m)]) i (by simpa using hi)
              rw [List.getElem?_cons_succ, this, List.take_succ_cons]
              simp [histOf, msgOf]
      | acc =>
          rw [hb] at hi
          simp only [List.length_singleton, Nat.lt_one_iff] at hi
          subst hi
          simp [histOf]
      | rej =>
          rw [hb] at hi
          simp only [List.length_singleton, Nat.lt_one_iff] at hi
          subst hi
          simp [histOf]

/-- The stopped prover's run agrees with the prover's before the stop. -/
theorem stopAt_take (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (t n : ℕ) :
    (interact ask (stopAt pm t stop) n []).take t = (interact ask pm n []).take t := by
  cases t with
  | zero => simp
  | succ j =>
      refine interact_take_congr ask pm (stopAt pm (j + 1) stop) n [] j fun i hi hlt => ?_
      have hl : (histOf ((interact ask pm n []).take i)).length = i := by
        rw [length_histOf_take]; omega
      simp only [List.nil_append]
      exact stopAt_of_lt (by rw [hl]; omega)

/-- **The stopped prover's run is its run cut**. -/
theorem interact_stopAt (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (t n : ℕ) :
    interact ask (stopAt pm t stop) n [] = cutT ask stop t n (interact ask pm n []) := by
  symm
  unfold cutT
  refine interact_congr ask (stopAt pm t stop) (cutMsg stop t (interact ask pm n [])) n [] fun i hi => ?_
  simp only [List.nil_append]
  have hl : (histOf ((interact ask (stopAt pm t stop) n []).take i)).length = i := by
    rw [length_histOf_take]; omega
  by_cases hit : i < t
  · rw [cutMsg_of_lt (by rw [hl]; exact hit), stopAt_of_lt (by rw [hl]; exact hit), hl]
    have htk := stopAt_take ask pm stop t n
    have e1 : (interact ask (stopAt pm t stop) n []).take i =
        ((interact ask (stopAt pm t stop) n []).take t).take i := by
      rw [List.take_take, Nat.min_eq_left hit.le]
    have e2 : (interact ask pm n []).take i = ((interact ask pm n []).take t).take i := by
      rw [List.take_take, Nat.min_eq_left hit.le]
    have hiτ : i < (interact ask pm n []).length := by
      have := congrArg List.length htk
      simp only [List.length_take] at this
      omega
    rw [interact_getElem_fst ask pm n [] i hiτ, e1, htk, ← e2]
    simp
  · rw [cutMsg_of_ge (by rw [hl]; exact hit), stopAt_of_ge (by rw [hl]; exact hit)]

/-! ## The split -/

namespace ZkServer

theorem pass_of_live {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) (hl : S.live wk x w ρ κ h = true) {i : ℕ} (hi : i ≤ h.length) :
    S.pass wk x w ρ κ (h.take i) = true :=
  List.all_eq_true.mp hl i (List.mem_range.mpr (by omega))

/-- Where every round so far passed and was faithful, the copy is the honest one. -/
theorem copy_eq {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) (hl : S.live wk x w ρ κ h = true)
    (hf : ∀ i ≤ h.length, S.Faithful wk x w ρ κ (h.take i)) :
    S.copy wk x w ρ h = S.hcopy x w ρ h := by
  funext i s
  unfold copy hcopy
  split_ifs with hc
  · obtain ⟨hi, hs, hm⟩ := hc
    have hlen : (h.take i).length = i := by simp only [List.length_take]; omega
    have hp := S.pass_of_live wk x w ρ κ h hl hi
    have hcs := List.all_eq_true.mp hp s (List.mem_range.mpr (by rw [hlen]; exact hs))
    rw [hf i hi s (by rw [hlen]; exact hs) (by rw [hlen]; exact hm) hcs, hlen]
  · rfl

/-- **Where every round so far passed and was faithful, the server releases the honest frame.** -/
theorem next_eq {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) (hl : S.live wk x w ρ κ h = true)
    (hf : ∀ i ≤ h.length, S.Faithful wk x w ρ κ (h.take i)) :
    S.next wk x w ρ κ h = S.refMsg x w ρ h := by
  unfold next refMsg
  simp only [hl, ↓reduceIte]
  rw [S.copy_eq wk x w ρ κ h hl hf]
  congr 1
  funext s
  have hp := S.pass_of_live wk x w ρ κ h hl le_rfl
  rw [List.take_length] at hp
  have hcs := List.all_eq_true.mp hp s.val (List.mem_range.mpr s.isLt)
  have hf0 := hf h.length le_rfl
  rw [List.take_length] at hf0
  unfold value
  cases hk : S.kind x h.length s.val with
  | masked => rw [hf0 s.val s.isLt hk hcs]
  | fixed => rfl
  | own => rfl
  | recomputed => rfl

theorem next_stop {W R : Type} (S : ZkServer W R) (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)
    (h : List (Msg × Msg)) (hl : S.live wk x w ρ κ h = false) : S.next wk x w ρ κ h = S.stop := by
  simp [next, hl]

/-- **The split, exactly**: where every masked check along `P`'s run is faithful, the server releases `P`'s run cut
at some round `t ≤ n`. -/
theorem split_exact {W R : Type} (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R) (x : Msg) (w : W)
    (hRef : S.Ref H P x w) (wk : S.Worker) (ask : List Msg → Out) (n : ℕ) (ρ : R) (κ : S.Chk)
    (hF : ∀ i < (interact ask (fun h => (((P w).next x h ρ).eval H).1) n []).length,
      S.Faithful wk x w ρ κ (histOf ((interact ask (fun h => (((P w).next x h ρ).eval H).1) n []).take i))) :
    ∃ t ≤ n, interact ask (S.next wk x w ρ κ) n [] =
      cutT ask S.stop t n (interact ask (fun h => (((P w).next x h ρ).eval H).1) n []) := by
  classical
  set pm : List (Msg × Msg) → Msg := fun h => (((P w).next x h ρ).eval H).1 with hpm
  set τ := interact ask pm n [] with hτ
  have hτn : τ.length ≤ n := interact_length_le ask pm n []
  have hlen : ∀ i ≤ τ.length, (histOf (τ.take i)).length = i := fun i hi => by
    rw [length_histOf_take]; omega
  obtain ⟨t, htn, hgood, hbad⟩ : ∃ t ≤ n,
      (∀ i < t, i < τ.length → S.pass wk x w ρ κ (histOf (τ.take i)) = true) ∧
      (t < n → t < τ.length ∧ S.pass wk x w ρ κ (histOf (τ.take t)) = false) := by
    by_cases hex : ∃ i, i < τ.length ∧ S.pass wk x w ρ κ (histOf (τ.take i)) = false
    · refine ⟨Nat.find hex, ((Nat.find_spec hex).1.le).trans hτn, fun i hi hiτ => ?_,
        fun _ => Nat.find_spec hex⟩
      have := Nat.find_min hex hi
      simp only [not_and, Bool.not_eq_false] at this
      exact this hiτ
    · refine ⟨n, le_rfl, fun i _ hiτ => ?_, fun h => absurd h (lt_irrefl n)⟩
      simp only [not_exists, not_and, Bool.not_eq_false] at hex
      exact hex i hiτ
  refine ⟨t, htn, ?_⟩
  rw [← interact_stopAt]
  refine interact_congr ask (stopAt pm t S.stop) (S.next wk x w ρ κ) n [] fun i hi => ?_
  simp only [List.nil_append]
  have hσn : (interact ask (stopAt pm t S.stop) n []).length ≤ n := interact_length_le ask _ n []
  have hst := stopAt_take ask pm S.stop t n
  rw [← hτ] at hst
  have hgl : (histOf ((interact ask (stopAt pm t S.stop) n []).take i)).length = i := by
    rw [length_histOf_take]; omega
  by_cases hit : i < t
  · have e : (interact ask (stopAt pm t S.stop) n []).take i = τ.take i := by
      have e1 : (interact ask (stopAt pm t S.stop) n []).take i =
          ((interact ask (stopAt pm t S.stop) n []).take t).take i := by
        rw [List.take_take, Nat.min_eq_left hit.le]
      rw [e1, hst, List.take_take, Nat.min_eq_left hit.le]
    have hiτ : i < τ.length := by
      have := congrArg List.length hst
      simp only [List.length_take] at this
      omega
    rw [stopAt_of_lt (by rw [hgl]; exact hit), e,
      show pm (histOf (τ.take i)) = S.refMsg x w ρ (histOf (τ.take i)) from hRef ρ _]
    refine S.next_eq wk x w ρ κ _ ?_ fun i' hi' => ?_
    · refine List.all_eq_true.mpr fun i' hi' => ?_
      rw [List.mem_range, hlen i hiτ.le] at hi'
      rw [histOf_take_take τ (by omega : i' ≤ i)]
      exact hgood i' (by omega) (by omega)
    · rw [hlen i hiτ.le] at hi'
      rw [histOf_take_take τ hi']
      exact hF i' (by omega)
  · have hin : i < n := lt_of_lt_of_le hi hσn
    obtain ⟨-, hpf⟩ := hbad (by omega)
    rw [stopAt_of_ge (by rw [hgl]; exact hit)]
    refine S.next_stop wk x w ρ κ _ ?_
    cases hlv : S.live wk x w ρ κ (histOf ((interact ask (stopAt pm t S.stop) n []).take i)) with
    | false => rfl
    | true =>
        have hp := S.pass_of_live wk x w ρ κ _ hlv (i := t) (by rw [hgl]; omega)
        rw [histOf_take_take _ (by omega : t ≤ i), hst] at hp
        rw [hp] at hpf
        exact absurd hpf (by decide)

end ZkServer

/-! ## Probability -/

/-- **The union bound** over a finite set. -/
theorem Pr_exists_finset_le {α I : Type} [Fintype α] (F : Finset I) (E : I → α → Prop) :
    Pr (fun a => ∃ i ∈ F, E i a) ≤ ∑ i ∈ F, Pr (E i) := by
  classical
  induction F using Finset.induction_on with
  | empty => simp [Pr_false]
  | insert j F hj ih =>
      rw [Finset.sum_insert hj]
      refine le_trans (Pr_mono fun a ha => ?_) ((ZKMal.Pr_or_le _ _).trans (add_le_add le_rfl ih))
      obtain ⟨i, hi, he⟩ := ha
      rcases Finset.mem_insert.mp hi with rfl | hi
      · exact Or.inl he
      · exact Or.inr ⟨i, hi, he⟩

/-- An event of two coordinates whose every slice is within `ε`. -/
theorem Pr_le_of_slices {α β : Type} [Fintype α] [Nonempty α] [Fintype β] (p : α × β → Prop) {ε : ℝ}
    (h : ∀ a, Pr (fun b => p (a, b)) ≤ ε) : Pr p ≤ ε := by
  rw [Pr_prod_left]
  calc _ ≤ Ex (fun _ : α => ε) := Ex_mono h
    _ = ε := Ex_const ε

/-- **An observation at a choice**: an event of `obs` at the choice `τ ω`, made by anyone who sees `ω`, is within the
sum over the choices of the event at each fixed one. -/
theorem Pr_choice_le {Ω T α : Type} [Fintype Ω] [Fintype T] (obs : T → Ω → α) (τ : Ω → T) (E : α → Prop) :
    Pr (fun ω => E (obs (τ ω) ω)) ≤ ∑ t, Pr (fun ω => E (obs t ω)) :=
  (Pr_mono fun ω h => ⟨τ ω, Finset.mem_univ _, h⟩).trans
    (Pr_exists_finset_le Finset.univ fun t ω => E (obs t ω))

namespace ZkServer

/-- **The split**: for every worker, the server releases `P`'s run cut at some round, but with probability the masked
checks' errors summed over the rounds and masked slots. -/
theorem split {W R : Type} [Fintype R] [Nonempty R] (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (x : Msg) (w : W) (hRef : S.Ref H P x w) (ε : Msg → ℕ → ℕ → ℝ) (hε : ∀ j s, 0 ≤ ε x j s)
    (hS : S.CheckSound ε) (wk : S.Worker) (ask : List Msg → Out) (n : ℕ) :
    Pr (fun ω : R × S.Chk => ¬ ∃ t ≤ n, interact ask (S.next wk x w ω.1 ω.2) n [] =
      cutT ask S.stop t n (interact ask (fun h => (((P w).next x h ω.1).eval H).1) n [])) ≤
      ∑ i ∈ Finset.range n, ∑ s ∈ (Finset.range (S.nS x i)).filter (fun s => S.kind x i s = .masked), ε x i s := by
  classical
  set τ : R → List (Msg × Out) := fun ρ => interact ask (fun h => (((P w).next x h ρ).eval H).1) n [] with hτ
  set hP : R → ℕ → List (Msg × Msg) := fun ρ i => histOf ((τ ρ).take i) with hhP
  let E : ℕ → ℕ → R × S.Chk → Prop := fun i s ω =>
    i < (τ ω.1).length ∧
      S.check x i s w ω.1 (hP ω.1 i) (S.copy wk x w ω.1 (hP ω.1 i)) ω.2 (wk x w ω.1 (hP ω.1 i) s) = true ∧
      wk x w ω.1 (hP ω.1 i) s ≠ some (S.honest x i s w ω.1 (hP ω.1 i))
  have hsub : ∀ ω : R × S.Chk, (¬ ∃ t ≤ n, interact ask (S.next wk x w ω.1 ω.2) n [] =
      cutT ask S.stop t n (τ ω.1)) →
      ∃ i ∈ Finset.range n, ∃ s ∈ (Finset.range (S.nS x i)).filter (fun s => S.kind x i s = .masked), E i s ω := by
    intro ω hω
    by_contra hno
    refine hω (S.split_exact H P x w hRef wk ask n ω.1 ω.2 fun i hi s hs hm hc => ?_)
    change i < (τ ω.1).length at hi
    change s < S.nS x (hP ω.1 i).length at hs
    change S.kind x (hP ω.1 i).length s = .masked at hm
    change S.check x (hP ω.1 i).length s w ω.1 (hP ω.1 i) (S.copy wk x w ω.1 (hP ω.1 i)) ω.2
      (wk x w ω.1 (hP ω.1 i) s) = true at hc
    change wk x w ω.1 (hP ω.1 i) s = some (S.honest x (hP ω.1 i).length s w ω.1 (hP ω.1 i))
    have hl : (hP ω.1 i).length = i := by
      show (histOf ((τ ω.1).take i)).length = i
      rw [length_histOf_take]; omega
    rw [hl] at hs hm hc ⊢
    have hn : (τ ω.1).length ≤ n := interact_length_le _ _ n []
    by_contra hne
    exact hno ⟨i, Finset.mem_range.mpr (by omega), s, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hs, hm⟩,
      hi, hc, hne⟩
  calc _ ≤ Pr (fun ω : R × S.Chk => ∃ i ∈ Finset.range n,
          ∃ s ∈ (Finset.range (S.nS x i)).filter (fun s => S.kind x i s = .masked), E i s ω) := Pr_mono hsub
    _ ≤ ∑ i ∈ Finset.range n, Pr (fun ω : R × S.Chk =>
          ∃ s ∈ (Finset.range (S.nS x i)).filter (fun s => S.kind x i s = .masked), E i s ω) :=
        Pr_exists_finset_le _ _
    _ ≤ ∑ i ∈ Finset.range n, ∑ s ∈ (Finset.range (S.nS x i)).filter (fun s => S.kind x i s = .masked),
          Pr (E i s) := Finset.sum_le_sum fun i _ => Pr_exists_finset_le _ _
    _ ≤ _ := by
        refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun s hs => ?_
        have hm := (Finset.mem_filter.mp hs).2
        refine Pr_le_of_slices _ fun ρ => ?_
        by_cases hiτ : i < (τ ρ).length
        · cases hu : wk x w ρ (hP ρ i) s with
          | none =>
              refine (Pr_mono fun κ (hκ : E i s (ρ, κ)) => ?_).trans (le_of_eq_of_le Pr_false (hε i s))
              obtain ⟨-, hc, -⟩ := hκ
              simp only [check, hm, hu] at hc
              exact Bool.false_ne_true hc
          | some v =>
              by_cases hv : v = S.honest x i s w ρ (hP ρ i)
              · refine (Pr_mono fun κ (hκ : E i s (ρ, κ)) => ?_).trans (le_of_eq_of_le Pr_false (hε i s))
                obtain ⟨-, -, hne⟩ := hκ
                exact hne (by rw [hu, hv])
              · refine (Pr_mono fun κ (hκ : E i s (ρ, κ)) => ?_).trans
                  (hS x i s w ρ (hP ρ i) (S.copy wk x w ρ (hP ρ i)) v hm hv)
                obtain ⟨-, hc, -⟩ := hκ
                simp only [check, hm, hu] at hc
                exact hc
        · exact (Pr_mono fun κ (hκ : E i s (ρ, κ)) => absurd hκ.1 hiτ).trans (le_of_eq_of_le Pr_false (hε i s))

/-- **The released transcript against `P`'s**: for every worker and every verifier, an event of the server's scheduled
transcript is within the sum over the stop rounds `t ≤ k` of the event at `P`'s run cut at `t`, plus the masked
checks' errors. -/
theorem real_le {W R Θ : Type} [Fintype R] [Nonempty R] (S : ZkServer W R) (H : Msg → Msg) (P : W → Prover R)
    (x : Msg) (w : W) (hRef : S.Ref H P x w) (ε : Msg → ℕ → ℕ → ℝ) (hε : ∀ j s, 0 ≤ ε x j s)
    (hS : S.CheckSound ε) (wk : S.Worker) (Vs : VStar Θ) (θ : Θ) (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : R × S.Chk => Ev (S.timed x (runStar H (S.prover wk w) Vs x ω θ).msgs)) ≤
      ∑ t ∈ Finset.range (Vs.k x + 1), Pr (fun ρ : R => Ev (S.timed x
        (cutT (fun as => ((Vs.next x as θ).eval H).1) S.stop t (Vs.k x) (runStar H (P w) Vs x ρ θ).msgs))) +
      ∑ i ∈ Finset.range (Vs.k x),
        ∑ s ∈ (Finset.range (S.nS x i)).filter (fun s => S.kind x i s = .masked), ε x i s := by
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  have hsrv : ∀ ω : R × S.Chk, (runStar H (S.prover wk w) Vs x ω θ).msgs =
      interact ask (S.next wk x w ω.1 ω.2) (Vs.k x) [] := fun ω => by
    rw [runStar_msgs]; rfl
  have hP : ∀ ρ : R, (runStar H (P w) Vs x ρ θ).msgs =
      interact ask (fun h => (((P w).next x h ρ).eval H).1) (Vs.k x) [] := fun ρ => runStar_msgs H (P w) Vs x ρ θ
  have hsplit := S.split H P x w hRef ε hε hS wk ask (Vs.k x)
  calc _ ≤ Pr (fun ω : R × S.Chk => (¬ ∃ t ≤ Vs.k x, interact ask (S.next wk x w ω.1 ω.2) (Vs.k x) [] =
          cutT ask S.stop t (Vs.k x) (interact ask (fun h => (((P w).next x h ω.1).eval H).1) (Vs.k x) [])) ∨
          ∃ t ∈ Finset.range (Vs.k x + 1),
            Ev (S.timed x (cutT ask S.stop t (Vs.k x) (runStar H (P w) Vs x ω.1 θ).msgs))) := by
        refine Pr_mono fun ω hω => ?_
        by_cases hb : ∃ t ≤ Vs.k x, interact ask (S.next wk x w ω.1 ω.2) (Vs.k x) [] =
            cutT ask S.stop t (Vs.k x) (interact ask (fun h => (((P w).next x h ω.1).eval H).1) (Vs.k x) [])
        · obtain ⟨t, ht, he⟩ := hb
          refine Or.inr ⟨t, Finset.mem_range.mpr (by omega), ?_⟩
          rw [hP, ← he, ← hsrv]
          exact hω
        · exact Or.inl hb
    _ ≤ _ := (ZKMal.Pr_or_le _ _).trans (add_le_add hsplit (Pr_exists_finset_le _ _))
    _ = _ := by
        rw [add_comm]
        congr 1
        refine Finset.sum_congr rfl fun t _ => ?_
        exact Pr_fst (fun ρ : R => Ev (S.timed x (cutT ask S.stop t (Vs.k x) (runStar H (P w) Vs x ρ θ).msgs)))

end ZkServer

/-! ## Zero knowledge, with the stop guessed -/

/-- **The simulator that guesses the stop**: it draws a round `t ≤ k` and `Sim`'s coins, and replays `Sim`'s
transcript cut at `t` against the verifier. -/
def leaky (Sim : Simulator) (k : Msg → ℕ) (stop : Msg) : Simulator where
  C x := Fin (k x + 1) × Sim.C x
  prog x c := spBind (Sim.prog x c.2) fun τ => spRun (fun h => .ret (cutMsg stop c.1 τ h)) (k x) []

theorem leaky_out {Θ : Type} (H : Msg → Msg) (Sim : Simulator) (k : Msg → ℕ) (stop : Msg) (Vs : VStar Θ) (θ : Θ)
    (x : Msg) (c : (leaky Sim k stop).C x) :
    (leaky Sim k stop).out H Vs θ x c =
      cutT (fun as => ((Vs.next x as θ).eval H).1) stop c.1 (k x) (Sim.out H Vs θ x c.2) := by
  unfold Simulator.out leaky cutT
  rw [spBind_eval, spRun_eval]
  rfl

namespace ZkServer

/-- **Zero knowledge of the server**, from `P`'s: for every worker, if `P` is zero knowledge against a malicious
verifier (`ZKmalR`), an event of the server's scheduled transcript has probability at most `k + 1` times its
probability at the guessing simulator `leaky` plus `ε_zk + γ·Fork`, plus the masked checks' errors. The factor `k + 1`
is the stop's leak: `log₂(k + 1)` bits a session. -/
theorem zk_server {W N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (S : ZkServer W (N × Ω))
    (H : Msg → Msg) (k : Msg → ℕ) (P : W → Prover (N × Ω)) (Rl : Rel W) (Sim : Simulator) (εzk γ : Msg → ℝ)
    (hZK : ZKmalR H k P Rl Sim εzk γ) (x : Msg) (w : W) (hR : Rl x w) (hRef : S.Ref H P x w)
    (ε : Msg → ℕ → ℕ → ℝ) (hε : ∀ j s, 0 ≤ ε x j s) (hS : S.CheckSound ε) (wk : S.Worker) (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = k x) (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (N × Ω) × S.Chk => Ev (S.timed x (runStar H (S.prover wk w) Vs x ω θ).msgs)) ≤
      (k x + 1) * (Pr (fun c : (leaky Sim k S.stop).C x => Ev (S.timed x ((leaky Sim k S.stop).out H Vs θ x c))) +
        εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) +
      ∑ i ∈ Finset.range (k x),
        ∑ s ∈ (Finset.range (S.nS x i)).filter (fun s => S.kind x i s = .masked), ε x i s := by
  have hr := S.real_le H P x w hRef ε hε hS wk Vs θ Ev
  rw [hk] at hr
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  set e := εzk x + γ x * (Fork H (P w) Vs θ x).Pr H with he
  set B : ℕ → ℝ := fun t => Pr (fun c : Sim.C x => Ev (S.timed x (cutT ask S.stop t (k x) (Sim.out H Vs θ x c))))
    with hB
  have hA : ∀ t, Pr (fun ρ : N × Ω => Ev (S.timed x (cutT ask S.stop t (k x) (runStar H (P w) Vs x ρ θ).msgs))) ≤
      B t + e := fun t => by
    have := hZK x w hR Θ Vs θ hk (fun τ => Ev (S.timed x (cutT ask S.stop t (k x) τ)))
    rw [he]
    linarith [le_abs_self (Pr (fun ρ : N × Ω => Ev (S.timed x (cutT ask S.stop t (k x)
      (runStar H (P w) Vs x ρ θ).msgs))) - B t)]
  have hsim : Pr (fun c : (leaky Sim k S.stop).C x => Ev (S.timed x ((leaky Sim k S.stop).out H Vs θ x c))) =
      (∑ t ∈ Finset.range (k x + 1), B t) / (k x + 1 : ℕ) := by
    rw [Finset.sum_range (fun t => B t)]
    simp only [leaky_out]
    exact Pr_guess (fun c : Fin (k x + 1) × Sim.C x =>
      Ev (S.timed x (cutT ask S.stop c.1 (k x) (Sim.out H Vs θ x c.2))))
  have hpos : (0 : ℝ) < (k x + 1 : ℕ) := by exact_mod_cast Nat.succ_pos _
  have hsum : ∑ t ∈ Finset.range (k x + 1), Pr (fun ρ : N × Ω => Ev (S.timed x
      (cutT ask S.stop t (k x) (runStar H (P w) Vs x ρ θ).msgs))) ≤
      (k x + 1) * (Pr (fun c : (leaky Sim k S.stop).C x => Ev (S.timed x ((leaky Sim k S.stop).out H Vs θ x c))) +
        e) := by
    calc _ ≤ ∑ t ∈ Finset.range (k x + 1), (B t + e) := Finset.sum_le_sum fun t _ => hA t
      _ = (∑ t ∈ Finset.range (k x + 1), B t) + (k x + 1) * e := by
          rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
          push_cast
          ring
      _ = _ := by
          rw [hsim]
          field_simp
          push_cast
          ring
  linarith

/-- **Exact recomputation is sound**: a masked check that compares with the honest value meets `CheckSound` at `0`. -/
theorem checkSound_exact {W R : Type} (S : ZkServer W R)
    (hrel : ∀ x j s w ρ h cp κ v, S.rel x j s w ρ h cp κ v = decide (v = S.honest x j s w ρ h)) :
    S.CheckSound fun _ _ _ => 0 := by
  intro x j s w ρ h cp v _ hv
  refine le_of_eq (Eq.trans (Pr_congr fun κ => ?_) Pr_false)
  rw [hrel]
  simp [hv]

end ZkServer

/-! ## The stop's leak over a stream -/

/-- **A stream's stop pattern**: each of `m` sessions runs through (`none`) or stops in one of `r` forms, at most `b`
of them stopping, the firewall's per-stream failure budget. -/
abbrev Pattern (m r b : ℕ) : Type :=
  {f : Fin m → Option (Fin r) // (Finset.univ.filter fun i => (f i).isSome).card ≤ b}

/-- **The number of stop patterns** is at most `∑_{i ≤ b} C(m, i)·r^i`. -/
theorem card_pattern_le (m r b : ℕ) :
    Fintype.card (Pattern m r b) ≤ ∑ i ∈ Finset.range (b + 1), m.choose i * r ^ i := by
  classical
  let T : Finset (Fin m) → Fin m → Finset (Option (Fin r)) := fun s i =>
    if i ∈ s then Finset.univ.map Function.Embedding.some else {none}
  let F := (Finset.univ : Finset (Fin m)).powerset.filter fun s => s.card ≤ b
  have hsub : (Finset.univ.filter fun f : Fin m → Option (Fin r) =>
      (Finset.univ.filter fun i => (f i).isSome).card ≤ b) ⊆ F.biUnion fun s => Fintype.piFinset (T s) := by
    intro f hf
    have hb := (Finset.mem_filter.mp hf).2
    refine Finset.mem_biUnion.mpr ⟨Finset.univ.filter fun i => (f i).isSome,
      Finset.mem_filter.mpr ⟨Finset.mem_powerset.mpr (Finset.subset_univ _), hb⟩,
      Fintype.mem_piFinset.mpr fun i => ?_⟩
    simp only [T, Finset.mem_filter, Finset.mem_univ, true_and]
    cases hfi : f i with
    | none => simp
    | some a => simp
  have hcard : ∀ s : Finset (Fin m), (Fintype.piFinset (T s)).card = r ^ s.card := fun s => by
    rw [Fintype.card_piFinset]
    have hc : ∀ i, (T s i).card = if i ∈ s then r else 1 := fun i => by
      simp only [T]
      split_ifs <;> simp
    rw [Finset.prod_congr rfl fun i _ => hc i, Finset.prod_ite_mem, Finset.univ_inter, Finset.prod_const]
  rw [Fintype.card_subtype]
  calc _ ≤ (F.biUnion fun s => Fintype.piFinset (T s)).card := Finset.card_le_card hsub
    _ ≤ ∑ s ∈ F, (Fintype.piFinset (T s)).card := Finset.card_biUnion_le
    _ = ∑ s ∈ (Finset.univ : Finset (Fin m)).powerset, if s.card ≤ b then r ^ s.card else 0 := by
        rw [Finset.sum_filter]
        exact Finset.sum_congr rfl fun s _ => by rw [hcard]
    _ = ∑ i ∈ Finset.range (m + 1), m.choose i * (if i ≤ b then r ^ i else 0) := by
        rw [Finset.sum_powerset_apply_card (fun i => if i ≤ b then r ^ i else 0)]
        simp [smul_eq_mul]
    _ = ∑ i ∈ (Finset.range (m + 1)).filter (· ≤ b), m.choose i * r ^ i := by
        rw [Finset.sum_filter]
        exact Finset.sum_congr rfl fun i _ => by split_ifs <;> simp
    _ ≤ _ := Finset.sum_le_sum_of_subset fun i hi => by
        simp only [Finset.mem_filter, Finset.mem_range] at hi ⊢
        omega

/-- **The stop's leak over a stream**: if every fixed stop pattern makes an event at most `q` likely, the pattern the
workers choose, from anything they see, makes it at most `∑_{i ≤ b} C(m, i)·r^i` times `q`: the stream leaks at most
`log₂ ∑_{i ≤ b} C(m, i)·r^i` bits. -/
theorem stream_leak {Ω α : Type} [Fintype Ω] (m r b : ℕ) (obs : Pattern m r b → Ω → α) (π : Ω → Pattern m r b)
    (E : α → Prop) {q : ℝ} (hq : ∀ p, Pr (fun ω => E (obs p ω)) ≤ q) :
    Pr (fun ω => E (obs (π ω) ω)) ≤ ((∑ i ∈ Finset.range (b + 1), m.choose i * r ^ i : ℕ) : ℝ) * q := by
  have hq0 : 0 ≤ q := (Pr_nonneg _).trans (hq ⟨fun _ => none, by simp⟩)
  calc _ ≤ ∑ p, Pr (fun ω => E (obs p ω)) := Pr_choice_le obs π E
    _ ≤ ∑ _p : Pattern m r b, q := Finset.sum_le_sum fun p _ => hq p
    _ = (Fintype.card (Pattern m r b) : ℝ) * q := by rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
    _ ≤ _ := mul_le_mul_of_nonneg_right (by exact_mod_cast card_pattern_le m r b) hq0

/-! ## At Flock's reference prover -/

open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The ZK server at `flock-circuit --zk`'s reference prover**: for every worker, the server whose frame at honest
workers is `refP`'s message (`ZkServer.Ref`) and whose masked checks are sound (`ZkServer.CheckSound`) releases, on
every event, at most `D.V.k x + 2` times the guessing simulator's probability plus `ε_zk + γ·Fork` of
`Flock.zk_deployZk_ref`, plus the checks' errors, under `Claims.Hm96Hiding` and the session's five Props. -/
theorem Flock.zk_server_ref (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg)
    (hc : S.Causal) (hT : S.TapeOK) (srv : ZkServer Flock.Wit (S.refP.Nonce × S.refP.Rand)) (x : Msg) (w : Flock.Wit)
    (hR : Flock.R (FlockDeploy.deployZk rel).parse x w) (hRef : srv.Ref H S.refP.prover x w)
    (ε : Msg → ℕ → ℕ → ℝ) (hε : ∀ j s, 0 ≤ ε x j s) (hCS : srv.CheckSound ε) (wk : srv.Worker) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × srv.Chk => Ev (srv.timed x (runStar H (srv.prover wk w) Vs x ω θ).msgs)) ≤
      ((FlockDeploy.deployZk rel).V.k x + 1 + 1) *
        (Pr (fun c : (leaky ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).C x =>
          Ev (srv.timed x ((leaky ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).out H Vs θ x c))) +
          (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H (S.refP.prover w) Vs θ x).Pr H) +
      ∑ i ∈ Finset.range ((FlockDeploy.deployZk rel).V.k x + 1),
        ∑ s ∈ (Finset.range (srv.nS x i)).filter (fun s => srv.kind x i s = .masked), ε x i s := by
  have h := srv.zk_server H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) S.refP.prover _ _ _ _
    (Flock.zk_deployZk_ref H rel S t ht hT1 hδ₁ hS hV hD hc hT).2 x w hR hRef ε hε hCS wk Vs θ hk Ev
  push_cast at h ⊢
  exact h

end Claims.ZKServer
