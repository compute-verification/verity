module

public import Proofs.Flock.ZkAny.RelayLoopStart

/-!
# The relay's auditor as `V*`

`LoopRead` asks at the `Hello`, the `Commit` and each frame round that the auditor answered as `V*` did
(`LoopModel.Ans`). Here `V*` is the relay's auditor itself, fed what the relay sends for the model's messages, and
`LoopReads` follows from what the relay reads at each slot without those answers:

* `audT`, `audView`: the auditor's view of the model's messages: each control slot's frame at the setup (`ctlOf`), and
  at each model slot the message's frames (`encK`), in the schedule's order (`tmplOf`), up to the last message's.
* `auditorV`: `V*` from an auditor `A` (its tape, then its view, to its answer): its answer to the model's messages so
  far is `A`'s at their view, bytes as a message (`audOut`), and no answer a rejection.
* `runStar_auditorV`: the model's run against `auditorV` answers each message as `A` does at the view of the messages
  through it, and goes on while `A` answers and rounds are left. `LoopModel.ans_auditorV`: so `Ans` holds at slot `j`
  for a request whose answer is `A`'s at that view, or none.
* `AudReads`: the relay's answers are the auditor's: at the `Hello`, the `Commit` and each frame round, `A`'s answer to
  everything that left so far, and none where nothing left there.
* `ReadsBy`: `LoopReads`' shape at any reading `Rd` of a slot. `AnsPart`: what `LoopRead` asks beyond such a reading,
  `Ans` at the `Hello`, the `Commit` and each frame round, and a message of the run at the `Finish`.
* **`loopReads_auditorV`**: where the model's run is against `auditorV`, the schedule has no more model slots than
  `V*` has rounds, each reading `Rd` with `AnsPart` gives `LoopRead`, and a reading holds with the answer erased,
  `LoopReads` holds wherever `EnvPre` follows from it (`hF`), the relay reads `Rd` at each slot and its answers are the
  auditor's. At a model slot the request with its answer erased is read so too, so `hF` gives `EnvOk` there, which
  reads no answer; then what left is the message's frames (`step_model`), the auditor's view the model's, and its
  answer `V*`'s.
* **`envPre_auditorV`**: `envPre_run` at `V* = auditorV`, its `LoopReads` from `ReadsBy` and `AudReads`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (V SessionSpec Stmt)

/-- **The auditor's view of the model's messages `as`** over the slots `ks`, after the rounds of the streams `seen`:
each control slot's frame `ctl k`, and at each model slot the next message's frames (`encK`), up to the last
message's. -/
def audT (ctl : Session.Kind → ByteArray) (spec : SessionSpec) :
    List String → List Session.Kind → List Claims.Msg → List ByteArray
  | _, [], _ => []
  | _, _ :: _, [] => []
  | seen, k :: ks, m :: ms =>
    if isModel k then encK spec seen k m ++ audT ctl spec (seenAfter seen k) ks ms
    else ctl k :: audT ctl spec (seenAfter seen k) ks (m :: ms)

/-- **The auditor's view of the model's messages `as`** at the setup `spec`, `sts`: the schedule's frames through the
last message's. -/
def audView (spec : SessionSpec) (sts : Array Stmt) (as : List Claims.Msg) : List ByteArray :=
  audT (ctlOf (.ok (spec, sts))) spec [] (tmplOf (.ok (spec, sts))) as

/-- An auditor's answer as `V*`'s: its bytes as a message, and no answer a rejection. -/
def audOut : Option ByteArray → Claims.Out
  | some a => .msg a.data.toList
  | none => .rej

/-- **`V*` from the relay's auditor `A`**, at `k` rounds: its answer to the model's messages so far is `A`'s at their
view at the setup `spec`, `sts` (`audView`), on its tape. -/
def auditorV {Θ : Type} (k : Claims.Msg → ℕ) (spec : SessionSpec) (sts : Array Stmt)
    (A : Θ → List ByteArray → Option ByteArray) : Claims.VStar Θ where
  k := k
  next _ as θ := .ret (audOut (A θ (audView spec sts as)))

/-- **The model's run against `auditorV` is the auditor's**: each of its messages answered as `A` answers the view of
the messages through it, and a message at each round while `A` answered every earlier one. -/
theorem runStar_auditorV {Ω Θ : Type} {H : Claims.Msg → Claims.Msg} {P : Claims.Prover Ω} {k : Claims.Msg → ℕ}
    {spec : SessionSpec} {sts : Array Stmt} {A : Θ → List ByteArray → Option ByteArray} {x : Claims.Msg} {ω : Ω}
    {θ : Θ} {τ : List (Claims.Msg × Claims.Out)}
    (hτ : τ = (Claims.runStar H P (auditorV k spec sts A) x ω θ).msgs) :
    (∀ j (hj : j < τ.length), (τ[j]'hj).2 = audOut (A θ (audView spec sts ((τ.take (j + 1)).map (·.1))))) ∧
      ∀ j, j < k x → (∀ i < j, (A θ (audView spec sts ((τ.take (i + 1)).map (·.1)))).isSome) → j < τ.length := by
  have e : τ = ZKMal.interact (fun as => audOut (A θ (audView spec sts as)))
      (fun h => ((P.next x h ω).eval H).1) (k x) [] :=
    hτ.trans (ZKMal.runStar_msgs _ _ _ _ _ _)
  have h1 : ∀ j (hj : j < τ.length),
      (τ[j]'hj).2 = audOut (A θ (audView spec sts ((τ.take (j + 1)).map (·.1)))) := by
    intro j hj
    subst e
    exact ZKMal.interact_snd _ _ _ j hj
  refine ⟨h1, fun j => ?_⟩
  induction j with
  | zero =>
    intro hk _
    obtain ⟨n, hn⟩ : ∃ n, k x = n + 1 := ⟨k x - 1, by omega⟩
    rw [e, hn]
    simp only [ZKMal.interact]
    split <;> simp
  | succ i ih =>
    intro hk hpre
    have hi : i < τ.length := ih (by omega) fun i' hi' => hpre i' (by omega)
    obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (hpre i (by omega))
    have hr : ZKMal.resp τ i = some a.data.toList := by
      rw [ZKMal.resp_eq_some]
      exact ⟨hi, (h1 i hi).trans (congrArg audOut ha)⟩
    rw [e] at hr ⊢
    exact ZKMal.interact_length_of_resp _ _ _ _ _ _ hr hk

/-- **The auditor answered the model's `j`-th message as `auditorV` did**: where the model's run is against
`auditorV`, `j` is below its rounds and `A` answered the view of every earlier message, a request whose answer is
none or `A`'s at the view of the messages through the `j`-th. -/
theorem LoopModel.ans_auditorV {W Rr Ω Θ : Type} {M : LoopModel W Rr} {H : Claims.Msg → Claims.Msg}
    {P : Claims.Prover Ω} {ω : Ω} {θ : Θ} {k : Claims.Msg → ℕ} {spec : SessionSpec} {sts : Array Stmt}
    {A : Θ → List ByteArray → Option ByteArray}
    (hτ : M.τ = (Claims.runStar H P (auditorV k spec sts A) M.x ω θ).msgs) {j : Nat} (hj : j < k M.x)
    (hpre : ∀ i < j, (A θ (audView spec sts (M.msgs.take (i + 1)))).isSome) {e : Env}
    (he : e.ans = none ∨ e.ans = A θ (audView spec sts (M.msgs.take (j + 1)))) : M.Ans j e := by
  obtain ⟨h1, h2⟩ := runStar_auditorV hτ
  have ht : ∀ i, (M.τ.take i).map (·.1) = M.msgs.take i := fun i => by rw [List.map_take]; rfl
  have hl : j < M.τ.length := h2 j hj fun i hi => by rw [ht]; exact hpre i hi
  refine ⟨hl, fun a ha _ => ?_⟩
  rw [h1 j hl, ht]
  rcases he with he | he
  · rw [he] at ha
    cases ha
  · exact congrArg audOut (he.symm.trans ha)

/-- **The relay's answers are the auditor's** `aud`, the frames that left so far being `v`: at the `Hello`, the
`Commit` and each frame round, its answer to everything that left through the slot, and none where nothing left
there; and so at each slot to come, up to the session's stop or end. -/
def AudReads (aud : List ByteArray → Option ByteArray) : List ByteArray → State → Placed → List Env → Prop
  | _, _, _, [] => True
  | v, st, pl, e :: rest => ∀ k, slot st pl = some k →
    (isModel k = true → k ≠ .finish →
      e.ans = if (stepK true st pl (some (tagOf k)) (.ok k) e).sent.isEmpty then none
        else aud (v ++ (stepK true st pl (some (tagOf k)) (.ok k) e).sent)) ∧
    (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
        (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
      AudReads aud (v ++ (stepK true st pl (some (tagOf k)) (.ok k) e).sent)
        (stepK true st pl (some (tagOf k)) (.ok k) e).st (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest)

/-- **The relay's requests `es` to come, each read as `Rd` reads its slot**, from the session `st`, placement `pl` and
the loop at `σ`: `LoopReads`' shape at the reading `Rd`. -/
inductive ReadsBy (Rd : Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop) :
    LoopAt → State → Placed → List Env → Prop
  | nil (σ : LoopAt) (st : State) (pl : Placed) : ReadsBy Rd σ st pl []
  | stop {σ : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} :
      slot st pl = none → ReadsBy Rd σ st pl (e :: rest)
  | step {σ σ' : LoopAt} {st : State} {pl : Placed} {e : Env} {rest : List Env} (k : Session.Kind) :
      slot st pl = some k → Rd k st pl e σ σ' →
      (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
          (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
        ReadsBy Rd σ' (stepK true st pl (some (tagOf k)) (.ok k) e).st
          (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest) →
      ReadsBy Rd σ st pl (e :: rest)

/-- **What `LoopRead` asks at a slot of kind `k` beyond what the relay reads**, the session's slots having taken `c`
of the model's messages: at the `Hello`, the `Commit` and each frame round the auditor answering as `V*` did, and at
the `Finish` the run having a message there. -/
def AnsPart {W Rr : Type} (M : LoopModel W Rr) (c : Nat) (e : Env) : Session.Kind → Prop
  | .hello => M.Ans c e
  | .commit => M.Ans c e
  | .round _ _ => M.Ans c e
  | .finish => c < M.τ.length
  | _ => True

theorem ansPart_of {W Rr : Type} {M : LoopModel W Rr} {c : Nat} {e : Env} {k : Session.Kind}
    (hm : isModel k = true) (hf : k ≠ .finish) (hA : M.Ans c e) : AnsPart M c e k := by
  cases k <;> first | exact hA | exact absurd rfl hf | cases hm

theorem ansPart_ctl {W Rr : Type} {M : LoopModel W Rr} {c : Nat} {e : Env} {k : Session.Kind}
    (hm : isModel k = false) : AnsPart M c e k := by
  cases k <;> first | exact True.intro | cases hm

/-- At a model slot `LoopRead` takes one of the model's messages. -/
theorem LoopRead.c_model {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {k : Session.Kind} {st : State} {pl : Placed} {e : Env} {σ σ' : LoopAt}
    (h : LoopRead M sid spec sts nonce k st pl e σ σ') (hm : isModel k = true) : σ'.c = σ.c + 1 := by
  cases k with
  | hello => exact h.2.2.2.1
  | commit => exact h.2.2.2.2.2.2
  | round s n => exact h.2.2.1
  | finish => exact h.2.2
  | _ => cases hm

/-- At a control slot `LoopRead` takes none of the model's messages. -/
theorem LoopRead.c_ctl {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {k : Session.Kind} {st : State} {pl : Placed} {e : Env} {σ σ' : LoopAt}
    (h : LoopRead M sid spec sts nonce k st pl e σ σ') (hm : isModel k = false) : σ'.c = σ.c := by
  cases k with
  | hello => cases hm
  | commit => cases hm
  | round s n => cases hm
  | finish => cases hm
  | _ => exact h.2.1

theorem audT_nil (ctl : Session.Kind → ByteArray) (spec : SessionSpec) (seen : List String) (ks : List Session.Kind) :
    audT ctl spec seen ks [] = [] := by
  cases ks <;> simp only [audT]

theorem audT_model {ctl : Session.Kind → ByteArray} {spec : SessionSpec} {seen : List String} {k : Session.Kind}
    {ks : List Session.Kind} {m : Claims.Msg} {ms : List Claims.Msg} (hm : isModel k = true) :
    audT ctl spec seen (k :: ks) (m :: ms) = encK spec seen k m ++ audT ctl spec (seenAfter seen k) ks ms := by
  simp only [audT, hm, ↓reduceIte]

theorem audT_ctl {ctl : Session.Kind → ByteArray} {spec : SessionSpec} {seen : List String} {k : Session.Kind}
    {ks : List Session.Kind} {m : Claims.Msg} {ms : List Claims.Msg} (hm : isModel k = false) :
    audT ctl spec seen (k :: ks) (m :: ms) = ctl k :: audT ctl spec (seenAfter seen k) ks (m :: ms) := by
  simp only [audT, hm, Bool.false_eq_true, ↓reduceIte]

/-- `EnvOk` reads no answer. -/
theorem envOk_ans {spec : SessionSpec} {seen : List String} {k : Session.Kind} {e : Env} {m : Claims.Msg}
    (a : Option ByteArray) : EnvOk spec seen k { e with ans := a } m ↔ EnvOk spec seen k e m := by
  cases k <;> exact Iff.rfl

/-- A model slot but the `Finish` sends at most one frame. -/
theorem encK_length_one {spec : SessionSpec} {seen : List String} {k : Session.Kind} {m : Claims.Msg}
    (hf : k ≠ .finish) : (encK spec seen k m).length ≤ 1 := by
  cases k with
  | finish => exact absurd rfl hf
  | hello => exact Nat.le_refl _
  | commit =>
    simp only [encK]
    split <;> simp
  | round s n =>
    simp only [encK]
    split <;> simp
  | _ => exact Nat.zero_le _

theorem prefix_eq_of_le_one {α : Type} {l l' : List α} (h : l <+: l') (hne : l ≠ []) (hl : l'.length ≤ 1) :
    l = l' :=
  h.eq_of_length (by have := h.length_le; have := List.length_pos_iff.mpr hne; omega)

theorem take_succ_headD {α : Type} {l : List α} {n : Nat} {d : α} (h : n < l.length) :
    l.take (n + 1) = l.take n ++ [(l.drop n).headD d] := by
  rw [List.take_succ_eq_append_getElem h, List.drop_eq_getElem_cons h]; rfl

theorem schedTrace_cons {st : State} {pl : Placed} {k : Session.Kind} {e : Env} {tail : List Env}
    (hsl : slot st pl = some k) :
    schedTrace true st pl (e :: tail) = (k, (stepK true st pl (some (tagOf k)) (.ok k) e).sent) ::
      if ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
          (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true then
        schedTrace true (stepK true st pl (some (tagOf k)) (.ok k) e).st
          (stepK true st pl (some (tagOf k)) (.ok k) e).pl tail
      else [] := by
  simp only [schedTrace, hsl]

theorem schedSteps_cons {st : State} {pl : Placed} {k : Session.Kind} {e : Env} {tail : List Env}
    (hsl : slot st pl = some k) :
    schedSteps true st pl (e :: tail) = (k, st, pl, e) ::
      if ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
          (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true then
        schedSteps true (stepK true st pl (some (tagOf k)) (.ok k) e).st
          (stepK true st pl (some (tagOf k)) (.ok k) e).pl tail
      else [] := by
  simp only [schedSteps, hsl]

/-- **The relay goes on past a `Hello`, a `Commit` or a frame round only with the auditor's answer.** -/
theorem ans_of_go {st : State} {pl : Placed} {k : Session.Kind} {e : Env} (h : st.stop = none)
    (hm : isModel k = true) (hf : k ≠ .finish)
    (hgo : ((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
      (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true) : e.ans.isSome := by
  obtain ⟨hg, hn⟩ := go_live hgo
  have hk : k ≠ .ping ∧ k ≠ .proof := by
    constructor <;> rintro rfl <;> cases hm
  rcases hr : st.request pl k with _ | ⟨st', o⟩
  · by_cases hh : k = .hello
    · subst hh
      rw [stepK_hello hr] at hg hn
      rcases hsh : st.sendHello e.setup e.nonce with ⟨st', o⟩ | ⟨st', o⟩
      · rw [hsh] at hn
        exact absurd hn (Option.isSome_iff_ne_none.mp (sendHello_error h hsh))
      · rcases ha : e.ans with _ | r
        · rw [hsh, ha] at hg
          simp at hg
        · rfl
    · rcases stepK_frame_ans hh hf hk.1 hk.2 hr hg with hst | ⟨r, ha, -⟩
      · exact absurd hn (Option.isSome_iff_ne_none.mp hst)
      · simp [ha]
  · obtain ⟨h9, h5, h6⟩ := tagOf_ne hk.1 hk.2 hf
    rw [stepK_refused h9 h5 h6 hr] at hn
    exact absurd hn (Option.isSome_iff_ne_none.mp (request_stop h hr))

/-- **`loopReads_auditorV`'s induction**, from the loop at `σ`, the session `st` going on with its setup's Σ and
domains, the frames `v` left so far, the streams `seen` and the slots `ksr` to come: where the auditor's view of the
model's messages from the `σ.c`-th on is `v` and theirs over `ksr`, `ksr` has no more model slots than `V*` has rounds
left, `A` answered the view of each message the slots took, `EnvPre` follows from `LoopReads` here, and what leaves
from here is at a prefix of `ksr` with each control slot's frame the setup's. -/
theorem loopReads_aud {W Rr Ω Θ : Type} {M : LoopModel W Rr} {H : Claims.Msg → Claims.Msg}
    {P : Claims.Prover Ω} {ω : Ω} {θ : Θ} {k : Claims.Msg → ℕ} {spec : SessionSpec} {sts : Array Stmt}
    {A : Θ → List ByteArray → Option ByteArray}
    (hτ : M.τ = (Claims.runStar H P (auditorV k spec sts A) M.x ω θ).msgs)
    {sid : Nat} {nonce : ByteArray} {Rd : Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop}
    (hRd : ∀ kk st pl e σ σ', Rd kk st pl e σ σ' → AnsPart M σ.c e kk →
      LoopRead M sid spec sts nonce kk st pl e σ σ')
    (hnone : ∀ kk st pl e σ σ', Rd kk st pl e σ σ' → Rd kk st pl { e with ans := none } σ σ') :
    ∀ {σ : LoopAt} {st : State} {pl : Placed} {rest : List Env}, ReadsBy Rd σ st pl rest →
      ∀ (v : List ByteArray) (seen : List String) (ksr : List Session.Kind),
      st.stop = none → SetupOk spec st →
      (∀ m ms, audView spec sts (M.msgs.take σ.c ++ m :: ms) =
        v ++ audT (ctlOf (.ok (spec, sts))) spec seen ksr (m :: ms)) →
      σ.c + ksr.countP isModel ≤ k M.x →
      (∀ i < σ.c, (A θ (audView spec sts (M.msgs.take (i + 1)))).isSome) →
      (∀ tail, (∀ e ∈ tail, e.setup = .ok (spec, sts)) → LoopReads M sid spec sts nonce σ st pl tail →
        EnvPre spec seen (schedSteps true st pl tail) (M.msgs.drop σ.c)) →
      (∀ tail, (∀ e ∈ tail, e.setup = .ok (spec, sts)) →
        (schedTrace true st pl tail).map (·.1) <+: ksr ∧
          Framed (ctlOf (.ok (spec, sts))) (schedTrace true st pl tail)) →
      (∀ e ∈ rest, e.setup = .ok (spec, sts)) → (∀ e ∈ rest, ∀ why, (e.barrier (some why)).isSome) →
      AudReads (A θ) v st pl rest →
      LoopReads M sid spec sts nonce σ st pl rest := by
  intro σ st pl rest hR
  induction hR with
  | nil σ st pl => intros; exact LoopReads.nil _ _ _
  | stop h => intros; exact LoopReads.stop h
  | @step σ σ' st pl e rest kk hsl hrd _ ih =>
    intro v seen ksr hst hσ hview hcnt hpre hKE hKT hset hbar haud
    have he_s := hset e List.mem_cons_self
    have he_b := hbar e List.mem_cons_self
    have hrest_s : ∀ e' ∈ rest, e'.setup = .ok (spec, sts) := fun e' he' => hset e' (List.mem_cons_of_mem _ he')
    have hrest_b : ∀ e' ∈ rest, ∀ why, (e'.barrier (some why)).isSome :=
      fun e' he' => hbar e' (List.mem_cons_of_mem _ he')
    have hcons : ∀ tail : List Env, (∀ e' ∈ tail, e'.setup = .ok (spec, sts)) →
        ∀ e' ∈ e :: tail, e'.setup = .ok (spec, sts) := by
      intro tail hts e' he'
      rcases List.mem_cons.mp he' with rfl | he'
      · exact he_s
      · exact hts e' he'
    simp only [AudReads] at haud
    obtain ⟨haA, haC⟩ := haud kk hsl
    obtain ⟨h1, h2⟩ := hKT [e] (hcons [] fun _ h => by cases h)
    simp only [schedTrace_cons hsl, List.map_cons] at h1
    rw [schedTrace_cons hsl] at h2
    obtain ⟨ksr', rfl, -⟩ := prefix_cons_inv h1
    have hKT' : ∀ hgo : ((stepK true st pl (some (tagOf kk)) (.ok kk) e).go &&
        (stepK true st pl (some (tagOf kk)) (.ok kk) e).st.stop.isNone) = true,
        ∀ tail, (∀ e' ∈ tail, e'.setup = .ok (spec, sts)) →
          (schedTrace true (stepK true st pl (some (tagOf kk)) (.ok kk) e).st
              (stepK true st pl (some (tagOf kk)) (.ok kk) e).pl tail).map (·.1) <+: ksr' ∧
            Framed (ctlOf (.ok (spec, sts))) (schedTrace true (stepK true st pl (some (tagOf kk)) (.ok kk) e).st
              (stepK true st pl (some (tagOf kk)) (.ok kk) e).pl tail) := by
      intro hgo tail hts
      obtain ⟨g1, g2⟩ := hKT (e :: tail) (hcons tail hts)
      rw [schedTrace_cons hsl, ite_eq_left hgo, List.map_cons] at g1
      rw [schedTrace_cons hsl, ite_eq_left hgo] at g2
      exact ⟨(List.cons_prefix_cons.mp g1).2, g2.2⟩
    cases hmk : isModel kk with
    | false =>
      have hLR := hRd _ _ _ _ _ _ hrd (ansPart_ctl hmk)
      have hcc := LoopRead.c_ctl hLR hmk
      refine LoopReads.step kk hsl hLR fun hgo => ?_
      obtain ⟨hg, hn⟩ := go_live hgo
      have hsent : (stepK true st pl (some (tagOf kk)) (.ok kk) e).sent = [ctlOf (.ok (spec, sts)) kk] := by
        rcases sent_ctl e hst hsl hmk with hs1 | ⟨-, hs2⟩
        · rcases h2.1 hmk with hf | ⟨hf, -⟩
          · exact hf
          · rw [hs1] at hf
            cases hf
        · exact absurd hn (Option.isSome_iff_ne_none.mp hs2)
      have hcp : (kk :: ksr').countP isModel = ksr'.countP isModel := List.countP_cons_of_neg (by simp [hmk])
      refine ih hgo _ (seenAfter seen kk) ksr' hn (stepK_setup he_s hσ) ?_ (by omega) (by rw [hcc]; exact hpre)
        ?_ (hKT' hgo) hrest_s hrest_b (haC hgo)
      · intro m ms
        rw [hcc, hview, audT_ctl hmk, hsent, List.append_assoc, List.singleton_append]
      · intro tail hts hLt
        have hE := hKE (e :: tail) (hcons tail hts) (LoopReads.step kk hsl hLR fun _ => hLt)
        rw [schedSteps_cons hsl, ite_eq_left hgo] at hE
        simp only [EnvPre, hmk, Bool.false_eq_true, ↓reduceIte] at hE
        rw [hcc]
        exact hE
    | true =>
      have hcp : (kk :: ksr').countP isModel = ksr'.countP isModel + 1 := List.countP_cons_of_pos hmk
      have hj : σ.c < k M.x := by omega
      obtain ⟨hl, -⟩ := LoopModel.ans_auditorV (e := { e with ans := none }) hτ hj hpre (.inl rfl)
      have hlm : σ.c < M.msgs.length := by rw [LoopModel.msgs, List.length_map]; exact hl
      have hEO : EnvOk spec seen kk { e with ans := none } ((M.msgs.drop σ.c).headD []) := by
        have hAP : AnsPart M σ.c { e with ans := none } kk := by
          by_cases hfin : kk = .finish
          · subst hfin
            exact hl
          · exact ansPart_of hmk hfin
              (LoopModel.ans_auditorV (e := { e with ans := none }) hτ hj hpre (.inl rfl))
        have hLR' := hRd _ _ _ _ _ _ (hnone _ _ _ _ _ _ hrd) hAP
        have hE := hKE [{ e with ans := none }]
          (fun e' he' => by rw [List.mem_singleton] at he'; subst he'; exact he_s)
          (LoopReads.step kk hsl hLR' fun _ => LoopReads.nil _ _ _)
        rw [schedSteps_cons hsl] at hE
        simp only [EnvPre, hmk, ↓reduceIte] at hE
        exact hE.1
      have hsm := step_model hst hsl hmk hσ he_s he_b ((envOk_ans (e := e) none).mp hEO)
      have hview1 : audView spec sts (M.msgs.take (σ.c + 1)) =
          v ++ encK spec seen kk ((M.msgs.drop σ.c).headD []) := by
        rw [take_succ_headD (d := []) hlm, hview, audT_model hmk, audT_nil, List.append_nil]
      have hAP : AnsPart M σ.c e kk := by
        by_cases hfin : kk = .finish
        · subst hfin
          exact hl
        · refine ansPart_of hmk hfin (LoopModel.ans_auditorV hτ hj hpre ?_)
          rw [haA hmk hfin]
          by_cases hse : (stepK true st pl (some (tagOf kk)) (.ok kk) e).sent.isEmpty = true
          · rw [ite_eq_left hse]
            exact .inl rfl
          · rw [ite_eq_right hse]
            refine .inr ?_
            have hne : (stepK true st pl (some (tagOf kk)) (.ok kk) e).sent ≠ [] :=
              fun h => hse (List.isEmpty_iff.mpr h)
            rw [prefix_eq_of_le_one hsm.1 hne (encK_length_one hfin), hview1]
      have hLR := hRd _ _ _ _ _ _ hrd hAP
      have hcc := LoopRead.c_model hLR hmk
      refine LoopReads.step kk hsl hLR fun hgo => ?_
      obtain ⟨hg, hn⟩ := go_live hgo
      have hfin : kk ≠ .finish := by
        rintro rfl
        rw [stepK_finish] at hg
        cases hg
      have hsent := hsm.2 hg hn
      refine ih hgo _ (seenAfter seen kk) ksr' hn (stepK_setup he_s hσ) ?_ (by omega) ?_ ?_ (hKT' hgo)
        hrest_s hrest_b (haC hgo)
      · intro m ms
        rw [hcc, take_succ_headD (d := []) hlm, List.append_assoc (M.msgs.take σ.c), List.singleton_append, hview,
          audT_model hmk, hsent, List.append_assoc]
      · intro i hi
        rw [hcc] at hi
        by_cases hie : i < σ.c
        · exact hpre i hie
        · obtain rfl : i = σ.c := by omega
          have hsome := ans_of_go hst hmk hfin hgo
          rw [haA hmk hfin] at hsome
          by_cases hse : (stepK true st pl (some (tagOf kk)) (.ok kk) e).sent.isEmpty = true
          · rw [ite_eq_left hse] at hsome
            simp at hsome
          · rw [ite_eq_right hse, hsent, ← hview1] at hsome
            exact hsome
      · intro tail hts hLt
        have hE := hKE (e :: tail) (hcons tail hts) (LoopReads.step kk hsl hLR fun _ => hLt)
        rw [schedSteps_cons hsl, ite_eq_left hgo] at hE
        simp only [EnvPre, hmk, ↓reduceIte] at hE
        rw [hcc, ← List.tail_drop]
        exact hE.2

/-- **The relay's requests read as the loop answers them, from what the relay reads and the auditor's answers**: where
the model's run is against `auditorV` (`hτ`), the session's schedule has no more model slots than `V*` has rounds
(`hK`), a reading `Rd` with what `LoopRead` asks beyond it gives `LoopRead` (`hRd`) and holds with the answer erased
(`hnone`), and `EnvPre` follows from `LoopReads` at the session's first request (`hF`): at the relay's requests `es`,
each reading the setup and passing no hold's barrier at a stop, read as `Rd` reads each slot and answered by the
auditor (`AudReads`), `LoopReads` holds. -/
theorem loopReads_auditorV {W Rr Ω Θ : Type} {M : LoopModel W Rr} {H : Claims.Msg → Claims.Msg}
    {P : Claims.Prover Ω} {ω : Ω} {θ : Θ} {k : Claims.Msg → ℕ} {spec : SessionSpec} {sts : Array Stmt}
    {A : Θ → List ByteArray → Option ByteArray}
    (hτ : M.τ = (Claims.runStar H P (auditorV k spec sts A) M.x ω θ).msgs)
    (hK : (tmplOf (.ok (spec, sts))).countP isModel ≤ k M.x)
    {sid : Nat} {nonce : ByteArray} {Rd : Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop}
    (hRd : ∀ kk st pl e σ σ', Rd kk st pl e σ σ' → AnsPart M σ.c e kk →
      LoopRead M sid spec sts nonce kk st pl e σ σ')
    (hnone : ∀ kk st pl e σ σ', Rd kk st pl e σ σ' → Rd kk st pl { e with ans := none } σ σ')
    {σ₀ : LoopAt} (hc : σ₀.c = 0)
    (hF : ∀ es, (∀ e ∈ es, e.setup = .ok (spec, sts)) → LoopReads M sid spec sts nonce σ₀ {} {} es →
      EnvPre spec [] (schedSteps true {} {} es) M.msgs)
    {es : List Env} (hs : ∀ e ∈ es, e.setup = .ok (spec, sts))
    (hb : ∀ e ∈ es, ∀ why, (e.barrier (some why)).isSome)
    (haud : AudReads (A θ) [] {} {} es) (hR : ReadsBy Rd σ₀ {} {} es) :
    LoopReads M sid spec sts nonce σ₀ {} {} es :=
  loopReads_aud hτ hRd hnone hR [] [] (tmplOf (.ok (spec, sts))) rfl (fun h => by cases h)
    (fun m ms => by rw [hc]; rfl) (by rw [hc]; omega)
    (fun i hi => absurd hi (by rw [hc]; omega))
    (fun tail hts hLt => by rw [hc, List.drop_zero]; exact hF tail hts hLt)
    (fun tail hts => relay_prefix true _ tail hts) hs hb haud

/-- **Each request that reaches a slot passes no hold's barrier at a stop**, where none the relay read does. -/
theorem pick_barrier {ins : List (ByteArray × Env)} (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) :
    ∀ e ∈ pick true {} {} ins, ∀ why, (e.barrier (some why)).isSome := by
  obtain ⟨hsub, -⟩ := out_sched true {} {} ins (.inr ⟨rfl, reach_init⟩) hb
  intro e he
  obtain ⟨p, hp, rfl⟩ := List.mem_map.mp (hsub.subset he)
  exact hb p hp

/-- **`EnvPre` at the relay's requests from its start, with the relay's auditor as `V*`**: `envPre_run` at
`V* = auditorV`, where the session's schedule has no more model slots than `V*` has rounds, its `LoopReads` from the
relay reading `Rd` at each slot (`ReadsBy`) and answering as the auditor does (`AudReads`), `Rd` with what `LoopRead`
asks beyond it giving `LoopRead` and holding with the answer erased. -/
theorem envPre_auditorV
    {L₁ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Claims.Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {fin : Claims.Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      List (Claims.Msg × Claims.Msg) → Array ByteArray}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M (LeanRelease.withFiles (LeanRelease.withLink
      (LeanRelease.withInner ZKMal.Masked.dataE L₁ (Loop.realOf src) fin) src) src) (Loop.realOf src) s0 spec sts)
    (hsrc : M.src = src M.x M.w M.ρ) (hfin : M.fin = fin M.x M.w M.ρ) (hS : Loop.Level0Src src M.x M.w)
    (hlive : ZKMal.Masked.Live M.x) (hT : M.Takes) (hcp : M.L.CrossPasses) {Θ : Type}
    {H : Claims.Msg → Claims.Msg} {k : Claims.Msg → ℕ} {A : Θ → List ByteArray → Option ByteArray} {θ : Θ}
    (hτ : M.τ = (Claims.runStar H (M.L.toSrv.proverD M.L.fillOf M.nN M.wk M.w) (auditorV k spec sts A) M.x
      (M.ρ, M.κ) θ).msgs)
    (hk : 0 < k M.x) (hK : (tmplOf (.ok (spec, sts))).countP isModel ≤ k M.x)
    (hlk : ∀ h : List (Claims.Msg × Claims.Msg), h.length = 1 →
      M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false))
    {sid : Nat} {nonce : ByteArray} (hn : nonce = ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩)
    {ins : List (ByteArray × Env)} (hs : ∀ p ∈ ins, p.2.setup = .ok (spec, sts))
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) {σ : LoopAt} (hc : σ.c = 0)
    (hnew : ∀ p ∈ σ.R.reps, p.sid ≠ sid)
    {Rd : Session.Kind → State → Placed → Env → LoopAt → LoopAt → Prop}
    (hRd : ∀ kk st pl e σ σ', Rd kk st pl e σ σ' → AnsPart M σ.c e kk →
      LoopRead M sid spec sts nonce kk st pl e σ σ')
    (hnone : ∀ kk st pl e σ σ', Rd kk st pl e σ σ' → Rd kk st pl { e with ans := none } σ σ')
    (haud : AudReads (A θ) [] {} {} (pick true {} {} ins)) (hR : ReadsBy Rd σ {} {} (pick true {} {} ins)) :
    EnvPre spec [] (relaySteps true ins) M.msgs := by
  have hOwn := LoopModel.own_of_runStar hτ
  have hne : 0 < M.τ.length := (runStar_auditorV hτ).2 0 hk fun i hi => absurd hi (Nat.not_lt_zero _)
  have h0 : M.msgs.headD [] = _root_.Flock.Firewall.Hello.msg spec nonce := by
    rw [hn]
    exact hD.head_hello hOwn hne
  have hwk := hlk (M.hist 1) (by rw [LoopModel.hist, length_histOf_take]; omega)
  exact envPre_run hD hsrc hfin hS hlive hT hcp hτ hk hlk hn hs hb hc hnew
    (loopReads_auditorV hτ hK hRd hnone hc
      (fun es hes hR' => envPre_withInner hD hsrc hfin hS hlive hT hcp hOwn h0 hwk es [] {} {} M.msgs
        (loopP_start hc hnew hR' hes))
      (pick_setup hs hb) (pick_barrier hb) haud hR)

end Claims.ZKServer.Relay
