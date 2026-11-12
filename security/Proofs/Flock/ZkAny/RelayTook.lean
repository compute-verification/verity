module

public import Proofs.Flock.ZkAny.RelayLoop

/-!
# What the loop took of the coins so far, against the model's release

The loop goes on past one of the session's slots only where the transport's coin check takes the auditor's answer and,
at a frame round, the rep's frame takes the coins (`Session.State.coins`, `Relay.afterAnswer`), and it runs no slot
after a refusal (`Session.State.request`). The model's release takes the coins after a history in three parts
(`LeanRelease.answered`): the transport's verdict, the frame's coin step on the last coins (`takeAt`'s
`Masker.answer`), and past it the rank stop or the next rep's first-round state, which the loop checks at the next
round's place, so only where that round goes out.

* `LeanRelease.verdictAt`: the transport's verdict on a history, the session's coin tree after it, or why there is
  none. `opens_of_verdictAt`: where it is a session, the check passes the history. `verdictAt_start`,
  `verdictAt_snoc`: `Hello`'s answer, then each opening, as the session loop takes them (`coinVerdict_eq_ok`,
  `reqsFrom_snoc`, `openAll_snoc`).
* `LeanRelease.fed`: the frame's coin step on the history's last coins. `fed_of_le`: there is none past `Hello` and the
  `Commit`.
* `LoopModel.TookAt`: at the loop's session and its first `c` of the model's messages, the verdict on the model's
  history so far the session's coin tree, the frame's taking of the coins after each earlier round, and the coin step
  on the last. `TookAt.zero`: at the first slot. `TookAt.congr`: at a slot that reads no model message.
  `TookAt.hello`: past the `Hello`. `TookAt.succ`: one opening on, where the frame took the last coins and steps on the
  next.
-/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock.Firewall.Coins (Release)

/-- The openings `ms ++ [a]` as `flock-firewall coins` takes them: `ms`'s, then `a` at its request. -/
theorem reqsFrom_snoc (req : ℕ → String × ℕ) :
    ∀ (j : ℕ) (ms : List Msg) (a : Msg),
      reqsFrom req j (ms ++ [a]) = reqsFrom req j ms ++ [((req (j + ms.length)).1, (req (j + ms.length)).2, a)]
  | _, [], _ => rfl
  | j, m :: ms, a => by
    show ((req j).1, (req j).2, m) :: reqsFrom req (j + 1) (ms ++ [a]) = _
    rw [reqsFrom_snoc req (j + 1) ms a, List.length_cons, show j + 1 + ms.length = j + (ms.length + 1) by omega]
    try rfl

/-- **One more opening**: where the session takes the openings `os` to `S'` and `S'` takes the next, so do `os` and
it. -/
theorem openAll_snoc : ∀ (S : Release) (os : List (String × ℕ × Msg)) {S' S'' : Release} {stream : String}
    {n : ℕ} {m c : Msg}, S.openAll os = .ok S' → S'.open stream n m = .ok (S'', c) →
      S.openAll (os ++ [(stream, n, m)]) = .ok S''
  | S, [], _, _, stream, n, m, _, h, ho => by
    simp only [Release.openAll, Except.ok.injEq] at h
    subst h
    simp only [List.nil_append, Release.openAll, ho]
  | S, (st, k, b) :: os, _, _, _, _, _, _, h, ho => by
    rw [List.cons_append]
    rcases hs : S.open st k b with e | ⟨S₁, c₁⟩
    · simp [Release.openAll, hs] at h
    · simp only [Release.openAll, hs] at h ⊢
      exact openAll_snoc S₁ os h ho

/-- **Where the coin check's verdict is a session**, it read the `Hello`, `Hello`'s answer began a session, and that
session took the openings. -/
theorem coinVerdict_eq_ok {hl : Except String _root_.Flock.Firewall.Coins.Hello} {a : Msg}
    {os : List (String × ℕ × Msg)} {C : Release} (H : coinVerdict hl a os = .ok C) :
    ∃ h S, hl = .ok h ∧ Release.start h a = .ok S ∧ S.openAll os = .ok C := by
  rcases hl with e | h
  · change Except.error e = Except.ok C at H
    cases H
  · change (Release.start h a).bind (fun S => S.openAll os) = .ok C at H
    rcases hs : Release.start h a with e | S
    · rw [hs] at H
      change Except.error e = Except.ok C at H
      cases H
    · rw [hs] at H
      exact ⟨h, S, rfl, by first | rfl | exact hs, H⟩

namespace LeanRelease

/-- **The transport's verdict on the history `h`** at the tape `ρ`: the session's coin tree after its reading of the
`Hello` at the tape, `Hello`'s answer and each later answer as the opening of the transport's request (`coinVerdict`),
as the session loop holds it (`Session.State.session`), or why it holds none. -/
def verdictAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) : List (Msg × Msg) → Except String Release
  | (_, a) :: ops => coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 (ops.map Prod.snd))
  | [] => .error "no answer to the Hello"

/-- **Where the transport's verdict on `h` is a session, its coin check passes `h`.** -/
theorem opens_of_verdictAt {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {C : Release} (hv : L.verdictAt x ρ h = .ok C) : L.opens x ρ h = true := by
  rcases h with _ | ⟨⟨m, a⟩, _ | ⟨o, ops⟩⟩
  · rfl
  · rfl
  · have hv' : coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd)) = .ok C := hv
    show (coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk = true
    rw [hv']
    rfl

/-- **`Hello`'s answer**: where the transport reads the `Hello` as `hl` and `Hello`'s answer `a` begins the session
`C`, the verdict on the `Hello` alone is `C`. -/
theorem verdictAt_start {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R}
    {hl : _root_.Flock.Firewall.Coins.Hello} {m a : Msg} {C : Release} (hh : L.helloChk x ρ = .ok hl)
    (hs : Release.start hl a = .ok C) : L.verdictAt x ρ [(m, a)] = .ok C := by
  show coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 []) = .ok C
  rw [hh]
  show (Release.start hl a).bind (fun S => S.openAll (reqsFrom (L.req x) 0 [])) = _
  rw [hs]
  try rfl

/-- **One more opening**: where the verdict on `h`, past `Hello`'s answer, is the session `C`, and `C` takes `a` as
the opening of the transport's next request, the verdict on `h` one round on is the session after it. -/
theorem verdictAt_snoc {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {C C' : Release} {m a c : Msg} (hv : L.verdictAt x ρ h = .ok C) (hn : h ≠ [])
    (ho : C.open (L.req x (h.length - 1)).1 (L.req x (h.length - 1)).2 a = .ok (C', c)) :
    L.verdictAt x ρ (h ++ [(m, a)]) = .ok C' := by
  rcases h with _ | ⟨⟨m₀, a₀⟩, ops⟩
  · exact absurd rfl hn
  obtain ⟨hl, S, hh, hs, hO⟩ := coinVerdict_eq_ok
    (show coinVerdict (L.helloChk x ρ) a₀ (reqsFrom (L.req x) 0 (ops.map Prod.snd)) = .ok C from hv)
  have hk : ((m₀, a₀) :: ops).length - 1 = ops.length := by simp
  rw [hk] at ho
  show coinVerdict (L.helloChk x ρ) a₀ (reqsFrom (L.req x) 0 ((ops ++ [(m, a)]).map Prod.snd)) = .ok C'
  rw [List.map_append, List.map_singleton, reqsFrom_snoc, Nat.zero_add, List.length_map, hh]
  show (Release.start hl a₀).bind _ = _
  rw [hs]
  exact openAll_snoc S _ hO ho

/-- **The frame's coin step on the stream's last coins** after `h`: past `Hello` and the `Commit`, whose coins no frame
takes, the frame at the state the rep that released the last round of `h` left (`prog`), at that rep's configuration,
takes them (`Masker.answer`, `takeAt`'s first step). -/
def fed {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  decide (h.length ≤ 2) || ((L.prog x h).answer (L.cfgAt x ρ (h.length - 1)) (L.coinsOf x h)).isOk

theorem fed_of_le {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) {h : List (Msg × Msg)}
    (hl : h.length ≤ 2) : L.fed x ρ h = true := by
  simp [fed, hl]

end LeanRelease

namespace Relay

open _root_.Flock.Firewall.Session (State)

open Claims.ZKMal in
/-- The run's history one message on: its `j`-th message and `V*`'s answer to it, read as the prover reads it. -/
theorem LoopModel.hist_snoc {W Rr : Type} (M : LoopModel W Rr) {j : Nat} (hj : j < M.τ.length) :
    M.hist (j + 1) = M.hist j ++ [((M.τ[j]'hj).1, msgOf (M.τ[j]'hj).2)] := by
  unfold LoopModel.hist histOf
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, List.map_append, List.map_cons,
    List.map_nil]

/-- **What the loop took of the coins so far, against the model's release**, at the session `st` once its slots took
the first `c` of the model's messages: wherever the session holds a coin tree, it is the transport's verdict on the
model's history so far (`verdictAt`); the frame took the coins after each earlier round (`tookEarlier`); and it takes
the last coins' step (`fed`). The rest of the model's taking of the last coins, the rank stop and a rep's first-round
state, the loop checks at the next round's place. -/
def LoopModel.TookAt {W Rr : Type} (M : LoopModel W Rr) (st : State) (c : Nat) : Prop :=
  (∀ C, st.session = some C → M.L.verdictAt M.x M.ρ (M.hist c) = .ok C) ∧
    M.L.tookEarlier M.x M.ρ (M.hist c) = true ∧ M.L.fed M.x M.ρ (M.hist c) = true

/-- **At the first slot**, where the session holds no coin tree yet. -/
theorem LoopModel.TookAt.zero {W Rr : Type} (M : LoopModel W Rr) {st : State} (hs : st.session = none) :
    M.TookAt st 0 :=
  ⟨fun C h => (by rw [hs] at h; cases h), by first | rfl | simp [LoopModel.hist, LeanRelease.tookEarlier],
    by first | rfl | simp [LoopModel.hist, LeanRelease.fed]⟩

/-- **At a slot that reads no model message**, which keeps the session's coin tree. -/
theorem LoopModel.TookAt.congr {W Rr : Type} {M : LoopModel W Rr} {st st' : State} {c : Nat}
    (hs : st'.session = st.session) (h : M.TookAt st c) : M.TookAt st' c :=
  ⟨fun C hC => h.1 C (hs.symm.trans hC), h.2⟩

open Claims.ZKMal in
/-- **Past the `Hello`**: where the transport reads the `Hello` as `hl` and the session's coin tree after it is the one
the model's first answer begins there (`Release.start`), the loop took the coins so far. -/
theorem LoopModel.TookAt.hello {W Rr : Type} {M : LoopModel W Rr} {st : State} (hlt : 0 < M.τ.length)
    {hl : _root_.Flock.Firewall.Coins.Hello} (hh : M.L.helloChk M.x M.ρ = .ok hl)
    (hs : ∀ C, st.session = some C → Release.start hl (msgOf (M.τ[0]'hlt).2) = .ok C) : M.TookAt st 1 := by
  have h1 : M.hist 1 = [((M.τ[0]'hlt).1, msgOf (M.τ[0]'hlt).2)] := M.hist_snoc hlt
  refine ⟨fun C hC => ?_, ?_, ?_⟩
  · rw [h1]
    exact LeanRelease.verdictAt_start hh (hs C hC)
  · rw [h1]
    exact M.L.tookEarlier_single M.x M.ρ _
  · rw [h1]
    exact M.L.fed_of_le M.x M.ρ (by simp)

open Claims.ZKMal in
/-- **One opening on**: past the `Hello`, where the session's coin tree `C` takes the model's next answer as the opening
of the transport's request there and the session after holds only the tree after it, the frame took the last coins
(`frameAnswer`), and it steps on the next, the loop took the coins one message on. -/
theorem LoopModel.TookAt.succ {W Rr : Type} {M : LoopModel W Rr} {st st' : State} {j : Nat} (hlt : j < M.τ.length)
    (h : M.TookAt st j) (hj : j ≠ 0) {C C' : Release} {c : Claims.Msg} (hC : st.session = some C)
    (ho : C.open (M.L.req M.x (j - 1)).1 (M.L.req M.x (j - 1)).2 (msgOf (M.τ[j]'hlt).2) = .ok (C', c))
    (hs' : ∀ D, st'.session = some D → D = C') (hf : (M.L.frameAnswer M.x M.ρ (M.hist j)).isOk = true)
    (hfed : M.L.fed M.x M.ρ (M.hist (j + 1)) = true) : M.TookAt st' (j + 1) := by
  have hlj : (M.hist j).length = j := by
    rw [LoopModel.hist, length_histOf_take, Nat.min_eq_left hlt.le]
  have hsn := M.hist_snoc hlt
  have hne : M.hist j ≠ [] := fun h0 => hj (by simpa [h0] using hlj.symm)
  refine ⟨fun D hD => ?_, ?_, hfed⟩
  · rw [hs' D hD, hsn]
    exact LeanRelease.verdictAt_snoc (h.1 C hC) hne (by rw [hlj]; exact ho)
  · simp only [hsn, LeanRelease.tookEarlier_append, h.2.1, hf, Bool.and_self]

end Relay

end Claims.ZKServer
