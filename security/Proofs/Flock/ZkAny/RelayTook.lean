module

public import Proofs.Flock.ZkAny.RelayLoop

/-! What the loop took of the coins so far, against the model's release (`LoopModel.TookAt`). -/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock.ZkServer.Coins (Release)

theorem reqsFrom_snoc (req : ℕ → String × ℕ) :
    ∀ (j : ℕ) (ms : List Msg) (a : Msg),
      reqsFrom req j (ms ++ [a]) = reqsFrom req j ms ++ [((req (j + ms.length)).1, (req (j + ms.length)).2, a)]
  | _, [], _ => rfl
  | j, m :: ms, a => by
    show ((req j).1, (req j).2, m) :: reqsFrom req (j + 1) (ms ++ [a]) = _
    rw [reqsFrom_snoc req (j + 1) ms a, List.length_cons, show j + 1 + ms.length = j + (ms.length + 1) by omega]
    try rfl

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

theorem coinVerdict_eq_ok {hl : Except String _root_.Flock.ZkServer.Coins.Hello} {a : Msg}
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

def verdictAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) : List (Msg × Msg) → Except String Release
  | (_, a) :: ops => coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 (ops.map Prod.snd))
  | [] => .error "no answer to the Hello"

theorem opens_of_verdictAt {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    {C : Release} (hv : L.verdictAt x ρ h = .ok C) : L.opens x ρ h = true := by
  rcases h with _ | ⟨⟨m, a⟩, _ | ⟨o, ops⟩⟩
  · rfl
  · rfl
  · have hv' : coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd)) = .ok C := hv
    show (coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk = true
    rw [hv']
    rfl

theorem verdictAt_start {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R}
    {hl : _root_.Flock.ZkServer.Coins.Hello} {m a : Msg} {C : Release} (hh : L.helloChk x ρ = .ok hl)
    (hs : Release.start hl a = .ok C) : L.verdictAt x ρ [(m, a)] = .ok C := by
  show coinVerdict (L.helloChk x ρ) a (reqsFrom (L.req x) 0 []) = .ok C
  rw [hh]
  show (Release.start hl a).bind (fun S => S.openAll (reqsFrom (L.req x) 0 [])) = _
  rw [hs]
  try rfl

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

def fed {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) : Bool :=
  decide (h.length ≤ 2) || ((L.prog x h).answer (L.cfgAt x ρ (h.length - 1)) (L.coinsOf x h)).isOk

theorem fed_of_le {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) {h : List (Msg × Msg)}
    (hl : h.length ≤ 2) : L.fed x ρ h = true := by
  simp [fed, hl]

end LeanRelease

namespace Relay

open _root_.Flock.ZkServer.Session (State)

open Claims.ZKMal in
theorem LoopModel.hist_snoc {W Rr : Type} (M : LoopModel W Rr) {j : Nat} (hj : j < M.τ.length) :
    M.hist (j + 1) = M.hist j ++ [((M.τ[j]'hj).1, msgOf (M.τ[j]'hj).2)] := by
  unfold LoopModel.hist histOf
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, List.map_append, List.map_cons,
    List.map_nil]

/-- The rank stop and a rep's first-round state, the rest of the model's taking of the last coins, the loop checks at the
next round's place. -/
def LoopModel.TookAt {W Rr : Type} (M : LoopModel W Rr) (st : State) (c : Nat) : Prop :=
  (∀ C, st.session = some C → M.L.verdictAt M.x M.ρ (M.hist c) = .ok C) ∧
    M.L.tookEarlier M.x M.ρ (M.hist c) = true ∧ M.L.fed M.x M.ρ (M.hist c) = true

theorem LoopModel.TookAt.zero {W Rr : Type} (M : LoopModel W Rr) {st : State} (hs : st.session = none) :
    M.TookAt st 0 :=
  ⟨fun C h => (by rw [hs] at h; cases h), by first | rfl | simp [LoopModel.hist, LeanRelease.tookEarlier],
    by first | rfl | simp [LoopModel.hist, LeanRelease.fed]⟩

theorem LoopModel.TookAt.congr {W Rr : Type} {M : LoopModel W Rr} {st st' : State} {c : Nat}
    (hs : st'.session = st.session) (h : M.TookAt st c) : M.TookAt st' c :=
  ⟨fun C hC => h.1 C (hs.symm.trans hC), h.2⟩

open Claims.ZKMal in
theorem LoopModel.TookAt.hello {W Rr : Type} {M : LoopModel W Rr} {st : State} (hlt : 0 < M.τ.length)
    {hl : _root_.Flock.ZkServer.Coins.Hello} (hh : M.L.helloChk M.x M.ρ = .ok hl)
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
