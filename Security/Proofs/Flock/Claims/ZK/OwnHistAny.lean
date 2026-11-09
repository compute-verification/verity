module

public import Proofs.Flock.Claims.ZK.Refine

/-!
# The deferred release's own histories

The deferred server meets only its own histories (`ZkServer.OwnHist`: each message is the deferred frame on the
history before it), and `LeanRelease.refD` hands each item of `ItemsRefine` that history's `hown`. The live equations
hold for any worker only there: `live` reads the worker's values at each prefix, never the history's messages, so off
`OwnHist` the released rounds a finish reads are arbitrary bytes, so they are asked there only
(`LeanRelease.LiveRoundEqOwn`, in `Refine` beside `LiveRoundEq`). This file holds what their proofs share:

* **The own histories**: `ZkServer.ownHist_nil`, `OwnHist.take` (closed under prefixes), `ownHist_concat_iff` (one
  more round is own exactly when its message is the deferred frame), `ownHist_cons_iff` (the live equations' histories
  `(m₀, ρ₀) :: ops`), and `ZkServer.ownOf` (the own history on given answers: `ownHist_ownOf`, `map_snd_ownOf`, and
  `OwnHist.eq_ownOf`, every own history is the one on its answers).
* **What an own history's messages are** at the Lean frame's release: at a frame round where the worker's run is live,
  the frame's release of the worker's round (`LeanRelease.ownHist_msg_live`); where it is not, the release of the
  stand-in, the ZK server's scheduled round as a stopped rep's (`LeanRelease.ownHist_msg_fill`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace ZkServer

variable {W R : Type} (S : ZkServer W R) (fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val)
  (wk : S.Worker) (x : Msg) (w : W) (ρ : R) (κ : S.Chk)

/-- The empty history is own. -/
theorem ownHist_nil : S.OwnHist fill wk x w ρ κ [] := fun _ hj => absurd hj (Nat.not_lt_zero _)

/-- **Every prefix of an own history is own.** -/
theorem OwnHist.take {S : ZkServer W R} {fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val} {wk : S.Worker}
    {x : Msg} {w : W} {ρ : R} {κ : S.Chk} {h : List (Msg × Msg)} (hown : S.OwnHist fill wk x w ρ κ h) (n : ℕ) :
    S.OwnHist fill wk x w ρ κ (h.take n) := by
  intro j hj
  rw [List.length_take] at hj
  have hjn : j ≤ n := by omega
  have hjh : j < h.length := by omega
  rw [List.getElem_take]
  refine (hown j hjh).trans ?_
  simp only [List.take_take, Nat.min_eq_left hjn]

/-- **One more round is own exactly when its message is the deferred frame** on the history before it. -/
theorem ownHist_concat_iff (h : List (Msg × Msg)) (p : Msg × Msg) :
    S.OwnHist fill wk x w ρ κ (h ++ [p]) ↔ S.OwnHist fill wk x w ρ κ h ∧ p.1 = S.frameD fill wk x w ρ κ h := by
  constructor
  · intro hown
    refine ⟨fun j hj => ?_, ?_⟩
    · have e := hown j (by simp only [List.length_append, List.length_singleton]; omega)
      rwa [List.getElem_append_left hj, List.take_append_of_le_length (Nat.le_of_lt hj)] at e
    · have e := hown h.length (by simp only [List.length_append, List.length_singleton]; omega)
      simpa using e
  · rintro ⟨hown, hp⟩ j hj
    simp only [List.length_append, List.length_singleton] at hj
    rcases Nat.lt_or_ge j h.length with hlt | hge
    · rw [List.getElem_append_left hlt, List.take_append_of_le_length (Nat.le_of_lt hlt)]
      exact hown j hlt
    · have hj' : j = h.length := by omega
      subst hj'
      simpa using hp

/-- **The live equations' histories**: `(m₀, ρ₀) :: ops` is own exactly when `m₀` is the deferred frame on the empty
history and each opening's message the deferred frame on the history before it. -/
theorem ownHist_cons_iff (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) :
    S.OwnHist fill wk x w ρ κ ((m₀, ρ₀) :: ops) ↔
      m₀ = S.frameD fill wk x w ρ κ [] ∧
        ∀ j (hj : j < ops.length), (ops[j]'hj).1 = S.frameD fill wk x w ρ κ ((m₀, ρ₀) :: ops.take j) := by
  constructor
  · intro hown
    exact ⟨hown 0 (by simp), fun j hj => hown (j + 1) (by simp only [List.length_cons]; omega)⟩
  · rintro ⟨h0, h1⟩ j hj
    cases j with
    | zero => simpa using h0
    | succ j => simpa using h1 j (by simp only [List.length_cons] at hj; omega)

/-- **The own history on the answers `as`**: each message the deferred frame on the history before it. -/
def ownOf (as : List Msg) : List (Msg × Msg) :=
  as.foldl (fun h a => h ++ [(S.frameD fill wk x w ρ κ h, a)]) []

theorem ownHist_foldl (as : List Msg) (acc : List (Msg × Msg)) (hacc : S.OwnHist fill wk x w ρ κ acc) :
    S.OwnHist fill wk x w ρ κ (as.foldl (fun h a => h ++ [(S.frameD fill wk x w ρ κ h, a)]) acc) := by
  induction as generalizing acc with
  | nil => exact hacc
  | cons a as ih =>
    exact ih _ ((S.ownHist_concat_iff fill wk x w ρ κ acc (S.frameD fill wk x w ρ κ acc, a)).mpr ⟨hacc, rfl⟩)

theorem map_snd_foldl (as : List Msg) (acc : List (Msg × Msg)) :
    (as.foldl (fun h a => h ++ [(S.frameD fill wk x w ρ κ h, a)]) acc).map Prod.snd = acc.map Prod.snd ++ as := by
  induction as generalizing acc with
  | nil => simp
  | cons a as ih => simp [ih]

/-- **An own history after `acc` is the own history on its answers after `acc`.** -/
theorem foldl_of_ownHist (rest acc : List (Msg × Msg)) (hown : S.OwnHist fill wk x w ρ κ (acc ++ rest)) :
    (rest.map Prod.snd).foldl (fun h a => h ++ [(S.frameD fill wk x w ρ κ h, a)]) acc = acc ++ rest := by
  induction rest generalizing acc with
  | nil => simp
  | cons r rest ih =>
    have hr : r.1 = S.frameD fill wk x w ρ κ acc := by
      have e := hown acc.length (by simp)
      simpa using e
    have e : acc ++ r :: rest = (acc ++ [r]) ++ rest := by simp
    rw [e] at hown ⊢
    simp only [List.map_cons, List.foldl_cons]
    have hp : (S.frameD fill wk x w ρ κ acc, r.2) = r := by rw [← hr]
    rw [hp]
    exact ih (acc ++ [r]) hown

theorem ownHist_ownOf (as : List Msg) : S.OwnHist fill wk x w ρ κ (S.ownOf fill wk x w ρ κ as) :=
  S.ownHist_foldl fill wk x w ρ κ as [] (S.ownHist_nil fill wk x w ρ κ)

theorem map_snd_ownOf (as : List Msg) : (S.ownOf fill wk x w ρ κ as).map Prod.snd = as := by
  simp [ownOf, map_snd_foldl]

/-- **An own history is the own history on its answers.** -/
theorem OwnHist.eq_ownOf {S : ZkServer W R} {fill : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → S.Val}
    {wk : S.Worker} {x : Msg} {w : W} {ρ : R} {κ : S.Chk} {h : List (Msg × Msg)}
    (hown : S.OwnHist fill wk x w ρ κ h) : h = S.ownOf fill wk x w ρ κ (h.map Prod.snd) := by
  have e := S.foldl_of_ownHist fill wk x w ρ κ h [] (by simpa using hown)
  simpa [ownOf] using e.symm

end ZkServer

namespace LeanRelease

section Msgs

variable {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R) (κ : L.toSrv.Chk)

/-- **An own history's message at a frame round where the worker's run is live** is the frame's release of the
worker's round there. -/
theorem ownHist_msg_live {h : List (Msg × Msg)} (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) {j : ℕ}
    (hj : j < h.length) (hk : L.kindAt x j = .masked) (hl : L.toSrv.live wk x w ρ κ (h.take j) = true) {v : Msg}
    (hv : wk x w ρ (h.take j) 0 = some (v, false)) : (h[j]'hj).1 = L.release x ρ (h.take j) false v := by
  have hkj : L.kindAt x (h.take j).length = .masked := by
    rw [List.length_take, Nat.min_eq_left hj.le]
    exact hk
  have hu : L.toSrv.wkD L.fillOf wk κ x w ρ (h.take j) 0 = some (v, false) := by
    simp only [ZkServer.wkD, hl, ↓reduceIte, hv]
    rfl
  exact (hown j hj).trans (L.frameD_toSrv L.fillOf wk x w ρ κ _ hkj hu)

/-- **An own history's message at a frame round where the worker's run is not live** is the frame's release of the
stand-in there: the ZK server's scheduled round, at the values its release log reports, as a stopped rep's. -/
theorem ownHist_msg_fill {h : List (Msg × Msg)} (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) {j : ℕ}
    (hj : j < h.length) (hk : L.kindAt x j = .masked) (hl : L.toSrv.live wk x w ρ κ (h.take j) = false) :
    (h[j]'hj).1 =
      L.release x ρ (h.take j) true (L.scheduled x w ρ (h.take j) (computedOf (L.computed x w ρ (h.take j)))) := by
  have hkj : L.kindAt x (h.take j).length = .masked := by
    rw [List.length_take, Nat.min_eq_left hj.le]
    exact hk
  have hu : L.toSrv.wkD L.fillOf wk κ x w ρ (h.take j) 0 =
      some (L.fillOf x (h.take j).length 0 w ρ (h.take j)) := by
    simp only [ZkServer.wkD, hl, Bool.false_eq_true, ↓reduceIte]
  exact (hown j hj).trans (L.frameD_toSrv L.fillOf wk x w ρ κ _ hkj hu)

end Msgs

end LeanRelease

end Claims.ZKServer
