module

public import Proofs.Properties.Verity.SampledTwo.PrefixFixed
public import Proofs.Properties.Verity.SampledTwo.MessagesOpen

/-! The play's log before a round is fixed by the coins its history shows; that it holds what the round's
message is fixed by is not proved here. -/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open Claims

open Classical in
noncomputable def before (L : Locate) (i : ℕ) :
    List Verity.Release → List Verity.Release → List Verity.Release × Option (Verity.Frame × ByteArray)
  | _, [] => ([], none)
  | pre, rl :: log =>
    if ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j then ([], some (rl.frame, rl.answers))
    else (rl :: (before L i (pre ++ [rl]) log).1, (before L i (pre ++ [rl]) log).2)

section Before

variable (L : Locate) {K : ℕ}

theorem before_cons_pos {i : ℕ} {pre : List Verity.Release} {rl : Verity.Release} (log : List Verity.Release)
    (h : ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j) :
    before L i pre (rl :: log) = ([], some (rl.frame, rl.answers)) := by
  simp only [before, ite_eq_left h]

theorem before_cons_neg {i : ℕ} {pre : List Verity.Release} {rl : Verity.Release} (log : List Verity.Release)
    (h : ¬ ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j) :
    before L i pre (rl :: log) = (rl :: (before L i (pre ++ [rl]) log).1, (before L i (pre ++ [rl]) log).2) := by
  simp only [before, ite_eq_right h]

theorem before_append {i : ℕ} : ∀ (q pre rest : List Verity.Release),
    (∀ a rl b, q = a ++ rl :: b → ¬ ∃ j, locAt L (pre ++ a) rl.frame rl.answers = some j ∧ i ≤ j) →
      (before L i pre (q ++ rest)).1 = q ++ (before L i (pre ++ q) rest).1 ∧
        (before L i pre (q ++ rest)).2 = (before L i (pre ++ q) rest).2
  | [], pre, rest, _ => by simp
  | rl :: q, pre, rest, h => by
    have h0 : ¬ ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j := by
      have h' := h [] rl q rfl
      rw [List.append_nil] at h'
      exact h'
    obtain ⟨h1, h2⟩ := before_append (i := i) q (pre ++ [rl]) rest fun a rl₂ b hq => by
      rw [show pre ++ [rl] ++ a = pre ++ rl :: a by simp]
      exact h (rl :: a) rl₂ b (by simp [hq])
    rw [show pre ++ [rl] ++ q = pre ++ rl :: q by simp] at h1 h2
    have e : before L i pre (rl :: q ++ rest) =
        (rl :: (before L i (pre ++ [rl]) (q ++ rest)).1, (before L i (pre ++ [rl]) (q ++ rest)).2) :=
      before_cons_neg L (q ++ rest) h0
    rw [e]
    exact ⟨congrArg (fun l => rl :: l) h1, h2⟩

theorem before_first {i : ℕ} {q rest : List Verity.Release} {rl : Verity.Release} {j : ℕ}
    (hq : ∀ a rl₀ b, q = a ++ rl₀ :: b → ¬ ∃ j, locAt L a rl₀.frame rl₀.answers = some j ∧ i ≤ j)
    (hj : locAt L q rl.frame rl.answers = some j) (hij : i ≤ j) :
    before L i [] (q ++ rl :: rest) = (q, some (rl.frame, rl.answers)) := by
  obtain ⟨h1, h2⟩ := before_append L (i := i) q [] (rl :: rest) fun a rl₀ b hqe => by
    rw [List.nil_append]
    exact hq a rl₀ b hqe
  rw [List.nil_append, before_cons_pos L (i := i) (pre := q) (rl := rl) rest ⟨j, hj, hij⟩] at h1 h2
  exact Prod.ext (h1.trans (List.append_nil q)) h2

theorem before_all {i : ℕ} {q : List Verity.Release}
    (hq : ∀ a rl₀ b, q = a ++ rl₀ :: b → ¬ ∃ j, locAt L a rl₀.frame rl₀.answers = some j ∧ i ≤ j) :
    before L i [] q = (q, none) := by
  obtain ⟨h1, h2⟩ := before_append L (i := i) q [] [] fun a rl₀ b hqe => by
    rw [List.nil_append]
    exact hq a rl₀ b hqe
  rw [List.append_nil] at h1 h2
  exact Prod.ext (h1.trans (List.append_nil q)) h2

theorem before_sameUntil {E : Prop} {r r' : Fin K → FlockDeploy.Coin} (i : ℕ) :
    ∀ (log pre log' : List Verity.Release), SameUntil L E r r' pre log log' →
      (∀ a rl b, log = a ++ rl :: b → ∀ j : Fin K, locAt L (pre ++ a) rl.frame rl.answers = some j.val →
        j.val < i → r j = r' j) →
      before L i pre log' = before L i pre log
  | [], pre, log', h, _ => by
    obtain ⟨rfl, -⟩ := h
    rfl
  | rl :: log, pre, log', h, hr => by
    obtain ⟨rl', rest, rfl, hf, ha, hstep⟩ := h
    by_cases hp : ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j
    · have hp' : ∃ j, locAt L pre rl'.frame rl'.answers = some j ∧ i ≤ j := by
        rw [hf, ha]
        exact hp
      rw [before_cons_pos L rest hp', before_cons_pos L log hp, hf, ha]
    · have hc : ChunkAgree L r r' pre rl := by
        intro j u hs a _ _
        obtain ⟨hl, -, -⟩ := slot_spec L hs
        have hji : j.val < i := Nat.lt_of_not_le fun hge => hp ⟨j.val, hl, hge⟩
        rw [hr [] rl log rfl j (by rw [List.append_nil]; exact hl) hji]
      obtain ⟨he, hs⟩ := hstep hc
      have ih := before_sameUntil i log (pre ++ [rl]) rest hs fun a rl₂ b hlog j hj hji =>
        hr (rl :: a) rl₂ b (by simp [hlog]) j
          (by rw [show pre ++ rl :: a = pre ++ [rl] ++ a by simp]; exact hj) hji
      rw [he, before_cons_neg L rest hp, before_cons_neg L log hp, ih]

end Before

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem before_place (L : Locate) {K : ℕ} (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (r r' : Fin K → FlockDeploy.Coin) (ω : Tape (V.game θ F c) σ) (i : ℕ)
    (h : ∀ a rl b, (Tape.out _ σ (place V θ F c σ L r ω)).log = a ++ rl :: b →
      ∀ j : Fin K, locAt L a rl.frame rl.answers = some j.val → j.val < i → r j = r' j) :
    before L i [] (Tape.out _ σ (place V θ F c σ L r' ω)).log =
      before L i [] (Tape.out _ σ (place V θ F c σ L r ω)).log := by
  have hs := same_place L V θ F c σ r r' ω
  unfold SameAt at hs
  rw [List.length_nil, List.drop_zero, List.drop_zero] at hs
  exact before_sameUntil L i _ [] _ hs fun a rl b hlog j hj hji =>
    h a rl b hlog j (by rw [List.nil_append] at hj; exact hj) hji

end Verity.SampledTwo.CoinMap

namespace Verity.SampledTwo.Messages

open Claims

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem before_shown (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) (ω : TapeOf V θ F c σ)
    (r r' : Coins A x)
    (hacc : Acc (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)))
    (hgood : Good (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω))) (i : ℕ)
    (hs : Shows (playHist A x r
      (playMsgs Rd (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω))) i) r') :
    CoinMap.before L i [] (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r' ω)).log =
      CoinMap.before L i [] (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)).log :=
  CoinMap.before_place L V θ F c σ r r' ω i fun a rl b hlog j hj hji =>
    (hs j _ (shownAt_playHist A x r _ j hji (hShow _ hacc hgood a rl b hlog j.val hj r j.isLt))).symm

theorem fixed_of_before (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) {Φ : Type}
    (Fx : Verity.Exchange → ℕ → Option Φ) (i : ℕ)
    (g : List Verity.Release × Option (Verity.Frame × ByteArray) → Option Φ) (ω : TapeOf V θ F c σ)
    (hFx : ∀ r₀, Fx (Fr.out ω r₀) i = g (CoinMap.before L i [] (Fr.out ω r₀).log)) (r r' : Coins A x)
    (hacc : Acc (Fr.out ω r)) (hgood : Good (Fr.out ω r))
    (hs : Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r') :
    Fx (Fr.out ω r') i = Fx (Fr.out ω r) i := by
  have e : ∀ r₀, Fr.out ω r₀ = Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r₀ ω) := fun r₀ =>
    congrArg (Verity.Game.Tape.out (V.game θ F c) σ) (hFr ω r₀)
  rw [hFx r', hFx r]
  rw [e r] at hacc hgood hs
  rw [e r', e r, before_shown V A x θ F c σ L Rd Acc Good hShow ω r r' hacc hgood i hs]

theorem fixedBefore_of_before (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) {Φ : Type}
    (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop) (i : ℕ)
    (hOpn : ∀ φ o, Opn φ o i → Acc o ∧ Good o)
    (g : TapeOf V θ F c σ → List Verity.Release × Option (Verity.Frame × ByteArray) → Option Φ)
    (hFx : ∀ ω r₀, Fx (Fr.out ω r₀) i = g ω (CoinMap.before L i [] (Fr.out ω r₀).log)) :
    FixedBefore Fr Rd Fx Opn i :=
  fixedBefore_of_eq Fr Rd Acc Good Fx Opn i hOpn fun ω r r' hacc hgood hs =>
    fixed_of_before V A x θ F c σ L Fr hFr Rd Acc Good hShow Fx i (g ω) ω (hFx ω) r r' hacc hgood hs

theorem fixedBefore_strings_before (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) {κ : Type}
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) (i : ℕ)
    (g : TapeOf V θ F c σ → List Verity.Release × Option (Verity.Frame × ByteArray) → Option (List Msg))
    (hDg : ∀ ω r₀, Dg (Fr.out ω r₀) i = g ω (CoinMap.before L i [] (Fr.out ω r₀).log))
    (hctx : ∀ (ω : TapeOf V θ F c σ) (r r' : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      Acc (Fr.out ω r') → Good (Fr.out ω r') → Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r' →
      ctx (Fr.out ω r') i = ctx (Fr.out ω r) i) :
    FixedBefore Fr Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx) i :=
  fixedBefore_strings Fr Rd Acc Good Dg raw ctx i fun ω r r' ha hg ha' hg' hs =>
    ⟨fixed_of_before V A x θ F c σ L Fr hFr Rd Acc Good hShow Dg i (g ω) ω (hDg ω) r r' ha hg hs,
      hctx ω r r' ha hg ha' hg' hs⟩

end Verity.SampledTwo.Messages

namespace Verity.SampledTwo.SessionCoins

theorem before_session {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hRise : LogRising A C Good tab) {o : Exchange}
    (hacc : FormAccepts C o) (hgood : Good o) {q rest : List Release} {rl : Release} {i : ℕ}
    (hlog : o.log = q ++ rl :: rest) (hi : CoinMap.locAt L q rl.frame rl.answers = some i)
    (hg : groupOf A i = .sessions) :
    CoinMap.before L i [] o.log = (q, some (rl.frame, rl.answers)) := by
  have hsi := stage_lt_of_sessions hg
  have h2 := (hL q rl.frame rl.answers i hg).1 hi
  have hq : ∀ a r b, q = a ++ r :: b → ¬ ∃ j, CoinMap.locAt L a r.frame r.answers = some j ∧ i ≤ j := by
    intro a r b hqe hex
    obtain ⟨j, hj, hij⟩ := hex
    have hgj : groupOf A j = .sessions := sessions_of_lt (by omega)
    have h1 := (hL a r.frame r.answers j hgj).1 hj
    have hsplit : o.log = a ++ r :: (b ++ rl :: rest) := by
      rw [hlog, hqe]
      simp
    have h2' : locOf A tab (a ++ r :: b) rl.frame = some i := by
      rw [← hqe]
      exact h2
    have hlt : j < i := hRise o hacc hgood a r b rl rest hsplit j i h1 h2'
    omega
  rw [hlog]
  exact CoinMap.before_first L hq hi (Nat.le_refl i)

end Verity.SampledTwo.SessionCoins
