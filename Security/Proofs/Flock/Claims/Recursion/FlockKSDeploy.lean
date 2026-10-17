import Proofs.Flock.Claims.Recursion.FlockKSAcc
import Proofs.Flock.Claims.DeployK

/-!
# The deployed verifier's code facts for the abort-law table bound

Flock's deployed verifier (`FlockDeploy.V`, plain or masked) meets three of the code facts
`tableBoundθAbort_of_onAcc` reads: it never accepts the empty message (`refusable_V`), its decision reads the link
message's coin and the round messages' coins but not the last message's (`decideFree_V`), and at SHA-512 it accepts
only at its last round (`accLast_V_sha512`, from `next_acc_length_sha512`). So at SHA-512 the recursion's KS over an
inner deployment whose verifier is `V zk` rests on `TableBoundθOnAcc` in place of `Inner.KSAbort`, beside
`KSFlock_of_flockKS`'s other hypotheses (`KSFlock_of_flockKS_V`).
-/

namespace FlockDeploy

open Claims.Recursion

/-- **The empty message is never accepted**: it is neither a coin request nor a last message. -/
theorem act_snoc_nil (zk : Bool) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    act zk x (as ++ [[]]) cs = .rej := by
  unfold act
  rcases setupOf zk x with _ | ⟨s, spec, sts⟩
  · rfl
  · rw [List.getLast?_concat]
    rfl

theorem refusable_V (H : Claims.Msg → Claims.Msg) (zk : Bool) : Refusable H (V zk) := fun x =>
  ⟨[], fun as cs => by
    show ((next zk x (as ++ [[]]) cs).eval H).1 ≠ .acc
    rw [next_rej H (act_snoc_nil zk x as cs)]
    exact fun h => Claims.Out.noConfusion h⟩

/-- **A decision reads no last coin**: past a first coin, where the last message asks for no coin, the step is the
same at any last coin. -/
theorem act_cons_snoc (zk : Bool) (x : Claims.Msg) (as : List Claims.Msg) (c₀ : Coin) (cs : List Coin) (c c' : Coin)
    (ha : ∀ t, as.getLast? ≠ some (0 :: t)) :
    act zk x as (c₀ :: cs ++ [c']) = act zk x as (c₀ :: cs ++ [c]) := by
  unfold act
  rcases setupOf zk x with _ | ⟨s, spec, sts⟩
  · rfl
  · dsimp only
    rw [List.getLast?_concat, List.getLast?_concat]
    rcases hl : as.getLast? with _ | a
    · rfl
    · have ha' : ∀ t, a ≠ 0 :: t := fun t h => ha t (by rw [hl, h])
      split
      · rename_i heq _
        exact absurd (Option.some.inj heq) (ha' _)
      · rename_i a₂ heq _
        obtain rfl := Option.some.inj heq
        symm
        split
        · rename_i heq' _
          exact absurd (Option.some.inj heq') (ha' _)
        · rename_i a₃ heq' _
          obtain rfl := Option.some.inj heq'
          rw [List.cons_append, List.cons_append]
          rcases hf : finalOf a with _ | f
          · rfl
          · rcases as with _ | ⟨a₁, rest⟩
            · rfl
            · simp only [List.dropLast_concat]
        · rename_i heq'
          cases heq'
      · rename_i heq
        cases heq

/-- **A coin request is not a decision.** -/
theorem act_snoc_zero {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {c : Coin}
    {t : Claims.Msg} (ha : as.getLast? = some (0 :: t)) :
    act zk x as (cs ++ [c]) = .rej ∨ act zk x as (cs ++ [c]) = .coin c := by
  unfold act
  rcases setupOf zk x with _ | ⟨s, spec, sts⟩
  · exact Or.inl rfl
  · dsimp only
    rw [ha, List.getLast?_concat]
    exact Or.inr rfl

theorem decideFree_V (H : Claims.Msg → Claims.Msg) (zk : Bool) : DecideFree H (V zk) := by
  intro x as cs c c' hl hacc
  show (next zk x as (cs ++ [c'])).eval H = (next zk x as (cs ++ [c])).eval H
  change ((next zk x as (cs ++ [c])).eval H).1 = .acc at hacc
  cases hact : act zk x as (cs ++ [c]) with
  | rej =>
    rw [next_rej H hact] at hacc
    cases hacc
  | coin c₂ =>
    rw [next_coin H hact] at hacc
    cases hacc
  | decide s spec sts link c₁ mids f =>
    have hlen := act_decide_length (by simp [hl]) hact
    obtain ⟨c₀, cs', rfl⟩ : ∃ c₀ cs', cs = c₀ :: cs' := by
      rcases cs with _ | ⟨c₀, cs'⟩
      · simp only [List.length_nil, zero_add] at hl
        omega
      · exact ⟨c₀, cs', rfl⟩
    have ha : ∀ t, as.getLast? ≠ some (0 :: t) := fun t h => by
      rcases act_snoc_zero (zk := zk) (x := x) (cs := c₀ :: cs') (c := c) h with h' | h' <;>
        rw [hact] at h' <;> cases h'
    unfold next
    rw [act_cons_snoc zk x as c₀ cs' c c' ha]

theorem accLast_V_sha512 (zk : Bool) : AccLast sha512 (V zk) := fun _ _ _ hcs h =>
  next_acc_length_sha512 zk hcs h

/-- **KS of the recursion over Flock's deployed verifier as the inner protocol**, at SHA-512: `KSFlock_of_flockKS` at an
inner deployment whose verifier is `V zk`, with its code facts `AccLast`, `DecideFree` and `Refusable` met by the
deployed verifier. -/
theorem KSFlock_of_flockKS_V {Col T : Type} [DecidableEq Col] [Zero Col] (zk : Bool) (D : Deploy Coin)
    (DI DO : Claims.Flock.Deploy Coin Col T) (hVI : DI.V = V zk) (opening : Opening)
    (decC : Claims.Msg → Claims.Msg → Coin) (hF : Claims.Flock.KS sha512 DO) (NI : Claims.Msg → ℕ)
    (Ok : Claims.Msg → Col → Prop) (hNI : ∀ x, (NI x : ℝ) ≤ DI.N₀ x) (hεI : ∀ x, 0 ≤ DI.εtable x)
    (hVkI : ∀ x, 0 < DI.V.k x) (hKI : ∀ x, 0 < DI.cap x)
    (hkI : ∀ x, (8 ≤ DI.k x ∧ Claims.Flock.Splits DI x) ∨ Claims.Flock.NoAcc sha512 DI.V x)
    (hone : ∀ x, (∃ s, DI.parse x = some s ∧ s.tables = 1) ∨ Claims.Flock.NoAcc sha512 DI.V x) (hz : ∀ x, Ok x 0)
    (hok : Claims.Flock.ColsOk sha512 DI NI Ok) (hconf : Claims.Flock.ConflictTied sha512 DI NI)
    (htab : TableBoundθOnAcc sha512 DI NI Ok) (hRef : Refuses sha512 D) (hRounds : ∀ x, DI.V.k (xI D x) ≤ D.V.k x)
    (hN : SessionNums D DO) (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock DI decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock DI decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock DI decI) DO)
    (hC : ∀ decI, CallsRecord sha512 D (Inner.ofFlock DI decI) DO)
    (hO : ∀ decI, CallsOpen sha512 D (Inner.ofFlock DI decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock DI decI) DO)
    (hpc : ∀ decI, PublicCoin sha512 (Inner.ofFlock DI decI) decC) (hN0 : CallsNil sha512 D) :
    KSFlockA sha512 D DI DO :=
  KSFlock_of_flockKS sha512 D DI DO opening decC hF NI Ok hNI hεI hVkI hKI hkI hone hz hok hconf htab
    (hVI ▸ accLast_V_sha512 zk) (hVI ▸ decideFree_V sha512 zk) (hVI ▸ refusable_V sha512 zk) hRef hRounds hN hε
    hRd hS hR hV hC hO hAcc hpc hN0

end FlockDeploy
